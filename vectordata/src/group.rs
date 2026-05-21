//! High-level entry point for loading vector datasets.
//!
//! [`TestDataGroup`] parses a `dataset.yaml` (from a local path or HTTP URL)
//! and exposes named profiles as [`TestDataView`](crate::view::TestDataView)
//! instances for reading vectors and metadata.

use crate::model::DatasetConfig;
use crate::view::{GenericTestDataView, TestDataView};
use crate::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use url::Url;

/// The location of the dataset source.
#[derive(Clone, Debug)]
pub enum DataSource {
    /// The dataset is located on the local file system.
    FileSystem(PathBuf),
    /// The dataset is located at a remote HTTP(S) URL.
    Http(Url),
}

/// Represents a loaded vector dataset configuration.
///
/// Use `TestDataGroup::load` to parse a `dataset.yaml` and prepare for data access.
#[derive(Debug)]
pub struct TestDataGroup {
    source: DataSource,
    config: DatasetConfig,
}

impl TestDataGroup {
    /// Loads a TestDataGroup from a path string which can be a file path or URL.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vectordata::TestDataGroup;
    ///
    /// // Load from local path
    /// let local_group = TestDataGroup::load("./data")?;
    ///
    /// // Load from remote URL
    /// let remote_group = TestDataGroup::load("https://example.com/data/")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn load(path_or_url: &str) -> Result<Self> {
        if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
            Self::load_from_url(path_or_url)
        } else {
            Self::load_from_path(path_or_url)
        }
    }

    /// Loads a TestDataGroup from a local directory path or a YAML
    /// catalog file.
    ///
    /// When `path` names a `.yaml`/`.yml` file, that file *is* the
    /// catalog regardless of its basename — the content is dispatched
    /// by shape (`dataset.yaml`-shaped struct vs `knn_entries`-shaped
    /// map), matching the back-end behaviour of jvector's
    /// `DataSetLoaderSimpleMFD`. When `path` names a directory, the
    /// canonical cascade applies: `dataset.yaml` → `knn_entries.yaml`.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Explicit catalog file: dispatch by content shape, no
        // sibling probing.
        if path.is_file()
            && path.extension().map_or(false, |ext| ext == "yaml" || ext == "yml")
        {
            let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            let yaml_content = fs::read_to_string(path).map_err(Error::ConfigIo)?;
            let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let config = parse_catalog_content_for(&yaml_content, dir_name)?;
            return Ok(Self {
                source: DataSource::FileSystem(dir),
                config,
            });
        }

        // Directory cascade.
        let dir = path.to_path_buf();
        let yaml_path = dir.join("dataset.yaml");
        if yaml_path.exists() {
            let yaml_content = fs::read_to_string(&yaml_path).map_err(Error::ConfigIo)?;
            let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;
            return Ok(Self {
                source: DataSource::FileSystem(dir),
                config,
            });
        }

        // Fall back to knn_entries.yaml. When the file describes
        // multiple datasets, prefer the one whose name matches the
        // containing directory; otherwise return the first
        // dataset's config (preserves the prior behavior for
        // single-dataset files).
        let knn_path = dir.join("knn_entries.yaml");
        if knn_path.exists() {
            let entries = crate::knn_entries::KnnEntries::load(&knn_path)
                .map_err(|e| Error::Other(e))?;
            let dir_name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let config = entries
                .to_config_for(dir_name)
                .unwrap_or_else(|| entries.to_config());
            return Ok(Self {
                source: DataSource::FileSystem(dir),
                config,
            });
        }

        // Neither found — return the original error
        let yaml_content = fs::read_to_string(&yaml_path)
            .map_err(Error::ConfigIo)?;
        let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;
        Ok(Self {
            source: DataSource::FileSystem(dir),
            config,
        })
    }

    /// Loads a TestDataGroup from a URL.
    ///
    /// When the URL path ends in `.yaml`/`.yml`, that file *is* the
    /// catalog regardless of its basename — the content is dispatched
    /// by shape (`dataset.yaml`-shaped struct vs `knn_entries`-shaped
    /// map), matching jvector's `DataSetLoaderSimpleMFD`. When the
    /// URL targets a directory, the canonical cascade applies:
    /// `dataset.yaml` → `knn_entries.yaml`.
    pub fn load_from_url(url_str: &str) -> Result<Self> {
        let url = Url::parse(url_str)?;
        let client = crate::transport::shared_client();

        // Explicit catalog file: dispatch by content shape, no
        // sibling probing. The base_url for resolving facet paths is
        // the URL's parent directory.
        if url.path().ends_with(".yaml") || url.path().ends_with(".yml") {
            let base_url = url.join(".")?;
            let resp = client.get(url.clone()).send()?.error_for_status()?;
            let yaml_content = resp.text()?;
            let dir_name = base_url
                .path_segments()
                .and_then(|s| s.collect::<Vec<_>>().iter().rev().find(|seg| !seg.is_empty()).cloned())
                .unwrap_or("");
            let config = parse_catalog_content_for(&yaml_content, dir_name)?;
            return Ok(Self {
                source: DataSource::Http(base_url),
                config,
            });
        }

        // Directory cascade.
        let mut url = url;
        if !url.path().ends_with('/') {
            url.set_path(&(url.path().to_owned() + "/"));
        }
        let base_url = url.clone();

        // Try dataset.yaml first
        let dataset_url = base_url.join("dataset.yaml")?;
        let resp = client.get(dataset_url.clone()).send()?;
        if resp.status().is_success() {
            let yaml_content = resp.text()?;
            let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;
            return Ok(Self {
                source: DataSource::Http(base_url),
                config,
            });
        }

        // Fall back to knn_entries.yaml. When the file describes
        // multiple datasets, prefer the one whose name matches the
        // last path segment of the URL; otherwise return the first.
        let knn_url = base_url.join("knn_entries.yaml")?;
        let resp = client.get(knn_url).send()?.error_for_status()?;
        let yaml_content = resp.text()?;
        let entries = crate::knn_entries::KnnEntries::parse(&yaml_content)
            .map_err(|e| Error::Other(e))?;
        let url_dir_name = base_url
            .path_segments()
            .and_then(|s| s.collect::<Vec<_>>().iter().rev().find(|seg| !seg.is_empty()).cloned())
            .unwrap_or("");
        let config = entries
            .to_config_for(url_dir_name)
            .unwrap_or_else(|| entries.to_config());

        Ok(Self {
            source: DataSource::Http(base_url),
            config,
        })
    }

    /// Retrieves a view for the specified profile name.
    ///
    /// Returns `None` if the profile name does not exist in the configuration.
    pub fn profile(&self, name: &str) -> Option<Arc<dyn TestDataView>> {
        let profile_config = self.config.profiles.get(name)?;
        let view = GenericTestDataView::with_attributes(
            self.source.clone(),
            profile_config.clone(),
            self.config.attributes.clone(),
        );
        Some(Arc::new(view))
    }

    /// Returns the concrete `GenericTestDataView` for typed facet access.
    ///
    /// Unlike `profile()` which returns a trait object, this returns the
    /// concrete type so clients can call `open_facet_typed::<T>()`.
    pub fn generic_view(&self, name: &str) -> Option<GenericTestDataView> {
        let profile_config = self.config.profiles.get(name)?;
        Some(GenericTestDataView::with_attributes(
            self.source.clone(),
            profile_config.clone(),
            self.config.attributes.clone(),
        ))
    }

    /// Returns the names of all available profiles.
    pub fn profile_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.config.profiles.keys().cloned().collect();
        names.sort_by(|a, b| {
            let a_bc = self.config.profiles.get(a).and_then(|p| p.base_count);
            let b_bc = self.config.profiles.get(b).and_then(|p| p.base_count);
            crate::dataset::profile::profile_sort_by_size(a, a_bc, b, b_bc)
        });
        names
    }

    /// Retrieves a top-level attribute from the dataset configuration.
    pub fn attribute(&self, name: &str) -> Option<&serde_yaml::Value> {
        self.config.attributes.get(name)
    }

    /// Direct read-only access to the underlying parsed
    /// `dataset.yaml`. Used by tooling that needs profile-level
    /// detail not exposed through [`TestDataView`] (e.g.
    /// `vectordata datasets derive` reads per-facet windows so it
    /// can materialize them into a self-standing dataset).
    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }

    /// Drive every facet of every profile in this dataset to
    /// fully-resident, zero-copy state.
    ///
    /// **Strict contract** (mirrors [`crate::TestDataView::prebuffer_all`]):
    /// returning `Ok(())` means every facet of every profile is
    /// local and mmap-promoted. Per-facet failure is propagated.
    ///
    /// If the announced total download size exceeds
    /// [`PREBUFFER_LARGE_WARNING_BYTES`] (250 MiB) and `warn_cb`
    /// is provided, `warn_cb` is invoked once with the total
    /// before the download begins. Precache continues regardless
    /// of the warning — the callback is purely advisory.
    pub fn prebuffer_all_profiles(&self) -> crate::Result<()> {
        self.prebuffer_all_profiles_with_progress(
            &mut |_, _, _| {},
            &mut |_| {},
        )
    }

    /// Same as [`prebuffer_all_profiles`] with progress and
    /// large-download warning callbacks.
    ///
    /// `progress_cb(profile, facet, prog)` fires per facet within
    /// each profile (after that facet's download completes; for
    /// already-resident facets it fires once with `total_chunks=0`).
    /// `warn_cb(total_bytes)` fires at most once if the announced
    /// total exceeds [`PREBUFFER_LARGE_WARNING_BYTES`] — the caller
    /// may print a warning to the user; precache continues
    /// regardless.
    pub fn prebuffer_all_profiles_with_progress(
        &self,
        progress_cb: &mut dyn FnMut(&str, &str, &crate::PrebufferProgress),
        warn_cb: &mut dyn FnMut(u64),
    ) -> crate::Result<()> {
        // Tally total announced size across every profile so we can
        // surface a large-download warning before doing the work.
        let mut total_bytes: u64 = 0;
        let mut profiles: Vec<String> = self.profile_names();
        // Stable order: profile_names() already returns
        // size-sorted; we fold dedupes implicitly because keys
        // are unique.
        profiles.sort();
        profiles.dedup();
        for profile_name in &profiles {
            if let Some(view) = self.profile(profile_name) {
                for (facet, _desc) in view.facet_manifest() {
                    if view.facet_element_type(&facet).is_err() { continue; }
                    if let Ok(storage) = view.open_facet_storage(&facet) {
                        // Only count cached/http facets toward the
                        // download tally; local-mmap facets are
                        // already resident.
                        if !storage.is_local() {
                            total_bytes += storage.total_size();
                        }
                    }
                }
            }
        }
        if total_bytes >= PREBUFFER_LARGE_WARNING_BYTES {
            warn_cb(total_bytes);
        }

        for profile_name in &profiles {
            let view = self.profile(profile_name)
                .ok_or_else(|| crate::Error::Other(format!(
                    "profile '{profile_name}' missing during precache")))?;
            view.prebuffer_all_with_progress(&mut |facet, prog| {
                progress_cb(profile_name, facet, prog);
            })?;
        }
        Ok(())
    }
}

/// Threshold above which [`TestDataGroup::prebuffer_all_profiles`]
/// fires its `warn_cb` to surface a "this is a lot of data" notice.
/// 250 MiB matches the documented operator guidance: a reminder, not
/// a hard limit.
pub const PREBUFFER_LARGE_WARNING_BYTES: u64 = 250 * 1024 * 1024;

impl TestDataGroup {
    /// Construct a `TestDataGroup` directly from a [`CatalogEntry`]
    /// whose layout already carries the full profile/facet
    /// description.
    ///
    /// For `knn_entries.yaml`-shape entries the catalog file *is*
    /// the dataset description — there is no separate
    /// `dataset.yaml` to fetch. The entry's `path` is a base
    /// location (potentially an `s3://` or `file://` URL the
    /// catalog publishes) and every facet path inside the layout
    /// has already been absolutised by the parser, so the
    /// data-source field is only used as a fallback when a
    /// downstream caller hands in a relative path. For canonical
    /// (`catalog.json`-shape) entries the layout is a summary,
    /// not the full config — use [`Self::load`] instead.
    pub fn from_catalog_entry(entry: &crate::dataset::CatalogEntry) -> Result<Self> {
        // Round-trip the layout through YAML so the canonical
        // `DatasetConfig` deserializer maps each `DSView` to a
        // `FacetConfig` (Simple or Detailed). The serialisers on
        // both sides agree on the wire shape — this is the
        // documented path for crossing the layout↔config gap
        // without hand-coding a converter that would drift.
        let layout_yaml = serde_yaml::to_string(&entry.layout)
            .map_err(Error::from)?;
        let config: DatasetConfig = serde_yaml::from_str(&layout_yaml)?;
        let source = data_source_for(&entry.path)?;
        Ok(Self { source, config })
    }
}

/// Choose a [`DataSource`] for an arbitrary catalog-entry path or
/// URL. `http(s)://` → `Http(url)`; `file://` is normalised to a
/// plain filesystem path; everything else is treated as a local
/// path. Returns an error for malformed URLs.
fn data_source_for(location: &str) -> Result<DataSource> {
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(DataSource::Http(Url::parse(location)?));
    }
    let path = if let Some(rest) = location.strip_prefix("file://") {
        // file:///abs → /abs; file://host/abs → /abs (host dropped).
        if rest.starts_with('/') {
            rest.to_string()
        } else if let Some(slash) = rest.find('/') {
            rest[slash..].to_string()
        } else {
            format!("/{rest}")
        }
    } else {
        location.to_string()
    };
    Ok(DataSource::FileSystem(PathBuf::from(path)))
}

/// Parse a catalog YAML by content shape and return the resulting
/// [`DatasetConfig`] for the dataset matching `prefer_name` (or the
/// first dataset if no match). Used by both the URL and filesystem
/// load paths when the location targets an explicit YAML file
/// regardless of its basename — the shape, not the name, decides
/// which parser handles it.
fn parse_catalog_content_for(content: &str, prefer_name: &str) -> Result<DatasetConfig> {
    // Try the canonical `dataset.yaml` shape first. We probe with a
    // generic YAML value so we can distinguish "wrong shape — try
    // the alternative" from "right shape but malformed — surface
    // the error".
    let value: serde_yaml::Value = serde_yaml::from_str(content)?;
    if let serde_yaml::Value::Mapping(ref m) = value {
        // dataset.yaml is identified by the presence of a top-level
        // `profiles:` key. knn_entries.yaml's top level is a flat
        // map of `name:profile` keys plus an optional `_defaults`.
        if m.contains_key(serde_yaml::Value::String("profiles".into())) {
            return serde_yaml::from_value(value).map_err(Error::from);
        }
    }

    // Fall through to the knn_entries shape.
    let entries = crate::knn_entries::KnnEntries::parse(content)
        .map_err(Error::Other)?;
    Ok(entries
        .to_config_for(prefer_name)
        .unwrap_or_else(|| entries.to_config()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_load_from_path_success() {
        let dir = tempdir().unwrap();
        let yaml_path = dir.path().join("dataset.yaml");
        let mut file = fs::File::create(&yaml_path).unwrap();
        writeln!(file, r#"
attributes:
  dimension: 128
profiles:
  default:
    base_vectors: base.fvec
"#).unwrap();

        let group = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap();
        assert!(group.profile("default").is_some());
        assert!(group.profile("nonexistent").is_none());
        
        let dim = group.attribute("dimension").unwrap();
        assert_eq!(dim.as_u64().unwrap(), 128);
    }

    #[test]
    fn test_load_from_path_file_success() {
        let dir = tempdir().unwrap();
        let yaml_path = dir.path().join("dataset.yaml");
        let mut file = fs::File::create(&yaml_path).unwrap();
        writeln!(file, "profiles: {{}}").unwrap();

        let group = TestDataGroup::load(yaml_path.to_str().unwrap()).unwrap();
        assert!(group.config.profiles.is_empty());
    }

    #[test]
    fn test_load_from_path_missing_file() {
        let dir = tempdir().unwrap();
        let result = TestDataGroup::load(dir.path().to_str().unwrap());
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::ConfigIo(_) => (), // Expected
            e => panic!("Expected ConfigIo error, got {:?}", e),
        }
    }
    
    #[test]
    fn test_load_from_path_invalid_yaml() {
        let dir = tempdir().unwrap();
        let yaml_path = dir.path().join("dataset.yaml");
        let mut file = fs::File::create(&yaml_path).unwrap();
        writeln!(file, "invalid_yaml: [").unwrap();

        let result = TestDataGroup::load(dir.path().to_str().unwrap());
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::ConfigParse(_) => (), // Expected
            e => panic!("Expected ConfigParse error, got {:?}", e),
        }
    }
}
