//! Library API. High-level entry point for loading vector datasets.
//!
//! [`TestDataGroup`] parses a `dataset.yaml` (from a local path or HTTP URL)
//! and exposes named profiles as [`TestDataView`]
//! instances for reading vectors and metadata.

use crate::dataset::selector::{self, ProfileFacts, SelectionError};
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
    /// URL the dataset description was loaded from — the
    /// `dataset.yaml` URL for canonical-shape catalogs, the
    /// `knn_entries.yaml` (or wherever the synthesised layout
    /// originated) for the legacy shape. Recorded in the
    /// per-dataset `<cache_root>/<dataset>/origin.json` so a
    /// catalog that moves can be migrated by editing one file.
    /// `None` only when the group was built via a constructor that
    /// doesn't have URL provenance (test fixtures); the cache
    /// then falls back to URL-derived layout.
    catalog_source: Option<String>,
    /// Dataset identifier used as the per-dataset cache directory
    /// name (`<cache_root>/<dataset_name>/`). Derived from the
    /// catalog entry name when opened through `Catalog::open_entry`,
    /// or from the last path/URL segment otherwise. `None` skips
    /// catalog-anchored layout (URL-derived fallback wins).
    dataset_name: Option<String>,
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
            && path.extension().is_some_and(|ext| ext == "yaml" || ext == "yml")
        {
            let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            let yaml_content = fs::read_to_string(path).map_err(Error::ConfigIo)?;
            let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let config = parse_catalog_content_for(&yaml_content, dir_name)?;
            let dataset_name = dir.file_name()
                .and_then(|n| n.to_str()).map(|s| s.to_string());
            return Ok(Self {
                source: DataSource::FileSystem(dir.clone()),
                config,
                catalog_source: dir.to_str().map(with_trailing_slash),
                dataset_name,
            });
        }

        // Directory cascade.
        let dir = path.to_path_buf();
        let yaml_path = dir.join("dataset.yaml");
        if yaml_path.exists() {
            let yaml_content = fs::read_to_string(&yaml_path).map_err(Error::ConfigIo)?;
            let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;
            let dataset_name = dir.file_name()
                .and_then(|n| n.to_str()).map(|s| s.to_string());
            return Ok(Self {
                source: DataSource::FileSystem(dir.clone()),
                config,
                catalog_source: dir.to_str().map(with_trailing_slash),
                dataset_name,
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
                .map_err(Error::Other)?;
            let dir_name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let config = entries
                .to_config_for(dir_name)
                .unwrap_or_else(|| entries.to_config());
            let dataset_name = dir.file_name()
                .and_then(|n| n.to_str()).map(|s| s.to_string());
            return Ok(Self {
                source: DataSource::FileSystem(dir.clone()),
                config,
                catalog_source: dir.to_str().map(with_trailing_slash),
                dataset_name,
            });
        }

        // Neither found — return the original error
        let yaml_content = fs::read_to_string(&yaml_path)
            .map_err(Error::ConfigIo)?;
        let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;
        let dataset_name = dir.file_name()
            .and_then(|n| n.to_str()).map(|s| s.to_string());
        Ok(Self {
            source: DataSource::FileSystem(dir.clone()),
            config,
            catalog_source: dir.to_str().map(|s| s.to_string()),
            dataset_name,
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
        let client = crate::transport::shared_client_for(url_str);

        // Explicit catalog file: dispatch by content shape, no
        // sibling probing. The base_url for resolving facet paths is
        // the URL's parent directory.
        if url.path().ends_with(".yaml") || url.path().ends_with(".yml") {
            let base_url = url.join(".")?;
            let yaml_content = fetch_descriptor(&client, &url, &base_url)?.required(&url)?;
            let dir_name = base_url
                .path_segments()
                .and_then(|s| s.collect::<Vec<_>>().iter().rev().find(|seg| !seg.is_empty()).cloned())
                .unwrap_or("");
            let config = parse_catalog_content_for(&yaml_content, dir_name)?;
            let dataset_name = base_url
                .path_segments()
                .and_then(|s| s.collect::<Vec<_>>()
                    .iter().rev().find(|seg| !seg.is_empty()).cloned())
                .map(|s| s.to_string());
            return Ok(Self {
                source: DataSource::Http(base_url.clone()),
                config,
                catalog_source: Some(with_trailing_slash(base_url.as_str())),
                dataset_name,
            });
        }

        // Directory cascade.
        let mut url = url;
        if !url.path().ends_with('/') {
            url.set_path(&(url.path().to_owned() + "/"));
        }
        let base_url = url.clone();

        // Try dataset.yaml first. When the server cannot be asked and no
        // copy of it was kept, the dataset may still be a kept
        // `knn_entries.yaml` one: go on to that before giving up.
        let dataset_url = base_url.join("dataset.yaml")?;
        let mut unavailable = None;
        match fetch_descriptor(&client, &dataset_url, &base_url)? {
            Descriptor::Text(yaml_content) => {
                let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;
                let dataset_name = base_url
                    .path_segments()
                    .and_then(|s| s.collect::<Vec<_>>()
                        .iter().rev().find(|seg| !seg.is_empty()).cloned())
                    .map(|s| s.to_string());
                return Ok(Self {
                    source: DataSource::Http(base_url.clone()),
                    config,
                    catalog_source: Some(with_trailing_slash(base_url.as_str())),
                    dataset_name,
                });
            }
            Descriptor::Absent => {}
            Descriptor::Unavailable(e) => unavailable = Some(e),
        }

        // Fall back to knn_entries.yaml. When the file describes
        // multiple datasets, prefer the one whose name matches the
        // last path segment of the URL; otherwise return the first.
        let knn_url = base_url.join("knn_entries.yaml")?;
        let yaml_content = match (fetch_descriptor(&client, &knn_url, &base_url)?, unavailable) {
            (Descriptor::Text(text), _) => text,
            // dataset.yaml went unasked: that is the reason, whatever
            // came of knn_entries.yaml.
            (_, Some(e)) | (Descriptor::Unavailable(e), None) => return Err(e),
            (Descriptor::Absent, None) => {
                return Err(Error::Other(format!("{base_url}: no dataset.yaml or knn_entries.yaml")));
            }
        };
        let entries = crate::knn_entries::KnnEntries::parse(&yaml_content)
            .map_err(Error::Other)?;
        let url_dir_name = base_url
            .path_segments()
            .and_then(|s| s.collect::<Vec<_>>().iter().rev().find(|seg| !seg.is_empty()).cloned())
            .unwrap_or("");
        let config = entries
            .to_config_for(url_dir_name)
            .unwrap_or_else(|| entries.to_config());

        let dataset_name = base_url
            .path_segments()
            .and_then(|s| s.collect::<Vec<_>>()
                .iter().rev().find(|seg| !seg.is_empty()).cloned())
            .map(|s| s.to_string());
        Ok(Self {
            source: DataSource::Http(base_url.clone()),
            config,
            catalog_source: Some(with_trailing_slash(base_url.as_str())),
            dataset_name,
        })
    }

    /// Retrieves a view for the specified profile name.
    ///
    /// Returns `None` if the profile name does not exist in the configuration.
    pub fn profile(&self, name: &str) -> Option<Arc<dyn TestDataView>> {
        let profile_config = self.config.profiles.get(name)?;
        let mut view = GenericTestDataView::with_attributes(
            self.source.clone(),
            profile_config.clone(),
            self.config.attributes.clone(),
        );
        if let (Some(name), Some(src)) = (&self.dataset_name, &self.catalog_source) {
            view = view.with_catalog_identity(name.clone(), src.clone());
        }
        Some(Arc::new(view))
    }

    /// Returns the concrete `GenericTestDataView` for typed facet access.
    ///
    /// Unlike `profile()` which returns a trait object, this returns the
    /// concrete type so clients can call `open_facet_typed::<T>()`.
    pub fn generic_view(&self, name: &str) -> Option<GenericTestDataView> {
        let profile_config = self.config.profiles.get(name)?;
        let mut view = GenericTestDataView::with_attributes(
            self.source.clone(),
            profile_config.clone(),
            self.config.attributes.clone(),
        );
        if let (Some(name), Some(src)) = (&self.dataset_name, &self.catalog_source) {
            view = view.with_catalog_identity(name.clone(), src.clone());
        }
        Some(view)
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

    /// Everything a selector can read of every profile, in the order
    /// `profile_names` lists them (PS-7, PS-9).
    pub fn profile_facts(&self) -> Vec<ProfileFacts> {
        self.profile_names()
            .iter()
            .filter_map(|n| self.config.profiles.get(n).map(|p| ProfileFacts::of_profile(n, p)))
            .collect()
    }

    /// The profiles a selector names, size-ordered (PS-9). `None` is
    /// `default`, `profile=*` is every profile, and a selector that
    /// matches nothing is an error listing what was on offer (PS-10).
    pub fn select(&self, selector: Option<&str>) -> std::result::Result<Vec<String>, SelectionError> {
        selector::resolve(selector, &self.profile_facts())
    }

    /// The one profile a selector names, for a surface that takes a
    /// single profile; more than one match is an error (PS-10).
    pub fn select_one(&self, selector: Option<&str>) -> std::result::Result<String, SelectionError> {
        selector::resolve_one(selector, &self.profile_facts())
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

    /// Fetch facets across several profiles, with progress — the
    /// multi-profile form of [`TestDataView::fetch`].
    ///
    /// `profiles` are fetched in the order given (what a selector
    /// resolved to, through [`select`](Self::select)); a name repeated
    /// is fetched once. `request` applies to each profile. Every event
    /// and report row carries its profile in
    /// [`FacetId::profile`](crate::fetch::FacetId::profile).
    ///
    /// The whole run is planned and checked before anything is fetched,
    /// so an unknown facet in the third profile fails the call before
    /// the first profile's bytes move.
    pub fn fetch(
        &self,
        profiles: &[String],
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> crate::Result<crate::fetch::FetchReport> {
        self.plan_fetch(profiles, request, progress)?.execute(progress)
    }

    /// Plan a [`fetch`](Self::fetch) across `profiles` without fetching.
    pub fn plan_fetch(
        &self,
        profiles: &[String],
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> crate::Result<crate::fetch::FetchPlan> {
        let mut seen = std::collections::HashSet::new();
        let mut facets = Vec::new();
        for name in profiles.iter().filter(|p| seen.insert(p.as_str())) {
            let view = self.profile(name).ok_or_else(|| crate::Error::Selection {
                dataset: self.dataset_name.clone().unwrap_or_default(),
                message: format!("profile '{name}' not found"),
            })?;
            crate::fetch::plan_view(&*view, Some(name), request, progress, &mut facets)?;
        }
        Ok(crate::fetch::plan_from(facets, request))
    }

    /// Deprecated: use [`fetch`](Self::fetch) over
    /// [`profile_names`](Self::profile_names) with
    /// [`FetchRequest::all`](crate::fetch::FetchRequest::all).
    #[deprecated(
        since = "2.5.0",
        note = "use `group.fetch(&group.profile_names(), &FetchRequest::all(), &mut Silent)` (vectordata::fetch)"
    )]
    pub fn prebuffer_all_profiles(&self) -> crate::Result<()> {
        self.fetch(&self.profile_names(), &crate::fetch::FetchRequest::all(), &mut crate::fetch::Silent)
            .map(|_| ())
    }

    /// Deprecated: use [`fetch`](Self::fetch) over
    /// [`profile_names`](Self::profile_names) with a
    /// [`FetchProgress`](crate::fetch::FetchProgress) sink.
    #[deprecated(
        since = "2.5.0",
        note = "use `group.fetch(&group.profile_names(), &FetchRequest::all(), &mut sink)` (vectordata::fetch)"
    )]
    pub fn prebuffer_all_profiles_with_progress(
        &self,
        fallback: crate::view::WholeFacetFallback,
        progress_cb: &mut dyn FnMut(&str, &str, &crate::PrebufferProgress),
        warn_cb: &mut dyn FnMut(u64),
    ) -> crate::Result<()> {
        #[allow(deprecated)]
        self.prebuffer_profiles_with_progress(&self.profile_names(), fallback, progress_cb, warn_cb)
    }

    /// Deprecated: use [`fetch`](Self::fetch), and compare
    /// [`FetchPlan::bytes_to_fetch`](crate::fetch::FetchPlan::bytes_to_fetch)
    /// from [`plan_fetch`](Self::plan_fetch) against
    /// [`PREBUFFER_LARGE_WARNING_BYTES`] for the large-download notice.
    ///
    /// `progress_cb(profile, facet, prog)` hears each facet; `warn_cb`
    /// hears the planned total once, before anything is fetched, when
    /// it reaches [`PREBUFFER_LARGE_WARNING_BYTES`].
    #[deprecated(
        since = "2.5.0",
        note = "use `group.fetch(&profiles, &FetchRequest::all(), &mut sink)` (vectordata::fetch)"
    )]
    pub fn prebuffer_profiles_with_progress(
        &self,
        profiles: &[String],
        fallback: crate::view::WholeFacetFallback,
        progress_cb: &mut dyn FnMut(&str, &str, &crate::PrebufferProgress),
        warn_cb: &mut dyn FnMut(u64),
    ) -> crate::Result<()> {
        let request = crate::fetch::FetchRequest::all().fallback(fallback);
        let mut sink = crate::view::prebuffer_progress_adapter(progress_cb);
        let plan = self.plan_fetch(profiles, &request, &mut sink)?;
        if plan.bytes_to_fetch() >= PREBUFFER_LARGE_WARNING_BYTES {
            warn_cb(plan.bytes_to_fetch());
        }
        plan.execute(&mut sink).map(|_| ())
    }
}

/// Fetch a dataset's descriptor (`dataset.yaml`, `knn_entries.yaml`, or
/// an explicitly named file) from `url`, keeping a copy in the dataset's
/// cache directory, and falling back to that copy when the server
/// cannot be reached.
///
/// The server is asked first because a descriptor is mutable — a
/// republished dataset can gain profiles — unlike the verified data it
/// describes, whose complete copies open with no network at all. The
/// kept copy is what lets a cached dataset still open when the server
/// is down or the machine is offline. Errors only for a server answer
/// that is neither the file nor "no such file", or a kept copy that
/// cannot be read.
fn fetch_descriptor(
    client: &reqwest::blocking::Client,
    url: &Url,
    base_url: &Url,
) -> Result<Descriptor> {
    let kept = kept_descriptor_path(url, base_url).filter(|p| p.is_file());
    let read_kept = |path: &std::path::Path| {
        std::fs::read_to_string(path).map(Descriptor::Text).map_err(Error::ConfigIo)
    };
    if crate::settings::offline() {
        return match &kept {
            Some(path) => read_kept(path),
            None => Ok(Descriptor::Unavailable(Error::Other(
                crate::transport::ensure_online(url.as_str()).unwrap_err().to_string(),
            ))),
        };
    }
    let resp = crate::transport::apply_read_auth(client.get(url.clone()), Some(url)).send();
    match resp {
        Ok(r) if r.status().is_success() => {
            let text = r.text()?;
            if let Some(path) = kept_descriptor_path(url, base_url) {
                // Best effort: a cache directory that cannot be written
                // only costs the offline fallback.
                let _ = path.parent().map(std::fs::create_dir_all);
                let _ = std::fs::write(path, &text);
            }
            Ok(Descriptor::Text(text))
        }
        Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => Ok(Descriptor::Absent),
        Ok(r) => Err(Error::Other(format!("{url}: {}", r.status()))),
        Err(e) if e.is_connect() || e.is_timeout() => match &kept {
            Some(path) => {
                log::warn!("{url} is unreachable ({e}); using the copy kept at {}", path.display());
                read_kept(path)
            }
            None => Ok(Descriptor::Unavailable(Error::Http(e))),
        },
        Err(e) => Err(Error::Http(e)),
    }
}

/// What [`fetch_descriptor`] found for one descriptor file.
enum Descriptor {
    /// The file's text: from the server, or the kept copy.
    Text(String),
    /// The server answered that there is no such file.
    Absent,
    /// The server could not be asked — unreachable, or offline mode —
    /// and no copy was kept. The error says which.
    Unavailable(Error),
}

impl Descriptor {
    /// The text of a descriptor the caller cannot do without: absent is
    /// "not found", unavailable is why it could not be asked for.
    fn required(self, url: &Url) -> Result<String> {
        match self {
            Descriptor::Text(text) => Ok(text),
            Descriptor::Absent => Err(Error::Other(format!("{url}: not found"))),
            Descriptor::Unavailable(e) => Err(e),
        }
    }
}

/// Fetch a dataset's `dataset.yaml` from a URL naming the dataset
/// directory or the file itself, through [`fetch_descriptor`]: read
/// auth, a kept copy, and that copy when the server is unreachable. For
/// callers that need the raw text rather than a parsed group.
pub(crate) fn fetch_dataset_yaml(url_str: &str) -> Result<String> {
    let mut url = Url::parse(url_str)?;
    if !url.path().ends_with(".yaml") && !url.path().ends_with(".yml") {
        if !url.path().ends_with('/') {
            url.set_path(&(url.path().to_owned() + "/"));
        }
        url = url.join("dataset.yaml")?;
    }
    let base_url = url.join(".")?;
    let client = crate::transport::shared_client_for(url.as_str());
    fetch_descriptor(&client, &url, &base_url)?.required(&url)
}

/// Where a descriptor fetched from `url` is kept: under the dataset's
/// cache directory, `<cache>/<dataset>/<path below the dataset's base>`
/// — the same place the published layout puts it. `None` without a
/// configured cache.
fn kept_descriptor_path(url: &Url, base_url: &Url) -> Option<std::path::PathBuf> {
    let cache_root = crate::settings::cache_dir().ok()?;
    let dataset = base_url
        .path_segments()?
        .rfind(|s| !s.is_empty())?
        .to_string();
    let rel = url.path().strip_prefix(base_url.path()).unwrap_or(url.path()).trim_start_matches('/');
    if rel.is_empty() || rel.contains("..") {
        return None;
    }
    let dir = crate::cache::layout::dataset_cache_dir(&cache_root, &dataset);
    // The directory belongs to one origin; a same-named dataset from
    // elsewhere neither reads nor overwrites this one's descriptor.
    crate::cache::layout::verify_or_record_origin(&dir, &with_trailing_slash(base_url.as_str())).ok()?;
    Some(dir.join(rel))
}

/// Planned download size at which a fetch deserves a "this is a lot of
/// data" notice before it starts: what `vectordata datasets precache`
/// warns at, comparing [`FetchPlan::bytes_to_fetch`](crate::fetch::FetchPlan::bytes_to_fetch)
/// against it.
/// 250 MiB matches the documented operator guidance: a reminder, not
/// a hard limit.
pub const PREBUFFER_LARGE_WARNING_BYTES: u64 = 250 * 1024 * 1024;

impl TestDataGroup {
    /// Construct a `TestDataGroup` directly from a [`CatalogEntry`](crate::dataset::CatalogEntry)
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
        // The dataset's *home URL* is the URL the cache should
        // mirror under `<cache>/<dataset_name>/`. For
        // knn_entries-shape catalogs `entry.path` is the catalog
        // base (no dataset segment) so we have to append the
        // dataset name. For canonical catalogs `entry.path` is the
        // `dataset.yaml` URL whose parent directory IS the home
        // URL. View facet URLs strip this prefix to derive the
        // relative path that the cache mirrors under the dataset
        // directory.
        let dataset_home_url = entry.dataset_home_url();
        Ok(Self {
            source,
            config,
            catalog_source: Some(dataset_home_url),
            dataset_name: Some(entry.name.clone()),
        })
    }
}

/// Choose a [`DataSource`] for an arbitrary catalog-entry path or
/// URL. `http(s)://` → `Http(url)`; `file://` is normalised to a
/// plain filesystem path; everything else is treated as a local
/// path. Returns an error for malformed URLs.
/// Normalise a dataset-home URL/path so the trailing `/` is always
/// present. Required because [`crate::view::GenericTestDataView::open_facet_storage`]
/// derives `file_relpath` via prefix strip; an inconsistent trailing
/// slash would either fail the strip (`<home>` vs `<home>/facet`) or
/// erase the first segment of the relpath.
fn with_trailing_slash<S: AsRef<str>>(s: S) -> String {
    let s = s.as_ref();
    if s.ends_with('/') { s.to_string() } else { format!("{s}/") }
}

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
    fn dataset_home_url_canonical_strips_dataset_yaml_basename() {
        // Build a CatalogEntry with the canonical `dataset.yaml` shape
        // and assert that the home URL we record as catalog_source is
        // the *directory containing* the dataset.yaml, not the URL of
        // the YAML itself. View facet URLs strip this prefix to get
        // a clean file_relpath like "base.fvec".
        let layout = crate::dataset::CatalogLayout {
            format_version: crate::model::FORMAT_VERSION_BASE,
            profile_tags: Default::default(),
            attributes: None,
            profiles: Default::default(),
        };
        let entry = crate::dataset::CatalogEntry {
            name: "vecs1m".to_string(),
            path: "https://example.com/datasets/vecs1m/dataset.yaml".to_string(),
            dataset_type: "dataset.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout,
        };
        let group = TestDataGroup::from_catalog_entry(&entry).unwrap();
        assert_eq!(
            group.catalog_source.as_deref(),
            Some("https://example.com/datasets/vecs1m/"),
        );
        assert_eq!(group.dataset_name.as_deref(), Some("vecs1m"));
    }

    #[test]
    fn dataset_home_url_knn_entries_appends_dataset_name() {
        // For knn_entries-shape catalogs, entry.path is the catalog
        // *base* (no dataset segment). The home URL needs the
        // dataset name appended so the strip-prefix in
        // view.open_facet_storage yields the right relpath
        // ("base.fvec", not the absolute URL).
        let layout = crate::dataset::CatalogLayout {
            format_version: crate::model::FORMAT_VERSION_BASE,
            profile_tags: Default::default(),
            attributes: None,
            profiles: Default::default(),
        };
        let entry = crate::dataset::CatalogEntry {
            name: "emb-002-100k".to_string(),
            path: "s3://vector-datasets-public/datasets-clean".to_string(),
            dataset_type: "knn_entries.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout,
        };
        let group = TestDataGroup::from_catalog_entry(&entry).unwrap();
        assert_eq!(
            group.catalog_source.as_deref(),
            Some("s3://vector-datasets-public/datasets-clean/emb-002-100k/"),
        );
        assert_eq!(group.dataset_name.as_deref(), Some("emb-002-100k"));
    }

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
