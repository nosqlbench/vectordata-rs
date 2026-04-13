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
use reqwest::blocking::Client;

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

    /// Loads a TestDataGroup from a local directory path or a dataset.yaml file.
    ///
    /// Falls back to `knn_entries.yaml` if `dataset.yaml` is not found.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let (dir, yaml_path) = if path.is_dir() {
            (path.to_path_buf(), path.join("dataset.yaml"))
        } else if path.extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
            (
                path.parent().unwrap_or(Path::new(".")).to_path_buf(),
                path.to_path_buf(),
            )
        } else {
             (path.to_path_buf(), path.join("dataset.yaml"))
        };

        // Try dataset.yaml first
        if yaml_path.exists() {
            let yaml_content = fs::read_to_string(&yaml_path)
                .map_err(Error::ConfigIo)?;
            let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;
            return Ok(Self {
                source: DataSource::FileSystem(dir),
                config,
            });
        }

        // Fall back to knn_entries.yaml
        let knn_path = dir.join("knn_entries.yaml");
        if knn_path.exists() {
            let entries = crate::knn_entries::KnnEntries::load(&knn_path)
                .map_err(|e| Error::Other(e))?;
            let config = entries.to_config();
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
    /// Falls back to `knn_entries.yaml` if `dataset.yaml` is not found (HTTP 404).
    pub fn load_from_url(url_str: &str) -> Result<Self> {
        let mut url = Url::parse(url_str)?;

        // If URL doesn't end in .yaml, assume it's a directory
        if !url.path().ends_with(".yaml") && !url.path().ends_with(".yml") {
            if !url.path().ends_with('/') {
                url.set_path(&(url.path().to_owned() + "/"));
            }
        }

        // Base URL is the directory
        let base_url = if url.path().ends_with('/') {
            url.clone()
        } else {
            url.join(".")?
        };

        let client = Client::new();

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

        // Fall back to knn_entries.yaml
        let knn_url = base_url.join("knn_entries.yaml")?;
        let resp = client.get(knn_url).send()?.error_for_status()?;
        let yaml_content = resp.text()?;
        let entries = crate::knn_entries::KnnEntries::parse(&yaml_content)
            .map_err(|e| Error::Other(e))?;
        let config = entries.to_config();

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
