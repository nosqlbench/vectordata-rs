//! High-level access to vector datasets.
//!
//! This module provides the `TestDataGroup` struct, which serves as the entry point
//! for loading and accessing vector datasets.

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
    /// use datatools_vectordata_rs::TestDataGroup;
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
            // Assume it is a directory if no extension, or let fs error handle it
             (path.to_path_buf(), path.join("dataset.yaml"))
        };

        let yaml_content = fs::read_to_string(&yaml_path)
            .map_err(Error::ConfigIo)?;
        
        let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;

        Ok(Self { 
            source: DataSource::FileSystem(dir),
            config 
        })
    }

    /// Loads a TestDataGroup from a URL.
    pub fn load_from_url(url_str: &str) -> Result<Self> {
        let mut url = Url::parse(url_str)?;
        
        // If URL doesn't end in .yaml, assume it's a directory and append dataset.yaml
        if !url.path().ends_with(".yaml") && !url.path().ends_with(".yml") {
            if !url.path().ends_with('/') {
                url.set_path(&(url.path().to_owned() + "/"));
            }
            url = url.join("dataset.yaml")?;
        }

        // Base URL is the parent directory
        let base_url = if url.path().ends_with('/') {
            url.clone()
        } else {
            url.join(".")?
        };

        let client = Client::new();
        let yaml_content = client.get(url.clone())
            .send()?
            .error_for_status()?
            .text()?;

        let config: DatasetConfig = serde_yaml::from_str(&yaml_content)?;

        Ok(Self {
            source: DataSource::Http(base_url),
            config
        })
    }

    /// Retrieves a view for the specified profile name.
    ///
    /// Returns `None` if the profile name does not exist in the configuration.
    pub fn profile(&self, name: &str) -> Option<Arc<dyn TestDataView>> {
        let profile_config = self.config.profiles.get(name)?;
        let view = GenericTestDataView::new(self.source.clone(), profile_config.clone());
        Some(Arc::new(view))
    }

    /// Retrieves a top-level attribute from the dataset configuration.
    pub fn attribute(&self, name: &str) -> Option<&serde_yaml::Value> {
        self.config.attributes.get(name)
    }
}
