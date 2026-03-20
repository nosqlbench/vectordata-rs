// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Catalog discovery and loading for dataset indexes.
//!
//! A catalog is a JSON or YAML array of [`CatalogEntry`] values. Each entry
//! embeds the publishable parts of a [`DatasetConfig`](super::config::DatasetConfig)
//! (attributes and profiles) under a `layout` key, alongside the dataset
//! `name`, relative `path` to the `dataset.yaml`, and a `dataset_type`
//! discriminator.
//!
//! Use [`find_catalog`] to locate a `catalog.json` or `catalog.yaml` in a
//! directory, and [`load_catalog`] to parse it. These are used by both
//! catalog generation (`veks catalog generate`) and catalog consumption
//! (`veks datasets list`).
//!
//! ## Catalog file format
//!
//! ```yaml
//! - name: sift-128
//!   path: vendor/sift/dataset.yaml
//!   dataset_type: dataset.yaml
//!   layout:
//!     attributes:
//!       distance_function: L2
//!       license: MIT
//!     profiles:
//!       default:
//!         base_vectors: base.fvec
//!         query_vectors: query.fvec
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::config::DatasetAttributes;
use super::profile::DSProfileGroup;

/// The layout block embedded within each catalog entry.
///
/// Contains the publishable subset of a `DatasetConfig`: attributes and
/// profiles. The `upstream` pipeline section is intentionally excluded —
/// it describes build steps, not dataset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogLayout {
    /// Dataset-level attributes (model, distance function, license, etc.).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub attributes: Option<DatasetAttributes>,
    /// Named profiles mapping view names to data sources.
    #[serde(default)]
    pub profiles: DSProfileGroup,
}

/// A layout-embedded catalog entry for one discovered dataset.
///
/// Matches the upstream Java catalog specification so that catalog files
/// produced by veks can be consumed by the Java vectordata toolchain and
/// vice versa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Dataset name (from `dataset.yaml`).
    pub name: String,
    /// Relative path from the catalog file to the `dataset.yaml`.
    pub path: String,
    /// Entry type discriminator — always `"dataset.yaml"` for layout entries.
    #[serde(default = "default_dataset_type")]
    pub dataset_type: String,
    /// Embedded dataset configuration (attributes + profiles).
    pub layout: CatalogLayout,
}

fn default_dataset_type() -> String {
    "dataset.yaml".to_string()
}

impl CatalogEntry {
    /// Returns view names from the default profile (convenience accessor).
    pub fn view_names(&self) -> Vec<&str> {
        self.layout.profiles.view_names()
    }

    /// Returns the number of profiles.
    pub fn profile_count(&self) -> usize {
        self.layout.profiles.profile_names().len()
    }

    /// Returns profile names.
    pub fn profile_names(&self) -> Vec<&str> {
        self.layout.profiles.profile_names()
    }
}

/// Load catalog entries from a `catalog.json` or `catalog.yaml` file.
///
/// Returns `Ok(entries)` on success, or an error string if the file cannot
/// be read or parsed.
pub fn load_catalog(path: &Path) -> Result<Vec<CatalogEntry>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "json" => serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse {}: {}", path.display(), e)),
        "yaml" | "yml" => serde_yaml::from_str(&content)
            .map_err(|e| format!("failed to parse {}: {}", path.display(), e)),
        _ => Err(format!("unsupported catalog format: {}", path.display())),
    }
}

/// Attempt to find and load a catalog file in the given directory.
///
/// Checks for `catalog.json` first, then `catalog.yaml`. Returns `None`
/// if neither exists.
pub fn find_catalog(dir: &Path) -> Option<Result<Vec<CatalogEntry>, String>> {
    let json_path = dir.join("catalog.json");
    if json_path.is_file() {
        return Some(load_catalog(&json_path));
    }
    let yaml_path = dir.join("catalog.yaml");
    if yaml_path.is_file() {
        return Some(load_catalog(&yaml_path));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_json() {
        let json = r#"[
  {
    "name": "test-ds",
    "path": "vendor/test/dataset.yaml",
    "dataset_type": "dataset.yaml",
    "layout": {
      "attributes": {
        "distance_function": "COSINE",
        "license": "MIT"
      },
      "profiles": {
        "default": {
          "base_vectors": "base.fvec",
          "query_vectors": "query.fvec"
        }
      }
    }
  }
]"#;
        let entries: Vec<CatalogEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test-ds");
        assert_eq!(entries[0].dataset_type, "dataset.yaml");
        assert_eq!(
            entries[0].layout.attributes.as_ref().unwrap().distance_function.as_deref(),
            Some("COSINE")
        );
        assert_eq!(entries[0].profile_count(), 1);
        assert_eq!(entries[0].view_names().len(), 2);
    }

    #[test]
    fn test_roundtrip_yaml() {
        let yaml = r#"
- name: sift-128
  path: sift/dataset.yaml
  dataset_type: dataset.yaml
  layout:
    profiles:
      default:
        base_vectors: base.fvec
"#;
        let entries: Vec<CatalogEntry> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "sift-128");
        assert!(entries[0].layout.attributes.is_none());
        assert_eq!(entries[0].profile_count(), 1);
    }

    #[test]
    fn test_default_dataset_type() {
        // Missing dataset_type should default to "dataset.yaml"
        let json = r#"[{
            "name": "x",
            "path": "x/dataset.yaml",
            "layout": { "profiles": {} }
        }]"#;
        let entries: Vec<CatalogEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(entries[0].dataset_type, "dataset.yaml");
    }

    #[test]
    fn test_load_catalog_json_file() {
        let tmp = tempfile::tempdir().unwrap();
        let json = r#"[{"name":"a","path":"a/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{"default":{"base_vectors":"b.fvec"}}}}]"#;
        std::fs::write(tmp.path().join("catalog.json"), json).unwrap();

        let entries = load_catalog(&tmp.path().join("catalog.json")).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "a");
    }

    #[test]
    fn test_find_catalog_prefers_json() {
        let tmp = tempfile::tempdir().unwrap();
        let json = r#"[{"name":"from-json","path":"a/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{}}}]"#;
        let yaml = "- name: from-yaml\n  path: a/dataset.yaml\n  dataset_type: dataset.yaml\n  layout:\n    profiles: {}\n";
        std::fs::write(tmp.path().join("catalog.json"), json).unwrap();
        std::fs::write(tmp.path().join("catalog.yaml"), yaml).unwrap();

        let entries = find_catalog(tmp.path()).unwrap().unwrap();
        assert_eq!(entries[0].name, "from-json");
    }

    #[test]
    fn test_find_catalog_falls_back_to_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = "- name: from-yaml\n  path: a/dataset.yaml\n  dataset_type: dataset.yaml\n  layout:\n    profiles: {}\n";
        std::fs::write(tmp.path().join("catalog.yaml"), yaml).unwrap();

        let entries = find_catalog(tmp.path()).unwrap().unwrap();
        assert_eq!(entries[0].name, "from-yaml");
    }

    #[test]
    fn test_find_catalog_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_catalog(tmp.path()).is_none());
    }
}
