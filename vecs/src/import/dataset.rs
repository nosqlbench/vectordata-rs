// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset YAML configuration — describes profiles, views, and upstream pipelines.
//!
//! A `dataset.yaml` file describes a complete benchmark or test dataset using
//! named profiles. Each profile maps canonical view names (facet names) to
//! data sources. A `"default"` profile provides the baseline configuration;
//! other profiles inherit from it automatically.
//!
//! The optional top-level `upstream` section defines a multi-step pipeline
//! for building the dataset from source data.

use std::path::Path;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::facet::Facet;
use super::profile::{DSProfile, DSProfileGroup, DSView};
use super::source::DSSource;
use crate::pipeline::schema::PipelineConfig;

/// A dataset.yaml configuration — describes profiles, views, and optionally
/// a multi-step command stream pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset name.
    pub name: String,

    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Top-level pipeline configuration (shared defaults and steps).
    ///
    /// When present, `vecs run` can execute a full multi-step pipeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<PipelineConfig>,

    /// Named profiles, each mapping canonical view names to data sources.
    #[serde(default)]
    pub profiles: DSProfileGroup,
}

impl DatasetConfig {
    /// Load from a YAML file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Returns the default profile, if one exists.
    pub fn default_profile(&self) -> Option<&DSProfile> {
        self.profiles.default_profile()
    }

    /// Look up a profile by name.
    pub fn profile(&self, name: &str) -> Option<&DSProfile> {
        self.profiles.profile(name)
    }

    /// List all profile names.
    pub fn profile_names(&self) -> Vec<&str> {
        self.profiles.profile_names()
    }

    /// Returns canonical view names from the default profile (for display).
    pub fn view_names(&self) -> Vec<&str> {
        self.profiles.view_names()
    }

    /// Validate that all view keys in all profiles are recognized canonical
    /// facet names and that source paths don't point into managed directories.
    /// Returns a list of errors.
    pub fn validate(&self, _base_dir: &Path) -> Vec<String> {
        let mut errors = Vec::new();

        for (profile_name, profile) in &self.profiles.0 {
            for (key, view) in &profile.views {
                // Validate view key is a recognized facet
                if Facet::from_key(key).is_none() {
                    errors.push(format!(
                        "Profile '{}': unknown view key '{}'",
                        profile_name, key
                    ));
                    continue;
                }

                // Reject paths into managed directories
                let path = Path::new(view.path());
                if let Some(first) = path.components().next() {
                    let s = first.as_os_str().to_string_lossy();
                    if s == ".scratch" || s == ".cache" {
                        errors.push(format!(
                            "Profile '{}', view '{}': path '{}' must not be under managed directory '{}'",
                            profile_name, key, view.path(), s
                        ));
                    }
                }
            }
        }

        errors
    }

    /// Generate a scaffold dataset.yaml with all facet keys in preferred order
    /// under a single default profile.
    pub fn scaffold(name: &str) -> Self {
        let mut views = IndexMap::new();

        for &facet in Facet::PREFERRED_ORDER {
            let out_ext = facet.default_format().name();
            views.insert(
                facet.key().to_string(),
                DSView {
                    source: DSSource {
                        path: format!("{}.{}", facet.key(), out_ext),
                        namespace: None,
                        window: Default::default(),
                    },
                    window: None,
                },
            );
        }

        let default_profile = DSProfile {
            maxk: None,
            views,
        };

        let mut profiles = IndexMap::new();
        profiles.insert("default".to_string(), default_profile);

        DatasetConfig {
            name: name.to_string(),
            description: None,
            upstream: None,
            profiles: DSProfileGroup(profiles),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaffold_generation() {
        let config = DatasetConfig::scaffold("test-dataset");
        assert_eq!(config.name, "test-dataset");
        let dp = config.default_profile().unwrap();
        assert!(dp.views.contains_key("base_vectors"));
        assert!(dp.views.contains_key("metadata_content"));
        assert!(dp.views.contains_key("metadata_results"));
        assert_eq!(dp.views.len(), 10);
    }

    #[test]
    fn test_scaffold_ordering() {
        let config = DatasetConfig::scaffold("test");
        let dp = config.default_profile().unwrap();
        let keys: Vec<&str> = dp.views.keys().map(|k| k.as_str()).collect();
        assert_eq!(keys[0], "base_vectors");
        assert_eq!(keys[1], "query_vectors");
        assert_eq!(keys[2], "neighbor_indices");
        assert_eq!(keys[3], "neighbor_distances");
        assert_eq!(keys[4], "metadata_content");
        assert_eq!(keys[5], "metadata_predicates");
        assert_eq!(keys[6], "metadata_results");
    }

    #[test]
    fn test_scaffold_roundtrip() {
        let config = DatasetConfig::scaffold("roundtrip-test");
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: DatasetConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.name, "roundtrip-test");
        let dp = parsed.default_profile().unwrap();
        assert_eq!(dp.views.len(), 10);
    }

    #[test]
    fn test_validate_rejects_scratch_path() {
        let yaml = r#"
name: test
profiles:
  default:
    base_vectors: ".scratch/base.fvec"
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let errors = config.validate(Path::new("."));
        assert!(
            errors.iter().any(|e| e.contains(".scratch")),
            "expected .scratch rejection, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_rejects_cache_path() {
        let yaml = r#"
name: test
profiles:
  default:
    query_vectors: ".cache/query.fvec"
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let errors = config.validate(Path::new("."));
        assert!(
            errors.iter().any(|e| e.contains(".cache")),
            "expected .cache rejection, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_unknown_view() {
        let yaml = r#"
name: test
profiles:
  default:
    nonexistent_facet: "x.fvec"
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let errors = config.validate(Path::new("."));
        assert!(errors.iter().any(|e| e.contains("unknown view key")));
    }

    #[test]
    fn test_profiles_yaml_parse() {
        let yaml = r#"
name: bigann
description: BIGANN 1B benchmark
profiles:
  default:
    maxk: 100
    base_vectors: base_vectors.fvec
    query_vectors: query_vectors.fvec
    neighbor_indices: gnd/idx.ivecs
    neighbor_distances: gnd/dis.fvecs
  1M:
    base_vectors:
      source: base_vectors.fvec
      window: "0..1000000"
    neighbor_indices: gnd/idx_1M.ivecs
    neighbor_distances: gnd/dis_1M.fvecs
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "bigann");
        assert_eq!(config.profile_names().len(), 2);

        let dp = config.default_profile().unwrap();
        assert_eq!(dp.maxk, Some(100));
        assert_eq!(dp.views.len(), 4);

        let child = config.profile("1M").unwrap();
        // Inherited maxk
        assert_eq!(child.maxk, Some(100));
        // Inherited query_vectors from default
        assert!(child.view("query_vectors").is_some());
        assert_eq!(child.view("query_vectors").unwrap().path(), "query_vectors.fvec");
    }

    #[test]
    fn test_profiles_with_aliases() {
        let yaml = r#"
name: test-aliases
profiles:
  default:
    base: base.fvec
    query: query.fvec
    indices: gnd/idx.ivecs
    distances: gnd/dis.fvecs
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let dp = config.default_profile().unwrap();
        assert!(dp.views.contains_key("base_vectors"));
        assert!(dp.views.contains_key("query_vectors"));
        assert!(dp.views.contains_key("neighbor_indices"));
        assert!(dp.views.contains_key("neighbor_distances"));
    }

    #[test]
    fn test_view_names_accessor() {
        let yaml = r#"
name: test
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let names = config.view_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_empty_profiles() {
        let yaml = r#"
name: minimal
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.profiles.is_empty());
        assert!(config.default_profile().is_none());
    }

    #[test]
    fn test_config_with_upstream_and_profiles() {
        let yaml = r#"
name: test-pipeline
upstream:
  steps:
    - run: import facet
      facet: base_vectors
      source: data/base.npy
      output: base_vectors.fvec
profiles:
  default:
    base_vectors: base_vectors.fvec
    neighbor_indices: neighbor_indices.ivec
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.upstream.is_some());
        assert_eq!(config.default_profile().unwrap().views.len(), 2);
    }
}
