// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset YAML configuration — the top-level `dataset.yaml` model.
//!
//! A `dataset.yaml` file describes a complete benchmark or test dataset using
//! named profiles. Each profile maps view names (facet names) to data sources.
//! A `"default"` profile provides the baseline configuration; other profiles
//! inherit from it automatically.
//!
//! The optional top-level `upstream` section defines a multi-step pipeline
//! for building the dataset from source data.

use std::path::Path;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::facet::StandardFacet;
use super::pipeline::PipelineConfig;
use super::profile::{DSProfile, DSProfileGroup};

/// Dataset-level attributes — metadata describing the dataset itself.
///
/// Mirrors the upstream Java `RootGroupAttributes` and is emitted as the
/// `layout.attributes` block in catalog entries. All fields are optional;
/// only non-`None` values are serialized.
///
/// Well-known keys: `model`, `url`, `distance_function`, `license`,
/// `vendor`, `notes`. Additional freeform tags live under `tags`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct DatasetAttributes {
    /// Name of the embedding model used to generate vectors (e.g. "CLIP ViT-B/32").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Canonical URL for the dataset source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Distance function — COSINE, L2, DOT_PRODUCT, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance_function: Option<String>,

    /// License identifier (e.g. "Apache-2.0", "CC-BY-4.0").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Dataset vendor/provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,

    /// Freeform notes about the dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,

    /// Freeform key-value tags for categorization.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub tags: IndexMap<String, String>,
}

/// Top-level `dataset.yaml` configuration.
///
/// Describes the dataset name, attributes, profiles (view mappings), and an
/// optional upstream pipeline for building the dataset from source data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset name.
    pub name: String,

    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Dataset-level attributes (model, distance function, license, etc.).
    ///
    /// Emitted as the `layout.attributes` block in catalog entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<DatasetAttributes>,

    /// Top-level pipeline configuration (shared defaults and steps).
    ///
    /// When present, a pipeline runner can execute a full multi-step pipeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<PipelineConfig>,

    /// Named profiles, each mapping view names to data sources.
    #[serde(default)]
    pub profiles: DSProfileGroup,
}

impl DatasetConfig {
    /// Load a `DatasetConfig` from a YAML file on disk.
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

    /// Returns view names from the default profile (for display).
    pub fn view_names(&self) -> Vec<&str> {
        self.profiles.view_names()
    }

    /// Validate structural rules for all profiles.
    ///
    /// Checks that source paths don't point into managed directories
    /// (`.scratch/`, `.cache/`). Custom facet keys are allowed and not
    /// flagged as errors.
    ///
    /// Returns a list of error messages.
    pub fn validate(&self, _base_dir: &Path) -> Vec<String> {
        let mut errors = Vec::new();

        for (profile_name, profile) in &self.profiles.0 {
            for (key, view) in &profile.views {
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

    /// Returns the standard facet keys present in the default profile.
    pub fn standard_facet_keys(&self) -> Vec<&str> {
        self.default_profile()
            .map(|p| {
                p.views
                    .keys()
                    .filter(|k| StandardFacet::from_key(k).is_some())
                    .map(|k| k.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the custom (non-standard) facet keys present in the default profile.
    pub fn custom_facet_keys(&self) -> Vec<&str> {
        self.default_profile()
            .map(|p| {
                p.views
                    .keys()
                    .filter(|k| StandardFacet::from_key(k).is_none())
                    .map(|k| k.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_validate_allows_custom_facets() {
        let yaml = r#"
name: test
profiles:
  default:
    base_vectors: base.fvec
    model_profile: model.json
    sketch_vectors: sketch.mvec
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let errors = config.validate(Path::new("."));
        assert!(errors.is_empty(), "custom facets should not produce errors: {:?}", errors);
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
    - run: import
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

    #[test]
    fn test_standard_and_custom_facet_keys() {
        let yaml = r#"
name: test
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
    model_profile: model.json
    sketch_vectors: sketch.mvec
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let standard = config.standard_facet_keys();
        let custom = config.custom_facet_keys();
        assert_eq!(standard.len(), 2);
        assert!(standard.contains(&"base_vectors"));
        assert!(standard.contains(&"query_vectors"));
        assert_eq!(custom.len(), 2);
        assert!(custom.contains(&"model_profile"));
        assert!(custom.contains(&"sketch_vectors"));
    }
}
