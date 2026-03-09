// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset YAML configuration — re-exports from `dataset` crate with
//! veks-specific extensions.
//!
//! The core `DatasetConfig`, profile, and pipeline types are owned by the
//! `dataset` crate. This module re-exports `DatasetConfig` and adds the
//! `scaffold()` function which depends on veks-specific types (`Facet`,
//! `VecFormat`).

pub use dataset::DatasetConfig;

use indexmap::IndexMap;

use dataset::{DSProfile, DSProfileGroup, DSSource, DSView};
use super::facet::Facet;

/// Extension trait adding veks-specific operations to `DatasetConfig`.
pub trait DatasetConfigExt {
    /// Generate a scaffold dataset.yaml with all facet keys in preferred order
    /// under a single default profile.
    fn scaffold(name: &str) -> DatasetConfig;
}

impl DatasetConfigExt for DatasetConfig {
    fn scaffold(name: &str) -> DatasetConfig {
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
            base_count: None,
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
    use std::path::Path;

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
    fn test_validate_allows_custom_facets() {
        let yaml = r#"
name: test
profiles:
  default:
    base_vectors: base.fvec
    nonexistent_facet: x.fvec
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
        assert_eq!(child.maxk, Some(100));
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
}
