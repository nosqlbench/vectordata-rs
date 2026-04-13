// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Parser for `knn_entries.yaml` — jvector-compatible dataset index.
//!
//! `knn_entries.yaml` is a simpler alternative to `dataset.yaml` for
//! describing vector datasets with base, query, and ground-truth files.
//! It's the format used by jvector's `DataSetLoaderSimpleMFD`.
//!
//! ## Format
//!
//! ```yaml
//! _defaults:
//!   base_url: https://example.com/datasets
//!
//! "dataset-name:profile":
//!   base: path/to/base.fvec
//!   query: path/to/query.fvec
//!   gt: path/to/groundtruth.ivec
//! ```

use std::collections::HashMap;
use std::path::Path;

use indexmap::IndexMap;
use serde::Deserialize;

use crate::model::{DatasetConfig, FacetConfig, ProfileConfig};

/// A single knn_entries.yaml entry.
#[derive(Debug, Clone, Deserialize)]
pub struct KnnEntry {
    pub base: String,
    pub query: String,
    pub gt: String,
}

/// Parsed knn_entries.yaml file.
#[derive(Debug)]
pub struct KnnEntries {
    pub base_url: Option<String>,
    pub entries: IndexMap<String, KnnEntry>,
}

impl KnnEntries {
    /// Parse a knn_entries.yaml string.
    pub fn parse(yaml: &str) -> Result<Self, String> {
        let raw: IndexMap<String, serde_yaml::Value> = serde_yaml::from_str(yaml)
            .map_err(|e| format!("parse knn_entries.yaml: {}", e))?;

        let mut base_url: Option<String> = None;
        let mut entries = IndexMap::new();

        for (key, value) in &raw {
            if key == "_defaults" {
                if let Some(url) = value.get("base_url").and_then(|v| v.as_str()) {
                    base_url = Some(url.trim_end_matches('/').to_string());
                }
                continue;
            }
            let entry: KnnEntry = serde_yaml::from_value(value.clone())
                .map_err(|e| format!("parse entry '{}': {}", key, e))?;
            entries.insert(key.clone(), entry);
        }

        Ok(KnnEntries { base_url, entries })
    }

    /// Load from a file path.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        Self::parse(&content)
    }

    /// Convert to a `DatasetConfig` for use with `TestDataGroup`.
    ///
    /// Groups entries by dataset name (before the `:`) and creates
    /// profiles for each.
    pub fn to_config(&self) -> DatasetConfig {
        let mut datasets: IndexMap<String, IndexMap<String, &KnnEntry>> = IndexMap::new();
        for (key, entry) in &self.entries {
            let (ds_name, profile_name) = if let Some(pos) = key.find(':') {
                (key[..pos].to_string(), key[pos + 1..].to_string())
            } else {
                (key.clone(), "default".to_string())
            };
            datasets.entry(ds_name).or_default().insert(profile_name, entry);
        }

        // Use the first dataset
        let (_ds_name, profiles_map) = match datasets.into_iter().next() {
            Some(pair) => pair,
            None => return DatasetConfig {
                attributes: HashMap::new(),
                profiles: HashMap::new(),
            },
        };

        let mut profiles = HashMap::new();
        for (profile_name, entry) in profiles_map {
            profiles.insert(profile_name, ProfileConfig {
                maxk: None,
                base_count: None,
                partition: false,
                base_vectors: Some(FacetConfig::Simple(entry.base.clone())),
                base_content: None,
                query_vectors: Some(FacetConfig::Simple(entry.query.clone())),
                query_terms: None,
                query_filters: None,
                neighbor_indices: Some(FacetConfig::Simple(entry.gt.clone())),
                neighbor_distances: None,
                filtered_neighbor_indices: None,
                filtered_neighbor_distances: None,
                metadata_content: None,
                metadata_predicates: None,
                predicate_results: None,
                metadata_layout: None,
            });
        }

        DatasetConfig {
            attributes: HashMap::new(),
            profiles,
        }
    }

    /// List dataset names (unique names before the `:` separator).
    pub fn dataset_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.entries.keys()
            .map(|k| k.split(':').next().unwrap_or(k).to_string())
            .collect();
        names.dedup();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let yaml = r#"
_defaults:
  base_url: https://example.com/data

"mydata:default":
  base: base.fvec
  query: query.fvec
  gt: gt.ivec

"mydata:10m":
  base: 10m/base.fvec
  query: query.fvec
  gt: 10m/gt.ivec
"#;
        let entries = KnnEntries::parse(yaml).unwrap();
        assert_eq!(entries.base_url, Some("https://example.com/data".to_string()));
        assert_eq!(entries.entries.len(), 2);
        assert_eq!(entries.entries["mydata:default"].base, "base.fvec");
        assert_eq!(entries.entries["mydata:10m"].gt, "10m/gt.ivec");
    }

    #[test]
    fn test_to_config() {
        let yaml = r#"
"sift1m:default":
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  gt: profiles/base/neighbor_indices.ivec
"#;
        let entries = KnnEntries::parse(yaml).unwrap();
        let config = entries.to_config();
        assert!(config.profiles.contains_key("default"));
        let default = &config.profiles["default"];
        assert_eq!(default.base_vectors.as_ref().unwrap().source(), "profiles/base/base_vectors.fvec");
        assert_eq!(default.neighbor_indices.as_ref().unwrap().source(), "profiles/base/neighbor_indices.ivec");
    }

    #[test]
    fn test_dataset_names() {
        let yaml = r#"
"ds1:default":
  base: a.fvec
  query: q.fvec
  gt: g.ivec
"ds2:default":
  base: c.fvec
  query: q2.fvec
  gt: g3.ivec
"#;
        let entries = KnnEntries::parse(yaml).unwrap();
        let names = entries.dataset_names();
        assert!(names.contains(&"ds1".to_string()));
        assert!(names.contains(&"ds2".to_string()));
    }
}
