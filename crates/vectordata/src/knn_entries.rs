// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Parser for `knn_entries.yaml` — jvector-compatible dataset index.
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

use std::path::Path;

use indexmap::IndexMap;
use serde::Deserialize;

use crate::dataset::StandardFacet;

/// A single knn_entries.yaml entry.
#[derive(Debug, Clone, Deserialize)]
pub struct KnnEntry {
    /// Path or URL of the base vectors file.
    pub base: String,
    /// Path or URL of the query vectors file.
    pub query: String,
    /// Path or URL of the ground-truth neighbor indices file.
    pub gt: String,
}

/// Parsed knn_entries.yaml file.
#[derive(Debug)]
pub struct KnnEntries {
    /// `_defaults.base_url`, with any trailing `/` removed; `None` when
    /// absent.
    pub base_url: Option<String>,
    /// Entries keyed by `dataset:profile`, in file order.
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

    /// The description of the dataset named `dataset_name`, or of the
    /// file's first dataset when none has that name: the preference a
    /// directory open makes, matching the directory's name. A file with
    /// no entries describes a dataset with no profiles.
    pub fn description_for(&self, dataset_name: &str) -> Result<crate::dataset::DatasetConfig, String> {
        let mut by_dataset = self.group_by_dataset();
        let chosen = match by_dataset.shift_remove_entry(dataset_name) {
            Some(found) => Some(found),
            None => by_dataset.shift_remove_index(0),
        };
        let Some((name, profiles)) = chosen else {
            return crate::dataset::DatasetConfig::builder(dataset_name).build();
        };
        profiles
            .into_iter()
            .fold(crate::dataset::DatasetConfig::builder(name), |dataset, (profile, entry)| {
                dataset.profile(&profile, |p| p
                    .facet(StandardFacet::BaseVectors, &entry.base)
                    .facet(StandardFacet::QueryVectors, &entry.query)
                    .facet(StandardFacet::NeighborIndices, &entry.gt))
            })
            .build()
    }

    fn group_by_dataset(&self) -> IndexMap<String, IndexMap<String, &KnnEntry>> {
        let mut datasets: IndexMap<String, IndexMap<String, &KnnEntry>> = IndexMap::new();
        for (key, entry) in &self.entries {
            let (ds_name, profile_name) = if let Some(pos) = key.find(':') {
                (key[..pos].to_string(), key[pos + 1..].to_string())
            } else {
                (key.clone(), "default".to_string())
            };
            datasets.entry(ds_name).or_default().insert(profile_name, entry);
        }
        datasets
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
"myset:default":
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  gt: profiles/base/neighbor_indices.ivec
"#;
        let entries = KnnEntries::parse(yaml).unwrap();
        let config = crate::model::DatasetConfig::opened(entries.description_for("elsewhere").unwrap());
        assert!(config.profiles.contains_key("default"));
        let default = &config.profiles["default"];
        assert_eq!(default.base_vectors.as_ref().unwrap().source(), Some("profiles/base/base_vectors.fvec"));
        assert_eq!(default.neighbor_indices.as_ref().unwrap().source(), Some("profiles/base/neighbor_indices.ivec"));
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
