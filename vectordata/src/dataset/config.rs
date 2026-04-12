// Copyright (c) Jonathan Shook
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

    /// Whether the base vectors are L2-normalized (all norms ≈ 1.0).
    ///
    /// When true, cosine similarity and dot product produce identical
    /// rankings, and the dot-product kernel can be used for cosine metric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_normalized: Option<bool>,

    /// Whether the dataset has been deduplicated (no exact duplicate vectors).
    /// Required for publishable datasets.
    #[serde(default)]
    pub is_duplicate_vector_free: Option<bool>,

    /// Whether the dataset contains no zero vectors.
    /// Required for publishable datasets.
    #[serde(default)]
    pub is_zero_vector_free: Option<bool>,

    /// Version of veks that created this dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub veks_version: Option<String>,

    /// Build hash of veks that created this dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub veks_build: Option<String>,

    /// Freeform key-value tags for categorization.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub tags: IndexMap<String, String>,
}

impl DatasetAttributes {
    /// Check for required attributes. Returns a list of missing fields.
    pub fn missing_required(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.is_zero_vector_free.is_none() {
            missing.push("is_zero_vector_free");
        }
        if self.is_duplicate_vector_free.is_none() {
            missing.push("is_duplicate_vector_free");
        }
        missing
    }
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

    /// Pipeline-produced variables (counts, flags, etc.).
    ///
    /// Synced from `variables.yaml` into `dataset.yaml` after each pipeline
    /// run so that consumers (catalogs, access layer, explore) can see
    /// dataset properties without a separate file.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub variables: IndexMap<String, String>,
}

impl DatasetConfig {
    /// Load a `DatasetConfig` from a YAML file on disk.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Load a dataset config and resolve deferred sized profiles from
    /// `variables.yaml` in the same directory (if it exists).
    ///
    /// This is the preferred loader for any context that needs the
    /// complete profile set — catalog resolution, explore, probe, check.
    /// The pipeline runner uses `resolve_all_steps()` which calls this
    /// internally.
    pub fn load_and_resolve(path: &Path) -> Result<Self, String> {
        let mut config = Self::load(path)?;
        if config.profiles.has_deferred() {
            let workspace = path.parent().unwrap_or(std::path::Path::new("."));
            let vars_path = workspace.join("variables.yaml");
            if vars_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&vars_path) {
                    if let Ok(vars) = serde_yaml::from_str::<indexmap::IndexMap<String, String>>(&content) {
                        let base_count: u64 = vars.get("base_count")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0);
                        if base_count > 0 {
                            config.profiles.expand_deferred_sized(&vars, base_count);
                        }
                    }
                }
            }
        }
        Ok(config)
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

    // -- Attribute accessors --------------------------------------------------

    /// Distance function (e.g., `"COSINE"`, `"L2"`, `"DOT_PRODUCT"`).
    pub fn distance_function(&self) -> Option<&str> {
        self.attributes.as_ref().and_then(|a| a.distance_function.as_deref())
    }

    /// Whether the base vectors are L2-normalized.
    pub fn is_normalized(&self) -> Option<bool> {
        self.attributes.as_ref().and_then(|a| a.is_normalized)
    }

    /// Whether the dataset contains no zero vectors.
    pub fn is_zero_vector_free(&self) -> Option<bool> {
        self.attributes.as_ref().and_then(|a| a.is_zero_vector_free)
    }

    /// Whether the dataset has been deduplicated.
    pub fn is_duplicate_vector_free(&self) -> Option<bool> {
        self.attributes.as_ref().and_then(|a| a.is_duplicate_vector_free)
    }

    // -- Variable accessors ---------------------------------------------------

    /// Get a pipeline variable by name. Variables are dynamic key-value
    /// pairs produced by pipeline stages — there is no fixed schema.
    pub fn variable(&self, name: &str) -> Option<&str> {
        self.variables.get(name).map(|s| s.as_str())
    }

    /// Get a pipeline variable parsed as a u64.
    pub fn variable_u64(&self, name: &str) -> Option<u64> {
        self.variables.get(name).and_then(|s| s.parse().ok())
    }

    /// Get a pipeline variable parsed as a bool.
    pub fn variable_bool(&self, name: &str) -> Option<bool> {
        self.variables.get(name).and_then(|s| match s.as_str() {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        })
    }

    /// Iterate all variables as `(key, value)` pairs.
    pub fn variables(&self) -> impl Iterator<Item = (&str, &str)> {
        self.variables.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    // -- Mutation helpers ------------------------------------------------------

    /// Add or replace a profile in the config.
    pub fn set_profile(&mut self, name: &str, profile: DSProfile) {
        self.profiles.profiles.insert(name.to_string(), profile);
    }

    /// Set a variable in the config.
    pub fn set_variable(&mut self, key: &str, value: &str) {
        self.variables.insert(key.to_string(), value.to_string());
    }

    /// Set an attribute field. Returns `false` if the key is not recognized.
    pub fn set_attribute(&mut self, key: &str, value: &str) -> bool {
        let attrs = self.attributes.get_or_insert_with(DatasetAttributes::default);
        match key {
            "is_normalized" => { attrs.is_normalized = Some(value == "true"); true }
            "is_zero_vector_free" => { attrs.is_zero_vector_free = Some(value == "true"); true }
            "is_duplicate_vector_free" => { attrs.is_duplicate_vector_free = Some(value == "true"); true }
            "distance_function" => { attrs.distance_function = Some(value.to_string()); true }
            "model" => { attrs.model = Some(value.to_string()); true }
            "url" => { attrs.url = Some(value.to_string()); true }
            "license" => { attrs.license = Some(value.to_string()); true }
            "vendor" => { attrs.vendor = Some(value.to_string()); true }
            "notes" => { attrs.notes = Some(value.to_string()); true }
            "veks_version" => { attrs.veks_version = Some(value.to_string()); true }
            "veks_build" => { attrs.veks_build = Some(value.to_string()); true }
            "personality" => {
                attrs.tags.insert("personality".to_string(), value.to_string());
                true
            }
            _ => false,
        }
    }

    // -- Save ---------------------------------------------------------------

    /// Save the config to a YAML file with canonical field ordering.
    ///
    /// Field order: name, description, attributes, upstream, profiles, variables.
    /// The license header comment is preserved if the file already has one,
    /// or added if absent.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        // If the file exists, extract the leading comment block
        let existing_header = if path.exists() {
            std::fs::read_to_string(path).ok().and_then(|content| {
                let header: String = content.lines()
                    .take_while(|l| l.starts_with('#') || l.is_empty())
                    .map(|l| format!("{}\n", l))
                    .collect();
                if header.is_empty() { None } else { Some(header) }
            })
        } else {
            None
        };

        let header = existing_header.unwrap_or_else(|| {
            "# Copyright (c) Jonathan Shook\n# SPDX-License-Identifier: Apache-2.0\n\n".to_string()
        });

        let mut out = header;

        // name
        out.push_str(&format!("name: {}\n", self.name));

        // description
        if let Some(ref desc) = self.description {
            out.push_str(&format!("description: >-\n  {}\n", desc));
        }

        // attributes
        if let Some(ref attrs) = self.attributes {
            out.push_str("\nattributes:\n");
            if let Some(ref v) = attrs.distance_function { out.push_str(&format!("  distance_function: {}\n", v)); }
            if let Some(v) = attrs.is_normalized { out.push_str(&format!("  is_normalized: {}\n", v)); }
            if let Some(v) = attrs.is_zero_vector_free { out.push_str(&format!("  is_zero_vector_free: {}\n", v)); }
            if let Some(v) = attrs.is_duplicate_vector_free { out.push_str(&format!("  is_duplicate_vector_free: {}\n", v)); }
            if let Some(ref v) = attrs.model { out.push_str(&format!("  model: {}\n", v)); }
            if let Some(ref v) = attrs.url { out.push_str(&format!("  url: {}\n", v)); }
            if let Some(ref v) = attrs.license { out.push_str(&format!("  license: {}\n", v)); }
            if let Some(ref v) = attrs.vendor { out.push_str(&format!("  vendor: {}\n", v)); }
            if let Some(ref v) = attrs.notes { out.push_str(&format!("  notes: {}\n", v)); }
            if let Some(ref v) = attrs.veks_version { out.push_str(&format!("  veks_version: {}\n", v)); }
            if let Some(ref v) = attrs.veks_build { out.push_str(&format!("  veks_build: {}\n", v)); }
            for (k, v) in &attrs.tags {
                out.push_str(&format!("  {}: {}\n", k, v));
            }
        }

        // upstream
        if let Some(ref upstream) = self.upstream {
            let upstream_yaml = serde_yaml::to_string(upstream)
                .map_err(|e| format!("serialize upstream: {}", e))?;
            out.push_str("\nupstream:\n");
            for line in upstream_yaml.lines() {
                if line.is_empty() { out.push('\n'); }
                else { out.push_str(&format!("  {}\n", line)); }
            }
        }

        // profiles
        if !self.profiles.is_empty() || self.profiles.has_deferred() {
            out.push_str("\nprofiles:\n");

            // Sized entries — preserved as compact syntax for round-tripping
            if !self.profiles.raw_sized.is_empty() {
                let entries: Vec<&str> = self.profiles.raw_sized.iter()
                    .map(|s| s.as_str()).collect();
                out.push_str(&format!("  sized: [{}]\n", entries.join(", ")));
            }

            // Materialized profiles (skip those generated from sized: entries —
            // they'll be re-generated from the raw_sized entries on reload)
            for name in self.profiles.profile_names() {
                if self.profiles.sized_profile_names.contains(&name.to_string()) {
                    continue;
                }
                let profile = self.profiles.profiles.get(name).unwrap();
                out.push_str(&format!("  {}:\n", name));
                if let Some(maxk) = profile.maxk {
                    out.push_str(&format!("    maxk: {}\n", maxk));
                }
                if let Some(base_count) = profile.base_count {
                    out.push_str(&format!("    base_count: {}\n", base_count));
                }
                for (key, view) in &profile.views {
                    if view.window.is_none()
                        && view.source.namespace.is_none()
                        && view.source.window.is_empty()
                    {
                        out.push_str(&format!("    {}: {}\n", key, view.source.path));
                    } else {
                        out.push_str(&format!("    {}:\n", key));
                        out.push_str(&format!("      source: {}\n", view.source.path));
                        if let Some(ref w) = view.window {
                            out.push_str(&format!("      window: \"{}\"\n", w));
                        } else if !view.source.window.is_empty() {
                            out.push_str(&format!("      window: \"{}\"\n", view.source.window));
                        }
                    }
                }
            }
        }

        // variables
        if !self.variables.is_empty() {
            out.push_str("\nvariables:\n");
            for (k, v) in &self.variables {
                out.push_str(&format!("  {}: '{}'\n", k, v));
            }
        }

        std::fs::write(path, &out)
            .map_err(|e| format!("write {}: {}", path.display(), e))
    }

    // -------------------------------------------------------------------------

    /// Validate structural rules for all profiles.
    ///
    /// Checks that source paths don't point into managed directories
    /// (`.scratch/`, `.cache/`). Custom facet keys are allowed and not
    /// flagged as errors.
    ///
    /// Returns a list of error messages.
    pub fn validate(&self, _base_dir: &Path) -> Vec<String> {
        let mut errors = Vec::new();

        for (profile_name, profile) in &self.profiles.profiles {
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

    #[test]
    fn test_save_roundtrip() {
        let yaml = r#"
name: roundtrip-test
description: >-
  A test dataset
attributes:
  distance_function: Cosine
  is_normalized: true
profiles:
  default:
    maxk: 100
    base_vectors: base.fvec
    query_vectors: query.fvec
variables:
  base_count: '500'
  query_count: '100'
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        config.save(tmp.path()).unwrap();

        let saved = std::fs::read_to_string(tmp.path()).unwrap();

        // Re-parse and verify
        let reloaded: DatasetConfig = serde_yaml::from_str(&saved).unwrap();
        assert_eq!(reloaded.name, "roundtrip-test");
        assert_eq!(reloaded.distance_function(), Some("Cosine"));
        assert_eq!(reloaded.is_normalized(), Some(true));
        assert_eq!(reloaded.default_profile().unwrap().maxk, Some(100));
        assert_eq!(reloaded.variable("base_count"), Some("500"));
        assert_eq!(reloaded.variable("query_count"), Some("100"));
    }

    #[test]
    fn test_save_canonical_ordering() {
        let yaml = r#"
name: order-test
variables:
  x: '1'
profiles:
  default:
    base_vectors: base.fvec
attributes:
  distance_function: L2
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        config.save(tmp.path()).unwrap();

        let saved = std::fs::read_to_string(tmp.path()).unwrap();

        // Verify canonical ordering: name < attributes < profiles < variables
        let name_pos = saved.find("\nname:").or_else(|| if saved.starts_with("name:") { Some(0) } else { None });
        let attr_pos = saved.find("\nattributes:");
        let prof_pos = saved.find("\nprofiles:");
        let var_pos = saved.find("\nvariables:");

        assert!(name_pos.is_some(), "name: should exist");
        assert!(attr_pos.is_some(), "attributes: should exist");
        assert!(prof_pos.is_some(), "profiles: should exist");
        assert!(var_pos.is_some(), "variables: should exist");

        assert!(name_pos.unwrap() < attr_pos.unwrap(), "name before attributes");
        assert!(attr_pos.unwrap() < prof_pos.unwrap(), "attributes before profiles");
        assert!(prof_pos.unwrap() < var_pos.unwrap(), "profiles before variables");
    }

    #[test]
    fn test_save_preserves_header_comment() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let header = "# Custom header\n# Another line\n\n";
        std::fs::write(tmp.path(), format!("{}name: test\n", header)).unwrap();

        let config: DatasetConfig = serde_yaml::from_str("name: test\n").unwrap();
        config.save(tmp.path()).unwrap();

        let saved = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(saved.starts_with("# Custom header\n# Another line\n"),
            "header should be preserved:\n{}", saved);
    }

    #[test]
    fn test_save_with_partition_profiles() {
        let yaml = r#"
name: partition-test
profiles:
  default:
    maxk: 50
    base_vectors: base.fvec
    query_vectors: query.fvec
"#;
        let mut config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();

        // Add partition profiles via the API
        config.set_profile("label-0", DSProfile {
            maxk: Some(50),
            base_count: Some(100),
            views: {
                use super::super::profile::DSView;
                use super::super::source::{DSSource, DSWindow};
                let mut v = IndexMap::new();
                v.insert("base_vectors".to_string(), DSView {
                    source: DSSource { path: "profiles/label-0/base_vectors.fvec".into(), namespace: None, window: DSWindow::default() },
                    window: None,
                });
                v.insert("query_vectors".to_string(), DSView {
                    source: DSSource { path: "profiles/label-0/query_vectors.fvec".into(), namespace: None, window: DSWindow::default() },
                    window: None,
                });
                v
            },
        });

        let tmp = tempfile::NamedTempFile::new().unwrap();
        config.save(tmp.path()).unwrap();

        let saved = std::fs::read_to_string(tmp.path()).unwrap();
        let reloaded: DatasetConfig = serde_yaml::from_str(&saved).unwrap();

        assert!(reloaded.profile("label-0").is_some());
        assert_eq!(reloaded.profile("label-0").unwrap().base_count, Some(100));
        assert!(reloaded.profile("default").is_some());
    }

    #[test]
    fn test_set_attribute() {
        let mut config: DatasetConfig = serde_yaml::from_str("name: test\n").unwrap();
        assert!(config.set_attribute("is_normalized", "true"));
        assert!(config.set_attribute("distance_function", "Cosine"));
        assert!(!config.set_attribute("nonexistent_field", "value"));

        assert_eq!(config.is_normalized(), Some(true));
        assert_eq!(config.distance_function(), Some("Cosine"));
    }
}
