//! Legacy data model types for `dataset.yaml` parsing.
//!
//! This module contains the original `DatasetConfig`, `ProfileConfig`, and
//! `FacetConfig` types used by [`TestDataGroup`](crate::group::TestDataGroup).
//! The newer, richer configuration model lives in [`crate::dataset::config`] and
//! [`crate::dataset::profile`].
//!
//! # Example `dataset.yaml`
//!
//! ```yaml
//! attributes:
//!   distance_function: COSINE
//!   dimension: 128
//!
//! profiles:
//!   default:
//!     base_vectors: base.mvec
//!     query_vectors: query.fvec
//!     neighbor_indices: ground_truth.ivec
//!     neighbor_distances: distances.fvec
//!
//!   small:
//!     base_vectors:
//!       source: base.mvec
//!       window: 0..1000
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Root configuration for a vector dataset.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetConfig {
    /// Arbitrary attributes describing the dataset (e.g., distance metric, dimension).
    pub attributes: HashMap<String, serde_yaml::Value>,
    /// Named profiles defining different views or subsets of the dataset.
    pub profiles: HashMap<String, ProfileConfig>,
}

// Hand-written deserialize so the client tolerates the compact `sized:`
// spec sitting next to concrete profile entries. The spec is a sequence
// (or mapping with a `ranges:` key) used by the pipeline as a shorthand
// for generating sized profiles; the client doesn't understand the
// grammar but must not reject datasets that carry it alongside the
// expanded entries. Any profile-map entry whose value isn't a struct
// (i.e. not a `ProfileConfig` shape) is skipped with a trace-level log.
impl<'de> Deserialize<'de> for DatasetConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            attributes: HashMap<String, serde_yaml::Value>,
            #[serde(default)]
            profiles: HashMap<String, serde_yaml::Value>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let mut profiles: HashMap<String, ProfileConfig> = HashMap::new();
        for (name, value) in raw.profiles {
            // Skip the compact sized-spec shorthand and any other
            // non-profile entries. A valid `ProfileConfig` is always
            // a mapping; sequences (`sized: ["mul:1m/2", ...]`) and
            // `sized: {ranges: [...], facets: {...}}` maps (which
            // contain no profile fields) are ignored here and
            // handled by the pipeline-side `vectordata::dataset`
            // parser instead.
            if name == "sized" {
                continue;
            }
            match serde_yaml::from_value::<ProfileConfig>(value) {
                Ok(cfg) => { profiles.insert(name, cfg); }
                Err(e) => {
                    log::trace!(
                        "skipping unparseable profile entry '{}': {}", name, e);
                }
            }
        }

        Ok(DatasetConfig {
            attributes: raw.attributes,
            profiles,
        })
    }
}

/// Configuration for a specific profile within a dataset.
///
/// A profile defines which files (facets) constitute the dataset view.
/// Facet names match the canonical keys from the Java `TestDataKind` enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Number of base vectors in this profile. Set for all profile types:
    /// default (full dataset), sized (windowed subset), partition (per-label).
    pub base_count: Option<u64>,

    /// Maximum k for KNN queries in this profile.
    pub maxk: Option<u32>,

    /// When true, this is an oracle partition profile with independent
    /// base vectors (not a windowed subset of the default profile).
    #[serde(default)]
    pub partition: bool,

    // -- Vector facets --

    /// Configuration for the base (database) vectors.
    pub base_vectors: Option<FacetConfig>,
    /// Optional original content associated with base vectors.
    pub base_content: Option<FacetConfig>,
    /// Configuration for the query vectors.
    pub query_vectors: Option<FacetConfig>,
    /// Optional query terms dataset.
    pub query_terms: Option<FacetConfig>,
    /// Optional query filters dataset.
    pub query_filters: Option<FacetConfig>,
    /// Configuration for the ground truth neighbor indices.
    pub neighbor_indices: Option<FacetConfig>,
    /// Configuration for the ground truth neighbor distances.
    pub neighbor_distances: Option<FacetConfig>,

    // -- Filtered neighbor facets --

    /// Filtered ground-truth neighbor indices (pre-conditioned on metadata predicates).
    pub filtered_neighbor_indices: Option<FacetConfig>,
    /// Filtered ground-truth neighbor distances (pre-conditioned on metadata predicates).
    pub filtered_neighbor_distances: Option<FacetConfig>,

    // -- Metadata facets --

    /// Metadata content records (MNode-encoded slab).
    pub metadata_content: Option<FacetConfig>,
    /// Metadata predicate trees (PNode-encoded slab).
    pub metadata_predicates: Option<FacetConfig>,
    /// Predicate result indices — ordinals matching metadata records for each predicate.
    #[serde(alias = "metadata_indices")]
    pub predicate_results: Option<FacetConfig>,
    /// Metadata layout describing the field schema.
    pub metadata_layout: Option<FacetConfig>,
}

/// Configuration for a single facet (file resource) of a dataset.
///
/// Can be a simple string (filename) or a detailed object with more options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FacetConfig {
    /// Simple filename string (e.g., "base.fvec").
    Simple(String),
    /// Detailed configuration object.
    Detailed {
        /// The source filename or path.
        source: String,
        /// Optional window/range string (e.g., "0..1000") to select a subset.
        #[serde(default)]
        window: Option<String>,
    },
}

impl FacetConfig {
    /// Returns the source path/filename.
    pub fn source(&self) -> &str {
        match self {
            FacetConfig::Simple(s) => s,
            FacetConfig::Detailed { source, .. } => source,
        }
    }

    /// Returns the optional window string.
    pub fn window(&self) -> Option<&str> {
        match self {
            FacetConfig::Simple(_) => None,
            FacetConfig::Detailed { window, .. } => window.as_deref(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_config() {
        let yaml = r#"
attributes:
  distance_function: COSINE
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.attributes.get("distance_function").unwrap().as_str().unwrap(), "COSINE");
        
        let profile = config.profiles.get("default").unwrap();
        assert_eq!(profile.base_vectors.as_ref().unwrap().source(), "base.fvec");
        assert!(profile.base_vectors.as_ref().unwrap().window().is_none());
    }

    #[test]
    fn test_parse_detailed_config() {
        let yaml = r#"
profiles:
  small:
    base_vectors:
      source: base.fvec
      window: 0..1000
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let profile = config.profiles.get("small").unwrap();
        
        match profile.base_vectors.as_ref().unwrap() {
            FacetConfig::Detailed { source, window } => {
                assert_eq!(source, "base.fvec");
                assert_eq!(window.as_deref(), Some("0..1000"));
            },
            _ => panic!("Expected Detailed config"),
        }
    }
}
