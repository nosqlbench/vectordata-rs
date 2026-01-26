//! Data models for parsing `dataset.yaml` configuration files.
//!
//! This module defines the structures that map to the `dataset.yaml` file format used
//! to describe vector datasets.
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
//!     base_vectors: base.fvec
//!     query_vectors: query.fvec
//!     neighbor_indices: ground_truth.ivec
//!     neighbor_distances: distances.fvec
//!   
//!   small:
//!     base_vectors:
//!       source: base.fvec
//!       window: 0..1000
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Root configuration for a vector dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Arbitrary attributes describing the dataset (e.g., distance metric, dimension).
    #[serde(default)]
    pub attributes: HashMap<String, serde_yaml::Value>,
    /// Named profiles defining different views or subsets of the dataset.
    pub profiles: HashMap<String, ProfileConfig>,
}

/// Configuration for a specific profile within a dataset.
///
/// A profile defines which files (facets) constitute the dataset view (e.g., base vectors, queries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Configuration for the base (database) vectors.
    pub base_vectors: Option<FacetConfig>,
    /// Configuration for the query vectors.
    pub query_vectors: Option<FacetConfig>,
    /// Configuration for the ground truth neighbor indices.
    pub neighbor_indices: Option<FacetConfig>,
    /// Configuration for the ground truth neighbor distances.
    pub neighbor_distances: Option<FacetConfig>,
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
