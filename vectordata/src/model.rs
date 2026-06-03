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
//
// After parsing, applies default-profile inheritance: non-default
// profiles that omit shared facets (`base_vectors`, `query_vectors`,
// etc.) pick them up from `default`. `base_vectors` and
// `metadata_content` additionally receive a `[0..base_count)` window
// when the child profile sets `base_count`. Mirrors the inheritance
// the pipeline-side `DSProfileGroup` deserializer already implements
// — without it, every sized profile in a canonical `dataset.yaml`
// loses its base/query facets and precache (or any consumer that
// iterates `facet_manifest`) silently skips them.
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

        apply_default_inheritance(&mut profiles);

        Ok(DatasetConfig {
            attributes: raw.attributes,
            profiles,
        })
    }
}

/// Apply default-profile inheritance. Non-default, non-partition
/// profiles inherit any missing shared-facet field from `default`.
/// `base_vectors` and `metadata_content` additionally receive a
/// `[0..base_count)` window suffix when the child profile declares
/// `base_count`. Per-profile output facets (`neighbor_*`,
/// `prefiltered_*`, `postfiltered_*`) are never inherited — a sized
/// profile's GT lives at `profiles/<name>/neighbor_*.ivecs`, not at
/// the default's path.
fn apply_default_inheritance(profiles: &mut HashMap<String, ProfileConfig>) {
    let default = match profiles.get("default").cloned() {
        Some(d) => d,
        None => return,
    };
    for (name, profile) in profiles.iter_mut() {
        if name == "default" { continue; }
        // Partition profiles are self-contained: they carry their
        // own base bytes and must not pick up default's shared facets.
        if profile.partition { continue; }

        let bc = profile.base_count;
        inherit_with_window(&mut profile.base_vectors, &default.base_vectors, bc);
        inherit_with_window(&mut profile.metadata_content, &default.metadata_content, bc);
        inherit(&mut profile.base_content, &default.base_content);
        inherit(&mut profile.query_vectors, &default.query_vectors);
        inherit(&mut profile.query_terms, &default.query_terms);
        inherit(&mut profile.query_filters, &default.query_filters);
        inherit(&mut profile.metadata_predicates, &default.metadata_predicates);
        inherit(&mut profile.predicate_results, &default.predicate_results);
        inherit(&mut profile.metadata_layout, &default.metadata_layout);
        if profile.maxk.is_none() { profile.maxk = default.maxk; }
    }
}

/// Copy `source` into `target` when `target.is_none()`.
fn inherit(target: &mut Option<FacetConfig>, source: &Option<FacetConfig>) {
    if target.is_none() {
        *target = source.clone();
    }
}

/// Inherit `source` into `target` and apply a `[0..base_count)`
/// window suffix to its source path. The window is *only* applied to
/// the inherited copy — an explicit per-profile facet is left alone.
/// No-op when `base_count.is_none()` (no meaningful window to apply)
/// or when the source path already has a `[...]` window suffix.
fn inherit_with_window(
    target: &mut Option<FacetConfig>,
    source: &Option<FacetConfig>,
    base_count: Option<u64>,
) {
    if target.is_some() { return; }
    let Some(src) = source.clone() else { return; };
    let Some(bc) = base_count else {
        *target = Some(src);
        return;
    };
    let windowed = match src {
        FacetConfig::Simple(path) => {
            if path.contains('[') {
                FacetConfig::Simple(path)
            } else {
                FacetConfig::Simple(format!("{path}[0..{bc})"))
            }
        }
        FacetConfig::Detailed { source, window } => {
            if window.is_some() || source.contains('[') {
                FacetConfig::Detailed { source, window }
            } else {
                FacetConfig::Detailed {
                    source,
                    window: Some(format!("0..{bc}")),
                }
            }
        }
    };
    *target = Some(windowed);
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

    // -- Filtered neighbor facets (F / E per docs/design/prefilter-postfilter-facets.md) --

    /// Pre-filter KNN ground-truth indices (**F** facet). Top-K over `X_p`
    /// (the predicate-passing base vectors). Full K when `|X_p| ≥ K`;
    /// perfect recall by construction. This is ACORN's `G_K` — the legacy
    /// filtered-knn shape.
    ///
    /// The legacy alias `filtered_neighbor_indices` resolves here, because
    /// files produced by the legacy `compute filtered-knn` carry pre-filter
    /// shape on disk.
    #[serde(default, alias = "filtered_neighbor_indices", alias = "prefilter_indices")]
    pub prefiltered_neighbor_indices: Option<FacetConfig>,
    /// Pre-filter KNN ground-truth distances (**F** facet).
    #[serde(default, alias = "filtered_neighbor_distances", alias = "prefilter_distances")]
    pub prefiltered_neighbor_distances: Option<FacetConfig>,

    /// Post-filter KNN ground-truth indices (**E** facet). `G ∩ R` — the
    /// unfiltered top-K intersected with the predicate-passing set.
    /// Sparse possible. New facet introduced alongside the F/E split.
    #[serde(default, alias = "postfilter_indices")]
    pub postfiltered_neighbor_indices: Option<FacetConfig>,
    /// Post-filter KNN ground-truth distances (**E** facet).
    #[serde(default, alias = "postfilter_distances")]
    pub postfiltered_neighbor_distances: Option<FacetConfig>,

    // -- Metadata facets --

    /// Metadata content records (MNode-encoded slab).
    pub metadata_content: Option<FacetConfig>,
    /// Metadata predicate trees (PNode-encoded slab).
    pub metadata_predicates: Option<FacetConfig>,
    /// Predicate result indices — ordinals matching metadata records for each
    /// predicate. Canonical key `metadata_results`; the legacy `metadata_indices`
    /// and the old field name `predicate_results` are accepted as aliases.
    #[serde(rename = "metadata_results", alias = "metadata_indices", alias = "predicate_results")]
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

    /// Legacy YAML keys `filtered_neighbor_indices` /
    /// `filtered_neighbor_distances` MUST populate the **pre-filter (F)**
    /// fields — files produced by the legacy `compute filtered-knn`
    /// carry pre-filter shape on disk, so the alias points at the
    /// matching typed slot. Regression-pin per
    /// `docs/design/prefilter-postfilter-facets.md` §3.1.
    #[test]
    fn test_legacy_filtered_yaml_keys_populate_prefiltered() {
        let yaml = r#"
profiles:
  default:
    base_vectors: base.fvec
    filtered_neighbor_indices: filtered.ivec
    filtered_neighbor_distances: filtered.fvec
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let profile = config.profiles.get("default").unwrap();

        // Legacy aliases populate the canonical F (pre-filter) fields.
        assert_eq!(
            profile.prefiltered_neighbor_indices.as_ref().map(|f| f.source()),
            Some("filtered.ivec"),
        );
        assert_eq!(
            profile.prefiltered_neighbor_distances.as_ref().map(|f| f.source()),
            Some("filtered.fvec"),
        );
        // E (post-filter) stays unset when only the legacy keys are used.
        assert!(profile.postfiltered_neighbor_indices.is_none());
        assert!(profile.postfiltered_neighbor_distances.is_none());
    }

    /// Canonical `prefiltered_*` and `postfiltered_*` keys parse into
    /// their respective fields and never collide.
    #[test]
    fn test_canonical_e_and_f_yaml_keys_parse() {
        let yaml = r#"
profiles:
  default:
    base_vectors: base.fvec
    prefiltered_neighbor_indices: prefiltered.ivec
    prefiltered_neighbor_distances: prefiltered.fvec
    postfiltered_neighbor_indices: postfiltered.ivec
    postfiltered_neighbor_distances: postfiltered.fvec
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let profile = config.profiles.get("default").unwrap();

        assert_eq!(
            profile.prefiltered_neighbor_indices.as_ref().map(|f| f.source()),
            Some("prefiltered.ivec"),
        );
        assert_eq!(
            profile.prefiltered_neighbor_distances.as_ref().map(|f| f.source()),
            Some("prefiltered.fvec"),
        );
        assert_eq!(
            profile.postfiltered_neighbor_indices.as_ref().map(|f| f.source()),
            Some("postfiltered.ivec"),
        );
        assert_eq!(
            profile.postfiltered_neighbor_distances.as_ref().map(|f| f.source()),
            Some("postfiltered.fvec"),
        );
    }

    /// Sized profile inherits shared facets from default and applies
    /// `[0..base_count)` window to base_vectors. Without inheritance,
    /// the precache iterator over `facet_manifest()` silently skips
    /// base/query for every sized profile in a canonical
    /// `dataset.yaml`.
    #[test]
    fn sized_profile_inherits_shared_facets_with_windowing() {
        let yaml = r#"
attributes:
  distance_function: COSINE
profiles:
  default:
    base_vectors: profiles/base/base_vectors.fvecs
    query_vectors: profiles/base/query_vectors.fvecs
    neighbor_indices: profiles/default/neighbor_indices.ivecs
    neighbor_distances: profiles/default/neighbor_distances.fvecs
  10m:
    base_count: 10000000
    neighbor_indices: profiles/10m/neighbor_indices.ivecs
    neighbor_distances: profiles/10m/neighbor_distances.fvecs
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let p = config.profiles.get("10m").expect("10m profile must parse");

        // Inherited from default with windowing applied.
        let bv = p.base_vectors.as_ref().expect("base_vectors inherited from default");
        assert_eq!(bv.source(), "profiles/base/base_vectors.fvecs[0..10000000)",
            "inherited base_vectors must carry a [0..base_count) window suffix");

        // Inherited as-is (query sets are shared across profiles).
        assert_eq!(
            p.query_vectors.as_ref().map(|f| f.source()),
            Some("profiles/base/query_vectors.fvecs"),
            "query_vectors must be inherited from default without windowing");

        // Per-profile output facets keep their own paths.
        assert_eq!(
            p.neighbor_indices.as_ref().map(|f| f.source()),
            Some("profiles/10m/neighbor_indices.ivecs"));
        assert_eq!(
            p.neighbor_distances.as_ref().map(|f| f.source()),
            Some("profiles/10m/neighbor_distances.fvecs"));
    }

    /// Partition profiles (`partition: true`) own their own base
    /// bytes and must NOT pick up default's shared facets, even when
    /// they don't declare their own. Defended explicitly because
    /// silent inheritance would mis-route partition reads to the
    /// default's full-base file.
    #[test]
    fn partition_profile_does_not_inherit_from_default() {
        let yaml = r#"
attributes: {}
profiles:
  default:
    base_vectors: profiles/base/base_vectors.fvecs
    query_vectors: profiles/base/query_vectors.fvecs
  label_03:
    partition: true
    base_vectors: profiles/label_03/base.fvecs
    neighbor_indices: profiles/label_03/neighbor_indices.ivecs
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let p = config.profiles.get("label_03").expect("label_03 profile must parse");

        // Own base_vectors — no windowing applied.
        assert_eq!(p.base_vectors.as_ref().map(|f| f.source()),
            Some("profiles/label_03/base.fvecs"));
        // query_vectors NOT inherited from default — partition profiles
        // are self-contained.
        assert!(p.query_vectors.is_none(),
            "partition profile must not inherit query_vectors from default");
    }

    /// When the sized profile has no `base_count`, inherited facets
    /// pass through unwindowed. The caller's responsibility to set
    /// base_count if they want windowing; absent that, we don't
    /// invent a value.
    #[test]
    fn sized_profile_without_base_count_inherits_without_window() {
        let yaml = r#"
attributes: {}
profiles:
  default:
    base_vectors: base.fvecs
  derived:
    neighbor_indices: derived/neighbor_indices.ivecs
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let p = config.profiles.get("derived").unwrap();
        assert_eq!(p.base_vectors.as_ref().map(|f| f.source()),
            Some("base.fvecs"),
            "no base_count → inherit unwindowed");
    }
}
