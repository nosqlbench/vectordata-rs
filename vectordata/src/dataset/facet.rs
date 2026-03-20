// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Standard facet kinds for vector datasets.
//!
//! [`StandardFacet`] enumerates the well-known facet types that define a
//! dataset's layout: `base_vectors`, `query_vectors`, `neighbor_indices`,
//! `neighbor_distances`, metadata content/predicates/results/layout, and
//! their filtered variants.
//!
//! Each facet has a canonical key name (used in `dataset.yaml` profile
//! definitions) and a set of shorthand aliases for convenience. This module
//! provides name resolution without any dependency on file formats or CLI
//! frameworks.
//!
//! Consumers that need format-specific behavior (e.g., preferred output format)
//! should extend `StandardFacet` in their own crate.

use std::fmt;

/// Canonical facet kinds for predicated vector datasets.
///
/// Each variant identifies a specific role within a test dataset. The
/// canonical key name is used in `dataset.yaml` profile definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StandardFacet {
    /// Base vectors (the corpus to search)
    BaseVectors,
    /// Query vectors (the queries to run)
    QueryVectors,
    /// Ground-truth neighbor indices
    NeighborIndices,
    /// Ground-truth neighbor distances
    NeighborDistances,
    /// Metadata content records (MNode)
    MetadataContent,
    /// Metadata predicate trees (PNode)
    MetadataPredicates,
    /// Metadata filter result bitmaps
    MetadataResults,
    /// Metadata field layout schema
    MetadataLayout,
    /// Filtered ground-truth neighbor indices
    FilteredNeighborIndices,
    /// Filtered ground-truth neighbor distances
    FilteredNeighborDistances,
}

impl StandardFacet {
    /// All facets in preferred ordering for dataset.yaml.
    pub const PREFERRED_ORDER: &[StandardFacet] = &[
        Self::BaseVectors,
        Self::QueryVectors,
        Self::NeighborIndices,
        Self::NeighborDistances,
        Self::MetadataContent,
        Self::MetadataPredicates,
        Self::MetadataResults,
        Self::MetadataLayout,
        Self::FilteredNeighborIndices,
        Self::FilteredNeighborDistances,
    ];

    /// Canonical key name used in dataset.yaml.
    pub fn key(self) -> &'static str {
        match self {
            Self::BaseVectors => "base_vectors",
            Self::QueryVectors => "query_vectors",
            Self::NeighborIndices => "neighbor_indices",
            Self::NeighborDistances => "neighbor_distances",
            Self::MetadataContent => "metadata_content",
            Self::MetadataPredicates => "metadata_predicates",
            Self::MetadataResults => "metadata_results",
            Self::MetadataLayout => "metadata_layout",
            Self::FilteredNeighborIndices => "filtered_neighbor_indices",
            Self::FilteredNeighborDistances => "filtered_neighbor_distances",
        }
    }

    /// Parse a facet from its canonical key name.
    pub fn from_key(key: &str) -> Option<Self> {
        match key {
            "base_vectors" => Some(Self::BaseVectors),
            "query_vectors" => Some(Self::QueryVectors),
            "neighbor_indices" => Some(Self::NeighborIndices),
            "neighbor_distances" => Some(Self::NeighborDistances),
            "metadata_content" => Some(Self::MetadataContent),
            "metadata_predicates" => Some(Self::MetadataPredicates),
            "metadata_results" => Some(Self::MetadataResults),
            "metadata_layout" => Some(Self::MetadataLayout),
            "filtered_neighbor_indices" => Some(Self::FilteredNeighborIndices),
            "filtered_neighbor_distances" => Some(Self::FilteredNeighborDistances),
            _ => None,
        }
    }

    /// Resolve a shorthand alias to its canonical facet.
    ///
    /// Matches Java's `TestDataKind.OtherNames` — allows YAML authors to use
    /// shorter, more natural names that get normalized to canonical keys.
    pub fn from_alias(name: &str) -> Option<Self> {
        match name {
            "base" | "train" => Some(Self::BaseVectors),
            "query" | "queries" | "test" => Some(Self::QueryVectors),
            "indices" | "neighbors" | "ground_truth" | "gt" => Some(Self::NeighborIndices),
            "distances" => Some(Self::NeighborDistances),
            "content" | "meta_content" | "meta_base" => Some(Self::MetadataContent),
            "meta_predicates" => Some(Self::MetadataPredicates),
            "meta_results" | "predicate_results" => Some(Self::MetadataResults),
            "layout" | "meta_layout" => Some(Self::MetadataLayout),
            "filtered_indices" | "filtered_gt" | "filtered_ground_truth" => {
                Some(Self::FilteredNeighborIndices)
            }
            "filtered_distances" | "filtered_neighbors" => {
                Some(Self::FilteredNeighborDistances)
            }
            _ => None,
        }
    }

    /// Returns `true` if the given key is a recognized standard facet
    /// (either canonical or alias).
    pub fn is_standard(key: &str) -> bool {
        Self::from_key(key).is_some() || Self::from_alias(key).is_some()
    }

    /// Whether this facet's slab content uses MNode encoding.
    pub fn is_mnode(self) -> bool {
        matches!(self, Self::MetadataContent)
    }
}

/// Resolve a view key to its canonical facet name.
///
/// Tries `StandardFacet::from_key` first (exact match), then
/// `StandardFacet::from_alias` (shorthand names). Returns the canonical
/// key string, or `None` if the key is not a recognized standard facet.
///
/// Non-standard keys are treated as custom facets and should be
/// preserved as-is by the caller.
pub fn resolve_standard_key(key: &str) -> Option<String> {
    if let Some(f) = StandardFacet::from_key(key) {
        return Some(f.key().to_string());
    }
    if let Some(f) = StandardFacet::from_alias(key) {
        return Some(f.key().to_string());
    }
    None
}

impl fmt::Display for StandardFacet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facet_roundtrip() {
        for facet in StandardFacet::PREFERRED_ORDER {
            let key = facet.key();
            let parsed = StandardFacet::from_key(key).unwrap();
            assert_eq!(*facet, parsed);
        }
    }

    #[test]
    fn test_preferred_order_is_complete() {
        assert_eq!(StandardFacet::PREFERRED_ORDER.len(), 10);
    }

    #[test]
    fn test_from_alias_base() {
        assert_eq!(StandardFacet::from_alias("base"), Some(StandardFacet::BaseVectors));
        assert_eq!(StandardFacet::from_alias("train"), Some(StandardFacet::BaseVectors));
    }

    #[test]
    fn test_from_alias_query() {
        assert_eq!(StandardFacet::from_alias("query"), Some(StandardFacet::QueryVectors));
        assert_eq!(StandardFacet::from_alias("queries"), Some(StandardFacet::QueryVectors));
        assert_eq!(StandardFacet::from_alias("test"), Some(StandardFacet::QueryVectors));
    }

    #[test]
    fn test_from_alias_indices() {
        assert_eq!(StandardFacet::from_alias("indices"), Some(StandardFacet::NeighborIndices));
        assert_eq!(StandardFacet::from_alias("neighbors"), Some(StandardFacet::NeighborIndices));
        assert_eq!(StandardFacet::from_alias("ground_truth"), Some(StandardFacet::NeighborIndices));
        assert_eq!(StandardFacet::from_alias("gt"), Some(StandardFacet::NeighborIndices));
    }

    #[test]
    fn test_from_alias_unknown() {
        assert_eq!(StandardFacet::from_alias("unknown"), None);
        assert_eq!(StandardFacet::from_alias("base_vectors"), None); // canonical, not alias
    }

    #[test]
    fn test_is_standard() {
        assert!(StandardFacet::is_standard("base_vectors"));
        assert!(StandardFacet::is_standard("base")); // alias
        assert!(!StandardFacet::is_standard("my_custom_facet"));
    }

    #[test]
    fn test_resolve_standard_key_canonical() {
        assert_eq!(resolve_standard_key("base_vectors"), Some("base_vectors".to_string()));
    }

    #[test]
    fn test_resolve_standard_key_alias() {
        assert_eq!(resolve_standard_key("base"), Some("base_vectors".to_string()));
        assert_eq!(resolve_standard_key("gt"), Some("neighbor_indices".to_string()));
    }

    #[test]
    fn test_resolve_standard_key_custom() {
        assert_eq!(resolve_standard_key("my_custom_facet"), None);
    }
}
