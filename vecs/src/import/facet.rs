// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Facet — canonical data facet types for predicated datasets.
//!
//! Each facet identifies a specific role within a test dataset and has a
//! preferred output format for import. Facets have a canonical ordering
//! used in dataset.yaml scaffolds and display.

use std::fmt;

use clap::ValueEnum;

use crate::formats::VecFormat;

/// Canonical facet kinds as defined by the Java `TestDataKind` registry.
///
/// Listed in preferred ordering for dataset.yaml.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Facet {
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

impl Facet {
    /// All facets in preferred ordering for dataset.yaml
    pub const PREFERRED_ORDER: &[Facet] = &[
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

    /// The preferred output format for this facet, considering the source
    /// element size to select the correct xvec variant.
    ///
    /// For vector facets, the element size determines the xvec type:
    /// 1 → bvec, 2 → hvec (IEEE 754 half), 4 → fvec, 8 → dvec.
    /// Index facets always use ivec. Metadata facets always use slab.
    pub fn preferred_format(self, element_size: usize) -> VecFormat {
        match self {
            Self::BaseVectors
            | Self::QueryVectors
            | Self::NeighborDistances
            | Self::FilteredNeighborDistances => xvec_for_element_size(element_size),
            Self::NeighborIndices
            | Self::FilteredNeighborIndices => VecFormat::Ivec,
            Self::MetadataContent
            | Self::MetadataPredicates
            | Self::MetadataResults
            | Self::MetadataLayout => VecFormat::Slab,
        }
    }

    /// Default preferred format assuming f32 elements (for scaffold generation)
    pub fn default_format(self) -> VecFormat {
        self.preferred_format(4)
    }

    /// Generic upstream source extension for scaffold generation.
    ///
    /// Uses `xvec` for vector facets (import picks the correct variant based
    /// on element size), `ivec` for index facets, and `slab` for metadata.
    pub fn upstream_extension(self) -> &'static str {
        match self {
            Self::BaseVectors
            | Self::QueryVectors
            | Self::NeighborDistances
            | Self::FilteredNeighborDistances => "xvec",
            Self::NeighborIndices
            | Self::FilteredNeighborIndices => "ivec",
            Self::MetadataContent
            | Self::MetadataPredicates
            | Self::MetadataResults
            | Self::MetadataLayout => "slab",
        }
    }

    /// Canonical key name used in dataset.yaml
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

    /// Whether this facet's slab content uses MNode encoding
    pub fn is_mnode(self) -> bool {
        matches!(self, Self::MetadataContent)
    }

}

/// Map source element byte width to the appropriate xvec output format.
///
/// Element sizes per Java xvec implementations:
/// - fvec: 4 bytes (f32)
/// - ivec: 4 bytes (i32)
/// - bvec: 4 bytes (i32 — "byte vectors stored as integers")
/// - dvec: 8 bytes (f64)
/// - hvec: 2 bytes (IEEE 754 half-precision f16)
/// - svec: 2 bytes (i16)
fn xvec_for_element_size(element_size: usize) -> VecFormat {
    match element_size {
        2 => VecFormat::Hvec,
        4 => VecFormat::Fvec,
        8 => VecFormat::Dvec,
        _ => VecFormat::Fvec, // fallback to f32
    }
}

impl fmt::Display for Facet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facet_roundtrip() {
        for facet in Facet::PREFERRED_ORDER {
            let key = facet.key();
            let parsed = Facet::from_key(key).unwrap();
            assert_eq!(*facet, parsed);
        }
    }

    #[test]
    fn test_preferred_formats_f32() {
        assert_eq!(Facet::BaseVectors.preferred_format(4), VecFormat::Fvec);
        assert_eq!(Facet::NeighborIndices.preferred_format(4), VecFormat::Ivec);
        assert_eq!(Facet::MetadataContent.preferred_format(4), VecFormat::Slab);
        assert_eq!(Facet::MetadataPredicates.preferred_format(4), VecFormat::Slab);
    }

    #[test]
    fn test_preferred_formats_by_element_size() {
        assert_eq!(Facet::BaseVectors.preferred_format(2), VecFormat::Hvec);
        assert_eq!(Facet::BaseVectors.preferred_format(4), VecFormat::Fvec);
        assert_eq!(Facet::BaseVectors.preferred_format(8), VecFormat::Dvec);
        // Index facets always ivec regardless of element size
        assert_eq!(Facet::NeighborIndices.preferred_format(2), VecFormat::Ivec);
        assert_eq!(Facet::NeighborIndices.preferred_format(8), VecFormat::Ivec);
    }

    #[test]
    fn test_preferred_order_is_complete() {
        assert_eq!(Facet::PREFERRED_ORDER.len(), 10);
    }

    #[test]
    fn test_from_alias_base() {
        assert_eq!(Facet::from_alias("base"), Some(Facet::BaseVectors));
        assert_eq!(Facet::from_alias("train"), Some(Facet::BaseVectors));
    }

    #[test]
    fn test_from_alias_query() {
        assert_eq!(Facet::from_alias("query"), Some(Facet::QueryVectors));
        assert_eq!(Facet::from_alias("queries"), Some(Facet::QueryVectors));
        assert_eq!(Facet::from_alias("test"), Some(Facet::QueryVectors));
    }

    #[test]
    fn test_from_alias_indices() {
        assert_eq!(Facet::from_alias("indices"), Some(Facet::NeighborIndices));
        assert_eq!(Facet::from_alias("neighbors"), Some(Facet::NeighborIndices));
        assert_eq!(Facet::from_alias("ground_truth"), Some(Facet::NeighborIndices));
        assert_eq!(Facet::from_alias("gt"), Some(Facet::NeighborIndices));
    }

    #[test]
    fn test_from_alias_distances() {
        assert_eq!(Facet::from_alias("distances"), Some(Facet::NeighborDistances));
    }

    #[test]
    fn test_from_alias_metadata() {
        assert_eq!(Facet::from_alias("content"), Some(Facet::MetadataContent));
        assert_eq!(Facet::from_alias("meta_content"), Some(Facet::MetadataContent));
        assert_eq!(Facet::from_alias("meta_predicates"), Some(Facet::MetadataPredicates));
        assert_eq!(Facet::from_alias("meta_results"), Some(Facet::MetadataResults));
        assert_eq!(Facet::from_alias("predicate_results"), Some(Facet::MetadataResults));
        assert_eq!(Facet::from_alias("layout"), Some(Facet::MetadataLayout));
        assert_eq!(Facet::from_alias("meta_layout"), Some(Facet::MetadataLayout));
    }

    #[test]
    fn test_from_alias_filtered() {
        assert_eq!(Facet::from_alias("filtered_indices"), Some(Facet::FilteredNeighborIndices));
        assert_eq!(Facet::from_alias("filtered_gt"), Some(Facet::FilteredNeighborIndices));
        assert_eq!(Facet::from_alias("filtered_distances"), Some(Facet::FilteredNeighborDistances));
        assert_eq!(Facet::from_alias("filtered_neighbors"), Some(Facet::FilteredNeighborDistances));
    }

    #[test]
    fn test_from_alias_unknown() {
        assert_eq!(Facet::from_alias("unknown"), None);
        assert_eq!(Facet::from_alias("base_vectors"), None); // canonical, not alias
    }
}
