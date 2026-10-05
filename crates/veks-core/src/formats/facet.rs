// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Facet — canonical data facet types for predicated datasets.
//!
//! Each facet identifies a specific role within a test dataset and has a
//! preferred output format for import. Facets have a canonical ordering
//! used in dataset.yaml scaffolds and display.
//!
//! See `docs/design/prefilter-postfilter-facets.md` for the E (pre-filter)
//! vs F (post-filter) facet split. The legacy keys `filtered_neighbor_*`
//! resolve to the F (post-filter) facet via [`Facet::from_alias`].

use std::fmt;

use super::VecFormat;

/// Canonical facet kinds as defined by the Java `TestDataKind` registry.
///
/// Listed in preferred ordering for dataset.yaml.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Pre-filter KNN ground-truth indices — facet code `F`
    /// (the legacy filtered-knn shape; ACORN `G_K`)
    PrefilteredNeighborIndices,
    /// Pre-filter KNN ground-truth distances — facet code `F`
    PrefilteredNeighborDistances,
    /// Post-filter KNN ground-truth indices — facet code `E`
    /// (`G ∩ R`, sparse new artifact)
    PostfilteredNeighborIndices,
    /// Post-filter KNN ground-truth distances — facet code `E`
    PostfilteredNeighborDistances,
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
        Self::PrefilteredNeighborIndices,
        Self::PrefilteredNeighborDistances,
        Self::PostfilteredNeighborIndices,
        Self::PostfilteredNeighborDistances,
    ];

    /// The preferred output format for this facet, considering the source
    /// element size to select the correct xvec variant.
    ///
    /// For vector facets, the element size determines the xvec type:
    /// 1 → bvec, 2 → mvec (IEEE 754 half), 4 → fvec, 8 → dvec.
    /// Index facets always use ivec. Metadata facets always use slab.
    pub fn preferred_format(self, element_size: usize) -> VecFormat {
        match self {
            Self::BaseVectors
            | Self::QueryVectors
            | Self::NeighborDistances
            | Self::PrefilteredNeighborDistances
            | Self::PostfilteredNeighborDistances => xvec_for_element_size(element_size),
            Self::NeighborIndices
            | Self::PrefilteredNeighborIndices
            | Self::PostfilteredNeighborIndices => VecFormat::Ivec,
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
            | Self::PrefilteredNeighborDistances
            | Self::PostfilteredNeighborDistances => "xvec",
            Self::NeighborIndices
            | Self::PrefilteredNeighborIndices
            | Self::PostfilteredNeighborIndices => "ivec",
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
            Self::PrefilteredNeighborIndices => "prefiltered_neighbor_indices",
            Self::PrefilteredNeighborDistances => "prefiltered_neighbor_distances",
            Self::PostfilteredNeighborIndices => "postfiltered_neighbor_indices",
            Self::PostfilteredNeighborDistances => "postfiltered_neighbor_distances",
        }
    }

    /// Single-character facet code used by partition sub-facet scoping.
    pub fn code(self) -> Option<char> {
        match self {
            Self::BaseVectors => Some('B'),
            Self::QueryVectors => Some('Q'),
            Self::NeighborIndices => Some('G'),
            Self::NeighborDistances => Some('D'),
            Self::MetadataContent => Some('M'),
            Self::MetadataPredicates => Some('P'),
            Self::MetadataResults => Some('R'),
            Self::MetadataLayout => None,
            Self::PrefilteredNeighborIndices | Self::PrefilteredNeighborDistances => Some('F'),
            Self::PostfilteredNeighborIndices | Self::PostfilteredNeighborDistances => Some('E'),
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
            "prefiltered_neighbor_indices" => Some(Self::PrefilteredNeighborIndices),
            "prefiltered_neighbor_distances" => Some(Self::PrefilteredNeighborDistances),
            "postfiltered_neighbor_indices" => Some(Self::PostfilteredNeighborIndices),
            "postfiltered_neighbor_distances" => Some(Self::PostfilteredNeighborDistances),
            _ => None,
        }
    }

    /// Resolve a shorthand alias to its canonical facet.
    ///
    /// Matches Java's `TestDataKind.OtherNames` — allows YAML authors to use
    /// shorter, more natural names that get normalized to canonical keys.
    /// Legacy `filtered_neighbor_*` keys resolve to the F (pre-filter)
    /// facet — matching the actual shape produced by the legacy
    /// `compute filtered-knn` command. See `is_legacy_filtered_alias`.
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
            "prefiltered_indices" | "prefiltered_gt" | "prefiltered_ground_truth"
                | "prefilter_indices" => Some(Self::PrefilteredNeighborIndices),
            "prefiltered_distances" | "prefilter_distances" | "prefiltered_neighbors" => {
                Some(Self::PrefilteredNeighborDistances)
            }
            "postfiltered_indices" | "postfiltered_gt" | "postfiltered_ground_truth"
                | "postfilter_indices" => Some(Self::PostfilteredNeighborIndices),
            "postfiltered_distances" | "postfilter_distances" | "postfiltered_neighbors" => {
                Some(Self::PostfilteredNeighborDistances)
            }
            // Legacy — pre-E/F-split datasets used these for pre-filter
            // ground truth (now facet code F). They resolve to the
            // prefiltered variants so the on-disk shape matches the new
            // typing.
            "filtered_neighbor_indices" | "filtered_indices" | "filtered_gt"
                | "filtered_ground_truth" => Some(Self::PrefilteredNeighborIndices),
            "filtered_neighbor_distances" | "filtered_distances" | "filtered_neighbors" => {
                Some(Self::PrefilteredNeighborDistances)
            }
            _ => None,
        }
    }

    /// Returns `true` if `name` is a legacy filtered-knn alias that loaders
    /// should treat as F while emitting a one-line migration note.
    pub fn is_legacy_filtered_alias(name: &str) -> bool {
        matches!(name, "filtered_neighbor_indices" | "filtered_neighbor_distances")
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
/// - mvec: 2 bytes (IEEE 754 half-precision f16)
/// - svec: 2 bytes (i16)
fn xvec_for_element_size(element_size: usize) -> VecFormat {
    match element_size {
        2 => VecFormat::Mvec,
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
        assert_eq!(Facet::BaseVectors.preferred_format(2), VecFormat::Mvec);
        assert_eq!(Facet::BaseVectors.preferred_format(4), VecFormat::Fvec);
        assert_eq!(Facet::BaseVectors.preferred_format(8), VecFormat::Dvec);
        // Index facets always ivec regardless of element size
        assert_eq!(Facet::NeighborIndices.preferred_format(2), VecFormat::Ivec);
        assert_eq!(Facet::NeighborIndices.preferred_format(8), VecFormat::Ivec);
    }

    #[test]
    fn test_preferred_order_is_complete() {
        assert_eq!(Facet::PREFERRED_ORDER.len(), 12);
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

    /// Pre- and post-filter facets are distinct. Aliases route to the
    /// canonical variants; legacy `filtered_*` resolves to F (post-filter)
    /// for backwards compat with pre-E/F-split datasets.
    #[test]
    fn test_from_alias_prefiltered() {
        assert_eq!(Facet::from_alias("prefiltered_indices"), Some(Facet::PrefilteredNeighborIndices));
        assert_eq!(Facet::from_alias("prefilter_indices"), Some(Facet::PrefilteredNeighborIndices));
        assert_eq!(Facet::from_alias("prefiltered_distances"), Some(Facet::PrefilteredNeighborDistances));
        assert_eq!(Facet::from_alias("prefiltered_gt"), Some(Facet::PrefilteredNeighborIndices));
    }

    #[test]
    fn test_from_alias_postfiltered() {
        assert_eq!(Facet::from_alias("postfiltered_indices"), Some(Facet::PostfilteredNeighborIndices));
        assert_eq!(Facet::from_alias("postfilter_indices"), Some(Facet::PostfilteredNeighborIndices));
        assert_eq!(Facet::from_alias("postfiltered_distances"), Some(Facet::PostfilteredNeighborDistances));
        assert_eq!(Facet::from_alias("postfiltered_gt"), Some(Facet::PostfilteredNeighborIndices));
    }

    /// Legacy `filtered_*` keys MUST resolve to **pre-filter** variants
    /// (facet code F) — matching the actual shape produced by the legacy
    /// `compute filtered-knn` command. Regression pin per
    /// docs/design/prefilter-postfilter-facets.md §3.1.
    #[test]
    fn test_legacy_filtered_aliases_resolve_to_prefilter() {
        assert_eq!(
            Facet::from_alias("filtered_neighbor_indices"),
            Some(Facet::PrefilteredNeighborIndices),
        );
        assert_eq!(
            Facet::from_alias("filtered_neighbor_distances"),
            Some(Facet::PrefilteredNeighborDistances),
        );
        assert_eq!(Facet::from_alias("filtered_indices"), Some(Facet::PrefilteredNeighborIndices));
        assert_eq!(Facet::from_alias("filtered_gt"), Some(Facet::PrefilteredNeighborIndices));
        assert_eq!(Facet::from_alias("filtered_distances"), Some(Facet::PrefilteredNeighborDistances));
        assert_eq!(Facet::from_alias("filtered_neighbors"), Some(Facet::PrefilteredNeighborDistances));
        assert!(Facet::is_legacy_filtered_alias("filtered_neighbor_indices"));
        assert!(Facet::is_legacy_filtered_alias("filtered_neighbor_distances"));
        assert!(!Facet::is_legacy_filtered_alias("postfiltered_neighbor_indices"));
        assert!(!Facet::is_legacy_filtered_alias("prefiltered_neighbor_indices"));
    }

    /// Facet code mapping per docs/design/prefilter-postfilter-facets.md:
    /// pre-filter → F (the legacy filtered-knn shape), post-filter → E
    /// (new G ∩ R artifact). Regression-pin so a future refactor can't
    /// silently swap them — that swap would mislabel every existing
    /// dataset.
    #[test]
    fn test_facet_codes() {
        assert_eq!(Facet::BaseVectors.code(), Some('B'));
        assert_eq!(Facet::QueryVectors.code(), Some('Q'));
        assert_eq!(Facet::NeighborIndices.code(), Some('G'));
        assert_eq!(Facet::NeighborDistances.code(), Some('D'));
        assert_eq!(Facet::MetadataContent.code(), Some('M'));
        assert_eq!(Facet::MetadataPredicates.code(), Some('P'));
        assert_eq!(Facet::MetadataResults.code(), Some('R'));
        assert_eq!(Facet::MetadataLayout.code(), None);
        assert_eq!(Facet::PrefilteredNeighborIndices.code(), Some('F'));
        assert_eq!(Facet::PrefilteredNeighborDistances.code(), Some('F'));
        assert_eq!(Facet::PostfilteredNeighborIndices.code(), Some('E'));
        assert_eq!(Facet::PostfilteredNeighborDistances.code(), Some('E'));
    }

    #[test]
    fn test_from_alias_unknown() {
        assert_eq!(Facet::from_alias("unknown"), None);
        assert_eq!(Facet::from_alias("base_vectors"), None); // canonical, not alias
    }
}
