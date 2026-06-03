// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Standard facet kinds for vector datasets.
//!
//! [`StandardFacet`] enumerates the well-known facet types that define a
//! dataset's layout: `base_vectors`, `query_vectors`, `neighbor_indices`,
//! `neighbor_distances`, metadata content/predicates/results/layout, and
//! the two predicated KNN ground-truth variants — pre-filter (`E`) and
//! post-filter (`F`).
//!
//! Each facet has a canonical key name (used in `dataset.yaml` profile
//! definitions) and a set of shorthand aliases for convenience. This module
//! provides name resolution without any dependency on file formats or CLI
//! frameworks.
//!
//! ## E vs F (see `docs/design/prefilter-postfilter-facets.md`)
//!
//! - **F** — [`PrefilteredNeighborIndices`] / [`PrefilteredNeighborDistances`]:
//!   pre-filter ground truth, ACORN's `G_K`. The top-K of `X_p` by distance —
//!   perfect recall, full K when `|X_p| ≥ K`. This is the **legacy filtered-
//!   knn shape**, retained under the F code so existing datasets keep their
//!   meaning.
//! - **E** — [`PostfilteredNeighborIndices`] / [`PostfilteredNeighborDistances`]:
//!   post-filter ground truth, `G ∩ R`. The unfiltered top-K intersected
//!   with the predicate-matching set. Sparse possible. New facet introduced
//!   alongside the E/F split.
//!
//! The legacy keys `filtered_neighbor_indices` / `filtered_neighbor_distances`
//! resolve to the F (pre-filter) facet — matching the actual shape of files
//! produced by the legacy `compute filtered-knn` command.

use std::fmt;

/// Canonical facet kinds for predicated vector datasets.
///
/// Each variant identifies a specific role within a test dataset. The
/// canonical key name is used in `dataset.yaml` profile definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StandardFacet {
    /// Base vectors (the corpus to search) — facet code `B`
    BaseVectors,
    /// Query vectors (the queries to run) — facet code `Q`
    QueryVectors,
    /// Unfiltered ground-truth neighbor indices — facet code `G`
    NeighborIndices,
    /// Unfiltered ground-truth neighbor distances — facet code `D`
    NeighborDistances,
    /// Metadata content records (MNode) — facet code `M`
    MetadataContent,
    /// Metadata predicate trees (PNode) — facet code `P`
    MetadataPredicates,
    /// Metadata filter result bitmaps — facet code `R`
    MetadataResults,
    /// Metadata field layout schema
    MetadataLayout,
    /// Pre-filter KNN ground-truth indices — facet code `F`.
    /// Top-K of `X_p` (the predicate-passing base vectors). Full K
    /// when `|X_p| ≥ K`; perfect recall. This is the legacy filtered-
    /// knn shape; the legacy YAML key `filtered_neighbor_indices`
    /// resolves here.
    PrefilteredNeighborIndices,
    /// Pre-filter KNN ground-truth distances — facet code `F`.
    PrefilteredNeighborDistances,
    /// Post-filter KNN ground-truth indices — facet code `E`.
    /// `G ∩ R` — the unfiltered top-K intersected with the
    /// predicate-passing set. Sparse possible.
    PostfilteredNeighborIndices,
    /// Post-filter KNN ground-truth distances — facet code `E`.
    PostfilteredNeighborDistances,
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
        Self::PrefilteredNeighborIndices,
        Self::PrefilteredNeighborDistances,
        Self::PostfilteredNeighborIndices,
        Self::PostfilteredNeighborDistances,
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
            Self::PrefilteredNeighborIndices => "prefiltered_neighbor_indices",
            Self::PrefilteredNeighborDistances => "prefiltered_neighbor_distances",
            Self::PostfilteredNeighborIndices => "postfiltered_neighbor_indices",
            Self::PostfilteredNeighborDistances => "postfiltered_neighbor_distances",
        }
    }

    /// Single-character facet code used by the partition sub-facet scoping
    /// strings (e.g. `"BQG"`). Layout has no code — it is metadata-about-
    /// metadata, not a content facet.
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
    ///
    /// The legacy keys `filtered_neighbor_indices` and
    /// `filtered_neighbor_distances` resolve to the **F** (pre-filter) facet
    /// — matching the actual shape of files produced by the legacy
    /// `compute filtered-knn` command. Loaders that hit these aliases
    /// should advise users to migrate to the canonical `prefiltered_*`
    /// keys.
    pub fn from_alias(name: &str) -> Option<Self> {
        match name {
            "base" | "train" => Some(Self::BaseVectors),
            "query" | "queries" | "test" => Some(Self::QueryVectors),
            "indices" | "neighbors" | "ground_truth" | "gt" => Some(Self::NeighborIndices),
            "distances" => Some(Self::NeighborDistances),
            "content" | "meta_content" | "meta_base" => Some(Self::MetadataContent),
            "meta_predicates" => Some(Self::MetadataPredicates),
            // `metadata_indices` is the legacy on-disk/key name for the
            // predicate-match index (the R facet); accept it as an alias so
            // extant datasets resolve under canonical `metadata_results`.
            "meta_results" | "predicate_results" | "metadata_indices" => Some(Self::MetadataResults),
            "layout" | "meta_layout" => Some(Self::MetadataLayout),
            "prefiltered_indices" | "prefiltered_gt" | "prefiltered_ground_truth"
                | "prefilter_indices" => Some(Self::PrefilteredNeighborIndices),
            "prefiltered_distances" | "prefilter_distances" | "prefiltered_neighbors" => {
                Some(Self::PrefilteredNeighborDistances)
            }
            // Canonical post-filter aliases.
            "postfiltered_indices" | "postfiltered_gt" | "postfiltered_ground_truth"
                | "postfilter_indices" => Some(Self::PostfilteredNeighborIndices),
            "postfiltered_distances" | "postfilter_distances" | "postfiltered_neighbors" => {
                Some(Self::PostfilteredNeighborDistances)
            }
            // Legacy aliases — pre-E/F-split datasets used these for
            // pre-filter ground truth (now facet code F). They resolve
            // to the prefiltered variants so the on-disk shape matches
            // the new typing without forcing regeneration.
            "filtered_neighbor_indices" | "filtered_indices" | "filtered_gt"
                | "filtered_ground_truth" => Some(Self::PrefilteredNeighborIndices),
            "filtered_neighbor_distances" | "filtered_distances" | "filtered_neighbors" => {
                Some(Self::PrefilteredNeighborDistances)
            }
            _ => None,
        }
    }

    /// Returns `true` if `name` is a legacy alias that should trigger a
    /// migration note ("use `postfiltered_*` going forward"). Used by
    /// loaders to surface a deprecation hint without refusing the file.
    pub fn is_legacy_filtered_alias(name: &str) -> bool {
        matches!(name, "filtered_neighbor_indices" | "filtered_neighbor_distances")
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
        assert_eq!(StandardFacet::PREFERRED_ORDER.len(), 12);
    }

    /// Canonical keys for E and F are distinct from the legacy `filtered_*`
    /// names. The legacy names are resolvable as aliases (see below).
    #[test]
    fn test_e_and_f_canonical_keys() {
        assert_eq!(StandardFacet::PrefilteredNeighborIndices.key(), "prefiltered_neighbor_indices");
        assert_eq!(StandardFacet::PrefilteredNeighborDistances.key(), "prefiltered_neighbor_distances");
        assert_eq!(StandardFacet::PostfilteredNeighborIndices.key(), "postfiltered_neighbor_indices");
        assert_eq!(StandardFacet::PostfilteredNeighborDistances.key(), "postfiltered_neighbor_distances");
    }

    /// Legacy `filtered_*` keys MUST resolve to the **pre-filter** (F)
    /// facet — matching the actual shape of files produced by the legacy
    /// `compute filtered-knn` command (full K, ACORN G_K). Regression-pin
    /// this so a future refactor can't silently reassign them to the
    /// post-filter variant (which is a different artifact shape).
    #[test]
    fn test_legacy_filtered_alias_resolves_to_prefilter() {
        assert_eq!(
            StandardFacet::from_alias("filtered_neighbor_indices"),
            Some(StandardFacet::PrefilteredNeighborIndices),
        );
        assert_eq!(
            StandardFacet::from_alias("filtered_neighbor_distances"),
            Some(StandardFacet::PrefilteredNeighborDistances),
        );
        // The legacy-alias predicate is what loaders use to emit the
        // migration note — keep it in sync with the two keys above.
        assert!(StandardFacet::is_legacy_filtered_alias("filtered_neighbor_indices"));
        assert!(StandardFacet::is_legacy_filtered_alias("filtered_neighbor_distances"));
        assert!(!StandardFacet::is_legacy_filtered_alias("postfiltered_neighbor_indices"));
        assert!(!StandardFacet::is_legacy_filtered_alias("prefiltered_neighbor_indices"));
    }

    /// Prefiltered aliases route to E; postfiltered aliases route to F.
    #[test]
    fn test_e_and_f_aliases() {
        assert_eq!(StandardFacet::from_alias("prefiltered_gt"), Some(StandardFacet::PrefilteredNeighborIndices));
        assert_eq!(StandardFacet::from_alias("prefilter_indices"), Some(StandardFacet::PrefilteredNeighborIndices));
        assert_eq!(StandardFacet::from_alias("postfiltered_gt"), Some(StandardFacet::PostfilteredNeighborIndices));
        assert_eq!(StandardFacet::from_alias("postfilter_indices"), Some(StandardFacet::PostfilteredNeighborIndices));
    }

    /// Facet codes per docs/design/prefilter-postfilter-facets.md.
    /// Pre-filter → **F** (the legacy filtered-knn shape; ACORN G_K).
    /// Post-filter → **E** (G ∩ R, new sparse artifact). Regression-pin
    /// this so a future refactor can't silently swap them — swapping
    /// would mislabel every existing published dataset.
    #[test]
    fn test_facet_codes() {
        assert_eq!(StandardFacet::BaseVectors.code(), Some('B'));
        assert_eq!(StandardFacet::QueryVectors.code(), Some('Q'));
        assert_eq!(StandardFacet::NeighborIndices.code(), Some('G'));
        assert_eq!(StandardFacet::NeighborDistances.code(), Some('D'));
        assert_eq!(StandardFacet::MetadataContent.code(), Some('M'));
        assert_eq!(StandardFacet::MetadataPredicates.code(), Some('P'));
        assert_eq!(StandardFacet::MetadataResults.code(), Some('R'));
        assert_eq!(StandardFacet::MetadataLayout.code(), None);
        assert_eq!(StandardFacet::PrefilteredNeighborIndices.code(), Some('F'));
        assert_eq!(StandardFacet::PrefilteredNeighborDistances.code(), Some('F'));
        assert_eq!(StandardFacet::PostfilteredNeighborIndices.code(), Some('E'));
        assert_eq!(StandardFacet::PostfilteredNeighborDistances.code(), Some('E'));
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
