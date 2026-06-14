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
/// The value-shape a facet's bytes take in a file/resource. This is the
/// coarse format taxonomy; specific element widths live in the extension
/// list each format owns. It is the authority for questions like
/// "can this facet be stored as an integer xvec?".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FacetFormat {
    /// Float xvec (`fvecs`/`fvec`/`dvecs`/…) — vectors and distances.
    FloatXvec,
    /// Fixed-width integer xvec (`ivecs`/`ivec`/`i32vecs`/…) — neighbor IDs.
    IntegerXvec,
    /// Variable-length integer xvec (`ivvecs`/`ivvec`/…) — the per-query
    /// predicate-match index (R).
    IntegerVarXvec,
    /// Raw packed scalars (`u8`/`i32`/…) — flat metadata columns.
    ScalarPacked,
    /// A slabtastic slab (`slab`) — MNode/PNode records, layout namespaces.
    Slab,
}

impl FacetFormat {
    /// Every recognized file extension for this format, plural (canonical)
    /// forms first.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            Self::FloatXvec => &[
                "fvecs", "dvecs", "mvecs", "fvec", "dvec", "mvec",
                "f32vecs", "f64vecs", "f16vecs", "f32vec", "f64vec", "f16vec",
            ],
            Self::IntegerXvec => &[
                "ivecs", "i32vecs", "u32vecs", "ivec", "i32vec", "u32vec",
                "i8vecs", "u8vecs", "bvecs", "i16vecs", "u16vecs", "svecs",
                "i8vec", "u8vec", "bvec", "i16vec", "u16vec", "svec",
                "i64vecs", "u64vecs", "i64vec", "u64vec",
            ],
            Self::IntegerVarXvec => &[
                "ivvecs", "i32vvecs", "u32vvecs", "ivvec", "i32vvec", "u32vvec",
            ],
            Self::ScalarPacked => &["u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64"],
            Self::Slab => &["slab"],
        }
    }

    /// The format an extension belongs to, if recognized.
    pub fn from_extension(ext: &str) -> Option<Self> {
        [Self::FloatXvec, Self::IntegerXvec, Self::IntegerVarXvec, Self::ScalarPacked, Self::Slab]
            .into_iter()
            .find(|f| f.extensions().contains(&ext))
    }
}

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

    /// Inverse of [`Self::code`]: resolve a capital facet code letter
    /// to its facet. `F` and `E` each denote a filtered *family*
    /// (indices + distances share the code); they resolve to the
    /// indices member, the family's primary facet. `None` for letters
    /// that aren't facet codes.
    ///
    /// This is user-input vocabulary (filters, pickers). The loader
    /// path canonicalizes via [`resolve_standard_key`], which
    /// deliberately does NOT consume single letters — a custom facet
    /// literally keyed `"B"` in a dataset.yaml is left alone.
    pub fn from_code(code: char) -> Option<Self> {
        match code {
            'B' => Some(Self::BaseVectors),
            'Q' => Some(Self::QueryVectors),
            'G' => Some(Self::NeighborIndices),
            'D' => Some(Self::NeighborDistances),
            'M' => Some(Self::MetadataContent),
            'P' => Some(Self::MetadataPredicates),
            'R' => Some(Self::MetadataResults),
            'F' => Some(Self::PrefilteredNeighborIndices),
            'E' => Some(Self::PostfilteredNeighborIndices),
            _ => None,
        }
    }

    /// The value-shape(s) this facet's bytes may legitimately take. The
    /// authority for "can facet X be stored as format Y?".
    pub fn formats(self) -> &'static [FacetFormat] {
        use FacetFormat::*;
        match self {
            Self::BaseVectors | Self::QueryVectors | Self::NeighborDistances
            | Self::PrefilteredNeighborDistances | Self::PostfilteredNeighborDistances => &[FloatXvec],
            Self::NeighborIndices
            | Self::PrefilteredNeighborIndices | Self::PostfilteredNeighborIndices => &[IntegerXvec],
            // Metadata content may be a slab of MNode records, raw packed
            // scalars, or an integer xvec (see `gen metadata`'s modes).
            Self::MetadataContent => &[Slab, ScalarPacked, IntegerXvec],
            // Predicate trees are a PNode slab; synthesis modes also emit
            // raw packed / integer-xvec encodings.
            Self::MetadataPredicates => &[Slab, ScalarPacked, IntegerXvec],
            // The predicate-match index (R): a variable-length integer xvec
            // by default, a fixed integer xvec or slab in other modes.
            Self::MetadataResults => &[IntegerVarXvec, IntegerXvec, Slab],
            // The schema: a slab (its own file, or a `#layout` namespace
            // inside a metadata slab).
            Self::MetadataLayout => &[Slab],
        }
    }

    /// On-disk basename(s) the facet's primary file may use — canonical
    /// first, then legacy names retained for extant datasets. (A facet is
    /// not 1:1 with a single filename.)
    pub fn basenames(self) -> &'static [&'static str] {
        match self {
            Self::BaseVectors => &["base_vectors"],
            Self::QueryVectors => &["query_vectors"],
            Self::NeighborIndices => &["neighbor_indices"],
            Self::NeighborDistances => &["neighbor_distances"],
            Self::MetadataContent => &["metadata_content"],
            Self::MetadataPredicates => &["metadata_predicates"],
            // Historically the predicate-match index was `metadata_indices`.
            Self::MetadataResults => &["metadata_results", "metadata_indices"],
            Self::MetadataLayout => &["metadata_layout"],
            // Legacy `filtered_*` names retained for extant datasets.
            Self::PrefilteredNeighborIndices => &["prefiltered_neighbor_indices", "filtered_neighbor_indices"],
            Self::PrefilteredNeighborDistances => &["prefiltered_neighbor_distances", "filtered_neighbor_distances"],
            Self::PostfilteredNeighborIndices => &["postfiltered_neighbor_indices"],
            Self::PostfilteredNeighborDistances => &["postfiltered_neighbor_distances"],
        }
    }

    /// The slab namespace(s) the facet's data may live in. Most facets use
    /// the default namespace (`""`); the layout schema may instead be a
    /// `layout` namespace co-located in a metadata slab.
    pub fn namespaces(self) -> &'static [&'static str] {
        match self {
            Self::MetadataLayout => &["layout", ""],
            _ => &[""],
        }
    }

    /// Does this facet accept the given value-shape?
    pub fn accepts_format(self, format: FacetFormat) -> bool {
        self.formats().contains(&format)
    }

    /// Every file extension valid for this facet (the union over its
    /// formats). Answers "can I store this facet in a `*.ivecs` file?".
    pub fn accepts_extension(self, ext: &str) -> bool {
        self.formats().iter().any(|f| f.extensions().contains(&ext))
    }

    /// Classify a file/resource name to the facet (and format) it belongs
    /// to, or `None` if it matches no facet. Strips a directory prefix, a
    /// `#namespace` suffix, and an `IDXFOR__…` sidecar wrapper.
    pub fn classify(name: &str) -> Option<(StandardFacet, FacetFormat)> {
        let name = name.rsplit(['/', '\\']).next().unwrap_or(name);
        let name = name.split('#').next().unwrap_or(name);
        // An `IDXFOR__<data>.<idx-ext>` sidecar belongs to the same facet as
        // its data file `<data>`; drop the wrapper and the index extension.
        let name = match name.strip_prefix("IDXFOR__") {
            Some(rest) => rest.rsplit_once('.').map(|(base, _)| base).unwrap_or(rest),
            None => name,
        };
        let (basename, ext) = name.rsplit_once('.')?;
        let fmt = FacetFormat::from_extension(ext)?;
        Self::PREFERRED_ORDER
            .iter()
            .copied()
            .find(|f| f.basenames().contains(&basename) && f.accepts_format(fmt))
            .map(|f| (f, fmt))
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

    /// The three questions the standardized spec must answer cheaply.
    #[test]
    fn spec_answers_facet_format_validity() {
        // "Can I store my metadata (content) in an integer xvec file?" — yes.
        assert!(StandardFacet::MetadataContent.accepts_format(FacetFormat::IntegerXvec));
        assert!(StandardFacet::MetadataContent.accepts_extension("ivecs"));
        // Base vectors are float xvec only — not integer.
        assert!(!StandardFacet::BaseVectors.accepts_format(FacetFormat::IntegerXvec));
        assert!(StandardFacet::BaseVectors.accepts_extension("fvecs"));
        // The R index accepts the variable-length integer xvec form.
        assert!(StandardFacet::MetadataResults.accepts_format(FacetFormat::IntegerVarXvec));
        assert!(StandardFacet::MetadataResults.accepts_extension("ivvecs"));
    }

    #[test]
    fn spec_classifies_resources_to_facets() {
        // "Does this file belong to the metadata_content facet?"
        assert_eq!(
            StandardFacet::classify("profiles/base/metadata_content.slab"),
            Some((StandardFacet::MetadataContent, FacetFormat::Slab)),
        );
        // The R index, under canonical and legacy names + its IDXFOR sidecar.
        assert_eq!(
            StandardFacet::classify("profiles/default/metadata_results.ivvecs").map(|(f, _)| f),
            Some(StandardFacet::MetadataResults),
        );
        assert_eq!(
            StandardFacet::classify("profiles/default/metadata_indices.ivvecs").map(|(f, _)| f),
            Some(StandardFacet::MetadataResults),
        );
        assert_eq!(
            StandardFacet::classify("IDXFOR__metadata_results.ivvecs.i32").map(|(f, _)| f),
            Some(StandardFacet::MetadataResults),
        );
        // A `#namespace` locator classifies by its file part.
        assert_eq!(
            StandardFacet::classify("metadata_content.slab#layout").map(|(f, _)| f),
            Some(StandardFacet::MetadataContent),
        );
        assert_eq!(StandardFacet::classify("random_unknown.xyz"), None);
    }

    #[test]
    fn spec_enumerates_facet_resources() {
        // "What files/resources/namespaces are associated with facet R?"
        let r = StandardFacet::MetadataResults;
        assert_eq!(r.basenames(), &["metadata_results", "metadata_indices"]);
        assert!(r.accepts_extension("ivvecs") && r.accepts_extension("slab"));
        // The layout schema may be a co-located `layout` namespace.
        assert!(StandardFacet::MetadataLayout.namespaces().contains(&"layout"));
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
