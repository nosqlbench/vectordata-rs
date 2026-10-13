//! Library API. The reader's view of a `dataset.yaml`.
//!
//! [`DatasetConfig`], [`ProfileConfig`] and [`FacetConfig`] are what
//! [`TestDataGroup`](crate::TestDataGroup) and the dataset views read:
//! each profile's standard facets, resolved. They are a **projection** of
//! the one description type, [`crate::dataset::DatasetConfig`], which
//! parses the file, applies the version gates and the inheritance rule,
//! and is what the pipeline reads; the reader never parses a file on its
//! own. Deserializing [`DatasetConfig`] runs that parse and projects it.
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
    /// The **minimum reader version** this dataset requires.
    ///
    /// Absent in the YAML means `1` — every dataset in circulation omits
    /// it, and they are all version 1 (V-2). Not a timestamp and not the
    /// writer's capability: the lowest version that describes what was
    /// actually written (V-4, V-5).
    ///
    /// See `docs/design/srd-dataset-format-version.md`.
    pub format_version: u32,
    /// Arbitrary attributes describing the dataset (e.g., distance metric, dimension).
    pub attributes: HashMap<String, serde_yaml::Value>,
    /// The profile tag schema (PS-19): the naming tags in order, each
    /// with a default or `~`. Empty when the dataset declares none.
    #[serde(default, skip_serializing_if = "indexmap::IndexMap::is_empty")]
    pub profile_tags: indexmap::IndexMap<String, serde_yaml::Value>,
    /// Named profiles defining different views or subsets of the dataset.
    pub profiles: HashMap<String, ProfileConfig>,
}

impl DatasetConfig {
    /// The lowest format version that can express this dataset.
    ///
    /// A fold over the tree, not a field: the version a dataset
    /// *requires* is a property of what it says, and deriving it is what
    /// stops a writer's stamp drifting from its content (V-19).
    pub fn min_format_version(&self) -> u32 {
        let facets = self
            .profiles
            .values()
            .flat_map(|p| p.facets())
            .map(|(_, f)| f.min_format_version())
            .max()
            .unwrap_or(FORMAT_VERSION_BASE);
        // A named parent or a tag schema is version 3 (V-7): a reader
        // that predates `inherits:` would fall back to `default` and
        // serve the wrong facets without a word.
        let tagged = !self.profile_tags.is_empty()
            || self.profiles.iter().any(|(name, p)| {
                name != "default"
                    && p.inherits.as_deref().is_some_and(|parent| parent != "default")
            });
        facets.max(if tagged { FORMAT_VERSION_TAGGED } else { FORMAT_VERSION_BASE })
    }

    /// Whether every facet is expressible in **v1**.
    ///
    /// `true` *is* the proof that a v1 reader can read this — there is
    /// no separate compatibility check to keep in step (V-20).
    pub fn is_v1(&self) -> bool {
        self.min_format_version() <= FORMAT_VERSION_BASE
    }
}

/// The `dataset.yaml` format version a dataset is assumed to be when it
/// declares none (V-2).
pub const FORMAT_VERSION_BASE: u32 = 1;

/// The version a sharded facet declaration requires (V-7).
pub const FORMAT_VERSION_SHARDED: u32 = 2;

/// The version a named parent or a profile tag schema requires (V-7):
/// profiles that state their parents and carry their tags.
pub const FORMAT_VERSION_TAGGED: u32 = 3;

/// The highest `dataset.yaml` format version this build can read.
///
/// A dataset declaring more than this is refused at load, naming both
/// numbers — the diagnosis the field exists to provide (V-9).
pub const FORMAT_VERSION_SUPPORTED: u32 = FORMAT_VERSION_TAGGED;

/// The version an absent field stands for (V-2).
pub fn base_format_version() -> u32 {
    FORMAT_VERSION_BASE
}

/// Whether a version is the base one, and so need not be written out.
///
/// A writer emits the lowest version that describes what it wrote, and
/// for version 1 that is nothing at all — which is what keeps an
/// unsharded output readable by every build that ever existed (V-5).
pub fn is_base_format_version(v: &u32) -> bool {
    *v == FORMAT_VERSION_BASE
}

/// Refuse a dataset this build cannot read, naming both numbers (V-9).
///
/// **Shared by both loaders rather than mirrored** (V-12). A dataset
/// accepted by one route and refused by the other would make the
/// transport decide whether it is readable, which is the same fault
/// duplicated shard realization would be (SH-90). One function is what
/// makes "both read the field identically" a property of the code
/// rather than a thing to keep true by hand.
///
/// Returns the effective version: an absent field means 1, which is
/// every dataset in circulation and not a distinct unversioned state
/// (V-2).
pub fn check_supported_version(stated: Option<u32>) -> Result<u32, String> {
    let version = stated.unwrap_or(FORMAT_VERSION_BASE);
    if version > FORMAT_VERSION_SUPPORTED {
        return Err(format!(
            "dataset requires format_version {version}; this build \
             supports up to {FORMAT_VERSION_SUPPORTED}. Upgrade vectordata to read it."
        ));
    }
    Ok(version)
}

/// Refuse a declaration that understates what it holds (V-14).
///
/// A *stated* version lower than the content requires is a declaration
/// contradicting itself — the same class of fault as a record count
/// that disagrees with its shards (SH-8).
///
/// An **absent** field is not a claim. It means 1 for the gate above
/// (V-2), but a dataset that never declared a version has not
/// understated one, and a reader new enough to notice the omission is
/// new enough to read the data (V-22). Under-annotation is a note from
/// `veks check`, not a load failure — refusing it here would reject
/// every hand-written sharded dataset over a field that helps no reader
/// which can already read it.
///
/// A version *higher* than the content requires is merely generous, and
/// is accepted (V-23).
pub fn check_stated_against_content(stated: Option<u32>, required: u32) -> Result<(), String> {
    match stated {
        Some(stated) if required > stated => Err(format!(
            "dataset declares format_version {stated} but its content \
             requires {required} — a declaration cannot understate what it holds"
        )),
        // Absent means 1, and the dataset is held to it (V-24): a file
        // that never said what it is is read as the least it could be,
        // and refused when its content needs more.
        None if required > FORMAT_VERSION_BASE => Err(format!(
            "dataset declares no format_version and carries content that \
             requires {required}; declare format_version: {required}"
        )),
        _ => Ok(()),
    }
}

/// The reader's view of a `dataset.yaml` is a **projection** of the one
/// description type, [`crate::dataset::DatasetConfig`]: the text is
/// parsed once, by that type, with its version gates, declaration checks
/// and the one inheritance rule
/// ([`DSProfileGroup`](crate::dataset::DSProfileGroup)), and this view is
/// computed from the result. The pipeline and a reader therefore see the
/// same profiles with the same facets.
///
/// Sized profiles written compactly (`strata:`, `sized:`) are expanded
/// where the file alone says enough: from its own `variables:` block.
/// Specs that wait on a pipeline's `variables.yaml` stay unexpanded, as
/// a reader of a published dataset has no such file.
impl<'de> Deserialize<'de> for DatasetConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let description = crate::dataset::DatasetConfig::deserialize(deserializer)?;
        Ok(DatasetConfig::opened(description))
    }
}

impl DatasetConfig {
    /// The reader's view of a description it opens: sized profiles
    /// written compactly are expanded from the description's own
    /// `variables:` block, then the description is projected. Every
    /// open goes through this, from a file, a URL, a catalog entry or a
    /// description built in code.
    pub(crate) fn opened(mut description: crate::dataset::DatasetConfig) -> Self {
        description.unify_strata();
        if description.profiles.has_deferred() {
            let vars = description.variables.clone();
            description.expand_sized_profiles(&vars);
        }
        DatasetConfig::from_description(&description)
    }

    /// The reader's view of a dataset description.
    pub fn from_description(description: &crate::dataset::DatasetConfig) -> Self {
        Self::from_parts(
            description.format_version,
            description.attributes.as_ref(),
            &description.profile_tags,
            &description.profiles,
        )
    }

    /// The reader's view of a description given by its parts, as a
    /// catalog entry's layout carries them.
    pub fn from_parts(
        format_version: u32,
        attributes: Option<&crate::dataset::DatasetAttributes>,
        profile_tags: &indexmap::IndexMap<String, serde_yaml::Value>,
        profiles: &crate::dataset::DSProfileGroup,
    ) -> Self {
        let attributes = attributes
            .and_then(|a| serde_yaml::to_value(a).ok())
            .and_then(|v| match v {
                serde_yaml::Value::Mapping(m) => Some(
                    m.into_iter()
                        // An attribute the file does not state is absent,
                        // not a null the reader would find by name.
                        .filter(|(_, v)| !v.is_null())
                        .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v)))
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        DatasetConfig {
            format_version,
            attributes,
            profile_tags: profile_tags.clone(),
            profiles: profiles
                .profiles
                .iter()
                .map(|(name, p)| (name.clone(), project_profile(p)))
                .collect(),
        }
    }
}

/// A profile as the reader holds it: the standard facets, by key. A
/// custom facet key has no slot here and is reached through the
/// description.
fn project_profile(p: &crate::dataset::DSProfile) -> ProfileConfig {
    let facet = |key: &str| p.views.get(key).map(project_view);
    ProfileConfig {
        base_count: p.base_count,
        maxk: p.maxk,
        partition: p.partition,
        attributes: p.attributes.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        inherits: p.inherits.clone(),
        base_vectors: facet("base_vectors"),
        base_content: facet("base_content"),
        query_vectors: facet("query_vectors"),
        query_terms: facet("query_terms"),
        query_filters: facet("query_filters"),
        neighbor_indices: facet("neighbor_indices"),
        neighbor_distances: facet("neighbor_distances"),
        prefiltered_neighbor_indices: facet("prefiltered_neighbor_indices"),
        prefiltered_neighbor_distances: facet("prefiltered_neighbor_distances"),
        postfiltered_neighbor_indices: facet("postfiltered_neighbor_indices"),
        postfiltered_neighbor_distances: facet("postfiltered_neighbor_distances"),
        metadata_content: facet("metadata_content"),
        metadata_predicates: facet("metadata_predicates"),
        predicate_results: facet("metadata_results"),
        metadata_layout: facet("metadata_layout"),
    }
}

/// A facet as the reader holds it. A single file keeps its source as
/// written (path, namespace and window, through `DSSource`'s `Display`,
/// the inverse of the source-string parser); a series keeps its pattern
/// or list, with the facet window in its own field (SH-67, SH-102).
fn project_view(v: &crate::dataset::DSView) -> FacetConfig {
    let window = v.window.as_ref().map(|w| w.to_string());
    if v.is_series() {
        if v.shard_count.is_some() {
            // A window written on the pattern is the facet's window.
            let (source, window) = if window.is_none() && !v.source.window.is_empty() {
                let bare = crate::dataset::source::DSSource {
                    window: Default::default(),
                    ..v.source.clone()
                };
                (bare.to_string(), Some(v.source.window.to_string()))
            } else {
                (v.source.to_string(), window)
            };
            return FacetConfig::Sharded(ShardedFacet::Uniform {
                source,
                shard_stride: v.shard_stride.unwrap_or_default(),
                shard_count: v.shard_count.unwrap_or_default(),
                record_count: v.record_count.unwrap_or_default(),
                window,
            });
        }
        return FacetConfig::Sharded(ShardedFacet::Explicit {
            source: v.declaration_sources(),
            record_count: v.record_count.unwrap_or_default(),
            window,
        });
    }
    match window {
        None => FacetConfig::Simple(v.source.to_string()),
        Some(window) => FacetConfig::Detailed { source: v.source.to_string(), window: Some(window) },
    }
}

/// A profile name that YAML may have read as a number — a rung such as
/// `100` — comes back as the name it spells.
fn name_or_number<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Option<serde_yaml::Value> = Option::deserialize(deserializer)?;
    match v {
        None | Some(serde_yaml::Value::Null) => Ok(None),
        Some(serde_yaml::Value::String(s)) => Ok(Some(s)),
        Some(serde_yaml::Value::Number(n)) => Ok(Some(n.to_string())),
        Some(other) => Err(serde::de::Error::custom(format!(
            "inherits: expected a profile name, found {}",
            serde_yaml::to_string(&other).unwrap_or_default().trim()
        ))),
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

    /// Open key/value map describing what this profile *is* — the
    /// parameter values that distinguish it from its siblings (P-4).
    ///
    /// `base_count` and `maxk` are fields because the reader acts on
    /// them. A selectivity is not something the reader acts on; it is
    /// something a consumer selects by. Without somewhere to put it, a
    /// family of selectivities is machine-readable only by parsing
    /// member names, and a name is an identifier, not a measurement.
    ///
    /// Conventional keys are `selectivity` (a fraction in `0.0..=1.0`),
    /// `predicate_count`, and `k` — conventions, not requirements
    /// (P-5). Unknown keys are preserved and reported, never rejected.
    ///
    /// Values record **what a run realized**, never what it targeted
    /// (P-6): a sweep aiming at 1% that achieves 1.2% records `0.012`.
    /// Absent means undescribed, which is not `selectivity: 0.0` (P-7)
    /// — hence a map that can be empty rather than fields that can be
    /// zero.
    ///
    /// See `docs/design/srd-profile-parameterization.md`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, serde_yaml::Value>,

    /// The profile this one inherits unstated facets from.
    ///
    /// Absent means `default`, which is every dataset written before
    /// parameterized profiles and the reason the field is optional
    /// rather than required (P-13).
    ///
    /// Naming a parent is what lets inheritance follow **the axis a
    /// family varies along** (P-2). On a size axis the unfiltered
    /// ground truth cannot be shared, because it depends on
    /// `base_count`; on a selectivity axis every member has one
    /// `base_count` and the ground truth is invariant, so a member
    /// naming its sized parent inherits it instead of restating a path
    /// per member and drifting the moment one is edited.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "name_or_number")]
    pub inherits: Option<String>,

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

impl ProfileConfig {
    /// Every declared facet of this profile, by canonical key.
    ///
    /// One list, so a consumer that must visit every facet — validation,
    /// the manifest, precache — cannot visit a different set than its
    /// neighbours by forgetting a field.
    pub fn facets(&self) -> Vec<(&'static str, &FacetConfig)> {
        [
            ("base_vectors", &self.base_vectors),
            ("base_content", &self.base_content),
            ("query_vectors", &self.query_vectors),
            ("query_terms", &self.query_terms),
            ("query_filters", &self.query_filters),
            ("neighbor_indices", &self.neighbor_indices),
            ("neighbor_distances", &self.neighbor_distances),
            ("prefiltered_neighbor_indices", &self.prefiltered_neighbor_indices),
            ("prefiltered_neighbor_distances", &self.prefiltered_neighbor_distances),
            ("postfiltered_neighbor_indices", &self.postfiltered_neighbor_indices),
            ("postfiltered_neighbor_distances", &self.postfiltered_neighbor_distances),
            ("metadata_content", &self.metadata_content),
            ("metadata_predicates", &self.metadata_predicates),
            ("metadata_results", &self.predicate_results),
            ("metadata_layout", &self.metadata_layout),
        ]
        .into_iter()
        .filter_map(|(name, slot)| slot.as_ref().map(|f| (name, f)))
        .collect()
    }
}

/// A **v2** facet declaration: one or more files in a series.
///
/// A separate case, not extra optional fields on the v1 shape, so the
/// version is visible in the type rather than probed from whether an
/// option happens to be set
/// (`docs/design/srd-dataset-format-version.md`, V-16).
///
/// Its numbers are **required**, which makes the malformed intermediate
/// states unrepresentable: "`NNNN` without `shard_stride`" is a parse
/// failure here rather than a rule someone has to remember to check
/// (V-21, SH-47).
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ShardedFacet {
    /// Filenames derived from an `NNNN` field, lengths from the stride
    /// (SRD SH-49).
    Uniform {
        /// Pattern carrying the `NNNN` shard field.
        source: String,
        /// Ordinals per shard, for every shard but the last.
        shard_stride: u64,
        /// Number of shards; always `>= 2` (SH-4).
        shard_count: u32,
        /// Total records across the series (SH-8).
        record_count: u64,
        /// Optional profile-level window, in facet ordinals (SH-67).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<String>,
    },
    /// An explicit list of files in ordinal order (SRD SH-50).
    Explicit {
        /// Source strings, in ordinal order.
        source: Vec<String>,
        /// Total records across the series (SH-8).
        record_count: u64,
        /// Optional profile-level window, in facet ordinals (SH-67).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<String>,
    },
}

/// The declaration as written, before the one normalisation the
/// loader applies (see [`ShardedFacet`]'s `Deserialize`).
#[derive(Deserialize)]
#[serde(untagged)]
enum ShardedFacetRaw {
    Uniform {
        source: String,
        shard_stride: u64,
        shard_count: u32,
        record_count: u64,
        #[serde(default)]
        window: Option<String>,
    },
    Explicit {
        source: Vec<String>,
        record_count: u64,
        #[serde(default)]
        window: Option<String>,
    },
}

impl<'de> Deserialize<'de> for ShardedFacet {
    /// A window written on a uniform pattern is the facet's window
    /// (SH-102). A pattern names no file, so the entry-window reading
    /// (file ordinals, SH-67) does not exist for it, and the loader
    /// moves the suffix to `window` so every reader sees one spelling.
    /// Giving both is a contradiction and is refused.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match ShardedFacetRaw::deserialize(d)? {
            ShardedFacetRaw::Uniform {
                source,
                shard_stride,
                shard_count,
                record_count,
                window,
            } => {
                let (source, window) = split_pattern_window(source, window)
                    .map_err(serde::de::Error::custom)?;
                ShardedFacet::Uniform {
                    source,
                    shard_stride,
                    shard_count,
                    record_count,
                    window,
                }
            }
            ShardedFacetRaw::Explicit {
                source,
                record_count,
                window,
            } => ShardedFacet::Explicit {
                source,
                record_count,
                window,
            },
        })
    }
}

/// Split a `[a..b)` suffix off a uniform pattern into the facet
/// window. The pattern comes back bare; a pattern without a suffix is
/// returned as it was.
fn split_pattern_window(
    pattern: String,
    window: Option<String>,
) -> Result<(String, Option<String>), String> {
    let parsed = crate::dataset::source::parse_source_string(&pattern)?;
    if parsed.window.is_empty() {
        return Ok((pattern, window));
    }
    if window.is_some() {
        return Err(format!(
            "'{pattern}': the facet window is given twice, on the pattern and as 'window'"
        ));
    }
    let facet_window = parsed.window.to_string();
    let bare = crate::dataset::source::DSSource {
        window: Default::default(),
        ..parsed
    };
    Ok((bare.to_string(), Some(facet_window)))
}

impl ShardedFacet {
    /// Source strings, in ordinal order — one pattern for the uniform
    /// form, the list for the explicit one.
    pub fn sources(&self) -> &[String] {
        match self {
            Self::Uniform { source, .. } => std::slice::from_ref(source),
            Self::Explicit { source, .. } => source,
        }
    }

    /// The optional profile-level window.
    pub fn window(&self) -> Option<&str> {
        match self {
            Self::Uniform { window, .. } | Self::Explicit { window, .. } => window.as_deref(),
        }
    }

    /// Declared total records.
    pub fn record_count(&self) -> u64 {
        match self {
            Self::Uniform { record_count, .. } | Self::Explicit { record_count, .. } => {
                *record_count
            }
        }
    }
}

/// Configuration for a single facet (file resource) of a dataset.
///
/// **The version is the shape.** `Simple` and `Detailed` are the v1
/// format, unchanged and unmoved; `Sharded` is what v2 adds. A v1
/// declaration *is* a v2 declaration, held as the case that carries it
/// rather than converted into anything
/// (`docs/design/srd-dataset-format-version.md`, V-17), and v2 adds a
/// case without redefining either of v1's (V-18).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FacetConfig {
    /// **v1** — simple filename string (e.g., "base.fvec").
    Simple(String),
    /// **v2** — a multi-file series.
    ///
    /// Ahead of `Detailed` in the untagged order because `Detailed`
    /// ignores unknown fields and would otherwise swallow a sharded
    /// declaration, dropping exactly the fields that make it one.
    Sharded(ShardedFacet),
    /// **v1** — detailed configuration object.
    Detailed {
        /// The source filename or path.
        source: String,
        /// Optional window/range string (e.g., "0..1000") to select a subset.
        #[serde(default)]
        window: Option<String>,
    },
}

impl FacetConfig {
    /// The lowest `dataset.yaml` format version that can express this
    /// facet.
    ///
    /// **Derived from the shape, never asserted** (V-19). A writer that
    /// stamped a version instead of folding this would be restating the
    /// rule, and a restatement can drift.
    pub fn min_format_version(&self) -> u32 {
        match self {
            Self::Simple(_) | Self::Detailed { .. } => FORMAT_VERSION_BASE,
            Self::Sharded(_) => FORMAT_VERSION_SHARDED,
        }
    }

    /// The facet's source strings, in ordinal order.
    pub fn sources(&self) -> &[String] {
        match self {
            Self::Simple(s) => std::slice::from_ref(s),
            Self::Detailed { source, .. } => std::slice::from_ref(source),
            Self::Sharded(sh) => sh.sources(),
        }
    }

    /// The single source string, or `None` for a series.
    ///
    /// Returning `None` rather than a stand-in is deliberate: a caller
    /// written before sharding must fail visibly rather than read a
    /// series as something it is not (SH-74). For the explicit form the
    /// stand-in would be the first shard; for the uniform form it would
    /// be the `NNNN` *pattern*, which names no file at all.
    pub fn source(&self) -> Option<&str> {
        match self {
            Self::Simple(s) => Some(s),
            Self::Detailed { source, .. } => Some(source),
            Self::Sharded(_) => None,
        }
    }

    /// Returns the optional window string.
    pub fn window(&self) -> Option<&str> {
        match self {
            Self::Simple(_) => None,
            Self::Detailed { window, .. } => window.as_deref(),
            Self::Sharded(sh) => sh.window(),
        }
    }

    /// Every file this facet names, with a uniform pattern expanded.
    ///
    /// [`Self::sources`] returns the declaration as written, which for
    /// the uniform form is one string containing `NNNN` — a pattern,
    /// not a filename. A caller reasoning about *files* (cache paths,
    /// publication, collisions) needs the expansion, and it is pure
    /// string work: the names follow from the pattern and the count
    /// with nothing to read.
    pub fn declared_files(&self) -> Vec<String> {
        match self {
            Self::Sharded(ShardedFacet::Uniform {
                source,
                shard_count,
                ..
            }) => (0..*shard_count)
                .map(|i| crate::dataset::shards::shard_filename(source, i))
                .collect(),
            other => other.sources().to_vec(),
        }
    }

    /// Ordinals per shard for a uniform series, if this is one.
    pub fn shard_stride(&self) -> Option<u64> {
        match self {
            Self::Sharded(ShardedFacet::Uniform { shard_stride, .. }) => Some(*shard_stride),
            _ => None,
        }
    }

    /// Shard count for a uniform series, if this is one.
    pub fn shard_count(&self) -> Option<u32> {
        match self {
            Self::Sharded(ShardedFacet::Uniform { shard_count, .. }) => Some(*shard_count),
            _ => None,
        }
    }

    /// Declared total records, if this facet declares one.
    pub fn record_count(&self) -> Option<u64> {
        match self {
            Self::Sharded(sh) => Some(sh.record_count()),
            _ => None,
        }
    }

    /// This facet's declaration, in the form the shard model consumes.
    ///
    /// The single adapter between the serde types and the ordinal model
    /// — both loaders realize through it, so neither can grow its own
    /// interpretation of a declaration (SH-90).
    pub(crate) fn declaration(&self) -> crate::dataset::shards::Declaration<'_> {
        crate::dataset::shards::Declaration {
            sources: self.sources(),
            is_array: self.is_explicit_series(),
            shard_stride: self.shard_stride(),
            shard_count: self.shard_count(),
            record_count: self.record_count(),
        }
    }

    /// Whether this facet declares an explicit series (an array
    /// `source`) rather than a single string.
    pub fn is_explicit_series(&self) -> bool {
        matches!(self, Self::Sharded(ShardedFacet::Explicit { .. }))
    }

    /// Whether this facet declares a multi-file series, in either form.
    ///
    /// Structural: the case says so, rather than being inferred from
    /// which optional fields are set (V-16).
    pub fn is_series(&self) -> bool {
        matches!(self, Self::Sharded(_))
    }

    /// This facet as a **v1** declaration, if it is expressible as one.
    ///
    /// `Some` *is* the proof that a v1 reader can read it — there is no
    /// separate compatibility check to keep in step (V-20).
    pub fn try_as_v1(&self) -> Option<&Self> {
        (self.min_format_version() <= FORMAT_VERSION_BASE).then_some(self)
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    /// **A reader sees the sized profiles a file declares compactly.**
    /// The reader once skipped `sized:` and served only the profiles
    /// written out, so a program never saw the sizes the pipeline
    /// builds and lists.
    #[test]
    fn a_reader_sees_compactly_declared_sized_profiles() {
        let config: DatasetConfig = serde_yaml::from_str(
            "name: ds\nprofiles:\n  default:\n    base_vectors: base.fvec\n    query_vectors: q.fvec\n\
             \x20 sized: [10k, 20k]\n",
        )
        .unwrap();
        let p = config.profiles.get("20k").expect("the 20k profile is visible");
        assert_eq!(p.base_count, Some(20_000));
        assert_eq!(p.query_vectors.as_ref().and_then(|f| f.source()), Some("q.fvec"));
    }

    /// **A partition reads only its own facets.** A profile whose base
    /// vectors live in its own directory holds other rows than the
    /// default's, so the default's metadata, cut to its count, would
    /// describe the wrong rows. The reader once served exactly that.
    #[test]
    fn a_partition_gets_none_of_the_defaults_metadata() {
        let config: DatasetConfig = serde_yaml::from_str(
            "name: ds\nprofiles:\n  default:\n    base_vectors: profiles/base/base.fvec\n\
             \x20   metadata_content: profiles/base/meta.u8\n\
             \x20 label_03:\n    base_count: 82915\n    base_vectors: profiles/label_03/base.fvec\n",
        )
        .unwrap();
        let p = &config.profiles["label_03"];
        assert!(p.partition);
        assert!(p.metadata_content.is_none(), "{:?}", p.metadata_content);
    }

    /// SH-102 at the model layer: the spelling tessera's sized profiles
    /// were published with reads as a facet window, and the reader that
    /// opens the profile sees it through `window()`.
    #[test]
    fn a_window_on_a_uniform_pattern_loads_as_the_facet_window() {
        let yaml = r#"
format_version: 2
profiles:
  default:
    base_vectors:
      source: profiles/base/base_vectors__NNNN.fvecs
      shard_stride: 100
      shard_count: 5
      record_count: 496
  10m:
    base_count: 10
    base_vectors:
      source: profiles/base/base_vectors__NNNN.fvecs[0..10]
      shard_stride: 100
      shard_count: 5
      record_count: 496
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let sized = config.profiles.get("10m").unwrap();
        let base = sized.base_vectors.as_ref().unwrap();
        assert_eq!(base.sources(), ["profiles/base/base_vectors__NNNN.fvecs"]);
        assert_eq!(base.window(), Some("[0..10]"));
        let twice = r#"
format_version: 2
profiles:
  default:
    base_vectors:
      source: b__NNNN.fvecs[0..10]
      shard_stride: 100
      shard_count: 5
      record_count: 496
      window: "[0..5]"
"#;
        assert!(serde_yaml::from_str::<DatasetConfig>(twice).is_err());
    }

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
        assert_eq!(profile.base_vectors.as_ref().unwrap().source(), Some("base.fvec"));
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
                // Windows come back in the canonical grammar the
                // description's parser writes; the bounds are what count.
                assert_eq!(window.as_deref(), Some("[0..1000]"));
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
            profile.prefiltered_neighbor_indices.as_ref().and_then(|f| f.source()),
            Some("filtered.ivec"),
        );
        assert_eq!(
            profile.prefiltered_neighbor_distances.as_ref().and_then(|f| f.source()),
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
            profile.prefiltered_neighbor_indices.as_ref().and_then(|f| f.source()),
            Some("prefiltered.ivec"),
        );
        assert_eq!(
            profile.prefiltered_neighbor_distances.as_ref().and_then(|f| f.source()),
            Some("prefiltered.fvec"),
        );
        assert_eq!(
            profile.postfiltered_neighbor_indices.as_ref().and_then(|f| f.source()),
            Some("postfiltered.ivec"),
        );
        assert_eq!(
            profile.postfiltered_neighbor_distances.as_ref().and_then(|f| f.source()),
            Some("postfiltered.fvec"),
        );
    }

    /// **The axis of a step is derived from `base_count`** (PL-5): a
    /// `20m` that names `10m` re-cuts the windows to twenty million and
    /// takes no ground truth, exactly as it would under `default`.
    #[test]
    fn the_axis_is_derived_from_base_count() {
        let yaml = r#"
format_version: 3
profiles:
  default:
    base_vectors: base.fvec
    metadata_content: meta.slab
    query_vectors: q.fvec
    neighbor_indices: profiles/default/gt.ivec
  10m:
    inherits: default
    base_count: 10000000
    neighbor_indices: profiles/10m/gt.ivec
  20m:
    inherits: 10m
    base_count: 20000000
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let p = &config.profiles["20m"];
        assert_eq!(p.base_vectors.as_ref().and_then(|f| f.source()), Some("base.fvec[0..20000000]"), "the window is re-cut, not the parent's");
        assert_eq!(p.metadata_content.as_ref().and_then(|f| f.source()), Some("meta.slab[0..20000000]"));
        assert_eq!(p.query_vectors.as_ref().and_then(|f| f.source()), Some("q.fvec"));
        assert!(p.neighbor_indices.is_none(), "ground truth does not cross a size step, whatever the parent is called");
    }

    /// **Layers stack** (PL-5, PL-2): `default` → `10m` → a set at the
    /// same count inherits the layer's windows and unfiltered ground
    /// truth unchanged and declares its predicate group; two sets may
    /// read one slab (case 7).
    #[test]
    fn stacked_layers_derive_each_axis_from_base_count() {
        let yaml = r#"
format_version: 3
profiles:
  default:
    base_vectors: base.fvec
    metadata_content: meta.slab
    query_vectors: q.fvec
    metadata_predicates: profiles/base/predicates.slab
    neighbor_indices: profiles/default/gt.ivec
  10m:
    inherits: default
    base_count: 10000000
    neighbor_indices: profiles/10m/gt.ivec
  10m-mixed:
    inherits: 10m
    base_count: 10000000
    metadata_results: profiles/10m-mixed/results.slab
    prefiltered_neighbor_indices: profiles/10m-mixed/pre.ivec
  10m-uniform-2-1e-2:
    inherits: 10m
    metadata_predicates: profiles/10m-uniform-2-1e-2/predicates.slab
    metadata_results: profiles/10m-uniform-2-1e-2/results.slab
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let set = &config.profiles["10m-mixed"];
        assert_eq!(set.base_vectors.as_ref().and_then(|f| f.source()), Some("base.fvec[0..10000000]"));
        assert_eq!(set.neighbor_indices.as_ref().and_then(|f| f.source()), Some("profiles/10m/gt.ivec"), "the unfiltered ground truth crosses a same-size step");
        assert_eq!(set.metadata_predicates.as_ref().and_then(|f| f.source()), Some("profiles/base/predicates.slab"), "the invariant slab reaches the set through the layer");
        assert_eq!(set.predicate_results.as_ref().and_then(|f| f.source()), Some("profiles/10m-mixed/results.slab"));
        let uniform = &config.profiles["10m-uniform-2-1e-2"];
        assert_eq!(uniform.base_count, Some(10000000), "a set without a count is at its layer's");
        assert_eq!(uniform.metadata_predicates.as_ref().and_then(|f| f.source()), Some("profiles/10m-uniform-2-1e-2/predicates.slab"), "a set's own slab overrides");
        assert_eq!(uniform.neighbor_indices.as_ref().and_then(|f| f.source()), Some("profiles/10m/gt.ivec"));
        let layer = &config.profiles["10m"];
        assert!(layer.predicate_results.is_none(), "a layer holds no predicate group of its own");
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
        assert_eq!(bv.source(), Some("profiles/base/base_vectors.fvecs[0..10000000]"),
            "inherited base_vectors must carry a [0..base_count) window suffix");

        // Inherited as-is (query sets are shared across profiles).
        assert_eq!(
            p.query_vectors.as_ref().and_then(|f| f.source()),
            Some("profiles/base/query_vectors.fvecs"),
            "query_vectors must be inherited from default without windowing");

        // Per-profile output facets keep their own paths.
        assert_eq!(
            p.neighbor_indices.as_ref().and_then(|f| f.source()),
            Some("profiles/10m/neighbor_indices.ivecs"));
        assert_eq!(
            p.neighbor_distances.as_ref().and_then(|f| f.source()),
            Some("profiles/10m/neighbor_distances.fvecs"));
    }

    /// Partition profiles (`partition: true`) own their own base
    /// bytes and must NOT pick up default's shared facets, even when
    /// they don't declare their own. Defended explicitly because
    /// silent inheritance would mis-route partition reads to the
    /// default's full-base file.
    /// A results index is derived from `base_count` like the ground
    /// truth: across a size step it is declared or absent, never the
    /// parent's; across a selectivity step it inherits like any
    /// invariant facet. The predicate slab crosses the size step, since
    /// the predicates are the same at every size.
    #[test]
    fn metadata_results_does_not_cross_the_size_axis() {
        let yaml = r#"
format_version: 3
attributes: {}
profiles:
  default:
    base_vectors: profiles/base/base_vectors.fvecs
    metadata_predicates: profiles/base/predicates.slab
    metadata_results: profiles/default/metadata_results.slab
  10m:
    inherits: default
    base_count: 10000000
    neighbor_indices: profiles/10m/neighbor_indices.ivecs
  10m-sel:
    inherits: 10m
    metadata_results: profiles/10m-sel/metadata_results.slab
  10m-again:
    inherits: 10m-sel
"#;
        let config: DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let sized = config.profiles.get("10m").unwrap();
        assert!(
            sized.predicate_results.is_none(),
            "a sized child must not take default's results index at another size"
        );
        assert_eq!(
            sized.metadata_predicates.as_ref().and_then(|f| f.source()),
            Some("profiles/base/predicates.slab"),
            "the predicate slab is the same at every size and crosses"
        );
        let same_size = config.profiles.get("10m-again").unwrap();
        assert_eq!(
            same_size.predicate_results.as_ref().and_then(|f| f.source()),
            Some("profiles/10m-sel/metadata_results.slab"),
            "across a step at one size the results index inherits like any invariant facet"
        );
    }

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
        assert_eq!(p.base_vectors.as_ref().and_then(|f| f.source()),
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
        assert_eq!(p.base_vectors.as_ref().and_then(|f| f.source()),
            Some("base.fvecs"),
            "no base_count → inherit unwindowed");
    }
}
