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
        self.profiles
            .values()
            .flat_map(|p| p.facets())
            .map(|(_, f)| f.min_format_version())
            .max()
            .unwrap_or(FORMAT_VERSION_BASE)
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

/// The highest `dataset.yaml` format version this build can read.
///
/// A dataset declaring more than this is refused at load, naming both
/// numbers — the diagnosis the field exists to provide (V-9).
pub const FORMAT_VERSION_SUPPORTED: u32 = FORMAT_VERSION_SHARDED;

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
        _ => Ok(()),
    }
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
            format_version: Option<u32>,
            #[serde(default)]
            attributes: HashMap<String, serde_yaml::Value>,
            #[serde(default)]
            profiles: HashMap<String, serde_yaml::Value>,
        }

        let raw = Raw::deserialize(deserializer)?;

        // Refuse anything this build cannot read, before a single facet
        // is opened (V-9, V-10). A dataset half-read is a view with a
        // hole in it, and a caller that checks only what it touched will
        // not find the hole.
        let format_version =
            check_supported_version(raw.format_version).map_err(serde::de::Error::custom)?;
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
            // A mapping carrying shard fields is unambiguously a
            // profile, so a parse failure there is a broken declaration
            // and not the compact `sized:` shorthand. Skipping it would
            // make the facet vanish rather than complain — the silent
            // shape SH-74 exists to forbid. Narrowed to the shard keys
            // so nothing that parses today changes behaviour.
            let declares_shards = value.as_mapping().is_some_and(|m| {
                m.values().any(|v| {
                    v.as_mapping().is_some_and(|f| {
                        ["shard_stride", "shard_count", "record_count"]
                            .iter()
                            .any(|k| f.contains_key(serde_yaml::Value::from(*k)))
                    })
                })
            });
            match serde_yaml::from_value::<ProfileConfig>(value) {
                Ok(cfg) => { profiles.insert(name, cfg); }
                Err(e) if declares_shards => {
                    return Err(serde::de::Error::custom(format!(
                        "profile '{name}' declares shard fields but does not parse: {e}"
                    )));
                }
                Err(e) => {
                    log::trace!(
                        "skipping unparseable profile entry '{}': {}", name, e);
                }
            }
        }

        // Realize every facet's declaration far enough to catch a
        // declaration that disagrees with itself, before anything else
        // sees it (SH-85). Lengths that only a file can answer are
        // deferred — there is no dataset root here to resolve a relative
        // path against.
        for (profile_name, cfg) in &profiles {
            for (facet_name, facet) in cfg.facets() {
                crate::dataset::shards::validate_declaration(facet_name, &facet.declaration())
                    .map_err(|e| {
                        serde::de::Error::custom(format!("profile '{profile_name}': {e}"))
                    })?;
            }
        }

        apply_default_inheritance(&mut profiles);

        // A *stated* version lower than the content requires is a
        // declaration contradicting itself, and is refused here — the
        // same class of fault as a record count that disagrees with its
        // shards (SH-8).
        //
        // An **absent** field is not a claim. It means 1 for the purpose
        // of the gate above (V-2), but a dataset that never declared a
        // version has not understated one, and a reader new enough to
        // notice is new enough to read it. Under-annotation is a note
        // from `veks check`, not a load failure — refusing it here would
        // reject every hand-written sharded dataset for a field that
        // helps no reader which can already read it.
        let required = profiles
            .values()
            .flat_map(|p| p.facets())
            .map(|(_, f)| f.min_format_version())
            .max()
            .unwrap_or(FORMAT_VERSION_BASE);
        check_stated_against_content(raw.format_version, required)
            .map_err(serde::de::Error::custom)?;

        Ok(DatasetConfig {
            format_version,
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
    // Resolve each profile against its parent, parents first, so a
    // chain (`1m-sel001` → `1m` → `default`) resolves in one pass with
    // the grandparent's facets already in place.
    for name in inheritance_order(profiles) {
        let Some(parent_name) = parent_of(profiles, &name) else {
            continue;
        };
        let Some(parent) = profiles.get(&parent_name).cloned() else {
            continue;
        };
        let Some(profile) = profiles.get_mut(&name) else {
            continue;
        };
        inherit_from(profile, &parent, parent_name == "default");
    }
}

/// The parent a profile inherits from, or `None` when it inherits from
/// nothing.
///
/// A named parent must exist and must not be the profile itself; an
/// unresolvable name falls back to `default` rather than failing the
/// load, because the facets a profile *does* declare are still readable
/// and `veks check` is where a broken declaration should be reported.
fn parent_of(profiles: &HashMap<String, ProfileConfig>, name: &str) -> Option<String> {
    if name == "default" {
        return None;
    }
    let profile = profiles.get(name)?;
    // Partition profiles are self-contained: they carry their own base
    // bytes and must not pick up another profile's shared facets.
    if profile.partition {
        return None;
    }
    match profile.inherits.as_deref() {
        Some(parent) if parent != name && profiles.contains_key(parent) => {
            Some(parent.to_string())
        }
        _ => profiles.contains_key("default").then(|| "default".to_string()),
    }
}

/// Profile names ordered so a parent is always resolved before its
/// children.
///
/// A cycle — `a: inherits: b`, `b: inherits: a` — leaves both members
/// unresolved rather than looping. They keep whatever they declared,
/// which is the same outcome as having no parent, and `veks check`
/// names the cycle.
fn inheritance_order(profiles: &HashMap<String, ProfileConfig>) -> Vec<String> {
    let mut names: Vec<String> = profiles.keys().cloned().collect();
    names.sort();
    let mut done: Vec<String> = Vec::with_capacity(names.len());
    let mut settled: std::collections::HashSet<String> =
        std::collections::HashSet::from(["default".to_string()]);
    // At most one profile is settled per pass, so the depth of the
    // deepest chain bounds the passes.
    for _ in 0..names.len() {
        let mut progressed = false;
        for name in &names {
            if settled.contains(name) {
                continue;
            }
            let ready = match parent_of(profiles, name) {
                None => true,
                Some(parent) => settled.contains(&parent),
            };
            if ready {
                done.push(name.clone());
                settled.insert(name.clone());
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    done
}

/// Copy the facets `profile` did not declare from `parent`.
///
/// **What inherits depends on the axis** (P-2). Per-profile outputs —
/// the unfiltered and filtered ground truth — cannot be shared across a
/// *size* family, because they are derived from `base_count`; a 1m
/// profile inheriting the default's full-base ground truth would
/// mis-route every verify. Across a *selectivity* family they are
/// invariant, and restating them per member is the drift hazard the
/// axis was supposed to remove.
///
/// The parent decides which it is. Inheriting from `default` keeps the
/// size-axis rule, which is what every dataset written before
/// parameterized profiles relies on (P-13). Naming a parent explicitly
/// is a statement that this profile is a parameterization *of that
/// one*, so everything it does not override comes across — the general
/// case of which the default rule is one instance (P-3).
fn inherit_from(profile: &mut ProfileConfig, parent: &ProfileConfig, size_axis: bool) {
    let bc = profile.base_count;
    inherit_with_window(&mut profile.base_vectors, &parent.base_vectors, bc);
    inherit_with_window(&mut profile.metadata_content, &parent.metadata_content, bc);
    inherit(&mut profile.base_content, &parent.base_content);
    inherit(&mut profile.query_vectors, &parent.query_vectors);
    inherit(&mut profile.query_terms, &parent.query_terms);
    inherit(&mut profile.query_filters, &parent.query_filters);
    inherit(&mut profile.metadata_predicates, &parent.metadata_predicates);
    inherit(&mut profile.predicate_results, &parent.predicate_results);
    inherit(&mut profile.metadata_layout, &parent.metadata_layout);
    if !size_axis {
        inherit(&mut profile.neighbor_indices, &parent.neighbor_indices);
        inherit(&mut profile.neighbor_distances, &parent.neighbor_distances);
        inherit(
            &mut profile.prefiltered_neighbor_indices,
            &parent.prefiltered_neighbor_indices,
        );
        inherit(
            &mut profile.prefiltered_neighbor_distances,
            &parent.prefiltered_neighbor_distances,
        );
        inherit(
            &mut profile.postfiltered_neighbor_indices,
            &parent.postfiltered_neighbor_indices,
        );
        inherit(
            &mut profile.postfiltered_neighbor_distances,
            &parent.postfiltered_neighbor_distances,
        );
        if profile.base_count.is_none() {
            profile.base_count = parent.base_count;
        }
    }
    if profile.maxk.is_none() {
        profile.maxk = parent.maxk;
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
            let window = if window.is_some() || source.contains('[') {
                window
            } else {
                Some(format!("0..{bc}"))
            };
            FacetConfig::Detailed { source, window }
        }
        // A profile window is in *facet* ordinals; an entry window
        // inside a series source is in *file* ordinals (SH-67). Only the
        // former is what inheritance sets, so a series is windowed
        // through its own field and never by inspecting its entries,
        // which describe a different coordinate space.
        FacetConfig::Sharded(sh) => FacetConfig::Sharded(match sh {
            ShardedFacet::Uniform {
                source,
                shard_stride,
                shard_count,
                record_count,
                window,
            } => ShardedFacet::Uniform {
                source,
                shard_stride,
                shard_count,
                record_count,
                window: window.or(Some(format!("0..{bc}"))),
            },
            ShardedFacet::Explicit {
                source,
                record_count,
                window,
            } => ShardedFacet::Explicit {
                source,
                record_count,
                window: window.or(Some(format!("0..{bc}"))),
            },
        }),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

    /// SH-102 at the model layer: the spelling tessera's sized profiles
    /// were published with reads as a facet window, and the reader that
    /// opens the profile sees it through `window()`.
    #[test]
    fn a_window_on_a_uniform_pattern_loads_as_the_facet_window() {
        let yaml = r#"
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
        assert_eq!(bv.source(), Some("profiles/base/base_vectors.fvecs[0..10000000)"),
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
