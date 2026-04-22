// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `datasets import` subcommand — bootstrap a new dataset directory.
//!
//! Implements the idempotent flow-state model from SRD §12. The generator
//! builds the superset pipeline graph, resolves each slot to `Materialized`,
//! `Identity`, or `Absent`, then emits only the materialized steps.

use std::path::{Path, PathBuf};

use crate::formats::VecFormat;

// ---------------------------------------------------------------------------
// Public args (matched from CLI)
// ---------------------------------------------------------------------------

/// Arguments for `datasets import`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ImportArgs {
    pub name: String,
    pub output: PathBuf,
    pub base_vectors: Option<PathBuf>,
    pub query_vectors: Option<PathBuf>,
    pub self_search: bool,
    pub query_count: u32,
    pub metadata: Option<PathBuf>,
    pub ground_truth: Option<PathBuf>,
    pub ground_truth_distances: Option<PathBuf>,
    pub metric: String,
    pub neighbors: u32,
    pub seed: u32,
    pub description: Option<String>,
    pub no_dedup: bool,
    pub no_zero_check: bool,
    /// User-supplied count asserted when `no_dedup=true`. When set,
    /// bootstrap writes `duplicate_count=<v>` into `variables.yaml` and
    /// skips `scan-duplicates` — the derived attribute
    /// `is_duplicate_vector_free` still lands in `dataset.yaml` through
    /// the normal sync path. `None` with `no_dedup=true` means "count
    /// unknown" and `scan-duplicates` still runs.
    pub duplicate_count: Option<u64>,
    /// User-supplied zero count. Same semantics as `duplicate_count`
    /// but for `no_zero_check` / `scan-zeros` / `is_zero_vector_free`.
    pub zero_count: Option<u64>,
    pub no_filtered: bool,
    pub normalize: bool,
    pub force: bool,
    /// Target format for base vectors precision conversion (e.g., "mvec" for f32→f16).
    /// When set, a `convert` step is emitted after the base import/identity step.
    pub base_convert_format: Option<String>,
    /// Target format for query vectors precision conversion.
    pub query_convert_format: Option<String>,
    /// Enable gzip compression for eligible cache artifacts.
    pub compress_cache: bool,
    /// Sized profile specification (e.g., "mul:1m..400m/2, 0m..400m/10m").
    /// When set, generates sized profiles in the dataset.yaml.
    pub sized_profiles: Option<String>,
    /// Fraction of base vectors to use (0.0–1.0, default 1.0 = all).
    /// When < 1.0, the base extraction range is capped so only this
    /// fraction of the available (post-dedup, post-query-split) vectors
    /// are used as the base set.
    pub base_fraction: f64,
    /// When true and base_fraction < 1.0, run dedup+zeros on the full
    /// input before subsetting. Slower but guarantees dedup is stable
    /// regardless of the fraction. When false (default), subset first
    /// then dedup on just the subset — faster but dedup results may
    /// differ from the full-input case.
    pub pedantic_dedup: bool,
    /// Required facets as a set of single-letter codes (BQGDMPRF).
    /// When `None`, facets are inferred from available inputs using
    /// the implication rules in SRD 2.8. When `Some`, only the
    /// specified facets are produced — overriding inference.
    pub required_facets: Option<String>,
    /// Provided facets as a set of single-letter codes (BQGDMPRF).
    /// Uses the same syntax as `required_facets`. When set, only the
    /// specified input facets are considered as provided — any detected
    /// inputs not in this set are disregarded. This helps when multiple
    /// input facets are detected but some should be ignored.
    pub provided_facets: Option<String>,
    /// Number of significant digits for computed counts (like base_end).
    /// Default 2 produces clean sizes (e.g., 180000000 instead of 184623729).
    /// Set to 10+ to effectively disable rounding.
    pub round_digits: u32,
    /// Selectivity ratio for synthesized predicates.
    /// Lower values produce harder predicates (fewer qualifying neighbors).
    /// Default 0.0001 means ~0.01% of base vectors match each predicate.
    pub selectivity: f64,
    /// Number of predicates to generate (default 10000).
    #[serde(default = "default_predicate_count")]
    pub predicate_count: u32,
    /// Predicate generation strategy: "eq" (single-field equality) or
    /// "compound" (multi-field AND).
    #[serde(default = "default_predicate_strategy")]
    pub predicate_strategy: String,
    /// Classic layout: store all artifacts in the dataset root directory
    /// instead of `profiles/base/` and `profiles/default/`. Simpler layout
    /// compatible with ann-benchmarks tooling.
    pub classic: bool,
    /// Pipeline personality: "native" (default) or "knn_utils".
    /// When "knn_utils", emit_steps generates BLAS/numpy-compatible
    /// commands (sort-knnutils, shuffle-knnutils, normalize-knnutils,
    /// knn-blas) instead of the native SimSIMD-based commands.
    #[serde(default = "default_personality")]
    pub personality: String,
    /// Synthesize metadata when none is provided.
    /// When true and metadata is required (M facet) but no `--metadata`
    /// source is given, a synthesis step is emitted.
    #[serde(default)]
    pub synthesize_metadata: bool,
    /// Synthesis mode: "simple-int-eq" (integer equality) or
    /// "conjugate" (compound predicates with selectivity control).
    #[serde(default = "default_synthesis_mode")]
    pub synthesis_mode: String,
    /// Storage format for synthesized metadata/predicates:
    /// "slab" (canonical MNode/PNode in slab files) or
    /// "ivec" (plain integer vectors, lightweight).
    #[serde(default = "default_synthesis_format")]
    pub synthesis_format: String,
    /// Number of integer fields for synthesized metadata (default 3).
    #[serde(default = "default_metadata_fields")]
    pub metadata_fields: u32,
    /// Minimum value (inclusive) for synthesized metadata integer range.
    #[serde(default)]
    pub metadata_range_min: i32,
    /// Maximum value (exclusive) for synthesized metadata integer range.
    #[serde(default = "default_metadata_range_max")]
    pub metadata_range_max: i32,
    /// Minimum value (inclusive) for synthesized predicate value range.
    /// Defaults to metadata_range_min.
    #[serde(default)]
    pub predicate_range_min: i32,
    /// Maximum value (exclusive) for synthesized predicate value range.
    /// Defaults to metadata_range_max.
    #[serde(default = "default_metadata_range_max")]
    pub predicate_range_max: i32,
    /// Number of queries to verify in verify-knn. 0 means all queries.
    #[serde(default)]
    pub verify_knn_sample: u32,
    /// Create oracle partition profiles (one per label value).
    #[serde(default)]
    pub partition_oracles: bool,
    /// Maximum number of partition profiles to create.
    #[serde(default = "default_max_partitions")]
    pub max_partitions: u32,
    /// Behavior when a partition has fewer than k vectors: "error", "warn", "include".
    #[serde(default = "default_on_undersized")]
    pub on_undersized: String,
    /// For COSINE metric: which cosine strategy the KNN engines should
    /// use. `Some("assume_normalized")` uses the inner product on the
    /// assumption that inputs are already unit-normalized (FAISS /
    /// numpy / knn_utils convention). `Some("proper")` computes cosine
    /// in-kernel as `dot / (|q| × |b|)`. `None` means unset — bootstrap
    /// will prompt (or error in non-interactive mode) when metric is
    /// COSINE. Ignored for L2 and DOT_PRODUCT.
    #[serde(default)]
    pub cosine_mode: Option<String>,
}

fn default_max_partitions() -> u32 { 100 }
fn default_on_undersized() -> String { "error".to_string() }

fn default_personality() -> String {
    "native".to_string()
}

fn default_metadata_fields() -> u32 { 3 }
fn default_metadata_range_max() -> i32 { 1000 }
fn default_predicate_count() -> u32 { 10000 }
fn default_predicate_strategy() -> String { "eq".to_string() }
fn default_synthesis_mode() -> String { "simple-int-eq".to_string() }
fn default_synthesis_format() -> String { "slab".to_string() }

impl ImportArgs {
    /// Path prefix for the base/default profile artifacts.
    /// Classic mode: `""` (root). Standard mode: `"profiles/base/"`.
    fn profile_prefix(&self) -> &str {
        if self.classic { "" } else { "profiles/base/" }
    }

    /// Path prefix for the default profile (computed artifacts like KNN).
    /// Classic mode: `""`. Standard mode: `"profiles/default/"`.
    fn default_prefix(&self) -> &str {
        if self.classic { "" } else { "profiles/default/" }
    }

    /// Profile path template variable for sized profiles.
    /// Classic mode: `""`. Standard mode: `"profiles/${profile}/"`.
    fn sized_prefix(&self) -> &str {
        if self.classic { "" } else { "profiles/${profile}/" }
    }
}

/// Canonical facet code definitions.
///
/// Each code maps to one or more profile facets. The codes are used
/// throughout the CLI, TUI, and pipeline generation.
pub const FACET_CODES: &[(&str, char, &str)] = &[
    ("base_vectors",               'B', "Base vectors"),
    ("query_vectors",              'Q', "Query vectors"),
    ("neighbor_indices",           'G', "Ground-truth KNN indices"),
    ("neighbor_distances",         'D', "Ground-truth KNN distances"),
    ("metadata_content",           'M', "Metadata content"),
    ("metadata_predicates",        'P', "Predicates"),
    ("metadata_indices",           'R', "Predicate evaluation results"),
    ("filtered_neighbor_indices",  'F', "Filtered KNN"),
    ("oracle_partitions",          'O', "Oracle partition profiles"),
];

/// Parse a facet specification string into a canonical code string.
///
/// Accepts:
/// - `"BQGD"` — compact codes
/// - `"B,Q,G,D"` or `"base,query,gt,dist"` — comma-separated
/// - `"base query gt dist"` — space-separated names
/// - Full facet names like `"base_vectors,query_vectors"`
/// All recognized single-letter facet codes.
const ALL_FACET_CHARS: &str = "BQGDMPRFObqgdmprfo";

pub fn parse_facet_spec(spec: &str) -> String {
    let spec = spec.trim();

    // Strip leading '=' that can appear from shell quoting like --flag ="value"
    let spec = spec.strip_prefix('=').unwrap_or(spec);

    // '*' or 'all' → every facet (O excluded — must be explicit)
    if spec == "*" || spec.eq_ignore_ascii_case("all") {
        return "BQGDMPRF".to_string();
    }

    // If it's all recognized facet letters (compact codes), treat directly
    if !spec.is_empty()
        && spec.chars().all(|c| ALL_FACET_CHARS.contains(c) || c == '+')
        && !spec.contains(',')
        && !spec.contains(' ')
    {
        // Handle '+' prefix: "+O" means "add O to inferred facets"
        // "+MPRFO" means "add MPRFO to inferred facets"
        // The '+' is stripped — the caller handles merging with inferred set
        return spec.replace('+', "").to_uppercase();
    }

    // Split on comma or space
    let parts: Vec<&str> = if spec.contains(',') {
        spec.split(',').map(|s| s.trim()).collect()
    } else {
        spec.split_whitespace().collect()
    };

    let mut codes = String::new();
    for part in parts {
        let p = part.to_lowercase();
        let code = match p.as_str() {
            "b" | "base" | "base_vectors" => 'B',
            "q" | "query" | "query_vectors" => 'Q',
            "g" | "gt" | "groundtruth" | "neighbor_indices" => 'G',
            "d" | "dist" | "distances" | "neighbor_distances" => 'D',
            "m" | "meta" | "metadata" | "metadata_content" => 'M',
            "p" | "pred" | "predicates" | "metadata_predicates" => 'P',
            "r" | "results" | "metadata_indices" => 'R',
            "f" | "filtered" | "filtered_neighbor_indices" => 'F',
            "o" | "oracle" | "oracles" | "oracle_partitions" | "partitions" => 'O',
            other => {
                // Single uppercase letter?
                if other.len() == 1 && "bqgdmprfo".contains(other) {
                    other.to_uppercase().chars().next().unwrap()
                } else {
                    eprintln!("Warning: unknown facet '{}', ignoring", other);
                    continue;
                }
            }
        };
        if !codes.contains(code) {
            codes.push(code);
        }
    }
    codes
}

/// Resolve required facets from explicit spec or input inference.
///
/// When `provided_facets` is set, inputs not matching the provided set
/// are disregarded before inference runs. This lets the user override
/// auto-detection when multiple input files are present but not all
/// should be used.
///
/// Returns a canonical code string (e.g., "BQGDMPRF").
pub fn resolve_facets(args: &ImportArgs) -> String {
    // '+' prefix means additive: infer first, then add the extra codes.
    // Without '+', the spec replaces inference entirely.
    if let Some(ref spec) = args.required_facets {
        if !spec.trim().starts_with('+') {
            return parse_facet_spec(spec);
        }
        // Fall through to inference, then merge below
    }

    // When provided_facets is set, only consider inputs matching those codes
    let provided = args.provided_facets.as_ref().map(|s| parse_facet_spec(s));

    let has_base = args.base_vectors.is_some()
        && provided.as_ref().map_or(true, |p| p.contains('B'));
    let _has_query = args.query_vectors.is_some()
        && provided.as_ref().map_or(true, |p| p.contains('Q'));
    let has_gt = args.ground_truth.is_some()
        && provided.as_ref().map_or(true, |p| p.contains('G'));
    let has_meta = args.metadata.is_some()
        && provided.as_ref().map_or(true, |p| p.contains('M'));

    let has_gt_dist = args.ground_truth_distances.is_some()
        && provided.as_ref().map_or(true, |p| p.contains('D'));

    // Inference from available inputs (SRD 2.8 implication rules)
    let mut facets = String::new();
    if has_base {
        facets.push('B');
        facets.push('Q'); // Q always implied by B
        facets.push('G'); // G always implied by B+Q
        // D only if distances are explicitly provided — computing D
        // requires a full KNN pass which is expensive and unnecessary
        // when GT indices are already provided
        if has_gt_dist {
            facets.push('D');
        }
        // MPRF: include when metadata is provided OR when BQG are all
        // provided (metadata can be synthesized)
        if has_meta || has_gt {
            facets.push('M');
            facets.push('P');
            facets.push('R');
            facets.push('F');
        }
    }

    // Merge additive extras from '+' prefix (e.g., "+O" → add O to inferred)
    if let Some(ref spec) = args.required_facets {
        if spec.trim().starts_with('+') {
            let extras = parse_facet_spec(spec);
            for c in extras.chars() {
                if !facets.contains(c) {
                    facets.push(c);
                }
            }
        }
    }

    facets
}

/// Extract the oracle partition sub-facets from a facet string.
///
/// The O code carries its own scope: characters after O (in lower or upper
/// case) specify which facets to compute within each partition.
///
/// - `BQGDMPRFO` or `+O` → `O` with default sub-facets `bqg`
/// - `BQGDMPRFObqg` → same (explicit)
/// - `BQGDMPRFObqgmprf` → partitions get full MPRF too
/// - `BQGDMPRFOBQGD` → partitions get BQG + distances
///
/// Returns `(main_facets, oracle_sub_facets)` where main_facets has O
/// stripped and oracle_sub_facets is the uppercase set of per-partition
/// facets (default "BQG" if none specified after O).
pub fn parse_oracle_scope(facets: &str) -> (String, Option<String>) {
    if let Some(o_pos) = facets.find('O') {
        let main = facets[..o_pos].to_string();
        let after_o = &facets[o_pos + 1..];

        let sub_facets = if after_o.is_empty() {
            // O alone → default sub-facets BQG
            "BQG".to_string()
        } else {
            // Characters after O are the partition sub-facets
            after_o.to_uppercase()
        };

        (main, Some(sub_facets))
    } else {
        (facets.to_string(), None)
    }
}

// ---------------------------------------------------------------------------
// Slot model (SRD §12.2)
// ---------------------------------------------------------------------------

/// A resolved artifact — either a real pipeline step or an alias.
#[derive(Clone)]
enum Artifact {
    /// A pipeline step produces this artifact.
    Materialized { step_id: String, output: String },
    /// No step — artifact is an alias to an existing path.
    Identity { path: String },
}

impl Artifact {
    /// The resolved output path regardless of materialization.
    fn path(&self) -> &str {
        match self {
            Artifact::Materialized { output, .. } => output,
            Artifact::Identity { path } => path,
        }
    }

    fn is_materialized(&self) -> bool {
        matches!(self, Artifact::Materialized { .. })
    }

    fn step_id(&self) -> &str {
        match self {
            Artifact::Materialized { step_id, .. } => step_id,
            Artifact::Identity { .. } => "",
        }
    }
}

/// Metadata sub-graph slots (all absent when no metadata provided).
struct MetadataSlots {
    metadata_all: Artifact,
    metadata_content: Artifact,
    survey: Artifact,
    predicates: Artifact,
    predicate_indices: Artifact,
}

/// The complete resolved pipeline graph (SRD §12, §20).
struct PipelineSlots {
    // Vector chain
    all_vectors: Artifact,
    /// Unified prepare step: sort + dedup + normalize + zero-detect + norm-stats
    prepare: Artifact,
    vector_count: Artifact,

    // Query chain
    self_search: bool,
    /// True when non-HDF5 B+Q are combined into a single source (Strategy 1)
    combined_bq: bool,
    shuffle: Option<Artifact>,
    query_vectors: Option<Artifact>,
    base_vectors: Artifact,
    base_count: Option<Artifact>,

    // Metadata chain
    metadata: Option<MetadataSlots>,

    // Ground truth chain
    knn: Option<Artifact>,
    filtered_knn: Option<Artifact>,
}

// ---------------------------------------------------------------------------
// Slot resolution (SRD §12.3)
// ---------------------------------------------------------------------------

/// Resolve all slots from user-provided inputs.
///
/// Path prefixes for profile artifacts are controlled by `args.classic`.
///
/// All input paths are relativized to the output directory so that
/// dataset.yaml never contains absolute paths (SRD requirement).
fn resolve_slots(args: &ImportArgs) -> PipelineSlots {
    let output_dir = &args.output;
    let pp = args.profile_prefix();
    let facets = resolve_facets(args);
    let has_base_facet = facets.contains('B');

    let base_source = args.base_vectors.as_ref()
        .map(|p| relativize_path(p, output_dir))
        .unwrap_or_default();

    // ── Vector chain ─────────────────────────────────────────────────
    let needs_import = has_base_facet && args.base_vectors.as_ref()
        .map(|p| !is_native_xvec_file(p))
        .unwrap_or(false);

    // Determine the output xvec extension from the source element size.
    // Probe the source to match its native precision instead of
    // hardcoding mvec (f16).
    let import_ext = if needs_import {
        args.base_vectors.as_ref()
            .and_then(|p| {
                let fmt = VecFormat::detect(p)?;
                let meta = veks_core::formats::reader::probe_source(p, fmt).ok()?;
                Some(match meta.element_size {
                    1 => "bvecs",
                    2 => "mvecs",
                    8 => "dvecs",
                    _ => "fvecs", // 4 bytes (f32/i32) → fvecs
                })
            })
            .unwrap_or("fvecs")
    } else {
        "fvecs"
    };

    let all_vectors = if needs_import {
        Artifact::Materialized {
            step_id: "convert-vectors".into(),
            output: format!("${{cache}}/all_vectors.{}", import_ext),
        }
    } else {
        Artifact::Identity { path: base_source.clone() }
    };

    // Unified prepare step: sort + dedup + normalize + zero-detect + norm-stats (SRD §20)
    let prepare = if !has_base_facet || args.no_dedup {
        Artifact::Identity { path: String::new() }
    } else {
        Artifact::Materialized {
            step_id: "prepare-vectors".into(),
            output: "${cache}/sorted_ordinals.ivecs".into(),
        }
    };

    let vector_count = if has_base_facet {
        Artifact::Materialized {
            step_id: "count-vectors".into(),
            output: String::new(), // variable, not a file
        }
    } else {
        Artifact::Identity { path: String::new() }
    };

    // ── Query chain ──────────────────────────────────────────────────
    let facets = resolve_facets(args);
    let wants_queries = facets.contains('Q');
    let has_separate_query = args.query_vectors.is_some();
    let is_hdf5_source = args.base_vectors.as_ref()
        .map(|p| VecFormat::detect_from_path(p) == Some(VecFormat::Hdf5))
        .unwrap_or(false);

    // Shuffling randomizes vector order for train/test splitting.
    // Disabled when ground truth is pre-provided (Identity) — shuffling
    // would change ordinals and invalidate the GT.
    let has_precomputed_gt = args.ground_truth.is_some();
    let shuffle_enabled = args.seed != 0 && !has_precomputed_gt;

    // Query generation strategy (SRD §7.4.1, §20.8):
    //   Strategy 1: Non-HDF5 B+Q → combine into single source, shuffle split
    //   Strategy 2: HDF5 B+Q     → independent processing (no combine)
    //   Strategy 3: B only       → self-search via shuffle
    //
    // `combined_bq` is the Strategy 1 path: combine separate B+Q into one
    // source so dedup/sort work over the union, then use the shuffle to
    // split back out. It requires shuffle to function — without shuffle
    // there's no way to recover the train/test split. So gate it on
    // `shuffle_enabled`. If the user disabled shuffle and provided Q
    // separately, treat the inputs as independent (use Q as-is) per the
    // sysref §7.4.1 rule: a user-supplied "no shuffle" is never silently
    // overridden, and Q-provided is categorically not self-search.
    let combined_bq = has_separate_query && !is_hdf5_source
        && !has_precomputed_gt && shuffle_enabled;
    let self_search = wants_queries && (!has_separate_query || combined_bq);
    let has_queries = has_separate_query || self_search;

    // Determine output extension from source format
    let vec_ext = if needs_import { import_ext } else {
        args.base_vectors.as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .and_then(VecFormat::canonical_extension)
            .unwrap_or("fvecs")
    };

    let (shuffle, query_vectors, base_vectors, base_count) = if self_search {
        // Strategy 1 (combined B+Q) or Strategy 3 (B only): shuffle + extract.
        // Self-search uses the shuffle output to choose which base rows
        // become queries (train/test split) — there is no other path.
        //
        // Hard rule (sysref §7): a user-supplied "no shuffle" answer is
        // never silently overridden. If the caller asked for self-search
        // AND disabled shuffle (seed=0), the configuration is internally
        // contradictory; refuse to emit a graph that ignores the
        // user's answer.
        if !shuffle_enabled {
            eprintln!(
                "Error: self-search requires the shuffle to pick the train/test split, \
                 but shuffle was disabled (seed=0).\n\
                 Resolve by either:\n  \
                   - providing a separate query file (--query-vectors), or\n  \
                   - enabling shuffle with a non-zero seed.\n\
                 The pipeline will not silently override your shuffle preference."
            );
            std::process::exit(1);
        }
        let shuffle = Artifact::Materialized {
            step_id: "generate-shuffle".into(),
            output: "${cache}/shuffle.ivecs".into(),
        };
        let qv = Artifact::Materialized {
            step_id: "extract-queries".into(),
            output: format!("{}query_vectors.{}", pp, vec_ext),
        };
        let bv = Artifact::Materialized {
            step_id: "extract-base".into(),
            output: format!("{}base_vectors.{}", pp, vec_ext),
        };
        let bc = Artifact::Materialized {
            step_id: "count-base".into(),
            output: String::new(),
        };
        (Some(shuffle), Some(qv), bv, Some(bc))
    } else if has_separate_query && is_hdf5_source {
        // Strategy 2: HDF5 with separate query — independent processing.
        // Queries get normalize + zero-filter via convert-queries.
        // Base gets full prepare-vectors + shuffle treatment.
        // Shuffle randomizes the base vector order after dedup (optional).
        let shuffle = if shuffle_enabled {
            Some(Artifact::Materialized {
                step_id: "generate-shuffle".into(),
                output: "${cache}/shuffle.ivecs".into(),
            })
        } else {
            None
        };
        let qv = Artifact::Materialized {
            step_id: "convert-queries".into(),
            output: format!("{}query_vectors.{}", pp, vec_ext),
        };
        let bv = Artifact::Materialized {
            step_id: "extract-base".into(),
            output: format!("{}base_vectors.{}", pp, vec_ext),
        };
        let bc = Artifact::Materialized {
            step_id: "count-base".into(),
            output: String::new(),
        };
        (shuffle, Some(qv), bv, Some(bc))
    } else if has_separate_query {
        // Separate query file with independent processing (e.g., pre-computed GT).
        let shuffle = if shuffle_enabled {
            Some(Artifact::Materialized {
                step_id: "generate-shuffle".into(),
                output: "${cache}/shuffle.ivecs".into(),
            })
        } else {
            None
        };
        // Query: needs convert step only if format conversion or normalize is needed
        let needs_query_convert = needs_import || args.normalize;
        let qv = if needs_query_convert {
            Artifact::Materialized {
                step_id: "convert-queries".into(),
                output: format!("{}query_vectors.{}", pp, vec_ext),
            }
        } else {
            Artifact::Identity {
                path: relativize_path(args.query_vectors.as_ref().unwrap(), output_dir),
            }
        };
        // Base: needs extract only if dedup, shuffle, or normalize
        let needs_base_extract = !args.no_dedup || shuffle_enabled || args.normalize;
        let bv = if needs_base_extract {
            Artifact::Materialized {
                step_id: "extract-base".into(),
                output: format!("{}base_vectors.{}", pp, vec_ext),
            }
        } else {
            Artifact::Identity {
                path: relativize_path(args.base_vectors.as_ref().unwrap(), output_dir),
            }
        };
        let bc = if needs_base_extract {
            Some(Artifact::Materialized {
                step_id: "count-base".into(),
                output: String::new(),
            })
        } else {
            None
        };
        (shuffle, Some(qv), bv, bc)
    } else {
        // No queries at all
        let bv = Artifact::Materialized {
            step_id: "extract-base".into(),
            output: format!("{}base_vectors.{}", pp, vec_ext),
        };
        let bc = Artifact::Materialized {
            step_id: "count-base".into(),
            output: String::new(),
        };
        (None, None, bv, Some(bc))
    };

    // ── Metadata chain ───────────────────────────────────────────────
    // Validate: predicates (P/F) without metadata is an error
    let wants_metadata = facets.contains('M');
    let wants_predicates = facets.contains('P') || facets.contains('F');
    if wants_predicates && !wants_metadata && args.metadata.is_none() && !args.synthesize_metadata {
        eprintln!("Error: predicates (P/F facets) require metadata (M facet).");
        eprintln!("  Provide --metadata <file> or use --synthesize-metadata to generate random metadata.");
        std::process::exit(1);
    }

    // When metadata synthesis is requested and no source is provided,
    // create a synthetic metadata source via `generate metadata`.
    let has_metadata_source = args.metadata.is_some();
    let synthesize = args.synthesize_metadata && !has_metadata_source && wants_metadata;

    let metadata = if has_metadata_source && wants_metadata {
        args.metadata.as_ref().map(|meta_source| {
        let needs_meta_import = !is_native_slab_file(meta_source);

        let metadata_all = if needs_meta_import {
            Artifact::Materialized {
                step_id: "convert-metadata".into(),
                output: "${cache}/metadata_all.slab".into(),
            }
        } else {
            Artifact::Identity { path: relativize_path(meta_source, output_dir) }
        };

        // Metadata extract: needed only in self-search (ordinal realignment)
        let metadata_content = if self_search {
            Artifact::Materialized {
                step_id: "extract-metadata".into(),
                output: format!("{}metadata_content.slab", pp),
            }
        } else {
            // Canonical profile path — symlinked to source during import
            Artifact::Identity { path: format!("{}metadata_content.slab", pp) }
        };

        let survey = Artifact::Materialized {
            step_id: "survey-metadata".into(),
            output: "${cache}/metadata_survey.json".into(),
        };
        let predicates = Artifact::Materialized {
            step_id: "generate-predicates".into(),
            output: format!("{}predicates.slab", pp),
        };
        let predicate_indices = Artifact::Materialized {
            step_id: "evaluate-predicates".into(),
            output: "metadata_indices.slab".into(),
        };

        MetadataSlots { metadata_all, metadata_content, survey, predicates, predicate_indices }
    })
    } else if synthesize {
        // Synthesize metadata — generate metadata step produces the data.
        // When not self-search (no shuffle/reorder), generate directly to
        // the final location — no extract step needed.
        let is_simple = args.synthesis_mode == "simple-int-eq";
        let ext = if is_simple && args.synthesis_format != "slab" {
            &args.synthesis_format
        } else {
            "slab"
        };

        let metadata_all = if self_search {
            Artifact::Materialized {
                step_id: "generate-metadata".into(),
                output: format!("${{cache}}/metadata_all.{}", ext),
            }
        } else {
            Artifact::Materialized {
                step_id: "generate-metadata".into(),
                output: format!("{}metadata_content.{}", pp, ext),
            }
        };
        let metadata_content = if self_search {
            Artifact::Materialized {
                step_id: "extract-metadata".into(),
                output: format!("{}metadata_content.{}", pp, ext),
            }
        } else {
            metadata_all.clone()
        };
        // Simple-int-eq: no survey needed (schema is known from config).
        // Predicates are generated directly from the configured ranges.
        let survey = if is_simple {
            Artifact::Identity { path: String::new() }
        } else {
            Artifact::Materialized {
                step_id: "survey-metadata".into(),
                output: "${cache}/metadata_survey.json".into(),
            }
        };
        let predicates = Artifact::Materialized {
            step_id: "generate-predicates".into(),
            output: format!("{}predicates.{}", pp, ext),
        };
        // R (predicate results) is always ivec — each record is a
        // variable-length list of matching ordinals, not a scalar value
        let predicate_indices = Artifact::Materialized {
            step_id: "evaluate-predicates".into(),
            output: "metadata_indices.ivvecs".into(),
        };
        Some(MetadataSlots { metadata_all, metadata_content, survey, predicates, predicate_indices })
    } else {
        None
    };

    // ── Ground truth chain ───────────────────────────────────────────
    let wants_gt = facets.contains('G');
    let knn = if !has_queries || !wants_gt {
        None
    } else if args.ground_truth.is_some() {
        Some(Artifact::Identity {
            path: relativize_path(args.ground_truth.as_ref().unwrap(), output_dir),
        })
    } else {
        Some(Artifact::Materialized {
            step_id: "compute-knn".into(),
            output: "neighbor_indices.ivecs".into(), // per_profile
        })
    };

    let wants_filtered = facets.contains('F') && !args.no_filtered;
    let filtered_knn = if !has_queries || metadata.is_none() || !wants_filtered {
        None
    } else {
        Some(Artifact::Materialized {
            step_id: "compute-filtered-knn".into(),
            output: "filtered_neighbor_indices.ivecs".into(), // per_profile
        })
    };

    PipelineSlots {
        all_vectors, prepare, vector_count,
        self_search, combined_bq, shuffle, query_vectors, base_vectors, base_count,
        metadata, knn, filtered_knn,
    }
}

// ---------------------------------------------------------------------------
// Pipeline emission (SRD §12.7)
// ---------------------------------------------------------------------------

/// A step to emit in the YAML.
struct Step {
    id: String,
    run: String,
    description: Option<String>,
    after: Vec<String>,
    per_profile: bool,
    /// Execution phase for profile ordering. Phase 0 (compute) runs
    /// across all profiles before phase 1 (verify) begins.
    #[allow(dead_code)]
    phase: u32,
    /// When true, this step runs in the finalization pass — after all
    /// compute phases (core, deferred sized, partition expansion) complete.
    finalize: bool,
    options: Vec<(String, String)>,
}

impl Default for Step {
    fn default() -> Self {
        Step {
            id: String::new(),
            run: String::new(),
            description: None,
            after: Vec::new(),
            per_profile: false,
            phase: 0,
            finalize: false,
            options: Vec::new(),
        }
    }
}

/// Command name mapping for the knn_utils personality.
///
/// Returns the knn_utils-compatible command name when personality is
/// "knn_utils", otherwise returns the native command name.
fn cmd_sort(personality: &str) -> &'static str {
    match personality {
        "knn_utils" => "compute sort-knnutils",
        _ => "compute sort",
    }
}

fn cmd_shuffle(personality: &str) -> &'static str {
    match personality {
        "knn_utils" => "generate shuffle-knnutils",
        _ => "generate shuffle",
    }
}

/// Build the `base:` option for a per-profile KNN step.
///
/// Three cases:
/// 1. **extract-base materialized** (`slots.base_count.is_some()`): the
///    pipeline produced a dedicated base file (deduped, normalized,
///    shuffled). compute-knn reads from that file with `[0..${base_count})`.
/// 2. **subset applied without extract-base**: subset-vectors (or
///    convert-vectors with `fraction`) wrote a smaller `all_vectors`.
///    compute-knn must read that subset, NOT the original source —
///    otherwise it sees vectors the rest of the pipeline pretends
///    aren't there. The subset's count is in `${vector_count}`.
/// 3. **No subset, no extract-base**: pure identity passthrough.
///    compute-knn reads the whole source file.
fn compute_knn_base_arg(
    slots: &PipelineSlots,
    working_vectors: &str,
    subset_applied: bool,
) -> String {
    if slots.base_count.is_some() {
        format!("{}[0..${{base_count}})", slots.base_vectors.path())
    } else if subset_applied {
        format!("{}[0..${{vector_count}})", working_vectors)
    } else {
        slots.base_vectors.path().to_string()
    }
}

fn cmd_knn(personality: &str) -> &'static str {
    // knn-blas is the canonical default: numpy / knn_utils kernel
    // parity with segment-cache + streaming pread. It requires the
    // `knnutils` feature (system BLAS at link time). When that
    // feature isn't compiled in, fall back to `compute knn` (the
    // stdarch alias) so generated dataset.yaml files reference a
    // command the binary actually registers.
    //
    // Datasets whose base is f16 (mvec) should point the `run:` field
    // at `compute knn` explicitly — knn-blas is f32-only.
    match personality {
        "knn_utils" => {
            #[cfg(feature = "knnutils")]
            { "compute knn-blas" }
            #[cfg(not(feature = "knnutils"))]
            {
                eprintln!(
                    "Warning: --personality knn_utils selected but the `knnutils` feature \
                     is not compiled in. Falling back to `compute knn` (stdarch)."
                );
                "compute knn"
            }
        }
        _ => {
            #[cfg(feature = "knnutils")]
            { "compute knn-blas" }
            #[cfg(not(feature = "knnutils"))]
            { "compute knn" }
        }
    }
}

/// Build the cosine-mode options for a KNN step. When metric is
/// COSINE, returns one of the two explicit flags based on the
/// `cosine_mode` the caller chose. For non-cosine metrics, returns
/// nothing (the flags don't apply).
///
/// The legacy `normalized` option is still emitted as a
/// back-compat alias when applicable — existing consumers of the
/// generated dataset.yaml that don't know the new flags will still
/// understand `normalized`.
fn knn_cosine_opts(args: &ImportArgs) -> Vec<(String, String)> {
    if args.metric.to_uppercase() != "COSINE" {
        // L2 / DOT_PRODUCT: carry the normalized hint through so
        // downstream steps (compute_filtered_knn, etc.) that still
        // read it can infer pipeline-level normalization.
        return vec![
            ("normalized".into(), if args.normalize { "true" } else { "false" }.into()),
        ];
    }
    let mode = args.cosine_mode.as_deref().unwrap_or("assume_normalized");
    match mode {
        "proper" => vec![
            ("use_proper_cosine_metric".into(), "true".into()),
            // If the pipeline still normalizes the data, note it for
            // downstream consumers — doesn't change kernel behavior
            // under `use_proper_cosine_metric`, just informational.
            ("normalized".into(), if args.normalize { "true" } else { "false" }.into()),
        ],
        _ => vec![
            ("assume_normalized_like_faiss".into(), "true".into()),
            ("normalized".into(), "true".into()),
        ],
    }
}

fn cmd_verify_knn(personality: &str) -> &'static str {
    match personality {
        "knn_utils" => "verify dataset-knnutils",
        _ => "verify knn-consolidated",
    }
}

/// Walk the resolved slots and emit pipeline steps for materialized slots.
fn emit_steps(slots: &PipelineSlots, args: &ImportArgs, _output_dir: &std::path::Path) -> Vec<Step> {
    let mut steps = Vec::new();
    let pp = args.profile_prefix();
    // Make source paths relative to the output directory when possible,
    // so the dataset.yaml is portable.
    let base_source = args.base_vectors.as_ref()
        .map(|p| relativize_path(p, &args.output))
        .unwrap_or_default();

    // ── Vector chain ─────────────────────────────────────────────────
    if let Artifact::Materialized { .. } = &slots.all_vectors {
        let source_vec_format = args.base_vectors.as_ref().and_then(|p| VecFormat::detect(p));
        let format = source_vec_format
            .map(|f| f.name().to_string())
            .unwrap_or_else(|| "auto".into());
        let mut convert_opts = vec![
            ("facet".into(), "base_vectors".into()),
            ("source".into(), base_source.clone()),
            ("from".into(), format),
            ("output".into(), slots.all_vectors.path().into()),
        ];
        // When base_fraction < 1.0 and not pedantic, limit the convert
        // step to only import the needed subset. The convert command
        // computes the actual limit from source record count × fraction.
        let uses_fraction = args.base_fraction < 1.0 && !args.pedantic_dedup;
        if uses_fraction {
            convert_opts.push(("fraction".into(), args.base_fraction.to_string()));
        }
        // Pin the compiled parquet→xvec fast path when the source is a
        // parquet file/dir and no option would block it. This makes
        // bootstrap-generated pipelines assert on the fast path: if
        // something later causes `transform convert` to silently fall
        // back to the per-record slow path for this step, the pipeline
        // errors out instead of quietly running 10× slower.
        if source_vec_format == Some(VecFormat::Parquet) && !uses_fraction {
            convert_opts.push(("require_fast".into(), "true".into()));
        }
        steps.push(Step {
            id: "convert-vectors".into(),
            run: "transform convert".into(),
            description: Some("Import base vectors from source format".into()),
            after: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            options: convert_opts,
        });
    }

    let mut last_vector_step = if slots.all_vectors.is_materialized() {
        "convert-vectors"
    } else {
        "" // no dependency
    };

    // ── Early subset for native sources with fraction < 1.0 ──────
    //
    // When the source is native xvec (identity, no convert step) and
    // the user wants a fraction < 1.0 and NOT pedantic dedup, we need
    // to extract the subset before dedup runs. This avoids expensive
    // dedup/sort on the full input.
    //
    // For foreign sources, the convert step's `fraction` option already
    // handles this. For native sources, we insert an explicit extract.
    if args.base_fraction < 1.0 && !args.pedantic_dedup && !slots.all_vectors.is_materialized() {
        let source_path = slots.all_vectors.path().to_string();
        let ext = args.base_vectors.as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .and_then(VecFormat::canonical_extension)
            .unwrap_or("fvec");
        let subset_output = format!("${{cache}}/all_vectors.{}", ext);
        steps.push(Step {
            id: "subset-vectors".into(),
            run: "transform convert".into(),
            description: Some(format!(
                "Subset to {:.0}% of source vectors (fast mode)",
                args.base_fraction * 100.0)),
            after: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("source".into(), source_path),
                ("output".into(), subset_output),
                ("fraction".into(), args.base_fraction.to_string()),
            ],
        });
        last_vector_step = "subset-vectors";
    }

    // Precision convert for base vectors (e.g., f32→f16 or f16→f32)
    if let Some(ref target_fmt) = args.base_convert_format {
        let source = slots.all_vectors.path().to_string();
        let ext = target_fmt;
        // Output replaces the all_vectors path for downstream steps
        let output = format!("${{cache}}/all_vectors.{}", ext);
        let after = if last_vector_step.is_empty() { vec![] } else { vec![last_vector_step.into()] };
        steps.push(Step {
            id: "convert-precision".into(),
            run: "transform convert".into(),
            description: Some(format!("Convert base vectors to {} precision", ext)),
            after,
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("source".into(), source),
                ("output".into(), output),
                ("to".into(), ext.clone()),
            ],
        });
        last_vector_step = "convert-precision";
    }

    // The working vector path: either the subset output or the original.
    // All downstream steps (count, sort, zeros, extract) should use this.
    let working_vectors = if last_vector_step == "subset-vectors" {
        let ext = args.base_vectors.as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .and_then(VecFormat::canonical_extension)
            .unwrap_or("fvec");
        format!("${{cache}}/all_vectors.{}", ext)
    } else {
        slots.all_vectors.path().to_string()
    };

    // set-vector-count (only when base vectors are materialized)
    if slots.vector_count.is_materialized() {
        steps.push(Step {
            id: "count-vectors".into(),
            run: "state set".into(),
            description: None,
            after: if last_vector_step.is_empty() { vec![] } else { vec![last_vector_step.into()] },
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("name".into(), "vector_count".into()),
                ("value".into(), format!("count:{}", working_vectors)),
            ],
        });
    }

    // Source counts: record the original base and query vector counts
    // before any processing (dedup, combine, zero removal).
    if let Some(ref base_path) = args.base_vectors {
        let rel = relativize_path(base_path, &args.output);
        steps.push(Step {
            id: "count-source-base".into(),
            run: "state set".into(),
            description: None,
            after: if last_vector_step.is_empty() { vec![] } else { vec![last_vector_step.into()] },
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("name".into(), "source_base_count".into()),
                ("value".into(), format!("count:{}", rel)),
            ],
        });
    }
    if let Some(ref query_path) = args.query_vectors {
        let rel = relativize_path(query_path, &args.output);
        steps.push(Step {
            id: "count-source-queries".into(),
            run: "state set".into(),
            description: None,
            after: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("name".into(), "source_query_count".into()),
                ("value".into(), format!("count:{}", rel)),
            ],
        });
    } else if args.self_search {
        // Self-search: no separate query source
        steps.push(Step {
            id: "count-source-queries".into(),
            run: "state set".into(),
            description: None,
            after: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("name".into(), "source_query_count".into()),
                ("value".into(), "0".into()),
            ],
        });
    }

    // Dataset metadata flags — persisted to variables.yaml for stats
    let is_shuffled = slots.shuffle.as_ref().map_or(false, |s| s.is_materialized());
    for (name, value) in [
        ("is_shuffled", is_shuffled.to_string()),
        ("is_self_search", slots.self_search.to_string()),
        ("combined_bq", slots.combined_bq.to_string()),
        ("k", args.neighbors.to_string()),
    ] {
        steps.push(Step {
            id: format!("set-{}", name),
            run: "state set".into(),
            description: None,
            after: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("name".into(), name.into()),
                ("value".into(), value),
            ],
        });
    }

    // prepare-vectors: sort + dedup (SRD §20)
    // Normalization and zero detection happen during extraction, where
    // each vector is already in memory for writing.
    if let Artifact::Materialized { .. } = &slots.prepare {
        steps.push(Step {
            id: "prepare-vectors".into(),
            run: cmd_sort(&args.personality).into(),
            description: Some(
                "External merge-sort + duplicate detection \
                 (builds sorted-ordinal index in lex order)".into(),
            ),
            after: if last_vector_step.is_empty() { vec![] } else { vec![last_vector_step.into()] },
            per_profile: false,
            phase: 0,
            finalize: false,
            options: {
                let mut opts = vec![
                    ("source".into(), working_vectors.clone()),
                    ("output".into(), "${cache}/sorted_ordinals.ivecs".into()),
                    ("duplicates".into(), "${cache}/dedup_duplicates.ivecs".into()),
                    ("report".into(), "${cache}/dedup_report.json".into()),
                ];
                if args.compress_cache {
                    opts.push(("compress_cache".into(), "true".into()));
                }
                opts
            },
        });
    }

    // count-duplicates (after prepare-vectors)
    if slots.prepare.is_materialized() {
        steps.push(Step {
            id: "count-duplicates".into(),
            run: "state set".into(),
            description: None,
            after: vec!["prepare-vectors".into()],
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("name".into(), "duplicate_count".into()),
                ("value".into(), "count:${cache}/dedup_duplicates.ivecs".into()),
            ],
        });

        // No separate filter-ordinals, count-clean, find-zeros, count-zeros,
        // or measure-normals steps. All absorbed into prepare-vectors (SRD §20).
        // The exclusion set (duplicates ∪ zeros) is applied directly by
        // extract-base and extract-queries.
    } else if !slots.base_vectors.path().is_empty() {
        // When prepare-vectors is NOT materialized (Identity base, no
        // removal requested), `zero_count` / `duplicate_count` must
        // come from one of:
        //   1. Standalone `scan-zeros` / `scan-duplicates` pipeline
        //      steps — a full-file scan that populates the variable.
        //   2. A user-asserted count seeded into `variables.yaml` at
        //      bootstrap (the wizard asks "how many zeros?" with
        //      default 0 when the user declines removal).
        //
        // Option 2 is the escape hatch for billion-vector sources
        // where the user already knows the source is clean — the
        // scan would take 15+ minutes only to confirm a zero. Skip
        // the step only when the count is explicitly asserted;
        // otherwise the scan MUST run so `veks check` can see
        // `is_*_free` populated in `dataset.yaml`.
        if args.zero_count.is_none() {
            steps.push(Step {
                id: "scan-zeros".into(),
                run: "analyze find-zeros".into(),
                description: Some("Scan source vectors for zero vectors (populates is_zero_vector_free)".into()),
                after: vec!["count-vectors".into()],
                per_profile: false,
                phase: 0,
                finalize: false,
                options: vec![
                    ("source".into(), slots.base_vectors.path().into()),
                ],
            });
        }
        if args.duplicate_count.is_none() {
            steps.push(Step {
                id: "scan-duplicates".into(),
                run: "analyze find-duplicates".into(),
                description: Some("Scan source vectors for duplicates (populates is_duplicate_vector_free)".into()),
                after: vec!["count-vectors".into()],
                per_profile: false,
                phase: 0,
                finalize: false,
                options: vec![
                    ("source".into(), slots.base_vectors.path().into()),
                ],
            });
        }
    }

    // Was the subset already applied by the subset-vectors or convert step?
    // When true, clean_count is already fractioned — don't apply base_end.
    let subset_applied = last_vector_step == "subset-vectors"
        || (slots.all_vectors.is_materialized() && args.base_fraction < 1.0 && !args.pedantic_dedup);

    // ── Shuffle (all strategies that need randomized base order) ────
    // Emitted whenever a shuffle artifact is materialized: self-search
    // (Strategies 1 & 3) AND HDF5 with separate queries (Strategy 2).
    // For self-search, the shuffle also provides the train/test split.
    // For HDF5, the shuffle randomizes base vector order after dedup.
    if let Some(ref shuffle) = slots.shuffle {
        if shuffle.is_materialized() {
            let shuffle_after = if slots.prepare.is_materialized() {
                vec!["prepare-vectors".into()]
            } else {
                vec!["count-vectors".into()]
            };
            let shuffle_interval = if slots.prepare.is_materialized() {
                "${clean_count}".to_string()
            } else {
                "${vector_count}".to_string()
            };
            let mut shuffle_opts = vec![
                ("output".into(), "${cache}/shuffle.ivecs".into()),
                ("interval".into(), shuffle_interval),
                ("seed".into(), format!("${{{}}}", "seed")),
            ];
            if slots.prepare.is_materialized() {
                shuffle_opts.push(("ordinals".into(), slots.prepare.path().into()));
            }
            steps.push(Step {
                id: "generate-shuffle".into(),
                run: cmd_shuffle(&args.personality).into(),
                description: Some("Reproducible random permutation of base vector order".into()),
                after: shuffle_after,
                per_profile: false,
                phase: 0,
                finalize: false,
                options: shuffle_opts,
            });
        }
    }

    // ── Query chain (self-search) ────────────────────────────────────
    if slots.self_search {
        if let Some(ref qv) = slots.query_vectors {
            if qv.is_materialized() {
                let vec_deps = if last_vector_step.is_empty() {
                    vec!["generate-shuffle".into()]
                } else {
                    vec![last_vector_step.into(), "generate-shuffle".into()]
                };
                let total_var = if slots.prepare.is_materialized() {
                    "${clean_count}"
                } else {
                    "${vector_count}"
                };

                // When base_fraction < 1.0, compute a capped base_end — BUT
                // only if the subset step didn't already apply the fraction.
                // If subset-vectors ran, clean_count is already fractioned.
                let base_upper = if args.base_fraction < 1.0 && !subset_applied {
                    let count_dep = if slots.prepare.is_materialized() {
                        "prepare-vectors"
                    } else {
                        "count-vectors"
                    };
                    // base_end = query_count + (total - query_count) * fraction
                    // We use scale: expression which operates on the already-
                    // interpolated variable value at runtime.
                    steps.push(Step {
                        id: "compute-base-end".into(),
                        run: "state set".into(),
                        description: Some(format!("Cap base set to {:.0}% of available vectors", args.base_fraction * 100.0)),
                        after: vec![count_dep.into()],
                        per_profile: false,
                        phase: 0,
                        finalize: false,
                        options: vec![
                            ("name".into(), "base_end".into()),
                            ("value".into(), format!(
                                "scale:{}*{}:round{}",
                                total_var, args.base_fraction, args.round_digits,
                            )),
                        ],
                    });
                    "${base_end}"
                } else {
                    total_var
                };

                let mut query_opts = vec![
                    ("source".into(), working_vectors.clone()),
                    ("ivec-file".into(), "${cache}/shuffle.ivecs".into()),
                    ("output".into(), slots.query_vectors.as_ref().map(|q| q.path().to_string()).unwrap_or_else(|| format!("{}query_vectors.fvecs", pp))),
                    ("range".into(), "[0,${query_count})".into()),
                    ("facet".into(), "query_vectors".into()),
                ];
                query_opts.push(("normalize".into(), "true".into()));
                // Perturb each query by a minimal epsilon on dim 0, scaled by
                // query index. This ensures every query produces a unique
                // distance surface — no two base vectors are ever equidistant
                // from a query, making ground truth neighborhoods canonical.
                query_opts.push(("perturb".into(), "true".into()));

                let mut extract_deps = vec_deps.clone();
                if args.base_fraction < 1.0 && !subset_applied {
                    extract_deps.push("compute-base-end".into());
                }
                steps.push(Step {
                    id: "extract-queries".into(),
                    run: "transform extract".into(),
                    description: Some(format!("First {} shuffled vectors -> query set (perturbed)", args.query_count)),
                    after: extract_deps.clone(),
                    per_profile: false,
                    phase: 0,
                    finalize: false,
                    options: query_opts,
                });
                let base_opts = vec![
                    ("source".into(), working_vectors.clone()),
                    ("ivec-file".into(), "${cache}/shuffle.ivecs".into()),
                    ("output".into(), slots.base_vectors.path().into()),
                    ("range".into(), format!("[${{query_count}},{})", base_upper)),
                    ("normalize".into(), "true".into()),
                ];
                steps.push(Step {
                    id: "extract-base".into(),
                    run: "transform extract".into(),
                    description: Some(if args.base_fraction < 1.0 {
                        format!("Shuffled vectors -> base set ({:.0}%)", args.base_fraction * 100.0)
                    } else {
                        "Remainder of shuffled vectors -> base set".into()
                    }),
                    after: extract_deps,
                    per_profile: false,
                    phase: 0,
                    finalize: false,
                    options: base_opts,
                });
                steps.push(Step {
                    id: "count-base".into(),
                    run: "state set".into(),
                    description: None,
                    after: vec!["extract-base".into()],
                    per_profile: false,
                    phase: 0,
                    finalize: false,
                    options: vec![
                        ("name".into(), "base_count".into()),
                        ("value".into(), format!("count:{}", slots.base_vectors.path())),
                    ],
                });
                // Normalization stats are computed by prepare-vectors (SRD §20).
                // No separate measure-normals step needed.
            }
        }
    } else if let Some(ref qv) = slots.query_vectors {
        // Separate query import
        if qv.is_materialized() {
            let query_source = args.query_vectors.as_ref().unwrap();
            let query_vec_format = VecFormat::detect(query_source);
            let format = query_vec_format
                .map(|f| f.name().to_string())
                .unwrap_or_else(|| "auto".into());
            // Strategy 2 (HDF5): convert queries directly to final output.
            // Normalize + zero-filter applied during conversion.
            steps.push(Step {
                id: "convert-queries".into(),
                run: "transform convert".into(),
                description: Some("Import and normalize query vectors from source format".into()),
                after: vec![],
                per_profile: false,
                phase: 0,
                finalize: false,
                options: {
                    let mut opts = vec![
                        ("facet".into(), "query_vectors".into()),
                        ("source".into(), relativize_path(query_source, &args.output)),
                        ("from".into(), format),
                        ("output".into(), qv.path().into()),
                    ];
                    if args.normalize {
                        opts.push(("normalize".into(), "true".into()));
                    }
                    // Pin the fast path when applicable (parquet source, no
                    // normalize). See `convert-vectors` above for rationale.
                    if query_vec_format == Some(VecFormat::Parquet) && !args.normalize {
                        opts.push(("require_fast".into(), "true".into()));
                    }
                    opts
                },
            });
        }
        // Extract base vectors to the profile path when materialized:
        // - HDF5/npy sources: extract from converted all_vectors
        // - Native xvec with dedup: extract clean subset via clean_ordinals
        if slots.base_vectors.is_materialized() {
            let working = slots.all_vectors.path();
            let has_shuffle = slots.shuffle.as_ref().map_or(false, |s| s.is_materialized());
            let after = if has_shuffle {
                vec!["generate-shuffle".into()]
            } else if slots.prepare.is_materialized() {
                vec!["prepare-vectors".into()]
            } else if slots.all_vectors.is_materialized() {
                vec!["convert-vectors".into()]
            } else {
                vec![]
            };
            let mut base_opts = vec![
                ("source".into(), working.into()),
                ("output".into(), slots.base_vectors.path().into()),
            ];
            if has_shuffle {
                // Use shuffled ordinals for randomized base vector order
                base_opts.push(("ivec-file".into(), "${cache}/shuffle.ivecs".into()));
                base_opts.push(("range".into(), format!("[0,{})", "${clean_count}")));
            } else if slots.prepare.is_materialized() {
                base_opts.push(("ivec-file".into(), slots.prepare.path().into()));
                base_opts.push(("range".into(), format!("[0,{})", "${vector_count}")));
            }
            if args.normalize {
                base_opts.push(("normalize".into(), "true".into()));
            }
            steps.push(Step {
                id: "extract-base".into(),
                run: "transform extract".into(),
                description: Some(if has_shuffle {
                    "Extract shuffled base vectors to profile".into()
                } else {
                    "Extract base vectors to profile".into()
                }),
                after,
                per_profile: false,
                phase: 0,
                finalize: false,
                options: base_opts,
            });
            // Count base vectors
            steps.push(Step {
                id: "count-base".into(),
                run: "state set".into(),
                description: None,
                after: vec!["extract-base".into()],
                per_profile: false,
                phase: 0,
                finalize: false,
                options: vec![
                    ("name".into(), "base_count".into()),
                    ("value".into(), format!("count:{}", slots.base_vectors.path())),
                ],
            });
            // Normalization stats are computed by prepare-vectors (SRD §20).
        }
    }

    // ── Metadata chain ───────────────────────────────────────────────
    if let Some(ref meta) = slots.metadata {
        if meta.metadata_all.is_materialized() {
            if let Some(meta_source) = args.metadata.as_ref() {
                // Import from existing metadata file
                let format = VecFormat::detect(meta_source)
                    .map(|f| f.name().to_string())
                    .unwrap_or_else(|| "parquet".into());
                let mut meta_opts = vec![
                    ("facet".into(), "metadata_content".into()),
                    ("source".into(), relativize_path(meta_source, &args.output)),
                    ("from".into(), format),
                    ("output".into(), "${cache}/metadata_all.slab".into()),
                ];
                if args.base_fraction < 1.0 && !args.pedantic_dedup {
                    meta_opts.push(("fraction".into(), args.base_fraction.to_string()));
                }
                steps.push(Step {
                    id: "convert-metadata".into(),
                    run: "transform convert".into(),
                    description: Some("Import metadata from source format".into()),
                    after: vec![],
                    per_profile: false,
                    phase: 0,
                    finalize: false,
                    options: meta_opts,
                });
            } else if args.synthesize_metadata {
                // Synthesize metadata via generate metadata command
                let mut gen_meta_opts = vec![
                    ("output".into(), meta.metadata_all.path().into()),
                    ("count".into(), "${vector_count}".into()),
                    ("fields".into(), args.metadata_fields.to_string()),
                    ("range-min".into(), args.metadata_range_min.to_string()),
                    ("range-max".into(), args.metadata_range_max.to_string()),
                    ("seed".into(), format!("${{{}}}", "seed")),
                ];
                if args.synthesis_mode == "simple-int-eq" && args.synthesis_format != "slab" {
                    gen_meta_opts.push(("format".into(), args.synthesis_format.clone()));
                }
                steps.push(Step {
                    id: "generate-metadata".into(),
                    run: "generate metadata".into(),
                    description: Some("Synthesize random integer metadata".into()),
                    after: vec!["count-vectors".into()],
                    per_profile: false,
                    phase: 0,
                    finalize: false,
                    options: gen_meta_opts,
                });
            }
        }

        // extract-metadata: only when metadata_content has its own step
        // (i.e., step_id is "extract-metadata", not shared with metadata_all)
        if meta.metadata_content.step_id() == "extract-metadata" {
            let mut after = vec![];
            if meta.metadata_all.is_materialized() {
                after.push(meta.metadata_all.step_id().into());
            }
            if slots.shuffle.as_ref().map(|s| s.is_materialized()).unwrap_or(false) {
                after.push("generate-shuffle".into());
            }
            if args.base_fraction < 1.0 && !subset_applied {
                after.push("compute-base-end".into());
            }
            // Determine metadata output extension from the source format
            let meta_ext = std::path::Path::new(meta.metadata_all.path())
                .extension().and_then(|e| e.to_str()).unwrap_or("slab");

            steps.push(Step {
                id: "extract-metadata".into(),
                run: "transform extract".into(),
                description: Some("Reorder metadata to match shuffled base vectors".into()),
                after,
                per_profile: false,
                phase: 0,
                finalize: false,
                options: vec![
                    ("source".into(), meta.metadata_all.path().into()),
                    ("ivec-file".into(), "${cache}/shuffle.ivecs".into()),
                    ("output".into(), format!("{}metadata_content.{}", pp, meta_ext)),
                    ("range".into(), if args.base_fraction < 1.0 && !subset_applied {
                        "[${query_count},${base_end})".into()
                    } else if slots.prepare.is_materialized() {
                        "[${query_count},${clean_count})".into()
                    } else {
                        "[${query_count},${vector_count})".into()
                    }),
                ],
            });
        }

        // survey — skip for simple-int-eq (schema is known from config)
        let is_simple = args.synthesis_mode == "simple-int-eq" && args.synthesize_metadata;
        if meta.survey.is_materialized() {
            let survey_after = if meta.metadata_all.is_materialized() {
                vec![meta.metadata_all.step_id().into()]
            } else {
                vec![]
            };
            steps.push(Step {
                id: "survey-metadata".into(),
                run: "analyze survey".into(),
                description: Some("Survey metadata to discover schema and value ranges".into()),
                after: survey_after.clone(),
                per_profile: false,
                phase: 0,
                finalize: false,
                options: vec![
                    ("source".into(), meta.metadata_all.path().into()),
                    ("output".into(), "${cache}/metadata_survey.json".into()),
                    ("samples".into(), "10000".into()),
                    ("max-distinct".into(), "100".into()),
                ],
            });
        }

        // synthesize predicates
        let pred_after = if meta.survey.is_materialized() {
            vec!["survey-metadata".into()]
        } else if meta.metadata_all.is_materialized() {
            vec![meta.metadata_all.step_id().into()]
        } else {
            vec![]
        };
        let _pred_ext = if is_simple && args.synthesis_format != "slab" {
            &args.synthesis_format
        } else {
            "slab"
        };
        let mut pred_opts = vec![
            ("output".into(), meta.predicates.path().into()),
            ("count".into(), args.predicate_count.to_string()),
            ("seed".into(), format!("${{{}}}", "seed")),
        ];
        if is_simple {
            // Simple-int-eq: predicates are generated from configured ranges,
            // no survey needed. Pass range params directly.
            pred_opts.push(("mode".into(), "simple-int-eq".into()));
            pred_opts.push(("fields".into(), args.metadata_fields.to_string()));
            pred_opts.push(("range-min".into(), args.predicate_range_min.to_string()));
            pred_opts.push(("range-max".into(), args.predicate_range_max.to_string()));
            pred_opts.push(("format".into(), args.synthesis_format.clone()));
        } else {
            // Survey-based: pass survey file and strategy
            pred_opts.push(("source".into(), meta.metadata_all.path().into()));
            pred_opts.push(("survey".into(), "${cache}/metadata_survey.json".into()));
            pred_opts.push(("strategy".into(), args.predicate_strategy.clone()));
            pred_opts.push(("selectivity".into(), args.selectivity.to_string()));
        }
        steps.push(Step {
            id: "generate-predicates".into(),
            run: "generate predicates".into(),
            description: Some(if is_simple {
                "Generate simple integer equality predicates".into()
            } else {
                "Generate test predicates from metadata survey".into()
            }),
            after: pred_after,
            per_profile: false,
            phase: 0,
            finalize: false,
            options: pred_opts,
        });

        // compute predicates (per_profile)
        let mut eval_after = vec!["generate-predicates".into()];
        if meta.metadata_content.step_id() == "extract-metadata" {
            eval_after.push("extract-metadata".into());
        } else if meta.metadata_content.is_materialized() {
            eval_after.push(meta.metadata_content.step_id().into());
        }
        let mut eval_opts = vec![
            ("source".into(), meta.metadata_content.path().into()),
            ("predicates".into(), meta.predicates.path().into()),
        ];
        if !is_simple {
            eval_opts.push(("survey".into(), "${cache}/metadata_survey.json".into()));
        }
        eval_opts.push(("selectivity".into(), args.selectivity.to_string()));
        if is_simple {
            eval_opts.push(("mode".into(), "simple-int-eq".into()));
            eval_opts.push(("fields".into(), args.metadata_fields.to_string()));
        }
        eval_opts.push(("range".into(), if slots.base_count.is_some() {
            "[0,${base_end})".into()
        } else if slots.prepare.is_materialized() {
            "[0,${clean_count})".into()
        } else {
            "[0,${vector_count})".into()
        }));
        eval_opts.push(("output".into(), meta.predicate_indices.path().into()));
        steps.push(Step {
            id: "evaluate-predicates".into(),
            run: "compute evaluate-predicates".into(),
            description: None,
            after: eval_after,
            per_profile: true,
            phase: 1,
            finalize: false,
            options: eval_opts,
        });
    }

    // ── Ground truth chain ───────────────────────────────────────────
    //
    // When KNN is computed (Materialized), a single per_profile compute-knn
    // step handles both the default and all sized profiles.
    //
    // When KNN is pre-provided (Identity) AND sized profiles are requested,
    // the pre-provided GT covers the default profile only. Sized profiles
    // still need computed KNN — add a per_profile compute-knn step.
    // The default profile will redundantly recompute, but sized profiles
    // will get the KNN they need.
    let facets_check = resolve_facets(args);
    let (_, oracle_check) = parse_oracle_scope(&facets_check);
    let wants_partition_verify = args.partition_oracles || oracle_check.is_some();

    let needs_computed_knn = if let Some(ref knn) = slots.knn {
        knn.is_materialized() || args.sized_profiles.is_some()
    } else {
        false
    };

    if needs_computed_knn && slots.knn.is_some() {
        let has_queries = slots.query_vectors.is_some();
        if has_queries {
            let mut after = vec![];
            if slots.base_count.is_some() {
                after.push("count-base".into());
            } else if slots.base_vectors.is_materialized() {
                after.push(slots.base_vectors.step_id().into());
            }
            if slots.all_vectors.is_materialized() {
                after.push("convert-vectors".into());
            }
            if let Some(ref qv) = slots.query_vectors {
                if qv.is_materialized() {
                    after.push(qv.step_id().into());
                }
            }

            // No separate overlap removal step (SRD §20):
            // - Strategy 1 (combined B+Q): combined before dedup, shuffle guarantees disjointness
            // - Strategy 2 (HDF5): independent processing, no intermixing
            // - Strategy 3 (B only): shuffle guarantees disjointness
            let query_path = slots.query_vectors.as_ref().unwrap().path().to_string();

            steps.push(Step {
                id: "compute-knn".into(),
                run: cmd_knn(&args.personality).into(),
                description: Some("Compute brute-force exact KNN ground truth".into()),
                after,
                per_profile: true,
                phase: 0,
                finalize: false,
                options: {
                    let mut opts = vec![
                        ("base".into(), compute_knn_base_arg(slots, &working_vectors, subset_applied)),
                        ("query".into(), query_path.clone()),
                        ("indices".into(), "neighbor_indices.ivecs".into()),
                        ("distances".into(), "neighbor_distances.fvecs".into()),
                        ("neighbors".into(), args.neighbors.to_string()),
                        ("metric".into(), args.metric.clone()),
                    ];
                    if args.compress_cache {
                        opts.push(("compress_cache".into(), "true".into()));
                    }
                    opts.extend(knn_cosine_opts(args));
                    opts
                },
            });
        }
    }

    // Per-partition KNN computation — only emitted when partitions are
    // enabled AND base GT is pre-provided (Identity). When KNN is already
    // Materialized (compute-knn template exists), partitions use that same
    // template via Phase 3 expansion. This template has "partition" in the
    // ID so it only expands for partition profiles, not default/sized.
    if wants_partition_verify && !needs_computed_knn && slots.knn.is_some() {
        let has_queries = slots.query_vectors.is_some();
        if has_queries {
            steps.push(Step {
                id: "compute-knn-partition".into(),
                run: cmd_knn(&args.personality).into(),
                description: Some("Compute KNN for oracle partition profiles".into()),
                after: vec!["partition-profiles".into(), "verify-knn".into()],
                per_profile: true,
                phase: 0,
                finalize: false,
                options: {
                    let mut opts = vec![
                        ("base".into(), "${profile_dir}base_vectors.fvecs".into()),
                        ("query".into(), "${profile_dir}query_vectors.fvecs".into()),
                        ("indices".into(), "neighbor_indices.ivecs".into()),
                        ("distances".into(), "neighbor_distances.fvecs".into()),
                        ("neighbors".into(), args.neighbors.to_string()),
                        ("metric".into(), args.metric.clone()),
                    ];
                    opts.extend(knn_cosine_opts(args));
                    opts
                },
            });
        }
    }

    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            let simple = args.synthesis_mode == "simple-int-eq" && args.synthesize_metadata;
            let mut after = if simple {
                vec!["verify-predicates-sqlite".into()]
            } else {
                vec!["evaluate-predicates".into()]
            };
            if slots.base_count.is_some() {
                after.push("count-base".into());
            }
            if let Some(ref qv) = slots.query_vectors {
                if qv.is_materialized() {
                    after.push(qv.step_id().into());
                }
            }
            steps.push(Step {
                id: "compute-filtered-knn".into(),
                run: "compute filtered-knn".into(),
                description: Some("Compute filtered KNN with predicate pre-filtering".into()),
                after,
                per_profile: true,
                phase: 2, // after all evaluate-predicates — base + pred indices I/O phase
                finalize: false,
                options: {
                    let mut opts = vec![
                        ("base".into(), compute_knn_base_arg(slots, &working_vectors, subset_applied)),
                        ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                        ("metadata-indices".into(), slots.metadata.as_ref().unwrap().predicate_indices.path().into()),
                        ("indices".into(), "filtered_neighbor_indices.ivecs".into()),
                        ("distances".into(), "filtered_neighbor_distances.fvecs".into()),
                        ("neighbors".into(), args.neighbors.to_string()),
                        ("metric".into(), args.metric.clone()),
                    ];
                    if args.compress_cache {
                        opts.push(("compress_cache".into(), "true".into()));
                    }
                    opts
                },
            });
        }
    }

    // ── Consolidated verification ─────────────────────────────────
    // Single-pass verifiers that scan base vectors once and verify all
    // profiles incrementally. NOT per-profile — runs once after all
    // compute phases complete.
    if slots.knn.is_some() {
        let mut verify_after = Vec::new();
        if needs_computed_knn {
            verify_after.push("compute-knn".into());
        } else {
            if slots.all_vectors.is_materialized() {
                verify_after.push("convert-vectors".into());
            }
            if let Some(ref qv) = slots.query_vectors {
                if qv.is_materialized() {
                    verify_after.push(qv.step_id().into());
                }
            }
        }
        // When ground truth is pre-provided (Identity), use the source
        // base vectors for verification — the GT ordinals reference the
        // original ordering, not the post-extraction/shuffle ordering.
        // Otherwise mirror compute-knn's `base:` exactly (subset-aware).
        // Mismatch here was the cause of false verify failures when
        // --base-fraction subsetted the source but verify still pointed
        // at the unsubsetted file.
        let verify_base = if !needs_computed_knn {
            slots.all_vectors.path().to_string()
        } else {
            compute_knn_base_arg(slots, &working_vectors, subset_applied)
        };
        let verify_query = if !needs_computed_knn && args.query_vectors.is_some() {
            // Use original query source when GT is pre-provided
            slots.query_vectors.as_ref()
                .map(|q| if q.is_materialized() { q.path().to_string() } else { q.path().to_string() })
                .unwrap_or_else(|| "query_vectors.fvecs".to_string())
        } else {
            slots.query_vectors.as_ref().unwrap().path().to_string()
        };

        steps.push(Step {
            id: "verify-knn".into(),
            run: cmd_verify_knn(&args.personality).into(),
            description: Some("Multi-threaded single-pass KNN verification across all profiles".into()),
            after: verify_after,
            per_profile: false, // NOT per-profile — one step verifies all
            phase: 0,
            finalize: false,
            options: {
                let mut opts = vec![
                    ("base".into(), verify_base),
                    ("query".into(), verify_query),
                    ("metric".into(), args.metric.clone()),
                    ("sample".into(), if args.verify_knn_sample == 0 {
                        "0".into()
                    } else {
                        args.verify_knn_sample.to_string()
                    }),
                    ("seed".into(), format!("${{{}}}", "seed")),
                    ("output".into(), "${cache}/verify_knn_consolidated.json".into()),
                ];
                // verify-knn-consolidated now shares the sgemm kernel
                // with knn-blas; both need the same cosine-mode flags
                // to agree on arithmetic.
                opts.extend(knn_cosine_opts(args));
                opts
            },
        });
    }

    // Per-partition KNN verification — runs after Phase 3 compute-knn
    // for each partition profile. Uses the per-profile groundtruth verifier
    // (not consolidated). Only expands for partition profiles via oracle
    // scope filtering (facet G).
    if wants_partition_verify && slots.knn.is_some() {
        steps.push(Step {
            id: "verify-knn-partition".into(),
            run: "verify knn-groundtruth".into(),
            description: Some("Verify partition KNN against brute-force recomputation".into()),
            after: if needs_computed_knn {
                vec!["compute-knn".into()]
            } else {
                vec!["compute-knn-partition".into()]
            },
            per_profile: true,
            phase: 1, // verify phase — runs after all phase 0 compute-knn
            finalize: false,
            options: vec![
                ("base".into(), "${profile_dir}base_vectors.fvecs".into()),
                ("query".into(), "${profile_dir}query_vectors.fvecs".into()),
                ("indices".into(), "${profile_dir}neighbor_indices.ivecs".into()),
                ("distances".into(), "${profile_dir}neighbor_distances.fvecs".into()),
                ("metric".into(), args.metric.clone()),
                ("neighbors".into(), args.neighbors.to_string()),
                ("sample".into(), if args.verify_knn_sample == 0 {
                    "50".into() // default sample for partition verify
                } else {
                    args.verify_knn_sample.to_string()
                }),
                ("seed".into(), format!("${{{}}}", "seed")),
                ("output".into(), "${cache}/${profile_name}_verify_knn_partition.json".into()),
            ],
        });
    }

    let is_simple_synth = args.synthesis_mode == "simple-int-eq" && args.synthesize_metadata;
    // ── Predicate verification ───────────────────────────────────────
    // verify-predicates-sqlite: SQLite oracle verification of R
    // Runs after evaluate-predicates, before compute-filtered-knn
    if is_simple_synth {
        if let Some(ref meta) = slots.metadata {
            if meta.predicate_indices.is_materialized() {
                steps.push(Step {
                    id: "verify-predicates-sqlite".into(),
                    run: "verify predicates-sqlite".into(),
                    description: Some("Verify predicate evaluations using SQLite oracle".into()),
                    after: vec!["evaluate-predicates".into()],
                    per_profile: true,
                    phase: 1,
                    finalize: false,
                    options: vec![
                        ("metadata".into(), meta.metadata_content.path().into()),
                        ("predicates".into(), meta.predicates.path().into()),
                        ("results".into(), meta.predicate_indices.path().into()),
                        ("fields".into(), args.metadata_fields.to_string()),
                        ("output".into(), "${cache}/${profile_name}_verify_predicates_sqlite.json".into()),
                    ],
                });
            }
        }
    } else if let Some(ref meta) = slots.metadata {
        // Slab mode: consolidated predicate verification
        if meta.predicate_indices.is_materialized() {
            steps.push(Step {
                id: "verify-predicates".into(),
                run: "verify predicates-consolidated".into(),
                description: Some("Verify predicate evaluations across all profiles".into()),
                after: vec!["evaluate-predicates".into()],
                per_profile: false,
                phase: 0,
                finalize: false,
                options: vec![
                    ("metadata".into(), meta.metadata_content.path().into()),
                    ("predicates".into(), meta.predicates.path().into()),
                    ("sample".into(), "50".into()),
                    ("metadata-sample".into(), "100000".into()),
                    ("seed".into(), format!("${{{}}}", "seed")),
                    ("output".into(), "${cache}/verify_predicates.json".into()),
                ],
            });
        }
    }

    // ── Filtered KNN + verification ─────────────────────────────────
    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            let meta = slots.metadata.as_ref().unwrap();
            // verify-filtered-knn: brute-force re-computation of filtered KNN
            steps.push(Step {
                id: "verify-filtered-knn".into(),
                run: "verify filtered-knn-consolidated".into(),
                description: Some("Verify filtered KNN results via brute-force recomputation".into()),
                after: vec!["compute-filtered-knn".into()],
                per_profile: false,
                phase: 0,
                finalize: false,
                options: vec![
                    ("base".into(), slots.base_vectors.path().into()),
                    ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                    ("metadata".into(), meta.metadata_content.path().into()),
                    ("predicates".into(), meta.predicates.path().into()),
                    ("metric".into(), args.metric.clone()),
                    ("normalized".into(), if args.normalize { "true" } else { "false" }.into()),
                    ("sample".into(), "50".into()),
                    ("seed".into(), format!("${{{}}}", "seed")),
                    ("output".into(), "${cache}/verify_filtered_knn.json".into()),
                ],
            });
        }
    }

    // ── dataset.json ──────────────────────────────────────────────
    // Generate a JSON copy of dataset.yaml for clients that prefer JSON.
    // Runs before merkle so the JSON file gets merkle coverage.
    let mut json_after = Vec::new();
    if let Some(ref knn) = slots.knn {
        if knn.is_materialized() || needs_computed_knn {
            json_after.push("verify-knn".into());
        }
    }
    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            json_after.push("verify-filtered-knn".into());
        }
    }
    if is_simple_synth && slots.metadata.is_some() {
        json_after.push("verify-predicates-sqlite".into());
    }
    // ── Oracle partition profiles ────────────────────────────────
    // O facet carries sub-facets: +O means Obqg (BQG per partition),
    // +Obqgmprf means full MPRF within each partition too.
    let facets = resolve_facets(args);
    let (_, oracle_scope) = parse_oracle_scope(&facets);
    let wants_oracles = args.partition_oracles || oracle_scope.is_some();
    if wants_oracles && slots.metadata.is_some() {
        // Partition-profiles depends on whatever produced the metadata labels.
        // If filtered KNN was computed, wait for its verification first.
        // Otherwise, just wait for metadata to be ready.
        let partition_after = if slots.filtered_knn.as_ref().map(|f| f.is_materialized()).unwrap_or(false) {
            vec!["verify-filtered-knn".into()]
        } else if let Some(ref meta) = slots.metadata {
            if meta.metadata_content.is_materialized() {
                vec![meta.metadata_content.step_id().into()]
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        steps.push(Step {
            id: "partition-profiles".into(),
            run: "compute partition-profiles".into(),
            description: Some("Create per-label oracle partition profiles".into()),
            after: partition_after,
            per_profile: false,
            phase: 0,
            finalize: false,
            options: vec![
                ("base".into(), slots.base_vectors.path().into()),
                ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                ("metadata".into(), slots.metadata.as_ref().unwrap().metadata_content.path().into()),
                ("predicates".into(), slots.metadata.as_ref().unwrap().predicates.path().into()),
                ("neighbors".into(), args.neighbors.to_string()),
                ("metric".into(), args.metric.clone()),
                ("allowed-partitions".into(), args.max_partitions.to_string()),
                ("on-undersized".into(), args.on_undersized.clone()),
            ],
        });
        json_after.push("partition-profiles".into());
    }

    if json_after.is_empty() {
        if let Some(last) = steps.last() {
            json_after.push(last.id.clone());
        }
    }
    steps.push(Step {
        id: "generate-dataset-json".into(),
        run: "generate dataset-json".into(),
        description: Some("Generate dataset.json from dataset.yaml".into()),
        after: json_after,
        per_profile: false,
        phase: 0,
        finalize: true,
        options: vec![],
    });

    // ── JSON copies of YAML artifacts ─────────────────────────
    steps.push(Step {
        id: "generate-variables-json".into(),
        run: "generate variables-json".into(),
        description: Some("Generate variables.json from variables.yaml".into()),
        after: vec!["generate-dataset-json".into()],
        per_profile: false,
        phase: 0,
        finalize: true,
        options: vec![],
    });
    steps.push(Step {
        id: "generate-dataset-log-jsonl".into(),
        run: "generate dataset-log-jsonl".into(),
        description: Some("Generate dataset.jsonl from dataset.log".into()),
        after: vec!["generate-dataset-json".into()],
        per_profile: false,
        phase: 0,
        finalize: true,
        options: vec![],
    });

    // ── Documentation generation ───────────────────────────────────
    // Produces docs/dataset.md (and optionally docs/exemplars.md).
    // Runs after dataset.yaml is finalized but before catalog/merkle
    // so docs/ files get catalog and merkle coverage.
    steps.push(Step {
        id: "generate-docs".into(),
        run: "analyze describe-dataset".into(),
        description: Some("Generate dataset documentation in docs/".into()),
        after: vec!["generate-variables-json".into(), "generate-dataset-log-jsonl".into()],
        per_profile: false,
        phase: 0,
        finalize: true,
        options: vec![
            ("exemplars".into(), "true".into()),
        ],
    });

    // ── Catalog generation ──────────────────────────────────────────
    // Produces catalog.json, catalog.yaml, and knn_entries.yaml.
    // Runs before merkle so knn_entries.yaml gets .mref coverage.
    steps.push(Step {
        id: "generate-catalog".into(),
        run: "catalog generate".into(),
        description: Some("Generate catalog index for the dataset directory".into()),
        after: vec!["generate-docs".into()],
        per_profile: false,
        phase: 0,
        finalize: true,
        options: vec![
            ("source".into(), ".".into()),
        ],
    });

    // ── Merkle hash trees ────────────────────────────────────────
    // Runs AFTER catalog so knn_entries.yaml is covered. Catalog
    // does not read .mref files — it reads dataset.yaml only.
    steps.push(Step {
        id: "generate-merkle".into(),
        run: "merkle create".into(),
        description: Some("Create merkle hash trees for all publishable data files".into()),
        after: vec!["generate-catalog".into()],
        per_profile: false,
        phase: 0,
        finalize: true,
        options: vec![
            ("source".into(), ".".into()),
            ("min-size".into(), "0".into()),
        ],
    });

    steps
}

// ---------------------------------------------------------------------------
// Profile view assembly
// ---------------------------------------------------------------------------

/// Assemble profile views from resolved slot paths.
fn profile_views(slots: &PipelineSlots, args: &ImportArgs, _output_dir: &std::path::Path) -> Vec<(String, String)> {
    let mut views = Vec::new();

    // Base vectors: when Identity (source used as-is), the canonical path is
    // the symlink in profiles/base/. When Materialized, the step output path
    // already includes the profile prefix.
    match &slots.base_vectors {
        Artifact::Identity { path } => {
            let ext = std::path::Path::new(path).extension()
                .and_then(|e| e.to_str())
                .and_then(crate::formats::VecFormat::canonical_extension)
                .unwrap_or("fvec");
            views.push(("base_vectors".into(),
                format!("{}base_vectors.{}", args.profile_prefix(), ext)));
        }
        Artifact::Materialized { .. } => {
            views.push(("base_vectors".into(), slots.base_vectors.path().into()));
        }
    }

    if let Some(ref qv) = slots.query_vectors {
        match qv {
            Artifact::Identity { path } => {
                let ext = std::path::Path::new(path).extension()
                    .and_then(|e| e.to_str())
                    .and_then(crate::formats::VecFormat::canonical_extension)
                    .unwrap_or("fvec");
                views.push(("query_vectors".into(),
                    format!("{}query_vectors.{}", args.profile_prefix(), ext)));
            }
            Artifact::Materialized { .. } => {
                views.push(("query_vectors".into(), qv.path().into()));
            }
        }
    }

    if let Some(ref meta) = slots.metadata {
        views.push(("metadata_content".into(), meta.metadata_content.path().into()));
        views.push(("metadata_predicates".into(), meta.predicates.path().into()));
    }

    if let Some(ref knn) = slots.knn {
        match knn {
            Artifact::Identity { path } => {
                let ext = std::path::Path::new(path).extension()
                    .and_then(|e| e.to_str())
                    .and_then(crate::formats::VecFormat::canonical_extension)
                    .unwrap_or("ivec");
                views.push(("neighbor_indices".into(),
                    format!("{}neighbor_indices.{}", args.profile_prefix(), ext)));
                // Include pre-provided distances if available
                if let Some(ref gt_dist) = args.ground_truth_distances {
                    let dext = gt_dist.extension()
                        .and_then(|e| e.to_str())
                        .and_then(crate::formats::VecFormat::canonical_extension)
                        .unwrap_or("fvec");
                    views.push(("neighbor_distances".into(),
                        format!("{}neighbor_distances.{}", args.profile_prefix(), dext)));
                }
            }
            Artifact::Materialized { .. } => {
                views.push(("neighbor_indices".into(), format!("{}neighbor_indices.ivecs", args.default_prefix())));
                views.push(("neighbor_distances".into(), format!("{}neighbor_distances.fvecs", args.default_prefix())));
            }
        }
    }

    if let Some(ref meta) = slots.metadata {
        views.push(("metadata_indices".into(),
            format!("{}{}", args.default_prefix(), meta.predicate_indices.path())));
    }

    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            views.push(("filtered_neighbor_indices".into(), format!("{}filtered_neighbor_indices.ivecs", args.default_prefix())));
            views.push(("filtered_neighbor_distances".into(), format!("{}filtered_neighbor_distances.fvecs", args.default_prefix())));
        }
    }

    views
}

// ---------------------------------------------------------------------------
// YAML generation
// ---------------------------------------------------------------------------

/// Produce a human-readable description for a sized profile spec.
fn describe_sized_spec(spec: &str) -> String {
    let spec = spec.trim();
    if let Some(rest) = spec.strip_prefix("mul:") {
        if let Some((range, factor)) = rest.split_once('/') {
            if range.contains("..") {
                let (start, end) = range.split_once("..").unwrap();
                format!("geometric series from {} to {}, multiplying by {} each step", start.trim(), end.trim(), factor.trim())
            } else {
                format!("geometric series from {}, multiplying by {} each step, up to dataset size", range.trim(), factor.trim())
            }
        } else {
            "geometric series".into()
        }
    } else if let Some(rest) = spec.strip_prefix("fib:") {
        if let Some((start, end)) = rest.split_once("..") {
            format!("fibonacci series from {} to {}", start.trim(), end.trim())
        } else {
            "fibonacci series".into()
        }
    } else if let Some((range, step)) = spec.split_once('/') {
        if let Some((start, end)) = range.split_once("..") {
            let step = step.trim();
            if step.bytes().any(|b| b.is_ascii_alphabetic()) {
                format!("linear series from {} to {}, stepping by {}", start.trim(), end.trim(), step)
            } else {
                format!("linear series from {} to {}, {} equal divisions", start.trim(), end.trim(), step)
            }
        } else {
            format!("profile at {}", spec)
        }
    } else {
        format!("single profile at {}", spec)
    }
}

fn generate_yaml(
    args: &ImportArgs,
    steps: &[Step],
    views: &[(String, String)],
    _slots: &PipelineSlots,
) -> String {
    let mut out = String::new();

    out.push_str("# Copyright (c) Jonathan Shook\n");
    out.push_str("# SPDX-License-Identifier: Apache-2.0\n\n");

    out.push_str(&format!("name: {}\n", args.name));
    if let Some(ref desc) = args.description {
        out.push_str(&format!("description: >-\n  {}\n", desc));
    }

    // Attributes
    out.push_str("\nattributes:\n");
    if args.personality != "native" {
        out.push_str(&format!("  personality: {}\n", args.personality));
    }
    out.push_str(&format!("  distance_function: {}\n", args.metric));
    out.push_str(&format!("  veks_version: {}\n", env!("CARGO_PKG_VERSION")));
    out.push_str(&format!("  veks_build: {}.{}\n", env!("VEKS_BUILD_HASH"), env!("VEKS_BUILD_NUMBER")));
    // Record oracle partition scope if O facet is active
    let facets_for_scope = resolve_facets(args);
    let (_, oracle_scope_opt) = parse_oracle_scope(&facets_for_scope);
    if let Some(ref scope) = oracle_scope_opt {
        out.push_str(&format!("  oracle_scope: {}\n", scope));
    }
    // Bake user-asserted counts directly into the attributes so they
    // survive any subsequent `veks run --reset` that wipes
    // `variables.yaml`. Without this, a reset cycle would drop the
    // zero/duplicate seeds, the pipeline would produce no scan steps
    // (bootstrap skipped them based on the same assertion), and
    // `update_dataset_attributes` would have nothing to sync — so
    // `veks check dataset-attributes` would go red again despite the
    // wizard answer being accurate.
    if let Some(z) = args.zero_count {
        out.push_str(&format!("  is_zero_vector_free: {}\n", z == 0));
    }
    if let Some(d) = args.duplicate_count {
        out.push_str(&format!("  is_duplicate_vector_free: {}\n", d == 0));
    }
    // is_normalized, is_duplicate_vector_free, and is_zero_vector_free
    // are set by the pipeline after the relevant steps complete — not
    // at generation time when correctness hasn't been verified yet.
    // Exception: when the user asserted the counts up front (above),
    // we can record the derived boolean immediately.

    // Upstream pipeline
    if !steps.is_empty() {
        out.push_str("\nupstream:\n");
        out.push_str("  defaults:\n");
        out.push_str(&format!("    query_count: {}\n", args.query_count));
        out.push_str(&format!("    seed: {}\n", args.seed));
        out.push_str("\n  steps:\n");

        for step in steps {
            out.push_str(&format!("    - id: {}\n", step.id));
            if let Some(ref desc) = step.description {
                out.push_str(&format!("      description: {}\n", desc));
            }
            out.push_str(&format!("      run: {}\n", step.run));
            if step.per_profile {
                out.push_str("      per_profile: true\n");
            }
            if step.phase > 0 {
                out.push_str(&format!("      phase: {}\n", step.phase));
            }
            if step.finalize {
                out.push_str("      finalize: true\n");
            }
            if !step.after.is_empty() {
                out.push_str(&format!("      after: [{}]\n", step.after.join(", ")));
            }
            for (k, v) in &step.options {
                // Quote values that contain YAML-special characters
                if v.contains('[') || v.contains('{') || v.contains('$')
                    || v.contains(':') || v.contains('#') || v.contains('&')
                {
                    out.push_str(&format!("      {}: \"{}\"\n", k, v));
                } else {
                    out.push_str(&format!("      {}: {}\n", k, v));
                }
            }
            out.push_str("\n");
        }
    }

    // strata — root-level sized-profile generator specs (the
    // `mul:/fib:/linear:/N..M/K` grammar). Moved out from under
    // `profiles:` so consumers reading the profile map never have
    // to special-case a non-profile key. Per-profile view paths
    // are derived at expand time from the per_profile pipeline
    // templates (`derive_views_from_templates`), so we no longer
    // need the `facets:` sub-map here either.
    if let Some(spec) = args.sized_profiles.as_ref() {
        let specs: Vec<&str> = spec.split(',').map(|s| s.trim()).collect();
        // Human-readable header comments explaining each expression,
        // then the flat sequence `serde` expects for `Vec<String>`.
        out.push_str("\n");
        for s in &specs {
            let desc = describe_sized_spec(s);
            out.push_str(&format!("# strata \"{}\": {}\n", s, desc));
        }
        let formatted: Vec<String> = specs.iter().map(|s| format!("\"{}\"", s)).collect();
        out.push_str(&format!("strata: [{}]\n", formatted.join(", ")));
    }

    // Profiles
    out.push_str("\nprofiles:\n");
    out.push_str("  default:\n");
    out.push_str(&format!("    maxk: {}\n", args.neighbors));
    for (facet, path) in views {
        out.push_str(&format!("    {}: {}\n", facet, path));
    }

    out
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the `datasets import` subcommand.
pub fn run(mut args: ImportArgs) {
    // Apply --provided-facets masking: null out inputs for facets not
    // in the provided set so the pipeline generates compute steps for them.
    if let Some(ref provided) = args.provided_facets {
        let p = parse_facet_spec(provided);
        if !p.contains('B') { args.base_vectors = None; }
        if !p.contains('Q') { args.query_vectors = None; }
        if !p.contains('G') { args.ground_truth = None; }
        if !p.contains('D') { args.ground_truth_distances = None; }
        if !p.contains('M') { args.metadata = None; }

        // Validate that the source data actually provides the declared facets.
        let mut missing = Vec::new();
        if p.contains('B') && args.base_vectors.is_none() { missing.push("B (base vectors)"); }
        if p.contains('Q') && args.query_vectors.is_none() { missing.push("Q (query vectors)"); }
        if p.contains('G') && args.ground_truth.is_none() { missing.push("G (ground truth)"); }
        if p.contains('D') && args.ground_truth_distances.is_none() { missing.push("D (distances)"); }
        if p.contains('M') && args.metadata.is_none() { missing.push("M (metadata)"); }
        if !missing.is_empty() {
            eprintln!("Error: --provided-facets declares {} but no matching inputs were found:",
                provided);
            for m in &missing {
                eprintln!("  - {}", m);
            }
            eprintln!("Provide the missing inputs via CLI flags (--base-vectors, --query-vectors, etc.)");
            eprintln!("or remove the unmatched facets from --provided-facets.");
            std::process::exit(1);
        }
    }

    let output_dir = &args.output;

    if output_dir.exists() && !args.force {
        if output_dir.join("dataset.yaml").exists() {
            eprintln!(
                "Error: {} already contains a dataset.yaml. Use --force to overwrite.",
                crate::check::rel_display(output_dir)
            );
            std::process::exit(1);
        }
    }

    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error: failed to create directory {}: {}", crate::check::rel_display(output_dir), e);
        std::process::exit(1);
    }

    // Detect normalization status of base vectors
    if let Some(ref base_path) = args.base_vectors {
        if let Some((is_normalized, sample_count, mean_norm)) = detect_normalized(base_path) {
            if is_normalized {
                println!("Vectors detected as L2-normalized (mean norm={:.4}, n={})", mean_norm, sample_count);
                println!("  Normalization during extraction will be a no-op for these vectors.");
            } else {
                println!("Vectors are NOT L2-normalized (mean norm={:.4}, n={})", mean_norm, sample_count);
                println!("  Normalization will be applied during extraction.");
            }
            println!();
        }
    }

    // Resolve all slots
    let slots = resolve_slots(&args);

    // Report what was resolved
    println!("Resolving pipeline slots:");
    print_slot("all_vectors", &slots.all_vectors);
    print_slot("prepare", &slots.prepare);
    print_slot("vector_count", &slots.vector_count);
    if slots.self_search {
        println!("  mode: self-search (query_count={})", args.query_count);
    }
    if let Some(ref s) = slots.shuffle { print_slot("shuffle", s); }
    if let Some(ref qv) = slots.query_vectors { print_slot("query_vectors", qv); }
    print_slot("base_vectors", &slots.base_vectors);
    if let Some(ref bc) = slots.base_count { print_slot("base_count", bc); }
    if let Some(ref meta) = slots.metadata {
        print_slot("metadata_all", &meta.metadata_all);
        print_slot("metadata_content", &meta.metadata_content);
        print_slot("survey", &meta.survey);
        print_slot("predicates", &meta.predicates);
        print_slot("predicate_indices", &meta.predicate_indices);
    } else {
        println!("  metadata: Absent");
    }
    if let Some(ref knn) = slots.knn { print_slot("knn", knn); } else { println!("  knn: Absent"); }
    if let Some(ref fknn) = slots.filtered_knn { print_slot("filtered_knn", fknn); } else { println!("  filtered_knn: Absent"); }

    // Emit steps
    let steps = emit_steps(&slots, &args, &args.output);
    let views = profile_views(&slots, &args, output_dir);

    println!();
    println!("{} pipeline step(s) to emit ({} identity-collapsed)",
        steps.len(),
        count_identity(&slots),
    );

    // Generate YAML
    let yaml = generate_yaml(&args, &steps, &views, &slots);

    let dataset_path = output_dir.join("dataset.yaml");
    if let Err(e) = std::fs::write(&dataset_path, &yaml) {
        eprintln!("Error: failed to write {}: {}", crate::check::rel_display(&dataset_path), e);
        std::process::exit(1);
    }

    // Seed variables.yaml with user-asserted counts when the wizard
    // (or the CLI) supplied them. This is the "I know this source is
    // already clean — don't scan 1.5 TB just to confirm zero_count=0"
    // shortcut that also stamps the `is_*_free` attributes during the
    // subsequent `veks run` via `update_dataset_attributes`, skipping
    // the standalone `scan-zeros` / `scan-duplicates` pipeline steps.
    if args.zero_count.is_some() || args.duplicate_count.is_some() {
        let vars_path = output_dir.join("variables.yaml");
        let mut vars: indexmap::IndexMap<String, String> =
            if vars_path.exists() {
                std::fs::read_to_string(&vars_path).ok()
                    .and_then(|s| serde_yaml::from_str::<std::collections::BTreeMap<String, String>>(&s).ok())
                    .map(|m| m.into_iter().collect())
                    .unwrap_or_default()
            } else {
                Default::default()
            };
        if let Some(z) = args.zero_count {
            vars.insert("zero_count".to_string(), z.to_string());
            vars.insert("source_zero_count".to_string(), z.to_string());
        }
        if let Some(d) = args.duplicate_count {
            vars.insert("duplicate_count".to_string(), d.to_string());
        }
        // Sorted for deterministic output (matches the runtime writer).
        let sorted: std::collections::BTreeMap<_, _> =
            vars.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        if let Ok(content) = serde_yaml::to_string(&sorted) {
            if let Err(e) = std::fs::write(&vars_path, content) {
                eprintln!("Warning: failed to seed {}: {}", vars_path.display(), e);
            }
        }
    }

    // Create .gitignore
    let gitignore_path = output_dir.join(".gitignore");
    if !gitignore_path.exists() {
        let _ = std::fs::write(&gitignore_path, ".scratch/\n.cache/\n");
    }

    // Create symlinks for Identity artifacts in the canonical profile structure.
    // When a source file is already in native format (no import/extract needed),
    // the profile view points to profiles/base/<facet>.<ext> and a symlink is
    // created there pointing to the actual source file.
    create_identity_symlinks(output_dir, &args, &slots);

    // Write import provenance to dataset.log
    write_import_log(output_dir, &args, &slots, steps.len());

    println!();
    println!("Created {}", crate::check::rel_display(&dataset_path));
    if !steps.is_empty() {
        // Show a concise run command — just `veks run` if the output dir
        // is the current working directory, since veks auto-detects dataset.yaml.
        let is_cwd = output_dir == Path::new(".") || output_dir == Path::new("");
        println!();
        println!("To prepare the dataset, run:");
        if is_cwd {
            println!("  veks run");
        } else {
            println!("  veks run {}", crate::check::rel_display(&dataset_path));
        }
    }

}

// ---------------------------------------------------------------------------
// Identity symlinks for canonical profile layout
// ---------------------------------------------------------------------------

/// Create symlinks in `profiles/base/` for source files that are used as-is
/// (Identity artifacts). Only creates symlinks for Identity artifacts —
/// Materialized artifacts will be created by pipeline steps and must NOT
/// have symlinks that point to source data (writes through symlinks
/// would destroy the original data).
fn create_identity_symlinks(output_dir: &std::path::Path, args: &ImportArgs, slots: &PipelineSlots) {
    // In classic mode, symlinks go directly in the dataset root
    let base_dir = if args.classic {
        output_dir.to_path_buf()
    } else {
        output_dir.join("profiles/base")
    };
    if let Err(e) = std::fs::create_dir_all(&base_dir) {
        eprintln!("Warning: failed to create {}: {}", base_dir.display(), e);
        return;
    }

    let self_search = args.query_vectors.is_none()
        && (args.self_search || args.base_vectors.is_some());

    // Base vectors: symlink ONLY when Identity (not Materialized).
    // When Materialized, the extract-base step will create the file.
    if !self_search && !slots.base_vectors.is_materialized() {
        if let Some(ref base_path) = args.base_vectors {
            if is_native_xvec_file(base_path) {
                let ext = base_path.extension()
                    .and_then(|e| e.to_str())
                    .and_then(VecFormat::canonical_extension)
                    .unwrap_or("fvec");
                let link = base_dir.join(format!("base_vectors.{}", ext));
                create_symlink(base_path, &link);
            }
        }
    }

    // Query vectors: when Identity (used as-is), create a query_vectors symlink.
    // When Materialized, the processing step creates the final file and a
    // query_vectors_raw symlink is created for the overlap step's input.
    if let Some(ref qv) = slots.query_vectors {
        if let Some(ref query_path) = args.query_vectors {
            if !slots.self_search && is_native_xvec_file(query_path) {
                let ext = query_path.extension()
                    .and_then(|e| e.to_str())
                    .and_then(VecFormat::canonical_extension)
                    .unwrap_or("fvec");
                if qv.is_materialized() {
                    // Overlap/convert step reads from _raw
                    let link = base_dir.join(format!("query_vectors_raw.{}", ext));
                    create_symlink(query_path, &link);
                } else {
                    // Identity: final query_vectors symlink
                    let link = base_dir.join(format!("query_vectors.{}", ext));
                    create_symlink(query_path, &link);
                }
            }
        }
    }

    // Ground truth: symlink when Identity and native xvec
    if let Some(ref knn) = slots.knn {
        if !knn.is_materialized() {
            if let Some(ref gt_path) = args.ground_truth {
                if is_native_xvec_file(gt_path) {
                    let ext = gt_path.extension()
                        .and_then(|e| e.to_str())
                        .and_then(VecFormat::canonical_extension)
                        .unwrap_or("ivec");
                    let link = base_dir.join(format!("neighbor_indices.{}", ext));
                    create_symlink(gt_path, &link);
                }
            }
            if let Some(ref gt_dist) = args.ground_truth_distances {
                if is_native_xvec_file(gt_dist) {
                    let ext = gt_dist.extension()
                        .and_then(|e| e.to_str())
                        .and_then(VecFormat::canonical_extension)
                        .unwrap_or("fvec");
                    let link = base_dir.join(format!("neighbor_distances.{}", ext));
                    create_symlink(gt_dist, &link);
                }
            }
        }
    }

    // Metadata content: symlink when native slab and not self-search
    if !self_search {
        if let Some(ref meta_path) = args.metadata {
            if is_native_slab_file(meta_path) {
                let link = base_dir.join("metadata_content.slab");
                create_symlink(meta_path, &link);
            }
        }
    }
}

/// Create a symlink with a relative target, removing any existing one first.
///
/// The symlink target is computed as a relative path from the link's parent
/// directory to the actual target file. This keeps datasets portable — they
/// can be moved or shared without breaking internal references.
pub(crate) fn create_symlink(target: &std::path::Path, link: &std::path::Path) {
    // Compute relative path from the link's parent directory to the target.
    let link_dir = link.parent().unwrap_or(std::path::Path::new("."));
    let rel_target = relative_path(link_dir, target);

    // Remove existing symlink or file at the link location
    if link.exists() || link.symlink_metadata().is_ok() {
        let _ = std::fs::remove_file(link);
    }

    match std::os::unix::fs::symlink(&rel_target, link) {
        Ok(()) => {
            println!("  Symlinked {} → {}", crate::check::rel_display(link), rel_target.display());
        }
        Err(e) => {
            eprintln!("  Warning: failed to create symlink {} → {}: {}",
                crate::check::rel_display(link), rel_target.display(), e);
        }
    }
}

/// Compute a relative path from `base` directory to `target` path.
///
/// Uses `std::path::Component` iteration rather than `canonicalize()` to
/// avoid requiring the paths to exist and to avoid producing absolute paths.
pub(crate) fn relative_path(base: &std::path::Path, target: &std::path::Path) -> std::path::PathBuf {
    use std::path::{Component, PathBuf};

    // Normalize both paths to absolute for comparison
    let abs_base = if base.is_absolute() {
        base.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(base)
    };
    let abs_target = if target.is_absolute() {
        target.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(target)
    };

    // Collect normalized components (resolve . and ..)
    let base_parts: Vec<_> = abs_base.components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect();
    let target_parts: Vec<_> = abs_target.components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect();

    // Find common prefix length
    let common = base_parts.iter().zip(target_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();

    // Build relative path: go up from base, then down to target
    let mut result = PathBuf::new();
    for _ in common..base_parts.len() {
        result.push("..");
    }
    for part in &target_parts[common..] {
        result.push(part);
    }

    if result.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// Import provenance log
// ---------------------------------------------------------------------------

/// Write import provenance to `dataset.log`.
///
/// Records the import inputs, detected starting scenario, and the dataflow
/// graph entry points so that `dataset.log` provides a complete audit trail
/// from the very first step.
fn write_import_log(
    output_dir: &std::path::Path,
    args: &ImportArgs,
    slots: &PipelineSlots,
    step_count: usize,
) {
    use std::io::Write;

    let path = output_dir.join("dataset.log");
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(_) => return, // best-effort
    };
    let mut w = std::io::BufWriter::new(file);

    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let _ = writeln!(w, "=== veks prepare bootstrap {} ===", ts);
    let _ = writeln!(w);

    // ── Inputs ───────────────────────────────────────────────────────
    let _ = writeln!(w, "Inputs:");
    let _ = writeln!(w, "  name:           {}", args.name);
    let _ = writeln!(w, "  output:         {}", args.output.display());
    match args.base_vectors {
        Some(ref p) => { let _ = writeln!(w, "  base_vectors:   {}", p.display()); }
        None => { let _ = writeln!(w, "  base_vectors:   (none)"); }
    }
    match args.query_vectors {
        Some(ref p) => { let _ = writeln!(w, "  query_vectors:  {}", p.display()); }
        None if args.self_search => {
            let _ = writeln!(w, "  query_vectors:  self-search (count={})", args.query_count);
        }
        None => { let _ = writeln!(w, "  query_vectors:  (none)"); }
    }
    match args.metadata {
        Some(ref p) => { let _ = writeln!(w, "  metadata:       {}", p.display()); }
        None => { let _ = writeln!(w, "  metadata:       (none)"); }
    }
    match args.ground_truth {
        Some(ref p) => { let _ = writeln!(w, "  ground_truth:   {}", p.display()); }
        None => { let _ = writeln!(w, "  ground_truth:   (compute)"); }
    }
    if let Some(ref p) = args.ground_truth_distances {
        let _ = writeln!(w, "  gt_distances:   {}", p.display());
    }
    let _ = writeln!(w, "  metric:         {}", args.metric);
    let _ = writeln!(w, "  neighbors:      {}", args.neighbors);
    let _ = writeln!(w, "  seed:           {}", args.seed);
    let _ = writeln!(w, "  normalize:      {}", args.normalize);
    let _ = writeln!(w, "  dedup:          {}", !args.no_dedup);
    let _ = writeln!(w, "  zero_check:     {}", !args.no_zero_check);
    let _ = writeln!(w, "  filtered_knn:   {}", !args.no_filtered);
    if args.base_fraction < 1.0 {
        let _ = writeln!(w, "  base_fraction:  {:.0}%", args.base_fraction * 100.0);
    }
    let _ = writeln!(w, "  compress_cache: {}", args.compress_cache);
    if let Some(ref sp) = args.sized_profiles {
        let _ = writeln!(w, "  sized_profiles: {}", sp);
    }
    if let Some(ref fmt) = args.base_convert_format {
        let _ = writeln!(w, "  base_convert:   {}", fmt);
    }
    if let Some(ref fmt) = args.query_convert_format {
        let _ = writeln!(w, "  query_convert:  {}", fmt);
    }

    // ── Detected starting scenario ───────────────────────────────────
    let _ = writeln!(w);
    let _ = writeln!(w, "Starting scenario:");
    let has_queries = args.query_vectors.is_some() || args.self_search;
    let has_metadata = args.metadata.is_some();
    let has_gt = args.ground_truth.is_some();
    let scenario = match (args.base_vectors.is_some(), has_queries, has_metadata, has_gt) {
        (true, true, true, true) => "full (base + query + metadata + pre-computed GT)",
        (true, true, true, false) => "full (base + query + metadata, compute GT)",
        (true, true, false, true) => "vectors + pre-computed GT (no metadata)",
        (true, true, false, false) => "vectors + compute GT (no metadata)",
        (true, false, true, _) => "base + metadata only (no queries)",
        (true, false, false, _) => "base vectors only (no queries, no metadata)",
        (false, _, true, _) => "metadata only (no vectors)",
        (false, _, _, _) => "minimal (no vectors, no metadata)",
    };
    let _ = writeln!(w, "  {}", scenario);

    // ── Dataflow graph entry points ──────────────────────────────────
    let _ = writeln!(w);
    let _ = writeln!(w, "Dataflow graph entry points ({} steps):", step_count);
    log_slot(&mut w, "all_vectors", &slots.all_vectors);
    log_slot(&mut w, "prepare", &slots.prepare);
    log_slot(&mut w, "vector_count", &slots.vector_count);
    if slots.self_search {
        let _ = writeln!(w, "  mode: self-search (query_count={})", args.query_count);
    }
    if let Some(ref s) = slots.shuffle { log_slot(&mut w, "shuffle", s); }
    if let Some(ref qv) = slots.query_vectors { log_slot(&mut w, "query_vectors", qv); }
    log_slot(&mut w, "base_vectors", &slots.base_vectors);
    if let Some(ref bc) = slots.base_count { log_slot(&mut w, "base_count", bc); }
    if let Some(ref meta) = slots.metadata {
        log_slot(&mut w, "metadata_all", &meta.metadata_all);
        log_slot(&mut w, "metadata_content", &meta.metadata_content);
        log_slot(&mut w, "survey", &meta.survey);
        log_slot(&mut w, "predicates", &meta.predicates);
        log_slot(&mut w, "predicate_indices", &meta.predicate_indices);
    } else {
        let _ = writeln!(w, "  metadata: Absent");
    }
    if let Some(ref knn) = slots.knn {
        log_slot(&mut w, "knn", knn);
    } else {
        let _ = writeln!(w, "  knn: Absent");
    }
    if let Some(ref fknn) = slots.filtered_knn {
        log_slot(&mut w, "filtered_knn", fknn);
    } else {
        let _ = writeln!(w, "  filtered_knn: Absent");
    }

    let _ = writeln!(w);
    let _ = w.flush();
}

/// Format a single slot for the provenance log.
fn log_slot(w: &mut impl std::io::Write, name: &str, artifact: &Artifact) {
    match artifact {
        Artifact::Materialized { step_id, output } => {
            if output.is_empty() {
                let _ = writeln!(w, "  {}: Materialized ({})", name, step_id);
            } else {
                let _ = writeln!(w, "  {}: Materialized ({}) -> {}", name, step_id, output);
            }
        }
        Artifact::Identity { path } => {
            if path.is_empty() {
                let _ = writeln!(w, "  {}: Identity (skipped)", name);
            } else {
                let _ = writeln!(w, "  {}: Identity -> {}", name, path);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample vectors from a file and check if they appear L2-normalized.
///
/// Returns `(is_normalized, sample_count, mean_norm)`. The normalization
/// threshold adapts to element precision and dimensionality per SRD §18.3:
/// `C × ε_mach(element_type) × √dim` (Higham & Mary 2019).
///
/// Uses mmap for sparse sampling — does not read the full file into memory.
pub fn detect_normalized(path: &Path) -> Option<(bool, usize, f64)> {
    use vectordata::VectorReader;
    use vectordata::io::MmapVectorReader;
    use crate::pipeline::element_type::ElementType;

    // Directories (npy) need special handling — probe first npy file directly
    if path.is_dir() {
        return probe_directory_vectors(path);
    }
    let etype = ElementType::from_path(path).ok()?;
    if !etype.is_float() {
        return None; // normalization only meaningful for float vectors
    }

    // Open via mmap — no full-file read
    let (count, dim, get_f64): (usize, usize, Box<dyn Fn(usize) -> Vec<f64>>) = match etype {
        ElementType::F32 => {
            let r = MmapVectorReader::<f32>::open_fvec(path).ok()?;
            let c = VectorReader::<f32>::count(&r);
            let d = VectorReader::<f32>::dim(&r);
            (c, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
        }
        ElementType::F16 => {
            let r = MmapVectorReader::<half::f16>::open_mvec(path).ok()?;
            let c = VectorReader::<half::f16>::count(&r);
            let d = VectorReader::<half::f16>::dim(&r);
            (c, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|v| v.to_f64()).collect()))
        }
        ElementType::F64 => {
            let r = MmapVectorReader::<f64>::open_dvec(path).ok()?;
            let c = VectorReader::<f64>::count(&r);
            let d = VectorReader::<f64>::dim(&r);
            (c, d, Box::new(move |i| r.get(i).unwrap_or_default()))
        }
        _ => return None,
    };

    if count == 0 || dim == 0 { return None; }

    // Sample strategy: rather than 100k scattered random-access reads
    // (which on a multi-TB EBS file means 100k cold-cache page faults
    // — minutes of silence), take a smaller number of small *contiguous
    // windows*. Same statistical coverage when the data is shuffled
    // (which it is for the wizard's typical input), one sequential
    // EBS read per window instead of thousands of random ones.
    //
    // 32 windows × 256 records ≈ 8k samples spread across the file.
    // At dim=384 / f32 that's 32 × 384 KiB = ~12 MiB of sequential I/O
    // total — under a second on cold EBS, vs minutes of random-access.
    const WINDOWS: usize = 32;
    const WINDOW_RECORDS: usize = 256;
    let sample_count = count.min(WINDOWS * WINDOW_RECORDS);
    let mut norms: Vec<f64> = Vec::with_capacity(sample_count);

    // Periodic progress feedback to stderr so the user sees we're alive
    // on big-file runs. Carriage return lets it overwrite in place.
    let mut next_tick = std::time::Instant::now();
    let tick_interval = std::time::Duration::from_millis(250);
    let print_progress = |done: usize, total: usize| {
        use std::io::Write;
        eprint!("\r  Sampling vectors for normalization status... [{}/{}]    ", done, total);
        let _ = std::io::stderr().flush();
    };

    let actual_windows = WINDOWS.min((count + WINDOW_RECORDS - 1) / WINDOW_RECORDS).max(1);
    let window_stride = if actual_windows == 1 {
        0
    } else {
        // Spread the windows evenly across the file with the last one
        // ending just before `count`. (count - WINDOW_RECORDS) span,
        // divided into (actual_windows - 1) gaps.
        count.saturating_sub(WINDOW_RECORDS) / (actual_windows - 1).max(1)
    };
    let mut done = 0usize;
    'outer: for w in 0..actual_windows {
        let start = w * window_stride;
        for j in 0..WINDOW_RECORDS {
            let idx = start + j;
            if idx >= count { break 'outer; }
            let vec = get_f64(idx);
            let sum: f64 = vec.iter().map(|v| v * v).sum();
            norms.push(sum.sqrt());
            done += 1;
            if std::time::Instant::now() >= next_tick {
                print_progress(done, sample_count);
                next_tick = std::time::Instant::now() + tick_interval;
            }
        }
    }
    // Wipe the in-place progress line so the caller's "is normalized"
    // message starts from column 0.
    {
        use std::io::Write;
        eprint!("\r  Sampling vectors for normalization status... ");
        let _ = std::io::stderr().flush();
    }

    if norms.is_empty() { return None; }

    let mean = norms.iter().sum::<f64>() / norms.len() as f64;
    // Precision-aware threshold (SRD §18.3, Higham & Mary 2019)
    let threshold = etype.normalization_threshold(dim)
        .unwrap_or(10.0 * 1e-7 * (dim as f64).sqrt());
    let mean_epsilon = (mean - 1.0).abs();
    let max_epsilon = norms.iter().map(|n| (n - 1.0).abs()).fold(0.0_f64, f64::max);
    let is_normalized = mean_epsilon < threshold && max_epsilon < threshold * 5.0;

    Some((is_normalized, norms.len(), mean))
}

/// Probe a source directory or file by opening it through the standard
/// reader infrastructure and sampling vectors for norm computation.
/// Uses the same code path as the pipeline — supports npy dirs, xvec
/// files, and any format that `VecFormat::detect` + `open_source` handles.
fn probe_directory_vectors(path: &Path) -> Option<(bool, usize, f64)> {
    use crate::pipeline::element_type::ElementType;
    use veks_core::formats::reader;
    let format = VecFormat::detect(path)?;
    let mut source = reader::open_source(path, format, 1, Some(100)).ok()?;

    let dim = source.dimension() as usize;
    if dim == 0 { return None; }

    // Detect element size from format
    let elem_size = match format {
        VecFormat::Mvec => 2,  // f16
        VecFormat::Dvec => 8,  // f64
        VecFormat::Bvec => 1,  // u8
        VecFormat::Svec => 2,  // i16
        VecFormat::Ivec => 4,  // i32
        _ => 4,                // f32 (fvec, npy, parquet)
    };
    let mut norms = Vec::with_capacity(100);
    while let Some(record) = source.next_record() {
        if record.is_empty() { continue; }
        let actual_elem_size = if record.len() == dim * 2 { 2 }
            else if record.len() == dim * 4 { 4 }
            else if record.len() == dim * 8 { 8 }
            else { elem_size };

        let norm: f64 = (0..dim)
            .map(|d| {
                let off = d * actual_elem_size;
                if off + actual_elem_size > record.len() { return 0.0; }
                let v: f64 = match actual_elem_size {
                    2 => half::f16::from_le_bytes([record[off], record[off + 1]]).to_f64(),
                    8 => f64::from_le_bytes(record[off..off + 8].try_into().unwrap_or([0; 8])),
                    _ => f32::from_le_bytes(record[off..off + 4].try_into().unwrap_or([0; 4])) as f64,
                };
                v * v
            })
            .sum::<f64>()
            .sqrt();
        norms.push(norm);
        if norms.len() >= 100 { break; }
    }

    if norms.is_empty() { return None; }

    let mean = norms.iter().sum::<f64>() / norms.len() as f64;
    // Infer element type from format for precision-aware threshold (SRD §18.3)
    let probe_etype = match format {
        VecFormat::Mvec => ElementType::F16,
        VecFormat::Dvec => ElementType::F64,
        _ => ElementType::F32,
    };
    let threshold = probe_etype.normalization_threshold(dim)
        .unwrap_or(10.0 * 1e-7 * (dim as f64).sqrt());
    let mean_epsilon = (mean - 1.0).abs();
    let max_epsilon = norms.iter().map(|n| (n - 1.0).abs()).fold(0.0_f64, f64::max);
    let is_normalized = mean_epsilon < threshold && max_epsilon < threshold * 5.0;

    Some((is_normalized, norms.len(), mean))
}

/// Detect the likely distance metric from vector data.
///
/// Always returns Cosine — it is the safest default and produces correct
/// rankings for both normalized and non-normalized data. The normalization
/// status is reported in the reason string so the pipeline can set the
/// `normalized` flag for kernel optimization (DotProduct kernel is faster
/// when norms are known to be 1.0).
///
/// Returns the metric name and a human-readable reason.
pub fn detect_metric(path: &Path) -> (String, String) {
    // First: infer metric from filename keywords. The filename (or parent
    // directory) often encodes the distance function used to train the model.
    if let Some(metric) = detect_metric_from_filename(path) {
        return metric;
    }

    // Fallback: probe vector data for normalization status
    let probe_result = if path.is_dir() {
        probe_directory_vectors(path)
    } else {
        detect_normalized(path)
    };
    if let Some((is_normalized, _n, mean_norm)) = probe_result {
        if is_normalized {
            ("COSINE".to_string(), format!("vectors are L2-normalized (mean norm={:.4})", mean_norm))
        } else {
            ("COSINE".to_string(), format!("vectors not normalized (mean norm={:.4})", mean_norm))
        }
    } else {
        let detail = if path.is_dir() {
            "directory probe returned no results"
        } else if !path.exists() {
            "path does not exist"
        } else {
            "unsupported format for probing"
        };
        ("COSINE".to_string(), format!("default — {} ({})", detail, path.display()))
    }
}

/// Detect distance metric from filename keywords.
///
/// Checks the source path (including parent directory and HDF5 dataset
/// name) for well-known metric keywords. Returns the metric and reason
/// if a match is found.
fn detect_metric_from_filename(path: &Path) -> Option<(String, String)> {
    // Check each name in the symlink chain: the original symlink name,
    // any intermediate symlinks, and the final target. The first match
    // in the chain is definitive.
    let mut current = path.to_path_buf();
    loop {
        let search = current.to_string_lossy().to_lowercase();
        if let Some(result) = check_metric_keywords(&search, &current) {
            return Some(result);
        }
        // Follow one symlink level
        match std::fs::read_link(&current) {
            Ok(target) => {
                // Resolve relative symlink targets against the parent dir
                current = if target.is_relative() {
                    current.parent().unwrap_or(Path::new(".")).join(&target)
                } else {
                    target
                };
            }
            Err(_) => break, // Not a symlink or error — stop
        }
    }
    None
}

/// Check a single path string for distance metric keywords.
fn check_metric_keywords(search: &str, display_path: &Path) -> Option<(String, String)> {
    if search.contains("dot_product") || search.contains("dotproduct")
        || search.contains("-dot-") || search.contains("_dot_")
        || search.contains("-dot.") || search.contains("_dot.")
        || search.contains("/dot-") || search.contains("/dot_") {
        Some(("DOT_PRODUCT".to_string(), format!("filename contains dot product keyword ({})", display_path.display())))
    } else if search.contains("euclidean") || search.contains("-l2-") || search.contains("_l2_") || search.contains("-l2.") || search.contains("_l2.") {
        Some(("L2".to_string(), format!("filename contains L2/euclidean keyword ({})", display_path.display())))
    } else if search.contains("angular") || search.contains("cosine") {
        Some(("COSINE".to_string(), format!("filename contains angular/cosine keyword ({})", display_path.display())))
    } else {
        None
    }
}

/// Make a path relative to a base directory. If the path can't be made
/// relative (different root), returns the original path as a string.
fn relativize_path(path: &Path, base: &Path) -> String {
    relative_path(base, path).to_string_lossy().to_string()
}

fn is_native_xvec_file(path: &Path) -> bool {
    if path.is_dir() { return false; }
    VecFormat::detect_from_path(path)
        .map(|f| f.is_xvec())
        .unwrap_or(false)
}

fn is_native_slab_file(path: &Path) -> bool {
    if path.is_dir() { return false; }
    VecFormat::detect_from_path(path)
        .map(|f| f == VecFormat::Slab)
        .unwrap_or(false)
}

fn print_slot(name: &str, artifact: &Artifact) {
    match artifact {
        Artifact::Materialized { step_id, output } => {
            if output.is_empty() {
                println!("  {}: Materialized ({})", name, step_id);
            } else {
                println!("  {}: Materialized ({}) -> {}", name, step_id, output);
            }
        }
        Artifact::Identity { path } => {
            if path.is_empty() {
                println!("  {}: Identity (skipped)", name);
            } else {
                println!("  {}: Identity -> {}", name, path);
            }
        }
    }
}

fn count_identity(slots: &PipelineSlots) -> usize {
    let mut n = 0;
    if !slots.all_vectors.is_materialized() { n += 1; }
    if !slots.prepare.is_materialized() { n += 1; }
    if !slots.base_vectors.is_materialized() { n += 1; }
    if let Some(ref qv) = slots.query_vectors { if !qv.is_materialized() { n += 1; } }
    if let Some(ref meta) = slots.metadata {
        if !meta.metadata_all.is_materialized() { n += 1; }
        if !meta.metadata_content.is_materialized() { n += 1; }
    }
    if let Some(ref knn) = slots.knn { if !knn.is_materialized() { n += 1; } }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal fvec file (N records, dim D).
    fn write_fvec(path: &Path, records: usize, dim: u32) {
        use std::io::Write;
        let f = std::fs::File::create(path).unwrap();
        let mut w = std::io::BufWriter::new(f);
        for _ in 0..records {
            w.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for _ in 0..dim {
                w.write_all(&0.0f32.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    fn default_args() -> ImportArgs {
        ImportArgs {
            name: "test".into(),
            output: PathBuf::from("/tmp/test-out"),
            base_vectors: None,
            query_vectors: None,
            self_search: false,
            query_count: 10000,
            metadata: None,
            ground_truth: None,
            ground_truth_distances: None,
            metric: "L2".into(),
            neighbors: 100,
            seed: 42,
            description: None,
            no_dedup: false,
            no_zero_check: false,
            duplicate_count: None,
            zero_count: None,
            no_filtered: false,
            normalize: false,
            force: false,
            base_convert_format: None,
            query_convert_format: None,
            compress_cache: true,
            sized_profiles: None,
            base_fraction: 1.0,
            required_facets: None,
            provided_facets: None,
            round_digits: 2,
            pedantic_dedup: false,
            selectivity: 0.0001,
            predicate_count: 10000,
            predicate_strategy: "eq".to_string(),
            classic: false,
            personality: "native".to_string(),
            synthesize_metadata: false,
            synthesis_mode: "simple-int-eq".to_string(),
            synthesis_format: "slab".to_string(),
            metadata_fields: 3,
            metadata_range_min: 0,
            metadata_range_max: 1000,
            predicate_range_min: 0,
            predicate_range_max: 1000,
            verify_knn_sample: 0,
            partition_oracles: false,
            max_partitions: 100,
            on_undersized: "error".to_string(),
            cosine_mode: None,
        }
    }

    // SRD §12.5 Example 1: Minimal — native fvec, no queries, no metadata
    // Expects: import collapses to identity, self-search activates,
    //          dedup + shuffle + extract + KNN materialize
    /// Regression: when --base-fraction subsets the source AND
    /// extract-base does NOT materialize (no dedup / no normalize /
    /// no shuffle), the compute-knn step's `base:` option must point
    /// at the SUBSET file with a `[0..${vector_count})` cap — not at
    /// the unsubsetted source.
    ///
    /// Caught only when the symptom appeared in production: compute-knn
    /// scanned the entire source file even though subset-vectors had
    /// reduced the effective dataset, then verify failed because the
    /// returned indices reached past the per-profile boundary.
    #[test]
    fn fractioned_compute_knn_reads_only_subset() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.fvec");
        let query = dir.path().join("query.fvec");
        write_fvec(&base, 1000, 4);
        write_fvec(&query, 10, 4);

        let mut args = default_args();
        args.base_vectors = Some(base);
        args.query_vectors = Some(query);
        args.no_dedup = true;
        args.no_zero_check = true;
        args.normalize = false;
        args.base_fraction = 0.01;
        args.seed = 0; // disable shuffle so extract-base does not materialize

        let slots = resolve_slots(&args);
        let steps = emit_steps(&slots, &args, &args.output);

        let knn = steps.iter().find(|s| s.id == "compute-knn")
            .expect("compute-knn step should be emitted");
        let base_opt = knn.options.iter().find(|(k, _)| k == "base")
            .map(|(_, v)| v.as_str())
            .expect("compute-knn should have a `base` option");
        // Must read from the SUBSET (working_vectors path) AND cap at
        // ${vector_count} (which counts the subset). Old buggy
        // behaviour: base = original source path with no cap → reads
        // the entire source file.
        assert!(
            base_opt.contains("all_vectors.fvec"),
            "compute-knn must read from the subset (`all_vectors.fvec`), got: {}",
            base_opt,
        );
        assert!(
            base_opt.contains("${vector_count}"),
            "compute-knn must cap range at ${{vector_count}}, got: {}",
            base_opt,
        );
    }

    #[test]
    fn example1_native_fvec_self_search() {
        let dir = tempfile::tempdir().unwrap();
        let fvec = dir.path().join("base.fvec");
        write_fvec(&fvec, 10, 3);

        let mut args = default_args();
        args.base_vectors = Some(fvec.clone());
        // no --query-vectors → self-search implied

        let slots = resolve_slots(&args);

        // all_vectors should be Identity (native fvec)
        assert!(!slots.all_vectors.is_materialized(), "native fvec should collapse to identity");
        assert!(slots.all_vectors.path().contains("base.fvec"));

        // sort should be materialized
        assert!(slots.prepare.is_materialized());

        // self-search should be active
        assert!(slots.self_search);
        assert!(slots.shuffle.as_ref().unwrap().is_materialized());
        assert!(slots.query_vectors.as_ref().unwrap().is_materialized());
        assert!(slots.base_vectors.is_materialized());

        // KNN should materialize
        assert!(slots.knn.as_ref().unwrap().is_materialized());

        // No metadata
        assert!(slots.metadata.is_none());
        assert!(slots.filtered_knn.is_none());

        let steps = emit_steps(&slots, &args, &args.output);
        // Expected: count-vectors, count-source-base, set-is_shuffled,
        // set-is_self_search, set-combined_bq, set-k, prepare-vectors,
        // count-duplicates, generate-shuffle, extract-queries, extract-base,
        // count-base, compute-knn, verify-knn, generate-dataset-json,
        // generate-variables-json, generate-dataset-log-jsonl, generate-docs,
        // generate-catalog, generate-merkle.
        assert_eq!(steps.len(), 20, "steps: {:?}", steps.iter().map(|s| &s.id).collect::<Vec<_>>());
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(step_ids.contains(&"count-duplicates"), "should have count-duplicates");
        assert!(step_ids.contains(&"count-source-base"), "should have count-source-base");
        assert!(step_ids.contains(&"set-is_shuffled"), "should have set-is_shuffled");
        assert!(step_ids.contains(&"generate-variables-json"), "should have generate-variables-json");
        assert!(step_ids.contains(&"generate-dataset-log-jsonl"), "should have generate-dataset-log-jsonl");
    }

    // SRD §12.5 Example 3: Native base + separate native query + metadata dir
    // With Strategy 1 (combined B+Q), base and query are combined before
    // dedup, then shuffled for train/test split. Metadata materializes.
    #[test]
    fn example3_native_base_separate_query_with_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.fvec");
        let query = dir.path().join("query.fvec");
        let meta = dir.path().join("meta_dir");
        write_fvec(&base, 10, 3);
        write_fvec(&query, 5, 3);
        std::fs::create_dir(&meta).unwrap();
        // Create a parquet file to trigger format detection
        std::fs::write(meta.join("data.parquet"), b"fake").unwrap();

        let mut args = default_args();
        args.base_vectors = Some(base.clone());
        args.query_vectors = Some(query.clone());
        args.metadata = Some(meta);

        let slots = resolve_slots(&args);

        // Strategy 1: combined B+Q → self_search with shuffle
        assert!(slots.self_search, "combined B+Q should be self-search");
        assert!(slots.combined_bq, "should be combined_bq");
        assert!(slots.shuffle.is_some(), "should have shuffle");
        assert!(slots.query_vectors.as_ref().unwrap().is_materialized(), "queries extracted from shuffle");
        assert!(slots.base_vectors.is_materialized(), "base extracted from shuffle");

        // Metadata: import materializes (parquet dir), extract materializes (shuffle reorders)
        let meta = slots.metadata.as_ref().unwrap();
        assert!(meta.metadata_all.is_materialized());

        // KNN + filtered KNN materialize
        assert!(slots.knn.as_ref().unwrap().is_materialized());
        assert!(slots.filtered_knn.as_ref().unwrap().is_materialized());

        let steps = emit_steps(&slots, &args, &args.output);
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(step_ids.contains(&"convert-metadata"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"survey-metadata"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"compute-knn"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"compute-filtered-knn"), "steps: {:?}", step_ids);
        // Strategy 1: shuffle and extract ARE present
        assert!(step_ids.contains(&"generate-shuffle"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"extract-queries"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"extract-base"), "steps: {:?}", step_ids);
    }

    // No dedup when --no-dedup is set
    #[test]
    fn no_dedup_flag() {
        let dir = tempfile::tempdir().unwrap();
        let fvec = dir.path().join("base.fvec");
        write_fvec(&fvec, 5, 2);

        let mut args = default_args();
        args.base_vectors = Some(fvec);
        args.no_dedup = true;

        let slots = resolve_slots(&args);
        assert!(!slots.prepare.is_materialized(), "sort should be identity when --no-dedup");

        let steps = emit_steps(&slots, &args, &args.output);
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(!step_ids.contains(&"sort-and-dedup"));
    }

    // Pre-computed ground truth collapses KNN to identity
    #[test]
    fn precomputed_ground_truth() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.fvec");
        let query = dir.path().join("query.fvec");
        let gt = dir.path().join("gt.ivec");
        write_fvec(&base, 10, 3);
        write_fvec(&query, 5, 3);
        std::fs::write(&gt, b"fake gt").unwrap();

        let mut args = default_args();
        args.base_vectors = Some(base);
        args.query_vectors = Some(query);
        args.ground_truth = Some(gt.clone());

        let slots = resolve_slots(&args);
        // KNN should be identity (pre-computed)
        let knn = slots.knn.as_ref().unwrap();
        assert!(!knn.is_materialized(), "KNN should be identity when GT provided");
        assert!(knn.path().contains("gt.ivec"));

        let steps = emit_steps(&slots, &args, &args.output);
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(!step_ids.contains(&"compute-knn"));
    }

    // No queries → no KNN, no filtered KNN
    #[test]
    fn no_queries_no_knn() {
        let dir = tempfile::tempdir().unwrap();
        let fvec = dir.path().join("base.fvec");
        write_fvec(&fvec, 5, 2);

        let mut args = default_args();
        args.base_vectors = Some(fvec);
        args.self_search = false;
        // Explicitly no query vectors and no self-search
        // But default behavior: self-search is implied when base_vectors present
        // and no query_vectors. Force no-query by providing neither.
        args.base_vectors = None; // no base → no query path at all

        let slots = resolve_slots(&args);
        assert!(slots.query_vectors.is_none());
        assert!(slots.knn.is_none());
        assert!(slots.filtered_knn.is_none());
    }

    #[test]
    fn relative_path_sibling() {
        let base = Path::new("/a/b/c");
        let target = Path::new("/a/b/d/file.txt");
        let rel = relative_path(base, target);
        assert_eq!(rel, PathBuf::from("../d/file.txt"));
    }

    #[test]
    fn relative_path_child() {
        let base = Path::new("/a/b");
        let target = Path::new("/a/b/c/file.txt");
        let rel = relative_path(base, target);
        assert_eq!(rel, PathBuf::from("c/file.txt"));
    }

    #[test]
    fn relative_path_same_dir() {
        let base = Path::new("/a/b");
        let target = Path::new("/a/b/file.txt");
        let rel = relative_path(base, target);
        assert_eq!(rel, PathBuf::from("file.txt"));
    }

    #[test]
    fn relative_path_upward() {
        let base = Path::new("/a/b/c/d");
        let target = Path::new("/a/x.txt");
        let rel = relative_path(base, target);
        assert_eq!(rel, PathBuf::from("../../../x.txt"));
    }

    #[test]
    fn relative_path_identical() {
        let base = Path::new("/a/b");
        let target = Path::new("/a/b");
        let rel = relative_path(base, target);
        assert_eq!(rel, PathBuf::from("."));
    }

    #[test]
    fn personality_native_uses_native_commands() {
        let mut args = default_args();
        args.personality = "native".into();
        args.base_vectors = Some(PathBuf::from("/fake/base.fvec"));
        args.self_search = true;
        args.normalize = true;
        let slots = resolve_slots(&args);
        let steps = emit_steps(&slots, &args, Path::new("/tmp"));
        let runs: Vec<&str> = steps.iter().map(|s| s.run.as_str()).collect();
        assert!(runs.contains(&"compute sort"), "native should use 'compute sort': {:?}", runs);
        assert!(runs.contains(&"generate shuffle"), "native should use 'generate shuffle': {:?}", runs);
        // Native personality now defaults to knn-blas for KNN (numpy
        // / knn_utils kernel parity). sort / shuffle remain native.
        assert!(runs.contains(&"compute knn-blas"),
            "native should use 'compute knn-blas' by default: {:?}", runs);
    }

    #[test]
    fn personality_knnutils_uses_knnutils_commands() {
        let mut args = default_args();
        args.personality = "knn_utils".into();
        args.base_vectors = Some(PathBuf::from("/fake/base.fvec"));
        args.self_search = true;
        args.normalize = true;
        let slots = resolve_slots(&args);
        let steps = emit_steps(&slots, &args, Path::new("/tmp"));
        let runs: Vec<&str> = steps.iter().map(|s| s.run.as_str()).collect();
        assert!(runs.contains(&"compute sort-knnutils"),
            "knn_utils personality should use 'compute sort-knnutils': {:?}", runs);
        assert!(runs.contains(&"generate shuffle-knnutils"),
            "knn_utils personality should use 'generate shuffle-knnutils': {:?}", runs);
        assert!(runs.contains(&"compute knn-blas"),
            "knn_utils personality should use 'compute knn-blas': {:?}", runs);
        assert!(runs.contains(&"verify dataset-knnutils"),
            "knn_utils personality should use 'verify dataset-knnutils': {:?}", runs);
    }

    #[test]
    fn parse_facet_spec_star_is_all() {
        assert_eq!(parse_facet_spec("*"), "BQGDMPRF");
    }

    #[test]
    fn parse_facet_spec_all_keyword() {
        assert_eq!(parse_facet_spec("all"), "BQGDMPRF");
        assert_eq!(parse_facet_spec("ALL"), "BQGDMPRF");
    }

    #[test]
    fn parse_facet_spec_compact_codes() {
        assert_eq!(parse_facet_spec("BQ"), "BQ");
        assert_eq!(parse_facet_spec("bqgd"), "BQGD");
    }

    #[test]
    fn parse_facet_spec_long_names() {
        assert_eq!(parse_facet_spec("base,query"), "BQ");
        assert_eq!(parse_facet_spec("base,query,gt,dist"), "BQGD");
    }
}
