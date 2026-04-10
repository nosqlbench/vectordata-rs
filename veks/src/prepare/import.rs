// Copyright (c) nosqlbench contributors
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
    /// source is given, a `generate metadata` step is emitted.
    #[serde(default)]
    pub synthesize_metadata: bool,
    /// Number of integer fields for synthesized metadata (default 3).
    #[serde(default = "default_metadata_fields")]
    pub metadata_fields: u32,
    /// Minimum value (inclusive) for synthesized metadata integer range.
    #[serde(default)]
    pub metadata_range_min: i32,
    /// Maximum value (exclusive) for synthesized metadata integer range.
    #[serde(default = "default_metadata_range_max")]
    pub metadata_range_max: i32,
}

fn default_personality() -> String {
    "native".to_string()
}

fn default_metadata_fields() -> u32 { 3 }
fn default_metadata_range_max() -> i32 { 1000 }

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
];

/// Parse a facet specification string into a canonical code string.
///
/// Accepts:
/// - `"BQGD"` — compact codes
/// - `"B,Q,G,D"` or `"base,query,gt,dist"` — comma-separated
/// - `"base query gt dist"` — space-separated names
/// - Full facet names like `"base_vectors,query_vectors"`
pub fn parse_facet_spec(spec: &str) -> String {
    let spec = spec.trim();

    // Strip leading '=' that can appear from shell quoting like --flag ="value"
    let spec = spec.strip_prefix('=').unwrap_or(spec);

    // '*' or 'all' → every facet
    if spec == "*" || spec.eq_ignore_ascii_case("all") {
        return "BQGDMPRF".to_string();
    }

    // If it's all uppercase letters from BQGDMPRF, treat as compact codes
    if !spec.is_empty()
        && spec.chars().all(|c| "BQGDMPRFbqgdmprf".contains(c))
        && !spec.contains(',')
        && !spec.contains(' ')
    {
        return spec.to_uppercase();
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
            other => {
                // Single uppercase letter?
                if other.len() == 1 && "bqgdmprf".contains(other) {
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
    if let Some(ref spec) = args.required_facets {
        return parse_facet_spec(spec);
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

    // Inference from available inputs (SRD 2.8 implication rules)
    match (has_base, has_meta) {
        (true, true) => "BQGDMPRF".to_string(),
        (false, true) => "BQGDMPR".to_string(),
        (true, false) => {
            if has_gt { "BQGD".to_string() }
            else { "BQGD".to_string() }
        }
        (false, false) => String::new(),
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
                    1 => "bvec",
                    2 => "mvec",
                    8 => "dvec",
                    _ => "fvec", // 4 bytes (f32/i32) → fvec
                })
            })
            .unwrap_or("fvec")
    } else {
        "fvec"
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
            output: "${cache}/sorted_ordinals.ivec".into(),
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

    // Query generation strategy (SRD §20.8):
    // Strategy 1: Non-HDF5 B+Q → combine into single source, shuffle split
    // Strategy 2: HDF5 B+Q → independent processing
    // Strategy 3: Non-HDF5 B only → self-search via shuffle
    let combined_bq = has_separate_query && !is_hdf5_source;
    let self_search = wants_queries && (!has_separate_query || combined_bq);
    let has_queries = has_separate_query || self_search;

    // Determine output extension from source format
    let vec_ext = if needs_import { import_ext } else {
        args.base_vectors.as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .and_then(VecFormat::canonical_extension)
            .unwrap_or("fvec")
    };

    let (shuffle, query_vectors, base_vectors, base_count) = if self_search {
        // Strategy 1 (combined B+Q) or Strategy 3 (B only): shuffle + extract
        // For Strategy 1, base+query are combined before prepare-vectors,
        // then shuffle produces disjoint train/test split.
        let shuffle = Artifact::Materialized {
            step_id: "generate-shuffle".into(),
            output: "${cache}/shuffle.ivec".into(),
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
        // The shuffle randomizes the base vector order after dedup,
        // same as non-HDF5 datasets. Queries are not included in the
        // shuffle since they come from a separate HDF5 dataset.
        let shuffle = Artifact::Materialized {
            step_id: "generate-shuffle".into(),
            output: "${cache}/shuffle.ivec".into(),
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
        (Some(shuffle), Some(qv), bv, Some(bc))
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
        // Synthesize metadata — generate metadata step produces the slab
        let metadata_all = Artifact::Materialized {
            step_id: "generate-metadata".into(),
            output: "${cache}/metadata_all.slab".into(),
        };
        let metadata_content = if self_search {
            Artifact::Materialized {
                step_id: "extract-metadata".into(),
                output: format!("{}metadata_content.slab", pp),
            }
        } else {
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
            output: "neighbor_indices.ivec".into(), // per_profile
        })
    };

    let wants_filtered = facets.contains('F') && !args.no_filtered;
    let filtered_knn = if !has_queries || metadata.is_none() || !wants_filtered {
        None
    } else {
        Some(Artifact::Materialized {
            step_id: "compute-filtered-knn".into(),
            output: "filtered_neighbor_indices.ivec".into(), // per_profile
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

fn cmd_knn(personality: &str) -> &'static str {
    match personality {
        "knn_utils" => "compute knn-blas",
        _ => "compute knn",
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
        let format = args.base_vectors.as_ref()
            .and_then(|p| VecFormat::detect(p))
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
        if args.base_fraction < 1.0 && !args.pedantic_dedup {
            convert_opts.push(("fraction".into(), args.base_fraction.to_string()));
        }
        steps.push(Step {
            id: "convert-vectors".into(),
            run: "transform convert".into(),
            description: Some("Import base vectors from source format".into()),
            after: vec![],
            per_profile: false,
            phase: 0,
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
            description: Some("Sort and deduplicate vectors".into()),
            after: if last_vector_step.is_empty() { vec![] } else { vec![last_vector_step.into()] },
            per_profile: false,
            phase: 0,
            options: {
                let mut opts = vec![
                    ("source".into(), working_vectors.clone()),
                    ("output".into(), "${cache}/sorted_ordinals.ivec".into()),
                    ("duplicates".into(), "${cache}/dedup_duplicates.ivec".into()),
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
            options: vec![
                ("name".into(), "duplicate_count".into()),
                ("value".into(), "count:${cache}/dedup_duplicates.ivec".into()),
            ],
        });

        // No separate filter-ordinals, count-clean, find-zeros, count-zeros,
        // or measure-normals steps. All absorbed into prepare-vectors (SRD §20).
        // The exclusion set (duplicates ∪ zeros) is applied directly by
        // extract-base and extract-queries.
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
                ("output".into(), "${cache}/shuffle.ivec".into()),
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
                    ("ivec-file".into(), "${cache}/shuffle.ivec".into()),
                    ("output".into(), slots.query_vectors.as_ref().map(|q| q.path().to_string()).unwrap_or_else(|| format!("{}query_vectors.fvec", pp))),
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
                    options: query_opts,
                });
                let base_opts = vec![
                    ("source".into(), working_vectors.clone()),
                    ("ivec-file".into(), "${cache}/shuffle.ivec".into()),
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
                    options: base_opts,
                });
                steps.push(Step {
                    id: "count-base".into(),
                    run: "state set".into(),
                    description: None,
                    after: vec!["extract-base".into()],
                    per_profile: false,
                    phase: 0,
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
            let format = VecFormat::detect(query_source)
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
                options: vec![
                    ("facet".into(), "query_vectors".into()),
                    ("source".into(), relativize_path(query_source, &args.output)),
                    ("from".into(), format),
                    ("output".into(), qv.path().into()),
                    ("normalize".into(), "true".into()),
                ],
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
                base_opts.push(("ivec-file".into(), "${cache}/shuffle.ivec".into()));
                base_opts.push(("range".into(), format!("[0,{})", "${clean_count}")));
            } else if slots.prepare.is_materialized() {
                base_opts.push(("ivec-file".into(), slots.prepare.path().into()));
                base_opts.push(("range".into(), format!("[0,{})", "${vector_count}")));
            }
            base_opts.push(("normalize".into(), "true".into()));
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
                    options: meta_opts,
                });
            } else if args.synthesize_metadata {
                // Synthesize metadata via generate metadata command
                steps.push(Step {
                    id: "generate-metadata".into(),
                    run: "generate metadata".into(),
                    description: Some("Synthesize random integer metadata".into()),
                    after: vec!["count-vectors".into()],
                    per_profile: false,
                    phase: 0,
                    options: vec![
                        ("output".into(), "${cache}/metadata_all.slab".into()),
                        ("count".into(), "${vector_count}".into()),
                        ("fields".into(), args.metadata_fields.to_string()),
                        ("range-min".into(), args.metadata_range_min.to_string()),
                        ("range-max".into(), args.metadata_range_max.to_string()),
                        ("seed".into(), args.seed.to_string()),
                    ],
                });
            }
        }

        if meta.metadata_content.is_materialized() {
            let mut after = vec![];
            if meta.metadata_all.is_materialized() {
                // Depends on whichever step produced metadata_all
                if args.metadata.is_some() {
                    after.push("convert-metadata".into());
                } else if args.synthesize_metadata {
                    after.push("generate-metadata".into());
                }
            }
            if slots.shuffle.as_ref().map(|s| s.is_materialized()).unwrap_or(false) {
                after.push("generate-shuffle".into());
            }
            if args.base_fraction < 1.0 && !subset_applied {
                after.push("compute-base-end".into());
            }
            steps.push(Step {
                id: "extract-metadata".into(),
                run: "transform extract".into(),
                description: Some("Reorder metadata to match shuffled base vectors".into()),
                after,
                per_profile: false,
                phase: 0,
                options: vec![
                    ("source".into(), meta.metadata_all.path().into()),
                    ("ivec-file".into(), "${cache}/shuffle.ivec".into()),
                    ("output".into(), format!("{}metadata_content.slab", pp)),
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

        // survey (always when metadata present)
        let survey_after = if meta.metadata_all.is_materialized() {
            vec!["convert-metadata".into()]
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
            options: vec![
                ("input".into(), meta.metadata_all.path().into()),
                ("output".into(), "${cache}/metadata_survey.json".into()),
                ("samples".into(), "10000".into()),
                ("max-distinct".into(), "100".into()),
            ],
        });

        // synthesize predicates
        steps.push(Step {
            id: "generate-predicates".into(),
            run: "generate predicates".into(),
            description: Some("Generate test predicates from metadata survey".into()),
            after: vec!["survey-metadata".into()],
            per_profile: false,
            phase: 0,
            options: vec![
                ("input".into(), meta.metadata_all.path().into()),
                ("survey".into(), "${cache}/metadata_survey.json".into()),
                ("output".into(), format!("{}predicates.slab", pp)),
                ("count".into(), "10000".into()),
                ("selectivity".into(), args.selectivity.to_string()),
                ("seed".into(), format!("${{{}}}", "seed")),
            ],
        });

        // compute predicates (per_profile)
        let mut eval_after = vec!["generate-predicates".into()];
        if meta.metadata_content.is_materialized() {
            eval_after.push("extract-metadata".into());
        }
        steps.push(Step {
            id: "evaluate-predicates".into(),
            run: "compute evaluate-predicates".into(),
            description: None,
            after: eval_after,
            per_profile: true,
            phase: 1, // after all compute-knn — metadata I/O phase
            options: vec![
                ("input".into(), meta.metadata_content.path().into()),
                ("predicates".into(), format!("{}predicates.slab", pp)),
                ("survey".into(), "${cache}/metadata_survey.json".into()),
                ("selectivity".into(), args.selectivity.to_string()),
                ("range".into(), if slots.base_count.is_some() {
                    "[0,${base_end})".into()
                } else if slots.prepare.is_materialized() {
                    "[0,${clean_count})".into()
                } else {
                    "[0,${vector_count})".into()
                }),
                ("output".into(), "metadata_indices.slab".into()),
            ],
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
                after.push("extract-base".into());
            }
            if slots.all_vectors.is_materialized() {
                after.push("convert-vectors".into());
            }
            if let Some(ref qv) = slots.query_vectors {
                if qv.is_materialized() {
                    let qid = if slots.self_search { "extract-queries" } else { "convert-queries" };
                    after.push(qid.into());
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
                options: {
                    let mut opts = vec![
                        ("base".into(), if slots.base_count.is_some() {
                            format!("{}[0..${{base_count}})", slots.base_vectors.path())
                        } else {
                            slots.base_vectors.path().to_string()
                        }),
                        ("query".into(), query_path.clone()),
                        ("indices".into(), "neighbor_indices.ivec".into()),
                        ("distances".into(), "neighbor_distances.fvec".into()),
                        ("neighbors".into(), args.neighbors.to_string()),
                        ("metric".into(), args.metric.clone()),
                    ];
                    if args.compress_cache {
                        opts.push(("compress_cache".into(), "true".into()));
                    }
                    opts.push(("normalized".into(), "true".into()));
                    opts
                },
            });
        }
    }

    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            let mut after = vec!["evaluate-predicates".into()];
            if slots.base_count.is_some() {
                after.push("count-base".into());
            }
            if let Some(ref qv) = slots.query_vectors {
                if qv.is_materialized() {
                    let qid = if slots.self_search { "extract-queries" } else { "convert-queries" };
                    after.push(qid.into());
                }
            }
            steps.push(Step {
                id: "compute-filtered-knn".into(),
                run: "compute filtered-knn".into(),
                description: Some("Compute filtered KNN with predicate pre-filtering".into()),
                after,
                per_profile: true,
                phase: 2, // after all evaluate-predicates — base + pred indices I/O phase
                options: {
                    let mut opts = vec![
                        ("base".into(), if slots.base_count.is_some() {
                        format!("{}[0..${{base_count}})", slots.base_vectors.path())
                    } else {
                        slots.base_vectors.path().to_string()
                    }),
                        ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                        ("metadata-indices".into(), "metadata_indices.slab".into()),
                        ("indices".into(), "filtered_neighbor_indices.ivec".into()),
                        ("distances".into(), "filtered_neighbor_distances.fvec".into()),
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
            if slots.query_vectors.as_ref().map(|q| q.is_materialized()).unwrap_or(false) {
                verify_after.push("extract-queries".into());
            }
        }
        steps.push(Step {
            id: "verify-knn".into(),
            run: cmd_verify_knn(&args.personality).into(),
            description: Some("Multi-threaded single-pass KNN verification across all profiles".into()),
            after: verify_after,
            per_profile: false, // NOT per-profile — one step verifies all
            phase: 0,
            options: vec![
                ("base".into(), slots.base_vectors.path().into()),
                ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                ("metric".into(), args.metric.clone()),
                ("normalized".into(), "true".into()),
                ("sample".into(), "100".into()),
                ("seed".into(), format!("${{{}}}", "seed")),
                ("output".into(), "${cache}/verify_knn_consolidated.json".into()),
            ],
        });
    }

    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            steps.push(Step {
                id: "verify-filtered-knn".into(),
                run: "verify filtered-knn-consolidated".into(),
                description: Some("Single-pass filtered KNN verification across all profiles".into()),
                after: vec!["compute-filtered-knn".into()],
                per_profile: false,
                phase: 0,
                options: vec![
                    ("base".into(), slots.base_vectors.path().into()),
                    ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                    ("metadata".into(), slots.metadata.as_ref().unwrap().metadata_content.path().into()),
                    ("predicates".into(), format!("{}predicates.slab", pp)),
                    ("metric".into(), args.metric.clone()),
                    ("normalized".into(), "true".into()),
                    ("sample".into(), "50".into()),
                    ("seed".into(), format!("${{{}}}", "seed")),
                    ("output".into(), "${cache}/verify_filtered_knn_consolidated.json".into()),
                ],
            });

            steps.push(Step {
                id: "verify-predicates".into(),
                run: "verify predicates-consolidated".into(),
                description: Some("Single-pass predicate verification across all profiles".into()),
                after: vec!["evaluate-predicates".into()],
                per_profile: false,
                phase: 0,
                options: vec![
                    ("metadata".into(), slots.metadata.as_ref().unwrap().metadata_content.path().into()),
                    ("predicates".into(), format!("{}predicates.slab", pp)),
                    ("sample".into(), "50".into()),
                    ("metadata-sample".into(), "100000".into()),
                    ("seed".into(), format!("${{{}}}", "seed")),
                    ("output".into(), "${cache}/verify_predicates_consolidated.json".into()),
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
            json_after.push("verify-predicates".into());
        }
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
        options: vec![],
    });
    steps.push(Step {
        id: "generate-dataset-log-jsonl".into(),
        run: "generate dataset-log-jsonl".into(),
        description: Some("Generate dataset.jsonl from dataset.log".into()),
        after: vec!["generate-dataset-json".into()],
        per_profile: false,
        phase: 0,
        options: vec![],
    });

    // ── Merkle hash trees ────────────────────────────────────────
    // Runs before catalog generation so that all data files have
    // .mref hashes before the catalog snapshot is taken.
    steps.push(Step {
        id: "generate-merkle".into(),
        run: "merkle create".into(),
        description: Some("Create merkle hash trees for all publishable data files".into()),
        after: vec!["generate-variables-json".into(), "generate-dataset-log-jsonl".into()],
        per_profile: false,
        phase: 0,
        options: vec![
            ("source".into(), ".".into()),
            ("min-size".into(), "0".into()),
        ],
    });

    // ── Catalog generation ──────────────────────────────────────────
    // MUST be the very last pipeline step — after merkle creation —
    // so the catalog reflects the final dataset state. The catalog
    // covers the full publish tree (all sibling datasets), not just
    // the current one.
    steps.push(Step {
        id: "generate-catalog".into(),
        run: "catalog generate".into(),
        description: Some("Generate catalog index for the dataset directory".into()),
        after: vec!["generate-merkle".into()],
        per_profile: false,
        phase: 0,
        options: vec![
            ("input".into(), ".".into()),
        ],
    });

    steps
}

// ---------------------------------------------------------------------------
// Profile view assembly
// ---------------------------------------------------------------------------

/// Assemble profile views from resolved slot paths.
fn profile_views(slots: &PipelineSlots, args: &ImportArgs, output_dir: &std::path::Path) -> Vec<(String, String)> {
    let mut views = Vec::new();

    views.push(("base_vectors".into(), slots.base_vectors.path().into()));

    if let Some(ref qv) = slots.query_vectors {
        views.push(("query_vectors".into(), qv.path().into()));
    }

    if let Some(ref meta) = slots.metadata {
        views.push(("metadata_content".into(), meta.metadata_content.path().into()));
        views.push(("metadata_predicates".into(), meta.predicates.path().into()));
    }

    if let Some(ref knn) = slots.knn {
        match knn {
            Artifact::Identity { path } => {
                views.push(("neighbor_indices".into(), path.clone()));
                // Include pre-provided distances if available
                if let Some(ref gt_dist) = args.ground_truth_distances {
                    views.push(("neighbor_distances".into(),
                        relativize_path(gt_dist, output_dir)));
                }
            }
            Artifact::Materialized { .. } => {
                views.push(("neighbor_indices".into(), format!("{}neighbor_indices.ivec", args.default_prefix())));
                views.push(("neighbor_distances".into(), format!("{}neighbor_distances.fvec", args.default_prefix())));
            }
        }
    }

    if let Some(ref meta) = slots.metadata {
        views.push(("metadata_indices".into(), format!("{}metadata_indices.slab", args.default_prefix())));
        let _ = &meta.predicate_indices; // used by per_profile steps
    }

    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            views.push(("filtered_neighbor_indices".into(), format!("{}filtered_neighbor_indices.ivec", args.default_prefix())));
            views.push(("filtered_neighbor_distances".into(), format!("{}filtered_neighbor_distances.fvec", args.default_prefix())));
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

    out.push_str("# Copyright (c) nosqlbench contributors\n");
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
    // is_normalized, is_duplicate_vector_free, and is_zero_vector_free
    // are set by the pipeline after the relevant steps complete — not
    // at generation time when correctness hasn't been verified yet.

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

    // Profiles
    out.push_str("profiles:\n");
    out.push_str("  default:\n");
    out.push_str(&format!("    maxk: {}\n", args.neighbors));
    for (facet, path) in views {
        out.push_str(&format!("    {}: {}\n", facet, path));
    }

    // Sized profiles — generate windowed sub-profiles at multiple scales
    if args.sized_profiles.is_some() {
        let spec = args.sized_profiles.as_ref().unwrap();
        out.push_str("\n  sized:\n");
        let specs: Vec<&str> = spec.split(',').map(|s| s.trim()).collect();
        // Add a human-readable comment explaining each expression
        for s in &specs {
            let desc = describe_sized_spec(s);
            out.push_str(&format!("    # \"{}\": {}\n", s, desc));
        }
        let formatted: Vec<String> = specs.iter().map(|s| format!("\"{}\"", s)).collect();
        out.push_str(&format!("    ranges: [{}]\n", formatted.join(", ")));
        out.push_str("    facets:\n");

        // Build sized facets from the default profile views plus any
        // computed per-profile facets needed by sized profiles.
        let mut sized_facets_written = std::collections::HashSet::new();

        for (facet, path) in views.iter() {
            let windowed = facet == "base_vectors" || facet == "metadata_content";
            let per_profile = facet == "neighbor_indices"
                || facet == "neighbor_distances"
                || facet == "filtered_neighbor_indices"
                || facet == "filtered_neighbor_distances"
                || facet == "metadata_indices";

            if windowed {
                out.push_str(&format!("      {}: \"{}:${{range}}\"\n", facet, path));
                sized_facets_written.insert(facet.as_str());
            } else if per_profile {
                if !args.classic && path.contains("profiles/") {
                    let templatized = path.replace("profiles/default/", "profiles/${profile}/");
                    out.push_str(&format!("      {}: \"{}\"\n", facet, templatized));
                    sized_facets_written.insert(facet.as_str());
                }
                // Pre-provided paths (e.g., _gt.ivec) are skipped here —
                // computed equivalents are added below if needed.
            } else {
                out.push_str(&format!("      {}: {}\n", facet, path));
                sized_facets_written.insert(facet.as_str());
            }
        }

        // When compute-knn runs per-profile (for sized profiles), add the
        // computed KNN facets even if the default profile uses pre-provided GT.
        let has_compute_knn = steps.iter().any(|s| s.id == "compute-knn");
        if has_compute_knn {
            if !sized_facets_written.contains("neighbor_indices") {
                let sp = args.sized_prefix();
                out.push_str(&format!("      neighbor_indices: \"{}neighbor_indices.ivec\"\n", sp));
            }
            if !sized_facets_written.contains("neighbor_distances") {
                let sp = args.sized_prefix();
                out.push_str(&format!("      neighbor_distances: \"{}neighbor_distances.fvec\"\n", sp));
            }
        }
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

    // Query vectors: create a query_vectors_raw symlink for native xvec files.
    // The overlap step reads from _raw and writes the final query_vectors.
    // For non-native (HDF5, npy), the convert-queries step handles this.
    if let Some(ref query_path) = args.query_vectors {
        if !slots.self_search && is_native_xvec_file(query_path) {
            let ext = query_path.extension()
                .and_then(|e| e.to_str())
                .and_then(VecFormat::canonical_extension)
                .unwrap_or("fvec");
            let link = base_dir.join(format!("query_vectors_raw.{}", ext));
            create_symlink(query_path, &link);
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

    let sample_count = count.min(100_000);
    let step = if count <= sample_count { 1 } else { count / sample_count };

    let mut norms: Vec<f64> = Vec::with_capacity(sample_count);

    for s in 0..sample_count {
        let idx = s * step;
        if idx >= count { break; }
        let vec = get_f64(idx);
        let sum: f64 = vec.iter().map(|v| v * v).sum();
        norms.push(sum.sqrt());
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
            classic: false,
            personality: "native".to_string(),
            synthesize_metadata: false,
            metadata_fields: 3,
            metadata_range_min: 0,
            metadata_range_max: 1000,
        }
    }

    // SRD §12.5 Example 1: Minimal — native fvec, no queries, no metadata
    // Expects: import collapses to identity, self-search activates,
    //          dedup + shuffle + extract + KNN materialize
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
        // Expected: set-vector-count, sort-vectors, set-duplicate-count,
        // zero-check, set-zero-count, clean-ordinals, set-clean-count,
        // count-source-base, set-is_shuffled, set-is_self_search, set-combined_bq, set-k,
        // shuffle, extract-query, extract-base, count-base,
        // compute-knn, verify-knn, generate-dataset-json, generate-variables-json,
        // generate-dataset-log-jsonl, generate-merkle, generate-catalog
        assert_eq!(steps.len(), 19, "steps: {:?}", steps.iter().map(|s| &s.id).collect::<Vec<_>>());
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
        assert!(runs.contains(&"compute knn"), "native should use 'compute knn': {:?}", runs);
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
