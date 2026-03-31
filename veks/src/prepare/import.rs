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

/// The complete resolved pipeline graph.
struct PipelineSlots {
    // Vector chain
    all_vectors: Artifact,
    sort: Artifact,
    zero_check: Artifact,
    clean_ordinals: Artifact,
    vector_count: Artifact,

    // Query chain
    self_search: bool,
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
/// All input paths are relativized to the output directory so that
/// dataset.yaml never contains absolute paths (SRD requirement).
fn resolve_slots(args: &ImportArgs) -> PipelineSlots {
    let output_dir = &args.output;
    let facets = resolve_facets(args);
    let has_base_facet = facets.contains('B');

    let base_source = args.base_vectors.as_ref()
        .map(|p| relativize_path(p, output_dir))
        .unwrap_or_default();

    // ── Vector chain ─────────────────────────────────────────────────
    let needs_import = has_base_facet && args.base_vectors.as_ref()
        .map(|p| !is_native_xvec_file(p))
        .unwrap_or(false);

    let all_vectors = if needs_import {
        Artifact::Materialized {
            step_id: "convert-vectors".into(),
            output: "${cache}/all_vectors.mvec".into(),
        }
    } else {
        Artifact::Identity { path: base_source.clone() }
    };

    let sort = if !has_base_facet || args.no_dedup {
        Artifact::Identity { path: String::new() } // no artifact
    } else {
        Artifact::Materialized {
            step_id: "sort-and-dedup".into(),
            output: "${cache}/sorted_ordinals.ivec".into(),
        }
    };

    let zero_check = if !has_base_facet || args.no_zero_check {
        Artifact::Identity { path: String::new() }
    } else {
        Artifact::Materialized {
            step_id: "find-zeros".into(),
            output: "${cache}/zero_ordinals.ivec".into(),
        }
    };

    let clean_ordinals = if !has_base_facet || (args.no_dedup && args.no_zero_check) {
        Artifact::Identity { path: String::new() }
    } else {
        Artifact::Materialized {
            step_id: "filter-ordinals".into(),
            output: "${cache}/clean_ordinals.ivec".into(),
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
    // Self-search when queries are wanted but not provided separately
    let self_search = wants_queries && !has_separate_query;
    let has_queries = has_separate_query || self_search;

    // Determine output extension from source format
    let vec_ext = if needs_import { "mvec" } else {
        args.base_vectors.as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .unwrap_or("fvec")
    };

    let (shuffle, query_vectors, base_vectors, base_count) = if self_search {
        // Self-search: shuffle + extract
        let shuffle = Artifact::Materialized {
            step_id: "generate-shuffle".into(),
            output: "${cache}/shuffle.ivec".into(),
        };
        let qv = Artifact::Materialized {
            step_id: "extract-queries".into(),
            output: format!("profiles/base/query_vectors.{}", vec_ext),
        };
        let bv = Artifact::Materialized {
            step_id: "extract-base".into(),
            output: format!("profiles/base/base_vectors.{}", vec_ext),
        };
        let bc = Artifact::Materialized {
            step_id: "count-base".into(),
            output: String::new(),
        };
        (Some(shuffle), Some(qv), bv, Some(bc))
    } else if has_separate_query {
        // Separate query file
        let query_source = args.query_vectors.as_ref().unwrap();
        let qv = if is_native_xvec_file(query_source) {
            let ext = query_source.extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_else(|| "fvec".to_string());
            Artifact::Identity { path: format!("profiles/base/query_vectors.{}", ext) }
        } else {
            Artifact::Materialized {
                step_id: "convert-queries".into(),
                output: format!("query_vectors.{}", vec_ext),
            }
        };
        // Base: when import was needed (HDF5, npy, etc.), extract from
        // the converted all_vectors to the profile path. When native xvec,
        // use a symlink (Identity).
        let bv = if needs_import {
            Artifact::Materialized {
                step_id: "extract-base".into(),
                output: format!("profiles/base/base_vectors.{}", vec_ext),
            }
        } else {
            Artifact::Identity { path: format!("profiles/base/base_vectors.{}", vec_ext) }
        };
        (None, Some(qv), bv, None)
    } else {
        // No queries at all — use canonical profile path (symlinked)
        let bv = Artifact::Identity { path: format!("profiles/base/base_vectors.{}", vec_ext) };
        (None, None, bv, None)
    };

    // ── Metadata chain ───────────────────────────────────────────────
    let wants_metadata = facets.contains('M');
    let metadata = args.metadata.as_ref().filter(|_| wants_metadata).map(|meta_source| {
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
                output: "profiles/base/metadata_content.slab".into(),
            }
        } else {
            // Canonical profile path — symlinked to source during import
            Artifact::Identity { path: "profiles/base/metadata_content.slab".into() }
        };

        let survey = Artifact::Materialized {
            step_id: "survey-metadata".into(),
            output: "${cache}/metadata_survey.json".into(),
        };
        let predicates = Artifact::Materialized {
            step_id: "generate-predicates".into(),
            output: "profiles/base/predicates.slab".into(),
        };
        let predicate_indices = Artifact::Materialized {
            step_id: "evaluate-predicates".into(),
            output: "metadata_indices.slab".into(),
        };

        MetadataSlots { metadata_all, metadata_content, survey, predicates, predicate_indices }
    });

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
        all_vectors, sort, zero_check, clean_ordinals, vector_count,
        self_search, shuffle, query_vectors, base_vectors, base_count,
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

/// Walk the resolved slots and emit pipeline steps for materialized slots.
fn emit_steps(slots: &PipelineSlots, args: &ImportArgs, _output_dir: &std::path::Path) -> Vec<Step> {
    let mut steps = Vec::new();
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
            ("output".into(), "${cache}/all_vectors.mvec".into()),
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

    // sort (lexicographic sort + duplicate detection as byproduct)
    if let Artifact::Materialized { .. } = &slots.sort {
        steps.push(Step {
            id: "sort-and-dedup".into(),
            run: "compute sort".into(),
            description: Some("Lexicographic sort producing sorted ordinals + duplicate report".into()),
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

    // set-duplicate-count (after sort, records number of elided duplicates)
    if slots.sort.is_materialized() {
        steps.push(Step {
            id: "count-duplicates".into(),
            run: "state set".into(),
            description: None,
            after: vec!["sort-and-dedup".into()],
            per_profile: false,
            phase: 0,
            options: vec![
                ("name".into(), "duplicate_count".into()),
                ("value".into(), "count:${cache}/dedup_duplicates.ivec".into()),
            ],
        });
    }

    // zero-check
    if let Artifact::Materialized { .. } = &slots.zero_check {
        let mut after = vec![];
        if slots.sort.is_materialized() {
            after.push("sort-and-dedup".into());
        } else if !last_vector_step.is_empty() {
            after.push(last_vector_step.into());
        }
        steps.push(Step {
            id: "find-zeros".into(),
            run: "analyze zeros".into(),
            description: Some("Binary search sorted index for zero vector".into()),
            after,
            per_profile: false,
            phase: 0,
            options: vec![
                ("source".into(), working_vectors.clone()),
                ("index".into(), slots.sort.path().into()),
                ("output".into(), "${cache}/zero_ordinals.ivec".into()),
            ],
        });
    }

    // set-zero-count (after zero-check, records number of zero vectors removed)
    if slots.zero_check.is_materialized() {
        steps.push(Step {
            id: "count-zeros".into(),
            run: "state set".into(),
            description: None,
            after: vec!["find-zeros".into()],
            per_profile: false,
            phase: 0,
            options: vec![
                ("name".into(), "zero_count".into()),
                ("value".into(), "count:${cache}/zero_ordinals.ivec".into()),
            ],
        });
    }

    // clean-ordinals
    if let Artifact::Materialized { .. } = &slots.clean_ordinals {
        let mut after = vec![];
        if slots.sort.is_materialized() {
            after.push("sort-and-dedup".into());
        }
        if slots.zero_check.is_materialized() {
            after.push("find-zeros".into());
        }
        steps.push(Step {
            id: "filter-ordinals".into(),
            run: "transform ordinals".into(),
            description: Some("Filter sorted ordinals excluding duplicates and zeros".into()),
            after,
            per_profile: false,
            phase: 0,
            options: vec![
                ("source".into(), slots.sort.path().into()),
                ("duplicates".into(), "${cache}/dedup_duplicates.ivec".into()),
                ("zeros".into(), slots.zero_check.path().into()),
                ("output".into(), "${cache}/clean_ordinals.ivec".into()),
            ],
        });
    }

    // set-clean-count (after clean-ordinals, used by shuffle)
    if slots.clean_ordinals.is_materialized() {
        steps.push(Step {
            id: "count-clean".into(),
            run: "state set".into(),
            description: None,
            after: vec!["filter-ordinals".into()],
            per_profile: false,
            phase: 0,
            options: vec![
                ("name".into(), "clean_count".into()),
                ("value".into(), "count:${cache}/clean_ordinals.ivec".into()),
            ],
        });
    }

    // Was the subset already applied by the subset-vectors or convert step?
    // When true, clean_count is already fractioned — don't apply base_end.
    let subset_applied = last_vector_step == "subset-vectors"
        || (slots.all_vectors.is_materialized() && args.base_fraction < 1.0 && !args.pedantic_dedup);

    // ── Query chain (self-search) ────────────────────────────────────
    if slots.self_search {
        if let Some(ref shuffle) = slots.shuffle {
            if shuffle.is_materialized() {
                // Shuffle depends on clean_count (not vector_count) so it
                // creates a permutation over only the clean ordinals.
                let shuffle_after = if slots.clean_ordinals.is_materialized() {
                    vec!["count-clean".into()]
                } else {
                    vec!["count-vectors".into()]
                };
                let shuffle_interval = if slots.clean_ordinals.is_materialized() {
                    "${clean_count}".to_string()
                } else {
                    "${vector_count}".to_string()
                };
                let mut shuffle_opts = vec![
                    ("output".into(), "${cache}/shuffle.ivec".into()),
                    ("interval".into(), shuffle_interval),
                    ("seed".into(), format!("${{{}}}", "seed")),
                ];
                // When clean_ordinals exists, pass it so the shuffle
                // permutes actual source ordinals (not [0, N) indices).
                // This ensures the extract steps get valid source ordinals
                // that exclude duplicates and zero vectors.
                if slots.clean_ordinals.is_materialized() {
                    shuffle_opts.push(("ordinals".into(), slots.clean_ordinals.path().into()));
                }
                steps.push(Step {
                    id: "generate-shuffle".into(),
                    run: "generate shuffle".into(),
                    description: Some("Reproducible random split via Fisher-Yates shuffle".into()),
                    after: shuffle_after,
                    per_profile: false,
                    phase: 0,
                    options: shuffle_opts,
                });
            }
        }

        if let Some(ref qv) = slots.query_vectors {
            if qv.is_materialized() {
                let vec_deps = if last_vector_step.is_empty() {
                    vec!["generate-shuffle".into()]
                } else {
                    vec![last_vector_step.into(), "generate-shuffle".into()]
                };
                let total_var = if slots.clean_ordinals.is_materialized() {
                    "${clean_count}"
                } else {
                    "${vector_count}"
                };

                // When base_fraction < 1.0, compute a capped base_end — BUT
                // only if the subset step didn't already apply the fraction.
                // If subset-vectors ran, clean_count is already fractioned.
                let base_upper = if args.base_fraction < 1.0 && !subset_applied {
                    let count_dep = if slots.clean_ordinals.is_materialized() {
                        "count-clean"
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
                    ("output".into(), slots.query_vectors.as_ref().map(|q| q.path().to_string()).unwrap_or_else(|| "profiles/base/query_vectors.fvec".into())),
                    ("range".into(), "[0,${query_count})".into()),
                ];
                if args.normalize {
                    query_opts.push(("normalize".into(), "true".into()));
                }
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
                let mut base_opts = vec![
                    ("source".into(), working_vectors.clone()),
                    ("ivec-file".into(), "${cache}/shuffle.ivec".into()),
                    ("output".into(), slots.base_vectors.path().into()),
                    ("range".into(), format!("[${{query_count}},{})", base_upper)),
                ];
                if args.normalize {
                    base_opts.push(("normalize".into(), "true".into()));
                }
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
                // Measure normalization precision after base vectors are clean
                steps.push(Step {
                    id: "measure-normals".into(),
                    run: "analyze measure-normals".into(),
                    description: Some("Measure L2 normalization precision at f64".into()),
                    after: vec!["extract-base".into()],
                    per_profile: false,
                    phase: 0,
                    options: vec![
                        ("input".into(), slots.base_vectors.path().into()),
                        ("sample".into(), "10000".into()),
                        ("seed".into(), format!("${{{}}}", "seed")),
                    ],
                });
            }
        }
    } else if let Some(ref qv) = slots.query_vectors {
        // Separate query import
        if qv.is_materialized() {
            let query_source = args.query_vectors.as_ref().unwrap();
            let format = VecFormat::detect(query_source)
                .map(|f| f.name().to_string())
                .unwrap_or_else(|| "auto".into());
            steps.push(Step {
                id: "convert-queries".into(),
                run: "transform convert".into(),
                description: Some("Import query vectors from source format".into()),
                after: vec![],
                per_profile: false,
                phase: 0,
                options: vec![
                    ("facet".into(), "query_vectors".into()),
                    ("source".into(), relativize_path(query_source, &args.output)),
                    ("from".into(), format),
                    ("output".into(), qv.path().into()),
                ],
            });
        }
        // When base needed import (HDF5, npy, etc.) with separate queries,
        // extract the converted all_vectors to the profile path.
        if slots.base_vectors.is_materialized() {
            let working = slots.all_vectors.path();
            let after = if slots.clean_ordinals.is_materialized() {
                vec!["filter-ordinals".into()]
            } else if slots.all_vectors.is_materialized() {
                vec!["convert-vectors".into()]
            } else {
                vec![]
            };
            let mut base_opts = vec![
                ("source".into(), working.into()),
                ("output".into(), slots.base_vectors.path().into()),
            ];
            if slots.clean_ordinals.is_materialized() {
                base_opts.push(("ivec-file".into(), slots.clean_ordinals.path().into()));
                base_opts.push(("range".into(), format!("[0,{})", "${vector_count}")));
            }
            if args.normalize {
                base_opts.push(("normalize".into(), "true".into()));
            }
            steps.push(Step {
                id: "extract-base".into(),
                run: "transform extract".into(),
                description: Some("Extract base vectors to profile".into()),
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
            // Measure normalization
            steps.push(Step {
                id: "measure-normals".into(),
                run: "analyze measure-normals".into(),
                description: Some("Measure L2 normalization precision at f64".into()),
                after: vec!["extract-base".into()],
                per_profile: false,
                phase: 0,
                options: vec![
                    ("input".into(), slots.base_vectors.path().into()),
                    ("sample".into(), "10000".into()),
                    ("seed".into(), format!("${{{}}}", "seed")),
                ],
            });
        }
    }

    // ── Metadata chain ───────────────────────────────────────────────
    if let Some(ref meta) = slots.metadata {
        let meta_source = args.metadata.as_ref().unwrap();

        if meta.metadata_all.is_materialized() {
            let format = VecFormat::detect(meta_source)
                .map(|f| f.name().to_string())
                .unwrap_or_else(|| "parquet".into());
            let mut meta_opts = vec![
                ("facet".into(), "metadata_content".into()),
                ("source".into(), relativize_path(meta_source, &args.output)),
                ("from".into(), format),
                ("output".into(), "${cache}/metadata_all.slab".into()),
            ];
            // When base_fraction < 1.0, only import metadata for the same
            // ordinal range as the base vectors. This preserves pairwise
            // ordinal alignment and avoids importing 100x more metadata
            // than needed.
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
        }

        if meta.metadata_content.is_materialized() {
            let mut after = vec![];
            if meta.metadata_all.is_materialized() {
                after.push("convert-metadata".into());
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
                    ("output".into(), "profiles/base/metadata_content.slab".into()),
                    ("range".into(), if args.base_fraction < 1.0 && !subset_applied {
                        "[${query_count},${base_end})".into()
                    } else if slots.clean_ordinals.is_materialized() {
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
                ("output".into(), "profiles/base/predicates.slab".into()),
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
                ("predicates".into(), "profiles/base/predicates.slab".into()),
                ("survey".into(), "${cache}/metadata_survey.json".into()),
                ("selectivity".into(), args.selectivity.to_string()),
                ("range".into(), if slots.base_count.is_some() {
                    "[0,${base_end})".into()
                } else if slots.clean_ordinals.is_materialized() {
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
            steps.push(Step {
                id: "compute-knn".into(),
                run: "compute knn".into(),
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
                        ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                        ("indices".into(), "neighbor_indices.ivec".into()),
                        ("distances".into(), "neighbor_distances.fvec".into()),
                        ("neighbors".into(), args.neighbors.to_string()),
                        ("metric".into(), args.metric.clone()),
                    ];
                    if args.compress_cache {
                        opts.push(("compress_cache".into(), "true".into()));
                    }
                    if args.normalize {
                        opts.push(("normalized".into(), "true".into()));
                    }
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
            run: "verify knn-consolidated".into(),
            description: Some("Multi-threaded single-pass KNN verification across all profiles".into()),
            after: verify_after,
            per_profile: false, // NOT per-profile — one step verifies all
            phase: 0,
            options: vec![
                ("base".into(), slots.base_vectors.path().into()),
                ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                ("metric".into(), args.metric.clone()),
                ("normalized".into(), if args.normalize { "true" } else { "false" }.into()),
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
                    ("predicates".into(), "profiles/base/predicates.slab".into()),
                    ("metric".into(), args.metric.clone()),
                    ("normalized".into(), if args.normalize { "true" } else { "false" }.into()),
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
                    ("predicates".into(), "profiles/base/predicates.slab".into()),
                    ("sample".into(), "50".into()),
                    ("metadata-sample".into(), "100000".into()),
                    ("seed".into(), format!("${{{}}}", "seed")),
                    ("output".into(), "${cache}/verify_predicates_consolidated.json".into()),
                ],
            });
        }
    }

    // ── Catalog generation ──────────────────────────────────────────
    // Generates catalog.json and catalog.yaml for the local dataset
    // directory so the dataset is discoverable by catalog queries.
    let mut catalog_after = Vec::new();
    if let Some(ref knn) = slots.knn {
        if knn.is_materialized() || needs_computed_knn {
            catalog_after.push("verify-knn".into());
        }
    }
    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            catalog_after.push("verify-filtered-knn".into());
            catalog_after.push("verify-predicates".into());
        }
    }
    if catalog_after.is_empty() {
        if let Some(last) = steps.last() {
            catalog_after.push(last.id.clone());
        }
    }
    steps.push(Step {
        id: "generate-catalog".into(),
        run: "catalog generate".into(),
        description: Some("Generate catalog index for the dataset directory".into()),
        after: catalog_after,
        per_profile: false,
        phase: 0, // not per-profile, depends on verify steps via after
        options: vec![
            ("input".into(), ".".into()),
        ],
    });

    // ── Merkle hash trees ────────────────────────────────────────
    // MUST be the very last pipeline step — after catalog generation —
    // so that catalog.json and catalog.yaml get .mref files too.
    steps.push(Step {
        id: "generate-merkle".into(),
        run: "merkle create".into(),
        description: Some("Create merkle hash trees for all publishable data files".into()),
        after: vec!["generate-catalog".into()],
        per_profile: false,
        phase: 0,
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
                views.push(("neighbor_indices".into(), "profiles/default/neighbor_indices.ivec".into()));
                views.push(("neighbor_distances".into(), "profiles/default/neighbor_distances.fvec".into()));
            }
        }
    }

    if let Some(ref meta) = slots.metadata {
        views.push(("metadata_indices".into(), "profiles/default/metadata_indices.slab".into()));
        let _ = &meta.predicate_indices; // used by per_profile steps
    }

    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            views.push(("filtered_neighbor_indices".into(), "profiles/default/filtered_neighbor_indices.ivec".into()));
            views.push(("filtered_neighbor_distances".into(), "profiles/default/filtered_neighbor_distances.fvec".into()));
        }
    }

    views
}

// ---------------------------------------------------------------------------
// YAML generation
// ---------------------------------------------------------------------------

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
                if path.contains("profiles/") {
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
                out.push_str("      neighbor_indices: \"profiles/${profile}/neighbor_indices.ivec\"\n");
            }
            if !sized_facets_written.contains("neighbor_distances") {
                out.push_str("      neighbor_distances: \"profiles/${profile}/neighbor_distances.fvec\"\n");
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
                if args.normalize {
                    println!("  --normalize specified but vectors are already normalized; normalization will be skipped.");
                }
            } else {
                println!("Vectors are NOT L2-normalized (mean norm={:.4}, n={})", mean_norm, sample_count);
                if args.normalize {
                    println!("  Normalization will be applied during extraction.");
                } else {
                    println!("  Consider using --normalize to L2-normalize during extraction.");
                }
            }
            println!();
        }
    }

    // Resolve all slots
    let slots = resolve_slots(&args);

    // Report what was resolved
    println!("Resolving pipeline slots:");
    print_slot("all_vectors", &slots.all_vectors);
    print_slot("sort", &slots.sort);
    print_slot("zero_check", &slots.zero_check);
    print_slot("clean_ordinals", &slots.clean_ordinals);
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
    create_identity_symlinks(output_dir, &args);

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
/// (Identity artifacts). This ensures the canonical profile structure exists
/// even when no import/extract step runs, so all profile views point to
/// paths under `profiles/base/` rather than raw source paths.
fn create_identity_symlinks(output_dir: &std::path::Path, args: &ImportArgs) {
    let base_dir = output_dir.join("profiles/base");
    if let Err(e) = std::fs::create_dir_all(&base_dir) {
        eprintln!("Warning: failed to create profiles/base/: {}", e);
        return;
    }

    let self_search = args.query_vectors.is_none()
        && (args.self_search || args.base_vectors.is_some());

    // Base vectors: symlink when not self-search (identity to source)
    if !self_search {
        if let Some(ref base_path) = args.base_vectors {
            if is_native_xvec_file(base_path) {
                let ext = base_path.extension()
                    .map(|e| e.to_string_lossy().to_string())
                    .unwrap_or_else(|| "mvec".to_string());
                let link = base_dir.join(format!("base_vectors.{}", ext));
                create_symlink(base_path, &link);
            }
        }
    }

    // Query vectors: symlink when separate native xvec query file
    if args.query_vectors.is_some() {
        let query_path = args.query_vectors.as_ref().unwrap();
        if is_native_xvec_file(query_path) {
            let ext = query_path.extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_else(|| "fvec".to_string());
            let link = base_dir.join(format!("query_vectors.{}", ext));
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
fn create_symlink(target: &std::path::Path, link: &std::path::Path) {
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
pub fn relative_path(base: &std::path::Path, target: &std::path::Path) -> std::path::PathBuf {
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
    log_slot(&mut w, "sort", &slots.sort);
    log_slot(&mut w, "zero_check", &slots.zero_check);
    log_slot(&mut w, "clean_ordinals", &slots.clean_ordinals);
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

    let sample_count = count.min(1000);
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
/// Heuristic:
/// - If vectors are L2-normalized → Cosine (norms ≈ 1.0)
/// - Otherwise → Cosine (safest default; L2 and Cosine rankings are
///   identical for normalized data, and Cosine is more commonly expected)
///
/// Returns the metric name and a human-readable reason.
pub fn detect_metric(path: &Path) -> (String, String) {
    // For directories (npy), try to probe via a small temp conversion
    let probe_result = if path.is_dir() {
        probe_directory_vectors(path)
    } else {
        detect_normalized(path)
    };
    if let Some((is_normalized, _n, mean_norm)) = probe_result {
        if is_normalized {
            ("DotProduct".to_string(), format!("vectors are L2-normalized (mean norm={:.4}) — DotProduct is optimal", mean_norm))
        } else {
            ("Cosine".to_string(), format!("vectors not normalized (mean norm={:.4})", mean_norm))
        }
    } else {
        let detail = if path.is_dir() {
            "directory probe returned no results"
        } else if !path.exists() {
            "path does not exist"
        } else {
            "unsupported format for probing"
        };
        ("Cosine".to_string(), format!("default — {} ({})", detail, path.display()))
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
    if !slots.sort.is_materialized() { n += 1; }
    if !slots.zero_check.is_materialized() { n += 1; }
    if !slots.clean_ordinals.is_materialized() { n += 1; }
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
        assert!(slots.sort.is_materialized());

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
        // shuffle, extract-query, extract-base, set-base-count,
        // compute-knn, verify-knn, merkle-all, catalog-generate
        assert_eq!(steps.len(), 15, "steps: {:?}", steps.iter().map(|s| &s.id).collect::<Vec<_>>());
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(step_ids.contains(&"count-duplicates"), "should have set-duplicate-count");
        assert!(step_ids.contains(&"count-zeros"), "should have set-zero-count");
    }

    // SRD §12.5 Example 3: Native base + separate native query + metadata dir
    // Expects: import/shuffle/extract all collapse, metadata import materializes,
    //          metadata extract collapses (no shuffle), compute steps materialize
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

        // all_vectors, query_vectors, base_vectors should be Identity
        assert!(!slots.all_vectors.is_materialized());
        assert!(!slots.query_vectors.as_ref().unwrap().is_materialized());
        assert!(!slots.base_vectors.is_materialized());

        // No shuffle (separate query)
        assert!(!slots.self_search);
        assert!(slots.shuffle.is_none());

        // Metadata: import materializes (parquet dir), extract is Identity (no shuffle)
        let meta = slots.metadata.as_ref().unwrap();
        assert!(meta.metadata_all.is_materialized());
        assert!(!meta.metadata_content.is_materialized(), "no shuffle → metadata_content should be identity");

        // KNN + filtered KNN materialize
        assert!(slots.knn.as_ref().unwrap().is_materialized());
        assert!(slots.filtered_knn.as_ref().unwrap().is_materialized());

        let steps = emit_steps(&slots, &args, &args.output);
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(step_ids.contains(&"convert-metadata"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"survey-metadata"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"compute-knn"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"compute-filtered-knn"), "steps: {:?}", step_ids);
        // No shuffle/extract steps
        assert!(!step_ids.contains(&"generate-shuffle"));
        assert!(!step_ids.contains(&"extract-queries"));
        assert!(!step_ids.contains(&"extract-base"));
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
        assert!(!slots.sort.is_materialized(), "sort should be identity when --no-dedup");

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
}
