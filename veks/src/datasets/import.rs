// Copyright (c) DataStax, Inc.
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
    pub no_filtered: bool,
    pub force: bool,
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
    dedup: Artifact,
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
fn resolve_slots(args: &ImportArgs) -> PipelineSlots {
    let base_source = args.base_vectors.as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    // ── Vector chain ─────────────────────────────────────────────────
    let needs_import = args.base_vectors.as_ref()
        .map(|p| !is_native_xvec_file(p))
        .unwrap_or(false);

    let all_vectors = if needs_import {
        Artifact::Materialized {
            step_id: "import-vectors".into(),
            output: "${cache}/all_vectors.mvec".into(),
        }
    } else {
        Artifact::Identity { path: base_source.clone() }
    };

    let dedup = if args.no_dedup {
        Artifact::Identity { path: String::new() } // no artifact
    } else {
        Artifact::Materialized {
            step_id: "dedup-vectors".into(),
            output: "${cache}/dedup_ordinals.ivec".into(),
        }
    };

    let vector_count = Artifact::Materialized {
        step_id: "set-vector-count".into(),
        output: String::new(), // variable, not a file
    };

    // ── Query chain ──────────────────────────────────────────────────
    let has_separate_query = args.query_vectors.is_some();
    // Separate query file takes precedence over --self-search flag
    let self_search = !has_separate_query
        && (args.self_search || args.base_vectors.is_some());
    let has_queries = has_separate_query || self_search;

    let (shuffle, query_vectors, base_vectors, base_count) = if self_search {
        // Self-search: shuffle + extract
        let shuffle = Artifact::Materialized {
            step_id: "shuffle-ordinals".into(),
            output: "${cache}/shuffle.ivec".into(),
        };
        let qv = Artifact::Materialized {
            step_id: "extract-query-vectors".into(),
            output: "profiles/base/query_vectors.mvec".into(),
        };
        let bv = Artifact::Materialized {
            step_id: "extract-base-vectors".into(),
            output: "profiles/base/base_vectors.mvec".into(),
        };
        let bc = Artifact::Materialized {
            step_id: "set-base-count".into(),
            output: String::new(),
        };
        (Some(shuffle), Some(qv), bv, Some(bc))
    } else if has_separate_query {
        // Separate query file
        let query_source = args.query_vectors.as_ref().unwrap();
        let qv = if is_native_xvec_file(query_source) {
            Artifact::Identity { path: query_source.to_string_lossy().to_string() }
        } else {
            Artifact::Materialized {
                step_id: "import-query".into(),
                output: "query_vectors.fvec".into(),
            }
        };
        // Base = all_vectors (identity)
        let bv = Artifact::Identity { path: all_vectors.path().to_string() };
        (None, Some(qv), bv, None)
    } else {
        // No queries at all
        let bv = Artifact::Identity { path: all_vectors.path().to_string() };
        (None, None, bv, None)
    };

    // ── Metadata chain ───────────────────────────────────────────────
    let metadata = args.metadata.as_ref().map(|meta_source| {
        let needs_meta_import = !is_native_slab_file(meta_source);

        let metadata_all = if needs_meta_import {
            Artifact::Materialized {
                step_id: "import-metadata".into(),
                output: "${cache}/metadata_all.slab".into(),
            }
        } else {
            Artifact::Identity { path: meta_source.to_string_lossy().to_string() }
        };

        // Metadata extract: needed only in self-search (ordinal realignment)
        let metadata_content = if self_search {
            Artifact::Materialized {
                step_id: "extract-metadata".into(),
                output: "profiles/base/metadata_content.slab".into(),
            }
        } else {
            Artifact::Identity { path: metadata_all.path().to_string() }
        };

        let survey = Artifact::Materialized {
            step_id: "survey-metadata".into(),
            output: "${cache}/metadata_survey.json".into(),
        };
        let predicates = Artifact::Materialized {
            step_id: "synthesize-predicates".into(),
            output: "predicates.slab".into(),
        };
        let predicate_indices = Artifact::Materialized {
            step_id: "evaluate-predicates".into(),
            output: "metadata_indices.slab".into(),
        };

        MetadataSlots { metadata_all, metadata_content, survey, predicates, predicate_indices }
    });

    // ── Ground truth chain ───────────────────────────────────────────
    let knn = if !has_queries {
        None
    } else if args.ground_truth.is_some() {
        Some(Artifact::Identity {
            path: args.ground_truth.as_ref().unwrap().to_string_lossy().to_string(),
        })
    } else {
        Some(Artifact::Materialized {
            step_id: "compute-knn".into(),
            output: "neighbor_indices.ivec".into(), // per_profile
        })
    };

    let filtered_knn = if !has_queries || metadata.is_none() || args.no_filtered {
        None
    } else {
        Some(Artifact::Materialized {
            step_id: "compute-filtered-knn".into(),
            output: "filtered_neighbor_indices.ivec".into(), // per_profile
        })
    };

    PipelineSlots {
        all_vectors, dedup, vector_count,
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
    options: Vec<(String, String)>,
}

/// Walk the resolved slots and emit pipeline steps for materialized slots.
fn emit_steps(slots: &PipelineSlots, args: &ImportArgs) -> Vec<Step> {
    let mut steps = Vec::new();
    let base_source = args.base_vectors.as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    // ── Vector chain ─────────────────────────────────────────────────
    if let Artifact::Materialized { .. } = &slots.all_vectors {
        let format = args.base_vectors.as_ref()
            .and_then(|p| VecFormat::detect(p))
            .map(|f| f.name().to_string())
            .unwrap_or_else(|| "auto".into());
        steps.push(Step {
            id: "import-vectors".into(),
            run: "import".into(),
            description: Some("Import base vectors from source format".into()),
            after: vec![],
            per_profile: false,
            options: vec![
                ("facet".into(), "base_vectors".into()),
                ("source".into(), base_source.clone()),
                ("from".into(), format),
                ("output".into(), "${cache}/all_vectors.mvec".into()),
            ],
        });
    }

    let import_id = if slots.all_vectors.is_materialized() {
        "import-vectors"
    } else {
        "" // no dependency
    };

    // set-vector-count (always)
    steps.push(Step {
        id: "set-vector-count".into(),
        run: "set variable".into(),
        description: None,
        after: if import_id.is_empty() { vec![] } else { vec![import_id.into()] },
        per_profile: false,
        options: vec![
            ("name".into(), "vector_count".into()),
            ("value".into(), format!("count:{}", slots.all_vectors.path())),
        ],
    });

    // dedup
    if let Artifact::Materialized { .. } = &slots.dedup {
        steps.push(Step {
            id: "dedup-vectors".into(),
            run: "compute dedup".into(),
            description: Some("Sort-based duplicate detection and optional elision".into()),
            after: if import_id.is_empty() { vec![] } else { vec![import_id.into()] },
            per_profile: false,
            options: vec![
                ("source".into(), slots.all_vectors.path().into()),
                ("output".into(), "${cache}/dedup_ordinals.ivec".into()),
                ("report".into(), "${cache}/dedup_report.json".into()),
                ("elide".into(), "true".into()),
            ],
        });
    }

    // ── Query chain (self-search) ────────────────────────────────────
    if slots.self_search {
        if let Some(ref shuffle) = slots.shuffle {
            if shuffle.is_materialized() {
                steps.push(Step {
                    id: "shuffle-ordinals".into(),
                    run: "generate ivec-shuffle".into(),
                    description: Some("Reproducible random split via Fisher-Yates shuffle".into()),
                    after: vec!["set-vector-count".into()],
                    per_profile: false,
                    options: vec![
                        ("output".into(), "${cache}/shuffle.ivec".into()),
                        ("interval".into(), "${vector_count}".into()),
                        ("seed".into(), format!("${{{}}}", "seed")),
                    ],
                });
            }
        }

        if let Some(ref qv) = slots.query_vectors {
            if qv.is_materialized() {
                let vec_deps = if import_id.is_empty() {
                    vec!["shuffle-ordinals".into()]
                } else {
                    vec![import_id.into(), "shuffle-ordinals".into()]
                };
                steps.push(Step {
                    id: "extract-query-vectors".into(),
                    run: "transform mvec-extract".into(),
                    description: Some(format!("First {} shuffled vectors -> query set", args.query_count)),
                    after: vec_deps.clone(),
                    per_profile: false,
                    options: vec![
                        ("mvec-file".into(), slots.all_vectors.path().into()),
                        ("ivec-file".into(), "${cache}/shuffle.ivec".into()),
                        ("output".into(), "profiles/base/query_vectors.mvec".into()),
                        ("range".into(), "[0,${query_count})".into()),
                    ],
                });
                steps.push(Step {
                    id: "extract-base-vectors".into(),
                    run: "transform mvec-extract".into(),
                    description: Some("Remainder of shuffled vectors -> base set".into()),
                    after: vec_deps,
                    per_profile: false,
                    options: vec![
                        ("mvec-file".into(), slots.all_vectors.path().into()),
                        ("ivec-file".into(), "${cache}/shuffle.ivec".into()),
                        ("output".into(), "profiles/base/base_vectors.mvec".into()),
                        ("range".into(), "[${query_count},${vector_count})".into()),
                    ],
                });
                steps.push(Step {
                    id: "set-base-count".into(),
                    run: "set variable".into(),
                    description: None,
                    after: vec!["extract-base-vectors".into()],
                    per_profile: false,
                    options: vec![
                        ("name".into(), "base_count".into()),
                        ("value".into(), "count:profiles/base/base_vectors.mvec".into()),
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
                id: "import-query".into(),
                run: "import".into(),
                description: Some("Import query vectors from source format".into()),
                after: vec![],
                per_profile: false,
                options: vec![
                    ("facet".into(), "query_vectors".into()),
                    ("source".into(), query_source.to_string_lossy().to_string()),
                    ("from".into(), format),
                    ("output".into(), "query_vectors.fvec".into()),
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
            steps.push(Step {
                id: "import-metadata".into(),
                run: "import".into(),
                description: Some("Import metadata from source format".into()),
                after: vec![],
                per_profile: false,
                options: vec![
                    ("facet".into(), "metadata_content".into()),
                    ("source".into(), meta_source.to_string_lossy().to_string()),
                    ("from".into(), format),
                    ("output".into(), "${cache}/metadata_all.slab".into()),
                ],
            });
        }

        if meta.metadata_content.is_materialized() {
            let mut after = vec![];
            if meta.metadata_all.is_materialized() {
                after.push("import-metadata".into());
            }
            if slots.shuffle.as_ref().map(|s| s.is_materialized()).unwrap_or(false) {
                after.push("shuffle-ordinals".into());
            }
            steps.push(Step {
                id: "extract-metadata".into(),
                run: "transform slab-extract".into(),
                description: Some("Reorder metadata to match shuffled base vectors".into()),
                after,
                per_profile: false,
                options: vec![
                    ("slab-file".into(), meta.metadata_all.path().into()),
                    ("ivec-file".into(), "${cache}/shuffle.ivec".into()),
                    ("output".into(), "profiles/base/metadata_content.slab".into()),
                    ("range".into(), "[${query_count},${vector_count})".into()),
                ],
            });
        }

        // survey (always when metadata present)
        let survey_after = if meta.metadata_all.is_materialized() {
            vec!["import-metadata".into()]
        } else {
            vec![]
        };
        steps.push(Step {
            id: "survey-metadata".into(),
            run: "survey".into(),
            description: Some("Survey metadata to discover schema and value ranges".into()),
            after: survey_after.clone(),
            per_profile: false,
            options: vec![
                ("input".into(), meta.metadata_all.path().into()),
                ("output".into(), "${cache}/metadata_survey.json".into()),
                ("samples".into(), "10000".into()),
                ("max-distinct".into(), "100".into()),
            ],
        });

        // synthesize predicates
        steps.push(Step {
            id: "synthesize-predicates".into(),
            run: "synthesize predicates".into(),
            description: Some("Generate test predicates from metadata survey".into()),
            after: vec!["survey-metadata".into()],
            per_profile: false,
            options: vec![
                ("input".into(), meta.metadata_all.path().into()),
                ("survey".into(), "${cache}/metadata_survey.json".into()),
                ("output".into(), "predicates.slab".into()),
                ("count".into(), "10000".into()),
                ("selectivity".into(), "0.0001".into()),
                ("seed".into(), format!("${{{}}}", "seed")),
            ],
        });

        // compute predicates (per_profile)
        let mut eval_after = vec!["synthesize-predicates".into()];
        if meta.metadata_content.is_materialized() {
            eval_after.push("extract-metadata".into());
        }
        steps.push(Step {
            id: "evaluate-predicates".into(),
            run: "compute predicates".into(),
            description: None,
            after: eval_after,
            per_profile: true,
            options: vec![
                ("input".into(), meta.metadata_content.path().into()),
                ("predicates".into(), "predicates.slab".into()),
                ("survey".into(), "${cache}/metadata_survey.json".into()),
                ("selectivity".into(), "0.0001".into()),
                ("range".into(), "[0,${base_count})".into()),
                ("output".into(), "metadata_indices.slab".into()),
            ],
        });
    }

    // ── Ground truth chain ───────────────────────────────────────────
    if let Some(ref knn) = slots.knn {
        if knn.is_materialized() {
            let mut after = vec![];
            if slots.base_count.is_some() {
                after.push("set-base-count".into());
            }
            if let Some(ref qv) = slots.query_vectors {
                if qv.is_materialized() {
                    let qid = if slots.self_search { "extract-query-vectors" } else { "import-query" };
                    after.push(qid.into());
                }
            }
            steps.push(Step {
                id: "compute-knn".into(),
                run: "compute knn".into(),
                description: Some("Compute brute-force exact KNN ground truth".into()),
                after,
                per_profile: true,
                options: vec![
                    ("base".into(), format!("{}[0..${{base_count}})", slots.base_vectors.path())),
                    ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                    ("indices".into(), "neighbor_indices.ivec".into()),
                    ("distances".into(), "neighbor_distances.fvec".into()),
                    ("neighbors".into(), args.neighbors.to_string()),
                    ("metric".into(), args.metric.clone()),
                ],
            });
        }
    }

    if let Some(ref fknn) = slots.filtered_knn {
        if fknn.is_materialized() {
            let mut after = vec!["evaluate-predicates".into()];
            if slots.base_count.is_some() {
                after.push("set-base-count".into());
            }
            if let Some(ref qv) = slots.query_vectors {
                if qv.is_materialized() {
                    let qid = if slots.self_search { "extract-query-vectors" } else { "import-query" };
                    after.push(qid.into());
                }
            }
            steps.push(Step {
                id: "compute-filtered-knn".into(),
                run: "compute filtered-knn".into(),
                description: Some("Compute filtered KNN with predicate pre-filtering".into()),
                after,
                per_profile: true,
                options: vec![
                    ("base".into(), format!("{}[0..${{base_count}})", slots.base_vectors.path())),
                    ("query".into(), slots.query_vectors.as_ref().unwrap().path().into()),
                    ("metadata-indices".into(), "metadata_indices.slab".into()),
                    ("indices".into(), "filtered_neighbor_indices.ivec".into()),
                    ("distances".into(), "filtered_neighbor_distances.fvec".into()),
                    ("neighbors".into(), args.neighbors.to_string()),
                    ("metric".into(), args.metric.clone()),
                ],
            });
        }
    }

    steps
}

// ---------------------------------------------------------------------------
// Profile view assembly
// ---------------------------------------------------------------------------

/// Assemble profile views from resolved slot paths.
fn profile_views(slots: &PipelineSlots) -> Vec<(String, String)> {
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
    slots: &PipelineSlots,
) -> String {
    let mut out = String::new();

    out.push_str("# Copyright (c) DataStax, Inc.\n");
    out.push_str("# SPDX-License-Identifier: Apache-2.0\n\n");

    out.push_str(&format!("name: {}\n", args.name));
    if let Some(ref desc) = args.description {
        out.push_str(&format!("description: >-\n  {}\n", desc));
    }

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
            if !step.after.is_empty() {
                out.push_str(&format!("      after: [{}]\n", step.after.join(", ")));
            }
            for (k, v) in &step.options {
                out.push_str(&format!("      {}: {}\n", k, v));
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

    // Sized profiles (only when self-search produces windowed base vectors)
    if slots.self_search && slots.metadata.is_some() {
        out.push_str("\n  sized:\n");
        out.push_str("    ranges: [\"0m..400m/10m\"]\n");
        out.push_str("    facets:\n");
        out.push_str("      base_vectors: \"profiles/base/base_vectors.mvec:${range}\"\n");
        out.push_str("      query_vectors: profiles/base/query_vectors.mvec\n");
        out.push_str("      metadata_predicates: predicates.slab\n");
        out.push_str("      metadata_content: \"profiles/base/metadata_content.slab:${range}\"\n");
        out.push_str("      neighbor_indices: \"profiles/${profile}/neighbor_indices.ivec\"\n");
        out.push_str("      neighbor_distances: \"profiles/${profile}/neighbor_distances.fvec\"\n");
        out.push_str("      metadata_indices: \"profiles/${profile}/metadata_indices.slab\"\n");
        out.push_str("      filtered_neighbor_indices: \"profiles/${profile}/filtered_neighbor_indices.ivec\"\n");
        out.push_str("      filtered_neighbor_distances: \"profiles/${profile}/filtered_neighbor_distances.fvec\"\n");
    }

    out
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the `datasets import` subcommand.
pub fn run(args: ImportArgs) {
    let output_dir = &args.output;

    if output_dir.exists() && !args.force {
        if output_dir.join("dataset.yaml").exists() {
            eprintln!(
                "Error: {} already contains a dataset.yaml. Use --force to overwrite.",
                output_dir.display()
            );
            std::process::exit(1);
        }
    }

    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error: failed to create directory {}: {}", output_dir.display(), e);
        std::process::exit(1);
    }

    // Resolve all slots
    let slots = resolve_slots(&args);

    // Report what was resolved
    println!("Resolving pipeline slots:");
    print_slot("all_vectors", &slots.all_vectors);
    print_slot("dedup", &slots.dedup);
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
    let steps = emit_steps(&slots, &args);
    let views = profile_views(&slots);

    println!();
    println!("{} pipeline step(s) to emit ({} identity-collapsed)",
        steps.len(),
        count_identity(&slots),
    );

    // Generate YAML
    let yaml = generate_yaml(&args, &steps, &views, &slots);

    let dataset_path = output_dir.join("dataset.yaml");
    if let Err(e) = std::fs::write(&dataset_path, &yaml) {
        eprintln!("Error: failed to write {}: {}", dataset_path.display(), e);
        std::process::exit(1);
    }

    // Create .gitignore
    let gitignore_path = output_dir.join(".gitignore");
    if !gitignore_path.exists() {
        let _ = std::fs::write(&gitignore_path, ".scratch/\n.cache/\n");
    }

    println!();
    println!("Created {}", dataset_path.display());
    if !steps.is_empty() {
        println!();
        println!("To prepare the dataset, run:");
        println!("  veks run {}", dataset_path.display());
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    if !slots.dedup.is_materialized() { n += 1; }
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
            no_filtered: false,
            force: false,
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

        // dedup should be materialized
        assert!(slots.dedup.is_materialized());

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

        let steps = emit_steps(&slots, &args);
        // Expected: set-vector-count, dedup, shuffle, extract-query, extract-base, set-base-count, compute-knn
        assert_eq!(steps.len(), 7, "steps: {:?}", steps.iter().map(|s| &s.id).collect::<Vec<_>>());
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

        let steps = emit_steps(&slots, &args);
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(step_ids.contains(&"import-metadata"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"survey-metadata"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"compute-knn"), "steps: {:?}", step_ids);
        assert!(step_ids.contains(&"compute-filtered-knn"), "steps: {:?}", step_ids);
        // No shuffle/extract steps
        assert!(!step_ids.contains(&"shuffle-ordinals"));
        assert!(!step_ids.contains(&"extract-query-vectors"));
        assert!(!step_ids.contains(&"extract-base-vectors"));
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
        assert!(!slots.dedup.is_materialized(), "dedup should be identity when --no-dedup");

        let steps = emit_steps(&slots, &args);
        let step_ids: Vec<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        assert!(!step_ids.contains(&"dedup-vectors"));
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

        let steps = emit_steps(&slots, &args);
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
}
