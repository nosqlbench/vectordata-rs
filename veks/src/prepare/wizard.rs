// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Interactive wizard for `datasets import --interactive`.
//!
//! Walks the user through the SRD §12 flowchart on the console, detecting
//! candidate data files in the working directory and prompting for each
//! meaningful option incrementally.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::formats::VecFormat;
use super::import::ImportArgs;

/// When true, all prompts return their default value without waiting for input.
static AUTO_ACCEPT: AtomicBool = AtomicBool::new(false);

/// When true, strict auto mode is active — all files must have recognized roles.
static AUTO_MODE: AtomicBool = AtomicBool::new(false);

/// Pre-seeded values from CLI flags. Each `Some` value overrides the
/// wizard's auto-detected default for that field. `None` means "let
/// the wizard decide."
#[derive(Default)]
pub struct WizardSeeds {
    pub name: Option<String>,
    pub output: Option<PathBuf>,
    pub base_vectors: Option<PathBuf>,
    pub query_vectors: Option<PathBuf>,
    pub self_search: Option<bool>,
    pub query_count: Option<u32>,
    pub metadata: Option<PathBuf>,
    pub ground_truth: Option<PathBuf>,
    pub ground_truth_distances: Option<PathBuf>,
    pub metric: Option<String>,
    pub neighbors: Option<u32>,
    pub seed: Option<u32>,
    pub description: Option<String>,
    pub no_dedup: Option<bool>,
    pub no_zero_check: Option<bool>,
    pub no_filtered: Option<bool>,
    pub normalize: Option<bool>,
    pub base_fraction: Option<f64>,
    pub pedantic_dedup: Option<bool>,
    pub required_facets: Option<String>,
    pub provided_facets: Option<String>,
    pub round_digits: Option<u32>,
    pub selectivity: Option<f64>,
    pub force: bool,
    pub classic: bool,
    pub sources: Vec<PathBuf>,
}

/// Run the wizard with auto-accept disabled (fully interactive).
pub fn run_wizard() -> ImportArgs {
    run_wizard_with_options(false, false, WizardSeeds::default())
}

/// Run the interactive import wizard.
///
/// - `auto_accept`: all prompts return their default value (-y flag)
/// - `auto_mode`: strict mode — all candidate files must be recognized by
///   role keywords, underscore-prefix renaming is assumed, and unrecognized
///   files cause a hard stop with guidance on naming conventions.
/// - `seeds`: pre-seeded values from CLI flags that override wizard defaults.
pub fn run_wizard_with_options(auto_accept: bool, auto_mode: bool, seeds: WizardSeeds) -> ImportArgs {
    AUTO_ACCEPT.store(auto_accept, Ordering::Relaxed);
    AUTO_MODE.store(auto_mode, Ordering::Relaxed);
    if auto_mode {
        println!("=== veks prepare bootstrap — auto mode ===");
    } else if auto_accept {
        println!("=== veks prepare bootstrap — auto-accept mode (-y) ===");
    }
    println!("=== veks prepare bootstrap — interactive wizard ===");
    println!();

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Pre-check: warn if dataset.yaml already exists
    if cwd.join("dataset.yaml").exists() {
        if seeds.force {
            println!("Overwriting existing dataset.yaml (--overwrite).");
        } else {
            println!("Warning: dataset.yaml already exists in this directory.");
            if !confirm("Overwrite it?", false) {
                eprintln!("Aborted.");
                std::process::exit(0);
            }
        }
        // Back up before overwriting
        let yaml_path = cwd.join("dataset.yaml");
        match crate::check::fix::create_backup(&yaml_path) {
            Ok(bp) => println!("  Backed up {} → {}", crate::check::rel_display(&yaml_path), crate::check::rel_display(&bp)),
            Err(e) => eprintln!("  Warning: backup failed: {}", e),
        }
        println!();
    }

    // Scan for candidate data files, or use explicit --source list
    let candidates = if !seeds.sources.is_empty() {
        // Build candidates from explicit --source paths
        let mut explicit = Vec::new();
        for src in &seeds.sources {
            let path = if src.is_absolute() { src.clone() } else { cwd.join(src) };
            if let Some(format) = VecFormat::detect_from_path(&path) {
                if format == VecFormat::Hdf5 {
                    if let Ok(datasets) = veks_core::formats::reader::hdf5::list_datasets(&path) {
                        for (ds_name, rows, cols, elem_size) in datasets {
                            let ds_size = rows * cols as u64 * elem_size as u64;
                            let fmt_name = match elem_size {
                                4 => if ds_name.contains("indic") { "ivec" } else { "fvec" },
                                8 => "dvec", 2 => "mvec", 1 => "bvec", _ => "fvec",
                            };
                            let rel = PathBuf::from(format!("{}#{}", src.display(), ds_name));
                            explicit.push((rel, fmt_name.to_string(), ds_size));
                        }
                    }
                } else {
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    explicit.push((src.clone(), format.name().to_string(), size));
                }
            } else {
                eprintln!("Warning: unrecognized format for --source {}", src.display());
            }
        }
        explicit
    } else {
        scan_candidates(&cwd)
    };
    if !candidates.is_empty() {
        println!("Detected files:");
        for (path, format, size) in &candidates {
            println!("  {} ({}, {})", path.display(), format, format_size(*size));
        }
        println!();
    } else {
        println!("No recognized data files found in current directory.");
        println!("You can still specify paths to files elsewhere.");
        println!();
    }

    // ── Filename-keyword role detection ──────────────────────────────
    let mut detected = detect_roles(&candidates);

    // Apply --provided-facets masking: null out detected inputs for
    // facets not in the provided set so the pipeline computes them.
    if let Some(ref pf) = seeds.provided_facets {
        let p = crate::prepare::import::parse_facet_spec(pf);
        if !p.contains('B') { detected.base_vectors = None; }
        if !p.contains('Q') { detected.query_vectors = None; }
        if !p.contains('G') { detected.neighbor_indices = None; }
        if !p.contains('D') { detected.neighbor_distances = None; }
        if !p.contains('M') { detected.metadata = None; }
    }

    // In --auto mode, all candidate files must have recognized roles.
    if AUTO_MODE.load(Ordering::Relaxed) && !detected.unassigned.is_empty() {
        eprintln!();
        eprintln!("Error: --auto mode requires all data files to have recognizable role keywords");
        eprintln!("in their filenames. The following files could not be assigned a role:");
        eprintln!();
        for p in &detected.unassigned {
            eprintln!("  {}", p.display());
        }
        eprintln!();
        eprintln!("Rename files to include one of these keywords (underscore-delimited):");
        eprintln!();
        eprintln!("  Role                        Keywords");
        eprintln!("  ──────────────────────────── ─────────────────────────────────────────");
        eprintln!("  Base vectors                 base, train");
        eprintln!("  Query vectors                query, queries, test");
        eprintln!("  Neighbor indices (GT)        groundtruth, gt, indices, neighbors");
        eprintln!("  Neighbor distances (GT)      distances");
        eprintln!("  Metadata content             metadata, content");
        eprintln!("  Metadata predicates          predicates");
        eprintln!("  Metadata results             results");
        eprintln!("  Filtered neighbor indices    filtered + (indices|neighbors|gt)");
        eprintln!("  Filtered neighbor distances  filtered + distances");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  base_vectors.fvec          → Base vectors");
        eprintln!("  query_test.mvec            → Query vectors");
        eprintln!("  gt_neighbors.ivec          → Neighbor indices");
        eprintln!("  metadata_content.slab      → Metadata content");
        eprintln!("  filtered_neighbors.ivec    → Filtered neighbor indices");
        eprintln!();
        eprintln!("A leading underscore (e.g., _base_vectors.mvec) is stripped before");
        eprintln!("keyword matching, so renamed source files are still recognized.");
        std::process::exit(1);
    }

    let roles_accepted = if detected.any_detected() {
        let mut accepted = false;
        loop {
            println!("Detected file roles:");
            detected.print_summary();
            println!();
            if confirm("Use detected assignments?", true) {
                accepted = true;
                break;
            }
            // User rejected — ask for a facet constraint and re-filter
            println!();
            println!("Enter provided facet codes to constrain detection");
            println!("  (e.g., BQ = only base + query, BQGD = all four)");
            println!("  Or press Enter to skip auto-detection entirely.");
            let spec = prompt_with_default("Provided facets", "");
            if spec.is_empty() {
                break; // skip detection
            }
            let p = crate::prepare::import::parse_facet_spec(&spec);
            // Re-run detection with the new constraint
            detected = detect_roles(&candidates);
            if !p.contains('B') { detected.base_vectors = None; }
            if !p.contains('Q') { detected.query_vectors = None; }
            if !p.contains('G') { detected.neighbor_indices = None; }
            if !p.contains('D') { detected.neighbor_distances = None; }
            if !p.contains('M') { detected.metadata = None; }
            println!();
        }
        accepted
    } else {
        false
    };

    // ── Rename source files early ─────────────────────────────────────
    // Prompt to underscore-prefix any detected source files that don't
    // already have it. Done early so the user sees it right after detection.
    if roles_accepted {
        rename_detected_sources(&mut detected);
    }

    // ── Facet inference from detected inputs ─────────────────────────
    // Infer which facets the pipeline should produce based on what
    // inputs were detected. Present as a checkbox for the user to
    // confirm or adjust BEFORE asking detailed questions.
    // Infer facets using the SRD §2.8 implication rules (same logic as
    // import::resolve_facets). Build a temporary ImportArgs to query it.
    let has_detected_base = roles_accepted && detected.base_vectors.is_some()
        || seeds.base_vectors.is_some();
    let has_detected_meta = roles_accepted && detected.metadata.is_some()
        || seeds.metadata.is_some();

    let inferred = {
        let probe = ImportArgs {
            name: String::new(),
            output: PathBuf::new(),
            base_vectors: if has_detected_base { Some(PathBuf::from("probe")) } else { None },
            query_vectors: None,
            self_search: false,
            query_count: 0,
            metadata: if has_detected_meta { Some(PathBuf::from("probe")) } else { None },
            ground_truth: None,
            ground_truth_distances: None,
            metric: String::new(),
            neighbors: 0,
            seed: 0,
            description: None,
            no_dedup: false,
            no_zero_check: false,
            no_filtered: false,
            normalize: true,
            force: false,
            base_convert_format: None,
            query_convert_format: None,
            compress_cache: false,
            sized_profiles: None,
            base_fraction: 1.0,
            required_facets: None,
            provided_facets: seeds.provided_facets.clone(),
            round_digits: 2,
            pedantic_dedup: false,
            selectivity: 0.0001,
            predicate_count: 10000,
            predicate_strategy: "eq".to_string(),
            classic: seeds.classic,
            personality: "native".to_string(),
            synthesize_metadata: false,
            synthesis_mode: "simple-int-eq".to_string(),
            synthesis_format: "slab".to_string(),
            metadata_fields: 3,
            metadata_range_min: 0,
            metadata_range_max: 1000,
            predicate_range_min: 0,
            predicate_range_max: 1000,
        };
        super::import::resolve_facets(&probe)
    };

    let facet_labels = [
        ('B', "base vectors"),
        ('Q', "query vectors"),
        ('G', "KNN ground-truth indices"),
        ('D', "KNN ground-truth distances"),
        ('M', "metadata"),
        ('P', "predicates"),
        ('R', "predicate results"),
        ('F', "filtered KNN ground-truth"),
    ];

    println!();
    println!("--- Required facets ---");
    println!("  Inferred from detected inputs (SRD §2.8):");
    println!();
    for &(code, label) in &facet_labels {
        let on = inferred.contains(code);
        println!("  [{}] {}  {}", if on { "x" } else { " " }, code, label);
    }
    println!();

    let implied_facets = &inferred;

    let confirmed_facets = if let Some(ref seeded) = seeds.required_facets {
        let parsed = crate::prepare::import::parse_facet_spec(seeded);
        println!("  (overridden by --required-facets → {})", parsed);
        parsed
    } else {
        let input = prompt_with_default(
            "Facets to include in dataset (* for all)",
            &implied_facets,
        );
        let parsed = crate::prepare::import::parse_facet_spec(input.trim());
        // Show what was selected, and note which facets will be generated
        let extra: String = parsed.chars()
            .filter(|c| !implied_facets.contains(*c))
            .collect();
        if !extra.is_empty() {
            println!("  → Facets: {}  ({}  will be generated)", parsed, extra);
        } else if parsed != input.trim().to_uppercase() {
            println!("  → Facets: {}", parsed);
        }
        parsed
    };

    // Parse confirmed facets into booleans for gating subsequent questions
    let mut confirmed_facets = confirmed_facets;
    let want_q = confirmed_facets.contains('Q');

    // ── Base data fraction ─────────────────────────────────────────
    // Asked early because it's a fundamental parameter that affects
    // all downstream processing. Immutable after first run (SRD §3.13).
    let frac_default = seeds.base_fraction
        .map(|f| format!("{}", (f * 100.0).round() as u32))
        .unwrap_or_else(|| "100".to_string());
    let base_fraction_str = prompt_with_default("Base data percentage (1-100)", &frac_default);
    let base_fraction = base_fraction_str.trim().parse::<f64>()
        .unwrap_or(100.0)
        .clamp(1.0, 100.0) / 100.0;
    let want_g = confirmed_facets.contains('G');
    let want_m = confirmed_facets.contains('M');

    // ── Dataset name ─────────────────────────────────────────────────
    let dir_name = cwd.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "my-dataset".to_string());
    let name_default = seeds.name.as_deref().unwrap_or(&dir_name);
    let name = prompt_with_prefill("Dataset name", name_default);

    // ── Description ──────────────────────────────────────────────────
    let description = prompt_optional("Description (optional)");

    // ── Output directory ─────────────────────────────────────────────
    let output = prompt_path_with_default("Output directory", ".");

    // ── Base vectors (B is always required) ───────────────────────────
    println!();
    println!("--- Vector source ---");
    let base_vectors = if roles_accepted && detected.base_vectors.is_some() {
        let bv = detected.base_vectors.as_ref().unwrap();
        println!("  Using detected: {}", bv.display());
        Some(bv.clone())
    } else {
        let vector_candidates: Vec<&(PathBuf, String, u64)> = candidates.iter()
            .filter(|(_, fmt, _)| {
                let f = fmt.as_str();
                f == "fvec" || f == "ivec" || f == "mvec" || f == "bvec" || f == "dvec" || f == "svec" || f == "npy"
            })
            .collect();

        if vector_candidates.len() == 1 {
            let candidate = &vector_candidates[0].0;
            println!("  Found: {}", candidate.display());
            if confirm("Use this as base vectors?", true) {
                Some(candidate.clone())
            } else {
                prompt_optional_path("Path to base vectors")
            }
        } else if vector_candidates.len() > 1 {
            println!("  Multiple vector files found:");
            for (i, (path, fmt, size)) in vector_candidates.iter().enumerate() {
                println!("    {}. {} ({}, {})", i + 1, path.display(), fmt, format_size(*size));
            }
            let choice = prompt_optional(&format!("Choose base vectors [1-{}] or enter path", vector_candidates.len()));
            match choice {
                Some(s) => {
                    if let Ok(idx) = s.parse::<usize>() {
                        if idx >= 1 && idx <= vector_candidates.len() {
                            Some(vector_candidates[idx - 1].0.clone())
                        } else {
                            Some(PathBuf::from(s))
                        }
                    } else {
                        Some(PathBuf::from(s))
                    }
                }
                None => None,
            }
        } else {
            prompt_optional_path("Path to base vectors")
        }
    };

    if base_vectors.is_none() {
        println!("  No base vectors specified. Only metadata operations will be available.");
    }

    // ── Precision confirmation for xvec formats ─────────────────────
    // When the source is already an xvec file, confirm with the user that
    // the native precision is correct for their use case.
    let base_convert_format = if let Some(ref bv) = base_vectors {
        check_vector_precision("Base vectors", bv)
    } else {
        None
    };

    // ── Query source — resolved from facets + detected inputs ──────────
    // Q confirmed + separate query file detected → use it
    // Q confirmed + no query file → self-search (only ask query count)
    // Q not confirmed → skip
    let (query_vectors, self_search, query_count) = if !want_q || base_vectors.is_none() {
        (None, false, seeds.query_count.unwrap_or(10000))
    } else if roles_accepted && detected.query_vectors.is_some() {
        let qv = detected.query_vectors.as_ref().unwrap().clone();
        (Some(qv), false, seeds.query_count.unwrap_or(10000))
    } else {
        // Self-search — only ask query count
        let qc_default = seeds.query_count.map(|n| n.to_string())
            .unwrap_or_else(|| "10000".to_string());
        let qc_str = prompt_with_default("Query count (self-search)", &qc_default);
        let qc = qc_str.parse().unwrap_or(10000);
        (None, true, qc)
    };

    let query_convert_format = if let Some(ref qv) = query_vectors {
        check_vector_precision("Query vectors", qv)
    } else {
        None
    };

    // ── Metadata — resolved from facets + detected inputs ──────────────
    // M confirmed + detected → use it. M confirmed + not detected → synthesis.
    // M not confirmed → skip.
    // For simple-int-eq mode, predicate count is asked here to avoid
    // a redundant predicate config section later.
    let mut simple_int_eq_predicate_count: Option<u32> = None;
    #[allow(clippy::type_complexity)]
    let (metadata, synthesize_metadata, synthesis_mode, synthesis_format,
         metadata_fields, metadata_range_min, metadata_range_max,
         predicate_range_min, predicate_range_max)
    : (Option<PathBuf>, bool, String, String, u32, i32, i32, i32, i32) = if !want_m {
        (None, false, "simple-int-eq".into(), "slab".into(), 3, 0, 1000, 0, 1000)
    } else if roles_accepted && detected.metadata.is_some() {
        let m = detected.metadata.as_ref().unwrap().clone();
        (Some(m), false, "simple-int-eq".into(), "slab".into(), 3, 0, 1000, 0, 1000)
    } else if let Some(ref seeded) = seeds.metadata {
        (Some(seeded.clone()), false, "simple-int-eq".into(), "slab".into(), 3, 0, 1000, 0, 1000)
    } else {
        // M facet was confirmed but no metadata file is available.
        // Offer synthesis modes.
        {
            println!();
            println!("--- Metadata & Predicate Synthesis ---");
            println!("  No metadata source detected. Choose a synthesis mode:");
            println!();
            println!("    1. Simple integer equality");
            println!("       Each record gets integer fields in a range.");
            println!("       Predicates are single-field equality checks.");
            println!();
            println!("    2. Conjugate & selectivity synthesis (not yet implemented)");
            println!("       Compound AND/OR predicates with selectivity control.");
            println!();

            let mode_choice = prompt_with_default("Synthesis mode [1/2]", "1");

            if mode_choice == "2" {
                println!("  Conjugate synthesis is not yet implemented.");
                println!("  Falling back to simple integer equality.");
            }

            if mode_choice == "1" || mode_choice == "2" {
                println!();
                println!("  Metadata — each record gets integer fields drawn from a range.");
                let fields_str = prompt_with_default("Number of integer fields per record", "1");
                let fields: u32 = fields_str.parse().unwrap_or(1);
                let rmin_str = prompt_with_default("Range minimum (inclusive)", "0");
                let rmin: i32 = rmin_str.parse().unwrap_or(0);
                let rmax_str = prompt_with_default("Range maximum (exclusive)", "100");
                let rmax: i32 = rmax_str.parse().unwrap_or(100);

                let range_size = (rmax - rmin).max(1) as f64;
                println!();
                println!("  Predicates — each predicate is \"field == value\".");
                println!("  Predicate values are drawn from a range (can differ from metadata).");
                let pred_rmin_str = prompt_with_default("Predicate range minimum (inclusive)",
                    &rmin.to_string());
                let pred_rmin: i32 = pred_rmin_str.parse().unwrap_or(rmin);
                let pred_rmax_str = prompt_with_default("Predicate range maximum (exclusive)",
                    &rmax.to_string());
                let pred_rmax: i32 = pred_rmax_str.parse().unwrap_or(rmax);
                let pred_range = (pred_rmax - pred_rmin).max(1) as f64;
                let match_pct = if pred_rmax <= rmax && pred_rmin >= rmin {
                    100.0 / range_size  // predicate values all within metadata range
                } else {
                    // some predicate values may be outside metadata range → lower match rate
                    let overlap = (rmax.min(pred_rmax) - rmin.max(pred_rmin)).max(0) as f64;
                    (overlap / pred_range) * (100.0 / range_size)
                };
                println!("  Each predicate matches ~{:.1}% of records.", match_pct);

                let count_str = prompt_with_default("Number of predicates to generate", "10000");
                let pred_count: u32 = count_str.parse().unwrap_or(10000);

                println!();
                println!("  Storage format for metadata and predicates:");
                println!("    slab  — canonical MNode/PNode in slab files (full type system)");
                println!("    ivec  — plain integer vectors (lightweight, fast)");
                let format = prompt_with_default("Storage format", "ivec");

                // Stash predicate config for later — avoids redundant prompts
                simple_int_eq_predicate_count = Some(pred_count);

                (None, true, "simple-int-eq".into(), format, fields, rmin, rmax, pred_rmin, pred_rmax)
            } else {
                // User entered something unexpected — drop M facets
                println!("  Dropping M facet (no metadata source).");
                confirmed_facets.retain(|c| !matches!(c, 'M' | 'P' | 'R' | 'F'));
                (None, false, "simple-int-eq".into(), "slab".into(), 3, 0, 1000, 0, 1000)
            }
        }
    };

    // ── Ground truth — resolved from facets + detected inputs ─────────
    // G confirmed + detected pre-computed → use it.
    // G confirmed + not detected → will be computed by the pipeline.
    let has_queries = query_vectors.is_some() || self_search;
    let (ground_truth, ground_truth_distances) = if !want_g || !has_queries {
        (None, None)
    } else if roles_accepted && detected.neighbor_indices.is_some() {
        let gt = detected.neighbor_indices.as_ref().unwrap().clone();
        let gtd = detected.neighbor_distances.clone();
        (Some(gt), gtd)
    } else if let Some(ref seeded) = seeds.ground_truth {
        (Some(seeded.clone()), seeds.ground_truth_distances.clone())
    } else {
        // G confirmed, no pre-computed GT — pipeline will compute it
        (None, None)
    };

    // ── Source metric and normalization ─────────────────────────────
    //
    // The distance metric is a property of the source data — it describes
    // how the embedding model was trained. The user can't arbitrarily
    // change it; the only meaningful transformation is L2-normalization,
    // which makes Cosine distance equivalent to (negated) Dot Product and
    // changes the L2 distance ranking to match Cosine ranking.
    println!();
    println!("--- Distance metric ---");
    // Auto-detect metric from vector data
    let (detected_metric, detect_reason) = base_vectors.as_ref()
        .map(|p| crate::prepare::import::detect_metric(p))
        .unwrap_or_else(|| ("COSINE".to_string(), "default".to_string()));
    println!("  Detected: {} ({})", detected_metric, detect_reason);
    println!("  Common metrics:");
    println!("    L2         — Euclidean distance (most embedding models)");
    println!("    Cosine     — Cosine similarity (angular distance)");
    println!("    DotProduct — Inner product (often used with normalized vectors)");
    println!();
    let metric_default = seeds.metric.as_deref().unwrap_or(&detected_metric);
    let source_metric = prompt_with_default(
        "What distance metric does this data use?",
        metric_default,
    );

    let metric = source_metric;

    // Normalization: detect current state and ask user. Default is enabled.
    println!();
    println!("--- Normalization ---");
    let already_normalized = if let Some(ref bv) = base_vectors {
        eprint!("  Sampling vectors for normalization status... ");
        use std::io::Write;
        let _ = std::io::stderr().flush();

        let detected = super::import::detect_normalized(bv);
        if let Some((is_norm, count, mean)) = detected {
            if is_norm {
                eprintln!("already normalized (mean norm={:.4}, n={}).", mean, count);
            } else {
                eprintln!("not normalized (mean norm={:.4}, n={}).", mean, count);
            }
            is_norm
        } else {
            eprintln!("could not detect (source format).");
            false
        }
    } else {
        false
    };

    println!("  L2-normalization ensures all vectors have unit length.");
    println!("  This is applied during extraction — source data is unchanged.");
    let has_precomputed_gt = ground_truth.is_some();
    let normalize = if let Some(seeded) = seeds.normalize {
        println!("  (seeded: normalize={})", seeded);
        seeded
    } else if already_normalized {
        println!("  Vectors are already normalized.");
        let force = confirm("Normalize anyway (re-normalize)?", false);
        if force {
            true
        } else {
            println!("  Skipping normalization (already normalized).");
            false
        }
    } else if has_precomputed_gt {
        println!("  Pre-computed ground truth provided — normalizing would");
        println!("  change distances and invalidate the GT neighbors.");
        confirm("L2-normalize vectors? (will invalidate GT)", false)
    } else {
        confirm("L2-normalize vectors?", true)
    };

    // ── Remaining configuration ─────────────────────────────────────
    println!();
    println!("--- Configuration ---");

    let has_precomputed_gt = ground_truth.is_some();
    let neighbors = if has_precomputed_gt {
        // Infer k from the ground truth ivec dimensionality (each row has k neighbor indices)
        let gt_path = ground_truth.as_ref().unwrap();
        let k = probe_vector_dim(gt_path).unwrap_or(100);
        println!("  Ground truth provided: k={} (inferred from {})", k, gt_path.display());
        k as u32
    } else if has_queries {
        let n_str = prompt_with_default("Number of neighbors (k)", "100");
        n_str.parse().unwrap_or(100)
    } else {
        100
    };

    // Shuffling: optional, seed 0 disables it
    let seed = if let Some(seeded) = seeds.seed {
        if seeded == 0 {
            println!("  (seeded: shuffle disabled)");
        } else {
            println!("  (seeded: shuffle seed={})", seeded);
        }
        seeded
    } else {
        let shuffle = confirm("Shuffle base vectors?", true);
        if shuffle {
            let seed_str = prompt_with_default("Shuffle seed", "42");
            seed_str.parse().unwrap_or(42)
        } else {
            0 // seed 0 disables shuffling
        }
    };

    // Predicate configuration — skip if already configured by simple-int-eq synthesis
    let (selectivity, predicate_count, predicate_strategy) = if let Some(count) = simple_int_eq_predicate_count {
        // Simple-int-eq: strategy is always "eq", selectivity is 1/range
        let range = (metadata_range_max - metadata_range_min).max(1) as f64;
        (1.0 / range, count, "eq".to_string())
    } else if confirmed_facets.contains('P') {
        // Non-synthesis predicates — full configuration prompt
        println!();
        println!("--- Predicates ---");

        let natural_selectivity = 0.0001;

        println!("  Predicate strategy:");
        println!("    eq       — single-field equality (simple, default)");
        println!("    compound — multi-field AND (harder, mixed operators)");
        let strategy = prompt_with_default("Strategy", "eq");

        let count_str = prompt_with_default("Number of predicates to generate", "10000");
        let count: u32 = count_str.parse().unwrap_or(10000);

        println!();
        println!("  Selectivity controls filtering difficulty.");
        println!("  Lower = harder (fewer matches per query).");
        println!("    0.1    = 10% of base qualifies   (easy)");
        println!("    0.01   = 1%                       (moderate)");
        println!("    0.001  = 0.1%                     (hard)");
        println!("    0.0001 = 0.01%                    (very hard)");
        let default_sel = seeds.selectivity
            .map(|s| format!("{}", s))
            .unwrap_or_else(|| format!("{}", natural_selectivity));
        let sel_str = prompt_with_default("Predicate selectivity", &default_sel);
        let sel = sel_str.parse::<f64>().unwrap_or(natural_selectivity);

        (sel, count, strategy)
    } else {
        (seeds.selectivity.unwrap_or(0.0001), 10000, "eq".to_string())
    };

    // ── Optional features ────────────────────────────────────────────
    // When pre-computed ground truth is provided, dedup and zero checks
    // are advisory only — the GT was computed against the original ordinal
    // space, so we cannot reindex without invalidating it. The checks
    // still run and report findings, but no exclusion ordinals are generated.
    let (no_dedup, no_zero_check) = if has_precomputed_gt && base_vectors.is_some() {
        println!("  Ground truth is pre-computed — dedup/zero checks will be advisory only.");
        println!("  (Cannot exclude vectors without invalidating ground truth ordinals.)");
        // Still run the checks but mark them as advisory
        (true, true) // no ordinal exclusion
    } else if base_vectors.is_some() {
        let nd = !confirm("Include deduplication stage?", true);
        let default_yes = !nd;
        let nz = !confirm("Check for and exclude zero vectors?", default_yes);
        (nd, nz)
    } else {
        (true, true)
    };

    if !no_dedup && !no_zero_check {
        println!("  A clean ordinal index will be produced excluding both duplicates and zeros.");
    } else if !no_dedup {
        println!("  A clean ordinal index will be produced excluding duplicates.");
    } else if !no_zero_check {
        println!("  A clean ordinal index will be produced excluding zeros.");
    }

    // no_filtered is determined by the facet confirmation above.
    let no_filtered = !confirmed_facets.contains('F');

    // ── Sized profiles ────────────────────────────────────────────────
    // Stratified profiles are independent of query vectors — they define
    // windowed subsets of the base data at multiple scales.
    let sized_profiles = if let Some(ref bv) = base_vectors {
        {
            let vec_count = probe_vector_count(bv);
            if vec_count.is_none() || vec_count == Some(0) {
                eprintln!("Error: could not determine vector count from '{}'", bv.display());
                eprintln!("The source must be readable and contain at least one vector.");
                std::process::exit(1);
            }
            let vec_count = vec_count.unwrap();
            let query_sub = if self_search { query_count as u64 } else { 0 };
            let effective_max = vec_count.saturating_sub(query_sub);

            if effective_max == 0 {
                println!();
                println!("  Warning: query_count ({}) >= vector count ({}) — no base vectors remain.",
                    query_count, vec_count);
                println!("  Sized profiles are not applicable.");
                None
            } else {
            let max_label = format_count_label(effective_max);

            println!();
            println!("--- Sized profiles ---");
            if self_search {
                println!("  Base vector count after query extraction: ~{}", max_label);
            } else {
                println!("  Base vector count: ~{}", max_label);
            }
            println!("  Sized profiles create windowed subsets of the base vectors at");
            println!("  different scales (e.g., 1M, 2M, 4M, ..., 10M, 20M, ...).");
            println!();
            println!("  Why this is useful:");
            println!("    - Benchmark search performance across dataset sizes");
            println!("    - Smaller profiles compute KNN much faster, giving early results");
            println!("    - Verification runs on smaller profiles first, catching errors");
            println!("      before investing hours on the full dataset");
            println!("    - KNN partition caches are shared across profiles — partitions");
            println!("      computed for a 10M profile are reused by 50M, 100M, etc.");
            println!("    - Incremental workflow: verify correctness at 1M, then 10M,");
            println!("      then scale up with confidence");
            println!();
            // Build a default spec. The upper bound is implicit — profiles
            // are only valid for sizes ≤ the default profile's base count.
            // Client libraries interpret the sized spec directly.
            let build_spec = || -> String {
                if effective_max >= 2_000_000 {
                    // Use binary units (1mi = 2^20 = 1,048,576) so the
                    // doubling series produces clean IEC names:
                    // 1mi, 2mi, 4mi, ..., 512mi, 1gi, 2gi, ...
                    "mul:1mi/2".to_string()
                } else if effective_max >= 100_000 {
                    let start = (effective_max / 10).max(1000);
                    let start_label = format_count_label(start);
                    format!("mul:{}/2", start_label)
                } else if effective_max >= 10_000 {
                    let half = format_count_label(effective_max / 2);
                    half
                } else {
                    format_count_label(effective_max / 2)
                }
            };

            if effective_max < 20_000_000 {
                println!("  Dataset has fewer than 20M vectors — sized profiles are optional.");
                if !confirm("Generate sized profiles anyway?", false) {
                    None
                } else {
                    let default_spec = build_spec();
                    let spec = prompt_optional(&format!("  Sized profile spec [{}]", default_spec));
                    Some(spec.unwrap_or(default_spec))
                }
            } else if confirm("Generate sized profiles?", true) {
                let default_spec = build_spec();
                println!();
                println!("  Default: {}", default_spec);
                println!("  Or enter a custom spec, or press Enter for the default.");
                let spec = prompt_optional("  Sized profile spec (Enter for default)");
                Some(spec.unwrap_or(default_spec))
            } else {
                None
            }
            } // close effective_max > 0 else
        }
    } else {
        None
    };

    // ── Cache compression ────────────────────────────────────────────
    println!();
    println!("--- Cache compression ---");
    println!("  Intermediate cache files can be gzip-compressed to save disk space.");
    println!("  This slows down processing and can always be done later with");
    println!("  `veks prepare cache-compress`. Recommended: No for initial runs.");
    let compress_cache = confirm("Compress cache files?", false);

    // The confirmed facets drive required_facets for ImportArgs
    let required_facets = if confirmed_facets == *implied_facets {
        None // user accepted inference — let import figure it out
    } else {
        Some(confirmed_facets.clone())
    };

    // ── Summary — artifact lineage view ──────────────────────────────
    use veks_core::term;

    println!();
    println!("{}", term::bold("=== Pipeline Summary ==="));
    println!();

    // Configuration line
    let frac_str = if base_fraction < 1.0 {
        format!("{:.0}%", base_fraction * 100.0)
    } else {
        "100%".into()
    };
    // Pad to visual width BEFORE applying ANSI color codes.
    // ANSI escapes are invisible but count in format width calculations.
    let pad = |s: &str, w: usize| -> String {
        if s.len() >= w { s.to_string() } else { format!("{}{}", s, " ".repeat(w - s.len())) }
    };

    println!("  {} {} {} {} {} {}",
        term::dim(&pad("Dataset:", 10)), term::bold(&pad(&name, 18)),
        term::dim(&pad("Output:", 10)), pad(&output.display().to_string(), 16),
        term::dim("Fraction:"), frac_str);
    println!("  {} {} {} {} {} {}",
        term::dim(&pad("Metric:", 10)), pad(&metric, 18),
        term::dim(&pad("k:", 10)), pad(&neighbors.to_string(), 16),
        term::dim("Seed:"), if seed == 0 { "none".to_string() } else { seed.to_string() });
    println!("  {} {} {} {}", term::dim(&pad("Normalize:", 10)),
        pad(&if normalize { term::info("yes") } else { "no".to_string() }, 18),
        term::dim("Shuffle:"),
        if seed == 0 { "no".to_string() } else { term::info("yes") });
    if !no_dedup || !no_zero_check {
        let cleaning: Vec<&str> = [
            if !no_dedup { Some("dedup") } else { None },
            if !no_zero_check { Some("zero-check") } else { None },
        ].into_iter().flatten().collect();
        println!("  {} {}", term::dim("Cleaning:"), cleaning.join(", "));
    }
    if let Some(ref sp) = sized_profiles {
        println!("  {} {}", term::dim("Stratify:"), sp);
    }
    println!();

    // (header printed after facet_rows is built)

    // Helper: format an input path for display (truncate to ~35 chars)
    let fmt_input = |path: &Path, fmt: &str| -> String {
        let name = path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        if name.len() > 30 {
            format!("{}.. ({})", &name[..28], fmt)
        } else {
            format!("{} ({})", name, fmt)
        }
    };

    // Determine source format label, including precision for non-xvec sources.
    // For HDF5 # paths and other container formats, probe to get element size.
    let detect_format = |p: &Path| -> String {
        let path_str = p.to_string_lossy();
        // HDF5 dataset paths
        if path_str.contains('#') {
            let fmt = VecFormat::detect(p);
            if let Some(f) = fmt {
                if let Ok(meta) = veks_core::formats::reader::probe_source(p, f) {
                    let prec = match meta.element_size {
                        1 => "u8", 2 => "f16", 4 => "f32", 8 => "f64",
                        n => return format!("hdf5 {}B", n),
                    };
                    return format!("hdf5#{} dim={}", prec, meta.dimension);
                }
            }
            return "hdf5".into();
        }
        if p.is_dir() {
            if std::fs::read_dir(p).ok()
                .map(|entries| entries.flatten().any(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("npy")
                })).unwrap_or(false) {
                return "npy dir".into();
            }
            if std::fs::read_dir(p).ok()
                .map(|entries| entries.flatten().any(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("parquet")
                })).unwrap_or(false) {
                return "parquet dir".into();
            }
            "directory".into()
        } else if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            ext.to_string()
        } else {
            VecFormat::detect(p)
                .map(|f: VecFormat| f.name().to_string())
                .unwrap_or_else(|| "file".into())
        }
    };

    let base_fmt = base_vectors.as_ref()
        .map(|p| detect_format(p))
        .unwrap_or_default();

    // Build facet rows: (code, label, connection_type, source_input)
    // Connection types: "provided" (pre-computed), "convert", "self-search",
    //                   "compute", "synthesize", "evaluate", "checked"
    struct FacetRow {
        code: char,
        label: &'static str,
        connection: &'static str,
        source: String,
    }

    let has_precomputed_gt = ground_truth.is_some();
    let base_label = base_vectors.as_ref()
        .map(|p| fmt_input(p, &base_fmt))
        .unwrap_or_default();
    let meta_label = metadata.as_ref()
        .map(|p| {
            let ext = detect_format(p);
            fmt_input(p, &ext)
        })
        .unwrap_or_default();

    let facet_rows: Vec<FacetRow> = vec![
        FacetRow {
            code: 'B', label: "base vectors",
            connection: if base_vectors.as_ref().map(|p| {
                VecFormat::detect(p).map(|f| f.is_xvec()).unwrap_or(false)
            }).unwrap_or(false) { "identity" } else { "convert" },
            source: base_label.clone(),
        },
        FacetRow {
            code: 'Q', label: "query vectors",
            connection: if query_vectors.is_some() {
                if query_vectors.as_ref().map(|p| VecFormat::detect(p).map(|f| f.is_xvec()).unwrap_or(false)).unwrap_or(false) {
                    "provided"
                } else { "convert" }
            } else if self_search { "self-search" } else { "—" },
            source: if query_vectors.is_some() {
                let qfmt = query_vectors.as_ref().map(|p| detect_format(p)).unwrap_or_default();
                query_vectors.as_ref().map(|p| fmt_input(p, &qfmt)).unwrap_or_default()
            } else if self_search {
                format!("{} ({})", base_label, query_count)
            } else { String::new() },
        },
        FacetRow {
            code: 'G', label: "KNN indices",
            connection: if has_precomputed_gt { "provided" } else { "compute" },
            source: if has_precomputed_gt {
                ground_truth.as_ref().map(|p| fmt_input(p, "ivec")).unwrap_or_default()
            } else { "B x Q brute-force".into() },
        },
        FacetRow {
            code: 'D', label: "KNN distances",
            connection: if ground_truth_distances.is_some() { "provided" } else { "compute" },
            source: if ground_truth_distances.is_some() {
                ground_truth_distances.as_ref().map(|p| fmt_input(p, "fvec")).unwrap_or_default()
            } else { "B x Q brute-force".into() },
        },
        FacetRow {
            code: 'M', label: "metadata",
            connection: if metadata.is_some() { "convert" } else { "—" },
            source: meta_label.clone(),
        },
        FacetRow {
            code: 'P', label: "predicates",
            connection: "synthesize",
            source: "M schema survey".into(),
        },
        FacetRow {
            code: 'R', label: "predicate results",
            connection: "evaluate",
            source: "M x P evaluation".into(),
        },
        FacetRow {
            code: 'F', label: "filtered KNN",
            connection: "compute",
            source: "B x Q x R filtered search".into(),
        },
    ];

    // Even/odd two-line layout: facet info on line 1, source on line 2
    println!("  {}", term::bold("Pipeline facets"));
    println!("  {}", term::dim("──────────────────────────────"));

    for row in &facet_rows {
        if !confirmed_facets.contains(row.code) {
            continue;
        }
        let tag = match row.connection {
            "identity"    => term::ok("[identity]"),
            "provided"    => term::green("[provided]"),
            "convert"     => term::info("[convert]"),
            "self-search" => term::info("[self-search]"),
            "compute"     => term::warn("[compute]"),
            "synthesize"  => term::warn("[synthesize]"),
            "evaluate"    => term::warn("[evaluate]"),
            "checked"     => term::ok("[checked]"),
            _             => format!("[{}]", row.connection),
        };
        let source = if row.source.is_empty() {
            term::dim("(none)")
        } else {
            row.source.clone()
        };
        println!("  {}  {}  {}",
            term::bold(&row.code.to_string()),
            pad(row.label, 22),
            tag);
        println!("      {}", term::dim(&source));
    }
    println!();
    println!();

    if !confirm("Proceed with this configuration?", true) {
        eprintln!("Aborted.");
        std::process::exit(0);
    }

    ImportArgs {
        name,
        output,
        base_vectors,
        query_vectors,
        self_search,
        query_count,
        metadata,
        ground_truth,
        ground_truth_distances,
        metric,
        neighbors,
        seed,
        description,
        no_dedup,
        no_zero_check,
        no_filtered,
        normalize,
        force: true, // user already confirmed overwrite above
        base_convert_format,
        query_convert_format,
        compress_cache,
        sized_profiles,
        base_fraction,
        required_facets,
        provided_facets: seeds.provided_facets.clone(),
        round_digits: seeds.round_digits.unwrap_or(2),
        pedantic_dedup: seeds.pedantic_dedup.unwrap_or(false),
        selectivity,
        predicate_count,
        predicate_strategy,
        classic: seeds.classic,
        personality: "native".to_string(),
        synthesize_metadata,
        synthesis_mode,
        synthesis_format,
        metadata_fields,
        metadata_range_min,
        metadata_range_max,
        predicate_range_min,
        predicate_range_max,
    }
}

// ---------------------------------------------------------------------------
// Source file renaming (early, right after detection)
// ---------------------------------------------------------------------------

/// Prompt to underscore-prefix all detected source files that don't already
/// have a `_` prefix. Updates the paths in `detected` in place.
fn rename_detected_sources(detected: &mut DetectedRoles) {
    fn needs_prefix(path: &Option<PathBuf>) -> bool {
        path.as_ref().map_or(false, |p| {
            let fname = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            !fname.starts_with('_') && !fname.is_empty()
        })
    }

    fn do_rename(path: &mut Option<PathBuf>) {
        if let Some(ref p) = *path {
            let fname = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if !fname.starts_with('_') && !fname.is_empty() {
                let parent = p.parent().unwrap_or(Path::new("."));
                let dest = parent.join(format!("_{}", fname));
                match std::fs::rename(p, &dest) {
                    Ok(()) => {
                        println!("  {} → {}", p.display(), dest.display());
                        *path = Some(dest);
                    }
                    Err(e) => {
                        println!("  WARNING: rename {} failed: {}", p.display(), e);
                    }
                }
            }
        }
    }

    // Collect labels for display
    let labels: Vec<(&str, bool)> = vec![
        ("Base vectors", needs_prefix(&detected.base_vectors)),
        ("Query vectors", needs_prefix(&detected.query_vectors)),
        ("Neighbor indices", needs_prefix(&detected.neighbor_indices)),
        ("Neighbor distances", needs_prefix(&detected.neighbor_distances)),
        ("Metadata", needs_prefix(&detected.metadata)),
        ("Metadata predicates", needs_prefix(&detected.metadata_predicates)),
        ("Metadata results", needs_prefix(&detected.metadata_results)),
        ("Filtered neighbor indices", needs_prefix(&detected.filtered_neighbor_indices)),
        ("Filtered neighbor distances", needs_prefix(&detected.filtered_neighbor_distances)),
    ];

    let any_need = labels.iter().any(|(_, need)| *need);
    if !any_need {
        return;
    }

    println!();
    println!("--- Source file prefixing ---");
    println!("  Source files are prefixed with _ to exclude them from published datasets.");
    println!("  The following detected files do not have a _ prefix:");
    println!();

    let all_fields: [(&str, &Option<PathBuf>); 9] = [
        ("Base vectors", &detected.base_vectors),
        ("Query vectors", &detected.query_vectors),
        ("Neighbor indices", &detected.neighbor_indices),
        ("Neighbor distances", &detected.neighbor_distances),
        ("Metadata", &detected.metadata),
        ("Metadata predicates", &detected.metadata_predicates),
        ("Metadata results", &detected.metadata_results),
        ("Filtered neighbor indices", &detected.filtered_neighbor_indices),
        ("Filtered neighbor distances", &detected.filtered_neighbor_distances),
    ];
    for (label, path) in &all_fields {
        if let Some(ref p) = **path {
            let fname = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if !fname.starts_with('_') && !fname.is_empty() {
                println!("    {}:  {}", label, p.display());
            }
        }
    }
    println!();

    if confirm("Rename these files with _ prefix?", true) {
        do_rename(&mut detected.base_vectors);
        do_rename(&mut detected.query_vectors);
        do_rename(&mut detected.neighbor_indices);
        do_rename(&mut detected.neighbor_distances);
        do_rename(&mut detected.metadata);
        do_rename(&mut detected.metadata_predicates);
        do_rename(&mut detected.metadata_results);
        do_rename(&mut detected.filtered_neighbor_indices);
        do_rename(&mut detected.filtered_neighbor_distances);
    }
}

// ---------------------------------------------------------------------------
// Source file location
// ---------------------------------------------------------------------------

/// Prompt the user about where to keep a source data file.
///
/// Options:
/// 1. Move to .cache/ (recommended — keeps workspace clean, not published)
/// 2. Keep in place but rename with _ prefix (not published, stays accessible)
/// 3. Keep as-is (will be flagged as extraneous if not in pipeline manifest)
///
/// Returns the final path to use in the pipeline.
/// Track HDF5 container files that have already been renamed so we
/// only prompt once per actual file, not per dataset within the file.
static HDF5_RENAMES: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());

fn prompt_source_location(source: &Path, output_dir: &Path, label: &str) -> PathBuf {
    let source_str = source.to_string_lossy();

    // HDF5 # paths: rename the container file once, update all references
    if let Some(hash_pos) = source_str.rfind('#') {
        let file_part = &source_str[..hash_pos];
        let dataset_part = &source_str[hash_pos..]; // includes #

        // Check if this container was already renamed
        if let Ok(renames) = HDF5_RENAMES.lock() {
            if let Some((_, new_file)) = renames.iter().find(|(old, _)| old == file_part) {
                return PathBuf::from(format!("{}{}", new_file, dataset_part));
            }
        }

        let file_path = Path::new(file_part);
        let filename = file_path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Already prefixed or in cache
        if filename.starts_with('_') || file_part.contains(".cache/") {
            return source.to_path_buf();
        }

        // Prompt once for this container file
        println!();
        println!("  --- HDF5 source file ---");
        println!("  {} contains multiple datasets used by this pipeline.", filename);
        println!("  Source files should not be published. Options:");
        println!("    1. Rename with _ prefix (not published, stays accessible locally)");
        println!("    2. Keep as-is (may be flagged as extraneous by veks check)");
        let choice = prompt_with_default("  Choice [1/2]", "1");

        if choice == "1" {
            let parent = file_path.parent().unwrap_or(Path::new("."));
            let new_name = format!("_{}", filename);
            let dest = parent.join(&new_name);
            println!("  Renaming {} → {}", file_path.display(), dest.display());
            match std::fs::rename(file_path, &dest) {
                Ok(()) => {
                    println!("  Renamed successfully.");
                    let new_file = dest.to_string_lossy().to_string();
                    if let Ok(mut renames) = HDF5_RENAMES.lock() {
                        renames.push((file_part.to_string(), new_file.clone()));
                    }
                    return PathBuf::from(format!("{}{}", new_file, dataset_part));
                }
                Err(e) => {
                    println!("  WARNING: rename failed: {}", e);
                }
            }
        } else {
            // Record as "kept" so we don't ask again
            if let Ok(mut renames) = HDF5_RENAMES.lock() {
                renames.push((file_part.to_string(), file_part.to_string()));
            }
        }
        return source.to_path_buf();
    }

    let filename = source.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    if source_str.contains(".cache/") || source_str.contains("/.cache/")
        || filename.starts_with('_')
    {
        return source.to_path_buf();
    }

    println!();
    println!("  --- {} source file location ---", label);
    println!("  Source files should not be published. Options:");
    println!("    1. Rename with _ prefix (not published, stays accessible locally)");
    println!("    2. Move to .cache/ (not published, workspace stays clean)");
    println!("    3. Keep as-is (may be flagged as extraneous by veks check)");
    let choice = prompt_with_default("  Choice [1/2/3]", "1");

    match choice.as_str() {
        "1" => {
            let parent = source.parent().unwrap_or(Path::new("."));
            let new_name = format!("_{}", filename);
            let dest = parent.join(&new_name);
            println!("  Renaming {} → {}", source.display(), dest.display());
            match std::fs::rename(source, &dest) {
                Ok(()) => {
                    println!("  Renamed successfully.");
                    dest
                }
                Err(e) => {
                    println!("  WARNING: rename failed: {}", e);
                    source.to_path_buf()
                }
            }
        }
        "2" => {
            let cache_dir = output_dir.join(".cache");
            let dest = cache_dir.join(&filename);
            println!("  Will move {} → {}", source.display(), dest.display());

            if let Err(e) = std::fs::create_dir_all(&cache_dir) {
                println!("  WARNING: failed to create .cache/: {}", e);
                println!("  Falling back to keeping file in place.");
                return source.to_path_buf();
            }
            match std::fs::rename(source, &dest) {
                Ok(()) => {
                    println!("  Moved successfully.");
                    dest
                }
                Err(_) => {
                    println!("  Cross-filesystem move — copying...");
                    match std::fs::copy(source, &dest) {
                        Ok(_) => {
                            let _ = std::fs::remove_file(source);
                            println!("  Copied and removed original.");
                            dest
                        }
                        Err(e) => {
                            println!("  WARNING: copy failed: {}", e);
                            println!("  Keeping file in place.");
                            source.to_path_buf()
                        }
                    }
                }
            }
        }
        _ => {
            println!("  Keeping file as-is.");
            source.to_path_buf()
        }
    }
}

// ---------------------------------------------------------------------------
// Vector probing
// ---------------------------------------------------------------------------

/// Probe a vector file for its record count using mmap (no full read).
fn probe_vector_count(path: &Path) -> Option<u64> {
    let format = VecFormat::detect(path)?;
    if path.is_dir() {
        // For directories (npy, parquet), open a source reader to get count
        let source = veks_core::formats::reader::open_source(path, format, 1, None).ok()?;
        return source.record_count();
    }
    let meta = veks_core::formats::reader::probe_source(path, format).ok()?;
    meta.record_count
}

fn probe_vector_dim(path: &Path) -> Option<u32> {
    let format = VecFormat::detect(path)?;
    if path.is_dir() {
        let source = veks_core::formats::reader::open_source(path, format, 1, Some(1)).ok()?;
        return Some(source.dimension());
    }
    let meta = veks_core::formats::reader::probe_source(path, format).ok()?;
    Some(meta.dimension)
}

/// Format a count as a clean suffix label (e.g., 407000000 → "407m").
fn format_count_label(n: u64) -> String {
    if n >= 1_000_000_000 && n % 1_000_000_000 == 0 {
        format!("{}b", n / 1_000_000_000)
    } else if n >= 1_000_000 && n % 1_000_000 == 0 {
        format!("{}m", n / 1_000_000)
    } else if n >= 1_000 && n % 1_000 == 0 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}

// ---------------------------------------------------------------------------
// Precision confirmation
// ---------------------------------------------------------------------------

/// Element type label for a given element size.
fn element_label(elem_size: usize) -> &'static str {
    match elem_size {
        2 => "float16 (f16, half-precision)",
        4 => "float32 (f32, single-precision)",
        8 => "float64 (f64, double-precision)",
        _ => "unknown",
    }
}

/// The xvec floating-point formats in ascending precision order.
const FLOAT_XVEC_FORMATS: &[(VecFormat, usize, &str)] = &[
    (VecFormat::Mvec, 2, "mvec / float16"),
    (VecFormat::Fvec, 4, "fvec / float32"),
    (VecFormat::Dvec, 8, "dvec / float64"),
];

/// Check whether the user is satisfied with the native precision of an xvec
/// vector file, and advise on up-conversion vs down-conversion if not.
///
/// Returns the target format name (e.g., `"mvec"`) if conversion is needed,
/// or `None` if the file should be used as-is.
fn check_vector_precision(label: &str, path: &Path) -> Option<String> {
    let format = match VecFormat::detect(path) {
        Some(f) => f,
        None => return None, // not a recognized format — nothing to confirm
    };

    // Only relevant for floating-point xvec formats
    let current = match format {
        VecFormat::Fvec => (4usize, "fvec / float32"),
        VecFormat::Mvec => (2, "mvec / float16"),
        VecFormat::Dvec => (8, "dvec / float64"),
        _ => return None, // ivec, bvec, svec, slab, npy, parquet — skip
    };

    // Probe for dimensions and record count
    let meta = match crate::formats::reader::probe_source(path, format) {
        Ok(m) => m,
        Err(_) => return None,
    };

    println!();
    println!("  {} format: {} — {} elements, dim={}, {} records",
        label, current.1, element_label(current.0),
        meta.dimension,
        meta.record_count.map_or("unknown".to_string(), |n| n.to_string()),
    );

    if confirm(&format!("  Use {} as-is ({})?", label.to_lowercase(), current.1), true) {
        return None;
    }

    // User wants a different precision — show options
    println!();
    println!("  Available floating-point precisions:");
    for (i, (_, elem_size, name)) in FLOAT_XVEC_FORMATS.iter().enumerate() {
        let marker = if *elem_size == current.0 { " (current)" } else { "" };
        println!("    {}. {}{}", i + 1, name, marker);
    }

    let choice_str = prompt_optional("  Target precision [1/2/3]");
    let chosen = match choice_str.as_deref() {
        Some("1") => Some(&FLOAT_XVEC_FORMATS[0]),
        Some("2") => Some(&FLOAT_XVEC_FORMATS[1]),
        Some("3") => Some(&FLOAT_XVEC_FORMATS[2]),
        _ => None,
    };

    let (target_fmt, target_size, target_name) = match chosen {
        Some((fmt, size, name)) => (*fmt, *size, *name),
        None => {
            println!("  No change — keeping {} format.", current.1);
            return None;
        }
    };

    if target_size == current.0 {
        println!("  No change — already {}.", current.1);
        return None;
    }

    if target_size > current.0 {
        // Up-conversion (e.g., f16 → f32)
        println!();
        println!("  NOTE: Up-converting from {} to {} does NOT improve accuracy.", current.1, target_name);
        println!("  The extra precision bits will be zero-filled. This is only useful");
        println!("  for compatibility with tools that require a specific element width.");
        println!();
        println!("  A convert step will be added to the pipeline.");
    } else {
        // Down-conversion (e.g., f32 → f16)
        println!();
        println!("  WARNING: Down-converting from {} to {} is a lossy operation.", current.1, target_name);
        println!("  Precision will be permanently reduced. IEEE 754 rounding rules apply");
        println!("  (round-to-nearest-even). Values outside the target range will saturate");
        println!("  to ±Inf.");
        println!();
        println!("  A convert step will be added to the pipeline.");
    }

    Some(target_fmt.name().to_string())
}

// ---------------------------------------------------------------------------
// Filename-keyword role detection
// ---------------------------------------------------------------------------

/// Auto-detected file-to-role assignments based on filename keyword hints.
///
/// After `scan_candidates()` finds all recognized data files, `detect_roles()`
/// examines each filename stem for keyword substrings and assigns files to
/// dataset roles. This eliminates unstable multi-select prompts and makes
/// `-y` (auto-accept) mode produce valid results when filenames contain
/// conventional hints like `base`, `query`, `groundtruth`, etc.
#[derive(Debug, Default)]
pub(crate) struct DetectedRoles {
    pub(crate) base_vectors: Option<PathBuf>,
    pub(crate) query_vectors: Option<PathBuf>,
    pub(crate) neighbor_indices: Option<PathBuf>,
    pub(crate) neighbor_distances: Option<PathBuf>,
    pub(crate) metadata: Option<PathBuf>,
    pub(crate) metadata_predicates: Option<PathBuf>,
    pub(crate) metadata_results: Option<PathBuf>,
    pub(crate) filtered_neighbor_indices: Option<PathBuf>,
    pub(crate) filtered_neighbor_distances: Option<PathBuf>,
    pub(crate) unassigned: Vec<PathBuf>,
}

impl DetectedRoles {
    /// Returns `true` if at least one role was detected.
    pub(crate) fn any_detected(&self) -> bool {
        self.base_vectors.is_some()
            || self.query_vectors.is_some()
            || self.neighbor_indices.is_some()
            || self.neighbor_distances.is_some()
            || self.metadata.is_some()
            || self.metadata_predicates.is_some()
            || self.metadata_results.is_some()
            || self.filtered_neighbor_indices.is_some()
            || self.filtered_neighbor_distances.is_some()
    }

    /// Print detected assignments for user confirmation.
    fn print_summary(&self) {
        if let Some(ref p) = self.base_vectors {
            println!("  Base vectors:              {}", p.display());
        }
        if let Some(ref p) = self.query_vectors {
            println!("  Query vectors:             {}", p.display());
        }
        if let Some(ref p) = self.neighbor_indices {
            println!("  Neighbor indices (GT):     {}", p.display());
        }
        if let Some(ref p) = self.neighbor_distances {
            println!("  Neighbor distances (GT):   {}", p.display());
        }
        if let Some(ref p) = self.metadata {
            println!("  Metadata content:          {}", p.display());
        }
        if let Some(ref p) = self.metadata_predicates {
            println!("  Metadata predicates:       {}", p.display());
        }
        if let Some(ref p) = self.metadata_results {
            println!("  Metadata results:          {}", p.display());
        }
        if let Some(ref p) = self.filtered_neighbor_indices {
            println!("  Filtered neighbor indices: {}", p.display());
        }
        if let Some(ref p) = self.filtered_neighbor_distances {
            println!("  Filtered neighbor distances: {}", p.display());
        }
        for p in &self.unassigned {
            println!("  (unassigned):              {}", p.display());
        }
    }
}

/// Vector format names for matching constraints.
const VECTOR_FORMATS: &[&str] = &["fvec", "ivec", "mvec", "bvec", "dvec", "svec", "npy"];
const FLOAT_VECTOR_FORMATS: &[&str] = &["fvec", "dvec", "mvec"];

/// Detect file roles from filename keyword hints.
///
/// Examines each candidate's filename stem (lowercased, `_` prefix stripped)
/// for keyword substrings and assigns it to a dataset role. If two files
/// claim the same role, neither is assigned (ambiguous — the wizard falls
/// through to manual selection for that role).
pub(crate) fn detect_roles(candidates: &[(PathBuf, String, u64)]) -> DetectedRoles {
    /// Role tag used during detection before resolving ambiguities.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Role {
        BaseVectors,
        QueryVectors,
        NeighborIndices,
        NeighborDistances,
        MetadataContent,
        MetadataPredicates,
        MetadataResults,
        FilteredNeighborIndices,
        FilteredNeighborDistances,
    }

    let mut assignments: Vec<(Role, &PathBuf)> = Vec::new();
    let mut unassigned: Vec<PathBuf> = Vec::new();

    for (path, fmt, _) in candidates {
        // For HDF5 #dataset paths, use the dataset name for role detection
        let path_str = path.to_string_lossy();
        let stem = if let Some(hash_pos) = path_str.rfind('#') {
            path_str[hash_pos + 1..].to_lowercase()
        } else {
            path.file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default()
                .to_string()
        };
        // Strip leading `_` prefix (source files renamed by prior import)
        let stem = stem.strip_prefix('_').unwrap_or(&stem);
        let fmt_str = fmt.as_str();
        let is_vector = VECTOR_FORMATS.contains(&fmt_str);
        let is_float_vector = FLOAT_VECTOR_FORMATS.contains(&fmt_str);
        let is_ivec = fmt_str == "ivec";
        let is_slab = fmt_str == "slab";
        let is_slab_or_parquet = is_slab || fmt_str == "parquet";

        // Split on common delimiters for word-boundary matching.
        // This avoids false positives like "test" matching "base_test"
        // or "gt" matching "lighgt".
        let tokens: Vec<&str> = stem.split(|c: char| c == '_' || c == '-' || c == '.')
            .filter(|s| !s.is_empty())
            .collect();
        let has_token = |keyword: &str| tokens.iter().any(|t| *t == keyword);
        let has_token_or_substring = |keyword: &str| stem.contains(keyword);

        // Some keywords are safe as substrings (long, unambiguous);
        // short ones like "gt", "test" use token-boundary matching.
        let has_filtered = has_token("filtered") || has_token("filter");
        let has_indices = has_token("indices") || has_token_or_substring("neighbors")
            || has_token("gt") || has_token_or_substring("groundtruth");
        let has_distances = has_token_or_substring("distance");
        let has_base = has_token("base") || has_token("train");
        let has_query = has_token("query") || has_token("queries")
            || has_token("test");
        let has_metadata = has_token_or_substring("metadata") || has_token("content");
        let has_predicates = has_token_or_substring("predicate");
        let has_results = has_token_or_substring("result");

        // Filtered variants take priority over non-filtered
        let role = if has_filtered && has_indices && is_ivec {
            Some(Role::FilteredNeighborIndices)
        } else if has_filtered && has_distances && is_float_vector {
            Some(Role::FilteredNeighborDistances)
        } else if has_indices && !has_filtered && is_ivec {
            Some(Role::NeighborIndices)
        } else if has_distances && !has_filtered && !has_indices && is_float_vector {
            Some(Role::NeighborDistances)
        } else if has_predicates && is_slab {
            Some(Role::MetadataPredicates)
        } else if has_results && is_slab {
            Some(Role::MetadataResults)
        } else if has_metadata && !has_predicates && !has_results && is_slab_or_parquet {
            Some(Role::MetadataContent)
        } else if has_base && is_vector {
            // "base" takes priority over "query"/"test" when both present
            Some(Role::BaseVectors)
        } else if has_query && is_vector {
            Some(Role::QueryVectors)
        } else {
            None
        };

        match role {
            Some(r) => assignments.push((r, path)),
            None => unassigned.push(path.clone()),
        }
    }

    // Resolve ambiguities: if two files claim the same role, neither wins.
    let mut result = DetectedRoles::default();
    let mut seen: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    for (i, (role, _)) in assignments.iter().enumerate() {
        let key = *role as u8;
        if seen.contains_key(&key) {
            // Mark as ambiguous — we'll push both to unassigned
            seen.insert(key, usize::MAX);
        } else {
            seen.insert(key, i);
        }
    }

    for (i, (role, path)) in assignments.iter().enumerate() {
        let key = *role as u8;
        if seen.get(&key) == Some(&usize::MAX) {
            // Ambiguous — push to unassigned
            unassigned.push((*path).clone());
            continue;
        }
        if seen.get(&key) != Some(&i) {
            // Not the winner (shouldn't happen, but defensive)
            unassigned.push((*path).clone());
            continue;
        }
        let p = (*path).clone();
        match role {
            Role::BaseVectors => result.base_vectors = Some(p),
            Role::QueryVectors => result.query_vectors = Some(p),
            Role::NeighborIndices => result.neighbor_indices = Some(p),
            Role::NeighborDistances => result.neighbor_distances = Some(p),
            Role::MetadataContent => result.metadata = Some(p),
            Role::MetadataPredicates => result.metadata_predicates = Some(p),
            Role::MetadataResults => result.metadata_results = Some(p),
            Role::FilteredNeighborIndices => result.filtered_neighbor_indices = Some(p),
            Role::FilteredNeighborDistances => result.filtered_neighbor_distances = Some(p),
        }
    }

    result.unassigned = unassigned;
    result
}

// ---------------------------------------------------------------------------
// File scanning
// ---------------------------------------------------------------------------

/// Scan a directory for recognized data files.
fn scan_candidates(dir: &Path) -> Vec<(PathBuf, String, u64)> {
    let mut results = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return results,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() { continue; }

        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden files, known non-data files, and pipeline artifacts.
        if name_str.starts_with('.') || name_str == "dataset.yaml"
            || name_str == "variables.yaml"
            || name_str.ends_with(".json") || name_str.ends_with(".yaml")
            || name_str.ends_with(".yml")
            || name_str.ends_with(".mref") || name_str.ends_with(".mrkl")
            || name_str.ends_with(".log")
            || veks_core::filters::is_infrastructure_file(&name_str)
        {
            continue;
        }
        // Skip pipeline-generated artifact names. These are canonical facet
        // output names that the pipeline produces — not source data.
        {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if matches!(stem,
                "base_vectors" | "base_vectors_raw"
                | "query_vectors" | "query_vectors_raw"
                | "query_vectors_clean"
                | "neighbor_indices" | "neighbor_distances"
                | "filtered_neighbor_indices" | "filtered_neighbor_distances"
                | "metadata_content" | "metadata_predicates" | "metadata_indices"
            ) {
                continue;
            }
        }

        if let Some(format) = VecFormat::detect_from_path(&path) {
            if format == VecFormat::Hdf5 {
                // HDF5 is a container — list internal datasets as separate candidates.
                // Each gets a path of the form `file.hdf5#dataset_name`.
                if let Ok(datasets) = veks_core::formats::reader::hdf5::list_datasets(&path) {
                    for (ds_name, rows, cols, elem_size) in datasets {
                        let ds_size = rows * cols as u64 * elem_size as u64;
                        // Map element size to the xvec format name that the
                        // downstream role detection uses.
                        let fmt_name = match elem_size {
                            4 => {
                                // Could be f32 or i32 — guess from dataset name
                                if ds_name.contains("indic") || ds_name.contains("neighbor_ind") {
                                    "ivec"
                                } else {
                                    "fvec"
                                }
                            }
                            8 => "dvec",
                            2 => "mvec",
                            1 => "bvec",
                            _ => "fvec",
                        };
                        let rel = PathBuf::from(format!(
                            "{}#{}",
                            entry.file_name().to_string_lossy(),
                            ds_name,
                        ));
                        results.push((rel, fmt_name.to_string(), ds_size));
                    }
                }
            } else {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                // Store as relative path (just the filename within the scan dir)
                let rel = PathBuf::from(entry.file_name());
                results.push((rel, format.name().to_string(), size));
            }
        }
    }

    // Also check for npy/parquet directories
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() { continue; }
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.') { continue; }

            if let Some(format) = VecFormat::detect_from_directory(&path) {
                let size = dir_size(&path);
                let rel = PathBuf::from(entry.file_name());
                results.push((rel, format.name().to_string(), size));
            }
        }
    }

    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

/// Total size of files in a directory (non-recursive).
fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| entries
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Prompt helpers
// ---------------------------------------------------------------------------

/// Prompt with the default pre-filled as editable text.
/// Uses crossterm raw mode so the user can backspace/edit the default.
fn prompt_with_prefill(label: &str, default: &str) -> String {
    if AUTO_ACCEPT.load(Ordering::Relaxed) {
        eprintln!("{}: {}", label, default);
        return default.to_string();
    }

    use crossterm::{terminal, event::{self, Event, KeyCode, KeyModifiers}};
    use std::io::Write;

    eprint!("{}: {}", label, default);
    io::stderr().flush().unwrap_or(());

    let mut buf = default.to_string();

    if terminal::enable_raw_mode().is_err() {
        // Fallback: plain prompt
        let _ = terminal::disable_raw_mode();
        eprint!("\r{} [{}]: ", label, default);
        io::stderr().flush().unwrap_or(());
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap_or(0);
        let trimmed = input.trim();
        return if trimmed.is_empty() { default.to_string() } else { trimmed.to_string() };
    }

    loop {
        match event::read() {
            Ok(Event::Key(key)) => match key.code {
                KeyCode::Enter => break,
                KeyCode::Backspace => {
                    if !buf.is_empty() {
                        buf.pop();
                        eprint!("\x08 \x08"); // erase char
                        io::stderr().flush().unwrap_or(());
                    }
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let _ = terminal::disable_raw_mode();
                    eprintln!();
                    std::process::exit(130);
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    // Clear line
                    let clear = "\x08 \x08".repeat(buf.len());
                    eprint!("{}", clear);
                    io::stderr().flush().unwrap_or(());
                    buf.clear();
                }
                KeyCode::Char(c) => {
                    buf.push(c);
                    eprint!("{}", c);
                    io::stderr().flush().unwrap_or(());
                }
                _ => {}
            },
            _ => {}
        }
    }

    let _ = terminal::disable_raw_mode();
    eprintln!(); // newline after Enter

    let trimmed = buf.trim().to_string();
    if trimmed.is_empty() { default.to_string() } else { trimmed }
}

/// Prompt with a default value. Returns the default if the user presses Enter
/// or if auto-accept mode is active.
fn prompt_with_default(label: &str, default: &str) -> String {
    if AUTO_ACCEPT.load(Ordering::Relaxed) {
        eprintln!("{} [{}]: {}", label, default, default);
        return default.to_string();
    }
    eprint!("{} [{}]: ", label, default);
    io::stderr().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    let trimmed = input.trim();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Prompt for an optional string. Returns None if empty or in auto-accept mode.
fn prompt_optional(label: &str) -> Option<String> {
    if AUTO_ACCEPT.load(Ordering::Relaxed) {
        eprintln!("{}: (default)", label);
        return None;
    }
    eprint!("{}: ", label);
    io::stderr().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    let trimmed = input.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Prompt for an optional path. Returns None if empty or in auto-accept mode.
fn prompt_optional_path(label: &str) -> Option<PathBuf> {
    prompt_optional(label).map(PathBuf::from)
}

/// Prompt for a path with a default.
fn prompt_path_with_default(label: &str, default: &str) -> PathBuf {
    PathBuf::from(prompt_with_default(label, default))
}

/// Yes/no confirmation prompt. Returns `default` on empty input or in
/// auto-accept mode.
fn confirm(question: &str, default: bool) -> bool {
    if AUTO_ACCEPT.load(Ordering::Relaxed) {
        let answer = if default { "Y" } else { "N" };
        eprintln!("{} [{}]: {}", question, if default { "Y/n" } else { "y/N" }, answer);
        return default;
    }
    let hint = if default { "Y/n" } else { "y/N" };
    eprint!("{} [{}]: ", question, hint);
    io::stderr().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    let trimmed = input.trim().to_lowercase();
    if trimmed.is_empty() {
        default
    } else {
        trimmed.starts_with('y')
    }
}

/// Exposed for testing.
pub(super) fn scan_candidates_for_test(dir: &Path) -> Vec<(PathBuf, String, u64)> {
    scan_candidates(dir)
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal fvec file.
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

    #[test]
    fn scan_finds_fvec_files() {
        let dir = tempfile::tempdir().unwrap();
        write_fvec(&dir.path().join("base.fvec"), 5, 3);
        write_fvec(&dir.path().join("query.mvec"), 2, 3);
        // Non-data file should be ignored
        std::fs::write(dir.path().join("readme.txt"), b"ignore").unwrap();
        std::fs::write(dir.path().join(".hidden"), b"ignore").unwrap();

        let results = scan_candidates(dir.path());
        let formats: Vec<&str> = results.iter().map(|(_, f, _)| f.as_str()).collect();
        assert!(formats.contains(&"fvec"), "should find fvec: {:?}", formats);
        assert!(formats.contains(&"mvec"), "should find mvec: {:?}", formats);
        assert_eq!(results.len(), 2, "should only find 2 data files");
    }

    #[test]
    fn scan_finds_npy_directory() {
        let dir = tempfile::tempdir().unwrap();
        let npy_dir = dir.path().join("embeddings");
        std::fs::create_dir(&npy_dir).unwrap();
        std::fs::write(npy_dir.join("shard_0.npy"), b"numpy data").unwrap();
        std::fs::write(npy_dir.join("shard_1.npy"), b"numpy data").unwrap();

        let results = scan_candidates(dir.path());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "npy");
        assert!(results[0].0.ends_with("embeddings"));
    }

    #[test]
    fn scan_ignores_hidden_and_yaml() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("dataset.yaml"), b"name: test").unwrap();
        std::fs::write(dir.path().join("variables.yaml"), b"seed: 42").unwrap();
        std::fs::write(dir.path().join(".gitignore"), b"*.tmp").unwrap();
        std::fs::write(dir.path().join("report.json"), b"{}").unwrap();

        let results = scan_candidates(dir.path());
        assert!(results.is_empty(), "should skip yaml/json/hidden: {:?}", results);
    }

    #[test]
    fn scan_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let results = scan_candidates(dir.path());
        assert!(results.is_empty());
    }

    // -- detect_roles tests ------------------------------------------------

    /// Helper to build a candidate list from (filename, format) pairs.
    fn make_candidates(items: &[(&str, &str)]) -> Vec<(PathBuf, String, u64)> {
        items.iter()
            .map(|(name, fmt)| (PathBuf::from(name), fmt.to_string(), 1024))
            .collect()
    }

    #[test]
    fn detect_roles_base_and_query() {
        let candidates = make_candidates(&[
            ("_base_test.mvec", "mvec"),
            ("query_vectors.mvec", "mvec"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.base_vectors.as_deref(), Some(Path::new("_base_test.mvec")));
        assert_eq!(roles.query_vectors.as_deref(), Some(Path::new("query_vectors.mvec")));
        assert!(roles.unassigned.is_empty());
    }

    #[test]
    fn detect_roles_gt_and_distances() {
        let candidates = make_candidates(&[
            ("groundtruth_indices.ivec", "ivec"),
            ("distances.fvec", "fvec"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.neighbor_indices.as_deref(), Some(Path::new("groundtruth_indices.ivec")));
        assert_eq!(roles.neighbor_distances.as_deref(), Some(Path::new("distances.fvec")));
    }

    #[test]
    fn detect_roles_filtered_knn() {
        let candidates = make_candidates(&[
            ("filtered_neighbors.ivec", "ivec"),
            ("filtered_distances.fvec", "fvec"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.filtered_neighbor_indices.as_deref(), Some(Path::new("filtered_neighbors.ivec")));
        assert_eq!(roles.filtered_neighbor_distances.as_deref(), Some(Path::new("filtered_distances.fvec")));
        // Should NOT be assigned to non-filtered roles
        assert!(roles.neighbor_indices.is_none());
        assert!(roles.neighbor_distances.is_none());
    }

    #[test]
    fn detect_roles_metadata_vs_predicates() {
        let candidates = make_candidates(&[
            ("metadata_content.slab", "slab"),
            ("predicates.slab", "slab"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.metadata.as_deref(), Some(Path::new("metadata_content.slab")));
        assert_eq!(roles.metadata_predicates.as_deref(), Some(Path::new("predicates.slab")));
    }

    #[test]
    fn detect_roles_metadata_results() {
        let candidates = make_candidates(&[
            ("metadata.slab", "slab"),
            ("results.slab", "slab"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.metadata.as_deref(), Some(Path::new("metadata.slab")));
        assert_eq!(roles.metadata_results.as_deref(), Some(Path::new("results.slab")));
    }

    #[test]
    fn detect_roles_ambiguous_duplicate() {
        let candidates = make_candidates(&[
            ("base_v1.fvec", "fvec"),
            ("base_v2.fvec", "fvec"),
        ]);
        let roles = detect_roles(&candidates);
        // Neither should be assigned — both are ambiguous
        assert!(roles.base_vectors.is_none(), "ambiguous base should be None");
        assert_eq!(roles.unassigned.len(), 2);
    }

    #[test]
    fn detect_roles_underscore_prefix_stripped() {
        let candidates = make_candidates(&[
            ("_base_test.mvec", "mvec"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.base_vectors.as_deref(), Some(Path::new("_base_test.mvec")));
    }

    #[test]
    fn detect_roles_no_keywords() {
        let candidates = make_candidates(&[
            ("embeddings.fvec", "fvec"),
            ("data.mvec", "mvec"),
        ]);
        let roles = detect_roles(&candidates);
        assert!(!roles.any_detected());
        assert_eq!(roles.unassigned.len(), 2);
    }

    #[test]
    fn detect_roles_train_keyword() {
        let candidates = make_candidates(&[
            ("train_vectors.fvec", "fvec"),
            ("test_queries.mvec", "mvec"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.base_vectors.as_deref(), Some(Path::new("train_vectors.fvec")));
        assert_eq!(roles.query_vectors.as_deref(), Some(Path::new("test_queries.mvec")));
    }

    #[test]
    fn detect_roles_gt_alias() {
        let candidates = make_candidates(&[
            ("gt_neighbors.ivec", "ivec"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.neighbor_indices.as_deref(), Some(Path::new("gt_neighbors.ivec")));
    }

    #[test]
    fn detect_roles_format_constraints() {
        // ivec file with "base" keyword should NOT match BaseVectors
        // because ivec is typically indices, not vectors — but our format
        // constraint allows it (ivec IS a vector format). This test
        // documents the current behavior.
        let candidates = make_candidates(&[
            ("base_indices.ivec", "ivec"),
        ]);
        let roles = detect_roles(&candidates);
        // ivec is in VECTOR_FORMATS, so "base" keyword matches BaseVectors
        // unless "indices" keyword takes priority → NeighborIndices wins
        // because has_indices is checked before has_base
        assert_eq!(roles.neighbor_indices.as_deref(), Some(Path::new("base_indices.ivec")));
        assert!(roles.base_vectors.is_none());
    }

    #[test]
    fn detect_roles_parquet_metadata() {
        let candidates = make_candidates(&[
            ("metadata.parquet", "parquet"),
        ]);
        let roles = detect_roles(&candidates);
        assert_eq!(roles.metadata.as_deref(), Some(Path::new("metadata.parquet")));
    }
}
