// Copyright (c) DataStax, Inc.
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

/// Run the wizard with auto-accept disabled (fully interactive).
pub fn run_wizard() -> ImportArgs {
    run_wizard_with_options(false, false)
}

/// Run the interactive import wizard.
///
/// - `auto_accept`: all prompts return their default value (-y flag)
/// - `auto_mode`: strict mode — all candidate files must be recognized by
///   role keywords, underscore-prefix renaming is assumed, and unrecognized
///   files cause a hard stop with guidance on naming conventions.
pub fn run_wizard_with_options(auto_accept: bool, auto_mode: bool) -> ImportArgs {
    AUTO_ACCEPT.store(auto_accept, Ordering::Relaxed);
    AUTO_MODE.store(auto_mode, Ordering::Relaxed);
    if auto_mode {
        println!("=== veks datasets import — auto mode ===");
    } else if auto_accept {
        println!("=== veks datasets import — auto-accept mode (-y) ===");
    }
    println!("=== veks datasets import — interactive wizard ===");
    println!();

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Pre-check: warn if dataset.yaml already exists
    if cwd.join("dataset.yaml").exists() {
        println!("Warning: dataset.yaml already exists in this directory.");
        if !confirm("Overwrite it?", false) {
            eprintln!("Aborted.");
            std::process::exit(0);
        }
        println!();
    }

    // Scan for candidate data files
    let candidates = scan_candidates(&cwd);
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
    let detected = detect_roles(&candidates);

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
        println!("Detected file roles:");
        detected.print_summary();
        println!();
        confirm("Use detected assignments?", true)
    } else {
        false
    };

    // ── Dataset name ─────────────────────────────────────────────────
    let dir_name = cwd.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "my-dataset".to_string());
    let name = prompt_with_default("Dataset name", &dir_name);

    // ── Description ──────────────────────────────────────────────────
    let description = prompt_optional("Description (optional)");

    // ── Output directory ─────────────────────────────────────────────
    let output = prompt_path_with_default("Output directory", &cwd.to_string_lossy());

    // ── Base vectors ─────────────────────────────────────────────────
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

    // ── Source file location ─────────────────────────────────────────
    let base_vectors = if let Some(bv) = base_vectors {
        Some(prompt_source_location(&bv, &output, "Base vectors"))
    } else {
        None
    };

    // ── Precision confirmation for xvec formats ─────────────────────
    // When the source is already an xvec file, confirm with the user that
    // the native precision is correct for their use case.
    let base_convert_format = if let Some(ref bv) = base_vectors {
        check_vector_precision("Base vectors", bv)
    } else {
        None
    };

    // ── Query source ─────────────────────────────────────────────────
    println!();
    println!("--- Query vectors ---");
    let (query_vectors, self_search, query_count) = if base_vectors.is_some() {
        if roles_accepted && detected.query_vectors.is_some() {
            // Detected a separate query file — use it directly
            let qv = detected.query_vectors.as_ref().unwrap().clone();
            println!("  Using detected: {}", qv.display());
            (Some(qv), false, 10000u32)
        } else {
            // Default: self-search when no query file detected
            let default_choice = if roles_accepted { "1" } else { "1" };
            println!("  Options:");
            println!("    1. Self-search — extract queries from base via shuffle (recommended)");
            println!("    2. Separate query file");
            println!("    3. No query vectors");
            let choice = prompt_with_default("Choice [1/2/3]", default_choice);
            match choice.as_str() {
                "2" => {
                    let qv = prompt_optional_path("Path to query vectors");
                    let qv = qv.map(|p| prompt_source_location(&p, &output, "Query vectors"));
                    (qv, false, 10000u32)
                }
                "3" => (None, false, 10000),
                _ => {
                    let qc_str = prompt_with_default("Query count", "10000");
                    let qc = qc_str.parse().unwrap_or(10000);
                    (None, true, qc)
                }
            }
        }
    } else {
        (None, false, 10000)
    };

    let query_convert_format = if let Some(ref qv) = query_vectors {
        check_vector_precision("Query vectors", qv)
    } else {
        None
    };

    // ── Metadata ─────────────────────────────────────────────────────
    println!();
    println!("--- Metadata ---");
    let metadata = if roles_accepted && detected.metadata.is_some() {
        let m = detected.metadata.as_ref().unwrap().clone();
        println!("  Using detected: {}", m.display());
        Some(m)
    } else {
        let meta_candidates: Vec<&(PathBuf, String, u64)> = candidates.iter()
            .filter(|(_, fmt, _)| fmt == "slab" || fmt == "parquet")
            .collect();

        if !meta_candidates.is_empty() {
            println!("  Found metadata candidates:");
            for (path, fmt, size) in &meta_candidates {
                println!("    {} ({}, {})", path.display(), fmt, format_size(*size));
            }
            if confirm("Include metadata in pipeline?", true) {
                if meta_candidates.len() == 1 {
                    Some(meta_candidates[0].0.clone())
                } else {
                    prompt_optional_path("Path to metadata source")
                }
            } else {
                None
            }
        } else {
            if confirm("Include metadata source?", false) {
                prompt_optional_path("Path to metadata")
            } else {
                None
            }
        }
    };

    // ── Ground truth ─────────────────────────────────────────────────
    println!();
    println!("--- Ground truth ---");
    let has_queries = query_vectors.is_some() || self_search;
    let (ground_truth, ground_truth_distances) = if has_queries {
        if roles_accepted && detected.neighbor_indices.is_some() {
            // Detected pre-computed ground truth — use it
            let gt = detected.neighbor_indices.as_ref().unwrap().clone();
            println!("  Using detected ground truth indices: {}", gt.display());
            let gtd = if let Some(ref d) = detected.neighbor_distances {
                println!("  Using detected ground truth distances: {}", d.display());
                Some(d.clone())
            } else {
                None
            };
            (Some(gt), gtd)
        } else {
            let gt_candidates: Vec<&(PathBuf, String, u64)> = candidates.iter()
                .filter(|(p, fmt, _)| {
                    fmt == "ivec" && p.to_string_lossy().contains("neighbor")
                })
                .collect();

            if !gt_candidates.is_empty() {
                println!("  Found potential ground truth:");
                for (path, _, size) in &gt_candidates {
                    println!("    {} ({})", path.display(), format_size(*size));
                }
            }

            if confirm("Compute KNN ground truth? (choose N to provide pre-computed)", true) {
                (None, None)
            } else {
                let gt = prompt_optional_path("Path to ground truth indices (ivec)");
                let gtd = prompt_optional_path("Path to ground truth distances (fvec)");
                (gt, gtd)
            }
        }
    } else {
        println!("  No query vectors — skipping ground truth.");
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
    println!("  The distance metric describes how the source vectors were trained.");
    println!("  Common metrics:");
    println!("    L2         — Euclidean distance (most embedding models)");
    println!("    Cosine     — Cosine similarity (angular distance)");
    println!("    DotProduct — Inner product (often used with normalized vectors)");
    println!();
    let source_metric = prompt_with_default(
        "What distance metric does this data use?",
        "L2",
    );

    // Normalization detection and metric implications
    let (normalize, metric) = if let Some(ref bv) = base_vectors {
        eprint!("Sampling vectors for normalization detection... ");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        if let Some((is_normalized, sample_count, mean_norm)) = super::import::detect_normalized(bv) {
            if is_normalized {
                eprintln!("done.");
                println!();
                println!("  Vectors are L2-normalized (mean norm={:.4}, n={})", mean_norm, sample_count);
                if source_metric.eq_ignore_ascii_case("Cosine") {
                    println!("  Since vectors are already normalized, Cosine and DotProduct are equivalent.");
                    println!("  KNN will use Cosine as specified.");
                }
                (false, source_metric)
            } else {
                eprintln!("done.");
                println!();
                println!("  Vectors are NOT L2-normalized (mean norm={:.4}, n={})", mean_norm, sample_count);
                println!();
                println!("  L2-normalizing converts vectors to unit length. This changes distance behavior:");
                println!("    - L2 distance ranking becomes equivalent to Cosine ranking");
                println!("    - DotProduct becomes equivalent to Cosine similarity");
                println!("    - Original magnitude information is lost (irreversible)");
                println!();
                if confirm("L2-normalize vectors during extraction?", false) {
                    println!();
                    println!("  Normalization will be applied. The effective metric for KNN:");
                    let effective = if source_metric.eq_ignore_ascii_case("L2") {
                        println!("    Source metric L2 on normalized vectors → Cosine-equivalent ranking.");
                        println!("    Using Cosine as the KNN metric.");
                        "Cosine".to_string()
                    } else if source_metric.eq_ignore_ascii_case("DotProduct") {
                        println!("    Source metric DotProduct on normalized vectors → Cosine-equivalent.");
                        println!("    Using Cosine as the KNN metric.");
                        "Cosine".to_string()
                    } else {
                        println!("    Keeping {} as the KNN metric.", source_metric);
                        source_metric.clone()
                    };
                    (true, effective)
                } else {
                    println!("  Keeping vectors unnormalized. KNN will use {} as specified.", source_metric);
                    (false, source_metric)
                }
            }
        } else {
            eprintln!("skipped (format not supported for sampling).");
            (false, source_metric)
        }
    } else {
        (false, source_metric)
    };

    // ── Remaining configuration ─────────────────────────────────────
    println!();
    println!("--- Configuration ---");

    let neighbors = if has_queries {
        let n_str = prompt_with_default("Number of neighbors (k)", "100");
        n_str.parse().unwrap_or(100)
    } else {
        100
    };

    let seed_str = prompt_with_default("Random seed", "42");
    let seed = seed_str.parse().unwrap_or(42);

    // ── Optional features ────────────────────────────────────────────
    let no_dedup = if base_vectors.is_some() {
        !confirm("Include deduplication stage?", true)
    } else {
        true
    };

    let no_zero_check = if base_vectors.is_some() {
        // Default to yes when dedup is enabled (most common case)
        let default_yes = !no_dedup;
        !confirm("Check for and exclude zero vectors?", default_yes)
    } else {
        true
    };

    if !no_dedup && !no_zero_check {
        println!("  A clean ordinal index will be produced excluding both duplicates and zeros.");
    } else if !no_dedup {
        println!("  A clean ordinal index will be produced excluding duplicates.");
    } else if !no_zero_check {
        println!("  A clean ordinal index will be produced excluding zeros.");
    }

    let no_filtered = if metadata.is_some() && has_queries {
        !confirm("Include filtered KNN (predicated search)?", true)
    } else {
        true
    };

    // ── Sized profiles ────────────────────────────────────────────────
    let sized_profiles = if let Some(ref bv) = base_vectors {
        if !has_queries {
            None
        } else {
            let vec_count = probe_vector_count(bv);
            let effective_max = vec_count
                .map(|n| n.saturating_sub(query_count as u64))
                .unwrap_or(0);
            let max_label = format_count_label(effective_max);

            println!();
            println!("--- Sized profiles ---");
            if effective_max > 0 {
                println!("  Base vector count after query extraction: ~{}", max_label);
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
            if effective_max < 2_000_000 {
                println!("  Dataset is small enough that sized profiles may not be needed.");
                if !confirm("Generate sized profiles anyway?", false) {
                    None
                } else {
                    let default_spec = format!("mul:1m..{}/2", max_label);
                    let spec = prompt_optional(&format!("  Sized profile spec [{}]", default_spec));
                    Some(spec.unwrap_or(default_spec))
                }
            } else if confirm("Generate sized profiles?", true) {
                let default_spec = format!("mul:1m..{}/2, 0m..{}/10m", max_label, max_label);
                println!();
                println!("  Default: {}", default_spec);
                println!("  Or enter a custom spec, or press Enter for the default.");
                let spec = prompt_optional("  Sized profile spec (Enter for default)");
                Some(spec.unwrap_or(default_spec))
            } else {
                None
            }
        }
    } else {
        None
    };

    // ── Cache compression ────────────────────────────────────────────
    println!();
    println!("--- Cache compression ---");
    println!("  Eligible cache files (sorted runs, KNN partitions, predicate segments)");
    println!("  can be gzip-compressed to save disk space. These files are only ever");
    println!("  read sequentially, so compression has no impact on access patterns.");
    let compress_cache = confirm("Compress eligible cache files?", true);

    // ── Summary ──────────────────────────────────────────────────────
    println!();
    println!("=== Summary ===");
    println!("  Name:          {}", name);
    if let Some(ref d) = description { println!("  Description:   {}", d); }
    println!("  Output:        {}", output.display());
    if let Some(ref bv) = base_vectors { println!("  Base vectors:  {}", bv.display()); }
    if let Some(ref qv) = query_vectors { println!("  Query vectors: {}", qv.display()); }
    if self_search { println!("  Query mode:    self-search ({})", query_count); }
    if let Some(ref m) = metadata { println!("  Metadata:      {}", m.display()); }
    if let Some(ref gt) = ground_truth { println!("  Ground truth:  {}", gt.display()); }
    println!("  Metric:        {}", metric);
    println!("  Neighbors:     {}", neighbors);
    println!("  Seed:          {}", seed);
    println!("  Normalize:     {}", if normalize { "yes" } else { "no" });
    println!("  Dedup:         {}", if no_dedup { "no" } else { "yes" });
    println!("  Zero check:    {}", if no_zero_check { "no" } else { "yes" });
    if metadata.is_some() && has_queries {
        println!("  Filtered KNN:  {}", if no_filtered { "no" } else { "yes" });
    }
    if let Some(ref sp) = sized_profiles {
        println!("  Sized profiles: {}", sp);
    }
    println!("  Compress cache: {}", if compress_cache { "yes" } else { "no" });
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
fn prompt_source_location(source: &Path, output_dir: &Path, label: &str) -> PathBuf {
    let filename = source.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    // If already in .cache/ or starts with _, no action needed
    let source_str = source.to_string_lossy();
    if source_str.contains(".cache/") || source_str.contains("/.cache/") || filename.starts_with('_') {
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
    if path.is_dir() { return None; }
    let meta = crate::formats::reader::probe_source(path, format).ok()?;
    meta.record_count
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
struct DetectedRoles {
    base_vectors: Option<PathBuf>,
    query_vectors: Option<PathBuf>,
    neighbor_indices: Option<PathBuf>,
    neighbor_distances: Option<PathBuf>,
    metadata: Option<PathBuf>,
    metadata_predicates: Option<PathBuf>,
    metadata_results: Option<PathBuf>,
    filtered_neighbor_indices: Option<PathBuf>,
    filtered_neighbor_distances: Option<PathBuf>,
    unassigned: Vec<PathBuf>,
}

impl DetectedRoles {
    /// Returns `true` if at least one role was detected.
    fn any_detected(&self) -> bool {
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
fn detect_roles(candidates: &[(PathBuf, String, u64)]) -> DetectedRoles {
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
        let stem = path.file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
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

        // Skip hidden files and known non-data files
        if name_str.starts_with('.') || name_str == "dataset.yaml"
            || name_str == "variables.yaml"
            || name_str.ends_with(".json") || name_str.ends_with(".yaml")
            || name_str.ends_with(".yml")
        {
            continue;
        }

        if let Some(format) = VecFormat::detect_from_path(&path) {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            results.push((path, format.name().to_string(), size));
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
                results.push((path, format.name().to_string(), size));
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
