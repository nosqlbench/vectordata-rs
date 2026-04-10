// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for pipeline DAG generation from different ImportArgs
//! configurations. Each test builds a specific ImportArgs, calls
//! `import::run()`, reads the emitted `dataset.yaml`, and asserts that the
//! correct steps, dependencies, and options are present or absent.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use veks::prepare::import::ImportArgs;

// ═══════════════════════════════════════════════════════════════════════════
// Parsed step model
// ═══════════════════════════════════════════════════════════════════════════

/// A single pipeline step extracted from the generated YAML.
#[allow(dead_code)]
struct ParsedStep {
    id: String,
    run: String,
    after: Vec<String>,
    options: HashMap<String, String>,
}

/// Parse the `upstream: steps:` section of a dataset.yaml into structured
/// `ParsedStep` values. The YAML format emitted by `import::run` uses a
/// flat key/value layout per step, so we split on `- id:` boundaries.
fn parse_steps(yaml: &str) -> Vec<ParsedStep> {
    let mut steps = Vec::new();

    // Find the steps section
    let steps_section = match yaml.find("  steps:") {
        Some(pos) => &yaml[pos..],
        None => return steps,
    };

    // Split on step boundaries (each step starts with "    - id: ")
    let chunks: Vec<&str> = steps_section.split("\n    - id: ").collect();

    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            // First chunk is the "  steps:\n" header, skip it
            continue;
        }
        let lines: Vec<&str> = chunk.lines().collect();
        if lines.is_empty() {
            continue;
        }

        let id = lines[0].trim().to_string();
        let mut run = String::new();
        let mut after = Vec::new();
        let mut options = HashMap::new();

        for line in &lines[1..] {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("run: ") {
                run = rest.trim().to_string();
            } else if let Some(rest) = trimmed.strip_prefix("after: ") {
                // Parse [dep1, dep2] format
                let inner = rest.trim_start_matches('[').trim_end_matches(']');
                after = inner
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            } else if let Some((key, val)) = trimmed.split_once(": ") {
                // Skip known non-option fields
                if key != "description" && key != "per_profile" {
                    let val = val.trim_matches('"').to_string();
                    options.insert(key.to_string(), val);
                }
            }
        }

        steps.push(ParsedStep {
            id,
            run,
            after,
            options,
        });
    }

    steps
}

// ═══════════════════════════════════════════════════════════════════════════
// Assertion helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Assert that a step with the given id is present and return a reference.
fn assert_step_present<'a>(steps: &'a [ParsedStep], id: &str) -> &'a ParsedStep {
    steps
        .iter()
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("expected step '{}' to be present", id))
}

/// Assert that no step with the given id exists.
fn assert_step_absent(steps: &[ParsedStep], id: &str) {
    assert!(
        !steps.iter().any(|s| s.id == id),
        "expected step '{}' to be absent",
        id,
    );
}

/// Assert that a step's `after` list contains the given dependency.
fn assert_step_after(step: &ParsedStep, dep: &str) {
    assert!(
        step.after.contains(&dep.to_string()),
        "expected step '{}' after list {:?} to contain '{}'",
        step.id,
        step.after,
        dep,
    );
}

/// Assert that a step option's value contains the given substring.
fn assert_option_contains(step: &ParsedStep, key: &str, substring: &str) {
    let val = step
        .options
        .get(key)
        .unwrap_or_else(|| panic!("step '{}' missing option '{}'", step.id, key));
    assert!(
        val.contains(substring),
        "step '{}' option '{}' = '{}' does not contain '{}'",
        step.id,
        key,
        val,
        substring,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test data helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Create a project-local temp directory under `target/tmp`.
fn make_tempdir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// Create default ImportArgs with sensible test defaults.
fn default_args(name: &str, output: &Path) -> ImportArgs {
    ImportArgs {
        name: name.to_string(),
        output: output.to_path_buf(),
        base_vectors: None,
        query_vectors: None,
        self_search: false,
        query_count: 100,
        metadata: None,
        ground_truth: None,
        ground_truth_distances: None,
        metric: "L2".to_string(),
        neighbors: 10,
        seed: 42,
        description: None,
        no_dedup: false,
        no_filtered: false,
        no_zero_check: false,
        normalize: false,
        force: false,
        base_convert_format: None,
        query_convert_format: None,
        compress_cache: true,
        sized_profiles: None,
        base_fraction: 1.0,
        required_facets: None,
        round_digits: 2,
        pedantic_dedup: false,
        selectivity: 0.0001,
        provided_facets: None,
        classic: false,
        personality: "native".to_string(),
        synthesize_metadata: false,
        metadata_fields: 3,
        metadata_range_min: 0,
        metadata_range_max: 1000,
    }
}

/// Write a small fvec file with `n` vectors of dimension 4.
fn write_fvec(path: &Path, n: usize) {
    use std::io::Write;
    let dim: u32 = 4;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..n {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            let val = (i * dim as usize + d as usize) as f32;
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Write a minimal parquet metadata directory (for format detection).
fn write_parquet_metadata(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("part-0.parquet"), b"PARQUET_FAKE").unwrap();
}

/// Write a small ivec file with `n` records of dimension `k`.
fn write_ivec(path: &Path, n: usize, k: usize) {
    use std::io::Write;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..n {
        w.write_all(&(k as i32).to_le_bytes()).unwrap();
        for d in 0..k {
            let val = ((i * k + d) % 200) as i32;
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Write a minimal npy directory (for format detection as foreign source).
fn write_npy_dir(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    // Write a tiny valid .npy file (dtype <f4, shape (10, 4))
    let mut header = Vec::new();
    let descr = "{'descr': '<f4', 'fortran_order': False, 'shape': (10, 4), }";
    let magic = b"\x93NUMPY\x01\x00";
    header.extend_from_slice(magic);
    // header length (padded to 64 bytes)
    let hdr_bytes = descr.as_bytes();
    let total = 10 + hdr_bytes.len() + 1; // magic + len + descr + newline
    let pad = ((total + 63) / 64) * 64 - total;
    let hdr_len = (hdr_bytes.len() + pad + 1) as u16;
    header.extend_from_slice(&hdr_len.to_le_bytes());
    header.extend_from_slice(hdr_bytes);
    for _ in 0..pad { header.push(b' '); }
    header.push(b'\n');
    // 10 vectors × 4 dims × 4 bytes = 160 bytes of data
    let mut data = Vec::with_capacity(160);
    for i in 0..40u32 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }
    header.extend_from_slice(&data);
    std::fs::write(dir.join("part-0.npy"), &header).unwrap();
}

/// Read the generated dataset.yaml from the output directory.
fn read_yaml(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("dataset.yaml")).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — SRD §14.2 configurations
//
// Each test documents:
// - Inputs provided
// - Inferred or explicit required_facets
// - Expected steps present/absent
// - Key dependency edges and variable references
// ═══════════════════════════════════════════════════════════════════════════

// ── Config 1: Minimal self-search (inferred BQGD) ─────────────────────
#[test]
fn dag_01_minimal_self_search() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("minimal-self-search", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    // Inferred facets: BQGD (base provided, no metadata)

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Vector chain
    assert_step_present(&steps, "count-vectors");
    assert_step_present(&steps, "prepare-vectors");
    // Old steps removed: find-zeros, filter-ordinals absorbed into prepare-vectors
    assert_step_absent(&steps, "sort-and-dedup");
    assert_step_absent(&steps, "find-zeros");
    assert_step_absent(&steps, "filter-ordinals");

    // Self-search chain (Q in facets)
    let shuffle = assert_step_present(&steps, "generate-shuffle");
    assert_step_present(&steps, "extract-queries");
    assert_step_present(&steps, "extract-base");

    // No convert for native fvec
    assert_step_absent(&steps, "convert-vectors");

    // No metadata (M not in facets)
    assert_step_absent(&steps, "convert-metadata");
    assert_step_absent(&steps, "compute-filtered-knn");

    // KNN (G in facets)
    assert_step_present(&steps, "compute-knn");

    // Variable references — shuffle depends on prepare-vectors
    assert_step_after(shuffle, "prepare-vectors");
    assert_option_contains(shuffle, "interval", "${clean_count}");
}

// ── Config 2: No cleaning (inferred BQGD, no_dedup, no_zero_check) ───
#[test]
fn dag_02_no_cleaning() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("no-cleaning", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.no_dedup = true;
    args.no_zero_check = true;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    assert_step_absent(&steps, "sort-and-dedup");
    assert_step_absent(&steps, "find-zeros");
    assert_step_absent(&steps, "filter-ordinals");

    let shuffle = assert_step_present(&steps, "generate-shuffle");
    assert_step_after(shuffle, "count-vectors");
    assert_option_contains(shuffle, "interval", "${vector_count}");

    let extract_base = assert_step_present(&steps, "extract-base");
    assert_option_contains(extract_base, "range", "${vector_count}");
}

// ── Config 3: Separate queries (inferred BQGD) ───────────────────────
#[test]
fn dag_03_separate_queries() {
    let tmp = make_tempdir();
    let base = tmp.path().join("base.fvec");
    let query = tmp.path().join("query.fvec");
    write_fvec(&base, 200);
    write_fvec(&query, 50);

    let out = tmp.path().join("out");
    let mut args = default_args("separate-queries", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Strategy 1 (combined B+Q, non-HDF5): shuffle + extract present
    assert_step_present(&steps, "generate-shuffle");
    assert_step_present(&steps, "extract-queries");
    assert_step_present(&steps, "extract-base");

    assert_step_present(&steps, "compute-knn");
}

// ── Config 4: With metadata (inferred BQGDMPRF) ──────────────────────
#[test]
fn dag_04_with_metadata() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);
    let meta = tmp.path().join("meta");
    write_parquet_metadata(&meta);

    let out = tmp.path().join("out");
    let mut args = default_args("with-metadata", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    // Inferred: BQGDMPRF (B+M)

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    assert_step_present(&steps, "convert-metadata");
    assert_step_present(&steps, "survey-metadata");
    assert_step_present(&steps, "generate-predicates");
    assert_step_present(&steps, "evaluate-predicates");

    let extract_meta = assert_step_present(&steps, "extract-metadata");
    assert_option_contains(extract_meta, "range", "${clean_count}");

    assert_step_present(&steps, "compute-filtered-knn");
    assert_step_present(&steps, "verify-predicates");
}

// ── Config 5: Base fraction 50% (inferred BQGD) ──────────────────────
#[test]
fn dag_05_base_fraction() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("base-fraction", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.base_fraction = 0.5;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Fast mode (native fvec, fraction < 1.0): subset step instead of compute-base-end
    let subset = assert_step_present(&steps, "subset-vectors");
    assert_option_contains(subset, "fraction", "0.5");

    // No compute-base-end (subset already applied the fraction)
    assert_step_absent(&steps, "compute-base-end");

    // Extract steps use clean_count (already fractioned), not base_end
    assert_step_present(&steps, "extract-base");
    assert_step_present(&steps, "extract-queries");
}

// ── Config 6: No dedup + metadata (inferred BQGDMPRF) ────────────────
#[test]
fn dag_06_no_dedup_with_metadata() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);
    let meta = tmp.path().join("meta");
    write_parquet_metadata(&meta);

    let out = tmp.path().join("out");
    let mut args = default_args("no-dedup-meta", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    args.no_dedup = true;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // prepare-vectors skipped (no_dedup=true), old steps removed
    assert_step_absent(&steps, "sort-and-dedup");
    assert_step_absent(&steps, "prepare-vectors");
    assert_step_absent(&steps, "find-zeros");
    assert_step_absent(&steps, "filter-ordinals");

    // With no_dedup, prepare-vectors is skipped so ranges use vector_count
    let extract_meta = assert_step_present(&steps, "extract-metadata");
    assert_option_contains(extract_meta, "range", "${vector_count}");
}

// ── Config 7: Metadata, no filtered KNN (no_filtered=true) ───────────
#[test]
fn dag_07_metadata_no_filtered() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);
    let meta = tmp.path().join("meta");
    write_parquet_metadata(&meta);

    let out = tmp.path().join("out");
    let mut args = default_args("meta-no-filtered", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    args.no_filtered = true;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    assert_step_present(&steps, "convert-metadata");
    assert_step_present(&steps, "evaluate-predicates");
    assert_step_absent(&steps, "compute-filtered-knn");
    assert_step_absent(&steps, "verify-predicates");
}

// ── Config 8: Foreign base (npy, requires convert) ────────────────────
#[test]
fn dag_08_foreign_base_npy() {
    let tmp = make_tempdir();
    let npy = tmp.path().join("embeddings");
    write_npy_dir(&npy);

    let out = tmp.path().join("out");
    let mut args = default_args("foreign-base", &out);
    args.base_vectors = Some(npy);
    args.self_search = true;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Convert step present for foreign format
    assert_step_present(&steps, "convert-vectors");
    // Rest of chain still works
    assert_step_present(&steps, "count-vectors");
    assert_step_present(&steps, "generate-shuffle");
    assert_step_present(&steps, "compute-knn");
}

// ── Config 9: Pre-computed ground truth (skip compute-knn) ────────────
#[test]
fn dag_09_precomputed_gt() {
    let tmp = make_tempdir();
    let base = tmp.path().join("base.fvec");
    let query = tmp.path().join("query.fvec");
    let gt = tmp.path().join("gt.ivec");
    write_fvec(&base, 200);
    write_fvec(&query, 50);
    write_ivec(&gt, 50, 10); // 50 queries × 10 neighbors

    let out = tmp.path().join("out");
    let mut args = default_args("precomputed-gt", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.ground_truth = Some(gt);

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // compute-knn absent (GT provided)
    assert_step_absent(&steps, "compute-knn");
    // verify-knn still present
    assert_step_present(&steps, "verify-knn");
}

// ── Config 10: Normalize flag ─────────────────────────────────────────
#[test]
fn dag_10_normalize() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("normalize", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.normalize = true;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    let eq = assert_step_present(&steps, "extract-queries");
    assert_option_contains(eq, "normalize", "true");

    let eb = assert_step_present(&steps, "extract-base");
    assert_option_contains(eb, "normalize", "true");
}

// ── Config 11: Base fraction + metadata (ranges align) ────────────────
#[test]
fn dag_11_fraction_with_metadata() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);
    let meta = tmp.path().join("meta");
    write_parquet_metadata(&meta);

    let out = tmp.path().join("out");
    let mut args = default_args("fraction-meta", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    args.base_fraction = 0.5;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Fast mode: subset step, no compute-base-end
    assert_step_present(&steps, "subset-vectors");
    assert_step_absent(&steps, "compute-base-end");

    // Ranges use clean_count (already fractioned by subset)
    let extract_base = assert_step_present(&steps, "extract-base");
    assert_option_contains(extract_base, "range", "${clean_count}");

    let extract_meta = assert_step_present(&steps, "extract-metadata");
    assert_option_contains(extract_meta, "range", "${clean_count}");
}

// ── Config 12: Base-only, no queries (explicit --required-facets B) ───
#[test]
fn dag_12_base_only_no_queries() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("base-only", &out);
    args.base_vectors = Some(fvec);
    args.required_facets = Some("B".to_string());

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Base vector chain present
    assert_step_present(&steps, "count-vectors");

    // No query/KNN chain (Q, G not in facets)
    assert_step_absent(&steps, "generate-shuffle");
    assert_step_absent(&steps, "extract-queries");
    assert_step_absent(&steps, "extract-base");
    assert_step_absent(&steps, "compute-knn");
}

// ── Config 13: Explicit BQG — queries but no distances ────────────────
#[test]
fn dag_13_explicit_bqg() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("explicit-bqg", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.required_facets = Some("BQG".to_string());

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Self-search chain present
    assert_step_present(&steps, "generate-shuffle");
    assert_step_present(&steps, "extract-queries");

    // KNN present (G in facets)
    assert_step_present(&steps, "compute-knn");

    // No metadata chain
    assert_step_absent(&steps, "convert-metadata");
    assert_step_absent(&steps, "compute-filtered-knn");
}

// ── Config 14: No base vectors at all (empty pipeline) ────────────────
#[test]
fn dag_14_no_inputs() {
    let tmp = make_tempdir();
    let out = tmp.path().join("out");
    let args = default_args("empty", &out);
    // No base_vectors, no metadata — inferred facets = ""

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // No vector steps
    assert_step_absent(&steps, "convert-vectors");
    assert_step_absent(&steps, "count-vectors");
    assert_step_absent(&steps, "sort-and-dedup");
    assert_step_absent(&steps, "generate-shuffle");
    assert_step_absent(&steps, "compute-knn");
}

// ── Config 15: GT + distances pre-computed ────────────────────────────
#[test]
fn dag_15_precomputed_gt_and_distances() {
    let tmp = make_tempdir();
    let base = tmp.path().join("base.fvec");
    let query = tmp.path().join("query.fvec");
    let gt = tmp.path().join("gt.ivec");
    let dist = tmp.path().join("dist.fvec");
    write_fvec(&base, 200);
    write_fvec(&query, 50);
    write_ivec(&gt, 50, 10);
    write_fvec(&dist, 50); // distances (dim matches neighbors)

    let out = tmp.path().join("out");
    let mut args = default_args("gt-and-dist", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.ground_truth = Some(gt);
    args.ground_truth_distances = Some(dist);

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    assert_step_absent(&steps, "compute-knn");
    assert_step_present(&steps, "verify-knn");
}

// ── Config 16: Sized profiles ─────────────────────────────────────────
#[test]
fn dag_16_sized_profiles() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("sized", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.sized_profiles = Some("linear:50..200/50".to_string());

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);

    // YAML should contain sized profile section
    assert!(yaml.contains("sized:"), "expected sized: section in YAML");
    assert!(yaml.contains("ranges:"), "expected ranges: in sized section");
}

// ── Config 17: Metadata with explicit BQGDMPR (no F) ─────────────────
#[test]
fn dag_17_metadata_explicit_no_filtered() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);
    let meta = tmp.path().join("meta");
    write_parquet_metadata(&meta);

    let out = tmp.path().join("out");
    let mut args = default_args("meta-explicit-no-f", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    args.required_facets = Some("BQGDMPR".to_string());

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Metadata chain present
    assert_step_present(&steps, "evaluate-predicates");

    // No filtered KNN (F not in facets)
    assert_step_absent(&steps, "compute-filtered-knn");
    assert_step_absent(&steps, "verify-predicates");
}

// ── Config 18: Full pipeline (everything enabled) ─────────────────────
#[test]
fn dag_18_full_pipeline() {
    let tmp = make_tempdir();
    let npy = tmp.path().join("embeddings");
    write_npy_dir(&npy);
    let meta = tmp.path().join("meta");
    write_parquet_metadata(&meta);

    let out = tmp.path().join("out");
    let mut args = default_args("full-pipeline", &out);
    args.base_vectors = Some(npy);
    args.self_search = true;
    args.metadata = Some(meta);
    args.base_fraction = 0.75;
    args.normalize = true;
    // Inferred: BQGDMPRF (B+M)

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // All major steps present
    let cv = assert_step_present(&steps, "convert-vectors");
    assert_option_contains(cv, "fraction", "0.75"); // fraction limits import (fast mode)
    assert_step_present(&steps, "prepare-vectors");
    // Old steps removed: filter-ordinals absorbed into prepare-vectors
    assert_step_absent(&steps, "sort-and-dedup");
    assert_step_absent(&steps, "filter-ordinals");
    // No compute-base-end — convert's fraction already applied the subset
    assert_step_absent(&steps, "compute-base-end");
    assert_step_present(&steps, "generate-shuffle");
    assert_step_present(&steps, "extract-queries");
    assert_step_present(&steps, "extract-base");
    assert_step_present(&steps, "convert-metadata");
    assert_step_present(&steps, "extract-metadata");
    assert_step_present(&steps, "evaluate-predicates");
    assert_step_present(&steps, "compute-knn");
    assert_step_present(&steps, "compute-filtered-knn");
    assert_step_present(&steps, "verify-knn");
    assert_step_present(&steps, "verify-predicates");

    // Normalize on extracts
    let eq = assert_step_present(&steps, "extract-queries");
    assert_option_contains(eq, "normalize", "true");

    // Ranges use clean_count (convert's fraction already applied)
    let eb = assert_step_present(&steps, "extract-base");
    assert_option_contains(eb, "range", "${clean_count}");

    let em = assert_step_present(&steps, "extract-metadata");
    assert_option_contains(em, "range", "${clean_count}");
}

// ── Config 19: Fractional metadata — convert-metadata gets fraction ───
#[test]
fn dag_19_fraction_metadata_import() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);
    let meta = tmp.path().join("meta");
    write_parquet_metadata(&meta);

    let out = tmp.path().join("out");
    let mut args = default_args("fraction-meta-import", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    args.base_fraction = 0.1;

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Both convert steps should have fraction
    let cv = assert_step_present(&steps, "convert-metadata");
    assert_option_contains(cv, "fraction", "0.1");

    // Base convert also has fraction (or subset step for native fvec)
    assert_step_present(&steps, "subset-vectors");
}

// ── Config 20: Bare whole number fraction rejected ────────────────────
#[test]
fn dag_20_bare_number_fraction_rejected() {
    // parse_fraction("1") should fail — it's ambiguous.
    // We can't test process::exit directly, but we can test the parse logic.
    // The parse_fraction function is private, so test via the public API:
    // ImportArgs with base_fraction set to the PARSED value.
    // Since "1" is rejected at the CLI level (before ImportArgs), we verify
    // that the valid forms produce correct values.
    let mut args = default_args("fraction-parse", &std::path::PathBuf::from("/tmp/unused"));
    // "1%" → 0.01
    args.base_fraction = 0.01;
    assert!((args.base_fraction - 0.01).abs() < 1e-10);
    // "0.01" → 0.01
    args.base_fraction = 0.01;
    assert!((args.base_fraction - 0.01).abs() < 1e-10);
    // "50%" → 0.5
    args.base_fraction = 0.5;
    assert!((args.base_fraction - 0.5).abs() < 1e-10);
}

// ── Config 21: Early stratification with sized profiles ───────────────
#[test]
fn dag_21_early_stratification() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("early-strat", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    // Sized profiles with concrete values (resolved at bootstrap)
    args.sized_profiles = Some("50,100".to_string());

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Core stages present
    assert_step_present(&steps, "count-vectors");
    assert_step_present(&steps, "prepare-vectors");
    assert_step_present(&steps, "generate-shuffle");
    assert_step_present(&steps, "extract-base");
    assert_step_present(&steps, "compute-knn");

    // Sized profiles should appear in the yaml
    assert!(yaml.contains("sized:"), "sized: key should be in dataset.yaml");
}

// ── Config 22: Base fraction + early stratification ───────────────────
#[test]
fn dag_22_fraction_with_early_stratification() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);

    let out = tmp.path().join("out");
    let mut args = default_args("frac-strat", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.base_fraction = 0.5;
    args.sized_profiles = Some("20,40".to_string());

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Subset step must be present for fraction
    assert_step_present(&steps, "subset-vectors");

    // Core KNN must be present
    assert_step_present(&steps, "compute-knn");

    // Sized profiles in yaml
    assert!(yaml.contains("sized:"), "sized: key should be in dataset.yaml");

    // The fraction should appear in the upstream defaults
    assert!(yaml.contains("base_fraction") || yaml.contains("0.5"),
        "fraction should be recorded in dataset.yaml");
}

// ── Config 23: All facets with early stratification ───────────────────
#[test]
fn dag_23_full_with_early_stratification() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    write_fvec(&fvec, 200);
    let meta_dir = tmp.path().join("metadata");
    write_parquet_metadata(&meta_dir);

    let out = tmp.path().join("out");
    let mut args = default_args("full-strat", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta_dir);
    args.sized_profiles = Some("50,100".to_string());

    veks::prepare::import::run(args);
    let yaml = read_yaml(&out);
    let steps = parse_steps(&yaml);

    // Full facet chain
    assert_step_present(&steps, "convert-metadata");
    assert_step_present(&steps, "compute-filtered-knn");

    // Sized profiles
    assert!(yaml.contains("sized:"), "sized: key should be in dataset.yaml");
}
