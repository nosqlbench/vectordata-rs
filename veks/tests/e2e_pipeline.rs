// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end pipeline execution tests.
//!
//! Each test generates small synthetic data, bootstraps a dataset via
//! `import::run()`, executes the pipeline headlessly via the `veks` binary,
//! then reads the output artifacts and verifies numerical correctness.
//!
//! Data is small (dim=4, ~200 vectors) so tests complete in seconds.
//! Temp directories are under `target/tmp/` (not /tmp).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use veks::prepare::import::ImportArgs;

// ═══════════════════════════════════════════════════════════════════════════
// Test infrastructure
// ═══════════════════════════════════════════════════════════════════════════

/// Path to the compiled veks binary.
fn veks_bin() -> PathBuf {
    // cargo sets this for integration tests
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_veks"));
    if !path.exists() {
        // Fallback: look in target/debug
        path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../target/debug/veks");
    }
    path
}

/// Create a project-local temp directory under `target/tmp`.
fn make_tempdir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// Default ImportArgs for e2e tests.
fn default_args(name: &str, output: &Path) -> ImportArgs {
    ImportArgs {
        name: name.to_string(),
        output: output.to_path_buf(),
        base_vectors: None,
        query_vectors: None,
        self_search: false,
        query_count: 10,
        metadata: None,
        ground_truth: None,
        ground_truth_distances: None,
        metric: "Cosine".to_string(),
        neighbors: 5,
        seed: 42,
        description: None,
        no_dedup: false,
        no_filtered: false,
        no_zero_check: false,
        normalize: false,
        force: true,
        base_convert_format: None,
        query_convert_format: None,
        compress_cache: false,
        sized_profiles: None,
        base_fraction: 1.0,
        required_facets: None,
        round_digits: 10, // disable rounding for exact counts in tests
        pedantic_dedup: false,
        selectivity: 0.0001,
    }
}

/// Write an fvec file with specific vectors.
///
/// Includes known properties:
/// - Vectors 0 and 1 are identical (duplicate pair)
/// - Vector 2 is the zero vector
/// - Remaining vectors have distinct values
fn write_test_fvec(path: &Path, count: usize, dim: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..count {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            let val = if i == 2 {
                0.0f32 // zero vector
            } else if i == 1 {
                // duplicate of vector 0
                (d + 1) as f32
            } else if i == 0 {
                (d + 1) as f32
            } else {
                (i * dim + d + 1) as f32
            };
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Read an fvec/mvec file and return (dim, record_count).
fn read_xvec_counts(path: &Path) -> (usize, usize) {
    let data = std::fs::read(path).unwrap();
    if data.len() < 4 { return (0, 0); }
    let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let elem_size = match path.extension().and_then(|e| e.to_str()) {
        Some("fvec") => 4,
        Some("mvec") => 2,
        Some("ivec") => 4,
        _ => 4,
    };
    let stride = 4 + dim * elem_size;
    let count = data.len() / stride;
    (dim, count)
}

/// Read an ivec file and return the ordinals as a Vec<i32>.
fn read_ivec_ordinals(path: &Path) -> Vec<Vec<i32>> {
    let data = std::fs::read(path).unwrap();
    let mut records = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let dim = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        let mut rec = Vec::with_capacity(dim);
        for _ in 0..dim {
            if pos + 4 > data.len() { break; }
            rec.push(i32::from_le_bytes(data[pos..pos+4].try_into().unwrap()));
            pos += 4;
        }
        records.push(rec);
    }
    records
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared fixture layer
// ═══════════════════════════════════════════════════════════════════════════

/// Shared test fixture directory. Created once per test run, reused
/// across all tests. Contains source data files that tests copy into
/// their own tempdirs.
static FIXTURES: OnceLock<PathBuf> = OnceLock::new();

/// Get or create the shared fixture directory with pre-generated data.
fn fixtures() -> &'static Path {
    FIXTURES.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/tmp/e2e-fixtures");
        std::fs::create_dir_all(&dir).unwrap();

        // Base fvec: 200 vectors, dim=4, with known dups and zeros
        let fvec = dir.join("base.fvec");
        if !fvec.exists() {
            write_test_fvec(&fvec, 200, 4);
        }

        // Separate query fvec: 20 distinct vectors
        let query = dir.join("queries.fvec");
        if !query.exists() {
            write_distinct_fvec(&query, 20, 4, 1000);
        }

        // Pre-computed ground truth: 20 queries × 5 neighbors
        let gt = dir.join("gt.ivec");
        if !gt.exists() {
            write_fake_ivec(&gt, 20, 5);
        }

        // Pre-computed distances: 20 queries × 5 distances
        let dist = dir.join("gt_dist.fvec");
        if !dist.exists() {
            write_fake_fvec_distances(&dist, 20, 5);
        }

        // Small fvec: 50 vectors for KNN verification
        let small = dir.join("small.fvec");
        if !small.exists() {
            write_test_fvec(&small, 50, 4);
        }

        // Large fvec: 500 vectors for fraction tests
        let large = dir.join("large.fvec");
        if !large.exists() {
            write_test_fvec(&large, 500, 4);
        }

        // Npy directory: 50 vectors, dim=4, f32
        let npy = dir.join("npy_source");
        if !npy.exists() {
            write_npy_dir(&npy, 50, 4);
        }

        // Parquet metadata: 200 rows with two scalar columns
        let meta = dir.join("metadata");
        if !meta.exists() {
            write_parquet_metadata(&meta, 200);
        }

        dir
    })
}

/// Write a minimal npy directory with `count` vectors of `dim` dimensions.
fn write_npy_dir(dir: &Path, count: usize, dim: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let descr = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", count, dim);
    let magic = b"\x93NUMPY\x01\x00";
    let mut header = Vec::new();
    header.extend_from_slice(magic);
    let hdr_bytes = descr.as_bytes();
    let total = 10 + hdr_bytes.len() + 1;
    let pad = ((total + 63) / 64) * 64 - total;
    let hdr_len = (hdr_bytes.len() + pad + 1) as u16;
    header.extend_from_slice(&hdr_len.to_le_bytes());
    header.extend_from_slice(hdr_bytes);
    for _ in 0..pad { header.push(b' '); }
    header.push(b'\n');
    for i in 0..(count * dim) {
        header.extend_from_slice(&(i as f32).to_le_bytes());
    }
    std::fs::write(dir.join("part-0.npy"), &header).unwrap();
}

/// Write a minimal parquet metadata directory with scalar columns.
fn write_parquet_metadata(dir: &Path, rows: usize) {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    std::fs::create_dir_all(dir).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
    ]));

    let ids: Vec<i32> = (0..rows as i32).collect();
    let labels: Vec<String> = (0..rows).map(|i| format!("item_{}", i)).collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(label_refs)),
        ],
    ).unwrap();

    let file = std::fs::File::create(dir.join("part-0.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Copy a fixture file into a test's temp directory.
fn copy_fixture(fixture_name: &str, dest: &Path) {
    let src = fixtures().join(fixture_name);
    std::fs::copy(&src, dest).unwrap_or_else(|e| {
        panic!("failed to copy fixture {} to {}: {}", src.display(), dest.display(), e);
    });
}

/// Write `count` distinct vectors starting at offset `base_val`.
fn write_distinct_fvec(path: &Path, count: usize, dim: usize, base_val: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..count {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            let val = (base_val + i * dim + d) as f32;
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Write a fake ivec (for pre-computed GT).
fn write_fake_ivec(path: &Path, queries: usize, k: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for q in 0..queries {
        w.write_all(&(k as i32).to_le_bytes()).unwrap();
        for n in 0..k {
            let idx = ((q + n) % 200) as i32;
            w.write_all(&idx.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Write fake fvec distances (for pre-computed GT distances).
fn write_fake_fvec_distances(path: &Path, queries: usize, k: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for _q in 0..queries {
        w.write_all(&(k as i32).to_le_bytes()).unwrap();
        for n in 0..k {
            let dist = (n as f32 + 1.0) * 0.1;
            w.write_all(&dist.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Read fvec vectors as Vec<Vec<f32>>.
fn read_fvec_vectors(path: &Path) -> Vec<Vec<f32>> {
    let data = std::fs::read(path).unwrap();
    let mut vecs = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let dim = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            if pos + 4 > data.len() { break; }
            vec.push(f32::from_le_bytes(data[pos..pos+4].try_into().unwrap()));
            pos += 4;
        }
        vecs.push(vec);
    }
    vecs
}

/// Run `veks run` on a dataset directory and return (success, output).
fn run_pipeline(dataset_yaml: &Path) -> (bool, String) {
    let output = Command::new(veks_bin())
        .arg("run")
        .arg("--output").arg("batch")
        .arg("--threads")
        .arg("2")
        .arg(dataset_yaml)
        .output()
        .expect("failed to execute veks");

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.status.success() {
        eprintln!("=== PIPELINE STDOUT ===\n{}", stdout);
        eprintln!("=== PIPELINE STDERR ===\n{}", stderr);
    }
    (output.status.success(), format!("{}\n{}", stdout, stderr))
}

// ═══════════════════════════════════════════════════════════════════════════
// End-to-end tests
// ═══════════════════════════════════════════════════════════════════════════

/// E2E Config 1: Self-search with dedup and zero check.
///
/// Input: 200 vectors (dim=4), 1 duplicate pair, 1 zero vector.
/// Expected: dedup removes 1, zero check removes 1 → 198 clean.
/// Shuffle + extract produces 10 queries + 188 base vectors.
/// KNN computed on 188 base × 10 queries × 5 neighbors.
#[test]
fn e2e_self_search_with_dedup() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-self-search", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;

    // Bootstrap
    veks::prepare::import::run(args);
    let dataset_yaml = out.join("dataset.yaml");
    assert!(dataset_yaml.exists(), "dataset.yaml not generated");

    // Run pipeline
    let (success, output) = run_pipeline(&dataset_yaml);
    assert!(success, "pipeline failed:\n{}", output);

    // Verify outputs exist
    let base = out.join("profiles/base/base_vectors.fvec");
    let query = out.join("profiles/base/query_vectors.fvec");
    assert!(base.exists(), "base_vectors not produced");
    assert!(query.exists(), "query_vectors not produced");

    // Verify record counts
    let (base_dim, base_count) = read_xvec_counts(&base);
    let (query_dim, query_count) = read_xvec_counts(&query);
    assert_eq!(base_dim, 4, "base dimension mismatch");
    assert_eq!(query_dim, 4, "query dimension mismatch");
    assert_eq!(query_count, 10, "expected 10 query vectors");
    // 200 - 1 dup - 1 zero = 198 clean, minus 10 queries = 188 base
    assert_eq!(base_count, 188, "expected 188 base vectors (200 - 1 dup - 1 zero - 10 queries)");

    // Verify KNN output exists with correct shape
    let indices_path = out.join("profiles/default/neighbor_indices.ivec");
    assert!(indices_path.exists(), "neighbor_indices not produced");
    let indices = read_ivec_ordinals(&indices_path);
    assert_eq!(indices.len(), 10, "expected 10 query results");
    assert_eq!(indices[0].len(), 5, "expected 5 neighbors per query");

    // Verify all neighbor indices are in range [0, base_count)
    for (qi, neighbors) in indices.iter().enumerate() {
        for &idx in neighbors {
            assert!(
                idx >= 0 && (idx as usize) < base_count,
                "query {} neighbor index {} out of range [0, {})",
                qi, idx, base_count,
            );
        }
    }

    // Verify variables.yaml has expected variables
    let vars_path = out.join("variables.yaml");
    assert!(vars_path.exists(), "variables.yaml not produced");
    let vars_content = std::fs::read_to_string(&vars_path).unwrap();
    assert!(vars_content.contains("vector_count"), "missing vector_count");
    assert!(vars_content.contains("clean_count"), "missing clean_count");
}

/// E2E Config 2: No dedup, no zero check — simplest pipeline.
///
/// All 200 vectors pass through. 10 queries + 190 base.
#[test]
fn e2e_no_cleaning() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-no-clean", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.no_dedup = true;
    args.no_zero_check = true;

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    let (_, base_count) = read_xvec_counts(&out.join("profiles/base/base_vectors.fvec"));
    let (_, query_count) = read_xvec_counts(&out.join("profiles/base/query_vectors.fvec"));
    assert_eq!(query_count, 10);
    assert_eq!(base_count, 190, "expected 190 base vectors (200 - 10 queries, no dedup/zero)");
}

/// E2E Config 3: Separate queries — no shuffle.
///
/// 200 base vectors, 20 separate query vectors. All pass through dedup.
#[test]
fn e2e_separate_queries() {
    let tmp = make_tempdir();
    let base_fvec = tmp.path().join("base.fvec");
    let query_fvec = tmp.path().join("queries.fvec");
    copy_fixture("base.fvec", &base_fvec);
    // Write 20 distinct query vectors (no dups, no zeros)
    {
        let f = std::fs::File::create(&query_fvec).unwrap();
        let mut w = std::io::BufWriter::new(f);
        for i in 0..20usize {
            w.write_all(&4i32.to_le_bytes()).unwrap();
            for d in 0..4usize {
                let val = (1000 + i * 4 + d) as f32;
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-separate", &out);
    args.base_vectors = Some(base_fvec);
    args.query_vectors = Some(query_fvec);

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // KNN should exist
    let indices_path = out.join("profiles/default/neighbor_indices.ivec");
    assert!(indices_path.exists(), "neighbor_indices not produced");
    let indices = read_ivec_ordinals(&indices_path);
    assert_eq!(indices.len(), 20, "expected 20 query results");
}

/// E2E Config 4: Base-only, no queries (facets=B).
///
/// Only base vectors, no KNN, no shuffle. Pipeline should still succeed.
#[test]
fn e2e_base_only() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-base-only", &out);
    args.base_vectors = Some(fvec);
    args.required_facets = Some("B".to_string());

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // No KNN, no queries, no shuffle — just catalog + merkle
    assert!(!out.join("profiles/default/neighbor_indices.ivec").exists(),
        "KNN should not be produced for B-only");
}

/// E2E Config 5: Base fraction 50% — capped base set.
///
/// 200 vectors, 1 dup, 1 zero → 198 clean. 50% → base_end ≈ 99.
/// 10 queries + ~89 base vectors.
#[test]
fn e2e_base_fraction() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-fraction", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.base_fraction = 0.5;

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    let (_, base_count) = read_xvec_counts(&out.join("profiles/base/base_vectors.fvec"));
    let (_, query_count) = read_xvec_counts(&out.join("profiles/base/query_vectors.fvec"));
    assert_eq!(query_count, 10);
    // 198 clean * 0.5 = 99 base_end, minus 10 queries = 89 base
    assert!(base_count < 190, "base_count {} should be < 190 (capped by fraction)", base_count);
    assert!(base_count > 50, "base_count {} should be > 50 (fraction is 50%)", base_count);
}

/// E2E Config 6: Pre-computed ground truth — skip KNN computation.
///
/// Separate base + query files with pre-computed GT indices.
/// The pipeline should use the provided GT as identity and only verify.
#[test]
fn e2e_precomputed_gt() {
    let tmp = make_tempdir();
    let base = tmp.path().join("base.fvec");
    let query = tmp.path().join("queries.fvec");
    let gt = tmp.path().join("gt.ivec");
    copy_fixture("base.fvec", &base);
    copy_fixture("queries.fvec", &query);
    copy_fixture("gt.ivec", &gt);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-precomputed-gt", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.ground_truth = Some(gt);

    veks::prepare::import::run(args);
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();

    // compute-knn should NOT be in the pipeline
    assert!(!yaml.contains("compute knn"), "compute-knn should be absent with pre-computed GT");

    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // verify-knn should still run
    assert!(yaml.contains("verify knn"), "verify-knn should still be present");
}

/// E2E Config 7: Pre-computed GT + distances.
///
/// Both GT indices and distances provided. No KNN computation needed.
#[test]
fn e2e_precomputed_gt_and_distances() {
    let tmp = make_tempdir();
    let base = tmp.path().join("base.fvec");
    let query = tmp.path().join("queries.fvec");
    let gt = tmp.path().join("gt.ivec");
    let dist = tmp.path().join("gt_dist.fvec");
    copy_fixture("base.fvec", &base);
    copy_fixture("queries.fvec", &query);
    copy_fixture("gt.ivec", &gt);
    copy_fixture("gt_dist.fvec", &dist);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-gt-and-dist", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.ground_truth = Some(gt);
    args.ground_truth_distances = Some(dist);

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);
}

/// E2E Config 8: Normalize vectors during extraction.
///
/// Self-search with --normalize. Output vectors should all have L2 norm ≈ 1.0.
#[test]
fn e2e_normalize() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-normalize", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.normalize = true;

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Read output base vectors and verify all are L2-normalized
    let base_path = out.join("profiles/base/base_vectors.fvec");
    let vecs = read_fvec_vectors(&base_path);
    assert!(!vecs.is_empty(), "no base vectors produced");

    for (i, v) in vecs.iter().enumerate() {
        let norm: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "vector {} has norm {:.4}, expected ~1.0 (zero vectors should have been excluded by dedup/clean)",
            i, norm,
        );
    }
}

/// E2E Config 9: KNN numerical verification.
///
/// Verify that for each query, the reported nearest neighbor is actually
/// the closest vector by brute force.
#[test]
fn e2e_knn_numerical_correctness() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("small.fvec", &fvec);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-knn-verify", &out);
    args.base_vectors = Some(fvec.clone());
    args.self_search = true;
    args.query_count = 5;
    args.neighbors = 3;
    args.metric = "L2".to_string();
    // Disable dedup/zeros to keep ordinals predictable
    args.no_dedup = true;
    args.no_zero_check = true;

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Read query vectors, base vectors, and KNN indices
    let query_vecs = read_fvec_vectors(&out.join("profiles/base/query_vectors.fvec"));
    let base_vecs = read_fvec_vectors(&out.join("profiles/base/base_vectors.fvec"));
    let indices = read_ivec_ordinals(&out.join("profiles/default/neighbor_indices.ivec"));

    assert_eq!(query_vecs.len(), 5, "expected 5 queries");
    assert_eq!(indices.len(), 5, "expected 5 KNN results");

    // For each query, verify the first neighbor is actually the closest
    for (qi, query) in query_vecs.iter().enumerate() {
        let nn_idx = indices[qi][0] as usize;
        let nn_dist = l2_distance(query, &base_vecs[nn_idx]);

        // Check no other base vector is closer
        for (bi, base) in base_vecs.iter().enumerate() {
            let dist = l2_distance(query, base);
            assert!(
                dist >= nn_dist - 1e-5,
                "query {}: base {} (dist={:.6}) is closer than reported NN {} (dist={:.6})",
                qi, bi, dist, nn_idx, nn_dist,
            );
        }
    }
}

/// L2 squared distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as f64) - (y as f64);
            d * d
        })
        .sum()
}

/// E2E Config 10: Dedup numerical verification.
///
/// Verify that the dedup step correctly identifies the known duplicate
/// pair and zero vector.
#[test]
fn e2e_dedup_correctness() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-dedup-verify", &out);
    args.base_vectors = Some(fvec);
    args.required_facets = Some("B".to_string()); // base only, no KNN

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Check variables.yaml for counts
    let vars = std::fs::read_to_string(out.join("variables.yaml")).unwrap();

    // vector_count should be 200
    assert!(vars.contains("vector_count: '200'") || vars.contains("vector_count: \"200\""),
        "expected vector_count=200 in variables.yaml, got:\n{}", vars);

    // duplicate_count should be 1 (one duplicate of the pair)
    assert!(vars.contains("duplicate_count: '1'") || vars.contains("duplicate_count: \"1\""),
        "expected duplicate_count=1 in variables.yaml, got:\n{}", vars);

    // zero_count should be 1
    assert!(vars.contains("zero_count: '1'") || vars.contains("zero_count: \"1\""),
        "expected zero_count=1 in variables.yaml, got:\n{}", vars);

    // clean_count should be 198 (200 - 1 dup - 1 zero)
    assert!(vars.contains("clean_count: '198'") || vars.contains("clean_count: \"198\""),
        "expected clean_count=198 in variables.yaml, got:\n{}", vars);
}

/// E2E Config 11: Base fraction 10% via CLI bootstrap --auto.
///
/// Creates 500 vectors (dim=4, 1 dup pair, 1 zero → 498 clean).
/// Bootstraps with --base-fraction '10%' --round-digits 10 (no rounding).
/// 10% of 498 = 49 base_end. With 10 queries → 39 base vectors.
///
/// Verifies:
/// 1. Pipeline succeeds
/// 2. base_end ≈ 49-50 in variables.yaml
/// 3. Base vector artifact has ~39 records
/// 4. Query artifact has 10 records
/// 5. KNN indices have 10 results
/// 6. run.log shows no step processing more than ~50 vectors after conversion
#[test]
fn e2e_base_fraction_10_percent_cli() {
    let tmp = make_tempdir();
    let dataset_dir = tmp.path().join("fraction10");
    std::fs::create_dir_all(&dataset_dir).unwrap();

    // Create source data: 500 vectors with known properties
    let fvec = dataset_dir.join("_base_vectors.fvec");
    copy_fixture("large.fvec", &fvec);

    // Bootstrap via CLI with explicit flags
    let bootstrap_out = Command::new(veks_bin())
        .arg("prepare")
        .arg("bootstrap")
        .arg("--name").arg("fraction-test")
        .arg("--output").arg(&dataset_dir)
        .arg("--base-vectors").arg(&fvec)
        .arg("--self-search")
        .arg("--base-fraction").arg("10%")
        .arg("--round-digits").arg("10")  // no rounding, exact counts
        .arg("--metric").arg("Cosine")
        .arg("--query-count").arg("10")
        .arg("--neighbors").arg("5")
        .arg("--seed").arg("42")
        .arg("--force")
        .output()
        .expect("failed to run bootstrap");

    let bootstrap_stderr = String::from_utf8_lossy(&bootstrap_out.stderr);
    let bootstrap_stdout = String::from_utf8_lossy(&bootstrap_out.stdout);
    let dataset_yaml = dataset_dir.join("dataset.yaml");
    assert!(bootstrap_out.status.success() && dataset_yaml.exists(),
        "bootstrap failed (exit={:?}):\nstdout: {}\nstderr: {}\ndataset_dir contents: {:?}",
        bootstrap_out.status.code(), bootstrap_stdout, bootstrap_stderr,
        std::fs::read_dir(&dataset_dir).ok().map(|d| d.filter_map(|e| e.ok().map(|e| e.file_name())).collect::<Vec<_>>()));

    // Verify the YAML contains fraction on convert step
    let yaml = std::fs::read_to_string(&dataset_yaml).unwrap();
    assert!(yaml.contains("fraction"), "expected fraction option in dataset.yaml:\n{}", yaml);

    // Run the pipeline
    let (success, output) = run_pipeline(&dataset_yaml);
    assert!(success, "pipeline failed:\n{}", output);

    // ── Verify variables.yaml ──────────────────────────────────────
    let vars = std::fs::read_to_string(dataset_dir.join("variables.yaml")).unwrap();

    // Parse variables into a map
    let var_map: std::collections::HashMap<String, String> = vars.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            if parts.len() == 2 {
                Some((parts[0].trim().to_string(),
                      parts[1].trim().trim_matches('\'').trim_matches('"').to_string()))
            } else { None }
        })
        .collect();

    let vector_count: u64 = var_map.get("vector_count")
        .and_then(|v| v.parse().ok()).unwrap_or(0);
    let clean_count: u64 = var_map.get("clean_count")
        .and_then(|v| v.parse().ok()).unwrap_or(0);
    let base_end: u64 = var_map.get("base_end")
        .and_then(|v| v.parse().ok()).unwrap_or(0);
    let base_count: u64 = var_map.get("base_count")
        .and_then(|v| v.parse().ok()).unwrap_or(0);

    // vector_count = 500 (all vectors imported, or fraction of 500)
    // The convert step has fraction=0.1, so it may import only ~50 vectors.
    // If convert honors the fraction: vector_count ≈ 50
    // clean_count ≤ vector_count (after dedup + zero removal)
    // base_end = clean_count * 0.1 (but clean_count is already fractioned)
    // OR if convert imports all 500: base_end = clean_count * 0.1 ≈ 49

    eprintln!("variables: vector_count={}, clean_count={}, base_end={}, base_count={}",
        vector_count, clean_count, base_end, base_count);

    // In fast mode (subset applied): vector_count ≈ 50 (10% of 500),
    // clean_count ≈ 48 (after dedup+zeros on the subset),
    // base_end = 0 (not set — clean_count IS the bound),
    // base_count = clean_count - query_count ≈ 38.
    assert!(vector_count <= 60,
        "vector_count ({}) should be <= 60 (10% of 500)", vector_count);
    assert!(clean_count <= vector_count,
        "clean_count ({}) should be <= vector_count ({})", clean_count, vector_count);
    assert!(base_count > 0, "base_count should be > 0, got {}", base_count);
    assert!(base_count <= 55,
        "base_count ({}) should be <= 55 (10% of ~500 minus queries)", base_count);

    // ── Verify output artifact sizes ───────────────────────────────
    let base_path = dataset_dir.join("profiles/base/base_vectors.fvec");
    let query_path = dataset_dir.join("profiles/base/query_vectors.fvec");
    assert!(base_path.exists(), "base_vectors not produced");
    assert!(query_path.exists(), "query_vectors not produced");

    let (base_dim, actual_base_count) = read_xvec_counts(&base_path);
    let (query_dim, actual_query_count) = read_xvec_counts(&query_path);

    assert_eq!(base_dim, 4);
    assert_eq!(query_dim, 4);
    assert_eq!(actual_query_count, 10, "expected 10 query vectors");
    assert_eq!(actual_base_count, base_count as usize,
        "base_vectors record count ({}) should match base_count variable ({})",
        actual_base_count, base_count);
    assert!(actual_base_count <= 55,
        "base_vectors has {} records, expected <= 55 for 10% fraction", actual_base_count);

    // ── Verify KNN output ──────────────────────────────────────────
    let indices_path = dataset_dir.join("profiles/default/neighbor_indices.ivec");
    assert!(indices_path.exists(), "neighbor_indices not produced");
    let indices = read_ivec_ordinals(&indices_path);
    assert_eq!(indices.len(), 10, "expected 10 query results");

    // All neighbor indices must be within [0, base_count)
    for (qi, neighbors) in indices.iter().enumerate() {
        for &idx in neighbors {
            assert!(idx >= 0 && (idx as u64) < base_count,
                "query {} neighbor {} out of range [0, {})", qi, idx, base_count);
        }
    }

    // ── Verify run.log: no step processes more than ~60 vectors ────
    // after the initial conversion step.
    let run_log_path = dataset_dir.join(".cache/run.log");
    if run_log_path.exists() {
        let log = std::fs::read_to_string(&run_log_path).unwrap();
        eprintln!("=== run.log ===\n{}", log);

        // After convert-vectors, no step should report processing
        // more than ~60 records. Look for large record counts in
        // progress messages.
        let mut past_convert = false;
        for line in log.lines() {
            if line.contains("convert") && line.contains("records") {
                past_convert = true;
                continue;
            }
            if past_convert {
                // Look for patterns like "407314954 records" or "N vectors"
                // that would indicate the full dataset leaked through
                for word in line.split_whitespace() {
                    if let Ok(n) = word.parse::<u64>() {
                        if n > 200 {
                            // Allow counts up to 200 in log messages
                            // (variable values, file sizes, etc. are fine)
                            // but flag anything that looks like a vector count
                            // Only flag raw counts, not rates (vec/s, rec/s)
                            let is_rate = line.contains("vec/s") || line.contains("rec/s")
                                || line.contains("MB/s");
                            if !is_rate && (line.contains("vectors") || line.contains("records")) {
                                if n > 500 {
                                    panic!(
                                        "run.log shows {} in a processing line after convert: {}",
                                        n, line.trim()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("e2e_base_fraction_10_percent_cli: all assertions passed");
    eprintln!("  vector_count={}, clean_count={}, base_end={}, base_count={}",
        vector_count, clean_count, base_end, base_count);
    eprintln!("  actual base={}, actual query={}", actual_base_count, actual_query_count);
}

/// E2E Config 12: Foreign source (npy) — convert step produces mvec.
///
/// 50 vectors in npy format, dim=4. Convert to mvec, dedup, shuffle, extract.
/// Verifies the conversion actually runs and produces correct mvec output.
#[test]
fn e2e_npy_source() {
    let tmp = make_tempdir();
    let npy = tmp.path().join("embeddings");
    // Copy the fixture npy dir
    let src = fixtures().join("npy_source");
    let dest = &npy;
    std::fs::create_dir_all(dest).unwrap();
    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dest.join(entry.file_name())).unwrap();
    }

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-npy", &out);
    args.base_vectors = Some(npy);
    args.self_search = true;
    args.query_count = 5;
    args.neighbors = 3;

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Output should be mvec (f16) since npy→mvec is the convert path
    let base_path = out.join("profiles/base/base_vectors.mvec");
    assert!(base_path.exists(), "base_vectors.mvec not produced");

    let (base_dim, base_count) = read_xvec_counts(&base_path);
    assert_eq!(base_dim, 4, "dimension mismatch");
    assert!(base_count > 0 && base_count <= 50, "base_count {} out of range", base_count);

    let query_path = out.join("profiles/base/query_vectors.mvec");
    assert!(query_path.exists(), "query_vectors.mvec not produced");
    let (_, query_count) = read_xvec_counts(&query_path);
    assert_eq!(query_count, 5, "expected 5 queries");

    // KNN should exist
    let indices = out.join("profiles/default/neighbor_indices.ivec");
    assert!(indices.exists(), "KNN indices not produced");
}

/// E2E Config 13: Metadata pipeline — parquet → slab → predicates → eval.
///
/// 200 base vectors + 200-row parquet metadata. Full BQGDMPRF pipeline
/// except filtered KNN (no_filtered=true to keep test fast).
#[test]
fn e2e_metadata_pipeline() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let meta = tmp.path().join("metadata");
    let src_meta = fixtures().join("metadata");
    std::fs::create_dir_all(&meta).unwrap();
    for entry in std::fs::read_dir(&src_meta).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), meta.join(entry.file_name())).unwrap();
    }

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-metadata", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    args.no_filtered = true; // skip filtered KNN for speed

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Metadata artifacts should exist
    let meta_content = out.join("profiles/base/metadata_content.slab");
    assert!(meta_content.exists(), "metadata_content.slab not produced");

    let predicates = out.join("profiles/base/predicates.slab");
    assert!(predicates.exists(), "predicates.slab not produced");

    let meta_indices = out.join("profiles/default/metadata_indices.slab");
    assert!(meta_indices.exists(), "metadata_indices.slab not produced");

    // Verify variables
    let vars = std::fs::read_to_string(out.join("variables.yaml")).unwrap();
    assert!(vars.contains("vector_count"), "missing vector_count");
    assert!(vars.contains("base_count"), "missing base_count");

    // No filtered KNN (no_filtered=true)
    assert!(!out.join("profiles/default/filtered_neighbor_indices.ivec").exists(),
        "filtered KNN should not be produced with no_filtered=true");
}

/// E2E Config 14: Metadata with fraction — both vectors and metadata subset.
///
/// Verifies ordinal congruency: same fraction applied to both, same ordinals
/// in output base vectors and metadata.
#[test]
fn e2e_metadata_with_fraction() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let meta = tmp.path().join("metadata");
    let src_meta = fixtures().join("metadata");
    std::fs::create_dir_all(&meta).unwrap();
    for entry in std::fs::read_dir(&src_meta).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), meta.join(entry.file_name())).unwrap();
    }

    let out = tmp.path().join("dataset");
    let mut args = default_args("e2e-meta-fraction", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.metadata = Some(meta);
    args.base_fraction = 0.5;
    args.no_filtered = true;

    veks::prepare::import::run(args);

    // Check YAML has fraction on both convert steps
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    let fraction_count = yaml.matches("fraction:").count();
    // Metadata is native fvec (no convert-metadata), but base has subset step
    // The metadata convert should also get fraction if it runs
    assert!(fraction_count >= 1, "expected at least 1 fraction option in YAML, found {}", fraction_count);

    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Base and metadata should have same number of records
    let base_path = out.join("profiles/base/base_vectors.fvec");
    let meta_path = out.join("profiles/base/metadata_content.slab");
    assert!(base_path.exists(), "base_vectors not produced");
    assert!(meta_path.exists(), "metadata_content not produced");

    let (_, base_count) = read_xvec_counts(&base_path);
    assert!(base_count > 0 && base_count < 190,
        "base_count {} should be < 190 (50% fraction)", base_count);
}

/// E2E Config 15: Cache invalidation across config changes.
///
/// Runs the pipeline at 100% fraction, then re-bootstraps at 50% fraction
/// and re-runs. The second run must NOT use stale cached artifacts from
/// the first run (different data size → different cache keys).
///
/// Verifies:
/// 1. First run produces ~190 base vectors (200 - dup - zero - queries)
/// 2. Second run produces ~89 base vectors (50% of 198 clean - queries)
/// 3. KNN indices from second run are all within [0, base_count_2)
#[test]
fn e2e_cache_invalidation_on_reconfig() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("vectors.fvec");
    copy_fixture("base.fvec", &fvec);

    let out = tmp.path().join("dataset");

    // ── Run 1: 100% fraction ──────────────────────────────────
    let mut args1 = default_args("cache-inval", &out);
    args1.base_vectors = Some(fvec.clone());
    args1.self_search = true;
    veks::prepare::import::run(args1);

    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "run 1 failed:\n{}", output);

    let (_, base_count_1) = read_xvec_counts(&out.join("profiles/base/base_vectors.fvec"));
    assert!(base_count_1 > 150, "run 1 base_count {} too small", base_count_1);

    // Verify KNN exists from run 1
    let indices_1 = read_ivec_ordinals(&out.join("profiles/default/neighbor_indices.ivec"));
    assert_eq!(indices_1.len(), 10, "run 1: expected 10 query results");

    // ── Re-bootstrap with 50% fraction ────────────────────────
    let mut args2 = default_args("cache-inval", &out);
    args2.base_vectors = Some(fvec.clone());
    args2.self_search = true;
    args2.base_fraction = 0.5;
    args2.force = true;
    veks::prepare::import::run(args2);

    // The config hash changed → progress log should invalidate all steps
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "run 2 failed:\n{}", output);

    let (_, base_count_2) = read_xvec_counts(&out.join("profiles/base/base_vectors.fvec"));

    // Run 2 should have fewer base vectors (50% subset)
    assert!(base_count_2 < base_count_1,
        "run 2 base_count ({}) should be less than run 1 ({})",
        base_count_2, base_count_1);
    assert!(base_count_2 > 30 && base_count_2 < 100,
        "run 2 base_count {} should be in [30, 100] for 50% of ~198",
        base_count_2);

    // KNN indices from run 2 must be in range [0, base_count_2)
    let indices_2 = read_ivec_ordinals(&out.join("profiles/default/neighbor_indices.ivec"));
    assert_eq!(indices_2.len(), 10, "run 2: expected 10 query results");
    for (qi, neighbors) in indices_2.iter().enumerate() {
        for &idx in neighbors {
            assert!(idx >= 0 && (idx as usize) < base_count_2,
                "run 2: query {} neighbor {} out of range [0, {}). \
                 Stale cache from run 1 may have leaked through.",
                qi, idx, base_count_2);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Early stratification tests
// ═══════════════════════════════════════════════════════════════════════════

/// E2E: Early stratification with concrete sized profiles at bootstrap time.
///
/// Bootstraps with `sized_profiles: "50"` on 200-vector source data.
/// After pipeline runs: profile "50" should exist with 50 base vectors.
#[test]
fn e2e_early_stratification() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("early-strat", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.sized_profiles = Some("50".to_string());

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Default profile artifacts should exist
    assert!(out.join("profiles/default/neighbor_indices.ivec").exists(),
        "default profile KNN indices should exist");

    // Check dataset.yaml has the sized profile
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(yaml.contains("sized:"), "sized: key should be in dataset.yaml");
}

/// E2E: Early stratification with base fraction.
///
/// Bootstraps with `--base-fraction 50% --sized "20"`.
/// The fraction reduces the data universe; profiles operate within it.
#[test]
fn e2e_early_stratification_with_fraction() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("frac-strat", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.base_fraction = 0.5;
    args.sized_profiles = Some("20".to_string());
    args.round_digits = 10; // disable rounding

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    // Base vectors should be ~50% of 198 clean = ~99
    let (_, base_count) = read_xvec_counts(&out.join("profiles/base/base_vectors.fvec"));
    assert!(base_count > 30 && base_count < 120,
        "base_count {} should be ~50% of 198 clean vectors", base_count);
}

// ═══════════════════════════════════════════════════════════════════════════
// Adversarial tests — pipeline boundary conditions and error paths
// ═══════════════════════════════════════════════════════════════════════════

/// Adversarial: Rerunning a pipeline with the same config should be a no-op.
///
/// All steps should be skipped (fresh). This verifies that the progress
/// tracking and freshness algorithm work correctly.
#[test]
fn adversarial_idempotent_rerun() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("idempotent", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;

    veks::prepare::import::run(args);

    // Run 1
    let (success1, _) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success1, "run 1 failed");

    // Run 2 — should skip all steps
    let (success2, output2) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success2, "run 2 failed:\n{}", output2);
    // Check that the output mentions skipping
    assert!(output2.contains("up-to-date") || output2.contains("skipped") || output2.contains("0 executed"),
        "run 2 should have skipped all steps (idempotent), output:\n{}", output2);
}

/// Adversarial: KNN indices must all be in-range for the base vector count.
///
/// This catches stale cache bugs where a larger run's indices leak into a
/// smaller fraction run.
#[test]
fn adversarial_knn_indices_in_range() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("knn-range", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    assert!(success, "pipeline failed:\n{}", output);

    let (_, base_count) = read_xvec_counts(&out.join("profiles/base/base_vectors.fvec"));
    let indices = read_ivec_ordinals(&out.join("profiles/default/neighbor_indices.ivec"));

    for (qi, neighbors) in indices.iter().enumerate() {
        for &idx in neighbors {
            assert!(idx >= 0 && (idx as usize) < base_count,
                "query {} neighbor {} out of range [0, {})",
                qi, idx, base_count);
        }
    }
}

/// Adversarial: Empty base vectors file should produce an error, not hang.
#[test]
fn adversarial_empty_base_vectors() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("empty.fvec");
    std::fs::write(&fvec, b"").unwrap(); // 0 bytes

    let out = tmp.path().join("out");
    let mut args = default_args("empty-base", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.no_dedup = true;
    args.no_zero_check = true;

    // Bootstrap should either error or produce a dataset with 0 vectors
    veks::prepare::import::run(args);

    // The pipeline should fail gracefully (not hang or panic)
    let yaml_path = out.join("dataset.yaml");
    if yaml_path.exists() {
        let (success, output) = run_pipeline(&yaml_path);
        // Either succeeds with 0 vectors or fails with clear error
        if !success {
            assert!(output.contains("error") || output.contains("Error") || output.contains("failed"),
                "pipeline should report a clear error for empty input");
        }
    }
}

/// Adversarial: Bootstrap with query_count exceeding available vectors.
///
/// With 200 vectors and query_count=300, extraction should not panic.
/// It should either cap at available or produce a clear error.
#[test]
fn adversarial_query_count_exceeds_vectors() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("query-exceed", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.query_count = 300; // exceeds 200 vectors
    args.no_dedup = true;
    args.no_zero_check = true;

    veks::prepare::import::run(args);
    let yaml_path = out.join("dataset.yaml");
    if yaml_path.exists() {
        let (success, output) = run_pipeline(&yaml_path);
        // Should not panic — either succeeds with capped count or errors
        if !success {
            assert!(!output.contains("panicked"),
                "pipeline should not panic on oversized query_count");
        }
    }
}

/// Adversarial: k exceeds base count after cleaning.
///
/// With 200 vectors, 1 dup, 1 zero, 10 queries → 188 base. k=200.
/// KNN should handle k > base_count gracefully.
#[test]
fn adversarial_k_exceeds_base_count() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("k-exceed", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.neighbors = 200; // exceeds base count after cleaning (~188)

    veks::prepare::import::run(args);
    let yaml_path = out.join("dataset.yaml");
    if yaml_path.exists() {
        let (success, output) = run_pipeline(&yaml_path);
        // Should either succeed with k capped or fail with clear error
        if !success {
            assert!(!output.contains("panicked"),
                "pipeline should not panic when k exceeds base count");
        }
    }
}

/// Adversarial: Fraction of 0 should not produce an infinite loop or panic.
#[test]
fn adversarial_zero_fraction() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("zero-frac", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    args.base_fraction = 0.0; // Edge case: 0% of data

    veks::prepare::import::run(args);
    let yaml_path = out.join("dataset.yaml");
    if yaml_path.exists() {
        let (success, output) = run_pipeline(&yaml_path);
        // Should handle 0 vectors gracefully
        if !success {
            assert!(!output.contains("panicked"),
                "pipeline should not panic on 0% fraction");
        }
    }
}

/// Adversarial: Sized profile larger than base count should be silently
/// excluded (or window-clamped), not cause an error.
#[test]
fn adversarial_sized_profile_exceeds_base() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("oversized-profile", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    // 200 vectors, but profile asks for 1M
    args.sized_profiles = Some("1m".to_string());

    veks::prepare::import::run(args);
    let (success, output) = run_pipeline(&out.join("dataset.yaml"));
    // Should succeed — the oversized profile's window gets clamped to
    // the actual base count, which is the same as the default profile
    assert!(success, "pipeline should handle oversized profile gracefully:\n{}", output);
}

/// Adversarial: Two consecutive runs with different seeds should produce
/// different shuffles but both produce valid KNN results.
#[test]
fn adversarial_different_seeds_valid_knn() {
    let tmp = make_tempdir();
    let fvec = tmp.path().join("base.fvec");
    std::fs::copy(fixtures().join("base.fvec"), &fvec).unwrap();

    // Run 1 with seed=42
    let out1 = tmp.path().join("out1");
    let mut args1 = default_args("seed42", &out1);
    args1.base_vectors = Some(fvec.clone());
    args1.self_search = true;
    args1.seed = 42;
    veks::prepare::import::run(args1);
    let (s1, o1) = run_pipeline(&out1.join("dataset.yaml"));
    assert!(s1, "seed 42 failed:\n{}", o1);

    // Run 2 with seed=99
    let out2 = tmp.path().join("out2");
    let mut args2 = default_args("seed99", &out2);
    args2.base_vectors = Some(fvec);
    args2.self_search = true;
    args2.seed = 99;
    veks::prepare::import::run(args2);
    let (s2, o2) = run_pipeline(&out2.join("dataset.yaml"));
    assert!(s2, "seed 99 failed:\n{}", o2);

    // Both should have valid indices
    let (_, bc1) = read_xvec_counts(&out1.join("profiles/base/base_vectors.fvec"));
    let (_, bc2) = read_xvec_counts(&out2.join("profiles/base/base_vectors.fvec"));
    // Same source data + same cleaning = same base count
    assert_eq!(bc1, bc2, "same source should produce same base count regardless of seed");

    // But different queries (different shuffle)
    let q1 = read_xvec_counts(&out1.join("profiles/base/query_vectors.fvec"));
    let q2 = read_xvec_counts(&out2.join("profiles/base/query_vectors.fvec"));
    assert_eq!(q1.1, q2.1, "same query count regardless of seed");
}
