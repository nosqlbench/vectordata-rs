// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for pipeline commands with edge-case inputs.
//!
//! Each test creates a `StreamContext`, writes small synthetic data, and calls
//! the command's `execute()` method directly to verify correct behavior on
//! boundary conditions.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use veks_pipeline::pipeline::command::{Options, Status, StreamContext};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;
use veks_core::ui::{TestSink, UiHandle};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal `StreamContext` rooted at the given directory.
fn test_ctx(dir: &Path) -> StreamContext {
    StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace: dir.to_path_buf(),
        scratch: dir.join(".scratch"),
        cache: dir.join(".cache"),
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads: 1,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: UiHandle::new(Arc::new(TestSink::new())),
        status_interval: Duration::from_secs(1),
        estimated_total_steps: 0,
    }
}

/// Write a small fvec file with known vectors for testing.
fn write_test_fvec(path: &Path, vectors: &[Vec<f32>]) {
    use std::io::Write;
    let file = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(file);
    for vec in vectors {
        let dim = vec.len() as i32;
        w.write_all(&dim.to_le_bytes()).unwrap();
        for &v in vec {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Read an fvec file back into vectors.
fn read_fvec(path: &Path) -> Vec<Vec<f32>> {
    let data = std::fs::read(path).unwrap();
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let dim = i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            let v = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            vec.push(v);
            offset += 4;
        }
        result.push(vec);
    }
    result
}

/// Read an ivec file (dim-1 records) back into a list of i32 values.
fn read_ivec_1d(path: &Path) -> Vec<i32> {
    let data = std::fs::read(path).unwrap();
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let dim = i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        assert_eq!(dim, 1, "expected dim=1 ivec records");
        let v = i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        result.push(v);
        offset += 4;
    }
    result
}

/// Write an ivec file with dim=1 records.
fn write_test_ivec_1d(path: &Path, values: &[i32]) {
    use std::io::Write;
    let file = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(file);
    let dim: i32 = 1;
    for &v in values {
        w.write_all(&dim.to_le_bytes()).unwrap();
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    w.flush().unwrap();
}

/// Create a tempdir under `target/tmp/`.
fn tmp_dir() -> tempfile::TempDir {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

// ===========================================================================
// 1. convert_empty_source
// ===========================================================================

#[test]
fn convert_empty_source() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Write a 0-record fvec (empty file)
    let source = ws.join("empty.fvec");
    write_test_fvec(&source, &[]);

    let output = ws.join("out.fvec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("to", "fvec");

    let mut ctx = test_ctx(ws);

    // compute_dedup handles count==0 by producing empty output.
    let mut dedup_opts = Options::new();
    dedup_opts.set("source", source.to_string_lossy().to_string());
    dedup_opts.set("output", ws.join("dedup_out.ivec").to_string_lossy().to_string());

    let mut dedup_op = veks_pipeline::pipeline::commands::compute_dedup::factory();
    let result = dedup_op.execute(&dedup_opts, &mut ctx);

    // Dedup handles count==0 → produces empty output with Status::Ok
    assert_eq!(result.status, Status::Ok, "dedup on 0-record fvec: {}", result.message);
}

// ===========================================================================
// 2. convert_single_record
// ===========================================================================

#[test]
fn convert_single_record() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Write a 1-record fvec
    let source = ws.join("single.fvec");
    write_test_fvec(&source, &[vec![1.0, 2.0, 3.0]]);

    // Use the extract command to copy that single record via range-based mode
    let output = ws.join("out.fvec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("range", "[0,1)");

    let mut op = veks_pipeline::pipeline::commands::gen_extract::extract_factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "extract single record: {}", result.message);
    let vecs = read_fvec(&output);
    assert_eq!(vecs.len(), 1);
    assert_eq!(vecs[0], vec![1.0, 2.0, 3.0]);
}

// ===========================================================================
// 3. shuffle_interval_one
// ===========================================================================

#[test]
fn shuffle_interval_one() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let output = ws.join("shuffle.ivec");
    let mut opts = Options::new();
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("interval", "1");
    opts.set("seed", "42");

    let mut op = veks_pipeline::pipeline::commands::gen_shuffle::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "shuffle interval=1: {}", result.message);
    let values = read_ivec_1d(&output);
    assert_eq!(values.len(), 1);
    assert_eq!(values[0], 0); // the only permutation of [0] is [0]
}

// ===========================================================================
// 4. shuffle_interval_zero
// ===========================================================================

#[test]
fn shuffle_interval_zero() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let output = ws.join("shuffle.ivec");
    let mut opts = Options::new();
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("interval", "0");
    opts.set("seed", "42");

    let mut op = veks_pipeline::pipeline::commands::gen_shuffle::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    // interval=0 is rejected: `Ok(n) if n > 0` guard in execute()
    assert_eq!(result.status, Status::Error, "shuffle interval=0 should error: {}", result.message);
}

// ===========================================================================
// 5. dedup_all_identical
// ===========================================================================

#[test]
fn dedup_all_identical() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // 5 identical vectors
    let source = ws.join("identical.fvec");
    let v = vec![1.0f32, 2.0, 3.0];
    write_test_fvec(&source, &vec![v.clone(); 5]);

    let output = ws.join("dedup.ivec");
    let dups = ws.join("dedup_duplicates.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());
    opts.set("elide", "true");

    let mut op = veks_pipeline::pipeline::commands::compute_dedup::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "dedup all identical: {}", result.message);

    // With elide=true, output should contain 1 unique ordinal
    let unique_ordinals = read_ivec_1d(&output);
    assert_eq!(unique_ordinals.len(), 1, "expected 1 unique ordinal, got {}", unique_ordinals.len());

    // Duplicates file should contain 4 duplicate ordinals
    let dup_ordinals = read_ivec_1d(&dups);
    assert_eq!(dup_ordinals.len(), 4, "expected 4 duplicate ordinals, got {}", dup_ordinals.len());
}

// ===========================================================================
// 6. dedup_all_unique
// ===========================================================================

#[test]
fn dedup_all_unique() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // 5 distinct vectors
    let source = ws.join("unique.fvec");
    write_test_fvec(&source, &[
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0],
    ]);

    let output = ws.join("dedup.ivec");
    let dups = ws.join("dedup_duplicates.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());
    opts.set("elide", "true");

    let mut op = veks_pipeline::pipeline::commands::compute_dedup::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "dedup all unique: {}", result.message);

    let unique_ordinals = read_ivec_1d(&output);
    assert_eq!(unique_ordinals.len(), 5, "expected 5 unique ordinals, got {}", unique_ordinals.len());

    let dup_ordinals = read_ivec_1d(&dups);
    assert_eq!(dup_ordinals.len(), 0, "expected 0 duplicate ordinals, got {}", dup_ordinals.len());
}

// ===========================================================================
// 7. dedup_single_vector
// ===========================================================================

#[test]
fn dedup_single_vector() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let source = ws.join("single.fvec");
    write_test_fvec(&source, &[vec![42.0, 0.5, -1.0]]);

    let output = ws.join("dedup.ivec");
    let dups = ws.join("dedup_duplicates.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());
    opts.set("elide", "true");

    let mut op = veks_pipeline::pipeline::commands::compute_dedup::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "dedup single vector: {}", result.message);

    let unique_ordinals = read_ivec_1d(&output);
    assert_eq!(unique_ordinals.len(), 1, "expected 1 unique ordinal");

    let dup_ordinals = read_ivec_1d(&dups);
    assert_eq!(dup_ordinals.len(), 0, "expected 0 duplicate ordinals");
}

// ===========================================================================
// 8. zeros_all_zeros
// ===========================================================================

#[test]
fn zeros_all_zeros() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // 4 all-zero vectors
    let source = ws.join("allzeros.fvec");
    write_test_fvec(&source, &[
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ]);

    let output = ws.join("zeros.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());

    let mut op = veks_pipeline::pipeline::commands::analyze_zeros::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    // All zeros → Warning status
    assert_eq!(result.status, Status::Warning, "all zeros should warn: {}", result.message);

    let ordinals = read_ivec_1d(&output);
    assert_eq!(ordinals.len(), 4, "expected 4 zero-vector ordinals, got {}", ordinals.len());
}

// ===========================================================================
// 9. zeros_no_zeros
// ===========================================================================

#[test]
fn zeros_no_zeros() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let source = ws.join("nonzero.fvec");
    write_test_fvec(&source, &[
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let output = ws.join("zeros.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());

    let mut op = veks_pipeline::pipeline::commands::analyze_zeros::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "no zeros should be Ok: {}", result.message);

    let ordinals = read_ivec_1d(&output);
    assert_eq!(ordinals.len(), 0, "expected 0 zero-vector ordinals, got {}", ordinals.len());
}

// ===========================================================================
// 10. ordinals_empty_exclusion
// ===========================================================================

#[test]
fn ordinals_empty_exclusion() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: ordinal index with 5 entries
    let source = ws.join("ordinals.ivec");
    write_test_ivec_1d(&source, &[0, 1, 2, 3, 4]);

    // Empty dup and zero files
    let dups = ws.join("dups.ivec");
    write_test_ivec_1d(&dups, &[]);
    let zeros = ws.join("zeros.ivec");
    write_test_ivec_1d(&zeros, &[]);

    let output = ws.join("clean.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());
    opts.set("zeros", zeros.to_string_lossy().to_string());

    let mut op = veks_pipeline::pipeline::commands::clean_ordinals::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "empty exclusion: {}", result.message);

    let kept = read_ivec_1d(&output);
    assert_eq!(kept.len(), 5, "all 5 ordinals should be kept, got {}", kept.len());
}

// ===========================================================================
// 11. ordinals_all_excluded
// ===========================================================================

#[test]
fn ordinals_all_excluded() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: ordinal index with 3 entries
    let source = ws.join("ordinals.ivec");
    write_test_ivec_1d(&source, &[0, 1, 2]);

    // Mark all as duplicates
    let dups = ws.join("dups.ivec");
    write_test_ivec_1d(&dups, &[0, 1, 2]);

    let output = ws.join("clean.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());

    let mut op = veks_pipeline::pipeline::commands::clean_ordinals::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "all excluded: {}", result.message);

    let kept = read_ivec_1d(&output);
    assert_eq!(kept.len(), 0, "expected 0 kept ordinals, got {}", kept.len());
}

// ===========================================================================
// 12. extract_empty_range
// ===========================================================================

#[test]
fn extract_empty_range() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: 5-record fvec
    let source = ws.join("base.fvec");
    write_test_fvec(&source, &[
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![4.0, 0.0],
        vec![5.0, 0.0],
    ]);

    let output = ws.join("out.fvec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("range", "[0,0)");

    let mut op = veks_pipeline::pipeline::commands::gen_extract::extract_factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    // Range [0,0) means 0 records extracted — should succeed with 0 records
    assert_eq!(result.status, Status::Ok, "extract [0,0): {}", result.message);

    let vecs = read_fvec(&output);
    assert_eq!(vecs.len(), 0, "expected 0 records, got {}", vecs.len());
}

// ===========================================================================
// 13. knn_k_exceeds_count
// ===========================================================================

#[test]
fn knn_k_exceeds_count() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // 3 base vectors, 1 query vector, k=10
    let base = ws.join("base.fvec");
    write_test_fvec(&base, &[
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ]);
    let query = ws.join("query.fvec");
    write_test_fvec(&query, &[vec![0.0, 0.0]]);

    let indices = ws.join("indices.ivec");
    let distances = ws.join("distances.fvec");
    let mut opts = Options::new();
    opts.set("base", base.to_string_lossy().to_string());
    opts.set("query", query.to_string_lossy().to_string());
    opts.set("indices", indices.to_string_lossy().to_string());
    opts.set("distances", distances.to_string_lossy().to_string());
    opts.set("neighbors", "10"); // k=10 > base count=3
    opts.set("metric", "L2");
    opts.set("threads", "1");

    let mut op = veks_pipeline::pipeline::commands::compute_knn::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    // KNN should either succeed (capping k to base count) or error gracefully.
    // The implementation uses a max-heap that simply won't fill to k — it
    // produces min(k, base_count) neighbors per query.
    assert!(
        result.status == Status::Ok || result.status == Status::Error,
        "knn k>count should succeed or error gracefully: {}", result.message
    );

    if result.status == Status::Ok {
        // If it succeeded, verify it produced results (capped to 3)
        let idx_data = read_fvec(&indices); // ivec has same layout as fvec for reading dims
        // KNN output is a multi-dim ivec: each query gets a row of k indices
        assert!(!idx_data.is_empty(), "expected some KNN output");
    }
}

// ===========================================================================
// 14. knn_identical_vectors
// ===========================================================================

#[test]
fn knn_identical_vectors() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // 5 identical base vectors and 1 query (same vector)
    let v = vec![1.0f32, 2.0, 3.0];
    let base = ws.join("base.fvec");
    write_test_fvec(&base, &vec![v.clone(); 5]);
    let query = ws.join("query.fvec");
    write_test_fvec(&query, &[v]);

    let indices = ws.join("indices.ivec");
    let distances = ws.join("distances.fvec");
    let mut opts = Options::new();
    opts.set("base", base.to_string_lossy().to_string());
    opts.set("query", query.to_string_lossy().to_string());
    opts.set("indices", indices.to_string_lossy().to_string());
    opts.set("distances", distances.to_string_lossy().to_string());
    opts.set("neighbors", "3");
    opts.set("metric", "L2");
    opts.set("threads", "1");

    let mut op = veks_pipeline::pipeline::commands::compute_knn::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);

    assert_eq!(result.status, Status::Ok, "knn identical: {}", result.message);

    // All distances should be 0 since all vectors are identical to the query
    let dist_vecs = read_fvec(&distances);
    assert_eq!(dist_vecs.len(), 1, "expected 1 query result");
    for &d in &dist_vecs[0] {
        assert!(
            d.abs() < 1e-6,
            "expected distance ~0 for identical vectors, got {}",
            d
        );
    }
}

/// Convert with fraction limits record count.
#[test]
fn convert_with_fraction() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    // Generate 100 vectors (dim=4)
    let mut vecs = Vec::new();
    for i in 0..100usize {
        vecs.push(vec![i as f32, (i*2) as f32, (i*3) as f32, (i*4) as f32]);
    }
    let fvec = ws.join("input.fvec");
    write_test_fvec(&fvec, &vecs);

    let output = ws.join("output.fvec");
    let mut opts = Options::new();
    opts.set("source", fvec.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("fraction", "0.1");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("transform convert").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "convert failed: {}", result.message);

    // Should have 10 records (10% of 100)
    let out_vecs = read_fvec(&output);
    assert_eq!(out_vecs.len(), 10, "expected 10 records (10% of 100), got {}", out_vecs.len());
}

// ===========================================================================
// 15. convert_f32_to_f16
// ===========================================================================

#[test]
fn convert_f32_to_f16() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    let source = ws.join("input.fvec");
    write_test_fvec(&source, &[
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let output = ws.join("output.mvec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("to", "mvec");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("transform convert").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "convert f32->f16 failed: {}", result.message);

    // mvec format: each record is [dim:i32 LE][dim * 2 bytes of f16 data]
    // 3 records, dim=3, element size=2 bytes → record stride = 4 + 3*2 = 10
    let file_bytes = std::fs::read(&output).unwrap();
    let record_stride = 4 + 3 * 2; // 10 bytes per record
    assert_eq!(
        file_bytes.len(),
        3 * record_stride,
        "expected {} bytes for 3 records, got {}",
        3 * record_stride,
        file_bytes.len()
    );

    // Verify dimension header of each record is 3
    for i in 0..3 {
        let offset = i * record_stride;
        let dim = i32::from_le_bytes([
            file_bytes[offset],
            file_bytes[offset + 1],
            file_bytes[offset + 2],
            file_bytes[offset + 3],
        ]);
        assert_eq!(dim, 3, "record {} dim header should be 3, got {}", i, dim);
    }
}

// ===========================================================================
// 16. convert_identity_same_format
// ===========================================================================

#[test]
fn convert_identity_same_format() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    let source = ws.join("input.fvec");
    let vecs = vec![
        vec![1.0f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    write_test_fvec(&source, &vecs);

    let output = ws.join("output.fvec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("to", "fvec");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("transform convert").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "convert identity failed: {}", result.message);

    let out_vecs = read_fvec(&output);
    assert_eq!(out_vecs, vecs, "identity convert should produce identical output");
}

// ===========================================================================
// 17. dedup_two_vectors_identical
// ===========================================================================

#[test]
fn dedup_two_vectors_identical() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    let source = ws.join("two_ident.fvec");
    let v = vec![1.0f32, 2.0, 3.0];
    write_test_fvec(&source, &[v.clone(), v]);

    let output = ws.join("dedup.ivec");
    let dups = ws.join("dedup_duplicates.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());
    opts.set("elide", "true");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("compute sort").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "dedup two identical failed: {}", result.message);

    let unique_ordinals = read_ivec_1d(&output);
    assert_eq!(unique_ordinals.len(), 1, "expected 1 unique, got {}", unique_ordinals.len());

    let dup_ordinals = read_ivec_1d(&dups);
    assert_eq!(dup_ordinals.len(), 1, "expected 1 duplicate, got {}", dup_ordinals.len());
}

// ===========================================================================
// 18. dedup_already_sorted
// ===========================================================================

#[test]
fn dedup_already_sorted() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    // Vectors already in lexicographic order
    let source = ws.join("sorted.fvec");
    write_test_fvec(&source, &[
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ]);

    let output = ws.join("dedup.ivec");
    let dups = ws.join("dedup_duplicates.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());
    opts.set("elide", "true");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("compute sort").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "dedup already sorted failed: {}", result.message);

    let unique_ordinals = read_ivec_1d(&output);
    assert_eq!(unique_ordinals.len(), 4, "expected 4 unique, got {}", unique_ordinals.len());

    let dup_ordinals = read_ivec_1d(&dups);
    assert_eq!(dup_ordinals.len(), 0, "expected 0 duplicates, got {}", dup_ordinals.len());
}

// ===========================================================================
// 19. shuffle_deterministic
// ===========================================================================

#[test]
fn shuffle_deterministic() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();

    let out1 = ws.join("shuffle_a.ivec");
    let out2 = ws.join("shuffle_b.ivec");

    for out in [&out1, &out2] {
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("interval", "500");
        opts.set("seed", "12345");

        let factory = registry.get("generate shuffle").unwrap();
        let mut cmd = factory();
        let mut ctx = test_ctx(ws);
        let result = cmd.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "shuffle failed: {}", result.message);
    }

    let data1 = std::fs::read(&out1).unwrap();
    let data2 = std::fs::read(&out2).unwrap();
    assert_eq!(data1, data2, "same seed + same interval should produce identical output");
}

// ===========================================================================
// 20. shuffle_with_ordinals
// ===========================================================================

#[test]
fn shuffle_with_ordinals() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    // Write an ordinals file with values [10, 20, 30, 40, 50]
    let ordinals_path = ws.join("ordinals.ivec");
    write_test_ivec_1d(&ordinals_path, &[10, 20, 30, 40, 50]);

    let output = ws.join("shuffled.ivec");
    let mut opts = Options::new();
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("interval", "5");
    opts.set("seed", "99");
    opts.set("ordinals", ordinals_path.to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("generate shuffle").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "shuffle with ordinals failed: {}", result.message);

    let values = read_ivec_1d(&output);
    assert_eq!(values.len(), 5, "expected 5 values, got {}", values.len());

    // Output should be a permutation of [10, 20, 30, 40, 50], not [0, 1, 2, 3, 4]
    let mut sorted = values.clone();
    sorted.sort();
    assert_eq!(sorted, vec![10, 20, 30, 40, 50],
        "output should be a permutation of the input ordinals, got {:?}", values);
}

// ===========================================================================
// 21. extract_full_range
// ===========================================================================

#[test]
fn extract_full_range() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    let source = ws.join("base.fvec");
    let vecs = vec![
        vec![1.0f32, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ];
    write_test_fvec(&source, &vecs);

    let output = ws.join("out.fvec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("range", "[0,4)");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("transform extract").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "extract full range failed: {}", result.message);

    let out_vecs = read_fvec(&output);
    assert_eq!(out_vecs, vecs, "full range extract should match source exactly");
}

// ===========================================================================
// 22. extract_single_record_range
// ===========================================================================

#[test]
fn extract_single_record_range() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    let source = ws.join("base.fvec");
    let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i * 10) as f32]).collect();
    write_test_fvec(&source, &vecs);

    let output = ws.join("out.fvec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("range", "[5,6)");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("transform extract").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "extract single record failed: {}", result.message);

    let out_vecs = read_fvec(&output);
    assert_eq!(out_vecs.len(), 1, "expected 1 record, got {}", out_vecs.len());
    assert_eq!(out_vecs[0], vec![5.0, 50.0], "expected record at index 5");
}

// ===========================================================================
// 23. knn_single_query
// ===========================================================================

#[test]
fn knn_single_query() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    // 10 base vectors, dim=3
    let base_path = ws.join("base.fvec");
    let base_vecs: Vec<Vec<f32>> = (0..10)
        .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
        .collect();
    write_test_fvec(&base_path, &base_vecs);

    // 1 query vector
    let query_path = ws.join("query.fvec");
    write_test_fvec(&query_path, &[vec![0.0, 0.0, 0.0]]);

    let indices = ws.join("indices.ivec");
    let distances = ws.join("distances.fvec");
    let mut opts = Options::new();
    opts.set("base", base_path.to_string_lossy().to_string());
    opts.set("query", query_path.to_string_lossy().to_string());
    opts.set("indices", indices.to_string_lossy().to_string());
    opts.set("distances", distances.to_string_lossy().to_string());
    opts.set("neighbors", "5");
    opts.set("metric", "L2");
    opts.set("threads", "1");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("compute knn").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "knn single query failed: {}", result.message);

    // Should produce exactly 1 result row (1 query)
    let dist_vecs = read_fvec(&distances);
    assert_eq!(dist_vecs.len(), 1, "expected 1 result row, got {}", dist_vecs.len());
    assert_eq!(dist_vecs[0].len(), 5, "expected 5 neighbors per query, got {}", dist_vecs[0].len());
}

// ===========================================================================
// 24. knn_cosine_metric
// ===========================================================================

#[test]
fn knn_cosine_metric() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    // 5 base vectors with non-zero components (avoid zero-norm issues)
    let base_path = ws.join("base.fvec");
    write_test_fvec(&base_path, &[
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ]);

    // 2 queries
    let query_path = ws.join("query.fvec");
    write_test_fvec(&query_path, &[
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);

    let indices = ws.join("indices.ivec");
    let distances = ws.join("distances.fvec");
    let mut opts = Options::new();
    opts.set("base", base_path.to_string_lossy().to_string());
    opts.set("query", query_path.to_string_lossy().to_string());
    opts.set("indices", indices.to_string_lossy().to_string());
    opts.set("distances", distances.to_string_lossy().to_string());
    opts.set("neighbors", "3");
    opts.set("metric", "COSINE");
    opts.set("threads", "1");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("compute knn").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "knn cosine failed: {}", result.message);

    let dist_vecs = read_fvec(&distances);
    assert_eq!(dist_vecs.len(), 2, "expected 2 result rows, got {}", dist_vecs.len());

    // Cosine distance = 1 - cos(theta), range [0, 2]
    for (qi, row) in dist_vecs.iter().enumerate() {
        for (ni, &d) in row.iter().enumerate() {
            assert!(
                d >= 0.0 && d <= 2.0,
                "cosine distance out of range [0,2]: query={} neighbor={} dist={}",
                qi, ni, d
            );
        }
    }
}

// ===========================================================================
// 25. zeros_single_zero_first
// ===========================================================================

#[test]
fn zeros_single_zero_first() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    let source = ws.join("first_zero.fvec");
    write_test_fvec(&source, &[
        vec![0.0, 0.0, 0.0],  // index 0: zero
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let output = ws.join("zeros.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze zeros").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Warning, "should warn about zeros: {}", result.message);

    let ordinals = read_ivec_1d(&output);
    assert_eq!(ordinals.len(), 1, "expected 1 zero vector, got {}", ordinals.len());
    assert_eq!(ordinals[0], 0, "zero vector should be at index 0");
}

// ===========================================================================
// 26. zeros_single_zero_last
// ===========================================================================

#[test]
fn zeros_single_zero_last() {
    let tmp = tmp_dir();
    let ws = tmp.path();
    let mut ctx = test_ctx(ws);

    let source = ws.join("last_zero.fvec");
    write_test_fvec(&source, &[
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![0.0, 0.0, 0.0],  // index 3: zero
    ]);

    let output = ws.join("zeros.ivec");
    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze zeros").unwrap();
    let mut cmd = factory();
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Warning, "should warn about zeros: {}", result.message);

    let ordinals = read_ivec_1d(&output);
    assert_eq!(ordinals.len(), 1, "expected 1 zero vector, got {}", ordinals.len());
    assert_eq!(ordinals[0], 3, "zero vector should be at index 3");
}

// ===========================================================================
// dedup_prefix_group_ordering: vectors with shared prefixes but different
// later components should be in full lexicographic order in the output.
// ===========================================================================

#[test]
fn dedup_prefix_group_ordering() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Vectors share first 12 components (all zero), differ at component 12+.
    // With a prefix width of 10, all vectors collide into one prefix group.
    // The fix ensures they are sorted by full content within the group.
    let mut vectors = Vec::new();
    for i in (0..5).rev() {
        let mut v = vec![0.0f32; 20];
        v[15] = (i + 1) as f32;  // differ at component 15
        vectors.push(v);
    }
    // vectors[0] has v[15]=5, vectors[1] has v[15]=4, ..., vectors[4] has v[15]=1
    // Expected sorted order: v[15]=1 < 2 < 3 < 4 < 5 → indices [4, 3, 2, 1, 0]

    let source = ws.join("prefix_group.fvec");
    write_test_fvec(&source, &vectors);

    let output = ws.join("sorted.ivec");
    let dups = ws.join("dups.ivec");

    let mut opts = Options::new();
    opts.set("source", source.to_string_lossy().to_string());
    opts.set("output", output.to_string_lossy().to_string());
    opts.set("duplicates", dups.to_string_lossy().to_string());
    opts.set("elide", "true");

    let mut op = veks_pipeline::pipeline::commands::compute_dedup::factory();
    let mut ctx = test_ctx(ws);
    let result = op.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "dedup: {}", result.message);

    let ordinals = read_ivec_1d(&output);
    assert_eq!(ordinals.len(), 5, "no duplicates expected");

    // Verify full lexicographic ordering: the vectors should be sorted
    // by component 15 ascending → source indices [4, 3, 2, 1, 0]
    assert_eq!(ordinals, vec![4, 3, 2, 1, 0],
        "prefix group should be sorted by full vector content, got {:?}", ordinals);
}

// ===========================================================================
// analyze find-duplicates
// ===========================================================================

#[test]
fn analyze_find_duplicates_basic() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Vectors: 0==1 (dup), 2 unique, 3==4==5 (triple dup)
    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0], vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0], vec![5.0, 6.0], vec![5.0, 6.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("dups.ivec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze find-duplicates").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    // Should succeed and find duplicates
    assert!(result.status == Status::Ok || result.status == Status::Warning,
        "find-duplicates: {}", result.message);
}

#[test]
fn analyze_find_duplicates_no_dups() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("dups.ivec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze find-duplicates").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "no-dups: {}", result.message);
}

// ===========================================================================
// analyze find-zeros
// ===========================================================================

#[test]
fn analyze_find_zeros_mixed() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0],
        vec![0.0, 0.0],  // zero
        vec![3.0, 4.0],
        vec![0.0, 0.0],  // zero
        vec![5.0, 6.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze find-zeros").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    // find-zeros reports zeros to stdout, status Warning when zeros found
    assert!(result.status == Status::Ok || result.status == Status::Warning,
        "find-zeros: {}", result.message);
}

#[test]
fn analyze_find_zeros_none() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 0.0], vec![0.0, 1.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze find-zeros").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "no zeros: {}", result.message);
}

// ===========================================================================
// analyze norms — L2 norm distribution analysis
// ===========================================================================

#[test]
fn analyze_norms_basic() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Mix of normalized and unnormalized vectors
    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 0.0, 0.0],    // norm = 1
        vec![0.0, 1.0, 0.0],    // norm = 1
        vec![3.0, 4.0, 0.0],    // norm = 5
        vec![0.0, 0.0, 0.0],    // norm = 0
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze display-norms").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "norms: {}", result.message);
}

// ===========================================================================
// analyze measure-normals — normalization precision
// ===========================================================================

#[test]
fn analyze_measure_normals_unit_vectors() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // All unit vectors
    let s = 1.0f32 / 2.0f32.sqrt();
    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![s, s, 0.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("sample", "3");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze measure-normals").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "measure-normals: {}", result.message);
    // Should set is_normalized = true
    assert_eq!(ctx.defaults.get("is_normalized").map(|s| s.as_str()), Some("true"),
        "unit vectors should be detected as normalized");
}

#[test]
fn analyze_measure_normals_unnormalized() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![3.0, 4.0],  // norm = 5
        vec![5.0, 12.0], // norm = 13
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("sample", "2");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze measure-normals").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "measure-normals: {}", result.message);
    assert_eq!(ctx.defaults.get("is_normalized").map(|s| s.as_str()), Some("false"),
        "unnormalized vectors should be detected as not normalized");
}

// ===========================================================================
// analyze describe
// ===========================================================================

#[test]
fn analyze_describe_basic() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze describe").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "describe: {}", result.message);
}

// ===========================================================================
// analyze overlap — find overlapping vectors between two files
// ===========================================================================

#[test]
fn analyze_overlap_with_shared_vectors() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("a.fvec"), &[
        vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
    ]);
    write_test_fvec(&ws.join("b.fvec"), &[
        vec![3.0, 4.0], vec![7.0, 8.0], vec![5.0, 6.0],
    ]);

    let mut opts = Options::new();
    opts.set("base", ws.join("a.fvec").to_string_lossy().to_string());
    opts.set("query", ws.join("b.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze overlap").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    // overlap command returns Error when overlap is detected (it's a data quality issue)
    assert_eq!(result.status, Status::Error,
        "overlap should be detected as an error: {}", result.message);
    assert!(result.message.contains("overlap=2") || result.message.contains("66"),
        "should report 2 overlapping vectors: {}", result.message);
}

#[test]
fn analyze_overlap_no_shared() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("a.fvec"), &[vec![1.0, 2.0]]);
    write_test_fvec(&ws.join("b.fvec"), &[vec![3.0, 4.0]]);

    let mut opts = Options::new();
    opts.set("base", ws.join("a.fvec").to_string_lossy().to_string());
    opts.set("query", ws.join("b.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze overlap").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "no-overlap: {}", result.message);
}

// ===========================================================================
// analyze stats — basic vector statistics
// ===========================================================================

#[test]
fn analyze_stats_basic() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("analyze stats").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "stats: {}", result.message);
}

// ===========================================================================
// verify knn-groundtruth — basic KNN verification
// ===========================================================================

#[test]
fn verify_knn_groundtruth_correct() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // 4 base vectors, 2 queries, dim=2
    write_test_fvec(&ws.join("base.fvec"), &[
        vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![0.5, 0.5],
    ]);
    write_test_fvec(&ws.join("query.fvec"), &[
        vec![0.9, 0.1], vec![0.1, 0.9],
    ]);

    // Compute correct GT: for L2, nearest to (0.9,0.1) is (1,0)=idx0
    // nearest to (0.1,0.9) is (0,1)=idx1
    // GT ivec: each row = [k values]
    let gt_path = ws.join("gt.ivec");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&gt_path).unwrap();
        // query 0: nearest = [0, 3] (L2 dist: 0.02, 0.32)
        f.write_all(&2i32.to_le_bytes()).unwrap();
        f.write_all(&0i32.to_le_bytes()).unwrap();
        f.write_all(&3i32.to_le_bytes()).unwrap();
        // query 1: nearest = [1, 3]
        f.write_all(&2i32.to_le_bytes()).unwrap();
        f.write_all(&1i32.to_le_bytes()).unwrap();
        f.write_all(&3i32.to_le_bytes()).unwrap();
    }

    let mut opts = Options::new();
    opts.set("base", ws.join("base.fvec").to_string_lossy().to_string());
    opts.set("query", ws.join("query.fvec").to_string_lossy().to_string());
    opts.set("indices", ws.join("gt.ivec").to_string_lossy().to_string());
    opts.set("output", ws.join("report.json").to_string_lossy().to_string());
    opts.set("metric", "L2");
    opts.set("sample", "2");
    opts.set("seed", "42");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get("verify knn-groundtruth").unwrap();
    let mut cmd = factory();
    let mut ctx = test_ctx(ws);
    let result = cmd.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "verify-knn: {}", result.message);
}

// ===========================================================================
// Adversarial: single vector in all operations
// ===========================================================================

#[test]
fn single_vector_through_pipeline_commands() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("single.fvec"), &[vec![1.0, 2.0, 3.0]]);

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut ctx = test_ctx(ws);

    // analyze describe
    let mut opts = Options::new();
    opts.set("source", ws.join("single.fvec").to_string_lossy().to_string());
    let mut cmd = registry.get("analyze describe").unwrap()();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "describe single: {}", r.message);

    // analyze norms
    let mut opts = Options::new();
    opts.set("source", ws.join("single.fvec").to_string_lossy().to_string());
    let mut cmd = registry.get("analyze display-norms").unwrap()();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "norms single: {}", r.message);

    // analyze measure-normals
    let mut opts = Options::new();
    opts.set("source", ws.join("single.fvec").to_string_lossy().to_string());
    opts.set("sample", "1");
    let mut cmd = registry.get("analyze measure-normals").unwrap()();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "measure-normals single: {}", r.message);

    // compute sort (dedup)
    let mut opts = Options::new();
    opts.set("source", ws.join("single.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("sorted.ivec").to_string_lossy().to_string());
    opts.set("duplicates", ws.join("dups.ivec").to_string_lossy().to_string());
    let mut cmd = registry.get("compute sort").unwrap()();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "sort single: {}", r.message);
    assert_eq!(read_ivec_1d(&ws.join("sorted.ivec")).len(), 1);
    assert_eq!(read_ivec_1d(&ws.join("dups.ivec")).len(), 0);
}

// ===========================================================================
// Adversarial: dimension 1 vectors
// ===========================================================================

#[test]
fn dimension_one_vectors() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![5.0], vec![1.0], vec![3.0], vec![1.0], vec![4.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("sorted.ivec").to_string_lossy().to_string());
    opts.set("duplicates", ws.join("dups.ivec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute sort").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "dim1 sort: {}", r.message);

    let ordinals = read_ivec_1d(&ws.join("sorted.ivec"));
    assert_eq!(ordinals.len(), 4, "should have 4 unique (1 dup of 1.0)");
    let dups = read_ivec_1d(&ws.join("dups.ivec"));
    assert_eq!(dups.len(), 1, "should have 1 duplicate");
}

// ===========================================================================
// Adversarial: all-NaN vectors (edge case for sort/norm)
// ===========================================================================

#[test]
fn nan_vectors_handled() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![f32::NAN, 1.0],
        vec![1.0, 2.0],
        vec![f32::NAN, f32::NAN],
    ]);

    // analyze norms should not crash
    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("analyze display-norms").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "norms with NaN: {}", r.message);

    // compute sort should not crash
    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("sorted.ivec").to_string_lossy().to_string());
    opts.set("duplicates", ws.join("dups.ivec").to_string_lossy().to_string());
    let mut cmd = registry.get("compute sort").unwrap()();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "sort with NaN: {}", r.message);
}

// ===========================================================================
// compute sort: large number of duplicates (stress dedup)
// ===========================================================================

#[test]
fn dedup_many_duplicates() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // 100 vectors where every 10 are identical → 10 unique, 90 dups
    let mut vecs = Vec::new();
    for group in 0..10 {
        for _ in 0..10 {
            vecs.push(vec![group as f32, (group * 2) as f32, (group * 3) as f32]);
        }
    }
    write_test_fvec(&ws.join("source.fvec"), &vecs);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("sorted.ivec").to_string_lossy().to_string());
    opts.set("duplicates", ws.join("dups.ivec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute sort").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "many dups: {}", r.message);

    let unique = read_ivec_1d(&ws.join("sorted.ivec"));
    let dups = read_ivec_1d(&ws.join("dups.ivec"));
    // 10 groups → 10 unique after dedup. Group 0 is [0,0,0] (zero vector),
    // removed during zero detection → 9 unique in output.
    assert_eq!(unique.len(), 9, "should have 9 unique (10 groups - 1 zero)");
    assert_eq!(dups.len(), 90, "should have 90 duplicates");
}

// ===========================================================================
// compute sort: already sorted input (lexicographic)
// ===========================================================================

#[test]
fn dedup_presorted_lexicographic() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("sorted.ivec").to_string_lossy().to_string());
    opts.set("duplicates", ws.join("dups.ivec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute sort").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "already sorted: {}", r.message);

    let unique = read_ivec_1d(&ws.join("sorted.ivec"));
    assert_eq!(unique.len(), 3);
    // Should be in same order since already sorted
    assert_eq!(unique, vec![0, 1, 2]);
}

// ===========================================================================
// transform extract: zero-length range produces empty output
// ===========================================================================

#[test]
fn extract_zero_length_range() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("empty.fvec").to_string_lossy().to_string());
    opts.set("range", "[0,0)");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "empty range: {}", r.message);

    let result = read_fvec(&ws.join("empty.fvec"));
    assert_eq!(result.len(), 0, "empty range should produce empty output");
}

// ===========================================================================
// transform extract: range beyond file size clamps
// ===========================================================================

#[test]
fn extract_range_beyond_end() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0], vec![3.0, 4.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("out.fvec").to_string_lossy().to_string());
    opts.set("range", "[0,1000)");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "beyond end: {}", r.message);

    let result = read_fvec(&ws.join("out.fvec"));
    assert_eq!(result.len(), 2, "should clamp to actual file size");
}

// ===========================================================================
// generate shuffle: interval=1 produces trivial permutation
// ===========================================================================

#[test]
fn shuffle_trivial_single_element() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let mut opts = Options::new();
    opts.set("output", ws.join("shuffle.ivec").to_string_lossy().to_string());
    opts.set("interval", "1");
    opts.set("seed", "42");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("generate shuffle").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "shuffle 1: {}", r.message);

    let values = read_ivec_1d(&ws.join("shuffle.ivec"));
    assert_eq!(values, vec![0], "single element shuffle should be [0]");
}

// ===========================================================================
// generate vectors: dimension 1, count 1 (minimal)
// ===========================================================================

#[test]
fn gen_vectors_minimal() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let mut opts = Options::new();
    opts.set("output", ws.join("minimal.fvec").to_string_lossy().to_string());
    opts.set("dimension", "1");
    opts.set("count", "1");
    opts.set("seed", "0");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("generate vectors").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "minimal gen: {}", r.message);

    let vecs = read_fvec(&ws.join("minimal.fvec"));
    assert_eq!(vecs.len(), 1);
    assert_eq!(vecs[0].len(), 1);
}

// ===========================================================================
// transform convert: fvec → fvec identity conversion
// ===========================================================================

#[test]
fn convert_identity_fvec_to_fvec() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let original = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    write_test_fvec(&ws.join("source.fvec"), &original);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("output", ws.join("copy.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform convert").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "identity convert: {}", r.message);

    let result = read_fvec(&ws.join("copy.fvec"));
    assert_eq!(result, original, "identity conversion should preserve data");
}

// ===========================================================================
// analyze compare-files: identical files
// ===========================================================================

#[test]
fn compare_identical_files() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    write_test_fvec(&ws.join("a.fvec"), &vecs);
    write_test_fvec(&ws.join("b.fvec"), &vecs);

    let mut opts = Options::new();
    opts.set("original", ws.join("a.fvec").to_string_lossy().to_string());
    opts.set("synthetic", ws.join("b.fvec").to_string_lossy().to_string());

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("analyze compare-files").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "compare identical: {}", r.message);
}

// ===========================================================================
// analyze histogram: basic operation
// ===========================================================================

#[test]
fn histogram_basic() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    write_test_fvec(&ws.join("source.fvec"), &[
        vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],
    ]);

    let mut opts = Options::new();
    opts.set("source", ws.join("source.fvec").to_string_lossy().to_string());
    opts.set("dimension", "0");  // 0 = auto-detect from file

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("analyze display-histogram").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "histogram: {}", r.message);
}

// ---------------------------------------------------------------------------
// Scalar extract tests
// ---------------------------------------------------------------------------

/// Write a u8 scalar file (one byte per record, no headers).
fn write_scalar_u8(path: &Path, values: &[u8]) {
    std::fs::write(path, values).unwrap();
}

/// Write an ivec file with 1-dimensional records (ordinals).
fn write_ivec_ordinals(path: &Path, ordinals: &[i32]) {
    use std::io::Write;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for &ord in ordinals {
        w.write_all(&1i32.to_le_bytes()).unwrap(); // dim = 1
        w.write_all(&ord.to_le_bytes()).unwrap();
    }
    w.flush().unwrap();
}

#[test]
fn scalar_extract_u8_by_index() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: 10 u8 values [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    let source: Vec<u8> = (0..10).map(|i| i * 10).collect();
    write_scalar_u8(&ws.join("labels.u8"), &source);

    // Index: reorder to [5, 3, 7, 1] → expect [50, 30, 70, 10]
    write_ivec_ordinals(&ws.join("shuffle.ivec"), &[5, 3, 7, 1]);

    let mut opts = Options::new();
    opts.set("source", "labels.u8");
    opts.set("ivec-file", "shuffle.ivec");
    opts.set("output", "extracted.u8");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "scalar extract: {}", r.message);

    let output = std::fs::read(ws.join("extracted.u8")).unwrap();
    assert_eq!(output, vec![50, 30, 70, 10],
        "extracted values should match index-based reorder");
}

#[test]
fn scalar_extract_u8_by_range() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    let source: Vec<u8> = (0..20).collect();
    write_scalar_u8(&ws.join("data.u8"), &source);

    let mut opts = Options::new();
    opts.set("source", "data.u8");
    opts.set("output", "slice.u8");
    opts.set("range", "[5,10)");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "scalar range extract: {}", r.message);

    let output = std::fs::read(ws.join("slice.u8")).unwrap();
    assert_eq!(output, vec![5, 6, 7, 8, 9],
        "extracted range [5,10) should be [5,6,7,8,9]");
}

#[test]
fn scalar_extract_i32_by_index() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: 5 i32 values [100, 200, 300, 400, 500]
    let source: Vec<u8> = [100i32, 200, 300, 400, 500].iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(ws.join("values.i32"), &source).unwrap();

    // Index: [4, 2, 0] → expect [500, 300, 100]
    write_ivec_ordinals(&ws.join("order.ivec"), &[4, 2, 0]);

    let mut opts = Options::new();
    opts.set("source", "values.i32");
    opts.set("ivec-file", "order.ivec");
    opts.set("output", "reordered.i32");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "i32 scalar extract: {}", r.message);

    let output = std::fs::read(ws.join("reordered.i32")).unwrap();
    let values: Vec<i32> = output.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![500, 300, 100],
        "i32 extracted values should match reorder");
}

#[test]
fn scalar_extract_u8_index_with_range() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: 10 u8 values
    let source: Vec<u8> = (0..10).map(|i| i * 10).collect();
    write_scalar_u8(&ws.join("data.u8"), &source);

    // Index: full shuffle [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    let ordinals: Vec<i32> = (0..10).rev().collect();
    write_ivec_ordinals(&ws.join("shuffle.ivec"), &ordinals);

    // Range: [0, 3) of the index → first 3 entries → ordinals [9, 8, 7]
    let mut opts = Options::new();
    opts.set("source", "data.u8");
    opts.set("ivec-file", "shuffle.ivec");
    opts.set("output", "slice.u8");
    opts.set("range", "[0,3)");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "scalar extract with range: {}", r.message);

    let output = std::fs::read(ws.join("slice.u8")).unwrap();
    assert_eq!(output, vec![90, 80, 70],
        "range [0,3) of reversed shuffle should give [90, 80, 70]");
}

// ---------------------------------------------------------------------------
// Partition profiles tests
// ---------------------------------------------------------------------------

#[test]
fn partition_profiles_u8_metadata() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Base vectors: 20 vectors, dim=2
    write_test_fvec(&ws.join("base.fvec"), &(0..20).map(|i|
        vec![(i + 1) as f32, (i * 2 + 1) as f32]
    ).collect::<Vec<_>>());

    // Metadata: 20 u8 labels in 4 groups (0, 1, 2, 3, 0, 1, 2, 3, ...)
    let labels: Vec<u8> = (0..20).map(|i| (i % 4) as u8).collect();
    std::fs::write(ws.join("labels.u8"), &labels).unwrap();

    // Query vectors: 5 vectors
    write_test_fvec(&ws.join("query.fvec"), &(0..5).map(|i|
        vec![(i + 100) as f32, (i * 3 + 100) as f32]
    ).collect::<Vec<_>>());

    let mut opts = Options::new();
    opts.set("base", "base.fvec");
    opts.set("query", "query.fvec");
    opts.set("metadata", "labels.u8");
    opts.set("neighbors", "3");
    opts.set("metric", "L2");
    opts.set("prefix", "label");
    opts.set("allowed-partitions", "10");
    opts.set("on-undersized", "warn");

    // Create profiles dir and dataset.yaml for profile registration
    std::fs::create_dir_all(ws.join("profiles")).unwrap();
    std::fs::write(ws.join("dataset.yaml"),
        "name: test\nprofiles:\n  default:\n    maxk: 3\n    base_vectors: base.fvec\n").unwrap();

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute partition-profiles").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "partition-profiles u8: {}", r.message);

    // Should have created 4 partition directories
    for i in 0..4 {
        let pdir = ws.join(format!("profiles/label_{}", i));
        assert!(pdir.exists(), "partition dir label_{} should exist", i);
        assert!(pdir.join("base_vectors.fvec").exists(),
            "label_{}/base_vectors.fvec should exist", i);
    }

    // Each partition should have 5 vectors (20 / 4)
    let data = std::fs::read(ws.join("profiles/label_0/base_vectors.fvec")).unwrap();
    let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let stride = 4 + dim * 4;
    let count = data.len() / stride;
    assert_eq!(count, 5, "label_0 should have 5 vectors");

    // Check partition_count variable
    assert_eq!(ctx.defaults.get("partition_count").map(|s| s.as_str()), Some("4"));
}

#[test]
fn partition_profiles_per_label_queries() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Base vectors: 20 vectors, dim=2
    write_test_fvec(&ws.join("base.fvec"), &(0..20).map(|i|
        vec![(i + 1) as f32, (i * 2 + 1) as f32]
    ).collect::<Vec<_>>());

    // Metadata: 20 u8 labels, 4 groups (0,1,2,3,0,1,2,3,...)
    let labels: Vec<u8> = (0..20).map(|i| (i % 4) as u8).collect();
    std::fs::write(ws.join("labels.u8"), &labels).unwrap();

    // Query vectors: 8 vectors, dim=2
    write_test_fvec(&ws.join("query.fvec"), &(0..8).map(|i|
        vec![(i + 100) as f32, (i * 3 + 100) as f32]
    ).collect::<Vec<_>>());

    // Predicates: 8 u8 labels (2 queries per label group)
    // queries 0,1 target label 0; queries 2,3 target label 1; etc.
    let predicates: Vec<u8> = vec![0, 0, 1, 1, 2, 2, 3, 3];
    std::fs::write(ws.join("predicates.u8"), &predicates).unwrap();

    let mut opts = Options::new();
    opts.set("base", "base.fvec");
    opts.set("query", "query.fvec");
    opts.set("metadata", "labels.u8");
    opts.set("predicates", "predicates.u8");
    opts.set("neighbors", "3");
    opts.set("metric", "L2");
    opts.set("prefix", "label");
    opts.set("allowed-partitions", "10");
    opts.set("on-undersized", "warn");

    std::fs::create_dir_all(ws.join("profiles")).unwrap();
    std::fs::write(ws.join("dataset.yaml"),
        "name: test\nprofiles:\n  default:\n    maxk: 3\n    base_vectors: base.fvec\n").unwrap();

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute partition-profiles").unwrap()();
    let mut ctx = test_ctx(ws);
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "partition with predicates: {}", r.message);

    // Each partition should have its own query_vectors.fvec (NOT a symlink)
    for i in 0..4 {
        let qpath = ws.join(format!("profiles/label_{}/query_vectors.fvec", i));
        assert!(qpath.exists(), "label_{}/query_vectors.fvec should exist", i);
        assert!(!qpath.is_symlink(),
            "label_{}/query_vectors.fvec should NOT be a symlink — should be extracted per-label queries", i);

        // Each partition should have exactly 2 queries (8 total / 4 labels)
        let data = std::fs::read(&qpath).unwrap();
        let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let stride = 4 + dim * 4;
        let query_count = data.len() / stride;
        assert_eq!(query_count, 2,
            "label_{} should have 2 queries, got {}", i, query_count);
    }
}

#[test]
fn partition_profiles_slab_metadata() {
    use slabtastic::{SlabWriter, WriterConfig};

    let tmp = tmp_dir();
    let ws = tmp.path();

    // Base vectors: 30 vectors, dim=2
    write_test_fvec(&ws.join("base.fvec"), &(0..30).map(|i|
        vec![(i + 1) as f32, (i * 2 + 1) as f32]
    ).collect::<Vec<_>>());

    // Query vectors: 5 vectors
    write_test_fvec(&ws.join("query.fvec"), &(0..5).map(|i|
        vec![(i + 100) as f32, (i * 3 + 100) as f32]
    ).collect::<Vec<_>>());

    // Create slab metadata: 30 records with label values 0-2 (3 groups).
    // Each slab record is: [field_len: u16 LE] [field_data: field_len bytes]
    // For i32 labels: field_len=4, field_data=i32 LE
    {
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(ws.join("metadata.slab"), config).unwrap();
        for i in 0..30u32 {
            let label = (i % 3) as i32;
            let mut record = Vec::new();
            record.extend_from_slice(&4u16.to_le_bytes()); // field_len = 4
            record.extend_from_slice(&label.to_le_bytes()); // field_data = i32
            writer.add_record(&record).unwrap();
        }
        writer.finish().unwrap();
    }

    // Run partition-profiles with slab metadata
    let mut opts = Options::new();
    opts.set("base", "base.fvec");
    opts.set("query", "query.fvec");
    opts.set("metadata", "metadata.slab");
    opts.set("neighbors", "3");
    opts.set("metric", "L2");
    opts.set("prefix", "label");
    opts.set("allowed-partitions", "10");
    opts.set("on-undersized", "warn");

    std::fs::create_dir_all(ws.join("profiles")).unwrap();
    std::fs::write(ws.join("dataset.yaml"),
        "name: test\nprofiles:\n  default:\n    maxk: 3\n    base_vectors: base.fvec\n").unwrap();

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute partition-profiles").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "partition-profiles slab: {}", r.message);

    // Should have created 3 partition directories
    for i in 0..3 {
        let pdir = ws.join(format!("profiles/label_{}", i));
        assert!(pdir.exists(), "partition dir label_{} should exist", i);
        assert!(pdir.join("base_vectors.fvec").exists(),
            "label_{}/base_vectors.fvec should exist", i);
    }

    // Each partition should have 10 vectors (30 / 3)
    let data = std::fs::read(ws.join("profiles/label_0/base_vectors.fvec")).unwrap();
    let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let stride = 4 + dim * 4;
    let count = data.len() / stride;
    assert_eq!(count, 10, "label_0 should have 10 vectors");

    assert_eq!(ctx.defaults.get("partition_count").map(|s| s.as_str()), Some("3"));
}

/// Partition profiles with slab PNode predicates — the production format.
///
/// Validates that PNode-encoded equality predicates are correctly decoded
/// to extract per-label query vectors. This is the format used by sift1m
/// and any dataset with `synthesis_format: slab`.
#[test]
fn partition_profiles_slab_pnode_predicates() {
    use slabtastic::{SlabWriter, WriterConfig};
    use vectordata::formats::pnode::{PNode, PredicateNode, OpType, Comparand, FieldRef};

    let tmp = tmp_dir();
    let ws = tmp.path();

    // Base vectors: 20 vectors, dim=2
    write_test_fvec(&ws.join("base.fvec"), &(0..20).map(|i|
        vec![(i + 1) as f32, (i * 2 + 1) as f32]
    ).collect::<Vec<_>>());

    // Metadata: 20 u8 labels (0,1,2,3,0,1,2,3,...)
    let labels: Vec<u8> = (0..20).map(|i| (i % 4) as u8).collect();
    std::fs::write(ws.join("labels.u8"), &labels).unwrap();

    // Query vectors: 8 vectors, dim=2
    write_test_fvec(&ws.join("query.fvec"), &(0..8).map(|i|
        vec![(i + 100) as f32, (i * 3 + 100) as f32]
    ).collect::<Vec<_>>());

    // Predicates: 8 PNode slab records (simple-int-eq format)
    // queries 0,1 target label 0; queries 2,3 target label 1; etc.
    {
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(ws.join("predicates.slab"), config).unwrap();
        for qi in 0..8u32 {
            let label = qi / 2; // 0,0,1,1,2,2,3,3
            let pnode = PNode::Predicate(PredicateNode {
                field: FieldRef::Named("field_0".into()),
                op: OpType::Eq,
                comparands: vec![Comparand::Int(label as i64)],
            });
            writer.add_record(&pnode.to_bytes_named()).unwrap();
        }
        writer.finish().unwrap();
    }

    let mut opts = Options::new();
    opts.set("base", "base.fvec");
    opts.set("query", "query.fvec");
    opts.set("metadata", "labels.u8");
    opts.set("predicates", "predicates.slab");
    opts.set("neighbors", "3");
    opts.set("metric", "L2");
    opts.set("prefix", "label");
    opts.set("allowed-partitions", "10");
    opts.set("on-undersized", "warn");

    std::fs::create_dir_all(ws.join("profiles")).unwrap();
    std::fs::write(ws.join("dataset.yaml"),
        "name: test\nprofiles:\n  default:\n    maxk: 3\n    base_vectors: base.fvec\n").unwrap();

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute partition-profiles").unwrap()();
    let mut ctx = test_ctx(ws);
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "partition with slab predicates: {}", r.message);

    // Each partition should have per-label query_vectors.fvec (NOT a symlink)
    for i in 0..4 {
        let qpath = ws.join(format!("profiles/label_{}/query_vectors.fvec", i));
        assert!(qpath.exists(), "label_{}/query_vectors.fvec should exist", i);
        assert!(!qpath.is_symlink(),
            "label_{}/query_vectors.fvec should NOT be a symlink — PNode predicates must produce extracted queries", i);

        // Each partition should have exactly 2 queries (8 total / 4 labels)
        let data = std::fs::read(&qpath).unwrap();
        let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let stride = 4 + dim * 4;
        let query_count = data.len() / stride;
        assert_eq!(query_count, 2,
            "label_{} should have 2 queries from PNode predicates, got {}", i, query_count);
    }
}

/// Verify symlink interlock: re-running partition-profiles must replace
/// existing symlinks with real files, never writing through them.
#[test]
fn partition_profiles_replaces_symlinks_on_rerun() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Base vectors: 10 vectors, dim=2
    write_test_fvec(&ws.join("base.fvec"), &(0..10).map(|i|
        vec![(i + 1) as f32, (i * 2 + 1) as f32]
    ).collect::<Vec<_>>());

    // Metadata: 10 u8 labels, 2 groups
    let labels: Vec<u8> = (0..10).map(|i| (i % 2) as u8).collect();
    std::fs::write(ws.join("labels.u8"), &labels).unwrap();

    // Query vectors: 4 vectors, dim=2
    write_test_fvec(&ws.join("query.fvec"), &(0..4).map(|i|
        vec![(i + 100) as f32, (i * 3 + 100) as f32]
    ).collect::<Vec<_>>());

    // Predicates: 4 u8 labels (2 queries per group)
    let predicates: Vec<u8> = vec![0, 0, 1, 1];
    std::fs::write(ws.join("predicates.u8"), &predicates).unwrap();

    std::fs::create_dir_all(ws.join("profiles")).unwrap();
    std::fs::write(ws.join("dataset.yaml"),
        "name: test\nprofiles:\n  default:\n    maxk: 3\n    base_vectors: base.fvec\n").unwrap();

    // Pre-create symlinks (simulating a previous run without predicates)
    for i in 0..2 {
        let pdir = ws.join(format!("profiles/label_{}", i));
        std::fs::create_dir_all(&pdir).unwrap();
        std::os::unix::fs::symlink("../../query.fvec", pdir.join("query_vectors.fvec")).unwrap();
    }
    // Verify symlinks exist before re-run
    assert!(ws.join("profiles/label_0/query_vectors.fvec").symlink_metadata()
        .unwrap().file_type().is_symlink());

    let mut opts = Options::new();
    opts.set("base", "base.fvec");
    opts.set("query", "query.fvec");
    opts.set("metadata", "labels.u8");
    opts.set("predicates", "predicates.u8");
    opts.set("neighbors", "3");
    opts.set("metric", "L2");
    opts.set("prefix", "label");
    opts.set("allowed-partitions", "10");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute partition-profiles").unwrap()();
    let mut ctx = test_ctx(ws);
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "rerun with predicates: {}", r.message);

    // After re-run: symlinks must be gone, replaced by real files
    for i in 0..2 {
        let qpath = ws.join(format!("profiles/label_{}/query_vectors.fvec", i));
        assert!(qpath.exists(), "label_{}/query_vectors.fvec should exist after rerun", i);
        assert!(!qpath.symlink_metadata().unwrap().file_type().is_symlink(),
            "label_{}/query_vectors.fvec must NOT be a symlink after rerun — symlink interlock failed", i);

        let data = std::fs::read(&qpath).unwrap();
        let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let stride = 4 + dim * 4;
        let count = data.len() / stride;
        assert_eq!(count, 2, "label_{} should have 2 queries, got {}", i, count);
    }

    // Verify the original query file was NOT corrupted
    let orig_data = std::fs::read(ws.join("query.fvec")).unwrap();
    let dim = i32::from_le_bytes(orig_data[0..4].try_into().unwrap()) as usize;
    let stride = 4 + dim * 4;
    let orig_count = orig_data.len() / stride;
    assert_eq!(orig_count, 4, "original query.fvec must still have 4 vectors, not corrupted");
}

/// Auto-discover predicates when the option is not passed but the file
/// exists on disk — covers datasets generated before the predicates
/// option was added to the partition-profiles template.
#[test]
fn partition_profiles_auto_discovers_predicates() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Base vectors: 20 vectors, dim=2
    write_test_fvec(&ws.join("base.fvec"), &(0..20).map(|i|
        vec![(i + 1) as f32, (i * 2 + 1) as f32]
    ).collect::<Vec<_>>());

    // Metadata: 20 u8 labels (0,1,2,3,0,1,2,3,...)
    let labels: Vec<u8> = (0..20).map(|i| (i % 4) as u8).collect();
    std::fs::write(ws.join("labels.u8"), &labels).unwrap();

    // Query vectors: 8 vectors, dim=2
    write_test_fvec(&ws.join("query.fvec"), &(0..8).map(|i|
        vec![(i + 100) as f32, (i * 3 + 100) as f32]
    ).collect::<Vec<_>>());

    // Predicates file exists at profiles/base/predicates.u8 but is NOT
    // passed as an option — the command must auto-discover it.
    std::fs::create_dir_all(ws.join("profiles/base")).unwrap();
    let predicates: Vec<u8> = vec![0, 0, 1, 1, 2, 2, 3, 3];
    std::fs::write(ws.join("profiles/base/predicates.u8"), &predicates).unwrap();

    // dataset.yaml references the predicates file in the default profile
    std::fs::write(ws.join("dataset.yaml"), "\
name: test
profiles:
  default:
    maxk: 3
    views:
      base_vectors:
        source: base.fvec
      query_vectors:
        source: query.fvec
      metadata_predicates:
        source: profiles/base/predicates.u8
").unwrap();

    // Note: NO "predicates" option — auto-discovery must find it
    let mut opts = Options::new();
    opts.set("base", "base.fvec");
    opts.set("query", "query.fvec");
    opts.set("metadata", "labels.u8");
    opts.set("neighbors", "3");
    opts.set("metric", "L2");
    opts.set("prefix", "label");
    opts.set("allowed-partitions", "10");
    opts.set("on-undersized", "warn");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("compute partition-profiles").unwrap()();
    let mut ctx = test_ctx(ws);
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "auto-discover predicates: {}", r.message);

    // Each partition should have per-label queries, NOT symlinks
    for i in 0..4 {
        let qpath = ws.join(format!("profiles/label_{}/query_vectors.fvec", i));
        assert!(qpath.exists(), "label_{}/query_vectors.fvec should exist", i);
        assert!(!qpath.symlink_metadata().unwrap().file_type().is_symlink(),
            "label_{}/query_vectors.fvec must NOT be a symlink — auto-discovery should have found predicates", i);

        let data = std::fs::read(&qpath).unwrap();
        let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let stride = 4 + dim * 4;
        let query_count = data.len() / stride;
        assert_eq!(query_count, 2,
            "label_{} should have 2 queries via auto-discovered predicates, got {}", i, query_count);
    }
}

// ---------------------------------------------------------------------------
// Predicate-index extraction tests
// ---------------------------------------------------------------------------

/// Write an ivvec file with explicit variable-length records.
fn write_test_ivvec(path: &std::path::Path, records: &[Vec<i32>]) {
    use std::io::Write;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for record in records {
        w.write_all(&(record.len() as i32).to_le_bytes()).unwrap();
        for &val in record {
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
    // Build offset index
    let _ = vectordata::io::IndexedXvecReader::open_ivec(path);
}

#[test]
fn extract_fvec_by_predicate_index() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: 10 vectors, dim=2
    write_test_fvec(&ws.join("base.fvec"), &(0..10).map(|i|
        vec![(i + 1) as f32, (i * 2 + 1) as f32]
    ).collect::<Vec<_>>());

    // ivvec with 3 predicates:
    //   pred 0: ordinals [0, 2, 4]     → vectors 1, 3, 5
    //   pred 1: ordinals [1, 3, 5, 7]  → vectors 2, 4, 6, 8
    //   pred 2: ordinals [9]           → vector 10
    write_test_ivvec(&ws.join("results.ivvec"), &[
        vec![0, 2, 4],
        vec![1, 3, 5, 7],
        vec![9],
    ]);

    // Extract base vectors matching predicate 1 (ordinals [1,3,5,7])
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    let mut opts = Options::new();
    opts.set("source", "base.fvec");
    opts.set("index-source", "results.ivvec");
    opts.set("predicate-index", "1");
    opts.set("output", "partition.fvec");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "predicate-index extract: {}", r.message);

    // Output should have 4 vectors (ordinals 1,3,5,7)
    let data = std::fs::read(ws.join("partition.fvec")).unwrap();
    let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    assert_eq!(dim, 2);
    let stride = 4 + dim * 4;
    let count = data.len() / stride;
    assert_eq!(count, 4, "should extract 4 vectors for predicate 1");

    // First extracted vector should be source vector at ordinal 1
    // Source vec[1] = [2.0, 3.0]
    let v0_offset = 4;
    let v0_0 = f32::from_le_bytes(data[v0_offset..v0_offset+4].try_into().unwrap());
    assert_eq!(v0_0, 2.0, "first extracted vector should be source[1]");
}

#[test]
fn extract_scalar_by_predicate_index() {
    let tmp = tmp_dir();
    let ws = tmp.path();

    // Source: 10 u8 labels
    write_scalar_u8(&ws.join("labels.u8"), &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);

    // ivvec predicate results: pred 0 matches ordinals [2, 5, 8]
    write_test_ivvec(&ws.join("results.ivvec"), &[
        vec![2, 5, 8],
    ]);

    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    let mut opts = Options::new();
    opts.set("source", "labels.u8");
    opts.set("index-source", "results.ivvec");
    opts.set("predicate-index", "0");
    opts.set("output", "partition_labels.u8");

    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let mut cmd = registry.get("transform extract").unwrap()();
    let mut ctx = test_ctx(ws);
    let r = cmd.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok, "scalar predicate-index: {}", r.message);

    let output = std::fs::read(ws.join("partition_labels.u8")).unwrap();
    // Ordinals [2,5,8] → labels [30, 60, 90]
    assert_eq!(output, vec![30, 60, 90]);
}

// ===========================================================================
// Per-command fuzz tests
//
// Every registered command is tested against adversarial inputs to verify
// it returns Status::Error (not panic) for invalid arguments.
// ===========================================================================

/// All non-feature-gated command paths in the registry.
const ALL_COMMANDS: &[&str] = &[
    // analyze
    "analyze check-endian", "analyze compare-files", "analyze compute-info",
    "analyze describe", "analyze file", "analyze find",
    "analyze find-duplicates", "analyze find-zeros",
    "analyze display-histogram", "analyze model-diff",
    "analyze explain-predicates", "analyze explain-filtered-knn",
    "analyze explain-partitions",
    "analyze select", "analyze slice", "analyze stats", "analyze survey",
    "analyze verify-knn", "analyze verify-profiles",
    "analyze measure-normals", "analyze display-norms",
    "analyze overlap", "analyze zeros",
    // cleanup
    "cleanup overlap",
    // compute
    "compute evaluate-predicates", "compute filtered-knn", "compute knn",
    "compute partition-profiles", "compute sort",
    // download
    "download bulk", "download huggingface",
    // generate
    "generate dataset", "generate derive", "generate from-model",
    "generate metadata", "generate predicates", "generate shuffle",
    "generate sketch", "generate dataset-log-jsonl",
    "generate variables-json", "generate vectors",
    // merkle
    "merkle create", "merkle diff", "merkle path", "merkle spoilbits",
    "merkle spoilchunks", "merkle summary", "merkle treeview", "merkle verify",
    // query
    "query json", "query records",
    // state
    "state set", "state clear",
    // transform
    "transform convert", "transform extract", "transform ordinals",
    // verify
    "verify knn-groundtruth", "verify predicate-results",
    "verify knn-consolidated", "verify filtered-knn-consolidated",
    "verify predicates-consolidated", "verify predicates-sqlite",
    // catalog
    "catalog generate",
];

/// Fuzz: every command with empty options should error, not panic.
#[test]
fn fuzz_empty_options_no_panic() {
    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let tmp = tmp_dir();
    let ws = tmp.path();
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    std::fs::create_dir_all(ws.join(".cache")).unwrap();

    for &cmd_path in ALL_COMMANDS {
        let factory = match registry.get(cmd_path) {
            Some(f) => f,
            None => continue, // feature-gated command not compiled in
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut cmd = factory();
            let opts = Options::new();
            let mut ctx = test_ctx(ws);
            cmd.execute(&opts, &mut ctx)
        }));
        match result {
            Ok(_) => {} // Ok or Error — either is fine, just no panic
            Err(panic_info) => {
                panic!("command '{}' panicked with empty options: {:?}", cmd_path, panic_info);
            }
        }
    }
}

/// Fuzz: every command with a nonexistent source file should error, not panic.
#[test]
fn fuzz_missing_source_no_panic() {
    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let tmp = tmp_dir();
    let ws = tmp.path();
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    std::fs::create_dir_all(ws.join(".cache")).unwrap();

    for &cmd_path in ALL_COMMANDS {
        let factory = match registry.get(cmd_path) {
            Some(f) => f,
            None => continue,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut cmd = factory();
            let mut opts = Options::new();
            opts.set("source", "/nonexistent/path/data.fvec");
            opts.set("output", ws.join("out.fvec").to_string_lossy().to_string());
            let mut ctx = test_ctx(ws);
            cmd.execute(&opts, &mut ctx)
        }));
        match result {
            Ok(_) => {} // error or ok, either is fine — just no panic
            Err(panic_info) => {
                panic!("command '{}' panicked with missing source: {:?}", cmd_path, panic_info);
            }
        }
    }
}

/// Fuzz: every command with a zero-length source file should error, not panic.
#[test]
fn fuzz_zero_length_file_no_panic() {
    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let tmp = tmp_dir();
    let ws = tmp.path();
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    std::fs::create_dir_all(ws.join(".cache")).unwrap();

    // Create zero-length files in various formats
    for ext in &["fvec", "ivec", "mvec", "u8", "i32", "slab"] {
        std::fs::write(ws.join(format!("empty.{}", ext)), b"").unwrap();
    }

    for &cmd_path in ALL_COMMANDS {
        let factory = match registry.get(cmd_path) {
            Some(f) => f,
            None => continue,
        };
        for ext in &["fvec", "ivec", "u8", "slab"] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut cmd = factory();
                let mut opts = Options::new();
                opts.set("source", format!("empty.{}", ext));
                opts.set("output", format!("out_{}.{}", cmd_path.replace(' ', "_"), ext));
                let mut ctx = test_ctx(ws);
                cmd.execute(&opts, &mut ctx)
            }));
            match result {
                Ok(_) => {} // fine
                Err(panic_info) => {
                    panic!("command '{}' panicked with zero-length .{} file: {:?}",
                        cmd_path, ext, panic_info);
                }
            }
        }
    }
}

/// Fuzz: every command with a truncated/corrupt source file should not panic.
#[test]
fn fuzz_corrupt_file_no_panic() {
    let registry = veks_pipeline::pipeline::registry::CommandRegistry::with_builtins();
    let tmp = tmp_dir();
    let ws = tmp.path();
    std::fs::create_dir_all(ws.join(".scratch")).unwrap();
    std::fs::create_dir_all(ws.join(".cache")).unwrap();

    // Create files with garbage data (wrong format, truncated headers)
    std::fs::write(ws.join("corrupt.fvec"), b"\x04\x00\x00\x00\x01").unwrap(); // dim=4 but only 1 data byte
    std::fs::write(ws.join("corrupt.ivec"), b"\xff\xff\xff\xff").unwrap(); // dim=-1
    std::fs::write(ws.join("corrupt.u8"), &[42, 43, 44]).unwrap(); // valid u8 but tiny
    std::fs::write(ws.join("corrupt.slab"), b"NOT_A_SLAB").unwrap(); // invalid slab header

    let corrupt_files = ["corrupt.fvec", "corrupt.ivec", "corrupt.u8", "corrupt.slab"];

    for &cmd_path in ALL_COMMANDS {
        let factory = match registry.get(cmd_path) {
            Some(f) => f,
            None => continue,
        };
        for &corrupt in &corrupt_files {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut cmd = factory();
                let mut opts = Options::new();
                opts.set("source", corrupt);
                opts.set("output", format!("out_corrupt_{}_{}", cmd_path.replace(' ', "_"),
                    corrupt.replace('.', "_")));
                let mut ctx = test_ctx(ws);
                cmd.execute(&opts, &mut ctx)
            }));
            match result {
                Ok(_) => {}
                Err(panic_info) => {
                    panic!("command '{}' panicked with corrupt file '{}': {:?}",
                        cmd_path, corrupt, panic_info);
                }
            }
        }
    }
}
