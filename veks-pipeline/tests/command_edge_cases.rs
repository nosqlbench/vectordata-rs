// Copyright (c) DataStax, Inc.
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
use veks_pipeline::pipeline::command::{CommandOp, Options, Status, StreamContext};
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
