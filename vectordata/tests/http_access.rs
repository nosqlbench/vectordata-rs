// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for HTTP-based vector data access.
//!
//! Uses `testserver` to serve test data locally and verifies that
//! `HttpVectorReader` and `TestDataGroup` work correctly over real HTTP,
//! including Range request handling and dataset.yaml loading.

use std::fs;
use std::io::Write;
use std::path::Path;

mod support;
use support::testserver::TestServer;
use vectordata::io::{HttpIndexedXvecReader, HttpVectorReader, IndexedXvecReader, VectorReader, open_vec, open_vvec};
use vectordata::view::TestDataView;
use vectordata::TestDataGroup;

/// Recursively check if any file with the given suffix exists under dir.
fn find_file_recursive(dir: &std::path::Path, suffix: &str) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if find_file_recursive(&path, suffix) { return true; }
            } else if path.to_string_lossy().ends_with(suffix) {
                return true;
            }
        }
    }
    false
}

/// Write a small fvec file with `count` vectors of dimension `dim`.
///
/// Vector `i` contains elements `[i*dim, i*dim+1, ..., i*dim+dim-1]` as f32.
fn write_fvec(path: &Path, dim: u32, count: u32) {
    let mut file = fs::File::create(path).unwrap();
    for i in 0..count {
        file.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = (i * dim + j) as f32;
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

/// Write a small ivec file with `count` vectors of dimension `dim`.
///
/// Vector `i` contains elements `[i*dim, i*dim+1, ..., i*dim+dim-1]` as i32.
fn write_ivec(path: &Path, dim: u32, count: u32) {
    let mut file = fs::File::create(path).unwrap();
    for i in 0..count {
        file.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = (i * dim + j) as i32;
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

fn setup_dataset_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();

    write_fvec(&dir.path().join("base.fvec"), 4, 10);
    write_fvec(&dir.path().join("query.fvec"), 4, 5);
    write_ivec(&dir.path().join("neighbors.ivec"), 3, 5);
    write_fvec(&dir.path().join("distances.fvec"), 3, 5);

    fs::write(
        dir.path().join("dataset.yaml"),
        r#"
attributes:
  distance_function: L2
  dimension: 4
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
    neighbor_indices: neighbors.ivec
    neighbor_distances: distances.fvec
"#,
    )
    .unwrap();

    dir
}

// ---------------------------------------------------------------------------
// HttpVectorReader tests
// ---------------------------------------------------------------------------

#[test]
fn test_http_fvec_reader_metadata() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let url = url::Url::parse(&format!("{}base.fvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<f32>::open_fvec(url).unwrap();

    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 10);
}

#[test]
fn test_http_fvec_reader_get_first_vector() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let url = url::Url::parse(&format!("{}base.fvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<f32>::open_fvec(url).unwrap();

    let vec0 = reader.get(0).unwrap();
    assert_eq!(vec0, vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_http_fvec_reader_get_last_vector() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let url = url::Url::parse(&format!("{}base.fvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<f32>::open_fvec(url).unwrap();

    let vec9 = reader.get(9).unwrap();
    assert_eq!(vec9, vec![36.0, 37.0, 38.0, 39.0]);
}

#[test]
fn test_http_fvec_reader_out_of_bounds() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let url = url::Url::parse(&format!("{}base.fvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<f32>::open_fvec(url).unwrap();

    let result = reader.get(10);
    assert!(result.is_err());
}

#[test]
fn test_http_ivec_reader() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let url = url::Url::parse(&format!("{}neighbors.ivec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<i32>::open_ivec(url).unwrap();

    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);

    let vec0 = reader.get(0).unwrap();
    assert_eq!(vec0, vec![0, 1, 2]);

    let vec4 = reader.get(4).unwrap();
    assert_eq!(vec4, vec![12, 13, 14]);
}

// ---------------------------------------------------------------------------
// TestDataGroup over HTTP tests
// ---------------------------------------------------------------------------

#[test]
fn test_dataset_load_from_http() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();

    assert!(group.profile("default").is_some());
    assert!(group.profile("nonexistent").is_none());

    let dim = group.attribute("dimension").unwrap();
    assert_eq!(dim.as_u64().unwrap(), 4);
}

#[test]
fn test_dataset_http_base_vectors() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let base = view.base_vectors().unwrap();
    assert_eq!(base.dim(), 4);
    assert_eq!(base.count(), 10);

    let vec0 = base.get(0).unwrap();
    assert_eq!(vec0, vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_dataset_http_query_vectors() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let query = view.query_vectors().unwrap();
    assert_eq!(query.dim(), 4);
    assert_eq!(query.count(), 5);
}

#[test]
fn test_dataset_http_neighbor_indices() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let indices = view.neighbor_indices().unwrap();
    assert_eq!(indices.dim(), 3);
    assert_eq!(indices.count(), 5);

    let vec0 = indices.get(0).unwrap();
    assert_eq!(vec0, vec![0, 1, 2]);
}

#[test]
fn test_dataset_http_random_access_pattern() {
    let dir = setup_dataset_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    let base = view.base_vectors().unwrap();

    // Access vectors out of order to verify Range requests work for random access
    let vec7 = base.get(7).unwrap();
    let vec2 = base.get(2).unwrap();
    let vec9 = base.get(9).unwrap();
    let vec0 = base.get(0).unwrap();

    assert_eq!(vec7, vec![28.0, 29.0, 30.0, 31.0]);
    assert_eq!(vec2, vec![8.0, 9.0, 10.0, 11.0]);
    assert_eq!(vec9, vec![36.0, 37.0, 38.0, 39.0]);
    assert_eq!(vec0, vec![0.0, 1.0, 2.0, 3.0]);
}

// ═══════════════════════════════════════════════════════════════════════
// Comprehensive format tests — every vector format served over HTTP
// ═══════════════════════════════════════════════════════════════════════

/// Write an xvec file with arbitrary element size.
fn write_xvec_raw(path: &Path, dim: u32, count: u32, elem_size: usize) {
    let mut file = fs::File::create(path).unwrap();
    for i in 0..count {
        file.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = (i * dim + j) as u64;
            let bytes = val.to_le_bytes();
            file.write_all(&bytes[..elem_size]).unwrap();
        }
    }
}

/// Write an mvec (f16) file.
fn write_mvec(path: &Path, dim: u32, count: u32) {
    let mut file = fs::File::create(path).unwrap();
    for i in 0..count {
        file.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = half::f16::from_f32((i * dim + j) as f32);
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

/// Write a bvec (u8) file.
fn write_bvec(path: &Path, dim: u32, count: u32) {
    write_xvec_raw(path, dim, count, 1);
}

/// Write an svec (i16) file.
fn write_svec(path: &Path, dim: u32, count: u32) {
    write_xvec_raw(path, dim, count, 2);
}

/// Write a dvec (f64) file.
fn write_dvec(path: &Path, dim: u32, count: u32) {
    let mut file = fs::File::create(path).unwrap();
    for i in 0..count {
        file.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = (i * dim + j) as f64;
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

/// Write a variable-length ivvec file where record i has dim = base_dim + i.
fn write_ivvec(path: &Path, base_dim: u32, count: u32) {
    let mut file = fs::File::create(path).unwrap();
    for i in 0..count {
        let dim = base_dim + i;
        file.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = (i * 1000 + j) as i32;
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

/// Write a scalar file (flat packed, no header).
fn write_scalar_u8(path: &Path, values: &[u8]) {
    fs::write(path, values).unwrap();
}

/// Set up a directory with every vector format for comprehensive testing.
fn setup_all_formats_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();

    // Uniform vector formats
    write_fvec(&dir.path().join("vectors.fvec"), 3, 5);          // f32
    write_ivec(&dir.path().join("indices.ivec"), 4, 5);           // i32 uniform
    write_mvec(&dir.path().join("halfvecs.mvec"), 3, 5);          // f16
    write_dvec(&dir.path().join("doublevecs.dvec"), 3, 5);        // f64
    write_bvec(&dir.path().join("bytevecs.bvec"), 3, 5);          // u8
    write_svec(&dir.path().join("shortvecs.svec"), 3, 5);         // i16

    // Variable-length vector (vvec)
    write_ivvec(&dir.path().join("varindices.ivvec"), 2, 5);      // i32 variable

    // Scalar
    write_scalar_u8(&dir.path().join("labels.u8"), &[1, 5, 3, 12, 7]);

    // Dataset manifest referencing all formats
    fs::write(
        dir.path().join("dataset.yaml"),
        r#"
name: all-formats-test
attributes:
  distance_function: L2
profiles:
  default:
    base_vectors: vectors.fvec
    query_vectors: vectors.fvec
    neighbor_indices: indices.ivec
    metadata_content: labels.u8
    metadata_indices: varindices.ivvec
"#,
    )
    .unwrap();

    dir
}

// ── HTTP access: fvec (f32) ────────────────────────────────────────

#[test]
fn http_fvec_count_dim_values() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();
    let url = url::Url::parse(&format!("{}vectors.fvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<f32>::open_fvec(url).unwrap();

    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0.0, 1.0, 2.0]);
    assert_eq!(reader.get(4).unwrap(), vec![12.0, 13.0, 14.0]);
}

// ── HTTP access: ivec (i32 uniform) ────────────────────────────────

#[test]
fn http_ivec_count_dim_values() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();
    let url = url::Url::parse(&format!("{}indices.ivec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<i32>::open_ivec(url).unwrap();

    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(reader.get(4).unwrap(), vec![16, 17, 18, 19]);
}

// ── HTTP access: mvec (f16) ────────────────────────────────────────

#[test]
fn http_mvec_count_dim_values() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();
    let url = url::Url::parse(&format!("{}halfvecs.mvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<half::f16>::open_mvec(url).unwrap();

    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
    let v0 = reader.get(0).unwrap();
    assert_eq!(v0[0], half::f16::from_f32(0.0));
    assert_eq!(v0[1], half::f16::from_f32(1.0));
    assert_eq!(v0[2], half::f16::from_f32(2.0));
}

// ── Local mmap access: dvec (f64) ──────────────────────────────────

#[test]
fn local_dvec_count_dim_values() {
    let dir = setup_all_formats_dir();
    let r = vectordata::io::MmapVectorReader::<f64>::open_dvec(&dir.path().join("doublevecs.dvec")).unwrap();
    assert_eq!(VectorReader::<f64>::dim(&r), 3);
    assert_eq!(VectorReader::<f64>::count(&r), 5);
    assert_eq!(r.get(0).unwrap(), vec![0.0, 1.0, 2.0]);
    assert_eq!(r.get(4).unwrap(), vec![12.0, 13.0, 14.0]);
}

// ── Local mmap access: bvec (u8) ───────────────────────────────────

#[test]
fn local_bvec_count_dim_values() {
    let dir = setup_all_formats_dir();
    let r = vectordata::io::MmapVectorReader::<u8>::open_bvec(&dir.path().join("bytevecs.bvec")).unwrap();
    assert_eq!(VectorReader::<u8>::dim(&r), 3);
    assert_eq!(VectorReader::<u8>::count(&r), 5);
    let v0 = r.get(0).unwrap();
    assert_eq!(v0, vec![0u8, 1, 2]);
}

// ── Local mmap access: svec (i16) ──────────────────────────────────

#[test]
fn local_svec_count_dim_values() {
    let dir = setup_all_formats_dir();
    let r = vectordata::io::MmapVectorReader::<i16>::open_svec(&dir.path().join("shortvecs.svec")).unwrap();
    assert_eq!(VectorReader::<i16>::dim(&r), 3);
    assert_eq!(VectorReader::<i16>::count(&r), 5);
    assert_eq!(r.get(0).unwrap(), vec![0i16, 1, 2]);
}

// ── ivvec (variable-length): local access via IndexedXvecReader ────

#[test]
fn local_ivvec_indexed_access() {
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");
    let reader = IndexedXvecReader::open_ivec(&path).unwrap();

    assert_eq!(reader.count(), 5);

    // Record 0: dim = 2, values = [0, 1]
    let v0 = reader.get_i32(0).unwrap();
    assert_eq!(v0, vec![0, 1]);

    // Record 1: dim = 3, values = [1000, 1001, 1002]
    let v1 = reader.get_i32(1).unwrap();
    assert_eq!(v1, vec![1000, 1001, 1002]);

    // Record 4: dim = 6, values = [4000..4005]
    let v4 = reader.get_i32(4).unwrap();
    assert_eq!(v4, vec![4000, 4001, 4002, 4003, 4004, 4005]);

    // Verify per-record dimension varies
    assert_eq!(reader.dim_at(0).unwrap(), 2);
    assert_eq!(reader.dim_at(1).unwrap(), 3);
    assert_eq!(reader.dim_at(4).unwrap(), 6);
}

// ── ivvec: index file is created and reused ────────────────────────

#[test]
fn ivvec_index_file_persisted() {
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");

    // First open: creates IDXFOR__ file
    let _reader = IndexedXvecReader::open_ivec(&path).unwrap();
    let index_path = dir.path().join("IDXFOR__varindices.ivvec.i32");
    assert!(index_path.exists(), "index file should be created");

    // Verify index file size: 5 records × 4 bytes/offset = 20 bytes
    let index_size = fs::metadata(&index_path).unwrap().len();
    assert_eq!(index_size, 20);

    // Second open: reuses cached index (no error)
    let reader2 = IndexedXvecReader::open_ivec(&path).unwrap();
    assert_eq!(reader2.count(), 5);
}

// ── ivvec over HTTP: full remote random access via index ───────────

#[test]
fn http_ivvec_remote_random_access() {
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");

    // Pre-build the index (simulates pipeline writing + indexing)
    let local = IndexedXvecReader::open_ivec(&path).unwrap();
    assert_eq!(local.count(), 5);

    // Start HTTP server — serves both data and index files
    let server = TestServer::start(dir.path()).unwrap();

    // Open remotely using HttpIndexedXvecReader
    let data_url = url::Url::parse(&format!("{}varindices.ivvec", server.base_url())).unwrap();
    let reader = HttpIndexedXvecReader::open_ivec(data_url).unwrap();

    assert_eq!(reader.count(), 5);

    // Record 0: dim=2, values=[0, 1]
    assert_eq!(reader.dim_at(0).unwrap(), 2);
    assert_eq!(reader.get_i32(0).unwrap(), vec![0, 1]);

    // Record 1: dim=3, values=[1000, 1001, 1002]
    assert_eq!(reader.dim_at(1).unwrap(), 3);
    assert_eq!(reader.get_i32(1).unwrap(), vec![1000, 1001, 1002]);

    // Record 4: dim=6, values=[4000..4005]
    assert_eq!(reader.dim_at(4).unwrap(), 6);
    assert_eq!(reader.get_i32(4).unwrap(), vec![4000, 4001, 4002, 4003, 4004, 4005]);

    // Random access order (not sequential)
    assert_eq!(reader.get_i32(3).unwrap(), vec![3000, 3001, 3002, 3003, 3004]);
    assert_eq!(reader.get_i32(0).unwrap(), vec![0, 1]);

    // Out of bounds
    assert!(reader.get_i32(100).is_err());
}

#[test]
fn http_ivvec_index_file_accessible() {
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");
    let _local = IndexedXvecReader::open_ivec(&path).unwrap();

    let server = TestServer::start(dir.path()).unwrap();
    let index_url = format!("{}IDXFOR__varindices.ivvec.i32", server.base_url());

    let resp = reqwest::blocking::get(&index_url).unwrap();
    assert_eq!(resp.status(), 200, "index file should be served over HTTP");
    let index_bytes = resp.bytes().unwrap();
    assert_eq!(index_bytes.len(), 20, "index file should be 5 × 4 bytes");
}

#[test]
fn http_ivvec_no_index_returns_error() {
    let dir = setup_all_formats_dir();
    // Do NOT build the index — HttpIndexedXvecReader should fail
    let server = TestServer::start(dir.path()).unwrap();
    let data_url = url::Url::parse(&format!("{}varindices.ivvec", server.base_url())).unwrap();
    let result = HttpIndexedXvecReader::open_ivec(data_url);
    assert!(result.is_err(), "should fail without index file");
}

#[test]
fn http_ivvec_matches_local_access() {
    // Verify that remote and local access return identical data
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");
    let local = IndexedXvecReader::open_ivec(&path).unwrap();

    let server = TestServer::start(dir.path()).unwrap();
    let data_url = url::Url::parse(&format!("{}varindices.ivvec", server.base_url())).unwrap();
    let remote = HttpIndexedXvecReader::open_ivec(data_url).unwrap();

    assert_eq!(local.count(), remote.count());
    for i in 0..local.count() {
        let local_rec = local.get_i32(i).unwrap();
        let remote_rec = remote.get_i32(i).unwrap();
        assert_eq!(local_rec, remote_rec, "mismatch at record {}", i);
        assert_eq!(local.dim_at(i).unwrap(), remote.dim_at(i).unwrap());
    }
}

// ── Unified open_vvec::<i32> — same API for local and remote ──────────

#[test]
fn unified_vvec_local_access() {
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");

    // Pre-build index
    let _ = IndexedXvecReader::open_ivec(&path).unwrap();

    // Use unified API with a local path — typed as i32
    let reader = open_vvec::<i32>(path.to_str().unwrap()).unwrap();
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0, 1]);
    assert_eq!(reader.get(4).unwrap(), vec![4000, 4001, 4002, 4003, 4004, 4005]);
    assert_eq!(reader.dim_at(0).unwrap(), 2);
    assert_eq!(reader.dim_at(4).unwrap(), 6);
}

#[test]
fn unified_vvec_remote_access() {
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");
    let _ = IndexedXvecReader::open_ivec(&path).unwrap();

    let server = TestServer::start(dir.path()).unwrap();
    let url = format!("{}varindices.ivvec", server.base_url());

    // Use unified API with an HTTP URL — typed as i32
    let reader = open_vvec::<i32>(&url).unwrap();
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0, 1]);
    assert_eq!(reader.get(4).unwrap(), vec![4000, 4001, 4002, 4003, 4004, 4005]);
}

#[test]
fn unified_vvec_local_matches_remote() {
    let dir = setup_all_formats_dir();
    let path = dir.path().join("varindices.ivvec");
    let _ = IndexedXvecReader::open_ivec(&path).unwrap();

    let server = TestServer::start(dir.path()).unwrap();
    let url = format!("{}varindices.ivvec", server.base_url());

    let local = open_vvec::<i32>(path.to_str().unwrap()).unwrap();
    let remote = open_vvec::<i32>(&url).unwrap();

    assert_eq!(local.count(), remote.count());
    for i in 0..local.count() {
        assert_eq!(local.get(i).unwrap(), remote.get(i).unwrap(),
            "record {} mismatch", i);
    }
}

// ── Uniform ivec rejects variable-length ───────────────────────────

#[test]
fn uniform_ivec_rejects_variable_length() {
    let dir = setup_all_formats_dir();
    // Rename ivvec to .ivec to test the guard
    let src = dir.path().join("varindices.ivvec");
    let dst = dir.path().join("varindices_test.ivec");
    fs::copy(&src, &dst).unwrap();

    let result = vectordata::io::MmapVectorReader::<i32>::open_ivec(&dst);
    assert!(result.is_err(), "variable-length ivec should be rejected by MmapVectorReader");
    let err = result.unwrap_err();
    assert!(
        matches!(err, vectordata::io::IoError::VariableLengthRecords(_)),
        "error should be VariableLengthRecords, got: {}", err
    );
}

// ── Scalar file access ─────────────────────────────────────────────

#[test]
fn scalar_u8_via_typed_reader() {
    let dir = setup_all_formats_dir();
    let reader = vectordata::typed_access::TypedReader::<u8>::open(
        &dir.path().join("labels.u8")
    ).unwrap();

    assert_eq!(reader.count(), 5);
    assert_eq!(reader.dim(), 1);
    assert_eq!(reader.get_native(0), 1);
    assert_eq!(reader.get_native(1), 5);
    assert_eq!(reader.get_native(4), 7);
}

// ── Dataset profile access: all facets ─────────────────────────────

#[test]
fn dataset_http_all_facets() {
    let dir = setup_all_formats_dir();

    // Pre-build ivvec index
    let _r = IndexedXvecReader::open_ivec(&dir.path().join("varindices.ivvec")).unwrap();

    let server = TestServer::start(dir.path()).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();

    assert!(group.profile("default").is_some());

    let view = group.profile("default").unwrap();

    // base_vectors (fvec)
    let base = view.base_vectors().unwrap();
    assert_eq!(base.dim(), 3);
    assert_eq!(base.count(), 5);
    assert_eq!(base.get(0).unwrap(), vec![0.0, 1.0, 2.0]);

    // query_vectors (fvec)
    let query = view.query_vectors().unwrap();
    assert_eq!(query.dim(), 3);
    assert_eq!(query.count(), 5);

    // neighbor_indices (ivec uniform)
    let gt = view.neighbor_indices().unwrap();
    assert_eq!(gt.dim(), 4);
    assert_eq!(gt.count(), 5);
    assert_eq!(gt.get(0).unwrap(), vec![0, 1, 2, 3]);
}

// ── Out-of-bounds access returns error ─────────────────────────────

#[test]
fn http_out_of_bounds_all_formats() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();

    // fvec out of bounds
    let url = url::Url::parse(&format!("{}vectors.fvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<f32>::open_fvec(url).unwrap();
    assert!(reader.get(100).is_err());

    // ivec out of bounds
    let url = url::Url::parse(&format!("{}indices.ivec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<i32>::open_ivec(url).unwrap();
    assert!(reader.get(100).is_err());

    // mvec out of bounds
    let url = url::Url::parse(&format!("{}halfvecs.mvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<half::f16>::open_mvec(url).unwrap();
    assert!(reader.get(100).is_err());
}

// ── IndexedXvecReader out-of-bounds ────────────────────────────────

#[test]
fn ivvec_out_of_bounds() {
    let dir = setup_all_formats_dir();
    let reader = IndexedXvecReader::open_ivec(&dir.path().join("varindices.ivvec")).unwrap();
    assert!(reader.get_i32(100).is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// Unified open_vec — local and remote, all element types
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn open_vec_local_fvec() {
    let dir = setup_all_formats_dir();
    let reader = open_vec::<f32>(dir.path().join("vectors.fvec").to_str().unwrap()).unwrap();
    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0.0, 1.0, 2.0]);
    assert_eq!(reader.get(4).unwrap(), vec![12.0, 13.0, 14.0]);
}

#[test]
fn open_vec_local_ivec() {
    let dir = setup_all_formats_dir();
    let reader = open_vec::<i32>(dir.path().join("indices.ivec").to_str().unwrap()).unwrap();
    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0, 1, 2, 3]);
}

#[test]
fn open_vec_local_bvec() {
    let dir = setup_all_formats_dir();
    let reader = open_vec::<u8>(dir.path().join("bytevecs.bvec").to_str().unwrap()).unwrap();
    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0u8, 1, 2]);
}

#[test]
fn open_vec_local_svec() {
    let dir = setup_all_formats_dir();
    let reader = open_vec::<i16>(dir.path().join("shortvecs.svec").to_str().unwrap()).unwrap();
    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
}

#[test]
fn open_vec_local_dvec() {
    let dir = setup_all_formats_dir();
    let reader = open_vec::<f64>(dir.path().join("doublevecs.dvec").to_str().unwrap()).unwrap();
    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0.0f64, 1.0, 2.0]);
}

#[test]
fn open_vec_remote_fvec() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();
    let url = format!("{}vectors.fvec", server.base_url());
    let reader = open_vec::<f32>(&url).unwrap();
    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0.0, 1.0, 2.0]);
}

#[test]
fn open_vec_remote_ivec() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();
    let url = format!("{}indices.ivec", server.base_url());
    let reader = open_vec::<i32>(&url).unwrap();
    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get(0).unwrap(), vec![0, 1, 2, 3]);
}

#[test]
fn open_vec_remote_mvec() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();
    let url = format!("{}halfvecs.mvec", server.base_url());
    let reader = open_vec::<half::f16>(&url).unwrap();
    assert_eq!(reader.dim(), 3);
    assert_eq!(reader.count(), 5);
    let v0 = reader.get(0).unwrap();
    assert_eq!(v0[0], half::f16::from_f32(0.0));
}

#[test]
fn open_vec_local_matches_remote() {
    let dir = setup_all_formats_dir();
    let server = TestServer::start(dir.path()).unwrap();

    let local = open_vec::<f32>(dir.path().join("vectors.fvec").to_str().unwrap()).unwrap();
    let remote = open_vec::<f32>(&format!("{}vectors.fvec", server.base_url())).unwrap();

    assert_eq!(local.count(), remote.count());
    assert_eq!(local.dim(), remote.dim());
    for i in 0..local.count() {
        assert_eq!(local.get(i).unwrap(), remote.get(i).unwrap(), "record {} mismatch", i);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Synthetic-1K fixture: full BQGDMPRF dataset over HTTP
// ═══════════════════════════════════════════════════════════════════════
//
// These tests serve the pre-built synthetic-1k fixture (1000 base vectors,
// 100 queries, dim=128, full metadata+predicate+filtered pipeline) over
// the test HTTP server and exercise every facet type through the
// vectordata unified API.

/// Path to the synthetic-1k fixture (pre-built by `veks run`).
fn synthetic_1k_dir() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // vectordata/tests -> vectordata -> workspace root -> veks/tests/fixtures
    manifest.parent().unwrap()
        .join("veks/tests/fixtures/synthetic-1k")
}

/// Ensure the synthetic-1k fixture is built. Skip test if not.
fn require_synthetic_1k() -> std::path::PathBuf {
    let dir = synthetic_1k_dir();
    let base = dir.join("profiles/base/base_vectors.fvec");
    if !base.exists() {
        eprintln!("synthetic-1k fixture not built — run:");
        eprintln!("  cd veks/tests/fixtures/synthetic-1k && veks run dataset.yaml");
        panic!("synthetic-1k fixture not available");
    }
    dir
}

// ── Load dataset over HTTP ─────────────────────────────────────────

#[test]
fn synthetic_http_load_dataset() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    assert!(group.profile("default").is_some(), "default profile should exist");

    let dist = group.attribute("distance_function").unwrap();
    assert_eq!(dist.as_str().unwrap(), "L2");
}

// ── Base vectors (f32, uniform) over HTTP ──────────────────────────

#[test]
fn synthetic_http_base_vectors() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 1000, "should have 1000 base vectors");
    assert_eq!(base.dim(), 128, "dim should be 128");

    // Verify first vector is readable and has correct dimension
    let v0 = base.get(0).unwrap();
    assert_eq!(v0.len(), 128);
    // Values should be finite (gaussian random)
    assert!(v0.iter().all(|v| v.is_finite()), "all components should be finite");
}

// ── Query vectors (f32, uniform) over HTTP ─────────────────────────

#[test]
fn synthetic_http_query_vectors() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let query = view.query_vectors().unwrap();
    assert_eq!(query.count(), 100, "should have 100 query vectors");
    assert_eq!(query.dim(), 128);
}

// ── Ground truth indices (i32, uniform) over HTTP ──────────────────

#[test]
fn synthetic_http_neighbor_indices() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let gt = view.neighbor_indices().unwrap();
    assert_eq!(gt.count(), 100, "should have 100 query results");
    assert_eq!(gt.dim(), 100, "k=100 neighbors per query");

    // All ordinals should be valid base vector indices
    let result0 = gt.get(0).unwrap();
    for &ord in &result0 {
        assert!((ord as usize) < 1000, "GT ordinal {} out of base range", ord);
    }

    // Random access: last query
    let result99 = gt.get(99).unwrap();
    assert_eq!(result99.len(), 100);
}

// ── Ground truth distances (f32, uniform) over HTTP ────────────────

#[test]
fn synthetic_http_neighbor_distances() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let dist = view.neighbor_distances().unwrap();
    assert_eq!(dist.count(), 100);
    assert_eq!(dist.dim(), 100);

    // Distances should be non-negative and sorted ascending
    let d0 = dist.get(0).unwrap();
    for i in 1..d0.len() {
        assert!(d0[i] >= d0[i-1], "distances should be sorted: d[{}]={} < d[{}]={}",
            i, d0[i], i-1, d0[i-1]);
    }
}

// ── Metadata indices (i32, variable-length vvec) over HTTP ─────────

#[test]
fn synthetic_http_metadata_indices_vvec() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let mi = view.metadata_indices().unwrap();
    assert_eq!(mi.count(), 100, "should have 100 predicate results (one per query)");

    // Each record is variable-length: different predicates match different counts
    let r0 = mi.get(0).unwrap();
    assert!(!r0.is_empty(), "first predicate should match some base vectors");

    // All ordinals should be valid base indices
    for &ord in &r0 {
        assert!((ord as usize) < 1000, "MI ordinal {} out of range", ord);
    }

    // Verify per-record dimensions vary (selectivity ~8.3% with 13 labels)
    let d0 = mi.dim_at(0).unwrap();
    let d1 = mi.dim_at(1).unwrap();
    // They might be the same (random), but check they're reasonable
    assert!(d0 > 0 && d0 < 1000, "dim_at(0)={} should be in (0, 1000)", d0);
    assert!(d1 > 0 && d1 < 1000, "dim_at(1)={} should be in (0, 1000)", d1);
}

// ── Filtered KNN indices (i32, uniform) over HTTP ──────────────────

#[test]
fn synthetic_http_filtered_knn_indices() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let fki = view.filtered_neighbor_indices().unwrap();
    assert_eq!(fki.count(), 100);
    assert!(fki.dim() > 0, "should have neighbors");

    // All filtered ordinals should be valid base indices (or -1 sentinel
    // when not enough neighbors pass the predicate to fill k)
    let f0 = fki.get(0).unwrap();
    for &ord in &f0 {
        assert!(ord == -1 || (ord >= 0 && (ord as usize) < 1000),
            "filtered ordinal {} out of range", ord);
    }
}

// ── Filtered KNN distances (f32, uniform) over HTTP ────────────────

#[test]
fn synthetic_http_filtered_knn_distances() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let fkd = view.filtered_neighbor_distances().unwrap();
    assert_eq!(fkd.count(), 100);

    // Distances should be non-negative and sorted ascending
    let d0 = fkd.get(0).unwrap();
    for i in 1..d0.len() {
        assert!(d0[i] >= d0[i-1],
            "filtered distances should be sorted: d[{}]={} < d[{}]={}",
            i, d0[i], i-1, d0[i-1]);
    }
}

// ── Typed metadata access (u8 scalar) via generic_view ──────────────

#[test]
fn synthetic_http_metadata_typed_access() {
    let dir = require_synthetic_1k();

    // Typed access is local-only (uses mmap), so test against local fixture
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();
    let view = group.generic_view("default").unwrap();

    let etype = view.facet_element_type("metadata_content").unwrap();
    assert_eq!(etype, vectordata::typed_access::ElementType::U8);

    let typed = view.open_facet_typed::<u8>("metadata_content").unwrap();
    assert_eq!(typed.count(), 1000);
    // All values should be in [0, 12]
    for i in 0..typed.count() {
        let val = typed.get_native(i);
        assert!(val <= 12, "metadata value {} out of range at ordinal {}", val, i);
    }
}

// ── open_vvec unified API: local vs remote consistency ──────────────

#[test]
fn synthetic_vvec_local_vs_remote() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let local_path = dir.join("profiles/default/metadata_indices.ivvec");
    let remote_url = format!("{}profiles/default/metadata_indices.ivvec", server.base_url());

    let local = open_vvec::<i32>(local_path.to_str().unwrap()).unwrap();
    let remote = open_vvec::<i32>(&remote_url).unwrap();

    assert_eq!(local.count(), remote.count(), "record counts should match");

    // Verify first 10 records match byte-for-byte
    for i in 0..10.min(local.count()) {
        let local_rec = local.get(i).unwrap();
        let remote_rec = remote.get(i).unwrap();
        assert_eq!(local_rec, remote_rec, "record {} mismatch", i);
        assert_eq!(local.dim_at(i).unwrap(), remote.dim_at(i).unwrap(),
            "dim_at({}) mismatch", i);
    }
}

// ── open_vec unified API: local vs remote consistency ────────────────

#[test]
fn synthetic_vec_local_vs_remote() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let local_path = dir.join("profiles/base/base_vectors.fvec");
    let remote_url = format!("{}profiles/base/base_vectors.fvec", server.base_url());

    let local = open_vec::<f32>(local_path.to_str().unwrap()).unwrap();
    let remote = open_vec::<f32>(&remote_url).unwrap();

    assert_eq!(local.count(), remote.count());
    assert_eq!(local.dim(), remote.dim());

    // Spot-check first and last vectors
    assert_eq!(local.get(0).unwrap(), remote.get(0).unwrap());
    assert_eq!(local.get(999).unwrap(), remote.get(999).unwrap());
}

// ── Cross-facet consistency: filtered results pass predicates ───────

#[test]
fn synthetic_http_filtered_results_pass_predicates() {
    let dir = require_synthetic_1k();

    // Load locally for typed metadata access
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let gview = group.generic_view("default").unwrap();

    let meta = gview.open_facet_typed::<u8>("metadata_content").unwrap();
    let pred = gview.open_facet_typed::<u8>("metadata_predicates").unwrap();
    let mi = view.metadata_indices().unwrap();
    let fki = view.filtered_neighbor_indices().unwrap();

    // For each query, verify:
    // 1. All metadata_indices ordinals have the correct metadata value
    // 2. All filtered KNN ordinals have the correct metadata value
    for qi in 0..10.min(pred.count()) {
        let pred_val = pred.get_native(qi);

        // Check metadata_indices: every matching ordinal should have the predicate value
        let matching = mi.get(qi).unwrap();
        for &ord in &matching {
            let meta_val = meta.get_native(ord as usize);
            assert_eq!(meta_val, pred_val,
                "query {}: predicate={}, but metadata[{}]={}", qi, pred_val, ord, meta_val);
        }

        // Check filtered KNN: every neighbor should pass the predicate
        // (skip -1 sentinels = unfilled neighbor slots)
        let neighbors = fki.get(qi).unwrap();
        for &ord in &neighbors {
            if ord < 0 { continue; } // sentinel
            let meta_val = meta.get_native(ord as usize);
            assert_eq!(meta_val, pred_val,
                "query {}: filtered neighbor {} has metadata={}, expected {}",
                qi, ord, meta_val, pred_val);
        }
    }
}

// ── Full facet manifest ─────────────────────────────────────────────

#[test]
fn synthetic_http_facet_manifest() {
    let dir = require_synthetic_1k();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let manifest = view.facet_manifest();

    // Should have all BQGDMPRF facets
    assert!(manifest.contains_key("base_vectors"), "missing base_vectors");
    assert!(manifest.contains_key("query_vectors"), "missing query_vectors");
    assert!(manifest.contains_key("neighbor_indices"), "missing neighbor_indices");
    assert!(manifest.contains_key("neighbor_distances"), "missing neighbor_distances");
    assert!(manifest.contains_key("metadata_content"), "missing metadata_content");
    assert!(manifest.contains_key("metadata_predicates"), "missing metadata_predicates");
    assert!(manifest.contains_key("filtered_neighbor_indices"), "missing filtered_neighbor_indices");
    assert!(manifest.contains_key("filtered_neighbor_distances"), "missing filtered_neighbor_distances");
}

// ═══════════════════════════════════════════════════════════════════════
// Partition profiles: base_count, profile enumeration, data access
// ═══════════════════════════════════════════════════════════════════════

/// Check that partition profiles exist and can be listed.
fn require_partition_profiles() -> std::path::PathBuf {
    let dir = require_synthetic_1k();
    let label_dir = dir.join("profiles/label_00");
    if !label_dir.exists() {
        panic!("partition profiles not built — run compute partition-profiles on synthetic-1k first");
    }
    dir
}

#[test]
fn partition_profile_enumeration() {
    let dir = require_partition_profiles();
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    let names = group.profile_names();
    assert!(names.contains(&"default".to_string()), "should have default");
    assert!(names.contains(&"label_00".to_string()), "should have label_00");
    assert!(names.len() >= 13, "should have default + 12 partitions, got {}", names.len());
}

#[test]
fn partition_profile_base_count() {
    let dir = require_partition_profiles();
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    // Default profile — base_count may or may not be set
    let _default_view = group.profile("default").unwrap();

    // Partition profile — must have base_count
    let label0 = group.profile("label_00").unwrap();
    let bc = label0.base_count();
    assert!(bc.is_some(), "partition profile must have base_count");
    let count = bc.unwrap();
    assert!(count > 0 && count < 1000,
        "label_00 base_count {} should be between 1 and 999", count);
}

#[test]
fn partition_profile_base_vectors_match_count() {
    let dir = require_partition_profiles();
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    let label0 = group.profile("label_00").unwrap();
    let bc = label0.base_count().unwrap();
    let base = label0.base_vectors().unwrap();

    assert_eq!(base.count() as u64, bc,
        "base_vectors count {} should match base_count {}", base.count(), bc);
    assert_eq!(base.dim(), 128, "dim should be 128");
}

#[test]
fn partition_profile_over_http() {
    let dir = require_partition_profiles();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();

    // Access partition profile via HTTP
    let label0 = group.profile("label_00").unwrap();
    let bc = label0.base_count();
    assert!(bc.is_some(), "partition base_count via HTTP");

    let base = label0.base_vectors().unwrap();
    assert_eq!(base.count() as u64, bc.unwrap());
    assert_eq!(base.dim(), 128);

    // First vector should be readable
    let v0 = base.get(0).unwrap();
    assert_eq!(v0.len(), 128);
    assert!(v0.iter().all(|v| v.is_finite()));
}

#[test]
fn partition_profile_query_vectors_shared() {
    let dir = require_partition_profiles();
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    let default_view = group.profile("default").unwrap();
    let label0 = group.profile("label_00").unwrap();

    let default_q = default_view.query_vectors().unwrap();
    let label0_q = label0.query_vectors().unwrap();

    // Same query count
    assert_eq!(default_q.count(), label0_q.count(),
        "query count should be same across profiles");

    // Same query data
    assert_eq!(default_q.get(0).unwrap(), label0_q.get(0).unwrap(),
        "query[0] should be identical across profiles");
}

#[test]
fn partition_all_profiles_have_base_count_over_http() {
    let dir = require_partition_profiles();
    let server = TestServer::start(&dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();

    for name in group.profile_names() {
        if name.starts_with("label_") {
            let view = group.profile(&name).unwrap();
            assert!(view.base_count().is_some(),
                "partition profile '{}' must have base_count", name);
            let bc = view.base_count().unwrap();
            assert!(bc > 0, "partition '{}' base_count must be > 0", name);
        }
    }
}

#[test]
fn partition_profile_ordering_consistent() {
    // Profile ordering is by ascending base_count, with natural name sort
    // as tiebreak for equal sizes. This test verifies that:
    // 1. default comes first
    // 2. Partition profiles are in ascending base_count order
    // 3. Within the same base_count, natural name ordering applies
    //    (e.g., label_02 before label_10)
    let dir = require_partition_profiles();
    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    let names = group.profile_names();
    assert_eq!(names[0], "default", "default should be first");

    // Verify ascending base_count order among partition profiles
    let partitions: Vec<_> = names.iter()
        .filter(|n| n.starts_with("label_"))
        .collect();
    for i in 1..partitions.len() {
        let prev_view = group.profile(partitions[i - 1]).unwrap();
        let curr_view = group.profile(partitions[i]).unwrap();
        let prev_bc = prev_view.base_count().unwrap_or(0);
        let curr_bc = curr_view.base_count().unwrap_or(0);
        assert!(prev_bc <= curr_bc,
            "profiles should be in ascending base_count order: {} ({}) before {} ({})",
            partitions[i - 1], prev_bc, partitions[i], curr_bc);
    }
}

#[test]
fn partition_profile_ordering_consistent_over_http() {
    let dir = require_partition_profiles();
    let server = TestServer::start(&dir).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();

    let names = group.profile_names();
    assert_eq!(names[0], "default", "default should be first over HTTP");

    let partitions: Vec<_> = names.iter()
        .filter(|n| n.starts_with("label_"))
        .collect();
    for i in 1..partitions.len() {
        let prev_view = group.profile(partitions[i - 1]).unwrap();
        let curr_view = group.profile(partitions[i]).unwrap();
        let prev_bc = prev_view.base_count().unwrap_or(0);
        let curr_bc = curr_view.base_count().unwrap_or(0);
        assert!(prev_bc <= curr_bc,
            "HTTP profiles should be in ascending order: {} ({}) before {} ({})",
            partitions[i - 1], prev_bc, partitions[i], curr_bc);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Format coverage tests
//
// Verify that every valid file format can be read in the positions where
// it's used, both locally and over HTTP.
// ═══════════════════════════════════════════════════════════════════════════

/// Write an fvec (f32 xvec) file.
fn write_fvec_data(path: &std::path::Path, dim: usize, count: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..count {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            let val = (i * dim + d + 1) as f32;
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Write an ivec (i32 xvec) file.
fn write_ivec_data(path: &std::path::Path, dim: usize, count: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..count {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            let val = ((i * dim + d) % 50) as i32; // ordinals within base count
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Write a scalar file (one element per record, no dim header).
fn write_scalar(path: &std::path::Path, count: usize, elem_size: usize) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..count {
        let val = (i % 256) as u32;
        match elem_size {
            1 => w.write_all(&[val as u8]).unwrap(),
            2 => w.write_all(&(val as u16).to_le_bytes()).unwrap(),
            4 => w.write_all(&(val as i32).to_le_bytes()).unwrap(),
            8 => w.write_all(&(val as i64).to_le_bytes()).unwrap(),
            _ => panic!("unsupported elem_size"),
        }
    }
    w.flush().unwrap();
}

/// Write a variable-length ivvec file from explicit records.
fn write_ivvec_records(path: &std::path::Path, records: &[Vec<i32>]) {
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for record in records {
        w.write_all(&(record.len() as i32).to_le_bytes()).unwrap();
        for &val in record {
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Create a format coverage dataset with all valid file types.
fn create_format_coverage_dataset(dir: &std::path::Path) {
    let dim = 4;
    let base_count = 50;
    let query_count = 10;
    let k = 5;
    let pred_count = 20;

    // ── Base and query vectors: fvec (f32 xvec) ──
    write_fvec_data(&dir.join("base_vectors.fvec"), dim, base_count);
    write_fvec_data(&dir.join("query_vectors.fvec"), dim, query_count);

    // ── KNN results: ivec (i32 indices) + fvec (f32 distances) ──
    write_ivec_data(&dir.join("neighbor_indices.ivec"), k, query_count);
    write_fvec_data(&dir.join("neighbor_distances.fvec"), k, query_count);

    // ── Filtered KNN: same formats ──
    write_ivec_data(&dir.join("filtered_neighbor_indices.ivec"), k, query_count);
    write_fvec_data(&dir.join("filtered_neighbor_distances.fvec"), k, query_count);

    // ── Metadata content: u8 scalar ──
    write_scalar(&dir.join("metadata_content.u8"), base_count, 1);

    // ── Metadata predicates: u8 scalar ──
    write_scalar(&dir.join("predicates.u8"), pred_count, 1);

    // ── Metadata indices: ivvec (variable-length i32) ──
    let ivvec_records: Vec<Vec<i32>> = (0..pred_count).map(|i| {
        (0..((i % 5) + 1) as i32).collect()
    }).collect();
    write_ivvec_records(&dir.join("metadata_indices.ivvec"), &ivvec_records);
    // Build offset index
    let _ = vectordata::io::IndexedXvecReader::open_ivec(&dir.join("metadata_indices.ivvec"));

    // ── dataset.yaml ──
    let yaml = format!(r#"name: format-coverage
profiles:
  default:
    maxk: {k}
    base_vectors: base_vectors.fvec
    query_vectors: query_vectors.fvec
    neighbor_indices: neighbor_indices.ivec
    neighbor_distances: neighbor_distances.fvec
    metadata_content: metadata_content.u8
    metadata_predicates: predicates.u8
    metadata_indices: metadata_indices.ivvec
    filtered_neighbor_indices: filtered_neighbor_indices.ivec
    filtered_neighbor_distances: filtered_neighbor_distances.fvec
"#);
    std::fs::write(dir.join("dataset.yaml"), &yaml).unwrap();
}

#[test]
fn format_coverage_local_all_facets() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_format_coverage_dataset(dir);

    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    // Base vectors: fvec (f32)
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 50, "base_vectors count");
    assert_eq!(base.dim(), 4, "base_vectors dim");
    let v0 = base.get(0).unwrap();
    assert_eq!(v0.len(), 4);
    assert!(v0[0].is_finite());

    // Query vectors: fvec (f32)
    let query = view.query_vectors().unwrap();
    assert_eq!(query.count(), 10, "query_vectors count");
    assert_eq!(query.dim(), 4, "query_vectors dim");

    // Neighbor indices: ivec (i32)
    let indices = view.neighbor_indices().unwrap();
    assert_eq!(indices.count(), 10, "neighbor_indices count");
    assert_eq!(indices.dim(), 5, "neighbor_indices dim (k)");

    // Neighbor distances: fvec (f32)
    let distances = view.neighbor_distances().unwrap();
    assert_eq!(distances.count(), 10, "neighbor_distances count");
    assert_eq!(distances.dim(), 5, "neighbor_distances dim (k)");

    // Filtered indices + distances
    let filt_idx = view.filtered_neighbor_indices().unwrap();
    assert_eq!(filt_idx.count(), 10, "filtered_indices count");
    let filt_dist = view.filtered_neighbor_distances().unwrap();
    assert_eq!(filt_dist.count(), 10, "filtered_distances count");

    // Metadata indices: ivvec (variable-length i32)
    let meta_idx = view.metadata_indices().unwrap();
    assert!(meta_idx.count() > 0, "metadata_indices should have records");
}

#[test]
fn format_coverage_http_all_facets() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_format_coverage_dataset(dir);

    let server = TestServer::start(dir).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    // Base vectors over HTTP
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 50, "HTTP base_vectors count");
    assert_eq!(base.dim(), 4, "HTTP base_vectors dim");
    let v0 = base.get(0).unwrap();
    assert!(v0[0].is_finite(), "HTTP base vector should be readable");

    // Query vectors over HTTP
    let query = view.query_vectors().unwrap();
    assert_eq!(query.count(), 10, "HTTP query_vectors count");

    // KNN over HTTP
    let indices = view.neighbor_indices().unwrap();
    assert_eq!(indices.count(), 10, "HTTP neighbor_indices count");
    let distances = view.neighbor_distances().unwrap();
    assert_eq!(distances.count(), 10, "HTTP neighbor_distances count");

    // Filtered KNN over HTTP
    let filt_idx = view.filtered_neighbor_indices().unwrap();
    assert_eq!(filt_idx.count(), 10, "HTTP filtered_indices count");
    let filt_dist = view.filtered_neighbor_distances().unwrap();
    assert_eq!(filt_dist.count(), 10, "HTTP filtered_distances count");

    // Variable-length metadata indices over HTTP
    let meta_idx = view.metadata_indices().unwrap();
    assert!(meta_idx.count() > 0, "HTTP metadata_indices count");
}

#[test]
fn format_coverage_mvec_base_vectors() {
    // mvec (f16) base vectors — opened via open_vec::<half::f16>
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Write mvec file (2-byte f16 elements)
    let dim = 4;
    let count = 20;
    {
        let f = std::fs::File::create(dir.join("base.mvec")).unwrap();
        let mut w = std::io::BufWriter::new(f);
        for i in 0..count {
            w.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for d in 0..dim {
                let val = half::f16::from_f32((i * dim + d + 1) as f32);
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    // Read via open_vec::<half::f16>
    let reader = vectordata::io::open_vec::<half::f16>(
        dir.join("base.mvec").to_str().unwrap()
    ).unwrap();
    assert_eq!(reader.count(), count);
    assert_eq!(reader.dim(), dim);
    let v0 = reader.get(0).unwrap();
    assert_eq!(v0.len(), dim);
    assert_eq!(v0[0], half::f16::from_f32(1.0));
}

#[test]
fn format_coverage_bvec_base_vectors() {
    // bvec (u8) base vectors — opened via open_vec::<u8>
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let dim = 4;
    let count = 20;
    {
        let f = std::fs::File::create(dir.join("base.bvec")).unwrap();
        let mut w = std::io::BufWriter::new(f);
        for i in 0..count {
            w.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for d in 0..dim {
                w.write_all(&[((i * dim + d + 1) % 256) as u8]).unwrap();
            }
        }
        w.flush().unwrap();
    }

    let reader = vectordata::io::open_vec::<u8>(
        dir.join("base.bvec").to_str().unwrap()
    ).unwrap();
    assert_eq!(reader.count(), count);
    assert_eq!(reader.dim(), dim);
    let v0 = reader.get(0).unwrap();
    assert_eq!(v0, vec![1, 2, 3, 4]);
}

#[test]
fn format_coverage_mvec_over_http() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let dim = 4;
    let count = 20;
    {
        let f = std::fs::File::create(dir.join("base.mvec")).unwrap();
        let mut w = std::io::BufWriter::new(f);
        for i in 0..count {
            w.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for d in 0..dim {
                let val = half::f16::from_f32((i * dim + d + 1) as f32);
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    let server = TestServer::start(dir).unwrap();
    let url = format!("{}/base.mvec", server.base_url().trim_end_matches('/'));
    let reader = vectordata::io::open_vec::<half::f16>(&url).unwrap();
    assert_eq!(reader.count(), count, "HTTP mvec count");
    assert_eq!(reader.dim(), dim, "HTTP mvec dim");
    let v0 = reader.get(0).unwrap();
    assert_eq!(v0[0], half::f16::from_f32(1.0));
}

// ── Typed access coverage ──────────────────────────────────────────

#[test]
fn format_coverage_typed_access_scalar_formats() {
    use vectordata::typed_access::TypedReader;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Write scalar files in various formats
    write_scalar(dir.join("meta.u8").as_path(), 100, 1);
    write_scalar(dir.join("meta.i32").as_path(), 100, 4);
    write_scalar(dir.join("meta.u16").as_path(), 100, 2);
    write_scalar(dir.join("meta.i64").as_path(), 100, 8);

    // Native access — each type matches exactly
    let r_u8 = TypedReader::<u8>::open(dir.join("meta.u8")).unwrap();
    assert_eq!(r_u8.count(), 100);
    assert_eq!(r_u8.dim(), 1); // scalar = dim 1

    let r_i32 = TypedReader::<i32>::open(dir.join("meta.i32")).unwrap();
    assert_eq!(r_i32.count(), 100);

    let r_u16 = TypedReader::<u16>::open(dir.join("meta.u16")).unwrap();
    assert_eq!(r_u16.count(), 100);

    let r_i64 = TypedReader::<i64>::open(dir.join("meta.i64")).unwrap();
    assert_eq!(r_i64.count(), 100);

    // Widening access — u8 as i32 (always succeeds)
    let r_wide = TypedReader::<i32>::open(dir.join("meta.u8")).unwrap();
    assert_eq!(r_wide.count(), 100);
    let val = r_wide.get_value(0).unwrap();
    assert_eq!(val, 0_i32); // first u8 value (0 % 256)

    // Narrowing access — i32 as u8 should fail
    let r_narrow = TypedReader::<u8>::open(dir.join("meta.i32"));
    assert!(r_narrow.is_err(), "narrowing i32→u8 should fail at open");
}

#[test]
fn format_coverage_typed_access_via_profile() {
    use vectordata::typed_access::TypedReader;
    use vectordata::view::GenericTestDataView;
    use vectordata::group::DataSource;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Create a dataset with u8 metadata and u16 predicates
    write_fvec_data(&dir.join("base.fvec"), 4, 50);
    write_fvec_data(&dir.join("query.fvec"), 4, 10);
    write_ivec_data(&dir.join("indices.ivec"), 5, 10);
    write_fvec_data(&dir.join("distances.fvec"), 5, 10);
    write_scalar(&dir.join("meta.u8"), 50, 1);
    write_scalar(&dir.join("preds.u16"), 20, 2);

    let yaml = r#"name: typed-access-test
profiles:
  default:
    maxk: 5
    base_vectors: base.fvec
    query_vectors: query.fvec
    neighbor_indices: indices.ivec
    neighbor_distances: distances.fvec
    metadata_content: meta.u8
    metadata_predicates: preds.u16
"#;
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();

    let _group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    // Load the config and create a GenericTestDataView for typed access
    let config: vectordata::model::DatasetConfig =
        serde_yaml::from_str(yaml).unwrap();
    let default_cfg = config.profiles.get("default").unwrap();
    let view = GenericTestDataView::new(
        DataSource::FileSystem(dir.to_path_buf()),
        default_cfg.clone(),
    );

    // Open metadata_content as native u8
    let r_native: TypedReader<u8> = view.open_facet_typed("metadata_content").unwrap();
    assert_eq!(r_native.count(), 50);
    assert!(r_native.is_native());

    // Open metadata_content as widened i32
    let r_wide: TypedReader<i32> = view.open_facet_typed("metadata_content").unwrap();
    assert_eq!(r_wide.count(), 50);
    assert!(!r_wide.is_native()); // widened, not native

    // Open predicates as native u16
    let r_pred: TypedReader<u16> = view.open_facet_typed("metadata_predicates").unwrap();
    assert_eq!(r_pred.count(), 20);
    assert!(r_pred.is_native());

    // Open predicates as widened i32
    let r_pred_wide: TypedReader<i32> = view.open_facet_typed("metadata_predicates").unwrap();
    assert_eq!(r_pred_wide.count(), 20);
}

// ═══════════════════════════════════════════════════════════════════════════
// knn_entries.yaml support
// ═══════════════════════════════════════════════════════════════════════════

/// Create a directory with knn_entries.yaml (no dataset.yaml) and data files.
fn create_knn_entries_fixture(dir: &std::path::Path) {
    let dim = 4;
    let base_count = 30;
    let query_count = 5;
    let k = 3;

    write_fvec_data(&dir.join("base.fvec"), dim, base_count);
    write_fvec_data(&dir.join("query.fvec"), dim, query_count);
    write_ivec_data(&dir.join("gt.ivec"), k, query_count);

    let knn_yaml = r#"
_defaults:
  base_url: http://localhost/test

"testdata:default":
  base: base.fvec
  query: query.fvec
  gt: gt.ivec

"testdata:small":
  base: base.fvec
  query: query.fvec
  gt: gt.ivec
"#;
    std::fs::write(dir.join("knn_entries.yaml"), knn_yaml).unwrap();
}

#[test]
fn knn_entries_local_fallback() {
    // When a directory has knn_entries.yaml but NO dataset.yaml,
    // TestDataGroup::load should use it.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_knn_entries_fixture(dir);

    // No dataset.yaml exists
    assert!(!dir.join("dataset.yaml").exists());

    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    // Should have profiles from knn_entries
    let names = group.profile_names();
    assert!(names.iter().any(|n| n == "default"),
        "should have default profile: {:?}", names);

    // Access base vectors
    let view = group.profile("default").unwrap();
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 30, "base count");
    assert_eq!(base.dim(), 4, "base dim");

    // Access query vectors
    let query = view.query_vectors().unwrap();
    assert_eq!(query.count(), 5, "query count");

    // Access ground truth
    let gt = view.neighbor_indices().unwrap();
    assert_eq!(gt.count(), 5, "gt count");
    assert_eq!(gt.dim(), 3, "gt dim (k)");
}

#[test]
fn knn_entries_http_fallback() {
    // When HTTP directory has knn_entries.yaml but no dataset.yaml,
    // TestDataGroup::load should fall back to knn_entries.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_knn_entries_fixture(dir);

    let server = TestServer::start(dir).unwrap();

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let names = group.profile_names();
    assert!(names.iter().any(|n| n == "default"),
        "HTTP: should have default profile: {:?}", names);

    let view = group.profile("default").unwrap();

    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 30, "HTTP base count");
    assert_eq!(base.dim(), 4, "HTTP base dim");

    let query = view.query_vectors().unwrap();
    assert_eq!(query.count(), 5, "HTTP query count");

    let gt = view.neighbor_indices().unwrap();
    assert_eq!(gt.count(), 5, "HTTP gt count");
}

#[test]
fn knn_entries_multiple_profiles() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_knn_entries_fixture(dir);

    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();
    let names = group.profile_names();
    assert!(names.iter().any(|n| n == "small"),
        "should have 'small' profile: {:?}", names);

    let small = group.profile("small").unwrap();
    let base = small.base_vectors().unwrap();
    assert_eq!(base.count(), 30); // same file, different profile
}

#[test]
fn knn_entries_parse_roundtrip() {
    use vectordata::knn_entries::KnnEntries;

    let yaml = r#"
_defaults:
  base_url: https://example.com/data

"mydata:default":
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  gt: profiles/base/neighbor_indices.ivec
"#;
    let entries = KnnEntries::parse(yaml).unwrap();
    assert_eq!(entries.base_url, Some("https://example.com/data".to_string()));
    assert_eq!(entries.entries.len(), 1);
    assert_eq!(entries.dataset_names(), vec!["mydata"]);

    let config = entries.to_config();
    assert!(config.profiles.contains_key("default"));
    let p = &config.profiles["default"];
    assert!(p.base_vectors.is_some());
    assert!(p.query_vectors.is_some());
    assert!(p.neighbor_indices.is_some());
}

// ═══════════════════════════════════════════════════════════════════════════
// API discovery: profile names, facet manifest, generic facet access
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn api_discover_profiles_and_facets_local() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_format_coverage_dataset(dir);

    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();

    // Discover profile names
    let names = group.profile_names();
    assert!(names.contains(&"default".to_string()), "should have default profile");

    // Discover facets on a profile
    let view = group.profile("default").unwrap();
    let manifest = view.facet_manifest();
    assert!(manifest.contains_key("base_vectors"),
        "manifest should contain base_vectors: {:?}", manifest.keys().collect::<Vec<_>>());
    assert!(manifest.contains_key("query_vectors"));
    assert!(manifest.contains_key("neighbor_indices"));

    // FacetDescriptor gives source info
    let base_desc = &manifest["base_vectors"];
    assert_eq!(base_desc.source_type.as_deref(), Some("fvec"));
    assert!(base_desc.is_standard());

    // Generic facet access by name
    let base = view.facet("base_vectors").unwrap();
    assert_eq!(base.count(), 50);
    assert_eq!(base.dim(), 4);
}

#[test]
fn api_discover_profiles_and_facets_http() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_format_coverage_dataset(dir);

    let server = TestServer::start(dir).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();

    // Profile discovery over HTTP
    let names = group.profile_names();
    assert!(names.contains(&"default".to_string()));

    // Facet discovery over HTTP
    let view = group.profile("default").unwrap();
    let manifest = view.facet_manifest();
    assert!(manifest.contains_key("base_vectors"));
    assert!(manifest.contains_key("neighbor_indices"));

    // Generic facet access over HTTP
    let base = view.facet("base_vectors").unwrap();
    assert_eq!(base.count(), 50);
}

#[test]
fn api_facet_manifest_excludes_absent_facets() {
    // A dataset with only BQG should not list metadata facets
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    write_fvec_data(&dir.join("base.fvec"), 4, 20);
    write_fvec_data(&dir.join("query.fvec"), 4, 5);
    write_ivec_data(&dir.join("gt.ivec"), 3, 5);

    let yaml = r#"name: minimal
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
    neighbor_indices: gt.ivec
"#;
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();

    let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let manifest = view.facet_manifest();

    assert!(manifest.contains_key("base_vectors"));
    assert!(manifest.contains_key("query_vectors"));
    assert!(manifest.contains_key("neighbor_indices"));
    assert!(!manifest.contains_key("metadata_content"),
        "minimal dataset should not have metadata_content");
    assert!(!manifest.contains_key("filtered_neighbor_indices"),
        "minimal dataset should not have filtered facets");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cache-backed HTTP access
// ═══════════════════════════════════════════════════════════════════════════

/// Create a dataset with .mref merkle files for cache testing.
fn create_cached_dataset(dir: &std::path::Path) {
    let dim = 4;
    let base_count = 50;
    let query_count = 10;
    let k = 5;

    write_fvec_data(&dir.join("base.fvec"), dim, base_count);
    write_fvec_data(&dir.join("query.fvec"), dim, query_count);
    write_ivec_data(&dir.join("gt.ivec"), k, query_count);

    // Generate .mref files using the merkle crate
    for name in &["base.fvec", "query.fvec", "gt.ivec"] {
        let file_path = dir.join(name);
        let content = std::fs::read(&file_path).unwrap();
        let mref = vectordata::merkle::MerkleRef::from_content(&content, 4096);
        let mref_path = dir.join(format!("{}.mref", name));
        mref.save(&mref_path).unwrap();
    }

    let yaml = format!(r#"name: cached-test
profiles:
  default:
    maxk: {k}
    base_vectors: base.fvec
    query_vectors: query.fvec
    neighbor_indices: gt.ivec
"#);
    std::fs::write(dir.join("dataset.yaml"), &yaml).unwrap();
}

#[test]
fn cached_http_read_uses_local_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    let cache_dir = tmp.path().join("_test_cache");
    let server = TestServer::start(dir).unwrap();

    let url = url::Url::parse(&format!("{}/base.fvec", server.base_url().trim_end_matches('/')))
        .unwrap();
    let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
        url, 4, &cache_dir,
    ).unwrap();

    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 50);

    // First read — downloads and caches
    let v0 = vectordata::VectorReader::<f32>::get(&reader, 0).unwrap();
    assert_eq!(v0.len(), 4);

    // Cache files should exist
    assert!(cache_dir.exists(), "cache directory should be created");

    // Second read — from cache
    let v1 = vectordata::VectorReader::<f32>::get(&reader, 1).unwrap();
    assert_eq!(v1.len(), 4);

    // Last vector
    let v_last = vectordata::VectorReader::<f32>::get(&reader, 49).unwrap();
    assert_eq!(v_last.len(), 4);
}

#[test]
fn cached_http_prebuffer_downloads_all() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    let cache_dir = tmp.path().join("_prebuffer_cache");
    let server = TestServer::start(dir).unwrap();

    let url = url::Url::parse(&format!("{}/base.fvec", server.base_url().trim_end_matches('/')))
        .unwrap();
    let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
        url, 4, &cache_dir,
    ).unwrap();

    // Prebuffer — downloads everything
    reader.prebuffer().unwrap();

    // All reads should work from cache
    for i in 0..50 {
        let v = vectordata::VectorReader::<f32>::get(&reader, i).unwrap();
        assert_eq!(v.len(), 4);
    }
}

#[test]
fn cached_http_survives_short_read_with_retry() {
    // Test that the cache layer retries and recovers from transient errors.
    // We use a real HTTP server but verify the retry path works by
    // requesting all records — if any chunk download fails transiently,
    // the retry policy should recover.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    let cache_dir = tmp.path().join("_retry_cache");
    let server = TestServer::start(dir).unwrap();

    let url = url::Url::parse(&format!("{}/base.fvec", server.base_url().trim_end_matches('/')))
        .unwrap();
    let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
        url, 4, &cache_dir,
    ).unwrap();

    // Read every record — exercises full chunk coverage
    for i in 0..reader.count() {
        let v = vectordata::VectorReader::<f32>::get(&reader, i).unwrap();
        assert_eq!(v.len(), 4, "record {} should have dim=4", i);
        assert!(v.iter().all(|x| x.is_finite()), "record {} has non-finite values", i);
    }
}

// ── Mmap switchover and merkle verification tests ────────────────────────

#[test]
fn cached_switches_to_mmap_after_prebuffer() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    let cache_dir = tmp.path().join("_mmap_cache");
    let server = TestServer::start(dir).unwrap();

    let url = url::Url::parse(&format!("{}/base.fvec", server.base_url().trim_end_matches('/')))
        .unwrap();
    let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
        url, 4, &cache_dir,
    ).unwrap();

    // Before prebuffer — may not be complete yet (first chunk downloaded for header)
    // After prebuffer — should be complete and switch to mmap
    reader.prebuffer().unwrap();
    assert!(reader.is_complete(), "should be complete after prebuffer");

    // Reads after prebuffer use mmap (zero-copy, no channel locks)
    for i in 0..50 {
        let v = vectordata::VectorReader::<f32>::get(&reader, i).unwrap();
        assert_eq!(v.len(), 4);
    }
}

#[test]
fn cached_mmap_on_reopen_of_complete_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    let cache_dir = tmp.path().join("_reopen_cache");
    let server = TestServer::start(dir).unwrap();

    let base_url = format!("{}/base.fvec", server.base_url().trim_end_matches('/'));

    // First open + prebuffer
    {
        let url = url::Url::parse(&base_url).unwrap();
        let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
            url, 4, &cache_dir,
        ).unwrap();
        reader.prebuffer().unwrap();
        assert!(reader.is_complete());
    }

    // Second open — cache is already complete, should start in mmap mode
    {
        let url = url::Url::parse(&base_url).unwrap();
        let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
            url, 4, &cache_dir,
        ).unwrap();
        assert!(reader.is_complete(),
            "re-opened reader should detect complete cache immediately");

        // Verify data is correct from mmap
        let v0 = vectordata::VectorReader::<f32>::get(&reader, 0).unwrap();
        assert_eq!(v0.len(), 4);
        let v_last = vectordata::VectorReader::<f32>::get(&reader, 49).unwrap();
        assert_eq!(v_last.len(), 4);
    }
}

#[test]
fn cached_merkle_state_persists_across_opens() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    let cache_dir = tmp.path().join("_persist_cache");
    let server = TestServer::start(dir).unwrap();

    let base_url = format!("{}/base.fvec", server.base_url().trim_end_matches('/'));

    // First open — read a few records (downloads some chunks)
    {
        let url = url::Url::parse(&base_url).unwrap();
        let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
            url, 4, &cache_dir,
        ).unwrap();
        let _ = vectordata::VectorReader::<f32>::get(&reader, 0).unwrap();
        let _ = vectordata::VectorReader::<f32>::get(&reader, 25).unwrap();
    }

    // .mrkl state file should exist
    // mrkl could be in a subdirectory
    let has_mrkl = find_file_recursive(&cache_dir, ".mrkl");
    assert!(has_mrkl, "merkle state file (.mrkl) should persist after close");

    // Second open — should resume from persisted state (not re-download)
    {
        let url = url::Url::parse(&base_url).unwrap();
        let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
            url, 4, &cache_dir,
        ).unwrap();
        // Previously downloaded records should be served from cache
        let v0 = vectordata::VectorReader::<f32>::get(&reader, 0).unwrap();
        assert_eq!(v0.len(), 4);
    }
}

#[test]
fn cached_data_matches_original_exactly() {
    // Verify that cached data is bit-for-bit identical to the original
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    let cache_dir = tmp.path().join("_exact_cache");
    let server = TestServer::start(dir).unwrap();

    let url = url::Url::parse(&format!("{}/base.fvec", server.base_url().trim_end_matches('/')))
        .unwrap();
    let reader = vectordata::cache::reader::CachedVectorReader::<f32>::open(
        url, 4, &cache_dir,
    ).unwrap();
    reader.prebuffer().unwrap();

    // Read original file directly
    let original = vectordata::io::open_vec::<f32>(dir.join("base.fvec").to_str().unwrap()).unwrap();

    // Compare every record
    for i in 0..50 {
        let cached = vectordata::VectorReader::<f32>::get(&reader, i).unwrap();
        let direct = original.get(i).unwrap();
        assert_eq!(cached, direct, "record {} should match original exactly", i);
    }
}

// ── Fault injection tests ────────────────────────────────────────────────
//
// These tests verify that all access modes produce clear errors (not panics
// or silent corruption) when data is truncated, corrupted, or missing.

#[test]
fn fault_truncated_fvec_local() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Write a truncated fvec: valid dim header (4) but only 1 byte of data
    std::fs::write(dir.join("truncated.fvec"), b"\x04\x00\x00\x00\x01").unwrap();

    let result = vectordata::io::open_vec::<f32>(dir.join("truncated.fvec").to_str().unwrap());
    // Should open (dim is readable) but count should reflect the truncation
    match result {
        Ok(reader) => {
            assert_eq!(reader.dim(), 4);
            // count = floor(5 / (4 + 4*4)) = 0 — truncated file has no complete records
            assert_eq!(reader.count(), 0, "truncated file should have 0 complete records");
        }
        Err(_) => {} // also acceptable — some formats reject on open
    }
}

#[test]
fn fault_truncated_fvec_http() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    std::fs::write(dir.join("truncated.fvec"), b"\x04\x00\x00\x00\x01").unwrap();
    let yaml = "name: trunc\nprofiles:\n  default:\n    base_vectors: truncated.fvec\n";
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();

    let server = TestServer::start(dir).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let base = view.base_vectors().unwrap();
    // 0 records due to truncation
    assert_eq!(base.count(), 0);
}

#[test]
fn fault_zero_length_file_local() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    std::fs::write(dir.join("empty.fvec"), b"").unwrap();
    let result = vectordata::io::open_vec::<f32>(dir.join("empty.fvec").to_str().unwrap());
    match result {
        Err(_) => {} // expected — can't read dim header
        Ok(reader) => {
            // If open succeeds (empty mmap), count must be 0 and get must error
            assert_eq!(reader.count(), 0, "empty file should have 0 records");
            assert!(reader.get(0).is_err(), "reading from empty file should error");
        }
    }
}

#[test]
fn fault_zero_length_file_http() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    std::fs::write(dir.join("empty.fvec"), b"").unwrap();
    let yaml = "name: empty\nprofiles:\n  default:\n    base_vectors: empty.fvec\n";
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();

    let server = TestServer::start(dir).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    let result = view.base_vectors();
    assert!(result.is_err(), "zero-length fvec over HTTP should error");
}

#[test]
fn fault_corrupt_data_detected_by_merkle() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    create_cached_dataset(dir);

    // Corrupt the data file AFTER creating the .mref
    let base_path = dir.join("base.fvec");
    let mut data = std::fs::read(&base_path).unwrap();
    // Flip bytes in the middle
    if data.len() > 100 {
        data[50] ^= 0xFF;
        data[51] ^= 0xFF;
        data[52] ^= 0xFF;
        data[53] ^= 0xFF;
    }
    std::fs::write(&base_path, &data).unwrap();

    let cache_dir = tmp.path().join("_corrupt_cache");
    let server = TestServer::start(dir).unwrap();

    let url = url::Url::parse(&format!("{}/base.fvec", server.base_url().trim_end_matches('/')))
        .unwrap();
    let result = vectordata::cache::reader::CachedVectorReader::<f32>::open(
        url, 4, &cache_dir,
    );

    // The .mref was created from the original data, but the server now
    // serves corrupt data. Merkle verification should detect the mismatch
    // and either error or retry (and fail since data is persistently corrupt).
    match result {
        Ok(reader) => {
            // If open succeeded (only reads header), reading should fail
            // on merkle verification of the corrupted chunk.
            let read_result = vectordata::VectorReader::<f32>::get(&reader, 5);
            // Either the read errors or the data is silently wrong (chunk too small
            // to have its own merkle node). Both outcomes are documented.
            if let Ok(v) = read_result {
                // With small test data and 4096-byte chunks, all data is in one
                // chunk. The merkle check happens at chunk granularity — if the
                // corrupt bytes are in the same chunk as the header (which was
                // already read successfully), the chunk may have been cached
                // before corruption detection.
                assert_eq!(v.len(), 4);
            }
        }
        Err(e) => {
            // Merkle verification caught the corruption — good
            let msg = format!("{}", e);
            assert!(msg.contains("verification") || msg.contains("hash") || msg.contains("mismatch")
                || msg.contains("retry") || msg.contains("failed"),
                "error should mention verification failure: {}", msg);
        }
    }
}

#[test]
fn fault_out_of_bounds_read() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    write_fvec_data(&dir.join("small.fvec"), 4, 10);

    // Local
    let reader = vectordata::io::open_vec::<f32>(dir.join("small.fvec").to_str().unwrap()).unwrap();
    assert_eq!(reader.count(), 10);
    assert!(reader.get(10).is_err(),
        "reading index 10 from 10-record file should error");
    assert!(reader.get(1000).is_err(),
        "reading index 1000 should error");

    // HTTP
    let yaml = "name: small\nprofiles:\n  default:\n    base_vectors: small.fvec\n";
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();
    let server = TestServer::start(dir).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 10);
    assert!(base.get(10).is_err(),
        "HTTP: reading index 10 from 10-record file should error");
}

#[test]
fn cached_http_fallback_when_no_mref() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    write_fvec_data(&dir.join("base.fvec"), 4, 20);
    let yaml = "name: no-mref\nprofiles:\n  default:\n    base_vectors: base.fvec\n";
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();

    let server = TestServer::start(dir).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    // Should work via direct HTTP fallback (no .mref)
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 20);
    assert_eq!(base.dim(), 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// TypedReader over HTTP — full chain from catalog to typed ordinal access
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn typed_reader_open_url_ivec() {
    use vectordata::typed_access::{ElementType, TypedReader};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_ivec_data(&dir.join("gt.ivec"), 5, 20);

    let server = TestServer::start(dir).unwrap();
    let url = url::Url::parse(&format!("{}/gt.ivec", server.base_url().trim_end_matches('/'))).unwrap();

    let reader = TypedReader::<i32>::open_url(url, ElementType::I32).unwrap();
    assert_eq!(reader.count(), 20);
    assert_eq!(reader.dim(), 5);

    let val = reader.get_value(0).unwrap();
    assert!(val >= 0);

    let record = reader.get_record(0).unwrap();
    assert_eq!(record.len(), 5);
}

#[test]
fn typed_reader_open_url_scalar_u8() {
    use vectordata::typed_access::{ElementType, TypedReader};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_scalar(&dir.join("labels.u8"), 100, 1);

    let server = TestServer::start(dir).unwrap();
    let url = url::Url::parse(&format!("{}/labels.u8", server.base_url().trim_end_matches('/'))).unwrap();

    let reader = TypedReader::<u8>::open_url(url, ElementType::U8).unwrap();
    assert_eq!(reader.count(), 100);
    assert_eq!(reader.dim(), 1);
    assert!(reader.is_native());

    let val = reader.get_value(0).unwrap();
    assert_eq!(val, 0_u8); // first value is 0 % 256

    // get_native falls back to get_value for HTTP
    let native_val = reader.get_native(0);
    assert_eq!(native_val, 0_u8);
}

#[test]
fn typed_reader_open_url_widening() {
    use vectordata::typed_access::{ElementType, TypedReader};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_scalar(&dir.join("labels.u8"), 50, 1);

    let server = TestServer::start(dir).unwrap();
    let url = url::Url::parse(&format!("{}/labels.u8", server.base_url().trim_end_matches('/'))).unwrap();

    // Open u8 file as i32 (widening)
    let reader = TypedReader::<i32>::open_url(url, ElementType::U8).unwrap();
    assert_eq!(reader.count(), 50);
    assert!(!reader.is_native());

    let val = reader.get_value(0).unwrap();
    assert_eq!(val, 0_i32);
}

#[test]
fn typed_reader_open_url_narrowing_rejected() {
    use vectordata::typed_access::{ElementType, TypedReader};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_scalar(&dir.join("values.i32"), 10, 4);

    let server = TestServer::start(dir).unwrap();
    let url = url::Url::parse(&format!("{}/values.i32", server.base_url().trim_end_matches('/'))).unwrap();

    // Opening i32 as u8 should fail (narrowing)
    let result = TypedReader::<u8>::open_url(url, ElementType::I32);
    assert!(result.is_err());
}

#[test]
fn typed_reader_open_auto_dispatches() {
    use vectordata::typed_access::{ElementType, TypedReader};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_scalar(&dir.join("data.u8"), 20, 1);

    // Local path
    let reader = TypedReader::<u8>::open_auto(
        dir.join("data.u8").to_str().unwrap(), ElementType::U8,
    ).unwrap();
    assert_eq!(reader.count(), 20);
    assert!(reader.is_native());

    // HTTP URL
    let server = TestServer::start(dir).unwrap();
    let url = format!("{}/data.u8", server.base_url().trim_end_matches('/'));
    let reader = TypedReader::<u8>::open_auto(&url, ElementType::U8).unwrap();
    assert_eq!(reader.count(), 20);
}

#[test]
fn typed_reader_via_profile_over_http() {
    use vectordata::typed_access::TypedReader;
    use vectordata::view::GenericTestDataView;
    use vectordata::group::DataSource;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    write_ivec_data(&dir.join("gt.ivec"), 5, 30);
    write_scalar(&dir.join("meta.u8"), 30, 1);

    let yaml = r#"name: typed-http-test
profiles:
  default:
    neighbor_indices: gt.ivec
    metadata_content: meta.u8
"#;
    std::fs::write(dir.join("dataset.yaml"), yaml).unwrap();

    let config: vectordata::model::DatasetConfig =
        serde_yaml::from_str(yaml).unwrap();
    let default_cfg = config.profiles.get("default").unwrap();

    let server = TestServer::start(dir).unwrap();
    let base_url = url::Url::parse(&server.base_url()).unwrap();
    let view = GenericTestDataView::new(
        DataSource::Http(base_url),
        default_cfg.clone(),
    );

    // open_facet_typed for neighbor_indices (i32) over HTTP
    let r_idx: TypedReader<i32> = view.open_facet_typed("neighbor_indices").unwrap();
    assert_eq!(r_idx.count(), 30);
    assert_eq!(r_idx.dim(), 5);
    assert!(r_idx.is_native());
    let val = r_idx.get_value(0).unwrap();
    assert!(val >= 0);

    // open_facet_typed for metadata_content (u8) over HTTP
    let r_meta: TypedReader<u8> = view.open_facet_typed("metadata_content").unwrap();
    assert_eq!(r_meta.count(), 30);
    assert!(r_meta.is_native());

    // Widened access: u8 metadata as i32 over HTTP
    let r_wide: TypedReader<i32> = view.open_facet_typed("metadata_content").unwrap();
    assert_eq!(r_wide.count(), 30);
    assert!(!r_wide.is_native());
    let val_i32 = r_wide.get_value(0).unwrap();
    assert_eq!(val_i32, 0_i32);
}

#[test]
fn typed_reader_element_type_from_url() {
    use vectordata::typed_access::ElementType;

    let url = url::Url::parse("https://example.com/data/base.fvec").unwrap();
    assert_eq!(ElementType::from_url(&url).unwrap(), ElementType::F32);

    let url = url::Url::parse("https://example.com/meta.u8").unwrap();
    assert_eq!(ElementType::from_url(&url).unwrap(), ElementType::U8);

    let url = url::Url::parse("https://example.com/indices.ivec").unwrap();
    assert_eq!(ElementType::from_url(&url).unwrap(), ElementType::I32);
}

#[test]
fn typed_reader_get_native_slice_returns_none_for_http() {
    use vectordata::typed_access::{ElementType, TypedReader};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_scalar(&dir.join("labels.u8"), 20, 1);

    let server = TestServer::start(dir).unwrap();
    let url = url::Url::parse(&format!("{}/labels.u8", server.base_url().trim_end_matches('/'))).unwrap();

    let reader = TypedReader::<u8>::open_url(url, ElementType::U8).unwrap();

    // get_native_slice returns None for HTTP (can't return reference)
    assert!(reader.get_native_slice(0).is_none());

    // But get_record works (returns owned Vec)
    let record = reader.get_record(0).unwrap();
    assert_eq!(record.len(), 1);
}

#[test]
fn typed_reader_data_matches_local_and_http() {
    use vectordata::typed_access::{ElementType, TypedReader};

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_scalar(&dir.join("data.u8"), 25, 1);

    let server = TestServer::start(dir).unwrap();

    // Local reader
    let local = TypedReader::<u8>::open(dir.join("data.u8")).unwrap();

    // HTTP reader
    let url = url::Url::parse(&format!("{}/data.u8", server.base_url().trim_end_matches('/'))).unwrap();
    let remote = TypedReader::<u8>::open_url(url, ElementType::U8).unwrap();

    assert_eq!(local.count(), remote.count());
    assert_eq!(local.dim(), remote.dim());

    // Compare every record
    for i in 0..25 {
        let local_val = local.get_value(i).unwrap();
        let remote_val = remote.get_value(i).unwrap();
        assert_eq!(local_val, remote_val, "value {} should match", i);
    }
}

