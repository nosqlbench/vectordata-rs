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

    // Pre-build the index (simulates pipeline generate-vvec-index step)
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

