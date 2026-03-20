// Copyright (c) DataStax, Inc.
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
use vectordata::io::{HttpVectorReader, VectorReader};
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
