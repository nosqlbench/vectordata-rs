// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Phase 4: facet manifest, generic facet access,
//! distance function, and mvec HTTP reader.

use std::fs;
use std::io::Write;
use std::path::Path;

mod support;
use support::testserver::TestServer;
use vectordata::io::{HttpVectorReader, VectorReader};
use vectordata::TestDataGroup;

/// Write a small fvec file with `count` vectors of dimension `dim`.
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

/// Write a small mvec file with `count` vectors of dimension `dim`.
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

fn setup_full_dataset() -> (tempfile::TempDir, TestServer) {
    let dir = tempfile::tempdir().unwrap();

    write_fvec(&dir.path().join("base.fvec"), 4, 10);
    write_fvec(&dir.path().join("query.fvec"), 4, 5);
    write_ivec(&dir.path().join("neighbors.ivec"), 3, 5);
    write_fvec(&dir.path().join("distances.fvec"), 3, 5);
    write_mvec(&dir.path().join("base_f16.mvec"), 4, 10);

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

    let server = TestServer::start(dir.path()).unwrap();
    (dir, server)
}

// ---------------------------------------------------------------------------
// Facet manifest tests
// ---------------------------------------------------------------------------

#[test]
fn test_facet_manifest_local() {
    let (dir, _server) = setup_full_dataset();
    let group = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let manifest = view.facet_manifest();

    assert!(manifest.contains_key("base_vectors"));
    assert!(manifest.contains_key("query_vectors"));
    assert!(manifest.contains_key("neighbor_indices"));
    assert!(manifest.contains_key("neighbor_distances"));
    assert_eq!(manifest.len(), 4);

    let base = &manifest["base_vectors"];
    assert!(base.is_standard());
    assert_eq!(base.source_path.as_deref(), Some("base.fvec"));
    assert_eq!(base.source_type.as_deref(), Some("fvec"));
}

#[test]
fn test_facet_manifest_http() {
    let (_dir, server) = setup_full_dataset();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let manifest = view.facet_manifest();
    assert_eq!(manifest.len(), 4);
    assert!(manifest["base_vectors"].is_standard());
}

// ---------------------------------------------------------------------------
// Generic facet access
// ---------------------------------------------------------------------------

#[test]
fn test_generic_facet_access_local() {
    let (dir, _server) = setup_full_dataset();
    let group = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    // Access base_vectors via generic facet()
    let reader = view.facet("base_vectors").unwrap();
    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 10);
    let v0 = reader.get(0).unwrap();
    assert_eq!(v0, vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_generic_facet_access_http() {
    let (_dir, server) = setup_full_dataset();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let reader = view.facet("query_vectors").unwrap();
    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 5);
}

#[test]
fn test_generic_facet_missing() {
    let (dir, _server) = setup_full_dataset();
    let group = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let result = view.facet("nonexistent_facet");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Distance function
// ---------------------------------------------------------------------------

#[test]
fn test_distance_function_present() {
    let (_dir, server) = setup_full_dataset();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    assert_eq!(view.distance_function(), Some("L2".to_string()));
}

#[test]
fn test_distance_function_from_local() {
    let (dir, _server) = setup_full_dataset();
    let group = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    assert_eq!(view.distance_function(), Some("L2".to_string()));
}

#[test]
fn test_distance_function_absent() {
    let dir = tempfile::tempdir().unwrap();
    write_fvec(&dir.path().join("base.fvec"), 4, 10);
    fs::write(
        dir.path().join("dataset.yaml"),
        "profiles:\n  default:\n    base_vectors: base.fvec\n",
    )
    .unwrap();

    let group = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    assert_eq!(view.distance_function(), None);
}

// ---------------------------------------------------------------------------
// mvec HTTP reader
// ---------------------------------------------------------------------------

#[test]
fn test_mvec_http_reader_metadata() {
    let (_dir, server) = setup_full_dataset();
    let url = url::Url::parse(&format!("{}base_f16.mvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<half::f16>::open_mvec(url).unwrap();

    assert_eq!(reader.dim(), 4);
    assert_eq!(reader.count(), 10);
}

#[test]
fn test_mvec_http_reader_get_vector() {
    let (_dir, server) = setup_full_dataset();
    let url = url::Url::parse(&format!("{}base_f16.mvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<half::f16>::open_mvec(url).unwrap();

    let v0 = reader.get(0).unwrap();
    assert_eq!(v0.len(), 4);
    assert_eq!(v0[0], half::f16::from_f32(0.0));
    assert_eq!(v0[1], half::f16::from_f32(1.0));
    assert_eq!(v0[2], half::f16::from_f32(2.0));
    assert_eq!(v0[3], half::f16::from_f32(3.0));
}

#[test]
fn test_mvec_http_reader_last_vector() {
    let (_dir, server) = setup_full_dataset();
    let url = url::Url::parse(&format!("{}base_f16.mvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<half::f16>::open_mvec(url).unwrap();

    let v9 = reader.get(9).unwrap();
    assert_eq!(v9[0], half::f16::from_f32(36.0));
    assert_eq!(v9[3], half::f16::from_f32(39.0));
}

#[test]
fn test_mvec_http_reader_out_of_bounds() {
    let (_dir, server) = setup_full_dataset();
    let url = url::Url::parse(&format!("{}base_f16.mvec", server.base_url())).unwrap();
    let reader = HttpVectorReader::<half::f16>::open_mvec(url).unwrap();

    assert!(reader.get(10).is_err());
}
