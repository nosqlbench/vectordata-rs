// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the data access layer.
//!
//! Uses TestServer to serve vector files over HTTP and verifies that
//! CachedVectorView correctly fetches, verifies, and caches chunks
//! on demand.

mod support;

use std::io::Write;
use std::path::Path;
use url::Url;
use vectordata::cache::CachedChannel;
use vectordata::dataset::view::{CachedVectorView, LocalVectorView, TypedVectorView, VecElementType};
use vectordata::merkle::MerkleRef;
use vectordata::transport::HttpTransport;

/// Write a test fvec file with sequential values.
fn write_fvec(path: &Path, dim: u32, count: u32) {
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..count {
        f.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = (i * dim + j) as f32;
            f.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

/// Write a test mvec (f16) file with sequential values.
fn write_mvec(path: &Path, dim: u32, count: u32) {
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..count {
        f.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for j in 0..dim {
            let val = half::f16::from_f32((i * dim + j) as f32);
            f.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// Local view tests
// ---------------------------------------------------------------------------

#[test]
fn test_local_vector_view_fvec() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.fvec");
    write_fvec(&path, 8, 1000);

    let view = LocalVectorView::open(&path, None).unwrap();
    assert_eq!(view.count(), 1000);
    assert_eq!(view.dim(), 8);
    assert!(view.is_local());

    let v = view.get_f64(0).unwrap();
    assert_eq!(v.len(), 8);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[7], 7.0);

    let v = view.get_f32(999).unwrap();
    assert_eq!(v[0], (999 * 8) as f32);
}

#[test]
fn test_local_vector_view_mvec() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.mvec");
    write_mvec(&path, 4, 100);

    let view = LocalVectorView::open(&path, None).unwrap();
    assert_eq!(view.count(), 100);
    assert_eq!(view.dim(), 4);

    let v = view.get_f32(0).unwrap();
    assert_eq!(v.len(), 4);
    // f16 values are approximate
    assert!((v[0] - 0.0).abs() < 0.01);
    assert!((v[3] - 3.0).abs() < 0.01);
}

#[test]
fn test_local_vector_view_windowed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.fvec");
    write_fvec(&path, 4, 1000);

    let view = LocalVectorView::open(&path, Some((100, 200))).unwrap();
    assert_eq!(view.count(), 200);
    assert_eq!(view.dim(), 4);

    // Index 0 of view = index 100 of file
    let v = view.get_f64(0).unwrap();
    assert_eq!(v[0], (100 * 4) as f64);

    // Index 199 = file index 299
    let v = view.get_f64(199).unwrap();
    assert_eq!(v[0], (299 * 4) as f64);

    // Out of window
    assert!(view.get_f64(200).is_none());
}

// ---------------------------------------------------------------------------
// Cached view tests (remote via TestServer)
// ---------------------------------------------------------------------------

#[test]
fn test_cached_vector_view_remote() {
    let dir = tempfile::tempdir().unwrap();
    let serve_dir = dir.path().join("serve");
    let cache_dir = dir.path().join("cache");
    std::fs::create_dir_all(&serve_dir).unwrap();
    std::fs::create_dir_all(&cache_dir).unwrap();

    // Create test data
    let fvec_path = serve_dir.join("vectors.fvec");
    write_fvec(&fvec_path, 4, 500);

    // Create merkle reference
    let content = std::fs::read(&fvec_path).unwrap();
    let mref = MerkleRef::from_content(&content, 4096); // 4KB chunks
    mref.save(&serve_dir.join("vectors.fvec.mref")).unwrap();

    // Start HTTP server
    let server = support::testserver::TestServer::start(&serve_dir).unwrap();
    let base_url = server.base_url();

    // Create transport + cached channel
    let url = Url::parse(&format!("{}vectors.fvec", base_url)).unwrap();
    let transport = HttpTransport::new(url);
    let channel = CachedChannel::open(
        Box::new(transport),
        mref,
        &cache_dir,
        "vectors.fvec",
    ).unwrap();

    // Create cached vector view
    let view = CachedVectorView::new(
        channel,
        cache_dir.join("vectors.fvec"),
        VecElementType::F32,
        None,
    ).unwrap();

    assert_eq!(view.count(), 500);
    assert_eq!(view.dim(), 4);
    assert!(!view.is_local()); // not promoted yet

    // Read a single vector — triggers chunk download
    let v = view.get_f64(0).unwrap();
    assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0]);

    // Read another vector (possibly different chunk)
    let v = view.get_f32(499).unwrap();
    assert_eq!(v[0], (499 * 4) as f32);
}

#[test]
fn test_cached_vector_view_partial_fetch() {
    let dir = tempfile::tempdir().unwrap();
    let serve_dir = dir.path().join("serve");
    let cache_dir = dir.path().join("cache");
    std::fs::create_dir_all(&serve_dir).unwrap();
    std::fs::create_dir_all(&cache_dir).unwrap();

    // Create larger test data (many chunks)
    let fvec_path = serve_dir.join("big.fvec");
    write_fvec(&fvec_path, 128, 10000); // ~5MB

    let content = std::fs::read(&fvec_path).unwrap();
    let mref = MerkleRef::from_content(&content, 65536); // 64KB chunks
    mref.save(&serve_dir.join("big.fvec.mref")).unwrap();

    let server = support::testserver::TestServer::start(&serve_dir).unwrap();
    let url = Url::parse(&format!("{}big.fvec", server.base_url())).unwrap();
    let transport = HttpTransport::new(url);
    let channel = CachedChannel::open(
        Box::new(transport),
        mref,
        &cache_dir,
        "big.fvec",
    ).unwrap();

    let view = CachedVectorView::new(
        channel,
        cache_dir.join("big.fvec"),
        VecElementType::F32,
        None,
    ).unwrap();

    // Read just one vector — should only download the chunk containing it
    let v = view.get_f64(5000).unwrap();
    assert_eq!(v[0], (5000 * 128) as f64);
    assert!(!view.is_local()); // not fully cached
}

#[test]
fn test_cached_vector_view_windowed() {
    let dir = tempfile::tempdir().unwrap();
    let serve_dir = dir.path().join("serve");
    let cache_dir = dir.path().join("cache");
    std::fs::create_dir_all(&serve_dir).unwrap();
    std::fs::create_dir_all(&cache_dir).unwrap();

    let fvec_path = serve_dir.join("vectors.fvec");
    write_fvec(&fvec_path, 4, 1000);

    let content = std::fs::read(&fvec_path).unwrap();
    let mref = MerkleRef::from_content(&content, 4096);
    mref.save(&serve_dir.join("vectors.fvec.mref")).unwrap();

    let server = support::testserver::TestServer::start(&serve_dir).unwrap();
    let url = Url::parse(&format!("{}vectors.fvec", server.base_url())).unwrap();
    let transport = HttpTransport::new(url);
    let channel = CachedChannel::open(
        Box::new(transport),
        mref,
        &cache_dir,
        "vectors.fvec",
    ).unwrap();

    // Window: only vectors 100..300
    let view = CachedVectorView::new(
        channel,
        cache_dir.join("vectors.fvec"),
        VecElementType::F32,
        Some((100, 200)),
    ).unwrap();

    assert_eq!(view.count(), 200);
    assert_eq!(view.dim(), 4);

    // View index 0 = file index 100
    let v = view.get_f64(0).unwrap();
    assert_eq!(v[0], (100 * 4) as f64);

    assert!(view.get_f64(200).is_none());
}

#[test]
fn test_remote_dataset_view() {
    let dir = tempfile::tempdir().unwrap();
    let serve_dir = dir.path().join("serve");
    let cache_dir = dir.path().join("cache");
    std::fs::create_dir_all(&serve_dir).unwrap();

    // Create a minimal dataset with base vectors
    let fvec_path = serve_dir.join("base.fvec");
    write_fvec(&fvec_path, 4, 200);

    // Create merkle reference
    let content = std::fs::read(&fvec_path).unwrap();
    let mref = MerkleRef::from_content(&content, 4096);
    mref.save(&serve_dir.join("base.fvec.mref")).unwrap();

    // Create a catalog entry
    let server = support::testserver::TestServer::start(&serve_dir).unwrap();
    let base_url = server.base_url();

    let entry = vectordata::dataset::catalog::CatalogEntry {
        name: "test-dataset".to_string(),
        path: format!("{}dataset.yaml", base_url),
        dataset_type: "dataset.yaml".to_string(),
        layout: vectordata::dataset::catalog::CatalogLayout {
            attributes: None,
            profiles: {
                use vectordata::dataset::profile::{DSProfile, DSProfileGroup, DSView};
                use vectordata::dataset::source::DSSource;
                use indexmap::IndexMap;

                let mut views = IndexMap::new();
                views.insert("base_vectors".to_string(), DSView {
                    source: DSSource {
                        path: "base.fvec".to_string(),
                        namespace: None,
                        window: vectordata::dataset::source::DSWindow::default(),
                    },
                    window: None,
                });

                let mut profiles = IndexMap::new();
                profiles.insert("default".to_string(), DSProfile {
                    maxk: Some(100),
                    base_count: None,
                    views,
                });
                DSProfileGroup::from_profiles(profiles)
            },
        },
    };

    // Open remote view
    let view = vectordata::dataset::remote::RemoteDatasetView::open(
        &entry,
        "default",
        &cache_dir,
    ).unwrap();

    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 200);
    assert_eq!(base.dim(), 4);

    // Read a vector — triggers on-demand download + merkle verify
    let v = base.get_f64(0).unwrap();
    assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0]);

    let v = base.get_f64(199).unwrap();
    assert_eq!(v[0], (199 * 4) as f64);
}

#[test]
fn test_dataset_loader_local_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.fvec");
    write_fvec(&path, 4, 50);

    let loader = vectordata::dataset::loader::DatasetLoader::new(dir.path().to_path_buf());
    let dataset = loader.load(path.to_str().unwrap(), &[]).unwrap();
    let base = dataset.base_vectors().unwrap();
    assert_eq!(base.count(), 50);
    assert_eq!(base.dim(), 4);
}

#[test]
fn test_cached_vector_view_prebuffer_promotes() {
    let dir = tempfile::tempdir().unwrap();
    let serve_dir = dir.path().join("serve");
    let cache_dir = dir.path().join("cache");
    std::fs::create_dir_all(&serve_dir).unwrap();
    std::fs::create_dir_all(&cache_dir).unwrap();

    let fvec_path = serve_dir.join("small.fvec");
    write_fvec(&fvec_path, 4, 100); // Small file — fits in one or two chunks

    let content = std::fs::read(&fvec_path).unwrap();
    let mref = MerkleRef::from_content(&content, 4096);
    mref.save(&serve_dir.join("small.fvec.mref")).unwrap();

    let server = support::testserver::TestServer::start(&serve_dir).unwrap();
    let url = Url::parse(&format!("{}small.fvec", server.base_url())).unwrap();
    let transport = HttpTransport::new(url);
    let channel = CachedChannel::open(
        Box::new(transport),
        mref,
        &cache_dir,
        "small.fvec",
    ).unwrap();

    let view = CachedVectorView::new(
        channel,
        cache_dir.join("small.fvec"),
        VecElementType::F32,
        None,
    ).unwrap();

    assert!(!view.is_local());

    // Prebuffer all
    view.prebuffer(0, 100).unwrap();

    // Should be promoted to mmap now
    assert!(view.is_local());

    // Still readable after promotion
    let v = view.get_f64(50).unwrap();
    assert_eq!(v[0], (50 * 4) as f64);
}
