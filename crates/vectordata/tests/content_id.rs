// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A facet's content identity (`FacetStorage::content_id`, IN-6/IN-7):
//! a facet fetched from a server that publishes no `.mref` records a
//! merkle tree as its chunks land, and its root is the root a `.mref`
//! of the same bytes would have — and a local copy's computed root.
//! A cached chunk changed since it landed is found (V9).

mod support;

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use support::testserver::TestServer;
use vectordata::merkle::{ContentIdKind, MerkleRef};
use vectordata::verify::{Check, NoProgress, VerifyOptions};
use vectordata::TestDataGroup;

static CACHE: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
});

/// A served dataset whose base spans two 8 MiB chunks; returns the
/// base's bytes.
fn served_dataset(root: &Path) -> Vec<u8> {
    let dim = 1024usize;
    let rows = 2100; // 4100 bytes a row: 8.2 MiB, two chunks
    let mut bytes = Vec::with_capacity(rows * (4 + dim * 4));
    for r in 0..rows {
        bytes.extend_from_slice(&(dim as i32).to_le_bytes());
        for k in 0..dim {
            bytes.extend_from_slice(&(((r * 7 + k) % 1000) as f32).to_le_bytes());
        }
    }
    std::fs::write(root.join("base.fvec"), &bytes).unwrap();
    std::fs::write(root.join("dataset.yaml"), "name: cid\nprofiles:\n  default:\n    base_vectors: base.fvec\n").unwrap();
    bytes
}

fn cached_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = cached_file(&p, name) {
                return Some(f);
            }
        } else if p.file_name().is_some_and(|n| n == name) {
            return Some(p);
        }
    }
    None
}

#[test]
fn a_fetched_facet_without_a_reference_records_its_content_root() {
    vectordata::settings::override_cache_dir_for_process(CACHE.path().to_path_buf());
    let served = tempfile::tempdir().unwrap();
    let bytes = served_dataset(served.path());
    let server = TestServer::start(served.path()).unwrap();
    let expected = *MerkleRef::from_content(&bytes, 8 * 1024 * 1024).root_hash();

    let group = TestDataGroup::load(&format!("{}/", server.base_url())).unwrap();
    let view = group.profile("default").unwrap();
    let storage = view.open_facet_storage("base_vectors").unwrap();
    assert_eq!(storage.content_id().unwrap(), None, "no root while chunks are missing");
    storage.precache().unwrap();
    let id = storage.content_id().unwrap().expect("complete: a root");
    assert_eq!((id.root, id.kind), (expected, ContentIdKind::Recorded), "the root a .mref of the same bytes has");

    // A local copy of the same bytes has the same identity, computed.
    let local = TestDataGroup::load(served.path().to_str().unwrap()).unwrap();
    let local_id = local.profile("default").unwrap().open_facet_storage("base_vectors").unwrap().content_id().unwrap().unwrap();
    assert_eq!((local_id.root, local_id.kind), (expected, ContentIdKind::Computed));

    // The cache is what landed: V9 passes, then finds a flipped byte.
    let options = VerifyOptions { cached_chunks: true, ..Default::default() };
    let report = view.verify_report(&options, &mut NoProgress);
    assert!(report.passed() && report.checked.contains(&Check::V9CachedChunks), "{:?}", report);
    let cache_file = cached_file(CACHE.path(), "base.fvec").expect("the cache copy");
    let mut cached = std::fs::read(&cache_file).unwrap();
    cached[8 * 1024 * 1024 + 100] ^= 0xff;
    std::fs::write(&cache_file, &cached).unwrap();
    let report = view.verify_report(&options, &mut NoProgress);
    let failure = report.failures.iter().find(|f| f.check == Check::V9CachedChunks).expect("V9 fails");
    assert!(failure.found.ends_with(": 1"), "chunk 1 holds 8 MiB + 100: {failure}");
}
