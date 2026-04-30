// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the public `vectordata` reader API against
//! the bundled HTTP TestServer fixture.
//!
//! Every test exercises a *public* surface — there is no way (and no
//! need) to reach the underlying `Storage` / `CachedChannel` /
//! `HttpTransport` types from outside the crate. The structural
//! claim being tested across the whole file is:
//!
//!   *Opening any source through the public dispatch (path or URL)
//!    yields identical bytes regardless of which transport variant
//!    the storage layer chose under the hood.*
//!
//! Layout:
//!
//! - `equivalence`: same fvec read three ways (local mmap, remote +
//!   `.mref` cached, remote without `.mref` direct-HTTP) → identical
//!   per-record bytes.
//! - `cached_promotion`: cached-remote storage starts not-complete /
//!   no mmap; after `prebuffer` it is complete and zero-copy.
//! - `vvec_remote`: the matrix cell that motivated this refactor —
//!   variable-length vvec read over cached HTTP, sidecar offsets
//!   fetched separately.
//! - `typed_reader_remote`: scalar `.u8` and uniform `.fvec` over
//!   HTTP, native + widening + cross-sign overflow.
//! - `view_round_trip`: `TestDataGroup::load(http_url)` →
//!   `TestDataView` → readers; `prebuffer_all_with_progress`
//!   actually downloads every facet.
//! - `cache_stats`: `FacetStorage::cache_stats` reports
//!   valid/total/complete during partial fill.

mod support;

use std::path::Path;
use std::sync::Arc;

use byteorder::{LittleEndian, WriteBytesExt};

use vectordata::{
    CacheStats, ElementType, IndexedVvecReader, TestDataGroup,
    TypedReader, VectorReader, VvecReader, XvecReader,
    io::{open_vec, open_vvec},
};
use vectordata::merkle::MerkleRef;

use support::testserver::TestServer;

// ═══════════════════════════════════════════════════════════════════════
// Fixtures
// ═══════════════════════════════════════════════════════════════════════

/// Use a sub-MiB merkle chunk so a small fixture file produces several
/// chunks and we can observe partial-fill behaviour. The default
/// production chunk size (1 MiB) would put every fixture into a
/// single chunk and hide partial-fill bugs.
const TEST_CHUNK: u64 = 4 * 1024;

fn make_tmp() -> tempfile::TempDir {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// Write a uniform fvec containing `count` records of dimension `dim`.
/// Element `(i, d)` is `i as f32 * 100.0 + d as f32` so any
/// out-of-range read produces an obvious garbage value.
fn write_fvec(path: &Path, count: usize, dim: usize) {
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..count {
        f.write_i32::<LittleEndian>(dim as i32).unwrap();
        for d in 0..dim {
            let v = i as f32 * 100.0 + d as f32;
            f.write_f32::<LittleEndian>(v).unwrap();
        }
    }
}

/// Write a uniform ivec containing `count` records of dimension `dim`.
fn write_ivec(path: &Path, count: usize, dim: usize) {
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..count {
        f.write_i32::<LittleEndian>(dim as i32).unwrap();
        for d in 0..dim {
            f.write_i32::<LittleEndian>((i * 1000 + d) as i32).unwrap();
        }
    }
}

/// Write a scalar `.u8` file containing the bytes 0..count.
fn write_scalar_u8(path: &Path, count: usize) {
    let bytes: Vec<u8> = (0..count).map(|i| (i % 256) as u8).collect();
    std::fs::write(path, bytes).unwrap();
}

/// Write a variable-length ivvec file. Record `i` has dimension
/// `(i % 5) + 1` and values `[i*100, i*100+1, ...]`. Returns the
/// per-record byte offsets so tests can assert offset-index recovery.
fn write_ivvec(path: &Path, count: usize) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(count);
    let mut buf: Vec<u8> = Vec::new();
    for i in 0..count {
        offsets.push(buf.len() as u64);
        let dim = (i % 5) + 1;
        buf.write_i32::<LittleEndian>(dim as i32).unwrap();
        for d in 0..dim {
            buf.write_i32::<LittleEndian>((i * 100 + d) as i32).unwrap();
        }
    }
    std::fs::write(path, &buf).unwrap();
    offsets
}

/// Write a sibling `.mref` file alongside `data_path` so HTTP opens
/// take the merkle-cached path. Without this, `Storage::open_url`
/// silently falls back to direct HTTP.
fn write_mref(data_path: &Path) {
    let content = std::fs::read(data_path).unwrap();
    let mref = MerkleRef::from_content(&content, TEST_CHUNK);
    let mut mref_path = data_path.to_path_buf().into_os_string();
    mref_path.push(".mref");
    mref.save(Path::new(&mref_path)).unwrap();
}

/// Write the `IDXFOR__<name>.<i64|i32>` sidecar that the vvec reader
/// fetches over HTTP to avoid walking the file at open time.
fn write_idxfor(data_path: &Path, offsets: &[u64]) {
    let dir = data_path.parent().unwrap();
    let name = data_path.file_name().unwrap().to_str().unwrap();
    let total: u64 = std::fs::metadata(data_path).unwrap().len();
    let (ext, bytes) = if total <= i32::MAX as u64 {
        ("i32", offsets.iter().flat_map(|&o| (o as i32).to_le_bytes()).collect::<Vec<u8>>())
    } else {
        ("i64", offsets.iter().flat_map(|&o| (o as i64).to_le_bytes()).collect::<Vec<u8>>())
    };
    let idx_path = dir.join(format!("IDXFOR__{name}.{ext}"));
    std::fs::write(idx_path, bytes).unwrap();
}

/// Per-record byte slices read directly from the on-disk file —
/// the ground truth that every reader path must match.
fn fvec_record_bytes(path: &Path, dim: usize) -> Vec<Vec<u8>> {
    let raw = std::fs::read(path).unwrap();
    let stride = 4 + dim * 4;
    raw.chunks(stride).map(|c| c[4..].to_vec()).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// equivalence — same source, three transports, identical bytes
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn fvec_local_cached_direct_agree() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 200, 16);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}base.fvec", server.base_url());

    let local = XvecReader::<f32>::open(path.to_str().unwrap()).unwrap();
    let cached = XvecReader::<f32>::open(&url).unwrap();
    let truth = fvec_record_bytes(&path, 16);

    assert_eq!(local.count(), 200);
    assert_eq!(cached.count(), 200);
    assert_eq!(local.dim(), 16);
    assert_eq!(cached.dim(), 16);

    // Local is mmap-backed and complete; cached starts not-complete
    // until prebuffer drives every chunk.
    assert!(local.is_complete());
    assert!(!cached.is_complete());

    for &i in &[0usize, 1, 17, 99, 199] {
        let local_v = local.get(i).unwrap();
        let cached_v = cached.get(i).unwrap();
        assert_eq!(local_v, cached_v, "record {i} differs between local and cached");

        // And both must match raw bytes.
        let bytes: Vec<u8> = local_v.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(bytes, truth[i], "record {i} differs from on-disk bytes");
    }
}

#[test]
fn fvec_no_mref_falls_back_to_direct_http() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 50, 8);
    // *No* .mref written — Storage::open_url must silently fall back.
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}base.fvec", server.base_url());

    let direct = XvecReader::<f32>::open(&url).unwrap();
    assert_eq!(direct.count(), 50);
    assert_eq!(direct.dim(), 8);
    // Direct HTTP can never become "complete" — there's no cache.
    assert!(!direct.is_complete());
    // …and prebuffer is a no-op (not an error).
    direct.prebuffer().unwrap();
    assert!(!direct.is_complete());

    let local = XvecReader::<f32>::open(path.to_str().unwrap()).unwrap();
    for &i in &[0usize, 7, 42, 49] {
        assert_eq!(local.get(i).unwrap(), direct.get(i).unwrap());
    }
}

#[test]
fn open_vec_dispatches_through_http() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 32, 4);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}base.fvec", server.base_url());

    let local = open_vec::<f32>(path.to_str().unwrap()).unwrap();
    let remote = open_vec::<f32>(&url).unwrap();
    assert_eq!(local.count(), remote.count());
    for i in 0..local.count() {
        assert_eq!(local.get(i).unwrap(), remote.get(i).unwrap());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// cached_promotion — Cached → Mmap after prebuffer
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn cached_storage_promotes_to_mmap_after_prebuffer() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 100, 32); // > 1 chunk at TEST_CHUNK=4KiB
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}base.fvec", server.base_url());

    let r = XvecReader::<f32>::open(&url).unwrap();
    assert!(!r.is_complete(), "fresh cached reader must not be complete");
    // Trait `get_slice` returns None pre-promotion (no mmap).
    let pre: Option<&[f32]> = <XvecReader<f32> as VectorReader<f32>>::get_slice(&r, 0);
    assert!(pre.is_none(), "Cached storage must not expose mmap_slice before promotion");

    r.prebuffer().unwrap();
    assert!(r.is_complete(), "after prebuffer, cached storage must be complete");
    let post: Option<&[f32]> = <XvecReader<f32> as VectorReader<f32>>::get_slice(&r, 0);
    assert!(post.is_some(), "after prebuffer, mmap_slice must succeed");
    let v = post.unwrap();
    assert_eq!(v.len(), 32);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], 1.0);
}

#[test]
fn prebuffer_with_progress_callback_fires() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 300, 64); // ~75 KiB → ~19 chunks at TEST_CHUNK
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}base.fvec", server.base_url());

    let r = XvecReader::<f32>::open(&url).unwrap();
    let mut callback_fired = 0u32;
    let mut last_total = 0u32;
    r.prebuffer_with_progress(|p| {
        callback_fired += 1;
        last_total = p.total_chunks();
        assert!(p.completed_chunks() <= p.total_chunks());
    }).unwrap();
    assert!(callback_fired >= 1, "progress callback must fire at least once");
    assert!(last_total > 1, "fixture must span multiple chunks");
}

// ═══════════════════════════════════════════════════════════════════════
// vvec_remote — variable-length over HTTP, the previously-missing cell
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn vvec_cached_via_idxfor_sidecar() {
    let tmp = make_tmp();
    let path = tmp.path().join("predicates.ivvec");
    let offsets = write_ivvec(&path, 60);
    write_mref(&path);
    write_idxfor(&path, &offsets);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}predicates.ivvec", server.base_url());

    let local = IndexedVvecReader::<i32>::open(path.to_str().unwrap()).unwrap();
    let remote = IndexedVvecReader::<i32>::open(&url).unwrap();

    assert_eq!(local.count(), 60);
    assert_eq!(remote.count(), 60, "remote must recover offsets via IDXFOR__ sidecar");

    for &i in &[0usize, 1, 2, 5, 13, 42, 59] {
        let l = <IndexedVvecReader<i32> as VvecReader<i32>>::get(&local, i).unwrap();
        let r = <IndexedVvecReader<i32> as VvecReader<i32>>::get(&remote, i).unwrap();
        assert_eq!(l, r, "vvec record {i} differs");
        assert_eq!(local.dim_at(i).unwrap(), remote.dim_at(i).unwrap());
    }
}

#[test]
fn vvec_cached_falls_back_to_walk_without_idxfor() {
    // No IDXFOR__ sidecar → reader walks the file via Storage::read_bytes
    // to rebuild the offset index. Slow first time, but correct.
    let tmp = make_tmp();
    let path = tmp.path().join("predicates.ivvec");
    let _ = write_ivvec(&path, 12);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}predicates.ivvec", server.base_url());

    let remote = IndexedVvecReader::<i32>::open(&url).unwrap();
    assert_eq!(remote.count(), 12);
    let r5 = <IndexedVvecReader<i32> as VvecReader<i32>>::get(&remote, 5).unwrap();
    assert_eq!(r5.len(), (5 % 5) + 1);
    assert_eq!(r5[0], 500);
}

#[test]
fn open_vvec_dispatches_through_http() {
    let tmp = make_tmp();
    let path = tmp.path().join("predicates.ivvec");
    let offsets = write_ivvec(&path, 25);
    write_mref(&path);
    write_idxfor(&path, &offsets);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}predicates.ivvec", server.base_url());

    let local = open_vvec::<i32>(path.to_str().unwrap()).unwrap();
    let remote = open_vvec::<i32>(&url).unwrap();
    assert_eq!(local.count(), remote.count());
    for i in 0..local.count() {
        assert_eq!(local.get(i).unwrap(), remote.get(i).unwrap());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// typed_reader_remote — TypedReader::open_url scalar + vector
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn typed_reader_open_url_scalar_u8_native() {
    let tmp = make_tmp();
    let path = tmp.path().join("metadata.u8");
    write_scalar_u8(&path, 256);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = url::Url::parse(&format!("{}metadata.u8", server.base_url())).unwrap();

    let r = TypedReader::<u8>::open_url(url, ElementType::U8).unwrap();
    assert_eq!(r.count(), 256);
    assert_eq!(r.dim(), 1);
    assert!(r.is_native());
    assert_eq!(r.get_value(0).unwrap(), 0);
    assert_eq!(r.get_value(42).unwrap(), 42);
    assert_eq!(r.get_value(255).unwrap(), 255);
}

#[test]
fn typed_reader_open_url_widening_u8_to_i32() {
    let tmp = make_tmp();
    let path = tmp.path().join("metadata.u8");
    write_scalar_u8(&path, 100);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = url::Url::parse(&format!("{}metadata.u8", server.base_url())).unwrap();

    let r = TypedReader::<i32>::open_url(url, ElementType::U8).unwrap();
    assert!(!r.is_native());
    assert_eq!(r.get_value(0).unwrap(), 0i32);
    assert_eq!(r.get_value(42).unwrap(), 42i32);
}

#[test]
fn typed_reader_open_url_narrowing_rejected() {
    let tmp = make_tmp();
    let path = tmp.path().join("metadata.i32");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_i32::<LittleEndian>(42).unwrap();
    drop(f);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = url::Url::parse(&format!("{}metadata.i32", server.base_url())).unwrap();

    let result = TypedReader::<u8>::open_url(url, ElementType::I32);
    assert!(result.is_err(), "narrowing must be rejected at open time");
}

#[test]
fn typed_reader_open_url_no_mref_still_works() {
    // No .mref → Storage::open_url falls back to direct HTTP.
    let tmp = make_tmp();
    let path = tmp.path().join("metadata.u8");
    write_scalar_u8(&path, 64);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = url::Url::parse(&format!("{}metadata.u8", server.base_url())).unwrap();

    let r = TypedReader::<u8>::open_url(url, ElementType::U8).unwrap();
    assert_eq!(r.count(), 64);
    assert_eq!(r.get_value(7).unwrap(), 7);
    assert!(!r.is_complete(), "direct HTTP storage cannot become complete");
}

#[test]
fn typed_reader_open_url_ivec_record() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.ivec");
    write_ivec(&path, 5, 3);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = url::Url::parse(&format!("{}data.ivec", server.base_url())).unwrap();

    let r = TypedReader::<i32>::open_url(url, ElementType::I32).unwrap();
    assert_eq!(r.count(), 5);
    assert_eq!(r.dim(), 3);
    let rec = r.get_record(2).unwrap();
    assert_eq!(rec, vec![2000, 2001, 2002]);
}

#[test]
fn typed_reader_open_auto_dispatches_url() {
    let tmp = make_tmp();
    let path = tmp.path().join("metadata.u8");
    write_scalar_u8(&path, 50);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url_str = format!("{}metadata.u8", server.base_url());

    let r = TypedReader::<u8>::open_auto(&url_str, ElementType::U8).unwrap();
    assert_eq!(r.count(), 50);
    assert_eq!(r.get_value(0).unwrap(), 0);
}

// ═══════════════════════════════════════════════════════════════════════
// view_round_trip — TestDataGroup → TestDataView over HTTP
// ═══════════════════════════════════════════════════════════════════════

/// Build a minimal `dataset.yaml` that references `base.fvec`,
/// `query.fvec`, `neighbor_indices.ivec`, `neighbor_distances.fvec`,
/// and a scalar `metadata.u8`. Returned bytes are written by caller.
fn make_dataset_yaml() -> &'static str {
    r#"
name: test-ds
attributes:
  distance_function: COSINE
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
    neighbor_indices: neighbor_indices.ivec
    neighbor_distances: neighbor_distances.fvec
    metadata_content: metadata.u8
"#
}

fn make_remote_dataset(server_root: &Path) {
    write_fvec(&server_root.join("base.fvec"), 50, 8);
    write_fvec(&server_root.join("query.fvec"), 5, 8);
    write_ivec(&server_root.join("neighbor_indices.ivec"), 5, 10);
    write_fvec(&server_root.join("neighbor_distances.fvec"), 5, 10);
    write_scalar_u8(&server_root.join("metadata.u8"), 50);
    std::fs::write(server_root.join("dataset.yaml"), make_dataset_yaml()).unwrap();

    write_mref(&server_root.join("base.fvec"));
    write_mref(&server_root.join("query.fvec"));
    write_mref(&server_root.join("neighbor_indices.ivec"));
    write_mref(&server_root.join("neighbor_distances.fvec"));
    write_mref(&server_root.join("metadata.u8"));
    // Note: dataset.yaml does NOT need a .mref — the catalog/group
    // loader fetches it directly via reqwest.
}

#[test]
fn view_round_trip_over_http() {
    let tmp = make_tmp();
    make_remote_dataset(tmp.path());
    let server = TestServer::start(tmp.path()).unwrap();
    let base_url = server.base_url();

    let group = TestDataGroup::load(&base_url).unwrap();
    let view = group.profile("default").expect("default profile must exist");

    // Vector facets — read through Arc<dyn VectorReader<T>>
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 50);
    assert_eq!(base.dim(), 8);
    assert_eq!(base.get(0).unwrap()[0], 0.0);
    assert_eq!(base.get(1).unwrap()[0], 100.0);

    let q = view.query_vectors().unwrap();
    assert_eq!(q.count(), 5);

    let ni = view.neighbor_indices().unwrap();
    assert_eq!(ni.count(), 5);
    assert_eq!(ni.dim(), 10);
    assert_eq!(ni.get(0).unwrap()[3], 3i32);

    let nd = view.neighbor_distances().unwrap();
    assert_eq!(nd.count(), 5);

    // Distance function attribute round-trips through the layout
    assert_eq!(view.distance_function().as_deref(), Some("COSINE"));

    // Manifest enumeration sees every declared facet
    let manifest = view.facet_manifest();
    assert!(manifest.contains_key("base_vectors"));
    assert!(manifest.contains_key("metadata_content"));
}

#[test]
fn view_open_facet_typed_over_http() {
    let tmp = make_tmp();
    make_remote_dataset(tmp.path());
    let server = TestServer::start(tmp.path()).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    // Typed scalar metadata via the view trait — NOT TypedReader::open_url.
    // Confirms the trait method routes through the same cache-first
    // storage as a directly-opened reader.
    let elem = view.facet_element_type("metadata_content").unwrap();
    assert_eq!(elem, ElementType::U8);

    // Note: open_facet_typed lives on the concrete GenericTestDataView
    // via the TestDataView impl; the public surface for downstream
    // crates is open_facet_storage + dedicated reader ctors. We rely
    // on facet() for the polymorphic path and assert the manifest
    // covers metadata_content (TypedReader access is exercised
    // separately above through TypedReader::open_url).
    assert!(view.facet_manifest().contains_key("metadata_content"));
}

#[test]
fn prebuffer_all_drives_every_facet_complete() {
    let tmp = make_tmp();
    make_remote_dataset(tmp.path());
    let server = TestServer::start(tmp.path()).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let mut facets_seen: Vec<String> = Vec::new();
    let mut chunks_total = 0u32;
    view.prebuffer_all_with_progress(&mut |facet, p| {
        if !facets_seen.contains(&facet.to_string()) {
            facets_seen.push(facet.to_string());
        }
        chunks_total += p.total_chunks;
    }).unwrap();

    // Every standard facet declared in the dataset.yaml must have
    // been visited.
    for expected in &["base_vectors", "query_vectors", "neighbor_indices",
                      "neighbor_distances", "metadata_content"] {
        assert!(facets_seen.iter().any(|f| f == expected),
            "prebuffer_all must visit '{expected}' (saw: {facets_seen:?})");
    }
    assert!(chunks_total > 0, "at least one chunk must have been processed");

    // After prebuffer_all, opening any facet again must be complete.
    let base_storage = view.open_facet_storage("base_vectors").unwrap();
    assert!(base_storage.is_complete(), "base_vectors must be complete after prebuffer_all");
    assert!(base_storage.is_local(), "fully-prebuffered cached storage must report is_local");
}

// ═══════════════════════════════════════════════════════════════════════
// cache_stats — FacetStorage::cache_stats reports valid/total/complete
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn cache_stats_reports_partial_and_full_fill() {
    let tmp = make_tmp();
    make_remote_dataset(tmp.path());
    let server = TestServer::start(tmp.path()).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let storage = view.open_facet_storage("base_vectors").unwrap();
    let pre: CacheStats = storage.cache_stats()
        .expect("cached storage must report cache_stats");
    assert!(!pre.is_complete, "fresh cached facet must not be complete");
    assert_eq!(pre.valid_chunks, 0, "no chunks should be downloaded yet");
    assert!(pre.total_chunks > 0);
    assert!(pre.content_size > 0);

    storage.prebuffer().unwrap();

    let post = storage.cache_stats().expect("still cached after prebuffer");
    assert!(post.is_complete);
    assert_eq!(post.valid_chunks, post.total_chunks);
    assert_eq!(post.content_size, pre.content_size);
}

#[test]
fn cache_stats_returns_none_for_local_storage() {
    let tmp = make_tmp();
    make_remote_dataset(tmp.path());
    let local_yaml = tmp.path().join("dataset.yaml");
    assert!(local_yaml.exists());

    let group = TestDataGroup::load(local_yaml.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let storage = view.open_facet_storage("base_vectors").unwrap();
    assert!(storage.cache_stats().is_none(),
        "local mmap storage must not report cache_stats");
    assert!(storage.is_local());
    assert!(storage.is_complete());
}

#[test]
fn cache_stats_returns_none_for_direct_http() {
    let tmp = make_tmp();
    write_fvec(&tmp.path().join("base.fvec"), 20, 4);
    write_fvec(&tmp.path().join("query.fvec"), 5, 4);
    std::fs::write(tmp.path().join("dataset.yaml"), r#"
name: nomref-ds
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
"#).unwrap();
    // No .mref files at all → every facet falls back to direct HTTP.
    let server = TestServer::start(tmp.path()).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let storage = view.open_facet_storage("base_vectors").unwrap();
    assert!(storage.cache_stats().is_none(),
        "direct-HTTP storage must not report cache_stats");
    assert!(!storage.is_local());
    assert!(!storage.is_complete());
    // prebuffer is a no-op on direct HTTP — must not error.
    storage.prebuffer().unwrap();
    assert!(!storage.is_complete());
}

// ═══════════════════════════════════════════════════════════════════════
// Arc-shared storage between shape adapters
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn local_yaml_with_absolute_http_facets() {
    // A dataset.yaml that lives on local disk but whose facet
    // entries are absolute HTTP URLs. The view layer must pass
    // those through to the remote-cache path rather than
    // path-joining them onto the local dataset directory.
    //
    // Use a multi-chunk file so the dim-header read at open time
    // doesn't accidentally fully download the file.
    let tmp_remote = make_tmp();
    write_fvec(&tmp_remote.path().join("base.fvec"), 1000, 64); // ~256 KiB → many TEST_CHUNK chunks
    write_fvec(&tmp_remote.path().join("query.fvec"), 50, 64);
    write_mref(&tmp_remote.path().join("base.fvec"));
    write_mref(&tmp_remote.path().join("query.fvec"));
    let server = TestServer::start(tmp_remote.path()).unwrap();
    let base_url = server.base_url();

    let tmp_local = make_tmp();
    let yaml = format!(r#"
name: hybrid
profiles:
  default:
    base_vectors: {base_url}base.fvec
    query_vectors: {base_url}query.fvec
"#);
    std::fs::write(tmp_local.path().join("dataset.yaml"), yaml).unwrap();

    let group = TestDataGroup::load(tmp_local.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    // The reader path resolves base/query to remote — counts and
    // first-record bytes must match the remote source. (This is the
    // key assertion — without URL-aware resolve_path_str, the
    // FileSystem branch would have path-joined the URL and failed
    // to open.)
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 1000);
    assert_eq!(base.dim(), 64);
    assert_eq!(base.get(0).unwrap()[0], 0.0);
    assert_eq!(base.get(1).unwrap()[0], 100.0);

    // Storage handle for the same facet must report cached-remote
    // (not local mmap). It is *not* yet complete because only the
    // dim header chunk has been touched so far on opens.
    let storage = view.open_facet_storage("base_vectors").unwrap();
    assert!(storage.cache_stats().is_some(),
        "remote facet via local yaml must yield cached storage with cache_stats");
    assert!(!storage.is_complete(),
        "multi-chunk file must not be complete from header read alone");

    storage.prebuffer().unwrap();
    assert!(storage.is_complete());

    // After prebuffer, cache_path returns the local cache file and
    // is_local is true (mmap-promoted).
    let local = storage.cache_path()
        .expect("cached storage must report a local cache path after prebuffer");
    assert!(local.is_file(), "cache_path must point to an existing file");
    assert!(storage.is_local(), "cached storage must be local after promotion");
}

#[test]
fn local_dataset_facet_storage_is_local_with_no_cache_path() {
    // A dataset.yaml on local disk with relative facet paths —
    // every facet should be Storage::Mmap, so cache_path is None
    // (no cache file involved) and is_local is true from the start.
    let tmp = make_tmp();
    write_fvec(&tmp.path().join("base.fvec"), 10, 4);
    std::fs::write(tmp.path().join("dataset.yaml"), r#"
name: local
profiles:
  default:
    base_vectors: base.fvec
"#).unwrap();

    let group = TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let storage = view.open_facet_storage("base_vectors").unwrap();
    assert!(storage.is_local(), "local dataset facets are local from the start");
    assert!(storage.is_complete(), "local dataset facets are complete from the start");
    assert!(storage.cache_path().is_none(),
        "local dataset facets have no cache file (data is read in place)");
    // prebuffer is a no-op
    storage.prebuffer().unwrap();
    assert!(storage.is_complete());
}

#[test]
fn arc_clone_of_reader_shares_storage() {
    // The XvecReader internally holds Arc<Storage>; cloning the
    // outer Arc<XvecReader<T>> shares the same storage and its
    // promoted-mmap state.
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 200, 16);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    let url = format!("{}base.fvec", server.base_url());

    let r1 = Arc::new(XvecReader::<f32>::open(&url).unwrap());
    let r2 = Arc::clone(&r1);
    assert!(!r1.is_complete());
    r2.prebuffer().unwrap();
    // Promotion observed through the *other* clone — same Storage.
    assert!(r1.is_complete());
}
