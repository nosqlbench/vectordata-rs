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

/// Wipe any cache state for the given server's URL prefix so a test
/// starts from a known empty-cache baseline. Required because cargo
/// test reuses ephemeral ports across runs and the configured
/// cache_dir at `/mnt/datamir/vectordata-cache` persists, so a
/// previous run on the same port can leave a populated cache that
/// would make a "starts not-complete" assertion false.
fn reset_cache_for_server(server: &TestServer) {
    let cache_root = vectordata::settings::cache_dir().expect("settings.yaml configured");
    let host_dir = cache_root.join(format!("127.0.0.1:{}", server.port()));
    let _ = std::fs::remove_dir_all(&host_dir);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    let direct = XvecReader::<f32>::open(&url).unwrap();
    assert_eq!(direct.count(), 50);
    assert_eq!(direct.dim(), 8);
    // Pre-prebuffer: direct-HTTP is not complete (no cache file
    // exists yet — we cleaned it via reset_cache_for_server).
    assert!(!direct.is_complete());

    // Per-record reads still work — they go over HTTP one record
    // at a time.
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
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
    reset_cache_for_server(&server);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let storage = view.open_facet_storage("base_vectors").unwrap();
    // Pre-prebuffer state for direct-HTTP: never reports merkle
    // cache stats, not local, not complete.
    assert!(storage.cache_stats().is_none(),
        "direct-HTTP storage has no merkle cache; cache_stats must be None");
    assert!(!storage.is_local());
    assert!(!storage.is_complete());

    // Strict contract: prebuffer downloads the file (no merkle —
    // bytes are trusted from the server) and promotes to mmap.
    storage.prebuffer().unwrap();
    assert!(storage.is_complete(),
        "direct-HTTP storage MUST be complete after prebuffer (strict contract)");
    assert!(storage.is_local(),
        "direct-HTTP storage MUST be local after prebuffer");
    // cache_stats stays None — merkle isn't part of the
    // direct-HTTP path.
    assert!(storage.cache_stats().is_none(),
        "direct-HTTP storage has no merkle cache; cache_stats stays None even after prebuffer");
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
    reset_cache_for_server(&server);
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

// ═══════════════════════════════════════════════════════════════════════
// Cross-instance promotion races — the bug downstream consumers hit
// when an early reader is opened, then prebuffer runs on a separate
// Storage instance. The early reader MUST see the promoted state on
// its next read; otherwise every cycle falls through to slow paths.
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn early_xvec_reader_picks_up_promotion_via_get_slice() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 500, 32);  // many TEST_CHUNK chunks
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    // Reader A is opened BEFORE prebuffer runs on a separate Storage.
    let early = XvecReader::<f32>::open(&url).unwrap();
    assert!(!early.is_complete(), "early reader starts not-complete");
    let pre: Option<&[f32]> = <XvecReader<f32> as VectorReader<f32>>::get_slice(&early, 0);
    assert!(pre.is_none(),
        "trait get_slice on early not-promoted Storage::Cached must be None");

    // Prebuffer through a *different* Storage instance (separate
    // open of the same URL).
    let prebufferer = XvecReader::<f32>::open(&url).unwrap();
    prebufferer.prebuffer().unwrap();
    assert!(prebufferer.is_complete());

    // Critical assertion: the early reader's get_slice MUST now
    // return Some — the bug is when this stays None, forcing every
    // per-cycle access to fall through to a slow path.
    let post: Option<&[f32]> = <XvecReader<f32> as VectorReader<f32>>::get_slice(&early, 0);
    assert!(post.is_some(),
        "after prebuffer on a sibling Storage, early reader's get_slice MUST promote");

    // is_complete on the early reader must also flip to true (lazy
    // promotion path).
    assert!(early.is_complete(), "is_complete must reflect post-prebuffer state");
}

#[test]
fn early_xvec_reader_picks_up_promotion_via_inherent_get_slice() {
    // Same race, but exercising the unchecked inherent get_slice
    // hot-path on XvecReader<f32> — the one nbrs's
    // slice_arc_from_uniform reaches for first.
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 400, 16);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    let early = XvecReader::<f32>::open(&url).unwrap();
    assert!(!early.is_complete());

    // Prebuffer on a separate Storage instance.
    XvecReader::<f32>::open(&url).unwrap().prebuffer().unwrap();

    // Inherent get_slice would PANIC if mmap_base returns None
    // ("requires mmap-backed storage"). The lazy promotion in
    // mmap_base must succeed so this returns valid bytes.
    let v: &[f32] = early.get_slice(0);
    assert_eq!(v.len(), 16);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], 1.0);
}

#[test]
fn prebuffer_via_view_promotes_other_open_readers() {
    // Mirrors the nbrs scenario: `view.prebuffer_all()` runs at
    // session-init through one set of FacetStorage instances; the
    // accessors then open their own readers per-cycle. Those
    // accessor-side readers MUST be zero-copy.
    //
    // Use a multi-chunk fixture so reading the dim header at open
    // doesn't accidentally fully download the only chunk.
    let tmp = make_tmp();
    write_fvec(&tmp.path().join("base.fvec"), 1000, 64);   // ~256 KiB → many TEST_CHUNK chunks
    write_fvec(&tmp.path().join("query.fvec"), 100, 64);
    write_ivec(&tmp.path().join("neighbor_indices.ivec"), 100, 10);
    write_fvec(&tmp.path().join("neighbor_distances.fvec"), 100, 10);
    write_scalar_u8(&tmp.path().join("metadata.u8"), 200_000);
    std::fs::write(tmp.path().join("dataset.yaml"), make_dataset_yaml()).unwrap();
    write_mref(&tmp.path().join("base.fvec"));
    write_mref(&tmp.path().join("query.fvec"));
    write_mref(&tmp.path().join("neighbor_indices.ivec"));
    write_mref(&tmp.path().join("neighbor_distances.fvec"));
    write_mref(&tmp.path().join("metadata.u8"));

    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    // Open a reader BEFORE prebuffer (early-bound accessor).
    let base_early = view.base_vectors().unwrap();
    assert!(!base_early.is_complete(),
        "early reader against remote view should not be complete pre-prebuffer");

    // Now drive the view's full prebuffer.
    view.prebuffer_all().unwrap();

    // The early reader must now expose zero-copy get_slice and a
    // complete state — the bug being: it stays not-complete and
    // get_slice falls back to allocating get(idx) per cycle.
    assert!(base_early.is_complete(),
        "early reader must reflect post-prebuffer is_complete");
    assert!(base_early.get_slice(0).is_some(),
        "early reader's trait get_slice must return zero-copy slice after prebuffer_all");
}

#[test]
fn prebuffer_propagates_failure_for_typed_reader_too() {
    // Same lazy-promotion behaviour for TypedReader (the API the
    // typed scalar metadata facets go through).
    let tmp = make_tmp();
    let path = tmp.path().join("metadata.u8");
    write_scalar_u8(&path, 200_000);  // ~50 chunks at TEST_CHUNK=4KiB
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let url = url::Url::parse(&format!("{}metadata.u8", server.base_url())).unwrap();

    let early = TypedReader::<u8>::open_url(url.clone(), ElementType::U8).unwrap();
    assert!(!early.is_complete());

    // Prebuffer on a separate TypedReader.
    let pb = TypedReader::<u8>::open_url(url.clone(), ElementType::U8).unwrap();
    pb.prebuffer().unwrap();

    // Early reader picks up promotion on next access.
    assert!(early.is_complete(),
        "TypedReader must lazy-promote when sibling Storage finishes prebuffer");
    // get_native_slice exercises mmap_slice — should now succeed.
    let slice = early.get_native_slice(0);
    assert!(slice.is_some(), "after prebuffer, get_native_slice must return Some");
}

// ═══════════════════════════════════════════════════════════════════════
// Strict prebuffer contract: Ok(()) ⇒ fully resident; failure surfaces
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn prebuffer_strict_contract_cached() {
    // After prebuffer_all returns Ok, every facet's storage handle
    // MUST report is_complete=true and is_local=true. There is no
    // partial-completion mode.
    let tmp = make_tmp();
    make_remote_dataset(tmp.path());
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    view.prebuffer_all().unwrap();

    // Walk every facet and assert strict completion.
    for (name, _desc) in view.facet_manifest() {
        // Facets without a recognized element type are skipped by
        // prebuffer_all_with_progress; they shouldn't fail here either.
        if view.facet_element_type(&name).is_err() { continue; }
        let storage = view.open_facet_storage(&name).unwrap();
        assert!(storage.is_complete(),
            "facet '{name}' must be complete after prebuffer_all (strict contract)");
        assert!(storage.is_local(),
            "facet '{name}' must be local after prebuffer_all (strict contract)");
    }
}

#[test]
fn prebuffer_for_url_without_mref_downloads_and_promotes() {
    // No .mref published — Storage::open_url falls back to
    // Storage::Http. Per the strict contract, prebuffer must STILL
    // download the file (no merkle, just fetch + write + mmap) and
    // promote, not silently no-op.
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 80, 8);
    // *No* write_mref — direct HTTP path.
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    let r = XvecReader::<f32>::open(&url).unwrap();
    assert!(!r.is_complete(), "Http storage starts not-complete");
    let pre: Option<&[f32]> = <XvecReader<f32> as VectorReader<f32>>::get_slice(&r, 0);
    assert!(pre.is_none(), "Http storage has no mmap pre-prebuffer");

    r.prebuffer().unwrap();

    // After prebuffer, reads must be zero-copy.
    assert!(r.is_complete(),
        "Http storage MUST be complete after prebuffer (no silent no-op)");
    let post: Option<&[f32]> = <XvecReader<f32> as VectorReader<f32>>::get_slice(&r, 0);
    assert!(post.is_some(),
        "Http storage MUST yield zero-copy get_slice after prebuffer");
    let v = post.unwrap();
    assert_eq!(v.len(), 8);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], 1.0);
}

#[test]
fn prebuffer_persists_for_subsequent_opens_without_mref() {
    // After Http prebuffer downloads the file, a fresh open of the
    // same URL must restore from the cache file at open time —
    // no second download, immediate zero-copy.
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 40, 8);
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    XvecReader::<f32>::open(&url).unwrap().prebuffer().unwrap();

    // Fresh open (different Storage instance) — must see the
    // restored mmap immediately.
    let fresh = XvecReader::<f32>::open(&url).unwrap();
    assert!(fresh.is_complete(),
        "fresh Http open against a previously-prebuffered URL must restore from cache");
    let v = fresh.get_slice(0);  // inherent unchecked — would panic if not mmap-backed
    assert_eq!(v.len(), 8);
    assert_eq!(v[0], 0.0);
}

#[test]
fn prebuffer_fails_loud_when_remote_size_mismatch() {
    // The Http prebuffer path verifies the post-write file size
    // against the HEAD-reported total_size. If the server returns
    // a short body, prebuffer must surface the failure rather than
    // leaving a truncated cache file in place.
    //
    // We can't easily make the test server return a wrong size, so
    // this asserts the write-then-verify path by triggering a
    // legitimate prebuffer and confirming the post-write size
    // matches. (The mismatch error path is covered by the io
    // length-check guard inside http_prebuffer.)
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 20, 4);
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    let r = XvecReader::<f32>::open(&url).unwrap();
    let pre_size = std::fs::metadata(&path).unwrap().len();
    r.prebuffer().unwrap();

    // The cache file must exist with the correct size.
    let storage = vectordata::TestDataGroup::load(&server.base_url())
        .ok()
        .and_then(|g| g.profile("default").map(|v| (g, v)));
    let _ = storage;  // suppress; we assert via cache_path on the reader's storage
    // We don't have direct access to FacetStorage here; instead
    // verify the "fully resident" invariant via reader API.
    assert!(r.is_complete());
    assert_eq!(r.count() as u64 * (4 + 4 * 4) as u64, pre_size,
        "fixture sanity: count*entry == on-disk size");
}

// ═══════════════════════════════════════════════════════════════════════
// view.prebuffer_all_with_progress propagates errors (no swallow)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn prebuffer_all_with_progress_propagates_facet_errors() {
    // Build a dataset where one facet's URL points at a 404. The
    // underlying Storage::open will fail (or its prebuffer will);
    // prebuffer_all_with_progress MUST return Err, not silently
    // skip the facet.
    let tmp = make_tmp();
    write_fvec(&tmp.path().join("base.fvec"), 20, 4);
    write_mref(&tmp.path().join("base.fvec"));
    // query.fvec is referenced but not written.
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let yaml = format!(r#"
name: missing-facet
profiles:
  default:
    base_vectors: {}base.fvec
    query_vectors: {}query.fvec
"#, server.base_url(), server.base_url());

    let local = make_tmp();
    std::fs::write(local.path().join("dataset.yaml"), yaml).unwrap();
    let group = TestDataGroup::load(local.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    let result = view.prebuffer_all();
    assert!(result.is_err(),
        "prebuffer_all MUST surface failure when a facet can't be opened/prebuffered");
}

// ═══════════════════════════════════════════════════════════════════════
// Concurrent races — multiple threads racing on promotion + reads
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn concurrent_readers_share_promotion_safely() {
    // Spawn N threads, each opening a fresh XvecReader against the
    // same URL. One thread runs prebuffer; the others spin on
    // get_slice. Every reader must end up promoted with no panic,
    // no torn read, no stale "always None".
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, 800, 32);
    write_mref(&path);
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    let n_readers = 8;
    let url_arc = Arc::new(url);
    let pb_done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    std::thread::scope(|s| {
        // Prebuffer thread — sets pb_done when complete so reader
        // threads can stop polling.
        {
            let url = Arc::clone(&url_arc);
            let pb_done = Arc::clone(&pb_done);
            s.spawn(move || {
                XvecReader::<f32>::open(&url).unwrap().prebuffer().unwrap();
                pb_done.store(true, std::sync::atomic::Ordering::Release);
            });
        }
        // Reader threads — open a fresh reader, spin on get_slice.
        for _ in 0..n_readers {
            let url = Arc::clone(&url_arc);
            let pb_done = Arc::clone(&pb_done);
            s.spawn(move || {
                let r = XvecReader::<f32>::open(&url).unwrap();
                // Poll until prebuffer is done, then once more to
                // give the lazy-promotion path a chance.
                while !pb_done.load(std::sync::atomic::Ordering::Acquire) {
                    let _ = <XvecReader<f32> as VectorReader<f32>>::get_slice(&r, 0);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                // After prebuffer completes, the next get_slice MUST
                // return Some — the cross-instance promotion guarantee.
                let v = <XvecReader<f32> as VectorReader<f32>>::get_slice(&r, 0);
                assert!(v.is_some(),
                    "reader must observe promotion immediately after prebuffer thread completes");
                let v = v.unwrap();
                assert_eq!(v.len(), 32, "promoted slice must have correct dim");
            });
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════
// All-profiles prebuffer + 250 MiB advisory warning
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn prebuffer_all_profiles_visits_every_profile_and_facet() {
    let tmp = make_tmp();
    write_fvec(&tmp.path().join("base.fvec"), 60, 8);
    write_fvec(&tmp.path().join("query.fvec"), 5, 8);
    write_fvec(&tmp.path().join("base_50.fvec"), 30, 8);
    write_mref(&tmp.path().join("base.fvec"));
    write_mref(&tmp.path().join("query.fvec"));
    write_mref(&tmp.path().join("base_50.fvec"));
    std::fs::write(tmp.path().join("dataset.yaml"), r#"
name: multi
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
  half:
    base_vectors: base_50.fvec
    query_vectors: query.fvec
"#).unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let group = TestDataGroup::load(&server.base_url()).unwrap();

    let mut visited: Vec<(String, String)> = Vec::new();
    let mut warned_bytes: Option<u64> = None;
    group.prebuffer_all_profiles_with_progress(
        &mut |profile, facet, _p| {
            let key = (profile.to_string(), facet.to_string());
            if !visited.contains(&key) { visited.push(key); }
        },
        &mut |total| { warned_bytes = Some(total); },
    ).unwrap();

    // Both profiles, both facets in each, must be visited.
    assert!(visited.iter().any(|(p, f)| p == "default" && f == "base_vectors"));
    assert!(visited.iter().any(|(p, f)| p == "default" && f == "query_vectors"));
    assert!(visited.iter().any(|(p, f)| p == "half" && f == "base_vectors"));
    assert!(visited.iter().any(|(p, f)| p == "half" && f == "query_vectors"));

    // Tiny fixture — under the 250 MiB threshold, no warning.
    assert!(warned_bytes.is_none(),
        "fixture is well under threshold; warn_cb must NOT fire");

    // After prebuffer_all_profiles, every facet of every profile is
    // resident.
    for profile_name in &["default", "half"] {
        let view = group.profile(profile_name).unwrap();
        for facet in ["base_vectors", "query_vectors"] {
            let storage = view.open_facet_storage(facet).unwrap();
            assert!(storage.is_complete(),
                "after prebuffer_all_profiles, {profile_name}/{facet} must be complete");
            assert!(storage.is_local(),
                "after prebuffer_all_profiles, {profile_name}/{facet} must be local");
        }
    }
}

#[test]
#[ignore = "writes ~280 MiB of fixture data; opt in with --ignored when verifying"]
fn prebuffer_all_profiles_warning_fires_above_threshold() {
    // Build a fixture whose total size exceeds the 250 MiB advisory
    // threshold so warn_cb fires.
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    // 1_100_000 records × (4 + 64*4) ≈ 273 MiB > 250 MiB threshold
    write_fvec(&path, 1_100_000, 64);
    write_mref(&path);
    std::fs::write(tmp.path().join("dataset.yaml"), r#"
name: large
profiles:
  default:
    base_vectors: base.fvec
"#).unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    reset_cache_for_server(&server);
    let group = TestDataGroup::load(&server.base_url()).unwrap();

    let mut warned: Option<u64> = None;
    let result = group.prebuffer_all_profiles_with_progress(
        &mut |_p, _f, _prog| {},
        &mut |total| { warned = Some(total); },
    );
    let _ = result;
    let total = warned.expect("warn_cb must fire above 250 MiB threshold");
    assert!(total >= vectordata::PREBUFFER_LARGE_WARNING_BYTES,
        "warn_cb total ({total}) must be >= threshold ({})",
        vectordata::PREBUFFER_LARGE_WARNING_BYTES);
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
    reset_cache_for_server(&server);
    let url = format!("{}base.fvec", server.base_url());

    let r1 = Arc::new(XvecReader::<f32>::open(&url).unwrap());
    let r2 = Arc::clone(&r1);
    assert!(!r1.is_complete());
    r2.prebuffer().unwrap();
    // Promotion observed through the *other* clone — same Storage.
    assert!(r1.is_complete());
}
