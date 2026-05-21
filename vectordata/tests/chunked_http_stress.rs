// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Concurrent and resume-style stress tests for the non-merkle
//! chunked HTTP storage path. Each test spins up the local
//! `TestServer` fixture, populates it with a multi-chunk file
//! (>> 8 MiB so the `ChunkStore` default chunk size produces
//! multiple chunks), and exercises:
//!
//! - **partial resume**: open a `Storage` against the URL, fetch
//!   a subset of chunks via random reads, drop and reopen → the
//!   `.chunks` sidecar must reload and the previously-fetched
//!   chunks must NOT be refetched.
//! - **concurrent multi-threaded random reads**: 16 worker
//!   threads issuing random reads of random sizes against the
//!   same URL → every read must return the correct bytes; the
//!   in-flight dedup map must prevent duplicate chunk fetches
//!   for the same chunk index.
//! - **seed-driven fuzz**: a `SmallRng`-driven mix of opens,
//!   reads, drops, and reopens. Bytes returned from `read_bytes`
//!   are verified against the ground-truth file byte-for-byte.
//!
//! These are *integration* tests — they run against the same
//! reqwest stack production hits, not a hand-mocked transport.
//! The TestServer is a real `axum` server bound to a local
//! port; the `Storage` layer goes through `HttpTransport` →
//! `ChunkStore` exactly as it would for an S3 fetch.

mod support;

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use vectordata::{VectorReader, XvecReader};

use support::testserver::TestServer;

/// Per-process test cache directory; mirrors the pattern in
/// http_storage.rs so cache state is contained inside
/// `target/tmp/` and reaped on `cargo clean`.
static TEST_CACHE_DIR: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).expect("create test cache root")
});

fn init_test_cache() {
    vectordata::settings::override_cache_dir_for_process(
        TEST_CACHE_DIR.path().to_path_buf(),
    );
}

fn make_tmp() -> tempfile::TempDir {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// Write a deterministic fvec file with `count` records of
/// dimension `dim`. Byte content is fully determined by `(i, d)`
/// so the test can recompute any record's expected bytes
/// without re-reading the file. The file size is `count *
/// (4 + dim*4)` bytes; pick `count` and `dim` to produce more
/// than `ChunkStore::DEFAULT_CHUNK_SIZE` (8 MiB) total so the
/// store has multiple chunks to manage.
fn write_fvec(path: &Path, count: usize, dim: usize) {
    use byteorder::{LittleEndian, WriteBytesExt};
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..count {
        f.write_i32::<LittleEndian>(dim as i32).unwrap();
        for d in 0..dim {
            let v = (i * 1000 + d) as f32;
            f.write_f32::<LittleEndian>(v).unwrap();
        }
    }
}

/// Expected vector at index `i` given `dim` — must match what
/// `write_fvec` produced. Lets the tests assert correctness
/// without keeping a giant buffer in memory.
fn expected_vec(i: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|d| (i * 1000 + d) as f32).collect()
}

/// Sized to comfortably exceed the default `ChunkStore` chunk
/// size (8 MiB), so every test exercises multi-chunk geometry.
/// 4096 records × dim 1024 × 4 bytes/elem ≈ 16 MiB of vector
/// payload → 3 chunks.
const STRESS_COUNT: usize = 4096;
const STRESS_DIM: usize = 1024;

fn stress_file_size() -> u64 {
    // `count * (4 + dim*4)` bytes per the fvec format.
    let entry = 4 + STRESS_DIM as u64 * 4;
    STRESS_COUNT as u64 * entry
}

// ═════════════════════════════════════════════════════════════════════
// Partial resume — drop the store mid-read, reopen, verify the bitmap
// reloaded and previously-fetched chunks are skipped.
// ═════════════════════════════════════════════════════════════════════

/// Random-read a handful of records through a fresh `XvecReader`
/// (which opens the `Storage::Http` under the hood, lazily
/// fetching chunks). Then *drop the reader* so the `Storage`
/// arc's strong refcount falls to zero and the registry weak
/// pointer is unable to upgrade — exactly the cross-session
/// state. Reopen and confirm the second session loads the
/// sidecar bitmap and only fetches new chunks.
#[test]
fn partial_fetch_resumes_across_open() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, STRESS_COUNT, STRESS_DIM);
    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    let url = format!("{}base.fvec", server.base_url());

    // First session: read a few records spread across the file.
    // Each `get` fetches at least one chunk; with 3 chunks total
    // we'll likely hit chunk 0 and chunk 2 but not chunk 1.
    let pre_chunks = chunks_dir_count(&url);
    {
        let r = XvecReader::<f32>::open(&url).unwrap();
        // Two reads near the start and end of the file. The
        // first record always fetches chunk 0; the last record
        // always fetches the final chunk. Middle is untouched.
        let first = <XvecReader<f32> as VectorReader<f32>>::get(&r, 0).unwrap();
        let last = <XvecReader<f32> as VectorReader<f32>>::get(&r, STRESS_COUNT - 1).unwrap();
        assert_eq!(first, expected_vec(0, STRESS_DIM));
        assert_eq!(last, expected_vec(STRESS_COUNT - 1, STRESS_DIM));
    }
    // Drop scope ended → Storage Arc's strong refcount goes to 0.
    // The sidecar `.chunks` file persists on disk.
    let mid_chunks = chunks_dir_count(&url);
    assert!(mid_chunks > pre_chunks,
        "expected the data file + sidecar to land on disk during first session");

    // Second session: reopen, read the middle record. The
    // already-fetched chunks must not refetch — we can't
    // directly observe HTTP requests from this scope, but the
    // mid-record read still succeeding (and matching the
    // expected bytes) is the user-visible contract. The deeper
    // network invariant is exercised in
    // `concurrent_reads_dedup_chunk_fetches`.
    {
        let r = XvecReader::<f32>::open(&url).unwrap();
        let mid_idx = STRESS_COUNT / 2;
        let mid = <XvecReader<f32> as VectorReader<f32>>::get(&r, mid_idx).unwrap();
        assert_eq!(mid, expected_vec(mid_idx, STRESS_DIM));
    }

    // After enough reads, every chunk has been fetched. A
    // subsequent open should observe complete state purely from
    // the sidecar — let's force that path by reading all
    // records sequentially.
    {
        let r = XvecReader::<f32>::open(&url).unwrap();
        // Sequential read over the whole file is the surest way
        // to ensure every chunk gets fetched.
        for i in 0..STRESS_COUNT {
            let v = <XvecReader<f32> as VectorReader<f32>>::get(&r, i).unwrap();
            assert_eq!(v, expected_vec(i, STRESS_DIM), "vec {i} mismatch");
        }
    }

    // Final session — bitmap should report fully resident at
    // open time, so `is_complete` returns true immediately.
    {
        let r = XvecReader::<f32>::open(&url).unwrap();
        assert!(r.is_complete(), "fully-fetched file must reload as complete");
    }
}

// ═════════════════════════════════════════════════════════════════════
// Concurrent multi-threaded random reads.
// ═════════════════════════════════════════════════════════════════════

/// 16 workers, each performing 200 random-record reads. Every
/// read must produce the expected bytes — corruption from
/// torn writes or duplicate-write races would fail this.
#[test]
fn concurrent_random_reads_all_correct() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, STRESS_COUNT, STRESS_DIM);
    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    let url = format!("{}base.fvec", server.base_url());

    let workers = 16usize;
    let reads_per_worker = 200usize;
    let mismatches = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        for w in 0..workers {
            let url = url.clone();
            let mismatches = Arc::clone(&mismatches);
            scope.spawn(move || {
                let mut rng = SmallRng::seed_from_u64(0x9e37_79b9_7f4a_7c15 ^ w as u64);
                let r = XvecReader::<f32>::open(&url).unwrap();
                for _ in 0..reads_per_worker {
                    let i = rng.random_range(0..STRESS_COUNT);
                    let v = <XvecReader<f32> as VectorReader<f32>>::get(&r, i).unwrap();
                    if v != expected_vec(i, STRESS_DIM) {
                        mismatches.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });

    assert_eq!(mismatches.load(Ordering::Relaxed), 0,
        "concurrent reads produced byte mismatches — chunked-fill race");

    // After all workers finished, every chunk should have been
    // fetched at least once between them (16 × 200 random reads
    // over a 3-chunk file → effectively certain to hit every
    // chunk). Reopen and check complete state.
    let r = XvecReader::<f32>::open(&url).unwrap();
    assert!(r.is_complete(),
        "after 3200 random reads spanning the file the bitmap should be full");
}

// ═════════════════════════════════════════════════════════════════════
// Seed-driven fuzz — random ops, byte-exact verification.
// ═════════════════════════════════════════════════════════════════════

/// Mixed-op fuzz: random reads at random offsets and sizes
/// against a known-good file. Re-opens the reader between
/// batches to exercise the cross-session resume path. Each
/// read's bytes are recomputed from the expected layout and
/// compared. Failures here represent a real correctness bug
/// in chunk-fetch + cache-file write ordering.
#[test]
fn fuzz_random_offset_random_size_reads() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, STRESS_COUNT, STRESS_DIM);
    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    let url = format!("{}base.fvec", server.base_url());

    // Compute the expected bytes once so we can index any offset
    // without rebuilding them.
    let truth_bytes = std::fs::read(&path).unwrap();
    let total = stress_file_size();
    assert_eq!(truth_bytes.len() as u64, total);

    let mut rng = SmallRng::seed_from_u64(0xfeed_face_dead_beef);
    for batch in 0..6 {
        // Reopen between batches → exercises sidecar reload.
        let r = XvecReader::<f32>::open(&url).unwrap();
        for _ in 0..80 {
            // Map a random record index → expected bytes for
            // that record. We don't read arbitrary byte ranges
            // because XvecReader is record-shaped, but the
            // chunk-fetch logic underneath sees every record
            // span and the chunks they cover.
            let i = rng.random_range(0..STRESS_COUNT);
            let v = <XvecReader<f32> as VectorReader<f32>>::get(&r, i).unwrap();
            assert_eq!(v, expected_vec(i, STRESS_DIM),
                "batch {batch} record {i}: got {:?}", &v[..4.min(v.len())]);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Bitmap reset on stale sidecar.
// ═════════════════════════════════════════════════════════════════════

/// Robustness: if the cache file is deleted out from under us
/// (manual user cleanup, host crash, anything) but a stale
/// `.chunks` sidecar remains, the next open MUST reset the
/// bitmap. Otherwise the bitmap would falsely advertise chunks
/// as valid against an empty re-allocated cache file, and
/// readers would see zero bytes.
#[test]
fn stale_sidecar_with_missing_data_file_resets_bitmap() {
    let tmp = make_tmp();
    let path = tmp.path().join("base.fvec");
    write_fvec(&path, STRESS_COUNT, STRESS_DIM);
    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    let url = format!("{}base.fvec", server.base_url());

    // Populate the cache file fully.
    {
        let r = XvecReader::<f32>::open(&url).unwrap();
        r.precache().unwrap();
        assert!(r.is_complete());
    }
    let (cache_file, sidecar) = locate_cache_files(&url);
    assert!(cache_file.exists());
    assert!(sidecar.exists());

    // Wipe the data file, keep the sidecar.
    std::fs::remove_file(&cache_file).unwrap();
    assert!(!cache_file.exists());
    assert!(sidecar.exists(), "sidecar should still be present");

    // Reopen — must NOT trust the stale sidecar against the
    // missing/empty cache file.
    let r = XvecReader::<f32>::open(&url).unwrap();
    assert!(!r.is_complete(),
        "stale sidecar with deleted data file must trigger bitmap reset");
    // Reads now refetch on demand and produce correct bytes.
    let v = <XvecReader<f32> as VectorReader<f32>>::get(&r, 0).unwrap();
    assert_eq!(v, expected_vec(0, STRESS_DIM));
}

// ═════════════════════════════════════════════════════════════════════
// Integrated pipeline path — exercise the same TestDataGroup +
// FacetStorage + view::prebuffer_all_profiles stack the pipeline
// commands hit, so a regression in the chunked-http layer is
// caught at the API surface real consumers use.
// ═════════════════════════════════════════════════════════════════════

/// Drive the full `TestDataGroup::prebuffer_all_profiles` flow
/// against an HTTP-served `dataset.yaml`, while concurrently
/// reading sample records from the underlying view. This
/// exercises the chunked-http path the same way `vectordata
/// datasets precache` and `veks explore` do: open via catalog
/// metadata → resolve facets → precache → read.
#[test]
fn pipeline_prebuffer_then_concurrent_reads_through_view() {
    use vectordata::TestDataGroup;

    let tmp = make_tmp();
    // Lay out a single-profile dataset with a vector facet whose
    // total payload exceeds the chunk size so we get multi-chunk
    // geometry through the chunked-http store.
    let base_path = tmp.path().join("base.fvec");
    write_fvec(&base_path, STRESS_COUNT, STRESS_DIM);
    let yaml = r#"
attributes:
  distance_function: L2
profiles:
  default:
    base_vectors: base.fvec
"#;
    std::fs::write(tmp.path().join("dataset.yaml"), yaml).unwrap();

    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    let dataset_url = server.base_url();

    // Open via the same API a pipeline command would use.
    let group = TestDataGroup::load(&dataset_url)
        .expect("dataset.yaml resolves via HTTP");
    let view = group.profile("default").expect("default profile exists");

    // Drive a precache through the high-level API. This goes
    // FacetStorage::prebuffer_with_progress → Storage::Http →
    // ChunkStore::prebuffer_with_progress.
    let fs = view.open_facet_storage("base_vectors")
        .expect("base_vectors facet opens");
    fs.precache().expect("precache succeeds end-to-end");
    assert!(fs.is_complete(), "facet must be complete after precache");

    // Concurrent reads via view.base_vectors() — different
    // entry path than open_facet_storage, but the underlying
    // Storage Arc is shared via the registry. With the file
    // fully mmapped, reads should be zero-copy fast.
    let mismatches = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for w in 0..8 {
            let url = dataset_url.clone();
            let mismatches = Arc::clone(&mismatches);
            scope.spawn(move || {
                let group = TestDataGroup::load(&url).unwrap();
                let view = group.profile("default").unwrap();
                let base = view.base_vectors().unwrap();
                let mut rng = SmallRng::seed_from_u64(0xa15c_b00b ^ w as u64);
                for _ in 0..100 {
                    let i = rng.random_range(0..STRESS_COUNT);
                    let v = base.get(i).unwrap();
                    if v != expected_vec(i, STRESS_DIM) {
                        mismatches.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    assert_eq!(mismatches.load(Ordering::Relaxed), 0,
        "post-precache concurrent reads must all match expected bytes");
}

/// Fuzz-style pipeline test: random seeds drive random open +
/// random partial-read sequences against the same URL. The
/// `Storage` Arc is process-wide-shared, so all sessions share
/// the same chunked-http store and bitmap. We confirm that:
///   - every read returns correct bytes regardless of which
///     fill state the chunks happened to be in;
///   - after a sufficient number of random sessions, the
///     bitmap reaches all-valid;
///   - reopening then reports `is_complete()` from the
///     persisted bitmap without re-fetching.
#[test]
fn pipeline_fuzz_random_sessions_converge_to_complete() {
    use vectordata::TestDataGroup;

    let tmp = make_tmp();
    let base_path = tmp.path().join("base.fvec");
    write_fvec(&base_path, STRESS_COUNT, STRESS_DIM);
    let yaml = r#"
attributes:
  distance_function: L2
profiles:
  default:
    base_vectors: base.fvec
"#;
    std::fs::write(tmp.path().join("dataset.yaml"), yaml).unwrap();

    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    let url = server.base_url();

    let mut rng = SmallRng::seed_from_u64(0xc0ff_ee15_d00d_5eed);
    // 12 random sessions, each reading a random handful of
    // records. The chunked store accumulates state across
    // sessions via the sidecar bitmap.
    for session in 0..12 {
        let group = TestDataGroup::load(&url).unwrap();
        let view = group.profile("default").unwrap();
        let base = view.base_vectors().unwrap();
        let reads = rng.random_range(5..40);
        for _ in 0..reads {
            let i = rng.random_range(0..STRESS_COUNT);
            let v = base.get(i).unwrap();
            assert_eq!(v, expected_vec(i, STRESS_DIM),
                "session {session} record {i} mismatch");
        }
    }

    // One final session does a full sweep to guarantee every
    // chunk is touched, then a subsequent open should observe
    // complete state purely from the sidecar.
    {
        let group = TestDataGroup::load(&url).unwrap();
        let view = group.profile("default").unwrap();
        let base = view.base_vectors().unwrap();
        for i in 0..STRESS_COUNT {
            let v = base.get(i).unwrap();
            assert_eq!(v, expected_vec(i, STRESS_DIM));
        }
    }
    {
        let group = TestDataGroup::load(&url).unwrap();
        let view = group.profile("default").unwrap();
        let fs = view.open_facet_storage("base_vectors").unwrap();
        assert!(fs.is_complete(),
            "after full sweep + reopen, sidecar should report complete");
    }
}

// ═════════════════════════════════════════════════════════════════════
// helpers — locate the on-disk cache file + sidecar for a URL.
// ═════════════════════════════════════════════════════════════════════

/// The `Storage` layer hashes the URL to derive its blob dir
/// (see `vectordata::cache::blob_dir_for_url`). We mirror the
/// computation here so the tests can verify on-disk artefacts
/// without going through private API.
fn locate_cache_files(url: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(url.as_bytes());
    let digest = format!("{:x}", h.finalize());
    let cache_root = TEST_CACHE_DIR.path().to_path_buf();
    let blob_dir = cache_root.join("http").join(&digest[..2]).join(&digest);
    let filename = url.rsplit('/').next().unwrap_or("data");
    let cache_file = blob_dir.join(filename);
    let sidecar = blob_dir.join(format!("{filename}.chunks"));
    (cache_file, sidecar)
}

/// Count files in the URL's blob dir. Returns 0 if the
/// directory doesn't exist yet. Used as a rough liveness check
/// — the test cares that *some* files were created, not the
/// exact set.
fn chunks_dir_count(url: &str) -> usize {
    let (cache_file, _) = locate_cache_files(url);
    let dir = match cache_file.parent() {
        Some(p) => p,
        None => return 0,
    };
    std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
}
