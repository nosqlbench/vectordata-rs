// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Internal. On-demand chunked partial-read for `Storage::Http` — the
//! `.mref`-less counterpart to [`crate::cache::CachedChannel`].
//!
//! Why this exists: a remote `Storage::Http` open against a
//! large file used to have only two read modes —
//! (a) per-call HTTP RANGE on every `read_bytes`, which is fine
//!     for tiny lookups but pathological for any consumer that
//!     reads thousands of small records (e.g. the explore TUI
//!     sampling 50k vectors → 50k round trips → 0 vec/s);
//! (b) a full [`crate::storage::Storage::precache`] that
//!     downloads the entire file before any read can proceed —
//!     prohibitive for multi-GB datasets the caller may only
//!     want to sample a slice of.
//!
//! [`ChunkStore`] fills the gap. On each `read`, it identifies
//! the fixed-size chunks covering the requested byte range,
//! fetches any chunks that aren't already present (deduping
//! concurrent fetches via per-chunk condvars, mirroring
//! `CachedChannel::ensure_chunks_valid`), writes them to a
//! sparse cache file, and records validity in an on-disk
//! sidecar bitmap. Subsequent process runs reload the bitmap
//! and only re-fetch chunks that weren't already filled.
//!
//! When every chunk has been filled, the parent `Storage::Http`
//! promotes the cache file to mmap, so subsequent reads take
//! the zero-copy path — same final state as
//! [`crate::storage::Storage::precache`] without forcing the
//! whole download up front.
//!
//! **No published tree to verify against.** The server's bytes are
//! trusted byte for byte, integrity being the transport's TLS: the
//! intentional difference from `Storage::Cached`, which needs a
//! published `.mref`. What this store does instead is *record* a tree:
//! each chunk is hashed as it lands, and the `.chunks` sidecar holds
//! the leaves and the validity bits in the `.mrkl` format. Once every
//! chunk is here the root is the file's content identity — the root a
//! publisher's `.mref` would have for the same bytes — and a cached
//! chunk can be checked against what was recorded for it
//! (`docs/design/srd-loose-files-overlays-integrity.md`, IN-7). The
//! name stays `.chunks`, so `.mrkl` keeps meaning "verified against a
//! published reference". A sidecar written before this — one byte per
//! chunk — is read as before; the leaves it lacks are hashed from the
//! cache file the first time a content identity is asked for.

use std::collections::HashMap;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::transport::{ChunkedTransport, DownloadProgress, HttpTransport};

/// Default chunk size for partial-read fills. Mirrors the
/// `MerkleRef` default that `CachedChannel` uses elsewhere so
/// the two paths produce similar over-the-wire footprints —
/// 8 MiB is the sweet spot between per-request overhead and
/// over-fetch amplification on small reads.
pub(crate) const DEFAULT_CHUNK_SIZE: u64 = 8 * 1024 * 1024;

/// Chunked partial-read store backing [`crate::storage::Storage::Http`].
///
/// Concurrency model mirrors [`crate::cache::CachedChannel`]:
/// `chunk_state` is the canonical bitmap of valid chunks
/// (lock-free reads, mutex-guarded writes), `in_flight` is the
/// per-chunk condvar table that lets concurrent readers wait
/// on an existing fetch instead of issuing duplicate requests.
pub(crate) struct ChunkStore {
    transport: HttpTransport,
    total_size: u64,
    chunk_size: u64,
    total_chunks: u32,
    cache_path: PathBuf,
    /// Persistent handle to the sparse cache file. Reads are
    /// positional (`pread` on Unix) so the hot read path doesn't
    /// re-open the file per call; chunk writes go through the same
    /// handle followed by an explicit `sync_data`.
    cache_file: crate::cache::CacheFile,
    /// Sidecar bitmap path (`<cache_path>.chunks`). Persists
    /// validity across process runs.
    chunks_path: PathBuf,
    /// One atomic byte per chunk: non-zero = valid, zero =
    /// missing. Per-chunk reads are lock-free `load(Acquire)`;
    /// stores use `store(Release)` so any thread that observes
    /// a chunk as valid is guaranteed to also observe the
    /// preceding write to the data file (which happens before
    /// the store with `sync_data` durability). Lock-free reads
    /// here are what lets us hold only the `in_flight` lock in
    /// the dedup path — no `chunk_state` lock means no inverse-
    /// order deadlock between the two.
    chunk_state: Arc<Vec<AtomicU8>>,
    /// In-flight fetch dedup. Cleared per chunk after fetch
    /// completes; condvar is signalled so waiters re-check the
    /// state bitmap. Also doubles as the serialisation point
    /// for sidecar writes — `save_sidecar` is called under this
    /// lock so concurrent finishers can't race on the rename.
    in_flight: Arc<Mutex<HashMap<u32, Arc<Condvar>>>>,
    /// Transient-failure retry policy for chunk fetches, matching the
    /// `.mref` [`crate::cache::CachedChannel`] path. A flaky connection on
    /// one chunk of a concurrent prebuffer must not fail the whole download.
    retry_policy: crate::transport::RetryPolicy,
    /// Serializes the whole-file FullTransfer download used when the
    /// server doesn't support byte ranges: the first chunk request
    /// streams the entire body; concurrent requesters block here and
    /// then observe every chunk valid.
    full_transfer: Mutex<()>,
    /// The recorded leaf hash of each chunk, as it landed; all zeros
    /// for one not recorded yet (no SHA-256 is all zeros in practice).
    leaves: Mutex<Vec<[u8; 32]>>,
    /// The validity bits as the sidecar's tree format holds them, for
    /// merging into the file on each chunk without rewriting the tree
    /// (`MerkleState::checkpoint`).
    bits: crate::merkle::MerkleState,
}

/// A leaf not recorded yet.
const UNRECORDED: [u8; 32] = [0u8; 32];

impl ChunkStore {
    /// Open a chunk store for `transport` (already pointed at
    /// the remote URL) with the given total size. Allocates the
    /// sparse cache file if absent and loads the `.chunks` bitmap
    /// sidecar if present. The sidecar is the sole source of
    /// truth for chunk validity: a data file at the expected size
    /// with no sidecar is treated as all-missing (its bytes are
    /// an unproven sparse placeholder, not a completed download),
    /// so reads never silently return zeroed regions.
    pub(crate) fn open(
        transport: HttpTransport,
        total_size: u64,
        cache_path: PathBuf,
    ) -> io::Result<Self> {
        let chunk_size = DEFAULT_CHUNK_SIZE;
        let total_chunks: u32 = total_size
            .div_ceil(chunk_size)
            .max(1)
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many chunks"))?;
        let chunks_path = chunks_sidecar_path(&cache_path);

        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Ensure the sparse data file exists at the declared
        // size — subsequent random-offset writes need it to be
        // pre-sized. We *reset* the bitmap whenever we create or
        // resize the data file: otherwise a stale `.chunks`
        // sidecar (left over from a previous URL that hashed
        // into the same blob dir, or from a manual cache wipe
        // that deleted the data file but not the sidecar) would
        // falsely advertise chunks as valid when the on-disk
        // bytes are actually zero.
        let file_existed = cache_path.exists();
        let file_size_matches = file_existed
            && std::fs::metadata(&cache_path)
                .map(|m| m.len() == total_size)
                .unwrap_or(false);
        let reset_bitmap = !file_existed || !file_size_matches;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(false)
            .open(&cache_path)?;
        if reset_bitmap {
            f.set_len(total_size)?;
        }
        let cache_file = crate::cache::CacheFile::new(f);

        // Load the bitmap. The `.chunks` sidecar is the *sole*
        // source of truth for which chunks are valid — a data
        // file present at the expected size is **not** proof of a
        // completed download, because `open` always pre-sizes the
        // sparse data file (`set_len`) so random-offset chunk
        // writes have somewhere to land. An interrupted or
        // size-probe-only open therefore leaves a full-size file
        // of zeros with no sidecar; trusting its size would mark
        // every chunk valid and silently serve zeroed bytes as
        // resident data (and skip the real download). Cases:
        // 1. Data file was just created/resized → bitmap MUST
        //    reset to all-zero. Stale sidecar bits are discarded
        //    and the file on disk (`.chunks`) is removed so a
        //    later open can't pick up the stale state again.
        // 2. `chunks_path` exists and matches `total_chunks`
        //    → use it verbatim.
        // 3. No sidecar (fresh open, or a prior open's unproven
        //    placeholder) → start empty and re-fetch every chunk.
        let (initial, leaves): (Vec<u8>, Vec<[u8; 32]>) = if reset_bitmap {
            let _ = std::fs::remove_file(&chunks_path);
            (vec![0u8; total_chunks as usize], vec![UNRECORDED; total_chunks as usize])
        } else {
            match read_sidecar(&chunks_path, total_size) {
                Some(sidecar) => (
                    sidecar.valid.iter().map(|&v| v as u8).collect(),
                    sidecar.leaves.unwrap_or_else(|| vec![UNRECORDED; total_chunks as usize]),
                ),
                None => (vec![0u8; total_chunks as usize], vec![UNRECORDED; total_chunks as usize]),
            }
        };

        // The validity bits as the sidecar's tree format holds them.
        let bits = {
            let shape = crate::merkle::MerkleShape::for_content(chunk_size, total_size);
            let blank = crate::merkle::MerkleRef::from_parts(shape.clone(), vec![UNRECORDED; shape.node_count as usize]);
            let bits = crate::merkle::MerkleState::from_ref(&blank);
            for (i, _) in initial.iter().enumerate().filter(|(_, v)| **v != 0) {
                bits.mark_valid(i as u32);
            }
            bits
        };
        let chunk_state: Vec<AtomicU8> = initial.into_iter().map(AtomicU8::new).collect();

        Ok(Self {
            transport,
            total_size,
            chunk_size,
            total_chunks,
            cache_path,
            cache_file,
            chunks_path,
            chunk_state: Arc::new(chunk_state),
            in_flight: Arc::new(Mutex::new(HashMap::new())),
            retry_policy: crate::transport::RetryPolicy::default(),
            full_transfer: Mutex::new(()),
            bits,
            leaves: Mutex::new(leaves),
        })
    }

    pub(crate) fn cache_path(&self) -> &Path {
        &self.cache_path
    }
    /// The URL this store fetches from.
    pub(crate) fn url(&self) -> &url::Url {
        self.transport.url()
    }

    pub(crate) fn total_size(&self) -> u64 {
        self.total_size
    }
    pub(crate) fn total_chunks(&self) -> u32 {
        self.total_chunks
    }
    pub(crate) fn chunk_size(&self) -> u64 {
        self.chunk_size
    }

    /// Whether every chunk has been downloaded. Lock-free —
    /// `Acquire` ordering ensures we observe the data-file
    /// writes that preceded any "valid" store.
    pub(crate) fn is_complete(&self) -> bool {
        self.chunk_state
            .iter()
            .all(|b| b.load(Ordering::Acquire) != 0)
    }

    /// Number of chunks currently marked valid.
    pub(crate) fn valid_count(&self) -> u32 {
        self.chunk_state
            .iter()
            .filter(|b| b.load(Ordering::Acquire) != 0)
            .count() as u32
    }

    /// Chunks marked valid within `[first, last]` inclusive.
    ///
    /// The range form is what a prefetch plan needs: whole-file fill
    /// says nothing about whether the particular window about to be
    /// read is the resident part or the missing one.
    pub(crate) fn valid_count_in_range(&self, first: u32, last: u32) -> u32 {
        if first > last || first as usize >= self.chunk_state.len() {
            return 0;
        }
        let last = (last as usize).min(self.chunk_state.len() - 1);
        self.chunk_state[first as usize..=last]
            .iter()
            .filter(|b| b.load(Ordering::Acquire) != 0)
            .count() as u32
    }

    /// The sidecar's bytes for the state now: the recorded leaves and
    /// the validity bits, in the `.mrkl` format, with the internal
    /// nodes and the root filled in once every chunk is here and
    /// recorded. The state may race with concurrent fetch completions;
    /// whichever snapshot lands is a valid prefix of the eventual
    /// state, and the next finish overwrites it with a fresher one.
    fn snapshot_sidecar(&self) -> Vec<u8> {
        let shape = crate::merkle::MerkleShape::for_content(self.chunk_size, self.total_size);
        let leaves = self.leaves.lock().unwrap().clone();
        let valid: Vec<bool> = self.chunk_state.iter().map(|b| b.load(Ordering::Acquire) != 0).collect();
        let complete = valid.iter().all(|&v| v) && leaves.iter().all(|l| *l != UNRECORDED);
        let tree = if complete && leaves.len() == shape.total_chunks as usize {
            crate::merkle::MerkleRef::from_leaves(shape.clone(), &leaves)
        } else {
            // Leaves only: an internal node over an unrecorded leaf is
            // not a hash of anything.
            let mut hashes = vec![UNRECORDED; shape.node_count as usize];
            for (i, leaf) in leaves.iter().enumerate().take(shape.total_chunks as usize) {
                hashes[shape.leaf_node_index(i as u32) as usize] = *leaf;
            }
            crate::merkle::MerkleRef::from_parts(shape.clone(), hashes)
        };
        let state = crate::merkle::MerkleState::from_ref(&tree);
        for (i, v) in valid.iter().enumerate() {
            if *v {
                state.mark_valid(i as u32);
            }
        }
        let mut out = Vec::new();
        state.write(&mut out).expect("write to a Vec cannot fail");
        out
    }

    /// Hash chunk `i` from the cache file and record it.
    fn record_from_cache(&self, i: u32) -> io::Result<()> {
        let mut buf = vec![0u8; self.chunk_len(i) as usize];
        self.cache_file.read_exact_at(&mut buf, self.chunk_start(i))?;
        self.leaves.lock().unwrap()[i as usize] = crate::merkle::sha256(&buf);
        Ok(())
    }

    /// The root of the tree recorded over the file, once every chunk is
    /// here: the file's content identity, the root a published `.mref`
    /// would have for the same bytes. Leaves not recorded yet — a
    /// sidecar written before chunks were hashed, or the whole file of
    /// one — are hashed from the cache file first, reporting `(done,
    /// total)` bytes to `progress`, and the sidecar is rewritten so
    /// this is paid once. `None` while any chunk is missing.
    pub(crate) fn content_root(&self, progress: &mut dyn FnMut(u64, u64)) -> io::Result<Option<[u8; 32]>> {
        if !self.is_complete() {
            return Ok(None);
        }
        let missing: Vec<u32> = {
            let leaves = self.leaves.lock().unwrap();
            (0..self.total_chunks).filter(|&i| leaves[i as usize] == UNRECORDED).collect()
        };
        if !missing.is_empty() {
            let total: u64 = missing.iter().map(|&i| self.chunk_len(i)).sum();
            let mut done = 0;
            progress(0, total);
            for i in missing {
                self.record_from_cache(i)?;
                done += self.chunk_len(i);
                progress(done, total);
            }
            let snapshot = self.snapshot_sidecar();
            let _guard = self.in_flight.lock().unwrap();
            save_sidecar(&self.chunks_path, &snapshot)?;
        }
        let shape = crate::merkle::MerkleShape::for_content(self.chunk_size, self.total_size);
        let leaves = self.leaves.lock().unwrap().clone();
        Ok(Some(*crate::merkle::MerkleRef::from_leaves(shape, &leaves).root_hash()))
    }

    /// The recorded root, when every chunk is here and recorded: what
    /// [`content_root`](Self::content_root) answers without reading
    /// the file. `None` otherwise.
    pub(crate) fn recorded_root(&self) -> Option<[u8; 32]> {
        if !self.is_complete() {
            return None;
        }
        let leaves = self.leaves.lock().unwrap().clone();
        if leaves.contains(&UNRECORDED) {
            return None;
        }
        let shape = crate::merkle::MerkleShape::for_content(self.chunk_size, self.total_size);
        Some(*crate::merkle::MerkleRef::from_leaves(shape, &leaves).root_hash())
    }

    /// Check every cached chunk with a recorded leaf against it, and
    /// return the chunks whose bytes no longer hash to what was
    /// recorded when they landed: local corruption of the cache file.
    /// Reports `(done, total)` bytes to `progress`.
    pub(crate) fn verify_recorded(&self, progress: &mut dyn FnMut(u64, u64)) -> io::Result<Vec<u32>> {
        let leaves = self.leaves.lock().unwrap().clone();
        let check: Vec<u32> = (0..self.total_chunks)
            .filter(|&i| self.chunk_state[i as usize].load(Ordering::Acquire) != 0 && leaves[i as usize] != UNRECORDED)
            .collect();
        let total: u64 = check.iter().map(|&i| self.chunk_len(i)).sum();
        let (mut done, mut bad) = (0, Vec::new());
        let mut buf = Vec::new();
        progress(0, total);
        for i in check {
            buf.resize(self.chunk_len(i) as usize, 0);
            self.cache_file.read_exact_at(&mut buf, self.chunk_start(i))?;
            if crate::merkle::sha256(&buf) != leaves[i as usize] {
                bad.push(i);
            }
            done += self.chunk_len(i);
            progress(done, total);
        }
        Ok(bad)
    }

    /// Read bytes from `[offset, offset+len)`, fetching any
    /// chunks needed to satisfy the range. Returns the assembled
    /// bytes from the local cache file.
    pub(crate) fn read(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read range overflow"))?;
        if end > self.total_size {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read past end: offset={offset} len={len} size={}",
                    self.total_size
                ),
            ));
        }
        let first = (offset / self.chunk_size) as u32;
        let last = ((end - 1) / self.chunk_size) as u32;
        self.ensure_chunks_valid(first, last)?;
        let mut buf = vec![0u8; len as usize];
        self.cache_file.read_exact_at(&mut buf, offset)?;
        Ok(buf)
    }

    /// Download every missing chunk, firing the callback after
    /// each chunk completes so callers can render in-flight
    /// progress. Idempotent — chunks already valid are skipped.
    ///
    /// Worker pool sized via [`crate::cache::download_concurrency`]
    /// so both the mref-backed `CachedChannel` and this no-mref
    /// path scale identically under one env knob. Workers pull
    /// chunk indices from a shared `Mutex<Vec<u32>>` queue and
    /// route each fetch through [`Self::fetch_chunk_with_dedup`],
    /// so concurrent partial-read traffic against the same store
    /// continues to dedup correctly with the prebuffer loop —
    /// the prebuffer queue only contains chunks currently not
    /// `Acquire`-valid, but if a concurrent partial-read worker
    /// fetches one of those chunks first, the queue's worker
    /// observes it as valid and skips re-fetching.
    pub(crate) fn prebuffer_with_progress<F>(&self, cb: F) -> io::Result<()>
    where
        F: FnMut(&DownloadProgress),
    {
        self.prebuffer_chunk_range(0, self.total_chunks, cb)
    }

    /// Same as [`Self::prebuffer_with_progress`] but bounded to the
    /// chunks covering `[byte_start, byte_end)`. Used by the view
    /// layer to honor profile windows — a windowed profile against a
    /// large base file only needs the chunks covering its window's
    /// byte range, not the whole file. `byte_end` is clamped to the
    /// total file size; an empty range is a no-op.
    pub(crate) fn prebuffer_range_with_progress<F>(
        &self,
        byte_start: u64,
        byte_end: u64,
        cb: F,
    ) -> io::Result<()>
    where
        F: FnMut(&DownloadProgress),
    {
        let end = byte_end.min(self.total_size);
        if byte_start >= end {
            let mut cb = cb;
            cb(&DownloadProgress::new(0, 0));
            return Ok(());
        }
        let first = (byte_start / self.chunk_size) as u32;
        let last_inclusive = ((end - 1) / self.chunk_size) as u32;
        let exclusive = last_inclusive + 1;
        self.prebuffer_chunk_range(first, exclusive, cb)
    }

    /// Common backend for the whole-file and ranged prebuffer entry
    /// points. Pulls only chunks in `[first..exclusive)`. The progress
    /// snapshot reports `range_bytes` / `range_chunks` so the UI meter
    /// shows progress against the bounded plan, not the full file
    /// size (otherwise a 100k-window profile's meter would top out
    /// at 0.1% even after the entire window is resident).
    fn prebuffer_chunk_range<F>(&self, first: u32, exclusive: u32, mut cb: F) -> io::Result<()>
    where
        F: FnMut(&DownloadProgress),
    {
        const PROGRESS_TICK: std::time::Duration = std::time::Duration::from_millis(250);

        let exclusive = exclusive.min(self.total_chunks);
        if first >= exclusive {
            cb(&DownloadProgress::new(0, 0));
            return Ok(());
        }
        let range_chunks = exclusive - first;
        let range_bytes: u64 = (first..exclusive).map(|i| self.chunk_len(i)).sum();

        let progress = DownloadProgress::new(range_bytes, range_chunks);
        let mut already_valid: u64 = 0;
        let mut pending: Vec<u32> = Vec::new();
        for i in first..exclusive {
            if self.chunk_state[i as usize].load(Ordering::Acquire) != 0 {
                already_valid += self.chunk_len(i);
                progress.increment_completed();
            } else {
                pending.push(i);
            }
        }
        progress.add_downloaded_bytes(already_valid);
        cb(&progress);
        if pending.is_empty() {
            return Ok(());
        }

        // The worker-pool / queue / abort / completion mechanics are owned
        // by `drain_parallel` (shared with the merkle-verified
        // `CachedChannel` path). `drain_parallel` reverses the work list so
        // `pop()` yields the lowest chunk index first — near-sequential
        // range requests, which S3 / CloudFront read-ahead prefers. This
        // path's only per-chunk work is the dedup-aware fetch (which writes
        // to the sparse cache file, marks the chunk valid, and persists the
        // `.chunks` sidecar itself) plus progress accounting — no merkle
        // verification, by design, since no `.mref` was published.
        let result = crate::transport::pump::drain_parallel(
            pending,
            crate::cache::download_concurrency(),
            PROGRESS_TICK,
            |&idx: &u32| -> io::Result<()> {
                let len = self.chunk_len(idx);
                match self.fetch_chunk_with_dedup(idx) {
                    Ok(()) => {
                        progress.add_downloaded_bytes(len);
                        progress.increment_completed();
                        Ok(())
                    }
                    Err(e) => {
                        progress.mark_failed();
                        Err(e)
                    }
                }
            },
            || cb(&progress),
        );

        // Terminal frame so the caller always sees the final (100% or
        // failed) snapshot — `drain_parallel` fires no event of its own.
        cb(&progress);
        result
    }

    /// Ensure chunks `[first..=last]` are valid. Mirrors
    /// `CachedChannel::ensure_chunks_valid` — dedups with
    /// `in_flight`, waits on condvar for chunks already being
    /// fetched by another thread, fetches its own assigned
    /// chunks serially.
    ///
    /// Lock discipline: **only the `in_flight` mutex is ever
    /// held across calls into other helpers**. Chunk-state
    /// reads are lock-free atomic loads. There is no path
    /// where two locks are held simultaneously, so the
    /// hierarchical-lock deadlock that a prior cut of this
    /// code suffered is impossible by construction.
    fn ensure_chunks_valid(&self, first: u32, last: u32) -> io::Result<()> {
        let (to_fetch, to_wait): (Vec<u32>, Vec<(u32, Arc<Condvar>)>) = {
            let mut in_flight = self.in_flight.lock().unwrap();
            let mut fetch = Vec::new();
            let mut wait = Vec::new();
            for i in first..=last {
                if self.chunk_state[i as usize].load(Ordering::Acquire) != 0 {
                    continue;
                }
                if let Some(cv) = in_flight.get(&i) {
                    wait.push((i, cv.clone()));
                } else {
                    let cv = Arc::new(Condvar::new());
                    in_flight.insert(i, cv.clone());
                    fetch.push(i);
                }
            }
            (fetch, wait)
        };

        for (chunk_idx, cv) in &to_wait {
            let mut in_flight = self.in_flight.lock().unwrap();
            loop {
                if self.chunk_state[*chunk_idx as usize].load(Ordering::Acquire) != 0 {
                    break;
                }
                if !in_flight.contains_key(chunk_idx) {
                    break;
                }
                in_flight = cv
                    .wait_timeout(in_flight, std::time::Duration::from_millis(50))
                    .unwrap()
                    .0;
            }
        }

        for i in to_fetch {
            let r = self.fetch_and_write_chunk(i);
            // Always remove the in_flight entry and notify
            // waiters, even on error — otherwise stuck waiters
            // would block forever on a chunk we abandoned.
            let cv = self.in_flight.lock().unwrap().remove(&i);
            if let Some(cv) = cv {
                cv.notify_all();
            }
            r?;
        }
        Ok(())
    }

    /// Single-chunk fetch with `in_flight` dedup. Used by the
    /// prebuffer loop so concurrent partial-read traffic
    /// against the same store reuses any in-flight fetch
    /// instead of duplicating it.
    fn fetch_chunk_with_dedup(&self, i: u32) -> io::Result<()> {
        // Fast path: chunk became valid since the caller's
        // last check.
        if self.chunk_state[i as usize].load(Ordering::Acquire) != 0 {
            return Ok(());
        }
        let waiter = {
            let mut in_flight = self.in_flight.lock().unwrap();
            if let Some(cv) = in_flight.get(&i) {
                Some(cv.clone())
            } else {
                let cv = Arc::new(Condvar::new());
                in_flight.insert(i, cv);
                None
            }
        };
        match waiter {
            // Another thread is fetching — wait for it.
            Some(cv) => {
                let mut in_flight = self.in_flight.lock().unwrap();
                loop {
                    if self.chunk_state[i as usize].load(Ordering::Acquire) != 0 {
                        break;
                    }
                    if !in_flight.contains_key(&i) {
                        break;
                    }
                    in_flight = cv
                        .wait_timeout(in_flight, std::time::Duration::from_millis(50))
                        .unwrap()
                        .0;
                }
                Ok(())
            }
            // We own the fetch — do it.
            None => {
                let r = self.fetch_and_write_chunk(i);
                let cv = self.in_flight.lock().unwrap().remove(&i);
                if let Some(cv) = cv {
                    cv.notify_all();
                }
                r
            }
        }
    }

    /// FullTransfer fallback: stream the entire body into the cache
    /// file with one plain GET, then mark every chunk valid. Used
    /// when the server doesn't advertise byte-range support (the
    /// probe's `Accept-Ranges` check). A server that supports ranges
    /// without advertising them costs one whole-file download here —
    /// correct bytes, suboptimal transfer; the advertisement is the
    /// only offline-safe signal we have.
    fn fetch_full_transfer(&self) -> io::Result<()> {
        let _guard = self.full_transfer.lock().unwrap();
        if self.is_complete() {
            return Ok(());
        }
        log::warn!(
            "{} does not advertise byte-range support; downloading the entire file \
             ({} bytes) before the first read can be served (FullTransfer)",
            self.transport.url(),
            self.total_size,
        );
        self.retry_policy.execute(|| {
            // Restart from offset 0 on every attempt — positional
            // writes make partial previous attempts harmless.
            let mut w = PositionalWriter {
                file: &self.cache_file,
                pos: 0,
            };
            let n = self.transport.fetch_full_to(&mut w)?;
            if n != self.total_size {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "full transfer short read: expected {} bytes, got {n}",
                        self.total_size
                    ),
                ));
            }
            Ok(())
        })?;
        self.cache_file.sync_data()?;
        // Record what landed: the body arrived as one stream, so each
        // chunk is hashed back from the cache file.
        for i in 0..self.total_chunks {
            self.record_from_cache(i)?;
        }
        for slot in self.chunk_state.iter() {
            slot.store(1, Ordering::Release);
        }
        let snapshot = self.snapshot_sidecar();
        {
            let _guard = self.in_flight.lock().unwrap();
            let _ = save_sidecar(&self.chunks_path, &snapshot);
        }
        Ok(())
    }

    fn chunk_start(&self, i: u32) -> u64 {
        i as u64 * self.chunk_size
    }
    fn chunk_len(&self, i: u32) -> u64 {
        let start = self.chunk_start(i);
        (self.total_size - start).min(self.chunk_size)
    }

    fn fetch_and_write_chunk(&self, i: u32) -> io::Result<()> {
        // Range-less servers can't serve chunks: the documented
        // FullTransfer semantic applies — the whole file downloads
        // once (streamed to the cache file) before any read returns.
        // Per-chunk callers converge here naturally: the first one
        // streams, the rest block on the full-transfer lock and then
        // find their chunk already valid.
        {
            use crate::transport::ChunkedTransport;
            if !self.transport.supports_range() {
                return self.fetch_full_transfer();
            }
        }
        let start = self.chunk_start(i);
        let len = self.chunk_len(i);
        // Retry transient fetch failures with backoff — a flaky connection on
        // one chunk of a concurrent prebuffer is recovered in place rather
        // than failing the whole download (parity with the `.mref` path).
        let bytes = self
            .retry_policy
            .execute(|| self.transport.fetch_range(start, len))?;
        if bytes.len() as u64 != len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("chunk {i} short read: expected {len}, got {}", bytes.len()),
            ));
        }
        // Ordering matters for crash safety:
        // 1. Write the chunk bytes to disk + fsync.
        // 2. `Release`-store the bitmap bit so any thread that
        //    observes "valid" with `Acquire` also observes the
        //    preceding data-file writes.
        // 3. Persist the sidecar bitmap (best-effort) under the
        //    `in_flight` lock so concurrent finishers serialise
        //    on the rename instead of racing.
        // A crash between (1) and (2) loses only the post-write
        // bookkeeping — chunk gets refetched on next open.
        // A crash after (2) but before (3) leaves the on-disk
        // bitmap behind the in-memory state; the next session
        // refetches a redundant chunk and re-saves. Either way
        // the data is correct.
        self.cache_file.write_all_at(&bytes, start)?;
        // `sync_data` rather than `sync_all` — we only care about the
        // data bytes, not metadata. Cheaper, same crash-safety
        // guarantee for the byte content.
        self.cache_file.sync_data()?;
        // The leaf, from the bytes as they arrived, in this worker:
        // leaves are independent, so the order chunks land in does not
        // matter.
        let leaf = crate::merkle::sha256(&bytes);
        self.leaves.lock().unwrap()[i as usize] = leaf;
        if let Some(slot) = self.chunk_state.get(i as usize) {
            slot.store(1, Ordering::Release);
        }
        // Serialise the sidecar write through the in_flight
        // lock — cheap, infrequent, and avoids tmp-file races
        // between concurrent finishers. The lock is released
        // before any read can observe the new "valid" bit
        // because the bit was set above; readers only need
        // memory visibility, not on-disk visibility.
        {
            let _guard = self.in_flight.lock().unwrap();
            let _ = self.persist_chunk(i, leaf);
        }
        Ok(())
    }

    /// Record chunk `i`'s leaf and validity in the sidecar: its 32
    /// bytes at its place in the tree and the validity bits merged in,
    /// never the whole tree — which on a 2 TB base is 16 MiB per 8 MiB
    /// chunk, the write amplification that once held `.mrkl` downloads
    /// to 6 MB/s. The whole sidecar is written only when it is not yet
    /// in the tree format, and once more on completion, with the root.
    /// Called under the `in_flight` lock.
    fn persist_chunk(&self, i: u32, leaf: [u8; 32]) -> io::Result<()> {
        self.bits.mark_valid(i);
        let shape = self.bits.shape().clone();
        let fresh = || self.snapshot_sidecar();
        persist_leaf(&self.chunks_path, &shape, &self.bits, i, &leaf, fresh)?;
        if self.is_complete() && self.leaves.lock().unwrap().iter().all(|l| *l != UNRECORDED) {
            save_sidecar(&self.chunks_path, &self.snapshot_sidecar())?;
        }
        Ok(())
    }
}

/// Write leaf `i` of the tree `shape` describes into the sidecar at
/// `path`, and merge `bits` into its validity bitset, rewriting the
/// file only when it is not this tree's layout yet (then with
/// `whole()`'s bytes). Bytes written per call: 32 for the leaf and
/// one bitset — not the tree.
fn persist_leaf(
    path: &Path,
    shape: &crate::merkle::MerkleShape,
    bits: &crate::merkle::MerkleState,
    i: u32,
    leaf: &[u8; 32],
    whole: impl FnOnce() -> Vec<u8>,
) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    // The layout by length, as `MerkleState::checkpoint` judges it: a
    // bitmap of the earlier format is one byte per chunk, never this.
    let hash_bytes = shape.node_count as u64 * crate::merkle::HASH_SIZE as u64;
    let layout = hash_bytes + bits.bitset_byte_size() as u64 + crate::merkle::FOOTER_SIZE_V2 as u64;
    if std::fs::metadata(path).map(|m| m.len()).ok() != Some(layout) {
        return save_sidecar(path, &whole());
    }
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(shape.leaf_node_index(i) as u64 * crate::merkle::HASH_SIZE as u64))?;
    file.write_all(leaf)?;
    drop(file);
    bits.checkpoint(path)
}

/// `io::Write` adapter over [`crate::cache::CacheFile`]'s positional
/// writes, for streaming a sequential body into the sparse cache file
/// without touching a shared cursor.
struct PositionalWriter<'a> {
    file: &'a crate::cache::CacheFile,
    pos: u64,
}

impl std::io::Write for PositionalWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write_all_at(buf, self.pos)?;
        self.pos += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The size of the file at `cache_path` when its `.chunks` bitmap
/// records every chunk as fetched — a complete download that can be
/// opened without asking the server for its length. `None` when the
/// file, its bitmap, or any chunk of it is missing.
pub(crate) fn complete_size_on_disk(cache_path: &Path, allow_partial: bool) -> Option<u64> {
    let size = std::fs::metadata(cache_path).ok()?.len();
    if size == 0 {
        return None;
    }
    let sidecar = read_sidecar(&chunks_sidecar_path(cache_path), size)?;
    (allow_partial || sidecar.valid.iter().all(|&v| v)).then_some(size)
}

/// A `.chunks` sidecar as read: which chunks are here, and — in the
/// current format — the leaf recorded for each (all zeros for one not
/// recorded).
pub(crate) struct ChunksSidecar {
    /// One flag per chunk.
    pub valid: Vec<bool>,
    /// The recorded leaves; `None` for a sidecar of the earlier,
    /// bitmap-only format.
    pub leaves: Option<Vec<[u8; 32]>>,
}

/// Read the `.chunks` sidecar of a cache file of `total_size` bytes, in
/// either format: the `.mrkl` format this store writes (recognised by
/// its footer's length marker, 45), or the earlier one byte per chunk
/// (whose bytes are 0 or 1). `None` when it is missing, or does not
/// describe a file of this size.
pub(crate) fn read_sidecar(path: &Path, total_size: u64) -> Option<ChunksSidecar> {
    let bytes = std::fs::read(path).ok()?;
    let chunks = total_size.div_ceil(DEFAULT_CHUNK_SIZE).max(1) as usize;
    if bytes.len() >= crate::merkle::FOOTER_SIZE_V2 && bytes.last() == Some(&(crate::merkle::FOOTER_SIZE_V2 as u8)) {
        let state = crate::merkle::MerkleState::from_bytes(&bytes).ok()?;
        let shape = state.shape().clone();
        if shape.total_content_size != total_size || shape.chunk_size != DEFAULT_CHUNK_SIZE {
            return None;
        }
        let tree = state.to_ref();
        let n = shape.total_chunks as usize;
        return Some(ChunksSidecar {
            valid: (0..n as u32).map(|i| state.is_valid(i)).collect(),
            leaves: Some((0..n as u32).map(|i| *tree.leaf_hash(i)).collect()),
        });
    }
    let bitmap = load_bitmap(path, chunks).ok()?;
    Some(ChunksSidecar { valid: bitmap.iter().map(|&b| b != 0).collect(), leaves: None })
}

fn chunks_sidecar_path(cache_path: &Path) -> PathBuf {
    let mut p = cache_path.as_os_str().to_owned();
    p.push(".chunks");
    PathBuf::from(p)
}

fn load_bitmap(path: &Path, expected_chunks: usize) -> io::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != expected_chunks {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "bitmap size {} != expected {}",
                bytes.len(),
                expected_chunks
            ),
        ));
    }
    Ok(bytes)
}

fn save_sidecar(path: &Path, bytes: &[u8]) -> io::Result<()> {
    // Write-rename for atomicity so a crash mid-write doesn't
    // leave a half-truncated bitmap. Include the PID in the tmp
    // filename so two processes racing on the same sidecar
    // don't clobber each other's tmp before the rename — the
    // rename itself is atomic on POSIX, so whichever process
    // wins the race ships its (newer) snapshot intact instead
    // of overwriting a half-written file.
    let tmp = path.with_extension(format!("chunks.tmp.{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_data()?;
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// In-memory transport that counts fetch_range calls so the
    /// dedup-and-cache behaviour can be observed. Uses the
    /// `ChunkedTransport` trait directly for testing in
    /// isolation — the real `ChunkStore` only accepts
    /// `HttpTransport` to keep its struct concrete, but the
    /// per-chunk fetch logic is what we want to exercise here.
    /// We test the higher-level read flow via the HTTP
    /// integration tests instead.
    struct CountedFakeTransport {
        data: Vec<u8>,
        calls: AtomicU32,
    }

    impl ChunkedTransport for CountedFakeTransport {
        fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let s = start as usize;
            let e = s + len as usize;
            Ok(self.data[s..e].to_vec())
        }
        fn content_length(&self) -> io::Result<u64> {
            Ok(self.data.len() as u64)
        }
        fn supports_range(&self) -> bool {
            true
        }
    }

    /// The chunk-shape math: chunk_len of the last chunk is the
    /// remainder, never exceeds the chunk size.
    #[test]
    fn chunk_geometry() {
        let cache = tempfile::tempdir().unwrap();
        let path = cache.path().join("data.bin");
        // 8 MiB chunk size; 20 MiB file → 3 chunks: 8 MiB,
        // 8 MiB, 4 MiB.
        let total: u64 = 20 * 1024 * 1024;
        std::fs::write(&path, vec![0u8; total as usize]).unwrap();
        let transport = HttpTransport::new(url::Url::parse("http://example.com/x").unwrap());
        let store = ChunkStore::open(transport, total, path).unwrap();
        assert_eq!(store.total_chunks(), 3);
        assert_eq!(store.chunk_len(0), 8 * 1024 * 1024);
        assert_eq!(store.chunk_len(1), 8 * 1024 * 1024);
        assert_eq!(store.chunk_len(2), 4 * 1024 * 1024);
    }

    /// A cache file present at the expected size but with **no**
    /// `.chunks` sidecar must NOT be assumed complete. `open`
    /// always pre-sizes the sparse data file so chunk writes have
    /// somewhere to land, so a full-size file with no sidecar is
    /// an unproven placeholder — typically left by a size-probe or
    /// an interrupted open. Trusting its size would silently serve
    /// its zeroed bytes as resident data and skip the real
    /// download. Regression guard: the sidecar is the sole source
    /// of truth, so this store reports *nothing* valid and will
    /// re-fetch every chunk.
    #[test]
    fn full_size_file_without_sidecar_is_not_complete() {
        let cache = tempfile::tempdir().unwrap();
        let path = cache.path().join("data.bin");
        let total: u64 = 1024;
        // Pre-existing full-size file (e.g. a prior open's sparse
        // allocation) with no sidecar alongside it.
        std::fs::write(&path, vec![0u8; total as usize]).unwrap();
        assert!(!chunks_sidecar_path(&path).exists());
        let transport = HttpTransport::new(url::Url::parse("http://example.com/x").unwrap());
        let store = ChunkStore::open(transport, total, path).unwrap();
        assert!(
            !store.is_complete(),
            "full-size file without a sidecar must not be trusted as complete"
        );
        assert_eq!(
            store.valid_count(),
            0,
            "no chunk is proven valid without a sidecar"
        );
    }

    /// **A sidecar of the earlier format is read as it was**, and
    /// written back in the recorded-tree format with the same chunks
    /// valid and their leaves unrecorded until asked for.
    #[test]
    fn a_legacy_bitmap_sidecar_reads_and_rewrites_in_the_tree_format() {
        let cache = tempfile::tempdir().unwrap();
        let path = cache.path().join("data.bin");
        // Use a small chunk multiple of DEFAULT_CHUNK_SIZE so
        // we have known chunk boundaries.
        let total: u64 = 24 * 1024 * 1024; // 3 chunks
        std::fs::write(&path, vec![0u8; total as usize]).unwrap();
        let chunks_path = chunks_sidecar_path(&path);
        save_sidecar(&chunks_path, &[1u8, 0, 1]).unwrap();
        let transport = HttpTransport::new(url::Url::parse("http://example.com/x").unwrap());
        let store = ChunkStore::open(transport, total, path.clone()).unwrap();
        assert_eq!(store.valid_count(), 2);
        save_sidecar(&chunks_path, &store.snapshot_sidecar()).unwrap();
        let reread = read_sidecar(&chunks_path, total).unwrap();
        assert_eq!(reread.valid, [true, false, true]);
        assert_eq!(reread.leaves, Some(vec![UNRECORDED; 3]), "no chunk was hashed by reading it");
        assert_eq!(*std::fs::read(&chunks_path).unwrap().last().unwrap(), crate::merkle::FOOTER_SIZE_V2 as u8);
    }

    /// **A complete file with a sidecar of the earlier format gets its
    /// content identity** by hashing its chunks from the cache file,
    /// once, with progress; the root is the one a publisher's `.mref`
    /// would have, and the sidecar is rewritten so the next ask reads
    /// it.
    #[test]
    fn a_legacy_complete_file_is_hashed_once_for_its_content_root() {
        let cache = tempfile::tempdir().unwrap();
        let path = cache.path().join("data.bin");
        let content: Vec<u8> = (0..(DEFAULT_CHUNK_SIZE as usize + 4096)).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &content).unwrap();
        save_sidecar(&chunks_sidecar_path(&path), &[1u8, 1]).unwrap();
        let transport = HttpTransport::new(url::Url::parse("http://127.0.0.1:9/never-asked").unwrap());
        let store = ChunkStore::open(transport, content.len() as u64, path.clone()).unwrap();
        let mut heard = Vec::new();
        let root = store.content_root(&mut |d, t| heard.push((d, t))).unwrap().unwrap();
        assert_eq!(root, *crate::merkle::MerkleRef::from_content(&content, DEFAULT_CHUNK_SIZE).root_hash());
        assert_eq!(heard.first(), Some(&(0, content.len() as u64)));
        assert_eq!(heard.last(), Some(&(content.len() as u64, content.len() as u64)));
        let reread = read_sidecar(&chunks_sidecar_path(&path), content.len() as u64).unwrap();
        assert!(reread.leaves.unwrap().iter().all(|l| *l != UNRECORDED), "the leaves were recorded");
        // A second ask hashes nothing.
        let mut again = 0;
        store.content_root(&mut |_, _| again += 1).unwrap();
        assert_eq!(again, 0);
        // A partial file has none.
        save_sidecar(&chunks_sidecar_path(&path), &[1u8, 0]).unwrap();
        let transport = HttpTransport::new(url::Url::parse("http://127.0.0.1:9/never-asked").unwrap());
        let partial = ChunkStore::open(transport, content.len() as u64, path).unwrap();
        assert_eq!(partial.content_root(&mut |_, _| {}).unwrap(), None);
    }

    /// **A recorded chunk writes its leaf and the bits, not the tree.**
    /// The sidecar of a 512 GiB file is a 4 MiB tree; rewriting it per
    /// 8 MiB chunk is the write amplification that once held `.mrkl`
    /// downloads to 6 MB/s. 200 chunks may write 200 leaves and 200
    /// bitsets.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_recorded_chunk_writes_its_leaf_and_bits_not_the_tree() {
        let thread_wchar = || -> u64 {
            std::fs::read_to_string("/proc/thread-self/io").unwrap()
                .lines().find_map(|l| l.strip_prefix("wchar:"))
                .and_then(|v| v.trim().parse().ok()).unwrap()
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.bin.chunks");
        let shape = crate::merkle::MerkleShape::for_content(DEFAULT_CHUNK_SIZE, 65536 * DEFAULT_CHUNK_SIZE);
        let blank = crate::merkle::MerkleRef::from_parts(shape.clone(), vec![UNRECORDED; shape.node_count as usize]);
        let bits = crate::merkle::MerkleState::from_ref(&blank);
        let whole = |bits: &crate::merkle::MerkleState| {
            let mut v = Vec::new();
            bits.write(&mut v).unwrap();
            v
        };
        bits.mark_valid(0);
        persist_leaf(&path, &shape, &bits, 0, &[7u8; 32], || whole(&bits)).unwrap();
        let tree = std::fs::metadata(&path).unwrap().len();
        assert!(tree > 4_000_000, "the tree is big enough for the guard to mean something: {tree}");

        let calls = 200u32;
        let before = thread_wchar();
        for i in 1..=calls {
            bits.mark_valid(i);
            persist_leaf(&path, &shape, &bits, i, &[i as u8; 32], || whole(&bits)).unwrap();
        }
        let written = thread_wchar() - before;
        let allowed = u64::from(calls) * (32 + bits.bitset_byte_size() as u64 + 256) + 64 * 1024;
        assert!(written <= allowed, "{written} bytes written for {calls} recorded chunks (tree {tree} B, limit {allowed}): the tree is being rewritten per chunk");
        let state = crate::merkle::MerkleState::load(&path).unwrap();
        assert!((0..=calls).all(|i| state.is_valid(i)));
        assert_eq!(*state.to_ref().leaf_hash(150), [150u8; 32]);
    }

    // Note: we don't expose a CountedFakeTransport-driven path
    // here because ChunkStore takes a concrete `HttpTransport`.
    // The full fetch-and-cache exercise lives in
    // tests/http_storage.rs against the test server fixture so
    // we exercise the same code that production hits.
    #[allow(dead_code)]
    fn _silence_unused(_: CountedFakeTransport) {}
}
