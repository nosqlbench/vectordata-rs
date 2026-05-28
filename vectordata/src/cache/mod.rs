// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cache-backed file channel with merkle verification.
//!
//! `CachedChannel` transparently downloads, verifies, and caches data from a
//! remote source using merkle tree integrity checking. Reads check local state
//! first; missing chunks are fetched on demand, verified against the reference
//! tree, and persisted to a local cache file.
//!
//! # Crash Recovery
//!
//! The `.mrkl` state file is the source of truth for what has been verified.
//! On restart, loading the state file resumes from the last checkpoint — only
//! unverified chunks are re-downloaded.

pub(crate) mod reader;

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

use crate::merkle::{MerkleRef, MerkleState};
use crate::transport::{
    ChunkRequest, ChunkedTransport, DownloadProgress, RetryPolicy,
};

/// Default concurrency for parallel chunk downloads. 16 lands well
/// short of NIC saturation on typical desktop / cloud VM links but
/// keeps every meaningful S3 / CloudFront origin busy: each TCP
/// connection caps around 100-300 MB/s; 16 streams sustain >1 GB/s
/// from a single bucket. Tunable via the `VECTORDATA_DOWNLOAD_CONCURRENCY`
/// env var for sites running on 25 / 100 Gbps links where pushing
/// concurrency to 64-128 is worthwhile.
const DEFAULT_CONCURRENCY: usize = 16;

/// Read the configured parallel-chunk-download worker count. Both
/// `Storage::Cached` (mref-backed) and `Storage::Http` (chunked-only)
/// route through this so a single env knob tunes both paths
/// identically. Floor of 1 — `n=0` would deadlock the queue-pull loop.
pub(crate) fn download_concurrency() -> usize {
    std::env::var("VECTORDATA_DOWNLOAD_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_CONCURRENCY)
}

/// Thread-safe positional file I/O for the cache file.
///
/// On Unix (Linux + macOS), [`pwrite(2)`] / [`pread(2)`] take an
/// explicit offset and do *not* touch the file's cursor, so multiple
/// threads can write and read different ranges of the same file
/// concurrently with no synchronization at all. The wrapper holds
/// the `File` directly with no mutex.
///
/// On Windows, `FileExt::seek_write` / `seek_read` mutate the
/// file cursor (the "positional" part is implemented client-side
/// by the std impl) — concurrent calls would race on the cursor.
/// The wrapper falls back to a `Mutex<File>` + `seek`/`write_all`
/// pair, which is what the cache module used everywhere before
/// this change.
///
/// The public surface is identical on both platforms so the rest
/// of the cache code is cfg-agnostic.
struct CacheFile {
    #[cfg(unix)]
    file: File,
    #[cfg(not(unix))]
    file: Mutex<File>,
}

impl CacheFile {
    fn new(file: File) -> Self {
        #[cfg(unix)]      { CacheFile { file } }
        #[cfg(not(unix))] { CacheFile { file: Mutex::new(file) } }
    }

    /// Read exactly `buf.len()` bytes from `offset`.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        #[cfg(unix)] {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset)
        }
        #[cfg(not(unix))] {
            use std::io::{Read, Seek};
            let mut f = self.file.lock().unwrap();
            f.seek(SeekFrom::Start(offset))?;
            f.read_exact(buf)
        }
    }

    /// Write exactly `buf.len()` bytes at `offset`.
    fn write_all_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        #[cfg(unix)] {
            use std::os::unix::fs::FileExt;
            self.file.write_all_at(buf, offset)
        }
        #[cfg(not(unix))] {
            use std::io::{Seek, Write};
            let mut f = self.file.lock().unwrap();
            f.seek(SeekFrom::Start(offset))?;
            f.write_all(buf)
        }
    }
}


/// Hex-encode a 32-byte hash for diagnostic messages. Avoids pulling in
/// the `hex` crate as a runtime dep just for one error string.
fn hex_short(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// A file channel that transparently downloads, verifies, and caches data
/// from a remote source using merkle tree integrity checking.
pub struct CachedChannel {
    /// Local cache file (random-access read/write).
    /// Wraps a single `File` handle; on Unix uses pwrite/pread for
    /// lock-free positional I/O, on Windows falls back to a mutex.
    cache_file: CacheFile,
    /// Path to the cache file (for reporting).
    cache_path: PathBuf,
    /// Merkle verification state (which chunks are valid).
    /// `mark_valid` and the read-side queries are lock-free via
    /// atomic `valid_words`; no surrounding `Mutex` needed.
    state: MerkleState,
    /// Path to the .mrkl state file.
    state_path: PathBuf,
    /// Remote data source. `Arc<dyn …>` (not `Box`) so two
    /// `CachedChannel`s opened against the same upstream — same URL,
    /// distinct cache dirs — can share one transport instance, and
    /// callers can clone the handle out before constructing the
    /// channel without losing the original.
    transport: Arc<dyn ChunkedTransport>,
    /// Reference tree for hash verification.
    reference: MerkleRef,
    /// Retry policy for downloads.
    retry_policy: RetryPolicy,
    /// Download concurrency limit.
    concurrency: usize,
    /// Tracks which chunks are currently being fetched.
    /// Concurrent readers of the same chunk wait on the Condvar instead
    /// of issuing duplicate requests.
    in_flight: Mutex<HashMap<u32, Arc<Condvar>>>,
}

#[allow(dead_code)] // pub(crate) helpers retained for Storage and tests
impl CachedChannel {
    /// Open a cached channel for a remote resource.
    ///
    /// - `transport`: the remote data source
    /// - `reference`: the merkle reference tree (.mref)
    /// - `cache_dir`: local directory for cache file and state
    /// - `name`: base name for cache files (e.g., "base_vectors.fvec")
    ///
    /// Uses a single `.mrkl` file as both reference and state (dual-mode),
    /// matching the Java `MerkleDataImpl` pattern. If a `.mrkl` exists,
    /// resumes from that checkpoint and derives the reference from its
    /// embedded hashes. Otherwise creates the `.mrkl` from the provided
    /// reference with a zeroed validity bitset.
    pub fn open(
        transport: Arc<dyn ChunkedTransport>,
        reference: MerkleRef,
        cache_dir: &Path,
        name: &str,
    ) -> io::Result<Self> {
        fs::create_dir_all(cache_dir)?;

        let cache_path = cache_dir.join(name);
        let state_path = cache_dir.join(format!("{}.mrkl", name));

        // Dual-mode: .mrkl is the single source of truth for *verified content*,
        // but the caller's `reference` argument is the single source of truth for
        // *what content the caller wants*. The two must agree, or the cache is
        // stale relative to the upstream resource and serving from it would
        // silently return wrong bytes. Compare root hashes and fail loudly on
        // mismatch — auto-recovery is the caller's responsibility (delete the
        // cache dir and reopen) because silent invalidation is what got us into
        // the stale-bytes regime in the first place.
        let (reference, state) = if state_path.exists() {
            let state = MerkleState::load(&state_path)?;
            let cached_ref = state.to_ref();
            if cached_ref.root_hash() != reference.root_hash() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "cache at {} is stale: cached merkle root {} disagrees with \
                         upstream merkle root {}. The remote resource has changed since \
                         this cache directory was populated. Delete {} and reopen to \
                         force a refetch.",
                        cache_dir.display(),
                        hex_short(cached_ref.root_hash()),
                        hex_short(reference.root_hash()),
                        cache_dir.display(),
                    ),
                ));
            }
            // Prefer the caller's reference (identical content, but it is the
            // authoritative copy and may carry shape metadata we want to keep).
            (reference, state)
        } else {
            let state = MerkleState::from_ref(&reference);
            if cache_path.exists() {
                // Cache file exists without merkle state (e.g., from a raw
                // download). Verify existing content against merkle hashes
                // to recover valid chunks without re-downloading.
                let shape = reference.shape();
                if let Ok(mut f) = fs::File::open(&cache_path) {
                    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0);
                    let mut verified = 0u32;
                    let mut buf = vec![0u8; shape.chunk_size as usize];
                    for i in 0..shape.total_chunks {
                        let start = shape.chunk_start(i);
                        let len = shape.chunk_len(i);
                        if start + len > file_len { break; }
                        use std::io::{Read, Seek};
                        if f.seek(SeekFrom::Start(start)).is_err() { break; }
                        let chunk_buf = &mut buf[..len as usize];
                        if f.read_exact(chunk_buf).is_err() { break; }
                        if reference.verify_chunk(i, chunk_buf) {
                            state.mark_valid(i);
                            verified += 1;
                        }
                    }
                    log::info!(
                        "{}: recovered {}/{} chunks from existing cache file",
                        name, verified, shape.total_chunks
                    );
                }
            }
            state.save(&state_path)?;
            (reference, state)
        };

        // Open or create the cache file, pre-allocating to full size
        let total_size = reference.shape().total_content_size;
        let cache_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&cache_path)?;

        // Extend file to full size if needed (sparse file on most FSes)
        let current_len = cache_file.metadata()?.len();
        if current_len < total_size {
            cache_file.set_len(total_size)?;
        }

        Ok(CachedChannel {
            cache_file: CacheFile::new(cache_file),
            cache_path,
            state,
            state_path,
            transport,
            reference,
            retry_policy: RetryPolicy::default(),
            concurrency: download_concurrency(),
            in_flight: Mutex::new(HashMap::new()),
        })
    }

    /// Set the retry policy.
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Set the download concurrency.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Read bytes from `[offset, offset+len)`, fetching and verifying chunks
    /// as needed.
    pub fn read(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        let shape = self.reference.shape();
        let end = offset + len;

        if end > shape.total_content_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "read range [{}, {}) exceeds content size {}",
                    offset, end, shape.total_content_size
                ),
            ));
        }

        // Determine which chunks overlap the requested range
        let first_chunk = (offset / shape.chunk_size) as u32;
        let last_chunk = ((end - 1) / shape.chunk_size) as u32;

        // Ensure all needed chunks are valid
        self.ensure_chunks_valid(first_chunk, last_chunk)?;

        // Read from the cache file (positional — no cursor mutation
        // on Unix, mutex-guarded on Windows).
        let mut buf = vec![0u8; len as usize];
        self.cache_file.read_exact_at(&mut buf, offset)?;
        Ok(buf)
    }

    /// Ensure chunks in range `[first, last]` (inclusive) are downloaded and
    /// verified. Deduplicates concurrent requests: if another thread is
    /// already fetching a chunk, this thread waits for it instead of
    /// issuing a duplicate request.
    fn ensure_chunks_valid(&self, first: u32, last: u32) -> io::Result<()> {
        // Collect chunks that need work — either missing or in-flight.
        // The `state.is_valid` check is lock-free; the in-flight
        // membership decision happens under the in_flight mutex so
        // we don't lose a race with a worker that's about to remove
        // an entry.
        let (to_fetch, to_wait): (Vec<u32>, Vec<(u32, Arc<Condvar>)>) = {
            let mut in_flight = self.in_flight.lock().unwrap();
            let mut fetch = Vec::new();
            let mut wait = Vec::new();
            for i in first..=last {
                if self.state.is_valid(i) {
                    continue; // Already cached
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

        // Wait for any chunks being fetched by other threads.
        // The condvar is paired with the `in_flight` mutex (the
        // worker calls `notify_all` after removing its entry from
        // the map). The release/acquire on the mutex ensures the
        // state-bit update is visible by the time we observe the
        // in_flight removal.
        for (chunk_idx, cv) in &to_wait {
            let mut in_flight = self.in_flight.lock().unwrap();
            while !self.state.is_valid(*chunk_idx) {
                if !in_flight.contains_key(chunk_idx) {
                    break; // Worker finished — re-check state
                }
                in_flight = cv.wait_timeout(in_flight,
                    std::time::Duration::from_millis(50)).unwrap().0;
            }
        }

        if to_fetch.is_empty() {
            return Ok(());
        }

        let shape = self.reference.shape();
        let requests: Vec<ChunkRequest> = to_fetch
            .iter()
            .map(|&i| ChunkRequest {
                index: i,
                start: shape.chunk_start(i),
                len: shape.chunk_len(i),
            })
            .collect();

        let total_bytes: u64 = requests.iter().map(|r| r.len).sum();
        let progress = DownloadProgress::new(total_bytes, requests.len() as u32);
        self.parallel_fetch_verify_write(&requests, &progress, |_| {})
    }

    /// Core worker-thread fetch pipeline.
    ///
    /// For each request, one worker thread performs the full
    /// **fetch → verify → write → mark → notify** sequence locally.
    /// SHA-256 verification and disk I/O happen on the same thread
    /// that did the network fetch, so no main-thread serial post-
    /// processing bottleneck.
    ///
    /// Invariants preserved from the previous (sequential)
    /// implementation:
    ///
    /// - **Differential merkle**: the caller has already filtered
    ///   out chunks that are valid or in-flight; this method
    ///   trusts that filter.
    /// - **In-flight dedup**: each successful or failing chunk
    ///   removes its entry from `in_flight` and notifies the
    ///   condvar so concurrent readers can proceed.
    /// - **Verify-before-trust**: a chunk's bytes are not exposed
    ///   to readers (and not committed to the state bitmap) until
    ///   SHA-256 against the merkle reference passes.
    /// - **Crash-safe resume**: the `.mrkl` file is saved
    ///   periodically (throttle: at most once per
    ///   `STATE_SAVE_INTERVAL_MS` from any worker, atomic CAS)
    ///   *and* unconditionally on completion, so a kill at any
    ///   moment loses at most a second of progress.
    /// - **Failure surfaces**: the first worker error short-
    ///   circuits the rest (subsequent workers see the abort
    ///   signal and bail) and is returned to the caller.
    ///
    /// `progress_cb` is invoked on a ticker (~4 Hz) so live UI
    /// updates land while the workers are still busy — the
    /// previous design only invoked it once at completion.
    fn parallel_fetch_verify_write<F>(
        &self,
        requests: &[ChunkRequest],
        progress: &DownloadProgress,
        mut progress_cb: F,
    ) -> io::Result<()>
    where
        F: FnMut(&DownloadProgress),
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;
        const STATE_SAVE_INTERVAL: std::time::Duration =
            std::time::Duration::from_millis(1000);
        const PROGRESS_TICK: std::time::Duration =
            std::time::Duration::from_millis(250);

        let abort = AtomicBool::new(false);
        let (tx, rx) = mpsc::channel::<io::Result<()>>();

        // Fixed-size worker pool. The previous implementation spawned
        // *one OS thread per ChunkRequest* and gated them through a
        // semaphore — fine for a few thousand chunks, but a 1.3 TiB
        // file at 8 MiB chunks is ~170 000 chunks, and pthread_create
        // refuses the spawn well before that (RLIMIT_NPROC, and each
        // thread's reserved stack would also exhaust virtual memory).
        // Workers now pull from a shared `Mutex<Vec<ChunkRequest>>`
        // queue, so the OS thread count equals `self.concurrency`
        // regardless of how many chunks are pending.
        let queue: std::sync::Mutex<Vec<ChunkRequest>> =
            std::sync::Mutex::new(requests.iter().rev().copied().collect());

        let result: io::Result<()> = std::thread::scope(|scope| {
            for _ in 0..self.concurrency.max(1) {
                let abort = &abort;
                let progress = &*progress;
                let this = self;
                let tx = tx.clone();
                let queue = &queue;

                scope.spawn(move || {
                    loop {
                        // Pull the next pending request. `pop` from
                        // the back gives the front of the requests
                        // slice because the queue was reversed at
                        // construction — keeps chunk-ordering close
                        // to the original sequential walk.
                        let req = match queue.lock().unwrap().pop() {
                            Some(r) => r,
                            None => break,
                        };

                        if abort.load(Ordering::Relaxed) {
                            // Still report so the main loop's
                            // done-counter can terminate.
                            let _ = tx.send(Ok(()));
                            continue;
                        }

                        let work = (|| -> io::Result<()> {
                            // ── 1. Fetch (with retry) ──────────
                            let data = match this.retry_policy.execute(|| {
                                this.transport.fetch_range(req.start, req.len)
                            }) {
                                Ok(d) => d,
                                Err(e) => {
                                    abort.store(true, Ordering::Relaxed);
                                    progress.mark_failed();
                                    this.drop_in_flight(req.index);
                                    return Err(e);
                                }
                            };

                            // ── 2. Verify (SHA-256 vs merkle ref) ──
                            if !this.reference.verify_chunk(req.index, &data) {
                                abort.store(true, Ordering::Relaxed);
                                progress.mark_failed();
                                this.drop_in_flight(req.index);
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("chunk {} failed integrity verification \
                                             (SHA-256 mismatch)", req.index)));
                            }

                            // ── 3. Write to cache file ─────────
                            // Positional write — on Unix this is a
                            // lock-free `pwrite(2)`, so workers
                            // don't serialize through a
                            // `Mutex<File>`. On Windows the
                            // wrapper takes a brief lock (no
                            // public OVERLAPPED-based API in std).
                            this.cache_file.write_all_at(&data, req.start)?;

                            // ── 4. Mark valid in state bitmap ──
                            // Lock-free: `MerkleState::mark_valid`
                            // does an atomic fetch_or on the
                            // bitmap word. Save is throttled on
                            // the main thread.
                            this.state.mark_valid(req.index);

                            progress.add_downloaded_bytes(data.len() as u64);
                            progress.increment_completed();

                            // ── 5. Notify in-flight waiters ────
                            this.drop_in_flight(req.index);
                            Ok(())
                        })();
                        let _ = tx.send(work);
                    }
                });
            }

            // Drop the spawner's tx so the channel closes once every
            // worker drops their clone — guards against hanging if
            // `done` undercounts due to a panic.
            drop(tx);

            // Main thread: collect chunk-completion events, fire
            // the user cb on each (or on timeout for ticker-style
            // updates), and run throttled .mrkl saves. All cb
            // calls happen here, so callers don't need a Send
            // bound — closures that touch !Send state work fine.
            let mut first_err: io::Result<()> = Ok(());
            let mut done: usize = 0;
            let mut last_save = std::time::Instant::now();
            while done < requests.len() {
                match rx.recv_timeout(PROGRESS_TICK) {
                    Ok(Ok(())) => { done += 1; progress_cb(progress); }
                    Ok(Err(e)) => {
                        done += 1;
                        if first_err.is_ok() { first_err = Err(e); }
                        progress_cb(progress);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        progress_cb(progress);
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if last_save.elapsed() >= STATE_SAVE_INTERVAL {
                    // No clone needed — `state.save` reads each
                    // bitmap word with `Relaxed` atomics, so it
                    // serialises a consistent-per-word snapshot
                    // even while workers are still flipping bits.
                    let _ = self.state.save(&self.state_path);
                    last_save = std::time::Instant::now();
                }
            }

            // One last cb so the caller always sees the final
            // (100% or failed) snapshot.
            progress_cb(progress);
            first_err
        });

        // Final save — even on error so partial progress is
        // preserved for resume.
        let _ = self.state.save(&self.state_path);

        // On error, clear any in-flight entries that aborted
        // workers didn't get to so blocked readers don't hang.
        if result.is_err() {
            let mut in_flight = self.in_flight.lock().unwrap();
            for req in requests {
                if let Some(cv) = in_flight.remove(&req.index) {
                    cv.notify_all();
                }
            }
        }

        result
    }

    /// Remove a chunk's in-flight entry and notify any blocked
    /// readers. Called from the worker thread after a chunk is
    /// successfully verified+written OR after a fatal error.
    fn drop_in_flight(&self, chunk_index: u32) {
        let cv = {
            let mut inf = self.in_flight.lock().unwrap();
            inf.remove(&chunk_index)
        };
        if let Some(cv) = cv { cv.notify_all(); }
    }

    /// Eagerly download and verify all unverified chunks.
    pub fn precache(&self) -> io::Result<()> {
        self.prebuffer_with_progress(|_| {})
    }

    /// Eagerly download all unverified chunks with a progress callback.
    ///
    /// The callback is invoked on a ticker (~4 Hz) while workers
    /// are downloading, *and* once more at completion — so long
    /// downloads emit live progress rather than going silent until
    /// every chunk lands. Registering missing chunks in the
    /// in-flight map first prevents a concurrent `read()` from
    /// double-fetching the same range while precache is running.
    pub fn prebuffer_with_progress<F: FnMut(&DownloadProgress)>(
        &self,
        callback: F,
    ) -> io::Result<()> {
        // Take a snapshot of which chunks need work, then claim
        // them in the in-flight map under the same lock that the
        // on-demand `ensure_chunks_valid` path uses. This is the
        // guard against the race "precache starts fetching a
        // chunk → on-demand reader sees `state.is_valid()=false`
        // and `in_flight.get()=None` → fires its own duplicate
        // fetch".
        let missing = {
            let mut in_flight = self.in_flight.lock().unwrap();
            let mut out = Vec::new();
            for i in self.state.missing_chunks() {
                if in_flight.contains_key(&i) {
                    // Another path already owns this chunk's
                    // fetch — let it race to completion; we'll
                    // skip it here.
                    continue;
                }
                in_flight.insert(i, Arc::new(Condvar::new()));
                out.push(i);
            }
            out
        };

        if missing.is_empty() {
            // Nothing to do — still fire the cb once so callers
            // see a "completed" snapshot.
            let shape = self.reference.shape();
            let progress = DownloadProgress::new(0, 0);
            let _ = progress;
            let mut cb = callback;
            cb(&DownloadProgress::new(shape.total_content_size, 0));
            return Ok(());
        }

        let shape = self.reference.shape();
        let requests: Vec<ChunkRequest> = missing
            .iter()
            .map(|&i| ChunkRequest {
                index: i,
                start: shape.chunk_start(i),
                len: shape.chunk_len(i),
            })
            .collect();

        let total_bytes: u64 = requests.iter().map(|r| r.len).sum();
        let progress = DownloadProgress::new(total_bytes, requests.len() as u32);
        self.parallel_fetch_verify_write(&requests, &progress, callback)
    }

    /// Whether all chunks have been downloaded and verified.
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Number of verified chunks.
    pub fn valid_count(&self) -> u32 {
        self.state.valid_count()
    }

    /// Total number of chunks.
    pub fn total_chunks(&self) -> u32 {
        self.reference.shape().total_chunks
    }

    /// Path to the local cache file.
    pub fn cache_path(&self) -> &Path {
        &self.cache_path
    }

    /// Path to the merkle state file.
    pub fn state_path(&self) -> &Path {
        &self.state_path
    }

    /// Total content size in bytes.
    pub fn content_size(&self) -> u64 {
        self.reference.shape().total_content_size
    }

    /// The merkle reference tree.
    pub fn reference(&self) -> &MerkleRef {
        &self.reference
    }

    /// Ensure chunks in range `[first, last]` (inclusive) are valid.
    /// Public wrapper for on-demand fetching.
    pub fn ensure_range(&self, first: u32, last: u32) -> io::Result<()> {
        self.ensure_chunks_valid(first, last)
    }

    /// Number of chunks currently being fetched by background threads.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.lock().map(|m| m.len()).unwrap_or(0)
    }
}

// CachedChannel holds Mutex<File> + Mutex<MerkleState>, both Send.
// The Mutexes ensure thread safety.
unsafe impl Send for CachedChannel {}
unsafe impl Sync for CachedChannel {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::MerkleRef;
    use crate::transport::ChunkedTransport;

    /// In-memory transport for testing cached channel behavior.
    struct MemoryTransport {
        data: Vec<u8>,
    }

    impl MemoryTransport {
        fn new(data: Vec<u8>) -> Self {
            MemoryTransport { data }
        }
    }

    impl ChunkedTransport for MemoryTransport {
        fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
            let s = start as usize;
            let e = s + len as usize;
            if e > self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("range [{}, {}) exceeds content length {}", s, e, self.data.len()),
                ));
            }
            Ok(self.data[s..e].to_vec())
        }

        fn content_length(&self) -> io::Result<u64> {
            Ok(self.data.len() as u64)
        }

        fn supports_range(&self) -> bool {
            true
        }
    }

    /// Create test data, merkle ref, and a CachedChannel in a temp dir.
    fn setup_cached_channel(
        data: &[u8],
        chunk_size: u64,
    ) -> (tempfile::TempDir, CachedChannel) {
        let dir = tempfile::tempdir().unwrap();
        let mref = MerkleRef::from_content(data, chunk_size);
        let transport = Arc::new(MemoryTransport::new(data.to_vec()));
        let channel = CachedChannel::open(transport, mref, dir.path(), "test.dat").unwrap();
        (dir, channel)
    }

    #[test]
    fn test_mref_content_size_mismatch_detected() {
        // Build a merkle ref for 1024 bytes of content, but provide a transport
        // that only has 512 bytes. When we try to read beyond the transport's
        // content, the fetch should fail.
        let real_data = vec![0xABu8; 512];
        let fake_data = vec![0xABu8; 1024];
        let mref = MerkleRef::from_content(&fake_data, 256);

        let dir = tempfile::tempdir().unwrap();
        let transport = Arc::new(MemoryTransport::new(real_data));
        let channel = CachedChannel::open(transport, mref, dir.path(), "mismatch.dat")
            .unwrap()
            .with_retry_policy(crate::transport::RetryPolicy {
                max_retries: 1,
                base_delay_ms: 1,
                max_delay_ms: 1,
                jitter_fraction: 0.0,
            });

        // Reading from chunk 2 (offset 512) should fail because the transport
        // only has 512 bytes
        let result = channel.read(512, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_download_resume_with_partial_cache() {
        let chunk_size = 256u64;
        let data: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let mref = MerkleRef::from_content(&data, chunk_size);

        let dir = tempfile::tempdir().unwrap();

        // Phase 1: open channel, download first two chunks, then drop
        {
            let transport = Arc::new(MemoryTransport::new(data.clone()));
            let channel = CachedChannel::open(
                transport, mref.clone(), dir.path(), "resume.dat",
            ).unwrap();

            // Read from the first two chunks to trigger download
            let chunk0 = channel.read(0, 256).unwrap();
            assert_eq!(&chunk0, &data[0..256]);
            let chunk1 = channel.read(256, 256).unwrap();
            assert_eq!(&chunk1, &data[256..512]);

            assert_eq!(channel.valid_count(), 2);
            assert!(!channel.is_complete());
        }
        // channel is dropped, state is persisted

        // Phase 2: reopen — should resume from checkpoint
        {
            let transport = Arc::new(MemoryTransport::new(data.clone()));
            let channel = CachedChannel::open(
                transport, mref.clone(), dir.path(), "resume.dat",
            ).unwrap();

            // Should already have 2 valid chunks from the previous session
            assert_eq!(channel.valid_count(), 2);

            // Read remaining chunks
            let chunk2 = channel.read(512, 256).unwrap();
            assert_eq!(&chunk2, &data[512..768]);
            let chunk3 = channel.read(768, 256).unwrap();
            assert_eq!(&chunk3, &data[768..1024]);

            assert!(channel.is_complete());
        }
    }

    #[test]
    fn test_basic_read_through() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let (_dir, channel) = setup_cached_channel(&data, 512);

        // Read that spans two chunks
        let result = channel.read(400, 300).unwrap();
        assert_eq!(&result, &data[400..700]);
        assert!(channel.valid_count() >= 2);
    }

    // ─── Perf harness ────────────────────────────────────────────
    //
    // Self-contained throughput probe — *not* a Criterion bench so
    // we don't add a dep just for a perf check. Inject controllable
    // latency per `fetch_range` call to simulate LAN / S3 / cross-
    // region transports, then measure precache wall-clock at each
    // concurrency level.
    //
    // Run with:
    //   cargo test --release -p vectordata --lib bench_prebuffer_throughput \
    //              -- --nocapture --ignored

    struct LatencyTransport {
        data: Vec<u8>,
        latency_ms: u64,
        /// Total fetch_range calls observed — exposed so the test
        /// can assert no duplicate fetches under the in-flight
        /// dedup contract.
        fetch_calls: std::sync::atomic::AtomicU64,
        /// Total bytes returned across all fetches.
        bytes_served: std::sync::atomic::AtomicU64,
    }

    impl LatencyTransport {
        fn new(data: Vec<u8>, latency_ms: u64) -> Self {
            Self {
                data, latency_ms,
                fetch_calls: std::sync::atomic::AtomicU64::new(0),
                bytes_served: std::sync::atomic::AtomicU64::new(0),
            }
        }
    }

    impl ChunkedTransport for LatencyTransport {
        fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
            self.fetch_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.latency_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.latency_ms));
            }
            let s = start as usize;
            let e = s + len as usize;
            if e > self.data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "range past EOF"));
            }
            self.bytes_served.fetch_add(len, std::sync::atomic::Ordering::Relaxed);
            Ok(self.data[s..e].to_vec())
        }
        fn content_length(&self) -> io::Result<u64> { Ok(self.data.len() as u64) }
        fn supports_range(&self) -> bool { true }
    }

    /// Stress-test the in-flight dedup invariant under concurrent
    /// reads + precache. Multiple reader threads request
    /// overlapping ranges while a precache is in flight; the
    /// `fetch_calls` counter on the transport must end up equal to
    /// `n_chunks` exactly — no double-fetches, no missed chunks.
    #[test]
    fn parallel_reads_dedup_with_prebuffer() {
        let chunk_size: u64 = 1024;
        let n_chunks: usize = 32;
        let data_size = chunk_size as usize * n_chunks;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 251) as u8).collect();
        let mref = MerkleRef::from_content(&data, chunk_size);

        let dir = tempfile::tempdir().unwrap();
        let transport = Arc::new(LatencyTransport::new(data.clone(), 5));
        let inspect = transport.clone(); // outlives `transport`, used for assertions below.
        let channel = std::sync::Arc::new(
            CachedChannel::open(transport, mref.clone(), dir.path(), "concurrent.dat")
                .unwrap()
                .with_concurrency(8));

        // Spawn 8 reader threads + 1 precache thread, all racing
        // to drive the same set of chunks resident.
        let mut handles = Vec::new();
        for t in 0..8 {
            let ch = channel.clone();
            handles.push(std::thread::spawn(move || {
                // Each thread reads a different overlapping range.
                let offset = (t as u64) * (chunk_size / 2);
                let len = chunk_size * (n_chunks as u64 / 2);
                let _ = ch.read(offset, len).expect("read should succeed");
            }));
        }
        {
            let ch = channel.clone();
            handles.push(std::thread::spawn(move || {
                ch.precache().expect("precache should succeed");
            }));
        }
        for h in handles { h.join().unwrap(); }

        // All chunks must be valid; no duplicate transport calls.
        assert!(channel.is_complete());
        let fetches = inspect.fetch_calls.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(fetches, n_chunks as u64,
            "in-flight dedup invariant violated: {fetches} fetches for {n_chunks} chunks");
    }

    /// Throughput probe across concurrency levels. Prints a single
    /// table to stdout. Marked `#[ignore]` so it doesn't run in the
    /// default test suite (the sleeps add ~30s wall-clock).
    #[test]
    #[ignore]
    fn bench_prebuffer_throughput() {
        // 200 chunks × 1 MiB = 200 MiB. Big enough that
        // overheads matter; small enough that the test finishes
        // in tens of seconds.
        let chunk_size: u64 = 1 * 1024 * 1024;
        let n_chunks: usize = 200;
        let data_size = chunk_size as usize * n_chunks;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 251) as u8).collect();
        let mref = MerkleRef::from_content(&data, chunk_size);

        println!();
        println!("=== precache throughput (200 MiB, 200×1 MiB chunks) ===");
        println!("{:<18} {:>12} {:>14} {:>14} {:>12}",
            "latency_ms x conc", "fetch calls", "wall_ms", "MB/s", "dup_calls");

        for &latency_ms in &[5u64, 50, 200] {
            for &concurrency in &[1usize, 4, 8, 16, 32, 64] {
                // Fresh tempdir per run so we always start cold.
                let dir = tempfile::tempdir().unwrap();
                let transport = Arc::new(LatencyTransport::new(data.clone(), latency_ms));
                let inspect = transport.clone();
                let channel = CachedChannel::open(
                    transport, mref.clone(), dir.path(), "bench.dat",
                ).unwrap()
                 .with_concurrency(concurrency);

                let start = std::time::Instant::now();
                channel.precache().unwrap();
                let elapsed = start.elapsed();

                let fetches = inspect.fetch_calls.load(std::sync::atomic::Ordering::Relaxed);
                let dup = fetches.saturating_sub(n_chunks as u64);
                let mb_per_s = (data_size as f64 / (1024.0 * 1024.0))
                    / elapsed.as_secs_f64();

                println!("{:>5}ms x {:>2}     {:>12} {:>14} {:>14.1} {:>12}",
                    latency_ms, concurrency,
                    fetches, elapsed.as_millis(), mb_per_s, dup);

                // Invariant: in-flight dedup must prevent duplicate
                // fetches. Even a single-threaded sequential
                // precache should issue exactly n_chunks calls.
                assert_eq!(fetches, n_chunks as u64,
                    "duplicate fetches detected at conc={concurrency}");
            }
        }
        println!();
    }

    #[test]
    fn test_full_prebuffer() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let (_dir, channel) = setup_cached_channel(&data, 1024);

        assert!(!channel.is_complete());
        channel.precache().unwrap();
        assert!(channel.is_complete());

        // Verify all data is correct
        let all = channel.read(0, 4096).unwrap();
        assert_eq!(&all, &data);
    }

    /// Reopening a cache dir with a different upstream merkle ref must
    /// fail loudly, not silently serve the previously cached bytes.
    /// This is the regression test for the stale-mref bug that the
    /// `<host>:<port>/` cache directory naming used to paper over.
    #[test]
    fn test_open_with_mismatched_reference_errors() {
        let chunk_size = 256u64;
        let original: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let original_ref = MerkleRef::from_content(&original, chunk_size);

        let dir = tempfile::tempdir().unwrap();

        // First open: populate cache against the original ref.
        {
            let transport = Arc::new(MemoryTransport::new(original.clone()));
            let channel = CachedChannel::open(
                transport, original_ref.clone(), dir.path(), "asset.dat",
            ).unwrap();
            channel.precache().unwrap();
            assert!(channel.is_complete());
        }

        // Reopen with a *different* upstream — same URL, new content.
        // This is exactly the scenario the old port-isolation hack
        // hid: now we surface it instead of trusting the stale state.
        let replaced: Vec<u8> = (0..1024).map(|i| ((i + 7) % 251) as u8).collect();
        let replaced_ref = MerkleRef::from_content(&replaced, chunk_size);
        assert_ne!(original_ref.root_hash(), replaced_ref.root_hash());

        let transport = Arc::new(MemoryTransport::new(replaced));
        let err = match CachedChannel::open(
            transport, replaced_ref, dir.path(), "asset.dat",
        ) {
            Ok(_) => panic!("expected stale-cache error, got Ok"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(msg.contains("stale"), "expected stale-cache phrasing: {msg}");
        assert!(msg.contains("Delete"), "expected actionable Delete hint: {msg}");
    }

    /// Reopening a cache dir with the *same* upstream merkle ref must
    /// succeed and reuse the cached state (no re-download).
    #[test]
    fn test_open_with_matching_reference_resumes() {
        let chunk_size = 256u64;
        let data: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let mref = MerkleRef::from_content(&data, chunk_size);

        let dir = tempfile::tempdir().unwrap();
        {
            let transport = Arc::new(MemoryTransport::new(data.clone()));
            let channel = CachedChannel::open(
                transport, mref.clone(), dir.path(), "asset.dat",
            ).unwrap();
            channel.precache().unwrap();
        }
        let transport = Arc::new(MemoryTransport::new(data.clone()));
        let channel = CachedChannel::open(
            transport, mref, dir.path(), "asset.dat",
        ).unwrap();
        assert!(channel.is_complete(), "matching reopen should resume from valid state");
    }
}
