// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fast directory-of-xvec → single-xvec gather, optimized for EBS.
//!
//! ## Architecture: parallel prefetch + single sequential writer
//!
//! EBS rewards **large sequential writes** to a single inode. Spraying
//! `pwrite`s from many threads at scattered offsets in one output file
//! looks like a high-IOPS scatter pattern from the volume's perspective:
//! the kernel may flush dirty pages out of offset order, the volume
//! sees fragmented I/O, and effective throughput collapses well below
//! the volume's nominal bandwidth.
//!
//! So the writer here is **single-threaded** and walks the input shards
//! in natural-sort order, issuing `copy_file_range` calls at strictly
//! monotonically increasing offsets in the output. From the volume's
//! point of view, the output is one continuous forward-streaming write —
//! exactly the pattern that lets the kernel coalesce dirty pages into
//! large sequential I/Os.
//!
//! Reads, on the other hand, *do* parallelize cleanly: each source shard
//! is a different inode, so the kernel issues independent I/O requests
//! across them. A small pool of prefetch threads runs ahead of the
//! writer, force-reading upcoming files into the page cache so the
//! writer's `copy_file_range` sees a cache hit on the source side and
//! is bottlenecked only by the EBS write path (not by source-read
//! latency).
//!
//! ## Other EBS-specific touches
//!
//! - Source files are advised `POSIX_FADV_SEQUENTIAL` before prefetch
//!   so the kernel widens its readahead window.
//! - After the writer copies a shard, both source and dest ranges are
//!   advised `POSIX_FADV_DONTNEED`, which lets writeback drain the dest
//!   pages and stops the growing output from displacing prefetched
//!   source pages from the page cache.
//! - The output is pre-sized via `set_len` so the kernel doesn't have
//!   to grow the file inline; copies always land at known offsets.
//! - Atomic-rename publish via `<output>.tmp` so a crash mid-run never
//!   leaves a half-written file at the published path.

use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use super::VecFormat;
use super::parquet_vector_compiler::{
    LogCallback, advise_dontneed, with_tmp_suffix,
};

/// Progress callback for the concat path. Reports both file granularity
/// (`shards_delta`, `shards_total` for the file-count progress bar) and
/// running record count (`records_total`) so callers can derive a
/// records-per-second rate independent of file size variance.
pub type ConcatProgress<'a> = &'a (dyn Fn(usize, u64, u64) + Sync);

/// Maximum reader threads regardless of caller request.
const MAX_LOADER_THREADS: usize = 64;

/// Number of prefetch reader threads. Reads can run concurrently — each
/// source is a different inode, so the kernel issues parallel I/O across
/// them — and that hides per-file open + readahead-warmup latency. The
/// writer is single-threaded by design (see module note on EBS write
/// patterns), so going higher here just keeps more pages cached without
/// helping write throughput. 4–8 saturates EBS read bandwidth on
/// gp3/io2 in practice.
const PREFETCH_THREADS: usize = 8;

/// Max number of files prefetched into the page cache ahead of where
/// the writer is currently working. Bounds the page-cache footprint to
/// roughly `MAX_LOOKAHEAD × shard_size`. At 120 MiB shards × 16 ahead
/// that's ~2 GiB resident — comfortable on any EBS host.
const MAX_LOOKAHEAD: usize = 16;

/// Discard buffer size used by reader threads to force file pages into
/// the page cache via `read()` syscalls. `posix_fadvise(WILLNEED)`
/// alone doesn't reliably populate large files (it's bounded by the
/// per-BDI `read_ahead_kb`, typically 128 KiB), so we top it up with
/// real reads into a throwaway buffer. The buffer itself is per-thread,
/// reused across files. 4 MiB matches Linux's default writeback chunk
/// and is large enough to amortize syscall overhead.
const PREFETCH_CHUNK: usize = 4 * 1024 * 1024;

/// Granularity at which the writer chunks `copy_file_range` calls and
/// kicks off writeback. EBS achieves peak sustained throughput when
/// the volume's write queue is *continuously* fed with large I/Os —
/// the kernel issues writeback in 1–4 MiB pieces, and what we want
/// is for those pieces to stream to disk back-to-back without ever
/// stalling on "no work queued."
///
/// 16 MiB is the sweet spot for that: large enough that the syscall
/// overhead is negligible (one syscall per 16 MiB of payload), small
/// enough that writeback for chunk *N* starts well before chunk
/// *N+1*'s data lands in the page cache. The pipeline fills steadily
/// instead of accumulating a giant dirty backlog and then bursting.
const WRITE_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// Cadence of progress-callback ticks from the ticker thread. Workers bump
/// a shared atomic per shard completed; the ticker reads it on this
/// interval and emits one `cb(delta, total)`. Decouples UI from the
/// per-file hot path.
const PROGRESS_TICK: std::time::Duration = std::time::Duration::from_millis(250);

/// Per-source metadata gathered during the probe phase.
#[derive(Debug)]
pub struct XvecDirProbe {
    pub files: Vec<PathBuf>,
    pub per_file_bytes: Vec<u64>,
    pub per_file_records: Vec<u64>,
    pub dimension: u32,
    pub element_size: usize,
    pub format: VecFormat,
}

impl XvecDirProbe {
    pub fn total_bytes(&self) -> u64 {
        self.per_file_bytes.iter().sum()
    }
    pub fn total_records(&self) -> u64 {
        self.per_file_records.iter().sum()
    }
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

/// Inspect a directory of xvec shards to determine: which files belong,
/// the shared dimension, and the per-file record counts. All cheap —
/// one `stat()` and one 4-byte read per file.
///
/// Validates that every file has the same `[dim]` header and a size that
/// is an exact multiple of the stride — the precondition for the fast
/// concat path. Anything else returns an error so the caller can fall
/// back to the slow per-record reader.
pub fn probe_xvec_directory(
    source: &Path,
    expected_format: VecFormat,
) -> Result<XvecDirProbe, String> {
    if !source.is_dir() {
        return Err(format!("{} is not a directory", source.display()));
    }
    let element_size = expected_format.element_size();
    if element_size == 0 || !expected_format.is_uniform_xvec() {
        return Err(format!(
            "format {} is not a uniform xvec format",
            expected_format.name(),
        ));
    }

    let mut files = collect_xvec_files(source, expected_format)?;
    if files.is_empty() {
        return Err(format!(
            "no .{} files found under {}",
            expected_format.preferred_extension(),
            source.display(),
        ));
    }
    files.sort_by(|a, b| natural_path_cmp(a, b));

    // Probe the first file's dim header so we can compute stride and
    // validate every other file against it.
    let dim = read_first_dim(&files[0])?;
    if dim <= 0 {
        return Err(format!(
            "first file {} reports non-positive dim {}",
            files[0].display(),
            dim,
        ));
    }
    let stride: u64 = 4 + dim as u64 * element_size as u64;

    let mut per_file_bytes = Vec::with_capacity(files.len());
    let mut per_file_records = Vec::with_capacity(files.len());
    for f in &files {
        let len = fs::metadata(f)
            .map_err(|e| format!("stat {}: {}", f.display(), e))?
            .len();
        if len == 0 {
            return Err(format!("file {} is empty", f.display()));
        }
        if len % stride != 0 {
            return Err(format!(
                "file {} size {} is not a multiple of stride {} (dim={}, element_size={}); \
                 file is variable-length or has a different dim — fast concat path requires \
                 uniform shards with the same dimension",
                f.display(), len, stride, dim, element_size,
            ));
        }
        // Cross-file dim check: peek the first 4 bytes of every file. Cheap
        // (one cached page read per file) and catches a corpus that mixes
        // dimensions, which would silently produce a corrupt output.
        let file_dim = read_first_dim(f)?;
        if file_dim != dim {
            return Err(format!(
                "file {} has dim {} but first file {} has dim {} — \
                 fast concat path requires every shard to share a dimension",
                f.display(), file_dim, files[0].display(), dim,
            ));
        }
        per_file_bytes.push(len);
        per_file_records.push(len / stride);
    }

    Ok(XvecDirProbe {
        files,
        per_file_bytes,
        per_file_records,
        dimension: dim as u32,
        element_size,
        format: expected_format,
    })
}

/// Concatenate a directory of uniform xvec shards into a single output
/// xvec file. Workers run in parallel; each reads one shard fully with
/// kernel readahead hints and `pwrite`s it to its pre-computed offset
/// in `<output>.tmp`. The output is atomically renamed into place once
/// every shard has landed.
///
/// Returns the total record count on success.
pub fn extract_xvec_dir_to_xvec_threaded(
    source: &Path,
    output: &Path,
    target_format: VecFormat,
    progress: Option<ConcatProgress<'_>>,
    log: Option<LogCallback<'_>>,
    threads: usize,
) -> Result<u64, String> {
    let probe = probe_xvec_directory(source, target_format)?;
    let total_bytes = probe.total_bytes();
    let total_records = probe.total_records();
    let total_files = probe.file_count();

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create dir {}: {}", parent.display(), e))?;
        }
    }

    // Idempotent fast-exit: with the atomic-rename pattern, the existence
    // of `output` at the right size is the contract of a completed run.
    if let Ok(meta) = fs::metadata(output) {
        if meta.len() == total_bytes {
            if let Some(cb) = progress {
                cb(total_files, total_files as u64, total_records);
            }
            return Ok(total_records);
        }
    }

    // Pre-compute the per-file output offset by cumulative-sum of shard
    // sizes. Workers can then write to disjoint regions of the output
    // without any coordination.
    let mut per_file_offset = Vec::with_capacity(total_files);
    let mut acc: u64 = 0;
    for &b in &probe.per_file_bytes {
        per_file_offset.push(acc);
        acc += b;
    }
    debug_assert_eq!(acc, total_bytes);

    let output_tmp = with_tmp_suffix(output);
    if output_tmp.exists() {
        let _ = fs::remove_file(&output_tmp);
    }

    let out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_tmp)
        .map_err(|e| format!("create {}: {}", output_tmp.display(), e))?;
    out_file
        .set_len(total_bytes)
        .map_err(|e| format!("set_len {}: {}", output_tmp.display(), e))?;
    let out_file = Arc::new(out_file);

    // copy_file_range only works when source and destination live on
    // the same filesystem (it returns EXDEV otherwise). Compare the
    // device IDs once, up front, and pick the per-shard copy strategy
    // accordingly: native kernel copy when same-device, chunked
    // read+pwrite (still streaming, still kicked-writeback) when not.
    let same_device = same_filesystem(&probe.files[0], &output_tmp);
    let copy_mode = if same_device {
        CopyMode::KernelCopyFileRange
    } else {
        CopyMode::ReadPwriteFallback
    };

    // `threads` is reused as a soft cap on prefetch concurrency. The
    // writer is always single-threaded by design.
    let prefetch_threads = threads
        .min(PREFETCH_THREADS)
        .min(total_files.saturating_sub(1).max(1))
        .max(1);

    if let Some(cb) = log {
        cb(&format!(
            "phase 1: gathering {} xvec shards ({} bytes, dim {}, element {} B) \
             into {} ({} prefetch thread(s) + 1 sequential writer, lookahead {}, copy via {})",
            total_files,
            total_bytes,
            probe.dimension,
            probe.element_size,
            output.display(),
            prefetch_threads,
            MAX_LOOKAHEAD,
            copy_mode.label(),
        ));
    }

    let files_shared = Arc::new(probe.files.clone());
    let bytes_shared = Arc::new(probe.per_file_bytes.clone());
    let records_shared = Arc::new(probe.per_file_records.clone());
    let offsets_shared = Arc::new(per_file_offset);

    // Coordination state.
    //   - `next_to_prefetch`: next file index a prefetch thread will claim.
    //   - `writer_advance`:   files the writer has fully copied — caps
    //                         how far ahead prefetchers may run.
    //   - `progress_total` / `records_total_atomic`: bumped by the
    //                         writer for the UI ticker.
    let next_to_prefetch = Arc::new(AtomicUsize::new(0));
    let writer_advance = Arc::new(AtomicUsize::new(0));
    let progress_total = Arc::new(AtomicU64::new(0));
    let records_total_atomic = Arc::new(AtomicU64::new(0));
    let progress_done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let first_err: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    thread::scope(|scope| {
        // Ticker → UI bridge. Same pattern as the parquet path.
        let ticker_total = Arc::clone(&progress_total);
        let ticker_records = Arc::clone(&records_total_atomic);
        let ticker_done = Arc::clone(&progress_done);
        let ticker_progress = progress;
        let ticker_handle = scope.spawn(move || {
            let mut last_seen: u64 = 0;
            loop {
                let cur = ticker_total.load(Ordering::Relaxed);
                let recs = ticker_records.load(Ordering::Relaxed);
                if cur > last_seen {
                    if let Some(cb) = ticker_progress {
                        cb((cur - last_seen) as usize, cur, recs);
                    }
                    last_seen = cur;
                }
                if ticker_done.load(Ordering::Relaxed) {
                    let cur = ticker_total.load(Ordering::Relaxed);
                    let recs = ticker_records.load(Ordering::Relaxed);
                    if cur > last_seen {
                        if let Some(cb) = ticker_progress {
                            cb((cur - last_seen) as usize, cur, recs);
                        }
                    }
                    break;
                }
                thread::sleep(PROGRESS_TICK);
            }
        });

        // Prefetch pool: pull upcoming source files into the page cache
        // so the writer's copy_file_range hits warm pages and stays
        // bottlenecked only on the EBS write side. Each thread claims
        // file indices via fetch_add and force-reads the file via a
        // discard buffer (more reliable than POSIX_FADV_WILLNEED, which
        // only honors the per-BDI readahead window).
        let mut prefetch_handles = Vec::with_capacity(prefetch_threads);
        for _ in 0..prefetch_threads {
            let next_to_prefetch = Arc::clone(&next_to_prefetch);
            let writer_advance = Arc::clone(&writer_advance);
            let files_shared = Arc::clone(&files_shared);
            let stop_flag = Arc::clone(&stop_flag);
            let first_err = Arc::clone(&first_err);
            prefetch_handles.push(scope.spawn(move || {
                prefetch_loop(
                    next_to_prefetch,
                    writer_advance,
                    files_shared,
                    stop_flag,
                    first_err,
                    total_files,
                );
            }));
        }

        // Single writer thread. Walks file indices 0..total in order
        // and writes them to the output at strictly monotonic offsets
        // — this is the EBS-friendly write pattern.
        let writer_files = Arc::clone(&files_shared);
        let writer_bytes = Arc::clone(&bytes_shared);
        let writer_records = Arc::clone(&records_shared);
        let writer_offsets = Arc::clone(&offsets_shared);
        let writer_advance_w = Arc::clone(&writer_advance);
        let writer_progress = Arc::clone(&progress_total);
        let writer_records_total = Arc::clone(&records_total_atomic);
        let writer_stop = Arc::clone(&stop_flag);
        let writer_err = Arc::clone(&first_err);
        let writer_out = Arc::clone(&out_file);
        let writer_handle = scope.spawn(move || {
            sequential_writer_loop(
                writer_files,
                writer_bytes,
                writer_records,
                writer_offsets,
                writer_advance_w,
                writer_progress,
                writer_records_total,
                writer_stop,
                writer_err,
                writer_out,
                total_files,
                copy_mode,
                log,
            );
        });

        let _ = writer_handle.join();
        // Writer is done — wake any prefetcher still parked on
        // backpressure so they exit cleanly.
        stop_flag.store(true, Ordering::Relaxed);
        for h in prefetch_handles { let _ = h.join(); }

        progress_done.store(true, Ordering::Relaxed);
        let _ = ticker_handle.join();
    });

    if let Some(e) = first_err.lock().unwrap().take() {
        return Err(e);
    }

    // Crash-safety boundary: drain every byte we wrote to disk before
    // renaming the .tmp into place. Without this, a power-loss window
    // exists where the rename is journal-committed but writeback hasn't
    // reached EBS yet, leaving the final-path file at correct size with
    // stale blocks that the next-run early-exit would mistake for a
    // completed run. fdatasync (`sync_data`) on the output is enough —
    // we don't need fsync since we have no metadata to persist beyond
    // what `set_len` already captured.
    out_file.sync_data().map_err(|e| {
        format!("fdatasync {}: {}", output_tmp.display(), e)
    })?;
    drop(out_file);

    // Atomic publish: rename <output>.tmp → <output>. Until the rename
    // succeeds, the final path does not exist — no observer can see a
    // partial write.
    fs::rename(&output_tmp, output).map_err(|e| {
        format!("rename {} → {}: {}", output_tmp.display(), output.display(), e)
    })?;

    if let Some(cb) = log {
        cb(&format!(
            "phase 2: published {} ({} bytes, {} records)",
            output.display(), total_bytes, total_records,
        ));
    }

    Ok(total_records)
}

/// Prefetch worker. Pulls upcoming source files into the page cache via
/// real `read()` syscalls (advisory `POSIX_FADV_WILLNEED` is unreliable
/// for files larger than the per-BDI readahead window). Backpressures
/// off the writer so the cached working set is bounded.
fn prefetch_loop(
    next_to_prefetch: Arc<AtomicUsize>,
    writer_advance: Arc<AtomicUsize>,
    files: Arc<Vec<PathBuf>>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    first_err: Arc<Mutex<Option<String>>>,
    total_files: usize,
) {
    let mut buf = vec![0u8; PREFETCH_CHUNK];
    loop {
        if stop_flag.load(Ordering::Relaxed) { break; }
        let idx = next_to_prefetch.fetch_add(1, Ordering::Relaxed);
        if idx >= total_files { break; }

        // Backpressure: don't run more than MAX_LOOKAHEAD ahead of the
        // writer. Sleep-poll is fine here — events are large (whole
        // shards) so 5 ms granularity costs nothing relative to the
        // copy-time of a 120 MiB shard.
        while !stop_flag.load(Ordering::Relaxed) {
            let advanced = writer_advance.load(Ordering::Relaxed);
            if idx < advanced + MAX_LOOKAHEAD { break; }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        if stop_flag.load(Ordering::Relaxed) { break; }

        let path = &files[idx];
        let result: Result<(), String> = (|| {
            let f = File::open(path)
                .map_err(|e| format!("prefetch open {}: {}", path.display(), e))?;
            // Widen the kernel's readahead window for this fd before
            // pulling pages — turns the discard reads below into the
            // large sequential I/O pattern EBS rewards.
            unsafe {
                libc::posix_fadvise(
                    f.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL,
                );
                libc::posix_fadvise(
                    f.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED,
                );
            }
            let mut reader = f;
            loop {
                let n = std::io::Read::read(&mut reader, &mut buf)
                    .map_err(|e| format!("prefetch read {}: {}", path.display(), e))?;
                if n == 0 { break; }
            }
            Ok(())
        })();

        if let Err(e) = result {
            let mut slot = first_err.lock().unwrap();
            if slot.is_none() { *slot = Some(e); }
            stop_flag.store(true, Ordering::Relaxed);
            break;
        }
    }
}

/// Per-shard copy strategy selected by the orchestrator based on
/// whether source and destination are on the same filesystem.
#[derive(Debug, Clone, Copy)]
enum CopyMode {
    /// `copy_file_range` — kernel-side, no user-space buffer. Requires
    /// source and destination on the same filesystem.
    KernelCopyFileRange,
    /// Chunked `read` into a scratch buffer + `pwrite` into the output.
    /// Used when the kernel-side path returns EXDEV. Same chunking and
    /// `sync_file_range` cadence as the kernel path, just with one
    /// extra memcpy per chunk.
    ReadPwriteFallback,
}

impl CopyMode {
    fn label(self) -> &'static str {
        match self {
            CopyMode::KernelCopyFileRange => "copy_file_range",
            CopyMode::ReadPwriteFallback => "read+pwrite (cross-filesystem)",
        }
    }
}

/// Same-filesystem detection for choosing between `copy_file_range`
/// and the read+pwrite fallback. We compare device IDs of the source
/// file and the destination's parent directory (the destination file
/// itself was just created so its st_dev is reliable too, but the
/// parent works whether or not the dest exists).
fn same_filesystem(src: &Path, dst: &Path) -> bool {
    let dst_dev = match fs::metadata(dst) {
        Ok(m) => m.dev(),
        Err(_) => match dst.parent().and_then(|p| fs::metadata(p).ok()) {
            Some(m) => m.dev(),
            None => return false,
        },
    };
    match fs::metadata(src) {
        Ok(m) => m.dev() == dst_dev,
        Err(_) => false,
    }
}

/// Single writer thread. Walks file indices 0..total in order and
/// streams each into the output at strictly monotonically increasing
/// offsets, so the EBS volume sees one continuous forward-streaming
/// write — the pattern its writeback path is designed around.
#[allow(clippy::too_many_arguments)]
fn sequential_writer_loop(
    files: Arc<Vec<PathBuf>>,
    bytes: Arc<Vec<u64>>,
    records: Arc<Vec<u64>>,
    offsets: Arc<Vec<u64>>,
    writer_advance: Arc<AtomicUsize>,
    progress_total: Arc<AtomicU64>,
    records_total_atomic: Arc<AtomicU64>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    first_err: Arc<Mutex<Option<String>>>,
    out_file: Arc<File>,
    total_files: usize,
    copy_mode: CopyMode,
    log: Option<LogCallback<'_>>,
) {
    // Single persistent scratch buffer for the read+pwrite fallback.
    // Allocated once, reused across every shard. Only touched when
    // `copy_mode == ReadPwriteFallback`, so same-filesystem runs pay
    // no allocation cost.
    let mut scratch: Vec<u8> = match copy_mode {
        CopyMode::KernelCopyFileRange => Vec::new(),
        CopyMode::ReadPwriteFallback => vec![0u8; WRITE_CHUNK_BYTES as usize],
    };
    for idx in 0..total_files {
        if stop_flag.load(Ordering::Relaxed) { return; }
        let path = &files[idx];
        let expected_size = bytes[idx];
        let expected_records = records[idx];
        let offset = offsets[idx];

        let shard_start = std::time::Instant::now();
        if let Some(cb) = log {
            cb(&format!(
                "[shard {}/{}] writing {} @ offset {}",
                idx + 1, total_files,
                path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                offset,
            ));
        }

        let result: Result<(), String> = (|| {
            let src = File::open(path)
                .map_err(|e| format!("open {}: {}", path.display(), e))?;
            let actual = src.metadata()
                .map_err(|e| format!("stat {}: {}", path.display(), e))?
                .len();
            if actual != expected_size {
                return Err(format!(
                    "size drift for {}: probe saw {}, now {}",
                    path.display(), expected_size, actual,
                ));
            }
            // Stream the shard in fixed-size chunks. Each chunk:
            //   1. Copy WRITE_CHUNK_BYTES into the output page cache —
            //      via copy_file_range when source and dest share a
            //      filesystem, or via read-into-scratch + pwrite when
            //      they don't. Same chunking either way; same EBS
            //      write footprint.
            //   2. sync_file_range(SYNC_FILE_RANGE_WRITE) tells the
            //      kernel to start writeback on that range *now*,
            //      asynchronously — without waiting on
            //      `dirty_background_bytes` or the periodic writeback
            //      timer. This is the lever that keeps the EBS write
            //      queue continuously fed.
            // No fsync per chunk: we rely on atomic-rename at end-of-
            // job to publish, and on the kernel to drain dirty pages
            // on its own. fsync would serialize the writer to disk
            // latency and undo the streaming.
            let mut copied: u64 = 0;
            while copied < expected_size {
                let chunk = WRITE_CHUNK_BYTES.min(expected_size - copied);
                let chunk_offset = offset + copied;
                copy_chunk(
                    copy_mode, &src, &out_file,
                    copied, chunk_offset, chunk, &mut scratch,
                ).map_err(|e| format!(
                    "{} {} → output@{} ({} bytes): {}",
                    copy_mode.label(), path.display(), chunk_offset, chunk, e,
                ))?;
                kick_writeback(&out_file, chunk_offset, chunk);
                copied += chunk;
            }
            // Both ranges are read-once / write-once: hint the kernel
            // it can drop them once they're clean. DONTNEED on dirty
            // pages is a soft hint that becomes effective once
            // writeback completes, so the order here is fine even
            // though sync_file_range(WRITE) above didn't wait.
            advise_dontneed(&out_file, offset, expected_size);
            advise_dontneed(&src, 0, expected_size);

            progress_total.fetch_add(1, Ordering::Relaxed);
            records_total_atomic.fetch_add(expected_records, Ordering::Relaxed);
            if let Some(cb) = log {
                let elapsed = shard_start.elapsed();
                let secs = elapsed.as_secs_f64().max(0.001);
                let mb = expected_size as f64 / (1024.0 * 1024.0);
                cb(&format!(
                    "[shard {}/{}] done {} ({:.1} MiB in {:.2}s, {:.1} MiB/s)",
                    idx + 1, total_files,
                    path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                    mb, secs, mb / secs,
                ));
            }
            Ok(())
        })();

        if let Err(e) = result {
            let mut slot = first_err.lock().unwrap();
            if slot.is_none() { *slot = Some(e); }
            stop_flag.store(true, Ordering::Relaxed);
            return;
        }

        // Publish writer progress for prefetcher backpressure.
        writer_advance.store(idx + 1, Ordering::Relaxed);
    }
}

/// List all files under `dir` whose extension matches the format
/// (preferred or any extension that resolves to it via `from_extension`).
fn collect_xvec_files(dir: &Path, format: VecFormat) -> Result<Vec<PathBuf>, String> {
    let preferred = format.preferred_extension().to_ascii_lowercase();
    let entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("read dir {}: {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file() && match p.extension().and_then(|s| s.to_str()) {
                Some(e) => {
                    let lower = e.to_ascii_lowercase();
                    lower == preferred || VecFormat::from_extension(&lower) == Some(format)
                }
                None => false,
            }
        })
        .collect();
    Ok(entries)
}

/// Per-chunk copy: dispatch to the chosen strategy.
///
/// `src_offset` is the byte offset into the source file (where this
/// chunk starts). `dst_offset` is the absolute output offset.
/// `scratch` is only consulted when `mode == ReadPwriteFallback`.
fn copy_chunk(
    mode: CopyMode,
    src: &File,
    dst: &File,
    src_offset: u64,
    dst_offset: u64,
    len: u64,
    scratch: &mut [u8],
) -> std::io::Result<()> {
    match mode {
        CopyMode::KernelCopyFileRange => {
            copy_file_range_at(src, src_offset, dst, dst_offset, len)
        }
        CopyMode::ReadPwriteFallback => {
            read_pwrite_chunk(src, src_offset, dst, dst_offset, len, scratch)
        }
    }
}

/// `copy_file_range` variant that takes a source offset (the
/// `copy_file_range_full` helper in parquet_vector_compiler always
/// starts at 0). Loops until `len` bytes are copied or EOF.
fn copy_file_range_at(
    src: &File,
    src_off: u64,
    dst: &File,
    dst_off: u64,
    len: u64,
) -> std::io::Result<()> {
    let mut so: i64 = src_off as i64;
    let mut dout: i64 = dst_off as i64;
    let mut remaining: usize = len as usize;
    while remaining > 0 {
        let n = unsafe {
            libc::copy_file_range(
                src.as_raw_fd(),
                &mut so as *mut i64,
                dst.as_raw_fd(),
                &mut dout as *mut i64,
                remaining,
                0,
            )
        };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "copy_file_range returned 0 before completing copy",
            ));
        }
        remaining = remaining.saturating_sub(n as usize);
    }
    Ok(())
}

/// Cross-filesystem fallback: read into a user-space scratch buffer,
/// then pwrite to the destination. Same chunking and same
/// `sync_file_range`-driven write cadence as the kernel-copy path —
/// only difference is one extra memcpy per chunk (the round trip
/// through user space). EBS still sees one streaming sequential
/// writer, just with slightly more CPU on the host.
fn read_pwrite_chunk(
    src: &File,
    src_off: u64,
    dst: &File,
    dst_off: u64,
    len: u64,
    scratch: &mut [u8],
) -> std::io::Result<()> {
    let len = len as usize;
    debug_assert!(scratch.len() >= len, "scratch too small for chunk");
    let buf = &mut scratch[..len];
    src.read_exact_at(buf, src_off)?;
    dst.write_all_at(buf, dst_off)?;
    Ok(())
}

/// Trigger asynchronous writeback for a range of `file` without
/// waiting. Equivalent to `posix_fadvise(WILLNEED)` for writes — the
/// kernel marks the range for immediate writeback rather than waiting
/// on its periodic timer or the `dirty_background_bytes` threshold.
///
/// Errors are non-fatal: the syscall is advisory, and a failure just
/// means writeback follows the kernel's default schedule rather than
/// our explicit kick.
fn kick_writeback(file: &File, offset: u64, len: u64) {
    if len == 0 { return; }
    unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            offset as i64,
            len as i64,
            libc::SYNC_FILE_RANGE_WRITE,
        );
    }
}

/// Read the four-byte little-endian dim header from the start of an
/// xvec file. Cheap enough that the probe phase calls it once per file
/// to validate cross-file dimension consistency.
fn read_first_dim(path: &Path) -> Result<i32, String> {
    use std::io::Read;
    let mut f = File::open(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)
        .map_err(|e| format!("read dim header {}: {}", path.display(), e))?;
    Ok(i32::from_le_bytes(buf))
}

/// Natural-order comparison on file basenames so unpadded shard
/// numbering (`1, 2, …, 10`) sorts numerically rather than lexically
/// (`1, 10, 2`). Mirrors the parquet path.
fn natural_path_cmp(a: &Path, b: &Path) -> std::cmp::Ordering {
    let an = a.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let bn = b.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    natural_str_cmp(&an, &bn)
}

fn natural_str_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let mut i = 0;
    let mut j = 0;
    while i < ab.len() && j < bb.len() {
        let ac = ab[i];
        let bc = bb[j];
        if ac.is_ascii_digit() && bc.is_ascii_digit() {
            let mut ai = i;
            while ai < ab.len() && ab[ai].is_ascii_digit() { ai += 1; }
            let mut bj = j;
            while bj < bb.len() && bb[bj].is_ascii_digit() { bj += 1; }
            let an: u128 = std::str::from_utf8(&ab[i..ai])
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let bn: u128 = std::str::from_utf8(&bb[j..bj])
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            match an.cmp(&bn) {
                Ordering::Equal => {
                    let alen = ai - i;
                    let blen = bj - j;
                    if alen != blen {
                        return alen.cmp(&blen);
                    }
                    i = ai;
                    j = bj;
                }
                other => return other,
            }
        } else {
            match ac.cmp(&bc) {
                Ordering::Equal => { i += 1; j += 1; }
                other => return other,
            }
        }
    }
    ab.len().cmp(&bb.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fvecs(path: &Path, dim: i32, vectors: &[&[f32]]) {
        let mut f = File::create(path).unwrap();
        for v in vectors {
            assert_eq!(v.len() as i32, dim);
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &x in *v { f.write_all(&x.to_le_bytes()).unwrap(); }
        }
    }

    #[test]
    fn probe_and_concat_uniform_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_fvecs(&dir.join("1.fvecs"), 3, &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        write_fvecs(&dir.join("2.fvecs"), 3, &[&[7.0, 8.0, 9.0]]);
        write_fvecs(&dir.join("10.fvecs"), 3, &[&[10.0, 11.0, 12.0]]);

        let probe = probe_xvec_directory(dir, VecFormat::Fvec).unwrap();
        // Natural sort: 1, 2, 10 (not 1, 10, 2).
        let names: Vec<_> = probe.files.iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["1.fvecs", "2.fvecs", "10.fvecs"]);
        assert_eq!(probe.dimension, 3);
        assert_eq!(probe.total_records(), 4);

        let out = tmp.path().join("all.fvecs");
        let n = extract_xvec_dir_to_xvec_threaded(
            dir, &out, VecFormat::Fvec, None, None, 4,
        ).unwrap();
        assert_eq!(n, 4);

        // Read back and verify ordering matches natural sort: 1.fvecs's two
        // records first, then 2.fvecs, then 10.fvecs.
        let bytes = fs::read(&out).unwrap();
        let stride = 4 + 3 * 4;
        assert_eq!(bytes.len(), stride * 4);
        for (record_idx, expected) in [
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ].iter().enumerate() {
            let off = record_idx * stride;
            let dim = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            assert_eq!(dim, 3);
            for (k, e) in expected.iter().enumerate() {
                let v = f32::from_le_bytes(
                    bytes[off + 4 + k * 4..off + 4 + (k + 1) * 4].try_into().unwrap()
                );
                assert_eq!(v, *e);
            }
        }
    }

    #[test]
    fn probe_rejects_mixed_dimensions() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_fvecs(&dir.join("1.fvecs"), 3, &[&[1.0, 2.0, 3.0]]);
        write_fvecs(&dir.join("2.fvecs"), 4, &[&[1.0, 2.0, 3.0, 4.0]]);
        let err = probe_xvec_directory(dir, VecFormat::Fvec).unwrap_err();
        assert!(err.contains("different dim") || err.contains("not a multiple"),
            "unexpected error: {}", err);
    }
}
