// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Sequential xvec writer with buffered I/O.
//!
//! The written pages are written back and released as the file grows
//! (Linux: every 64 MiB, `sync_file_range` then `FADV_DONTNEED`), so a
//! write of a file larger than memory keeps a bounded page cache and the
//! kernel never stalls the writer flushing gigabytes at once. A symlink
//! at the target is an error: what it links to is not the writer's to
//! overwrite, and the link is not the writer's to remove. The first I/O
//! error stops the writer and is what
//! [`finish`](VecSink::finish) returns.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::traits::VecSink;

/// Size of the `BufWriter`'s internal buffer.
///
/// 16 MiB is a balance between two concerns:
/// - Larger buffers reduce the number of `write` syscalls.
/// - Each flush is a blocking `write` of the whole buffer; making it too
///   large (e.g. 64 MiB) causes long stalls under writeback throttling
///   (Linux's `dirty_ratio`) on cold-disk targets like EBS, because the
///   producer side serializes on the flush.
const WRITE_BUF_SIZE: usize = 16 << 20; // 16 MiB

/// Bytes written between writeback cycles.
const WRITEBACK_INTERVAL: u64 = 64 << 20; // 64 MiB

/// [`guard_against_symlink_write`](crate::atomic::guard_against_symlink_write)
/// as the writers' `String` error.
pub(crate) fn refuse_symlink(path: &Path) -> Result<(), String> {
    crate::atomic::guard_against_symlink_write(path).map_err(|e| e.to_string())
}

/// Open an xvec file for sequential writing.
///
/// The `dimension` is written as the header of every record.
/// Element size is inferred from the output format (determined by
/// the file extension at the call site).
///
/// # Example
///
/// Two `i8` records; the writer prefixes each with the dimension and
/// copies the bytes as given, so the file is `2 × (4 + 3)` bytes:
///
/// ```
/// use veks_io::xvec;
/// # let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("codes.i8vecs");
/// let mut writer = xvec::writer::open(&path, 3).unwrap();
/// for row in [[-1i8, 2, -3], [i8::MIN, 0, i8::MAX]] {
///     let bytes: Vec<u8> = row.iter().map(|v| v.to_le_bytes()[0]).collect();
///     writer.write_record(0, &bytes);
/// }
/// writer.finish().unwrap();
/// assert_eq!(std::fs::metadata(&path).unwrap().len(), 2 * (4 + 3));
///
/// let mut reader = veks_io::open(&path).unwrap();
/// reader.next_record().unwrap();
/// let second = reader.next_record().unwrap();
/// assert_eq!(second.iter().map(|&b| i8::from_le_bytes([b])).collect::<Vec<_>>(), [i8::MIN, 0, i8::MAX]);
/// ```
pub fn open(path: &Path, dimension: u32) -> Result<Box<dyn VecSink>, String> {
    refuse_symlink(path)?;
    let file = File::create(path)
        .map_err(|e| format!("create {}: {}", path.display(), e))?;
    let writer = BufWriter::with_capacity(WRITE_BUF_SIZE, file);

    Ok(Box::new(XvecWriter {
        writer,
        // `dim_prefix` is the 4-byte little-endian dim header emitted in
        // front of every record. Pre-computed once at open time so the hot
        // loop is a plain copy of a small stack slice.
        dim_prefix: (dimension as i32).to_le_bytes(),
        // Reusable per-batch scratch buffer for the coalesced batch-write
        // path. Grows on demand and is reused across batches so the steady
        // state has zero allocations in the hot loop.
        scratch: Vec::new(),
        error: None,
        writeback: Writeback::default(),
    }))
}

struct XvecWriter {
    writer: BufWriter<File>,
    dim_prefix: [u8; 4],
    scratch: Vec<u8>,
    /// The first write that failed: kept, and returned by `finish`, so a
    /// full disk is an error rather than a short file.
    error: Option<std::io::Error>,
    /// Page-cache release as the file grows.
    writeback: Writeback,
}

impl XvecWriter {
    fn keep(&mut self, result: std::io::Result<()>) {
        if let Err(e) = result
            && self.error.is_none()
        {
            self.error = Some(e);
        }
    }

    /// Count `n` bytes written, and once an interval has passed, flush
    /// and write back what is on disk so far.
    fn wrote(&mut self, n: usize) {
        self.writeback.written += n as u64;
        if self.writeback.written - self.writeback.started >= WRITEBACK_INTERVAL {
            let flushed = self.writer.flush();
            self.keep(flushed);
            self.writeback.cycle(self.writer.get_ref());
        }
    }
}

/// Page-cache release for a file written front to back: start
/// writeback on the newly written range, and drop the range written
/// back the cycle before, which is clean by then.
#[derive(Debug, Default)]
struct Writeback {
    /// Bytes written so far.
    written: u64,
    /// Where the last writeback started up to.
    started: u64,
    /// Where pages have been released up to.
    released: u64,
}

impl Writeback {
    fn cycle(&mut self, file: &File) {
        if self.released < self.started {
            release(file, self.released, self.started);
            self.released = self.started;
        }
        if self.written > self.started {
            start_writeback(file, self.started, self.written);
            self.started = self.written;
        }
    }

    /// The last cycle: write back the rest, wait for it and release it.
    fn finish(&mut self, file: &File) {
        self.cycle(file);
        if self.released < self.started {
            release(file, self.released, self.started);
            self.released = self.started;
        }
    }
}

#[cfg(target_os = "linux")]
fn start_writeback(file: &File, start: u64, end: u64) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: a hint over a range of an open file; it reads no memory.
    unsafe {
        libc::sync_file_range(file.as_raw_fd(), start as i64, (end - start) as i64, libc::SYNC_FILE_RANGE_WRITE);
    }
}

#[cfg(target_os = "linux")]
fn release(file: &File, start: u64, end: u64) {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: as above. Waiting first makes the pages clean, so
    // DONTNEED drops them rather than leaving dirty ones behind.
    unsafe {
        libc::sync_file_range(fd, start as i64, (end - start) as i64, libc::SYNC_FILE_RANGE_WAIT_BEFORE);
        libc::posix_fadvise(fd, start as i64, (end - start) as i64, libc::POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn start_writeback(_file: &File, _start: u64, _end: u64) {}

#[cfg(not(target_os = "linux"))]
fn release(_file: &File, _start: u64, _end: u64) {}


impl VecSink for XvecWriter {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        if self.error.is_some() {
            return;
        }
        let header = self.writer.write_all(&self.dim_prefix);
        self.keep(header);
        let body = self.writer.write_all(data);
        self.keep(body);
        self.wrote(4 + data.len());
    }

    /// Write `count` fixed-dim records from a contiguous packed buffer of
    /// raw element bytes. The hot path on the parquet → xvec extractor.
    ///
    /// Implementation: build a single batch-sized interleaved buffer
    /// `[dim][rec0][dim][rec1]...` in `scratch`, then issue ONE
    /// `write_all` to the underlying `BufWriter`. When the scratch slice
    /// is larger than the BufWriter's internal buffer, std's `BufWriter`
    /// bypasses its own buffer and writes the slice directly to the
    /// inner `File` in a single `write()` syscall — turning what was
    /// previously `2 * count` per-record `write_all` calls into one
    /// kernel write per batch.
    ///
    /// Tradeoff: one extra memcpy per record (Arrow → scratch, then
    /// scratch → kernel), but each memcpy is a tight `copy_from_slice`
    /// at memory bandwidth (~5 GiB/s) — cheaper than the per-record
    /// virtual + bounds-check overhead it replaces.
    fn write_records_fixed_dim(
        &mut self,
        _ordinal_start: i64,
        packed: &[u8],
        count: usize,
        record_size: usize,
    ) {
        debug_assert_eq!(packed.len(), count * record_size);
        let stride = 4 + record_size;
        let total = count * stride;
        // resize fills with zeros; the loop overwrites every byte. Could
        // be replaced with `set_len` after `reserve` for an unsafe fast
        // path, but the zero-fill is well-vectorized by the compiler and
        // not the bottleneck (tested under perf).
        self.scratch.resize(total, 0);
        let mut dst = 0;
        let mut src = 0;
        for _ in 0..count {
            self.scratch[dst..dst + 4].copy_from_slice(&self.dim_prefix);
            self.scratch[dst + 4..dst + stride]
                .copy_from_slice(&packed[src..src + record_size]);
            dst += stride;
            src += record_size;
        }
        if self.error.is_none() {
            let result = self.writer.write_all(&self.scratch);
            self.keep(result);
            self.wrote(total);
        }
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        if let Some(e) = self.error.take() {
            return Err(format!("write: {e}"));
        }
        self.writer.flush().map_err(|e| format!("flush: {}", e))?;
        self.writeback.finish(self.writer.get_ref());
        Ok(())
    }
}

