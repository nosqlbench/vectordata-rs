// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use super::VecSink;

/// Write buffer size (4 MiB).
const WRITE_BUF_SIZE: usize = 4 << 20;

/// Interval (in bytes written) between page cache release cycles.
///
/// Every interval, we start async writeback on the current dirty range
/// and release the previous range's now-clean pages via `FADV_DONTNEED`.
const WRITEBACK_INTERVAL: u64 = 64 << 20; // 64 MiB

/// Writes any xvec format (fvec, ivec, bvec, dvec, mvec, svec).
///
/// Wire format per record: `[dimension: i32 LE][elements: dim * element_size bytes]`
///
/// Uses a two-phase page cache management pipeline:
/// 1. `sync_file_range(SYNC_FILE_RANGE_WRITE)` to start async writeback
///    on newly written pages
/// 2. On the next interval, wait for the previous writeback to complete,
///    then `FADV_DONTNEED` to release those now-clean pages
///
/// This prevents dirty page accumulation that triggers the kernel's
/// synchronous writeback throttle (`dirty_ratio`) on large files.
pub struct XvecWriter {
    writer: BufWriter<File>,
    dimension: u32,
    bytes_written: u64,
    /// End offset of the range where we last started async writeback.
    last_writeback: u64,
    /// End offset of the range we last released via `FADV_DONTNEED`.
    last_dontneed: u64,
}

impl XvecWriter {
    /// Open an xvec output file for writing
    pub fn open(path: &Path, dimension: u32) -> Result<Box<dyn VecSink>, String> {
        // Safety: remove any symlink before writing — symlinks are
        // read-only aliases to source data.
        if path.is_symlink() {
            std::fs::remove_file(path)
                .map_err(|e| format!("Failed to remove symlink {}: {}", path.display(), e))?;
            eprintln!("  safety: removed symlink {} before write", path.display());
        }
        let file = File::create(path)
            .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
        Ok(Box::new(XvecWriter {
            writer: BufWriter::with_capacity(WRITE_BUF_SIZE, file),
            dimension,
            bytes_written: 0,
            last_writeback: 0,
            last_dontneed: 0,
        }))
    }
}

impl VecSink for XvecWriter {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        self.writer
            .write_i32::<LittleEndian>(self.dimension as i32)
            .expect("failed to write dimension");
        self.writer.write_all(data).expect("failed to write data");
        self.bytes_written += 4 + data.len() as u64;

        if self.bytes_written - self.last_writeback >= WRITEBACK_INTERVAL {
            self.writer.flush().ok();
            release_written_pages(
                &self.writer,
                &mut self.last_writeback,
                &mut self.last_dontneed,
                self.bytes_written,
            );
        }
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        self.writer.flush().map_err(|e| e.to_string())?;
        // Final flush: sync and release any remaining pages
        release_written_pages(
            &self.writer,
            &mut self.last_writeback,
            &mut self.last_dontneed,
            self.bytes_written,
        );
        // Wait for final writeback and release
        if self.last_dontneed < self.last_writeback {
            wait_and_release(
                &self.writer,
                self.last_dontneed,
                self.last_writeback,
            );
        }
        Ok(())
    }
}

/// Two-phase page cache release pipeline.
///
/// Phase 1: If there's a range with pending async writeback from last
/// time, wait for it to finish and release those now-clean pages.
///
/// Phase 2: Start async writeback on newly written dirty pages so
/// they'll be clean by the time we come back next interval.
#[cfg(target_os = "linux")]
fn release_written_pages(
    writer: &BufWriter<File>,
    last_writeback: &mut u64,
    last_dontneed: &mut u64,
    bytes_written: u64,
) {
    use std::os::unix::io::AsRawFd;
    let fd = writer.get_ref().as_raw_fd();

    // Phase 1: release pages from the previous writeback range
    if *last_dontneed < *last_writeback {
        wait_and_release(writer, *last_dontneed, *last_writeback);
        *last_dontneed = *last_writeback;
    }

    // Phase 2: start async writeback on the current dirty range
    if bytes_written > *last_writeback {
        unsafe {
            libc::sync_file_range(
                fd,
                *last_writeback as i64,
                (bytes_written - *last_writeback) as i64,
                libc::SYNC_FILE_RANGE_WRITE,
            );
        }
        *last_writeback = bytes_written;
    }
}

#[cfg(not(target_os = "linux"))]
fn release_written_pages(
    _writer: &BufWriter<File>,
    _last_writeback: &mut u64,
    _last_dontneed: &mut u64,
    _bytes_written: u64,
) {}

/// Wait for async writeback to complete on a range, then release the
/// now-clean pages from the page cache.
#[cfg(target_os = "linux")]
fn wait_and_release(writer: &BufWriter<File>, start: u64, end: u64) {
    use std::os::unix::io::AsRawFd;
    let fd = writer.get_ref().as_raw_fd();
    let len = end - start;
    unsafe {
        // Wait for any ongoing writeback on this range to finish
        libc::sync_file_range(
            fd,
            start as i64,
            len as i64,
            libc::SYNC_FILE_RANGE_WAIT_BEFORE,
        );
        // Pages are now clean — release them from the page cache
        libc::posix_fadvise(
            fd,
            start as i64,
            len as i64,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn wait_and_release(_writer: &BufWriter<File>, _start: u64, _end: u64) {}
