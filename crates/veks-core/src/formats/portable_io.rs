// Copyright 2024-present nosqlbench / vectordata-rs contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-platform `pread` / `pwrite` wrappers.
//!
//! Linux-tuned compilers in this crate (`xvec_dir_compiler`,
//! `parquet_vector_compiler`) lean on positional I/O so a single
//! shared `File` handle can serve concurrent readers/writers without
//! needing `&mut self` for each operation. The Unix idiom is
//! `std::os::unix::fs::FileExt::{read_exact_at, write_all_at}`; the
//! Windows equivalents live on `std::os::windows::fs::FileExt` as
//! `seek_read` / `seek_write`. The two surfaces aren't quite
//! interchangeable (Windows's variants return short reads/writes
//! that we have to loop over to match Unix's `_exact`/`_all`
//! semantics), so this module hides the difference behind a uniform
//! interface used by every positional-I/O call site in the crate.
//!
//! On targets that are neither `unix` nor `windows`, both functions
//! return `ErrorKind::Unsupported` so callers can choose to fail
//! gracefully or report unsupported.

use std::fs::File;
use std::io;

/// Read exactly `buf.len()` bytes from `file` starting at `offset`.
/// Mirrors `FileExt::read_exact_at` semantics on every platform we
/// support.
///
/// **`#[inline(always)]`**: this is a per-vector hot-path call site
/// in `gen_extract`, `compute_dedup`, and `analyze_find_zeros`. The
/// caller crates have no LTO, so plain `#[inline]` is not always
/// honored across crate boundaries. Forcing inlining lets the
/// `#[cfg(unix)]` body fold to a direct `read_exact_at` call with
/// the per-platform dead-code arms eliminated.
#[inline(always)]
pub fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut read = 0;
        while read < buf.len() {
            let n = file.seek_read(&mut buf[read..], offset + read as u64)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "seek_read returned 0 before filling buffer",
                ));
            }
            read += n;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pread_exact: unsupported target",
        ))
    }
}

/// Read up to `buf.len()` bytes from `file` starting at `offset`.
/// Returns the number of bytes read (may be short or 0 at EOF).
/// Mirrors `FileExt::read_at` semantics on every platform we
/// support — does NOT loop, lets the caller decide what to do
/// with a short read.
#[inline(always)]
pub fn pread(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(buf, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pread: unsupported target",
        ))
    }
}

/// Write all of `buf` to `file` starting at `offset`. Mirrors
/// `FileExt::write_all_at` semantics on every platform we support.
#[inline(always)]
pub fn pwrite_all(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut written = 0;
        while written < buf.len() {
            let n = file.seek_write(&buf[written..], offset + written as u64)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "seek_write returned 0",
                ));
            }
            written += n;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pwrite_all: unsupported target",
        ))
    }
}

/// A facet opened for positional reads, whether it is one file or a
/// series of shards.
///
/// The hot scan paths read with `pread` into heap buffers rather than
/// mapping the file, so they hold a `File` rather than a reader — and
/// a `File` is one file. This presents the shards of a series the same
/// way [`Storage::Series`](vectordata) presents them to the mapped
/// readers: as one byte space, with offsets that mean what the
/// unsharded file's offsets meant.
///
/// A read never crosses a shard here because callers read whole
/// records and records never straddle (SH-13) — but one that did would
/// be stitched rather than truncated, so the abstraction does not
/// depend on the caller knowing that.
pub struct SpanFile {
    parts: Vec<File>,
    /// Byte offset at which each part begins. Ascending, first zero.
    starts: Vec<u64>,
    total: u64,
}

impl SpanFile {
    /// Open `path`, or its shards when `path` itself is absent.
    ///
    /// The same resolution rule the mapped readers use: a facet
    /// written as a series has nothing at the unsharded name, and a
    /// caller handed that name means the facet.
    pub fn open(path: &std::path::Path) -> io::Result<Self> {
        let files: Vec<std::path::PathBuf> = if path.exists() {
            vec![path.to_path_buf()]
        } else {
            let shards = vectordata::dataset::discover_shards(path);
            if shards.is_empty() {
                // Let `File::open` produce the usual not-found error,
                // naming the path the caller asked for.
                vec![path.to_path_buf()]
            } else {
                shards
            }
        };
        let mut parts = Vec::with_capacity(files.len());
        let mut starts = Vec::with_capacity(files.len());
        let mut total = 0u64;
        for f in files {
            let file = File::open(&f)?;
            let len = file.metadata()?.len();
            starts.push(total);
            total += len;
            parts.push(file);
        }
        Ok(Self { parts, starts, total })
    }

    /// Total bytes across every part.
    pub fn len(&self) -> u64 {
        self.total
    }

    /// Whether this facet holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// How many files back this facet. `1` for an unsharded one.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Fill `buf` from `offset` in the joined space.
    #[inline]
    pub fn pread_exact(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        // The overwhelmingly common case, and the one the scan loops
        // take on every record: one file, or a read wholly inside one
        // shard. Kept to a single positional read.
        if self.parts.len() == 1 {
            return pread_exact(&self.parts[0], buf, offset);
        }
        let end = offset + buf.len() as u64;
        if end > self.total {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read past end of series: offset={offset} len={} size={}", buf.len(), self.total),
            ));
        }
        let mut want = offset;
        let mut at = 0usize;
        while want < end {
            let i = match self.starts.binary_search(&want) {
                Ok(i) => i,
                Err(i) => i - 1,
            };
            let part_start = self.starts[i];
            let part_len = self.parts[i].metadata()?.len();
            let within = want - part_start;
            let take = ((part_len - within).min(end - want)) as usize;
            pread_exact(&self.parts[i], &mut buf[at..at + take], within)?;
            at += take;
            want += take as u64;
        }
        Ok(())
    }
}
