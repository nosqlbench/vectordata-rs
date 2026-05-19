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
