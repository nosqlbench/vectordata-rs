// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Atomic file writing via temp-then-rename.
//!
//! [`AtomicWriter`] writes to a temporary file in the same directory as the
//! target, then atomically renames it to the final path on success. This
//! prevents partially-written files from being visible at the final path
//! if the process is interrupted.
//!
//! The rename is an inode swap (same filesystem), not a data copy.
//!
//! Usage:
//! ```
//! use std::io::Write;
//! use veks_pipeline::pipeline::atomic_write::AtomicWriter;
//!
//! let dir = tempfile::tempdir().unwrap();
//! let final_path = dir.path().join("output.fvec");
//! let mut w = AtomicWriter::new(&final_path).unwrap();
//! w.write_all(b"data").unwrap();
//! w.finish().unwrap(); // renames temp → final
//! assert!(final_path.exists());
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Atomic file writer: writes to a temp file, renames on `finish()`.
pub struct AtomicWriter {
    writer: BufWriter<File>,
    temp_path: PathBuf,
    final_path: PathBuf,
    finished: bool,
}

impl AtomicWriter {
    /// Create a new atomic writer targeting `final_path`.
    ///
    /// The temp file is created in the same directory as `final_path`
    /// (ensuring same-filesystem rename) with a `.tmp` suffix.
    pub fn new(final_path: &Path) -> io::Result<Self> {
        Self::with_capacity(1 << 20, final_path) // 1 MiB buffer
    }

    /// Create with a specific buffer capacity.
    pub fn with_capacity(capacity: usize, final_path: &Path) -> io::Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = final_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let temp_path = temp_path_for(final_path);
        let file = File::create(&temp_path)?;
        Ok(AtomicWriter {
            writer: BufWriter::with_capacity(capacity, file),
            temp_path,
            final_path: final_path.to_path_buf(),
            finished: false,
        })
    }

    /// Flush and atomically rename the temp file to the final path.
    ///
    /// This is an inode-level rename (same filesystem) — no data copy.
    /// Must be called explicitly; if not called, the temp file is removed
    /// on drop.
    pub fn finish(mut self) -> io::Result<()> {
        self.writer.flush()?;
        // Safety: if the final path is a symlink, remove it before renaming.
        // POSIX rename() replaces the directory entry (not the symlink target),
        // but we add an explicit guard to prevent any risk of clobbering
        // source data through a stale identity symlink.
        if self.final_path.is_symlink() {
            std::fs::remove_file(&self.final_path)?;
        }
        std::fs::rename(&self.temp_path, &self.final_path)?;
        self.finished = true;
        Ok(())
    }

    /// Path to the temp file (for diagnostics only — callers should not
    /// depend on this).
    pub fn temp_path(&self) -> &Path {
        &self.temp_path
    }

    /// Path to the final target file.
    pub fn final_path(&self) -> &Path {
        &self.final_path
    }
}

impl Write for AtomicWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.writer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl Drop for AtomicWriter {
    fn drop(&mut self) {
        if !self.finished {
            // Clean up temp file on failure/panic
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

/// Compute the temp path for a given final path.
///
/// Uses the same directory (same filesystem for atomic rename) with a
/// `.tmp` suffix appended.
fn temp_path_for(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".tmp");
    PathBuf::from(p)
}

/// Reject writes to symlinks (SRD §12 safety invariant).
///
/// Symlinks in the dataset workspace are read aliases — they point to
/// source data that must never be overwritten. Any code path that opens
/// a file for writing must call this first. If `path` is a symlink, the
/// symlink is removed so a subsequent `File::create` produces a new
/// regular file instead of writing through the link to the source.
///
/// Returns `Err` only if the symlink exists but cannot be removed.
pub fn guard_against_symlink_write(path: &Path) -> io::Result<()> {
    if path.is_symlink() {
        eprintln!(
            "  safety: removing symlink {} before write (symlinks are read-only aliases)",
            path.display(),
        );
        std::fs::remove_file(path)?;
    }
    Ok(())
}

/// Create a file for writing, with symlink safety.
///
/// If `path` is a symlink, removes it first so the write creates a new
/// regular file instead of clobbering the symlink target. This enforces
/// the SRD invariant that symlinks are read-only aliases to source data.
pub fn safe_create_file(path: &Path) -> io::Result<File> {
    guard_against_symlink_write(path)?;
    File::create(path)
}

/// Write bytes to a file, with symlink safety.
///
/// If `path` is a symlink, removes it first. Equivalent to
/// `guard_against_symlink_write` + `std::fs::write`.
pub fn safe_write(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    guard_against_symlink_write(path)?;
    std::fs::write(path, contents)
}

/// Copy a file, with symlink safety on the destination.
///
/// If `to` is a symlink, removes it first so the copy creates a new
/// regular file instead of overwriting the symlink target.
pub fn safe_copy(from: &Path, to: &Path) -> io::Result<u64> {
    guard_against_symlink_write(to)?;
    std::fs::copy(from, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_success() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("output.fvec");

        let mut w = AtomicWriter::new(&final_path).unwrap();
        w.write_all(b"hello world").unwrap();
        w.finish().unwrap();

        assert!(final_path.exists());
        assert_eq!(std::fs::read(&final_path).unwrap(), b"hello world");
        // Temp file should be gone
        assert!(!temp_path_for(&final_path).exists());
    }

    #[test]
    fn atomic_write_drop_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("output.fvec");

        {
            let mut w = AtomicWriter::new(&final_path).unwrap();
            w.write_all(b"partial data").unwrap();
            // drop without finish()
        }

        // Final path should NOT exist
        assert!(!final_path.exists());
        // Temp file should be cleaned up
        assert!(!temp_path_for(&final_path).exists());
    }

    #[test]
    fn atomic_write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("sub/dir/output.fvec");

        let mut w = AtomicWriter::new(&final_path).unwrap();
        w.write_all(b"nested").unwrap();
        w.finish().unwrap();

        assert!(final_path.exists());
    }
}
