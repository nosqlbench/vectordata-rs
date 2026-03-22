// Copyright (c) DataStax, Inc.
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
//! ```ignore
//! let mut w = AtomicWriter::new(&final_path)?;
//! w.write_all(b"data")?;
//! w.finish()?; // renames temp → final
//! // If finish() is not called (e.g., error/panic), the temp file is
//! // cleaned up on drop.
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
