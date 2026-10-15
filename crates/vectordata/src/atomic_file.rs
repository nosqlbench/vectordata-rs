// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Internal. Replace a file's contents so a reader sees the old file or
//! the new one, never an empty or half-written one.
//!
//! `std::fs::write` truncates the file and then writes it, so another
//! process reading it in between (a second `vectordata` reading
//! `catalogs.yaml` while the explorer edits it) finds it empty. These
//! functions write a temporary file beside the target and rename it over
//! the target, which is atomic on one filesystem.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes temporary files written by one process at once.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Replace the file at `path` with `data`: the path itself is replaced,
/// so a symlink there becomes a regular file. For data files a program
/// owns (published objects, restored copies). Creates the parent
/// directory when it is missing.
pub(crate) fn replace(path: &Path, data: &[u8]) -> io::Result<()> {
    write_then_rename(path, data)
}

fn write_then_rename(path: &Path, data: &[u8]) -> io::Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(dir)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.tmp.{}.{}", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    let written = OpenOptions::new().write(true).create_new(true).open(&tmp).and_then(|mut file| {
        file.write_all(data)?;
        file.flush()
    });
    match written.and_then(|()| std::fs::rename(&tmp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_files(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp."))
            .collect()
    }

    /// **A replacement never changes the file a reader already has open**:
    /// the old file stays whole for it, and the path names the new one.
    /// `std::fs::write` truncates the open file in place, which is how a
    /// reader came to find an empty `catalogs.yaml`.
    #[cfg(unix)]
    #[test]
    fn a_reader_sees_the_old_file_or_the_new_never_a_truncated_one() {
        use std::io::Read;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("catalogs.yaml");
        std::fs::write(&path, "old: content\n").unwrap();
        let mut reader = std::fs::File::open(&path).unwrap();

        replace(&path, b"new: content\n").unwrap();

        let mut seen = String::new();
        reader.read_to_string(&mut seen).unwrap();
        assert_eq!(seen, "old: content\n", "the open file was rewritten in place");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new: content\n");
        assert!(temp_files(tmp.path()).is_empty());
    }

    /// **A replacement that fails leaves the old file and no temporary
    /// file**: here the target is a directory, so the rename fails.
    #[test]
    fn a_failed_replacement_leaves_no_temporary_file() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("taken");
        std::fs::create_dir_all(target.join("inside")).unwrap();
        assert!(replace(&target, b"x").is_err());
        assert!(target.join("inside").is_dir());
        assert!(temp_files(tmp.path()).is_empty());
    }
}
