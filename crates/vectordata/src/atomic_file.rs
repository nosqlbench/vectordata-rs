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

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes temporary files written by one process at once.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Replace the file at `path` with `data`: the path itself is replaced,
/// so a symlink there becomes a regular file. For data files a program
/// owns (published objects, restored copies). Creates the parent
/// directory when it is missing.
pub(crate) fn replace(path: &Path, data: &[u8]) -> io::Result<()> {
    write_then_rename(path, data, None)
}

/// Replace a configuration file the user may keep by hand: a symlink is
/// followed and its target replaced, as `std::fs::write` would write
/// through it, and the file keeps the permissions it had. With
/// `private`, the file is made readable by its owner only (`0600` on
/// unix) before any byte is written, so a secret is never visible under
/// looser permissions.
pub(crate) fn replace_config(path: &Path, data: &[u8], private: bool) -> io::Result<()> {
    let target = link_target(path)?;
    let mode = if private { Some(0o600) } else { existing_mode(&target) };
    write_then_rename(&target, data, mode)
}

/// The file `path` names: `path` itself, or the end of the symlink chain
/// it starts, each link read relative to its own directory.
fn link_target(path: &Path) -> io::Result<PathBuf> {
    let mut current = path.to_path_buf();
    for _ in 0..40 {
        match std::fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let next = std::fs::read_link(&current)?;
                current = match current.parent() {
                    Some(dir) if next.is_relative() => dir.join(next),
                    _ => next,
                };
            }
            _ => return Ok(current),
        }
    }
    Err(io::Error::other(format!("too many levels of symbolic links at {}", path.display())))
}

/// The permission bits of the file at `path`, when there is one.
fn existing_mode(path: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).ok().map(|m| m.permissions().mode() & 0o7777)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

fn write_then_rename(path: &Path, data: &[u8], mode: Option<u32>) -> io::Result<()> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(dir)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.tmp.{}.{}", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    let written = create_with_mode(&tmp, mode).and_then(|mut file| {
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

/// A new file at `path`, with `mode` from its creation on unix.
fn create_with_mode(path: &Path, mode: Option<u32>) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    if let Some(mode) = mode {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(mode);
    }
    #[cfg(not(unix))]
    let _ = mode;
    let file = options.open(path)?;
    // The creation mode is masked by the umask; set it exactly.
    #[cfg(unix)]
    if let Some(mode) = mode {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(mode))?;
    }
    Ok(file)
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
        let mut reader = File::open(&path).unwrap();

        replace_config(&path, b"new: content\n", false).unwrap();

        let mut seen = String::new();
        reader.read_to_string(&mut seen).unwrap();
        assert_eq!(seen, "old: content\n", "the open file was rewritten in place");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new: content\n");
        assert!(temp_files(tmp.path()).is_empty());
    }

    /// **A symlinked configuration file stays a symlink**, and its target
    /// gets the new content, as a write through the link would.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_config_file_keeps_its_link() {
        let tmp = tempfile::tempdir().unwrap();
        let dotfiles = tmp.path().join("dotfiles");
        std::fs::create_dir_all(&dotfiles).unwrap();
        std::fs::write(dotfiles.join("settings.yaml"), "a: 1\n").unwrap();
        let home = tmp.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        std::os::unix::fs::symlink("../dotfiles/settings.yaml", home.join("settings.yaml")).unwrap();

        replace_config(&home.join("settings.yaml"), b"a: 2\n", false).unwrap();

        assert!(std::fs::symlink_metadata(home.join("settings.yaml")).unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read_to_string(dotfiles.join("settings.yaml")).unwrap(), "a: 2\n");
        assert!(temp_files(&dotfiles).is_empty() && temp_files(&home).is_empty());
    }

    /// **A configuration file keeps its permissions**, and a private one
    /// is owner-only from its creation.
    #[cfg(unix)]
    #[test]
    fn permissions_are_kept_and_private_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let kept = tmp.path().join("settings.yaml");
        std::fs::write(&kept, "a: 1\n").unwrap();
        std::fs::set_permissions(&kept, std::fs::Permissions::from_mode(0o640)).unwrap();
        replace_config(&kept, b"a: 2\n", false).unwrap();
        assert_eq!(mode(&kept), 0o640);

        let secret = tmp.path().join("credentials.toml");
        replace_config(&secret, b"token = \"x\"\n", true).unwrap();
        assert_eq!(mode(&secret), 0o600);
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
