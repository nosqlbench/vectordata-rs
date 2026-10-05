// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Workspace sweep for legacy singular-extension xvec files.
//!
//! The canonical xvec extensions in this codebase normalise to the
//! plural form (`fvecs`, `ivecs`, …; see [`crate::formats::VecFormat`]).
//! Datasets bootstrapped before that normalisation still carry the
//! singular-form symlinks (`base_vectors.fvec` &c). `merkle create`
//! walks them recursively and produces redundant `.mref` siblings,
//! and `veks check extraneous-files` correctly flags them since the
//! manifest only references the plural form.
//!
//! [`sweep`] walks a workspace and removes every singular-extension
//! file/symlink whose plural sibling exists at the same path,
//! together with its `.mref` companion. The sibling-presence guard
//! makes the sweep safe on any dataset: a `.fvec` that is the *only*
//! copy (no `.fvecs` sibling) is preserved.
//!
//! Used by:
//!
//! - `veks prepare bootstrap` (at end, regardless of which slots ran
//!   through the per-link sweep in `create_symlink`).
//! - `veks run` (at the very start, so existing datasets self-heal
//!   without a re-bootstrap).

use std::path::Path;

use crate::filters::is_excluded_dir;
use crate::formats::VecFormat;

/// Walk `workspace` and remove every legacy singular-extension file
/// or symlink (e.g. `*.fvec`, `*.ivec`) whose canonical plural
/// sibling exists at the same path. Also removes the `.mref`
/// companion of any swept file.
///
/// Returns the number of legacy data files removed (mref companions
/// are not counted separately).
pub fn sweep(workspace: &Path) -> usize {
    let mut removed = 0usize;
    sweep_recursive(workspace, &mut removed);
    removed
}

fn sweep_recursive(dir: &Path, removed: &mut usize) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue; };
        if ft.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if is_excluded_dir(&name_str) { continue; }
            sweep_recursive(&path, removed);
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue; };
        if ext == "mref" { continue; } // rides along with the data file
        let Some(canonical) = VecFormat::canonical_extension(ext) else { continue; };
        if canonical == ext { continue; } // already canonical
        let plural_path = path.with_extension(canonical);
        if !(plural_path.exists() || plural_path.symlink_metadata().is_ok()) {
            continue;
        }
        let data_existed = path.exists() || path.symlink_metadata().is_ok();
        remove_legacy_pair(&path);
        let still_there = path.exists() || path.symlink_metadata().is_ok();
        if data_existed && !still_there {
            *removed += 1;
        }
    }
}

fn remove_legacy_pair(legacy_path: &Path) {
    let legacy_mref = {
        let mut p = legacy_path.to_path_buf().into_os_string();
        p.push(".mref");
        std::path::PathBuf::from(p)
    };
    for p in [legacy_path, legacy_mref.as_path()] {
        if p.exists() || p.symlink_metadata().is_ok() {
            match std::fs::remove_file(p) {
                Ok(()) => println!("  Removed legacy sibling {}", p.display()),
                Err(e) => eprintln!("  Warning: failed to remove legacy {}: {}",
                    p.display(), e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_removes_singular_when_plural_present() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        // Plural canonical file (the "kept" form).
        std::fs::write(dir.join("base.fvecs"), b"plural").unwrap();
        // Legacy singular sibling + its .mref.
        std::fs::write(dir.join("base.fvec"), b"singular").unwrap();
        std::fs::write(dir.join("base.fvec.mref"), b"hash").unwrap();

        let n = sweep(dir);
        assert_eq!(n, 1, "should remove 1 data file");
        assert!(dir.join("base.fvecs").exists());
        assert!(!dir.join("base.fvec").exists());
        assert!(!dir.join("base.fvec.mref").exists());
    }

    #[test]
    fn sweep_preserves_singular_when_no_plural_sibling() {
        // A `.fvec` that is the only copy must not be removed —
        // it's the legitimate canonical form of *this* file.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("data.fvec"), b"only").unwrap();
        std::fs::write(dir.join("data.fvec.mref"), b"hash").unwrap();

        let n = sweep(dir);
        assert_eq!(n, 0);
        assert!(dir.join("data.fvec").exists());
        assert!(dir.join("data.fvec.mref").exists());
    }

    #[test]
    fn sweep_descends_subdirectories_and_skips_excluded() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::create_dir_all(dir.join("profiles/base")).unwrap();
        std::fs::create_dir_all(dir.join(".cache")).unwrap();
        std::fs::write(dir.join("profiles/base/x.fvecs"), b"a").unwrap();
        std::fs::write(dir.join("profiles/base/x.fvec"),  b"b").unwrap();
        std::fs::write(dir.join("profiles/base/x.fvec.mref"), b"h").unwrap();
        // Inside excluded dir — must NOT be swept.
        std::fs::write(dir.join(".cache/y.fvecs"), b"a").unwrap();
        std::fs::write(dir.join(".cache/y.fvec"),  b"b").unwrap();

        let n = sweep(dir);
        assert_eq!(n, 1);
        assert!(!dir.join("profiles/base/x.fvec").exists());
        assert!(!dir.join("profiles/base/x.fvec.mref").exists());
        assert!(dir.join("profiles/base/x.fvecs").exists());
        // Excluded dir untouched.
        assert!(dir.join(".cache/y.fvec").exists());
        assert!(dir.join(".cache/y.fvecs").exists());
    }

    #[test]
    fn sweep_handles_symlinks_via_symlink_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let target = dir.join("source.fvecs");
        std::fs::write(&target, b"data").unwrap();

        let canonical = dir.join("link.fvecs");
        let legacy = dir.join("link.fvec");
        std::os::unix::fs::symlink(&target, &canonical).unwrap();
        std::os::unix::fs::symlink(&target, &legacy).unwrap();

        let n = sweep(dir);
        assert_eq!(n, 1);
        assert!(canonical.symlink_metadata().is_ok());
        assert!(legacy.symlink_metadata().is_err());
    }
}
