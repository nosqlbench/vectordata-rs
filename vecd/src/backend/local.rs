// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `local:` filesystem backend — atomic temp+rename writes under a
//! root directory. This is the I19/I20 lesson from the push spec: a write
//! is never observed half-applied.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::backend::Backend;
use crate::model::VecdError;

/// Directory (under the backend root) that holds in-progress upload staging
/// blobs. Hidden from object listings and on the same filesystem as the
/// objects, so finalize is an atomic same-fs rename.
const STAGING_DIR: &str = ".uploads";

pub struct LocalBackend {
    root: PathBuf,
}

/// Normalize a user-supplied local-backend directory to the canonical stored
/// form, `local:<absolute-dir>`.
///
/// Accepts any of `store`, `./data`, `/abs/dir`, or an already-prefixed
/// `local:store` — the `local:` prefix is optional on input. The directory part
/// is taken relative to the current directory (where the admin command runs),
/// created if missing, and canonicalized to an absolute path. Two reasons this
/// matters: the on-disk format `open()` expects is `local:DIR`, and the path
/// must be absolute because `vecd serve` may run from a different working
/// directory than the `vecd backends add` that created it — a relative endpoint
/// would silently resolve against the *server's* CWD (or nowhere).
///
/// Returns the canonical `local:<abs>` string; the caller can compare it against
/// the raw input to tell the user when a rewrite happened.
pub fn resolve_dir(raw: &str) -> Result<String, VecdError> {
    let dir_part = raw.strip_prefix("local:").unwrap_or(raw);
    if dir_part.trim().is_empty() {
        return Err(VecdError::usage("local backend directory must not be empty"));
    }
    let p = Path::new(dir_part);
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| {
                VecdError::usage(format!("resolving local backend directory '{dir_part}': {e}"))
            })?
            .join(p)
    };
    std::fs::create_dir_all(&abs).map_err(|e| {
        VecdError::usage(format!(
            "local backend directory '{}' can't be created or written: {e}",
            abs.display()
        ))
    })?;
    // canonicalize requires existence (created above); fall back to the absolute
    // path if the FS can't canonicalize (e.g. exotic mounts).
    let resolved = std::fs::canonicalize(&abs).unwrap_or(abs);
    Ok(format!("local:{}", resolved.to_string_lossy()))
}

impl LocalBackend {
    pub fn new(dir: &str) -> Result<Self, VecdError> {
        let root = PathBuf::from(dir);
        std::fs::create_dir_all(&root)?;
        Ok(LocalBackend { root })
    }

    /// Map a confined relative key onto a filesystem path. Keys are
    /// already validated by [`crate::store`] (no `..`, no leading `/`), so
    /// this is a plain join; we still reject anything that would escape
    /// the root as defense in depth.
    fn path_of(&self, key: &str) -> Result<PathBuf, VecdError> {
        if key.is_empty() || key.starts_with('/') || key.split('/').any(|c| c == ".." || c == ".") {
            return Err(VecdError::usage(format!("invalid object key '{key}'")));
        }
        Ok(self.root.join(key))
    }

    /// Filesystem path of a staging blob. The `staging_key` is an opaque,
    /// server-generated id (no path separators), so a plain join is safe.
    fn staging_path(&self, staging_key: &str) -> PathBuf {
        self.root.join(STAGING_DIR).join(staging_key)
    }
}

impl Backend for LocalBackend {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, VecdError> {
        match std::fs::read(self.path_of(key)?) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<(), VecdError> {
        let path = self.path_of(key)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // temp + rename within the same directory → atomic replace.
        let tmp = path.with_extension(format!(
            "tmp-{}",
            std::process::id()
        ));
        std::fs::write(&tmp, data)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn put_at(&self, staging_key: &str, offset: u64, chunk: &[u8]) -> Result<(), VecdError> {
        let path = self.staging_path(staging_key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Open-or-create, then seek to the offset. Seeking past the current
        // end leaves a hole that reads back as zeros (a sparse file) — the
        // gap a later out-of-order chunk will fill.
        let mut f = std::fs::OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(chunk)?;
        Ok(())
    }

    fn finalize_staged(&self, staging_key: &str, final_key: &str) -> Result<(), VecdError> {
        let from = self.staging_path(staging_key);
        let to = self.path_of(final_key)?;
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Same-filesystem rename → atomic promotion; no torn read.
        std::fs::rename(&from, &to)?;
        Ok(())
    }

    fn discard_staged(&self, staging_key: &str) -> Result<(), VecdError> {
        match std::fs::remove_file(self.staging_path(staging_key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn head(&self, key: &str) -> Result<Option<u64>, VecdError> {
        match std::fs::metadata(self.path_of(key)?) {
            Ok(m) => Ok(Some(m.len())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn delete(&self, key: &str) -> Result<(), VecdError> {
        match std::fs::remove_file(self.path_of(key)?) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, VecdError> {
        let mut out = Vec::new();
        walk(&self.root, &self.root, &mut out)?;
        if !prefix.is_empty() {
            out.retain(|k| k.starts_with(prefix));
        }
        Ok(out)
    }

    fn describe(&self) -> String {
        format!("local:{}", self.root.display())
    }
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), VecdError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // The staging area is internal — never an object listing.
            if path.file_name().and_then(|n| n.to_str()) == Some(STAGING_DIR) {
                continue;
            }
            walk(root, &path, out)?;
        } else if let Ok(rel) = path.strip_prefix(root) {
            // Skip any in-flight temp files.
            let name = rel.to_string_lossy().to_string();
            if !name.contains(".tmp-") {
                out.push(name.replace(std::path::MAIN_SEPARATOR, "/"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    #[test]
    fn resolve_dir_normalizes_to_absolute_local_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("store");

        // An absolute directory → `local:<abs>`, and it gets created.
        let out = resolve_dir(sub.to_str().unwrap()).unwrap();
        assert!(out.starts_with("local:/"), "want absolute local: prefix, got {out}");
        assert!(sub.is_dir(), "the directory is created");

        // The `local:` prefix is optional on input and yields the same result.
        let out2 = resolve_dir(&format!("local:{}", sub.display())).unwrap();
        assert_eq!(out, out2);

        // Empty / prefix-only → a clear error, not a silent bad endpoint.
        assert!(resolve_dir("local:").is_err());
        assert!(resolve_dir("   ").is_err());
    }
}
