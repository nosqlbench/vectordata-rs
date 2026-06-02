// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Local / `file://` push transport — a filesystem copy preserving the
//! relative tree. Used for local catalogs, mounts, and as the
//! fully-exercisable transport in tests.

use std::path::{Path, PathBuf};

use super::{PushError, PushTransport, RemoteObject};
use crate::push::checksums::sha256_bytes;

pub struct LocalTransport {
    root: PathBuf,
}

impl LocalTransport {
    /// Build from a `file://` URL. `file:///abs/path/` maps to the
    /// absolute path `/abs/path/`; the trailing slash is irrelevant.
    pub fn from_url(url: &str) -> Result<Self, String> {
        let path = url
            .strip_prefix("file://")
            .ok_or_else(|| format!("not a file:// url: {url}"))?;
        if path.is_empty() {
            return Err("file:// url has no path".to_string());
        }
        Ok(LocalTransport { root: PathBuf::from(path) })
    }

    fn path_for(&self, rel: &str) -> PathBuf {
        if rel.is_empty() {
            self.root.clone()
        } else {
            self.root.join(rel.trim_start_matches('/'))
        }
    }
}

impl PushTransport for LocalTransport {
    fn head(&self, rel: &str) -> Result<Option<RemoteObject>, PushError> {
        let p = self.path_for(rel);
        match std::fs::metadata(&p) {
            Ok(m) if m.is_file() => {
                // etag = content sha256, so conditional puts have a real
                // basis on a single-machine "remote".
                let bytes = std::fs::read(&p).map_err(io)?;
                Ok(Some(RemoteObject { size: m.len(), etag: Some(sha256_bytes(&bytes)) }))
            }
            Ok(_) => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io(e)),
        }
    }

    fn get(&self, rel: &str) -> Result<Option<Vec<u8>>, PushError> {
        match std::fs::read(self.path_for(rel)) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io(e)),
        }
    }

    fn put_file(&self, rel: &str, src: &Path) -> Result<(), PushError> {
        let dst = self.path_for(rel);
        let bytes = std::fs::read(src).map_err(io)?;
        atomic_write(&dst, &bytes)
    }

    fn put_bytes(&self, rel: &str, data: &[u8], if_match: Option<&str>) -> Result<(), PushError> {
        let dst = self.path_for(rel);
        if let Some(expected) = if_match {
            let current = self.head(rel)?;
            match (expected, current) {
                // "" means must-not-exist
                ("", Some(_)) => return Err(PushError::PreconditionFailed),
                ("", None) => {}
                (etag, Some(obj)) if obj.etag.as_deref() == Some(etag) => {}
                (_, _) => return Err(PushError::PreconditionFailed),
            }
        }
        atomic_write(&dst, data)
    }

    fn preflight(&self) -> Result<(), PushError> {
        std::fs::create_dir_all(&self.root).map_err(|e| {
            PushError::Other(format!("cannot create publish root {}: {e}", self.root.display()))
        })
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, PushError> {
        let base = self.path_for(prefix);
        let mut out = Vec::new();
        if base.is_dir() {
            walk(&self.root, &base, &mut out).map_err(io)?;
        }
        out.sort();
        Ok(out)
    }

    fn delete(&self, rel: &str) -> Result<(), PushError> {
        match std::fs::remove_file(self.path_for(rel)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(io(e)),
        }
    }

    fn describe(&self) -> String {
        format!("file://{}", self.root.display())
    }
}

/// Recursively collect file keys relative to `root` (forward-slashed).
fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_dir() {
            walk(root, &entry.path(), out)?;
        } else if ft.is_file()
            && let Ok(rel) = entry.path().strip_prefix(root)
        {
            out.push(
                rel.components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/"),
            );
        }
    }
    Ok(())
}

fn io(e: std::io::Error) -> PushError {
    PushError::Other(e.to_string())
}

/// Write `data` to `dst` crash-atomically: stage in a sibling temp file,
/// then `rename` over the target (atomic on the same filesystem). A crash
/// mid-write leaves either the old file or the new one — never a torn
/// half-written object.
fn atomic_write(dst: &Path, data: &[u8]) -> Result<(), PushError> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let file_name = dst.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dst.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()));
    std::fs::write(&tmp, data).map_err(io)?;
    match std::fs::rename(&tmp, dst) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(io(e))
        }
    }
}
