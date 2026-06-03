// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `local:` filesystem backend — atomic temp+rename writes under a
//! root directory. This is the I19/I20 lesson from the push spec: a write
//! is never observed half-applied.

use std::path::{Path, PathBuf};

use crate::backend::Backend;
use crate::model::VecdError;

pub struct LocalBackend {
    root: PathBuf,
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
