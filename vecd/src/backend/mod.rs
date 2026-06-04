// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Storage backends — the *plain byte store* beneath a namespace.
//!
//! All backends sit behind one synchronous [`Backend`] trait
//! (`get`/`put`/`head`/`delete`/`list`), so namespaces, AAA, and the CAS
//! authority are backend-agnostic. **Backends need not support
//! conditional writes**: `vecd`'s DB is the CAS authority (see
//! [`crate::store`]), so an ordinary S3 bucket or a local directory works
//! whether or not it offers native conditional `PutObject`.
//!
//! Built-in kinds (decision #8): `local` (atomic temp+rename), `s3`
//! (S3-compatible via the `aws` CLI, matching `vectordata push`'s s3
//! transport), and `mem` (in-process, ephemeral — tests / scratch).

use std::sync::Arc;

use crate::db::BackendRow;
use crate::model::VecdError;

pub mod local;
pub mod mem;
pub mod s3;

/// A plain byte store. Keys are already-normalized, namespace-confined
/// relative paths (see [`crate::store`] — the caller appends the
/// content-key substructure, the backend only moves bytes).
pub trait Backend: Send + Sync {
    /// Fetch an object's bytes, or `None` if absent.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, VecdError>;
    /// Store (create or overwrite) an object. Must be atomic enough that a
    /// reader never sees a torn write (local uses temp+rename).
    fn put(&self, key: &str, data: &[u8]) -> Result<(), VecdError>;

    /// Write `chunk` at byte `offset` of the **staging** blob `staging_key`,
    /// creating it if absent and zero-filling any hole before `offset`. This
    /// is the streaming/sparse write that backs both the plain whole-object
    /// `PUT` (sequential chunks) and resumable `PATCH` (out-of-order, sparse)
    /// — bytes go straight to the backend, never buffered whole in memory.
    /// Staging blobs are transient (an in-progress upload) and never appear
    /// in [`Backend::list`]; they become an object only at
    /// [`Backend::finalize_staged`].
    fn put_at(&self, staging_key: &str, offset: u64, chunk: &[u8]) -> Result<(), VecdError>;

    /// Atomically promote a completed staging blob to the object `final_key`
    /// (local: a same-filesystem rename). The move must be atomic enough that
    /// a reader never sees a torn object.
    fn finalize_staged(&self, staging_key: &str, final_key: &str) -> Result<(), VecdError>;

    /// Discard a staging blob — an abandoned (`DELETE` upload) or rejected
    /// (failed precondition / quota) upload. Idempotent if already gone.
    fn discard_staged(&self, staging_key: &str) -> Result<(), VecdError>;
    /// Object size, or `None` if absent.
    fn head(&self, key: &str) -> Result<Option<u64>, VecdError>;
    /// Remove an object; succeeds (idempotent) if already absent.
    fn delete(&self, key: &str) -> Result<(), VecdError>;
    /// List keys under a prefix. Used for orphan cleanup and quota sums.
    fn list(&self, prefix: &str) -> Result<Vec<String>, VecdError>;
    /// A human-readable description (for errors / `ns show`).
    fn describe(&self) -> String;
}

/// Open the [`Backend`] for a stored backend config row.
///
/// `mem` backends share a process-global store keyed by their `mem:<id>`
/// endpoint, so two namespaces pointing at the same `mem:` endpoint see
/// the same bytes within one daemon — matching the one-endpoint
/// semantics of durable backends.
pub fn open(row: &BackendRow) -> Result<Arc<dyn Backend>, VecdError> {
    match row.kind.as_str() {
        "local" => {
            let dir = row
                .endpoint
                .strip_prefix("local:")
                .ok_or_else(|| VecdError::usage(format!("local endpoint must be local:DIR, got '{}'", row.endpoint)))?;
            Ok(Arc::new(local::LocalBackend::new(dir)?))
        }
        "mem" => Ok(mem::open_shared(&row.endpoint)),
        "s3" => Ok(Arc::new(s3::S3Backend::new(row)?)),
        other => Err(VecdError::usage(format!("unknown backend kind '{other}'"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::BackendRow;

    fn row(kind: &str, endpoint: &str) -> BackendRow {
        BackendRow {
            name: "t".into(),
            kind: kind.into(),
            endpoint: endpoint.into(),
            endpoint_url: None,
            region: None,
            creds_ref: None,
            active: true,
        }
    }

    fn roundtrip(b: &dyn Backend) {
        assert!(b.get("a/b").unwrap().is_none());
        assert!(b.head("a/b").unwrap().is_none());
        b.put("a/b", b"hello").unwrap();
        assert_eq!(b.get("a/b").unwrap().unwrap(), b"hello");
        assert_eq!(b.head("a/b").unwrap(), Some(5));
        b.put("a/c", b"x").unwrap();
        let mut listed = b.list("a/").unwrap();
        listed.sort();
        assert_eq!(listed, vec!["a/b".to_string(), "a/c".to_string()]);
        b.delete("a/b").unwrap();
        assert!(b.get("a/b").unwrap().is_none());
        b.delete("a/b").unwrap(); // idempotent
    }

    /// Staging: out-of-order sparse writes assemble into the final object,
    /// finalize promotes it (and the staging blob never shows up in `list`),
    /// and discard removes an abandoned upload.
    fn staging_roundtrip(b: &dyn Backend) {
        // Sparse, out of order: tail then head → "HELLOworld".
        b.put_at("up1", 5, b"world").unwrap();
        b.put_at("up1", 0, b"HELLO").unwrap();
        // Not visible as an object yet, nor in listings.
        assert!(b.get("dst/obj").unwrap().is_none());
        assert!(b.list("").unwrap().iter().all(|k| !k.contains("up1")));
        b.finalize_staged("up1", "dst/obj").unwrap();
        assert_eq!(b.get("dst/obj").unwrap().unwrap(), b"HELLOworld");
        // Discard is idempotent and leaves no object behind.
        b.put_at("up2", 0, b"temp").unwrap();
        b.discard_staged("up2").unwrap();
        b.discard_staged("up2").unwrap();
        b.finalize_staged("up2", "dst/never").ok();
        assert!(b.get("dst/never").unwrap().is_none());
    }

    #[test]
    fn local_staging() {
        let dir = tempfile::tempdir().unwrap();
        let b = open(&row("local", &format!("local:{}", dir.path().display()))).unwrap();
        staging_roundtrip(b.as_ref());
    }

    #[test]
    fn mem_staging() {
        let b = open(&row("mem", "mem:staging")).unwrap();
        staging_roundtrip(b.as_ref());
    }

    #[test]
    fn local_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let b = open(&row("local", &format!("local:{}", dir.path().display()))).unwrap();
        roundtrip(b.as_ref());
    }

    #[test]
    fn mem_roundtrip_and_sharing() {
        let b = open(&row("mem", "mem:t1")).unwrap();
        roundtrip(b.as_ref());
        // Same endpoint → shared store.
        let b1 = open(&row("mem", "mem:shared")).unwrap();
        b1.put("k", b"v").unwrap();
        let b2 = open(&row("mem", "mem:shared")).unwrap();
        assert_eq!(b2.get("k").unwrap().unwrap(), b"v");
    }
}
