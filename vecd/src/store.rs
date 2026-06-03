// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The object store — *the `objectstore.rs` test mock grown up*: a real
//! blob store, real ETags, and **conditional writes honored**, with
//! `vecd`'s DB as the **CAS authority** (decision #6).
//!
//! Each object's identity is its **content-key**, which is also the ETag.
//! Per the design's terminology invariant (`docs/design/vecd-daemon.md`
//! § *Hashing & addressing*) a content-key is a **canonical descriptive-
//! metadata hash — never a whole-content hash**: a sha256 over a canonical
//! serialization of the object's descriptors (its byte length and its
//! **full-content fingerprint, computed via a merkle tree**). Because the
//! descriptor embeds a full-content merkle root, the content-key is
//! *content-determining and global* (byte-identical objects share one
//! content-key everywhere — the basis of dedup), while remaining, by
//! construction, a metadata hash. Byte-level integrity rides on the merkle
//! tree, not on the content-key.
//!
//! The content-key lives in the `objects` table; `If-Match` /
//! `If-None-Match` are evaluated against that row **inside an immediate
//! transaction**, so the single-provenance guarantee `vectordata push`
//! relies on holds over any (plain byte-store) backend — even one with no
//! conditional-write API. The SQLite write lock held across the
//! check-then-write is the serialization point; bytes are written to the
//! content-addressed path `<rel_key>/<content_key>` so a write is
//! idempotent and never tears a concurrent read of the prior version.
//!
//! Phase 1 keeps a flat per-namespace manifest (one row per live key); the
//! COW *version* tree is a Phase 2 refinement (see [`crate::db`] docs).

use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};
use vectordata::merkle::MerkleRef;

use crate::db::Db;
use crate::model::VecdError;
use crate::namespace::Resolved;

/// Chunk size for the full-content merkle fingerprint embedded in a
/// content-key. Matches `vectordata`'s `.mref` generation convention
/// (`datasets::derive`'s 1 MiB) so the fingerprint is consistent with the
/// rest of the system and chunk-resumable verification lines up.
const MERKLE_CHUNK_SIZE: u64 = 1024 * 1024;

/// Version tag on the canonical content-key descriptor serialization, so
/// the scheme can evolve without silently changing existing keys.
const CONTENT_KEY_SCHEME: &str = "vecd-ck/1";

/// A conditional-write precondition parsed from the request headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Precondition {
    /// No condition — unconditional PUT.
    None,
    /// `If-None-Match: *` — succeed only if the object is absent.
    IfNoneMatchStar,
    /// `If-Match: "<etag>"` — succeed only if the current ETag matches.
    IfMatch(String),
}

/// The result of a conditional PUT.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PutResult {
    /// Stored; the new ETag (content-key).
    Written { etag: String },
    /// A precondition (`If-Match`/`If-None-Match`) was not satisfied → 412.
    PreconditionFailed,
    /// The write would exceed the namespace's quota → 507.
    QuotaExceeded { quota: u64 },
}

/// Object metadata for `HEAD`/`GET` responses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectMeta {
    pub size: u64,
    pub etag: String,
}

/// Compute an object's **content-key** — its identity and ETag.
///
/// This is a *canonical descriptive-metadata hash*, **not** a hash of the
/// byte stream: a sha256 over a canonical serialization of the object's
/// descriptors — its byte length and its full-content fingerprint (the
/// **merkle root**, full content hashing via a merkle tree). Embedding the
/// merkle root makes the key content-determining and global (so dedup and
/// content-addressed fetch work) while keeping it, by construction, a
/// metadata hash distinct from whole-content hashing.
pub fn content_key(data: &[u8]) -> String {
    let merkle_root = if data.is_empty() {
        // An empty tree has no root node; use the canonical empty-content
        // hash (the same value merkle padding leaves use for empty chunks).
        let mut h = Sha256::new();
        h.update(b"");
        hex::encode(h.finalize())
    } else {
        hex::encode(MerkleRef::from_content(data, MERKLE_CHUNK_SIZE).root_hash())
    };
    // Canonical descriptor: a versioned, fixed-order serialization of the
    // metadata. Hash the descriptor, never the bytes.
    let descriptor = format!(
        "{CONTENT_KEY_SCHEME}\nsize={}\nmerkle_sha256={merkle_root}\n",
        data.len(),
    );
    let mut h = Sha256::new();
    h.update(descriptor.as_bytes());
    hex::encode(h.finalize())
}

/// The content-addressed physical key beneath the logical key.
pub fn physical(rel_key: &str, ck: &str) -> String {
    format!("{rel_key}/{ck}")
}

/// Compute an object's content-key and write its bytes to the backend at
/// the content-addressed path (idempotent — same content → same path).
/// Shared by the lone-write and session-staging paths. Returns
/// `(content_key, size)`.
pub fn write_blob(r: &Resolved, data: &[u8]) -> Result<(String, u64), VecdError> {
    let ck = content_key(data);
    r.backend.put(&physical(&r.rel_key, &ck), data)?;
    Ok((ck, data.len() as u64))
}

/// `HEAD` — object metadata, or `None` if absent.
pub fn head(db: &Db, r: &Resolved) -> Result<Option<ObjectMeta>, VecdError> {
    let row: Option<(String, i64)> = db
        .conn()
        .query_row(
            "SELECT content_key, size FROM objects WHERE namespace_path=?1 AND key=?2",
            params![r.storage_ns, r.rel_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    Ok(row.map(|(etag, size)| ObjectMeta { size: size.max(0) as u64, etag }))
}

/// `GET` — object bytes + metadata, or `None` if absent.
pub fn get(db: &Db, r: &Resolved) -> Result<Option<(Vec<u8>, ObjectMeta)>, VecdError> {
    let Some(meta) = head(db, r)? else { return Ok(None) };
    match r.backend.get(&physical(&r.rel_key, &meta.etag))? {
        Some(bytes) => Ok(Some((bytes, meta))),
        // DB says it exists but the backend lost the blob — a torn store.
        None => Err(VecdError::op(format!(
            "object '{}' is recorded but its content is missing from {}",
            r.rel_key,
            r.backend.describe()
        ))),
    }
}

/// `PUT` with a conditional-write precondition, enforced transactionally.
/// `ignore_quota` is set when the caller holds `IGNORE-QUOTAS`.
pub fn put(
    db: &mut Db,
    r: &Resolved,
    data: &[u8],
    precondition: &Precondition,
    quota_bytes: u64,
    ignore_quota: bool,
) -> Result<PutResult, VecdError> {
    let new_ck = content_key(data);
    let new_size = data.len() as u64;

    // Write the content-addressed blob first (idempotent: same content →
    // same path). A blob with no committed row is a harmless orphan,
    // reclaimable later — the DB row is the authority for existence.
    r.backend.put(&physical(&r.rel_key, &new_ck), data)?;

    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

    let current: Option<(String, i64)> = tx
        .query_row(
            "SELECT content_key, size FROM objects WHERE namespace_path=?1 AND key=?2",
            params![r.storage_ns, r.rel_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;

    // Evaluate the precondition against the committed state.
    match precondition {
        Precondition::None => {}
        Precondition::IfNoneMatchStar => {
            if current.is_some() {
                return Ok(PutResult::PreconditionFailed);
            }
        }
        Precondition::IfMatch(etag) => {
            match &current {
                Some((ck, _)) if ck == etag => {}
                _ => return Ok(PutResult::PreconditionFailed),
            }
        }
    }

    // Quota: deduped-content usage of the subtree must not exceed the cap.
    if !ignore_quota {
        let usage = subtree_usage(&tx, &r.storage_ns)?;
        let already_present: bool = tx
            .query_row(
                "SELECT 1 FROM objects WHERE content_key=?1 AND (namespace_path=?2 OR namespace_path LIKE ?2||'/%') LIMIT 1",
                params![new_ck, r.storage_ns],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        // Conservative: new content adds to usage only when not already
        // deduplicated within the subtree.
        if !already_present && usage.saturating_add(new_size) > quota_bytes {
            return Ok(PutResult::QuotaExceeded { quota: quota_bytes });
        }
    }

    tx.execute(
        "INSERT INTO objects(namespace_path,key,content_key,size,created)
         VALUES(?1,?2,?3,?4,strftime('%s','now'))
         ON CONFLICT(namespace_path,key) DO UPDATE SET content_key=excluded.content_key, size=excluded.size",
        params![r.storage_ns, r.rel_key, new_ck, new_size as i64],
    )?;
    tx.commit()?;

    Ok(PutResult::Written { etag: new_ck })
}

/// `DELETE` — remove the live key. The blob is removed too when no other
/// object row references its content-key (Phase 1 has no version history
/// to protect; cross-key dedup is still respected).
pub fn delete(db: &mut Db, r: &Resolved) -> Result<(), VecdError> {
    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let ck: Option<String> = tx
        .query_row(
            "SELECT content_key FROM objects WHERE namespace_path=?1 AND key=?2",
            params![r.storage_ns, r.rel_key],
            |row| row.get(0),
        )
        .optional()?;
    tx.execute(
        "DELETE FROM objects WHERE namespace_path=?1 AND key=?2",
        params![r.storage_ns, r.rel_key],
    )?;
    let orphaned = if let Some(ck) = &ck {
        let still_referenced: bool = tx
            .query_row(
                "SELECT 1 FROM objects WHERE content_key=?1 LIMIT 1",
                params![ck],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        (!still_referenced).then(|| ck.clone())
    } else {
        None
    };
    tx.commit()?;
    // Best-effort blob reclamation outside the txn.
    if let Some(ck) = orphaned {
        let _ = r.backend.delete(&physical(&r.rel_key, &ck));
    }
    Ok(())
}

/// List live logical keys under the storage namespace (for `?list` and
/// push `--delete` orphan detection).
pub fn list(db: &Db, storage_ns: &str) -> Result<Vec<(String, String)>, VecdError> {
    let mut stmt = db
        .conn()
        .prepare("SELECT key, content_key FROM objects WHERE namespace_path=?1 ORDER BY key")?;
    let rows = stmt
        .query_map(params![storage_ns], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Deduped-content usage (bytes) of a namespace subtree.
fn subtree_usage(tx: &rusqlite::Transaction, storage_ns: &str) -> Result<u64, VecdError> {
    let total: i64 = tx.query_row(
        "SELECT COALESCE(SUM(size),0) FROM (
            SELECT DISTINCT content_key, size FROM objects
            WHERE namespace_path=?1 OR namespace_path LIKE ?1||'/%'
         )",
        params![storage_ns],
        |r| r.get(0),
    )?;
    Ok(total.max(0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authz::Snapshot;
    use crate::db::{BackendRow, ControlPlane, NamespaceRow};
    use crate::namespace;

    /// A store fixture: a DB + a snapshot with one mem-backed namespace.
    fn fixture(quota: u64) -> (tempfile::TempDir, Db, Snapshot) {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::init(&dir.path().join("vecd.db")).unwrap();
        let mut cp = ControlPlane::default();
        cp.backends.push(BackendRow {
            name: "b".into(),
            kind: "mem".into(),
            // unique per test dir so stores don't collide across tests
            endpoint: format!("mem:{}", dir.path().display()),
            endpoint_url: None,
            region: None,
            creds_ref: None,
            active: true,
        });
        cp.namespaces.push(NamespaceRow {
            path: "ds".into(),
            owner: "@admin".into(),
            backend_config: Some("b".into()),
            active: true,
            listable: "grantees".into(),
            quota_bytes: quota as i64,
            ttl_seconds: None,
        });
        (dir, db, Snapshot::build(&cp))
    }

    fn resolved(snap: &Snapshot, path: &str) -> Resolved {
        namespace::resolve(snap, path).unwrap()
    }

    #[test]
    fn content_key_is_a_metadata_hash_not_whole_content_sha256() {
        let data = b"some object bytes for the content key";
        let ck = content_key(data);
        // It must NOT equal the flat sha256 of the bytes (the invariant).
        let flat = {
            let mut h = Sha256::new();
            h.update(data);
            hex::encode(h.finalize())
        };
        assert_ne!(ck, flat, "content-key must be a descriptive-metadata hash, not whole-content sha256");
        // Content-determining & stable: same bytes → same key.
        assert_eq!(ck, content_key(data));
        // Different bytes (and a differing merkle root) → different key.
        assert_ne!(ck, content_key(b"other bytes"));
        // Empty content is handled without panicking and is stable.
        assert_eq!(content_key(b""), content_key(b""));
        assert_ne!(content_key(b""), ck);
    }

    #[test]
    fn put_get_head_roundtrip() {
        let (_d, mut db, snap) = fixture(1 << 30);
        let r = resolved(&snap, "ds/base.fvec");
        assert!(head(&db, &r).unwrap().is_none());
        let res = put(&mut db, &r, b"HELLO", &Precondition::None, 1 << 30, false).unwrap();
        let etag = match res {
            PutResult::Written { etag } => etag,
            other => panic!("{other:?}"),
        };
        assert_eq!(head(&db, &r).unwrap().unwrap(), ObjectMeta { size: 5, etag: etag.clone() });
        let (bytes, meta) = get(&db, &r).unwrap().unwrap();
        assert_eq!(bytes, b"HELLO");
        assert_eq!(meta.etag, etag);
    }

    #[test]
    fn if_none_match_star_blocks_overwrite() {
        let (_d, mut db, snap) = fixture(1 << 30);
        let r = resolved(&snap, "ds/x");
        assert!(matches!(
            put(&mut db, &r, b"a", &Precondition::IfNoneMatchStar, 1 << 30, false).unwrap(),
            PutResult::Written { .. }
        ));
        assert_eq!(
            put(&mut db, &r, b"b", &Precondition::IfNoneMatchStar, 1 << 30, false).unwrap(),
            PutResult::PreconditionFailed
        );
    }

    #[test]
    fn if_match_requires_current_etag() {
        let (_d, mut db, snap) = fixture(1 << 30);
        let r = resolved(&snap, "ds/x");
        let etag = match put(&mut db, &r, b"a", &Precondition::None, 1 << 30, false).unwrap() {
            PutResult::Written { etag } => etag,
            o => panic!("{o:?}"),
        };
        // Stale etag refused; correct etag accepted.
        assert_eq!(
            put(&mut db, &r, b"b", &Precondition::IfMatch("deadbeef".into()), 1 << 30, false).unwrap(),
            PutResult::PreconditionFailed
        );
        assert!(matches!(
            put(&mut db, &r, b"b", &Precondition::IfMatch(etag), 1 << 30, false).unwrap(),
            PutResult::Written { .. }
        ));
    }

    #[test]
    fn quota_exceeded_is_507() {
        let (_d, mut db, snap) = fixture(4); // 4-byte cap
        let r = resolved(&snap, "ds/x");
        assert!(matches!(
            put(&mut db, &r, b"ab", &Precondition::None, 4, false).unwrap(),
            PutResult::Written { .. }
        ));
        let r2 = resolved(&snap, "ds/y");
        assert_eq!(
            put(&mut db, &r2, b"cde", &Precondition::None, 4, false).unwrap(),
            PutResult::QuotaExceeded { quota: 4 }
        );
        // IGNORE-QUOTAS bypasses it.
        assert!(matches!(
            put(&mut db, &r2, b"cde", &Precondition::None, 4, true).unwrap(),
            PutResult::Written { .. }
        ));
    }

    #[test]
    fn delete_removes_row_and_unreferenced_blob() {
        let (_d, mut db, snap) = fixture(1 << 30);
        let r = resolved(&snap, "ds/x");
        put(&mut db, &r, b"data", &Precondition::None, 1 << 30, false).unwrap();
        delete(&mut db, &r).unwrap();
        assert!(head(&db, &r).unwrap().is_none());
        assert!(get(&db, &r).unwrap().is_none());
    }

    #[test]
    fn list_returns_live_keys() {
        let (_d, mut db, snap) = fixture(1 << 30);
        put(&mut db, &resolved(&snap, "ds/a"), b"1", &Precondition::None, 1 << 30, false).unwrap();
        put(&mut db, &resolved(&snap, "ds/b"), b"2", &Precondition::None, 1 << 30, false).unwrap();
        let keys: Vec<String> = list(&db, "ds").unwrap().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
    }
}
