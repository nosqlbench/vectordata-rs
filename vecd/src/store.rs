// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The object store — *the `objectstore.rs` test mock grown up*: a real
//! blob store, real ETags, and **conditional writes honored**, with
//! `vecd`'s DB as the **CAS authority** (decision #6).
//!
//! Each object's identity is its **envelope ETag** — a sha256 over a
//! canonical serialization of the write's *envelope metadata* (`namespace`,
//! `key`, `size`, and a monotonic per-write `seq`), **never of the content
//! bytes**. `vecd` does not hash, merkle, or otherwise inspect object
//! content (the non-negotiable ruling in
//! `local/vecd-resumable-upload-SPEC.md` §1): it authenticates and
//! authorizes the caller, then stores the bytes faithfully. Integrity is
//! the client's job — `vectordata push` verifies via its separate
//! `SHA256SUMS` round-trip and treats ETags as opaque.
//!
//! Because the `seq` advances on every committed write, the ETag **changes
//! on every write** (even when identical bytes are re-published), so
//! `If-Match`/`If-None-Match` CAS stays meaningful and each committed write
//! lands at a distinct content-addressed blob path `<rel_key>/<etag>`
//! (copy-on-write friendly). The flip side — dropped from the prior
//! merkle-keyed scheme — is **cross-object content dedup**: two byte-
//! identical objects now occupy two blobs. The user accepts that trade.
//!
//! The ETag lives in the `objects` table (its column is still named
//! `content_key` for schema continuity); `If-Match`/`If-None-Match` are
//! evaluated against that row **inside an immediate transaction**, so the
//! single-provenance guarantee `vectordata push` relies on holds over any
//! (plain byte-store) backend — even one with no conditional-write API. The
//! SQLite write lock held across the check-then-write is the serialization
//! point; the blob path is unique per write, so a write never tears a
//! concurrent read of the prior version.

use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::db::Db;
use crate::model::VecdError;
use crate::namespace::Resolved;

/// Version tag on the canonical envelope-ETag descriptor serialization, so
/// the scheme can evolve without silently changing how ETags are minted.
const ETAG_SCHEME: &str = "vecd-etag/1";

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

/// Mint an object's **envelope ETag** — its identity, ETag, and the blob
/// path component.
///
/// This is a sha256 over a canonical serialization of the write's
/// *envelope metadata* — never of the content bytes (no hashing, no merkle;
/// the bytes are never read here). The `seq` is a per-write monotonic value
/// (see [`Db::next_etag_seq`]), so the ETag is unique to this write: CAS
/// stays meaningful and the blob lands at a distinct path even when the same
/// bytes are re-published.
pub fn envelope_etag(storage_ns: &str, key: &str, size: u64, seq: i64) -> String {
    // Canonical descriptor: a versioned, fixed-order serialization of the
    // envelope. Hash the descriptor, never the bytes.
    let descriptor =
        format!("{ETAG_SCHEME}\nns={storage_ns}\nkey={key}\nsize={size}\nseq={seq}\n");
    let mut h = Sha256::new();
    h.update(descriptor.as_bytes());
    hex::encode(h.finalize())
}

/// The content-addressed physical key beneath the logical key.
pub fn physical(rel_key: &str, ck: &str) -> String {
    format!("{rel_key}/{ck}")
}

/// Allocate a fresh envelope ETag and write the object's bytes to the
/// backend at its (unique) content-addressed path. Shared by the lone-write
/// and session-staging paths. Returns `(etag, size)`.
pub fn write_blob(db: &Db, r: &Resolved, data: &[u8]) -> Result<(String, u64), VecdError> {
    let seq = db.next_etag_seq()?;
    let etag = envelope_etag(&r.storage_ns, &r.rel_key, data.len() as u64, seq);
    r.backend.put(&physical(&r.rel_key, &etag), data)?;
    Ok((etag, data.len() as u64))
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

/// `PUT` a whole, buffered object with a conditional-write precondition,
/// enforced transactionally. `ignore_quota` is set when the caller holds
/// `IGNORE-QUOTAS`.
pub fn put(
    db: &mut Db,
    r: &Resolved,
    data: &[u8],
    precondition: &Precondition,
    quota_bytes: u64,
    ignore_quota: bool,
) -> Result<PutResult, VecdError> {
    // Mint the envelope ETag and write its blob first. The blob path is
    // unique per write, so a blob with no committed row is a harmless
    // orphan — the DB row is the authority for existence.
    let (new_ck, new_size) = write_blob(db, r, data)?;
    finish_put(db, r, new_ck, new_size, precondition, quota_bytes, ignore_quota)
}

/// `PUT` a **streamed** object whose bytes are already assembled in a
/// backend staging blob (`staging_key`, `size` bytes, written chunk-by-chunk
/// via [`Backend::put_at`]). The same CAS + quota contract as [`put`], with
/// no whole-object buffering: the staged blob is promoted to its content
/// path and then the live manifest row is recorded transactionally.
///
/// [`Backend::put_at`]: crate::backend::Backend::put_at
pub fn put_staged(
    db: &mut Db,
    r: &Resolved,
    staging_key: &str,
    size: u64,
    precondition: &Precondition,
    quota_bytes: u64,
    ignore_quota: bool,
) -> Result<PutResult, VecdError> {
    // Promote the staged bytes to a unique content path first (orphan-
    // tolerant, exactly like the buffered `put`).
    let (new_ck, new_size) = commit_staged_blob(db, r, staging_key, size)?;
    finish_put(db, r, new_ck, new_size, precondition, quota_bytes, ignore_quota)
}

/// Promote an upload's staged bytes into a uniquely-keyed object blob: mint
/// the envelope ETag for `size` and atomically move the staging blob to its
/// content path. Returns `(etag, size)`. Shared by the lone-write and
/// session-staging streamed paths.
pub fn commit_staged_blob(
    db: &Db,
    r: &Resolved,
    staging_key: &str,
    size: u64,
) -> Result<(String, u64), VecdError> {
    let seq = db.next_etag_seq()?;
    let etag = envelope_etag(&r.storage_ns, &r.rel_key, size, seq);
    r.backend.finalize_staged(staging_key, &physical(&r.rel_key, &etag))?;
    Ok((etag, size))
}

/// The transactional tail shared by [`put`] and [`put_staged`]: evaluate the
/// precondition + quota against the committed state and, on success, record
/// the live manifest row. The blob (`new_ck`) is already written.
fn finish_put(
    db: &mut Db,
    r: &Resolved,
    new_ck: String,
    new_size: u64,
    precondition: &Precondition,
    quota_bytes: u64,
    ignore_quota: bool,
) -> Result<PutResult, VecdError> {
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

    // Quota: the subtree's live footprint after this write must not exceed
    // the cap. Content dedup is gone (envelope ETags are unique per write),
    // so usage is a plain sum of live sizes; an overwrite of this key
    // replaces its prior size rather than adding to it.
    if !ignore_quota {
        let usage = subtree_usage(&tx, &r.storage_ns)?;
        let prior = current.as_ref().map(|(_, sz)| (*sz).max(0) as u64).unwrap_or(0);
        let projected = usage.saturating_sub(prior).saturating_add(new_size);
        if projected > quota_bytes {
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

/// `DELETE` — remove the live key. Its blob is reclaimed only when **no
/// retained version, live manifest entry, or open session** still
/// references it. A committed version's frozen manifest (`version_objects`)
/// keeps a pinned read (`@v3/key`) intact even after the live key is
/// deleted; a purge is the path that removes those. (This mirrors
/// [`crate::lifetime::purge`]'s reference check — without it, envelope
/// ETags, which carry no dedup safety net, would let a lone delete strand a
/// version's bytes.)
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
                "SELECT 1 FROM (
                    SELECT content_key FROM objects WHERE content_key=?1
                    UNION ALL SELECT content_key FROM version_objects WHERE content_key=?1
                    UNION ALL SELECT content_key FROM staging_objects WHERE content_key=?1
                 ) LIMIT 1",
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

/// Live footprint (bytes) of a namespace subtree — the sum of the live
/// manifest's object sizes. With envelope ETags there is no cross-object
/// content dedup, so this is a plain per-key sum.
fn subtree_usage(tx: &rusqlite::Transaction, storage_ns: &str) -> Result<u64, VecdError> {
    let total: i64 = tx.query_row(
        "SELECT COALESCE(SUM(size),0) FROM objects
         WHERE namespace_path=?1 OR namespace_path LIKE ?1||'/%'",
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
    fn envelope_etag_is_metadata_only_and_unique_per_write() {
        // Stable for a fixed (ns, key, size, seq) tuple — but it never sees
        // content, so it cannot be the whole-content sha256.
        let etag = envelope_etag("ds", "x", 5, 1);
        assert_eq!(etag, envelope_etag("ds", "x", 5, 1));
        let flat = {
            let mut h = Sha256::new();
            h.update(b"HELLO");
            hex::encode(h.finalize())
        };
        assert_ne!(etag, flat, "envelope ETag must not be a whole-content hash");
        // Distinct per write (the seq bumps), so re-publishing identical
        // bytes still changes the ETag and keeps CAS meaningful.
        assert_ne!(envelope_etag("ds", "x", 5, 1), envelope_etag("ds", "x", 5, 2));
        // Distinct per envelope coordinate (ns / key / size) too.
        assert_ne!(envelope_etag("ds", "x", 5, 1), envelope_etag("ds", "y", 5, 1));
        assert_ne!(envelope_etag("ds", "x", 5, 1), envelope_etag("ns2", "x", 5, 1));
        assert_ne!(envelope_etag("ds", "x", 5, 1), envelope_etag("ds", "x", 6, 1));
    }

    #[test]
    fn identical_bytes_get_distinct_etags_no_dedup() {
        let (_d, mut db, snap) = fixture(1 << 30);
        let e1 = match put(&mut db, &resolved(&snap, "ds/a"), b"same", &Precondition::None, 1 << 30, false).unwrap() {
            PutResult::Written { etag } => etag,
            o => panic!("{o:?}"),
        };
        let e2 = match put(&mut db, &resolved(&snap, "ds/b"), b"same", &Precondition::None, 1 << 30, false).unwrap() {
            PutResult::Written { etag } => etag,
            o => panic!("{o:?}"),
        };
        // No cross-object content dedup: byte-identical objects, distinct ETags.
        assert_ne!(e1, e2);
        assert_eq!(get(&db, &resolved(&snap, "ds/a")).unwrap().unwrap().0, b"same");
        assert_eq!(get(&db, &resolved(&snap, "ds/b")).unwrap().unwrap().0, b"same");
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
