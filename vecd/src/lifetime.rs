// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Data lifetimes — **expire to stasis, never auto-delete**.
//!
//! The unit of lifecycle is the committed **version** (a version gets
//! `expires_at = committed_at + ns.ttl`). A background sweep moves expired
//! committed versions to `stasis` (records `stasis_at`): they become
//! invisible to clients (`GET`/`HEAD` → `410 Gone`) while every byte and
//! manifest row is retained. An admin then either **extends** (re-lifecycle
//! + restore) or **purges** (the only path that physically removes bytes,
//! always explicit) from the cleanup queue.
//!
//! When a namespace's current (latest) version moves to stasis, the live
//! manifest (`objects`) is rebuilt from the next non-stasis committed
//! version — a rollback — or cleared if none remain.

use rusqlite::{params, OptionalExtension};

use crate::backend::Backend;
use crate::db::Db;
use crate::model::VecdError;

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// A version awaiting an admin decision (in the cleanup queue).
#[derive(Clone, Debug)]
pub struct StasisItem {
    pub namespace_path: String,
    pub seq: i64,
    pub tag: String,
    pub stasis_at: Option<i64>,
    pub manifest_hash: String,
}

/// Move every committed version past its `expires_at` to stasis, rebuilding
/// each affected namespace's live manifest from its newest surviving
/// version. Returns how many versions were swept. Idempotent.
pub fn sweep(db: &mut Db) -> Result<usize, VecdError> {
    let ts = now();
    let expired: Vec<(i64, String)> = db
        .conn()
        .prepare(
            "SELECT id, namespace_path FROM versions
             WHERE state='committed' AND expires_at IS NOT NULL AND expires_at <= ?1",
        )?
        .query_map(params![ts], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    if expired.is_empty() {
        return Ok(0);
    }
    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    for (id, _) in &expired {
        tx.execute(
            "UPDATE versions SET state='stasis', stasis_at=?1 WHERE id=?2",
            params![ts, id],
        )?;
    }
    let mut seen = std::collections::HashSet::new();
    for (_, ns) in &expired {
        if seen.insert(ns.clone()) {
            rebuild_live(&tx, ns)?;
        }
    }
    tx.commit()?;
    Ok(expired.len())
}

/// Rebuild the live manifest (`objects`) for a namespace from its current
/// (latest committed, non-stasis) version, or clear it if none remain.
fn rebuild_live(tx: &rusqlite::Transaction, ns: &str) -> Result<(), VecdError> {
    tx.execute("DELETE FROM objects WHERE namespace_path=?1", params![ns])?;
    let current: Option<i64> = tx
        .query_row(
            "SELECT id FROM versions WHERE namespace_path=?1 AND state='committed'
             ORDER BY seq DESC LIMIT 1",
            params![ns],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(vid) = current {
        tx.execute(
            "INSERT INTO objects(namespace_path,key,content_key,size,created)
             SELECT ?1,key,content_key,size,?2 FROM version_objects WHERE version_id=?3",
            params![ns, now(), vid],
        )?;
    }
    Ok(())
}

/// The cleanup queue — every version currently in stasis.
pub fn pending(db: &Db) -> Result<Vec<StasisItem>, VecdError> {
    Ok(db
        .conn()
        .prepare(
            "SELECT namespace_path,seq,tag,stasis_at,manifest_hash FROM versions
             WHERE state='stasis' ORDER BY namespace_path, seq",
        )?
        .query_map([], |r| {
            Ok(StasisItem {
                namespace_path: r.get(0)?,
                seq: r.get(1)?,
                tag: r.get(2)?,
                stasis_at: r.get(3)?,
                manifest_hash: r.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?)
}

/// True iff the namespace has versions but no committed (non-stasis) one —
/// i.e. its data has wholly expired (so `@latest` answers `410`).
pub fn ns_in_stasis(db: &Db, ns: &str) -> Result<bool, VecdError> {
    let total: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM versions WHERE namespace_path=?1", params![ns], |r| r.get(0))?;
    if total == 0 {
        return Ok(false);
    }
    let live: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM versions WHERE namespace_path=?1 AND state='committed'",
        params![ns],
        |r| r.get(0),
    )?;
    Ok(live == 0)
}

/// Restore a stasis version: re-lifecycle it (`duration` seconds, or no
/// limit when `None`) and bring it back as a committed version, rebuilding
/// the live manifest.
pub fn extend(db: &mut Db, ns: &str, selector: &str, duration: Option<i64>) -> Result<(), VecdError> {
    let Some(v) = crate::session::find_version(db, ns, selector)? else {
        return Err(VecdError::usage(format!("no version '{selector}' in namespace '{ns}'")));
    };
    let expires_at = duration.map(|d| now() + d);
    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute(
        "UPDATE versions SET state='committed', stasis_at=NULL, expires_at=?1 WHERE id=?2",
        params![expires_at, v.id],
    )?;
    rebuild_live(&tx, ns)?;
    tx.commit()?;
    Ok(())
}

/// Physically delete a stasis version: drop its manifest + version row and
/// reclaim every content blob it referenced that no other retained version
/// or the live manifest still uses. The **only** path that removes bytes.
/// Returns the number of blobs reclaimed.
pub fn purge(
    db: &mut Db,
    backend: &dyn Backend,
    ns: &str,
    selector: &str,
) -> Result<usize, VecdError> {
    let Some(v) = crate::session::find_version(db, ns, selector)? else {
        return Err(VecdError::usage(format!("no version '{selector}' in namespace '{ns}'")));
    };
    if v.state != "stasis" {
        return Err(VecdError::usage(format!(
            "version '{}' is {} — only stasis versions can be purged (expire it first)",
            v.tag, v.state
        )));
    }
    let objects = crate::session::version_manifest(db, v.id)?; // (key, content_key, size)

    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    // Drop the version first so reference checks don't see its own rows.
    tx.execute("DELETE FROM version_objects WHERE version_id=?1", params![v.id])?;
    tx.execute("DELETE FROM versions WHERE id=?1", params![v.id])?;

    let mut orphans: Vec<(String, String)> = Vec::new();
    for (key, ck, _size) in &objects {
        let still_used: bool = tx
            .query_row(
                "SELECT 1 FROM (
                    SELECT content_key FROM version_objects WHERE content_key=?1
                    UNION ALL SELECT content_key FROM objects WHERE content_key=?1
                    UNION ALL SELECT content_key FROM staging_objects WHERE content_key=?1
                 ) LIMIT 1",
                params![ck],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !still_used {
            orphans.push((key.clone(), ck.clone()));
        }
    }
    tx.commit()?;

    // Reclaim bytes outside the txn (best-effort).
    let mut reclaimed = 0;
    for (key, ck) in orphans {
        if backend.delete(&crate::store::physical(&key, &ck)).is_ok() {
            reclaimed += 1;
        }
    }
    Ok(reclaimed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authz::Snapshot;
    use crate::db::{BackendRow, ControlPlane, NamespaceRow};
    use crate::namespace::{self, Resolved};
    use crate::{session, store};

    fn fixture() -> (tempfile::TempDir, Db, Snapshot) {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::init(&dir.path().join("vecd.db")).unwrap();
        let mut cp = ControlPlane::default();
        cp.backends.push(BackendRow {
            name: "b".into(),
            kind: "mem".into(),
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
            quota_bytes: 1 << 40,
            ttl_seconds: None,
        });
        (dir, db, Snapshot::build(&cp))
    }
    fn r(snap: &Snapshot, p: &str) -> Resolved {
        namespace::resolve(snap, p).unwrap()
    }

    /// Commit a version, then force its expiry into the past.
    fn commit_expired(db: &mut Db, snap: &Snapshot, key: &str, data: &[u8]) -> i64 {
        session::open(db, "ds", None, &[]).unwrap();
        session::stage_put_bytes(db, &r(snap, &format!("ds/{key}")), data).unwrap();
        let info = session::commit(db, "ds", None).unwrap();
        db.conn()
            .execute("UPDATE versions SET expires_at=1 WHERE seq=?1 AND namespace_path='ds'", params![info.seq])
            .unwrap();
        info.seq
    }

    #[test]
    fn sweep_moves_expired_to_stasis_and_rolls_back() {
        let (_d, mut db, snap) = fixture();
        // v1 with content "one", expired.
        commit_expired(&mut db, &snap, "f", b"one");
        // v2 with content "two", NOT expired.
        session::open(&mut db, "ds", None, &[]).unwrap();
        session::stage_put_bytes(&mut db, &r(&snap, "ds/f"), b"two").unwrap();
        session::commit(&mut db, "ds", None).unwrap();

        // Sweep expires v1 only; live manifest still reflects v2.
        assert_eq!(sweep(&mut db).unwrap(), 1);
        assert_eq!(store::get(&db, &r(&snap, "ds/f")).unwrap().unwrap().0, b"two");
        // v1 is now in the cleanup queue.
        let q = pending(&db).unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].tag, "v1");
    }

    #[test]
    fn sweep_of_only_version_clears_live_and_marks_ns_stasis() {
        let (_d, mut db, snap) = fixture();
        commit_expired(&mut db, &snap, "f", b"only");
        assert_eq!(sweep(&mut db).unwrap(), 1);
        assert!(store::get(&db, &r(&snap, "ds/f")).unwrap().is_none());
        assert!(ns_in_stasis(&db, "ds").unwrap());
    }

    #[test]
    fn extend_restores_a_stasis_version() {
        let (_d, mut db, snap) = fixture();
        commit_expired(&mut db, &snap, "f", b"data");
        sweep(&mut db).unwrap();
        assert!(ns_in_stasis(&db, "ds").unwrap());
        extend(&mut db, "ds", "v1", Some(3600)).unwrap();
        assert!(!ns_in_stasis(&db, "ds").unwrap());
        assert_eq!(store::get(&db, &r(&snap, "ds/f")).unwrap().unwrap().0, b"data");
    }

    #[test]
    fn purge_reclaims_unreferenced_blobs_only() {
        let (_d, mut db, snap) = fixture();
        // v1 has unique content; v2 shares a key's content with nothing.
        commit_expired(&mut db, &snap, "f", b"v1only");
        session::open(&mut db, "ds", None, &[]).unwrap();
        session::stage_put_bytes(&mut db, &r(&snap, "ds/f"), b"v2content").unwrap();
        session::commit(&mut db, "ds", None).unwrap();
        sweep(&mut db).unwrap(); // v1 → stasis

        let backend = crate::backend::open(&BackendRow {
            name: "b".into(),
            kind: "mem".into(),
            endpoint: format!("mem:{}", _d.path().display()),
            endpoint_url: None,
            region: None,
            creds_ref: None,
            active: true,
        })
        .unwrap();
        // Purging v1 reclaims its unique blob; v2's content is untouched.
        let reclaimed = purge(&mut db, backend.as_ref(), "ds", "v1").unwrap();
        assert_eq!(reclaimed, 1);
        assert!(pending(&db).unwrap().is_empty());
        assert_eq!(store::get(&db, &r(&snap, "ds/f")).unwrap().unwrap().0, b"v2content");
        // Purging a non-stasis version is refused.
        assert!(purge(&mut db, backend.as_ref(), "ds", "v2").is_err());
    }
}
