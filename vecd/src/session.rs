// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Transactional upload sessions and the copy-on-write version tree.
//!
//! A session is the push `begin`→`complete` bracket. While it is open, an
//! object PUT **stages** into `staging_objects` (invisible to readers, who
//! resolve the live `objects` manifest) instead of mutating the live
//! manifest — this is what closes push's "in-flux window". At `complete`
//! the session **commits in one transaction**: the staged manifest
//! atomically *replaces* the live manifest **and** is snapshotted as a new
//! immutable [`versions`] row (the pointer flip). An `abort` discards the
//! staging with no effect on readers.
//!
//! The staging manifest is **copy-on-write initialised** from the current
//! live manifest at `begin`, so unchanged keys are inherited and a push
//! that re-uploads only changed files still publishes a complete dataset.
//! The `begin` event's declared `deletes` are applied to the staging
//! manifest, so orphan removals are a versioned manifest omission rather
//! than a physical erase (bytes are reclaimed only by `cleanup purge`).
//!
//! All of this is object-store state (not control plane), so it uses plain
//! transactions and never bumps `auth_generation`.

use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::db::Db;
use crate::model::VecdError;
use crate::namespace::Resolved;
use crate::store;

/// What a commit produced.
#[derive(Clone, Debug)]
pub struct Committed {
    pub seq: i64,
    pub tag: String,
    pub manifest_hash: String,
    pub expires_at: Option<i64>,
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Is an upload session currently open for this namespace?
pub fn is_open(db: &Db, ns: &str) -> Result<bool, VecdError> {
    Ok(db
        .conn()
        .query_row("SELECT 1 FROM sessions WHERE namespace_path=?1", params![ns], |_| Ok(()))
        .optional()?
        .is_some())
}

/// Open a fresh session: discard any prior staging, COW-initialise the
/// staging manifest from the live manifest, apply the begin event's
/// `deletes`, and mark the session open.
pub fn open(db: &mut Db, ns: &str, actor: Option<&str>, deletes: &[String]) -> Result<(), VecdError> {
    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute("DELETE FROM staging_objects WHERE namespace_path=?1", params![ns])?;
    tx.execute("DELETE FROM sessions WHERE namespace_path=?1", params![ns])?;
    // COW init: copy the live manifest into staging.
    tx.execute(
        "INSERT INTO staging_objects(namespace_path,key,content_key,size)
         SELECT namespace_path,key,content_key,size FROM objects WHERE namespace_path=?1",
        params![ns],
    )?;
    for key in deletes {
        tx.execute(
            "DELETE FROM staging_objects WHERE namespace_path=?1 AND key=?2",
            params![ns, key],
        )?;
    }
    tx.execute(
        "INSERT INTO sessions(namespace_path,opened_at,actor) VALUES(?1,?2,?3)",
        params![ns, now(), actor],
    )?;
    tx.commit()?;
    Ok(())
}

/// Stage `data` under the resolved key (write bytes + record in the
/// staging manifest). Returns the content-key (ETag). Unconditional — the
/// only conditional writes push makes are on the pushlog, handled by the
/// caller against the committed manifest.
pub fn stage_put_bytes(db: &mut Db, r: &Resolved, data: &[u8]) -> Result<String, VecdError> {
    let (ck, size) = store::write_blob(r, data)?;
    db.conn().execute(
        "INSERT INTO staging_objects(namespace_path,key,content_key,size) VALUES(?1,?2,?3,?4)
         ON CONFLICT(namespace_path,key) DO UPDATE SET content_key=excluded.content_key, size=excluded.size",
        params![r.storage_ns, r.rel_key, ck, size as i64],
    )?;
    Ok(ck)
}

/// Remove a key from the staging manifest.
pub fn stage_delete(db: &mut Db, ns: &str, rel_key: &str) -> Result<(), VecdError> {
    db.conn().execute(
        "DELETE FROM staging_objects WHERE namespace_path=?1 AND key=?2",
        params![ns, rel_key],
    )?;
    Ok(())
}

/// Discard an open session's staging with no effect on the live manifest.
pub fn abort(db: &mut Db, ns: &str) -> Result<(), VecdError> {
    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute("DELETE FROM staging_objects WHERE namespace_path=?1", params![ns])?;
    tx.execute("DELETE FROM sessions WHERE namespace_path=?1", params![ns])?;
    tx.commit()?;
    Ok(())
}

/// Commit the open session: atomically replace the live manifest with the
/// staged manifest and snapshot it as a new committed version (the pointer
/// flip). `ttl_seconds` (the namespace default TTL) sets `expires_at`.
pub fn commit(db: &mut Db, ns: &str, ttl_seconds: Option<i64>) -> Result<Committed, VecdError> {
    let ts = now();
    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

    // The staged manifest, sorted — the basis for the manifest hash and the
    // new live + snapshot rows.
    let staged: Vec<(String, String, i64)> = tx
        .prepare("SELECT key,content_key,size FROM staging_objects WHERE namespace_path=?1 ORDER BY key")?
        .query_map(params![ns], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<Vec<_>, _>>()?;

    let manifest_hash = manifest_hash(&staged);
    let seq: i64 = tx.query_row(
        "SELECT COALESCE(MAX(seq),0)+1 FROM versions WHERE namespace_path=?1",
        params![ns],
        |r| r.get(0),
    )?;
    let tag = format!("v{seq}");
    let expires_at = ttl_seconds.map(|t| ts + t);

    // Replace the live manifest with the staged one.
    tx.execute("DELETE FROM objects WHERE namespace_path=?1", params![ns])?;
    for (key, ck, size) in &staged {
        tx.execute(
            "INSERT INTO objects(namespace_path,key,content_key,size,created) VALUES(?1,?2,?3,?4,?5)",
            params![ns, key, ck, size, ts],
        )?;
    }

    // Snapshot the immutable version.
    tx.execute(
        "INSERT INTO versions(namespace_path,seq,tag,manifest_hash,state,created,committed_at,expires_at)
         VALUES(?1,?2,?3,?4,'committed',?5,?5,?6)",
        params![ns, seq, tag, manifest_hash, ts, expires_at],
    )?;
    let version_id = tx.last_insert_rowid();
    for (key, ck, size) in &staged {
        tx.execute(
            "INSERT INTO version_objects(version_id,key,content_key,size) VALUES(?1,?2,?3,?4)",
            params![version_id, key, ck, size],
        )?;
    }

    // Clear the session.
    tx.execute("DELETE FROM staging_objects WHERE namespace_path=?1", params![ns])?;
    tx.execute("DELETE FROM sessions WHERE namespace_path=?1", params![ns])?;
    tx.commit()?;

    Ok(Committed { seq, tag, manifest_hash, expires_at })
}

/// A committed (or stasis) version's metadata.
#[derive(Clone, Debug)]
pub struct VersionRow {
    pub id: i64,
    pub seq: i64,
    pub tag: String,
    pub manifest_hash: String,
    pub state: String,
    pub committed_at: Option<i64>,
    pub expires_at: Option<i64>,
    pub stasis_at: Option<i64>,
}

fn map_version(r: &rusqlite::Row) -> rusqlite::Result<VersionRow> {
    Ok(VersionRow {
        id: r.get(0)?,
        seq: r.get(1)?,
        tag: r.get(2)?,
        manifest_hash: r.get(3)?,
        state: r.get(4)?,
        committed_at: r.get(5)?,
        expires_at: r.get(6)?,
        stasis_at: r.get(7)?,
    })
}

const VERSION_COLS: &str =
    "id,seq,tag,manifest_hash,state,committed_at,expires_at,stasis_at";

/// The current (latest committed, non-stasis) version of a namespace.
pub fn current_version(db: &Db, ns: &str) -> Result<Option<VersionRow>, VecdError> {
    Ok(db
        .conn()
        .query_row(
            &format!(
                "SELECT {VERSION_COLS} FROM versions
                 WHERE namespace_path=?1 AND state='committed'
                 ORDER BY seq DESC LIMIT 1"
            ),
            params![ns],
            map_version,
        )
        .optional()?)
}

/// Resolve a version selector: `latest`, `v<seq>`, a tag, or a
/// manifest-hash prefix.
pub fn find_version(db: &Db, ns: &str, selector: &str) -> Result<Option<VersionRow>, VecdError> {
    if selector == "latest" {
        return current_version(db, ns);
    }
    // Exact tag (which includes the default `v<seq>`).
    if let Some(v) = db
        .conn()
        .query_row(
            &format!("SELECT {VERSION_COLS} FROM versions WHERE namespace_path=?1 AND tag=?2"),
            params![ns, selector],
            map_version,
        )
        .optional()?
    {
        return Ok(Some(v));
    }
    // `v<seq>` numeric form.
    if let Some(seq) = selector.strip_prefix('v').and_then(|s| s.parse::<i64>().ok()) {
        if let Some(v) = db
            .conn()
            .query_row(
                &format!("SELECT {VERSION_COLS} FROM versions WHERE namespace_path=?1 AND seq=?2"),
                params![ns, seq],
                map_version,
            )
            .optional()?
        {
            return Ok(Some(v));
        }
    }
    // manifest-hash prefix.
    Ok(db
        .conn()
        .query_row(
            &format!(
                "SELECT {VERSION_COLS} FROM versions
                 WHERE namespace_path=?1 AND manifest_hash LIKE ?2||'%'
                 ORDER BY seq DESC LIMIT 1"
            ),
            params![ns, selector],
            map_version,
        )
        .optional()?)
}

/// All versions of a namespace, newest first.
pub fn list_versions(db: &Db, ns: &str) -> Result<Vec<VersionRow>, VecdError> {
    Ok(db
        .conn()
        .prepare(&format!(
            "SELECT {VERSION_COLS} FROM versions WHERE namespace_path=?1 ORDER BY seq DESC"
        ))?
        .query_map(params![ns], map_version)?
        .collect::<Result<Vec<_>, _>>()?)
}

/// A version's `(content_key, size)` for one key.
pub fn version_object(db: &Db, version_id: i64, key: &str) -> Result<Option<(String, u64)>, VecdError> {
    Ok(db
        .conn()
        .query_row(
            "SELECT content_key,size FROM version_objects WHERE version_id=?1 AND key=?2",
            params![version_id, key],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as u64)),
        )
        .optional()?)
}

/// A version's full manifest: `(key, content_key, size)`, sorted by key.
pub fn version_manifest(db: &Db, version_id: i64) -> Result<Vec<(String, String, u64)>, VecdError> {
    Ok(db
        .conn()
        .prepare("SELECT key,content_key,size FROM version_objects WHERE version_id=?1 ORDER BY key")?
        .query_map(params![version_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)? as u64))
        })?
        .collect::<Result<Vec<_>, _>>()?)
}

/// sha256 over the canonical sorted `key\0content_key\n` manifest — a
/// *metadata* hash (per the terminology invariant), citable as the
/// version's integrity check.
fn manifest_hash(sorted: &[(String, String, i64)]) -> String {
    let mut h = Sha256::new();
    for (key, ck, _size) in sorted {
        h.update(key.as_bytes());
        h.update(b"\0");
        h.update(ck.as_bytes());
        h.update(b"\n");
    }
    hex::encode(h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authz::Snapshot;
    use crate::db::{BackendRow, ControlPlane, NamespaceRow};
    use crate::namespace;
    use crate::store::{self, Precondition};

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

    #[test]
    fn session_isolates_then_commits() {
        let (_d, mut db, snap) = fixture();
        // A committed baseline via a lone write.
        store::put(&mut db, &r(&snap, "ds/a"), b"v1a", &Precondition::None, 1 << 40, false).unwrap();

        // Open a session, stage a change + a new key.
        session_open(&mut db, "ds", &[]);
        stage_put_bytes(&mut db, &r(&snap, "ds/a"), b"v2a").unwrap();
        stage_put_bytes(&mut db, &r(&snap, "ds/b"), b"newb").unwrap();

        // Readers still see the prior committed manifest (in-flux closed).
        assert_eq!(store::get(&db, &r(&snap, "ds/a")).unwrap().unwrap().0, b"v1a");
        assert!(store::get(&db, &r(&snap, "ds/b")).unwrap().is_none());
        assert!(is_open(&db, "ds").unwrap());

        // Commit → the new manifest is live and snapshotted.
        let info = commit(&mut db, "ds", None).unwrap();
        assert_eq!(info.seq, 1);
        assert_eq!(info.tag, "v1");
        assert_eq!(store::get(&db, &r(&snap, "ds/a")).unwrap().unwrap().0, b"v2a");
        assert_eq!(store::get(&db, &r(&snap, "ds/b")).unwrap().unwrap().0, b"newb");
        assert!(!is_open(&db, "ds").unwrap());
    }

    #[test]
    fn abort_discards_staging() {
        let (_d, mut db, snap) = fixture();
        store::put(&mut db, &r(&snap, "ds/a"), b"keep", &Precondition::None, 1 << 40, false).unwrap();
        session_open(&mut db, "ds", &[]);
        stage_put_bytes(&mut db, &r(&snap, "ds/a"), b"discarded").unwrap();
        abort(&mut db, "ds").unwrap();
        assert!(!is_open(&db, "ds").unwrap());
        assert_eq!(store::get(&db, &r(&snap, "ds/a")).unwrap().unwrap().0, b"keep");
    }

    #[test]
    fn begin_deletes_are_manifest_omissions() {
        let (_d, mut db, snap) = fixture();
        store::put(&mut db, &r(&snap, "ds/a"), b"a", &Precondition::None, 1 << 40, false).unwrap();
        store::put(&mut db, &r(&snap, "ds/old"), b"old", &Precondition::None, 1 << 40, false).unwrap();
        // begin declares 'old' as a delete.
        session_open(&mut db, "ds", &["old".to_string()]);
        stage_put_bytes(&mut db, &r(&snap, "ds/a"), b"a2").unwrap();
        commit(&mut db, "ds", None).unwrap();
        assert_eq!(store::get(&db, &r(&snap, "ds/a")).unwrap().unwrap().0, b"a2");
        assert!(store::get(&db, &r(&snap, "ds/old")).unwrap().is_none());
    }

    #[test]
    fn seq_is_monotonic_per_namespace() {
        let (_d, mut db, _snap) = fixture();
        session_open(&mut db, "ds", &[]);
        assert_eq!(commit(&mut db, "ds", None).unwrap().seq, 1);
        session_open(&mut db, "ds", &[]);
        assert_eq!(commit(&mut db, "ds", None).unwrap().seq, 2);
        session_open(&mut db, "ds", &[]);
        let third = commit(&mut db, "ds", Some(3600)).unwrap();
        assert_eq!(third.seq, 3);
        assert!(third.expires_at.is_some());
    }

    fn session_open(db: &mut Db, ns: &str, deletes: &[String]) {
        open(db, ns, Some("tester"), deletes).unwrap();
    }
}
