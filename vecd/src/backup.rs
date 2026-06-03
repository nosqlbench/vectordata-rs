// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Control-plane DB backup — consistent, no-downtime snapshots of the one
//! piece of state `vecd` alone owns, shipped to a private destination.
//!
//! A snapshot uses SQLite's `VACUUM INTO` (a one-shot live page copy, safe
//! under WAL while the daemon keeps serving) into a temp file, which is
//! then placed at the destination — a local directory (`file://…` or a
//! bare path) or a private S3 prefix (`s3://…`, via the `aws` CLI).
//! Snapshots are timestamped (`vecd-<epoch>.db`); `retain` keeps the most
//! recent N and prunes the rest.
//!
//! When built with the `sqlcipher` feature and a DB key is configured, the
//! DB — and therefore every snapshot — is encrypted by construction, so
//! the bytes at rest are never readable without the key.

use std::path::Path;
use std::process::{Command, Stdio};

use crate::db::Db;
use crate::model::VecdError;

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Whether a destination is an S3 prefix.
fn is_s3(dest: &str) -> bool {
    dest.starts_with("s3://")
}

/// Strip an optional `file://` scheme and normalize to a directory path.
fn local_dir(dest: &str) -> String {
    dest.strip_prefix("file://").unwrap_or(dest).trim_end_matches('/').to_string()
}

/// Write a consistent snapshot of the DB to a fresh temp file; returns its
/// path. `VACUUM INTO` refuses to overwrite, so the target must not exist.
fn vacuum_into_temp(db: &Db) -> Result<std::path::PathBuf, VecdError> {
    let tmp = std::env::temp_dir().join(format!("vecd-snap-{}-{}.db", std::process::id(), now_secs()));
    if tmp.exists() {
        std::fs::remove_file(&tmp)?;
    }
    // VACUUM INTO takes a string literal path; escape single quotes.
    let escaped = tmp.to_string_lossy().replace('\'', "''");
    db.conn()
        .execute_batch(&format!("VACUUM INTO '{escaped}'"))
        .map_err(|e| VecdError::op(format!("VACUUM INTO snapshot failed: {e}")))?;
    Ok(tmp)
}

/// Take a snapshot and place it at `dest` with a timestamped name. Returns
/// the snapshot's full URI/path. Applies `retain` (keep newest N) when set.
pub fn backup_now(db: &Db, dest: &str, retain: Option<usize>) -> Result<String, VecdError> {
    let tmp = vacuum_into_temp(db)?;
    let name = format!("vecd-{}.db", now_secs());

    let placed = if is_s3(dest) {
        let uri = format!("{}/{name}", dest.trim_end_matches('/'));
        let out = Command::new("aws")
            .arg("s3")
            .arg("cp")
            .arg(&tmp)
            .arg(&uri)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| VecdError::op(format!("invoking aws for backup: {e}")))?;
        let _ = std::fs::remove_file(&tmp);
        if !out.status.success() {
            return Err(VecdError::op(format!(
                "uploading snapshot to {uri} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        uri
    } else {
        let dir = local_dir(dest);
        std::fs::create_dir_all(&dir)?;
        let target = Path::new(&dir).join(&name);
        std::fs::rename(&tmp, &target).or_else(|_| {
            // rename across filesystems fails — fall back to copy.
            std::fs::copy(&tmp, &target).map(|_| ()).and_then(|_| std::fs::remove_file(&tmp))
        })?;
        target.to_string_lossy().to_string()
    };

    if let Some(keep) = retain {
        prune(dest, keep)?;
    }
    Ok(placed)
}

/// List snapshot names at the destination, newest first (by the embedded
/// epoch).
pub fn list_backups(dest: &str) -> Result<Vec<String>, VecdError> {
    let mut names: Vec<String> = if is_s3(dest) {
        let out = Command::new("aws")
            .arg("s3")
            .arg("ls")
            .arg(format!("{}/", dest.trim_end_matches('/')))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| VecdError::op(format!("invoking aws: {e}")))?;
        if !out.status.success() {
            return Err(VecdError::op(format!(
                "listing {dest} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.split_whitespace().last())
            .filter(|n| n.starts_with("vecd-") && n.ends_with(".db"))
            .map(|s| s.to_string())
            .collect()
    } else {
        let dir = local_dir(dest);
        match std::fs::read_dir(&dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.starts_with("vecd-") && n.ends_with(".db"))
                .collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.into()),
        }
    };
    names.sort_by_key(|n| std::cmp::Reverse(epoch_of(n)));
    Ok(names)
}

/// Keep the newest `keep` snapshots, removing the rest.
fn prune(dest: &str, keep: usize) -> Result<(), VecdError> {
    let names = list_backups(dest)?;
    for old in names.into_iter().skip(keep) {
        if is_s3(dest) {
            let uri = format!("{}/{old}", dest.trim_end_matches('/'));
            let _ = Command::new("aws").arg("s3").arg("rm").arg(&uri).output();
        } else {
            let _ = std::fs::remove_file(Path::new(&local_dir(dest)).join(&old));
        }
    }
    Ok(())
}

/// Parse the epoch out of a `vecd-<epoch>.db` name (0 if unparseable).
fn epoch_of(name: &str) -> i64 {
    name.strip_prefix("vecd-")
        .and_then(|s| s.strip_suffix(".db"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Install a snapshot as the active DB at `db_path`. Refuses to clobber a
/// DB whose data dir has a live pidfile (stop the daemon first).
pub fn restore(snapshot_uri: &str, db_path: &Path) -> Result<(), VecdError> {
    if let Some(parent) = db_path.parent() {
        if pidfile_is_live(parent) {
            return Err(VecdError::usage(
                "a vecd daemon appears to be running on this data-dir; stop it before restoring",
            ));
        }
        std::fs::create_dir_all(parent)?;
    }
    if is_s3(snapshot_uri) {
        let out = Command::new("aws")
            .arg("s3")
            .arg("cp")
            .arg(snapshot_uri)
            .arg(db_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| VecdError::op(format!("invoking aws: {e}")))?;
        if !out.status.success() {
            return Err(VecdError::op(format!(
                "fetching snapshot {snapshot_uri} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
    } else {
        let src = snapshot_uri.strip_prefix("file://").unwrap_or(snapshot_uri);
        std::fs::copy(src, db_path)?;
    }
    Ok(())
}

fn pidfile_is_live(data_dir: &Path) -> bool {
    let pidfile = data_dir.join("vecd.pid");
    let Ok(text) = std::fs::read_to_string(&pidfile) else { return false };
    let Ok(pid) = text.trim().parse::<i32>() else { return false };
    process_alive(pid)
}

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

#[cfg(unix)]
fn process_alive(pid: i32) -> bool {
    // signal 0 probes existence without delivering a signal.
    unsafe { kill(pid, 0) == 0 }
}

#[cfg(not(unix))]
fn process_alive(_pid: i32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admin;
    use crate::model::Level;

    #[test]
    fn backup_list_restore_roundtrip_local() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("vecd.db");
        let mut db = Db::init(&db_path).unwrap();
        admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();

        let dest = dir.path().join("backups");
        let dest_str = dest.to_string_lossy().to_string();

        let snap1 = backup_now(&db, &dest_str, Some(5)).unwrap();
        assert!(Path::new(&snap1).exists());
        let listed = list_backups(&dest_str).unwrap();
        assert_eq!(listed.len(), 1);

        // Restore into a fresh data dir and confirm the user survived.
        let restore_dir = tempfile::tempdir().unwrap();
        let restored_db = restore_dir.path().join("vecd.db");
        restore(&snap1, &restored_db).unwrap();
        let db2 = Db::open(&restored_db).unwrap();
        assert!(db2.load_control_plane().unwrap().users.iter().any(|u| u.name == "alice"));
    }

    #[test]
    fn retain_prunes_old_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::init(&dir.path().join("vecd.db")).unwrap();
        let dest = dir.path().join("b").to_string_lossy().to_string();
        // Hand-place three snapshots with distinct epochs, then prune to 2.
        std::fs::create_dir_all(&dest).unwrap();
        for epoch in [100, 200, 300] {
            std::fs::write(Path::new(&dest).join(format!("vecd-{epoch}.db")), b"x").unwrap();
        }
        let _ = &db; // keep db alive
        prune(&dest, 2).unwrap();
        let remaining = list_backups(&dest).unwrap();
        assert_eq!(remaining, vec!["vecd-300.db".to_string(), "vecd-200.db".to_string()]);
    }
}
