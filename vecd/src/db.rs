// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The SQLite control plane and CAS authority.
//!
//! `vecd` keeps *who-can-do-what* and *what-objects-exist* in one SQLite
//! database opened in **WAL mode** with a `busy_timeout` and
//! `foreign_keys=ON`, so the daemon and the local admin CLI share it
//! safely with no lock files or bespoke IPC (see `docs/design/
//! vecd-daemon.md` § *Concurrency*).
//!
//! Every **control-plane write** bumps `meta.auth_generation` inside its
//! own transaction; the daemon watches that (gated by the cheap built-in
//! `PRAGMA data_version`) to live-reload its in-memory snapshot.
//! Best-effort writes (`access_log`, `tokens.last_used`) deliberately do
//! *not* bump it, so they never trigger a reload storm.
//!
//! This is the **CAS authority**: object existence and each object's
//! content-key live in `objects`, and `If-Match`/`If-None-Match` are
//! enforced against that row inside the write transaction — so the
//! single-provenance guarantee holds over any (plain byte-store) backend.
//!
//! Phase 1 stores objects as a flat per-namespace manifest (`objects`);
//! the COW *version* tree (`versions`/`version_objects`, tags, `@latest`)
//! from the design's *Upload sessions* section is a Phase 2 refinement.

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::model::{DEFAULT_QUOTA_BYTES, VecdError};

/// The schema version this binary writes. Bumped when the schema changes
/// in a way that needs a migration.
pub const SCHEMA_VERSION: i64 = 2;

/// A handle to the control-plane database.
pub struct Db {
    conn: Connection,
}

impl Db {
    /// Open an existing database with the standard PRAGMAs applied. Fails
    /// if the file does not exist (use [`Db::init`] first).
    pub fn open(path: &Path) -> Result<Self, VecdError> {
        if !path.exists() {
            return Err(VecdError::usage(format!(
                "no vecd database at {} — run `vecd init` first",
                path.display()
            )));
        }
        let conn = Connection::open(path)?;
        Self::apply_pragmas(&conn)?;
        let db = Db { conn };
        db.check_schema()?;
        Ok(db)
    }

    /// Open read-only (for completions / inspection); a no-op-safe path
    /// that never creates or migrates.
    pub fn open_readonly(path: &Path) -> Result<Self, VecdError> {
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
        Ok(Db { conn })
    }

    /// Create the database and schema if absent; idempotent.
    pub fn init(path: &Path) -> Result<Self, VecdError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Self::apply_pragmas(&conn)?;
        let db = Db { conn };
        db.create_schema()?;
        Ok(db)
    }

    fn apply_pragmas(conn: &Connection) -> Result<(), VecdError> {
        // SQLCipher key, when built with the `sqlcipher` feature and a key
        // is provided via the environment (set from the config file by the
        // caller). Applied before any other access, per SQLCipher rules.
        #[cfg(feature = "sqlcipher")]
        if let Ok(key) = std::env::var("VECD_DB_KEY") {
            if !key.is_empty() {
                conn.pragma_update(None, "key", &key)?;
            }
        }
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        Ok(())
    }

    fn check_schema(&self) -> Result<(), VecdError> {
        let v: Option<i64> = self
            .conn
            .query_row("SELECT value FROM meta WHERE key='schema_version'", [], |r| {
                let s: String = r.get(0)?;
                Ok(s.parse::<i64>().unwrap_or(-1))
            })
            .optional()
            .ok()
            .flatten();
        match v {
            Some(v) if v == SCHEMA_VERSION => Ok(()),
            Some(v) => Err(VecdError::op(format!(
                "vecd database schema version {v} != supported {SCHEMA_VERSION}"
            ))),
            None => Err(VecdError::op("vecd database is missing its schema; re-run `vecd init`")),
        }
    }

    /// Borrow the underlying connection (for backup / advanced ops).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Mutable connection access — needed to open a transaction for the
    /// object-store CAS path (which deliberately does *not* bump
    /// `auth_generation`).
    pub fn conn_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    // ── live-reload signals ─────────────────────────────────────────

    /// The built-in `PRAGMA data_version` — changes whenever *another*
    /// connection commits. The cheap gate the daemon polls.
    pub fn data_version(&self) -> Result<i64, VecdError> {
        Ok(self.conn.query_row("PRAGMA data_version", [], |r| r.get(0))?)
    }

    /// The monotonic control-plane generation; advances on every authz /
    /// namespace / backend / quota / profile change.
    pub fn auth_generation(&self) -> Result<i64, VecdError> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key='auth_generation'", [], |r| {
                let s: String = r.get(0)?;
                Ok(s.parse::<i64>().unwrap_or(0))
            })
            .optional()?
            .unwrap_or(0))
    }

    /// Bump `auth_generation` — call inside any control-plane mutation's
    /// transaction so the daemon reloads.
    fn bump_generation(tx: &rusqlite::Transaction) -> Result<(), VecdError> {
        tx.execute(
            "INSERT INTO meta(key,value) VALUES('auth_generation','1')
             ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER)+1 AS TEXT)",
            [],
        )?;
        Ok(())
    }

    /// Run `f` inside a transaction that bumps `auth_generation` on commit.
    /// Use for every control-plane mutation.
    pub fn with_cp_txn<T>(
        &mut self,
        f: impl FnOnce(&rusqlite::Transaction) -> Result<T, VecdError>,
    ) -> Result<T, VecdError> {
        let tx = self.conn.transaction()?;
        let out = f(&tx)?;
        Db::bump_generation(&tx)?;
        tx.commit()?;
        Ok(out)
    }

    // ── schema ──────────────────────────────────────────────────────

    fn create_schema(&self) -> Result<(), VecdError> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        // Seed meta + built-in roles. Idempotent (INSERT OR IGNORE).
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES('auth_generation','0')",
            [],
        )?;
        for (name, actions) in [
            ("read", "read"),
            ("publish", "read,write"),
            ("maintain", "read,write,delete"),
            ("curate", "read,write,delete,admin"),
        ] {
            self.conn.execute(
                "INSERT OR IGNORE INTO roles(name,actions,builtin) VALUES(?1,?2,1)",
                params![name, actions],
            )?;
        }
        // The root namespace always exists, owned by @superuser, config-only.
        self.conn.execute(
            "INSERT OR IGNORE INTO namespaces(path,owner,backend_config,active,listable,quota_bytes,ttl_seconds,created)
             VALUES('', '@superuser', NULL, 0, 'grantees', ?1, NULL, strftime('%s','now'))",
            params![DEFAULT_QUOTA_BYTES as i64],
        )?;
        Ok(())
    }
}

/// The full control-plane + object-store schema. Phase 1 omits the COW
/// `versions`/`version_objects` tables (see module docs); `objects` is the
/// flat per-namespace CAS manifest.
const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    name          TEXT UNIQUE NOT NULL,
    created       INTEGER NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0,
    level         TEXT NOT NULL DEFAULT 'user',
    password_hash TEXT
);

CREATE TABLE IF NOT EXISTS tokens (
    id          INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    profile     TEXT,                       -- JSON [(class,scope)]; NULL = full issuer authority
    created     INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,           -- epoch seconds; MANDATORY
    last_used   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tokens_hash ON tokens(token_hash);

CREATE TABLE IF NOT EXISTS roles (
    name    TEXT PRIMARY KEY,
    actions TEXT NOT NULL,                  -- csv subset of read,write,delete,admin
    builtin INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS profiles (
    name    TEXT PRIMARY KEY,
    owner   TEXT NOT NULL,
    spec    TEXT NOT NULL,                  -- (class,scope-with-{placeholders}) template
    created INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS system_privileges (
    principal  TEXT NOT NULL,
    privilege  TEXT NOT NULL,
    granted_by TEXT,
    created    INTEGER NOT NULL,
    PRIMARY KEY(principal, privilege)
);

CREATE TABLE IF NOT EXISTS backends (
    name         TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,             -- local|s3|mem
    endpoint     TEXT NOT NULL,             -- local:DIR | s3://bucket/prefix | mem:<id>
    endpoint_url TEXT,
    region       TEXT,
    creds_ref    TEXT,
    active       INTEGER NOT NULL DEFAULT 0,
    created      INTEGER NOT NULL
);
-- one endpoint, one active config
CREATE UNIQUE INDEX IF NOT EXISTS idx_backends_active_endpoint
    ON backends(endpoint) WHERE active = 1;

CREATE TABLE IF NOT EXISTS namespaces (
    path           TEXT PRIMARY KEY,        -- '' is the root
    owner          TEXT NOT NULL,           -- user name or system role (@admin)
    backend_config TEXT REFERENCES backends(name),
    active         INTEGER NOT NULL DEFAULT 0,
    listable       TEXT NOT NULL DEFAULT 'grantees',
    quota_bytes    INTEGER NOT NULL,
    ttl_seconds    INTEGER,
    created        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS role_bindings (
    id             INTEGER PRIMARY KEY,
    principal      TEXT NOT NULL,           -- user name | 'PUBLIC' | 'KNOWN'
    role           TEXT NOT NULL REFERENCES roles(name),
    namespace_path TEXT NOT NULL,
    created_by     TEXT,
    created        INTEGER NOT NULL,
    UNIQUE(principal, role, namespace_path)
);
CREATE INDEX IF NOT EXISTS idx_bindings_ns ON role_bindings(namespace_path);

-- The *live* current manifest: what `@latest` serves and what non-session
-- (lone) writes mutate in place. content_key is the descriptive-metadata
-- content-key (see store::content_key), which is also the ETag.
CREATE TABLE IF NOT EXISTS objects (
    namespace_path TEXT NOT NULL,
    key            TEXT NOT NULL,           -- namespace-relative logical key
    content_key    TEXT NOT NULL,           -- descriptive-metadata content-key / ETag
    size           INTEGER NOT NULL,
    created        INTEGER NOT NULL,
    PRIMARY KEY(namespace_path, key)
);

-- Immutable, named version snapshots of a namespace's manifest (the COW
-- tree). A version is committed by an atomic pointer flip at push
-- `complete`; `state` moves committed → stasis on expiry (never deleted
-- until `cleanup purge`).
CREATE TABLE IF NOT EXISTS versions (
    id             INTEGER PRIMARY KEY,
    namespace_path TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    tag            TEXT NOT NULL,           -- user label, default 'v<seq>'
    manifest_hash  TEXT NOT NULL,           -- sha256 over the sorted key→content_key manifest
    state          TEXT NOT NULL,           -- committed | stasis
    created        INTEGER NOT NULL,
    committed_at   INTEGER,
    expires_at     INTEGER,                 -- committed_at + ns ttl, or NULL
    stasis_at      INTEGER,
    UNIQUE(namespace_path, seq),
    UNIQUE(namespace_path, tag)
);
CREATE INDEX IF NOT EXISTS idx_versions_ns ON versions(namespace_path, seq);

-- A committed version's frozen manifest: logical key → content_key.
CREATE TABLE IF NOT EXISTS version_objects (
    version_id  INTEGER NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    content_key TEXT NOT NULL,
    size        INTEGER NOT NULL,
    PRIMARY KEY(version_id, key)
);

-- At most one open upload session per namespace (the push begin→complete
-- bracket). Its presence means content PUTs stage instead of mutating the
-- live manifest.
CREATE TABLE IF NOT EXISTS sessions (
    namespace_path TEXT PRIMARY KEY,
    opened_at      INTEGER NOT NULL,
    actor          TEXT
);

-- Staged manifest entries for an open session — invisible to readers until
-- the session commits (this is what closes push's "in-flux window").
CREATE TABLE IF NOT EXISTS staging_objects (
    namespace_path TEXT NOT NULL,
    key            TEXT NOT NULL,
    content_key    TEXT NOT NULL,
    size           INTEGER NOT NULL,
    PRIMARY KEY(namespace_path, key)
);

CREATE TABLE IF NOT EXISTS access_log (
    id          INTEGER PRIMARY KEY,
    ts          INTEGER NOT NULL,
    principal   TEXT,
    token_desc  TEXT,
    action      TEXT,
    key         TEXT,
    status      INTEGER,
    bytes       INTEGER,
    remote_addr TEXT
);
CREATE INDEX IF NOT EXISTS idx_access_ts ON access_log(ts);
"#;

// ── row types (raw control-plane rows the snapshot is built from) ───

#[derive(Clone, Debug)]
pub struct UserRow {
    pub id: i64,
    pub name: String,
    pub disabled: bool,
    pub level: String,
    pub password_hash: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TokenRow {
    pub id: i64,
    pub user_id: i64,
    pub token_hash: String,
    pub description: String,
    pub profile: Option<String>,
    pub expires_at: i64,
}

#[derive(Clone, Debug)]
pub struct RoleRow {
    pub name: String,
    pub actions: String,
    pub builtin: bool,
}

#[derive(Clone, Debug)]
pub struct BindingRow {
    pub principal: String,
    pub role: String,
    pub namespace_path: String,
}

#[derive(Clone, Debug)]
pub struct BackendRow {
    pub name: String,
    pub kind: String,
    pub endpoint: String,
    pub endpoint_url: Option<String>,
    pub region: Option<String>,
    pub creds_ref: Option<String>,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub struct NamespaceRow {
    pub path: String,
    pub owner: String,
    pub backend_config: Option<String>,
    pub active: bool,
    pub listable: String,
    pub quota_bytes: i64,
    pub ttl_seconds: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct ProfileRow {
    pub name: String,
    pub owner: String,
    pub spec: String,
}

#[derive(Clone, Debug)]
pub struct SystemPrivRow {
    pub principal: String,
    pub privilege: String,
}

/// The complete control-plane state, read in one consistent pass. The
/// daemon turns this into an indexed [`crate::authz::Snapshot`]; the admin
/// CLI reads pieces of it for listing.
#[derive(Clone, Debug, Default)]
pub struct ControlPlane {
    pub users: Vec<UserRow>,
    pub tokens: Vec<TokenRow>,
    pub roles: Vec<RoleRow>,
    pub bindings: Vec<BindingRow>,
    pub backends: Vec<BackendRow>,
    pub namespaces: Vec<NamespaceRow>,
    pub profiles: Vec<ProfileRow>,
    pub system_privileges: Vec<SystemPrivRow>,
    pub generation: i64,
}

impl Db {
    /// Read the entire control plane in one pass (used to build the
    /// daemon's reloadable snapshot).
    pub fn load_control_plane(&self) -> Result<ControlPlane, VecdError> {
        let users = self
            .conn
            .prepare("SELECT id,name,disabled,level,password_hash FROM users")?
            .query_map([], |r| {
                Ok(UserRow {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    disabled: r.get::<_, i64>(2)? != 0,
                    level: r.get(3)?,
                    password_hash: r.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let tokens = self
            .conn
            .prepare("SELECT id,user_id,token_hash,description,profile,expires_at FROM tokens")?
            .query_map([], |r| {
                Ok(TokenRow {
                    id: r.get(0)?,
                    user_id: r.get(1)?,
                    token_hash: r.get(2)?,
                    description: r.get(3)?,
                    profile: r.get(4)?,
                    expires_at: r.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let roles = self
            .conn
            .prepare("SELECT name,actions,builtin FROM roles")?
            .query_map([], |r| {
                Ok(RoleRow { name: r.get(0)?, actions: r.get(1)?, builtin: r.get::<_, i64>(2)? != 0 })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let bindings = self
            .conn
            .prepare("SELECT principal,role,namespace_path FROM role_bindings")?
            .query_map([], |r| {
                Ok(BindingRow {
                    principal: r.get(0)?,
                    role: r.get(1)?,
                    namespace_path: r.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let backends = self
            .conn
            .prepare("SELECT name,kind,endpoint,endpoint_url,region,creds_ref,active FROM backends")?
            .query_map([], |r| {
                Ok(BackendRow {
                    name: r.get(0)?,
                    kind: r.get(1)?,
                    endpoint: r.get(2)?,
                    endpoint_url: r.get(3)?,
                    region: r.get(4)?,
                    creds_ref: r.get(5)?,
                    active: r.get::<_, i64>(6)? != 0,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let namespaces = self
            .conn
            .prepare(
                "SELECT path,owner,backend_config,active,listable,quota_bytes,ttl_seconds FROM namespaces",
            )?
            .query_map([], |r| {
                Ok(NamespaceRow {
                    path: r.get(0)?,
                    owner: r.get(1)?,
                    backend_config: r.get(2)?,
                    active: r.get::<_, i64>(3)? != 0,
                    listable: r.get(4)?,
                    quota_bytes: r.get(5)?,
                    ttl_seconds: r.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let profiles = self
            .conn
            .prepare("SELECT name,owner,spec FROM profiles")?
            .query_map([], |r| {
                Ok(ProfileRow { name: r.get(0)?, owner: r.get(1)?, spec: r.get(2)? })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let system_privileges = self
            .conn
            .prepare("SELECT principal,privilege FROM system_privileges")?
            .query_map([], |r| {
                Ok(SystemPrivRow { principal: r.get(0)?, privilege: r.get(1)? })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let generation = self.auth_generation()?;

        Ok(ControlPlane {
            users,
            tokens,
            roles,
            bindings,
            backends,
            namespaces,
            profiles,
            system_privileges,
            generation,
        })
    }

    /// Append an access-log row. Best-effort: does *not* bump
    /// `auth_generation` (so it never triggers a reload).
    pub fn log_access(
        &self,
        principal: Option<&str>,
        token_desc: Option<&str>,
        action: &str,
        key: &str,
        status: u16,
        bytes: u64,
        remote_addr: &str,
    ) {
        let _ = self.conn.execute(
            "INSERT INTO access_log(ts,principal,token_desc,action,key,status,bytes,remote_addr)
             VALUES(strftime('%s','now'),?1,?2,?3,?4,?5,?6,?7)",
            params![principal, token_desc, action, key, status as i64, bytes as i64, remote_addr],
        );
    }

    /// Verify a password grant: returns the user id if the user exists, is
    /// enabled, has a password set, and `presented_hash` matches.
    pub fn password_ok(&self, user: &str, presented_hash: &str) -> Result<Option<i64>, VecdError> {
        let row: Option<(i64, Option<String>, bool)> = self
            .conn
            .query_row(
                "SELECT id, password_hash, disabled FROM users WHERE name=?1",
                params![user],
                |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? != 0)),
            )
            .optional()?;
        Ok(match row {
            Some((id, Some(hash), false)) if hash == presented_hash => Some(id),
            _ => None,
        })
    }

    /// The issuing user id of a token (for revocation ownership checks).
    pub fn token_owner(&self, token_id: i64) -> Result<Option<i64>, VecdError> {
        Ok(self
            .conn
            .query_row("SELECT user_id FROM tokens WHERE id=?1", params![token_id], |r| r.get(0))
            .optional()?)
    }

    /// A user's id by name.
    pub fn user_id(&self, name: &str) -> Result<Option<i64>, VecdError> {
        Ok(self
            .conn
            .query_row("SELECT id FROM users WHERE name=?1", params![name], |r| r.get(0))
            .optional()?)
    }

    /// Best-effort `last_used` stamp on a token (no generation bump).
    pub fn touch_token(&self, token_id: i64) {
        let _ = self.conn.execute(
            "UPDATE tokens SET last_used = strftime('%s','now') WHERE id = ?1",
            params![token_id],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_db() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::init(&dir.path().join("vecd.db")).unwrap();
        (dir, db)
    }

    #[test]
    fn init_seeds_roles_and_root() {
        let (_d, db) = tmp_db();
        let cp = db.load_control_plane().unwrap();
        assert_eq!(cp.roles.len(), 4);
        assert!(cp.roles.iter().all(|r| r.builtin));
        assert!(cp.namespaces.iter().any(|n| n.path.is_empty() && n.owner == "@superuser"));
        assert_eq!(cp.generation, 0);
    }

    #[test]
    fn cp_txn_bumps_generation() {
        let (_d, mut db) = tmp_db();
        assert_eq!(db.auth_generation().unwrap(), 0);
        db.with_cp_txn(|tx| {
            tx.execute(
                "INSERT INTO users(id,name,created,level) VALUES(1,'alice',strftime('%s','now'),'user')",
                [],
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(db.auth_generation().unwrap(), 1);
    }

    #[test]
    fn reopen_checks_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecd.db");
        Db::init(&path).unwrap();
        // Reopening a healthy DB succeeds.
        assert!(Db::open(&path).is_ok());
        // Opening a nonexistent DB is a usage error, not a panic.
        assert!(matches!(Db::open(&dir.path().join("missing.db")), Err(VecdError::Usage(_))));
    }
}
