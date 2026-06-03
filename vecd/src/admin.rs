// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Control-plane administration — the library behind the `vecd` admin
//! subcommands (`users`/`tokens`/`roles`/`backends`/`ns`/`bind`/`priv`/
//! `profiles`). These operate **directly on the SQLite DB**, so holding
//! the DB file *is* the authority (the local admin CLI is
//! superuser-by-filesystem-access, per `docs/design/vecd-daemon.md`
//! § *Privilege levels*) — no management-plane level check is applied
//! here. Every mutation bumps `auth_generation` (via [`Db::with_cp_txn`])
//! so a running daemon live-reloads.
//!
//! Functions are written to be called both from the CLI and from tests,
//! so the integration test sets up an endpoint exactly as an operator
//! would.

use rusqlite::{params, OptionalExtension};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::auth;
use crate::authz::ProfileEntry;
use crate::db::Db;
use crate::model::{
    Class, Level, Listable, VecdError, DEFAULT_TOKEN_TTL_SECS, MAX_TOKEN_TTL_SECS,
};

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Render an epoch as RFC3339 for human-facing output.
pub fn fmt_epoch(epoch: i64) -> String {
    OffsetDateTime::from_unix_timestamp(epoch)
        .ok()
        .and_then(|t| t.format(&Rfc3339).ok())
        .unwrap_or_else(|| epoch.to_string())
}

/// Parse a duration like `90d`, `12h`, `30m`, `3600s`, or a bare integer
/// (seconds). Used for token expiry and TTLs.
pub fn parse_duration(s: &str) -> Result<i64, VecdError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(VecdError::usage("empty duration"));
    }
    let (num, mult) = match s.chars().last().unwrap() {
        'd' => (&s[..s.len() - 1], 86_400),
        'h' => (&s[..s.len() - 1], 3_600),
        'm' => (&s[..s.len() - 1], 60),
        's' => (&s[..s.len() - 1], 1),
        c if c.is_ascii_digit() => (s, 1),
        other => return Err(VecdError::usage(format!("bad duration unit '{other}' (use d/h/m/s)"))),
    };
    let n: i64 = num
        .trim()
        .parse()
        .map_err(|_| VecdError::usage(format!("bad duration '{s}'")))?;
    if n < 0 {
        return Err(VecdError::usage("duration must not be negative"));
    }
    Ok(n * mult)
}

/// Parse a human size like `50T`, `100G`, `512M`, `1024K`, or bare bytes.
pub fn parse_size(s: &str) -> Result<u64, VecdError> {
    let s = s.trim();
    let (num, mult): (&str, u64) = match s.chars().last() {
        Some('T') | Some('t') => (&s[..s.len() - 1], 1u64 << 40),
        Some('G') | Some('g') => (&s[..s.len() - 1], 1u64 << 30),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1u64 << 20),
        Some('K') | Some('k') => (&s[..s.len() - 1], 1u64 << 10),
        Some(c) if c.is_ascii_digit() => (s, 1),
        _ => return Err(VecdError::usage(format!("bad size '{s}'"))),
    };
    num.trim().parse::<u64>().map(|n| n * mult).map_err(|_| VecdError::usage(format!("bad size '{s}'")))
}

/// Parse an access-profile spec — comma-separated `<class> <scope>`
/// entries (`"read datasets/glove, publish datasets/scratch"`) — into the
/// JSON stored on a token / profile. Validates each class.
pub fn parse_profile_spec(spec: &str) -> Result<String, VecdError> {
    let mut entries = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (class, scope) = part
            .split_once(char::is_whitespace)
            .ok_or_else(|| VecdError::usage(format!("profile entry '{part}' must be '<class> <scope>'")))?;
        // validate the class
        let _: Class = class.parse()?;
        entries.push(ProfileEntry {
            class: class.trim().to_ascii_lowercase(),
            scope: scope.trim().to_string(),
        });
    }
    if entries.is_empty() {
        return Err(VecdError::usage("empty profile spec"));
    }
    serde_json::to_string(&entries).map_err(|e| VecdError::op(e.to_string()))
}

/// Resolve a `--role` argument to a canonical, existing role name. Accepts
/// the four class shorthands and their `-er` aliases, or any custom role
/// already in the `roles` table.
fn resolve_role(db: &Db, name: &str) -> Result<String, VecdError> {
    let exists: bool = db
        .conn()
        .query_row("SELECT 1 FROM roles WHERE name=?1", params![name], |_| Ok(()))
        .optional()?
        .is_some();
    if exists {
        return Ok(name.to_string());
    }
    // Map an alias (reader/publisher/…) to its canonical class role.
    let class: Class = name.parse()?;
    Ok(class.name().to_string())
}

fn user_id(db: &Db, name: &str) -> Result<i64, VecdError> {
    db.conn()
        .query_row("SELECT id FROM users WHERE name=?1", params![name], |r| r.get(0))
        .optional()?
        .ok_or_else(|| VecdError::usage(format!("no such user '{name}'")))
}

// ── users ───────────────────────────────────────────────────────────

/// Add a user. When `home_backend` is set, also auto-provisions a private
/// home namespace `users/<name>/` owned by the user on that backend.
pub fn add_user(
    db: &mut Db,
    name: &str,
    level: Level,
    password: Option<&str>,
    home_backend: Option<&str>,
) -> Result<(), VecdError> {
    let pw_hash = password.map(auth::hash_token);
    db.with_cp_txn(|tx| {
        tx.execute(
            "INSERT INTO users(name,created,disabled,level,password_hash) VALUES(?1,?2,0,?3,?4)",
            params![name, now(), level.name(), pw_hash],
        )
        .map_err(|e| dup(e, format!("user '{name}' already exists")))?;
        if let Some(backend) = home_backend {
            let path = format!("users/{name}");
            tx.execute(
                "INSERT INTO namespaces(path,owner,backend_config,active,listable,quota_bytes,ttl_seconds,created)
                 VALUES(?1,?2,?3,1,'grantees',?4,NULL,?5)",
                params![path, name, backend, crate::model::DEFAULT_QUOTA_BYTES as i64, now()],
            )?;
        }
        Ok(())
    })
}

pub fn set_level(db: &mut Db, name: &str, level: Level) -> Result<(), VecdError> {
    let id = user_id(db, name)?;
    db.with_cp_txn(|tx| {
        tx.execute("UPDATE users SET level=?1 WHERE id=?2", params![level.name(), id])?;
        Ok(())
    })
}

pub fn set_password(db: &mut Db, name: &str, password: &str) -> Result<(), VecdError> {
    let id = user_id(db, name)?;
    let hash = auth::hash_token(password);
    db.with_cp_txn(|tx| {
        tx.execute("UPDATE users SET password_hash=?1 WHERE id=?2", params![hash, id])?;
        Ok(())
    })
}

pub fn set_disabled(db: &mut Db, name: &str, disabled: bool) -> Result<(), VecdError> {
    let id = user_id(db, name)?;
    db.with_cp_txn(|tx| {
        tx.execute("UPDATE users SET disabled=?1 WHERE id=?2", params![disabled as i64, id])?;
        Ok(())
    })
}

pub fn remove_user(db: &mut Db, name: &str) -> Result<(), VecdError> {
    let id = user_id(db, name)?;
    db.with_cp_txn(|tx| {
        tx.execute("DELETE FROM users WHERE id=?1", params![id])?;
        Ok(())
    })
}

/// (name, level, disabled) for every user.
pub fn list_users(db: &Db) -> Result<Vec<(String, String, bool)>, VecdError> {
    Ok(db
        .conn()
        .prepare("SELECT name,level,disabled FROM users ORDER BY name")?
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)? != 0)))?
        .collect::<Result<Vec<_>, _>>()?)
}

// ── tokens ──────────────────────────────────────────────────────────

/// A freshly minted token — the plaintext is shown once and not stored.
pub struct TokenCreated {
    pub id: i64,
    pub plaintext: String,
    pub expires_at: i64,
}

/// Mint a token for `user`. `expires` defaults to 90d and is hard-capped
/// at 365d. `profile_json` is `None` for the user's full authority, or the
/// JSON from [`parse_profile_spec`] for a delegated subset.
pub fn create_token(
    db: &mut Db,
    user: &str,
    description: &str,
    expires: Option<&str>,
    profile_json: Option<String>,
) -> Result<TokenCreated, VecdError> {
    if description.trim().is_empty() {
        return Err(VecdError::usage("a token description is mandatory (shown at every usage point)"));
    }
    let uid = user_id(db, user)?;
    let ttl = match expires {
        Some(d) => parse_duration(d)?,
        None => DEFAULT_TOKEN_TTL_SECS,
    };
    if ttl <= 0 {
        return Err(VecdError::usage("token expiry must be positive"));
    }
    if ttl > MAX_TOKEN_TTL_SECS {
        return Err(VecdError::usage(format!(
            "token expiry {ttl}s exceeds the hard maximum of {MAX_TOKEN_TTL_SECS}s (365d)"
        )));
    }
    let expires_at = now() + ttl;
    let (plaintext, hash) = auth::generate_token();
    let id = db.with_cp_txn(|tx| {
        tx.execute(
            "INSERT INTO tokens(user_id,token_hash,description,profile,created,expires_at)
             VALUES(?1,?2,?3,?4,?5,?6)",
            params![uid, hash, description, profile_json, now(), expires_at],
        )?;
        Ok(tx.last_insert_rowid())
    })?;
    Ok(TokenCreated { id, plaintext, expires_at })
}

pub fn revoke_token(db: &mut Db, id: i64) -> Result<(), VecdError> {
    db.with_cp_txn(|tx| {
        let n = tx.execute("DELETE FROM tokens WHERE id=?1", params![id])?;
        if n == 0 {
            return Err(VecdError::usage(format!("no token with id {id}")));
        }
        Ok(())
    })
}

/// (id, user, description, expires_at) for tokens, optionally one user's.
pub fn list_tokens(db: &Db, user: Option<&str>) -> Result<Vec<(i64, String, String, i64)>, VecdError> {
    let sql = "SELECT t.id, u.name, t.description, t.expires_at
               FROM tokens t JOIN users u ON u.id=t.user_id
               WHERE (?1 IS NULL OR u.name=?1) ORDER BY t.id";
    Ok(db
        .conn()
        .prepare(sql)?
        .query_map(params![user], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)?))
        })?
        .collect::<Result<Vec<_>, _>>()?)
}

// ── roles ───────────────────────────────────────────────────────────

pub fn add_role(db: &mut Db, name: &str, actions_csv: &str) -> Result<(), VecdError> {
    let set = crate::model::ActionSet::parse_csv(actions_csv)?;
    db.with_cp_txn(|tx| {
        tx.execute(
            "INSERT INTO roles(name,actions,builtin) VALUES(?1,?2,0)",
            params![name, set.to_csv()],
        )
        .map_err(|e| dup(e, format!("role '{name}' already exists")))?;
        Ok(())
    })
}

pub fn remove_role(db: &mut Db, name: &str) -> Result<(), VecdError> {
    db.with_cp_txn(|tx| {
        let builtin: Option<bool> = tx
            .query_row("SELECT builtin FROM roles WHERE name=?1", params![name], |r| {
                Ok(r.get::<_, i64>(0)? != 0)
            })
            .optional()?;
        match builtin {
            None => return Err(VecdError::usage(format!("no such role '{name}'"))),
            Some(true) => return Err(VecdError::usage(format!("cannot remove built-in role '{name}'"))),
            Some(false) => {}
        }
        tx.execute("DELETE FROM roles WHERE name=?1", params![name])?;
        Ok(())
    })
}

/// (name, actions, builtin) for every role.
pub fn list_roles(db: &Db) -> Result<Vec<(String, String, bool)>, VecdError> {
    Ok(db
        .conn()
        .prepare("SELECT name,actions,builtin FROM roles ORDER BY name")?
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)? != 0)))?
        .collect::<Result<Vec<_>, _>>()?)
}

// ── backends ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn add_backend(
    db: &mut Db,
    name: &str,
    kind: &str,
    endpoint: &str,
    endpoint_url: Option<&str>,
    region: Option<&str>,
    aws_profile: Option<&str>,
    active: bool,
) -> Result<(), VecdError> {
    if !["local", "s3", "mem"].contains(&kind) {
        return Err(VecdError::usage(format!("unknown backend kind '{kind}' (local|s3|mem)")));
    }
    db.with_cp_txn(|tx| {
        if active {
            ensure_endpoint_free(tx, endpoint, None)?;
        }
        tx.execute(
            "INSERT INTO backends(name,kind,endpoint,endpoint_url,region,creds_ref,active,created)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![name, kind, endpoint, endpoint_url, region, aws_profile, active as i64, now()],
        )
        .map_err(|e| dup(e, format!("backend '{name}' already exists")))?;
        Ok(())
    })
}

pub fn set_backend_active(db: &mut Db, name: &str, active: bool) -> Result<(), VecdError> {
    db.with_cp_txn(|tx| {
        let endpoint: String = tx
            .query_row("SELECT endpoint FROM backends WHERE name=?1", params![name], |r| r.get(0))
            .optional()?
            .ok_or_else(|| VecdError::usage(format!("no such backend '{name}'")))?;
        if active {
            ensure_endpoint_free(tx, &endpoint, Some(name))?;
        }
        tx.execute("UPDATE backends SET active=?1 WHERE name=?2", params![active as i64, name])?;
        Ok(())
    })
}

pub fn remove_backend(db: &mut Db, name: &str) -> Result<(), VecdError> {
    db.with_cp_txn(|tx| {
        let in_use: bool = tx
            .query_row(
                "SELECT 1 FROM namespaces WHERE backend_config=?1 LIMIT 1",
                params![name],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if in_use {
            return Err(VecdError::usage(format!(
                "backend '{name}' is referenced by a namespace; repoint it first"
            )));
        }
        let n = tx.execute("DELETE FROM backends WHERE name=?1", params![name])?;
        if n == 0 {
            return Err(VecdError::usage(format!("no such backend '{name}'")));
        }
        Ok(())
    })
}

/// (name, kind, endpoint, active) for every backend.
pub fn list_backends(db: &Db) -> Result<Vec<(String, String, String, bool)>, VecdError> {
    Ok(db
        .conn()
        .prepare("SELECT name,kind,endpoint,active FROM backends ORDER BY name")?
        .query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)? != 0))
        })?
        .collect::<Result<Vec<_>, _>>()?)
}

/// Enforce one-endpoint-one-active-config (the partial unique index also
/// guards this, but we give a clear error first).
fn ensure_endpoint_free(
    tx: &rusqlite::Transaction,
    endpoint: &str,
    excluding: Option<&str>,
) -> Result<(), VecdError> {
    let claimed: Option<String> = tx
        .query_row(
            "SELECT name FROM backends WHERE endpoint=?1 AND active=1 AND (?2 IS NULL OR name<>?2) LIMIT 1",
            params![endpoint, excluding],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(other) = claimed {
        return Err(VecdError::usage(format!(
            "endpoint '{endpoint}' is already held by active backend '{other}' \
             (one endpoint, one active config); deactivate it first"
        )));
    }
    Ok(())
}

// ── namespaces ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn add_namespace(
    db: &mut Db,
    path: &str,
    owner: &str,
    backend_config: Option<&str>,
    active: bool,
    listable: Listable,
    ttl: Option<&str>,
    quota: Option<&str>,
) -> Result<(), VecdError> {
    let path = crate::authz::normalize(path);
    if path.is_empty() {
        return Err(VecdError::usage("the root namespace already exists and cannot be re-created"));
    }
    let ttl_seconds = ttl.map(parse_duration).transpose()?;
    let quota_bytes = match quota {
        Some(q) => parse_size(q)? as i64,
        None => crate::model::DEFAULT_QUOTA_BYTES as i64,
    };
    db.with_cp_txn(|tx| {
        if let Some(bc) = backend_config {
            backend_exists(tx, bc)?;
        }
        tx.execute(
            "INSERT INTO namespaces(path,owner,backend_config,active,listable,quota_bytes,ttl_seconds,created)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![path, owner, backend_config, active as i64, listable.name(), quota_bytes, ttl_seconds, now()],
        )
        .map_err(|e| dup(e, format!("namespace '{path}' already exists")))?;
        Ok(())
    })
}

/// Mutate one or more namespace fields. `None` fields are left unchanged.
#[allow(clippy::too_many_arguments)]
pub fn set_namespace(
    db: &mut Db,
    path: &str,
    owner: Option<&str>,
    backend_config: Option<Option<&str>>,
    active: Option<bool>,
    listable: Option<Listable>,
    ttl: Option<Option<&str>>,
    quota: Option<&str>,
) -> Result<(), VecdError> {
    let path = crate::authz::normalize(path);
    let ttl_seconds: Option<Option<i64>> = match ttl {
        Some(Some(d)) => Some(Some(parse_duration(d)?)),
        Some(None) => Some(None),
        None => None,
    };
    let quota_bytes = match quota {
        Some(q) => Some(parse_size(q)? as i64),
        None => None,
    };
    db.with_cp_txn(|tx| {
         namespace_exists(tx, &path)?;
        if let Some(o) = owner {
            tx.execute("UPDATE namespaces SET owner=?1 WHERE path=?2", params![o, path])?;
        }
        if let Some(bc) = backend_config {
            if let Some(name) = bc {
                backend_exists(tx, name)?;
            }
            tx.execute("UPDATE namespaces SET backend_config=?1 WHERE path=?2", params![bc, path])?;
        }
        if let Some(a) = active {
            tx.execute("UPDATE namespaces SET active=?1 WHERE path=?2", params![a as i64, path])?;
        }
        if let Some(l) = listable {
            tx.execute("UPDATE namespaces SET listable=?1 WHERE path=?2", params![l.name(), path])?;
        }
        if let Some(t) = ttl_seconds {
            tx.execute("UPDATE namespaces SET ttl_seconds=?1 WHERE path=?2", params![t, path])?;
        }
        if let Some(q) = quota_bytes {
            tx.execute("UPDATE namespaces SET quota_bytes=?1 WHERE path=?2", params![q, path])?;
        }
        Ok(())
    })
}

pub fn remove_namespace(db: &mut Db, path: &str) -> Result<(), VecdError> {
    let path = crate::authz::normalize(path);
    if path.is_empty() {
        return Err(VecdError::usage("cannot remove the root namespace"));
    }
    db.with_cp_txn(|tx| {
        // Phase 1: a direct retire (the non-destructive stasis path is
        // Phase 2). Refuse if it still holds objects so bytes aren't
        // silently orphaned.
        let has_objects: bool = tx
            .query_row(
                "SELECT 1 FROM objects WHERE namespace_path=?1 OR namespace_path LIKE ?1||'/%' LIMIT 1",
                params![path],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if has_objects {
            return Err(VecdError::usage(format!(
                "namespace '{path}' still holds objects; (stasis-based removal lands in Phase 2 — \
                 delete its objects first in Phase 1)"
            )));
        }
        let n = tx.execute("DELETE FROM namespaces WHERE path=?1", params![path])?;
        if n == 0 {
            return Err(VecdError::usage(format!("no such namespace '{path}'")));
        }
        tx.execute("DELETE FROM role_bindings WHERE namespace_path=?1", params![path])?;
        Ok(())
    })
}

/// (path, owner, backend_config, active, listable) for every namespace.
pub fn list_namespaces(
    db: &Db,
) -> Result<Vec<(String, String, Option<String>, bool, String)>, VecdError> {
    Ok(db
        .conn()
        .prepare("SELECT path,owner,backend_config,active,listable FROM namespaces ORDER BY path")?
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, i64>(3)? != 0,
                r.get::<_, String>(4)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?)
}

fn backend_exists(tx: &rusqlite::Transaction, name: &str) -> Result<(), VecdError> {
    let ok: bool = tx
        .query_row("SELECT 1 FROM backends WHERE name=?1", params![name], |_| Ok(()))
        .optional()?
        .is_some();
    if ok {
        Ok(())
    } else {
        Err(VecdError::usage(format!("no such backend config '{name}'")))
    }
}

fn namespace_exists(tx: &rusqlite::Transaction, path: &str) -> Result<(), VecdError> {
    let ok: bool = tx
        .query_row("SELECT 1 FROM namespaces WHERE path=?1", params![path], |_| Ok(()))
        .optional()?
        .is_some();
    if ok {
        Ok(())
    } else {
        Err(VecdError::usage(format!("no such namespace '{path}'")))
    }
}

// ── bindings & privileges ───────────────────────────────────────────

/// Bind a role to a principal (`PUBLIC`/`KNOWN`/user name) on a namespace
/// subtree.
pub fn bind(db: &mut Db, principal: &str, role: &str, ns: &str) -> Result<(), VecdError> {
    let role = resolve_role(db, role)?;
    let ns = crate::authz::normalize(ns);
    validate_principal(db, principal)?;
    db.with_cp_txn(|tx| {
        namespace_exists(tx, &ns)?;
        tx.execute(
            "INSERT OR IGNORE INTO role_bindings(principal,role,namespace_path,created_by,created)
             VALUES(?1,?2,?3,'admin-cli',?4)",
            params![principal, role, ns, now()],
        )?;
        Ok(())
    })
}

/// Remove a binding. `role = None` removes every binding for the principal
/// on the namespace.
pub fn unbind(db: &mut Db, principal: &str, role: Option<&str>, ns: &str) -> Result<(), VecdError> {
    let ns = crate::authz::normalize(ns);
    let role = role.map(|r| resolve_role(db, r)).transpose()?;
    db.with_cp_txn(|tx| {
        match &role {
            Some(r) => tx.execute(
                "DELETE FROM role_bindings WHERE principal=?1 AND role=?2 AND namespace_path=?3",
                params![principal, r, ns],
            )?,
            None => tx.execute(
                "DELETE FROM role_bindings WHERE principal=?1 AND namespace_path=?2",
                params![principal, ns],
            )?,
        };
        Ok(())
    })
}

/// (principal, role, namespace) for every binding.
pub fn list_bindings(db: &Db) -> Result<Vec<(String, String, String)>, VecdError> {
    Ok(db
        .conn()
        .prepare("SELECT principal,role,namespace_path FROM role_bindings ORDER BY namespace_path,principal")?
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?
        .collect::<Result<Vec<_>, _>>()?)
}

fn validate_principal(db: &Db, principal: &str) -> Result<(), VecdError> {
    if principal == "PUBLIC" || principal == "KNOWN" {
        return Ok(());
    }
    let _ = user_id(db, principal)?;
    Ok(())
}

pub fn grant_privilege(db: &mut Db, principal: &str, privilege: &str) -> Result<(), VecdError> {
    db.with_cp_txn(|tx| {
        tx.execute(
            "INSERT OR IGNORE INTO system_privileges(principal,privilege,granted_by,created)
             VALUES(?1,?2,'admin-cli',?3)",
            params![principal, privilege, now()],
        )?;
        Ok(())
    })
}

pub fn revoke_privilege(db: &mut Db, principal: &str, privilege: &str) -> Result<(), VecdError> {
    db.with_cp_txn(|tx| {
        tx.execute(
            "DELETE FROM system_privileges WHERE principal=?1 AND privilege=?2",
            params![principal, privilege],
        )?;
        Ok(())
    })
}

// ── named privilege profiles ────────────────────────────────────────

/// Add a named, persisted, parameterized profile (placeholders like
/// `{dataset}` are filled at token-create time).
pub fn add_profile(db: &mut Db, name: &str, owner: &str, spec: &str) -> Result<(), VecdError> {
    // Validate the template's classes, allowing `{placeholder}` in scopes.
    validate_profile_template(spec)?;
    db.with_cp_txn(|tx| {
        tx.execute(
            "INSERT INTO profiles(name,owner,spec,created) VALUES(?1,?2,?3,?4)",
            params![name, owner, spec, now()],
        )
        .map_err(|e| dup(e, format!("profile '{name}' already exists")))?;
        Ok(())
    })
}

pub fn remove_profile(db: &mut Db, name: &str) -> Result<(), VecdError> {
    db.with_cp_txn(|tx| {
        let n = tx.execute("DELETE FROM profiles WHERE name=?1", params![name])?;
        if n == 0 {
            return Err(VecdError::usage(format!("no such profile '{name}'")));
        }
        Ok(())
    })
}

/// (name, owner, spec) for every profile.
pub fn list_profiles(db: &Db) -> Result<Vec<(String, String, String)>, VecdError> {
    Ok(db
        .conn()
        .prepare("SELECT name,owner,spec FROM profiles ORDER BY name")?
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?
        .collect::<Result<Vec<_>, _>>()?)
}

/// Expand a named profile into a concrete profile JSON, filling
/// `{placeholder}` positions from `sets` (`pos=val`). Bounded later by the
/// issuer's authority at token-create time.
pub fn expand_profile(db: &Db, name: &str, sets: &[(String, String)]) -> Result<String, VecdError> {
    let spec: String = db
        .conn()
        .query_row("SELECT spec FROM profiles WHERE name=?1", params![name], |r| r.get(0))
        .optional()?
        .ok_or_else(|| VecdError::usage(format!("no such profile '{name}'")))?;
    let mut expanded = spec;
    for (pos, val) in sets {
        expanded = expanded.replace(&format!("{{{pos}}}"), val);
    }
    if let Some(start) = expanded.find('{') {
        let frag = &expanded[start..];
        return Err(VecdError::usage(format!(
            "profile '{name}' still has an unfilled placeholder near '{frag}' — use --set pos=val"
        )));
    }
    parse_profile_spec(&expanded)
}

/// Validate a profile *template*: each entry is `<class> <scope>` and the
/// class is known; scopes may contain `{placeholders}`.
fn validate_profile_template(spec: &str) -> Result<(), VecdError> {
    let mut any = false;
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        any = true;
        let (class, _scope) = part
            .split_once(char::is_whitespace)
            .ok_or_else(|| VecdError::usage(format!("profile entry '{part}' must be '<class> <scope>'")))?;
        let _: Class = class.parse()?;
    }
    if any {
        Ok(())
    } else {
        Err(VecdError::usage("empty profile spec"))
    }
}

// ── helpers ─────────────────────────────────────────────────────────

/// Map a UNIQUE-constraint violation to a friendly usage error.
fn dup(e: rusqlite::Error, msg: String) -> VecdError {
    if let rusqlite::Error::SqliteFailure(err, _) = &e {
        if err.code == rusqlite::ErrorCode::ConstraintViolation {
            return VecdError::usage(msg);
        }
    }
    VecdError::Db(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::init(&dir.path().join("vecd.db")).unwrap();
        (dir, db)
    }

    #[test]
    fn duration_and_size_parsing() {
        assert_eq!(parse_duration("90d").unwrap(), 90 * 86400);
        assert_eq!(parse_duration("12h").unwrap(), 12 * 3600);
        assert_eq!(parse_duration("3600").unwrap(), 3600);
        assert!(parse_duration("5y").is_err());
        assert_eq!(parse_size("50T").unwrap(), 50u64 << 40);
        assert_eq!(parse_size("512M").unwrap(), 512u64 << 20);
    }

    #[test]
    fn profile_spec_roundtrip() {
        let json = parse_profile_spec("read datasets/glove, publish datasets/scratch").unwrap();
        let entries: Vec<ProfileEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].class, "read");
        assert_eq!(entries[1].scope, "datasets/scratch");
        assert!(parse_profile_spec("bogus datasets/x").is_err());
    }

    #[test]
    fn users_tokens_lifecycle() {
        let (_d, mut db) = db();
        add_user(&mut db, "alice", Level::User, None, None).unwrap();
        assert!(add_user(&mut db, "alice", Level::User, None, None).is_err()); // dup
        let tok = create_token(&mut db, "alice", "ci", Some("7d"), None).unwrap();
        assert!(tok.plaintext.starts_with("vd_"));
        assert!(tok.expires_at > now());
        // mandatory description
        assert!(create_token(&mut db, "alice", "  ", None, None).is_err());
        // max-lifetime enforced
        assert!(create_token(&mut db, "alice", "x", Some("400d"), None).is_err());
        revoke_token(&mut db, tok.id).unwrap();
        assert!(revoke_token(&mut db, tok.id).is_err());
    }

    #[test]
    fn endpoint_exclusivity_enforced() {
        let (_d, mut db) = db();
        add_backend(&mut db, "a", "s3", "s3://b/p", None, None, None, true).unwrap();
        // A second *active* config on the same endpoint is refused.
        assert!(add_backend(&mut db, "b", "s3", "s3://b/p", None, None, None, true).is_err());
        // Inactive is fine (standby).
        add_backend(&mut db, "b", "s3", "s3://b/p", None, None, None, false).unwrap();
        // Activating it now collides.
        assert!(set_backend_active(&mut db, "b", true).is_err());
    }

    #[test]
    fn namespace_and_binding_setup() {
        let (_d, mut db) = db();
        add_backend(&mut db, "store", "mem", "mem:x", None, None, None, true).unwrap();
        add_user(&mut db, "alice", Level::User, None, None).unwrap();
        add_namespace(&mut db, "datasets/glove", "alice", Some("store"), true, Listable::Grantees, None, None)
            .unwrap();
        bind(&mut db, "PUBLIC", "reader", "datasets/glove").unwrap();
        bind(&mut db, "alice", "curate", "datasets/glove").unwrap();
        let binds = list_bindings(&db).unwrap();
        assert!(binds.iter().any(|(p, r, _)| p == "PUBLIC" && r == "read"));
        assert!(binds.iter().any(|(p, r, _)| p == "alice" && r == "curate"));
        // Binding to an unknown user fails.
        assert!(bind(&mut db, "nobody", "read", "datasets/glove").is_err());
    }

    #[test]
    fn profile_expansion_fills_placeholders() {
        let (_d, mut db) = db();
        add_profile(&mut db, "collab", "alice", "read datasets/{dataset}/, publish datasets/{dataset}/scratch/")
            .unwrap();
        let json = expand_profile(&db, "collab", &[("dataset".into(), "glove-100".into())]).unwrap();
        assert!(json.contains("datasets/glove-100/"));
        // Missing placeholder is an error.
        assert!(expand_profile(&db, "collab", &[]).is_err());
    }
}
