// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Namespace → backend resolution and key confinement.
//!
//! A request path is *governed* by the deepest covering namespace (for
//! AAA — handled by [`crate::authz::Snapshot::allowed`]) but *stored*
//! through the **nearest covering namespace that has an active backend**
//! (an ancestor walk, so a config-only child inherits its parent's
//! storage, and a child with its own backend overrides it). The two are
//! independent concerns.

use std::sync::Arc;

use crate::authz::{covers, normalize, Snapshot};
use crate::backend::{self, Backend};
use crate::db::BackendRow;
use crate::model::VecdError;

/// Where a request's bytes live.
pub struct Resolved {
    /// The namespace whose backend is used (the storage extent).
    pub storage_ns: String,
    /// The opened backend.
    pub backend: Arc<dyn Backend>,
    /// The key relative to `storage_ns` — what the backend stores under.
    pub rel_key: String,
    /// The resolved backend config (for diagnostics / `ns show`).
    pub backend_config: BackendRow,
}

/// Reject keys that would escape a namespace's storage extent: `..`/`.`
/// segments, absolute paths, or empty path segments (`a//b`). One
/// namespace can never read or write into another's storage.
pub fn validate_key(key: &str) -> Result<(), VecdError> {
    if key.starts_with('/') {
        return Err(VecdError::usage("object key must not be absolute".to_string()));
    }
    for seg in key.split('/') {
        if seg.is_empty() || seg == "." || seg == ".." {
            return Err(VecdError::usage(format!("object key '{key}' has an illegal path segment")));
        }
    }
    Ok(())
}

/// Resolve the storage backend and namespace-relative key for a request
/// path. `request_path` is the URL path with the leading `/` already
/// stripped (e.g. `datasets/glove/base.fvec`).
pub fn resolve(snap: &Snapshot, request_path: &str) -> Result<Resolved, VecdError> {
    let key = normalize(request_path);
    validate_key(&key)?;

    // Covering namespaces, deepest first.
    let mut covering: Vec<_> = snap.namespaces().filter(|n| covers(&n.path, &key)).collect();
    covering.sort_by_key(|n| std::cmp::Reverse(n.path.len()));

    for ns in &covering {
        if !ns.active {
            continue;
        }
        let Some(bname) = &ns.backend_config else { continue };
        let Some(brow) = snap.backend(bname) else { continue };
        if !brow.active {
            continue;
        }
        let rel_key = relativize(&ns.path, &key);
        if rel_key.is_empty() {
            return Err(VecdError::usage(format!(
                "'{request_path}' addresses a namespace, not an object"
            )));
        }
        let backend = backend::open(brow)?;
        return Ok(Resolved {
            storage_ns: ns.path.clone(),
            backend,
            rel_key,
            backend_config: brow.clone(),
        });
    }

    // Nothing active + backed served the key. Diagnose from the most-specific
    // covering namespace so the caller learns the actual fix (inactive vs. no
    // backend vs. a dead backend) instead of a generic "no backend".
    if let Some(ns) = covering.first() {
        let label = if ns.path.is_empty() { "/".to_string() } else { ns.path.clone() };
        if !ns.active {
            return Err(VecdError::usage(format!(
                "namespace '{label}' is inactive (config-only) — activate it with \
                 `vecd ns set {label} --active`"
            )));
        }
        return match &ns.backend_config {
            None => Err(VecdError::usage(format!(
                "namespace '{label}' has no storage backend — attach one with \
                 `vecd ns set {label} --backend-config <name>`"
            ))),
            Some(bn) => Err(VecdError::usage(format!(
                "namespace '{label}' uses backend '{bn}', which is missing or inactive — \
                 check `vecd backends list` (activate with `vecd backends set {bn} --active`)"
            ))),
        };
    }

    Err(VecdError::usage(format!("no namespace covers '{request_path}'")))
}

/// Resolve a *prefix* (which may name a namespace itself) to its storage
/// namespace and the prefix relative to it — for listing. Unlike
/// [`resolve`], an empty relative prefix is allowed (listing a whole
/// namespace). Returns `(storage_ns, rel_prefix)`.
pub fn resolve_for_list(snap: &Snapshot, request_path: &str) -> Result<(String, String), VecdError> {
    let key = normalize(request_path);
    if !key.is_empty() {
        validate_key(&key)?;
    }
    let mut covering: Vec<_> = snap.namespaces().filter(|n| covers(&n.path, &key)).collect();
    covering.sort_by_key(|n| std::cmp::Reverse(n.path.len()));
    for ns in covering {
        if !ns.active {
            continue;
        }
        let Some(bname) = &ns.backend_config else { continue };
        let Some(brow) = snap.backend(bname) else { continue };
        if !brow.active {
            continue;
        }
        return Ok((ns.path.clone(), relativize(&ns.path, &key)));
    }
    Err(VecdError::usage(format!("no active storage backend serves '{request_path}'")))
}

/// `key` relative to `ns_path` (which is known to cover it).
fn relativize(ns_path: &str, key: &str) -> String {
    let ns_path = ns_path.trim_matches('/');
    if ns_path.is_empty() {
        key.to_string()
    } else if key == ns_path {
        String::new()
    } else {
        key.strip_prefix(&format!("{ns_path}/")).unwrap_or(key).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{BackendRow, ControlPlane, NamespaceRow};

    fn cp_with(active_backend: bool, ns_active: bool) -> ControlPlane {
        let mut cp = ControlPlane::default();
        cp.backends.push(BackendRow {
            name: "b".into(),
            kind: "mem".into(),
            endpoint: "mem:rt".into(),
            endpoint_url: None,
            region: None,
            creds_ref: None,
            active: active_backend,
        });
        cp.namespaces.push(NamespaceRow {
            path: "datasets/glove".into(),
            owner: "@admin".into(),
            backend_config: Some("b".into()),
            active: ns_active,
            listable: "grantees".into(),
            quota_bytes: 1 << 40,
            ttl_seconds: None,
        });
        cp
    }

    #[test]
    fn resolves_to_nearest_active_backend() {
        let snap = Snapshot::build(&cp_with(true, true));
        let r = resolve(&snap, "datasets/glove/base.fvec").unwrap();
        assert_eq!(r.storage_ns, "datasets/glove");
        assert_eq!(r.rel_key, "base.fvec");
    }

    #[test]
    fn inactive_backend_or_namespace_has_no_storage() {
        let snap = Snapshot::build(&cp_with(false, true));
        assert!(resolve(&snap, "datasets/glove/x").is_err());
        let snap2 = Snapshot::build(&cp_with(true, false));
        assert!(resolve(&snap2, "datasets/glove/x").is_err());
    }

    #[test]
    fn key_confinement() {
        assert!(validate_key("a/b/c").is_ok());
        assert!(validate_key("../escape").is_err());
        assert!(validate_key("/abs").is_err());
        assert!(validate_key("a//b").is_err());
        assert!(validate_key("a/./b").is_err());
    }

    #[test]
    fn child_overrides_parent_backend() {
        let mut cp = cp_with(true, true);
        // Parent datasets/ also active with its own backend.
        cp.backends.push(BackendRow {
            name: "parent".into(),
            kind: "mem".into(),
            endpoint: "mem:parent".into(),
            endpoint_url: None,
            region: None,
            creds_ref: None,
            active: true,
        });
        cp.namespaces.push(NamespaceRow {
            path: "datasets".into(),
            owner: "@admin".into(),
            backend_config: Some("parent".into()),
            active: true,
            listable: "grantees".into(),
            quota_bytes: 1 << 40,
            ttl_seconds: None,
        });
        let snap = Snapshot::build(&cp);
        // The deepest (child) backend wins for keys under it.
        let r = resolve(&snap, "datasets/glove/x").unwrap();
        assert_eq!(r.storage_ns, "datasets/glove");
        // A key only under the parent resolves to the parent backend.
        let r2 = resolve(&snap, "datasets/other.txt").unwrap();
        assert_eq!(r2.storage_ns, "datasets");
        assert_eq!(r2.rel_key, "other.txt");
    }
}
