// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-endpoint client credentials and bearer-token resolution.
//!
//! `vectordata login <url>` stores a bearer token bound to the request's
//! **origin** (`scheme://host[:port]`) in
//! `~/.config/vectordata/credentials.toml` (mode `0600`). Push and pull
//! then resolve a token transparently — no `--token` on every call — with
//! this precedence (first present wins):
//!
//! 1. an explicit `--token` / call-site override;
//! 2. `$VECTORDATA_PUSH_TOKEN` (push) / `$VECTORDATA_TOKEN`;
//! 3. the stored credential for the request's origin;
//! 4. none → anonymous.
//!
//! Public datasets keep working with no token at all — token resolution is
//! purely additive: absent everywhere, the request stays anonymous.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use url::Url;

/// Config directory, mirroring `settings.rs`: `$VECTORDATA_HOME`, else
/// `$HOME/.config/vectordata`. Tests isolate via `$VECTORDATA_HOME`.
fn config_dir() -> PathBuf {
    if let Some(root) = std::env::var_os("VECTORDATA_HOME") {
        PathBuf::from(root)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config/vectordata")
    } else {
        PathBuf::from(".config/vectordata")
    }
}

/// Path to the credentials file.
pub fn credentials_path() -> PathBuf {
    config_dir().join("credentials.toml")
}

/// One stored credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    /// The bearer secret.
    pub token: String,
    /// Informational: the user the token was minted for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Informational: the token's expiry, if known (RFC3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
}

/// The credential store — a map of origin → [`Entry`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Store {
    #[serde(flatten)]
    entries: BTreeMap<String, Entry>,
}

impl Store {
    /// Load the store, or an empty one if the file is absent/unreadable.
    pub fn load() -> Store {
        match std::fs::read_to_string(credentials_path()) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Store::default(),
        }
    }

    /// Persist the store (creating the config dir `0700`, the file `0600`).
    pub fn save(&self) -> std::io::Result<()> {
        let dir = config_dir();
        std::fs::create_dir_all(&dir)?;
        set_mode(&dir, 0o700);
        let path = credentials_path();
        let text = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, text)?;
        set_mode(&path, 0o600);
        Ok(())
    }

    pub fn get(&self, origin: &str) -> Option<&Entry> {
        self.entries.get(origin)
    }

    pub fn set(&mut self, origin: String, entry: Entry) {
        self.entries.insert(origin, entry);
    }

    /// Remove an origin's credential; returns whether one was present.
    pub fn remove(&mut self, origin: &str) -> bool {
        self.entries.remove(origin).is_some()
    }

    /// (origin, user) for every stored credential.
    pub fn list(&self) -> Vec<(String, Option<String>)> {
        self.entries.iter().map(|(o, e)| (o.clone(), e.user.clone())).collect()
    }
}

/// The origin (`scheme://host[:port]`) of a URL — the credential key.
pub fn origin_of(url: &Url) -> Option<String> {
    let scheme = url.scheme();
    let host = url.host_str()?;
    Some(match url.port() {
        Some(p) => format!("{scheme}://{host}:{p}"),
        None => format!("{scheme}://{host}"),
    })
}

/// Parse a URL string and return its origin.
pub fn origin_of_str(url: &str) -> Option<String> {
    Url::parse(url).ok().as_ref().and_then(origin_of)
}

/// The cached store for this process (so per-request reads don't re-read
/// the file). A short-lived CLI invocation loads it once.
fn cached_store() -> &'static Store {
    static STORE: OnceLock<Store> = OnceLock::new();
    STORE.get_or_init(Store::load)
}

/// Resolve a bearer token for a **read/pull** to `url`: `$VECTORDATA_TOKEN`,
/// then the stored credential for the origin, else `None` (anonymous).
pub fn resolve_read_token(url: &Url) -> Option<String> {
    if let Ok(t) = std::env::var("VECTORDATA_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    let origin = origin_of(url)?;
    cached_store().get(&origin).map(|e| e.token.clone())
}

/// Resolve a token by origin from the store only (no env) — used by the
/// CLI commands that act on a specific endpoint.
pub fn stored_token(url: &str) -> Option<String> {
    let origin = origin_of_str(url)?;
    Store::load().get(&origin).map(|e| e.token.clone())
}

#[cfg(unix)]
fn set_mode(path: &std::path::Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
}
#[cfg(not(unix))]
fn set_mode(_path: &std::path::Path, _mode: u32) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_extraction() {
        let u = Url::parse("https://vecd-host:8443/datasets/glove/base.fvec").unwrap();
        assert_eq!(origin_of(&u).unwrap(), "https://vecd-host:8443");
        let u2 = Url::parse("https://example.com/x").unwrap();
        assert_eq!(origin_of(&u2).unwrap(), "https://example.com");
    }

    #[test]
    fn store_set_get_remove_roundtrip() {
        let mut s = Store::default();
        s.set(
            "https://h:1".into(),
            Entry { token: "vd_abc".into(), user: Some("alice".into()), expires: None },
        );
        assert_eq!(s.get("https://h:1").unwrap().token, "vd_abc");
        assert_eq!(s.list().len(), 1);
        assert!(s.remove("https://h:1"));
        assert!(!s.remove("https://h:1"));
    }

    #[test]
    fn toml_roundtrip_quotes_origin_keys() {
        let mut s = Store::default();
        s.set(
            "https://vecd-host:8443".into(),
            Entry { token: "vd_x".into(), user: Some("bob".into()), expires: Some("2026-01-01T00:00:00Z".into()) },
        );
        let text = toml::to_string_pretty(&s).unwrap();
        let back: Store = toml::from_str(&text).unwrap();
        assert_eq!(back.get("https://vecd-host:8443").unwrap().token, "vd_x");
    }
}
