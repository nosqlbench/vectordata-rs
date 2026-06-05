// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Client-side credential store for `vecd login` — bearer tokens keyed by
//! endpoint **origin** (`scheme://host[:port]`), so a stored token is applied
//! automatically to any request `vecd` makes to that endpoint (today:
//! `vecd whoami`).
//!
//! The store is `credentials.json` in the resolved config dir (so `--conf` /
//! `$VECD_CONFIG` isolate it, defaulting to `~/.config/vecd/`), created
//! `0600`. Origin parsing is shared with the `vectordata` client
//! ([`vectordata::credentials::origin_of_str`]) so both tools key tokens the
//! same way.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::model::VecdError;

/// One stored credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    /// The bearer secret.
    pub token: String,
    /// Informational: the user the token was minted for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Informational: expiry as epoch seconds (stringified), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
}

/// A map of endpoint origin → [`Entry`].
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Store {
    #[serde(flatten)]
    entries: BTreeMap<String, Entry>,
}

impl Store {
    /// Load the store from `<dir>/credentials.json`, or an empty store if it
    /// is absent or unreadable.
    pub fn load(dir: &Path) -> Store {
        std::fs::read_to_string(path(dir))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Persist the store (dir `0700`, file `0600`).
    pub fn save(&self, dir: &Path) -> Result<(), VecdError> {
        std::fs::create_dir_all(dir)?;
        set_mode(dir, 0o700);
        let p = path(dir);
        let text = serde_json::to_string_pretty(self).map_err(|e| VecdError::op(e.to_string()))?;
        std::fs::write(&p, text)?;
        set_mode(&p, 0o600);
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

    /// `(origin, entry)` for every stored credential.
    pub fn list(&self) -> Vec<(&String, &Entry)> {
        self.entries.iter().collect()
    }
}

fn path(dir: &Path) -> PathBuf {
    dir.join("credentials.json")
}

/// The bearer token stored for `url`'s origin, if any — applied automatically
/// to requests `vecd` makes to that endpoint.
pub fn stored_token(dir: &Path, url: &str) -> Option<String> {
    let origin = vectordata::credentials::origin_of_str(url)?;
    Store::load(dir).get(&origin).map(|e| e.token.clone())
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
}
#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_roundtrip_by_origin() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Store::default();
        s.set(
            "http://127.0.0.1:8443".into(),
            Entry { token: "vd_abc".into(), user: Some("alice".into()), expires: None },
        );
        s.save(dir.path()).unwrap();
        // Reload and look up by an endpoint URL whose origin matches.
        assert_eq!(
            stored_token(dir.path(), "http://127.0.0.1:8443/datasets/x/y.fvec").as_deref(),
            Some("vd_abc")
        );
        // A different origin has no credential.
        assert_eq!(stored_token(dir.path(), "http://other:9000/").as_deref(), None);
        // Remove.
        let mut s = Store::load(dir.path());
        assert!(s.remove("http://127.0.0.1:8443"));
        s.save(dir.path()).unwrap();
        assert_eq!(stored_token(dir.path(), "http://127.0.0.1:8443/").as_deref(), None);
    }
}
