// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Operator configuration and standard paths.
//!
//! `vecd` reads settings an operator shouldn't pass on the command line —
//! the DB encryption key, default data dir / bind address / TLS paths, and
//! the DB-backup destination — from a config file in a config directory
//! (`$HOME/.config/vecd/`, overridable via `--config` / `$VECD_CONFIG`).
//! CLI flags override the file. Phase 1 keeps the format minimal (simple
//! `key = value` lines); secret-bearing files are created `0600`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::model::VecdError;

/// Default config directory: `$HOME/.config/vecd/` (or `$VECD_CONFIG`).
pub fn config_dir() -> PathBuf {
    if let Ok(c) = std::env::var("VECD_CONFIG") {
        return PathBuf::from(c);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".config").join("vecd")
}

/// Default data directory (DB + pidfile + local-backend objects):
/// `<config_dir>/data` unless overridden by `--data-dir`.
pub fn default_data_dir() -> PathBuf {
    config_dir().join("data")
}

/// The DB path under a data dir.
pub fn db_path(data_dir: &Path) -> PathBuf {
    data_dir.join("vecd.db")
}

/// Parsed config file. Unknown keys are tolerated (forward compatible).
#[derive(Debug, Default, Clone)]
pub struct Config {
    values: HashMap<String, String>,
}

impl Config {
    /// Load `<config_dir>/vecd.conf`, or an empty config if it is absent.
    pub fn load(dir: &Path) -> Result<Self, VecdError> {
        let path = dir.join("vecd.conf");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Config::default()),
            Err(e) => return Err(e.into()),
        };
        Self::parse(&text)
    }

    pub(crate) fn parse(text: &str) -> Result<Self, VecdError> {
        let mut values = HashMap::new();
        for (n, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (k, v) = line.split_once('=').ok_or_else(|| {
                VecdError::usage(format!("vecd.conf line {}: expected key = value", n + 1))
            })?;
            values.insert(k.trim().to_string(), v.trim().trim_matches('"').to_string());
        }
        Ok(Config { values })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    /// The DB encryption key (`db_key`), if set. Surfaced into the
    /// environment for the SQLCipher PRAGMA when built with that feature.
    pub fn db_key(&self) -> Option<&str> {
        self.get("db_key")
    }

    /// The DB-backup destination (`db_backup`), e.g. `s3://vecd-private/…`.
    pub fn db_backup(&self) -> Option<&str> {
        self.get("db_backup")
    }
}

/// Write a starter config file (mode 0600) into `dir`, created `0700`.
/// Idempotent: leaves an existing file untouched.
pub fn write_starter(dir: &Path) -> Result<PathBuf, VecdError> {
    std::fs::create_dir_all(dir)?;
    set_mode(dir, 0o700);
    let path = dir.join("vecd.conf");
    if !path.exists() {
        let starter = "# vecd configuration\n\
            # data_dir   = /var/lib/vecd\n\
            # bind       = 0.0.0.0:8443\n\
            # tls_cert   = /etc/vecd/cert.pem\n\
            # tls_key    = /etc/vecd/key.pem\n\
            # db_key     = <SQLCipher key — build with --features sqlcipher>\n\
            # db_backup  = s3://vecd-private/backups/\n";
        std::fs::write(&path, starter)?;
        set_mode(&path, 0o600);
    }
    Ok(path)
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
    fn parse_basic() {
        let c = Config::parse(
            "# comment\nbind = 0.0.0.0:8443\ndb_key = \"sekret\"  # inline\n\n",
        )
        .unwrap();
        assert_eq!(c.get("bind"), Some("0.0.0.0:8443"));
        assert_eq!(c.db_key(), Some("sekret"));
        assert_eq!(c.get("missing"), None);
    }

    #[test]
    fn malformed_line_is_usage_error() {
        assert!(matches!(Config::parse("no equals here"), Err(VecdError::Usage(_))));
    }
}
