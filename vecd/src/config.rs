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

    /// Parse a byte-rate config key (e.g. `ratelimit_connection_download`)
    /// into bytes/sec. Absent → `0` (unlimited). See [`parse_byte_rate`]
    /// for the accepted syntax.
    pub fn byte_rate(&self, key: &str) -> Result<u64, VecdError> {
        match self.get(key) {
            Some(v) => parse_byte_rate(v)
                .map_err(|m| VecdError::usage(format!("vecd.conf '{key}': {m}"))),
            None => Ok(0),
        }
    }
}

/// Parse a human-friendly byte-rate into **bytes per second**.
///
/// Accepts a bare integer (bytes/sec), or an integer/decimal with a unit
/// suffix: binary `KiB`/`MiB`/`GiB` (×1024ⁿ) or decimal `KB`/`MB`/`GB` and
/// the bare `K`/`M`/`G` (×1000ⁿ). Case-insensitive; surrounding and
/// internal spaces are ignored. `0` (any unit) means **unlimited**.
///
/// Examples: `8MiB` → 8 388 608, `1MB` → 1 000 000, `1048576` → 1 048 576,
/// `0` → 0.
pub fn parse_byte_rate(s: &str) -> Result<u64, String> {
    let t = s.trim().replace(' ', "");
    if t.is_empty() {
        return Err("empty rate".to_string());
    }
    let lower = t.to_ascii_lowercase();
    // Longest suffixes first so "kib" isn't shadowed by "k"/"b".
    let units: &[(&str, f64)] = &[
        ("kib", 1024.0),
        ("mib", 1024.0 * 1024.0),
        ("gib", 1024.0 * 1024.0 * 1024.0),
        ("kb", 1000.0),
        ("mb", 1_000_000.0),
        ("gb", 1_000_000_000.0),
        ("k", 1000.0),
        ("m", 1_000_000.0),
        ("g", 1_000_000_000.0),
        ("b", 1.0),
    ];
    let (num_str, mult) = units
        .iter()
        .find(|(suf, _)| lower.ends_with(suf))
        .map(|(suf, m)| (&lower[..lower.len() - suf.len()], *m))
        .unwrap_or((lower.as_str(), 1.0));
    let num: f64 = num_str
        .parse()
        .map_err(|_| format!("invalid byte rate '{s}'"))?;
    if num < 0.0 {
        return Err(format!("negative byte rate '{s}'"));
    }
    Ok((num * mult).round() as u64)
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
            # db_backup  = s3://vecd-private/backups/\n\
            #\n\
            # Bandwidth rate limits (bytes/sec; suffixes KiB/MiB/GiB, KB/MB/GB; 0 = unlimited).\n\
            # Per-connection caps shape each TCP connection — opening more connections scales\n\
            # aggregate throughput. Per-client caps shape the sum across one host's (IP's)\n\
            # connections — concurrency can't exceed the cap. Each --ratelimit-* flag overrides\n\
            # the matching key below.\n\
            # ratelimit_connection_download = 0\n\
            # ratelimit_connection_upload   = 0\n\
            # ratelimit_client_download     = 0\n\
            # ratelimit_client_upload       = 0\n";
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

    #[test]
    fn byte_rate_units() {
        assert_eq!(parse_byte_rate("0"), Ok(0));
        assert_eq!(parse_byte_rate("1048576"), Ok(1024 * 1024));
        assert_eq!(parse_byte_rate("8MiB"), Ok(8 * 1024 * 1024));
        assert_eq!(parse_byte_rate("1MB"), Ok(1_000_000));
        assert_eq!(parse_byte_rate("2 mib"), Ok(2 * 1024 * 1024));
        assert_eq!(parse_byte_rate("512KiB"), Ok(512 * 1024));
        assert_eq!(parse_byte_rate("1G"), Ok(1_000_000_000));
        assert!(parse_byte_rate("nonsense").is_err());
        assert!(parse_byte_rate("-5MiB").is_err());
    }

    #[test]
    fn byte_rate_accessor_defaults_to_zero() {
        let c = Config::parse("ratelimit_connection_download = 8MiB").unwrap();
        assert_eq!(c.byte_rate("ratelimit_connection_download").unwrap(), 8 * 1024 * 1024);
        assert_eq!(c.byte_rate("ratelimit_client_upload").unwrap(), 0);
    }
}
