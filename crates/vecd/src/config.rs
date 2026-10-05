// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Operator configuration and standard paths.
//!
//! `vecd` reads settings an operator shouldn't pass on the command line —
//! the DB encryption key, the data dir / bind address / TLS paths, and the
//! DB-backup destination — from a `vecd.conf` file. CLI flags override the
//! file. The format is minimal (`key = value` lines); secret-bearing files
//! are created `0600`.
//!
//! **Locating `vecd.conf`** ([`resolve`]): `$VECD_CONFIG` (an explicit config
//! dir) wins outright. Otherwise vecd looks in the current directory
//! (`./vecd.conf`) and the home config (`~/.config/vecd/vecd.conf`); if a
//! config exists in exactly one it is used, and if it exists in *both* the
//! caller must disambiguate with `--config-is-local` / `--config-is-home`
//! (or `PREFER_CONFIG=local|home` for scripts). When none exists, every
//! command except `config` and `completions` refuses and points the user at
//! `vecd config auto` — vecd is never run against an unconfigured base path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::model::VecdError;

/// Default config directory: `$HOME/.config/vecd/` (or `$VECD_CONFIG`).
/// Prefer [`resolve`] for command dispatch — this is the plain default used
/// where the local/home tie-break doesn't apply (e.g. computing a write
/// target's data dir).
pub fn config_dir() -> PathBuf {
    if let Ok(c) = std::env::var("VECD_CONFIG") {
        return PathBuf::from(c);
    }
    home_config_dir()
}

/// The home config dir (`$HOME/.config/vecd`), ignoring `$VECD_CONFIG`.
pub fn home_config_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".config").join("vecd")
}

/// Which location a resolved config dir came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    /// The `--conf` flag.
    Flag,
    /// `$VECD_CONFIG` was set.
    Env,
    /// The current directory (`./vecd.conf`).
    Local,
    /// The home config (`~/.config/vecd/vecd.conf`).
    Home,
}

impl ConfigSource {
    /// Human label for messages.
    pub fn label(self) -> &'static str {
        match self {
            ConfigSource::Flag => "--conf",
            ConfigSource::Env => "$VECD_CONFIG",
            ConfigSource::Local => "./vecd.conf",
            ConfigSource::Home => "~/.config/vecd/vecd.conf",
        }
    }
}

/// Disambiguation preference between the local and home configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prefer {
    Local,
    Home,
}

/// `PREFER_CONFIG=local|home`, if set and valid.
pub fn prefer_from_env() -> Option<Prefer> {
    match std::env::var("PREFER_CONFIG").ok().as_deref() {
        Some("local") => Some(Prefer::Local),
        Some("home") => Some(Prefer::Home),
        _ => None,
    }
}

/// A resolved config location: where the dir is, where it came from, and
/// whether a `vecd.conf` actually exists there yet.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub dir: PathBuf,
    pub source: ConfigSource,
    pub exists: bool,
}

impl Resolved {
    /// The `vecd.conf` path at this location.
    pub fn conf_path(&self) -> PathBuf {
        self.dir.join("vecd.conf")
    }
}

/// Resolve the config location (see the module docs). `conf` is the `--conf`
/// flag (a config dir) and wins over `$VECD_CONFIG` — with a warning if both
/// are set. `prefer_flag` comes from `--config-is-local`/`--config-is-home`
/// and falls back to `PREFER_CONFIG`. Errors only when a config exists in
/// *both* the current dir and the home dir with no preference to break the tie.
pub fn resolve(conf: Option<&Path>, prefer_flag: Option<Prefer>) -> Result<Resolved, VecdError> {
    let env_conf = std::env::var("VECD_CONFIG").ok();
    if let Some(conf) = conf {
        if env_conf.is_some() {
            eprintln!(
                "vecd: warning: --conf and $VECD_CONFIG are both set; honoring --conf ({})",
                conf.display()
            );
        }
        if conf.is_file() {
            return Err(VecdError::usage(format!(
                "--conf must be a config directory, but {} is a file. To seed config from a \
                 json/yaml file use `vecd config set --from <file>` or `vecd init --config-import <file>`.",
                conf.display()
            )));
        }
        return Ok(Resolved {
            dir: conf.to_path_buf(),
            source: ConfigSource::Flag,
            exists: conf.join("vecd.conf").is_file(),
        });
    }
    if let Some(c) = env_conf {
        let dir = PathBuf::from(c);
        let exists = dir.join("vecd.conf").is_file();
        return Ok(Resolved { dir, source: ConfigSource::Env, exists });
    }
    let local = PathBuf::from(".");
    let home = home_config_dir();
    let local_exists = local.join("vecd.conf").is_file();
    let home_exists = home.join("vecd.conf").is_file();
    let prefer = prefer_flag.or_else(prefer_from_env);
    resolve_core(local, local_exists, home, home_exists, prefer)
}

/// Pure core of [`resolve`] — the existence flags and preference are inputs,
/// so the tie-break policy is unit-tested without the filesystem or env.
fn resolve_core(
    local: PathBuf,
    local_exists: bool,
    home: PathBuf,
    home_exists: bool,
    prefer: Option<Prefer>,
) -> Result<Resolved, VecdError> {
    let pick = |dir: PathBuf, source: ConfigSource, exists: bool| Ok(Resolved { dir, source, exists });
    match (local_exists, home_exists) {
        (true, true) => match prefer {
            Some(Prefer::Local) => pick(local, ConfigSource::Local, true),
            Some(Prefer::Home) => pick(home, ConfigSource::Home, true),
            None => Err(VecdError::usage(
                "a vecd config exists in BOTH the current directory (./vecd.conf) and your home \
                 config (~/.config/vecd/vecd.conf) — choose one with --config-is-local or \
                 --config-is-home, or set PREFER_CONFIG=local|home (handy for scripts/CI)."
                    .to_string(),
            )),
        },
        (true, false) => pick(local, ConfigSource::Local, true),
        (false, true) => pick(home, ConfigSource::Home, true),
        // Neither exists: hand back the *write target* per preference (home by
        // default) so `config auto` knows where to create the file, and the
        // gate knows nothing is configured yet (`exists: false`).
        (false, false) => match prefer {
            Some(Prefer::Local) => pick(local, ConfigSource::Local, false),
            _ => pick(home, ConfigSource::Home, false),
        },
    }
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

    /// Set `key = value`, validating both (see [`validate_kv`]).
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), VecdError> {
        validate_kv(key, value)?;
        self.values.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Remove a key; returns whether it was present.
    pub fn remove(&mut self, key: &str) -> bool {
        self.values.remove(key).is_some()
    }

    /// Whether `lock_config` is on — when set, config changes are refused
    /// until it is turned off (see the CLI's change guard).
    pub fn is_locked(&self) -> bool {
        self.get("lock_config").and_then(as_bool).unwrap_or(false)
    }

    /// The configured [`DaemonMode`] — [`DaemonMode::Adhoc`] when unset
    /// or unrecognized. Drives whether `start`/`stop`/`status`/`restart`
    /// self-daemonize or delegate to systemd.
    pub fn daemon_mode(&self) -> DaemonMode {
        match self.get("daemon_mode") {
            Some(v) if v.eq_ignore_ascii_case("systemd") => DaemonMode::Systemd,
            _ => DaemonMode::Adhoc,
        }
    }

    /// Validate every key/value (used after a bulk import/load).
    pub fn validate_all(&self) -> Result<(), VecdError> {
        for (k, v) in &self.values {
            validate_kv(k, v)?;
        }
        Ok(())
    }

    /// `(key, value)` pairs in canonical order: known keys in declared order,
    /// then any extras alphabetically.
    pub fn ordered(&self) -> Vec<(&str, &str)> {
        let mut out: Vec<(&str, &str)> = Vec::new();
        for k in KNOWN_KEYS {
            if let Some(v) = self.values.get(*k) {
                out.push((k, v.as_str()));
            }
        }
        let mut extra: Vec<&String> = self.values.keys().filter(|k| !is_known_key(k)).collect();
        extra.sort();
        for k in extra {
            out.push((k.as_str(), self.values[k].as_str()));
        }
        out
    }

    /// Serialize to canonical `key = value` text — the native `vecd.conf`
    /// format, round-trippable through [`parse`](Self::parse).
    pub fn to_text(&self) -> String {
        let mut out =
            String::from("# vecd configuration — see docs/guides/vecd-config.md (or `vecd config get`)\n");
        for (k, v) in self.ordered() {
            out.push_str(&format!("{k} = {v}\n"));
        }
        out
    }

    /// Serialize to a flat JSON object of `key: value` (all values strings).
    pub fn to_json(&self) -> String {
        let map: std::collections::BTreeMap<&str, &str> = self.ordered().into_iter().collect();
        serde_json::to_string_pretty(&map).unwrap_or_else(|_| "{}".to_string())
    }

    /// Serialize to a flat YAML mapping of `key: value`.
    pub fn to_yaml(&self) -> String {
        let map: std::collections::BTreeMap<&str, &str> = self.ordered().into_iter().collect();
        serde_yaml::to_string(&map).unwrap_or_default()
    }

    /// Write this config as `<dir>/vecd.conf` (dir `0700`, file `0600`), in
    /// the native format. Returns the file path.
    pub fn write_to(&self, dir: &Path) -> Result<PathBuf, VecdError> {
        std::fs::create_dir_all(dir)?;
        set_mode(dir, 0o700);
        let path = dir.join("vecd.conf");
        std::fs::write(&path, self.to_text())?;
        set_mode(&path, 0o600);
        Ok(path)
    }
}

/// Every recognized `vecd.conf` key, in canonical display order. `set`/`get`/
/// `import`/`load` validate against this; it mirrors the `vecd serve` flags.
pub const KNOWN_KEYS: &[&str] = &[
    "data_dir",
    "bind",
    "tls_cert",
    "tls_key",
    "db_key",
    "db_backup",
    "ratelimit_connection_download",
    "ratelimit_connection_upload",
    "ratelimit_client_download",
    "ratelimit_client_upload",
    "daemon_mode",
    "lock_config",
];

/// Whether `key` is a recognized config key.
pub fn is_known_key(key: &str) -> bool {
    KNOWN_KEYS.contains(&key)
}

/// How the background lifecycle verbs (`start`/`stop`/`status`/
/// `restart`) behave, per the `daemon_mode` config key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonMode {
    /// Self-daemonize via `fork`/`setsid` — the default (see
    /// [`crate::daemon`]).
    Adhoc,
    /// Delegate to `systemctl`; set by `vecd daemon install`.
    Systemd,
}

impl DaemonMode {
    /// The canonical `daemon_mode` config-value spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonMode::Adhoc => "adhoc",
            DaemonMode::Systemd => "systemd",
        }
    }
}

/// Parse a boolean-ish config value: `on/true/1/yes` → `true`,
/// `off/false/0/no` → `false` (case-insensitive). Used by `lock_config`.
pub fn as_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Some(true),
        "off" | "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

/// Validate a `key = value` pair: the key must be known, and known keys get a
/// light value check so a typo (`bind = nonsense`, `ratelimit_… = blah`) is
/// caught at set/import time rather than at server start.
pub fn validate_kv(key: &str, value: &str) -> Result<(), VecdError> {
    if !is_known_key(key) {
        return Err(VecdError::usage(format!(
            "unknown config key '{key}'. Known keys: {}",
            KNOWN_KEYS.join(", ")
        )));
    }
    match key {
        "bind" => {
            value.parse::<std::net::SocketAddr>().map_err(|e| {
                VecdError::usage(format!("config 'bind' = '{value}': not a host:port ({e})"))
            })?;
        }
        k if k.starts_with("ratelimit_") => {
            parse_byte_rate(value).map_err(|m| VecdError::usage(format!("config '{key}': {m}")))?;
        }
        "lock_config" => {
            as_bool(value).ok_or_else(|| {
                VecdError::usage(format!("config 'lock_config' = '{value}': expected on/off"))
            })?;
        }
        "daemon_mode" => {
            match value.trim().to_ascii_lowercase().as_str() {
                "adhoc" | "systemd" => {}
                _ => {
                    return Err(VecdError::usage(format!(
                        "config 'daemon_mode' = '{value}': expected adhoc or systemd"
                    )));
                }
            }
        }
        _ => {} // paths / secret keys accepted as-is
    }
    Ok(())
}

/// Reasonable starter settings for `config auto`: the safe, local-only bind
/// and the data dir pinned under the config dir. TLS / backups / rate limits
/// stay unset (off / unlimited). The data dir is recorded explicitly so the
/// base path is pinned regardless of where the config later moves.
pub fn auto_defaults(config_dir: &Path) -> Config {
    let mut c = Config::default();
    let _ = c.set("bind", "127.0.0.1:8443");
    let _ = c.set("data_dir", &config_dir.join("data").to_string_lossy());
    c
}

/// Import config from `path`: a `.json` / `.yaml` / `.yml` file, a `vecd.conf`
/// (or any other file, parsed as the native format), or a **directory** (its
/// `vecd.conf`). Keys/values are validated. Numbers and booleans in JSON/YAML
/// are stringified, so `8388608` and `"8MiB"` both work for a byte-rate key.
pub fn import_from(path: &Path) -> Result<Config, VecdError> {
    if path.is_dir() {
        let cfg = Config::load(path)?;
        cfg.validate_all()?;
        return Ok(cfg);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| VecdError::usage(format!("reading config import {}: {e}", path.display())))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let cfg = match ext.as_str() {
        "json" => config_from_pairs(json_pairs(&text)?)?,
        "yaml" | "yml" => config_from_pairs(yaml_pairs(&text)?)?,
        _ => Config::parse(&text)?, // native key = value
    };
    cfg.validate_all()?;
    Ok(cfg)
}

/// Export `cfg` to `path` by extension: `.json`, `.yaml`/`.yml`, a directory
/// (native `vecd.conf` inside), or `-` for native text on stdout. Other file
/// names get the native format.
pub fn export_to(cfg: &Config, path: &Path) -> Result<(), VecdError> {
    if path.as_os_str() == "-" {
        print!("{}", cfg.to_text());
        return Ok(());
    }
    if path.is_dir() {
        cfg.write_to(path)?;
        return Ok(());
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let text = match ext.as_str() {
        "json" => cfg.to_json(),
        "yaml" | "yml" => cfg.to_yaml(),
        _ => cfg.to_text(),
    };
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)?;
    Ok(())
}

/// Build a [`Config`] from already-extracted `(key, value)` pairs (validating).
fn config_from_pairs(pairs: Vec<(String, String)>) -> Result<Config, VecdError> {
    let mut c = Config::default();
    for (k, v) in pairs {
        c.set(&k, &v)?;
    }
    Ok(c)
}

/// Flatten a JSON object into `(key, string-value)` pairs. Rejects non-object
/// roots and nested/array values (config is flat & scalar).
fn json_pairs(text: &str) -> Result<Vec<(String, String)>, VecdError> {
    let v: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| VecdError::usage(format!("invalid JSON config: {e}")))?;
    let obj = v.as_object().ok_or_else(|| {
        VecdError::usage("JSON config must be an object of key: value".to_string())
    })?;
    let mut out = Vec::new();
    for (k, val) in obj {
        out.push((k.clone(), json_scalar(k, val)?));
    }
    Ok(out)
}

fn json_scalar(key: &str, v: &serde_json::Value) -> Result<String, VecdError> {
    match v {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        _ => Err(VecdError::usage(format!(
            "config '{key}' must be a scalar (string/number/bool), not an array or object"
        ))),
    }
}

/// Flatten a YAML mapping into `(key, string-value)` pairs (same rules as JSON).
fn yaml_pairs(text: &str) -> Result<Vec<(String, String)>, VecdError> {
    let v: serde_yaml::Value = serde_yaml::from_str(text)
        .map_err(|e| VecdError::usage(format!("invalid YAML config: {e}")))?;
    let map = v.as_mapping().ok_or_else(|| {
        VecdError::usage("YAML config must be a mapping of key: value".to_string())
    })?;
    let mut out = Vec::new();
    for (k, val) in map {
        let key = k.as_str().ok_or_else(|| {
            VecdError::usage("YAML config keys must be strings".to_string())
        })?;
        out.push((key.to_string(), yaml_scalar(key, val)?));
    }
    Ok(out)
}

fn yaml_scalar(key: &str, v: &serde_yaml::Value) -> Result<String, VecdError> {
    match v {
        serde_yaml::Value::String(s) => Ok(s.clone()),
        serde_yaml::Value::Number(n) => Ok(n.to_string()),
        serde_yaml::Value::Bool(b) => Ok(b.to_string()),
        _ => Err(VecdError::usage(format!(
            "config '{key}' must be a scalar (string/number/bool), not a sequence or mapping"
        ))),
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
    fn resolve_core_tie_break() {
        let local = || PathBuf::from(".");
        let home = || PathBuf::from("/home/u/.config/vecd");
        // Only one present → use it, no preference needed.
        assert_eq!(
            resolve_core(local(), true, home(), false, None).unwrap().source,
            ConfigSource::Local
        );
        assert_eq!(
            resolve_core(local(), false, home(), true, None).unwrap().source,
            ConfigSource::Home
        );
        // Both present, no preference → ambiguity error.
        assert!(resolve_core(local(), true, home(), true, None).is_err());
        // Both present, preference breaks the tie.
        assert_eq!(
            resolve_core(local(), true, home(), true, Some(Prefer::Home)).unwrap().source,
            ConfigSource::Home
        );
        // Neither present → home is the default write target, marked absent.
        let none = resolve_core(local(), false, home(), false, None).unwrap();
        assert_eq!(none.source, ConfigSource::Home);
        assert!(!none.exists);
        // …unless local is preferred.
        assert_eq!(
            resolve_core(local(), false, home(), false, Some(Prefer::Local)).unwrap().source,
            ConfigSource::Local
        );
    }

    #[test]
    fn set_get_validate_and_roundtrip() {
        let mut c = Config::default();
        c.set("bind", "0.0.0.0:9000").unwrap();
        c.set("ratelimit_client_download", "8MiB").unwrap();
        assert_eq!(c.get("bind"), Some("0.0.0.0:9000"));
        // Unknown key and bad values are rejected at set time.
        assert!(c.set("nope", "x").is_err());
        assert!(c.set("bind", "garbage").is_err());
        assert!(c.set("ratelimit_client_upload", "fast").is_err());
        // Native text round-trips through parse.
        let re = Config::parse(&c.to_text()).unwrap();
        assert_eq!(re.get("bind"), Some("0.0.0.0:9000"));
        assert_eq!(re.get("ratelimit_client_download"), Some("8MiB"));
    }

    #[test]
    fn json_and_yaml_import_are_equivalent() {
        let from_json = config_from_pairs(json_pairs(
            r#"{"bind":"127.0.0.1:8443","ratelimit_client_download":8388608}"#,
        ).unwrap()).unwrap();
        let from_yaml = config_from_pairs(yaml_pairs(
            "bind: 127.0.0.1:8443\nratelimit_client_download: 8388608\n",
        ).unwrap()).unwrap();
        // Numbers are stringified, so both match.
        assert_eq!(from_json.get("ratelimit_client_download"), Some("8388608"));
        assert_eq!(from_json.ordered(), from_yaml.ordered());
        // A nested value is rejected (config is flat).
        assert!(json_pairs(r#"{"bind":{"host":"x"}}"#).is_err());
    }

    #[test]
    fn lock_config_flag() {
        assert_eq!(as_bool("on"), Some(true));
        assert_eq!(as_bool("OFF"), Some(false));
        assert_eq!(as_bool("maybe"), None);
        let mut c = Config::default();
        assert!(!c.is_locked());
        c.set("lock_config", "on").unwrap();
        assert!(c.is_locked());
        c.set("lock_config", "off").unwrap();
        assert!(!c.is_locked());
        assert!(c.set("lock_config", "perhaps").is_err());
    }

    #[test]
    fn daemon_mode_default_set_validate() {
        let mut c = Config::default();
        // Default (unset) is adhoc.
        assert_eq!(c.daemon_mode(), DaemonMode::Adhoc);
        c.set("daemon_mode", "systemd").unwrap();
        assert_eq!(c.daemon_mode(), DaemonMode::Systemd);
        // Case-insensitive read, canonical spelling round-trips.
        c.set("daemon_mode", "ADHOC").unwrap();
        assert_eq!(c.daemon_mode(), DaemonMode::Adhoc);
        assert_eq!(DaemonMode::Systemd.as_str(), "systemd");
        // Bad value rejected at set time.
        assert!(c.set("daemon_mode", "upstart").is_err());
    }

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
