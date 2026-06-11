// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-user `vectordata` settings.
//!
//! Loaded from `~/.config/vectordata/settings.yaml` (or
//! `$VECTORDATA_HOME/settings.yaml` when that env var is set —
//! used by tests). The single source of truth for cache-directory
//! resolution: every public reader path that fetches remote data
//! routes through here, so the user's `cache_dir:` setting is
//! honored uniformly across vectordata and any consuming crate.
//!
//! There is no silent fallback. If the settings file is missing or
//! does not declare `cache_dir:`, [`cache_dir`] returns
//! [`SettingsError::NotConfigured`], whose `Display` impl prints a
//! ready-to-paste set of commands the user can run to configure it.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::mounts;

const CONFIG_DIR: &str = ".config/vectordata";
const SETTINGS_FILE: &str = "settings.yaml";

/// Process-wide override for the cache directory. When set, [`cache_dir`]
/// returns this value verbatim and never reads `settings.yaml`. Intended
/// for tests that need to isolate cache state in a tempdir without
/// touching the user's real configuration or racing on `$VECTORDATA_HOME`
/// (which is a process-wide env var and tests share a process when run
/// with `cargo test`).
static CACHE_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Install a process-wide override for [`cache_dir`]. First call wins;
/// subsequent calls are silent no-ops so racing test initializers
/// converge on a single value instead of panicking.
///
/// Use this from a one-time test initializer (a `LazyLock<TempDir>` or
/// equivalent) — never from production code. There is intentionally no
/// way to clear the override once set, so tests cannot stomp on each
/// other by alternating overrides.
pub fn override_cache_dir_for_process(path: PathBuf) {
    let _ = CACHE_DIR_OVERRIDE.set(path);
}

/// Settings resolution failure.
#[derive(Debug)]
pub enum SettingsError {
    /// `settings.yaml` is missing, or present but lacks a `cache_dir:`
    /// entry. The `Display` impl carries actionable instructions for
    /// the user — surface it directly without paraphrasing.
    NotConfigured {
        /// Where we looked.
        settings_path: PathBuf,
        /// Whether the file existed (true ⇒ it just lacks `cache_dir:`).
        file_exists: bool,
    },
}

impl std::fmt::Display for SettingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettingsError::NotConfigured { settings_path, file_exists } => {
                let why = if *file_exists {
                    format!("{} exists but does not declare 'cache_dir:'.",
                        settings_path.display())
                } else {
                    format!("{} does not exist.", settings_path.display())
                };
                let parent = settings_path.parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "$HOME/.config/vectordata".to_string());

                // Pull a concrete recommendation from the live mount
                // table when possible. If `auto` would land on $HOME
                // (the auto-bootstrap path that *should* have already
                // fired) we don't end up here; mentioning `auto`
                // still makes sense for callers that constructed
                // `NotConfigured` synthetically (e.g. tests) or for
                // future readers writing the file by hand.
                let auto_hint = match auto_resolved_cache_dir() {
                    Ok(a) => format!("auto would pick: {}", a.path.display()),
                    Err(_) => "auto-resolution unavailable on this host".into(),
                };
                let example_path = match auto_resolved_cache_dir() {
                    Ok(a) => a.path.display().to_string(),
                    Err(_) => "/mnt/example/vectordata-cache".into(),
                };

                write!(f,
                    "vectordata is not configured: {why}\n\
                     \n\
                     # Pick the largest writable mount automatically:\n\
                     vectordata config set cache auto\n\
                     ({auto_hint})\n\
                     \n\
                     # Or set an explicit path:\n\
                     vectordata config set cache <path>\n\
                     # (veks alias: `veks datasets config set cache <path>`)\n\
                     \n\
                     # Or write the file by hand:\n\
                     mkdir -p {parent}\n\
                     cat > {} <<'YAML'\n\
                     cache_dir: {example_path}\n\
                     protect_settings: true\n\
                     YAML",
                    settings_path.display(),
                )
            }
        }
    }
}

impl std::error::Error for SettingsError {}

/// Failure modes for [`write_cache_dir`].
#[derive(Debug)]
pub enum SettingsWriteError {
    /// `settings.yaml` already exists with `protect_settings: true`
    /// and the caller did not pass `force=true`. Refusing to overwrite
    /// is the safe default; callers should surface this clearly so
    /// users know which flag unlocks the operation.
    Protected {
        /// Where the existing settings live.
        settings_path: PathBuf,
        /// The cache_dir currently recorded there (if any).
        existing_cache_dir: Option<PathBuf>,
    },
    /// Filesystem failure: directory creation, file write, or read of
    /// the existing settings file. The wrapped [`io::Error`] carries
    /// the underlying cause.
    Io(io::Error),
}

impl std::fmt::Display for SettingsWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettingsWriteError::Protected { settings_path, existing_cache_dir } => {
                let existing = existing_cache_dir.as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(not set)".to_string());
                write!(f,
                    "{} already exists with `protect_settings: true` \
                     (current cache_dir: {existing}). \
                     Re-run with --force to overwrite.",
                    settings_path.display())
            }
            SettingsWriteError::Io(e) => write!(f, "settings I/O error: {e}"),
        }
    }
}

impl std::error::Error for SettingsWriteError {}

impl From<io::Error> for SettingsWriteError {
    fn from(e: io::Error) -> Self { SettingsWriteError::Io(e) }
}

/// Result of a successful [`write_cache_dir`]. Distinguishes a
/// real write from a same-value no-op so the CLI can print
/// "Configuration updated" vs "Already set" without re-doing the
/// equality check itself.
#[derive(Debug, PartialEq, Eq)]
pub enum WriteCacheOutcome {
    /// `settings.yaml` was written with a new `cache_dir:` value.
    Wrote,
    /// The requested value already matched the existing one —
    /// `settings.yaml` was not touched.
    AlreadySet,
}

/// Path to the `settings.yaml` file. `$VECTORDATA_HOME` takes
/// precedence over `$HOME/.config/vectordata/` so tests can isolate
/// without touching the user's real config.
pub fn settings_path() -> PathBuf {
    if let Some(root) = std::env::var_os("VECTORDATA_HOME") {
        PathBuf::from(root).join(SETTINGS_FILE)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(CONFIG_DIR).join(SETTINGS_FILE)
    } else {
        PathBuf::from(CONFIG_DIR).join(SETTINGS_FILE)
    }
}

/// Resolve the configured cache directory.
///
/// Returns the `cache_dir:` value from the user's `settings.yaml`.
/// If the file is missing or does not declare `cache_dir:`:
///
/// - When `$VECTORDATA_HOME` is set, the cache defaults to
///   `$VECTORDATA_HOME/cache`. That env var already roots the config
///   ([`settings_path`]), so it is a complete isolation boundary —
///   pointing it at a throwaway directory isolates *all* client state
///   (config, credentials, and cache) under one root, never touching
///   the user's real `~/.config` or `~/.cache`.
/// - Otherwise this function *auto-bootstraps* to `$HOME/.cache/vectordata`
///   — **only** when the largest writable mount visible to this process
///   is on the same filesystem as `$HOME` (so we aren't silently
///   burying a cache on a small volume when there's a larger disk the
///   user should be choosing). The bootstrap writes the resolved
///   value to `settings.yaml` and prints a one-line warning to stderr
///   the first time it runs.
///
/// If neither applies (`$HOME` isn't on the largest writable mount,
/// `$HOME` unreadable, etc.), returns [`SettingsError::NotConfigured`] —
/// its `Display` impl carries ready-to-paste commands.
///
/// If [`override_cache_dir_for_process`] has been called, that value
/// takes precedence and `settings.yaml` is not consulted.
pub fn cache_dir() -> Result<PathBuf, SettingsError> {
    if let Some(p) = CACHE_DIR_OVERRIDE.get() {
        return Ok(p.clone());
    }
    let path = settings_path();
    match cache_dir_from(&path) {
        Ok(p) => Ok(p),
        // An explicit `cache_dir:` wins; only fall back when unconfigured.
        // `$VECTORDATA_HOME` isolates the cache alongside the config it
        // already roots, without the mount-table auto-bootstrap dance.
        Err(e) => match vectordata_home_cache_from(std::env::var_os("VECTORDATA_HOME")) {
            Some(dir) => Ok(dir),
            None => try_auto_bootstrap(&path).ok_or(e),
        },
    }
}

/// Pure resolver: `<vectordata_home>/cache` when `$VECTORDATA_HOME` is set,
/// else `None`. Tested directly so the env-var mapping needs no env mutation.
fn vectordata_home_cache_from(vectordata_home: Option<std::ffi::OsString>) -> Option<PathBuf> {
    vectordata_home.map(|root| PathBuf::from(root).join("cache"))
}

/// Why [`auto_resolved_cache_dir`] picked the path it picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheDirResolutionReason {
    /// Largest writable mount is the same filesystem as `$HOME` — we
    /// picked `$HOME/.cache/vectordata` because there's no bigger
    /// disk to put it on anyway.
    HomeIsLargestMount,
    /// Largest writable mount is on a *different* filesystem than
    /// `$HOME`. The path is `<that-mount>/vectordata-cache`. The
    /// auto-bootstrap path in [`cache_dir`] refuses this case — it
    /// requires the user to explicitly opt in via `set-cache` so we
    /// don't silently land cache data somewhere unexpected.
    DifferentMountIsLargest,
    /// The mount table was unusable — no writable mounts visible
    /// (containers, restricted /proc, exotic platforms) or device
    /// inspection failed. `$HOME` demonstrably exists (the process
    /// is running with one), so the XDG default
    /// `~/.cache/vectordata` is the answer; a missing mount table
    /// must never block `config set cache auto`.
    MountTableUnavailable,
}

/// A cache-directory candidate auto-picked from the live mount table.
#[derive(Debug, Clone)]
pub struct AutoResolved {
    pub path: PathBuf,
    pub reason: CacheDirResolutionReason,
    /// The mount-point string that was picked (for diagnostic
    /// messages).
    pub mount: String,
}

/// Pick a cache directory based on the live mount table. Shared by
/// `set-cache auto` (which honors the choice unconditionally) and
/// the [`cache_dir`] auto-bootstrap (which only honors
/// [`CacheDirResolutionReason::HomeIsLargestMount`]).
///
/// Bubbles up a printable error when `$HOME` isn't set, no writable
/// mounts are visible, or `statvfs`/`stat` fail.
pub fn auto_resolved_cache_dir() -> Result<AutoResolved, String> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| "HOME is not set; cannot auto-resolve cache_dir".to_string())?;
    let home = PathBuf::from(home);

    // The mount-table walk below is an *optimization* — it exists to
    // find a bigger data disk than $HOME's filesystem. When the
    // table is unusable (no writable mounts visible, device
    // inspection failing — containers and minimal cloud images get
    // here), the XDG default under $HOME is the answer, not an
    // error.
    let xdg_default = || AutoResolved {
        path: xdg_cache_home(&home).join("vectordata"),
        reason: CacheDirResolutionReason::MountTableUnavailable,
        mount: home.display().to_string(),
    };

    let Ok(home_dev) = mounts::device_id(&home) else {
        return Ok(xdg_default());
    };

    // Walk rw-mounted filesystems from largest to smallest. The
    // first acceptable one wins:
    //   - same filesystem as $HOME → the XDG default under $HOME
    //     (no bigger disk to prefer);
    //   - a different filesystem → `<mount>/vectordata-cache`, but
    //     only when the current user can actually create that
    //     directory at the mount root (a root-owned `/` is
    //     rw-mounted yet unusable for this).
    for candidate in mounts::enumerate().into_iter().filter(|m| m.writable) {
        let path = Path::new(&candidate.path);
        let Ok(dev) = mounts::device_id(path) else { continue };
        if dev == home_dev {
            return Ok(AutoResolved {
                path: xdg_cache_home(&home).join("vectordata"),
                reason: CacheDirResolutionReason::HomeIsLargestMount,
                mount: candidate.path,
            });
        }
        if mounts::is_writable(path) {
            return Ok(AutoResolved {
                path: PathBuf::from(&candidate.path).join("vectordata-cache"),
                reason: CacheDirResolutionReason::DifferentMountIsLargest,
                mount: candidate.path,
            });
        }
    }
    // Mount table empty, unusable, or nothing acceptable — $HOME
    // exists, so the XDG default is the answer, never an error.
    Ok(xdg_default())
}

/// The XDG cache base: `$XDG_CACHE_HOME` when set (absolute), else
/// `<home>/.cache` per the XDG Base Directory spec.
fn xdg_cache_home(home: &Path) -> PathBuf {
    xdg_cache_home_from(std::env::var_os("XDG_CACHE_HOME"), home)
}

/// Pure core of [`xdg_cache_home`]: the env value arrives as a
/// parameter so tests don't mutate process env.
fn xdg_cache_home_from(xdg: Option<std::ffi::OsString>, home: &Path) -> PathBuf {
    match xdg {
        Some(p) if Path::new(&p).is_absolute() => PathBuf::from(p),
        _ => home.join(".cache"),
    }
}

/// Try to auto-bootstrap `settings.yaml` on first read. Returns
/// `Some(path)` only when the resolution rule fires
/// (`HomeIsLargestMount` / `MountTableUnavailable` — both resolve
/// under `$HOME`) and the write succeeds. Prints a one-line
/// stderr warning on success — the user sees what just happened.
///
/// On any failure (no `$HOME`, different mount is largest, write
/// fails) returns `None` so the caller surfaces the original
/// [`SettingsError`].
fn try_auto_bootstrap(settings: &Path) -> Option<PathBuf> {
    let resolved = auto_resolved_cache_dir().ok()?;
    if !matches!(resolved.reason,
        CacheDirResolutionReason::HomeIsLargestMount
        | CacheDirResolutionReason::MountTableUnavailable)
    {
        return None;
    }
    match write_cache_dir_at(settings, &resolved.path, false) {
        Ok(WriteCacheOutcome::Wrote) => {
            eprintln!("warning: no cache_dir configured; auto-set to {} \
                (largest writable mount is $HOME's filesystem). \
                To choose a different location, run \
                `vectordata config set cache <path>`.",
                resolved.path.display());
            Some(resolved.path)
        }
        Ok(WriteCacheOutcome::AlreadySet) => Some(resolved.path),
        Err(_) => None,
    }
}

/// Variant of [`cache_dir`] that reads from an explicit settings
/// file path — used by tests so they don't have to mutate the
/// process-wide `$VECTORDATA_HOME` env var (which races other tests
/// in the same process).
pub fn cache_dir_from(settings: &Path) -> Result<PathBuf, SettingsError> {
    let file_exists = settings.is_file();
    if let Ok(content) = std::fs::read_to_string(settings) {
        for line in content.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("cache_dir:") {
                let val = val.trim().trim_matches('"').trim_matches('\'');
                if !val.is_empty() {
                    return Ok(PathBuf::from(val));
                }
            }
        }
    }
    Err(SettingsError::NotConfigured {
        settings_path: settings.to_path_buf(),
        file_exists,
    })
}

/// Whether `settings.yaml` declares `protect_settings: true`. Used by
/// [`write_cache_dir`] (and surfaced to callers like
/// `vectordata config get`) so users can see at a glance that the
/// file is guarded against accidental overwrites.
///
/// Returns `false` if the file does not exist or does not declare the
/// flag — matching the historical default which only set it on
/// explicit `set-cache` writes.
pub fn protect_settings() -> bool {
    protect_settings_from(&settings_path())
}

/// Variant of [`protect_settings`] reading from an explicit path.
pub fn protect_settings_from(settings: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(settings) else { return false; };
    for line in content.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("protect_settings:") {
            return val.trim() == "true";
        }
    }
    false
}

/// Client TLS-trust policy, read from `settings.yaml`.
///
/// ```yaml
/// # Extra CA / leaf certificates (PEM) to ADD to the system trust roots —
/// # the secure way to trust a private vecd (export its cert with `vecd tls
/// # export`). Verification stays on.
/// trusted_ca_certs:
///   - /home/me/.config/vectordata/vecd-ca.pem
/// # Endpoint origins for which an invalid/self-signed server cert is accepted
/// # WITHOUT verification — insecure, off by default. Use only for local dev.
/// trust_self_signed:
///   - https://127.0.0.1:18443
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TlsTrust {
    /// PEM files added to the client's trusted roots (verification stays on).
    pub trusted_ca_certs: Vec<PathBuf>,
    /// Origins (`scheme://host[:port]`) whose server cert is accepted without
    /// verification — insecure; empty by default.
    pub trust_self_signed: Vec<String>,
}

/// Read the [`TlsTrust`] policy from the user's `settings.yaml` (empty when the
/// file is absent or declares neither key).
pub fn tls_trust() -> TlsTrust {
    tls_trust_from(&settings_path())
}

/// [`tls_trust`] reading from an explicit path. Pure given the file content, so
/// tests pass a tempdir path rather than mutating process env.
pub fn tls_trust_from(settings: &Path) -> TlsTrust {
    let Ok(content) = std::fs::read_to_string(settings) else { return TlsTrust::default() };
    let doc: serde_yaml::Value = serde_yaml::from_str(&content).unwrap_or(serde_yaml::Value::Null);
    let strings = |key: &str| -> Vec<String> {
        doc.get(key)
            .and_then(|v| v.as_sequence())
            .map(|seq| seq.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect())
            .unwrap_or_default()
    };
    TlsTrust {
        trusted_ca_certs: strings("trusted_ca_certs").into_iter().map(PathBuf::from).collect(),
        trust_self_signed: strings("trust_self_signed"),
    }
}

/// Write a new `cache_dir:` into the user's `settings.yaml`,
/// creating the file and its parent directory as needed. Always
/// writes `protect_settings: true` (for backwards compat with older
/// readers that still inspect the flag — the writer itself no
/// longer uses it as a gate).
///
/// Refuses to overwrite an existing `cache_dir:` with a *different*
/// value unless `force=true`. A second call with the *same* value
/// is a no-op (idempotent — does not touch the file). Returns
/// [`SettingsWriteError::Protected`] with the existing value so the
/// caller can show the user what's currently configured.
///
/// Also creates `cache_dir` itself if it doesn't exist — having a
/// `cache_dir:` value pointing at a non-existent path is a common
/// support trap, so this constructor closes that gap.
pub fn write_cache_dir(cache_dir: &Path, force: bool) -> Result<WriteCacheOutcome, SettingsWriteError> {
    write_cache_dir_at(&settings_path(), cache_dir, force)
}

/// Variant of [`write_cache_dir`] that targets an explicit settings
/// path — used by tests so they can exercise the writer without
/// stomping on `$HOME`.
///
/// Behavior when `settings` already exists:
///
/// - Existing `cache_dir:` equals the requested one → no-op (idempotent).
/// - Different value, `force == false` → [`SettingsWriteError::Protected`].
/// - Different value, `force == true` → overwrites unconditionally.
///
/// The `protect_settings:` flag in the file is *no longer load-bearing*
/// for this decision — every write is protected against silent
/// overwrite. The flag is still written so older readers see the
/// expected shape.
pub fn write_cache_dir_at(
    settings: &Path,
    cache_dir: &Path,
    force: bool,
) -> Result<WriteCacheOutcome, SettingsWriteError> {
    if settings.is_file() {
        let existing = cache_dir_from(settings).ok();
        // Same value → no-op; don't even touch the file (avoids
        // gratuitous mtime bumps that would invalidate any
        // staleness-tracked downstream).
        if existing.as_deref() == Some(cache_dir) {
            return Ok(WriteCacheOutcome::AlreadySet);
        }
        if !force {
            return Err(SettingsWriteError::Protected {
                settings_path: settings.to_path_buf(),
                existing_cache_dir: existing,
            });
        }
    }

    if let Some(parent) = settings.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir_all(cache_dir)?;

    let content = if settings.is_file() {
        // Forced overwrite of an existing file: edit ONLY the
        // `cache_dir:` line and keep every other byte — comments and
        // any additional keys are the user's, not ours to discard.
        let existing = std::fs::read_to_string(settings)?;
        rewrite_cache_dir_line(&existing, cache_dir)
    } else {
        format!(
            "cache_dir: {}\nprotect_settings: true\n",
            cache_dir.display(),
        )
    };
    std::fs::write(settings, content)?;
    Ok(WriteCacheOutcome::Wrote)
}

/// Replace the value of the top-level `cache_dir:` line in an existing
/// `settings.yaml`, preserving every other line byte-for-byte
/// (comments, unknown keys, formatting). Appends the line when no
/// uncommented `cache_dir:` exists.
fn rewrite_cache_dir_line(existing: &str, cache_dir: &Path) -> String {
    let mut replaced = false;
    let mut lines: Vec<String> = existing.lines()
        .map(|line| {
            let t = line.trim_start();
            if !replaced && t.starts_with("cache_dir:") && !t.starts_with('#') {
                replaced = true;
                format!("cache_dir: {}", cache_dir.display())
            } else {
                line.to_string()
            }
        })
        .collect();
    if !replaced {
        lines.push(format!("cache_dir: {}", cache_dir.display()));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Auto-resolution lands on the XDG default under $HOME — with
    /// `$XDG_CACHE_HOME` honored when absolute, ignored otherwise.
    #[test]
    fn xdg_cache_home_resolution() {
        let home = Path::new("/home/u");
        assert_eq!(xdg_cache_home_from(None, home), PathBuf::from("/home/u/.cache"));
        // "Absolute" is platform-defined: a Unix-style path has no
        // drive letter, so on Windows it is NOT absolute and must be
        // ignored like any other relative value.
        #[cfg(unix)]
        let abs = "/fast/cache";
        #[cfg(windows)]
        let abs = r"C:\fast\cache";
        assert_eq!(
            xdg_cache_home_from(Some(abs.into()), home),
            PathBuf::from(abs));
        // Relative XDG_CACHE_HOME must be ignored per the XDG spec.
        assert_eq!(
            xdg_cache_home_from(Some("relative".into()), home),
            PathBuf::from("/home/u/.cache"));
        #[cfg(windows)]
        assert_eq!(
            xdg_cache_home_from(Some("/unix/style/not/absolute/here".into()), home),
            PathBuf::from("/home/u/.cache"));
    }

    /// A forced `cache_dir` overwrite must edit only the
    /// `cache_dir:` line — comments and unrelated keys in
    /// `settings.yaml` are the user's and were previously destroyed
    /// by the whole-file template rewrite.
    #[test]
    fn forced_cache_dir_write_preserves_comments_and_other_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let settings = tmp.path().join("settings.yaml");
        let new_cache = tmp.path().join("cache");
        std::fs::write(&settings, "\
# tuned by hand — do not lose this comment
cache_dir: /old/path
protect_settings: true
# experimental:
# download_concurrency: 32
").unwrap();

        let outcome = write_cache_dir_at(&settings, &new_cache, true).unwrap();
        assert!(matches!(outcome, WriteCacheOutcome::Wrote));
        let content = std::fs::read_to_string(&settings).unwrap();
        assert!(content.contains("# tuned by hand — do not lose this comment"));
        assert!(content.contains("# download_concurrency: 32"));
        assert!(content.contains("protect_settings: true"));
        assert!(content.contains(&format!("cache_dir: {}", new_cache.display())));
        assert!(!content.contains("/old/path"));
    }

    #[test]
    fn tls_trust_parses_both_lists() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("settings.yaml");
        std::fs::write(
            &f,
            "cache_dir: /x\n\
             trusted_ca_certs:\n  - /a/ca.pem\n  - /b/ca.pem\n\
             trust_self_signed:\n  - https://127.0.0.1:18443\n",
        )
        .unwrap();
        let t = tls_trust_from(&f);
        assert_eq!(
            t.trusted_ca_certs,
            vec![PathBuf::from("/a/ca.pem"), PathBuf::from("/b/ca.pem")]
        );
        assert_eq!(t.trust_self_signed, vec!["https://127.0.0.1:18443".to_string()]);

        // A file without the keys, or no file at all, yields the empty policy.
        std::fs::write(&f, "cache_dir: /x\n").unwrap();
        assert_eq!(tls_trust_from(&f), TlsTrust::default());
        assert_eq!(tls_trust_from(&tmp.path().join("absent.yaml")), TlsTrust::default());
    }

    #[test]
    fn cache_dir_honors_settings_override() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = tmp.path().join("settings.yaml");
        std::fs::write(&yaml,
            "cache_dir: /mnt/datamir/vectordata-cache\nprotect_settings: true\n",
        ).unwrap();
        assert_eq!(cache_dir_from(&yaml).unwrap(),
            PathBuf::from("/mnt/datamir/vectordata-cache"));
    }

    #[test]
    fn vectordata_home_defaults_cache_under_the_home() {
        // When $VECTORDATA_HOME is set and settings.yaml pins no cache_dir,
        // the cache lands at <home>/cache — so isolating the home isolates
        // the cache too. Tested via the pure resolver (no env mutation).
        assert_eq!(
            vectordata_home_cache_from(Some(std::ffi::OsString::from("/tmp/iso-home"))),
            Some(PathBuf::from("/tmp/iso-home/cache"))
        );
        assert_eq!(vectordata_home_cache_from(None), None);
    }

    #[test]
    fn cache_dir_quoted_value_is_unwrapped() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = tmp.path().join("settings.yaml");
        std::fs::write(&yaml, "cache_dir: \"/some/quoted/path\"\n").unwrap();
        assert_eq!(cache_dir_from(&yaml).unwrap(), PathBuf::from("/some/quoted/path"));
    }

    #[test]
    fn cache_dir_errors_when_settings_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let nope = tmp.path().join("absent.yaml");
        let err = cache_dir_from(&nope).unwrap_err();
        match err {
            SettingsError::NotConfigured { ref settings_path, file_exists } => {
                assert_eq!(settings_path, &nope);
                assert!(!file_exists);
            }
        }
        // Display message must contain the actionable commands so
        // callers can surface it directly.
        let msg = err.to_string();
        assert!(msg.contains("vectordata config set cache auto"),
            "missing auto-pick hint: {msg}");
        assert!(msg.contains("vectordata config set cache <path>"),
            "missing explicit-path hint: {msg}");
        assert!(msg.contains("veks datasets config set cache"),
            "missing veks alias hint: {msg}");
        assert!(msg.contains("mkdir -p"), "missing manual mkdir hint: {msg}");
        assert!(msg.contains("cache_dir:"), "missing cache_dir line: {msg}");
        assert!(msg.contains("protect_settings: true"),
            "missing protect_settings line: {msg}");
    }

    #[test]
    fn write_cache_dir_creates_settings_and_target_dir() {
        let home = tempfile::tempdir().unwrap();
        let settings = home.path().join("vectordata/settings.yaml");
        let target = home.path().join("cache-root");
        write_cache_dir_at(&settings, &target, false).unwrap();

        assert!(settings.is_file(), "settings file should be created");
        assert!(target.is_dir(), "cache_dir should be created");
        assert_eq!(cache_dir_from(&settings).unwrap(), target);
        assert!(protect_settings_from(&settings),
            "writer must always set protect_settings: true");
    }

    #[test]
    fn write_cache_dir_refuses_overwrite_of_protected_settings() {
        let home = tempfile::tempdir().unwrap();
        let settings = home.path().join("settings.yaml");
        let first = home.path().join("first");
        let second = home.path().join("second");
        write_cache_dir_at(&settings, &first, false).unwrap();

        let err = write_cache_dir_at(&settings, &second, false).unwrap_err();
        match err {
            SettingsWriteError::Protected { ref existing_cache_dir, .. } => {
                assert_eq!(existing_cache_dir.as_deref(), Some(first.as_path()));
            }
            other => panic!("expected Protected, got {other:?}"),
        }
        // Force overrides.
        write_cache_dir_at(&settings, &second, true).unwrap();
        assert_eq!(cache_dir_from(&settings).unwrap(), second);
    }

    #[test]
    fn write_cache_dir_refuses_overwrite_even_without_protect_flag() {
        // The `protect_settings:` flag in the file is no longer
        // load-bearing — every existing-with-different-value write
        // is refused without --force. The earlier "unprotected
        // settings overwrite freely" rule turned a `set-cache` typo
        // into a silent reconfiguration.
        let home = tempfile::tempdir().unwrap();
        let settings = home.path().join("settings.yaml");
        let target = home.path().join("cache");
        // Pre-existing settings WITHOUT protect_settings.
        std::fs::write(&settings, "cache_dir: /old/path\n").unwrap();

        let err = write_cache_dir_at(&settings, &target, false).unwrap_err();
        match err {
            SettingsWriteError::Protected { ref existing_cache_dir, .. } => {
                assert_eq!(existing_cache_dir.as_deref(),
                    Some(std::path::Path::new("/old/path")));
            }
            other => panic!("expected Protected, got {other:?}"),
        }
        // Force overrides.
        write_cache_dir_at(&settings, &target, true).unwrap();
        assert_eq!(cache_dir_from(&settings).unwrap(), target);
    }

    #[test]
    fn write_cache_dir_is_noop_for_same_value() {
        let home = tempfile::tempdir().unwrap();
        let settings = home.path().join("settings.yaml");
        let target = home.path().join("cache");
        write_cache_dir_at(&settings, &target, false).unwrap();
        let before = std::fs::metadata(&settings).unwrap().modified().unwrap();
        // Force one filesystem-tick of separation so a real write
        // would be detectable.
        std::thread::sleep(std::time::Duration::from_millis(20));
        // Writing the same value (no --force) must succeed without
        // touching the file.
        write_cache_dir_at(&settings, &target, false)
            .expect("same-value write must be idempotent");
        let after = std::fs::metadata(&settings).unwrap().modified().unwrap();
        assert_eq!(before, after,
            "same-value write must not touch the settings file");
    }

    #[test]
    fn cache_dir_errors_when_setting_omitted_from_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = tmp.path().join("settings.yaml");
        std::fs::write(&yaml, "protect_settings: true\n").unwrap();
        let err = cache_dir_from(&yaml).unwrap_err();
        match err {
            SettingsError::NotConfigured { ref settings_path, file_exists } => {
                assert_eq!(settings_path, &yaml);
                assert!(file_exists, "file_exists must be true when settings.yaml is present");
            }
        }
        let msg = err.to_string();
        assert!(msg.contains("exists but does not declare 'cache_dir:'"),
            "expected 'exists but does not declare' phrasing: {msg}");
    }
}
