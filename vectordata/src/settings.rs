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
                let example_path = "/mnt/example/vectordata-cache";
                let parent = settings_path.parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "$HOME/.config/vectordata".to_string());
                write!(f,
                    "vectordata is not configured: {why}\n\
                     \n\
                     # If you have the `veks` CLI installed:\n\
                     veks datasets config set-cache {example_path}\n\
                     \n\
                     # Or configure manually:\n\
                     mkdir -p {parent}\n\
                     echo \"cache_dir: {example_path}\" >> {}\n\
                     echo \"protect_settings: true\" >> {}",
                    settings_path.display(), settings_path.display(),
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
/// If the file is missing or does not declare `cache_dir:`, returns
/// [`SettingsError::NotConfigured`] — there is **no silent
/// fallback**. Print the error directly: its `Display` impl carries
/// ready-to-paste commands the user can run to configure the cache.
///
/// If [`override_cache_dir_for_process`] has been called, that value
/// takes precedence and `settings.yaml` is not consulted.
pub fn cache_dir() -> Result<PathBuf, SettingsError> {
    if let Some(p) = CACHE_DIR_OVERRIDE.get() {
        return Ok(p.clone());
    }
    cache_dir_from(&settings_path())
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
/// `vectordata config show`) so users can see at a glance that the
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

/// Write a new `cache_dir:` into the user's `settings.yaml`,
/// creating the file and its parent directory as needed. Always
/// writes `protect_settings: true` so subsequent `set-cache`
/// invocations have to be deliberate.
///
/// If the file already exists with `protect_settings: true`,
/// refuses to overwrite unless `force=true`, returning
/// [`SettingsWriteError::Protected`] with the existing value so the
/// caller can show the user what's currently configured.
///
/// Also creates `cache_dir` itself if it doesn't exist — having a
/// `cache_dir:` value pointing at a non-existent path is a common
/// support trap, so this constructor closes that gap.
pub fn write_cache_dir(cache_dir: &Path, force: bool) -> Result<(), SettingsWriteError> {
    write_cache_dir_at(&settings_path(), cache_dir, force)
}

/// Variant of [`write_cache_dir`] that targets an explicit settings
/// path — used by tests so they can exercise the writer without
/// stomping on `$HOME`.
pub fn write_cache_dir_at(
    settings: &Path,
    cache_dir: &Path,
    force: bool,
) -> Result<(), SettingsWriteError> {
    if settings.is_file() && !force {
        let existing = cache_dir_from(settings).ok();
        if protect_settings_from(settings) {
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

    let content = format!(
        "cache_dir: {}\nprotect_settings: true\n",
        cache_dir.display(),
    );
    std::fs::write(settings, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(msg.contains("veks datasets config set-cache"), "missing CLI hint: {msg}");
        assert!(msg.contains("mkdir -p"), "missing manual hint: {msg}");
        assert!(msg.contains("echo \"cache_dir:"), "missing echo hint: {msg}");
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
    fn write_cache_dir_overwrites_unprotected_existing_settings() {
        let home = tempfile::tempdir().unwrap();
        let settings = home.path().join("settings.yaml");
        let target = home.path().join("cache");
        // Pre-existing settings without protect_settings.
        std::fs::write(&settings, "cache_dir: /old/path\n").unwrap();
        write_cache_dir_at(&settings, &target, false)
            .expect("unprotected settings should overwrite freely");
        assert_eq!(cache_dir_from(&settings).unwrap(), target);
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
