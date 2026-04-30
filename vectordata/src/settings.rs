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

use std::path::{Path, PathBuf};

const CONFIG_DIR: &str = ".config/vectordata";
const SETTINGS_FILE: &str = "settings.yaml";

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
pub fn cache_dir() -> Result<PathBuf, SettingsError> {
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
