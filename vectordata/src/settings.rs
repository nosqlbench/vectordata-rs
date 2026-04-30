// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-user `vectordata` settings.
//!
//! Loaded from `~/.config/vectordata/settings.yaml` (or
//! `$VECTORDATA_HOME/settings.yaml` when that env var is set —
//! used by tests). The single source of truth for cache-directory
//! resolution: every public reader path that fetches remote data
//! routes through here, so the user's `cache_dir:` override is
//! honored uniformly across vectordata and any consuming crate.

use std::path::PathBuf;

const CONFIG_DIR: &str = ".config/vectordata";
const SETTINGS_FILE: &str = "settings.yaml";
const FALLBACK_CACHE_REL: &str = ".cache/vectordata";

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
/// Order of precedence:
/// 1. `cache_dir:` entry in the user's `settings.yaml` (or
///    `$VECTORDATA_HOME/settings.yaml`).
/// 2. `$HOME/.cache/vectordata/` fallback.
/// 3. Relative `.cache/vectordata` (no `$HOME` set — pathological).
///
/// The settings parser is whitespace-tolerant and strips matching
/// surrounding quotes from the value.
pub fn cache_dir() -> PathBuf {
    let settings = settings_path();
    if let Ok(content) = std::fs::read_to_string(&settings) {
        for line in content.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("cache_dir:") {
                let val = val.trim().trim_matches('"').trim_matches('\'');
                if !val.is_empty() {
                    return PathBuf::from(val);
                }
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(FALLBACK_CACHE_REL)
    } else {
        PathBuf::from(FALLBACK_CACHE_REL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `$VECTORDATA_HOME` overrides settings_path so the
    /// settings.yaml override is honored end-to-end.
    #[test]
    fn cache_dir_honors_settings_override() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = tmp.path().join("settings.yaml");
        std::fs::write(&yaml,
            "cache_dir: /mnt/datamir/vectordata-cache\nprotect_settings: true\n",
        ).unwrap();
        // SAFETY: tests in this module are serialized by the env var
        // contention pattern; we restore on the way out.
        let prev = std::env::var_os("VECTORDATA_HOME");
        unsafe { std::env::set_var("VECTORDATA_HOME", tmp.path()); }
        let resolved = cache_dir();
        match prev {
            Some(v) => unsafe { std::env::set_var("VECTORDATA_HOME", v); },
            None    => unsafe { std::env::remove_var("VECTORDATA_HOME"); },
        }
        assert_eq!(resolved, PathBuf::from("/mnt/datamir/vectordata-cache"));
    }

    #[test]
    fn cache_dir_quoted_value_is_unwrapped() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = tmp.path().join("settings.yaml");
        std::fs::write(&yaml, "cache_dir: \"/some/quoted/path\"\n").unwrap();
        let prev = std::env::var_os("VECTORDATA_HOME");
        unsafe { std::env::set_var("VECTORDATA_HOME", tmp.path()); }
        let resolved = cache_dir();
        match prev {
            Some(v) => unsafe { std::env::set_var("VECTORDATA_HOME", v); },
            None    => unsafe { std::env::remove_var("VECTORDATA_HOME"); },
        }
        assert_eq!(resolved, PathBuf::from("/some/quoted/path"));
    }

    #[test]
    fn cache_dir_falls_back_when_setting_missing() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("settings.yaml"), "protect_settings: true\n").unwrap();
        let prev = std::env::var_os("VECTORDATA_HOME");
        unsafe { std::env::set_var("VECTORDATA_HOME", tmp.path()); }
        let resolved = cache_dir();
        match prev {
            Some(v) => unsafe { std::env::set_var("VECTORDATA_HOME", v); },
            None    => unsafe { std::env::remove_var("VECTORDATA_HOME"); },
        }
        // Falls back to $HOME/.cache/vectordata
        if let Some(home) = std::env::var_os("HOME") {
            assert_eq!(resolved, PathBuf::from(home).join(".cache/vectordata"));
        }
    }
}
