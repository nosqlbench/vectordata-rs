// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Thin wrappers over [`vectordata::settings`] for the rest of
//! veks-pipeline.
//!
//! The user-facing `config` admin operations (show, set-cache,
//! list-mounts, add/remove/list catalogs) all live in
//! [`vectordata::config`] now — both the `vectordata` binary and the
//! `veks` CLI dispatch there. This module keeps only the two helpers
//! that other veks-pipeline code calls when it needs the configured
//! cache directory.

use std::path::PathBuf;

/// Resolve the configured cache directory from settings.yaml.
///
/// Delegates to [`vectordata::settings::cache_dir`] — the single
/// source of truth for cache resolution. Returns
/// [`vectordata::settings::SettingsError`] when `cache_dir:` is not
/// configured; print the error directly via its `Display` impl,
/// which carries actionable commands.
pub fn configured_cache_dir() -> Result<PathBuf, vectordata::settings::SettingsError> {
    vectordata::settings::cache_dir()
}

/// CLI-style wrapper around [`configured_cache_dir`]: prints the
/// configuration error to stderr and exits with code 2 when
/// `cache_dir:` is unset. Used by veks subcommands that prefer a
/// clean exit over propagating the error up the stack.
pub fn configured_cache_dir_or_exit() -> PathBuf {
    match configured_cache_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}
