// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cache directory layout helpers.
//!
//! The reader struct (`CachedVectorReader`) that used to live here is
//! gone — its responsibilities are absorbed by the canonical
//! [`crate::storage::Storage::Cached`] variant. Only the path
//! resolution helpers remain, used by `Storage::open_url_cached`.

use std::path::{Path, PathBuf};
use url::Url;

/// Crate-internal alias for [`crate::settings::cache_dir`]. Kept as a
/// thin wrapper so existing call sites stay terse; the actual
/// resolution lives in `settings.rs` so `vectordata` and consuming
/// crates share one canonical implementation.
///
/// Returns [`crate::settings::SettingsError`] when `cache_dir:` is
/// not configured — there is no silent fallback. Callers that need
/// to surface the error to the user should print it directly; its
/// `Display` impl carries actionable commands.
pub(crate) fn default_cache_dir() -> Result<PathBuf, crate::settings::SettingsError> {
    crate::settings::cache_dir()
}

/// Resolve the cache directory for a dataset URL.
///
/// Includes the URL port in the host segment so concurrent local
/// servers (e.g., test fixtures on different ephemeral ports) get
/// isolated cache directories. Without the port, two test runs
/// against `127.0.0.1:RAND1/foo.fvec` and `127.0.0.1:RAND2/foo.fvec`
/// would share state and fail with stale-merkle errors.
pub(crate) fn cache_dir_for_url(url: &Url, cache_root: &Path) -> PathBuf {
    let host = match (url.host_str(), url.port()) {
        (Some(h), Some(p)) => format!("{h}:{p}"),
        (Some(h), None)    => h.to_string(),
        (None,    _)       => "local".to_string(),
    };
    let path = url.path().trim_start_matches('/');
    let dir = if let Some(pos) = path.rfind('/') {
        &path[..pos]
    } else {
        path
    };
    cache_root.join(host).join(dir)
}

