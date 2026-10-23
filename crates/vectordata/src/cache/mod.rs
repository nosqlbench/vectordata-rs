// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Internal. The cache's dataset-facing side: where a dataset's files
//! land ([`layout`]), what is under the cache root ([`reader`]), and
//! whether the cache directory can hold a fetch.
//!
//! The cache files themselves — the merkle-verified channel and the
//! chunk store — are `veks-transport`'s; this crate tells them where
//! each file and sidecar lives (`docs/design/srd-crate-boundaries.md`
//! TR-7).

pub(crate) mod layout;
pub(crate) mod reader;

use std::path::Path;

/// Workers a chunked download runs at once: the transport's
/// [`download_concurrency`](veks_transport::Transport::download_concurrency),
/// set from `VECTORDATA_DOWNLOAD_CONCURRENCY` when vectordata builds its
/// transport ([`crate::transport::config_from`]). Both the merkle-verified
/// and the chunk-store paths run this many, so worker count and the
/// HTTP runtime count stay matched.
pub(crate) fn download_concurrency() -> usize {
    crate::transport::shared().download_concurrency()
}

pub use veks_apicore::error::InsufficientCacheSpace;

/// Fail fast when the cache directory `dir` cannot hold `needed`
/// additional bytes. Returns `Ok(())` — i.e. lets the download
/// proceed — in every case where the check can't be made meaningfully:
///   - `needed == 0` (nothing remote to fetch),
///   - the cache directory can't be resolved (the download itself will
///     surface the real configuration error — we don't invent a new
///     one here), or
///   - free space can't be queried (non-Unix target, or `statvfs`
///     failed). The invariant only *rejects* on a known shortfall;
///     it never blocks on an unverifiable one.
///
/// `dir` is the cache a dataset's catalog names, if any: the directory
/// checked is the dataset's cache ([`crate::settings::dataset_cache_location`]).
pub(crate) fn ensure_cache_capacity_in(dir: Option<&Path>, needed: u64) -> Result<(), InsufficientCacheSpace> {
    if needed == 0 {
        return Ok(());
    }
    let Ok(dir) = crate::settings::dataset_cache_location(dir) else {
        return Ok(());
    };
    let available = available_space(&dir);
    capacity_verdict(&dir, needed, available)
}

/// Pure capacity decision, split out from the filesystem/settings
/// lookups so the threshold logic is unit-testable without real
/// directories or process-env mutation. `available == None` means
/// "couldn't measure" and is treated as a pass.
fn capacity_verdict(
    dir: &Path,
    needed: u64,
    available: Option<u64>,
) -> Result<(), InsufficientCacheSpace> {
    match available {
        Some(avail) if avail < needed => Err(InsufficientCacheSpace {
            cache_dir: dir.to_path_buf(),
            needed,
            available: avail,
        }),
        _ => Ok(()),
    }
}

/// Bytes free to an unprivileged process on the filesystem that holds
/// `path`. `path` need not exist yet — the cache directory is often
/// created lazily on first download — so this walks up to the nearest
/// existing ancestor (the same filesystem) before measuring. Returns
/// `None` when the space can't be determined (no existing ancestor,
/// non-Unix target, or the syscall failed).
fn available_space(path: &Path) -> Option<u64> {
    let mut probe = path;
    loop {
        if probe.exists() {
            break;
        }
        probe = probe.parent()?;
    }
    statvfs_available(probe)
}

#[cfg(unix)]
fn statvfs_available(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let cpath = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: an all-zero `statvfs` is a valid uninitialised buffer;
    // `cpath` is a valid NUL-terminated C string for the duration of
    // the call, and we read `stat` only after confirming `rc == 0`.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(cpath.as_ptr(), &mut stat) };
    if rc != 0 {
        return None;
    }
    // `f_bavail` is blocks available to non-root callers (honors the
    // reserved-block fraction); `f_frsize` is the fundamental block
    // size that count is expressed in.
    Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

#[cfg(not(unix))]
fn statvfs_available(_path: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The capacity decision rejects only a *known* shortfall, passes
    /// when it fits or exactly meets, and never blocks when the free
    /// space couldn't be measured.
    #[test]
    fn capacity_verdict_rejects_only_known_shortfall() {
        let dir = Path::new("/cache/root");

        // Fits with room to spare.
        assert!(capacity_verdict(dir, 50, Some(100)).is_ok());
        // Exactly fits — equal is not a shortfall.
        assert!(capacity_verdict(dir, 100, Some(100)).is_ok());
        // Unmeasurable free space (non-Unix / statvfs failure) must
        // not block the download.
        assert!(capacity_verdict(dir, 1_000_000, None).is_ok());

        // Genuine shortfall — rejected, carrying the numbers needed
        // for the operator-facing message.
        let err = capacity_verdict(dir, 100, Some(40)).unwrap_err();
        assert_eq!(
            err,
            InsufficientCacheSpace {
                cache_dir: dir.to_path_buf(),
                needed: 100,
                available: 40,
            }
        );
        // The message names the directory and both magnitudes.
        let msg = err.to_string();
        assert!(
            msg.contains("/cache/root"),
            "message must name the cache dir: {msg}"
        );
        assert!(
            msg.contains("100"),
            "message must state bytes needed: {msg}"
        );
        assert!(
            msg.contains("40"),
            "message must state bytes available: {msg}"
        );
    }

    /// `ensure_cache_capacity` short-circuits to Ok when nothing needs
    /// fetching, so a fully-resident (or all-local) precache never
    /// touches the filesystem-space probe.
    #[test]
    fn ensure_capacity_is_a_noop_for_zero_bytes() {
        assert!(ensure_cache_capacity_in(None, 0).is_ok());
    }

    /// `available_space` reports a positive figure for a real
    /// directory, and tolerates a not-yet-created cache path by
    /// measuring the nearest existing ancestor's filesystem.
    #[cfg(unix)]
    #[test]
    fn available_space_measures_existing_ancestor() {
        let tmp = tempfile::tempdir().unwrap();
        let existing = available_space(tmp.path());
        assert!(
            existing.is_some_and(|b| b > 0),
            "an existing dir must report some free space"
        );

        // A nested path that doesn't exist yet still measures (walks
        // up to `tmp`, which shares the filesystem).
        let nested = tmp.path().join("not/created/yet/cache");
        assert!(!nested.exists());
        let via_ancestor = available_space(&nested);
        assert!(
            via_ancestor.is_some_and(|b| b > 0),
            "a not-yet-created cache path must measure via its ancestor"
        );
    }
}
