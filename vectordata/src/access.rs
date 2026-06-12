// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Access-mode taxonomy for vector data sources.
//!
//! Datasets reach the runtime through one of four [`AccessMode`]
//! variants. Two are sparse — chunks are fetched on demand and
//! recorded in a local sidecar so subsequent runs only re-fetch what's
//! missing. The third (`FullTransfer`) is the catch-all for sources
//! where no sparse path can apply: variable-length record files
//! without a published sibling offset index, formats we don't
//! recognise, or transports that don't support HTTP RANGE.
//!
//! The split between [`AccessMode::MerkleHashed`] and
//! [`AccessMode::MerkleChunked`] is about *trust*, not access pattern.
//! Both stream chunks; the hashed variant verifies each one against a
//! published merkle root, the chunked variant trusts server bytes
//! byte-for-byte and leans on TLS for integrity.
//!
//! The variants map onto the crate-private `Storage` enum:
//!
//! | `AccessMode`     | `Storage` variant | Sidecar              | Verification        |
//! |------------------|-------------------|----------------------|---------------------|
//! | `Local`          | `Storage::Mmap`   | (none)               | n/a — local         |
//! | `MerkleHashed`   | `Storage::Cached` | `<file>.mrkl`        | per-chunk merkle    |
//! | `MerkleChunked`  | `Storage::Http`   | `<file>.chunks`      | TLS only            |
//! | `FullTransfer`   | (forced full DL)  | (none)               | TLS only            |

use std::path::Path;

/// How a dataset facet source will be accessed at open time.
///
/// Use [`AccessMode::classify`] to predict the mode for a source
/// without performing any network I/O. The classifier defaults to the
/// best-effort path (`MerkleChunked`) for remote uniform-format
/// sources whose `.mref` status hasn't been observed yet; once a
/// source has been opened once and left a `.mrkl` sidecar in the
/// cache, subsequent classifications upgrade to `MerkleHashed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Every byte is already on disk: a local source file
    /// (mmap-backed reads), or a remote facet whose cache copy is
    /// complete — a finished download, or a chunk-store file with a
    /// fully valid bitmap. Either way, no download takes place.
    Local,

    /// Remote source with a published `.mref`. Chunks download on
    /// demand and each one is verified against the merkle tree
    /// before being served. A `<file>.mrkl` sidecar in the cache
    /// records which chunks are present so subsequent runs skip
    /// re-fetching.
    MerkleHashed,

    /// Remote source without `.mref` but with HTTP RANGE support.
    /// Chunks download on demand and a local `<file>.chunks` bitmap
    /// records which ones are present — same incremental UX as
    /// `MerkleHashed`, but the bytes are trusted byte-for-byte and
    /// integrity comes from TLS rather than a per-chunk hash chain.
    MerkleChunked,

    /// Remote source where no sparse path applies. The full file
    /// must download before any read succeeds. Triggered by
    /// variable-length record formats without a sibling offset
    /// index, unrecognised extensions, or transports that refuse
    /// HTTP RANGE requests.
    FullTransfer,
}

impl AccessMode {
    /// Predict the access mode for a source without performing
    /// network I/O.
    ///
    /// `resolved_source` is the full source string the underlying
    /// storage layer would see (absolute path, `file://` URI, or
    /// remote URL — relative paths must already be resolved against
    /// their catalog's location).
    ///
    /// `cache_dir` is the configured cache root. When the file's
    /// `.mrkl` sidecar is already present under this root, the
    /// classification can upgrade `MerkleChunked` → `MerkleHashed`
    /// (proof from the prior open). Pass an empty path to skip the
    /// sidecar lookup.
    ///
    /// Conservative bias: when offline information can't distinguish
    /// merkle-hashed from merkle-chunked, returns `MerkleChunked`
    /// (the safe sparse path that works without `.mref`). When
    /// sparse access can't apply at all (vvec format, unknown
    /// extension), returns `FullTransfer`.
    pub fn classify(resolved_source: &str, cache_dir: &Path) -> AccessMode {
        if !crate::transport::is_remote_url(resolved_source) {
            return AccessMode::Local;
        }
        Self::classify_remote(resolved_source, cache_dir)
    }

    /// Predict the access mode when the caller has already established
    /// the source is remote. Use this from contexts where the
    /// local-vs-remote question was answered upstream (catalog metadata,
    /// existing file-presence check) and the input is just the
    /// remote file's path or name.
    ///
    /// Same offline classification as [`AccessMode::classify`] minus
    /// the URL-scheme check: vvec formats and unknown extensions return
    /// `FullTransfer`; uniform xvec extensions return `MerkleHashed`
    /// when a `.mrkl` sidecar is present under `cache_dir`, otherwise
    /// `MerkleChunked`.
    pub fn classify_remote(remote_path: &str, cache_dir: &Path) -> AccessMode {
        let ext = crate::io::ext_of(remote_path);
        if crate::io::infer_elem_size(ext) == 0 {
            return AccessMode::FullTransfer;
        }
        if crate::io::is_vvec_ext(ext) {
            return AccessMode::FullTransfer;
        }
        if mrkl_sidecar_present(remote_path, cache_dir) {
            AccessMode::MerkleHashed
        } else {
            AccessMode::MerkleChunked
        }
    }

    /// Short, fixed-width label suitable for a single picker column.
    pub fn short_label(self) -> &'static str {
        match self {
            AccessMode::Local         => "local",
            AccessMode::MerkleHashed  => "hashed",
            AccessMode::MerkleChunked => "chunked",
            AccessMode::FullTransfer  => "full-DL",
        }
    }

    /// Human-readable description for the detail panel. Kept under
    /// ~55 display columns so the picker's bottom panel line
    /// (`" access: "` prefix included) fits an 80-column terminal
    /// without truncating or wrapping.
    pub fn description(self) -> &'static str {
        match self {
            AccessMode::Local =>
                "local — fully on disk; no download",
            AccessMode::MerkleHashed =>
                "hashed — sparse chunks, merkle-verified",
            AccessMode::MerkleChunked =>
                "chunked — sparse chunks, TLS trust only",
            AccessMode::FullTransfer =>
                "full-transfer — whole file downloads first",
        }
    }
}

/// Best-effort check: is a `.mrkl` sidecar already present under the
/// cache root that matches this source's filename?
///
/// We don't try to compute the exact natural-layout cache path
/// (`<cache>/<dataset>/<facet_relpath>.mrkl`) from a source string
/// alone — that needs catalog context (dataset_name, file_relpath
/// stripping) we don't have at access-mode classification time.
/// Walk the cache root for any `<filename>.mrkl` match instead;
/// false positives across datasets are unlikely because dataset
/// directories namespace the sidecar names.
fn mrkl_sidecar_present(resolved_source: &str, cache_dir: &Path) -> bool {
    if cache_dir.as_os_str().is_empty() { return false; }
    let filename = resolved_source
        .rsplit('/')
        .next()
        .unwrap_or(resolved_source);
    let sidecar_name = format!("{filename}.mrkl");
    walk_for_filename(cache_dir, &sidecar_name, 4)
}

/// Bounded-depth filename search. Stops descending past `max_depth`
/// to avoid wandering into deep workspace trees on misconfigured
/// caches.
fn walk_for_filename(root: &Path, target: &str, max_depth: usize) -> bool {
    if max_depth == 0 { return false; }
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path.file_name().map(|n| n == target).unwrap_or(false)
        {
            return true;
        }
        if path.is_dir() && walk_for_filename(&path, target, max_depth - 1) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_paths_classify_as_local() {
        let cache = std::path::PathBuf::new();
        assert_eq!(AccessMode::classify("/tmp/base.fvec", &cache), AccessMode::Local);
        assert_eq!(AccessMode::classify("file:///tmp/base.fvec", &cache), AccessMode::Local);
        assert_eq!(AccessMode::classify("./base.fvec", &cache), AccessMode::Local);
    }

    #[test]
    fn remote_uniform_xvec_defaults_to_chunked() {
        let cache = std::path::PathBuf::new();
        assert_eq!(
            AccessMode::classify("https://example.com/data/base.fvec", &cache),
            AccessMode::MerkleChunked,
        );
        assert_eq!(
            AccessMode::classify("https://example.com/data/base.bvec", &cache),
            AccessMode::MerkleChunked,
        );
    }

    #[test]
    fn remote_vvec_classifies_as_full_transfer() {
        let cache = std::path::PathBuf::new();
        assert_eq!(
            AccessMode::classify("https://example.com/data/meta.fvvec", &cache),
            AccessMode::FullTransfer,
        );
    }

    #[test]
    fn remote_unknown_extension_classifies_as_full_transfer() {
        let cache = std::path::PathBuf::new();
        assert_eq!(
            AccessMode::classify("https://example.com/data/meta.bin", &cache),
            AccessMode::FullTransfer,
        );
    }

    #[test]
    fn mrkl_sidecar_upgrades_to_hashed() {
        let tmp = tempfile::tempdir().unwrap();
        // Drop a sidecar matching the expected filename pattern.
        let nested = tmp.path().join("ab").join("cd");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("base.fvec.mrkl"), b"fake").unwrap();
        assert_eq!(
            AccessMode::classify("https://example.com/data/base.fvec", tmp.path()),
            AccessMode::MerkleHashed,
        );
    }
}
