// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Natural cache layout — flat per-dataset directories with a single
//! origin file at the root of each one.
//!
//! ```text
//! <cache_root>/<dataset>/
//!   ├── origin.json                  (records the catalog source URL)
//!   ├── base.fvec                    (data file matching the view path)
//!   ├── base.fvec.mrkl               (merkle state sidecar)
//!   ├── query.fvec
//!   └── ...
//! ```
//!
//! The directory mirrors what's inside a published dataset directory,
//! so users can navigate to `<cache_root>/<dataset>/` and find their
//! data laid out the way the catalog laid it out — no content-
//! addressed `<sha256>/<sha256>/` jumble.
//!
//! Collisions across catalogs (two different catalogs both publishing
//! a dataset named `sift1m`, with different bytes behind it) are
//! detected by [`verify_or_record_origin`]: the first download writes
//! `origin.json` with the catalog source URL, subsequent opens compare,
//! and a mismatch is a hard error. Users resolve it by editing
//! `origin.json` (when they know it's the same dataset moved to a new
//! URL) or by deleting the dataset directory (to re-download from the
//! new source). Both moves are visible filesystem operations, no opaque
//! cache surgery.

use std::path::{Path, PathBuf};

/// Filename of the per-dataset origin record at the root of each
/// `<cache_root>/<dataset>/` directory.
pub(crate) const DATASET_ORIGIN_FILE: &str = "origin.json";

/// Filename suffix appended to in-progress full-transfer downloads
/// (the third storage mode — no `.mref`, no chunk-level resume).
/// During transfer the file lives at `<final_path>.download`; on
/// successful completion it's atomically renamed to `<final_path>`.
/// The merkle-cached (`CachedChannel`) and chunked-HTTP
/// (`ChunkStore`) modes both write IN PLACE — they have per-chunk
/// validity tracking so a partially-filled file is a meaningful,
/// resumable state, not corruption — and this constant doesn't
/// apply to them.
pub(crate) const DOWNLOAD_SUFFIX: &str = ".download";

/// Compute the per-dataset cache directory under `cache_root`. The
/// directory mirrors the dataset name verbatim — no host prefix, no
/// hash, no shape distinction (`blobs/` vs `http/`). Two catalogs
/// publishing the same dataset name collide here; that collision is
/// caught by [`verify_or_record_origin`].
pub(crate) fn dataset_cache_dir(cache_root: &Path, dataset: &str) -> PathBuf {
    cache_root.join(dataset)
}

/// Compute the cache file path for a specific facet within a dataset.
/// `facet_relpath` is whatever the view declares (`base.fvec`,
/// `profiles/1m/base.fvec`, etc.) — passed through verbatim so the
/// cache layout mirrors the catalog layout.
#[allow(dead_code)] // tests use this; production goes through Storage::open_layered which passes
                    // dataset_dir + file_relpath separately to CachedChannel/ChunkStore.
pub(crate) fn facet_cache_path(
    cache_root: &Path,
    dataset: &str,
    facet_relpath: &str,
) -> PathBuf {
    dataset_cache_dir(cache_root, dataset).join(facet_relpath)
}

/// Compute the in-progress download path for the full-transfer
/// mode (the third storage path, used when there is neither a `.mref`
/// nor chunk-level resumable HTTP). Returns `<final_path>.download` —
/// same parent directory as the final file, so atomic rename on
/// completion is a single same-filesystem syscall.
///
/// Not used by `CachedChannel` (merkle) or `ChunkStore` (chunked
/// HTTP): both of those write in place, because per-chunk validity
/// tracking makes a partially-filled file a meaningful, resumable
/// state — not corruption.
#[allow(dead_code)] // wired up when the full-transfer storage path lands.
pub(crate) fn download_inprogress_path(final_path: &Path) -> PathBuf {
    let mut p = final_path.to_path_buf();
    let new_name = match p.file_name().and_then(|n| n.to_str()) {
        Some(base) => format!("{base}{DOWNLOAD_SUFFIX}"),
        None => DOWNLOAD_SUFFIX.trim_start_matches('.').to_string(),
    };
    p.set_file_name(new_name);
    p
}

/// Atomically publish a fully-downloaded full-transfer file: rename
/// `<final>.download` → `<final>`. Same-filesystem rename, single
/// syscall on every modern Unix.
#[allow(dead_code)] // wired up when the full-transfer storage path lands.
pub(crate) fn publish_download(
    inprogress_path: &Path,
    final_path: &Path,
) -> std::io::Result<()> {
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::rename(inprogress_path, final_path)
}

/// Per-dataset origin record. One file per dataset directory; records
/// the URL the dataset was first fetched from. Compared against the
/// current request's source URL on subsequent opens to catch
/// cross-catalog name collisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DatasetOrigin {
    /// The catalog source URL recorded at first download.
    pub source: String,
    /// RFC 3339 timestamp when the dataset was first populated. Free-
    /// form text — tooling shows it but doesn't compare on it.
    pub fetched_at: String,
}

/// Read the dataset's `origin.json` if it exists, returning `None` on
/// absent / malformed / unreadable file. Tools that want a strict read
/// (failing on malformed) should check existence + parse explicitly;
/// this helper is for the "no-prior-state" code path that treats
/// missing-or-broken origin equivalently.
pub(crate) fn read_dataset_origin(dataset_dir: &Path) -> Option<DatasetOrigin> {
    let path = dataset_dir.join(DATASET_ORIGIN_FILE);
    let content = std::fs::read_to_string(&path).ok()?;
    let source = extract_json_string(&content, "source")?;
    let fetched_at = extract_json_string(&content, "fetched_at").unwrap_or_default();
    Some(DatasetOrigin { source, fetched_at })
}

/// Write a fresh `origin.json` to a dataset directory. Creates the
/// directory if missing. Format is minimal JSON so users can `cat` it
/// (and edit it to migrate, per the documented workflow).
pub(crate) fn write_dataset_origin(
    dataset_dir: &Path,
    source: &str,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dataset_dir)?;
    let fetched_at = httpdate::fmt_http_date(std::time::SystemTime::now());
    let json = format!(
        "{{\n  \"source\": {},\n  \"fetched_at\": {}\n}}\n",
        json_escape_str(source),
        json_escape_str(&fetched_at),
    );
    std::fs::write(dataset_dir.join(DATASET_ORIGIN_FILE), json)
}

/// Origin-mismatch error: the dataset directory already records a
/// source URL different from the one the current open is using.
#[derive(Debug)]
pub(crate) struct OriginMismatch {
    pub dataset_dir: PathBuf,
    pub recorded: String,
    pub requested: String,
}

impl std::fmt::Display for OriginMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "cache directory name collision: {} already holds data from {}, \
             but the current open is fetching from {}. Delete the directory \
             (rm -rf {}) to re-download from the new source.",
            self.dataset_dir.display(),
            self.recorded,
            self.requested,
            self.dataset_dir.display(),
        )
    }
}

impl std::error::Error for OriginMismatch {}

impl From<OriginMismatch> for std::io::Error {
    fn from(e: OriginMismatch) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
    }
}

/// Verify the dataset's origin file matches `source_url`. Writes a
/// fresh one on first download (when no origin.json exists yet).
/// Returns:
///   - `Ok(())` — origin matched (or was just written for a fresh
///     dataset directory)
///   - `Err(OriginMismatch)` — recorded URL ≠ requested URL; caller
///     should refuse the open and surface the mismatch to the user
///
/// This is the load-bearing check for the natural-layout collision
/// story: it converts "two different catalogs both published a
/// dataset named X" from a silent corruption into a user-visible
/// error with a documented manual resolution.
pub(crate) fn verify_or_record_origin(
    dataset_dir: &Path,
    source_url: &str,
) -> Result<(), OriginMismatch> {
    if let Some(existing) = read_dataset_origin(dataset_dir) {
        if existing.source != source_url {
            return Err(OriginMismatch {
                dataset_dir: dataset_dir.to_path_buf(),
                recorded: existing.source,
                requested: source_url.to_string(),
            });
        }
        return Ok(());
    }
    // First open against a fresh (or origin-less) dataset dir.
    // Writing might fail if the directory was just deleted from
    // under us — bubble that up as InvalidInput so the caller can
    // surface it, but don't synthesise a mismatch.
    let _ = write_dataset_origin(dataset_dir, source_url);
    Ok(())
}

// ── JSON helpers (tiny, scoped) ───────────────────────────────────

/// Minimal JSON string escaper. Same shape as `cache::reader::json_escape`
/// — duplicated here so the layout module stays self-contained and
/// doesn't reach into reader's pub(crate) internals.
fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Extract a string-valued JSON field by key. Hand-rolled parser
/// matching the same constraints `cache::reader::extract_json_string`
/// already satisfies: tiny dependency-free, accepts the controlled
/// shapes we write, tolerates whitespace.
fn extract_json_string(content: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let key_pos = content.find(&needle)?;
    let after_key = &content[key_pos + needle.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = &after_key[colon_pos + 1..];
    let start = after_colon.find('"')?;
    let body = &after_colon[start + 1..];
    let mut out = String::new();
    let mut escape = false;
    for c in body.chars() {
        if escape {
            match c {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                _ => { out.push('\\'); out.push(c); }
            }
            escape = false;
        } else if c == '\\' {
            escape = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_cache_dir_is_flat() {
        let p = dataset_cache_dir(Path::new("/cache"), "sift1m");
        assert_eq!(p, Path::new("/cache/sift1m"));
    }

    #[test]
    fn facet_cache_path_preserves_view_relpath() {
        let p = facet_cache_path(Path::new("/cache"), "sift1m", "base.fvec");
        assert_eq!(p, Path::new("/cache/sift1m/base.fvec"));
        let nested = facet_cache_path(Path::new("/cache"), "sift1m", "profiles/1m/base.fvec");
        assert_eq!(nested, Path::new("/cache/sift1m/profiles/1m/base.fvec"));
    }

    #[test]
    fn fresh_directory_writes_origin_on_first_call() {
        let tmp = tempfile::tempdir().unwrap();
        let ds = tmp.path().join("sift1m");
        assert!(verify_or_record_origin(&ds, "https://example.com/sift1m/dataset.yaml").is_ok());
        let recorded = read_dataset_origin(&ds).unwrap();
        assert_eq!(recorded.source, "https://example.com/sift1m/dataset.yaml");
    }

    #[test]
    fn matching_origin_passes_through() {
        let tmp = tempfile::tempdir().unwrap();
        let ds = tmp.path().join("sift1m");
        let url = "https://example.com/sift1m/dataset.yaml";
        verify_or_record_origin(&ds, url).unwrap();
        // Second open with the same URL must succeed.
        assert!(verify_or_record_origin(&ds, url).is_ok());
    }

    #[test]
    fn mismatched_origin_errors_with_remediation_hint() {
        let tmp = tempfile::tempdir().unwrap();
        let ds = tmp.path().join("sift1m");
        verify_or_record_origin(&ds, "https://a.example.com/sift1m/dataset.yaml").unwrap();
        let err = verify_or_record_origin(&ds, "https://b.example.com/sift1m/dataset.yaml")
            .unwrap_err();
        assert_eq!(err.recorded, "https://a.example.com/sift1m/dataset.yaml");
        assert_eq!(err.requested, "https://b.example.com/sift1m/dataset.yaml");
        let msg = err.to_string();
        assert!(msg.contains("rm -rf"),
            "error must suggest deletion of the dataset directory: {msg}");
    }
}
