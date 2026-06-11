// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cache-root inspection helpers — the read side of the natural cache
//! layout owned by [`crate::cache::layout`].
//!
//! # Layout in force
//!
//! ```text
//! <cache_root>/
//!   <dataset_a>/                ← one directory per dataset
//!     origin.json               ← marker: source URL + fetched_at
//!     base.fvec
//!     base.fvec.mrkl
//!     profiles/1m/base.fvec
//!     ...
//!   <dataset_b>/
//!     origin.json
//!     ...
//!
//!   # Legacy detritus (purgeable via prune_legacy_layout):
//!   blobs/<hex>/<hex>/<file>    ← content-addressed cache, pre-cutover
//!   http/<hex>/<hex>/<file>     ← URL-keyed direct-HTTP cache, pre-cutover
//!   <host>[:<port>]/...         ← pre-1.1 layout (older than blobs/http)
//! ```
//!
//! # What this module does
//!
//! - [`list_entries`] classifies every top-level entry under `cache_root`
//!   into one of three buckets: natural-layout datasets (the marker is
//!   the per-dataset `origin.json` written by [`crate::cache::layout`]),
//!   legacy detritus, or unrecognised "other".
//! - [`prune_by_filter`] removes natural-layout dataset directories
//!   whose name matches a glob.
//! - [`prune_legacy_layout`] removes every legacy directory in one
//!   sweep — `blobs/`, `http/`, and any pre-1.1 `<host>[:<port>]/`
//!   shape. The user-visible affordance for "clean up everything that
//!   came before the cutover".
//!
//! # What this module does NOT do
//!
//! - It does not write anything. The write side of the natural layout
//!   (origin recording, collision detection) lives in
//!   [`crate::cache::layout`]. This module is read-only.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Pre-cutover top-level directory name: content-addressed merkle-
/// verified blob cache. Surfaced as legacy; pruned by
/// [`prune_legacy_layout`].
pub const LEGACY_BLOBS_DIR: &str = "blobs";
/// Pre-cutover top-level directory name: URL-keyed direct-HTTP cache.
/// Surfaced as legacy; pruned by [`prune_legacy_layout`].
pub const LEGACY_HTTP_DIR: &str = "http";

/// True when `name` is a top-level cache directory from any
/// pre-cutover layout. Three shapes qualify:
///   - `"blobs"` — the content-addressed bucket (post-1.1, pre-cutover)
///   - `"http"`  — the URL-keyed direct-HTTP bucket (post-1.1, pre-cutover)
///   - `<host>[:<port>]` — the bare-authority layout (pre-1.1)
///
/// All three are purgeable detritus. The natural-layout marker is
/// `origin.json` *inside* a dataset directory; this function operates
/// at the level above, classifying top-level names without filesystem
/// access.
pub fn is_legacy_layout_dir(name: &str) -> bool {
    if name == LEGACY_BLOBS_DIR || name == LEGACY_HTTP_DIR {
        return true;
    }
    // Bare authority shape: optional `<host>:<port>` suffix. We
    // intentionally accept only what the pre-1.1 emitter could
    // produce — anything else (catalog datasets, user-managed dirs)
    // stays untouched. IPv6 bracket form was never emitted by the
    // old code so we don't try to recognise it.
    let (host, port) = match name.split_once(':') {
        Some((h, p)) => (h, Some(p)),
        None => (name, None),
    };
    if host.is_empty() {
        return false;
    }
    let host_looks_like_host = host == "localhost"
        || host.split('.').all(|seg| !seg.is_empty()
            && seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
            && host.contains('.');
    if !host_looks_like_host {
        return false;
    }
    if let Some(p) = port {
        return !p.is_empty() && p.chars().all(|c| c.is_ascii_digit());
    }
    true
}

/// True when a directory tree contains a natural-layout
/// `origin.json` marker anywhere beneath it (bounded depth).
///
/// Authority-shaped top-level names are NOT sufficient evidence of
/// pre-cutover detritus: the URL-derived natural layout
/// (`crate::storage::layout_for_url`) also creates
/// `<cache>/<host>/<path…>/` trees — for direct URL opens that have
/// no catalog identity to anchor a dataset directory on — and
/// those carry `origin.json` at each dataset directory inside.
/// Classifying (and pruning!) by name alone would delete live
/// caches. Name shape narrows the candidates; this marker decides.
pub(crate) fn contains_origin_json(root: &Path) -> bool {
    fn walk(dir: &Path, depth: usize) -> bool {
        if depth == 0 { return false; }
        let Ok(entries) = fs::read_dir(dir) else { return false; };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file()
                && p.file_name().is_some_and(|n| n == crate::cache::layout::DATASET_ORIGIN_FILE)
            {
                return true;
            }
            if p.is_dir() && walk(&p, depth - 1) {
                return true;
            }
        }
        false
    }
    walk(root, 8)
}

// ─── Targeted prune ─────────────────────────────────────────────────

/// Filter for [`prune_by_filter`]: dataset directory names matching
/// `dataset` (glob with `*`/`?`, or exact) are removed. Profile-level
/// filtering is intentionally absent — the natural layout does not
/// surface profiles as a cache-level concept (a single facet file
/// may be shared by many windowed profiles), so a per-profile delete
/// would be meaningless at this layer.
#[derive(Debug, Default, Clone)]
pub struct PruneFilter {
    /// Glob pattern (with `*`/`?`) matched against each dataset
    /// directory's name.
    pub dataset: Option<String>,
}

impl PruneFilter {
    /// True when no filter is set. Callers should refuse to prune
    /// with an empty filter — that would wipe every cached dataset,
    /// which `prune_legacy_layout` already handles for the case it
    /// makes sense (legacy cleanup).
    pub fn is_empty(&self) -> bool {
        self.dataset.is_none()
    }
}

/// Summary returned by [`prune_by_filter`]. Reports both what matched
/// and what was actually removed so dry-run callers can preview
/// without performing the deletion.
#[derive(Debug, Default)]
pub struct PruneReport {
    /// Entries whose dataset name matched the filter.
    pub matched: Vec<CacheEntry>,
    /// Subset of `matched` that were successfully deleted (empty
    /// for `dry_run=true`).
    pub removed: Vec<CacheEntry>,
    /// Total on-disk bytes across `removed`.
    pub bytes_freed: u64,
}

/// Walk `cache_root` and remove every natural-layout dataset
/// directory whose name matches `filter.dataset`. Pass `dry_run=true`
/// to preview.
///
/// Only natural-layout datasets are candidates — legacy `blobs/` /
/// `http/` / `<host>:<port>/` directories are deliberately skipped.
/// Legacy cleanup is the job of [`prune_legacy_layout`], which has a
/// different (no-filter) shape because legacy entries don't have a
/// reliable dataset identity to filter on.
pub fn prune_by_filter(
    cache_root: &Path,
    filter: &PruneFilter,
    dry_run: bool,
) -> io::Result<PruneReport> {
    let listing = list_entries(cache_root)?;
    let mut report = PruneReport::default();
    let Some(pattern) = filter.dataset.as_deref() else { return Ok(report); };

    for entry in &listing.datasets {
        let name = entry.path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !glob_match(pattern, name) { continue; }
        report.matched.push(entry.clone());
        if !dry_run {
            match fs::remove_dir_all(&entry.path) {
                Ok(()) => {
                    report.bytes_freed += entry.size_bytes;
                    report.removed.push(entry.clone());
                }
                Err(e) => {
                    eprintln!("warning: failed to remove {}: {}",
                        entry.path.display(), e);
                }
            }
        }
    }
    Ok(report)
}

/// Minimal glob matcher supporting `*` (any run of characters) and
/// `?` (one character). Bare strings (no wildcards) match exactly.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    fn inner(p: &[char], t: &[char], pi: usize, ti: usize) -> bool {
        if pi == p.len() { return ti == t.len(); }
        match p[pi] {
            '*' => (ti..=t.len()).any(|i| inner(p, t, pi + 1, i)),
            '?' => ti < t.len() && inner(p, t, pi + 1, ti + 1),
            c   => ti < t.len() && t[ti] == c && inner(p, t, pi + 1, ti + 1),
        }
    }
    inner(&p, &t, 0, 0)
}

/// One row in a [`CacheListing`].
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Absolute on-disk path of the entry's top-level cache directory.
    pub path: PathBuf,
    /// Bytes actually allocated on disk under `path` (`du`
    /// semantics). Sparse cache files report only their downloaded
    /// chunks, not the pre-sized apparent length.
    pub size_bytes: u64,
    /// Number of regular files under `path`.
    pub file_count: usize,
    /// Originating URL recovered from `origin.json` (natural-layout
    /// datasets only; `None` for legacy and other entries).
    pub origin_url: Option<String>,
    /// Host of `origin_url`, if known.
    pub origin_host: Option<String>,
}

/// A structured view of the cache root, suitable for rendering by a
/// CLI or programmatic inspection. The walker never errors on an
/// unrecognised top-level entry; it goes into [`Self::other`] so the
/// user can review it.
#[derive(Debug, Default)]
pub struct CacheListing {
    /// Natural-layout datasets — one entry per top-level directory
    /// containing an `origin.json` recorded by the cache write side.
    pub datasets: Vec<CacheEntry>,
    /// URL-derived natural caches — authority-named top-level trees
    /// holding per-dataset `origin.json` markers deeper inside.
    /// Produced only by direct URL opens that carry no catalog
    /// identity (catalog-anchored opens always use the dataset-keyed
    /// layout). Live data — never pruned by [`prune_legacy_layout`].
    pub url_derived: Vec<CacheEntry>,
    /// Pre-cutover detritus — `blobs/`, `http/`, and the older
    /// `<host>[:<port>]/` shape. Cleaned up by
    /// [`prune_legacy_layout`].
    pub legacy: Vec<CacheEntry>,
    /// Top-level entries we did not recognise. Left untouched by
    /// every helper in this module so users can review them.
    pub other: Vec<CacheEntry>,
}

impl CacheListing {
    /// Total bytes across every category.
    pub fn total_bytes(&self) -> u64 {
        let s = |v: &[CacheEntry]| v.iter().map(|e| e.size_bytes).sum::<u64>();
        s(&self.datasets) + s(&self.url_derived) + s(&self.legacy) + s(&self.other)
    }
}

/// Walk `cache_root` and classify every top-level entry. Read-only.
///
/// The classification rule:
///   - Directory contains `origin.json` → [`CacheListing::datasets`]
///   - Directory name matches [`is_legacy_layout_dir`] → [`CacheListing::legacy`]
///   - Anything else → [`CacheListing::other`]
///
/// This is the function `vectordata cache list` is built on.
pub fn list_entries(cache_root: &Path) -> io::Result<CacheListing> {
    let mut listing = CacheListing::default();
    let entries = match fs::read_dir(cache_root) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(listing),
        Err(e) => return Err(e),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() { continue; }
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (size, count) = dir_size_and_count(&path);
        // Natural-layout marker: the per-dataset `origin.json` written
        // by `crate::cache::layout::write_dataset_origin`. Reading it
        // here also extracts the URL for the listing.
        let origin = crate::cache::layout::read_dataset_origin(&path);
        let cache_entry = CacheEntry {
            path: path.clone(),
            size_bytes: size,
            file_count: count,
            origin_url: origin.as_ref().map(|o| o.source.clone()),
            origin_host: origin.as_ref().and_then(|o| origin_host(&o.source)),
        };
        if origin.is_some() {
            listing.datasets.push(cache_entry);
        } else if contains_origin_json(&path) {
            listing.url_derived.push(cache_entry);
        } else if is_legacy_layout_dir(&name) {
            listing.legacy.push(cache_entry);
        } else {
            listing.other.push(cache_entry);
        }
    }
    sort_by_size_desc(&mut listing.datasets);
    sort_by_size_desc(&mut listing.url_derived);
    sort_by_size_desc(&mut listing.legacy);
    sort_by_size_desc(&mut listing.other);
    Ok(listing)
}

/// Extract a hostname from a URL string. Used to decorate
/// [`CacheEntry`]s with their origin host for grouped CLI display
/// without forcing every caller to re-parse the URL.
fn origin_host(url: &str) -> Option<String> {
    url::Url::parse(url).ok().and_then(|u| u.host_str().map(str::to_string))
}

fn sort_by_size_desc(v: &mut [CacheEntry]) {
    v.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
}

/// Bytes a file actually occupies on disk (`du` semantics): allocated
/// blocks, not apparent length. The distinction is load-bearing for
/// every cache-size surface — chunk stores pre-size sparse cache
/// files to the full remote size at open, so `metadata().len()`
/// reports a one-chunk download as the whole dataset. On non-Unix
/// targets (no block count in std) this falls back to apparent size.
pub(crate) fn allocated_size(meta: &std::fs::Metadata) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.blocks() * 512
    }
    #[cfg(not(unix))]
    {
        meta.len()
    }
}

fn dir_size_and_count(path: &Path) -> (u64, usize) {
    let mut size = 0u64;
    let mut count = 0usize;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for e in entries.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(p),
                Ok(_) => {
                    if let Ok(meta) = e.metadata() {
                        size += allocated_size(&meta);
                        count += 1;
                    }
                }
                Err(_) => {}
            }
        }
    }
    (size, count)
}

/// Remove every top-level entry in `cache_root` whose name matches
/// [`is_legacy_layout_dir`]. Returns the names deleted, in arbitrary
/// order. Idempotent.
///
/// Natural-layout dataset directories are never touched — they're
/// removed via [`prune_by_filter`] (selective) or the picker's
/// Purge action (per-dataset, catalog-aware).
pub fn prune_legacy_layout(cache_root: &Path) -> io::Result<Vec<String>> {
    let mut removed = Vec::new();
    let entries = match fs::read_dir(cache_root) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(removed),
        Err(e) => return Err(e),
    };
    for entry in entries.flatten() {
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        if !is_legacy_layout_dir(&name) {
            continue;
        }
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // Authority-shaped name is necessary but not sufficient: a
        // URL-derived natural cache has the same top-level shape and
        // must survive. The origin.json marker is the discriminator.
        if contains_origin_json(&path) {
            continue;
        }
        fs::remove_dir_all(&path)?;
        removed.push(name);
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_layout_dir_recognizes_pre_cutover_shapes() {
        // The two cutover-era content-addressed buckets.
        assert!(is_legacy_layout_dir("blobs"));
        assert!(is_legacy_layout_dir("http"));
        // The pre-1.1 bare-authority shape.
        assert!(is_legacy_layout_dir("127.0.0.1:32775"));
        assert!(is_legacy_layout_dir("127.0.0.1"));
        assert!(is_legacy_layout_dir("localhost:8080"));
        assert!(is_legacy_layout_dir("localhost"));
        assert!(is_legacy_layout_dir(
            "vectordata-pvs-testing.s3.us-east-1.amazonaws.com"));
    }

    #[test]
    fn legacy_layout_dir_rejects_natural_dataset_names() {
        // Catalog dataset names (no dots, no auth-shape) — should be
        // left alone.
        assert!(!is_legacy_layout_dir("vecs1m"));
        assert!(!is_legacy_layout_dir("example-1b"));
        assert!(!is_legacy_layout_dir(""));
    }

    /// A sparse pre-sized cache file (the shape every chunk store
    /// leaves behind at open) must report only its written bytes,
    /// not the apparent length — this is the regression that made
    /// one downloaded chunk list as a fully-cached dataset.
    #[cfg(unix)]
    #[test]
    fn allocated_size_ignores_sparse_holes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sparse.bin");
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(64 * 1024 * 1024).unwrap(); // 64 MiB apparent
        drop(f);
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&vec![7u8; 1024 * 1024]).unwrap(); // 1 MiB real
            f.sync_all().unwrap();
        }
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 64 * 1024 * 1024);
        let alloc = allocated_size(&meta);
        assert!(alloc >= 1024 * 1024, "written bytes must be counted, got {alloc}");
        assert!(alloc < 8 * 1024 * 1024,
            "sparse holes must not be counted (apparent 64 MiB, got {alloc})");
    }

    #[test]
    fn glob_match_basic_patterns() {
        assert!(glob_match("vecs1m", "vecs1m"));
        assert!(!glob_match("vecs1m", "vecs1m-2"));
        assert!(glob_match("vecs*", "vecs1m"));
        assert!(glob_match("*1m", "vecs1m"));
        assert!(glob_match("vecs?m", "vecs1m"));
        assert!(!glob_match("vecs?m", "vecs10m"));
        assert!(glob_match("label_*", "label_03"));
        assert!(!glob_match("label_*", "default"));
        assert!(glob_match("*", "anything"));
    }

    /// Seed a natural-layout dataset directory by writing the
    /// `origin.json` the cache write side would have produced.
    fn seed_dataset(root: &Path, name: &str, url: &str) -> PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        crate::cache::layout::write_dataset_origin(&dir, url).unwrap();
        fs::write(dir.join("base.fvec"), b"data").unwrap();
        dir
    }

    #[test]
    fn list_entries_classifies_natural_legacy_and_other() {
        // Only exercises shapes that can actually exist on every
        // platform we ship for. The pre-1.1 `<host>[:<port>]/`
        // cache layout was a defect — NTFS reserves `:` in path
        // components, so it could never have produced such a
        // directory on Windows — and we don't reproduce that
        // mistake here just to assert detection. Name-shape
        // recognition is covered by
        // `legacy_layout_dir_recognizes_pre_cutover_shapes` as a
        // pure-string unit test that runs everywhere.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Natural-layout dataset.
        seed_dataset(root, "vecs1m", "https://example.com/vecs1m/");
        // Legacy buckets — the two cutover-era shapes that are
        // cross-platform safe to construct.
        fs::create_dir_all(root.join("blobs/aa/bb")).unwrap();
        fs::write(root.join("blobs/aa/bb/data"), b"x").unwrap();
        fs::create_dir_all(root.join("http/cc/dd")).unwrap();
        fs::write(root.join("http/cc/dd/data"), b"y").unwrap();
        // Unrecognised top-level dir.
        fs::create_dir_all(root.join("weirdo")).unwrap();
        fs::write(root.join("weirdo/file"), b"z").unwrap();

        let listing = list_entries(root).unwrap();
        let names = |v: &[CacheEntry]| v.iter()
            .map(|e| e.path.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>();

        assert_eq!(names(&listing.datasets), vec!["vecs1m"]);
        let mut leg = names(&listing.legacy); leg.sort();
        assert_eq!(leg, vec!["blobs", "http"]);
        assert_eq!(names(&listing.other), vec!["weirdo"]);
    }

    #[test]
    fn list_entries_records_dataset_origin() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_dataset(root, "vecs1m", "https://example.com/vecs1m/");
        let listing = list_entries(root).unwrap();
        assert_eq!(listing.datasets.len(), 1);
        let entry = &listing.datasets[0];
        assert_eq!(entry.origin_url.as_deref(),
            Some("https://example.com/vecs1m/"));
        assert_eq!(entry.origin_host.as_deref(), Some("example.com"));
    }

    #[test]
    fn prune_by_filter_dry_run_lists_matches_without_removal() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_dataset(root, "vecs1m", "https://example.com/vecs1m/");
        seed_dataset(root, "glove", "https://example.com/glove/");

        let report = prune_by_filter(root,
            &PruneFilter { dataset: Some("vecs*".into()) }, true).unwrap();
        assert_eq!(report.matched.len(), 1);
        assert_eq!(report.removed.len(), 0, "dry-run must not delete");
        assert!(root.join("vecs1m").exists() && root.join("glove").exists());
    }

    #[test]
    fn prune_by_filter_removes_matching_datasets() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        seed_dataset(root, "vecs1m", "https://example.com/vecs1m/");
        seed_dataset(root, "glove", "https://example.com/glove/");

        let report = prune_by_filter(root,
            &PruneFilter { dataset: Some("vecs1m".into()) }, false).unwrap();
        assert_eq!(report.matched.len(), 1);
        assert_eq!(report.removed.len(), 1);
        assert!(!root.join("vecs1m").exists(), "matched dataset must be removed");
        assert!(root.join("glove").exists(),   "non-matched dataset must survive");
    }

    #[test]
    fn prune_by_filter_skips_legacy_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Glob `*` would otherwise sweep everything; legacy entries
        // must be invisible to dataset-level prune.
        fs::create_dir_all(root.join("blobs/aa/bb")).unwrap();
        fs::write(root.join("blobs/aa/bb/data"), b"x").unwrap();
        seed_dataset(root, "vecs1m", "https://example.com/vecs1m/");

        let report = prune_by_filter(root,
            &PruneFilter { dataset: Some("*".into()) }, false).unwrap();
        assert_eq!(report.removed.len(), 1);
        assert!(!root.join("vecs1m").exists());
        assert!(root.join("blobs").exists(), "legacy must survive dataset prune");
    }

    // Unix-only: the legacy <host>:<port> shape this test simulates
    // could never have existed on Windows (the `:` is reserved in
    // path components), so there's nothing for prune_legacy_layout
    // to clean up there.
    #[cfg(not(windows))]
    #[test]
    fn prune_legacy_layout_removes_all_legacy_shapes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for name in ["127.0.0.1:32775", "127.0.0.1", "localhost:8080",
                     "blobs", "http"] {
            fs::create_dir_all(root.join(name)).unwrap();
            fs::write(root.join(name).join("marker"), name).unwrap();
        }
        // A natural-layout dataset that must NOT be touched.
        seed_dataset(root, "vecs1m", "https://example.com/vecs1m/");

        // A URL-derived natural cache: authority-shaped TOP-LEVEL name
        // but with a nested per-dataset origin.json — the layout that
        // direct URL opens and out-of-home catalog facets produce.
        // Must NOT be classified or pruned as legacy.
        let url_derived = root.join("example.s3.amazonaws.com/data/vecs");
        fs::create_dir_all(&url_derived).unwrap();
        fs::write(url_derived.join("origin.json"),
            r#"{"source":"https://example.s3.amazonaws.com/data/vecs/","fetched_at":"x"}"#).unwrap();
        fs::write(url_derived.join("base.fvec"), b"data").unwrap();

        let mut removed = prune_legacy_layout(root).unwrap();
        removed.sort();
        assert_eq!(removed, vec![
            "127.0.0.1".to_string(),
            "127.0.0.1:32775".to_string(),
            "blobs".to_string(),
            "http".to_string(),
            "localhost:8080".to_string(),
        ]);
        assert!(root.join("vecs1m/origin.json").exists(),
            "natural-layout datasets must survive legacy purge");
        assert!(url_derived.join("base.fvec").exists(),
            "URL-derived natural caches must survive legacy purge");

        let listing = list_entries(root).unwrap();
        assert!(listing.legacy.is_empty(),
            "nothing legacy-shaped remains after prune");
        assert!(listing.url_derived.iter().any(|e|
            e.path.file_name().is_some_and(|n| n == "example.s3.amazonaws.com")),
            "URL-derived cache must list in its own live group, not as detritus");
    }
}
