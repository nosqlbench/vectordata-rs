// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cache-directory layout helpers used by
//! [`crate::storage::Storage::open_url_cached`] and
//! [`crate::storage::Storage::open_url_http`].
//!
//! # Layout
//!
//! The cache root is resolved once via [`crate::settings::cache_dir`].
//! Underneath it, two parallel subtrees hold all remotely-fetched bytes:
//!
//! ```text
//! <cache_root>/
//!   blobs/<hex[..2]>/<full-hex>/
//!     <filename>            cached file body
//!     <filename>.mrkl       merkle verification state
//!     origin.json           { url, fetched_at } — tooling sidecar
//!   http/<hex[..2]>/<full-hex>/
//!     <filename>            direct-HTTP cached body (no merkle)
//!     origin.json
//! ```
//!
//! - `blobs/` is *content-addressed*: the directory key is the hex of
//!   the merkle reference's root hash. Two URLs serving identical
//!   bytes share one blob dir; two URLs at the same path serving
//!   different bytes occupy disjoint dirs. There is no `<host>:<port>`
//!   anywhere — ephemeral test fixtures can no longer leak per-port
//!   directories.
//!
//! - `http/` is *URL-addressed* and used only when the upstream does
//!   not publish a `.mref`. The key is SHA-256 of the canonical URL
//!   string. Without a merkle reference we cannot dedupe by content,
//!   so the URL is the next-best stable identity. The fix to stale
//!   bytes on this path is handled at the storage layer (length check
//!   on restore) — there is no merkle to reconcile.
//!
//! The two-character `<hex[..2]>` prefix bounds fanout under each
//! subtree so single directories never grow unbounded.
//!
//! # Why a sidecar
//!
//! `origin.json` records the URL that originally populated the blob.
//! User-facing tools (`veks datasets cache`, `veks datasets drop-cache`)
//! use it to group entries by dataset, hostname, or any other
//! URL-derived attribute. Without it, a content-addressed layout would
//! be opaque to humans.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use url::Url;

use sha2::{Digest, Sha256};

use crate::merkle::MerkleRef;

/// Top-level subdir for content-addressed (merkle-verified) blobs.
pub const BLOBS_DIR: &str = "blobs";
/// Top-level subdir for URL-addressed (no-`.mref`) cached bodies.
pub const HTTP_DIR: &str = "http";
/// Sidecar filename recording the originating URL for a cache entry.
pub(crate) const ORIGIN_FILE: &str = "origin.json";

/// Whether `name` is a top-level subdir name reserved by the cache
/// layout. Tools that enumerate the cache root (e.g. `veks datasets
/// cache` / `drop-cache`) skip these so they don't display the
/// content-addressed buckets as if they were catalog-managed datasets.
pub fn is_reserved_layout_name(name: &str) -> bool {
    name == BLOBS_DIR || name == HTTP_DIR
}

/// Crate-internal alias for [`crate::settings::cache_dir`]. Kept as a
/// thin wrapper so existing call sites stay terse; the actual
/// resolution lives in `settings.rs` so `vectordata` and consuming
/// crates share one canonical implementation.
pub(crate) fn default_cache_dir() -> Result<PathBuf, crate::settings::SettingsError> {
    crate::settings::cache_dir()
}

/// Hex-encode bytes (lowercase) without pulling the `hex` crate into
/// runtime deps.
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Two-character prefix dir for fanout bounding, plus full hex.
fn split_hex(hex: &str) -> (&str, &str) {
    // hex is always at least 4 chars for our uses (SHA-256 / merkle-root
    // sizes), so this is safe.
    (&hex[..2], hex)
}

/// Resolve the blob directory for a merkle-verified resource.
///
/// The directory is keyed solely by the merkle root hash — it does
/// not depend on URL, host, or port. Two callers requesting different
/// URLs that resolve to identical bytes share this directory.
pub(crate) fn blob_dir_for_mref(cache_root: &Path, mref: &MerkleRef) -> PathBuf {
    let hex = hex_encode(mref.root_hash());
    let (prefix, full) = split_hex(&hex);
    cache_root.join(BLOBS_DIR).join(prefix).join(full)
}

/// Resolve the cache directory for a URL with no published `.mref`.
///
/// The directory is keyed by SHA-256 of the canonical URL string.
/// This is the fallback path: without merkle hashes we cannot content-
/// dedupe, so URL identity is the next-best key. Filesystem-safe in
/// all cases (lowercase hex only).
pub(crate) fn blob_dir_for_url(cache_root: &Path, url: &Url) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(url.as_str().as_bytes());
    let hex = hex_encode(&hasher.finalize());
    let (prefix, full) = split_hex(&hex);
    cache_root.join(HTTP_DIR).join(prefix).join(full)
}

/// Originating-URL sidecar recorded into each blob dir for tooling.
///
/// Tools that want to display the cache as a human-friendly list
/// (grouped by dataset / hostname) read this sidecar instead of trying
/// to invert the content hash. Format is intentionally tiny JSON so it
/// stays readable with `cat`.
#[derive(Debug)]
pub(crate) struct Origin {
    pub url: String,
    /// RFC 3339 timestamp of when the blob was first populated.
    /// Read into `list_entries` but not surfaced through `CacheEntry`
    /// yet — the CLI prints fetched_at only on per-blob detail rows
    /// once we add that flag. Kept for forward-compat.
    #[allow(dead_code)]
    pub fetched_at: String,
}

/// Write `origin.json` into a blob directory. Best-effort: a failure
/// here doesn't break the cache, it just leaves an unattributable
/// blob behind.
pub(crate) fn write_origin(dir: &Path, url: &Url) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join(ORIGIN_FILE);
    if path.exists() {
        // First writer wins — don't overwrite a recorded URL when a
        // second URL ends up dedup'd into the same content blob. We
        // could append, but for the immediate use cases (display +
        // grouping) one canonical URL is enough.
        return Ok(());
    }
    let fetched_at = httpdate::fmt_http_date(std::time::SystemTime::now());
    let json = format!(
        "{{\n  \"url\": {},\n  \"fetched_at\": {}\n}}\n",
        json_escape(url.as_str()),
        json_escape(&fetched_at),
    );
    fs::write(&path, json)
}

/// Read an origin sidecar. Returns `None` on absent / malformed file —
/// origin metadata is advisory, never load-bearing.
#[allow(dead_code)] // Used by the (forthcoming) `vectordata cache list` admin command.
pub(crate) fn read_origin(dir: &Path) -> Option<Origin> {
    let path = dir.join(ORIGIN_FILE);
    let content = fs::read_to_string(&path).ok()?;
    let url = extract_json_string(&content, "url")?;
    let fetched_at = extract_json_string(&content, "fetched_at")
        .unwrap_or_else(|| "".to_string());
    Some(Origin { url, fetched_at })
}

/// Minimal JSON string escaper for sidecar writing — we control the
/// content shape so we only need to handle `"` and `\`.
fn json_escape(s: &str) -> String {
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

/// Extract `"key": "value"` from the sidecar without a JSON dep. The
/// file is single-writer and tiny, so a regex-light scan is fine.
fn extract_json_string(content: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let after_key = content.split_once(&needle)?.1;
    let after_colon = after_key.split_once(':')?.1;
    let trimmed = after_colon.trim_start();
    if !trimmed.starts_with('"') {
        return None;
    }
    let body = &trimmed[1..];
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    None
}

// ─── Targeted prune ─────────────────────────────────────────────────

/// Filter for [`prune_by_filter`]: blobs match when *both* fields
/// (if set) match the entry's extracted dataset and profile names.
#[derive(Debug, Default, Clone)]
pub struct PruneFilter {
    /// Glob pattern (with `*`/`?`) matched against the dataset name
    /// extracted from the entry's origin URL.
    pub dataset: Option<String>,
    /// Glob pattern matched against the profile name extracted from
    /// the entry's origin URL.
    pub profile: Option<String>,
}

impl PruneFilter {
    /// True when at least one filter dimension is set. Callers
    /// should refuse to prune with an empty filter — that would
    /// nuke the entire cache, which is what `prune-legacy` exists
    /// for in a more deliberate form.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_none() && self.profile.is_none()
    }
}

/// Summary returned by [`prune_by_filter`]. Reports both what
/// matched and what was actually removed so dry-run callers can
/// preview without performing the deletion.
#[derive(Debug, Default)]
pub struct PruneReport {
    /// Entries whose extracted dataset/profile matched the filter.
    pub matched: Vec<CacheEntry>,
    /// Subset of `matched` that were successfully deleted (empty
    /// for `dry_run=true`).
    pub removed: Vec<CacheEntry>,
    /// Total on-disk bytes across `removed`.
    pub bytes_freed: u64,
}

/// Walk `cache_root` and remove every content-addressed / URL-keyed
/// cache entry whose origin URL matches `filter`. The matcher
/// extracts the dataset name and profile name from the URL path —
/// the standard published shape is
/// `…/<dataset>/profiles/<profile>/<file>` — and runs a `*`/`?`
/// glob against each.
///
/// Entries with no `origin.json` sidecar, or where the URL doesn't
/// fit the expected shape, are *not* pruned regardless of filter —
/// "unknown origin" is treated conservatively.
///
/// Pass `dry_run=true` to preview which entries would be removed.
pub fn prune_by_filter(
    cache_root: &Path,
    filter: &PruneFilter,
    dry_run: bool,
) -> io::Result<PruneReport> {
    let listing = list_entries(cache_root)?;
    let mut report = PruneReport::default();

    let candidates: Vec<&CacheEntry> = listing.blobs.iter()
        .chain(listing.http.iter())
        .collect();

    for entry in candidates {
        if !entry_matches_filter(entry, filter) { continue; }
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

fn entry_matches_filter(entry: &CacheEntry, filter: &PruneFilter) -> bool {
    // No URL → don't match. The user can't reason about it via the
    // filter anyway, and we don't want to nuke things we can't
    // describe.
    let Some(url) = entry.origin_url.as_deref() else { return false; };
    let (dataset, profile) = extract_dataset_profile(url);
    if let Some(pat) = &filter.dataset {
        let Some(name) = dataset.as_deref() else { return false; };
        if !glob_match(pat, name) { return false; }
    }
    if let Some(pat) = &filter.profile {
        let Some(name) = profile.as_deref() else { return false; };
        if !glob_match(pat, name) { return false; }
    }
    true
}

/// Pull `(dataset_name, profile_name)` out of an origin URL.
///
/// Recognises the canonical published shape
/// `…/<dataset>/profiles/<profile>/<file>` — dataset is the path
/// segment immediately preceding `profiles`, profile is the segment
/// immediately following it.
///
/// Falls back to `(<parent-dir>, None)` for URLs that don't carry a
/// `profiles` segment — the parent directory in the URL path is
/// typically the dataset name (e.g. `…/sift1m/dataset.yaml`).
fn extract_dataset_profile(url: &str) -> (Option<String>, Option<String>) {
    let parsed = match Url::parse(url) { Ok(u) => u, Err(_) => return (None, None) };
    let segments: Vec<&str> = match parsed.path_segments() {
        Some(it) => it.filter(|s| !s.is_empty()).collect(),
        None => return (None, None),
    };
    if let Some(idx) = segments.iter().position(|s| *s == "profiles") {
        let dataset = if idx > 0 { Some(segments[idx - 1].to_string()) } else { None };
        let profile = segments.get(idx + 1).map(|s| s.to_string());
        return (dataset, profile);
    }
    if segments.len() >= 2 {
        (Some(segments[segments.len() - 2].to_string()), None)
    } else {
        (None, None)
    }
}

/// Minimal glob matcher supporting `*` (any run of characters) and
/// `?` (one character). Bare strings (no wildcards) match exactly.
/// Used by [`prune_by_filter`]; intentionally not exposed publicly
/// to keep the crate's surface focused.
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

/// Match an old-layout top-level cache-dir name (`<host>` or
/// `<host>:<port>`) so the legacy-prune utility can delete them. We
/// intentionally accept only the shapes that the old code emitted —
/// anything else (catalog datasets, user-managed dirs) is left alone.
pub fn is_legacy_layout_name(name: &str) -> bool {
    if name == BLOBS_DIR || name == HTTP_DIR {
        return false;
    }
    // Bare IPv4 / hostname with optional `:port` suffix. We don't try
    // to be clever about IPv6 — the old code didn't emit `[::1]:port`
    // form, so we never had to deal with bracketed-host directories.
    let (host, port) = match name.split_once(':') {
        Some((h, p)) => (h, Some(p)),
        None => (name, None),
    };
    if host.is_empty() {
        return false;
    }
    // Host: at least one dot or a known localhost name. We don't want
    // to match arbitrary user-named dirs like "sift1m".
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

/// One row in a [`CacheListing`].
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Absolute on-disk path to the cache directory holding the file.
    pub path: PathBuf,
    /// Sum of all file sizes under `path` (bytes).
    pub size_bytes: u64,
    /// Number of regular files under `path`.
    pub file_count: usize,
    /// Originating URL recovered from `origin.json`, if present.
    pub origin_url: Option<String>,
    /// Host of `origin_url`, if known — extracted so callers can
    /// group entries without re-parsing the URL.
    pub origin_host: Option<String>,
}

/// A structured view of the cache root, suitable for rendering by a
/// CLI or programmatic inspection.
///
/// Each field corresponds to one layout-level category — callers
/// decide how to render. The walker does not error on unrecognised
/// top-level entries; they appear in [`Self::other`] so users can see
/// (and curate) anything unexpected.
#[derive(Debug, Default)]
pub struct CacheListing {
    /// `<cache_root>/<name>/dataset.yaml` — catalog-managed datasets.
    pub catalog_datasets: Vec<CacheEntry>,
    /// Content-addressed blobs under `<cache_root>/blobs/`.
    pub blobs: Vec<CacheEntry>,
    /// URL-addressed direct-HTTP cache under `<cache_root>/http/`.
    pub http: Vec<CacheEntry>,
    /// Pre-content-addressed legacy `<host>[:<port>]` dirs awaiting
    /// migration via [`prune_legacy_layout`].
    pub legacy: Vec<CacheEntry>,
    /// Anything else at the top level we did not recognise — left
    /// untouched by tools so the user can review.
    pub other: Vec<CacheEntry>,
}

impl CacheListing {
    /// Total bytes across every category.
    pub fn total_bytes(&self) -> u64 {
        let s = |v: &[CacheEntry]| v.iter().map(|e| e.size_bytes).sum::<u64>();
        s(&self.catalog_datasets) + s(&self.blobs) + s(&self.http)
            + s(&self.legacy) + s(&self.other)
    }
}

/// Walk `cache_root` and classify every top-level entry. Read-only —
/// callers compose this with [`prune_legacy_layout`] or their own
/// deletion logic when they want to act on the result.
///
/// This is the function `vectordata cache list` is built on; veks
/// tools should use it too rather than re-implementing the walk.
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
        match name.as_str() {
            BLOBS_DIR => collect_addressed_entries(&path, &mut listing.blobs)?,
            HTTP_DIR  => collect_addressed_entries(&path, &mut listing.http)?,
            _ if is_legacy_layout_name(&name) => {
                let (size, count) = dir_size_and_count(&path);
                listing.legacy.push(CacheEntry {
                    path, size_bytes: size, file_count: count,
                    origin_url: None, origin_host: None,
                });
            }
            _ => {
                // Catalog-managed datasets carry a top-level
                // `dataset.yaml`; anything else falls into `other`.
                let (size, count) = dir_size_and_count(&path);
                let entry = CacheEntry {
                    path: path.clone(), size_bytes: size, file_count: count,
                    origin_url: None, origin_host: None,
                };
                if path.join("dataset.yaml").exists() {
                    listing.catalog_datasets.push(entry);
                } else {
                    listing.other.push(entry);
                }
            }
        }
    }
    sort_by_size_desc(&mut listing.catalog_datasets);
    sort_by_size_desc(&mut listing.blobs);
    sort_by_size_desc(&mut listing.http);
    sort_by_size_desc(&mut listing.legacy);
    sort_by_size_desc(&mut listing.other);
    Ok(listing)
}

/// Walk a `blobs/` or `http/` subtree: two levels (prefix, full-hex)
/// of fanout, then each leaf is one cache entry.
fn collect_addressed_entries(root: &Path, out: &mut Vec<CacheEntry>) -> io::Result<()> {
    let prefixes = match fs::read_dir(root) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for prefix in prefixes.flatten() {
        let pp = prefix.path();
        if !pp.is_dir() { continue; }
        let leaves = match fs::read_dir(&pp) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for leaf in leaves.flatten() {
            let lp = leaf.path();
            if !lp.is_dir() { continue; }
            let (size, count) = dir_size_and_count(&lp);
            let origin = read_origin(&lp);
            let (origin_url, origin_host) = match origin {
                Some(o) => {
                    let host = Url::parse(&o.url).ok().and_then(|u| u.host_str().map(str::to_string));
                    (Some(o.url), host)
                }
                None => (None, None),
            };
            out.push(CacheEntry {
                path: lp, size_bytes: size, file_count: count,
                origin_url, origin_host,
            });
        }
    }
    Ok(())
}

fn sort_by_size_desc(v: &mut [CacheEntry]) {
    v.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
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
                        size += meta.len();
                        count += 1;
                    }
                }
                Err(_) => {}
            }
        }
    }
    (size, count)
}

/// Remove every top-level entry in `cache_root` that matches the old
/// `<host>` / `<host>:<port>` layout. Returns the names deleted.
///
/// Idempotent and bounded — does not recurse into `blobs/` or `http/`,
/// the catalog-dataset subtrees, or anything else. Used by the
/// migration CLI.
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
        if !is_legacy_layout_name(&name) {
            continue;
        }
        let path = entry.path();
        if !path.is_dir() {
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
    fn blob_dir_for_mref_is_content_addressed() {
        let tmp = tempfile::tempdir().unwrap();
        let data = b"some bytes that will be hashed into the merkle root";
        let m1 = MerkleRef::from_content(data, 16);
        let m2 = MerkleRef::from_content(data, 16);
        let d1 = blob_dir_for_mref(tmp.path(), &m1);
        let d2 = blob_dir_for_mref(tmp.path(), &m2);
        assert_eq!(d1, d2, "identical content must map to identical blob dirs");

        let different = b"different bytes producing a different merkle root";
        let m3 = MerkleRef::from_content(different, 16);
        let d3 = blob_dir_for_mref(tmp.path(), &m3);
        assert_ne!(d1, d3, "different content must map to different blob dirs");

        // The layout shape is blobs/<hex[..2]>/<full-hex>
        let rel = d1.strip_prefix(tmp.path()).unwrap();
        let comps: Vec<_> = rel.components().collect();
        assert_eq!(comps.len(), 3, "expected blobs/<prefix>/<full>: {:?}", rel);
        assert_eq!(comps[0].as_os_str(), BLOBS_DIR);
    }

    #[test]
    fn blob_dir_for_url_is_url_addressed() {
        let tmp = tempfile::tempdir().unwrap();
        let u1 = Url::parse("http://example.com/path/to/file.fvec").unwrap();
        let u2 = Url::parse("http://example.com/path/to/file.fvec").unwrap();
        let u3 = Url::parse("http://example.com/other/file.fvec").unwrap();
        assert_eq!(blob_dir_for_url(tmp.path(), &u1), blob_dir_for_url(tmp.path(), &u2));
        assert_ne!(blob_dir_for_url(tmp.path(), &u1), blob_dir_for_url(tmp.path(), &u3));

        // The URL's host (which can carry a `:` for the port) must not
        // leak into the URL-derived cache path components. Check only
        // the suffix relative to `cache_root`; the root itself can
        // legitimately contain `:` on Windows (drive letter).
        let with_port = Url::parse("http://127.0.0.1:32775/base.fvec").unwrap();
        let dir = blob_dir_for_url(tmp.path(), &with_port);
        let suffix = dir.strip_prefix(tmp.path())
            .expect("blob dir must live under cache_root");
        let s = suffix.to_string_lossy();
        assert!(!s.contains(':'),
            "no `:` should appear in URL-derived cache dir components: {s}");
    }

    #[test]
    fn origin_sidecar_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let url = Url::parse("http://example.com/data.fvec").unwrap();
        write_origin(tmp.path(), &url).unwrap();
        let got = read_origin(tmp.path()).expect("sidecar present");
        assert_eq!(got.url, "http://example.com/data.fvec");
        assert!(!got.fetched_at.is_empty());
    }

    #[test]
    fn origin_sidecar_handles_quotes_and_backslashes() {
        let tmp = tempfile::tempdir().unwrap();
        // A real URL won't contain unescaped quotes, but the escaper
        // still has to handle whatever url::Url produces — guard
        // against shape drift.
        let url = Url::parse("http://example.com/with%20space.fvec").unwrap();
        write_origin(tmp.path(), &url).unwrap();
        let got = read_origin(tmp.path()).unwrap();
        assert_eq!(got.url, url.as_str());
    }

    #[test]
    fn legacy_layout_name_recognizes_old_shapes() {
        // Exact shapes the old `cache_dir_for_url` emitted.
        assert!(is_legacy_layout_name("127.0.0.1:32775"));
        assert!(is_legacy_layout_name("127.0.0.1"));
        assert!(is_legacy_layout_name("localhost:8080"));
        assert!(is_legacy_layout_name("localhost"));
        assert!(is_legacy_layout_name(
            "vectordata-pvs-testing.s3.us-east-1.amazonaws.com"));
    }

    #[test]
    fn legacy_layout_name_rejects_unrelated_dirs() {
        // Reserved new-layout dirs must not be pruned.
        assert!(!is_legacy_layout_name("blobs"));
        assert!(!is_legacy_layout_name("http"));
        // Catalog datasets — single-word names without dots.
        assert!(!is_legacy_layout_name("sift1m"));
        assert!(!is_legacy_layout_name("ibm-datapile-1b"));
        // Anything with a slash isn't a top-level dir name anyway.
        assert!(!is_legacy_layout_name(""));
    }

    #[test]
    fn extract_dataset_profile_canonical_shape() {
        let (d, p) = extract_dataset_profile(
            "https://example.com/path/sift1m/profiles/default/base.fvec");
        assert_eq!(d.as_deref(), Some("sift1m"));
        assert_eq!(p.as_deref(), Some("default"));
    }

    #[test]
    fn extract_dataset_profile_partition_profile() {
        let (d, p) = extract_dataset_profile(
            "https://example.com/path/sift1m/profiles/label_03/neighbor_indices.ivecs");
        assert_eq!(d.as_deref(), Some("sift1m"));
        assert_eq!(p.as_deref(), Some("label_03"));
    }

    #[test]
    fn extract_dataset_profile_no_profiles_segment() {
        let (d, p) = extract_dataset_profile(
            "https://example.com/path/sift1m/dataset.yaml");
        assert_eq!(d.as_deref(), Some("sift1m"));
        assert!(p.is_none());
    }

    #[test]
    fn glob_match_basic_patterns() {
        assert!(glob_match("sift1m", "sift1m"));
        assert!(!glob_match("sift1m", "sift1m-2"));
        assert!(glob_match("sift*", "sift1m"));
        assert!(glob_match("*1m", "sift1m"));
        assert!(glob_match("sift?m", "sift1m"));
        assert!(!glob_match("sift?m", "sift10m"));
        assert!(glob_match("label_*", "label_03"));
        assert!(!glob_match("label_*", "default"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn prune_filter_matches_dataset_and_profile() {
        let entry = CacheEntry {
            path: PathBuf::from("/cache/blobs/aa/abc/data"),
            size_bytes: 1024,
            file_count: 1,
            origin_url: Some(
                "https://example.com/p/sift1m/profiles/default/file.fvec".into()),
            origin_host: Some("example.com".into()),
        };
        // dataset-only filter matches
        assert!(entry_matches_filter(&entry, &PruneFilter {
            dataset: Some("sift*".into()), profile: None }));
        // profile-only filter matches
        assert!(entry_matches_filter(&entry, &PruneFilter {
            dataset: None, profile: Some("default".into()) }));
        // both, both match
        assert!(entry_matches_filter(&entry, &PruneFilter {
            dataset: Some("sift1m".into()),
            profile: Some("def*".into()) }));
        // both, dataset mismatches
        assert!(!entry_matches_filter(&entry, &PruneFilter {
            dataset: Some("ibm-*".into()),
            profile: Some("default".into()) }));
        // both, profile mismatches
        assert!(!entry_matches_filter(&entry, &PruneFilter {
            dataset: Some("sift1m".into()),
            profile: Some("label_*".into()) }));
    }

    #[test]
    fn prune_filter_rejects_entries_with_no_origin() {
        let entry = CacheEntry {
            path: PathBuf::from("/cache/blobs/aa/abc/data"),
            size_bytes: 1024, file_count: 1,
            origin_url: None, origin_host: None,
        };
        assert!(!entry_matches_filter(&entry, &PruneFilter {
            dataset: Some("*".into()), profile: None }));
    }

    #[test]
    fn prune_by_filter_dry_run_lists_matches_without_removal() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Two blobs: one matches, one doesn't.
        let blob1 = root.join("blobs/aa/aabbcc/");
        let blob2 = root.join("blobs/bb/bbccdd/");
        fs::create_dir_all(&blob1).unwrap();
        fs::create_dir_all(&blob2).unwrap();
        fs::write(blob1.join("data"), b"yes").unwrap();
        fs::write(blob2.join("data"), b"no").unwrap();
        let url1 = Url::parse("https://h/p/sift1m/profiles/default/base.fvec").unwrap();
        let url2 = Url::parse("https://h/p/glove/profiles/default/base.fvec").unwrap();
        write_origin(&blob1, &url1).unwrap();
        write_origin(&blob2, &url2).unwrap();

        let report = prune_by_filter(root,
            &PruneFilter { dataset: Some("sift*".into()), profile: None },
            true).unwrap();
        assert_eq!(report.matched.len(), 1);
        assert_eq!(report.removed.len(), 0, "dry-run must not delete");
        assert!(blob1.exists() && blob2.exists());
    }

    #[test]
    fn prune_by_filter_removes_matching_blobs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let blob1 = root.join("blobs/aa/aabbcc/");
        let blob2 = root.join("blobs/bb/bbccdd/");
        fs::create_dir_all(&blob1).unwrap();
        fs::create_dir_all(&blob2).unwrap();
        fs::write(blob1.join("data"), b"yes").unwrap();
        fs::write(blob2.join("data"), b"no").unwrap();
        write_origin(&blob1, &Url::parse(
            "https://h/p/sift1m/profiles/label_00/base.fvec").unwrap()).unwrap();
        write_origin(&blob2, &Url::parse(
            "https://h/p/sift1m/profiles/default/base.fvec").unwrap()).unwrap();

        let report = prune_by_filter(root,
            &PruneFilter { dataset: Some("sift1m".into()),
                profile: Some("label_*".into()) },
            false).unwrap();
        assert_eq!(report.matched.len(), 1);
        assert_eq!(report.removed.len(), 1);
        assert!(!blob1.exists(), "matched blob must be removed");
        assert!(blob2.exists(),  "non-matched blob must be preserved");
    }

    // Unix-only: the legacy cache layout this test simulates used
    // host:port directory names (e.g. `127.0.0.1:32775`). Windows
    // reserves `:` and refuses such mkdir calls outright, so these
    // legacy directories could never have existed on Windows in the
    // first place — there's nothing for prune_legacy_layout to clean
    // up there. The Unix-side test continues to exercise the cleanup.
    #[cfg(not(windows))]
    #[test]
    fn prune_legacy_layout_removes_old_shapes_only() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for name in ["127.0.0.1:32775", "127.0.0.1", "localhost:8080",
                     "blobs", "http", "sift1m"] {
            fs::create_dir_all(root.join(name)).unwrap();
            fs::write(root.join(name).join("marker"), name).unwrap();
        }
        let mut removed = prune_legacy_layout(root).unwrap();
        removed.sort();
        assert_eq!(removed, vec![
            "127.0.0.1".to_string(),
            "127.0.0.1:32775".to_string(),
            "localhost:8080".to_string(),
        ]);
        // Reserved + catalog dirs untouched.
        assert!(root.join("blobs/marker").exists());
        assert!(root.join("http/marker").exists());
        assert!(root.join("sift1m/marker").exists());
    }
}
