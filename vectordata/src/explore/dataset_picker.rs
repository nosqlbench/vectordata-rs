// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Interactive dataset picker TUI.
//!
//! Displays a filterable, scrollable list of datasets from configured
//! catalogs. The user selects a dataset:profile with arrow keys and Enter.

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use unicode_width::UnicodeWidthStr;
use crate::dataset::CatalogEntry;

// Neon/synthwave palette — saturated but desaturated enough to stay
// readable across long sessions. Picked over the standard 8-colour
// names so the look is stable across terminal themes (the standard
// `Color::Cyan` etc. would change meaning under Solarized, Dracula,
// Nord, etc.). Truecolor-capable terminals render these verbatim;
// older terminals downsample to the nearest 256-colour cube cell.
const ACCENT_CYAN:    Color = Color::Rgb( 64, 230, 220); // primary highlight / local
const ACCENT_MINT:    Color = Color::Rgb(135, 240, 195); // success / merkle-hashed
const ACCENT_LAVENDR: Color = Color::Rgb(190, 160, 255); // info / merkle-chunked
const ACCENT_AMBER:   Color = Color::Rgb(255, 195, 120); // warning / GT-style
const ACCENT_PINK:    Color = Color::Rgb(255, 130, 200); // metadata family
const ACCENT_CORAL:   Color = Color::Rgb(255, 115, 145); // error / full-transfer
const TEXT_PRIMARY:   Color = Color::Rgb(225, 230, 245); // default row text
const TEXT_MUTED:     Color = Color::Rgb(135, 145, 175); // column headers, footer
const TEXT_DIM:       Color = Color::Rgb( 90, 100, 130); // tertiary text, "—" rows
const SELECTED_BG:    Color = Color::Rgb( 40,  55,  95); // selection band
const SELECTED_FG:    Color = Color::Rgb(245, 250, 255); // selection text

/// Pad a string to `target` DISPLAY columns (not bytes). Rust's built-in
/// `format!("{:<w$}", ...)` pads by byte count, which under-reserves
/// columns whenever the content contains multi-byte glyphs like the
/// tree-drawing unicode (`▸`, `▾`, `└` — 3 bytes each, 1 column each).
/// In the picker, that bug let the PROFILE column drift left and kiss
/// the DATASET column on any row whose name had a tree prefix.
fn pad_display(s: &str, target: usize) -> String {
    let w = UnicodeWidthStr::width(s);
    if w >= target {
        s.to_string()
    } else {
        let mut out = String::with_capacity(s.len() + (target - w));
        out.push_str(s);
        for _ in 0..(target - w) {
            out.push(' ');
        }
        out
    }
}

/// Truncate a string to at most `target` DISPLAY columns, then pad to
/// exactly `target`. Matches the behaviour the original code tried to
/// get out of `format!("{:.w$}", ...)` — that formatter also counts
/// bytes, not columns, so long UTF-8 strings would overflow the cell.
fn fit_display(s: &str, target: usize) -> String {
    let mut out = String::new();
    let mut used = 0usize;
    for ch in s.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + cw > target { break; }
        out.push(ch);
        used += cw;
    }
    for _ in used..target {
        out.push(' ');
    }
    out
}

use crate::AccessMode;
use crate::catalog::resolver::Catalog;
use crate::catalog::sources::CatalogSources;

/// A flattened row: one entry per dataset:profile pair.
struct PickerRow {
    dataset: String,
    profile: String,
    /// Numeric base_count for sorting (0 for default profile).
    base_count: u64,
    metric: String,
    facets: String,
    size: String,
    cache_status: String,
    /// Predicted access mode for this profile's `base_vectors` source.
    access: AccessMode,
}

impl PickerRow {
    fn specifier(&self) -> String {
        format!("{}:{}", self.dataset, self.profile)
    }
}

fn build_rows(
    entries: &[CatalogEntry],
    cache_survey: &std::collections::HashMap<String, (u32, u32)>,
) -> Vec<PickerRow> {
    use crate::dataset::StandardFacet;

    let cache_dir = crate::explore::cache_dir_or_exit();
    let mut rows = Vec::new();
    for entry in entries {
        let metric = entry.layout.attributes.as_ref()
            .and_then(|a| a.distance_function.as_deref())
            .unwrap_or("")
            .to_string();

        let ds_cache = cache_dir.join(&entry.name);

        // The default profile's `base_vectors` file is the dataset's
        // shared base. Non-default profiles whose base path resolves
        // to the same file (after stripping window notation) are
        // windows into that shared data — no extra download.
        let default_base_path: Option<String> = entry.layout.profiles
            .profile("default")
            .and_then(|p| p.views.get(StandardFacet::BaseVectors.key()))
            .map(|v| strip_window_suffix(&v.source.path).to_string());

        for profile_name in entry.profile_names() {
            let profile = entry.layout.profiles.profile(profile_name);
            let bc = crate::dataset::profile::profile_sort_key(
                profile_name, profile.and_then(|p| p.base_count));
            // Show count from profile's base_count (partition profiles, sized profiles)
            let explicit_count = profile.and_then(|p| p.base_count);
            let shares_default_base = profile_name != "default"
                && profile.map(|p| !p.partition).unwrap_or(false)
                && profile
                    .and_then(|p| p.views.get(StandardFacet::BaseVectors.key()))
                    .map(|v| strip_window_suffix(&v.source.path).to_string())
                    == default_base_path
                && default_base_path.is_some();
            let size = match explicit_count {
                Some(c) if c > 0 => {
                    let s = format_count(c);
                    if shares_default_base { format!("{}*", s) } else { s }
                }
                _ => String::new(),
            };

            let facets = profile.map(|p| facet_indicators(&p.views)).unwrap_or_default();

            // Per-profile cache: consult the global cache survey
            // (authoritative; covers content-addressed blob/http
            // entries the runtime actually produces) and fall back
            // to a direct on-disk file check for the legacy
            // workspace layout `derive` produces.
            let ds_workspace = ds_cache.clone();
            let (valid_chunks, total_chunks) = if let Some(p) = profile {
                profile_cache_coverage(entry, p, &ds_workspace, cache_survey)
            } else {
                (0, 0)
            };

            let cache_status = if total_chunks == 0 {
                "—".to_string()
            } else if valid_chunks == total_chunks {
                format!("100% ({}/{})", valid_chunks, total_chunks)
            } else {
                let pct = 100.0 * valid_chunks as f64 / total_chunks as f64;
                format!("{:.0}% ({}/{})", pct, valid_chunks, total_chunks)
            };

            let access = profile_access_mode(&ds_cache, profile);

            rows.push(PickerRow {
                dataset: entry.name.clone(),
                profile: profile_name.to_string(),
                base_count: bc,
                metric: metric.clone(),
                facets,
                size,
                cache_status,
                access,
            });
        }
    }
    // Pre-compute which datasets have any cached profiles
    let cached_datasets: std::collections::HashSet<String> = rows.iter()
        .filter(|r| r.cache_status != "—")
        .map(|r| r.dataset.clone())
        .collect();

    // Sort: cached datasets first, then default profile first, then cached profiles, then by name
    rows.sort_by(|a, b| {
        let a_ds_cached = cached_datasets.contains(&a.dataset);
        let b_ds_cached = cached_datasets.contains(&b.dataset);
        b_ds_cached.cmp(&a_ds_cached)
            .then_with(|| a.dataset.cmp(&b.dataset))
            .then_with(|| {
                let a_def = a.profile == "default";
                let b_def = b.profile == "default";
                b_def.cmp(&a_def)
            })
            .then_with(|| {
                let a_cached = a.cache_status != "—";
                let b_cached = b.cache_status != "—";
                b_cached.cmp(&a_cached)
            })
            .then_with(|| a.base_count.cmp(&b.base_count))
            .then_with(|| a.profile.cmp(&b.profile))
    });
    rows
}

/// Per-profile cache coverage. Consults the cache survey built at
/// picker startup (`survey_cache_state`) — that map already contains
/// authoritative `(valid_chunks, total_chunks)` for every URL the
/// runtime has touched, keyed by the canonical URL form `Storage`
/// itself uses. Per-facet logic:
///   1. The legacy `<cache_root>/<dataset>/<source>` materialised
///      layout (what `derive` produces) — file exists → fully cached.
///   2. Otherwise resolve the facet's absolute URL, normalise, look
///      up in the survey.
///   3. Otherwise count as one missing chunk (we know the facet
///      exists but have no evidence of it being cached).
fn profile_cache_coverage(
    entry: &CatalogEntry,
    profile: &crate::dataset::profile::DSProfile,
    ds_workspace: &std::path::Path,
    survey: &std::collections::HashMap<String, (u32, u32)>,
) -> (u32, u32) {
    let mut valid = 0u32;
    let mut total = 0u32;
    for (_facet, view) in &profile.views {
        let source_path = &view.source.path;
        if source_path.is_empty() { continue; }
        let clean = strip_window_suffix(source_path);
        // Skip facets the runtime's precache path also skips: anything
        // whose extension doesn't infer to a known vector element type
        // (sidecar YAMLs, layout files, etc.). Counting them as
        // "missing chunks" was the reason a freshly-precached
        // ada002-100k displayed `97% (73/75)` — two non-data facets in
        // the manifest each added 1 to `total` without ever showing up
        // in the cache.
        let ext = clean.rsplit('.').next().unwrap_or("");
        if crate::io::infer_elem_size(ext) == 0 { continue; }

        let local_path = ds_workspace.join(clean);
        if local_path.exists() {
            valid += 1;
            total += 1;
            continue;
        }
        if let Some(url) = resolve_facet_url(entry, view) {
            let canonical = canonicalize_cache_url(&url);
            if let Some(&(v, t)) = survey.get(&canonical) {
                valid += v;
                total += t;
                continue;
            }
        }
        total += 1;
    }
    (valid, total)
}

/// Compose stats lines for the Ctrl-D details overlay. Pulls
/// dataset-level attributes (model / license / vendor / distance)
/// from the catalog entry plus profile-level facets (each view's
/// source path, with window notation kept inline for clarity).
fn build_details_lines(
    row: &PickerRow,
    entry: Option<&CatalogEntry>,
) -> Vec<Line<'static>> {
    let dim_gray = Style::default().fg(TEXT_MUTED);
    let val_white = Style::default().fg(TEXT_PRIMARY);
    let head_cyan = Style::default().fg(ACCENT_CYAN);

    let mut lines: Vec<Line<'static>> = Vec::new();

    if let Some(entry) = entry
        && let Some(attrs) = entry.layout.attributes.as_ref()
    {
        if let Some(v) = attrs.model.as_deref()
            { lines.push(kv("model", v, dim_gray, val_white)); }
        if let Some(v) = attrs.distance_function.as_deref()
            { lines.push(kv("metric", v, dim_gray, val_white)); }
        if let Some(v) = attrs.vendor.as_deref()
            { lines.push(kv("vendor", v, dim_gray, val_white)); }
        if let Some(v) = attrs.license.as_deref()
            { lines.push(kv("license", v, dim_gray, val_white)); }
        if let Some(v) = attrs.url.as_deref()
            { lines.push(kv("upstream", v, dim_gray, val_white)); }
        if let Some(true) = attrs.is_normalized
            { lines.push(kv("normalized", "yes", dim_gray, val_white)); }
        if let Some(v) = attrs.notes.as_deref()
            { lines.push(kv("notes", v, dim_gray, val_white)); }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(" Profile ", head_cyan)));
    if let Some(entry) = entry
        && let Some(profile) = entry.layout.profiles.profile(&row.profile)
    {
        if let Some(maxk) = profile.maxk
            { lines.push(kv("maxk", &maxk.to_string(), dim_gray, val_white)); }
        if let Some(bc) = profile.base_count
            { lines.push(kv("base_count", &format_count(bc), dim_gray, val_white)); }
        if profile.partition
            { lines.push(kv("partition", "yes (independent base)", dim_gray, val_white)); }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Facets ", head_cyan)));
        for (facet, view) in &profile.views {
            let path = if view.window.is_some() {
                format!("{} (view window)", view.source.path)
            } else {
                view.source.path.clone()
            };
            lines.push(kv(facet, &path, dim_gray, val_white));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(" Access ", head_cyan)));
    let access_color = match row.access {
        AccessMode::Local         => ACCENT_CYAN,
        AccessMode::MerkleHashed  => ACCENT_MINT,
        AccessMode::MerkleChunked => ACCENT_LAVENDR,
        AccessMode::FullTransfer  => ACCENT_CORAL,
    };
    lines.push(Line::from(Span::styled(
        format!(" {}", row.access.description()),
        Style::default().fg(access_color),
    )));
    lines.push(kv("cache", &row.cache_status, dim_gray, val_white));

    lines
}

/// Two-column "  key   value" line used by the details overlay.
/// Returns owned `Line<'static>` so callers can push it into a vec
/// without lifetime juggling around per-line temporary strings.
fn kv(key: &str, value: &str, key_style: Style, val_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("{key:<12}"), key_style),
        Span::raw(" "),
        Span::styled(value.to_string(), val_style),
    ])
}

/// Strip a window notation (`[lo..hi)` or `(lo..hi]`) from a source
/// path, returning just the underlying file path. Source strings in
/// `dataset.yaml` use these brackets to denote subranges of a shared
/// base file (e.g. `base.fvec[0..1000000)`); the picker compares
/// stripped paths to detect when a profile is a window into the
/// dataset's default base data rather than its own download.
fn strip_window_suffix(source_path: &str) -> &str {
    if let Some(bracket) = source_path.find(|c: char| c == '[' || c == '(') {
        &source_path[..bracket]
    } else {
        source_path
    }
}

/// Walk the cache root once and build a `canonical_url → (valid_chunks,
/// total_chunks)` map. Picker consults this in [`build_rows`] to give
/// every dataset an authoritative CACHED reading without opening any
/// `Storage` (no `.mref` HEAD round-trips, no network).
///
/// The cache layout has two trees:
///   - `<cache_root>/blobs/<sha256[..2]>/<sha256>/`  — `Storage::Cached`
///     (mref-backed). Sidecar: `<filename>.mrkl` (full [`MerkleState`]).
///   - `<cache_root>/http/<sha256-of-url[..2]>/<sha256-of-url>/`  —
///     `Storage::Http` (no mref). Sidecar: `<filename>.chunks` (one
///     byte per chunk, non-zero = valid).
///
/// Each leaf carries an `origin.json` that records the URL that
/// populated it. The lookup key is the URL after the same
/// normalisation [`crate::storage`] applies on `open` (s3 → https,
/// then `Url::parse`'s canonical form), so a row's facet URL hashed
/// here matches whatever the runtime would have hashed.
fn survey_cache_state(cache_dir: &std::path::Path)
    -> std::collections::HashMap<String, (u32, u32)>
{
    let mut map: std::collections::HashMap<String, (u32, u32)> =
        std::collections::HashMap::new();
    for subdir in ["blobs", "http"] {
        scan_addressed_subtree(&cache_dir.join(subdir), &mut map);
    }
    map
}

fn scan_addressed_subtree(
    root: &std::path::Path,
    map: &mut std::collections::HashMap<String, (u32, u32)>,
) {
    let Ok(prefixes) = std::fs::read_dir(root) else { return; };
    for prefix in prefixes.flatten() {
        let pp = prefix.path();
        if !pp.is_dir() { continue; }
        let Ok(leaves) = std::fs::read_dir(&pp) else { continue; };
        for leaf in leaves.flatten() {
            let lp = leaf.path();
            if !lp.is_dir() { continue; }
            scan_cache_leaf(&lp, map);
        }
    }
}

fn scan_cache_leaf(
    leaf: &std::path::Path,
    map: &mut std::collections::HashMap<String, (u32, u32)>,
) {
    let Some(origin) = crate::cache::reader::read_origin(leaf) else { return; };
    let Ok(entries) = std::fs::read_dir(leaf) else { return; };
    for dirent in entries.flatten() {
        let path = dirent.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue; };
        match ext {
            "mrkl" => {
                if let Ok(state) = crate::merkle::MerkleState::load(&path) {
                    map.insert(
                        canonicalize_cache_url(&origin.url),
                        (state.valid_count(), state.shape().total_chunks),
                    );
                    return;
                }
            }
            "chunks" => {
                if let Ok(bytes) = std::fs::read(&path) {
                    let total = bytes.len() as u32;
                    let valid = bytes.iter().filter(|&&b| b != 0).count() as u32;
                    map.insert(canonicalize_cache_url(&origin.url), (valid, total));
                    return;
                }
            }
            _ => {}
        }
    }
}

/// Normalise a URL the same way `Storage::open` does (s3 → https, then
/// `Url::parse`'s canonical string form). The cache `origin.json` keys
/// are written by `Storage::open` against the post-translation URL, so
/// to look them up we have to translate the row's facet URL the same
/// way before hashing.
fn canonicalize_cache_url(url: &str) -> String {
    let translated = crate::transport::normalize_remote_url(url);
    match url::Url::parse(translated.as_ref()) {
        Ok(u) => u.to_string(),
        Err(_) => translated.into_owned(),
    }
}

/// Collect every absolute facet URL declared by every profile of an
/// entry. Used by both [`purge_cache_for_entry`] (to know which cache
/// leaves to delete) and the cache survey (indirectly — via
/// `resolve_facet_url` per row, but conceptually the same set).
fn entry_facet_urls(entry: &CatalogEntry) -> std::collections::HashSet<String> {
    let mut urls = std::collections::HashSet::new();
    for (_pname, profile) in &entry.layout.profiles.profiles {
        for (_facet, view) in &profile.views {
            if let Some(url) = resolve_facet_url(entry, view) {
                urls.insert(canonicalize_cache_url(&url));
            }
        }
    }
    urls
}

/// Purge every cache leaf whose `origin.json` URL belongs to this
/// dataset (any facet of any profile). Mirrors the survey's discovery
/// path — same URL canonicalisation, same walk over
/// `cache_root/{blobs,http}/<prefix>/<hash>/` — so the delete set is
/// exactly the set of leaves the survey would have attributed to this
/// dataset.
///
/// Returns `(removed_paths, freed_bytes)`. A leaf removal failure
/// (permission, missing dir) is silently skipped so a single bad
/// entry doesn't abort the whole purge.
pub(super) fn purge_cache_for_entry(
    entry: &CatalogEntry,
    cache_dir: &std::path::Path,
) -> (Vec<std::path::PathBuf>, u64) {
    let facet_urls = entry_facet_urls(entry);
    let mut removed = Vec::new();
    let mut freed: u64 = 0;
    for subdir in ["blobs", "http"] {
        let root = cache_dir.join(subdir);
        let Ok(prefixes) = std::fs::read_dir(&root) else { continue; };
        for prefix in prefixes.flatten() {
            let pp = prefix.path();
            if !pp.is_dir() { continue; }
            let Ok(leaves) = std::fs::read_dir(&pp) else { continue; };
            for leaf in leaves.flatten() {
                let lp = leaf.path();
                if !lp.is_dir() { continue; }
                let Some(origin) = crate::cache::reader::read_origin(&lp) else { continue; };
                let canonical = canonicalize_cache_url(&origin.url);
                if !facet_urls.contains(&canonical) { continue; }
                let leaf_bytes = leaf_size_bytes(&lp);
                if std::fs::remove_dir_all(&lp).is_ok() {
                    freed += leaf_bytes;
                    removed.push(lp);
                }
            }
        }
    }
    (removed, freed)
}

fn leaf_size_bytes(dir: &std::path::Path) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue; };
        for e in entries.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(p),
                Ok(_) => {
                    if let Ok(meta) = e.metadata() { total += meta.len(); }
                }
                Err(_) => {}
            }
        }
    }
    total
}

/// Resolve a facet's `view.source.path` to the absolute URL the runtime
/// would call `Storage::open` with. Two cases:
///   1. Already absolute (`http://`, `https://`, `s3://`, `file://`,
///      or a path-shaped value with a leading `/`) — use verbatim.
///      knn_entries-shape catalogs land here because the parser
///      resolves facet paths against the catalog base at load time.
///   2. Relative — join against the parent of `entry.path`. Canonical
///      catalogs land here; `entry.path` is the absolute dataset.yaml
///      URL so its parent is the dataset directory.
fn resolve_facet_url(
    entry: &CatalogEntry,
    view: &crate::dataset::DSView,
) -> Option<String> {
    let raw = strip_window_suffix(&view.source.path);
    if raw.is_empty() { return None; }
    if raw.starts_with("http://")
        || raw.starts_with("https://")
        || raw.starts_with("s3://")
        || raw.starts_with("file://")
        || raw.starts_with('/')
    {
        return Some(raw.to_string());
    }
    let parent = entry.path.rsplit_once('/').map(|(p, _)| p)?;
    Some(format!("{parent}/{raw}"))
}

/// Classify a profile by its `base_vectors` source — the file the
/// explore TUI opens first. The picker's source-of-truth column is
/// based on this facet because everything else (queries, GT) gets
/// fetched only when the user runs a KNN sample, and the dominant
/// latency-shaping question is "how does base_vectors arrive?".
///
/// Returns [`AccessMode::FullTransfer`] when the profile has no
/// `base_vectors` declared — there's nothing to predict an access
/// mode against. Returns [`AccessMode::Local`] when the source path
/// resolves to a file already present under the cache root (the
/// dataset has been precached or was locally prepared). Otherwise
/// delegates to [`AccessMode::classify`], which handles the remote
/// uniform-vs-vvec / `.mrkl`-or-not branches.
fn profile_access_mode(
    ds_cache: &std::path::Path,
    profile: Option<&crate::dataset::profile::DSProfile>,
) -> AccessMode {
    use crate::dataset::StandardFacet;

    let profile = match profile {
        Some(p) => p,
        None => return AccessMode::FullTransfer,
    };
    let view = match profile.views.get(StandardFacet::BaseVectors.key()) {
        Some(v) => v,
        None => return AccessMode::FullTransfer,
    };

    let clean = strip_window_suffix(&view.source.path);

    // Materialised locally already? Same check `profile_cache_coverage`
    // uses for its "this file exists locally, count as cached" branch.
    if ds_cache.join(clean).exists() {
        return AccessMode::Local;
    }
    AccessMode::classify_remote(clean, ds_cache)
}

/// Build a compact facet indicator string from a profile's view keys.
///
/// Letters: B(base) Q(query) G(ground-truth indices) D(distances)
///          M(metadata) P(predicates) R(results) F(filtered-knn)
fn facet_indicators(views: &indexmap::IndexMap<String, crate::dataset::DSView>) -> String {
    use crate::dataset::StandardFacet;

    let mut out = String::with_capacity(10);
    let has = |facet: StandardFacet| -> bool {
        views.contains_key(facet.key())
    };

    if has(StandardFacet::BaseVectors)                { out.push('B'); }
    if has(StandardFacet::QueryVectors)               { out.push('Q'); }
    if has(StandardFacet::NeighborIndices)            { out.push('G'); }
    if has(StandardFacet::NeighborDistances)          { out.push('D'); }
    if has(StandardFacet::MetadataContent)            { out.push('M'); }
    if has(StandardFacet::MetadataPredicates)         { out.push('P'); }
    if has(StandardFacet::MetadataResults)            { out.push('R'); }
    // F (pre-filter, the legacy filtered-knn shape) and E (post-filter,
    // G ∩ R) are distinct facets now; show both letters when both are
    // present. The legacy `filtered_*` YAML key resolves to F, so older
    // catalogs that only set that one still light up F.
    if has(StandardFacet::PrefilteredNeighborIndices)  { out.push('F'); }
    if has(StandardFacet::PostfilteredNeighborIndices) { out.push('E'); }

    out
}

/// Format a vector count for the picker's SIZE column.
///
/// Profile names use `m` (10⁶) and `mi` (2²⁰) as distinct suffixes — a
/// `1mi` profile holds 1,048,576 vectors, not 1,000,000 — so the SIZE
/// column must preserve that distinction, not collapse both onto `M`.
/// The magnitude tier is chosen by raw value (so 2.5M doesn't degrade
/// to 2500K), and within that tier the IEC variant wins iff the count
/// is an exact power-of-2 multiple; otherwise it falls back to SI with
/// one decimal.
fn format_count(n: u64) -> String {
    const KI: u64 = 1 << 10;
    const MI: u64 = 1 << 20;
    const GI: u64 = 1 << 30;
    const K: u64 = 1_000;
    const M: u64 = 1_000_000;
    const B: u64 = 1_000_000_000;

    // (tier_threshold, iec_unit, iec_label, si_unit, si_label) from
    // largest to smallest. The first tier whose threshold `n` clears
    // determines the unit family.
    let tiers: [(u64, u64, &str, u64, &str); 3] = [
        (B, GI, "Gi", B, "B"),
        (M, MI, "Mi", M, "M"),
        (K, KI, "Ki", K, "K"),
    ];
    for (threshold, iec, iec_label, si, si_label) in tiers {
        if n >= threshold {
            if n % iec == 0 {
                return format!("{}{}", n / iec, iec_label);
            } else if n % si == 0 {
                return format!("{}{}", n / si, si_label);
            } else {
                return format!("{:.1}{}", n as f64 / si as f64, si_label);
            }
        }
    }
    n.to_string()
}

fn matches_filter(row: &PickerRow, filter: &str) -> bool {
    if filter.is_empty() { return true; }

    // If filter is all uppercase letters that are valid facet codes, match by facets.
    // Valid codes: B Q G D M P R F
    let is_facet_filter = !filter.is_empty()
        && filter.chars().all(|c| "BQGDMPRF".contains(c));

    if is_facet_filter {
        // Every code in the filter must be present in the row's facets
        filter.chars().all(|c| row.facets.contains(c))
    } else {
        let lower = filter.to_lowercase();
        row.dataset.to_lowercase().contains(&lower)
            || row.profile.to_lowercase().contains(&lower)
            || row.metric.to_lowercase().contains(&lower)
    }
}

/// Action the user selected from the per-row action menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerAction {
    /// Launch the unified vector-space explorer for this profile.
    Visualize,
    /// Download every facet's bytes into the local cache directory.
    Precache,
    /// Delete the dataset's cached files from disk.
    Purge,
    /// Probe the dataset's catalog: verify reachability, list facets,
    /// HTTP-RANGE the first record of each one.
    Ping,
}

impl PickerAction {
    fn label(self) -> &'static str {
        match self {
            PickerAction::Visualize => "Visualize",
            PickerAction::Precache  => "Precache",
            PickerAction::Purge     => "Purge",
            PickerAction::Ping      => "Ping",
        }
    }

    fn description(self) -> &'static str {
        match self {
            PickerAction::Visualize =>
                "Open the interactive explorer: norms, distances, eigenvalues, PCA.",
            PickerAction::Precache =>
                "Download every facet of this profile into the cache directory.",
            PickerAction::Purge =>
                "Delete this dataset's cached files from disk (does not affect the catalog).",
            PickerAction::Ping =>
                "Verify catalog reachability: list facets and HTTP-range each one's first record.",
        }
    }

    /// Menu order — visualize first (the dominant action) so a
    /// double-Enter from the picker still lands on the explorer.
    fn menu_order() -> &'static [PickerAction] {
        &[
            PickerAction::Visualize,
            PickerAction::Precache,
            PickerAction::Ping,
            PickerAction::Purge,
        ]
    }
}

/// Whether the picker should continue running after an action
/// completes, or close down because the action signalled a hard exit
/// (e.g. user pressed `q` inside the visualizer).
pub enum ActionFlow {
    /// Re-enter the picker with all UI state preserved (cursor,
    /// expanded set, filter text, scroll position, last menu cursor).
    Stay,
    /// Close the picker too — caller is done.
    Exit,
}

/// Outcome of running the catalog dataset picker.
pub enum PickerOutcome {
    /// Picker closed cleanly. Reached either by the user quitting the
    /// picker (Esc / q / Ctrl-C) or by an action returning
    /// [`ActionFlow::Exit`].
    Done,
    /// Picker could not start (no catalogs, no datasets, no TTY,
    /// terminal-init failure). An error message has already been
    /// printed to stderr.
    Failed,
}

/// Run the dataset picker. The caller provides a `dispatch` closure
/// invoked when the user confirms an action from the per-row menu.
///
/// The picker suspends its alt-screen + raw mode before each
/// dispatch call so the action can print freely to stderr / take over
/// the terminal (the visualizer enters its own raw mode), and
/// restores its own chrome immediately after. Picker state (cursor,
/// expanded set, filter, scroll, menu cursor) is preserved across
/// dispatches because the function never exits between them.
pub fn run_picker<F>(mut dispatch: F) -> PickerOutcome
where F: FnMut(&str, PickerAction) -> ActionFlow,
{
    // Load catalog
    let sources = CatalogSources::new().configure_default();
    if sources.is_empty() {
        eprintln!("error: no catalog sources configured");
        eprintln!();
        eprintln!("Add one with `vectordata config add-catalog <URL-or-path>`.");
        return PickerOutcome::Failed;
    }
    let catalog = Catalog::of(&sources);
    let entries = catalog.datasets();
    if entries.is_empty() {
        eprintln!("error: no datasets found in any configured catalog");
        return PickerOutcome::Failed;
    }

    let cache_dir = crate::explore::cache_dir_or_exit();
    let mut cache_survey = survey_cache_state(&cache_dir);
    let mut all_rows = build_rows(entries, &cache_survey);
    if all_rows.is_empty() {
        eprintln!("error: no dataset profiles available in the configured catalogs");
        return PickerOutcome::Failed;
    }

    // Need a real terminal — refuse with a clear message when we're
    // piped or redirected. Otherwise the user gets a silent exit.
    if !std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        eprintln!("error: dataset picker requires an interactive terminal");
        eprintln!("       (stdout is not a TTY — were you redirecting output?)");
        eprintln!();
        eprintln!("Pass `--dataset <name>` or `--source <path>` to explore non-interactively,");
        eprintln!("or run `vectordata datasets list` for a text-friendly catalog view.");
        return PickerOutcome::Failed;
    }

    if enable_raw_mode().is_err() {
        eprintln!("error: failed to enable terminal raw mode");
        return PickerOutcome::Failed;
    }
    let mut stdout = stdout();
    if execute!(stdout, EnterAlternateScreen).is_err() {
        let _ = crossterm::terminal::disable_raw_mode();
        eprintln!("error: failed to enter alternate screen");
        return PickerOutcome::Failed;
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            let _ = crossterm::terminal::disable_raw_mode();
            let _ = execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
            eprintln!("error: terminal initialisation failed: {e}");
            return PickerOutcome::Failed;
        }
    };

    let mut filter = String::new();
    let mut cursor: usize = 0;
    let mut scroll: usize = 0;
    let mut show_all_profiles = false;
    let mut expanded: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut show_help = false;
    let mut show_details = false;
    let mut menu_open = false;
    let mut menu_cursor: usize = 0;
    let result: PickerOutcome;

    loop {
        // Filter rows:
        // - show_all_profiles: show everything
        // - otherwise: show "default" profile + all profiles of expanded datasets
        let visible: Vec<&PickerRow> = all_rows.iter()
            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
            .filter(|r| matches_filter(r, &filter))
            .collect();

        if cursor >= visible.len() && !visible.is_empty() {
            cursor = visible.len() - 1;
        }

        let drew = terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // filter input
                    Constraint::Min(5),    // list
                    Constraint::Length(3),  // detail panel
                    Constraint::Length(1),  // footer
                ]).split(frame.area());

            // Filter input — neon "Filter:" label, cyan caret, lavender
            // chrome on the box so the picker reads as a single tinted
            // panel instead of plain ASCII.
            let filter_line = Line::from(vec![
                Span::styled(" Filter: ", Style::default().fg(TEXT_MUTED)),
                Span::styled(filter.clone(), Style::default().fg(TEXT_PRIMARY)),
                Span::styled("█", Style::default().fg(ACCENT_CYAN)),
            ]);
            frame.render_widget(
                Paragraph::new(filter_line)
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(Style::default().fg(ACCENT_LAVENDR))
                        .title(Span::styled(
                            format!(" Select Dataset — {} shown ({}) ",
                                visible.len(),
                                if show_all_profiles { "all profiles" } else { "default only — Tab for all" }),
                            Style::default().fg(ACCENT_CYAN),
                        ))),
                chunks[0],
            );

            // Dataset list
            let list_height = chunks[1].height.saturating_sub(2) as usize;
            if cursor >= scroll + list_height && list_height > 0 {
                scroll = cursor - list_height + 1;
            }
            if cursor < scroll {
                scroll = cursor;
            }

            if show_help {
                let help = vec![
                    Line::from(Span::styled(" Dataset Picker — Keyboard Shortcuts", Style::default().fg(ACCENT_CYAN))),
                    Line::from(""),
                    Line::from(" Navigation"),
                    Line::from("   ↑ / ↓               Move selection"),
                    Line::from("   PgUp / PgDn          Jump by page"),
                    Line::from("   Enter / →            Expand dataset / select profile"),
                    Line::from("   ←                    Collapse expanded dataset"),
                    Line::from("   Tab                  Toggle: default profiles / all profiles"),
                    Line::from(""),
                    Line::from(" Filtering"),
                    Line::from("   type letters          Filter by name, profile, or metric"),
                    Line::from("   type BQGDMPRF         Filter by facet codes (all caps)"),
                    Line::from("   Backspace             Delete last filter character"),
                    Line::from(""),
                    Line::from(" Facet Codes"),
                    Line::from("   B=Base  Q=Query  G=GroundTruth  D=Distances"),
                    Line::from("   M=Metadata  P=Predicates  R=Results  F=Filtered"),
                    Line::from(""),
                    Line::from(" Inspection"),
                    Line::from("   Ctrl-D                Toggle dataset details overlay"),
                    Line::from(""),
                    Line::from(" Exit"),
                    Line::from("   Esc                   Close details → collapse → clear filter → exit (one step per press)"),
                    Line::from("   q / Ctrl-C            Quit immediately"),
                    Line::from("   ?                     Toggle this help"),
                ];
                frame.render_widget(
                    Paragraph::new(help).block(Block::default().borders(Borders::ALL)
                        .border_style(Style::default().fg(ACCENT_LAVENDR))
                        .title(Span::styled(" Help ", Style::default().fg(ACCENT_CYAN)))),
                    chunks[1],
                );
            } else {

            // Column widths measured in DISPLAY columns. The tree prefix
            // (" ▸ ", " ▾ ", "  └ ", "   ") is up to 4 display cols wide,
            // so reserve 4 on top of the longest dataset name, plus 2
            // trailing blanks so the PROFILE column has a visible gap
            // even when the widest row fills its cell edge-to-edge.
            let name_w = visible.iter()
                .map(|r| UnicodeWidthStr::width(r.dataset.as_str()) + 4 + 2)
                .max()
                .unwrap_or(20)
                .max(12);
            // PROFILE column sized to fit the longest profile name plus
            // a trailing gap. Default of 14 is plenty for `default`,
            // `1m`, `40mi` etc., but partition / user-named profiles
            // can run longer.
            let prof_w = visible.iter()
                .map(|r| UnicodeWidthStr::width(r.profile.as_str()) + 2)
                .max()
                .unwrap_or(14)
                .max(10);
            let facet_w = 12;
            let metric_w = 14;
            let size_w = 11; // +1 vs metric/cache widths to leave room for the shared-base "*"
            let access_w = 10;
            let cache_w = 18;

            // Header and data rows share the same cell widths so columns
            // line up edge-to-edge. Data rows get their leading whitespace
            // from the tree prefix (3–4 display columns depending on
            // shape); the header pads DATASET to the same `name_w` so
            // the PROFILE column lands at identical offsets in both.
            let mut lines = vec![Line::from(vec![
                Span::styled(pad_display("DATASET", name_w), Style::default().fg(TEXT_MUTED)),
                Span::styled(pad_display("PROFILE", prof_w), Style::default().fg(TEXT_MUTED)),
                Span::styled(pad_display("FACETS", facet_w), Style::default().fg(TEXT_MUTED)),
                Span::styled(pad_display("METRIC", metric_w), Style::default().fg(TEXT_MUTED)),
                Span::styled(pad_display("SIZE", size_w), Style::default().fg(TEXT_MUTED)),
                Span::styled(pad_display("ACCESS", access_w), Style::default().fg(TEXT_MUTED)),
                Span::styled(pad_display("CACHED", cache_w), Style::default().fg(TEXT_MUTED)),
            ])];

            for (i, row) in visible.iter().enumerate().skip(scroll).take(list_height.saturating_sub(1)) {
                let selected = i == cursor;
                let (fg, bg) = if selected {
                    (SELECTED_FG, SELECTED_BG)
                } else if row.cache_status != "—" {
                    (ACCENT_MINT, Color::Reset)
                } else {
                    (TEXT_PRIMARY, Color::Reset)
                };
                let style = Style::default().fg(fg).bg(bg);
                let cache_style = if selected {
                    style
                } else if row.cache_status != "—" {
                    Style::default().fg(ACCENT_MINT).bg(bg)
                } else {
                    Style::default().fg(TEXT_DIM).bg(bg)
                };

                // Tree indicator for collapsed/expanded datasets
                let is_expanded = expanded.contains(&row.dataset);
                let has_siblings = all_rows.iter().filter(|r| r.dataset == row.dataset).count() > 1;
                let is_child = is_expanded && row.profile != "default";

                let (prefix, name_display) = if show_all_profiles {
                    (" ", row.dataset.as_str())
                } else if is_child {
                    ("  └ ", row.dataset.as_str())
                } else if has_siblings && !is_expanded {
                    (" ▸ ", row.dataset.as_str())
                } else if has_siblings && is_expanded {
                    (" ▾ ", row.dataset.as_str())
                } else {
                    ("   ", row.dataset.as_str())
                };

                let name_field = format!("{}{}", prefix, name_display);
                let name_cell = fit_display(&name_field, name_w);

                let access_color = match row.access {
                    AccessMode::Local         => ACCENT_CYAN,
                    AccessMode::MerkleHashed  => ACCENT_MINT,
                    AccessMode::MerkleChunked => ACCENT_LAVENDR,
                    AccessMode::FullTransfer  => ACCENT_CORAL,
                };
                let access_style = if selected {
                    style
                } else {
                    Style::default().fg(access_color).bg(bg)
                };

                lines.push(Line::from(vec![
                    Span::styled(name_cell, style),
                    Span::styled(pad_display(&row.profile, prof_w), style),
                    Span::styled(pad_display(&row.facets, facet_w), style),
                    Span::styled(pad_display(&row.metric, metric_w), style),
                    Span::styled(pad_display(&row.size, size_w), style),
                    Span::styled(pad_display(row.access.short_label(), access_w), access_style),
                    Span::styled(pad_display(&row.cache_status, cache_w), cache_style),
                ]));
            }

            frame.render_widget(
                Paragraph::new(lines)
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(Style::default().fg(ACCENT_LAVENDR))),
                chunks[1],
            );

            } // close help/list else

            // Detail panel for selected row. The dataset:profile name
            // is already shown highlighted in the list above, so the
            // detail line skips it and just expands the facet legend +
            // access mode.
            let (detail_line, access_line) = if let Some(row) = visible.get(cursor) {
                let mut spans: Vec<Span<'_>> = vec![Span::raw(" ")];
                let facet_details: &[(&str, char, Color)] = &[
                    ("Base",     'B', ACCENT_MINT),
                    ("Query",    'Q', ACCENT_MINT),
                    ("GT",       'G', ACCENT_AMBER),
                    ("Dist",     'D', ACCENT_AMBER),
                    ("Meta",     'M', ACCENT_PINK),
                    ("Pred",     'P', ACCENT_PINK),
                    ("Results",  'R', ACCENT_PINK),
                    ("Filtered", 'F', ACCENT_CORAL),
                ];
                for (label, code, color) in facet_details {
                    if row.facets.contains(*code) {
                        spans.push(Span::styled(format!("{} ", label), Style::default().fg(*color)));
                    } else {
                        spans.push(Span::styled(format!("{} ", label), Style::default().fg(TEXT_DIM)));
                    }
                }
                let access_color = match row.access {
                    AccessMode::Local         => ACCENT_CYAN,
                    AccessMode::MerkleHashed  => ACCENT_MINT,
                    AccessMode::MerkleChunked => ACCENT_LAVENDR,
                    AccessMode::FullTransfer  => ACCENT_CORAL,
                };
                let access = Line::from(Span::styled(
                    format!(" access: {}", row.access.description()),
                    Style::default().fg(access_color),
                ));
                (Line::from(spans), access)
            } else {
                (Line::from(""), Line::from(""))
            };
            frame.render_widget(
                Paragraph::new(vec![
                    detail_line,
                    access_line,
                ]).block(Block::default().borders(Borders::TOP)
                    .border_style(Style::default().fg(ACCENT_LAVENDR))),
                chunks[2],
            );

            // Footer
            frame.render_widget(
                Paragraph::new(Span::styled(
                    " ↑↓: navigate | Enter/→: expand | ←: collapse | Tab: all profiles | type to filter | Ctrl-D: details | Esc: back | q: quit",
                    Style::default().fg(TEXT_MUTED))),
                chunks[3],
            );

            // Stats overlay (Ctrl-D toggle). Renders on top of the
            // list area, anchored to the bottom of the screen and
            // covering roughly the lower 60% horizontally centred,
            // so the filter line and footer stay visible.
            if show_details {
                if let Some(row) = visible.get(cursor) {
                    let outer = frame.area();
                    let popup_h = outer.height.saturating_sub(6).max(6).min(20);
                    let popup_w = outer.width.saturating_sub(8).max(40);
                    let popup_x = (outer.width.saturating_sub(popup_w)) / 2;
                    let popup_y = outer.height.saturating_sub(popup_h + 1);
                    let area = ratatui::layout::Rect {
                        x: outer.x + popup_x,
                        y: outer.y + popup_y,
                        width: popup_w,
                        height: popup_h,
                    };
                    let entry = entries.iter().find(|e| e.name == row.dataset);
                    let lines = build_details_lines(row, entry);
                    frame.render_widget(ratatui::widgets::Clear, area);
                    frame.render_widget(
                        Paragraph::new(lines)
                            .block(Block::default().borders(Borders::ALL)
                                .border_style(Style::default().fg(ACCENT_LAVENDR))
                                .title(Span::styled(
                                    format!(" Dataset Details — {}:{} (Ctrl-D / Esc to close) ", row.dataset, row.profile),
                                    Style::default().fg(ACCENT_CYAN),
                                ))),
                        area,
                    );
                }
            }

            // Action menu (opens on Enter against a leaf row). Drawn
            // last so it lands on top of any other overlay.
            if menu_open {
                if let Some(row) = visible.get(cursor) {
                    let outer = frame.area();
                    let menu_h: u16 = (PickerAction::menu_order().len() as u16) + 5;
                    let menu_w: u16 = 60;
                    let area = ratatui::layout::Rect {
                        x: outer.x + outer.width.saturating_sub(menu_w) / 2,
                        y: outer.y + outer.height.saturating_sub(menu_h) / 2,
                        width: menu_w.min(outer.width),
                        height: menu_h.min(outer.height),
                    };
                    let actions = PickerAction::menu_order();
                    let mut lines: Vec<Line<'static>> = Vec::new();
                    for (i, action) in actions.iter().enumerate() {
                        let marker = if i == menu_cursor { " ▸ " } else { "   " };
                        let style = if i == menu_cursor {
                            Style::default().fg(SELECTED_FG).bg(SELECTED_BG)
                        } else {
                            Style::default().fg(TEXT_PRIMARY)
                        };
                        lines.push(Line::from(Span::styled(
                            format!("{marker}{}", action.label()),
                            style,
                        )));
                    }
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        format!(" {}", actions[menu_cursor].description()),
                        Style::default().fg(TEXT_MUTED),
                    )));
                    frame.render_widget(ratatui::widgets::Clear, area);
                    frame.render_widget(
                        Paragraph::new(lines)
                            .block(Block::default().borders(Borders::ALL)
                                .border_style(Style::default().fg(ACCENT_CYAN))
                                .title(Span::styled(
                                    format!(" Action — {}:{} (Enter to confirm · Esc to cancel) ", row.dataset, row.profile),
                                    Style::default().fg(ACCENT_CYAN),
                                ))),
                        area,
                    );
                }
            }
        });
        if let Err(e) = drew {
            let _ = disable_raw_mode();
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
            eprintln!("error: terminal draw failed: {e}");
            return PickerOutcome::Failed;
        }

        // Events
        if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    result = PickerOutcome::Done;
                    break;
                }
                if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    show_details = !show_details;
                    continue;
                }

                // Action menu is modal — when open, swallow everything
                // but the menu's own navigation/select/cancel keys so
                // the underlying list doesn't drift.
                if menu_open {
                    match key.code {
                        KeyCode::Esc => { menu_open = false; }
                        KeyCode::Up => {
                            if menu_cursor > 0 { menu_cursor -= 1; }
                        }
                        KeyCode::Down => {
                            if menu_cursor + 1 < PickerAction::menu_order().len() {
                                menu_cursor += 1;
                            }
                        }
                        KeyCode::Enter => {
                            let vis: Vec<&PickerRow> = all_rows.iter()
                                .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                                .filter(|r| matches_filter(r, &filter))
                                .collect();
                            if let Some(row) = vis.get(cursor) {
                                let action = PickerAction::menu_order()[menu_cursor];
                                let specifier = row.specifier();
                                menu_open = false;
                                // Suspend our chrome so the action can
                                // own the terminal (visualizer enters
                                // its own raw mode; precache/ping/purge
                                // print to stderr). State across the
                                // suspension is naturally preserved —
                                // we never exit run_picker.
                                let _ = disable_raw_mode();
                                let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
                                let flow = dispatch(&specifier, action);
                                let _ = execute!(std::io::stdout(), EnterAlternateScreen);
                                let _ = enable_raw_mode();
                                let _ = terminal.clear();
                                if matches!(flow, ActionFlow::Exit) {
                                    result = PickerOutcome::Done;
                                    break;
                                }
                                // Re-survey the cache after the action.
                                // precache fills it, purge empties it,
                                // visualize's sparse access streams
                                // chunks into it. Re-walking the cache
                                // tree is cheap (filesystem only — one
                                // pass over each leaf's origin.json +
                                // sidecar, no network, no Storage
                                // opens) and gives a complete view —
                                // every dataset, not just the one the
                                // user touched. Cursor restored by
                                // dataset:profile so the row the user
                                // just acted on stays under the
                                // highlight even if the new sort order
                                // moved it.
                                cache_survey = survey_cache_state(&cache_dir);
                                all_rows = build_rows(entries, &cache_survey);
                                let new_visible_idx = all_rows.iter()
                                    .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                                    .filter(|r| matches_filter(r, &filter))
                                    .position(|r| r.specifier() == specifier);
                                if let Some(pos) = new_visible_idx {
                                    cursor = pos;
                                } else {
                                    cursor = 0;
                                    scroll = 0;
                                }
                            } else {
                                menu_open = false;
                            }
                        }
                        _ => {}
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') => {
                        result = PickerOutcome::Done;
                        break;
                    }
                    KeyCode::Esc => {
                        // Single-Esc cascade: whichever piece of state
                        // is "in the way" of leaving the picker gets
                        // peeled off first. No double-tap; the user can
                        // just keep tapping Esc until they're out.
                        if show_details {
                            show_details = false;
                        } else if !expanded.is_empty() && !show_all_profiles {
                            expanded.clear();
                        } else if !filter.is_empty() {
                            filter.clear();
                            cursor = 0;
                            scroll = 0;
                        } else {
                            result = PickerOutcome::Done;
                            break;
                        }
                    }
                    KeyCode::Enter => {
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .collect();
                        if let Some(row) = vis.get(cursor) {
                            // If in collapsed mode and this dataset has multiple profiles, expand it
                            let profile_count = all_rows.iter()
                                .filter(|r| r.dataset == row.dataset)
                                .count();
                            if !show_all_profiles && !expanded.contains(&row.dataset) && profile_count > 1 {
                                expanded.insert(row.dataset.clone());
                            } else {
                                // Leaf row — open the action menu rather
                                // than launching the explorer directly.
                                menu_open = true;
                                menu_cursor = 0;
                            }
                        }
                    }
                    KeyCode::Left => {
                        // Collapse: if current row's dataset is expanded, collapse it
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .collect();
                        if let Some(row) = vis.get(cursor) {
                            if expanded.remove(&row.dataset) {
                                // Move cursor to the default row of this dataset
                                let new_vis: Vec<&PickerRow> = all_rows.iter()
                                    .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                                    .filter(|r| matches_filter(r, &filter))
                                    .collect();
                                cursor = new_vis.iter().position(|r| r.dataset == row.dataset).unwrap_or(0);
                            }
                        }
                    }
                    KeyCode::Right => {
                        // Expand: same as Enter for collapsed datasets
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .collect();
                        if let Some(row) = vis.get(cursor) {
                            if !expanded.contains(&row.dataset) {
                                expanded.insert(row.dataset.clone());
                            }
                        }
                    }
                    KeyCode::Up => {
                        cursor = cursor.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        let visible_count = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .count();
                        if cursor + 1 < visible_count {
                            cursor += 1;
                        }
                    }
                    KeyCode::PageUp => {
                        let h = crossterm::terminal::size().map(|(_, h)| h as usize).unwrap_or(20);
                        cursor = cursor.saturating_sub(h.saturating_sub(6));
                    }
                    KeyCode::PageDown => {
                        let h = crossterm::terminal::size().map(|(_, h)| h as usize).unwrap_or(20);
                        let visible_count = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .count();
                        cursor = (cursor + h.saturating_sub(6)).min(visible_count.saturating_sub(1));
                    }
                    KeyCode::Tab => {
                        show_all_profiles = !show_all_profiles;
                        if !show_all_profiles {
                            expanded.clear();
                        }
                        cursor = 0;
                        scroll = 0;
                    }
                    KeyCode::Backspace => {
                        filter.pop();
                        cursor = 0;
                        scroll = 0;
                    }
                    KeyCode::Char('?') => {
                        show_help = !show_help;
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        show_help = false;
                        filter.push(c);
                        cursor = 0;
                        scroll = 0;
                    }
                    _ => {}
                }
            }
        }
    }

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    result
}

#[cfg(test)]
mod tests {
    use super::format_count;

    #[test]
    fn format_count_iec_vs_si() {
        // Decimal-mega counts render as "M".
        assert_eq!(format_count(1_000_000), "1M");
        assert_eq!(format_count(2_000_000), "2M");
        assert_eq!(format_count(13_000_000), "13M");
        // Mebi (2^20) counts render as "Mi" — these are what `1mi`,
        // `2mi`, `8mi`, `16mi` profiles produce, and were previously
        // mis-shown as 1.0M / 2.1M / 8.4M / 16.8M.
        assert_eq!(format_count(1 << 20), "1Mi");
        assert_eq!(format_count(2 << 20), "2Mi");
        assert_eq!(format_count(4 << 20), "4Mi");
        assert_eq!(format_count(8 << 20), "8Mi");
        assert_eq!(format_count(16 << 20), "16Mi");
        // Gibi vs giga.
        assert_eq!(format_count(1_000_000_000), "1B");
        assert_eq!(format_count(1 << 30), "1Gi");
        // Off-grid values fall back to one-decimal SI.
        assert_eq!(format_count(2_500_000), "2.5M");
        // Counts below the K threshold render verbatim.
        assert_eq!(format_count(42), "42");
    }
}
