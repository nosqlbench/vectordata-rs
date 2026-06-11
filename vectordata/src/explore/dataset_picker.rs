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
    style::{Color, Modifier, Style},
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
// Selection uses `Modifier::REVERSED` rather than explicit fg/bg
// colours so the highlight bar is independent of palette state —
// works under the neon theme, the mono toggle, and broken-ANSI
// terminals alike.

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
    cache_survey: &std::collections::HashMap<String, FacetCacheView>,
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
///
/// For *windowed* views (sized profiles where `view.source.window`
/// carries a `[start..end)` range), the coverage is bounded to the
/// chunks covering the window's byte range — otherwise every sized
/// profile sharing a base file would report the whole file's chunk
/// count and the picker's CACHED column would show identical
/// numerators / denominators across every row.
fn profile_cache_coverage(
    entry: &CatalogEntry,
    profile: &crate::dataset::profile::DSProfile,
    ds_workspace: &std::path::Path,
    survey: &std::collections::HashMap<String, FacetCacheView>,
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
        // emb-002-100k displayed `97% (73/75)` — two non-data facets in
        // the manifest each added 1 to `total` without ever showing up
        // in the cache.
        let ext = clean.rsplit('.').next().unwrap_or("");
        let elem_size = crate::io::infer_elem_size(ext);
        if elem_size == 0 { continue; }

        // Prefer the survey (which reads the `.mrkl` / `.chunks`
        // sidecar for true chunk-level coverage). Post-cutover the
        // natural-layout cache path is *exactly* `<cache>/<dataset>/
        // <facet>` — the same place a `derive`-style flat file
        // would live, but Storage pre-allocates the file as sparse
        // before downloading any chunks. A `file.exists()` check
        // therefore lights up as `100% (1/1)` from the moment
        // precache opens the facet, masking the real chunk progress.
        // Survey first, file-existence second (as the `derive`-style
        // flat-file fallback when there is no sidecar to consult).
        if let Some(url) = resolve_facet_url(entry, view) {
            let canonical = canonicalize_cache_url(&url);
            if let Some(cv) = survey.get(&canonical) {
                let (v, t) = window_bounded_coverage(cv, view, source_path, ext, elem_size);
                valid += v;
                total += t;
                continue;
            }
        }
        let local_path = ds_workspace.join(clean);
        if local_path.exists() {
            valid += 1;
            total += 1;
            continue;
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
    colors_on: bool,
) -> Vec<Line<'static>> {
    let tint = |c: Color| if colors_on { Style::default().fg(c) } else { Style::default() };
    let dim_gray = tint(TEXT_MUTED);
    let val_white = tint(TEXT_PRIMARY);
    let head_cyan = tint(ACCENT_CYAN);

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
        tint(access_color),
    )));
    lines.push(kv("cache", &row.cache_status, dim_gray, val_white));

    lines
}

/// Compose the full descriptor for the scrollable Describe overlay.
/// Goes deeper than [`build_details_lines`]: every view's path,
/// namespace (when set), and explicit window range; the origin URL
/// pulled from the catalog entry; access mode rationale; per-row
/// cache state.
fn build_descriptor_lines(
    row: &PickerRow,
    entry: Option<&CatalogEntry>,
    colors_on: bool,
) -> Vec<Line<'static>> {
    let tint = |c: Color| if colors_on { Style::default().fg(c) } else { Style::default() };
    let dim_gray = tint(TEXT_MUTED);
    let val_white = tint(TEXT_PRIMARY);
    let head_cyan = tint(ACCENT_CYAN);

    let mut lines: Vec<Line<'static>> = Vec::new();

    lines.push(Line::from(vec![
        Span::styled(" Dataset: ", dim_gray),
        Span::styled(row.dataset.clone(), val_white),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" Profile: ", dim_gray),
        Span::styled(row.profile.clone(), val_white),
    ]));

    if let Some(entry) = entry {
        lines.push(kv("path", &entry.path, dim_gray, val_white));
        lines.push(kv("type", &entry.dataset_type, dim_gray, val_white));
    }

    if let Some(entry) = entry
        && let Some(attrs) = entry.layout.attributes.as_ref()
    {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Attributes ", head_cyan)));
        if let Some(v) = attrs.model.as_deref()
            { lines.push(kv("model", v, dim_gray, val_white)); }
        if let Some(v) = attrs.distance_function.as_deref()
            { lines.push(kv("distance", v, dim_gray, val_white)); }
        if let Some(v) = attrs.vendor.as_deref()
            { lines.push(kv("vendor", v, dim_gray, val_white)); }
        if let Some(v) = attrs.license.as_deref()
            { lines.push(kv("license", v, dim_gray, val_white)); }
        if let Some(v) = attrs.url.as_deref()
            { lines.push(kv("upstream", v, dim_gray, val_white)); }
        if let Some(b) = attrs.is_normalized
            { lines.push(kv("normalized", if b { "yes" } else { "no" }, dim_gray, val_white)); }
        if let Some(v) = attrs.notes.as_deref()
            { lines.push(kv("notes", v, dim_gray, val_white)); }
    }

    if let Some(entry) = entry
        && let Some(profile) = entry.layout.profiles.profile(&row.profile)
    {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Profile ", head_cyan)));
        if let Some(maxk) = profile.maxk
            { lines.push(kv("maxk", &maxk.to_string(), dim_gray, val_white)); }
        if let Some(bc) = profile.base_count
            { lines.push(kv("base_count", &format_count(bc), dim_gray, val_white)); }
        if profile.partition
            { lines.push(kv("partition", "yes (independent base)", dim_gray, val_white)); }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Facets ", head_cyan)));
        for (facet, view) in &profile.views {
            // Header line: facet name in primary tint, then nested
            // fields underneath for source path / namespace / window.
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{facet}:"), val_white),
            ]));
            lines.push(kv_indent(2, "source", &view.source.path, dim_gray, val_white));
            if let Some(ns) = view.source.namespace.as_deref() {
                lines.push(kv_indent(2, "namespace", ns, dim_gray, val_white));
            }
            // Both window fields the catalog YAML can populate. The
            // view-level override (sibling of `source`) shadows the
            // source-level one; show both if set so users can see
            // which one wins.
            if let Some(view_w) = view.window.as_ref()
                && let Some(iv) = view_w.0.first()
            {
                lines.push(kv_indent(2, "window",
                    &format!("[{}..{}) (view override)", iv.min_incl, iv.max_excl),
                    dim_gray, val_white));
            }
            if let Some(iv) = view.source.window.0.first() {
                lines.push(kv_indent(2, "source.window",
                    &format!("[{}..{})", iv.min_incl, iv.max_excl),
                    dim_gray, val_white));
            }
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(" Access & Cache ", head_cyan)));
    let access_color = match row.access {
        AccessMode::Local         => ACCENT_CYAN,
        AccessMode::MerkleHashed  => ACCENT_MINT,
        AccessMode::MerkleChunked => ACCENT_LAVENDR,
        AccessMode::FullTransfer  => ACCENT_CORAL,
    };
    lines.push(kv("access", row.access.short_label(), dim_gray, tint(access_color)));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("{:<12}", ""), dim_gray),
        Span::raw(" "),
        Span::styled(row.access.description().to_string(), dim_gray),
    ]));
    lines.push(kv("cache", &row.cache_status, dim_gray, val_white));
    lines.push(kv("size", &row.size, dim_gray, val_white));
    lines.push(kv("metric", &row.metric, dim_gray, val_white));

    lines
}

/// Locate and load the raw catalog YAML backing a dataset, return
/// `(title_suffix, lines)`. For canonical `dataset.yaml` catalogs
/// the file is returned verbatim; for `knn_entries.yaml` catalogs
/// only the entries matching `entry.name` (plus `_defaults:`) are
/// kept so a 1268-line / 253-entry file collapses to the ~100 lines
/// the user actually cares about.
///
/// Both local paths and HTTP(S) URLs are handled. On failure the
/// returned line vector carries a single error message — the overlay
/// still opens so the user can see what went wrong.
fn build_source_lines(
    entry: &CatalogEntry,
    colors_on: bool,
) -> (String, Vec<Line<'static>>) {
    let tint = |c: Color| if colors_on { Style::default().fg(c) } else { Style::default() };
    let body_style = tint(TEXT_PRIMARY);
    let error_style = tint(ACCENT_CORAL);

    // Source location. `entry.catalog_file` is the document the
    // resolver actually parsed this entry from — authoritative when
    // present. For knn_entries-shape catalogs it is the ONLY correct
    // answer: `entry.path` holds `_defaults.base_url` (where the DATA
    // lives, possibly another host and scheme), so the old
    // `<path>/knn_entries.yaml` reconstruction pointed at a file that
    // never existed. Canonical entries fall back to `entry.path`,
    // which IS their dataset.yaml location.
    let (source_url, title_suffix) = match &entry.catalog_file {
        Some(file) => {
            let basename = file.rsplit('/').next().unwrap_or("catalog").to_string();
            (file.clone(), basename)
        }
        None if entry.dataset_type == "knn_entries.yaml" => (
            format!("{}/knn_entries.yaml", entry.path.trim_end_matches('/')),
            "knn_entries.yaml".to_string(),
        ),
        None => (entry.path.clone(), "dataset.yaml".to_string()),
    };

    let raw = match fetch_yaml_text(&source_url) {
        Ok(s) => s,
        Err(e) => {
            return (
                format!("Source error — {title_suffix}"),
                vec![
                    Line::from(Span::styled(format!(" {source_url}"), tint(TEXT_MUTED))),
                    Line::from(""),
                    Line::from(Span::styled(format!(" error: {e}"), error_style)),
                ],
            );
        }
    };

    let filtered = if entry.dataset_type == "knn_entries.yaml" {
        filter_knn_entries_for(&raw, &entry.name)
    } else {
        raw
    };

    // Preserve verbatim whitespace by rendering each line as a raw
    // span. The Paragraph widget already honours the line breaks;
    // the leading space keeps content from kissing the border.
    let lines: Vec<Line<'static>> = std::iter::once(
        Line::from(Span::styled(format!(" {source_url}"), tint(TEXT_MUTED))),
    )
    .chain(std::iter::once(Line::from("")))
    .chain(filtered.lines().map(|l| {
        Line::from(Span::styled(format!(" {l}"), body_style))
    }))
    .collect();

    (format!("Source — {title_suffix}"), lines)
}

/// Read raw text from a URL or filesystem path. `file://` URIs are
/// stripped to their path component; bare `http(s)://` URLs route
/// through the shared transport client (warm connection pool, same
/// 32-way runtime fanout as the rest of the runtime); everything
/// else is treated as a filesystem path.
fn fetch_yaml_text(location: &str) -> Result<String, String> {
    // `s3://` catalogs fetch over their virtual-hosted HTTPS form,
    // same as every data read.
    if crate::transport::is_remote_url(location) {
        let location = crate::transport::normalize_remote_url(location);
        let location = location.as_ref();
        let client = crate::transport::shared_client_for(location);
        let parsed = url::Url::parse(location).ok();
        return crate::transport::apply_read_auth(client.get(location), parsed.as_ref()).send()
            .map_err(|e| format!("HTTP fetch: {e}"))?
            .error_for_status()
            .map_err(|e| format!("HTTP status: {e}"))?
            .text()
            .map_err(|e| format!("HTTP decode: {e}"));
    }
    let fs_path: &str = if let Some(rest) = location.strip_prefix("file://") {
        // file:///abs → /abs; file://host/abs → /abs.
        if rest.starts_with('/') { rest }
        else if let Some(slash) = rest.find('/') { &rest[slash..] }
        else { rest }
    } else {
        location
    };
    std::fs::read_to_string(fs_path)
        .map_err(|e| format!("read {fs_path}: {e}"))
}

/// Filter a `knn_entries.yaml` body to the blocks belonging to a
/// specific dataset. Keeps `_defaults:` (the file's shared base_url
/// / cache_dir context) plus every entry whose key starts with
/// `"<dataset>:`. A "block" is a top-level key line plus its
/// indented continuation. Blank lines between kept blocks are
/// preserved so the output stays readable; other blank lines are
/// dropped to keep the filtered view tight.
fn filter_knn_entries_for(content: &str, dataset_name: &str) -> String {
    let mut out = String::new();
    let mut iter = content.lines().peekable();
    while let Some(line) = iter.next() {
        // Top-level key lines: not blank, not indented, contain `:`.
        let is_top_level = !line.is_empty()
            && !line.starts_with(' ')
            && !line.starts_with('\t')
            && line.contains(':');
        if !is_top_level {
            // Carry through file-leading comments verbatim; drop
            // everything else at the top level (blank lines between
            // entries we're filtering out).
            if line.starts_with('#') {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        // Extract the key (quoted or bare).
        let key_raw = line.split(':').next().unwrap_or("");
        let key = key_raw.trim().trim_matches('"');
        let keep = key == "_defaults"
            || key.split(':').next().is_some_and(|name| name == dataset_name);

        if keep {
            out.push_str(line);
            out.push('\n');
            // Consume continuation: indented lines and blank lines
            // that sit between continuation lines.
            while let Some(next) = iter.peek() {
                if next.is_empty()
                    || next.starts_with(' ')
                    || next.starts_with('\t')
                {
                    out.push_str(next);
                    out.push('\n');
                    iter.next();
                } else {
                    break;
                }
            }
        } else {
            // Skip block: consume its continuation lines without
            // emitting them.
            while let Some(next) = iter.peek() {
                if next.is_empty()
                    || next.starts_with(' ')
                    || next.starts_with('\t')
                {
                    iter.next();
                } else {
                    break;
                }
            }
        }
    }
    if out.is_empty() {
        format!("# no entries matched '{dataset_name}' in knn_entries.yaml\n")
    } else {
        out
    }
}

/// Like [`kv`] but with a custom leading-indent depth (in spaces of
/// 2). Used by the descriptor view's nested facet fields.
fn kv_indent(depth: usize, key: &str, value: &str, key_style: Style, val_style: Style) -> Line<'static> {
    let pad = " ".repeat(depth * 2 + 2);
    Line::from(vec![
        Span::raw(pad),
        Span::styled(format!("{key:<12}"), key_style),
        Span::raw(" "),
        Span::styled(value.to_string(), val_style),
    ])
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
    if let Some(bracket) = source_path.find(['[', '(']) {
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
/// Walk every per-dataset directory under `cache_root` (identified by
/// the presence of an `origin.json`), then within each one read every
/// `.mrkl` / `.chunks` sidecar to extract `(valid, total)` chunk
/// counts. Returns a map keyed by the *facet URL* — the per-dataset
/// origin URL joined with the sidecar's filesystem path relative to
/// the dataset directory. Callers consult this map by canonicalising
/// the row's facet URL the same way and looking it up.
///
/// Layout assumed (the natural-cache layout established by
/// [`crate::storage::layout_for_url`]):
///
/// ```text
/// <cache_root>/<authority>/<.../subdirs>/
///   ├── origin.json
///   ├── base.fvec
///   ├── base.fvec.mrkl
///   ├── query.fvec
///   └── query.fvec.chunks
/// ```
fn survey_cache_state(cache_dir: &std::path::Path)
    -> std::collections::HashMap<String, FacetCacheView>
{
    let mut map: std::collections::HashMap<String, FacetCacheView> =
        std::collections::HashMap::new();
    walk_dataset_dirs(cache_dir, &mut |dataset_dir| {
        let Some(origin) = crate::cache::layout::read_dataset_origin(dataset_dir) else { return; };
        let base_url = origin.source;
        walk_sidecars(dataset_dir, dataset_dir, &base_url, &mut map);
    });
    map
}

/// Per-facet cache state captured by [`survey_cache_state`]. Carries
/// enough chunk-level detail that windowed coverage can be computed
/// at picker time — without the per-chunk bitmap, every sized profile
/// sharing a base file would report the whole file's chunk count and
/// the CACHED column would look identical across windows of wildly
/// different sizes.
pub(crate) struct FacetCacheView {
    /// Chunk size in bytes (from the merkle reference, or the
    /// `.chunks` sidecar's implied 8 MiB).
    chunk_size: u64,
    /// Total chunk count for the whole file.
    total_chunks: u32,
    /// Total file size in bytes.
    total_size: u64,
    /// Per-chunk validity bitmap, `true` when the chunk is downloaded
    /// and (for merkle-backed entries) verified.
    valid_bits: Vec<bool>,
    /// Absolute path to the cache file. Used by windowed-coverage
    /// computation to peek at byte 0..4 — the xvec dim header — to
    /// derive bytes-per-record without separate format-specific
    /// metadata.
    cache_file_path: std::path::PathBuf,
}

impl FacetCacheView {
    /// Total valid chunks across the whole file.
    fn whole_file_valid(&self) -> u32 {
        self.valid_bits.iter().filter(|&&b| b).count() as u32
    }
}

/// Compute the picker's `(valid, total)` chunk contribution for a
/// single facet. For windowed views (sized profiles), this bounds
/// counting to chunks covering the window's byte range — so the
/// CACHED column reflects what the profile actually needs, not the
/// shared base file's full chunk count.
///
/// Falls back to whole-file counts when:
///   - the view has no window (default profile);
///   - the format isn't a uniform-stride xvec (vvec / parquet / etc.
///     would need format-specific record→byte logic we don't have at
///     this layer);
///   - bytes-per-record can't be read from the cache file (chunk 0
///     isn't downloaded yet, so the xvec dim header is unavailable).
fn window_bounded_coverage(
    cv: &FacetCacheView,
    view: &crate::dataset::profile::DSView,
    source_path: &str,
    ext: &str,
    elem_size: usize,
) -> (u32, u32) {
    // Window: `view.effective_window()` returns the view-level
    // override when set, else the source-level window. The catalog's
    // explicit form (`base_vectors: { source: ..., window: [...] }`)
    // populates `view.window` (sibling of `source`), so reading from
    // `view.source.window` alone misses it entirely — and was the
    // reason every sized profile reported the whole-file chunk count
    // even after windowed coverage was wired up. Fall back to a
    // `[start..end)` suffix embedded in the source path for catalogs
    // that use the legacy `Simple("path[0..N)")` form.
    let effective = view.effective_window();
    let window: Option<(u64, u64)> = effective.0.first()
        .map(|iv| (iv.min_incl, iv.max_excl))
        .or_else(|| parse_path_suffix_window(source_path));

    let Some((win_start, win_end)) = window else {
        return (cv.whole_file_valid(), cv.total_chunks);
    };
    if win_end <= win_start {
        return (cv.whole_file_valid(), cv.total_chunks);
    }

    // Format guard: only uniform-stride xvec is record→byte computable
    // from `4 + dim * elem_size`. vvec / parquet need metadata we
    // don't carry at this layer.
    if crate::io::is_vvec_ext(ext) {
        return (cv.whole_file_valid(), cv.total_chunks);
    }

    let Some(bpr) = read_xvec_bpr(&cv.cache_file_path, elem_size) else {
        // Cache file isn't downloaded far enough to read the dim
        // header — windowed coverage can't be computed; reporting
        // the whole-file count is the only honest fallback.
        return (cv.whole_file_valid(), cv.total_chunks);
    };

    let byte_start = win_start.saturating_mul(bpr).min(cv.total_size);
    let byte_end = win_end.saturating_mul(bpr).min(cv.total_size);
    chunks_for_byte_range(cv, byte_start, byte_end)
}

/// Read the 4-byte xvec dim header at byte 0 of the cache file, then
/// compute bytes-per-record as `4 + dim * elem_size`. Returns `None`
/// when chunk 0 isn't downloaded (sparse hole at the start of the
/// file → reads as zeros → dim parses as 0 → rejected by sanity
/// bounds) or when the file can't be opened.
fn read_xvec_bpr(cache_file: &std::path::Path, elem_size: usize) -> Option<u64> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    let mut f = std::fs::File::open(cache_file).ok()?;
    f.read_exact(&mut buf).ok()?;
    let dim = i32::from_le_bytes(buf);
    if dim <= 0 || dim > 1_000_000 { return None; }
    Some(4 + (dim as u64) * (elem_size as u64))
}

/// Count `(valid_chunks_in_range, total_chunks_in_range)` for the
/// chunks intersecting `[byte_start, byte_end)`. An empty range
/// returns `(0, 0)`.
fn chunks_for_byte_range(
    cv: &FacetCacheView,
    byte_start: u64,
    byte_end: u64,
) -> (u32, u32) {
    if cv.chunk_size == 0 || byte_start >= byte_end { return (0, 0); }
    let first = (byte_start / cv.chunk_size) as u32;
    let last_inclusive = ((byte_end - 1) / cv.chunk_size) as u32;
    let last_inclusive = last_inclusive.min(cv.total_chunks.saturating_sub(1));
    if first > last_inclusive { return (0, 0); }
    let total = last_inclusive - first + 1;
    let mut valid = 0u32;
    for i in first..=last_inclusive {
        if let Some(&b) = cv.valid_bits.get(i as usize)
            && b { valid += 1; }
    }
    (valid, total)
}

/// Parse a `[start..end)` window suffix from a source path. Returns
/// `None` when the path has no suffix or the contents don't parse —
/// callers fall back to either the explicit `view.source.window`
/// field or treat the facet as unwindowed.
fn parse_path_suffix_window(path: &str) -> Option<(u64, u64)> {
    let parsed = crate::dataset::source::parse_source_string(path).ok()?;
    let iv = parsed.window.0.first()?;
    Some((iv.min_incl, iv.max_excl))
}

/// Recursively walk `root`, invoking `visit` on every directory that
/// contains an `origin.json` (i.e., every per-dataset cache
/// directory). Descends through the in-between authority/path-segment
/// directories that have no origin.json.
fn walk_dataset_dirs<F: FnMut(&std::path::Path)>(root: &std::path::Path, visit: &mut F) {
    let Ok(entries) = std::fs::read_dir(root) else { return; };
    let mut has_origin = false;
    let mut sub_dirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path.file_name().is_some_and(|n| n == crate::cache::layout::DATASET_ORIGIN_FILE) {
                has_origin = true;
            }
        } else if path.is_dir() {
            sub_dirs.push(path);
        }
    }
    if has_origin {
        visit(root);
        // A dataset directory can still nest further (sized
        // profiles, partition dirs) — keep walking subdirs in case
        // they're separate datasets in their own right with their
        // own `origin.json`.
    }
    for sub in sub_dirs {
        walk_dataset_dirs(&sub, visit);
    }
}

/// Within a single dataset directory (where origin.json lives), walk
/// every file and record per-chunk cache state. The URL recorded in
/// the map is `<base_url><relpath>` where `relpath` is the sidecar's
/// logical filename (`.mrkl` / `.chunks` trimmed) relative to the
/// dataset directory.
fn walk_sidecars(
    dataset_dir: &std::path::Path,
    current: &std::path::Path,
    base_url: &str,
    map: &mut std::collections::HashMap<String, FacetCacheView>,
) {
    let Ok(entries) = std::fs::read_dir(current) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_sidecars(dataset_dir, &path, base_url, map);
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue; };
        // Strip `.mrkl` / `.chunks` to get the data file's path;
        // that path is the cache file we'll later peek at (for the
        // xvec dim header) when computing windowed coverage.
        let cache_file_path = path.with_extension("");
        let view = match ext {
            "mrkl" => crate::merkle::MerkleState::load(&path).ok().map(|s| {
                let shape = s.shape();
                let total_chunks = shape.total_chunks;
                let valid_bits: Vec<bool> = (0..total_chunks).map(|i| s.is_valid(i)).collect();
                FacetCacheView {
                    chunk_size: shape.chunk_size,
                    total_chunks,
                    total_size: shape.total_content_size,
                    valid_bits,
                    cache_file_path: cache_file_path.clone(),
                }
            }),
            "chunks" => std::fs::read(&path).ok().map(|bytes| {
                // The `.chunks` sidecar carries one byte per chunk —
                // non-zero means valid. We don't have a shape sidecar
                // alongside it; chunk_size and total_size come from
                // the cache file itself (the chunk size used by
                // `ChunkStore` is fixed at the storage layer; we
                // approximate total_size from the file metadata so
                // windowed coverage still works).
                let total = bytes.len() as u32;
                let valid_bits: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
                let chunk_size = crate::chunked_http::DEFAULT_CHUNK_SIZE;
                let total_size = std::fs::metadata(&cache_file_path)
                    .ok()
                    .map(|m| m.len())
                    .unwrap_or((total as u64) * chunk_size);
                FacetCacheView {
                    chunk_size,
                    total_chunks: total,
                    total_size,
                    valid_bits,
                    cache_file_path: cache_file_path.clone(),
                }
            }),
            _ => None,
        };
        let Some(view) = view else { continue; };
        let Ok(relpath) = cache_file_path.strip_prefix(dataset_dir) else { continue; };
        let relpath_str = relpath.to_string_lossy().replace('\\', "/");
        let facet_url = format!("{}{}", base_url.trim_end_matches('/'), {
            // base_url already ends with `/` (recorded as the parent
            // URL). If it doesn't, add one. Use `.starts_with('/')`
            // on the relpath to avoid a double slash either way.
            if relpath_str.starts_with('/') {
                relpath_str.clone()
            } else {
                format!("/{relpath_str}")
            }
        });
        map.insert(canonicalize_cache_url(&facet_url), view);
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
    // Walk dataset directories the same way the survey does. For each
    // one, check whether ANY facet URL we know belongs to this entry
    // would resolve into this dataset_dir — if so the whole tree
    // belongs to us and gets removed atomically.
    walk_dataset_dirs(cache_dir, &mut |dataset_dir| {
        let Some(origin) = crate::cache::layout::read_dataset_origin(dataset_dir) else { return; };
        // Canonicalise the recorded origin before comparing — without
        // this, an s3:// origin and a facet URL already-canonicalised
        // to https:// (by `entry_facet_urls`) would never share a
        // prefix and purge would silently match nothing. Both sides
        // must travel the same translation through
        // `transport::normalize_remote_url`.
        let base_url = canonicalize_cache_url(&origin.source);
        // A dataset directory's recorded origin is the *parent URL*
        // of every facet that ought to live there. We check whether
        // any facet URL we know has the dataset directory's origin
        // as a URL prefix. Conservative — only delete what's clearly
        // ours, never sibling caches that happen to share an origin
        // host.
        let matches = facet_urls.iter().any(|u| base_url_matches_facet(&base_url, u));
        if !matches { return; }
        let bytes = leaf_size_bytes(dataset_dir);
        if std::fs::remove_dir_all(dataset_dir).is_ok() {
            freed += bytes;
            removed.push(dataset_dir.to_path_buf());
        }
    });
    (removed, freed)
}

/// True iff a facet URL is a child of `base_url` (the recorded
/// parent-URL origin of a dataset directory). The picker computes
/// every facet URL declared by the dataset entry; if any of them
/// has the dataset directory's origin as a prefix, the directory
/// belongs to this dataset.
fn base_url_matches_facet(base_url: &str, facet_url: &str) -> bool {
    let base = base_url.trim_end_matches('/');
    facet_url.starts_with(&format!("{base}/")) || facet_url == base
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
                    if let Ok(meta) = e.metadata() {
                        total += crate::cache::reader::allocated_size(&meta);
                    }
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
            if n.is_multiple_of(iec) {
                return format!("{}{}", n / iec, iec_label);
            } else if n.is_multiple_of(si) {
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
    /// Open a scrollable text view of the full catalog descriptor
    /// for this dataset+profile (attributes, facets with windows,
    /// origin URLs, cache state). Renders inside the picker as an
    /// overlay — does not exit the picker.
    Describe,
    /// Open a scrollable text view of the raw catalog YAML — the
    /// `dataset.yaml` file for canonical catalogs, or the relevant
    /// entries pulled out of `knn_entries.yaml` for the legacy
    /// shape. Picker-local like Describe.
    Source,
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
            PickerAction::Describe  => "Describe",
            PickerAction::Source    => "Source",
            PickerAction::Precache  => "Precache",
            PickerAction::Purge     => "Purge",
            PickerAction::Ping      => "Ping",
        }
    }

    fn description(self) -> &'static str {
        match self {
            PickerAction::Visualize =>
                "Open the interactive explorer: norms, distances, eigenvalues, PCA.",
            PickerAction::Describe =>
                "Show full descriptor: attributes, facets (paths + windows), origin, cache state. Scrollable.",
            PickerAction::Source =>
                "Show the raw catalog YAML — dataset.yaml verbatim, or the relevant entries from knn_entries.yaml. Scrollable.",
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
    /// Describe sits second so a quick "what's in this profile?"
    /// look is one keystroke away; Source is right after for the
    /// "show me the raw YAML" path.
    fn menu_order() -> &'static [PickerAction] {
        &[
            PickerAction::Visualize,
            PickerAction::Describe,
            PickerAction::Source,
            PickerAction::Precache,
            PickerAction::Ping,
            PickerAction::Purge,
        ]
    }

    /// True for actions handled inside the picker (overlay state only)
    /// rather than dispatched to the runner. Picker-local actions
    /// don't suspend the chrome and don't go through `dispatch`.
    fn is_picker_local(self) -> bool {
        matches!(self, PickerAction::Describe | PickerAction::Source)
    }

    /// True when this action can sensibly run against a set of
    /// selected rows in sequence. `Visualize` is excluded because
    /// it opens an interactive single-target viewer; `Describe` and
    /// `Source` are picker-local overlays that only have meaning
    /// for one row at a time. The remaining three — precache,
    /// ping, purge — chain naturally across a batch.
    fn batch_safe(self) -> bool {
        matches!(self,
            PickerAction::Precache
            | PickerAction::Ping
            | PickerAction::Purge)
    }

    /// Batch menu order — same precedence as `menu_order` but
    /// filtered to actions that make sense across many rows.
    fn batch_menu_order() -> Vec<PickerAction> {
        Self::menu_order().iter().copied().filter(|a| a.batch_safe()).collect()
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
where F: FnMut(&str, PickerAction, bool) -> ActionFlow,
{
    // Third dispatch argument is `pause_after`: when true, the runner
    // pauses for a keypress after printing its output so the user can
    // read it before the picker's chrome takes the screen back. The
    // picker passes `true` for single-row dispatches and for the
    // *last* item in a batch loop only, so a 10-row batch precache
    // pauses once at the end instead of 10 times in a row.
    // Load catalog
    let sources = CatalogSources::new().configure_default();
    if sources.is_empty() {
        eprintln!("error: no catalog sources configured");
        eprintln!();
        eprintln!("Add one with `vectordata config catalog add <URL-or-path>`.");
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
    // ── Multi-select state ─────────────────────────────────────────
    // `selected` is the persistent set of toggled row specifiers
    // (`<dataset>:<profile>`). Space toggles the cursor row in/out.
    //
    // `range_anchor` is the *temporary* shift-arrow extend anchor.
    // While Some, the rows between anchor and cursor render with the
    // selection (REVERSED) bar drawn over the whole range, and a
    // subsequent Space commits that range into `selected` (toggling
    // each row's membership) and clears the anchor. Plain arrow keys
    // clear the anchor — shift-arrow is the only way to enter range
    // mode, so any other movement is treated as exiting it.
    let mut selected: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut range_anchor: Option<usize> = None;
    // When the action menu opens, this flag records whether it's a
    // single-target menu (cursor row only) or a batch menu (every
    // member of `selected`). The two differ in their action list
    // (batch excludes Visualize / Describe / Source) and in how
    // Enter dispatches.
    let mut menu_is_batch = false;
    // Disambiguation prompt: opens when the user hits Enter with a
    // non-empty `selected` set but the cursor row is *not* itself
    // selected. We can't tell whether they meant "apply to the row
    // I'm pointing at" or "apply to the N rows I previously
    // toggled" — the prompt asks. Three options: highlighted row,
    // batch, cancel.
    let mut disambig_open = false;
    let mut disambig_cursor: usize = 0;
    // Scrollable text overlay. Reused by Describe (parsed descriptor)
    // and Source (raw catalog YAML) — both populate `descriptor_lines`
    // + `descriptor_title` and flip `descriptor_open`. Lines are
    // recomputed per-open from the current row + entry so cache
    // state and remote fetch results stay fresh.
    let mut descriptor_open = false;
    let mut descriptor_scroll: u16 = 0;
    let mut descriptor_lines: Vec<Line<'static>> = Vec::new();
    let mut descriptor_title: String = String::new();
    // Reading colour-naming detection from terminals is unreliable
    // (TERM, COLORTERM, NO_COLOR all lie or are missing in tmux /
    // ssh / screen / older xterm-256 setups). Instead, default to
    // colour-on and let the user toggle with Ctrl-T when their
    // terminal mangles the palette into illegible blocks. Selection
    // is rendered via `Modifier::REVERSED` in both modes so it
    // remains visible even when fg/bg colours are stripped.
    let mut colors_enabled = true;
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

        // Local closures bound to the current theme state. `tinted`
        // applies an accent colour only when colours are enabled; in
        // mono mode it returns terminal-default styling (with modifier
        // bits preserved). `selected_style` is always `REVERSED` so
        // the highlight bar stays visible whether the user has colour
        // turned on or has Ctrl-T'd it off because their terminal
        // butchers ANSI colours.
        let theme_on = colors_enabled;
        let tinted = |c: Color| -> Style {
            if theme_on { Style::default().fg(c) } else { Style::default() }
        };
        let bordered = |c: Color| -> Style {
            if theme_on { Style::default().fg(c) } else { Style::default() }
        };
        let selected_style = || -> Style {
            Style::default().add_modifier(Modifier::REVERSED)
        };

        let drew = terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // filter input
                    Constraint::Min(5),    // list
                    Constraint::Length(3),  // detail panel
                    Constraint::Length(2),  // footer: selection summary + keybinds
                ]).split(frame.area());

            // Filter input — neon "Filter:" label, cyan caret, lavender
            // chrome on the box so the picker reads as a single tinted
            // panel instead of plain ASCII. In mono mode all tints
            // collapse to terminal default.
            let filter_line = Line::from(vec![
                Span::styled(" Filter: ", tinted(TEXT_MUTED)),
                Span::styled(filter.clone(), tinted(TEXT_PRIMARY)),
                Span::styled("█", tinted(ACCENT_CYAN)),
            ]);
            frame.render_widget(
                Paragraph::new(filter_line)
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(bordered(ACCENT_LAVENDR))
                        .title(Span::styled(
                            format!(" Select Dataset — {} shown ({}) ",
                                visible.len(),
                                if show_all_profiles { "all profiles" } else { "default only — Tab for all" }),
                            tinted(ACCENT_CYAN),
                        ))),
                chunks[0],
            );

            // Dataset list. `list_height` is the inner height of the
            // bordered box (frame minus top/bottom borders). Of that,
            // one row goes to the column header — leaving
            // `data_rows = list_height - 1` for selectable rows. The
            // scroll calculation below must use `data_rows`, not
            // `list_height`, otherwise the cursor can park one row
            // below the last rendered data row and visibly land on
            // the box's bottom border.
            let list_height = chunks[1].height.saturating_sub(2) as usize;
            let data_rows = list_height.saturating_sub(1);
            if data_rows > 0 && cursor >= scroll + data_rows {
                scroll = cursor - data_rows + 1;
            }
            if cursor < scroll {
                scroll = cursor;
            }

            if show_help {
                let help = vec![
                    Line::from(Span::styled(" Dataset Picker — Keyboard Shortcuts", tinted(ACCENT_CYAN))),
                    Line::from(""),
                    Line::from(" Navigation"),
                    Line::from("   ↑ / ↓                 Move cursor"),
                    Line::from("   PgUp / PgDn           Jump by page"),
                    Line::from("   Enter / →             Expand dataset / open action menu"),
                    Line::from("   ←                     Collapse expanded dataset"),
                    Line::from("   Tab                   Toggle: default profiles / all profiles"),
                    Line::from(""),
                    Line::from(" Multi-select"),
                    Line::from("   Space                 Toggle the cursor row into / out of selection"),
                    Line::from("   Shift+↑ / Shift+↓     Extend a temporary highlighted range"),
                    Line::from("   Space (in range)      Toggle every row in the highlighted range"),
                    Line::from("   Enter (with selection) Open batch action menu (Precache / Ping / Purge)"),
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
                    Line::from("   Ctrl-T                Toggle colour / mono theme"),
                    Line::from(""),
                    Line::from(" Exit"),
                    Line::from("   Esc                   Drop range → clear selection → close details → collapse → clear filter → exit"),
                    Line::from("   q / Ctrl-C            Quit immediately"),
                    Line::from("   ?                     Toggle this help"),
                ];
                frame.render_widget(
                    Paragraph::new(help).block(Block::default().borders(Borders::ALL)
                        .border_style(bordered(ACCENT_LAVENDR))
                        .title(Span::styled(" Help ", tinted(ACCENT_CYAN)))),
                    chunks[1],
                );
            } else {

            // Column widths measured in DISPLAY columns. The prefix
            // is split into independent slots so the cursor pip and
            // the multi-select toggle marker never collide:
            //   col 0  cursor pip ('▶' or ' ')
            //   col 1  toggle mark ('●' or ' ')
            //   col 2-4 tree glyph (up to 3 cols: " └ ", "▸ ", etc.)
            // Reserve 5 cols on top of the longest dataset name plus
            // 2 trailing blanks so the PROFILE column has a visible
            // gap even when the widest row fills its cell edge-to-edge.
            let name_w = visible.iter()
                .map(|r| UnicodeWidthStr::width(r.dataset.as_str()) + 5 + 2)
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
                Span::styled(pad_display("DATASET", name_w), tinted(TEXT_MUTED)),
                Span::styled(pad_display("PROFILE", prof_w), tinted(TEXT_MUTED)),
                Span::styled(pad_display("FACETS", facet_w), tinted(TEXT_MUTED)),
                Span::styled(pad_display("METRIC", metric_w), tinted(TEXT_MUTED)),
                Span::styled(pad_display("SIZE", size_w), tinted(TEXT_MUTED)),
                Span::styled(pad_display("ACCESS", access_w), tinted(TEXT_MUTED)),
                Span::styled(pad_display("CACHED", cache_w), tinted(TEXT_MUTED)),
            ])];

            // Compute the inclusive range of visible-row indices
            // that should render with the REVERSED bar. When
            // `range_anchor` is set, the whole anchor..=cursor span
            // is highlighted — that's the shift-arrow "I am about
            // to toggle these" preview. Otherwise just the cursor.
            let (highlight_lo, highlight_hi) = match range_anchor {
                Some(a) => (a.min(cursor), a.max(cursor)),
                None    => (cursor, cursor),
            };
            for (i, row) in visible.iter().enumerate().skip(scroll).take(list_height.saturating_sub(1)) {
                let is_cursor = i == cursor;
                let in_range = i >= highlight_lo && i <= highlight_hi;
                let is_toggled = selected.contains(&row.specifier());
                // Selection is always REVERSED — independent of the
                // colour state, so the highlight bar stays visible
                // on terminals that mangle ANSI 24-bit colour or
                // when the user has Ctrl-T'd colours off. The
                // REVERSED bar now covers any row in the visual
                // range (cursor or shift-arrow extend).
                let style = if in_range {
                    selected_style()
                } else if is_toggled {
                    // Persistent multi-select: tinted accent so a
                    // toggled-but-not-current row is visible without
                    // overwhelming the cursor's REVERSED bar.
                    tinted(ACCENT_LAVENDR)
                } else if row.cache_status != "—" {
                    tinted(ACCENT_MINT)
                } else {
                    tinted(TEXT_PRIMARY)
                };
                let cache_style = if in_range {
                    style
                } else if row.cache_status != "—" {
                    tinted(ACCENT_MINT)
                } else {
                    tinted(TEXT_DIM)
                };

                // Row prefix layout (display columns):
                //   col 0  cursor pip   '▶' or ' '
                //   col 1  toggle mark  '●' or ' '
                //   col 2+ tree glyph + dataset name
                // Two independent marker columns so a row that's
                // both the cursor AND toggled shows '▶●' — neither
                // marker hides the other. In mono mode (Ctrl-T) both
                // remain visible by shape alone; with colour, the
                // REVERSED highlight bar layers on top of the cursor
                // row and the lavender tint on toggled rows still
                // distinguishes them at a glance.
                let is_expanded = expanded.contains(&row.dataset);
                let has_siblings = all_rows.iter().filter(|r| r.dataset == row.dataset).count() > 1;
                let is_child = is_expanded && row.profile != "default";

                let cursor_pip = if is_cursor { '▶' } else { ' ' };
                let toggle_mark = if is_toggled { '●' } else { ' ' };
                let (tree_glyph, name_display) = if show_all_profiles {
                    ("", row.dataset.as_str())
                } else if is_child {
                    (" └ ", row.dataset.as_str())
                } else if has_siblings && !is_expanded {
                    ("▸ ", row.dataset.as_str())
                } else if has_siblings && is_expanded {
                    ("▾ ", row.dataset.as_str())
                } else {
                    ("  ", row.dataset.as_str())
                };

                let name_field = format!("{cursor_pip}{toggle_mark}{tree_glyph}{name_display}");
                let name_cell = fit_display(&name_field, name_w);

                let access_color = match row.access {
                    AccessMode::Local         => ACCENT_CYAN,
                    AccessMode::MerkleHashed  => ACCENT_MINT,
                    AccessMode::MerkleChunked => ACCENT_LAVENDR,
                    AccessMode::FullTransfer  => ACCENT_CORAL,
                };
                let access_style = if in_range {
                    style
                } else {
                    tinted(access_color)
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
                        .border_style(bordered(ACCENT_LAVENDR))),
                chunks[1],
            );

            } // close help/list else

            // Detail panel for selected row. Facet legend turns into
            // present/absent emphasis via brackets in mono mode so
            // users without colour can still tell which facets are
            // declared by this profile (mint/amber/pink/coral in the
            // colour palette collapse to undifferentiated terminal
            // default otherwise).
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
                    let present = row.facets.contains(*code);
                    if theme_on {
                        let style = if present { Style::default().fg(*color) }
                                    else { Style::default().fg(TEXT_DIM) };
                        spans.push(Span::styled(format!("{} ", label), style));
                    } else {
                        // Mono: bracket present facets, strike absent ones with
                        // surrounding dots so the legend still reads at a glance.
                        let glyph = if present { format!("[{label}] ") }
                                    else { format!(" {label}  ") };
                        spans.push(Span::raw(glyph));
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
                    tinted(access_color),
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
                    .border_style(bordered(ACCENT_LAVENDR))),
                chunks[2],
            );

            // Footer: two lines. The first is the selection summary
            // — always present per the multi-select UX contract, so
            // the user can see at a glance how many rows their next
            // batch action will hit. The second is the keybind
            // cheat sheet. The range-mode hint only renders when
            // shift-arrow is active, to avoid noise the rest of the
            // time.
            let selected_count = selected.len();
            let range_count = range_anchor.map(|a| {
                let lo = a.min(cursor);
                let hi = a.max(cursor);
                hi - lo + 1
            });
            let summary = match (selected_count, range_count) {
                (0, None)        => " No rows selected — Space toggles · Shift+↑↓ extends range".to_string(),
                (n, None)        => format!(" Selected: {n} · Space toggles cursor · Enter for batch menu"),
                (n, Some(r))     => format!(" Selected: {n} · Range: {r} highlighted · Space toggles range"),
            };
            let mono_hint = if theme_on { "Ctrl-T: mono" } else { "Ctrl-T: colour" };
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(summary, tinted(ACCENT_CYAN))),
                    Line::from(Span::styled(
                        format!(" ↑↓: navigate | Shift+↑↓: range | Space: toggle | Enter/→: expand | ←: collapse | Tab: all profiles | Ctrl-D: details | {mono_hint} | Esc: back | q: quit"),
                        tinted(TEXT_MUTED))),
                ]),
                chunks[3],
            );

            // Stats overlay (Ctrl-D toggle). Renders on top of the
            // list area, anchored to the bottom of the screen and
            // covering roughly the lower 60% horizontally centred,
            // so the filter line and footer stay visible.
            if show_details
                && let Some(row) = visible.get(cursor) {
                    let outer = frame.area();
                    let popup_h = outer.height.saturating_sub(6).clamp(6, 20);
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
                    let lines = build_details_lines(row, entry, theme_on);
                    frame.render_widget(ratatui::widgets::Clear, area);
                    frame.render_widget(
                        Paragraph::new(lines)
                            .block(Block::default().borders(Borders::ALL)
                                .border_style(bordered(ACCENT_LAVENDR))
                                .title(Span::styled(
                                    format!(" Dataset Details — {}:{} (Ctrl-D / Esc to close) ", row.dataset, row.profile),
                                    tinted(ACCENT_CYAN),
                                ))),
                        area,
                    );
                }

            // Descriptor overlay (Describe action). Larger than the
            // Ctrl-D details popup so the full facet list, windows,
            // and origin URLs stay readable; scrollable to handle
            // dataset.yaml descriptors that overflow the screen.
            if descriptor_open {
                // No row guard: descriptor_title was captured when
                // the overlay opened, so the popup keeps making
                // sense even if the user's filter or expansion
                // state shifts the row index behind the overlay.
                let outer = frame.area();
                let popup_w = outer.width.saturating_sub(6).max(40);
                let popup_h = outer.height.saturating_sub(4).max(8);
                let popup_x = (outer.width.saturating_sub(popup_w)) / 2;
                let popup_y = (outer.height.saturating_sub(popup_h)) / 2;
                let area = ratatui::layout::Rect {
                    x: outer.x + popup_x,
                    y: outer.y + popup_y,
                    width: popup_w,
                    height: popup_h,
                };
                // Clamp scroll so the bottom of the descriptor
                // never disappears off-screen — if the user
                // resizes the terminal smaller, the scroll
                // position needs to slide back up.
                let inner_h = popup_h.saturating_sub(2); // borders
                let max_scroll = (descriptor_lines.len() as u16).saturating_sub(inner_h);
                if descriptor_scroll > max_scroll {
                    descriptor_scroll = max_scroll;
                }
                frame.render_widget(ratatui::widgets::Clear, area);
                frame.render_widget(
                    Paragraph::new(descriptor_lines.clone())
                        .scroll((descriptor_scroll, 0))
                        .block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(ACCENT_LAVENDR))
                            .title(Span::styled(
                                format!(" {} ({}/{} · ↑↓/PgUp/PgDn/Home/End · Esc to close) ",
                                    descriptor_title,
                                    descriptor_scroll + 1,
                                    descriptor_lines.len().max(1)),
                                tinted(ACCENT_CYAN),
                            ))),
                    area,
                );
            }

            // Disambiguation prompt: cursor row is outside the toggled
            // selection, so we ask which target the user meant. Three
            // options live in the popup: highlighted row, batch over
            // the toggled set, cancel.
            if disambig_open {
                let cursor_spec = visible.get(cursor)
                    .map(|r| r.specifier())
                    .unwrap_or_else(|| "(no row)".to_string());
                let options: [String; 3] = [
                    format!("Apply action to highlighted row ({cursor_spec})"),
                    format!("Apply action to {} selected row(s)", selected.len()),
                    "Cancel".to_string(),
                ];
                if disambig_cursor >= options.len() { disambig_cursor = 0; }
                let outer = frame.area();
                let popup_h: u16 = (options.len() as u16) + 4;
                let popup_w: u16 = 70;
                let area = ratatui::layout::Rect {
                    x: outer.x + outer.width.saturating_sub(popup_w) / 2,
                    y: outer.y + outer.height.saturating_sub(popup_h) / 2,
                    width: popup_w.min(outer.width),
                    height: popup_h.min(outer.height),
                };
                let mut lines: Vec<Line<'static>> = Vec::new();
                for (i, opt) in options.iter().enumerate() {
                    let marker = if i == disambig_cursor { " ▸ " } else { "   " };
                    let style = if i == disambig_cursor {
                        selected_style()
                    } else {
                        tinted(TEXT_PRIMARY)
                    };
                    lines.push(Line::from(Span::styled(
                        format!("{marker}{opt}"),
                        style,
                    )));
                }
                lines.push(Line::from(""));
                frame.render_widget(ratatui::widgets::Clear, area);
                frame.render_widget(
                    Paragraph::new(lines)
                        .block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(ACCENT_LAVENDR))
                            .title(Span::styled(
                                " Which target? (↑↓ · Enter · Esc) ",
                                tinted(ACCENT_LAVENDR),
                            ))),
                    area,
                );
            }

            // Action menu (opens on Enter against a leaf row). Drawn
            // last so it lands on top of any other overlay. Selection
            // cursor `▸` doubles the colour-based reverse-video so it
            // is unambiguous in mono mode too.
            if menu_open {
                let actions: Vec<PickerAction> = if menu_is_batch {
                    PickerAction::batch_menu_order()
                } else {
                    PickerAction::menu_order().to_vec()
                };
                if menu_cursor >= actions.len() { menu_cursor = 0; }
                // Single-target: title shows the cursor's row. Batch:
                // title shows the selection count instead.
                let title_text = if menu_is_batch {
                    format!(" Batch action — {} row(s) selected (Enter to confirm · Esc to cancel) ",
                        selected.len())
                } else if let Some(row) = visible.get(cursor) {
                    format!(" Action — {}:{} (Enter to confirm · Esc to cancel) ",
                        row.dataset, row.profile)
                } else {
                    " Action ".to_string()
                };
                let outer = frame.area();
                let menu_h: u16 = (actions.len() as u16) + 5;
                let menu_w: u16 = 64;
                let area = ratatui::layout::Rect {
                    x: outer.x + outer.width.saturating_sub(menu_w) / 2,
                    y: outer.y + outer.height.saturating_sub(menu_h) / 2,
                    width: menu_w.min(outer.width),
                    height: menu_h.min(outer.height),
                };
                let mut lines: Vec<Line<'static>> = Vec::new();
                for (i, action) in actions.iter().enumerate() {
                    let marker = if i == menu_cursor { " ▸ " } else { "   " };
                    let style = if i == menu_cursor {
                        selected_style()
                    } else {
                        tinted(TEXT_PRIMARY)
                    };
                    lines.push(Line::from(Span::styled(
                        format!("{marker}{}", action.label()),
                        style,
                    )));
                }
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!(" {}", actions[menu_cursor].description()),
                    tinted(TEXT_MUTED),
                )));
                frame.render_widget(ratatui::widgets::Clear, area);
                frame.render_widget(
                    Paragraph::new(lines)
                        .block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(ACCENT_CYAN))
                            .title(Span::styled(title_text, tinted(ACCENT_CYAN)))),
                    area,
                );
            }
        });
        if let Err(e) = drew {
            let _ = disable_raw_mode();
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
            eprintln!("error: terminal draw failed: {e}");
            return PickerOutcome::Failed;
        }

        // Events
        if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false)
            && let Ok(Event::Key(key)) = event::read() {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    result = PickerOutcome::Done;
                    break;
                }
                if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    show_details = !show_details;
                    continue;
                }
                if key.code == KeyCode::Char('t') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    // Ctrl-T: toggle colour theme. The rendering
                    // codepaths fall back to terminal-default fg/bg
                    // when this is off; selection still inverts via
                    // `Modifier::REVERSED` so the highlight is
                    // visible regardless.
                    colors_enabled = !colors_enabled;
                    continue;
                }

                // Descriptor overlay is modal — when open, swallow
                // everything but scroll/close keys so the underlying
                // list doesn't drift behind the popup. Checked
                // *before* `menu_open` because Describe closes the
                // menu before opening this overlay (they're never
                // both open at once, but the ordering still matters
                // for the Esc cascade further down).
                if descriptor_open {
                    let page = 10u16;
                    match key.code {
                        KeyCode::Esc | KeyCode::Char('q') => {
                            descriptor_open = false;
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            descriptor_scroll = descriptor_scroll.saturating_sub(1);
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            descriptor_scroll = descriptor_scroll.saturating_add(1);
                        }
                        KeyCode::PageUp => {
                            descriptor_scroll = descriptor_scroll.saturating_sub(page);
                        }
                        KeyCode::PageDown => {
                            descriptor_scroll = descriptor_scroll.saturating_add(page);
                        }
                        KeyCode::Home | KeyCode::Char('g') => {
                            descriptor_scroll = 0;
                        }
                        KeyCode::End | KeyCode::Char('G') => {
                            descriptor_scroll = u16::MAX;
                            // Renderer clamps on next draw — saves
                            // us from re-computing the popup height
                            // here.
                        }
                        _ => {}
                    }
                    continue;
                }

                // Disambiguation prompt is modal — three-way choice
                // between "highlighted row", "batch over selected",
                // and "cancel". Confirmation routes to the appropriate
                // action menu shape.
                if disambig_open {
                    match key.code {
                        KeyCode::Esc => {
                            disambig_open = false;
                        }
                        KeyCode::Up => {
                            disambig_cursor = disambig_cursor.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            if disambig_cursor + 1 < 3 { disambig_cursor += 1; }
                        }
                        KeyCode::Enter => {
                            match disambig_cursor {
                                0 => {
                                    // Highlighted row → single-target
                                    // action menu against the cursor.
                                    disambig_open = false;
                                    menu_open = true;
                                    menu_is_batch = false;
                                    menu_cursor = 0;
                                }
                                1 => {
                                    // Batch → menu over `selected`.
                                    disambig_open = false;
                                    menu_open = true;
                                    menu_is_batch = true;
                                    menu_cursor = 0;
                                }
                                _ => {
                                    // Cancel — close the prompt and
                                    // leave the selection intact.
                                    disambig_open = false;
                                }
                            }
                        }
                        _ => {}
                    }
                    continue;
                }

                // Action menu is modal — when open, swallow everything
                // but the menu's own navigation/select/cancel keys so
                // the underlying list doesn't drift.
                if menu_open {
                    match key.code {
                        KeyCode::Esc => { menu_open = false; }
                        KeyCode::Up => {
                            menu_cursor = menu_cursor.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            let menu_len = if menu_is_batch {
                                PickerAction::batch_menu_order().len()
                            } else {
                                PickerAction::menu_order().len()
                            };
                            if menu_cursor + 1 < menu_len {
                                menu_cursor += 1;
                            }
                        }
                        KeyCode::Enter => {
                            let vis: Vec<&PickerRow> = all_rows.iter()
                                .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                                .filter(|r| matches_filter(r, &filter))
                                .collect();
                            // Batch path: loop the chosen action
                            // over every specifier in the persistent
                            // `selected` set, in sorted order so
                            // re-runs are deterministic. Picker-local
                            // actions don't make batch sense (the
                            // overlay is single-target) — they were
                            // excluded from `batch_menu_order` so
                            // there's no need to special-case here.
                            if menu_is_batch {
                                let actions = PickerAction::batch_menu_order();
                                if menu_cursor >= actions.len() { menu_cursor = 0; }
                                let action = actions[menu_cursor];
                                let mut specs: Vec<String> = selected.iter().cloned().collect();
                                specs.sort();
                                menu_open = false;
                                let _ = disable_raw_mode();
                                let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
                                let mut exit_after = false;
                                for (idx, spec) in specs.iter().enumerate() {
                                    eprintln!("── [{}/{}] {} {} ──",
                                        idx + 1, specs.len(), action.label(), spec);
                                    // Pause only after the LAST item.
                                    // Intermediate items chain through
                                    // so the user reads the whole batch
                                    // output once and presses Enter once.
                                    let is_last = idx + 1 == specs.len();
                                    let flow = dispatch(spec, action, is_last);
                                    if matches!(flow, ActionFlow::Exit) { exit_after = true; break; }
                                }
                                let _ = execute!(std::io::stdout(), EnterAlternateScreen);
                                let _ = enable_raw_mode();
                                let _ = terminal.clear();
                                if exit_after {
                                    result = PickerOutcome::Done;
                                    break;
                                }
                                // Cache survey reflects every change
                                // the batch made.
                                cache_survey = survey_cache_state(&cache_dir);
                                all_rows = build_rows(entries, &cache_survey);
                                // Selection stays toggled — user can
                                // chain another batch action without
                                // re-selecting. Press Esc to clear.
                                continue;
                            }
                            if let Some(row) = vis.get(cursor) {
                                let action = PickerAction::menu_order()[menu_cursor];
                                let specifier = row.specifier();
                                menu_open = false;
                                // Picker-local actions never leave the
                                // chrome: they flip an overlay flag and
                                // skip the dispatch+chrome-suspend cycle.
                                if action.is_picker_local() {
                                    match action {
                                        PickerAction::Describe => {
                                            let entry = entries.iter().find(|e| e.name == row.dataset);
                                            descriptor_lines = build_descriptor_lines(row, entry, theme_on);
                                            descriptor_title = format!(
                                                "Descriptor — {}:{}", row.dataset, row.profile);
                                            descriptor_scroll = 0;
                                            descriptor_open = true;
                                        }
                                        PickerAction::Source => {
                                            let entry = entries.iter().find(|e| e.name == row.dataset);
                                            let (title, lines) = match entry {
                                                Some(e) => build_source_lines(e, theme_on),
                                                None => (
                                                    "Source — error".to_string(),
                                                    vec![Line::from(
                                                        Span::styled(" no catalog entry found".to_string(),
                                                            if theme_on { Style::default().fg(ACCENT_CORAL) }
                                                            else { Style::default() }),
                                                    )],
                                                ),
                                            };
                                            descriptor_lines = lines;
                                            descriptor_title = title;
                                            descriptor_scroll = 0;
                                            descriptor_open = true;
                                        }
                                        _ => {} // future picker-local actions land here
                                    }
                                    continue;
                                }
                                // Suspend our chrome so the action can
                                // own the terminal (visualizer enters
                                // its own raw mode; precache/ping/purge
                                // print to stderr). State across the
                                // suspension is naturally preserved —
                                // we never exit run_picker.
                                let _ = disable_raw_mode();
                                let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
                                // Single-target dispatch — always pause
                                // after so the user sees the output.
                                let flow = dispatch(&specifier, action, true);
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
                        // Order matters — multi-select state peels off
                        // first (most ephemeral), then UI overlays,
                        // then list shape, then filter, then exit.
                        if range_anchor.is_some() {
                            range_anchor = None;
                        } else if !selected.is_empty() {
                            selected.clear();
                        } else if show_details {
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
                        let cursor_in_selection = vis.get(cursor)
                            .map(|r| selected.contains(&r.specifier()))
                            .unwrap_or(false);
                        // Decision tree:
                        //   selected empty                 → single-target menu (cursor row)
                        //   selected non-empty + cursor in it → batch menu directly
                        //   selected non-empty + cursor outside → disambig prompt
                        //     (the user might mean either; ask explicitly)
                        if selected.is_empty() {
                            if let Some(row) = vis.get(cursor) {
                                let profile_count = all_rows.iter()
                                    .filter(|r| r.dataset == row.dataset)
                                    .count();
                                if !show_all_profiles && !expanded.contains(&row.dataset) && profile_count > 1 {
                                    expanded.insert(row.dataset.clone());
                                } else {
                                    menu_open = true;
                                    menu_is_batch = false;
                                    menu_cursor = 0;
                                }
                            }
                        } else if cursor_in_selection {
                            menu_open = true;
                            menu_is_batch = true;
                            menu_cursor = 0;
                        } else {
                            disambig_open = true;
                            disambig_cursor = 0;
                        }
                    }
                    KeyCode::Left => {
                        // Collapse: if current row's dataset is expanded, collapse it
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .collect();
                        if let Some(row) = vis.get(cursor)
                            && expanded.remove(&row.dataset) {
                                // Move cursor to the default row of this dataset
                                let new_vis: Vec<&PickerRow> = all_rows.iter()
                                    .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                                    .filter(|r| matches_filter(r, &filter))
                                    .collect();
                                cursor = new_vis.iter().position(|r| r.dataset == row.dataset).unwrap_or(0);
                            }
                    }
                    KeyCode::Right => {
                        // Expand: same as Enter for collapsed datasets
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .collect();
                        if let Some(row) = vis.get(cursor)
                            && !expanded.contains(&row.dataset) {
                                expanded.insert(row.dataset.clone());
                            }
                    }
                    KeyCode::Up => {
                        // Shift+↑ enters range mode (or extends it):
                        // capture the anchor on first press, then let
                        // the cursor move freely while keeping the
                        // REVERSED bar drawn over the whole span.
                        // Plain ↑ leaves range mode if it was active.
                        if key.modifiers.contains(KeyModifiers::SHIFT) {
                            if range_anchor.is_none() { range_anchor = Some(cursor); }
                        } else {
                            range_anchor = None;
                        }
                        cursor = cursor.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        if key.modifiers.contains(KeyModifiers::SHIFT) {
                            if range_anchor.is_none() { range_anchor = Some(cursor); }
                        } else {
                            range_anchor = None;
                        }
                        let visible_count = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .count();
                        if cursor + 1 < visible_count {
                            cursor += 1;
                        }
                    }
                    KeyCode::Char(' ') => {
                        // Toggle persistent multi-select membership.
                        // Range-mode behaviour: toggle every row in
                        // anchor..=cursor (commits the shift-arrow
                        // preview into the actual selected set) and
                        // exits range mode. Otherwise toggle just the
                        // cursor row.
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| matches_filter(r, &filter))
                            .collect();
                        match range_anchor {
                            Some(a) => {
                                let lo = a.min(cursor);
                                let hi = a.max(cursor);
                                for i in lo..=hi {
                                    if let Some(row) = vis.get(i) {
                                        let spec = row.specifier();
                                        if !selected.remove(&spec) {
                                            selected.insert(spec);
                                        }
                                    }
                                }
                                range_anchor = None;
                            }
                            None => {
                                if let Some(row) = vis.get(cursor) {
                                    let spec = row.specifier();
                                    if !selected.remove(&spec) {
                                        selected.insert(spec);
                                    }
                                }
                            }
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
