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
// Chrome colors come from the session theme — derived from the same
// configured (palette, curve) pair that drives data visualization.
// See `explore::theme()` / `palette::Theme::derive`. Role mapping:
//   primary  — highlight / local        success — merkle-hashed
//   info     — merkle-chunked           warning — GT-style
//   meta     — metadata family          error   — full-transfer
use super::theme;
// Selection uses `Modifier::REVERSED` rather than explicit fg/bg
// colours so the highlight bar is independent of palette state —
// works under the neon theme, the mono toggle, and broken-ANSI
// terminals alike.

/// Truncate a string to at most `target` DISPLAY columns (not
/// bytes), then pad to exactly `target`. Byte-counting formatters
/// (`format!("{:<w$}")` / `format!("{:.w$}")`) under-reserve columns
/// whenever content contains multi-byte glyphs like the tree-drawing
/// unicode (`▸`, `▾`, `└` — 3 bytes each, 1 column each) and let
/// long values overflow their cell. Every picker cell renders
/// through this, at exactly its column width — inter-column gaps
/// come from explicit separator spans, never from width headroom.
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
/// Optional picker columns, in display order. DATASET and PROFILE
/// are row identity and always shown; everything else can be hidden
/// via the settings screen (persisted as the `disabled_columns`
/// settings key, by `name()`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PickerColumn {
    Catalog,
    Facets,
    Metric,
    /// Base-vector record count (cardinality).
    Count,
    /// Storage bytes across the profile's facet files.
    Size,
    Access,
    Cached,
}

const ALL_COLUMNS: &[PickerColumn] = &[
    PickerColumn::Catalog,
    PickerColumn::Facets,
    PickerColumn::Metric,
    PickerColumn::Count,
    PickerColumn::Size,
    PickerColumn::Access,
    PickerColumn::Cached,
];

impl PickerColumn {
    /// Settings token (the `disabled_columns` entry).
    fn name(self) -> &'static str {
        match self {
            PickerColumn::Catalog => "catalog",
            PickerColumn::Facets  => "facets",
            PickerColumn::Metric  => "metric",
            PickerColumn::Count   => "count",
            PickerColumn::Size    => "size",
            PickerColumn::Access  => "access",
            PickerColumn::Cached  => "cached",
        }
    }
    /// Column header text. Catalog is deliberately terse — it sits
    /// left of DATASET and usually holds short names/indexes.
    fn header(self) -> &'static str {
        match self {
            PickerColumn::Catalog => "C",
            PickerColumn::Facets  => "FACETS",
            PickerColumn::Metric  => "METRIC",
            PickerColumn::Count   => "COUNT",
            PickerColumn::Size    => "SIZE",
            PickerColumn::Access  => "ACCESS",
            PickerColumn::Cached  => "CACHED",
        }
    }
}

/// One actionable row of the settings screen (Ctrl-G). Headers are
/// rendered between sections but are not items — the cursor only
/// lands on these.
#[derive(Clone, PartialEq, Eq, Debug)]
enum SettingsItem {
    /// Toggle one configured catalog by name.
    CatalogToggle(String),
    /// Open the inline editor to add a new catalog source.
    AddCatalog,
    /// Toggle one optional column.
    ColumnToggle(PickerColumn),
    /// Cycle the global palette.
    PaletteCycle,
    /// Cycle the global curve.
    CurveCycle,
    /// Persist palette/curve to settings.yaml.
    SaveTheme,
    /// Remove display-related settings keys; back to the standard.
    ResetDisplay,
}

/// Which field the inline "Add catalog" editor is currently capturing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AddField {
    /// The symbolic name the catalog will be saved under.
    Name,
    /// The catalog URL or local path.
    Url,
    /// An optional bearer token for a catalog whose endpoint needs auth.
    Token,
}

/// Transient state of the settings screen's "Add catalog" flow: a name,
/// a URL/path, then an optional auth token, each typed into a single-line
/// field. A catalog must be given a name (the [`AddField::Name`] stage
/// refuses to advance on an empty one), mirroring `config catalog add
/// --name`. When a token is supplied it is recorded (keyed by the
/// catalog URL, tagged with the name) *before* verification, so a
/// protected endpoint authenticates during the add.
#[derive(Clone, Debug)]
struct AddCatalogInput {
    /// The field keystrokes currently land in.
    field: AddField,
    /// The name buffer.
    name: String,
    /// The URL/path buffer.
    url: String,
    /// The optional token buffer (blank → add anonymously).
    token: String,
    /// When editing an existing catalog, the prior name to drop on submit
    /// (so a fix replaces in place); `None` for a fresh add.
    replacing: Option<String>,
}

impl AddCatalogInput {
    /// A fresh editor, parked on the name field.
    fn new() -> Self {
        AddCatalogInput {
            field: AddField::Name,
            name: String::new(),
            url: String::new(),
            token: String::new(),
            replacing: None,
        }
    }

    /// An editor pre-filled to fix an existing catalog in place.
    fn editing(name: String, url: String, token: String) -> Self {
        AddCatalogInput {
            field: AddField::Url,
            replacing: Some(name.clone()),
            name,
            url,
            token,
        }
    }
}

/// Transient state of the "Set auth" flow for an already-configured
/// catalog: a single masked token field. Submitting empty clears any
/// stored credential; submitting a token records one keyed by the
/// catalog's URL and tagged with its name.
#[derive(Clone, Debug)]
struct AuthInput {
    /// The catalog name (the credential's name-tag, and the prompt label).
    name: String,
    /// The catalog URL the credential is keyed against.
    url: String,
    /// The token buffer.
    token: String,
}

/// The faceted settings screen (Ctrl-G) is a tabbed view; each tab owns
/// one group of actionable items. Tab / Shift-Tab move between tabs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Catalogs,
    Columns,
    Theme,
    Maintenance,
}

impl SettingsTab {
    /// Tabs in display order.
    const ALL: [SettingsTab; 4] = [
        SettingsTab::Catalogs,
        SettingsTab::Columns,
        SettingsTab::Theme,
        SettingsTab::Maintenance,
    ];

    fn title(self) -> &'static str {
        match self {
            SettingsTab::Catalogs => "Catalogs",
            SettingsTab::Columns => "Columns",
            SettingsTab::Theme => "Theme",
            SettingsTab::Maintenance => "Maintenance",
        }
    }

    fn idx(self) -> usize {
        Self::ALL.iter().position(|&t| t == self).unwrap_or(0)
    }
    fn next(self) -> Self {
        Self::ALL[(self.idx() + 1) % Self::ALL.len()]
    }
    fn prev(self) -> Self {
        Self::ALL[(self.idx() + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    /// The actionable items shown on this tab, in order. The cursor only
    /// lands on these; the tab bar itself is navigated with Tab.
    fn items(self, catalog_list: &[(String, usize)]) -> Vec<SettingsItem> {
        match self {
            SettingsTab::Catalogs => {
                let mut v: Vec<SettingsItem> = catalog_list.iter()
                    .map(|(name, _)| SettingsItem::CatalogToggle(name.clone()))
                    .collect();
                v.push(SettingsItem::AddCatalog);
                v
            }
            SettingsTab::Columns =>
                ALL_COLUMNS.iter().map(|c| SettingsItem::ColumnToggle(*c)).collect(),
            SettingsTab::Theme => vec![
                SettingsItem::PaletteCycle,
                SettingsItem::CurveCycle,
                SettingsItem::SaveTheme,
            ],
            SettingsTab::Maintenance => vec![SettingsItem::ResetDisplay],
        }
    }
}

struct PickerRow {
    dataset: String,
    /// Symbolic name of the configured catalog this row came from.
    catalog: String,
    profile: String,
    /// Numeric base_count for sorting (0 for default profile).
    base_count: u64,
    metric: String,
    facets: String,
    /// COUNT cell: base-vector cardinality ("1M", "983K", with `*`
    /// marking profiles that share the default profile's base file).
    count: String,
    /// SIZE cell: storage bytes across the profile's facet files.
    /// Both cells are filled by [`apply_size_cells`] from the
    /// session's facet-knowledge map, not by `build_rows`.
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
    cache_survey: &std::collections::HashMap<std::path::PathBuf, FacetCacheView>,
) -> Vec<PickerRow> {
    let cache_dir = crate::explore::cache_dir_or_exit();
    let mut rows = Vec::new();
    for entry in entries {
        // Declared metric first (canonical dataset.yaml attributes);
        // knn_entries-shape catalogs declare none, but the jvector
        // SimpleMFD ecosystem encodes the metric in ground-truth
        // filenames (`..._gt_..._ip_k100.ivecs`) — recover it from
        // there rather than showing an empty column.
        let metric = entry.layout.attributes.as_ref()
            .and_then(|a| a.distance_function.as_deref())
            .map(|s| s.to_string())
            .or_else(|| infer_metric_from_entry(entry))
            .unwrap_or_default();

        let ds_cache = cache_dir.join(&entry.name);

        let catalog_name = entry.catalog_name.clone()
            .unwrap_or_else(|| "?".to_string());
        for profile_name in entry.profile_names() {
            let profile = entry.layout.profiles.profile(profile_name);
            let bc = crate::dataset::profile::profile_sort_key(
                profile_name, profile.and_then(|p| p.base_count));

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

            let access = profile_access_mode(entry, &ds_cache, profile, cache_survey);

            rows.push(PickerRow {
                dataset: entry.name.clone(),
                catalog: catalog_name.clone(),
                profile: profile_name.to_string(),
                base_count: bc,
                metric: metric.clone(),
                facets,
                count: String::new(),
                size: String::new(),
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
    survey: &std::collections::HashMap<std::path::PathBuf, FacetCacheView>,
) -> (u32, u32) {
    // Same home URL the storage open anchors the cache layout on —
    // single authority, so the picker's expectation of where a facet
    // caches cannot drift from where the runtime actually puts it.
    let home = {
        let h = entry.dataset_home_url();
        if h.ends_with('/') { h } else { format!("{h}/") }
    };
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
        // sidecar for true chunk-level coverage). The expected cache
        // path is derived FORWARD — facet URL + home →
        // `facet_cache_relpath`, the same function the storage open
        // uses — and looked up by path. (The previous version
        // reconstructed a URL from the on-disk relpath and matched
        // URLs; that guess broke for any facet whose cache relpath
        // isn't home-relative, e.g. out-of-home facets cached flat
        // by basename.) A bare `file.exists()` check can't replace
        // the survey: Storage pre-allocates sparse files before any
        // chunk lands, so existence alone would show `100% (1/1)`
        // the moment a facet is first opened.
        if let Some(url) = resolve_facet_url(entry, view) {
            let relpath = crate::view::facet_cache_relpath(&url, &home);
            let expected = ds_workspace.join(relpath);
            if let Some(cv) = survey.get(&expected) {
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

/// One-line cardinality + byte summary for a facet, from the
/// session's knowledge map: `"983K records, 1.5G"`. Windowed views
/// that could only be costed whole-file say so. `None` when nothing
/// is known (and no probe is in flight) — the line is omitted
/// rather than shown empty.
fn facet_stat_string(
    entry: &CatalogEntry,
    view: &crate::dataset::profile::DSView,
    knowledge: &std::collections::HashMap<String, FacetKnowledge>,
) -> Option<String> {
    let shape = facet_shape(entry, view)?;
    let res = resolve_facet(&shape, knowledge);
    let mut parts: Vec<String> = Vec::new();
    if let Some(n) = res.records {
        parts.push(format!("{} records", format_count(n)));
    }
    if let Some(b) = res.bytes {
        let suffix = if res.whole_file_for_window { " (whole file)" } else { "" };
        parts.push(format!("{}{suffix}", crate::explore::format_bytes_short(b)));
    }
    if parts.is_empty() {
        return res.pending.then(|| "resolving…".to_string());
    }
    Some(parts.join(", "))
}

/// Compose the full descriptor for the scrollable Describe overlay — the
/// single dataset-inspect view. Covers every view's path, namespace (when
/// set), explicit window range, and cardinality / byte size when known; the
/// origin URL pulled from the catalog entry; access mode rationale; per-row
/// cache state.
fn build_descriptor_lines(
    row: &PickerRow,
    entry: Option<&CatalogEntry>,
    knowledge: &std::collections::HashMap<String, FacetKnowledge>,
    colors_on: bool,
) -> Vec<Line<'static>> {
    let tint = |c: Color| if colors_on { Style::default().fg(c) } else { Style::default() };
    let dim_gray = tint(theme().text_muted);
    let val_white = tint(theme().text_primary);
    let head_cyan = tint(theme().primary);

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
            if let Some(stat) = facet_stat_string(entry, view, knowledge) {
                lines.push(kv_indent(2, "shape", &stat, dim_gray, val_white));
            }
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
        AccessMode::Local         => theme().primary,
        AccessMode::MerkleHashed  => theme().success,
        AccessMode::MerkleChunked => theme().info,
        AccessMode::FullTransfer  => theme().error,
    };
    lines.push(kv("access", row.access.short_label(), dim_gray, tint(access_color)));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("{:<12}", ""), dim_gray),
        Span::raw(" "),
        Span::styled(row.access.description().to_string(), dim_gray),
    ]));
    lines.push(kv("cache", &row.cache_status, dim_gray, val_white));
    lines.push(kv("count", &row.count, dim_gray, val_white));
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
    let body_style = tint(theme().text_primary);
    let error_style = tint(theme().error);

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
                    Line::from(Span::styled(format!(" {source_url}"), tint(theme().text_muted))),
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
        Line::from(Span::styled(format!(" {source_url}"), tint(theme().text_muted))),
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
/// path — see [`crate::dataset::catalog::strip_window_suffix`], the
/// single authority this delegates to. The picker compares stripped
/// paths to detect when a profile is a window into the dataset's
/// default base data rather than its own download.
fn strip_window_suffix(source_path: &str) -> &str {
    crate::dataset::catalog::strip_window_suffix(source_path)
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
    -> std::collections::HashMap<std::path::PathBuf, FacetCacheView>
{
    let mut map: std::collections::HashMap<std::path::PathBuf, FacetCacheView> =
        std::collections::HashMap::new();
    walk_dataset_dirs(cache_dir, &mut |dataset_dir| {
        // origin.json marks a real dataset directory; the survey is
        // keyed by cache-file path, so the origin's URL itself isn't
        // needed here — the coverage lookup derives each facet's
        // expected cache path forward, with the same function the
        // storage open uses, and matches on the path.
        if crate::cache::layout::read_dataset_origin(dataset_dir).is_none() { return; }
        walk_sidecars(dataset_dir, &mut map);
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
/// every file and record per-chunk cache state, keyed by the data
/// file's absolute cache path (the sidecar path with `.mrkl` /
/// `.chunks` trimmed). Path-keyed on purpose: reconstructing the
/// facet URL from `<origin><relpath>` breaks the moment the relpath
/// isn't home-relative (out-of-home facets cache flat by basename),
/// whereas the forward derivation used by the coverage lookup is the
/// same one the storage open uses — they can't disagree.
fn walk_sidecars(
    current: &std::path::Path,
    map: &mut std::collections::HashMap<std::path::PathBuf, FacetCacheView>,
) {
    let Ok(entries) = std::fs::read_dir(current) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_sidecars(&path, map);
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
        map.insert(cache_file_path, view);
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

/// Purge a dataset's cache. Two sweeps:
///
/// 1. **The dataset-keyed directory** — `<cache_root>/<entry.name>/`
///    is where the mandated layout puts everything the dataset
///    pulled, including out-of-home facets cached flat by basename
///    (whose URLs are NOT under the recorded origin, which is why
///    the old facet-URL-prefix match silently removed nothing for
///    them). Guarded by the origin binding: when `origin.json`
///    names a different home than this entry's, the directory
///    belongs to a same-named dataset from another catalog and is
///    reported as skipped instead of clobbered.
/// 2. **URL-derived locations** — any other origin-bearing directory
///    whose recorded origin is a URL-prefix of one of this entry's
///    facet URLs (direct-URL-open caches of the same files).
///
/// Returns `(removed_paths, freed_bytes, skipped_notes)`. A removal
/// failure (permission, missing dir) is silently skipped so a single
/// bad entry doesn't abort the whole purge.
pub(super) fn purge_cache_for_entry(
    entry: &CatalogEntry,
    cache_dir: &std::path::Path,
) -> (Vec<std::path::PathBuf>, u64, Vec<String>) {
    let mut removed = Vec::new();
    let mut freed: u64 = 0;
    let mut skipped = Vec::new();
    let remove_dir = |dir: &std::path::Path, removed: &mut Vec<std::path::PathBuf>, freed: &mut u64| {
        let bytes = leaf_size_bytes(dir);
        if std::fs::remove_dir_all(dir).is_ok() {
            *freed += bytes;
            removed.push(dir.to_path_buf());
        }
    };

    // Sweep 1: the dataset-keyed directory.
    let name_dir = cache_dir.join(&entry.name);
    if name_dir.is_dir() {
        let home = canonicalize_cache_url(&entry.dataset_home_url());
        match crate::cache::layout::read_dataset_origin(&name_dir) {
            Some(origin) if canonicalize_cache_url(&origin.source) != home => {
                skipped.push(format!(
                    "{} — origin.json records {} (a different catalog's dataset \
                     with the same name); not removed. Delete it manually if \
                     that's really what you want.",
                    name_dir.display(), origin.source,
                ));
            }
            // Matching origin, or no origin recorded (derive-style
            // workspace) — the directory is this dataset's by layout.
            _ => remove_dir(&name_dir, &mut removed, &mut freed),
        }
    }

    // Sweep 2: URL-derived caches of the same files (origin is the
    // facet's parent URL → prefix-matches the facet URL). Both sides
    // canonicalised so an s3:// origin and an https:// facet URL
    // compare in the same scheme.
    let facet_urls = entry_facet_urls(entry);
    walk_dataset_dirs(cache_dir, &mut |dataset_dir| {
        if removed.iter().any(|r| dataset_dir.starts_with(r)) { return; }
        let Some(origin) = crate::cache::layout::read_dataset_origin(dataset_dir) else { return; };
        let base_url = canonicalize_cache_url(&origin.source);
        let matches = facet_urls.iter().any(|u| base_url_matches_facet(&base_url, u));
        if !matches { return; }
        remove_dir(dataset_dir, &mut removed, &mut freed);
    });
    (removed, freed, skipped)
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
    entry.resolve_facet_url(&view.source.path)
}

/// Classify a profile by its `base_vectors` source — the file the
/// explore TUI opens first. The picker's source-of-truth column is
/// based on this facet because everything else (queries, GT) gets
/// fetched only when the user runs a KNN sample, and the dominant
/// latency-shaping question is "how does base_vectors arrive?".
///
/// Returns [`AccessMode::FullTransfer`] when the profile has no
/// `base_vectors` declared — there's nothing to predict an access
/// mode against. Returns [`AccessMode::Local`] when the facet is
/// already fully on disk: a complete sidecar-less copy at the
/// forward-derived cache path (URL + home → `facet_cache_relpath`,
/// the same single authority the storage open and the coverage
/// survey use — joining the raw source path broke for knn_entries
/// catalogs whose source paths are absolute URLs, leaving fully
/// downloaded datasets labelled "chunked"), a chunk-store file whose
/// survey bitmap is fully valid, or the legacy workspace layout's
/// materialised file. Otherwise delegates to
/// [`AccessMode::classify_remote`] for the remote uniform-vs-vvec /
/// `.mrkl`-or-not branches.
fn profile_access_mode(
    entry: &CatalogEntry,
    ds_cache: &std::path::Path,
    profile: Option<&crate::dataset::profile::DSProfile>,
    survey: &std::collections::HashMap<std::path::PathBuf, FacetCacheView>,
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

    if let Some(url) = resolve_facet_url(entry, view) {
        let home = {
            let h = entry.dataset_home_url();
            if h.ends_with('/') { h } else { format!("{h}/") }
        };
        let expected = ds_cache.join(crate::view::facet_cache_relpath(&url, &home));
        match survey.get(&expected) {
            // Chunk-store file: fully valid bitmap means every read
            // is served from disk — no download left to do.
            Some(cv) => {
                if cv.total_chunks > 0 && cv.whole_file_valid() == cv.total_chunks {
                    return AccessMode::Local;
                }
            }
            // No sidecar: a plain file at the derived path is a
            // complete copy (only sparse chunk-store files misreport
            // completeness, and those always carry a sidecar).
            None => {
                if expected.is_file() {
                    return AccessMode::Local;
                }
            }
        }
    }

    // Legacy workspace layout (`derive` output): source-relative
    // file materialised under the dataset directory.
    if ds_cache.join(clean).exists() {
        return AccessMode::Local;
    }
    AccessMode::classify_remote(clean, ds_cache)
}

/// Build a compact facet indicator string from a profile's view keys.
///
/// Letters: B(base) Q(query) G(ground-truth indices) D(distances)
///          M(metadata) P(predicates) R(results) F(filtered-knn)
/// Recover the distance metric from facet filenames when the catalog
/// declares none. The jvector SimpleMFD convention encodes it as an
/// underscore-delimited token in the ground-truth (and sometimes
/// base) filename: `_ip_`, `_l2_`, `_cos`/`_cosine`, `_angular`,
/// `_dot`. Returns the canonical lowercase token.
fn infer_metric_from_entry(entry: &CatalogEntry) -> Option<String> {
    for (_pname, profile) in &entry.layout.profiles.profiles {
        for (_facet, view) in &profile.views {
            if let Some(m) = infer_metric_from_filename(&view.source.path) {
                return Some(m);
            }
        }
    }
    None
}

/// Token scan of one filename. Pure; see [`infer_metric_from_entry`].
fn infer_metric_from_filename(path: &str) -> Option<String> {
    let name = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    for t in name.split(['_', '.', '-']) {
        match t {
            "ip" | "dot" | "dotproduct" => return Some("ip".to_string()),
            "l2" | "euclidean" => return Some("l2".to_string()),
            "cos" | "cosine" | "angular" => return Some("cosine".to_string()),
            _ => {}
        }
    }
    None
}

/// Marker shown in a COUNT/SIZE cell while a background remote
/// probe that could still improve it is in flight.
const SIZE_PENDING: &str = "…";

/// What the session knows about one facet file, keyed by its
/// resolved source URL. Seeded from local sources (the cache
/// survey, complete local copies) by [`collect_local_knowledge`]
/// and completed by background remote probes. Lives across row
/// rebuilds so an answer is never fetched twice.
#[derive(Clone, Copy, Default)]
struct FacetKnowledge {
    /// Total file size in bytes.
    bytes: Option<u64>,
    /// Bytes per record (`4 + dim * elem_size`, uniform xvec only).
    bpr: Option<u64>,
    /// A remote probe is queued or in flight.
    pending: bool,
    /// A remote probe answered (even with nothing) — the facet is
    /// settled; whatever is still `None` is unknowable this session.
    probed: bool,
}

/// Static per-facet shape derived from the catalog entry alone:
/// resolved URL, element size, uniformity, window. The bridge
/// between a profile's views and the knowledge map.
struct FacetShape {
    url: String,
    elem_size: usize,
    /// Uniform-stride xvec — record counts derivable from byte math.
    uniform: bool,
    /// `[start..end)` record window, when the view declares one.
    window: Option<(u64, u64)>,
}

/// Resolve one view to its [`FacetShape`]. `None` for facets that
/// aren't vector data (sidecar YAMLs etc. — same skip rule as cache
/// coverage) or whose URL can't be resolved.
fn facet_shape(entry: &CatalogEntry, view: &crate::dataset::profile::DSView) -> Option<FacetShape> {
    let clean = strip_window_suffix(&view.source.path);
    let ext = clean.rsplit('.').next().unwrap_or("");
    let elem_size = crate::io::infer_elem_size(ext);
    if elem_size == 0 { return None; }
    let url = resolve_facet_url(entry, view)?;
    let window = view.effective_window().0.first()
        .map(|iv| (iv.min_incl, iv.max_excl))
        .or_else(|| parse_path_suffix_window(&view.source.path))
        .filter(|(s, e)| e > s);
    Some(FacetShape {
        url,
        elem_size,
        uniform: !crate::io::is_vvec_ext(ext),
        window,
    })
}

/// Seed the knowledge map from everything answerable without the
/// network, for every facet of every profile:
///   - the cache survey: sparse chunk-store files carry the full
///     remote size in their sidecar, and `bpr` when chunk 0 (the
///     xvec dim header) is resident — a ping leaves it there;
///   - a complete local copy with no chunk sidecar (a finished
///     download, or a locally-derived workspace): a plain file's
///     length IS the whole size — only sparse chunk-store files
///     misreport it, and those always carry the sidecar above.
///
/// Re-run after cache re-surveys to pick up facts the cache didn't
/// hold before (e.g. a download landing the dim header). Facets
/// already fully answered are skipped — local and remote answers
/// describe the same file, so there is nothing to upgrade.
fn collect_local_knowledge(
    entries: &[CatalogEntry],
    cache_dir: &std::path::Path,
    survey: &std::collections::HashMap<std::path::PathBuf, FacetCacheView>,
    knowledge: &mut std::collections::HashMap<String, FacetKnowledge>,
) {
    for entry in entries {
        let ds_workspace = cache_dir.join(&entry.name);
        let home = {
            let h = entry.dataset_home_url();
            if h.ends_with('/') { h } else { format!("{h}/") }
        };
        for pname in entry.profile_names() {
            let Some(profile) = entry.layout.profiles.profile(pname) else { continue };
            for (_facet, view) in &profile.views {
                let Some(shape) = facet_shape(entry, view) else { continue };
                let known = knowledge.get(&shape.url).copied().unwrap_or_default();
                if known.bytes.is_some() && (!shape.uniform || known.bpr.is_some()) {
                    continue; // already fully answered
                }
                let relpath = crate::view::facet_cache_relpath(&shape.url, &home);
                let (bytes, header_file) = if let Some(cv) = survey.get(&ds_workspace.join(relpath)) {
                    (Some(cv.total_size), Some(cv.cache_file_path.clone()))
                } else {
                    let clean = strip_window_suffix(&view.source.path);
                    [ds_workspace.join(relpath), ds_workspace.join(clean)].into_iter()
                        .find_map(|p| std::fs::metadata(&p).ok()
                            .filter(|m| m.is_file())
                            .map(|m| (Some(m.len()), Some(p))))
                        .unwrap_or((None, None))
                };
                if bytes.is_none() { continue; }
                let bpr = header_file
                    .filter(|_| shape.uniform)
                    .and_then(|p| read_xvec_bpr(&p, shape.elem_size));
                let slot = knowledge.entry(shape.url).or_default();
                slot.bytes = bytes.filter(|&b| b > 0).or(slot.bytes);
                slot.bpr = bpr.or(slot.bpr);
            }
        }
    }
}

/// One remote lookup: a facet URL whose bytes (or bytes-per-record)
/// the local pass couldn't answer.
struct SizeProbeJob {
    url: String,
    elem_size: usize,
    uniform: bool,
}

/// Answer for one [`SizeProbeJob`]; merged into the knowledge map.
struct SizeProbeResult {
    url: String,
    bytes: Option<u64>,
    bpr: Option<u64>,
}

/// Collect one probe job per distinct remote facet URL the local
/// pass left incompletely answered, marking each as pending in the
/// knowledge map. No jobs → no threads spawned.
fn collect_size_probe_jobs(
    entries: &[CatalogEntry],
    knowledge: &mut std::collections::HashMap<String, FacetKnowledge>,
) -> Vec<SizeProbeJob> {
    let mut jobs: Vec<SizeProbeJob> = Vec::new();
    for entry in entries {
        for pname in entry.profile_names() {
            let Some(profile) = entry.layout.profiles.profile(pname) else { continue };
            for (_facet, view) in &profile.views {
                let Some(shape) = facet_shape(entry, view) else { continue };
                if !crate::transport::is_remote_url(&shape.url) { continue; }
                let slot = knowledge.entry(shape.url.clone()).or_default();
                if slot.pending || slot.probed { continue; }
                if slot.bytes.is_some() && (!shape.uniform || slot.bpr.is_some()) { continue; }
                slot.pending = true;
                jobs.push(SizeProbeJob {
                    url: shape.url,
                    elem_size: shape.elem_size,
                    uniform: shape.uniform,
                });
            }
        }
    }
    jobs
}

/// Run `jobs` on a small worker pool, sending each answer over the
/// returned channel as it lands. Workers are detached: they stop at
/// the next job boundary once `cancel` is set or the receiver is
/// dropped. Probes are read-only — an authenticated HEAD for the
/// byte size plus (for uniform xvecs on range-capable servers) a
/// 4-byte ranged GET for the dim header. Nothing touches the cache
/// directory, so an untouched dataset stays untouched on disk.
fn spawn_size_probes(
    jobs: Vec<SizeProbeJob>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> std::sync::mpsc::Receiver<SizeProbeResult> {
    use std::sync::atomic::Ordering;
    let (tx, rx) = std::sync::mpsc::channel();
    let queue = std::sync::Arc::new(std::sync::Mutex::new(
        std::collections::VecDeque::from(jobs)));
    let workers = crate::cache::download_concurrency()
        .min(queue.lock().map(|q| q.len()).unwrap_or(0));
    for _ in 0..workers {
        let queue = queue.clone();
        let tx = tx.clone();
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            while !cancel.load(Ordering::Relaxed) {
                let job = match queue.lock() {
                    Ok(mut q) => q.pop_front(),
                    Err(_) => None,
                };
                let Some(job) = job else { break };
                let (bytes, bpr) = probe_facet(&job);
                let result = SizeProbeResult { url: job.url, bytes, bpr };
                if tx.send(result).is_err() { break; }
            }
        });
    }
    rx
}

/// One remote probe: HEAD for `Content-Length` (with read auth and
/// S3 wrong-region handling via `HttpTransport`), then — when the
/// facet is a uniform xvec on a range-capable server — a 4-byte
/// ranged GET of the dim header for bytes-per-record, the same
/// `4 + dim * elem` math ping and the cache survey use.
fn probe_facet(job: &SizeProbeJob) -> (Option<u64>, Option<u64>) {
    use crate::transport::ChunkedTransport;
    let normalized = crate::transport::normalize_remote_url(&job.url);
    let Some(parsed) = url::Url::parse(normalized.as_ref()).ok() else {
        return (None, None);
    };
    let transport = crate::transport::HttpTransport::new(parsed);
    let Ok(total) = transport.content_length() else { return (None, None) };
    let mut bpr = None;
    if job.uniform && transport.supports_range()
        && let Ok(hdr) = transport.fetch_range(0, 4)
            && hdr.len() == 4 {
                let dim = i32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
                if dim > 0 && dim <= 1_000_000 {
                    bpr = Some(4 + dim as u64 * job.elem_size as u64);
                }
            }
    ((total > 0).then_some(total), bpr)
}

/// Resolved cardinality and byte weight for one facet, merged from
/// its [`FacetShape`] and the knowledge map. Single authority for
/// the window math — the COUNT/SIZE cells and the details overlays
/// all read from here.
struct FacetResolution {
    /// Record count: the window length when windowed, else
    /// `bytes / bpr` when both are known.
    records: Option<u64>,
    /// Bytes this facet contributes to its profile: the window's
    /// byte span when it can be computed, else the whole file.
    bytes: Option<u64>,
    /// `bytes` covers the whole file although the view is a window
    /// into it (bpr unknown, so the span couldn't be computed).
    whole_file_for_window: bool,
    /// A probe that could still improve this facet is in flight.
    pending: bool,
}

/// Merge one facet's static shape with the session's knowledge.
fn resolve_facet(
    shape: &FacetShape,
    knowledge: &std::collections::HashMap<String, FacetKnowledge>,
) -> FacetResolution {
    let known = knowledge.get(&shape.url).copied().unwrap_or_default();
    let derived_records = match (known.bytes, known.bpr) {
        (Some(bytes), Some(bpr)) if bpr > 0 => Some(bytes / bpr),
        _ => None,
    };
    match shape.window {
        Some((start, end)) => {
            let span = known.bpr.map(|bpr| {
                let span = (end - start).saturating_mul(bpr);
                match known.bytes {
                    Some(total) => span.min(total),
                    None => span,
                }
            });
            FacetResolution {
                records: Some(end - start),
                bytes: span.or(known.bytes),
                whole_file_for_window: span.is_none() && known.bytes.is_some(),
                pending: known.pending && span.is_none(),
            }
        }
        None => FacetResolution {
            records: derived_records,
            bytes: known.bytes,
            whole_file_for_window: false,
            pending: known.pending
                && (known.bytes.is_none() || (shape.uniform && known.bpr.is_none())),
        },
    }
}

/// Fill every row's COUNT and SIZE cells from the catalog entries
/// and the knowledge map. Called after each row rebuild and after
/// each batch of probe answers — cells are recomputed wholesale, so
/// the function is the single authority on what the two columns
/// mean:
///   COUNT — base-vector cardinality: the declared `base_count`,
///           else the base view's window length, else
///           `bytes / bpr`; `*` marks profiles sharing the default
///           profile's base file (no extra download).
///   SIZE  — storage bytes summed over the profile's distinct facet
///           files, window-scaled when the math is derivable; `…`
///           while a probe is still in flight, a `+` suffix when
///           some facet stayed unknowable (sum is a floor), `*`
///           when a shared base file is counted whole.
fn apply_size_cells(
    rows: &mut [PickerRow],
    entries: &[CatalogEntry],
    knowledge: &std::collections::HashMap<String, FacetKnowledge>,
) {
    use crate::dataset::StandardFacet;
    for row in rows.iter_mut() {
        let Some(entry) = entries.iter().find(|e| e.name == row.dataset) else { continue };
        let Some(profile) = entry.layout.profiles.profile(&row.profile) else { continue };

        let default_base_path: Option<String> = entry.layout.profiles
            .profile("default")
            .and_then(|p| p.views.get(StandardFacet::BaseVectors.key()))
            .map(|v| strip_window_suffix(&v.source.path).to_string());
        let shares_default_base = row.profile != "default"
            && !profile.partition
            && default_base_path.is_some()
            && profile.views.get(StandardFacet::BaseVectors.key())
                .map(|v| strip_window_suffix(&v.source.path).to_string())
                == default_base_path;

        let base_shape = profile.views.get(StandardFacet::BaseVectors.key())
            .and_then(|v| facet_shape(entry, v));
        let base_res = base_shape.as_ref().map(|s| resolve_facet(s, knowledge));

        row.count = match profile.base_count {
            Some(c) if c > 0 => {
                let s = format_count(c);
                if shares_default_base { format!("{s}*") } else { s }
            }
            _ => match &base_res {
                Some(r) => match r.records {
                    Some(n) if n > 0 => {
                        let s = format_count(n);
                        if shares_default_base { format!("{s}*") } else { s }
                    }
                    _ if r.pending => SIZE_PENDING.to_string(),
                    _ => String::new(),
                },
                None => String::new(),
            },
        };

        // SIZE: sum each distinct facet FILE once (keyed by URL).
        // Two views of the same file with different windows are
        // still one download — count the larger contribution.
        let mut urls: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        let mut pending = false;
        let mut unknown = false;
        for (_facet, view) in &profile.views {
            let Some(shape) = facet_shape(entry, view) else { continue };
            let res = resolve_facet(&shape, knowledge);
            pending |= res.pending;
            match res.bytes {
                Some(b) => {
                    let slot = urls.entry(shape.url).or_default();
                    *slot = (*slot).max(b);
                }
                None => unknown = true,
            }
        }
        // `*`: the sum counts the default profile's base file in
        // full although this profile only shares (or windows into)
        // it — downloading the default profile covers it.
        let shared_whole_base = shares_default_base
            && base_shape.as_ref().zip(base_res.as_ref()).is_some_and(|(s, r)|
                r.bytes.is_some() && (s.window.is_none() || r.whole_file_for_window));
        let total: u64 = urls.values().sum();
        row.size = if pending {
            SIZE_PENDING.to_string()
        } else if total == 0 {
            String::new()
        } else {
            let mut s = crate::explore::format_bytes_short(total);
            if unknown { s.push('+'); }
            if shared_whole_base { s.push('*'); }
            s
        };
    }
}

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

/// Row visibility: the text/facet filter AND the catalog toggle
/// screen's enabled set. Single predicate so every visible-rows
/// recomputation agrees.
fn row_visible(
    row: &PickerRow,
    filter: &str,
    disabled_catalogs: &std::collections::HashSet<String>,
) -> bool {
    !disabled_catalogs.contains(&row.catalog) && matches_filter(row, filter)
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
    /// Print the dataset's cache location (commented header + path,
    /// or a commented not-cached note), then return to the picker.
    /// Batch-safe: selecting several datasets prints one block each.
    Locate,
    /// Open a scrollable text view of the full catalog descriptor for this
    /// dataset+profile: attributes, profile (maxk/base_count), facets with
    /// their paths, windows, and per-facet dims/counts/byte-sizes, origin
    /// URLs, access mode, and cache state. The single "inspect this dataset"
    /// view. Renders inside the picker as an overlay — does not exit it.
    Describe,
    /// Open a scrollable text view of the raw catalog YAML — the
    /// `dataset.yaml` file for canonical catalogs, or the relevant
    /// entries pulled out of `knn_entries.yaml` for the legacy
    /// shape. Picker-local like Describe.
    Source,
    /// Download every facet's bytes into the local cache directory.
    /// (Drives the `datasets precache` machinery; the menu label is
    /// "Download" because that's what the user experiences.)
    Download,
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
            PickerAction::Locate    => "Locate",
            PickerAction::Describe  => "Describe",
            PickerAction::Source    => "Source",
            PickerAction::Download  => "Download",
            PickerAction::Purge     => "Purge",
            PickerAction::Ping      => "Ping",
        }
    }

    fn description(self) -> &'static str {
        match self {
            PickerAction::Visualize =>
                "Open the interactive explorer: norms, distances, eigenvalues, PCA.",
            PickerAction::Locate =>
                "Print the dataset's cache directory (commented note when not cached), then return to the picker. Works on a multi-selection.",
            PickerAction::Describe =>
                "Show full descriptor: attributes, facets (paths, windows, dims/counts/sizes), origin, access + cache state. Scrollable.",
            PickerAction::Source =>
                "Show the raw catalog YAML — dataset.yaml verbatim, or the relevant entries from knn_entries.yaml. Scrollable.",
            PickerAction::Download =>
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
            PickerAction::Locate,
            PickerAction::Download,
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
    /// for one row at a time. The rest — locate, download, ping,
    /// purge — chain naturally across a batch.
    fn batch_safe(self) -> bool {
        matches!(self,
            PickerAction::Locate
            | PickerAction::Download
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
    /// The catalog set changed (a source was added or removed in the
    /// settings editor). The caller should re-run the picker so the
    /// dataset list, counts, and probes rebuild against the new
    /// configuration. Distinct from [`Self::Done`] so a reload never
    /// looks like the user closing the picker.
    Reload,
}

/// Window a single-line input field's value to `width` display columns. The
/// active field anchors to the END (so the caret — where you're typing — stays
/// visible as a long URL scrolls left); other fields anchor to the start. The
/// returned `u16` is the caret's column offset from the field's value start
/// (only meaningful for the active field).
fn field_window(value: &str, width: u16, anchor_end: bool) -> (String, u16) {
    let w = width as usize;
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= w {
        return (value.to_string(), chars.len() as u16);
    }
    if anchor_end {
        (chars[chars.len() - w..].iter().collect(), w as u16)
    } else {
        (chars[..w].iter().collect(), w as u16)
    }
}

/// Run `body` with the picker's chrome suspended — so it can print freely
/// to the real console (verification progress, prompts, the persistent
/// failure message) — then pause for Enter and restore the chrome.
fn with_console<F: FnOnce()>(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>, body: F) {
    use std::io::Write as _;
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    body();
    println!();
    print!("Press Enter to return to the explorer… ");
    let _ = std::io::stdout().flush();
    let mut _line = String::new();
    let _ = std::io::stdin().read_line(&mut _line);
    let _ = execute!(std::io::stdout(), EnterAlternateScreen);
    let _ = enable_raw_mode();
    let _ = terminal.clear();
}

/// A console y/N prompt (default No). Used on the suspended-chrome path.
fn prompt_yes_no(question: &str) -> bool {
    use std::io::Write as _;
    print!("{question} ");
    let _ = std::io::stdout().flush();
    let mut ans = String::new();
    let _ = std::io::stdin().read_line(&mut ans);
    matches!(ans.trim(), "y" | "Y" | "yes" | "YES")
}

/// Add (or edit) a catalog, on the console (chrome already suspended).
///
/// A named write needs the name-based (`map`) form of `catalogs.yaml`; if
/// the file is the legacy list form the user is prompted to convert it
/// (recommended) and may decline (aborting). On success the catalog is
/// recorded active and its disabled/unvalidated flags cleared; on a
/// verification failure it is saved **disabled** + flagged *unvalidated*
/// (so a typo isn't lost and can be fixed in place). `replacing` names a
/// prior entry to drop first (edit); a non-empty `token` records a
/// credential before verifying so a protected endpoint authenticates.
#[allow(clippy::too_many_arguments)]
fn add_or_edit_console(
    name: &str,
    url: &str,
    token: &str,
    replacing: Option<&str>,
    catalog_locations: &std::collections::HashMap<String, String>,
    disabled: &mut std::collections::HashSet<String>,
    unvalidated: &mut std::collections::HashSet<String>,
) {
    // Format gate: keeping a name requires the map form.
    match crate::config::catalogs_format() {
        crate::config::CatalogsFormat::List => {
            println!("Your catalogs.yaml is list-formatted, which can't store catalog names.");
            println!("To keep the name '{name}' it must be rewritten to name-based form (recommended).");
            if !prompt_yes_no("Convert catalogs.yaml now? [y/N]") {
                println!("Cancelled — catalogs.yaml left unchanged; nothing was added.");
                return;
            }
            match crate::config::convert_catalogs_to_map() {
                Ok(n) => println!("Converted {n} catalog entr{} to name-based form.",
                    if n == 1 { "y" } else { "ies" }),
                Err(e) => { eprintln!("Conversion failed: {e} — aborting."); return; }
            }
        }
        crate::config::CatalogsFormat::Mixed => {
            eprintln!("catalogs.yaml mixes list and map entries — fix it manually first. Aborting.");
            return;
        }
        crate::config::CatalogsFormat::Map | crate::config::CatalogsFormat::Empty => {}
    }

    // Editing: drop the prior entry + its flags first so a successful
    // re-add lands clean (and a name change doesn't dup).
    if let Some(old) = replacing {
        let target = catalog_locations.get(old).cloned().unwrap_or_else(|| old.to_string());
        let _ = crate::config::try_remove_catalog(&target);
        if disabled.remove(old) {
            let _ = crate::settings::write_disabled_catalogs(disabled);
        }
        if unvalidated.remove(old) {
            let _ = crate::settings::write_unvalidated_catalogs(unvalidated);
        }
    }

    if !token.is_empty()
        && let Err(e) = crate::credentials::set_catalog_credential(name, url, token, None)
    {
        eprintln!("auth save failed: {e}");
    }

    println!("Adding catalog '{name}' — {url}");
    match crate::config::try_add_catalog(url, Some(name)) {
        Ok(crate::config::AddCatalogOutcome::Added { count, .. }) => {
            if unvalidated.remove(name) {
                let _ = crate::settings::write_unvalidated_catalogs(unvalidated);
            }
            if disabled.remove(name) {
                let _ = crate::settings::write_disabled_catalogs(disabled);
            }
            println!("OK — {count} dataset(s). Saved and enabled.");
        }
        Ok(crate::config::AddCatalogOutcome::AlreadyConfigured) => {
            println!("Already configured: {url}");
        }
        Err(reason) => {
            eprintln!("Verification failed: {reason}");
            match crate::config::force_add_catalog(name, url) {
                Ok(_) => {
                    disabled.insert(name.to_string());
                    let _ = crate::settings::write_disabled_catalogs(disabled);
                    unvalidated.insert(name.to_string());
                    let _ = crate::settings::write_unvalidated_catalogs(unvalidated);
                    eprintln!("Saved '{name}' but left it DISABLED until it validates.");
                    eprintln!("Fix the URL with 'e' (edit) in the config view, or press Space on it to re-check.");
                }
                Err(e) => {
                    if !token.is_empty() {
                        let _ = crate::credentials::clear_catalog_credential(url);
                    }
                    eprintln!("Could not save '{name}': {e}");
                }
            }
        }
    }
}

/// Add/edit a catalog with the chrome suspended (see [`add_or_edit_console`]).
#[allow(clippy::too_many_arguments)]
fn add_or_edit_catalog(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    name: &str,
    url: &str,
    token: &str,
    replacing: Option<&str>,
    catalog_locations: &std::collections::HashMap<String, String>,
    disabled: &mut std::collections::HashSet<String>,
    unvalidated: &mut std::collections::HashSet<String>,
) {
    with_console(terminal, || {
        add_or_edit_console(name, url, token, replacing, catalog_locations, disabled, unvalidated);
    });
}

/// Re-verify an already-configured (unvalidated) catalog with the chrome
/// suspended. No file rewrite — the entry already exists; on success it
/// just clears the disabled/unvalidated flags (enabling it), else it stays
/// disabled with the failure on the console.
fn revalidate_catalog(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    name: &str,
    url: &str,
    disabled: &mut std::collections::HashSet<String>,
    unvalidated: &mut std::collections::HashSet<String>,
) {
    with_console(terminal, || {
        println!("Re-checking catalog '{name}' — {url}");
        match crate::config::verify_catalog_source(url) {
            Ok(count) => {
                if unvalidated.remove(name) {
                    let _ = crate::settings::write_unvalidated_catalogs(unvalidated);
                }
                if disabled.remove(name) {
                    let _ = crate::settings::write_disabled_catalogs(disabled);
                }
                println!("OK — {count} dataset(s). Enabled.");
            }
            Err(e) => {
                eprintln!("Still failing: {e}");
                eprintln!("Left disabled. Fix the URL with 'e' (edit) in the config view.");
            }
        }
    });
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
pub fn run_picker<F>(mut dispatch: F, start_in_settings: bool) -> PickerOutcome
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
    // Session-long facet knowledge (url → bytes/bpr): seeded from
    // local sources now, completed by background remote probes once
    // the terminal is up. Drives the COUNT and SIZE cells and the
    // per-facet cardinality lines in the detail overlays.
    let mut facet_knowledge: std::collections::HashMap<String, FacetKnowledge> =
        std::collections::HashMap::new();
    collect_local_knowledge(entries, &cache_dir, &cache_survey, &mut facet_knowledge);
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

    // Background remote-size probes for facets the local pass left
    // incompletely answered (untouched remote datasets). Answers
    // stream in over the channel, drained at the top of the event
    // loop into `facet_knowledge`, which lives across row rebuilds
    // so nothing is ever probed twice.
    let size_probe_cancel = std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(false));
    let size_probe_rx = {
        let jobs = collect_size_probe_jobs(entries, &mut facet_knowledge);
        spawn_size_probes(jobs, size_probe_cancel.clone())
    };
    apply_size_cells(&mut all_rows, entries, &facet_knowledge);

    let mut filter = String::new();
    // Last theme-knob outcome, shown on the footer's summary line
    // until any other key clears it.
    let mut flash: Option<String> = None;
    // Catalog toggle screen (Ctrl-G): the ordered catalog names with
    // dataset counts, the disabled set (session-scoped — comment the
    // entry out in catalogs.yaml to disable persistently), and the
    // screen's open/cursor state.
    // name → configured location, from catalogs.yaml (the authoritative
    // name↔location binding) — EVERY configured catalog, including disabled
    // and unvalidated ones that resolve to no datasets. Drives the config
    // pane's catalog list, the URL shown under the highlighted catalog, and
    // removal by exact source line.
    let catalog_locations: std::collections::HashMap<String, String> =
        crate::catalog::sources::named_catalog_entries(
            &crate::catalog::sources::config_dir())
            .into_iter()
            .map(|s| (s.name, s.location))
            .collect();
    // Every configured catalog with its dataset count: resolved catalogs
    // (from `entries`) first, then any configured catalog that resolved to
    // 0 datasets (disabled / unvalidated / unreachable) so it still shows
    // in the config pane and can be fixed or re-enabled.
    let catalog_list: Vec<(String, usize)> = {
        let mut ordered: Vec<(String, usize)> = Vec::new();
        for e in entries {
            let name = e.catalog_name.clone().unwrap_or_else(|| "?".to_string());
            match ordered.iter_mut().find(|(n, _)| *n == name) {
                Some((_, count)) => *count += 1,
                None => ordered.push((name, 1)),
            }
        }
        let mut extra: Vec<&String> = catalog_locations.keys()
            .filter(|n| !ordered.iter().any(|(x, _)| x == *n))
            .collect();
        extra.sort();
        for name in extra {
            ordered.push((name.clone(), 0));
        }
        ordered
    };
    // Catalog names that have a credential recorded for their URL (the
    // auth indicator). Joined pairwise by name: catalogs.yaml gives
    // name→url, credentials.toml is keyed by url. Rebuilt on reload (the
    // whole picker re-runs), so it always reflects what's on disk.
    let authed_catalogs: std::collections::HashSet<String> = catalog_locations.iter()
        .filter(|(_, url)| crate::credentials::credential_for_url(url).is_some())
        .map(|(name, _)| name.clone())
        .collect();
    // Seed from the persisted `disabled_catalogs` setting; only
    // names that actually exist in this session's catalog list count
    // (a stale name in settings silently waits for its catalog to
    // come back).
    let mut disabled_catalogs: std::collections::HashSet<String> =
        crate::settings::disabled_catalogs().into_iter().collect();
    // Catalogs saved but failing verification — kept disabled and flagged
    // for fixing (edit / re-check). Mutated + persisted alongside
    // disabled_catalogs by the add/edit/re-validate flow.
    let mut unvalidated_catalogs: std::collections::HashSet<String> =
        crate::settings::unvalidated_catalogs().into_iter().collect();
    // Hidden columns, persisted as the `disabled_columns` key.
    let mut disabled_columns: std::collections::HashSet<String> =
        crate::settings::disabled_columns().into_iter().collect();
    // Opens straight onto the config view after a reload triggered by a
    // catalog change (add/edit/remove/auth), so the user sees the result
    // (and the colour-coded status) on the Catalogs pane they came from.
    let mut catalog_screen = start_in_settings;
    let mut catalog_cursor: usize = 0;
    // Which settings tab is active (the screen is a tabbed/faceted view);
    // Tab / Shift-Tab move between tabs, resetting catalog_cursor.
    let mut settings_tab = SettingsTab::Catalogs;
    // Settings-screen sub-modes (meaningful only while `catalog_screen`):
    //   add_input      — capturing a new catalog's name, URL, token.
    //   auth_input     — setting/clearing auth for an existing catalog.
    //   confirm_remove — holds a catalog name pending a remove keystroke.
    let mut add_input: Option<AddCatalogInput> = None;
    let mut auth_input: Option<AuthInput> = None;
    let mut confirm_remove: Option<String> = None;
    let mut cursor: usize = 0;
    let mut scroll: usize = 0;
    let mut show_all_profiles = false;
    let mut expanded: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut show_help = false;
    let mut menu_open = false;
    // Two-press quit guard for the picker's final exit (Esc / Ctrl-D):
    // the first press arms + prompts, a second within QUIT_CONFIRM quits.
    // Ctrl-C stays the unguarded escape hatch.
    const QUIT_CONFIRM: std::time::Duration = std::time::Duration::from_secs(1);
    let mut quit_armed: Option<std::time::Instant> = None;
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
        // Fold finished remote-size probes into the knowledge map,
        // then recompute the COUNT/SIZE cells they could affect.
        let mut probes_landed = false;
        while let Ok(result) = size_probe_rx.try_recv() {
            let slot = facet_knowledge.entry(result.url).or_default();
            // Local knowledge wins — a probe never downgrades it.
            slot.bytes = slot.bytes.or(result.bytes);
            slot.bpr = slot.bpr.or(result.bpr);
            slot.pending = false;
            slot.probed = true;
            probes_landed = true;
        }
        if probes_landed {
            apply_size_cells(&mut all_rows, entries, &facet_knowledge);
        }

        // Filter rows:
        // - show_all_profiles: show everything
        // - otherwise: show "default" profile + all profiles of expanded datasets
        let visible: Vec<&PickerRow> = all_rows.iter()
            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
            .collect();

        if cursor >= visible.len() && !visible.is_empty() {
            cursor = visible.len() - 1;
        }

        // Catalog activity for the title summary: `[n catalogs]` when
        // they're all on, `[m/n catalogs] active` once any has been
        // toggled off (Ctrl-G), so the row count is read against how
        // many catalogs are actually contributing.
        let total_catalogs = catalog_list.len();
        let active_catalogs = catalog_list.iter()
            .filter(|(n, _)| !disabled_catalogs.contains(n))
            .count();
        let catalog_summary = if active_catalogs == total_catalogs {
            format!("[{total_catalogs} catalogs]")
        } else {
            format!("[{active_catalogs}/{total_catalogs} catalogs] active")
        };

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
                Span::styled(" Filter: ", tinted(theme().text_muted)),
                Span::styled(filter.clone(), tinted(theme().text_primary)),
                Span::styled("█", tinted(theme().primary)),
            ]);
            frame.render_widget(
                Paragraph::new(filter_line)
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(bordered(theme().info))
                        .title(Span::styled(
                            format!(" Select Dataset — {} shown ({}) {} ",
                                visible.len(),
                                if show_all_profiles { "all profiles" } else { "default only — Tab for all" },
                                catalog_summary),
                            tinted(theme().primary),
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
                // Compact on purpose: every line ≤78 columns and the
                // whole card ≤19 rows, so it renders complete on an
                // 80×24 terminal (the Paragraph doesn't scroll —
                // anything taller silently truncates). One group per
                // label; the "Overlays" line documents the shared
                // conventions of every sub-screen (settings,
                // describe/source, menus) in one place.
                let group = |label: &str, text: &str| {
                    Line::from(vec![
                        Span::styled(format!(" {label:<10}"), tinted(theme().primary)),
                        Span::styled(text.to_string(), tinted(theme().text_primary)),
                    ])
                };
                let cont = |text: &str| {
                    Line::from(vec![
                        Span::raw(" ".repeat(11)),
                        Span::styled(text.to_string(), tinted(theme().text_primary)),
                    ])
                };
                let help = vec![
                    Line::from(Span::styled(" Dataset Picker — Keys", tinted(theme().primary))),
                    Line::from(""),
                    group("Navigate",  "↑↓ move · PgUp/PgDn page · Home/End ends · Tab all profiles"),
                    cont("Enter/→ expand or open menu · ← collapse"),
                    group("Filter",    "type to match name/profile/metric · Backspace edits"),
                    cont("ALL-CAPS filters facets: B=Base Q=Query G=GroundTruth"),
                    cont("D=Distances M=Meta P=Predicates R=Results F=Filtered"),
                    group("Select",    "Space toggles row · Shift+↑↓ extends range, Space commits"),
                    cont("Enter with a selection → batch menu (Download/Ping/Purge)"),
                    group("Inspect",   "Enter menu → Describe / Source / Locate"),
                    group("Configure", "Ctrl-G settings: catalogs · columns · theme · reset"),
                    cont("Ctrl-T mono/colour (terminals that mangle ANSI)"),
                    group("Overlays",  "↑↓ or j/k scroll · g/G Home/End ends · Esc or q close"),
                    group("Exit",      "Esc/Ctrl-D peel: help → range → selection →"),
                    cont("expanded → filter, then again to quit · Ctrl-C quits now"),
                ];
                frame.render_widget(
                    Paragraph::new(help).block(Block::default().borders(Borders::ALL)
                        .border_style(bordered(theme().info))
                        .title(Span::styled(" Help ", tinted(theme().primary)))),
                    chunks[1],
                );
            } else {

            // Column widths measured in DISPLAY columns. The prefix
            // is split into independent slots so the cursor pip and
            // the multi-select toggle marker never collide:
            //   col 0  cursor pip ('▶' or ' ')
            //   col 1  toggle mark ('●' or ' ')
            //   col 2-4 tree glyph (up to 3 cols: " └ ", "▸ ", etc.)
            // Reserve 5 cols on top of the longest dataset name.
            // Inter-column gaps come from explicit one-space
            // separator spans (see the header/row assembly below),
            // never from width headroom — every cell is rendered at
            // exactly its column width, so a value filling its cell
            // edge-to-edge still gets a gap before the next column.
            let name_w = visible.iter()
                .map(|r| UnicodeWidthStr::width(r.dataset.as_str()) + 5)
                .max()
                .unwrap_or(20)
                .max(12);
            // PROFILE column sized to fit the longest profile name.
            // Default of 14 is plenty for `default`, `1m`, `40mi`
            // etc., but partition / user-named profiles can run
            // longer.
            let prof_w = visible.iter()
                .map(|r| UnicodeWidthStr::width(r.profile.as_str()))
                .max()
                .unwrap_or(14)
                .max(10);
            // Per-column widths. CATALOG sizes to its content
            // (catalog names are user-chosen); the rest are fixed.
            let catalog_w = visible.iter()
                .map(|r| UnicodeWidthStr::width(r.catalog.as_str()))
                .max()
                .unwrap_or(1)
                .max(1);
            let col_w = |c: PickerColumn| -> usize {
                match c {
                    PickerColumn::Catalog => catalog_w,
                    PickerColumn::Facets  => 12,
                    PickerColumn::Metric  => 14,
                    // Room for the shared-base "*" after the value.
                    PickerColumn::Count   => 8,
                    // Room for the "+" (sum is a floor) / "*" marks.
                    PickerColumn::Size    => 9,
                    PickerColumn::Access  => 10,
                    PickerColumn::Cached  => 18,
                }
            };
            let enabled_cols: Vec<PickerColumn> = ALL_COLUMNS.iter().copied()
                .filter(|c| !disabled_columns.contains(c.name()))
                .collect();
            // The catalog column sits LEFT of DATASET (provenance
            // before identity); the rest follow PROFILE.
            let catalog_enabled = enabled_cols.contains(&PickerColumn::Catalog);
            let trailing_cols: Vec<PickerColumn> = enabled_cols.iter().copied()
                .filter(|c| *c != PickerColumn::Catalog)
                .collect();

            // Header and data rows share the same cell widths so columns
            // line up edge-to-edge. Data rows prefix the dataset name
            // with marker + tree glyph columns; the DATASET header is
            // indented by the same width so it aligns with the names
            // themselves (2 marker cols + 2 glyph cols in tree mode,
            // markers only when every profile is flattened).
            let name_indent = if show_all_profiles { 2 } else { 4 };
            let dataset_header = format!("{}DATASET", " ".repeat(name_indent));
            let mut header_spans = Vec::new();
            if catalog_enabled {
                header_spans.push(Span::styled(
                    fit_display("C", catalog_w), tinted(theme().text_muted)));
                header_spans.push(Span::raw(" "));
            }
            header_spans.push(Span::styled(fit_display(&dataset_header, name_w), tinted(theme().text_muted)));
            header_spans.push(Span::raw(" "));
            header_spans.push(Span::styled(fit_display("PROFILE", prof_w), tinted(theme().text_muted)));
            for c in &trailing_cols {
                header_spans.push(Span::raw(" "));
                header_spans.push(Span::styled(
                    fit_display(c.header(), col_w(*c)), tinted(theme().text_muted)));
            }
            let mut lines = vec![Line::from(header_spans)];

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
                    tinted(theme().info)
                } else if row.cache_status != "—" {
                    tinted(theme().success)
                } else {
                    tinted(theme().text_primary)
                };
                let cache_style = if in_range {
                    style
                } else if row.cache_status != "—" {
                    tinted(theme().success)
                } else {
                    tinted(theme().text_dim)
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
                    AccessMode::Local         => theme().primary,
                    AccessMode::MerkleHashed  => theme().success,
                    AccessMode::MerkleChunked => theme().info,
                    AccessMode::FullTransfer  => theme().error,
                };
                let access_style = if in_range {
                    style
                } else {
                    tinted(access_color)
                };

                // Separator spans carry the row's base style so the
                // REVERSED highlight bar stays continuous across
                // column gaps on the cursor/range rows.
                let mut row_spans = Vec::new();
                if catalog_enabled {
                    row_spans.push(Span::styled(
                        fit_display(&row.catalog, catalog_w),
                        if in_range { style } else { tinted(theme().text_muted) }));
                    row_spans.push(Span::styled(" ", style));
                }
                row_spans.push(Span::styled(name_cell, style));
                row_spans.push(Span::styled(" ", style));
                row_spans.push(Span::styled(fit_display(&row.profile, prof_w), style));
                for c in &trailing_cols {
                    let w = col_w(*c);
                    row_spans.push(Span::styled(" ", style));
                    row_spans.push(match c {
                        PickerColumn::Facets =>
                            Span::styled(fit_display(&row.facets, w), style),
                        PickerColumn::Metric =>
                            Span::styled(fit_display(&row.metric, w), style),
                        PickerColumn::Count =>
                            Span::styled(fit_display(&row.count, w), style),
                        PickerColumn::Size =>
                            Span::styled(fit_display(&row.size, w), style),
                        PickerColumn::Access =>
                            Span::styled(fit_display(row.access.short_label(), w), access_style),
                        PickerColumn::Cached =>
                            Span::styled(fit_display(&row.cache_status, w), cache_style),
                        // Catalog renders before DATASET, never here.
                        PickerColumn::Catalog => unreachable!("catalog is filtered out of trailing_cols"),
                    });
                }
                lines.push(Line::from(row_spans));
            }

            frame.render_widget(
                Paragraph::new(lines)
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(bordered(theme().info))),
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
                // Name the legend so users know what the indicator
                // letters describe.
                if theme_on {
                    spans.push(Span::styled("Facets: ", Style::default().fg(theme().text_muted)));
                } else {
                    spans.push(Span::raw("Facets: "));
                }
                let facet_details: &[(&str, char, Color)] = &[
                    ("Base",     'B', theme().success),
                    ("Query",    'Q', theme().success),
                    ("GT",       'G', theme().warning),
                    ("Dist",     'D', theme().warning),
                    ("Meta",     'M', theme().meta),
                    ("Pred",     'P', theme().meta),
                    ("Results",  'R', theme().meta),
                    ("Filtered", 'F', theme().error),
                ];
                for (label, code, color) in facet_details {
                    let present = row.facets.contains(*code);
                    if theme_on {
                        let style = if present { Style::default().fg(*color) }
                                    else { Style::default().fg(theme().text_dim) };
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
                    AccessMode::Local         => theme().primary,
                    AccessMode::MerkleHashed  => theme().success,
                    AccessMode::MerkleChunked => theme().info,
                    AccessMode::FullTransfer  => theme().error,
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
                    .border_style(bordered(theme().info))),
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
            // The flash slot shows the latest theme-knob action
            // (cycle/save outcome) until the next keypress replaces
            // the summary line's left edge.
            let summary_line = match &flash {
                Some(f) => format!(" {f}"),
                None => summary,
            };
            // Essentials only — the full keymap lives in the `?`
            // help overlay. One line, 78 display columns, so an
            // 80-column terminal shows it whole instead of
            // truncating a cheat sheet. `^G` (terminal notation, as
            // in nano) keeps the config entry point discoverable
            // without blowing the width budget — settings must be
            // findable without first finding help.
            // When a quit is armed, the hint line becomes the confirm
            // prompt (in the warning colour) so the second Esc/Ctrl-D is
            // a deliberate one.
            let quit_pending = quit_armed.map(|t| t.elapsed() < QUIT_CONFIRM).unwrap_or(false);
            let (hint, hint_color) = if quit_pending {
                (" Press Esc (or Ctrl-D) again to quit — any other key cancels",
                 theme().warning)
            } else {
                (" type: filter · Space: select · Enter: menu · ^G: config · ?: help · Esc/^D: quit",
                 theme().text_muted)
            };
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(summary_line, tinted(theme().primary))),
                    Line::from(Span::styled(hint, tinted(hint_color))),
                ]),
                chunks[3],
            );

            // Descriptor overlay (Describe action) — the single dataset-inspect
            // view. Scrollable to handle descriptors that overflow the screen.
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
                            .border_style(bordered(theme().info))
                            .title(Span::styled(
                                format!(" {} ({}/{} · ↑↓ j/k scroll · g/G ends · Esc/q close) ",
                                    descriptor_title,
                                    descriptor_scroll + 1,
                                    descriptor_lines.len().max(1)),
                                tinted(theme().primary),
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
                        tinted(theme().text_primary)
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
                            .border_style(bordered(theme().info))
                            .title(Span::styled(
                                " Which target? (↑↓ · Enter · Esc) ",
                                tinted(theme().info),
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
                        tinted(theme().text_primary)
                    };
                    lines.push(Line::from(Span::styled(
                        format!("{marker}{}", action.label()),
                        style,
                    )));
                }
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!(" {}", actions[menu_cursor].description()),
                    tinted(theme().text_muted),
                )));
                frame.render_widget(ratatui::widgets::Clear, area);
                frame.render_widget(
                    Paragraph::new(lines)
                        .block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(theme().primary))
                            .title(Span::styled(title_text, tinted(theme().primary)))),
                    area,
                );
            }

            // Settings screen (Ctrl-G): a faceted, tabbed view —
            // Catalogs / Columns / Theme / Maintenance. Tab and Shift-Tab
            // move between tabs; ↑↓ move the cursor within the active tab.
            // Toggles persist immediately; theme cycles are session state
            // with an explicit save item.
            if catalog_screen {
                let items = settings_tab.items(&catalog_list);
                let outer = frame.area();
                // Only the highlighted catalog reveals its URL + count +
                // auth (three detail rows); every other catalog shows name
                // only.
                let highlighted_catalog = match items.get(catalog_cursor) {
                    Some(SettingsItem::CatalogToggle(n)) => Some(n.clone()),
                    _ => None,
                };
                let highlighted_unvalidated = highlighted_catalog.as_ref()
                    .map(|n| unvalidated_catalogs.contains(n)).unwrap_or(false);
                // Validated catalog detail = url + count + auth (3 rows); an
                // unvalidated one = url + the ⚠ fix line (2 rows).
                let detail_h: u16 = match (&highlighted_catalog, highlighted_unvalidated) {
                    (Some(_), false) => 3,
                    (Some(_), true) => 2,
                    (None, _) => 0,
                };
                // tab bar (1) + blank (1) + items + detail + blank (1)
                // + hint (1) + borders (2).
                let h: u16 = (items.len() as u16) + 6 + detail_h;
                let w: u16 = 76;
                let area = ratatui::layout::Rect {
                    x: outer.x + outer.width.saturating_sub(w) / 2,
                    y: outer.y + outer.height.saturating_sub(h) / 2,
                    width: w.min(outer.width),
                    height: h.min(outer.height),
                };
                let (cur_palette, cur_curve) = super::theme_palette_curve();
                // Palette/curve are session state persisted only by "save
                // theme"; flag them when the live value differs from what's
                // saved in settings.yaml so unsaved changes are visible.
                let (saved_palette, saved_curve) = super::resolve_palette_curve(None, None);
                let palette_dirty = cur_palette.name() != saved_palette.name();
                let curve_dirty = cur_curve.name() != saved_curve.name();
                let theme_dirty = palette_dirty || curve_dirty;
                // Warning-coloured style for a modified-but-unsaved row.
                let dirty_style = |selected: bool| if selected {
                    tinted(theme().warning).add_modifier(Modifier::REVERSED)
                } else {
                    tinted(theme().warning)
                };
                let mut lines: Vec<Line> = Vec::new();
                // Tab bar: the active tab inverted, the rest dim.
                let mut tab_spans: Vec<Span> = vec![Span::raw(" ")];
                for (i, t) in SettingsTab::ALL.iter().enumerate() {
                    if i > 0 {
                        tab_spans.push(Span::styled(" · ", tinted(theme().text_dim)));
                    }
                    let style = if *t == settings_tab {
                        selected_style()
                    } else {
                        tinted(theme().text_dim)
                    };
                    tab_spans.push(Span::styled(format!(" {} ", t.title()), style));
                }
                lines.push(Line::from(tab_spans));
                lines.push(Line::from(""));
                for (i, item) in items.iter().enumerate() {
                    let marker = if i == catalog_cursor { " ▸ " } else { "   " };
                    let (text, enabled) = match item {
                        // Name + checkbox only, with a lock when a
                        // credential is recorded for it; the URL/count/auth
                        // detail appears below for the highlighted one.
                        SettingsItem::CatalogToggle(name) => {
                            let enabled = !disabled_catalogs.contains(name);
                            let lock = if authed_catalogs.contains(name) { "  🔒" } else { "" };
                            let warn = if unvalidated_catalogs.contains(name) { "  ⚠" } else { "" };
                            (format!("[{}] {name}{lock}{warn}",
                                if enabled { "x" } else { " " }), enabled)
                        }
                        SettingsItem::AddCatalog =>
                            ("+ add a catalog…".to_string(), true),
                        SettingsItem::ColumnToggle(col) => {
                            let enabled = !disabled_columns.contains(col.name());
                            (format!("[{}] {} column",
                                if enabled { "x" } else { " " }, col.name()), enabled)
                        }
                        SettingsItem::PaletteCycle =>
                            (format!("palette: {}{}  (Space cycles)", cur_palette.name(),
                                if palette_dirty { " *modified" } else { "" }), true),
                        SettingsItem::CurveCycle =>
                            (format!("curve: {}{}  (Space cycles)", cur_curve.name(),
                                if curve_dirty { " *modified" } else { "" }), true),
                        SettingsItem::SaveTheme =>
                            (if theme_dirty {
                                "save theme to settings as default  *unsaved".to_string()
                            } else {
                                "save theme to settings as default".to_string()
                            }, true),
                        SettingsItem::ResetDisplay =>
                            ("reset display options (palette, curve, columns)".to_string(), true),
                    };
                    // Catalog rows are colour-coded by status — green active,
                    // dim disabled, warning unvalidated — so state reads at a
                    // glance; the cursor row keeps that colour but reversed.
                    let style = match item {
                        SettingsItem::CatalogToggle(name) => {
                            let sc = if unvalidated_catalogs.contains(name) {
                                theme().warning
                            } else if disabled_catalogs.contains(name) {
                                theme().text_dim
                            } else {
                                theme().success
                            };
                            if i == catalog_cursor {
                                tinted(sc).add_modifier(Modifier::REVERSED)
                            } else {
                                tinted(sc)
                            }
                        }
                        SettingsItem::PaletteCycle if palette_dirty =>
                            dirty_style(i == catalog_cursor),
                        SettingsItem::CurveCycle if curve_dirty =>
                            dirty_style(i == catalog_cursor),
                        SettingsItem::SaveTheme if theme_dirty =>
                            dirty_style(i == catalog_cursor),
                        _ if i == catalog_cursor => selected_style(),
                        _ if enabled => tinted(theme().text_primary),
                        _ => tinted(theme().text_dim),
                    };
                    lines.push(Line::from(Span::styled(format!("{marker}{text}"), style)));
                }
                // Detail for the highlighted catalog: source URL, then either
                // the count + auth (validated) or the ⚠ fix line (unvalidated).
                if let Some(SettingsItem::CatalogToggle(name)) = items.get(catalog_cursor) {
                    let loc = catalog_locations.get(name).map(String::as_str)
                        .unwrap_or("(not a catalogs.yaml entry — not removable here)");
                    lines.push(Line::from(Span::styled(
                        format!("      {loc}"), tinted(theme().text_muted))));
                    if unvalidated_catalogs.contains(name) {
                        lines.push(Line::from(Span::styled(
                            "      ⚠ not validated — e: edit & re-check · Space: re-check",
                            tinted(theme().warning))));
                    } else {
                        let count = catalog_list.iter()
                            .find(|(n, _)| n == name).map(|(_, c)| *c).unwrap_or(0);
                        lines.push(Line::from(Span::styled(
                            format!("      {count} dataset{}", if count == 1 { "" } else { "s" }),
                            tinted(theme().text_dim))));
                        // Auth line: who when a credential is stored, else a
                        // hint that `c` provides one.
                        let auth_line = match catalog_locations.get(name)
                            .and_then(|u| crate::credentials::credential_for_url(u))
                        {
                            Some(e) => {
                                let who = e.user.as_deref().unwrap_or("token");
                                format!("      🔒 auth: {who} (c: change · c then empty: clear)")
                            }
                            None => "      no auth — c: provide a token".to_string(),
                        };
                        lines.push(Line::from(Span::styled(auth_line, tinted(theme().info))));
                    }
                }
                lines.push(Line::from(""));
                let hint = if matches!(settings_tab, SettingsTab::Catalogs) {
                    " ←→ tabs · ↑↓ move · Space toggle · a:add e:edit c:auth x:remove"
                } else {
                    " ←→ tabs · ↑↓ move · Space: toggle/activate"
                };
                lines.push(Line::from(Span::styled(hint, tinted(theme().text_muted))));
                frame.render_widget(ratatui::widgets::Clear, area);
                frame.render_widget(
                    Paragraph::new(lines)
                        .block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(theme().primary))
                            .title(Span::styled(
                                " Settings — ←→/Tab: tabs · Esc/Ctrl-D: close ",
                                tinted(theme().primary)))),
                    area,
                );

                // Add-catalog input overlay (on top of the settings box).
                if let Some(input) = &add_input {
                    // Use most of the terminal width (capped) so long URLs have
                    // room; the active field also scrolls to keep the caret in
                    // view (see field_window), so it's never clipped off-screen.
                    let iw: u16 = outer.width.saturating_sub(6).clamp(40, 100).min(outer.width);
                    let ih: u16 = 9;
                    let ia = ratatui::layout::Rect {
                        x: outer.x + outer.width.saturating_sub(iw) / 2,
                        y: outer.y + outer.height.saturating_sub(ih) / 2,
                        width: iw,
                        height: ih.min(outer.height),
                    };
                    let field_style = |active: bool| if active {
                        tinted(theme().info)
                    } else {
                        tinted(theme().text_dim)
                    };
                    // The token is a secret — show it masked.
                    let masked = "•".repeat(input.token.chars().count());
                    // Each field line is "  <label>  <value>"; the value starts
                    // at a fixed column (FIELD_COL). `vis_w` is how many value
                    // columns fit inside the box.
                    const FIELD_COL: u16 = 10;
                    let vis_w = iw.saturating_sub(2 + FIELD_COL);
                    let is = |f: AddField| input.field == f;
                    let (name_v, name_caret) = field_window(&input.name, vis_w, is(AddField::Name));
                    let (url_v, url_caret) = field_window(&input.url, vis_w, is(AddField::Url));
                    let (tok_v, tok_caret) = field_window(&masked, vis_w, is(AddField::Token));
                    let il = vec![
                        Line::from(""),
                        Line::from(Span::styled(format!("  name:   {name_v}"), field_style(is(AddField::Name)))),
                        Line::from(Span::styled(format!("  url:    {url_v}"), field_style(is(AddField::Url)))),
                        Line::from(Span::styled(format!("  token:  {tok_v}"), field_style(is(AddField::Token)))),
                        Line::from(""),
                        Line::from(Span::styled(
                            "  Tab/↑↓: fields · Enter: verify+save · Esc: cancel · token optional",
                            tinted(theme().text_muted))),
                    ];
                    frame.render_widget(ratatui::widgets::Clear, ia);
                    frame.render_widget(
                        Paragraph::new(il).block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(theme().primary))
                            .title(Span::styled(" Add catalog ", tinted(theme().primary)))),
                        ia,
                    );
                    // Real (terminal-native, blinking) cursor at the caret of
                    // the active field — replaces the painted underscore.
                    let (line_idx, caret) = match input.field {
                        AddField::Name => (1u16, name_caret),
                        AddField::Url => (2, url_caret),
                        AddField::Token => (3, tok_caret),
                    };
                    let cx = (ia.x + 1 + FIELD_COL + caret)
                        .min(ia.x + ia.width.saturating_sub(2));
                    frame.set_cursor_position((cx, ia.y + 1 + line_idx));
                }

                // Set-auth input overlay (token for an existing catalog).
                if let Some(auth) = &auth_input {
                    let aw: u16 = outer.width.saturating_sub(6).clamp(40, 100).min(outer.width);
                    let ah: u16 = 8;
                    let aa = ratatui::layout::Rect {
                        x: outer.x + outer.width.saturating_sub(aw) / 2,
                        y: outer.y + outer.height.saturating_sub(ah) / 2,
                        width: aw,
                        height: ah.min(outer.height),
                    };
                    let masked = "•".repeat(auth.token.chars().count());
                    const AUTH_FIELD_COL: u16 = 11; // width of "  token:   "
                    let (tok_v, tok_caret) =
                        field_window(&masked, aw.saturating_sub(2 + AUTH_FIELD_COL), true);
                    // The (display-only) URL line: prefix "  " = 2 cols.
                    let (url_disp, _) = field_window(&auth.url, aw.saturating_sub(4), false);
                    let al = vec![
                        Line::from(""),
                        Line::from(Span::styled(
                            format!("  catalog: {}", auth.name), tinted(theme().text_muted))),
                        Line::from(Span::styled(
                            format!("  {url_disp}"), tinted(theme().text_dim))),
                        Line::from(Span::styled(
                            format!("  token:   {tok_v}"), tinted(theme().info))),
                        Line::from(""),
                        Line::from(Span::styled(
                            "  Enter: save (empty clears) · Esc: cancel",
                            tinted(theme().text_muted))),
                    ];
                    frame.render_widget(ratatui::widgets::Clear, aa);
                    frame.render_widget(
                        Paragraph::new(al).block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(theme().info))
                            .title(Span::styled(" Set auth ", tinted(theme().info)))),
                        aa,
                    );
                    // Real (blinking) cursor at the caret of the token field.
                    let cx = (aa.x + 1 + AUTH_FIELD_COL + tok_caret)
                        .min(aa.x + aa.width.saturating_sub(2));
                    frame.set_cursor_position((cx, aa.y + 1 + 3));
                }

                // Remove-confirmation overlay.
                if let Some(name) = &confirm_remove {
                    let cw: u16 = 64;
                    let ch: u16 = 7;
                    let ca = ratatui::layout::Rect {
                        x: outer.x + outer.width.saturating_sub(cw) / 2,
                        y: outer.y + outer.height.saturating_sub(ch) / 2,
                        width: cw.min(outer.width),
                        height: ch.min(outer.height),
                    };
                    let loc = catalog_locations.get(name).map(String::as_str).unwrap_or("");
                    let cl = vec![
                        Line::from(""),
                        Line::from(Span::styled(
                            format!("  Remove catalog '{name}'?"), tinted(theme().text_primary))),
                        Line::from(Span::styled(format!("  {loc}"), tinted(theme().text_muted))),
                        Line::from(""),
                        Line::from(Span::styled(
                            "  y: remove from catalogs.yaml · any other key: cancel",
                            tinted(theme().text_muted))),
                    ];
                    frame.render_widget(ratatui::widgets::Clear, ca);
                    frame.render_widget(
                        Paragraph::new(cl).block(Block::default().borders(Borders::ALL)
                            .border_style(bordered(theme().warning))
                            .title(Span::styled(" Confirm remove ", tinted(theme().warning)))),
                        ca,
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
        if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false)
            && let Ok(Event::Key(key)) = event::read() {
                // Ctrl-C: immediate hard quit (the escape hatch), before
                // any rewrite below.
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    result = PickerOutcome::Done;
                    break;
                }
                // Ctrl-D is an alternate Esc — the soft "back / quit" key,
                // matching the visualizer. Rewriting it to Esc here means it
                // flows through the same cascade + double-tap-to-quit below
                // (and closes modals) with no special-casing.
                let key = if key.code == KeyCode::Char('d')
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                {
                    crossterm::event::KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)
                } else {
                    key
                };
                // Any key disarms a pending quit-confirm; the Esc cascade's
                // final step re-arms it, so two Escs (or Ctrl-Ds) in a row
                // quit while a stray key cancels.
                let prev_quit = quit_armed.take();
                if key.code == KeyCode::Char('t') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    // Ctrl-T: toggle colour theme. The rendering
                    // codepaths fall back to terminal-default fg/bg
                    // when this is off; selection still inverts via
                    // `Modifier::REVERSED` so the highlight is
                    // visible regardless.
                    colors_enabled = !colors_enabled;
                    continue;
                }
                // Theme knobs (palette/curve cycle, save) live on the
                // Ctrl-G settings screen only — global chords for
                // them were dropped to keep the keymap small. Ctrl-T
                // stays global: it is the accessibility escape hatch
                // for terminals that mangle color, so it must work
                // without navigating a (colored) settings screen.
                if key.code == KeyCode::Char('g') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    catalog_screen = !catalog_screen;
                    catalog_cursor = 0;
                    continue;
                }
                // Settings screen is modal: swallow everything except
                // its own navigation/activate/close keys. Toggles
                // persist immediately (checkbox semantics); theme
                // cycles are session state with an explicit save item.
                if catalog_screen {
                    // Sub-modal: typing a new catalog's name, URL, token.
                    // Captures every key; Enter advances/saves, Esc cancels.
                    if add_input.is_some() {
                        match key.code {
                            KeyCode::Esc => { add_input = None; }
                            // Tab / ↓ go to the next field, Shift-Tab / ↑ the
                            // previous — so any field can be edited, not just
                            // forward via Enter.
                            KeyCode::Tab | KeyCode::Down => {
                                let input = add_input.as_mut().unwrap();
                                input.field = match input.field {
                                    AddField::Name => AddField::Url,
                                    AddField::Url => AddField::Token,
                                    AddField::Token => AddField::Name,
                                };
                            }
                            KeyCode::BackTab | KeyCode::Up => {
                                let input = add_input.as_mut().unwrap();
                                input.field = match input.field {
                                    AddField::Name => AddField::Token,
                                    AddField::Url => AddField::Name,
                                    AddField::Token => AddField::Url,
                                };
                            }
                            KeyCode::Backspace => {
                                let input = add_input.as_mut().unwrap();
                                match input.field {
                                    AddField::Name => { input.name.pop(); }
                                    AddField::Url => { input.url.pop(); }
                                    AddField::Token => { input.token.pop(); }
                                }
                            }
                            KeyCode::Char(c)
                                if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                                let input = add_input.as_mut().unwrap();
                                match input.field {
                                    AddField::Name => input.name.push(c),
                                    AddField::Url => input.url.push(c),
                                    AddField::Token => input.token.push(c),
                                }
                            }
                            KeyCode::Enter => {
                                let field = add_input.as_ref().unwrap().field;
                                match field {
                                    // Enter advances linearly; Tab jumps freely.
                                    AddField::Name =>
                                        add_input.as_mut().unwrap().field = AddField::Url,
                                    AddField::Url =>
                                        add_input.as_mut().unwrap().field = AddField::Token,
                                    // Token (optional) → validate the required
                                    // fields (Tab may have skipped them), record
                                    // any credential first (so a protected
                                    // endpoint authenticates), then verify + save.
                                    AddField::Token => {
                                        let (name, url, token) = {
                                            let inp = add_input.as_ref().unwrap();
                                            (inp.name.trim().to_string(),
                                             inp.url.trim().to_string(),
                                             inp.token.trim().to_string())
                                        };
                                        if name.is_empty() {
                                            flash = Some("a catalog must be given a name".into());
                                            add_input.as_mut().unwrap().field = AddField::Name;
                                            continue;
                                        }
                                        if url.is_empty() {
                                            flash = Some("enter a catalog URL or path".into());
                                            add_input.as_mut().unwrap().field = AddField::Url;
                                            continue;
                                        }
                                        let replacing = add_input.as_ref()
                                            .unwrap().replacing.clone();
                                        // Verify with the chrome suspended so
                                        // progress + any failure message land
                                        // on the console; saves disabled +
                                        // flags unvalidated on failure.
                                        add_or_edit_catalog(
                                            &mut terminal, &name, &url, &token,
                                            replacing.as_deref(), &catalog_locations,
                                            &mut disabled_catalogs, &mut unvalidated_catalogs);
                                        result = PickerOutcome::Reload;
                                        break;
                                    }
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }
                    // Sub-modal: setting/clearing auth for an existing catalog.
                    if auth_input.is_some() {
                        match key.code {
                            KeyCode::Esc => { auth_input = None; }
                            KeyCode::Backspace => { auth_input.as_mut().unwrap().token.pop(); }
                            KeyCode::Char(c)
                                if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                                auth_input.as_mut().unwrap().token.push(c);
                            }
                            KeyCode::Enter => {
                                let AuthInput { name, url, token } = auth_input.take().unwrap();
                                let token = token.trim();
                                let outcome = if token.is_empty() {
                                    crate::credentials::clear_catalog_credential(&url)
                                        .map(|removed| if removed {
                                            format!("cleared auth for '{name}'")
                                        } else {
                                            format!("no auth was set for '{name}'")
                                        })
                                } else {
                                    crate::credentials::set_catalog_credential(
                                        &name, &url, token, None)
                                        .map(|()| format!("auth set for '{name}'"))
                                };
                                match outcome {
                                    // Reload so a now-authed catalog's datasets
                                    // load (or a cleared one drops them).
                                    Ok(_) => { result = PickerOutcome::Reload; break; }
                                    Err(e) => flash = Some(format!("auth failed: {e}")),
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }
                    // Sub-modal: confirming a catalog removal.
                    if let Some(name) = confirm_remove.clone() {
                        if matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y')) {
                            confirm_remove = None;
                            // Remove by exact source (works for both the map
                            // and legacy-list forms); fall back to the name.
                            let target = catalog_locations.get(&name).cloned()
                                .unwrap_or_else(|| name.clone());
                            match crate::config::try_remove_catalog(&target) {
                                Ok(true) => {
                                    if disabled_catalogs.remove(&name) {
                                        let _ = crate::settings::write_disabled_catalogs(
                                            &disabled_catalogs);
                                    }
                                    // Keep credentials.toml in step with the
                                    // catalogs: drop any token now left without a
                                    // configured catalog. (The edit flow's
                                    // internal remove does NOT prune — it re-adds.)
                                    crate::credentials::prune_orphan_credentials();
                                    // Reload; the catalog vanishing from the
                                    // list is the confirmation.
                                    result = PickerOutcome::Reload;
                                    break;
                                }
                                Ok(false) =>
                                    flash = Some(format!("'{name}' not found in catalogs.yaml")),
                                Err(e) => flash = Some(format!("remove failed: {e}")),
                            }
                        } else {
                            confirm_remove = None;
                        }
                        continue;
                    }
                    let items = settings_tab.items(&catalog_list);
                    match key.code {
                        KeyCode::Esc | KeyCode::Char('q') => { catalog_screen = false; }
                        // Tab / Shift-Tab and ←/→ move between the faceted tabs.
                        KeyCode::Tab | KeyCode::Right => {
                            settings_tab = settings_tab.next();
                            catalog_cursor = 0;
                        }
                        KeyCode::BackTab | KeyCode::Left => {
                            settings_tab = settings_tab.prev();
                            catalog_cursor = 0;
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            catalog_cursor = catalog_cursor.saturating_sub(1);
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if catalog_cursor + 1 < items.len() {
                                catalog_cursor += 1;
                            }
                        }
                        // Add a catalog from anywhere on the screen (jumps
                        // to the Catalogs tab implicitly via the add modal).
                        KeyCode::Char('a') => { add_input = Some(AddCatalogInput::new()); }
                        // Set/clear auth for the highlighted catalog.
                        KeyCode::Char('c') => {
                            if let Some(SettingsItem::CatalogToggle(name)) =
                                items.get(catalog_cursor)
                            {
                                match catalog_locations.get(name) {
                                    Some(url) => auth_input = Some(AuthInput {
                                        name: name.clone(),
                                        url: url.clone(),
                                        token: String::new(),
                                    }),
                                    None => flash = Some(format!(
                                        "'{name}' has no URL — can't attach auth")),
                                }
                            }
                        }
                        // Edit the highlighted catalog in place (fix a typo,
                        // change URL/token), then re-validate on submit.
                        KeyCode::Char('e') => {
                            if let Some(SettingsItem::CatalogToggle(name)) =
                                items.get(catalog_cursor)
                            {
                                match catalog_locations.get(name) {
                                    Some(url) => {
                                        let token = crate::credentials::credential_for_url(url)
                                            .map(|e| e.token).unwrap_or_default();
                                        add_input = Some(AddCatalogInput::editing(
                                            name.clone(), url.clone(), token));
                                    }
                                    None => flash = Some(format!(
                                        "'{name}' isn't a catalogs.yaml entry — can't edit here")),
                                }
                            }
                        }
                        // Remove the highlighted catalog (with confirm).
                        KeyCode::Char('x') | KeyCode::Delete => {
                            if let Some(SettingsItem::CatalogToggle(name)) =
                                items.get(catalog_cursor)
                            {
                                if catalog_list.len() <= 1 {
                                    flash = Some(
                                        "can't remove the only catalog — add another first".into());
                                } else if catalog_locations.contains_key(name) {
                                    confirm_remove = Some(name.clone());
                                } else {
                                    flash = Some(format!(
                                        "'{name}' isn't a removable catalogs.yaml entry"));
                                }
                            }
                        }
                        KeyCode::Char(' ') | KeyCode::Enter => {
                            match items.get(catalog_cursor) {
                                Some(SettingsItem::CatalogToggle(name)) => {
                                    if unvalidated_catalogs.contains(name) {
                                        // Not enableable while unvalidated:
                                        // re-verify in place; it only turns on
                                        // if it now passes (no file rewrite).
                                        if let Some(url) = catalog_locations.get(name).cloned() {
                                            let nm = name.clone();
                                            revalidate_catalog(
                                                &mut terminal, &nm, &url,
                                                &mut disabled_catalogs,
                                                &mut unvalidated_catalogs);
                                            result = PickerOutcome::Reload;
                                            break;
                                        }
                                    } else {
                                        if !disabled_catalogs.remove(name) {
                                            disabled_catalogs.insert(name.clone());
                                        }
                                        if let Err(e) = crate::settings::write_disabled_catalogs(
                                            &disabled_catalogs)
                                        {
                                            flash = Some(format!(
                                                "could not persist disabled_catalogs: {e}"));
                                        }
                                        // The visible row set just changed out
                                        // from under the main cursor — snap back.
                                        cursor = 0;
                                        scroll = 0;
                                    }
                                }
                                Some(SettingsItem::AddCatalog) => {
                                    add_input = Some(AddCatalogInput::new());
                                }
                                Some(SettingsItem::ColumnToggle(col)) => {
                                    let name = col.name().to_string();
                                    if !disabled_columns.remove(&name) {
                                        disabled_columns.insert(name);
                                    }
                                    if let Err(e) =
                                        crate::settings::write_disabled_columns(&disabled_columns)
                                    {
                                        flash = Some(format!("could not persist disabled_columns: {e}"));
                                    }
                                }
                                Some(SettingsItem::PaletteCycle) => {
                                    let (p, c) = super::theme_palette_curve();
                                    super::set_theme(p.next(), c);
                                }
                                Some(SettingsItem::CurveCycle) => {
                                    let (p, c) = super::theme_palette_curve();
                                    super::set_theme(p, c.next());
                                }
                                Some(SettingsItem::SaveTheme) => {
                                    let (p, c) = super::theme_palette_curve();
                                    flash = Some(match super::save_theme_to_settings() {
                                        Ok(path) => format!(
                                            "saved theme ({}/{}) to {}",
                                            p.name(), c.name(), path.display()),
                                        Err(e) => format!("save failed: {e}"),
                                    });
                                }
                                Some(SettingsItem::ResetDisplay) => {
                                    flash = Some(match super::reset_display_options() {
                                        Ok(()) => {
                                            disabled_columns.clear();
                                            "display options reset to the project standard".to_string()
                                        }
                                        Err(e) => format!("reset failed: {e}"),
                                    });
                                }
                                None => {}
                            }
                        }
                        _ => {}
                    }
                    continue;
                }
                // Any other key clears the theme-knob flash.
                flash = None;

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
                                .filter(|r| row_visible(r, &filter, &disabled_catalogs))
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
                                // the batch made; fresh local facts
                                // (finished downloads) refresh the
                                // knowledge map before cells refill.
                                cache_survey = survey_cache_state(&cache_dir);
                                all_rows = build_rows(entries, &cache_survey);
                                collect_local_knowledge(
                                    entries, &cache_dir, &cache_survey, &mut facet_knowledge);
                                apply_size_cells(&mut all_rows, entries, &facet_knowledge);
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
                                            descriptor_lines = build_descriptor_lines(row, entry, &facet_knowledge, theme_on);
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
                                                            if theme_on { Style::default().fg(theme().error) }
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
                                collect_local_knowledge(
                                    entries, &cache_dir, &cache_survey, &mut facet_knowledge);
                                apply_size_cells(&mut all_rows, entries, &facet_knowledge);
                                let new_visible_idx = all_rows.iter()
                                    .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                                    .filter(|r| row_visible(r, &filter, &disabled_catalogs))
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
                    // No `q`-quit on the main list: every printable
                    // key belongs to the filter (datasets named
                    // "quora"/"query" were unfilterable while `q`
                    // quit). Exit is the Esc cascade or Ctrl-C;
                    // sub-screens keep `q`-close since they have no
                    // filter to collide with.
                    KeyCode::Esc => {
                        // Single-Esc cascade: whichever piece of state
                        // is "in the way" of leaving the picker gets
                        // peeled off first. No double-tap; the user can
                        // just keep tapping Esc until they're out.
                        // Order matters — the topmost overlay (help)
                        // peels first, then multi-select state (most
                        // ephemeral), then list shape, then filter, then exit.
                        if show_help {
                            show_help = false;
                        } else if range_anchor.is_some() {
                            range_anchor = None;
                        } else if !selected.is_empty() {
                            selected.clear();
                        } else if !expanded.is_empty() && !show_all_profiles {
                            expanded.clear();
                        } else if !filter.is_empty() {
                            filter.clear();
                            cursor = 0;
                            scroll = 0;
                        } else {
                            // Nothing left to peel: two presses within
                            // QUIT_CONFIRM quit. The first only arms +
                            // prompts (see the footer). `prev_quit` was
                            // taken at the top of key handling, so any
                            // intervening key cancels the arm.
                            match prev_quit {
                                Some(t) if t.elapsed() < QUIT_CONFIRM => {
                                    result = PickerOutcome::Done;
                                    break;
                                }
                                _ => { quit_armed = Some(std::time::Instant::now()); }
                            }
                        }
                    }
                    KeyCode::Enter => {
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
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
                            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
                            .collect();
                        if let Some(row) = vis.get(cursor)
                            && expanded.remove(&row.dataset) {
                                // Move cursor to the default row of this dataset
                                let new_vis: Vec<&PickerRow> = all_rows.iter()
                                    .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                                    .filter(|r| row_visible(r, &filter, &disabled_catalogs))
                                    .collect();
                                cursor = new_vis.iter().position(|r| r.dataset == row.dataset).unwrap_or(0);
                            }
                    }
                    KeyCode::Right => {
                        // Expand: same as Enter for collapsed datasets
                        let vis: Vec<&PickerRow> = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
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
                            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
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
                            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
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
                            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
                            .count();
                        cursor = (cursor + h.saturating_sub(6)).min(visible_count.saturating_sub(1));
                    }
                    // Same ends-jumping the overlays offer (there via
                    // g/G as well — letters here belong to the
                    // filter, so only the dedicated keys apply).
                    KeyCode::Home => {
                        cursor = 0;
                        scroll = 0;
                    }
                    KeyCode::End => {
                        let visible_count = all_rows.iter()
                            .filter(|r| show_all_profiles || r.profile == "default" || expanded.contains(&r.dataset))
                            .filter(|r| row_visible(r, &filter, &disabled_catalogs))
                            .count();
                        cursor = visible_count.saturating_sub(1);
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

    // Stop the size-probe workers at their next job boundary; the
    // receiver drops with this frame, so any in-flight send also
    // ends its worker.
    size_probe_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    // On final exit (not a reload), leave a persistent note on the console
    // for any catalog still saved-but-unvalidated, so it isn't forgotten.
    if !matches!(result, PickerOutcome::Reload) && !unvalidated_catalogs.is_empty() {
        let mut names: Vec<&str> = unvalidated_catalogs.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        eprintln!();
        eprintln!("⚠ {} catalog(s) saved but disabled — failed validation, needs fixing:",
            names.len());
        for n in names {
            eprintln!("    {n}");
        }
        eprintln!("  In `vectordata explore`, open the config view (Ctrl-G) and press 'e' to edit & re-validate.");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::format_count;

    /// Metric recovery from jvector SimpleMFD filenames — fills the
    /// picker's metric column for catalogs that declare no
    /// attributes.
    #[test]
    fn metric_inferred_from_filename_tokens() {
        use super::infer_metric_from_filename as infer;
        assert_eq!(infer("cap/cap_1m_gt_norm_shuffle_ip_k100.ivecs"), Some("ip".into()));
        assert_eq!(infer("x/some_gt_l2_k100.ivecs"), Some("l2".into()));
        assert_eq!(infer("a_cosine_gt.ivecs"), Some("cosine".into()));
        assert_eq!(infer("a_angular_gt.ivecs"), Some("cosine".into()));
        assert_eq!(infer("dpr_gemma_gt_ip_100.ivecs"), Some("ip".into()));
        // No token → no claim. ("ship.fvecs" must not match "ip".)
        assert_eq!(infer("base_vectors.fvecs"), None);
        assert_eq!(infer("ship.fvecs"), None);
    }

    fn knn_entry(name: &str, base: &str) -> crate::dataset::CatalogEntry {
        crate::dataset::CatalogEntry {
            name: name.to_string(),
            path: base.to_string(),
            dataset_type: "knn_entries.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout: crate::dataset::CatalogLayout {
                attributes: None,
                profiles: Default::default(),
            },
        }
    }

    /// Purge must remove the dataset-keyed directory even when every
    /// facet lives OUTSIDE the dataset's home URL — the previous
    /// facet-URL-prefix match found nothing for such datasets and
    /// silently removed nothing.
    #[test]
    fn purge_removes_dataset_dir_with_out_of_home_facets() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        let ds = cache.join("ds-a");
        std::fs::create_dir_all(&ds).unwrap();
        // Origin records the dataset HOME — facets are elsewhere.
        std::fs::write(ds.join("origin.json"),
            r#"{"source":"https://h/cat/ds-a/","fetched_at":"x"}"#).unwrap();
        std::fs::write(ds.join("query.fvecs"), b"data").unwrap();

        let entry = knn_entry("ds-a", "https://h/cat");
        let (removed, _freed, skipped) = super::purge_cache_for_entry(&entry, cache);
        assert_eq!(removed, vec![ds.clone()], "dataset dir must be purged");
        assert!(skipped.is_empty(), "no skip notes expected: {skipped:?}");
        assert!(!ds.exists());
    }

    /// A same-named directory bound to a DIFFERENT catalog's home
    /// must be skipped with a note, never clobbered.
    #[test]
    fn purge_skips_dataset_dir_bound_to_other_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        let ds = cache.join("ds-a");
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("origin.json"),
            r#"{"source":"https://OTHER/cat/ds-a/","fetched_at":"x"}"#).unwrap();
        std::fs::write(ds.join("query.fvecs"), b"data").unwrap();

        let entry = knn_entry("ds-a", "https://h/cat");
        let (removed, freed, skipped) = super::purge_cache_for_entry(&entry, cache);
        assert!(removed.is_empty(), "foreign-bound dir must not be removed");
        assert_eq!(freed, 0);
        assert_eq!(skipped.len(), 1, "must report the binding mismatch");
        assert!(ds.join("query.fvecs").exists());
    }

    /// The CACHED column's coverage must agree with where the storage
    /// layer actually puts cache files — for home-relative facets AND
    /// for out-of-home facets (cached flat by basename under the
    /// dataset directory). The previous survey reconstructed facet
    /// URLs from on-disk relpaths, which silently missed any facet
    /// whose cache relpath isn't home-relative, reporting fully
    /// cached datasets as missing.
    #[test]
    fn cache_coverage_matches_storage_layout_for_all_facet_shapes() {
        use crate::dataset::profile::{DSProfile, DSView};
        use crate::dataset::source::DSSource;

        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        let ds = cache.join("ds-a");
        std::fs::create_dir_all(ds.join("profiles")).unwrap();
        std::fs::write(ds.join("origin.json"),
            r#"{"source":"https://h/cat/ds-a/","fetched_at":"x"}"#).unwrap();
        // Home-relative facet: profiles/base.fvecs, fully chunk-valid.
        std::fs::write(ds.join("profiles/base.fvecs"), b"data").unwrap();
        std::fs::write(ds.join("profiles/base.fvecs.chunks"), [1u8]).unwrap();
        // Out-of-home facet: cached flat by basename.
        std::fs::write(ds.join("query.fvecs"), b"data").unwrap();
        std::fs::write(ds.join("query.fvecs.chunks"), [1u8]).unwrap();

        let entry = crate::dataset::CatalogEntry {
            name: "ds-a".to_string(),
            path: "https://h/cat".to_string(),
            dataset_type: "knn_entries.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout: crate::dataset::CatalogLayout {
                attributes: None,
                profiles: Default::default(),
            },
        };
        let view_for = |path: &str| DSView {
            source: DSSource {
                path: path.to_string(),
                namespace: None,
                window: Default::default(),
            },
            window: None,
        };
        let mut profile = DSProfile {
            maxk: None,
            base_count: None,
            partition: false,
            views: indexmap::IndexMap::new(),
        };
        profile.views.insert("base_vectors".into(),
            view_for("https://h/cat/ds-a/profiles/base.fvecs"));
        profile.views.insert("query_vectors".into(),
            view_for("https://h/cat/shared/query.fvecs"));
        profile.views.insert("neighbor_indices".into(),
            view_for("https://h/cat/shared/missing.ivecs"));

        let survey = super::survey_cache_state(cache);
        let (valid, total) = super::profile_cache_coverage(
            &entry, &profile, &ds, &survey);
        // base (1 chunk) + query (1 chunk) cached; missing.ivecs
        // contributes 1 uncached unit.
        assert_eq!((valid, total), (2, 3),
            "home-relative and flat-basename facets must both be found");
    }

    fn test_row(dataset: &str, profile: &str) -> super::PickerRow {
        super::PickerRow {
            dataset: dataset.to_string(),
            catalog: "0".to_string(),
            profile: profile.to_string(),
            base_count: 0,
            metric: String::new(),
            facets: String::new(),
            count: String::new(),
            size: String::new(),
            cache_status: "—".to_string(),
            access: crate::AccessMode::FullTransfer,
        }
    }

    fn view_for(path: &str) -> crate::dataset::profile::DSView {
        crate::dataset::profile::DSView {
            source: crate::dataset::source::DSSource {
                path: path.to_string(),
                namespace: None,
                window: Default::default(),
            },
            window: None,
        }
    }

    fn profile_with_base(path: &str) -> crate::dataset::profile::DSProfile {
        let mut profile = crate::dataset::profile::DSProfile {
            maxk: None,
            base_count: None,
            partition: false,
            views: indexmap::IndexMap::new(),
        };
        profile.views.insert("base_vectors".into(), view_for(path));
        profile
    }

    /// COUNT and SIZE must answer from local knowledge wherever
    /// possible: a window IS the record count (and window × bpr the
    /// byte span), and a complete local copy with no chunk sidecar
    /// (finished download) reports from its real file — no survey
    /// entry, no network. vvec gets bytes but no record math.
    #[test]
    fn size_cells_from_local_knowledge() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        let ds = cache.join("ds-a");
        std::fs::create_dir_all(&ds).unwrap();
        // Complete sidecar-less local file: dim=2 f32 → 12 bytes per
        // record, 24 bytes on disk → 2 records.
        let mut bytes = Vec::new();
        for _ in 0..2 {
            bytes.extend_from_slice(&2i32.to_le_bytes());
            bytes.extend_from_slice(&[0u8; 8]);
        }
        std::fs::write(ds.join("base.fvecs"), &bytes).unwrap();
        std::fs::write(ds.join("base.ivvecs"), [0u8; 64]).unwrap();

        let mut entry = knn_entry("ds-a", "https://h/cat");
        entry.layout.profiles.profiles.insert(
            "default".into(), profile_with_base("https://h/cat/ds-a/base.fvecs"));
        entry.layout.profiles.profiles.insert(
            "win".into(), profile_with_base("https://h/cat/ds-a/base.fvecs[0..1)"));
        entry.layout.profiles.profiles.insert(
            "vv".into(), profile_with_base("https://h/cat/ds-a/base.ivvecs"));
        let entries = vec![entry];

        let survey = std::collections::HashMap::new();
        let mut knowledge = std::collections::HashMap::new();
        super::collect_local_knowledge(&entries, cache, &survey, &mut knowledge);

        let mut rows = vec![
            test_row("ds-a", "default"),
            test_row("ds-a", "win"),
            test_row("ds-a", "vv"),
        ];
        super::apply_size_cells(&mut rows, &entries, &knowledge);

        assert_eq!(rows[0].count, format_count(2));
        assert_eq!(rows[0].size, crate::explore::format_bytes_short(24));
        // Window: 1 record, shares the default base (`*` on count),
        // and SIZE is the window's byte span — 1 × 12 — not the
        // whole shared file.
        assert_eq!(rows[1].count, format!("{}*", format_count(1)));
        assert_eq!(rows[1].size, crate::explore::format_bytes_short(12));
        // vvec: no record math, bytes still report.
        assert_eq!(rows[2].count, "");
        assert_eq!(rows[2].size, crate::explore::format_bytes_short(64));

        // The details overlays read the same knowledge: cardinality
        // and bytes for the uniform facet.
        let stat = super::facet_stat_string(
            &entries[0],
            entries[0].layout.profiles.profile("default").unwrap()
                .views.get("base_vectors").unwrap(),
            &knowledge);
        assert_eq!(stat.as_deref(), Some(format!(
            "{} records, {}", format_count(2),
            crate::explore::format_bytes_short(24)).as_str()));
    }

    /// Probe-job collection: one job per distinct remote facet URL,
    /// non-remote and already-answered facets skipped, and pending
    /// facets surface as `…` cells until their answer merges in.
    #[test]
    fn size_probe_jobs_dedupe_and_mark_pending() {
        let mut a = knn_entry("ds-a", "https://h/cat");
        a.layout.profiles.profiles.insert(
            "default".into(), profile_with_base("https://h/cat/ds-a/base.fvecs"));
        a.layout.profiles.profiles.insert(
            "alt".into(), profile_with_base("https://h/cat/ds-a/base.fvecs"));
        let mut b = knn_entry("ds-b", "https://h/cat");
        b.layout.profiles.profiles.insert(
            "default".into(), profile_with_base("/local/base.fvecs"));
        let mut c = knn_entry("ds-c", "https://h/cat");
        c.layout.profiles.profiles.insert(
            "default".into(), profile_with_base("https://h/cat/ds-c/base.fvecs"));
        let entries = vec![a, b, c];

        // ds-c is already fully answered locally.
        let mut knowledge = std::collections::HashMap::new();
        knowledge.insert("https://h/cat/ds-c/base.fvecs".to_string(),
            super::FacetKnowledge {
                bytes: Some(1_200), bpr: Some(12), pending: false, probed: false,
            });

        let mut jobs = super::collect_size_probe_jobs(&entries, &mut knowledge);
        assert_eq!(jobs.len(), 1, "one job for the one unanswered remote URL");
        let job = jobs.pop().unwrap();
        assert_eq!(job.url, "https://h/cat/ds-a/base.fvecs");
        assert!(job.uniform);

        let mut rows = vec![
            test_row("ds-a", "default"),
            test_row("ds-a", "alt"),
            test_row("ds-b", "default"),
            test_row("ds-c", "default"),
        ];
        super::apply_size_cells(&mut rows, &entries, &knowledge);
        assert_eq!(rows[0].count, super::SIZE_PENDING);
        assert_eq!(rows[0].size, super::SIZE_PENDING);
        assert_eq!(rows[1].size, super::SIZE_PENDING);
        assert_eq!(rows[2].count, "", "non-remote, unknowable: blank");
        assert_eq!(rows[2].size, "");
        assert_eq!(rows[3].count, format_count(100));
        assert_eq!(rows[3].size, crate::explore::format_bytes_short(1_200));

        // A probe answer merges in: both ds-a rows resolve, and the
        // profile sharing the default base gets its `*` markers
        // (whole shared file counted in SIZE).
        let slot = knowledge.get_mut("https://h/cat/ds-a/base.fvecs").unwrap();
        slot.bytes = Some(2_400);
        slot.bpr = Some(12);
        slot.pending = false;
        slot.probed = true;
        super::apply_size_cells(&mut rows, &entries, &knowledge);
        assert_eq!(rows[0].count, format_count(200));
        assert_eq!(rows[0].size, crate::explore::format_bytes_short(2_400));
        assert_eq!(rows[1].count, format!("{}*", format_count(200)));
        assert_eq!(rows[1].size,
            format!("{}*", crate::explore::format_bytes_short(2_400)));
    }

    /// The ACCESS column must say "local" the moment a facet is
    /// fully on disk — including knn_entries-style catalogs whose
    /// source paths are absolute URLs. The previous check joined the
    /// raw source path under the dataset dir (never exists for a
    /// URL), so fully downloaded datasets stayed labelled "chunked".
    #[test]
    fn fully_cached_knn_dataset_classifies_local() {
        use crate::AccessMode;
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        let ds = cache.join("ds-a");
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("origin.json"),
            r#"{"source":"https://h/cat/ds-a/","fetched_at":"x"}"#).unwrap();
        std::fs::write(ds.join("base.fvecs"), b"data").unwrap();

        let entry = knn_entry("ds-a", "https://h/cat");
        let profile = profile_with_base("https://h/cat/ds-a/base.fvecs");

        // Chunk-store copy, every chunk valid → local.
        std::fs::write(ds.join("base.fvecs.chunks"), [1u8, 1]).unwrap();
        let survey = super::survey_cache_state(cache);
        assert_eq!(
            super::profile_access_mode(&entry, &ds, Some(&profile), &survey),
            AccessMode::Local);

        // One chunk missing → still sparse remote access.
        std::fs::write(ds.join("base.fvecs.chunks"), [1u8, 0]).unwrap();
        let survey = super::survey_cache_state(cache);
        assert_eq!(
            super::profile_access_mode(&entry, &ds, Some(&profile), &survey),
            AccessMode::MerkleChunked);

        // Sidecar-less complete copy (finished download) → local.
        std::fs::remove_file(ds.join("base.fvecs.chunks")).unwrap();
        let survey = super::survey_cache_state(cache);
        assert_eq!(
            super::profile_access_mode(&entry, &ds, Some(&profile), &survey),
            AccessMode::Local);

        // Untouched facet → the remote classification stands.
        let absent = profile_with_base("https://h/cat/ds-a/missing.fvecs");
        assert_eq!(
            super::profile_access_mode(&entry, &ds, Some(&absent), &survey),
            AccessMode::MerkleChunked);
    }

    /// A failed probe settles the facet: pending markers clear and
    /// the cells go blank instead of showing `…` forever.
    #[test]
    fn failed_probe_clears_pending_cells() {
        let mut a = knn_entry("ds-a", "https://h/cat");
        a.layout.profiles.profiles.insert(
            "default".into(), profile_with_base("https://h/cat/ds-a/base.fvecs"));
        let entries = vec![a];
        let mut knowledge = std::collections::HashMap::new();
        knowledge.insert("https://h/cat/ds-a/base.fvecs".to_string(),
            super::FacetKnowledge {
                bytes: None, bpr: None, pending: false, probed: true,
            });
        let mut rows = vec![test_row("ds-a", "default")];
        super::apply_size_cells(&mut rows, &entries, &knowledge);
        assert_eq!(rows[0].count, "");
        assert_eq!(rows[0].size, "");
    }

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
