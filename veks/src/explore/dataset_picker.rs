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
use vectordata::dataset::CatalogEntry;

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
}

impl PickerRow {
    fn specifier(&self) -> String {
        format!("{}:{}", self.dataset, self.profile)
    }
}

fn build_rows(entries: &[CatalogEntry]) -> Vec<PickerRow> {
    let cache_dir = crate::pipeline::commands::config::configured_cache_dir_or_exit();
    let mut rows = Vec::new();
    for entry in entries {
        let metric = entry.layout.attributes.as_ref()
            .and_then(|a| a.distance_function.as_deref())
            .unwrap_or("")
            .to_string();

        let ds_cache = cache_dir.join(&entry.name);

        for profile_name in entry.profile_names() {
            let profile = entry.layout.profiles.profile(profile_name);
            let bc = vectordata::dataset::profile::profile_sort_key(
                profile_name, profile.and_then(|p| p.base_count));
            // Show count from profile's base_count (partition profiles, sized profiles)
            let explicit_count = profile.and_then(|p| p.base_count);
            let size = match explicit_count {
                Some(c) if c > 0 => format_count(c),
                _ => String::new(),
            };

            let facets = profile.map(|p| facet_indicators(&p.views)).unwrap_or_default();

            // Per-profile cache: check local files or .mrkl download state
            let ds_workspace = ds_cache.clone(); // download cache mirrors workspace layout
            let (valid_chunks, total_chunks) = if let Some(p) = profile {
                profile_cache_coverage(&ds_cache, &ds_workspace, p)
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

            rows.push(PickerRow {
                dataset: entry.name.clone(),
                profile: profile_name.to_string(),
                base_count: bc,
                metric: metric.clone(),
                facets,
                size,
                cache_status,
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

/// Per-profile cache coverage: check `.mrkl` files for each source file
/// that this profile references.
fn profile_cache_coverage(
    ds_cache: &std::path::Path,
    ds_workspace: &std::path::Path,
    profile: &vectordata::dataset::profile::DSProfile,
) -> (u32, u32) {
    use vectordata::merkle::MerkleState;

    let mut valid = 0u32;
    let mut total = 0u32;

    for (_facet, view) in &profile.views {
        let source_path = &view.source.path;
        if source_path.is_empty() { continue; }

        // Strip window notation: "base.fvec[0..1000)" → "base.fvec"
        let clean = if let Some(bracket) = source_path.find(|c: char| c == '[' || c == '(') {
            &source_path[..bracket]
        } else {
            source_path.as_str()
        };

        // First: check if the actual file exists in the workspace.
        // For locally-prepared datasets, files exist directly without
        // merkle download tracking.
        let local_path = ds_workspace.join(clean);
        if local_path.exists() {
            // File exists locally — count as fully cached (1 chunk = 1 valid)
            valid += 1;
            total += 1;
            continue;
        }

        // Second: check for .mrkl download state in the cache
        if ds_cache.is_dir() {
            let mrkl_path = ds_cache.join(format!("{}.mrkl", clean));
            if let Ok(state) = MerkleState::load(&mrkl_path) {
                valid += state.valid_count();
                total += state.shape().total_chunks;
                continue;
            }
        }

        // File not present locally and no cache state — count as missing
        total += 1;
    }

    (valid, total)
}

/// Build a compact facet indicator string from a profile's view keys.
///
/// Letters: B(base) Q(query) G(ground-truth indices) D(distances)
///          M(metadata) P(predicates) R(results) F(filtered-knn)
fn facet_indicators(views: &indexmap::IndexMap<String, vectordata::dataset::DSView>) -> String {
    use vectordata::dataset::StandardFacet;

    let mut out = String::with_capacity(10);
    let has = |facet: StandardFacet| -> bool {
        views.contains_key(facet.key())
    };

    if has(StandardFacet::BaseVectors)            { out.push('B'); }
    if has(StandardFacet::QueryVectors)           { out.push('Q'); }
    if has(StandardFacet::NeighborIndices)         { out.push('G'); }
    if has(StandardFacet::NeighborDistances)       { out.push('D'); }
    if has(StandardFacet::MetadataContent)         { out.push('M'); }
    if has(StandardFacet::MetadataPredicates)      { out.push('P'); }
    if has(StandardFacet::MetadataResults)         { out.push('R'); }
    if has(StandardFacet::FilteredNeighborIndices) { out.push('F'); }

    out
}

fn format_bytes(n: u64) -> String {
    if n >= 1_073_741_824 { format!("{:.1} GiB", n as f64 / 1_073_741_824.0) }
    else if n >= 1_048_576 { format!("{:.1} MiB", n as f64 / 1_048_576.0) }
    else if n >= 1024 { format!("{:.1} KiB", n as f64 / 1024.0) }
    else { format!("{} B", n) }
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

/// Run the interactive dataset picker. Returns `Some("dataset:profile")` on
/// selection, or `None` if the user cancels (q/Esc).
pub fn run_picker() -> Option<String> {
    // Load catalog
    let sources = CatalogSources::new().configure_default();
    if sources.is_empty() {
        eprintln!("No catalog sources configured.");
        eprintln!("Add one with: veks datasets config add-catalog <URL>");
        return None;
    }
    let catalog = Catalog::of(&sources);
    let entries = catalog.datasets();
    if entries.is_empty() {
        eprintln!("No datasets found in configured catalogs.");
        return None;
    }

    let all_rows = build_rows(entries);
    if all_rows.is_empty() {
        eprintln!("No dataset profiles found.");
        return None;
    }

    enable_raw_mode().ok()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen).ok()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).ok()?;

    let mut filter = String::new();
    let mut cursor: usize = 0;
    let mut scroll: usize = 0;
    let mut show_all_profiles = false;
    let mut expanded: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut last_esc: Option<std::time::Instant> = None;
    let mut show_help = false;
    let result;

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

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),  // filter input
                    Constraint::Min(5),    // list
                    Constraint::Length(3),  // detail panel
                    Constraint::Length(1),  // footer
                ]).split(frame.area());

            // Filter input
            let filter_display = format!(" Filter: {}█", filter);
            frame.render_widget(
                Paragraph::new(filter_display)
                    .block(Block::default().borders(Borders::ALL)
                        .title(format!(" Select Dataset — {} shown ({}) ",
                            visible.len(),
                            if show_all_profiles { "all profiles" } else { "default only — Tab for all" }))),
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
                    Line::from(Span::styled(" Dataset Picker — Keyboard Shortcuts", Style::default().fg(Color::Cyan))),
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
                    Line::from(" Exit"),
                    Line::from("   Esc                   Collapse → clear filter → double-tap to exit"),
                    Line::from("   q / Ctrl-C            Quit immediately"),
                    Line::from("   ?                     Toggle this help"),
                ];
                frame.render_widget(
                    Paragraph::new(help).block(Block::default().borders(Borders::ALL).title(" Help ")),
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
            let size_w = 10;
            let cache_w = 18;

            // Header and data rows share the same cell widths so columns
            // line up edge-to-edge. Data rows get their leading whitespace
            // from the tree prefix (3–4 display columns depending on
            // shape); the header pads DATASET to the same `name_w` so
            // the PROFILE column lands at identical offsets in both.
            let mut lines = vec![Line::from(vec![
                Span::styled(pad_display("DATASET", name_w), Style::default().fg(Color::DarkGray)),
                Span::styled(pad_display("PROFILE", prof_w), Style::default().fg(Color::DarkGray)),
                Span::styled(pad_display("FACETS", facet_w), Style::default().fg(Color::DarkGray)),
                Span::styled(pad_display("METRIC", metric_w), Style::default().fg(Color::DarkGray)),
                Span::styled(pad_display("SIZE", size_w), Style::default().fg(Color::DarkGray)),
                Span::styled(pad_display("CACHED", cache_w), Style::default().fg(Color::DarkGray)),
            ])];

            for (i, row) in visible.iter().enumerate().skip(scroll).take(list_height.saturating_sub(1)) {
                let selected = i == cursor;
                let (fg, bg) = if selected {
                    (Color::Black, Color::Cyan)
                } else if row.cache_status != "—" {
                    (Color::Green, Color::Reset)
                } else {
                    (Color::White, Color::Reset)
                };
                let style = Style::default().fg(fg).bg(bg);
                let cache_style = if selected {
                    style
                } else if row.cache_status != "—" {
                    Style::default().fg(Color::Green).bg(bg)
                } else {
                    Style::default().fg(Color::DarkGray).bg(bg)
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

                lines.push(Line::from(vec![
                    Span::styled(name_cell, style),
                    Span::styled(pad_display(&row.profile, prof_w), style),
                    Span::styled(pad_display(&row.facets, facet_w), style),
                    Span::styled(pad_display(&row.metric, metric_w), style),
                    Span::styled(pad_display(&row.size, size_w), style),
                    Span::styled(pad_display(&row.cache_status, cache_w), cache_style),
                ]));
            }

            frame.render_widget(
                Paragraph::new(lines)
                    .block(Block::default().borders(Borders::ALL)),
                chunks[1],
            );

            } // close help/list else

            // Detail panel for selected row
            let detail_line = if let Some(row) = visible.get(cursor) {
                let mut spans = vec![
                    Span::styled(format!(" {}:{} ", row.dataset, row.profile), Style::default().fg(Color::Cyan)),
                    Span::raw("  "),
                ];
                // Facet details with colors
                let facet_details: &[(&str, char, Color)] = &[
                    ("Base",     'B', Color::Green),
                    ("Query",    'Q', Color::Green),
                    ("GT",       'G', Color::Yellow),
                    ("Dist",     'D', Color::Yellow),
                    ("Meta",     'M', Color::Magenta),
                    ("Pred",     'P', Color::Magenta),
                    ("Results",  'R', Color::Magenta),
                    ("Filtered", 'F', Color::Red),
                ];
                for (label, code, color) in facet_details {
                    if row.facets.contains(*code) {
                        spans.push(Span::styled(format!("{} ", label), Style::default().fg(*color)));
                    } else {
                        spans.push(Span::styled(format!("{} ", label), Style::default().fg(Color::DarkGray)));
                    }
                }
                Line::from(spans)
            } else {
                Line::from("")
            };
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(""),
                    detail_line,
                ]).block(Block::default().borders(Borders::TOP)),
                chunks[2],
            );

            // Footer
            frame.render_widget(
                Paragraph::new(Span::styled(
                    " ↑↓: navigate | Enter/→: expand | ←: collapse | Tab: all profiles | type to filter | Esc×2: exit | q: quit",
                    Style::default().fg(Color::DarkGray))),
                chunks[3],
            );
        }).ok()?;

        // Events
        if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    result = None;
                    break;
                }
                match key.code {
                    KeyCode::Char('q') => {
                        result = None;
                        break;
                    }
                    KeyCode::Esc => {
                        // Priority: collapse expanded → clear filter → double-tap to exit
                        if !expanded.is_empty() && !show_all_profiles {
                            // Collapse all expanded trees
                            expanded.clear();
                            last_esc = None;
                        } else if !filter.is_empty() {
                            filter.clear();
                            cursor = 0;
                            scroll = 0;
                            last_esc = None;
                        } else {
                            // Double-tap to exit
                            let now = std::time::Instant::now();
                            if let Some(prev) = last_esc {
                                if now.duration_since(prev).as_millis() < 500 {
                                    result = None;
                                    break;
                                }
                            }
                            last_esc = Some(now);
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
                                result = Some(row.specifier());
                                break;
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
