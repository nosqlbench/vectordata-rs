// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata datasets` (no subcommand) — TUI catalog browser.
//!
//! Launches a two-pane terminal UI: the left pane is a scrolling
//! list of datasets pulled from every configured catalog; the
//! right pane shows the highlighted dataset's metadata (profile
//! count, sizes, dimensions, vtype, distance metric, facet URLs).
//!
//! Keybindings:
//!   - `↑`/`↓` / `k`/`j`  — move the highlight
//!   - `Home`/`End`        — jump to top / bottom
//!   - `PgUp`/`PgDn`       — page move
//!   - `q` / `Esc`         — quit
//!
//! Designed to be the fallback when `vectordata datasets` is run
//! with no subcommand — the user sees a usable browser instead of
//! a wall of `--help` text.

use std::io::{stdout, Stdout};
use std::time::Duration;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Terminal,
};
use unicode_width::UnicodeWidthStr;

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::CatalogSources;
use crate::dataset::CatalogEntry;

/// Entry point invoked when the user runs `vectordata datasets`
/// with no subcommand. Loads every configured catalog and hands
/// over to the TUI event loop.
pub fn run(configdir: &str, extra_catalogs: &[String], at: &[String]) -> i32 {
    let sources = build_sources(configdir, extra_catalogs, at);
    if sources.is_empty() {
        eprintln!("No catalog sources configured.");
        eprintln!("Add one with:  vectordata datasets config add-catalog <URL-or-path>");
        eprintln!("Or use --catalog/--at for one-off access.");
        return 1;
    }
    let catalog = Catalog::of(&sources);
    if catalog.is_empty() {
        eprintln!("No datasets found in any configured catalog.");
        return 1;
    }

    match tui_loop(&catalog) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("browser error: {e}");
            1
        }
    }
}

fn build_sources(configdir: &str, extra: &[String], at: &[String]) -> CatalogSources {
    let mut s = CatalogSources::new();
    if !at.is_empty() {
        s = s.add_catalogs(at);
    } else {
        s = s.configure(configdir);
        if !extra.is_empty() {
            s = s.add_catalogs(extra);
        }
    }
    s
}

/// Set up the terminal, drive the event loop, then restore.
fn tui_loop(catalog: &Catalog) -> Result<(), String> {
    enable_raw_mode().map_err(|e| format!("enable raw mode: {e}"))?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture)
        .map_err(|e| format!("enter alt screen: {e}"))?;
    let backend = CrosstermBackend::new(out);
    let mut terminal =
        Terminal::new(backend).map_err(|e| format!("terminal init: {e}"))?;

    let entries = catalog.datasets();
    let mut state = BrowserState::new(entries.len());
    let result = event_loop(&mut terminal, entries, &mut state);

    // Restore terminal — best-effort so even a panic in the inner
    // loop leaves the terminal usable.
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture);
    let _ = terminal.show_cursor();
    result
}

struct BrowserState {
    list_state: ListState,
    count: usize,
}

impl BrowserState {
    fn new(count: usize) -> Self {
        let mut list_state = ListState::default();
        if count > 0 { list_state.select(Some(0)); }
        BrowserState { list_state, count }
    }

    fn move_selection(&mut self, delta: i64) {
        if self.count == 0 { return; }
        let cur = self.list_state.selected().unwrap_or(0) as i64;
        let next = (cur + delta).clamp(0, self.count as i64 - 1);
        self.list_state.select(Some(next as usize));
    }

    fn jump(&mut self, idx: usize) {
        if self.count == 0 { return; }
        self.list_state.select(Some(idx.min(self.count - 1)));
    }
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    entries: &[CatalogEntry],
    state: &mut BrowserState,
) -> Result<(), String> {
    loop {
        terminal
            .draw(|f| draw(f, entries, state))
            .map_err(|e| format!("draw: {e}"))?;

        if !event::poll(Duration::from_millis(150)).map_err(|e| format!("poll: {e}"))? {
            continue;
        }
        let ev = event::read().map_err(|e| format!("read: {e}"))?;
        let Event::Key(key) = ev else { continue };
        if key.kind != KeyEventKind::Press { continue; }
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => return Ok(()),
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => return Ok(()),
            (KeyCode::Up, _) | (KeyCode::Char('k'), _) => state.move_selection(-1),
            (KeyCode::Down, _) | (KeyCode::Char('j'), _) => state.move_selection(1),
            (KeyCode::PageUp, _) => state.move_selection(-10),
            (KeyCode::PageDown, _) => state.move_selection(10),
            (KeyCode::Home, _) | (KeyCode::Char('g'), _) => state.jump(0),
            (KeyCode::End, _) | (KeyCode::Char('G'), _) => state.jump(usize::MAX),
            _ => {}
        }
    }
}

fn draw(
    f: &mut ratatui::Frame,
    entries: &[CatalogEntry],
    state: &mut BrowserState,
) {
    // Layout: header (1) + body (split 35%/65%) + footer (1).
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(f.area());

    let header = format!(
        " vectordata datasets — {} dataset{}",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" },
    );
    f.render_widget(
        Paragraph::new(header).style(Style::default().add_modifier(Modifier::BOLD)),
        outer[0],
    );

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(outer[1]);

    // Left: list of datasets.
    let items: Vec<ListItem> = entries
        .iter()
        .map(|e| {
            let n_profiles = e.profile_count();
            let label = format!("{}  ({} profile{})", e.name, n_profiles, if n_profiles == 1 { "" } else { "s" });
            ListItem::new(label)
        })
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Datasets "))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, body[0], &mut state.list_state);

    // Right: details for the highlighted dataset.
    let detail_text: Vec<Line> = match state.list_state.selected().and_then(|i| entries.get(i)) {
        Some(entry) => render_detail(entry),
        None => vec![Line::from(Span::raw("(no dataset selected)"))],
    };
    let detail = Paragraph::new(detail_text)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(" Details "));
    f.render_widget(detail, body[1]);

    // Footer with keybinding hint.
    let footer = " ↑/↓ or j/k  move    PgUp/PgDn  page    Home/End  jump    q/Esc  quit ";
    f.render_widget(
        Paragraph::new(footer).style(Style::default().add_modifier(Modifier::DIM)),
        outer[2],
    );
}

fn render_detail(entry: &CatalogEntry) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        entry.name.clone(),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::styled(
        entry.path.clone(),
        Style::default().add_modifier(Modifier::DIM),
    )));
    lines.push(Line::raw(""));

    // Attribute block (model / metric / dataset_type when present).
    let attrs = entry.layout.attributes.as_ref();
    if let Some(a) = attrs {
        if let Some(ref m) = a.model {
            lines.push(field_line("model", m));
        }
        if let Some(ref d) = a.distance_function {
            lines.push(field_line("distance", d));
        }
        if let Some(ref n) = a.notes {
            lines.push(field_line("notes", n));
        }
        lines.push(Line::raw(""));
    }
    lines.push(field_line("dataset_type", &entry.dataset_type));

    // Profile summaries.
    let profile_names = entry.profile_names();
    lines.push(field_line(
        "profiles",
        &if profile_names.is_empty() {
            "(none)".to_string()
        } else {
            profile_names.join(", ")
        },
    ));
    lines.push(Line::raw(""));

    // Per-profile detail: facets + base_count + maxk.
    for pname in &profile_names {
        let Some(profile) = entry.layout.profiles.profile(pname) else { continue };
        lines.push(Line::from(Span::styled(
            format!("  ▸ {}", pname),
            Style::default().add_modifier(Modifier::BOLD),
        )));
        if let Some(bc) = profile.base_count {
            lines.push(indented_field("base_count", &fmt_count(bc)));
        }
        if let Some(maxk) = profile.maxk {
            lines.push(indented_field("maxk", &maxk.to_string()));
        }
        if !profile.views.is_empty() {
            let view_names: Vec<&str> = profile.views.keys().map(|s| s.as_str()).collect();
            lines.push(indented_field("facets", &view_names.join(", ")));
        }
        lines.push(Line::raw(""));
    }
    lines
}

fn field_line(label: &str, value: &str) -> Line<'static> {
    let label_owned = format!("{}: ", label);
    let pad_width = 14_usize.saturating_sub(UnicodeWidthStr::width(label_owned.as_str()));
    Line::from(vec![
        Span::styled(label_owned, Style::default().add_modifier(Modifier::DIM)),
        Span::raw(" ".repeat(pad_width)),
        Span::raw(value.to_string()),
    ])
}

fn indented_field(label: &str, value: &str) -> Line<'static> {
    let label_owned = format!("    {}: ", label);
    let pad_width = 20_usize.saturating_sub(UnicodeWidthStr::width(label_owned.as_str()));
    Line::from(vec![
        Span::styled(label_owned, Style::default().add_modifier(Modifier::DIM)),
        Span::raw(" ".repeat(pad_width)),
        Span::raw(value.to_string()),
    ])
}

/// Compact human-readable size — "12.3M", "4.5B", etc.
fn fmt_count(n: u64) -> String {
    let nf = n as f64;
    if nf < 1_000.0 { return n.to_string(); }
    if nf < 1_000_000.0 { return format!("{:.1}K", nf / 1_000.0); }
    if nf < 1_000_000_000.0 { return format!("{:.1}M", nf / 1_000_000.0); }
    format!("{:.2}B", nf / 1_000_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_count_thresholds() {
        assert_eq!(fmt_count(0), "0");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1_500), "1.5K");
        assert_eq!(fmt_count(2_500_000), "2.5M");
        assert_eq!(fmt_count(1_200_000_000), "1.20B");
    }

    /// Browser state navigation: cursor moves stay clamped to the
    /// list bounds and ignore moves on an empty list. Exercises
    /// the pure key-handling logic without touching the TUI.
    #[test]
    fn browser_state_clamps_at_bounds() {
        let mut s = BrowserState::new(3);
        assert_eq!(s.list_state.selected(), Some(0));
        s.move_selection(-1);
        assert_eq!(s.list_state.selected(), Some(0), "clamps at 0");
        s.move_selection(2);
        assert_eq!(s.list_state.selected(), Some(2));
        s.move_selection(10);
        assert_eq!(s.list_state.selected(), Some(2), "clamps at count-1");
        s.jump(0);
        assert_eq!(s.list_state.selected(), Some(0));
        s.jump(usize::MAX);
        assert_eq!(s.list_state.selected(), Some(2));
    }

    #[test]
    fn browser_state_empty_list_noop() {
        let mut s = BrowserState::new(0);
        assert_eq!(s.list_state.selected(), None);
        s.move_selection(5);
        s.jump(0);
        assert_eq!(s.list_state.selected(), None);
    }
}
