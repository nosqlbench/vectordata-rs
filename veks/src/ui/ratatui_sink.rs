// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rich terminal [`UiSink`] using ratatui's `Viewport::Inline` mode.
//!
//! Renders progress bars and a resource status line in a fixed region at the
//! bottom of the terminal. Log lines scroll above the fixed area via
//! `insert_before()`.
//!
//! Events are sent over a channel to a dedicated render thread that owns the
//! ratatui `Terminal`. This keeps the `RatatuiSink` itself `Send + Sync`.

use std::collections::HashMap;
use std::io::{self, Stdout};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::cursor;
use crossterm::event::{self as ct_event, Event as CtEvent, KeyCode, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Widget},
};

use super::event::{ProgressId, ProgressKind, UiEvent};
use super::sink::UiSink;

/// Internal message type for the render thread.
///
/// Wraps `UiEvent` with a `Shutdown` sentinel for clean teardown.
enum RenderMsg {
    Event(UiEvent),
    Shutdown,
}

/// Rich terminal sink using ratatui `Viewport::Inline`.
///
/// Thread-safe: holds only a channel sender and an atomic id counter.
/// The actual terminal rendering happens on a background thread.
pub struct RatatuiSink {
    tx: mpsc::Sender<RenderMsg>,
    next_id: AtomicU32,
    render_thread: Option<thread::JoinHandle<()>>,
}

impl RatatuiSink {
    /// Create a new ratatui-based sink.
    ///
    /// `inline_height` is the number of terminal lines reserved for the
    /// fixed progress region (typically 4-8).
    pub fn new(inline_height: u16) -> io::Result<Self> {
        let (tx, rx) = mpsc::channel::<RenderMsg>();

        let handle = thread::Builder::new()
            .name("ratatui-render".into())
            .spawn(move || {
                if let Err(e) = render_loop(rx, inline_height) {
                    eprintln!("ratatui render error: {}", e);
                }
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(RatatuiSink {
            tx,
            next_id: AtomicU32::new(0),
            render_thread: Some(handle),
        })
    }
}

impl Drop for RatatuiSink {
    fn drop(&mut self) {
        let _ = self.tx.send(RenderMsg::Shutdown);
        if let Some(h) = self.render_thread.take() {
            let _ = h.join();
        }
    }
}

impl UiSink for RatatuiSink {
    fn send(&self, event: UiEvent) {
        let _ = self.tx.send(RenderMsg::Event(event));
    }

    fn next_progress_id(&self) -> ProgressId {
        ProgressId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

// ── Render thread internals ────────────────────────────────────────────

/// State for a single tracked progress indicator.
struct BarState {
    kind: ProgressKind,
    total: u64,
    position: u64,
    label: String,
    message: String,
    created_at: Instant,
}

/// Ring buffer for time-series data, storing the most recent `capacity` samples.
///
/// Each sample is stored as `(x, y)` where `x` is the sample index (time)
/// and `y` is the value. This format is directly usable by ratatui's `Chart`.
struct MetricsHistory {
    data: Vec<(f64, f64)>,
    capacity: usize,
    sample_count: u64,
    max_value: f64,
}

impl MetricsHistory {
    fn new(capacity: usize) -> Self {
        MetricsHistory {
            data: Vec::with_capacity(capacity),
            capacity,
            sample_count: 0,
            max_value: 1.0,
        }
    }

    fn push(&mut self, value: f64) {
        self.sample_count += 1;
        if self.data.len() >= self.capacity {
            self.data.remove(0);
        }
        self.data.push((self.sample_count as f64, value));
        if value > self.max_value {
            self.max_value = value;
        }
    }

    fn as_slice(&self) -> &[(f64, f64)] {
        &self.data
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn x_bounds(&self) -> [f64; 2] {
        if self.data.is_empty() {
            return [0.0, 1.0];
        }
        let start = self.data.first().map(|p| p.0).unwrap_or(0.0);
        let end = self.data.last().map(|p| p.0).unwrap_or(1.0);
        [start, end.max(start + 1.0)]
    }

    fn y_bounds(&self) -> [f64; 2] {
        [0.0, self.max_value * 1.25] // 25% headroom so the line isn't clipped at the top
    }
}

/// Mutable state owned by the render thread.
struct RenderState {
    bars: HashMap<ProgressId, BarState>,
    /// Insertion-ordered list of active progress ids.
    bar_order: Vec<ProgressId>,
    resource_status: String,
    /// Context label (dataset name, profile, step).
    context_label: String,
    /// RSS history for chart (values in MB).
    rss_history: MetricsHistory,
    /// Thread count history for chart.
    thread_history: MetricsHistory,
    /// CPU usage history for chart (total percentage).
    cpu_history: MetricsHistory,
    /// I/O read rate history (MB/s).
    io_read_history: MetricsHistory,
    /// I/O write rate history (MB/s).
    io_write_history: MetricsHistory,
    /// Page fault rate history (major faults/s).
    pgfault_history: MetricsHistory,
    /// Previous cumulative I/O read bytes (for delta computation).
    prev_io_read: f64,
    /// Previous cumulative I/O write bytes.
    prev_io_write: f64,
    /// Previous cumulative major fault count.
    prev_major_faults: f64,
    /// RSS ceiling in MB (from resource budget), for chart reference line.
    rss_ceiling_mb: f64,
    /// Thread ceiling (from resource budget).
    thread_ceiling: f64,
    /// CPU ceiling (total percentage across all cores).
    cpu_ceiling: f64,
    /// Whether to show the keystroke help overlay.
    show_help: bool,
    /// Whether rendering is paused (`p` key toggle).
    paused: bool,
    /// One-shot flag: redraw once to show/hide PAUSED indicator.
    pause_redraw_pending: bool,
    suspended: bool,
    dirty: bool,
}

impl RenderState {
    fn new() -> Self {
        RenderState {
            bars: HashMap::new(),
            bar_order: Vec::new(),
            resource_status: String::new(),
            context_label: String::new(),
            rss_history: MetricsHistory::new(120),
            thread_history: MetricsHistory::new(120),
            cpu_history: MetricsHistory::new(120),
            io_read_history: MetricsHistory::new(120),
            io_write_history: MetricsHistory::new(120),
            pgfault_history: MetricsHistory::new(120),
            prev_io_read: 0.0,
            prev_io_write: 0.0,
            prev_major_faults: 0.0,
            rss_ceiling_mb: 0.0,
            thread_ceiling: 0.0,
            cpu_ceiling: 0.0,
            show_help: false,
            paused: false,
            pause_redraw_pending: false,
            suspended: false,
            dirty: false,
        }
    }

    /// Number of lines needed for the progress region.
    fn visible_lines(&self) -> u16 {
        if self.show_help {
            return 8; // fixed help overlay height
        }
        let context_line = if self.context_label.is_empty() { 0u16 } else { 1 };
        let bar_lines = self.bar_order.len() as u16;
        let status_lines = if self.resource_status.is_empty() {
            0
        } else if self.rss_history.data.len() >= 2 {
            11 // status text + 10-line chart (8 data + 2 border)
        } else {
            1
        };
        let paused_line = if self.paused { 1 } else { 0 };
        context_line + bar_lines + status_lines + paused_line
    }
}

/// Restore the terminal to a usable state: show cursor, disable raw mode,
/// and flush stdout.
fn restore_terminal<B: ratatui::backend::Backend + io::Write>(terminal: &mut Terminal<B>) {
    let _ = terminal.clear();
    let _ = crossterm::execute!(terminal.backend_mut(), cursor::Show);
    let _ = io::Write::flush(terminal.backend_mut());
    let _ = disable_raw_mode();
}

/// Main render loop running on the background thread.
fn render_loop(rx: mpsc::Receiver<RenderMsg>, max_height: u16) -> io::Result<()> {
    enable_raw_mode()?;
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(max_height),
        },
    )?;

    let mut state = RenderState::new();
    let min_redraw_interval = Duration::from_millis(50);
    let mut last_draw = Instant::now() - min_redraw_interval;
    let poll_timeout = Duration::from_millis(50);
    let mut last_escape: Option<Instant> = None;

    loop {
        // Poll for keyboard events (Ctrl-C, Ctrl-Z, Esc×2) without blocking.
        if ct_event::poll(poll_timeout)? {
            if let CtEvent::Key(key) = ct_event::read()? {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    match key.code {
                        KeyCode::Char('c') => {
                            // Ctrl-C: clean up terminal and raise SIGINT.
                            restore_terminal(&mut terminal);
                            #[cfg(unix)]
                            unsafe {
                                libc::raise(libc::SIGINT);
                            }
                            std::process::exit(130);
                        }
                        KeyCode::Char('z') => {
                            // Ctrl-Z: suspend the process (SIGTSTP).
                            #[cfg(unix)]
                            {
                                restore_terminal(&mut terminal);
                                unsafe {
                                    libc::raise(libc::SIGTSTP);
                                }
                                // On resume (SIGCONT), re-enter raw mode and
                                // mark the display as dirty.
                                enable_raw_mode()?;
                                state.dirty = true;
                            }
                        }
                        _ => {}
                    }
                } else if key.code == KeyCode::Esc {
                    if state.show_help {
                        // Dismiss help overlay on Esc.
                        state.show_help = false;
                        state.dirty = true;
                    } else {
                        // Double-Escape within 1 second: terminate like Ctrl-C.
                        let now = Instant::now();
                        if let Some(prev) = last_escape {
                            if now.duration_since(prev) < Duration::from_secs(1) {
                                restore_terminal(&mut terminal);
                                #[cfg(unix)]
                                unsafe {
                                    libc::raise(libc::SIGINT);
                                }
                                std::process::exit(130);
                            }
                        }
                        last_escape = Some(now);
                    }
                } else if key.code == KeyCode::Char('?') {
                    state.show_help = !state.show_help;
                    state.dirty = true;
                } else if key.code == KeyCode::Char('p') {
                    state.paused = !state.paused;
                    // Force immediate redraw to show/hide the PAUSED indicator.
                    state.pause_redraw_pending = true;
                    state.dirty = true;
                }
            }
        }

        // Drain all queued render messages without blocking.
        let mut pending_logs: Vec<String> = Vec::new();
        let mut pending_emits: Vec<(String, bool)> = Vec::new();
        let mut got_message = false;

        while let Ok(msg) = rx.try_recv() {
            if matches!(msg, RenderMsg::Shutdown) {
                // Flush any pending output before shutdown.
                flush_logs(&mut terminal, &pending_logs)?;
                flush_emits(&mut terminal, &pending_emits)?;
                restore_terminal(&mut terminal);
                return Ok(());
            }
            process_msg(&msg, &mut state, &mut pending_logs, &mut pending_emits);
            got_message = true;
        }

        if !got_message && !state.dirty {
            continue;
        }

        // Flush text output above the progress region.
        flush_logs(&mut terminal, &pending_logs)?;
        flush_emits(&mut terminal, &pending_emits)?;

        // Rate-limited redraw. When paused, only redraw once (to show the
        // PAUSED indicator), then suppress until unpaused.
        let should_redraw = state.dirty && !state.suspended
            && (!state.paused || state.pause_redraw_pending);
        if should_redraw {
            state.pause_redraw_pending = false;
            let elapsed = last_draw.elapsed();
            if elapsed >= min_redraw_interval {
                draw_progress(&mut terminal, &state, max_height)?;
                state.dirty = false;
                last_draw = Instant::now();
            }
        }
    }
}

/// Process a single message, updating state and collecting text output.
fn process_msg(
    msg: &RenderMsg,
    state: &mut RenderState,
    logs: &mut Vec<String>,
    emits: &mut Vec<(String, bool)>,
) {
    let event = match msg {
        RenderMsg::Event(e) => e,
        RenderMsg::Shutdown => return,
    };

    match event {
        UiEvent::ProgressCreate { id, kind, total, label } => {
            state.bars.insert(*id, BarState {
                kind: kind.clone(),
                total: *total,
                position: 0,
                label: label.clone(),
                message: String::new(),
                created_at: Instant::now(),
            });
            state.bar_order.push(*id);
            state.dirty = true;
        }

        UiEvent::ProgressUpdate { id, position } => {
            if let Some(bar) = state.bars.get_mut(id) {
                bar.position = *position;
                state.dirty = true;
            }
        }

        UiEvent::ProgressInc { id, delta } => {
            if let Some(bar) = state.bars.get_mut(id) {
                bar.position = bar.position.saturating_add(*delta);
                state.dirty = true;
            }
        }

        UiEvent::ProgressMessage { id, message } => {
            if let Some(bar) = state.bars.get_mut(id) {
                bar.message = message.clone();
                state.dirty = true;
            }
        }

        UiEvent::ProgressFinish { id } => {
            state.bars.remove(id);
            state.bar_order.retain(|x| x != id);
            state.dirty = true;
        }

        UiEvent::Log { message } => {
            logs.push(message.clone());
        }

        UiEvent::Emit { text } => {
            emits.push((text.clone(), false));
        }

        UiEvent::EmitLn { text } => {
            emits.push((text.clone(), true));
        }

        UiEvent::ResourceStatus { line } => {
            state.resource_status = line.clone();
            // Parse RSS and thread values for chart history.
            // Format: "rss: 1.2G/4G | threads: 4/8 | ..."
            for part in line.split(" | ") {
                if let Some(rest) = part.strip_prefix("rss: ") {
                    // "1.2 GiB/4.0 GiB" → current/ceiling
                    let mut parts = rest.split('/');
                    if let Some(current) = parts.next().and_then(|s| parse_size_to_mb(s.trim())) {
                        state.rss_history.push(current);
                    }
                    if let Some(ceiling) = parts.next().and_then(|s| parse_size_to_mb(s.trim())) {
                        state.rss_ceiling_mb = ceiling;
                    }
                } else if let Some(rest) = part.strip_prefix("cpu: ") {
                    // "42%/800" → current pct / ceiling pct
                    let mut parts = rest.split('/');
                    if let Some(current) = parts.next().and_then(|s| s.trim().strip_suffix('%').and_then(|n| n.parse::<f64>().ok())) {
                        state.cpu_history.push(current);
                    }
                    if let Some(ceiling) = parts.next().and_then(|s| s.trim().parse::<f64>().ok()) {
                        state.cpu_ceiling = ceiling;
                    }
                } else if let Some(rest) = part.strip_prefix("threads: ") {
                    let mut parts = rest.split('/');
                    if let Some(current) = parts.next().and_then(|s| s.trim().parse::<f64>().ok()) {
                        state.thread_history.push(current);
                    }
                    if let Some(ceiling) = parts.next().and_then(|s| s.trim().parse::<f64>().ok()) {
                        state.thread_ceiling = ceiling;
                    }
                } else if let Some(rest) = part.strip_prefix("io_r: ") {
                    // Cumulative read bytes — compute rate as delta/interval
                    if let Some(current_mb) = parse_size_to_mb(rest.trim()) {
                        let rate = if state.prev_io_read > 0.0 {
                            (current_mb - state.prev_io_read) * 2.0 // ×2 because 500ms interval
                        } else {
                            0.0
                        };
                        state.prev_io_read = current_mb;
                        state.io_read_history.push(rate.max(0.0));
                    }
                } else if let Some(rest) = part.strip_prefix("io_w: ") {
                    if let Some(current_mb) = parse_size_to_mb(rest.trim()) {
                        let rate = if state.prev_io_write > 0.0 {
                            (current_mb - state.prev_io_write) * 2.0
                        } else {
                            0.0
                        };
                        state.prev_io_write = current_mb;
                        state.io_write_history.push(rate.max(0.0));
                    }
                } else if let Some(rest) = part.strip_prefix("pgfault: ") {
                    // "major/minor" — track major fault rate
                    let mut parts = rest.split('/');
                    if let Some(major) = parts.next().and_then(|s| s.trim().parse::<f64>().ok()) {
                        let rate = if state.prev_major_faults > 0.0 {
                            (major - state.prev_major_faults) * 2.0
                        } else {
                            0.0
                        };
                        state.prev_major_faults = major;
                        state.pgfault_history.push(rate.max(0.0));
                    }
                }
            }
            state.dirty = true;
        }

        UiEvent::SetContext { label } => {
            state.context_label = label.clone();
            state.dirty = true;
        }

        UiEvent::SuspendBegin => {
            state.suspended = true;
        }

        UiEvent::SuspendEnd => {
            state.suspended = false;
            state.dirty = true;
        }

        UiEvent::Clear => {
            state.bars.clear();
            state.bar_order.clear();
            state.resource_status.clear();
            state.dirty = true;
        }
    }
}

/// Colorize a log line based on pipeline status keywords.
///
/// Recognizes patterns like `[1/5] step-id — OK (...)`, `— SKIP (...)`,
/// `— ERROR (...)`, `— WARNING (...)`, and the pipeline banner.
fn colorize_log<'a>(s: &'a str) -> Line<'a> {
    // Step status lines: "[N/M] id — STATUS ..."
    if let Some(dash_pos) = s.find(" — ") {
        let prefix = &s[..dash_pos];
        let rest = &s[dash_pos + " — ".len()..];

        let (status_word, after_status) = rest
            .find(|c: char| c == ' ' || c == ':' || c == '(')
            .map(|i| (&rest[..i], &rest[i..]))
            .unwrap_or((rest, ""));

        let status_style = match status_word {
            "OK" => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            "SKIP" => Style::default().fg(Color::DarkGray),
            "ERROR" => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            "WARNING" => Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            "WOULD" => Style::default().fg(Color::Blue),
            w if w.ends_with("step(s)") || w == "pipeline" => {
                // Banner line: "dataset [profile] — N step(s) to evaluate"
                return Line::from(vec![
                    Span::styled(prefix, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(" — "),
                    Span::raw(rest),
                ]);
            }
            _ => Style::default(),
        };

        // Dim style for skipped step details
        let detail_style = if status_word == "SKIP" {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };

        return Line::from(vec![
            Span::styled(prefix, Style::default().fg(Color::Cyan)),
            Span::raw(" — "),
            Span::styled(status_word, status_style),
            Span::styled(after_status, detail_style),
        ]);
    }

    // "Pipeline complete" or similar
    if s.contains("pipeline complete") || s.contains("Pipeline complete") {
        return Line::from(Span::styled(s, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)));
    }

    Line::raw(s)
}

/// Insert log lines above the inline progress region.
fn flush_logs(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    logs: &[String],
) -> io::Result<()> {
    if logs.is_empty() {
        return Ok(());
    }
    let lines: Vec<Line> = logs.iter().map(|s| colorize_log(s)).collect();
    terminal.insert_before(lines.len() as u16, |buf| {
        let area = buf.area;
        let text: Vec<Line> = lines.iter().cloned().collect();
        let paragraph = Paragraph::new(text);
        paragraph.render(area, buf);
    })?;
    Ok(())
}

/// Write raw emit text above the inline progress region.
fn flush_emits(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    emits: &[(String, bool)],
) -> io::Result<()> {
    if emits.is_empty() {
        return Ok(());
    }
    // Combine consecutive emits into lines.
    let mut combined = String::new();
    let mut line_count = 0u16;
    for (text, has_newline) in emits {
        combined.push_str(text);
        if *has_newline {
            combined.push('\n');
            line_count += 1;
        }
    }
    // If there's trailing content without a newline, count it as a line.
    if !combined.is_empty() && !combined.ends_with('\n') {
        line_count += 1;
    }
    if line_count == 0 {
        return Ok(());
    }
    let lines: Vec<Line> = combined.lines().map(Line::raw).collect();
    let count = lines.len().max(1) as u16;
    terminal.insert_before(count, |buf| {
        let area = buf.area;
        let paragraph = Paragraph::new(lines.clone());
        paragraph.render(area, buf);
    })?;
    Ok(())
}

/// Draw the progress region in the inline area.
fn draw_progress<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: &RenderState,
    _max_height: u16,
) -> io::Result<()> {
    let needed = state.visible_lines();
    if needed == 0 {
        terminal.draw(|frame| {
            // Empty frame — nothing to render.
            let _ = frame.area();
        })?;
        return Ok(());
    }

    terminal.draw(|frame| {
        let area = frame.area();

        // Help overlay replaces the normal layout.
        if state.show_help {
            render_help(frame, area);
            return;
        }

        // Build constraints: context line + bars + status/chart + paused.
        let has_chart = state.rss_history.data.len() >= 2;
        let mut constraints: Vec<Constraint> = Vec::new();

        if !state.context_label.is_empty() {
            constraints.push(Constraint::Length(1)); // context header
        }
        for _ in &state.bar_order {
            constraints.push(Constraint::Length(1)); // progress bar
        }
        if !state.resource_status.is_empty() {
            constraints.push(Constraint::Length(1)); // status text
            if has_chart {
                constraints.push(Constraint::Length(10)); // chart area (8 data + 2 border)
            }
        }
        if state.paused {
            constraints.push(Constraint::Length(1)); // paused indicator
        }
        // Fill remaining space.
        constraints.push(Constraint::Min(0));

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        let mut idx = 0;

        // Context header line
        if !state.context_label.is_empty() {
            let ctx_line = Line::from(vec![
                Span::styled("▶ ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(
                    &state.context_label,
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  (? help)", Style::default().fg(Color::DarkGray)),
            ]);
            frame.render_widget(Paragraph::new(ctx_line), chunks[idx]);
            idx += 1;
        }

        for id in &state.bar_order {
            if let Some(bar) = state.bars.get(id) {
                render_bar(frame, chunks[idx], bar);
            }
            idx += 1;
        }

        if !state.resource_status.is_empty() {
            if idx < chunks.len() {
                let spans = colorize_resource_status(&state.resource_status);
                let status = Paragraph::new(Line::from(spans));
                frame.render_widget(status, chunks[idx]);
            }
            idx += 1;

            // Chart: RSS + threads over time
            if has_chart {
                if idx < chunks.len() {
                    render_resource_chart(frame, chunks[idx], state);
                }
                idx += 1;
            }
        }

        if state.paused && idx < chunks.len() {
            let paused_line = Line::from(Span::styled(
                " ⏸  PAUSED — press 'p' to resume",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ));
            frame.render_widget(Paragraph::new(paused_line), chunks[idx]);
        }
    })?;

    Ok(())
}

/// Parse a human-readable size string to megabytes (as `f64`).
///
/// Handles formats produced by `format_value()` in the resource governor:
/// `"1.2 GiB"`, `"512.0 MiB"`, `"1.5 KiB"`, `"1024 B"`, as well as
/// compact forms like `"1.2G"`, `"512M"`.
fn parse_size_to_mb(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Try "number unit" format (with space): "1.2 GiB"
    if let Some((num_str, unit)) = s.rsplit_once(' ') {
        let num: f64 = num_str.trim().parse().ok()?;
        let mb = match unit {
            "TiB" => num * 1_048_576.0,
            "GiB" => num * 1_024.0,
            "MiB" => num,
            "KiB" => num / 1_024.0,
            "B" => num / 1_048_576.0,
            "TB" => num * 1_000_000.0,
            "GB" => num * 1_000.0,
            "MB" => num,
            "KB" => num / 1_000.0,
            _ => return None,
        };
        return Some(mb);
    }

    // Compact format: "1.2G", "512M"
    let (num, mult) = if let Some(n) = s.strip_suffix('T') {
        (n.parse::<f64>().ok()?, 1_000_000.0)
    } else if let Some(n) = s.strip_suffix('G') {
        (n.parse::<f64>().ok()?, 1_000.0)
    } else if let Some(n) = s.strip_suffix('M') {
        (n.parse::<f64>().ok()?, 1.0)
    } else if let Some(n) = s.strip_suffix('K') {
        (n.parse::<f64>().ok()?, 0.001)
    } else {
        // Plain number — assume bytes
        (s.parse::<f64>().ok()?, 1.0 / 1_000_000.0)
    };
    Some(num * mult)
}

/// Colorize the resource status line.
///
/// Parses the `"rss: 1.2G/4G | threads: 4/8 | ⚠ EMERGENCY"` format
/// produced by `ResourceStatusSource::status_line()` and applies color:
/// - Label ("rss:", "threads:") in dim white
/// - Values in default color
/// - EMERGENCY in bold red
/// - Throttle in yellow
fn colorize_resource_status(status: &str) -> Vec<Span<'_>> {
    let mut spans = Vec::new();
    spans.push(Span::styled(
        " ▸ ",
        Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
    ));

    for (i, part) in status.split(" | ").enumerate() {
        if i > 0 {
            spans.push(Span::styled(" │ ", Style::default().fg(Color::DarkGray)));
        }

        if part.contains("EMERGENCY") {
            spans.push(Span::styled(
                part,
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ));
        } else if part.contains("throttle") {
            spans.push(Span::styled(
                part,
                Style::default().fg(Color::Yellow),
            ));
        } else if let Some(colon_pos) = part.find(':') {
            // "label: value/ceiling"
            let label = &part[..=colon_pos];
            let value = &part[colon_pos + 1..];
            spans.push(Span::styled(label, Style::default().fg(Color::DarkGray)));
            spans.push(Span::styled(value, Style::default().fg(Color::White)));
        } else {
            spans.push(Span::raw(part));
        }
    }

    spans
}

/// Format a megabyte value as a compact human-readable string.
fn format_mb(mb: f64) -> String {
    if mb >= 1_048_576.0 {
        format!("{:.1}T", mb / 1_048_576.0)
    } else if mb >= 1_024.0 {
        format!("{:.1}G", mb / 1_024.0)
    } else if mb >= 1.0 {
        format!("{:.0}M", mb)
    } else {
        format!("{:.0}K", mb * 1_024.0)
    }
}

/// Render the keystroke help overlay.
fn render_help(frame: &mut ratatui::Frame, area: Rect) {
    let help_lines = vec![
        Line::from(Span::styled(
            " Keyboard shortcuts",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ?       ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle this help"),
        ]),
        Line::from(vec![
            Span::styled("  p       ", Style::default().fg(Color::Yellow)),
            Span::raw("Pause / resume rendering"),
        ]),
        Line::from(vec![
            Span::styled("  Esc Esc ", Style::default().fg(Color::Yellow)),
            Span::raw("Terminate (same as Ctrl-C)"),
        ]),
        Line::from(vec![
            Span::styled("  Ctrl-C  ", Style::default().fg(Color::Yellow)),
            Span::raw("Interrupt process"),
        ]),
        Line::from(vec![
            Span::styled("  Ctrl-Z  ", Style::default().fg(Color::Yellow)),
            Span::raw("Suspend process"),
        ]),
    ];
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" help ", Style::default().fg(Color::Cyan)));
    let paragraph = Paragraph::new(help_lines).block(block);
    frame.render_widget(paragraph, area);
}

/// Build a single-dataset chart widget for a metric time series.
fn metric_chart<'a>(
    history: &'a MetricsHistory,
    title: &str,
    color: Color,
    y_label: String,
    dim: Style,
) -> Chart<'a> {
    let datasets = vec![
        Dataset::default()
            .name(title.to_string())
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(color))
            .data(history.as_slice()),
    ];
    Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(format!(" {} ", title), Style::default().fg(color)))
                .border_style(dim),
        )
        .x_axis(Axis::default().bounds(history.x_bounds()).style(dim))
        .y_axis(
            Axis::default()
                .bounds(history.y_bounds())
                .labels(vec![
                    Span::styled("0", dim),
                    Span::styled(y_label, dim),
                ])
                .style(dim),
        )
}

/// Render the resource metrics charts: RSS, CPU, I/O, threads, page faults.
fn render_resource_chart(frame: &mut ratatui::Frame, area: Rect, state: &RenderState) {
    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25), // RSS
            Constraint::Percentage(20), // CPU
            Constraint::Percentage(20), // I/O
            Constraint::Percentage(15), // threads
            Constraint::Percentage(20), // faults
        ])
        .split(area);

    let dim = Style::default().fg(Color::DarkGray);

    // RSS (magenta)
    if !state.rss_history.is_empty() {
        let label = format_mb(state.rss_history.y_bounds()[1]);
        frame.render_widget(
            metric_chart(&state.rss_history, "rss", Color::Magenta, label, dim),
            panels[0],
        );
    }

    // CPU (green)
    if !state.cpu_history.is_empty() {
        let label = format!("{}%", state.cpu_history.y_bounds()[1] as u64);
        frame.render_widget(
            metric_chart(&state.cpu_history, "cpu", Color::Green, label, dim),
            panels[1],
        );
    }

    // I/O — read (cyan) and write (yellow) on the same chart
    if !state.io_read_history.is_empty() || !state.io_write_history.is_empty() {
        let mut datasets = Vec::new();
        // Compute combined Y bounds from both read and write
        let mut y_max = 1.0f64;
        if !state.io_read_history.is_empty() {
            y_max = y_max.max(state.io_read_history.max_value * 1.25);
            datasets.push(
                Dataset::default()
                    .name("read")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(state.io_read_history.as_slice()),
            );
        }
        if !state.io_write_history.is_empty() {
            y_max = y_max.max(state.io_write_history.max_value * 1.25);
            datasets.push(
                Dataset::default()
                    .name("write")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Yellow))
                    .data(state.io_write_history.as_slice()),
            );
        }
        // Use the read history for X bounds (both are populated together)
        let x_bounds = if !state.io_read_history.is_empty() {
            state.io_read_history.x_bounds()
        } else {
            state.io_write_history.x_bounds()
        };
        let label = format!("{}/s", format_mb(y_max));
        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Line::from(vec![
                        Span::styled(" io ", Style::default().fg(Color::Cyan)),
                        Span::styled("r", Style::default().fg(Color::Cyan)),
                        Span::styled("/", dim),
                        Span::styled("w ", Style::default().fg(Color::Yellow)),
                    ]))
                    .border_style(dim),
            )
            .x_axis(Axis::default().bounds(x_bounds).style(dim))
            .y_axis(
                Axis::default()
                    .bounds([0.0, y_max])
                    .labels(vec![
                        Span::styled("0", dim),
                        Span::styled(label, dim),
                    ])
                    .style(dim),
            );
        frame.render_widget(chart, panels[2]);
    }

    // Threads (blue)
    if !state.thread_history.is_empty() {
        let label = format!("{}", state.thread_history.y_bounds()[1] as u64);
        frame.render_widget(
            metric_chart(&state.thread_history, "threads", Color::Blue, label, dim),
            panels[3],
        );
    }

    // Page faults (red)
    if !state.pgfault_history.is_empty() {
        let label = format!("{}/s", state.pgfault_history.y_bounds()[1] as u64);
        frame.render_widget(
            metric_chart(&state.pgfault_history, "faults", Color::Red, label, dim),
            panels[4],
        );
    }
}

/// Render a single progress bar or spinner into a one-line area.
fn render_bar(frame: &mut ratatui::Frame, area: Rect, bar: &BarState) {
    match bar.kind {
        ProgressKind::Bar => {
            let ratio = if bar.total > 0 {
                (bar.position as f64 / bar.total as f64).min(1.0)
            } else {
                0.0
            };
            let pct = (ratio * 100.0) as u16;
            let elapsed = bar.created_at.elapsed().as_secs_f64();
            let rps_str = if elapsed > 0.5 && bar.position > 0 {
                let rps = bar.position as f64 / elapsed;
                let rate = format_rate(rps);
                if rate.is_empty() { String::new() } else { format!(" {}", rate) }
            } else {
                String::new()
            };
            let label_str = if bar.message.is_empty() {
                format!("{} [{}/{}] {}%{}", bar.label, format_count(bar.position), format_count(bar.total), pct, rps_str)
            } else {
                format!("{} [{}/{}] {}%{} {}", bar.label, format_count(bar.position), format_count(bar.total), pct, rps_str, bar.message)
            };

            let gauge = Gauge::default()
                .gauge_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .ratio(ratio)
                .label(label_str);
            frame.render_widget(gauge, area);
        }
        ProgressKind::Ratio => {
            let ratio = if bar.total > 0 {
                (bar.position as f64 / bar.total as f64).min(1.0)
            } else {
                0.0
            };
            let label_str = if bar.message.is_empty() {
                bar.label.clone()
            } else {
                format!("{} {}", bar.label, bar.message)
            };
            let gauge = Gauge::default()
                .gauge_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                .ratio(ratio)
                .label(label_str);
            frame.render_widget(gauge, area);
        }
        ProgressKind::Spinner => {
            // Simple text-based spinner (no animation frame needed).
            let dots = match (bar.position % 4) as usize {
                0 => "   ",
                1 => ".  ",
                2 => ".. ",
                _ => "...",
            };
            let text = if bar.message.is_empty() {
                format!("{} {}", dots, bar.label)
            } else {
                format!("{} {} — {}", dots, bar.label, bar.message)
            };
            let paragraph = Paragraph::new(Line::from(text));
            frame.render_widget(paragraph, area);
        }
    }
}

/// Format a rate value as a compact human-readable string (e.g. "1.2M/s").
fn format_rate(rps: f64) -> String {
    if rps >= 1_000_000_000.0 {
        format!("{:.1}B/s", rps / 1_000_000_000.0)
    } else if rps >= 1_000_000.0 {
        format!("{:.1}M/s", rps / 1_000_000.0)
    } else if rps >= 1_000.0 {
        format!("{:.1}K/s", rps / 1_000.0)
    } else if rps >= 1.0 {
        format!("{:.0}/s", rps)
    } else {
        String::new()
    }
}

/// Format a count with thousands separators.
fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_count_thousands() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1_234_567), "1,234,567");
    }

    #[test]
    fn format_rate_compact() {
        assert_eq!(format_rate(0.5), "");
        assert_eq!(format_rate(42.0), "42/s");
        assert_eq!(format_rate(1_500.0), "1.5K/s");
        assert_eq!(format_rate(2_300_000.0), "2.3M/s");
        assert_eq!(format_rate(1_200_000_000.0), "1.2B/s");
    }

    #[test]
    fn render_state_visible_lines() {
        let mut state = RenderState::new();
        assert_eq!(state.visible_lines(), 0);

        state.bar_order.push(ProgressId(0));
        state.bars.insert(ProgressId(0), BarState {
            kind: ProgressKind::Bar,
            total: 100,
            position: 0,
            label: "test".into(),
            message: String::new(),
            created_at: Instant::now(),
        });
        assert_eq!(state.visible_lines(), 1);

        state.resource_status = "mem: 1.2G / 4G".into();
        assert_eq!(state.visible_lines(), 2);
    }

    #[test]
    fn process_msg_lifecycle() {
        let mut state = RenderState::new();
        let mut logs = Vec::new();
        let mut emits = Vec::new();

        let id = ProgressId(0);

        // Create
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressCreate {
                id,
                kind: ProgressKind::Bar,
                total: 100,
                label: "test".into(),
            }),
            &mut state, &mut logs, &mut emits,
        );
        assert_eq!(state.bars.len(), 1);
        assert_eq!(state.bar_order.len(), 1);

        // Inc
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressInc { id, delta: 50 }),
            &mut state, &mut logs, &mut emits,
        );
        assert_eq!(state.bars[&id].position, 50);

        // Message
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressMessage {
                id,
                message: "halfway".into(),
            }),
            &mut state, &mut logs, &mut emits,
        );
        assert_eq!(state.bars[&id].message, "halfway");

        // Finish
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressFinish { id }),
            &mut state, &mut logs, &mut emits,
        );
        assert!(state.bars.is_empty());
        assert!(state.bar_order.is_empty());
    }

    #[test]
    fn process_msg_logs_and_emits() {
        let mut state = RenderState::new();
        let mut logs = Vec::new();
        let mut emits = Vec::new();

        process_msg(
            &RenderMsg::Event(UiEvent::Log { message: "hello".into() }),
            &mut state, &mut logs, &mut emits,
        );
        process_msg(
            &RenderMsg::Event(UiEvent::EmitLn { text: "world".into() }),
            &mut state, &mut logs, &mut emits,
        );

        assert_eq!(logs, vec!["hello"]);
        assert_eq!(emits.len(), 1);
        assert_eq!(emits[0], ("world".into(), true));
    }

    #[test]
    fn suspend_defers_dirty() {
        let mut state = RenderState::new();
        let mut logs = Vec::new();
        let mut emits = Vec::new();

        process_msg(
            &RenderMsg::Event(UiEvent::SuspendBegin),
            &mut state, &mut logs, &mut emits,
        );
        assert!(state.suspended);

        process_msg(
            &RenderMsg::Event(UiEvent::SuspendEnd),
            &mut state, &mut logs, &mut emits,
        );
        assert!(!state.suspended);
        assert!(state.dirty);
    }

    #[test]
    fn render_bar_widget_on_test_backend() {
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(60, 4);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = RenderState::new();
        let id = ProgressId(0);
        state.bars.insert(id, BarState {
            kind: ProgressKind::Bar,
            total: 100,
            position: 50,
            label: "importing".into(),
            message: String::new(),
            created_at: Instant::now(),
        });
        state.bar_order.push(id);

        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Min(0)])
                .split(area);

            if let Some(bar) = state.bars.get(&id) {
                render_bar(frame, chunks[0], bar);
            }
        }).unwrap();

        // Verify the rendered buffer contains the label and percentage.
        let buf = terminal.backend().buffer().clone();
        let content: String = (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).unwrap().symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("importing"), "Expected 'importing' in: {}", content);
        assert!(content.contains("50%"), "Expected '50%' in: {}", content);
    }

    #[test]
    fn render_spinner_widget_on_test_backend() {
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(60, 4);
        let mut terminal = Terminal::new(backend).unwrap();

        let bar = BarState {
            kind: ProgressKind::Spinner,
            total: 0,
            position: 2,
            label: "scanning".into(),
            message: "file.fvec".into(),
            created_at: Instant::now(),
        };

        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Min(0)])
                .split(area);
            render_bar(frame, chunks[0], &bar);
        }).unwrap();

        let buf = terminal.backend().buffer().clone();
        let content: String = (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).unwrap().symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("scanning"), "Expected 'scanning' in: {}", content);
        assert!(content.contains("file.fvec"), "Expected 'file.fvec' in: {}", content);
    }

    #[test]
    fn render_resource_status_on_test_backend() {
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(60, 4);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = RenderState::new();
        let id = ProgressId(0);
        state.bars.insert(id, BarState {
            kind: ProgressKind::Bar,
            total: 1000,
            position: 250,
            label: "computing".into(),
            message: String::new(),
            created_at: Instant::now(),
        });
        state.bar_order.push(id);
        state.resource_status = "mem: 1.2G/4G | threads: 8".into();

        // Use draw_progress directly with test backend
        terminal.draw(|frame| {
            let area = frame.area();
            let constraints = vec![
                Constraint::Length(1), // bar
                Constraint::Length(1), // resource status
                Constraint::Min(0),
            ];
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(area);

            if let Some(bar) = state.bars.get(&id) {
                render_bar(frame, chunks[0], bar);
            }

            let status = Paragraph::new(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(
                    state.resource_status.as_str(),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
            frame.render_widget(status, chunks[1]);
        }).unwrap();

        let buf = terminal.backend().buffer().clone();
        // Check row 0 has the bar
        let row0: String = (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).unwrap().symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(row0.contains("computing"), "Expected 'computing' in row 0: {}", row0);

        // Check row 1 has the resource status
        let row1: String = (0..buf.area.width)
            .map(|x| buf.cell((x, 1)).unwrap().symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(row1.contains("mem:"), "Expected 'mem:' in row 1: {}", row1);
        assert!(row1.contains("threads:"), "Expected 'threads:' in row 1: {}", row1);
    }

    #[test]
    fn multiple_bars_render_in_order() {
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(60, 6);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = RenderState::new();
        for i in 0..3 {
            let id = ProgressId(i);
            state.bars.insert(id, BarState {
                kind: ProgressKind::Bar,
                total: 100,
                position: (i as u64 + 1) * 25,
                label: format!("step-{}", i),
                message: String::new(),
                created_at: Instant::now(),
            });
            state.bar_order.push(id);
        }

        terminal.draw(|frame| {
            let area = frame.area();
            let constraints: Vec<Constraint> = (0..3)
                .map(|_| Constraint::Length(1))
                .chain(std::iter::once(Constraint::Min(0)))
                .collect();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(area);

            for (i, id) in state.bar_order.iter().enumerate() {
                if let Some(bar) = state.bars.get(id) {
                    render_bar(frame, chunks[i], bar);
                }
            }
        }).unwrap();

        let buf = terminal.backend().buffer().clone();
        for i in 0..3 {
            let row: String = (0..buf.area.width)
                .map(|x| buf.cell((x, i as u16)).unwrap().symbol().chars().next().unwrap_or(' '))
                .collect();
            assert!(
                row.contains(&format!("step-{}", i)),
                "Expected 'step-{}' in row {}: {}",
                i, i, row
            );
        }
    }

    #[test]
    fn clear_removes_all_state() {
        let mut state = RenderState::new();
        let mut logs = Vec::new();
        let mut emits = Vec::new();

        // Add some state
        let id = ProgressId(0);
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressCreate {
                id,
                kind: ProgressKind::Bar,
                total: 100,
                label: "test".into(),
            }),
            &mut state, &mut logs, &mut emits,
        );
        process_msg(
            &RenderMsg::Event(UiEvent::ResourceStatus {
                line: "status".into(),
            }),
            &mut state, &mut logs, &mut emits,
        );

        assert!(!state.bars.is_empty());
        assert!(!state.resource_status.is_empty());

        // Clear
        process_msg(
            &RenderMsg::Event(UiEvent::Clear),
            &mut state, &mut logs, &mut emits,
        );

        assert!(state.bars.is_empty());
        assert!(state.bar_order.is_empty());
        assert!(state.resource_status.is_empty());
    }

    // ── Layout introspection helpers ─────────────────────────────────

    /// Extract the text content of a single row from the buffer.
    fn row_text(buf: &ratatui::buffer::Buffer, row: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf.cell((x, row)).unwrap().symbol().chars().next().unwrap_or(' '))
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    /// Check that a row contains non-whitespace content.
    fn row_has_content(buf: &ratatui::buffer::Buffer, row: u16) -> bool {
        (0..buf.area.width)
            .any(|x| {
                let ch = buf.cell((x, row)).unwrap().symbol();
                ch != " " && ch != ""
            })
    }

    /// Render the full progress region into a test buffer and return it.
    fn render_state_to_buffer(state: &RenderState, width: u16, height: u16) -> ratatui::buffer::Buffer {
        use ratatui::backend::TestBackend;

        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        draw_progress(&mut terminal, state, height).unwrap();
        terminal.backend().buffer().clone()
    }

    /// Build a RenderState with a progress bar, resource status, and chart data.
    fn state_with_chart(rss_samples: &[f64], thread_samples: &[f64]) -> RenderState {
        let mut state = RenderState::new();

        // Add one progress bar
        let id = ProgressId(0);
        state.bars.insert(id, BarState {
            kind: ProgressKind::Bar,
            total: 1000,
            position: 500,
            label: "computing".into(),
            message: String::new(),
            created_at: Instant::now(),
        });
        state.bar_order.push(id);

        // Set resource status
        state.resource_status = "rss: 2.5 GiB/8.0 GiB | threads: 4/8".into();

        // Push RSS samples
        for &v in rss_samples {
            state.rss_history.push(v);
        }

        // Push thread samples
        for &v in thread_samples {
            state.thread_history.push(v);
        }

        state.rss_ceiling_mb = 8192.0; // 8 GiB
        state.thread_ceiling = 8.0;
        state.dirty = true;
        state
    }

    // ── Layout introspection tests ───────────────────────────────────

    #[test]
    fn layout_chart_not_clipped_at_top() {
        // RSS data that should fill most of the chart vertically
        let rss: Vec<f64> = (0..20).map(|i| 1000.0 + (i as f64) * 100.0).collect();
        let threads: Vec<f64> = (0..20).map(|i| 2.0 + (i as f64) * 0.3).collect();
        let state = state_with_chart(&rss, &threads);

        let height = state.visible_lines() + 2; // +2 for filler
        let buf = render_state_to_buffer(&state, 120, height);

        // Row 0: progress bar
        let bar_row = row_text(&buf, 0);
        assert!(bar_row.contains("computing"), "Row 0 should contain the progress bar label, got: '{}'", bar_row);

        // Row 1: resource status line
        let status_row = row_text(&buf, 1);
        assert!(status_row.contains("rss:"), "Row 1 should contain resource status, got: '{}'", status_row);

        // Rows 2..12: chart area (10 lines: 8 data + 2 border)
        // The chart should have content in multiple rows, not just the edges.
        let chart_rows_with_content: Vec<u16> = (2..12)
            .filter(|&r| row_has_content(&buf, r))
            .collect();
        assert!(
            chart_rows_with_content.len() >= 3,
            "Chart should have content in at least 3 rows (got {} rows with content: {:?})",
            chart_rows_with_content.len(),
            chart_rows_with_content,
        );

        // The top chart row (row 2) should have Y-axis label content
        let top_chart_row = row_text(&buf, 2);
        assert!(
            !top_chart_row.is_empty(),
            "Top chart row should have Y-axis label, got empty"
        );

        // The Y-axis max label should NOT be a raw number like "907991M"
        // It should be formatted compactly like "3.6G"
        assert!(
            !top_chart_row.contains("M") || top_chart_row.len() < 15 || top_chart_row.contains("G"),
            "Y-axis label should be compact, got: '{}'", top_chart_row
        );
    }

    #[test]
    fn layout_chart_data_uses_full_height() {
        // Linearly increasing RSS — the line should span from bottom to top
        let rss: Vec<f64> = (0..30).map(|i| (i as f64) * 100.0).collect();
        let threads: Vec<f64> = (0..30).map(|i| (i as f64) * 0.25).collect();
        let state = state_with_chart(&rss, &threads);

        let height = state.visible_lines() + 2;
        let buf = render_state_to_buffer(&state, 120, height);

        // Count rows in the chart area (rows 2-11) that have braille characters.
        // Braille block is U+2800..U+28FF.
        let braille_rows: Vec<u16> = (2..12)
            .filter(|&r| {
                (0..buf.area.width).any(|x| {
                    let ch = buf.cell((x, r)).unwrap().symbol().chars().next().unwrap_or(' ');
                    ('\u{2800}'..='\u{28FF}').contains(&ch)
                })
            })
            .collect();

        assert!(
            braille_rows.len() >= 3,
            "Braille data should appear in at least 3 chart rows for linearly increasing data. \
             Got {} rows: {:?}",
            braille_rows.len(),
            braille_rows,
        );
    }

    #[test]
    fn layout_no_overlap_between_bar_and_chart() {
        let rss: Vec<f64> = vec![500.0; 10];
        let threads: Vec<f64> = vec![4.0; 10];
        let state = state_with_chart(&rss, &threads);

        let height = state.visible_lines() + 2;
        let buf = render_state_to_buffer(&state, 120, height);

        // Row 0 should be the progress bar (contains "computing" and percentage)
        let row0 = row_text(&buf, 0);
        assert!(row0.contains("computing"), "Row 0 should be the progress bar");
        assert!(row0.contains("50%"), "Row 0 should show progress percentage");

        // Row 0 should NOT contain chart elements (box-drawing or braille)
        let has_chart_chars = row0.chars().any(|ch|
            ('\u{2500}'..='\u{257F}').contains(&ch) || // box drawing
            ('\u{2800}'..='\u{28FF}').contains(&ch)     // braille
        );
        assert!(
            !has_chart_chars,
            "Progress bar row should not contain chart characters: '{}'", row0
        );

        // Row 1 should be the status line
        let row1 = row_text(&buf, 1);
        assert!(row1.contains("rss:"), "Row 1 should be resource status, got: '{}'", row1);
    }

    #[test]
    fn layout_status_only_no_chart_when_insufficient_data() {
        let mut state = RenderState::new();
        state.resource_status = "rss: 1.0 GiB | threads: 4".into();
        // Only 1 data point — not enough for chart
        state.rss_history.push(1024.0);
        state.dirty = true;

        assert_eq!(state.visible_lines(), 1, "Should be 1 line (status only) with < 2 data points");

        let buf = render_state_to_buffer(&state, 80, 4);
        let row0 = row_text(&buf, 0);
        assert!(row0.contains("rss:"), "Should show status text without chart");

        // No braille characters anywhere
        for r in 0..4 {
            let has_braille = (0..80u16).any(|x| {
                let ch = buf.cell((x, r)).unwrap().symbol().chars().next().unwrap_or(' ');
                ('\u{2800}'..='\u{28FF}').contains(&ch)
            });
            assert!(!has_braille, "Row {} should not have braille with only 1 data point", r);
        }
    }

    #[test]
    fn layout_y_axis_label_format() {
        // Verify format_mb produces readable labels
        assert_eq!(format_mb(0.5), "512K");
        assert_eq!(format_mb(100.0), "100M");
        assert_eq!(format_mb(2048.0), "2.0G");
        assert_eq!(format_mb(1_500_000.0), "1.4T");
    }

    #[test]
    fn parse_size_to_mb_governor_format() {
        // These are the exact formats produced by format_value() in resource.rs
        assert!((parse_size_to_mb("1.2 GiB").unwrap() - 1228.8).abs() < 0.1);
        assert!((parse_size_to_mb("512.0 MiB").unwrap() - 512.0).abs() < 0.1);
        assert!((parse_size_to_mb("1.5 KiB").unwrap() - 0.00146).abs() < 0.01);
        assert!((parse_size_to_mb("1024 B").unwrap() - 0.000977).abs() < 0.001);
        // Also handles compact forms
        assert!((parse_size_to_mb("1.2G").unwrap() - 1200.0).abs() < 0.1);
        assert!((parse_size_to_mb("512M").unwrap() - 512.0).abs() < 0.1);
    }
}
