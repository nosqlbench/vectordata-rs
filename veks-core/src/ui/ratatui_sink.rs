// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Rich terminal [`UiSink`] using ratatui's alternate screen mode.
//!
//! Renders progress bars and a resource status line in a fixed region at the
//! bottom of the terminal. Log lines scroll above the fixed area via
//! `insert_before()`.
//!
//! Events are sent over a channel to a dedicated render thread that owns the
//! ratatui `Terminal`. This keeps the `RatatuiSink` itself `Send + Sync`.

use std::collections::HashMap;
use std::io::{self, Stdout};
use std::sync::{Arc, Mutex};
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
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Sparkline, Widget},
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

/// Rich terminal sink using ratatui alternate screen mode.
///
/// Thread-safe: holds only a channel sender and an atomic id counter.
/// The actual terminal rendering happens on a background thread.
///
/// Log messages are buffered in `console_log` so they can be printed
/// to stdout after the TUI exits (the alternate screen wipes TUI output).
pub struct RatatuiSink {
    tx: mpsc::Sender<RenderMsg>,
    next_id: AtomicU32,
    render_thread: Mutex<Option<thread::JoinHandle<()>>>,
    /// Buffered log messages for post-TUI console output.
    console_log: Arc<Mutex<Vec<String>>>,
}

impl RatatuiSink {
    /// Create a new ratatui-based sink.
    ///
    /// `inline_height` is the number of terminal lines reserved for the
    /// fixed progress region (typically 4-8).
    pub fn new(inline_height: u16) -> io::Result<Self> {
        Self::with_redraw_interval(inline_height, Duration::from_millis(250))
    }

    /// Create a new ratatui-based sink with a custom redraw interval.
    pub fn with_redraw_interval(inline_height: u16, redraw_interval: Duration) -> io::Result<Self> {
        let console_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let console_log_clone = Arc::clone(&console_log);

        // Install a panic hook that restores the terminal before printing
        // the panic message. Without this, a panic leaves the terminal in
        // raw/alternate screen mode and the error message is invisible.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let mut stdout = io::stdout();
            let _ = crossterm::execute!(
                stdout,
                crossterm::terminal::LeaveAlternateScreen,
                cursor::Show,
            );
            let _ = io::Write::flush(&mut stdout);
            default_hook(info);
        }));

        let (tx, rx) = mpsc::channel::<RenderMsg>();

        let handle = thread::Builder::new()
            .name("ratatui-render".into())
            .spawn(move || {
                if let Err(e) = render_loop(rx, inline_height, redraw_interval, console_log_clone) {
                    // Unconditionally restore terminal state — render_loop may
                    // have entered raw/alternate screen before the error.
                    let _ = disable_raw_mode();
                    let mut stdout = io::stdout();
                    let _ = crossterm::execute!(
                        stdout,
                        crossterm::terminal::LeaveAlternateScreen,
                        cursor::Show,
                    );
                    let _ = io::Write::flush(&mut stdout);
                    eprintln!("ratatui render error: {}", e);
                }
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(RatatuiSink {
            tx,
            next_id: AtomicU32::new(0),
            render_thread: Mutex::new(Some(handle)),
            console_log,
        })
    }
}

impl Drop for RatatuiSink {
    fn drop(&mut self) {
        // shutdown() may have already joined the thread
        let _ = self.tx.send(RenderMsg::Shutdown);
        if let Some(h) = self.render_thread.lock().unwrap().take() {
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

    fn take_console_log(&self) -> Vec<String> {
        std::mem::take(&mut self.console_log.lock().unwrap())
    }

    fn shutdown(&self) {
        let _ = self.tx.send(RenderMsg::Shutdown);
        if let Some(h) = self.render_thread.lock().unwrap().take() {
            let _ = h.join();
        }
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
    /// Optional `(position, time)` anchor for rate calculation. When
    /// present, the displayed rate is `(position - p0) / (now - t0)`
    /// instead of `position / (now - created_at)`. Set via
    /// [`UiEvent::ProgressAnchorRate`] to exclude an early burst of
    /// trivial-cost progress (e.g. cache-skipped shards) from the
    /// throughput display.
    rate_anchor: Option<(u64, Instant)>,
    /// The unit label for rate display (e.g., "rec", "files", "chunks").
    unit: String,
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

/// Time-series buffer that keeps up to `capacity` data points and
/// automatically downsamples when full (pinned mode) or drops old samples
/// (scrolling mode).
///
/// **Scrolling mode** (default): keeps only the most recent `capacity` raw
/// samples, providing a moving window. Y-max is recomputed from the visible
/// window so the scale adapts to recent throughput.
///
/// **Pinned mode**: retains all samples from the beginning. When the buffer
/// reaches capacity, consecutive pairs are merged (averaged), halving the
/// point count and doubling the effective sampling period. The x-coordinates
/// are normalized to `0..N` so the chart always stretches from left to right.
///
/// Press `t` in the TUI to toggle between modes. In pinned mode, the chart
/// title shows `[pinned]`.
struct DownsamplingHistory {
    data: Vec<(f64, f64)>,
    capacity: usize,
    /// How many raw samples have been accumulated into the current pending slot.
    pending_count: u64,
    /// Accumulated value for the current pending slot (to be averaged).
    pending_sum: f64,
    /// Current downsampling factor: every `stride` raw samples become one point.
    stride: u64,
    /// Total raw samples received (for stride accounting).
    raw_count: u64,
    max_value: f64,
    /// When true, keep all history (downsample when full). When false,
    /// use a scrolling window that drops old samples.
    pinned: bool,
}

impl DownsamplingHistory {
    fn new(capacity: usize) -> Self {
        DownsamplingHistory {
            data: Vec::with_capacity(capacity),
            capacity,
            pending_count: 0,
            pending_sum: 0.0,
            stride: 1,
            raw_count: 0,
            max_value: 1.0,
            pinned: false,
        }
    }

    fn push(&mut self, value: f64) {
        self.raw_count += 1;

        if self.pinned {
            // Pinned mode: accumulate into stride-averaged slots, compact when full
            self.pending_count += 1;
            self.pending_sum += value;

            if self.pending_count >= self.stride {
                let avg = self.pending_sum / self.pending_count as f64;
                self.pending_count = 0;
                self.pending_sum = 0.0;

                if self.data.len() >= self.capacity {
                    self.compact();
                }
                let x = self.data.len() as f64;
                self.data.push((x, avg));
                if avg > self.max_value {
                    self.max_value = avg;
                }
            }
        } else {
            // Scrolling mode: keep most recent `capacity` raw samples
            if self.data.len() >= self.capacity {
                self.data.remove(0);
                // Re-index x-coordinates so the chart always starts at 0
                for (i, pt) in self.data.iter_mut().enumerate() {
                    pt.0 = i as f64;
                }
                // Recompute max from visible window
                self.max_value = self.data.iter().map(|p| p.1).fold(1.0f64, f64::max);
            }
            let x = self.data.len() as f64;
            self.data.push((x, value));
            if value > self.max_value {
                self.max_value = value;
            }
        }
    }

    /// Merge consecutive pairs, halving the data and doubling the stride.
    /// Only used in pinned mode.
    fn compact(&mut self) {
        let mut compacted = Vec::with_capacity(self.data.len() / 2 + 1);
        let mut i = 0;
        while i + 1 < self.data.len() {
            let avg = (self.data[i].1 + self.data[i + 1].1) / 2.0;
            compacted.push((compacted.len() as f64, avg));
            i += 2;
        }
        // If odd number of points, keep the last one.
        if i < self.data.len() {
            compacted.push((compacted.len() as f64, self.data[i].1));
        }
        self.data = compacted;
        self.stride *= 2;
        // Recompute max after compaction (averages may be lower than peaks).
        self.max_value = self.data.iter().map(|p| p.1).fold(1.0f64, f64::max);
    }

    /// Toggle between scrolling and pinned modes.
    ///
    /// When switching to pinned mode, the current data is retained and
    /// future samples use the keep+resample strategy. When switching
    /// back to scrolling, data is truncated to the most recent `capacity`
    /// samples and the stride is reset.
    fn toggle_pinned(&mut self) {
        self.pinned = !self.pinned;
        if !self.pinned {
            // Switching from pinned → scrolling: reset downsampling state
            self.stride = 1;
            self.pending_count = 0;
            self.pending_sum = 0.0;
            // Truncate to last `capacity` points
            if self.data.len() > self.capacity {
                let start = self.data.len() - self.capacity;
                self.data = self.data[start..].to_vec();
            }
            // Re-index x-coordinates
            for (i, pt) in self.data.iter_mut().enumerate() {
                pt.0 = i as f64;
            }
            self.max_value = self.data.iter().map(|p| p.1).fold(1.0f64, f64::max);
        }
    }

    fn is_pinned(&self) -> bool {
        self.pinned
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
        [0.0, (self.data.len() as f64).max(1.0)]
    }

    fn y_bounds(&self) -> [f64; 2] {
        [0.0, self.max_value * 1.25]
    }

    /// Compute a moving average over the data with the given window size.
    ///
    /// Returns a new Vec of (x, avg) points suitable for rendering as a
    /// second chart line. The window is centered on each point.
    fn moving_average(&self, window: usize) -> Vec<(f64, f64)> {
        if self.data.len() < 2 || window < 2 {
            return Vec::new();
        }
        let half = window / 2;
        let mut result = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(self.data.len());
            let sum: f64 = self.data[start..end].iter().map(|p| p.1).sum();
            let avg = sum / (end - start) as f64;
            result.push((self.data[i].0, avg));
        }
        result
    }

    fn clear(&mut self) {
        self.data.clear();
        self.pending_count = 0;
        self.pending_sum = 0.0;
        self.stride = 1;
        self.raw_count = 0;
        self.max_value = 1.0;
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
    /// Per-core CPU utilization percentages (0–100 per core).
    cpu_cores: Vec<u64>,
    /// I/O read rate history (MB/s).
    io_read_history: MetricsHistory,
    /// I/O write rate history (MB/s).
    io_write_history: MetricsHistory,
    /// Page fault rate history (major faults/s).
    pgfault_history: MetricsHistory,
    /// System page cache size history (MB).
    pcache_history: MetricsHistory,
    /// I/O queue depth history — read inflight requests.
    ioq_read_history: MetricsHistory,
    /// I/O queue depth history — write inflight requests.
    ioq_write_history: MetricsHistory,
    /// Previous cumulative I/O read bytes (for delta computation).
    prev_io_read: f64,
    /// Previous cumulative I/O write bytes.
    prev_io_write: f64,
    /// Timestamp of last I/O sample (for rate computation).
    last_io_time: Instant,
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
    /// RPS (records per second) history with auto-downsampling.
    rps_history: DownsamplingHistory,
    /// Previous bar positions for computing RPS deltas.
    prev_positions: HashMap<ProgressId, u64>,
    /// Timestamp of the last RPS sample.
    last_rps_sample: Instant,
    /// Extra budget items not covered by dedicated charts (e.g., "segmentsize: 1M/1M").
    extra_budget: Vec<String>,
    /// Alert indicator: "EMERGENCY", "throttle", or empty.
    alert: String,
    /// Current RSS readout string for chart title (e.g., "2.5G/806G").
    rss_readout: String,
    /// Current thread readout string for chart title (e.g., "128/128").
    thread_readout: String,
    /// Current pgfault readout string for chart title (e.g., "0/26M").
    pgfault_readout: String,
    /// Current page cache readout string for chart title (e.g., "120G hit:98%").
    pcache_readout: String,
    /// YAML snippet of the current pipeline step.
    step_yaml: String,
    /// Height of the step YAML panel (user-adjustable via up/down arrows).
    step_yaml_height: u16,
    /// Width percentage of the YAML panel vs RPS chart (user-adjustable via left/right arrows).
    yaml_width_pct: u16,
    /// Whether rendering is paused (`p` key toggle).
    paused: bool,
    /// One-shot flag: redraw once to show/hide PAUSED indicator.
    pause_redraw_pending: bool,
    suspended: bool,
    dirty: bool,
    /// Recent log messages for the bottom log window.
    recent_logs: Vec<String>,
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
            cpu_cores: Vec::new(),
            io_read_history: MetricsHistory::new(120),
            io_write_history: MetricsHistory::new(120),
            pgfault_history: MetricsHistory::new(120),
            pcache_history: MetricsHistory::new(120),
            ioq_read_history: MetricsHistory::new(120),
            ioq_write_history: MetricsHistory::new(120),
            rps_history: DownsamplingHistory::new(240), // ~1 minute at 250ms sampling
            prev_positions: HashMap::new(),
            last_rps_sample: Instant::now(),
            prev_io_read: 0.0,
            prev_io_write: 0.0,
            last_io_time: Instant::now(),
            prev_major_faults: 0.0,
            rss_ceiling_mb: 0.0,
            thread_ceiling: 0.0,
            cpu_ceiling: 0.0,
            extra_budget: Vec::new(),
            alert: String::new(),
            rss_readout: String::new(),
            thread_readout: String::new(),
            pgfault_readout: String::new(),
            pcache_readout: String::new(),
            show_help: false,
            step_yaml: String::new(),
            step_yaml_height: 8,
            yaml_width_pct: 45,
            paused: false,
            pause_redraw_pending: false,
            suspended: false,
            dirty: false,
            recent_logs: Vec::new(),
        }
    }

    /// Number of lines needed for the progress region.
    fn visible_lines(&self) -> u16 {
        if self.show_help {
            return 9; // fixed help overlay height
        }
        let context_line = if self.context_label.is_empty() { 0u16 } else { 1 };
        let bar_lines = self.bar_order.len() as u16;
        let has_rps = self.rps_history.data.len() >= 2;
        let has_yaml = !self.step_yaml.is_empty();
        // The top panel shows YAML + throughput side-by-side. Its height
        // is the YAML panel height when YAML is present, otherwise just
        // the RPS chart height (8 lines).
        let top_panel_lines = if has_yaml {
            self.step_yaml_height
        } else if has_rps {
            8u16
        } else {
            0
        };
        let has_extra = !self.alert.is_empty() || !self.extra_budget.is_empty();
        let extra_line = if has_extra { 1u16 } else { 0 };
        let chart_lines = if !self.resource_status.is_empty() && self.rss_history.data.len() >= 2 {
            10u16 // base chart height (capped at render time)
        } else {
            0
        };
        let status_lines = extra_line + chart_lines;
        let paused_line = if self.paused { 1 } else { 0 };
        // Log pane fills remaining space — no fixed cap. The layout
        // gives it Min(1) and it scrolls internally.
        let log_lines = 0u16; // 0 = use Min constraint instead of fixed Length
        context_line + bar_lines + top_panel_lines + status_lines + paused_line + log_lines
    }
}

/// Restore the terminal to a usable state: show cursor, disable raw mode,
/// and flush stdout.
fn restore_terminal<B: ratatui::backend::Backend + io::Write>(terminal: &mut Terminal<B>) {
    // Leave raw mode first so the terminal processes escape sequences normally
    let _ = disable_raw_mode();
    let _ = crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        cursor::Show,
        // Move cursor to start of line and clear below to prevent
        // TUI frame artifacts from bleeding into the normal screen
        crossterm::cursor::MoveToColumn(0),
        crossterm::terminal::Clear(crossterm::terminal::ClearType::FromCursorDown),
    );
    let _ = io::Write::flush(terminal.backend_mut());
}

/// Main render loop running on the background thread.
fn render_loop(
    rx: mpsc::Receiver<RenderMsg>,
    max_height: u16,
    redraw_interval: Duration,
    console_log: Arc<Mutex<Vec<String>>>,
) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = RenderState::new();
    let min_redraw_interval = redraw_interval;
    let mut last_draw = Instant::now() - min_redraw_interval;
    let poll_timeout = Duration::from_millis(100);
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
                            if now.duration_since(prev) < Duration::from_millis(250) {
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
                } else if key.code == KeyCode::Up {
                    if state.step_yaml_height > 4 {
                        state.step_yaml_height -= 1;
                        state.dirty = true;
                    }
                } else if key.code == KeyCode::Down {
                    if state.step_yaml_height < 30 {
                        state.step_yaml_height += 1;
                        state.dirty = true;
                    }
                } else if key.code == KeyCode::Left {
                    if state.yaml_width_pct > 15 {
                        state.yaml_width_pct -= 5;
                        state.dirty = true;
                    }
                } else if key.code == KeyCode::Right {
                    if state.yaml_width_pct < 85 {
                        state.yaml_width_pct += 5;
                        state.dirty = true;
                    }
                } else if key.code == KeyCode::Char('t') {
                    state.rps_history.toggle_pinned();
                    state.dirty = true;
                }
            }
        }

        // Drain all queued render messages without blocking.
        let mut pending_logs: Vec<String> = Vec::new();
        let mut pending_emits: Vec<(String, bool)> = Vec::new();
        let mut got_message = false;

        while let Ok(msg) = rx.try_recv() {
            // Buffer log messages for post-TUI console output
            if let RenderMsg::Event(UiEvent::Log { ref message }) = msg {
                if let Ok(mut buf) = console_log.lock() {
                    buf.push(message.clone());
                }
            }
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

        // Sample RPS from bar position deltas.
        let rps_dt = state.last_rps_sample.elapsed().as_secs_f64();
        if rps_dt >= 0.25 && !state.bar_order.is_empty() {
            let mut total_delta: u64 = 0;
            for id in &state.bar_order {
                if let Some(bar) = state.bars.get(id) {
                    if bar.kind == ProgressKind::Bar {
                        let prev = state.prev_positions.get(id).copied().unwrap_or(0);
                        total_delta += bar.position.saturating_sub(prev);
                        state.prev_positions.insert(*id, bar.position);
                    }
                }
            }
            let rps = total_delta as f64 / rps_dt;
            state.rps_history.push(rps);
            state.last_rps_sample = Instant::now();
            state.dirty = true;
        }

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
        UiEvent::ProgressCreate { id, kind, total, label, unit } => {
            state.bars.insert(*id, BarState {
                kind: kind.clone(),
                total: *total,
                position: 0,
                label: label.clone(),
                message: String::new(),
                created_at: Instant::now(),
                rate_anchor: None,
                unit: unit.clone(),
            });
            state.bar_order.push(*id);
            state.dirty = true;
        }

        UiEvent::ProgressAnchorRate { id } => {
            if let Some(bar) = state.bars.get_mut(id) {
                bar.rate_anchor = Some((bar.position, Instant::now()));
            }
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
            state.prev_positions.remove(id);
            state.dirty = true;
        }

        UiEvent::Log { message } => {
            logs.push(message.clone());
            // Keep recent logs for the TUI log window
            state.recent_logs.push(message.clone());
            state.dirty = true;
        }

        UiEvent::Emit { text } => {
            emits.push((text.clone(), false));
        }

        UiEvent::EmitLn { text } => {
            emits.push((text.clone(), true));
        }

        UiEvent::ResourceStatus { line, metrics } => {
            let now = Instant::now();
            state.resource_status = line.clone();
            state.extra_budget.clear();
            state.alert.clear();

            // Use structured metrics directly for chart data — no text parsing.
            let m = &metrics;

            // RSS
            let rss_mb = bytes_to_mb(m.rss_bytes);
            state.rss_history.push(rss_mb);
            if m.rss_ceiling_bytes > 0 {
                state.rss_ceiling_mb = bytes_to_mb(m.rss_ceiling_bytes);
                state.rss_readout = format!("{}/{}", format_mb(rss_mb), format_mb(state.rss_ceiling_mb));
            } else {
                state.rss_readout = format_mb(rss_mb);
            }

            // CPU
            state.cpu_history.push(m.cpu_pct);
            state.cpu_ceiling = m.cpu_ceiling;

            // Per-core CPU
            state.cpu_cores = m.cpu_cores.clone();

            // Threads
            state.thread_history.push(m.threads as f64);
            state.thread_ceiling = m.thread_ceiling as f64;
            if m.thread_ceiling > 0 {
                state.thread_readout = format!("{}/{}", m.threads, m.thread_ceiling);
            } else {
                state.thread_readout = format!("{}", m.threads);
            }

            // I/O throughput — cumulative bytes, compute rate from deltas
            let io_read_mb = bytes_to_mb(m.io_read_bytes);
            let io_write_mb = bytes_to_mb(m.io_write_bytes);
            let dt = now.duration_since(state.last_io_time).as_secs_f64();
            if state.prev_io_read > 0.0 && dt > 0.0 {
                state.io_read_history.push(((io_read_mb - state.prev_io_read) / dt).max(0.0));
            } else {
                state.io_read_history.push(0.0);
            }
            if state.prev_io_write > 0.0 && dt > 0.0 {
                state.io_write_history.push(((io_write_mb - state.prev_io_write) / dt).max(0.0));
            } else {
                state.io_write_history.push(0.0);
            }
            state.prev_io_read = io_read_mb;
            state.prev_io_write = io_write_mb;

            // I/O queue depth
            state.ioq_read_history.push(m.ioq_read as f64);
            state.ioq_write_history.push(m.ioq_write as f64);

            // Page cache
            let pcache_mb = bytes_to_mb(m.page_cache_bytes);
            state.pcache_history.push(pcache_mb);
            state.pcache_readout = match m.page_cache_hit_pct {
                Some(pct) => format!("{} hit:{:.0}%", format_mb(pcache_mb), pct),
                None => format_mb(pcache_mb),
            };

            // Page faults — track major fault rate from cumulative deltas
            let major = m.major_faults as f64;
            if state.prev_major_faults > 0.0 && dt > 0.0 {
                state.pgfault_history.push(((major - state.prev_major_faults) / dt).max(0.0));
            } else {
                state.pgfault_history.push(0.0);
            }
            state.prev_major_faults = major;
            state.pgfault_readout = format!("major:{} minor:{}",
                format_count(m.major_faults), format_count(m.minor_faults));

            // Alert state
            if m.emergency {
                state.alert = "⚠ EMERGENCY".to_string();
            } else if m.throttle {
                state.alert = "⏳ throttle".to_string();
            }

            // Collect extra budget items from the text line (non-standard resources)
            for part in line.split(" | ") {
                let trimmed = part.trim();
                if trimmed.starts_with("rss:") || trimmed.starts_with("cpu:")
                    || trimmed.starts_with("cpu_cores:") || trimmed.starts_with("threads:")
                    || trimmed.starts_with("io_r:") || trimmed.starts_with("io_w:")
                    || trimmed.starts_with("ioq_r:") || trimmed.starts_with("ioq_w:")
                    || trimmed.starts_with("ioq:") || trimmed.starts_with("pcache:")
                    || trimmed.starts_with("pgfault:")
                    || trimmed.contains("EMERGENCY") || trimmed.contains("throttle")
                    || trimmed.starts_with("segmentsize")
                {
                    continue;
                }
                state.extra_budget.push(trimmed.to_string());
            }

            state.last_io_time = now;
            state.dirty = true;
        }

        UiEvent::SetContext { label } => {
            state.context_label = label.clone();
            state.dirty = true;
        }

        UiEvent::SetStepYaml { yaml } => {
            state.step_yaml = yaml.clone();
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
            // Reset per-step state but preserve persistent bars (like the
            // overall pipeline progress bar). Bars from the previous step
            // will have already been finished/dropped by their ProgressHandle.
            // Only remove bars that are already finished (position == total).
            let finished_ids: Vec<ProgressId> = state.bars.iter()
                .filter(|(_, bar)| bar.kind == ProgressKind::Bar && bar.position >= bar.total && bar.total > 0)
                .map(|(id, _)| *id)
                .collect();
            for id in &finished_ids {
                state.bars.remove(id);
                state.bar_order.retain(|x| x != id);
                state.prev_positions.remove(id);
            }
            state.resource_status.clear();
            state.rps_history.clear();
            state.step_yaml.clear();
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

        // Build constraints for full-height layout.
        //
        // Layout priority (top to bottom):
        //   1. Context header line (1 line, if present)
        //   2. Progress bars (1 line each)
        //   3. Top panel: YAML + RPS chart (20 lines or 1/3 of remaining, whichever is greater)
        //   4. Alert/extra budget line (1 line, if present)
        //   5. Metrics chart (10 lines or 1/4 of total height, whichever is more, capped)
        //   6. Paused indicator (1 line, if paused)
        //   7. Log window (fills remaining vertical space)
        let has_chart = state.rss_history.data.len() >= 2;
        let has_rps_chart = state.rps_history.data.len() >= 2;
        let has_yaml = !state.step_yaml.is_empty();
        let has_top_panel = has_yaml || has_rps_chart;
        let has_extra = !state.alert.is_empty() || !state.extra_budget.is_empty();
        let has_resource_chart = !state.resource_status.is_empty() && has_chart;

        // Calculate fixed-height rows consumed by chrome
        let context_lines: u16 = if state.context_label.is_empty() { 0 } else { 1 };
        let bar_lines = state.bar_order.len() as u16;
        let extra_lines: u16 = if has_extra { 1 } else { 0 };
        let paused_lines: u16 = if state.paused { 1 } else { 0 };

        // Metrics chart height
        let chart_height: u16 = if has_resource_chart {
            let quarter = area.height / 4;
            quarter.max(10).min(area.height.saturating_sub(8))
        } else {
            0
        };

        // Remaining height after chrome and metrics
        let chrome = context_lines + bar_lines + extra_lines + paused_lines + chart_height;
        let remaining = area.height.saturating_sub(chrome);

        // Reserve minimum space for the log pane
        let min_log_lines: u16 = 6;

        // Top panel: 20 lines or 1/3 of remaining, whichever is greater,
        // but always leave room for the log pane
        let top_height: u16 = if has_top_panel {
            let max_top = remaining.saturating_sub(min_log_lines);
            let third = remaining / 3;
            third.max(12).min(max_top)
        } else {
            0
        };

        // Log window: everything left over (at least min_log_lines)
        let log_height = remaining.saturating_sub(top_height).max(min_log_lines.min(remaining));

        let mut constraints: Vec<Constraint> = Vec::new();

        if context_lines > 0 {
            constraints.push(Constraint::Length(1));
        }
        for _ in &state.bar_order {
            constraints.push(Constraint::Length(1));
        }
        if has_top_panel {
            constraints.push(Constraint::Length(top_height));
        }
        if has_extra {
            constraints.push(Constraint::Length(1));
        }
        if has_resource_chart {
            constraints.push(Constraint::Length(chart_height));
        }
        if state.paused {
            constraints.push(Constraint::Length(1));
        }
        // Log window always present — fills remaining space
        constraints.push(Constraint::Length(log_height));
        let has_logs = true; // always render the log panel

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
                Span::styled("  (? help  ↑↓ height  ←→ width)", Style::default().fg(Color::DarkGray)),
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

        // Top panel: YAML panel (left) + RPS chart (right), or just RPS
        if has_top_panel {
            if idx < chunks.len() {
                let top_area = chunks[idx];
                if has_yaml && has_rps_chart {
                    // Side-by-side: YAML left, RPS right
                    let halves = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([
                            Constraint::Percentage(state.yaml_width_pct),
                            Constraint::Percentage(100 - state.yaml_width_pct),
                        ])
                        .split(top_area);
                    render_step_yaml(frame, halves[0], state);
                    render_rps_chart(frame, halves[1], state);
                } else if has_yaml {
                    // YAML only (no RPS data yet)
                    render_step_yaml(frame, top_area, state);
                } else {
                    // RPS only (no step YAML)
                    render_rps_chart(frame, top_area, state);
                }
            }
            idx += 1;
        }

        // Alert / extra budget line
        if has_extra {
            if idx < chunks.len() {
                let mut spans: Vec<Span> = Vec::new();
                spans.push(Span::styled(
                    " ▸ ",
                    Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
                ));
                if !state.alert.is_empty() {
                    let (alert_color, alert_mod) = if state.alert.contains("EMERGENCY") {
                        (Color::Red, Modifier::BOLD)
                    } else {
                        (Color::Yellow, Modifier::empty())
                    };
                    spans.push(Span::styled(
                        &state.alert,
                        Style::default().fg(alert_color).add_modifier(alert_mod),
                    ));
                    if !state.extra_budget.is_empty() {
                        spans.push(Span::styled(" │ ", Style::default().fg(Color::DarkGray)));
                    }
                }
                for (i, item) in state.extra_budget.iter().enumerate() {
                    if i > 0 {
                        spans.push(Span::styled(" │ ", Style::default().fg(Color::DarkGray)));
                    }
                    if let Some(colon_pos) = item.find(':') {
                        let label = &item[..=colon_pos];
                        let value = &item[colon_pos + 1..];
                        spans.push(Span::styled(label, Style::default().fg(Color::DarkGray)));
                        spans.push(Span::styled(value, Style::default().fg(Color::White)));
                    } else {
                        spans.push(Span::raw(item));
                    }
                }
                frame.render_widget(Paragraph::new(Line::from(spans)), chunks[idx]);
            }
            idx += 1;
        }

        // Resource charts
        if !state.resource_status.is_empty() && has_chart {
            if idx < chunks.len() {
                render_resource_chart(frame, chunks[idx], state);
            }
            idx += 1;
        }

        if state.paused && idx < chunks.len() {
            let paused_line = Line::from(Span::styled(
                " ⏸  PAUSED — press 'p' to resume",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ));
            frame.render_widget(Paragraph::new(paused_line), chunks[idx]);
            idx += 1;
        }

        // Log window at the bottom — show the last N lines that fit
        if has_logs && idx < chunks.len() {
            let visible_height = chunks[idx].height as usize;
            let skip = state.recent_logs.len().saturating_sub(visible_height);
            let log_lines: Vec<Line> = state.recent_logs.iter()
                .skip(skip)
                .map(|s| colorize_log(s))
                .collect();
            let log_widget = Paragraph::new(log_lines)
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(log_widget, chunks[idx]);
        }
    })?;

    Ok(())
}

/// Convert raw bytes to megabytes (MiB) for chart data.
fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
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
        Line::from(vec![
            Span::styled("  ↑ / ↓   ", Style::default().fg(Color::Yellow)),
            Span::raw("Adjust panel height"),
        ]),
        Line::from(vec![
            Span::styled("  ← / →   ", Style::default().fg(Color::Yellow)),
            Span::raw("Adjust step detail / RPS chart split"),
        ]),
        Line::from(vec![
            Span::styled("  t       ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle throughput: scrolling window / pinned history"),
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
    title: &'a str,
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
                .title(Span::styled(title, Style::default().fg(color)))
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

/// Render the current step's YAML snippet in a bordered panel.
fn render_step_yaml(frame: &mut ratatui::Frame, area: Rect, state: &RenderState) {
    let dim = Style::default().fg(Color::DarkGray);
    let input_label = Style::default().fg(Color::Green).add_modifier(Modifier::BOLD);
    let output_label = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);
    let config_label = Style::default().fg(Color::DarkGray);
    let key_style = Style::default().fg(Color::Cyan);
    let val_style = Style::default().fg(Color::White);
    let input_val = Style::default().fg(Color::Green);
    let output_val = Style::default().fg(Color::Yellow);

    // Track current section to color values appropriately
    let mut current_section = "";

    let lines: Vec<Line> = state.step_yaml.lines().map(|line| {
        let trimmed = line.trim();
        // Section headers
        if trimmed == "inputs:" {
            current_section = "inputs";
            return Line::from(Span::styled(line, input_label));
        } else if trimmed == "outputs:" {
            current_section = "outputs";
            return Line::from(Span::styled(line, output_label));
        } else if trimmed == "config:" {
            current_section = "config";
            return Line::from(Span::styled(line, config_label));
        }

        // Key: value lines
        if let Some(colon_pos) = line.find(':') {
            let key_part = &line[..colon_pos + 1];
            let val_part = &line[colon_pos + 1..];
            let val_color = match current_section {
                "inputs" => input_val,
                "outputs" => output_val,
                _ => val_style,
            };
            Line::from(vec![
                Span::styled(key_part, key_style),
                Span::styled(val_part, val_color),
            ])
        } else {
            Line::from(Span::styled(line, dim))
        }
    }).collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" step ", Style::default().fg(Color::Cyan)))
        .border_style(dim);
    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

/// Render the RPS time-series chart.
fn render_rps_chart(frame: &mut ratatui::Frame, area: Rect, state: &RenderState) {
    let dim = Style::default().fg(Color::DarkGray);
    // Use the most recent bar's unit for the chart, falling back to "rec".
    let chart_unit = state.bar_order.last()
        .and_then(|id| state.bars.get(id))
        .map(|b| b.unit.as_str())
        .unwrap_or("rec");
    let y_max = state.rps_history.y_bounds()[1];
    let y_label = format_rate(y_max, chart_unit);
    let title_label = {
        let data = state.rps_history.as_slice();
        let current = data.last().map(|p| p.1).unwrap_or(0.0);
        // 1-minute windowed average (~240 samples at 250ms interval)
        let window = 240usize;
        let avg_1m = if data.len() >= 2 {
            let start = data.len().saturating_sub(window);
            let slice = &data[start..];
            let sum: f64 = slice.iter().map(|p| p.1).sum();
            sum / slice.len() as f64
        } else {
            current
        };
        let pin_tag = if state.rps_history.is_pinned() { " [pinned]" } else { "" };
        if chart_unit == "bytes" {
            format!(" throughput: {} (1m avg: {}){} ",
                format_rate(current, chart_unit), format_rate(avg_1m, chart_unit), pin_tag)
        } else {
            format!(" throughput: {}{} ", format_rate(current, chart_unit), pin_tag)
        }
    };

    // Compute the 1-minute moving average (240 samples at 250ms)
    let ma_data = state.rps_history.moving_average(240);

    let mut datasets = vec![
        Dataset::default()
            .name("rps")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(state.rps_history.as_slice()),
    ];
    // Render moving average as a dimmer second line (only when there
    // are enough samples for it to differ meaningfully from the raw data).
    if ma_data.len() >= 10 {
        datasets.push(
            Dataset::default()
                .name("1m avg")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(&ma_data),
        );
    }

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(title_label, Style::default().fg(Color::Yellow)))
                .border_style(dim),
        )
        .x_axis(Axis::default().bounds(state.rps_history.x_bounds()).style(dim))
        .y_axis(
            Axis::default()
                .bounds(state.rps_history.y_bounds())
                .labels(vec![
                    Span::styled("0", dim),
                    Span::styled(y_label, dim),
                ])
                .style(dim),
        );

    frame.render_widget(chart, area);
}

/// Render a dual-series (read/write) time-series chart.
fn rw_chart<'a>(
    read_history: &'a MetricsHistory,
    write_history: &'a MetricsHistory,
    title_spans: Vec<Span<'a>>,
    dim: Style,
    y_format: fn(f64) -> String,
) -> Chart<'a> {
    let mut datasets = Vec::new();
    let mut y_max = 1.0f64;
    if !read_history.is_empty() {
        y_max = y_max.max(read_history.max_value * 1.25);
        datasets.push(
            Dataset::default()
                .name("read")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(read_history.as_slice()),
        );
    }
    if !write_history.is_empty() {
        y_max = y_max.max(write_history.max_value * 1.25);
        datasets.push(
            Dataset::default()
                .name("write")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(write_history.as_slice()),
        );
    }
    let x_bounds = if !read_history.is_empty() {
        read_history.x_bounds()
    } else {
        write_history.x_bounds()
    };
    let label = y_format(y_max);
    Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Line::from(title_spans))
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
        )
}

/// Render the resource metrics charts: RSS, CPU, I/O, threads, page faults.
///
/// Each chart's title includes the current readout value so that the separate
/// status text line is no longer needed.
fn render_resource_chart(frame: &mut ratatui::Frame, area: Rect, state: &RenderState) {
    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(16), // RSS
            Constraint::Percentage(14), // page cache (adjacent to RSS)
            Constraint::Percentage(10), // faults (adjacent to page cache)
            Constraint::Percentage(14), // CPU
            Constraint::Percentage(18), // I/O throughput
            Constraint::Percentage(14), // I/O queue
            Constraint::Percentage(14), // threads
        ])
        .split(area);

    let dim = Style::default().fg(Color::DarkGray);

    // RSS (magenta) — title shows current/ceiling
    if !state.rss_history.is_empty() {
        let title = if state.rss_readout.is_empty() {
            " rss ".to_string()
        } else {
            format!(" rss {} ", state.rss_readout)
        };
        let label = format_mb(state.rss_history.y_bounds()[1]);
        frame.render_widget(
            metric_chart(&state.rss_history, &title, Color::Magenta, label, dim),
            panels[0],
        );
    }

    // Page cache (yellow) — adjacent to RSS
    if !state.pcache_history.is_empty() {
        let title = if state.pcache_readout.is_empty() {
            " pcache ".to_string()
        } else {
            format!(" pcache {} ", state.pcache_readout)
        };
        let label = format_mb(state.pcache_history.y_bounds()[1]);
        frame.render_widget(
            metric_chart(&state.pcache_history, &title, Color::Yellow, label, dim),
            panels[1],
        );
    }

    // Page faults (red) — adjacent to page cache for correlation
    if !state.pgfault_history.is_empty() {
        let title = if state.pgfault_readout.is_empty() {
            " faults ".to_string()
        } else {
            format!(" faults {} ", state.pgfault_readout)
        };
        let label = format!("{}/s", state.pgfault_history.y_bounds()[1] as u64);
        frame.render_widget(
            metric_chart(&state.pgfault_history, &title, Color::Red, label, dim),
            panels[2],
        );
    }

    // CPU — per-core sparkline when available, otherwise time-series chart
    if !state.cpu_cores.is_empty() {
        let active = state.cpu_cores.iter().filter(|&&v| v > 0).count();
        let total = state.cpu_cores.len();
        let avg = if total > 0 {
            state.cpu_cores.iter().sum::<u64>() / total as u64
        } else {
            0
        };
        let title = format!(" cpu {}/{} avg {}% ", active, total, avg);
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(Span::styled(title, Style::default().fg(Color::Green)))
                    .border_style(dim),
            )
            .data(&state.cpu_cores)
            .max(100)
            .style(Style::default().fg(Color::Green));
        frame.render_widget(sparkline, panels[3]);
    } else if !state.cpu_history.is_empty() {
        let label = format!("{}%", state.cpu_history.y_bounds()[1] as u64);
        frame.render_widget(
            metric_chart(&state.cpu_history, " cpu ", Color::Green, label, dim),
            panels[3],
        );
    }

    // I/O throughput — read (cyan) and write (yellow)
    if !state.io_read_history.is_empty() || !state.io_write_history.is_empty() {
        let r_rate = state.io_read_history.as_slice().last().map(|p| p.1).unwrap_or(0.0);
        let w_rate = state.io_write_history.as_slice().last().map(|p| p.1).unwrap_or(0.0);
        let title_spans = vec![
            Span::styled(" io ", Style::default().fg(Color::White)),
            Span::styled(format!("r:{}/s", format_mb(r_rate)), Style::default().fg(Color::Cyan)),
            Span::styled(" ", dim),
            Span::styled(format!("w:{}/s ", format_mb(w_rate)), Style::default().fg(Color::Yellow)),
        ];
        frame.render_widget(
            rw_chart(
                &state.io_read_history,
                &state.io_write_history,
                title_spans,
                dim,
                |y| format!("{}/s", format_mb(y)),
            ),
            panels[4],
        );
    }

    // I/O queue depth — read (cyan) and write (yellow)
    if !state.ioq_read_history.is_empty() || !state.ioq_write_history.is_empty() {
        let r_cur = state.ioq_read_history.as_slice().last().map(|p| p.1 as u64).unwrap_or(0);
        let w_cur = state.ioq_write_history.as_slice().last().map(|p| p.1 as u64).unwrap_or(0);
        let title_spans = vec![
            Span::styled(" ioq ", Style::default().fg(Color::White)),
            Span::styled(format!("r:{}", r_cur), Style::default().fg(Color::Cyan)),
            Span::styled(" ", dim),
            Span::styled(format!("w:{} ", w_cur), Style::default().fg(Color::Yellow)),
        ];
        frame.render_widget(
            rw_chart(
                &state.ioq_read_history,
                &state.ioq_write_history,
                title_spans,
                dim,
                |y| format!("{}", y as u64),
            ),
            panels[5],
        );
    }

    // Threads (blue) — title shows current/ceiling
    if !state.thread_history.is_empty() {
        let title = if state.thread_readout.is_empty() {
            " threads ".to_string()
        } else {
            format!(" threads {} ", state.thread_readout)
        };
        let label = format!("{}", state.thread_history.y_bounds()[1] as u64);
        frame.render_widget(
            metric_chart(&state.thread_history, &title, Color::Blue, label, dim),
            panels[6],
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
            // Rate calc respects an explicit anchor (set when the
            // caller wants to exclude an early burst of free progress
            // from the rate average — e.g. cache-skipped shards).
            // Falls back to bar-creation time and full position when
            // unanchored.
            let (rate_pos, rate_elapsed) = match bar.rate_anchor {
                Some((p0, t0)) => (
                    bar.position.saturating_sub(p0),
                    t0.elapsed().as_secs_f64(),
                ),
                None => (bar.position, bar.created_at.elapsed().as_secs_f64()),
            };
            let rate_eta = if rate_elapsed > 0.5 && rate_pos > 0 {
                let rps = rate_pos as f64 / rate_elapsed;
                let rate = format_rate(rps, &bar.unit);
                let eta = if bar.total > bar.position && rps > 0.0 {
                    let remaining = (bar.total - bar.position) as f64;
                    format!(" eta {}", format_duration(remaining / rps))
                } else {
                    String::new()
                };
                if rate.is_empty() { eta } else { format!(" {}{}", rate, eta) }
            } else {
                String::new()
            };
            let (pos_str, total_str) = if bar.unit == "bytes" {
                (format_bytes(bar.position), format_bytes(bar.total))
            } else {
                (format_count(bar.position), format_count(bar.total))
            };
            let label_str = if bar.message.is_empty() {
                format!("{} [{}/{}] {}%{}", bar.label, pos_str, total_str, pct, rate_eta)
            } else {
                format!("{} [{}/{}] {}%{} {}", bar.label, pos_str, total_str, pct, rate_eta, bar.message)
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
            let elapsed = bar.created_at.elapsed().as_secs_f64();
            let elapsed_str = if elapsed > 0.5 {
                format!(" [{}]", format_duration(elapsed))
            } else {
                String::new()
            };
            let label_str = if bar.message.is_empty() {
                format!("{}{}", bar.label, elapsed_str)
            } else {
                format!("{} {}{}", bar.label, bar.message, elapsed_str)
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

/// Format a rate value as a compact human-readable string.
///
/// When rate >= 1/s, shows records per second (e.g. "42 rec/s", "1.5K rec/s").
/// When rate < 1/s, flips to seconds per unit (e.g. "2.0s/file",
/// "1.5m/chunk") so slow operations still show meaningful progress info.
///
/// The `unit` parameter controls the label (e.g., "rec", "files", "chunks").
fn format_rate(rps: f64, unit: &str) -> String {
    // Byte rates use IEC-style formatting (MB/s, GB/s)
    if unit == "bytes" {
        if rps >= 1024.0 * 1024.0 * 1024.0 {
            return format!("{:.1} GB/s", rps / (1024.0 * 1024.0 * 1024.0));
        } else if rps >= 1024.0 * 1024.0 {
            return format!("{:.1} MB/s", rps / (1024.0 * 1024.0));
        } else if rps >= 1024.0 {
            return format!("{:.1} KB/s", rps / 1024.0);
        } else if rps >= 1.0 {
            return format!("{:.0} B/s", rps);
        } else if rps > 0.0 {
            return String::new();
        } else {
            return String::new();
        }
    }

    if rps >= 1_000_000_000.0 {
        format!("{:.1}G {}/s", rps / 1_000_000_000.0, unit)
    } else if rps >= 1_000_000.0 {
        format!("{:.1}M {}/s", rps / 1_000_000.0, unit)
    } else if rps >= 1_000.0 {
        format!("{:.1}K {}/s", rps / 1_000.0, unit)
    } else if rps >= 1.0 {
        format!("{:.0} {}/s", rps, unit)
    } else if rps > 0.0 {
        // Inverted rate: show time per unit (e.g., "14.6s /step").
        // The space before the slash prevents visual misreading — without
        // it "14.6s/steps" scans as "14.6 steps/s" at a glance.
        // Also singularize the unit ("step" not "steps") since it's per-one.
        let spt = 1.0 / rps;
        let singular_owned;
        let singular: &str = if unit == "bytes" || unit.len() <= 1 {
            unit
        } else if unit.ends_with("ies") && unit.len() > 3 {
            // queries → query, entries → entry
            singular_owned = format!("{}y", &unit[..unit.len() - 3]);
            &singular_owned
        } else if unit.ends_with('s') {
            &unit[..unit.len() - 1]
        } else {
            unit
        };
        if spt >= 3600.0 {
            format!("{:.0}h /{}", spt / 3600.0, singular)
        } else if spt >= 60.0 {
            format!("{:.1}m /{}", spt / 60.0, singular)
        } else {
            format!("{:.1}s /{}", spt, singular)
        }
    } else {
        String::new()
    }
}

/// Format a duration in seconds as a compact human-readable string.
fn format_duration(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        format!("{}m{}s", s / 60, s % 60)
    } else {
        let h = s / 3600;
        let m = (s % 3600) / 60;
        format!("{}h{}m", h, m)
    }
}

/// Format a byte count as human-readable (KB/MB/GB/TB).
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes < 1024u64 * 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{:.2} TB", bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0))
    }
}

/// Format a count with thousands separators.
fn format_count(n: u64) -> String {
    let n = n as usize;
    if n >= 1_000_000_000 && n % 1_000_000_000 == 0 {
        format!("{}B", n / 1_000_000_000)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 && n % 1_000_000 == 0 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 && n % 1_000 == 0 {
        format!("{}K", n / 1_000)
    } else if n >= 10_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::event::ResourceMetrics;

    #[test]
    fn format_count_iso_suffixes() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1K");
        assert_eq!(format_count(1_234_567), "1.2M");
        assert_eq!(format_count(5_000_000), "5M");
        assert_eq!(format_count(2_000_000_000), "2B");
    }

    #[test]
    fn format_rate_compact() {
        // High rates: records per second (default unit)
        assert_eq!(format_rate(42.0, "rec"), "42 rec/s");
        assert_eq!(format_rate(1_500.0, "rec"), "1.5K rec/s");
        assert_eq!(format_rate(2_300_000.0, "rec"), "2.3M rec/s");
        assert_eq!(format_rate(1_200_000_000.0, "rec"), "1.2G rec/s");

        // Sub-1/s: flips to time per unit with space before slash
        assert_eq!(format_rate(0.5, "rec"), "2.0s /rec");
        assert_eq!(format_rate(0.1, "rec"), "10.0s /rec");
        assert_eq!(format_rate(1.0 / 90.0, "rec"), "1.5m /rec");
        assert_eq!(format_rate(1.0 / 7200.0, "rec"), "2h /rec");

        // Zero: empty
        assert_eq!(format_rate(0.0, "rec"), "");

        // Custom units
        assert_eq!(format_rate(42.0, "files"), "42 files/s");
        assert_eq!(format_rate(1_500.0, "chunks"), "1.5K chunks/s");
        // Plural unit is singularized in inverted format
        assert_eq!(format_rate(0.5, "vectors"), "2.0s /vector");
        assert_eq!(format_rate(0.5, "steps"), "2.0s /step");
        assert_eq!(format_rate(0.5, "queries"), "2.0s /query");
        assert_eq!(format_rate(0.5, "entries"), "2.0s /entry");
    }

    #[test]
    fn downsampling_history_pinned() {
        let mut h = DownsamplingHistory::new(8);
        h.pinned = true; // test pinned (keep+resample) mode
        for i in 0..8 {
            h.push(i as f64);
        }
        assert_eq!(h.data.len(), 8);
        assert_eq!(h.stride, 1);

        // 9th push: stride is still 1 when pending check runs, so value is
        // emitted immediately. But data.len()==8 triggers compact first
        // (8→4, stride=2), then the point is appended → 5 total.
        h.push(100.0);
        assert_eq!(h.stride, 2);
        assert_eq!(h.data.len(), 5);
        assert!((h.data[4].1 - 100.0).abs() < 0.001);

        // 10th push: stride=2, pending_count=1 < 2 → still pending
        h.push(200.0);
        assert_eq!(h.data.len(), 5);

        // 11th push: pending_count=2 >= 2 → emit average of 200+300=250
        h.push(300.0);
        assert_eq!(h.data.len(), 6);
        assert!((h.data[5].1 - 250.0).abs() < 0.001);
    }

    #[test]
    fn downsampling_history_scrolling() {
        let mut h = DownsamplingHistory::new(8);
        // Default is scrolling mode
        assert!(!h.is_pinned());
        for i in 0..8 {
            h.push(i as f64);
        }
        assert_eq!(h.data.len(), 8);

        // 9th push drops oldest, keeps 8 most recent
        h.push(100.0);
        assert_eq!(h.data.len(), 8);
        assert_eq!(h.stride, 1); // no compaction in scrolling mode
        assert!((h.data[7].1 - 100.0).abs() < 0.001);
        // First element should be 1.0 (0.0 was dropped)
        assert!((h.data[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn downsampling_history_toggle() {
        let mut h = DownsamplingHistory::new(8);
        for i in 0..5 {
            h.push(i as f64);
        }
        assert!(!h.is_pinned());

        // Pin it
        h.toggle_pinned();
        assert!(h.is_pinned());
        assert_eq!(h.data.len(), 5);

        // Unpin
        h.toggle_pinned();
        assert!(!h.is_pinned());
        assert_eq!(h.data.len(), 5); // data preserved
    }

    #[test]
    fn downsampling_history_x_stretches() {
        let mut h = DownsamplingHistory::new(16);
        for i in 0..10 {
            h.push(i as f64);
        }
        let bounds = h.x_bounds();
        assert_eq!(bounds[0], 0.0);
        assert_eq!(bounds[1], 10.0);
        // X coords are sequential 0..N
        for (i, pt) in h.as_slice().iter().enumerate() {
            assert_eq!(pt.0, i as f64);
        }
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
            rate_anchor: None,
            unit: "rec".into(),
        });
        assert_eq!(state.visible_lines(), 1);

        // Resource status without chart data — no extra lines (no status text line)
        state.resource_status = "mem: 1.2G / 4G".into();
        assert_eq!(state.visible_lines(), 1);

        // With alert, adds an extra line
        state.alert = "⏳ throttle".into();
        assert_eq!(state.visible_lines(), 2);
        state.alert.clear();

        // With chart data, adds chart area
        for i in 0..5 {
            state.rss_history.push(i as f64 * 100.0);
        }
        assert_eq!(state.visible_lines(), 1 + 10); // bar + chart
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
                unit: "rec".into(),
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
            rate_anchor: None,
            unit: "rec".into(),
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
            rate_anchor: None,
            unit: "rec".into(),
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
            rate_anchor: None,
            unit: "rec".into(),
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
                rate_anchor: None,
                unit: "rec".into(),
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
    fn clear_removes_finished_bars_and_resets_status() {
        let mut state = RenderState::new();
        let mut logs = Vec::new();
        let mut emits = Vec::new();

        // Add an unfinished bar (position < total) and a finished bar
        let live_id = ProgressId(0);
        let done_id = ProgressId(1);
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressCreate {
                id: live_id,
                kind: ProgressKind::Bar,
                total: 100,
                label: "live".into(),
                unit: "rec".into(),
            }),
            &mut state, &mut logs, &mut emits,
        );
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressCreate {
                id: done_id,
                kind: ProgressKind::Bar,
                total: 50,
                label: "done".into(),
                unit: "rec".into(),
            }),
            &mut state, &mut logs, &mut emits,
        );
        // Finish the second bar
        process_msg(
            &RenderMsg::Event(UiEvent::ProgressUpdate { id: done_id, position: 50 }),
            &mut state, &mut logs, &mut emits,
        );
        process_msg(
            &RenderMsg::Event(UiEvent::ResourceStatus {
                line: "status".into(),
                metrics: ResourceMetrics::default(),
            }),
            &mut state, &mut logs, &mut emits,
        );

        assert_eq!(state.bars.len(), 2);
        assert!(!state.resource_status.is_empty());

        // Clear: removes finished bars, keeps live ones, clears status
        process_msg(
            &RenderMsg::Event(UiEvent::Clear),
            &mut state, &mut logs, &mut emits,
        );

        // Live bar survives, finished bar removed
        assert_eq!(state.bars.len(), 1, "live bar should survive clear");
        assert!(state.bars.contains_key(&live_id), "live bar should still exist");
        assert!(!state.bars.contains_key(&done_id), "finished bar should be removed");
        assert!(state.resource_status.is_empty(), "resource status should be cleared");
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
            rate_anchor: None,
            unit: "rec".into(),
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

        // Row 1: chart area starts (no separate status line)
        // Charts should have content in multiple rows
        let chart_rows_with_content: Vec<u16> = (1..height)
            .filter(|&r| row_has_content(&buf, r))
            .collect();
        assert!(
            chart_rows_with_content.len() >= 3,
            "Chart should have content in at least 3 rows (got {} rows with content: {:?})",
            chart_rows_with_content.len(),
            chart_rows_with_content,
        );

        // The first chart row should have the title/border
        let top_chart_row = row_text(&buf, 1);
        assert!(
            top_chart_row.contains("rss"),
            "First chart row should contain chart title, got: '{}'", top_chart_row
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

        // Row 1 should be the start of chart area (box-drawing border with title)
        let row1 = row_text(&buf, 1);
        assert!(row1.contains("rss"), "Row 1 should be chart border with rss title, got: '{}'", row1);
    }

    #[test]
    fn layout_no_chart_when_insufficient_data() {
        let mut state = RenderState::new();
        state.resource_status = "rss: 1.0 GiB | threads: 4".into();
        // Only 1 data point — not enough for chart
        state.rss_history.push(1024.0);
        state.dirty = true;

        // No chart, no extra budget/alert — visible_lines is 0
        assert_eq!(state.visible_lines(), 0, "Should be 0 lines with no bars, no chart, no extra");

        let buf = render_state_to_buffer(&state, 80, 4);

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
