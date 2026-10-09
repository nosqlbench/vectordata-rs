// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. [`TextMeter`], the line-oriented progress renderer
//! `vectordata datasets precache` draws, as a [`FetchProgress`] sink
//! any caller can attach.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{FacetId, FetchEvent, FetchProgress};

/// A shared, lockable output: the meter's ticker thread and the
/// fetching thread both write to it.
type Out = Arc<Mutex<Box<dyn Write + Send>>>;

/// A progress meter that draws a fetch as text: a status line that keeps
/// moving while facets are opened, then a single in-place line per facet
/// with its own and the run's progress, rate and time remaining, closed
/// by a `✓` line per facet and a summary.
///
/// This is the renderer `vectordata datasets precache` uses; a library
/// caller that wants the same display passes one to
/// [`TestDataView::fetch`](crate::TestDataView::fetch):
///
/// ```no_run
/// # use vectordata::fetch::{FetchRequest, TextMeter};
/// # fn demo(view: &dyn vectordata::TestDataView) -> vectordata::Result<()> {
/// view.fetch(&FetchRequest::all(), &mut TextMeter::stderr("Fetch"))?;
/// # Ok(()) }
/// ```
///
/// Lines are overwritten in place with carriage returns, so the output
/// is meant for a terminal; for a log, use [`LogMeter`]. `title` heads the summary line
/// (`"<title> done: …"`, `"<title>: failed — …"`).
pub struct TextMeter {
    title: String,
    out: Out,
    ticker: Option<Ticker>,
    live: Option<Live>,
}

impl TextMeter {
    /// A meter that draws on standard error.
    pub fn stderr(title: &str) -> Self {
        Self::to_writer(title, Box::new(std::io::stderr()))
    }

    /// A meter that draws on `out` — a file, a buffer, a pipe.
    pub fn to_writer(title: &str, out: Box<dyn Write + Send>) -> Self {
        TextMeter {
            title: title.to_string(),
            out: Arc::new(Mutex::new(out)),
            ticker: None,
            live: None,
        }
    }

    fn write(&self, text: &str) {
        let mut out = self.out.lock().expect("meter output");
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }

    fn stop_ticker(&mut self) {
        if let Some(t) = self.ticker.take() {
            t.stop();
        }
    }
}

impl FetchProgress for TextMeter {
    fn on_event(&mut self, event: &FetchEvent<'_>) {
        match event {
            FetchEvent::PlanBegin { index, count, facet, files } => {
                let span = match files {
                    Some(n) if *n > 1 => format!(" ({n} files)"),
                    _ => String::new(),
                };
                let out = self.out.clone();
                self.ticker
                    .get_or_insert_with(|| Ticker::start(out))
                    .begin(format!("[{index}/{count}] {facet}{span}: opening"));
            }
            FetchEvent::PlanEnd { facet, plan } => {
                let outcome = if plan.degrades_to_full_download {
                    format!("{} whole, its window cannot be mapped", fmt_bytes(plan.facet_bytes))
                } else if plan.is_resident() {
                    "already resident".to_string()
                } else {
                    format!("{} to fetch", fmt_bytes(plan.bytes_to_fetch()))
                };
                if let Some(t) = &self.ticker {
                    t.end(format!("{facet}: {outcome}"));
                }
            }
            FetchEvent::Begin { facets, bytes } => {
                self.stop_ticker();
                self.live = Some(Live::new(*facets, *bytes));
            }
            FetchEvent::FacetBegin { index, facet, bytes, resident, .. } => {
                if let Some(live) = &mut self.live {
                    live.begin(*index, facet, *bytes, *resident);
                    let line = live.line(0);
                    self.write(&line);
                }
            }
            FetchEvent::Progress { bytes, .. } => {
                let Some(live) = &mut self.live else { return };
                live.current_bytes = *bytes;
                if live.last_render.elapsed() >= Duration::from_millis(250) {
                    live.last_render = Instant::now();
                    let line = live.line(*bytes);
                    self.write(&line);
                }
            }
            FetchEvent::FacetEnd { bytes, .. } => {
                if let Some(live) = &mut self.live {
                    let line = live.end(*bytes);
                    self.write(&line);
                }
            }
            FetchEvent::End { report } => {
                if let Some(live) = &self.live {
                    let secs = live.started.elapsed().as_secs_f64();
                    let done = report.bytes_fetched();
                    let line = format!(
                        "{} done: {} facet(s), {} in {:.1}s ({}/s).\n",
                        self.title,
                        live.count,
                        fmt_bytes(done),
                        secs,
                        fmt_bytes((done as f64 / secs.max(0.001)) as u64)
                    );
                    self.write(&line);
                }
                self.live = None;
            }
            FetchEvent::Failed { error, .. } => {
                self.stop_ticker();
                // A failure while planning is the caller's to report —
                // it knows what it asked for. One mid-fetch also has a
                // half-drawn line to close.
                if let Some(live) = self.live.take() {
                    let line = format!("\r{}\u{1b}[K\n{}: failed — {error}\n", live.clear(), self.title);
                    self.write(&line);
                }
            }
        }
    }
}

impl Drop for TextMeter {
    fn drop(&mut self) {
        self.stop_ticker();
    }
}

// ─── Planning ticker ─────────────────────────────────────────────────

/// A status line that keeps moving while the planner opens files.
///
/// Planning opens every facet: on a large sharded dataset that can be a
/// merkle reference per shard and a slab index, seconds during which
/// nothing would otherwise be printed. The ticker repaints the current
/// step with its elapsed time four times a second, and closes each step
/// with its result on its own line, so the screen always says what is
/// happening and how long it has been happening.
struct Ticker {
    state: Arc<Mutex<TickState>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

struct TickState {
    out: Out,
    /// The step being worked on and when it began; `None` between steps.
    current: Option<(String, Instant)>,
    stop: bool,
}

impl TickState {
    fn write(&self, text: &str) {
        let mut out = self.out.lock().expect("meter output");
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }
}

impl Ticker {
    fn start(out: Out) -> Self {
        let state = Arc::new(Mutex::new(TickState {
            out,
            current: None,
            stop: false,
        }));
        let shared = state.clone();
        let handle = std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(250));
                // Painting under the lock keeps a repaint from landing
                // after the step's closing line.
                let st = shared.lock().expect("ticker state");
                if st.stop {
                    break;
                }
                if let Some((label, since)) = &st.current {
                    st.write(&format!(
                        "\r  {label}\u{2026} {:.1}s\u{1b}[K",
                        since.elapsed().as_secs_f64()
                    ));
                }
            }
        });
        Ticker {
            state,
            handle: Some(handle),
        }
    }

    /// Begin a step: paint its label now, then keep painting its age.
    fn begin(&self, label: String) {
        let mut st = self.state.lock().expect("ticker state");
        st.write(&format!("\r  {label}\u{2026}\u{1b}[K"));
        st.current = Some((label, Instant::now()));
    }

    /// Close the current step with its result, on its own line.
    fn end(&self, line: String) {
        let mut st = self.state.lock().expect("ticker state");
        st.current = None;
        st.write(&format!("\r  {line}\u{1b}[K\n"));
    }

    fn stop(mut self) {
        {
            let mut st = self.state.lock().expect("ticker state");
            st.stop = true;
            st.current = None;
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        if let Ok(mut st) = self.state.lock() {
            st.stop = true;
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ─── Fetch meter ─────────────────────────────────────────────────────

/// The in-place line drawn while facets are fetched.
struct Live {
    count: usize,
    total: u64,
    /// Bytes fetched by facets already finished.
    finished: u64,
    index: usize,
    facet: String,
    facet_total: u64,
    resident: bool,
    current_bytes: u64,
    last_render: Instant,
    started: Instant,
}

impl Live {
    fn new(count: usize, total: u64) -> Self {
        Live {
            count,
            total,
            finished: 0,
            index: 0,
            facet: String::new(),
            facet_total: 0,
            resident: false,
            current_bytes: 0,
            last_render: Instant::now(),
            started: Instant::now(),
        }
    }

    fn begin(&mut self, index: usize, facet: &FacetId, bytes: u64, resident: bool) {
        self.index = index;
        self.facet = facet.to_string();
        self.facet_total = bytes;
        self.resident = resident;
        self.current_bytes = 0;
        self.last_render = Instant::now();
    }

    /// The in-place line for the current facet at `bytes` done.
    fn line(&self, bytes: u64) -> String {
        let aggregate = self.finished + bytes;
        let trailing = match self.rate_eta(aggregate) {
            Some((rate, eta)) => format!(" \u{2022} {}/s \u{2022} ETA {}", fmt_bytes(rate), fmt_duration(eta)),
            None => String::new(),
        };
        format!(
            "\r  [{}/{}] {}: {} \u{2022} total {}% ({}/{}){}\u{1b}[K",
            self.index,
            self.count,
            self.facet,
            self.facet_state(bytes),
            pct(aggregate, self.total),
            fmt_bytes(aggregate),
            fmt_bytes(self.total),
            trailing
        )
    }

    /// The current facet's own progress at `bytes` done.
    fn facet_state(&self, bytes: u64) -> String {
        if self.resident {
            "already resident".to_string()
        } else {
            format!("{}% ({}/{})", pct(bytes, self.facet_total), fmt_bytes(bytes), fmt_bytes(self.facet_total))
        }
    }

    /// The run's rate in bytes per second and its time remaining in
    /// seconds, at `aggregate` bytes done.
    ///
    /// Held back until the run has been going long enough for them to
    /// mean something: the first second is connection set-up, and the
    /// rate it implies would promise an absurd wait right when the user
    /// is most likely to look.
    fn rate_eta(&self, aggregate: u64) -> Option<(u64, u64)> {
        let elapsed = self.started.elapsed().as_secs_f64();
        if elapsed > 1.5 && aggregate > 0 && self.total > aggregate {
            let rate = aggregate as f64 / elapsed;
            let eta = ((self.total - aggregate) as f64 / rate.max(1.0)) as u64;
            Some((rate as u64, eta))
        } else {
            None
        }
    }

    /// The current facet at `bytes` done, as plain text for one log
    /// line: the in-place line without its terminal controls.
    fn status(&self, bytes: u64, aggregate: u64) -> String {
        let trailing = match self.rate_eta(aggregate) {
            Some((rate, eta)) => format!(", {}/s, ETA {}", fmt_bytes(rate), fmt_duration(eta)),
            None => String::new(),
        };
        format!(
            "[{}/{}] {}: {}, total {}% ({}/{}){}",
            self.index,
            self.count,
            self.facet,
            self.facet_state(bytes),
            pct(aggregate, self.total),
            fmt_bytes(aggregate),
            fmt_bytes(self.total),
            trailing
        )
    }

    /// Close the current facet with a permanent `✓` line.
    fn end(&mut self, bytes: u64) -> String {
        // The run's total counts each facet at its planned size, so a
        // facet's share of it is the plan, whatever the transport
        // reported against chunk boundaries.
        self.finished += self.facet_total;
        let what = if self.resident {
            "already resident".to_string()
        } else {
            fmt_bytes(bytes)
        };
        format!("\r  [{}/{}] {} \u{2713} {}\u{1b}[K\n", self.index, self.count, self.facet, what)
    }

    /// Clear the in-place line, leaving the facet that was in progress.
    fn clear(&self) -> String {
        format!("  [{}/{}] {}", self.index, self.count, self.facet)
    }
}

// ─── Log meter ───────────────────────────────────────────────────────

/// Millionths: the resolution of a [`ProgressStep`].
const PPM: u64 = 1_000_000;

/// How far a run's total must advance between two progress lines of a
/// [`LogMeter`]: a step of `10%` writes at 10%, 20%, … 100%.
///
/// Parsed from a fraction (`0.1`) or a percentage (`10%`); either must
/// be more than zero and at most the whole run. A bare number is always
/// a fraction, so `1` is the whole run, not one percent.
///
/// ```
/// use vectordata::fetch::ProgressStep;
///
/// let step: ProgressStep = "10%".parse().unwrap();
/// assert_eq!(step, "0.1".parse().unwrap());
/// assert_eq!(step.to_string(), "10%");
/// assert!("50".parse::<ProgressStep>().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressStep {
    /// The step, in millionths of the run.
    ppm: u32,
}

impl ProgressStep {
    /// A step of `fraction` of the run, which must be in `(0, 1]`.
    pub fn fraction(fraction: f64) -> Result<Self, String> {
        let ppm = (fraction * PPM as f64).round();
        if !(fraction > 0.0 && fraction <= 1.0) || ppm < 1.0 {
            return Err(format!(
                "a progress step must be more than 0 and at most 1 (100%), not {fraction}"
            ));
        }
        Ok(ProgressStep { ppm: ppm as u32 })
    }

    /// The step as a fraction of the run.
    pub fn as_fraction(self) -> f64 {
        self.ppm as f64 / PPM as f64
    }

    /// Steps completed at `done` of `total` bytes; all of them when
    /// there is nothing to do.
    fn index(self, done: u64, total: u64) -> u64 {
        if total == 0 {
            return PPM / self.ppm as u64;
        }
        (done.min(total) as u128 * PPM as u128 / total as u128 / self.ppm as u128) as u64
    }
}

impl std::str::FromStr for ProgressStep {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let text = s.trim();
        let (number, scale) = match text.strip_suffix('%') {
            Some(p) => (p.trim(), 100.0),
            None => (text, 1.0),
        };
        let value: f64 = number
            .parse()
            .map_err(|_| format!("'{s}' is not a progress step; give a fraction (0.1) or a percentage (10%)"))?;
        ProgressStep::fraction(value / scale)
            .map_err(|_| format!("'{s}' is not a progress step: it must be more than 0 and at most 1 (100%)"))
    }
}

/// As a percentage: `10%`, `0.5%`.
impl std::fmt::Display for ProgressStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}%", self.ppm as f64 / 10_000.0)
    }
}

/// Where a [`LogMeter`] sends each finished line, with its level.
type LineSink = Box<dyn FnMut(log::Level, &str) + Send>;

/// A progress meter that reports a fetch as whole lines, for logs, CI
/// output and anything else that is not a terminal.
///
/// Every line is complete and stands alone: no carriage returns, no
/// terminal controls, ASCII punctuation, each starting with the title.
/// It writes:
///
/// - a line as each facet is planned, and its cost once it is;
/// - a line as each facet starts and finishes;
/// - **a progress line each time the run's total crosses a step** —
///   10% by default, set with [`step`](Self::step) — but never sooner
///   than [`min_interval`](Self::min_interval) after the last one
///   (2 s by default), which only holds back a burst;
/// - a progress line at 100% when the run ends, always;
/// - **a warning when nothing has moved for a minute** — set with
///   [`stall_warning`](Self::stall_warning) — repeated each minute the
///   stall lasts, in the same form as a progress line;
/// - the summary line [`TextMeter`] ends with.
///
/// ```text
/// Fetch: fetching 2 facet(s), 1.5 GiB
/// Fetch: [1/2] base_vectors: started, 1.0 GiB to fetch
/// Fetch: [1/2] base_vectors: 10% (102.4 MiB/1.0 GiB), total 6% (102.4 MiB/1.5 GiB), 51.2 MiB/s, ETA 28s
/// Fetch: warning: no progress for 1m 00s, [1/2] base_vectors: 14% (143.4 MiB/1.0 GiB), total 9% (…)
/// …
/// Fetch: [2/2] query_vectors: 100% (512.0 MiB/512.0 MiB), total 100% (1.5 GiB/1.5 GiB)
/// Fetch done: 2 facet(s), 1.5 GiB in 31.4s (48.9 MiB/s).
/// ```
///
/// ```no_run
/// # use std::time::Duration;
/// # use vectordata::fetch::{FetchRequest, LogMeter};
/// # fn demo(view: &dyn vectordata::TestDataView) -> vectordata::Result<()> {
/// let mut meter = LogMeter::to_log("Fetch")
///     .step(Some("5%".parse().unwrap()))
///     .min_interval(Duration::from_secs(10));
/// view.fetch(&FetchRequest::all(), &mut meter)?;
/// # Ok(()) }
/// ```
///
/// The stall warning is timed by a thread the meter starts with its
/// first event and stops when it is dropped.
pub struct LogMeter {
    shared: Arc<LogShared>,
    watchdog: Option<std::thread::JoinHandle<()>>,
}

struct LogShared {
    state: Mutex<LogState>,
    wake: std::sync::Condvar,
}

impl LogMeter {
    /// A meter that writes its lines to standard error.
    pub fn stderr(title: &str) -> Self {
        Self::to_writer(title, Box::new(std::io::stderr()))
    }

    /// A meter that writes its lines to `out` — a file, a buffer, a
    /// pipe — one line per write.
    pub fn to_writer(title: &str, mut out: Box<dyn Write + Send>) -> Self {
        Self::to_lines(title, move |_, line| {
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        })
    }

    /// A meter that logs its lines through the [`log`] crate, under the
    /// target `vectordata::fetch`: progress at `Info`, a stall at
    /// `Warn`, a failure at `Error`.
    pub fn to_log(title: &str) -> Self {
        Self::to_lines(title, |level, line| log::log!(target: "vectordata::fetch", level, "{line}"))
    }

    /// A meter that hands each line, without a line ending, to `sink`
    /// with its level: progress at `Info`, a stall at `Warn`, a failure
    /// at `Error`.
    pub fn to_lines(title: &str, sink: impl FnMut(log::Level, &str) + Send + 'static) -> Self {
        LogMeter {
            shared: Arc::new(LogShared {
                state: Mutex::new(LogState {
                    title: title.to_string(),
                    sink: Box::new(sink),
                    step: Some(ProgressStep { ppm: (PPM / 10) as u32 }),
                    min_interval: Duration::from_secs(2),
                    stall_after: Duration::from_secs(60),
                    phase: LogPhase::Idle,
                    advanced: Instant::now(),
                    warned: None,
                    last_progress: None,
                    reported_step: 0,
                    reported_bytes: None,
                    stop: false,
                }),
                wake: std::sync::Condvar::new(),
            }),
            watchdog: None,
        }
    }

    /// Write a progress line each time the run's total crosses a
    /// multiple of `step` (10% by default); `None` writes one for every
    /// advance, as often as [`min_interval`](Self::min_interval) allows.
    pub fn step(self, step: Option<ProgressStep>) -> Self {
        self.lock().step = step;
        self
    }

    /// The least time between two progress lines (2 s by default): a
    /// step crossed sooner is reported by the first progress after it.
    /// Zero lets every step through.
    pub fn min_interval(self, interval: Duration) -> Self {
        self.lock().min_interval = interval;
        self
    }

    /// Warn when nothing has moved for `after` (a minute by default),
    /// and again each `after` the stall lasts.
    pub fn stall_warning(self, after: Duration) -> Self {
        self.lock().stall_after = after;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LogState> {
        self.shared.state.lock().expect("meter state")
    }

    /// Start the stall watchdog, once.
    fn ensure_watchdog(&mut self) {
        if self.watchdog.is_some() {
            return;
        }
        let shared = self.shared.clone();
        self.watchdog = std::thread::Builder::new()
            .name("vectordata-fetch-watchdog".into())
            .spawn(move || watch(&shared))
            .ok();
    }
}

impl FetchProgress for LogMeter {
    fn on_event(&mut self, event: &FetchEvent<'_>) {
        self.ensure_watchdog();
        self.lock().on_event(event);
        self.shared.wake.notify_all();
    }
}

impl Drop for LogMeter {
    fn drop(&mut self) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.stop = true;
        }
        self.shared.wake.notify_all();
        if let Some(h) = self.watchdog.take() {
            let _ = h.join();
        }
    }
}

/// The watchdog: sleep until the run would have stalled, and warn if it
/// has.
fn watch(shared: &LogShared) {
    let mut st = shared.state.lock().expect("meter state");
    loop {
        if st.stop {
            return;
        }
        let wait = match st.stall_deadline() {
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    st.warn_stalled(now);
                    continue;
                }
                deadline - now
            }
            None => st.stall_after,
        };
        st = shared.wake.wait_timeout(st, wait).expect("meter state").0;
    }
}

/// What a [`LogMeter`] is in the middle of.
enum LogPhase {
    Idle,
    /// Planning the facet with this label.
    Planning(String),
    Fetching(Live),
}

struct LogState {
    title: String,
    sink: LineSink,
    step: Option<ProgressStep>,
    min_interval: Duration,
    stall_after: Duration,
    phase: LogPhase,
    /// When the run last moved.
    advanced: Instant,
    /// When the last stall warning was written, if one was since the
    /// run last moved.
    warned: Option<Instant>,
    /// When the last progress line was written.
    last_progress: Option<Instant>,
    /// Steps reported so far.
    reported_step: u64,
    /// The run's total bytes the last progress line showed.
    reported_bytes: Option<u64>,
    stop: bool,
}

impl LogState {
    fn emit(&mut self, level: log::Level, body: &str) {
        let line = format!("{}: {body}", self.title);
        (self.sink)(level, &line);
    }

    fn advance(&mut self) {
        self.advanced = Instant::now();
        self.warned = None;
    }

    /// When the run counts as stalled, if it is doing anything.
    fn stall_deadline(&self) -> Option<Instant> {
        match self.phase {
            LogPhase::Idle => None,
            _ => Some(self.warned.unwrap_or(self.advanced) + self.stall_after),
        }
    }

    fn warn_stalled(&mut self, now: Instant) {
        let idle = fmt_duration(now.duration_since(self.advanced).as_secs());
        let what = match &self.phase {
            LogPhase::Idle => return,
            LogPhase::Planning(label) => format!("still planning {label}"),
            LogPhase::Fetching(live) => live.status(live.current_bytes, live.finished + live.current_bytes),
        };
        self.emit(log::Level::Warn, &format!("warning: no progress for {idle}, {what}"));
        self.warned = Some(now);
    }

    /// Write a progress line at `aggregate` bytes of the run, and note
    /// it as reported.
    fn report(&mut self, facet_bytes: u64, aggregate: u64, step: u64) {
        let LogPhase::Fetching(live) = &self.phase else { return };
        let body = live.status(facet_bytes, aggregate);
        self.emit(log::Level::Info, &body);
        self.reported_step = step;
        self.reported_bytes = Some(aggregate);
        self.last_progress = Some(Instant::now());
    }

    fn on_event(&mut self, event: &FetchEvent<'_>) {
        match event {
            FetchEvent::PlanBegin { index, count, facet, files } => {
                let span = match files {
                    Some(n) if *n > 1 => format!(" ({n} files)"),
                    _ => String::new(),
                };
                let label = format!("[{index}/{count}] {facet}{span}");
                self.emit(log::Level::Info, &format!("planning {label}"));
                self.phase = LogPhase::Planning(label);
                self.advance();
            }
            FetchEvent::PlanEnd { facet, plan } => {
                let outcome = if plan.degrades_to_full_download {
                    format!("{} whole, its window cannot be mapped", fmt_bytes(plan.facet_bytes))
                } else if plan.is_resident() {
                    "already resident".to_string()
                } else {
                    format!("{} to fetch", fmt_bytes(plan.bytes_to_fetch()))
                };
                let label = match std::mem::replace(&mut self.phase, LogPhase::Idle) {
                    LogPhase::Planning(label) => label,
                    _ => facet.to_string(),
                };
                self.emit(log::Level::Info, &format!("planned {label}: {outcome}"));
                self.advance();
            }
            FetchEvent::Begin { facets, bytes } => {
                self.emit(log::Level::Info, &format!("fetching {facets} facet(s), {}", fmt_bytes(*bytes)));
                self.phase = LogPhase::Fetching(Live::new(*facets, *bytes));
                self.reported_step = 0;
                self.reported_bytes = None;
                self.last_progress = None;
                self.advance();
            }
            FetchEvent::FacetBegin { index, facet, bytes, resident, .. } => {
                let LogPhase::Fetching(live) = &mut self.phase else { return };
                live.begin(*index, facet, *bytes, *resident);
                if !*resident {
                    let body = format!("[{index}/{}] {facet}: started, {} to fetch", live.count, fmt_bytes(*bytes));
                    self.emit(log::Level::Info, &body);
                }
                self.advance();
            }
            FetchEvent::Progress { bytes, .. } => {
                let LogPhase::Fetching(live) = &mut self.phase else { return };
                let moved = *bytes > live.current_bytes;
                live.current_bytes = *bytes;
                let (aggregate, total) = (live.finished + bytes, live.total);
                if moved {
                    self.advance();
                }
                let due = progress_due(
                    self.step,
                    self.reported_step,
                    aggregate,
                    total,
                    self.last_progress.map(|t| t.elapsed()),
                    self.min_interval,
                );
                if let Some(step) = due {
                    self.report(*bytes, aggregate, step);
                }
            }
            FetchEvent::FacetEnd { bytes, .. } => {
                let LogPhase::Fetching(live) = &mut self.phase else { return };
                // As in the terminal meter, the run counts each facet at
                // its planned size.
                live.finished += live.facet_total;
                live.current_bytes = 0;
                let what = if live.resident {
                    "already resident".to_string()
                } else {
                    format!("done, {}", fmt_bytes(*bytes))
                };
                let body = format!("[{}/{}] {}: {what}", live.index, live.count, live.facet);
                self.emit(log::Level::Info, &body);
                self.advance();
            }
            FetchEvent::End { report } => {
                let LogPhase::Fetching(live) = &self.phase else { return };
                let (total, facet_total, count, secs) =
                    (live.total, live.facet_total, live.count, live.started.elapsed().as_secs_f64());
                if self.reported_bytes != Some(total) {
                    self.report(facet_total, total, self.reported_step);
                }
                let done = report.bytes_fetched();
                let line = format!(
                    "{} done: {} facet(s), {} in {:.1}s ({}/s).",
                    self.title,
                    count,
                    fmt_bytes(done),
                    secs,
                    fmt_bytes((done as f64 / secs.max(0.001)) as u64)
                );
                (self.sink)(log::Level::Info, &line);
                self.phase = LogPhase::Idle;
            }
            FetchEvent::Failed { error, .. } => {
                // As in the terminal meter, a failure while planning is
                // the caller's to report.
                if let LogPhase::Fetching(live) = std::mem::replace(&mut self.phase, LogPhase::Idle) {
                    let body = format!("failed at [{}/{}] {}: {error}", live.index, live.count, live.facet);
                    self.emit(log::Level::Error, &body);
                }
            }
        }
    }
}

/// Whether a progress line is due at `done` of `total` bytes, and the
/// step it reports.
///
/// Due when the run has crossed a step past the `reported` one (or, with
/// no step, at every advance) and the last progress line is at least
/// `min_interval` old. A step crossed too soon is not lost: the next
/// progress after the interval reports it.
fn progress_due(
    step: Option<ProgressStep>,
    reported: u64,
    done: u64,
    total: u64,
    since_last: Option<Duration>,
    min_interval: Duration,
) -> Option<u64> {
    let reached = step.map_or(reported, |s| s.index(done, total));
    let crossed = step.is_none() || reached > reported;
    let rested = since_last.is_none_or(|d| d >= min_interval);
    (crossed && rested).then_some(reached)
}

/// The meter a CLI command draws on standard error: [`TextMeter`] on a
/// terminal, [`LogMeter`] when standard error is a file or a pipe,
/// where the in-place line would be a wall of carriage returns.
pub(crate) fn stderr_meter(title: &str) -> Box<dyn FetchProgress> {
    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        Box::new(TextMeter::stderr(title))
    } else {
        Box::new(LogMeter::stderr(title))
    }
}

// ─── Formatting ──────────────────────────────────────────────────────

/// `done` as a whole percentage of `total`; 100 when there is nothing
/// to do.
pub(crate) fn pct(done: u64, total: u64) -> u32 {
    if total == 0 {
        return 100;
    }
    ((done.min(total) as u128 * 100) / total as u128) as u32
}

/// Binary-unit byte count: `512 B`, `1.5 KiB` … `2.0 TiB`.
pub(crate) fn fmt_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    const TIB: u64 = 1024 * GIB;
    if bytes >= TIB {
        format!("{:.1} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// A duration in seconds as the largest unit pair: `45s`, `3m 22s`,
/// `1h 12m`, `2d 04h`. Two units keep the resolution useful at the
/// boundaries, so a 60-minute estimate does not read as `1h 00m` next to
/// a 59-second one with no seconds shown.
pub(crate) fn fmt_duration(secs: u64) -> String {
    const M: u64 = 60;
    const H: u64 = 60 * M;
    const D: u64 = 24 * H;
    if secs < M {
        format!("{secs}s")
    } else if secs < H {
        format!("{}m {:02}s", secs / M, secs % M)
    } else if secs < D {
        format!("{}h {:02}m", secs / H, (secs % H) / M)
    } else {
        format!("{}d {:02}h", secs / D, (secs % D) / H)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::{FacetFetch, FetchReport};
    use crate::view::PrefetchPlan;

    /// A writer the test can read back.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn id(name: &str) -> FacetId {
        FacetId::new(None, name)
    }

    /// **The run's total spans every facet.** Fetching a subset used to
    /// build one meter per facet, so the "total" each showed was that
    /// facet alone and the run had no total at all.
    #[test]
    fn the_total_spans_every_facet_in_the_run() {
        let cap = Capture::default();
        let mut meter = TextMeter::to_writer("Fetch", Box::new(cap.clone()));
        let (a, b) = (id("a"), id("b"));
        meter.on_event(&FetchEvent::Begin { facets: 2, bytes: 3000 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 2, facet: &a, bytes: 1000, chunks: 1, resident: false });
        meter.on_event(&FetchEvent::FacetEnd { facet: &a, bytes: 1000 });
        meter.on_event(&FetchEvent::FacetBegin { index: 2, count: 2, facet: &b, bytes: 2000, chunks: 1, resident: false });
        let text = cap.text();
        assert!(text.contains("[1/2] a ✓ 1000 B"), "{text}");
        assert!(
            text.contains("[2/2] b: 0% (0 B/2.0 KiB) • total 33% (1000 B/2.9 KiB)"),
            "the second facet opens at the first facet's share of the run's total: {text}"
        );
    }

    /// Planning keeps the screen alive and closes each facet with its
    /// cost; the fetch closes with a summary naming the title.
    #[test]
    fn planning_and_completion_are_both_reported() {
        let cap = Capture::default();
        let mut meter = TextMeter::to_writer("Precache", Box::new(cap.clone()));
        let a = id("a");
        let plan = PrefetchPlan::default();
        meter.on_event(&FetchEvent::PlanBegin { index: 1, count: 1, facet: &a, files: Some(3) });
        meter.on_event(&FetchEvent::PlanEnd { facet: &a, plan: &plan });
        meter.on_event(&FetchEvent::Begin { facets: 1, bytes: 0 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 1, facet: &a, bytes: 0, chunks: 0, resident: true });
        meter.on_event(&FetchEvent::FacetEnd { facet: &a, bytes: 0 });
        let report = FetchReport {
            facets: vec![FacetFetch {
                id: a.clone(),
                planned: plan.clone(),
                ranges_fetched: 0,
                bytes_fetched: 0,
                complete: true,
                upstream_checked: true,
            }],
            elapsed: Duration::ZERO,
        };
        meter.on_event(&FetchEvent::End { report: &report });
        let text = cap.text();
        assert!(text.contains("[1/1] a (3 files): opening"), "{text}");
        assert!(text.contains("a: already resident"), "{text}");
        assert!(text.contains("[1/1] a ✓ already resident"), "{text}");
        assert!(text.contains("Precache done: 1 facet(s), 0 B"), "{text}");
    }

    /// A failure mid-fetch closes the line and names the title.
    #[test]
    fn a_failure_mid_fetch_is_reported_once() {
        let cap = Capture::default();
        let mut meter = TextMeter::to_writer("Precache", Box::new(cap.clone()));
        let a = id("a");
        meter.on_event(&FetchEvent::Begin { facets: 1, bytes: 10 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 1, facet: &a, bytes: 10, chunks: 1, resident: false });
        let error = crate::Error::Other("boom".into());
        meter.on_event(&FetchEvent::Failed { facet: Some(&a), error: &error });
        let text = cap.text();
        assert_eq!(text.matches("Precache: failed — boom").count(), 1, "{text}");
    }

    /// The lines a [`LogMeter`] wrote, with their levels.
    type Lines = Arc<Mutex<Vec<(log::Level, String)>>>;

    fn log_meter(title: &str) -> (LogMeter, Lines) {
        let lines: Lines = Arc::default();
        let sink = lines.clone();
        let meter = LogMeter::to_lines(title, move |level, line| sink.lock().unwrap().push((level, line.to_string())));
        (meter, lines)
    }

    fn texts(lines: &Lines) -> Vec<String> {
        lines.lock().unwrap().iter().map(|(_, l)| l.clone()).collect()
    }

    fn progress(meter: &mut LogMeter, facet: &FacetId, bytes: u64, total: u64) {
        meter.on_event(&FetchEvent::Progress { facet, bytes, total, chunks: 0 });
    }

    /// **Log lines are whole and plain.** No carriage returns, no
    /// terminal controls, ASCII only, each headed by the title; and a
    /// progress line is written once per step crossed, not per event.
    #[test]
    fn log_lines_are_whole_plain_and_stepped() {
        let (meter, lines) = log_meter("Fetch");
        let mut meter = meter.min_interval(Duration::ZERO);
        let (a, b) = (id("a"), id("b"));
        let plan = PrefetchPlan::default();
        meter.on_event(&FetchEvent::PlanBegin { index: 1, count: 2, facet: &a, files: Some(3) });
        meter.on_event(&FetchEvent::PlanEnd { facet: &a, plan: &plan });
        meter.on_event(&FetchEvent::Begin { facets: 2, bytes: 1000 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 2, facet: &a, bytes: 600, chunks: 6, resident: false });
        for bytes in (0..=600).step_by(25) {
            progress(&mut meter, &a, bytes, 600);
        }
        meter.on_event(&FetchEvent::FacetEnd { facet: &a, bytes: 600 });
        meter.on_event(&FetchEvent::FacetBegin { index: 2, count: 2, facet: &b, bytes: 400, chunks: 4, resident: false });
        for bytes in (0..=400).step_by(25) {
            progress(&mut meter, &b, bytes, 400);
        }
        meter.on_event(&FetchEvent::FacetEnd { facet: &b, bytes: 400 });

        let text = texts(&lines);
        for line in &text {
            assert!(line.is_ascii() && !line.contains('\r') && !line.contains('\n'), "{line:?}");
            assert!(line.starts_with("Fetch: "), "{line:?}");
        }
        assert!(text.contains(&"Fetch: planning [1/2] a (3 files)".to_string()), "{text:#?}");
        assert!(text.contains(&"Fetch: planned [1/2] a (3 files): already resident".to_string()), "{text:#?}");
        assert!(text.contains(&"Fetch: [1/2] a: started, 600 B to fetch".to_string()), "{text:#?}");
        assert!(text.contains(&"Fetch: [1/2] a: done, 600 B".to_string()), "{text:#?}");
        let stepped: Vec<&String> = text.iter().filter(|l| l.contains(", total ")).collect();
        let totals: Vec<String> = stepped
            .iter()
            .map(|l| l.split(", total ").nth(1).unwrap().split('%').next().unwrap().to_string())
            .collect();
        assert_eq!(totals, ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"], "{stepped:#?}");
        assert_eq!(*stepped[0], "Fetch: [1/2] a: 16% (100 B/600 B), total 10% (100 B/1000 B)");
    }

    /// **The interval only holds back a burst.** Steps crossed inside
    /// the interval are not written one by one; the run still ends with
    /// its 100% line and the summary.
    #[test]
    fn the_interval_throttles_and_the_run_ends_at_100_percent() {
        let (meter, lines) = log_meter("Precache");
        let mut meter = meter.min_interval(Duration::from_secs(3600));
        let a = id("a");
        meter.on_event(&FetchEvent::Begin { facets: 1, bytes: 1000 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 1, facet: &a, bytes: 1000, chunks: 10, resident: false });
        for bytes in (0..=950).step_by(50) {
            progress(&mut meter, &a, bytes, 1000);
        }
        meter.on_event(&FetchEvent::FacetEnd { facet: &a, bytes: 1000 });
        let report = FetchReport { facets: vec![], elapsed: Duration::ZERO };
        meter.on_event(&FetchEvent::End { report: &report });

        let text = texts(&lines);
        let stepped: Vec<&String> = text.iter().filter(|l| l.contains(", total ")).collect();
        assert_eq!(stepped.len(), 2, "the first step, then nothing until the end: {stepped:#?}");
        assert!(stepped[0].contains("total 10%"), "{stepped:#?}");
        assert_eq!(*stepped[1], "Precache: [1/1] a: 100% (1000 B/1000 B), total 100% (1000 B/1000 B)");
        assert!(text.last().unwrap().starts_with("Precache done: 1 facet(s)"), "{text:#?}");
    }

    /// The closing 100% line is not repeated when the last step already
    /// said 100%, and is written even for a run with nothing to fetch.
    #[test]
    fn the_100_percent_line_is_written_once() {
        let a = id("a");
        let report = FetchReport { facets: vec![], elapsed: Duration::ZERO };

        let (meter, lines) = log_meter("Fetch");
        let mut meter = meter.min_interval(Duration::ZERO);
        meter.on_event(&FetchEvent::Begin { facets: 1, bytes: 100 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 1, facet: &a, bytes: 100, chunks: 1, resident: false });
        progress(&mut meter, &a, 100, 100);
        meter.on_event(&FetchEvent::FacetEnd { facet: &a, bytes: 100 });
        meter.on_event(&FetchEvent::End { report: &report });
        assert_eq!(texts(&lines).iter().filter(|l| l.contains("total 100%")).count(), 1, "{:#?}", texts(&lines));

        let (mut meter, lines) = log_meter("Fetch");
        meter.on_event(&FetchEvent::Begin { facets: 1, bytes: 0 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 1, facet: &a, bytes: 0, chunks: 0, resident: true });
        meter.on_event(&FetchEvent::FacetEnd { facet: &a, bytes: 0 });
        meter.on_event(&FetchEvent::End { report: &report });
        let text = texts(&lines);
        assert!(text.contains(&"Fetch: [1/1] a: already resident".to_string()), "{text:#?}");
        assert!(text.contains(&"Fetch: [1/1] a: already resident, total 100% (0 B/0 B)".to_string()), "{text:#?}");
    }

    /// **A stall is warned in the progress line's form**, at `Warn`,
    /// and repeated while it lasts; an idle meter never warns.
    #[test]
    fn a_stall_is_warned_in_the_progress_form() {
        let (meter, lines) = log_meter("Fetch");
        let mut meter = meter.stall_warning(Duration::from_millis(40));
        let a = id("a");
        meter.on_event(&FetchEvent::Begin { facets: 1, bytes: 1000 });
        meter.on_event(&FetchEvent::FacetBegin { index: 1, count: 1, facet: &a, bytes: 1000, chunks: 10, resident: false });
        progress(&mut meter, &a, 400, 1000);
        let warnings = || -> Vec<String> {
            lines.lock().unwrap().iter().filter(|(l, _)| *l == log::Level::Warn).map(|(_, t)| t.clone()).collect()
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        while warnings().len() < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let seen = warnings();
        assert!(seen.len() >= 2, "a stall is warned again while it lasts: {seen:#?}");
        assert!(
            seen[0].starts_with("Fetch: warning: no progress for 0s, [1/1] a: 40% (400 B/1000 B), total 40%"),
            "{seen:#?}"
        );

        meter.on_event(&FetchEvent::FacetEnd { facet: &a, bytes: 1000 });
        let report = FetchReport { facets: vec![], elapsed: Duration::ZERO };
        meter.on_event(&FetchEvent::End { report: &report });
        let after_end = warnings().len();
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(warnings().len(), after_end, "a finished run does not stall");
    }

    /// A slow planning step is warned about too.
    #[test]
    fn a_stalled_plan_is_warned() {
        let (meter, lines) = log_meter("Fetch");
        let mut meter = meter.stall_warning(Duration::from_millis(40));
        let a = id("a");
        meter.on_event(&FetchEvent::PlanBegin { index: 1, count: 1, facet: &a, files: Some(12) });
        let deadline = Instant::now() + Duration::from_secs(10);
        let found = loop {
            let text = texts(&lines);
            if let Some(l) = text.iter().find(|l| l.contains("warning")) {
                break l.clone();
            }
            assert!(Instant::now() < deadline, "no stall warning: {text:#?}");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(found, "Fetch: warning: no progress for 0s, still planning [1/1] a (12 files)");
    }

    #[test]
    fn a_progress_line_is_due_on_a_new_step_after_the_interval() {
        let ten: ProgressStep = "10%".parse().unwrap();
        let two = Duration::from_secs(2);
        // First step crossed, nothing written yet.
        assert_eq!(progress_due(Some(ten), 0, 100, 1000, None, two), Some(1));
        // Same step again: not due.
        assert_eq!(progress_due(Some(ten), 1, 150, 1000, Some(Duration::from_secs(9)), two), None);
        // Next step, but too soon: held back, not lost.
        assert_eq!(progress_due(Some(ten), 1, 250, 1000, Some(Duration::from_secs(1)), two), None);
        assert_eq!(progress_due(Some(ten), 1, 260, 1000, Some(two), two), Some(2));
        // Without a step, the interval alone decides.
        assert_eq!(progress_due(None, 0, 1, 1000, Some(two), two), Some(0));
        assert_eq!(progress_due(None, 0, 1, 1000, Some(Duration::from_secs(1)), two), None);
    }

    #[test]
    fn a_progress_step_is_a_fraction_or_a_percentage() {
        let parse = |s: &str| s.parse::<ProgressStep>();
        assert_eq!(parse("10%").unwrap(), parse("0.1").unwrap());
        assert_eq!(parse(" 2.5 % ").unwrap().as_fraction(), 0.025);
        assert_eq!(parse("1").unwrap().to_string(), "100%");
        assert_eq!(parse("0.5%").unwrap().to_string(), "0.5%");
        for bad in ["0", "0%", "-0.1", "50", "150%", "ten", "", "NaN"] {
            assert!(parse(bad).is_err(), "{bad:?} parsed");
        }
        assert!(parse("50").unwrap_err().contains("at most 1 (100%)"));
    }

    #[test]
    fn formatting_is_stable() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(1536), "1.5 KiB");
        assert_eq!(fmt_duration(45), "45s");
        assert_eq!(fmt_duration(3 * 60 + 22), "3m 22s");
        assert_eq!(pct(5, 0), 100);
        assert_eq!(pct(20, 10), 100, "progress past the plan is clamped");
    }
}
