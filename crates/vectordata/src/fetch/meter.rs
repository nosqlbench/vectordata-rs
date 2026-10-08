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
/// is meant for a terminal. `title` heads the summary line
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
        let facet_state = if self.resident {
            "already resident".to_string()
        } else {
            format!(
                "{}% ({}/{})",
                pct(bytes, self.facet_total),
                fmt_bytes(bytes),
                fmt_bytes(self.facet_total)
            )
        };
        // Rate and time remaining are held back until the run has been
        // going long enough for them to mean something: the first second
        // is connection set-up, and the rate it implies would promise an
        // absurd wait right when the user is most likely to look.
        let elapsed = self.started.elapsed().as_secs_f64();
        let trailing = if elapsed > 1.5 && aggregate > 0 && self.total > aggregate {
            let rate = aggregate as f64 / elapsed;
            let eta = ((self.total - aggregate) as f64 / rate.max(1.0)) as u64;
            format!(
                " \u{2022} {}/s \u{2022} ETA {}",
                fmt_bytes(rate as u64),
                fmt_duration(eta)
            )
        } else {
            String::new()
        };
        format!(
            "\r  [{}/{}] {}: {} \u{2022} total {}% ({}/{}){}\u{1b}[K",
            self.index,
            self.count,
            self.facet,
            facet_state,
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
