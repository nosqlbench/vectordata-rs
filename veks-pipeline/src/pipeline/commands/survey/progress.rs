// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Survey-side progress and status reporting.
//!
//! Owns the live progress bar over total records, batch-level
//! milestone log lines, and the `status_line()` snapshot that an
//! external observer (`veks status` or the runner's resource
//! polling thread) can read at any time.
//!
//! Two concerns are deliberately kept separate:
//!
//! - **Progress** (`SurveyProgress`): cumulative state — pass index,
//!   records, rate, milestones. Updated from the main loop.
//! - **Status snapshot** (`SurveyStatusSource`): a cheap read-only
//!   view that summarizes current state into a single string for
//!   live display.
//!
//! The snapshot is produced on demand rather than maintained as a
//! cache; the survey is not a hot path for status polling, and
//! recomputing keeps the locking surface small.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use veks_core::ui::{ProgressHandle, UiHandle};

/// Which pass the survey is currently running.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurveyPass {
    /// Pre-loop / setup.
    Setup = 0,
    /// Pass 1 — exploration / template discovery.
    Pass1 = 1,
    /// Pass 2 — measure-suite profiling.
    Pass2 = 2,
    /// Cross-field analyzer finalize.
    CrossField = 3,
    /// Findings rendering.
    Findings = 4,
    /// Complete.
    Done = 5,
}

impl SurveyPass {
    fn label(self) -> &'static str {
        match self {
            SurveyPass::Setup => "setup",
            SurveyPass::Pass1 => "Pass 1/2 — surveying",
            SurveyPass::Pass2 => "Pass 2/2 — profiling",
            SurveyPass::CrossField => "cross-field finalize",
            SurveyPass::Findings => "rendering findings",
            SurveyPass::Done => "done",
        }
    }
}

/// Shared progress state. Atomics for the hot fields so a status
/// poller can read without locking.
pub struct SurveyProgress {
    pass: AtomicU8,
    records_processed: AtomicU64,
    records_total: AtomicU64,
    fields_classified: AtomicU64,
    unstable_count: AtomicU64,
    started: Instant,
}

impl Default for SurveyProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl SurveyProgress {
    pub fn new() -> Self {
        SurveyProgress {
            pass: AtomicU8::new(SurveyPass::Setup as u8),
            records_processed: AtomicU64::new(0),
            records_total: AtomicU64::new(0),
            fields_classified: AtomicU64::new(0),
            unstable_count: AtomicU64::new(0),
            started: Instant::now(),
        }
    }

    pub fn set_pass(&self, p: SurveyPass) {
        self.pass.store(p as u8, Ordering::Release);
    }

    pub fn current_pass(&self) -> SurveyPass {
        match self.pass.load(Ordering::Acquire) {
            0 => SurveyPass::Setup,
            1 => SurveyPass::Pass1,
            2 => SurveyPass::Pass2,
            3 => SurveyPass::CrossField,
            4 => SurveyPass::Findings,
            _ => SurveyPass::Done,
        }
    }

    pub fn set_records_total(&self, n: u64) {
        self.records_total.store(n, Ordering::Release);
    }

    pub fn records_total(&self) -> u64 {
        self.records_total.load(Ordering::Acquire)
    }

    pub fn add_records(&self, delta: u64) {
        self.records_processed.fetch_add(delta, Ordering::AcqRel);
    }

    pub fn reset_records(&self) {
        self.records_processed.store(0, Ordering::Release);
    }

    pub fn records_processed(&self) -> u64 {
        self.records_processed.load(Ordering::Acquire)
    }

    pub fn set_fields_classified(&self, n: u64) {
        self.fields_classified.store(n, Ordering::Release);
    }

    pub fn fields_classified(&self) -> u64 {
        self.fields_classified.load(Ordering::Acquire)
    }

    pub fn set_unstable_count(&self, n: u64) {
        self.unstable_count.store(n, Ordering::Release);
    }

    pub fn unstable_count(&self) -> u64 {
        self.unstable_count.load(Ordering::Acquire)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Records per second since the survey started. Returns 0 if the
    /// clock hasn't advanced (immediately after construction).
    pub fn rate(&self) -> f64 {
        let secs = self.elapsed_secs();
        if secs <= 0.0 {
            return 0.0;
        }
        self.records_processed() as f64 / secs
    }

    /// Estimated remaining seconds for the current pass, based on
    /// the current rate and the configured `records_total`. Returns
    /// 0 if the rate is 0 or the total is unknown.
    pub fn eta_secs(&self) -> f64 {
        let rate = self.rate();
        let total = self.records_total();
        let done = self.records_processed();
        if rate <= 0.0 || total == 0 || done >= total {
            return 0.0;
        }
        (total - done) as f64 / rate
    }

    /// Single-line status snapshot suitable for `veks status` or the
    /// runner's resource-status display.
    pub fn status_line(&self) -> String {
        let pass = self.current_pass();
        let done = self.records_processed();
        let total = self.records_total();
        let rate = self.rate();
        let eta = self.eta_secs();
        let fields = self.fields_classified();
        let unstable = self.unstable_count();
        let elapsed = self.elapsed_secs();
        format!(
            "[{label}]  records={done}/{total}  rate={rate:.0}/s  elapsed={elapsed:.0}s  ETA={eta:.0}s  fields={fields}  unstable={unstable}",
            label = pass.label(),
            done = done,
            total = total,
            rate = rate,
            elapsed = elapsed,
            eta = eta,
            fields = fields,
            unstable = unstable,
        )
    }
}

// ---------------------------------------------------------------------------
// Live-progress driver
// ---------------------------------------------------------------------------

/// Drives the [`UiHandle`] progress bar from a [`SurveyProgress`].
///
/// Owns the per-pass `ProgressHandle`; the orchestrator calls
/// [`tick`](Self::tick) at batch boundaries to update both the
/// shared `SurveyProgress` state and the live bar.
pub struct ProgressDriver {
    progress: Arc<SurveyProgress>,
    ui: Option<UiHandle>,
    current_bar: Option<ProgressHandle>,
    /// How often to emit milestone log lines (every Nth batch).
    log_every_n_batches: u32,
    batches_since_log: u32,
}

impl ProgressDriver {
    pub fn new(progress: Arc<SurveyProgress>, ui: Option<UiHandle>, log_every_n_batches: u32) -> Self {
        ProgressDriver {
            progress,
            ui,
            current_bar: None,
            log_every_n_batches: log_every_n_batches.max(1),
            batches_since_log: 0,
        }
    }

    /// Begin a new pass: store the pass identifier, refresh the
    /// records-total estimate, and rotate the active progress bar.
    pub fn begin_pass(&mut self, pass: SurveyPass, records_total: u64) {
        // Drop any old bar first so the UI doesn't double-display.
        self.current_bar = None;
        self.progress.set_pass(pass);
        self.progress.reset_records();
        self.progress.set_records_total(records_total);
        if let Some(ui) = &self.ui {
            self.current_bar = Some(ui.bar_with_unit(records_total, pass.label(), "records"));
        }
        self.batches_since_log = 0;
    }

    /// Tick for a completed batch of `batch_size` records.
    pub fn tick(&mut self, batch_size: u64) {
        self.progress.add_records(batch_size);
        if let Some(bar) = &self.current_bar {
            bar.set_position(self.progress.records_processed());
        }
        self.batches_since_log += 1;
        if self.batches_since_log >= self.log_every_n_batches {
            self.batches_since_log = 0;
            if let Some(ui) = &self.ui {
                ui.log(&self.progress.status_line());
            }
        }
    }

    /// Finalize the current pass: print summary line, drop the
    /// active bar.
    pub fn end_pass(&mut self) {
        if let Some(ui) = &self.ui {
            ui.log(&self.progress.status_line());
        }
        self.current_bar = None;
    }

    /// Emit a one-off milestone log line. Used by the orchestrator
    /// for events that don't fit the batch cadence (template
    /// synthesis summary, downscale events, etc.).
    pub fn log(&self, msg: impl Into<String>) {
        if let Some(ui) = &self.ui {
            ui.log(&msg.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_labels_match() {
        assert_eq!(SurveyPass::Pass1.label(), "Pass 1/2 — surveying");
        assert_eq!(SurveyPass::Pass2.label(), "Pass 2/2 — profiling");
        assert_eq!(SurveyPass::CrossField.label(), "cross-field finalize");
    }

    #[test]
    fn progress_atomics_update() {
        let p = SurveyProgress::new();
        assert_eq!(p.records_processed(), 0);
        p.add_records(100);
        p.add_records(50);
        assert_eq!(p.records_processed(), 150);
        p.reset_records();
        assert_eq!(p.records_processed(), 0);
    }

    #[test]
    fn rate_zero_before_clock_advance() {
        let p = SurveyProgress::new();
        // Immediately after construction, elapsed ≈ 0 → rate may be
        // 0 or a finite value depending on clock resolution. The
        // contract is "no division by zero".
        let _ = p.rate();
    }

    #[test]
    fn eta_zero_when_total_unknown() {
        let p = SurveyProgress::new();
        p.add_records(10);
        assert_eq!(p.eta_secs(), 0.0);
    }

    #[test]
    fn status_line_includes_pass_and_counts() {
        let p = SurveyProgress::new();
        p.set_pass(SurveyPass::Pass1);
        p.set_records_total(1_000);
        p.add_records(250);
        p.set_fields_classified(42);
        p.set_unstable_count(3);
        let line = p.status_line();
        assert!(line.contains("Pass 1/2"));
        assert!(line.contains("records=250/1000"));
        assert!(line.contains("fields=42"));
        assert!(line.contains("unstable=3"));
    }

    #[test]
    fn driver_without_ui_still_tracks_progress() {
        let progress = Arc::new(SurveyProgress::new());
        let mut driver = ProgressDriver::new(progress.clone(), None, 10);
        driver.begin_pass(SurveyPass::Pass1, 1_000);
        driver.tick(100);
        driver.tick(200);
        assert_eq!(progress.records_processed(), 300);
        driver.end_pass();
        // After end_pass the current pass is still Pass1 (driver
        // doesn't auto-advance to Done); the orchestrator sets that.
        assert_eq!(progress.current_pass(), SurveyPass::Pass1);
    }
}
