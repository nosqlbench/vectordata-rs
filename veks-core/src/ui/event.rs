// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! UI-agnostic event primitives.
//!
//! Every visual action the pipeline can express is modeled as a value in the
//! [`UiEvent`] enum.  No rendering, no ANSI codes, no framework types —
//! just algebraic data that any backend can interpret.

use std::fmt;

/// Structured resource metrics for direct use by chart rendering.
///
/// Produced by `ResourceStatusSource::sample_metrics()` alongside the
/// formatted status line.  Chart code reads these numeric values directly
/// rather than re-parsing the text representation.
#[derive(Debug, Clone, Default)]
pub struct ResourceMetrics {
    /// Resident set size in bytes.
    pub rss_bytes: u64,
    /// RSS memory budget ceiling in bytes (0 if no budget).
    pub rss_ceiling_bytes: u64,
    /// Instantaneous CPU usage as percentage (0–num_cpus×100).
    pub cpu_pct: f64,
    /// CPU ceiling percentage (num_cpus × 100).
    pub cpu_ceiling: f64,
    /// Per-core CPU utilization percentages (0–100 each).
    pub cpu_cores: Vec<u64>,
    /// Active thread count.
    pub threads: usize,
    /// Thread budget ceiling (0 if no budget).
    pub thread_ceiling: u64,
    /// Cumulative I/O read bytes (for delta-based rate computation).
    pub io_read_bytes: u64,
    /// Cumulative I/O write bytes.
    pub io_write_bytes: u64,
    /// Read I/O requests in flight.
    pub ioq_read: u64,
    /// Write I/O requests in flight.
    pub ioq_write: u64,
    /// System page cache size in bytes.
    pub page_cache_bytes: u64,
    /// Page cache hit ratio (0.0–100.0), or `None` if no interval data yet.
    pub page_cache_hit_pct: Option<f64>,
    /// Cumulative major page faults.
    pub major_faults: u64,
    /// Cumulative minor page faults.
    pub minor_faults: u64,
    /// Whether EMERGENCY state is active.
    pub emergency: bool,
    /// Whether throttle is active.
    pub throttle: bool,
}

/// Opaque handle that identifies a progress indicator within a session.
///
/// Created by [`UiSink::next_progress_id`] and referenced in subsequent
/// progress events.  The value is meaningful only to the sink that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgressId(pub u32);

impl fmt::Display for ProgressId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "progress#{}", self.0)
    }
}

/// The shape of a progress indicator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressKind {
    /// Determinate — a known total, rendered as a bar / gauge.
    Bar,
    /// Indeterminate — no known total, rendered as a spinner.
    Spinner,
    /// Unit-interval gauge — position/total maps to 0.0–1.0.
    /// Label shows only the label text and message, not `[pos/total] pct%`.
    Ratio,
}

/// A single UI-agnostic event.
///
/// Pipeline code emits these; a [`super::sink::UiSink`] implementation
/// decides how (or whether) to render them.
#[derive(Debug, Clone)]
pub enum UiEvent {
    // ── Progress lifecycle ──────────────────────────────────────────

    /// Create a new progress indicator.
    ProgressCreate {
        id: ProgressId,
        kind: ProgressKind,
        total: u64,
        label: String,
        /// The unit label for rate display (e.g., "rec", "files", "chunks").
        unit: String,
    },

    /// Update the absolute position of a progress indicator.
    ProgressUpdate {
        id: ProgressId,
        position: u64,
    },

    /// Increment the position of a progress indicator by a delta.
    ProgressInc {
        id: ProgressId,
        delta: u64,
    },

    /// Change the trailing message on a progress indicator.
    ProgressMessage {
        id: ProgressId,
        message: String,
    },

    /// Mark a progress indicator as finished and remove it.
    ProgressFinish {
        id: ProgressId,
    },

    // ── Text output ─────────────────────────────────────────────────

    /// A log line that scrolls above the progress region.
    ///
    /// Backends that support a fixed progress area insert this text above
    /// that area.  Non-TTY backends simply write a line.
    Log {
        message: String,
    },

    /// Raw text output (no trailing newline).
    Emit {
        text: String,
    },

    /// Raw text output with a trailing newline.
    EmitLn {
        text: String,
    },

    // ── Resource status ─────────────────────────────────────────────

    /// Update the resource-utilization status line.
    ///
    /// `line` is the formatted text for display; `metrics` carries the
    /// structured numeric values for chart rendering.
    ResourceStatus {
        line: String,
        metrics: ResourceMetrics,
    },

    // ── Lifecycle / batching ────────────────────────────────────────

    /// Begin a batch of creates — suppress intermediate redraws.
    SuspendBegin,

    /// End a batch — perform a single atomic redraw.
    SuspendEnd,

    /// Set the context label (dataset name, profile, current step).
    SetContext {
        label: String,
    },

    /// Set the YAML snippet for the current pipeline step.
    ///
    /// Displayed as a scrollable panel alongside the throughput chart.
    /// Send an empty string to clear it.
    SetStepYaml {
        yaml: String,
    },

    /// Clear the entire progress region.
    Clear,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_id_display() {
        assert_eq!(ProgressId(7).to_string(), "progress#7");
    }

    #[test]
    fn event_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<UiEvent>();
    }
}
