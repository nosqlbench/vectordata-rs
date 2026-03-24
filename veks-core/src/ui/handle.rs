// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Ergonomic handle for progress indicators.
//!
//! [`ProgressHandle`] wraps a [`ProgressId`] and an `Arc<dyn UiSink>` so
//! that call sites can write `handle.set_position(n)` instead of manually
//! constructing [`UiEvent`] values.  On drop the progress indicator is
//! automatically finished.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::event::{ProgressId, ProgressKind, ResourceMetrics, UiEvent};
use super::sink::UiSink;

/// Owned handle to a single progress indicator.
///
/// Created via [`UiHandle::bar`] or [`UiHandle::spinner`].
/// Sends a `ProgressFinish` event on drop.
pub struct ProgressHandle {
    id: ProgressId,
    sink: Arc<dyn UiSink>,
    finished: AtomicBool,
}

impl ProgressHandle {
    /// The underlying id, in case callers need to correlate.
    pub fn id(&self) -> ProgressId {
        self.id
    }

    /// Set the absolute position.
    pub fn set_position(&self, position: u64) {
        self.sink.send(UiEvent::ProgressUpdate {
            id: self.id,
            position,
        });
    }

    /// Increment the position by `delta`.
    pub fn inc(&self, delta: u64) {
        self.sink.send(UiEvent::ProgressInc {
            id: self.id,
            delta,
        });
    }

    /// Update the trailing message.
    pub fn set_message(&self, message: impl Into<String>) {
        self.sink.send(UiEvent::ProgressMessage {
            id: self.id,
            message: message.into(),
        });
    }

    /// Explicitly finish and remove the indicator.
    pub fn finish(&self) {
        if !self.finished.swap(true, Ordering::Relaxed) {
            self.sink.send(UiEvent::ProgressFinish { id: self.id });
        }
    }
}

impl Drop for ProgressHandle {
    fn drop(&mut self) {
        self.finish();
    }
}

/// High-level facade over a [`UiSink`].
///
/// This is the primary API that pipeline code uses.  It wraps
/// `Arc<dyn UiSink>` and provides convenience methods that map to
/// [`UiEvent`] values.
///
/// `UiHandle` is cheap to clone (just an `Arc` bump).
#[derive(Clone)]
pub struct UiHandle {
    sink: Arc<dyn UiSink>,
}

impl UiHandle {
    /// Wrap a sink in a handle.
    pub fn new(sink: Arc<dyn UiSink>) -> Self {
        UiHandle { sink }
    }

    /// Access the underlying sink.
    pub fn sink(&self) -> &Arc<dyn UiSink> {
        &self.sink
    }

    /// Create a determinate progress bar with the default unit "rec".
    pub fn bar(&self, total: u64, label: impl Into<String>) -> ProgressHandle {
        self.bar_with_unit(total, label, "rec")
    }

    /// Create a determinate progress bar with a custom unit label.
    ///
    /// The `unit` controls how throughput is displayed (e.g., "42 files/s"
    /// instead of "42 rec/s").
    pub fn bar_with_unit(&self, total: u64, label: impl Into<String>, unit: impl Into<String>) -> ProgressHandle {
        let id = self.sink.next_progress_id();
        self.sink.send(UiEvent::ProgressCreate {
            id,
            kind: ProgressKind::Bar,
            total,
            label: label.into(),
            unit: unit.into(),
        });
        ProgressHandle {
            id,
            sink: Arc::clone(&self.sink),
            finished: AtomicBool::new(false),
        }
    }

    /// Create a unit-interval ratio gauge (0.0–1.0).
    ///
    /// Position maps to `pos / total` where total defines the precision.
    /// The gauge shows the label and message text without `[pos/total] pct%`.
    pub fn ratio(&self, total: u64, label: impl Into<String>) -> ProgressHandle {
        let id = self.sink.next_progress_id();
        self.sink.send(UiEvent::ProgressCreate {
            id,
            kind: ProgressKind::Ratio,
            total,
            label: label.into(),
            unit: "rec".into(),
        });
        ProgressHandle {
            id,
            sink: Arc::clone(&self.sink),
            finished: AtomicBool::new(false),
        }
    }

    /// Create an indeterminate spinner.
    pub fn spinner(&self, label: impl Into<String>) -> ProgressHandle {
        let id = self.sink.next_progress_id();
        self.sink.send(UiEvent::ProgressCreate {
            id,
            kind: ProgressKind::Spinner,
            total: 0,
            label: label.into(),
            unit: "rec".into(),
        });
        ProgressHandle {
            id,
            sink: Arc::clone(&self.sink),
            finished: AtomicBool::new(false),
        }
    }

    /// Retrieve buffered log messages for post-TUI console output.
    ///
    /// Call after the TUI session ends to get all log messages that were
    /// displayed during the session. For non-TUI sinks this returns empty.
    pub fn take_console_log(&self) -> Vec<String> {
        self.sink.take_console_log()
    }

    /// Emit a log line above the progress region.
    pub fn log(&self, message: &str) {
        self.sink.send(UiEvent::Log {
            message: message.to_string(),
        });
    }

    /// Emit raw text (no newline).
    pub fn emit(&self, text: impl Into<String>) {
        self.sink.send(UiEvent::Emit { text: text.into() });
    }

    /// Emit raw text with trailing newline.
    pub fn emitln(&self, text: impl Into<String>) {
        self.sink.send(UiEvent::EmitLn { text: text.into() });
    }

    /// Update the resource status line (text only, no structured metrics).
    ///
    /// Prefer [`resource_status_with_metrics`] when structured data is available.
    pub fn resource_status(&self, line: impl Into<String>) {
        self.sink.send(UiEvent::ResourceStatus {
            line: line.into(),
            metrics: ResourceMetrics::default(),
        });
    }

    /// Update the resource status with both formatted text and structured metrics.
    ///
    /// The TUI uses `metrics` directly for chart rendering, avoiding
    /// re-parsing the formatted text.
    pub fn resource_status_with_metrics(&self, line: String, metrics: ResourceMetrics) {
        self.sink.send(UiEvent::ResourceStatus { line, metrics });
    }

    /// Begin a batch — suppress intermediate redraws until [`suspend_end`].
    pub fn suspend_begin(&self) {
        self.sink.send(UiEvent::SuspendBegin);
    }

    /// End a batch — perform a single atomic redraw.
    pub fn suspend_end(&self) {
        self.sink.send(UiEvent::SuspendEnd);
    }

    /// Increment a progress indicator by id (for cross-thread sharing).
    ///
    /// Use this when multiple tasks need to update the same bar.
    /// The `ProgressId` is `Copy`, so it can be shared freely across threads.
    pub fn inc_by_id(&self, id: ProgressId, delta: u64) {
        self.sink.send(UiEvent::ProgressInc { id, delta });
    }

    /// Set a progress message by id (for cross-thread sharing).
    pub fn set_message_by_id(&self, id: ProgressId, message: impl Into<String>) {
        self.sink.send(UiEvent::ProgressMessage {
            id,
            message: message.into(),
        });
    }

    /// Finish a progress indicator by id.
    pub fn finish_by_id(&self, id: ProgressId) {
        self.sink.send(UiEvent::ProgressFinish { id });
    }

    /// Set the context label shown in the progress region header.
    pub fn set_context(&self, label: impl Into<String>) {
        self.sink.send(UiEvent::SetContext {
            label: label.into(),
        });
    }

    /// Set the YAML snippet for the current pipeline step.
    ///
    /// Displayed as a scrollable panel alongside the throughput chart.
    /// Pass an empty string to clear it.
    pub fn set_step_yaml(&self, yaml: impl Into<String>) {
        self.sink.send(UiEvent::SetStepYaml {
            yaml: yaml.into(),
        });
    }

    /// Clear the entire progress region.
    pub fn clear(&self) {
        self.sink.send(UiEvent::Clear);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_sink::TestSink;

    #[test]
    fn bar_lifecycle() {
        let sink = Arc::new(TestSink::new());
        let ui = UiHandle::new(sink.clone());

        let pb = ui.bar(100, "testing");
        pb.set_position(50);
        pb.set_message("halfway");
        pb.finish();

        let events = sink.events();
        assert!(matches!(events[0], UiEvent::ProgressCreate { .. }));
        assert!(matches!(events[1], UiEvent::ProgressUpdate { position: 50, .. }));
        assert!(matches!(events[2], UiEvent::ProgressMessage { .. }));
        assert!(matches!(events[3], UiEvent::ProgressFinish { .. }));
        // drop should not double-finish
        drop(pb);
        assert_eq!(sink.events().len(), 4);
    }

    #[test]
    fn spinner_auto_finish_on_drop() {
        let sink = Arc::new(TestSink::new());
        let ui = UiHandle::new(sink.clone());

        {
            let _sp = ui.spinner("working");
        }

        let events = sink.events();
        assert_eq!(events.len(), 2); // Create + Finish
        assert!(matches!(events[1], UiEvent::ProgressFinish { .. }));
    }

    #[test]
    fn log_and_emit() {
        let sink = Arc::new(TestSink::new());
        let ui = UiHandle::new(sink.clone());

        ui.log("step 1 complete");
        ui.emit("partial");
        ui.emitln("full line");

        let events = sink.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], UiEvent::Log { message } if message == "step 1 complete"));
        assert!(matches!(&events[1], UiEvent::Emit { text } if text == "partial"));
        assert!(matches!(&events[2], UiEvent::EmitLn { text } if text == "full line"));
    }
}
