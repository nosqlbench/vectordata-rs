// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! UI-agnostic event primitives.
//!
//! Every visual action the pipeline can express is modeled as a value in the
//! [`UiEvent`] enum.  No rendering, no ANSI codes, no framework types —
//! just algebraic data that any backend can interpret.

use std::fmt;

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
    ResourceStatus {
        line: String,
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
