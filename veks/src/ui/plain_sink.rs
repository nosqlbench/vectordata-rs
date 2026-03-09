// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Plain-text [`UiSink`] for non-TTY output (pipes, files, CI).
//!
//! Progress events are silently ignored (no bars in piped output).
//! Log and emit events write directly to stdout.

use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};

use super::event::{ProgressId, UiEvent};
use super::sink::UiSink;

/// Sink that writes plain text to stdout and discards progress events.
pub struct PlainSink {
    next_id: AtomicU32,
}

impl PlainSink {
    pub fn new() -> Self {
        PlainSink {
            next_id: AtomicU32::new(0),
        }
    }
}

impl Default for PlainSink {
    fn default() -> Self {
        Self::new()
    }
}

impl UiSink for PlainSink {
    fn send(&self, event: UiEvent) {
        match event {
            // Progress events are invisible in plain mode.
            UiEvent::ProgressCreate { .. }
            | UiEvent::ProgressUpdate { .. }
            | UiEvent::ProgressInc { .. }
            | UiEvent::ProgressMessage { .. }
            | UiEvent::ProgressFinish { .. }
            | UiEvent::ResourceStatus { .. }
            | UiEvent::SetContext { .. }
            | UiEvent::SetStepYaml { .. }
            | UiEvent::SuspendBegin
            | UiEvent::SuspendEnd
            | UiEvent::Clear => {}

            UiEvent::Log { message } => {
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{}", message);
            }

            UiEvent::Emit { text } => {
                let mut out = std::io::stdout().lock();
                let _ = write!(out, "{}", text);
            }

            UiEvent::EmitLn { text } => {
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{}", text);
            }
        }
    }

    fn next_progress_id(&self) -> ProgressId {
        ProgressId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_sink_ids_are_unique() {
        let sink = PlainSink::new();
        let a = sink.next_progress_id();
        let b = sink.next_progress_id();
        assert_ne!(a, b);
    }

    #[test]
    fn plain_sink_progress_is_silent() {
        // Just verify no panic — progress events are no-ops.
        let sink = PlainSink::new();
        let id = sink.next_progress_id();
        sink.send(UiEvent::ProgressCreate {
            id,
            kind: super::super::event::ProgressKind::Bar,
            total: 100,
            label: "test".into(),
        });
        sink.send(UiEvent::ProgressUpdate { id, position: 50 });
        sink.send(UiEvent::ProgressFinish { id });
    }
}
