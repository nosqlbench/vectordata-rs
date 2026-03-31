// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! The [`UiSink`] trait — the only UI surface that pipeline code touches.
//!
//! All rendering decisions live behind this trait.  Commands, the runner,
//! and any other pipeline code communicate visual intent exclusively by
//! calling methods on a `dyn UiSink`.

use super::event::{ProgressId, UiEvent};

/// Receives UI events and decides how to render them.
///
/// Implementations include:
/// - **`PlainSink`** — non-TTY / pipe output (plain text, no ANSI)
/// - **`TestSink`** — collects events for test assertions
/// - *(future)* **`RatatuiSink`** — rich terminal rendering
///
/// The trait is `Send + Sync` so an `Arc<dyn UiSink>` can be shared across
/// threads (e.g., background resource monitor, parallel segment processing).
pub trait UiSink: Send + Sync {
    /// Dispatch a single UI event.
    fn send(&self, event: UiEvent);

    /// Allocate a fresh [`ProgressId`] unique within this sink.
    fn next_progress_id(&self) -> ProgressId;

    /// Retrieve buffered log messages for post-session console output.
    ///
    /// Sinks that render to an alternate screen (ratatui) buffer all log
    /// messages so they can be replayed to stdout after the TUI exits.
    /// Other sinks return an empty vec (their output already went to stdout).
    fn take_console_log(&self) -> Vec<String> {
        Vec::new()
    }

    /// Shut down the sink, restoring the terminal to normal mode.
    ///
    /// Must be called before printing to stdout/stderr after TUI mode.
    /// The global `log` crate logger holds a leaked `Arc<dyn UiSink>`,
    /// so `Drop` alone is insufficient — the refcount never reaches zero.
    fn shutdown(&self) {}
}
