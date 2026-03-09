// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! UI-agnostic eventing layer.
//!
//! Pipeline code communicates visual intent through algebraic [`event::UiEvent`]
//! values dispatched to a [`sink::UiSink`].  The [`handle::UiHandle`] facade
//! provides an ergonomic API (`.bar()`, `.log()`, `.emit()`) that maps directly
//! to events.
//!
//! Concrete rendering backends implement [`sink::UiSink`]:
//!
//! | Sink | When |
//! |------|------|
//! | [`plain_sink::PlainSink`] | stdout is a pipe or file (no progress bars) |
//! | [`test_sink::TestSink`] | unit tests (records events for assertions) |
//! | *(future)* `RatatuiSink` | TTY — rich terminal UI |
//!
//! No pipeline code should ever import a rendering library directly.

pub mod event;
pub mod handle;
pub mod plain_sink;
pub mod ratatui_sink;
pub mod sink;
pub mod test_sink;

// Re-export the handful of types that most callers need.
pub use event::{ProgressId, ProgressKind, ResourceMetrics, UiEvent};
pub use handle::{ProgressHandle, UiHandle};
pub use plain_sink::PlainSink;
pub use ratatui_sink::RatatuiSink;
pub use sink::UiSink;
pub use test_sink::TestSink;

/// Create a `UiHandle` with the appropriate sink for the current environment.
///
/// Uses `RatatuiSink` when stdout is a TTY, `PlainSink` otherwise (pipes, CI).
/// Falls back to `PlainSink` if the ratatui terminal cannot be initialized.
pub fn auto_ui_handle() -> UiHandle {
    auto_ui_handle_with_interval(std::time::Duration::from_secs(1))
}

/// Create a UI handle with a custom redraw interval.
pub fn auto_ui_handle_with_interval(redraw_interval: std::time::Duration) -> UiHandle {
    let sink: std::sync::Arc<dyn UiSink> = if atty::is(atty::Stream::Stdout) {
        match RatatuiSink::with_redraw_interval(16, redraw_interval) {
            Ok(s) => std::sync::Arc::new(s),
            Err(_) => std::sync::Arc::new(PlainSink::new()),
        }
    } else {
        std::sync::Arc::new(PlainSink::new())
    };
    UiHandle::new(sink)
}
