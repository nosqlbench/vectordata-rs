// Copyright (c) DataStax, Inc.
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
}
