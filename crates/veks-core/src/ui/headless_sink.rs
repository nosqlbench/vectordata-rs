// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Headless [`UiSink`] for non-interactive execution.
//!
//! Does nothing with events — all output goes through the `log` crate
//! via [`install_logger`](super::install_logger). This sink is used
//! when `--headless` is passed, in CI, and in integration tests.
//!
//! Since `UiHandle::log()` calls `log::info!()` and `UiHandle::bar()`
//! creates progress IDs that are logged on create/finish, headless mode
//! still produces full diagnostic output through the log channel.

use std::sync::atomic::{AtomicU32, Ordering};

use super::event::{ProgressId, UiEvent};
use super::sink::UiSink;

/// A no-op sink. All output flows through the `log` crate instead.
pub struct HeadlessSink {
    next_id: AtomicU32,
}

impl HeadlessSink {
    pub fn new() -> Self {
        HeadlessSink {
            next_id: AtomicU32::new(0),
        }
    }
}

impl Default for HeadlessSink {
    fn default() -> Self {
        Self::new()
    }
}

impl UiSink for HeadlessSink {
    fn send(&self, _event: UiEvent) {
        // No-op: all meaningful output goes through log::info!() etc.
        // Progress lifecycle events are logged by install_logger's
        // PipelineLogger when it receives them through UiHandle methods.
    }

    fn next_progress_id(&self) -> ProgressId {
        ProgressId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}
