// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Test harness [`UiSink`] that records events for assertions.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use super::event::{ProgressId, UiEvent};
use super::sink::UiSink;

/// Sink that collects all events into a `Vec` for test inspection.
pub struct TestSink {
    next_id: AtomicU32,
    events: Mutex<Vec<UiEvent>>,
}

impl TestSink {
    pub fn new() -> Self {
        TestSink {
            next_id: AtomicU32::new(0),
            events: Mutex::new(Vec::new()),
        }
    }

    /// Return a snapshot of all recorded events.
    pub fn events(&self) -> Vec<UiEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Number of events recorded so far.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Whether any events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all recorded events.
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    /// Return only log messages.
    pub fn log_messages(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                UiEvent::Log { message } => Some(message.clone()),
                _ => None,
            })
            .collect()
    }
}

impl Default for TestSink {
    fn default() -> Self {
        Self::new()
    }
}

impl UiSink for TestSink {
    fn send(&self, event: UiEvent) {
        self.events.lock().unwrap().push(event);
    }

    fn next_progress_id(&self) -> ProgressId {
        ProgressId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::event::ProgressKind;

    #[test]
    fn records_events_in_order() {
        let sink = TestSink::new();
        sink.send(UiEvent::Log {
            message: "hello".into(),
        });
        sink.send(UiEvent::EmitLn {
            text: "world".into(),
        });
        assert_eq!(sink.len(), 2);
        assert_eq!(sink.log_messages(), vec!["hello"]);
    }

    #[test]
    fn clear_resets() {
        let sink = TestSink::new();
        sink.send(UiEvent::Clear);
        assert_eq!(sink.len(), 1);
        sink.clear();
        assert!(sink.is_empty());
    }

    #[test]
    fn ids_increment() {
        let sink = TestSink::new();
        assert_eq!(sink.next_progress_id(), ProgressId(0));
        assert_eq!(sink.next_progress_id(), ProgressId(1));
        assert_eq!(sink.next_progress_id(), ProgressId(2));
    }

    #[test]
    fn full_progress_lifecycle() {
        let sink = TestSink::new();
        let id = sink.next_progress_id();
        sink.send(UiEvent::ProgressCreate {
            id,
            kind: ProgressKind::Bar,
            total: 1000,
            label: "scan".into(),
            unit: "rec".into(),
        });
        sink.send(UiEvent::ProgressUpdate { id, position: 500 });
        sink.send(UiEvent::ProgressInc { id, delta: 100 });
        sink.send(UiEvent::ProgressMessage {
            id,
            message: "halfway".into(),
        });
        sink.send(UiEvent::ProgressFinish { id });
        assert_eq!(sink.len(), 5);
    }
}
