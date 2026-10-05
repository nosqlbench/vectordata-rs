// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Thread-based asynchronous task execution with progress polling.
//!
//! This module provides [`SlabTask`] and [`SlabProgress`] for running
//! long-lived slab operations (bulk reads, bulk writes) on a background
//! thread while the caller polls for progress.
//!
//! # Architecture
//!
//! Since the crate has no async runtime dependency, background work is
//! dispatched via [`std::thread::spawn`]. The caller receives a
//! [`SlabTask<T>`] handle that exposes:
//!
//! - [`SlabTask::progress`] — a [`SlabProgress`] snapshot (total items,
//!   completed items, done flag)
//! - [`SlabTask::is_done`] — quick check without joining
//! - [`SlabTask::wait`] — blocks until the background thread finishes
//!   and returns the result
//!
//! # Example
//!
//! ```rust,no_run
//! use slabtastic::task::{SlabProgress, SlabTask};
//!
//! // (Typically you'd get a SlabTask from SlabReader::read_to_sink_async
//! //  or SlabWriter::write_from_iter_async — this is a conceptual sketch.)
//! # fn example(task: SlabTask<u64>) {
//! while !task.is_done() {
//!     let p = task.progress();
//!     println!("{}/{} ({:.1}%)", p.completed(), p.total(),
//!              p.fraction() * 100.0);
//!     std::thread::sleep(std::time::Duration::from_millis(100));
//! }
//! let result = task.wait().expect("task succeeded");
//! println!("done — processed {result} records");
//! # }
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::error::Result;

/// Read-only view of a background task's progress.
///
/// All accessors perform relaxed atomic loads, so values are eventually
/// consistent but always safe to read from any thread.
#[derive(Clone)]
pub struct SlabProgress {
    total: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    done: Arc<AtomicBool>,
}

impl SlabProgress {
    /// Return the total number of items the task expects to process.
    ///
    /// This may be zero if the total is not yet known.
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Return the number of items processed so far.
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }

    /// Return `true` once the background thread has finished.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// Return the fraction of work completed (`completed / total`).
    ///
    /// Returns `0.0` if `total` is zero.
    pub fn fraction(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            return 0.0;
        }
        self.completed() as f64 / t as f64
    }
}

/// Handle to a background slab operation.
///
/// Wraps a [`JoinHandle`] and a [`SlabProgress`] so the caller can poll
/// progress and eventually collect the result.
pub struct SlabTask<T> {
    handle: Option<JoinHandle<Result<T>>>,
    progress: SlabProgress,
}

impl<T> SlabTask<T> {
    /// Return a reference to the task's progress counters.
    pub fn progress(&self) -> &SlabProgress {
        &self.progress
    }

    /// Return `true` if the background thread has finished.
    pub fn is_done(&self) -> bool {
        self.progress.is_done()
    }

    /// Block until the background thread finishes and return its result.
    ///
    /// Consumes the task handle. If the thread panicked, the panic is
    /// propagated to the caller.
    pub fn wait(mut self) -> Result<T> {
        self.handle
            .take()
            .expect("handle already consumed")
            .join()
            .expect("background thread panicked")
    }
}

/// Mutable side of the progress counters, held by the background thread.
///
/// This is an internal helper — not part of the public API.
pub(crate) struct SlabProgressTracker {
    total: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    done: Arc<AtomicBool>,
}

impl SlabProgressTracker {
    /// Set the total number of items to process.
    pub(crate) fn set_total(&self, n: u64) {
        self.total.store(n, Ordering::Relaxed);
    }

    /// Increment the completed counter by one.
    pub(crate) fn inc(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark the task as done.
    pub(crate) fn mark_done(&self) {
        self.done.store(true, Ordering::Release);
    }
}

/// Create a paired `(SlabProgress, SlabProgressTracker)`.
pub(crate) fn new_progress() -> (SlabProgress, SlabProgressTracker) {
    let total = Arc::new(AtomicU64::new(0));
    let completed = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));

    let progress = SlabProgress {
        total: Arc::clone(&total),
        completed: Arc::clone(&completed),
        done: Arc::clone(&done),
    };
    let tracker = SlabProgressTracker {
        total,
        completed,
        done,
    };
    (progress, tracker)
}

/// Create a [`SlabTask`] from a join handle and progress counters.
pub(crate) fn new_task<T>(handle: JoinHandle<Result<T>>, progress: SlabProgress) -> SlabTask<T> {
    SlabTask {
        handle: Some(handle),
        progress,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify progress tracking: create, increment, fraction, done flag.
    #[test]
    fn test_progress_tracking() {
        let (progress, tracker) = new_progress();

        assert_eq!(progress.total(), 0);
        assert_eq!(progress.completed(), 0);
        assert!(!progress.is_done());
        assert_eq!(progress.fraction(), 0.0);

        tracker.set_total(10);
        assert_eq!(progress.total(), 10);

        tracker.inc();
        tracker.inc();
        tracker.inc();
        assert_eq!(progress.completed(), 3);
        assert!((progress.fraction() - 0.3).abs() < 1e-9);

        tracker.mark_done();
        assert!(progress.is_done());
    }

    /// Verify that SlabTask::wait returns the correct result.
    #[test]
    fn test_task_wait_returns_result() {
        let (progress, tracker) = new_progress();
        let handle = std::thread::spawn(move || {
            tracker.set_total(5);
            for _ in 0..5 {
                tracker.inc();
            }
            tracker.mark_done();
            Ok(42u64)
        });

        let task = new_task(handle, progress);
        let result = task.wait().unwrap();
        assert_eq!(result, 42);
    }
}
