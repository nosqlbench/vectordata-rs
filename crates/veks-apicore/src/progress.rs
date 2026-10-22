// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! How long work says how far it has got.
//!
//! Two shapes, by what is being measured:
//!
//! - [`ProgressSink`]: a labelled scan or pass, `done` of `total` units
//!   (rows verified, bytes hashed). Every long phase of the API reports
//!   through it, so a caller writes one sink and hears all of them. A
//!   closure `FnMut(&str, u64, u64)` is a sink; [`NoProgress`] hears
//!   nothing.
//! - [`DownloadProgress`]: a transfer in flight, shared by the threads
//!   moving it: bytes and chunks done of planned, and whether a chunk
//!   failed. [`ByteSource::prebuffer_with_progress`](crate::ByteSource::prebuffer_with_progress)
//!   hands it to its callback.
//!
//! A sink is called through `&mut dyn`, so work reports at most once
//! per batch (a chunk, a page run, a block of rows), never per record
//! (`srd-crate-boundaries.md` PR-5).
//!
//! ```
//! use veks_apicore::{NoProgress, ProgressSink};
//!
//! fn scan(rows: u64, progress: &mut dyn ProgressSink) {
//!     progress.report("scanning", 0, rows);
//!     for done in (0..=rows).step_by(1000).skip(1).chain([rows]) {
//!         progress.report("scanning", done.min(rows), rows);
//!     }
//! }
//!
//! let mut heard = Vec::new();
//! scan(2500, &mut |label: &str, done, total| heard.push(format!("{label} {done}/{total}")));
//! assert_eq!(heard.first().unwrap(), "scanning 0/2500");
//! assert_eq!(heard.last().unwrap(), "scanning 2500/2500");
//! scan(10, &mut NoProgress);
//! ```

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Hears how far a labelled pass has got: `done` of `total` units of
/// the pass `label` (`scanning neighbor_indices`, `hashing base.fvecs`).
///
/// Reported once at the start (`done == 0`) and once at the end
/// (`done == total`), and between them at most once per batch of work.
pub trait ProgressSink {
    /// `done` of `total` units of `label` are finished.
    fn report(&mut self, label: &str, done: u64, total: u64);
}

impl<F: FnMut(&str, u64, u64)> ProgressSink for F {
    #[inline]
    fn report(&mut self, label: &str, done: u64, total: u64) {
        self(label, done, total)
    }
}

/// A sink that hears nothing: for a caller with no use for progress.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    #[inline]
    fn report(&mut self, _label: &str, _done: u64, _total: u64) {}
}

/// Thread-safe download progress tracker.
///
/// Uses atomic counters so multiple download threads can update progress
/// concurrently without locking.
#[derive(Debug)]
pub struct DownloadProgress {
    /// Total bytes to download.
    total_bytes: u64,
    /// Bytes downloaded so far.
    downloaded_bytes: AtomicU64,
    /// Total chunks to download.
    total_chunks: u32,
    /// Chunks completed so far.
    completed_chunks: AtomicU32,
    /// Set to true if any chunk fails permanently.
    failed: AtomicBool,
}

impl DownloadProgress {
    /// Create a new progress tracker.
    #[inline]
    pub fn new(total_bytes: u64, total_chunks: u32) -> Self {
        DownloadProgress {
            total_bytes,
            downloaded_bytes: AtomicU64::new(0),
            total_chunks,
            completed_chunks: AtomicU32::new(0),
            failed: AtomicBool::new(false),
        }
    }

    /// Total bytes expected.
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Bytes downloaded so far.
    #[inline]
    pub fn downloaded_bytes(&self) -> u64 {
        self.downloaded_bytes.load(Ordering::Relaxed)
    }

    /// Add to downloaded byte count.
    #[inline]
    pub fn add_downloaded_bytes(&self, bytes: u64) {
        self.downloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Total chunks expected.
    #[inline]
    pub fn total_chunks(&self) -> u32 {
        self.total_chunks
    }

    /// Chunks completed so far.
    #[inline]
    pub fn completed_chunks(&self) -> u32 {
        self.completed_chunks.load(Ordering::Relaxed)
    }

    /// Increment completed chunk count.
    #[inline]
    pub fn increment_completed(&self) {
        self.completed_chunks.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether the download has failed.
    #[inline]
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    /// Signal that the download has failed.
    #[inline]
    pub fn mark_failed(&self) {
        self.failed.store(true, Ordering::Relaxed);
    }

    /// Fraction complete (0.0 to 1.0).
    #[inline]
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        self.downloaded_bytes() as f64 / self.total_bytes as f64
    }

    /// Whether all chunks have been completed.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.completed_chunks() >= self.total_chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracking() {
        let p = DownloadProgress::new(4096, 4);
        assert_eq!(p.fraction(), 0.0);
        assert!(!p.is_complete());

        p.add_downloaded_bytes(1024);
        p.increment_completed();
        assert_eq!(p.downloaded_bytes(), 1024);
        assert_eq!(p.completed_chunks(), 1);
        assert!((p.fraction() - 0.25).abs() < 0.001);

        p.add_downloaded_bytes(3072);
        p.increment_completed();
        p.increment_completed();
        p.increment_completed();
        assert!(p.is_complete());
        assert!((p.fraction() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_failure_flag() {
        let p = DownloadProgress::new(1000, 1);
        assert!(!p.is_failed());
        p.mark_failed();
        assert!(p.is_failed());
    }

    #[test]
    fn test_empty_progress() {
        let p = DownloadProgress::new(0, 0);
        assert_eq!(p.fraction(), 1.0);
        assert!(p.is_complete());
    }
}
