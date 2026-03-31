// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Download progress tracking with atomic counters.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

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
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Bytes downloaded so far.
    pub fn downloaded_bytes(&self) -> u64 {
        self.downloaded_bytes.load(Ordering::Relaxed)
    }

    /// Add to downloaded byte count.
    pub fn add_downloaded_bytes(&self, bytes: u64) {
        self.downloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Total chunks expected.
    pub fn total_chunks(&self) -> u32 {
        self.total_chunks
    }

    /// Chunks completed so far.
    pub fn completed_chunks(&self) -> u32 {
        self.completed_chunks.load(Ordering::Relaxed)
    }

    /// Increment completed chunk count.
    pub fn increment_completed(&self) {
        self.completed_chunks.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether the download has failed.
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    /// Signal that the download has failed.
    pub fn mark_failed(&self) {
        self.failed.store(true, Ordering::Relaxed);
    }

    /// Fraction complete (0.0 to 1.0).
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        self.downloaded_bytes() as f64 / self.total_bytes as f64
    }

    /// Whether all chunks have been completed.
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
