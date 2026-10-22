// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Byte ranges of one resource, fetched in chunks: [`ChunkedTransport`]
//! (where the bytes come from), [`ChunkRequest`] (one chunk) and
//! [`fetch_chunks_parallel`] (many, bounded, with retry and progress).
//!
//! The chunk stores ([`CachedChannel`](crate::CachedChannel),
//! [`ChunkStore`](crate::ChunkStore)) fetch through a
//! `ChunkedTransport`; [`HttpTransport`](crate::HttpTransport) is the
//! network one, and a test supplies one held in memory.

use std::io;
use std::sync::{Condvar, Mutex, PoisonError};

use crate::RetryPolicy;
use veks_apicore::DownloadProgress;

/// Byte-range data fetcher — abstracts HTTP vs local file access.
pub trait ChunkedTransport: Send + Sync {
    /// Fetch bytes in range `[start, start+len)` from the resource.
    fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>>;

    /// Total size of the resource in bytes.
    fn content_length(&self) -> io::Result<u64>;

    /// Whether the source supports byte-range requests.
    fn supports_range(&self) -> bool;
}

/// A chunk descriptor for parallel download scheduling.
#[derive(Debug, Clone, Copy)]
pub struct ChunkRequest {
    /// Chunk index (for merkle verification).
    pub index: u32,
    /// Byte offset within the content.
    pub start: u64,
    /// Byte length of this chunk.
    pub len: u64,
}

/// Fetch multiple chunks in parallel, at most `concurrency` at once,
/// each with `retry_policy`.
///
/// Returns `(chunk_index, data)` for each chunk, or its error. Stops
/// starting chunks once one has failed (`progress` records the failure);
/// `progress` counts every chunk that arrives.
pub fn fetch_chunks_parallel(
    transport: &dyn ChunkedTransport,
    chunks: &[ChunkRequest],
    retry_policy: &RetryPolicy,
    progress: &DownloadProgress,
    concurrency: usize,
) -> Vec<io::Result<(u32, Vec<u8>)>> {
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let semaphore = std::sync::Arc::new(Semaphore::new(concurrency));

        for chunk in chunks {
            if progress.is_failed() {
                break;
            }

            let permit = semaphore.clone();
            let chunk = *chunk;

            let handle = scope.spawn(move || {
                let _permit = permit.acquire();

                if progress.is_failed() {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "download aborted"));
                }

                let result = retry_policy.execute(|| transport.fetch_range(chunk.start, chunk.len));

                match &result {
                    Ok(data) => {
                        progress.add_downloaded_bytes(data.len() as u64);
                        progress.increment_completed();
                    }
                    Err(_) => {
                        progress.mark_failed();
                    }
                }

                result.map(|data| (chunk.index, data))
            });

            handles.push(handle);
        }

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(io::Error::other("thread panicked"))))
            .collect()
    })
}

/// Simple counting semaphore for bounding concurrency.
struct Semaphore {
    state: Mutex<usize>,
    cond: Condvar,
}

struct SemaphoreGuard<'a> {
    sem: &'a Semaphore,
}

impl Semaphore {
    fn new(permits: usize) -> Self {
        Semaphore { state: Mutex::new(permits.max(1)), cond: Condvar::new() }
    }

    fn acquire(&self) -> SemaphoreGuard<'_> {
        let mut count = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        while *count == 0 {
            count = self.cond.wait(count).unwrap_or_else(PoisonError::into_inner);
        }
        *count -= 1;
        SemaphoreGuard { sem: self }
    }
}

impl Drop for SemaphoreGuard<'_> {
    fn drop(&mut self) {
        let mut count = self.sem.state.lock().unwrap_or_else(PoisonError::into_inner);
        *count += 1;
        self.sem.cond.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory transport for testing.
    struct MemTransport {
        data: Vec<u8>,
    }

    impl ChunkedTransport for MemTransport {
        fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
            let start = start as usize;
            let end = start + len as usize;
            if end > self.data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "range exceeds content"));
            }
            Ok(self.data[start..end].to_vec())
        }

        fn content_length(&self) -> io::Result<u64> {
            Ok(self.data.len() as u64)
        }

        fn supports_range(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_fetch_chunks_parallel_basic() {
        let data = vec![0u8; 4096];
        let transport = MemTransport { data };
        let policy = RetryPolicy::default();
        let progress = DownloadProgress::new(4096, 4);

        let chunks: Vec<ChunkRequest> =
            (0..4).map(|i| ChunkRequest { index: i, start: i as u64 * 1024, len: 1024 }).collect();

        let results = fetch_chunks_parallel(&transport, &chunks, &policy, &progress, 4);
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.is_ok());
        }
        assert_eq!(progress.completed_chunks(), 4);
        assert_eq!(progress.downloaded_bytes(), 4096);
    }

    #[test]
    fn test_fetch_chunks_parallel_early_abort() {
        // Transport that fails on chunk 1
        struct FailTransport;
        impl ChunkedTransport for FailTransport {
            fn fetch_range(&self, start: u64, _len: u64) -> io::Result<Vec<u8>> {
                if start == 1024 {
                    Err(io::Error::new(io::ErrorKind::ConnectionReset, "boom"))
                } else {
                    Ok(vec![0u8; 1024])
                }
            }
            fn content_length(&self) -> io::Result<u64> {
                Ok(4096)
            }
            fn supports_range(&self) -> bool {
                true
            }
        }

        let transport = FailTransport;
        let policy = RetryPolicy { max_retries: 1, base_delay_ms: 1, max_delay_ms: 1, jitter_fraction: 0.0 };
        let progress = DownloadProgress::new(4096, 4);

        let chunks: Vec<ChunkRequest> =
            (0..4).map(|i| ChunkRequest { index: i, start: i as u64 * 1024, len: 1024 }).collect();

        let results = fetch_chunks_parallel(&transport, &chunks, &policy, &progress, 2);
        assert!(progress.is_failed());
        // At least one result should be an error
        assert!(results.iter().any(|r| r.is_err()));
    }
}
