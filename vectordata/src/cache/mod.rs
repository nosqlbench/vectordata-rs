// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Cache-backed file channel with merkle verification.
//!
//! `CachedChannel` transparently downloads, verifies, and caches data from a
//! remote source using merkle tree integrity checking. Reads check local state
//! first; missing chunks are fetched on demand, verified against the reference
//! tree, and persisted to a local cache file.
//!
//! # Crash Recovery
//!
//! The `.mrkl` state file is the source of truth for what has been verified.
//! On restart, loading the state file resumes from the last checkpoint — only
//! unverified chunks are re-downloaded.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::merkle::{MerkleRef, MerkleState};
use crate::transport::{
    ChunkRequest, ChunkedTransport, DownloadProgress, RetryPolicy,
    fetch_chunks_parallel,
};

/// Default concurrency for parallel chunk downloads.
const DEFAULT_CONCURRENCY: usize = 8;

/// A file channel that transparently downloads, verifies, and caches data
/// from a remote source using merkle tree integrity checking.
pub struct CachedChannel {
    /// Local cache file (random-access read/write).
    cache_file: Mutex<File>,
    /// Path to the cache file (for reporting).
    cache_path: PathBuf,
    /// Merkle verification state (which chunks are valid).
    state: Mutex<MerkleState>,
    /// Path to the .mrkl state file.
    state_path: PathBuf,
    /// Remote data source.
    transport: Box<dyn ChunkedTransport>,
    /// Reference tree for hash verification.
    reference: MerkleRef,
    /// Retry policy for downloads.
    retry_policy: RetryPolicy,
    /// Download concurrency limit.
    concurrency: usize,
}

impl CachedChannel {
    /// Open a cached channel for a remote resource.
    ///
    /// - `transport`: the remote data source
    /// - `reference`: the merkle reference tree (.mref)
    /// - `cache_dir`: local directory for cache file and state
    /// - `name`: base name for cache files (e.g., "base_vectors.fvec")
    ///
    /// If a `.mrkl` state file exists in `cache_dir`, resumes from that
    /// checkpoint. Otherwise starts fresh.
    pub fn open(
        transport: Box<dyn ChunkedTransport>,
        reference: MerkleRef,
        cache_dir: &Path,
        name: &str,
    ) -> io::Result<Self> {
        fs::create_dir_all(cache_dir)?;

        let cache_path = cache_dir.join(name);
        let state_path = cache_dir.join(format!("{}.mrkl", name));

        // Open or create the cache file, pre-allocating to full size
        let total_size = reference.shape().total_content_size;
        let cache_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&cache_path)?;

        // Extend file to full size if needed (sparse file on most FSes)
        let current_len = cache_file.metadata()?.len();
        if current_len < total_size {
            cache_file.set_len(total_size)?;
        }

        // Load or create state
        let state = if state_path.exists() {
            MerkleState::load(&state_path)?
        } else {
            MerkleState::from_ref(&reference)
        };

        Ok(CachedChannel {
            cache_file: Mutex::new(cache_file),
            cache_path,
            state: Mutex::new(state),
            state_path,
            transport,
            reference,
            retry_policy: RetryPolicy::default(),
            concurrency: DEFAULT_CONCURRENCY,
        })
    }

    /// Set the retry policy.
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Set the download concurrency.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Read bytes from `[offset, offset+len)`, fetching and verifying chunks
    /// as needed.
    pub fn read(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        let shape = self.reference.shape();
        let end = offset + len;

        if end > shape.total_content_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "read range [{}, {}) exceeds content size {}",
                    offset, end, shape.total_content_size
                ),
            ));
        }

        // Determine which chunks overlap the requested range
        let first_chunk = (offset / shape.chunk_size) as u32;
        let last_chunk = ((end - 1) / shape.chunk_size) as u32;

        // Ensure all needed chunks are valid
        self.ensure_chunks_valid(first_chunk, last_chunk)?;

        // Read from the cache file
        let mut file = self.cache_file.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Ensure chunks in range `[first, last]` (inclusive) are downloaded and
    /// verified.
    fn ensure_chunks_valid(&self, first: u32, last: u32) -> io::Result<()> {
        // Collect missing chunks
        let missing: Vec<u32> = {
            let state = self.state.lock().unwrap();
            (first..=last).filter(|&i| !state.is_valid(i)).collect()
        };

        if missing.is_empty() {
            return Ok(());
        }

        let shape = self.reference.shape();

        // Build chunk requests
        let requests: Vec<ChunkRequest> = missing
            .iter()
            .map(|&i| ChunkRequest {
                index: i,
                start: shape.chunk_start(i),
                len: shape.chunk_len(i),
            })
            .collect();

        let total_bytes: u64 = requests.iter().map(|r| r.len).sum();
        let progress = DownloadProgress::new(total_bytes, requests.len() as u32);

        // Fetch chunks (parallel if multiple)
        let results = fetch_chunks_parallel(
            self.transport.as_ref(),
            &requests,
            &self.retry_policy,
            &progress,
            self.concurrency,
        );

        // Verify and write each chunk
        for result in results {
            let (chunk_index, data) = result?;
            self.verify_and_cache(chunk_index, &data)?;
        }

        Ok(())
    }

    /// Verify a chunk against the reference tree, write to cache, mark valid.
    fn verify_and_cache(&self, chunk_index: u32, data: &[u8]) -> io::Result<()> {
        // Verify against reference
        if !self.reference.verify_chunk(chunk_index, data) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "chunk {} failed integrity verification (SHA-256 mismatch)",
                    chunk_index
                ),
            ));
        }

        let shape = self.reference.shape();
        let offset = shape.chunk_start(chunk_index);

        // Write to cache file
        {
            let mut file = self.cache_file.lock().unwrap();
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(data)?;
        }

        // Mark valid and persist state
        {
            let mut state = self.state.lock().unwrap();
            state.mark_valid(chunk_index);
            state.save(&self.state_path)?;
        }

        Ok(())
    }

    /// Eagerly download and verify all unverified chunks.
    pub fn prebuffer(&self) -> io::Result<()> {
        self.prebuffer_with_progress(|_| {})
    }

    /// Eagerly download all unverified chunks with a progress callback.
    ///
    /// The callback is invoked after each batch of chunks is downloaded.
    pub fn prebuffer_with_progress<F: FnMut(&DownloadProgress)>(
        &self,
        mut callback: F,
    ) -> io::Result<()> {
        let missing = {
            let state = self.state.lock().unwrap();
            state.missing_chunks()
        };

        if missing.is_empty() {
            return Ok(());
        }

        let shape = self.reference.shape();

        let requests: Vec<ChunkRequest> = missing
            .iter()
            .map(|&i| ChunkRequest {
                index: i,
                start: shape.chunk_start(i),
                len: shape.chunk_len(i),
            })
            .collect();

        let total_bytes: u64 = requests.iter().map(|r| r.len).sum();
        let progress = DownloadProgress::new(total_bytes, requests.len() as u32);

        let results = fetch_chunks_parallel(
            self.transport.as_ref(),
            &requests,
            &self.retry_policy,
            &progress,
            self.concurrency,
        );

        for result in results {
            let (chunk_index, data) = result?;
            self.verify_and_cache(chunk_index, &data)?;
        }

        callback(&progress);
        Ok(())
    }

    /// Whether all chunks have been downloaded and verified.
    pub fn is_complete(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.is_complete()
    }

    /// Number of verified chunks.
    pub fn valid_count(&self) -> u32 {
        let state = self.state.lock().unwrap();
        state.valid_count()
    }

    /// Total number of chunks.
    pub fn total_chunks(&self) -> u32 {
        self.reference.shape().total_chunks
    }

    /// Path to the local cache file.
    pub fn cache_path(&self) -> &Path {
        &self.cache_path
    }

    /// Path to the merkle state file.
    pub fn state_path(&self) -> &Path {
        &self.state_path
    }
}

// CachedChannel holds Mutex<File> + Mutex<MerkleState>, both Send.
// The Mutexes ensure thread safety.
unsafe impl Send for CachedChannel {}
unsafe impl Sync for CachedChannel {}
