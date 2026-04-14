// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cache-backed vector reader for remote datasets.
//!
//! `CachedVectorReader` wraps a `CachedChannel` to provide
//! `VectorReader<T>` access to remote vector files. Data is
//! downloaded on first access, verified against merkle hashes,
//! and served from the local cache on subsequent reads.
//!
//! When the cache is fully populated (all chunks verified), the
//! reader switches to mmap for zero-copy local access — no more
//! mutex locks or chunk validity checks.

use std::io;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use reqwest::blocking::Client;
use url::Url;

use crate::io::{IoError, VvecElement};
use crate::merkle::MerkleRef;
use crate::transport::http::HttpTransport;

use super::CachedChannel;

/// Default cache directory: `~/.cache/vectordata/`
pub fn default_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("vectordata")
    } else {
        PathBuf::from(".cache/vectordata")
    }
}

/// Resolve the cache directory for a dataset URL.
pub fn cache_dir_for_url(url: &Url, cache_root: &Path) -> PathBuf {
    let host = url.host_str().unwrap_or("local");
    let path = url.path().trim_start_matches('/');
    let dir = if let Some(pos) = path.rfind('/') {
        &path[..pos]
    } else {
        path
    };
    cache_root.join(host).join(dir)
}

/// A vector reader backed by a local cache with merkle verification.
///
/// Two modes:
/// - **Downloading**: reads go through `CachedChannel` (mutex + chunk check)
/// - **Complete**: all chunks verified → switches to mmap (zero-copy, no locks)
pub struct CachedVectorReader<T> {
    channel: CachedChannel,
    /// Lazily initialized mmap — set when all chunks are verified.
    mmap: OnceLock<Mmap>,
    dim: usize,
    count: usize,
    entry_size: usize,
    elem_size: usize,
    _phantom: PhantomData<T>,
}

impl<T: VvecElement> CachedVectorReader<T> {
    /// Open a cached reader for a remote xvec file.
    pub fn open(url: Url, elem_size: usize, cache_root: &Path) -> Result<Self, IoError> {
        let client = Client::new();

        // Fetch the .mref file
        let mref_url_str = format!("{}.mref", url.as_str());
        let mref_url = Url::parse(&mref_url_str)
            .map_err(|e| IoError::InvalidFormat(format!("invalid mref URL: {}", e)))?;

        let mref_resp = client.get(mref_url).send()?.error_for_status()
            .map_err(|e| IoError::Io(io::Error::new(io::ErrorKind::NotFound,
                format!("no .mref for {}: {}", url, e))))?;
        let mref_bytes = mref_resp.bytes()?;
        let reference = MerkleRef::from_bytes(&mref_bytes)?;

        let cache_dir = cache_dir_for_url(&url, cache_root);
        let filename = url.path_segments()
            .and_then(|s| s.last())
            .unwrap_or("data");

        let transport = HttpTransport::with_client(client, url.clone());
        let channel = CachedChannel::open(
            Box::new(transport),
            reference,
            &cache_dir,
            filename,
        )?;

        // Read dimension from first 4 bytes
        let header = channel.read(0, 4)?;
        let dim = (&header[..]).read_i32::<LittleEndian>()? as usize;
        if dim == 0 {
            return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * elem_size;
        let total_size = channel.content_size();
        let count = (total_size / entry_size as u64) as usize;

        let reader = Self {
            channel,
            mmap: OnceLock::new(),
            dim,
            count,
            entry_size,
            elem_size,
            _phantom: PhantomData,
        };

        // If already fully cached, initialize mmap immediately
        reader.try_init_mmap();

        Ok(reader)
    }

    /// Prebuffer the entire file into the local cache.
    pub fn prebuffer(&self) -> io::Result<()> {
        self.channel.prebuffer()?;
        self.try_init_mmap();
        Ok(())
    }

    /// Prebuffer with a progress callback.
    pub fn prebuffer_with_progress<F: FnMut(&crate::transport::DownloadProgress)>(
        &self,
        callback: F,
    ) -> io::Result<()> {
        self.channel.prebuffer_with_progress(callback)?;
        self.try_init_mmap();
        Ok(())
    }

    /// The cache directory path for this reader's data.
    pub fn cache_path(&self) -> &Path {
        self.channel.cache_path()
    }

    /// Whether the entire file is cached locally.
    pub fn is_complete(&self) -> bool {
        self.channel.is_complete()
    }

    /// Try to switch to mmap mode if all chunks are verified.
    fn try_init_mmap(&self) {
        if self.channel.is_complete() {
            let _ = self.mmap.get_or_init(|| {
                let file = std::fs::File::open(self.channel.cache_path())
                    .expect("cache file should be readable when complete");
                unsafe { Mmap::map(&file).expect("mmap should succeed on complete cache file") }
            });
        }
    }

    /// Read record bytes — mmap if complete, channel otherwise.
    fn read_record(&self, index: usize) -> Result<&[u8], IoError> {
        // This returns a reference only when mmap is available
        if let Some(mmap) = self.mmap.get() {
            let start = index * self.entry_size;
            let end = start + self.entry_size;
            if end > mmap.len() {
                return Err(IoError::OutOfBounds(index));
            }
            Ok(&mmap[start..end])
        } else {
            Err(IoError::OutOfBounds(usize::MAX)) // signal to use channel path
        }
    }
}

impl<T: VvecElement> crate::VectorReader<T> for CachedVectorReader<T> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        // Fast path: mmap (zero-copy, no locks)
        if let Some(mmap) = self.mmap.get() {
            let start = index * self.entry_size + 4; // skip dim header
            let end = start + self.dim * self.elem_size;
            if end > mmap.len() {
                return Err(IoError::OutOfBounds(index));
            }
            let data = &mmap[start..end];
            return Ok(data.chunks_exact(T::ELEM_SIZE)
                .map(|c| T::from_le_bytes(c))
                .collect());
        }

        // Slow path: channel read (downloads chunks if needed)
        let offset = (index * self.entry_size) as u64;
        let bytes = self.channel.read(offset, self.entry_size as u64)?;
        let data = &bytes[4..]; // skip dim header
        Ok(data.chunks_exact(T::ELEM_SIZE)
            .map(|c| T::from_le_bytes(c))
            .collect())
    }
}
