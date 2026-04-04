// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vector I/O: readers for fvec, ivec, mvec, dvec, bvec, and svec binary formats.
//!
//! Provides the [`VectorReader`] trait and two concrete implementations:
//!
//! - [`MmapVectorReader`] — memory-mapped local files (zero-copy where possible).
//! - [`HttpVectorReader`] — remote files accessed via HTTP Range requests.
//!
//! # Supported formats
//!
//! All formats share the same per-record layout: a 4-byte little-endian
//! `i32` dimension header followed by `dim` elements.
//!
//! | Format | Element type | Element size | Open method |
//! |--------|-------------|-------------|-------------|
//! | **fvec** | `f32` | 4 bytes | `open_fvec` |
//! | **ivec** | `i32` | 4 bytes | `open_ivec` |
//! | **mvec** | `f16` (half) | 2 bytes | `open_mvec` |
//! | **dvec** | `f64` | 8 bytes | `open_dvec` |
//! | **bvec** | `u8` | 1 byte | `open_bvec` |
//! | **svec** | `i16` | 2 bytes | `open_svec` |

use std::fs::File;
use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::path::Path;
use memmap2::Mmap;
use byteorder::{LittleEndian, ReadBytesExt};
use thiserror::Error;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, RANGE};
use url::Url;

/// Sentinel value for dimension when the file has zero records.
///
/// Dimensionality is undefined (moot) when cardinality is zero — there are
/// no records to derive a dimension from. Callers must check `count() > 0`
/// before relying on `dim()`. Using `usize::MAX` rather than 0 avoids
/// confusion with a hypothetical zero-dimensional vector.
pub const DIM_UNDEFINED: usize = usize::MAX;

/// Validate that a file's extension matches the expected format.
///
/// Returns `Ok(())` if the extension matches or the file has no extension
/// (for programmatic paths). Returns an error for mismatches.
fn validate_extension(path: &Path, expected: &str) -> Result<(), IoError> {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        // Accept both singular (e.g. "fvec") and plural (e.g. "fvecs") forms
        let plural = format!("{}s", expected);
        if ext != expected && ext != plural {
            return Err(IoError::InvalidFormat(format!(
                "file extension '.{}' does not match expected '.{}' for {}: {}",
                ext, expected, expected, path.display(),
            )));
        }
    }
    Ok(())
}

/// Errors that can occur during vector I/O operations.
#[derive(Error, Debug)]
pub enum IoError {
    /// Wrapper for standard IO errors.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Wrapper for HTTP errors from reqwest.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// Error indicating invalid file format or data.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    /// Error indicating an access out of valid bounds.
    #[error("Index out of bounds: {0}")]
    OutOfBounds(usize),
}

/// A trait for reading vectors from a data source.
///
/// Implementations handle the underlying storage details (e.g., file, network).
pub trait VectorReader<T>: Send + Sync {
    /// Returns the dimension of the vectors.
    fn dim(&self) -> usize;
    /// Returns the total number of vectors available.
    fn count(&self) -> usize;
    /// Retrieves the vector at the specified index.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;
}

/// A `VectorReader` backed by a memory-mapped file.
///
/// This implementation is efficient for local files as it relies on the OS page cache.
#[derive(Debug)]
pub struct MmapVectorReader<T> {
    mmap: Mmap,
    dim: usize,
    count: usize,
    entry_size: usize,
    #[allow(dead_code)]
    header_size: usize, // usually 4 bytes for dim
    _phantom: PhantomData<T>,
}

impl<T> MmapVectorReader<T> {
    /// Returns the byte size of each vector entry (header + data).
    pub fn entry_size(&self) -> usize {
        self.entry_size
    }

    /// Advise the kernel to prefetch vector data for indices `[start, end)`.
    ///
    /// Issues `madvise(MADV_WILLNEED)` on the byte range covering the
    /// requested vectors. This is non-blocking — the kernel begins
    /// asynchronous readahead.
    pub fn prefetch_range(&self, start: usize, end: usize) {
        let byte_start = start * self.entry_size;
        let byte_end = std::cmp::min(end * self.entry_size, self.mmap.len());
        let byte_len = byte_end.saturating_sub(byte_start);
        if byte_len > 0 {
            let _ = self.mmap.advise_range(memmap2::Advice::WillNeed, byte_start, byte_len);
        }
    }

    /// Advise the kernel that the entire mmap will be accessed sequentially.
    ///
    /// Issues `madvise(MADV_SEQUENTIAL)` on the full mapping. This enables
    /// aggressive kernel readahead and allows the kernel to free pages
    /// behind the access cursor.
    pub fn advise_sequential(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Sequential);
    }

    /// Advise the kernel that vector data for indices `[start, end)` is
    /// no longer needed.
    ///
    /// Issues `madvise(MADV_DONTNEED)` on the byte range, allowing the
    /// kernel to evict those pages from the page cache. Use this after
    /// completing a partition scan to reduce page cache pressure.
    pub fn release_range(&self, start: usize, end: usize) {
        let byte_start = start * self.entry_size;
        let byte_end = std::cmp::min(end * self.entry_size, self.mmap.len());
        let byte_len = byte_end.saturating_sub(byte_start);
        if byte_len > 0 {
            // memmap2's Advice enum doesn't expose DONTNEED; use libc directly.
            #[cfg(unix)]
            unsafe {
                libc::madvise(
                    self.mmap.as_ptr().add(byte_start) as *mut libc::c_void,
                    byte_len,
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    /// Force vector data for indices `[start, end)` into the page cache.
    ///
    /// Issues `madvise(MADV_WILLNEED)` to hint the kernel, then
    /// sequentially touches every page in the range via a volatile read.
    /// This **blocks** until all pages are resident.
    ///
    /// Call from a **background prefetch thread** that runs 2+ partitions
    /// ahead of compute. The blocking is intentional — it guarantees
    /// pages are warm when compute starts. Without the touch loop,
    /// `madvise` alone is a hint that the kernel may not honor in time,
    /// causing all compute threads to stall on page faults simultaneously.
    ///
    /// The `bytes_paged` counter is incremented after each page touch
    /// so callers can display I/O progress from another thread.
    pub fn prefetch_pages(
        &self,
        start: usize,
        end: usize,
        bytes_paged: Option<&std::sync::atomic::AtomicU64>,
    ) {
        let byte_start = start * self.entry_size;
        let byte_end = std::cmp::min(end * self.entry_size, self.mmap.len());
        let byte_len = byte_end.saturating_sub(byte_start);
        if byte_len == 0 {
            return;
        }

        // Hint the kernel for sequential readahead
        let _ = self.mmap.advise_range(memmap2::Advice::WillNeed, byte_start, byte_len);

        // Touch every page to guarantee residence. Sequential access
        // creates steady I/O instead of a thundering herd of random
        // page faults across all compute threads.
        let data = &self.mmap[byte_start..byte_end];
        let page_size = 4096;
        for offset in (0..data.len()).step_by(page_size) {
            unsafe { std::ptr::read_volatile(&data[offset]); }
            if let Some(counter) = bytes_paged {
                counter.fetch_add(page_size as u64, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

impl MmapVectorReader<f32> {
    /// Opens a local `.fvec` file for reading floating-point vectors.
    pub fn open_fvec(path: &Path) -> Result<Self, IoError> {
        validate_extension(path, "fvec")?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.is_empty() {
            return Ok(Self {
                mmap,
                dim: DIM_UNDEFINED,
                count: 0,
                entry_size: 0,
                header_size: 4,
                _phantom: PhantomData,
            });
        }

        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;

        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 4;
        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl MmapVectorReader<f32> {
    /// Returns a zero-copy slice into the mmap'd data for the vector at `index`.
    ///
    /// No allocation, no per-element parsing, no dimension validation.
    /// The caller must ensure `index < self.count()`.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[f32] {
        let data_start = index * self.entry_size + 4; // skip 4-byte dim header
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(data_start) as *const f32,
                self.dim,
            )
        }
    }
}

impl VectorReader<f32> for MmapVectorReader<f32> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<f32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        // Verify the dimension marker at start of record
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 4;
        
        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);
        
        for _ in 0..self.dim {
            vector.push(cursor.read_f32::<LittleEndian>()?);
        }

        Ok(vector)
    }
}

impl MmapVectorReader<i32> {
    /// Opens a local `.ivec` file for reading integer vectors (e.g., indices).
    pub fn open_ivec(path: &Path) -> Result<Self, IoError> {
        validate_extension(path, "ivec")?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.is_empty() {
            return Ok(Self {
                mmap, dim: DIM_UNDEFINED, count: 0, entry_size: 0,
                header_size: 4, _phantom: PhantomData,
            });
        }

        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;

        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 4;
        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl MmapVectorReader<i32> {
    /// Returns a zero-copy slice into the mmap'd data for the vector at `index`.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[i32] {
        let data_start = index * self.entry_size + 4;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(data_start) as *const i32,
                self.dim,
            )
        }
    }
}

impl VectorReader<i32> for MmapVectorReader<i32> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<i32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 4;
        
        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);
        
        for _ in 0..self.dim {
            vector.push(cursor.read_i32::<LittleEndian>()?);
        }

        Ok(vector)
    }
}

impl MmapVectorReader<half::f16> {
    /// Opens a local `.mvec` file for reading half-precision (f16) vectors.
    pub fn open_mvec(path: &Path) -> Result<Self, IoError> {
        validate_extension(path, "mvec")?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.is_empty() {
            return Ok(Self {
                mmap, dim: DIM_UNDEFINED, count: 0, entry_size: 0,
                header_size: 4, _phantom: PhantomData,
            });
        }

        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;

        if dim == 0 {
            return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 2;

        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl MmapVectorReader<half::f16> {
    /// Returns a zero-copy slice into the mmap'd data for the vector at `index`.
    ///
    /// No allocation, no per-element parsing, no dimension validation.
    /// The caller must ensure `index < self.count()`.
    ///
    /// Safe on little-endian architectures (x86/ARM) where the mmap bytes
    /// are already in the correct f16 bit pattern. `half::f16` is
    /// `#[repr(transparent)]` around `u16`.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[half::f16] {
        let data_start = index * self.entry_size + 4; // skip 4-byte dim header
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(data_start) as *const half::f16,
                self.dim,
            )
        }
    }
}

impl VectorReader<half::f16> for MmapVectorReader<half::f16> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<half::f16>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
            return Err(IoError::InvalidFormat(format!(
                "Record at index {} has mismatched dimension {}",
                index, dim
            )));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 2;

        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);

        for _ in 0..self.dim {
            vector.push(half::f16::from_bits(cursor.read_u16::<LittleEndian>()?));
        }

        Ok(vector)
    }
}

impl MmapVectorReader<f64> {
    /// Opens a local `.dvec` file for reading double-precision floating-point vectors.
    pub fn open_dvec(path: &Path) -> Result<Self, IoError> {
        validate_extension(path, "dvec")?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.is_empty() {
            return Ok(Self {
                mmap, dim: DIM_UNDEFINED, count: 0, entry_size: 0,
                header_size: 4, _phantom: PhantomData,
            });
        }

        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;

        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 8;

        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl MmapVectorReader<f64> {
    /// Returns a zero-copy slice into the mmap'd data for the vector at `index`.
    ///
    /// No allocation, no per-element parsing, no dimension validation.
    /// The caller must ensure `index < self.count()`.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[f64] {
        let data_start = index * self.entry_size + 4; // skip 4-byte dim header
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(data_start) as *const f64,
                self.dim,
            )
        }
    }
}

impl VectorReader<f64> for MmapVectorReader<f64> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<f64>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 8;

        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);

        for _ in 0..self.dim {
            vector.push(cursor.read_f64::<LittleEndian>()?);
        }

        Ok(vector)
    }
}

impl MmapVectorReader<u8> {
    /// Opens a local `.bvec` file for reading byte vectors.
    pub fn open_bvec(path: &Path) -> Result<Self, IoError> {
        validate_extension(path, "bvec")?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.is_empty() {
            return Ok(Self {
                mmap, dim: DIM_UNDEFINED, count: 0, entry_size: 0,
                header_size: 4, _phantom: PhantomData,
            });
        }

        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;

        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 1;

        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl MmapVectorReader<u8> {
    /// Returns a zero-copy slice into the mmap'd data for the vector at `index`.
    ///
    /// No allocation, no per-element parsing, no dimension validation.
    /// The caller must ensure `index < self.count()`.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[u8] {
        let data_start = index * self.entry_size + 4; // skip 4-byte dim header
        &self.mmap[data_start..data_start + self.dim]
    }
}

impl VectorReader<u8> for MmapVectorReader<u8> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<u8>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim;

        Ok(self.mmap[vector_start..vector_end].to_vec())
    }
}

impl MmapVectorReader<i16> {
    /// Opens a local `.svec` file for reading signed 16-bit integer vectors.
    pub fn open_svec(path: &Path) -> Result<Self, IoError> {
        validate_extension(path, "svec")?;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.is_empty() {
            return Ok(Self {
                mmap, dim: DIM_UNDEFINED, count: 0, entry_size: 0,
                header_size: 4, _phantom: PhantomData,
            });
        }

        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;

        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 2;

        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl MmapVectorReader<i16> {
    /// Returns a zero-copy slice into the mmap'd data for the vector at `index`.
    ///
    /// No allocation, no per-element parsing, no dimension validation.
    /// The caller must ensure `index < self.count()`.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[i16] {
        let data_start = index * self.entry_size + 4; // skip 4-byte dim header
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(data_start) as *const i16,
                self.dim,
            )
        }
    }
}

impl VectorReader<i16> for MmapVectorReader<i16> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<i16>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 2;

        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);

        for _ in 0..self.dim {
            vector.push(cursor.read_i16::<LittleEndian>()?);
        }

        Ok(vector)
    }
}

/// A VectorReader that reads data from an HTTP(S) URL using Range requests.
///
/// This reader expects the remote file to be formatted as a standard vector file
/// (dimension header followed by vectors). It performs minimal network requests:
/// 1. Initial GET/HEAD to determine dimension and file size.
/// 2. Range GET requests for specific vector reads.
#[derive(Debug)]
pub struct HttpVectorReader<T> {
    client: Client,
    url: Url,
    dim: usize,
    count: usize,
    entry_size: usize,
    #[allow(dead_code)]
    total_size: u64,
    _phantom: PhantomData<T>,
}

impl HttpVectorReader<f32> {
    /// Opens a floating-point vector file from a URL.
    pub fn open_fvec(url: Url) -> Result<Self, IoError> {
        let client = Client::new();
        
        // Read header (dimension)
        let resp = client.get(url.clone())
            .header(RANGE, "bytes=0-3")
            .send()?
            .error_for_status()?;
            
        let bytes = resp.bytes()?;
        if bytes.len() < 4 {
             return Err(IoError::InvalidFormat("File too short".into()));
        }
        let mut cursor = Cursor::new(bytes);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        // Get total size via HEAD
        let resp = client.head(url.clone()).send()?.error_for_status()?;
        let total_size = resp.headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| IoError::InvalidFormat("Missing Content-Length".into()))?;

        let entry_size = 4 + dim * 4;
        let count = (total_size / entry_size as u64) as usize;

        Ok(Self {
            client,
            url,
            dim,
            count,
            entry_size,
            total_size,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<f32> for HttpVectorReader<f32> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<f32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let end = start + self.entry_size - 1;

        let resp = self.client.get(self.url.clone())
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()?
            .error_for_status()?;

        let bytes = resp.bytes()?;
        let mut cursor = Cursor::new(bytes);
        
        // Read and check dimension
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let mut vector = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            vector.push(cursor.read_f32::<LittleEndian>()?);
        }
        Ok(vector)
    }
}

impl HttpVectorReader<i32> {
    /// Opens an integer vector file from a URL.
    pub fn open_ivec(url: Url) -> Result<Self, IoError> {
        let client = Client::new();
        
        let resp = client.get(url.clone())
            .header(RANGE, "bytes=0-3")
            .send()?
            .error_for_status()?;
            
        let bytes = resp.bytes()?;
        let mut cursor = Cursor::new(bytes);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        
        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let resp = client.head(url.clone()).send()?.error_for_status()?;
        let total_size = resp.headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| IoError::InvalidFormat("Missing Content-Length".into()))?;

        let entry_size = 4 + dim * 4;
        let count = (total_size / entry_size as u64) as usize;

        Ok(Self {
            client,
            url,
            dim,
            count,
            entry_size,
            total_size,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<i32> for HttpVectorReader<i32> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<i32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let end = start + self.entry_size - 1;

        let resp = self.client.get(self.url.clone())
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()?
            .error_for_status()?;

        let bytes = resp.bytes()?;
        let mut cursor = Cursor::new(bytes);

        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let mut vector = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            vector.push(cursor.read_i32::<LittleEndian>()?);
        }
        Ok(vector)
    }
}

impl HttpVectorReader<half::f16> {
    /// Opens a half-precision vector file from a URL.
    pub fn open_mvec(url: Url) -> Result<Self, IoError> {
        let client = Client::new();

        let resp = client.get(url.clone())
            .header(RANGE, "bytes=0-3")
            .send()?
            .error_for_status()?;

        let bytes = resp.bytes()?;
        if bytes.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }
        let mut cursor = Cursor::new(bytes);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim == 0 {
            return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let resp = client.head(url.clone()).send()?.error_for_status()?;
        let total_size = resp.headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| IoError::InvalidFormat("Missing Content-Length".into()))?;

        let entry_size = 4 + dim * 2; // 4 bytes dim header + dim * 2 bytes per f16
        let count = (total_size / entry_size as u64) as usize;

        Ok(Self {
            client,
            url,
            dim,
            count,
            entry_size,
            total_size,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<half::f16> for HttpVectorReader<half::f16> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<half::f16>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let end = start + self.entry_size - 1;

        let resp = self.client.get(self.url.clone())
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()?
            .error_for_status()?;

        let bytes = resp.bytes()?;
        let mut cursor = Cursor::new(bytes);

        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
            return Err(IoError::InvalidFormat(format!(
                "Record at index {} has mismatched dimension {}", index, dim
            )));
        }

        let mut vector = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            vector.push(half::f16::from_bits(cursor.read_u16::<LittleEndian>()?));
        }
        Ok(vector)
    }
}

