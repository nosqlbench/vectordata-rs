// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector I/O: readers for fvec, ivec, mvec, dvec, bvec, and svec binary formats.
//!
//! Provides the [`VectorReader`] trait and two concrete implementations:
//!
//! - `MmapVectorReader` — memory-mapped local files (zero-copy where possible).
//! - `HttpVectorReader` — remote files accessed via HTTP Range requests.
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

/// Validate that a file's extension is compatible with the expected format.
///
/// Accepts the expected extension, its plural form, and canonical aliases
/// (e.g., `.u8` and `.u8vec` are compatible with `.bvec`). Returns `Ok(())`
/// if compatible or if the file has no extension.
fn validate_extension(path: &Path, expected: &str) -> Result<(), IoError> {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let ext_lower = ext.to_lowercase();
        let expected_lower = expected.to_lowercase();

        // Direct match or plural
        if ext_lower == expected_lower || ext_lower == format!("{}s", expected_lower) {
            return Ok(());
        }

        // Check if both map to the same element size via the format system
        let ext_size = infer_elem_size(&ext_lower);
        let expected_size = infer_elem_size(&expected_lower);
        if ext_size > 0 && ext_size == expected_size {
            return Ok(());
        }

        return Err(IoError::InvalidFormat(format!(
            "file extension '.{}' does not match expected '.{}' for {}: {}",
            ext, expected, expected, path.display(),
        )));
    }
    Ok(())
}

/// Infer element size from an extension string (0 = unknown).
fn infer_elem_size(ext: &str) -> usize {
    match ext {
        "fvec" | "fvecs" | "f32vec" | "f32vecs" | "ivec" | "ivecs" | "i32vec" | "i32vecs"
        | "u32vec" | "u32vecs" | "fvvec" | "fvvecs" | "ivvec" | "ivvecs"
        | "f32vvec" | "f32vvecs" | "i32vvec" | "i32vvecs" | "u32vvec" | "u32vvecs"
        | "u32" | "i32" => 4,
        "dvec" | "dvecs" | "f64vec" | "f64vecs" | "dvvec" | "dvvecs"
        | "i64vec" | "i64vecs" | "u64vec" | "u64vecs"
        | "f64vvec" | "f64vvecs" | "i64vvec" | "i64vvecs" | "u64vvec" | "u64vvecs"
        | "u64" | "i64" => 8,
        "mvec" | "mvecs" | "f16vec" | "f16vecs" | "svec" | "svecs" | "i16vec" | "i16vecs"
        | "u16vec" | "u16vecs" | "mvvec" | "mvvecs" | "svvec" | "svvecs"
        | "f16vvec" | "f16vvecs" | "i16vvec" | "i16vvecs" | "u16vvec" | "u16vvecs"
        | "u16" | "i16" => 2,
        "bvec" | "bvecs" | "u8vec" | "u8vecs" | "i8vec" | "i8vecs"
        | "bvvec" | "bvvecs" | "u8vvec" | "u8vvecs" | "i8vvec" | "i8vvecs"
        | "u8" | "i8" => 1,
        _ => 0,
    }
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
    /// File has variable-length records — cannot use uniform-stride random access.
    /// Use `IndexedXvecReader` for random access on variable-length xvec files.
    #[error("Variable-length records: {0}")]
    VariableLengthRecords(String),
}

/// A trait for reading vectors from a data source.
///
/// Implementations handle the underlying storage details (e.g., file, network).
/// Assumes uniform record dimension (all records same length).
pub trait VectorReader<T>: Send + Sync {
    /// Returns the dimension of the vectors.
    fn dim(&self) -> usize;
    /// Returns the total number of vectors available.
    fn count(&self) -> usize;
    /// Retrieves the vector at the specified index.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;

    /// Try to access the vector at `index` as a borrowed slice
    /// without copying. Returns `None` when the underlying storage
    /// can't satisfy the request without materializing — e.g.,
    /// HTTP-backed readers that need to range-fetch on each call,
    /// or formats that need per-element decoding.
    ///
    /// The mmap-backed concrete readers in this crate override
    /// this to return `Some(&self.mmap[...])` — a true zero-copy
    /// view into the file's mapped pages. Hot-path consumers can
    /// branch on `Some` to avoid the `get(...)` allocation.
    ///
    /// The default returns `None`, so non-mmap implementations
    /// continue to work unchanged.
    fn get_slice(&self, _index: usize) -> Option<&[T]> {
        None
    }
}

/// Trait for types that can be decoded from little-endian bytes.
pub trait VvecElement: Copy + Send + Sync + 'static {
    /// Size of one element in bytes.
    const ELEM_SIZE: usize;
    /// Decode one element from a little-endian byte slice.
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl VvecElement for u8 {
    const ELEM_SIZE: usize = 1;
    fn from_le_bytes(bytes: &[u8]) -> Self { bytes[0] }
}
impl VvecElement for i8 {
    const ELEM_SIZE: usize = 1;
    fn from_le_bytes(bytes: &[u8]) -> Self { bytes[0] as i8 }
}
impl VvecElement for u16 {
    const ELEM_SIZE: usize = 2;
    fn from_le_bytes(bytes: &[u8]) -> Self { u16::from_le_bytes([bytes[0], bytes[1]]) }
}
impl VvecElement for i16 {
    const ELEM_SIZE: usize = 2;
    fn from_le_bytes(bytes: &[u8]) -> Self { i16::from_le_bytes([bytes[0], bytes[1]]) }
}
impl VvecElement for u32 {
    const ELEM_SIZE: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) }
}
impl VvecElement for i32 {
    const ELEM_SIZE: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) }
}
impl VvecElement for f32 {
    const ELEM_SIZE: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) }
}
impl VvecElement for u64 {
    const ELEM_SIZE: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]) }
}
impl VvecElement for i64 {
    const ELEM_SIZE: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { i64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]) }
}
impl VvecElement for f64 {
    const ELEM_SIZE: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { f64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]) }
}
impl VvecElement for half::f16 {
    const ELEM_SIZE: usize = 2;
    fn from_le_bytes(bytes: &[u8]) -> Self { half::f16::from_le_bytes([bytes[0], bytes[1]]) }
}

/// A trait for reading variable-length vector records.
///
/// Unlike `VectorReader`, each record may have a different dimension.
/// Implementations handle local (mmap + offset index) and remote
/// (HTTP Range + index) access transparently.
///
/// The type parameter `T` determines the element type (e.g., `i32`,
/// `f32`, `u8`). Use [`open_vvec`] to get a type-erased reader, or
/// call the typed methods on `IndexedXvecReader` / `HttpIndexedXvecReader`
/// directly.
pub trait VvecReader<T: VvecElement>: Send + Sync {
    /// Total number of records in the file.
    fn count(&self) -> usize;
    /// Dimension of the record at the given ordinal.
    fn dim_at(&self, index: usize) -> Result<usize, IoError>;
    /// Read the raw bytes of the record data (past the dim header).
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError>;

    /// Read the record at the given ordinal as a Vec of typed elements.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        let bytes = self.get_bytes(index)?;
        Ok(bytes.chunks_exact(T::ELEM_SIZE)
            .map(|c| T::from_le_bytes(c))
            .collect())
    }
}

impl<T: VvecElement> VvecReader<T> for IndexedXvecReader {
    fn count(&self) -> usize { self.count() }
    fn dim_at(&self, index: usize) -> Result<usize, IoError> { self.dim_at(index) }
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        self.get_raw(index).map(|s| s.to_vec())
    }
}

impl<T: VvecElement> VvecReader<T> for HttpIndexedXvecReader {
    fn count(&self) -> usize { self.count() }
    fn dim_at(&self, index: usize) -> Result<usize, IoError> { self.dim_at(index) }
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        self.get_bytes(index)
    }
}

/// Open a variable-length vector file (vvec) for typed random access.
///
/// Transparently handles:
/// - **Local files**: uses `IndexedXvecReader` (mmap + offset index)
/// - **HTTP URLs**: uses `HttpIndexedXvecReader` (fetches index, Range requests)
///
/// The element size is inferred from the file extension. The companion
/// `IDXFOR__` index file must exist (created automatically at write
/// time by pipeline steps, or by `IndexedXvecReader::open`).
///
/// ```no_run
/// use vectordata::io::open_vvec;
///
/// let reader = open_vvec::<i32>("profiles/default/metadata_indices.ivvec").unwrap();
/// println!("{} records", reader.count());
/// let record = reader.get(42).unwrap();
///
/// let reader = open_vvec::<u8>("https://example.com/data.u8vvec").unwrap();
/// let bytes = reader.get(0).unwrap();
/// ```
pub fn open_vvec<T: VvecElement>(path_or_url: &str) -> Result<Box<dyn VvecReader<T>>, IoError> {
    // Infer element size from extension
    let name = if let Some(pos) = path_or_url.rfind('/') {
        &path_or_url[pos+1..]
    } else {
        path_or_url
    };
    let ext = name.rsplit('.').next().unwrap_or("");
    let elem_size = infer_vvec_elem_size(ext)?;

    if elem_size != T::ELEM_SIZE {
        return Err(IoError::InvalidFormat(format!(
            "extension '.{}' implies {}-byte elements but type {} requires {}-byte elements",
            ext, elem_size, std::any::type_name::<T>(), T::ELEM_SIZE)));
    }

    if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
        let url = Url::parse(path_or_url)
            .map_err(|e| IoError::InvalidFormat(format!("invalid URL: {}", e)))?;
        let reader = HttpIndexedXvecReader::open(url, elem_size)?;
        Ok(Box::new(reader))
    } else {
        let path = Path::new(path_or_url);
        let reader = IndexedXvecReader::open(path, elem_size)?;
        Ok(Box::new(reader))
    }
}

/// Open a variable-length vector file without compile-time type checking.
///
/// Returns a reader where the caller uses `get_bytes()` and decodes
/// manually, or calls the typed `get()` with matching element size.
pub fn open_vvec_untyped(path_or_url: &str) -> Result<Box<dyn VvecReader<u8>>, IoError> {
    let name = if let Some(pos) = path_or_url.rfind('/') {
        &path_or_url[pos+1..]
    } else {
        path_or_url
    };
    let ext = name.rsplit('.').next().unwrap_or("");
    let elem_size = infer_vvec_elem_size(ext)?;

    if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
        let url = Url::parse(path_or_url)
            .map_err(|e| IoError::InvalidFormat(format!("invalid URL: {}", e)))?;
        let reader = HttpIndexedXvecReader::open(url, elem_size)?;
        Ok(Box::new(reader))
    } else {
        let path = Path::new(path_or_url);
        let reader = IndexedXvecReader::open(path, elem_size)?;
        Ok(Box::new(reader))
    }
}

/// Sealed trait enabling `open_vec<T>` dispatch.
/// Implemented for every supported element type.
pub trait OpenableElement: VvecElement {
    /// Open a local uniform xvec file as a boxed VectorReader.
    fn open_local(path: &Path) -> Result<Box<dyn VectorReader<Self>>, IoError>;
    /// Open a remote uniform xvec file as a boxed VectorReader.
    fn open_remote(url: Url) -> Result<Box<dyn VectorReader<Self>>, IoError>;
}

macro_rules! impl_openable {
    ($t:ty) => {
        impl OpenableElement for $t {
            fn open_local(path: &Path) -> Result<Box<dyn VectorReader<Self>>, IoError> {
                let reader = MmapVectorReader::<$t>::open_xvec_generic(path, <$t as VvecElement>::ELEM_SIZE)?;
                Ok(Box::new(reader))
            }
            fn open_remote(url: Url) -> Result<Box<dyn VectorReader<Self>>, IoError> {
                // Try cache-backed reader first (downloads + caches with merkle verification).
                // Falls back to direct HTTP if no .mref exists for this file.
                let cache_root = crate::cache::reader::default_cache_dir();
                match crate::cache::reader::CachedVectorReader::<$t>::open(
                    url.clone(), <$t as VvecElement>::ELEM_SIZE, &cache_root,
                ) {
                    Ok(reader) => {
                        log::debug!("opened cached reader for {}", url);
                        Ok(Box::new(reader))
                    }
                    Err(_) => {
                        // No .mref available — fall back to direct HTTP Range requests
                        log::debug!("no .mref for {}, using direct HTTP", url);
                        let reader = HttpVectorReader::<$t>::open_xvec_generic(url, <$t as VvecElement>::ELEM_SIZE)?;
                        Ok(Box::new(reader))
                    }
                }
            }
        }
    };
}

impl_openable!(f32);
impl_openable!(f64);
impl_openable!(half::f16);
impl_openable!(i32);
impl_openable!(i16);
impl_openable!(u8);
impl_openable!(i8);
impl_openable!(u16);
impl_openable!(u32);
impl_openable!(u64);
impl_openable!(i64);

/// Open a uniform vector file for typed random access.
///
/// Transparently handles local files (mmap) and HTTP URLs (Range requests).
/// The type parameter `T` determines the element type and must match the
/// file extension.
///
/// Supported types: `f32`, `f64`, `half::f16`, `i32`, `i16`, `i8`,
/// `u8`, `u16`, `u32`, `u64`, `i64`.
///
/// ```no_run
/// use vectordata::io::{open_vec, VectorReader};
///
/// // Local
/// let reader = open_vec::<f32>("base_vectors.fvec").unwrap();
/// let v = reader.get(0).unwrap();
///
/// // Remote — same call
/// let reader = open_vec::<f32>("https://example.com/base.fvec").unwrap();
/// let v = reader.get(0).unwrap();
/// ```
pub fn open_vec<T: OpenableElement>(path_or_url: &str) -> Result<Box<dyn VectorReader<T>>, IoError> {
    if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
        let url = Url::parse(path_or_url)
            .map_err(|e| IoError::InvalidFormat(format!("invalid URL: {}", e)))?;
        T::open_remote(url)
    } else {
        T::open_local(Path::new(path_or_url))
    }
}

// Generic VectorReader impl for HttpVectorReader via get_record_bytes.
// Replaces the per-type specialized impls.
impl<T: VvecElement> VectorReader<T> for HttpVectorReader<T> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        let bytes = self.get_record_bytes(index)?;
        if bytes.len() < 4 {
            return Err(IoError::InvalidFormat("record too short".into()));
        }
        let data = &bytes[4..];
        Ok(data.chunks_exact(T::ELEM_SIZE)
            .map(|c| T::from_le_bytes(c))
            .collect())
    }
}

fn infer_vvec_elem_size(ext: &str) -> Result<usize, IoError> {
    match ext.to_lowercase().as_str() {
        "ivvec" | "ivvecs" | "i32vvec" | "i32vvecs" |
        "fvvec" | "fvvecs" | "f32vvec" | "f32vvecs" |
        "u32vvec" | "u32vvecs" => Ok(4),
        "dvvec" | "dvvecs" | "f64vvec" | "f64vvecs" |
        "i64vvec" | "i64vvecs" | "u64vvec" | "u64vvecs" => Ok(8),
        "svvec" | "svvecs" | "i16vvec" | "i16vvecs" |
        "mvvec" | "mvvecs" | "f16vvec" | "f16vvecs" |
        "u16vvec" | "u16vvecs" => Ok(2),
        "bvvec" | "bvvecs" | "u8vvec" | "u8vvecs" |
        "i8vvec" | "i8vvecs" => Ok(1),
        // Legacy: .ivec may be variable-length
        "ivec" | "ivecs" => Ok(4),
        _ => Err(IoError::InvalidFormat(format!(
            "cannot infer element size from extension '.{}'", ext))),
    }
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

    /// Advise the kernel that the mmap will be accessed in essentially
    /// random order (no useful readahead). Issues
    /// `madvise(MADV_RANDOM)` on the full mapping.
    ///
    /// Important for *small-payload-per-vector* read patterns (e.g.,
    /// reading just the leading prefix of each vector for sort-key
    /// construction): without this, the kernel's default readahead
    /// fetches ~256 KB on each access, mapping in 64× more pages than
    /// the read actually needs and inflating process RSS proportionally.
    /// On a multi-TB source file with parallel readers, that
    /// over-fetch is the dominant cause of "RSS over ceiling — aborting
    /// to prevent system lockup."
    pub fn advise_random(&self) {
        let _ = self.mmap.advise(memmap2::Advice::Random);
    }

    /// Advise the kernel that vector data for indices `[start, end)` is
    /// no longer needed.
    ///
    /// Issues `madvise(MADV_DONTNEED)` on the byte range, allowing the
    /// kernel to drop those pages from the process's page tables (so
    /// they no longer count against RSS) and reuse them under memory
    /// pressure. Use this after completing a partition scan to reduce
    /// resident-set growth on streaming reads.
    ///
    /// Requires page-aligned start and length. The vector entry size
    /// is generally not a multiple of the page size (e.g., 1540 bytes
    /// at dim=384 fvec), so we align the start UP and the length DOWN
    /// to the nearest page. Bytes in the sub-page slack at the
    /// boundaries stay mapped, but it's at most one page on each side
    /// — negligible compared to the megabyte-or-larger windows
    /// typical callers pass in. Without alignment, madvise rejects
    /// the call with `EINVAL` silently and resident size keeps
    /// growing — see verify dataset-knnutils on TB-scale base files.
    pub fn release_range(&self, start: usize, end: usize) {
        let byte_start = start * self.entry_size;
        let byte_end = std::cmp::min(end * self.entry_size, self.mmap.len());
        if byte_start >= byte_end { return; }

        #[cfg(unix)]
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            if page == 0 { return; }
            // Align start up, end down.
            let aligned_start = (byte_start + page - 1) & !(page - 1);
            let aligned_end = byte_end & !(page - 1);
            if aligned_end <= aligned_start { return; }
            let len = aligned_end - aligned_start;
            libc::madvise(
                self.mmap.as_ptr().add(aligned_start) as *mut libc::c_void,
                len,
                libc::MADV_DONTNEED,
            );
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

/// Streaming-scan resource manager for `MmapVectorReader`.
///
/// The problem this solves: a sequential mmap scan over a TB-scale
/// file accumulates pages in process RSS once they're touched. The
/// kernel only reclaims under memory pressure, so on RAM-rich
/// machines RSS climbs to the file size — well past any sensible
/// resource ceiling — before any reclaim happens.
///
/// `MADV_SEQUENTIAL` alone is insufficient: it lets the kernel free
/// pages behind the cursor when convenient, but doesn't force it.
/// `MADV_DONTNEED` is the active form, but must be issued
/// page-aligned (handled by [`MmapVectorReader::release_range`]).
///
/// `StreamReclaim` wraps the pattern: advise sequential up front,
/// release a fixed-byte trailing window each time the cursor
/// crosses a window boundary, and release any remainder on drop.
///
/// Usage:
/// ```ignore
/// let mut rec = StreamReclaim::new(&base_reader, start, end);
/// for i in start..end {
///     let v = base_reader.get_slice(i);
///     // ... do work ...
///     rec.advance(i);
/// }
/// // drop releases the trailing tail
/// ```
pub struct StreamReclaim<'a, T> {
    reader: &'a MmapVectorReader<T>,
    last_released: usize,
    end: usize,
    reclaim_window_records: usize,
}

impl<'a, T> StreamReclaim<'a, T> {
    /// Default 256 MiB reclaim windows. A reasonable trade-off:
    /// large enough that the per-window `madvise` overhead is
    /// negligible, small enough that resident-set growth is bounded.
    pub fn new(reader: &'a MmapVectorReader<T>, start: usize, end: usize) -> Self {
        Self::with_reclaim_bytes(reader, start, end, 256 * 1024 * 1024)
    }

    pub fn with_reclaim_bytes(
        reader: &'a MmapVectorReader<T>,
        start: usize,
        end: usize,
        reclaim_bytes: usize,
    ) -> Self {
        reader.advise_sequential();
        let entry_size = reader.entry_size().max(1);
        let reclaim_window_records = (reclaim_bytes / entry_size).max(1024);
        Self {
            reader,
            last_released: start,
            end,
            reclaim_window_records,
        }
    }

    /// Call after touching index `i`. Releases the trailing window
    /// when `i` crosses a window boundary. Cheap when the boundary
    /// hasn't been crossed yet (one comparison + branch).
    #[inline]
    pub fn advance(&mut self, i: usize) {
        if i >= self.last_released + self.reclaim_window_records {
            self.reader.release_range(
                self.last_released,
                self.last_released + self.reclaim_window_records,
            );
            self.last_released += self.reclaim_window_records;
        }
    }
}

impl<'a, T> Drop for StreamReclaim<'a, T> {
    fn drop(&mut self) {
        if self.last_released < self.end {
            self.reader.release_range(self.last_released, self.end);
        }
    }
}

impl<T: Copy + 'static> MmapVectorReader<T> {
    /// Generic constructor for any uniform xvec file.
    ///
    /// `elem_size` is the byte width of each element (e.g., 4 for f32/i32).
    /// The caller is responsible for ensuring `T` matches `elem_size`.
    fn open_xvec_generic(path: &Path, elem_size: usize) -> Result<Self, IoError> {
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

        let entry_size = 4 + dim * elem_size;

        if mmap.len() % entry_size != 0 {
            return Err(IoError::VariableLengthRecords(format!(
                "{}: file size {} is not a multiple of stride {} (dim={}). \
                 Use IndexedXvecReader for variable-length files.",
                path.display(), mmap.len(), entry_size, dim,
            )));
        }

        let count = mmap.len() / entry_size;

        Ok(Self { mmap, dim, count, entry_size, header_size: 4, _phantom: PhantomData })
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
        let dim_i32 = cursor.read_i32::<LittleEndian>()?;
        if dim_i32 <= 0 {
             return Err(IoError::InvalidFormat(
                format!("invalid dimension {} in {}", dim_i32, path.display())));
        }
        let dim = dim_i32 as usize;

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

    fn get_slice(&self, index: usize) -> Option<&[f32]> {
        if index >= self.count { return None; }
        Some(MmapVectorReader::<f32>::get_slice(self, index))
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
        let dim_i32 = cursor.read_i32::<LittleEndian>()?;
        if dim_i32 <= 0 {
             return Err(IoError::InvalidFormat(
                format!("invalid dimension {} in {}", dim_i32, path.display())));
        }
        let dim = dim_i32 as usize;

        let entry_size = 4 + dim * 4;

        if mmap.len() % entry_size != 0 {
            return Err(IoError::VariableLengthRecords(format!(
                "{}: file size {} is not a multiple of stride {} (dim={}). \
                 Use IndexedXvecReader for variable-length ivec files.",
                path.display(), mmap.len(), entry_size, dim,
            )));
        }

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

    fn get_slice(&self, index: usize) -> Option<&[i32]> {
        if index >= self.count { return None; }
        Some(MmapVectorReader::<i32>::get_slice(self, index))
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

    fn get_slice(&self, index: usize) -> Option<&[half::f16]> {
        if index >= self.count { return None; }
        Some(MmapVectorReader::<half::f16>::get_slice(self, index))
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

    fn get_slice(&self, index: usize) -> Option<&[f64]> {
        if index >= self.count { return None; }
        Some(MmapVectorReader::<f64>::get_slice(self, index))
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

    fn get_slice(&self, index: usize) -> Option<&[u8]> {
        if index >= self.count { return None; }
        Some(MmapVectorReader::<u8>::get_slice(self, index))
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

    fn get_slice(&self, index: usize) -> Option<&[i16]> {
        if index >= self.count { return None; }
        Some(MmapVectorReader::<i16>::get_slice(self, index))
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

// ── Additional VectorReader impls for MmapVectorReader ─────────────
// Types not covered by the zero-copy specialized impls above.

macro_rules! impl_mmap_vector_reader_generic {
    ($t:ty, $elem_size:expr) => {
        impl VectorReader<$t> for MmapVectorReader<$t> {
            fn dim(&self) -> usize { self.dim }
            fn count(&self) -> usize { self.count }

            fn get(&self, index: usize) -> Result<Vec<$t>, IoError> {
                if index >= self.count {
                    return Err(IoError::OutOfBounds(index));
                }
                let start = index * self.entry_size + 4; // skip dim header
                let end = start + self.dim * $elem_size;
                let data = &self.mmap[start..end];
                Ok(data.chunks_exact($elem_size)
                    .map(|c| <$t as VvecElement>::from_le_bytes(c))
                    .collect())
            }
        }
    };
}

impl_mmap_vector_reader_generic!(i8, 1);
impl_mmap_vector_reader_generic!(u16, 2);
impl_mmap_vector_reader_generic!(u32, 4);
impl_mmap_vector_reader_generic!(u64, 8);
impl_mmap_vector_reader_generic!(i64, 8);

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

impl<T: Copy + 'static> HttpVectorReader<T> {
    /// Generic constructor for any uniform xvec file over HTTP.
    fn open_xvec_generic(url: Url, elem_size: usize) -> Result<Self, IoError> {
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

        let entry_size = 4 + dim * elem_size;
        let count = (total_size / entry_size as u64) as usize;

        Ok(Self { client, url, dim, count, entry_size, total_size, _phantom: PhantomData })
    }

    /// Read raw record bytes at the given index (dim header + data).
    fn get_record_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }
        let start = index * self.entry_size;
        let end = start + self.entry_size - 1;

        let resp = self.client.get(self.url.clone())
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()?
            .error_for_status()?;

        Ok(resp.bytes()?.to_vec())
    }
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

// Legacy convenience constructors for HttpVectorReader.
// The generic open_xvec_generic handles all types; these are kept for
// backward compatibility with existing call sites.
impl HttpVectorReader<i32> {
    /// Opens an integer vector file from a URL.
    pub fn open_ivec(url: Url) -> Result<Self, IoError> {
        Self::open_xvec_generic(url, 4)
    }
}
impl HttpVectorReader<half::f16> {
    /// Opens a half-precision vector file from a URL.
    pub fn open_mvec(url: Url) -> Result<Self, IoError> {
        Self::open_xvec_generic(url, 2)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// IndexedXvecReader — random access on variable-length xvec files
// ═══════════════════════════════════════════════════════════════════════

/// Random-access reader for variable-length xvec files.
///
/// On first open, walks the file to build an offset index mapping
/// ordinal → byte offset. The index is persisted as a sibling file
/// named `IDXFOR__<original_name>.<i32|i64>` (flat-packed offsets).
/// Subsequent opens reuse the cached index if it is newer than the
/// data file.
///
/// Element size is determined by the xvec format (4 for ivec, etc.).
pub struct IndexedXvecReader {
    mmap: Mmap,
    offsets: Vec<u64>,
    elem_size: usize,
}

impl IndexedXvecReader {
    /// Open a variable-length xvec file with offset indexing.
    ///
    /// - Checks for an existing `IDXFOR__<name>.<i32|i64>` index file.
    /// - If the index is missing or stale (older than the data file),
    ///   walks the data file to build a new index and writes it.
    /// - Returns a reader with O(1) random access by ordinal.
    pub fn open(path: &Path, elem_size: usize) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let file_size = mmap.len() as u64;

        // Check for cached index
        let index_path = index_path_for(path, file_size);
        let offsets = if let Some(cached) = load_index(&index_path, path)? {
            cached
        } else {
            // Build index by walking records
            let offsets = build_offset_index(&mmap, elem_size)?;
            // Best-effort write — index is a cache, not critical
            let _ = write_index(&index_path, &offsets, file_size);
            offsets
        };

        Ok(IndexedXvecReader { mmap, offsets, elem_size })
    }

    /// Open a variable-length ivec file (element size = 4).
    pub fn open_ivec(path: &Path) -> Result<Self, IoError> {
        Self::open(path, 4)
    }

    /// Number of records in the file.
    pub fn count(&self) -> usize {
        self.offsets.len()
    }

    /// Read the dimension of record at the given ordinal.
    pub fn dim_at(&self, index: usize) -> Result<usize, IoError> {
        if index >= self.offsets.len() {
            return Err(IoError::OutOfBounds(index));
        }
        let offset = self.offsets[index] as usize;
        if offset + 4 > self.mmap.len() {
            return Err(IoError::InvalidFormat("offset beyond file end".into()));
        }
        let dim = i32::from_le_bytes([
            self.mmap[offset], self.mmap[offset+1],
            self.mmap[offset+2], self.mmap[offset+3],
        ]) as usize;
        Ok(dim)
    }

    /// Read record at the given ordinal as a `Vec<i32>`.
    pub fn get_i32(&self, index: usize) -> Result<Vec<i32>, IoError> {
        if index >= self.offsets.len() {
            return Err(IoError::OutOfBounds(index));
        }
        let offset = self.offsets[index] as usize;
        if offset + 4 > self.mmap.len() {
            return Err(IoError::InvalidFormat("offset beyond file end".into()));
        }
        let dim = i32::from_le_bytes([
            self.mmap[offset], self.mmap[offset+1],
            self.mmap[offset+2], self.mmap[offset+3],
        ]) as usize;
        let data_start = offset + 4;
        let data_end = data_start + dim * self.elem_size;
        if data_end > self.mmap.len() {
            return Err(IoError::InvalidFormat(format!(
                "record {} truncated: need {} bytes at offset {}", index, dim * self.elem_size, data_start)));
        }
        let bytes = &self.mmap[data_start..data_end];
        let vals: Vec<i32> = bytes.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(vals)
    }

    /// Zero-copy slice access to record data (past the dim header).
    /// Returns the raw bytes of the record data.
    pub fn get_raw(&self, index: usize) -> Result<&[u8], IoError> {
        if index >= self.offsets.len() {
            return Err(IoError::OutOfBounds(index));
        }
        let offset = self.offsets[index] as usize;
        if offset + 4 > self.mmap.len() {
            return Err(IoError::InvalidFormat("offset beyond file end".into()));
        }
        let dim = i32::from_le_bytes([
            self.mmap[offset], self.mmap[offset+1],
            self.mmap[offset+2], self.mmap[offset+3],
        ]) as usize;
        let data_start = offset + 4;
        let data_end = data_start + dim * self.elem_size;
        if data_end > self.mmap.len() {
            return Err(IoError::InvalidFormat("record truncated".into()));
        }
        Ok(&self.mmap[data_start..data_end])
    }
}

/// Compute the index file path for a given data file.
/// Uses `.i32` for files up to 2GB, `.i64` for larger.
/// Remove any existing IDXFOR__ index files for a data file.
///
/// Call this before rewriting a vvec file to ensure stale indices
/// are cleaned up. The index will be rebuilt by `IndexedXvecReader::open`.
pub fn remove_vvec_index(data_path: &Path) {
    let parent = data_path.parent().unwrap_or(Path::new("."));
    let name = data_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if name.is_empty() { return; }
    for ext in &["i32", "i64"] {
        let idx = parent.join(format!("IDXFOR__{}.{}", name, ext));
        if idx.exists() {
            let _ = std::fs::remove_file(&idx);
        }
    }
}

fn index_path_for(data_path: &Path, file_size: u64) -> std::path::PathBuf {
    let parent = data_path.parent().unwrap_or(Path::new("."));
    let name = data_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("data");
    let ext = if file_size <= i32::MAX as u64 { "i32" } else { "i64" };
    parent.join(format!("IDXFOR__{}.{}", name, ext))
}

/// Load a cached offset index if it exists and is fresh.
fn load_index(index_path: &Path, data_path: &Path) -> Result<Option<Vec<u64>>, IoError> {
    if !index_path.is_file() {
        return Ok(None);
    }

    // Check freshness: index must be newer than data file
    let data_mtime = std::fs::metadata(data_path)?
        .modified().map_err(|e| IoError::InvalidFormat(format!("mtime: {}", e)))?;
    let index_mtime = std::fs::metadata(index_path)?
        .modified().map_err(|e| IoError::InvalidFormat(format!("mtime: {}", e)))?;

    if index_mtime < data_mtime {
        return Ok(None); // stale
    }

    let data = std::fs::read(index_path)?;
    let ext = index_path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let offsets = match ext {
        "i32" => {
            data.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64)
                .collect()
        }
        "i64" => {
            data.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u64)
                .collect()
        }
        _ => return Ok(None),
    };

    Ok(Some(offsets))
}

/// Build an offset index by walking all record dim headers.
fn build_offset_index(mmap: &[u8], elem_size: usize) -> Result<Vec<u64>, IoError> {
    let file_size = mmap.len() as u64;
    let mut offsets = Vec::new();
    let mut offset: u64 = 0;

    while offset + 4 <= file_size {
        offsets.push(offset);
        let o = offset as usize;
        let dim = i32::from_le_bytes([mmap[o], mmap[o+1], mmap[o+2], mmap[o+3]]);
        // Negative dimension is invalid. Zero dimension is valid for
        // variable-length (vvec) files — it represents an empty record
        // (e.g., a predicate with no matching vectors).
        if dim < 0 {
            return Err(IoError::InvalidFormat(format!(
                "negative dimension {} at offset {}", dim, offset)));
        }
        offset += 4 + dim as u64 * elem_size as u64;
    }

    if offset != file_size {
        return Err(IoError::InvalidFormat(format!(
            "file does not end at a record boundary: {} bytes remaining at offset {}",
            file_size - offset, offset)));
    }

    Ok(offsets)
}

/// Write an offset index to disk.
fn write_index(index_path: &Path, offsets: &[u64], file_size: u64) -> Result<(), IoError> {
    use std::io::Write;

    let mut f = File::create(index_path)?;

    if file_size <= i32::MAX as u64 {
        for &off in offsets {
            f.write_all(&(off as i32).to_le_bytes())?;
        }
    } else {
        for &off in offsets {
            f.write_all(&(off as i64).to_le_bytes())?;
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// HttpIndexedXvecReader — remote random access on variable-length xvec
// ═══════════════════════════════════════════════════════════════════════

/// Remote random-access reader for variable-length xvec files.
///
/// Fetches the companion `IDXFOR__<name>.<i32|i64>` index file via HTTP
/// (small — one offset per record), then uses the offsets to issue
/// Range requests for individual records from the data file.
pub struct HttpIndexedXvecReader {
    client: Client,
    data_url: Url,
    offsets: Vec<u64>,
    #[allow(dead_code)]
    data_size: u64,
    elem_size: usize,
}

impl HttpIndexedXvecReader {
    /// Open a remote variable-length xvec file.
    ///
    /// `data_url` points to the data file (e.g. `metadata_indices.ivvec`).
    /// The index file URL is derived by replacing the filename with
    /// `IDXFOR__<name>.<i32|i64>`.
    ///
    /// Tries `.i64` first (large files), then `.i32`.
    pub fn open(data_url: Url, elem_size: usize) -> Result<Self, IoError> {
        let client = Client::new();

        // Determine data file size
        let resp = client.head(data_url.clone()).send()?.error_for_status()?;
        let data_size = resp.headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| IoError::InvalidFormat("Missing Content-Length on data file".into()))?;

        // Derive index URL: same directory, IDXFOR__<name>.<ext>
        let data_name = data_url.path_segments()
            .and_then(|s| s.last())
            .unwrap_or("")
            .to_string();

        let base_url = {
            let mut u = data_url.clone();
            u.path_segments_mut()
                .map_err(|_| IoError::InvalidFormat("cannot modify URL path".into()))?
                .pop();
            u
        };

        // Try i64 index first, then i32
        let index_candidates = [
            format!("{}/IDXFOR__{}.i64", base_url.as_str().trim_end_matches('/'), data_name),
            format!("{}/IDXFOR__{}.i32", base_url.as_str().trim_end_matches('/'), data_name),
        ];

        let mut offsets: Option<Vec<u64>> = None;
        for candidate in &index_candidates {
            let url = match Url::parse(candidate) {
                Ok(u) => u,
                Err(_) => continue,
            };
            match client.get(url).send() {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes()?;
                    let ext = if candidate.ends_with(".i64") { "i64" } else { "i32" };
                    offsets = Some(parse_index_bytes(&bytes, ext));
                    break;
                }
                _ => continue,
            }
        }

        let offsets = offsets.ok_or_else(|| IoError::InvalidFormat(format!(
            "no IDXFOR__ index file found for {}. \
             Run `veks run` to create it (indices are built automatically at write time).",
            data_name,
        )))?;

        Ok(HttpIndexedXvecReader { client, data_url, offsets, data_size, elem_size })
    }

    /// Open a remote variable-length ivec/ivvec file (element size = 4).
    pub fn open_ivec(data_url: Url) -> Result<Self, IoError> {
        Self::open(data_url, 4)
    }

    /// Number of records.
    pub fn count(&self) -> usize {
        self.offsets.len()
    }

    /// Read the dimension of record at the given ordinal via Range request.
    pub fn dim_at(&self, index: usize) -> Result<usize, IoError> {
        if index >= self.offsets.len() {
            return Err(IoError::OutOfBounds(index));
        }
        let offset = self.offsets[index];
        let resp = self.client.get(self.data_url.clone())
            .header(RANGE, format!("bytes={}-{}", offset, offset + 3))
            .send()?
            .error_for_status()?;
        let bytes = resp.bytes()?;
        if bytes.len() < 4 {
            return Err(IoError::InvalidFormat("short dim header read".into()));
        }
        let dim = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        Ok(dim)
    }

    /// Read raw bytes of record data at the given ordinal via Range request.
    pub fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        if index >= self.offsets.len() {
            return Err(IoError::OutOfBounds(index));
        }
        let offset = self.offsets[index];

        // First fetch the dim header to know how many bytes to read
        let resp = self.client.get(self.data_url.clone())
            .header(RANGE, format!("bytes={}-{}", offset, offset + 3))
            .send()?
            .error_for_status()?;
        let dim_bytes = resp.bytes()?;
        if dim_bytes.len() < 4 {
            return Err(IoError::InvalidFormat("short dim header".into()));
        }
        let dim = i32::from_le_bytes([dim_bytes[0], dim_bytes[1], dim_bytes[2], dim_bytes[3]]) as usize;

        // Fetch the record data (past dim header)
        let data_start = offset + 4;
        let data_end = data_start + (dim as u64 * self.elem_size as u64) - 1;
        let resp = self.client.get(self.data_url.clone())
            .header(RANGE, format!("bytes={}-{}", data_start, data_end))
            .send()?
            .error_for_status()?;
        Ok(resp.bytes()?.to_vec())
    }

    /// Read record at the given ordinal as a `Vec<i32>` via Range request.
    pub fn get_i32(&self, index: usize) -> Result<Vec<i32>, IoError> {
        let data = self.get_bytes(index)?;
        let vals: Vec<i32> = data.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(vals)
    }
}

/// Parse raw index bytes into offset vector.
fn parse_index_bytes(bytes: &[u8], ext: &str) -> Vec<u64> {
    match ext {
        "i32" => bytes.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64)
            .collect(),
        "i64" => bytes.chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u64)
            .collect(),
        _ => vec![],
    }
}

