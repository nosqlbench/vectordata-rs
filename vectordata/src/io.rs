// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector I/O — uniform and variable-length record readers.
//!
//! Two reader shapes, parameterised on element type, both backed by
//! the crate-private [`crate::storage::Storage`] abstraction:
//!
//! - [`XvecReader<T>`] — uniform-stride records (`fvec`, `ivec`, `mvec`,
//!   `dvec`, `bvec`, `svec`, …). All records have the same dimension.
//! - [`IndexedVvecReader<T>`] — variable-length records (`ivvec`,
//!   `fvvec`, `bvvec`, …). Each record may have a different dimension.
//!
//! Both readers are transport-agnostic: a single `open(source)` call
//! handles local files (`mmap`), remote URLs with a published `.mref`
//! (merkle-cached, auto-promoted to `mmap` once complete), and remote
//! URLs without a `.mref` (direct HTTP RANGE, slow fallback).
//!
//! The traits [`VectorReader<T>`] and [`VvecReader<T>`] are the
//! consumer-facing abstractions; the concrete struct types are
//! provided for callers that want to reach for the storage-aware
//! methods like `prebuffer()`, `prefetch_range()`, `release_range()`.

use std::fs::File;
use std::io;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use byteorder::{ByteOrder, LittleEndian};
use thiserror::Error;
use url::Url;

use crate::storage::Storage;

/// Sentinel value for dimension when the file has zero records.
///
/// Dimensionality is undefined (moot) when cardinality is zero — there
/// are no records to derive a dimension from. Callers must check
/// `count() > 0` before relying on `dim()`.
pub const DIM_UNDEFINED: usize = usize::MAX;

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
    /// File has variable-length records — cannot use uniform-stride
    /// random access. Use [`IndexedVvecReader`] for variable-length
    /// xvec files.
    #[error("Variable-length records: {0}")]
    VariableLengthRecords(String),
}

// ═══════════════════════════════════════════════════════════════════════
// Element trait
// ═══════════════════════════════════════════════════════════════════════

/// Trait for types that can be decoded from little-endian bytes.
pub trait VvecElement: Copy + Send + Sync + 'static {
    /// Size of one element in bytes.
    const ELEM_SIZE: usize;
    /// Decode one element from a little-endian byte slice.
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl VvecElement for u8  { const ELEM_SIZE: usize = 1; fn from_le_bytes(b: &[u8]) -> Self { b[0] } }
impl VvecElement for i8  { const ELEM_SIZE: usize = 1; fn from_le_bytes(b: &[u8]) -> Self { b[0] as i8 } }
impl VvecElement for u16 { const ELEM_SIZE: usize = 2; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_u16(b) } }
impl VvecElement for i16 { const ELEM_SIZE: usize = 2; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_i16(b) } }
impl VvecElement for u32 { const ELEM_SIZE: usize = 4; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_u32(b) } }
impl VvecElement for i32 { const ELEM_SIZE: usize = 4; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_i32(b) } }
impl VvecElement for f32 { const ELEM_SIZE: usize = 4; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_f32(b) } }
impl VvecElement for u64 { const ELEM_SIZE: usize = 8; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_u64(b) } }
impl VvecElement for i64 { const ELEM_SIZE: usize = 8; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_i64(b) } }
impl VvecElement for f64 { const ELEM_SIZE: usize = 8; fn from_le_bytes(b: &[u8]) -> Self { LittleEndian::read_f64(b) } }
impl VvecElement for half::f16 {
    const ELEM_SIZE: usize = 2;
    fn from_le_bytes(b: &[u8]) -> Self { half::f16::from_le_bytes([b[0], b[1]]) }
}

// ═══════════════════════════════════════════════════════════════════════
// Traits — VectorReader / VvecReader
// ═══════════════════════════════════════════════════════════════════════

/// Random-access reader for uniform-stride vector files.
///
/// Implementations may be backed by local mmap, merkle-cached remote,
/// or direct-HTTP storage; consumers don't need to know which.
pub trait VectorReader<T>: Send + Sync {
    /// Returns the dimension of the vectors.
    fn dim(&self) -> usize;
    /// Returns the total number of vectors available.
    fn count(&self) -> usize;
    /// Retrieves the vector at the specified index.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;

    /// Try to access the vector at `index` as a borrowed slice
    /// without copying. Returns `None` when the underlying storage
    /// can't satisfy the request without materializing — e.g., HTTP-
    /// backed readers, or merkle-cached storage that hasn't been
    /// promoted to mmap yet.
    fn get_slice(&self, _index: usize) -> Option<&[T]> {
        None
    }

    /// Force-download the underlying storage into the local cache so
    /// subsequent reads are zero-copy. No-op for local files and for
    /// non-cacheable HTTP. Idempotent.
    fn prebuffer(&self) -> io::Result<()> { Ok(()) }

    /// Whether all bytes are locally accessible. `true` for local
    /// files, `true` for cached storage once every chunk is verified,
    /// `false` for direct HTTP.
    fn is_complete(&self) -> bool { true }
}

/// Random-access reader for variable-length vector record files.
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
        Ok(bytes.chunks_exact(T::ELEM_SIZE).map(T::from_le_bytes).collect())
    }
    /// See [`VectorReader::prebuffer`].
    fn prebuffer(&self) -> io::Result<()> { Ok(()) }
    /// See [`VectorReader::is_complete`].
    fn is_complete(&self) -> bool { true }
}

// ═══════════════════════════════════════════════════════════════════════
// Extension / size inference helpers
// ═══════════════════════════════════════════════════════════════════════

/// Infer element size from an extension string (0 = unknown).
fn infer_elem_size(ext: &str) -> usize {
    match ext.to_lowercase().as_str() {
        // 4-byte
        "fvec" | "fvecs" | "f32vec" | "f32vecs" | "ivec" | "ivecs" | "i32vec" | "i32vecs"
        | "u32vec" | "u32vecs" | "fvvec" | "fvvecs" | "ivvec" | "ivvecs"
        | "f32vvec" | "f32vvecs" | "i32vvec" | "i32vvecs" | "u32vvec" | "u32vvecs"
        | "u32" | "i32" => 4,
        // 8-byte
        "dvec" | "dvecs" | "f64vec" | "f64vecs" | "dvvec" | "dvvecs"
        | "i64vec" | "i64vecs" | "u64vec" | "u64vecs"
        | "f64vvec" | "f64vvecs" | "i64vvec" | "i64vvecs" | "u64vvec" | "u64vvecs"
        | "u64" | "i64" => 8,
        // 2-byte
        "mvec" | "mvecs" | "f16vec" | "f16vecs" | "svec" | "svecs" | "i16vec" | "i16vecs"
        | "u16vec" | "u16vecs" | "mvvec" | "mvvecs" | "svvec" | "svvecs"
        | "f16vvec" | "f16vvecs" | "i16vvec" | "i16vvecs" | "u16vvec" | "u16vvecs"
        | "u16" | "i16" => 2,
        // 1-byte
        "bvec" | "bvecs" | "u8vec" | "u8vecs" | "i8vec" | "i8vecs"
        | "bvvec" | "bvvecs" | "u8vvec" | "u8vvecs" | "i8vvec" | "i8vvecs"
        | "u8" | "i8" => 1,
        _ => 0,
    }
}

/// Whether the extension implies a variable-length record file.
fn is_vvec_ext(ext: &str) -> bool {
    let e = ext.to_lowercase();
    e.contains("vvec") || e == "ivec" || e == "ivecs"
}

/// Validate that a file's extension is compatible with `T`'s element width.
fn validate_element_for_source(source: &str) -> Result<(), IoError> {
    let name = source.rsplit('/').next().unwrap_or(source);
    let ext = name.rsplit('.').next().unwrap_or("");
    let size = infer_elem_size(ext);
    if size == 0 {
        return Err(IoError::InvalidFormat(format!(
            "cannot infer element size from extension '.{ext}'")));
    }
    Ok(())
}

fn ext_of(source: &str) -> &str {
    let name = source.rsplit('/').next().unwrap_or(source);
    name.rsplit('.').next().unwrap_or("")
}

// ═══════════════════════════════════════════════════════════════════════
// XvecReader<T> — uniform-stride canonical reader
// ═══════════════════════════════════════════════════════════════════════

/// Canonical uniform-vector reader. Holds an `Arc<Storage>` so the same
/// underlying transport can be shared by multiple shape adapters
/// (e.g., the typed-access view of the same file).
///
/// All transport choice is hidden inside `Storage`. There is no public
/// way to construct an `XvecReader` against a specific transport
/// variant — `open` dispatches based on the source string.
pub struct XvecReader<T> {
    storage: Arc<Storage>,
    dim: usize,
    count: usize,
    entry_size: usize,
    _phantom: PhantomData<T>,
}

impl<T> std::fmt::Debug for XvecReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XvecReader")
            .field("dim", &self.dim)
            .field("count", &self.count)
            .field("entry_size", &self.entry_size)
            .field("storage", &self.storage)
            .finish()
    }
}

impl<T> XvecReader<T> {
    // ---- shape accessors that don't need a VvecElement bound ----

    pub fn dim(&self) -> usize { self.dim }
    pub fn count(&self) -> usize { self.count }
    /// Byte size of each record (4-byte dim header + `dim * ELEM_SIZE`).
    pub fn entry_size(&self) -> usize { self.entry_size }

    // ---- storage advice / reclaim API (no T bound) ----

    /// Drive the underlying storage to fully resident state. No-op
    /// for local files and for non-cacheable HTTP.
    pub fn prebuffer(&self) -> io::Result<()> { self.storage.prebuffer() }

    pub fn prebuffer_with_progress<F>(&self, cb: F) -> io::Result<()>
    where F: FnMut(&crate::transport::DownloadProgress)
    { self.storage.prebuffer_with_progress(cb) }

    pub fn is_complete(&self) -> bool { self.storage.is_complete() }
    pub fn advise_sequential(&self) { self.storage.advise_sequential() }
    pub fn advise_random(&self)     { self.storage.advise_random() }

    pub fn prefetch_range(&self, start: usize, end: usize) {
        let bs = (start * self.entry_size) as u64;
        let be = (end.min(self.count) * self.entry_size) as u64;
        self.storage.prefetch_range_bytes(bs, be);
    }

    pub fn release_range(&self, start: usize, end: usize) {
        let bs = (start * self.entry_size) as u64;
        let be = (end.min(self.count) * self.entry_size) as u64;
        self.storage.release_range_bytes(bs, be);
    }

    pub fn prefetch_pages(
        &self,
        start: usize,
        end: usize,
        bytes_paged: Option<&AtomicU64>,
    ) {
        let bs = (start * self.entry_size) as u64;
        let be = (end.min(self.count) * self.entry_size) as u64;
        self.storage.prefetch_pages_bytes(bs, be, bytes_paged);
    }
}

impl<T: VvecElement> XvecReader<T> {
    /// Open from a path or URL string. Local paths use mmap; URLs try
    /// the merkle-cache path first and fall back to direct HTTP.
    pub fn open(source: &str) -> Result<Self, IoError> {
        validate_element_for_source(source)?;
        let storage = Arc::new(Storage::open(source)?);
        Self::from_storage(storage)
    }

    /// Open a local file by path. Functionally equivalent to
    /// `open(path.to_str().unwrap())` but skips URL parsing.
    pub fn open_path(path: &Path) -> Result<Self, IoError> {
        let storage = Arc::new(Storage::open_path(path)?);
        Self::from_storage(storage)
    }

    /// Open a remote URL with cache-first dispatch. Equivalent to
    /// `open(url.as_str())` but takes a parsed `Url` directly.
    pub fn open_url(url: Url) -> Result<Self, IoError> {
        let storage = Arc::new(Storage::open_url(url)?);
        Self::from_storage(storage)
    }

    /// Construct from a pre-opened storage. **Crate-internal**.
    pub(crate) fn from_storage(storage: Arc<Storage>) -> Result<Self, IoError> {
        let total_size = storage.total_size();
        if total_size == 0 {
            return Ok(Self { storage, dim: DIM_UNDEFINED, count: 0, entry_size: 0, _phantom: PhantomData });
        }
        if total_size < 4 {
            return Err(IoError::InvalidFormat("file too short for dim header".into()));
        }
        let header = storage.read_bytes(0, 4)?;
        let dim_i32 = LittleEndian::read_i32(&header);
        if dim_i32 <= 0 {
            return Err(IoError::InvalidFormat(format!("invalid dimension {dim_i32}")));
        }
        let dim = dim_i32 as usize;
        let entry_size = 4 + dim * T::ELEM_SIZE;
        if total_size % entry_size as u64 != 0 {
            return Err(IoError::VariableLengthRecords(format!(
                "file size {total_size} is not a multiple of stride {entry_size} (dim={dim}). \
                 Use IndexedVvecReader for variable-length files.",
            )));
        }
        let count = (total_size / entry_size as u64) as usize;
        Ok(Self { storage, dim, count, entry_size, _phantom: PhantomData })
    }

}

impl XvecReader<f32> {
    /// Zero-copy slice of the vector at `index`. **Panics** if the
    /// underlying storage is not mmap-backed.
    ///
    /// **No bounds check** on `index` — hot-path inner-loop form
    /// used by KNN scans and normalisation. Reading past `count` is
    /// undefined behaviour; the caller is responsible for
    /// `index < self.count()`. Use [`VectorReader::get_slice`] for
    /// the bounds-checked `Option<&[T]>` form.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[f32] {
        let data_start = index * self.entry_size + 4;
        let base = self.storage.mmap_base()
            .expect("XvecReader::<f32>::get_slice requires mmap-backed storage");
        unsafe {
            core::slice::from_raw_parts(base.add(data_start) as *const f32, self.dim)
        }
    }
}

macro_rules! impl_get_slice {
    ($t:ty) => {
        impl XvecReader<$t> {
            /// See [`XvecReader::<f32>::get_slice`] — same semantics.
            #[inline]
            pub fn get_slice(&self, index: usize) -> &[$t] {
                let data_start = index * self.entry_size + 4;
                let base = self.storage.mmap_base()
                    .expect(concat!("XvecReader::<", stringify!($t),
                                     ">::get_slice requires mmap-backed storage"));
                unsafe {
                    core::slice::from_raw_parts(base.add(data_start) as *const $t, self.dim)
                }
            }
        }
    };
}
impl_get_slice!(half::f16);
impl_get_slice!(f64);
impl_get_slice!(i32);
impl_get_slice!(u8);
impl_get_slice!(i16);

impl<T: VvecElement> VectorReader<T> for XvecReader<T> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }
        let data_start = (index * self.entry_size + 4) as u64;
        let data_len = (self.dim * T::ELEM_SIZE) as u64;
        if let Some(slice) = self.storage.mmap_slice(data_start, data_len) {
            return Ok(slice.chunks_exact(T::ELEM_SIZE).map(T::from_le_bytes).collect());
        }
        let bytes = self.storage.read_bytes(data_start, data_len)?;
        Ok(bytes.chunks_exact(T::ELEM_SIZE).map(T::from_le_bytes).collect())
    }

    fn get_slice(&self, index: usize) -> Option<&[T]> {
        if index >= self.count { return None; }
        let data_start = (index * self.entry_size + 4) as u64;
        let data_len = (self.dim * T::ELEM_SIZE) as u64;
        let bytes = self.storage.mmap_slice(data_start, data_len)?;
        // Cast bytes → &[T]. Safe under the LE-host assumption that
        // the rest of the crate already makes (storage values are
        // little-endian on disk; native LE on x86/ARM means the byte
        // order is identical). Alignment is satisfied by mmap (page-
        // aligned base) and the per-record offset (4-byte dim header
        // keeps T::ELEM_SIZE-aligned data starts aligned for sizes ≤
        // 8). For 8-byte types, the leading dim header forces all
        // records to start at offsets that are 4-aligned, not 8-
        // aligned — so we conservatively bail out on 8-byte types.
        if T::ELEM_SIZE > 4 { return None; }
        let ptr = bytes.as_ptr() as *const T;
        if (ptr as usize) % core::mem::align_of::<T>() != 0 { return None; }
        Some(unsafe { core::slice::from_raw_parts(ptr, self.dim) })
    }

    fn prebuffer(&self) -> io::Result<()> { self.storage.prebuffer() }
    fn is_complete(&self) -> bool { self.storage.is_complete() }
}

// ═══════════════════════════════════════════════════════════════════════
// IndexedVvecReader<T> — variable-length canonical reader
// ═══════════════════════════════════════════════════════════════════════

/// Canonical variable-length vvec reader. Holds an `Arc<Storage>`
/// plus a precomputed offset index mapping ordinal → byte offset.
///
/// On open, reads the existing `IDXFOR__<name>.<i32|i64>` sidecar
/// index when present; otherwise walks the file once to build it and
/// persists for next time. For remote-cached storage the sidecar is
/// fetched from the same URL prefix; for local files it lives next
/// to the data file.
pub struct IndexedVvecReader<T> {
    storage: Arc<Storage>,
    offsets: Vec<u64>,
    elem_size: usize,
    _phantom: PhantomData<T>,
}

impl<T> std::fmt::Debug for IndexedVvecReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedVvecReader")
            .field("count", &self.offsets.len())
            .field("elem_size", &self.elem_size)
            .field("storage", &self.storage)
            .finish()
    }
}

impl<T: VvecElement> IndexedVvecReader<T> {
    /// Open a local variable-length file by path. The element width
    /// is taken from `T::ELEM_SIZE`. Errors if the file extension's
    /// implied size disagrees with `T`.
    pub fn open_path(path: &Path) -> Result<Self, IoError> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let inferred = infer_elem_size(ext);
        if inferred != 0 && inferred != T::ELEM_SIZE {
            return Err(IoError::InvalidFormat(format!(
                "extension '.{ext}' implies {inferred}-byte elements but type {} requires {}",
                std::any::type_name::<T>(), T::ELEM_SIZE)));
        }
        let storage = Arc::new(Storage::open_path(path)?);
        let offsets = load_or_build_local_offsets(path, &storage, T::ELEM_SIZE)?;
        Ok(Self { storage, offsets, elem_size: T::ELEM_SIZE, _phantom: PhantomData })
    }

    /// Open from a path or URL string. Auto-dispatches transport.
    pub fn open(source: &str) -> Result<Self, IoError> {
        let ext = ext_of(source);
        let elem_size = infer_elem_size(ext);
        if elem_size == 0 {
            return Err(IoError::InvalidFormat(format!(
                "cannot infer element size from extension '.{ext}'")));
        }
        if elem_size != T::ELEM_SIZE {
            return Err(IoError::InvalidFormat(format!(
                "extension '.{ext}' implies {elem_size}-byte elements but type {} requires {}",
                std::any::type_name::<T>(), T::ELEM_SIZE)));
        }
        if !is_vvec_ext(ext) && ext != "ivec" {
            return Err(IoError::InvalidFormat(format!(
                "extension '.{ext}' is not a variable-length format; use XvecReader for uniform-stride files")));
        }

        let is_remote = source.starts_with("http://") || source.starts_with("https://");
        let storage = Arc::new(Storage::open(source)?);
        let offsets = if is_remote {
            load_or_fetch_remote_offsets(source, &storage, elem_size)?
        } else {
            load_or_build_local_offsets(Path::new(source), &storage, elem_size)?
        };
        Ok(Self { storage, offsets, elem_size, _phantom: PhantomData })
    }

    pub fn count(&self) -> usize { self.offsets.len() }

    pub fn dim_at(&self, index: usize) -> Result<usize, IoError> {
        if index >= self.offsets.len() {
            return Err(IoError::OutOfBounds(index));
        }
        let offset = self.offsets[index];
        let bytes = self.storage.read_bytes(offset, 4)?;
        if bytes.len() < 4 {
            return Err(IoError::InvalidFormat("short dim header".into()));
        }
        Ok(LittleEndian::read_i32(&bytes) as usize)
    }

    pub fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        if index >= self.offsets.len() {
            return Err(IoError::OutOfBounds(index));
        }
        let offset = self.offsets[index];
        // Read header + body in one shot when storage is local; when
        // remote-direct, two requests is what it costs.
        let dim = self.dim_at(index)?;
        let body_start = offset + 4;
        let body_len = (dim * self.elem_size) as u64;
        self.storage.read_bytes(body_start, body_len).map_err(IoError::Io)
    }

    /// Zero-copy access to record bytes when storage is mmap-backed.
    pub fn get_raw(&self, index: usize) -> Option<&[u8]> {
        if index >= self.offsets.len() { return None; }
        let offset = self.offsets[index];
        let dim_bytes = self.storage.mmap_slice(offset, 4)?;
        let dim = LittleEndian::read_i32(dim_bytes) as usize;
        let body_start = offset + 4;
        let body_len = (dim * self.elem_size) as u64;
        self.storage.mmap_slice(body_start, body_len)
    }

    pub fn prebuffer(&self) -> io::Result<()> { self.storage.prebuffer() }
    pub fn is_complete(&self) -> bool { self.storage.is_complete() }
}

impl IndexedVvecReader<i32> {
    /// Read a record as `Vec<i32>`. Convenience alias for the
    /// trait-method `get(index)`.
    pub fn get_i32(&self, index: usize) -> Result<Vec<i32>, IoError> { self.get(index) }
}

impl<T: VvecElement> VvecReader<T> for IndexedVvecReader<T> {
    fn count(&self) -> usize { self.offsets.len() }
    fn dim_at(&self, index: usize) -> Result<usize, IoError> { self.dim_at(index) }
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> { self.get_bytes(index) }
    fn prebuffer(&self) -> io::Result<()> { self.storage.prebuffer() }
    fn is_complete(&self) -> bool { self.storage.is_complete() }
}

// ═══════════════════════════════════════════════════════════════════════
// Streaming-reclaim helper for bounded-RSS scans
// ═══════════════════════════════════════════════════════════════════════

/// Bounded-RSS streaming-scan helper for [`XvecReader`].
///
/// A sequential mmap scan over a TB-scale file accumulates pages in
/// process RSS once they're touched. The kernel only reclaims under
/// memory pressure, so on RAM-rich machines RSS climbs to the file
/// size before any reclaim happens. `MADV_SEQUENTIAL` alone is
/// insufficient; `MADV_DONTNEED` is the active form.
///
/// `StreamReclaim` advises sequential up front, releases a fixed-byte
/// trailing window each time the cursor crosses a window boundary,
/// and releases any remainder on drop. A no-op when storage is not
/// mmap-backed (the underlying madvise calls just return).
///
/// ```ignore
/// let mut rec = StreamReclaim::new(&reader, start, end);
/// for i in start..end {
///     let v = reader.get_slice(i).unwrap();
///     // ... do work ...
///     rec.advance(i);
/// }
/// // drop releases the trailing tail
/// ```
pub struct StreamReclaim<'a, T: VvecElement> {
    reader: &'a XvecReader<T>,
    last_released: usize,
    end: usize,
    reclaim_window_records: usize,
}

impl<'a, T: VvecElement> StreamReclaim<'a, T> {
    /// Default 256 MiB reclaim windows. Large enough that the per-
    /// window `madvise` overhead is negligible, small enough that
    /// resident-set growth is bounded.
    pub fn new(reader: &'a XvecReader<T>, start: usize, end: usize) -> Self {
        Self::with_reclaim_bytes(reader, start, end, 256 * 1024 * 1024)
    }

    pub fn with_reclaim_bytes(
        reader: &'a XvecReader<T>,
        start: usize,
        end: usize,
        reclaim_bytes: usize,
    ) -> Self {
        reader.advise_sequential();
        let entry_size = reader.entry_size().max(1);
        let reclaim_window_records = (reclaim_bytes / entry_size).max(1024);
        Self { reader, last_released: start, end, reclaim_window_records }
    }

    /// Call after touching index `i`. Releases the trailing window
    /// when `i` crosses a window boundary.
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

impl<'a, T: VvecElement> Drop for StreamReclaim<'a, T> {
    fn drop(&mut self) {
        if self.last_released < self.end {
            self.reader.release_range(self.last_released, self.end);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Top-level dispatchers
// ═══════════════════════════════════════════════════════════════════════

/// Open a uniform vector file for typed random access. Local or
/// remote — the source string is the dispatch.
///
/// ```no_run
/// use vectordata::io::open_vec;
/// let reader = open_vec::<f32>("base_vectors.fvec").unwrap();
/// let v = reader.get(0).unwrap();
/// ```
pub fn open_vec<T: VvecElement>(source: &str) -> Result<Box<dyn VectorReader<T>>, IoError> {
    Ok(Box::new(XvecReader::<T>::open(source)?))
}

/// Open a variable-length vector file for typed random access.
pub fn open_vvec<T: VvecElement>(source: &str) -> Result<Box<dyn VvecReader<T>>, IoError> {
    Ok(Box::new(IndexedVvecReader::<T>::open(source)?))
}

/// Open a variable-length vector file as raw `u8` records (no
/// per-element typing). Useful when the caller decodes records itself.
pub fn open_vvec_untyped(source: &str) -> Result<Box<dyn VvecReader<u8>>, IoError> {
    let ext = ext_of(source);
    let elem_size = infer_elem_size(ext);
    if elem_size == 0 {
        return Err(IoError::InvalidFormat(format!(
            "cannot infer element size from extension '.{ext}'")));
    }
    let is_remote = source.starts_with("http://") || source.starts_with("https://");
    let storage = Arc::new(Storage::open(source)?);
    let offsets = if is_remote {
        load_or_fetch_remote_offsets(source, &storage, elem_size)?
    } else {
        load_or_build_local_offsets(Path::new(source), &storage, elem_size)?
    };
    Ok(Box::new(IndexedVvecReader::<u8> {
        storage, offsets, elem_size, _phantom: PhantomData,
    }))
}

// ═══════════════════════════════════════════════════════════════════════
// Offset-index management (private)
// ═══════════════════════════════════════════════════════════════════════

fn load_or_build_local_offsets(
    data_path: &Path,
    storage: &Storage,
    elem_size: usize,
) -> Result<Vec<u64>, IoError> {
    let total_size = storage.total_size();
    let index_path = index_path_for(data_path, total_size);
    if let Some(cached) = load_local_index(&index_path, data_path)? {
        return Ok(cached);
    }
    // Build by walking the mmap (local storage always has mmap).
    let bytes = storage.mmap_slice(0, total_size)
        .ok_or_else(|| IoError::InvalidFormat("local storage missing mmap".into()))?;
    let offsets = walk_offsets(bytes, elem_size)?;
    let _ = write_index(&index_path, &offsets, total_size); // best-effort
    Ok(offsets)
}

fn load_or_fetch_remote_offsets(
    data_url: &str,
    storage: &Storage,
    elem_size: usize,
) -> Result<Vec<u64>, IoError> {
    // Try sibling IDXFOR__ index URLs (i64 then i32).
    let url = Url::parse(data_url)
        .map_err(|e| IoError::InvalidFormat(format!("invalid URL: {e}")))?;
    let data_name = url.path_segments()
        .and_then(|mut s| s.next_back())
        .unwrap_or("")
        .to_string();
    let mut base = url.clone();
    if base.path_segments_mut().is_ok() {
        let _ = base.path_segments_mut().map(|mut s| { s.pop(); });
    }
    let base_str = base.as_str().trim_end_matches('/').to_string();
    let candidates = [
        (format!("{base_str}/IDXFOR__{data_name}.i64"), "i64"),
        (format!("{base_str}/IDXFOR__{data_name}.i32"), "i32"),
    ];
    let client = reqwest::blocking::Client::new();
    for (cand, ext) in &candidates {
        if let Ok(resp) = client.get(cand).send()
            && resp.status().is_success()
            && let Ok(bytes) = resp.bytes()
        {
            return Ok(parse_index_bytes(&bytes, ext));
        }
    }
    // Fallback: walk the file via Storage::read_bytes (slow for direct
    // HTTP; for cached storage this prebuffers chunks as it walks).
    walk_offsets_via_storage(storage, elem_size)
}

fn walk_offsets(mmap: &[u8], elem_size: usize) -> Result<Vec<u64>, IoError> {
    let file_size = mmap.len() as u64;
    let mut offsets = Vec::new();
    let mut offset: u64 = 0;
    while offset + 4 <= file_size {
        offsets.push(offset);
        let o = offset as usize;
        let dim = LittleEndian::read_i32(&mmap[o..o + 4]);
        if dim < 0 {
            return Err(IoError::InvalidFormat(format!(
                "negative dimension {dim} at offset {offset}")));
        }
        offset += 4 + dim as u64 * elem_size as u64;
    }
    if offset != file_size {
        return Err(IoError::InvalidFormat(format!(
            "file does not end at a record boundary: {} bytes remaining at offset {offset}",
            file_size - offset)));
    }
    Ok(offsets)
}

fn walk_offsets_via_storage(storage: &Storage, elem_size: usize) -> Result<Vec<u64>, IoError> {
    let total_size = storage.total_size();
    let mut offsets = Vec::new();
    let mut offset: u64 = 0;
    while offset + 4 <= total_size {
        offsets.push(offset);
        let header = storage.read_bytes(offset, 4)?;
        let dim = LittleEndian::read_i32(&header);
        if dim < 0 {
            return Err(IoError::InvalidFormat(format!(
                "negative dimension {dim} at offset {offset}")));
        }
        offset += 4 + dim as u64 * elem_size as u64;
    }
    if offset != total_size {
        return Err(IoError::InvalidFormat(format!(
            "file does not end at a record boundary: {} bytes remaining at offset {offset}",
            total_size - offset)));
    }
    Ok(offsets)
}

fn parse_index_bytes(bytes: &[u8], ext: &str) -> Vec<u64> {
    match ext {
        "i32" => bytes.chunks_exact(4)
            .map(|c| LittleEndian::read_i32(c) as u64)
            .collect(),
        "i64" => bytes.chunks_exact(8)
            .map(|c| LittleEndian::read_i64(c) as u64)
            .collect(),
        _ => Vec::new(),
    }
}

fn index_path_for(data_path: &Path, file_size: u64) -> std::path::PathBuf {
    let parent = data_path.parent().unwrap_or(Path::new("."));
    let name = data_path.file_name().and_then(|n| n.to_str()).unwrap_or("data");
    let ext = if file_size <= i32::MAX as u64 { "i32" } else { "i64" };
    parent.join(format!("IDXFOR__{name}.{ext}"))
}

fn load_local_index(index_path: &Path, data_path: &Path) -> Result<Option<Vec<u64>>, IoError> {
    if !index_path.is_file() { return Ok(None); }
    let data_mtime = std::fs::metadata(data_path)?.modified()
        .map_err(|e| IoError::InvalidFormat(format!("mtime: {e}")))?;
    let index_mtime = std::fs::metadata(index_path)?.modified()
        .map_err(|e| IoError::InvalidFormat(format!("mtime: {e}")))?;
    if index_mtime < data_mtime { return Ok(None); }
    let data = std::fs::read(index_path)?;
    let ext = index_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    Ok(Some(parse_index_bytes(&data, ext)))
}

fn write_index(index_path: &Path, offsets: &[u64], file_size: u64) -> Result<(), IoError> {
    use std::io::Write;
    let mut f = File::create(index_path)?;
    if file_size <= i32::MAX as u64 {
        for &o in offsets { f.write_all(&(o as i32).to_le_bytes())?; }
    } else {
        for &o in offsets { f.write_all(&(o as i64).to_le_bytes())?; }
    }
    Ok(())
}

/// Remove any existing IDXFOR__ index files for a data file.
///
/// Call this before rewriting a vvec file to ensure stale indices are
/// cleaned up. The index will be rebuilt by [`IndexedVvecReader::open`].
pub fn remove_vvec_index(data_path: &Path) {
    let parent = data_path.parent().unwrap_or(Path::new("."));
    let name = data_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.is_empty() { return; }
    for ext in &["i32", "i64"] {
        let idx = parent.join(format!("IDXFOR__{name}.{ext}"));
        if idx.exists() {
            let _ = std::fs::remove_file(&idx);
        }
    }
}
