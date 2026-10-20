// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Vector I/O — uniform and variable-length record readers.
//!
//! Two reader shapes, parameterised on element type, both backed by
//! the crate-private `Storage` abstraction:
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
//! methods like `precache()`, `prefetch_range()`, `release_range()`.

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
#[non_exhaustive]
pub enum IoError {
    /// The file holds one element type and the reader was asked for
    /// another. A raw reader never reinterprets; see
    /// [`ElementType::conversion_to`] for what a dataset view converts.
    #[error("the file holds {file} elements and cannot be read as {requested}: {reason}")]
    ElementMismatch {
        /// The element type the file holds, by its extension.
        file: ElementType,
        /// The element type the reader was asked for.
        requested: ElementType,
        /// Why not, in a few words.
        reason: &'static str,
    },
    /// A value of the file's type does not fit the requested type,
    /// under a conversion that checks each value.
    #[error("value {value} at ordinal {index} does not fit in {target}")]
    ValueOverflow {
        /// The ordinal of the value, in the reader's own numbering.
        index: usize,
        /// The value as the file holds it.
        value: veks_datatypes::Value,
        /// The type it was asked to become.
        target: ElementType,
    },
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
    /// A remote variable-length file has no published offset index,
    /// and the only other way to find its record boundaries is to read
    /// every byte of it.
    ///
    /// Opening a reader never does that implicitly: a whole-file
    /// download behind a constructor is the trap this refusal replaces.
    /// Fetch the facet first — which downloads it with progress and
    /// consent — and the reader then rebuilds the index from the local
    /// copy; or publish the `IDXFOR__` sidecar beside the file.
    #[error(
        "no published offset index for {url}; finding its record boundaries means \
         downloading the whole file ({bytes} bytes). Fetch the facet first \
         (TestDataView::fetch, or `vectordata datasets precache`) or publish its \
         IDXFOR__ sidecar"
    )]
    OffsetIndexUnavailable {
        /// The file.
        url: String,
        /// Its size: what fetching it costs.
        bytes: u64,
    },
}

// ═══════════════════════════════════════════════════════════════════════
// Element trait
// ═══════════════════════════════════════════════════════════════════════

pub use veks_datatypes::{Container, Element, ElementConversion, ElementType};

/// The bound the readers put on their element type before the shared
/// model. Every [`Element`] is one; bound on [`Element`] instead.
#[deprecated(note = "bound on `Element` (re-exported here and at the crate root)")]
pub trait VvecElement: Element {}
#[allow(deprecated)]
impl<T: Element> VvecElement for T {}

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

    /// Copy consecutive records `start, start + 1, …` into `out`,
    /// contiguously — record `start + k` lands at
    /// `out[k * dim .. (k + 1) * dim]` — and return how many were copied:
    /// as many as fit, up to the end of the facet.
    ///
    /// **The bulk read.** One call per batch instead of one per record:
    /// a single-file facet reads the whole span in one range request (or
    /// one copy from its mapping, once fetched), and a series or a
    /// windowed facet is split across files here rather than by the
    /// caller. Use it to load a base set, or to stream one a window at a
    /// time into a buffer you reuse.
    ///
    /// `out.len()` must be a multiple of [`dim`](Self::dim). `start ==
    /// count()` copies nothing; past that is
    /// [`IoError::OutOfBounds`].
    fn read_into(&self, start: usize, out: &mut [T]) -> Result<usize, IoError>
    where
        T: Copy,
    {
        let dim = self.dim();
        let n = records_to_read(start, out.len(), dim, self.count())?;
        for (k, record) in out[..n * dim].chunks_exact_mut(dim).enumerate() {
            match self.get_slice(start + k) {
                Some(s) => record.copy_from_slice(s),
                None => record.copy_from_slice(&self.get(start + k)?),
            }
        }
        Ok(n)
    }

    /// Low-level: download this reader's file whole into the local
    /// cache, so later reads are zero-copy. Most callers want
    /// [`TestDataView::fetch`](crate::TestDataView::fetch), which plans,
    /// checks cache space and reports progress. No-op for local files.
    /// Idempotent.
    fn precache(&self) -> io::Result<()> {
        Ok(())
    }

    /// Whether all bytes are locally accessible. `true` for local
    /// files, `true` for cached storage once every chunk is verified,
    /// `false` for direct HTTP.
    fn is_complete(&self) -> bool {
        true
    }

    /// Every element of every record, contiguous, borrowed from the
    /// mapping: `count() * dim()` elements with no copy.
    ///
    /// `Some` only for a header-free column in one resident mapped file
    /// ([`ScalarReader`]), or a window over one. `None` for xvec files,
    /// whose record headers interleave the elements; for a series; for
    /// storage that is not mapped; and for a converting reader, which
    /// has nothing of `T` to borrow.
    fn as_slice(&self) -> Option<&[T]> {
        None
    }

    /// How values reach `T`: [`Native`](ElementConversion::Native)
    /// unless this reader converts them from the file's own type.
    fn conversion(&self) -> ElementConversion {
        ElementConversion::Native
    }

    /// The element type the file holds, when the reader knows it.
    fn native_element_type(&self) -> Option<ElementType> {
        None
    }
}

/// How many records a [`VectorReader::read_into`] call copies: as many
/// whole records as `out_len` holds, up to the end of the facet.
pub(crate) fn records_to_read(
    start: usize,
    out_len: usize,
    dim: usize,
    count: usize,
) -> Result<usize, IoError> {
    if dim == 0 {
        return Ok(0);
    }
    if !out_len.is_multiple_of(dim) {
        return Err(IoError::InvalidFormat(format!(
            "read_into: buffer of {out_len} elements is not a whole number of {dim}-element records"
        )));
    }
    if start > count {
        return Err(IoError::OutOfBounds(start));
    }
    Ok((out_len / dim).min(count - start))
}

/// Copy `n` xvec records out of `bytes` — each a 4-byte dimension header
/// followed by `dim` little-endian elements, `entry_size` bytes apart —
/// into `out`, dropping the headers.
fn copy_records<T: Element>(bytes: &[u8], entry_size: usize, dim: usize, out: &mut [T]) {
    let body = dim * T::SIZE;
    for (k, record) in out.chunks_exact_mut(dim).enumerate() {
        let src = &bytes[k * entry_size + 4..k * entry_size + 4 + body];
        #[cfg(target_endian = "little")]
        {
            debug_assert_eq!(core::mem::size_of::<T>(), T::SIZE);
            // SAFETY: `VvecElement` types are plain little-endian numbers
            // whose in-memory form, on a little-endian host, is exactly
            // their on-disk bytes; `record` holds `dim` of them, which is
            // `body` bytes, and the regions cannot overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(src.as_ptr(), record.as_mut_ptr() as *mut u8, body);
            }
        }
        #[cfg(not(target_endian = "little"))]
        for (dst, b) in record.iter_mut().zip(src.chunks_exact(T::SIZE)) {
            *dst = T::from_le_bytes(b);
        }
    }
}

/// Random-access reader for variable-length vector record files.
pub trait VvecReader<T: Element>: Send + Sync {
    /// Total number of records in the file.
    fn count(&self) -> usize;
    /// Dimension of the record at the given ordinal.
    fn dim_at(&self, index: usize) -> Result<usize, IoError>;
    /// Read the raw bytes of the record data (past the dim header).
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError>;
    /// Read the record at the given ordinal as a Vec of typed elements.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        let bytes = self.get_bytes(index)?;
        Ok(bytes
            .chunks_exact(T::SIZE)
            .map(T::from_le_bytes)
            .collect())
    }
    /// See [`VectorReader::precache`].
    fn precache(&self) -> io::Result<()> {
        Ok(())
    }
    /// See [`VectorReader::is_complete`].
    fn is_complete(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Extension / size inference helpers
// ═══════════════════════════════════════════════════════════════════════

/// Infer element size from an extension string (0 = unknown). Answers
/// from the element model's one table.
pub(crate) fn infer_elem_size(ext: &str) -> usize {
    ElementType::from_extension(ext).map_or(0, ElementType::byte_width)
}

/// Whether the extension implies a variable-length record file. The
/// bare `ivec(s)` is counted: the earliest datasets used it for
/// variable-length files, and the vvec reader still opens them.
pub(crate) fn is_vvec_ext(ext: &str) -> bool {
    matches!(ElementType::parse_extension(ext), Some((_, Container::Vvec)))
        || ext.eq_ignore_ascii_case("ivec")
        || ext.eq_ignore_ascii_case("ivecs")
}

/// Whether `ext` names a *scalar* facet: raw packed values with no
/// per-record header, addressed at `ordinal * elem_size`.
///
/// The single authority for the question — [`crate::typed_access`]
/// classifies both paths and URLs through here. Keeping one table
/// matters because the answer decides whether the first four bytes of
/// a file are a dimension header or data: classify a scalar file as
/// xvec and every offset past the first is wrong. The table is the
/// element model's: every element type has a column spelling, the
/// floats included (`.f32`, `.bf16`).
pub(crate) fn is_scalar_ext(ext: &str) -> bool {
    matches!(ElementType::parse_extension(ext), Some((_, Container::Scalar)))
}

/// The one check a typed open makes: the extension of `source` must
/// name a file whose element type is `T`'s. Returns the file's type and
/// layout.
///
/// A reader never reinterprets. A `.bvecs` is not opened as f32 because
/// the stride happens to divide, and a `.dvec` is not opened as f32
/// because both are floats: anything but `Native` under
/// [`ElementType::conversion_to`] is an [`IoError::ElementMismatch`].
/// A dataset view converts where the table allows it; a raw reader
/// reads a file at its own type only.
pub(crate) fn require_native<T: Element>(source: &str) -> Result<(ElementType, Container), IoError> {
    let ext = ext_of(source);
    let (file, container) = ElementType::parse_extension(ext).ok_or_else(|| {
        IoError::InvalidFormat(format!("cannot infer element type from extension '.{ext}'"))
    })?;
    match file.conversion_to(T::TYPE) {
        ElementConversion::Native => Ok((file, container)),
        ElementConversion::Widening => Err(IoError::ElementMismatch {
            file,
            requested: T::TYPE,
            reason: "a raw reader reads a file at its own type; open it through a dataset view to widen",
        }),
        ElementConversion::CrossSign => Err(IoError::ElementMismatch {
            file,
            requested: T::TYPE,
            reason: "a raw reader reads a file at its own type; open it through a dataset view to check each value",
        }),
        ElementConversion::Refused => Err(IoError::ElementMismatch {
            file,
            requested: T::TYPE,
            reason: "no conversion between the two",
        }),
    }
}

pub(crate) fn ext_of(source: &str) -> &str {
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
///
/// It reads a file at its own type only (`T` must be what the
/// extension names). A record borrows from the mapping when its bytes
/// are aligned for `T` — an 8-byte type at a 4-byte header is not,
/// every other record — and reads by copy otherwise:
///
/// ```
/// use vectordata::io::XvecReader;
/// use vectordata::VectorReader;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("dist.dvecs"))?;
/// # for v in [[0.25f64], [-1e300]] { f.write_all(&1i32.to_le_bytes())?; for x in v { f.write_all(&f64::to_le_bytes(x))?; } }
/// # drop(f);
/// let r = XvecReader::<f64>::open_path(&dir.path().join("dist.dvecs"))?;
/// assert_eq!((r.count(), r.dim()), (2, 1));
/// assert_eq!(VectorReader::get(&r, 1)?, vec![-1e300]);              // always: a decoded copy
/// assert!(VectorReader::get_slice(&r, 0).is_none(), "offset 4 is not 8-aligned");
/// assert_eq!(VectorReader::get_slice(&r, 1), Some(&[-1e300][..]));   // offset 16 is: zero-copy
/// # Ok(()) }
/// ```
pub struct XvecReader<T> {
    storage: Arc<Storage>,
    dim: usize,
    count: usize,
    entry_size: usize,
    /// Mapped base address of each shard, when this reads a series.
    ///
    /// Empty for a single file, which keeps the hot path a plain
    /// pointer add. Held as `usize` rather than `*const u8` so the
    /// reader stays `Send + Sync`: the addresses belong to mmaps that
    /// `storage` owns for this reader's whole life, so sharing them
    /// across threads is exactly as safe as sharing the one base
    /// pointer a single-file reader already shares.
    part_bases: Vec<usize>,
    /// Records in each shard but the last, so an ordinal resolves to
    /// its shard with one division rather than a search (SH-64).
    shard_records: usize,
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

    /// Elements per record (every record has the same dimension).
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Records in the file.
    pub fn count(&self) -> usize {
        self.count
    }
    /// Byte size of each record (4-byte dim header + `dim * SIZE`).
    pub fn entry_size(&self) -> usize {
        self.entry_size
    }

    // ---- storage advice / reclaim API (no T bound) ----

    /// Low-level: download this one file whole. Most callers want
    /// [`TestDataView::fetch`](crate::TestDataView::fetch), which plans,
    /// checks cache space and reports progress across a profile's facets.
    /// No-op for local files.
    pub fn precache(&self) -> io::Result<()> {
        self.storage.precache()
    }

    /// Low-level: [`precache`](Self::precache) with the transport's
    /// progress per chunk. Most callers want
    /// [`TestDataView::fetch`](crate::TestDataView::fetch).
    pub fn prebuffer_with_progress<F>(&self, cb: F) -> io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        self.storage.prebuffer_with_progress(cb)
    }

    /// Whether every byte of the file is resident locally, so reads
    /// make no network requests.
    pub fn is_complete(&self) -> bool {
        self.storage.is_complete()
    }
    /// Hint that the file will be read front to back
    /// (`madvise(SEQUENTIAL)`). No-op unless the file is memory-mapped,
    /// and on non-Unix targets.
    pub fn advise_sequential(&self) {
        self.storage.advise_sequential()
    }
    /// Hint that the file will be read in no particular order
    /// (`madvise(RANDOM)`). No-op unless memory-mapped, and on non-Unix
    /// targets.
    pub fn advise_random(&self) {
        self.storage.advise_random()
    }

    /// Hint that records `start..end` will be read soon
    /// (`madvise(WILLNEED)`); returns without waiting. `end` is clamped
    /// to [`count`](Self::count).
    pub fn prefetch_range(&self, start: usize, end: usize) {
        let bs = (start * self.entry_size) as u64;
        let be = (end.min(self.count) * self.entry_size) as u64;
        self.storage.prefetch_range_bytes(bs, be);
    }

    /// Let the kernel drop the pages of records `start..end`
    /// (`madvise(DONTNEED)`), bounding resident memory during a scan.
    /// The data stays readable; a later read pages it back in.
    pub fn release_range(&self, start: usize, end: usize) {
        let bs = (start * self.entry_size) as u64;
        let be = (end.min(self.count) * self.entry_size) as u64;
        self.storage.release_range_bytes(bs, be);
    }

    /// Page records `start..end` into memory and wait until they are
    /// resident, adding the bytes touched to `bytes_paged` when given.
    /// The blocking counterpart of [`prefetch_range`](Self::prefetch_range).
    pub fn prefetch_pages(&self, start: usize, end: usize, bytes_paged: Option<&AtomicU64>) {
        let bs = (start * self.entry_size) as u64;
        let be = (end.min(self.count) * self.entry_size) as u64;
        self.storage.prefetch_pages_bytes(bs, be, bytes_paged);
    }
}

impl<T: Element> XvecReader<T> {
    /// Open from a path or URL string. Local paths use mmap; URLs try
    /// the merkle-cache path first and fall back to direct HTTP.
    ///
    /// For a remote dataset, prefer opening it by
    /// [`Catalog`](crate::catalog::Catalog) or
    /// [`TestDataGroup`](crate::TestDataGroup) and reading the facet from
    /// its view; a URL opened here shares that dataset's cache directory
    /// when one covers it, but has no fetch or profiles around it.
    ///
    /// Opening a URL reads only the header. Reads then fetch the chunks
    /// they touch — except against a server without HTTP range support,
    /// where the first read downloads the whole file, because no other
    /// way to read it exists. Fetch first
    /// ([`TestDataView::fetch`](crate::TestDataView::fetch)) to see that
    /// download with progress.
    pub fn open(source: &str) -> Result<Self, IoError> {
        Self::require_vectors(source)?;
        let storage = Storage::open(source)?;
        Self::from_storage(storage)
    }

    /// Open a local file by path. Functionally equivalent to
    /// `open(path.to_str().unwrap())` but skips URL parsing.
    pub fn open_path(path: &Path) -> Result<Self, IoError> {
        Self::require_vectors(&path.to_string_lossy())?;
        let storage = Storage::open_path(path)?;
        Self::from_storage(storage)
    }

    /// The extension must name `T` ([`require_native`]) in a vector
    /// layout: a column has no record header to read a dimension from.
    fn require_vectors(source: &str) -> Result<(), IoError> {
        match require_native::<T>(source)? {
            (file, Container::Scalar) => Err(IoError::InvalidFormat(format!(
                "'{}' is a column of {file}, not vectors; read it with ScalarReader or TypedReader",
                ext_of(source)
            ))),
            _ => Ok(()),
        }
    }

    /// Deprecated: open a remote dataset through
    /// [`Catalog`](crate::catalog::Catalog) or
    /// [`TestDataGroup`](crate::TestDataGroup) and read its facets from
    /// the view, which fetch, plan and track it as one dataset. A URL
    /// opened here has no dataset around it: no fetch, no profiles, no
    /// windows. Its bytes do land in that dataset's cache directory when
    /// one covers the URL.
    #[deprecated(
        since = "2.5.0",
        note = "open the dataset by catalog or TestDataGroup and read the facet from its view"
    )]
    pub fn open_url(url: Url) -> Result<Self, IoError> {
        let storage = Storage::open_url(url)?;
        Self::from_storage(storage)
    }

    /// Construct from a pre-opened storage. **Crate-internal**.
    pub(crate) fn from_storage(storage: Arc<Storage>) -> Result<Self, IoError> {
        let total_size = storage.total_size();
        if total_size == 0 {
            return Ok(Self {
                storage,
                dim: DIM_UNDEFINED,
                count: 0,
                entry_size: 0,
                part_bases: Vec::new(),
                shard_records: 0,
                _phantom: PhantomData,
            });
        }
        if total_size < 4 {
            return Err(IoError::InvalidFormat(
                "file too short for dim header".into(),
            ));
        }
        let header = storage.read_bytes(0, 4)?;
        let dim_i32 = LittleEndian::read_i32(&header);
        if dim_i32 <= 0 {
            return Err(IoError::InvalidFormat(format!(
                "invalid dimension {dim_i32}"
            )));
        }
        let dim = dim_i32 as usize;
        let entry_size = 4 + dim * T::SIZE;
        if !total_size.is_multiple_of(entry_size as u64) {
            return Err(IoError::VariableLengthRecords(format!(
                "file size {total_size} is not a multiple of stride {entry_size} (dim={dim}). \
                 Use IndexedVvecReader for variable-length files.",
            )));
        }
        let count = (total_size / entry_size as u64) as usize;
        // For a series, remember where each shard is mapped and how
        // many records it holds, so the zero-copy path can resolve an
        // ordinal to a shard-local address.
        let parts = storage.parts();
        let (part_bases, shard_records) = if parts.is_empty() {
            (Vec::new(), 0)
        } else {
            let bases: Vec<usize> = parts
                .iter()
                .map(|p| p.mmap_base().map_or(0, |b| b as usize))
                .collect();
            // Every shard but the last is full, so shard 0's record
            // count is the stride (SH-35).
            let first = (parts[0].total_size() / entry_size as u64) as usize;
            (bases, first)
        };
        Ok(Self {
            storage,
            dim,
            count,
            entry_size,
            part_bases,
            shard_records,
            _phantom: PhantomData,
        })
    }

    /// The mapped address of record `index` and its offset within the
    /// shard that holds it.
    ///
    /// For a single file this is the file's base and `index` itself —
    /// the same arithmetic that was here before a facet could be a
    /// series. For a series it is one division and an index.
    #[inline]
    fn slice_base(&self, index: usize) -> (*const u8, usize) {
        if self.part_bases.is_empty() {
            let base = self
                .storage
                .mmap_base()
                .expect("XvecReader::get_slice requires mmap-backed storage");
            return (base, index);
        }
        let shard = index / self.shard_records;
        let local = index % self.shard_records;
        let base = self.part_bases[shard];
        assert!(base != 0, "shard {shard} of this series is not mmap-backed");
        (base as *const u8, local)
    }
}

impl<T: Element> XvecReader<T> {
    /// Zero-copy slice of the vector at `index`. **Panics** if the
    /// underlying storage is not mmap-backed.
    ///
    /// **No bounds check** on `index` — hot-path inner-loop form
    /// used by KNN scans and normalisation. Reading past `count` is
    /// undefined behaviour; the caller is responsible for
    /// `index < self.count()`. Use [`VectorReader::get_slice`] for
    /// the bounds-checked `Option<&[T]>` form.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[T] {
        let (base, local) = self.slice_base(index);
        let data_start = local * self.entry_size + 4;
        unsafe { core::slice::from_raw_parts(base.add(data_start) as *const T, self.dim) }
    }
}

impl<T: Element> VectorReader<T> for XvecReader<T> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn count(&self) -> usize {
        self.count
    }
    fn native_element_type(&self) -> Option<ElementType> {
        Some(T::TYPE)
    }

    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }
        let data_start = (index * self.entry_size + 4) as u64;
        let data_len = (self.dim * T::SIZE) as u64;
        if let Some(slice) = self.storage.mmap_slice(data_start, data_len) {
            return Ok(slice
                .chunks_exact(T::SIZE)
                .map(T::from_le_bytes)
                .collect());
        }
        let bytes = self.storage.read_bytes(data_start, data_len)?;
        Ok(bytes
            .chunks_exact(T::SIZE)
            .map(T::from_le_bytes)
            .collect())
    }

    fn get_slice(&self, index: usize) -> Option<&[T]> {
        if index >= self.count {
            return None;
        }
        let data_start = (index * self.entry_size + 4) as u64;
        let data_len = (self.dim * T::SIZE) as u64;
        let bytes = self.storage.mmap_slice(data_start, data_len)?;
        // Cast bytes → &[T]. Safe under the LE-host assumption that
        // the rest of the crate already makes (storage values are
        // little-endian on disk; native LE on x86/ARM means the byte
        // order is identical). The mapping is page-aligned, so a
        // record's data is aligned for T when its offset is: always
        // for sizes ≤ 4; for an 8-byte type the 4-byte header puts
        // every other record at 4 mod 8, and those read by copy. The
        // same rule veks-io's MmapReader applies, record by record.
        let ptr = bytes.as_ptr() as *const T;
        if !(ptr as usize).is_multiple_of(core::mem::align_of::<T>()) {
            return None;
        }
        Some(unsafe { core::slice::from_raw_parts(ptr, self.dim) })
    }

    /// One range of the file for the whole span: a copy out of the
    /// mapping once resident, else a single read.
    fn read_into(&self, start: usize, out: &mut [T]) -> Result<usize, IoError>
    where
        T: Copy,
    {
        let n = records_to_read(start, out.len(), self.dim, self.count)?;
        if n == 0 {
            return Ok(0);
        }
        let from = (start * self.entry_size) as u64;
        let len = (n * self.entry_size) as u64;
        let out = &mut out[..n * self.dim];
        match self.storage.mmap_slice(from, len) {
            Some(bytes) => copy_records(bytes, self.entry_size, self.dim, out),
            None => copy_records(&self.storage.read_bytes(from, len)?, self.entry_size, self.dim, out),
        }
        Ok(n)
    }

    fn precache(&self) -> io::Result<()> {
        self.storage.precache()
    }
    fn is_complete(&self) -> bool {
        self.storage.is_complete()
    }
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
///
/// ```
/// use vectordata::io::IndexedVvecReader;
/// use vectordata::VvecReader;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("labels.u8vvecs"))?;
/// # for v in [vec![9u8, 8, 7], vec![], vec![255]] { f.write_all(&(v.len() as i32).to_le_bytes())?; f.write_all(&v)?; }
/// # drop(f);
/// let r = IndexedVvecReader::<u8>::open_path(&dir.path().join("labels.u8vvecs"))?;   // builds IDXFOR__labels.u8vvecs.i32 beside it
/// assert_eq!(r.count(), 3);
/// assert_eq!((r.dim_at(0)?, r.dim_at(1)?, r.dim_at(2)?), (3, 0, 1));
/// assert_eq!(VvecReader::get(&r, 2)?, vec![255u8]);
/// # Ok(()) }
/// ```
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

impl<T: Element> IndexedVvecReader<T> {
    /// Open a local variable-length file by path. The element width
    /// is taken from `T::SIZE`. Errors if the file extension's
    /// implied size disagrees with `T`.
    pub fn open_path(path: &Path) -> Result<Self, IoError> {
        Self::validate_source(&path.to_string_lossy())?;
        let storage = Storage::open_path(path)?;
        let offsets = load_or_build_local_offsets(path, &storage, T::SIZE)?;
        Ok(Self {
            storage,
            offsets,
            elem_size: T::SIZE,
            _phantom: PhantomData,
        })
    }

    /// Check that `source` names a variable-length file whose element
    /// width is `T`'s, before any storage is opened.
    pub(crate) fn validate_source(source: &str) -> Result<(), IoError> {
        let ext = ext_of(source);
        require_native::<T>(source)?;
        if !is_vvec_ext(ext) {
            return Err(IoError::InvalidFormat(format!(
                "extension '.{ext}' is not a variable-length format; use XvecReader for uniform-stride files"
            )));
        }
        Ok(())
    }

    /// Open from a path or URL string. Auto-dispatches transport.
    ///
    /// Records vary in length, so random access needs an offset index:
    /// a published `IDXFOR__` sidecar, one persisted from an earlier
    /// open, or a walk of a complete local copy. A remote file with none
    /// of those is refused with [`IoError::OffsetIndexUnavailable`]
    /// rather than downloaded whole behind the call; fetch it first
    /// ([`TestDataView::fetch`](crate::TestDataView::fetch)).
    pub fn open(source: &str) -> Result<Self, IoError> {
        Self::validate_source(source)?;
        let elem_size = T::SIZE;

        // `s3://` URLs are remote and get dispatched through the
        // same HTTPS-based offset-fetch path as `http(s)://`. The
        // storage layer translates the scheme internally so the
        // remote offset fetcher needs to see the translated URL
        // when it composes sibling `IDXFOR__` requests.
        let is_remote = crate::transport::is_remote_url(source);
        let storage = Storage::open(source)?;
        let offsets = if is_remote {
            let translated = crate::transport::normalize_remote_url(source);
            load_or_fetch_remote_offsets(
                translated.as_ref(),
                &storage,
                elem_size,
            )?
        } else {
            load_or_build_local_offsets(Path::new(source), &storage, elem_size)?
        };
        Ok(Self {
            storage,
            offsets,
            elem_size,
            _phantom: PhantomData,
        })
    }

    /// Build over an already-opened storage.
    ///
    /// `source` is still needed: it names the file whose sibling
    /// `IDXFOR__` sidecar holds the offsets, and for a series that is
    /// the *shard's own* file — each carries its own index, in its own
    /// file ordinals (SH-17, SH-82).
    pub(crate) fn from_storage(
        storage: Arc<Storage>,
        source: &str,
        elem_size: usize,
    ) -> Result<Self, IoError> {
        let offsets = if crate::transport::is_remote_url(source) {
            let translated = crate::transport::normalize_remote_url(source);
            load_or_fetch_remote_offsets(
                translated.as_ref(),
                &storage,
                elem_size,
            )?
        } else {
            load_or_build_local_offsets(Path::new(source), &storage, elem_size)?
        };
        Ok(Self {
            storage,
            offsets,
            elem_size,
            _phantom: PhantomData,
        })
    }

    /// Records in the file.
    pub fn count(&self) -> usize {
        self.offsets.len()
    }

    /// Elements in record `index` — records vary in length.
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

    /// Record `index`'s elements as raw little-endian bytes, without the
    /// dimension header.
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
        self.storage
            .read_bytes(body_start, body_len)
            .map_err(IoError::Io)
    }

    /// Zero-copy access to record bytes when storage is mmap-backed.
    pub fn get_raw(&self, index: usize) -> Option<&[u8]> {
        if index >= self.offsets.len() {
            return None;
        }
        let offset = self.offsets[index];
        let dim_bytes = self.storage.mmap_slice(offset, 4)?;
        let dim = LittleEndian::read_i32(dim_bytes) as usize;
        let body_start = offset + 4;
        let body_len = (dim * self.elem_size) as u64;
        self.storage.mmap_slice(body_start, body_len)
    }

    /// Low-level: download this one file whole. Most callers want
    /// [`TestDataView::fetch`](crate::TestDataView::fetch), which plans,
    /// checks cache space and reports progress.
    pub fn precache(&self) -> io::Result<()> {
        self.storage.precache()
    }
    /// Whether every byte of the file is resident locally.
    pub fn is_complete(&self) -> bool {
        self.storage.is_complete()
    }
}

impl IndexedVvecReader<i32> {
    /// Read a record as `Vec<i32>`. Convenience alias for the
    /// trait-method `get(index)`.
    pub fn get_i32(&self, index: usize) -> Result<Vec<i32>, IoError> {
        self.get(index)
    }
}

impl<T: Element> VvecReader<T> for IndexedVvecReader<T> {
    fn count(&self) -> usize {
        self.offsets.len()
    }
    fn dim_at(&self, index: usize) -> Result<usize, IoError> {
        self.dim_at(index)
    }
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        self.get_bytes(index)
    }
    fn precache(&self) -> io::Result<()> {
        self.storage.precache()
    }
    fn is_complete(&self) -> bool {
        self.storage.is_complete()
    }
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
pub struct StreamReclaim<'a, T: Element> {
    reader: &'a XvecReader<T>,
    last_released: usize,
    end: usize,
    reclaim_window_records: usize,
}

impl<'a, T: Element> StreamReclaim<'a, T> {
    /// Default 256 MiB reclaim windows. Large enough that the per-
    /// window `madvise` overhead is negligible, small enough that
    /// resident-set growth is bounded.
    pub fn new(reader: &'a XvecReader<T>, start: usize, end: usize) -> Self {
        Self::with_reclaim_bytes(reader, start, end, 256 * 1024 * 1024)
    }

    /// [`new`](Self::new) with an explicit window: the trailing
    /// `reclaim_bytes` of records (at least 1024 records) are released
    /// each time the scan crosses a window boundary.
    pub fn with_reclaim_bytes(
        reader: &'a XvecReader<T>,
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

impl<'a, T: Element> Drop for StreamReclaim<'a, T> {
    fn drop(&mut self) {
        if self.last_released < self.end {
            self.reader.release_range(self.last_released, self.end);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Top-level dispatchers
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
// Generic access: a column reader, conversion, and type erasure
// ═══════════════════════════════════════════════════════════════════════

/// Whether `bytes` starts on a `T` boundary, so a slice of `T` may be
/// borrowed from it.
fn aligned_for<T>(bytes: &[u8]) -> bool {
    (bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<T>())
}

/// Decode `out.len()` elements from their little-endian bytes: one copy
/// on a little-endian host, a decode per element elsewhere.
fn copy_elements<T: Element>(bytes: &[u8], out: &mut [T]) {
    debug_assert_eq!(bytes.len(), out.len() * T::SIZE);
    #[cfg(target_endian = "little")]
    {
        // SAFETY: every `Element` is a plain number whose in-memory form
        // on this host is its little-endian encoding; `out` holds
        // exactly `bytes.len()` bytes of them, and the regions cannot
        // overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        }
    }
    #[cfg(not(target_endian = "little"))]
    for (dst, b) in out.iter_mut().zip(bytes.chunks_exact(T::SIZE)) {
        *dst = T::from_le_bytes(b);
    }
}

/// A zero of `T`, for a scratch buffer.
fn zero<T: Element>() -> T {
    T::from_value(veks_datatypes::Value::Int(0)).expect("every element type holds zero")
}

/// A header-free column: `T` after `T`, nothing else. The reader for
/// a scalar facet (`.u8`, `.i32`, `.f32`, `.bf16`, …), with `dim() ==
/// 1` and the whole column borrowable as one slice.
///
/// Like [`XvecReader`], it reads a file at its own type only; a dataset
/// view converts where [`ElementType::conversion_to`] allows it.
///
/// ```
/// use vectordata::io::ScalarReader;
/// use vectordata::VectorReader;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("scale.f32"))?;
/// # for x in [0.5f32, 0.25, 2.0] { f.write_all(&x.to_le_bytes())?; }
/// # drop(f);
/// # let mut g = std::fs::File::create(dir.path().join("hash.u64"))?;
/// # for x in [u64::MAX, 1] { g.write_all(&x.to_le_bytes())?; }
/// # drop(g);
/// let scale = ScalarReader::<f32>::open_path(&dir.path().join("scale.f32"))?;
/// assert_eq!((scale.count(), scale.dim()), (3, 1));
/// assert_eq!(scale.as_slice(), Some(&[0.5f32, 0.25, 2.0][..]));     // the whole column, zero-copy
/// let hash = ScalarReader::<u64>::open_path(&dir.path().join("hash.u64"))?;
/// assert_eq!(hash.get(0)?, vec![u64::MAX]);
/// # Ok(()) }
/// ```
pub struct ScalarReader<T> {
    storage: Arc<Storage>,
    count: usize,
    _phantom: PhantomData<T>,
}

impl<T: Element> ScalarReader<T> {
    /// Open a column from a path or URL string whose extension names
    /// `T` in the column layout.
    pub fn open(source: &str) -> Result<Self, IoError> {
        Self::require_column(source)?;
        Self::from_storage(Storage::open(source)?)
    }

    /// Open a local column file by path.
    pub fn open_path(path: &Path) -> Result<Self, IoError> {
        Self::require_column(&path.to_string_lossy())?;
        Self::from_storage(Storage::open_path(path)?)
    }

    fn require_column(source: &str) -> Result<(), IoError> {
        match require_native::<T>(source)? {
            (_, Container::Scalar) => Ok(()),
            (file, _) => Err(IoError::InvalidFormat(format!(
                "'{}' holds vectors of {file}, not a column; read it with XvecReader",
                ext_of(source)
            ))),
        }
    }

    /// Construct from a pre-opened storage. **Crate-internal**: the
    /// caller has established that the bytes are a column of `T`.
    pub(crate) fn from_storage(storage: Arc<Storage>) -> Result<Self, IoError> {
        let total = storage.total_size();
        if !total.is_multiple_of(T::SIZE as u64) {
            return Err(IoError::InvalidFormat(format!(
                "a column of {total} bytes is not a whole number of {}-byte {} elements",
                T::SIZE,
                T::TYPE
            )));
        }
        Ok(Self { storage, count: (total / T::SIZE as u64) as usize, _phantom: PhantomData })
    }

    /// Number of elements.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The bytes of elements `start..start + n`, borrowed when the
    /// storage is mapped, read otherwise.
    fn bytes(&self, start: usize, n: usize) -> Result<std::borrow::Cow<'_, [u8]>, IoError> {
        let (offset, len) = ((start * T::SIZE) as u64, (n * T::SIZE) as u64);
        Ok(match self.storage.mmap_slice(offset, len) {
            Some(slice) => std::borrow::Cow::Borrowed(slice),
            None => std::borrow::Cow::Owned(self.storage.read_bytes(offset, len)?),
        })
    }
}

impl<T: Element> std::fmt::Debug for ScalarReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarReader")
            .field("element", &T::TYPE)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl<T: Element> VectorReader<T> for ScalarReader<T> {
    fn dim(&self) -> usize {
        1
    }
    fn count(&self) -> usize {
        self.count
    }
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }
        Ok(vec![T::from_le_bytes(&self.bytes(index, 1)?)])
    }
    fn get_slice(&self, index: usize) -> Option<&[T]> {
        self.as_slice()?.get(index..index + 1)
    }
    fn read_into(&self, start: usize, out: &mut [T]) -> Result<usize, IoError> {
        let n = records_to_read(start, out.len(), 1, self.count)?;
        copy_elements(&self.bytes(start, n)?, &mut out[..n]);
        Ok(n)
    }
    fn precache(&self) -> io::Result<()> {
        self.storage.precache()
    }
    fn is_complete(&self) -> bool {
        self.storage.is_complete()
    }
    fn as_slice(&self) -> Option<&[T]> {
        let bytes = self.storage.mmap_slice(0, (self.count * T::SIZE) as u64)?;
        if !aligned_for::<T>(bytes) {
            return None;
        }
        // SAFETY: `bytes` is `count * T::SIZE` bytes of the mapping on a
        // `T` boundary, and every `Element` is a plain number.
        Some(unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const T, self.count) })
    }
    fn native_element_type(&self) -> Option<ElementType> {
        Some(T::TYPE)
    }
}

/// A reader of `T` over a reader of the file's own type `N`, for the
/// conversions [`ElementType::conversion_to`] allows:
/// [`Widening`](ElementConversion::Widening), which never fails, and
/// [`CrossSign`](ElementConversion::CrossSign), where a value outside
/// `T`'s range is [`IoError::ValueOverflow`] at its ordinal.
///
/// Nothing of `T` exists to borrow, so [`get_slice`](VectorReader::get_slice)
/// and [`as_slice`](VectorReader::as_slice) are `None`; [`get`](VectorReader::get)
/// and [`read_into`](VectorReader::read_into) convert. It is never
/// built for a native read, so the native path costs nothing.
///
/// ```
/// use std::sync::Arc;
/// use vectordata::io::{ConvertingReader, IoError, XvecReader};
/// use vectordata::{ElementConversion, VectorReader};
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("counts.u16vecs"))?;
/// # for v in [[1u16, 2], [3, 40000]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&u16::to_le_bytes(x))?; } }
/// # drop(f);
/// let native: Arc<dyn VectorReader<u16>> = Arc::new(XvecReader::<u16>::open_path(&dir.path().join("counts.u16vecs"))?);
/// let checked = ConvertingReader::<u16, i16>::new(native.clone())?;      // same width, other sign: per value
/// assert_eq!(checked.conversion(), ElementConversion::CrossSign);
/// assert_eq!(checked.get(0)?, vec![1i16, 2]);
/// assert!(matches!(checked.get(1), Err(IoError::ValueOverflow { index: 1, .. })), "40000 is not an i16");
/// let wide = ConvertingReader::<u16, f32>::new(native)?;                 // wider: never fails
/// assert_eq!(wide.get(1)?, vec![3.0, 40000.0]);
/// assert!(ConvertingReader::<u16, u16>::new(Arc::new(XvecReader::<u16>::open_path(&dir.path().join("counts.u16vecs"))?)).is_err(), "never built for a native pair");
/// # Ok(()) }
/// ```
pub struct ConvertingReader<N: Element, T: Element> {
    inner: Arc<dyn VectorReader<N>>,
    mode: ElementConversion,
    _t: PhantomData<T>,
}

/// Bytes of the file's own type per scratch buffer in a converting bulk
/// read.
const CONVERT_SCRATCH_BYTES: usize = 64 << 10;

impl<N: Element, T: Element> ConvertingReader<N, T> {
    /// Wrap `inner`, or refuse: a native pair needs no conversion and
    /// a refused pair has none.
    pub fn new(inner: Arc<dyn VectorReader<N>>) -> Result<Self, IoError> {
        let mode = N::TYPE.conversion_to(T::TYPE);
        let reason = match mode {
            ElementConversion::Widening | ElementConversion::CrossSign => {
                return Ok(Self { inner, mode, _t: PhantomData });
            }
            ElementConversion::Native => "the reader already holds this type",
            ElementConversion::Refused => "no conversion between the two",
        };
        Err(IoError::ElementMismatch { file: N::TYPE, requested: T::TYPE, reason })
    }

    fn convert_run(&self, from: &[N], into: &mut [T], first_ordinal: usize) -> Result<(), IoError> {
        let dim = self.inner.dim().max(1);
        for (k, (n, slot)) in from.iter().zip(into.iter_mut()).enumerate() {
            *slot = veks_datatypes::convert::<N, T>(*n).map_err(|e| IoError::ValueOverflow {
                index: first_ordinal + k / dim,
                value: e.value,
                target: T::TYPE,
            })?;
        }
        Ok(())
    }
}

impl<N: Element, T: Element> std::fmt::Debug for ConvertingReader<N, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvertingReader")
            .field("from", &N::TYPE)
            .field("to", &T::TYPE)
            .field("mode", &self.mode)
            .field("dim", &self.inner.dim())
            .field("count", &self.inner.count())
            .finish()
    }
}

impl<N: Element, T: Element> VectorReader<T> for ConvertingReader<N, T> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn count(&self) -> usize {
        self.inner.count()
    }
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        let native = self.inner.get(index)?;
        let mut out = vec![zero::<T>(); native.len()];
        self.convert_run(&native, &mut out, index)?;
        Ok(out)
    }
    fn get_slice(&self, _index: usize) -> Option<&[T]> {
        None
    }
    fn read_into(&self, start: usize, out: &mut [T]) -> Result<usize, IoError> {
        let dim = self.dim();
        let n = records_to_read(start, out.len(), dim, self.count())?;
        let per_scratch = (CONVERT_SCRATCH_BYTES / (dim * N::SIZE).max(1)).max(1);
        let mut scratch = vec![zero::<N>(); per_scratch * dim];
        let mut done = 0;
        while done < n {
            let want = (n - done).min(per_scratch);
            let got = self.inner.read_into(start + done, &mut scratch[..want * dim])?;
            if got == 0 {
                break;
            }
            let span = &mut out[done * dim..(done + got) * dim];
            self.convert_run(&scratch[..got * dim], span, start + done)?;
            done += got;
        }
        Ok(done)
    }
    fn precache(&self) -> io::Result<()> {
        self.inner.precache()
    }
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    fn as_slice(&self) -> Option<&[T]> {
        None
    }
    fn conversion(&self) -> ElementConversion {
        self.mode
    }
    fn native_element_type(&self) -> Option<ElementType> {
        self.inner.native_element_type().or(Some(N::TYPE))
    }
}

/// The element types a reader can be erased to and recovered from.
/// Implemented for every [`Element`]; crate-internal.
pub(crate) trait AnyElement: Element {
    fn into_any(reader: Arc<dyn VectorReader<Self>>) -> AnyVectorReader;
    fn into_any_vvec(reader: Arc<dyn VvecReader<Self>>) -> AnyVvecReader;
}

fn downcast<N: Element, T: Element>(reader: Arc<dyn VectorReader<N>>) -> Result<Arc<dyn VectorReader<T>>, IoError> {
    let boxed: Box<dyn std::any::Any> = Box::new(reader);
    boxed.downcast::<Arc<dyn VectorReader<T>>>().map(|b| *b).map_err(|_| IoError::ElementMismatch {
        file: N::TYPE,
        requested: T::TYPE,
        reason: "the reader holds the file's own type; ask for that, or convert through a dataset view",
    })
}

fn downcast_vvec<N: Element, T: Element>(reader: Arc<dyn VvecReader<N>>) -> Result<Arc<dyn VvecReader<T>>, IoError> {
    let boxed: Box<dyn std::any::Any> = Box::new(reader);
    boxed.downcast::<Arc<dyn VvecReader<T>>>().map(|b| *b).map_err(|_| IoError::ElementMismatch {
        file: N::TYPE,
        requested: T::TYPE,
        reason: "a variable-length reader holds the file's own type only",
    })
}

macro_rules! any_readers {
    ($($variant:ident $t:ty),* $(,)?) => {
        /// A reader at one of the element types, before the caller's
        /// `T` is known to the compiler.
        ///
        /// What [`open_vec_native`] and
        /// [`TestDataView::open_facet_native`](crate::TestDataView::open_facet_native)
        /// return: the file read at its own type. Recover the typed
        /// reader with [`into_typed`](Self::into_typed), convert with
        /// [`as_type`](Self::as_type), or match on the variant.
        ///
        /// ```
        /// use vectordata::io::{open_vec_native, AnyVectorReader, IoError};
        /// use vectordata::typed_access::bf16;
        /// use vectordata::{ElementType, VectorReader};
        /// # use std::io::Write;
        /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
        /// # let dir = tempfile::tempdir()?;
        /// # let mut f = std::fs::File::create(dir.path().join("emb.bf16vecs"))?;
        /// # for v in [[bf16::from_f32(1.5), bf16::from_f32(-0.125)]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&x.to_le_bytes())?; } }
        /// # drop(f);
        /// # let path = dir.path().join("emb.bf16vecs");
        /// let any = open_vec_native(path.to_str().unwrap())?;
        /// assert_eq!(any.element_type(), ElementType::BF16);
        /// assert!(matches!(any, AnyVectorReader::BF16(_)));
        /// let as_f64 = any.as_type::<f64>()?;                                // widened: exact, converted on read
        /// assert_eq!(as_f64.get(0)?, vec![1.5, -0.125]);
        /// let any = open_vec_native(path.to_str().unwrap())?;
        /// assert!(matches!(any.into_typed::<f32>(), Err(IoError::ElementMismatch { .. })), "into_typed takes the native type only");
        /// # Ok(()) }
        /// ```
        #[non_exhaustive]
        pub enum AnyVectorReader {
            $(
                #[doc = concat!("A reader of `", stringify!($t), "`.")]
                $variant(Arc<dyn VectorReader<$t>>),
            )*
        }

        impl AnyVectorReader {
            /// The element type the reader holds.
            pub fn element_type(&self) -> ElementType {
                match self { $(Self::$variant(_) => ElementType::$variant,)* }
            }

            /// Number of records.
            pub fn count(&self) -> usize {
                match self { $(Self::$variant(r) => r.count(),)* }
            }

            /// Elements per record.
            pub fn dim(&self) -> usize {
                match self { $(Self::$variant(r) => r.dim(),)* }
            }

            /// The reader as `T`, which must be the type it holds:
            /// [`IoError::ElementMismatch`] otherwise. No conversion
            /// happens here; see [`as_type`](Self::as_type) for one.
            pub fn into_typed<T: Element>(self) -> Result<Arc<dyn VectorReader<T>>, IoError> {
                match self { $(Self::$variant(r) => downcast::<$t, T>(r),)* }
            }

            /// The reader converted to `T` by the table.
            pub(crate) fn converted_to<T: Element>(self) -> Result<Arc<dyn VectorReader<T>>, IoError> {
                match self { $(Self::$variant(r) => Ok(Arc::new(ConvertingReader::<$t, T>::new(r)?)),)* }
            }

            /// The reader as `T`: in place when `T` is the file's type,
            /// converted where [`ElementType::conversion_to`] allows,
            /// refused otherwise.
            pub fn as_type<T: Element>(self) -> Result<Arc<dyn VectorReader<T>>, IoError> {
                match self.element_type().conversion_to(T::TYPE) {
                    ElementConversion::Native => self.into_typed::<T>(),
                    ElementConversion::Widening | ElementConversion::CrossSign => self.converted_to::<T>(),
                    ElementConversion::Refused => Err(IoError::ElementMismatch {
                        file: self.element_type(),
                        requested: T::TYPE,
                        reason: "no conversion between the two",
                    }),
                }
            }
        }

        impl std::fmt::Debug for AnyVectorReader {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("AnyVectorReader")
                    .field("element", &self.element_type())
                    .field("dim", &self.dim())
                    .field("count", &self.count())
                    .finish()
            }
        }

        /// A variable-length reader at one of the element types, before
        /// the caller's `T` is known to the compiler. Recovered with
        /// [`into_typed`](Self::into_typed) at the type it holds; a
        /// variable-length facet is never converted.
        #[non_exhaustive]
        pub enum AnyVvecReader {
            $(
                #[doc = concat!("A reader of `", stringify!($t), "` records.")]
                $variant(Arc<dyn VvecReader<$t>>),
            )*
        }

        impl AnyVvecReader {
            /// The element type the reader holds.
            pub fn element_type(&self) -> ElementType {
                match self { $(Self::$variant(_) => ElementType::$variant,)* }
            }

            /// Number of records.
            pub fn count(&self) -> usize {
                match self { $(Self::$variant(r) => r.count(),)* }
            }

            /// The reader as `T`, which must be the type it holds.
            pub fn into_typed<T: Element>(self) -> Result<Arc<dyn VvecReader<T>>, IoError> {
                match self { $(Self::$variant(r) => downcast_vvec::<$t, T>(r),)* }
            }
        }

        impl std::fmt::Debug for AnyVvecReader {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("AnyVvecReader")
                    .field("element", &self.element_type())
                    .field("count", &self.count())
                    .finish()
            }
        }

        $(
            impl AnyElement for $t {
                fn into_any(reader: Arc<dyn VectorReader<$t>>) -> AnyVectorReader {
                    AnyVectorReader::$variant(reader)
                }
                fn into_any_vvec(reader: Arc<dyn VvecReader<$t>>) -> AnyVvecReader {
                    AnyVvecReader::$variant(reader)
                }
            }
        )*
    };
}

any_readers! {
    U8 u8, I8 i8, U16 u16, I16 i16, U32 u32, I32 i32, U64 u64, I64 i64,
    F16 half::f16, BF16 half::bf16, F32 f32, F64 f64,
}

/// Open a vector or column file at the type its extension names, for
/// a caller that does not know the type ahead of time. A column opens
/// as a [`ScalarReader`] (`dim() == 1`), anything else as an
/// [`XvecReader`].
///
/// ```
/// use vectordata::io::{open_vec_native, AnyVectorReader};
/// use vectordata::ElementType;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("scores.i16vecs"))?;
/// # for v in [[-3i16, 7], [i16::MIN, i16::MAX]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&i16::to_le_bytes(x))?; } }
/// # drop(f);
/// let any = open_vec_native(dir.path().join("scores.i16vecs").to_str().unwrap())?;
/// assert_eq!((any.element_type(), any.count(), any.dim()), (ElementType::I16, 2, 2));
/// match any {
///     AnyVectorReader::I16(r) => assert_eq!(r.get(1)?, vec![i16::MIN, i16::MAX]),   // zero-copy, its own type
///     other => panic!("an .i16vecs is i16, not {}", other.element_type()),
/// }
/// # Ok(()) }
/// ```
pub fn open_vec_native(source: &str) -> Result<AnyVectorReader, IoError> {
    let ext = ext_of(source);
    let (file, container) = ElementType::parse_extension(ext).ok_or_else(|| {
        IoError::InvalidFormat(format!("cannot infer element type from extension '.{ext}'"))
    })?;
    veks_datatypes::dispatch_element!(file, N => Ok(match container {
        Container::Scalar => <N as AnyElement>::into_any(Arc::new(ScalarReader::<N>::open(source)?)),
        _ => <N as AnyElement>::into_any(Arc::new(XvecReader::<N>::open(source)?)),
    }))
}

/// Open a vector or column file as `T`: in place when `T` is the type
/// the extension names, converted by [`ElementType::conversion_to`]
/// where the table allows (an `.mvec` as f32, a `.bvecs` as i32),
/// refused otherwise. [`open_vec`] is the native-only form.
///
/// One call, four outcomes, by the table:
///
/// ```
/// use vectordata::io::{open_vec_as, IoError};
/// use vectordata::{ElementConversion, ElementType, VectorReader};
/// use vectordata::typed_access::f16;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("half.mvecs"))?;
/// # for v in [[f16::from_f32(0.5), f16::from_f32(-2.0)]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&f16::to_le_bytes(x))?; } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("ids.ivecs"))?;
/// # for v in [[5i32], [-1]] { f.write_all(&1i32.to_le_bytes())?; for x in v { f.write_all(&i32::to_le_bytes(x))?; } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("codes.bvecs"))?;
/// # for v in [[100u8], [200]] { f.write_all(&1i32.to_le_bytes())?; for x in v { f.write_all(&u8::to_le_bytes(x))?; } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("base.fvecs"))?;
/// # for v in [[1.0f32]] { f.write_all(&1i32.to_le_bytes())?; for x in v { f.write_all(&f32::to_le_bytes(x))?; } }
/// # drop(f);
/// # let p = |n: &str| dir.path().join(n).to_str().unwrap().to_string();
/// // Widening: an f16 file as f32 — exact, one conversion per element read.
/// let wide = open_vec_as::<f32>(&p("half.mvecs"))?;
/// assert_eq!((wide.conversion(), wide.native_element_type()), (ElementConversion::Widening, Some(ElementType::F16)));
/// assert_eq!(wide.get(0)?, vec![0.5, -2.0]);
/// assert!(wide.get_slice(0).is_none(), "nothing of f32 exists to borrow");
///
/// // Widening across integers: i32 ground truth as i64, -1 included.
/// assert_eq!(open_vec_as::<i64>(&p("ids.ivecs"))?.get(1)?, vec![-1i64]);
///
/// // CrossSign: u8 codes as i8 — read what fits, refuse the value that does not, at its ordinal.
/// let checked = open_vec_as::<i8>(&p("codes.bvecs"))?;
/// assert_eq!(checked.get(0)?, vec![100i8]);
/// assert!(matches!(checked.get(1), Err(IoError::ValueOverflow { index: 1, .. })));
///
/// // Refused: f32 as u8 is not lossless, and never a reinterpretation.
/// assert!(matches!(open_vec_as::<u8>(&p("base.fvecs")), Err(IoError::ElementMismatch { file: ElementType::F32, requested: ElementType::U8, .. })));
/// # Ok(()) }
/// ```
pub fn open_vec_as<T: Element>(source: &str) -> Result<Arc<dyn VectorReader<T>>, IoError> {
    open_vec_native(source)?.as_type::<T>()
}

/// Open a uniform vector file for typed random access. Local or
/// remote — the source string is the dispatch. For a remote dataset,
/// prefer [`Catalog`](crate::catalog::Catalog) or
/// [`TestDataGroup`](crate::TestDataGroup) and the view's readers.
///
/// The open is native-only: `T` must be the type the extension names.
///
/// ```
/// use vectordata::io::{open_vec, IoError};
/// use vectordata::ElementType;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("ords.u32vecs"))?;
/// # for v in [[1u32, 2, u32::MAX]] { f.write_all(&3i32.to_le_bytes())?; for x in v { f.write_all(&u32::to_le_bytes(x))?; } }
/// # drop(f);
/// # let path = dir.path().join("ords.u32vecs");
/// let reader = open_vec::<u32>(path.to_str().unwrap())?;              // zero-copy
/// assert_eq!(reader.get(0)?, vec![1, 2, u32::MAX]);
/// // Same width, other sign: refused here; open_vec_as checks it per value.
/// assert!(matches!(open_vec::<i32>(path.to_str().unwrap()), Err(IoError::ElementMismatch { file: ElementType::U32, requested: ElementType::I32, .. })));
/// # Ok(()) }
/// ```
pub fn open_vec<T: Element>(source: &str) -> Result<Box<dyn VectorReader<T>>, IoError> {
    Ok(Box::new(XvecReader::<T>::open(source)?))
}

/// Open a variable-length vector file for typed random access.
///
/// A remote file with no published offset index is refused rather than
/// downloaded whole; see [`IndexedVvecReader::open`].
///
/// ```
/// use vectordata::io::open_vvec;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("matches.i64vvecs"))?;
/// # for v in [vec![7i64, -7], vec![i64::MAX]] { f.write_all(&(v.len() as i32).to_le_bytes())?; for x in v { f.write_all(&x.to_le_bytes())?; } }
/// # drop(f);
/// let r = open_vvec::<i64>(dir.path().join("matches.i64vvecs").to_str().unwrap())?;
/// assert_eq!((r.count(), r.dim_at(0)?, r.dim_at(1)?), (2, 2, 1));
/// assert_eq!(r.get(1)?, vec![i64::MAX]);
/// # Ok(()) }
/// ```
pub fn open_vvec<T: Element>(source: &str) -> Result<Box<dyn VvecReader<T>>, IoError> {
    Ok(Box::new(IndexedVvecReader::<T>::open(source)?))
}

/// Open a variable-length vector file as raw `u8` records (no
/// per-element typing). Useful when the caller decodes records itself.
///
/// A remote file with no published offset index is refused rather than
/// downloaded whole; see [`IndexedVvecReader::open`].
pub fn open_vvec_untyped(source: &str) -> Result<Box<dyn VvecReader<u8>>, IoError> {
    let ext = ext_of(source);
    let elem_size = infer_elem_size(ext);
    if elem_size == 0 {
        return Err(IoError::InvalidFormat(format!(
            "cannot infer element size from extension '.{ext}'"
        )));
    }
    let is_remote = crate::transport::is_remote_url(source);
    let storage = Storage::open(source)?;
    let offsets = if is_remote {
        let translated = crate::transport::normalize_remote_url(source);
        load_or_fetch_remote_offsets(
            translated.as_ref(),
            &storage,
            elem_size,
        )?
    } else {
        load_or_build_local_offsets(Path::new(source), &storage, elem_size)?
    };
    Ok(Box::new(IndexedVvecReader::<u8> {
        storage,
        offsets,
        elem_size,
        _phantom: PhantomData,
    }))
}

// ═══════════════════════════════════════════════════════════════════════
// Offset-index management (private)
// ═══════════════════════════════════════════════════════════════════════

/// Load the record-offset index for a variable-length file, without
/// needing the element type.
///
/// The offsets are the only thing a caller needs to turn an ordinal
/// range into a byte range, and nothing about them depends on `T` —
/// only the walk fallback does, and only to compute a stride it should
/// not be computing in the first place. Exposing them untyped is what
/// lets the prefetch path window a vvec facet without opening a typed
/// reader for a format whose element type it does not care about.
///
/// This is the same lookup [`IndexedVvecReader::open`] performs, in the
/// same order, so a windowed prefetch and a subsequent read agree about
/// where record `i` begins.
pub(crate) fn load_offsets(
    source: &str,
    storage: &Storage,
    elem_size: usize,
) -> Result<Vec<u64>, IoError> {
    if crate::transport::is_remote_url(source) {
        let translated = crate::transport::normalize_remote_url(source);
        load_or_fetch_remote_offsets(translated.as_ref(), storage, elem_size)
    } else {
        load_or_build_local_offsets(Path::new(source), storage, elem_size)
    }
}

fn load_or_build_local_offsets(
    data_path: &Path,
    storage: &Storage,
    elem_size: usize,
) -> Result<Vec<u64>, IoError> {
    let total_size = storage.total_size();
    // The sidecar lives beside the file the storage actually opened,
    // which is not necessarily what the caller named. A `dataset.yaml`
    // facet is written relative to the dataset root ("meta.ivvec"), so
    // resolving it here against the process CWD both misses a published
    // sidecar and writes the rebuilt one into whatever directory the
    // process happens to be sitting in.
    let opened = storage.local_path();
    let data_path = opened.as_deref().unwrap_or(data_path);
    let index_path = index_path_for(data_path, total_size);
    if let Some(cached) = load_local_index(&index_path, data_path, true)? {
        return Ok(cached);
    }
    // Build by walking the mmap (local storage always has mmap).
    let bytes = storage
        .mmap_slice(0, total_size)
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
    // Try sibling IDXFOR__ index URLs.
    let url =
        Url::parse(data_url).map_err(|e| IoError::InvalidFormat(format!("invalid URL: {e}")))?;
    let data_name = url
        .path_segments()
        .and_then(|mut s| s.next_back())
        .unwrap_or("")
        .to_string();
    let mut base = url.clone();
    if base.path_segments_mut().is_ok() {
        let _ = base.path_segments_mut().map(|mut s| {
            s.pop();
        });
    }
    let base_str = base.as_str().trim_end_matches('/').to_string();
    let total_size = storage.total_size();
    // Probe the width this file's size calls for first — the same rule
    // `index_path_for` writes by — so the common case costs one request
    // instead of a guaranteed 404 followed by the real one.
    let (expected, other) = if index_ext_for(total_size) == "i32" {
        ("i32", "i64")
    } else {
        ("i64", "i32")
    };
    let candidates = [
        (
            format!("{base_str}/IDXFOR__{data_name}.{expected}"),
            expected,
        ),
        (format!("{base_str}/IDXFOR__{data_name}.{other}"), other),
    ];
    // The local copy first: an index this cache already holds — one a
    // previous open fetched, or rebuilt by walking a complete copy —
    // needs no network, and a complete dataset must open without one.
    let cache_path = storage.local_path();
    if let Some(cp) = &cache_path {
        let index_path = index_path_for(cp, total_size);
        if let Some(cached) = load_local_index(&index_path, cp, false)? {
            return Ok(cached);
        }
    }
    // Then the published sidecar, kept beside the cache file once
    // fetched so the next open is local. Same read-side auth as every
    // data fetch, or a token-protected server answers 401 and the open
    // falls through to the walk.
    let client = crate::transport::shared_client_for(&base_str);
    let candidates: &[(String, &str)] = if crate::settings::offline() { &[] } else { &candidates };
    for (cand, ext) in candidates {
        let parsed = Url::parse(cand).ok();
        if let Ok(resp) = crate::transport::apply_read_auth(client.get(cand), parsed.as_ref()).send()
            && resp.status().is_success()
            && let Ok(bytes) = resp.bytes()
        {
            let offsets = parse_index_bytes(&bytes, ext, total_size);
            if let Some(cp) = &cache_path {
                let _ = write_index(&index_path_for(cp, total_size), &offsets, total_size);
            }
            return Ok(offsets);
        }
    }
    // Everything cheap is exhausted; only the walk remains. Over a
    // complete local copy that is a read of the cache file. Over a
    // remote file it is a download of every byte, which no open may do
    // behind the caller's back — and which a planner, asking what a
    // transfer would cost, must not do in order to answer. Refuse, and
    // say what to do instead.
    if !storage.is_complete() {
        return Err(IoError::OffsetIndexUnavailable {
            url: data_url.to_string(),
            bytes: storage.total_size(),
        });
    }
    let offsets = walk_offsets_via_storage(storage, elem_size)?;
    // Persist so the walk is paid once per cache lifetime. Best-
    // effort: a read-only cache directory only costs us the rebuild.
    if let Some(cp) = &cache_path {
        let _ = write_index(
            &index_path_for(cp, storage.total_size()),
            &offsets,
            storage.total_size(),
        );
    }
    Ok(offsets)
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
                "negative dimension {dim} at offset {offset}"
            )));
        }
        offset += 4 + dim as u64 * elem_size as u64;
    }
    if offset != file_size {
        return Err(IoError::InvalidFormat(format!(
            "file does not end at a record boundary: {} bytes remaining at offset {offset}",
            file_size - offset
        )));
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
                "negative dimension {dim} at offset {offset}"
            )));
        }
        offset += 4 + dim as u64 * elem_size as u64;
    }
    if offset != total_size {
        return Err(IoError::InvalidFormat(format!(
            "file does not end at a record boundary: {} bytes remaining at offset {offset}",
            total_size - offset
        )));
    }
    Ok(offsets)
}

/// Parse an `IDXFOR__` sidecar into **record start offsets**.
///
/// Two layouts are in circulation next to real data. This crate's own
/// [`write_index`] persists `N` starts; nbdatatools publishes `N + 1`
/// entries whose last is the payload size — an end sentinel that makes
/// every record's extent readable as `offsets[i + 1] - offsets[i]`
/// without a special case for the final one.
///
/// No record can *start* at the payload size, so a trailing entry equal
/// to `payload_size` is unambiguously the sentinel and is dropped here.
/// Without that, a sentinel sidecar reads as `N + 1` records, the last
/// a phantom beginning at EOF: `count()` overstates by one and a window
/// reaching the tail resolves to an empty range.
///
/// Both layouts therefore yield `N` starts, and the record count is the
/// length of what this returns.
fn parse_index_bytes(bytes: &[u8], ext: &str, payload_size: u64) -> Vec<u64> {
    let mut offsets: Vec<u64> = match ext {
        "i32" => bytes
            .chunks_exact(4)
            .map(|c| LittleEndian::read_i32(c) as u64)
            .collect(),
        "i64" => bytes
            .chunks_exact(8)
            .map(|c| LittleEndian::read_i64(c) as u64)
            .collect(),
        _ => Vec::new(),
    };
    if offsets.last() == Some(&payload_size) {
        offsets.pop();
    }
    offsets
}

/// The sidecar entry width a payload of `file_size` bytes calls for:
/// `i32` while every offset still fits one, `i64` beyond that.
///
/// Shared by the local sidecar name and the remote sidecar probe so
/// both agree on which file to expect.
fn index_ext_for(file_size: u64) -> &'static str {
    if file_size <= i32::MAX as u64 {
        "i32"
    } else {
        "i64"
    }
}

fn index_path_for(data_path: &Path, file_size: u64) -> std::path::PathBuf {
    let parent = data_path.parent().unwrap_or(Path::new("."));
    let name = data_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("data");
    parent.join(format!("IDXFOR__{name}.{}", index_ext_for(file_size)))
}

/// Load an offset sidecar beside `data_path`.
///
/// `data_can_change` says whether the data file's mtime means anything:
/// for a local source file it does — an index older than its data was
/// built from a previous version — but a cache copy's mtime moves every
/// time a chunk lands while its content stays fixed, so for one the
/// check would only throw away good indexes. A cache copy whose upstream
/// changed is caught by revalidation, not here.
fn load_local_index(
    index_path: &Path,
    data_path: &Path,
    data_can_change: bool,
) -> Result<Option<Vec<u64>>, IoError> {
    if !index_path.is_file() {
        return Ok(None);
    }
    let data_meta = std::fs::metadata(data_path)?;
    if data_can_change {
        let data_mtime = data_meta
            .modified()
            .map_err(|e| IoError::InvalidFormat(format!("mtime: {e}")))?;
        let index_mtime = std::fs::metadata(index_path)?
            .modified()
            .map_err(|e| IoError::InvalidFormat(format!("mtime: {e}")))?;
        if index_mtime < data_mtime {
            return Ok(None);
        }
    }
    let data = std::fs::read(index_path)?;
    let ext = index_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    // A whole number of offsets, or the file is not one. Belt and
    // braces beside the atomic write above: a sidecar left short by an
    // older build, a truncated copy, or a partial download should send
    // us back to the walk rather than yield a reader that cannot see
    // the records the tail describes.
    let width = if ext == "i64" { 8 } else { 4 };
    if data.is_empty() || !data.len().is_multiple_of(width) {
        return Ok(None);
    }
    Ok(Some(parse_index_bytes(&data, ext, data_meta.len())))
}

/// Write the offset sidecar **atomically**.
///
/// `File::create` truncates in place and the offsets are written one at
/// a time, so a reader that opens the sidecar mid-write sees a short
/// file. `load_local_index` would then parse fewer offsets than the
/// data has records and hand back a reader that silently cannot see the
/// tail of the file — no error, just missing records.
///
/// That is reachable whenever two readers open the same vvec at once,
/// in one process or across processes: both walk, both write. Writing
/// to a sibling temporary and renaming makes the swap atomic, so a
/// reader sees either the previous index or the whole new one.
fn write_index(index_path: &Path, offsets: &[u64], file_size: u64) -> Result<(), IoError> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Unique per writer: the pid separates processes, the counter
    // separates threads within one.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = index_path.parent().unwrap_or(Path::new("."));
    let stem = index_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("IDXFOR__");
    let tmp = dir.join(format!(
        ".{stem}.{}.{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));

    let write = || -> Result<(), IoError> {
        let mut f = File::create(&tmp)?;
        if file_size <= i32::MAX as u64 {
            for &o in offsets {
                f.write_all(&(o as i32).to_le_bytes())?;
            }
        } else {
            for &o in offsets {
                f.write_all(&(o as i64).to_le_bytes())?;
            }
        }
        f.flush()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, index_path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
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
    if name.is_empty() {
        return;
    }
    for ext in &["i32", "i64"] {
        let idx = parent.join(format!("IDXFOR__{name}.{ext}"));
        if idx.exists() {
            let _ = std::fs::remove_file(&idx);
        }
    }
}

/// The `IDXFOR__` sidecar as it is actually published.
///
/// Two layouts are in circulation and both sit next to real data, so
/// the parse has to read either one as the same N records. These pin
/// that, and pin the extension tables the parse depends on.
#[cfg(test)]
mod sidecar_layouts {
    use super::*;

    /// Starts-only — what this crate's own `write_index` persists.
    #[test]
    fn a_starts_only_sidecar_keeps_every_entry() {
        // Three records of 4 bytes each in a 12-byte payload.
        let bytes: Vec<u8> = [0i32, 4, 8].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(parse_index_bytes(&bytes, "i32", 12), vec![0, 4, 8]);
    }

    /// Sentinel — what nbdatatools publishes. The trailing entry is the
    /// payload size, which lets a consumer take every record's extent
    /// as `offsets[i + 1] - offsets[i]`. It is not a record.
    #[test]
    fn an_end_sentinel_is_not_a_record() {
        let bytes: Vec<u8> = [0i32, 4, 8, 12]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            parse_index_bytes(&bytes, "i32", 12),
            vec![0, 4, 8],
            "a final entry equal to the payload size is the sentinel; kept, \
             it reads as a fourth record starting at EOF"
        );
    }

    /// The same, at the wider entry width.
    #[test]
    fn the_sentinel_is_dropped_at_i64_width_too() {
        let bytes: Vec<u8> = [0i64, 20, 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(parse_index_bytes(&bytes, "i64", 40), vec![0, 20]);
    }

    /// Only the *last* entry can be the sentinel. An interior offset
    /// equal to the payload size would be a corrupt index, not a
    /// layout, and silently dropping it would hide that.
    #[test]
    fn only_a_trailing_entry_is_treated_as_the_sentinel() {
        let bytes: Vec<u8> = [0i32, 12, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(parse_index_bytes(&bytes, "i32", 12), vec![0, 12, 4]);
    }

    /// A record needs at least a 4-byte header, so no record can start
    /// at the payload size — which is what makes the sentinel test
    /// unambiguous rather than a heuristic.
    #[test]
    fn the_sentinel_value_is_never_a_legal_record_start() {
        let dims: [i32; 3] = [2, 5, 1];
        let mut at = 0u64;
        let starts: Vec<u64> = dims
            .iter()
            .map(|&d| {
                let s = at;
                at += 4 + d as u64 * 4;
                s
            })
            .collect();
        assert!(
            starts.iter().all(|&s| s < at),
            "every record start is strictly inside the payload"
        );
    }

    /// The width a payload calls for decides both the sidecar's name
    /// and which remote sidecar is probed first. One rule, so the two
    /// cannot drift apart and cost a guaranteed 404 on every open.
    #[test]
    fn the_entry_width_follows_the_payload_size() {
        assert_eq!(index_ext_for(0), "i32");
        assert_eq!(index_ext_for(i32::MAX as u64), "i32");
        assert_eq!(index_ext_for(i32::MAX as u64 + 1), "i64");
        let p = std::path::Path::new("/d/x.ivvec");
        assert_eq!(
            index_path_for(p, 1024).file_name().unwrap(),
            "IDXFOR__x.ivvec.i32"
        );
        assert_eq!(
            index_path_for(p, i32::MAX as u64 + 1).file_name().unwrap(),
            "IDXFOR__x.ivvec.i64"
        );
    }
}

#[cfg(test)]
mod index_atomicity {
    use super::*;

    /// **A half-written sidecar must never be read as a whole one.**
    ///
    /// `write_index` used to truncate the target in place and fill it
    /// offset by offset, so a concurrent reader saw a short file and
    /// parsed fewer offsets than the data had records — a reader that
    /// silently could not see the tail of its own file. It surfaced as
    /// a one-in-six flake in a test that opened one vvec from eight
    /// threads.
    #[test]
    fn many_writers_never_leave_a_readable_partial_index() {
        let tmp = tempfile::tempdir().unwrap();
        let index = tmp.path().join("IDXFOR__x.ivvec.i32");
        let offsets: Vec<u64> = (0..4096).map(|i| i * 12).collect();

        // Writers racing on one path, readers observing it throughout.
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let index = index.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut bad = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        if let Ok(bytes) = std::fs::read(&index)
                            && !bytes.is_empty()
                            && bytes.len() != offsets_len_bytes()
                        {
                            bad += 1;
                        }
                    }
                    bad
                })
            })
            .collect();

        let writers: Vec<_> = (0..4)
            .map(|_| {
                let index = index.clone();
                let offsets = offsets.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        write_index(&index, &offsets, 1 << 20).unwrap();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);

        let partial: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
        assert_eq!(
            partial, 0,
            "readers observed {partial} partial sidecars; the swap is not atomic"
        );

        // And no temporaries are left lying about.
        let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temporaries left behind: {leftovers:?}"
        );
    }

    fn offsets_len_bytes() -> usize {
        4096 * 4
    }
}

/// How a facet's extension decides whether its first four bytes are a
/// dimension header or data.
#[cfg(test)]
mod scalar_classification {
    use super::*;

    /// Scalar facets are the eight integer widths and nothing else.
    ///
    /// The set decides whether a file's first four bytes are a
    /// dimension header or data, so it must be one table: the element
    /// model's, where every type has a column spelling.
    #[test]
    fn the_scalar_extensions_are_every_element_type() {
        for ty in ElementType::ALL {
            let ext = ty.extension(Container::Scalar);
            assert!(is_scalar_ext(ext), "{ext} is a scalar facet");
            assert!(!is_vvec_ext(ext), "{ext} is not variable-length");
            assert_eq!(infer_elem_size(ext), ty.byte_width(), "{ext} has its width");
        }
        for ext in ["fvec", "ivvec", "bf16vecs", "parquet", ""] {
            assert!(!is_scalar_ext(ext), "{ext} is not a scalar facet");
        }
        assert!(is_scalar_ext("I32"), "extension case must not matter");
        assert!(is_scalar_ext("f32") && is_scalar_ext("bf16"), "float columns are scalar facets");
    }

    /// The path and URL classifiers must answer from that one table.
    /// A local open and a remote open of the same facet that disagree
    /// read the same bytes two different ways, one of them wrong.
    #[test]
    fn path_and_url_classification_agree() {
        for ext in [
            "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "fvec", "ivvec", "f32", "f64",
        ] {
            let by_path = <crate::typed_access::ElementType as crate::typed_access::ElementTypeExt>::is_scalar_format(format!("/d/x.{ext}"));
            let by_ext = is_scalar_ext(ext);
            assert_eq!(by_path, by_ext, "path classification of .{ext}");
        }
    }
}

#[cfg(test)]
mod series_reader_tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// Write records `from..from+n` of a dim-`d` fvec into shard `i`.
    /// Record `r` is `[r, r+0.5, ...]`, so a value identifies its own
    /// ordinal.
    fn shard(dir: &std::path::Path, i: u32, d: usize, from: usize, n: usize) {
        let p = dir.join(crate::dataset::shards::shard_name("base", "fvecs", i));
        let mut bytes = Vec::new();
        for r in from..from + n {
            bytes.extend_from_slice(&(d as i32).to_le_bytes());
            for k in 0..d {
                bytes.extend_from_slice(&((r * 10 + k) as f32).to_le_bytes());
            }
        }
        std::fs::write(p, bytes).unwrap();
    }

    /// **A sharded facet reads as one vector space.** Count, dim, and
    /// every record come back as if the shards were one file — which
    /// is what lets a kernel scan a series without knowing it is one.
    #[test]
    fn a_sharded_facet_reads_as_one_vector_space() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 4, 0, 10);
        shard(tmp.path(), 1, 4, 10, 10);
        shard(tmp.path(), 2, 4, 20, 5);

        let r = XvecReader::<f32>::open_path(&tmp.path().join("base.fvecs")).unwrap();
        assert_eq!(VectorReader::count(&r), 25);
        assert_eq!(VectorReader::dim(&r), 4);

        for o in 0..25usize {
            let want: Vec<f32> = (0..4).map(|k| (o * 10 + k) as f32).collect();
            assert_eq!(r.get(o).unwrap(), want, "record {o} via get");
        }
    }

    /// **The zero-copy path crosses shards.** `get_slice` is the KNN
    /// inner loop; if it resolved every ordinal against the first
    /// shard's base it would read adjacent memory and return plausible
    /// garbage rather than failing.
    #[test]
    fn the_zero_copy_path_resolves_per_shard() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 3, 0, 8);
        shard(tmp.path(), 1, 3, 8, 8);
        shard(tmp.path(), 2, 3, 16, 4);

        let r = XvecReader::<f32>::open_path(&tmp.path().join("base.fvecs")).unwrap();
        assert_eq!(VectorReader::count(&r), 20);

        for o in 0..20usize {
            let want: Vec<f32> = (0..3).map(|k| (o * 10 + k) as f32).collect();
            assert_eq!(r.get_slice(o), want.as_slice(), "record {o} via get_slice");
        }
        // The two paths agree, which is the invariant the hot loop
        // relies on.
        for o in [0usize, 7, 8, 15, 16, 19] {
            assert_eq!(r.get_slice(o), r.get(o).unwrap().as_slice(), "record {o}");
        }
    }

    /// A seam is where an off-by-one hides: the last record of each
    /// shard and the first of the next are checked explicitly.
    #[test]
    fn records_either_side_of_a_seam_are_correct() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 2, 0, 6);
        shard(tmp.path(), 1, 2, 6, 6);
        let r = XvecReader::<f32>::open_path(&tmp.path().join("base.fvecs")).unwrap();

        assert_eq!(r.get_slice(5), [50.0f32, 51.0].as_slice(), "last of shard 0");
        assert_eq!(r.get_slice(6), [60.0f32, 61.0].as_slice(), "first of shard 1");
    }

    /// An unsharded file behaves exactly as it did — the series
    /// machinery is inert when there is one file.
    #[test]
    fn a_single_file_is_unaffected() {
        let tmp = tmpdir();
        let path = tmp.path().join("solo.fvecs");
        let mut bytes = Vec::new();
        for r in 0..5usize {
            bytes.extend_from_slice(&2i32.to_le_bytes());
            for k in 0..2 {
                bytes.extend_from_slice(&((r * 10 + k) as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, bytes).unwrap();

        let r = XvecReader::<f32>::open_path(&path).unwrap();
        assert_eq!(VectorReader::count(&r), 5);
        assert_eq!(r.get_slice(4), [40.0f32, 41.0].as_slice());
    }

    /// A series of integer records reads the same way — the shard
    /// resolution is about layout, not element type.
    #[test]
    fn an_integer_series_reads_across_shards() {
        let tmp = tmpdir();
        for (i, from) in [(0u32, 0usize), (1, 4)] {
            let p = tmp.path().join(crate::dataset::shards::shard_name("n", "ivecs", i));
            let mut bytes = Vec::new();
            for r in from..from + 4 {
                bytes.extend_from_slice(&3i32.to_le_bytes());
                for k in 0..3 {
                    bytes.extend_from_slice(&((r * 10 + k) as i32).to_le_bytes());
                }
            }
            std::fs::write(p, bytes).unwrap();
        }
        let r = XvecReader::<i32>::open_path(&tmp.path().join("n.ivecs")).unwrap();
        assert_eq!(VectorReader::count(&r), 8);
        assert_eq!(r.get_slice(7), [70, 71, 72].as_slice());
        assert_eq!(r.get(4).unwrap(), vec![40, 41, 42]);
    }
}
