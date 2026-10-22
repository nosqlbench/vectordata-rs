// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The uniform-stride reader: every record a 4-byte dimension and the
//! same number of elements.

use std::marker::PhantomData;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;

use byteorder::{ByteOrder, LittleEndian};
use veks_datatypes::{Container, Element, ElementType};

use super::{
    DIM_UNDEFINED, IoError, VectorReader, aligned_for, copy_records, decode_vec, ext_of, records_to_read,
    require_native,
};
use crate::source::{ByteSource, MappedFile, OpenSource};

/// A uniform vector file: every record a little-endian `i32` dimension
/// followed by that many elements of `T`, the type its extension names
/// (`f32` for `.fvecs`, `u8` for `.bvecs`, [`bf16`](crate::bf16) for
/// `.bf16vecs`, through every [`ElementType`]).
///
/// It reads a file at its own type only. A record borrows from the
/// mapping when its bytes are aligned for `T` — an 8-byte type at a
/// 4-byte header is not, every other record — and reads by copy
/// otherwise:
///
/// ```
/// use veks_io::{VectorReader, XvecReader};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("dist.dvecs")).unwrap();
/// # for v in [[0.25f64], [-1e300]] { f.write_all(&1i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&f64::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// let r = XvecReader::<f64>::open_path(&dir.path().join("dist.dvecs")).unwrap();
/// assert_eq!((r.count(), r.dim()), (2, 1));
/// assert_eq!(VectorReader::get(&r, 1).unwrap(), vec![-1e300]);           // always: a decoded copy
/// assert!(VectorReader::get_slice(&r, 0).is_none(), "offset 4 is not 8-aligned");
/// assert_eq!(VectorReader::get_slice(&r, 1), Some(&[-1e300][..]));        // offset 16 is: zero-copy
/// ```
///
/// A `bf16` file, natively, widened by the caller:
///
/// ```
/// use veks_io::{VectorReader, XvecReader, bf16};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("base.bf16vecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # for row in [[1.5f32, -2.0], [0.0, 64.0]] {
/// #     f.write_all(&2i32.to_le_bytes()).unwrap();
/// #     for v in row { f.write_all(&bf16::from_f32(v).to_le_bytes()).unwrap(); }
/// # }
/// # drop(f);
/// let base = XvecReader::<bf16>::open_path(&path).unwrap();
/// let row: &[bf16] = VectorReader::get_slice(&base, 0).unwrap(); // zero-copy
/// assert_eq!(row.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), [1.5, -2.0]);
/// ```
///
/// A file of another type is refused, never reinterpreted:
///
/// ```
/// use veks_io::{ElementType, IoError, XvecReader};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("codes.bvecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # f.write_all(&4i32.to_le_bytes()).unwrap();
/// # f.write_all(&[1u8, 2, 3, 4]).unwrap();
/// # drop(f);
/// assert_eq!(XvecReader::<u8>::open_path(&path).unwrap().get_slice(0), &[1, 2, 3, 4]);
/// assert!(matches!(
///     XvecReader::<f32>::open_path(&path),
///     Err(IoError::ElementMismatch { file: ElementType::U8, requested: ElementType::F32, .. })
/// ));
/// ```
pub struct XvecReader<T, S: ByteSource = MappedFile> {
    source: S,
    dim: usize,
    count: usize,
    entry_size: usize,
    /// The source's mapped base, kept once it is known: a mapping never
    /// moves for the life of its source, so after the first read a
    /// zero-copy read is a pointer add (IO-4: 1.3× over asking the
    /// source each time).
    resident: OnceLock<usize>,
    /// Mapped base address of each part, when this reads a series.
    ///
    /// Empty for a single file, which keeps the hot path a plain
    /// pointer add. Held as `usize` rather than `*const u8` so the
    /// reader stays `Send + Sync`: the addresses belong to mappings
    /// the source owns for this reader's whole life.
    part_bases: Vec<usize>,
    /// Records in each part but the last, so an ordinal resolves to its
    /// part with one division rather than a search.
    shard_records: usize,
    _phantom: PhantomData<T>,
}

impl<T: Element, S: ByteSource> std::fmt::Debug for XvecReader<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XvecReader")
            .field("element", &T::TYPE)
            .field("dim", &self.dim)
            .field("count", &self.count)
            .field("entry_size", &self.entry_size)
            .field("source", &self.source)
            .finish()
    }
}

impl<T, S: ByteSource> XvecReader<T, S> {
    /// Elements per record (every record has the same dimension);
    /// [`DIM_UNDEFINED`] for an empty file.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Records in the file.
    pub fn count(&self) -> usize {
        self.count
    }
    /// Bytes per record: the 4-byte dimension and `dim` elements.
    pub fn entry_size(&self) -> usize {
        self.entry_size
    }
    /// The source the reader reads.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Low-level: bring this one file here whole. A no-op for a local
    /// file.
    pub fn precache(&self) -> std::io::Result<()> {
        self.source.precache()
    }

    /// Low-level: [`precache`](Self::precache) with the source's
    /// progress.
    pub fn prebuffer_with_progress<F>(&self, mut cb: F) -> std::io::Result<()>
    where
        F: FnMut(&veks_apicore::DownloadProgress),
    {
        self.source.prebuffer_with_progress(&mut cb)
    }

    /// Whether every byte is here, so reads make no requests.
    pub fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
    /// Hint a front-to-back read (`madvise(SEQUENTIAL)`). A no-op
    /// unless mapped, and off Unix.
    pub fn advise_sequential(&self) {
        self.source.advise_sequential()
    }
    /// Hint reads in no order (`madvise(RANDOM)`). A no-op unless
    /// mapped, and off Unix.
    pub fn advise_random(&self) {
        self.source.advise_random()
    }

    /// Hint that records `start..end` will be read soon
    /// (`madvise(WILLNEED)`); does not wait. `end` is clamped to
    /// [`count`](Self::count).
    pub fn prefetch_range(&self, start: usize, end: usize) {
        let (bs, be) = self.byte_range(start, end);
        self.source.prefetch_range_bytes(bs, be);
    }

    /// Let the kernel drop the pages of records `start..end`
    /// (`madvise(DONTNEED)`), bounding resident memory during a scan.
    /// The data stays readable; a later read pages it back in.
    pub fn release_range(&self, start: usize, end: usize) {
        let (bs, be) = self.byte_range(start, end);
        self.source.release_range_bytes(bs, be);
    }

    /// Page records `start..end` in and wait until they are resident,
    /// adding the bytes touched to `bytes_paged` when given.
    pub fn prefetch_pages(&self, start: usize, end: usize, bytes_paged: Option<&AtomicU64>) {
        let (bs, be) = self.byte_range(start, end);
        self.source.prefetch_pages_bytes(bs, be, bytes_paged);
    }

    fn byte_range(&self, start: usize, end: usize) -> (u64, u64) {
        ((start * self.entry_size) as u64, (end.min(self.count) * self.entry_size) as u64)
    }

    /// The source's mapped base, kept after the first answer.
    #[inline]
    fn resident_base(&self) -> Option<*const u8> {
        if let Some(&base) = self.resident.get() {
            return Some(base as *const u8);
        }
        let base = self.source.mmap_base()?;
        let _ = self.resident.set(base as usize);
        Some(base)
    }
}

impl<T: Element, S: ByteSource> XvecReader<T, S> {
    /// Open the file `source` names — a path, or whatever else `S`
    /// reaches — whose extension names `T` in a vector layout.
    pub fn open(source: &str) -> Result<Self, IoError>
    where
        S: OpenSource,
    {
        Self::require_vectors(source)?;
        Self::from_source(S::open(source)?)
    }

    /// Open a local file whose extension names `T`.
    pub fn open_path(path: &Path) -> Result<Self, IoError>
    where
        S: OpenSource,
    {
        Self::require_vectors(&path.to_string_lossy())?;
        Self::from_source(S::open_path(path)?)
    }

    /// The extension must name `T` ([`require_native`]) in the
    /// uniform layout: a column has no record header to read a
    /// dimension from, and a variable-length file's records are not one
    /// stride even when its first few happen to be.
    fn require_vectors(source: &str) -> Result<(), IoError> {
        match require_native::<T>(source)? {
            (file, Container::Scalar) => Err(IoError::InvalidFormat(format!(
                "'{}' is a column of {file}, not vectors; read it with ScalarReader",
                ext_of(source)
            ))),
            (file, Container::Vvec) => Err(IoError::InvalidFormat(format!(
                "'{}' holds variable-length records of {file}; read it with IndexedVvecReader",
                ext_of(source)
            ))),
            (_, Container::Vec) => Ok(()),
        }
    }

    /// Read the file `source` holds as records of `T`. The caller has
    /// established that the bytes are a vector file of `T`: the
    /// extension check is [`open`](Self::open)'s.
    ///
    /// An empty file is zero records of [`DIM_UNDEFINED`] dimension. A
    /// file is refused when its first dimension is not positive, a
    /// record of it would overflow, or its size is not a whole number
    /// of records ([`IoError::VariableLengthRecords`]).
    pub fn from_source(source: S) -> Result<Self, IoError> {
        let total_size = source.total_size();
        let empty = |source| Self {
            source,
            dim: DIM_UNDEFINED,
            count: 0,
            entry_size: 0,
            resident: OnceLock::new(),
            part_bases: Vec::new(),
            shard_records: 0,
            _phantom: PhantomData,
        };
        if total_size == 0 {
            return Ok(empty(source));
        }
        if total_size < 4 {
            return Err(IoError::InvalidFormat(format!("{total_size} bytes is too short for a record header")));
        }
        let header = source.read_bytes(0, 4)?;
        let dim_i32 = LittleEndian::read_i32(&header);
        let entry_size = super::xvec_record_bytes(dim_i32, T::TYPE)?;
        let dim = dim_i32 as usize;
        if !total_size.is_multiple_of(entry_size as u64) {
            return Err(IoError::VariableLengthRecords(format!(
                "file size {total_size} is not a multiple of stride {entry_size} (dim={dim}). \
                 Use IndexedVvecReader for variable-length files.",
            )));
        }
        let count = (total_size / entry_size as u64) as usize;
        // For a series, remember where each part is mapped and how many
        // records it holds, so the zero-copy path can resolve an
        // ordinal to a part-local address.
        let parts = source.parts();
        let (part_bases, shard_records) = if parts.is_empty() {
            (Vec::new(), 0)
        } else {
            let bases = parts.iter().map(|p| p.base.map_or(0, |b| b as usize)).collect();
            // Every part but the last is full, so part 0's record count
            // is the stride.
            (bases, (parts[0].size / entry_size as u64) as usize)
        };
        Ok(Self {
            source,
            dim,
            count,
            entry_size,
            resident: OnceLock::new(),
            part_bases,
            shard_records,
            _phantom: PhantomData,
        })
    }

    /// The mapped address of record `index`'s part and its index
    /// within that part: the file's base and `index` for one file, one
    /// division and an index for a series.
    #[inline]
    fn slice_base(&self, index: usize) -> (*const u8, usize) {
        if self.part_bases.is_empty() {
            let base = self.resident_base().expect("XvecReader::get_slice requires mapped bytes");
            return (base, index);
        }
        let shard = index / self.shard_records;
        let local = index % self.shard_records;
        let base = self.part_bases[shard];
        assert!(base != 0, "part {shard} of this series is not mapped");
        (base as *const u8, local)
    }

    /// Zero-copy slice of the record at `index`. **Panics** when the
    /// bytes are not mapped.
    ///
    /// **No bounds check** on `index`: the inner-loop form KNN scans
    /// and normalisation use. Reading past `count` is undefined
    /// behaviour; the caller keeps `index < self.count()`. Use
    /// [`VectorReader::get_slice`] for the checked `Option<&[T]>` form.
    #[inline]
    pub fn get_slice(&self, index: usize) -> &[T] {
        let (base, local) = self.slice_base(index);
        let data_start = local * self.entry_size + 4;
        // SAFETY: the caller keeps `index < count`, so the record lies
        // inside the mapping the source holds for this reader's life.
        unsafe { std::slice::from_raw_parts(base.add(data_start) as *const T, self.dim) }
    }
}

impl<T: Element, S: ByteSource> VectorReader<T> for XvecReader<T, S> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn count(&self) -> usize {
        self.count
    }
    fn native_element_type(&self) -> Option<ElementType> {
        Some(T::TYPE)
    }

    #[inline]
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }
        let data_start = index * self.entry_size + 4;
        let data_len = self.dim * T::SIZE;
        if self.part_bases.is_empty()
            && let Some(base) = self.resident_base()
        {
            // SAFETY: `index < count`, so the record is inside the
            // mapping the source holds for this reader's life.
            return Ok(decode_vec(unsafe { std::slice::from_raw_parts(base.add(data_start), data_len) }));
        }
        if let Some(slice) = self.source.mmap_slice(data_start as u64, data_len as u64) {
            return Ok(decode_vec(slice));
        }
        Ok(decode_vec(&self.source.read_bytes(data_start as u64, data_len as u64)?))
    }

    #[inline]
    fn get_slice(&self, index: usize) -> Option<&[T]> {
        if index >= self.count {
            return None;
        }
        let data_start = index * self.entry_size + 4;
        let len = self.dim * T::SIZE;
        let bytes: &[u8] = if self.part_bases.is_empty() {
            let base = self.resident_base()?;
            // SAFETY: `index < count`, so the record is inside the
            // mapping the source holds for this reader's life.
            unsafe { std::slice::from_raw_parts(base.add(data_start), len) }
        } else {
            self.source.mmap_slice(data_start as u64, len as u64)?
        };
        // The mapping is page-aligned, so a record's data is aligned
        // for T when its offset is: always for sizes ≤ 4; for an 8-byte
        // type the 4-byte header puts every other record at 4 mod 8,
        // and those read by copy.
        if !aligned_for::<T>(bytes) {
            return None;
        }
        // SAFETY: `bytes` is `dim * T::SIZE` bytes on a `T` boundary,
        // and every `Element` is a plain number.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, self.dim) })
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
        match self.source.mmap_slice(from, len) {
            Some(bytes) => copy_records(bytes, self.entry_size, self.dim, out),
            None => copy_records(&self.source.read_bytes(from, len)?, self.entry_size, self.dim, out),
        }
        Ok(n)
    }

    fn precache(&self) -> std::io::Result<()> {
        self.source.precache()
    }
    fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
}

/// Bounded-memory streaming scan over an [`XvecReader`].
///
/// A sequential scan of a mapped file keeps every page it touches
/// resident until the kernel needs the memory, so on a machine with a
/// lot of it the resident set climbs to the file's size. Advising
/// sequential access alone does not stop that; releasing the pages
/// behind the cursor does.
///
/// `StreamReclaim` advises sequential access up front, releases a
/// fixed-size window behind the cursor each time it crosses a window
/// boundary, and releases the rest on drop. A no-op for bytes that are
/// not mapped.
///
/// ```
/// use veks_io::{StreamReclaim, XvecReader};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("base.fvecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # for i in 0..3000 { f.write_all(&2i32.to_le_bytes()).unwrap(); for x in [i as f32, 1.0] { f.write_all(&x.to_le_bytes()).unwrap(); } }
/// # drop(f);
/// let reader = XvecReader::<f32>::open_path(&path).unwrap();
/// let mut sum = 0.0;
/// let mut reclaim = StreamReclaim::with_reclaim_bytes(&reader, 0, reader.count(), 4096);
/// for i in 0..reader.count() {
///     sum += reader.get_slice(i)[0];
///     reclaim.advance(i);
/// }
/// drop(reclaim); // releases what is left behind the cursor
/// assert_eq!(sum, (0..3000).sum::<i32>() as f32);
/// ```
pub struct StreamReclaim<'a, T: Element, S: ByteSource = MappedFile> {
    reader: &'a XvecReader<T, S>,
    last_released: usize,
    end: usize,
    reclaim_window_records: usize,
}

impl<'a, T: Element, S: ByteSource> StreamReclaim<'a, T, S> {
    /// A scan of records `start..end` releasing 256 MiB windows: large
    /// enough that the per-window `madvise` costs nothing, small enough
    /// that the resident set stays bounded.
    pub fn new(reader: &'a XvecReader<T, S>, start: usize, end: usize) -> Self {
        Self::with_reclaim_bytes(reader, start, end, 256 * 1024 * 1024)
    }

    /// [`new`](Self::new) with an explicit window: the trailing
    /// `reclaim_bytes` of records (at least 1024 records) are released
    /// each time the scan crosses a window boundary.
    pub fn with_reclaim_bytes(reader: &'a XvecReader<T, S>, start: usize, end: usize, reclaim_bytes: usize) -> Self {
        reader.advise_sequential();
        let entry_size = reader.entry_size().max(1);
        let reclaim_window_records = (reclaim_bytes / entry_size).max(1024);
        Self { reader, last_released: start, end, reclaim_window_records }
    }

    /// Call after touching record `i`. Releases the window behind the
    /// cursor when `i` crosses a window boundary.
    #[inline]
    pub fn advance(&mut self, i: usize) {
        if i >= self.last_released + self.reclaim_window_records {
            self.reader.release_range(self.last_released, self.last_released + self.reclaim_window_records);
            self.last_released += self.reclaim_window_records;
        }
    }
}

impl<T: Element, S: ByteSource> Drop for StreamReclaim<'_, T, S> {
    fn drop(&mut self) {
        if self.last_released < self.end {
            self.reader.release_range(self.last_released, self.end);
        }
    }
}
