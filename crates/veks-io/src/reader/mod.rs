// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Random-access readers for every file layout: uniform vectors
//! ([`XvecReader`]), variable-length vectors ([`IndexedVvecReader`]),
//! header-free columns ([`ScalarReader`]), conversion between element
//! types ([`ConvertingReader`]), and readers whose type is chosen at
//! run time ([`AnyVectorReader`]).
//!
//! Each reader is generic over its [`ByteSource`], defaulting to
//! [`MappedFile`], a local file. This is the one implementation of the
//! formats in the workspace: `vectordata`'s readers are these readers
//! over its own source.
//!
//! A reader reads a file at its own type only, the type its extension
//! names. A different `T` is [`IoError::ElementMismatch`], never a
//! reinterpretation of the bytes; [`ConvertingReader`] and
//! [`AnyVectorReader::as_type`] convert where
//! [`ElementType::conversion_to`] allows it.
//!
//! [`ByteSource`]: crate::source::ByteSource
//! [`MappedFile`]: crate::source::MappedFile

use std::io;

use veks_datatypes::{Container, Element, ElementConversion, ElementType};

mod any;
mod scalar;
mod vvec;
mod xvec;

pub use any::{AnyElement, AnyVectorReader, AnyVvecReader, ConvertingReader};
pub use any::{open_vec, open_vec_as, open_vec_as_with, open_vec_native, open_vec_native_with, open_vec_with};
pub use any::{open_vvec, open_vvec_untyped, open_vvec_untyped_with, open_vvec_with};
pub use scalar::ScalarReader;
pub use vvec::{IndexedVvecReader, load_offsets, remove_vvec_index};
pub(crate) use vvec::write_offset_index;
pub use xvec::{StreamReclaim, XvecReader};

/// The dimension a uniform reader reports for an empty file: there is
/// no record to read it from. Check `count() > 0` before relying on
/// `dim()`.
pub const DIM_UNDEFINED: usize = usize::MAX;

/// What went wrong opening or reading a file.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum IoError {
    /// The file holds one element type and the reader was asked for
    /// another. A reader never reinterprets; see
    /// [`ElementType::conversion_to`] for what converts.
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
    /// An I/O error from the file or its source.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// An error from the transport a remote source fetches through.
    #[error("transport error: {0}")]
    Transport(Box<dyn std::error::Error + Send + Sync>),
    /// The bytes are not a file of the expected layout.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    /// A record index at or past the end.
    #[error("Index out of bounds: {0}")]
    OutOfBounds(usize),
    /// The records are not all one length, so a uniform reader cannot
    /// address them. Read the file with [`IndexedVvecReader`].
    #[error("Variable-length records: {0}")]
    VariableLengthRecords(String),
    /// A remote variable-length file has no published offset index,
    /// and the only other way to find its record boundaries is to read
    /// every byte of it.
    ///
    /// Opening a reader never does that implicitly: a whole-file
    /// download behind a constructor is the trap this refusal replaces.
    /// Fetch the file first, and the reader then rebuilds the index
    /// from the local copy; or publish the `IDXFOR__` sidecar beside it.
    #[error(
        "no published offset index for {url}; finding its record boundaries means \
         downloading the whole file ({bytes} bytes). Fetch it first \
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

/// Random access to a uniform-stride file, or anything that reads like
/// one: every record has [`dim`](Self::dim) elements of `T`.
///
/// Implementations may read a local mapping, a cache that fills as it
/// is read, or a remote file; a consumer does not need to know which.
pub trait VectorReader<T>: Send + Sync {
    /// Elements per record.
    fn dim(&self) -> usize;
    /// Records available.
    fn count(&self) -> usize;
    /// The record at `index`, copied.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;

    /// The record at `index` borrowed, with no copy. `None` when the
    /// bytes are not mapped (a remote file not yet fetched), when they
    /// are not aligned for `T`, or when the reader converts.
    fn get_slice(&self, _index: usize) -> Option<&[T]> {
        None
    }

    /// Copy consecutive records `start, start + 1, …` into `out`,
    /// contiguously — record `start + k` lands at
    /// `out[k * dim .. (k + 1) * dim]` — and return how many were copied:
    /// as many as fit, up to the end.
    ///
    /// **The bulk read.** One call per batch instead of one per record:
    /// one file reads the whole span in one request (or one copy from
    /// its mapping), and a series or a window is split across files
    /// here rather than by the caller. Use it to load a base set, or to
    /// stream one a window at a time into a buffer you reuse.
    ///
    /// `out.len()` must be a multiple of [`dim`](Self::dim). `start ==
    /// count()` copies nothing; past that is [`IoError::OutOfBounds`].
    ///
    /// Three `i16` records in an `.svecs` file; a buffer for two reads
    /// records 1 and 2 in one copy, and from the last record only what
    /// remains:
    ///
    /// ```
    /// use veks_io::{IoError, VectorReader, XvecReader};
    /// # use std::io::Write;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let path = dir.path().join("short.svecs");
    /// # let mut f = std::fs::File::create(&path).unwrap();
    /// # for row in [[1i16, -1], [200, -200], [i16::MAX, i16::MIN]] {
    /// #     f.write_all(&2i32.to_le_bytes()).unwrap();
    /// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
    /// # }
    /// # drop(f);
    /// let short = XvecReader::<i16>::open_path(&path).unwrap();
    /// let mut out = vec![0i16; short.dim() * 2];
    /// assert_eq!(short.read_into(1, &mut out).unwrap(), 2);
    /// assert_eq!(out, [200, -200, i16::MAX, i16::MIN]);
    /// assert_eq!(short.read_into(2, &mut out).unwrap(), 1); // only one record remains
    /// assert_eq!(&out[..2], &[i16::MAX, i16::MIN]);
    /// assert!(matches!(short.read_into(4, &mut out), Err(IoError::OutOfBounds(4))));
    /// ```
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

    /// Bring the whole file here, so later reads are zero-copy. A no-op
    /// for a local file. Idempotent.
    fn precache(&self) -> io::Result<()> {
        Ok(())
    }

    /// Whether every byte is here, so no read makes a request.
    fn is_complete(&self) -> bool {
        true
    }

    /// Every element of every record, contiguous, borrowed: `count() *
    /// dim()` elements with no copy.
    ///
    /// `Some` only for a header-free column in one mapped file
    /// ([`ScalarReader`]), or a window over one. `None` for xvec files,
    /// whose record headers interleave the elements; for a series; for
    /// bytes not mapped; and for a converting reader, which has nothing
    /// of `T` to borrow.
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

/// Random access to a file whose records vary in length.
pub trait VvecReader<T: Element>: Send + Sync {
    /// Records in the file.
    fn count(&self) -> usize;
    /// Elements in record `index`.
    fn dim_at(&self, index: usize) -> Result<usize, IoError>;
    /// Record `index`'s elements as raw little-endian bytes, without
    /// the dimension header.
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError>;
    /// Record `index` as elements.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        let bytes = self.get_bytes(index)?;
        Ok(decode_vec(&bytes))
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

/// How many records a [`VectorReader::read_into`] call copies: as many
/// whole records as `out_len` holds, up to the end.
pub fn records_to_read(start: usize, out_len: usize, dim: usize, count: usize) -> Result<usize, IoError> {
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

/// The bytes of one record of a uniform vector file whose first header
/// says `dim`, for elements of `elem`: the 4-byte header and `dim`
/// elements. Refused when `dim` is not positive or the record would
/// overflow.
///
/// ```
/// use veks_io::ElementType;
/// use veks_io::reader::xvec_record_bytes;
/// assert_eq!(xvec_record_bytes(128, ElementType::F32).unwrap(), 4 + 512);
/// assert_eq!(xvec_record_bytes(3, ElementType::BF16).unwrap(), 4 + 6);
/// assert!(xvec_record_bytes(0, ElementType::U8).is_err());
/// assert!(xvec_record_bytes(-4, ElementType::F32).unwrap_err().to_string().contains("dimension -4 is not positive"));
/// ```
pub fn xvec_record_bytes(dim: i32, elem: ElementType) -> Result<usize, IoError> {
    if dim <= 0 {
        return Err(IoError::InvalidFormat(format!("dimension {dim} is not positive")));
    }
    (dim as usize)
        .checked_mul(elem.byte_width())
        .and_then(|payload| payload.checked_add(4))
        .ok_or_else(|| IoError::InvalidFormat(format!("a record of dimension {dim} overflows")))
}

/// The extension of a path or URL: what follows the last `.` of its
/// last segment.
pub fn ext_of(source: &str) -> &str {
    let name = source.rsplit('/').next().unwrap_or(source);
    name.rsplit('.').next().unwrap_or("")
}

/// Bytes per element the extension names, `0` when it names none.
pub fn infer_elem_size(ext: &str) -> usize {
    ElementType::from_extension(ext).map_or(0, ElementType::byte_width)
}

/// Whether the extension names a variable-length record file. The bare
/// `ivec(s)` is counted: the earliest datasets used it for
/// variable-length files, and the vvec reader still opens them.
pub fn is_vvec_ext(ext: &str) -> bool {
    matches!(ElementType::parse_extension(ext), Some((_, Container::Vvec)))
        || ext.eq_ignore_ascii_case("ivec")
        || ext.eq_ignore_ascii_case("ivecs")
}

/// Whether the extension names a header-free column: values packed with
/// no record header, addressed at `ordinal * size`.
///
/// The one authority for the question. The answer decides whether a
/// file's first four bytes are a dimension or data: classify a column
/// as xvec and every offset past the first is wrong.
pub fn is_scalar_ext(ext: &str) -> bool {
    matches!(ElementType::parse_extension(ext), Some((_, Container::Scalar)))
}

/// The one check a typed open makes: the extension of `source` must
/// name a file whose element type is `T`'s. Returns the file's type and
/// layout.
///
/// A `.bvecs` is not opened as f32 because the stride happens to
/// divide, and a `.dvec` is not opened as f32 because both are floats:
/// anything but `Native` under [`ElementType::conversion_to`] is
/// [`IoError::ElementMismatch`].
///
/// ```
/// use veks_io::{Container, ElementType, IoError};
/// use veks_io::reader::require_native;
///
/// assert_eq!(require_native::<u8>("codes.bvecs").unwrap(), (ElementType::U8, Container::Vec));
/// assert_eq!(require_native::<half::bf16>("scale.bf16").unwrap(), (ElementType::BF16, Container::Scalar));
/// // u8 would widen to f32 losslessly, but a raw reader never converts.
/// assert!(matches!(
///     require_native::<f32>("codes.bvecs"),
///     Err(IoError::ElementMismatch { file: ElementType::U8, requested: ElementType::F32, .. })
/// ));
/// assert!(matches!(require_native::<f32>("rows.xyz"), Err(IoError::InvalidFormat(_))));
/// ```
pub fn require_native<T: Element>(source: &str) -> Result<(ElementType, Container), IoError> {
    let ext = ext_of(source);
    let (file, container) = ElementType::parse_extension(ext)
        .ok_or_else(|| IoError::InvalidFormat(format!("cannot infer element type from extension '.{ext}'")))?;
    let reason = match file.conversion_to(T::TYPE) {
        ElementConversion::Native => return Ok((file, container)),
        ElementConversion::Widening => {
            "a raw reader reads a file at its own type; open it through a dataset view or a converting reader to widen"
        }
        ElementConversion::CrossSign => {
            "a raw reader reads a file at its own type; open it through a dataset view or a converting reader to check each value"
        }
        ElementConversion::Refused => "no conversion between the two",
    };
    Err(IoError::ElementMismatch { file, requested: T::TYPE, reason })
}

/// Whether `bytes` starts on a `T` boundary, so a slice of `T` may be
/// borrowed from it.
pub(crate) fn aligned_for<T>(bytes: &[u8]) -> bool {
    (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>())
}

/// Decode `out.len()` elements from their little-endian bytes: one copy
/// on a little-endian host, a decode per element elsewhere.
#[inline]
pub(crate) fn copy_elements<T: Element>(bytes: &[u8], out: &mut [T]) {
    debug_assert_eq!(bytes.len(), out.len() * T::SIZE);
    #[cfg(target_endian = "little")]
    // SAFETY: every `Element` is a plain number whose in-memory form on
    // this host is its little-endian encoding; `out` holds exactly
    // `bytes.len()` bytes of them, and the regions cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    #[cfg(not(target_endian = "little"))]
    for (dst, b) in out.iter_mut().zip(bytes.chunks_exact(T::SIZE)) {
        *dst = T::from_le_bytes(b);
    }
}

/// The elements `bytes` encodes, as a new `Vec`: one allocation and,
/// on a little-endian host, one copy. Never a loop the compiler must
/// inline to make fast (the IO-4 measurement of an out-of-line
/// per-element decode: 2.4× slower).
#[inline]
pub(crate) fn decode_vec<T: Element>(bytes: &[u8]) -> Vec<T> {
    let n = bytes.len() / T::SIZE;
    let mut out = Vec::<T>::with_capacity(n);
    #[cfg(target_endian = "little")]
    // SAFETY: the capacity is `n` elements, `n * T::SIZE` bytes are
    // copied into it, and every bit pattern of those bytes is a valid
    // `T` (every `Element` is a plain number).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, n * T::SIZE);
        out.set_len(n);
    }
    #[cfg(not(target_endian = "little"))]
    out.extend(bytes.chunks_exact(T::SIZE).map(T::from_le_bytes));
    out
}

/// Copy `n` xvec records out of `bytes` — each a 4-byte dimension header
/// followed by `dim` little-endian elements, `entry_size` bytes apart —
/// into `out`, dropping the headers.
pub(crate) fn copy_records<T: Element>(bytes: &[u8], entry_size: usize, dim: usize, out: &mut [T]) {
    let body = dim * T::SIZE;
    for (k, record) in out.chunks_exact_mut(dim).enumerate() {
        let src = &bytes[k * entry_size + 4..k * entry_size + 4 + body];
        copy_elements(src, record);
    }
}

/// A zero of `T`, for a scratch buffer.
pub(crate) fn zero<T: Element>() -> T {
    T::from_value(veks_datatypes::Value::Int(0)).expect("every element type holds zero")
}

/// How an extension decides whether a file's first four bytes are a
/// dimension header or data.
#[cfg(test)]
mod scalar_classification {
    use super::*;

    /// The column extensions are every element type's column spelling,
    /// and nothing else. The set decides whether a file's first four
    /// bytes are a dimension header or data, so it is one table: the
    /// element model's.
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
}
