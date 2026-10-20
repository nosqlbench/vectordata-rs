// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The element model as this crate applies it: the error a reader
//! raises, and the one check every typed open makes.
//!
//! The model itself — [`ElementType`], [`Element`], the extension
//! table and the conversion table — is `veks-datatypes`, re-exported
//! here so a program that reads through this crate names them from
//! one place.

use std::path::Path;

pub use veks_datatypes::{
    Container, Element, ElementConversion, ElementKind, ElementType, FloatFormat, bf16, f16,
};

/// What went wrong opening or reading a memory-mapped file.
#[derive(Debug)]
#[non_exhaustive]
pub enum MmapError {
    /// The file could not be opened.
    Open(std::io::Error),
    /// The file could not be memory-mapped.
    Mmap(std::io::Error),
    /// The file's bytes are not a whole file of the expected layout:
    /// empty, too short for a header, a dimension that is not
    /// positive, a size that is not a whole number of records, or an
    /// extension that names no layout.
    InvalidFormat(String),
    /// The extension says the file holds one element type and the
    /// reader was asked for another. A reader reads a file at its own
    /// type only; see [`ElementType::conversion_to`] for what a
    /// converting read would allow.
    ElementMismatch {
        /// The element type the file holds, by its extension.
        file: ElementType,
        /// The element type the reader was asked for.
        requested: ElementType,
    },
    /// A record index at or past the end.
    OutOfBounds {
        /// The index asked for.
        index: usize,
        /// How many records there are.
        count: usize,
    },
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::Open(e) => write!(f, "open: {e}"),
            MmapError::Mmap(e) => write!(f, "mmap: {e}"),
            MmapError::InvalidFormat(s) => write!(f, "invalid format: {s}"),
            MmapError::ElementMismatch { file, requested } => {
                write!(f, "the file holds {file} elements and cannot be read as {requested}")
            }
            MmapError::OutOfBounds { index, count } => {
                write!(f, "record {index} is out of bounds (count {count})")
            }
        }
    }
}

impl std::error::Error for MmapError {}

/// The one check a typed open makes: the extension must name a file
/// of the given layout whose element type is `T`'s. Returns the file's
/// type.
///
/// A reader never reinterprets: a `.bvecs` file is not opened as f32
/// because the stride happens to divide, and a `.dvec` is not opened
/// as f32 because both are floats. A caller that wants a conversion
/// reads the file at its own type and converts, or uses a converting
/// reader where one exists.
///
/// # Example
///
/// Through the public readers: a `.bvecs` file holds `u8`, and
/// opening it as `f32` is refused with [`MmapError::ElementMismatch`]
/// before the file is mapped, even though the conversion table would
/// allow the widening in a converting reader.
///
/// ```
/// use veks_io::{ElementType, MmapError, MmapReader};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("codes.bvecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # f.write_all(&4i32.to_le_bytes()).unwrap();
/// # f.write_all(&[1u8, 2, 3, 4]).unwrap();
/// # drop(f);
/// let codes = MmapReader::<u8>::open(&path).unwrap(); // native: zero-copy
/// assert_eq!(codes.get_slice(0).unwrap(), &[1, 2, 3, 4]);
///
/// match MmapReader::<f32>::open(&path) {
///     Err(MmapError::ElementMismatch { file, requested }) => {
///         assert_eq!((file, requested), (ElementType::U8, ElementType::F32)); // refused
///     }
///     other => panic!("expected ElementMismatch, got {other:?}"),
/// }
/// ```
pub(crate) fn require_native<T: Element>(path: &Path, container: Container) -> Result<ElementType, MmapError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| MmapError::InvalidFormat(format!("{}: no extension says what it holds", path.display())))?;
    let (file, found) = ElementType::parse_extension(ext)
        .ok_or_else(|| MmapError::InvalidFormat(format!("{}: unrecognised extension .{ext}", path.display())))?;
    if found != container {
        return Err(MmapError::InvalidFormat(format!(
            "{}: a {} file, not a {} file",
            path.display(),
            container_name(found),
            container_name(container)
        )));
    }
    match file.conversion_to(T::TYPE) {
        ElementConversion::Native => Ok(file),
        _ => Err(MmapError::ElementMismatch { file, requested: T::TYPE }),
    }
}

fn container_name(container: Container) -> &'static str {
    match container {
        Container::Scalar => "column",
        Container::Vec => "vector",
        Container::Vvec => "variable-length vector",
    }
}

/// Whether `bytes` starts on a `T` boundary, so a slice of `T` may be
/// borrowed from it.
pub(crate) fn aligned_for<T>(bytes: &[u8]) -> bool {
    (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>())
}

/// Decode `n` elements of `T` from their little-endian bytes into `out`.
/// On a little-endian host this is one copy; elsewhere each element is
/// decoded.
pub(crate) fn decode_into<T: Element>(bytes: &[u8], out: &mut [T]) {
    debug_assert_eq!(bytes.len(), out.len() * T::SIZE);
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `out` is `out.len()` initialised `T`s, every `Element`
        // is a plain number whose in-memory form is its little-endian
        // encoding on this host, and `bytes` holds exactly that many.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        for (chunk, slot) in bytes.chunks_exact(T::SIZE).zip(out.iter_mut()) {
            *slot = T::from_le_bytes(chunk);
        }
    }
}
