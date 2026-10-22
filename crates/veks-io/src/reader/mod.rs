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

use veks_datatypes::{Container, Element, ElementConversion, ElementType};

mod any;
mod scalar;
mod vvec;
mod xvec;

pub use veks_apicore::reader::{
    AnyElement, AnyVectorReader, AnyVvecReader, ConvertingReader, DIM_UNDEFINED, IoError, VectorReader, VvecReader,
    records_to_read,
};
pub(crate) use veks_apicore::reader::decode_vec;
pub use any::{open_vec, open_vec_as, open_vec_as_with, open_vec_native, open_vec_native_with, open_vec_with};
pub use any::{open_vvec, open_vvec_untyped, open_vvec_untyped_with, open_vvec_with};
pub use scalar::ScalarReader;
pub use vvec::{IndexedVvecReader, load_offsets, remove_vvec_index};
pub(crate) use vvec::write_offset_index;
pub use xvec::{StreamReclaim, XvecReader};


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
