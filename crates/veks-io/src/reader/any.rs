// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The open functions that pick a reader by extension, at the file's
//! own type or converted. The conversion and run-time-typed readers
//! they return are veks-apicore's.

use std::sync::Arc;

use veks_datatypes::{Container, Element, ElementType};

use super::{
    AnyElement, AnyVectorReader, IndexedVvecReader, IoError, ScalarReader, VectorReader, VvecReader, XvecReader,
    ext_of, infer_elem_size,
};
use crate::source::{MappedFile, OpenSource};

/// [`open_vec_native`] over the source `S`.
pub fn open_vec_native_with<S: OpenSource>(source: &str) -> Result<AnyVectorReader, IoError> {
    let ext = ext_of(source);
    let (file, container) = ElementType::parse_extension(ext)
        .ok_or_else(|| IoError::InvalidFormat(format!("cannot infer element type from extension '.{ext}'")))?;
    veks_datatypes::dispatch_element!(file, N => Ok(match container {
        Container::Scalar => <N as AnyElement>::into_any(Arc::new(ScalarReader::<N, S>::open(source)?)),
        _ => <N as AnyElement>::into_any(Arc::new(XvecReader::<N, S>::open(source)?)),
    }))
}

/// Open a vector or column file at the type its extension names, for a
/// caller that does not know the type ahead of time. A column opens as
/// a [`ScalarReader`] (`dim() == 1`), anything else as an
/// [`XvecReader`].
///
/// ```
/// use veks_io::{AnyVectorReader, ElementType, VectorReader, open_vec_native};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("scores.i16vecs")).unwrap();
/// # for v in [[-3i16, 7], [i16::MIN, i16::MAX]] { f.write_all(&2i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&i16::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// let any = open_vec_native(dir.path().join("scores.i16vecs").to_str().unwrap()).unwrap();
/// assert_eq!((any.element_type(), any.count(), any.dim()), (ElementType::I16, 2, 2));
/// match any {
///     AnyVectorReader::I16(r) => assert_eq!(r.get(1).unwrap(), vec![i16::MIN, i16::MAX]),
///     other => panic!("an .i16vecs is i16, not {}", other.element_type()),
/// }
/// ```
pub fn open_vec_native(source: &str) -> Result<AnyVectorReader, IoError> {
    open_vec_native_with::<MappedFile>(source)
}

/// [`open_vec_as`] over the source `S`.
pub fn open_vec_as_with<S: OpenSource, T: Element>(source: &str) -> Result<Arc<dyn VectorReader<T>>, IoError> {
    open_vec_native_with::<S>(source)?.as_type::<T>()
}

/// Open a vector or column file as `T`: in place when `T` is the type
/// the extension names, converted by [`ElementType::conversion_to`]
/// where the table allows (an `.mvec` as f32, a `.bvecs` as i32),
/// refused otherwise. [`open_vec`] is the native-only form.
///
/// ```
/// use veks_io::{ElementConversion, ElementType, IoError, VectorReader, f16, open_vec_as};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("half.mvecs")).unwrap();
/// # for v in [[f16::from_f32(0.5), f16::from_f32(-2.0)]] { f.write_all(&2i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&f16::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("ids.ivecs")).unwrap();
/// # for v in [[5i32], [-1]] { f.write_all(&1i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&i32::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("codes.bvecs")).unwrap();
/// # for v in [[100u8], [200]] { f.write_all(&1i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&u8::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("base.fvecs")).unwrap();
/// # for v in [[1.0f32]] { f.write_all(&1i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&f32::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// # let p = |n: &str| dir.path().join(n).to_str().unwrap().to_string();
/// // Widening: an f16 file as f32 — exact, one conversion per element read.
/// let wide = open_vec_as::<f32>(&p("half.mvecs")).unwrap();
/// assert_eq!((wide.conversion(), wide.native_element_type()), (ElementConversion::Widening, Some(ElementType::F16)));
/// assert_eq!(wide.get(0).unwrap(), vec![0.5, -2.0]);
/// assert!(wide.get_slice(0).is_none(), "nothing of f32 exists to borrow");
///
/// // Widening across integers: i32 ground truth as i64, -1 included.
/// assert_eq!(open_vec_as::<i64>(&p("ids.ivecs")).unwrap().get(1).unwrap(), vec![-1i64]);
///
/// // CrossSign: u8 codes as i8 — read what fits, refuse the value that does not, at its ordinal.
/// let checked = open_vec_as::<i8>(&p("codes.bvecs")).unwrap();
/// assert_eq!(checked.get(0).unwrap(), vec![100i8]);
/// assert!(matches!(checked.get(1), Err(IoError::ValueOverflow { index: 1, .. })));
///
/// // Refused: f32 as u8 is not lossless, and never a reinterpretation.
/// assert!(matches!(open_vec_as::<u8>(&p("base.fvecs")), Err(IoError::ElementMismatch { file: ElementType::F32, requested: ElementType::U8, .. })));
/// ```
pub fn open_vec_as<T: Element>(source: &str) -> Result<Arc<dyn VectorReader<T>>, IoError> {
    open_vec_as_with::<MappedFile, T>(source)
}

/// [`open_vec`] over the source `S`.
pub fn open_vec_with<S: OpenSource, T: Element>(source: &str) -> Result<Box<dyn VectorReader<T>>, IoError> {
    Ok(Box::new(XvecReader::<T, S>::open(source)?))
}

/// Open a uniform vector file for typed random access, at the type its
/// extension names only.
///
/// ```
/// use veks_io::{ElementType, IoError, open_vec};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("ords.u32vecs")).unwrap();
/// # for v in [[1u32, 2, u32::MAX]] { f.write_all(&3i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&u32::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// # let path = dir.path().join("ords.u32vecs");
/// let reader = open_vec::<u32>(path.to_str().unwrap()).unwrap();
/// assert_eq!(reader.get(0).unwrap(), vec![1, 2, u32::MAX]);
/// // Same width, other sign: refused here; open_vec_as checks it per value.
/// assert!(matches!(open_vec::<i32>(path.to_str().unwrap()), Err(IoError::ElementMismatch { file: ElementType::U32, requested: ElementType::I32, .. })));
/// ```
pub fn open_vec<T: Element>(source: &str) -> Result<Box<dyn VectorReader<T>>, IoError> {
    open_vec_with::<MappedFile, T>(source)
}

/// [`open_vvec`] over the source `S`.
pub fn open_vvec_with<S: OpenSource, T: Element>(source: &str) -> Result<Box<dyn VvecReader<T>>, IoError> {
    Ok(Box::new(IndexedVvecReader::<T, S>::open(source)?))
}

/// Open a variable-length vector file for typed random access.
///
/// ```
/// use veks_io::{VvecReader, open_vvec};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("matches.i64vvecs")).unwrap();
/// # for v in [vec![7i64, -7], vec![i64::MAX]] { f.write_all(&(v.len() as i32).to_le_bytes()).unwrap(); for x in v { f.write_all(&x.to_le_bytes()).unwrap(); } }
/// # drop(f);
/// let r = open_vvec::<i64>(dir.path().join("matches.i64vvecs").to_str().unwrap()).unwrap();
/// assert_eq!((r.count(), r.dim_at(0).unwrap(), r.dim_at(1).unwrap()), (2, 2, 1));
/// assert_eq!(r.get(1).unwrap(), vec![i64::MAX]);
/// ```
pub fn open_vvec<T: Element>(source: &str) -> Result<Box<dyn VvecReader<T>>, IoError> {
    open_vvec_with::<MappedFile, T>(source)
}

/// [`open_vvec_untyped`] over the source `S`.
pub fn open_vvec_untyped_with<S: OpenSource>(source: &str) -> Result<Box<dyn VvecReader<u8>>, IoError> {
    let ext = ext_of(source);
    let elem_size = infer_elem_size(ext);
    if elem_size == 0 {
        return Err(IoError::InvalidFormat(format!("cannot infer element size from extension '.{ext}'")));
    }
    Ok(Box::new(IndexedVvecReader::<u8, S>::from_source_sized(S::open(source)?, source, elem_size)?))
}

/// Open a variable-length vector file as raw `u8` records, for a caller
/// that decodes them itself.
pub fn open_vvec_untyped(source: &str) -> Result<Box<dyn VvecReader<u8>>, IoError> {
    open_vvec_untyped_with::<MappedFile>(source)
}
