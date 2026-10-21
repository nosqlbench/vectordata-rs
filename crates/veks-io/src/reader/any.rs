// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Conversion between element types, readers whose type is chosen at
//! run time, and the open functions that pick a reader by extension.

use std::marker::PhantomData;
use std::sync::Arc;

use veks_datatypes::{Container, Element, ElementConversion, ElementType};

use super::{
    IndexedVvecReader, IoError, ScalarReader, VectorReader, VvecReader, XvecReader, ext_of, infer_elem_size,
    records_to_read, zero,
};
use crate::source::{ByteSource, MappedFile};

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
/// use veks_io::{ConvertingReader, ElementConversion, IoError, VectorReader, XvecReader};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("counts.u16vecs")).unwrap();
/// # for v in [[1u16, 2], [3, 40000]] { f.write_all(&2i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&u16::to_le_bytes(x)).unwrap(); } }
/// # drop(f);
/// # let path = dir.path().join("counts.u16vecs");
/// let native: Arc<dyn VectorReader<u16>> = Arc::new(XvecReader::<u16>::open_path(&path).unwrap());
/// let checked = ConvertingReader::<u16, i16>::new(native.clone()).unwrap();  // same width, other sign: per value
/// assert_eq!(checked.conversion(), ElementConversion::CrossSign);
/// assert_eq!(checked.get(0).unwrap(), vec![1i16, 2]);
/// assert!(matches!(checked.get(1), Err(IoError::ValueOverflow { index: 1, .. })), "40000 is not an i16");
/// let wide = ConvertingReader::<u16, f32>::new(native.clone()).unwrap();      // wider: never fails
/// assert_eq!(wide.get(1).unwrap(), vec![3.0, 40000.0]);
/// assert!(ConvertingReader::<u16, u16>::new(native).is_err(), "never built for a native pair");
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
    fn precache(&self) -> std::io::Result<()> {
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

/// The element types a reader can be erased to and recovered from:
/// every [`Element`]. What lets a function generic over `N` hand its
/// reader back as an [`AnyVectorReader`].
pub trait AnyElement: Element {
    /// `reader` as the [`AnyVectorReader`] variant of its type.
    fn into_any(reader: Arc<dyn VectorReader<Self>>) -> AnyVectorReader;
    /// `reader` as the [`AnyVvecReader`] variant of its type.
    fn into_any_vvec(reader: Arc<dyn VvecReader<Self>>) -> AnyVvecReader;
}

fn downcast<N: Element, T: Element>(reader: Arc<dyn VectorReader<N>>) -> Result<Arc<dyn VectorReader<T>>, IoError> {
    let boxed: Box<dyn std::any::Any> = Box::new(reader);
    boxed.downcast::<Arc<dyn VectorReader<T>>>().map(|b| *b).map_err(|_| IoError::ElementMismatch {
        file: N::TYPE,
        requested: T::TYPE,
        reason: "the reader holds the file's own type; ask for that, or convert",
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
        /// `T` is known to the compiler: the file read at its own type.
        /// Recover the typed reader with [`into_typed`](Self::into_typed),
        /// convert with [`as_type`](Self::as_type), or match on the
        /// variant.
        ///
        /// ```
        /// use veks_io::{AnyVectorReader, ElementType, IoError, VectorReader, bf16, open_vec_native};
        /// # use std::io::Write;
        /// # let dir = tempfile::tempdir().unwrap();
        /// # let mut f = std::fs::File::create(dir.path().join("emb.bf16vecs")).unwrap();
        /// # for v in [[bf16::from_f32(1.5), bf16::from_f32(-0.125)]] { f.write_all(&2i32.to_le_bytes()).unwrap(); for x in v { f.write_all(&x.to_le_bytes()).unwrap(); } }
        /// # drop(f);
        /// # let path = dir.path().join("emb.bf16vecs");
        /// let any = open_vec_native(path.to_str().unwrap()).unwrap();
        /// assert_eq!(any.element_type(), ElementType::BF16);
        /// assert!(matches!(any, AnyVectorReader::BF16(_)));
        /// let as_f64 = any.as_type::<f64>().unwrap();                        // widened: exact, converted on read
        /// assert_eq!(as_f64.get(0).unwrap(), vec![1.5, -0.125]);
        /// let any = open_vec_native(path.to_str().unwrap()).unwrap();
        /// assert!(matches!(any.into_typed::<f32>(), Err(IoError::ElementMismatch { .. })), "into_typed takes the native type only");
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

            /// The reader converted to `T` through a [`ConvertingReader`]:
            /// for a pair the table widens or checks per value, refused
            /// for a native or refused pair.
            pub fn converted_to<T: Element>(self) -> Result<Arc<dyn VectorReader<T>>, IoError> {
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
        /// variable-length file is never converted.
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

/// [`open_vec_native`] over the source `S`.
pub fn open_vec_native_with<S: ByteSource>(source: &str) -> Result<AnyVectorReader, IoError> {
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
pub fn open_vec_as_with<S: ByteSource, T: Element>(source: &str) -> Result<Arc<dyn VectorReader<T>>, IoError> {
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
pub fn open_vec_with<S: ByteSource, T: Element>(source: &str) -> Result<Box<dyn VectorReader<T>>, IoError> {
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
pub fn open_vvec_with<S: ByteSource, T: Element>(source: &str) -> Result<Box<dyn VvecReader<T>>, IoError> {
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
pub fn open_vvec_untyped_with<S: ByteSource>(source: &str) -> Result<Box<dyn VvecReader<u8>>, IoError> {
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
