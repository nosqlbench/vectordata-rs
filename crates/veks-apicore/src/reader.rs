// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The reader contracts: a file's records by ordinal at one element
//! type ([`VectorReader`], [`VvecReader`]), the error every read
//! returns ([`IoError`]), the conversion between element types
//! ([`ConvertingReader`]), and a reader whose type is chosen at run time
//! ([`AnyVectorReader`], [`AnyVvecReader`]).
//!
//! veks-io implements the readers for every layout (`XvecReader`,
//! `IndexedVvecReader`, `ScalarReader`), generic over a
//! [`ByteSource`](crate::ByteSource); vectordata hands them out through
//! its dataset views, windowed and sharded. A reader reads a file at
//! its own type; a different `T` is [`IoError::ElementMismatch`], never
//! a reinterpretation of the bytes.

use std::io;
use std::marker::PhantomData;
use std::sync::Arc;

use veks_datatypes::{Element, ElementConversion, ElementType};

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
    /// address them. Read the file with veks-io's `IndexedVvecReader`.
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
    /// Three `i16` records; a buffer for two reads records 1 and 2 in
    /// one call, and from the last record only what remains. The
    /// default reads record by record through
    /// [`get_slice`](Self::get_slice) or [`get`](Self::get); a reader
    /// over contiguous bytes overrides it with one copy.
    ///
    /// ```
    /// use veks_apicore::{IoError, VectorReader};
    ///
    /// struct Rows(Vec<[i16; 2]>);
    /// impl VectorReader<i16> for Rows {
    ///     fn dim(&self) -> usize { 2 }
    ///     fn count(&self) -> usize { self.0.len() }
    ///     fn get(&self, i: usize) -> Result<Vec<i16>, IoError> {
    ///         self.0.get(i).map(|r| r.to_vec()).ok_or(IoError::OutOfBounds(i))
    ///     }
    ///     fn get_slice(&self, i: usize) -> Option<&[i16]> { self.0.get(i).map(|r| &r[..]) }
    /// }
    ///
    /// let short = Rows(vec![[1, -1], [200, -200], [i16::MAX, i16::MIN]]);
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
    /// (veks-io's `ScalarReader`), or a window over one. `None` for xvec files,
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
///
/// An implementation answers a record's length and bytes; typed records
/// come with the trait:
///
/// ```
/// use veks_apicore::{IoError, VvecReader};
///
/// /// Ragged rows of i32 held in memory.
/// struct Ragged(Vec<Vec<i32>>);
///
/// impl VvecReader<i32> for Ragged {
///     fn count(&self) -> usize {
///         self.0.len()
///     }
///     fn dim_at(&self, i: usize) -> Result<usize, IoError> {
///         self.0.get(i).map(Vec::len).ok_or(IoError::OutOfBounds(i))
///     }
///     fn get_bytes(&self, i: usize) -> Result<Vec<u8>, IoError> {
///         let row = self.0.get(i).ok_or(IoError::OutOfBounds(i))?;
///         Ok(row.iter().flat_map(|v| v.to_le_bytes()).collect())
///     }
/// }
///
/// let r = Ragged(vec![vec![1], vec![2, 3, 4]]);
/// assert_eq!((r.dim_at(1).unwrap(), r.get(1).unwrap()), (3, vec![2, 3, 4]));
/// ```
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
#[inline]
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

/// The elements `bytes` encodes, as a new `Vec`: one allocation and,
/// on a little-endian host, one copy. Never a loop the compiler must
/// inline to make fast (the IO-4 measurement of an out-of-line
/// per-element decode: 2.4× slower).
#[inline]
pub fn decode_vec<T: Element>(bytes: &[u8]) -> Vec<T> {
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

/// The zero of `T`, a buffer's fill before a conversion writes it.
#[inline]
fn zero<T: Element>() -> T {
    T::from_le_bytes(&[0u8; 16][..T::SIZE])
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
/// use veks_apicore::{ConvertingReader, ElementConversion, IoError, VectorReader};
/// # struct Rows(Vec<[u16; 2]>);
/// # impl VectorReader<u16> for Rows {
/// #     fn dim(&self) -> usize { 2 }
/// #     fn count(&self) -> usize { self.0.len() }
/// #     fn get(&self, i: usize) -> Result<Vec<u16>, IoError> { self.0.get(i).map(|r| r.to_vec()).ok_or(IoError::OutOfBounds(i)) }
/// # }
/// let native: Arc<dyn VectorReader<u16>> = Arc::new(Rows(vec![[1, 2], [3, 40000]]));
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
        /// use std::sync::Arc;
        /// use veks_apicore::{AnyElement, AnyVectorReader, ElementType, IoError, VectorReader, bf16};
        /// # struct Rows(Vec<[bf16; 2]>);
        /// # impl VectorReader<bf16> for Rows {
        /// #     fn dim(&self) -> usize { 2 }
        /// #     fn count(&self) -> usize { self.0.len() }
        /// #     fn get(&self, i: usize) -> Result<Vec<bf16>, IoError> { self.0.get(i).map(|r| r.to_vec()).ok_or(IoError::OutOfBounds(i)) }
        /// # }
        /// let reader = || Arc::new(Rows(vec![[bf16::from_f32(1.5), bf16::from_f32(-0.125)]]));
        /// let any = bf16::into_any(reader());                              // what an open by extension returns
        /// assert_eq!(any.element_type(), ElementType::BF16);
        /// assert!(matches!(any, AnyVectorReader::BF16(_)));
        /// let as_f64 = any.as_type::<f64>().unwrap();                        // widened: exact, converted on read
        /// assert_eq!(as_f64.get(0).unwrap(), vec![1.5, -0.125]);
        /// let any = bf16::into_any(reader());
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
            #[inline]
            pub fn element_type(&self) -> ElementType {
                match self { $(Self::$variant(_) => ElementType::$variant,)* }
            }

            /// Number of records.
            #[inline]
            pub fn count(&self) -> usize {
                match self { $(Self::$variant(r) => r.count(),)* }
            }

            /// Elements per record.
            #[inline]
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
            #[inline]
            pub fn element_type(&self) -> ElementType {
                match self { $(Self::$variant(_) => ElementType::$variant,)* }
            }

            /// Number of records.
            #[inline]
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
    F16 veks_datatypes::f16, BF16 veks_datatypes::bf16, F32 f32, F64 f64,
}
