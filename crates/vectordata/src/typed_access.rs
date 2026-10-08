// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Typed data access with runtime type negotiation.
//!
//! [`TypedReader<T>`] opens vector and scalar files with compile-time
//! type safety and runtime width/signedness validation. The transport
//! choice (local mmap, merkle-cached remote, direct HTTP) is hidden
//! inside the crate-private `Storage` abstraction.
//!
//! # Access modes
//!
//! - **Native**: open with the exact native type → zero-copy `&[T]` access
//! - **Widening**: open a narrower file as a wider type → checked conversion, always succeeds
//! - **Cross-sign same width**: u8↔i8, u16↔i16, etc. → checked per-value, fails on overflow
//! - **Narrowing**: rejected at open time
//!
//! # Examples
//!
//! ```rust,no_run
//! use vectordata::typed_access::{ElementType, TypedReader};
//!
//! // Open with native type — zero-copy
//! let reader = TypedReader::<u8>::open("metadata.u8").unwrap();
//! let val: u8 = reader.get_native(42).unwrap();
//!
//! // Open with wider type — always succeeds
//! let reader = TypedReader::<i32>::open("metadata.u8").unwrap();
//! let val: i32 = reader.get_value(42).unwrap();
//! ```

use std::path::Path;
use std::sync::Arc;

use crate::storage::Storage;

// ═══════════════════════════════════════════════════════════════════════
// ElementType — file-format → native-type mapping
// ═══════════════════════════════════════════════════════════════════════

/// Element type of a data file, inferred from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// Unsigned 8-bit integer (`.u8`, `.bvec`, `.u8vec`, ...).
    U8,
    /// Signed 8-bit integer (`.i8`, `.i8vec`, ...).
    I8,
    /// Unsigned 16-bit integer (`.u16`, `.u16vec`, ...).
    U16,
    /// Signed 16-bit integer (`.i16`, `.svec`, `.i16vec`, ...).
    I16,
    /// Unsigned 32-bit integer (`.u32`, `.u32vec`, ...).
    U32,
    /// Signed 32-bit integer (`.i32`, `.ivec`, `.i32vec`, ...).
    I32,
    /// Unsigned 64-bit integer (`.u64`, `.u64vec`, ...).
    U64,
    /// Signed 64-bit integer (`.i64`, `.i64vec`, ...).
    I64,
    /// IEEE 754 half-precision float (`.mvec`, `.f16vec`, ...).
    F16,
    /// IEEE 754 single-precision float (`.fvec`, `.f32vec`, ...).
    F32,
    /// IEEE 754 double-precision float (`.dvec`, `.f64vec`, ...).
    F64,
}

impl ElementType {
    /// Infer the element type from the extension of a local path.
    ///
    /// Errors when the path has no extension or the extension is not
    /// one [`ElementType::from_extension`] recognises.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no extension: {}", path.display()))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}': {}", path.display()))
    }

    /// Infer the element type from the extension of a URL's path
    /// (the text after the last `.`).
    ///
    /// Errors when that text is not a recognised extension.
    pub fn from_url(url: &url::Url) -> Result<Self, String> {
        let path = url.path();
        let ext = path.rsplit('.').next()
            .ok_or_else(|| format!("no extension in URL: {url}"))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}' in URL: {url}"))
    }

    /// Map a file extension (without the leading `.`, case-insensitive)
    /// to its element type.
    ///
    /// Accepts the bare scalar forms (`u8`, `i32`, ...) and every
    /// uniform and variable-length vector spelling (`fvec`, `f32vecs`,
    /// `ivvec`, ...). Returns `None` for anything else.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "u8" => Some(Self::U8),
            "i8" => Some(Self::I8),
            "u16" => Some(Self::U16),
            "i16" => Some(Self::I16),
            "u32" => Some(Self::U32),
            "i32" => Some(Self::I32),
            "u64" => Some(Self::U64),
            "i64" => Some(Self::I64),
            "bvec" | "bvecs" | "u8vec" | "u8vecs" | "bvvec" | "bvvecs" | "u8vvec" | "u8vvecs" => Some(Self::U8),
            "i8vec" | "i8vecs" | "i8vvec" | "i8vvecs" => Some(Self::I8),
            "svec" | "svecs" | "i16vec" | "i16vecs" | "svvec" | "svvecs" | "i16vvec" | "i16vvecs" => Some(Self::I16),
            "u16vec" | "u16vecs" | "u16vvec" | "u16vvecs" => Some(Self::U16),
            "ivec" | "ivecs" | "i32vec" | "i32vecs" | "ivvec" | "ivvecs" | "i32vvec" | "i32vvecs" => Some(Self::I32),
            "u32vec" | "u32vecs" | "u32vvec" | "u32vvecs" => Some(Self::U32),
            "i64vec" | "i64vecs" | "i64vvec" | "i64vvecs" => Some(Self::I64),
            "u64vec" | "u64vecs" | "u64vvec" | "u64vvecs" => Some(Self::U64),
            "fvec" | "fvecs" | "fvvec" | "fvvecs" | "f32vec" | "f32vecs" | "f32vvec" | "f32vvecs" => Some(Self::F32),
            "mvec" | "mvecs" | "mvvec" | "mvvecs" | "f16vec" | "f16vecs" | "f16vvec" | "f16vvecs" => Some(Self::F16),
            "dvec" | "dvecs" | "dvvec" | "dvvecs" | "f64vec" | "f64vecs" | "f64vvec" | "f64vvecs" => Some(Self::F64),
            _ => None,
        }
    }

    /// Size of one element in bytes: 1, 2, 4 or 8.
    pub fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Whether `path` names a scalar facet — raw packed values with no
    /// per-record header. See `crate::io::is_scalar_ext`, the single
    /// table both this and the URL form answer from.
    pub fn is_scalar_format(path: impl AsRef<Path>) -> bool {
        let ext = path.as_ref().extension().and_then(|e| e.to_str()).unwrap_or("");
        crate::io::is_scalar_ext(ext)
    }

    /// Short lowercase type name (`"u8"`, `"f32"`, ...), as used by
    /// `Display` and in error messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8", Self::I8 => "i8",
            Self::U16 => "u16", Self::I16 => "i16",
            Self::U32 => "u32", Self::I32 => "i32",
            Self::U64 => "u64", Self::I64 => "i64",
            Self::F16 => "f16", Self::F32 => "f32", Self::F64 => "f64",
        }
    }

    /// Whether a file of this type may be opened as a target type of
    /// `target_width` bytes — true unless that would narrow. Signedness
    /// is not considered here; cross-sign values are checked per read.
    pub fn can_open_as(self, target_width: usize) -> bool {
        target_width >= self.byte_width()
    }
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Errors
// ═══════════════════════════════════════════════════════════════════════

/// Failure opening or reading a [`TypedReader`].
#[derive(Debug)]
pub enum TypedAccessError {
    /// Target type is narrower than native type.
    Narrowing {
        /// Element type of the file.
        native: ElementType,
        /// Name of the requested Rust element type.
        target: &'static str,
    },
    /// Value at ordinal doesn't fit in the target type.
    ValueOverflow {
        /// Record ordinal being read.
        ordinal: usize,
        /// The stored value, widened to `i128`.
        value: i128,
        /// Name of the requested Rust element type.
        target: &'static str,
    },
    /// I/O error.
    Io(String),
}

impl std::fmt::Display for TypedAccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Narrowing { native, target } =>
                write!(f, "cannot open {native} file as {target} (narrowing)"),
            Self::ValueOverflow { ordinal, value, target } =>
                write!(f, "value {value} at ordinal {ordinal} does not fit in {target}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TypedAccessError {}

// ═══════════════════════════════════════════════════════════════════════
// TypedElement trait
// ═══════════════════════════════════════════════════════════════════════

/// Rust integer type a [`TypedReader`] can yield.
///
/// Implemented for `u8`..`i64`. Values travel through `i128` so that
/// every supported native type can be widened or range-checked into
/// every target type.
pub trait TypedElement: Copy + Send + Sync + 'static {
    /// Short lowercase type name (`"u8"`, `"i32"`, ...), used in errors.
    fn type_name() -> &'static str;
    /// Size of the type in bytes.
    fn width() -> usize;
    /// Convert from `i128`, or `None` when the value is out of range.
    fn from_i128(val: i128) -> Option<Self>;
    /// Widen to `i128` losslessly.
    fn to_i128(self) -> i128;
}

macro_rules! impl_typed_element {
    ($t:ty, $name:expr) => {
        impl TypedElement for $t {
            fn type_name() -> &'static str { $name }
            fn width() -> usize { std::mem::size_of::<$t>() }
            fn from_i128(val: i128) -> Option<Self> { <$t>::try_from(val).ok() }
            fn to_i128(self) -> i128 { self as i128 }
        }
    };
}

impl_typed_element!(u8,  "u8");
impl_typed_element!(i8,  "i8");
impl_typed_element!(u16, "u16");
impl_typed_element!(i16, "i16");
impl_typed_element!(u32, "u32");
impl_typed_element!(i32, "i32");
impl_typed_element!(u64, "u64");
impl_typed_element!(i64, "i64");

/// URL form of [`ElementType::is_scalar_format`] — same table, applied
/// to a URL path where `Path::extension` cannot be used.
///
/// It must stay the same table: a local open and a remote open of the
/// same facet that disagree here read the same bytes two different
/// ways, one of them wrong.
fn is_scalar_url_path(path: &str) -> bool {
    let ext = match path.rfind('.') {
        Some(p) => &path[p + 1..],
        None => return false,
    };
    crate::io::is_scalar_ext(ext)
}

fn read_native_value(data: &[u8], native: ElementType) -> i128 {
    match native {
        ElementType::U8  => data[0] as i128,
        ElementType::I8  => data[0] as i8 as i128,
        ElementType::U16 => u16::from_le_bytes([data[0], data[1]]) as i128,
        ElementType::I16 => i16::from_le_bytes([data[0], data[1]]) as i128,
        ElementType::U32 => u32::from_le_bytes(data[..4].try_into().unwrap()) as i128,
        ElementType::I32 => i32::from_le_bytes(data[..4].try_into().unwrap()) as i128,
        ElementType::U64 => u64::from_le_bytes(data[..8].try_into().unwrap()) as i128,
        ElementType::I64 => i64::from_le_bytes(data[..8].try_into().unwrap()) as i128,
        ElementType::F16 | ElementType::F32 | ElementType::F64 => 0, // float not supported here
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TypedReader<T>
// ═══════════════════════════════════════════════════════════════════════

/// Typed reader for vector or scalar data files.
///
/// Single concrete struct over the crate's internal `Storage`. The
/// transport choice (local mmap, merkle-cached remote, direct HTTP)
/// is hidden inside the storage and selected by `Storage::open`.
pub struct TypedReader<T: TypedElement> {
    storage: Arc<Storage>,
    /// The remaining shards, when this facet is a series.
    ///
    /// `None` for a single file — the case that must stay free of any
    /// added indirection (SH-73). The offset arithmetic below is
    /// unchanged either way; a series only changes *which* storage it
    /// is applied to, and at which ordinal within that file (SH-64).
    series: Option<Arc<crate::view::Series>>,
    native_type: ElementType,
    native_width: usize,
    is_scalar: bool,
    dim: usize,
    count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TypedElement> TypedReader<T> {
    /// Open a local file by path. Inferred element type from extension.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TypedAccessError> {
        let path = path.as_ref();
        let native_type = ElementType::from_path(path)
            .map_err(TypedAccessError::Io)?;
        if T::width() < native_type.byte_width() {
            return Err(TypedAccessError::Narrowing { native: native_type, target: T::type_name() });
        }
        let storage = Storage::open_path(path)
            .map_err(|e| TypedAccessError::Io(format!("open {}: {e}", path.display())))?;
        let is_scalar = ElementType::is_scalar_format(path);
        Self::from_storage(storage, native_type, is_scalar)
    }

    /// Open a remote URL with cache-first dispatch (merkle cache when
    /// `.mref` is published, direct HTTP otherwise).
    pub fn open_url(url: url::Url, native_type: ElementType) -> Result<Self, TypedAccessError> {
        if T::width() < native_type.byte_width() {
            return Err(TypedAccessError::Narrowing { native: native_type, target: T::type_name() });
        }
        let is_scalar = is_scalar_url_path(url.path());
        let storage = Storage::open_url(url.clone())
            .map_err(|e| TypedAccessError::Io(format!("open_url {url}: {e}")))?;
        Self::from_storage(storage, native_type, is_scalar)
    }

    /// Open from a path-or-URL string, dispatching automatically.
    pub fn open_auto(path_or_url: &str, native_type: ElementType) -> Result<Self, TypedAccessError> {
        if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
            let url = url::Url::parse(path_or_url)
                .map_err(|e| TypedAccessError::Io(format!("invalid URL: {e}")))?;
            Self::open_url(url, native_type)
        } else {
            Self::open(path_or_url)
        }
    }

    /// **Crate-internal**: build from an already-opened storage.
    /// Used by [`crate::view::TestDataView::open_facet_typed`] so a
    /// single shared storage backs both the typed and the
    /// uniform-vector view of the same facet.
    pub(crate) fn from_storage(
        storage: Arc<Storage>,
        native_type: ElementType,
        is_scalar: bool,
    ) -> Result<Self, TypedAccessError> {
        let native_width = native_type.byte_width();
        let total_size = storage.total_size();
        let (dim, count) = if is_scalar {
            (1, (total_size / native_width as u64) as usize)
        } else if total_size < 4 {
            (1, 0)
        } else {
            let header = storage.read_bytes(0, 4)
                .map_err(|e| TypedAccessError::Io(format!("read header: {e}")))?;
            let dim = i32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
            let record_bytes = 4 + dim * native_width;
            let count = if record_bytes > 0 { (total_size / record_bytes as u64) as usize } else { 0 };
            (dim, count)
        };
        Ok(Self {
            storage, series: None, native_type, native_width, is_scalar, dim, count,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Build over a multi-file series.
    ///
    /// Every shard shares the element type and dimension (SRD invariant
    /// 4), so the shape comes from the first file and the count comes
    /// from the declaration rather than from dividing bytes.
    pub(crate) fn from_series(
        series: Arc<crate::view::Series>,
        native_type: ElementType,
        is_scalar: bool,
    ) -> Result<Self, TypedAccessError> {
        let first = series
            .file(0)
            .map_err(|e| TypedAccessError::Io(e.to_string()))?
            .clone();
        let count = series.shards().count() as usize;
        let probe = Self::from_storage(first.clone(), native_type, is_scalar)?;
        Ok(Self {
            storage: first,
            series: Some(series),
            native_type,
            native_width: native_type.byte_width(),
            is_scalar,
            dim: probe.dim,
            count,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The storage holding `ordinal`, and the ordinal within that
    /// file's own numbering (SH-64).
    ///
    /// For a single file this is the identity — the ordinal is already
    /// a file ordinal — so the common path pays one `Option` test.
    fn at(&self, ordinal: usize) -> Result<(Arc<Storage>, usize), TypedAccessError> {
        match &self.series {
            None => Ok((self.storage.clone(), ordinal)),
            Some(s) => {
                let located = s.shards().locate(ordinal as u64).ok_or_else(|| {
                    TypedAccessError::Io(format!(
                        "ordinal {ordinal} out of range (count {})",
                        self.count
                    ))
                })?;
                let file = s
                    .file_index_of_shard(located.shard)
                    .map_err(|e| TypedAccessError::Io(e.to_string()))?;
                let storage = s.file(file).map_err(|e| TypedAccessError::Io(e.to_string()))?;
                Ok((storage, located.file_ordinal as usize))
            }
        }
    }

    /// Read from a named storage, so a series can read from the file
    /// that actually holds the ordinal.
    fn read_from(
        storage: &Storage,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>, TypedAccessError> {
        storage
            .read_bytes(offset as u64, len as u64)
            .map_err(|e| TypedAccessError::Io(e.to_string()))
    }

    fn mmap_slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        self.storage.mmap_slice(offset as u64, len as u64)
    }

    /// Element type stored in the file (from its extension).
    pub fn native_type(&self) -> ElementType { self.native_type }
    /// Whether `T` has the same byte width as the native type, i.e. no
    /// widening is involved.
    pub fn is_native(&self) -> bool { T::width() == self.native_width }
    /// Number of records (values, for a scalar file); for a series, the
    /// declared total across all shards.
    pub fn count(&self) -> usize { self.count }
    /// Elements per record: 1 for a scalar file, otherwise the
    /// dimension from the first record header.
    pub fn dim(&self) -> usize { self.dim }

    /// Force-download every byte into the local cache. No-op for
    /// local files and for non-cacheable HTTP. Idempotent.
    pub fn precache(&self) -> std::io::Result<()> {
        match &self.series {
            None => self.storage.precache(),
            Some(s) => {
                for i in 0..s.file_count() {
                    s.file(i)
                        .map_err(|e| std::io::Error::other(e.to_string()))?
                        .precache()?;
                }
                Ok(())
            }
        }
    }

    /// Whether all bytes are locally accessible without network round-trips.
    pub fn is_complete(&self) -> bool {
        match &self.series {
            None => self.storage.is_complete(),
            Some(s) => (0..s.file_count()).all(|i| s.file(i).is_ok_and(|f| f.is_complete())),
        }
    }

    /// Get a single value from a scalar file (dim=1), with checked conversion.
    pub fn get_value(&self, ordinal: usize) -> Result<T, TypedAccessError> {
        if ordinal >= self.count {
            return Err(TypedAccessError::Io(
                format!("ordinal {ordinal} out of range (count {})", self.count)));
        }
        // The offset formula is unchanged by sharding; only *which*
        // file it applies to, and at which ordinal within that file
        // (SH-64). For a single facet `at` is the identity.
        let (storage, at) = self.at(ordinal)?;
        let offset = if self.is_scalar {
            at * self.native_width
        } else {
            let record_bytes = 4 + self.dim * self.native_width;
            at * record_bytes + 4
        };
        let bytes = Self::read_from(&storage, offset, self.native_width)?;
        let val = read_native_value(&bytes, self.native_type);
        T::from_i128(val).ok_or(TypedAccessError::ValueOverflow {
            ordinal, value: val, target: T::type_name(),
        })
    }

    /// Get a record as `Vec<T>`, with checked conversion per element.
    pub fn get_record(&self, ordinal: usize) -> Result<Vec<T>, TypedAccessError> {
        if ordinal >= self.count {
            return Err(TypedAccessError::Io(
                format!("ordinal {ordinal} out of range (count {})", self.count)));
        }
        let (storage, at) = self.at(ordinal)?;
        let offset = if self.is_scalar {
            at * self.native_width * self.dim
        } else {
            let record_bytes = 4 + self.dim * self.native_width;
            at * record_bytes + 4
        };
        let total_bytes = self.dim * self.native_width;
        let bytes = Self::read_from(&storage, offset, total_bytes)?;
        let mut result = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            let elem_offset = d * self.native_width;
            let val = read_native_value(&bytes[elem_offset..], self.native_type);
            result.push(T::from_i128(val).ok_or(TypedAccessError::ValueOverflow {
                ordinal, value: val, target: T::type_name(),
            })?);
        }
        Ok(result)
    }
}

// ─── Native zero-copy access (only when T matches native type) ──────────

impl<T: TypedElement> TypedReader<T> {
    /// The mapped bytes at `offset` of a single-file facet. `None` for a
    /// series — an offset is only meaningful within the file that holds
    /// the ordinal, which `get_value` finds — and for storage that is
    /// not memory-mapped.
    fn single_file_slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if self.series.is_some() {
            return None;
        }
        self.mmap_slice(offset, len)
    }
}

impl TypedReader<u8> {
    /// Native read of a record's first element (the value itself for a
    /// scalar file), zero-copy when the file is memory-mapped and read
    /// through [`get_value`](Self::get_value) otherwise — including for
    /// a sharded facet.
    pub fn get_native(&self, ordinal: usize) -> Result<u8, TypedAccessError> {
        debug_assert!(self.native_type == ElementType::U8);
        let offset = if self.is_scalar { ordinal } else { 4 + ordinal * (4 + self.dim) };
        match self.single_file_slice(offset, 1) {
            Some(slice) if ordinal < self.count => Ok(slice[0]),
            _ => self.get_value(ordinal),
        }
    }

    /// Zero-copy slice of a record's native bytes. `None` when the
    /// storage is not memory-mapped, when `ordinal` is out of range, and
    /// for a sharded facet; read through
    /// [`get_record`](Self::get_record) then.
    pub fn get_native_slice(&self, ordinal: usize) -> Option<&[u8]> {
        debug_assert!(self.native_type == ElementType::U8);
        if ordinal >= self.count {
            return None;
        }
        let (offset, len) = if self.is_scalar {
            (ordinal * self.dim, self.dim)
        } else {
            (ordinal * (4 + self.dim) + 4, self.dim)
        };
        self.single_file_slice(offset, len)
    }
}

impl TypedReader<i32> {
    /// Native read of a record's first element (the value itself for a
    /// scalar file), zero-copy when the file is memory-mapped and read
    /// through [`get_value`](Self::get_value) otherwise — including for
    /// a sharded facet.
    pub fn get_native(&self, ordinal: usize) -> Result<i32, TypedAccessError> {
        debug_assert!(self.native_type == ElementType::I32);
        let offset = if self.is_scalar {
            ordinal * 4
        } else {
            ordinal * (4 + self.dim * 4) + 4
        };
        match self.single_file_slice(offset, 4) {
            Some(slice) if ordinal < self.count => {
                Ok(i32::from_le_bytes(slice.try_into().expect("four bytes")))
            }
            _ => self.get_value(ordinal),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// dispatch_typed! macro — branch on a file's native element type
// ═══════════════════════════════════════════════════════════════════════

/// Dispatch on a file's native element type, opening it as that type
/// and binding the resulting reader to `$reader`.
#[macro_export]
macro_rules! dispatch_typed {
    ($path:expr, $reader:ident => $body:expr) => {{
        let etype = $crate::typed_access::ElementType::from_path($path)
            .map_err(|e| e.to_string())?;
        match etype {
            $crate::typed_access::ElementType::U8  => { let $reader = $crate::typed_access::TypedReader::<u8> ::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::I8  => { let $reader = $crate::typed_access::TypedReader::<i8> ::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::U16 => { let $reader = $crate::typed_access::TypedReader::<u16>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::I16 => { let $reader = $crate::typed_access::TypedReader::<i16>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::U32 => { let $reader = $crate::typed_access::TypedReader::<u32>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::I32 => { let $reader = $crate::typed_access::TypedReader::<i32>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::U64 => { let $reader = $crate::typed_access::TypedReader::<u64>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::I64 => { let $reader = $crate::typed_access::TypedReader::<i64>::open($path).map_err(|e| e.to_string())?; $body }
            _ => Err(format!("unsupported element type {:?} for typed access", etype)),
        }
    }};
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_tmp() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    #[test]
    fn detect_scalar_types() {
        assert_eq!(ElementType::from_extension("u8"), Some(ElementType::U8));
        assert_eq!(ElementType::from_extension("i32"), Some(ElementType::I32));
        assert_eq!(ElementType::from_extension("u64"), Some(ElementType::U64));
    }

    #[test]
    fn detect_vector_types() {
        assert_eq!(ElementType::from_extension("ivec"), Some(ElementType::I32));
        assert_eq!(ElementType::from_extension("bvec"), Some(ElementType::U8));
        assert_eq!(ElementType::from_extension("fvec"), Some(ElementType::F32));
        assert_eq!(ElementType::from_extension("u16vec"), Some(ElementType::U16));
    }

    #[test]
    fn scalar_u8_native() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [0u8, 42, 127, 255]).unwrap();
        let r = TypedReader::<u8>::open(&path).unwrap();
        assert_eq!(r.count(), 4);
        assert_eq!(r.dim(), 1);
        assert!(r.is_native());
        assert_eq!(r.get_native(0).unwrap(), 0);
        assert_eq!(r.get_native(3).unwrap(), 255);
    }

    #[test]
    fn scalar_u8_as_i32_widening() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [0u8, 42, 255]).unwrap();
        let r = TypedReader::<i32>::open(&path).unwrap();
        assert!(!r.is_native());
        assert_eq!(r.get_value(0).unwrap(), 0i32);
        assert_eq!(r.get_value(2).unwrap(), 255);
    }

    #[test]
    fn scalar_u8_as_i8_overflow() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [128u8]).unwrap();
        let r = TypedReader::<i8>::open(&path).unwrap();
        assert!(r.get_value(0).is_err());
    }

    #[test]
    fn narrowing_rejected() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&42i32.to_le_bytes()).unwrap();
        assert!(TypedReader::<u8>::open(&path).is_err());
        assert!(TypedReader::<i8>::open(&path).is_err());
    }

    #[test]
    fn ivec_i32_native() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.ivec");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&3i32.to_le_bytes()).unwrap();
        f.write_all(&10i32.to_le_bytes()).unwrap();
        f.write_all(&20i32.to_le_bytes()).unwrap();
        f.write_all(&30i32.to_le_bytes()).unwrap();
        drop(f);
        let r = TypedReader::<i32>::open(&path).unwrap();
        assert_eq!(r.count(), 1);
        assert_eq!(r.dim(), 3);
        let rec = r.get_record(0).unwrap();
        assert_eq!(rec, vec![10, 20, 30]);
    }

    #[test]
    fn out_of_bounds() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        let r = TypedReader::<u8>::open(&path).unwrap();
        assert!(r.get_value(3).is_err());
    }

    #[test]
    fn empty_scalar() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, []).unwrap();
        let r = TypedReader::<u8>::open(&path).unwrap();
        assert_eq!(r.count(), 0);
    }
}
