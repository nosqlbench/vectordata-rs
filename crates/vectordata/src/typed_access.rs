// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The element model, and typed data access with runtime
//! type negotiation.
//!
//! The model — [`ElementType`], the [`Element`] trait, the extension
//! table and the conversion table ([`ElementType::conversion_to`]) —
//! is `veks-datatypes`, re-exported here and at the crate root;
//! [`ElementTypeExt`] adds the path and URL forms this crate reads
//! extensions from.
//!
//! [`TypedReader<T>`] opens vector and scalar files with compile-time
//! type safety and runtime width/signedness validation. The transport
//! choice (local mmap, merkle-cached remote, direct HTTP) is hidden
//! inside the crate-private `Storage` abstraction.
//!
//! # Access modes
//!
//! What a file of one type may be read as is one table,
//! [`ElementType::conversion_to`]:
//!
//! - **Native**: open with the exact native type → zero-copy `&[T]` access
//! - **Widening**: open a file as a type that holds every value of its own → checked conversion, always succeeds
//! - **Cross-sign same width**: u8↔i8, u16↔i16, etc. → checked per-value, fails on overflow
//! - **Refused**: narrowing, a float as an integer, a lossy integer-to-float → rejected at open time
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
// ElementType — the shared element model, plus its path forms
// ═══════════════════════════════════════════════════════════════════════

pub use veks_datatypes::{
    Container, Element, ElementConversion, ElementKind, ElementType, FloatFormat, Value, bf16, f16,
};

/// The path and URL forms of [`ElementType::from_extension`], and the
/// scalar-layout question, as this crate asks them.
///
/// Brought into scope with `use vectordata::ElementTypeExt` (or
/// `vectordata::typed_access::ElementTypeExt`).
pub trait ElementTypeExt: Sized {
    /// Infer the element type from the extension of a local path.
    ///
    /// Errors when the path has no extension or the extension is not
    /// one [`ElementType::from_extension`] recognises.
    fn from_path(path: impl AsRef<Path>) -> Result<Self, String>;

    /// Infer the element type from the extension of a URL's path
    /// (the text after the last `.`).
    ///
    /// Errors when that text is not a recognised extension.
    fn from_url(url: &url::Url) -> Result<Self, String>;

    /// Whether `path` names a scalar facet — raw packed values with no
    /// per-record header. See `crate::io::is_scalar_ext`, the single
    /// table both this and the URL form answer from.
    fn is_scalar_format(path: impl AsRef<Path>) -> bool;

    /// Whether a file of this type may be opened as a target type of
    /// `target_width` bytes — true unless that would narrow. Width
    /// alone no longer decides: [`ElementType::conversion_to`] does.
    #[deprecated(note = "use `ElementType::conversion_to`, which answers by type rather than width")]
    fn can_open_as(self, target_width: usize) -> bool;
}

impl ElementTypeExt for ElementType {
    fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no extension: {}", path.display()))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}': {}", path.display()))
    }

    fn from_url(url: &url::Url) -> Result<Self, String> {
        let path = url.path();
        let ext = path.rsplit('.').next()
            .ok_or_else(|| format!("no extension in URL: {url}"))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}' in URL: {url}"))
    }

    fn is_scalar_format(path: impl AsRef<Path>) -> bool {
        let ext = path.as_ref().extension().and_then(|e| e.to_str()).unwrap_or("");
        crate::io::is_scalar_ext(ext)
    }

    fn can_open_as(self, target_width: usize) -> bool {
        target_width >= self.byte_width()
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
// TypedElement — the bound this reader used before the element model
// ═══════════════════════════════════════════════════════════════════════

/// The bound [`TypedReader`] put on its element type before the shared
/// model. Every [`Element`] is one; bound on [`Element`] instead.
#[deprecated(note = "bound on `Element` (re-exported here and at the crate root)")]
pub trait TypedElement: Element {}
#[allow(deprecated)]
impl<T: Element> TypedElement for T {}

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

/// One element of the file's own type, decoded from its bytes, as a
/// [`Value`] for conversion to `T`.
fn read_native_value(data: &[u8], native: ElementType) -> Value {
    veks_datatypes::dispatch_element!(native, N => <N as Element>::from_le_bytes(data).to_value())
}

/// The value a [`TypedAccessError::ValueOverflow`] reports: the integer
/// as stored. A float never overflows here, because the table never
/// lets a float be read as an integer.
fn overflow_value(value: Value) -> i128 {
    match value {
        Value::Int(i) => i,
        Value::Float(x) => x as i128,
    }
}

/// The open-time check: the file's type must convert to `T` by the
/// table, or the reader does not open.
fn check_open<T: Element>(native: ElementType) -> Result<(), TypedAccessError> {
    match native.conversion_to(T::TYPE) {
        ElementConversion::Refused => Err(TypedAccessError::Narrowing { native, target: T::TYPE.name() }),
        _ => Ok(()),
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
pub struct TypedReader<T: Element> {
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

impl<T: Element> std::fmt::Debug for TypedReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedReader")
            .field("as", &T::TYPE)
            .field("native", &self.native_type)
            .field("dim", &self.dim)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl<T: Element> TypedReader<T> {
    /// Open a local file by path. Inferred element type from extension.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TypedAccessError> {
        let path = path.as_ref();
        let native_type = ElementType::from_path(path)
            .map_err(TypedAccessError::Io)?;
        check_open::<T>(native_type)?;
        let storage = Storage::open_path(path)
            .map_err(|e| TypedAccessError::Io(format!("open {}: {e}", path.display())))?;
        let is_scalar = ElementType::is_scalar_format(path);
        Self::from_storage(storage, native_type, is_scalar)
    }

    /// Deprecated: open a remote dataset through
    /// [`Catalog`](crate::catalog::Catalog) or
    /// [`TestDataGroup`](crate::TestDataGroup) and read the facet with
    /// [`open_facet_typed`](crate::open_facet_typed), which fetch, plan
    /// and track it as part of its dataset. Its bytes do land in that
    /// dataset's cache directory when one covers the URL.
    #[deprecated(
        since = "2.5.0",
        note = "open the dataset by catalog or TestDataGroup and use open_facet_typed"
    )]
    pub fn open_url(url: url::Url, native_type: ElementType) -> Result<Self, TypedAccessError> {
        Self::open_remote(url, native_type)
    }

    /// Open a remote URL with cache-first dispatch (merkle cache when
    /// `.mref` is published, direct HTTP otherwise).
    fn open_remote(url: url::Url, native_type: ElementType) -> Result<Self, TypedAccessError> {
        check_open::<T>(native_type)?;
        let is_scalar = is_scalar_url_path(url.path());
        let storage = Storage::open_url(url.clone())
            .map_err(|e| TypedAccessError::Io(format!("open_url {url}: {e}")))?;
        Self::from_storage(storage, native_type, is_scalar)
    }

    /// Open from a path-or-URL string, dispatching automatically. For a
    /// remote dataset, prefer opening it by catalog or
    /// [`TestDataGroup`](crate::TestDataGroup) and reading the facet with
    /// [`open_facet_typed`](crate::open_facet_typed).
    pub fn open_auto(path_or_url: &str, native_type: ElementType) -> Result<Self, TypedAccessError> {
        if path_or_url.starts_with("http://") || path_or_url.starts_with("https://") {
            let url = url::Url::parse(path_or_url)
                .map_err(|e| TypedAccessError::Io(format!("invalid URL: {e}")))?;
            Self::open_remote(url, native_type)
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
    pub fn is_native(&self) -> bool { T::TYPE == self.native_type }
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
        T::from_value(val).ok_or(TypedAccessError::ValueOverflow {
            ordinal, value: overflow_value(val), target: T::TYPE.name(),
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
            result.push(T::from_value(val).ok_or(TypedAccessError::ValueOverflow {
                ordinal, value: overflow_value(val), target: T::TYPE.name(),
            })?);
        }
        Ok(result)
    }
}

// ─── Native zero-copy access (only when T matches native type) ──────────

impl<T: Element> TypedReader<T> {
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
        let etype = <$crate::typed_access::ElementType as $crate::typed_access::ElementTypeExt>::from_path($path)
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
            $crate::typed_access::ElementType::F16 => { let $reader = $crate::typed_access::TypedReader::<$crate::typed_access::f16>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::BF16 => { let $reader = $crate::typed_access::TypedReader::<$crate::typed_access::bf16>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::F32 => { let $reader = $crate::typed_access::TypedReader::<f32>::open($path).map_err(|e| e.to_string())?; $body }
            $crate::typed_access::ElementType::F64 => { let $reader = $crate::typed_access::TypedReader::<f64>::open($path).map_err(|e| e.to_string())?; $body }
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
