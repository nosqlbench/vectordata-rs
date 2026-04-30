// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Typed data access with runtime type negotiation.
//!
//! [`TypedReader<T>`] opens vector and scalar files with compile-time
//! type safety and runtime width/signedness validation. The transport
//! choice (local mmap, merkle-cached remote, direct HTTP) is hidden
//! inside the crate-private [`crate::storage::Storage`] abstraction.
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
//! let val: u8 = reader.get_native(42);
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
    U8, I8, U16, I16, U32, I32, U64, I64,
    F16, F32, F64,
}

impl ElementType {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no extension: {}", path.display()))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}': {}", path.display()))
    }

    pub fn from_url(url: &url::Url) -> Result<Self, String> {
        let path = url.path();
        let ext = path.rsplit('.').next()
            .ok_or_else(|| format!("no extension in URL: {url}"))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}' in URL: {url}"))
    }

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

    pub fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    pub fn is_scalar_format(path: impl AsRef<Path>) -> bool {
        let ext = path.as_ref().extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(ext.to_lowercase().as_str(),
            "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64")
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8", Self::I8 => "i8",
            Self::U16 => "u16", Self::I16 => "i16",
            Self::U32 => "u32", Self::I32 => "i32",
            Self::U64 => "u64", Self::I64 => "i64",
            Self::F16 => "f16", Self::F32 => "f32", Self::F64 => "f64",
        }
    }

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

#[derive(Debug)]
pub enum TypedAccessError {
    /// Target type is narrower than native type.
    Narrowing { native: ElementType, target: &'static str },
    /// Value at ordinal doesn't fit in the target type.
    ValueOverflow { ordinal: usize, value: i128, target: &'static str },
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

pub trait TypedElement: Copy + Send + Sync + 'static {
    fn type_name() -> &'static str;
    fn width() -> usize;
    fn from_i128(val: i128) -> Option<Self>;
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

fn is_scalar_url_path(path: &str) -> bool {
    let ext = match path.rfind('.') {
        Some(p) => &path[p + 1..],
        None => return false,
    };
    matches!(ext.to_lowercase().as_str(),
        "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64" | "f32" | "f64")
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
/// Single concrete struct over [`crate::storage::Storage`]. The
/// transport choice (local mmap, merkle-cached remote, direct HTTP)
/// is hidden inside the storage and selected by [`Storage::open`].
pub struct TypedReader<T: TypedElement> {
    storage: Arc<Storage>,
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
        let storage = Arc::new(Storage::open_path(path)
            .map_err(|e| TypedAccessError::Io(format!("open {}: {e}", path.display())))?);
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
        let storage = Arc::new(Storage::open_url(url.clone())
            .map_err(|e| TypedAccessError::Io(format!("open_url {url}: {e}")))?);
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
            storage, native_type, native_width, is_scalar, dim, count,
            _phantom: std::marker::PhantomData,
        })
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, TypedAccessError> {
        self.storage.read_bytes(offset as u64, len as u64)
            .map_err(|e| TypedAccessError::Io(e.to_string()))
    }

    fn mmap_slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        self.storage.mmap_slice(offset as u64, len as u64)
    }

    pub fn native_type(&self) -> ElementType { self.native_type }
    pub fn is_native(&self) -> bool { T::width() == self.native_width }
    pub fn count(&self) -> usize { self.count }
    pub fn dim(&self) -> usize { self.dim }

    /// Force-download every byte into the local cache. No-op for
    /// local files and for non-cacheable HTTP. Idempotent.
    pub fn prebuffer(&self) -> std::io::Result<()> { self.storage.prebuffer() }

    /// Whether all bytes are locally accessible without network round-trips.
    pub fn is_complete(&self) -> bool { self.storage.is_complete() }

    /// Get a single value from a scalar file (dim=1), with checked conversion.
    pub fn get_value(&self, ordinal: usize) -> Result<T, TypedAccessError> {
        if ordinal >= self.count {
            return Err(TypedAccessError::Io(
                format!("ordinal {ordinal} out of range (count {})", self.count)));
        }
        let offset = if self.is_scalar {
            ordinal * self.native_width
        } else {
            let record_bytes = 4 + self.dim * self.native_width;
            ordinal * record_bytes + 4
        };
        let bytes = self.read_bytes(offset, self.native_width)?;
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
        let offset = if self.is_scalar {
            ordinal * self.native_width * self.dim
        } else {
            let record_bytes = 4 + self.dim * self.native_width;
            ordinal * record_bytes + 4
        };
        let total_bytes = self.dim * self.native_width;
        let bytes = self.read_bytes(offset, total_bytes)?;
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

impl TypedReader<u8> {
    /// Zero-copy native read. For non-mmap storage falls back to get_value.
    pub fn get_native(&self, ordinal: usize) -> u8 {
        debug_assert!(self.native_type == ElementType::U8);
        let offset = if self.is_scalar { ordinal } else { 4 + ordinal * (4 + self.dim) };
        if let Some(slice) = self.mmap_slice(offset, 1) {
            slice[0]
        } else {
            self.get_value(ordinal).unwrap_or(0)
        }
    }

    /// Zero-copy slice of a record's native bytes. `None` when storage
    /// is not mmap-backed.
    pub fn get_native_slice(&self, ordinal: usize) -> Option<&[u8]> {
        debug_assert!(self.native_type == ElementType::U8);
        let (offset, len) = if self.is_scalar {
            (ordinal * self.dim, self.dim)
        } else {
            (ordinal * (4 + self.dim) + 4, self.dim)
        };
        self.mmap_slice(offset, len)
    }
}

impl TypedReader<i32> {
    pub fn get_native(&self, ordinal: usize) -> i32 {
        debug_assert!(self.native_type == ElementType::I32);
        let offset = if self.is_scalar {
            ordinal * 4
        } else {
            ordinal * (4 + self.dim * 4) + 4
        };
        if let Some(slice) = self.mmap_slice(offset, 4) {
            i32::from_le_bytes(slice.try_into().unwrap())
        } else {
            self.get_value(ordinal).unwrap_or(0)
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
        std::fs::write(&path, &[0u8, 42, 127, 255]).unwrap();
        let r = TypedReader::<u8>::open(&path).unwrap();
        assert_eq!(r.count(), 4);
        assert_eq!(r.dim(), 1);
        assert!(r.is_native());
        assert_eq!(r.get_native(0), 0);
        assert_eq!(r.get_native(3), 255);
    }

    #[test]
    fn scalar_u8_as_i32_widening() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, &[0u8, 42, 255]).unwrap();
        let r = TypedReader::<i32>::open(&path).unwrap();
        assert!(!r.is_native());
        assert_eq!(r.get_value(0).unwrap(), 0i32);
        assert_eq!(r.get_value(2).unwrap(), 255);
    }

    #[test]
    fn scalar_u8_as_i8_overflow() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, &[128u8]).unwrap();
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
        std::fs::write(&path, &[1u8, 2, 3]).unwrap();
        let r = TypedReader::<u8>::open(&path).unwrap();
        assert!(r.get_value(3).is_err());
    }

    #[test]
    fn empty_scalar() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, &[]).unwrap();
        let r = TypedReader::<u8>::open(&path).unwrap();
        assert_eq!(r.count(), 0);
    }
}
