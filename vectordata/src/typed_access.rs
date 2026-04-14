// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Typed data access with runtime type negotiation.
//!
//! Provides [`TypedReader`] for opening vector and scalar files with
//! compile-time type safety and runtime width/signedness validation.
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
//! // Interrogate the native type
//! let etype = ElementType::from_path("metadata.u8").unwrap();
//! assert_eq!(etype, ElementType::U8);
//! assert_eq!(etype.byte_width(), 1);
//!
//! // Open with native type — zero-copy
//! let reader = TypedReader::<u8>::open("metadata.u8").unwrap();
//! let val: u8 = reader.get_native(42);
//!
//! // Open with wider type — always succeeds
//! let reader = TypedReader::<i32>::open("metadata.u8").unwrap();
//! let val: i32 = reader.get_value(42).unwrap();
//!
//! // Open with same-width cross-sign — checked per value
//! let reader = TypedReader::<i8>::open("metadata.u8").unwrap();
//! let val = reader.get_value(42); // Ok if value ≤ 127, Err if > 127
//! ```

use std::path::Path;

/// Element type of a data file, inferred from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    U8, I8, U16, I16, U32, I32, U64, I64,
    F16, F32, F64,
}

impl ElementType {
    /// Detect from a file path's extension.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no extension: {}", path.display()))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{}': {}", ext, path.display()))
    }

    /// Detect from a URL path extension.
    pub fn from_url(url: &url::Url) -> Result<Self, String> {
        let path = url.path();
        let ext = path.rsplit('.').next()
            .ok_or_else(|| format!("no extension in URL: {url}"))?;
        Self::from_extension(ext)
            .ok_or_else(|| format!("unknown extension '.{ext}' in URL: {url}"))
    }

    /// Detect from a bare extension string.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            // Scalar formats
            "u8" => Some(Self::U8),
            "i8" => Some(Self::I8),
            "u16" => Some(Self::U16),
            "i16" => Some(Self::I16),
            "u32" => Some(Self::U32),
            "i32" => Some(Self::I32),
            "u64" => Some(Self::U64),
            "i64" => Some(Self::I64),
            // Vector formats (xvec — uniform and variable)
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

    /// Byte width of this element type.
    pub fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Whether the file is a scalar format (no dim header).
    pub fn is_scalar_format(path: impl AsRef<Path>) -> bool {
        let ext = path.as_ref().extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        matches!(ext.to_lowercase().as_str(),
            "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64")
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8", Self::I8 => "i8",
            Self::U16 => "u16", Self::I16 => "i16",
            Self::U32 => "u32", Self::I32 => "i32",
            Self::U64 => "u64", Self::I64 => "i64",
            Self::F16 => "f16", Self::F32 => "f32", Self::F64 => "f64",
        }
    }

    /// Whether opening as target type T is valid (width check only).
    pub fn can_open_as(self, target_width: usize) -> bool {
        target_width >= self.byte_width()
    }
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Error from typed access operations.
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
                write!(f, "cannot open {} file as {} (narrowing)", native, target),
            Self::ValueOverflow { ordinal, value, target } =>
                write!(f, "value {} at ordinal {} does not fit in {}", value, ordinal, target),
            Self::Io(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for TypedAccessError {}

/// Trait for types that can be read from typed data files.
///
/// Implemented for all integer types. Provides the conversion logic
/// from raw bytes to the target type.
pub trait TypedElement: Copy + Send + Sync + 'static {
    /// Type name for error messages.
    fn type_name() -> &'static str;
    /// Byte width of this type.
    fn width() -> usize;
    /// Convert from a raw i128 value (read from the file in its native type).
    /// Returns None if the value doesn't fit.
    fn from_i128(val: i128) -> Option<Self>;
    /// Convert this value to i128 for comparison.
    fn to_i128(self) -> i128;
}

macro_rules! impl_typed_element {
    ($t:ty, $name:expr) => {
        impl TypedElement for $t {
            fn type_name() -> &'static str { $name }
            fn width() -> usize { std::mem::size_of::<$t>() }
            fn from_i128(val: i128) -> Option<Self> {
                <$t>::try_from(val as i128).ok()
            }
            fn to_i128(self) -> i128 { self as i128 }
        }
    };
}

impl_typed_element!(u8, "u8");
impl_typed_element!(i8, "i8");
impl_typed_element!(u16, "u16");
impl_typed_element!(i16, "i16");
impl_typed_element!(u32, "u32");
impl_typed_element!(i32, "i32");
impl_typed_element!(u64, "u64");
impl_typed_element!(i64, "i64");

/// Read a single native value from raw bytes.
fn read_native_value(data: &[u8], native: ElementType) -> i128 {
    match native {
        ElementType::U8 => data[0] as i128,
        ElementType::I8 => data[0] as i8 as i128,
        ElementType::U16 => u16::from_le_bytes([data[0], data[1]]) as i128,
        ElementType::I16 => i16::from_le_bytes([data[0], data[1]]) as i128,
        ElementType::U32 => u32::from_le_bytes(data[..4].try_into().unwrap()) as i128,
        ElementType::I32 => i32::from_le_bytes(data[..4].try_into().unwrap()) as i128,
        ElementType::U64 => u64::from_le_bytes(data[..8].try_into().unwrap()) as i128,
        ElementType::I64 => i64::from_le_bytes(data[..8].try_into().unwrap()) as i128,
        ElementType::F16 | ElementType::F32 | ElementType::F64 => 0, // float not supported here
    }
}

/// A typed reader for vector or scalar data files.
///
/// Opens local files via mmap or remote files via HTTP range requests.
/// Provides ordinal-based access as type T, with runtime validation
/// that T is wide enough to hold the native values.
pub struct TypedReader<T: TypedElement> {
    backend: TypedBackend,
    native_type: ElementType,
    native_width: usize,
    is_scalar: bool,
    dim: usize,
    count: usize,
    _phantom: std::marker::PhantomData<T>,
}

enum TypedBackend {
    Mmap(memmap2::Mmap),
    Http {
        client: reqwest::blocking::Client,
        url: url::Url,
        total_size: u64,
    },
}

impl TypedBackend {
    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, TypedAccessError> {
        match self {
            TypedBackend::Mmap(mmap) => {
                if offset + len > mmap.len() {
                    return Err(TypedAccessError::Io(
                        format!("read past end: offset={offset} len={len} size={}", mmap.len())));
                }
                Ok(mmap[offset..offset + len].to_vec())
            }
            TypedBackend::Http { client, url, .. } => {
                use reqwest::header::RANGE;
                let end = offset + len - 1;
                let resp = client.get(url.clone())
                    .header(RANGE, format!("bytes={offset}-{end}"))
                    .send()
                    .map_err(|e| TypedAccessError::Io(format!("HTTP range request: {e}")))?
                    .error_for_status()
                    .map_err(|e| TypedAccessError::Io(format!("HTTP error: {e}")))?;
                Ok(resp.bytes()
                    .map_err(|e| TypedAccessError::Io(format!("HTTP read: {e}")))?
                    .to_vec())
            }
        }
    }

    fn mmap_slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        match self {
            TypedBackend::Mmap(mmap) => Some(&mmap[offset..offset + len]),
            TypedBackend::Http { .. } => None,
        }
    }
}

impl<T: TypedElement> TypedReader<T> {
    /// Open a local file for typed reading.
    ///
    /// Fails immediately if T is narrower than the native element type.
    /// Same-width cross-sign is allowed (checked per-value on access).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TypedAccessError> {
        let path = path.as_ref();
        let native_type = ElementType::from_path(path)
            .map_err(|e| TypedAccessError::Io(e))?;

        if T::width() < native_type.byte_width() {
            return Err(TypedAccessError::Narrowing {
                native: native_type,
                target: T::type_name(),
            });
        }

        let file = std::fs::File::open(path)
            .map_err(|e| TypedAccessError::Io(format!("open {}: {}", path.display(), e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| TypedAccessError::Io(format!("mmap {}: {}", path.display(), e)))?;

        let is_scalar = ElementType::is_scalar_format(path);
        let native_width = native_type.byte_width();

        let (dim, count) = if is_scalar {
            let count = mmap.len() / native_width;
            (1, count)
        } else {
            if mmap.len() < 4 {
                (1, 0)
            } else {
                let dim = i32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
                let record_bytes = 4 + dim * native_width;
                let count = if record_bytes > 0 { mmap.len() / record_bytes } else { 0 };
                (dim, count)
            }
        };

        Ok(TypedReader {
            backend: TypedBackend::Mmap(mmap),
            native_type,
            native_width,
            is_scalar,
            dim,
            count,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Open a remote file for typed reading via HTTP range requests.
    ///
    /// Reads the file header to determine dimension and count,
    /// then provides ordinal-based access via range requests.
    pub fn open_url(url: url::Url, native_type: ElementType) -> Result<Self, TypedAccessError> {
        if T::width() < native_type.byte_width() {
            return Err(TypedAccessError::Narrowing {
                native: native_type,
                target: T::type_name(),
            });
        }

        let client = reqwest::blocking::Client::new();
        let native_width = native_type.byte_width();

        // Get total file size
        let resp = client.head(url.clone())
            .send()
            .map_err(|e| TypedAccessError::Io(format!("HTTP HEAD: {e}")))?
            .error_for_status()
            .map_err(|e| TypedAccessError::Io(format!("HTTP HEAD error: {e}")))?;
        let total_size: u64 = resp.headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| TypedAccessError::Io("Missing Content-Length".into()))?;

        // Determine if scalar by URL extension
        let is_scalar = url.path().ends_with(".u8")
            || url.path().ends_with(".i8")
            || url.path().ends_with(".u16")
            || url.path().ends_with(".i16")
            || url.path().ends_with(".u32")
            || url.path().ends_with(".i32")
            || url.path().ends_with(".u64")
            || url.path().ends_with(".i64")
            || url.path().ends_with(".f32")
            || url.path().ends_with(".f64");

        let (dim, count) = if is_scalar {
            (1, (total_size / native_width as u64) as usize)
        } else {
            // Read first 4 bytes for dim header
            use reqwest::header::RANGE;
            let resp = client.get(url.clone())
                .header(RANGE, "bytes=0-3")
                .send()
                .map_err(|e| TypedAccessError::Io(format!("HTTP range: {e}")))?;
            let bytes = resp.bytes()
                .map_err(|e| TypedAccessError::Io(format!("HTTP read: {e}")))?;
            if bytes.len() < 4 {
                (1, 0)
            } else {
                let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
                let record_bytes = 4 + dim * native_width;
                let count = if record_bytes > 0 { (total_size / record_bytes as u64) as usize } else { 0 };
                (dim, count)
            }
        };

        Ok(TypedReader {
            backend: TypedBackend::Http { client, url, total_size },
            native_type,
            native_width,
            is_scalar,
            dim,
            count,
            _phantom: std::marker::PhantomData,
        })
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

    /// The native element type of the underlying file.
    pub fn native_type(&self) -> ElementType { self.native_type }

    /// Whether T exactly matches the native type (zero-copy possible).
    pub fn is_native(&self) -> bool {
        T::width() == self.native_width
    }

    /// Number of records in the file.
    pub fn count(&self) -> usize { self.count }

    /// Dimension (elements per record). Always 1 for scalar files.
    pub fn dim(&self) -> usize { self.dim }

    /// Get a single value from a scalar file (dim=1), with checked conversion.
    ///
    /// Returns Err if the native value doesn't fit in T (cross-sign overflow).
    pub fn get_value(&self, ordinal: usize) -> Result<T, TypedAccessError> {
        if ordinal >= self.count {
            return Err(TypedAccessError::Io(
                format!("ordinal {} out of range (count {})", ordinal, self.count)));
        }
        let offset = if self.is_scalar {
            ordinal * self.native_width
        } else {
            let record_bytes = 4 + self.dim * self.native_width;
            ordinal * record_bytes + 4
        };
        let bytes = self.backend.read_bytes(offset, self.native_width)?;
        let val = read_native_value(&bytes, self.native_type);
        T::from_i128(val).ok_or_else(|| TypedAccessError::ValueOverflow {
            ordinal,
            value: val,
            target: T::type_name(),
        })
    }

    /// Get a record as Vec<T>, with checked conversion per element.
    ///
    /// For scalar files (dim=1), returns a single-element vec.
    /// For xvec files, returns all elements of the record.
    pub fn get_record(&self, ordinal: usize) -> Result<Vec<T>, TypedAccessError> {
        if ordinal >= self.count {
            return Err(TypedAccessError::Io(
                format!("ordinal {} out of range (count {})", ordinal, self.count)));
        }
        let offset = if self.is_scalar {
            ordinal * self.native_width * self.dim
        } else {
            let record_bytes = 4 + self.dim * self.native_width;
            ordinal * record_bytes + 4
        };
        let total_bytes = self.dim * self.native_width;
        let bytes = self.backend.read_bytes(offset, total_bytes)?;
        let mut result = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            let elem_offset = d * self.native_width;
            let val = read_native_value(&bytes[elem_offset..], self.native_type);
            result.push(T::from_i128(val).ok_or_else(|| TypedAccessError::ValueOverflow {
                ordinal,
                value: val,
                target: T::type_name(),
            })?);
        }
        Ok(result)
    }
}

// ── Native zero-copy access (only when T matches native type) ────────

impl TypedReader<u8> {
    /// Zero-copy access to the raw native value. Only works for mmap backend.
    /// Falls back to get_value for HTTP.
    pub fn get_native(&self, ordinal: usize) -> u8 {
        debug_assert!(self.native_type == ElementType::U8);
        let offset = if self.is_scalar { ordinal } else {
            4 + ordinal * (4 + self.dim)
        };
        if let Some(slice) = self.backend.mmap_slice(offset, 1) {
            slice[0]
        } else {
            self.get_value(ordinal).unwrap_or(0)
        }
    }

    /// Zero-copy slice of a record's native data. Only works for mmap backend.
    /// Returns None for HTTP backend (use get_record instead).
    pub fn get_native_slice(&self, ordinal: usize) -> Option<&[u8]> {
        debug_assert!(self.native_type == ElementType::U8);
        let (offset, len) = if self.is_scalar {
            (ordinal * self.dim, self.dim)
        } else {
            let record_bytes = 4 + self.dim;
            (ordinal * record_bytes + 4, self.dim)
        };
        self.backend.mmap_slice(offset, len)
    }
}

impl TypedReader<i32> {
    /// Zero-copy access to native i32 value at ordinal.
    pub fn get_native(&self, ordinal: usize) -> i32 {
        debug_assert!(self.native_type == ElementType::I32);
        let offset = if self.is_scalar {
            ordinal * 4
        } else {
            let record_bytes = 4 + self.dim * 4;
            ordinal * record_bytes + 4
        };
        if let Some(slice) = self.backend.mmap_slice(offset, 4) {
            i32::from_le_bytes(slice.try_into().unwrap())
        } else {
            self.get_value(ordinal).unwrap_or(0)
        }
    }
}

/// Macro for dispatching on the native element type of a file.
///
/// Opens the file with its native type and calls the body with
/// the reader bound to the native type.
///
/// ```rust,ignore
/// use vectordata::dispatch_typed;
/// use vectordata::typed_access::{ElementType, TypedReader};
///
/// let path = "metadata.u8";
/// dispatch_typed!(path, reader => {
///     println!("count={}, dim={}", reader.count(), reader.dim());
///     for i in 0..reader.count().min(5) {
///         println!("[{}] = {:?}", i, reader.get_record(i).unwrap());
///     }
/// });
/// ```
#[macro_export]
macro_rules! dispatch_typed {
    ($path:expr, $reader:ident => $body:expr) => {{
        let etype = $crate::typed_access::ElementType::from_path($path)
            .map_err(|e| e.to_string())?;
        match etype {
            $crate::typed_access::ElementType::U8 => {
                let $reader = $crate::typed_access::TypedReader::<u8>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            $crate::typed_access::ElementType::I8 => {
                let $reader = $crate::typed_access::TypedReader::<i8>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            $crate::typed_access::ElementType::U16 => {
                let $reader = $crate::typed_access::TypedReader::<u16>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            $crate::typed_access::ElementType::I16 => {
                let $reader = $crate::typed_access::TypedReader::<i16>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            $crate::typed_access::ElementType::U32 => {
                let $reader = $crate::typed_access::TypedReader::<u32>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            $crate::typed_access::ElementType::I32 => {
                let $reader = $crate::typed_access::TypedReader::<i32>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            $crate::typed_access::ElementType::U64 => {
                let $reader = $crate::typed_access::TypedReader::<u64>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            $crate::typed_access::ElementType::I64 => {
                let $reader = $crate::typed_access::TypedReader::<i64>::open($path)
                    .map_err(|e| e.to_string())?;
                $body
            }
            _ => Err(format!("unsupported element type {:?} for typed access", etype)),
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_tmp() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    // ── ElementType detection ──────────────────────────────────────

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

    // ── Scalar u8 file ─────────────────────────────────────────────

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
        assert_eq!(r.get_native(1), 42);
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
        assert_eq!(r.get_value(1).unwrap(), 42);
        assert_eq!(r.get_value(2).unwrap(), 255); // widening: always works
    }

    #[test]
    fn scalar_u8_as_i8_cross_sign_ok() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, &[0u8, 42, 127]).unwrap();

        let r = TypedReader::<i8>::open(&path).unwrap();
        assert_eq!(r.get_value(0).unwrap(), 0i8);
        assert_eq!(r.get_value(1).unwrap(), 42i8);
        assert_eq!(r.get_value(2).unwrap(), 127i8);
    }

    #[test]
    fn scalar_u8_as_i8_overflow() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, &[128u8]).unwrap();

        let r = TypedReader::<i8>::open(&path).unwrap();
        assert!(r.get_value(0).is_err()); // 128 doesn't fit in i8
    }

    #[test]
    fn scalar_i8_as_u8_negative_fails() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i8");
        std::fs::write(&path, &[(-1i8) as u8]).unwrap();

        let r = TypedReader::<u8>::open(&path).unwrap();
        assert!(r.get_value(0).is_err()); // -1 doesn't fit in u8
    }

    // ── Narrowing rejected at open ─────────────────────────────────

    #[test]
    fn narrowing_rejected() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&42i32.to_le_bytes()).unwrap();

        assert!(TypedReader::<u8>::open(&path).is_err());
        assert!(TypedReader::<i8>::open(&path).is_err());
        assert!(TypedReader::<u16>::open(&path).is_err());
    }

    #[test]
    fn same_width_allowed() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&42i32.to_le_bytes()).unwrap();

        // u32 is same width as i32 — allowed
        let r = TypedReader::<u32>::open(&path).unwrap();
        assert_eq!(r.get_value(0).unwrap(), 42u32);
    }

    // ── Scalar i32 file ────────────────────────────────────────────

    #[test]
    fn scalar_i32_native() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        for v in [-1i32, 0, 42, i32::MAX, i32::MIN] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let r = TypedReader::<i32>::open(&path).unwrap();
        assert_eq!(r.count(), 5);
        assert_eq!(r.get_native(0), -1);
        assert_eq!(r.get_native(3), i32::MAX);
    }

    #[test]
    fn scalar_i32_as_i64_widening() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&i32::MIN.to_le_bytes()).unwrap();
        drop(f);

        let r = TypedReader::<i64>::open(&path).unwrap();
        assert_eq!(r.get_value(0).unwrap(), i32::MIN as i64);
    }

    // ── Ivec (xvec with dim header) ────────────────────────────────

    #[test]
    fn ivec_i32_native() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.ivec");
        let mut f = std::fs::File::create(&path).unwrap();
        // Record: dim=3, values=[10, 20, 30]
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
    fn ivec_as_i64_widening() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.ivec");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&2i32.to_le_bytes()).unwrap();
        f.write_all(&100i32.to_le_bytes()).unwrap();
        f.write_all(&(-50i32).to_le_bytes()).unwrap();
        drop(f);

        let r = TypedReader::<i64>::open(&path).unwrap();
        let rec = r.get_record(0).unwrap();
        assert_eq!(rec, vec![100i64, -50]);
    }

    // ── Out of bounds ──────────────────────────────────────────────

    #[test]
    fn out_of_bounds() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, &[1u8, 2, 3]).unwrap();

        let r = TypedReader::<u8>::open(&path).unwrap();
        assert!(r.get_value(3).is_err());
    }

    // ── Empty file ─────────────────────────────────────────────────

    #[test]
    fn empty_scalar() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        std::fs::write(&path, &[]).unwrap();

        let r = TypedReader::<u8>::open(&path).unwrap();
        assert_eq!(r.count(), 0);
    }

    // ── Full widening matrix ───────────────────────────────────────

    #[test]
    fn widening_always_succeeds() {
        let tmp = make_tmp();

        // u8 → everything wider
        let path = tmp.path().join("w.u8");
        std::fs::write(&path, &[255u8]).unwrap();
        assert!(TypedReader::<u16>::open(&path).unwrap().get_value(0).is_ok());
        assert!(TypedReader::<i16>::open(&path).unwrap().get_value(0).is_ok());
        assert!(TypedReader::<u32>::open(&path).unwrap().get_value(0).is_ok());
        assert!(TypedReader::<i32>::open(&path).unwrap().get_value(0).is_ok());
        assert!(TypedReader::<u64>::open(&path).unwrap().get_value(0).is_ok());
        assert!(TypedReader::<i64>::open(&path).unwrap().get_value(0).is_ok());

        // i8 → wider signed
        let path = tmp.path().join("w.i8");
        std::fs::write(&path, &[(-128i8) as u8]).unwrap();
        assert!(TypedReader::<i16>::open(&path).unwrap().get_value(0).is_ok());
        assert!(TypedReader::<i32>::open(&path).unwrap().get_value(0).is_ok());
        assert!(TypedReader::<i64>::open(&path).unwrap().get_value(0).is_ok());
    }
}
