// Copyright (c) nosqlbench contributors
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
            // Vector formats (xvec)
            "bvec" | "bvecs" | "u8vec" | "u8vecs" => Some(Self::U8),
            "i8vec" | "i8vecs" => Some(Self::I8),
            "svec" | "svecs" | "i16vec" | "i16vecs" => Some(Self::I16),
            "u16vec" | "u16vecs" => Some(Self::U16),
            "ivec" | "ivecs" | "i32vec" | "i32vecs" => Some(Self::I32),
            "u32vec" | "u32vecs" => Some(Self::U32),
            "i64vec" | "i64vecs" => Some(Self::I64),
            "u64vec" | "u64vecs" => Some(Self::U64),
            "fvec" | "fvecs" => Some(Self::F32),
            "mvec" | "mvecs" => Some(Self::F16),
            "dvec" | "dvecs" => Some(Self::F64),
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
/// Opens a file and provides access as type T, with runtime validation
/// that T is wide enough to hold the native values.
pub struct TypedReader<T: TypedElement> {
    data: memmap2::Mmap,
    native_type: ElementType,
    native_width: usize,
    is_scalar: bool,
    dim: usize,
    count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TypedElement> TypedReader<T> {
    /// Open a file for typed reading.
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
            // xvec: read dim from first record header
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
            data: mmap,
            native_type,
            native_width,
            is_scalar,
            dim,
            count,
            _phantom: std::marker::PhantomData,
        })
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
            ordinal * record_bytes + 4 // skip dim header
        };
        let val = read_native_value(&self.data[offset..], self.native_type);
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
        let mut result = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            let elem_offset = offset + d * self.native_width;
            let val = read_native_value(&self.data[elem_offset..], self.native_type);
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
    /// Zero-copy access to the raw native value. Panics if not native u8.
    pub fn get_native(&self, ordinal: usize) -> u8 {
        debug_assert!(self.native_type == ElementType::U8);
        let offset = if self.is_scalar { ordinal } else {
            4 + ordinal * (4 + self.dim)
        };
        self.data[offset]
    }

    /// Zero-copy slice of a record's native data. Panics if not native u8.
    pub fn get_native_slice(&self, ordinal: usize) -> &[u8] {
        debug_assert!(self.native_type == ElementType::U8);
        let (offset, len) = if self.is_scalar {
            (ordinal * self.dim, self.dim)
        } else {
            let record_bytes = 4 + self.dim;
            (ordinal * record_bytes + 4, self.dim)
        };
        &self.data[offset..offset + len]
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
        i32::from_le_bytes(self.data[offset..offset+4].try_into().unwrap())
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
