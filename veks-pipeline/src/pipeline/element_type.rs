// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared element type enum for vector and scalar file formats.
//!
//! [`ElementType`] maps file extensions to their element types and provides
//! convenience methods for querying type properties such as element size,
//! floating-point status, and SIMD distance support.

use std::path::Path;

/// Element type of a vector or scalar file, inferred from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// 64-bit IEEE 754 double-precision float (`.dvec`).
    F64,
    /// 32-bit IEEE 754 single-precision float (`.fvec`).
    F32,
    /// 16-bit IEEE 754 half-precision float (`.mvec`).
    F16,
    /// 8-bit unsigned integer (`.bvec`, `.u8vec`, `.u8`).
    U8,
    /// 8-bit signed integer (`.i8vec`, `.i8`).
    I8,
    /// 16-bit unsigned integer (`.u16vec`, `.u16`).
    U16,
    /// 16-bit signed integer (`.svec`, `.i16vec`, `.i16`).
    I16,
    /// 32-bit unsigned integer (`.u32vec`, `.u32`).
    U32,
    /// 32-bit signed integer (`.ivec`, `.i32vec`, `.i32`).
    I32,
    /// 64-bit unsigned integer (`.u64vec`, `.u64`).
    U64,
    /// 64-bit signed integer (`.i64vec`, `.i64`).
    I64,
}

impl ElementType {
    /// Detect the element type from a file path's extension.
    ///
    /// Returns an error if the extension is missing or unrecognized.
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no file extension: {}", path.display()))?;

        match ext.to_lowercase().as_str() {
            "dvec" | "dvecs" => Ok(Self::F64),
            "fvec" | "fvecs" => Ok(Self::F32),
            "mvec" | "mvecs" => Ok(Self::F16),
            "bvec" | "bvecs" | "u8vec" | "u8vecs" | "u8" => Ok(Self::U8),
            "i8vec" | "i8vecs" | "i8" => Ok(Self::I8),
            "u16vec" | "u16vecs" | "u16" => Ok(Self::U16),
            "svec" | "svecs" | "i16vec" | "i16vecs" | "i16" => Ok(Self::I16),
            "u32vec" | "u32vecs" | "u32" => Ok(Self::U32),
            "ivec" | "ivecs" | "i32vec" | "i32vecs" | "i32" => Ok(Self::I32),
            "u64vec" | "u64vecs" | "u64" => Ok(Self::U64),
            "i64vec" | "i64vecs" | "i64" => Ok(Self::I64),
            _ => Err(format!("unrecognized vector extension '.{}': {}", ext, path.display())),
        }
    }

    /// Returns `true` if this is a floating-point element type.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F64 | Self::F32 | Self::F16)
    }

    /// Returns `true` if SIMD-accelerated distance functions are available.
    pub fn supports_simd_distance(&self) -> bool {
        matches!(self, Self::F64 | Self::F32 | Self::F16 | Self::I8)
    }

    /// Machine epsilon for floating-point types. `None` for integers.
    pub fn machine_epsilon(&self) -> Option<f64> {
        match self {
            Self::F16 => Some(9.77e-4_f64),
            Self::F32 => Some(1.19e-7_f64),
            Self::F64 => Some(2.22e-16_f64),
            _ => None,
        }
    }

    /// Fixed normalization threshold per element type. `None` for non-float.
    pub fn normalization_threshold(&self, _dim: usize) -> Option<f64> {
        match self {
            Self::F16 => Some(1e-1),
            Self::F32 => Some(1e-5),
            Self::F64 => Some(1e-14),
            _ => None,
        }
    }

    /// Size of a single element in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::I16 | Self::U16 => 2,
            Self::U8 | Self::I8 => 1,
        }
    }
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
        };
        write!(f, "{}", label)
    }
}

/// Trait for converting vector elements to `f64` for analysis.
pub trait ToF64: Copy {
    fn to_f64(self) -> f64;
}

impl ToF64 for f64 { #[inline] fn to_f64(self) -> f64 { self } }
impl ToF64 for f32 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for half::f16 { #[inline] fn to_f64(self) -> f64 { self.to_f64() } }
impl ToF64 for i32 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for i16 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for u8 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for i8 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for u16 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for u32 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for u64 { #[inline] fn to_f64(self) -> f64 { self as f64 } }
impl ToF64 for i64 { #[inline] fn to_f64(self) -> f64 { self as f64 } }

/// Dispatch macro for vector element types.
///
/// Opens the appropriate `MmapVectorReader<T>` based on `ElementType` and
/// calls the provided expression block with the reader bound as `$reader`.
#[macro_export]
macro_rules! dispatch_reader {
    ($etype:expr, $path:expr, $reader:ident => $body:expr) => {
        match $etype {
            $crate::pipeline::element_type::ElementType::F64 => {
                let $reader = vectordata::io::MmapVectorReader::<f64>::open_dvec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::F32 => {
                let $reader = vectordata::io::MmapVectorReader::<f32>::open_fvec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::F16 => {
                let $reader = vectordata::io::MmapVectorReader::<half::f16>::open_mvec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::I32 => {
                let $reader = vectordata::io::MmapVectorReader::<i32>::open_ivec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::I16 => {
                let $reader = vectordata::io::MmapVectorReader::<i16>::open_svec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::U8 => {
                let $reader = vectordata::io::MmapVectorReader::<u8>::open_bvec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::I8 => {
                // i8vec: reinterpret as u8 (same byte layout)
                let $reader = vectordata::io::MmapVectorReader::<u8>::open_bvec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::U16 => {
                // u16vec: same layout as i16 but unsigned
                let $reader = vectordata::io::MmapVectorReader::<i16>::open_svec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::U32 => {
                // u32vec: same layout as i32 but unsigned
                let $reader = vectordata::io::MmapVectorReader::<i32>::open_ivec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::U64 => {
                // u64vec: same byte layout as i64
                let $reader = vectordata::io::MmapVectorReader::<f64>::open_dvec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
            $crate::pipeline::element_type::ElementType::I64 => {
                // i64vec: same byte layout as f64 (8 bytes)
                let $reader = vectordata::io::MmapVectorReader::<f64>::open_dvec($path)
                    .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
                $body
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_from_path_legacy_extensions() {
        let cases = [
            ("data.dvec", ElementType::F64),
            ("data.fvec", ElementType::F32),
            ("data.mvec", ElementType::F16),
            ("data.ivec", ElementType::I32),
            ("data.svec", ElementType::I16),
            ("data.bvec", ElementType::U8),
        ];
        for (filename, expected) in &cases {
            assert_eq!(ElementType::from_path(&PathBuf::from(filename)).unwrap(), *expected,
                "failed for {}", filename);
        }
    }

    #[test]
    fn test_from_path_explicit_vec_extensions() {
        let cases = [
            ("data.u8vec", ElementType::U8),
            ("data.i8vec", ElementType::I8),
            ("data.u16vec", ElementType::U16),
            ("data.i16vec", ElementType::I16),
            ("data.u32vec", ElementType::U32),
            ("data.i32vec", ElementType::I32),
            ("data.u64vec", ElementType::U64),
            ("data.i64vec", ElementType::I64),
        ];
        for (filename, expected) in &cases {
            assert_eq!(ElementType::from_path(&PathBuf::from(filename)).unwrap(), *expected,
                "failed for {}", filename);
        }
    }

    #[test]
    fn test_from_path_scalar_extensions() {
        let cases = [
            ("data.u8", ElementType::U8),
            ("data.i8", ElementType::I8),
            ("data.u16", ElementType::U16),
            ("data.i16", ElementType::I16),
            ("data.u32", ElementType::U32),
            ("data.i32", ElementType::I32),
            ("data.u64", ElementType::U64),
            ("data.i64", ElementType::I64),
        ];
        for (filename, expected) in &cases {
            assert_eq!(ElementType::from_path(&PathBuf::from(filename)).unwrap(), *expected,
                "failed for {}", filename);
        }
    }

    #[test]
    fn test_from_path_unknown_extension() {
        assert!(ElementType::from_path(&PathBuf::from("data.csv")).is_err());
    }

    #[test]
    fn test_from_path_no_extension() {
        assert!(ElementType::from_path(&PathBuf::from("data")).is_err());
    }

    #[test]
    fn test_is_float() {
        assert!(ElementType::F64.is_float());
        assert!(ElementType::F32.is_float());
        assert!(ElementType::F16.is_float());
        assert!(!ElementType::I32.is_float());
        assert!(!ElementType::U8.is_float());
        assert!(!ElementType::U64.is_float());
    }

    #[test]
    fn test_element_size() {
        assert_eq!(ElementType::F64.element_size(), 8);
        assert_eq!(ElementType::F32.element_size(), 4);
        assert_eq!(ElementType::F16.element_size(), 2);
        assert_eq!(ElementType::I32.element_size(), 4);
        assert_eq!(ElementType::I16.element_size(), 2);
        assert_eq!(ElementType::U8.element_size(), 1);
        assert_eq!(ElementType::I8.element_size(), 1);
        assert_eq!(ElementType::U16.element_size(), 2);
        assert_eq!(ElementType::U32.element_size(), 4);
        assert_eq!(ElementType::U64.element_size(), 8);
        assert_eq!(ElementType::I64.element_size(), 8);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ElementType::F64), "f64");
        assert_eq!(format!("{}", ElementType::U16), "u16");
        assert_eq!(format!("{}", ElementType::I64), "i64");
    }

    #[test]
    fn test_machine_epsilon() {
        assert!(ElementType::F16.machine_epsilon().is_some());
        assert!(ElementType::F32.machine_epsilon().is_some());
        assert!(ElementType::F64.machine_epsilon().is_some());
        assert!(ElementType::I32.machine_epsilon().is_none());
        assert!(ElementType::U64.machine_epsilon().is_none());
    }
}
