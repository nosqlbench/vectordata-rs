// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shared element type enum for vector file formats.
//!
//! [`ElementType`] maps file extensions to their element types and provides
//! convenience methods for querying type properties such as element size,
//! floating-point status, and SIMD distance support.

use std::path::Path;

/// Element type of a vector file, inferred from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// 64-bit IEEE 754 double-precision float (`.dvec`).
    F64,
    /// 32-bit IEEE 754 single-precision float (`.fvec`).
    F32,
    /// 16-bit IEEE 754 half-precision float (`.mvec`).
    F16,
    /// 32-bit signed integer (`.ivec`).
    I32,
    /// 16-bit signed integer (`.svec`).
    I16,
    /// 8-bit unsigned integer (`.bvec`).
    U8,
    /// 8-bit signed integer (no standard extension; used programmatically).
    I8,
}

impl ElementType {
    /// Detect the element type from a file path's extension.
    ///
    /// Returns an error if the extension is missing or unrecognized.
    ///
    /// # Extension mapping
    ///
    /// | Extension | ElementType |
    /// |-----------|-------------|
    /// | `.dvec`   | `F64`       |
    /// | `.fvec`   | `F32`       |
    /// | `.mvec`   | `F16`       |
    /// | `.ivec`   | `I32`       |
    /// | `.svec`   | `I16`       |
    /// | `.bvec`   | `U8`        |
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no file extension: {}", path.display()))?;

        match ext {
            "dvec" => Ok(ElementType::F64),
            "fvec" => Ok(ElementType::F32),
            "mvec" => Ok(ElementType::F16),
            "ivec" => Ok(ElementType::I32),
            "svec" => Ok(ElementType::I16),
            "bvec" => Ok(ElementType::U8),
            _ => Err(format!("unrecognized vector extension '.{}': {}", ext, path.display())),
        }
    }

    /// Returns `true` if this is a floating-point element type (F64, F32, F16).
    pub fn is_float(&self) -> bool {
        matches!(self, ElementType::F64 | ElementType::F32 | ElementType::F16)
    }

    /// Returns `true` if SIMD-accelerated distance functions are available
    /// for this element type.
    ///
    /// Currently supported: F64, F32, F16, and I8.
    pub fn supports_simd_distance(&self) -> bool {
        matches!(
            self,
            ElementType::F64 | ElementType::F32 | ElementType::F16 | ElementType::I8
        )
    }

    /// Returns the size of a single element in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            ElementType::F64 => 8,
            ElementType::F32 | ElementType::I32 => 4,
            ElementType::F16 | ElementType::I16 => 2,
            ElementType::U8 | ElementType::I8 => 1,
        }
    }
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            ElementType::F64 => "f64",
            ElementType::F32 => "f32",
            ElementType::F16 => "f16",
            ElementType::I32 => "i32",
            ElementType::I16 => "i16",
            ElementType::U8 => "u8",
            ElementType::I8 => "i8",
        };
        write!(f, "{}", label)
    }
}

/// Trait for converting vector elements to `f64` for analysis.
///
/// Implemented for all xvec element types. This enables generic analysis code
/// to accumulate statistics in f64 precision regardless of the source type.
pub trait ToF64: Copy {
    fn to_f64(self) -> f64;
}

impl ToF64 for f64 {
    #[inline]
    fn to_f64(self) -> f64 { self }
}
impl ToF64 for f32 {
    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
}
impl ToF64 for half::f16 {
    #[inline]
    fn to_f64(self) -> f64 { self.to_f64() }
}
impl ToF64 for i32 {
    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
}
impl ToF64 for i16 {
    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
}
impl ToF64 for u8 {
    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
}
impl ToF64 for i8 {
    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
}

/// Dispatch macro for vector element types.
///
/// Opens the appropriate `MmapVectorReader<T>` based on `ElementType` and
/// calls the provided expression block with the reader bound as `$reader`.
/// The block must return the same type for every arm.
///
/// Usage:
/// ```ignore
/// dispatch_reader!(etype, &source_path, reader => {
///     // `reader` is MmapVectorReader<T> where T: ToF64
///     let count = VectorReader::<_>::count(&reader);
///     // ...
/// })
/// ```
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
                // i8 has no standard xvec extension; reinterpret bvec as i8
                let $reader = vectordata::io::MmapVectorReader::<u8>::open_bvec($path)
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
    fn test_from_path_all_extensions() {
        let cases = [
            ("data.dvec", ElementType::F64),
            ("data.fvec", ElementType::F32),
            ("data.mvec", ElementType::F16),
            ("data.ivec", ElementType::I32),
            ("data.svec", ElementType::I16),
            ("data.bvec", ElementType::U8),
        ];
        for (filename, expected) in &cases {
            let path = PathBuf::from(filename);
            assert_eq!(
                ElementType::from_path(&path).unwrap(),
                *expected,
                "failed for {}",
                filename,
            );
        }
    }

    #[test]
    fn test_from_path_nested() {
        let path = PathBuf::from("/data/profiles/10M/base_vectors.mvec");
        assert_eq!(ElementType::from_path(&path).unwrap(), ElementType::F16);
    }

    #[test]
    fn test_from_path_unknown_extension() {
        let path = PathBuf::from("data.csv");
        assert!(ElementType::from_path(&path).is_err());
    }

    #[test]
    fn test_from_path_no_extension() {
        let path = PathBuf::from("data");
        assert!(ElementType::from_path(&path).is_err());
    }

    #[test]
    fn test_is_float() {
        assert!(ElementType::F64.is_float());
        assert!(ElementType::F32.is_float());
        assert!(ElementType::F16.is_float());
        assert!(!ElementType::I32.is_float());
        assert!(!ElementType::I16.is_float());
        assert!(!ElementType::U8.is_float());
        assert!(!ElementType::I8.is_float());
    }

    #[test]
    fn test_supports_simd_distance() {
        assert!(ElementType::F64.supports_simd_distance());
        assert!(ElementType::F32.supports_simd_distance());
        assert!(ElementType::F16.supports_simd_distance());
        assert!(ElementType::I8.supports_simd_distance());
        assert!(!ElementType::I32.supports_simd_distance());
        assert!(!ElementType::I16.supports_simd_distance());
        assert!(!ElementType::U8.supports_simd_distance());
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
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ElementType::F64), "f64");
        assert_eq!(format!("{}", ElementType::F32), "f32");
        assert_eq!(format!("{}", ElementType::I8), "i8");
    }
}
