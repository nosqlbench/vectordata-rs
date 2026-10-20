// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The element model as the pipeline uses it.
//!
//! The types, the extension table and the conversion rules are the
//! shared model ([`ElementType`], [`Element`], from `veks-datatypes`
//! through vectordata), re-exported here under the path the pipeline
//! has always imported them from. What this module adds is the
//! pipeline's own numeric policy for each type — which types the SIMD
//! kernels take, what counts as a unit-norm vector at each precision —
//! as [`ElementTypeKnn`], and [`dispatch_reader!`](crate::dispatch_reader), which opens a file
//! at the type its extension names.

pub use vectordata::typed_access::{Element, ElementConversion, ElementType, ElementTypeExt};

/// The pipeline's numeric policy per element type.
pub trait ElementTypeKnn: Copy {
    /// Whether the SIMD distance kernels take this type directly: the
    /// IEEE floats. A `bf16` or integer corpus reaches them widened to
    /// f32 by the conversion table.
    fn supports_simd_distance(self) -> bool;

    /// The machine epsilon of a float type; `None` for an integer.
    fn machine_epsilon(self) -> Option<f64>;

    /// How far from 1.0 a vector's norm may be and still count as
    /// normalized, for a float type; `None` for an integer.
    fn normalization_threshold(self, dim: usize) -> Option<f64>;
}

impl ElementTypeKnn for ElementType {
    fn supports_simd_distance(self) -> bool {
        matches!(self, Self::F64 | Self::F32 | Self::F16)
    }

    fn machine_epsilon(self) -> Option<f64> {
        match self {
            Self::F16 => Some(9.77e-4_f64),
            Self::BF16 => Some(7.81e-3_f64),
            Self::F32 => Some(1.19e-7_f64),
            Self::F64 => Some(2.22e-16_f64),
            _ => None,
        }
    }

    fn normalization_threshold(self, _dim: usize) -> Option<f64> {
        match self {
            Self::F16 => Some(1e-1),
            Self::BF16 => Some(1e-1),
            Self::F32 => Some(1e-5),
            Self::F64 => Some(1e-14),
            _ => None,
        }
    }
}

/// Open a uniform vector file at the type `$etype` names and run
/// `$body` with `$reader` bound to the `XvecReader<T>` of that type.
/// Every type opens as itself; the conversion table decides what a
/// caller may then read it as.
#[macro_export]
macro_rules! dispatch_reader {
    ($etype:expr, $path:expr, $reader:ident => $body:expr) => {
        $crate::pipeline::element_type::dispatch_element!($etype, T => {
            let $reader = vectordata::io::XvecReader::<T>::open_path($path)
                .map_err(|e| format!("failed to open {}: {}", $path.display(), e))?;
            $body
        })
    };
}

pub use veks_datatypes::dispatch_element;

/// A uniform vector file read as f64 whatever element type it holds,
/// for the analysis commands that upcast every component (SRD §19.2).
/// The file is opened at its own type and each element is widened by
/// [`Element::as_f64`]: exact for every float and for integers up to
/// 53 bits, rounded above that.
pub struct F64Vectors {
    reader: vectordata::io::AnyVectorReader,
}

impl F64Vectors {
    /// Open `path` at the type its extension names.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        vectordata::io::open_vec_native(&path.to_string_lossy())
            .map(|reader| Self { reader })
            .map_err(|e| e.to_string())
    }

    /// Number of records.
    pub fn count(&self) -> usize {
        self.reader.count()
    }

    /// Elements per record.
    pub fn dim(&self) -> usize {
        self.reader.dim()
    }

    /// The element type the file holds.
    pub fn element_type(&self) -> ElementType {
        self.reader.element_type()
    }

    /// Record `index` as f64, or an empty vector past the end — the
    /// shape the analysis loops have always read.
    pub fn get(&self, index: usize) -> Vec<f64> {
        use vectordata::io::AnyVectorReader as A;
        macro_rules! widen {
            ($($v:ident),*) => {
                match &self.reader {
                    $(A::$v(r) => r.get(index).unwrap_or_default().iter().map(|x| x.as_f64()).collect(),)*
                    _ => Vec::new(),
                }
            };
        }
        widen!(U8, I8, U16, I16, U32, I32, U64, I64, F16, BF16, F32, F64)
    }

    /// The reader as the record-fetching closure the analysis commands
    /// run across their thread pools.
    pub fn into_get(self) -> Box<dyn Fn(usize) -> Vec<f64> + Send + Sync> {
        Box::new(move |i| self.get(i))
    }
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
            ("data.bf16vecs", ElementType::BF16),
        ];
        for (filename, expected) in &cases {
            assert_eq!(ElementType::from_path(PathBuf::from(filename)).unwrap(), *expected,
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
            assert_eq!(ElementType::from_path(PathBuf::from(filename)).unwrap(), *expected,
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
            ("data.f32", ElementType::F32),
            ("data.bf16", ElementType::BF16),
        ];
        for (filename, expected) in &cases {
            assert_eq!(ElementType::from_path(PathBuf::from(filename)).unwrap(), *expected,
                "failed for {}", filename);
        }
    }

    #[test]
    fn test_from_path_unknown_extension() {
        assert!(ElementType::from_path(PathBuf::from("data.csv")).is_err());
    }

    #[test]
    fn test_from_path_no_extension() {
        assert!(ElementType::from_path(PathBuf::from("data")).is_err());
    }

    #[test]
    fn test_is_float() {
        assert!(ElementType::F64.is_float());
        assert!(ElementType::F32.is_float());
        assert!(ElementType::F16.is_float());
        assert!(ElementType::BF16.is_float());
        assert!(!ElementType::I32.is_float());
        assert!(!ElementType::U8.is_float());
        assert!(!ElementType::U64.is_float());
    }

    /// The shared model and the pipeline's policy agree on which types
    /// are floats, and only the IEEE floats go to the kernels as they
    /// are.
    #[test]
    fn bf16_is_a_float_without_simd_distance() {
        for ty in ElementType::ALL {
            assert_eq!(ty.machine_epsilon().is_some(), ty.is_float(), "{ty}");
            assert_eq!(ty.normalization_threshold(128).is_some(), ty.is_float(), "{ty}");
            if ty.supports_simd_distance() {
                assert!(ty.is_float() && *ty != ElementType::BF16, "{ty}");
            }
        }
        assert!(!ElementType::BF16.supports_simd_distance());
        assert!(ElementType::F16.supports_simd_distance());
    }

    /// The path the pipeline imports from names the shared type.
    #[test]
    fn the_pipeline_element_type_is_the_shared_one() {
        fn same<T: 'static>() -> bool {
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<veks_datatypes::ElementType>()
        }
        assert!(same::<ElementType>());
        assert!(same::<vectordata::ElementType>());
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

    /// The dispatch opens every file as the type its extension names:
    /// an `.i8vecs` as `XvecReader<i8>`, not as the u8 reader of the
    /// same width.
    #[test]
    fn dispatch_reader_opens_an_i8_file_as_i8() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("codes.i8vecs");
        std::fs::write(&path, [2i32.to_le_bytes().as_slice(), &[0xff, 0x7f]].concat()).unwrap();
        let opened: Result<&'static str, String> = (|| {
            crate::dispatch_reader!(ElementType::I8, &path, reader => {
                Ok(std::any::type_name_of_val(&reader))
            })
        })();
        assert_eq!(opened.unwrap(), std::any::type_name::<vectordata::io::XvecReader<i8>>());
    }
}
