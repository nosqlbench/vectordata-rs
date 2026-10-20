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

/// An element type the f32 kernels score after widening it to f32 —
/// exactly, since these are the types the conversion table widens to
/// f32 (`ElementType::conversion_to(F32) == Widening`): `f16`, `bf16`
/// and the 8- and 16-bit integers. A KNN engine scans such a base at
/// its own width, widening one vector at a time into an f32 scratch,
/// so the corpus is never copied as f32 (SRD element access, §10).
pub trait WidenToF32: Element {
    /// Write `src` widened into `dst[..src.len()]`.
    fn widen_into(src: &[Self], dst: &mut [f32]);
}

macro_rules! widen_by_value {
    ($($t:ty),*) => {$(
        impl WidenToF32 for $t {
            #[inline]
            fn widen_into(src: &[Self], dst: &mut [f32]) {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = s.as_f64() as f32;
                }
            }
        }
    )*};
}
widen_by_value!(u8, i8, u16, i16, half::bf16);

impl WidenToF32 for half::f16 {
    #[inline]
    fn widen_into(src: &[Self], dst: &mut [f32]) {
        (crate::pipeline::simd_distance::select_f16_to_f32())(src, &mut dst[..src.len()]);
    }
}

/// Whether a file of `ty` reads exactly as f32: f32 itself, or a type
/// the conversion table widens to f32. The test every f32 engine puts
/// on a base or query before scanning it.
pub fn reads_as_f32(ty: ElementType) -> bool {
    ty == ElementType::F32 || ty.conversion_to(ElementType::F32) == ElementConversion::Widening
}

/// Decode `dst.len()` little-endian elements of `ty` from `src` into
/// f32: a copy for f32, the SIMD converter for f16, element by element
/// for the other types that widen to f32. The pread-and-unpack step of
/// the streaming engines, which scan raw record bytes rather than a
/// typed reader; `src` need not be aligned for `ty`.
pub fn widen_le_into(ty: ElementType, src: &[u8], dst: &mut [f32]) -> Result<(), String> {
    let n = dst.len();
    let need = n * ty.byte_width();
    if src.len() < need {
        return Err(format!("{} bytes hold fewer than {n} {ty} elements", src.len()));
    }
    fn by_value<T: WidenToF32>(src: &[u8], dst: &mut [f32]) {
        for (d, chunk) in dst.iter_mut().zip(src.chunks_exact(T::SIZE)) {
            *d = <T as Element>::from_le_bytes(chunk).as_f64() as f32;
        }
    }
    match ty {
        // A verbatim copy: the bytes are the f32s (little-endian host,
        // as every reader in the workspace assumes).
        ElementType::F32 => {
            // SAFETY: `dst` is a valid, aligned `[f32]` of `n` elements;
            // viewing it as `4n` bytes is sound for writing.
            let out = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, need) };
            out.copy_from_slice(&src[..need]);
        }
        // The SIMD converter over the bytes in place when they are
        // aligned for f16, as a streaming chunk's records are; element
        // by element otherwise.
        ElementType::F16 if (src.as_ptr() as usize).is_multiple_of(std::mem::align_of::<half::f16>()) => {
            // SAFETY: aligned, `need` bytes long, and every bit pattern
            // is a valid f16.
            let halves = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const half::f16, n) };
            half::f16::widen_into(halves, dst);
        }
        ElementType::F16 => by_value::<half::f16>(src, dst),
        ElementType::BF16 => by_value::<half::bf16>(src, dst),
        ElementType::U8 => by_value::<u8>(src, dst),
        ElementType::I8 => by_value::<i8>(src, dst),
        ElementType::U16 => by_value::<u16>(src, dst),
        ElementType::I16 => by_value::<i16>(src, dst),
        other => {
            return Err(format!(
                "{other} elements are not exact as f32; an f32 engine scans f32 and the types that widen to it \
                 (f16, bf16, and the 8- and 16-bit integers)"
            ));
        }
    }
    Ok(())
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

    /// **Exactly the types the table widens to f32 widen exactly**: for
    /// each `WidenToF32` type, its samples land on their own value as
    /// f32, and the set of implementors is the set of types whose
    /// `conversion_to(F32)` is `Widening`.
    #[test]
    fn widen_to_f32_is_the_tables_widening_set_and_is_exact() {
        fn exact<T: WidenToF32 + super::samples::Samples>() -> ElementType {
            let src = T::samples();
            let mut dst = vec![0f32; src.len()];
            T::widen_into(&src, &mut dst);
            for (s, d) in src.iter().zip(&dst) {
                let (a, b) = (s.as_f64(), *d as f64);
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{}: {a} became {b}", T::TYPE);
            }
            T::TYPE
        }
        let widened = [exact::<u8>(), exact::<i8>(), exact::<u16>(), exact::<i16>(), exact::<half::f16>(), exact::<half::bf16>()];
        let table: Vec<ElementType> = ElementType::ALL
            .iter()
            .copied()
            .filter(|t| t.conversion_to(ElementType::F32) == ElementConversion::Widening)
            .collect();
        assert_eq!(widened.to_vec(), table);
    }

    /// **`widen_le_into` decodes exactly the types that read as f32**,
    /// from unaligned bytes, to the values `as_f64` gives; every other
    /// type is refused by name.
    #[test]
    fn widen_le_into_decodes_every_type_that_reads_as_f32() {
        for &ty in ElementType::ALL {
            veks_datatypes::dispatch_element!(ty, T => {
                let values: Vec<T> = <T as super::samples::Samples>::samples();
                let mut bytes = vec![0xAAu8]; // one byte of offset: unaligned for every width
                let mut elem = vec![0u8; T::SIZE];
                for v in &values {
                    v.write_le_bytes(&mut elem);
                    bytes.extend_from_slice(&elem);
                }
                let mut out = vec![0f32; values.len()];
                let decoded = widen_le_into(ty, &bytes[1..], &mut out);
                assert_eq!(decoded.is_ok(), reads_as_f32(ty), "{ty}: {decoded:?}");
                match decoded {
                    Ok(()) => {
                        for (v, d) in values.iter().zip(&out) {
                            let (a, b) = (v.as_f64(), *d as f64);
                            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{ty}: {a} decoded as {b}");
                        }
                    }
                    Err(e) => assert!(e.contains(ty.name()), "{ty}: the refusal names the type"),
                }
            });
        }
        assert!(widen_le_into(ElementType::U8, &[1, 2], &mut [0f32; 3]).is_err(), "too few bytes");
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

/// The per-type sample values the integration tests share
/// (`tests/support/samples.rs`), included by path so that both sides
/// draw the same values. Relative to this file's directory.
#[cfg(test)]
#[path = "../../tests/support/samples.rs"]
mod samples;

/// The type matrix (SRD §13.10): every scenario here iterates
/// `ElementType::ALL`, with the sample values the integration tests
/// share, so a thirteenth type cannot be added without coverage.
#[cfg(test)]
mod type_matrix {
    use super::*;
    use std::path::{Path, PathBuf};
    use veks_datatypes::Container;

    use super::samples::{Samples, same_f64};

    /// Write uniform vectors of `T` with the extension `ext`: an `i32`
    /// dimension header per record, then little-endian elements.
    fn write_vec<T: Element>(dir: &Path, stem: &str, ext: &str, records: &[Vec<T>]) -> PathBuf {
        let path = dir.join(format!("{stem}.{ext}"));
        let mut bytes = Vec::new();
        for record in records {
            bytes.extend_from_slice(&(record.len() as i32).to_le_bytes());
            let mut buf = vec![0u8; T::SIZE];
            for &v in record {
                v.write_le_bytes(&mut buf);
                bytes.extend_from_slice(&buf);
            }
        }
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// Two records of `T`'s samples: the samples, then the samples
    /// reversed, so that the two records differ and the reversed one
    /// checks the record offset.
    fn sample_records<T: Samples>() -> Vec<Vec<T>> {
        let samples = T::samples();
        let mut reversed = samples.clone();
        reversed.reverse();
        vec![samples, reversed]
    }

    /// Every element type opens through `F64Vectors` under its canonical
    /// extension and, where one exists, its legacy spelling; it reports
    /// its own type, count and dimension, and every element reads as the
    /// `as_f64` of the sample written (NaN as NaN); a record past the end
    /// is empty, and the closure form reads the same.
    #[test]
    fn f64_vectors_reads_every_type_as_its_as_f64() {
        let dir = tempfile::tempdir().unwrap();
        for &ty in ElementType::ALL {
            dispatch_element!(ty, T => {
                let records = sample_records::<T>();
                let mut spellings = vec![ty.extension(Container::Vec)];
                spellings.extend(ty.legacy_extension(Container::Vec));
                for ext in spellings {
                    let path = write_vec(dir.path(), "samples", ext, &records);
                    let src = F64Vectors::open(&path).unwrap_or_else(|e| panic!("{ty} as .{ext}: {e}"));
                    assert_eq!(src.element_type(), ty, ".{ext}");
                    assert_eq!(src.count(), 2, "{ty}");
                    assert_eq!(src.dim(), records[0].len(), "{ty}");
                    for (i, record) in records.iter().enumerate() {
                        let got = src.get(i);
                        assert_eq!(got.len(), record.len(), "{ty} record {i}");
                        for (j, (&want, &have)) in record.iter().zip(&got).enumerate() {
                            assert!(same_f64(want.as_f64(), have), "{ty} record {i} element {j}: want {} got {have}", want.as_f64());
                        }
                    }
                    assert!(src.get(2).is_empty(), "{ty}: a record past the end is empty");
                    let get = src.into_get();
                    for (j, &want) in records[1].iter().enumerate() {
                        assert!(same_f64(want.as_f64(), get(1)[j]), "{ty} through into_get, element {j}");
                    }
                    assert!(get(2).is_empty(), "{ty}: into_get past the end is empty");
                }
            });
        }
    }

    /// `dispatch_reader!` opens a file of every element type as the
    /// `XvecReader<T>` of that type.
    #[test]
    fn dispatch_reader_opens_every_type_as_its_own_xvec_reader() {
        let dir = tempfile::tempdir().unwrap();
        for &ty in ElementType::ALL {
            let (path, want) = dispatch_element!(ty, T => {
                let path = write_vec(dir.path(), "dispatch", ty.extension(Container::Vec), &sample_records::<T>());
                (path, std::any::type_name::<vectordata::io::XvecReader<T>>())
            });
            let opened: Result<(&'static str, usize), String> = (|| {
                crate::dispatch_reader!(ty, &path, reader => {
                    Ok((std::any::type_name_of_val(&reader), reader.count()))
                })
            })();
            let (name, count) = opened.unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!(name, want, "{ty}");
            assert_eq!(count, 2, "{ty}");
        }
    }

    /// The machine epsilon is pinned per float type and absent for
    /// every integer type.
    #[test]
    fn machine_epsilon_is_pinned_per_float_and_none_per_integer() {
        for &ty in ElementType::ALL {
            let want = match ty {
                ElementType::F16 => Some(9.77e-4),
                ElementType::BF16 => Some(7.81e-3),
                ElementType::F32 => Some(1.19e-7),
                ElementType::F64 => Some(2.22e-16),
                _ => None,
            };
            assert_eq!(ty.machine_epsilon(), want, "{ty}");
            assert_eq!(ty.machine_epsilon().is_some(), ty.is_float(), "{ty}");
        }
    }

    /// The normalization threshold is pinned per float type, is the
    /// same at dims 1, 128 and 4096 (the dimension does not enter), and
    /// is absent for every integer type at every dim.
    #[test]
    fn normalization_threshold_is_pinned_per_float_and_independent_of_dim() {
        for &ty in ElementType::ALL {
            let want = match ty {
                ElementType::F16 | ElementType::BF16 => Some(1e-1),
                ElementType::F32 => Some(1e-5),
                ElementType::F64 => Some(1e-14),
                _ => None,
            };
            for dim in [1, 128, 4096] {
                assert_eq!(ty.normalization_threshold(dim), want, "{ty} at dim {dim}");
            }
        }
    }

    /// The SIMD distance kernels take exactly the IEEE floats — F64, F32
    /// and F16 — and none of the other nine types, each named.
    #[test]
    fn simd_distance_is_supported_for_exactly_the_ieee_floats() {
        use ElementType::*;
        const SIMD: [ElementType; 3] = [F64, F32, F16];
        const NO_SIMD: [ElementType; 9] = [U8, I8, U16, I16, U32, I32, U64, I64, BF16];
        for ty in SIMD {
            assert!(ty.supports_simd_distance(), "{ty} takes the SIMD kernels");
        }
        for ty in NO_SIMD {
            assert!(!ty.supports_simd_distance(), "{ty} does not take the SIMD kernels");
        }
        let mut named: Vec<ElementType> = SIMD.iter().chain(&NO_SIMD).copied().collect();
        named.sort();
        assert_eq!(named, ElementType::ALL, "every type is named in one of the two lists");
        for &ty in ElementType::ALL {
            assert_eq!(ty.supports_simd_distance(), SIMD.contains(&ty), "{ty}");
        }
    }
}
