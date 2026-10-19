// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector and scalar data format detection and metadata.

use std::path::Path;

use veks_datatypes::{Container, ElementKind, ElementType};

/// Supported data formats.
///
/// **Vector formats** (xvec family) store one record per vector as
/// `[dim:i32, data...]`. Extensions end in `vec` or `vecs`.
///
/// **Scalar formats** store one value per ordinal as a flat packed array
/// with no header. Extension is the bare type name (e.g., `.u8`, `.f32`).
///
/// **Container formats** (npy, parquet, slab) use their own structure.
///
/// A vector or scalar format is a [`Container`] and an [`ElementType`]
/// ([`container`](Self::container), [`element_type`](Self::element_type),
/// and [`of`](Self::of) to go the other way); the extension spellings
/// are the element model's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VecFormat {
    // ── Container formats ──────────────────────────────────────────
    /// NumPy `.npy` — float arrays (requires `npy` feature).
    Npy,
    /// Apache Parquet — columnar float vectors (requires `parquet` feature).
    Parquet,
    /// Page-oriented slab binary format (requires `slab` feature).
    Slab,

    // ── Vector formats (xvec: [dim:i32, data...] per record) ──────
    /// xvec float32 (4 bytes/element). Extension: `.fvec`
    Fvec,
    /// xvec float64 (8 bytes/element). Extension: `.dvec`
    Dvec,
    /// xvec float16 / half (2 bytes/element). Extension: `.mvec`
    Mvec,
    /// xvec bfloat16 (2 bytes/element). Extension: `.bf16vec`
    Bf16vec,
    /// xvec uint8 (1 byte/element). Extension: `.bvec` or `.u8vec`
    Bvec,
    /// xvec int8 (1 byte/element). Extension: `.i8vec`
    I8vec,
    /// xvec int16 (2 bytes/element). Extension: `.svec` or `.i16vec`
    Svec,
    /// xvec uint16 (2 bytes/element). Extension: `.u16vec`
    U16vec,
    /// xvec int32 (4 bytes/element). Extension: `.ivec` or `.i32vec`
    Ivec,
    /// xvec uint32 (4 bytes/element). Extension: `.u32vec`
    U32vec,
    /// xvec int64 (8 bytes/element). Extension: `.i64vec`
    I64vec,
    /// xvec uint64 (8 bytes/element). Extension: `.u64vec`
    U64vec,

    // ── Scalar formats (flat packed, no header) ───────────────────
    /// Scalar uint8 (1 byte/element). Extension: `.u8`
    ScalarU8,
    /// Scalar int8 (1 byte/element). Extension: `.i8`
    ScalarI8,
    /// Scalar uint16 (2 bytes/element). Extension: `.u16`
    ScalarU16,
    /// Scalar int16 (2 bytes/element). Extension: `.i16`
    ScalarI16,
    /// Scalar uint32 (4 bytes/element). Extension: `.u32`
    ScalarU32,
    /// Scalar int32 (4 bytes/element). Extension: `.i32`
    ScalarI32,
    /// Scalar uint64 (8 bytes/element). Extension: `.u64`
    ScalarU64,
    /// Scalar int64 (8 bytes/element). Extension: `.i64`
    ScalarI64,
    /// Scalar float16 (2 bytes/element). Extension: `.f16`
    ScalarF16,
    /// Scalar bfloat16 (2 bytes/element). Extension: `.bf16`
    ScalarBf16,
    /// Scalar float32 (4 bytes/element). Extension: `.f32`
    ScalarF32,
    /// Scalar float64 (8 bytes/element). Extension: `.f64`
    ScalarF64,
}

impl VecFormat {
    /// Every format.
    pub const ALL: &[VecFormat] = &[
        Self::Npy, Self::Parquet, Self::Slab,
        Self::Fvec, Self::Dvec, Self::Mvec, Self::Bf16vec,
        Self::Bvec, Self::I8vec, Self::Svec, Self::U16vec,
        Self::Ivec, Self::U32vec, Self::I64vec, Self::U64vec,
        Self::ScalarU8, Self::ScalarI8, Self::ScalarU16, Self::ScalarI16,
        Self::ScalarU32, Self::ScalarI32, Self::ScalarU64, Self::ScalarI64,
        Self::ScalarF16, Self::ScalarBf16, Self::ScalarF32, Self::ScalarF64,
    ];

    /// Human-readable name matching the file extension.
    pub fn name(self) -> &'static str {
        match self {
            Self::Npy => "npy",
            Self::Parquet => "parquet",
            Self::Slab => "slab",
            Self::Fvec => "fvec",
            Self::Dvec => "dvec",
            Self::Mvec => "mvec",
            Self::Bf16vec => "bf16vec",
            Self::Bvec => "bvec",
            Self::I8vec => "i8vec",
            Self::Svec => "svec",
            Self::U16vec => "u16vec",
            Self::Ivec => "ivec",
            Self::U32vec => "u32vec",
            Self::I64vec => "i64vec",
            Self::U64vec => "u64vec",
            Self::ScalarU8 => "u8",
            Self::ScalarI8 => "i8",
            Self::ScalarU16 => "u16",
            Self::ScalarI16 => "i16",
            Self::ScalarU32 => "u32",
            Self::ScalarI32 => "i32",
            Self::ScalarU64 => "u64",
            Self::ScalarI64 => "i64",
            Self::ScalarF16 => "f16",
            Self::ScalarBf16 => "bf16",
            Self::ScalarF32 => "f32",
            Self::ScalarF64 => "f64",
        }
    }

    /// The element type a vector or scalar format holds. `None` for the
    /// container formats, which describe their own.
    pub fn element_type(self) -> Option<ElementType> {
        Some(match self {
            Self::Npy | Self::Parquet | Self::Slab => return None,
            Self::Fvec | Self::ScalarF32 => ElementType::F32,
            Self::Dvec | Self::ScalarF64 => ElementType::F64,
            Self::Mvec | Self::ScalarF16 => ElementType::F16,
            Self::Bf16vec | Self::ScalarBf16 => ElementType::BF16,
            Self::Bvec | Self::ScalarU8 => ElementType::U8,
            Self::I8vec | Self::ScalarI8 => ElementType::I8,
            Self::Svec | Self::ScalarI16 => ElementType::I16,
            Self::U16vec | Self::ScalarU16 => ElementType::U16,
            Self::Ivec | Self::ScalarI32 => ElementType::I32,
            Self::U32vec | Self::ScalarU32 => ElementType::U32,
            Self::I64vec | Self::ScalarI64 => ElementType::I64,
            Self::U64vec | Self::ScalarU64 => ElementType::U64,
        })
    }

    /// The layout of a vector or scalar format: [`Container::Vec`] for
    /// the xvec family, [`Container::Scalar`] for the packed columns.
    /// `None` for the container formats.
    pub fn container(self) -> Option<Container> {
        match self {
            Self::Npy | Self::Parquet | Self::Slab => None,
            Self::Fvec | Self::Dvec | Self::Mvec | Self::Bf16vec
            | Self::Bvec | Self::I8vec | Self::Svec | Self::U16vec
            | Self::Ivec | Self::U32vec | Self::I64vec | Self::U64vec => Some(Container::Vec),
            Self::ScalarU8 | Self::ScalarI8 | Self::ScalarU16 | Self::ScalarI16
            | Self::ScalarU32 | Self::ScalarI32 | Self::ScalarU64 | Self::ScalarI64
            | Self::ScalarF16 | Self::ScalarBf16 | Self::ScalarF32 | Self::ScalarF64 => Some(Container::Scalar),
        }
    }

    /// The format that holds `element_type` in `container`. `None` for
    /// [`Container::Vvec`], which this crate reads through
    /// [`open_varlen`](crate::open_varlen) under the vector format's
    /// name rather than as a format of its own.
    pub fn of(container: Container, element_type: ElementType) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|f| f.container() == Some(container) && f.element_type() == Some(element_type))
    }

    /// True for the xvec family (vector formats with `[dim:i32, data...]` header).
    pub fn is_xvec(self) -> bool {
        self.container() == Some(Container::Vec)
    }

    /// True for scalar formats (flat packed, no header, one value per ordinal).
    pub fn is_scalar(self) -> bool {
        self.container() == Some(Container::Scalar)
    }

    /// True for integer element types (scalar or vector).
    pub fn is_integer(self) -> bool {
        self.element_type().is_some_and(ElementType::is_integer)
    }

    /// True for signed types: every float, and the signed integers.
    pub fn is_signed(self) -> bool {
        self.element_type().is_some_and(|t| t.kind() != ElementKind::Unsigned)
    }

    /// Bytes per element. Returns 0 for container formats.
    pub fn element_size(self) -> usize {
        self.element_type().map_or(0, ElementType::byte_width)
    }

    /// True if this format supports writing.
    pub fn is_writable(self) -> bool {
        self.is_xvec() || self.is_scalar() || self == Self::Slab
    }

    /// Preferred file extension for writing a new file of this format.
    ///
    /// For xvec formats this is the plural form, and the historical
    /// spelling where there is one (`fvecs`, `ivecs`, `bvecs`, `svecs`,
    /// `mvecs`, `dvecs`): the convention every published dataset is
    /// named by. A type with no historical spelling gets its typed
    /// plural (`i8vecs`, `bf16vecs`). For scalar and container formats
    /// it matches [`name`](Self::name) (they have no singular/plural
    /// split).
    ///
    /// Reading accepts every spelling; see
    /// [`from_extension`](Self::from_extension).
    pub fn preferred_extension(self) -> &'static str {
        match (self.container(), self.element_type()) {
            (Some(Container::Vec), Some(ty)) => {
                ty.legacy_extension(Container::Vec).unwrap_or(ty.extension(Container::Vec))
            }
            _ => self.name(),
        }
    }

    /// Normalize an extension string to the preferred canonical form.
    ///
    /// Accepts any recognized synonym (e.g. `"fvec"`, `"fvecs"`, `"f32vec"`)
    /// and returns the preferred extension string to use when writing a new
    /// file of the same format — the plural form for xvec/vvec types.
    pub fn canonical_extension(ext: &str) -> Option<&'static str> {
        Self::from_extension(ext).map(|f| f.preferred_extension())
    }

    /// Detect format from a file extension string.
    ///
    /// Every spelling the element model reads is accepted; a
    /// variable-length spelling (`fvvecs`) names no format here.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "slab" => Some(Self::Slab),
            "npy" => Some(Self::Npy),
            "parquet" => Some(Self::Parquet),
            other => {
                let (ty, container) = ElementType::parse_extension(other)?;
                Self::of(container, ty)
            }
        }
    }

    /// Detect format from a file path.
    ///
    /// Handles HDF5 `#` notation (returns `None` — HDF5 not supported).
    /// For directories, checks contents for npy/parquet files.
    pub fn detect(path: &Path) -> Option<Self> {
        let s = path.to_string_lossy();

        // HDF5 # notation → not supported
        if s.contains('#') {
            let hash_pos = s.rfind('#').unwrap();
            let file_part = &s[..hash_pos];
            if let Some(ext) = Path::new(file_part).extension().and_then(|e| e.to_str())
                && (ext == "hdf5" || ext == "h5") {
                return None;
            }
        }

        // Try file extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str())
            && let Some(fmt) = Self::from_extension(ext) {
            return Some(fmt);
        }

        // For directories, detect by contents
        if path.is_dir() {
            return Self::detect_from_directory(path);
        }

        None
    }

    /// Detect format from directory contents.
    fn detect_from_directory(dir: &Path) -> Option<Self> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut has_npy = false;
        let mut has_parquet = false;
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                match ext {
                    "npy" => has_npy = true,
                    "parquet" => has_parquet = true,
                    _ => {}
                }
            }
        }
        if has_npy { return Some(Self::Npy); }
        if has_parquet { return Some(Self::Parquet); }
        None
    }

    /// Human-readable data type name.
    pub fn data_type_name(self) -> &'static str {
        match self.element_type() {
            Some(ElementType::F32) => "float32",
            Some(ElementType::F64) => "float64",
            Some(ElementType::F16) => "float16",
            Some(ElementType::BF16) => "bfloat16",
            Some(ElementType::U8) => "uint8",
            Some(ElementType::I8) => "int8",
            Some(ElementType::U16) => "uint16",
            Some(ElementType::I16) => "int16",
            Some(ElementType::U32) => "uint32",
            Some(ElementType::I32) => "int32",
            Some(ElementType::U64) => "uint64",
            Some(ElementType::I64) => "int64",
            _ => "float",
        }
    }
}

impl std::fmt::Display for VecFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Extension detection ────────────────────────────────────────

    #[test]
    fn test_legacy_extensions() {
        assert_eq!(VecFormat::from_extension("fvec"), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::from_extension("fvecs"), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::from_extension("ivec"), Some(VecFormat::Ivec));
        assert_eq!(VecFormat::from_extension("ivecs"), Some(VecFormat::Ivec));
        assert_eq!(VecFormat::from_extension("bvec"), Some(VecFormat::Bvec));
        assert_eq!(VecFormat::from_extension("svec"), Some(VecFormat::Svec));
        assert_eq!(VecFormat::from_extension("dvec"), Some(VecFormat::Dvec));
        assert_eq!(VecFormat::from_extension("mvec"), Some(VecFormat::Mvec));
    }

    #[test]
    fn test_explicit_vec_extensions() {
        assert_eq!(VecFormat::from_extension("u8vec"), Some(VecFormat::Bvec));
        assert_eq!(VecFormat::from_extension("u8vecs"), Some(VecFormat::Bvec));
        assert_eq!(VecFormat::from_extension("i8vec"), Some(VecFormat::I8vec));
        assert_eq!(VecFormat::from_extension("i8vecs"), Some(VecFormat::I8vec));
        assert_eq!(VecFormat::from_extension("u16vec"), Some(VecFormat::U16vec));
        assert_eq!(VecFormat::from_extension("u16vecs"), Some(VecFormat::U16vec));
        assert_eq!(VecFormat::from_extension("i16vec"), Some(VecFormat::Svec));
        assert_eq!(VecFormat::from_extension("i16vecs"), Some(VecFormat::Svec));
        assert_eq!(VecFormat::from_extension("u32vec"), Some(VecFormat::U32vec));
        assert_eq!(VecFormat::from_extension("i32vec"), Some(VecFormat::Ivec));
        assert_eq!(VecFormat::from_extension("i64vec"), Some(VecFormat::I64vec));
        assert_eq!(VecFormat::from_extension("u64vec"), Some(VecFormat::U64vec));
        assert_eq!(VecFormat::from_extension("f32vecs"), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::from_extension("bf16vec"), Some(VecFormat::Bf16vec));
        assert_eq!(VecFormat::from_extension("bf16vecs"), Some(VecFormat::Bf16vec));
        assert_eq!(VecFormat::from_extension("f16vecs"), Some(VecFormat::Mvec));
    }

    /// The scalar set is every element type: a column of any of them.
    #[test]
    fn test_scalar_extensions() {
        assert_eq!(VecFormat::from_extension("u8"), Some(VecFormat::ScalarU8));
        assert_eq!(VecFormat::from_extension("i8"), Some(VecFormat::ScalarI8));
        assert_eq!(VecFormat::from_extension("u16"), Some(VecFormat::ScalarU16));
        assert_eq!(VecFormat::from_extension("i16"), Some(VecFormat::ScalarI16));
        assert_eq!(VecFormat::from_extension("u32"), Some(VecFormat::ScalarU32));
        assert_eq!(VecFormat::from_extension("i32"), Some(VecFormat::ScalarI32));
        assert_eq!(VecFormat::from_extension("u64"), Some(VecFormat::ScalarU64));
        assert_eq!(VecFormat::from_extension("i64"), Some(VecFormat::ScalarI64));
        assert_eq!(VecFormat::from_extension("f16"), Some(VecFormat::ScalarF16));
        assert_eq!(VecFormat::from_extension("bf16"), Some(VecFormat::ScalarBf16));
        assert_eq!(VecFormat::from_extension("f32"), Some(VecFormat::ScalarF32));
        assert_eq!(VecFormat::from_extension("f64"), Some(VecFormat::ScalarF64));
    }

    #[test]
    fn test_unknown_extension() {
        assert_eq!(VecFormat::from_extension("txt"), None);
        assert_eq!(VecFormat::from_extension("csv"), None);
        assert_eq!(VecFormat::from_extension("fvvecs"), None, "variable-length files are not a VecFormat");
    }

    // ── The element model ──────────────────────────────────────────

    /// Every vector and scalar format is one (layout, element type) pair
    /// and back, and every element type has both layouts.
    #[test]
    fn vec_format_names_every_element_type_and_layout() {
        for &ty in ElementType::ALL {
            for container in [Container::Vec, Container::Scalar] {
                let fmt = VecFormat::of(container, ty).unwrap_or_else(|| panic!("{ty} in {container:?}"));
                assert_eq!(fmt.element_type(), Some(ty));
                assert_eq!(fmt.container(), Some(container));
                assert_eq!(fmt.element_size(), ty.byte_width());
                assert_eq!(VecFormat::from_extension(ty.extension(container)), Some(fmt));
                for alias in ty.aliases(container) {
                    assert_eq!(VecFormat::from_extension(alias), Some(fmt), "{alias}");
                }
            }
            assert_eq!(VecFormat::of(Container::Vvec, ty), None);
        }
        for fmt in VecFormat::ALL {
            match (fmt.container(), fmt.element_type()) {
                (None, None) => assert!(!fmt.is_xvec() && !fmt.is_scalar()),
                (Some(_), Some(_)) => {}
                other => panic!("{fmt}: {other:?}"),
            }
        }
    }

    /// What a new file is called does not change with the model: the
    /// historical plural where there is one, the typed plural otherwise.
    #[test]
    fn preferred_extension_keeps_the_historical_spelling() {
        for (fmt, want) in [
            (VecFormat::Fvec, "fvecs"), (VecFormat::Dvec, "dvecs"), (VecFormat::Mvec, "mvecs"),
            (VecFormat::Bvec, "bvecs"), (VecFormat::Svec, "svecs"), (VecFormat::Ivec, "ivecs"),
            (VecFormat::I8vec, "i8vecs"), (VecFormat::U16vec, "u16vecs"), (VecFormat::U32vec, "u32vecs"),
            (VecFormat::I64vec, "i64vecs"), (VecFormat::U64vec, "u64vecs"), (VecFormat::Bf16vec, "bf16vecs"),
            (VecFormat::ScalarU8, "u8"), (VecFormat::ScalarF32, "f32"), (VecFormat::ScalarBf16, "bf16"),
            (VecFormat::Slab, "slab"),
        ] {
            assert_eq!(fmt.preferred_extension(), want, "{fmt:?}");
        }
    }

    // ── Classification ─────────────────────────────────────────────

    #[test]
    fn test_is_xvec() {
        assert!(VecFormat::Fvec.is_xvec());
        assert!(VecFormat::Ivec.is_xvec());
        assert!(VecFormat::Bvec.is_xvec());
        assert!(VecFormat::I8vec.is_xvec());
        assert!(VecFormat::U16vec.is_xvec());
        assert!(VecFormat::U32vec.is_xvec());
        assert!(VecFormat::I64vec.is_xvec());
        assert!(VecFormat::U64vec.is_xvec());
        assert!(VecFormat::Bf16vec.is_xvec());
        assert!(!VecFormat::ScalarU8.is_xvec());
        assert!(!VecFormat::ScalarI32.is_xvec());
        assert!(!VecFormat::Npy.is_xvec());
    }

    #[test]
    fn test_is_scalar() {
        assert!(VecFormat::ScalarU8.is_scalar());
        assert!(VecFormat::ScalarI8.is_scalar());
        assert!(VecFormat::ScalarU16.is_scalar());
        assert!(VecFormat::ScalarI16.is_scalar());
        assert!(VecFormat::ScalarU32.is_scalar());
        assert!(VecFormat::ScalarI32.is_scalar());
        assert!(VecFormat::ScalarU64.is_scalar());
        assert!(VecFormat::ScalarI64.is_scalar());
        assert!(VecFormat::ScalarF32.is_scalar());
        assert!(VecFormat::ScalarBf16.is_scalar());
        assert!(!VecFormat::Fvec.is_scalar());
        assert!(!VecFormat::Ivec.is_scalar());
    }

    #[test]
    fn test_is_integer() {
        assert!(VecFormat::Ivec.is_integer());
        assert!(VecFormat::Bvec.is_integer());
        assert!(VecFormat::ScalarU8.is_integer());
        assert!(VecFormat::ScalarI64.is_integer());
        assert!(!VecFormat::Fvec.is_integer());
        assert!(!VecFormat::Dvec.is_integer());
        assert!(!VecFormat::Mvec.is_integer());
        assert!(!VecFormat::Bf16vec.is_integer());
        assert!(!VecFormat::ScalarF32.is_integer());
    }

    #[test]
    fn test_is_signed() {
        assert!(VecFormat::Ivec.is_signed());
        assert!(VecFormat::Svec.is_signed());
        assert!(VecFormat::I8vec.is_signed());
        assert!(VecFormat::ScalarI8.is_signed());
        assert!(VecFormat::Fvec.is_signed());
        assert!(VecFormat::Bf16vec.is_signed());
        assert!(!VecFormat::Bvec.is_signed());
        assert!(!VecFormat::U16vec.is_signed());
        assert!(!VecFormat::ScalarU32.is_signed());
    }

    // ── Element sizes ──────────────────────────────────────────────

    #[test]
    fn test_element_sizes() {
        assert_eq!(VecFormat::Bvec.element_size(), 1);
        assert_eq!(VecFormat::I8vec.element_size(), 1);
        assert_eq!(VecFormat::ScalarU8.element_size(), 1);
        assert_eq!(VecFormat::ScalarI8.element_size(), 1);
        assert_eq!(VecFormat::Svec.element_size(), 2);
        assert_eq!(VecFormat::U16vec.element_size(), 2);
        assert_eq!(VecFormat::Mvec.element_size(), 2);
        assert_eq!(VecFormat::Bf16vec.element_size(), 2);
        assert_eq!(VecFormat::ScalarU16.element_size(), 2);
        assert_eq!(VecFormat::ScalarI16.element_size(), 2);
        assert_eq!(VecFormat::Fvec.element_size(), 4);
        assert_eq!(VecFormat::Ivec.element_size(), 4);
        assert_eq!(VecFormat::U32vec.element_size(), 4);
        assert_eq!(VecFormat::ScalarU32.element_size(), 4);
        assert_eq!(VecFormat::ScalarI32.element_size(), 4);
        assert_eq!(VecFormat::ScalarF32.element_size(), 4);
        assert_eq!(VecFormat::Dvec.element_size(), 8);
        assert_eq!(VecFormat::I64vec.element_size(), 8);
        assert_eq!(VecFormat::U64vec.element_size(), 8);
        assert_eq!(VecFormat::ScalarU64.element_size(), 8);
        assert_eq!(VecFormat::ScalarI64.element_size(), 8);
        assert_eq!(VecFormat::ScalarF64.element_size(), 8);
        assert_eq!(VecFormat::Npy.element_size(), 0);
    }

    // ── Aliases ────────────────────────────────────────────────────

    #[test]
    fn test_u8vec_is_bvec() {
        // u8vec and bvec are the same format
        assert_eq!(VecFormat::from_extension("u8vec"), VecFormat::from_extension("bvec"));
    }

    #[test]
    fn test_i16vec_is_svec() {
        assert_eq!(VecFormat::from_extension("i16vec"), VecFormat::from_extension("svec"));
    }

    #[test]
    fn test_i32vec_is_ivec() {
        assert_eq!(VecFormat::from_extension("i32vec"), VecFormat::from_extension("ivec"));
    }

    // ── Path detection ─────────────────────────────────────────────

    #[test]
    fn test_detect_scalar_path() {
        assert_eq!(VecFormat::detect(Path::new("data.u8")), Some(VecFormat::ScalarU8));
        assert_eq!(VecFormat::detect(Path::new("data.i32")), Some(VecFormat::ScalarI32));
        assert_eq!(VecFormat::detect(Path::new("data.u64")), Some(VecFormat::ScalarU64));
        assert_eq!(VecFormat::detect(Path::new("scale.f32")), Some(VecFormat::ScalarF32));
    }

    #[test]
    fn test_detect_vec_path() {
        assert_eq!(VecFormat::detect(Path::new("data.u16vec")), Some(VecFormat::U16vec));
        assert_eq!(VecFormat::detect(Path::new("data.i64vecs")), Some(VecFormat::I64vec));
        assert_eq!(VecFormat::detect(Path::new("data.bf16vecs")), Some(VecFormat::Bf16vec));
    }

    #[test]
    fn data_type_names_and_display_cover_every_format() {
        let names = [
            (ElementType::U8, "uint8"), (ElementType::I8, "int8"),
            (ElementType::U16, "uint16"), (ElementType::I16, "int16"),
            (ElementType::U32, "uint32"), (ElementType::I32, "int32"),
            (ElementType::U64, "uint64"), (ElementType::I64, "int64"),
            (ElementType::F16, "float16"), (ElementType::BF16, "bfloat16"),
            (ElementType::F32, "float32"), (ElementType::F64, "float64"),
        ];
        for &fmt in VecFormat::ALL {
            assert_eq!(fmt.to_string(), fmt.name());
            let want = match fmt.element_type() {
                Some(ty) => names.iter().find(|(t, _)| *t == ty).map(|(_, n)| *n).unwrap(),
                None => "float",
            };
            assert_eq!(fmt.data_type_name(), want, "{fmt}");
        }
    }

    #[test]
    fn a_directory_is_detected_by_what_it_holds() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(VecFormat::detect(tmp.path()), None);
        std::fs::write(tmp.path().join("part-0.parquet"), b"").unwrap();
        assert_eq!(VecFormat::detect(tmp.path()), Some(VecFormat::Parquet));
        std::fs::write(tmp.path().join("arr.npy"), b"").unwrap();
        assert_eq!(VecFormat::detect(tmp.path()), Some(VecFormat::Npy), "npy wins when both are present");
        assert_eq!(VecFormat::detect(&tmp.path().join("absent")), None);
    }

    #[test]
    fn test_hdf5_path_returns_none() {
        assert_eq!(VecFormat::detect(Path::new("data.hdf5#train")), None);
        assert_eq!(VecFormat::detect(Path::new("data.h5#vectors")), None);
    }

    // ── Writable ───────────────────────────────────────────────────

    #[test]
    fn test_all_xvec_and_scalar_writable() {
        for fmt in VecFormat::ALL.iter().filter(|f| f.is_xvec() || f.is_scalar()) {
            assert!(fmt.is_writable(), "{} should be writable", fmt);
        }
    }

    #[test]
    fn test_container_not_writable() {
        assert!(!VecFormat::Npy.is_writable());
        assert!(!VecFormat::Parquet.is_writable());
    }

    // ── Canonical extension roundtrip ──────────────────────────────

    #[test]
    fn test_canonical_extension_roundtrip() {
        // Every format's name should detect back to itself
        for &fmt in VecFormat::ALL {
            let ext = fmt.name();
            let detected = VecFormat::from_extension(ext);
            assert_eq!(detected, Some(fmt), "roundtrip failed for {}", ext);
        }
    }

    /// `preferred_extension()` returns the plural form for xvec types and
    /// equals `name()` for everything else. Reading the plural form must
    /// always round-trip back to the same format.
    #[test]
    fn preferred_extension_round_trips() {
        for &fmt in VecFormat::ALL {
            let ext = fmt.preferred_extension();
            let detected = VecFormat::from_extension(ext);
            assert_eq!(detected, Some(fmt), "round-trip via preferred_extension failed for {:?}", fmt);
            if fmt.is_xvec() {
                assert!(ext.ends_with('s'),
                    "xvec format {:?} should have plural preferred_extension, got {:?}", fmt, ext);
                // The singular form must still parse to the same format.
                assert_eq!(VecFormat::from_extension(fmt.name()), Some(fmt));
            } else {
                // Scalars and containers: plural and singular are identical.
                assert_eq!(ext, fmt.name(), "non-xvec format {:?} should have no plural form", fmt);
            }
        }
    }

    /// `canonical_extension` is the project-wide normalizer: a singular
    /// input must produce the plural output (the form we now write).
    #[test]
    fn canonical_extension_normalizes_singular_to_plural() {
        assert_eq!(VecFormat::canonical_extension("fvec"), Some("fvecs"));
        assert_eq!(VecFormat::canonical_extension("fvecs"), Some("fvecs"));
        assert_eq!(VecFormat::canonical_extension("ivec"), Some("ivecs"));
        assert_eq!(VecFormat::canonical_extension("u8vec"), Some("bvecs"));
        assert_eq!(VecFormat::canonical_extension("f32vecs"), Some("fvecs"));
        assert_eq!(VecFormat::canonical_extension("bf16vec"), Some("bf16vecs"));
        // Scalars and containers: no plural change.
        assert_eq!(VecFormat::canonical_extension("u8"), Some("u8"));
        assert_eq!(VecFormat::canonical_extension("slab"), Some("slab"));
        assert_eq!(VecFormat::canonical_extension("parquet"), Some("parquet"));
    }
}
