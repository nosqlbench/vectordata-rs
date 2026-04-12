// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector and scalar data format detection and metadata.

use std::path::Path;

/// Supported data formats.
///
/// **Vector formats** (xvec family) store one record per vector as
/// `[dim:i32, data...]`. Extensions end in `vec` or `vecs`.
///
/// **Scalar formats** store one value per ordinal as a flat packed array
/// with no header. Extension is the bare type name (e.g., `.u8`, `.i32`).
///
/// **Container formats** (npy, parquet, slab) use their own structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
}

impl VecFormat {
    /// Human-readable name matching the file extension.
    pub fn name(self) -> &'static str {
        match self {
            Self::Npy => "npy",
            Self::Parquet => "parquet",
            Self::Slab => "slab",
            Self::Fvec => "fvec",
            Self::Dvec => "dvec",
            Self::Mvec => "mvec",
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
        }
    }

    /// True for the xvec family (vector formats with `[dim:i32, data...]` header).
    pub fn is_xvec(self) -> bool {
        matches!(self,
            Self::Fvec | Self::Dvec | Self::Mvec |
            Self::Bvec | Self::I8vec |
            Self::Svec | Self::U16vec |
            Self::Ivec | Self::U32vec |
            Self::I64vec | Self::U64vec
        )
    }

    /// True for scalar formats (flat packed, no header, one value per ordinal).
    pub fn is_scalar(self) -> bool {
        matches!(self,
            Self::ScalarU8 | Self::ScalarI8 |
            Self::ScalarU16 | Self::ScalarI16 |
            Self::ScalarU32 | Self::ScalarI32 |
            Self::ScalarU64 | Self::ScalarI64
        )
    }

    /// True for integer element types (scalar or vector).
    pub fn is_integer(self) -> bool {
        matches!(self,
            Self::Bvec | Self::I8vec | Self::Svec | Self::U16vec |
            Self::Ivec | Self::U32vec | Self::I64vec | Self::U64vec |
            Self::ScalarU8 | Self::ScalarI8 | Self::ScalarU16 | Self::ScalarI16 |
            Self::ScalarU32 | Self::ScalarI32 | Self::ScalarU64 | Self::ScalarI64
        )
    }

    /// True for signed types.
    pub fn is_signed(self) -> bool {
        matches!(self,
            Self::I8vec | Self::Svec | Self::Ivec | Self::I64vec |
            Self::ScalarI8 | Self::ScalarI16 | Self::ScalarI32 | Self::ScalarI64 |
            Self::Fvec | Self::Dvec | Self::Mvec
        )
    }

    /// Bytes per element. Returns 0 for container formats.
    pub fn element_size(self) -> usize {
        match self {
            Self::Bvec | Self::I8vec | Self::ScalarU8 | Self::ScalarI8 => 1,
            Self::Mvec | Self::Svec | Self::U16vec | Self::ScalarU16 | Self::ScalarI16 => 2,
            Self::Fvec | Self::Ivec | Self::U32vec | Self::ScalarU32 | Self::ScalarI32 => 4,
            Self::Dvec | Self::I64vec | Self::U64vec | Self::ScalarU64 | Self::ScalarI64 => 8,
            Self::Npy | Self::Parquet | Self::Slab => 0,
        }
    }

    /// True if this format supports writing.
    pub fn is_writable(self) -> bool {
        self.is_xvec() || self.is_scalar() || self == Self::Slab
    }

    /// Normalize extension variants (e.g., `"fvecs"` → `"fvec"`).
    pub fn canonical_extension(ext: &str) -> Option<&'static str> {
        Self::from_extension(ext).map(|f| f.name())
    }

    /// Detect format from a file extension string.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            // Float vector
            "fvec" | "fvecs" => Some(Self::Fvec),
            "dvec" | "dvecs" => Some(Self::Dvec),
            "mvec" | "mvecs" => Some(Self::Mvec),
            // Integer vector (legacy names)
            "bvec" | "bvecs" => Some(Self::Bvec),
            "svec" | "svecs" => Some(Self::Svec),
            "ivec" | "ivecs" => Some(Self::Ivec),
            // Integer vector (explicit names)
            "u8vec" | "u8vecs" => Some(Self::Bvec),
            "i8vec" | "i8vecs" => Some(Self::I8vec),
            "u16vec" | "u16vecs" => Some(Self::U16vec),
            "i16vec" | "i16vecs" => Some(Self::Svec),
            "u32vec" | "u32vecs" => Some(Self::U32vec),
            "i32vec" | "i32vecs" => Some(Self::Ivec),
            "i64vec" | "i64vecs" => Some(Self::I64vec),
            "u64vec" | "u64vecs" => Some(Self::U64vec),
            // Scalar (flat packed)
            "u8" => Some(Self::ScalarU8),
            "i8" => Some(Self::ScalarI8),
            "u16" => Some(Self::ScalarU16),
            "i16" => Some(Self::ScalarI16),
            "u32" => Some(Self::ScalarU32),
            "i32" => Some(Self::ScalarI32),
            "u64" => Some(Self::ScalarU64),
            "i64" => Some(Self::ScalarI64),
            // Container
            "slab" => Some(Self::Slab),
            "npy" => Some(Self::Npy),
            "parquet" => Some(Self::Parquet),
            _ => None,
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
        match self {
            Self::Fvec => "float32",
            Self::Dvec => "float64",
            Self::Mvec => "float16",
            Self::Bvec | Self::ScalarU8 => "uint8",
            Self::I8vec | Self::ScalarI8 => "int8",
            Self::U16vec | Self::ScalarU16 => "uint16",
            Self::Svec | Self::ScalarI16 => "int16",
            Self::U32vec | Self::ScalarU32 => "uint32",
            Self::Ivec | Self::ScalarI32 => "int32",
            Self::U64vec | Self::ScalarU64 => "uint64",
            Self::I64vec | Self::ScalarI64 => "int64",
            Self::Npy | Self::Parquet | Self::Slab => "float",
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
    }

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
    }

    #[test]
    fn test_unknown_extension() {
        assert_eq!(VecFormat::from_extension("txt"), None);
        assert_eq!(VecFormat::from_extension("csv"), None);
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
    }

    #[test]
    fn test_is_signed() {
        assert!(VecFormat::Ivec.is_signed());
        assert!(VecFormat::Svec.is_signed());
        assert!(VecFormat::I8vec.is_signed());
        assert!(VecFormat::ScalarI8.is_signed());
        assert!(VecFormat::Fvec.is_signed());
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
        assert_eq!(VecFormat::ScalarU16.element_size(), 2);
        assert_eq!(VecFormat::ScalarI16.element_size(), 2);
        assert_eq!(VecFormat::Fvec.element_size(), 4);
        assert_eq!(VecFormat::Ivec.element_size(), 4);
        assert_eq!(VecFormat::U32vec.element_size(), 4);
        assert_eq!(VecFormat::ScalarU32.element_size(), 4);
        assert_eq!(VecFormat::ScalarI32.element_size(), 4);
        assert_eq!(VecFormat::Dvec.element_size(), 8);
        assert_eq!(VecFormat::I64vec.element_size(), 8);
        assert_eq!(VecFormat::U64vec.element_size(), 8);
        assert_eq!(VecFormat::ScalarU64.element_size(), 8);
        assert_eq!(VecFormat::ScalarI64.element_size(), 8);
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
    }

    #[test]
    fn test_detect_vec_path() {
        assert_eq!(VecFormat::detect(Path::new("data.u16vec")), Some(VecFormat::U16vec));
        assert_eq!(VecFormat::detect(Path::new("data.i64vecs")), Some(VecFormat::I64vec));
    }

    #[test]
    fn test_hdf5_path_returns_none() {
        assert_eq!(VecFormat::detect(Path::new("data.hdf5#train")), None);
        assert_eq!(VecFormat::detect(Path::new("data.h5#vectors")), None);
    }

    // ── Writable ───────────────────────────────────────────────────

    #[test]
    fn test_all_xvec_writable() {
        for fmt in [VecFormat::Fvec, VecFormat::Ivec, VecFormat::Bvec, VecFormat::Dvec,
                    VecFormat::Mvec, VecFormat::Svec, VecFormat::I8vec, VecFormat::U16vec,
                    VecFormat::U32vec, VecFormat::I64vec, VecFormat::U64vec] {
            assert!(fmt.is_writable(), "{} should be writable", fmt);
        }
    }

    #[test]
    fn test_all_scalar_writable() {
        for fmt in [VecFormat::ScalarU8, VecFormat::ScalarI8, VecFormat::ScalarU16,
                    VecFormat::ScalarI16, VecFormat::ScalarU32, VecFormat::ScalarI32,
                    VecFormat::ScalarU64, VecFormat::ScalarI64] {
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
        let all_formats = [
            VecFormat::Fvec, VecFormat::Dvec, VecFormat::Mvec,
            VecFormat::Bvec, VecFormat::I8vec, VecFormat::Svec, VecFormat::U16vec,
            VecFormat::Ivec, VecFormat::U32vec, VecFormat::I64vec, VecFormat::U64vec,
            VecFormat::ScalarU8, VecFormat::ScalarI8, VecFormat::ScalarU16, VecFormat::ScalarI16,
            VecFormat::ScalarU32, VecFormat::ScalarI32, VecFormat::ScalarU64, VecFormat::ScalarI64,
            VecFormat::Slab, VecFormat::Npy, VecFormat::Parquet,
        ];
        for fmt in all_formats {
            let ext = fmt.name();
            let detected = VecFormat::from_extension(ext);
            assert_eq!(detected, Some(fmt), "roundtrip failed for {}", ext);
        }
    }
}
