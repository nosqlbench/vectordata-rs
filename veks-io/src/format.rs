// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vector data format detection and metadata.

use std::path::Path;

/// Supported vector data formats.
///
/// xvec formats (fvec, ivec, bvec, dvec, mvec, svec) store one record per
/// vector as `[dim:element_type, data...]`. Container formats (npy, parquet,
/// slab) use their own internal structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecFormat {
    /// NumPy `.npy` — float arrays (requires `npy` feature).
    Npy,
    /// Apache Parquet — columnar float vectors (requires `parquet` feature).
    Parquet,
    /// xvec float32 (4 bytes/element).
    Fvec,
    /// xvec int32 (4 bytes/element).
    Ivec,
    /// xvec uint8 (1 byte/element, stored in 4-byte groups).
    Bvec,
    /// xvec float64 (8 bytes/element).
    Dvec,
    /// xvec float16 / half (2 bytes/element).
    Mvec,
    /// xvec int16 / short (2 bytes/element).
    Svec,
    /// Page-oriented slab binary format (requires `slab` feature).
    Slab,
}

impl VecFormat {
    /// Human-readable name matching the file extension.
    pub fn name(self) -> &'static str {
        match self {
            Self::Npy => "npy",
            Self::Parquet => "parquet",
            Self::Fvec => "fvec",
            Self::Ivec => "ivec",
            Self::Bvec => "bvec",
            Self::Dvec => "dvec",
            Self::Mvec => "mvec",
            Self::Svec => "svec",
            Self::Slab => "slab",
        }
    }

    /// True for the xvec family (fvec, ivec, bvec, dvec, mvec, svec).
    pub fn is_xvec(self) -> bool {
        matches!(self, Self::Fvec | Self::Ivec | Self::Bvec | Self::Dvec | Self::Mvec | Self::Svec)
    }

    /// Bytes per element for xvec formats. Returns 0 for container formats.
    pub fn element_size(self) -> usize {
        match self {
            Self::Fvec | Self::Ivec | Self::Bvec => 4,
            Self::Dvec => 8,
            Self::Mvec | Self::Svec => 2,
            Self::Npy | Self::Parquet | Self::Slab => 0,
        }
    }

    /// True if this format supports writing.
    pub fn is_writable(self) -> bool {
        match self {
            Self::Fvec | Self::Ivec | Self::Bvec | Self::Dvec | Self::Mvec | Self::Svec | Self::Slab => true,
            Self::Npy | Self::Parquet => false,
        }
    }

    /// Normalize extension variants (e.g., `"fvecs"` → `"fvec"`).
    pub fn canonical_extension(ext: &str) -> Option<&'static str> {
        match ext {
            "fvec" | "fvecs" => Some("fvec"),
            "ivec" | "ivecs" => Some("ivec"),
            "bvec" | "bvecs" => Some("bvec"),
            "dvec" | "dvecs" => Some("dvec"),
            "mvec" | "mvecs" => Some("mvec"),
            "svec" | "svecs" => Some("svec"),
            "slab" => Some("slab"),
            "npy" => Some("npy"),
            "parquet" => Some("parquet"),
            _ => None,
        }
    }

    /// Detect format from a file extension string.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "fvec" | "fvecs" => Some(Self::Fvec),
            "ivec" | "ivecs" => Some(Self::Ivec),
            "bvec" | "bvecs" => Some(Self::Bvec),
            "dvec" | "dvecs" => Some(Self::Dvec),
            "mvec" | "mvecs" => Some(Self::Mvec),
            "svec" | "svecs" => Some(Self::Svec),
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
            if let Some(ext) = Path::new(file_part).extension().and_then(|e| e.to_str()) {
                if ext == "hdf5" || ext == "h5" {
                    return None;
                }
            }
        }

        // Try file extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if let Some(fmt) = Self::from_extension(ext) {
                return Some(fmt);
            }
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
            Self::Ivec => "int32",
            Self::Bvec => "uint8",
            Self::Dvec => "float64",
            Self::Mvec => "float16",
            Self::Svec => "int16",
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

    #[test]
    fn test_from_extension() {
        assert_eq!(VecFormat::from_extension("fvec"), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::from_extension("fvecs"), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::from_extension("ivec"), Some(VecFormat::Ivec));
        assert_eq!(VecFormat::from_extension("mvec"), Some(VecFormat::Mvec));
        assert_eq!(VecFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_detect_from_path() {
        assert_eq!(VecFormat::detect(Path::new("data.fvec")), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::detect(Path::new("data.fvecs")), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::detect(Path::new("data.ivec")), Some(VecFormat::Ivec));
        assert_eq!(VecFormat::detect(Path::new("data.parquet")), Some(VecFormat::Parquet));
        assert_eq!(VecFormat::detect(Path::new("data.txt")), None);
    }

    #[test]
    fn test_hdf5_path_returns_none() {
        assert_eq!(VecFormat::detect(Path::new("data.hdf5#train")), None);
        assert_eq!(VecFormat::detect(Path::new("data.h5#vectors")), None);
    }

    #[test]
    fn test_element_size() {
        assert_eq!(VecFormat::Fvec.element_size(), 4);
        assert_eq!(VecFormat::Mvec.element_size(), 2);
        assert_eq!(VecFormat::Dvec.element_size(), 8);
        assert_eq!(VecFormat::Npy.element_size(), 0);
    }

    #[test]
    fn test_is_writable() {
        assert!(VecFormat::Fvec.is_writable());
        assert!(VecFormat::Ivec.is_writable());
        assert!(!VecFormat::Npy.is_writable());
        assert!(!VecFormat::Parquet.is_writable());
    }
}
