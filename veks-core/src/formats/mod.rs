// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vector data format support.
//!
//! This module provides a unified interface for reading and writing vector data
//! in various formats (xvec, npy, parquet, slab). It is designed to be used by
//! multiple tools — not just `convert` — and follows a registry pattern similar
//! to the Java `VectorFileIO` approach.

// Wire format codecs re-exported from vectordata so that both crates share
// a single definition of MNode, PNode, ANode, etc.
#[allow(dead_code)]
pub use vectordata::formats::mnode;
#[allow(dead_code)]
pub use vectordata::formats::pnode;
#[allow(dead_code)]
pub use vectordata::formats::anode;
#[allow(dead_code)]
pub use vectordata::formats::anode_vernacular;

// Arrow-based parquet-to-mnode compiler — stays in veks because it depends on
// the `arrow` crate which is not a vectordata dependency.
pub mod facet;
pub mod parquet_compiler;
pub mod convert;
pub mod reader;
pub mod writer;

use std::fmt;
use std::path::Path;

use clap::ValueEnum;

/// A vector data format recognized by veks.
///
/// Each variant maps to a file extension and encodes vectors with a specific
/// element type and on-disk layout. The xvec family (`fvec`, `ivec`, etc.)
/// stores one `[dim_header | elements...]` record per vector. `Npy` and
/// `Parquet` are third-party container formats. `Slab` is the page-oriented
/// binary format from the `slabtastic` crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VecFormat {
    /// NumPy `.npy` — float32 arrays.
    Npy,
    /// Apache Parquet — columnar float vectors.
    Parquet,
    /// xvec float32 (4 bytes/element).
    Fvec,
    /// xvec int32 (4 bytes/element).
    Ivec,
    /// xvec uint8 packed as 4-byte groups (4 bytes/element).
    Bvec,
    /// xvec float64 (8 bytes/element).
    Dvec,
    /// xvec float16 / half (2 bytes/element).
    Mvec,
    /// xvec int16 / short (2 bytes/element).
    Svec,
    /// Page-oriented slab binary format.
    Slab,
    /// HDF5 container — one file may hold multiple named datasets.
    ///
    /// Paths use `file.hdf5#dataset` notation to select an internal dataset.
    Hdf5,
}

impl VecFormat {
    /// Human-readable name matching the file extension (e.g. `"fvec"`, `"slab"`).
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
            Self::Hdf5 => "hdf5",
        }
    }

    /// Returns `true` for the xvec family (fvec, ivec, bvec, dvec, mvec, svec).
    pub fn is_xvec(self) -> bool {
        matches!(
            self,
            Self::Fvec | Self::Ivec | Self::Bvec | Self::Dvec | Self::Mvec | Self::Svec
        )
    }

    /// Bytes per element for xvec formats. Returns `0` for non-xvec formats.
    pub fn element_size(self) -> usize {
        match self {
            Self::Fvec | Self::Ivec | Self::Bvec => 4,
            Self::Dvec => 8,
            Self::Mvec | Self::Svec => 2,
            Self::Npy | Self::Parquet | Self::Slab | Self::Hdf5 => 0,
        }
    }

    /// Return the canonical (preferred) extension for a recognized extension
    /// string. For xvec formats the shorter singular form is canonical
    /// (e.g. `"fvecs"` → `"fvec"`). Returns `None` for unrecognized extensions.
    pub fn canonical_extension(ext: &str) -> Option<&'static str> {
        Self::from_extension(ext).map(|f| f.name())
    }

    /// Detect format from a bare file extension (e.g. `"fvec"`, `"parquet"`).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "npy" => Some(Self::Npy),
            "parquet" => Some(Self::Parquet),
            "slab" => Some(Self::Slab),
            "fvec" | "fvecs" => Some(Self::Fvec),
            "ivec" | "ivecs" => Some(Self::Ivec),
            "bvec" | "bvecs" => Some(Self::Bvec),
            "dvec" | "dvecs" => Some(Self::Dvec),
            "mvec" | "mvecs" => Some(Self::Mvec),
            "svec" | "svecs" => Some(Self::Svec),
            "hdf5" | "h5" => Some(Self::Hdf5),
            _ => None,
        }
    }

    /// Detect format from a file path by inspecting its extension.
    ///
    /// Handles HDF5 `#` notation: `file.hdf5#dataset` → `Hdf5`.
    pub fn detect_from_path(path: &Path) -> Option<Self> {
        let s = path.to_string_lossy();
        // Check for HDF5 # notation before standard extension detection
        if let Some(hash_pos) = s.rfind('#') {
            let file_part = &s[..hash_pos];
            let ext = file_part.rsplit('.').next()?;
            if ext == "hdf5" || ext == "h5" {
                return Some(Self::Hdf5);
            }
        }
        let ext = path.extension()?.to_str()?;
        Self::from_extension(ext)
    }

    /// Detect format by scanning a directory's files. Returns the first
    /// recognized format, preferring npy > parquet > xvec > slab.
    pub fn detect_from_directory(dir: &Path) -> Option<Self> {
        let entries: Vec<_> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .collect();

        if entries.is_empty() {
            return None;
        }

        let mut has_npy = false;
        let mut has_parquet = false;
        let mut has_xvec = None;
        let mut has_slab = false;

        for entry in &entries {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(ext) = name.rsplit('.').next() {
                match ext {
                    "npy" => has_npy = true,
                    "parquet" => has_parquet = true,
                    "slab" => has_slab = true,
                    _ => {
                        if has_xvec.is_none() {
                            has_xvec = Self::from_extension(ext).filter(|f| f.is_xvec());
                        }
                    }
                }
            }
        }

        if has_npy {
            Some(Self::Npy)
        } else if has_parquet {
            Some(Self::Parquet)
        } else if let Some(xvec) = has_xvec {
            Some(xvec)
        } else if has_slab {
            Some(Self::Slab)
        } else {
            None
        }
    }

    /// Auto-detect format from either a file path or a directory.
    pub fn detect(path: &Path) -> Option<Self> {
        if path.is_dir() {
            Self::detect_from_directory(path)
        } else {
            Self::detect_from_path(path)
        }
    }

    /// Compute the expected file size for a given record count and dimension.
    ///
    /// For xvec formats: `records × (4 + dimension × element_size)` where
    /// the 4-byte prefix is the LE i32 dimension header per record.
    ///
    /// Returns `None` for formats where exact size cannot be predicted
    /// (slab, npy, parquet).
    pub fn expected_file_size(self, record_count: u64, dimension: u32) -> Option<u64> {
        if self.is_xvec() {
            let stride = 4u64 + (dimension as u64) * (self.element_size() as u64);
            Some(record_count * stride)
        } else {
            None
        }
    }

    /// Human-readable data type name and bytes per element.
    ///
    /// Returns `(type_name, bytes_per_element)` for display purposes.
    pub fn data_type_name(self) -> &'static str {
        match self {
            Self::Fvec => "float",
            Self::Ivec => "int",
            Self::Bvec => "byte",
            Self::Dvec => "double",
            Self::Mvec => "half",
            Self::Svec => "short",
            Self::Slab => "bytes",
            Self::Npy => "float",
            Self::Parquet => "float",
            Self::Hdf5 => "float",
        }
    }

    /// Whether this format can be used as an output/sink format
    pub fn is_writable(self) -> bool {
        matches!(
            self,
            Self::Slab
                | Self::Fvec
                | Self::Ivec
                | Self::Bvec
                | Self::Dvec
                | Self::Mvec
                | Self::Svec
        )
    }
}

impl fmt::Display for VecFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
