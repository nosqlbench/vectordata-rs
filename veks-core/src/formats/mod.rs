// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vector data format support.
//!
//! This module provides a unified interface for reading and writing vector data
//! in various formats (xvec, scalar, npy, parquet, slab). It is designed to be
//! used by multiple tools and follows a registry pattern similar to the Java
//! `VectorFileIO` approach.

// Re-export veks-io traits so downstream code can access them via veks-core.
pub use veks_io::{VecSource as IoVecSource, VecSink as IoVecSink, SourceMeta as IoSourceMeta};
pub use veks_io::MmapReader as IoMmapReader;

// Wire format codecs re-exported from vectordata.
#[allow(dead_code)]
pub use vectordata::formats::mnode;
#[allow(dead_code)]
pub use vectordata::formats::pnode;
#[allow(dead_code)]
pub use vectordata::formats::anode;
#[allow(dead_code)]
pub use vectordata::formats::anode_vernacular;

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
/// **Vector formats** (xvec family) store one record per vector as
/// `[dim:i32, data...]`. Extensions end in `vec` or `vecs`.
///
/// **Scalar formats** store one value per ordinal as a flat packed array
/// with no header. Extension is the bare type name (e.g., `.u8`, `.i32`).
///
/// **Container formats** (npy, parquet, slab, hdf5) use their own structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VecFormat {
    // ── Container formats ──────────────────────────────────────────
    /// NumPy `.npy` — float32 arrays.
    Npy,
    /// Apache Parquet — columnar float vectors.
    Parquet,
    /// Page-oriented slab binary format.
    Slab,
    /// HDF5 container — `file.hdf5#dataset` notation.
    Hdf5,

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
            Self::Hdf5 => "hdf5",
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

    /// Bytes per element. Returns `0` for container formats.
    pub fn element_size(self) -> usize {
        match self {
            Self::Bvec | Self::I8vec | Self::ScalarU8 | Self::ScalarI8 => 1,
            Self::Mvec | Self::Svec | Self::U16vec | Self::ScalarU16 | Self::ScalarI16 => 2,
            Self::Fvec | Self::Ivec | Self::U32vec | Self::ScalarU32 | Self::ScalarI32 => 4,
            Self::Dvec | Self::I64vec | Self::U64vec | Self::ScalarU64 | Self::ScalarI64 => 8,
            Self::Npy | Self::Parquet | Self::Slab | Self::Hdf5 => 0,
        }
    }

    /// Canonical extension for a recognized extension string.
    pub fn canonical_extension(ext: &str) -> Option<&'static str> {
        Self::from_extension(ext).map(|f| f.name())
    }

    /// Detect format from a bare file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            // Container
            "npy" => Some(Self::Npy),
            "parquet" => Some(Self::Parquet),
            "slab" => Some(Self::Slab),
            "hdf5" | "h5" => Some(Self::Hdf5),
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
            _ => None,
        }
    }

    /// Detect format from a file path. Handles HDF5 `#` notation.
    pub fn detect_from_path(path: &Path) -> Option<Self> {
        let s = path.to_string_lossy();
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

    /// Detect format by scanning a directory's files.
    pub fn detect_from_directory(dir: &Path) -> Option<Self> {
        let entries: Vec<_> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .collect();
        if entries.is_empty() { return None; }

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
        if has_npy { Some(Self::Npy) }
        else if has_parquet { Some(Self::Parquet) }
        else if let Some(xvec) = has_xvec { Some(xvec) }
        else if has_slab { Some(Self::Slab) }
        else { None }
    }

    /// Auto-detect format from either a file path or a directory.
    pub fn detect(path: &Path) -> Option<Self> {
        if path.is_dir() { Self::detect_from_directory(path) }
        else { Self::detect_from_path(path) }
    }

    /// Compute expected file size for a given record count and dimension.
    ///
    /// For xvec: `records × (4 + dimension × element_size)`.
    /// For scalar: `records × element_size`.
    pub fn expected_file_size(self, record_count: u64, dimension: u32) -> Option<u64> {
        if self.is_xvec() {
            let stride = 4u64 + (dimension as u64) * (self.element_size() as u64);
            Some(record_count * stride)
        } else if self.is_scalar() {
            Some(record_count * self.element_size() as u64)
        } else {
            None
        }
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
            Self::Slab => "bytes",
            Self::Npy | Self::Parquet | Self::Hdf5 => "float",
        }
    }

    /// Whether this format can be used as an output/sink format.
    pub fn is_writable(self) -> bool {
        self.is_xvec() || self.is_scalar() || self == Self::Slab
    }
}

impl fmt::Display for VecFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
