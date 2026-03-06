// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Vector data format support.
//!
//! This module provides a unified interface for reading and writing vector data
//! in various formats (xvec, npy, parquet, slab). It is designed to be used by
//! multiple tools — not just `convert` — and follows a registry pattern similar
//! to the Java `VectorFileIO` approach.

// mnode and pnode: wire format codecs for metadata and predicate records.
// The canonical encode/decode types (MNode, MValue, PNode, etc.) are used in
// tests to verify the wire format produced by the compiled writers at runtime.
#[allow(dead_code)]
pub mod mnode;
#[allow(dead_code)]
pub mod pnode;
#[allow(dead_code)]
pub mod anode;
#[allow(dead_code)]
pub mod anode_vernacular;
pub mod convert;
pub mod reader;
pub mod writer;

use std::fmt;
use std::path::Path;

use clap::ValueEnum;

/// A vector data format supported for reading
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VecFormat {
    Npy,
    Parquet,
    Fvec,
    Ivec,
    Bvec,
    Dvec,
    Hvec,
    Svec,
    Slab,
}

impl VecFormat {
    /// Human-readable name
    pub fn name(self) -> &'static str {
        match self {
            Self::Npy => "npy",
            Self::Parquet => "parquet",
            Self::Fvec => "fvec",
            Self::Ivec => "ivec",
            Self::Bvec => "bvec",
            Self::Dvec => "dvec",
            Self::Hvec => "hvec",
            Self::Svec => "svec",
            Self::Slab => "slab",
        }
    }

    /// Whether this is an xvec format
    pub fn is_xvec(self) -> bool {
        matches!(
            self,
            Self::Fvec | Self::Ivec | Self::Bvec | Self::Dvec | Self::Hvec | Self::Svec
        )
    }

    /// Element byte width for xvec formats (0 for non-xvec)
    pub fn element_size(self) -> usize {
        match self {
            Self::Fvec | Self::Ivec | Self::Bvec => 4,
            Self::Dvec => 8,
            Self::Hvec | Self::Svec => 2,
            Self::Npy | Self::Parquet | Self::Slab => 0,
        }
    }

    /// Detect format from a file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "npy" => Some(Self::Npy),
            "parquet" => Some(Self::Parquet),
            "slab" => Some(Self::Slab),
            "fvec" | "fvecs" => Some(Self::Fvec),
            "ivec" | "ivecs" => Some(Self::Ivec),
            "bvec" | "bvecs" => Some(Self::Bvec),
            "dvec" | "dvecs" => Some(Self::Dvec),
            "hvec" | "hvecs" => Some(Self::Hvec),
            "svec" | "svecs" => Some(Self::Svec),
            _ => None,
        }
    }

    /// Detect format from a file path (by extension)
    pub fn detect_from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?;
        Self::from_extension(ext)
    }

    /// Detect format from a directory's contents (first matching extension wins)
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

    /// Auto-detect format from either a file or directory
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
            Self::Hvec => "half",
            Self::Svec => "short",
            Self::Slab => "bytes",
            Self::Npy => "float",
            Self::Parquet => "float",
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
                | Self::Hvec
                | Self::Svec
        )
    }
}

impl fmt::Display for VecFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
