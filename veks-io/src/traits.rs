// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Core traits for reading and writing vector data.

/// Metadata about a vector data source.
#[derive(Debug, Clone)]
pub struct SourceMeta {
    /// Vector dimension (number of elements per record).
    pub dimension: u32,
    /// Bytes per element (4 for f32, 2 for f16, 8 for f64, etc.).
    pub element_size: usize,
    /// Total number of records, if known without a full scan.
    pub record_count: Option<u64>,
}

/// Streaming reader for vector data.
///
/// Records are returned as raw little-endian bytes (no dimension prefix).
/// The format and element type are determined at open time; the consumer
/// interprets bytes according to `element_size()`.
///
/// Implementations exist for all xvec formats (fvec, ivec, mvec, dvec,
/// bvec, svec) and optionally for npy, parquet, and slab formats.
pub trait VecSource: Send {
    /// Vector dimension (number of elements per record).
    fn dimension(&self) -> u32;

    /// Bytes per element in the source data.
    fn element_size(&self) -> usize;

    /// Total record count, if known without a full scan.
    fn record_count(&self) -> Option<u64>;

    /// Read the next record as raw bytes (element data only, no dim prefix).
    /// Returns `None` when exhausted.
    fn next_record(&mut self) -> Option<Vec<u8>>;
}

/// Streaming writer for vector data.
///
/// Records are raw element bytes (no dimension prefix). The dimension is
/// configured at writer construction time.
pub trait VecSink {
    /// Write a record. `data` is raw element bytes (length = dim × element_size).
    /// `ordinal` is the logical record index (used by slab for addressing).
    fn write_record(&mut self, ordinal: i64, data: &[u8]);

    /// Finalize the output (flush buffers, write footers, etc.).
    fn finish(self: Box<Self>) -> Result<(), String>;
}

// ── Variable-length (non-uniform dimension) types ──────────────────────

/// A single record from a variable-length vector source.
#[derive(Debug, Clone)]
pub struct VarlenRecord {
    /// Dimension of this specific record.
    pub dimension: u32,
    /// Raw element bytes (length = dimension × element_size).
    pub data: Vec<u8>,
}

/// Streaming reader for variable-length vector data.
///
/// Unlike [`VecSource`], which assumes all records share the same
/// dimension, this reader handles files where each record may have a
/// different dimension. The xvec on-disk format naturally supports this
/// since every record carries its own dimension header.
///
/// No random-access (mmap) support — non-uniform records have no fixed
/// stride, so indexing requires a separate offset table.
pub trait VarlenSource: Send {
    /// Bytes per element (same for all records — only dimension varies).
    fn element_size(&self) -> usize;

    /// Total record count, if known without a full scan.
    fn record_count(&self) -> Option<u64>;

    /// Read the next record, returning its dimension and data.
    /// Returns `None` when exhausted.
    fn next_record(&mut self) -> Option<VarlenRecord>;
}

/// Streaming writer for variable-length vector data.
///
/// Unlike [`VecSink`], which writes a fixed dimension header for every
/// record, this writer accepts a per-record dimension. Each record is
/// written with its own `[dim:i32, data...]` header.
pub trait VarlenSink {
    /// Write a record with the given dimension and raw element bytes.
    fn write_record(&mut self, dimension: u32, data: &[u8]);

    /// Finalize the output.
    fn finish(self: Box<Self>) -> Result<(), String>;
}
