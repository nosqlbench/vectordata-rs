// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Vector data readers for various formats.

pub mod npy;
pub mod parquet;
pub mod parquet_mnode;
pub mod slab;
pub mod xvec;

use std::path::Path;

use super::VecFormat;
use crate::import::Facet;

/// Trait for reading vector records from a source format.
///
/// Records are raw element bytes (no dimension prefix). The dimension is
/// metadata on the source itself.
pub trait VecSource {
    /// The vector dimension (number of elements per record)
    fn dimension(&self) -> u32;

    /// Bytes per element in the source data.
    ///
    /// Used to select the correct xvec output variant during import:
    /// 1 → bvec, 2 → hvec, 4 → fvec, 8 → dvec.
    fn element_size(&self) -> usize;

    /// Total record count, if known ahead of time
    fn record_count(&self) -> Option<u64>;

    /// Read the next record as raw bytes (element data only, no dim prefix).
    /// Returns `None` when exhausted.
    fn next_record(&mut self) -> Option<Vec<u8>>;
}

/// Lightweight metadata from probing a source without opening a full reader.
///
/// This avoids spawning background threads or loading data — only reads
/// headers and file metadata.
pub struct SourceMeta {
    pub dimension: u32,
    pub element_size: usize,
    pub record_count: Option<u64>,
}

/// Probe a source to extract metadata without starting a full reader.
///
/// For npy directories, this only reads file headers (a few hundred bytes
/// each). For other formats, it opens and immediately queries the reader
/// since those are already lightweight.
pub fn probe_source(path: &Path, format: VecFormat) -> Result<SourceMeta, String> {
    match format {
        VecFormat::Npy => npy::NpyDirReader::probe(path),
        _ => {
            // Other formats are lightweight to open — just open and extract
            let source = open_source(path, format, 0)?;
            Ok(SourceMeta {
                dimension: source.dimension(),
                element_size: source.element_size(),
                record_count: source.record_count(),
            })
        }
    }
}

/// Open a source reader for the given path and format.
///
/// `threads` controls how many loader threads to use for parallel readers
/// (npy, parquet). Pass `0` to auto-detect from available CPU parallelism.
/// Readers that don't support multi-threading ignore this parameter.
pub fn open_source(path: &Path, format: VecFormat, threads: usize) -> Result<Box<dyn VecSource>, String> {
    match format {
        VecFormat::Fvec
        | VecFormat::Ivec
        | VecFormat::Bvec
        | VecFormat::Dvec
        | VecFormat::Hvec
        | VecFormat::Svec => xvec::open_xvec(path, format),
        VecFormat::Npy => npy::NpyDirReader::open(path, threads),
        VecFormat::Parquet => parquet::ParquetDirReader::open(path, threads),
        VecFormat::Slab => slab::SlabReader::open(path),
    }
}

/// Probe a source with facet context, dispatching metadata parquet to
/// [`ParquetMnodeReader`](parquet_mnode::ParquetMnodeReader).
///
/// For `MetadataContent` facets with `Parquet` format, uses the MNode reader
/// which handles scalar columns. Everything else delegates to [`probe_source`].
pub fn probe_source_for_facet(
    path: &Path,
    format: VecFormat,
    facet: Facet,
) -> Result<SourceMeta, String> {
    if facet.is_mnode() && format == VecFormat::Parquet {
        parquet_mnode::ParquetMnodeReader::probe(path)
    } else {
        probe_source(path, format)
    }
}

/// Open a source reader with facet context, dispatching metadata parquet to
/// [`ParquetMnodeReader`](parquet_mnode::ParquetMnodeReader).
///
/// For `MetadataContent` facets with `Parquet` format, opens the MNode reader
/// which converts scalar parquet rows to MNode wire bytes. Everything else
/// delegates to [`open_source`].
///
/// `threads` controls how many loader threads to use for parallel readers.
/// Pass `0` to auto-detect from available CPU parallelism.
pub fn open_source_for_facet(
    path: &Path,
    format: VecFormat,
    facet: Facet,
    threads: usize,
) -> Result<Box<dyn VecSource>, String> {
    if facet.is_mnode() && format == VecFormat::Parquet {
        parquet_mnode::ParquetMnodeReader::open(path, threads)
    } else {
        open_source(path, format, threads)
    }
}
