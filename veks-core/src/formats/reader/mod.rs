// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vector data readers for various formats.

pub mod hdf5;
pub mod npy;
pub mod parquet;
pub mod parquet_mnode;
pub mod slab;
pub mod xvec;

use std::path::Path;

use super::VecFormat;
use crate::formats::facet::Facet;

/// Trait for reading vector records from a source format.
///
/// Records are raw element bytes (no dimension prefix). The dimension is
/// metadata on the source itself.
pub trait VecSource: Send {
    /// The vector dimension (number of elements per record)
    fn dimension(&self) -> u32;

    /// Bytes per element in the source data.
    ///
    /// Used to select the correct xvec output variant during import:
    /// 1 → bvec, 2 → mvec, 4 → fvec, 8 → dvec.
    fn element_size(&self) -> usize;

    /// Total record count, if known ahead of time
    fn record_count(&self) -> Option<u64>;

    /// Read the next record as raw bytes (element data only, no dim prefix).
    /// Returns `None` when exhausted.
    fn next_record(&mut self) -> Option<Vec<u8>>;
}

/// Adapter to wrap a `veks_io::VecSource` as a `veks_core::formats::reader::VecSource`.
struct IoSourceAdapter(Box<dyn veks_io::VecSource>);

impl VecSource for IoSourceAdapter {
    fn dimension(&self) -> u32 { self.0.dimension() }
    fn element_size(&self) -> usize { self.0.element_size() }
    fn record_count(&self) -> Option<u64> { self.0.record_count() }
    fn next_record(&mut self) -> Option<Vec<u8>> { self.0.next_record() }
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
        VecFormat::Hdf5 => hdf5::probe(path),
        VecFormat::Slab => probe_slab(path),
        _ => {
            // Other formats are lightweight to open — just open and extract
            let source = open_source(path, format, 0, None)?;
            Ok(SourceMeta {
                dimension: source.dimension(),
                element_size: source.element_size(),
                record_count: source.record_count(),
            })
        }
    }
}

/// Probe a slab file for metadata without building the full page index.
///
/// Uses [`slabtastic::SlabReader::probe`] which reads only the pages-page
/// and at most two data page footers — orders of magnitude faster than
/// [`open`] for large files.
fn probe_slab(path: &Path) -> Result<SourceMeta, String> {
    let stats = slabtastic::SlabReader::probe(path)
        .map_err(|e| format!("Failed to probe slab {}: {}", path.display(), e))?;

    if stats.page_count == 0 {
        return Err("Empty slab file".to_string());
    }

    // Assume f32 elements (4 bytes each) as default
    let dimension = (stats.first_record_size / 4) as u32;

    Ok(SourceMeta {
        dimension,
        element_size: 4,
        record_count: Some(stats.total_records),
    })
}

/// Open a source reader for the given path and format.
///
/// `threads` controls how many loader threads to use for parallel readers
/// (npy, parquet). Pass `0` to auto-detect from available CPU parallelism.
/// Readers that don't support multi-threading ignore this parameter.
pub fn open_source(path: &Path, format: VecFormat, threads: usize, max_count: Option<u64>) -> Result<Box<dyn VecSource>, String> {
    if format.is_xvec() {
        return xvec::open_xvec(path, format);
    }
    if format.is_scalar() {
        let io_fmt = veks_io::VecFormat::from_extension(format.name())
            .unwrap_or(veks_io::VecFormat::ScalarU8);
        let io_reader = veks_io::scalar::reader::open(path, io_fmt)?;
        return Ok(Box::new(IoSourceAdapter(io_reader)));
    }
    match format {
        VecFormat::Npy => npy::NpyDirReader::open(path, threads, max_count),
        VecFormat::Parquet => parquet::ParquetDirReader::open(path, threads),
        VecFormat::Slab => slab::SlabReader::open(path),
        VecFormat::Hdf5 => hdf5::open(path),
        _ => Err(format!("{} format not supported for reading", format)),
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
    max_count: Option<u64>,
) -> Result<Box<dyn VecSource>, String> {
    if facet.is_mnode() && format == VecFormat::Parquet {
        parquet_mnode::ParquetMnodeReader::open(path, threads)
    } else {
        open_source(path, format, threads, max_count)
    }
}
