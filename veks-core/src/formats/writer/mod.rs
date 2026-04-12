// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector data writers for various formats.

pub mod mmap_xvec;
pub mod slab;
pub mod xvec;

use std::path::Path;

use super::VecFormat;

/// Trait for writing vector records to an output format.
///
/// Records are raw element bytes (no dimension prefix). The dimension is
/// configured at writer construction time.
pub trait VecSink {
    /// Write a record with the given ordinal. Data is raw element bytes.
    fn write_record(&mut self, ordinal: i64, data: &[u8]);

    /// Finalize the output (e.g., write pages page for slab).
    ///
    /// Consumes self — an unfinished slab file is corrupt, so the caller
    /// must handle any I/O errors.
    fn finish(self: Box<Self>) -> Result<(), String>;
}

/// Adapter to wrap a `veks_io::VecSink` as a `veks_core::formats::writer::VecSink`.
struct IoSinkAdapter(Option<Box<dyn veks_io::VecSink>>);

impl VecSink for IoSinkAdapter {
    fn write_record(&mut self, ordinal: i64, data: &[u8]) {
        if let Some(ref mut inner) = self.0 {
            inner.write_record(ordinal, data);
        }
    }
    fn finish(mut self: Box<Self>) -> Result<(), String> {
        if let Some(inner) = self.0.take() {
            inner.finish()
        } else {
            Ok(())
        }
    }
}

/// Configuration for opening a sink writer
pub struct SinkConfig {
    pub dimension: u32,
    pub source_format: VecFormat,
    /// Preferred slab page size override. `None` uses the slabtastic default.
    pub slab_page_size: Option<u32>,
    pub slab_namespace: u8,
}

/// Open a sink writer for the given path, format, and configuration
pub fn open_sink(
    path: &Path,
    format: VecFormat,
    config: &SinkConfig,
) -> Result<Box<dyn VecSink>, String> {
    match format {
        VecFormat::Slab => {
            let element_size = if config.source_format.is_xvec() || config.source_format.is_scalar() {
                config.source_format.element_size()
            } else {
                4 // default to f32 for npy/parquet sources
            };
            let record_byte_len = config.dimension as usize * element_size;
            slab::SlabWriter::open(
                path,
                record_byte_len,
                config.slab_page_size,
                config.slab_namespace,
            )
        }
        _ if format.is_xvec() => xvec::XvecWriter::open(path, config.dimension),
        _ if format.is_scalar() => {
            let io_fmt = veks_io::VecFormat::from_extension(format.name())
                .unwrap_or(veks_io::VecFormat::ScalarU8);
            let io_writer = veks_io::scalar::writer::open(path, io_fmt)?;
            Ok(Box::new(IoSinkAdapter(Some(io_writer))))
        }
        _ => Err(format!("{} is not a supported output format", format)),
    }
}
