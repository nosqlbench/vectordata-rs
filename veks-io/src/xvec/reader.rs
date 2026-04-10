// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Streaming xvec reader.
//!
//! Reads xvec files record-by-record with buffered I/O and periodic
//! page cache release for large files.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::format::VecFormat;
use crate::traits::{SourceMeta, VecSource};

const READ_BUF_SIZE: usize = 4 << 20; // 4 MiB

/// Open an xvec file as a streaming [`VecSource`].
pub fn open(path: &Path, format: VecFormat) -> Result<Box<dyn VecSource>, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let file_size = file.metadata()
        .map_err(|e| format!("metadata {}: {}", path.display(), e))?
        .len();
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} is not an xvec format", format));
    }

    let mut reader = BufReader::with_capacity(READ_BUF_SIZE, file);

    // Read dimension from first record header
    let dim = reader.read_i32::<LittleEndian>()
        .map_err(|e| format!("read dim: {}", e))? as u32;
    if dim == 0 {
        return Err("zero dimension".into());
    }

    let record_bytes = 4 + (dim as usize) * elem_size;
    let total_records = file_size / record_bytes as u64;

    // Seek back to start
    use std::io::Seek;
    reader.seek(std::io::SeekFrom::Start(0))
        .map_err(|e| format!("seek: {}", e))?;

    Ok(Box::new(XvecReader {
        reader,
        dim,
        elem_size,
        total_records,
        records_read: 0,
    }))
}

/// Probe an xvec file for metadata without reading all data.
pub fn probe(path: &Path, format: VecFormat) -> Result<SourceMeta, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let file_size = file.metadata()
        .map_err(|e| format!("metadata: {}", e))?
        .len();
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} is not an xvec format", format));
    }

    let mut reader = BufReader::new(file);
    let dim = reader.read_i32::<LittleEndian>()
        .map_err(|e| format!("read dim: {}", e))? as u32;
    let record_bytes = 4 + (dim as usize) * elem_size;
    let count = file_size / record_bytes as u64;

    Ok(SourceMeta {
        dimension: dim,
        element_size: elem_size,
        record_count: Some(count),
    })
}

struct XvecReader {
    reader: BufReader<File>,
    dim: u32,
    elem_size: usize,
    total_records: u64,
    records_read: u64,
}

impl VecSource for XvecReader {
    fn dimension(&self) -> u32 { self.dim }
    fn element_size(&self) -> usize { self.elem_size }
    fn record_count(&self) -> Option<u64> { Some(self.total_records) }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        if self.records_read >= self.total_records {
            return None;
        }

        // Read and validate dimension header
        let dim = match self.reader.read_i32::<LittleEndian>() {
            Ok(d) => d as u32,
            Err(_) => return None,
        };
        if dim != self.dim {
            return None; // dimension mismatch
        }

        // Read element data
        let data_bytes = self.dim as usize * self.elem_size;
        let mut buf = vec![0u8; data_bytes];
        if self.reader.read_exact(&mut buf).is_err() {
            return None;
        }

        self.records_read += 1;
        Some(buf)
    }
}
