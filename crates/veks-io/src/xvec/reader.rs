// Copyright (c) Jonathan Shook
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

/// The shape of an xvec file from its first header and its size: the
/// dimension, the record size, and the record count. The file must be
/// a whole number of records of a positive dimension; a short file is
/// refused here rather than read up to where it stops, because a tail
/// that is silently dropped is a record a consumer never sees.
fn shape(path: &Path, dim: i32, file_size: u64, elem_size: usize) -> Result<(u32, usize, u64), String> {
    if dim <= 0 {
        return Err(format!("{}: dimension {dim} is not positive", path.display()));
    }
    let record_bytes = (dim as usize)
        .checked_mul(elem_size)
        .and_then(|payload| payload.checked_add(4))
        .ok_or_else(|| format!("{}: a record of dimension {dim} overflows", path.display()))?;
    if !file_size.is_multiple_of(record_bytes as u64) {
        return Err(format!(
            "{}: file size {file_size} is not a whole number of {record_bytes}-byte records",
            path.display()
        ));
    }
    Ok((dim as u32, record_bytes, file_size / record_bytes as u64))
}

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
        .map_err(|e| format!("read dim: {}", e))?;
    let (dim, _, total_records) = shape(path, dim, file_size, elem_size)?;

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
        .map_err(|e| format!("read dim: {}", e))?;
    let (dim, _, count) = shape(path, dim, file_size, elem_size)?;

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
