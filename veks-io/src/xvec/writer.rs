// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Sequential xvec writer with buffered I/O.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::traits::VecSink;

const WRITE_BUF_SIZE: usize = 4 << 20; // 4 MiB

/// Open an xvec file for sequential writing.
///
/// The `dimension` is written as the header of every record.
/// Element size is inferred from the output format (determined by
/// the file extension at the call site).
pub fn open(path: &Path, dimension: u32) -> Result<Box<dyn VecSink>, String> {
    let file = File::create(path)
        .map_err(|e| format!("create {}: {}", path.display(), e))?;
    let writer = BufWriter::with_capacity(WRITE_BUF_SIZE, file);

    Ok(Box::new(XvecWriter {
        writer,
        dimension,
    }))
}

struct XvecWriter {
    writer: BufWriter<File>,
    dimension: u32,
}

impl VecSink for XvecWriter {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        let _ = self.writer.write_i32::<LittleEndian>(self.dimension as i32);
        let _ = self.writer.write_all(data);
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        self.writer.flush().map_err(|e| format!("flush: {}", e))
    }
}
