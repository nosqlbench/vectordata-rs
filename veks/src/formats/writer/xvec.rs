// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use super::VecSink;

/// Writes any xvec format (fvec, ivec, bvec, dvec, hvec, svec).
///
/// Wire format per record: `[dimension: i32 LE][elements: dim * element_size bytes]`
pub struct XvecWriter {
    writer: BufWriter<File>,
    dimension: u32,
}

impl XvecWriter {
    /// Open an xvec output file for writing
    pub fn open(path: &Path, dimension: u32) -> Result<Box<dyn VecSink>, String> {
        let file = File::create(path)
            .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
        Ok(Box::new(XvecWriter {
            writer: BufWriter::with_capacity(1 << 20, file),
            dimension,
        }))
    }
}

impl VecSink for XvecWriter {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        self.writer
            .write_i32::<LittleEndian>(self.dimension as i32)
            .expect("failed to write dimension");
        self.writer.write_all(data).expect("failed to write data");
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        self.writer.flush().map_err(|e| e.to_string())
    }
}
