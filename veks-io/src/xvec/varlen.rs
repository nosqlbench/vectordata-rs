// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Variable-length xvec reader and writer.
//!
//! Handles xvec files where records may have different dimensions.
//! Each record is `[dim:i32_le, element_0, ..., element_dim-1]` and
//! the dimension can vary from record to record.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::format::VecFormat;
use crate::traits::{VarlenRecord, VarlenSink, VarlenSource};

const BUF_SIZE: usize = 4 << 20; // 4 MiB

/// Open an xvec file as a variable-length streaming reader.
///
/// Unlike the uniform reader, this does not assume all records share
/// the same dimension. Each call to `next_record()` reads the dimension
/// from the record header.
pub fn open_reader(path: &Path, format: VecFormat) -> Result<Box<dyn VarlenSource>, String> {
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} is not an xvec format", format));
    }

    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let file_size = file.metadata()
        .map_err(|e| format!("metadata {}: {}", path.display(), e))?
        .len();

    let reader = BufReader::with_capacity(BUF_SIZE, file);

    Ok(Box::new(XvecVarlenReader {
        reader,
        elem_size,
        file_size,
        bytes_read: 0,
    }))
}

/// Open an xvec file for variable-length writing.
///
/// Each record is written with its own dimension header, allowing
/// different dimensions per record.
pub fn open_writer(path: &Path) -> Result<Box<dyn VarlenSink>, String> {
    let file = File::create(path)
        .map_err(|e| format!("create {}: {}", path.display(), e))?;
    let writer = BufWriter::with_capacity(BUF_SIZE, file);

    Ok(Box::new(XvecVarlenWriter { writer }))
}

// ── Reader ─────────────────────────────────────────────────────────────

struct XvecVarlenReader {
    reader: BufReader<File>,
    elem_size: usize,
    file_size: u64,
    bytes_read: u64,
}

impl VarlenSource for XvecVarlenReader {
    fn element_size(&self) -> usize {
        self.elem_size
    }

    fn record_count(&self) -> Option<u64> {
        // Cannot know without scanning — dimension varies per record
        None
    }

    fn next_record(&mut self) -> Option<VarlenRecord> {
        if self.bytes_read >= self.file_size {
            return None;
        }

        // Read dimension header
        let dim = match self.reader.read_i32::<LittleEndian>() {
            Ok(d) if d > 0 => d as u32,
            _ => return None,
        };
        self.bytes_read += 4;

        // Read element data
        let data_bytes = dim as usize * self.elem_size;
        let mut data = vec![0u8; data_bytes];
        if self.reader.read_exact(&mut data).is_err() {
            return None;
        }
        self.bytes_read += data_bytes as u64;

        Some(VarlenRecord { dimension: dim, data })
    }
}

// ── Writer ─────────────────────────────────────────────────────────────

struct XvecVarlenWriter {
    writer: BufWriter<File>,
}

impl VarlenSink for XvecVarlenWriter {
    fn write_record(&mut self, dimension: u32, data: &[u8]) {
        let _ = self.writer.write_i32::<LittleEndian>(dimension as i32);
        let _ = self.writer.write_all(data);
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        self.writer.flush().map_err(|e| format!("flush: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tmp() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    #[test]
    fn varlen_roundtrip_uniform() {
        let tmp = make_tmp();
        let path = tmp.path().join("uniform.fvec");

        // Write 3 records all with dim=2
        let mut writer = open_writer(&path).unwrap();
        writer.write_record(2, &[1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        writer.write_record(2, &[3.0f32, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        writer.write_record(2, &[5.0f32, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        writer.finish().unwrap();

        let mut reader = open_reader(&path, VecFormat::Fvec).unwrap();
        let r0 = reader.next_record().unwrap();
        assert_eq!(r0.dimension, 2);
        assert_eq!(r0.data.len(), 8);

        let r1 = reader.next_record().unwrap();
        assert_eq!(r1.dimension, 2);

        let r2 = reader.next_record().unwrap();
        assert_eq!(r2.dimension, 2);

        assert!(reader.next_record().is_none());
    }

    #[test]
    fn varlen_roundtrip_nonuniform() {
        let tmp = make_tmp();
        let path = tmp.path().join("nonuniform.fvec");

        // Write records with different dimensions
        let mut writer = open_writer(&path).unwrap();
        // dim=1: single float
        writer.write_record(1, &1.0f32.to_le_bytes());
        // dim=3: three floats
        writer.write_record(3, &[2.0f32, 3.0, 4.0].iter()
            .flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        // dim=2: two floats
        writer.write_record(2, &[5.0f32, 6.0].iter()
            .flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        writer.finish().unwrap();

        let mut reader = open_reader(&path, VecFormat::Fvec).unwrap();

        let r0 = reader.next_record().unwrap();
        assert_eq!(r0.dimension, 1);
        assert_eq!(r0.data.len(), 4);
        assert_eq!(f32::from_le_bytes(r0.data[0..4].try_into().unwrap()), 1.0);

        let r1 = reader.next_record().unwrap();
        assert_eq!(r1.dimension, 3);
        assert_eq!(r1.data.len(), 12);
        assert_eq!(f32::from_le_bytes(r1.data[0..4].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_le_bytes(r1.data[8..12].try_into().unwrap()), 4.0);

        let r2 = reader.next_record().unwrap();
        assert_eq!(r2.dimension, 2);
        assert_eq!(r2.data.len(), 8);

        assert!(reader.next_record().is_none());
    }

    #[test]
    fn varlen_record_count_unknown() {
        let tmp = make_tmp();
        let path = tmp.path().join("varlen.fvec");

        let mut writer = open_writer(&path).unwrap();
        writer.write_record(1, &0.0f32.to_le_bytes());
        writer.write_record(5, &[0u8; 20]);
        writer.finish().unwrap();

        let reader = open_reader(&path, VecFormat::Fvec).unwrap();
        // Cannot know count without scanning
        assert_eq!(reader.record_count(), None);
    }

    #[test]
    fn varlen_empty_file() {
        let tmp = make_tmp();
        let path = tmp.path().join("empty.fvec");
        std::fs::File::create(&path).unwrap();

        let mut reader = open_reader(&path, VecFormat::Fvec).unwrap();
        assert!(reader.next_record().is_none());
    }

    #[test]
    fn varlen_truncated_stops_gracefully() {
        let tmp = make_tmp();
        let path = tmp.path().join("trunc.fvec");

        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).unwrap();
            // Complete record: dim=2, two f32s
            f.write_all(&2i32.to_le_bytes()).unwrap();
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
            f.write_all(&2.0f32.to_le_bytes()).unwrap();
            // Truncated: dim=3 but only one f32
            f.write_all(&3i32.to_le_bytes()).unwrap();
            f.write_all(&3.0f32.to_le_bytes()).unwrap();
        }

        let mut reader = open_reader(&path, VecFormat::Fvec).unwrap();
        let r0 = reader.next_record().unwrap();
        assert_eq!(r0.dimension, 2);
        // Truncated record should return None
        assert!(reader.next_record().is_none());
    }

    #[test]
    fn varlen_ivec_nonuniform() {
        let tmp = make_tmp();
        let path = tmp.path().join("nonuniform.ivec");

        let mut writer = open_writer(&path).unwrap();
        writer.write_record(2, &[10i32, 20].iter()
            .flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        writer.write_record(4, &[30i32, 40, 50, 60].iter()
            .flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        writer.finish().unwrap();

        let mut reader = open_reader(&path, VecFormat::Ivec).unwrap();
        let r0 = reader.next_record().unwrap();
        assert_eq!(r0.dimension, 2);
        assert_eq!(r0.data.len(), 8);

        let r1 = reader.next_record().unwrap();
        assert_eq!(r1.dimension, 4);
        assert_eq!(r1.data.len(), 16);

        assert!(reader.next_record().is_none());
    }
}
