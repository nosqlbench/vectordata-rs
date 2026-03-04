// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use slabtastic::{SlabWriter as SlabtasticWriter, WriterConfig};

use super::VecSink;

/// Writes a slab file using the `slabtastic` crate.
///
/// Delegates page management, alignment, and pages-page indexing
/// to the crate's `SlabWriter`.
pub struct SlabWriter {
    writer: SlabtasticWriter,
    path: PathBuf,
}

impl SlabWriter {
    /// Open a slab file for writing.
    ///
    /// When `preferred_page_size` is `None`, the slabtastic library default
    /// is used (currently 4 MiB).
    pub fn open(
        path: &Path,
        _record_byte_len: usize,
        preferred_page_size: Option<u32>,
        _namespace_index: u8,
    ) -> Result<Box<dyn VecSink>, String> {
        let config = match preferred_page_size {
            Some(size) => WriterConfig::new(512, size, u32::MAX, false)
                .map_err(|e| format!("Invalid writer config: {}", e))?,
            None => WriterConfig::default(),
        };

        let writer = SlabtasticWriter::new(path, config)
            .map_err(|e| format!("Failed to create slab {}: {}", path.display(), e))?;

        Ok(Box::new(SlabWriter {
            writer,
            path: path.to_path_buf(),
        }))
    }
}

impl VecSink for SlabWriter {
    fn write_record(&mut self, _ordinal: i64, data: &[u8]) {
        self.writer
            .add_record(data)
            .unwrap_or_else(|e| panic!("Failed to write record to slab: {}", e));
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        self.writer
            .finish()
            .map_err(|e| format!("Failed to finalize slab {}: {}", self.path.display(), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::reader::slab::SlabReader;

    #[test]
    fn test_slab_write_read_roundtrip() {
        let records: Vec<(i64, Vec<u8>)> = (0..10)
            .map(|i| {
                let val = (i as f32).to_le_bytes();
                let mut data = Vec::new();
                // 4-dim f32 vector
                for _ in 0..4 {
                    data.extend_from_slice(&val);
                }
                (i, data)
            })
            .collect();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut writer = SlabWriter::open(&path, 16, None, 1).unwrap();
            for (ordinal, data) in &records {
                writer.write_record(*ordinal, data);
            }
            writer.finish().unwrap();
        }

        // Read back
        let mut reader = SlabReader::open(&path).unwrap();
        assert_eq!(reader.dimension(), 4);
        assert_eq!(reader.record_count(), Some(10));

        for (i, (_ordinal, expected_data)) in records.iter().enumerate() {
            let actual = reader
                .next_record()
                .unwrap_or_else(|| panic!("Expected record {} but got None", i));
            assert_eq!(actual, *expected_data, "Record {} data mismatch", i);
        }
        assert!(reader.next_record().is_none());
    }

    #[test]
    fn test_slab_many_records() {
        let records: Vec<Vec<u8>> = (0..100)
            .map(|i| vec![i as u8; 16])
            .collect();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut writer = SlabWriter::open(&path, 16, Some(512), 1).unwrap();
            for (i, data) in records.iter().enumerate() {
                writer.write_record(i as i64, data);
            }
            writer.finish().unwrap();
        }

        let mut reader = SlabReader::open(&path).unwrap();
        assert_eq!(reader.record_count(), Some(100));

        for i in 0..100 {
            let data = reader.next_record().unwrap();
            assert_eq!(data, vec![i as u8; 16]);
        }
        assert!(reader.next_record().is_none());
    }
}
