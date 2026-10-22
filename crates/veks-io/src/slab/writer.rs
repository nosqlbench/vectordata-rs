// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! [`VecSink`] that writes vector records into a `.slab` file.

use std::path::{Path, PathBuf};

use slabtastic::{SlabWriter as SlabtasticWriter, WriterConfig};
use veks_apicore::metadata_schema::{MetadataSchema, SCHEMA_NAMESPACE, VectorSchema};

use crate::element::ElementType;
use crate::traits::VecSink;

/// Writes a slab file using the `slabtastic` crate.
///
/// Delegates page management, alignment, and pages-page indexing
/// to the crate's `SlabWriter`. When constructed with a schema
/// sidecar or an element type, [`finish`](Self::finish) seals the
/// content (default) namespace and emits a single JSON record into
/// the `:schema` namespace before finalizing the file: the
/// [`MetadataSchema`] given, or a [`VectorSchema`] declaring the
/// element type, which the slab reader then reports.
pub struct SlabWriter {
    writer: SlabtasticWriter,
    path: PathBuf,
    sidecar: Option<Vec<u8>>,
}

impl SlabWriter {
    /// Open a slab file for writing.
    ///
    /// When `preferred_page_size` is `None`, the slabtastic library default
    /// is used (currently 4 MiB).
    ///
    /// `schema_sidecar` (metadata records) or `element_type` (vector
    /// records) is written as a single JSON record into the
    /// [`SCHEMA_NAMESPACE`] (`"schema"`) namespace at finish time; the
    /// two describe different records, so passing both is an error.
    /// Pass neither for a content-only slab, whose reader then reports
    /// its records as bytes of an undeclared type.
    pub fn open(
        path: &Path,
        _record_byte_len: usize,
        preferred_page_size: Option<u32>,
        _namespace_index: u8,
        schema_sidecar: Option<MetadataSchema>,
        element_type: Option<ElementType>,
    ) -> Result<Box<dyn VecSink>, String> {
        let sidecar = match (schema_sidecar, element_type) {
            (Some(_), Some(ty)) => {
                return Err(format!(
                    "{}: a metadata schema sidecar describes metadata records, not {ty} vectors",
                    path.display()
                ));
            }
            (Some(schema), None) => Some(schema.to_json_bytes()),
            (None, Some(ty)) => Some(VectorSchema::new(ty).to_json_bytes()),
            (None, None) => None,
        }
        .transpose()
        .map_err(|e| format!("{}: {e}", path.display()))?;

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
            sidecar,
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
        if let Some(bytes) = self.sidecar.take() {
            self.writer
                .start_namespace(SCHEMA_NAMESPACE)
                .map_err(|e| {
                    format!("Failed to seal content namespace in {}: {e}", self.path.display())
                })?;
            self.writer
                .add_record(&bytes)
                .map_err(|e| {
                    format!("Failed to write schema record in {}: {e}", self.path.display())
                })?;
        }
        self.writer
            .finish()
            .map_err(|e| format!("Failed to finalize slab {}: {}", self.path.display(), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slab::reader::SlabReader;

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
            let mut writer = SlabWriter::open(&path, 16, None, 1, None, Some(ElementType::F32)).unwrap();
            for (ordinal, data) in &records {
                writer.write_record(*ordinal, data);
            }
            writer.finish().unwrap();
        }

        // Read back
        let mut reader = SlabReader::open(&path).unwrap();
        assert_eq!(reader.dimension(), 4);
        assert_eq!(reader.element_type(), Some(ElementType::F32));
        assert_eq!(reader.record_count(), Some(10));

        for (i, (_ordinal, expected_data)) in records.iter().enumerate() {
            let actual = reader
                .next_record()
                .unwrap()
                .unwrap_or_else(|| panic!("Expected record {} but got None", i));
            assert_eq!(actual, *expected_data, "Record {} data mismatch", i);
        }
        assert!(reader.next_record().unwrap().is_none());
    }

    /// When a schema sidecar is passed in, `finish` lands it in the
    /// `:schema` namespace as a single JSON record while leaving the
    /// content namespace untouched.
    #[test]
    fn test_slab_writes_schema_sidecar() {
        use veks_apicore::metadata_schema::{
            MetadataSchema, SchemaField, SCHEMA_NAMESPACE,
        };

        let schema = MetadataSchema::new(
            "parquet:/test/source",
            vec![
                SchemaField {
                    name: "id".into(),
                    type_name: "int".into(),
                    nullable: false,
                },
                SchemaField {
                    name: "label".into(),
                    type_name: "text".into(),
                    nullable: true,
                },
            ],
        )
        .with_record_count(3);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut writer =
                SlabWriter::open(&path, 16, None, 1, Some(schema.clone()), None).unwrap();
            writer.write_record(0, b"content-0");
            writer.write_record(1, b"content-1");
            writer.write_record(2, b"content-2");
            writer.finish().unwrap();
        }

        // Content namespace still readable as before.
        let content_reader = slabtastic::SlabReader::open(&path).unwrap();
        assert_eq!(content_reader.get(0).unwrap(), b"content-0");
        assert_eq!(content_reader.get(2).unwrap(), b"content-2");

        // Schema namespace carries the descriptor as a single JSON record.
        let schema_reader =
            slabtastic::SlabReader::open_namespace(&path, Some(SCHEMA_NAMESPACE))
                .unwrap();
        let record = schema_reader.get(0).unwrap();
        let parsed = MetadataSchema::from_json_bytes(&record).unwrap();
        assert_eq!(parsed.fields, schema.fields);
        assert_eq!(parsed.record_count, Some(3));
        assert!(schema_reader.get(1).is_err(), "schema namespace has exactly one record");
    }

    /// A metadata sidecar and an element type describe different
    /// records; the writer refuses the pair rather than drop one.
    #[test]
    fn test_slab_refuses_a_metadata_sidecar_with_an_element_type() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let schema = MetadataSchema::new("parquet:/test/source", vec![]);
        let err = SlabWriter::open(tmp.path(), 4, None, 1, Some(schema), Some(ElementType::F32))
            .err()
            .unwrap();
        assert!(err.contains("not f32 vectors"), "{err}");
    }

    #[test]
    fn test_slab_many_records() {
        let records: Vec<Vec<u8>> = (0..100)
            .map(|i| vec![i as u8; 16])
            .collect();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let mut writer = SlabWriter::open(&path, 16, Some(512), 1, None, Some(ElementType::U8)).unwrap();
            for (i, data) in records.iter().enumerate() {
                writer.write_record(i as i64, data);
            }
            writer.finish().unwrap();
        }

        let mut reader = SlabReader::open(&path).unwrap();
        assert_eq!(reader.record_count(), Some(100));
        assert_eq!((reader.dimension(), reader.element_type()), (16, Some(ElementType::U8)));

        for i in 0..100 {
            let data = reader.next_record().unwrap().unwrap();
            assert_eq!(data, vec![i as u8; 16]);
        }
        assert!(reader.next_record().unwrap().is_none());
    }
}
