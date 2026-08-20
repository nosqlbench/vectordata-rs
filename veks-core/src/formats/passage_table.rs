// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Parquet table codec for passage corpora.
//!
//! This module is the single authority for the on-disk schema of the two
//! passage-pipeline artifacts:
//!
//! - **`passages.parquet`** — one row per passage, in parent-block order
//!   (all passages of a document contiguous). The global passage ordinal is
//!   the row index; passage identity is the (corpusid, section, ordinal)
//!   triple. `char_start`/`char_end` are character (not byte) offsets into
//!   the source document text — the publishable passage *coordinates*.
//! - **`parents.parquet`** — one row per parent document: its passage count
//!   and the global row index of its first passage (`row_start`).
//!
//! Writers stage output in a sibling `.partial` file and rename on
//! [`finish`](PassageTableWriter::finish), so a killed run never leaves a
//! torn artifact at the final path. Output is deterministic: identical rows
//! produce byte-identical files.
//!
//! This is a *table* codec, deliberately separate from the vector-sink API
//! ([`writer::open_sink`](super::writer::open_sink) /
//! [`VecFormat::is_writable`](super::VecFormat::is_writable)), which models
//! fixed-dimension vector records rather than heterogeneous rows.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, Int32Array, Int32Builder, Int64Array, Int64Builder, RecordBatch, StringArray,
    StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Rows buffered before a `RecordBatch` is flushed to the writer.
const BATCH_ROWS: usize = 4096;

/// One passage row of `passages.parquet`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassageRow {
    /// Parent document id (S2 corpusid).
    pub corpusid: i64,
    /// Section label (whitespace-collapsed header text; empty when the
    /// passage precedes any section header).
    pub section: String,
    /// Ordinal within (corpusid, section) — the identity triple's third leg.
    pub ordinal: i32,
    /// Character offset of the passage's first paragraph in the source text.
    pub char_start: i64,
    /// Character offset one past the passage's last paragraph in the source text.
    pub char_end: i64,
    /// Passage prose (upstream artifact only; never a published facet).
    pub text: String,
}

/// One parent row of `parents.parquet`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentRow {
    /// Parent document id (S2 corpusid).
    pub corpusid: i64,
    /// Number of passages derived from this document (0 when the document
    /// was selected but yielded no chunkable paragraphs).
    pub passage_count: i32,
    /// Global row index (in `passages.parquet`) of this parent's first
    /// passage. For a zero-passage parent this is the next parent's start.
    pub row_start: i64,
}

/// The authoritative `passages.parquet` schema.
pub fn passages_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("corpusid", DataType::Int64, false),
        Field::new("section", DataType::Utf8, false),
        Field::new("ordinal", DataType::Int32, false),
        Field::new("char_start", DataType::Int64, false),
        Field::new("char_end", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]))
}

/// The authoritative `parents.parquet` schema.
pub fn parents_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("corpusid", DataType::Int64, false),
        Field::new("passage_count", DataType::Int32, false),
        Field::new("row_start", DataType::Int64, false),
    ]))
}

/// Shared staged-file plumbing: an [`ArrowWriter`] over `<path>.partial`,
/// renamed to `path` on finish.
pub(crate) struct StagedParquetWriter {
    writer: ArrowWriter<File>,
    final_path: PathBuf,
    partial_path: PathBuf,
    rows_written: u64,
}

impl StagedParquetWriter {
    pub(crate) fn create(path: &Path, schema: SchemaRef) -> Result<Self, String> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {}", parent.display(), e))?;
        }
        let partial_path = path.with_extension("parquet.partial");
        let file = File::create(&partial_path)
            .map_err(|e| format!("failed to create {}: {}", partial_path.display(), e))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| format!("failed to open parquet writer: {}", e))?;
        Ok(Self {
            writer,
            final_path: path.to_path_buf(),
            partial_path,
            rows_written: 0,
        })
    }

    pub(crate) fn write_batch(&mut self, batch: RecordBatch) -> Result<(), String> {
        self.rows_written += batch.num_rows() as u64;
        self.writer
            .write(&batch)
            .map_err(|e| format!("parquet write failed: {}", e))
    }

    pub(crate) fn finish(self) -> Result<u64, String> {
        self.writer
            .close()
            .map_err(|e| format!("parquet close failed: {}", e))?;
        std::fs::rename(&self.partial_path, &self.final_path).map_err(|e| {
            format!(
                "failed to rename {} to {}: {}",
                self.partial_path.display(),
                self.final_path.display(),
                e
            )
        })?;
        Ok(self.rows_written)
    }
}

/// Streaming writer for `passages.parquet`.
pub struct PassageTableWriter {
    staged: StagedParquetWriter,
    corpusid: Int64Builder,
    section: StringBuilder,
    ordinal: Int32Builder,
    char_start: Int64Builder,
    char_end: Int64Builder,
    text: StringBuilder,
    buffered: usize,
}

impl PassageTableWriter {
    /// Open a writer staging to `<path>.partial`.
    pub fn create(path: &Path) -> Result<Self, String> {
        Ok(Self {
            staged: StagedParquetWriter::create(path, passages_schema())?,
            corpusid: Int64Builder::new(),
            section: StringBuilder::new(),
            ordinal: Int32Builder::new(),
            char_start: Int64Builder::new(),
            char_end: Int64Builder::new(),
            text: StringBuilder::new(),
            buffered: 0,
        })
    }

    /// Append one passage row. Rows must arrive in the intended final row
    /// order — the global passage ordinal is the row index.
    pub fn push(&mut self, row: &PassageRow) -> Result<(), String> {
        self.corpusid.append_value(row.corpusid);
        self.section.append_value(&row.section);
        self.ordinal.append_value(row.ordinal);
        self.char_start.append_value(row.char_start);
        self.char_end.append_value(row.char_end);
        self.text.append_value(&row.text);
        self.buffered += 1;
        if self.buffered >= BATCH_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        if self.buffered == 0 {
            return Ok(());
        }
        let batch = RecordBatch::try_new(
            passages_schema(),
            vec![
                Arc::new(self.corpusid.finish()),
                Arc::new(self.section.finish()),
                Arc::new(self.ordinal.finish()),
                Arc::new(self.char_start.finish()),
                Arc::new(self.char_end.finish()),
                Arc::new(self.text.finish()),
            ],
        )
        .map_err(|e| format!("failed to build record batch: {}", e))?;
        self.buffered = 0;
        self.staged.write_batch(batch)
    }

    /// Flush, close, and rename into place. Returns the total row count.
    pub fn finish(mut self) -> Result<u64, String> {
        self.flush()?;
        self.staged.finish()
    }
}

/// Streaming writer for `parents.parquet`.
pub struct ParentTableWriter {
    staged: StagedParquetWriter,
    corpusid: Int64Builder,
    passage_count: Int32Builder,
    row_start: Int64Builder,
    buffered: usize,
}

impl ParentTableWriter {
    /// Open a writer staging to `<path>.partial`.
    pub fn create(path: &Path) -> Result<Self, String> {
        Ok(Self {
            staged: StagedParquetWriter::create(path, parents_schema())?,
            corpusid: Int64Builder::new(),
            passage_count: Int32Builder::new(),
            row_start: Int64Builder::new(),
            buffered: 0,
        })
    }

    /// Append one parent row, in parent-block order.
    pub fn push(&mut self, row: &ParentRow) -> Result<(), String> {
        self.corpusid.append_value(row.corpusid);
        self.passage_count.append_value(row.passage_count);
        self.row_start.append_value(row.row_start);
        self.buffered += 1;
        if self.buffered >= BATCH_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        if self.buffered == 0 {
            return Ok(());
        }
        let batch = RecordBatch::try_new(
            parents_schema(),
            vec![
                Arc::new(self.corpusid.finish()),
                Arc::new(self.passage_count.finish()),
                Arc::new(self.row_start.finish()),
            ],
        )
        .map_err(|e| format!("failed to build record batch: {}", e))?;
        self.buffered = 0;
        self.staged.write_batch(batch)
    }

    /// Flush, close, and rename into place. Returns the total row count.
    pub fn finish(mut self) -> Result<u64, String> {
        self.flush()?;
        self.staged.finish()
    }
}

/// Row count of any parquet file, read from footer metadata only.
pub fn parquet_row_count(path: &Path) -> Result<u64, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("failed to read parquet metadata from {}: {}", path.display(), e))?;
    let rows = builder.metadata().file_metadata().num_rows();
    if rows < 0 {
        return Err(format!("negative row count in {}", path.display()));
    }
    Ok(rows as u64)
}

/// Read a complete `passages.parquet` back into rows (schema-checked).
pub fn read_passages(path: &Path) -> Result<Vec<PassageRow>, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("failed to read parquet {}: {}", path.display(), e))?
        .build()
        .map_err(|e| format!("failed to build parquet reader: {}", e))?;

    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| format!("parquet read failed: {}", e))?;
        let corpusid = column_as::<Int64Array>(&batch, "corpusid")?;
        let section = column_as::<StringArray>(&batch, "section")?;
        let ordinal = column_as::<Int32Array>(&batch, "ordinal")?;
        let char_start = column_as::<Int64Array>(&batch, "char_start")?;
        let char_end = column_as::<Int64Array>(&batch, "char_end")?;
        let text = column_as::<StringArray>(&batch, "text")?;
        for i in 0..batch.num_rows() {
            rows.push(PassageRow {
                corpusid: corpusid.value(i),
                section: section.value(i).to_string(),
                ordinal: ordinal.value(i),
                char_start: char_start.value(i),
                char_end: char_end.value(i),
                text: text.value(i).to_string(),
            });
        }
    }
    Ok(rows)
}

/// Read one Utf8 column of any parquet table in row order (projected —
/// other columns are not decoded). Used by embedding stages that must
/// preserve the ordinal contract: element i is row i's value.
pub fn read_text_column(path: &Path, column: &str) -> Result<Vec<String>, String> {
    use parquet::arrow::ProjectionMask;
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("failed to read parquet {}: {}", path.display(), e))?;
    let schema = builder.parquet_schema();
    let indices: Vec<usize> = (0..schema.num_columns())
        .filter(|&i| schema.column(i).name() == column)
        .collect();
    if indices.is_empty() {
        return Err(format!("no column '{}' in {}", column, path.display()));
    }
    let mask = ProjectionMask::leaves(schema, indices);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| format!("failed to build parquet reader: {}", e))?;

    let mut values = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| format!("parquet read failed: {}", e))?;
        let col = column_as::<StringArray>(&batch, column)?;
        for i in 0..batch.num_rows() {
            values.push(col.value(i).to_string());
        }
    }
    Ok(values)
}

/// Read a single Int64 column of a parquet file in row order (projected —
/// other columns are not decoded). Same ordinal contract as
/// [`read_text_column`]: element i is row i's value.
pub fn read_i64_column(path: &Path, column: &str) -> Result<Vec<i64>, String> {
    use parquet::arrow::ProjectionMask;
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("failed to read parquet {}: {}", path.display(), e))?;
    let schema = builder.parquet_schema();
    let indices: Vec<usize> = (0..schema.num_columns())
        .filter(|&i| schema.column(i).name() == column)
        .collect();
    if indices.is_empty() {
        return Err(format!("no column '{}' in {}", column, path.display()));
    }
    let mask = ProjectionMask::leaves(schema, indices);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| format!("failed to build parquet reader: {}", e))?;

    let mut values = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| format!("parquet read failed: {}", e))?;
        let col = column_as::<Int64Array>(&batch, column)?;
        for i in 0..batch.num_rows() {
            values.push(col.value(i));
        }
    }
    Ok(values)
}

/// Read a complete `parents.parquet` back into rows (schema-checked).
pub fn read_parents(path: &Path) -> Result<Vec<ParentRow>, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("failed to read parquet {}: {}", path.display(), e))?
        .build()
        .map_err(|e| format!("failed to build parquet reader: {}", e))?;

    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| format!("parquet read failed: {}", e))?;
        let corpusid = column_as::<Int64Array>(&batch, "corpusid")?;
        let passage_count = column_as::<Int32Array>(&batch, "passage_count")?;
        let row_start = column_as::<Int64Array>(&batch, "row_start")?;
        for i in 0..batch.num_rows() {
            rows.push(ParentRow {
                corpusid: corpusid.value(i),
                passage_count: passage_count.value(i),
                row_start: row_start.value(i),
            });
        }
    }
    Ok(rows)
}

/// Downcast a named column, with a schema-mismatch error naming the column.
fn column_as<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T, String> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("missing column '{}'", name))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| format!("column '{}' has unexpected type", name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_passages() -> Vec<PassageRow> {
        (0..10_000i64)
            .map(|i| PassageRow {
                corpusid: i / 100,
                section: if i % 2 == 0 { "Introduction".into() } else { "".into() },
                ordinal: (i % 100) as i32,
                char_start: i * 10,
                char_end: i * 10 + 9,
                text: format!("passage text {}", i),
            })
            .collect()
    }

    #[test]
    fn passages_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("passages.parquet");
        let rows = sample_passages();

        let mut w = PassageTableWriter::create(&path).unwrap();
        for row in &rows {
            w.push(row).unwrap();
        }
        assert_eq!(w.finish().unwrap(), rows.len() as u64);

        assert_eq!(parquet_row_count(&path).unwrap(), rows.len() as u64);
        let back = read_passages(&path).unwrap();
        assert_eq!(back, rows);
        assert!(!path.with_extension("parquet.partial").exists());
    }

    #[test]
    fn parents_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("parents.parquet");
        let rows: Vec<ParentRow> = (0..100)
            .map(|i| ParentRow { corpusid: i, passage_count: (i % 7) as i32, row_start: i * 5 })
            .collect();

        let mut w = ParentTableWriter::create(&path).unwrap();
        for row in &rows {
            w.push(row).unwrap();
        }
        assert_eq!(w.finish().unwrap(), rows.len() as u64);
        assert_eq!(read_parents(&path).unwrap(), rows);
    }

    #[test]
    fn writes_are_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.parquet");
        let b = tmp.path().join("b.parquet");
        for path in [&a, &b] {
            let mut w = PassageTableWriter::create(path).unwrap();
            for row in sample_passages() {
                w.push(&row).unwrap();
            }
            w.finish().unwrap();
        }
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    }

    #[test]
    fn unfinished_writer_leaves_no_final_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("passages.parquet");
        let mut w = PassageTableWriter::create(&path).unwrap();
        w.push(&sample_passages()[0]).unwrap();
        drop(w);
        assert!(!path.exists());
    }
}
