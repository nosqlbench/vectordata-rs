// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Schema authority and streaming writer for the per-passage metadata table
//! (`metadata.parquet`) — the M-facet raw input for predicated (PVS)
//! datasets.
//!
//! One row per passage, **in passage row order**: row i of the metadata
//! table describes row i of `passages.parquet` and therefore row i of the
//! embedded vectors — the same ordinal-identity contract `verify alignment`
//! gates. Columns are scalars only, the shape the parquet→MNode reader
//! consumes when a dataset build imports metadata content.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanBuilder, Int32Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;

use super::passage_table::StagedParquetWriter;

/// Rows buffered before a record batch is flushed.
const BATCH_ROWS: usize = 65_536;

/// One per-passage metadata row. Parent-level fields (everything but
/// `section`) are broadcast from the passage's parent document; unknown
/// values use the documented defaults rather than nulls so every column
/// stays non-nullable and MNode-friendly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataRow {
    /// Parent document id (S2 corpusid).
    pub corpusid: i64,
    /// Passage section label (from `passages.parquet`).
    pub section: String,
    /// Publication year (0 = unknown).
    pub year: i32,
    /// Citation count (0 = unknown/none).
    pub citationcount: i64,
    /// Open-access flag (false = unknown/closed).
    pub isopenaccess: bool,
    /// Primary s2fieldsofstudy category ("" = unknown).
    pub field: String,
    /// Publication venue name ("" = unknown).
    pub venue: String,
}

/// The authoritative `metadata.parquet` schema.
pub fn metadata_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("corpusid", DataType::Int64, false),
        Field::new("section", DataType::Utf8, false),
        Field::new("year", DataType::Int32, false),
        Field::new("citationcount", DataType::Int64, false),
        Field::new("isopenaccess", DataType::Boolean, false),
        Field::new("field", DataType::Utf8, false),
        Field::new("venue", DataType::Utf8, false),
    ]))
}

/// Streaming writer for `metadata.parquet`.
pub struct MetadataTableWriter {
    staged: StagedParquetWriter,
    corpusid: Int64Builder,
    section: StringBuilder,
    year: Int32Builder,
    citationcount: Int64Builder,
    isopenaccess: BooleanBuilder,
    field: StringBuilder,
    venue: StringBuilder,
    buffered: usize,
}

impl MetadataTableWriter {
    /// Open a writer staging to `<path>.partial`.
    pub fn create(path: &Path) -> Result<Self, String> {
        Ok(Self {
            staged: StagedParquetWriter::create(path, metadata_schema())?,
            corpusid: Int64Builder::new(),
            section: StringBuilder::new(),
            year: Int32Builder::new(),
            citationcount: Int64Builder::new(),
            isopenaccess: BooleanBuilder::new(),
            field: StringBuilder::new(),
            venue: StringBuilder::new(),
            buffered: 0,
        })
    }

    pub fn push(&mut self, row: &MetadataRow) -> Result<(), String> {
        self.corpusid.append_value(row.corpusid);
        self.section.append_value(&row.section);
        self.year.append_value(row.year);
        self.citationcount.append_value(row.citationcount);
        self.isopenaccess.append_value(row.isopenaccess);
        self.field.append_value(&row.field);
        self.venue.append_value(&row.venue);
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
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.corpusid.finish()),
            Arc::new(self.section.finish()),
            Arc::new(self.year.finish()),
            Arc::new(self.citationcount.finish()),
            Arc::new(self.isopenaccess.finish()),
            Arc::new(self.field.finish()),
            Arc::new(self.venue.finish()),
        ];
        let batch = RecordBatch::try_new(metadata_schema(), cols)
            .map_err(|e| format!("metadata batch build failed: {}", e))?;
        self.staged.write_batch(batch)?;
        self.buffered = 0;
        Ok(())
    }

    /// Flush and atomically move the staged file into place; returns rows
    /// written.
    pub fn finish(mut self) -> Result<u64, String> {
        self.flush()?;
        self.staged.finish()
    }
}
