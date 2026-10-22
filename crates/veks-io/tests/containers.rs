// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The container readers report the element type they read
//! (`docs/design/srd-crate-boundaries.md` FL-5): a parquet vector column
//! of any numeric type, in a `List` or a `FixedSizeList`, and an npy
//! file of f16, each at its own type and width — not 4 bytes whatever
//! the data.

#![cfg(all(feature = "npy", feature = "parquet"))]

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, ListArray, UInt8Array};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use veks_io::{ElementType, VecFormat, VecSource};

fn write_parquet(path: &std::path::Path, column: ArrayRef) {
    let schema = Arc::new(Schema::new(vec![Field::new("emb", column.data_type().clone(), false)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![column]).unwrap();
    let mut w = ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn all(mut source: Box<dyn VecSource>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(r) = source.next_record().unwrap() {
        out.push(r);
    }
    out
}

/// **A parquet column of u8 vectors reads as u8**, one byte an
/// element, its type stated.
#[test]
fn a_parquet_u8_column_reads_at_its_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("codes.parquet");
    let values = UInt8Array::from(vec![1u8, 2, 3, 4, 5, 6]);
    let item = Arc::new(Field::new("item", DataType::UInt8, false));
    let list = ListArray::new(item, OffsetBuffer::from_lengths([3, 3]), Arc::new(values), None);
    write_parquet(&path, Arc::new(list));
    let source = veks_io::open_format(&path, VecFormat::Parquet).unwrap();
    assert_eq!((source.dimension(), source.element_size(), source.element_type()), (3, 1, Some(ElementType::U8)));
    assert_eq!(all(source), vec![vec![1, 2, 3], vec![4, 5, 6]]);
}

/// **A `FixedSizeList` column decodes**: it was found as the vector
/// column and then read as a `List`, which it is not, so every row was
/// missing.
#[test]
fn a_parquet_fixed_size_list_column_decodes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("emb.parquet");
    let values = Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]);
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let list = FixedSizeListArray::new(item, 2, Arc::new(values), None);
    write_parquet(&path, Arc::new(list));
    let source = veks_io::open_format(&path, VecFormat::Parquet).unwrap();
    assert_eq!(source.element_type(), Some(ElementType::F32));
    let rows = all(source);
    assert_eq!(rows.len(), 2);
    assert_eq!(f32::from_le_bytes(rows[1][4..8].try_into().unwrap()), 4.0);
}

/// **An npy file of f16 states f16**, two bytes an element.
#[test]
fn an_npy_f16_file_states_its_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("emb.npy");
    // The npy v1 layout: magic, version, header length, a header dict
    // padded so the data starts 64-byte aligned, then the data.
    let mut header = "{'descr': '<f2', 'fortran_order': False, 'shape': (2, 3), }".to_string();
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    for v in 0..6 {
        bytes.extend(half::f16::from_f32(v as f32).to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();
    let meta = veks_io::npy::NpyDirReader::probe(&path).unwrap();
    assert_eq!((meta.dimension, meta.element_size, meta.element_type), (3, 2, Some(ElementType::F16)));
    let source = veks_io::open_format(&path, VecFormat::Npy).unwrap();
    assert_eq!(source.element_type(), Some(ElementType::F16));
    assert_eq!(all(source).len(), 2);
}
