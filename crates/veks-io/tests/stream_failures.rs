// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A streaming source reports a file it cannot read
//! (`docs/design/srd-crate-boundaries.md` PR-8). The npy and parquet
//! directory readers load their files on worker threads; one that
//! failed to load was logged at info level and skipped, so the stream
//! ended early with fewer records than `record_count` promised and no
//! caller saw an error.

#![cfg(all(feature = "npy", feature = "parquet"))]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use veks_io::{VecFormat, VecSource};

const DIM: usize = 2;
const ROWS: usize = 4;

fn scratch() -> tempfile::TempDir {
    tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap()
}

/// The values of file `file`: row `r` is `[file*100 + r, -(file*100 + r)]`.
fn values(file: usize) -> Vec<f32> {
    (0..ROWS).flat_map(|r| {
        let v = (file * 100 + r) as f32;
        [v, -v]
    }).collect()
}

/// An f32 npy file of `ROWS × DIM` whose header is whole; `keep` of
/// its data bytes are written, so a short `keep` makes a file whose
/// header scans and whose load fails.
fn write_npy(path: &Path, data: &[f32], keep: usize) {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({ROWS}, {DIM}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    let body: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes.extend_from_slice(&body[..keep.min(body.len())]);
    std::fs::write(path, bytes).unwrap();
}

fn write_parquet(path: &Path, data: &[f32]) {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let column: ArrayRef = Arc::new(FixedSizeListArray::new(
        item,
        DIM as i32,
        Arc::new(Float32Array::from(data.to_vec())),
        None,
    ));
    let schema = Arc::new(Schema::new(vec![Field::new("emb", column.data_type().clone(), false)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![column]).unwrap();
    let mut w = ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

/// Overwrite the first data page's header — just past the leading
/// `PAR1` — with garbage, leaving the footer whole: the file still
/// opens and states its row count, and its rows cannot be decoded.
fn corrupt_parquet_pages(path: &Path) {
    let mut bytes = std::fs::read(path).unwrap();
    for b in &mut bytes[4..36] {
        *b = 0xFF;
    }
    std::fs::write(path, bytes).unwrap();
}

/// Read every record, until the end or the first error.
fn drain(source: &mut dyn VecSource) -> (Vec<Vec<u8>>, Option<String>) {
    let mut records = Vec::new();
    loop {
        match source.next_record() {
            Ok(Some(r)) => records.push(r),
            Ok(None) => return (records, None),
            Err(e) => return (records, Some(e.to_string())),
        }
    }
}

fn row(file: usize, r: usize) -> Vec<u8> {
    let v = (file * 100 + r) as f32;
    [v, -v].iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// **An npy directory with a corrupt file in the middle** yields the
/// records before it, then an error naming the file.
#[test]
fn a_corrupt_npy_file_mid_directory_is_an_error() {
    let dir = scratch();
    write_npy(&dir.path().join("part_0.npy"), &values(0), usize::MAX);
    write_npy(&dir.path().join("part_1.npy"), &values(1), 5); // header says 4 rows; 5 data bytes
    write_npy(&dir.path().join("part_2.npy"), &values(2), usize::MAX);

    let mut source = veks_io::open_format(dir.path(), VecFormat::Npy).unwrap();
    assert_eq!(source.record_count(), Some(3 * ROWS as u64));
    let (records, error) = drain(source.as_mut());
    assert_eq!(records, (0..ROWS).map(|r| row(0, r)).collect::<Vec<_>>(), "the records before the corrupt file, in order");
    let error = error.unwrap_or_else(|| panic!(
        "the stream ended after {} of {:?} records with no error",
        records.len(),
        source.record_count()
    ));
    assert!(error.contains("part_1.npy"), "the error names the file: {error}");
    assert!(!error.contains("loaders stopped"), "the file's own load failure, not a closed channel: {error}");
    assert!(source.next_record().is_err(), "the stream does not go on past the error");
}

/// **A parquet directory with a corrupt file in the middle** yields the
/// records before it, then an error naming the file.
#[test]
fn a_corrupt_parquet_file_mid_directory_is_an_error() {
    let dir = scratch();
    for file in 0..3 {
        write_parquet(&dir.path().join(format!("part_{file}.parquet")), &values(file));
    }
    corrupt_parquet_pages(&dir.path().join("part_1.parquet"));

    let mut source = veks_io::open_format(dir.path(), VecFormat::Parquet).unwrap();
    assert_eq!(source.record_count(), Some(3 * ROWS as u64));
    let (records, error) = drain(source.as_mut());
    assert_eq!(records, (0..ROWS).map(|r| row(0, r)).collect::<Vec<_>>(), "the records before the corrupt file, in order");
    let error = error.unwrap_or_else(|| panic!(
        "the stream ended after {} of {:?} records with no error",
        records.len(),
        source.record_count()
    ));
    assert!(error.contains("part_1.parquet"), "the error names the file: {error}");
    assert!(!error.contains("loaders stopped"), "the file's own load failure, not a closed channel: {error}");
    assert!(source.next_record().is_err(), "the stream does not go on past the error");
}

/// **A whole parquet directory streams every file**, in name order.
#[test]
fn a_parquet_directory_streams_every_file() {
    let dir = scratch();
    for file in 0..3 {
        write_parquet(&dir.path().join(format!("part_{file}.parquet")), &values(file));
    }
    let mut source = veks_io::open_format(dir.path(), VecFormat::Parquet).unwrap();
    let (records, error) = drain(source.as_mut());
    assert_eq!(error, None);
    let expected: Vec<_> = (0..3).flat_map(|f| (0..ROWS).map(move |r| row(f, r))).collect();
    assert_eq!(records, expected);
}

/// **A whole npy directory streams every file**, in name order.
#[test]
fn an_npy_directory_streams_every_file() {
    let dir = scratch();
    for file in 0..3 {
        write_npy(&dir.path().join(format!("part_{file}.npy")), &values(file), usize::MAX);
    }
    let mut source = veks_io::open_format(dir.path(), VecFormat::Npy).unwrap();
    let (records, error) = drain(source.as_mut());
    assert_eq!(error, None);
    let expected: Vec<_> = (0..3).flat_map(|f| (0..ROWS).map(move |r| row(f, r))).collect();
    assert_eq!(records, expected);
}
