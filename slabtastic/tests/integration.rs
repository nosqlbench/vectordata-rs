// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the slabtastic library.
//!
//! These tests exercise the public API through realistic end-to-end
//! workflows: write → read, iteration, append-mode, alignment, and
//! error paths. Each test creates a temporary slabtastic file, writes
//! records through `SlabWriter`, then verifies correctness via
//! `SlabReader`.

use slabtastic::{SlabError, SlabReader, SlabWriter, WriterConfig, SLAB_EXTENSION};
use tempfile::NamedTempFile;

/// Write 50 records and verify each can be retrieved by ordinal.
///
/// Exercises the basic write → random-read round-trip with multiple
/// records in a single page (default 64 KiB preferred page size is
/// large enough to hold all 50 small records).
#[test]
fn test_write_then_read() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();

    let records: Vec<Vec<u8>> = (0..50)
        .map(|i| format!("record-{i:04}").into_bytes())
        .collect();

    for rec in &records {
        writer.add_record(rec).unwrap();
    }
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    for (i, expected) in records.iter().enumerate() {
        let actual = reader.get(i as i64).unwrap();
        assert_eq!(&actual, expected, "mismatch at ordinal {i}");
    }
}

/// Write 3 records and iterate them all via `iter()`.
///
/// Verifies that sequential iteration returns `(ordinal, data)` pairs
/// in ordinal order and that the ordinal assignments are correct
/// (0-based, monotonically increasing).
#[test]
fn test_iterate_all_records() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"first").unwrap();
    writer.add_record(b"second").unwrap();
    writer.add_record(b"third").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    let all = reader.iter().unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0], (0, b"first".to_vec()));
    assert_eq!(all[1], (1, b"second".to_vec()));
    assert_eq!(all[2], (2, b"third".to_vec()));
}

/// Write 2 records, close, append 2 more, then read all 4.
///
/// Exercises `SlabWriter::append` which reads the existing pages page,
/// positions after the last data page, and writes new data pages
/// followed by a new pages page that references both old and new pages.
/// Verifies ordinal continuity across append boundaries and that the
/// page count reflects both the original and appended page.
#[test]
fn test_append_mode() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Write initial records
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config.clone()).unwrap();
    writer.add_record(b"alpha").unwrap();
    writer.add_record(b"beta").unwrap();
    writer.finish().unwrap();

    // Append more records
    let mut writer = SlabWriter::append(&path, config).unwrap();
    writer.add_record(b"gamma").unwrap();
    writer.add_record(b"delta").unwrap();
    writer.finish().unwrap();

    // Read all back
    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"alpha");
    assert_eq!(reader.get(1).unwrap(), b"beta");
    assert_eq!(reader.get(2).unwrap(), b"gamma");
    assert_eq!(reader.get(3).unwrap(), b"delta");
    assert_eq!(reader.page_count(), 2); // original page + appended page
}

/// Write a single record and verify that out-of-range ordinals produce
/// `OrdinalNotFound`.
///
/// Tests the error path for `get()` with ordinals beyond the file's
/// range (positive, negative, and far out of range) and confirms that
/// `contains()` agrees.
#[test]
fn test_ordinal_not_found() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"only").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert!(reader.contains(0));
    assert!(!reader.contains(1));
    assert!(!reader.contains(-1));

    match reader.get(99) {
        Err(SlabError::OrdinalNotFound(99)) => {}
        Err(SlabError::WithContext { ref source, .. })
            if matches!(**source, SlabError::OrdinalNotFound(99)) => {}
        other => panic!("expected OrdinalNotFound(99), got {other:?}"),
    }
}

/// Write a single small record with alignment enabled (512-byte pages)
/// and verify the data page on disk is padded to a 512-byte boundary.
///
/// Reads back the data to confirm the alignment padding doesn't corrupt
/// the record, then inspects file layout to verify the data page size
/// is a multiple of the configured `min_page_size`.
#[test]
fn test_page_alignment() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::new(512, 512, u32::MAX, true).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"short").unwrap();
    writer.finish().unwrap();

    let metadata = std::fs::metadata(&path).unwrap();
    let file_len = metadata.len();
    // The data page should be aligned to 512 bytes
    // We can at least check the file has proper structure
    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"short");

    // Data page should be a multiple of 512
    let entries = reader.page_entries();
    assert_eq!(entries.len(), 1);
    // The file should consist of aligned data page + pages page
    // The data page starts at offset 0 and should be 512 bytes
    let data_page_offset = entries[0].file_offset as u64;
    assert_eq!(data_page_offset, 0);
    // Pages page starts right after (which means data page is 512)
    let pages_page_start = file_len - {
        // read pages page size from footer
        let mut f = std::fs::File::open(&path).unwrap();
        use std::io::{Read, Seek, SeekFrom};
        f.seek(SeekFrom::End(-16)).unwrap();
        let mut fb = [0u8; 16];
        f.read_exact(&mut fb).unwrap();
        let footer = slabtastic::Footer::read_from(&fb).unwrap();
        footer.page_size as u64
    };
    // data page occupies 0..pages_page_start, check it's multiple of 512
    assert_eq!(pages_page_start % 512, 0, "data page not aligned to 512");
}

/// Write a 10 000-byte record alongside a tiny record and read both back.
///
/// Verifies that large records (exceeding the header+footer overhead
/// multiple times) are stored and retrieved correctly, and that they
/// coexist with small records in the same page without corruption.
#[test]
fn test_large_records() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();

    let big_data = vec![0xABu8; 10_000];
    writer.add_record(&big_data).unwrap();
    writer.add_record(b"small").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), big_data);
    assert_eq!(reader.get(1).unwrap(), b"small");
}

/// Write zero-length records interleaved with a non-empty record.
///
/// Validates that the offset array correctly handles zero-length records
/// (consecutive offsets with the same value) and that they round-trip
/// as empty byte slices.
#[test]
fn test_empty_records() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"").unwrap();
    writer.add_record(b"nonempty").unwrap();
    writer.add_record(b"").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"");
    assert_eq!(reader.get(1).unwrap(), b"nonempty");
    assert_eq!(reader.get(2).unwrap(), b"");
}

/// Write 200 records with a 512-byte preferred page size to force
/// multiple data pages, then verify every record by ordinal.
///
/// With 13-byte records (`"record-NNNNNN"`) and 512-byte pages, each
/// page holds roughly 25–30 records, producing 7+ data pages. Tests
/// that the pages page correctly indexes across multiple data pages
/// and that binary search finds the right page for each ordinal.
#[test]
fn test_many_records_multi_page() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Use a small preferred page size to force multiple pages
    let config = WriterConfig::new(512, 512, u32::MAX, false).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();

    let count = 200;
    for i in 0..count {
        let data = format!("record-{i:06}").into_bytes();
        writer.add_record(&data).unwrap();
    }
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert!(reader.page_count() > 1, "expected multiple pages");

    for i in 0..count {
        let expected = format!("record-{i:06}").into_bytes();
        let actual = reader.get(i as i64).unwrap();
        assert_eq!(actual, expected, "mismatch at ordinal {i}");
    }
}

/// Write a single record and confirm `page_count()` returns 1.
///
/// Exercises the minimal file layout: one data page + one pages page.
#[test]
fn test_page_count() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"one").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.page_count(), 1);
}

/// Async read with progress polling loop.
///
/// Writes 100 records, then reads them back via `read_to_sink_async`,
/// polling progress until done.
#[test]
fn test_async_read_with_progress() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    for i in 0..100u32 {
        writer.add_record(&i.to_le_bytes()).unwrap();
    }
    writer.finish().unwrap();

    let callback_called = Arc::new(AtomicBool::new(false));
    let cb = Arc::clone(&callback_called);
    let sink: Vec<u8> = Vec::new();

    let task = SlabReader::read_to_sink_async(
        path,
        sink,
        move |count| {
            assert_eq!(count, 100);
            cb.store(true, Ordering::Release);
        },
    );

    // Poll progress until done
    while !task.is_done() {
        let _frac = task.progress().fraction();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let count = task.wait().unwrap();
    assert_eq!(count, 100);
    assert!(callback_called.load(Ordering::Acquire));
}

/// Async write with progress polling loop.
///
/// Writes 50 records via `write_from_iter_async` and polls progress
/// until done.
#[test]
fn test_async_write_with_progress() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let records: Vec<Vec<u8>> = (0..50)
        .map(|i| format!("write-{i:03}").into_bytes())
        .collect();

    let reported_count = Arc::new(AtomicU64::new(0));
    let rc = Arc::clone(&reported_count);

    let task = SlabWriter::write_from_iter_async(
        path.clone(),
        WriterConfig::default(),
        records.clone().into_iter(),
        move |count| {
            rc.store(count, Ordering::Release);
        },
    );

    // Poll progress until done
    while !task.is_done() {
        let _completed = task.progress().completed();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let count = task.wait().unwrap();
    assert_eq!(count, 50);
    assert_eq!(reported_count.load(Ordering::Acquire), 50);

    // Verify records are readable
    let reader = SlabReader::open(&path).unwrap();
    for (i, expected) in records.iter().enumerate() {
        assert_eq!(reader.get(i as i64).unwrap(), *expected);
    }
}

/// Round-trip: `write_from_iter_async` → `batch_iter` verify.
///
/// Writes records asynchronously, then reads them back via batch_iter
/// and verifies all data matches.
#[test]
fn test_async_write_then_batch_iter_roundtrip() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let records: Vec<Vec<u8>> = (0..30)
        .map(|i| format!("roundtrip-{i:03}").into_bytes())
        .collect();

    let task = SlabWriter::write_from_iter_async(
        path.clone(),
        WriterConfig::default(),
        records.clone().into_iter(),
        |_| {},
    );
    let written = task.wait().unwrap();
    assert_eq!(written, 30);

    // Read back via batch_iter with batch_size=7
    let reader = SlabReader::open(&path).unwrap();
    let mut iter = reader.batch_iter(7);
    let mut all: Vec<(i64, Vec<u8>)> = Vec::new();
    loop {
        let batch = iter.next_batch().unwrap();
        if batch.is_empty() {
            break;
        }
        all.extend(batch);
    }

    assert_eq!(all.len(), 30);
    for (i, (ordinal, data)) in all.iter().enumerate() {
        assert_eq!(*ordinal, i as i64);
        assert_eq!(data, &records[i]);
    }
}

/// Verify the SLAB_EXTENSION constant value.
#[test]
fn test_slab_extension_constant() {
    assert_eq!(SLAB_EXTENSION, ".slab");
}
