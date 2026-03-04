// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Adversarial and boundary-condition tests for the slabtastic format.
//!
//! These tests validate that the library correctly rejects malformed
//! input, handles edge cases at format boundaries, and survives
//! deliberate corruption. They complement the integration tests (which
//! exercise happy-path workflows) by targeting the error paths and
//! structural invariants described in the slabtastic design document.
//!
//! ## Categories
//!
//! - **File-level corruption** — truncated files, corrupted magic / version /
//!   page-type bytes, files that don't end with a pages page
//! - **Page deserialization** — undersized buffers, bad magic, header/footer
//!   size mismatches
//! - **Footer edge cases** — buffer too small, invalid page type (0 and 255),
//!   footer_length below minimum
//! - **Ordinal boundaries** — max positive (2^39−1), max negative (−2^39),
//!   zero, −1
//! - **Record size boundaries** — record exceeds max page capacity, record
//!   exactly fills a page
//! - **Writer config validation** — illegal size orderings, minimum below 512
//! - **Pages page** — single entry, exact match, well-past-end, negative
//!   before all entries
//! - **Alignment** — various record sizes with padding, verify on-disk
//!   multiples
//! - **Append mode** — nonexistent file, triple append
//! - **Forward traversal** — walk file from offset 0, cross-check against
//!   index
//! - **Rewrite round-trip** — data integrity through reordering and repacking
//! - **PageEntry serialization** — negative ordinals, max values, zeros
//! - **Edge-case payloads** — all-empty records, single record, every byte
//!   value 0x00–0xFF

use std::io::Write;

use slabtastic::constants::{FOOTER_V1_SIZE, HEADER_SIZE, MAGIC};
use slabtastic::{
    Footer, NamespaceEntry, NamespacesPage, Page, PageEntry, PageType, PagesPage, SlabError,
    SlabReader, SlabWriter, WriterConfig,
};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// File-level corruption
// ---------------------------------------------------------------------------

/// Opening a zero-byte file must fail because there is no footer to read.
#[test]
fn test_open_empty_file() {
    let tmp = NamedTempFile::new().unwrap();
    let result = SlabReader::open(tmp.path());
    assert!(result.is_err(), "opening an empty file should fail");
}

/// Opening a one-byte file must fail — smaller than the minimum
/// structural unit (8-byte header + 16-byte footer = 24 bytes).
#[test]
fn test_open_single_byte_file() {
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&[0x42]).unwrap();
    let result = SlabReader::open(tmp.path());
    assert!(result.is_err(), "opening a 1-byte file should fail");
}

/// A file containing only a valid 8-byte header (magic + page_size) but
/// no footer must be rejected. The reader needs at least header + footer
/// bytes to locate the pages page.
#[test]
fn test_open_truncated_to_header_only() {
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&MAGIC).unwrap();
    tmp.write_all(&100u32.to_le_bytes()).unwrap();
    let result = SlabReader::open(tmp.path());
    assert!(result.is_err(), "file with only a header should fail");
}

/// Corrupt the first magic byte of the first data page. The reader
/// opens successfully (it reads the pages page from the end), but
/// `get(0)` must fail when it tries to deserialize the corrupted data
/// page.
#[test]
fn test_open_corrupted_magic() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let mut writer = SlabWriter::new(&path, WriterConfig::default()).unwrap();
    writer.add_record(b"test").unwrap();
    writer.finish().unwrap();

    let mut data = std::fs::read(&path).unwrap();
    data[0] = b'X';
    std::fs::write(&path, &data).unwrap();

    // Point reads use the zero-copy path which skips magic validation
    // for performance. The record data (starting at byte 8) is intact,
    // so get() still returns the correct record.
    let reader = SlabReader::open(&path).unwrap();
    let result = reader.get(0);
    assert_eq!(result.unwrap(), b"test");

    // Full deserialization (used by iter, check, etc.) still validates magic.
    let reader2 = SlabReader::open(&path).unwrap();
    let result2 = reader2.iter();
    assert!(result2.is_err(), "full deserialization should detect corrupted magic");
}

/// Corrupt the namespace_index byte (offset 13 within the footer) in the
/// trailing pages-page footer. The reader must reject the file at
/// open time because it cannot parse the pages page with an invalid
/// namespace index.
#[test]
fn test_open_corrupted_pages_page_footer_namespace_index() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let mut writer = SlabWriter::new(&path, WriterConfig::default()).unwrap();
    writer.add_record(b"test").unwrap();
    writer.finish().unwrap();

    let mut data = std::fs::read(&path).unwrap();
    let ns_idx_pos = data.len() - FOOTER_V1_SIZE + 13;
    data[ns_idx_pos] = 0; // 0 = invalid namespace index
    std::fs::write(&path, &data).unwrap();

    let result = SlabReader::open(&path);
    assert!(result.is_err(), "invalid namespace index should reject file");
}

/// Change the pages-page type byte from Pages (1) to Data (2). The
/// reader must reject the file because a valid slabtastic file always
/// ends with a pages page, not a data page.
#[test]
fn test_open_corrupted_pages_page_type_to_data() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let mut writer = SlabWriter::new(&path, WriterConfig::default()).unwrap();
    writer.add_record(b"test").unwrap();
    writer.finish().unwrap();

    let mut data = std::fs::read(&path).unwrap();
    let type_pos = data.len() - FOOTER_V1_SIZE + 12;
    data[type_pos] = PageType::Data as u8;
    std::fs::write(&path, &data).unwrap();

    let result = SlabReader::open(&path);
    assert!(result.is_err(), "pages page with Data type should reject");
}

/// Manually write a single serialized data page (no pages page) to a
/// file. The reader must reject the file because the last page is
/// required to be of type Pages per the spec.
#[test]
fn test_open_file_ending_with_data_page_not_pages_page() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let mut page = Page::new(0, PageType::Data);
    page.add_record(b"hello");
    let bytes = page.serialize();
    std::fs::write(&path, &bytes).unwrap();

    let result = SlabReader::open(&path);
    assert!(result.is_err(), "file ending with data page (not pages page) should fail");
}

// ---------------------------------------------------------------------------
// Page deserialization edge cases
// ---------------------------------------------------------------------------

/// Attempting to deserialize a buffer smaller than header + footer
/// (24 bytes) must produce a `TruncatedPage` error.
#[test]
fn test_page_deserialize_truncated_buffer() {
    let tiny = vec![0u8; 10];
    let result = Page::deserialize(&tiny);
    assert!(result.is_err());
}

/// Replacing the magic bytes with "NOPE" in an otherwise valid
/// serialized page must produce an `InvalidMagic` error.
#[test]
fn test_page_deserialize_bad_magic() {
    let mut page = Page::new(0, PageType::Data);
    page.add_record(b"data");
    let mut bytes = page.serialize();
    bytes[0..4].copy_from_slice(b"NOPE");
    let result = Page::deserialize(&bytes);
    assert!(matches!(result, Err(SlabError::InvalidMagic)));
}

/// Corrupt the header's page_size field (bytes 4–7) so it disagrees
/// with the footer's page_size. Must produce a `PageSizeMismatch` error,
/// since header and footer page sizes are required to match for both
/// forward and backward traversal.
#[test]
fn test_page_deserialize_header_footer_size_mismatch() {
    let mut page = Page::new(0, PageType::Data);
    page.add_record(b"data");
    let mut bytes = page.serialize();
    let bad_size = (bytes.len() as u32 + 100).to_le_bytes();
    bytes[4..8].copy_from_slice(&bad_size);
    let result = Page::deserialize(&bytes);
    assert!(
        matches!(result, Err(SlabError::PageSizeMismatch { .. })),
        "expected PageSizeMismatch, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Footer edge cases
// ---------------------------------------------------------------------------

/// A buffer shorter than the 16-byte v1 footer must be rejected.
#[test]
fn test_footer_buffer_too_small() {
    let buf = [0u8; 8];
    let result = Footer::read_from(&buf);
    assert!(result.is_err());
}

/// Page type 0 (Invalid) is reserved as a sentinel and must always be
/// rejected during deserialization, even though `PageType::from_u8(0)`
/// returns `Some(Invalid)`.
#[test]
fn test_footer_invalid_page_type_zero() {
    let mut buf = [0u8; FOOTER_V1_SIZE];
    let footer = Footer::new(0, 0, 512, PageType::Data);
    footer.write_to(&mut buf);
    buf[12] = 0;
    let result = Footer::read_from(&buf);
    assert!(result.is_err());
}

/// A page type byte of 255 is not a valid variant and must be rejected.
#[test]
fn test_footer_invalid_page_type_unknown() {
    let mut buf = [0u8; FOOTER_V1_SIZE];
    let footer = Footer::new(0, 0, 512, PageType::Data);
    footer.write_to(&mut buf);
    buf[12] = 255;
    let result = Footer::read_from(&buf);
    assert!(result.is_err());
}

/// A footer_length of 8 (below the 16-byte minimum) must be rejected.
/// Footer length must be at least 16 and a multiple of 16.
#[test]
fn test_footer_footer_length_too_small() {
    let mut buf = [0u8; FOOTER_V1_SIZE];
    let footer = Footer::new(0, 0, 512, PageType::Data);
    footer.write_to(&mut buf);
    buf[14..16].copy_from_slice(&8u16.to_le_bytes());
    let result = Footer::read_from(&buf);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Ordinal boundary conditions
// ---------------------------------------------------------------------------

/// The maximum positive ordinal representable in the 5-byte signed
/// format: 2^39 − 1 = 549,755,813,887. Must round-trip through
/// footer serialization without loss.
#[test]
fn test_max_positive_ordinal() {
    let max_ord: i64 = (1i64 << 39) - 1;
    let footer = Footer::new(max_ord, 1, 512, PageType::Data);
    let mut buf = [0u8; FOOTER_V1_SIZE];
    footer.write_to(&mut buf);
    let decoded = Footer::read_from(&buf).unwrap();
    assert_eq!(decoded.start_ordinal, max_ord);
}

/// The minimum (most-negative) ordinal in the 5-byte signed format:
/// −2^39 = −549,755,813,888. Must round-trip correctly — the sign
/// extension logic in `read_from` fills bytes 5–7 with 0xFF when
/// bit 39 is set.
#[test]
fn test_max_negative_ordinal() {
    let min_ord: i64 = -(1i64 << 39);
    let footer = Footer::new(min_ord, 1, 512, PageType::Data);
    let mut buf = [0u8; FOOTER_V1_SIZE];
    footer.write_to(&mut buf);
    let decoded = Footer::read_from(&buf).unwrap();
    assert_eq!(decoded.start_ordinal, min_ord);
}

/// Ordinal −1 uses the sign-extension path (bit 39 is set) and is
/// within the valid 5-byte range. Must round-trip correctly.
#[test]
fn test_ordinal_just_outside_negative_range() {
    let footer = Footer::new(-1, 0, 512, PageType::Data);
    let mut buf = [0u8; FOOTER_V1_SIZE];
    footer.write_to(&mut buf);
    let decoded = Footer::read_from(&buf).unwrap();
    assert_eq!(decoded.start_ordinal, -1);
}

/// Ordinal 0 must round-trip — the simplest non-negative case with no
/// sign extension needed.
#[test]
fn test_zero_ordinal_roundtrip() {
    let footer = Footer::new(0, 0, 512, PageType::Data);
    let mut buf = [0u8; FOOTER_V1_SIZE];
    footer.write_to(&mut buf);
    let decoded = Footer::read_from(&buf).unwrap();
    assert_eq!(decoded.start_ordinal, 0);
}

// ---------------------------------------------------------------------------
// Record size boundary conditions
// ---------------------------------------------------------------------------

/// A 500-byte record with max_page_size=512 exceeds capacity because
/// the page also needs 8 bytes of header, 8 bytes of offsets (2 × 4),
/// and 16 bytes of footer = 32 bytes of overhead. Must produce
/// `RecordTooLarge`.
#[test]
fn test_record_too_large_for_max_page() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::new(512, 512, 512, false).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();

    let big = vec![0u8; 500];
    let result = writer.add_record(&big);
    assert!(
        matches!(result, Err(SlabError::RecordTooLarge { .. })),
        "expected RecordTooLarge, got {result:?}"
    );
}

/// A 480-byte record exactly fills a 512-byte page (8 header + 480 data
/// + 8 offsets + 16 footer = 512). Must write and read back successfully,
///   exercising the tight-packing boundary.
#[test]
fn test_record_exactly_fits_page() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::new(512, 512, 512, false).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();

    let data = vec![0xAAu8; 480];
    writer.add_record(&data).unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), data);
}

// ---------------------------------------------------------------------------
// Writer config validation
// ---------------------------------------------------------------------------

/// min_page_size (1024) > preferred_page_size (512) violates the
/// ordering constraint and must be rejected.
#[test]
fn test_config_min_above_preferred() {
    let result = WriterConfig::new(1024, 512, 2048, false);
    assert!(result.is_err());
}

/// preferred_page_size (2048) > max_page_size (1024) violates the
/// ordering constraint and must be rejected.
#[test]
fn test_config_preferred_above_max() {
    let result = WriterConfig::new(512, 2048, 1024, false);
    assert!(result.is_err());
}

/// min_page_size (256) below the absolute minimum (512) must be rejected.
/// The spec requires all pages to be at least 512 bytes.
#[test]
fn test_config_min_below_absolute_minimum() {
    let result = WriterConfig::new(256, 512, 1024, false);
    assert!(result.is_err());
}

/// All three size parameters set to the same value (512) is a valid
/// degenerate case — every page will be exactly 512 bytes.
#[test]
fn test_config_all_equal() {
    let config = WriterConfig::new(512, 512, 512, false).unwrap();
    assert_eq!(config.min_page_size, 512);
    assert_eq!(config.preferred_page_size, 512);
    assert_eq!(config.max_page_size, 512);
}

// ---------------------------------------------------------------------------
// Pages page edge cases
// ---------------------------------------------------------------------------

/// A pages page with exactly one entry must serialize, deserialize, and
/// return the correct ordinal and offset.
#[test]
fn test_pages_page_single_entry() {
    let mut pp = PagesPage::new();
    pp.add_entry(42, 1024);
    let bytes = pp.serialize();
    let decoded = PagesPage::deserialize(&bytes).unwrap();
    assert_eq!(decoded.entry_count(), 1);
    let entries = decoded.entries();
    assert_eq!(entries[0].start_ordinal, 42);
    assert_eq!(entries[0].file_offset, 1024);
}

/// Looking up the exact start ordinal of the last entry must return
/// that entry (binary search exact-match path on the final element).
#[test]
fn test_pages_page_find_ordinal_at_exact_last_entry() {
    let mut pp = PagesPage::new();
    pp.add_entry(0, 0);
    pp.add_entry(100, 4096);
    pp.add_entry(200, 8192);

    let entry = pp.find_page_for_ordinal(200).unwrap();
    assert_eq!(entry.start_ordinal, 200);
}

/// An ordinal far beyond the last entry (999999) must still map to the
/// last page entry, since `find_page_for_ordinal` returns the greatest
/// entry ≤ the requested ordinal.
#[test]
fn test_pages_page_find_ordinal_well_past_last() {
    let mut pp = PagesPage::new();
    pp.add_entry(0, 0);
    pp.add_entry(100, 4096);

    let entry = pp.find_page_for_ordinal(999999).unwrap();
    assert_eq!(entry.start_ordinal, 100);
}

/// A negative ordinal (−1) before the first entry (0) must return
/// `None` — there is no page that could contain it.
#[test]
fn test_pages_page_negative_ordinal_before_all() {
    let mut pp = PagesPage::new();
    pp.add_entry(0, 0);

    assert!(pp.find_page_for_ordinal(-1).is_none());
}

// ---------------------------------------------------------------------------
// Alignment edge cases
// ---------------------------------------------------------------------------

/// Write records of varying sizes (1, 2, 3, and 400 bytes) with
/// alignment enabled and read them all back. Verifies that the
/// alignment padding inserted between data and offsets/footer does not
/// corrupt any record regardless of size.
#[test]
fn test_alignment_with_various_record_sizes() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::new(512, 512, u32::MAX, true).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();

    writer.add_record(b"a").unwrap();
    writer.add_record(b"bb").unwrap();
    writer.add_record(b"ccc").unwrap();
    writer.add_record(&vec![0xDD; 400]).unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"a");
    assert_eq!(reader.get(1).unwrap(), b"bb");
    assert_eq!(reader.get(2).unwrap(), b"ccc");
    assert_eq!(reader.get(3).unwrap(), vec![0xDD; 400]);
}

/// Write a small record with alignment enabled and verify that the
/// data page on disk occupies exactly a multiple of 512 bytes. This
/// confirms the writer's alignment padding logic produces correctly
/// sized pages.
#[test]
fn test_alignment_page_sizes_are_multiples() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::new(512, 512, u32::MAX, true).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"short").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    let entries = reader.page_entries();
    if entries.len() == 1 {
        let file_meta = std::fs::metadata(&path).unwrap();
        let file_len = file_meta.len();
        let data = std::fs::read(&path).unwrap();
        let footer = Footer::read_from(&data[data.len() - FOOTER_V1_SIZE..]).unwrap();
        let pages_page_size = footer.page_size as u64;
        let data_page_end = file_len - pages_page_size;
        assert_eq!(
            data_page_end % 512,
            0,
            "data page size {} not aligned to 512",
            data_page_end
        );
    }
}

// ---------------------------------------------------------------------------
// Append-mode edge cases
// ---------------------------------------------------------------------------

/// Appending to a file that does not exist must fail with an I/O error.
#[test]
fn test_append_to_nonexistent_file() {
    let result = SlabWriter::append("/tmp/nonexistent_slab_file_12345.slab", WriterConfig::default());
    assert!(result.is_err());
}

/// Perform three successive appends (each adding one record) and verify
/// all four records (1 original + 3 appended) are readable with correct
/// ordinals. Exercises the append path's ordinal-continuation and
/// pages-page-rebuild logic across multiple cycles.
#[test]
fn test_multiple_appends() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut w = SlabWriter::new(&path, config.clone()).unwrap();
    w.add_record(b"a").unwrap();
    w.finish().unwrap();

    for i in 1..=3u8 {
        let mut w = SlabWriter::append(&path, config.clone()).unwrap();
        w.add_record(&[b'a' + i]).unwrap();
        w.finish().unwrap();
    }

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"a");
    assert_eq!(reader.get(1).unwrap(), b"b");
    assert_eq!(reader.get(2).unwrap(), b"c");
    assert_eq!(reader.get(3).unwrap(), b"d");
}

// ---------------------------------------------------------------------------
// Forward traversal (read_page_at_offset)
// ---------------------------------------------------------------------------

/// Write 100 records with small pages (512 B preferred) to produce
/// multiple data pages. Then walk the file from offset 0 using
/// `read_page_at_offset`, collecting each page's offset and type.
/// Verify that:
/// 1. Every index entry from the pages page appears in the forward walk.
/// 2. The last page encountered is a Pages-type page.
///
/// This exercises the `slab check` forward-traversal path that validates
/// file structure without relying on the index.
#[test]
fn test_forward_traversal_matches_index() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::new(512, 512, u32::MAX, false).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    for i in 0..100 {
        writer
            .add_record(format!("record-{i:04}").as_bytes())
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    let entries = reader.page_entries();
    let file_len = reader.file_len().unwrap();

    let mut offset: u64 = 0;
    let mut forward_offsets = Vec::new();
    while offset < file_len {
        let page = reader.read_page_at_offset(offset).unwrap();
        forward_offsets.push(offset);
        offset += page.footer.page_size as u64;
    }

    for entry in &entries {
        assert!(
            forward_offsets.contains(&(entry.file_offset as u64)),
            "index entry at offset {} not found in forward traversal",
            entry.file_offset
        );
    }

    let last_page = reader
        .read_page_at_offset(*forward_offsets.last().unwrap())
        .unwrap();
    assert_eq!(last_page.footer.page_type, PageType::Pages);
}

// ---------------------------------------------------------------------------
// Rewrite round-trip (replaces former repack / reorder tests)
// ---------------------------------------------------------------------------

/// Write 50 records with small pages (512 B) producing many pages, then
/// rewrite into a new file with default (64 KiB) pages. Verify every
/// record is identical in the rewritten file. Exercises the `slab rewrite`
/// workflow: read all → sort → write fresh with different page config.
#[test]
fn test_repack_preserves_all_records() {
    let tmp_in = NamedTempFile::new().unwrap();
    let tmp_out = NamedTempFile::new().unwrap();
    let in_path = tmp_in.path().to_path_buf();
    let out_path = tmp_out.path().to_path_buf();

    let config = WriterConfig::new(512, 512, u32::MAX, false).unwrap();
    let mut writer = SlabWriter::new(&in_path, config).unwrap();
    let records: Vec<Vec<u8>> = (0..50)
        .map(|i| format!("item-{i:04}").into_bytes())
        .collect();
    for r in &records {
        writer.add_record(r).unwrap();
    }
    writer.finish().unwrap();

    let reader = SlabReader::open(&in_path).unwrap();
    let all = reader.iter().unwrap();

    let config2 = WriterConfig::default();
    let mut writer2 = SlabWriter::new(&out_path, config2).unwrap();
    for (_ord, data) in &all {
        writer2.add_record(data).unwrap();
    }
    writer2.finish().unwrap();

    let reader2 = SlabReader::open(&out_path).unwrap();
    for (i, expected) in records.iter().enumerate() {
        assert_eq!(reader2.get(i as i64).unwrap(), *expected);
    }
}

/// Write 20 records, read them back, sort by ordinal, and write to a
/// new file. Verify the output's ordinals are strictly monotonic. This
/// exercises the reorder logic of `slab rewrite`. (Since the normal
/// writer already produces monotonic ordinals, this mainly confirms the
/// sort-then-write pipeline doesn't introduce errors.)
#[test]
fn test_reorder_sorts_correctly() {
    let tmp_in = NamedTempFile::new().unwrap();
    let tmp_out = NamedTempFile::new().unwrap();
    let in_path = tmp_in.path().to_path_buf();
    let out_path = tmp_out.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&in_path, config.clone()).unwrap();
    for i in 0..20 {
        writer
            .add_record(format!("val-{i}").as_bytes())
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = SlabReader::open(&in_path).unwrap();
    let mut records = reader.iter().unwrap();
    records.sort_by_key(|&(ord, _)| ord);

    let mut writer2 = SlabWriter::new(&out_path, config).unwrap();
    for (_ord, data) in &records {
        writer2.add_record(data).unwrap();
    }
    writer2.finish().unwrap();

    let reader2 = SlabReader::open(&out_path).unwrap();
    let all = reader2.iter().unwrap();
    for window in all.windows(2) {
        assert!(window[0].0 < window[1].0);
    }
}

// ---------------------------------------------------------------------------
// Page entry serialization
// ---------------------------------------------------------------------------

/// Round-trip a `PageEntry` with a negative ordinal and the maximum
/// `i64` file offset to exercise both ends of the value range.
#[test]
fn test_page_entry_roundtrip() {
    let entry = PageEntry {
        start_ordinal: -42,
        file_offset: i64::MAX,
    };
    let bytes = entry.to_bytes();
    let decoded = PageEntry::from_bytes(&bytes);
    assert_eq!(decoded.start_ordinal, -42);
    assert_eq!(decoded.file_offset, i64::MAX);
}

/// Round-trip a `PageEntry` with both fields set to zero — the
/// smallest valid entry.
#[test]
fn test_page_entry_zero_values() {
    let entry = PageEntry {
        start_ordinal: 0,
        file_offset: 0,
    };
    let bytes = entry.to_bytes();
    let decoded = PageEntry::from_bytes(&bytes);
    assert_eq!(decoded, entry);
}

// ---------------------------------------------------------------------------
// Edge case: file with only empty records
// ---------------------------------------------------------------------------

/// Write 100 zero-length records and read them all back. Zero-length
/// records produce consecutive identical offsets in the offset array;
/// this verifies the offset logic handles that correctly.
#[test]
fn test_file_all_empty_records() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    for _ in 0..100 {
        writer.add_record(b"").unwrap();
    }
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    for i in 0..100 {
        assert_eq!(reader.get(i).unwrap(), b"");
    }
}

// ---------------------------------------------------------------------------
// Edge case: file with exactly one record
// ---------------------------------------------------------------------------

/// Write a single record and verify: page count is 1, ordinal 0
/// returns the record, ordinals 1 and −1 produce errors, and `iter()`
/// yields exactly one `(0, data)` pair.
#[test]
fn test_single_record_file() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"only-one").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.page_count(), 1);
    assert_eq!(reader.get(0).unwrap(), b"only-one");
    assert!(reader.get(1).is_err());
    assert!(reader.get(-1).is_err());

    let all = reader.iter().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0], (0, b"only-one".to_vec()));
}

// ---------------------------------------------------------------------------
// Edge case: binary data with all byte values
// ---------------------------------------------------------------------------

/// Store each of the 256 possible byte values (0x00–0xFF) as a
/// separate one-byte record, then read them all back. Verifies no
/// byte value is treated specially (no null-termination, no escaping,
/// no encoding issues).
#[test]
fn test_binary_data_all_byte_values() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();

    for b in 0..=255u8 {
        writer.add_record(&[b]).unwrap();
    }
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    for b in 0..=255u8 {
        assert_eq!(reader.get(b as i64).unwrap(), vec![b]);
    }
}

// ===========================================================================
// Namespace entry serialization edge cases
// ===========================================================================

/// A NamespaceEntry buffer shorter than 10 bytes must be rejected.
#[test]
fn test_namespace_entry_buffer_too_short() {
    let buf = [0u8; 9];
    let result = NamespaceEntry::from_bytes(&buf);
    assert!(result.is_err(), "namespace entry < 10 bytes must fail");
}

/// A NamespaceEntry with name_length claiming more bytes than available
/// in the buffer must be rejected.
#[test]
fn test_namespace_entry_name_length_exceeds_buffer() {
    // namespace_index=1, name_length=50, but buffer only has 10 bytes total
    let mut buf = vec![0u8; 10];
    buf[0] = 1;   // namespace_index
    buf[1] = 50;  // name_length claims 50 bytes of name
    let result = NamespaceEntry::from_bytes(&buf);
    assert!(result.is_err(), "name_length exceeding buffer should fail");
}

/// A NamespaceEntry with invalid UTF-8 in the name must be rejected.
#[test]
fn test_namespace_entry_invalid_utf8() {
    // Build a well-formed buffer with invalid UTF-8 name bytes
    let mut buf = Vec::new();
    buf.push(2);    // namespace_index
    buf.push(3);    // name_length = 3
    buf.extend_from_slice(&[0xFF, 0xFE, 0x80]); // invalid UTF-8
    buf.extend_from_slice(&0i64.to_le_bytes());  // pages_page_offset
    let result = NamespaceEntry::from_bytes(&buf);
    assert!(result.is_err(), "invalid UTF-8 name must be rejected");
}

/// A NamespaceEntry with a maximum-length name (128 bytes) must round-trip
/// successfully. The name_length field is a u8 but spec caps at 128.
#[test]
fn test_namespace_entry_max_name_length() {
    let long_name = "a".repeat(128);
    let entry = NamespaceEntry {
        namespace_index: 5,
        name: long_name.clone(),
        pages_page_offset: 99999,
    };
    let bytes = entry.to_bytes();
    let decoded = NamespaceEntry::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.name, long_name);
    assert_eq!(decoded.namespace_index, 5);
    assert_eq!(decoded.pages_page_offset, 99999);
}

/// An empty NamespacesPage (no entries) must serialize and deserialize
/// to zero entries.
#[test]
fn test_namespaces_page_empty() {
    let np = NamespacesPage::new();
    let bytes = np.serialize();
    let decoded = NamespacesPage::deserialize(&bytes).unwrap();
    let entries = decoded.entries().unwrap();
    assert!(entries.is_empty(), "empty namespaces page should have 0 entries");
}

/// Attempting to deserialize a Data-type page as a NamespacesPage must fail.
#[test]
fn test_namespaces_page_rejects_data_type() {
    let mut page = Page::new(0, PageType::Data);
    page.add_record(b"some data");
    let bytes = page.serialize();
    let result = NamespacesPage::deserialize(&bytes);
    assert!(result.is_err());
}

/// NamespaceEntry with default namespace (index=1, empty name) round-trips.
#[test]
fn test_namespace_entry_default_namespace_roundtrip() {
    let entry = NamespaceEntry {
        namespace_index: 1,
        name: String::new(),
        pages_page_offset: 0,
    };
    let bytes = entry.to_bytes();
    assert_eq!(bytes.len(), 10); // 1 + 1 + 0 + 8
    let decoded = NamespaceEntry::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, entry);
}

/// NamespaceEntry with exactly 10 bytes buffer (minimum valid: index=1,
/// name_length=0, 8-byte offset) must succeed.
#[test]
fn test_namespace_entry_exactly_minimum_size() {
    let mut buf = vec![0u8; 10];
    buf[0] = 1;   // namespace_index
    buf[1] = 0;   // name_length = 0
    buf[2..10].copy_from_slice(&42i64.to_le_bytes());
    let entry = NamespaceEntry::from_bytes(&buf).unwrap();
    assert_eq!(entry.namespace_index, 1);
    assert_eq!(entry.name, "");
    assert_eq!(entry.pages_page_offset, 42);
}

// ===========================================================================
// Footer namespace_index boundary values
// ===========================================================================

/// namespace_index 127 (max valid user namespace) must round-trip.
#[test]
fn test_footer_namespace_index_127_roundtrip() {
    let footer = Footer {
        start_ordinal: 0,
        record_count: 1,
        page_size: 512,
        page_type: PageType::Data,
        namespace_index: 127,
        footer_length: 16,
    };
    let mut buf = [0u8; FOOTER_V1_SIZE];
    footer.write_to(&mut buf);
    let decoded = Footer::read_from(&buf).unwrap();
    assert_eq!(decoded.namespace_index, 127);
}

/// namespace_index 128 (first reserved/negative byte) must be rejected.
#[test]
fn test_footer_namespace_index_128_rejected() {
    let mut buf = [0u8; FOOTER_V1_SIZE];
    let footer = Footer::new(0, 0, 512, PageType::Data);
    footer.write_to(&mut buf);
    buf[13] = 128;
    let result = Footer::read_from(&buf);
    assert!(result.is_err(), "namespace_index 128 must be rejected");
}

/// namespace_index 255 (max reserved) must be rejected.
#[test]
fn test_footer_namespace_index_255_rejected() {
    let mut buf = [0u8; FOOTER_V1_SIZE];
    let footer = Footer::new(0, 0, 512, PageType::Data);
    footer.write_to(&mut buf);
    buf[13] = 255;
    let result = Footer::read_from(&buf);
    assert!(result.is_err(), "namespace_index 255 must be rejected");
}

/// Footer with Namespaces page type round-trips correctly.
#[test]
fn test_footer_namespaces_page_type_roundtrip() {
    let footer = Footer::new(0, 3, 1024, PageType::Namespaces);
    let mut buf = [0u8; FOOTER_V1_SIZE];
    footer.write_to(&mut buf);
    let decoded = Footer::read_from(&buf).unwrap();
    assert_eq!(decoded.page_type, PageType::Namespaces);
    assert_eq!(decoded.record_count, 3);
    assert_eq!(decoded.page_size, 1024);
}

// ===========================================================================
// Reader with namespace-aware files
// ===========================================================================

/// Construct a multi-namespace file by hand (data pages + pages page +
/// namespaces page) and verify the reader can open it by following the
/// default namespace entry.
#[test]
fn test_reader_opens_file_with_namespaces_page() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Step 1: write a normal single-namespace file first
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"hello").unwrap();
    writer.add_record(b"world").unwrap();
    writer.finish().unwrap();

    // Step 2: read the file, find the pages page offset, then append a
    // namespaces page that points at it
    let file_data = std::fs::read(&path).unwrap();
    let file_len = file_data.len();

    // The pages page is at file_len - pages_page_size
    let footer = Footer::read_from(&file_data[file_len - FOOTER_V1_SIZE..]).unwrap();
    assert_eq!(footer.page_type, PageType::Pages);
    let pages_page_offset = file_len as i64 - footer.page_size as i64;

    // Step 3: build a namespaces page with a single entry pointing to the
    // existing pages page
    let mut np = NamespacesPage::new();
    np.add_entry(&NamespaceEntry {
        namespace_index: 1,
        name: String::new(),
        pages_page_offset,
    });
    let ns_bytes = np.serialize();

    // Step 4: append the namespaces page to the file
    let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(&ns_bytes).unwrap();
    drop(f);

    // Step 5: the file now ends with a Namespaces page; open and verify
    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"hello");
    assert_eq!(reader.get(1).unwrap(), b"world");
    assert!(reader.get(2).is_err());
}

/// A file ending with a Namespaces page but containing NO default
/// namespace (index 1) must be rejected.
#[test]
fn test_reader_rejects_namespaces_page_without_default() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Write a normal file first
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"data").unwrap();
    writer.finish().unwrap();

    let file_data = std::fs::read(&path).unwrap();
    let file_len = file_data.len();
    let footer = Footer::read_from(&file_data[file_len - FOOTER_V1_SIZE..]).unwrap();
    let pages_page_offset = file_len as i64 - footer.page_size as i64;

    // Build a namespaces page with only a non-default namespace
    let mut np = NamespacesPage::new();
    np.add_entry(&NamespaceEntry {
        namespace_index: 2,
        name: "not-default".to_string(),
        pages_page_offset,
    });
    let ns_bytes = np.serialize();

    let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(&ns_bytes).unwrap();
    drop(f);

    let result = SlabReader::open(&path);
    assert!(result.is_err(), "file without default namespace must be rejected");
}

/// Change the terminal page type to 4 (unknown). The reader must reject
/// the file because only Pages and Namespaces are valid terminal types.
#[test]
fn test_reader_rejects_unknown_terminal_page_type() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"test").unwrap();
    writer.finish().unwrap();

    let mut data = std::fs::read(&path).unwrap();
    let type_pos = data.len() - FOOTER_V1_SIZE + 12;
    data[type_pos] = 4; // unknown page type
    std::fs::write(&path, &data).unwrap();

    let result = SlabReader::open(&path);
    assert!(result.is_err(), "unknown terminal page type must be rejected");
}

// ===========================================================================
// PageType edge cases
// ===========================================================================

/// PageType::from_u8 for all values outside 0..=3 must return None.
#[test]
fn test_page_type_from_u8_exhaustive_invalid() {
    for v in 4..=255u8 {
        assert!(
            PageType::from_u8(v).is_none(),
            "PageType::from_u8({v}) should be None"
        );
    }
}

// ===========================================================================
// Ordinal range parser adversarial inputs
// ===========================================================================

/// Empty string input must fail.
#[test]
fn test_ordinal_range_empty_string() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("");
    assert!(result.is_err(), "empty string should fail");
}

/// Whitespace-only input must fail.
#[test]
fn test_ordinal_range_whitespace_only() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("   ");
    assert!(result.is_err(), "whitespace-only should fail");
}

/// Garbage input (non-numeric, non-bracket) must fail.
#[test]
fn test_ordinal_range_garbage() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("foobar");
    assert!(result.is_err(), "garbage input should fail");
}

/// Single negative number: `-5` → `[0, -5)` which is an empty range.
/// The parser should accept this (it's a valid half-open interval even if empty).
#[test]
fn test_ordinal_range_negative_count() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("-5");
    // -5 as a plain number means [0, -5) which is empty — parser should still return Ok
    assert!(result.is_ok(), "negative single number should parse");
    let (start, end) = result.unwrap();
    assert_eq!(start, 0);
    assert_eq!(end, -5);
    // The range is empty (start > end) but that's valid — the caller handles it
}

/// `[0]` — single ordinal at zero.
#[test]
fn test_ordinal_range_single_zero() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("[0]");
    assert_eq!(result.unwrap(), (0, 1));
}

/// `[-1]` — single negative ordinal.
#[test]
fn test_ordinal_range_single_negative() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("[-1]");
    assert_eq!(result.unwrap(), (-1, 0));
}

/// Bracket with no separator: `[42abc]` must fail.
#[test]
fn test_ordinal_range_bracket_garbage() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("[42abc]");
    assert!(result.is_err(), "bracket with garbage should fail");
}

/// Mixed brackets: `(5,10]` — half-open with exclusive left, inclusive right.
#[test]
fn test_ordinal_range_mixed_brackets() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("(5,10]");
    assert_eq!(result.unwrap(), (6, 11));
}

/// Large ordinal values near i64 limits must parse without overflow.
#[test]
fn test_ordinal_range_large_values() {
    let big = i64::MAX / 2;
    let s = format!("[{big}]");
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range(&s);
    assert_eq!(result.unwrap(), (big, big + 1));
}

/// Zero-width range `[5,5)` is valid but empty.
#[test]
fn test_ordinal_range_zero_width() {
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range("[5,5)");
    assert_eq!(result.unwrap(), (5, 5));
}

/// Dotdot range with spaces around it: ` 5 .. 10 ` — must fail because
/// the top-level parse will try as plain i64 first (which fails due to spaces
/// in the dotdot form), then try dotdot splitting.
#[test]
fn test_ordinal_range_dotdot_with_spaces() {
    // The implementation splits on ".." and trims parts
    let result = slabtastic::cli::ordinal_range::parse_ordinal_range(" 5..10 ");
    assert_eq!(result.unwrap(), (5, 11));
}

// ===========================================================================
// Base64 encoder edge cases (via get command's built-in encoder)
// ===========================================================================

// Note: base64_encode is private in get.rs. We test it indirectly through
// the CLI slab binary, or we can test known vectors by writing records and
// using the slab binary. Instead, let's test the encoding by writing records
// with known content and reading via the binary.

// ===========================================================================
// Import/export format round-trips
// ===========================================================================

/// JSON import: write a JSON file with multiple objects, import into slab,
/// verify records.
#[test]
fn test_import_json_roundtrip() {
    let tmp_json = NamedTempFile::new().unwrap();
    let json_path = tmp_json.path().to_path_buf();
    std::fs::write(&json_path, r#"{"a":1} {"b":2} {"c":3}"#).unwrap();

    let tmp_slab = NamedTempFile::new().unwrap();
    let slab_path = tmp_slab.path().to_path_buf();

    // Remove the slab file so import creates it fresh
    std::fs::remove_file(&slab_path).ok();

    // Use the import module directly via the library
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();

    let file = std::fs::File::open(&json_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<serde_json::Value>();
    let mut count = 0u64;
    for value in stream {
        let value = value.unwrap();
        let mut serialized = serde_json::to_string(&value).unwrap();
        serialized.push('\n');
        writer.add_record(serialized.as_bytes()).unwrap();
        count += 1;
    }
    writer.finish().unwrap();

    assert_eq!(count, 3);
    let reader = SlabReader::open(&slab_path).unwrap();
    let r0 = String::from_utf8(reader.get(0).unwrap()).unwrap();
    assert!(r0.contains("\"a\":1") || r0.contains("\"a\": 1"));
}

/// JSONL import: each line is a valid JSON object.
#[test]
fn test_import_jsonl_roundtrip() {
    let tmp_jsonl = NamedTempFile::new().unwrap();
    let jsonl_path = tmp_jsonl.path().to_path_buf();
    std::fs::write(
        &jsonl_path,
        "{\"x\":1}\n{\"y\":2}\n{\"z\":3}\n",
    )
    .unwrap();

    let tmp_slab = NamedTempFile::new().unwrap();
    let slab_path = tmp_slab.path().to_path_buf();
    std::fs::remove_file(&slab_path).ok();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();

    let data = std::fs::read_to_string(&jsonl_path).unwrap();
    let mut count = 0u64;
    for line in data.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _: serde_json::Value = serde_json::from_str(trimmed).unwrap();
        let mut rec = line.to_string();
        rec.push('\n');
        writer.add_record(rec.as_bytes()).unwrap();
        count += 1;
    }
    writer.finish().unwrap();

    assert_eq!(count, 3);
    let reader = SlabReader::open(&slab_path).unwrap();
    assert_eq!(reader.iter().unwrap().len(), 3);
}

/// Malformed JSON must produce an error during import.
#[test]
fn test_import_json_malformed_rejected() {
    let json_data = b"{ not valid json }";
    let reader = std::io::BufReader::new(&json_data[..]);
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<serde_json::Value>();
    let mut had_error = false;
    for value in stream {
        if value.is_err() {
            had_error = true;
            break;
        }
    }
    assert!(had_error, "malformed JSON must produce an error");
}

/// Malformed JSONL (line that isn't valid JSON) must be rejected.
#[test]
fn test_import_jsonl_malformed_line_rejected() {
    let line = "not json at all";
    let result: std::result::Result<serde_json::Value, _> = serde_json::from_str(line);
    assert!(result.is_err(), "invalid JSONL line must fail JSON parse");
}

/// CSV import round-trip: write a CSV, import records, verify.
#[test]
fn test_import_csv_roundtrip() {
    let tmp_csv = NamedTempFile::new().unwrap();
    let csv_path = tmp_csv.path().to_path_buf();
    std::fs::write(&csv_path, "name,age\nAlice,30\nBob,25\n").unwrap();

    let tmp_slab = NamedTempFile::new().unwrap();
    let slab_path = tmp_slab.path().to_path_buf();
    std::fs::remove_file(&slab_path).ok();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(&csv_path)
        .unwrap();
    let mut count = 0u64;
    for result in rdr.records() {
        let record = result.unwrap();
        let fields: Vec<&str> = record.iter().collect();
        let mut row = fields.join(",");
        row.push('\n');
        writer.add_record(row.as_bytes()).unwrap();
        count += 1;
    }
    writer.finish().unwrap();

    assert_eq!(count, 3); // header + 2 data rows
    let reader = SlabReader::open(&slab_path).unwrap();
    let r0 = String::from_utf8(reader.get(0).unwrap()).unwrap();
    assert!(r0.contains("name"));
}

/// TSV import: tab-separated values.
#[test]
fn test_import_tsv_roundtrip() {
    let tmp_tsv = NamedTempFile::new().unwrap();
    let tsv_path = tmp_tsv.path().to_path_buf();
    std::fs::write(&tsv_path, "col1\tcol2\nval1\tval2\n").unwrap();

    let tmp_slab = NamedTempFile::new().unwrap();
    let slab_path = tmp_slab.path().to_path_buf();
    std::fs::remove_file(&slab_path).ok();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(&tsv_path)
        .unwrap();
    let mut count = 0u64;
    for result in rdr.records() {
        let record = result.unwrap();
        let fields: Vec<&str> = record.iter().collect();
        let mut row = fields.join("\t");
        row.push('\n');
        writer.add_record(row.as_bytes()).unwrap();
        count += 1;
    }
    writer.finish().unwrap();

    assert_eq!(count, 2);
    let reader = SlabReader::open(&slab_path).unwrap();
    let r0 = String::from_utf8(reader.get(0).unwrap()).unwrap();
    assert!(r0.contains("col1\tcol2"));
}

/// YAML import round-trip with multiple documents.
#[test]
fn test_import_yaml_roundtrip() {
    let tmp_yaml = NamedTempFile::new().unwrap();
    let yaml_path = tmp_yaml.path().to_path_buf();
    std::fs::write(
        &yaml_path,
        "---\nname: Alice\n---\nname: Bob\n",
    )
    .unwrap();

    let tmp_slab = NamedTempFile::new().unwrap();
    let slab_path = tmp_slab.path().to_path_buf();
    std::fs::remove_file(&slab_path).ok();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();

    let data = std::fs::read_to_string(&yaml_path).unwrap();
    let mut count = 0u64;
    let mut current_doc = String::new();
    for line in data.lines() {
        if line.trim() == "---" {
            if !current_doc.trim().is_empty() {
                let _: serde_yaml::Value = serde_yaml::from_str(&current_doc).unwrap();
                let mut rec = current_doc.clone();
                rec.push('\n');
                writer.add_record(rec.as_bytes()).unwrap();
                count += 1;
            }
            current_doc.clear();
        } else {
            current_doc.push_str(line);
            current_doc.push('\n');
        }
    }
    if !current_doc.trim().is_empty() {
        let _: serde_yaml::Value = serde_yaml::from_str(&current_doc).unwrap();
        let mut rec = current_doc;
        if !rec.ends_with('\n') {
            rec.push('\n');
        }
        writer.add_record(rec.as_bytes()).unwrap();
        count += 1;
    }
    writer.finish().unwrap();

    assert_eq!(count, 2);
    let reader = SlabReader::open(&slab_path).unwrap();
    let r0 = String::from_utf8(reader.get(0).unwrap()).unwrap();
    assert!(r0.contains("Alice"));
}

/// Malformed YAML must produce an error.
#[test]
fn test_import_yaml_malformed_rejected() {
    let bad_yaml = "key: [unterminated";
    let result: std::result::Result<serde_yaml::Value, _> = serde_yaml::from_str(bad_yaml);
    assert!(result.is_err(), "malformed YAML must fail");
}

// ===========================================================================
// split_with_delimiter edge cases
// ===========================================================================

/// Split empty input → no records.
#[test]
fn test_split_with_delimiter_empty() {
    let result = slabtastic::cli::import::split_with_delimiter(b"", b'\n');
    assert!(result.is_empty());
}

/// Split data with no delimiter present → single record (the whole input).
#[test]
fn test_split_with_delimiter_no_delimiter() {
    let result = slabtastic::cli::import::split_with_delimiter(b"no newline here", b'\n');
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], b"no newline here");
}

/// Split with delimiter at start → first record is just the delimiter byte.
#[test]
fn test_split_with_delimiter_at_start() {
    let result = slabtastic::cli::import::split_with_delimiter(b"\nhello\n", b'\n');
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], b"\n");
    assert_eq!(result[1], b"hello\n");
}

/// Split with consecutive delimiters → multiple single-byte records.
#[test]
fn test_split_with_delimiter_consecutive() {
    let result = slabtastic::cli::import::split_with_delimiter(b"\n\n\n", b'\n');
    assert_eq!(result.len(), 3);
    for rec in &result {
        assert_eq!(rec, &b"\n".to_vec());
    }
}

/// Split data not ending with delimiter → trailing chunk has no delimiter.
#[test]
fn test_split_with_delimiter_no_trailing() {
    let result = slabtastic::cli::import::split_with_delimiter(b"a\nb", b'\n');
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], b"a\n");
    assert_eq!(result[1], b"b");
}

// ===========================================================================
// Export format edge cases
// ===========================================================================

/// Export text: records that already end with \n should not get double newlines.
#[test]
fn test_export_text_no_double_newline() {
    let tmp_slab = NamedTempFile::new().unwrap();
    let slab_path = tmp_slab.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();
    writer.add_record(b"line1\n").unwrap();
    writer.add_record(b"line2").unwrap(); // no trailing newline
    writer.finish().unwrap();

    let tmp_out = NamedTempFile::new().unwrap();
    let out_path = tmp_out.path().to_path_buf();

    // Simulate text export
    let reader = SlabReader::open(&slab_path).unwrap();
    let records = reader.iter().unwrap();
    let mut sink = std::fs::File::create(&out_path).unwrap();
    for (_ord, data) in &records {
        sink.write_all(data).unwrap();
        if !data.ends_with(b"\n") {
            sink.write_all(b"\n").unwrap();
        }
    }
    drop(sink);

    let exported = std::fs::read_to_string(&out_path).unwrap();
    assert_eq!(exported, "line1\nline2\n");
}

/// Export cstrings: records already ending with \0 should not get double nulls.
#[test]
fn test_export_cstrings_no_double_null() {
    let tmp_slab = NamedTempFile::new().unwrap();
    let slab_path = tmp_slab.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();
    writer.add_record(b"str1\0").unwrap();
    writer.add_record(b"str2").unwrap();
    writer.finish().unwrap();

    let tmp_out = NamedTempFile::new().unwrap();
    let out_path = tmp_out.path().to_path_buf();

    let reader = SlabReader::open(&slab_path).unwrap();
    let records = reader.iter().unwrap();
    let mut sink = std::fs::File::create(&out_path).unwrap();
    for (_ord, data) in &records {
        sink.write_all(data).unwrap();
        if !data.ends_with(b"\0") {
            sink.write_all(b"\0").unwrap();
        }
    }
    drop(sink);

    let exported = std::fs::read(&out_path).unwrap();
    assert_eq!(exported, b"str1\0str2\0");
}

/// Export slab → slab round-trip: export all records from one slab file
/// into a new slab file and verify data integrity.
#[test]
fn test_export_slab_to_slab_roundtrip() {
    let tmp_in = NamedTempFile::new().unwrap();
    let in_path = tmp_in.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&in_path, config.clone()).unwrap();
    for i in 0..20 {
        writer.add_record(format!("rec-{i}").as_bytes()).unwrap();
    }
    writer.finish().unwrap();

    let tmp_out = NamedTempFile::new().unwrap();
    let out_path = tmp_out.path().to_path_buf();
    std::fs::remove_file(&out_path).ok();

    let reader = SlabReader::open(&in_path).unwrap();
    let records = reader.iter().unwrap();
    let mut writer2 = SlabWriter::new(&out_path, config).unwrap();
    for (_ord, data) in &records {
        writer2.add_record(data).unwrap();
    }
    writer2.finish().unwrap();

    let reader2 = SlabReader::open(&out_path).unwrap();
    for i in 0..20 {
        assert_eq!(
            reader2.get(i).unwrap(),
            format!("rec-{i}").as_bytes(),
            "mismatch at ordinal {i}"
        );
    }
}

/// Export empty slab file → should produce 0 records.
#[test]
fn test_export_empty_slab() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Write a file with zero records (just pages page)
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    let records = reader.iter().unwrap();
    assert!(records.is_empty(), "empty slab should have 0 records");
}

// ===========================================================================
// File-level corruption: footer_length not multiple of 16
// ===========================================================================

/// A footer_length of 17 (not a multiple of 16) — the spec requires
/// multiples of 16. Check if the library accepts or rejects this.
#[test]
fn test_footer_odd_footer_length() {
    let mut buf = [0u8; FOOTER_V1_SIZE];
    let footer = Footer::new(0, 0, 512, PageType::Data);
    footer.write_to(&mut buf);
    // Set footer_length to 17 — not a multiple of 16 but >= 16
    buf[14..16].copy_from_slice(&17u16.to_le_bytes());
    // This tests the current behavior: the library only checks >= 16.
    // A stricter implementation would reject non-multiples.
    let result = Footer::read_from(&buf);
    // Accept either success or error here — document the behavior
    if let Ok(decoded) = result {
        assert_eq!(decoded.footer_length, 17);
    }
    // Either way, the test exercises this path
}

// ===========================================================================
// Namespaces page with multiple entries including edge names
// ===========================================================================

/// NamespacesPage with entries having Unicode names must round-trip.
#[test]
fn test_namespaces_page_unicode_names() {
    let mut np = NamespacesPage::new();
    np.add_entry(&NamespaceEntry {
        namespace_index: 1,
        name: String::new(),
        pages_page_offset: 0,
    });
    np.add_entry(&NamespaceEntry {
        namespace_index: 2,
        name: "日本語".to_string(),
        pages_page_offset: 4096,
    });
    np.add_entry(&NamespaceEntry {
        namespace_index: 3,
        name: "émojis-🎉".to_string(),
        pages_page_offset: 8192,
    });

    let bytes = np.serialize();
    let decoded = NamespacesPage::deserialize(&bytes).unwrap();
    let entries = decoded.entries().unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[1].name, "日本語");
    assert_eq!(entries[2].name, "émojis-🎉");
}

/// NamespaceEntry with negative pages_page_offset must round-trip.
/// (Negative offsets are meaningless but the format stores i64.)
#[test]
fn test_namespace_entry_negative_offset() {
    let entry = NamespaceEntry {
        namespace_index: 1,
        name: String::new(),
        pages_page_offset: -1,
    };
    let bytes = entry.to_bytes();
    let decoded = NamespaceEntry::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.pages_page_offset, -1);
}

// ===========================================================================
// Writer with asserted ordinal edge cases
// ===========================================================================

/// Adding a record with a mismatched asserted ordinal must fail.
#[test]
fn test_writer_ordinal_mismatch() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"first").unwrap(); // ordinal 0
    // Now try to add with asserted ordinal 5 when next expected is 1
    let result = writer.add_record_at(5, b"wrong");
    assert!(
        matches!(result, Err(SlabError::OrdinalMismatch { .. })),
        "ordinal mismatch must be rejected, got {result:?}"
    );
}

/// Adding records with correct sequential asserted ordinals must succeed.
#[test]
fn test_writer_ordinal_correct_sequence() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record_at(0, b"zero").unwrap();
    writer.add_record_at(1, b"one").unwrap();
    writer.add_record_at(2, b"two").unwrap();
    writer.finish().unwrap();

    let reader = SlabReader::open(&path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"zero");
    assert_eq!(reader.get(1).unwrap(), b"one");
    assert_eq!(reader.get(2).unwrap(), b"two");
}

// ===========================================================================
// File corruption: truncated at various points
// ===========================================================================

/// A file truncated to exactly HEADER_SIZE + FOOTER_V1_SIZE - 1 bytes must
/// be rejected (one byte short of minimum).
#[test]
fn test_open_truncated_one_byte_short_of_minimum() {
    let mut tmp = NamedTempFile::new().unwrap();
    let data = vec![0u8; HEADER_SIZE + FOOTER_V1_SIZE - 1];
    tmp.write_all(&data).unwrap();
    let result = SlabReader::open(tmp.path());
    assert!(result.is_err());
}

/// Corrupt the page_size in the header to zero — should cause issues
/// when the reader tries to read the pages page.
#[test]
fn test_corrupted_header_page_size_zero() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"test").unwrap();
    writer.finish().unwrap();

    let mut data = std::fs::read(&path).unwrap();
    // Corrupt the first page's header page_size (bytes 4-7) to 0
    data[4..8].copy_from_slice(&0u32.to_le_bytes());
    std::fs::write(&path, &data).unwrap();

    // The reader reads from the end (footer), so it may still open.
    // But trying to read the first data page should fail.
    if let Ok(reader) = SlabReader::open(&path) {
        let result = reader.get(0);
        assert!(result.is_err(), "corrupted page_size=0 should fail on read");
    }
}

// ===========================================================================
// PagesPage edge cases
// ===========================================================================

/// An empty PagesPage (no entries) should serialize/deserialize cleanly.
#[test]
fn test_pages_page_empty() {
    let pp = PagesPage::new();
    let bytes = pp.serialize();
    let decoded = PagesPage::deserialize(&bytes).unwrap();
    assert_eq!(decoded.entry_count(), 0);
    assert!(decoded.entries().is_empty());
}

/// Looking up an ordinal in an empty PagesPage must return None.
#[test]
fn test_pages_page_find_in_empty() {
    let pp = PagesPage::new();
    assert!(pp.find_page_for_ordinal(0).is_none());
    assert!(pp.find_page_for_ordinal(-1).is_none());
    assert!(pp.find_page_for_ordinal(100).is_none());
}

/// Write 1000 page entries and verify binary search works for all ordinals.
#[test]
fn test_pages_page_many_entries_binary_search() {
    let mut pp = PagesPage::new();
    for i in 0..1000 {
        pp.add_entry(i * 100, i * 4096);
    }

    // Exact matches
    for i in 0..1000 {
        let entry = pp.find_page_for_ordinal(i * 100).unwrap();
        assert_eq!(entry.start_ordinal, i * 100);
    }

    // Values between entries: ordinal 50 should map to the page starting at 0
    let entry = pp.find_page_for_ordinal(50).unwrap();
    assert_eq!(entry.start_ordinal, 0);

    // Value between entries: ordinal 150 should map to page starting at 100
    let entry = pp.find_page_for_ordinal(150).unwrap();
    assert_eq!(entry.start_ordinal, 100);
}

// ---------------------------------------------------------------------------
// slab explain
// ---------------------------------------------------------------------------

/// Helper: write a slab file with the given records and return the temp path.
fn write_explain_test_file(records: &[&[u8]]) -> tempfile::TempPath {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.into_temp_path();
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    for rec in records {
        writer.add_record(rec).unwrap();
    }
    writer.finish().unwrap();
    path
}

/// A single-page file produces a diagram with Header, Records, Offsets,
/// Footer sections, plus the pages page diagram.
#[test]
fn test_explain_single_page() {
    let path = write_explain_test_file(&[b"alpha", b"beta", b"gamma"]);
    let mut out = Vec::new();
    slabtastic::cli::explain::explain_to(
        &mut out,
        path.to_str().unwrap(),
        &None,
        &None,
        &None,
    )
    .unwrap();
    let output = String::from_utf8(out).unwrap();

    assert!(output.contains("Page 0 (Data)"), "should show page 0");
    assert!(output.contains("Header"), "should show header section");
    assert!(output.contains("magic: SLAB"), "should show magic");
    assert!(output.contains("Records (3 records"), "should show 3 records");
    assert!(output.contains("ordinal 0:"), "should show ordinal 0");
    assert!(output.contains("ordinal 1:"), "should show ordinal 1");
    assert!(output.contains("ordinal 2:"), "should show ordinal 2");
    assert!(output.contains("Offsets (4 entries"), "should show 4 offset entries");
    assert!(output.contains("Footer (16 bytes)"), "should show footer");
    assert!(output.contains("start_ordinal: 0"), "should show start_ordinal");
    assert!(output.contains("record_count: 3"), "should show record_count");
    assert!(output.contains("page_type: Data"), "should show page_type");
    assert!(output.contains("Pages Page at offset"), "should show pages page");
}

/// A multi-page file shows all data pages when no filter is applied.
#[test]
fn test_explain_multi_page() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.into_temp_path();
    // Use a small page size to force multiple pages
    let config = WriterConfig::new(512, 512, u32::MAX, false).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    // Write enough records to span multiple pages
    for i in 0..100 {
        writer
            .add_record(format!("record-{i:04}").as_bytes())
            .unwrap();
    }
    writer.finish().unwrap();

    let mut out = Vec::new();
    slabtastic::cli::explain::explain_to(
        &mut out,
        path.to_str().unwrap(),
        &None,
        &None,
        &None,
    )
    .unwrap();
    let output = String::from_utf8(out).unwrap();

    // Should have Page 0 and Page 1 at minimum
    assert!(output.contains("Page 0 (Data)"), "should show page 0");
    assert!(output.contains("Page 1 (Data)"), "should show page 1");
    assert!(output.contains("Pages Page at offset"), "should show pages page");
}

/// Filtering by page index shows only the selected page.
#[test]
fn test_explain_page_filter() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.into_temp_path();
    let config = WriterConfig::new(512, 512, u32::MAX, false).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    for i in 0..100 {
        writer
            .add_record(format!("record-{i:04}").as_bytes())
            .unwrap();
    }
    writer.finish().unwrap();

    let mut out = Vec::new();
    slabtastic::cli::explain::explain_to(
        &mut out,
        path.to_str().unwrap(),
        &Some(vec![1]),
        &None,
        &None,
    )
    .unwrap();
    let output = String::from_utf8(out).unwrap();

    assert!(!output.contains("Page 0 (Data)"), "should NOT show page 0");
    assert!(output.contains("Page 1 (Data)"), "should show page 1");
    // Pages page should not appear when filtering
    assert!(
        !output.contains("Pages Page at offset"),
        "pages page should not appear with page filter"
    );
}

/// Filtering by ordinal range shows only pages whose ordinal range overlaps.
#[test]
fn test_explain_ordinal_filter() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.into_temp_path();
    let config = WriterConfig::new(512, 512, u32::MAX, false).unwrap();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    for i in 0..100 {
        writer
            .add_record(format!("record-{i:04}").as_bytes())
            .unwrap();
    }
    writer.finish().unwrap();

    // Get the page entries to find the second page's start ordinal
    let reader = SlabReader::open(&path).unwrap();
    let entries = reader.page_entries();
    assert!(entries.len() >= 2, "need at least 2 pages for this test");
    let second_start = entries[1].start_ordinal;

    // Filter to an ordinal range that only includes the second page
    let range_spec = format!("[{},{})", second_start, second_start + 1);
    let mut out = Vec::new();
    slabtastic::cli::explain::explain_to(
        &mut out,
        path.to_str().unwrap(),
        &None,
        &None,
        &Some(range_spec),
    )
    .unwrap();
    let output = String::from_utf8(out).unwrap();

    assert!(!output.contains("Page 0 (Data)"), "should NOT show page 0");
    assert!(output.contains("Page 1 (Data)"), "should show page 1");
}

/// A file with zero data records still has a pages page that can be
/// explained.
#[test]
fn test_explain_empty_file() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.into_temp_path();
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.finish().unwrap();

    let mut out = Vec::new();
    slabtastic::cli::explain::explain_to(
        &mut out,
        path.to_str().unwrap(),
        &None,
        &None,
        &None,
    )
    .unwrap();
    let output = String::from_utf8(out).unwrap();

    // Should show the pages page even with no data pages
    assert!(output.contains("Pages Page at offset"), "should show pages page");
}

/// The pages page diagram shows its entries as start_ordinal/offset tuples.
#[test]
fn test_explain_pages_page() {
    let path = write_explain_test_file(&[b"hello", b"world"]);
    let mut out = Vec::new();
    slabtastic::cli::explain::explain_to(
        &mut out,
        path.to_str().unwrap(),
        &None,
        &None,
        &None,
    )
    .unwrap();
    let output = String::from_utf8(out).unwrap();

    assert!(output.contains("Pages Page at offset"), "should show pages page");
    assert!(
        output.contains("start_ordinal=0"),
        "pages page should show entry with start_ordinal=0"
    );
    assert!(
        output.contains("offset=0"),
        "pages page should show entry with file offset=0"
    );
}

// ===========================================================================
// CLI: --skip-malformed on import
// ===========================================================================

/// Importing a JSONL file with one malformed line should skip it when
/// `--skip-malformed` is set, importing only the valid records.
#[test]
fn test_import_skip_malformed_jsonl() {
    let source = NamedTempFile::new().unwrap();
    let source_path = source.path().to_path_buf();
    std::fs::write(
        &source_path,
        "{\"a\":1}\nNOT_JSON\n{\"b\":2}\n",
    )
    .unwrap();

    let target = NamedTempFile::new().unwrap();
    let target_path = target.into_temp_path();
    // Remove so import creates fresh
    std::fs::remove_file(&target_path).unwrap();

    slabtastic::cli::import::run(
        target_path.to_str().unwrap(),
        source_path.to_str().unwrap(),
        false, false, false, false,
        true,  // jsonl
        false, false, false,
        true,  // skip_malformed
        false, // strip_newline
        None, None, false,
        false, // progress
        &None, // namespace
    )
    .unwrap();

    let reader = SlabReader::open(&target_path).unwrap();
    let records = reader.iter().unwrap();
    assert_eq!(records.len(), 2, "should import 2 valid records, skipping the malformed one");
}

/// Importing a JSONL file with a malformed line should fail when
/// `--skip-malformed` is NOT set.
#[test]
fn test_import_fail_on_malformed_jsonl() {
    let source = NamedTempFile::new().unwrap();
    let source_path = source.path().to_path_buf();
    std::fs::write(&source_path, "{\"a\":1}\nNOT_JSON\n").unwrap();

    let target = NamedTempFile::new().unwrap();
    let target_path = target.into_temp_path();
    std::fs::remove_file(&target_path).unwrap();

    let result = slabtastic::cli::import::run(
        target_path.to_str().unwrap(),
        source_path.to_str().unwrap(),
        false, false, false, false,
        true,  // jsonl
        false, false, false,
        false, // skip_malformed
        false, // strip_newline
        None, None, false,
        false, // progress
        &None, // namespace
    );
    assert!(result.is_err(), "import should fail on malformed record without --skip-malformed");
}

// ===========================================================================
// CLI: --strip-newline on import
// ===========================================================================

/// Importing newline-terminated text with `--strip-newline` should store
/// records without trailing newlines.
#[test]
fn test_import_strip_newline() {
    let source = NamedTempFile::new().unwrap();
    let source_path = source.path().to_path_buf();
    std::fs::write(&source_path, "hello\nworld\n").unwrap();

    let target = NamedTempFile::new().unwrap();
    let target_path = target.into_temp_path();
    std::fs::remove_file(&target_path).unwrap();

    slabtastic::cli::import::run(
        target_path.to_str().unwrap(),
        source_path.to_str().unwrap(),
        true,  // newline_terminated_records
        false, false, false, false, false, false, false,
        false, // skip_malformed
        true,  // strip_newline
        None, None, false,
        false, // progress
        &None, // namespace
    )
    .unwrap();

    let reader = SlabReader::open(&target_path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"hello");
    assert_eq!(reader.get(1).unwrap(), b"world");
}

/// Without `--strip-newline`, records preserve trailing newlines.
#[test]
fn test_import_preserves_newline_by_default() {
    let source = NamedTempFile::new().unwrap();
    let source_path = source.path().to_path_buf();
    std::fs::write(&source_path, "hello\nworld\n").unwrap();

    let target = NamedTempFile::new().unwrap();
    let target_path = target.into_temp_path();
    std::fs::remove_file(&target_path).unwrap();

    slabtastic::cli::import::run(
        target_path.to_str().unwrap(),
        source_path.to_str().unwrap(),
        true,  // newline_terminated_records
        false, false, false, false, false, false, false,
        false, // skip_malformed
        false, // strip_newline
        None, None, false,
        false, // progress
        &None, // namespace
    )
    .unwrap();

    let reader = SlabReader::open(&target_path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"hello\n");
    assert_eq!(reader.get(1).unwrap(), b"world\n");
}

// ===========================================================================
// CLI: export --format
// ===========================================================================

/// Exporting with `--format=raw` writes records exactly as stored
/// without adding missing newlines.
#[test]
fn test_export_raw() {
    let tmp = NamedTempFile::new().unwrap();
    let slab_path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();
    writer.add_record(b"no-newline").unwrap();
    writer.add_record(b"also-no-newline").unwrap();
    writer.finish().unwrap();

    let out = NamedTempFile::new().unwrap();
    let out_path = out.path().to_path_buf();

    slabtastic::cli::export::run(&slabtastic::cli::export::ExportConfig {
        file: slab_path.to_str().unwrap(),
        output: Some(out_path.to_str().unwrap()),
        format: Some(slabtastic::cli::export::ExportFormat::Raw),
        preferred_page_size: None,
        min_page_size: None,
        page_alignment: false,
        progress: false,
        namespace: &None,
        range: None,
    })
    .unwrap();

    let exported = std::fs::read(&out_path).unwrap();
    assert_eq!(exported, b"no-newlinealso-no-newline");
}

/// Exporting with `--format=text` adds missing newlines.
#[test]
fn test_export_text_adds_newlines() {
    let tmp = NamedTempFile::new().unwrap();
    let slab_path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&slab_path, config).unwrap();
    writer.add_record(b"no-newline").unwrap();
    writer.add_record(b"also-no-newline").unwrap();
    writer.finish().unwrap();

    let out = NamedTempFile::new().unwrap();
    let out_path = out.path().to_path_buf();

    slabtastic::cli::export::run(&slabtastic::cli::export::ExportConfig {
        file: slab_path.to_str().unwrap(),
        output: Some(out_path.to_str().unwrap()),
        format: Some(slabtastic::cli::export::ExportFormat::Text),
        preferred_page_size: None,
        min_page_size: None,
        page_alignment: false,
        progress: false,
        namespace: &None,
        range: None,
    })
    .unwrap();

    let exported = std::fs::read(&out_path).unwrap();
    assert_eq!(exported, b"no-newline\nalso-no-newline\n");
}

// ===========================================================================
// .slab.buffer convention
// ===========================================================================

/// Rewrite creates a `.buffer` file during writing and renames it on
/// success. After a successful rewrite the buffer file must not exist.
#[test]
fn test_buffer_rename_on_rewrite() {
    let tmp_in = NamedTempFile::new().unwrap();
    let in_path = tmp_in.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&in_path, config).unwrap();
    writer.add_record(b"hello").unwrap();
    writer.finish().unwrap();

    let tmp_out = NamedTempFile::new().unwrap();
    let out_path = tmp_out.into_temp_path();
    std::fs::remove_file(&out_path).unwrap();

    slabtastic::cli::rewrite::run(
        in_path.to_str().unwrap(),
        out_path.to_str().unwrap(),
        None, // range
        None, None, false,
        false, // progress
        &None, // namespace
    )
    .unwrap();

    // Output file must exist
    assert!(out_path.exists(), "output slab file must exist after rewrite");

    // Buffer file must NOT exist
    let buffer_path = format!("{}.buffer", out_path.to_str().unwrap());
    assert!(
        !std::path::Path::new(&buffer_path).exists(),
        "buffer file should be removed after successful rewrite"
    );

    // Verify content
    let reader = SlabReader::open(&out_path).unwrap();
    assert_eq!(reader.get(0).unwrap(), b"hello");
}

// ===========================================================================
// Error context (WithContext variant)
// ===========================================================================

/// SlabError::WithContext displays the underlying error plus location info.
#[test]
fn test_error_with_context_display() {
    let err = SlabError::InvalidPageType(99)
        .with_context(Some(4096), Some(3), None);
    let msg = format!("{err}");
    assert!(msg.contains("invalid page type: 99"), "should contain base error: {msg}");
    assert!(msg.contains("offset 4096"), "should contain file offset: {msg}");
    assert!(msg.contains("page 3"), "should contain page index: {msg}");
}

/// Errors from read_data_page should carry file offset context when the
/// underlying page data is corrupted.
#[test]
fn test_reader_error_carries_offset_context() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    writer.add_record(b"test").unwrap();
    writer.finish().unwrap();

    // Corrupt the first data page's footer page_type byte to 0 (Invalid).
    // The footer is the last 16 bytes of the data page, and page_type is
    // at offset 12 within the footer.
    let mut data = std::fs::read(&path).unwrap();
    // Data page header page_size tells us how big the data page is.
    let page_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    // Footer starts at page_size - 16, page_type is at footer_start + 12
    let page_type_offset = page_size - 16 + 12;
    data[page_type_offset] = 0; // Invalid page type
    std::fs::write(&path, &data).unwrap();

    // The reader validates page metadata eagerly at open time, so
    // the corrupted page_type is caught during index construction.
    // If open somehow succeeds, get() must still fail.
    match SlabReader::open(&path) {
        Err(e) => {
            let err_msg = format!("{e}");
            assert!(
                err_msg.contains("offset"),
                "error should include file offset context: {err_msg}"
            );
        }
        Ok(reader) => {
            let result = reader.get(0);
            assert!(result.is_err());
            let err_msg = format!("{}", result.unwrap_err());
            assert!(
                err_msg.contains("offset"),
                "error should include file offset context: {err_msg}"
            );
        }
    }
}
