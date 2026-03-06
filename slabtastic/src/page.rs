// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Page structure: the fundamental unit of a slabtastic file.
//!
//! Each page is a self-contained blob with header, packed record data,
//! an offset array, and a footer. Pages can be serialized and
//! deserialized independently.
//!
//! ## Record offset calculation
//!
//! From the end of the page: back up `footer_length` bytes to reach the
//! end of the offset array, then back up `4 * (record_count + 1)` bytes
//! to reach the first offset. Each offset is a little-endian `u32`
//! measured from the **start of the page**, so the first record always
//! begins at byte 8 (after the 8-byte header).
//!
//! ## Size constraints
//!
//! Page sizes range from 2^9 (512 bytes) to 2^32 bytes. A single record
//! that would exceed the maximum page size is an error in v1 — see
//! [`SlabError::RecordTooLarge`].
//!
//! ## Bidirectional traversal
//!
//! Both header and footer carry the page size, so a file can be
//! traversed **forward** (header → next page) and **backward** (footer →
//! previous page) without consulting the pages page index.

use crate::constants::{FOOTER_V1_SIZE, HEADER_SIZE, MAGIC, PageType};
use crate::error::{Result, SlabError};
use crate::footer::Footer;

/// A slabtastic page containing a header, records, offset array, and footer.
///
/// ## Wire layout
///
/// ```text
/// [magic:4 "SLAB"][page_size:4][record data ...][offsets:(n+1)*4][footer:16]
/// ```
///
/// The **offset array** contains `(record_count + 1)` little-endian `u32`
/// values. Each value is a byte position measured from the start of the
/// page, so the first record always starts at offset 8 (after the 8-byte
/// header). The extra sentinel offset marks the end of the last record.
///
/// ## Examples
///
/// ```
/// use slabtastic::{Page, PageType};
///
/// let mut page = Page::new(0, PageType::Data);
/// page.add_record(b"hello");
/// page.add_record(b"world");
///
/// let bytes = page.serialize();
/// let decoded = Page::deserialize(&bytes).unwrap();
/// assert_eq!(decoded.record_count(), 2);
/// assert_eq!(decoded.get_record(0).unwrap(), b"hello");
/// ```
#[derive(Debug, Clone)]
pub struct Page {
    /// The page footer (metadata).
    pub footer: Footer,
    /// The raw record data blobs.
    pub records: Vec<Vec<u8>>,
    /// Running total of record byte lengths (maintained incrementally to
    /// avoid re-summing on every `serialized_size()` call).
    record_data_len: usize,
}

impl Page {
    /// Create a new empty page.
    pub fn new(start_ordinal: i64, page_type: PageType) -> Self {
        Page {
            footer: Footer::new(start_ordinal, 0, 0, page_type),
            records: Vec::new(),
            record_data_len: 0,
        }
    }

    /// Append a record to this page.
    pub fn add_record(&mut self, data: &[u8]) {
        self.record_data_len += data.len();
        self.records.push(data.to_vec());
        self.footer.record_count = self.records.len() as u32;
    }

    /// Compute the total serialized size of this page.
    pub fn serialized_size(&self) -> usize {
        let offset_count = self.records.len() + 1;
        HEADER_SIZE + self.record_data_len + (offset_count * 4) + FOOTER_V1_SIZE
    }

    /// Serialize this page to a byte vector.
    ///
    /// Layout: `[magic:4][page_size:4][records...][offsets:(n+1)*4][footer:16]`
    pub fn serialize(&self) -> Vec<u8> {
        let total_size = self.serialized_size();
        let mut buf = Vec::with_capacity(total_size);

        // Header: magic + page_size
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&(total_size as u32).to_le_bytes());

        // Records: packed contiguous data
        for record in &self.records {
            buf.extend_from_slice(record);
        }

        // Offsets: (record_count + 1) x 4-byte LE from page start
        let mut offset = HEADER_SIZE as u32;
        for record in &self.records {
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += record.len() as u32;
        }
        // Final sentinel offset (end of last record)
        buf.extend_from_slice(&offset.to_le_bytes());

        // Footer
        let mut footer = self.footer.clone();
        footer.page_size = total_size as u32;
        footer.record_count = self.records.len() as u32;
        let mut footer_buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut footer_buf);
        buf.extend_from_slice(&footer_buf);

        debug_assert_eq!(buf.len(), total_size);
        buf
    }

    /// Deserialize a page from a byte buffer.
    pub fn deserialize(buf: &[u8]) -> Result<Page> {
        if buf.len() < HEADER_SIZE + FOOTER_V1_SIZE {
            return Err(SlabError::TruncatedPage {
                expected: HEADER_SIZE + FOOTER_V1_SIZE,
                actual: buf.len(),
            });
        }

        // Validate magic
        if buf[0..4] != MAGIC {
            return Err(SlabError::InvalidMagic);
        }

        // Read header page_size
        let header_page_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);

        // Read footer from last 16 bytes
        let footer_start = buf.len() - FOOTER_V1_SIZE;
        let footer = Footer::read_from(&buf[footer_start..])?;

        // Verify header and footer page sizes match
        if header_page_size != footer.page_size {
            return Err(SlabError::PageSizeMismatch {
                header: header_page_size,
                footer: footer.page_size,
            });
        }

        if buf.len() < footer.page_size as usize {
            return Err(SlabError::TruncatedPage {
                expected: footer.page_size as usize,
                actual: buf.len(),
            });
        }

        let record_count = footer.record_count as usize;
        let offset_count = record_count + 1;

        // Offsets sit immediately before the footer
        let offsets_size = offset_count * 4;
        let offsets_start = footer_start - offsets_size;

        // Read offsets
        let mut offsets = Vec::with_capacity(offset_count);
        for i in 0..offset_count {
            let pos = offsets_start + i * 4;
            let o = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
            offsets.push(o as usize);
        }

        // Extract records
        let mut records = Vec::with_capacity(record_count);
        for i in 0..record_count {
            let start = offsets[i];
            let end = offsets[i + 1];
            records.push(buf[start..end].to_vec());
        }

        let record_data_len: usize = records.iter().map(|r| r.len()).sum();
        Ok(Page {
            footer: Footer::new(
                footer.start_ordinal,
                footer.record_count,
                footer.page_size,
                footer.page_type,
            ),
            records,
            record_data_len,
        })
    }

    /// Get a record by its local (zero-based) index.
    pub fn get_record(&self, index: usize) -> Option<&[u8]> {
        self.records.get(index).map(|r| r.as_slice())
    }

    /// Return the number of records in this page.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Return the starting ordinal for this page.
    pub fn start_ordinal(&self) -> i64 {
        self.footer.start_ordinal
    }

    /// Read the record count from a serialized page buffer by inspecting
    /// only the footer bytes. No heap allocation.
    ///
    /// The record count is stored as a 3-byte unsigned LE integer at
    /// footer offset 5–7 (bytes `[len-11..len-8]` of the page buffer).
    ///
    /// ## Errors
    ///
    /// - [`SlabError::TruncatedPage`] if the buffer is too small for a
    ///   header + footer.
    pub fn record_count_from_buf(buf: &[u8]) -> Result<usize> {
        if buf.len() < HEADER_SIZE + FOOTER_V1_SIZE {
            return Err(SlabError::TruncatedPage {
                expected: HEADER_SIZE + FOOTER_V1_SIZE,
                actual: buf.len(),
            });
        }
        let footer_start = buf.len() - FOOTER_V1_SIZE;
        let mut rc_bytes = [0u8; 4];
        rc_bytes[0..3].copy_from_slice(&buf[footer_start + 5..footer_start + 8]);
        Ok(u32::from_le_bytes(rc_bytes) as usize)
    }

    /// Return a borrowed slice of a single record from a serialized page
    /// buffer without deserializing the entire page.
    ///
    /// Like [`get_record_from_buf`](Self::get_record_from_buf) but returns
    /// `&[u8]` instead of `Vec<u8>`, eliminating the per-record heap
    /// allocation. The `record_count` parameter avoids re-parsing the
    /// footer on every call when iterating all records in a page.
    ///
    /// Magic validation is intentionally skipped since callers are
    /// expected to have obtained the buffer through a validated path.
    ///
    /// ## Errors
    ///
    /// - [`SlabError::TruncatedPage`] if the buffer is too small.
    /// - [`SlabError::OrdinalNotFound`] if `local_index` is out of range.
    pub fn get_record_ref_from_buf(
        buf: &[u8], local_index: usize, record_count: usize,
    ) -> Result<&[u8]> {
        if buf.len() < HEADER_SIZE + FOOTER_V1_SIZE {
            return Err(SlabError::TruncatedPage {
                expected: HEADER_SIZE + FOOTER_V1_SIZE,
                actual: buf.len(),
            });
        }

        if local_index >= record_count {
            return Err(SlabError::OrdinalNotFound(-1));
        }

        let footer_start = buf.len() - FOOTER_V1_SIZE;
        let offset_count = record_count + 1;
        let offsets_size = offset_count * 4;
        let offsets_start = footer_start - offsets_size;

        let pos_a = offsets_start + local_index * 4;
        let pos_b = pos_a + 4;
        let start = u32::from_le_bytes([buf[pos_a], buf[pos_a + 1], buf[pos_a + 2], buf[pos_a + 3]]) as usize;
        let end = u32::from_le_bytes([buf[pos_b], buf[pos_b + 1], buf[pos_b + 2], buf[pos_b + 3]]) as usize;

        Ok(&buf[start..end])
    }

    /// Extract a single record from a serialized page buffer without
    /// deserializing the entire page.
    ///
    /// This reads only the footer and the two offset entries needed to
    /// locate the record at `local_index`, then copies just that
    /// record's bytes. No `Vec<Vec<u8>>` allocation for all records and
    /// no parsing of all N+1 offsets.
    ///
    /// Magic validation is intentionally skipped since callers are
    /// expected to have obtained the buffer through a validated path.
    ///
    /// Note: [`SlabReader::get`](crate::SlabReader::get) performs
    /// targeted I/O directly (reading only geometry + record bytes)
    /// rather than buffering the full page. This method is useful when
    /// a full page buffer is already in memory for other reasons.
    ///
    /// ## Errors
    ///
    /// - [`SlabError::TruncatedPage`] if the buffer is too small.
    /// - [`SlabError::OrdinalNotFound`] if `local_index` is out of range
    ///   (uses ordinal −1 as a placeholder since the caller maps ordinals
    ///   to local indices).
    pub fn get_record_from_buf(buf: &[u8], local_index: usize) -> Result<Vec<u8>> {
        if buf.len() < HEADER_SIZE + FOOTER_V1_SIZE {
            return Err(SlabError::TruncatedPage {
                expected: HEADER_SIZE + FOOTER_V1_SIZE,
                actual: buf.len(),
            });
        }

        // Read footer from last 16 bytes
        let footer_start = buf.len() - FOOTER_V1_SIZE;
        let footer = Footer::read_from(&buf[footer_start..])?;

        let record_count = footer.record_count as usize;
        if local_index >= record_count {
            return Err(SlabError::OrdinalNotFound(-1));
        }

        // Offset array sits immediately before the footer.
        // We only need offsets[local_index] and offsets[local_index + 1].
        let offset_count = record_count + 1;
        let offsets_size = offset_count * 4;
        let offsets_start = footer_start - offsets_size;

        let pos_a = offsets_start + local_index * 4;
        let pos_b = pos_a + 4;
        let start = u32::from_le_bytes([buf[pos_a], buf[pos_a + 1], buf[pos_a + 2], buf[pos_a + 3]]) as usize;
        let end = u32::from_le_bytes([buf[pos_b], buf[pos_b + 1], buf[pos_b + 2], buf[pos_b + 3]]) as usize;

        Ok(buf[start..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty page (zero records) must serialize and deserialize
    /// to a page with record_count 0 and the original start_ordinal.
    #[test]
    fn test_page_empty_roundtrip() {
        let page = Page::new(0, PageType::Data);
        let bytes = page.serialize();
        let decoded = Page::deserialize(&bytes).unwrap();
        assert_eq!(decoded.record_count(), 0);
        assert_eq!(decoded.start_ordinal(), 0);
    }

    /// A page with a single record at a non-zero start ordinal must
    /// round-trip with the correct data and ordinal preserved.
    #[test]
    fn test_page_single_record() {
        let mut page = Page::new(100, PageType::Data);
        page.add_record(b"hello world");
        let bytes = page.serialize();
        let decoded = Page::deserialize(&bytes).unwrap();
        assert_eq!(decoded.record_count(), 1);
        assert_eq!(decoded.get_record(0).unwrap(), b"hello world");
        assert_eq!(decoded.start_ordinal(), 100);
    }

    /// Three records of different lengths must round-trip in order.
    /// Verifies the offset array correctly delimits each record.
    #[test]
    fn test_page_multiple_records() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"alpha");
        page.add_record(b"beta");
        page.add_record(b"gamma");
        let bytes = page.serialize();
        let decoded = Page::deserialize(&bytes).unwrap();
        assert_eq!(decoded.record_count(), 3);
        assert_eq!(decoded.get_record(0).unwrap(), b"alpha");
        assert_eq!(decoded.get_record(1).unwrap(), b"beta");
        assert_eq!(decoded.get_record(2).unwrap(), b"gamma");
    }

    /// Corrupting the first byte of the magic ("SLAB" → "XLAB") must
    /// cause `deserialize` to fail with `InvalidMagic`.
    #[test]
    fn test_page_magic_validation() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"test");
        let mut bytes = page.serialize();
        bytes[0] = b'X'; // corrupt magic
        let result = Page::deserialize(&bytes);
        assert!(result.is_err());
    }

    /// Verify that `serialized_size()` matches the actual length of the
    /// serialized buffer. For one 5-byte record the expected size is:
    /// header(8) + data(5) + offsets(2×4=8) + footer(16) = 37.
    #[test]
    fn test_page_serialized_size() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"12345");
        // header(8) + data(5) + offsets(2*4=8) + footer(16) = 37
        assert_eq!(page.serialized_size(), 37);
        let bytes = page.serialize();
        assert_eq!(bytes.len(), 37);
    }

    /// `get_record_from_buf` extracts a single record from a serialized
    /// page without deserializing all records.
    #[test]
    fn test_get_record_from_buf_single() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"only-record");
        let bytes = page.serialize();
        let record = Page::get_record_from_buf(&bytes, 0).unwrap();
        assert_eq!(record, b"only-record");
    }

    /// `get_record_from_buf` correctly indexes into a multi-record page.
    #[test]
    fn test_get_record_from_buf_multiple() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"alpha");
        page.add_record(b"beta");
        page.add_record(b"gamma");
        let bytes = page.serialize();

        assert_eq!(Page::get_record_from_buf(&bytes, 0).unwrap(), b"alpha");
        assert_eq!(Page::get_record_from_buf(&bytes, 1).unwrap(), b"beta");
        assert_eq!(Page::get_record_from_buf(&bytes, 2).unwrap(), b"gamma");
    }

    /// `get_record_from_buf` with an out-of-bounds index returns an error.
    #[test]
    fn test_get_record_from_buf_out_of_bounds() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"one");
        let bytes = page.serialize();
        assert!(Page::get_record_from_buf(&bytes, 1).is_err());
        assert!(Page::get_record_from_buf(&bytes, 100).is_err());
    }

    /// `get_record_from_buf` still works when magic bytes are corrupted
    /// since it intentionally skips magic validation for performance.
    #[test]
    fn test_get_record_from_buf_skips_magic_validation() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"test");
        let mut bytes = page.serialize();
        bytes[0] = b'X';
        // Should still extract the record — magic is not checked
        assert_eq!(Page::get_record_from_buf(&bytes, 0).unwrap(), b"test");
    }

    /// `get_record_ref_from_buf` returns borrowed slices without copying.
    #[test]
    fn test_get_record_ref_from_buf_multiple() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"alpha");
        page.add_record(b"beta");
        page.add_record(b"gamma");
        let bytes = page.serialize();

        assert_eq!(Page::get_record_ref_from_buf(&bytes, 0, 3).unwrap(), b"alpha");
        assert_eq!(Page::get_record_ref_from_buf(&bytes, 1, 3).unwrap(), b"beta");
        assert_eq!(Page::get_record_ref_from_buf(&bytes, 2, 3).unwrap(), b"gamma");
    }

    /// `get_record_ref_from_buf` with an out-of-bounds index returns an error.
    #[test]
    fn test_get_record_ref_from_buf_out_of_bounds() {
        let mut page = Page::new(0, PageType::Data);
        page.add_record(b"one");
        let bytes = page.serialize();
        assert!(Page::get_record_ref_from_buf(&bytes, 1, 1).is_err());
        assert!(Page::get_record_ref_from_buf(&bytes, 100, 1).is_err());
    }
}
