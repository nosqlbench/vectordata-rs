// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Page footer serialization and deserialization.
//!
//! Every slabtastic page ends with a fixed-size footer that carries all
//! metadata needed to interpret the page without external context.
//!
//! ## Namespace index (byte 13)
//!
//! Byte 13 of the v1 footer is the `namespace_index`. In pre-namespace
//! files this byte was called `version` and was always `1`. Since
//! namespace index 1 is the default namespace `""`, existing files are
//! backward compatible. Index 0 is invalid/reserved, indices 2–127 are
//! user-defined, and negative indices (−128 to −1) are reserved.
//!
//! ## Format versioning
//!
//! Page types implicitly carry their format version. Types 1 (pages
//! page), 2 (data page), and 3 (namespaces page) are all v1-era types.
//! Future revisions will introduce new page type values rather than
//! incrementing a version field.
//!
//! ## v1 capacity limits
//!
//! The 16-byte v1 footer encodes the start ordinal in 5 bytes (range
//! ±2^39 ≈ ±549 billion) and the record count in 3 bytes (max 2^24 − 1
//! = 16,777,215 records per page).

use crate::constants::{DEFAULT_NAMESPACE_INDEX, FOOTER_V1_SIZE, PageType};
use crate::error::{Result, SlabError};

/// A 16-byte page footer (v1 layout, little-endian).
///
/// ## Wire format
///
/// ```text
/// Byte   Field              Width   Encoding
/// 0–4    start_ordinal      5       signed LE (±2^39 range)
/// 5–7    record_count       3       unsigned LE (max 2^24 − 1 = 16,777,215)
/// 8–11   page_size          4       unsigned LE (512 .. 2^32)
/// 12     page_type          1       enum (0=Invalid, 1=Pages, 2=Data, 3=Namespaces)
/// 13     namespace_index    1       signed (0=invalid, 1=default "", 2–127=user)
/// 14–15  footer_length      2       unsigned LE (≥ 16, multiple of 16)
/// ```
///
/// ## Examples
///
/// ```
/// use slabtastic::Footer;
/// use slabtastic::PageType;
/// use slabtastic::constants::FOOTER_V1_SIZE;
///
/// let footer = Footer::new(42, 10, 4096, PageType::Data);
/// let mut buf = [0u8; FOOTER_V1_SIZE];
/// footer.write_to(&mut buf);
/// let decoded = Footer::read_from(&buf).unwrap();
/// assert_eq!(footer, decoded);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Footer {
    /// Starting ordinal for this page (5-byte signed, range ±2^39).
    pub start_ordinal: i64,
    /// Number of records in this page (3-byte unsigned, max 2^24-1).
    pub record_count: u32,
    /// Total page size in bytes.
    pub page_size: u32,
    /// Discriminator for the page type.
    pub page_type: PageType,
    /// Namespace index (0=invalid, 1=default `""`, 2–127=user-defined).
    pub namespace_index: u8,
    /// Length of the footer in bytes (>= 16, multiple of 16).
    pub footer_length: u16,
}

impl Footer {
    /// Create a new v1 footer with the given parameters and the default
    /// namespace index (`1`).
    pub fn new(start_ordinal: i64, record_count: u32, page_size: u32, page_type: PageType) -> Self {
        Footer {
            start_ordinal,
            record_count,
            page_size,
            page_type,
            namespace_index: DEFAULT_NAMESPACE_INDEX,
            footer_length: FOOTER_V1_SIZE as u16,
        }
    }

    /// Serialize this footer into exactly 16 bytes (little-endian).
    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(
            buf.len() >= FOOTER_V1_SIZE,
            "buffer too small for footer: {} < {FOOTER_V1_SIZE}",
            buf.len()
        );

        // start_ordinal: 5 bytes LE (signed, mask to 5 bytes)
        let ord_bytes = self.start_ordinal.to_le_bytes();
        buf[0..5].copy_from_slice(&ord_bytes[0..5]);

        // record_count: 3 bytes LE
        let rc_bytes = self.record_count.to_le_bytes();
        buf[5..8].copy_from_slice(&rc_bytes[0..3]);

        // page_size: 4 bytes LE
        buf[8..12].copy_from_slice(&self.page_size.to_le_bytes());

        // page_type: 1 byte
        buf[12] = self.page_type as u8;

        // namespace_index: 1 byte
        buf[13] = self.namespace_index;

        // footer_length: 2 bytes LE
        buf[14..16].copy_from_slice(&self.footer_length.to_le_bytes());
    }

    /// Deserialize a footer from exactly 16 bytes (little-endian).
    pub fn read_from(buf: &[u8]) -> Result<Footer> {
        if buf.len() < FOOTER_V1_SIZE {
            return Err(SlabError::InvalidFooter(format!(
                "buffer too small: {} < {FOOTER_V1_SIZE}",
                buf.len()
            )));
        }

        // start_ordinal: 5 bytes LE, sign-extend to i64
        let mut ord_bytes = [0u8; 8];
        ord_bytes[0..5].copy_from_slice(&buf[0..5]);
        // sign-extend: if bit 39 is set, fill bytes 5..8 with 0xFF
        if ord_bytes[4] & 0x80 != 0 {
            ord_bytes[5] = 0xFF;
            ord_bytes[6] = 0xFF;
            ord_bytes[7] = 0xFF;
        }
        let start_ordinal = i64::from_le_bytes(ord_bytes);

        // record_count: 3 bytes LE
        let mut rc_bytes = [0u8; 4];
        rc_bytes[0..3].copy_from_slice(&buf[5..8]);
        let record_count = u32::from_le_bytes(rc_bytes);

        // page_size: 4 bytes LE
        let page_size = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);

        // page_type: 1 byte
        let page_type_raw = buf[12];
        let page_type =
            PageType::from_u8(page_type_raw).ok_or(SlabError::InvalidPageType(page_type_raw))?;
        if page_type == PageType::Invalid {
            return Err(SlabError::InvalidPageType(page_type_raw));
        }

        // namespace_index: 1 byte (0 = invalid, 1 = default, 2–127 = user, 128–255 = reserved)
        let namespace_index = buf[13];
        if namespace_index == 0 || namespace_index >= 128 {
            return Err(SlabError::InvalidNamespaceIndex(namespace_index));
        }

        // footer_length: 2 bytes LE
        let footer_length = u16::from_le_bytes([buf[14], buf[15]]);
        if footer_length < FOOTER_V1_SIZE as u16 {
            return Err(SlabError::InvalidFooter(format!(
                "footer_length {footer_length} < {FOOTER_V1_SIZE}"
            )));
        }

        Ok(Footer {
            start_ordinal,
            record_count,
            page_size,
            page_type,
            namespace_index,
            footer_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize a footer with typical values and deserialize it back,
    /// confirming all fields survive the round-trip unchanged.
    #[test]
    fn test_footer_roundtrip() {
        let footer = Footer::new(42, 10, 4096, PageType::Data);
        let mut buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut buf);
        let decoded = Footer::read_from(&buf).unwrap();
        assert_eq!(footer, decoded);
    }

    /// Verify that a negative ordinal (−1) round-trips correctly through
    /// the 5-byte sign-extended encoding.
    #[test]
    fn test_footer_negative_ordinal() {
        let footer = Footer::new(-1, 1, 512, PageType::Data);
        let mut buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut buf);
        let decoded = Footer::read_from(&buf).unwrap();
        assert_eq!(decoded.start_ordinal, -1);
    }

    /// Verify the maximum positive 5-byte signed ordinal (2^39 − 1)
    /// round-trips without truncation or sign-extension errors.
    #[test]
    fn test_footer_large_ordinal() {
        // Max positive 5-byte signed: 2^39 - 1 = 549_755_813_887
        let max_ord: i64 = (1i64 << 39) - 1;
        let footer = Footer::new(max_ord, 100, 65536, PageType::Pages);
        let mut buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut buf);
        let decoded = Footer::read_from(&buf).unwrap();
        assert_eq!(decoded.start_ordinal, max_ord);
    }

    /// Verify the maximum 3-byte unsigned record count (2^24 − 1 =
    /// 16,777,215) round-trips correctly.
    #[test]
    fn test_footer_max_record_count() {
        // Max 3-byte unsigned: 2^24 - 1 = 16_777_215
        let footer = Footer::new(0, 0x00FF_FFFF, 65536, PageType::Data);
        let mut buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut buf);
        let decoded = Footer::read_from(&buf).unwrap();
        assert_eq!(decoded.record_count, 0x00FF_FFFF);
    }

    /// A footer with namespace_index 0 (invalid/reserved) must be rejected
    /// during deserialization with an `InvalidNamespaceIndex` error.
    #[test]
    fn test_footer_invalid_namespace_index_zero() {
        let footer = Footer {
            start_ordinal: 0,
            record_count: 0,
            page_size: 512,
            page_type: PageType::Data,
            namespace_index: 0,
            footer_length: 16,
        };
        let mut buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut buf);
        let result = Footer::read_from(&buf);
        assert!(result.is_err());
    }

    /// A footer with namespace_index 200 (in the reserved negative range
    /// when interpreted as signed) must be rejected.
    #[test]
    fn test_footer_invalid_namespace_index_reserved() {
        let footer = Footer {
            start_ordinal: 0,
            record_count: 0,
            page_size: 512,
            page_type: PageType::Data,
            namespace_index: 200,
            footer_length: 16,
        };
        let mut buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut buf);
        let result = Footer::read_from(&buf);
        assert!(result.is_err());
    }

    /// A footer with a valid user-defined namespace_index (e.g. 42) must
    /// round-trip successfully.
    #[test]
    fn test_footer_user_namespace_index() {
        let footer = Footer {
            start_ordinal: 0,
            record_count: 1,
            page_size: 512,
            page_type: PageType::Data,
            namespace_index: 42,
            footer_length: 16,
        };
        let mut buf = [0u8; FOOTER_V1_SIZE];
        footer.write_to(&mut buf);
        let decoded = Footer::read_from(&buf).unwrap();
        assert_eq!(decoded.namespace_index, 42);
    }

    /// Corrupting the page_type byte to 0 (Invalid) after serializing a
    /// valid footer must cause `read_from` to reject it.
    #[test]
    fn test_footer_invalid_page_type() {
        let mut buf = [0u8; FOOTER_V1_SIZE];
        // Write a valid footer then corrupt page_type
        let footer = Footer::new(0, 0, 512, PageType::Data);
        footer.write_to(&mut buf);
        buf[12] = 0; // Invalid
        let result = Footer::read_from(&buf);
        assert!(result.is_err());
    }
}
