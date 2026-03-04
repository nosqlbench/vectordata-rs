// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Format constants for the slabtastic wire format.
//!
//! These constants define the structural invariants shared by all pages:
//! magic bytes, header/footer sizes, page size limits, and version tags.
//!
//! ## File capacity
//!
//! Slabtastic supports files of up to 2^63 bytes. All file-level offsets
//! are twos-complement signed 8-byte little-endian integers for easy
//! interop with language stacks that lack unsigned 64-bit types.
//!
//! ## Page size bounds
//!
//! Pages must be between 2^9 (512) and 2^32 bytes. The 512-byte minimum
//! ensures pages are large enough to hold the header, at least one
//! offset entry, and the footer. The 2^32 upper limit keeps page sizes
//! addressable by a 4-byte field and within single `mmap` call limits on
//! older Java runtimes.

/// Magic bytes identifying a slabtastic page: ASCII "SLAB".
///
/// Every page begins with these four bytes, used both for file-type
/// identification and as a structural anchor during forward traversal.
pub const MAGIC: [u8; 4] = *b"SLAB";

/// Size of the page header in bytes (4 magic + 4 page_size).
pub const HEADER_SIZE: usize = 8;

/// Minimum allowed page size in bytes (2^9).
pub const MIN_PAGE_SIZE: u32 = 512;

/// Maximum allowed page size in bytes (2^32 - 1).
pub const MAX_PAGE_SIZE: u32 = u32::MAX;

/// Size of the v1 page footer in bytes.
pub const FOOTER_V1_SIZE: usize = 16;

/// Default namespace index (byte 13 of the v1 footer).
///
/// In the v1 format this byte was called `version` and was always `1`.
/// With namespace support the same byte identifies the namespace a page
/// belongs to. Index `1` is the default namespace `""`, so existing v1
/// files are backward compatible without migration.
pub const DEFAULT_NAMESPACE_INDEX: u8 = 1;

/// Conventional file extension for slabtastic files.
///
/// By convention, slabtastic files use the `.slab` extension. This
/// constant includes the leading dot for easy use with path manipulation.
pub const SLAB_EXTENSION: &str = ".slab";

/// Page type discriminator (1-byte enum in the footer).
///
/// The page type distinguishes data pages (which hold user records) from
/// the pages page (the index) and the namespaces page. A value of 0 is
/// reserved as invalid. Page types implicitly carry their format version:
/// types 1, 2, and 3 are all v1-era types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    /// Invalid / uninitialized page (value 0). Used as a sentinel; a
    /// page with this type is always rejected during deserialization.
    Invalid = 0,
    /// Pages page — the file-level index (value 1). The last page in a
    /// single-namespace slabtastic file is of this type. Its records are
    /// `(start_ordinal:8, file_offset:8)` tuples sorted by ordinal.
    Pages = 1,
    /// Data page — holds user records (value 2). Records are packed
    /// contiguously and indexed by the trailing offset array.
    Data = 2,
    /// Namespaces page (value 3). Maps namespace names to indices and
    /// locates each namespace's pages page. When present, this is always
    /// the last page in the file.
    Namespaces = 3,
}

impl PageType {
    /// Convert a raw byte to a `PageType`.
    pub fn from_u8(value: u8) -> Option<PageType> {
        match value {
            0 => Some(PageType::Invalid),
            1 => Some(PageType::Pages),
            2 => Some(PageType::Data),
            3 => Some(PageType::Namespaces),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the magic constant is the ASCII bytes "SLAB".
    #[test]
    fn test_magic_bytes() {
        assert_eq!(&MAGIC, b"SLAB");
    }

    /// Convert each valid PageType variant (0, 1, 2, 3) to u8 and back,
    /// confirming round-trip identity. Values outside the enum (4, 255)
    /// must return `None`.
    #[test]
    fn test_page_type_roundtrip() {
        for val in 0..=3u8 {
            let pt = PageType::from_u8(val).unwrap();
            assert_eq!(pt as u8, val);
        }
        assert!(PageType::from_u8(4).is_none());
        assert!(PageType::from_u8(255).is_none());
    }
}
