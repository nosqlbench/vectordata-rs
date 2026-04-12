// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The pages page — the file-level index of a slabtastic file.
//!
//! The pages page is always the **last page** in the file. It uses the
//! standard page layout to store `PageEntry` records — tuples of
//! `(start_ordinal:8, file_offset:8)` — sorted by ordinal to support
//! O(log₂ n) binary-search lookup via [`PagesPage::find_page_for_ordinal`].
//!
//! ## Constraints
//!
//! - **Single-page requirement**: the pages page must fit in a single
//!   page, which puts a hard upper bound on the number of data pages in
//!   a v1 file (page capacity / 16 bytes per entry).
//! - **Authoritative last page**: a valid slabtastic file always ends
//!   with a pages page. The last pages page in the file is authoritative;
//!   earlier pages pages are logically dead.
//! - **Logical deletion**: data pages not referenced by the pages page
//!   are logically deleted and must not be used by readers.
//!
//! ## Ordering
//!
//! Entries are sorted by `start_ordinal` for binary search. However,
//! the underlying data pages they reference are **not** required to
//! appear in monotonic file-offset order — pages may be out of order on
//! disk when append-only revisions rewrite earlier ordinal ranges.
//!
//! ## Examples
//!
//! ```
//! use slabtastic::PagesPage;
//!
//! let mut pp = PagesPage::new();
//! pp.add_entry(0, 0);
//! pp.add_entry(100, 4096);
//!
//! let entry = pp.find_page_for_ordinal(50).unwrap();
//! assert_eq!(entry.start_ordinal, 0);
//! assert_eq!(entry.file_offset, 0);
//! ```

use crate::constants::PageType;
use crate::error::{Result, SlabError};
use crate::page::Page;

/// An entry in the pages page mapping a starting ordinal to a file offset.
///
/// Wire format: `[start_ordinal:8][file_offset:8]` (16 bytes, little-endian).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageEntry {
    /// The starting ordinal of the referenced data page.
    pub start_ordinal: i64,
    /// The byte offset of the data page within the slabtastic file.
    pub file_offset: i64,
}

impl PageEntry {
    /// Serialize this entry to 16 bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.start_ordinal.to_le_bytes());
        buf[8..16].copy_from_slice(&self.file_offset.to_le_bytes());
        buf
    }

    /// Deserialize an entry from 16 bytes (little-endian).
    pub fn from_bytes(buf: &[u8]) -> PageEntry {
        let start_ordinal = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        let file_offset = i64::from_le_bytes(buf[8..16].try_into().unwrap());
        PageEntry {
            start_ordinal,
            file_offset,
        }
    }
}

/// A pages page (index page) that stores `PageEntry` records using the
/// standard page layout with `page_type = Pages`.
///
/// Entries are sorted by `start_ordinal` so that
/// [`find_page_for_ordinal`](Self::find_page_for_ordinal) can binary
/// search in O(log₂ n).
///
/// A pre-built `sorted_entries` cache is populated at construction and
/// deserialization time so that lookups are allocation-free.
#[derive(Debug, Clone)]
pub struct PagesPage {
    /// The underlying page.
    pub page: Page,
    /// Cached entries sorted by `start_ordinal`, built once at
    /// construction / deserialization time.
    sorted_entries: Vec<PageEntry>,
}

impl PagesPage {
    /// Create a new empty pages page.
    pub fn new() -> Self {
        PagesPage {
            page: Page::new(0, PageType::Pages),
            sorted_entries: Vec::new(),
        }
    }

    /// Add an entry mapping a start ordinal to a file offset.
    ///
    /// Entries must be added in non-decreasing `start_ordinal` order (which
    /// is the natural order during sequential writes). The sorted invariant
    /// is maintained by insertion order alone — no sort is performed.
    pub fn add_entry(&mut self, start_ordinal: i64, file_offset: i64) {
        let entry = PageEntry {
            start_ordinal,
            file_offset,
        };
        self.page.add_record(&entry.to_bytes());
        debug_assert!(
            self.sorted_entries.last().is_none_or(|prev| prev.start_ordinal <= start_ordinal),
            "add_entry called out of order: last={}, new={}",
            self.sorted_entries.last().unwrap().start_ordinal,
            start_ordinal,
        );
        self.sorted_entries.push(entry);
    }

    /// Return the cached entries (sorted by `start_ordinal`).
    pub fn entries(&self) -> Vec<PageEntry> {
        self.sorted_entries.clone()
    }

    /// Return a borrowed slice of the cached entries (sorted by
    /// `start_ordinal`), with zero allocation.
    pub fn sorted_entries_ref(&self) -> &[PageEntry] {
        &self.sorted_entries
    }

    /// Binary search for the page entry containing the given ordinal.
    ///
    /// Returns the entry whose `start_ordinal` is the greatest value
    /// less than or equal to `ordinal`. Uses the pre-built sorted
    /// entries cache — zero allocation per call.
    pub fn find_page_for_ordinal(&self, ordinal: i64) -> Option<PageEntry> {
        if self.sorted_entries.is_empty() {
            return None;
        }

        // Binary search: find the rightmost entry where start_ordinal <= ordinal
        match self
            .sorted_entries
            .binary_search_by_key(&ordinal, |e| e.start_ordinal)
        {
            Ok(i) => Some(self.sorted_entries[i]),
            Err(0) => None, // ordinal is before all entries
            Err(i) => Some(self.sorted_entries[i - 1]),
        }
    }

    /// Serialize this pages page to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        self.page.serialize()
    }

    /// Deserialize a pages page from bytes.
    ///
    /// The sorted entries cache is populated eagerly so that subsequent
    /// [`find_page_for_ordinal`](Self::find_page_for_ordinal) calls are
    /// allocation-free.
    pub fn deserialize(buf: &[u8]) -> Result<PagesPage> {
        let page = Page::deserialize(buf)?;
        if page.footer.page_type != PageType::Pages {
            return Err(SlabError::InvalidPageType(page.footer.page_type as u8));
        }
        let mut sorted_entries: Vec<PageEntry> = page
            .records
            .iter()
            .map(|r| PageEntry::from_bytes(r))
            .collect();
        sorted_entries.sort_by_key(|e| e.start_ordinal);
        Ok(PagesPage {
            page,
            sorted_entries,
        })
    }

    /// Return the number of page entries.
    pub fn entry_count(&self) -> usize {
        self.page.record_count()
    }
}

impl Default for PagesPage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize a pages page with 3 entries and deserialize it back.
    /// All ordinals and offsets must survive the round-trip unchanged.
    #[test]
    fn test_pages_page_roundtrip() {
        let mut pp = PagesPage::new();
        pp.add_entry(0, 0);
        pp.add_entry(100, 4096);
        pp.add_entry(200, 8192);

        let bytes = pp.serialize();
        let decoded = PagesPage::deserialize(&bytes).unwrap();
        let entries = decoded.entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], PageEntry { start_ordinal: 0, file_offset: 0 });
        assert_eq!(entries[1], PageEntry { start_ordinal: 100, file_offset: 4096 });
        assert_eq!(entries[2], PageEntry { start_ordinal: 200, file_offset: 8192 });
    }

    /// Looking up an ordinal that exactly matches an entry's
    /// `start_ordinal` must return that entry (binary search
    /// exact-match path).
    #[test]
    fn test_find_page_exact() {
        let mut pp = PagesPage::new();
        pp.add_entry(0, 0);
        pp.add_entry(100, 4096);
        pp.add_entry(200, 8192);

        let entry = pp.find_page_for_ordinal(100).unwrap();
        assert_eq!(entry.start_ordinal, 100);
        assert_eq!(entry.file_offset, 4096);
    }

    /// Looking up ordinal 150, which falls between entries 100 and 200,
    /// must return the entry with start_ordinal 100 (greatest entry ≤
    /// the requested ordinal).
    #[test]
    fn test_find_page_between() {
        let mut pp = PagesPage::new();
        pp.add_entry(0, 0);
        pp.add_entry(100, 4096);
        pp.add_entry(200, 8192);

        let entry = pp.find_page_for_ordinal(150).unwrap();
        assert_eq!(entry.start_ordinal, 100);
    }

    /// Looking up ordinal 5 when the first entry starts at 10 must
    /// return `None` — no page covers ordinals before its first entry.
    #[test]
    fn test_find_page_before_first() {
        let mut pp = PagesPage::new();
        pp.add_entry(10, 0);

        assert!(pp.find_page_for_ordinal(5).is_none());
    }

    /// Looking up any ordinal in an empty pages page must return `None`.
    #[test]
    fn test_find_page_empty() {
        let pp = PagesPage::new();
        assert!(pp.find_page_for_ordinal(0).is_none());
    }
}
