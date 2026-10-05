// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The namespaces page — maps namespace names to indices and pages page offsets.
//!
//! When a slab file contains multiple namespaces, a namespaces page
//! (page type 3) is written as the final page. Each record maps a
//! namespace name to its index and the file offset of that namespace's
//! pages page.
//!
//! ## Record format
//!
//! ```text
//! [namespace_index:1][name_length:1][name_bytes:N][namespace_pages_page_offset:8]
//! ```
//!
//! - `namespace_index` (1 byte, unsigned): must match data page footer indices
//! - `name_length` (1 byte, unsigned): UTF-8 name length (0 for default, max 128)
//! - `name_bytes` (N bytes): UTF-8 encoded namespace name
//! - `namespace_pages_page_offset` (8 bytes, signed LE): file offset of the
//!   namespace's pages page

use crate::constants::PageType;
use crate::error::{Result, SlabError};
use crate::page::Page;

/// A single entry in the namespaces page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceEntry {
    /// The namespace index (1–127).
    pub namespace_index: u8,
    /// The namespace name (UTF-8, empty string for default).
    pub name: String,
    /// File offset of this namespace's pages page.
    pub pages_page_offset: i64,
}

impl NamespaceEntry {
    /// Serialize this entry to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut buf = Vec::with_capacity(2 + name_bytes.len() + 8);
        buf.push(self.namespace_index);
        buf.push(name_bytes.len() as u8);
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&self.pages_page_offset.to_le_bytes());
        buf
    }

    /// Deserialize an entry from bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<NamespaceEntry> {
        if buf.len() < 10 {
            return Err(SlabError::InvalidFooter(
                "namespace entry too short".to_string(),
            ));
        }
        let namespace_index = buf[0];
        let name_length = buf[1] as usize;
        if buf.len() < 2 + name_length + 8 {
            return Err(SlabError::InvalidFooter(
                "namespace entry truncated".to_string(),
            ));
        }
        let name = String::from_utf8(buf[2..2 + name_length].to_vec()).map_err(|_| {
            SlabError::InvalidFooter("namespace name is not valid UTF-8".to_string())
        })?;
        let offset_bytes: [u8; 8] = buf[2 + name_length..2 + name_length + 8]
            .try_into()
            .unwrap();
        let pages_page_offset = i64::from_le_bytes(offset_bytes);
        Ok(NamespaceEntry {
            namespace_index,
            name,
            pages_page_offset,
        })
    }
}

/// A namespaces page containing [`NamespaceEntry`] records.
///
/// Uses the standard page layout with `page_type = Namespaces` (3).
#[derive(Debug, Clone)]
pub struct NamespacesPage {
    /// The underlying page.
    pub page: Page,
}

impl NamespacesPage {
    /// Create a new empty namespaces page.
    pub fn new() -> Self {
        NamespacesPage {
            page: Page::new(0, PageType::Namespaces),
        }
    }

    /// Add a namespace entry.
    pub fn add_entry(&mut self, entry: &NamespaceEntry) {
        self.page.add_record(&entry.to_bytes());
    }

    /// Parse all namespace entries from this page.
    pub fn entries(&self) -> Result<Vec<NamespaceEntry>> {
        let mut entries = Vec::with_capacity(self.page.record_count());
        for i in 0..self.page.record_count() {
            let rec = self.page.get_record(i).unwrap();
            entries.push(NamespaceEntry::from_bytes(rec)?);
        }
        Ok(entries)
    }

    /// Serialize this namespaces page to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        self.page.serialize()
    }

    /// Deserialize a namespaces page from bytes.
    pub fn deserialize(buf: &[u8]) -> Result<NamespacesPage> {
        let page = Page::deserialize(buf)?;
        if page.footer.page_type != PageType::Namespaces {
            return Err(SlabError::InvalidPageType(page.footer.page_type as u8));
        }
        Ok(NamespacesPage { page })
    }
}

impl Default for NamespacesPage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize a namespaces page with the default namespace and a
    /// user-defined namespace, then deserialize and verify all fields.
    #[test]
    fn test_namespaces_page_roundtrip() {
        let mut np = NamespacesPage::new();
        np.add_entry(&NamespaceEntry {
            namespace_index: 1,
            name: "".to_string(),
            pages_page_offset: 4096,
        });
        np.add_entry(&NamespaceEntry {
            namespace_index: 2,
            name: "vectors".to_string(),
            pages_page_offset: 8192,
        });

        let bytes = np.serialize();
        let decoded = NamespacesPage::deserialize(&bytes).unwrap();
        let entries = decoded.entries().unwrap();
        assert_eq!(entries.len(), 2);

        assert_eq!(entries[0].namespace_index, 1);
        assert_eq!(entries[0].name, "");
        assert_eq!(entries[0].pages_page_offset, 4096);

        assert_eq!(entries[1].namespace_index, 2);
        assert_eq!(entries[1].name, "vectors");
        assert_eq!(entries[1].pages_page_offset, 8192);
    }

    /// NamespaceEntry round-trip with a long name.
    #[test]
    fn test_namespace_entry_roundtrip() {
        let entry = NamespaceEntry {
            namespace_index: 42,
            name: "my-namespace".to_string(),
            pages_page_offset: 123456,
        };
        let bytes = entry.to_bytes();
        let decoded = NamespaceEntry::from_bytes(&bytes).unwrap();
        assert_eq!(entry, decoded);
    }

    /// Deserializing a namespaces page from a pages-typed page must fail.
    #[test]
    fn test_namespaces_page_wrong_type() {
        let page = Page::new(0, PageType::Pages);
        let bytes = page.serialize();
        let result = NamespacesPage::deserialize(&bytes);
        assert!(result.is_err());
    }
}
