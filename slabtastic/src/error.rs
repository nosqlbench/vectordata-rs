// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Error types for slabtastic operations.
//!
//! All fallible library functions return [`Result<T>`] which is an alias
//! for `std::result::Result<T, SlabError>`. I/O errors from the
//! underlying file are wrapped in [`SlabError::Io`].

use std::fmt;
use std::io;

/// Errors produced by slabtastic operations.
///
/// Most variants carry enough context to diagnose the problem without
/// re-reading the file (e.g. expected vs. actual sizes, the offending
/// byte value, etc.).
#[derive(Debug)]
pub enum SlabError {
    /// The magic bytes do not match "SLAB".
    InvalidMagic,
    /// The namespace index is not valid (0 = reserved, negative = reserved).
    InvalidNamespaceIndex(u8),
    /// The page type byte is not a valid variant.
    InvalidPageType(u8),
    /// The page size in the header does not match the footer.
    PageSizeMismatch { header: u32, footer: u32 },
    /// The page size is below the minimum (512 bytes).
    PageTooSmall(u32),
    /// The page size exceeds the maximum.
    PageTooLarge(u64),
    /// A single record exceeds the capacity of a page.
    RecordTooLarge { record_size: usize, max_size: usize },
    /// The requested ordinal is not present in the file.
    OrdinalNotFound(i64),
    /// The caller-specified ordinal does not match the writer's next expected ordinal.
    OrdinalMismatch { expected: i64, actual: i64 },
    /// The footer data is malformed.
    InvalidFooter(String),
    /// The page data is truncated or incomplete.
    TruncatedPage { expected: usize, actual: usize },
    /// An underlying I/O error.
    Io(io::Error),
    /// An error with additional location context (file offset, page index,
    /// ordinal).
    WithContext {
        /// The underlying error.
        source: Box<SlabError>,
        /// Byte offset in the file where the error occurred.
        file_offset: Option<u64>,
        /// Zero-based page index where the error occurred.
        page_index: Option<usize>,
        /// Ordinal associated with the error.
        ordinal: Option<i64>,
    },
}

impl fmt::Display for SlabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlabError::InvalidMagic => write!(f, "invalid magic bytes (expected SLAB)"),
            SlabError::InvalidNamespaceIndex(v) => write!(f, "invalid namespace index: {v}"),
            SlabError::InvalidPageType(t) => write!(f, "invalid page type: {t}"),
            SlabError::PageSizeMismatch { header, footer } => {
                write!(f, "page size mismatch: header={header}, footer={footer}")
            }
            SlabError::PageTooSmall(s) => write!(f, "page size {s} below minimum 512"),
            SlabError::PageTooLarge(s) => write!(f, "page size {s} exceeds maximum"),
            SlabError::RecordTooLarge {
                record_size,
                max_size,
            } => write!(
                f,
                "record size {record_size} exceeds max page capacity {max_size}"
            ),
            SlabError::OrdinalNotFound(o) => write!(f, "ordinal {o} not found"),
            SlabError::OrdinalMismatch { expected, actual } => {
                write!(f, "ordinal mismatch: expected {expected}, got {actual}")
            }
            SlabError::InvalidFooter(msg) => write!(f, "invalid footer: {msg}"),
            SlabError::TruncatedPage { expected, actual } => {
                write!(f, "truncated page: expected {expected} bytes, got {actual}")
            }
            SlabError::Io(e) => write!(f, "I/O error: {e}"),
            SlabError::WithContext {
                source,
                file_offset,
                page_index,
                ordinal,
            } => {
                write!(f, "{source}")?;
                let mut parts = Vec::new();
                if let Some(off) = file_offset {
                    parts.push(format!("offset {off}"));
                }
                if let Some(pg) = page_index {
                    parts.push(format!("page {pg}"));
                }
                if let Some(ord) = ordinal {
                    parts.push(format!("ordinal {ord}"));
                }
                if !parts.is_empty() {
                    write!(f, " (at {})", parts.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for SlabError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SlabError::Io(e) => Some(e),
            SlabError::WithContext { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl SlabError {
    /// Wrap this error with file location context.
    ///
    /// Returns a `WithContext` variant carrying the original error plus
    /// optional file offset, page index, and ordinal.
    pub fn with_context(
        self,
        file_offset: Option<u64>,
        page_index: Option<usize>,
        ordinal: Option<i64>,
    ) -> SlabError {
        SlabError::WithContext {
            source: Box::new(self),
            file_offset,
            page_index,
            ordinal,
        }
    }
}

impl From<io::Error> for SlabError {
    fn from(err: io::Error) -> Self {
        SlabError::Io(err)
    }
}

/// Convenience alias for `Result<T, SlabError>`.
pub type Result<T> = std::result::Result<T, SlabError>;
