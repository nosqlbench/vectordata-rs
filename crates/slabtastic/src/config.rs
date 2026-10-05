// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Writer configuration for page sizing and alignment.
//!
//! Page sizes must fall in the 512–4 GB range. The **preferred** size
//! governs when the writer flushes the current page: once accumulated
//! record data reaches the preferred threshold, the page is written to
//! disk and a new one started. **Alignment**, when enabled, pads every
//! page to a multiple of `min_page_size`.

use crate::constants::{MAX_PAGE_SIZE, MIN_PAGE_SIZE};
use crate::error::{Result, SlabError};

/// Configuration for a [`SlabWriter`](crate::SlabWriter).
///
/// ## Examples
///
/// ```
/// use slabtastic::WriterConfig;
///
/// // Aligned to 4 KiB pages
/// let cfg = WriterConfig::new(4096, 65536, u32::MAX, true).unwrap();
/// assert_eq!(cfg.aligned_size(5000), 8192); // rounds up to 2 * 4096
/// ```
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Minimum page size in bytes (must be >= 512).
    pub min_page_size: u32,
    /// Preferred page size in bytes. Pages are flushed when reaching this threshold.
    pub preferred_page_size: u32,
    /// Maximum page size in bytes (must be <= u32::MAX).
    pub max_page_size: u32,
    /// When true, pad pages to multiples of `min_page_size`.
    pub page_alignment: bool,
}

impl WriterConfig {
    /// Create a new `WriterConfig`, validating the constraints.
    pub fn new(
        min_page_size: u32,
        preferred_page_size: u32,
        max_page_size: u32,
        page_alignment: bool,
    ) -> Result<Self> {
        if min_page_size < MIN_PAGE_SIZE {
            return Err(SlabError::PageTooSmall(min_page_size));
        }
        if (max_page_size as u64) > (MAX_PAGE_SIZE as u64) {
            return Err(SlabError::PageTooLarge(max_page_size as u64));
        }
        if min_page_size > preferred_page_size || preferred_page_size > max_page_size {
            return Err(SlabError::InvalidFooter(format!(
                "page size ordering violated: min={min_page_size} <= preferred={preferred_page_size} <= max={max_page_size}"
            )));
        }
        Ok(WriterConfig {
            min_page_size,
            preferred_page_size,
            max_page_size,
            page_alignment,
        })
    }

    /// Compute the aligned page size for a given raw size.
    ///
    /// When alignment is enabled, rounds up to the next multiple of
    /// `min_page_size`.
    pub fn aligned_size(&self, raw_size: usize) -> usize {
        if !self.page_alignment {
            return raw_size;
        }
        let align = self.min_page_size as usize;
        let remainder = raw_size % align;
        if remainder == 0 {
            raw_size
        } else {
            raw_size + (align - remainder)
        }
    }
}

impl Default for WriterConfig {
    fn default() -> Self {
        WriterConfig {
            min_page_size: MIN_PAGE_SIZE,
            preferred_page_size: 4 * 1024 * 1024,
            max_page_size: MAX_PAGE_SIZE,
            page_alignment: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the `Default` impl produces the expected values:
    /// min=512, preferred=4 MiB, max=u32::MAX, alignment off.
    #[test]
    fn test_default_config() {
        let cfg = WriterConfig::default();
        assert_eq!(cfg.min_page_size, 512);
        assert_eq!(cfg.preferred_page_size, 4 * 1024 * 1024);
        assert_eq!(cfg.max_page_size, u32::MAX);
        assert!(!cfg.page_alignment);
    }

    /// A min_page_size below the absolute minimum (512) must be rejected.
    #[test]
    fn test_config_validation_min_too_small() {
        let result = WriterConfig::new(256, 512, 1024, false);
        assert!(result.is_err());
    }

    /// min (1024) > preferred (512) violates the ordering constraint
    /// and must be rejected.
    #[test]
    fn test_config_validation_ordering() {
        let result = WriterConfig::new(1024, 512, 2048, false);
        assert!(result.is_err());
    }

    /// With alignment enabled and min_page_size=512, `aligned_size`
    /// must round up to the next multiple of 512. Values already on a
    /// boundary are unchanged.
    #[test]
    fn test_aligned_size() {
        let cfg = WriterConfig::new(512, 65536, u32::MAX, true).unwrap();
        assert_eq!(cfg.aligned_size(512), 512);
        assert_eq!(cfg.aligned_size(513), 1024);
        assert_eq!(cfg.aligned_size(1023), 1024);
        assert_eq!(cfg.aligned_size(1024), 1024);
    }

    /// With alignment disabled, `aligned_size` must return the input
    /// unchanged regardless of whether it's a multiple of min_page_size.
    #[test]
    fn test_aligned_size_disabled() {
        let cfg = WriterConfig::new(512, 65536, u32::MAX, false).unwrap();
        assert_eq!(cfg.aligned_size(513), 513);
    }
}
