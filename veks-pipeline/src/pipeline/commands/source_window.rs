// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared utility for parsing source strings with optional window ranges.
//!
//! Commands that accept vector file paths can receive them with inline
//! window sugar (e.g., `base.fvec[0..1M]`) or as separate path + window
//! option pairs. This module provides a unified way to extract the file
//! path and effective vector range.
//!
//! ## Motivation
//!
//! When multiple profiles share the same base vectors file, each profile
//! can specify a window into that file rather than requiring a separate
//! extracted copy. For example:
//!
//! ```yaml
//! profiles:
//!   default:
//!     base_vectors: profiles/default/base_vectors.fvec
//!   1m:
//!     base_count: 1000000
//!     base_vectors: "profiles/default/base_vectors.fvec[0..1M)"
//! ```
//!
//! Commands that load vectors parse the source string to obtain both the
//! physical file path and the logical record window.

use std::path::{Path, PathBuf};
use vectordata::dataset::source::parse_source_string;

/// A resolved source: physical file path plus an optional record range.
#[derive(Debug, Clone)]
pub struct ResolvedSource {
    /// Physical file path (resolved against workspace).
    pub path: PathBuf,
    /// Optional record window `[start, end)`. `None` means all records.
    pub window: Option<(usize, usize)>,
}

impl ResolvedSource {
    /// Returns the effective record range given the file's total count.
    ///
    /// If no window is set, returns `(0, file_count)`.
    /// If a window is set, clamps to `file_count`.
    pub fn effective_range(&self, file_count: usize) -> (usize, usize) {
        match self.window {
            Some((start, end)) => (start.min(file_count), end.min(file_count)),
            None => (0, file_count),
        }
    }

    /// Returns the effective count of records in the window.
    pub fn effective_count(&self, file_count: usize) -> usize {
        let (start, end) = self.effective_range(file_count);
        end.saturating_sub(start)
    }
}

/// Parse a source string with optional window and resolve the path.
///
/// Accepts forms like:
/// - `"base.fvec"` — plain path, no window
/// - `"base.fvec[0..1M]"` — path with bracket window
/// - `"base.fvec(0..1000)"` — path with paren window
/// - `"../default/base.fvec[0..1M)"` — relative path with window
///
/// The window intervals are converted to `(usize, usize)` ranges.
/// Multiple intervals are not supported for vector commands — only the
/// first interval is used.
pub fn resolve_source(source_str: &str, workspace: &Path) -> Result<ResolvedSource, String> {
    let ds = parse_source_string(source_str)?;

    let path = {
        let p = PathBuf::from(&ds.path);
        if p.is_absolute() { p } else { workspace.join(p) }
    };

    let window = if ds.window.is_empty() {
        None
    } else {
        let interval = &ds.window.0[0];
        let start = interval.min_incl as usize;
        let end = if interval.max_excl == u64::MAX {
            usize::MAX
        } else {
            interval.max_excl as usize
        };
        Some((start, end))
    };

    Ok(ResolvedSource { path, window })
}

/// Resolve a source path relative to the workspace.
///
/// Relative paths are joined with the workspace directory so that file
/// operations work regardless of the process CWD. The result stays
/// relative when the workspace itself is relative (e.g., `.`), keeping
/// user-visible output free of absolute paths per SRD §1.6.7.
pub fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_resolve_plain_path() {
        let ws = Path::new("/workspace");
        let src = resolve_source("base.fvec", ws).unwrap();
        assert_eq!(src.path, PathBuf::from("/workspace/base.fvec"));
        assert_eq!(src.window, None);
        assert_eq!(src.effective_range(10000), (0, 10000));
    }

    #[test]
    fn test_resolve_with_bracket_window() {
        let ws = Path::new("/workspace");
        let src = resolve_source("base.fvec[0..1000000]", ws).unwrap();
        assert_eq!(src.path, PathBuf::from("/workspace/base.fvec"));
        // Brackets in source sugar are structural delimiters, not interval
        // bound indicators. Inner "0..1000000" parses as [0, 1000000).
        assert_eq!(src.window, Some((0, 1_000_000)));
    }

    #[test]
    fn test_resolve_with_paren_window() {
        let ws = Path::new("/workspace");
        let src = resolve_source("base.fvec(0..1000)", ws).unwrap();
        assert_eq!(src.path, PathBuf::from("/workspace/base.fvec"));
        assert_eq!(src.window, Some((0, 1000)));
    }

    #[test]
    fn test_resolve_relative_path_with_window() {
        let ws = Path::new("/workspace");
        let src = resolve_source("../default/base.fvec[0..1M]", ws).unwrap();
        // Path::join handles ../
        assert_eq!(src.path, PathBuf::from("/workspace/../default/base.fvec"));
        assert!(src.window.is_some());
    }

    #[test]
    fn test_effective_range_clamps() {
        let src = ResolvedSource {
            path: PathBuf::from("test.fvec"),
            window: Some((0, 1_000_000)),
        };
        // File has only 500 vectors
        assert_eq!(src.effective_range(500), (0, 500));
        assert_eq!(src.effective_count(500), 500);
        // File has 2M vectors
        assert_eq!(src.effective_range(2_000_000), (0, 1_000_000));
        assert_eq!(src.effective_count(2_000_000), 1_000_000);
    }

    #[test]
    fn test_effective_range_no_window() {
        let src = ResolvedSource {
            path: PathBuf::from("test.fvec"),
            window: None,
        };
        assert_eq!(src.effective_range(5000), (0, 5000));
        assert_eq!(src.effective_count(5000), 5000);
    }

    #[test]
    fn test_effective_range_with_offset() {
        let src = ResolvedSource {
            path: PathBuf::from("test.fvec"),
            window: Some((1000, 5000)),
        };
        assert_eq!(src.effective_range(10000), (1000, 5000));
        assert_eq!(src.effective_count(10000), 4000);
    }
}
