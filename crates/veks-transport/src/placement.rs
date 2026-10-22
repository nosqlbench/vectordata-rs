// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! [`CachePlacement`]: where one remote file's cache copy and its
//! sidecars live, as the caller decides it.
//!
//! The transport never computes a cache path (`srd-crate-boundaries.md`
//! TR-7, principle 5): which directory a dataset's files land in, and
//! what a sidecar is called, is the layout's business — vectordata's
//! `cache::layout`. The transport writes exactly the paths it is given.

use std::path::{Path, PathBuf};

/// The paths of one remote file's cache copy.
///
/// ```
/// use std::path::Path;
/// use veks_transport::CachePlacement;
///
/// let p = CachePlacement {
///     root: "/cache/ds".into(),
///     data: "/cache/ds/base.fvecs".into(),
///     merkle_state: "/cache/ds/base.fvecs.mrkl".into(),
///     chunk_map: "/cache/ds/base.fvecs.chunks".into(),
/// };
/// assert_eq!(p.data, Path::new("/cache/ds/base.fvecs"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePlacement {
    /// The directory the copy belongs to: what a user removes to clear
    /// it, named in the messages that tell them to.
    pub root: PathBuf,
    /// The cache copy of the file's bytes, written in place (sparse
    /// while it fills).
    pub data: PathBuf,
    /// The merkle state of a copy verified against a published
    /// reference (the `.mrkl` sidecar).
    pub merkle_state: PathBuf,
    /// The chunk map of a copy fetched without a reference (the
    /// `.chunks` sidecar): which chunks are here, and the tree recorded
    /// as they landed.
    pub chunk_map: PathBuf,
}

impl CachePlacement {
    /// The cache copy's path.
    pub fn data(&self) -> &Path {
        &self.data
    }
}
