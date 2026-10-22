// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A facet's bytes, wherever they live: [`FacetBytes`], what
//! [`TestDataView::open_facet_storage`](crate::TestDataView::open_facet_storage)
//! hands out, and [`ContentId`], what the bytes are.
//!
//! A handle answers questions about the bytes (how many, how many are
//! here, what they hash to) and brings them here (whole, by range, by
//! shard of a series). It is what fetch planning and execution
//! ([`crate::fetch`]) are written against, so a view of any
//! implementation plans and fetches the same way.
//!
//! Every call here is on a cold path — opening, planning, a whole range
//! fetched, a whole file hashed — and goes through `dyn`; reading
//! records is a reader's, which the implementation builds from its own
//! concrete handle (`srd-crate-boundaries.md` DA-3, PR-1).

use std::any::Any;
use std::io;
use std::path::PathBuf;

use crate::progress::{DownloadProgress, ProgressSink};
use crate::records::{RecordError, RecordFacet};
use crate::view::{CacheStats, MappedRange, RangeFill};

/// A facet's content identity: the root of a merkle tree over its
/// bytes at the 8 MiB chunk size, so the same bytes have the same root
/// however it was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentId {
    /// The tree's root.
    pub root: [u8; 32],
    /// How the tree was obtained.
    pub kind: ContentIdKind,
}

/// How a [`ContentId`]'s tree was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentIdKind {
    /// The publisher's `.mref`, against which every fetched chunk is
    /// verified.
    Published,
    /// Recorded as the chunks of a file published without one were
    /// fetched (the `.chunks` sidecar).
    Recorded,
    /// Computed from a local file.
    Computed,
}

impl ContentId {
    /// The root as lowercase hex.
    pub fn hex(&self) -> String {
        self.root.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl std::fmt::Display for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.kind {
            ContentIdKind::Published => "published",
            ContentIdKind::Recorded => "recorded",
            ContentIdKind::Computed => "computed",
        };
        write!(f, "{} ({kind})", self.hex())
    }
}

/// An open handle on one facet's bytes: a local file, a remote or
/// cached one, or a series of files.
///
/// Byte offsets are facet offsets for a single file; the `shard_*`
/// methods and a series' [`shard_sizes`](Self::shard_sizes) address one
/// file of a series.
///
/// Only [`total_size`](Self::total_size), [`read_prefix`](Self::read_prefix)
/// and [`as_any`](Self::as_any) are required. The defaults describe
/// bytes that are already here: complete, local, nothing to fetch, no
/// cache and no identity. vectordata's handle overrides every one.
///
/// ```
/// use std::any::Any;
/// use veks_apicore::FacetBytes;
///
/// /// A facet held in memory: a test double, or a program's own bytes.
/// struct InMemory(Vec<u8>);
///
/// impl FacetBytes for InMemory {
///     fn total_size(&self) -> u64 {
///         self.0.len() as u64
///     }
///     fn read_prefix(&self, len: u64) -> std::io::Result<Vec<u8>> {
///         Ok(self.0[..(len as usize).min(self.0.len())].to_vec())
///     }
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
/// }
///
/// let bytes: Box<dyn FacetBytes> = Box::new(InMemory(vec![2, 0, 0, 0, 1, 2]));
/// assert_eq!(bytes.read_prefix(4).unwrap(), vec![2, 0, 0, 0]);
/// assert!(bytes.is_complete() && bytes.resident_bytes() == 6, "the defaults: already here");
/// assert_eq!(bytes.content_id().unwrap(), None, "and no identity unless it says so");
/// ```
pub trait FacetBytes: Send + Sync + 'static {
    /// The facet's size in bytes, across every file of a series.
    fn total_size(&self) -> u64;

    /// [`total_size`](Self::total_size), reporting a failure to learn
    /// it (a remote that cannot be reached) instead of answering zero.
    fn try_total_size(&self) -> crate::Result<u64> {
        Ok(self.total_size())
    }

    /// Whether every byte is here, so no read makes a request.
    fn is_complete(&self) -> bool {
        true
    }

    /// Whether the bytes are a local file, never fetched.
    fn is_local(&self) -> bool {
        true
    }

    /// Whether the facet is a series of files.
    fn is_series(&self) -> bool {
        false
    }

    /// The size of each file of a series, in order; one entry, the
    /// facet's size, for a single file.
    fn shard_sizes(&self) -> Vec<u64> {
        vec![self.total_size()]
    }

    /// Bytes already here.
    fn resident_bytes(&self) -> u64 {
        if self.is_complete() { self.total_size() } else { 0 }
    }

    /// The local file holding the bytes, when there is one: the file
    /// itself, or the cache copy of a remote one.
    fn cache_path(&self) -> Option<PathBuf> {
        None
    }

    /// Every local file the facet's bytes are in: one, or one per
    /// shard. `None` when nothing is local yet.
    fn local_files(&self) -> Option<Vec<PathBuf>> {
        None
    }

    /// The cache's chunk accounting for a remote facet; `None` for a
    /// local one.
    fn cache_stats(&self) -> Option<CacheStats> {
        None
    }

    /// The chunks a byte range spans and how many are here.
    fn range_fill(&self, byte_start: u64, byte_end: u64) -> Option<RangeFill> {
        let _ = (byte_start, byte_end);
        None
    }

    /// [`range_fill`](Self::range_fill) within one file of a series.
    fn shard_range_fill(&self, shard: usize, byte_start: u64, byte_end: u64) -> Option<RangeFill> {
        let _ = shard;
        self.range_fill(byte_start, byte_end)
    }

    /// The first `len` bytes (a header, a magic), fetched if needed.
    fn read_prefix(&self, len: u64) -> io::Result<Vec<u8>>;

    /// The byte ranges records `start..end` of the facet at `path`
    /// occupy, for a format whose records map to bytes from the file
    /// alone (a uniform stride, or a published offset index). `None`
    /// when they do not, which sends the request to a whole-facet fetch.
    fn map_records(&self, path: &str, start: u64, end: u64) -> Option<MappedRange> {
        let _ = (path, start, end);
        None
    }

    /// Bring every byte here.
    fn precache(&self) -> io::Result<()> {
        Ok(())
    }

    /// [`precache`](Self::precache), reporting the transfer.
    fn prebuffer_with_progress(&self, cb: &mut dyn FnMut(&DownloadProgress)) -> io::Result<()> {
        let _ = cb;
        self.precache()
    }

    /// [`precache`](Self::precache) file by file, reporting each file
    /// of a series by its index.
    fn prebuffer_parts(&self, cb: &mut dyn FnMut(usize, &DownloadProgress)) -> io::Result<()> {
        self.prebuffer_with_progress(&mut |p| cb(0, p))
    }

    /// Bring bytes `byte_start..byte_end` here, reporting the transfer.
    fn prebuffer_range_with_progress(
        &self,
        byte_start: u64,
        byte_end: u64,
        cb: &mut dyn FnMut(&DownloadProgress),
    ) -> io::Result<()> {
        let _ = (byte_start, byte_end, cb);
        self.precache()
    }

    /// [`prebuffer_range_with_progress`](Self::prebuffer_range_with_progress)
    /// within one file of a series.
    fn prebuffer_shard_range(
        &self,
        shard: usize,
        byte_start: u64,
        byte_end: u64,
        cb: &mut dyn FnMut(&DownloadProgress),
    ) -> io::Result<()> {
        let _ = shard;
        self.prebuffer_range_with_progress(byte_start, byte_end, cb)
    }

    /// Ask the source whether a complete local copy is still current:
    /// `Ok(true)` when the source confirmed it, `Ok(false)` when it
    /// could not be reached and the copy stands; an error when it has
    /// changed.
    fn revalidate(&self) -> io::Result<bool> {
        Ok(true)
    }

    /// The facet's content identity ([`ContentId`]): the publisher's
    /// root, the one recorded as its chunks were fetched (`None` until
    /// every chunk is here), or a local file's, computed once. Equal
    /// bytes have equal roots. `None` for a series.
    /// [`content_id_with_progress`](Self::content_id_with_progress)
    /// reports any hashing it takes.
    fn content_id(&self) -> io::Result<Option<ContentId>> {
        self.content_id_with_progress(&mut crate::progress::NoProgress)
    }

    /// [`content_id`](Self::content_id), reporting any hashing to
    /// `progress` (bytes hashed of the facet's size).
    fn content_id_with_progress(&self, progress: &mut dyn ProgressSink) -> io::Result<Option<ContentId>> {
        let _ = progress;
        Ok(None)
    }

    /// The content identity known without reading anything: a published
    /// root, or a recorded one once fetched. `None` for a local file
    /// and a series.
    fn known_content_id(&self) -> Option<ContentId> {
        None
    }

    /// Check the cached chunks against what they were verified or
    /// recorded as when they landed: the chunk numbers whose bytes no
    /// longer match (local corruption), `None` for a local file.
    fn verify_cached(&self, progress: &mut dyn ProgressSink) -> io::Result<Option<Vec<u32>>> {
        let _ = progress;
        Ok(None)
    }

    /// Open the facet as records: a slab of opaque records, `namespace`
    /// selecting a document inside it. The default says the bytes are
    /// not a record container.
    fn open_records(&self, facet: &str, namespace: Option<&str>) -> Result<RecordFacet, RecordError> {
        let _ = namespace;
        Err(RecordError::Container(format!("facet '{facet}': these bytes are not a record container")))
    }

    /// The handle as [`Any`], so the implementation that opened it can
    /// recover its own concrete type on its own cold paths (building a
    /// reader, classifying its cache).
    fn as_any(&self) -> &dyn Any;
}
