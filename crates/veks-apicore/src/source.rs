// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Where a reader's bytes come from: [`ByteSource`].
//!
//! Every reader is generic over its source, so the one implementation
//! of each format (veks-io's) serves a local mapped file, a remote or
//! cached one, and a series of files alike. A source answers byte
//! questions; the readers own the layout. Opening a source is its
//! implementation's business (a path, a URL, a catalog's facet), so the
//! contract names no path or URL scheme.
//!
//! The source is a type parameter, not a trait object: each reader is
//! compiled for its source, and a read from a local mapping is a
//! pointer add (`srd-crate-boundaries.md` PR-1, PR-3).
//!
//! ```
//! use std::io;
//! use std::sync::Arc;
//! use veks_apicore::ByteSource;
//!
//! /// Bytes held in memory: enough to be read by any veks-io reader.
//! #[derive(Clone, Debug)]
//! struct InMemory(Arc<Vec<u8>>);
//!
//! impl ByteSource for InMemory {
//!     fn total_size(&self) -> u64 {
//!         self.0.len() as u64
//!     }
//!     fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
//!         let (a, b) = (offset as usize, (offset + len) as usize);
//!         self.0.get(a..b).map(<[u8]>::to_vec).ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))
//!     }
//!     fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
//!         self.0.get(offset as usize..(offset + len) as usize)
//!     }
//!     fn mmap_base(&self) -> Option<*const u8> {
//!         Some(self.0.as_ptr())
//!     }
//! }
//!
//! let source = InMemory(Arc::new(vec![1, 2, 3, 4]));
//! assert_eq!(source.read_bytes(1, 2).unwrap(), vec![2, 3]);
//! assert!(source.is_complete(), "a source is complete unless it says otherwise");
//! ```

use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;

use crate::progress::DownloadProgress;

/// One file of a source that is a series of files: where it is
/// mapped, when it is, and how many bytes it holds.
#[derive(Debug, Clone, Copy)]
pub struct SourcePart {
    /// The part's mapped base address, `None` when it is not mapped.
    pub base: Option<*const u8>,
    /// The part's size in bytes.
    pub size: u64,
}

// SAFETY: the address is only ever dereferenced by a reader that also
// holds the source owning the mapping, for as long as it holds it.
unsafe impl Send for SourcePart {}
// SAFETY: as above; the mapping is read-only.
unsafe impl Sync for SourcePart {}

/// The bytes of one file, as a reader asks for them.
///
/// A source is cheap to clone (it shares what it holds), and once it
/// answers [`mmap_base`](Self::mmap_base) with `Some`, it answers with
/// the same address for as long as it lives: a reader may keep it.
pub trait ByteSource: Clone + Send + Sync + fmt::Debug + 'static {
    /// The file's size in bytes.
    fn total_size(&self) -> u64;

    /// `len` bytes at `offset`, fetched if they are not here.
    fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>>;

    /// `len` bytes at `offset`, borrowed, when they are mapped. `None`
    /// sends the caller to [`read_bytes`](Self::read_bytes).
    fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]>;

    /// The whole file's mapped base address, when it is one mapping.
    fn mmap_base(&self) -> Option<*const u8>;

    /// The files of a series, in order. Empty for one file.
    fn parts(&self) -> Vec<SourcePart> {
        Vec::new()
    }

    /// The local file holding the bytes, when there is one: the file
    /// itself, or a cache copy of a remote one.
    fn local_path(&self) -> Option<PathBuf> {
        None
    }

    /// Whether the bytes live somewhere else and `local_path` is a
    /// copy of them. A copy's mtime changes as it fills, so it says
    /// nothing about whether a sidecar beside it is stale.
    fn is_remote(&self) -> bool {
        false
    }

    /// Whether every byte is here, so no read makes a request.
    fn is_complete(&self) -> bool {
        true
    }

    /// Bring every byte here. A no-op for a local file.
    fn precache(&self) -> io::Result<()> {
        Ok(())
    }

    /// [`precache`](Self::precache), reporting the transfer as it goes.
    fn prebuffer_with_progress(&self, _cb: &mut dyn FnMut(&DownloadProgress)) -> io::Result<()> {
        self.precache()
    }

    /// The bytes of `name`, a file published beside `data_source`
    /// (an `IDXFOR__` offset index), or `None` when there is none or
    /// it cannot be asked for. A local source has nothing to fetch:
    /// its sidecars are read from beside [`local_path`](Self::local_path).
    fn fetch_sibling(&self, _data_source: &str, _name: &str) -> Option<Vec<u8>> {
        None
    }

    /// Hint a front-to-back read of the whole file.
    fn advise_sequential(&self) {}

    /// Hint reads in no order.
    fn advise_random(&self) {}

    /// Hint that `start..end` will be read soon; does not wait.
    fn prefetch_range_bytes(&self, _start: u64, _end: u64) {}

    /// Let the pages of `start..end` go; they read back in on demand.
    fn release_range_bytes(&self, _start: u64, _end: u64) {}

    /// Page `start..end` in and wait, adding the bytes touched to
    /// `paged`.
    fn prefetch_pages_bytes(&self, _start: u64, _end: u64, _paged: Option<&AtomicU64>) {}
}
