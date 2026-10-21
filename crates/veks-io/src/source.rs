// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Where a reader's bytes come from: [`ByteSource`], and
//! [`MappedFile`], the source for a local file.
//!
//! Every reader in [`crate::reader`] is generic over its source, so the
//! one implementation of each format serves a local file mapped here
//! and any other place bytes can be fetched from. `vectordata` supplies
//! its own source (remote, cached, fetched on demand, a series of
//! files), and its readers are these readers over it. A source answers
//! byte questions; the readers own the layout.
//!
//! The source is a type parameter, not a trait object, so each reader
//! is compiled for its source and a read from a local mapping is a
//! pointer add (`srd-loose-files-overlays-integrity.md`, IO-2 and the
//! IO-4 measurements).

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use memmap2::Mmap;

use crate::reader::IoError;

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
    /// What [`prebuffer_with_progress`](Self::prebuffer_with_progress)
    /// reports as the source fills itself.
    type Progress;

    /// Open the file `source` names: a path, or whatever else this
    /// source can reach.
    fn open(source: &str) -> Result<Self, IoError>;

    /// Open a local file.
    fn open_path(path: &Path) -> Result<Self, IoError>;

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

    /// [`precache`](Self::precache), reporting progress.
    fn prebuffer_with_progress(&self, _cb: &mut dyn FnMut(&Self::Progress)) -> io::Result<()> {
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

/// A local file, memory-mapped: the source of the readers a program
/// uses on its own files.
#[derive(Clone)]
pub struct MappedFile {
    mmap: Arc<Mmap>,
    path: Arc<PathBuf>,
}

impl MappedFile {
    /// Map `path` read-only.
    pub fn map(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only; a file changed underneath a mapping is the
        // documented hazard of every mapped reader.
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Self { mmap: Arc::new(mmap), path: Arc::new(path.to_path_buf()) })
    }

    /// The mapping.
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }
}

impl fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedFile").field("path", &self.path).field("len", &self.mmap.len()).finish()
    }
}

impl ByteSource for MappedFile {
    type Progress = ();

    fn open(source: &str) -> Result<Self, IoError> {
        Self::open_path(Path::new(source))
    }

    fn open_path(path: &Path) -> Result<Self, IoError> {
        Ok(Self::map(path)?)
    }

    fn total_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        self.mmap_slice(offset, len).map(<[u8]>::to_vec).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read past end: offset={offset} len={len} size={}", self.mmap.len()),
            )
        })
    }

    fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(usize::try_from(len).ok()?)?;
        self.mmap.get(start..end)
    }

    fn mmap_base(&self) -> Option<*const u8> {
        Some(self.mmap.as_ptr())
    }

    fn local_path(&self) -> Option<PathBuf> {
        Some(self.path.as_ref().clone())
    }

    fn advise_sequential(&self) {
        mapping::advise_sequential_bytes(&self.mmap);
    }

    fn advise_random(&self) {
        mapping::advise_random_bytes(&self.mmap);
    }

    fn prefetch_range_bytes(&self, start: u64, end: u64) {
        mapping::prefetch_range_bytes(&self.mmap, start, end);
    }

    fn release_range_bytes(&self, start: u64, end: u64) {
        mapping::release_range_bytes(&self.mmap, start, end);
    }

    fn prefetch_pages_bytes(&self, start: u64, end: u64, paged: Option<&AtomicU64>) {
        mapping::prefetch_pages_bytes(&self.mmap, start, end, paged);
    }
}

/// Paging advice over a mapping, for every source that has one: the
/// one implementation of `madvise` use in the workspace.
///
/// Each takes the mapped bytes (a whole mapping, page-aligned at its
/// start) and does nothing off Unix.
pub mod mapping {
    use std::sync::atomic::{AtomicU64, Ordering};

    #[cfg(unix)]
    fn page_size() -> usize {
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page > 0 { page as usize } else { 4096 }
    }

    /// `madvise(advice)` over `bytes[start..end]`, widened out to whole
    /// pages (`WILLNEED`, `SEQUENTIAL`, `RANDOM` are hints, and a
    /// wider range costs nothing).
    #[cfg(unix)]
    fn advise_out(bytes: &[u8], start: usize, end: usize, advice: libc::c_int) {
        let end = end.min(bytes.len());
        if end <= start {
            return;
        }
        let page = page_size();
        let base = bytes.as_ptr() as usize;
        let from = (base + start) & !(page - 1);
        // SAFETY: the range starts on a page of the live mapping and ends
        // inside it; advice changes no contents.
        unsafe {
            libc::madvise(from as *mut libc::c_void, base + end - from, advice);
        }
    }

    /// `madvise(SEQUENTIAL)` over the whole mapping.
    pub fn advise_sequential_bytes(bytes: &[u8]) {
        #[cfg(unix)]
        advise_out(bytes, 0, bytes.len(), libc::MADV_SEQUENTIAL);
        #[cfg(not(unix))]
        let _ = bytes;
    }

    /// `madvise(RANDOM)` over the whole mapping.
    pub fn advise_random_bytes(bytes: &[u8]) {
        #[cfg(unix)]
        advise_out(bytes, 0, bytes.len(), libc::MADV_RANDOM);
        #[cfg(not(unix))]
        let _ = bytes;
    }

    /// `madvise(WILLNEED)` over `start..end`, clipped to the mapping.
    /// Does not wait.
    pub fn prefetch_range_bytes(bytes: &[u8], start: u64, end: u64) {
        #[cfg(unix)]
        advise_out(bytes, start as usize, end as usize, libc::MADV_WILLNEED);
        #[cfg(not(unix))]
        let _ = (bytes, start, end);
    }

    /// `madvise(DONTNEED)` over the whole pages inside `start..end`.
    ///
    /// The start rounds up and the end down to a page: dropping a page
    /// is not a hint, so only pages wholly inside the range go, and the
    /// sub-page slack a streaming scan leaves mapped is negligible
    /// beside its windows.
    pub fn release_range_bytes(bytes: &[u8], start: u64, end: u64) {
        if start >= end {
            return;
        }
        #[cfg(unix)]
        {
            let page = page_size();
            let base = bytes.as_ptr() as usize;
            let from = (base + start as usize + page - 1) & !(page - 1);
            let to = (base + (end as usize).min(bytes.len())) & !(page - 1);
            if to <= from {
                return;
            }
            // SAFETY: the range is whole pages inside the live, read-only
            // mapping; dropped pages read back in from the file.
            unsafe {
                libc::madvise(from as *mut libc::c_void, to - from, libc::MADV_DONTNEED);
            }
        }
        #[cfg(not(unix))]
        let _ = bytes;
    }

    /// Page `start..end` in and wait: `madvise(WILLNEED)`, then one
    /// read per page, adding 4096 to `paged` per page touched. The
    /// reads are what make it wait; the advice is a hint.
    pub fn prefetch_pages_bytes(bytes: &[u8], start: u64, end: u64, paged: Option<&AtomicU64>) {
        let start = start as usize;
        let end = (end as usize).min(bytes.len());
        if end <= start {
            return;
        }
        #[cfg(unix)]
        advise_out(bytes, start, end, libc::MADV_WILLNEED);
        let data = &bytes[start..end];
        const PAGE: usize = 4096;
        for offset in (0..data.len()).step_by(PAGE) {
            // SAFETY: `offset` is inside `data`.
            unsafe {
                std::ptr::read_volatile(&data[offset]);
            }
            if let Some(counter) = paged {
                counter.fetch_add(PAGE as u64, Ordering::Relaxed);
            }
        }
    }
}
