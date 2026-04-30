// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Byte-level storage abstraction. **Crate-private.**
//!
//! `Storage` hides the transport choice (local mmap, direct HTTP RANGE,
//! merkle-cached HTTP-with-mmap-promotion) behind a single byte-oriented
//! interface. Shape adapters (`XvecReader`, `IndexedVvecReader`,
//! `TypedReader`) are parameterised on `Arc<Storage>` so they share one
//! transport implementation regardless of data shape.
//!
//! This module is `pub(crate)` on purpose: callers outside the crate
//! must not be able to construct or name a `Storage` value, because
//! every public reader funnels through `Storage::open` and inherits the
//! cache-first remote behaviour automatically. The matrix-cell
//! omissions that motivated this refactor (typed-scalar over remote
//! had no cache; vvec over remote had no cache) are impossible by
//! construction once every shape adapter routes through `Storage`.

use std::io;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;

use memmap2::Mmap;
use url::Url;

use crate::cache::CachedChannel;
use crate::cache::reader::{cache_dir_for_url, default_cache_dir};
use crate::merkle::MerkleRef;
use crate::transport::HttpTransport;

/// Byte-oriented storage over an opaque transport.
///
/// Variants are deliberately not exposed: `pub(crate)` on the enum and
/// every constructor. Public reader types embed `Arc<Storage>` and
/// expose only shape-aware methods.
pub(crate) enum Storage {
    /// Local file, mmap'd. Zero-copy, no fallback.
    Mmap(Mmap),

    /// Remote URL, direct HTTP RANGE per read. Used only when no
    /// `.mref` is published. No accumulation, no mmap promotion. Slow
    /// — kept exclusively as the silent fallback when caching isn't
    /// possible.
    Http {
        transport: HttpTransport,
        total_size: u64,
    },

    /// Remote URL with a published `.mref`. Reads route through
    /// `CachedChannel`, which downloads + merkle-verifies chunks on
    /// demand and writes them to the local cache directory. When every
    /// chunk is verified, the storage flips to mmap so subsequent
    /// reads are zero-copy.
    Cached {
        channel: CachedChannel,
        /// Set exactly once when the cache is fully verified.
        mmap: OnceLock<Mmap>,
    },
}

impl Storage {
    /// Open from a path or URL string. Local paths get `Mmap`; URLs
    /// try `Cached` first (fetches `.mref`), fall back to `Http` when
    /// no `.mref` is published.
    pub(crate) fn open(source: &str) -> io::Result<Self> {
        if source.starts_with("http://") || source.starts_with("https://") {
            let url = Url::parse(source)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("invalid URL '{source}': {e}")))?;
            Self::open_url(url)
        } else {
            Self::open_path(Path::new(source))
        }
    }

    /// Open a local file by mmap. Returns `Storage::Mmap`.
    pub(crate) fn open_path(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Storage::Mmap(mmap))
    }

    /// Open a remote URL with cache-first dispatch. Tries to fetch the
    /// `.mref` and open a `Storage::Cached`; on `.mref` absence
    /// (404 / no published reference) falls back to `Storage::Http`.
    /// The fallback covers the legitimate "remote source has no
    /// merkle reference published" case; it does **not** mask
    /// configuration problems.
    ///
    /// If `vectordata::settings::cache_dir()` returns
    /// `Err(SettingsError::NotConfigured)`, this function surfaces
    /// the actionable error — there is no silent fallback to a
    /// default cache directory.
    pub(crate) fn open_url(url: Url) -> io::Result<Self> {
        let cache_root = default_cache_dir()
            .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
        match Self::open_url_cached(url.clone(), &cache_root) {
            Ok(s) => Ok(s),
            Err(_) => Self::open_url_http(url),
        }
    }

    /// Open a remote URL through the merkle-cache path. Requires a
    /// published `.mref`. Crate-internal; callers go through
    /// `open_url` so the fallback path runs uniformly.
    pub(crate) fn open_url_cached(url: Url, cache_root: &Path) -> io::Result<Self> {
        let client = reqwest::blocking::Client::new();
        let mref_url_str = format!("{}.mref", url.as_str());
        let mref_url = Url::parse(&mref_url_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("invalid mref URL: {e}")))?;
        let resp = client.get(mref_url).send()
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, format!("mref fetch: {e}")))?
            .error_for_status()
            .map_err(|e| io::Error::new(io::ErrorKind::NotFound, format!("no .mref for {url}: {e}")))?;
        let mref_bytes = resp.bytes()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("mref read: {e}")))?;
        let reference = MerkleRef::from_bytes(&mref_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("mref parse: {e}")))?;

        let cache_dir = cache_dir_for_url(&url, cache_root);
        let filename: String = url.path_segments()
            .and_then(|mut s| s.next_back())
            .unwrap_or("data")
            .to_string();
        let transport = HttpTransport::with_client(client, url);
        let channel = CachedChannel::open(
            Box::new(transport),
            reference,
            &cache_dir,
            &filename,
        )?;

        let mmap = OnceLock::new();
        if channel.is_complete() {
            if let Ok(file) = std::fs::File::open(channel.cache_path())
                && let Ok(m) = unsafe { Mmap::map(&file) }
            {
                let _ = mmap.set(m);
            }
        }

        Ok(Storage::Cached { channel, mmap })
    }

    /// Open a remote URL via direct HTTP RANGE. Crate-internal; the
    /// fallback path of `open_url`. No caching, no promotion.
    pub(crate) fn open_url_http(url: Url) -> io::Result<Self> {
        let transport = HttpTransport::new(url);
        let total_size = ChunkedTransportExt::content_length_for(&transport)?;
        Ok(Storage::Http { transport, total_size })
    }

    /// Read `len` bytes at `offset`. Always succeeds for an in-bounds
    /// range — slow-path downloads chunks first if necessary.
    pub(crate) fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 { return Ok(Vec::new()); }
        match self {
            Storage::Mmap(m) => {
                let end = offset.checked_add(len)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read range overflow"))?;
                if end > m.len() as u64 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof,
                        format!("read past end: offset={offset} len={len} size={}", m.len())));
                }
                Ok(m[offset as usize..end as usize].to_vec())
            }
            Storage::Http { transport, .. } => {
                use crate::transport::ChunkedTransport;
                transport.fetch_range(offset, len)
            }
            Storage::Cached { channel, mmap } => {
                if let Some(m) = mmap.get() {
                    let end = (offset + len) as usize;
                    if end > m.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof,
                            format!("read past end of cache mmap: offset={offset} len={len} size={}", m.len())));
                    }
                    return Ok(m[offset as usize..end].to_vec());
                }
                let bytes = channel.read(offset, len)?;
                if mmap.get().is_none() && channel.is_complete() {
                    if let Ok(file) = std::fs::File::open(channel.cache_path())
                        && let Ok(m) = unsafe { Mmap::map(&file) }
                    {
                        let _ = mmap.set(m);
                    }
                }
                Ok(bytes)
            }
        }
    }

    /// Raw base pointer of the underlying mmap, when one exists.
    /// Returned for hot-path zero-copy slice construction (see
    /// [`crate::io::XvecReader::get_slice`]); the caller is
    /// responsible for staying in bounds. `None` when the storage is
    /// not mmap-backed.
    pub(crate) fn mmap_base(&self) -> Option<*const u8> {
        match self {
            Storage::Mmap(m) => Some(m.as_ptr()),
            Storage::Cached { mmap, .. } => mmap.get().map(|m| m.as_ptr()),
            Storage::Http { .. } => None,
        }
    }

    /// Borrow a slice if the bytes are locally accessible without
    /// allocation. `Some` for `Mmap` always, for `Cached` once the
    /// file is fully promoted to mmap. `None` otherwise — caller
    /// falls back to `read_bytes`.
    pub(crate) fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let (start, end) = (offset as usize, (offset + len) as usize);
        match self {
            Storage::Mmap(m) => m.get(start..end),
            Storage::Cached { mmap, .. } => mmap.get().and_then(|m| m.get(start..end)),
            Storage::Http { .. } => None,
        }
    }

    /// Total size in bytes.
    pub(crate) fn total_size(&self) -> u64 {
        match self {
            Storage::Mmap(m) => m.len() as u64,
            Storage::Http { total_size, .. } => *total_size,
            Storage::Cached { channel, .. } => channel.content_size(),
        }
    }

    /// Whether all bytes are locally accessible. `true` for `Mmap`
    /// always, `true` for `Cached` once every chunk is verified,
    /// always `false` for `Http`.
    pub(crate) fn is_complete(&self) -> bool {
        match self {
            Storage::Mmap(_) => true,
            Storage::Http { .. } => false,
            Storage::Cached { channel, .. } => channel.is_complete(),
        }
    }

    /// Whether reads avoid network round-trips.
    pub(crate) fn is_local(&self) -> bool {
        match self {
            Storage::Mmap(_) => true,
            Storage::Cached { mmap, .. } => mmap.get().is_some(),
            Storage::Http { .. } => false,
        }
    }

    /// Force-download every chunk into the local cache. No-op for
    /// `Mmap` (already local). For `Cached`, drives `prebuffer` and
    /// promotes to mmap. For `Http`, no-op (no cache to populate).
    pub(crate) fn prebuffer(&self) -> io::Result<()> {
        self.prebuffer_with_progress(|_| {})
    }

    /// Same as [`prebuffer`] with a progress callback fired after the
    /// underlying download completes.
    pub(crate) fn prebuffer_with_progress<F>(&self, cb: F) -> io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        match self {
            Storage::Mmap(_) => Ok(()),
            Storage::Http { .. } => Ok(()), // no cache to populate
            Storage::Cached { channel, mmap } => {
                channel.prebuffer_with_progress(cb)?;
                if mmap.get().is_none() && channel.is_complete() {
                    if let Ok(file) = std::fs::File::open(channel.cache_path())
                        && let Ok(m) = unsafe { Mmap::map(&file) }
                    {
                        let _ = mmap.set(m);
                    }
                }
                Ok(())
            }
        }
    }

    /// Hint that the full mapping will be accessed sequentially.
    /// No-op for non-mmap variants (and for `Cached` until promotion).
    pub(crate) fn advise_sequential(&self) {
        match self {
            Storage::Mmap(m) => { let _ = m.advise(memmap2::Advice::Sequential); }
            Storage::Cached { mmap, .. } => {
                if let Some(m) = mmap.get() { let _ = m.advise(memmap2::Advice::Sequential); }
            }
            Storage::Http { .. } => {}
        }
    }

    /// Hint that the full mapping will be accessed in random order.
    pub(crate) fn advise_random(&self) {
        match self {
            Storage::Mmap(m) => { let _ = m.advise(memmap2::Advice::Random); }
            Storage::Cached { mmap, .. } => {
                if let Some(m) = mmap.get() { let _ = m.advise(memmap2::Advice::Random); }
            }
            Storage::Http { .. } => {}
        }
    }

    /// `madvise(WILLNEED)` over the byte range. Non-blocking hint.
    pub(crate) fn prefetch_range_bytes(&self, byte_start: u64, byte_end: u64) {
        let mmap = match self {
            Storage::Mmap(m) => Some(m),
            Storage::Cached { mmap, .. } => mmap.get(),
            Storage::Http { .. } => None,
        };
        if let Some(m) = mmap {
            let start = byte_start as usize;
            let end = (byte_end as usize).min(m.len());
            if end > start {
                let _ = m.advise_range(memmap2::Advice::WillNeed, start, end - start);
            }
        }
    }

    /// `madvise(DONTNEED)` over the byte range. Page-aligns start UP
    /// and end DOWN — sub-page slack stays mapped, but on a
    /// streaming scan that's negligible compared to the megabyte-or-
    /// larger windows callers pass in. Without alignment, madvise
    /// rejects with EINVAL silently.
    pub(crate) fn release_range_bytes(&self, byte_start: u64, byte_end: u64) {
        let mmap = match self {
            Storage::Mmap(m) => Some(m),
            Storage::Cached { mmap, .. } => mmap.get(),
            Storage::Http { .. } => None,
        };
        let Some(m) = mmap else { return; };
        if byte_start >= byte_end { return; }

        #[cfg(unix)]
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            if page == 0 { return; }
            let start = byte_start as usize;
            let end = (byte_end as usize).min(m.len());
            let aligned_start = (start + page - 1) & !(page - 1);
            let aligned_end = end & !(page - 1);
            if aligned_end <= aligned_start { return; }
            libc::madvise(
                m.as_ptr().add(aligned_start) as *mut libc::c_void,
                aligned_end - aligned_start,
                libc::MADV_DONTNEED,
            );
        }
    }

    /// Force the byte range into the page cache via `madvise(WILLNEED)`
    /// + sequential volatile read. Blocks until the pages are
    /// resident. Caller increments `bytes_paged` per page touched.
    pub(crate) fn prefetch_pages_bytes(
        &self,
        byte_start: u64,
        byte_end: u64,
        bytes_paged: Option<&AtomicU64>,
    ) {
        let mmap = match self {
            Storage::Mmap(m) => Some(m),
            Storage::Cached { mmap, .. } => mmap.get(),
            Storage::Http { .. } => None,
        };
        let Some(m) = mmap else { return; };
        let start = byte_start as usize;
        let end = (byte_end as usize).min(m.len());
        if end <= start { return; }
        let _ = m.advise_range(memmap2::Advice::WillNeed, start, end - start);

        let data = &m[start..end];
        let page_size = 4096;
        for offset in (0..data.len()).step_by(page_size) {
            unsafe { std::ptr::read_volatile(&data[offset]); }
            if let Some(counter) = bytes_paged {
                counter.fetch_add(page_size as u64, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Path to the cache file, when this storage is `Cached`. Used by
    /// debugging / diagnostic surfaces only.
    #[allow(dead_code)]
    pub(crate) fn cache_path(&self) -> Option<&Path> {
        match self {
            Storage::Cached { channel, .. } => Some(channel.cache_path()),
            _ => None,
        }
    }

    /// Borrow the underlying [`CachedChannel`] when this storage is
    /// `Cached`. Used by [`crate::view::FacetStorage::cache_stats`]
    /// to report fill state. **Crate-internal**: the channel itself
    /// is not part of the public API.
    pub(crate) fn cached_channel(&self) -> Option<&CachedChannel> {
        match self {
            Storage::Cached { channel, .. } => Some(channel),
            _ => None,
        }
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Mmap(m) => f.debug_struct("Storage::Mmap").field("size", &m.len()).finish(),
            Storage::Http { total_size, .. } => f.debug_struct("Storage::Http").field("size", total_size).finish(),
            Storage::Cached { channel, mmap } => f.debug_struct("Storage::Cached")
                .field("size", &channel.content_size())
                .field("complete", &channel.is_complete())
                .field("promoted", &mmap.get().is_some())
                .finish(),
        }
    }
}

/// Helper to call `ChunkedTransport::content_length` without exposing
/// the trait through the public surface.
struct ChunkedTransportExt;
impl ChunkedTransportExt {
    fn content_length_for(t: &HttpTransport) -> io::Result<u64> {
        use crate::transport::ChunkedTransport;
        t.content_length()
    }
}
