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

    /// Remote URL with no published `.mref`. Per-read fetches go
    /// through HTTP RANGE; calling [`prebuffer`] downloads the whole
    /// file once into the cache directory (no merkle verification —
    /// we don't have hashes for it) and promotes to mmap, so
    /// subsequent reads on this and any other `Storage` instance
    /// pointed at the same URL take the zero-copy path.
    Http {
        transport: HttpTransport,
        total_size: u64,
        /// Local path where `prebuffer` lands the downloaded file.
        /// Computed from the URL via `cache_dir_for_url` so multiple
        /// `Storage::Http` instances against the same URL share one
        /// on-disk file.
        cache_path: std::path::PathBuf,
        /// Promoted-mmap view of `cache_path`, set by `prebuffer` or
        /// at open time if a prior prebuffer left the file with the
        /// expected size.
        mmap: OnceLock<Mmap>,
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
    /// fallback path of `open_url` (used when no `.mref` is
    /// published). Per-read fetches go over HTTP until the caller
    /// runs [`prebuffer`], which downloads the whole file into
    /// `cache_dir/<host:port>/<url-path-prefix>/<filename>` and
    /// promotes to mmap.
    ///
    /// At open time, if the cache file already exists with the
    /// expected size (from a prior prebuffer), this constructor
    /// mmap-promotes immediately so the new `Storage::Http` starts
    /// in the zero-copy state.
    pub(crate) fn open_url_http(url: Url) -> io::Result<Self> {
        let cache_root = crate::settings::cache_dir()
            .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
        let cache_dir = cache_dir_for_url(&url, &cache_root);
        let filename: String = url.path_segments()
            .and_then(|mut s| s.next_back())
            .unwrap_or("data")
            .to_string();
        let cache_path = cache_dir.join(&filename);

        let transport = HttpTransport::new(url);
        let total_size = ChunkedTransportExt::content_length_for(&transport)?;

        let mmap = OnceLock::new();
        // Restore promotion from a prior prebuffer if the cache file
        // is present with the matching size. We don't have hashes —
        // the size match is the strongest invariant we can check
        // without a `.mref` published.
        if let Ok(meta) = std::fs::metadata(&cache_path) {
            if meta.len() == total_size {
                if let Ok(file) = std::fs::File::open(&cache_path)
                    && let Ok(m) = unsafe { Mmap::map(&file) }
                {
                    let _ = mmap.set(m);
                }
            }
        }

        Ok(Storage::Http { transport, total_size, cache_path, mmap })
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
            Storage::Http { transport, mmap, cache_path, total_size } => {
                // Mmap fast path — promoted by a prior prebuffer or
                // restored at open time from a leftover cache file.
                try_promote_http(cache_path, *total_size, mmap);
                if let Some(m) = mmap.get() {
                    let end = (offset + len) as usize;
                    if end > m.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof,
                            format!("read past end of HTTP cache mmap: offset={offset} len={len} size={}", m.len())));
                    }
                    return Ok(m[offset as usize..end].to_vec());
                }
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
    ///
    /// For `Storage::Cached`, attempts a lazy mmap promotion when
    /// the local cache file is complete but this `Storage` instance
    /// hasn't promoted yet. This handles the cross-instance case:
    /// reader X is opened against URL U; later, *another* `Storage`
    /// instance (e.g. one built by `view.prebuffer_all`) finishes
    /// downloading the cache file. Without lazy promotion, reader X
    /// would never see the zero-copy state and stay on the slow
    /// channel-read path forever.
    pub(crate) fn mmap_base(&self) -> Option<*const u8> {
        match self {
            Storage::Mmap(m) => Some(m.as_ptr()),
            Storage::Cached { channel, mmap } => {
                try_promote_cached(channel, mmap);
                mmap.get().map(|m| m.as_ptr())
            }
            Storage::Http { cache_path, total_size, mmap, .. } => {
                try_promote_http(cache_path, *total_size, mmap);
                mmap.get().map(|m| m.as_ptr())
            }
        }
    }

    /// Borrow a slice if the bytes are locally accessible without
    /// allocation. `Some` for `Mmap` always, for `Cached` once the
    /// file is fully promoted to mmap. `None` otherwise — caller
    /// falls back to `read_bytes`.
    ///
    /// Same lazy-promotion behaviour as [`mmap_base`].
    pub(crate) fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let (start, end) = (offset as usize, (offset + len) as usize);
        match self {
            Storage::Mmap(m) => m.get(start..end),
            Storage::Cached { channel, mmap } => {
                try_promote_cached(channel, mmap);
                mmap.get().and_then(|m| m.get(start..end))
            }
            Storage::Http { cache_path, total_size, mmap, .. } => {
                try_promote_http(cache_path, *total_size, mmap);
                mmap.get().and_then(|m| m.get(start..end))
            }
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

    /// Whether all bytes are locally accessible.
    /// `true` for `Mmap` always; `true` for `Cached` once every
    /// chunk is verified (consults the on-disk `.mrkl` state if
    /// the in-memory state is stale, e.g., a sibling instance
    /// completed it); `true` for `Http` once `prebuffer` has
    /// downloaded the whole file.
    pub(crate) fn is_complete(&self) -> bool {
        match self {
            Storage::Mmap(_) => true,
            Storage::Http { cache_path, total_size, mmap, .. } => {
                try_promote_http(cache_path, *total_size, mmap);
                mmap.get().is_some()
            }
            Storage::Cached { channel, mmap } => {
                try_promote_cached(channel, mmap);
                mmap.get().is_some() || channel.is_complete()
            }
        }
    }

    /// Whether reads avoid network round-trips.
    pub(crate) fn is_local(&self) -> bool {
        match self {
            Storage::Mmap(_) => true,
            Storage::Cached { channel, mmap } => {
                try_promote_cached(channel, mmap);
                mmap.get().is_some()
            }
            Storage::Http { cache_path, total_size, mmap, .. } => {
                try_promote_http(cache_path, *total_size, mmap);
                mmap.get().is_some()
            }
        }
    }

    /// Drive this storage to fully-resident, zero-copy state.
    ///
    /// **Strict contract**: when `Ok(())` returns, every byte is
    /// locally accessible *and* mmap-promoted. There is no variant
    /// where this is a no-op that quietly leaves reads going over
    /// HTTP. Failure is always surfaced, never swallowed.
    ///
    /// - `Storage::Mmap` → already local; returns immediately.
    /// - `Storage::Cached` → downloads + merkle-verifies every chunk;
    ///   strict completion check; mmap-promotes; errors otherwise.
    /// - `Storage::Http` → downloads the full file via HTTP RANGE
    ///   into the configured cache directory (no merkle — no `.mref`
    ///   was published, so we trust the bytes byte-for-byte from the
    ///   server); mmap-promotes. Errors on download or write
    ///   failure.
    pub(crate) fn prebuffer(&self) -> io::Result<()> {
        self.prebuffer_with_progress(|_| {})
    }

    /// Same as [`prebuffer`] with a progress callback fired after
    /// the underlying download completes. Same strict contract:
    /// either every byte is resident on `Ok`, or an `Err` is
    /// returned.
    pub(crate) fn prebuffer_with_progress<F>(&self, cb: F) -> io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        match self {
            Storage::Mmap(_) => Ok(()),
            Storage::Http { transport, total_size, cache_path, mmap } => {
                if mmap.get().is_some() { return Ok(()); }
                http_prebuffer(transport, *total_size, cache_path, mmap, cb)
            }
            Storage::Cached { channel, mmap } => {
                channel.prebuffer_with_progress(cb)?;
                // Strict-completion check: prebuffer downloaded every
                // missing chunk; if any chunk is still unverified
                // (verification mismatch, race with a concurrent
                // writer, anything), surface the failure now.
                if !channel.is_complete() {
                    let valid = channel.valid_count();
                    let total = channel.total_chunks();
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "prebuffer did not complete: {valid}/{total} chunks verified"
                        ),
                    ));
                }
                // Promotion is part of the contract: after Ok(())
                // every read on *this* Storage takes the zero-copy
                // path. Other Storage instances pointed at the same
                // cache file pick up the promotion lazily on their
                // next mmap_slice/mmap_base call.
                if mmap.get().is_none() {
                    let file = std::fs::File::open(channel.cache_path())?;
                    let m = unsafe { Mmap::map(&file) }?;
                    let _ = mmap.set(m);
                }
                Ok(())
            }
        }
    }

    /// Hint that the full mapping will be accessed sequentially.
    /// No-op for non-mmap variants (and for `Cached`/`Http` until
    /// promotion).
    pub(crate) fn advise_sequential(&self) {
        if let Some(m) = self.promoted_mmap() {
            let _ = m.advise(memmap2::Advice::Sequential);
        }
    }

    /// Hint that the full mapping will be accessed in random order.
    pub(crate) fn advise_random(&self) {
        if let Some(m) = self.promoted_mmap() {
            let _ = m.advise(memmap2::Advice::Random);
        }
    }

    /// Internal helper: borrow the promoted mmap, lazy-promoting
    /// the Cached and Http variants if their underlying file is
    /// ready.
    fn promoted_mmap(&self) -> Option<&Mmap> {
        match self {
            Storage::Mmap(m) => Some(m),
            Storage::Cached { channel, mmap } => {
                try_promote_cached(channel, mmap);
                mmap.get()
            }
            Storage::Http { cache_path, total_size, mmap, .. } => {
                try_promote_http(cache_path, *total_size, mmap);
                mmap.get()
            }
        }
    }

    /// `madvise(WILLNEED)` over the byte range. Non-blocking hint.
    pub(crate) fn prefetch_range_bytes(&self, byte_start: u64, byte_end: u64) {
        let mmap = self.promoted_mmap();
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
        let Some(m) = self.promoted_mmap() else { return; };
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
        let Some(m) = self.promoted_mmap() else { return; };
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

    /// Best-effort path to the local file backing this storage. `Some`
    /// when the bytes are reachable through the filesystem — i.e.,
    /// `Storage::Cached` (the cache file) and `Storage::Http` after
    /// `prebuffer` has downloaded the file (or if a prior prebuffer
    /// left the cache file at the expected path).
    ///
    /// For `Storage::Mmap` we don't track the originating path
    /// (the `Mmap` doesn't carry it), so the caller is expected to
    /// remember it from the open arguments. `None` for
    /// `Storage::Http` when no cache file exists yet.
    pub(crate) fn local_path(&self) -> Option<std::path::PathBuf> {
        match self {
            Storage::Cached { channel, .. } => Some(channel.cache_path().to_path_buf()),
            Storage::Http { cache_path, .. } if cache_path.is_file() =>
                Some(cache_path.clone()),
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
            Storage::Http { total_size, mmap, .. } => f.debug_struct("Storage::Http")
                .field("size", total_size)
                .field("promoted", &mmap.get().is_some())
                .finish(),
            Storage::Cached { channel, mmap } => f.debug_struct("Storage::Cached")
                .field("size", &channel.content_size())
                .field("complete", &channel.is_complete())
                .field("promoted", &mmap.get().is_some())
                .finish(),
        }
    }
}

/// Lazy promotion: if the local cache file is fully verified but
/// this `Storage::Cached`'s `OnceLock<Mmap>` is empty, open + mmap
/// the cache file now. Idempotent under races: `OnceLock::set`
/// keeps only one of the concurrent attempts.
///
/// Two completeness signals are checked: the in-memory state on the
/// `CachedChannel` (cheap, takes a Mutex), and — only as a
/// fallback when in-memory says incomplete — the on-disk `.mrkl`
/// state. The latter handles the case where a *sibling* `Storage`
/// instance has populated the cache without notifying this
/// channel's in-memory state. Without this fallback, an early
/// reader opened before `prebuffer` would never see the promotion
/// triggered by another instance.
///
/// Cheap when already promoted (atomic load on `OnceLock::get`).
/// One `Mutex` lock per call until promoted; one extra disk
/// `.mrkl` read per call until promoted *and* the in-memory state
/// is incomplete. After promotion, all calls are O(1) atomic-load.
fn try_promote_cached(
    channel: &CachedChannel,
    mmap: &OnceLock<Mmap>,
) {
    if mmap.get().is_some() { return; }
    let in_memory_complete = channel.is_complete();
    let on_disk_complete = !in_memory_complete
        && crate::merkle::MerkleState::load(channel.state_path())
            .map(|s| s.is_complete())
            .unwrap_or(false);
    if !(in_memory_complete || on_disk_complete) { return; }
    if let Ok(file) = std::fs::File::open(channel.cache_path())
        && let Ok(m) = unsafe { Mmap::map(&file) }
    {
        let _ = mmap.set(m);
    }
}

/// Same as [`try_promote_cached`] but for the merkle-less Http
/// variant: if the cache file at `path` exists with the expected
/// `total_size`, mmap it. Used both at open time and lazily on
/// the read paths so that an Http storage opened *before* a
/// `prebuffer` on a sibling instance picks up the promotion the
/// first time it's read.
fn try_promote_http(
    path: &std::path::Path,
    total_size: u64,
    mmap: &OnceLock<Mmap>,
) {
    if mmap.get().is_some() { return; }
    let Ok(meta) = std::fs::metadata(path) else { return; };
    if meta.len() != total_size { return; }
    if let Ok(file) = std::fs::File::open(path)
        && let Ok(m) = unsafe { Mmap::map(&file) }
    {
        let _ = mmap.set(m);
    }
}

/// Download the full file via HTTP RANGE, atomic-rename into
/// `cache_path`, and mmap-promote. Used by
/// [`Storage::prebuffer_with_progress`] for the `Http` variant.
///
/// We don't have merkle hashes (no `.mref` is published), so the
/// only integrity check is the post-write file size. If the server
/// returns fewer bytes than the HEAD-reported `total_size`, this
/// errors instead of leaving a short file. The download is written
/// to a `.tmp` sibling and atomically renamed so concurrent readers
/// either see the fully-resident file or none at all.
fn http_prebuffer<F>(
    transport: &HttpTransport,
    total_size: u64,
    cache_path: &std::path::Path,
    mmap: &OnceLock<Mmap>,
    mut cb: F,
) -> io::Result<()>
where F: FnMut(&crate::transport::DownloadProgress),
{
    use crate::transport::ChunkedTransport;
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Stream the download in 8 MiB chunks so memory cost is bounded
    // regardless of total size, then assemble the file.
    let chunk_size: u64 = 8 * 1024 * 1024;
    let progress = crate::transport::DownloadProgress::new(
        total_size,
        ((total_size + chunk_size - 1) / chunk_size).max(1) as u32,
    );

    let tmp_path = cache_path.with_extension(format!(
        "{}.tmp",
        cache_path.extension().and_then(|s| s.to_str()).unwrap_or("dl"),
    ));
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path)?;
        let mut written: u64 = 0;
        while written < total_size {
            let want = chunk_size.min(total_size - written);
            let bytes = transport.fetch_range(written, want)?;
            if bytes.len() as u64 != want {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "HTTP prebuffer short read at offset {written}: \
                         expected {want}, got {}", bytes.len()
                    ),
                ));
            }
            f.write_all(&bytes)?;
            written += want;
            progress.add_downloaded_bytes(want);
            progress.increment_completed();
        }
        f.sync_all()?;
    }
    std::fs::rename(&tmp_path, cache_path)?;

    // Verify the written file matches what the server told us and
    // mmap-promote.
    let meta = std::fs::metadata(cache_path)?;
    if meta.len() != total_size {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("HTTP prebuffer size mismatch: expected {total_size}, got {}", meta.len()),
        ));
    }
    let file = std::fs::File::open(cache_path)?;
    let m = unsafe { Mmap::map(&file) }?;
    let _ = mmap.set(m);
    cb(&progress);
    Ok(())
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
