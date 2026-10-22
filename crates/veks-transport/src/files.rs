// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Remote files as their cache copies: [`VerifiedFile`] (a published
//! `.mref` verifies every chunk) and [`ChunkedFile`] (no reference; a
//! tree is recorded as chunks land), each mapped for zero-copy reads
//! once complete; and the [`ByteSource`]s over them — [`RemoteSource`],
//! cached, and [`RangeSource`], uncached — which veks-io's readers read
//! (`srd-crate-boundaries.md` TR-8).
//!
//! Where a copy and its sidecars live is the caller's [`CachePlacement`]
//! (TR-7). Once every chunk is here a file promotes to a mapping, and
//! every read after is a borrow from it; until then a read fetches the
//! chunks it touches. A copy completed by another handle (another
//! process's precache, a sibling in this one) is picked up lazily on the
//! next read.

use std::io;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use memmap2::Mmap;
use url::Url;
use veks_apicore::{ByteSource, DownloadProgress};

use crate::chunk_store::{ChunkStore, complete_size_on_disk};
use crate::merkle::MerkleRef;
use crate::{CachePlacement, CachedChannel, HttpTransport, Transport};

/// Map `path` read-only, when it opens and maps.
fn map(path: &std::path::Path) -> Option<Mmap> {
    let file = std::fs::File::open(path).ok()?;
    // SAFETY: the cache copy is complete and is never written again
    // while mapped: every chunk is valid, and a valid chunk is not
    // fetched twice.
    unsafe { Mmap::map(&file) }.ok()
}

/// `len` bytes at `offset` of a mapping, copied, or the error for a read
/// past its end. `what` names the mapping in the message.
#[inline]
fn copy_out(m: &Mmap, offset: u64, len: u64, what: &str) -> io::Result<Vec<u8>> {
    let end = (offset + len) as usize;
    if end > m.len() {
        return Err(past_end(offset, len, m.len(), what));
    }
    Ok(m[offset as usize..end].to_vec())
}

#[cold]
fn past_end(offset: u64, len: u64, size: usize, what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!("read past end of {what}: offset={offset} len={len} size={size}"),
    )
}

/// The `.mref` published beside `url`, fetched with the read
/// credentials. An error when offline, when the server has none (kind
/// `NotFound`), or when it does not parse.
pub fn fetch_reference(transport: &Transport, url: &Url) -> io::Result<MerkleRef> {
    transport.ensure_online(url.as_str())?;
    let client = transport.client_for(url.as_str())?;
    let mref_url_str = format!("{}.mref", url.as_str());
    let mref_url = Url::parse(&mref_url_str)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("invalid mref URL: {e}")))?;
    // Same read-side auth as every data fetch — without it, a
    // token-protected server 401s the `.mref` probe and every open
    // silently falls back to the unverified chunked path even though a
    // merkle reference is published.
    let resp = transport
        .apply_read_auth(client.get(mref_url.clone()), mref_url.as_str())
        .send()
        .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, format!("mref fetch: {e}")))?
        .error_for_status()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, format!("no .mref for {url}: {e}")))?;
    let mref_bytes = resp.bytes().map_err(|e| io::Error::other(format!("mref read: {e}")))?;
    MerkleRef::from_bytes(&mref_bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("mref parse: {e}")))
}

/// A remote file verified against its published merkle reference,
/// chunk by chunk, as it is read; mapped once every chunk is verified.
pub struct VerifiedFile {
    channel: CachedChannel,
    /// Set exactly once, when the cache is fully verified.
    mmap: OnceLock<Mmap>,
}

impl VerifiedFile {
    /// The copy of `http`'s file verified against `reference`, at
    /// `placement`: resumed from its merkle state when there is one, a
    /// fresh sparse copy otherwise. An error when the state there was
    /// verified against another reference (the upstream changed).
    pub fn open(http: HttpTransport, reference: MerkleRef, placement: &CachePlacement) -> io::Result<Self> {
        let url = http.url().to_string();
        let workers = http.transport().download_concurrency();
        let channel =
            CachedChannel::open(Arc::new(http), reference, placement)?.with_source_url(&url).with_concurrency(workers);
        Ok(Self::from_channel(channel))
    }

    /// The copy at `placement` from its merkle state alone, with no
    /// network: when it is complete, or with `allow_partial` (offline
    /// mode) when it is not. `None` when there is no such copy.
    pub fn open_complete(http: HttpTransport, placement: &CachePlacement, allow_partial: bool) -> Option<Self> {
        let url = http.url().to_string();
        let workers = http.transport().download_concurrency();
        let channel = CachedChannel::open_complete(Arc::new(http), placement, allow_partial)?
            .with_source_url(&url)
            .with_concurrency(workers);
        Some(Self::from_channel(channel))
    }

    /// A file over a channel already open: mapped at once when the
    /// channel is complete.
    pub fn from_channel(channel: CachedChannel) -> Self {
        let mmap = OnceLock::new();
        if channel.is_complete()
            && let Some(m) = map(channel.cache_path())
        {
            let _ = mmap.set(m);
        }
        VerifiedFile { channel, mmap }
    }

    /// The channel the bytes are verified through.
    pub fn channel(&self) -> &CachedChannel {
        &self.channel
    }

    /// Lazy promotion: if the cache file is fully verified but this
    /// file is not mapped yet, map it now. Idempotent under races:
    /// `OnceLock::set` keeps one of the concurrent attempts.
    ///
    /// Two completeness signals are checked: the channel's in-memory
    /// state (cheap, a bitmap count), and — only when that says
    /// incomplete — the on-disk `.mrkl` via
    /// [`CachedChannel::completed_by_sibling`], which sees a copy a
    /// *sibling* completed without telling this channel.
    ///
    /// Cheap when already mapped (one atomic load). Until then each call
    /// costs the in-memory count plus one `stat` of the `.mrkl`; the
    /// on-disk bitset is re-read only when that file changes. This sits
    /// under every record read of a partially cached shard, so it must
    /// never load the state file per call: on a 410 GB shard that file
    /// is 32 MB, and doing so held an explorer to about 50 vectors a
    /// second (guarded by
    /// `verified::tests::reads_on_a_partial_cache_do_not_reload_the_state_file`).
    #[inline]
    fn promote(&self) {
        if self.mmap.get().is_none() {
            self.promote_slow();
        }
    }

    fn promote_slow(&self) {
        if !(self.channel.is_complete() || self.channel.completed_by_sibling()) {
            return;
        }
        if let Some(m) = map(self.channel.cache_path()) {
            let _ = self.mmap.set(m);
        }
    }

    /// The whole file mapped, once it is complete (promoting it now if
    /// it just became so).
    #[inline]
    pub fn mapped(&self) -> Option<&[u8]> {
        self.promote();
        self.mmap.get().map(|m| &m[..])
    }

    /// Whether this file is mapped already, without promoting it.
    pub fn is_mapped(&self) -> bool {
        self.mmap.get().is_some()
    }

    /// `len` bytes at `offset`, fetched and verified if they are not
    /// here.
    #[inline]
    pub fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if let Some(m) = self.mmap.get() {
            return copy_out(m, offset, len, "cache mmap");
        }
        let bytes = self.channel.read(offset, len)?;
        if self.mmap.get().is_none()
            && self.channel.is_complete()
            && let Some(m) = map(self.channel.cache_path())
        {
            let _ = self.mmap.set(m);
        }
        Ok(bytes)
    }

    /// The mapping's base address, once complete.
    #[inline]
    pub fn mmap_base(&self) -> Option<*const u8> {
        self.promote();
        self.mmap.get().map(|m| m.as_ptr())
    }

    /// `len` bytes at `offset`, borrowed, once complete.
    #[inline]
    pub fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        self.promote();
        self.mmap.get().and_then(|m| m.get(offset as usize..(offset + len) as usize))
    }

    /// The file's size in bytes.
    #[inline]
    pub fn total_size(&self) -> u64 {
        self.channel.content_size()
    }

    /// Whether every chunk is verified (counting a sibling's work).
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.promote();
        self.mmap.get().is_some() || self.channel.is_complete()
    }

    /// Whether reads are served from the mapping, with no request.
    #[inline]
    pub fn is_local(&self) -> bool {
        self.promote();
        self.mmap.get().is_some()
    }

    /// The cache copy's path.
    pub fn local_path(&self) -> PathBuf {
        self.channel.cache_path().to_path_buf()
    }

    /// Fetch and verify every missing chunk, reporting as it goes, then
    /// map the file. Strict: `Ok` means every byte is here and mapped.
    pub fn prebuffer_with_progress(&self, cb: &mut dyn FnMut(&DownloadProgress)) -> io::Result<()> {
        self.channel.prebuffer_with_progress(cb)?;
        // Strict-completion check: precache downloaded every missing
        // chunk; if any chunk is still unverified (verification
        // mismatch, race with a concurrent writer, anything), surface
        // the failure now.
        if !self.channel.is_complete() {
            let valid = self.channel.valid_count();
            let total = self.channel.total_chunks();
            return Err(io::Error::other(format!("precache did not complete: {valid}/{total} chunks verified")));
        }
        // Promotion is part of the contract: after Ok(()) every read on
        // *this* file takes the zero-copy path. Other handles on the
        // same cache file pick it up lazily on their next read.
        if self.mmap.get().is_none() {
            let file = std::fs::File::open(self.channel.cache_path())?;
            // SAFETY: as in `map`: the copy is complete and not written again.
            let m = unsafe { Mmap::map(&file) }?;
            let _ = self.mmap.set(m);
        }
        Ok(())
    }

    /// Fetch and verify the chunks covering `[byte_start, byte_end)`,
    /// reporting as it goes. Does not map: the file need not be whole.
    pub fn prebuffer_range_with_progress(
        &self,
        byte_start: u64,
        byte_end: u64,
        cb: &mut dyn FnMut(&DownloadProgress),
    ) -> io::Result<()> {
        self.channel.prebuffer_range_with_progress(byte_start, byte_end, cb)
    }
}

impl std::fmt::Debug for VerifiedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedFile")
            .field("size", &self.channel.content_size())
            .field("complete", &self.channel.is_complete())
            .field("promoted", &self.mmap.get().is_some())
            .finish()
    }
}

/// A remote file with no published reference, fetched in 8 MiB chunks
/// on demand into a sparse cache copy; mapped once every chunk is here.
///
/// The server's bytes are trusted as they arrive (integrity being the
/// connection's TLS); each chunk's hash is recorded as it lands, so once
/// complete the copy has a content identity and a chunk can be checked
/// against what it was.
pub struct ChunkedFile {
    store: Arc<ChunkStore>,
    /// Set once every chunk is here.
    mmap: OnceLock<Mmap>,
}

impl ChunkedFile {
    /// The copy of `http`'s file, `total_size` bytes, at `placement`:
    /// its chunk map loaded when there is one, a fresh sparse copy
    /// otherwise.
    pub fn open(http: HttpTransport, total_size: u64, placement: &CachePlacement) -> io::Result<Self> {
        Ok(Self::from_store(Arc::new(ChunkStore::open(http, total_size, placement)?)))
    }

    /// The copy at `placement` from its chunk map alone, with no
    /// network: when every chunk is here, or with `allow_partial`
    /// (offline mode) when not. `None` when there is no such copy.
    pub fn open_complete(http: HttpTransport, placement: &CachePlacement, allow_partial: bool) -> io::Result<Option<Self>> {
        match complete_size_on_disk(placement, allow_partial) {
            Some(total_size) => Self::open(http, total_size, placement).map(Some),
            None => Ok(None),
        }
    }

    fn from_store(store: Arc<ChunkStore>) -> Self {
        let mmap = OnceLock::new();
        if store.is_complete()
            && let Some(m) = map(store.cache_path())
        {
            let _ = mmap.set(m);
        }
        ChunkedFile { store, mmap }
    }

    /// The chunk store the bytes come through.
    pub fn store(&self) -> &ChunkStore {
        &self.store
    }

    /// The URL the bytes come from.
    pub fn url(&self) -> &Url {
        self.store.url()
    }

    /// Lazy promotion, as [`VerifiedFile`]'s: if every chunk is valid
    /// (the persisted map says so — a sparse file's size proves
    /// nothing), map the cache copy.
    #[inline]
    fn promote(&self) {
        if self.mmap.get().is_none() {
            self.promote_slow();
        }
    }

    fn promote_slow(&self) {
        if !self.store.is_complete() {
            return;
        }
        if let Some(m) = map(self.store.cache_path()) {
            let _ = self.mmap.set(m);
        }
    }

    /// The whole file mapped, once it is complete.
    #[inline]
    pub fn mapped(&self) -> Option<&[u8]> {
        self.promote();
        self.mmap.get().map(|m| &m[..])
    }

    /// Whether this file is mapped already, without promoting it.
    pub fn is_mapped(&self) -> bool {
        self.mmap.get().is_some()
    }

    /// `len` bytes at `offset`, fetched if they are not here.
    #[inline]
    pub fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        // Mapped fast path — at open, if a prior precache filled every
        // chunk, or after the read below fills the last one.
        self.promote();
        if let Some(m) = self.mmap.get() {
            return copy_out(m, offset, len, "HTTP cache mmap");
        }
        // Lazy chunked read: fetch only the chunks covering the range,
        // cache them to disk; map if this read completed the file.
        let bytes = self.store.read(offset, len)?;
        if self.mmap.get().is_none()
            && self.store.is_complete()
            && let Some(m) = map(self.store.cache_path())
        {
            let _ = self.mmap.set(m);
        }
        Ok(bytes)
    }

    /// The mapping's base address, once complete.
    #[inline]
    pub fn mmap_base(&self) -> Option<*const u8> {
        self.promote();
        self.mmap.get().map(|m| m.as_ptr())
    }

    /// `len` bytes at `offset`, borrowed, once complete.
    #[inline]
    pub fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        self.promote();
        self.mmap.get().and_then(|m| m.get(offset as usize..(offset + len) as usize))
    }

    /// The file's size in bytes.
    #[inline]
    pub fn total_size(&self) -> u64 {
        self.store.total_size()
    }

    /// Whether every chunk is here (and the file mapped).
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.promote();
        self.mmap.get().is_some()
    }

    /// Whether reads are served from the mapping, with no request.
    #[inline]
    pub fn is_local(&self) -> bool {
        self.is_complete()
    }

    /// The cache copy's path, once there is a file there.
    pub fn local_path(&self) -> Option<PathBuf> {
        let p = self.store.cache_path();
        p.is_file().then(|| p.to_path_buf())
    }

    /// Fetch every missing chunk, reporting as it goes, then map the
    /// file. Strict: `Ok` means every byte is here and mapped.
    pub fn prebuffer_with_progress(&self, cb: &mut dyn FnMut(&DownloadProgress)) -> io::Result<()> {
        if self.mmap.get().is_some() {
            return Ok(());
        }
        self.store.prebuffer_with_progress(cb)?;
        if !self.store.is_complete() {
            return Err(io::Error::other(format!(
                "HTTP precache did not complete: {}/{} chunks downloaded",
                self.store.valid_count(),
                self.store.total_chunks(),
            )));
        }
        if self.mmap.get().is_none() {
            let file = std::fs::File::open(self.store.cache_path())?;
            // SAFETY: as in `map`: the copy is complete and not written again.
            let m = unsafe { Mmap::map(&file) }?;
            let _ = self.mmap.set(m);
        }
        Ok(())
    }

    /// Fetch the chunks covering `[byte_start, byte_end)`, reporting as
    /// it goes. Does not map.
    pub fn prebuffer_range_with_progress(
        &self,
        byte_start: u64,
        byte_end: u64,
        cb: &mut dyn FnMut(&DownloadProgress),
    ) -> io::Result<()> {
        self.store.prebuffer_range_with_progress(byte_start, byte_end, cb)
    }
}

impl std::fmt::Debug for ChunkedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkedFile")
            .field("size", &self.store.total_size())
            .field("chunks_valid", &self.store.valid_count())
            .field("chunks_total", &self.store.total_chunks())
            .field("promoted", &self.mmap.get().is_some())
            .finish()
    }
}

/// A remote file's cache copy, by how it is kept.
#[derive(Debug)]
pub enum RemoteFile {
    /// Verified against a published `.mref`.
    Verified(VerifiedFile),
    /// Fetched without a reference.
    Chunked(ChunkedFile),
}

impl RemoteFile {
    /// The copy of `url` at `placement`: a complete copy from disk with
    /// no request (or any copy, offline); else verified against the
    /// `.mref` the server publishes; else, when it publishes none,
    /// fetched without one. `url` may be an `s3://` URL.
    pub fn open(transport: &Transport, url: &str, placement: &CachePlacement) -> io::Result<Self> {
        let wire = transport.normalize_url(url);
        let parsed = Url::parse(wire.as_ref()).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("invalid URL '{url}' (translated to '{wire}'): {e}"))
        })?;
        let http = HttpTransport::new(transport, parsed.clone());
        if placement.data.is_file() {
            let allow_partial = transport.is_offline();
            if let Some(f) = VerifiedFile::open_complete(http.clone(), placement, allow_partial) {
                return Ok(RemoteFile::Verified(f));
            }
            if let Some(f) = ChunkedFile::open_complete(http.clone(), placement, allow_partial)? {
                return Ok(RemoteFile::Chunked(f));
            }
        }
        match fetch_reference(transport, &parsed) {
            Ok(reference) => Ok(RemoteFile::Verified(VerifiedFile::open(http, reference, placement)?)),
            Err(_) => {
                let total = http.content_length()?;
                Ok(RemoteFile::Chunked(ChunkedFile::open(http, total, placement)?))
            }
        }
    }
}

/// A remote file read through its cache copy: the [`ByteSource`] a
/// program hands veks-io's readers to read a remote file without
/// vectordata (`srd-crate-boundaries.md` TR-8).
///
/// Reads fetch the chunks they touch; [`precache`](ByteSource::precache)
/// brings the whole file and maps it.
#[derive(Clone, Debug)]
pub struct RemoteSource {
    file: Arc<RemoteFile>,
    transport: Transport,
}

impl RemoteSource {
    /// The file at `url`, cached at `placement` ([`RemoteFile::open`]).
    pub fn open(transport: &Transport, url: &str, placement: &CachePlacement) -> io::Result<Self> {
        Ok(RemoteSource { file: Arc::new(RemoteFile::open(transport, url, placement)?), transport: transport.clone() })
    }

    /// The cache copy behind the source.
    pub fn file(&self) -> &RemoteFile {
        &self.file
    }
}

impl ByteSource for RemoteSource {
    #[inline]
    fn total_size(&self) -> u64 {
        match &*self.file {
            RemoteFile::Verified(f) => f.total_size(),
            RemoteFile::Chunked(f) => f.total_size(),
        }
    }

    fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        match &*self.file {
            RemoteFile::Verified(f) => f.read_bytes(offset, len),
            RemoteFile::Chunked(f) => f.read_bytes(offset, len),
        }
    }

    #[inline]
    fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        match &*self.file {
            RemoteFile::Verified(f) => f.mmap_slice(offset, len),
            RemoteFile::Chunked(f) => f.mmap_slice(offset, len),
        }
    }

    #[inline]
    fn mmap_base(&self) -> Option<*const u8> {
        match &*self.file {
            RemoteFile::Verified(f) => f.mmap_base(),
            RemoteFile::Chunked(f) => f.mmap_base(),
        }
    }

    fn local_path(&self) -> Option<PathBuf> {
        match &*self.file {
            RemoteFile::Verified(f) => Some(f.local_path()),
            RemoteFile::Chunked(f) => f.local_path(),
        }
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn is_complete(&self) -> bool {
        match &*self.file {
            RemoteFile::Verified(f) => f.is_complete(),
            RemoteFile::Chunked(f) => f.is_complete(),
        }
    }

    fn precache(&self) -> io::Result<()> {
        self.prebuffer_with_progress(&mut |_| {})
    }

    fn prebuffer_with_progress(&self, cb: &mut dyn FnMut(&DownloadProgress)) -> io::Result<()> {
        match &*self.file {
            RemoteFile::Verified(f) => f.prebuffer_with_progress(cb),
            RemoteFile::Chunked(f) => f.prebuffer_with_progress(cb),
        }
    }

    /// The sibling published beside `data_source`, with the read
    /// credentials; `None` offline and for anything but a success.
    fn fetch_sibling(&self, data_source: &str, name: &str) -> Option<Vec<u8>> {
        fetch_sibling(&self.transport, data_source, name)
    }
}

/// The file `name` published beside `data_source`, with the read
/// credentials: `None` offline, for a local `data_source`, and for
/// anything but a success.
pub fn fetch_sibling(transport: &Transport, data_source: &str, name: &str) -> Option<Vec<u8>> {
    if transport.is_offline() || !crate::is_remote_url(data_source) {
        return None;
    }
    let url = Url::parse(data_source).ok()?.join(name).ok()?;
    let client = transport.client_for(url.as_str()).ok()?;
    let resp = transport.apply_read_auth(client.get(url.clone()), url.as_str()).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.bytes().ok().map(|b| b.to_vec())
}

/// A remote file read by range requests with no cache: every read is a
/// request. For a few reads of a file not worth keeping (a header, a
/// probe); a reader that reads much should use [`RemoteSource`].
#[derive(Clone, Debug)]
pub struct RangeSource {
    http: Arc<HttpTransport>,
    size: u64,
}

impl RangeSource {
    /// The file at `url` (`s3://` included), its size asked once now.
    pub fn open(transport: &Transport, url: &str) -> io::Result<Self> {
        let wire = transport.normalize_url(url);
        let parsed = Url::parse(wire.as_ref())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("invalid URL '{url}': {e}")))?;
        let http = HttpTransport::new(transport, parsed);
        let size = http.content_length()?;
        Ok(RangeSource { http: Arc::new(http), size })
    }
}

impl ByteSource for RangeSource {
    #[inline]
    fn total_size(&self) -> u64 {
        self.size
    }

    fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if offset.checked_add(len).is_none_or(|end| end > self.size) {
            return Err(past_end_of_source(offset, len, self.size));
        }
        self.http.fetch_range(offset, len)
    }

    #[inline]
    fn mmap_slice(&self, _offset: u64, _len: u64) -> Option<&[u8]> {
        None
    }

    #[inline]
    fn mmap_base(&self) -> Option<*const u8> {
        None
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn is_complete(&self) -> bool {
        false
    }

    fn fetch_sibling(&self, data_source: &str, name: &str) -> Option<Vec<u8>> {
        fetch_sibling(self.http.transport(), data_source, name)
    }

    fn prefetch_pages_bytes(&self, _start: u64, _end: u64, _paged: Option<&AtomicU64>) {}
}

#[cold]
fn past_end_of_source(offset: u64, len: u64, size: u64) -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, format!("read past end: offset={offset} len={len} size={size}"))
}
