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
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::AtomicU64;

use memmap2::Mmap;
use url::Url;

use crate::cache::CachedChannel;
use crate::cache::layout;
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

    /// Remote URL with no published `.mref`. Reads route through a
    /// [`ChunkStore`] that lazily downloads 8 MiB chunks on demand
    /// — first read covering an unmapped chunk fetches it, writes
    /// it into a sparse cache file, and records validity in a
    /// sidecar bitmap. Calling [`precache`] forces every chunk to
    /// fill. When all chunks are valid the cache file is mmap-
    /// promoted so subsequent reads take the zero-copy path.
    ///
    /// No merkle verification — server bytes are trusted byte-
    /// for-byte (the intentional difference from
    /// [`Storage::Cached`], which requires a published `.mref`).
    Http {
        /// Chunked lazy-fill backend. Shared across every
        /// `Storage::Http` instance against the same URL via the
        /// process-wide registry, so concurrent readers benefit
        /// from a single bitmap and a single in-flight dedup map.
        chunks: Arc<crate::chunked_http::ChunkStore>,
        /// Promoted-mmap view of `chunks.cache_path()`. Set by
        /// `precache`/`prebuffer_with_progress` once every chunk
        /// is filled, or at open time when an already-complete
        /// cache file (with bitmap) is detected.
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

/// Process-wide registry of `Storage` instances keyed on the
/// canonical source identity (URL string for remote, absolute
/// canonicalised path for local). Multiple opens against the same
/// source return the same `Arc<Storage>` so all readers share one
/// `CachedChannel`, one `MerkleState`, one cache file handle, and
/// one mmap promotion.
///
/// Without this, two `Storage::Cached` against the same URL would
/// each have independent state, and concurrent reads could observe
/// the cache file mid-write (the .mrkl saved as "chunk valid" by
/// thread A while thread B's open is processing — A's writes haven't
/// been flushed through B's separate file handle yet, so B reads
/// the pre-allocated zero bytes and fails with "invalid dimension 0").
///
/// Entries are stored as `Weak<Storage>` so a `Storage` is freed
/// once every reader against it drops. The next open re-creates it.
fn registry() -> &'static std::sync::Mutex<std::collections::HashMap<String, std::sync::Weak<Storage>>> {
    static REGISTRY: OnceLock<std::sync::Mutex<std::collections::HashMap<String, std::sync::Weak<Storage>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Per-key open-in-progress coordinator. When N threads concurrently
/// call `Storage::open(source)` against the same source, only the
/// first thread actually does the I/O; the rest block on the
/// `Condvar` and pick up the result without re-doing the work.
///
/// Without this, parallel opens against the same URL would each do
/// their own .mref fetch + chunk-0 download, wasting ~N round trips
/// for the same single resource (and producing the connection
/// storm a downstream consumer's flame graph showed).
struct OpenInFlight {
    result: std::sync::Mutex<Option<Result<Arc<Storage>, String>>>,
    notify: std::sync::Condvar,
}

fn in_flight_opens() -> &'static std::sync::Mutex<std::collections::HashMap<String, Arc<OpenInFlight>>> {
    static MAP: OnceLock<std::sync::Mutex<std::collections::HashMap<String, Arc<OpenInFlight>>>> = OnceLock::new();
    MAP.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Compute the registry key for a source string. URLs key on the
/// URL itself (after parse round-trip for normalisation). Local
/// paths key on canonicalised absolute path so `./foo` and the
/// equivalent absolute path map to the same instance. `file://`
/// URIs are normalised to their plain filesystem path before
/// keying so a catalog entry that references a local file via
/// `file:///x` shares an Arc with an explicit `/x` open. `s3://`
/// URLs are normalised through the transport translation layer
/// so an `s3://` open shares its Arc with the HTTPS form that
/// the actual fetch uses.
fn registry_key(source: &str) -> String {
    if crate::transport::is_remote_url(source) {
        let translated = crate::transport::normalize_remote_url(source);
        match Url::parse(translated.as_ref()) {
            Ok(u) => u.to_string(),
            Err(_) => translated.into_owned(),
        }
    } else {
        let local = local_source_path(source);
        std::fs::canonicalize(local.as_ref())
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| local.into_owned())
    }
}

/// Convert a source string to a plain filesystem path string. A
/// `file://` URI is stripped to its path component (`file:///x/y`
/// → `/x/y`); everything else passes through unchanged. The point
/// is to give a single short-circuit for "this is a local file"
/// regardless of which syntactic form the catalog uses.
pub(crate) fn local_source_path(source: &str) -> std::borrow::Cow<'_, str> {
    if let Some(rest) = source.strip_prefix("file://") {
        // file:///abs/path → /abs/path
        // file://localhost/abs/path → /abs/path  (host stripped)
        if let Some(after_root) = rest.strip_prefix('/') {
            return std::borrow::Cow::Owned(format!("/{after_root}"));
        }
        if let Some(slash) = rest.find('/') {
            return std::borrow::Cow::Owned(rest[slash..].to_string());
        }
        return std::borrow::Cow::Owned(format!("/{rest}"));
    }
    std::borrow::Cow::Borrowed(source)
}

/// Concrete decision of where bytes for a particular open land on
/// disk. Built once by [`layout_for_url`] (the URL-only path) and
/// shared between the `Storage::Cached` and `Storage::Http` opens
/// so they both write into the same natural-layout dataset
/// directory.
pub(crate) struct LayoutChoice {
    /// Per-dataset cache directory: `<cache_root>/<dataset>/`. Holds
    /// the data files and the per-dataset `origin.json`.
    pub dataset_dir: PathBuf,
    /// Data file path relative to `dataset_dir`. May contain `/` if
    /// the dataset's view declares nested layout
    /// (`profiles/1m/base.fvec`). Subdirs are created on demand.
    pub file_relpath: String,
    /// URL recorded in `<dataset_dir>/origin.json`. Subsequent
    /// opens must produce the same `origin_source` or the open
    /// fails with [`crate::cache::layout::OriginMismatch`] —
    /// that's how a moved catalog is migrated by user-visible
    /// `origin.json` edit instead of an opaque cache surgery.
    pub origin_source: String,
}

/// Derive a [`LayoutChoice`] from a URL for URL-only opens (no
/// catalog context). The dataset directory is the URL's host plus
/// the directory portion of the path; the file relpath is the URL
/// path's basename; `origin.json` records the URL's parent
/// directory (so a sibling file under the same parent shares the
/// dataset directory naturally).
///
/// Examples:
///   - `https://example.com/datasets/sift1m/base.fvec`
///     → dataset_dir: `<cache>/example.com/datasets/sift1m`
///     → file_relpath: `base.fvec`
///     → origin: `https://example.com/datasets/sift1m/`
///   - `https://example.com/data.fvec`
///     → dataset_dir: `<cache>/example.com`
///     → file_relpath: `data.fvec`
///     → origin: `https://example.com/`
pub(crate) fn layout_for_url(url: &Url) -> io::Result<LayoutChoice> {
    let cache_root = crate::settings::cache_dir()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
    let host = url.host_str().unwrap_or("_remote");
    // Include explicit port in the authority component so two
    // servers on the same host (different ports) — common in
    // tests, also possible in production — land in distinct
    // dataset directories. Default ports (80 / 443) are
    // omitted so a URL with an implicit port and an equivalent
    // explicit-default URL converge.
    let authority = match url.port() {
        Some(port) => format!("{host}_{port}"),
        None => host.to_string(),
    };
    let raw_path = url.path().trim_start_matches('/');
    let (parent, basename) = match raw_path.rsplit_once('/') {
        Some((p, b)) if !b.is_empty() => (p, b),
        _ => ("", raw_path),
    };
    let basename = if basename.is_empty() { "data" } else { basename };
    let mut dataset_dir = cache_root.join(authority);
    if !parent.is_empty() {
        for segment in parent.split('/') {
            dataset_dir.push(segment);
        }
    }
    // Origin = URL with the basename trimmed off, preserving scheme
    // + host + port so a sibling URL under the same parent path
    // canonicalises identically.
    let mut origin = url.clone();
    let new_path = if parent.is_empty() {
        "/".to_string()
    } else {
        format!("/{parent}/")
    };
    origin.set_path(&new_path);
    origin.set_query(None);
    origin.set_fragment(None);
    Ok(LayoutChoice {
        dataset_dir,
        file_relpath: basename.to_string(),
        origin_source: origin.to_string(),
    })
}

impl Storage {
    /// Open from a path or URL string. Returns a process-wide shared
    /// `Arc<Storage>` per source — concurrent opens against the same
    /// URL or path get the same instance, so they share the cached
    /// channel, file handle, mmap promotion, and download state.
    /// This is what makes "many threads reading the same dataset"
    /// safe by construction: there is no possible second writer to
    /// race with.
    ///
    /// Per-source serialization: when N threads call `open()` for
    /// the same source concurrently, only the first thread actually
    /// performs the open I/O (`.mref` fetch, cache file open, etc).
    /// The rest block on a per-key condvar and clone the resulting
    /// `Arc<Storage>` without re-doing the work.
    pub(crate) fn open(source: &str) -> io::Result<Arc<Self>> {
        Self::open_inner(source, None)
    }

    /// Catalog-aware open. The view layer routes every facet through
    /// here so the cache lands at `<cache_root>/<dataset_name>/
    /// <file_relpath>` (stable across catalog moves) and
    /// `origin.json` records `catalog_source` (editable by the user
    /// when the catalog moves but the bytes are the same).
    ///
    /// `fetch_url` is the actual byte source. `dataset_name` +
    /// `file_relpath` are the natural cache coordinates. The
    /// pre-flight origin check (against `catalog_source`) runs
    /// *before* the registry lookup so a mismatch fails fast even
    /// if the URL is already in the process-wide cache.
    pub(crate) fn open_layered(
        fetch_url: &str,
        dataset_name: &str,
        file_relpath: &str,
        catalog_source: &str,
    ) -> io::Result<Arc<Self>> {
        // Local sources route to `open_path_uncached` (plain mmap) —
        // there is no cache directory to lay out for them. Skip the
        // dataset-dir + origin.json setup that would otherwise leave
        // an empty `<cache>/<dataset_name>/origin.json` behind. This
        // is what was producing the `.tmp*` debris: every test
        // fixture that built a TestDataGroup from a tempdir's
        // dataset.yaml routed through here, the tempdir basename
        // became `dataset_name`, and a stub origin file landed under
        // the real cache root even though the actual bytes were
        // mmap'd locally.
        if !crate::transport::is_remote_url(fetch_url) {
            let _ = (dataset_name, file_relpath, catalog_source);
            return Self::open(fetch_url);
        }
        let cache_root = crate::settings::cache_dir()
            .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
        let dataset_dir = layout::dataset_cache_dir(&cache_root, dataset_name);
        // Pre-flight: surface OriginMismatch before anything else.
        layout::verify_or_record_origin(&dataset_dir, catalog_source)
            .map_err(io::Error::from)?;
        let layout = LayoutChoice {
            dataset_dir,
            file_relpath: file_relpath.to_string(),
            origin_source: catalog_source.to_string(),
        };
        Self::open_inner(fetch_url, Some(layout))
    }

    /// Common registry-coordinated open. `layout_override`, when set,
    /// is passed to [`Self::open_url_uncached`] instead of the
    /// URL-derived layout — that's how `open_layered` plumbs the
    /// catalog-anchored dataset path through this routine without
    /// duplicating the in-flight + condvar machinery.
    fn open_inner(source: &str, layout_override: Option<LayoutChoice>) -> io::Result<Arc<Self>> {
        let key = registry_key(source);

        // Fast path: already in registry, still alive.
        if let Some(arc) = registry().lock().unwrap().get(&key).and_then(|w| w.upgrade()) {
            return Ok(arc);
        }

        // Per-key coordinator: insert a pending entry if we're
        // first; otherwise wait for the in-flight open to complete.
        let (in_flight, we_are_opener) = {
            let mut map = in_flight_opens().lock().unwrap();
            if let Some(existing) = map.get(&key).cloned() {
                (existing, false)
            } else {
                let entry = Arc::new(OpenInFlight {
                    result: std::sync::Mutex::new(None),
                    notify: std::sync::Condvar::new(),
                });
                map.insert(key.clone(), Arc::clone(&entry));
                (entry, true)
            }
        };

        if !we_are_opener {
            // Block until the chosen opener publishes a result.
            let mut result = in_flight.result.lock().unwrap();
            while result.is_none() {
                result = in_flight.notify.wait(result).unwrap();
            }
            return match result.as_ref().unwrap() {
                Ok(arc) => Ok(Arc::clone(arc)),
                Err(msg) => Err(io::Error::new(io::ErrorKind::Other, msg.clone())),
            };
        }

        // We're the chosen opener. Do the open exactly once.
        let opened: io::Result<Arc<Storage>> = (|| {
            // Recheck registry: another thread may have inserted
            // and dropped between our fast-path check and now.
            if let Some(arc) = registry().lock().unwrap().get(&key).and_then(|w| w.upgrade()) {
                return Ok(arc);
            }
            let storage = if crate::transport::is_remote_url(source) {
                // `s3://bucket/key` is rewritten to its
                // virtual-hosted HTTPS form here — the rest of the
                // transport pipeline never sees s3-scheme URLs.
                let translated = crate::transport::normalize_remote_url(source);
                let url = Url::parse(translated.as_ref())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("invalid URL '{source}' (translated to '{translated}'): {e}")))?;
                Self::open_url_uncached(url, layout_override)?
            } else {
                // `file://` URIs and bare paths both open as local
                // mmap — `local_source_path` normalises both to a
                // filesystem path string before we hand it to Path.
                // Layout override is meaningless for local mmap
                // (no cache directory involved), so it's dropped.
                let _ = layout_override;
                let local = local_source_path(source);
                Self::open_path_uncached(Path::new(local.as_ref()))?
            };
            let arc = Arc::new(storage);
            registry().lock().unwrap().insert(key.clone(), Arc::downgrade(&arc));
            Ok(arc)
        })();

        // Publish result and notify waiters before returning.
        {
            let mut result = in_flight.result.lock().unwrap();
            *result = Some(match &opened {
                Ok(arc) => Ok(Arc::clone(arc)),
                Err(e) => Err(e.to_string()),
            });
            in_flight.notify.notify_all();
        }
        // Remove the in-flight entry so future opens (after Arc
        // drop) can re-trigger the open path.
        in_flight_opens().lock().unwrap().remove(&key);
        opened
    }

    /// Open a local file by mmap, returned via the shared registry.
    pub(crate) fn open_path(path: &Path) -> io::Result<Arc<Self>> {
        Self::open(path.to_str().ok_or_else(|| io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("non-UTF8 path: {}", path.display()),
        ))?)
    }

    /// Crate-internal: build a fresh local-file `Storage::Mmap` (no
    /// registry lookup). Used by `Storage::open` after a registry miss.
    fn open_path_uncached(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Storage::Mmap(mmap))
    }

    /// Open a remote URL via the shared registry.
    pub(crate) fn open_url(url: Url) -> io::Result<Arc<Self>> {
        Self::open(url.as_str())
    }

    /// Crate-internal: open a fresh URL-backed `Storage` (no
    /// registry lookup). Tries `.mref`-cached first, falls back to
    /// direct HTTP. Used by `Storage::open` after a registry miss.
    fn open_url_uncached(url: Url, layout_override: Option<LayoutChoice>) -> io::Result<Self> {
        let layout = match layout_override {
            Some(l) => l,
            None => layout_for_url(&url)?,
        };
        match Self::open_url_cached(url.clone(), &layout) {
            Ok(s) => Ok(s),
            Err(_) => Self::open_url_http(url, &layout),
        }
    }

    /// Open a remote URL through the merkle-cache path. Requires a
    /// published `.mref`. Crate-internal; callers go through
    /// `open_url` so the fallback path runs uniformly.
    pub(crate) fn open_url_cached(url: Url, layout: &LayoutChoice) -> io::Result<Self> {
        let client = crate::transport::shared_client();
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

        layout::verify_or_record_origin(&layout.dataset_dir, &layout.origin_source)
            .map_err(io::Error::from)?;
        let transport = HttpTransport::with_client(client, url);
        let channel = CachedChannel::open(
            Arc::new(transport),
            reference,
            &layout.dataset_dir,
            &layout.file_relpath,
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
    /// runs [`precache`], which downloads the whole file into
    /// `<dataset_dir>/<file_relpath>` (per the natural layout) and
    /// promotes to mmap.
    ///
    /// At open time, if the cache file already exists with the
    /// expected size (from a prior precache), this constructor
    /// mmap-promotes immediately so the new `Storage::Http` starts
    /// in the zero-copy state.
    pub(crate) fn open_url_http(url: Url, layout: &LayoutChoice) -> io::Result<Self> {
        layout::verify_or_record_origin(&layout.dataset_dir, &layout.origin_source)
            .map_err(io::Error::from)?;
        let cache_path = layout.dataset_dir.join(&layout.file_relpath);

        let transport = HttpTransport::new(url);
        let total_size = ChunkedTransportExt::content_length_for(&transport)?;

        let chunks = Arc::new(crate::chunked_http::ChunkStore::open(
            transport, total_size, cache_path)?);

        let mmap = OnceLock::new();
        if chunks.is_complete()
            && let Ok(file) = std::fs::File::open(chunks.cache_path())
            && let Ok(m) = unsafe { Mmap::map(&file) }
        {
            let _ = mmap.set(m);
        }

        Ok(Storage::Http { chunks, mmap })
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
            Storage::Http { chunks, mmap } => {
                // Mmap fast path — promoted at open time if a
                // prior precache populated every chunk, or after
                // the chunked reader fills the last chunk below.
                try_promote_http(chunks, mmap);
                if let Some(m) = mmap.get() {
                    let end = (offset + len) as usize;
                    if end > m.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof,
                            format!("read past end of HTTP cache mmap: offset={offset} len={len} size={}", m.len())));
                    }
                    return Ok(m[offset as usize..end].to_vec());
                }
                // Lazy chunked read — fetches only the chunks
                // covering the requested range, caches them to
                // disk. If this read happened to complete the
                // file, promote to mmap so the next read takes
                // the zero-copy path.
                let bytes = chunks.read(offset, len)?;
                if mmap.get().is_none() && chunks.is_complete()
                    && let Ok(file) = std::fs::File::open(chunks.cache_path())
                    && let Ok(m) = unsafe { Mmap::map(&file) }
                {
                    let _ = mmap.set(m);
                }
                Ok(bytes)
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
            Storage::Http { chunks, mmap } => {
                try_promote_http(chunks, mmap);
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
            Storage::Http { chunks, mmap } => {
                try_promote_http(chunks, mmap);
                mmap.get().and_then(|m| m.get(start..end))
            }
        }
    }

    /// Total size in bytes.
    pub(crate) fn total_size(&self) -> u64 {
        match self {
            Storage::Mmap(m) => m.len() as u64,
            Storage::Http { chunks, .. } => chunks.total_size(),
            Storage::Cached { channel, .. } => channel.content_size(),
        }
    }

    /// Whether all bytes are locally accessible.
    /// `true` for `Mmap` always; `true` for `Cached` once every
    /// chunk is verified (consults the on-disk `.mrkl` state if
    /// the in-memory state is stale, e.g., a sibling instance
    /// completed it); `true` for `Http` once `precache` has
    /// downloaded the whole file.
    pub(crate) fn is_complete(&self) -> bool {
        match self {
            Storage::Mmap(_) => true,
            Storage::Http { chunks, mmap } => {
                try_promote_http(chunks, mmap);
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
            Storage::Http { chunks, mmap } => {
                try_promote_http(chunks, mmap);
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
    pub(crate) fn precache(&self) -> io::Result<()> {
        self.prebuffer_with_progress(|_| {})
    }

    /// Drive the storage to fully-resident state over the byte range
    /// `[byte_start, byte_end)`. Same strict-contract / progress
    /// semantics as [`prebuffer_with_progress`], but only the chunks
    /// covering that range are fetched. Used by the view layer to
    /// honor profile windows so a windowed profile against a large
    /// base file doesn't fetch the whole file.
    ///
    /// `byte_end` is clamped to `total_size()`. An empty range
    /// (`byte_start >= byte_end`) fires the completion callback once
    /// and returns `Ok(())`. mmap promotion is *not* attempted on the
    /// ranged path — the file isn't fully resident, so the mmap
    /// promotion contract (every read takes the zero-copy path)
    /// doesn't apply. Promotion still happens lazily on later reads
    /// or on a subsequent full prebuffer.
    pub(crate) fn prebuffer_range_with_progress<F>(
        &self,
        byte_start: u64,
        byte_end: u64,
        cb: F,
    ) -> io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        match self {
            // Local mmap is already fully resident — fire one
            // completion event so callers' meters land at 100%.
            Storage::Mmap(m) => {
                let mut cb = cb;
                cb(&crate::transport::DownloadProgress::new(m.len() as u64, 0));
                Ok(())
            }
            Storage::Http { chunks, mmap: _ } => {
                chunks.prebuffer_range_with_progress(byte_start, byte_end, cb)
            }
            Storage::Cached { channel, mmap: _ } => {
                channel.prebuffer_range_with_progress(byte_start, byte_end, cb)
            }
        }
    }

    /// Same as [`precache`] with a progress callback fired in
    /// flight — once at start (`downloaded_bytes == 0`) and then
    /// after each completed chunk, so UIs can render a live meter
    /// rather than wait for completion. Same strict contract:
    /// either every byte is resident on `Ok`, or an `Err` is
    /// returned.
    pub(crate) fn prebuffer_with_progress<F>(&self, cb: F) -> io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        match self {
            Storage::Mmap(_) => Ok(()),
            Storage::Http { chunks, mmap } => {
                if mmap.get().is_some() { return Ok(()); }
                chunks.prebuffer_with_progress(cb)?;
                if !chunks.is_complete() {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "HTTP precache did not complete: {}/{} chunks downloaded",
                            chunks.valid_count(), chunks.total_chunks(),
                        ),
                    ));
                }
                if mmap.get().is_none() {
                    let file = std::fs::File::open(chunks.cache_path())?;
                    let m = unsafe { Mmap::map(&file) }?;
                    let _ = mmap.set(m);
                }
                Ok(())
            }
            Storage::Cached { channel, mmap } => {
                channel.prebuffer_with_progress(cb)?;
                // Strict-completion check: precache downloaded every
                // missing chunk; if any chunk is still unverified
                // (verification mismatch, race with a concurrent
                // writer, anything), surface the failure now.
                if !channel.is_complete() {
                    let valid = channel.valid_count();
                    let total = channel.total_chunks();
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "precache did not complete: {valid}/{total} chunks verified"
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
    /// promotion). No-op on non-Unix targets — `memmap2::Mmap::advise`
    /// is Unix-only (mirrors the `release_range_bytes` pattern).
    pub(crate) fn advise_sequential(&self) {
        #[cfg(unix)]
        if let Some(m) = self.promoted_mmap() {
            let _ = m.advise(memmap2::Advice::Sequential);
        }
        #[cfg(not(unix))]
        let _ = self;
    }

    /// Hint that the full mapping will be accessed in random order.
    /// No-op on non-Unix targets.
    pub(crate) fn advise_random(&self) {
        #[cfg(unix)]
        if let Some(m) = self.promoted_mmap() {
            let _ = m.advise(memmap2::Advice::Random);
        }
        #[cfg(not(unix))]
        let _ = self;
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
            Storage::Http { chunks, mmap } => {
                try_promote_http(chunks, mmap);
                mmap.get()
            }
        }
    }

    /// `madvise(WILLNEED)` over the byte range. Non-blocking hint.
    /// No-op on non-Unix targets — `Mmap::advise_range` is Unix-only.
    pub(crate) fn prefetch_range_bytes(&self, byte_start: u64, byte_end: u64) {
        #[cfg(unix)]
        {
            let mmap = self.promoted_mmap();
            if let Some(m) = mmap {
                let start = byte_start as usize;
                let end = (byte_end as usize).min(m.len());
                if end > start {
                    let _ = m.advise_range(memmap2::Advice::WillNeed, start, end - start);
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (self, byte_start, byte_end);
        }
    }

    /// `madvise(DONTNEED)` over the byte range. Page-aligns start UP
    /// and end DOWN — sub-page slack stays mapped, but on a
    /// streaming scan that's negligible compared to the megabyte-or-
    /// larger windows callers pass in. Without alignment, madvise
    /// rejects with EINVAL silently.
    pub(crate) fn release_range_bytes(&self, byte_start: u64, byte_end: u64) {
        // `m` is bound only inside the `#[cfg(unix)]` block — leading
        // underscore keeps the non-Unix build clean of "unused
        // variable" warnings.
        let Some(_m) = self.promoted_mmap() else { return; };
        if byte_start >= byte_end { return; }

        #[cfg(unix)]
        unsafe {
            let m = _m;
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
    /// The `madvise` portion is Unix-only; the volatile-read fallback
    /// still executes on all platforms (it's the load-bearing part of
    /// the prefetch — `madvise` is just a hint).
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
        #[cfg(unix)]
        {
            let _ = m.advise_range(memmap2::Advice::WillNeed, start, end - start);
        }

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
    /// `precache` has downloaded the file (or if a prior precache
    /// left the cache file at the expected path).
    ///
    /// For `Storage::Mmap` we don't track the originating path
    /// (the `Mmap` doesn't carry it), so the caller is expected to
    /// remember it from the open arguments. `None` for
    /// `Storage::Http` when no cache file exists yet.
    pub(crate) fn local_path(&self) -> Option<std::path::PathBuf> {
        match self {
            Storage::Cached { channel, .. } => Some(channel.cache_path().to_path_buf()),
            Storage::Http { chunks, .. } if chunks.cache_path().is_file() =>
                Some(chunks.cache_path().to_path_buf()),
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
            Storage::Http { chunks, mmap } => f.debug_struct("Storage::Http")
                .field("size", &chunks.total_size())
                .field("chunks_valid", &chunks.valid_count())
                .field("chunks_total", &chunks.total_chunks())
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
/// reader opened before `precache` would never see the promotion
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
/// variant: if the underlying [`ChunkStore`] reports every chunk
/// is valid, mmap the cache file. Used both at open time and
/// lazily on the read paths so that an Http storage opened
/// *before* a `precache` on a sibling instance picks up the
/// promotion the first time it's read.
///
/// Note: we no longer rely on a bare file-size check —
/// `chunks.is_complete()` consults the persisted chunk bitmap,
/// which is the only safe signal when partial-fill cache files
/// may exist on disk (a sparse file with holes still reports
/// `total_size` from the filesystem).
fn try_promote_http(
    chunks: &crate::chunked_http::ChunkStore,
    mmap: &OnceLock<Mmap>,
) {
    if mmap.get().is_some() { return; }
    if !chunks.is_complete() { return; }
    if let Ok(file) = std::fs::File::open(chunks.cache_path())
        && let Ok(m) = unsafe { Mmap::map(&file) }
    {
        let _ = mmap.set(m);
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
