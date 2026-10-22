// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Internal. Byte-level storage abstraction. **Crate-private.**
//!
//! `Storage` hides the transport choice (local mmap, direct HTTP RANGE,
//! merkle-cached HTTP-with-mmap-promotion) behind a single byte-oriented
//! interface. Shape adapters (`XvecReader`, `ScalarReader`,
//! `IndexedVvecReader`) are parameterised on `Arc<Storage>` so they share one
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
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use memmap2::Mmap;
use url::Url;

use veks_transport::{ChunkedFile, VerifiedFile};

use crate::cache::layout;
use crate::merkle::MerkleRef;

/// Byte-oriented storage over an opaque transport.
///
/// Variants are deliberately not exposed: `pub(crate)` on the enum and
/// every constructor. Public reader types embed `Arc<Storage>` and
/// expose only shape-aware methods.
/// What [`Storage::revalidate`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Revalidation {
    /// The local copy is what the upstream publishes, or is local.
    Current,
    /// The upstream could not be reached; the local copy stands.
    Unreachable,
}

pub(crate) enum Storage {
    /// Local file, mmap'd. Zero-copy, no fallback.
    Mmap {
        mmap: Mmap,
        /// The file this was opened from.
        ///
        /// Carried because sidecars belong beside the file the storage
        /// actually opened, and only the storage knows which that is.
        /// A caller-supplied source string may be relative to some
        /// other directory — a `dataset.yaml` entry is relative to the
        /// dataset root, not the process CWD — so resolving an
        /// `IDXFOR__` path from it looks in the wrong place and, worse,
        /// *writes* the rebuilt index there.
        path: std::path::PathBuf,
    },

    /// Remote URL with no published `.mref`: veks-transport's
    /// [`ChunkedFile`], which lazily downloads 8 MiB chunks on demand —
    /// the first read covering a missing chunk fetches it, writes it
    /// into a sparse cache file, and records it in the `.chunks`
    /// sidecar. [`precache`](Self::precache) fills every chunk. When
    /// all chunks are valid the cache file is mapped, so later reads
    /// take the zero-copy path.
    ///
    /// No merkle verification — server bytes are trusted byte-
    /// for-byte (the intentional difference from
    /// [`Storage::Cached`], which requires a published `.mref`). One
    /// instance per URL (the process-wide registry), so concurrent
    /// readers share one bitmap and one in-flight dedup map.
    Http(ChunkedFile),

    /// Remote URL with a published `.mref`: veks-transport's
    /// [`VerifiedFile`], which downloads and merkle-verifies chunks on
    /// demand into the cache file. When every chunk is verified, it is
    /// mapped, so later reads are zero-copy.
    Cached(VerifiedFile),

    /// A multi-file series presented as one byte space (SH-38).
    ///
    /// The concatenation of a uniform facet's shards **is** the
    /// facet's byte stream — records are fixed width and never
    /// straddle a shard (SH-13) — so a reader that asks this for
    /// bytes at an offset gets the same answer it would from the
    /// unsharded file. That is what lets a facet be split without
    /// every reader learning about shards.
    ///
    /// What it deliberately cannot do is hand back a single base
    /// pointer: there is no one mapping. [`Storage::mmap_base`]
    /// returns `None` here, and callers that need zero-copy work
    /// per-part through [`Storage::parts`] instead of pretending the
    /// shards are contiguous in memory.
    Series {
        /// The shards, in ordinal order.
        parts: Vec<Arc<Storage>>,
        /// Byte offset at which each part begins in the joined space.
        /// One entry per part, ascending, first is zero.
        starts: Vec<u64>,
        /// Sum of every part's size.
        total: u64,
    },
}

/// A local file's merkle root at the 8 MiB chunk size, computed once
/// per process for each (path, size, mtime) and kept.
fn local_root(path: &Path, len: u64, progress: &mut dyn FnMut(u64, u64)) -> io::Result<[u8; 32]> {
    static ROOTS: std::sync::Mutex<Vec<((std::path::PathBuf, u64, Option<std::time::SystemTime>), [u8; 32])>> =
        std::sync::Mutex::new(Vec::new());
    let mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
    let key = (path.to_path_buf(), len, mtime);
    if let Some((_, root)) = ROOTS.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key) {
        return Ok(*root);
    }
    progress(0, len);
    let tree = crate::merkle::MerkleRef::from_path_with_progress(path, veks_transport::DEFAULT_CHUNK_SIZE, |done| progress(done, len))?;
    let root = *tree.root_hash();
    ROOTS.lock().unwrap_or_else(|e| e.into_inner()).push((key, root));
    Ok(root)
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
fn registry()
-> &'static std::sync::Mutex<std::collections::HashMap<String, std::sync::Weak<Storage>>> {
    static REGISTRY: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, std::sync::Weak<Storage>>>,
    > = OnceLock::new();
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

fn in_flight_opens()
-> &'static std::sync::Mutex<std::collections::HashMap<String, Arc<OpenInFlight>>> {
    static MAP: OnceLock<std::sync::Mutex<std::collections::HashMap<String, Arc<OpenInFlight>>>> =
        OnceLock::new();
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
        let local = crate::location::file_uri_path(source);
        std::fs::canonicalize(local.as_ref())
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| local.into_owned())
    }
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

impl LayoutChoice {
    /// Where the file's cache copy and sidecars live, by the cache
    /// layout's naming ([`layout::placement`]).
    pub(crate) fn placement(&self) -> veks_transport::CachePlacement {
        layout::placement(&self.dataset_dir, &self.file_relpath)
    }
}

/// Derive a [`LayoutChoice`] from a URL for URL-only opens (no
/// catalog context). The dataset directory is the URL's host plus
/// the directory portion of the path; the file relpath is the URL
/// path's basename; `origin.json` records the URL's parent
/// directory (so a sibling file under the same parent shares the
/// dataset directory naturally).
///
/// Examples:
///   - `https://example.com/datasets/vecs1m/base.fvec`
///     → dataset_dir: `<cache>/example.com/datasets/vecs1m`
///     → file_relpath: `base.fvec`
///     → origin: `https://example.com/datasets/vecs1m/`
///   - `https://example.com/data.fvec`
///     → dataset_dir: `<cache>/example.com`
///     → file_relpath: `data.fvec`
///     → origin: `https://example.com/`
pub(crate) fn layout_for_url(url: &Url) -> io::Result<LayoutChoice> {
    let cache_root = crate::settings::data_cache_dir()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
    // A file of a dataset already cached under its own name belongs in
    // that dataset's directory: one cache file per remote file, however
    // it is opened.
    if let Some((dataset_dir, origin_source, file_relpath)) =
        layout::dataset_dir_covering(&cache_root, url)
    {
        return Ok(LayoutChoice {
            dataset_dir,
            file_relpath,
            origin_source,
        });
    }
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
    let basename = if basename.is_empty() {
        "data"
    } else {
        basename
    };
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
        let cache_root = crate::settings::data_cache_dir()
            .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
        let dataset_dir = layout::dataset_cache_dir(&cache_root, dataset_name);
        // Pre-flight: surface OriginMismatch before anything else.
        layout::verify_or_record_origin(&dataset_dir, catalog_source).map_err(io::Error::from)?;
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
        if let Some(arc) = registry()
            .lock()
            .unwrap()
            .get(&key)
            .and_then(|w| w.upgrade())
        {
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
                Err(msg) => Err(io::Error::other(msg.clone())),
            };
        }

        // We're the chosen opener. Do the open exactly once.
        let opened: io::Result<Arc<Storage>> = (|| {
            // Recheck registry: another thread may have inserted
            // and dropped between our fast-path check and now.
            if let Some(arc) = registry()
                .lock()
                .unwrap()
                .get(&key)
                .and_then(|w| w.upgrade())
            {
                return Ok(arc);
            }
            let storage = if crate::transport::is_remote_url(source) {
                // `s3://bucket/key` is rewritten to its
                // virtual-hosted HTTPS form here — the rest of the
                // transport pipeline never sees s3-scheme URLs.
                let translated = crate::transport::normalize_remote_url(source);
                let url = Url::parse(translated.as_ref()).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("invalid URL '{source}' (translated to '{translated}'): {e}"),
                    )
                })?;
                Self::open_url_uncached(url, layout_override)?
            } else {
                // `file://` URIs and bare paths both open as local
                // mmap — `location::file_uri_path` normalises both to a
                // filesystem path string before we hand it to Path.
                // Layout override is meaningless for local mmap
                // (no cache directory involved), so it's dropped.
                let _ = layout_override;
                let local = crate::location::file_uri_path(source);
                Self::open_path_uncached(Path::new(local.as_ref()))?
            };
            let arc = Arc::new(storage);
            registry()
                .lock()
                .unwrap()
                .insert(key.clone(), Arc::downgrade(&arc));
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
    ///
    /// **A path whose file is absent resolves to its series, if one is
    /// there.** A facet written across shards is named
    /// `base__0000.fvecs`, `base__0001.fvecs`, … and nothing at
    /// `base.fvecs`; a reader handed the unsharded name would
    /// otherwise fail with "no such file" for a facet that is
    /// completely present. The combination "this file is missing and
    /// its shards exist" has no other meaning, so resolving it turns
    /// an error into the right answer and changes nothing else.
    pub(crate) fn open_path(path: &Path) -> io::Result<Arc<Self>> {
        if !path.exists() {
            let shards = crate::dataset::shards::discover_shards(path);
            if !shards.is_empty() {
                return Self::open_series(&shards);
            }
        }
        Self::open(path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("non-UTF8 path: {}", path.display()),
            )
        })?)
    }

    /// Open the given shard files as one joined byte space.
    ///
    /// Order is the caller's: shard ordinal order, which is what the
    /// concatenation has to be for the joined offsets to mean what a
    /// reader thinks they mean (SH-38).
    pub(crate) fn open_series(shards: &[std::path::PathBuf]) -> io::Result<Arc<Self>> {
        if shards.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a series needs at least one shard",
            ));
        }
        let mut parts = Vec::with_capacity(shards.len());
        let mut starts = Vec::with_capacity(shards.len());
        let mut total = 0u64;
        for path in shards {
            let part = Self::open(path.to_str().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("non-UTF8 path: {}", path.display()),
                )
            })?)?;
            starts.push(total);
            total += part.total_size();
            parts.push(part);
        }
        // One shard is not a series: returning the file itself keeps
        // every zero-copy path available, which a joined space cannot
        // offer (SH-83 says the writer collapses these, so this is the
        // case of a series that lost shards, or a hand-made one).
        if parts.len() == 1 {
            return Ok(parts.pop().expect("one part"));
        }
        Ok(Arc::new(Storage::Series { parts, starts, total }))
    }

    /// The shards behind this storage, in ordinal order.
    ///
    /// A single file is one part, so a caller that wants to work
    /// per-file does not need to know which it has.
    pub(crate) fn parts(&self) -> Vec<&Arc<Storage>> {
        match self {
            Storage::Series { parts, .. } => parts.iter().collect(),
            _ => Vec::new(),
        }
    }


    /// Crate-internal: build a fresh local-file `Storage::Mmap` (no
    /// registry lookup). Used by `Storage::open` after a registry miss.
    fn open_path_uncached(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Storage::Mmap {
            mmap,
            path: path.to_path_buf(),
        })
    }

    /// Crate-internal: open a fresh URL-backed `Storage` (no
    /// registry lookup). Tries `.mref`-cached first, falls back to
    /// direct HTTP. Used by `Storage::open` after a registry miss.
    fn open_url_uncached(url: Url, layout_override: Option<LayoutChoice>) -> io::Result<Self> {
        let layout = match layout_override {
            Some(l) => l,
            None => layout_for_url(&url)?,
        };
        // A complete copy opens from disk: no `.mref` fetch, no HEAD.
        // Its content was verified as it arrived, and reading it must
        // not depend on the network being there.
        if let Some(local) = Self::open_url_complete(&url, &layout)? {
            return Ok(local);
        }
        match Self::open_url_cached(url.clone(), &layout) {
            Ok(s) => Ok(s),
            Err(_) => Self::open_url_http(url, &layout),
        }
    }

    /// Open a remote URL whose cache copy is already complete, without
    /// touching the network: a merkle-verified copy from its `.mrkl`
    /// state, else a chunk-store copy whose bitmap is full. `Ok(None)`
    /// when there is no complete copy, and the caller goes online.
    fn open_url_complete(url: &Url, layout: &LayoutChoice) -> io::Result<Option<Self>> {
        let placement = layout.placement();
        if !placement.data.is_file() {
            return Ok(None);
        }
        layout::verify_or_record_origin(&layout.dataset_dir, &layout.origin_source)
            .map_err(io::Error::from)?;
        // Offline, a partial copy opens too and serves what it holds;
        // a read of a missing chunk then says offline mode is on.
        let allow_partial = crate::settings::offline();
        if let Some(file) = VerifiedFile::open_complete(crate::transport::http(url.clone()), &placement, allow_partial) {
            return Ok(Some(Storage::Cached(file)));
        }
        if let Some(file) = ChunkedFile::open_complete(crate::transport::http(url.clone()), &placement, allow_partial)? {
            return Ok(Some(Storage::Http(file)));
        }
        Ok(None)
    }

    /// Ask the upstream whether a complete local copy is still what it
    /// publishes — the check a complete open skips, made at the points
    /// whose job is to talk to the server: a fetch, a ping.
    ///
    /// [`Revalidation::Unreachable`] when the server cannot be asked: a
    /// complete copy stays usable offline. An upstream that has changed
    /// is an error naming the stale cache, as a fresh open has always
    /// reported it. Local files and incomplete copies — which went
    /// online to open — have nothing to revalidate.
    pub(crate) fn revalidate(&self) -> io::Result<Revalidation> {
        if crate::settings::offline() && !matches!(self, Storage::Mmap { .. }) {
            return Ok(Revalidation::Unreachable);
        }
        match self {
            Storage::Mmap { .. } => Ok(Revalidation::Current),
            Storage::Series { parts, .. } => {
                let mut outcome = Revalidation::Current;
                for p in parts {
                    if p.revalidate()? == Revalidation::Unreachable {
                        outcome = Revalidation::Unreachable;
                    }
                }
                Ok(outcome)
            }
            Storage::Cached(file) => {
                let Some(url) = file.channel().source_url().map(str::to_string) else {
                    return Ok(Revalidation::Current);
                };
                let mref_url = format!("{url}.mref");
                let resp = match crate::transport::shared().send(crate::transport::Request::get(mref_url.as_str())) {
                    Ok(r) => r,
                    Err(e) if e.is_unreachable() => return Ok(Revalidation::Unreachable),
                    Err(e) => return Err(io::Error::other(format!("revalidate {url}: {e}"))),
                };
                if !resp.is_success() {
                    return Err(io::Error::other(format!(
                        "revalidate {url}: the server no longer publishes its .mref ({})",
                        resp.status_line()
                    )));
                }
                let bytes = resp.bytes().map_err(|e| io::Error::other(format!("revalidate {url}: {e}")))?;
                let upstream = MerkleRef::from_bytes(&bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("mref parse: {e}")))?;
                if upstream.root_hash() != file.channel().reference().root_hash() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "cache for {url} is stale: the remote resource has changed since it \
                             was cached. Remove its dataset with `vectordata cache prune` and \
                             fetch again."
                        ),
                    ));
                }
                Ok(Revalidation::Current)
            }
            Storage::Http(file) => {
                let transport = crate::transport::http(file.url().clone());
                match transport.content_length() {
                    Ok(len) if len == file.total_size() => Ok(Revalidation::Current),
                    Ok(len) => Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "cache for {} is stale: the remote file is {len} bytes, the cached copy \
                             {}. Remove its dataset with `vectordata cache prune` and fetch again.",
                            file.url(),
                            file.total_size()
                        ),
                    )),
                    Err(_) => Ok(Revalidation::Unreachable),
                }
            }
        }
    }

    /// Open a remote URL through the merkle-cache path. Requires a
    /// published `.mref`. Crate-internal; callers go through
    /// `open_url_uncached` so the fallback path runs uniformly.
    pub(crate) fn open_url_cached(url: Url, layout: &LayoutChoice) -> io::Result<Self> {
        let reference = veks_transport::fetch_reference(crate::transport::shared(), &url)?;
        layout::verify_or_record_origin(&layout.dataset_dir, &layout.origin_source)
            .map_err(io::Error::from)?;
        Ok(Storage::Cached(VerifiedFile::open(crate::transport::http(url), reference, &layout.placement())?))
    }

    /// Open a remote URL via direct HTTP RANGE. Crate-internal; the
    /// fallback path of `open_url_uncached` (used when no `.mref` is
    /// published). Per-read fetches fill chunks of the cache copy at
    /// `<dataset_dir>/<file_relpath>` (per the natural layout) until
    /// the caller runs [`precache`](Self::precache), which fills the
    /// rest and maps the file.
    ///
    /// At open time, if the cache copy is already complete (from a
    /// prior precache), the new `Storage::Http` starts mapped.
    pub(crate) fn open_url_http(url: Url, layout: &LayoutChoice) -> io::Result<Self> {
        layout::verify_or_record_origin(&layout.dataset_dir, &layout.origin_source)
            .map_err(io::Error::from)?;
        let transport = crate::transport::http(url);
        let total_size = transport.content_length()?;
        Ok(Storage::Http(ChunkedFile::open(transport, total_size, &layout.placement())?))
    }

    /// Read `len` bytes at `offset`. Always succeeds for an in-bounds
    /// range — slow-path downloads chunks first if necessary.
    pub(crate) fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        match self {
            Storage::Mmap { mmap: m, .. } => {
                let end = offset.checked_add(len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "read range overflow")
                })?;
                if end > m.len() as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("read past end: offset={offset} len={len} size={}", m.len()),
                    ));
                }
                Ok(m[offset as usize..end as usize].to_vec())
            }
            Storage::Http(f) => f.read_bytes(offset, len),
            Storage::Cached(f) => f.read_bytes(offset, len),
            Storage::Series { parts, starts, total } => {
                let end = offset.checked_add(len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "read range overflow")
                })?;
                if end > *total {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("read past end of series: offset={offset} len={len} size={total}"),
                    ));
                }
                // Walk only the parts the range touches. A read inside
                // one shard — which every record-sized read is, because
                // records never straddle (SH-13) — copies once.
                let mut out = Vec::with_capacity(len as usize);
                let mut want = offset;
                while want < end {
                    let i = match starts.binary_search(&want) {
                        Ok(i) => i,
                        Err(i) => i - 1,
                    };
                    let part_start = starts[i];
                    let part_len = parts[i].total_size();
                    let within = want - part_start;
                    let take = (part_len - within).min(end - want);
                    out.extend_from_slice(&parts[i].read_bytes(within, take)?);
                    want += take;
                }
                Ok(out)
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
            Storage::Mmap { mmap: m, .. } => Some(m.as_ptr()),
            Storage::Cached(f) => f.mmap_base(),
            Storage::Http(f) => f.mmap_base(),
            // A series has no single mapping, and saying it has one is
            // the mistake that would make a zero-copy read walk off
            // the end of the first shard into unrelated memory.
            // Callers that need pointers take them per part.
            Storage::Series { .. } => None,
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
            Storage::Mmap { mmap: m, .. } => m.get(start..end),
            Storage::Cached(f) => f.mmap_slice(offset, len),
            Storage::Http(f) => f.mmap_slice(offset, len),
            // Borrowable only when the range lies inside one shard.
            // A range crossing a seam has no contiguous backing, and
            // the caller falls back to `read_bytes` as it does for any
            // other non-borrowable storage.
            Storage::Series { parts, starts, .. } => {
                let i = match starts.binary_search(&offset) {
                    Ok(i) => i,
                    Err(0) => return None,
                    Err(i) => i - 1,
                };
                let within = offset - starts[i];
                parts[i].mmap_slice(within, len)
            }
        }
    }

    /// Total size in bytes.
    pub(crate) fn total_size(&self) -> u64 {
        match self {
            Storage::Mmap { mmap: m, .. } => m.len() as u64,
            Storage::Http(f) => f.total_size(),
            Storage::Cached(f) => f.total_size(),
            Storage::Series { total, .. } => *total,
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
            // A series is resident only when every shard is: a facet
            // half of whose shards are here is not readable.
            Storage::Series { parts, .. } => parts.iter().all(|p| p.is_complete()),
            Storage::Mmap { .. } => true,
            Storage::Http(f) => f.is_complete(),
            Storage::Cached(f) => f.is_complete(),
        }
    }

    /// The content identity of the bytes (IN-6): the publisher's root
    /// for a merkle-cached file; the recorded root for a file fetched
    /// without a reference, once every chunk is here (`None` before);
    /// a local file's root, computed once per process and kept. `None`
    /// for a series, which has no one tree. `progress` hears `(done,
    /// total)` bytes of any hashing this takes.
    pub(crate) fn content_id(&self, progress: &mut dyn FnMut(u64, u64)) -> io::Result<Option<crate::merkle::ContentId>> {
        use crate::merkle::{ContentId, ContentIdKind};
        Ok(match self {
            Storage::Cached(f) => {
                Some(ContentId { root: *f.channel().reference().root_hash(), kind: ContentIdKind::Published })
            }
            Storage::Http(f) => f.store().content_root(progress)?.map(|root| ContentId { root, kind: ContentIdKind::Recorded }),
            Storage::Mmap { path, mmap } => Some(ContentId { root: local_root(path, mmap.len() as u64, progress)?, kind: ContentIdKind::Computed }),
            Storage::Series { .. } => None,
        })
    }

    /// The content identity the cache already knows, reading nothing:
    /// a published root, or a recorded one once the fetch is complete.
    /// `None` for a local file (whose identity is a read of it) and a
    /// series.
    pub(crate) fn known_content_id(&self) -> Option<crate::merkle::ContentId> {
        use crate::merkle::{ContentId, ContentIdKind};
        match self {
            Storage::Cached(f) => Some(ContentId { root: *f.channel().reference().root_hash(), kind: ContentIdKind::Published }),
            Storage::Http(f) => f.store().recorded_root().map(|root| ContentId { root, kind: ContentIdKind::Recorded }),
            Storage::Mmap { .. } | Storage::Series { .. } => None,
        }
    }

    /// Check the cached chunks against what they were verified or
    /// recorded as, returning the chunks that no longer match: `None`
    /// for a local file, which has nothing to check against.
    pub(crate) fn verify_cached(&self, progress: &mut dyn FnMut(u64, u64)) -> io::Result<Option<Vec<u32>>> {
        Ok(match self {
            Storage::Cached(f) => Some(f.channel().verify_cached(progress)?),
            Storage::Http(f) => Some(f.store().verify_recorded(progress)?),
            Storage::Mmap { .. } => None,
            Storage::Series { parts, .. } => {
                let mut bad = Vec::new();
                let mut any = false;
                for p in parts {
                    if let Some(b) = p.verify_cached(progress)? {
                        any = true;
                        bad.extend(b);
                    }
                }
                any.then_some(bad)
            }
        })
    }

    /// Whether the bytes live on a server and the local file, if any, is
    /// a copy of them: a cache file whose mtime moves as it fills.
    pub(crate) fn is_remote(&self) -> bool {
        match self {
            Storage::Mmap { .. } => false,
            Storage::Http(_) | Storage::Cached(_) => true,
            Storage::Series { parts, .. } => parts.iter().any(|p| p.is_remote()),
        }
    }

    /// Whether reads avoid network round-trips.
    pub(crate) fn is_local(&self) -> bool {
        match self {
            Storage::Series { parts, .. } => parts.iter().all(|p| p.is_local()),
            Storage::Mmap { .. } => true,
            Storage::Cached(f) => f.is_local(),
            Storage::Http(f) => f.is_local(),
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
        mut cb: F,
    ) -> io::Result<()>
    where
        F: FnMut(&crate::DownloadProgress),
    {
        self.prebuffer_range_with_progress_dyn(byte_start, byte_end, &mut cb)
    }

    /// The body of [`Self::prebuffer_range_with_progress`], with the
    /// callback dynamically dispatched.
    ///
    /// A series delegates to its shards, and a generic callback would
    /// re-instantiate this function at `&mut F`, then `&mut &mut F`,
    /// for as long as the compiler was willing — so the recursive form
    /// takes a trait object and the recursion stays monomorphic.
    fn prebuffer_range_with_progress_dyn(
        &self,
        byte_start: u64,
        byte_end: u64,
        cb: &mut dyn FnMut(&crate::DownloadProgress),
    ) -> io::Result<()> {
        match self {
            // Each shard buffers the slice of the range that falls in
            // it, so a window over a series costs the shards it spans
            // rather than the whole facet (SH-92).
            Storage::Series { parts, starts, total } => {
                let end = byte_end.min(*total);
                for (i, part) in parts.iter().enumerate() {
                    let part_start = starts[i];
                    let part_end = part_start + part.total_size();
                    if part_end <= byte_start || part_start >= end {
                        continue;
                    }
                    let from = byte_start.saturating_sub(part_start);
                    let to = end.min(part_end) - part_start;
                    part.prebuffer_range_with_progress_dyn(from, to, cb)?;
                }
                Ok(())
            }
            // Local mmap is already fully resident — fire one
            // completion event so callers' meters land at 100%.
            Storage::Mmap { mmap: m, .. } => {
                cb(&crate::DownloadProgress::new(m.len() as u64, 0));
                Ok(())
            }
            Storage::Http(f) => f.prebuffer_range_with_progress(byte_start, byte_end, cb),
            Storage::Cached(f) => f.prebuffer_range_with_progress(byte_start, byte_end, cb),
        }
    }

    /// Same as [`precache`] with a progress callback fired in
    /// flight — once at start (`downloaded_bytes == 0`) and then
    /// after each completed chunk, so UIs can render a live meter
    /// rather than wait for completion. Same strict contract:
    /// either every byte is resident on `Ok`, or an `Err` is
    /// returned.
    pub(crate) fn prebuffer_with_progress<F>(&self, mut cb: F) -> io::Result<()>
    where
        F: FnMut(&crate::DownloadProgress),
    {
        self.prebuffer_with_progress_dyn(&mut cb)
    }

    /// The body of [`Self::prebuffer_with_progress`]. See
    /// [`Self::prebuffer_range_with_progress_dyn`] for why the
    /// recursive form takes a trait object.
    fn prebuffer_with_progress_dyn(
        &self,
        cb: &mut dyn FnMut(&crate::DownloadProgress),
    ) -> io::Result<()> {
        match self {
            Storage::Series { parts, .. } => {
                for part in parts {
                    part.prebuffer_with_progress_dyn(cb)?;
                }
                Ok(())
            }
            Storage::Mmap { .. } => Ok(()),
            Storage::Http(f) => f.prebuffer_with_progress(cb),
            Storage::Cached(f) => f.prebuffer_with_progress(cb),
        }
    }

    /// Hint that the full mapping will be accessed sequentially.
    /// No-op for non-mmap variants (and for `Cached`/`Http` until
    /// promotion). No-op on non-Unix targets — `memmap2::Mmap::advise`
    /// is Unix-only (mirrors the `release_range_bytes` pattern).
    pub(crate) fn advise_sequential(&self) {
        if let Some(m) = self.promoted_mmap() {
            veks_io::source::mapping::advise_sequential_bytes(m);
        }
    }

    /// Hint that the full mapping will be accessed in random order.
    /// No-op on non-Unix targets.
    pub(crate) fn advise_random(&self) {
        if let Some(m) = self.promoted_mmap() {
            veks_io::source::mapping::advise_random_bytes(m);
        }
    }

    /// Internal helper: borrow the promoted mmap, lazy-promoting
    /// the Cached and Http variants if their underlying file is
    /// ready.
    fn promoted_mmap(&self) -> Option<&[u8]> {
        match self {
            // No single mapping to promote; see `mmap_base`.
            Storage::Series { .. } => None,
            Storage::Mmap { mmap: m, .. } => Some(&m[..]),
            Storage::Cached(f) => f.mapped(),
            Storage::Http(f) => f.mapped(),
        }
    }

    /// `madvise(WILLNEED)` over the byte range. Non-blocking hint.
    /// No-op on non-Unix targets — `Mmap::advise_range` is Unix-only.
    pub(crate) fn prefetch_range_bytes(&self, byte_start: u64, byte_end: u64) {
        if let Some(m) = self.promoted_mmap() {
            veks_io::source::mapping::prefetch_range_bytes(m, byte_start, byte_end);
        }
    }

    /// `madvise(DONTNEED)` over the byte range. Page-aligns start UP
    /// and end DOWN — sub-page slack stays mapped, but on a
    /// streaming scan that's negligible compared to the megabyte-or-
    /// larger windows callers pass in. Without alignment, madvise
    /// rejects with EINVAL silently.
    pub(crate) fn release_range_bytes(&self, byte_start: u64, byte_end: u64) {
        // A series has no single mapping: the range is split over the
        // shards it touches and each releases its own part. Without
        // this a sharded base never drops a partition's pages, and a
        // pass that touches most of the base carries most of it as
        // resident file-backed memory (tessera, 2026-09-03: 667 GB at
        // 200m rows, aborted by the governor).
        if let Storage::Series { parts, .. } = self {
            for (i, local_start, local_end) in self.part_ranges(byte_start, byte_end) {
                parts[i].release_range_bytes(local_start, local_end);
            }
            return;
        }
        if let Some(m) = self.promoted_mmap() {
            veks_io::source::mapping::release_range_bytes(m, byte_start, byte_end);
        }
    }

    /// Force the byte range into the page cache via
    /// `madvise(WILLNEED)` + sequential volatile read. Blocks until
    /// the pages are resident. Caller increments `bytes_paged` per
    /// page touched. The `madvise` portion is Unix-only; the
    /// volatile-read fallback still executes on all platforms (it's
    /// the load-bearing part of the prefetch — `madvise` is just a
    /// hint).
    pub(crate) fn prefetch_pages_bytes(
        &self,
        byte_start: u64,
        byte_end: u64,
        bytes_paged: Option<&AtomicU64>,
    ) {
        // A series prefetches shard by shard, like it releases.
        if let Storage::Series { parts, .. } = self {
            for (i, local_start, local_end) in self.part_ranges(byte_start, byte_end) {
                parts[i].prefetch_pages_bytes(local_start, local_end, bytes_paged);
            }
            return;
        }
        if let Some(m) = self.promoted_mmap() {
            veks_io::source::mapping::prefetch_pages_bytes(m, byte_start, byte_end, bytes_paged);
        }
    }

    /// The shards a byte range of the joined space touches, each with
    /// the range's part local to it: `(part index, local start, local
    /// end)`, in order. A single file is one part covering the range
    /// clipped to its size; an empty or inverted range is no parts.
    pub(crate) fn part_ranges(&self, byte_start: u64, byte_end: u64) -> Vec<(usize, u64, u64)> {
        let mut out = Vec::new();
        let end = byte_end.min(self.total_size());
        if byte_start >= end {
            return out;
        }
        match self {
            Storage::Series { parts, starts, .. } => {
                let mut want = byte_start;
                while want < end {
                    let i = match starts.binary_search(&want) {
                        Ok(i) => i,
                        Err(i) => i - 1,
                    };
                    let within = want - starts[i];
                    let take = (parts[i].total_size() - within).min(end - want);
                    out.push((i, within, within + take));
                    want += take;
                }
            }
            _ => out.push((0, byte_start, end)),
        }
        out
    }

    /// Path to the cache file, when this storage is `Cached`. Used by
    /// debugging / diagnostic surfaces only.
    #[allow(dead_code)]
    pub(crate) fn cache_path(&self) -> Option<&Path> {
        match self {
            Storage::Cached(f) => Some(f.channel().cache_path()),
            _ => None,
        }
    }

    /// Best-effort path to the local file backing this storage. `Some`
    /// when the bytes are reachable through the filesystem — i.e.,
    /// `Storage::Cached` (the cache file) and `Storage::Http` after
    /// `precache` has downloaded the file (or if a prior precache
    /// left the cache file at the expected path).
    ///
    /// For `Storage::Mmap` it is the file that was opened. `None` for
    /// `Storage::Http` when no cache file exists yet.
    pub(crate) fn local_path(&self) -> Option<std::path::PathBuf> {
        match self {
            Storage::Mmap { path, .. } => Some(path.clone()),
            Storage::Cached(f) => Some(f.local_path()),
            Storage::Http(f) => f.local_path(),
            _ => None,
        }
    }

    /// Chunk-level residency for a byte range, or `None` when the
    /// storage has no chunks — local mmap is always fully resident and
    /// a range question about it has no meaning beyond "yes".
    ///
    /// Returns `(first_chunk, last_chunk, chunk_size, resident_chunks,
    /// resident_bytes)` with both chunk indices inclusive.
    /// `resident_bytes` counts each resident chunk at its real length:
    /// the file's last chunk holds only what is left of the file. The
    /// caller turns that into a cost; this layer does not know what the
    /// range was *for*.
    pub(crate) fn range_fill(
        &self,
        byte_start: u64,
        byte_end: u64,
    ) -> Option<(u32, u32, u64, u32, u64)> {
        let (_, total_chunks, chunk_size, content_size, _) = self.fill_stats()?;
        let (first, last) =
            chunk_span(byte_start, byte_end, chunk_size, total_chunks, content_size)?;
        let resident_in = |a: u32, b: u32| match self {
            Storage::Http(f) => Some(f.store().valid_count_in_range(a, b)),
            Storage::Cached(f) => Some(f.channel().valid_count_in_range(a, b)),
            Storage::Mmap { .. } => None,
            // Chunk fill is a property of one transport. A series
            // spans several, each with its own chunk map, so there is
            // no single span to report — residency for a series is
            // asked per shard.
            Storage::Series { .. } => None,
        };
        let resident = resident_in(first, last)?;
        let last_resident = resident_in(last, last)? == 1;
        let last_len = ((last as u64 + 1) * chunk_size).min(content_size) - last as u64 * chunk_size;
        let resident_bytes = (resident - last_resident as u32) as u64 * chunk_size
            + if last_resident { last_len } else { 0 };
        Some((first, last, chunk_size, resident, resident_bytes))
    }

    /// Live cache-fill statistics for chunk-backed remote storage:
    /// `(valid_chunks, total_chunks, chunk_size, content_size,
    /// is_complete)`. Covers both the merkle-verified (`Cached`) and
    /// merkle-less (`Http`) chunk stores — a status UI watching a
    /// no-`.mref` download must see the same fill progress as a
    /// `.mref` one. `None` for local mmap, which is always fully
    /// resident and has no fill state to report.
    pub(crate) fn fill_stats(&self) -> Option<(u32, u32, u64, u64, bool)> {
        match self {
            Storage::Mmap { .. } => None,
            // See `range_fill`: one chunk map per shard, not per facet.
            Storage::Series { .. } => None,
            Storage::Http(f) => Some((
                f.store().valid_count(),
                f.store().total_chunks(),
                f.store().chunk_size(),
                f.store().total_size(),
                f.store().is_complete(),
            )),
            Storage::Cached(f) => Some((
                f.channel().valid_count(),
                f.channel().total_chunks(),
                f.channel().chunk_size(),
                f.channel().content_size(),
                f.channel().is_complete(),
            )),
        }
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Series { parts, total, .. } => f
                .debug_struct("Storage::Series")
                .field("shards", &parts.len())
                .field("size", total)
                .finish(),
            Storage::Mmap { mmap: m, .. } => f
                .debug_struct("Storage::Mmap")
                .field("size", &m.len())
                .finish(),
            Storage::Http(file) => f
                .debug_struct("Storage::Http")
                .field("size", &file.store().total_size())
                .field("chunks_valid", &file.store().valid_count())
                .field("chunks_total", &file.store().total_chunks())
                .field("promoted", &file.is_mapped())
                .finish(),
            Storage::Cached(file) => f
                .debug_struct("Storage::Cached")
                .field("size", &file.channel().content_size())
                .field("complete", &file.channel().is_complete())
                .field("promoted", &file.is_mapped())
                .finish(),
        }
    }
}

/// Chunks covering `[byte_start, byte_end)`, inclusive at both ends.
///
/// Pure arithmetic, separated from storage so the off-by-ones can be
/// tested without a server. Getting these wrong does not fail loudly —
/// it fetches a window adjacent to the one asked for, and the reader
/// then faults in the right chunks anyway, so the only visible symptom
/// is a prefetch that silently did nothing useful.
///
/// `None` when there is nothing to span: no chunks, a zero chunk size,
/// an empty or inverted range, or a range starting past the content.
pub(crate) fn chunk_span(
    byte_start: u64,
    byte_end: u64,
    chunk_size: u64,
    total_chunks: u32,
    content_size: u64,
) -> Option<(u32, u32)> {
    if chunk_size == 0 || total_chunks == 0 || byte_end <= byte_start {
        return None;
    }
    if byte_start >= content_size {
        return None;
    }
    let end = byte_end.min(content_size);
    let first = (byte_start / chunk_size) as u32;
    let last = (((end - 1) / chunk_size) as u32).min(total_chunks - 1);
    if first > last {
        return None;
    }
    Some((first, last))
}

#[cfg(test)]
mod chunk_span_tests {
    use super::chunk_span;

    /// 10 chunks of 100 bytes over a 1000-byte file.
    const CS: u64 = 100;
    const N: u32 = 10;
    const SIZE: u64 = 1000;

    fn span(start: u64, end: u64) -> Option<(u32, u32)> {
        chunk_span(start, end, CS, N, SIZE)
    }

    #[test]
    fn a_range_inside_one_chunk_spans_that_chunk() {
        assert_eq!(span(0, 1), Some((0, 0)));
        assert_eq!(span(0, 100), Some((0, 0)), "exactly one chunk, not two");
        assert_eq!(span(99, 100), Some((0, 0)));
        assert_eq!(span(100, 101), Some((1, 1)));
    }

    /// The end is exclusive, so a range ending on a boundary must not
    /// drag in the chunk that starts there. This is the off-by-one that
    /// turns every prefetch into one chunk more than it needed.
    #[test]
    fn an_exclusive_end_on_a_boundary_does_not_span_the_next_chunk() {
        assert_eq!(span(0, 200), Some((0, 1)));
        assert_eq!(span(0, 201), Some((0, 2)));
        assert_eq!(span(150, 250), Some((1, 2)));
    }

    #[test]
    fn a_range_is_clamped_to_the_content() {
        assert_eq!(span(0, SIZE), Some((0, 9)));
        assert_eq!(span(0, SIZE * 4), Some((0, 9)), "past the end clamps");
        assert_eq!(span(950, u64::MAX), Some((9, 9)));
    }

    #[test]
    fn degenerate_ranges_span_nothing() {
        assert_eq!(span(5, 5), None, "empty");
        assert_eq!(span(9, 4), None, "inverted");
        assert_eq!(span(SIZE, SIZE + 10), None, "starts past the content");
        assert_eq!(chunk_span(0, 10, 0, N, SIZE), None, "zero chunk size");
        assert_eq!(chunk_span(0, 10, CS, 0, SIZE), None, "no chunks");
    }

    /// A final short chunk is still a chunk.
    #[test]
    fn a_ragged_tail_chunk_is_spanned() {
        // 250 bytes in 100-byte chunks = 3 chunks, the last holding 50.
        assert_eq!(chunk_span(200, 250, CS, 3, 250), Some((2, 2)));
        assert_eq!(chunk_span(0, 250, CS, 3, 250), Some((0, 2)));
    }
}

#[cfg(test)]
mod series_tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// Write shard `i` of a ramp: byte `n` of the joined space is
    /// `n as u8`, so any read reports where it came from.
    fn shard(dir: &Path, i: u32, from: usize, len: usize) -> std::path::PathBuf {
        let p = dir.join(crate::dataset::shards::shard_name("f", "fvecs", i));
        let bytes: Vec<u8> = (from..from + len).map(|n| n as u8).collect();
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// **The joined space is the concatenation.** A read at any offset
    /// returns the bytes the unsharded file would have had there.
    #[test]
    fn a_series_reads_as_one_byte_space() {
        let tmp = tmpdir();
        let shards = vec![
            shard(tmp.path(), 0, 0, 100),
            shard(tmp.path(), 1, 100, 100),
            shard(tmp.path(), 2, 200, 50),
        ];
        let s = Storage::open_series(&shards).unwrap();
        assert_eq!(s.total_size(), 250);

        // Whole space, one read.
        let all = s.read_bytes(0, 250).unwrap();
        assert_eq!(all, (0..250).map(|n| n as u8).collect::<Vec<u8>>());

        // Reads inside one shard.
        assert_eq!(s.read_bytes(0, 4).unwrap(), [0, 1, 2, 3]);
        assert_eq!(s.read_bytes(150, 4).unwrap(), [150, 151, 152, 153]);
        assert_eq!(s.read_bytes(246, 4).unwrap(), [246, 247, 248, 249]);
    }

    /// A read that crosses a seam is stitched, not truncated — the
    /// case a reader hits whenever a window straddles two shards.
    #[test]
    fn a_read_across_a_seam_is_stitched() {
        let tmp = tmpdir();
        let shards = vec![
            shard(tmp.path(), 0, 0, 100),
            shard(tmp.path(), 1, 100, 100),
        ];
        let s = Storage::open_series(&shards).unwrap();

        assert_eq!(s.read_bytes(98, 4).unwrap(), [98, 99, 100, 101]);
        // And a read spanning all of both.
        assert_eq!(s.read_bytes(0, 200).unwrap().len(), 200);
    }

    /// Past the end is refused rather than silently short.
    #[test]
    fn a_read_past_the_end_is_refused() {
        let tmp = tmpdir();
        let shards = vec![shard(tmp.path(), 0, 0, 10), shard(tmp.path(), 1, 10, 10)];
        let s = Storage::open_series(&shards).unwrap();
        assert!(s.read_bytes(19, 4).is_err());
        assert!(s.read_bytes(20, 1).is_err());
        assert_eq!(s.read_bytes(19, 1).unwrap(), [19]);
    }

    /// **A series has no single base pointer**, and says so — the
    /// alternative is a zero-copy read walking off the end of the
    /// first shard into unrelated memory.
    #[test]
    fn a_series_offers_no_single_mapping() {
        let tmp = tmpdir();
        let shards = vec![shard(tmp.path(), 0, 0, 10), shard(tmp.path(), 1, 10, 10)];
        let s = Storage::open_series(&shards).unwrap();
        assert!(s.mmap_base().is_none());
        // But a range inside one shard is still borrowable.
        assert_eq!(s.mmap_slice(2, 4).unwrap(), &[2, 3, 4, 5]);
        assert_eq!(s.mmap_slice(12, 4).unwrap(), &[12, 13, 14, 15]);
        // And one that crosses a seam is not.
        assert!(s.mmap_slice(8, 4).is_none());
    }

    /// A single shard is not a series: it opens as the file itself, so
    /// every zero-copy path stays available.
    #[test]
    fn one_shard_is_just_a_file() {
        let tmp = tmpdir();
        let shards = vec![shard(tmp.path(), 0, 0, 10)];
        let s = Storage::open_series(&shards).unwrap();
        assert!(s.mmap_base().is_some(), "still a plain mapping");
        assert_eq!(s.total_size(), 10);
    }

    /// **A path whose file is absent resolves to its shards.** This is
    /// what lets a step that writes a series and a step that reads the
    /// unsharded name meet without either knowing about the other.
    #[test]
    fn an_absent_file_resolves_to_its_series() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 0, 100);
        shard(tmp.path(), 1, 100, 100);

        let s = Storage::open_path(&tmp.path().join("f.fvecs")).unwrap();
        assert_eq!(s.total_size(), 200);
        assert_eq!(s.read_bytes(99, 2).unwrap(), [99, 100]);
    }

    /// The plain file wins when it is there, so nothing about an
    /// ordinary unsharded facet changes.
    #[test]
    fn a_present_file_is_preferred_over_stray_shards() {
        let tmp = tmpdir();
        std::fs::write(tmp.path().join("f.fvecs"), [9u8; 7]).unwrap();
        shard(tmp.path(), 0, 0, 100);

        let s = Storage::open_path(&tmp.path().join("f.fvecs")).unwrap();
        assert_eq!(s.total_size(), 7, "the file itself, not the shard");
    }

    /// **A gap stops discovery**, so a facet missing a shard is not
    /// read as the prefix that happens to be there.
    #[test]
    fn a_missing_shard_truncates_discovery_rather_than_skipping_it() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 0, 10);
        shard(tmp.path(), 1, 10, 10);
        // No shard 2.
        shard(tmp.path(), 3, 30, 10);

        let found = crate::dataset::shards::discover_shards(&tmp.path().join("f.fvecs"));
        assert_eq!(found.len(), 2, "stops at the gap: {found:?}");
    }

    /// A path with no shards at all discovers nothing — the ordinary
    /// case for every unsharded file.
    #[test]
    fn an_ordinary_path_discovers_no_shards() {
        let tmp = tmpdir();
        std::fs::write(tmp.path().join("f.fvecs"), [1u8; 4]).unwrap();
        assert!(crate::dataset::shards::discover_shards(&tmp.path().join("f.fvecs")).is_empty());
        assert!(crate::dataset::shards::discover_shards(&tmp.path().join("absent.fvecs")).is_empty());
    }
}

#[cfg(test)]
mod part_range_tests {
    use super::Storage;
    use std::path::PathBuf;

    fn series(dir: &std::path::Path, sizes: &[usize]) -> std::sync::Arc<Storage> {
        let mut paths: Vec<PathBuf> = Vec::new();
        for (i, size) in sizes.iter().enumerate() {
            let p = dir.join(format!("s__{:04}.bin", i));
            std::fs::write(&p, vec![i as u8 + 1; *size]).unwrap();
            paths.push(p);
        }
        Storage::open_series(&paths).unwrap()
    }

    /// A range over the joined space maps onto the shards it touches,
    /// each in its own coordinates.
    #[test]
    fn a_joined_range_splits_over_the_shards_it_touches() {
        let dir = tempfile::tempdir().unwrap();
        let s = series(dir.path(), &[100, 50, 80]);
        assert_eq!(s.part_ranges(0, 100), vec![(0, 0, 100)]);
        assert_eq!(s.part_ranges(90, 120), vec![(0, 90, 100), (1, 0, 20)]);
        assert_eq!(s.part_ranges(100, 150), vec![(1, 0, 50)]);
        assert_eq!(s.part_ranges(120, 230), vec![(1, 20, 50), (2, 0, 80)]);
        assert_eq!(s.part_ranges(0, 1000), vec![(0, 0, 100), (1, 0, 50), (2, 0, 80)], "clipped to the series");
        assert!(s.part_ranges(50, 50).is_empty());
        assert!(s.part_ranges(60, 40).is_empty());
        assert!(s.part_ranges(230, 300).is_empty());
        let one = Storage::open_series(&[dir.path().join("s__0000.bin")]).unwrap();
        assert_eq!(one.part_ranges(10, 500), vec![(0, 10, 100)]);
    }

    /// Releasing a range of a series reaches the shards: the pages
    /// leave the process's resident set (they stay in the page cache).
    #[cfg(target_os = "linux")]
    #[test]
    fn releasing_a_series_range_drops_its_resident_pages() {
        fn rss_file_kb() -> u64 {
            std::fs::read_to_string("/proc/self/status")
                .unwrap()
                .lines()
                .find_map(|l| l.strip_prefix("RssFile:"))
                .and_then(|v| v.trim().trim_end_matches(" kB").trim().parse().ok())
                .unwrap()
        }
        let dir = tempfile::tempdir().unwrap();
        let shard = 16 << 20;
        let s = series(dir.path(), &[shard, shard]);
        // Touch every page of both shards through the mapping.
        let total = s.total_size();
        let mut sum = 0u64;
        let mut off = 0u64;
        while off < total {
            sum += s.read_bytes(off, 1).unwrap()[0] as u64;
            off += 4096;
        }
        assert!(sum > 0);
        let before = rss_file_kb();
        s.release_range_bytes(0, total);
        let after = rss_file_kb();
        // Two shards of 16 MiB were resident; at least one shard's
        // worth must have gone, whatever else the test process maps.
        assert!(
            before.saturating_sub(after) >= (shard as u64 / 1024),
            "RssFile {} kB → {} kB after releasing {} bytes",
            before, after, total
        );
    }
}
