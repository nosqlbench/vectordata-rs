// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Profile views for accessing dataset components.
//!
//! Defines the [`TestDataView`] trait and `GenericTestDataView` implementation
//! for uniform access to base vectors, query vectors, ground-truth neighbors,
//! and metadata facets regardless of backing storage (local mmap or HTTP).
//!
//! The [`FacetDescriptor`] type supports discover-then-load patterns by
//! describing available facets without materializing data.

use crate::dataset::facet::StandardFacet;
use crate::location::Location;
use crate::io::IoError;
use crate::io::{self, Element, VectorReader, VvecReader};
use crate::model::{FacetConfig, ProfileConfig};
use crate::{Error, FacetBytes, Result};

pub use veks_apicore::view::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Parse a window string using the canonical dataset-source parser
/// (`crate::dataset::source::parse_window`). Returns the FIRST
/// interval as `(start, end)` with `end` exclusive — the multi-
/// interval form `"[0..1K, 2K..3K]"` is documented but the reader
/// API only handles a single contiguous range; callers wanting a
/// disjoint window should split into separate facet configs.
///
/// `None` for malformed input or empty windows — callers fall back
/// to the unwrapped reader.
fn parse_window_first(s: &str) -> Option<(usize, usize)> {
    let dsw = crate::dataset::source::parse_window(s).ok()?;
    let iv = dsw.0.into_iter().next()?;
    let start = iv.min_incl as usize;
    let end = iv.max_excl as usize;
    if end < start {
        return None;
    }
    Some((start, end))
}

/// The single source of a facet, or an error naming sharding as the
/// reason there is not one.
///
/// Every caller of this predates multi-file facets and reads one path.
/// A series has no single path, and a caller written before sharding
/// must fail visibly rather than resolve to its first shard
/// (`docs/design/srd-multifile-facet-shards.md`, SH-74). These call
/// sites go away when the reader layer consumes the realized `Shards`
/// model.
fn single_source<'a>(facet: &'a FacetConfig, what: &str) -> Result<&'a str> {
    facet.source().ok_or_else(|| {
        Error::Other(format!(
            "{what}: this facet declares a multi-file series, which this \
             path does not resolve yet — see \
             docs/design/srd-multifile-facet-shards.md"
        ))
    })
}

/// How many records a file holds — the inverse of
/// [`record_range_to_bytes`], sharing its format dispatch so the two
/// cannot disagree about what a record is.
///
/// Consulted only for a bare series entry, whose length the declaration
/// does not state (SH-63). `None` when the format offers no answer from
/// the file alone.
pub(crate) fn records_in(path: &str, storage: &std::sync::Arc<crate::storage::Storage>) -> Option<u64> {
    let ext = path.rsplit('.').next().unwrap_or("");
    let elem_size = crate::io::infer_elem_size(ext);
    if elem_size == 0 {
        return None;
    }
    let total = storage.total_size();
    if crate::io::is_vvec_ext(ext) {
        // Variable-length: only the index knows. A bare entry is
        // local-only, so the walk this may cost is a local walk.
        let offsets = crate::io::load_offsets(path, storage, elem_size).ok()?;
        return Some(offsets.len() as u64);
    }
    if crate::io::is_scalar_ext(ext) {
        return Some(total / elem_size as u64);
    }
    // Uniform xvec: one dim header governs the whole file.
    let bpr = uniform_record_bytes(ext, storage)?;
    Some(total / bpr)
}

/// Bytes per record of a uniform xvec file, from its first header as
/// veks-io reads it — four bytes, so a sparse remote file stays sparse.
///
/// `None` when the extension names no element type, the header is not
/// a positive dimension, or it claims more than a million elements (a
/// corrupt header, not a vector).
fn uniform_record_bytes(ext: &str, storage: &std::sync::Arc<crate::storage::Storage>) -> Option<u64> {
    let elem = veks_io::ElementType::from_extension(ext)?;
    let probe = veks_io::reader::probe_source(&crate::io::StorageSource::new(storage.clone()), elem).ok()?;
    (probe.dim <= 1_000_000).then_some(probe.record_bytes)
}


/// Map a record range to shard-qualified byte ranges.
///
/// For a single file this is one range in shard `0` — the old meaning
/// under the new shape. For a series the window is decomposed per shard
/// (SH-14) and each sub-window mapped against **its own file**, in that
/// file's ordinals (SH-64).
pub(crate) fn record_range_to_bytes(
    path_no_window: &str,
    win_start: u64,
    win_end: u64,
    storage: &FacetStorage,
) -> Option<MappedRange> {
    if win_end <= win_start {
        return None;
    }
    match storage.series_ref() {
        None => {
            let m = map_in_file(
                path_no_window,
                win_start,
                win_end,
                &storage.storage,
                &|elem| storage.published_offsets(path_no_window, elem),
            )?;
            Some(MappedRange {
                ranges: vec![ShardRange::whole(m.0, m.1)],
                prerequisite_bytes: m.2,
            })
        }
        Some(series) => {
            let mut ranges = Vec::new();
            let mut prerequisite_bytes = 0u64;
            for w in series.shards().decompose(win_start, win_end) {
                let file = series.file_index_of_shard(w.shard).ok()?;
                let st = series.file(file).ok()?;
                let path = series.shards().entries()[w.shard].source.path.clone();
                let m = map_in_file(&path, w.file.0, w.file.1, &st, &|elem| {
                    series.published_offsets(file, &path, elem)
                })?;
                // Summed, not maxed: each shard's index is a separate
                // read, and a window touching three shards pays for
                // three (SH-29).
                prerequisite_bytes = prerequisite_bytes.saturating_add(m.2);
                ranges.push(ShardRange {
                    shard: w.shard,
                    start: m.0,
                    end: m.1,
                });
            }
            (!ranges.is_empty()).then_some(MappedRange {
                ranges,
                prerequisite_bytes,
            })
        }
    }
}

/// Map a record range **within one file** to `(start, end, prerequisite)`.
///
/// `offsets` is consulted only for variable-length formats, and is a
/// closure so the caller decides where the index comes from: a facet
/// handle's cache for a single file, the series' per-file cache for a
/// shard.
fn map_in_file(
    path_no_window: &str,
    lo: u64,
    hi: u64,
    storage: &std::sync::Arc<crate::storage::Storage>,
    offsets: &dyn Fn(usize) -> Option<std::sync::Arc<Vec<u64>>>,
) -> Option<(u64, u64, u64)> {
    if hi <= lo {
        return None;
    }
    // A slab carries its own ordinal index in its tail, so a window
    // maps to the pages it spans. Without this branch the extension
    // fails `infer_elem_size` below and every slab window degrades to a
    // whole-facet fetch — which would leave the planner claiming a cost
    // the reader does not pay, since a record read costs its page.
    if slab_ext(path_no_window) {
        return map_in_slab(path_no_window, lo, hi, storage);
    }

    let ext = path_no_window.rsplit('.').next().unwrap_or("");
    let elem_size = crate::io::infer_elem_size(ext);
    if elem_size == 0 {
        return None;
    }
    let total = storage.total_size();

    if crate::io::is_vvec_ext(ext) {
        // Planning must not move bytes to decide what to move: a remote
        // vvec with no published sidecar returns `None` here and the
        // window degrades to a whole-facet fetch, which the fallback
        // policy then consents to or refuses. Rebuilding the index by
        // walking the file would download all of it *while planning*,
        // defeating the gate.
        let offsets = offsets(elem_size)?;
        if offsets.is_empty() {
            return None;
        }
        let count = offsets.len() as u64;
        if lo >= count {
            return None;
        }
        let byte_start = offsets[lo as usize];
        // A window running past the last record ends at the file, not at
        // a record that does not exist.
        let byte_end = if hi >= count {
            total
        } else {
            offsets[hi as usize]
        };
        if byte_end <= byte_start {
            return None;
        }
        // The index had to be read whole to answer this at all.
        //
        // Counted as **record starts × 8**, not sidecar-file bytes: the
        // published widths differ (`i32` vs `i64`) and the sentinel
        // layout carries one entry more than there are records, so the
        // on-disk size is not a stable measure of the same thing.
        // `offsets` is post-sentinel — see `io::parse_index_bytes` — so
        // both sidecar layouts report the same figure here.
        let prereq = (offsets.len() * std::mem::size_of::<u64>()) as u64;
        return Some((byte_start, byte_end, prereq));
    }

    if crate::io::is_scalar_ext(ext) {
        // Scalar facets are raw packed values with no per-record
        // header — `ScalarReader` addresses them at `ordinal *
        // elem_size`. Falling through to the xvec path below would
        // read the first *value* as a dimension: a needless degrade
        // when it fails the sanity check, and silently wrong byte
        // ranges when it happens to pass.
        let stride = elem_size as u64;
        let byte_start = lo.saturating_mul(stride).min(total);
        let byte_end = hi.saturating_mul(stride).min(total);
        return (byte_start < byte_end).then_some((byte_start, byte_end, 0));
    }

    // Uniform xvec: one dim header governs the file.
    let bpr = uniform_record_bytes(ext, storage)?;
    let byte_start = lo.saturating_mul(bpr).min(total);
    let byte_end = hi.saturating_mul(bpr).min(total);
    // A uniform-stride mapping costs a 4-byte header read, which every
    // reader pays on first access anyway — nothing to report.
    (byte_start < byte_end).then_some((byte_start, byte_end, 0))
}

/// Whether a locator names a slab, ignoring any `:namespace` suffix.
fn slab_ext(locator: &str) -> bool {
    let path = locator.split(':').next().unwrap_or(locator);
    path.rsplit('.')
        .next()
        .is_some_and(|e| e.eq_ignore_ascii_case("slab"))
}

/// Map an ordinal window in a slab to the byte range of the pages that
/// hold it.
///
/// Pages are laid out in ordinal order, so a contiguous window is a
/// contiguous span of pages and therefore one byte range — the same
/// shape every other branch returns. The index comes from the tail
/// (`records::read_slab_index`), which is a bounded read of two short
/// pages rather than a walk of the file, so planning stays cheap in the
/// sense SH-21 requires: it must not move the bytes it is deciding
/// whether to move.
fn map_in_slab(
    locator: &str,
    lo: u64,
    hi: u64,
    storage: &crate::storage::Storage,
) -> Option<(u64, u64, u64)> {
    let namespace = locator.split_once(':').map(|(_, ns)| ns);
    let index = crate::records::read_slab_index(storage, namespace, locator)
        .ok()
        .flatten()?;
    if index.total() == 0 {
        return None;
    }
    let first = index.page_of(lo)?;
    // A window running past the last record ends at the last page, not
    // at one that does not exist.
    let last = index.page_of(hi - 1).unwrap_or(index.page_count() - 1);
    let byte_start = index.page_offset(first)?;
    let last_start = index.page_offset(last)?;
    // The final page's extent comes from its own header, which is the
    // only place its size is written.
    let size = storage
        .read_bytes(last_start + 4, 4)
        .ok()
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)? as u64;
    let byte_end = (last_start + size).min(storage.total_size());
    (byte_start < byte_end).then_some((byte_start, byte_end, index.prerequisite_bytes()))
}

/// Reader wrapper that clips access to a sub-range of the underlying
/// reader. `count()` reports the window length; `get(i)` reads from
/// `underlying.get(i + start)` and rejects `i >= window length`.
///
/// Built when a `FacetConfig` carries a `window` field — the documented
/// sub-ordinal suffix model used by sized profiles to express "first
/// `base_count` rows of the shared base file" without having to copy
/// the file or trust every consumer to honor `view.base_count()`.
struct WindowedVectorReader<T> {
    inner: Box<dyn VectorReader<T>>,
    start: usize,
    /// Window length, capped to the underlying file's count.
    len: usize,
}

impl<T> WindowedVectorReader<T> {
    fn new(inner: Box<dyn VectorReader<T>>, start: usize, end: usize) -> Self {
        let total = inner.count();
        let s = start.min(total);
        let e = end.min(total);
        let len = e.saturating_sub(s);
        WindowedVectorReader {
            inner,
            start: s,
            len,
        }
    }
}

impl<T: Send + Sync> VectorReader<T> for WindowedVectorReader<T> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn count(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> std::result::Result<Vec<T>, IoError> {
        if index >= self.len {
            return Err(IoError::InvalidFormat(format!(
                "index {} out of range for windowed reader (len {}, start {})",
                index, self.len, self.start,
            )));
        }
        self.inner.get(self.start + index)
    }
    fn get_slice(&self, index: usize) -> Option<&[T]> {
        (index < self.len).then(|| self.inner.get_slice(self.start + index)).flatten()
    }
    fn read_into(&self, start: usize, out: &mut [T]) -> std::result::Result<usize, IoError>
    where
        T: Copy,
    {
        let n = crate::io::records_to_read(start, out.len(), self.dim(), self.len)?;
        self.inner.read_into(self.start + start, &mut out[..n * self.dim()])
    }
    fn precache(&self) -> std::io::Result<()> {
        self.inner.precache()
    }
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    fn as_slice(&self) -> Option<&[T]> {
        let dim = self.inner.dim();
        self.inner.as_slice().map(|all| &all[self.start * dim..(self.start + self.len) * dim])
    }
    fn conversion(&self) -> crate::typed_access::ElementConversion {
        self.inner.conversion()
    }
    fn native_element_type(&self) -> Option<crate::typed_access::ElementType> {
        self.inner.native_element_type()
    }
}

/// Adapt a [`FetchProgress`](crate::fetch::FetchProgress) stream to the
/// per-facet callback the deprecated `prebuffer_*_with_progress` forms
/// take: `(profile, facet, progress)`, with `profile` empty for a
/// single view.
pub(crate) fn prebuffer_progress_adapter<'c>(
    cb: &'c mut dyn FnMut(&str, &str, &PrebufferProgress),
) -> impl FnMut(&crate::fetch::FetchEvent<'_>) + 'c {
    use crate::fetch::FetchEvent;
    let mut fired = false;
    let (mut total, mut total_chunks) = (0u64, 0u32);
    move |event: &FetchEvent<'_>| match event {
        FetchEvent::FacetBegin { facet, bytes, chunks, .. } => {
            fired = false;
            (total, total_chunks) = (*bytes, *chunks);
            let p = PrebufferProgress {
                verified_chunks: 0,
                total_chunks,
                verified_bytes: 0,
                total_bytes: total,
            };
            cb(facet.profile.as_deref().unwrap_or(""), &facet.facet, &p);
        }
        FetchEvent::Progress { facet, bytes, total, chunks } => {
            fired = true;
            let p = PrebufferProgress {
                verified_chunks: *chunks,
                total_chunks: total_chunks.max(*chunks),
                verified_bytes: *bytes,
                total_bytes: *total,
            };
            cb(facet.profile.as_deref().unwrap_or(""), &facet.facet, &p);
        }
        FetchEvent::FacetEnd { facet, .. } if !fired => {
            let p = PrebufferProgress {
                verified_chunks: total_chunks,
                total_chunks,
                verified_bytes: total,
                total_bytes: total,
            };
            cb(facet.profile.as_deref().unwrap_or(""), &facet.facet, &p);
        }
        _ => {}
    }
}


/// Snapshot of one facet's progress, as the deprecated
/// `prebuffer_*_with_progress` callbacks receive it. New code reads
/// [`FetchEvent`](crate::fetch::FetchEvent)s instead.
#[derive(Debug, Clone)]
pub struct PrebufferProgress {
    /// Chunks fetched and verified so far.
    pub verified_chunks: u32,
    /// Chunks the facet's fetch covers.
    pub total_chunks: u32,
    /// Bytes fetched so far.
    pub verified_bytes: u64,
    /// Bytes the facet's fetch covers.
    pub total_bytes: u64,
}

/// How to open one of a series' files, recorded at handle creation so
/// the open itself can be deferred (SH-26).
///
/// Resolution — catalog anchoring, cache relpath, collision rejection —
/// happens once, up front, because it is string work and its errors
/// belong at handle creation. Only the `Storage::open` is lazy.
#[derive(Debug, Clone)]
pub(crate) enum ShardOpen {
    /// Catalog-anchored: cached under `<root>/<dataset>/<relpath>`,
    /// `root` the catalog's named cache directory or, without one, the
    /// configured cache.
    Layered {
        resolved: String,
        dataset: String,
        relpath: String,
        home: String,
        cache_dir: Option<std::path::PathBuf>,
    },
    /// Plain open by resolved path or URL.
    Plain { resolved: String },
}

impl ShardOpen {
    fn open(&self) -> std::io::Result<std::sync::Arc<crate::storage::Storage>> {
        match self {
            Self::Layered {
                resolved,
                dataset,
                relpath,
                home,
                cache_dir,
            } => crate::storage::Storage::open_layered(resolved, dataset, relpath, home, cache_dir.as_deref()),
            Self::Plain { resolved } => crate::storage::Storage::open(resolved),
        }
    }

    fn resolved(&self) -> &str {
        match self {
            Self::Layered { resolved, .. } | Self::Plain { resolved } => resolved,
        }
    }
}

/// How many of a series' files may be open at once (SH-59).
///
/// **Derived from the process file-descriptor limit, not hardcoded.** A
/// fixed constant is wrong in both directions: it strands descriptors on
/// a generous host and thrashes on a constrained one, and it is
/// invisible to the operator who raised `ulimit -n` precisely to make
/// this work.
///
/// A quarter of the soft limit, leaving the rest for everything else the
/// process holds open — transport sockets, other facets, sidecars. The
/// fraction is an implementation constant; the limit it applies to is
/// not.
pub fn open_file_cap() -> usize {
    resolve_open_file_cap(
        std::env::var("VECTORDATA_SHARD_FD_CAP").ok().as_deref(),
        fd_soft_limit(),
    )
}

/// The cap, given the two things that decide it.
///
/// Split from [`open_file_cap`] so it is answerable without a process
/// to configure: the environment value and the limit arrive as
/// arguments, which is the only way to test the derivation without
/// mutating the environment out from under a parallel test.
///
/// An explicit `VECTORDATA_SHARD_FD_CAP` wins outright — an operator
/// setting it knows something the fraction cannot. A value that is not
/// a positive integer is ignored rather than fatal: a typo in an
/// environment variable should not stop a dataset from opening.
pub(crate) fn resolve_open_file_cap(env: Option<&str>, soft_limit: Option<u64>) -> usize {
    const FRACTION: u64 = 4;
    const FLOOR: usize = 8;
    if let Some(n) = env
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    let soft = soft_limit.unwrap_or(1024);
    ((soft / FRACTION) as usize).max(FLOOR)
}

/// The process's soft `RLIMIT_NOFILE`, if it can be read.
fn fd_soft_limit() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `getrlimit` writes into a fully-initialised struct we
        // own; no aliasing and no lifetime concerns.
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) } == 0 {
            return Some(lim.rlim_cur);
        }
        None
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Whether the open-file cap would bind on fetch concurrency (SH-77).
///
/// The cap is a **backstop, never a throttle**: it is expected to sit
/// above whatever the transport wants to run in parallel. When it does
/// not, that is a misconfiguration and must be *reported* — silently
/// reducing parallelism surfaces as unexplained slowness with no line
/// of output pointing at `ulimit`.
pub(crate) fn warn_if_cap_binds_concurrency() {
    let cap = open_file_cap();
    let want = crate::cache::download_concurrency();
    if cap < want {
        log::warn!(
            "open-file cap {cap} (from RLIMIT_NOFILE) is below the download \
             concurrency {want}: shard fetches will be limited by descriptors \
             rather than by the link. Raise `ulimit -n`, or lower \
             VECTORDATA_DOWNLOAD_CONCURRENCY to match."
        );
    }
}

/// One distinct file a series draws from.
///
/// Held behind a `Mutex<Option<..>>` rather than a `OnceLock` because an
/// open file may be **evicted**: a series larger than the descriptor
/// budget must still be readable, and that means the set of open files
/// changes over the handle's life (SH-59).
#[derive(Debug)]
struct SeriesFile {
    spec: ShardOpen,
    opened: std::sync::Mutex<Option<std::sync::Arc<crate::storage::Storage>>>,
}

/// A facet's shards, and the files behind them.
///
/// **Shards and files are counted separately** (SH-81): the ordinal
/// model has one entry per shard, while storage has one entry per
/// *distinct file*. Two shards sliced from one file share a `Storage`,
/// a cache file, a descriptor, and a fetch — so anything byte-shaped
/// here iterates `files`, and anything ordinal-shaped iterates shards.
// Consumed by the reader and prefetch layers, which arrive in the next
// steps; until then the ordinal side of a series has no caller. Remove
// this attribute when they land — it covers a staged landing, not a
// permanently unused surface.
#[derive(Debug)]
pub struct Series {
    shards: crate::dataset::shards::Shards,
    files: Vec<SeriesFile>,
    /// Shard index → index into `files`.
    file_of_shard: Vec<usize>,
    /// Summed file sizes, computed on first ask. Answering it opens
    /// every file, so a caller that never asks never pays.
    total_bytes: std::sync::OnceLock<u64>,
    /// Open files, most-recently-used last. Bounded by
    /// [`open_file_cap`]; the least-recently-used is closed when the
    /// budget is reached (SH-59).
    ///
    /// This list is the authority on what the series holds: a file is
    /// installed in its slot and recorded here under one lock, so the
    /// two can never disagree and the bound holds under concurrent
    /// readers.
    lru: std::sync::Mutex<std::collections::VecDeque<usize>>,
    /// How many files may be open at once.
    cap: usize,
    /// Per-**file** offset indexes for variable-length shards (SH-82).
    ///
    /// One index per file, not per shard: two shards sliced from one
    /// file read that file's index at their own offsets, and load it
    /// once between them.
    offsets: Vec<std::sync::OnceLock<std::sync::Arc<Vec<u64>>>>,
}

impl Series {
    /// Build from the ordinal model and a per-shard open spec,
    /// collapsing shards that name one file onto one entry.
    pub(crate) fn new(shards: crate::dataset::shards::Shards, per_shard: Vec<ShardOpen>) -> Self {
        Self::with_cap(shards, per_shard, open_file_cap())
    }

    /// As [`Self::new`], with the descriptor budget supplied.
    ///
    /// The budget is a property of the process, so `new` reads it from
    /// the process. Naming it here is what lets eviction be exercised
    /// at a size a test can build: with the real budget a test would
    /// need more shards than the host has descriptors, so the branch
    /// that closes a file would never run.
    pub(crate) fn with_cap(
        shards: crate::dataset::shards::Shards,
        per_shard: Vec<ShardOpen>,
        budget: usize,
    ) -> Self {
        let mut files: Vec<SeriesFile> = Vec::new();
        let mut file_of_shard = Vec::with_capacity(per_shard.len());
        for spec in per_shard {
            let existing = files
                .iter()
                .position(|f| f.spec.resolved() == spec.resolved());
            let idx = match existing {
                Some(i) => i,
                None => {
                    files.push(SeriesFile {
                        spec,
                        opened: std::sync::Mutex::new(None),
                    });
                    files.len() - 1
                }
            };
            file_of_shard.push(idx);
        }
        let offsets = (0..files.len())
            .map(|_| std::sync::OnceLock::new())
            .collect();
        // A series never needs more descriptors than it has files, so
        // the cap is the smaller of the budget and the series.
        if files.len() > budget {
            warn_if_cap_binds_concurrency();
        }
        let cap = budget.min(files.len().max(1));
        Self {
            shards,
            files,
            file_of_shard,
            total_bytes: std::sync::OnceLock::new(),
            offsets,
            lru: std::sync::Mutex::new(std::collections::VecDeque::new()),
            cap,
        }
    }

    /// The published offset index for file `i`, loaded once.
    ///
    /// Restricted to what is already cheap, exactly as the single-file
    /// path is: planning must not walk-download a shard to price it
    /// (SH-21).
    pub(crate) fn published_offsets(
        &self,
        i: usize,
        source: &str,
        elem_size: usize,
    ) -> Option<std::sync::Arc<Vec<u64>>> {
        if let Some(o) = self.offsets.get(i)?.get() {
            return Some(o.clone());
        }
        let storage = self.file(i).ok()?;
        let loaded = crate::io::load_offsets(source, &storage, elem_size).ok()?;
        let _ = self.offsets[i].set(std::sync::Arc::new(loaded));
        self.offsets.get(i)?.get().cloned()
    }

    /// The ordinal model.
    pub(crate) fn shards(&self) -> &crate::dataset::shards::Shards {
        &self.shards
    }

    /// How many distinct files back this series — never more than the
    /// shard count, and fewer when shards share a file.
    pub(crate) fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Open (or reuse) the storage for file `i`.
    ///
    /// Returns an owned handle rather than a borrow: a file may be
    /// evicted while a caller still holds its storage, and the `Arc`
    /// keeps that read valid until the caller is done (SH-59).
    pub(crate) fn file(&self, i: usize) -> Result<std::sync::Arc<crate::storage::Storage>> {
        let slot = self
            .files
            .get(i)
            .ok_or_else(|| Error::Other(format!("shard file {i} out of range")))?;

        // The recency list is taken before a slot, and that order is
        // never reversed in this type — an eviction holds the list
        // while it closes its victims, so a slot lock is only ever
        // acquired by a thread that already holds the list or holds
        // nothing.
        {
            let mut lru = self.lru.lock().expect("lru");
            if let Some(s) = slot.opened.lock().expect("shard slot").as_ref() {
                let s = s.clone();
                Self::touch_locked(&mut lru, i);
                return Ok(s);
            }
        }

        // Opening happens outside every lock. It may hit the filesystem
        // or the network, and a series is meant to fetch its shards in
        // parallel — holding the recency list across an open would
        // serialise exactly what sharding exists to spread out. Two
        // threads may therefore open the same file at once; one
        // installs and the other's handle is dropped.
        let opened = slot.spec.open().map_err(|e| {
            Error::Other(format!("open shard file '{}': {e}", slot.spec.resolved()))
        })?;

        // Install, record, and evict as **one** critical section. Any
        // of the three alone lets the open set and the recency list
        // disagree, and the descriptor bound is precisely their
        // agreement: two threads opening different shards would each
        // install before either evicted, and the series would hold
        // more files than its budget.
        //
        // The bound is on files this series *holds*. A handle opened
        // above but not yet installed is a descriptor in flight, and
        // those are bounded by fetch concurrency rather than by the
        // cap — the case `warn_if_cap_binds_concurrency` reports.
        let mut lru = self.lru.lock().expect("lru");
        Self::touch_locked(&mut lru, i);
        while lru.len() > self.cap {
            let Some(victim) = lru.pop_front() else { break };
            // The file just opened is the most recently used, so it is
            // never the victim. Asserting that here rather than relying
            // on it keeps a cap of one from closing what it just
            // opened, which would make the series unable to advance.
            if victim == i {
                lru.push_back(victim);
                break;
            }
            if let Some(slot) = self.files.get(victim) {
                *slot.opened.lock().expect("shard slot") = None;
            }
        }
        // Installed **after** the eviction, so the set of held files
        // never passes through a state above the budget. Installing
        // first and trimming after would hold cap+1 descriptors for the
        // width of the eviction loop. Nothing outside can observe that
        // window — every observer goes through the same list — but on a
        // host where the cap is the real ceiling, the descriptor is
        // claimed whether or not anyone is counting.
        let held = {
            let mut slotted = slot.opened.lock().expect("shard slot");
            slotted.get_or_insert(opened).clone()
        };
        Ok(held)
    }

    /// Mark file `i` as most recently used, under a list already held.
    ///
    /// Takes the guard rather than the lock so a caller can record a
    /// use and act on the result without releasing the list in between.
    fn touch_locked(lru: &mut std::collections::VecDeque<usize>, i: usize) {
        if let Some(pos) = lru.iter().position(|&x| x == i) {
            lru.remove(pos);
        }
        lru.push_back(i);
    }

    /// The byte range shard `s` can address in its file.
    ///
    /// The whole file for a whole-file shard; the window's extent for a
    /// sliced one. This is what makes residency mean **addressable**
    /// bytes rather than every byte of every file (SH-92) — a facet
    /// slicing a tenth of a large file would otherwise report incomplete
    /// forever unless the other nine tenths were downloaded.
    pub(crate) fn shard_byte_extent(&self, s: usize) -> Option<(u64, u64)> {
        let entry = self.shards.entries().get(s)?;
        let i = self.file_index_of_shard(s).ok()?;
        let storage = self.file(i).ok()?;
        let (lo, hi) = (entry.file_base, entry.file_base + entry.len);
        let m = map_in_file(&entry.source.path, lo, hi, &storage, &|elem| {
            self.published_offsets(i, &entry.source.path, elem)
        })?;
        Some((m.0, m.1))
    }

    /// The resolved source of file `i` — the string it was opened
    /// from, which is what classification reads.
    pub(crate) fn file_source(&self, i: usize) -> &str {
        self.files[i].spec.resolved()
    }

    /// How many files are open right now.
    ///
    /// Never exceeds the descriptor budget (SH-59) — that bound is what
    /// this exists to make observable.
    ///
    /// Counted under the recency list, not by walking the slots freely.
    /// A free walk answers from a torn view: it can read one slot
    /// before an eviction closes it and a later slot after the open
    /// that displaced it, and report a total that was never held at any
    /// instant. Holding the list makes the answer a real moment,
    /// because every install and every eviction passes through it.
    pub fn open_file_count(&self) -> usize {
        let _lru = self.lru.lock().expect("lru");
        self.files
            .iter()
            .filter(|f| f.opened.lock().expect("shard slot").is_some())
            .count()
    }

    /// Which file backs shard `s`.
    pub(crate) fn file_index_of_shard(&self, s: usize) -> Result<usize> {
        self.file_of_shard
            .get(s)
            .copied()
            .ok_or_else(|| Error::Other(format!("shard {s} out of range")))
    }

    /// Every file, opened. Used by the whole-facet operations —
    /// precache, residency, total size — which are byte-shaped and so
    /// per file (SH-81).
    fn all_files(&self) -> Result<Vec<std::sync::Arc<crate::storage::Storage>>> {
        (0..self.files.len()).map(|i| self.file(i)).collect()
    }

    /// Summed size of the distinct files, counting a shared file once.
    ///
    /// Fallible on purpose. A declared shard that will not open is a
    /// broken facet, and the infallible answer for one is `0` — which is
    /// indistinguishable from an empty facet and is exactly the silent
    /// shape this design forbids.
    fn try_total_size(&self) -> Result<u64> {
        if let Some(n) = self.total_bytes.get() {
            return Ok(*n);
        }
        let n: u64 = self.all_files()?.iter().map(|s| s.total_size()).sum();
        let _ = self.total_bytes.set(n);
        Ok(n)
    }
}

/// Low-level: an open handle on a facet's bytes — one file, or every
/// shard of a series — for cache inspection and range fetches. Most
/// callers want [`TestDataView::fetch`] to bring bytes in and a reader
/// ([`TestDataViewExt::facet_as`] and the per-facet `*_as` calls) to read them.
///
/// There is no public API for reading bytes through this handle; it
/// answers what is resident ([`cache_stats`](Self::cache_stats),
/// [`range_fill`](Self::range_fill)) and moves bytes on request.
pub struct FacetStorage {
    storage: std::sync::Arc<crate::storage::Storage>,
    /// The remaining shards, when this facet is a series.
    ///
    /// `None` for a single file — which is every facet written before
    /// multi-file support, and the case that must add no indirection
    /// (SH-73). Every method below branches once on this and then runs
    /// exactly the code it ran before.
    series: Option<std::sync::Arc<Series>>,
    /// Record offsets for variable-length facets, loaded once per
    /// handle.
    ///
    /// Deliberately per-handle rather than process-wide. The index is
    /// eight bytes per record — 8 GB at a billion — so a global cache
    /// would be an unbounded one, and the point at which it should be
    /// dropped is a question only the caller can answer. Holding a
    /// handle is how a caller says "I am going to ask about this facet
    /// repeatedly"; dropping it is how they say they are done.
    offsets: std::sync::OnceLock<CachedOffsets>,
}

/// A loaded offset index, with the element width it was loaded for.
///
/// The width is carried so a handle reused across element types cannot
/// silently answer with offsets computed for a different stride — that
/// would resolve a window to plausible, wrong bytes.
#[derive(Debug)]
struct CachedOffsets {
    elem_size: usize,
    offsets: std::sync::Arc<Vec<u64>>,
}

impl FacetStorage {
    pub(crate) fn new(storage: std::sync::Arc<crate::storage::Storage>) -> Self {
        Self {
            storage,
            series: None,
            offsets: std::sync::OnceLock::new(),
        }
    }

    /// A handle over a multi-file series.
    ///
    /// `storage` is the first shard's file, so every single-file path
    /// below keeps working against something real rather than a
    /// placeholder — but any *whole-facet* answer comes from `series`.
    pub(crate) fn series(first: std::sync::Arc<crate::storage::Storage>, series: Series) -> Self {
        Self {
            storage: first,
            series: Some(std::sync::Arc::new(series)),
            offsets: std::sync::OnceLock::new(),
        }
    }

    /// The series behind this facet, if it has one.
    pub fn series_ref(&self) -> Option<&std::sync::Arc<Series>> {
        self.series.as_ref()
    }

    /// The record-offset index for a variable-length facet, loaded on
    /// first use and reused for the life of this handle.
    ///
    /// Only cheap sources: a published sidecar, a persisted rebuild, or
    /// a walk of a complete local copy. This handle exists to answer
    /// *planning* questions ("what would this window cost?"), and the
    /// remaining way to build an index for a remote vvec is to walk the
    /// file, which downloads all of it — and a plan that transfers the
    /// facet in order to report the cost of transferring part of it is
    /// not a plan. Readers follow the same rule.
    ///
    /// `None` when no such index is available, or when this handle
    /// already holds offsets for a different element width — see
    /// [`CachedOffsets`].
    pub(crate) fn published_offsets(
        &self,
        source: &str,
        elem_size: usize,
    ) -> Option<std::sync::Arc<Vec<u64>>> {
        if let Some(cached) = self.offsets.get() {
            return (cached.elem_size == elem_size).then(|| cached.offsets.clone());
        }
        let loaded = crate::io::load_offsets(source, &self.storage, elem_size).ok()?;
        // A race here means two loads and one winner, which is wasteful
        // but never wrong — both produce the same offsets. Return what
        // is *cached* rather than what this call built, so every caller
        // observes the same allocation.
        let _ = self.offsets.set(CachedOffsets {
            elem_size,
            offsets: std::sync::Arc::new(loaded),
        });
        let winner = self.offsets.get()?;
        (winner.elem_size == elem_size).then(|| winner.offsets.clone())
    }

    /// Read the first `len` bytes of the facet (clamped to its size).
    /// Crate-internal shape probe: pulls the covering chunk on remote
    /// storage, which is exactly what `datasets ping` wants — after a
    /// ping, the cache survey can derive record counts from the now-
    /// resident header.
    pub(crate) fn read_prefix(&self, len: u64) -> std::io::Result<Vec<u8>> {
        let len = len.min(self.total_size());
        self.storage.read_bytes(0, len)
    }
    /// Low-level: download everything this facet can address. Most
    /// callers want [`TestDataView::fetch`], which plans, checks cache
    /// space and reports progress.
    ///
    /// **Strict**: returns `Err` on any failure — never silently
    /// leaves the facet in a partially-resident state. After
    /// `Ok(())`, every read on every reader against this source
    /// (this `FacetStorage` and any other) takes the zero-copy
    /// path on next access.
    pub fn precache(&self) -> std::io::Result<()> {
        match &self.series {
            None => self.storage.precache(),
            // Drives exactly what the facet can address (SH-92). A
            // whole-file shard asks for the whole file, so the common
            // case is unchanged; a sliced one asks only for its window
            // rather than dragging bytes it can never read.
            Some(s) => {
                for shard in 0..s.shards().entries().len() {
                    match s.shard_byte_extent(shard) {
                        Some((lo, hi)) => {
                            self.prebuffer_shard_range(shard, lo, hi, |_| {})?;
                        }
                        None => {
                            let i = s
                                .file_index_of_shard(shard)
                                .map_err(|e| std::io::Error::other(e.to_string()))?;
                            s.file(i)
                                .map_err(|e| std::io::Error::other(e.to_string()))?
                                .precache()?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
    /// Low-level: [`precache`](Self::precache) with the transport's
    /// per-file progress. Most callers want [`TestDataView::fetch`].
    ///
    /// **Walks the series**, exactly as [`Self::precache`] does.
    /// `self.storage` is the first shard's, so delegating to it would
    /// fetch one file and return — leaving a facet that reports itself
    /// precached while most of it is still a hole, which is the worst
    /// of the three possible outcomes.
    ///
    /// Progress arrives **per file**: each shard's fetch reports its
    /// own totals, so a meter driven by this advances once per shard
    /// rather than once across the series. That is a display question;
    /// the alternative — synthesising a combined `DownloadProgress`
    /// the transport never produced — would misreport what is actually
    /// in flight.
    pub fn prebuffer_with_progress<F>(&self, mut cb: F) -> std::io::Result<()>
    where
        F: FnMut(&crate::DownloadProgress),
    {
        self.prebuffer_parts(&mut |_, p| cb(p))
    }

    /// [`Self::prebuffer_with_progress`], telling the callback which
    /// part each progress snapshot belongs to: the shard index for a
    /// series, `0` for a single file.
    ///
    /// Each part's fetch reports its own totals, so a caller summing
    /// bytes across a series needs to know where one part ends and the
    /// next begins. Guessing that from the numbers alone (a total that
    /// changes, a count that drops) misreads two equal-sized shards as
    /// one.
    pub(crate) fn prebuffer_parts(
        &self,
        cb: &mut dyn FnMut(usize, &crate::DownloadProgress),
    ) -> std::io::Result<()> {
        let Some(s) = &self.series else {
            return self.storage.prebuffer_with_progress(|p| cb(0, p));
        };
        for shard in 0..s.shards().entries().len() {
            // Addressable bytes, not whole files (SH-92): a sliced
            // shard pulls its window, not bytes it can never read.
            match s.shard_byte_extent(shard) {
                Some((lo, hi)) => self.prebuffer_shard_range(shard, lo, hi, |p| cb(shard, p))?,
                None => {
                    let i = s
                        .file_index_of_shard(shard)
                        .map_err(|e| std::io::Error::other(e.to_string()))?;
                    s.file(i)
                        .map_err(|e| std::io::Error::other(e.to_string()))?
                        .prebuffer_with_progress(|p| cb(shard, p))?;
                }
            }
        }
        Ok(())
    }

    /// Low-level: fetch the chunks covering bytes
    /// `[byte_start, byte_end)` of the facet's joined byte space. Most
    /// callers want [`TestDataView::fetch`] with a record window, which
    /// maps records to bytes for them. Used by windowed
    /// profile precache so a `:1m` window against a 1.3 TiB base
    /// only pulls the chunks for the window's byte range. The
    /// view layer computes the byte range from the window record
    /// count and the format's bytes-per-record.
    ///
    /// **Walks the series.** The facet's byte space is its shards' files
    /// laid end to end (SH-38), and `self.storage` is only the first of
    /// them: delegating to it would fetch a range in shard 0 and
    /// silently do nothing for a range that lives in any later shard.
    /// The explorer's prefetcher found that out on tessera, where every
    /// chunk past the first shard fell back to the reader's serial
    /// on-demand fetch.
    pub fn prebuffer_range_with_progress<F>(
        &self,
        byte_start: u64,
        byte_end: u64,
        mut cb: F,
    ) -> std::io::Result<()>
    where
        F: FnMut(&crate::DownloadProgress),
    {
        let Some(s) = &self.series else {
            return self
                .storage
                .prebuffer_range_with_progress(byte_start, byte_end, cb);
        };
        let mut file_start = 0u64;
        for i in 0..s.file_count() {
            if file_start >= byte_end {
                break;
            }
            let storage = s
                .file(i)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let file_end = file_start + storage.total_size();
            if file_end > byte_start && file_start < byte_end {
                let from = byte_start.saturating_sub(file_start);
                let to = byte_end.min(file_end) - file_start;
                storage.prebuffer_range_with_progress(from, to, &mut cb)?;
            }
            file_start = file_end;
        }
        Ok(())
    }

    /// [`Self::prebuffer_range_with_progress`] against a named shard's
    /// file — the byte range is in that file's own coordinates.
    pub(crate) fn prebuffer_shard_range<F>(
        &self,
        shard: usize,
        byte_start: u64,
        byte_end: u64,
        cb: F,
    ) -> std::io::Result<()>
    where
        F: FnMut(&crate::DownloadProgress),
    {
        let storage = match &self.series {
            None => self.storage.clone(),
            Some(s) => {
                let i = s
                    .file_index_of_shard(shard)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                s.file(i)
                    .map_err(|e| std::io::Error::other(e.to_string()))?
            }
        };
        storage.prebuffer_range_with_progress(byte_start, byte_end, cb)
    }
    /// Ask the upstream whether this facet's complete local copy is
    /// still what it publishes: `Ok(true)` when it is (or the facet is
    /// local, or not yet complete — then the open itself went online),
    /// `Ok(false)` when the server could not be reached and the copy
    /// stands on its own, an error when the upstream has changed.
    pub(crate) fn revalidate(&self) -> std::io::Result<bool> {
        use crate::storage::Revalidation;
        let parts: Vec<std::sync::Arc<crate::storage::Storage>> = match &self.series {
            None => vec![self.storage.clone()],
            Some(s) => (0..s.file_count())
                .map(|i| s.file(i).map_err(|e| std::io::Error::other(e.to_string())))
                .collect::<std::io::Result<_>>()?,
        };
        let mut reached = true;
        for p in parts {
            if p.revalidate()? == Revalidation::Unreachable {
                reached = false;
            }
        }
        Ok(reached)
    }

    /// Whether every byte this facet can address is resident.
    ///
    /// For whole-file shards — every form but a sliced series — that is
    /// the conjunction over the files it draws from. A *sliced* facet
    /// should answer over its addressable range only (SH-92); that
    /// needs byte ranges, which arrive with the reader layer, and no
    /// writer produces a sliced series yet.
    pub fn is_complete(&self) -> bool {
        match &self.series {
            None => self.storage.is_complete(),
            Some(s) => (0..s.shards().entries().len()).all(|shard| {
                // Addressable bytes, not whole files (SH-92). For a
                // whole-file shard the two coincide, so the common case
                // is unchanged; where they diverge, the addressable
                // range is what "complete" has to mean, or a sliced
                // facet can never reach it.
                match s.shard_byte_extent(shard) {
                    Some((lo, hi)) => self
                        .shard_range_fill(shard, lo, hi)
                        .is_none_or(|f| f.is_resident()),
                    // No mapping to work from: fall back to the file,
                    // which is the conservative answer.
                    None => s
                        .file_index_of_shard(shard)
                        .and_then(|i| s.file(i))
                        .is_ok_and(|f| f.is_complete()),
                }
            }),
        }
    }
    /// The storage behind this facet.
    ///
    /// A reader that parses a container of its own reads through this
    /// rather than mapping a file, which is what keeps such a facet
    /// incremental: a chunked, merkle-verified source fetches and
    /// verifies the chunks covering each range as it is asked for.
    pub(crate) fn storage_handle(&self) -> std::sync::Arc<crate::storage::Storage> {
        self.storage.clone()
    }

    /// The access mode this facet promises, which is the weakest among
    /// its files (SH-93).
    ///
    /// A facet's mode is a promise about every read it will serve. A
    /// series whose shards were published differently — one with a
    /// `.mref`, one without, one on a server with no range support —
    /// must answer for the worst of them, or a caller plans against an
    /// average it will not get. Understating a good shard costs
    /// efficiency; overstating a bad one costs correctness.
    ///
    /// Per-file detail stays available to `describe` and the explore
    /// TUI, where it is information rather than a promise (SH-46).
    /// `source` is the facet's declared source, which is all a
    /// single-file facet needs. A series ignores it — it is the `NNNN`
    /// pattern, which names no file — and classifies each shard from
    /// the source it was actually opened from.
    pub fn access_mode(
        &self,
        source: &str,
        cache_dir: &std::path::Path,
    ) -> crate::access::AccessMode {
        let Some(s) = &self.series else {
            return crate::access::AccessMode::classify(source, cache_dir);
        };
        crate::access::AccessMode::weakest(
            (0..s.file_count())
                .map(|i| crate::access::AccessMode::classify(s.file_source(i), cache_dir)),
        )
        .unwrap_or(crate::access::AccessMode::Local)
    }

    /// Whether reads make no network requests: a local source file, or
    /// a remote one whose cache copy is complete and memory-mapped.
    pub fn is_local(&self) -> bool {
        match &self.series {
            None => self.storage.is_local(),
            Some(s) => (0..s.file_count()).all(|i| s.file(i).is_ok_and(|f| f.is_local())),
        }
    }
    /// Bytes this facet occupies, summed over the distinct files it
    /// draws from (a shared file counted once).
    ///
    /// Returns `0` for a series with an unopenable shard, having logged
    /// why. That is the only answer an infallible accessor can give, and
    /// it is why [`Self::try_total_size`] exists: every caller that can
    /// report a failure should use that one instead.
    pub fn total_size(&self) -> u64 {
        match self.try_total_size() {
            Ok(n) => n,
            Err(e) => {
                log::error!("facet size unavailable: {e}");
                0
            }
        }
    }

    /// [`Self::total_size`], propagating the reason a series cannot be
    /// sized — a declared shard that is absent or will not open.
    pub fn try_total_size(&self) -> Result<u64> {
        match &self.series {
            None => Ok(self.storage.total_size()),
            Some(s) => s.try_total_size(),
        }
    }

    /// Records this facet holds, when its layout states them.
    ///
    /// `None` for a single file, whose count is the reader's to derive
    /// from the bytes; `Some` for a series, whose declaration states it.
    #[allow(dead_code)]
    pub(crate) fn record_count(&self) -> Option<u64> {
        self.series.as_ref().map(|s| s.shards().count())
    }

    /// Low-level: the path of the local file behind this facet, for
    /// diagnostics and cache administration — showing where a facet is
    /// cached, measuring it. **Not an access path for the data.** A
    /// cache file is pre-sized and sparse until complete, its validity
    /// lives in a separate chunk bitmap, and a series spans several
    /// files; only vectordata's readers account for all of that. Read
    /// through [`TestDataViewExt::base_vectors_as`], [`TestDataViewExt::facet_as`]
    /// and the other facet readers, which are zero-copy once the facet
    /// is fetched.
    ///
    /// `Some` for cached remote storage and for local source files;
    /// `None` for direct HTTP before its first fetch. **For a series
    /// this is the first shard's file only**; see
    /// [`local_files`](Self::local_files).
    pub fn cache_path(&self) -> Option<std::path::PathBuf> {
        self.storage.local_path()
    }

    /// Low-level: the local file behind every file this facet spans, in
    /// order — one per shard for a series — for tools that manage files
    /// (hash, copy, inspect the cache). **Not an access path for the
    /// data:** read vectors through the facet's readers
    /// ([`TestDataViewExt::base_vectors_as`], [`TestDataViewExt::facet_as`], …), which
    /// are zero-copy once the facet is fetched and know the record
    /// layout, shard boundaries and windows that a raw file does not.
    /// `None` when any file has no local copy yet.
    pub fn local_files(&self) -> Option<Vec<std::path::PathBuf>> {
        match &self.series {
            None => self.storage.local_path().map(|p| vec![p]),
            Some(s) => (0..s.file_count())
                .map(|i| s.file(i).ok().and_then(|f| f.local_path()))
                .collect(),
        }
    }

    /// Bytes of this facet already held, summed over its shards, as the
    /// chunk state records them: each resident chunk at its real length.
    /// A file with no chunk state — a local file, or a cache copy opened
    /// complete — is held whole. Never read from the cache file's size:
    /// it is pre-sized to the whole facet, and where the platform
    /// reports no sparse allocation that size says everything is held.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.shard_sizes()
            .into_iter()
            .enumerate()
            .map(|(shard, size)| match self.shard_range_fill(shard, 0, size) {
                Some(fill) => fill.resident_bytes,
                None => size,
            })
            .sum()
    }

    /// Live cache-fill statistics. Reports chunk fill for both the
    /// merkle-verified (`.mref`) and merkle-less chunked-HTTP remote
    /// paths, so progress UIs work no matter which transport the
    /// open resolved to. Returns `None` only for local mmap storage,
    /// which is always fully resident.
    /// What fetching `[byte_start, byte_end)` would cost, without
    /// fetching any of it.
    ///
    /// `None` for storage with no chunks — a local mmap is resident by
    /// definition and a fully-downloaded cache file has nothing left to
    /// plan. Callers should treat `None` as "free".
    /// Byte size of each shard, in shard order.
    ///
    /// One entry for a single-file facet — the facet's own size — so a
    /// caller never has to ask which case it is in.
    pub(crate) fn shard_sizes(&self) -> Vec<u64> {
        match &self.series {
            None => vec![self.storage.total_size()],
            Some(s) => (0..s.shards().entries().len())
                .map(|shard| {
                    s.file_index_of_shard(shard)
                        .and_then(|i| s.file(i).map(|f| f.total_size()))
                        .unwrap_or(0)
                })
                .collect(),
        }
    }

    /// [`Self::range_fill`] against a named shard's file.
    pub(crate) fn shard_range_fill(
        &self,
        shard: usize,
        byte_start: u64,
        byte_end: u64,
    ) -> Option<RangeFill> {
        let storage = match &self.series {
            None => self.storage.clone(),
            Some(s) => {
                let i = s.file_index_of_shard(shard).ok()?;
                s.file(i).ok()?
            }
        };
        let (first, last, chunk_size, resident, resident_bytes) = storage.range_fill(byte_start, byte_end)?;
        let aligned_start = first as u64 * chunk_size;
        let aligned_end = ((last as u64 + 1) * chunk_size).min(storage.total_size());
        Some(RangeFill {
            first_chunk: first,
            last_chunk: last,
            chunk_size,
            chunks: last - first + 1,
            chunks_resident: resident,
            resident_bytes,
            aligned_start,
            aligned_end,
        })
    }

    /// What fetching bytes `[byte_start, byte_end)` would cost, at chunk
    /// granularity, and how much of it is already resident. `None` for
    /// storage without chunks (local files).
    pub fn range_fill(&self, byte_start: u64, byte_end: u64) -> Option<RangeFill> {
        let (first, last, chunk_size, resident, resident_bytes) = self.storage.range_fill(byte_start, byte_end)?;
        let aligned_start = first as u64 * chunk_size;
        let aligned_end = ((last as u64 + 1) * chunk_size).min(self.total_size());
        Some(RangeFill {
            first_chunk: first,
            last_chunk: last,
            chunk_size,
            chunks: last - first + 1,
            chunks_resident: resident,
            resident_bytes,
            aligned_start,
            aligned_end,
        })
    }

    /// Cache fill for the whole facet.
    ///
    /// **Summed across a series' files** (SH-81). `self.storage` is the
    /// first shard's, so reporting its numbers would say "37% cached"
    /// about a five-file facet on the strength of one file — a
    /// percentage that is right about nothing. Chunk counts and content
    /// bytes add; completeness is the conjunction, since a facet with
    /// one missing shard is not cached.
    ///
    /// `None` when no file is cache-backed (local mmap, direct HTTP).
    /// A series where only some files are cache-backed reports the ones
    /// that are, which is the same partial answer a single file gives
    /// while filling.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        let Some(series) = &self.series else {
            return self.storage.fill_stats().map(
                |(valid_chunks, total_chunks, chunk_size, content_size, is_complete)| CacheStats {
                    valid_chunks,
                    total_chunks,
                    chunk_size,
                    content_size,
                    is_complete,
                },
            );
        };
        let mut acc: Option<CacheStats> = None;
        for i in 0..series.file_count() {
            let Ok(storage) = series.file(i) else { continue };
            let Some((valid, total, chunk_size, content, complete)) = storage.fill_stats() else {
                continue;
            };
            acc = Some(match acc {
                None => CacheStats {
                    valid_chunks: valid,
                    total_chunks: total,
                    chunk_size,
                    content_size: content,
                    is_complete: complete,
                },
                Some(a) => CacheStats {
                    valid_chunks: a.valid_chunks + valid,
                    total_chunks: a.total_chunks + total,
                    // One chunk size across the series; files agreeing
                    // is the normal case, and the larger is the honest
                    // answer if they ever do not.
                    chunk_size: a.chunk_size.max(chunk_size),
                    content_size: a.content_size + content,
                    is_complete: a.is_complete && complete,
                },
            });
        }
        acc
    }
}

/// vectordata's facet handle as the API's [`FacetBytes`]: each method
/// is the inherent one of the same name, or, for content identity and
/// the cache check, the storage layer's, reporting as a [`ProgressSink`](crate::ProgressSink).
impl FacetBytes for FacetStorage {
    fn total_size(&self) -> u64 {
        FacetStorage::total_size(self)
    }

    fn try_total_size(&self) -> Result<u64> {
        FacetStorage::try_total_size(self)
    }

    fn is_complete(&self) -> bool {
        FacetStorage::is_complete(self)
    }

    fn is_local(&self) -> bool {
        FacetStorage::is_local(self)
    }

    fn is_series(&self) -> bool {
        self.series.is_some()
    }

    fn shard_sizes(&self) -> Vec<u64> {
        FacetStorage::shard_sizes(self)
    }

    fn resident_bytes(&self) -> u64 {
        FacetStorage::resident_bytes(self)
    }

    fn cache_path(&self) -> Option<std::path::PathBuf> {
        FacetStorage::cache_path(self)
    }

    fn local_files(&self) -> Option<Vec<std::path::PathBuf>> {
        FacetStorage::local_files(self)
    }

    fn cache_stats(&self) -> Option<CacheStats> {
        FacetStorage::cache_stats(self)
    }

    fn range_fill(&self, byte_start: u64, byte_end: u64) -> Option<RangeFill> {
        FacetStorage::range_fill(self, byte_start, byte_end)
    }

    fn shard_range_fill(&self, shard: usize, byte_start: u64, byte_end: u64) -> Option<RangeFill> {
        FacetStorage::shard_range_fill(self, shard, byte_start, byte_end)
    }

    fn read_prefix(&self, len: u64) -> std::io::Result<Vec<u8>> {
        FacetStorage::read_prefix(self, len)
    }

    fn map_records(&self, path: &str, start: u64, end: u64) -> Option<MappedRange> {
        record_range_to_bytes(path, start, end, self)
    }

    fn precache(&self) -> std::io::Result<()> {
        FacetStorage::precache(self)
    }

    fn prebuffer_with_progress(&self, cb: &mut dyn FnMut(&crate::DownloadProgress)) -> std::io::Result<()> {
        FacetStorage::prebuffer_with_progress(self, cb)
    }

    fn prebuffer_parts(&self, cb: &mut dyn FnMut(usize, &crate::DownloadProgress)) -> std::io::Result<()> {
        FacetStorage::prebuffer_parts(self, cb)
    }

    fn prebuffer_range_with_progress(
        &self,
        byte_start: u64,
        byte_end: u64,
        cb: &mut dyn FnMut(&crate::DownloadProgress),
    ) -> std::io::Result<()> {
        FacetStorage::prebuffer_range_with_progress(self, byte_start, byte_end, cb)
    }

    fn prebuffer_shard_range(
        &self,
        shard: usize,
        byte_start: u64,
        byte_end: u64,
        cb: &mut dyn FnMut(&crate::DownloadProgress),
    ) -> std::io::Result<()> {
        FacetStorage::prebuffer_shard_range(self, shard, byte_start, byte_end, cb)
    }

    fn revalidate(&self) -> std::io::Result<bool> {
        FacetStorage::revalidate(self)
    }

    fn content_id_with_progress(
        &self,
        progress: &mut dyn crate::ProgressSink,
    ) -> std::io::Result<Option<crate::ContentId>> {
        match &self.series {
            None => self.storage.content_id(&mut |done, total| progress.report("hashing", done, total)),
            Some(_) => Ok(None),
        }
    }

    fn known_content_id(&self) -> Option<crate::ContentId> {
        match &self.series {
            None => self.storage.known_content_id(),
            Some(_) => None,
        }
    }

    fn verify_cached(&self, progress: &mut dyn crate::ProgressSink) -> std::io::Result<Option<Vec<u32>>> {
        let mut hear = |done: u64, total: u64| progress.report("verifying cached chunks", done, total);
        match &self.series {
            None => self.storage.verify_cached(&mut hear),
            Some(series) => {
                let mut bad = Vec::new();
                let mut any = false;
                for i in 0..series.file_count() {
                    if let Some(b) = series.file(i).map_err(std::io::Error::other)?.verify_cached(&mut hear)? {
                        any = true;
                        bad.extend(b);
                    }
                }
                Ok(any.then_some(bad))
            }
        }
    }

    fn open_records(
        &self,
        facet: &str,
        namespace: Option<&str>,
    ) -> std::result::Result<crate::records::RecordFacet, crate::records::RecordError> {
        crate::records::open_record_facet(self, facet, namespace)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A facet spread across several files, read as one vector space.
///
/// Presents the **same surface** as a single-file reader (SH-22): a
/// caller that never asks about layout never learns there is one.
/// `count()` is the series total, `dim()` the dimension every shard
/// shares, and `get(o)` resolves through the ordinal model to the file
/// that holds `o`.
///
/// One reader per **file**, not per shard (SH-81) — two shards sliced
/// from one file read through one reader at their own offsets. `R` is
/// the single-file reader for the facet's layout:
/// [`XvecReader`](crate::io::XvecReader) for vectors,
/// [`ScalarReader`](crate::io::ScalarReader) for a column. Locating an
/// ordinal, reading across a seam and the shape check are written once
/// here for both.
pub(crate) struct ShardedReader<T, R> {
    series: std::sync::Arc<Series>,
    /// Per-file readers, opened on first touch (SH-26).
    readers: Vec<std::sync::OnceLock<R>>,
    /// How a file's storage becomes an `R`.
    open: fn(std::sync::Arc<crate::storage::Storage>) -> std::result::Result<R, crate::io::IoError>,
    dim: usize,
    count: usize,
    _t: std::marker::PhantomData<T>,
}

pub(crate) type ShardedXvecReader<T> = ShardedReader<T, crate::io::XvecReader<T>>;
pub(crate) type ShardedScalarReader<T> = ShardedReader<T, crate::io::ScalarReader<T>>;

impl<T, R> std::fmt::Debug for ShardedReader<T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedReader")
            .field("shards", &self.series.shards().entries().len())
            .field("files", &self.series.file_count())
            .field("dim", &self.dim)
            .field("count", &self.count)
            .finish()
    }
}

impl<T: Element> ShardedXvecReader<T> {
    fn open_xvec(series: std::sync::Arc<Series>) -> Result<Self> {
        Self::open(series, crate::io::xvec_from_storage::<T>)
    }
}

impl<T: Element> ShardedScalarReader<T> {
    fn open_scalar(series: std::sync::Arc<Series>) -> Result<Self> {
        Self::open(series, crate::io::scalar_from_storage::<T>)
    }
}

impl<T: Element, R: VectorReader<T>> ShardedReader<T, R> {
    /// Open every file's reader and check that the shards agree on
    /// one shape. A file whose dimension disagrees resolves ordinals to
    /// records of the wrong width, and there is no later moment at
    /// which that becomes visible (SH-15). Paying one header read per
    /// file at open is what makes every read after it trustworthy.
    fn open(
        series: std::sync::Arc<Series>,
        open: fn(std::sync::Arc<crate::storage::Storage>) -> std::result::Result<R, crate::io::IoError>,
    ) -> Result<Self> {
        let n = series.file_count();
        let readers: Vec<std::sync::OnceLock<R>> = (0..n).map(|_| std::sync::OnceLock::new()).collect();
        let me = Self {
            series,
            readers,
            open,
            dim: 0,
            count: 0,
            _t: std::marker::PhantomData,
        };
        let first = me.reader(0)?;
        let dim = first.dim();
        for i in 1..n {
            let d = me.reader(i)?.dim();
            if d != dim {
                return Err(Error::Other(format!(
                    "shard file {i} has dim {d}, but the series is dim {dim} — \
                     every shard of a facet must share one shape"
                )));
            }
        }
        Ok(Self {
            count: me.series.shards().count() as usize,
            dim,
            ..me
        })
    }

    fn reader(&self, i: usize) -> Result<&R> {
        let slot = self
            .readers
            .get(i)
            .ok_or_else(|| Error::Other(format!("shard file {i} out of range")))?;
        if let Some(r) = slot.get() {
            return Ok(r);
        }
        let storage = self.series.file(i)?.clone();
        let opened = (self.open)(storage)
            .map_err(|e| Error::Other(format!("open shard file {i}: {e}")))?;
        let _ = slot.set(opened);
        slot.get()
            .ok_or_else(|| Error::Other("shard reader vanished after set".into()))
    }

    fn locate(&self, index: usize) -> Result<(&R, usize)> {
        let at = self
            .series
            .shards()
            .locate(index as u64)
            .ok_or_else(|| Error::Other(format!("ordinal {index} out of range")))?;
        let file = self.series.file_index_of_shard(at.shard)?;
        Ok((self.reader(file)?, at.file_ordinal as usize))
    }
}

impl<T: Element, R: VectorReader<T>> VectorReader<T> for ShardedReader<T, R> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn count(&self) -> usize {
        self.count
    }
    fn native_element_type(&self) -> Option<crate::typed_access::ElementType> {
        Some(T::TYPE)
    }

    fn get(&self, index: usize) -> std::result::Result<Vec<T>, crate::io::IoError> {
        let (reader, at) = self
            .locate(index)
            .map_err(|_| crate::io::IoError::OutOfBounds(index))?;
        crate::io::VectorReader::get(reader, at)
    }

    /// A borrowed slice, when the record lies in a resident,
    /// mmap-promoted file.
    ///
    /// Never synthesized across files (SH-25): a record spans no shard
    /// boundary and a shard spans no file, so the delegation is exact —
    /// but a caller asking for a slice of a file that is not mapped gets
    /// an honest `None` rather than a copy pretending to be a borrow.
    fn get_slice(&self, index: usize) -> Option<&[T]> {
        let (reader, at) = self.locate(index).ok()?;
        // The trait's checked form: the inherent `get_slice` on a mapped
        // reader is the unchecked hot-path one.
        VectorReader::get_slice(reader, at)
    }

    /// One bulk read per shard the span touches.
    fn read_into(&self, start: usize, out: &mut [T]) -> std::result::Result<usize, crate::io::IoError>
    where
        T: Copy,
    {
        let n = crate::io::records_to_read(start, out.len(), self.dim, self.count)?;
        let mut done = 0;
        while done < n {
            let at = self
                .series
                .shards()
                .locate((start + done) as u64)
                .ok_or(crate::io::IoError::OutOfBounds(start + done))?;
            let in_shard = (self.series.shards().entries()[at.shard].len - at.local) as usize;
            let run = in_shard.min(n - done);
            let file = self
                .series
                .file_index_of_shard(at.shard)
                .map_err(|e| crate::io::IoError::InvalidFormat(e.to_string()))?;
            let reader = self
                .reader(file)
                .map_err(|e| crate::io::IoError::InvalidFormat(e.to_string()))?;
            let span = &mut out[done * self.dim..(done + run) * self.dim];
            let got = crate::io::VectorReader::read_into(reader, at.file_ordinal as usize, span)?;
            if got != run {
                return Err(crate::io::IoError::InvalidFormat(format!(
                    "shard {} holds {got} of the {run} records its declaration promises",
                    at.shard
                )));
            }
            done += run;
        }
        Ok(n)
    }

    fn precache(&self) -> std::io::Result<()> {
        for i in 0..self.series.file_count() {
            self.series
                .file(i)
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .precache()?;
        }
        Ok(())
    }

    fn is_complete(&self) -> bool {
        (0..self.series.file_count()).all(|i| self.series.file(i).is_ok_and(|f| f.is_complete()))
    }
}

/// A variable-length reader over a facet spread across several files.
///
/// Each **file** carries its own `IDXFOR__` sidecar, whose offsets are
/// local to that file (SH-17, SH-82). A sliced shard reads the
/// sub-range its window names; no index is rebuilt or re-based to
/// support slicing, and two shards drawn from one file load its index
/// once between them.
pub(crate) struct ShardedVvecReader<T> {
    series: std::sync::Arc<Series>,
    readers: Vec<std::sync::OnceLock<crate::io::IndexedVvecReader<T>>>,
    /// The source string per file, for the sidecar lookup.
    sources: Vec<String>,
    elem_size: usize,
    count: usize,
}

impl<T> std::fmt::Debug for ShardedVvecReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedVvecReader")
            .field("files", &self.series.file_count())
            .field("count", &self.count)
            .finish()
    }
}

impl<T: Element> ShardedVvecReader<T> {
    fn open(series: std::sync::Arc<Series>, elem_size: usize) -> Result<Self> {
        // One source per distinct file, in file order — the sidecar is
        // found beside the file, so the shard's own path is what names
        // it.
        let mut sources = vec![String::new(); series.file_count()];
        for (shard, entry) in series.shards().entries().iter().enumerate() {
            let i = series.file_index_of_shard(shard)?;
            if sources[i].is_empty() {
                sources[i] = entry.source.path.clone();
            }
        }
        let readers = (0..series.file_count())
            .map(|_| std::sync::OnceLock::new())
            .collect();
        let count = series.shards().count() as usize;
        Ok(Self {
            series,
            readers,
            sources,
            elem_size,
            count,
        })
    }

    fn reader(&self, i: usize) -> Result<&crate::io::IndexedVvecReader<T>> {
        let slot = self
            .readers
            .get(i)
            .ok_or_else(|| Error::Other(format!("shard file {i} out of range")))?;
        if let Some(r) = slot.get() {
            return Ok(r);
        }
        let storage = self.series.file(i)?.clone();
        let opened = crate::io::vvec_from_storage::<T>(
            storage,
            &self.sources[i],
            self.elem_size,
        )
        .map_err(|e| Error::Other(format!("open shard file {i}: {e}")))?;
        let _ = slot.set(opened);
        slot.get()
            .ok_or_else(|| Error::Other("shard reader vanished after set".into()))
    }

    fn locate(&self, index: usize) -> Result<(&crate::io::IndexedVvecReader<T>, usize)> {
        let at = self
            .series
            .shards()
            .locate(index as u64)
            .ok_or_else(|| Error::Other(format!("ordinal {index} out of range")))?;
        let file = self.series.file_index_of_shard(at.shard)?;
        Ok((self.reader(file)?, at.file_ordinal as usize))
    }
}

impl<T: Element> VvecReader<T> for ShardedVvecReader<T> {
    fn count(&self) -> usize {
        self.count
    }

    fn dim_at(&self, index: usize) -> std::result::Result<usize, crate::io::IoError> {
        let (r, at) = self
            .locate(index)
            .map_err(|_| crate::io::IoError::OutOfBounds(index))?;
        r.dim_at(at)
    }

    fn get_bytes(&self, index: usize) -> std::result::Result<Vec<u8>, crate::io::IoError> {
        let (r, at) = self
            .locate(index)
            .map_err(|_| crate::io::IoError::OutOfBounds(index))?;
        r.get_bytes(at)
    }

    fn precache(&self) -> std::io::Result<()> {
        for i in 0..self.series.file_count() {
            self.series
                .file(i)
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .precache()?;
        }
        Ok(())
    }

    fn is_complete(&self) -> bool {
        (0..self.series.file_count()).all(|i| self.series.file(i).is_ok_and(|f| f.is_complete()))
    }
}

/// A generic implementation of `TestDataView`.
///
/// This struct holds the configuration for a profile and the data source location,
/// creating the appropriate `VectorReader` (Mmap or Http) on demand.
#[derive(Debug)]
pub struct GenericTestDataView {
    /// Where the dataset lives; relative facet paths resolve against it.
    source: Location,
    config: ProfileConfig,
    /// Dataset-level attributes for metadata accessors.
    attributes: HashMap<String, serde_yaml::Value>,
    /// Dataset name (matches the `name:` field in `dataset.yaml`,
    /// or the catalog entry name for knn_entries-shape catalogs).
    /// When set, [`Self::open_facet_storage`] routes every facet
    /// through [`crate::storage::Storage::open_layered`] so cached
    /// bytes land at `<cache_root>/<dataset_name>/<facet_relpath>`
    /// — stable across catalog moves. `None` falls back to URL-
    /// derived layout (sufficient for direct URL opens that have
    /// no catalog context to anchor on).
    dataset_name: Option<String>,
    /// Catalog source URL recorded in the per-dataset `origin.json`.
    /// Verified on subsequent opens; editable by the user when the
    /// catalog moves but the bytes are the same.
    catalog_source: Option<String>,
    /// The cache root the dataset's catalog names for it, in place of
    /// the configured cache.
    cache_dir: Option<std::path::PathBuf>,
    /// What overlays claim about this dataset's counts, checked as the
    /// facets they concern open (OV-5).
    overlay_claims: Option<Arc<crate::overlay::Claims>>,
    /// Where each attribute that did not come from the dataset itself
    /// came from: an overlay, or loose files' names.
    attribute_sources: Arc<HashMap<String, crate::overlay::AttributeSource>>,
}

impl GenericTestDataView {
    /// Creates a new `GenericTestDataView`.
    pub fn new(source: Location, config: ProfileConfig) -> Self {
        Self {
            source,
            config,
            attributes: HashMap::new(),
            dataset_name: None,
            catalog_source: None,
            cache_dir: None,
            overlay_claims: None,
            attribute_sources: Arc::default(),
        }
    }

    /// Creates a new `GenericTestDataView` with dataset attributes.
    pub fn with_attributes(
        source: Location,
        config: ProfileConfig,
        attributes: HashMap<String, serde_yaml::Value>,
    ) -> Self {
        Self {
            source,
            config,
            attributes,
            dataset_name: None,
            catalog_source: None,
            cache_dir: None,
            overlay_claims: None,
            attribute_sources: Arc::default(),
        }
    }

    /// Keep the dataset's cached copies under `dir`, the cache its
    /// catalog names, in place of the configured cache.
    pub(crate) fn with_cache_dir(mut self, dir: Option<std::path::PathBuf>) -> Self {
        self.cache_dir = dir;
        self
    }

    /// Check what overlays claim about the dataset's counts against
    /// each facet they concern as it opens, warning on a disagreement
    /// (OV-5).
    pub(crate) fn with_overlay_claims(mut self, claims: Option<Arc<crate::overlay::Claims>>) -> Self {
        self.overlay_claims = claims;
        self
    }

    /// Where the attributes that did not come from the dataset itself
    /// came from.
    pub(crate) fn with_attribute_sources(mut self, sources: HashMap<String, crate::overlay::AttributeSource>) -> Self {
        self.attribute_sources = Arc::new(sources);
        self
    }

    /// Warn about every overlay claim the facet `name`, just opened as
    /// `count` records of `dim`, contradicts. The base's count is the
    /// claim's only for the whole base: a sized profile's window over
    /// it holds fewer records by design.
    fn probe_overlay(&self, facet: &FacetConfig, name: &str, count: usize, dim: usize) {
        let Some(claims) = &self.overlay_claims else { return };
        let windowed = facet.window().is_some() || facet.source().is_some_and(|s| s.contains('['));
        let whole_base = !windowed;
        for (key, claimed, observed, source) in claims.contradictions(name, count, dim, whole_base) {
            crate::overlay::warn_contradiction(
                &claims.dataset,
                key,
                &claimed.to_string(),
                &format!("{name} holding {observed}"),
                &source,
            );
        }
    }

    /// Set the catalog-anchored cache identity. Both must be set
    /// together — `dataset_name` is the per-dataset cache
    /// directory name, `catalog_source` is the URL recorded in
    /// `origin.json` (typically the dataset.yaml URL, or the
    /// knn_entries.yaml URL for that catalog shape).
    pub fn with_catalog_identity(
        mut self,
        dataset_name: impl Into<String>,
        catalog_source: impl Into<String>,
    ) -> Self {
        self.dataset_name = Some(dataset_name.into());
        self.catalog_source = Some(catalog_source.into());
        self
    }

    /// Open a uniform vector facet with the unified `open_vec` API.
    /// Handles local (mmap) and remote (HTTP) transparently.
    ///
    /// When the facet config carries a `window` field
    /// (`profiles.X.base_vectors.window: "0..N"`), the returned
    /// reader is wrapped so that `count()` reports `N - 0` and
    /// `get(i)` is offset into the underlying file. This is the
    /// documented sub-ordinal suffix model — sized profiles inherit
    /// `base_vectors` from default with `[0..base_count)` so every
    /// consumer reading via the trait gets a clipped reader without
    /// having to honor `view.base_count()` manually.
    /// Open an element facet as `T`: at the file's own type when `T`
    /// is it, converted where [`ElementType::conversion_to`] allows,
    /// refused otherwise. The one place in the crate a conversion is
    /// accepted; the raw readers never convert.
    ///
    /// [`ElementType::conversion_to`]: crate::typed_access::ElementType::conversion_to
    fn open_uniform<T: Element>(
        &self,
        facet_opt: Option<&FacetConfig>,
        name: &str,
    ) -> Result<Arc<dyn VectorReader<T>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        // `VectorReader` promises `dim()` and a run of `T` per record.
        // A record-shaped facet has neither, so it is refused here with
        // the reader that does open it — rather than failing further in
        // with "cannot infer element size", which describes the symptom
        // and points nowhere.
        self.require_facet_shape(name, crate::dataset::facet::FacetShape::Elements)?;
        let native = self.facet_element_type(name)?;
        let any = self.open_any_native(facet, name, native)?;
        Ok(any.as_type::<T>()?)
    }

    /// Open an element facet at the file's own type, erased, so a
    /// runtime [`ElementType`](crate::typed_access::ElementType)
    /// reaches the monomorphic reader.
    fn open_any_native(
        &self,
        facet: &FacetConfig,
        name: &str,
        native: crate::typed_access::ElementType,
    ) -> Result<crate::io::AnyVectorReader> {
        veks_datatypes::dispatch_element!(native, N => {
            let reader = self.open_native::<N>(facet, name)?;
            self.probe_overlay(facet, name, reader.count(), reader.dim());
            Ok(<N as crate::io::AnyElement>::into_any(reader))
        })
    }

    /// Open an element facet at its own type `N`, with the window and
    /// series logic written once for vectors and for columns.
    fn open_native<N: Element>(
        &self,
        facet: &FacetConfig,
        name: &str,
    ) -> Result<Arc<dyn VectorReader<N>>> {

        // A series reads through the shard model. The profile window
        // still applies on top, in *facet* ordinals (SH-67) — an entry
        // window inside the series is in file ordinals and has already
        // been folded into the shards.
        if facet.is_series() {
            let storage = self.facet_storage(name)?;
            let series = storage
                .series_ref()
                .ok_or_else(|| Error::Other(format!("facet '{name}': series not realized")))?
                .clone();
            let column = series.shards().entries().first().is_some_and(|e| {
                crate::io::is_scalar_ext(e.source.path.rsplit('.').next().unwrap_or(""))
            });
            let reader: Box<dyn VectorReader<N>> = if column {
                Box::new(ShardedScalarReader::<N>::open_scalar(series)?)
            } else {
                Box::new(ShardedXvecReader::<N>::open_xvec(series)?)
            };
            return Ok(match facet.window().and_then(parse_window_first) {
                Some((start, end)) => Arc::new(WindowedVectorReader::new(reader, start, end)),
                None => Arc::from(reader),
            });
        }

        // Two ways the dataset config can carry a sub-ordinal range:
        //   1. `Detailed { source, window }` — explicit `window:` field.
        //   2. `Simple("path[0..N)")` — the documented suffix sugar
        //      that `parse_source_string` understands.
        // Try the suffix form first (so `Simple` strings work without
        // forcing the consumer to use `Detailed`); fall back to the
        // explicit field. Either path produces the same windowed
        // reader behind the trait.
        let raw = single_source(facet, "open")?;
        // A source string only fails to parse when it *looks* like it
        // carries a `[..]` window suffix and that suffix is malformed —
        // a plain path always parses with an empty window. So an error
        // here is a broken window, not a broken path, and treating the
        // whole string as a filename turns "your window is malformed"
        // into "no such file: base.fvec[0,1000)". Surface the real
        // reason instead.
        let (path_str, window_from_suffix): (String, Option<(usize, usize)>) =
            match crate::dataset::source::parse_source_string(raw) {
                Ok(parsed) if !parsed.window.is_empty() => {
                    let iv = &parsed.window.0[0];
                    (
                        parsed.path,
                        Some((iv.min_incl as usize, iv.max_excl as usize)),
                    )
                }
                Ok(parsed) => (parsed.path, None),
                Err(e) => {
                    return Err(Error::Other(format!(
                        "facet '{name}': source '{raw}' has a malformed window: {e}"
                    )));
                }
            };
        // The reader is built on the facet's own storage — the one open
        // that anchors the cache by dataset — so it reads the bytes a
        // fetch brought in. Opening the resolved URL directly would key a
        // second cache entry by URL: everything downloaded twice, and
        // every read a range request against the empty one.
        let (_, container) = io::require_native::<N>(&path_str)?;
        let storage = self.facet_storage(name)?;
        let reader: Box<dyn VectorReader<N>> = match container {
            crate::typed_access::Container::Scalar => {
                Box::new(io::scalar_from_storage::<N>(storage.storage_handle())?)
            }
            _ => Box::new(io::xvec_from_storage::<N>(storage.storage_handle())?),
        };

        let window = window_from_suffix.or_else(|| facet.window().and_then(parse_window_first));
        if let Some((start, end)) = window {
            return Ok(Arc::new(WindowedVectorReader::new(reader, start, end)));
        }
        Ok(Arc::from(reader))
    }

    /// Resolve a bare path string (no window suffix) against the
    /// dataset's location, by [`Location::resolve`]: a URL or `file://`
    /// URI stands alone, a leading `/` is a local path (a remote
    /// dataset may point a facet at a local copy), and anything else
    /// is relative to the dataset. The picker, `datasets list` and the
    /// cache drop resolve through the same join, so they name the file
    /// this reads.
    fn resolve_path_str(&self, path: &str) -> Result<String> {
        self.source.resolve(path)
    }

    /// Open a variable-length vector facet with the unified `open_vvec` API.
    /// Handles local (mmap + index) and remote (HTTP + index) transparently.
    fn open_variable<T: Element>(
        &self,
        facet_opt: Option<&FacetConfig>,
        name: &str,
    ) -> Result<Arc<dyn VvecReader<T>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        if facet.is_series() {
            let native = self.facet_element_type(name)?;
            if native != T::TYPE {
                return Err(io::IoError::ElementMismatch {
                    file: native,
                    requested: T::TYPE,
                    reason: "a variable-length facet is read at its own type only",
                }
                .into());
            }
            let storage = self.facet_storage(name)?;
            let series = storage
                .series_ref()
                .ok_or_else(|| Error::Other(format!("facet '{name}': series not realized")))?
                .clone();
            return Ok(Arc::new(ShardedVvecReader::<T>::open(
                series,
                T::SIZE,
            )?));
        }
        // On the facet's own storage, as `open_uniform` is, so the cache
        // is the one a fetch filled. The source still names where the
        // offset index's sidecar lives.
        let path_or_url = self.resolve_as_string(facet)?;
        io::IndexedVvecReader::<T>::validate_source(&path_or_url)?;
        let storage = self.facet_storage(name)?;
        Ok(Arc::new(io::vvec_from_storage::<T>(
            storage.storage_handle(),
            &path_or_url,
            T::SIZE,
        )?))
    }

    /// Resolve a facet to a path string (local) or URL string (remote).
    fn resolve_as_string(&self, facet: &FacetConfig) -> Result<String> {
        self.resolve_path_str(single_source(facet, "resolve")?)
    }

    /// Collect all facets declared in the profile config.
    fn collect_facets(&self) -> HashMap<String, FacetDescriptor> {
        let mut manifest = HashMap::new();

        // Every facet this view can open, whether or not the spec
        // names it. `base_content`, `query_terms` and `query_filters`
        // are declarable in `dataset.yaml` and openable by name, so a
        // manifest that omitted them made them invisible to everything
        // that asks what a profile holds — precache among them.
        let standard_facets: &[(&str, Option<&FacetConfig>, Option<StandardFacet>)] = &[
            (
                "base_vectors",
                self.config.base_vectors.as_ref(),
                Some(StandardFacet::BaseVectors),
            ),
            (
                "query_vectors",
                self.config.query_vectors.as_ref(),
                Some(StandardFacet::QueryVectors),
            ),
            (
                "neighbor_indices",
                self.config.neighbor_indices.as_ref(),
                Some(StandardFacet::NeighborIndices),
            ),
            (
                "neighbor_distances",
                self.config.neighbor_distances.as_ref(),
                Some(StandardFacet::NeighborDistances),
            ),
            (
                "metadata_content",
                self.config.metadata_content.as_ref(),
                Some(StandardFacet::MetadataContent),
            ),
            (
                "metadata_predicates",
                self.config.metadata_predicates.as_ref(),
                Some(StandardFacet::MetadataPredicates),
            ),
            (
                "metadata_results",
                self.config.predicate_results.as_ref(),
                Some(StandardFacet::MetadataResults),
            ),
            (
                "metadata_layout",
                self.config.metadata_layout.as_ref(),
                Some(StandardFacet::MetadataLayout),
            ),
            (
                "prefiltered_neighbor_indices",
                self.config.prefiltered_neighbor_indices.as_ref(),
                Some(StandardFacet::PrefilteredNeighborIndices),
            ),
            (
                "prefiltered_neighbor_distances",
                self.config.prefiltered_neighbor_distances.as_ref(),
                Some(StandardFacet::PrefilteredNeighborDistances),
            ),
            (
                "postfiltered_neighbor_indices",
                self.config.postfiltered_neighbor_indices.as_ref(),
                Some(StandardFacet::PostfilteredNeighborIndices),
            ),
            (
                "postfiltered_neighbor_distances",
                self.config.postfiltered_neighbor_distances.as_ref(),
                Some(StandardFacet::PostfilteredNeighborDistances),
            ),
            ("quant_scale", self.config.quant_scale.as_ref(), Some(StandardFacet::QuantScale)),
            (
                "quant_zero_point",
                self.config.quant_zero_point.as_ref(),
                Some(StandardFacet::QuantZeroPoint),
            ),
            ("quant_sum", self.config.quant_sum.as_ref(), Some(StandardFacet::QuantSum)),
            ("base_content", self.config.base_content.as_ref(), None),
            ("query_terms", self.config.query_terms.as_ref(), None),
            ("query_filters", self.config.query_filters.as_ref(), None),
        ];

        for (name, facet_opt, kind) in standard_facets {
            if let Some(facet) = facet_opt {
                // A series has no single source path. Reporting `None`
                // makes a windowed plan degrade to a whole-facet fetch,
                // which the fallback gate then refuses — the safe
                // transitional answer until the planner consumes the
                // realized `Shards` model. Reporting the first shard
                // instead would silently map the facet against a
                // fraction of itself.
                let source = facet.source().map(str::to_string);
                // The **format** is answerable for a series even though
                // the path is not: every shard shares one format (SRD
                // invariant 4), so the first declared source names it.
                // Leaving this `None` for a series made a sharded facet
                // look formatless to anything that asks what it is.
                let source_type = facet
                    .sources()
                    .first()
                    .map(|s| s.as_str())
                    .and_then(FacetDescriptor::infer_type);
                manifest.insert(
                    name.to_string(),
                    FacetDescriptor {
                        name: name.to_string(),
                        source_type,
                        source_path: source,
                        window: facet.window().map(|w| w.to_string()),
                        standard_kind: *kind,
                        shard_count: facet.shard_count(),
                    },
                );
            }
        }

        manifest
    }
}

impl GenericTestDataView {
    /// Look up a facet by name and return its FacetConfig.
    fn facet_config_by_name(&self, name: &str) -> Option<&FacetConfig> {
        match name {
            "base_vectors" => self.config.base_vectors.as_ref(),
            "query_vectors" => self.config.query_vectors.as_ref(),
            // Declared in `ProfileConfig` and therefore declarable in
            // `dataset.yaml`. Without an arm here a dataset can name
            // one and nothing can open it: `facet()` reports it
            // missing, and `derive` — which asks this route for an
            // element type — treats it as a non-data key and drops it
            // from the output.
            "base_content" => self.config.base_content.as_ref(),
            "query_terms" => self.config.query_terms.as_ref(),
            "query_filters" => self.config.query_filters.as_ref(),
            "neighbor_indices" => self.config.neighbor_indices.as_ref(),
            "neighbor_distances" => self.config.neighbor_distances.as_ref(),
            "metadata_content" => self.config.metadata_content.as_ref(),
            "metadata_predicates" => self.config.metadata_predicates.as_ref(),
            // Canonical spec key first (`StandardFacet::basenames`), then
            // the legacy names — the same set the config field accepts
            // as serde aliases. Lookup must know every spelling the
            // reader methods pass, or a facet resolves by one route and
            // not another.
            "metadata_results" | "metadata_indices" | "predicate_results" => {
                self.config.predicate_results.as_ref()
            }
            "metadata_layout" => self.config.metadata_layout.as_ref(),
            // Canonical pre-filter keys + legacy `filtered_*` aliases
            // (legacy on-disk files carry pre-filter shape).
            "prefiltered_neighbor_indices" | "filtered_neighbor_indices" => {
                self.config.prefiltered_neighbor_indices.as_ref()
            }
            "prefiltered_neighbor_distances" | "filtered_neighbor_distances" => {
                self.config.prefiltered_neighbor_distances.as_ref()
            }
            // Canonical post-filter keys.
            "postfiltered_neighbor_indices" => self.config.postfiltered_neighbor_indices.as_ref(),
            "postfiltered_neighbor_distances" => {
                self.config.postfiltered_neighbor_distances.as_ref()
            }
            "quant_scale" => self.config.quant_scale.as_ref(),
            "quant_zero_point" => self.config.quant_zero_point.as_ref(),
            "quant_sum" => self.config.quant_sum.as_ref(),
            _ => None,
        }
    }
}


/// Cache-relative path for a facet inside its dataset's cache
/// directory (`<cache_root>/<dataset>/<relpath>`): the home-relative
/// path when the facet lives under the dataset's home URL, otherwise
/// the URL basename — flat under the dataset directory, per the
/// mandated dataset-keyed layout. The result is `/`-separated.
///
/// A local facet resolves through `Path::join`, which on Windows
/// yields mixed separators (`C:\ds\a/b.fvec`) while the home is
/// written `C:\ds/`; both sides are compared with `/` separators, or
/// the prefix never matches and every local facet falls back to its
/// basename — colliding with any other facet that shares one.
pub(crate) fn facet_cache_relpath(resolved: &str, home_norm: &str) -> String {
    let resolved = forward_slashes(resolved);
    let home_norm = forward_slashes(home_norm);
    if let Some(rel) = resolved.strip_prefix(home_norm.as_ref()) {
        return rel.to_string();
    }
    resolved.rsplit('/').next().unwrap_or(&resolved).to_string()
}

/// `s` with Windows path separators rewritten to `/`. A no-op on
/// other platforms, where `\` is an ordinary filename character.
fn forward_slashes(s: &str) -> std::borrow::Cow<'_, str> {
    if cfg!(windows) && s.contains('\\') {
        std::borrow::Cow::Owned(s.replace('\\', "/"))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

impl GenericTestDataView {
    /// Hard-stop guard for cache-file collisions: two facets of this
    /// profile must never map to the same `<dataset>/<relpath>` from
    /// different source URLs — chunk sidecars are keyed to the file,
    /// so a shared path with divergent bytes is silent corruption.
    /// Realistically reachable only via basename collisions between
    /// out-of-home facets; cheap to check (≤ a dozen facets).
    fn reject_relpath_collisions(
        &self,
        facet_name: &str,
        resolved: &str,
        relpath: &str,
        home_norm: &str,
    ) -> Result<()> {
        for other_name in self.facet_manifest().keys() {
            if other_name == facet_name {
                continue;
            }
            // Every file the other facet names, not its single source.
            // A series has no single source, so reading one would skip
            // series facets entirely — and a shard is exactly the kind
            // of file that collides, since shard names are derived from
            // a basename many facets could share (SH-33).
            let Some(other) = self.facet_config_by_name(other_name) else {
                continue;
            };
            for raw in other.declared_files() {
                let other_path = match crate::dataset::source::parse_source_string(&raw) {
                    Ok(parsed) => parsed.path,
                    Err(_) => raw.clone(),
                };
                let Ok(other_resolved) = self.resolve_path_str(&other_path) else {
                    continue;
                };
                if other_resolved == resolved {
                    continue;
                }
                if facet_cache_relpath(&other_resolved, home_norm) == relpath {
                    return Err(Error::Other(format!(
                        "cache filename collision in dataset '{}': facets \
                         '{facet_name}' ({resolved}) and '{other_name}' \
                         ({other_resolved}) both cache as '{relpath}'. Rename \
                         one of the source files (or move it under the \
                         dataset's home URL) so each facet has a distinct \
                         filename.",
                        self.dataset_name.as_deref().unwrap_or("?"),
                    )));
                }
            }
        }
        Ok(())
    }
}

impl TestDataView for GenericTestDataView {
    fn metadata_content(&self) -> Option<&FacetConfig> {
        self.config.metadata_content.as_ref()
    }

    fn metadata_predicates(&self) -> Option<&FacetConfig> {
        self.config.metadata_predicates.as_ref()
    }

    fn predicate_results(&self) -> Option<&FacetConfig> {
        self.config.predicate_results.as_ref()
    }

    fn metadata_layout(&self) -> Option<&FacetConfig> {
        self.config.metadata_layout.as_ref()
    }

    fn facet_manifest(&self) -> HashMap<String, FacetDescriptor> {
        self.collect_facets()
    }

    fn open_facet_as_type(
        &self,
        name: &str,
        target: crate::typed_access::ElementType,
    ) -> Result<crate::io::AnyVectorReader> {
        let name = match name {
            "filtered_neighbor_indices" => "prefiltered_neighbor_indices",
            "filtered_neighbor_distances" => "prefiltered_neighbor_distances",
            other => other,
        };
        veks_datatypes::dispatch_element!(target, T => {
            let reader = self.open_uniform::<T>(self.facet_config_by_name(name), name)?;
            Ok(<T as crate::io::AnyElement>::into_any(reader))
        })
    }

    fn open_vvec_facet_native(&self, name: &str) -> Result<crate::io::AnyVvecReader> {
        let (config, name) = match name {
            "metadata_results" | "metadata_indices" | "predicate_results" => {
                (self.config.predicate_results.as_ref(), "metadata_results")
            }
            other => (self.facet_config_by_name(other), other),
        };
        let native = self.facet_element_type(name)?;
        veks_datatypes::dispatch_element!(native, T => {
            let reader = self.open_variable::<T>(config, name)?;
            Ok(<T as crate::io::AnyElement>::into_any_vvec(reader))
        })
    }

    fn facet_element_type(&self, name: &str) -> Result<crate::typed_access::ElementType> {
        // A record-shaped facet has no element width, and saying so as
        // "unknown element type" reads like a gap in this build rather
        // than a property of the data. Name the shape and the reader
        // that opens it.
        self.require_facet_shape(name, crate::dataset::facet::FacetShape::Elements)?;
        let facet = self
            .facet_config_by_name(name)
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        // Strip the `[start..end)` window suffix before inspecting
        // the extension — otherwise "base.fvec[0..1000)" splits at the
        // dots inside the window and yields "1000)" as the "extension",
        // which infers to no element type at all. The whole facet then
        // looks unrecognised to the precache iterator and gets silently
        // skipped — exactly the symptom that started this fix.
        // Any entry answers this: every shard of a series shares one
        // format and element type (SRD invariant 4), so the first is not
        // a stand-in for the facet — it is the same answer the others
        // would give.
        let raw = facet
            .sources()
            .first()
            .ok_or_else(|| Error::Other(format!("facet '{name}' declares no source")))?;
        let source = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        crate::typed_access::ElementType::from_extension(source.rsplit('.').next().unwrap_or(""))
            .ok_or_else(|| Error::Other(format!("unknown element type for facet '{name}'")))
    }

    fn base_count(&self) -> Option<u64> {
        self.config.base_count
    }

    fn dataset_attribute(&self, key: &str) -> Option<serde_yaml::Value> {
        self.attributes.get(key).cloned()
    }

    fn dataset_attribute_source(&self, key: &str) -> Option<crate::overlay::AttributeSource> {
        match self.attribute_sources.get(key) {
            Some(source) => Some(source.clone()),
            None => self.attributes.contains_key(key).then_some(crate::overlay::AttributeSource::Dataset),
        }
    }

    fn cache_name(&self) -> Option<String> {
        self.dataset_name.clone()
    }

    fn cache_dir(&self) -> Option<std::path::PathBuf> {
        self.cache_dir.clone()
    }

    fn distance_function(&self) -> Option<String> {
        self.attributes
            .get("distance_function")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn quantization(&self) -> Option<std::result::Result<crate::dataset::Quantization, String>> {
        self.config
            .attributes
            .get("quantization")
            .or_else(|| self.attributes.get("quantization"))
            .map(crate::dataset::Quantization::from_value)
    }

    fn facet_source(&self, name: &str) -> Option<String> {
        let facet = self.facet_config_by_name(name)?;
        // `None` for a series: it has no single path, and answering with
        // the first shard would resolve the facet to a fraction of
        // itself (SH-74).
        let raw = facet.source()?;
        let path_str = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        self.resolve_path_str(&path_str).ok()
    }

    fn open_facet_storage(&self, name: &str) -> Result<Box<dyn FacetBytes>> {
        Ok(Box::new(self.facet_storage(name)?))
    }

    fn cache_capacity(&self) -> crate::fetch::CapacityCheck {
        let dir = self.cache_dir.clone();
        crate::fetch::CapacityCheck::new(move |bytes| Ok(crate::cache::ensure_cache_capacity_in(dir.as_deref(), bytes)?))
    }

    fn normalization(&self) -> Result<crate::verify::Normalization> {
        crate::verify::normalization(self)
    }

    fn verify_report(
        &self,
        options: &crate::verify::VerifyOptions,
        progress: &mut dyn crate::ProgressSink,
    ) -> crate::verify::VerifyReport {
        crate::verify::verify_view(self, options, progress)
    }
}

impl GenericTestDataView {
    /// The facet's bytes as vectordata's own handle: what the trait's
    /// [`open_facet_storage`](TestDataView::open_facet_storage) boxes,
    /// and what the readers here are built from.
    pub fn facet_storage(&self, name: &str) -> Result<FacetStorage> {
        let facet = self
            .facet_config_by_name(name)
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        // A series resolves every shard's file; a single facet resolves
        // one. Both go through the same per-file resolution below, so a
        // shard is anchored, cached, and collision-checked exactly as a
        // lone facet is (SH-32).
        if facet.is_series() {
            return self.open_facet_series(name, facet);
        }
        // Strip any window suffix from the source so Storage opens
        // the underlying file/url, not a half-parsed token.
        let raw = single_source(facet, "open storage for")?;
        let path_str = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        let resolved = self.resolve_path_str(&path_str)?;
        // Catalog-anchored open. The cache layout is fixed by design:
        //
        //   <cache_root>/<dataset_name>/<dataset files>
        //
        // For facets under the dataset's home URL (`catalog_source`)
        // the file keeps its home-relative path ("base.fvec",
        // "profiles/1m/base.fvec") so the cache mirrors the published
        // layout. A facet living OUTSIDE the home URL (legal in
        // knn_entries catalogs — several sized datasets sharing one
        // remote directory of files) still caches under THIS
        // dataset's directory, flat, by URL basename. Datasets that
        // share a remote file each hold their own copy — the layout
        // is dataset-keyed, never URL-keyed. (The previous fallback
        // used the absolute URL as a "relative path", producing
        // `<dataset>/s3:/...` trees whose colon broke Windows.)
        //
        // `facet_cache_relpath` collisions (two facets of this
        // profile mapping to one cache file from different URLs) are
        // rejected before any byte lands — see the guard below.
        let storage = match (&self.dataset_name, &self.catalog_source) {
            (Some(ds_name), Some(home)) => {
                let home_norm = crate::location::dir_form(home);
                let file_relpath = facet_cache_relpath(&resolved, &home_norm);
                self.reject_relpath_collisions(name, &resolved, &file_relpath, &home_norm)?;
                crate::storage::Storage::open_layered(&resolved, ds_name, &file_relpath, home, self.cache_dir.as_deref())
            }
            _ => crate::storage::Storage::open(&resolved),
        }
        .map_err(|e| Error::Other(format!("storage open '{name}': {e}")))?;
        Ok(FacetStorage::new(storage))
    }
}

impl GenericTestDataView {
    /// How to open one resolved path, using this view's catalog
    /// anchoring. Shared by the single-file and series paths so a shard
    /// caches exactly where a lone facet would.
    fn shard_open_spec(&self, facet: &str, path_str: &str) -> Result<ShardOpen> {
        let resolved = self.resolve_path_str(path_str)?;
        Ok(match (&self.dataset_name, &self.catalog_source) {
            (Some(ds_name), Some(home)) => {
                let home_norm = crate::location::dir_form(home);
                let relpath = facet_cache_relpath(&resolved, &home_norm);
                self.reject_relpath_collisions(facet, &resolved, &relpath, &home_norm)?;
                ShardOpen::Layered {
                    resolved,
                    dataset: ds_name.clone(),
                    relpath,
                    home: home.clone(),
                    cache_dir: self.cache_dir.clone(),
                }
            }
            _ => ShardOpen::Plain { resolved },
        })
    }

    /// Open a facet declared as a multi-file series.
    ///
    /// The declaration is realized into the ordinal model first, so a
    /// series that disagrees with itself fails here rather than at the
    /// first read. Only the first shard's file is opened; the rest wait
    /// until something reads them (SH-26).
    fn open_facet_series(&self, name: &str, facet: &FacetConfig) -> Result<FacetStorage> {
        // Bare local entries need a record count from the file itself,
        // which means resolving the path and opening it — the local
        // convenience SH-63 permits.
        let probe = |src: &crate::dataset::source::DSSource| -> std::result::Result<u64, String> {
            let spec = self
                .shard_open_spec(name, &src.path)
                .map_err(|e| e.to_string())?;
            let storage = spec.open().map_err(|e| e.to_string())?;
            records_in(&src.path, &storage)
                .ok_or_else(|| format!("cannot derive a record count from '{}'", src.path))
        };
        let shards = crate::dataset::shards::realize(name, &facet.declaration(), &probe)
            .map_err(|e| Error::Other(e.to_string()))?;

        let specs = shards
            .entries()
            .iter()
            .map(|e| self.shard_open_spec(name, &e.source.path))
            .collect::<Result<Vec<_>>>()?;

        // Two shards naming one file is a declared relationship and
        // shares a cache entry by design (SH-66, SH-80). Two *different*
        // files landing on one cache path is a collision, and the
        // cross-facet guard cannot see it — it compares facets, and
        // these are shards of the same one.
        let mut by_relpath: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();
        for spec in &specs {
            if let ShardOpen::Layered {
                resolved, relpath, ..
            } = spec
                && let Some(prev) = by_relpath.insert(relpath, resolved)
                && prev != resolved
            {
                return Err(Error::Other(format!(
                    "facet '{name}': shards '{prev}' and '{resolved}' both cache as \
                     '{relpath}'. Two shards may share a file, but two files cannot \
                     share a cache entry — rename one so each has a distinct filename."
                )));
            }
        }
        let series = Series::new(shards, specs);
        let first = series.file(0)?.clone();
        Ok(FacetStorage::series(first, series))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use url::Url;

    /// **The catalog and the open name the same file for a facet.** A
    /// listing, the picker's cache column and its cache drop resolve a
    /// facet through [`CatalogEntry::resolve_facet_url`]; the open
    /// resolves it through the view. A `..` segment or a character the
    /// request encodes once made them differ, and the cache drop then
    /// derived a cache path outside the dataset's directory.
    #[test]
    fn the_catalog_and_the_open_resolve_a_facet_alike() {
        let entry: crate::dataset::CatalogEntry = serde_json::from_str(
            r#"{"name":"ds","path":"https://h/cat/ds/dataset.yaml",
                "dataset_type":"dataset.yaml","layout":{"profiles":{}}}"#,
        )
        .unwrap();
        let home = crate::location::dir_form(entry.dataset_home_url());
        let config: crate::model::DatasetConfig =
            serde_yaml::from_str("name: ds\nprofiles:\n  default:\n    base_vectors: b.fvec\n").unwrap();
        let view = GenericTestDataView::new(
            Location::parse(&home).unwrap(),
            config.profiles["default"].clone(),
        );
        for raw in [
            "profiles/base/b.fvec",
            "../shared/b.fvec",
            "a b.fvec",
            "/local/b.fvec",
            "file:///local/b.fvec",
            "https://elsewhere/x/b.fvec",
        ] {
            let listed = entry.resolve_facet_url(raw).unwrap();
            let opened = view.resolve_path_str(raw).unwrap();
            assert_eq!(listed, opened, "{raw}");
            let relpath = facet_cache_relpath(&listed, &home);
            assert!(!relpath.split('/').any(|seg| seg == ".."), "{raw} caches at {relpath}");
        }
    }

    #[test]
    fn facet_cache_relpath_keeps_the_home_relative_path() {
        assert_eq!(facet_cache_relpath("https://h/ds/a/b.fvec", "https://h/ds/"), "a/b.fvec");
        assert_eq!(facet_cache_relpath("https://elsewhere/x/b.fvec", "https://h/ds/"), "b.fvec");
    }

    /// A local facet resolved through `Path::join` on Windows mixes
    /// separators; it must still cache under its home-relative path,
    /// not collapse to a basename another facet may share.
    #[cfg(windows)]
    #[test]
    fn facet_cache_relpath_reads_windows_separators() {
        assert_eq!(facet_cache_relpath(r"C:\tmp\ds\a/b__0001.fvec", r"C:\tmp\ds/"), "a/b__0001.fvec");
        assert_eq!(facet_cache_relpath(r"C:\tmp\ds\a\b.fvec", r"C:\tmp\ds\"), "a/b.fvec");
    }

    /// When a remote-loaded `TestDataGroup` has a facet whose source
    /// is a `file://` URI (a fully-precached local file), the resolver
    /// must yield the filesystem path — not try to join
    /// the URI onto the catalog's HTTP base. Symmetric for an
    /// absolute filesystem path. Same semantic as "remote facet that
    /// already lives on disk".
    #[test]
    fn remote_catalog_with_local_facet_short_circuits_to_filesystem() {
        use crate::model::FacetConfig;

        fn empty_profile() -> crate::model::ProfileConfig {
            crate::model::ProfileConfig {
                attributes: Default::default(),
                inherits: None,
                base_count: None,
                maxk: None,
                partition: false,
                base_vectors: None,
                base_content: None,
                query_vectors: None,
                query_terms: None,
                query_filters: None,
                neighbor_indices: None,
                neighbor_distances: None,
                prefiltered_neighbor_indices: None,
                prefiltered_neighbor_distances: None,
                postfiltered_neighbor_indices: None,
                postfiltered_neighbor_distances: None,
                metadata_content: None,
                metadata_predicates: None,
                predicate_results: None,
                metadata_layout: None,
                quant_scale: None,
                quant_zero_point: None,
                quant_sum: None,
            }
        }

        let mut profile = empty_profile();
        profile.base_vectors = Some(FacetConfig::Simple(
            "file:///tank/share/base.fvec".to_string(),
        ));
        profile.query_vectors = Some(FacetConfig::Simple("/tank/share/query.fvec".to_string()));

        let base_url = Url::parse("https://catalog.example.com/datasets/x/").unwrap();
        let view = GenericTestDataView::new(Location::Url(base_url), profile);

        let bv = view.config.base_vectors.as_ref().unwrap();
        let q = view.config.query_vectors.as_ref().unwrap();
        // The storage open resolves the same way (no `https://.../tank/...` join).
        assert_eq!(view.resolve_path_str("file:///tank/share/base.fvec").unwrap(), "/tank/share/base.fvec");
        assert_eq!(view.resolve_path_str("/tank/share/query.fvec").unwrap(), "/tank/share/query.fvec");
        // And so does the variable-length readers' resolution.
        assert_eq!(view.resolve_as_string(bv).unwrap(), "/tank/share/base.fvec");
        assert_eq!(view.resolve_as_string(q).unwrap(), "/tank/share/query.fvec");
    }

    /// Build a minimal uniform `.fvec` (dim-prefixed records) on disk
    /// and wrap it in a `FacetStorage` so the window-byte-range logic
    /// can read the real dim header at byte 0.
    fn fvec_storage(dir: &std::path::Path, dim: i32, records: usize) -> (PathBuf, FacetStorage) {
        let path = dir.join("base.fvec");
        let mut bytes = Vec::new();
        for r in 0..records {
            bytes.extend_from_slice(&dim.to_le_bytes());
            for d in 0..dim {
                bytes.extend_from_slice(&((r as f32) + d as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let storage = crate::storage::Storage::open_path(&path).unwrap();
        (path, FacetStorage::new(storage))
    }

    /// A facet declares its window in one of two encodings, and the
    /// whole-profile precache reads both, the way the reader does: the
    /// `[start..end)` suffix on a single source, else the `window:`
    /// field. The suffix wins when both are present; neither means no
    /// window. Consulting only the suffix once made the precache pull
    /// the entire base file for every field-form sized profile.
    #[test]
    fn a_declared_window_is_read_from_either_encoding() {
        let desc = |source: Option<&str>, window: Option<&str>| FacetDescriptor {
            name: "base_vectors".into(),
            source_path: source.map(str::to_string),
            source_type: None,
            window: window.map(str::to_string),
            standard_kind: None,
            shard_count: None,
        };
        let bounds = |d: &FacetDescriptor| {
            let w = facet_declared_window(d).unwrap();
            w.0.first().map(|iv| (iv.min_incl, iv.max_excl))
        };
        assert_eq!(bounds(&desc(Some("base.fvec[1..3)"), None)), Some((1, 3)));
        assert_eq!(bounds(&desc(Some("base.fvec"), Some("1..3"))), Some((1, 3)));
        assert_eq!(
            bounds(&desc(Some("base.fvec[1..3)"), Some("0..4"))),
            Some((1, 3)),
            "the path suffix takes precedence over the window field"
        );
        assert_eq!(bounds(&desc(Some("base.fvec"), None)), None);
        // A series has no single source; its window lives in the field.
        assert_eq!(bounds(&desc(None, Some("[0..500]"))), Some((0, 500)));
        assert_eq!(bounds(&desc(None, None)), None);
    }

    /// **A malformed window is an error, not an absent one.**
    ///
    /// Returning `None` here is what a facet with no window returns, and
    /// the caller reads that as "fetch the whole file". Conflating the
    /// two is how `base.fvec[0,1000)` came to download a terabyte in
    /// silence: the reader clipped to zero records, the precache path
    /// saw no window, and neither said why.
    #[test]
    fn a_malformed_window_is_an_error_not_an_absent_window() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, _storage) = fvec_storage(tmp.path(), 2, 4);
        let src = path.to_string_lossy().to_string();

        // And the declared-window helper the whole-profile precache
        // plans from propagates rather than quietly answering "no
        // window" — which would size, and fetch, the whole file.
        let desc = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: Some(format!("{src}[0,1000)")),
            source_type: None,
            window: None,
            standard_kind: None,
            shard_count: None,
        };
        assert!(
            facet_declared_window(&desc).is_err(),
            "a plan built on a malformed window must fail, not size the whole file"
        );
        let series_desc = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: None,
            source_type: None,
            window: Some("[0..500]".into()),
            standard_kind: None,
            shard_count: None,
        };
        let w = facet_declared_window(&series_desc).unwrap();
        assert_eq!((w.0[0].min_incl, w.0[0].max_excl), (0, 500), "a series declares its window in the field");
        let bad_field = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: Some(src.clone()),
            source_type: None,
            window: Some("0,1000".into()),
            standard_kind: None,
            shard_count: None,
        };
        let err = facet_declared_window(&bad_field).expect_err("the window field is checked too");
        assert!(err.to_string().contains("malformed"), "{err}");
        let bad_series = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: None,
            source_type: None,
            window: Some("0,1000".into()),
            standard_kind: None,
            shard_count: None,
        };
        assert!(facet_declared_window(&bad_series).is_err(), "a series' malformed window is an error too");
    }

    // ---- Coalescing -------------------------------------------------

    /// 100-byte chunks, so chunk N covers bytes [N*100, N*100+100).
    const CS: Option<u64> = Some(100);

    /// A range in the only shard — what every single-file facet has.
    fn sr(start: u64, end: u64) -> ShardRange {
        ShardRange::whole(start, end)
    }

    /// A range in a named shard.
    fn sr_at(shard: usize, start: u64, end: u64) -> ShardRange {
        ShardRange { shard, start, end }
    }

    #[test]
    fn coalescing_leaves_a_single_range_alone() {
        assert_eq!(coalesce_ranges(vec![], CS), Vec::<ShardRange>::new());
        assert_eq!(coalesce_ranges(vec![sr(10, 20)], CS), vec![sr(10, 20)]);
    }

    /// Overlapping and touching byte ranges merge under any granularity.
    #[test]
    fn overlapping_and_touching_ranges_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr(0, 50), sr(25, 75)], CS),
            vec![sr(0, 75)]
        );
        assert_eq!(
            coalesce_ranges(vec![sr(0, 50), sr(50, 75)], CS),
            vec![sr(0, 75)]
        );
        assert_eq!(
            coalesce_ranges(vec![sr(0, 50), sr(50, 75)], None),
            vec![sr(0, 75)]
        );
    }

    /// **Ranges in different shards never merge**, however adjacent
    /// their byte offsets look.
    ///
    /// Two shards are two files: byte 90 of shard 0 and byte 10 of
    /// shard 1 are in the same chunk *of different files*, and one
    /// request cannot cover both (SH-28). Without the shard in the key
    /// this is exactly the case that would silently fold together.
    #[test]
    fn ranges_in_different_shards_never_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr_at(0, 10, 20), sr_at(1, 20, 30)], CS),
            vec![sr_at(0, 10, 20), sr_at(1, 20, 30)]
        );
        // Even byte-identical ranges stay separate across shards.
        assert_eq!(
            coalesce_ranges(vec![sr_at(0, 0, 50), sr_at(1, 0, 50)], CS).len(),
            2
        );
        // And within one shard they still merge as before.
        assert_eq!(
            coalesce_ranges(vec![sr_at(2, 10, 20), sr_at(2, 80, 90)], CS),
            vec![sr_at(2, 10, 90)]
        );
    }

    /// **Two ranges in one chunk are already one fetch.** Issuing them
    /// separately asks the device for the same bytes twice.
    #[test]
    fn ranges_sharing_a_chunk_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(80, 90)], CS),
            vec![sr(10, 90)]
        );
        // Without chunking they are plainly disjoint and stay so.
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(80, 90)], None),
            vec![sr(10, 20), sr(80, 90)]
        );
    }

    /// Adjacent chunks are contiguous on the device, so one request
    /// covering both beats two.
    #[test]
    fn ranges_in_adjacent_chunks_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(150, 160)], CS),
            vec![sr(10, 160)]
        );
    }

    /// **But a whole chunk of gap is not bridged.** Merging across it
    /// would fetch a chunk nobody asked for — the exact failure the
    /// plan exists to make visible rather than commit.
    #[test]
    fn a_chunk_of_gap_is_not_bridged() {
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(250, 260)], CS),
            vec![sr(10, 20), sr(250, 260)]
        );
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(350, 360)], CS),
            vec![sr(10, 20), sr(350, 360)]
        );
    }

    /// Input order is not the caller's problem.
    #[test]
    fn ranges_are_sorted_before_merging() {
        // Chunks 0, 1 and 4. The first two are a run; chunks 2 and 3
        // lie untouched between that run and the last range.
        assert_eq!(
            coalesce_ranges(vec![sr(450, 460), sr(10, 20), sr(150, 160)], CS),
            vec![sr(10, 160), sr(450, 460)],
            "sorting happens first, then merging by chunk adjacency"
        );
    }

    /// A chain merges transitively, and an empty range contributes
    /// nothing rather than acting as a bridge.
    #[test]
    fn a_chain_merges_and_empty_ranges_drop_out() {
        assert_eq!(
            coalesce_ranges(vec![sr(0, 10), sr(100, 110), sr(200, 210)], CS),
            vec![sr(0, 210)],
            "chunks 0, 1 and 2 are a run"
        );
        assert_eq!(
            coalesce_ranges(vec![sr(0, 10), sr(50, 50), sr(300, 310)], CS),
            vec![sr(0, 10), sr(300, 310)],
            "an empty range is not a bridge"
        );
    }

    /// A degenerate chunk size must not divide by zero or merge
    /// everything into one range.
    #[test]
    fn a_zero_chunk_size_falls_back_to_byte_adjacency() {
        assert_eq!(
            coalesce_ranges(vec![sr(0, 10), sr(300, 310)], Some(0)),
            vec![sr(0, 10), sr(300, 310)]
        );
    }

    // ---- RangeFill ---------------------------------------------------

    /// `RangeFill`'s derived numbers are what a caller decides on: a
    /// fetch is chunk-granular, and both "already warm" and "this is
    /// secretly a full download" have to be readable off the struct.
    #[test]
    fn range_fill_reports_cost_at_chunk_granularity() {
        let f = RangeFill {
            first_chunk: 0,
            last_chunk: 1,
            chunk_size: 8 << 20,
            chunks: 2,
            chunks_resident: 1,
            resident_bytes: 8 << 20,
            aligned_start: 0,
            aligned_end: 16 << 20,
        };
        assert_eq!(f.chunks_to_fetch(), 1);
        assert_eq!(f.bytes_to_fetch(), 8 << 20, "resident chunks are free");
        assert!(!f.is_resident());
        assert_eq!(f.overfetch_bytes(0, 4096), (16 << 20) - 4096);

        let warm = RangeFill {
            chunks_resident: 2,
            resident_bytes: 16 << 20,
            ..f
        };
        assert!(warm.is_resident());
        assert_eq!(warm.bytes_to_fetch(), 0);

        // A span ending at the end of a file ends with a short chunk,
        // charged at its length, not at the chunk size.
        let tail = RangeFill {
            chunks_resident: 0,
            resident_bytes: 0,
            aligned_end: (8 << 20) + 100,
            ..f
        };
        assert_eq!(tail.bytes_to_fetch(), (8 << 20) + 100);
    }

    /// Local storage has no chunks, so it has no plan — and a caller
    /// must read that as "free", not as "unknown".
    #[test]
    fn local_storage_has_no_range_plan() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, storage) = fvec_storage(tmp.path(), 2, 4);
        assert!(storage.is_local());
        assert_eq!(storage.range_fill(0, 12), None);
    }

    // ---- The offset-index cache --------------------------------------

    /// Build a variable-length file: 4-byte dim then `dim` i32s, so no
    /// two records need be the same size.
    fn ivvec_storage(dir: &std::path::Path, dims: &[i32]) -> (PathBuf, FacetStorage) {
        use std::io::Write as _;
        let path = dir.join("meta.ivvec");
        let mut f = std::fs::File::create(&path).unwrap();
        for &d in dims {
            f.write_all(&d.to_le_bytes()).unwrap();
            for e in 0..d {
                f.write_all(&e.to_le_bytes()).unwrap();
            }
        }
        drop(f);
        let storage = crate::storage::Storage::open_path(&path).unwrap();
        (path, FacetStorage::new(storage))
    }

    /// **The index is loaded once per handle.** Two asks return the
    /// same allocation, not two equal ones — which is the difference
    /// between a cache and a coincidence.
    #[test]
    fn a_handle_loads_its_offset_index_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, storage) = ivvec_storage(tmp.path(), &[3, 1, 8, 2]);
        let src = path.to_string_lossy().to_string();

        let first = storage.published_offsets(&src, 4).expect("offsets load");
        let second = storage.published_offsets(&src, 4).expect("offsets load");
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "the second ask must be served from the handle, not reloaded"
        );
        assert_eq!(first.len(), 4);
    }

    /// A handle holding offsets for one element width will not answer
    /// for another. Offsets are computed by walking with a stride, so
    /// serving them for a different width resolves a window to
    /// plausible, wrong bytes.
    #[test]
    fn a_handle_refuses_offsets_for_a_different_element_width() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, storage) = ivvec_storage(tmp.path(), &[3, 1, 8, 2]);
        let src = path.to_string_lossy().to_string();

        assert!(storage.published_offsets(&src, 4).is_some());
        assert!(
            storage.published_offsets(&src, 8).is_none(),
            "a width mismatch degrades to no window rather than wrong bytes"
        );
    }

    /// **The cache is safe to race.** Two threads asking at once may
    /// both load — wasteful, never wrong — but every caller must come
    /// away with the *same* allocation, or "cached on the handle" is a
    /// claim that quietly does not hold under concurrency.
    #[test]
    fn concurrent_asks_all_see_one_allocation() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, storage) = ivvec_storage(tmp.path(), &[3, 1, 8, 2, 5, 9]);
        let src = path.to_string_lossy().to_string();
        let storage = std::sync::Arc::new(storage);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let storage = storage.clone();
                let src = src.clone();
                std::thread::spawn(move || {
                    storage.published_offsets(&src, 4).expect("offsets load")
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = &results[0];
        for (i, r) in results.iter().enumerate() {
            assert!(
                std::sync::Arc::ptr_eq(first, r),
                "thread {i} got a different allocation; the cache did not settle"
            );
        }
        assert_eq!(first.len(), 6);
    }

    // ── the descriptor budget ──────────────────────────────────────

    /// `n` one-file shards, each holding four bytes of the shard's own
    /// index, opened under an explicit budget.
    fn byte_series(dir: &std::path::Path, n: usize, cap: usize) -> Series {
        use crate::dataset::shards::{Entry, Shards};
        let mut entries = Vec::new();
        let mut opens = Vec::new();
        for i in 0..n {
            let path = dir.join(format!("part__{i:04}.u8"));
            std::fs::write(&path, [i as u8; 4]).unwrap();
            entries.push(Entry {
                source: crate::dataset::source::parse_source_string(
                    path.to_str().unwrap(),
                )
                .unwrap(),
                file_base: 0,
                len: 4,
            });
            opens.push(ShardOpen::Plain {
                resolved: path.to_str().unwrap().to_string(),
            });
        }
        Series::with_cap(Shards::new(entries).unwrap(), opens, cap)
    }

    /// Whether the series is currently holding file `i` open. Reads
    /// the slot directly rather than widening `Series`: eviction order
    /// is an internal policy, and this test module is inside the module
    /// that owns it.
    fn is_open(s: &Series, i: usize) -> bool {
        s.files[i].opened.lock().expect("shard slot").is_some()
    }

    /// An explicit setting wins; otherwise a quarter of the limit,
    /// never below the floor, and a junk value is ignored rather than
    /// fatal (SH-59).
    #[test]
    fn the_cap_derivation_answers_from_its_two_inputs() {
        assert_eq!(resolve_open_file_cap(Some("3"), Some(1024)), 3);
        assert_eq!(resolve_open_file_cap(Some(" 12 "), Some(1024)), 12);
        assert_eq!(resolve_open_file_cap(None, Some(1024)), 256);
        // No limit to read: the documented default stands in.
        assert_eq!(resolve_open_file_cap(None, None), 256);
        // The floor keeps a constrained host workable.
        assert_eq!(resolve_open_file_cap(None, Some(4)), 8);
        // Junk, zero, and negatives fall through to the derivation
        // rather than stopping a dataset from opening.
        for junk in ["", "lots", "0", "-1", "8.5"] {
            assert_eq!(
                resolve_open_file_cap(Some(junk), Some(1024)),
                256,
                "{junk:?} should not have been honoured"
            );
        }
    }

    /// **A series never claims more descriptors than it has files.** The
    /// budget is an upper bound, not a target.
    #[test]
    fn the_cap_never_exceeds_the_file_count() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 3, 4096);
        assert_eq!(s.cap, 3);
    }

    /// **Reading past the budget closes the least-recently-used file**
    /// (SH-59). With a real budget this branch needs more shards than
    /// the host has descriptors, which is why the cap is injectable.
    #[test]
    fn a_series_over_its_budget_closes_the_least_recently_used_file() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 8, 3);

        for i in 0..8 {
            let _ = s.file(i).unwrap();
            assert!(
                s.open_file_count() <= 3,
                "after opening {i}, {} files were open",
                s.open_file_count()
            );
        }
        // The budget bound is met exactly, and it is the *oldest* three
        // that were closed rather than an arbitrary three.
        assert_eq!(s.open_file_count(), 3);
        let open: Vec<usize> = (0..8).filter(|&i| is_open(&s, i)).collect();
        assert_eq!(open, vec![5, 6, 7]);
    }

    /// **An evicted file reopens and reads the same bytes.** Eviction
    /// is a descriptor economy, not a loss of data — a second pass over
    /// a series wider than the budget must answer exactly as the first.
    #[test]
    fn an_evicted_file_reopens_and_reads_the_same_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 8, 2);

        let first: Vec<u8> = (0..8)
            .map(|i| s.file(i).unwrap().read_bytes(0, 4).unwrap()[0])
            .collect();
        assert_eq!(first, (0..8).map(|i| i as u8).collect::<Vec<_>>());

        // Every file but the last two has been evicted by now; reading
        // backwards evicts the rest and reopens the front.
        let second: Vec<u8> = (0..8)
            .rev()
            .map(|i| s.file(i).unwrap().read_bytes(0, 4).unwrap()[0])
            .collect();
        assert_eq!(second, (0..8).rev().map(|i| i as u8).collect::<Vec<_>>());
        assert!(s.open_file_count() <= 2);
    }

    /// **Eviction never pulls a file out from under a reader** (SH-59).
    /// `file()` hands back an owned `Arc`, so a caller mid-read keeps
    /// its own claim after the series has closed its slot.
    #[test]
    fn eviction_cannot_invalidate_a_handle_a_caller_still_holds() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 6, 2);

        let held = s.file(0).unwrap();
        for i in 1..6 {
            let _ = s.file(i).unwrap();
        }
        assert!(!is_open(&s, 0), "file 0 should have been evicted");
        // The evicted slot is empty, but the handle taken before the
        // eviction still reads.
        assert_eq!(held.read_bytes(0, 4).unwrap(), vec![0u8; 4]);
    }

    /// **Parallel readers stay inside the budget and see their own
    /// bytes.** The LRU is shared mutable state behind a lock; a series
    /// read from several threads must neither exceed the cap nor return
    /// one shard's bytes for another's.
    #[test]
    fn parallel_readers_respect_the_budget_and_read_correctly() {
        let tmp = tempfile::tempdir().unwrap();
        let s = std::sync::Arc::new(byte_series(tmp.path(), 16, 4));

        std::thread::scope(|scope| {
            for t in 0..8 {
                let s = std::sync::Arc::clone(&s);
                scope.spawn(move || {
                    for pass in 0..40 {
                        let i = (t * 7 + pass * 3) % 16;
                        let bytes = s.file(i).unwrap().read_bytes(0, 4).unwrap();
                        assert_eq!(
                            bytes,
                            vec![i as u8; 4],
                            "thread {t} pass {pass} read shard {i}"
                        );
                        assert!(
                            s.open_file_count() <= 4,
                            "budget exceeded: {}",
                            s.open_file_count()
                        );
                    }
                });
            }
        });
        assert!(s.open_file_count() <= 4);
    }

    /// Separate handles are separate caches — the point of putting the
    /// cache on the handle. A caller wanting reuse holds one; a caller
    /// that does not pays per handle and nothing leaks.
    #[test]
    fn separate_handles_do_not_share_the_index() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, first) = ivvec_storage(tmp.path(), &[3, 1, 8, 2]);
        let src = path.to_string_lossy().to_string();
        let second = FacetStorage::new(crate::storage::Storage::open_path(&path).unwrap());

        let a = first.published_offsets(&src, 4).unwrap();
        let b = second.published_offsets(&src, 4).unwrap();
        assert_eq!(a, b, "same file, same offsets");
        assert!(
            !std::sync::Arc::ptr_eq(&a, &b),
            "but a distinct handle holds its own copy, with its own lifetime"
        );
    }
}
