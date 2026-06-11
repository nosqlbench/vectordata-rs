// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for explore subcommands: data access, sampling,
//! caching, and common utilities.

use std::path::PathBuf;


/// Sampling strategy for visualization commands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleMode {
    /// Read the first N vectors sequentially. Fast for remote/cached data
    /// because it accesses contiguous chunks.
    Streaming,
    /// Clumped sparse sampling — evenly-spaced clumps of consecutive vectors.
    /// Covers the full distribution while maintaining access locality.
    /// Each clump reads `clump_size` consecutive vectors, positioned at
    /// evenly-spaced intervals across the file.
    Clumped,
    /// Pseudo-random sparse sampling via Fisher-Yates shuffle. Best
    /// distribution coverage but worst access locality — slow for remote data.
    Sparse,
}

/// Parse a `SampleMode` from a CLI string.
pub(crate) fn parse_sample_mode(s: &str) -> Result<SampleMode, String> {
    match s.to_lowercase().as_str() {
        "streaming" | "stream" | "sequential" | "seq" => Ok(SampleMode::Streaming),
        "clumped" | "clump" => Ok(SampleMode::Clumped),
        "sparse" | "random" => Ok(SampleMode::Sparse),
        _ => Err(format!("unknown sample mode '{}': use streaming, clumped, or sparse", s)),
    }
}

impl std::str::FromStr for SampleMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_sample_mode(s)
    }
}

/// Resolve a non-local source — either a `dataset[:profile]` catalog
/// specifier or a remote URL pointing at a dataset directory — and
/// return the canonical [`TestDataView`]. URL sources use the default
/// profile (URLs don't carry a `:profile` suffix because the colon is
/// reserved for the scheme separator).
pub(super) fn open_dataset_view(source: &str) -> std::sync::Arc<dyn crate::TestDataView> {
    if crate::transport::is_remote_url(source) {
        let group = crate::TestDataGroup::load(source).unwrap_or_else(|e| {
            eprintln!("error: failed to load {source}: {e}");
            std::process::exit(1);
        });
        return group.profile("default").unwrap_or_else(|| {
            eprintln!(
                "error: profile 'default' not found at {source}. Available: {}",
                group.profile_names().join(", "),
            );
            std::process::exit(1);
        });
    }
    let (name, profile) = match source.find(':') {
        Some(pos) => (&source[..pos], &source[pos + 1..]),
        None => (source, "default"),
    };
    let sources = crate::catalog::sources::CatalogSources::new().configure_default();
    let catalog = crate::catalog::resolver::Catalog::of(&sources);
    catalog.open_profile(name, profile).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    })
}

/// Check if a source specifier refers to a local file. Remote URLs
/// (`http://`, `https://`, `s3://`) and bare catalog specs
/// (`dataset[:profile]`) both classify as non-local — only paths
/// (either existing on disk, prefixed with a directory separator, or
/// using a `file://` scheme) are treated as local.
pub(super) fn is_local_source(source: &str) -> bool {
    if crate::transport::is_remote_url(source) { return false; }
    if source.starts_with("file://") { return true; }
    if source.contains('/') { return true; }
    std::path::Path::new(source).exists()
}

/// Unified data reader that wraps both local (AnyVectorReader) and
/// remote (AnyDatasetReader) access behind the same interface.
pub(super) enum UnifiedReader {
    Local(AnyVectorReader),
    Remote(AnyDatasetReader),
}

impl UnifiedReader {
    pub(super) fn open(source: &str) -> Self {
        if is_local_source(source) {
            let path = resolve_source(source);
            UnifiedReader::Local(AnyVectorReader::open(&path))
        } else {
            UnifiedReader::Remote(AnyDatasetReader::from_source(source))
        }
    }
    pub(super) fn count(&self) -> usize {
        match self { UnifiedReader::Local(r) => r.count(), UnifiedReader::Remote(r) => r.count() }
    }
    pub(super) fn dim(&self) -> usize {
        match self { UnifiedReader::Local(r) => r.dim(), UnifiedReader::Remote(r) => r.dim() }
    }
    pub(super) fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        match self { UnifiedReader::Local(r) => r.get_f32(index), UnifiedReader::Remote(r) => r.get_f32(index) }
    }
    pub(super) fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        match self {
            UnifiedReader::Local(r)  => r.get_f32_range(start, count),
            UnifiedReader::Remote(r) => r.get_f32_range(start, count),
        }
    }
    pub(super) fn cache_stats(&self) -> Option<crate::CacheStats> {
        match self {
            UnifiedReader::Local(_) => None,
            UnifiedReader::Remote(r) => r.cache_stats(),
        }
    }
    /// Chunk geometry of the underlying remote storage, when the
    /// source is chunk-backed. `None` for local files.
    pub(super) fn chunk_geometry(&self) -> Option<ChunkGeometry> {
        match self {
            UnifiedReader::Local(_) => None,
            UnifiedReader::Remote(r) => r.chunk_geometry(),
        }
    }
    /// See [`AnyDatasetReader::prefetch_byte_range`]. No-op for local.
    pub(super) fn prefetch_byte_range(&self, byte_start: u64, byte_end: u64) -> std::io::Result<()> {
        match self {
            UnifiedReader::Local(_) => Ok(()),
            UnifiedReader::Remote(r) => r.prefetch_byte_range(byte_start, byte_end),
        }
    }
}

/// Resolve a *local* source specifier to a filesystem path.
///
/// Only meant to be called when [`is_local_source`] returns true.
/// Remote sources (URL, catalog spec) bypass this helper entirely
/// and route through [`AnyDatasetReader::from_source`], which opens
/// them in sparse mode (chunked HTTP cache) without writing a
/// fully-materialised local file. The picker's Precache action is
/// the explicit channel for triggering a full download.
pub(super) fn resolve_source(source: &str) -> PathBuf {
    if let Some(rest) = source.strip_prefix("file://") {
        return PathBuf::from(rest);
    }
    let p = std::path::Path::new(source);
    if p.exists() {
        return p.to_path_buf();
    }
    eprintln!("error: file not found: {source}");
    std::process::exit(1);
}

/// Format-agnostic vector reader that returns f64 values.
///
/// Wraps any supported vector format (fvec, mvec, dvec) and converts
/// individual vectors to f64 on read. No bulk file conversion needed.
pub(super) enum AnyVectorReader {
    F32(crate::io::XvecReader<f32>),
    F16(crate::io::XvecReader<half::f16>),
}

impl AnyVectorReader {
    /// Open a vector file, auto-detecting format from extension.
    pub(super) fn open(path: &std::path::Path) -> Self {
        use crate::io::XvecReader;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match ext {
            "fvec" | "fvecs" => {
                AnyVectorReader::F32(XvecReader::<f32>::open_path(path).unwrap_or_else(|e| {
                    eprintln!("error: failed to open {}: {}", path.display(), e);
                    std::process::exit(1);
                }))
            }
            "mvec" | "mvecs" => {
                AnyVectorReader::F16(XvecReader::<half::f16>::open_path(path).unwrap_or_else(|e| {
                    eprintln!("error: failed to open {}: {}", path.display(), e);
                    std::process::exit(1);
                }))
            }
            _ => {
                eprintln!("error: unsupported format '.{}' for visualization. Use fvec or mvec.", ext);
                std::process::exit(1);
            }
        }
    }

    pub(super) fn count(&self) -> usize {
        use crate::VectorReader;
        match self {
            AnyVectorReader::F32(r) => <crate::io::XvecReader<f32> as VectorReader<f32>>::count(r),
            AnyVectorReader::F16(r) => <crate::io::XvecReader<half::f16> as VectorReader<half::f16>>::count(r),
        }
    }

    pub(super) fn dim(&self) -> usize {
        use crate::VectorReader;
        match self {
            AnyVectorReader::F32(r) => <crate::io::XvecReader<f32> as VectorReader<f32>>::dim(r),
            AnyVectorReader::F16(r) => <crate::io::XvecReader<half::f16> as VectorReader<half::f16>>::dim(r),
        }
    }

    /// Read a single vector as f32 values (for simsimd hot paths).
    pub(super) fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        use crate::VectorReader;
        match self {
            AnyVectorReader::F32(r) => {
                r.get(index).ok().map(|v| v.to_vec())
            }
            AnyVectorReader::F16(r) => {
                r.get(index).ok().map(|v| v.iter().map(|x| x.to_f32()).collect())
            }
        }
    }

    pub(super) fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        (0..count).map(|i| self.get_f32(start + i)).collect()
    }

}

/// Byte-level geometry of a remote facet, resolved once at open time.
/// Lets sampling and prefetch reason in transfer-chunk units: which
/// chunk a record lands in, and how many records one chunk holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ChunkGeometry {
    /// Bytes per record: 4-byte dim header + `dim × element size`.
    pub entry_bytes: u64,
    /// Transfer-chunk size of the backing store (8 MiB for the
    /// merkle-less chunked-HTTP path, whatever the published `.mref`
    /// declares for the merkle path).
    pub chunk_bytes: u64,
    /// Ordinal of the window's first record in the underlying file —
    /// 0 for unwindowed facets. Chunk boundaries are file-global, so
    /// windowed profiles must offset before mapping records to chunks.
    pub window_start: u64,
}

impl ChunkGeometry {
    /// Records per chunk, floored (a record straddling the boundary
    /// counts toward neither side). Never zero.
    pub(super) fn records_per_chunk(&self) -> usize {
        ((self.chunk_bytes / self.entry_bytes.max(1)) as usize).max(1)
    }

    /// File-global chunk ordinal containing the FIRST byte of the
    /// window-local record `i`.
    fn chunk_of(&self, i: usize) -> u64 {
        ((self.window_start + i as u64) * self.entry_bytes) / self.chunk_bytes
    }

    /// File-global chunk ordinal containing the LAST byte of the
    /// window-local record `i` — differs from [`Self::chunk_of`] when
    /// the record straddles a chunk boundary.
    fn chunk_of_end(&self, i: usize) -> u64 {
        ((self.window_start + i as u64 + 1) * self.entry_bytes - 1) / self.chunk_bytes
    }
}

/// Wrapper that opens `base_vectors` (and exposes a cache-stats view)
/// through the canonical [`crate::TestDataView`] path.
pub(super) struct AnyDatasetReader {
    view: std::sync::Arc<dyn crate::TestDataView>,
    base: std::sync::Arc<dyn crate::VectorReader<f32>>,
    facet_storage: Option<crate::FacetStorage>,
    dim: usize,
    count: usize,
    /// Bytes per record in the underlying file (dim header + payload).
    entry_bytes: u64,
    /// Window offset of this profile's base facet into the underlying
    /// file, in records. 0 when the facet is unwindowed.
    window_start: u64,
}

impl AnyDatasetReader {
    pub(super) fn from_source(source: &str) -> Self {
        let view = open_dataset_view(source);
        // Sparse access mode: don't prebuffer base_vectors at open
        // time. Reads route through the chunked-HTTP storage layer
        // (Storage::Cached or Storage::Http) which fetches 8 MiB
        // chunks on demand and caches them locally. A 50k-vector
        // streaming sample window at dim=384 f32 is ~75 MB ≈ 10
        // chunks — well under the previous "download everything
        // first" cost. The user can still trigger a full
        // prebuffer via the picker's Precache action.
        //
        // `open_facet_storage` is still called so the resulting
        // `AnyDatasetReader::facet_storage` handle can serve
        // cache_stats() to the TUI's status line and the chunk
        // geometry to the sampler/prefetcher.
        let facet_storage = view.open_facet_storage("base_vectors").ok();
        let base = view.base_vectors().unwrap_or_else(|e| {
            eprintln!("error: failed to open base_vectors: {e}");
            std::process::exit(1);
        });
        let dim = base.dim();
        let count = base.count();
        let elem = crate::io::infer_elem_size(crate::io::ext_of(
            view.facet_source("base_vectors").as_deref().unwrap_or(""),
        ));
        // Unknown extension: assume f32, the only element type the
        // remote f32 reader path can have opened anyway.
        let elem = if elem == 0 { 4 } else { elem };
        let entry_bytes = 4 + (dim as u64) * (elem as u64);
        let window_start = base_window_start(&*view);
        AnyDatasetReader { view, base, facet_storage, dim, count, entry_bytes, window_start }
    }

    /// Chunk geometry of the base facet, when it is chunk-backed
    /// remote storage. `None` for local mmap (no transfer chunks)
    /// and when the facet storage handle could not be opened.
    pub(super) fn chunk_geometry(&self) -> Option<ChunkGeometry> {
        let stats = self.facet_storage.as_ref()?.cache_stats()?;
        if stats.chunk_size == 0 { return None; }
        Some(ChunkGeometry {
            entry_bytes: self.entry_bytes,
            chunk_bytes: stats.chunk_size,
            window_start: self.window_start,
        })
    }

    /// Drive the chunks covering `[byte_start, byte_end)` of the base
    /// facet to locally-resident state. Blocking; parallel within the
    /// range. No-op when there is no storage handle.
    pub(super) fn prefetch_byte_range(&self, byte_start: u64, byte_end: u64) -> std::io::Result<()> {
        match &self.facet_storage {
            Some(fs) => fs.prebuffer_range_with_progress(byte_start, byte_end, |_| {}),
            None => Ok(()),
        }
    }

    pub(super) fn count(&self) -> usize { self.count }
    pub(super) fn dim(&self) -> usize { self.dim }

    pub(super) fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        self.base.get(index).ok()
    }
    pub(super) fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        (0..count).map(|i| self.get_f32(start + i)).collect()
    }

    pub(super) fn cache_stats(&self) -> Option<crate::CacheStats> {
        self.facet_storage.as_ref()?.cache_stats()
    }

    /// Suppress unused-field warning — the view handle keeps the
    /// underlying dataset alive for the lifetime of `base`/`facet_storage`.
    #[allow(dead_code)]
    pub(super) fn view(&self) -> &dyn crate::TestDataView { &*self.view }
}

/// Resolve the base facet's window offset (in records) into the
/// underlying file. Sized profiles express their window either as a
/// `[start..end)` suffix on the source string (the canonical sugar)
/// or as an explicit `window:` field on a `Detailed` facet config —
/// the facet manifest carries both forms, so consult both. 0 when
/// the facet is unwindowed or the window is unparseable (reads stay
/// correct either way; only chunk alignment degrades).
fn base_window_start(view: &dyn crate::TestDataView) -> u64 {
    let manifest = view.facet_manifest();
    let Some(desc) = manifest.get("base_vectors") else { return 0 };
    if let Some(raw) = &desc.source_path
        && let Ok(parsed) = crate::dataset::source::parse_source_string(raw)
        && let Some(iv) = parsed.window.0.first()
    {
        return iv.min_incl;
    }
    if let Some(w) = &desc.window
        && let Ok(parsed) = crate::dataset::source::parse_window(w)
        && let Some(iv) = parsed.0.first()
    {
        return iv.min_incl;
    }
    0
}

/// Default number of consecutive vectors per clump, used when the
/// source has no chunk geometry to align with (local mmap). Pure
/// access-locality clumping — large enough to amortize per-record
/// overhead, small enough that clumps stay well spread.
pub(super) const DEFAULT_CLUMP_SIZE: usize = 32;

/// Effective clump size for the given source geometry: exactly the
/// number of records one transfer chunk holds, so a clump consumes
/// whole downloaded blocks. Falls back to [`DEFAULT_CLUMP_SIZE`]
/// when there is no chunk geometry (local files) or for very small
/// records where a single chunk would make clumps absurdly small.
pub(super) fn clump_size_for(geometry: Option<ChunkGeometry>) -> usize {
    match geometry {
        Some(g) => g.records_per_chunk().max(DEFAULT_CLUMP_SIZE),
        None => DEFAULT_CLUMP_SIZE,
    }
}

/// Generate sample indices according to the sampling mode.
///
/// When `geometry` is known (chunk-backed remote source), `Clumped`
/// mode produces *chunk-aligned* clumps: evenly-spaced whole transfer
/// chunks, each contributing every record fully contained in it. This
/// is the intended remote trade-off — pay for whole blocks, spread
/// them across the distribution, and waste none of the downloaded
/// bytes. Without geometry, `Clumped` falls back to evenly-spaced
/// fixed-size clumps (access locality only).
pub(super) fn sample_indices(
    total: usize,
    effective: usize,
    seed: u64,
    mode: SampleMode,
    clump_size: usize,
    geometry: Option<ChunkGeometry>,
) -> Vec<usize> {
    if effective >= total {
        return (0..total).collect();
    }
    match mode {
        SampleMode::Streaming => {
            (0..effective).collect()
        }
        SampleMode::Clumped => {
            if let Some(g) = geometry {
                return chunk_aligned_clumps(total, effective, g);
            }
            // No chunk geometry — evenly-spaced clumps of consecutive
            // vectors. num_clumps × clump_size ≈ effective.
            let num_clumps = effective.div_ceil(clump_size);
            let stride = total / num_clumps.max(1);

            let mut indices = Vec::with_capacity(effective);
            for c in 0..num_clumps {
                let start = c * stride;
                for offset in 0..clump_size {
                    let idx = start + offset;
                    if idx < total && indices.len() < effective {
                        indices.push(idx);
                    }
                }
            }
            indices
        }
        SampleMode::Sparse => {
            let mut rng = crate::explore::seeded_rng(seed);
            use rand::Rng;
            let mut idx: Vec<usize> = (0..total).collect();
            for i in 0..effective {
                let j = rng.random_range(i..total);
                idx.swap(i, j);
            }
            idx.truncate(effective);
            idx
        }
    }
}

/// Chunk-aligned clumped sampling: pick `ceil(effective /
/// records_per_chunk)` whole transfer chunks, evenly spaced across
/// the chunks the window covers, and take every record *fully
/// contained* in each picked chunk (a record straddling a chunk
/// boundary would force a second chunk download, defeating the
/// whole-block trade-off). If edge clipping leaves the sample short,
/// additional unpicked chunks are appended (in ascending order) until
/// `effective` is met or the window is exhausted.
pub(super) fn chunk_aligned_clumps(total: usize, effective: usize, g: ChunkGeometry) -> Vec<usize> {
    let (e, c) = (g.entry_bytes, g.chunk_bytes);
    if e == 0 || c == 0 {
        return (0..effective.min(total)).collect();
    }
    let first_chunk = g.chunk_of(0);
    let last_chunk = g.chunk_of_end(total.saturating_sub(1));
    let span_chunks = last_chunk - first_chunk + 1;

    // Window-local ordinals of the records fully inside chunk `q`.
    let recs_in = |q: u64| -> std::ops::Range<usize> {
        let lo_global = (q * c).div_ceil(e);
        let hi_global = ((q + 1) * c) / e;
        let lo = lo_global.saturating_sub(g.window_start).min(total as u64) as usize;
        let hi = hi_global.saturating_sub(g.window_start).min(total as u64) as usize;
        lo..hi.max(lo)
    };

    let needed = (effective.div_ceil(g.records_per_chunk()) as u64)
        .clamp(1, span_chunks);
    let stride = (span_chunks / needed).max(1);
    let mut picked: Vec<u64> = (0..needed).map(|k| first_chunk + k * stride).collect();
    let mut chosen: std::collections::HashSet<u64> = picked.iter().copied().collect();

    let mut indices = Vec::with_capacity(effective);
    let mut cursor = 0usize;
    while indices.len() < effective && cursor < picked.len() {
        for r in recs_in(picked[cursor]) {
            if indices.len() >= effective { break; }
            indices.push(r);
        }
        cursor += 1;
        // Top-up: boundary clipping can leave each chunk a few records
        // short of records_per_chunk. Pull in the next unpicked chunk
        // until the target is met or the window has no chunks left.
        if cursor == picked.len() && indices.len() < effective
            && let Some(q) = (first_chunk..=last_chunk).find(|q| !chosen.contains(q)) {
                chosen.insert(q);
                picked.push(q);
            }
    }
    indices
}

/// Map sample indices to the distinct file-global transfer chunks
/// they touch, in first-touch order. A record straddling a chunk
/// boundary contributes both chunks. This is the prefetcher's work
/// list: warming exactly these chunks (in this order) keeps the
/// parallel readahead aligned with the serial read order.
pub(super) fn chunks_for_indices(indices: &[usize], g: ChunkGeometry) -> Vec<u64> {
    let mut seen = std::collections::HashSet::new();
    let mut chunks = Vec::new();
    for &i in indices {
        for q in [g.chunk_of(i), g.chunk_of_end(i)] {
            if seen.insert(q) {
                chunks.push(q);
            }
        }
    }
    chunks
}

/// Spawn parallel readahead for a sample over a chunk-backed remote
/// source. Maps `indices` to their covering chunks and warms them
/// with [`crate::cache::download_concurrency`] worker threads, each
/// driving one chunk at a time through the storage layer's
/// prebuffer path. The storage layer dedups in-flight chunk fetches,
/// so the serial read thread consuming the same sample blocks on
/// chunks already being fetched here instead of downloading them
/// itself — turning the read path's one-connection serial fetch into
/// a pool-wide parallel one without restructuring the reader.
///
/// `cancel` is checked between chunks; setting it abandons the
/// remaining work list (bounded overshoot: at most one in-flight
/// chunk per worker). No-op for local sources.
pub(super) fn spawn_chunk_prefetch(
    reader: std::sync::Arc<UnifiedReader>,
    indices: &[usize],
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let Some(g) = reader.chunk_geometry() else { return };
    let chunks = chunks_for_indices(indices, g);
    if chunks.is_empty() { return; }
    let queue = std::sync::Arc::new(std::sync::Mutex::new(
        std::collections::VecDeque::from(chunks),
    ));
    let workers = crate::cache::download_concurrency()
        .min(queue.lock().unwrap().len())
        .max(1);
    for _ in 0..workers {
        let reader = reader.clone();
        let queue = queue.clone();
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            loop {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) { break; }
                let q = queue.lock().unwrap().pop_front();
                let Some(q) = q else { break };
                // Errors are not surfaced here — the serial read path
                // hits the same chunk next and reports the failure
                // with full context.
                if reader.prefetch_byte_range(q * g.chunk_bytes, (q + 1) * g.chunk_bytes).is_err() {
                    break;
                }
            }
        });
    }
}

/// Split sample indices into contiguous runs of at most `cap`
/// indices each, returned as `(start_index, run_length)` pairs.
///
/// Used by the explore read thread to drive `get_f32_range`: a run
/// is served with one range read, and the cap bounds how many
/// vectors that read collects before control returns to the caller.
/// Without the cap, Streaming mode (one contiguous block spanning
/// the whole sample) would gather every vector — and on a remote
/// source, download every covering chunk — inside a single call,
/// so no progress batch could reach the UI until the entire sample
/// was resident. With the cap, each segment completes after at most
/// a chunk fetch or two and the vectors-loaded counter advances as
/// data arrives.
///
/// Sparse mode produces no contiguous neighbors, so every run has
/// length 1 and behaviour matches a per-index loop.
pub(super) fn contiguous_runs(indices: &[usize], cap: usize) -> Vec<(usize, usize)> {
    let cap = cap.max(1);
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < indices.len() {
        let mut end = i + 1;
        while end < indices.len()
            && end - i < cap
            && indices[end] == indices[end - 1] + 1 {
            end += 1;
        }
        runs.push((indices[i], end - i));
        i = end;
    }
    runs
}

/// Install a SIGINT (Ctrl-C) handler.
///
/// Sets the abort flag AND restores the terminal. The handler runs
/// in signal context so it uses only async-signal-safe operations.
/// After restoring the terminal, it force-exits the process to
/// ensure we don't hang on blocking I/O.
pub(super) fn install_abort_handler(flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    // POSIX-only: signal-hook's `SIGINT` constant and flag registration
    // are not available on Windows. On Windows the OS default Ctrl-C
    // handler runs immediately (same behaviour as the
    // `register_conditional_default` second-press branch on Unix), so
    // the function degrades to a no-op rather than offering the
    // cooperative-first-press-then-hard-second-press flow.
    #[cfg(unix)]
    {
        // First: register the cooperative flag
        let _ = signal_hook::flag::register(signal_hook::consts::SIGINT, flag.clone());
        // Second: register conditional default — if flag already set (second Ctrl-C),
        // the OS default handler runs (immediate termination)
        let _ = signal_hook::flag::register_conditional_default(
            signal_hook::consts::SIGINT,
            flag,
        );
    }
    #[cfg(not(unix))]
    {
        let _ = flag; // suppress unused-variable warning
    }
}

/// Normalize a vector to unit length.
pub(super) fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A streaming-mode sample (one fully contiguous block) must be
    /// split into cap-sized segments — an uncapped single run is the
    /// regression where the whole remote sample downloaded before
    /// the first progress batch reached the UI.
    #[test]
    fn streaming_block_splits_at_cap() {
        let indices: Vec<usize> = (0..1250).collect();
        let runs = contiguous_runs(&indices, 500);
        assert_eq!(runs, vec![(0, 500), (500, 500), (1000, 250)]);
    }

    /// Sparse (non-adjacent) indices each form a length-1 run.
    #[test]
    fn sparse_indices_yield_unit_runs() {
        let indices = vec![7, 3, 100, 42];
        let runs = contiguous_runs(&indices, 500);
        assert_eq!(runs, vec![(7, 1), (3, 1), (100, 1), (42, 1)]);
    }

    /// Clumped samples keep clump boundaries; clumps longer than the
    /// cap split, shorter ones pass through intact.
    #[test]
    fn clumps_split_only_when_longer_than_cap() {
        let mut indices: Vec<usize> = (0..8).collect();      // clump of 8
        indices.extend(1000..1003);                           // clump of 3
        let runs = contiguous_runs(&indices, 4);
        assert_eq!(runs, vec![(0, 4), (4, 4), (1000, 3)]);
    }

    #[test]
    fn empty_and_degenerate_cap() {
        assert!(contiguous_runs(&[], 500).is_empty());
        // cap 0 is clamped to 1 instead of looping forever
        assert_eq!(contiguous_runs(&[5, 6], 0), vec![(5, 1), (6, 1)]);
    }

    fn geo(entry: u64, chunk: u64, window: u64) -> ChunkGeometry {
        ChunkGeometry { entry_bytes: entry, chunk_bytes: chunk, window_start: window }
    }

    /// Clump size equals whole-chunk capacity when geometry is known;
    /// falls back to the locality floor for local files.
    #[test]
    fn clump_size_tracks_chunk_capacity() {
        // dim=256 f32: entry 1028 B; 8 MiB chunk → 8160 records.
        assert_eq!(clump_size_for(Some(geo(1028, 8 << 20, 0))), 8160);
        // Tiny chunks never shrink the clump below the floor.
        assert_eq!(clump_size_for(Some(geo(1028, 4096, 0))), DEFAULT_CLUMP_SIZE);
        assert_eq!(clump_size_for(None), DEFAULT_CLUMP_SIZE);
    }

    /// Whole-block invariant: every index a chunk-aligned clump
    /// produces maps to a record fully contained in one of the
    /// evenly-spaced picked chunks — no byte of any other chunk is
    /// needed to serve the sample.
    #[test]
    fn aligned_clumps_stay_within_whole_chunks() {
        // entry 300 B, chunk 1000 B → records straddle boundaries.
        let g = geo(300, 1000, 0);
        let total = 100;       // 30 000 bytes → chunks 0..=29
        // 3 recs/chunk → 3 picked chunks (stride 10), which after
        // boundary clipping yield exactly 3 + 2 + 3 = 8 records —
        // ask for all 8 so no top-up chunk is pulled in.
        let effective = 8;
        let indices = chunk_aligned_clumps(total, effective, g);
        assert_eq!(indices.len(), effective);
        for &r in &indices {
            let (lo, hi) = (r as u64 * 300, (r as u64 + 1) * 300);
            let q = lo / 1000;
            assert_eq!(hi.div_ceil(1000) - 1, q,
                "record {r} ({lo}..{hi}) must not straddle a chunk boundary");
        }
        // Even spread: picked chunks are 0, 10, 20.
        let chunks = chunks_for_indices(&indices, g);
        assert_eq!(chunks, vec![0, 10, 20]);
    }

    /// Boundary clipping makes each chunk yield fewer records than
    /// `records_per_chunk`; the top-up pass appends more chunks until
    /// the requested sample size is met.
    #[test]
    fn aligned_clumps_top_up_after_clipping() {
        let g = geo(300, 1000, 0);
        let total = 100;
        // 3 fully-contained records per 1000-byte chunk (floor of
        // 3.33), so 33 records cannot come from ceil(33/3)=11 chunks
        // alone if any clip — request a size that forces top-up.
        let effective = 34;
        let indices = chunk_aligned_clumps(total, effective, g);
        assert_eq!(indices.len(), effective);
        // No duplicates.
        let unique: std::collections::HashSet<_> = indices.iter().collect();
        assert_eq!(unique.len(), indices.len());
    }

    /// Windowed profiles map records to file-global chunks: window
    /// start 5 at entry 100 puts local record 0 at byte 500, inside
    /// chunk 0, and local record 5 at byte 1000 — chunk 1.
    #[test]
    fn window_offsets_shift_chunk_mapping() {
        let g = geo(100, 1000, 5);
        assert_eq!(chunks_for_indices(&[0], g), vec![0]);
        assert_eq!(chunks_for_indices(&[5], g), vec![1]);
        // Aligned clumps express results in window-local ordinals.
        let indices = chunk_aligned_clumps(20, 5, g);
        assert_eq!(indices.len(), 5);
        for &r in &indices {
            assert!(r < 20, "window-local ordinal expected, got {r}");
        }
    }

    /// A record straddling a chunk boundary contributes both covering
    /// chunks to the prefetch work list, in first-touch order.
    #[test]
    fn straddling_records_prefetch_both_chunks() {
        let g = geo(300, 1000, 0);
        // record 3 spans bytes 900..1200 → chunks 0 and 1.
        assert_eq!(chunks_for_indices(&[3], g), vec![0, 1]);
        // Dedup across indices: records 0..=3 touch chunks 0 and 1 once.
        assert_eq!(chunks_for_indices(&[0, 1, 2, 3], g), vec![0, 1]);
    }
}
