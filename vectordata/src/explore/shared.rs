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
    pub(super) fn is_remote(&self) -> bool {
        matches!(self, UnifiedReader::Remote(_))
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

/// Wrapper that opens `base_vectors` (and exposes a cache-stats view)
/// through the canonical [`crate::TestDataView`] path.
pub(super) struct AnyDatasetReader {
    view: std::sync::Arc<dyn crate::TestDataView>,
    base: std::sync::Arc<dyn crate::VectorReader<f32>>,
    facet_storage: Option<crate::FacetStorage>,
    dim: usize,
    count: usize,
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
        // cache_stats() to the TUI's status line.
        let facet_storage = view.open_facet_storage("base_vectors").ok();
        let base = view.base_vectors().unwrap_or_else(|e| {
            eprintln!("error: failed to open base_vectors: {e}");
            std::process::exit(1);
        });
        let dim = base.dim();
        let count = base.count();
        AnyDatasetReader { view, base, facet_storage, dim, count }
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

/// Default number of consecutive vectors per clump.
///
/// For remote data, the effective clump size should ideally match the number
/// of vectors that fit in one merkle chunk (~1MB). With 1024-dim f16 vectors
/// (2KB each), that's ~512 vectors per chunk. A larger clump means each
/// downloaded chunk is more fully utilized, reducing total HTTP requests.
///
/// This default is a conservative floor for local data. The `clump_size_for`
/// helper scales it up based on vector size and chunk geometry.
pub(super) const DEFAULT_CLUMP_SIZE: usize = 32;

/// Compute an effective clump size based on vector entry size.
///
/// For remote access, aligns clumps to merkle chunk boundaries so each
/// downloaded chunk is maximally utilized. Falls back to DEFAULT_CLUMP_SIZE
/// for very small vectors or local access.
pub(super) fn clump_size_for_dim(dim: usize, is_remote: bool) -> usize {
    if !is_remote {
        return DEFAULT_CLUMP_SIZE;
    }
    // Estimate bytes per vector entry: dim * element_size + 4 (dimension header)
    // Use 2 bytes/element (f16) as conservative estimate; f32 would be 4.
    let entry_size = dim * 2 + 4;
    // Merkle default chunk size is 1MB
    let chunk_size = 1024 * 1024;
    let vecs_per_chunk = chunk_size / entry_size.max(1);
    // Use the full chunk capacity, with a floor of DEFAULT_CLUMP_SIZE
    vecs_per_chunk.max(DEFAULT_CLUMP_SIZE)
}

/// Generate sample indices according to the sampling mode.
pub(super) fn sample_indices(total: usize, effective: usize, seed: u64, mode: SampleMode, clump_size: usize) -> Vec<usize> {
    if effective >= total {
        return (0..total).collect();
    }
    match mode {
        SampleMode::Streaming => {
            (0..effective).collect()
        }
        SampleMode::Clumped => {
            // Evenly-spaced clumps of consecutive vectors.
            // num_clumps × clump_size ≈ effective
            let clump_size = clump_size;
            let num_clumps = (effective + clump_size - 1) / clump_size;
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
