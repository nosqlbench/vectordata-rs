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

/// Resolve a `dataset[:profile]` specifier through the catalog and
/// return the canonical [`TestDataView`].
pub(super) fn open_dataset_view(source: &str) -> std::sync::Arc<dyn vectordata::TestDataView> {
    let (name, profile) = match source.find(':') {
        Some(pos) => (&source[..pos], &source[pos + 1..]),
        None => (source, "default"),
    };
    let sources = crate::catalog::sources::CatalogSources::new().configure_default();
    let catalog = crate::catalog::resolver::Catalog::of(&sources);
    catalog.open_profile(name, profile).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    })
}

/// Check if a source specifier refers to a local file.
pub(super) fn is_local_source(source: &str) -> bool {
    let as_path = std::path::Path::new(source);
    as_path.exists() || source.contains('/') || source.contains('.')
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
    pub(super) fn get_f64(&self, index: usize) -> Option<Vec<f64>> {
        match self { UnifiedReader::Local(r) => r.get_f64(index), UnifiedReader::Remote(r) => r.get_f64(index) }
    }
    pub(super) fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        match self { UnifiedReader::Local(r) => r.get_f32(index), UnifiedReader::Remote(r) => r.get_f32(index) }
    }
    /// Batched range read; lets remote-backed views satisfy the whole
    /// visible window of the values grid in a single HTTP-fetch round
    /// instead of `count` serialized round trips.
    pub(super) fn get_f64_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f64>>> {
        match self {
            UnifiedReader::Local(r)  => r.get_f64_range(start, count),
            UnifiedReader::Remote(r) => r.get_f64_range(start, count),
        }
    }
    pub(super) fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        match self {
            UnifiedReader::Local(r)  => r.get_f32_range(start, count),
            UnifiedReader::Remote(r) => r.get_f32_range(start, count),
        }
    }
    pub(super) fn cache_stats(&self) -> Option<vectordata::CacheStats> {
        match self {
            UnifiedReader::Local(_) => None,
            UnifiedReader::Remote(r) => r.cache_stats(),
        }
    }
    pub(super) fn is_remote(&self) -> bool {
        matches!(self, UnifiedReader::Remote(_))
    }
}

/// Resolve a source specifier to a local file path.
///
/// For local files, returns the path directly.
/// For catalog specifiers, this function should NOT be used —
/// use `open_dataset()` instead for on-demand remote access.
pub(super) fn resolve_source(source: &str) -> PathBuf {
    let as_path = std::path::Path::new(source);
    if as_path.exists() {
        return as_path.to_path_buf();
    }

    if source.contains('/') || source.contains('.') {
        eprintln!("Error: file not found: {}", source);
        std::process::exit(1);
    }

    // Parse as dataset:profile[:facet]
    let parts: Vec<&str> = source.split(':').collect();
    let (dataset_name, profile_name, facet_name) = match parts.len() {
        1 => (parts[0], "default", "base_vectors"),
        2 => (parts[0], parts[1], "base_vectors"),
        3 => (parts[0], parts[1], parts[2]),
        _ => {
            eprintln!("Error: invalid source specifier '{}'. Use dataset:profile[:facet]", source);
            std::process::exit(1);
        }
    };

    // Resolve via catalog
    let sources = crate::catalog::sources::CatalogSources::new().configure_default();
    if sources.is_empty() {
        eprintln!("Error: '{}' is not a local file and no catalogs are configured.", source);
        eprintln!("Create ~/.config/vectordata/catalogs.yaml or use a local file path.");
        std::process::exit(1);
    }

    let catalog = crate::catalog::resolver::Catalog::of(&sources);
    let entry = match catalog.find_exact(dataset_name) {
        Some(e) => e,
        None => {
            eprintln!("Error: dataset '{}' not found in catalog.", dataset_name);
            catalog.list_datasets(dataset_name);
            std::process::exit(1);
        }
    };

    let profile = match entry.layout.profiles.profile(profile_name) {
        Some(p) => p,
        None => {
            eprintln!("Error: profile '{}' not found in '{}'. Available: {}",
                profile_name, entry.name, entry.profile_names().join(", "));
            std::process::exit(1);
        }
    };

    let view = match profile.view(facet_name) {
        Some(v) => v,
        None => {
            eprintln!("Error: facet '{}' not found in {}:{}. Available: {}",
                facet_name, entry.name, profile_name, profile.view_names().join(", "));
            std::process::exit(1);
        }
    };

    let source_path = &view.source.path;
    let base_url = entry.path.rsplit_once('/').map(|(base, _)| base).unwrap_or("");

    // Check local cache first
    let cache_dir = dirs_cache_dir().join(&entry.name);
    let cached_path = cache_dir.join(source_path);

    if cached_path.exists() {
        eprintln!("Using cached: {}", crate::check::rel_display(&cached_path));
        return cached_path;
    }

    // Download to cache
    let full_url = if source_path.starts_with("http://") || source_path.starts_with("https://") {
        source_path.clone()
    } else {
        format!("{}/{}", base_url, source_path)
    };

    if !full_url.starts_with("http://") && !full_url.starts_with("https://") {
        // Local source relative to catalog entry
        let local_src = std::path::Path::new(&full_url);
        if local_src.exists() {
            return local_src.to_path_buf();
        }
        eprintln!("Error: source not available: {}", full_url);
        std::process::exit(1);
    }

    eprintln!("Downloading {}:{} ({})...", entry.name, profile_name, facet_name);
    if let Some(parent) = cached_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    match crate::datasets::prebuffer::download_file(&full_url, &cached_path) {
        Ok(size) => {
            eprintln!("Downloaded {} bytes to {}", size, crate::check::rel_display(&cached_path));
            cached_path
        }
        Err(e) => {
            eprintln!("Error: failed to download {}: {}", full_url, e);
            std::process::exit(1);
        }
    }
}

/// Get the configured vectordata cache directory from settings.yaml.
pub(super) fn dirs_cache_dir() -> PathBuf {
    crate::pipeline::commands::config::configured_cache_dir()
}

/// Format-agnostic vector reader that returns f64 values.
///
/// Wraps any supported vector format (fvec, mvec, dvec) and converts
/// individual vectors to f64 on read. No bulk file conversion needed.
pub(super) enum AnyVectorReader {
    F32(vectordata::io::XvecReader<f32>),
    F16(vectordata::io::XvecReader<half::f16>),
}

impl AnyVectorReader {
    /// Open a vector file, auto-detecting format from extension.
    pub(super) fn open(path: &std::path::Path) -> Self {
        use vectordata::io::XvecReader;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match ext {
            "fvec" | "fvecs" => {
                AnyVectorReader::F32(XvecReader::<f32>::open_path(path).unwrap_or_else(|e| {
                    eprintln!("Error: failed to open {}: {}", path.display(), e);
                    std::process::exit(1);
                }))
            }
            "mvec" | "mvecs" => {
                AnyVectorReader::F16(XvecReader::<half::f16>::open_path(path).unwrap_or_else(|e| {
                    eprintln!("Error: failed to open {}: {}", path.display(), e);
                    std::process::exit(1);
                }))
            }
            _ => {
                eprintln!("Error: unsupported format '.{}' for visualization. Use fvec or mvec.", ext);
                std::process::exit(1);
            }
        }
    }

    pub(super) fn count(&self) -> usize {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => <vectordata::io::XvecReader<f32> as VectorReader<f32>>::count(r),
            AnyVectorReader::F16(r) => <vectordata::io::XvecReader<half::f16> as VectorReader<half::f16>>::count(r),
        }
    }

    pub(super) fn dim(&self) -> usize {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => <vectordata::io::XvecReader<f32> as VectorReader<f32>>::dim(r),
            AnyVectorReader::F16(r) => <vectordata::io::XvecReader<half::f16> as VectorReader<half::f16>>::dim(r),
        }
    }

    /// Read a single vector as f64 values. Converts from native format on the fly.
    pub(super) fn get_f64(&self, index: usize) -> Option<Vec<f64>> {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => {
                r.get(index).ok().map(|v| v.iter().map(|&x| x as f64).collect())
            }
            AnyVectorReader::F16(r) => {
                r.get(index).ok().map(|v| v.iter().map(|x| x.to_f64()).collect())
            }
        }
    }

    /// Read a single vector as f32 values (for simsimd hot paths).
    pub(super) fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => {
                r.get(index).ok().map(|v| v.to_vec())
            }
            AnyVectorReader::F16(r) => {
                r.get(index).ok().map(|v| v.iter().map(|x| x.to_f32()).collect())
            }
        }
    }

    /// Range read of `count` vectors as f64. mmap-backed, so it just
    /// loops single-vector reads — local files are I/O-cheap and the
    /// trait-level batch path exists primarily to amortize remote
    /// (HTTP-cached) chunk fetches.
    pub(super) fn get_f64_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f64>>> {
        (0..count).map(|i| self.get_f64(start + i)).collect()
    }
    pub(super) fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        (0..count).map(|i| self.get_f32(start + i)).collect()
    }

}

/// Wrapper that opens `base_vectors` (and exposes a cache-stats view)
/// through the canonical [`vectordata::TestDataView`] path.
pub(super) struct AnyDatasetReader {
    view: std::sync::Arc<dyn vectordata::TestDataView>,
    base: std::sync::Arc<dyn vectordata::VectorReader<f32>>,
    facet_storage: Option<vectordata::FacetStorage>,
    dim: usize,
    count: usize,
}

impl AnyDatasetReader {
    pub(super) fn from_source(source: &str) -> Self {
        let view = open_dataset_view(source);
        let base = view.base_vectors().unwrap_or_else(|e| {
            eprintln!("Error: failed to open base_vectors: {e}");
            std::process::exit(1);
        });
        let dim = base.dim();
        let count = base.count();
        let facet_storage = view.open_facet_storage("base_vectors").ok();
        AnyDatasetReader { view, base, facet_storage, dim, count }
    }

    pub(super) fn count(&self) -> usize { self.count }
    pub(super) fn dim(&self) -> usize { self.dim }

    pub(super) fn get_f64(&self, index: usize) -> Option<Vec<f64>> {
        self.base.get(index).ok().map(|v| v.into_iter().map(|x| x as f64).collect())
    }
    pub(super) fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        self.base.get(index).ok()
    }
    pub(super) fn get_f64_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f64>>> {
        (0..count).map(|i| self.get_f64(start + i)).collect()
    }
    pub(super) fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        (0..count).map(|i| self.get_f32(start + i)).collect()
    }

    pub(super) fn cache_stats(&self) -> Option<vectordata::CacheStats> {
        self.facet_storage.as_ref()?.cache_stats()
    }

    /// Suppress unused-field warning — the view handle keeps the
    /// underlying dataset alive for the lifetime of `base`/`facet_storage`.
    #[allow(dead_code)]
    pub(super) fn view(&self) -> &dyn vectordata::TestDataView { &*self.view }
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
            let mut rng = crate::pipeline::rng::seeded_rng(seed);
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
    // First: register the cooperative flag
    let _ = signal_hook::flag::register(signal_hook::consts::SIGINT, flag.clone());

    // Second: register conditional default — if flag already set (second Ctrl-C),
    // the OS default handler runs (immediate termination)
    let _ = signal_hook::flag::register_conditional_default(
        signal_hook::consts::SIGINT,
        flag,
    );
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

/// Compute axis bounds with 5% padding.
pub(super) fn compute_bounds(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let x_pad = (x_max - x_min) * 0.05;
    let y_pad = (y_max - y_min) * 0.05;
    (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
}

/// Cache file for PCA results: stores mean, eigenvectors, eigenvalues,
/// and projected points so recomputation is avoided.
pub(super) const PCA_CACHE_FILE: &str = "pca_projection.bin";

/// Partition size for incremental PCA (number of vectors per partition).
pub(super) const PCA_PARTITION_SIZE: usize = 1_000_000;

/// Magic bytes for 5D PCA cache format.
pub(super) const PCA_5D_MAGIC: &[u8; 4] = b"PC5D";

/// Check if cache file is newer than the source file.
pub(super) fn is_cache_fresh(cache: &std::path::Path, source: &std::path::Path) -> bool {
    let cache_mtime = std::fs::metadata(cache).ok().and_then(|m| m.modified().ok());
    let source_mtime = std::fs::metadata(source).ok().and_then(|m| m.modified().ok());
    match (cache_mtime, source_mtime) {
        (Some(c), Some(s)) => c > s,
        _ => false,
    }
}

/// Save 5D PCA projection to cache.
pub(super) fn save_pca_cache(
    path: &std::path::Path,
    projected: &[[f64; 5]],
    eigenvalues: &[f64],
    dim: usize,
    total: usize,
    filename: &str,
) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(PCA_5D_MAGIC).map_err(|e| e.to_string())?;
    let fname_bytes = filename.as_bytes();
    f.write_all(&(dim as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(total as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(fname_bytes.len() as u16).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(fname_bytes).map_err(|e| e.to_string())?;
    // Write 10 eigenvalues (padded with zeros if fewer)
    for i in 0..10 {
        let ev = eigenvalues.get(i).copied().unwrap_or(0.0);
        f.write_all(&ev.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    f.write_all(&(projected.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    for p in projected {
        for &v in p {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Load 5D PCA projection from cache.
pub(super) fn load_pca_5d_cache(path: &std::path::Path) -> Option<(Vec<[f64; 5]>, Vec<f64>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];
    let mut buf2 = [0u8; 2];

    f.read_exact(&mut buf4).ok()?;
    if &buf4 != PCA_5D_MAGIC {
        return None; // Old format — trigger recomputation
    }

    f.read_exact(&mut buf4).ok()?; // dim
    f.read_exact(&mut buf8).ok()?; // total
    f.read_exact(&mut buf2).ok()?;
    let fname_len = u16::from_le_bytes(buf2) as usize;
    let mut fname_buf = vec![0u8; fname_len];
    f.read_exact(&mut fname_buf).ok()?;

    let mut eigenvalues = Vec::with_capacity(10);
    for _ in 0..10 {
        f.read_exact(&mut buf8).ok()?;
        eigenvalues.push(f64::from_le_bytes(buf8));
    }

    f.read_exact(&mut buf8).ok()?;
    let n_points = u64::from_le_bytes(buf8) as usize;
    if n_points > 100_000_000 { return None; }

    let mut projected = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        let mut p = [0.0f64; 5];
        for v in &mut p {
            f.read_exact(&mut buf8).ok()?;
            *v = f64::from_le_bytes(buf8);
        }
        projected.push(p);
    }

    Some((projected, eigenvalues))
}

/// Per-partition statistics for incremental PCA.
pub(super) struct PartitionStats {
    pub(super) count: usize,
    pub(super) mean: Vec<f64>,
}

/// Save partition statistics to a binary cache file.
pub(super) fn save_partition_cache(path: &std::path::Path, stats: &PartitionStats) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(&(stats.count as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(stats.mean.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    for &v in &stats.mean {
        f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Load partition statistics from a binary cache file.
pub(super) fn load_partition_cache(path: &std::path::Path) -> Option<PartitionStats> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];

    f.read_exact(&mut buf8).ok()?;
    let count = u64::from_le_bytes(buf8) as usize;

    f.read_exact(&mut buf4).ok()?;
    let dim = u32::from_le_bytes(buf4) as usize;

    let mut mean = vec![0.0f64; dim];
    for i in 0..dim {
        f.read_exact(&mut buf8).ok()?;
        mean[i] = f64::from_le_bytes(buf8);
    }

    Some(PartitionStats { count, mean })
}

/// Compute the mean vector across sampled indices.
#[allow(dead_code)]
pub(super) fn compute_mean(
    reader: &UnifiedReader,
    indices: &[usize],
    dim: usize,
) -> Vec<f64> {
    let n = indices.len();

    let partial_sums: Vec<Vec<f64>> = indices
        .chunks(4096)
        .map(|chunk| {
            let mut sum = vec![0.0f64; dim];
            for &i in chunk {
                if let Some(v) = reader.get_f64(i) {
                    for d in 0..dim {
                        sum[d] += v[d];
                    }
                }
            }
            sum
        })
        .collect();

    let mut mean = vec![0.0f64; dim];
    for ps in &partial_sums {
        for d in 0..dim {
            mean[d] += ps[d];
        }
    }
    for d in 0..dim {
        mean[d] /= n as f64;
    }
    mean
}

/// Compute top-k eigenvectors of the covariance matrix using power iteration.
///
/// For each eigenvector:
/// 1. Start with a random vector
/// 2. Repeatedly multiply by the covariance matrix (implicitly, via the data)
/// 3. Normalize
/// 4. Deflate the covariance by subtracting the found eigenvector's contribution
///
/// The covariance-vector product is computed implicitly:
///   Cv = (1/n) Σ (x_i - μ)(x_i - μ)ᵀ v = (1/n) Σ (x_i - μ) · ((x_i - μ)ᵀ v)
/// This avoids materializing the d×d covariance matrix.
#[allow(dead_code)]
pub(super) fn compute_top_eigenvectors(
    reader: &UnifiedReader,
    indices: &[usize],
    mean: &[f64],
    dim: usize,
    k: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    use simsimd::SpatialSimilarity;

    let n = indices.len();
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut eigenvalues: Vec<f64> = Vec::with_capacity(k);

    // Pre-compute f32 mean for simsimd dot products
    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();

    for _ki in 0..k {
        // Initialize with a deterministic pseudo-random vector
        let mut v: Vec<f64> = (0..dim).map(|d| ((d * 7 + 13) % 97) as f64 - 48.0).collect();
        normalize(&mut v);

        // Power iteration: 30 iterations is more than enough for convergence
        for _iter in 0..30 {
            // Implicit covariance-vector product: Cv = (1/n) Σ (x-μ)((x-μ)·v)
            // Use simsimd for the (x-μ)·v dot product in the inner loop.
            let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            let partial_products: Vec<Vec<f64>> = indices
                .chunks(4096)
                .map(|chunk| {
                    let mut product = vec![0.0f64; dim];
                    let mut centered = vec![0.0f32; dim];
                    for &i in chunk {
                        if let Some(x) = reader.get_f32(i) {
                            // Center: centered = x - μ
                            for d in 0..dim {
                                centered[d] = x[d] - mean_f32[d];
                            }
                            // SIMD dot product: (x-μ)·v
                            let dot = <f32 as SpatialSimilarity>::dot(&centered, &v_f32)
                                .unwrap_or(0.0) as f64;
                            // Accumulate outer product contribution in f64
                            for d in 0..dim {
                                product[d] += centered[d] as f64 * dot;
                            }
                        }
                    }
                    product
                })
                .collect();

            // Merge partial products
            let mut new_v = vec![0.0f64; dim];
            for pp in &partial_products {
                for d in 0..dim {
                    new_v[d] += pp[d];
                }
            }
            for d in 0..dim {
                new_v[d] /= n as f64;
            }

            // Deflate: remove components along previously found eigenvectors
            for prev in &eigenvectors {
                let proj: f64 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for d in 0..dim {
                    new_v[d] -= proj * prev[d];
                }
            }

            normalize(&mut new_v);
            v = new_v;
        }

        // Compute eigenvalue: λ = vᵀ C v (via the same implicit product)
        let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let eigenvalue_sum: f64 = indices
            .chunks(4096)
            .map(|chunk| {
                let mut sum = 0.0f64;
                let mut centered = vec![0.0f32; dim];
                for &i in chunk {
                    if let Some(x) = reader.get_f32(i) {
                        for d in 0..dim {
                            centered[d] = x[d] - mean_f32[d];
                        }
                        let dot = <f32 as SpatialSimilarity>::dot(&centered, &v_f32)
                            .unwrap_or(0.0) as f64;
                        sum += dot * dot;
                    }
                }
                sum
            })
            .sum();

        let eigenvalue = eigenvalue_sum / n as f64;
        eigenvalues.push(eigenvalue);
        eigenvectors.push(v);
    }

    (eigenvectors, eigenvalues)
}

/// Project sampled vectors onto the top-k eigenvectors.
///
/// Uses simsimd for the dot products: (x - μ) · eigenvector.
#[allow(dead_code)]
pub(super) fn project_vectors(
    reader: &UnifiedReader,
    indices: &[usize],
    mean: &[f64],
    eigenvectors: &[Vec<f64>],
    dim: usize,
) -> Vec<(f64, f64)> {
    use simsimd::SpatialSimilarity;

    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
    let ev0_f32: Vec<f32> = eigenvectors[0].iter().map(|&x| x as f32).collect();
    let ev1_f32: Vec<f32> = eigenvectors[1].iter().map(|&x| x as f32).collect();

    let mut centered = vec![0.0f32; dim];
    indices
        .iter()
        .filter_map(|&i| {
            let x = reader.get_f32(i)?;
            for d in 0..dim {
                centered[d] = x[d] - mean_f32[d];
            }
            let pc1 = <f32 as SpatialSimilarity>::dot(&centered, &ev0_f32)
                .unwrap_or(0.0) as f64;
            let pc2 = <f32 as SpatialSimilarity>::dot(&centered, &ev1_f32)
                .unwrap_or(0.0) as f64;
            Some((pc1, pc2))
        })
        .collect()
}

/// Project sampled vectors onto the top-3 eigenvectors (3D).
#[allow(dead_code)]
pub(super) fn project_vectors_3d(
    reader: &UnifiedReader,
    indices: &[usize],
    mean: &[f64],
    eigenvectors: &[Vec<f64>],
    dim: usize,
) -> Vec<[f64; 3]> {
    use simsimd::SpatialSimilarity;

    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
    let ev0_f32: Vec<f32> = eigenvectors[0].iter().map(|&x| x as f32).collect();
    let ev1_f32: Vec<f32> = eigenvectors[1].iter().map(|&x| x as f32).collect();
    let ev2_f32: Vec<f32> = if eigenvectors.len() > 2 {
        eigenvectors[2].iter().map(|&x| x as f32).collect()
    } else {
        vec![0.0f32; dim]
    };

    let mut centered = vec![0.0f32; dim];
    indices
        .iter()
        .filter_map(|&i| {
            let x = reader.get_f32(i)?;
            for d in 0..dim {
                centered[d] = x[d] - mean_f32[d];
            }
            let pc1 = <f32 as SpatialSimilarity>::dot(&centered, &ev0_f32).unwrap_or(0.0) as f64;
            let pc2 = <f32 as SpatialSimilarity>::dot(&centered, &ev1_f32).unwrap_or(0.0) as f64;
            let pc3 = <f32 as SpatialSimilarity>::dot(&centered, &ev2_f32).unwrap_or(0.0) as f64;
            Some([pc1, pc2, pc3])
        })
        .collect()
}
