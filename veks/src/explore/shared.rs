// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for explore subcommands: data access, sampling,
//! caching, and common utilities.

use std::ffi::OsStr;
use std::path::PathBuf;

use clap_complete::engine::CompletionCandidate;

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

/// Recognized data file extensions for explore commands.
pub(super) const DATA_EXTENSIONS: &[&str] = &[
    "fvec", "fvecs", "ivec", "ivecs", "mvec", "mvecs",
    "bvec", "bvecs", "dvec", "dvecs", "svec", "svecs",
    "npy", "slab", "parquet",
];

/// Completion for the source argument: local data files + catalog datasets.
/// Dataset-only completer (three-tier, no local files).
///
/// Used by `--dataset` which constrains to catalog entries only.
pub(crate) fn dataset_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let prefix = current.to_string_lossy();
    let tier = source_completion_tier();

    let mut candidates = Vec::new();

    // Tier 1: cached datasets only
    candidates.extend(cached_dataset_candidates(&prefix));

    // Tier 2: all dataset names
    if tier >= 2 {
        candidates.extend(catalog_name_candidates(&prefix));
    }

    // Tier 3: dataset:profile combinations
    if tier >= 3 {
        candidates.extend(catalog_profile_candidates(&prefix));
    }

    candidates
}

/// Three-tier dataset source completion:
///
/// - **Tier 1** (first tab): local files + locally cached datasets
/// - **Tier 2** (second tab): + all catalog dataset names (no profiles)
/// - **Tier 3** (third tab): + all dataset:profile combinations
///
/// Uses a tap-count file to track repeated completions for the same input.
pub(crate) fn source_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let prefix = current.to_string_lossy();
    let tier = source_completion_tier();

    let mut candidates = Vec::new();

    // Tier 1: local files + cached datasets (always shown)
    candidates.extend(local_file_candidates(&prefix));
    candidates.extend(cached_dataset_candidates(&prefix));

    // Tier 2: add all dataset names (no profiles)
    if tier >= 2 {
        candidates.extend(catalog_name_candidates(&prefix));
    }

    // Tier 3: add dataset:profile combinations
    if tier >= 3 {
        candidates.extend(catalog_profile_candidates(&prefix));
    }

    candidates
}

const SOURCE_TAP_FILE: &str = "/tmp/.veks_source_tab";
const SOURCE_TAP_MAX_AGE: u64 = 10;

/// Determine which completion tier to show based on repeated tab presses.
/// Returns 1, 2, or 3.
fn source_completion_tier() -> usize {
    use std::time::SystemTime;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let state_key: String = std::env::args().skip(1).collect::<Vec<_>>().join("\x00");

    let (prev_key, prev_count, prev_time) = std::fs::read_to_string(SOURCE_TAP_FILE)
        .ok()
        .and_then(|content| {
            let mut lines = content.lines();
            let count: usize = lines.next()?.parse().ok()?;
            let timestamp: u64 = lines.next()?.parse().ok()?;
            let key = lines.next().unwrap_or("").to_string();
            Some((key, count, timestamp))
        })
        .unwrap_or_default();

    let is_stale = now.saturating_sub(prev_time) > SOURCE_TAP_MAX_AGE;

    let count = if state_key == prev_key && !is_stale {
        prev_count + 1
    } else {
        1
    };

    let _ = std::fs::write(SOURCE_TAP_FILE, format!("{}\n{}\n{}", count, now, state_key));

    // Shells invoke completion twice per Tab (compute + display).
    // count 1-2 = tier 1 (first tab), 3-4 = tier 2 (second tab), 5+ = tier 3
    if count >= 5 {
        let _ = std::fs::write(SOURCE_TAP_FILE, ""); // reset
        3
    } else if count >= 3 {
        2
    } else {
        1
    }
}

/// Find local files with recognized data extensions matching the prefix.
fn local_file_candidates(prefix: &str) -> Vec<CompletionCandidate> {
    let mut candidates = Vec::new();

    // Determine directory to scan and filename prefix
    let path = std::path::Path::new(prefix);
    let (dir, file_prefix) = if prefix.ends_with('/') || prefix.ends_with(std::path::MAIN_SEPARATOR) {
        (path.to_path_buf(), String::new())
    } else if let Some(parent) = path.parent() {
        let dir = if parent.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            parent.to_path_buf()
        };
        let fname = path.file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        (dir, fname)
    } else {
        (PathBuf::from("."), prefix.to_string())
    };

    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip hidden files
            if name_str.starts_with('.') {
                continue;
            }

            // Include directories (for navigation) and data files
            let entry_path = entry.path();
            if entry_path.is_dir() {
                if name_str.starts_with(&*file_prefix) {
                    let display = if dir == PathBuf::from(".") {
                        format!("{}/", name_str)
                    } else {
                        format!("{}/{}/", dir.display(), name_str)
                    };
                    candidates.push(CompletionCandidate::new(display));
                }
            } else if name_str.starts_with(&*file_prefix) {
                let ext = entry_path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if DATA_EXTENSIONS.contains(&ext) {
                    let display = if dir == PathBuf::from(".") {
                        name_str.to_string()
                    } else {
                        format!("{}/{}", dir.display(), name_str)
                    };
                    candidates.push(
                        CompletionCandidate::new(display)
                            .help(Some(ext.to_string().into()))
                    );
                }
            }
        }
    }

    candidates
}

/// Tier 1 catalog: datasets that exist in the local cache directory.
fn cached_dataset_candidates(prefix: &str) -> Vec<CompletionCandidate> {
    if prefix.contains('/') || prefix.contains('.') {
        return Vec::new();
    }

    let cache_dir = crate::pipeline::commands::config::configured_cache_dir();
    let mut candidates = Vec::new();
    let prefix_lower = prefix.to_lowercase();

    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.to_lowercase().starts_with(&prefix_lower) && entry.path().is_dir() {
                candidates.push(
                    CompletionCandidate::new(name_str.to_string())
                        .help(Some("cached".into()))
                );
            }
        }
    }

    candidates
}

/// Tier 2 catalog: all dataset names without profiles.
fn catalog_name_candidates(prefix: &str) -> Vec<CompletionCandidate> {
    if prefix.contains('/') || prefix.contains('.') {
        return Vec::new();
    }

    let prefix_lower = prefix.to_lowercase();
    let entries = crate::datasets::filter::completion_entries();
    let mut candidates = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for entry in entries {
        if entry.name.to_lowercase().starts_with(&prefix_lower) && seen.insert(entry.name.clone()) {
            let help = entry.layout.attributes.as_ref()
                .and_then(|a| a.distance_function.as_deref())
                .unwrap_or("");
            let profiles = entry.profile_names();
            let summary = format!("{} ({} profiles)", help, profiles.len());
            candidates.push(
                CompletionCandidate::new(&entry.name)
                    .help(Some(summary.into()))
            );
        }
    }

    candidates
}

/// Tier 3 catalog: all dataset:profile combinations.
fn catalog_profile_candidates(prefix: &str) -> Vec<CompletionCandidate> {
    if prefix.contains('/') || prefix.contains('.') {
        return Vec::new();
    }

    let prefix_lower = prefix.to_lowercase();
    let entries = crate::datasets::filter::completion_entries();
    let mut candidates = Vec::new();

    for entry in entries {
        let profiles = entry.profile_names();
        for profile in &profiles {
            let candidate = format!("{}:{}", entry.name, profile);
            if candidate.to_lowercase().starts_with(&prefix_lower)
                || entry.name.to_lowercase().starts_with(&prefix_lower)
            {
                let help = entry.layout.attributes.as_ref()
                    .and_then(|a| a.distance_function.as_deref())
                    .unwrap_or("");
                candidates.push(
                    CompletionCandidate::new(&candidate)
                        .help(Some(help.to_string().into()))
                );
            }
        }
    }

    candidates
}

/// Open a data source via the data access layer.
///
/// Returns a `LoadedDataset` that owns all the views. The caller accesses
/// views via `dataset.base_vectors()` etc.
pub(super) fn open_dataset(source: &str) -> vectordata::dataset::loader::LoadedDataset {
    use vectordata::dataset::loader::DatasetLoader;

    let cache_dir = crate::pipeline::commands::config::configured_cache_dir();
    let loader = DatasetLoader::new(cache_dir);

    // Load catalog entries for catalog specifier resolution
    let catalog_entries = {
        let sources = crate::catalog::sources::CatalogSources::new().configure_default();
        let catalog = crate::catalog::resolver::Catalog::of(&sources);
        catalog.datasets().to_vec()
    };

    loader.load(source, &catalog_entries).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
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
            let dataset = open_dataset(source);
            UnifiedReader::Remote(AnyDatasetReader(dataset))
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
    pub(super) fn cache_stats(&self) -> Option<vectordata::dataset::view::CacheStats> {
        match self {
            UnifiedReader::Local(_) => None,
            UnifiedReader::Remote(r) => r.0.base_vectors()?.cache_stats(),
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
    F32(vectordata::io::MmapVectorReader<f32>),
    F16(vectordata::io::MmapVectorReader<half::f16>),
}

impl AnyVectorReader {
    /// Open a vector file, auto-detecting format from extension.
    pub(super) fn open(path: &std::path::Path) -> Self {
        use vectordata::io::MmapVectorReader;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "fvec" | "fvecs" => {
                AnyVectorReader::F32(MmapVectorReader::<f32>::open_fvec(path).unwrap_or_else(|e| {
                    eprintln!("Error: failed to open {}: {}", path.display(), e);
                    std::process::exit(1);
                }))
            }
            "mvec" | "mvecs" => {
                AnyVectorReader::F16(MmapVectorReader::<half::f16>::open_mvec(path).unwrap_or_else(|e| {
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
            AnyVectorReader::F32(r) => <vectordata::io::MmapVectorReader<f32> as VectorReader<f32>>::count(r),
            AnyVectorReader::F16(r) => <vectordata::io::MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(r),
        }
    }

    pub(super) fn dim(&self) -> usize {
        use vectordata::VectorReader;
        match self {
            AnyVectorReader::F32(r) => <vectordata::io::MmapVectorReader<f32> as VectorReader<f32>>::dim(r),
            AnyVectorReader::F16(r) => <vectordata::io::MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(r),
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

    /// Create from a LoadedDataset's base_vectors view.
    ///
    /// Wraps the dataset so the AnyVectorReader can be used uniformly
    /// for both local files and remote catalog datasets.
    pub(super) fn from_dataset(dataset: vectordata::dataset::loader::LoadedDataset) -> AnyDatasetReader {
        AnyDatasetReader(dataset)
    }
}

/// Wrapper around LoadedDataset that provides the same interface as AnyVectorReader.
pub(super) struct AnyDatasetReader(pub(super) vectordata::dataset::loader::LoadedDataset);

impl AnyDatasetReader {
    pub(super) fn count(&self) -> usize {
        self.0.base_vectors().map(|v| v.count()).unwrap_or(0)
    }
    pub(super) fn dim(&self) -> usize {
        self.0.base_vectors().map(|v| v.dim()).unwrap_or(0)
    }
    pub(super) fn get_f64(&self, index: usize) -> Option<Vec<f64>> {
        self.0.base_vectors()?.get_f64(index)
    }
    pub(super) fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        self.0.base_vectors()?.get_f32(index)
    }
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
