// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared segment-cache infrastructure for brute-force KNN engines.
//!
//! Every full-scan KNN engine in this crate (`knn-stdarch`, `knn-blas`,
//! ...) computes per-query top-K neighbors by scanning a contiguous
//! range of base vectors and merging each segment's contribution into a
//! global per-query heap. This module provides the pieces shared across
//! engines:
//!
//! - [`Metric`] — metric-selection enum with canonical cache tags
//! - [`build_cache_path`], [`validate_cache_file`] — file-naming and
//!   size-based validation for cached segment artifacts
//! - [`write_segment_cache`], [`load_segment_cache`] — write/replay of
//!   a single segment's per-query top-K
//! - [`scan_cached_segments`] — cross-profile cache discovery plus
//!   salvage of sibling profiles' published outputs
//! - [`merge_segment_into_heaps`] — append-with-pruning merge
//!
//! Each engine supplies:
//!   - an `engine` string (`"knn-stdarch"`, `"knn-blas"`, ...) that
//!     distinguishes its cache files on disk. Engines do not share
//!     caches because their f32 accumulation orders differ at ULP
//!     level and the cached distances would disagree.
//!   - a distance kernel (per-vector SIMD loop, BLAS GEMM, ...).

use std::collections::BinaryHeap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use super::compute_knn::Neighbor;

/// Cache-format version. Bump when the segment-compute algorithm
/// changes in a way that would affect stored neighbors or distances.
///
/// v1 → v2: every engine can opt in to top-(k + margin) candidates
/// per query (margin defaults to 0 in production). v1 caches assume
/// `k` entries per row; v2 file rows may contain more.
///
/// v2 → v3: every on-disk fvec — both segment caches and published
/// `neighbor_distances.fvecs` — now uses **FAISS publication
/// convention** uniformly:
///   - `L2`         → `+L2sq`     (positive, smaller = better)
///   - `DotProduct` → `+dot`      (positive, larger  = better)
///   - `Cosine`     → `+cos_sim`  (`[-1, 1]`, larger = better)
/// The kernel still operates on monotonic-distance values internally
/// (smaller = better), but conversion to/from publication convention
/// happens at every disk boundary via [`kernel_to_published`] and
/// [`published_to_kernel`]. v2 cache files were a mix (segment caches
/// in kernel convention, published files in publication convention)
/// and would be misinterpreted under the new uniform convention.
pub const CACHE_VERSION: &str = "v3";

/// Default rerank margin ratio for production `compute knn*` calls.
/// `0` means each engine tracks top-k only — same heap size and
/// pruning aggressiveness as the pre-margin code, no slowdown.
///
/// The canonical f64 rerank post-pass still runs (it just reorders
/// the engine's existing top-k via f64 direct math — cheap), so
/// every engine's output is canonicalized; only the *boundary
/// recovery* is gated by margin.
///
/// Cross-engine parity testing (`verify engine-parity`) opts in to a
/// non-zero margin via `rerank_margin_ratio`/`rerank_margin` options
/// to recover the last 1% of boundary cases that surface only on
/// pathological synthetic distributions (uniform random at very
/// high dim).
pub const RERANK_MARGIN_RATIO_DEFAULT: usize = 0;

/// Resolve the per-call rerank margin from pipeline options. Returns
/// the number of *extra* candidates each engine should keep beyond k.
///
/// Recognized options (first wins):
///   - `rerank_margin`: explicit count of extra candidates (e.g. 30)
///   - `rerank_margin_ratio`: count expressed as a multiplier on k
///     (e.g. 3 ⇒ margin = 3·k ⇒ internal_k = 4·k)
///
/// Defaults to `RERANK_MARGIN_RATIO_DEFAULT * k = 0`.
pub(super) fn rerank_margin_from(opts: &super::super::command::Options, k: usize) -> usize {
    if let Some(m) = opts.get("rerank_margin").and_then(|s| s.parse::<usize>().ok()) {
        return m;
    }
    if let Some(r) = opts.get("rerank_margin_ratio").and_then(|s| s.parse::<usize>().ok()) {
        return r.saturating_mul(k);
    }
    RERANK_MARGIN_RATIO_DEFAULT.saturating_mul(k)
}

/// Returns the per-query candidate count an engine should track:
/// `k + margin`. With `margin = 0` (the production default), the
/// engine reverts to top-k tracking.
pub(super) fn internal_k(k: usize, margin: usize) -> usize {
    k.saturating_add(margin)
}

/// Which distance metric a KNN engine is computing. Filename tokens
/// (`cache_tag`) are stable across releases — changing them
/// invalidates existing caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    DotProduct,
    Cosine,
}

impl Metric {
    pub(super) fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "L2" => Some(Metric::L2),
            "DOT_PRODUCT" | "IP" | "DOT" => Some(Metric::DotProduct),
            "COSINE" => Some(Metric::Cosine),
            _ => None,
        }
    }

    pub(super) fn cache_tag(self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::DotProduct => "dot_product",
            Metric::Cosine => "cosine",
        }
    }

    /// If the caller asserts the input is already unit-normalized,
    /// cosine similarity is exactly the dot product. Collapsing both
    /// to `DotProduct` lets a normalized run share caches with a dot
    /// run on the same vectors.
    pub(super) fn kernel_metric(self, normalized: bool) -> Self {
        if normalized && matches!(self, Metric::Cosine | Metric::DotProduct) {
            Metric::DotProduct
        } else {
            self
        }
    }
}

/// Convert a kernel-internal distance (always smaller = better, used
/// by every per-query heap) to FAISS publication convention (the
/// shape every on-disk fvec uses):
///
///   `L2`         → `+L2sq`     (unchanged — kernel and publication agree)
///   `DotProduct` → `+dot`      (negated; kernel stores `-dot`)
///   `Cosine`     → `+cos_sim`  (mapped from kernel's `1 - cos_sim`)
///
/// `f32::INFINITY` round-trips as `f32::INFINITY` for the
/// padding-sentinel case (rows shorter than k).
#[inline]
pub(super) fn kernel_to_published(d_kernel: f32, metric: Metric) -> f32 {
    if d_kernel.is_infinite() {
        return f32::INFINITY;
    }
    match metric {
        Metric::L2 => d_kernel,
        Metric::DotProduct => -d_kernel,
        Metric::Cosine => 1.0 - d_kernel,
    }
}

/// Inverse of [`kernel_to_published`]: convert an on-disk publication
/// distance back to the kernel-internal monotonic form so it can feed
/// a heap whose ordering is "smaller = better."
#[inline]
pub(super) fn published_to_kernel(d_published: f32, metric: Metric) -> f32 {
    if d_published.is_infinite() {
        return f32::INFINITY;
    }
    match metric {
        Metric::L2 => d_published,
        Metric::DotProduct => -d_published,
        Metric::Cosine => 1.0 - d_published,
    }
}

/// The cosine-handling strategy. One must be chosen explicitly
/// when `metric = COSINE`; L2 and DOT_PRODUCT ignore this.
///
/// The choice matters because:
///  - Assuming normalization (FAISS/numpy/knn_utils convention) lets
///    the kernel use a single dot product per candidate, and makes
///    results bit-compatible with BLAS-based tools on pre-normalized
///    inputs. But if inputs aren't actually normalized, rankings are
///    wrong.
///  - Computing cosine properly (divide dot by `|q| × |b|`) works on
///    any input at the cost of extra norm computations. This is
///    what `numpy.dot(q, b) / (np.linalg.norm(q) * np.linalg.norm(b))`
///    does when you compute cosine manually.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CosineMode {
    /// Treat vectors as already unit-normalized; evaluate cosine as
    /// the inner product. Cheaper; bit-matches BLAS/FAISS reference.
    AssumeNormalized,
    /// Compute cosine in-kernel: divide dot by `|q| × |b|`. Correct
    /// for arbitrary inputs; costs two norm computations per pair.
    ProperMetric,
}

/// Resolve the cosine strategy from pipeline options, erroring when
/// `metric = COSINE` but the caller didn't specify exactly one of the
/// two explicit flags.
///
/// Returns `Ok(None)` for non-cosine metrics (the flags don't apply).
///
/// Back-compat: a lone `normalized = true` (without either new flag)
/// is accepted as `AssumeNormalized`. New pipelines should always
/// write one of the explicit flags.
pub(super) fn resolve_cosine_mode(
    metric: Metric,
    opts: &super::super::command::Options,
) -> Result<Option<CosineMode>, String> {
    resolve_cosine_mode_for(matches!(metric, Metric::Cosine), opts)
}

/// Same as [`resolve_cosine_mode`] but takes an `is_cosine` boolean
/// instead of a `Metric`. Lets engines that carry their own metric
/// enum (like the SimSIMD backend, which also supports L1) share the
/// validator without a type conversion.
pub(super) fn resolve_cosine_mode_for(
    is_cosine: bool,
    opts: &super::super::command::Options,
) -> Result<Option<CosineMode>, String> {
    if !is_cosine {
        return Ok(None);
    }
    let assume = opts.get("assume_normalized_like_faiss").map(|s| s == "true").unwrap_or(false);
    let proper = opts.get("use_proper_cosine_metric").map(|s| s == "true").unwrap_or(false);
    let legacy_normalized = opts.get("normalized").map(|s| s == "true").unwrap_or(false);

    match (assume, proper) {
        (true, true) => Err(
            "metric=COSINE: set exactly one of assume_normalized_like_faiss / use_proper_cosine_metric — both were true".into()
        ),
        (true, false) => Ok(Some(CosineMode::AssumeNormalized)),
        (false, true) => Ok(Some(CosineMode::ProperMetric)),
        (false, false) => {
            if legacy_normalized {
                // Deprecated path: `normalized=true` alone is
                // accepted as AssumeNormalized for back-compat with
                // pre-existing dataset.yaml files. Bootstrap should
                // set the new flags explicitly going forward.
                Ok(Some(CosineMode::AssumeNormalized))
            } else {
                Err(format!(
                    "metric=COSINE requires one of:\n  \
                      assume_normalized_like_faiss=true  (FAISS/numpy convention — vectors are pre-normalized, use inner product)\n  \
                      use_proper_cosine_metric=true      (compute cosine in-kernel from raw vectors)"
                ))
            }
        }
    }
}

/// Derive a cache-key prefix from the base and query file paths.
///
/// Returns `{base_stem}.{query_stem}.{base_size}_{query_size}` — the
/// per-dataset-pair component of the cache path. The full cache
/// filename is built by [`build_cache_path`], which prepends the
/// **engine name** (`knn-metal`, `knn-stdarch`, `knn-blas`, …) so
/// engines never replay each other's output. That engine prefix is
/// the load-bearing differentiator: every `compute knn*`
/// implementation declares a unique single-word name (see each
/// engine's `ENGINE_NAME` constant), and the contract is that two
/// engines with different numerical behavior MUST have different
/// names.
///
/// Same-engine, same-file-paths, same-file-sizes, different-content
/// inputs still collide on this prefix by design: users running
/// repeated experiments on top of the same output paths are
/// responsible for either regenerating through the pipeline (which
/// bumps the pipeline fingerprint), pointing the engine at a fresh
/// workspace, or clearing `<workspace>/.cache`.
pub(super) fn cache_prefix_for(base_path: &Path, query_path: &Path) -> String {
    let base_stem = base_path.file_stem().unwrap_or_default().to_string_lossy();
    let query_stem = query_path.file_stem().unwrap_or_default().to_string_lossy();
    let base_size = std::fs::metadata(base_path).map(|m| m.len()).unwrap_or(0);
    let query_size = std::fs::metadata(query_path).map(|m| m.len()).unwrap_or(0);
    format!("{}.{}.{}_{}", base_stem, query_stem, base_size, query_size)
}

/// Build a cache file path for a segment of the current
/// `(base, query, k, metric)` tuple. `engine` distinguishes engines
/// that compute numerically-different distances.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_cache_path(
    cache_dir: &Path,
    engine: &str,
    cache_prefix: &str,
    start: usize,
    end: usize,
    k: usize,
    metric: Metric,
    suffix: &str,
    ext: &str,
) -> PathBuf {
    cache_dir.join(format!(
        "{}.{}.{}.range_{:012}_{:012}.k{}.{}.{}.{}",
        engine,
        CACHE_VERSION,
        cache_prefix,
        start, end, k,
        metric.cache_tag(),
        suffix,
        ext,
    ))
}

/// A segment cache's two files are each `query_count × (4 + k × elem_size)`
/// bytes — one row per query consisting of a 4-byte dim header plus
/// `k` 4-byte payload values (i32 for indices, f32 for distances).
/// Validate by exact byte-size equality; any mismatch means the cache
/// was written with different settings and must not be loaded.
pub(super) fn validate_cache_file(path: &Path, query_count: usize, k: usize, elem_size: usize) -> bool {
    let expected = query_count as u64 * (4 + k as u64 * elem_size as u64);
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() == expected,
        Err(_) => false,
    }
}

/// Write a segment's per-query top-k contribution to `ivec_path`
/// (neighbor indices) + `fvec_path` (matching distances). The layout
/// matches the final-output format so cache files are directly
/// inspectable with the same readers. Atomic via `.tmp + rename` —
/// a crash mid-write never leaves a half-file that looks valid.
///
/// Distances are converted from kernel-internal convention (the heap's
/// "smaller = better" representation) to FAISS publication convention
/// before writing — that's the single on-disk convention every fvec
/// in the codebase uses (see [`kernel_to_published`] for the mapping).
pub(super) fn write_segment_cache(
    ivec_path: &Path,
    fvec_path: &Path,
    per_query: &[Vec<Neighbor>],
    k: usize,
    metric: Metric,
) -> Result<(), String> {
    use std::io::BufWriter;
    let dim_bytes = (k as i32).to_le_bytes();

    let ivec_tmp = ivec_path.with_extension("tmp");
    let fvec_tmp = fvec_path.with_extension("tmp");

    let ivec_f = std::fs::File::create(&ivec_tmp)
        .map_err(|e| format!("create {}: {}", ivec_tmp.display(), e))?;
    let fvec_f = std::fs::File::create(&fvec_tmp)
        .map_err(|e| format!("create {}: {}", fvec_tmp.display(), e))?;
    let mut iw = BufWriter::with_capacity(1 << 20, ivec_f);
    let mut fw = BufWriter::with_capacity(1 << 20, fvec_f);

    for row in per_query {
        iw.write_all(&dim_bytes).map_err(|e| e.to_string())?;
        fw.write_all(&dim_bytes).map_err(|e| e.to_string())?;
        for j in 0..k {
            let (idx_i32, dist_f32) = if j < row.len() {
                (row[j].index as i32, kernel_to_published(row[j].distance, metric))
            } else {
                (-1i32, f32::INFINITY)
            };
            iw.write_all(&idx_i32.to_le_bytes()).map_err(|e| e.to_string())?;
            fw.write_all(&dist_f32.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    iw.flush().map_err(|e| e.to_string())?;
    fw.flush().map_err(|e| e.to_string())?;
    drop(iw);
    drop(fw);
    std::fs::rename(&ivec_tmp, ivec_path)
        .map_err(|e| format!("rename {} → {}: {}", ivec_tmp.display(), ivec_path.display(), e))?;
    std::fs::rename(&fvec_tmp, fvec_path)
        .map_err(|e| format!("rename {} → {}: {}", fvec_tmp.display(), fvec_path.display(), e))?;
    Ok(())
}

/// Load a previously-written segment cache into a per-query structure.
/// Padding entries (index `-1`, distance `+∞`) are stripped.
///
/// The on-disk fvec is in FAISS publication convention; this loader
/// converts each distance back to kernel-internal convention via
/// [`published_to_kernel`] so the heap-merging code can compare
/// distances uniformly as "smaller = better."
pub(super) fn load_segment_cache(
    ivec_path: &Path,
    fvec_path: &Path,
    k: usize,
    query_count: usize,
    metric: Metric,
) -> Result<Vec<Vec<Neighbor>>, String> {
    use std::io::BufReader;
    let mut iw = BufReader::with_capacity(1 << 20, std::fs::File::open(ivec_path)
        .map_err(|e| format!("open {}: {}", ivec_path.display(), e))?);
    let mut fw = BufReader::with_capacity(1 << 20, std::fs::File::open(fvec_path)
        .map_err(|e| format!("open {}: {}", fvec_path.display(), e))?);

    let mut result: Vec<Vec<Neighbor>> = Vec::with_capacity(query_count);
    let mut hdr = [0u8; 4];
    let mut idx_buf = [0u8; 4];
    let mut dist_buf = [0u8; 4];
    for _ in 0..query_count {
        iw.read_exact(&mut hdr).map_err(|e| format!("read ivec dim: {}", e))?;
        fw.read_exact(&mut hdr).map_err(|e| format!("read fvec dim: {}", e))?;
        let mut row: Vec<Neighbor> = Vec::with_capacity(k);
        for _ in 0..k {
            iw.read_exact(&mut idx_buf).map_err(|e| format!("read ivec body: {}", e))?;
            fw.read_exact(&mut dist_buf).map_err(|e| format!("read fvec body: {}", e))?;
            let idx = i32::from_le_bytes(idx_buf);
            let raw = f32::from_le_bytes(dist_buf);
            let dist = published_to_kernel(raw, metric);
            if idx >= 0 {
                row.push(Neighbor { index: idx as u32, distance: dist });
            }
        }
        result.push(row);
    }
    Ok(result)
}

/// Merge a segment's per-query top-k contribution into the global
/// running heaps, updating the per-query thresholds so downstream
/// segments can prune more aggressively.
pub(super) fn merge_segment_into_heaps(
    segment: &[Vec<Neighbor>],
    all_heaps: &mut [BinaryHeap<Neighbor>],
    all_thresholds: &mut [f32],
    k: usize,
) {
    for (qi, neighbors) in segment.iter().enumerate() {
        for n in neighbors {
            if n.distance < all_thresholds[qi] {
                all_heaps[qi].push(*n);
                if all_heaps[qi].len() > k {
                    all_heaps[qi].pop();
                    all_thresholds[qi] = all_heaps[qi].peek().unwrap().distance;
                }
            }
        }
    }
}

/// One cached segment discovered during cache scan — either an entry
/// written by this engine in `.cache/`, or a salvaged published
/// output from a smaller sibling profile.
///
/// Distance sign convention: every on-disk fvec uses FAISS publication
/// convention (see [`kernel_to_published`]); [`load_segment_cache`]
/// converts back to kernel convention via the metric. Both Path-1 and
/// Path-2 sources are uniform now — there's no per-segment sign flag.
pub(super) struct CachedSegment {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) ivec_path: PathBuf,
    pub(super) fvec_path: PathBuf,
}

/// Scan the cache directory plus sibling profile outputs for every
/// segment whose per-query top-K results could feed this run's KNN.
///
/// Two discovery paths:
///   1. `<cache_dir>/<engine>.<CACHE_VERSION>.<prefix>.range_*.k{k}.{metric}.neighbors.ivec`
///      — segments written by this engine on a prior run.
///   2. `<workspace>/profiles/<size>/neighbor_indices.ivecs` + `..._distances.fvecs`
///      — completed ground truth from a smaller sibling profile. The
///      directory name encodes the profile's base count N, so the
///      file covers `[0..N)` natively. This catches the "100m profile
///      reuses the 50m profile's final output as its [0..50m)
///      segment" case without needing the 50m profile to have been
///      computed through this engine's cache path.
#[allow(clippy::too_many_arguments)]
pub(super) fn scan_cached_segments(
    cache_dir: &Path,
    engine: &str,
    cache_prefix: &str,
    k: usize,
    metric: Metric,
    query_count: usize,
    workspace: &Path,
    base_end: usize,
    ui: &veks_core::ui::UiHandle,
) -> Vec<CachedSegment> {
    let metric_tag = metric.cache_tag();
    let prefix_pat = format!("{}.{}.{}.range_", engine, CACHE_VERSION, cache_prefix);
    let suffix_pat = format!(".k{}.{}.neighbors.ivec", k, metric_tag);

    let mut segments: Vec<CachedSegment> = Vec::new();

    // ── Path 1: scan <cache_dir>/ for segments written by this engine
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with(&prefix_pat) { continue; }
            if !name_str.ends_with(&suffix_pat) { continue; }

            let after_prefix = &name_str[prefix_pat.len()..];
            let range_end = after_prefix.find('.').unwrap_or(after_prefix.len());
            let range_str = &after_prefix[..range_end];
            let (s_str, e_str) = match range_str.split_once('_') {
                Some(pair) => pair,
                None => continue,
            };
            let (s, e) = match (s_str.parse::<usize>(), e_str.parse::<usize>()) {
                (Ok(a), Ok(b)) if b > a => (a, b),
                _ => continue,
            };

            let ivec = build_cache_path(cache_dir, engine, cache_prefix, s, e, k, metric, "neighbors", "ivec");
            let fvec = build_cache_path(cache_dir, engine, cache_prefix, s, e, k, metric, "distances", "fvec");
            if validate_cache_file(&ivec, query_count, k, 4)
                && validate_cache_file(&fvec, query_count, k, 4)
            {
                segments.push(CachedSegment {
                    start: s, end: e, ivec_path: ivec, fvec_path: fvec,
                });
            }
        }
    }

    // ── Path 2: scan <workspace>/profiles/ for smaller-profile outputs
    let profiles_dir = workspace.join("profiles");
    if profiles_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&profiles_dir) {
            for entry in entries.flatten() {
                if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let pname = entry.file_name();
                let pname_str = pname.to_string_lossy();
                if pname_str == "default" { continue; }

                let pbc = match vectordata::dataset::source::parse_number_with_suffix(&pname_str) {
                    Ok(v) => v as usize,
                    Err(_) => continue,
                };
                if pbc == 0 || pbc >= base_end { continue; }

                let idx_path = entry.path().join("neighbor_indices.ivecs");
                let dist_path = entry.path().join("neighbor_distances.fvecs");
                if validate_cache_file(&idx_path, query_count, k, 4)
                    && validate_cache_file(&dist_path, query_count, k, 4)
                {
                    let already = segments.iter().any(|s| s.start == 0 && s.end == pbc);
                    if !already {
                        // All on-disk fvecs use FAISS publication
                        // convention now (see [`kernel_to_published`]).
                        // `load_segment_cache` converts via the metric;
                        // there's no per-source sign flag.
                        segments.push(CachedSegment {
                            start: 0, end: pbc,
                            ivec_path: idx_path,
                            fvec_path: dist_path,
                        });
                    }
                }
            }
        }
    }

    // Sort by start, then by size descending — the greedy matcher
    // picks the largest segment starting at a given position.
    segments.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));

    // One concise summary of what's available. The actual selection
    // (which is at most one segment per starting offset — the
    // largest one that fits) is logged later by the engine as
    // `▸ [seg N/M] REUSED range [X..Y) from FILE`.
    if !segments.is_empty() {
        let mut path1 = 0usize;
        let mut path2 = 0usize;
        let mut largest_path2 = 0usize;
        for s in &segments {
            // Path-1 segments live in `cache_dir`, Path-2 in `<workspace>/profiles/`.
            let from_path2 = s.ivec_path.parent()
                .and_then(|p| p.parent())
                .map(|p| p.ends_with("profiles"))
                .unwrap_or(false);
            if from_path2 {
                path2 += 1;
                largest_path2 = largest_path2.max(s.end);
            } else {
                path1 += 1;
            }
        }
        ui.log(&format!(
            "  cache discovery: {} candidate(s) — {} from .cache/, {} from sibling profiles/ \
             (largest sibling segment ends at {}); planner uses at most one per offset",
            segments.len(), path1, path2, largest_path2,
        ));
    }
    segments
}

/// Re-rank candidate neighbors using f64-direct arithmetic to produce
/// the canonical top-k for a single query.
///
/// Why this is the canonical form: all our sgemm-path engines (BLAS,
/// FAISS, blas-mirror) compute L2² via the expansion
/// `‖a‖² + ‖b‖² − 2·a·b`, which has f32 catastrophic-cancellation
/// risk on concentrated-distance data (curse-of-dim uniform random
/// at high dim, or clustered data with tight within-cluster
/// distances). The direct form `Σ(a−b)²` doesn't have that risk,
/// and promoting the accumulation to f64 removes the residual
/// single-pair f32 rounding too. Using this as the post-sort step
/// across every engine guarantees bit-identical canonical output
/// — the "right answer" that all engines agree on.
///
/// `candidates` should already hold the engine's top-`(k + margin)`
/// guesses. `margin > 0` is what catches boundary misses — if an
/// engine's f32 ranking puts a genuine top-k neighbor just outside
/// its top-k window, it still appears in top-`(k + margin)`, and the
/// f64 rerank elevates it. With `margin = 0` we only fix order
/// within the engine's existing top-k set, which is enough when the
/// engine's native scan was also k-exact (e.g. partitioned scanners
/// that keep every visited candidate within k).
///
/// Result is sorted ascending by canonical distance with the same
/// tiebreaker as [`Neighbor`]'s `Ord` impl (lower index wins on
/// distance tie).
pub(super) fn rerank_topk_f64<F>(
    query: &[f32],
    candidates: &[super::compute_knn::Neighbor],
    k: usize,
    metric: Metric,
    mut get_base: F,
) -> Vec<super::compute_knn::Neighbor>
where
    F: FnMut(u32) -> Option<Vec<f32>>,
{
    let q64: Vec<f64> = query.iter().map(|&x| x as f64).collect();

    // Precompute query-side scalars the metric needs once.
    let (qnorm_sq, qnorm): (f64, f64) = match metric {
        Metric::L2 => (0.0, 0.0), // unused
        Metric::DotProduct => (0.0, 0.0),
        Metric::Cosine => {
            let ns = q64.iter().map(|x| x * x).sum::<f64>();
            (ns, ns.sqrt().max(f64::MIN_POSITIVE))
        }
    };
    let _ = qnorm_sq;

    let mut rescored: Vec<super::compute_knn::Neighbor> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let base = match get_base(c.index) {
            Some(v) => v,
            None => continue,
        };
        if base.len() != query.len() {
            continue;
        }
        let d64: f64 = match metric {
            Metric::L2 => {
                let mut acc = 0.0f64;
                for i in 0..base.len() {
                    let d = q64[i] - base[i] as f64;
                    acc += d * d;
                }
                acc
            }
            Metric::DotProduct => {
                // Engines store -dot as "distance" so smaller is
                // better. Preserve that convention here.
                let mut acc = 0.0f64;
                for i in 0..base.len() {
                    acc += q64[i] * base[i] as f64;
                }
                -acc
            }
            Metric::Cosine => {
                let mut dot = 0.0f64;
                let mut bn = 0.0f64;
                for i in 0..base.len() {
                    dot += q64[i] * base[i] as f64;
                    bn += (base[i] as f64) * (base[i] as f64);
                }
                let denom = qnorm * bn.sqrt().max(f64::MIN_POSITIVE);
                1.0 - dot / denom
            }
        };
        rescored.push(super::compute_knn::Neighbor {
            index: c.index,
            distance: d64 as f32,
        });
    }

    rescored.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    });
    rescored.truncate(k);
    rescored
}

/// Post-pass that rewrites an engine's just-written ivec (and
/// optionally distances fvec) with the canonical f64-direct ranking.
///
/// Why a post-pass and not in-line: every KNN engine has its own
/// scan/merge architecture. A uniform post-pass that operates only
/// on the engine's final top-k output (read back from disk) is the
/// cheapest way to canonicalize every engine without restructuring
/// each one's main loop. Each query becomes
/// O((k + margin) × dim) f64 FMAs — trivial vs the full scan.
///
/// `margin` extra candidates per query are rerank-eligible; pass 0
/// to only reorder within the engine's existing top-k. With
/// `margin > 0`, the input ivec must already hold top-(k + margin)
/// per row, otherwise we just have top-k to work with and the
/// margin is implicitly truncated.
///
/// On error (mismatched dims, base too small, etc.) the original
/// files are left untouched — atomic via `.tmp` + rename.
///
/// Emits a spinner/progress-bar via `ui` so the TUI doesn't go
/// silent on large outputs — the per-query f64 recompute is fast
/// per row but stacks up at high query count × high dim.
pub(super) fn rerank_output_post_pass(
    indices_path: &Path,
    distances_path: Option<&Path>,
    base_reader: &vectordata::io::MmapVectorReader<f32>,
    query_reader: &vectordata::io::MmapVectorReader<f32>,
    metric: Metric,
    k: usize,
    base_offset: usize,
    ui: &veks_core::ui::UiHandle,
) -> Result<(), String> {
    use rayon::prelude::*;
    use std::io::{Read, Write};
    use vectordata::VectorReader;

    // Read the existing indices file. Each row is `[k_header:i32,
    // i32 × k_header]`.
    let mut idx_bytes = Vec::new();
    std::fs::File::open(indices_path)
        .map_err(|e| format!("open {}: {}", indices_path.display(), e))?
        .read_to_end(&mut idx_bytes)
        .map_err(|e| format!("read {}: {}", indices_path.display(), e))?;

    if idx_bytes.is_empty() {
        return Ok(()); // empty file — nothing to rerank
    }

    let header_k = i32::from_le_bytes([idx_bytes[0], idx_bytes[1], idx_bytes[2], idx_bytes[3]]) as usize;
    let row_bytes = 4 + header_k * 4;
    if idx_bytes.len() % row_bytes != 0 {
        return Err(format!("indices file {} has unaligned size", indices_path.display()));
    }
    let n_rows = idx_bytes.len() / row_bytes;
    let n_query = <vectordata::io::MmapVectorReader<f32> as VectorReader<f32>>::count(query_reader);
    if n_rows != n_query {
        return Err(format!(
            "indices file rows={} but query reader has count={}",
            n_rows, n_query));
    }

    ui.log(&format!(
        "  canonical rerank: {} queries × {} candidates, metric={:?}",
        n_rows, header_k, metric,
    ));

    // Parse rows → Vec<Vec<Neighbor>> with f32::INFINITY distance
    // (we don't have the original distances at parse time — they get
    // recomputed in the rerank).
    let mut rows: Vec<Vec<super::compute_knn::Neighbor>> = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let off = r * row_bytes;
        let mut row = Vec::with_capacity(header_k);
        for j in 0..header_k {
            let p = off + 4 + j * 4;
            let idx = i32::from_le_bytes([idx_bytes[p], idx_bytes[p+1], idx_bytes[p+2], idx_bytes[p+3]]);
            if idx >= 0 {
                row.push(super::compute_knn::Neighbor {
                    index: idx as u32,
                    distance: f32::INFINITY, // recomputed below
                });
            }
        }
        rows.push(row);
    }

    // Rerank each query's row. `base_offset` lets engines that
    // store window-relative indices in ivec (like compute_knn.rs
    // metal) still point the rerank at the correct base-reader
    // slot; engines that store absolute indices pass `base_offset = 0`.
    //
    // Parallelized via rayon so large outputs don't serialize on
    // one core. Progress is tracked via a bar updated in batches
    // (every ~1% of rows) so the TUI refresh doesn't become the
    // bottleneck.
    let pb = ui.bar(n_rows as u64, "canonical rerank");
    let progress_tick = (n_rows / 100).max(1);
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let reranked: Vec<Vec<super::compute_knn::Neighbor>> = rows.into_par_iter()
        .enumerate()
        .map(|(qi, cands)| {
            let q = query_reader.get_slice(qi);
            let result = rerank_topk_f64(
                q, &cands, k, metric,
                |idx| Some(base_reader.get_slice(base_offset + idx as usize).to_vec()),
            );
            let done = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % progress_tick == 0 || done == n_rows {
                pb.set_position(done as u64);
            }
            result
        })
        .collect();
    pb.finish();

    // The ivec stores (possibly-window-relative) indices; rerank
    // needs to map those back when writing. The helper already
    // produces Neighbor.index matching the candidate input, so
    // writes the same scheme back out (no shift needed — the index
    // round-trips the caller's scheme).

    // Atomically rewrite the indices file.
    let dim_bytes = (k as i32).to_le_bytes();
    let tmp_idx = indices_path.with_extension("rerank.tmp");
    {
        let mut f = std::fs::File::create(&tmp_idx)
            .map_err(|e| format!("create {}: {}", tmp_idx.display(), e))?;
        for row in &reranked {
            f.write_all(&dim_bytes).map_err(|e| e.to_string())?;
            for j in 0..k {
                let idx = if j < row.len() { row[j].index as i32 } else { -1 };
                f.write_all(&idx.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
    }
    std::fs::rename(&tmp_idx, indices_path)
        .map_err(|e| format!("rename {} → {}: {}", tmp_idx.display(), indices_path.display(), e))?;

    // If the engine emitted distances, rewrite those too with the
    // canonical f64-cast-to-f32 values that pair with the reranked
    // indices.
    //
    // Sign-convention contract for the published fvec — load-bearing
    // for `scan_cached_segments` Path-2 sibling reuse and for the
    // verifier, which both expect the user-facing convention rather
    // than the kernel-internal one:
    //
    //   L2          → `‖a − b‖²`  (positive, smaller = better)
    //   DotProduct  → `+dot`      (positive, larger  = better)
    //   Cosine      → `+cos_sim`  (`[-1, 1]`, larger = better)
    //
    // `rerank_topk_f64` returns kernel-internal distances (always
    // smaller = better, so sorting works uniformly): for DotProduct
    // that's `-dot`, for Cosine that's `1 - cos_sim`. Convert here
    // before writing so the on-disk file matches what compute_knn_blas
    // and friends used to write — otherwise Path-2 sibling segments
    // get loaded with the wrong sign and the merging heap accumulates
    // worst-of-k instead of best-of-k.
    if let Some(dp) = distances_path {
        let tmp_dist = dp.with_extension("rerank.tmp");
        {
            let mut f = std::fs::File::create(&tmp_dist)
                .map_err(|e| format!("create {}: {}", tmp_dist.display(), e))?;
            for row in &reranked {
                f.write_all(&dim_bytes).map_err(|e| e.to_string())?;
                for j in 0..k {
                    let d_kernel = if j < row.len() { row[j].distance } else { f32::INFINITY };
                    let d_published = kernel_to_published(d_kernel, metric);
                    f.write_all(&d_published.to_le_bytes()).map_err(|e| e.to_string())?;
                }
            }
        }
        std::fs::rename(&tmp_dist, dp)
            .map_err(|e| format!("rename {} → {}: {}", tmp_dist.display(), dp.display(), e))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Regression: two engines (different `engine` strings) scanning
    /// the same input files MUST land on different cache paths. This
    /// guards the contract that each `compute knn*` implementation
    /// writes under its own single-word namespace and never replays
    /// another engine's output.
    #[test]
    fn build_cache_path_distinguishes_engines() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("base.fvec");
        let query = tmp.path().join("query.fvec");
        std::fs::File::create(&base).unwrap().write_all(&[0u8; 64]).unwrap();
        std::fs::File::create(&query).unwrap().write_all(&[0u8; 32]).unwrap();

        let prefix = cache_prefix_for(&base, &query);
        let p_metal   = build_cache_path(tmp.path(), "knn-metal",   &prefix, 0, 1000, 10, Metric::L2, "neighbors", "ivec");
        let p_stdarch = build_cache_path(tmp.path(), "knn-stdarch", &prefix, 0, 1000, 10, Metric::L2, "neighbors", "ivec");
        let p_blas    = build_cache_path(tmp.path(), "knn-blas",    &prefix, 0, 1000, 10, Metric::L2, "neighbors", "ivec");

        assert_ne!(p_metal, p_stdarch);
        assert_ne!(p_stdarch, p_blas);
        assert_ne!(p_metal, p_blas);

        // Engine name must be the leading filename component so
        // directory-scan discovery (which filters by engine prefix)
        // can distinguish them.
        assert!(p_metal.file_name().unwrap().to_string_lossy().starts_with("knn-metal."));
        assert!(p_stdarch.file_name().unwrap().to_string_lossy().starts_with("knn-stdarch."));
        assert!(p_blas.file_name().unwrap().to_string_lossy().starts_with("knn-blas."));
    }
}
