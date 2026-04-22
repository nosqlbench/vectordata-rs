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
pub const CACHE_VERSION: &str = "v1";

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
pub(super) fn write_segment_cache(
    ivec_path: &Path,
    fvec_path: &Path,
    per_query: &[Vec<Neighbor>],
    k: usize,
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
                (row[j].index as i32, row[j].distance)
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
pub(super) fn load_segment_cache(
    ivec_path: &Path,
    fvec_path: &Path,
    k: usize,
    query_count: usize,
    flip_sign: bool,
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
            let dist = if flip_sign { -raw } else { raw };
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
pub(super) struct CachedSegment {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) ivec_path: PathBuf,
    pub(super) fvec_path: PathBuf,
    /// True when the loaded distances must be sign-flipped to match the
    /// kernel's internal convention. The published profile-dir GT for
    /// IP/DOT/cosine-as-IP stores `+dot` (positive, larger = better)
    /// because that's what users expect, but the kernel ranks with
    /// `-dot` so the standard min-distance heap math works. Path-1
    /// `.cache/` files are written in kernel sign and need no flip;
    /// Path-2 sibling profile files do, when the metric is one of the
    /// inner-product variants.
    pub(super) flip_sign: bool,
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
                    flip_sign: false,
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
                        // Published GT for IP/DOT/cosine-as-IP stores
                        // `+dot` (user-facing convention: larger = better)
                        // while the kernel ranks with `-dot`. Flag the
                        // segment so the loader negates each distance
                        // before merging into the heap. L2 stores `+L2sq`
                        // in both places — no flip needed.
                        let flip_sign = matches!(metric, Metric::DotProduct | Metric::Cosine);
                        segments.push(CachedSegment {
                            start: 0, end: pbc,
                            ivec_path: idx_path,
                            fvec_path: dist_path,
                            flip_sign,
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
