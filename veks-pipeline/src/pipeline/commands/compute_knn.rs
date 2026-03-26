// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute exact K-nearest neighbors.
//!
//! Computes ground truth KNN for a set of query vectors against a base vector
//! set. Outputs both neighbor indices (ivec) and distances (fvec).
//!
//! Supports both f32 (fvec) and f16 (mvec) input formats. The element type is
//! detected from the file extension of the base vectors path. When mvec files
//! are used, native f16 SIMD distance kernels operate directly on half-precision
//! data without an explicit upcast to f32.
//!
//! Supports distance metrics: L2 (Euclidean), Cosine, DotProduct, L1 (Manhattan).
//!
//! Uses multi-threaded query processing with a max-heap for top-k selection.
//! Base vectors are mmap'd for O(1) random access.
//!
//! For large base vector sets, supports **partitioned computation**: the base
//! space is split into partitions, each partition's per-query top-K results are
//! cached in `.cache/`, and a merge phase combines partition results into the
//! final global top-K. Cached partitions are reusable across runs with different
//! base-vector ranges.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{BufReader, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use veks_core::ui::ProgressHandle;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use crate::pipeline::simd_distance::{self, Metric};
use super::source_window::resolve_source;

/// Pipeline command: compute exact KNN ground truth.
pub struct ComputeKnnOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeKnnOp)
}

// -- Top-K heap ---------------------------------------------------------------

/// A neighbor candidate with index and distance, ordered by distance descending
/// (max-heap: worst neighbor at top, easily evicted).
#[derive(Clone, Copy, Debug)]
pub(super) struct Neighbor {
    pub(super) index: u32,
    pub(super) distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority (gets evicted first)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Compute KNN for a batch of f32 queries against base vectors `[start, end)`.
///
/// Scans base vectors exactly once per batch, amortizing memory loads.
/// Uses transposed SIMD kernels when available for the given metric.
#[inline(never)]
fn find_top_k_batch_f32(
    queries: &[&[f32]],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    batched_fn: Option<simd_distance::BatchedDistFnF32>,
    metric: Metric,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    if let Some(bfn) = batched_fn {
        find_top_k_batch_transposed_f32(queries, base_reader, start, end, k, bfn, metric, dim, results);
    } else {
        find_top_k_batch_pairwise_f32(queries, base_reader, start, end, k, dist_fn, results);
    }
}

/// Per-pair fallback for f32 batch processing.
#[inline(never)]
fn find_top_k_batch_pairwise_f32(
    queries: &[&[f32]],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    results: &mut [Vec<Neighbor>],
) {
    let batch_size = queries.len();
    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..batch_size)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; batch_size];

    for i in start..end {
        let base_vec = base_reader.get_slice(i);
        let idx = i as u32;

        for qi in 0..batch_size {
            let dist = dist_fn(queries[qi], base_vec);

            if dist < thresholds[qi] {
                heaps[qi].push(Neighbor { index: idx, distance: dist });
                if heaps[qi].len() > k {
                    heaps[qi].pop();
                }
                if heaps[qi].len() == k {
                    thresholds[qi] = heaps[qi].peek().unwrap().distance;
                }
            }
        }
    }

    for (qi, heap) in heaps.into_iter().enumerate() {
        let mut v: Vec<Neighbor> = heap.into_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results[qi] = v;
    }
}

/// Transposed SIMD batch processing for f32.
///
/// Splits queries into sub-batches of [`simd_distance::SIMD_BATCH_WIDTH`] (16),
/// transposes each sub-batch into dimension-major layout, then scans base
/// vectors once computing 16 distances per SIMD FMA instruction.
#[inline(never)]
fn find_top_k_batch_transposed_f32(
    queries: &[&[f32]],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    batched_fn: simd_distance::BatchedDistFnF32,
    metric: Metric,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    use simd_distance::{SIMD_BATCH_WIDTH, TransposedBatch};

    let batch_size = queries.len();
    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..batch_size)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; batch_size];

    // Pre-transpose queries into SIMD-width sub-batches
    let mut sub_batches: Vec<TransposedBatch> = Vec::new();
    let mut sub_offsets: Vec<usize> = Vec::new();
    let mut offset = 0;
    while offset < batch_size {
        let sub_end = std::cmp::min(offset + SIMD_BATCH_WIDTH, batch_size);
        sub_batches.push(TransposedBatch::from_f32(&queries[offset..sub_end], dim));
        sub_offsets.push(offset);
        offset = sub_end;
    }

    // Select dual kernel for paired sub-batch processing (2 FMA ports)
    let dual_fn = simd_distance::select_dual_batched_fn_f32(metric);

    let mut dist_buf_16 = [0.0f32; SIMD_BATCH_WIDTH];
    let mut dist_buf_32 = [0.0f32; 32];

    for i in start..end {
        let base_vec = base_reader.get_slice(i);
        let idx = i as u32;

        // Process sub-batches in pairs with dual kernel
        let mut si = 0;
        if let Some(dfn) = dual_fn {
            while si + 1 < sub_batches.len() {
                dfn(&sub_batches[si], &sub_batches[si + 1], base_vec, &mut dist_buf_32);
                // First 16 queries
                let off_a = sub_offsets[si];
                for qi in 0..sub_batches[si].count() {
                    let gqi = off_a + qi;
                    let dist = dist_buf_32[qi];
                    if dist < thresholds[gqi] {
                        heaps[gqi].push(Neighbor { index: idx, distance: dist });
                        if heaps[gqi].len() > k { heaps[gqi].pop(); }
                        if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                    }
                }
                // Second 16 queries
                let off_b = sub_offsets[si + 1];
                for qi in 0..sub_batches[si + 1].count() {
                    let gqi = off_b + qi;
                    let dist = dist_buf_32[16 + qi];
                    if dist < thresholds[gqi] {
                        heaps[gqi].push(Neighbor { index: idx, distance: dist });
                        if heaps[gqi].len() > k { heaps[gqi].pop(); }
                        if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                    }
                }
                si += 2;
            }
        }
        // Handle leftover (odd sub-batch or no dual kernel)
        while si < sub_batches.len() {
            batched_fn(&sub_batches[si], base_vec, &mut dist_buf_16);
            let sub_offset = sub_offsets[si];
            for qi in 0..sub_batches[si].count() {
                let gqi = sub_offset + qi;
                let dist = dist_buf_16[qi];
                if dist < thresholds[gqi] {
                    heaps[gqi].push(Neighbor { index: idx, distance: dist });
                    if heaps[gqi].len() > k { heaps[gqi].pop(); }
                    if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                }
            }
            si += 1;
        }
    }

    for (qi, heap) in heaps.into_iter().enumerate() {
        let mut v: Vec<Neighbor> = heap.into_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results[qi] = v;
    }
}


// -- Partitioned computation --------------------------------------------------

/// Metadata for a single partition of the base vector space.
struct PartitionMeta {
    start: usize,
    end: usize,
    neighbors_path: PathBuf,
    distances_path: PathBuf,
    cached: bool,
}

/// Build a cache file path for a partition segment.
///
/// The cache key is derived from the base/query file stems, partition range,
/// k, and metric — NOT from the pipeline step ID. This allows partition
/// caches to be shared across profiles that use overlapping base-vector
/// windows with the same query set, k, and metric.
fn build_cache_path(
    cache_dir: &Path,
    cache_prefix: &str,
    start: usize,
    end: usize,
    k: usize,
    metric: Metric,
    suffix: &str,
    ext: &str,
) -> PathBuf {
    let metric_str = match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        Metric::DotProduct => "dot_product",
        Metric::L1 => "l1",
    };
    cache_dir.join(format!(
        "{}.range_{:06}_{:06}.k{}.{}.{}.{}",
        cache_prefix, start, end, k, metric_str, suffix, ext
    ))
}

/// Derive a cache key prefix from the base and query file paths.
///
/// Uses file stems so that partitions computed for the same physical files
/// are reusable regardless of which pipeline step or profile triggered them.
fn cache_prefix_for(base_path: &Path, query_path: &Path) -> String {
    let base_stem = base_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let query_stem = query_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    // Include file sizes in the cache key so partitions are invalidated
    // when the data changes (e.g., different base_fraction → different vectors).
    let base_size = std::fs::metadata(base_path).map(|m| m.len()).unwrap_or(0);
    let query_size = std::fs::metadata(query_path).map(|m| m.len()).unwrap_or(0);
    format!("{}.{}.{}_{}", base_stem, query_stem, base_size, query_size)
}

/// Validate that a cache file exists and has the expected byte size.
fn validate_cache_file(path: &Path, query_count: usize, k: usize, elem_size: usize) -> bool {
    let expected = query_count as u64 * (4 + k as u64 * elem_size as u64);
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() == expected,
        Err(_) => false,
    }
}

/// Query-partitioned KNN for f32 vectors.
/// Each thread owns a disjoint subset of queries and scans all base vectors.
/// Thread count capped so each gets a full QUERY_BATCH_SIZE of queries.
fn compute_partition(
    query_reader: &MmapVectorReader<f32>,
    query_count: usize,
    base_reader: &Arc<MmapVectorReader<f32>>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    metric: Metric,
    dim: usize,
    threads: usize,
    pb: &ProgressHandle,
) -> Vec<Vec<Neighbor>> {
    // For normalized cosine/dot data, dist_fn already uses the optimal
    // per-pair kernel. Select the matching batched kernel: if dist_fn was
    // chosen for DotProduct (normalized cosine), the batched kernel should
    // also be DotProduct (neg-dot, no norm division).
    let batched_fn = simd_distance::select_batched_fn_f32(metric);
    let mut results: Vec<Vec<Neighbor>> = (0..query_count).map(|_| Vec::new()).collect();

    if threads > 1 && query_count > 1 {
        let effective_threads = std::cmp::min(threads, query_count);
        let chunk_size = (query_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            results.chunks_mut(chunk_size).collect();

        std::thread::scope(|scope| {
            for (ci, chunk) in result_chunks.into_iter().enumerate() {
                let chunk_start = ci * chunk_size;
                let chunk_len = chunk.len();
                let base_ref = Arc::clone(base_reader);

                scope.spawn(move || {
                    let mut offset = 0;
                    while offset < chunk_len {
                        let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, chunk_len);
                        let batch_size = batch_end - offset;

                        let queries: Vec<&[f32]> = (0..batch_size)
                            .map(|i| query_reader.get_slice(chunk_start + offset + i))
                            .collect();

                        find_top_k_batch_f32(
                            &queries, &base_ref, start, end, k, dist_fn,
                            batched_fn, metric, dim, &mut chunk[offset..batch_end],
                        );

                        pb.inc(batch_size as u64);
                        offset = batch_end;
                    }
                });
            }
        });
    } else {
        let mut offset = 0;
        while offset < query_count {
            let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, query_count);
            let batch_size = batch_end - offset;
            let queries: Vec<&[f32]> = (0..batch_size)
                .map(|i| query_reader.get_slice(offset + i))
                .collect();
            find_top_k_batch_f32(
                &queries, base_reader, start, end, k, dist_fn,
                batched_fn, metric, dim, &mut results[offset..batch_end],
            );
            pb.inc(batch_size as u64);
            offset = batch_end;
        }
    }

    results
}

// -- Element type detection ---------------------------------------------------

/// Detected element type based on file extension.
use crate::pipeline::element_type::ElementType;

// -- f16 KNN functions --------------------------------------------------------

/// Number of queries processed per base-vector scan in the batched KNN loop.
///
/// Each batch scans all base vectors ONCE and computes distances to all queries
/// in the batch simultaneously. Larger batches reduce memory bandwidth (base
/// vectors are loaded once per batch instead of once per query) at the cost of
/// more working memory for heaps.
///
/// 256 queries × (heap ~1.6 KB + threshold 4B + query ptr 8B) ≈ 410 KB per
/// batch — fits comfortably in per-core L2 cache.
const QUERY_BATCH_SIZE: usize = 256;

/// Compute KNN for a batch of f16 queries against base vectors `[start, end)`.
///
/// Scans base vectors exactly once. Uses transposed SIMD kernels when
/// available for the given metric. The transposed path converts each base
/// vector f16→f32 once (via SIMD bulk conversion) and uses the f32 kernel,
/// since `TransposedBatch` already stores f32-converted query data.
#[inline(never)]
fn find_top_k_batch_f16(
    queries: &[&[half::f16]],
    base_reader: &MmapVectorReader<half::f16>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    batched_fn: Option<simd_distance::BatchedDistFnF32>,
    metric: Metric,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    if let Some(bfn) = batched_fn {
        find_top_k_batch_transposed_f16(queries, base_reader, start, end, k, bfn, metric, dim, results);
    } else {
        find_top_k_batch_pairwise_f16(queries, base_reader, start, end, k, dist_fn, results);
    }
}

/// Per-pair fallback for f16 batch processing.
#[inline(never)]
fn find_top_k_batch_pairwise_f16(
    queries: &[&[half::f16]],
    base_reader: &MmapVectorReader<half::f16>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    results: &mut [Vec<Neighbor>],
) {
    let batch_size = queries.len();
    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..batch_size)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; batch_size];

    for i in start..end {
        let base_vec = base_reader.get_slice(i);
        let idx = i as u32;

        for qi in 0..batch_size {
            let dist = dist_fn(queries[qi], base_vec);

            if dist < thresholds[qi] {
                heaps[qi].push(Neighbor { index: idx, distance: dist });
                if heaps[qi].len() > k {
                    heaps[qi].pop();
                }
                if heaps[qi].len() == k {
                    thresholds[qi] = heaps[qi].peek().unwrap().distance;
                }
            }
        }
    }

    for (qi, heap) in heaps.into_iter().enumerate() {
        let mut v: Vec<Neighbor> = heap.into_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results[qi] = v;
    }
}

/// Transposed SIMD batch processing for f16.
///
/// Splits queries into sub-batches of [`simd_distance::SIMD_BATCH_WIDTH`] (16),
/// transposes each sub-batch into dimension-major f32 layout, then scans base
/// vectors once computing 16 distances per SIMD FMA instruction.
///
/// Each f16 base vector is bulk-converted to f32 **once** via SIMD (32 AVX-512
/// conversions for 512 dims), then the f32 kernel processes all sub-batches.
/// This eliminates the 16× redundant per-sub-batch f16→f32 conversion that
/// would otherwise dominate the inner loop.
#[inline(never)]
fn find_top_k_batch_transposed_f16(
    queries: &[&[half::f16]],
    base_reader: &MmapVectorReader<half::f16>,
    start: usize,
    end: usize,
    k: usize,
    batched_fn: simd_distance::BatchedDistFnF32,
    metric: Metric,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    use simd_distance::{SIMD_BATCH_WIDTH, TransposedBatch};

    let batch_size = queries.len();
    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..batch_size)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; batch_size];

    // Pre-transpose queries into SIMD-width sub-batches (already f32)
    let mut sub_batches: Vec<TransposedBatch> = Vec::new();
    let mut sub_offsets: Vec<usize> = Vec::new();
    let mut offset = 0;
    while offset < batch_size {
        let sub_end = std::cmp::min(offset + SIMD_BATCH_WIDTH, batch_size);
        sub_batches.push(TransposedBatch::from_f16(&queries[offset..sub_end], dim));
        sub_offsets.push(offset);
        offset = sub_end;
    }

    // Reusable f32 buffer for base vector conversion
    let mut base_f32 = vec![0.0f32; dim];
    let mut dist_buf_16 = [0.0f32; SIMD_BATCH_WIDTH];
    let mut dist_buf_32 = [0.0f32; 32];

    // Select dual kernel for paired sub-batch processing (2 FMA ports)
    let dual_fn = simd_distance::select_dual_batched_fn_f32(metric);

    for i in start..end {
        let base_f16 = base_reader.get_slice(i);
        let idx = i as u32;

        // Convert base vector f16→f32 ONCE via SIMD bulk conversion
        simd_distance::convert_f16_to_f32_bulk(base_f16, &mut base_f32);

        // Process sub-batches in pairs with dual kernel (2 FMA ports)
        let mut si = 0;
        if let Some(dfn) = dual_fn {
            while si + 1 < sub_batches.len() {
                dfn(&sub_batches[si], &sub_batches[si + 1], &base_f32, &mut dist_buf_32);
                let off_a = sub_offsets[si];
                for qi in 0..sub_batches[si].count() {
                    let gqi = off_a + qi;
                    let dist = dist_buf_32[qi];
                    if dist < thresholds[gqi] {
                        heaps[gqi].push(Neighbor { index: idx, distance: dist });
                        if heaps[gqi].len() > k { heaps[gqi].pop(); }
                        if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                    }
                }
                let off_b = sub_offsets[si + 1];
                for qi in 0..sub_batches[si + 1].count() {
                    let gqi = off_b + qi;
                    let dist = dist_buf_32[16 + qi];
                    if dist < thresholds[gqi] {
                        heaps[gqi].push(Neighbor { index: idx, distance: dist });
                        if heaps[gqi].len() > k { heaps[gqi].pop(); }
                        if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                    }
                }
                si += 2;
            }
        }
        // Handle leftover
        while si < sub_batches.len() {
            batched_fn(&sub_batches[si], &base_f32, &mut dist_buf_16);
            let sub_offset = sub_offsets[si];
            for qi in 0..sub_batches[si].count() {
                let gqi = sub_offset + qi;
                let dist = dist_buf_16[qi];
                if dist < thresholds[gqi] {
                    heaps[gqi].push(Neighbor { index: idx, distance: dist });
                    if heaps[gqi].len() > k { heaps[gqi].pop(); }
                    if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                }
            }
            si += 1;
        }
    }

    for (qi, heap) in heaps.into_iter().enumerate() {
        let mut v: Vec<Neighbor> = heap.into_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results[qi] = v;
    }
}

/// Compute KNN for a single partition `[start, end)` across all f16 queries.
///
/// Query-partitioned KNN for f16 vectors.
/// Each thread owns a disjoint subset of queries and scans all base vectors.
/// Thread count is capped so each gets a full SIMD batch worth of queries.
fn compute_partition_f16(
    query_reader: &MmapVectorReader<half::f16>,
    query_count: usize,
    base_reader: &Arc<MmapVectorReader<half::f16>>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    metric: Metric,
    dim: usize,
    threads: usize,
    pb: &ProgressHandle,
) -> Vec<Vec<Neighbor>> {
    // For normalized cosine/dot data, dist_fn already uses the optimal
    // per-pair kernel. Select the matching batched kernel: if dist_fn was
    // chosen for DotProduct (normalized cosine), the batched kernel should
    // also be DotProduct (neg-dot, no norm division).
    let batched_fn = simd_distance::select_batched_fn_f32(metric);
    let mut results: Vec<Vec<Neighbor>> = (0..query_count).map(|_| Vec::new()).collect();

    if threads > 1 && query_count > 1 {
        let effective_threads = std::cmp::min(threads, query_count);
        let chunk_size = (query_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            results.chunks_mut(chunk_size).collect();

        std::thread::scope(|scope| {
            for (ci, chunk) in result_chunks.into_iter().enumerate() {
                let chunk_start = ci * chunk_size;
                let chunk_len = chunk.len();
                let base_ref = Arc::clone(base_reader);

                scope.spawn(move || {
                    let mut offset = 0;
                    while offset < chunk_len {
                        let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, chunk_len);
                        let batch_size = batch_end - offset;

                        let queries: Vec<&[half::f16]> = (0..batch_size)
                            .map(|i| query_reader.get_slice(chunk_start + offset + i))
                            .collect();

                        find_top_k_batch_f16(
                            &queries, &base_ref, start, end, k, dist_fn,
                            batched_fn, metric, dim, &mut chunk[offset..batch_end],
                        );

                        pb.inc(batch_size as u64);
                        offset = batch_end;
                    }
                });
            }
        });
    } else {
        let mut offset = 0;
        while offset < query_count {
            let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, query_count);
            let batch_size = batch_end - offset;
            let queries: Vec<&[half::f16]> = (0..batch_size)
                .map(|i| query_reader.get_slice(offset + i))
                .collect();
            find_top_k_batch_f16(
                &queries, base_reader, start, end, k, dist_fn,
                batched_fn, metric, dim, &mut results[offset..batch_end],
            );
            pb.inc(batch_size as u64);
            offset = batch_end;
        }
    }

    results
}

/// Compute KNN for a single partition `[start, end)` across all f64 queries.
///
/// Uses pairwise distance computation (no transposed batch kernel for f64 yet).
#[allow(clippy::too_many_arguments)]
fn compute_partition_f64(
    query_reader: &MmapVectorReader<f64>,
    query_count: usize,
    base_reader: &Arc<MmapVectorReader<f64>>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f64], &[f64]) -> f32,
    _metric: Metric,
    _dim: usize,
    threads: usize,
    pb: &ProgressHandle,
) -> Vec<Vec<Neighbor>> {
    let mut results: Vec<Vec<Neighbor>> = (0..query_count).map(|_| Vec::new()).collect();

    if threads > 1 && query_count > 1 {
        // Cap threads so each gets at least QUERY_BATCH_SIZE queries.
        // More threads = more redundant base-vector memory reads.
        let effective_threads = std::cmp::min(threads, query_count);
        let chunk_size = (query_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            results.chunks_mut(chunk_size).collect();

        std::thread::scope(|scope| {
            for (ci, chunk) in result_chunks.into_iter().enumerate() {
                let chunk_start = ci * chunk_size;
                let chunk_len = chunk.len();
                let base_ref = Arc::clone(base_reader);

                scope.spawn(move || {
                    let mut offset = 0;
                    while offset < chunk_len {
                        let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, chunk_len);
                        let batch_size = batch_end - offset;

                        let queries: Vec<&[f64]> = (0..batch_size)
                            .map(|i| query_reader.get_slice(chunk_start + offset + i))
                            .collect();

                        find_top_k_batch_pairwise_f64(
                            &queries, &base_ref, start, end, k, dist_fn,
                            &mut chunk[offset..batch_end],
                        );

                        pb.inc(batch_size as u64);
                        offset = batch_end;
                    }
                });
            }
        });
    } else {
        let mut offset = 0;
        while offset < query_count {
            let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, query_count);
            let batch_size = batch_end - offset;

            let queries: Vec<&[f64]> = (0..batch_size)
                .map(|i| query_reader.get_slice(offset + i))
                .collect();

            find_top_k_batch_pairwise_f64(
                &queries, base_reader, start, end, k, dist_fn,
                &mut results[offset..batch_end],
            );

            pb.inc(batch_size as u64);
            offset = batch_end;
        }
    }

    results
}

/// Per-pair distance computation for f64 vectors.
#[inline(never)]
fn find_top_k_batch_pairwise_f64(
    queries: &[&[f64]],
    base_reader: &MmapVectorReader<f64>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f64], &[f64]) -> f32,
    results: &mut [Vec<Neighbor>],
) {
    let batch_size = queries.len();
    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..batch_size)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; batch_size];

    for i in start..end {
        let base_vec = base_reader.get_slice(i);
        let idx = i as u32;

        for qi in 0..batch_size {
            let dist = dist_fn(queries[qi], base_vec);

            if dist < thresholds[qi] {
                heaps[qi].push(Neighbor { index: idx, distance: dist });
                if heaps[qi].len() > k {
                    heaps[qi].pop();
                }
                if heaps[qi].len() == k {
                    thresholds[qi] = heaps[qi].peek().unwrap().distance;
                }
            }
        }
    }

    for (qi, heap) in heaps.into_iter().enumerate() {
        let mut sorted: Vec<Neighbor> = heap.into_sorted_vec();
        sorted.reverse();
        results[qi] = sorted;
    }
}

/// Write partition results to cache files (both ivec and fvec).
fn write_partition_cache(
    meta: &PartitionMeta,
    results: &[Vec<Neighbor>],
    k: usize,
) -> Result<(), String> {
    write_indices(&meta.neighbors_path, results, k, 0)?;
    write_distances(&meta.distances_path, results, k)?;
    Ok(())
}

/// Merge all partition cache files into final output files.
///
/// For each query row, reads that row from every partition's cached ivec and
/// fvec, collects all (index, distance) candidates, sorts by distance, and
/// takes the top-K. Memory usage is O(num_partitions * k) per query.
fn merge_partitions(
    partitions: &[PartitionMeta],
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    query_count: usize,
    base_offset: usize,
    compress_cache: bool,
    ui: &veks_core::ui::UiHandle,
) -> Result<(), String> {
    // Open all partition files
    let load_pb = ui.bar_with_unit(
        partitions.len() as u64,
        "loading partition caches for merge",
        "partitions",
    );
    let mut ivec_readers: Vec<Box<dyn Read>> = Vec::with_capacity(partitions.len());
    let mut fvec_readers: Vec<Box<dyn Read>> = Vec::with_capacity(partitions.len());

    for (pi, part) in partitions.iter().enumerate() {
        if compress_cache || crate::pipeline::gz_cache::gz_exists(&part.neighbors_path) {
            let ivec_data = crate::pipeline::gz_cache::load_gz(&part.neighbors_path)?;
            ivec_readers.push(Box::new(Cursor::new(ivec_data)));

            let fvec_data = crate::pipeline::gz_cache::load_gz(&part.distances_path)?;
            fvec_readers.push(Box::new(Cursor::new(fvec_data)));
        } else {
            let ivec_file = std::fs::File::open(&part.neighbors_path)
                .map_err(|e| format!("open {}: {}", part.neighbors_path.display(), e))?;
            ivec_readers.push(Box::new(BufReader::with_capacity(1 << 16, ivec_file)));

            let fvec_file = std::fs::File::open(&part.distances_path)
                .map_err(|e| format!("open {}: {}", part.distances_path.display(), e))?;
            fvec_readers.push(Box::new(BufReader::with_capacity(1 << 16, fvec_file)));
        }
        load_pb.set_position((pi + 1) as u64);
    }
    load_pb.finish();

    // Open output files via AtomicWriter (temp-then-rename)
    use crate::pipeline::atomic_write::AtomicWriter;
    let mut idx_writer = AtomicWriter::with_capacity(1 << 20, indices_path)
        .map_err(|e| format!("create {}: {}", indices_path.display(), e))?;

    let mut dist_writer: Option<AtomicWriter> = match distances_path {
        Some(p) => Some(AtomicWriter::with_capacity(1 << 20, p)
            .map_err(|e| format!("create {}: {}", p.display(), e))?),
        None => None,
    };

    let row_bytes = 4 + k * 4; // dim header + k elements
    let mut ivec_row = vec![0u8; row_bytes];
    let mut fvec_row = vec![0u8; row_bytes];

    let pb = make_query_progress_bar(query_count as u64, ui);
    for _qi in 0..query_count {
        // Collect candidates from all partitions
        let mut candidates: Vec<Neighbor> = Vec::with_capacity(partitions.len() * k);

        for (pi, _part) in partitions.iter().enumerate() {
            ivec_readers[pi]
                .read_exact(&mut ivec_row)
                .map_err(|e| format!("read ivec partition: {}", e))?;
            fvec_readers[pi]
                .read_exact(&mut fvec_row)
                .map_err(|e| format!("read fvec partition: {}", e))?;

            // Parse k (index, distance) pairs from this partition's row
            for j in 0..k {
                let idx_offset = 4 + j * 4;
                let idx = i32::from_le_bytes([
                    ivec_row[idx_offset],
                    ivec_row[idx_offset + 1],
                    ivec_row[idx_offset + 2],
                    ivec_row[idx_offset + 3],
                ]);
                let dist = f32::from_le_bytes([
                    fvec_row[idx_offset],
                    fvec_row[idx_offset + 1],
                    fvec_row[idx_offset + 2],
                    fvec_row[idx_offset + 3],
                ]);

                if idx >= 0 && dist.is_finite() {
                    candidates.push(Neighbor {
                        index: idx as u32,
                        distance: dist,
                    });
                }
            }
        }

        // Sort by distance ascending and take top-K
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        candidates.truncate(k);

        // Write merged row to indices output
        let dim = k as i32;
        idx_writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        let offset32 = base_offset as u32;
        for j in 0..k {
            let idx: i32 = if j < candidates.len() {
                (candidates[j].index - offset32) as i32
            } else {
                -1
            };
            idx_writer
                .write_all(&idx.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }

        // Write merged row to distances output
        if let Some(ref mut dw) = dist_writer {
            dw.write_all(&dim.to_le_bytes())
                .map_err(|e| e.to_string())?;
            for j in 0..k {
                let dist: f32 = if j < candidates.len() {
                    candidates[j].distance
                } else {
                    f32::INFINITY
                };
                dw.write_all(&dist.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }

        pb.inc(1);
    }
    pb.finish();

    idx_writer.finish().map_err(|e| e.to_string())?;
    if let Some(dw) = dist_writer {
        dw.finish().map_err(|e| e.to_string())?;
    }

    Ok(())
}

// -- CommandOp impl -----------------------------------------------------------

impl CommandOp for ComputeKnnOp {
    fn command_path(&self) -> &str {
        "compute knn"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Brute-force exact K-nearest-neighbor ground truth computation".into(),
            body: format!(r#"# compute knn

Brute-force exact K-nearest-neighbor ground truth computation.

## Description

Computes ground truth KNN for a set of query vectors against a base vector
set. Outputs both neighbor indices (ivec) and distances (fvec).

This command performs an **exhaustive brute-force search**: for every query
vector it computes the distance to every base vector, maintaining a max-heap
of the top-K closest results. Because every (query, base) pair is evaluated,
this is the most computationally expensive step in a typical dataset
preparation pipeline — but the results it produces are exact, making them
the gold-standard "correct answers" that approximate nearest-neighbor (ANN)
algorithms are benchmarked against.

### Distance Metrics

Supports distance metrics: L2 (Euclidean), Cosine, DotProduct, L1 (Manhattan).
The metric chosen here must match the metric used during ANN evaluation so
that the ground truth is meaningful.

### Input Formats

Supports both f32 (fvec) and f16 (mvec) input formats. The element type is
detected from the file extension of the base vectors path. When mvec files
are used, native f16 SIMD distance kernels operate directly on half-precision
data without an explicit upcast to f32.

### Memory-Partitioned Computation

For large base vector sets that exceed available memory, the command supports
partitioned computation. The base space is split into partitions of
`partition_size` vectors. Each partition's per-query top-K results are cached
in a `.cache/` directory, and a merge phase combines partition results into
the final global top-K. Cached partitions are reusable across runs, so if a
run is interrupted it can resume from the last completed partition.

### Windowed Ranges

Both the base and query paths accept range notation to select a subset of
vectors (e.g., `base.mvec[0..1000000)`). This allows computing ground truth
over a specific slice of a large dataset without copying or splitting the
file.

### Per-Profile Execution

When used inside a sized profile, this command runs once per profile with the
profile's base range substituted automatically. This lets a single pipeline
definition produce ground truth files for every dataset size (e.g., 1M, 10M,
100M) in one invocation.

### Output Files

The command produces two files:

- **neighbor_indices.ivec** — for each query, the ordinal indices of its K
  nearest neighbors in the base set.
- **neighbor_distances.fvec** — the corresponding distances, in the same
  order.

These files are consumed downstream by recall-evaluation commands that
compare an ANN index's approximate results against this exact ground truth.

## Options

{}

## Notes

- Element type (f32 vs f16) is auto-detected from the base file extension.
- Partition caching allows incremental and resumable computation.
- Thread count of 0 uses the system default (all available cores).
- This is typically the most expensive step in a pipeline; plan for hours on
  billion-scale datasets even with high thread counts.
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Partition results and top-K heaps".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Parallel query processing".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential prefetch for mmap'd base vectors".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("indices") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let k: usize = match options.require("neighbors") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid neighbors: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let metric_str = options.get("metric").unwrap_or("L2");
        let metric = match Metric::from_str(metric_str) {
            Some(m) => m,
            None => {
                return error_result(
                    format!(
                        "unknown metric: '{}'. Use L2, COSINE, DOT_PRODUCT, or L1",
                        metric_str
                    ),
                    start,
                )
            }
        };
        let threads: usize = match options.parse_opt::<usize>("threads") {
            Ok(Some(v)) => v,
            Ok(None) => ctx.governor.current_or("threads", ctx.threads as u64) as usize,
            Err(e) => return error_result(e, start),
        };

        // Default partition_size is computed in execute_with_partitions
        // using actual entry size from the reader. 0 = auto.
        let partition_size: usize = match options.parse_opt::<usize>("partition_size") {
            Ok(Some(v)) => v,
            Ok(None) => ctx.governor.current("segmentsize")
                .map(|v| v as usize)
                .unwrap_or(0), // 0 = auto-size in execute_with_partitions
            Err(e) => return error_result(e, start),
        };

        let compress_cache = options.get("compress-cache").map(|s| s == "true").unwrap_or(false);
        let normalized = options.get("normalized").map(|s| s == "true").unwrap_or(false);

        // When vectors are normalized, Cosine = 1 - dot and DotProduct = -dot.
        // Use DotProduct kernel (no norm division) for both — faster and
        // numerically identical on unit vectors.
        let kernel_metric = if normalized && (metric == Metric::Cosine || metric == Metric::DotProduct) {
            log::info!("vectors are normalized — using DotProduct kernel for {} metric", metric_str);
            Metric::DotProduct
        } else {
            metric
        };

        let base_source = match resolve_source(base_str, &ctx.workspace) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);

        // Optional distances output
        let distances_path = options
            .get("distances")
            .map(|s| resolve_path(s, &ctx.workspace));

        // Create output directories
        for path in std::iter::once(&indices_path).chain(distances_path.iter()) {
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return error_result(
                            format!("failed to create directory: {}", e),
                            start,
                        );
                    }
                }
            }
        }

        let base_window = base_source.window;

        // Detect element type from base file extension and dispatch
        let etype = match ElementType::from_path(&base_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };
        match etype {
            ElementType::F16 => {
                let dist_fn = simd_distance::select_distance_fn_f16(kernel_metric);
                execute_f16(
                    &base_path,
                    &query_path,
                    &indices_path,
                    distances_path.as_deref(),
                    k,
                    kernel_metric,
                    dist_fn,
                    threads,
                    partition_size,
                    base_window,
                    compress_cache,
                    ctx,
                    start,
                )
            }
            ElementType::F64 => {
                let dist_fn = simd_distance::select_distance_fn_f64(kernel_metric);
                execute_f64(
                    &base_path,
                    &query_path,
                    &indices_path,
                    distances_path.as_deref(),
                    k,
                    kernel_metric,
                    dist_fn,
                    threads,
                    partition_size,
                    base_window,
                    compress_cache,
                    ctx,
                    start,
                )
            }
            ElementType::F32 | _ => {
                let dist_fn = simd_distance::select_distance_fn(kernel_metric);
                execute_f32(
                    &base_path,
                    &query_path,
                    &indices_path,
                    distances_path.as_deref(),
                    k,
                    kernel_metric,
                    dist_fn,
                    threads,
                    partition_size,
                    base_window,
                    compress_cache,
                    ctx,
                    start,
                )
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Base vectors file (fvec, mvec, or dvec)".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "query".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Query vectors file (fvec, mvec, or dvec)".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "indices".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output neighbor indices (ivec)".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "distances".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output neighbor distances (fvec)".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "neighbors".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of nearest neighbors (k)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "metric".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("L2".to_string()),
                description: "Distance metric: L2, COSINE, DOT_PRODUCT, L1".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "threads".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Thread count (0 = auto)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "partition_size".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1000000".to_string()),
                description: "Base vectors per partition for cache-backed computation".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "compress-cache".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Gzip-compress partition cache files".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query"],
            &["indices", "distances"],
        )
    }
}

/// Execute KNN computation using f32 vectors.
#[allow(clippy::too_many_arguments)]
fn execute_f32(
    base_path: &Path,
    query_path: &Path,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    threads: usize,
    partition_size: usize,
    base_window: Option<(usize, usize)>,
    compress_cache: bool,
    ctx: &mut StreamContext,
    start: Instant,
) -> CommandResult {
    ctx.ui.log(&format!("  opening base vectors: {}", base_path.display()));
    let base_reader = match MmapVectorReader::<f32>::open_fvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            return error_result(
                format!("failed to open base {}: {}", base_path.display(), e),
                start,
            )
        }
    };
    ctx.ui.log(&format!("  opening query vectors: {}", query_path.display()));
    let query_reader = match MmapVectorReader::<f32>::open_fvec(query_path) {
        Ok(r) => r,
        Err(e) => {
            return error_result(
                format!("failed to open query {}: {}", query_path.display(), e),
                start,
            )
        }
    };

    let file_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&*base_reader);
    let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
    let base_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&*base_reader);
    let query_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&query_reader);

    // Apply window to base vectors
    let (base_offset, base_count) = match base_window {
        Some((ws, we)) => {
            let ws = ws.min(file_count);
            let we = we.min(file_count);
            (ws, we.saturating_sub(ws))
        }
        None => (0, file_count),
    };

    if base_dim != query_dim {
        return error_result(
            format!("dimension mismatch: base={}, query={}", base_dim, query_dim),
            start,
        );
    }

    let base_bytes = base_count as u64 * base_dim as u64 * 4;
    let query_bytes = query_count as u64 * query_dim as u64 * 4;
    let batched_mode = if simd_distance::select_batched_fn_f32(metric).is_some() {
        "transposed AVX-512 (16-wide)"
    } else {
        "per-pair SimSIMD"
    };
    if base_offset > 0 {
        ctx.ui.log(&format!(
            "KNN: {} queries x {} base vectors (f32, dim={}, window=[{}..{})), k={}, metric={:?}, threads={}, simd={}, batch={}",
            format_count(query_count), format_count(base_count), base_dim,
            format_count(base_offset), format_count(base_offset + base_count),
            k, metric, threads, simd_distance::simd_level(), batched_mode
        ));
    } else {
        ctx.ui.log(&format!(
            "KNN: {} queries x {} base vectors (f32, dim={}), k={}, metric={:?}, threads={}, simd={}, batch={}",
            format_count(query_count), format_count(base_count), base_dim,
            k, metric, threads, simd_distance::simd_level(), batched_mode
        ));
    }
    ctx.ui.log(&format!(
        "  base: {} ({})  query: {} ({})",
        base_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(base_bytes),
        query_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(query_bytes),
    ));

    execute_with_partitions(
        &query_reader,
        &base_reader,
        base_offset,
        base_count,
        query_count,
        base_dim,
        indices_path,
        distances_path,
        k,
        metric,
        dist_fn,
        threads,
        partition_size,
        base_path,
        query_path,
        compress_cache,
        ctx,
        start,
        compute_partition,
    )
}

/// Execute KNN computation using f16 vectors.
#[allow(clippy::too_many_arguments)]
fn execute_f16(
    base_path: &Path,
    query_path: &Path,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    threads: usize,
    partition_size: usize,
    base_window: Option<(usize, usize)>,
    compress_cache: bool,
    ctx: &mut StreamContext,
    start: Instant,
) -> CommandResult {
    ctx.ui.log(&format!("  opening base vectors: {}", base_path.display()));
    let base_reader = match MmapVectorReader::<half::f16>::open_mvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            return error_result(
                format!("failed to open base {}: {}", base_path.display(), e),
                start,
            )
        }
    };
    ctx.ui.log(&format!("  opening query vectors: {}", query_path.display()));
    let query_reader = match MmapVectorReader::<half::f16>::open_mvec(query_path) {
        Ok(r) => r,
        Err(e) => {
            return error_result(
                format!("failed to open query {}: {}", query_path.display(), e),
                start,
            )
        }
    };

    let file_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&*base_reader);
    let query_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&query_reader);
    let base_dim = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&*base_reader);
    let query_dim = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&query_reader);

    // Apply window to base vectors
    let (base_offset, base_count) = match base_window {
        Some((ws, we)) => {
            let ws = ws.min(file_count);
            let we = we.min(file_count);
            (ws, we.saturating_sub(ws))
        }
        None => (0, file_count),
    };

    if base_dim != query_dim {
        return error_result(
            format!("dimension mismatch: base={}, query={}", base_dim, query_dim),
            start,
        );
    }

    let base_bytes = base_count as u64 * base_dim as u64 * 2;
    let query_bytes = query_count as u64 * query_dim as u64 * 2;
    let batched_mode = if simd_distance::select_batched_fn_f16(metric).is_some() {
        "transposed AVX-512 (16-wide)"
    } else {
        "per-pair SimSIMD"
    };
    if base_offset > 0 {
        ctx.ui.log(&format!(
            "KNN: {} queries x {} base vectors (f16, dim={}, window=[{}..{})), k={}, metric={:?}, threads={}, simd={}, batch={}",
            format_count(query_count), format_count(base_count), base_dim,
            format_count(base_offset), format_count(base_offset + base_count),
            k, metric, threads, simd_distance::simd_level(), batched_mode
        ));
    } else {
        ctx.ui.log(&format!(
            "KNN: {} queries x {} base vectors (f16, dim={}), k={}, metric={:?}, threads={}, simd={}, batch={}",
            format_count(query_count), format_count(base_count), base_dim,
            k, metric, threads, simd_distance::simd_level(), batched_mode
        ));
    }
    ctx.ui.log(&format!(
        "  base: {} ({})  query: {} ({})",
        base_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(base_bytes),
        query_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(query_bytes),
    ));

    execute_with_partitions(
        &query_reader,
        &base_reader,
        base_offset,
        base_count,
        query_count,
        base_dim,
        indices_path,
        distances_path,
        k,
        metric,
        dist_fn,
        threads,
        partition_size,
        base_path,
        query_path,
        compress_cache,
        ctx,
        start,
        compute_partition_f16,
    )
}

/// Execute KNN computation using f64 vectors.
#[allow(clippy::too_many_arguments)]
fn execute_f64(
    base_path: &Path,
    query_path: &Path,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[f64], &[f64]) -> f32,
    threads: usize,
    partition_size: usize,
    base_window: Option<(usize, usize)>,
    compress_cache: bool,
    ctx: &mut StreamContext,
    start: Instant,
) -> CommandResult {
    ctx.ui.log(&format!("  opening base vectors: {}", base_path.display()));
    let base_reader = match MmapVectorReader::<f64>::open_dvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => return error_result(format!("failed to open base {}: {}", base_path.display(), e), start),
    };
    ctx.ui.log(&format!("  opening query vectors: {}", query_path.display()));
    let query_reader = match MmapVectorReader::<f64>::open_dvec(query_path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("failed to open query {}: {}", query_path.display(), e), start),
    };

    let file_count = VectorReader::<f64>::count(&*base_reader);
    let query_count = VectorReader::<f64>::count(&query_reader);
    let base_dim = VectorReader::<f64>::dim(&*base_reader);
    let query_dim = VectorReader::<f64>::dim(&query_reader);

    let (base_offset, base_count) = match base_window {
        Some((ws, we)) => {
            let ws = ws.min(file_count);
            let we = we.min(file_count);
            (ws, we.saturating_sub(ws))
        }
        None => (0, file_count),
    };

    if base_dim != query_dim {
        return error_result(format!("dimension mismatch: base={}, query={}", base_dim, query_dim), start);
    }

    let base_bytes = base_count as u64 * base_dim as u64 * 8;
    let query_bytes = query_count as u64 * query_dim as u64 * 8;
    if base_offset > 0 {
        ctx.ui.log(&format!(
            "KNN: {} queries x {} base vectors (f64, dim={}, window=[{}..{})), k={}, metric={:?}, threads={}, simd={}",
            format_count(query_count), format_count(base_count), base_dim,
            format_count(base_offset), format_count(base_offset + base_count),
            k, metric, threads, simd_distance::simd_level()
        ));
    } else {
        ctx.ui.log(&format!(
            "KNN: {} queries x {} base vectors (f64, dim={}), k={}, metric={:?}, threads={}, simd={}",
            format_count(query_count), format_count(base_count), base_dim,
            k, metric, threads, simd_distance::simd_level()
        ));
    }
    ctx.ui.log(&format!(
        "  base: {} ({})  query: {} ({})",
        base_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(base_bytes),
        query_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(query_bytes),
    ));

    execute_with_partitions(
        &query_reader,
        &base_reader,
        base_offset,
        base_count,
        query_count,
        base_dim,
        indices_path,
        distances_path,
        k,
        metric,
        dist_fn,
        threads,
        partition_size,
        base_path,
        query_path,
        compress_cache,
        ctx,
        start,
        compute_partition_f64,
    )
}

/// Shared partition/merge logic for both f32 and f16 paths.
///
/// Query vectors are accessed via mmap random access on demand — no bulk
/// loading. Each base-vector partition's results are cached independently,
/// then merged into the final output.
#[allow(clippy::too_many_arguments)]
fn execute_with_partitions<T>(
    query_reader: &MmapVectorReader<T>,
    base_reader: &Arc<MmapVectorReader<T>>,
    base_offset: usize,
    base_count: usize,
    query_count: usize,
    dim: usize,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[T], &[T]) -> f32,
    threads: usize,
    partition_size: usize,
    base_path: &Path,
    query_path: &Path,
    compress_cache: bool,
    ctx: &mut StreamContext,
    start: Instant,
    compute_fn: fn(&MmapVectorReader<T>, usize, &Arc<MmapVectorReader<T>>, usize, usize, usize, fn(&[T], &[T]) -> f32, Metric, usize, usize, &ProgressHandle) -> Vec<Vec<Neighbor>>,
) -> CommandResult
where
    T: Send + Sync + 'static,
{
    // Segment size = cache artifact granularity (default 1M from governor).
    // This determines how cache files are partitioned for cross-profile reuse.
    // Profile sizes (10M, 20M, etc.) should be multiples of segment_size.
    let segment_size = partition_size.max(1);

    // Pass size = how many vectors to process per memory pass.
    // Auto-sized to ~50% of system RAM. Multiple segments are computed
    // per pass to amortize page-in overhead, but each segment's results
    // are written to separate cache files for reuse.
    let pass_size = {
        let total_ram: u64 = std::fs::read_to_string("/proc/meminfo").ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|n| n.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(8 * 1024 * 1024 * 1024);
        let half_ram = (total_ram / 2) as usize;
        let entry_size = base_reader.entry_size().max(1);
        let auto = (half_ram / entry_size).max(segment_size);
        // Round down to segment boundary
        let auto = (auto / segment_size) * segment_size;
        auto.max(segment_size)
    };
    let segments_per_pass = pass_size / segment_size;

    ctx.ui.log(&format!(
        "  segment size: {} vectors, pass size: {} vectors ({} segments/pass, {:.1} GiB/pass)",
        format_count(segment_size), format_count(pass_size), segments_per_pass,
        (pass_size as u64 * base_reader.entry_size() as u64) as f64 / (1024.0 * 1024.0 * 1024.0),
    ));

    // For cache path generation and partitioning, we use segment_size
    // as the partition unit. The pass_size controls prefetch grouping.
    let partition_size = segment_size;

    // Single-partition fast path
    if base_count <= partition_size {
        let pb = make_query_progress_bar(query_count as u64, &ctx.ui);
        let results = compute_fn(query_reader, query_count, base_reader, base_offset, base_offset + base_count, k, dist_fn, metric, dim, threads, &pb);
        pb.finish();

        ctx.ui.log("  writing results...");
        if let Err(e) = write_indices(indices_path, &results, k, base_offset) {
            return error_result(e, start);
        }

        let mut produced = vec![indices_path.to_path_buf()];

        if let Some(dist_path) = distances_path {
            if let Err(e) = write_distances(dist_path, &results, k) {
                return error_result(e, start);
            }
            produced.push(dist_path.to_path_buf());
        }

        // Write verified counts for the bound checker
        for xvec_path in &produced {
            let var_name = format!("verified_count:{}",
                xvec_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &query_count.to_string());
            ctx.defaults.insert(var_name, query_count.to_string());
        }

        return CommandResult {
            status: Status::Ok,
            message: format!(
                "computed KNN: {} queries, k={}, metric={:?}, {} base vectors",
                query_count, k, metric, base_count
            ),
            produced,
            elapsed: start.elapsed(),
        };
    }

    // Partitioned path

    // Memory-aware partition sizing (REQ-RM-09).
    //
    // The main memory cost per partition is the result set:
    //   query_count × k × 8 bytes (Neighbor = u32 index + f32 distance)
    // Plus per-thread heaps and batch buffers (~small).
    // If the result set for the configured partition_size exceeds the memory
    // budget, we don't need to change partition_size — the result set size
    // depends on query_count and k, not partition_size. But base vectors
    // consume page cache proportional to partition_size × dim × sizeof(T),
    // so we can reduce partition_size to limit page cache pressure.
    let partition_size = if let Some(mem_ceiling) = ctx.governor.mem_ceiling() {
        let snapshot = super::super::resource::SystemSnapshot::sample();
        let target = (mem_ceiling as f64 * 0.85) as u64;
        let available = if snapshot.rss_bytes < target {
            target - snapshot.rss_bytes
        } else {
            (mem_ceiling as f64 * 0.10) as u64
        };

        // Result set memory (fixed regardless of partition_size)
        let result_bytes = query_count as u64 * k as u64 * 8;
        // Per-thread heap memory
        let heap_bytes = threads as u64 * k as u64 * 8;
        let fixed_overhead = result_bytes + heap_bytes;

        if fixed_overhead > available {
            ctx.ui.log(&format!(
                "  WARNING: KNN result set alone ({}) exceeds available memory ({})",
                format_bytes(fixed_overhead), format_bytes(available),
            ));
            ctx.ui.log(&format!(
                "    result set: {} queries × k={} × 8B = {}",
                format_count(query_count), k, format_bytes(result_bytes),
            ));
        }

        // Remaining memory for base vector page cache
        let remaining = available.saturating_sub(fixed_overhead);
        let entry_size = base_reader.entry_size() as u64;
        if entry_size > 0 {
            let max_partition = (remaining / entry_size) as usize;
            if max_partition < partition_size && max_partition >= 1000 {
                ctx.ui.log(&format!(
                    "  memory-aware partition sizing: {} → {} (available: {}, RSS: {}, ceiling: {})",
                    format_count(partition_size), format_count(max_partition),
                    format_bytes(available), format_bytes(snapshot.rss_bytes),
                    format_bytes(mem_ceiling),
                ));
                max_partition
            } else {
                partition_size
            }
        } else {
            partition_size
        }
    } else {
        partition_size
    };

    if !ctx.cache.exists() {
        if let Err(e) = std::fs::create_dir_all(&ctx.cache) {
            return error_result(
                format!("failed to create cache directory: {}", e),
                start,
            );
        }
    }

    // Phase 1: Plan partitions and validate cache
    let estimated_partitions = (base_count + partition_size - 1) / partition_size;
    ctx.ui.log(&format!(
        "  planning ~{} partitions (partition_size={})...",
        estimated_partitions, format_count(partition_size)
    ));

    let plan_pb = ctx.ui.bar_with_unit(estimated_partitions as u64, "validating cache", "partitions");

    let cache_prefix = cache_prefix_for(base_path, query_path);
    let base_end = base_offset + base_count;
    let mut partitions: Vec<PartitionMeta> = Vec::new();
    let mut part_start = base_offset;

    // Scan for existing super-partitions (merged results from smaller profiles)
    // that cover a prefix of the range. Use the largest available cached partition
    // starting at part_start before falling back to partition_size chunks.
    let find_largest_cached = |start: usize, max_end: usize| -> Option<usize> {
        // Scan candidate endpoints: try the full remaining range first, then
        // halve down to find the largest cached partition.
        let mut try_end = max_end;
        while try_end > start + partition_size {
            let n_path = build_cache_path(
                &ctx.cache, &cache_prefix, start, try_end, k, metric, "neighbors", "ivec",
            );
            let d_path = build_cache_path(
                &ctx.cache, &cache_prefix, start, try_end, k, metric, "distances", "fvec",
            );
            let exists = if compress_cache {
                crate::pipeline::gz_cache::gz_exists(&n_path)
                    && crate::pipeline::gz_cache::gz_exists(&d_path)
            } else {
                validate_cache_file(&n_path, query_count, k, 4)
                    && validate_cache_file(&d_path, query_count, k, 4)
            };
            if exists {
                return Some(try_end);
            }
            // Try the next smaller multiple of partition_size
            try_end = ((try_end - start - 1) / partition_size) * partition_size + start;
            if try_end <= start { break; }
        }
        None
    };

    while part_start < base_end {
        // Check for a super-partition first
        let (part_end, neighbors_path, dist_cache_path, cached) =
            if let Some(super_end) = find_largest_cached(part_start, base_end) {
                ctx.ui.log(&format!(
                    "  reusing cached super-partition [{}, {}) from smaller profile",
                    format_count(part_start), format_count(super_end),
                ));
                let n = build_cache_path(&ctx.cache, &cache_prefix, part_start, super_end, k, metric, "neighbors", "ivec");
                let d = build_cache_path(&ctx.cache, &cache_prefix, part_start, super_end, k, metric, "distances", "fvec");
                (super_end, n, d, true)
            } else {
                let pe = std::cmp::min(part_start + partition_size, base_end);
                let n = build_cache_path(&ctx.cache, &cache_prefix, part_start, pe, k, metric, "neighbors", "ivec");
                let d = build_cache_path(&ctx.cache, &cache_prefix, part_start, pe, k, metric, "distances", "fvec");
                let c = if compress_cache {
                    crate::pipeline::gz_cache::gz_exists(&n) && crate::pipeline::gz_cache::gz_exists(&d)
                } else {
                    validate_cache_file(&n, query_count, k, 4) && validate_cache_file(&d, query_count, k, 4)
                };
                (pe, n, d, c)
            };

        partitions.push(PartitionMeta {
            start: part_start,
            end: part_end,
            neighbors_path,
            distances_path: dist_cache_path,
            cached,
        });
        plan_pb.inc(1);
        part_start = part_end;
    }
    plan_pb.finish();

    let num_partitions = partitions.len();
    let cached_count = partitions.iter().filter(|p| p.cached).count();
    let to_compute = num_partitions - cached_count;

    ctx.ui.log(&format!(
        "  {} partitions of {} base vectors ({} cached, {} to compute)",
        num_partitions, format_count(partition_size), cached_count, to_compute
    ));

    // Phase 2: Compute missing partitions
    //
    // 3-stage pipeline: load → compute → save
    //
    // The bottleneck is I/O (both load and save), while compute is fast
    // (in-memory). The pipeline is designed to keep the I/O channel
    // saturated at all times:
    //
    //   - Load uses 2-partition lookahead: load N+1 and N+2 are both
    //     started before compute N begins, so by the time we need N+1
    //     its pages are already warm.
    //   - Save and load may overlap (read vs write I/O), keeping the
    //     channel busy in both directions.
    //   - If load and save DO contend for the same I/O resource, save
    //     (drain) is waited on first to bias toward pipeline clearing
    //     and steady-state throughput.
    //
    // Steady-state per partition:
    //   1. Wait save N-1    (drain output first if pending)
    //   2. Wait load N      (should be warm — started 2 iterations ago)
    //   3. Compute N        (CPU-bound; load N+1, N+2 in flight)
    //   4. Spawn save N     (background write)
    //   5. Spawn load N+2   (refill lookahead queue)

    // Hint sequential access pattern for base vectors (REQ-RM-10)
    base_reader.advise_sequential();

    // Collect uncached partition indices for lookahead
    let uncached_indices: Vec<usize> = partitions.iter()
        .enumerate()
        .filter(|(_, p)| !p.cached)
        .map(|(i, _)| i)
        .collect();

    let total_prefetch_bytes: u64 = uncached_indices.iter()
        .map(|&i| ((partitions[i].end - partitions[i].start) as u64) * (base_reader.entry_size() as u64))
        .sum();
    ctx.ui.log(&format!(
        "  pipeline: {} uncached partition{} ({} to page in)",
        uncached_indices.len(),
        if uncached_indices.len() == 1 { "" } else { "s" },
        format_bytes(total_prefetch_bytes),
    ));

    // Shared counter for the prefetch progress bar
    let prefetch_bytes_paged = Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Prefetch progress bar — updated by a background tick thread
    let prefetch_pb = ctx.ui.bar_with_unit(total_prefetch_bytes, "paging in", "bytes");
    let prefetch_bytes_for_tick = Arc::clone(&prefetch_bytes_paged);
    let prefetch_tick_handle = std::thread::spawn(move || {
        loop {
            let paged = prefetch_bytes_for_tick.load(std::sync::atomic::Ordering::Relaxed);
            prefetch_pb.set_position(paged);
            if paged >= total_prefetch_bytes {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        prefetch_pb.finish();
    });

    let mut computed = 0usize;
    let mut prev_writer: Option<std::thread::JoinHandle<Result<(), String>>> = None;
    let mut prev_write_start: Option<Instant> = None;

    // Lookahead queue for load (prefetch) handles.
    // We maintain up to LOAD_DEPTH outstanding loads so the I/O channel
    // is continuously fed. Each entry corresponds to a future uncached
    // partition whose pages are being warmed.
    const LOAD_DEPTH: usize = 2;
    let mut load_queue: std::collections::VecDeque<std::thread::JoinHandle<()>> =
        std::collections::VecDeque::with_capacity(LOAD_DEPTH);

    // Overall partition progress bar — tracks how many partitions have been
    // computed out of the total uncached count.
    let overall_pb = if to_compute > 1 {
        Some(ctx.ui.bar_with_unit(to_compute as u64, "partitions", "partitions"))
    } else {
        None
    };

    // Seed the load queue: prefetch by pass_size (multiple segments at once).
    // This amortizes I/O overhead — one large sequential prefetch instead
    // of many small ones.
    let mut next_load_slot = 0usize;
    while next_load_slot < uncached_indices.len() && load_queue.len() < LOAD_DEPTH {
        // Prefetch a contiguous run of up to segments_per_pass segments
        let first_idx = uncached_indices[next_load_slot];
        let prefetch_start = partitions[first_idx].start;
        let mut prefetch_end = partitions[first_idx].end;
        let mut slots_covered = 1;
        while slots_covered < segments_per_pass
            && next_load_slot + slots_covered < uncached_indices.len()
        {
            let next_idx = uncached_indices[next_load_slot + slots_covered];
            // Only batch contiguous segments
            if partitions[next_idx].start == prefetch_end {
                prefetch_end = partitions[next_idx].end;
                slots_covered += 1;
            } else {
                break;
            }
        }
        let reader = Arc::clone(base_reader);
        let counter = Arc::clone(&prefetch_bytes_paged);
        load_queue.push_back(std::thread::spawn(move || {
            reader.prefetch_pages(prefetch_start, prefetch_end, Some(&counter));
        }));
        next_load_slot += slots_covered;
    }

    for (_ui_idx, &pi) in uncached_indices.iter().enumerate() {
        let part = &partitions[pi];

        // Governor checkpoint at partition boundary
        if ctx.governor.checkpoint() {
            log::info!(
                "partition {}/{} — governor throttle active, continuing with current settings",
                pi + 1, num_partitions,
            );
        }

        computed += 1;
        let part_size = part.end - part.start;
        let part_bytes = part_size as u64 * base_reader.entry_size() as u64;

        // Partition header — visible in the TUI log pane
        ctx.ui.log(&format!(
            "  partition {}/{} [{},{}) — {} queries x {} base ({}) metric={:?}  [{}/{}]",
            pi + 1, num_partitions,
            format_count(part.start), format_count(part.end),
            format_count(query_count), format_count(part_size),
            format_bytes(part_bytes),
            metric,
            computed, to_compute,
        ));

        if let Some(ref opb) = overall_pb {
            opb.set_message(format!(
                "partition {}/{} [{},{})",
                computed, to_compute,
                format_count(part.start), format_count(part.end),
            ));
        }

        // ── Stage 1: Ensure prefetch issued for this partition ─────
        // With async madvise, prefetch threads return instantly after
        // issuing the kernel hint. Just join to confirm it was issued.
        let stage_t = Instant::now();
        if let Some(handle) = load_queue.pop_front() {
            let _ = handle.join();
        }
        let prefetch_ms = stage_t.elapsed().as_millis();
        if prefetch_ms > 100 {
            ctx.ui.log(&format!("    prefetch join: {}ms", prefetch_ms));
        }

        // ── Stage 2: Compute ─────────────────────────────────────────
        // CPU-bound. Save of the previous partition runs concurrently
        // in the background. Kernel readahead for future partitions
        // runs asynchronously (madvise). No blocking waits needed.
        let part_start_time = Instant::now();
        let pb = ctx.ui.bar_with_unit(
            query_count as u64,
            &format!("KNN {:?} [{},{}) [{}/{}]",
                metric,
                format_count(part.start), format_count(part.end),
                computed, to_compute),
            "queries",
        );
        let results = compute_fn(
            query_reader, query_count, base_reader, part.start, part.end, k, dist_fn, metric, dim, threads, &pb,
        );
        pb.finish();
        let compute_elapsed = part_start_time.elapsed();

        // Partition completion log with throughput
        let pairs_per_sec = (query_count as f64 * part_size as f64) / compute_elapsed.as_secs_f64();
        ctx.ui.log(&format!(
            "    done in {:.1}s ({}/s distance evaluations)",
            compute_elapsed.as_secs_f64(),
            format_count(pairs_per_sec as usize),
        ));

        if let Some(ref opb) = overall_pb {
            opb.inc(1);
        }

        // ── Stage 3: Drain previous save, then spawn new save ────────
        // Block on the previous save only now — right before spawning
        // a new one. This means save(N-1) ran fully concurrent with
        // compute(N). We only block here to bound memory to one result
        // set in flight at a time.
        if let Some(handle) = prev_writer.take() {
            if !handle.is_finished() {
                ctx.ui.log("    waiting for previous partition save...");
            }
            let write_result = join_writer_with_spinner(handle, "previous", &ctx.ui);
            let write_elapsed = prev_write_start.take()
                .map(|s| s.elapsed())
                .unwrap_or_default();
            match write_result {
                Ok(()) => {
                    log::info!("previous save completed in {:.1}s", write_elapsed.as_secs_f64());
                }
                Err(msg) => {
                    if let Some(opb) = overall_pb.as_ref() { opb.finish(); }
                    return error_result(msg, start);
                }
            }
        }

        // Spawn save for this partition — runs concurrently with next
        // iteration's compute.
        let neighbors_path = part.neighbors_path.clone();
        let distances_path = part.distances_path.clone();
        let part_start_idx = part.start;
        let part_end_idx = part.end;
        let part_query_count = query_count;
        prev_write_start = Some(Instant::now());
        prev_writer = Some(std::thread::spawn(move || {
            if compress_cache {
                // Serialize to in-memory buffers, then gzip-compress and write
                let ivec_data = serialize_indices(&results, k, 0)?;
                crate::pipeline::gz_cache::save_gz(&neighbors_path, &ivec_data)?;

                let fvec_data = serialize_distances(&results, k)?;
                crate::pipeline::gz_cache::save_gz(&distances_path, &fvec_data)?;

                let gz_ivec = crate::pipeline::gz_cache::gz_path(&neighbors_path);
                let gz_fvec = crate::pipeline::gz_cache::gz_path(&distances_path);
                let ivec_size = std::fs::metadata(&gz_ivec)
                    .map(|m| m.len()).unwrap_or(0);
                let fvec_size = std::fs::metadata(&gz_fvec)
                    .map(|m| m.len()).unwrap_or(0);
                log::info!(
                    "→ {}  ({})",
                    gz_ivec.file_name().unwrap_or_default().to_string_lossy(),
                    format_bytes(ivec_size),
                );
                log::info!(
                    "→ {}  ({})",
                    gz_fvec.file_name().unwrap_or_default().to_string_lossy(),
                    format_bytes(fvec_size),
                );
            } else {
                write_indices(&neighbors_path, &results, k, 0)?;
                write_distances(&distances_path, &results, k)?;

                if !validate_cache_file(&neighbors_path, part_query_count, k, 4)
                    || !validate_cache_file(&distances_path, part_query_count, k, 4)
                {
                    return Err(format!(
                        "partition [{}, {}) cache files failed size validation after write",
                        part_start_idx, part_end_idx,
                    ));
                }

                let ivec_size = std::fs::metadata(&neighbors_path)
                    .map(|m| m.len()).unwrap_or(0);
                let fvec_size = std::fs::metadata(&distances_path)
                    .map(|m| m.len()).unwrap_or(0);
                log::info!(
                    "→ {}  ({})",
                    neighbors_path.file_name().unwrap_or_default().to_string_lossy(),
                    format_bytes(ivec_size),
                );
                log::info!(
                    "→ {}  ({})",
                    distances_path.file_name().unwrap_or_default().to_string_lossy(),
                    format_bytes(fvec_size),
                );
            }
            Ok(())
        }));

        // ── Stage 4: Refill prefetch queue ──────────────────────────
        // Batch contiguous uncached segments into a single large prefetch
        // (up to segments_per_pass segments per prefetch call).
        if load_queue.is_empty() && next_load_slot < uncached_indices.len() {
            let first_idx = uncached_indices[next_load_slot];
            let prefetch_start = partitions[first_idx].start;
            let mut prefetch_end = partitions[first_idx].end;
            let mut slots_covered = 1;
            while slots_covered < segments_per_pass
                && next_load_slot + slots_covered < uncached_indices.len()
            {
                let next_idx = uncached_indices[next_load_slot + slots_covered];
                if partitions[next_idx].start == prefetch_end {
                    prefetch_end = partitions[next_idx].end;
                    slots_covered += 1;
                } else {
                    break;
                }
            }
            let reader = Arc::clone(base_reader);
            let counter = Arc::clone(&prefetch_bytes_paged);
            load_queue.push_back(std::thread::spawn(move || {
                reader.prefetch_pages(prefetch_start, prefetch_end, Some(&counter));
            }));
            next_load_slot += slots_covered;
        }

        // Release completed partition's pages from page cache (REQ-RM-10)
        base_reader.release_range(part.start, part.end);

        log::info!(
            "partition {}/{} — compute done in {:.1}s (save in background)",
            pi + 1, num_partitions,
            compute_elapsed.as_secs_f64(),
        );
    }

    // Drain final save
    if let Some(handle) = prev_writer.take() {
        let write_result = join_writer_with_spinner(handle, "final", &ctx.ui);
        let write_elapsed = prev_write_start.take()
            .map(|s| s.elapsed())
            .unwrap_or_default();
        match write_result {
            Ok(()) => {
                log::info!("final save completed in {:.1}s", write_elapsed.as_secs_f64());
            }
            Err(msg) => {
                if let Some(opb) = overall_pb.as_ref() { opb.finish(); }
                return error_result(msg, start);
            }
        }
    }

    // Drain any remaining loads (shouldn't happen but be safe)
    for handle in load_queue {
        let _ = handle.join();
    }

    // Finish the overall partition progress bar
    if let Some(opb) = overall_pb {
        opb.finish();
    }

    // Stop the prefetch progress bar tick thread
    let _ = prefetch_tick_handle.join();

    // Phase 3: Merge
    ctx.ui.log(&format!("  merging {} partitions ({} queries)...", num_partitions, query_count));

    if let Err(e) = merge_partitions(
        &partitions,
        indices_path,
        distances_path,
        k,
        query_count,
        base_offset,
        compress_cache,
        &ctx.ui,
    ) {
        return error_result(format!("merge failed: {}", e), start);
    }

    // Save merged result as a cache partition covering the full range.
    // This allows larger profiles to reuse this as a single "super-partition"
    // instead of re-reading all the individual smaller partitions.
    if num_partitions > 1 {
        let full_neighbors = build_cache_path(
            &ctx.cache, &cache_prefix, base_offset, base_end, k, metric, "neighbors", "ivec",
        );
        let full_distances = build_cache_path(
            &ctx.cache, &cache_prefix, base_offset, base_end, k, metric, "distances", "fvec",
        );
        // Copy the final output to the cache as the full-range partition
        let cache_sp = ctx.ui.spinner("caching merged result as super-partition");
        if !full_neighbors.exists() {
            let _ = std::fs::copy(indices_path, &full_neighbors);
        }
        if let Some(dp) = distances_path {
            if !full_distances.exists() {
                let _ = std::fs::copy(dp, &full_distances);
            }
        }
        cache_sp.finish();
        ctx.ui.log(&format!(
            "  cached merged result as partition [{}, {}) for reuse by larger profiles",
            format_count(base_offset), format_count(base_end),
        ));
    }

    let mut produced = vec![indices_path.to_path_buf()];
    if let Some(dp) = distances_path {
        produced.push(dp.to_path_buf());
    }

    // Write verified counts for the bound checker
    for xvec_path in &produced {
        let var_name = format!("verified_count:{}",
            xvec_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &query_count.to_string());
        ctx.defaults.insert(var_name, query_count.to_string());
    }

    CommandResult {
        status: Status::Ok,
        message: format!(
            "computed KNN: {} queries, k={}, metric={:?}, {} base vectors ({} partitions)",
            query_count, k, metric, base_count, num_partitions
        ),
        produced,
        elapsed: start.elapsed(),
    }
}

/// Write neighbor indices as ivec (each row: k i32 indices).
///
/// `base_offset` is subtracted from each neighbor index so output indices are
/// 0-based relative to the window start.
fn write_indices(path: &Path, results: &[Vec<Neighbor>], k: usize, base_offset: usize) -> Result<(), String> {
    use crate::pipeline::atomic_write::AtomicWriter;
    let mut writer = AtomicWriter::with_capacity(1 << 20, path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;

    let dim = k as i32;
    let offset32 = base_offset as u32;
    for row in results {
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for i in 0..k {
            let idx: i32 = if i < row.len() {
                (row[i].index - offset32) as i32
            } else {
                -1 // Padding if fewer than k neighbors
            };
            writer
                .write_all(&idx.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    writer.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write neighbor distances as fvec (each row: k f32 distances).
fn write_distances(path: &Path, results: &[Vec<Neighbor>], k: usize) -> Result<(), String> {
    use crate::pipeline::atomic_write::AtomicWriter;
    let mut writer = AtomicWriter::with_capacity(1 << 20, path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;

    let dim = k as i32;
    for row in results {
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for i in 0..k {
            let dist: f32 = if i < row.len() {
                row[i].distance
            } else {
                f32::INFINITY
            };
            writer
                .write_all(&dist.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    writer.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Serialize neighbor indices to an in-memory buffer (ivec format).
///
/// Same layout as [`write_indices`] but returns bytes instead of writing to a file.
fn serialize_indices(results: &[Vec<Neighbor>], k: usize, base_offset: usize) -> Result<Vec<u8>, String> {
    let row_bytes = 4 + k * 4;
    let mut buf = Vec::with_capacity(results.len() * row_bytes);
    let dim = k as i32;
    let offset32 = base_offset as u32;
    for row in results {
        buf.extend_from_slice(&dim.to_le_bytes());
        for i in 0..k {
            let idx: i32 = if i < row.len() {
                (row[i].index - offset32) as i32
            } else {
                -1
            };
            buf.extend_from_slice(&idx.to_le_bytes());
        }
    }
    Ok(buf)
}

/// Serialize neighbor distances to an in-memory buffer (fvec format).
///
/// Same layout as [`write_distances`] but returns bytes instead of writing to a file.
fn serialize_distances(results: &[Vec<Neighbor>], k: usize) -> Result<Vec<u8>, String> {
    let row_bytes = 4 + k * 4;
    let mut buf = Vec::with_capacity(results.len() * row_bytes);
    let dim = k as i32;
    for row in results {
        buf.extend_from_slice(&dim.to_le_bytes());
        for i in 0..k {
            let dist: f32 = if i < row.len() {
                row[i].distance
            } else {
                f32::INFINITY
            };
            buf.extend_from_slice(&dist.to_le_bytes());
        }
    }
    Ok(buf)
}

/// Create a progress bar for query processing within a partition.
fn make_query_progress_bar(total: u64, ui: &veks_core::ui::UiHandle) -> ProgressHandle {
    ui.bar_with_unit(total, "queries", "queries")
}

/// Format a count with thousands separators.
fn format_count(n: usize) -> String {
    if n >= 1_000_000_000 && n % 1_000_000_000 == 0 {
        format!("{}B", n / 1_000_000_000)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 && n % 1_000_000 == 0 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 && n % 1_000 == 0 {
        format!("{}K", n / 1_000)
    } else if n >= 10_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Format byte count as human-readable size.
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * 1024 * 1024 * 1024;
    if bytes >= TIB {
        format!("{:.1} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

/// Wait for a background cache-writer thread to finish, showing a spinner
/// if the thread hasn't completed yet.
fn join_writer_with_spinner(
    handle: std::thread::JoinHandle<Result<(), String>>,
    label: &str,
    ui: &veks_core::ui::UiHandle,
) -> Result<(), String> {
    if !handle.is_finished() {
        let sp = ui.spinner(&format!("flushing {} cache write...", label));
        let result = handle.join();
        sp.finish();
        match result {
            Ok(inner) => inner,
            Err(_) => Err("partition cache writer thread panicked".to_string()),
        }
    } else {
        match handle.join() {
            Ok(inner) => inner,
            Err(_) => Err("partition cache writer thread panicked".to_string()),
        }
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    /// Generate test vectors with a fixed seed.
    fn gen_vectors(ctx: &mut StreamContext, path: &Path, dim: &str, count: &str) {
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", dim);
        opts.set("count", count);
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        let r = gen_op.execute(&opts, ctx);
        assert_eq!(r.status, Status::Ok);
    }

    /// Read raw ivec/fvec row data for comparison.
    fn read_rows(path: &Path, k: usize) -> Vec<Vec<u8>> {
        let data = std::fs::read(path).unwrap();
        let row_bytes = 4 + k * 4;
        data.chunks(row_bytes).map(|c| c.to_vec()).collect()
    }

    #[test]
    fn test_simd_distance_selection() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
            let f = simd_distance::select_distance_fn(metric);
            let d = f(&a, &b);
            assert!(d.is_finite(), "metric {:?} returned non-finite", metric);
        }
    }

    #[test]
    fn test_knn_small() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "50");
        gen_vectors(&mut ctx, &query_path, "4", "5");

        let indices_path = workspace.join("indices.ivec");
        let distances_path = workspace.join("distances.fvec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("distances", distances_path.to_string_lossy().to_string());
        opts.set("neighbors", "3");
        opts.set("metric", "L2");

        let mut knn = ComputeKnnOp;
        let result = knn.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        // indices: 5 rows x (4 + 3*4) = 5 * 16 = 80 bytes
        let idx_size = std::fs::metadata(&indices_path).unwrap().len();
        assert_eq!(idx_size, 5 * (4 + 3 * 4));

        let dist_size = std::fs::metadata(&distances_path).unwrap().len();
        assert_eq!(dist_size, 5 * (4 + 3 * 4));
    }

    #[test]
    fn test_knn_results_sorted_by_distance() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "8", "30");
        gen_vectors(&mut ctx, &query_path, "8", "3");

        let distances_path = workspace.join("distances.fvec");
        let indices_path = workspace.join("indices.ivec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("distances", distances_path.to_string_lossy().to_string());
        opts.set("neighbors", "5");
        opts.set("metric", "L2");

        let mut knn = ComputeKnnOp;
        knn.execute(&opts, &mut ctx);

        // Read distances and verify they're sorted ascending per row
        let data = std::fs::read(&distances_path).unwrap();
        let k = 5;
        let record_size = 4 + k * 4;
        for row in 0..3 {
            let mut prev_dist = f32::NEG_INFINITY;
            for col in 0..k {
                let offset = row * record_size + 4 + col * 4;
                let dist = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                assert!(
                    dist >= prev_dist,
                    "row {} distances not sorted: {} < {}",
                    row,
                    dist,
                    prev_dist
                );
                prev_dist = dist;
            }
        }
    }

    #[test]
    fn test_knn_threaded() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "20");
        gen_vectors(&mut ctx, &query_path, "4", "10");

        // Single-threaded
        let idx1 = workspace.join("idx1.ivec");
        let mut opts1 = Options::new();
        opts1.set("base", base_path.to_string_lossy().to_string());
        opts1.set("query", query_path.to_string_lossy().to_string());
        opts1.set("indices", idx1.to_string_lossy().to_string());
        opts1.set("neighbors", "3");
        opts1.set("threads", "1");
        let mut knn1 = ComputeKnnOp;
        knn1.execute(&opts1, &mut ctx);

        // Multi-threaded
        let idx2 = workspace.join("idx2.ivec");
        let mut opts2 = Options::new();
        opts2.set("base", base_path.to_string_lossy().to_string());
        opts2.set("query", query_path.to_string_lossy().to_string());
        opts2.set("indices", idx2.to_string_lossy().to_string());
        opts2.set("neighbors", "3");
        opts2.set("threads", "4");
        let mut knn2 = ComputeKnnOp;
        ctx.threads = 4;
        knn2.execute(&opts2, &mut ctx);

        let data1 = std::fs::read(&idx1).unwrap();
        let data2 = std::fs::read(&idx2).unwrap();
        assert_eq!(data1, data2, "threaded and single-threaded results differ");
    }

    #[test]
    fn test_metric_parsing() {
        assert_eq!(Metric::from_str("L2"), Some(Metric::L2));
        assert_eq!(Metric::from_str("EUCLIDEAN"), Some(Metric::L2));
        assert_eq!(Metric::from_str("cosine"), Some(Metric::Cosine));
        assert_eq!(Metric::from_str("DOT_PRODUCT"), Some(Metric::DotProduct));
        assert_eq!(Metric::from_str("L1"), Some(Metric::L1));
        assert_eq!(Metric::from_str("invalid"), None);
    }

    #[test]
    fn test_knn_partitioned() {
        // 50 base vectors, 5 queries, partition_size=10 → 5 partitions.
        // Verify partitioned results match single-partition computation.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "50");
        gen_vectors(&mut ctx, &query_path, "4", "5");

        let k = 3;

        // Single-partition reference (partition_size > base_count)
        let ref_idx = workspace.join("ref.ivec");
        let ref_dist = workspace.join("ref.fvec");
        let mut opts_ref = Options::new();
        opts_ref.set("base", base_path.to_string_lossy().to_string());
        opts_ref.set("query", query_path.to_string_lossy().to_string());
        opts_ref.set("indices", ref_idx.to_string_lossy().to_string());
        opts_ref.set("distances", ref_dist.to_string_lossy().to_string());
        opts_ref.set("neighbors", k.to_string());
        opts_ref.set("metric", "L2");
        let mut knn_ref = ComputeKnnOp;
        let r = knn_ref.execute(&opts_ref, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Partitioned computation (partition_size=10 → 5 partitions)
        let part_idx = workspace.join("part.ivec");
        let part_dist = workspace.join("part.fvec");
        let mut opts_part = Options::new();
        opts_part.set("base", base_path.to_string_lossy().to_string());
        opts_part.set("query", query_path.to_string_lossy().to_string());
        opts_part.set("indices", part_idx.to_string_lossy().to_string());
        opts_part.set("distances", part_dist.to_string_lossy().to_string());
        opts_part.set("neighbors", k.to_string());
        opts_part.set("metric", "L2");
        opts_part.set("partition_size", "10");
        ctx.step_id = "test-knn-part".to_string();
        let mut knn_part = ComputeKnnOp;
        let r = knn_part.execute(&opts_part, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Results should be identical
        let ref_idx_data = read_rows(&ref_idx, k);
        let part_idx_data = read_rows(&part_idx, k);
        assert_eq!(ref_idx_data, part_idx_data, "partitioned indices differ from reference");

        let ref_dist_data = read_rows(&ref_dist, k);
        let part_dist_data = read_rows(&part_dist, k);
        assert_eq!(ref_dist_data, part_dist_data, "partitioned distances differ from reference");

        // Verify cache files exist (5 partitions)
        let cache_dir = workspace.join(".cache");
        let cache_files: Vec<_> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(
            cache_files.len(),
            12, // 5 partitions x 2 files (ivec + fvec) + 1 full-range super-partition x 2 files
            "expected 12 cache files, found {}",
            cache_files.len()
        );
    }

    #[test]
    fn test_knn_cache_reuse() {
        // Run partitioned KNN twice; verify the second run reuses cache
        // (produces identical results without recomputing).
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "50");
        gen_vectors(&mut ctx, &query_path, "4", "5");

        let k = 3;
        ctx.step_id = "cache-reuse".to_string();

        // First run
        let idx1 = workspace.join("idx1.ivec");
        let dist1 = workspace.join("dist1.fvec");
        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", idx1.to_string_lossy().to_string());
        opts.set("distances", dist1.to_string_lossy().to_string());
        opts.set("neighbors", k.to_string());
        opts.set("metric", "L2");
        opts.set("partition_size", "10");
        let mut knn = ComputeKnnOp;
        let r = knn.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Record cache file mtimes
        let cache_dir = workspace.join(".cache");
        let mut mtimes: Vec<_> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| {
                let meta = e.metadata().unwrap();
                (e.file_name(), meta.modified().unwrap())
            })
            .collect();
        mtimes.sort_by(|a, b| a.0.cmp(&b.0));

        // Brief pause so mtime would differ if rewritten
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Second run with different output paths
        let idx2 = workspace.join("idx2.ivec");
        let dist2 = workspace.join("dist2.fvec");
        let mut opts2 = Options::new();
        opts2.set("base", base_path.to_string_lossy().to_string());
        opts2.set("query", query_path.to_string_lossy().to_string());
        opts2.set("indices", idx2.to_string_lossy().to_string());
        opts2.set("distances", dist2.to_string_lossy().to_string());
        opts2.set("neighbors", k.to_string());
        opts2.set("metric", "L2");
        opts2.set("partition_size", "10");
        let mut knn2 = ComputeKnnOp;
        let r2 = knn2.execute(&opts2, &mut ctx);
        assert_eq!(r2.status, Status::Ok);

        // Cache files should not have been rewritten
        let mut mtimes2: Vec<_> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| {
                let meta = e.metadata().unwrap();
                (e.file_name(), meta.modified().unwrap())
            })
            .collect();
        mtimes2.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(mtimes, mtimes2, "cache files were unexpectedly rewritten");

        // Results should be identical
        let data1 = std::fs::read(&idx1).unwrap();
        let data2 = std::fs::read(&idx2).unwrap();
        assert_eq!(data1, data2, "second run produced different results");
    }

    #[test]
    fn test_knn_cache_validation() {
        // Truncate a cache file and verify it's recomputed.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "30");
        gen_vectors(&mut ctx, &query_path, "4", "3");

        let k = 2;
        ctx.step_id = "cache-val".to_string();

        // First run
        let idx1 = workspace.join("idx1.ivec");
        let dist1 = workspace.join("dist1.fvec");
        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", idx1.to_string_lossy().to_string());
        opts.set("distances", dist1.to_string_lossy().to_string());
        opts.set("neighbors", k.to_string());
        opts.set("metric", "L2");
        opts.set("partition_size", "10");
        let mut knn = ComputeKnnOp;
        let r = knn.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Truncate one cache file
        let cache_dir = workspace.join(".cache");
        let first_ivec = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(".neighbors.ivec")
            })
            .unwrap();
        let ivec_path = first_ivec.path();
        std::fs::write(&ivec_path, b"truncated").unwrap();

        // Second run — should recompute the truncated partition
        let idx2 = workspace.join("idx2.ivec");
        let dist2 = workspace.join("dist2.fvec");
        let mut opts2 = Options::new();
        opts2.set("base", base_path.to_string_lossy().to_string());
        opts2.set("query", query_path.to_string_lossy().to_string());
        opts2.set("indices", idx2.to_string_lossy().to_string());
        opts2.set("distances", dist2.to_string_lossy().to_string());
        opts2.set("neighbors", k.to_string());
        opts2.set("metric", "L2");
        opts2.set("partition_size", "10");
        let mut knn2 = ComputeKnnOp;
        let r2 = knn2.execute(&opts2, &mut ctx);
        assert_eq!(r2.status, Status::Ok);

        // Results should still be correct (match original)
        let data1 = std::fs::read(&idx1).unwrap();
        let data2 = std::fs::read(&idx2).unwrap();
        assert_eq!(data1, data2, "recomputed results differ from original");
    }
}
