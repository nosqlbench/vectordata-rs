// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: sparse-sample KNN verification.
//!
//! Recomputes brute-force exact KNN for a sparse random sample of query
//! vectors and compares against stored ground-truth results. This catches
//! data corruption, byte-order issues, shuffle bugs, off-by-one errors,
//! and distance function mismatches.
//!
//! Uses the same optimized SIMD distance infrastructure and batched
//! parallel processing pattern as `compute knn`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use rand::seq::index::sample;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::Serialize;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status,
    StreamContext, render_options_table,
};
use crate::pipeline::element_type::ElementType;
use crate::pipeline::simd_distance::{self, Metric};
use super::source_window::{resolve_path, resolve_source};

/// Pipeline command: sparse-sample KNN verification.
pub struct VerifyKnnOp;

/// Returns a boxed `VerifyKnnOp` for command registration.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyKnnOp)
}

// -- Top-K heap ---------------------------------------------------------------

/// A neighbor candidate with index and distance, ordered by distance descending
/// (max-heap: worst neighbor at top, easily evicted).
#[derive(Clone, Debug)]
struct Neighbor {
    index: u32,
    distance: f32,
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

// -- Query batch size ---------------------------------------------------------

/// Number of queries processed per base-vector scan in the batched KNN loop.
const QUERY_BATCH_SIZE: usize = 256;

// -- Batched pairwise KNN (f32) -----------------------------------------------

/// Compute KNN for a batch of f32 queries against base vectors `[start, end)`.
#[inline(never)]
fn find_top_k_batch_f32(
    queries: &[&[f32]],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    batched_fn: Option<simd_distance::BatchedDistFnF32>,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    if let Some(bfn) = batched_fn {
        find_top_k_batch_transposed_f32(queries, base_reader, start, end, k, bfn, dim, results);
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
#[inline(never)]
fn find_top_k_batch_transposed_f32(
    queries: &[&[f32]],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    batched_fn: simd_distance::BatchedDistFnF32,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    use simd_distance::{SIMD_BATCH_WIDTH, TransposedBatch};

    let batch_size = queries.len();
    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..batch_size)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; batch_size];

    let mut sub_batches: Vec<TransposedBatch> = Vec::new();
    let mut sub_offsets: Vec<usize> = Vec::new();
    let mut offset = 0;
    while offset < batch_size {
        let sub_end = std::cmp::min(offset + SIMD_BATCH_WIDTH, batch_size);
        sub_batches.push(TransposedBatch::from_f32(&queries[offset..sub_end], dim));
        sub_offsets.push(offset);
        offset = sub_end;
    }

    let mut dist_buf = [0.0f32; SIMD_BATCH_WIDTH];

    for i in start..end {
        let base_vec = base_reader.get_slice(i);
        let idx = i as u32;

        for (si, sub_batch) in sub_batches.iter().enumerate() {
            batched_fn(sub_batch, base_vec, &mut dist_buf);
            let sub_offset = sub_offsets[si];

            for qi in 0..sub_batch.count() {
                let global_qi = sub_offset + qi;
                let dist = dist_buf[qi];

                if dist < thresholds[global_qi] {
                    heaps[global_qi].push(Neighbor { index: idx, distance: dist });
                    if heaps[global_qi].len() > k {
                        heaps[global_qi].pop();
                    }
                    if heaps[global_qi].len() == k {
                        thresholds[global_qi] = heaps[global_qi].peek().unwrap().distance;
                    }
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

// -- Batched pairwise KNN (f16) -----------------------------------------------

/// Compute KNN for a batch of f16 queries against base vectors `[start, end)`.
#[inline(never)]
fn find_top_k_batch_f16(
    queries: &[&[half::f16]],
    base_reader: &MmapVectorReader<half::f16>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    batched_fn: Option<simd_distance::BatchedDistFnF32>,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    if let Some(bfn) = batched_fn {
        find_top_k_batch_transposed_f16(queries, base_reader, start, end, k, bfn, dim, results);
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

/// Transposed SIMD batch processing for f16 (converts base to f32 once).
#[inline(never)]
fn find_top_k_batch_transposed_f16(
    queries: &[&[half::f16]],
    base_reader: &MmapVectorReader<half::f16>,
    start: usize,
    end: usize,
    k: usize,
    batched_fn: simd_distance::BatchedDistFnF32,
    dim: usize,
    results: &mut [Vec<Neighbor>],
) {
    use simd_distance::{SIMD_BATCH_WIDTH, TransposedBatch};

    let batch_size = queries.len();
    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..batch_size)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; batch_size];

    let mut sub_batches: Vec<TransposedBatch> = Vec::new();
    let mut sub_offsets: Vec<usize> = Vec::new();
    let mut offset = 0;
    while offset < batch_size {
        let sub_end = std::cmp::min(offset + SIMD_BATCH_WIDTH, batch_size);
        sub_batches.push(TransposedBatch::from_f16(&queries[offset..sub_end], dim));
        sub_offsets.push(offset);
        offset = sub_end;
    }

    let mut base_f32 = vec![0.0f32; dim];
    let mut dist_buf = [0.0f32; SIMD_BATCH_WIDTH];

    for i in start..end {
        let base_f16 = base_reader.get_slice(i);
        let idx = i as u32;

        simd_distance::convert_f16_to_f32_bulk(base_f16, &mut base_f32);

        for (si, sub_batch) in sub_batches.iter().enumerate() {
            batched_fn(sub_batch, &base_f32, &mut dist_buf);
            let sub_offset = sub_offsets[si];

            for qi in 0..sub_batch.count() {
                let global_qi = sub_offset + qi;
                let dist = dist_buf[qi];

                if dist < thresholds[global_qi] {
                    heaps[global_qi].push(Neighbor { index: idx, distance: dist });
                    if heaps[global_qi].len() > k {
                        heaps[global_qi].pop();
                    }
                    if heaps[global_qi].len() == k {
                        thresholds[global_qi] = heaps[global_qi].peek().unwrap().distance;
                    }
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

// -- Batched pairwise KNN (f64) -----------------------------------------------

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
        let mut v: Vec<Neighbor> = heap.into_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results[qi] = v;
    }
}

// -- Parallel verification per element type -----------------------------------

/// Verify sampled queries using f32 vectors with parallel batched processing.
#[allow(clippy::too_many_arguments)]
fn verify_f32(
    base_reader: &Arc<MmapVectorReader<f32>>,
    query_reader: &MmapVectorReader<f32>,
    indices_reader: &MmapVectorReader<i32>,
    sampled_indices: &[usize],
    base_offset: usize,
    base_end: usize,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    phi: f32,
    threads: usize,
    ui: &crate::ui::UiHandle,
) -> Vec<QueryVerification> {
    let batched_fn = simd_distance::select_batched_fn_f32(metric);
    let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&**base_reader);
    let sample_count = sampled_indices.len();

    let pb = ui.bar_with_unit(sample_count as u64, "verify", "queries");
    let pb_ref = &pb;

    // Pre-collect query slices and ground truth for sampled indices
    let queries_data: Vec<(usize, Vec<i32>)> = sampled_indices
        .iter()
        .map(|&qi| {
            let gt = indices_reader.get(qi).unwrap_or_default();
            (qi, gt)
        })
        .collect();

    // Compute recomputed KNN for all sampled queries in parallel batches
    let mut recomputed: Vec<Vec<Neighbor>> = (0..sample_count).map(|_| Vec::new()).collect();

    if threads > 1 && sample_count > 1 {
        let effective_threads = std::cmp::min(threads, sample_count);
        let chunk_size = (sample_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            recomputed.chunks_mut(chunk_size).collect();

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
                            .map(|i| {
                                let qi = sampled_indices[chunk_start + offset + i];
                                query_reader.get_slice(qi)
                            })
                            .collect();

                        find_top_k_batch_f32(
                            &queries,
                            &base_ref,
                            base_offset,
                            base_end,
                            k,
                            dist_fn,
                            batched_fn,
                            dim,
                            &mut chunk[offset..batch_end],
                        );

                        pb_ref.inc(batch_size as u64);
                        offset = batch_end;
                    }
                });
            }
        });
    } else {
        let mut offset = 0;
        while offset < sample_count {
            let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, sample_count);
            let batch_size = batch_end - offset;

            let queries: Vec<&[f32]> = (0..batch_size)
                .map(|i| {
                    let qi = sampled_indices[offset + i];
                    query_reader.get_slice(qi)
                })
                .collect();

            find_top_k_batch_f32(
                &queries,
                base_reader,
                base_offset,
                base_end,
                k,
                dist_fn,
                batched_fn,
                dim,
                &mut recomputed[offset..batch_end],
            );

            pb_ref.inc(batch_size as u64);
            offset = batch_end;
        }
    }
    pb.finish();

    // Compare recomputed results against ground truth
    compare_results_generic(&recomputed, &queries_data, base_offset, k, phi)
}

/// Compare recomputed KNN results against ground truth with tie detection.
///
/// Verify sampled queries using f16 vectors with parallel batched processing.
#[allow(clippy::too_many_arguments)]
fn verify_f16(
    base_reader: &Arc<MmapVectorReader<half::f16>>,
    query_reader: &MmapVectorReader<half::f16>,
    indices_reader: &MmapVectorReader<i32>,
    sampled_indices: &[usize],
    base_offset: usize,
    base_end: usize,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    phi: f32,
    threads: usize,
    ui: &crate::ui::UiHandle,
) -> Vec<QueryVerification> {
    let batched_fn = simd_distance::select_batched_fn_f32(metric);
    let dim = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&**base_reader);
    let sample_count = sampled_indices.len();

    let pb = ui.bar_with_unit(sample_count as u64, "verify", "queries");
    let pb_ref = &pb;

    let queries_data: Vec<(usize, Vec<i32>)> = sampled_indices
        .iter()
        .map(|&qi| {
            let gt = indices_reader.get(qi).unwrap_or_default();
            (qi, gt)
        })
        .collect();

    let mut recomputed: Vec<Vec<Neighbor>> = (0..sample_count).map(|_| Vec::new()).collect();

    if threads > 1 && sample_count > 1 {
        let effective_threads = std::cmp::min(threads, sample_count);
        let chunk_size = (sample_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            recomputed.chunks_mut(chunk_size).collect();

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
                            .map(|i| {
                                let qi = sampled_indices[chunk_start + offset + i];
                                query_reader.get_slice(qi)
                            })
                            .collect();

                        find_top_k_batch_f16(
                            &queries,
                            &base_ref,
                            base_offset,
                            base_end,
                            k,
                            dist_fn,
                            batched_fn,
                            dim,
                            &mut chunk[offset..batch_end],
                        );

                        pb_ref.inc(batch_size as u64);
                        offset = batch_end;
                    }
                });
            }
        });
    } else {
        let mut offset = 0;
        while offset < sample_count {
            let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, sample_count);
            let batch_size = batch_end - offset;

            let queries: Vec<&[half::f16]> = (0..batch_size)
                .map(|i| {
                    let qi = sampled_indices[offset + i];
                    query_reader.get_slice(qi)
                })
                .collect();

            find_top_k_batch_f16(
                &queries,
                base_reader,
                base_offset,
                base_end,
                k,
                dist_fn,
                batched_fn,
                dim,
                &mut recomputed[offset..batch_end],
            );

            pb_ref.inc(batch_size as u64);
            offset = batch_end;
        }
    }
    pb.finish();

    compare_results_generic(&recomputed, &queries_data, base_offset, k, phi)
}

/// Verify sampled queries using f64 vectors with parallel batched processing.
#[allow(clippy::too_many_arguments)]
fn verify_f64(
    base_reader: &Arc<MmapVectorReader<f64>>,
    query_reader: &MmapVectorReader<f64>,
    indices_reader: &MmapVectorReader<i32>,
    sampled_indices: &[usize],
    base_offset: usize,
    base_end: usize,
    k: usize,
    _metric: Metric,
    dist_fn: fn(&[f64], &[f64]) -> f32,
    phi: f32,
    threads: usize,
    ui: &crate::ui::UiHandle,
) -> Vec<QueryVerification> {
    let sample_count = sampled_indices.len();

    let pb = ui.bar_with_unit(sample_count as u64, "verify", "queries");
    let pb_ref = &pb;

    let queries_data: Vec<(usize, Vec<i32>)> = sampled_indices
        .iter()
        .map(|&qi| {
            let gt = indices_reader.get(qi).unwrap_or_default();
            (qi, gt)
        })
        .collect();

    let mut recomputed: Vec<Vec<Neighbor>> = (0..sample_count).map(|_| Vec::new()).collect();

    if threads > 1 && sample_count > 1 {
        let effective_threads = std::cmp::min(threads, sample_count);
        let chunk_size = (sample_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            recomputed.chunks_mut(chunk_size).collect();

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
                            .map(|i| {
                                let qi = sampled_indices[chunk_start + offset + i];
                                query_reader.get_slice(qi)
                            })
                            .collect();

                        find_top_k_batch_pairwise_f64(
                            &queries,
                            &base_ref,
                            base_offset,
                            base_end,
                            k,
                            dist_fn,
                            &mut chunk[offset..batch_end],
                        );

                        pb_ref.inc(batch_size as u64);
                        offset = batch_end;
                    }
                });
            }
        });
    } else {
        let mut offset = 0;
        while offset < sample_count {
            let batch_end = std::cmp::min(offset + QUERY_BATCH_SIZE, sample_count);
            let batch_size = batch_end - offset;

            let queries: Vec<&[f64]> = (0..batch_size)
                .map(|i| {
                    let qi = sampled_indices[offset + i];
                    query_reader.get_slice(qi)
                })
                .collect();

            find_top_k_batch_pairwise_f64(
                &queries,
                base_reader,
                base_offset,
                base_end,
                k,
                dist_fn,
                &mut recomputed[offset..batch_end],
            );

            pb_ref.inc(batch_size as u64);
            offset = batch_end;
        }
    }
    pb.finish();

    compare_results_generic(&recomputed, &queries_data, base_offset, k, phi)
}

/// Generic comparison that does not require the base reader (tie detection
/// uses the recomputed result's distance profile).
fn compare_results_generic(
    recomputed: &[Vec<Neighbor>],
    queries_data: &[(usize, Vec<i32>)],
    base_offset: usize,
    k: usize,
    phi: f32,
) -> Vec<QueryVerification> {
    recomputed
        .iter()
        .zip(queries_data.iter())
        .map(|(true_neighbors, (qi, provided_indices))| {
            let true_set: HashSet<u32> = true_neighbors.iter().map(|n| n.index).collect();

            let matching = provided_indices
                .iter()
                .filter(|&&idx| idx >= 0 && true_set.contains(&((idx as usize + base_offset) as u32)))
                .count();

            if matching == k {
                return QueryVerification {
                    query_index: *qi,
                    status: VerifyStatus::Pass,
                    matching_count: k,
                    tie_adjusted_count: k,
                    mismatched_indices: vec![],
                };
            }

            // Check for distance ties at the k-th boundary
            let worst_true_dist = true_neighbors
                .last()
                .map(|n| n.distance)
                .unwrap_or(0.0);

            let tie_count_at_boundary = true_neighbors
                .iter()
                .filter(|n| (n.distance - worst_true_dist).abs() <= phi)
                .count();

            let mut tie_adjusted = matching;
            let mut mismatched = Vec::new();
            for &idx in provided_indices {
                if idx >= 0 && !true_set.contains(&((idx as usize + base_offset) as u32)) {
                    mismatched.push(idx);
                    if tie_count_at_boundary > 1 {
                        tie_adjusted += 1;
                    }
                }
            }

            let status = if tie_adjusted >= k {
                VerifyStatus::Tie
            } else {
                VerifyStatus::Fail
            };

            QueryVerification {
                query_index: *qi,
                status,
                matching_count: matching,
                tie_adjusted_count: tie_adjusted,
                mismatched_indices: mismatched,
            }
        })
        .collect()
}

// -- Verification report types ------------------------------------------------

/// Status of a single query verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum VerifyStatus {
    Pass,
    Tie,
    Fail,
}

/// Verification result for a single query.
#[derive(Debug, Clone, Serialize)]
struct QueryVerification {
    query_index: usize,
    status: VerifyStatus,
    matching_count: usize,
    tie_adjusted_count: usize,
    mismatched_indices: Vec<i32>,
}

/// JSON report written to the output path.
#[derive(Debug, Clone, Serialize)]
struct VerificationReport {
    sample_count: usize,
    pass_count: usize,
    tie_count: usize,
    fail_count: usize,
    recall_at_k: f64,
    k: usize,
    metric: String,
    phi: f32,
    seed: u64,
    elapsed_secs: f64,
    failures: Vec<QueryVerification>,
}

// -- CommandOp impl -----------------------------------------------------------

impl CommandOp for VerifyKnnOp {
    fn command_path(&self) -> &str {
        "verify knn-groundtruth"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Sparse-sample KNN verification against ground truth".into(),
            body: format!(
                "# verify knn-groundtruth\n\n\
                Sparse-sample KNN verification against ground truth.\n\n\
                ## Description\n\n\
                Recomputes brute-force exact KNN for a sparse random sample of query \
                vectors and compares against stored ground-truth results. This catches \
                data corruption, byte-order issues, shuffle bugs, off-by-one errors, \
                and distance function mismatches.\n\n\
                ## Algorithm\n\n\
                1. **Sample selection**: Choose `sample` (default: 100) query indices \
                uniformly at random using a deterministic seed.\n\
                2. **Parallel brute-force recomputation**: For each sampled query, \
                recompute exact top-k nearest neighbors against the full base vector \
                set using SIMD-accelerated distance functions and governor-limited \
                thread pool.\n\
                3. **Comparison**: Compare recomputed top-k index sets against stored \
                indices. Distance ties within `phi` tolerance are counted as passes.\n\
                4. **Report**: Write `verify_knn.json` with sample_count, pass_count, \
                fail_count, tie_count, recall@k, per-failure detail, and elapsed time.\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description: "Base vector mmap and distance recomputation".into(),
                adjustable: false,
            },
            ResourceDesc {
                name: "threads".into(),
                description: "Parallel distance recomputation".into(),
                adjustable: true,
            },
            ResourceDesc {
                name: "readahead".into(),
                description: "Sequential read prefetch".into(),
                adjustable: false,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // -- Parse required options -------------------------------------------

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
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        // -- Parse optional options -------------------------------------------

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

        let sample_count: usize = options
            .get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        let seed: u64 = options
            .get("seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);

        let phi: f32 = options
            .get("phi")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.001);

        let threads: usize = options
            .get("threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                ctx.governor.current_or("threads", ctx.threads as u64) as usize
            });

        // -- Resolve paths ----------------------------------------------------

        let base_source = match resolve_source(base_str, &ctx.workspace) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Create output directory if needed
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(
                        format!("failed to create output directory: {}", e),
                        start,
                    );
                }
            }
        }

        // -- Open indices reader (type-independent) ---------------------------

        let indices_reader = match MmapVectorReader::<i32>::open_ivec(&indices_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open indices {}: {}", indices_path.display(), e),
                    start,
                )
            }
        };

        let indices_count =
            <MmapVectorReader<i32> as VectorReader<i32>>::count(&indices_reader);
        let k = <MmapVectorReader<i32> as VectorReader<i32>>::dim(&indices_reader);

        // -- Detect element type and dispatch ---------------------------------

        let etype = match ElementType::from_path(&base_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        // -- Sample query indices deterministically ---------------------------

        let effective_sample = std::cmp::min(sample_count, indices_count);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let sampled: Vec<usize> = if effective_sample >= indices_count {
            (0..indices_count).collect()
        } else {
            sample(&mut rng, indices_count, effective_sample)
                .into_vec()
        };

        ctx.ui.log(&format!(
            "verify knn: {} sampled queries out of {}, k={}, metric={:?}, phi={}, seed={}, threads={}, type={}",
            sampled.len(), indices_count, k, metric, phi, seed, threads, etype
        ));

        // -- Dispatch by element type -----------------------------------------

        let verifications = match etype {
            ElementType::F32 => {
                let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
                    Ok(r) => Arc::new(r),
                    Err(e) => {
                        return error_result(
                            format!("failed to open base {}: {}", base_path.display(), e),
                            start,
                        )
                    }
                };
                let query_reader = match MmapVectorReader::<f32>::open_fvec(&query_path) {
                    Ok(r) => r,
                    Err(e) => {
                        return error_result(
                            format!("failed to open query {}: {}", query_path.display(), e),
                            start,
                        )
                    }
                };

                let file_count =
                    <MmapVectorReader<f32> as VectorReader<f32>>::count(&*base_reader);
                let (base_offset, base_end) =
                    base_source.effective_range(file_count);

                let dist_fn = simd_distance::select_distance_fn(metric);

                verify_f32(
                    &base_reader,
                    &query_reader,
                    &indices_reader,
                    &sampled,
                    base_offset,
                    base_end,
                    k,
                    metric,
                    dist_fn,
                    phi,
                    threads,
                    &ctx.ui,
                )
            }
            ElementType::F16 => {
                let base_reader =
                    match MmapVectorReader::<half::f16>::open_mvec(&base_path) {
                        Ok(r) => Arc::new(r),
                        Err(e) => {
                            return error_result(
                                format!(
                                    "failed to open base {}: {}",
                                    base_path.display(),
                                    e
                                ),
                                start,
                            )
                        }
                    };
                let query_reader =
                    match MmapVectorReader::<half::f16>::open_mvec(&query_path) {
                        Ok(r) => r,
                        Err(e) => {
                            return error_result(
                                format!(
                                    "failed to open query {}: {}",
                                    query_path.display(),
                                    e
                                ),
                                start,
                            )
                        }
                    };

                let file_count =
                    <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(
                        &*base_reader,
                    );
                let (base_offset, base_end) =
                    base_source.effective_range(file_count);

                let dist_fn = simd_distance::select_distance_fn_f16(metric);

                verify_f16(
                    &base_reader,
                    &query_reader,
                    &indices_reader,
                    &sampled,
                    base_offset,
                    base_end,
                    k,
                    metric,
                    dist_fn,
                    phi,
                    threads,
                    &ctx.ui,
                )
            }
            ElementType::F64 => {
                let base_reader = match MmapVectorReader::<f64>::open_dvec(&base_path) {
                    Ok(r) => Arc::new(r),
                    Err(e) => {
                        return error_result(
                            format!("failed to open base {}: {}", base_path.display(), e),
                            start,
                        )
                    }
                };
                let query_reader = match MmapVectorReader::<f64>::open_dvec(&query_path) {
                    Ok(r) => r,
                    Err(e) => {
                        return error_result(
                            format!("failed to open query {}: {}", query_path.display(), e),
                            start,
                        )
                    }
                };

                let file_count =
                    <MmapVectorReader<f64> as VectorReader<f64>>::count(&*base_reader);
                let (base_offset, base_end) =
                    base_source.effective_range(file_count);

                let dist_fn = simd_distance::select_distance_fn_f64(metric);

                verify_f64(
                    &base_reader,
                    &query_reader,
                    &indices_reader,
                    &sampled,
                    base_offset,
                    base_end,
                    k,
                    metric,
                    dist_fn,
                    phi,
                    threads,
                    &ctx.ui,
                )
            }
            _ => {
                return error_result(
                    format!(
                        "unsupported element type {:?} for verify knn (use fvec, mvec, or dvec)",
                        etype
                    ),
                    start,
                )
            }
        };

        // -- Aggregate results ------------------------------------------------

        let pass_count = verifications
            .iter()
            .filter(|v| v.status == VerifyStatus::Pass)
            .count();
        let tie_count = verifications
            .iter()
            .filter(|v| v.status == VerifyStatus::Tie)
            .count();
        let fail_count = verifications
            .iter()
            .filter(|v| v.status == VerifyStatus::Fail)
            .count();
        let total = verifications.len();
        let recall = if total > 0 {
            (pass_count + tie_count) as f64 / total as f64
        } else {
            1.0
        };

        let failures: Vec<QueryVerification> = verifications
            .into_iter()
            .filter(|v| v.status == VerifyStatus::Fail)
            .collect();

        for f in &failures {
            ctx.ui.log(&format!(
                "  FAIL query {}: {}/{} matching ({} with ties), mismatched: {:?}",
                f.query_index, f.matching_count, k, f.tie_adjusted_count, f.mismatched_indices
            ));
        }

        let elapsed_secs = start.elapsed().as_secs_f64();

        // -- Write JSON report ------------------------------------------------

        let report = VerificationReport {
            sample_count: total,
            pass_count,
            tie_count,
            fail_count,
            recall_at_k: recall,
            k,
            metric: format!("{:?}", metric),
            phi,
            seed,
            elapsed_secs,
            failures,
        };

        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                use crate::pipeline::atomic_write::AtomicWriter;
                use std::io::Write;
                let write_result = AtomicWriter::new(&output_path)
                    .and_then(|mut w| { w.write_all(json.as_bytes())?; w.finish() });
                if let Err(e) = write_result {
                    return error_result(
                        format!("failed to write report {}: {}", output_path.display(), e),
                        start,
                    );
                }
            }
            Err(e) => {
                return error_result(
                    format!("failed to serialize report: {}", e),
                    start,
                );
            }
        }

        ctx.ui.log(&format!(
            "verify knn: {}/{} pass, {}/{} tie, {}/{} fail, recall@{}={:.4}",
            pass_count, total, tie_count, total, fail_count, total, k, recall
        ));

        let status = if fail_count == 0 {
            Status::Ok
        } else {
            Status::Error
        };

        CommandResult {
            status,
            message: format!(
                "verified {}/{} sampled queries: {} pass, {} tie, {} fail (k={}, metric={:?}, recall@k={:.4})",
                total, indices_count, pass_count, tie_count, fail_count, k, metric, recall
            ),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> crate::pipeline::command::ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query", "indices", "distances"],
            &["output"],
        )
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
                description: "Precomputed neighbor indices (ivec)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "distances".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Precomputed neighbor distances (fvec, optional)".to_string(),
                role: OptionRole::Input,
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
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("100".to_string()),
                description: "Number of query vectors to spot-check".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "Random seed for deterministic sample selection".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "phi".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.001".to_string()),
                description: "Floating-point tolerance for distance tie detection".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output verification report (JSON)".to_string(),
                role: OptionRole::Output,
            },
        ]
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
