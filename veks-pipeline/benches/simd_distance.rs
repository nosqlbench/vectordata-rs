// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Microbenchmarks for SIMD distance kernels and KNN inner loop variants.
//!
//! Run all:          cargo bench -p veks-pipeline --bench simd_distance
//! Run specific:     cargo bench -p veks-pipeline --bench simd_distance -- "kernel_plus_heap"
//! List benchmarks:  cargo bench -p veks-pipeline --bench simd_distance -- --list

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use veks_pipeline::pipeline::simd_distance::{
    self, Metric, TransposedBatch,
    SIMD_BATCH_WIDTH,
    select_distance_fn, select_distance_fn_f16,
    select_batched_fn_f32, select_batched_fn_f16,
    convert_f16_to_f32_bulk,
};

// ═══════════════════════════════════════════════════════════════════════════
// Neighbor types for top-k benchmarks
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct Neighbor { index: u32, distance: f32 }

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool { self.distance == other.distance }
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Data generators
// ═══════════════════════════════════════════════════════════════════════════

fn xorshift(rng: &mut u64) -> f32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    ((*rng as f32) / (u64::MAX as f32)) * 2.0 - 1.0
}

fn random_f32_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = seed;
    (0..n).map(|_| (0..dim).map(|_| xorshift(&mut rng)).collect()).collect()
}

fn random_f16_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<half::f16>> {
    random_f32_vectors(n, dim, seed).into_iter()
        .map(|v| v.into_iter().map(half::f16::from_f32).collect())
        .collect()
}

fn normalize_f32(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in v.iter_mut() { *x /= norm; } }
}

fn random_normalized_f32(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut vecs = random_f32_vectors(n, dim, seed);
    for v in &mut vecs { normalize_f32(v); }
    vecs
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Per-pair distance benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn bench_pairwise(c: &mut Criterion) {
    let dims = [128, 512, 768, 1024];
    let metrics = [("L2", Metric::L2), ("Cosine", Metric::Cosine),
                   ("DotProduct", Metric::DotProduct), ("L1", Metric::L1)];

    let mut group = c.benchmark_group("pairwise_f32");
    for &dim in &dims {
        let a = random_f32_vectors(1, dim, 42)[0].clone();
        let b = random_f32_vectors(1, dim, 99)[0].clone();
        for &(name, metric) in &metrics {
            let dist_fn = select_distance_fn(metric);
            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(BenchmarkId::new(name, dim), &dim, |bench, _| {
                bench.iter(|| black_box(dist_fn(black_box(&a), black_box(&b))));
            });
        }
    }
    group.finish();

    let mut group = c.benchmark_group("pairwise_f16");
    for &dim in &dims {
        let a = random_f16_vectors(1, dim, 42)[0].clone();
        let b = random_f16_vectors(1, dim, 99)[0].clone();
        for &(name, metric) in &metrics {
            let dist_fn = select_distance_fn_f16(metric);
            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(BenchmarkId::new(name, dim), &dim, |bench, _| {
                bench.iter(|| black_box(dist_fn(black_box(&a), black_box(&b))));
            });
        }
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Batched kernel benchmarks (single call, 16 queries)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_batched(c: &mut Criterion) {
    let dims = [128, 512, 768, 1024];
    let metrics = [("L2", Metric::L2), ("Cosine", Metric::Cosine),
                   ("DotProduct", Metric::DotProduct), ("L1", Metric::L1)];

    let mut group = c.benchmark_group("batched_f32");
    for &dim in &dims {
        let queries = random_f32_vectors(16, dim, 42);
        let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
        let batch = TransposedBatch::from_f32(&query_refs, dim);
        let base = random_f32_vectors(1, dim, 99)[0].clone();
        let mut out = [0.0f32; 16];

        for &(name, metric) in &metrics {
            if let Some(bfn) = select_batched_fn_f32(metric) {
                group.throughput(Throughput::Elements((dim * 16) as u64));
                group.bench_with_input(BenchmarkId::new(name, dim), &dim, |bench, _| {
                    bench.iter(|| bfn(black_box(&batch), black_box(&base), black_box(&mut out)));
                });
            }
        }
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. f16→f32 conversion benchmark
// ═══════════════════════════════════════════════════════════════════════════

fn bench_convert_f16_to_f32(c: &mut Criterion) {
    let dims = [128, 512, 768, 1024];
    let mut group = c.benchmark_group("convert_f16_to_f32");
    for &dim in &dims {
        let src = random_f16_vectors(1, dim, 42)[0].clone();
        let mut dst = vec![0.0f32; dim];
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| convert_f16_to_f32_bulk(black_box(&src), black_box(&mut dst)));
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. OPTIMIZATION A: Sorted array top-k vs BinaryHeap
// ═══════════════════════════════════════════════════════════════════════════

/// Sorted array top-k: maintains a sorted Vec of at most k elements.
/// For small k (≤128), insertion sort into a sorted array is faster than
/// BinaryHeap because of better cache behavior and fewer pointer chases.
struct SortedTopK {
    items: Vec<Neighbor>,
    k: usize,
    threshold: f32,
}

impl SortedTopK {
    fn new(k: usize) -> Self {
        SortedTopK { items: Vec::with_capacity(k), k, threshold: f32::INFINITY }
    }

    #[inline(always)]
    fn push(&mut self, n: Neighbor) {
        if n.distance >= self.threshold && self.items.len() == self.k {
            return;
        }
        // Binary search for insertion point (sorted ascending by distance)
        let pos = self.items.partition_point(|x| x.distance < n.distance);
        if self.items.len() < self.k {
            self.items.insert(pos, n);
        } else {
            // Full: only insert if better than worst (last element)
            if pos < self.k {
                self.items.pop(); // remove worst
                self.items.insert(pos, n);
            }
        }
        if self.items.len() == self.k {
            self.threshold = self.items.last().unwrap().distance;
        }
    }

    fn into_sorted(self) -> Vec<Neighbor> { self.items }
    fn threshold(&self) -> f32 { self.threshold }
}

fn bench_topk_structures(c: &mut Criterion) {
    let dim = 512;
    let k = 100;
    let base_count = 100_000;

    let queries = random_normalized_f32(16, dim, 42);
    let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
    let batch = TransposedBatch::from_f32(&query_refs, dim);
    let base_data: Vec<f32> = random_normalized_f32(base_count, dim, 99)
        .into_iter().flatten().collect();
    let bfn = select_batched_fn_f32(Metric::DotProduct).unwrap();

    let mut group = c.benchmark_group("topk_structure");
    group.sample_size(10);
    group.throughput(Throughput::Elements((base_count * 16) as u64));

    // A: BinaryHeap (current)
    group.bench_function("binary_heap", |bench| {
        let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..16)
            .map(|_| BinaryHeap::with_capacity(k + 1)).collect();
        let mut thresholds = [f32::INFINITY; 16];
        let mut dist_buf = [0.0f32; 16];

        bench.iter(|| {
            for h in heaps.iter_mut() { h.clear(); }
            thresholds.fill(f32::INFINITY);
            for i in 0..base_count {
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch), black_box(base_vec), &mut dist_buf);
                let idx = i as u32;
                for qi in 0..16 {
                    let dist = dist_buf[qi];
                    if dist < thresholds[qi] {
                        heaps[qi].push(Neighbor { index: idx, distance: dist });
                        if heaps[qi].len() > k { heaps[qi].pop(); }
                        if heaps[qi].len() == k {
                            thresholds[qi] = heaps[qi].peek().unwrap().distance;
                        }
                    }
                }
            }
            black_box(&heaps);
        });
    });

    // B: Sorted array
    group.bench_function("sorted_array", |bench| {
        let mut topks: Vec<SortedTopK> = (0..16).map(|_| SortedTopK::new(k)).collect();
        let mut dist_buf = [0.0f32; 16];

        bench.iter(|| {
            topks.iter_mut().for_each(|t| *t = SortedTopK::new(k));
            for i in 0..base_count {
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch), black_box(base_vec), &mut dist_buf);
                let idx = i as u32;
                for qi in 0..16 {
                    let dist = dist_buf[qi];
                    if dist < topks[qi].threshold() {
                        topks[qi].push(Neighbor { index: idx, distance: dist });
                    }
                }
            }
            black_box(&topks);
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. OPTIMIZATION B: Software prefetch
// ═══════════════════════════════════════════════════════════════════════════

fn bench_prefetch(c: &mut Criterion) {
    let dim = 512;
    let base_count = 100_000;

    let queries = random_normalized_f32(16, dim, 42);
    let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
    let batch = TransposedBatch::from_f32(&query_refs, dim);
    let base_data: Vec<f32> = random_normalized_f32(base_count, dim, 99)
        .into_iter().flatten().collect();
    let bfn = select_batched_fn_f32(Metric::DotProduct).unwrap();

    let mut group = c.benchmark_group("prefetch");
    group.sample_size(10);
    group.throughput(Throughput::Elements((base_count * 16) as u64));

    // Without prefetch
    group.bench_function("no_prefetch", |bench| {
        let mut dist_buf = [0.0f32; 16];
        bench.iter(|| {
            for i in 0..base_count {
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch), black_box(base_vec), &mut dist_buf);
                black_box(&dist_buf);
            }
        });
    });

    // With software prefetch (2 vectors ahead)
    group.bench_function("prefetch_2", |bench| {
        let mut dist_buf = [0.0f32; 16];
        bench.iter(|| {
            for i in 0..base_count {
                // Prefetch 2 vectors ahead
                if i + 2 < base_count {
                    let prefetch_ptr = base_data[(i + 2) * dim..].as_ptr();
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            prefetch_ptr as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch), black_box(base_vec), &mut dist_buf);
                black_box(&dist_buf);
            }
        });
    });

    // With software prefetch (4 vectors ahead)
    group.bench_function("prefetch_4", |bench| {
        let mut dist_buf = [0.0f32; 16];
        bench.iter(|| {
            for i in 0..base_count {
                if i + 4 < base_count {
                    let prefetch_ptr = base_data[(i + 4) * dim..].as_ptr();
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            prefetch_ptr as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch), black_box(base_vec), &mut dist_buf);
                black_box(&dist_buf);
            }
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. OPTIMIZATION C: Double-wide batch (32 queries per kernel call)
//    Uses two zmm accumulators in parallel to exploit dual FMA pipes.
// ═══════════════════════════════════════════════════════════════════════════

fn bench_double_wide(c: &mut Criterion) {
    let dim = 512;
    let base_count = 100_000;

    let bfn = select_batched_fn_f32(Metric::DotProduct).unwrap();

    // 16-wide: one TransposedBatch
    let queries16 = random_normalized_f32(16, dim, 42);
    let refs16: Vec<&[f32]> = queries16.iter().map(|v| v.as_slice()).collect();
    let batch16 = TransposedBatch::from_f32(&refs16, dim);

    // 32-wide: two TransposedBatches processed per base vector
    let queries32 = random_normalized_f32(32, dim, 42);
    let refs32a: Vec<&[f32]> = queries32[..16].iter().map(|v| v.as_slice()).collect();
    let refs32b: Vec<&[f32]> = queries32[16..].iter().map(|v| v.as_slice()).collect();
    let batch32a = TransposedBatch::from_f32(&refs32a, dim);
    let batch32b = TransposedBatch::from_f32(&refs32b, dim);

    let base_data: Vec<f32> = random_normalized_f32(base_count, dim, 99)
        .into_iter().flatten().collect();

    let mut group = c.benchmark_group("batch_width");
    group.sample_size(10);

    // 16-wide scan
    group.throughput(Throughput::Elements((base_count * 16) as u64));
    group.bench_function("16_wide", |bench| {
        let mut out = [0.0f32; 16];
        bench.iter(|| {
            for i in 0..base_count {
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch16), black_box(base_vec), &mut out);
            }
            black_box(&out);
        });
    });

    // 32-wide scan (two 16-wide calls per base vector)
    group.throughput(Throughput::Elements((base_count * 32) as u64));
    group.bench_function("32_wide_2x16", |bench| {
        let mut out_a = [0.0f32; 16];
        let mut out_b = [0.0f32; 16];
        bench.iter(|| {
            for i in 0..base_count {
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch32a), black_box(base_vec), &mut out_a);
                bfn(black_box(&batch32b), black_box(base_vec), &mut out_b);
            }
            black_box((&out_a, &out_b));
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// 6b. True dual-accumulator 32-wide kernel benchmark
// ═══════════════════════════════════════════════════════════════════════════

/// A 32-wide transposed batch: two groups of 16 queries.
struct DualBatch {
    data_a: Vec<f32>, // first 16 queries, dim-major
    data_b: Vec<f32>, // second 16 queries, dim-major
    dim: usize,
}

impl DualBatch {
    fn from_f32(queries: &[&[f32]], dim: usize) -> Self {
        assert!(queries.len() <= 32);
        let count_a = std::cmp::min(queries.len(), 16);
        let count_b = queries.len().saturating_sub(16);
        let mut data_a = vec![0.0f32; dim * 16];
        let mut data_b = vec![0.0f32; dim * 16];
        for (qi, q) in queries[..count_a].iter().enumerate() {
            for d in 0..dim { data_a[d * 16 + qi] = q[d]; }
        }
        for (qi, q) in queries[count_a..].iter().enumerate() {
            for d in 0..dim { data_b[d * 16 + qi] = q[d]; }
        }
        DualBatch { data_a, data_b, dim }
    }
}

/// Dual-accumulator neg-dot kernel: 32 queries per base vector, 2 FMA ports.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dual_neg_dot_f32_avx512(
    batch: &DualBatch, base: &[f32], out: &mut [f32; 32],
) {
    use std::arch::x86_64::*;
    let dim = batch.dim;
    let a_ptr = batch.data_a.as_ptr();
    let b_ptr = batch.data_b.as_ptr();
    let base_ptr = base.as_ptr();

    let mut acc_a = _mm512_setzero_ps();
    let mut acc_b = _mm512_setzero_ps();

    for d in 0..dim {
        let bval = _mm512_set1_ps(*base_ptr.add(d));
        let qa = _mm512_loadu_ps(a_ptr.add(d * 16));
        let qb = _mm512_loadu_ps(b_ptr.add(d * 16));
        // Two independent FMAs — hits both FMA ports
        acc_a = _mm512_fmadd_ps(bval, qa, acc_a);
        acc_b = _mm512_fmadd_ps(bval, qb, acc_b);
    }

    let zero = _mm512_setzero_ps();
    _mm512_storeu_ps(out.as_mut_ptr(), _mm512_sub_ps(zero, acc_a));
    _mm512_storeu_ps(out.as_mut_ptr().add(16), _mm512_sub_ps(zero, acc_b));
}

fn bench_dual_accumulator(c: &mut Criterion) {
    let dim = 512;
    let base_count = 100_000;
    let bfn = select_batched_fn_f32(Metric::DotProduct).unwrap();

    let queries = random_normalized_f32(32, dim, 42);
    let refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
    let refs16a: Vec<&[f32]> = queries[..16].iter().map(|v| v.as_slice()).collect();
    let refs16b: Vec<&[f32]> = queries[16..].iter().map(|v| v.as_slice()).collect();

    let batch_dual = DualBatch::from_f32(&refs, dim);
    let batch_a = TransposedBatch::from_f32(&refs16a, dim);
    let batch_b = TransposedBatch::from_f32(&refs16b, dim);

    let base_data: Vec<f32> = random_normalized_f32(base_count, dim, 99)
        .into_iter().flatten().collect();

    let mut group = c.benchmark_group("dual_accumulator");
    group.sample_size(10);
    group.throughput(Throughput::Elements((base_count * 32) as u64));

    // Two sequential 16-wide calls
    group.bench_function("2x16_sequential", |bench| {
        let mut out_a = [0.0f32; 16];
        let mut out_b = [0.0f32; 16];
        bench.iter(|| {
            for i in 0..base_count {
                let bv = &base_data[i * dim..(i + 1) * dim];
                bfn(black_box(&batch_a), black_box(bv), &mut out_a);
                bfn(black_box(&batch_b), black_box(bv), &mut out_b);
            }
            black_box((&out_a, &out_b));
        });
    });

    // True dual-accumulator 32-wide
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        group.bench_function("32_dual_fma", |bench| {
            let mut out32 = [0.0f32; 32];
            bench.iter(|| {
                for i in 0..base_count {
                    let bv = &base_data[i * dim..(i + 1) * dim];
                    unsafe {
                        dual_neg_dot_f32_avx512(
                            black_box(&batch_dual), black_box(bv), black_box(&mut out32),
                        );
                    }
                }
                black_box(&out32);
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. COMBINED: Full inner loop with all optimizations
// ═══════════════════════════════════════════════════════════════════════════

fn bench_combined(c: &mut Criterion) {
    let dim = 512;
    let k = 100;
    let base_count = 100_000;
    let query_count = 78; // 128-thread share of 10K queries

    let queries = random_normalized_f32(query_count, dim, 42);
    let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();

    // Build sub-batches
    let mut sub_batches = Vec::new();
    let mut sub_offsets = Vec::new();
    let mut off = 0;
    while off < query_count {
        let end = std::cmp::min(off + 16, query_count);
        sub_batches.push(TransposedBatch::from_f32(&query_refs[off..end], dim));
        sub_offsets.push(off);
        off = end;
    }

    let base_data: Vec<f32> = random_normalized_f32(base_count, dim, 99)
        .into_iter().flatten().collect();
    let bfn = select_batched_fn_f32(Metric::DotProduct).unwrap();

    let mut group = c.benchmark_group("combined_inner_loop");
    group.sample_size(10);
    group.throughput(Throughput::Elements((base_count * query_count) as u64));

    // Baseline: BinaryHeap, no prefetch
    group.bench_function("baseline_heap_noprefetch", |bench| {
        let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..query_count)
            .map(|_| BinaryHeap::with_capacity(k + 1)).collect();
        let mut thresholds = vec![f32::INFINITY; query_count];
        let mut dist_buf = [0.0f32; 16];

        bench.iter(|| {
            for h in heaps.iter_mut() { h.clear(); }
            thresholds.fill(f32::INFINITY);
            for i in 0..base_count {
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                for (si, sb) in sub_batches.iter().enumerate() {
                    bfn(black_box(sb), black_box(base_vec), &mut dist_buf);
                    let sub_off = sub_offsets[si];
                    for qi in 0..sb.count() {
                        let gqi = sub_off + qi;
                        let dist = dist_buf[qi];
                        if dist < thresholds[gqi] {
                            heaps[gqi].push(Neighbor { index: i as u32, distance: dist });
                            if heaps[gqi].len() > k { heaps[gqi].pop(); }
                            if heaps[gqi].len() == k {
                                thresholds[gqi] = heaps[gqi].peek().unwrap().distance;
                            }
                        }
                    }
                }
            }
            black_box(&heaps);
        });
    });

    // Opt A: Sorted array top-k
    group.bench_function("sorted_topk_noprefetch", |bench| {
        let mut topks: Vec<SortedTopK> = (0..query_count).map(|_| SortedTopK::new(k)).collect();
        let mut dist_buf = [0.0f32; 16];

        bench.iter(|| {
            topks.iter_mut().for_each(|t| *t = SortedTopK::new(k));
            for i in 0..base_count {
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                for (si, sb) in sub_batches.iter().enumerate() {
                    bfn(black_box(sb), black_box(base_vec), &mut dist_buf);
                    let sub_off = sub_offsets[si];
                    for qi in 0..sb.count() {
                        let gqi = sub_off + qi;
                        let dist = dist_buf[qi];
                        if dist < topks[gqi].threshold() {
                            topks[gqi].push(Neighbor { index: i as u32, distance: dist });
                        }
                    }
                }
            }
            black_box(&topks);
        });
    });

    // Opt B: BinaryHeap + prefetch
    group.bench_function("heap_prefetch", |bench| {
        let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..query_count)
            .map(|_| BinaryHeap::with_capacity(k + 1)).collect();
        let mut thresholds = vec![f32::INFINITY; query_count];
        let mut dist_buf = [0.0f32; 16];

        bench.iter(|| {
            for h in heaps.iter_mut() { h.clear(); }
            thresholds.fill(f32::INFINITY);
            for i in 0..base_count {
                if i + 2 < base_count {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            base_data[(i + 2) * dim..].as_ptr() as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                for (si, sb) in sub_batches.iter().enumerate() {
                    bfn(black_box(sb), black_box(base_vec), &mut dist_buf);
                    let sub_off = sub_offsets[si];
                    for qi in 0..sb.count() {
                        let gqi = sub_off + qi;
                        let dist = dist_buf[qi];
                        if dist < thresholds[gqi] {
                            heaps[gqi].push(Neighbor { index: i as u32, distance: dist });
                            if heaps[gqi].len() > k { heaps[gqi].pop(); }
                            if heaps[gqi].len() == k {
                                thresholds[gqi] = heaps[gqi].peek().unwrap().distance;
                            }
                        }
                    }
                }
            }
            black_box(&heaps);
        });
    });

    // Opt A+B: Sorted array + prefetch
    group.bench_function("sorted_topk_prefetch", |bench| {
        let mut topks: Vec<SortedTopK> = (0..query_count).map(|_| SortedTopK::new(k)).collect();
        let mut dist_buf = [0.0f32; 16];

        bench.iter(|| {
            topks.iter_mut().for_each(|t| *t = SortedTopK::new(k));
            for i in 0..base_count {
                if i + 2 < base_count {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            base_data[(i + 2) * dim..].as_ptr() as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                let base_vec = &base_data[i * dim..(i + 1) * dim];
                for (si, sb) in sub_batches.iter().enumerate() {
                    bfn(black_box(sb), black_box(base_vec), &mut dist_buf);
                    let sub_off = sub_offsets[si];
                    for qi in 0..sb.count() {
                        let gqi = sub_off + qi;
                        let dist = dist_buf[qi];
                        if dist < topks[gqi].threshold() {
                            topks[gqi].push(Neighbor { index: i as u32, distance: dist });
                        }
                    }
                }
            }
            black_box(&topks);
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_pairwise,
    bench_batched,
    bench_convert_f16_to_f32,
    bench_topk_structures,
    bench_prefetch,
    bench_double_wide,
    bench_dual_accumulator,
    bench_combined,
);
criterion_main!(benches);
