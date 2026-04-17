// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute exact KNN using pure `std::arch` SIMD.
//!
//! Zero external SIMD dependencies — uses only Rust's `std::arch` intrinsics
//! (AVX-512, AVX2, scalar fallback). Same threading and batching strategy as
//! knn-metal, but with no SimSIMD or FAISS dependency.
//!
//! This is a reference implementation demonstrating that the full KNN pipeline
//! can be built on `std::arch` alone. Runtime feature detection selects the
//! best available ISA.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use super::compute_knn::Neighbor;
use super::source_window::resolve_source;

pub struct ComputeKnnStdarchOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeKnnStdarchOp)
}

// ═══════════════════════════════════════════════════════════════════════
// Pure std::arch distance functions with runtime ISA detection
// ═══════════════════════════════════════════════════════════════════════

/// Distance function type.
type DistFn = fn(&[f32], &[f32]) -> f32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Metric { L2, DotProduct, Cosine }

impl Metric {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "L2" => Some(Metric::L2),
            "DOT_PRODUCT" | "IP" => Some(Metric::DotProduct),
            "COSINE" => Some(Metric::Cosine),
            _ => None,
        }
    }
}

fn select_dist_fn(metric: Metric) -> (DistFn, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            let f = match metric {
                Metric::L2 => l2sq_avx512 as DistFn,
                Metric::DotProduct => neg_dot_avx512 as DistFn,
                Metric::Cosine => cosine_avx512 as DistFn,
            };
            return (f, "AVX-512");
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let f = match metric {
                Metric::L2 => l2sq_avx2 as DistFn,
                Metric::DotProduct => neg_dot_avx2 as DistFn,
                Metric::Cosine => cosine_avx2 as DistFn,
            };
            return (f, "AVX2+FMA");
        }
    }
    let f = match metric {
        Metric::L2 => l2sq_scalar as DistFn,
        Metric::DotProduct => neg_dot_scalar as DistFn,
        Metric::Cosine => cosine_scalar as DistFn,
    };
    (f, "scalar")
}

// -- Scalar fallback ----------------------------------------------------------

fn l2sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum()
}

fn neg_dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    -(a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>())
}

fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y; na += x * x; nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

// -- AVX-512 ------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn l2sq_avx512(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l2sq_avx512_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2sq_avx512_inner(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm512_setzero_ps();
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let chunks = n / 16;
        for i in 0..chunks {
            let va = _mm512_loadu_ps(ap.add(i * 16));
            let vb = _mm512_loadu_ps(bp.add(i * 16));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..n {
            let d = a[i] - b[i]; sum += d * d;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
fn neg_dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    unsafe { neg_dot_avx512_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn neg_dot_avx512_inner(a: &[f32], b: &[f32]) -> f32 {    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm512_setzero_ps();
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let chunks = n / 16;
        for i in 0..chunks {
            let va = _mm512_loadu_ps(ap.add(i * 16));
            let vb = _mm512_loadu_ps(bp.add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..n {
            sum += a[i] * b[i];
        }
        -sum

    }
}

#[cfg(target_arch = "x86_64")]
fn cosine_avx512(a: &[f32], b: &[f32]) -> f32 {
    unsafe { cosine_avx512_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cosine_avx512_inner(a: &[f32], b: &[f32]) -> f32 {    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let (mut dot_acc, mut na_acc, mut nb_acc) = (
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let chunks = n / 16;
        for i in 0..chunks {
            let va = _mm512_loadu_ps(ap.add(i * 16));
            let vb = _mm512_loadu_ps(bp.add(i * 16));
            dot_acc = _mm512_fmadd_ps(va, vb, dot_acc);
            na_acc = _mm512_fmadd_ps(va, va, na_acc);
            nb_acc = _mm512_fmadd_ps(vb, vb, nb_acc);
        }
        let (mut dot, mut na, mut nb) = (
            _mm512_reduce_add_ps(dot_acc),
            _mm512_reduce_add_ps(na_acc),
            _mm512_reduce_add_ps(nb_acc));
        for i in (chunks * 16)..n {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
        }
        let denom = (na * nb).sqrt();
        if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }

    }
}

// -- AVX2+FMA -----------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn l2sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l2sq_avx2_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2sq_avx2_inner(a: &[f32], b: &[f32]) -> f32 {    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm256_setzero_ps();
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let chunks = n / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        let mut sum = hsum256(acc);
        for i in (chunks * 8)..n {
            let d = a[i] - b[i]; sum += d * d;
        }
        sum

    }
}

#[cfg(target_arch = "x86_64")]
fn neg_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe { neg_dot_avx2_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn neg_dot_avx2_inner(a: &[f32], b: &[f32]) -> f32 {    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm256_setzero_ps();
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let chunks = n / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        let mut sum = hsum256(acc);
        for i in (chunks * 8)..n {
            sum += a[i] * b[i];
        }
        -sum

    }
}

#[cfg(target_arch = "x86_64")]
fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe { cosine_avx2_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cosine_avx2_inner(a: &[f32], b: &[f32]) -> f32 {    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let (mut dot_acc, mut na_acc, mut nb_acc) = (
            _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let chunks = n / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
            na_acc = _mm256_fmadd_ps(va, va, na_acc);
            nb_acc = _mm256_fmadd_ps(vb, vb, nb_acc);
        }
        let (mut dot, mut na, mut nb) = (hsum256(dot_acc), hsum256(na_acc), hsum256(nb_acc));
        for i in (chunks * 8)..n {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
        }
        let denom = (na * nb).sqrt();
        if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }

    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

// ═══════════════════════════════════════════════════════════════════════
// KNN computation — same strategy as knn-metal
// ═══════════════════════════════════════════════════════════════════════

const BATCH_WIDTH: usize = 16;

/// Transpose queries into dimension-major layout for SIMD batch processing.
/// Layout: `data[d * BATCH_WIDTH + qi]` — one contiguous f32x16 per dimension.
struct TransposedQueries {
    data: Vec<f32>,
    dim: usize,
    count: usize,
}

impl TransposedQueries {
    fn new(queries: &[&[f32]], dim: usize) -> Self {
        let count = queries.len();
        assert!(count <= BATCH_WIDTH);
        let mut data = vec![0.0f32; dim * BATCH_WIDTH];
        for (qi, q) in queries.iter().enumerate() {
            for d in 0..dim {
                data[d * BATCH_WIDTH + qi] = q[d];
            }
        }
        Self { data, dim, count }
    }
}

/// Compute L2sq distances from one base vector to 16 transposed queries.
/// Pure std::arch AVX-512: broadcast each base dim, subtract, FMA accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2sq_batch16_avx512(
    base_vec: &[f32],
    transposed: &TransposedQueries,
    out: &mut [f32; BATCH_WIDTH],
) {
    use std::arch::x86_64::*;
    unsafe {
        let dim = transposed.dim;
        let tp = transposed.data.as_ptr();
        let bp = base_vec.as_ptr();
        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let bval = _mm512_set1_ps(*bp.add(d));
            let qvals = _mm512_loadu_ps(tp.add(d * BATCH_WIDTH));
            let diff = _mm512_sub_ps(bval, qvals);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        _mm512_storeu_ps(out.as_mut_ptr(), acc);
    }
}

/// Compute negative dot product from one base vector to 16 transposed queries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn neg_dot_batch16_avx512(
    base_vec: &[f32],
    transposed: &TransposedQueries,
    out: &mut [f32; BATCH_WIDTH],
) {
    use std::arch::x86_64::*;
    unsafe {
        let dim = transposed.dim;
        let tp = transposed.data.as_ptr();
        let bp = base_vec.as_ptr();
        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let bval = _mm512_set1_ps(*bp.add(d));
            let qvals = _mm512_loadu_ps(tp.add(d * BATCH_WIDTH));
            acc = _mm512_fmadd_ps(bval, qvals, acc);
        }

        // Negate: lower dot product = higher distance
        let neg = _mm512_sub_ps(_mm512_setzero_ps(), acc);
        _mm512_storeu_ps(out.as_mut_ptr(), neg);
    }
}

/// Batch distance function type for transposed queries.
type BatchDistFn = unsafe fn(&[f32], &TransposedQueries, &mut [f32; BATCH_WIDTH]);

fn select_batch_dist_fn(metric: Metric) -> Option<BatchDistFn> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return Some(match metric {
                Metric::L2 => l2sq_batch16_avx512,
                Metric::DotProduct => neg_dot_batch16_avx512,
                Metric::Cosine => return None, // cosine needs norms, use pairwise
            });
        }
    }
    None
}

/// Scan one base-vector segment for a set of queries. Returns per-query
/// top-k candidates (not globally merged yet). Accepts initial thresholds
/// from prior segments for early pruning.
fn find_top_k_segment(
    queries: &[&[f32]],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: DistFn,
    metric: Metric,
    dim: usize,
    initial_thresholds: &[f32],
) -> Vec<Vec<Neighbor>> {
    let n_queries = queries.len();
    let batch_fn = select_batch_dist_fn(metric);

    // Build transposed sub-batches
    let mut sub_batches: Vec<TransposedQueries> = Vec::new();
    let mut sub_offsets: Vec<usize> = Vec::new();
    if batch_fn.is_some() {
        let mut offset = 0;
        while offset < n_queries {
            let sub_end = (offset + BATCH_WIDTH).min(n_queries);
            sub_batches.push(TransposedQueries::new(&queries[offset..sub_end], dim));
            sub_offsets.push(offset);
            offset = sub_end;
        }
    }

    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..n_queries)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds: Vec<f32> = initial_thresholds.to_vec();
    let mut dist_buf = [0.0f32; BATCH_WIDTH];

    for j in start..end {
        let base_vec = base_reader.get_slice(j);
        let idx = j as u32;

        if let Some(bfn) = batch_fn {
            for (si, batch) in sub_batches.iter().enumerate() {
                unsafe { bfn(base_vec, batch, &mut dist_buf) };
                let sub_offset = sub_offsets[si];
                let count = batch.count;
                for qi in 0..count {
                    let gqi = sub_offset + qi;
                    let dist = dist_buf[qi];
                    if dist < thresholds[gqi] {
                        heaps[gqi].push(Neighbor { index: idx, distance: dist });
                        if heaps[gqi].len() > k { heaps[gqi].pop(); }
                        if heaps[gqi].len() == k {
                            thresholds[gqi] = heaps[gqi].peek().unwrap().distance;
                        }
                    }
                }
            }
        } else {
            for qi in 0..n_queries {
                let dist = dist_fn(queries[qi], base_vec);
                if dist < thresholds[qi] {
                    heaps[qi].push(Neighbor { index: idx, distance: dist });
                    if heaps[qi].len() > k { heaps[qi].pop(); }
                    if heaps[qi].len() == k {
                        thresholds[qi] = heaps[qi].peek().unwrap().distance;
                    }
                }
            }
        }
    }

    heaps.into_iter().map(|h| h.into_vec()).collect()
}

/// Per-thread KNN scan with transposed 16-wide SIMD batching.
///
/// Groups queries into sub-batches of 16, transposes each into
/// dimension-major layout, then for each base vector computes 16
/// distances with one FMA per dimension. Falls back to pairwise
/// for the tail sub-batch and non-AVX-512 platforms.
fn find_top_k(
    queries: &[&[f32]],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: DistFn,
    metric: Metric,
    dim: usize,
    results: &mut [Vec<Neighbor>],
    progress: &AtomicU64,
) {
    let n_queries = queries.len();
    let batch_fn = select_batch_dist_fn(metric);

    // Build transposed sub-batches of 16 queries each
    let mut sub_batches: Vec<TransposedQueries> = Vec::new();
    let mut sub_offsets: Vec<usize> = Vec::new();
    if batch_fn.is_some() {
        let mut offset = 0;
        while offset < n_queries {
            let sub_end = (offset + BATCH_WIDTH).min(n_queries);
            sub_batches.push(TransposedQueries::new(&queries[offset..sub_end], dim));
            sub_offsets.push(offset);
            offset = sub_end;
        }
    }

    let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..n_queries)
        .map(|_| BinaryHeap::with_capacity(k + 1))
        .collect();
    let mut thresholds = vec![f32::INFINITY; n_queries];
    let mut dist_buf = [0.0f32; BATCH_WIDTH];

    let stride = ((end - start) / 10).max(1);
    let mut i = start;
    while i < end {
        let stride_end = (i + stride).min(end);
        for j in i..stride_end {
            let base_vec = base_reader.get_slice(j);
            let idx = j as u32;

            if let Some(bfn) = batch_fn {
                // Transposed batch path: 16 distances per SIMD pass
                for (si, batch) in sub_batches.iter().enumerate() {
                    unsafe { bfn(base_vec, batch, &mut dist_buf) };
                    let sub_offset = sub_offsets[si];
                    let count = batch.count;
                    for qi in 0..count {
                        let gqi = sub_offset + qi;
                        let dist = dist_buf[qi];
                        if dist < thresholds[gqi] {
                            heaps[gqi].push(Neighbor { index: idx, distance: dist });
                            if heaps[gqi].len() > k { heaps[gqi].pop(); }
                            if heaps[gqi].len() == k {
                                thresholds[gqi] = heaps[gqi].peek().unwrap().distance;
                            }
                        }
                    }
                }
            } else {
                // Pairwise fallback
                for qi in 0..n_queries {
                    let dist = dist_fn(queries[qi], base_vec);
                    if dist < thresholds[qi] {
                        heaps[qi].push(Neighbor { index: idx, distance: dist });
                        if heaps[qi].len() > k { heaps[qi].pop(); }
                        if heaps[qi].len() == k {
                            thresholds[qi] = heaps[qi].peek().unwrap().distance;
                        }
                    }
                }
            }
        }
        progress.fetch_add((stride_end - i) as u64, AtomicOrdering::Relaxed);
        i = stride_end;
    }

    for (qi, heap) in heaps.into_iter().enumerate() {
        let mut v: Vec<Neighbor> = heap.into_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
            .then(a.index.cmp(&b.index)));
        results[qi] = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CommandOp implementation
// ═══════════════════════════════════════════════════════════════════════

fn resolve_path(s: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::Path::new(s);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult { status: Status::Error, message: msg.into(), produced: vec![], elapsed: start.elapsed() }
}

impl CommandOp for ComputeKnnStdarchOp {
    fn command_path(&self) -> &str {
        "compute knn-stdarch"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Brute-force exact KNN using pure std::arch SIMD (no external deps)".into(),
            body: format!(r#"# compute knn-stdarch

Brute-force exact KNN ground truth using only Rust `std::arch` intrinsics.

## Description

Zero external SIMD dependencies. Runtime feature detection selects the best
available ISA: AVX-512 → AVX2+FMA → scalar fallback. Uses the same
multi-threaded query-batching strategy as knn-metal.

This is a reference implementation proving the full KNN pipeline can run
on `std::arch` alone, without SimSIMD, FAISS, or BLAS.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Base vectors (mmap)".into(), adjustable: false },
            ResourceDesc { name: "threads".into(), description: "Parallel KNN threads".into(), adjustable: true },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let query_str = match options.require("query") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let indices_str = match options.require("indices") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let k: usize = match options.require("neighbors") {
            Ok(s) => match s.parse() { Ok(n) if n > 0 => n, _ => return error_result(format!("invalid neighbors: '{}'", s), start) },
            Err(e) => return error_result(e, start),
        };
        let metric_str = options.get("metric").unwrap_or("L2");
        let normalized = options.get("normalized").map(|s| s == "true").unwrap_or(false);
        let metric = match Metric::from_str(metric_str) {
            Some(m) => m,
            None => return error_result(format!("unknown metric: '{}'", metric_str), start),
        };
        let kernel_metric = if normalized && (metric == Metric::Cosine || metric == Metric::DotProduct) {
            Metric::DotProduct
        } else { metric };

        let logical_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads: usize = options.get("threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(logical_cpus);

        // Resolve paths
        let base_source = match resolve_source(base_str, &ctx.workspace) {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
        let distances_path = options.get("distances").map(|s| resolve_path(s, &ctx.workspace));

        for path in std::iter::once(&indices_path).chain(distances_path.iter()) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        // Open readers
        let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => Arc::new(r), Err(e) => return error_result(format!("open base: {}", e), start),
        };
        let query_reader = match MmapVectorReader::<f32>::open_fvec(&query_path) {
            Ok(r) => Arc::new(r), Err(e) => return error_result(format!("open query: {}", e), start),
        };

        let base_count = VectorReader::<f32>::count(base_reader.as_ref());
        let query_count = VectorReader::<f32>::count(query_reader.as_ref());
        let dim = VectorReader::<f32>::dim(base_reader.as_ref());

        let (base_offset, base_n) = match base_source.window {
            Some((ws, we)) => (ws.min(base_count), we.min(base_count).saturating_sub(ws.min(base_count))),
            None => (0, base_count),
        };

        let (dist_fn, isa_label) = select_dist_fn(kernel_metric);

        let batch_label = if select_batch_dist_fn(kernel_metric).is_some() {
            "transposed 16-wide"
        } else {
            "pairwise"
        };
        ctx.ui.log(&format!(
            "KNN-stdarch: {} queries × {} base, dim={}, k={}, metric={}, ISA={} ({}), threads={}",
            query_count, base_n, dim, k, metric_str, isa_label, batch_label, threads));

        // ── Partitioned base-vector scan ────────────────────────────
        // Split base vectors into cache-friendly segments. All threads
        // work on the same segment together (queries split across threads),
        // keeping base data hot in L3. Then move to the next segment.
        // Per-query heaps persist across segments and accumulate the
        // global top-k.
        let segment_size = {
            // Target: ~50% of system RAM for the segment's page cache footprint.
            // Each base vector is dim*4 bytes. With all threads reading the
            // same segment, only one copy lives in page cache.
            let total_ram: u64 = std::fs::read_to_string("/proc/meminfo").ok()
                .and_then(|s| s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|n| n.parse::<u64>().ok())
                    .map(|kb| kb * 1024))
                .unwrap_or(8 * 1024 * 1024 * 1024);
            let budget = (total_ram / 2) as usize;
            let entry_bytes = dim * 4;
            let auto = (budget / entry_bytes.max(1)).max(10_000);
            auto.min(base_n) // don't exceed actual base count
        };

        let n_segments = (base_n + segment_size - 1) / segment_size;
        ctx.ui.log(&format!("  {} base-vector segments ({} vectors each)",
            n_segments, segment_size));

        let chunk_size = (query_count + threads - 1) / threads;
        let pb = ctx.ui.bar(base_n as u64, "KNN-stdarch");

        // Per-query heaps persist across segments
        let mut all_heaps: Vec<BinaryHeap<Neighbor>> = (0..query_count)
            .map(|_| BinaryHeap::with_capacity(k + 1))
            .collect();
        let mut all_thresholds = vec![f32::INFINITY; query_count];

        for seg_idx in 0..n_segments {
            let seg_start = base_offset + seg_idx * segment_size;
            let seg_end = (seg_start + segment_size).min(base_offset + base_n);

            // Each thread scans this segment for its subset of queries,
            // returning partial heaps that we merge into the global heaps.
            let segment_results: Vec<Vec<(usize, Vec<Neighbor>)>> =
                std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..threads).filter_map(|ti| {
                        let q_start = ti * chunk_size;
                        if q_start >= query_count { return None; }
                        let q_end = (q_start + chunk_size).min(query_count);
                        let q_len = q_end - q_start;

                        let base_ref = Arc::clone(&base_reader);
                        let query_ref = Arc::clone(&query_reader);
                        // Pass current thresholds to the thread for early pruning
                        let thresh: Vec<f32> = all_thresholds[q_start..q_end].to_vec();

                        Some(scope.spawn(move || {
                            let queries: Vec<&[f32]> = (0..q_len)
                                .map(|i| query_ref.get_slice(q_start + i))
                                .collect();

                            find_top_k_segment(
                                &queries, &base_ref,
                                seg_start, seg_end,
                                k, dist_fn, kernel_metric, dim,
                                &thresh,
                            ).into_iter()
                                .enumerate()
                                .map(|(i, heap)| (q_start + i, heap))
                                .collect::<Vec<_>>()
                        }))
                    }).collect();

                    handles.into_iter()
                        .map(|h| h.join().unwrap())
                        .collect()
                });

            // Merge segment results into global heaps
            for thread_results in segment_results {
                for (qi, neighbors) in thread_results {
                    for n in neighbors {
                        if n.distance < all_thresholds[qi] {
                            all_heaps[qi].push(n);
                            if all_heaps[qi].len() > k {
                                all_heaps[qi].pop();
                                all_thresholds[qi] = all_heaps[qi].peek().unwrap().distance;
                            }
                        }
                    }
                }
            }

            pb.set_position((seg_end - base_offset) as u64);
        }
        pb.finish();

        let compute_secs = start.elapsed().as_secs_f64();
        let evals = query_count as f64 * base_n as f64;
        ctx.ui.log(&format!("  compute: {:.2}s ({:.1}B dist/s)",
            compute_secs, evals / compute_secs / 1e9));

        // Convert heaps to sorted results
        let all_results: Vec<Vec<Neighbor>> = all_heaps.into_iter().map(|heap| {
            let mut v: Vec<Neighbor> = heap.into_vec();
            v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
                .then(a.index.cmp(&b.index)));
            v
        }).collect();

        // Write output
        let mut idx_file = match std::fs::File::create(&indices_path) {
            Ok(f) => std::io::BufWriter::new(f),
            Err(e) => return error_result(format!("create {}: {}", indices_path.display(), e), start),
        };
        let mut dist_file = distances_path.as_ref().map(|dp| {
            std::fs::File::create(dp)
                .map(std::io::BufWriter::new)
                .unwrap_or_else(|_| panic!("create {}", dp.display()))
        });

        let dim_bytes = (k as i32).to_le_bytes();
        for qi in 0..query_count {
            idx_file.write_all(&dim_bytes).unwrap();
            if let Some(ref mut dw) = dist_file {
                dw.write_all(&dim_bytes).unwrap();
            }
            for j in 0..k {
                if j < all_results[qi].len() {
                    let n = &all_results[qi][j];
                    idx_file.write_all(&(n.index as i32).to_le_bytes()).unwrap();
                    if let Some(ref mut dw) = dist_file {
                        dw.write_all(&n.distance.to_le_bytes()).unwrap();
                    }
                } else {
                    idx_file.write_all(&(-1i32).to_le_bytes()).unwrap();
                    if let Some(ref mut dw) = dist_file {
                        dw.write_all(&f32::INFINITY.to_le_bytes()).unwrap();
                    }
                }
            }
        }
        idx_file.flush().unwrap();
        if let Some(ref mut dw) = dist_file { dw.flush().unwrap(); }

        let mut produced = vec![indices_path.clone()];
        if let Some(ref dp) = distances_path { produced.push(dp.clone()); }

        let elapsed = start.elapsed();
        ctx.ui.log(&format!("  total: {:.2}s", elapsed.as_secs_f64()));

        CommandResult {
            status: Status::Ok,
            message: format!("{} queries, k={}, {} base, metric={}, ISA={}, engine=stdarch",
                query_count, k, base_n, metric_str, isa_label),
            elapsed,
            produced,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "base".into(), type_name: "Path".into(), required: true, default: None, description: "Base vectors (fvec)".into(), role: OptionRole::Input },
            OptionDesc { name: "query".into(), type_name: "Path".into(), required: true, default: None, description: "Query vectors (fvec)".into(), role: OptionRole::Input },
            OptionDesc { name: "indices".into(), type_name: "Path".into(), required: true, default: None, description: "Output neighbor indices (ivec)".into(), role: OptionRole::Output },
            OptionDesc { name: "distances".into(), type_name: "Path".into(), required: false, default: None, description: "Output neighbor distances (fvec)".into(), role: OptionRole::Output },
            OptionDesc { name: "neighbors".into(), type_name: "int".into(), required: true, default: None, description: "k (number of neighbors)".into(), role: OptionRole::Config },
            OptionDesc { name: "metric".into(), type_name: "enum".into(), required: false, default: Some("L2".into()), description: "L2, DOT_PRODUCT, COSINE, IP".into(), role: OptionRole::Config },
            OptionDesc { name: "normalized".into(), type_name: "bool".into(), required: false, default: Some("false".into()), description: "Vectors are L2-normalized".into(), role: OptionRole::Config },
            OptionDesc { name: "threads".into(), type_name: "int".into(), required: false, default: Some("0".into()), description: "Thread count (0 = auto)".into(), role: OptionRole::Config },
            // Accept but ignore knn-metal options for drop-in compatibility
            OptionDesc { name: "partition_size".into(), type_name: "int".into(), required: false, default: None, description: "Ignored (no partitioning)".into(), role: OptionRole::Config },
            OptionDesc { name: "compress-cache".into(), type_name: "bool".into(), required: false, default: Some("false".into()), description: "Ignored".into(), role: OptionRole::Config },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::{Options, Status};

    fn make_ctx(workspace: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
            defaults: indexmap::IndexMap::new(),
            dry_run: false,
            progress: crate::pipeline::progress::ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
        }
    }

    fn write_fvec(path: &std::path::Path, vectors: &[Vec<f32>]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    fn read_ivec_rows(path: &std::path::Path) -> Vec<Vec<i32>> {
        let reader = MmapVectorReader::<i32>::open_ivec(path).unwrap();
        let count = VectorReader::<i32>::count(&reader);
        (0..count).map(|i| reader.get_slice(i).to_vec()).collect()
    }

    #[test]
    fn test_stdarch_l2_known() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let query = vec![vec![0.1, 0.1]];
        write_fvec(&tmp.path().join("b.fvec"), &base);
        write_fvec(&tmp.path().join("q.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", "b.fvec"); opts.set("query", "q.fvec");
        opts.set("indices", "out.ivec"); opts.set("neighbors", "2"); opts.set("metric", "L2");

        let mut op = ComputeKnnStdarchOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        let rows = read_ivec_rows(&tmp.path().join("out.ivec"));
        assert_eq!(rows[0][0], 0, "nearest to [0.1,0.1] should be [0,0]");
    }

    #[test]
    fn test_stdarch_matches_metal() {
        use std::collections::HashSet;

        let tmp = tempfile::tempdir().unwrap();

        let mut rng = 42u64;
        let mut next_f32 = || -> f32 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        let base: Vec<Vec<f32>> = (0..100).map(|_| (0..8).map(|_| next_f32()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..10).map(|_| (0..8).map(|_| next_f32()).collect()).collect();
        write_fvec(&tmp.path().join("b.fvec"), &base);
        write_fvec(&tmp.path().join("q.fvec"), &query);

        // stdarch
        {
            let mut ctx = make_ctx(tmp.path());
            let mut opts = Options::new();
            opts.set("base", "b.fvec"); opts.set("query", "q.fvec");
            opts.set("indices", "stdarch.ivec"); opts.set("neighbors", "5"); opts.set("metric", "L2");
            let mut op = ComputeKnnStdarchOp;
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(r.status, Status::Ok);
        }

        // metal (SimSIMD)
        {
            let mut ctx = make_ctx(tmp.path());
            let mut opts = Options::new();
            opts.set("base", "b.fvec"); opts.set("query", "q.fvec");
            opts.set("indices", "metal.ivec"); opts.set("neighbors", "5"); opts.set("metric", "L2");
            let mut op = crate::pipeline::commands::compute_knn::factory();
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(r.status, Status::Ok);
        }

        let stdarch_rows = read_ivec_rows(&tmp.path().join("stdarch.ivec"));
        let metal_rows = read_ivec_rows(&tmp.path().join("metal.ivec"));

        for q in 0..stdarch_rows.len() {
            let sa: HashSet<i32> = stdarch_rows[q].iter().copied().collect();
            let mt: HashSet<i32> = metal_rows[q].iter().copied().collect();
            let diff = sa.symmetric_difference(&mt).count();
            assert!(diff <= 2,
                "query {}: stdarch {:?} vs metal {:?} — {} differ", q, &stdarch_rows[q], &metal_rows[q], diff);
        }
    }

    #[test]
    fn test_distance_kernels() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];

        let l2 = l2sq_scalar(&a, &b);
        assert!((l2 - 64.0).abs() < 1e-6, "L2sq should be 64, got {}", l2);

        let dot = neg_dot_scalar(&a, &b);
        assert!((dot - (-70.0)).abs() < 1e-6, "neg_dot should be -70, got {}", dot);

        let cos = cosine_scalar(&a, &b);
        assert!(cos >= 0.0 && cos < 0.05, "cosine dist should be near 0, got {}", cos);
    }
}
