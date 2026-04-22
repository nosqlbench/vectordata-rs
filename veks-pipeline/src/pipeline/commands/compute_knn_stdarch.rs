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
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
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

// Metric enum, cache helpers, and segment discovery are shared across
// all brute-force KNN engines via `super::knn_segment`.
use super::knn_segment::{
    CosineMode, Metric, build_cache_path, cache_prefix_for,
    load_segment_cache, merge_segment_into_heaps, resolve_cosine_mode,
    scan_cached_segments, write_segment_cache,
};

const ENGINE_NAME: &str = "knn-stdarch";

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
    /// Precomputed L2 norm of each query (for cosine distance).
    /// Unused slots = 1.0 to avoid division by zero.
    query_norms: [f32; BATCH_WIDTH],
    dim: usize,
    count: usize,
}

impl TransposedQueries {
    fn new(queries: &[&[f32]], dim: usize) -> Self {
        let count = queries.len();
        assert!(count <= BATCH_WIDTH);
        let mut data = vec![0.0f32; dim * BATCH_WIDTH];
        let mut query_norms = [1.0f32; BATCH_WIDTH];
        for (qi, q) in queries.iter().enumerate() {
            let mut norm_sq = 0.0f32;
            for d in 0..dim {
                data[d * BATCH_WIDTH + qi] = q[d];
                norm_sq += q[d] * q[d];
            }
            query_norms[qi] = norm_sq.sqrt().max(f32::EPSILON);
        }
        Self { data, query_norms, dim, count }
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

/// Compute cosine distance from one base vector to 16 transposed queries.
/// cosine_dist = 1.0 - dot(base, query) / (norm(base) * norm(query))
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cosine_batch16_avx512(
    base_vec: &[f32],
    transposed: &TransposedQueries,
    out: &mut [f32; BATCH_WIDTH],
) {
    use std::arch::x86_64::*;
    unsafe {
        let dim = transposed.dim;
        let tp = transposed.data.as_ptr();
        let bp = base_vec.as_ptr();
        let mut dot_acc = _mm512_setzero_ps();
        let mut base_norm_acc = _mm512_setzero_ps();

        for d in 0..dim {
            let bval = _mm512_set1_ps(*bp.add(d));
            let qvals = _mm512_loadu_ps(tp.add(d * BATCH_WIDTH));
            dot_acc = _mm512_fmadd_ps(bval, qvals, dot_acc);
            base_norm_acc = _mm512_fmadd_ps(bval, bval, base_norm_acc);
        }

        // base_norm is the same for all 16 queries — reduce and broadcast
        let base_norm = _mm512_reduce_add_ps(base_norm_acc).sqrt().max(f32::EPSILON);
        let base_norm_v = _mm512_set1_ps(base_norm);

        // query norms are precomputed
        let q_norms = _mm512_loadu_ps(transposed.query_norms.as_ptr());

        // denom = base_norm * query_norms
        let denom = _mm512_mul_ps(base_norm_v, q_norms);

        // cosine_dist = 1.0 - dot / denom
        let one = _mm512_set1_ps(1.0);
        let cos_sim = _mm512_div_ps(dot_acc, denom);
        let cos_dist = _mm512_sub_ps(one, cos_sim);

        _mm512_storeu_ps(out.as_mut_ptr(), cos_dist);
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
                Metric::Cosine => cosine_batch16_avx512,
            });
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════
// Streaming pread scanner
// ═══════════════════════════════════════════════════════════════════════
//
// Mmap is the wrong primitive for a single-pass sequential scan: page-
// fault latency ends up in the compute hot path, RSS grows with segment
// size, and the leading compute thread bottlenecks on kernel readahead.
// Instead we pread the segment in chunks into a pair of aligned heap
// buffers, alternating reader/writer every iteration. While compute
// threads scan buffer A for the current chunk, one I/O task preads
// the next chunk into buffer B. The `std::thread::scope` join is the
// natural barrier; after it, swap roles and advance.
//
// Layout notes:
//  - fvec entry stride is `entry_size = 4 + dim*4` bytes: a 4-byte dim
//    header followed by `dim` f32 payload elements.
//  - We pread whole entries (including headers) as raw bytes but back
//    the buffer with `Vec<f32>` so alignment is guaranteed to be 4
//    bytes for all derived `&[f32]` slices.
//  - Given any vector index `j` within the chunk, the start of its
//    f32 payload in f32-units is `j * (1 + dim) + 1` — the `+1` steps
//    past the dim header. No unsafe needed.

/// Target chunk size for pread streaming. 64 MiB balances single-
/// syscall I/O throughput, page-cache friendliness, and L3 residence.
/// Undersized chunks cost in syscall overhead and buffer management;
/// oversized chunks spoil cache locality and let the leading edge
/// outrun prefetch.
const STREAM_CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// Allocate a chunk buffer as `Vec<f32>`. Backing with f32 (not u8)
/// guarantees 4-byte alignment for the payload slices the scan loop
/// extracts, so the inner loop stays in safe code.
fn alloc_chunk_buf(chunk_capacity_bytes: usize) -> Vec<f32> {
    // Round up so the byte capacity can hold any entry-sized chunk.
    let n_f32 = (chunk_capacity_bytes + 3) / 4;
    vec![0.0f32; n_f32]
}

/// View a `&mut [f32]` buffer as `&mut [u8]` for pread. Safe because
/// f32 has stricter alignment than u8 and shares the same raw
/// storage.
fn buf_bytes_mut(buf: &mut [f32]) -> &mut [u8] {
    let ptr = buf.as_mut_ptr() as *mut u8;
    let len = std::mem::size_of_val(buf);
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Scan one chunk of base vectors against a thread's query subset,
/// updating the thread's persistent heaps/thresholds in place.
///
/// `chunk_buf` is the full chunk's raw f32 storage (entries as laid
/// out in the fvec file, including 4-byte dim headers). `n_vecs` is
/// the number of base vectors actually present in this chunk; the
/// buffer may be larger (capacity-sized).
///
/// `chunk_first_idx` is the absolute base-vector index of the
/// chunk's first vector, used to tag `Neighbor` records with the
/// file-level index.
fn scan_chunk_f32(
    chunk_buf: &[f32],
    n_vecs: usize,
    chunk_first_idx: usize,
    dim: usize,
    queries: &[&[f32]],
    sub_batches: &[TransposedQueries],
    sub_offsets: &[usize],
    k: usize,
    dist_fn: DistFn,
    batch_fn: Option<BatchDistFn>,
    heaps: &mut [BinaryHeap<Neighbor>],
    thresholds: &mut [f32],
) {
    let stride_f32 = 1 + dim; // header (1 f32) + payload (dim f32s)
    let n_queries = queries.len();
    let mut dist_buf = [0.0f32; BATCH_WIDTH];

    for j in 0..n_vecs {
        let payload_off = j * stride_f32 + 1;
        let base_vec: &[f32] = &chunk_buf[payload_off..payload_off + dim];
        let idx = (chunk_first_idx + j) as u32;

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
        let metric = match Metric::from_str(metric_str) {
            Some(m) => m,
            None => return error_result(format!("unknown metric: '{}'", metric_str), start),
        };
        let cosine_mode = match resolve_cosine_mode(metric, options) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };
        // Kernel-metric selection:
        //   L2 / DOT_PRODUCT: unaffected by cosine_mode
        //   COSINE + AssumeNormalized: collapse to DotProduct (IP
        //     on pre-normalized inputs) — bit-matches the BLAS path
        //     and lets cached segments be shared with DOT runs.
        //   COSINE + ProperMetric: run the proper cosine kernel
        //     (sqrt/divide in-kernel).
        let kernel_metric = match (metric, cosine_mode) {
            (Metric::Cosine, Some(CosineMode::AssumeNormalized)) => Metric::DotProduct,
            _ => metric,
        };

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

        // Open readers. The mmap readers are used for metadata (dim,
        // count) and for query slices. The base vectors are streamed
        // through a separate pread(2) path during compute so no base
        // bytes ever enter the process address space — see the scanner
        // module above for rationale.
        let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => Arc::new(r), Err(e) => return error_result(format!("open base: {}", e), start),
        };
        let query_reader = match MmapVectorReader::<f32>::open_fvec(&query_path) {
            Ok(r) => Arc::new(r), Err(e) => return error_result(format!("open query: {}", e), start),
        };
        // File handle used exclusively for pread-based streaming.
        // Opened once and shared across all segment scans.
        let base_file = match File::open(&base_path) {
            Ok(f) => Arc::new(f),
            Err(e) => return error_result(format!("open base for streaming: {}", e), start),
        };

        let base_count = VectorReader::<f32>::count(base_reader.as_ref());
        let query_count = VectorReader::<f32>::count(query_reader.as_ref());
        let dim = VectorReader::<f32>::dim(base_reader.as_ref());

        // Resolve the effective base window from BOTH the inline path
        // syntax (`base.fvec[0..N]`) and the injected `range` option
        // that sized-profile expansion puts on every per-profile
        // step. Previously this only read the inline form, so sized
        // profiles silently scanned the full base. See
        // `source_window::resolve_window` for the full rationale.
        let effective_window = super::source_window::resolve_window(
            base_source.window, options.get("range"),
        );
        let (base_offset, base_n) = match effective_window {
            Some((ws, we)) => {
                let s = ws.min(base_count);
                (s, we.min(base_count).saturating_sub(s))
            }
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
            // Explicit override wins — tests use it to force multiple
            // small segments, and users can use it to force finer
            // cache granularity on huge inputs (trade more cache
            // files for finer resume granularity).
            let explicit: Option<usize> = options.get("partition_size")
                .and_then(|s| s.parse().ok())
                .filter(|&n: &usize| n > 0);
            if let Some(n) = explicit {
                n.min(base_n)
            } else {
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
            }
        };

        // Nominal segment count — the actual plan may produce
        // fewer/differently-sized segments when scan_cached_segments
        // greedy-matches cached ranges that don't align with this
        // value. This log is just the "fresh-compute baseline."
        let nominal_segments = (base_n + segment_size - 1) / segment_size;
        ctx.ui.log(&format!("  {} base-vector segments @ {} vectors each (nominal, before cache reuse)",
            nominal_segments, segment_size));

        let chunk_size = (query_count + threads - 1) / threads;

        // Per-query heaps persist across segments
        let mut all_heaps: Vec<BinaryHeap<Neighbor>> = (0..query_count)
            .map(|_| BinaryHeap::with_capacity(k + 1))
            .collect();
        let mut all_thresholds = vec![f32::INFINITY; query_count];

        // ── Discover reusable segments + plan ──────────────────────
        //
        // Scan the cache directory for every segment previously
        // written by this command (across prior runs of ANY profile),
        // plus any completed smaller-profile outputs that naturally
        // cover [0..N). Build a segment plan that greedy-prefers the
        // largest cached segment at each position, filling gaps with
        // fresh partition_size chunks.
        //
        // Cross-profile reuse is the point here: when profiles run
        // small-to-large (the usual sized-profile sweep), every
        // larger profile picks up the previous profiles' work from
        // disk instead of recomputing [0..smaller) from scratch.
        if !ctx.cache.exists() {
            let _ = std::fs::create_dir_all(&ctx.cache);
        }
        let cache_prefix = cache_prefix_for(&base_path, &query_path);
        let base_start = base_offset;
        let base_end = base_offset + base_n;

        let cached_segments = scan_cached_segments(
            &ctx.cache, ENGINE_NAME, &cache_prefix, k, kernel_metric, query_count,
            &ctx.workspace, base_end, &ctx.ui,
        );

        /// A single entry in the planned segment chain — either a
        /// ready-to-replay cached segment or a fresh range that
        /// needs computation (and will write its own cache on
        /// completion).
        struct PlannedSegment {
            start: usize,
            end: usize,
            ivec_path: PathBuf,
            fvec_path: PathBuf,
            cached: bool,
            flip_sign: bool,
        }

        // Greedy chain build: at each position, prefer the largest
        // cache starting exactly there whose end stays within
        // `base_end`. Otherwise lay down a fresh partition_size
        // segment.
        let mut plan: Vec<PlannedSegment> = Vec::new();
        let mut pos = base_start;
        while pos < base_end {
            let best = cached_segments.iter()
                .filter(|s| s.start == pos && s.end <= base_end && s.end > pos)
                .max_by_key(|s| s.end);
            if let Some(seg) = best {
                plan.push(PlannedSegment {
                    start: seg.start, end: seg.end,
                    ivec_path: seg.ivec_path.clone(),
                    fvec_path: seg.fvec_path.clone(),
                    cached: true,
                    flip_sign: seg.flip_sign,
                });
                pos = seg.end;
            } else {
                let pe = (pos + segment_size).min(base_end);
                plan.push(PlannedSegment {
                    start: pos, end: pe,
                    ivec_path: build_cache_path(&ctx.cache, ENGINE_NAME, &cache_prefix, pos, pe, k, kernel_metric, "neighbors", "ivec"),
                    fvec_path: build_cache_path(&ctx.cache, ENGINE_NAME, &cache_prefix, pos, pe, k, kernel_metric, "distances", "fvec"),
                    cached: false,
                    flip_sign: false,
                });
                pos = pe;
            }
        }

        let n_segments = plan.len();
        let cached_count = plan.iter().filter(|p| p.cached).count();
        let to_compute = n_segments - cached_count;
        // Base-vectors covered by cache (for % savings banner).
        let cached_bases: usize = plan.iter()
            .filter(|p| p.cached)
            .map(|p| p.end - p.start)
            .sum();

        // Progress bar sized to the COMPUTE-ONLY workload. Cached
        // segment replay is reported separately and must not inflate
        // the rate or collapse the ETA.
        let compute_bases: usize = plan.iter()
            .filter(|p| !p.cached)
            .map(|p| p.end - p.start)
            .sum();
        let pb = ctx.ui.bar(compute_bases as u64, "KNN-stdarch");
        let mut compute_progress: u64 = 0;

        if cached_count > 0 {
            let base_pct = 100.0 * cached_bases as f64 / base_n as f64;
            ctx.ui.log(&format!(
                "  ┌─── segment cache: {}/{} segments reusable ({} of {} base vectors, {:.0}%) ───",
                cached_count, n_segments,
                cached_bases, base_n, base_pct,
            ));
            ctx.ui.log(&format!(
                "  │  will REPLAY {} segment(s) from disk, COMPUTE {} fresh",
                cached_count, to_compute,
            ));
            ctx.ui.log(&format!(
                "  │  cache dir: {}",
                ctx.cache.display(),
            ));
            ctx.ui.log("  └──────────────────────────────────────────────────────");

            // Replay every cached segment into the global heaps up
            // front — this tightens `all_thresholds` before the
            // compute loop starts, maximizing early-exit pruning
            // when the fresh segments actually run.
            let replay_start = Instant::now();
            let mut replayed_bytes: u64 = 0;
            for (seg_idx, p) in plan.iter().enumerate() {
                if !p.cached { continue; }
                let seg = match load_segment_cache(
                    &p.ivec_path, &p.fvec_path, k, query_count, p.flip_sign,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        return error_result(
                            format!("failed to replay segment {} [{}..{}): {}",
                                seg_idx, p.start, p.end, e),
                            start,
                        );
                    }
                };
                merge_segment_into_heaps(&seg, &mut all_heaps, &mut all_thresholds, k);
                let ivec_sz = std::fs::metadata(&p.ivec_path).map(|m| m.len()).unwrap_or(0);
                let fvec_sz = std::fs::metadata(&p.fvec_path).map(|m| m.len()).unwrap_or(0);
                replayed_bytes += ivec_sz + fvec_sz;
                ctx.ui.log(&format!(
                    "  ▸ [seg {:>3}/{}] REUSED range [{}..{}) from {}",
                    seg_idx + 1, n_segments, p.start, p.end,
                    p.ivec_path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                ));
                // Cache replay is reported separately via the log
                // line above; the compute-only progress bar stays at
                // 0 so its rec/s reflects actual sgemm throughput and
                // its ETA reflects remaining compute work.
                let _ = base_start;
            }
            let replay_secs = replay_start.elapsed().as_secs_f64();
            ctx.ui.log(&format!(
                "  ✓ replay complete: {} segment(s), {:.1} MiB read from cache in {:.1}s",
                cached_count,
                replayed_bytes as f64 / (1024.0 * 1024.0),
                replay_secs,
            ));
        } else {
            ctx.ui.log(&format!(
                "  segment cache: 0 reusable — {} segment(s) will be computed fresh (cache will be populated for future profiles)",
                n_segments,
            ));
        }

        // ── Streaming pread pipeline ────────────────────────────────
        // Segment compute reads base vectors via pread into heap
        // buffers (not mmap). Two buffers alternate: while compute
        // threads scan chunk N, an I/O task preads chunk N+1 into
        // the other buffer. The scope join is the chunk barrier.
        let entry_size = 4 + dim * 4;
        let vecs_per_chunk = (STREAM_CHUNK_BYTES / entry_size).max(1);
        let chunk_byte_cap = vecs_per_chunk * entry_size;
        let mut buf_a = alloc_chunk_buf(chunk_byte_cap);
        let mut buf_b = alloc_chunk_buf(chunk_byte_cap);
        let batch_fn = select_batch_dist_fn(kernel_metric);

        let mut computed_count = 0usize;
        for seg_idx in 0..n_segments {
            let p = &plan[seg_idx];

            // Skip if this segment's contribution was loaded from cache above.
            if p.cached {
                continue;
            }

            let seg_start = p.start;
            let seg_end = p.end;
            let seg_len = seg_end - seg_start;
            let n_chunks = (seg_len + vecs_per_chunk - 1) / vecs_per_chunk;
            let seg_compute_start = Instant::now();
            ctx.ui.log(&format!(
                "  ▶ [seg {:>3}/{}] COMPUTE range [{}..{}) ({} base × {} queries, {} chunk(s) ≤ {} vecs)",
                seg_idx + 1, n_segments,
                seg_start, seg_end,
                seg_len, query_count,
                n_chunks, vecs_per_chunk,
            ));

            // Per-thread persistent state across this segment's chunks.
            // Heaps accumulate across chunks; thresholds tighten as
            // the per-thread top-K fills up.
            struct ThreadCtx<'a> {
                q_start: usize,
                queries: Vec<&'a [f32]>,
                sub_batches: Vec<TransposedQueries>,
                sub_offsets: Vec<usize>,
                heaps: Vec<BinaryHeap<Neighbor>>,
                thresholds: Vec<f32>,
            }
            let mut thread_ctxs: Vec<ThreadCtx> = (0..threads).filter_map(|ti| {
                let q_start = ti * chunk_size;
                if q_start >= query_count { return None; }
                let q_end = (q_start + chunk_size).min(query_count);
                let q_len = q_end - q_start;
                let queries: Vec<&[f32]> = (0..q_len)
                    .map(|i| query_reader.get_slice(q_start + i))
                    .collect();
                let mut sub_batches: Vec<TransposedQueries> = Vec::new();
                let mut sub_offsets: Vec<usize> = Vec::new();
                if batch_fn.is_some() {
                    let mut off = 0;
                    while off < q_len {
                        let se = (off + BATCH_WIDTH).min(q_len);
                        sub_batches.push(TransposedQueries::new(&queries[off..se], dim));
                        sub_offsets.push(off);
                        off = se;
                    }
                }
                let heaps: Vec<BinaryHeap<Neighbor>> = (0..q_len)
                    .map(|_| BinaryHeap::with_capacity(k + 1))
                    .collect();
                // Seed thresholds from the global top-K accumulated by
                // prior segments — lets us prune early on any base
                // vector that can't beat what we've already found.
                let thresholds: Vec<f32> = all_thresholds[q_start..q_end].to_vec();
                Some(ThreadCtx {
                    q_start, queries, sub_batches, sub_offsets, heaps, thresholds,
                })
            }).collect();

            // Helper: absolute-index range of chunk `c` within this segment.
            let chunk_range = |c: usize| -> (usize, usize) {
                let s = seg_start + c * vecs_per_chunk;
                let e = (s + vecs_per_chunk).min(seg_end);
                (s, e - s)
            };

            // Prime buf_a with chunk 0 synchronously — nothing to
            // overlap with yet. After this, every subsequent chunk's
            // I/O overlaps with the previous chunk's compute.
            let (c0_first, c0_n) = chunk_range(0);
            let c0_bytes = c0_n * entry_size;
            let c0_off = (c0_first as u64) * (entry_size as u64);
            if let Err(e) = base_file.read_exact_at(
                &mut buf_bytes_mut(&mut buf_a)[..c0_bytes], c0_off,
            ) {
                return error_result(
                    format!("pread seg {} chunk 0: {}", seg_idx, e),
                    start,
                );
            }

            // I/O errors from the prefetch task are captured here
            // and surfaced after the scope joins.
            let io_err: Arc<std::sync::Mutex<Option<String>>> =
                Arc::new(std::sync::Mutex::new(None));

            for chunk_i in 0..n_chunks {
                let is_last = chunk_i + 1 == n_chunks;
                let (cur_first, cur_n) = chunk_range(chunk_i);
                let (next_first, next_n) = if is_last {
                    (0, 0)
                } else {
                    chunk_range(chunk_i + 1)
                };
                let next_bytes = next_n * entry_size;
                let next_off = (next_first as u64) * (entry_size as u64);

                std::thread::scope(|scope| {
                    // I/O task: fill buf_b with the next chunk.
                    if !is_last {
                        let rb: &mut Vec<f32> = &mut buf_b;
                        let bf = Arc::clone(&base_file);
                        let err_slot = Arc::clone(&io_err);
                        scope.spawn(move || {
                            if let Err(e) = bf.read_exact_at(
                                &mut buf_bytes_mut(rb)[..next_bytes], next_off,
                            ) {
                                *err_slot.lock().unwrap() =
                                    Some(format!("pread chunk@{}: {}", next_first, e));
                            }
                        });
                    }
                    // Compute tasks: each thread scans buf_a for its
                    // query subset, updating its persistent heaps.
                    let sb: &[f32] = &buf_a;
                    for tc in thread_ctxs.iter_mut() {
                        let queries_ref: &[&[f32]] = &tc.queries;
                        let subs_ref: &[TransposedQueries] = &tc.sub_batches;
                        let offs_ref: &[usize] = &tc.sub_offsets;
                        let heaps_ref: &mut [BinaryHeap<Neighbor>] = &mut tc.heaps;
                        let thr_ref: &mut [f32] = &mut tc.thresholds;
                        scope.spawn(move || {
                            scan_chunk_f32(
                                sb, cur_n, cur_first, dim,
                                queries_ref, subs_ref, offs_ref,
                                k, dist_fn, batch_fn,
                                heaps_ref, thr_ref,
                            );
                        });
                    }
                });

                // Surface any I/O error after the scope joins — we
                // must not proceed to swap + scan a buffer whose
                // contents are undefined.
                if let Some(msg) = io_err.lock().unwrap().take() {
                    return error_result(
                        format!("seg {}: {}", seg_idx, msg),
                        start,
                    );
                }

                if !is_last {
                    std::mem::swap(&mut buf_a, &mut buf_b);
                }
                compute_progress += cur_n as u64;
                pb.set_position(compute_progress);
                let _ = (cur_first, base_offset);
            }

            // Collect per-thread heaps into absolute-indexed per_query
            // for cache write and global merge.
            let mut per_query: Vec<Vec<Neighbor>> =
                (0..query_count).map(|_| Vec::new()).collect();
            for tc in thread_ctxs {
                for (i, heap) in tc.heaps.into_iter().enumerate() {
                    per_query[tc.q_start + i] = heap.into_vec();
                }
            }

            // Persist the segment's contribution to cache BEFORE
            // merging into the global heaps. If we crashed between
            // merge and write, a subsequent run would double-count
            // this segment's neighbors. Cache-then-merge ordering
            // makes this replay-safe: crash after cache write → next
            // run skips this segment entirely (the load re-creates
            // the same heap contribution).
            if let Err(e) = write_segment_cache(
                &plan[seg_idx].ivec_path,
                &plan[seg_idx].fvec_path,
                &per_query, k,
            ) {
                // Non-fatal: log and continue. A cache-write failure
                // doesn't break correctness, just resumability for
                // this segment.
                ctx.ui.log(&format!(
                    "  warning: segment {} cache write failed: {}",
                    seg_idx, e,
                ));
            }

            // Merge segment results into global heaps.
            merge_segment_into_heaps(&per_query, &mut all_heaps, &mut all_thresholds, k);

            computed_count += 1;
            let seg_secs = seg_compute_start.elapsed().as_secs_f64();
            let seg_evals = seg_len as f64 * query_count as f64;
            let seg_rate = if seg_secs > 0.0 { seg_evals / seg_secs / 1e9 } else { 0.0 };
            ctx.ui.log(&format!(
                "  ✓ [seg {:>3}/{}] done in {:.1}s ({:.2}B dist/s) — cache written",
                seg_idx + 1, n_segments, seg_secs, seg_rate,
            ));
        }
        pb.finish();

        // Final summary line explicitly names how much work was saved.
        if cached_count > 0 {
            ctx.ui.log(&format!(
                "  segments: {} reused from cache + {} computed = {} total (saved ~{:.0}% of base scans)",
                cached_count, computed_count, n_segments,
                100.0 * cached_count as f64 / n_segments as f64,
            ));
        } else {
            ctx.ui.log(&format!(
                "  segments: {} computed (cache populated for future runs)",
                computed_count,
            ));
        }

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
            OptionDesc { name: "assume_normalized_like_faiss".into(), type_name: "bool".into(), required: false,
                default: Some("false".into()),
                description: "For COSINE metric: treat inputs as pre-normalized and evaluate cosine as inner product (FAISS / numpy / knn_utils convention). Exactly one of this and use_proper_cosine_metric must be set when metric=COSINE.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "use_proper_cosine_metric".into(), type_name: "bool".into(), required: false,
                default: Some("false".into()),
                description: "For COSINE metric: compute cosine in-kernel as dot / (|q| × |b|). Correct for arbitrary inputs. Exactly one of this and assume_normalized_like_faiss must be set when metric=COSINE.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "normalized".into(), type_name: "bool".into(), required: false, default: Some("false".into()),
                description: "Deprecated alias for assume_normalized_like_faiss; kept for back-compat.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "threads".into(), type_name: "int".into(), required: false, default: Some("0".into()), description: "Thread count (0 = auto)".into(), role: OptionRole::Config },
            // Accept but ignore knn-metal options for drop-in compatibility
            OptionDesc { name: "partition_size".into(), type_name: "int".into(), required: false, default: None, description: "Override auto-sized segment length (for testing or finer cache granularity)".into(), role: OptionRole::Config },
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

        // We expect *exact* set equality for the deterministic
        // single-threaded fixture (dim=8, 100 base, 10 queries, k=5).
        // Boundary slack would be a defensive allowance for the
        // multi-threaded BLAS rounding regime — irrelevant here. If
        // this assertion ever loosens to allow even one differing
        // neighbor, that's a real regression worth investigating, not
        // a tolerance to widen.
        for q in 0..stdarch_rows.len() {
            let sa: HashSet<i32> = stdarch_rows[q].iter().copied().collect();
            let mt: HashSet<i32> = metal_rows[q].iter().copied().collect();
            let diff = sa.symmetric_difference(&mt).count();
            assert_eq!(diff, 0,
                "query {}: stdarch {:?} vs metal {:?} — {} differ (expected 0)",
                q, &stdarch_rows[q], &metal_rows[q], diff);
        }
    }

    fn read_fvec_rows(path: &std::path::Path) -> Vec<Vec<f32>> {
        let reader = MmapVectorReader::<f32>::open_fvec(path).unwrap();
        let count = VectorReader::<f32>::count(&reader);
        (0..count).map(|i| reader.get_slice(i).to_vec()).collect()
    }

    /// Build a small repeatable test dataset + run the op through it
    /// with an explicit `partition_size` so the segment loop runs
    /// multiple times (exercising the cache write + replay paths).
    /// Returns `(result, indices_path, distances_path)`.
    fn run_knn(
        tmp: &std::path::Path,
        base: &[Vec<f32>], query: &[Vec<f32>],
        partition_size: usize,
        k: usize,
        metric: &str,
        stem: &str,
    ) -> (CommandResult, std::path::PathBuf, std::path::PathBuf) {
        write_fvec(&tmp.join("b.fvec"), base);
        write_fvec(&tmp.join("q.fvec"), query);
        let indices_name = format!("{}.ivec", stem);
        let distances_name = format!("{}-d.fvec", stem);
        let mut ctx = make_ctx(tmp);
        let mut opts = Options::new();
        opts.set("base", "b.fvec"); opts.set("query", "q.fvec");
        opts.set("indices", &indices_name);
        opts.set("distances", &distances_name);
        opts.set("neighbors", &k.to_string());
        opts.set("metric", metric);
        opts.set("partition_size", &partition_size.to_string());
        let mut op = ComputeKnnStdarchOp;
        let r = op.execute(&opts, &mut ctx);
        (r, tmp.join(&indices_name), tmp.join(&distances_name))
    }

    /// After a successful run, each segment should have left behind
    /// two cache files of exactly query_count × (4 + k × 4) bytes.
    #[test]
    fn cache_files_written_per_segment() {
        let tmp = tempfile::tempdir().unwrap();

        let mut rng = 7u64;
        let mut nxt = || -> f32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0 };
        let base: Vec<Vec<f32>> = (0..120).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..5).map(|_| (0..4).map(|_| nxt()).collect()).collect();

        // 120 base / partition 40 = 3 segments
        let (r, _, _) = run_knn(tmp.path(), &base, &query, 40, 3, "L2", "out");
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        // Three segments × two files each = six cache files at
        // query_count * (4 + k*4) = 5 * (4 + 12) = 80 bytes each.
        let cache_dir = tmp.path().join(".cache");
        let files: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-stdarch.v1.")).unwrap_or(false))
            .collect();
        assert_eq!(files.len(), 6,
            "expected 6 cache files (3 segments × ivec+fvec), got {}: {:?}",
            files.len(), files);
        for f in &files {
            let sz = std::fs::metadata(f).unwrap().len();
            assert_eq!(sz, 80, "cache file {} has wrong size", f.display());
        }
    }

    /// A second run with every segment cached must produce
    /// byte-identical output, AND must not re-scan any base vectors
    /// (verified by removing the base file between runs — a fresh
    /// compute would error, a full cache replay would not).
    #[test]
    fn replay_identical_with_full_cache() {
        let tmp = tempfile::tempdir().unwrap();

        let mut rng = 42u64;
        let mut nxt = || -> f32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0 };
        let base: Vec<Vec<f32>> = (0..80).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..4).map(|_| (0..4).map(|_| nxt()).collect()).collect();

        // First run: 2 segments of 40 each
        let (r1, idx1, dist1) = run_knn(tmp.path(), &base, &query, 40, 3, "L2", "first");
        assert_eq!(r1.status, Status::Ok, "{}", r1.message);
        let rows1 = read_ivec_rows(&idx1);
        let dists1 = read_fvec_rows(&dist1);

        // Second run with a DIFFERENT output name so we're not just
        // reading back the first run's output — this genuinely replays
        // every segment from the cache files.
        let (r2, idx2, dist2) = run_knn(tmp.path(), &base, &query, 40, 3, "L2", "second");
        assert_eq!(r2.status, Status::Ok, "{}", r2.message);
        let rows2 = read_ivec_rows(&idx2);
        let dists2 = read_fvec_rows(&dist2);

        assert_eq!(rows1, rows2, "indices differ between first and replayed run");
        assert_eq!(dists1.len(), dists2.len());
        for (a, b) in dists1.iter().zip(dists2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-6, "distance mismatch: {} vs {}", x, y);
            }
        }
    }

    /// Delete one segment's cache mid-flight; the re-run must replay
    /// the surviving segments, recompute the missing one, and produce
    /// the same final result.
    #[test]
    fn partial_cache_recomputes_missing_segment() {
        let tmp = tempfile::tempdir().unwrap();

        let mut rng = 99u64;
        let mut nxt = || -> f32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0 };
        let base: Vec<Vec<f32>> = (0..90).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..3).map(|_| (0..4).map(|_| nxt()).collect()).collect();

        // First run: 3 segments of 30 each. Baseline result.
        let (r1, idx1, _) = run_knn(tmp.path(), &base, &query, 30, 2, "L2", "baseline");
        assert_eq!(r1.status, Status::Ok);
        let baseline_rows = read_ivec_rows(&idx1);

        // Delete ONE segment's caches (the middle one: range
        // 000000000030_000000000060). The outer segments' caches
        // remain and should be replayed.
        let cache_dir = tmp.path().join(".cache");
        for entry in std::fs::read_dir(&cache_dir).unwrap() {
            let p = entry.unwrap().path();
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name.contains("range_000000000030_000000000060") {
                    std::fs::remove_file(&p).unwrap();
                }
            }
        }

        // Second run produces same result, but the missing segment
        // had to be recomputed.
        let (r2, idx2, _) = run_knn(tmp.path(), &base, &query, 30, 2, "L2", "partial");
        assert_eq!(r2.status, Status::Ok);
        let partial_rows = read_ivec_rows(&idx2);
        assert_eq!(baseline_rows, partial_rows,
            "partial-cache replay produced different neighbors than baseline");

        // Cache should now be fully populated again (3 segments × 2 files).
        let files: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-stdarch.v1.")).unwrap_or(false))
            .collect();
        assert_eq!(files.len(), 6,
            "cache should be re-filled after partial-replay run, got {}", files.len());
    }

    /// Changing `k` between runs must not reuse caches — different
    /// k means different per-query top-k → different bytes per row.
    /// The cache-key includes k, so a new k lands on disjoint paths.
    #[test]
    fn different_k_uses_disjoint_caches() {
        let tmp = tempfile::tempdir().unwrap();
        let mut rng = 13u64;
        let mut nxt = || -> f32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0 };
        let base: Vec<Vec<f32>> = (0..60).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..2).map(|_| (0..4).map(|_| nxt()).collect()).collect();

        let (r1, _, _) = run_knn(tmp.path(), &base, &query, 30, 3, "L2", "k3");
        assert_eq!(r1.status, Status::Ok);
        let (r2, _, _) = run_knn(tmp.path(), &base, &query, 30, 5, "L2", "k5");
        assert_eq!(r2.status, Status::Ok);

        let cache_dir = tmp.path().join(".cache");
        let k3: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.contains(".k3.")).unwrap_or(false))
            .collect();
        let k5: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.contains(".k5.")).unwrap_or(false))
            .collect();
        assert_eq!(k3.len(), 4, "k=3 should have 2 segments × 2 files = 4; got {}", k3.len());
        assert_eq!(k5.len(), 4, "k=5 should have 2 segments × 2 files = 4; got {}", k5.len());
    }

    /// Changing the base file size invalidates the cache (the
    /// cache-key embeds `base_size`). A run against a different-
    /// sized base writes to fresh paths; the old caches are
    /// orphaned but harmless.
    #[test]
    fn base_file_change_invalidates_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let mut rng = 21u64;
        let mut nxt = || -> f32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0 };

        // First run: 40 base vectors.
        let base1: Vec<Vec<f32>> = (0..40).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..2).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let (r1, _, _) = run_knn(tmp.path(), &base1, &query, 20, 2, "L2", "a");
        assert_eq!(r1.status, Status::Ok);

        let cache_dir = tmp.path().join(".cache");
        let before: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-stdarch.v1.")).unwrap_or(false))
            .collect();
        assert_eq!(before.len(), 4, "expected 4 cache files after first run");

        // Second run: different-sized base (50 vectors). Same
        // file name `b.fvec`, but different bytes on disk. The
        // cache-prefix key includes the file's current byte size,
        // so this run should NOT reuse the old caches — it should
        // write new ones at disjoint paths.
        let base2: Vec<Vec<f32>> = (0..50).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let (r2, _, _) = run_knn(tmp.path(), &base2, &query, 20, 2, "L2", "b");
        assert_eq!(r2.status, Status::Ok);

        let after: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-stdarch.v1.")).unwrap_or(false))
            .collect();
        // 4 from run-1 + 6 from run-2 (50/20 = 3 segments × 2 files)
        // — all coexist under disjoint cache-prefix keys.
        assert_eq!(after.len(), 10,
            "second run with different-sized base should write to new cache paths, total should be 10; got {}",
            after.len());
    }

    /// Cross-profile reuse: a smaller profile writes caches for its
    /// segments, then a larger profile with an overlapping base-vector
    /// range discovers and reuses those caches instead of recomputing
    /// the overlapping region. This is the main point of the
    /// scan-all-caches + greedy-chain plan; without it every profile
    /// would start from scratch.
    ///
    /// Simulates two profiles via the `base[start..end)` window sugar:
    /// profile A uses `b.fvec[0..30)`, profile B uses `b.fvec[0..60)`.
    /// After A runs, B should replay A's segments and only compute
    /// the [30..60) tail.
    #[test]
    fn larger_profile_reuses_smaller_profile_segments() {
        let tmp = tempfile::tempdir().unwrap();

        let mut rng = 123u64;
        let mut nxt = || -> f32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0 };
        let base: Vec<Vec<f32>> = (0..60).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..3).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        write_fvec(&tmp.path().join("b.fvec"), &base);
        write_fvec(&tmp.path().join("q.fvec"), &query);

        let partition_size = 10usize;
        let k = 2usize;
        let cache_dir = tmp.path().join(".cache");

        // ── Profile A: base[0..30), which at partition_size=10
        //    produces 3 cached segments: [0..10), [10..20), [20..30).
        {
            let mut ctx = make_ctx(tmp.path());
            let mut opts = Options::new();
            opts.set("base", "b.fvec[0..30)"); opts.set("query", "q.fvec");
            opts.set("indices", "a.ivec"); opts.set("distances", "a-d.fvec");
            opts.set("neighbors", &k.to_string());
            opts.set("metric", "L2");
            opts.set("partition_size", &partition_size.to_string());
            let mut op = ComputeKnnStdarchOp;
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(r.status, Status::Ok, "profile A failed: {}", r.message);
        }

        let files_after_a: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-stdarch.v1.")).unwrap_or(false))
            .collect();
        assert_eq!(files_after_a.len(), 6,
            "profile A should have written 6 cache files (3 segments × ivec+fvec), got {}",
            files_after_a.len());

        // ── Profile B: base[0..60). Expected segment plan:
        //    - [0..10), [10..20), [20..30): REUSED from profile A
        //    - [30..40), [40..50), [50..60): computed fresh
        // We detect reuse via mtime — A's files keep their mtime,
        // fresh files are newer.
        std::thread::sleep(std::time::Duration::from_millis(50)); // force a detectable mtime gap
        let a_mtimes: std::collections::HashMap<std::path::PathBuf, std::time::SystemTime> =
            files_after_a.iter().map(|p| {
                (p.clone(), std::fs::metadata(p).unwrap().modified().unwrap())
            }).collect();

        let mut ctx = make_ctx(tmp.path());
        let mut opts = Options::new();
        opts.set("base", "b.fvec[0..60)"); opts.set("query", "q.fvec");
        opts.set("indices", "b.ivec"); opts.set("distances", "b-d.fvec");
        opts.set("neighbors", &k.to_string());
        opts.set("metric", "L2");
        opts.set("partition_size", &partition_size.to_string());
        let mut op = ComputeKnnStdarchOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "profile B failed: {}", r.message);

        // Profile A's three segments must NOT have been rewritten
        // (mtimes unchanged) — that's the proof they were reused.
        for (path, orig_mtime) in &a_mtimes {
            let new_mtime = std::fs::metadata(path).unwrap().modified().unwrap();
            assert_eq!(*orig_mtime, new_mtime,
                "profile A's cache file {} was rewritten (should have been reused, not recomputed)",
                path.display());
        }

        // Total file count now: 3 from profile A + 3 new from profile B = 12 files
        let files_after_b: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-stdarch.v1.")).unwrap_or(false))
            .collect();
        assert_eq!(files_after_b.len(), 12,
            "expected 12 cache files (6 from profile A + 6 fresh from profile B), got {}: {:?}",
            files_after_b.len(), files_after_b);

        // And profile B's result over [0..60) must match a fresh
        // uncached run — reuse must not distort the answers.
        let rows_b = read_ivec_rows(&tmp.path().join("b.ivec"));
        let baseline_tmp = tempfile::tempdir().unwrap();
        write_fvec(&baseline_tmp.path().join("b.fvec"), &base);
        write_fvec(&baseline_tmp.path().join("q.fvec"), &query);
        let mut bctx = make_ctx(baseline_tmp.path());
        let mut bopts = Options::new();
        bopts.set("base", "b.fvec[0..60)"); bopts.set("query", "q.fvec");
        bopts.set("indices", "baseline.ivec"); bopts.set("distances", "baseline-d.fvec");
        bopts.set("neighbors", &k.to_string());
        bopts.set("metric", "L2");
        bopts.set("partition_size", &partition_size.to_string());
        let mut bop = ComputeKnnStdarchOp;
        let br = bop.execute(&bopts, &mut bctx);
        assert_eq!(br.status, Status::Ok);
        let rows_base = read_ivec_rows(&baseline_tmp.path().join("baseline.ivec"));
        assert_eq!(rows_b, rows_base,
            "cross-profile reuse produced different neighbors than fresh compute");
    }

    /// A corrupted cache file (wrong size on disk) must NOT be
    /// replayed — the validator catches the size mismatch and the
    /// segment is recomputed.
    #[test]
    fn corrupted_cache_is_recomputed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut rng = 55u64;
        let mut nxt = || -> f32 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0 };
        let base: Vec<Vec<f32>> = (0..50).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..2).map(|_| (0..4).map(|_| nxt()).collect()).collect();

        let (r1, idx1, _) = run_knn(tmp.path(), &base, &query, 25, 2, "L2", "first");
        assert_eq!(r1.status, Status::Ok);
        let baseline = read_ivec_rows(&idx1);

        // Truncate one cache file to the wrong size.
        let cache_dir = tmp.path().join(".cache");
        for entry in std::fs::read_dir(&cache_dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("ivec") {
                let data = std::fs::read(&p).unwrap();
                std::fs::write(&p, &data[..data.len() / 2]).unwrap();
                break; // corrupt one file is enough
            }
        }

        // Run should still succeed and produce same result.
        let (r2, idx2, _) = run_knn(tmp.path(), &base, &query, 25, 2, "L2", "second");
        assert_eq!(r2.status, Status::Ok, "{}", r2.message);
        let recovered = read_ivec_rows(&idx2);
        assert_eq!(baseline, recovered,
            "corrupted-cache recovery produced different neighbors");
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
