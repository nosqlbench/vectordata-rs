// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! SIMD-accelerated vector distance functions.
//!
//! L2, Cosine, and DotProduct use the [`simsimd`] crate for hardware-dispatched
//! SIMD (AVX-512, AVX2, NEON, SVE). L1 (Manhattan) uses hand-rolled AVX2/AVX-512
//! since SimSIMD does not provide L1.
//!
//! Feature detection is done once per call to [`select_distance_fn`], which
//! returns a function pointer for the best available implementation.

use simsimd::SpatialSimilarity;

/// Distance metric enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
    DotProduct,
    L1,
}

impl Metric {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "L2" | "EUCLIDEAN" => Some(Metric::L2),
            "COSINE" => Some(Metric::Cosine),
            "DOT_PRODUCT" | "DOTPRODUCT" | "DOT" => Some(Metric::DotProduct),
            "L1" | "MANHATTAN" => Some(Metric::L1),
            _ => None,
        }
    }
}

/// Select the best distance function for the given metric.
///
/// Returns a function pointer that uses the best available SIMD instructions.
pub fn select_distance_fn(metric: Metric) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        Metric::L2 => select_l2(),
        Metric::Cosine => select_cosine(),
        Metric::DotProduct => select_dot_product(),
        Metric::L1 => select_l1(),
    }
}

// -- L2 (Euclidean) distance via SimSIMD --------------------------------------

fn select_l2() -> fn(&[f32], &[f32]) -> f32 {
    // Use squared L2 — sqrt is monotonic so ordering is preserved for KNN.
    // Avoids ~15 cycles per distance in the inner loop.
    |a, b| <f32 as SpatialSimilarity>::l2sq(a, b).unwrap_or(0.0) as f32
}

#[cfg(test)]
fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum // squared L2 — matches select_distance_fn(Metric::L2)
}

// -- Cosine distance via SimSIMD ----------------------------------------------

fn select_cosine() -> fn(&[f32], &[f32]) -> f32 {
    |a, b| <f32 as SpatialSimilarity>::cos(a, b).unwrap_or(1.0) as f32
}

#[cfg(test)]
fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

// -- Dot product via SimSIMD --------------------------------------------------

fn select_dot_product() -> fn(&[f32], &[f32]) -> f32 {
    |a, b| -(<f32 as SpatialSimilarity>::dot(a, b).unwrap_or(0.0) as f32)
}

#[cfg(test)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    -dot // Negative: higher dot = more similar = lower distance
}

// -- L1 (Manhattan) distance (hand-rolled, SimSIMD lacks L1) ------------------

fn select_l1() -> fn(&[f32], &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return l1_avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return l1_avx2;
        }
    }
    l1_scalar
}

fn l1_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_avx2_inner(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let mut sum_vec = _mm256_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_and_ps(diff, sign_mask);
            sum_vec = _mm256_add_ps(sum_vec, abs_diff);
        }

        let mut sum = hsum256(sum_vec);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += (a[tail_start + i] - b[tail_start + i]).abs();
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
fn l1_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l1_avx2_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l1_avx512_inner(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 16;
        let remainder = n % 16;

        let mut sum_vec = _mm512_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a_ptr.add(offset));
            let vb = _mm512_loadu_ps(b_ptr.add(offset));
            let diff = _mm512_sub_ps(va, vb);
            let abs_diff = _mm512_abs_ps(diff);
            sum_vec = _mm512_add_ps(sum_vec, abs_diff);
        }

        let mut sum = _mm512_reduce_add_ps(sum_vec);

        let tail_start = chunks * 16;
        for i in 0..remainder {
            sum += (a[tail_start + i] - b[tail_start + i]).abs();
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
fn l1_avx512(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l1_avx512_inner(a, b) }
}

// -- AVX2 horizontal sum helper (used by L1) ----------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

// -- f16 distance functions ---------------------------------------------------

/// Select the best distance function for the given metric operating on f16 vectors.
///
/// L2, Cosine, and Dot use SimSIMD's native f16 SIMD kernels (widening to f32
/// happens at the register level inside the kernel). L1 uses hand-rolled
/// AVX-512/AVX2 kernels that convert f16 to f32 in-register via `vcvtph2ps`.
///
/// Both `half::f16` and `simsimd::f16` are `#[repr(transparent)]` wrappers
/// around `u16` with IEEE 754 half-precision layout, so the pointer cast
/// between them is safe.
pub fn select_distance_fn_f16(metric: Metric) -> fn(&[half::f16], &[half::f16]) -> f32 {
    match metric {
        Metric::L2 => select_l2_f16(),
        Metric::Cosine => select_cosine_f16(),
        Metric::DotProduct => select_dot_product_f16(),
        Metric::L1 => select_l1_f16(),
    }
}

/// Reinterpret a `&[half::f16]` slice as `&[simsimd::f16]`.
///
/// # Safety
///
/// Both types are `#[repr(transparent)]` wrappers around `u16` with identical
/// IEEE 754 half-precision layout, so this pointer cast is sound.
unsafe fn as_simsimd_f16(s: &[half::f16]) -> &[simsimd::f16] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const simsimd::f16, s.len()) }
}

fn select_l2_f16() -> fn(&[half::f16], &[half::f16]) -> f32 {
    // Use squared L2 — sqrt is monotonic so ordering is preserved for KNN.
    |a, b| {
        let (sa, sb) = unsafe { (as_simsimd_f16(a), as_simsimd_f16(b)) };
        <simsimd::f16 as SpatialSimilarity>::l2sq(sa, sb).unwrap_or(0.0) as f32
    }
}

fn select_cosine_f16() -> fn(&[half::f16], &[half::f16]) -> f32 {
    |a, b| {
        let (sa, sb) = unsafe { (as_simsimd_f16(a), as_simsimd_f16(b)) };
        <simsimd::f16 as SpatialSimilarity>::cos(sa, sb).unwrap_or(1.0) as f32
    }
}

fn select_dot_product_f16() -> fn(&[half::f16], &[half::f16]) -> f32 {
    |a, b| {
        let (sa, sb) = unsafe { (as_simsimd_f16(a), as_simsimd_f16(b)) };
        -(<simsimd::f16 as SpatialSimilarity>::dot(sa, sb).unwrap_or(0.0) as f32)
    }
}

fn select_l1_f16() -> fn(&[half::f16], &[half::f16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return l1_f16_avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return l1_f16_avx2;
        }
    }
    l1_f16_scalar
}

fn l1_f16_scalar(a: &[half::f16], b: &[half::f16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i].to_f32() - b[i].to_f32()).abs();
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn l1_f16_avx2_inner(a: &[half::f16], b: &[half::f16]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let mut sum_vec = _mm256_setzero_ps();
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;

        for i in 0..chunks {
            let offset = i * 8;
            // Load 8 f16 values as 128-bit integer, convert to 8 f32 values
            let va_i = _mm_loadu_si128(a_ptr.add(offset) as *const __m128i);
            let vb_i = _mm_loadu_si128(b_ptr.add(offset) as *const __m128i);
            let va = _mm256_cvtph_ps(va_i);
            let vb = _mm256_cvtph_ps(vb_i);
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_and_ps(diff, sign_mask);
            sum_vec = _mm256_add_ps(sum_vec, abs_diff);
        }

        let mut sum = hsum256(sum_vec);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += (a[tail_start + i].to_f32() - b[tail_start + i].to_f32()).abs();
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
fn l1_f16_avx2(a: &[half::f16], b: &[half::f16]) -> f32 {
    unsafe { l1_f16_avx2_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "f16c")]
unsafe fn l1_f16_avx512_inner(a: &[half::f16], b: &[half::f16]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let n = a.len();
        let chunks = n / 16;
        let remainder = n % 16;

        let mut sum_vec = _mm512_setzero_ps();
        let a_ptr = a.as_ptr() as *const u16;
        let b_ptr = b.as_ptr() as *const u16;

        for i in 0..chunks {
            let offset = i * 16;
            // Load 16 f16 values as 256-bit integer, convert to 16 f32 values
            let va_i = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
            let vb_i = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);
            let va = _mm512_cvtph_ps(va_i);
            let vb = _mm512_cvtph_ps(vb_i);
            let diff = _mm512_sub_ps(va, vb);
            let abs_diff = _mm512_abs_ps(diff);
            sum_vec = _mm512_add_ps(sum_vec, abs_diff);
        }

        let mut sum = _mm512_reduce_add_ps(sum_vec);

        let tail_start = chunks * 16;
        for i in 0..remainder {
            sum += (a[tail_start + i].to_f32() - b[tail_start + i].to_f32()).abs();
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
fn l1_f16_avx512(a: &[half::f16], b: &[half::f16]) -> f32 {
    unsafe { l1_f16_avx512_inner(a, b) }
}

// -- Batched multi-query distance (transposed SIMD) ---------------------------

/// Width of the transposed SIMD batch — matches AVX-512 f32 lane count.
///
/// Each base vector dimension is broadcast to all 16 lanes, and 16 query
/// distances are computed simultaneously with a single FMA instruction.
pub const SIMD_BATCH_WIDTH: usize = 16;

/// Transposed query batch for SIMD-optimized multi-query distance computation.
///
/// Stores up to [`SIMD_BATCH_WIDTH`] queries in dimension-major (columnar)
/// layout, with all values pre-converted to f32. Layout:
/// `data[d * SIMD_BATCH_WIDTH + qi]` = f32 value for dimension `d` of query `qi`.
/// Unused lanes (fewer than 16 queries) are zero-padded.
///
/// This layout enables the inner distance loop to process all queries in
/// parallel using a single SIMD instruction per dimension:
/// 1. Broadcast `base[d]` to all 16 lanes
/// 2. Load `transposed[d][0..16]` (contiguous f32 vector)
/// 3. FMA: `acc += (base[d] - query[d])²` for all 16 queries
///
/// Pre-converting f16 to f32 during construction eliminates redundant
/// per-base-vector conversion in the inner loop (512× fewer conversions
/// for 512-dim vectors in a batch of 256 queries).
pub struct TransposedBatch {
    data: Vec<f32>,
    /// Precomputed L2 norm of each query (for cosine distance).
    /// Length = SIMD_BATCH_WIDTH, unused slots = 1.0 to avoid division issues.
    pub query_norms: [f32; SIMD_BATCH_WIDTH],
    dim: usize,
    count: usize,
}

impl TransposedBatch {
    /// Create a transposed batch from f16 query slices.
    ///
    /// Converts all values to f32 and transposes from query-major
    /// `[queries][dims]` to dimension-major `[dims][SIMD_BATCH_WIDTH]`.
    pub fn from_f16(queries: &[&[half::f16]], dim: usize) -> Self {
        assert!(queries.len() <= SIMD_BATCH_WIDTH);
        let count = queries.len();
        let mut data = vec![0.0f32; dim * SIMD_BATCH_WIDTH];
        let mut query_norms = [1.0f32; SIMD_BATCH_WIDTH]; // default 1.0 for unused slots
        for (qi, q) in queries.iter().enumerate() {
            let mut norm_sq = 0.0f32;
            for d in 0..dim {
                let v = q[d].to_f32();
                data[d * SIMD_BATCH_WIDTH + qi] = v;
                norm_sq += v * v;
            }
            query_norms[qi] = norm_sq.sqrt().max(f32::EPSILON);
        }
        Self { data, query_norms, dim, count }
    }

    /// Create a transposed batch from f32 query slices.
    pub fn from_f32(queries: &[&[f32]], dim: usize) -> Self {
        assert!(queries.len() <= SIMD_BATCH_WIDTH);
        let count = queries.len();
        let mut data = vec![0.0f32; dim * SIMD_BATCH_WIDTH];
        let mut query_norms = [1.0f32; SIMD_BATCH_WIDTH];
        for (qi, q) in queries.iter().enumerate() {
            let mut norm_sq = 0.0f32;
            for d in 0..dim {
                data[d * SIMD_BATCH_WIDTH + qi] = q[d];
                norm_sq += q[d] * q[d];
            }
            query_norms[qi] = norm_sq.sqrt().max(f32::EPSILON);
        }
        Self { data, query_norms, dim, count }
    }

    /// Number of actual queries in this batch (≤ SIMD_BATCH_WIDTH).
    pub fn count(&self) -> usize {
        self.count
    }

    /// Raw pointer to the transposed data.
    ///
    /// Layout: `data[d * SIMD_BATCH_WIDTH + qi]` = f32 value for dimension
    /// `d` of query `qi`.
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Dimensionality of the vectors in this batch.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Contiguous packed layout for all sub-batches in a thread's query set.
///
/// Layout: `data[d * n_batches * SIMD_BATCH_WIDTH + si * SIMD_BATCH_WIDTH + qi]`
/// where `d` is the dimension, `si` is the sub-batch index, and `qi` is the
/// query index within the sub-batch (0..16).
///
/// This puts all sub-batches for a given dimension in adjacent cache lines,
/// giving the hardware prefetcher a single sequential stream to track instead
/// of N scattered streams from separate heap allocations.
pub struct PackedBatches {
    data: Vec<f32>,
    dim: usize,
    n_batches: usize,
    /// Number of actual queries per sub-batch (last may be partial).
    counts: Vec<usize>,
    /// Query offset for each sub-batch (index into the original query array).
    offsets: Vec<usize>,
}

impl PackedBatches {
    /// Pack f32 query slices into the contiguous interleaved layout.
    pub fn from_f32(queries: &[&[f32]], dim: usize) -> Self {
        let n_queries = queries.len();
        let n_batches = (n_queries + SIMD_BATCH_WIDTH - 1) / SIMD_BATCH_WIDTH;
        let stride = n_batches * SIMD_BATCH_WIDTH; // elements per dimension
        let mut data = vec![0.0f32; dim * stride];
        let mut counts = Vec::with_capacity(n_batches);
        let mut offsets = Vec::with_capacity(n_batches);

        let mut qi_global = 0;
        for si in 0..n_batches {
            let batch_start = si * SIMD_BATCH_WIDTH;
            let batch_end = (batch_start + SIMD_BATCH_WIDTH).min(n_queries);
            let count = batch_end - batch_start;
            counts.push(count);
            offsets.push(batch_start);

            for qi in 0..count {
                let q = queries[batch_start + qi];
                for d in 0..dim {
                    data[d * stride + si * SIMD_BATCH_WIDTH + qi] = q[d];
                }
                qi_global += 1;
            }
        }
        let _ = qi_global;

        Self { data, dim, n_batches, counts, offsets }
    }

    /// Pack f16 query slices into the contiguous interleaved layout (f32).
    pub fn from_f16(queries: &[&[half::f16]], dim: usize) -> Self {
        let n_queries = queries.len();
        let n_batches = (n_queries + SIMD_BATCH_WIDTH - 1) / SIMD_BATCH_WIDTH;
        let stride = n_batches * SIMD_BATCH_WIDTH;
        let mut data = vec![0.0f32; dim * stride];
        let mut counts = Vec::with_capacity(n_batches);
        let mut offsets = Vec::with_capacity(n_batches);

        for si in 0..n_batches {
            let batch_start = si * SIMD_BATCH_WIDTH;
            let batch_end = (batch_start + SIMD_BATCH_WIDTH).min(n_queries);
            counts.push(batch_end - batch_start);
            offsets.push(batch_start);

            for qi in 0..(batch_end - batch_start) {
                let q = queries[batch_start + qi];
                for d in 0..dim {
                    data[d * stride + si * SIMD_BATCH_WIDTH + qi] = q[d].to_f32();
                }
            }
        }

        Self { data, dim, n_batches, counts, offsets }
    }

    /// Number of sub-batches.
    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    /// Number of actual queries in sub-batch `si`.
    pub fn count(&self, si: usize) -> usize {
        self.counts[si]
    }

    /// Query offset for sub-batch `si` (index into the original query array).
    pub fn offset(&self, si: usize) -> usize {
        self.offsets[si]
    }

    /// Raw pointer to the packed data.
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Elements per dimension row (n_batches × SIMD_BATCH_WIDTH).
    pub fn row_stride(&self) -> usize {
        self.n_batches * SIMD_BATCH_WIDTH
    }
}

/// Whether `metric` should use the packed neg-dot kernel path.
///
/// Only DotProduct is eligible: the packed kernel computes raw `-dot(a, b)`
/// which is only a valid distance for unit-length vectors where cosine
/// reduces to negative dot product. All other metrics (Cosine, L2, L1) must
/// use the TransposedBatch path which applies the correct per-metric formula
/// (norm division, squared differences, etc.).
///
/// Every command that chooses between packed and TransposedBatch paths must
/// call this function instead of making the decision locally — duplicating
/// this logic is how compute-knn and verify-knn diverged in the past.
pub fn metric_uses_packed_path(metric: Metric) -> bool {
    metric == Metric::DotProduct
}

/// Function type for batched distance computation on f16 base vectors.
pub type BatchedDistFnF16 = fn(&TransposedBatch, &[half::f16], &mut [f32]);

/// Function type for batched distance computation on f32 base vectors.
pub type BatchedDistFnF32 = fn(&TransposedBatch, &[f32], &mut [f32]);

/// Select a batched SIMD distance function for f16 vectors, if available.
///
/// Returns `Some(fn)` for metrics with batched SIMD implementations,
/// `None` otherwise (caller should fall back to per-pair distance).
pub fn select_batched_fn_f16(metric: Metric) -> Option<BatchedDistFnF16> {
    match metric {
        Metric::L2 => Some(batched_l2sq_f16),
        Metric::DotProduct => Some(batched_neg_dot_f16),
        Metric::Cosine => Some(batched_cosine_f16),
        Metric::L1 => Some(batched_l1_f16),
    }
}

/// Select a batched SIMD distance function for f32 vectors, if available.
pub fn select_batched_fn_f32(metric: Metric) -> Option<BatchedDistFnF32> {
    match metric {
        Metric::L2 => Some(batched_l2sq_f32),
        Metric::DotProduct => Some(batched_neg_dot_f32),
        Metric::Cosine => Some(batched_cosine_f32),
        Metric::L1 => Some(batched_l1_f32),
    }
}

/// Function type for dual-accumulator batched distance computation.
/// Processes two TransposedBatches (32 queries) per base vector load,
/// feeding both FMA ports simultaneously. Output is 32 f32 distances.
pub type DualBatchedDistFnF32 = fn(&TransposedBatch, &TransposedBatch, &[f32], &mut [f32; 32]);

/// Select a dual-accumulator batched distance function for f32 vectors.
pub fn select_dual_batched_fn_f32(metric: Metric) -> Option<DualBatchedDistFnF32> {
    match metric {
        Metric::L2 => Some(dual_batched_l2sq_f32),
        Metric::DotProduct => Some(dual_batched_neg_dot_f32),
        Metric::Cosine => Some(dual_batched_cosine_f32),
        Metric::L1 => Some(dual_batched_l1_f32),
    }
}

// -- Batched L2 squared -------------------------------------------------------

/// Compute L2 squared distances from one f16 base vector to all queries
/// in a transposed batch. Dispatches to AVX-512 when available.
fn batched_l2sq_f16(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_l2sq_f16_avx512(batch, base, out); }
            return;
        }
    }
    batched_l2sq_scalar_from_f16(batch, base, out);
}

fn batched_l2sq_scalar_from_f16(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    let dim = batch.dim;
    for qi in 0..batch.count {
        let mut sum = 0.0f32;
        for d in 0..dim {
            let diff = base[d].to_f32() - batch.data[d * SIMD_BATCH_WIDTH + qi];
            sum += diff * diff;
        }
        out[qi] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_l2sq_f16_avx512(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();

        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let bval = base[d].to_f32();
            let base_bc = _mm512_set1_ps(bval);
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            let diff = _mm512_sub_ps(base_bc, q16);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        _mm512_storeu_ps(out.as_mut_ptr(), acc);
    }
}

/// Compute L2 squared distances from one f32 base vector to all queries.
fn batched_l2sq_f32(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_l2sq_f32_avx512(batch, base, out); }
            return;
        }
    }
    batched_l2sq_scalar_from_f32(batch, base, out);
}

fn batched_l2sq_scalar_from_f32(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    let dim = batch.dim;
    for qi in 0..batch.count {
        let mut sum = 0.0f32;
        for d in 0..dim {
            let diff = base[d] - batch.data[d * SIMD_BATCH_WIDTH + qi];
            sum += diff * diff;
        }
        out[qi] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_l2sq_f32_avx512(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();
        let b_ptr = base.as_ptr();

        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let base_bc = _mm512_set1_ps(*b_ptr.add(d));
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            let diff = _mm512_sub_ps(base_bc, q16);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        _mm512_storeu_ps(out.as_mut_ptr(), acc);
    }
}

// -- Batched negative dot product ---------------------------------------------

/// Compute negative dot product distances from one f16 base vector to all
/// queries in a transposed batch.
fn batched_neg_dot_f16(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_neg_dot_f16_avx512(batch, base, out); }
            return;
        }
    }
    let dim = batch.dim;
    for qi in 0..batch.count {
        let mut dot = 0.0f32;
        for d in 0..dim {
            dot += base[d].to_f32() * batch.data[d * SIMD_BATCH_WIDTH + qi];
        }
        out[qi] = -dot;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_neg_dot_f16_avx512(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();
        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let bval = base[d].to_f32();
            let base_bc = _mm512_set1_ps(bval);
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            acc = _mm512_fmadd_ps(base_bc, q16, acc);
        }

        // Negate: distance = -dot (higher dot = more similar = lower distance)
        let neg = _mm512_sub_ps(_mm512_setzero_ps(), acc);
        _mm512_storeu_ps(out.as_mut_ptr(), neg);
    }
}

/// Compute negative dot product distances from one f32 base vector.
fn batched_neg_dot_f32(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_neg_dot_f32_avx512(batch, base, out); }
            return;
        }
    }
    let dim = batch.dim;
    for qi in 0..batch.count {
        let mut dot = 0.0f32;
        for d in 0..dim {
            dot += base[d] * batch.data[d * SIMD_BATCH_WIDTH + qi];
        }
        out[qi] = -dot;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_neg_dot_f32_avx512(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();
        let b_ptr = base.as_ptr();
        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let base_bc = _mm512_set1_ps(*b_ptr.add(d));
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            acc = _mm512_fmadd_ps(base_bc, q16, acc);
        }

        let neg = _mm512_sub_ps(_mm512_setzero_ps(), acc);
        _mm512_storeu_ps(out.as_mut_ptr(), neg);
    }
}

// -- Batched Cosine -----------------------------------------------------------
//
// Cosine distance = 1 - dot(a,b)/(|a|*|b|).
// Query norms are precomputed in TransposedBatch::query_norms.
// Base norm is computed per-call (same for all 16 queries).
// This is correct for both normalized and unnormalized vectors.
// For normalized vectors, norms are 1.0 and the division is a no-op.

/// Compute cosine distances from one f16 base vector to all queries.
fn batched_cosine_f16(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_cosine_f16_avx512(batch, base, out); }
            return;
        }
    }
    let dim = batch.dim;
    let mut base_norm_sq = 0.0f32;
    for qi in 0..batch.count {
        let mut dot = 0.0f32;
        for d in 0..dim {
            let bv = base[d].to_f32();
            dot += bv * batch.data[d * SIMD_BATCH_WIDTH + qi];
            if qi == 0 { base_norm_sq += bv * bv; }
        }
        let base_norm = if qi == 0 { base_norm_sq.sqrt().max(f32::EPSILON) } else { base_norm_sq.sqrt().max(f32::EPSILON) };
        out[qi] = 1.0 - dot / (batch.query_norms[qi] * base_norm);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_cosine_f16_avx512(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();
        let mut dot_acc = _mm512_setzero_ps();
        let mut base_norm_sq = 0.0f32;

        for d in 0..dim {
            let bval = base[d].to_f32();
            base_norm_sq += bval * bval;
            let base_bc = _mm512_set1_ps(bval);
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            dot_acc = _mm512_fmadd_ps(base_bc, q16, dot_acc);
        }

        let base_norm = base_norm_sq.sqrt().max(f32::EPSILON);
        // Load query norms and compute base_norm * query_norm per lane
        let q_norms = _mm512_loadu_ps(batch.query_norms.as_ptr());
        let b_norm_bc = _mm512_set1_ps(base_norm);
        let norm_product = _mm512_mul_ps(b_norm_bc, q_norms);

        // cosine_dist = 1 - dot / (|b| * |q|)
        let similarity = _mm512_div_ps(dot_acc, norm_product);
        let ones = _mm512_set1_ps(1.0);
        let result = _mm512_sub_ps(ones, similarity);
        _mm512_storeu_ps(out.as_mut_ptr(), result);
    }
}

/// Compute cosine distances from one f32 base vector to all queries.
fn batched_cosine_f32(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_cosine_f32_avx512(batch, base, out); }
            return;
        }
    }
    let dim = batch.dim;
    let mut base_norm_sq = 0.0f32;
    for d in 0..dim {
        base_norm_sq += base[d] * base[d];
    }
    let base_norm = base_norm_sq.sqrt().max(f32::EPSILON);
    for qi in 0..batch.count {
        let mut dot = 0.0f32;
        for d in 0..dim {
            dot += base[d] * batch.data[d * SIMD_BATCH_WIDTH + qi];
        }
        out[qi] = 1.0 - dot / (batch.query_norms[qi] * base_norm);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_cosine_f32_avx512(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();
        let b_ptr = base.as_ptr();
        let mut dot_acc = _mm512_setzero_ps();
        let mut base_norm_sq = 0.0f32;

        for d in 0..dim {
            let bv = *b_ptr.add(d);
            base_norm_sq += bv * bv;
            let bval = _mm512_set1_ps(bv);
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            dot_acc = _mm512_fmadd_ps(bval, q16, dot_acc);
        }

        let base_norm = base_norm_sq.sqrt().max(f32::EPSILON);

        let q_norms = _mm512_loadu_ps(batch.query_norms.as_ptr());
        let b_norm_bc = _mm512_set1_ps(base_norm);
        let norm_product = _mm512_mul_ps(b_norm_bc, q_norms);

        let similarity = _mm512_div_ps(dot_acc, norm_product);
        let ones = _mm512_set1_ps(1.0);
        let result = _mm512_sub_ps(ones, similarity);
        _mm512_storeu_ps(out.as_mut_ptr(), result);
    }
}

// -- Batched L1 (Manhattan) ---------------------------------------------------

/// Compute L1 distances from one f16 base vector to all queries.
fn batched_l1_f16(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_l1_f16_avx512(batch, base, out); }
            return;
        }
    }
    let dim = batch.dim;
    for qi in 0..batch.count {
        let mut sum = 0.0f32;
        for d in 0..dim {
            sum += (base[d].to_f32() - batch.data[d * SIMD_BATCH_WIDTH + qi]).abs();
        }
        out[qi] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_l1_f16_avx512(batch: &TransposedBatch, base: &[half::f16], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();
        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let bval = base[d].to_f32();
            let base_bc = _mm512_set1_ps(bval);
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            let diff = _mm512_sub_ps(base_bc, q16);
            let abs_diff = _mm512_abs_ps(diff);
            acc = _mm512_add_ps(acc, abs_diff);
        }

        _mm512_storeu_ps(out.as_mut_ptr(), acc);
    }
}

/// Compute L1 distances from one f32 base vector to all queries.
fn batched_l1_f32(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { batched_l1_f32_avx512(batch, base, out); }
            return;
        }
    }
    let dim = batch.dim;
    for qi in 0..batch.count {
        let mut sum = 0.0f32;
        for d in 0..dim {
            sum += (base[d] - batch.data[d * SIMD_BATCH_WIDTH + qi]).abs();
        }
        out[qi] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batched_l1_f32_avx512(batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let dim = batch.dim;
        let t_ptr = batch.data.as_ptr();
        let b_ptr = base.as_ptr();
        let mut acc = _mm512_setzero_ps();

        for d in 0..dim {
            let base_bc = _mm512_set1_ps(*b_ptr.add(d));
            let q16 = _mm512_loadu_ps(t_ptr.add(d * SIMD_BATCH_WIDTH));
            let diff = _mm512_sub_ps(base_bc, q16);
            let abs_diff = _mm512_abs_ps(diff);
            acc = _mm512_add_ps(acc, abs_diff);
        }

        _mm512_storeu_ps(out.as_mut_ptr(), acc);
    }
}

// -- Dual-accumulator kernels (32-wide, 2 FMA ports) -------------------------
//
// Each dual kernel processes TWO TransposedBatches per base vector load,
// using two independent zmm accumulators that feed both FMA execution
// ports simultaneously. Benchmarked at 1.7× throughput vs sequential calls.

/// Dual L2 squared: 32 queries per base vector.
fn dual_batched_l2sq_f32(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { dual_batched_l2sq_f32_avx512(a, b, base, out); }
            return;
        }
    }
    // Scalar fallback
    let dim = a.dim;
    for qi in 0..a.count() {
        let mut sum = 0.0f32;
        for d in 0..dim { let diff = base[d] - a.data[d * SIMD_BATCH_WIDTH + qi]; sum += diff * diff; }
        out[qi] = sum;
    }
    for qi in 0..b.count() {
        let mut sum = 0.0f32;
        for d in 0..dim { let diff = base[d] - b.data[d * SIMD_BATCH_WIDTH + qi]; sum += diff * diff; }
        out[16 + qi] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dual_batched_l2sq_f32_avx512(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    unsafe {
        use std::arch::x86_64::*;
        let dim = a.dim;
        let a_ptr = a.data.as_ptr();
        let b_ptr = b.data.as_ptr();
        let base_ptr = base.as_ptr();
        let mut acc_a = _mm512_setzero_ps();
        let mut acc_b = _mm512_setzero_ps();
        for d in 0..dim {
            let bval = _mm512_set1_ps(*base_ptr.add(d));
            let da = _mm512_sub_ps(bval, _mm512_loadu_ps(a_ptr.add(d * 16)));
            let db = _mm512_sub_ps(bval, _mm512_loadu_ps(b_ptr.add(d * 16)));
            acc_a = _mm512_fmadd_ps(da, da, acc_a);
            acc_b = _mm512_fmadd_ps(db, db, acc_b);
        }
        _mm512_storeu_ps(out.as_mut_ptr(), acc_a);
        _mm512_storeu_ps(out.as_mut_ptr().add(16), acc_b);
    }
}

/// Dual negative dot product: 32 queries per base vector.
fn dual_batched_neg_dot_f32(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { dual_batched_neg_dot_f32_avx512(a, b, base, out); }
            return;
        }
    }
    let dim = a.dim;
    for qi in 0..a.count() {
        let mut dot = 0.0f32;
        for d in 0..dim { dot += base[d] * a.data[d * SIMD_BATCH_WIDTH + qi]; }
        out[qi] = -dot;
    }
    for qi in 0..b.count() {
        let mut dot = 0.0f32;
        for d in 0..dim { dot += base[d] * b.data[d * SIMD_BATCH_WIDTH + qi]; }
        out[16 + qi] = -dot;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dual_batched_neg_dot_f32_avx512(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    unsafe {
        use std::arch::x86_64::*;
        let dim = a.dim;
        let a_ptr = a.data.as_ptr();
        let b_ptr = b.data.as_ptr();
        let base_ptr = base.as_ptr();
        let mut acc_a = _mm512_setzero_ps();
        let mut acc_b = _mm512_setzero_ps();
        for d in 0..dim {
            let bval = _mm512_set1_ps(*base_ptr.add(d));
            acc_a = _mm512_fmadd_ps(bval, _mm512_loadu_ps(a_ptr.add(d * 16)), acc_a);
            acc_b = _mm512_fmadd_ps(bval, _mm512_loadu_ps(b_ptr.add(d * 16)), acc_b);
        }
        let zero = _mm512_setzero_ps();
        _mm512_storeu_ps(out.as_mut_ptr(), _mm512_sub_ps(zero, acc_a));
        _mm512_storeu_ps(out.as_mut_ptr().add(16), _mm512_sub_ps(zero, acc_b));
    }
}

/// Dual cosine distance: 32 queries per base vector (with norm division).
fn dual_batched_cosine_f32(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { dual_batched_cosine_f32_avx512(a, b, base, out); }
            return;
        }
    }
    let dim = a.dim;
    let mut base_norm_sq = 0.0f32;
    for d in 0..dim { base_norm_sq += base[d] * base[d]; }
    let base_norm = base_norm_sq.sqrt().max(f32::EPSILON);
    for qi in 0..a.count() {
        let mut dot = 0.0f32;
        for d in 0..dim { dot += base[d] * a.data[d * SIMD_BATCH_WIDTH + qi]; }
        out[qi] = 1.0 - dot / (a.query_norms[qi] * base_norm);
    }
    for qi in 0..b.count() {
        let mut dot = 0.0f32;
        for d in 0..dim { dot += base[d] * b.data[d * SIMD_BATCH_WIDTH + qi]; }
        out[16 + qi] = 1.0 - dot / (b.query_norms[qi] * base_norm);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dual_batched_cosine_f32_avx512(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    unsafe {
        use std::arch::x86_64::*;
        let dim = a.dim;
        let a_ptr = a.data.as_ptr();
        let b_ptr = b.data.as_ptr();
        let base_ptr = base.as_ptr();
        let mut dot_a = _mm512_setzero_ps();
        let mut dot_b = _mm512_setzero_ps();
        let mut base_norm_sq = 0.0f32;
        for d in 0..dim {
            let bv = *base_ptr.add(d);
            base_norm_sq += bv * bv;
            let bval = _mm512_set1_ps(bv);
            dot_a = _mm512_fmadd_ps(bval, _mm512_loadu_ps(a_ptr.add(d * 16)), dot_a);
            dot_b = _mm512_fmadd_ps(bval, _mm512_loadu_ps(b_ptr.add(d * 16)), dot_b);
        }
        let base_norm = base_norm_sq.sqrt().max(f32::EPSILON);
        let b_norm_bc = _mm512_set1_ps(base_norm);
        let ones = _mm512_set1_ps(1.0);
        let norm_a = _mm512_mul_ps(b_norm_bc, _mm512_loadu_ps(a.query_norms.as_ptr()));
        let norm_b = _mm512_mul_ps(b_norm_bc, _mm512_loadu_ps(b.query_norms.as_ptr()));
        _mm512_storeu_ps(out.as_mut_ptr(), _mm512_sub_ps(ones, _mm512_div_ps(dot_a, norm_a)));
        _mm512_storeu_ps(out.as_mut_ptr().add(16), _mm512_sub_ps(ones, _mm512_div_ps(dot_b, norm_b)));
    }
}

/// Dual L1 (Manhattan): 32 queries per base vector.
fn dual_batched_l1_f32(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { dual_batched_l1_f32_avx512(a, b, base, out); }
            return;
        }
    }
    let dim = a.dim;
    for qi in 0..a.count() {
        let mut sum = 0.0f32;
        for d in 0..dim { sum += (base[d] - a.data[d * SIMD_BATCH_WIDTH + qi]).abs(); }
        out[qi] = sum;
    }
    for qi in 0..b.count() {
        let mut sum = 0.0f32;
        for d in 0..dim { sum += (base[d] - b.data[d * SIMD_BATCH_WIDTH + qi]).abs(); }
        out[16 + qi] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dual_batched_l1_f32_avx512(a: &TransposedBatch, b: &TransposedBatch, base: &[f32], out: &mut [f32; 32]) {
    unsafe {
        use std::arch::x86_64::*;
        let dim = a.dim;
        let a_ptr = a.data.as_ptr();
        let b_ptr = b.data.as_ptr();
        let base_ptr = base.as_ptr();
        let mut acc_a = _mm512_setzero_ps();
        let mut acc_b = _mm512_setzero_ps();
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
        for d in 0..dim {
            let bval = _mm512_set1_ps(*base_ptr.add(d));
            let da = _mm512_and_ps(_mm512_sub_ps(bval, _mm512_loadu_ps(a_ptr.add(d * 16))), sign_mask);
            let db = _mm512_and_ps(_mm512_sub_ps(bval, _mm512_loadu_ps(b_ptr.add(d * 16))), sign_mask);
            acc_a = _mm512_add_ps(acc_a, da);
            acc_b = _mm512_add_ps(acc_b, db);
        }
        _mm512_storeu_ps(out.as_mut_ptr(), acc_a);
        _mm512_storeu_ps(out.as_mut_ptr().add(16), acc_b);
    }
}

// -- Tiled multi-batch distance -----------------------------------------------

/// Dimension tile size for cache-friendly multi-batch distance computation.
///
/// Each tile accesses `num_sub_batches × TILE × SIMD_BATCH_WIDTH × 4` bytes
/// of transposed query data. With 10 sub-batches and TILE=48:
/// 10 × 48 × 16 × 4 = 30 KB — fits comfortably in L1 (48 KB per core).
const DIM_TILE: usize = 64;

/// Compute neg-dot-product distances from one f32 base vector to ALL
/// transposed sub-batches simultaneously, using dimension tiling for L1
/// cache locality.
///
/// This replaces the per-pair `dual_batched_neg_dot_f32` calls with a
/// single pass that tiles across dimensions. Each base-vector element is
/// broadcast once and used by ALL sub-batches before advancing to the
/// next tile — amortizing the broadcast cost and keeping the working set
/// in L1.
///
/// `accumulators` must have length `sub_batches.len()` and each inner
/// slice must have length `SIMD_BATCH_WIDTH` (16), pre-zeroed by caller.
#[inline(never)]
pub fn tiled_neg_dot_all_batches_f32(
    base: &[f32],
    sub_batches: &[TransposedBatch],
    accumulators: &mut [&mut [f32; SIMD_BATCH_WIDTH]],
    dim: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                tiled_neg_dot_all_batches_f32_avx512(base, sub_batches, accumulators, dim);
            }
            return;
        }
    }
    // Scalar fallback
    for d in 0..dim {
        let bv = base[d];
        for (si, batch) in sub_batches.iter().enumerate() {
            let acc = &mut accumulators[si];
            for qi in 0..batch.count() {
                acc[qi] -= bv * batch.data[d * SIMD_BATCH_WIDTH + qi];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn tiled_neg_dot_all_batches_f32_avx512(
    base: &[f32],
    sub_batches: &[TransposedBatch],
    accumulators: &mut [&mut [f32; SIMD_BATCH_WIDTH]],
    dim: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        let base_ptr = base.as_ptr();
        let n = sub_batches.len();

        const MAX_BATCHES: usize = 20;
        debug_assert!(n <= MAX_BATCHES);
        let mut ptrs: [*const f32; MAX_BATCHES] = [std::ptr::null(); MAX_BATCHES];
        let mut accs: [__m512; MAX_BATCHES] = [_mm512_setzero_ps(); MAX_BATCHES];
        for i in 0..n {
            ptrs[i] = sub_batches[i].as_ptr();
        }

        // Process ALL sub-batches per dimension within each tile.
        // The inner dimension loop is the tightest loop — for each dimension,
        // broadcast once and FMA into all accumulators.
        //
        // Working set per tile: n × TILE × 16 × 4 bytes.
        // With n=10 and TILE=48: 30 KB — fits in L1 (48 KB).
        //
        // To avoid the variable-length inner loop penalty, use a macro
        // that unrolls the sub-batch FMA sequence at compile time.
        macro_rules! fma_unrolled {
            ($bval:expr, $off:expr, $idx:expr) => {
                accs[$idx] = _mm512_fmadd_ps(
                    $bval,
                    _mm512_loadu_ps(ptrs[$idx].add($off)),
                    accs[$idx],
                );
            };
        }

        macro_rules! tiled_loop {
            ($($idx:expr),+) => {{
                let mut d = 0usize;
                while d < dim {
                    let tile_end = (d + DIM_TILE).min(dim);
                    let mut dd = d;
                    while dd < tile_end {
                        let bval = _mm512_set1_ps(*base_ptr.add(dd));
                        let off = dd * SIMD_BATCH_WIDTH;
                        $( fma_unrolled!(bval, off, $idx); )+
                        dd += 1;
                    }
                    d = tile_end;
                }
            }};
        }

        // Dispatch to fully-unrolled variants for common sub-batch counts.
        // 128 threads / 10K queries → 78 queries/thread → ceil(78/16) = 5
        // 64 threads / 10K queries → 156 queries/thread → ceil(156/16) = 10
        match n {
            1 => tiled_loop!(0),
            2 => tiled_loop!(0, 1),
            3 => tiled_loop!(0, 1, 2),
            4 => tiled_loop!(0, 1, 2, 3),
            5 => tiled_loop!(0, 1, 2, 3, 4),
            6 => tiled_loop!(0, 1, 2, 3, 4, 5),
            7 => tiled_loop!(0, 1, 2, 3, 4, 5, 6),
            8 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7),
            9 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8),
            10 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
            11 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            12 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
            13 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            14 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
            15 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
            16 => tiled_loop!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _ => {
                // Fallback for >16 sub-batches: dynamic loop
                let mut d = 0usize;
                while d < dim {
                    let tile_end = (d + DIM_TILE).min(dim);
                    for dd in d..tile_end {
                        let bval = _mm512_set1_ps(*base_ptr.add(dd));
                        let off = dd * SIMD_BATCH_WIDTH;
                        for si in 0..n {
                            fma_unrolled!(bval, off, si);
                        }
                    }
                    d = tile_end;
                }
            }
        }

        // Store negated results
        let zero = _mm512_setzero_ps();
        for si in 0..n {
            _mm512_storeu_ps(
                accumulators[si].as_mut_ptr(),
                _mm512_sub_ps(zero, accs[si]),
            );
        }
    }
}

// -- Packed tiled distance (contiguous layout) --------------------------------

/// Compute neg-dot-product distances from one f32 base vector to ALL
/// sub-batches in a [`PackedBatches`] using the contiguous interleaved layout.
///
/// The packed layout places all sub-batches for each dimension in adjacent
/// cache lines, giving the hardware prefetcher a single sequential stream.
/// Combined with dimension tiling, the active working set fits in L1.
///
/// `out` must have length `n_batches * SIMD_BATCH_WIDTH`, pre-zeroed by caller.
#[inline(never)]
pub fn packed_neg_dot_f32(
    base: &[f32],
    packed: &PackedBatches,
    out: &mut [f32],
) {
    let dim = packed.dim;
    let n = packed.n_batches;
    let stride = packed.row_stride();
    debug_assert!(out.len() >= n * SIMD_BATCH_WIDTH);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                packed_neg_dot_f32_avx512(base, packed.as_ptr(), dim, n, stride, out);
            }
            return;
        }
    }
    // Scalar fallback
    for d in 0..dim {
        let bv = base[d];
        let row = d * stride;
        for si in 0..n {
            let off = row + si * SIMD_BATCH_WIDTH;
            for qi in 0..SIMD_BATCH_WIDTH {
                out[si * SIMD_BATCH_WIDTH + qi] -= bv * packed.data[off + qi];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn packed_neg_dot_f32_avx512(
    base: &[f32],
    packed_ptr: *const f32,
    dim: usize,
    n_batches: usize,
    stride: usize,
    out: &mut [f32],
) {
    unsafe {
        use std::arch::x86_64::*;

        let base_ptr = base.as_ptr();

        // Use a small fixed-size array of accumulators with the match
        // ensuring the compiler sees a constant size. Each match arm
        // uses a const N so the inner loop is fully unrolled.
        macro_rules! packed_kernel {
            ($n:expr) => {{
                let mut accs = [_mm512_setzero_ps(); $n];
                let mut dd = 0usize;
                while dd < dim {
                    let bval = _mm512_set1_ps(*base_ptr.add(dd));
                    let row_ptr = packed_ptr.add(dd * stride);
                    let mut si = 0;
                    while si < $n {
                        accs[si] = _mm512_fmadd_ps(
                            bval,
                            _mm512_loadu_ps(row_ptr.add(si * SIMD_BATCH_WIDTH)),
                            accs[si],
                        );
                        si += 1;
                    }
                    dd += 1;
                }
                let zero = _mm512_setzero_ps();
                let mut si = 0;
                while si < $n {
                    _mm512_storeu_ps(
                        out.as_mut_ptr().add(si * SIMD_BATCH_WIDTH),
                        _mm512_sub_ps(zero, accs[si]),
                    );
                    si += 1;
                }
            }};
        }

        match n_batches {
            1 => packed_kernel!(1),
            2 => packed_kernel!(2),
            3 => packed_kernel!(3),
            4 => packed_kernel!(4),
            5 => packed_kernel!(5),
            6 => packed_kernel!(6),
            7 => packed_kernel!(7),
            8 => packed_kernel!(8),
            9 => packed_kernel!(9),
            10 => packed_kernel!(10),
            _ => {
                // Fallback for >10: use array (may spill)
                const MAX_BATCHES: usize = 20;
                debug_assert!(n_batches <= MAX_BATCHES);
                let mut accs: [__m512; MAX_BATCHES] = [_mm512_setzero_ps(); MAX_BATCHES];
                let mut dd = 0usize;
                while dd < dim {
                    let bval = _mm512_set1_ps(*base_ptr.add(dd));
                    let row_ptr = packed_ptr.add(dd * stride);
                    for si in 0..n_batches {
                        accs[si] = _mm512_fmadd_ps(
                            bval,
                            _mm512_loadu_ps(row_ptr.add(si * SIMD_BATCH_WIDTH)),
                            accs[si],
                        );
                    }
                    dd += 1;
                }
                let zero = _mm512_setzero_ps();
                for si in 0..n_batches {
                    _mm512_storeu_ps(
                        out.as_mut_ptr().add(si * SIMD_BATCH_WIDTH),
                        _mm512_sub_ps(zero, accs[si]),
                    );
                }
            }
        }
    }
}

// -- Bulk f16 → f32 conversion ------------------------------------------------

/// Bulk-convert f16 values to f32 using AVX-512 when available.
///
/// For 512 dimensions, uses 32 SIMD conversion instructions instead of
/// 512 scalar conversions. This should be called once per base vector,
/// not inside the per-sub-batch distance kernel.
pub fn convert_f16_to_f32_bulk(src: &[half::f16], dst: &mut [f32]) {
    debug_assert!(dst.len() >= src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            unsafe { convert_f16_to_f32_avx512(src, dst); }
            return;
        }
        if is_x86_feature_detected!("f16c") {
            unsafe { convert_f16_to_f32_avx2(src, dst); }
            return;
        }
    }
    for (i, h) in src.iter().enumerate() {
        dst[i] = h.to_f32();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "f16c")]
unsafe fn convert_f16_to_f32_avx512(src: &[half::f16], dst: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let n = src.len();
        let src_ptr = src.as_ptr() as *const u16;
        let dst_ptr = dst.as_mut_ptr();
        let chunks = n / 16;
        let remainder = n % 16;

        for i in 0..chunks {
            let offset = i * 16;
            let f16_vals = _mm256_loadu_si256(src_ptr.add(offset) as *const __m256i);
            let f32_vals = _mm512_cvtph_ps(f16_vals);
            _mm512_storeu_ps(dst_ptr.add(offset), f32_vals);
        }

        let tail_start = chunks * 16;
        for i in 0..remainder {
            *dst_ptr.add(tail_start + i) =
                half::f16::from_bits(*src_ptr.add(tail_start + i)).to_f32();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn convert_f16_to_f32_avx2(src: &[half::f16], dst: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let n = src.len();
        let src_ptr = src.as_ptr() as *const u16;
        let dst_ptr = dst.as_mut_ptr();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let f16_vals = _mm_loadu_si128(src_ptr.add(offset) as *const __m128i);
            let f32_vals = _mm256_cvtph_ps(f16_vals);
            _mm256_storeu_ps(dst_ptr.add(offset), f32_vals);
        }

        let tail_start = chunks * 8;
        for i in 0..remainder {
            *dst_ptr.add(tail_start + i) =
                half::f16::from_bits(*src_ptr.add(tail_start + i)).to_f32();
        }
    }
}

// -- f64 distance functions ---------------------------------------------------

/// Select the best distance function for the given metric operating on f64 vectors.
///
/// L2, Cosine, and Dot use SimSIMD's native f64 SIMD kernels. L1 uses a scalar
/// fallback since SimSIMD does not provide L1.
pub fn select_distance_fn_f64(metric: Metric) -> fn(&[f64], &[f64]) -> f32 {
    match metric {
        Metric::L2 => select_l2_f64(),
        Metric::Cosine => select_cosine_f64(),
        Metric::DotProduct => select_dot_product_f64(),
        Metric::L1 => select_l1_f64(),
    }
}

fn select_l2_f64() -> fn(&[f64], &[f64]) -> f32 {
    |a, b| <f64 as SpatialSimilarity>::l2sq(a, b).unwrap_or(0.0) as f32
}

fn select_cosine_f64() -> fn(&[f64], &[f64]) -> f32 {
    |a, b| <f64 as SpatialSimilarity>::cos(a, b).unwrap_or(1.0) as f32
}

fn select_dot_product_f64() -> fn(&[f64], &[f64]) -> f32 {
    |a, b| -(<f64 as SpatialSimilarity>::dot(a, b).unwrap_or(0.0) as f32)
}

fn select_l1_f64() -> fn(&[f64], &[f64]) -> f32 {
    l1_f64_scalar
}

fn l1_f64_scalar(a: &[f64], b: &[f64]) -> f32 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum as f32
}

#[cfg(test)]
fn l2_f64_scalar(a: &[f64], b: &[f64]) -> f32 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum as f32
}

#[cfg(test)]
fn cosine_f64_scalar(a: &[f64], b: &[f64]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 1.0 } else { (1.0 - dot / denom) as f32 }
}

#[cfg(test)]
fn dot_product_f64_scalar(a: &[f64], b: &[f64]) -> f32 {
    let mut dot = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    -dot as f32
}

// -- i8 distance functions ----------------------------------------------------

/// Select the best distance function for the given metric operating on i8 vectors.
///
/// L2, Cosine, and Dot use SimSIMD's native i8 SIMD kernels. L1 uses a scalar
/// fallback since SimSIMD does not provide L1.
pub fn select_distance_fn_i8(metric: Metric) -> fn(&[i8], &[i8]) -> f32 {
    match metric {
        Metric::L2 => select_l2_i8(),
        Metric::Cosine => select_cosine_i8(),
        Metric::DotProduct => select_dot_product_i8(),
        Metric::L1 => select_l1_i8(),
    }
}

fn select_l2_i8() -> fn(&[i8], &[i8]) -> f32 {
    |a, b| <i8 as SpatialSimilarity>::l2sq(a, b).unwrap_or(0.0) as f32
}

fn select_cosine_i8() -> fn(&[i8], &[i8]) -> f32 {
    |a, b| <i8 as SpatialSimilarity>::cos(a, b).unwrap_or(1.0) as f32
}

fn select_dot_product_i8() -> fn(&[i8], &[i8]) -> f32 {
    |a, b| -(<i8 as SpatialSimilarity>::dot(a, b).unwrap_or(0.0) as f32)
}

fn select_l1_i8() -> fn(&[i8], &[i8]) -> f32 {
    l1_i8_scalar
}

fn l1_i8_scalar(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i64;
    for i in 0..a.len() {
        sum += ((a[i] as i32) - (b[i] as i32)).abs() as i64;
    }
    sum as f32
}

#[cfg(test)]
fn l2_i8_scalar(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i64;
    for i in 0..a.len() {
        let d = (a[i] as i32) - (b[i] as i32);
        sum += (d * d) as i64;
    }
    sum as f32
}

#[cfg(test)]
fn cosine_i8_scalar(a: &[i8], b: &[i8]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 1.0 } else { (1.0 - dot / denom) as f32 }
}

#[cfg(test)]
fn dot_product_i8_scalar(a: &[i8], b: &[i8]) -> f32 {
    let mut dot = 0i64;
    for i in 0..a.len() {
        dot += (a[i] as i64) * (b[i] as i64);
    }
    -(dot as f32)
}

// -- Reporting ---------------------------------------------------------------

/// Report which SIMD level is available on this system.
///
/// Returns "SimSIMD" for L2/Cosine/Dot (SimSIMD handles its own dispatch),
/// plus the L1 SIMD level in parentheses.
pub fn simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "SimSIMD (L1: AVX-512)";
        }
        if is_x86_feature_detected!("avx2") {
            return "SimSIMD (L1: AVX2)";
        }
        return "SimSIMD (L1: scalar)";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "SimSIMD (L1: scalar)";
    }
    #[allow(unreachable_code)]
    "SimSIMD (L1: scalar)"
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
fn l2_f16_scalar(a: &[half::f16], b: &[half::f16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i].to_f32() - b[i].to_f32();
        sum += d * d;
    }
    sum // squared L2 — matches select_distance_fn_f16(Metric::L2)
}

#[cfg(test)]
fn cosine_f16_scalar(a: &[half::f16], b: &[half::f16]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        let ai = a[i].to_f32();
        let bi = b[i].to_f32();
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

#[cfg(test)]
fn dot_product_f16_scalar(a: &[half::f16], b: &[half::f16]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i].to_f32() * b[i].to_f32();
    }
    -dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_scalar_identity() {
        let a = vec![1.0, 2.0, 3.0];
        let d = l2_scalar(&a, &a);
        assert!(d.abs() < 1e-6, "expected 0, got {}", d);
    }

    #[test]
    fn test_l2_scalar_known() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = l2_scalar(&a, &b);
        assert!(
            (d - 2.0).abs() < 1e-5, // squared L2: (1-0)² + (0-1)² = 2
            "expected 2.0 (l2sq), got {}",
            d
        );
    }

    #[test]
    fn test_cosine_scalar_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = cosine_scalar(&a, &b);
        assert!((d - 1.0).abs() < 1e-5, "expected 1.0, got {}", d);
    }

    #[test]
    fn test_cosine_scalar_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = cosine_scalar(&a, &a);
        assert!(d.abs() < 1e-5, "expected 0, got {}", d);
    }

    #[test]
    fn test_dot_product_scalar() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let d = dot_product_scalar(&a, &b);
        assert!((d - (-1.0)).abs() < 1e-5, "expected -1.0, got {}", d);
    }

    #[test]
    fn test_l1_scalar() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = l1_scalar(&a, &b);
        assert!((d - 9.0).abs() < 1e-5, "expected 9.0, got {}", d);
    }

    #[test]
    fn test_select_returns_working_fn() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
            let f = select_distance_fn(metric);
            let d = f(&a, &b);
            assert!(d.is_finite(), "metric {:?} returned non-finite: {}", metric, d);
        }
    }

    #[test]
    fn test_simd_matches_scalar_l2() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32) * 0.2 - 5.0).collect();

        let scalar = l2_scalar(&a, &b);
        let simd = select_distance_fn(Metric::L2)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-2,
            "L2 mismatch: scalar={}, simd={}",
            scalar,
            simd
        );
    }

    #[test]
    fn test_simd_matches_scalar_cosine() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32) * 0.2 - 5.0).collect();

        let scalar = cosine_scalar(&a, &b);
        let simd = select_distance_fn(Metric::Cosine)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-3,
            "Cosine mismatch: scalar={}, simd={}",
            scalar,
            simd
        );
    }

    #[test]
    fn test_simd_matches_scalar_dot() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32) * 0.2 - 5.0).collect();

        let scalar = dot_product_scalar(&a, &b);
        let simd = select_distance_fn(Metric::DotProduct)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-1,
            "Dot mismatch: scalar={}, simd={}",
            scalar,
            simd
        );
    }

    #[test]
    fn test_simd_matches_scalar_l1() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32) * 0.2 - 5.0).collect();

        let scalar = l1_scalar(&a, &b);
        let simd = select_distance_fn(Metric::L1)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-2,
            "L1 mismatch: scalar={}, simd={}",
            scalar,
            simd
        );
    }

    #[test]
    fn test_odd_length_vectors() {
        // Test with lengths that aren't multiples of 8 or 16
        for len in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 2.0).collect();

            for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
                let f = select_distance_fn(metric);
                let d = f(&a, &b);
                assert!(d.is_finite(), "metric {:?} len={} non-finite", metric, len);
            }
        }
    }

    #[test]
    fn test_simd_level_reports_something() {
        let level = simd_level();
        assert!(!level.is_empty());
        assert!(level.contains("SimSIMD"), "expected SimSIMD in level: {}", level);
    }

    // -- f16 tests ------------------------------------------------------------

    fn make_f16_vecs(len: usize) -> (Vec<half::f16>, Vec<half::f16>) {
        let a: Vec<half::f16> = (0..len).map(|i| half::f16::from_f32((i as f32) * 0.1)).collect();
        let b: Vec<half::f16> = (0..len).map(|i| half::f16::from_f32((i as f32) * 0.2 - 5.0)).collect();
        (a, b)
    }

    #[test]
    fn test_f16_select_returns_working_fn() {
        let (a, b) = make_f16_vecs(8);
        for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
            let f = select_distance_fn_f16(metric);
            let d = f(&a, &b);
            assert!(d.is_finite(), "f16 metric {:?} returned non-finite: {}", metric, d);
        }
    }

    #[test]
    fn test_f16_simd_matches_scalar_l2() {
        let (a, b) = make_f16_vecs(128);
        let scalar = l2_f16_scalar(&a, &b);
        let simd = select_distance_fn_f16(Metric::L2)(&a, &b);
        assert!(
            (scalar - simd).abs() < 0.5,
            "f16 L2 mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f16_simd_matches_scalar_cosine() {
        let (a, b) = make_f16_vecs(128);
        let scalar = cosine_f16_scalar(&a, &b);
        let simd = select_distance_fn_f16(Metric::Cosine)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-2,
            "f16 Cosine mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f16_simd_matches_scalar_dot() {
        let (a, b) = make_f16_vecs(128);
        let scalar = dot_product_f16_scalar(&a, &b);
        let simd = select_distance_fn_f16(Metric::DotProduct)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1.0,
            "f16 Dot mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f16_simd_matches_scalar_l1() {
        let (a, b) = make_f16_vecs(128);
        let scalar = l1_f16_scalar(&a, &b);
        let simd = select_distance_fn_f16(Metric::L1)(&a, &b);
        assert!(
            (scalar - simd).abs() < 0.5,
            "f16 L1 mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f16_odd_length_vectors() {
        for len in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65] {
            let a: Vec<half::f16> = (0..len).map(|i| half::f16::from_f32(i as f32)).collect();
            let b: Vec<half::f16> = (0..len).map(|i| half::f16::from_f32((i as f32) * 2.0)).collect();

            for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
                let f = select_distance_fn_f16(metric);
                let d = f(&a, &b);
                assert!(d.is_finite(), "f16 metric {:?} len={} non-finite", metric, len);
            }
        }
    }

    // -- Transposed batch tests -----------------------------------------------

    #[test]
    fn test_transposed_batch_l2sq_f16_matches_pairwise() {
        let dim = 128;
        let queries: Vec<Vec<half::f16>> = (0..16)
            .map(|qi| {
                (0..dim)
                    .map(|d| half::f16::from_f32((qi * dim + d) as f32 * 0.01))
                    .collect()
            })
            .collect();
        let base: Vec<half::f16> = (0..dim)
            .map(|d| half::f16::from_f32(d as f32 * 0.05 - 3.0))
            .collect();

        let query_refs: Vec<&[half::f16]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = TransposedBatch::from_f16(&query_refs, dim);

        let mut batched_out = vec![0.0f32; SIMD_BATCH_WIDTH];
        batched_l2sq_f16(&batch, &base, &mut batched_out);

        let pairwise_fn = select_distance_fn_f16(Metric::L2);
        for qi in 0..16 {
            let pairwise = pairwise_fn(&queries[qi], &base);
            assert!(
                (batched_out[qi] - pairwise).abs() < 0.5,
                "L2sq f16 mismatch at qi={}: batched={}, pairwise={}",
                qi, batched_out[qi], pairwise
            );
        }
    }

    #[test]
    fn test_transposed_batch_l2sq_f32_matches_pairwise() {
        let dim = 128;
        let queries: Vec<Vec<f32>> = (0..16)
            .map(|qi| {
                (0..dim)
                    .map(|d| (qi * dim + d) as f32 * 0.01)
                    .collect()
            })
            .collect();
        let base: Vec<f32> = (0..dim).map(|d| d as f32 * 0.05 - 3.0).collect();

        let query_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = TransposedBatch::from_f32(&query_refs, dim);

        let mut batched_out = vec![0.0f32; SIMD_BATCH_WIDTH];
        batched_l2sq_f32(&batch, &base, &mut batched_out);

        let pairwise_fn = select_distance_fn(Metric::L2);
        for qi in 0..16 {
            let pairwise = pairwise_fn(&queries[qi], &base);
            // Relative tolerance: SimSIMD uses different accumulation order
            let tol = pairwise.abs() * 1e-4 + 0.1;
            assert!(
                (batched_out[qi] - pairwise).abs() < tol,
                "L2sq f32 mismatch at qi={}: batched={}, pairwise={}",
                qi, batched_out[qi], pairwise
            );
        }
    }

    #[test]
    fn test_transposed_batch_partial_fill() {
        let dim = 64;
        let queries: Vec<Vec<f32>> = (0..5)
            .map(|qi| (0..dim).map(|d| (qi * dim + d) as f32 * 0.1).collect())
            .collect();
        let base: Vec<f32> = (0..dim).map(|d| d as f32 * 0.2).collect();

        let query_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = TransposedBatch::from_f32(&query_refs, dim);
        assert_eq!(batch.count(), 5);

        let mut batched_out = vec![0.0f32; SIMD_BATCH_WIDTH];
        batched_l2sq_f32(&batch, &base, &mut batched_out);

        let pairwise_fn = select_distance_fn(Metric::L2);
        for qi in 0..5 {
            let pairwise = pairwise_fn(&queries[qi], &base);
            assert!(
                (batched_out[qi] - pairwise).abs() < 1e-2,
                "partial L2sq f32 mismatch at qi={}: batched={}, pairwise={}",
                qi, batched_out[qi], pairwise
            );
        }
    }

    #[test]
    fn test_transposed_batch_neg_dot_f16() {
        let dim = 64;
        let queries: Vec<Vec<half::f16>> = (0..8)
            .map(|qi| {
                (0..dim)
                    .map(|d| half::f16::from_f32((qi * dim + d) as f32 * 0.01))
                    .collect()
            })
            .collect();
        let base: Vec<half::f16> = (0..dim)
            .map(|d| half::f16::from_f32(d as f32 * 0.05))
            .collect();

        let query_refs: Vec<&[half::f16]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = TransposedBatch::from_f16(&query_refs, dim);

        let mut batched_out = vec![0.0f32; SIMD_BATCH_WIDTH];
        batched_neg_dot_f16(&batch, &base, &mut batched_out);

        let pairwise_fn = select_distance_fn_f16(Metric::DotProduct);
        for qi in 0..8 {
            let pairwise = pairwise_fn(&queries[qi], &base);
            assert!(
                (batched_out[qi] - pairwise).abs() < 1.0,
                "neg dot f16 mismatch at qi={}: batched={}, pairwise={}",
                qi, batched_out[qi], pairwise
            );
        }
    }

    #[test]
    fn test_transposed_batch_l1_f16() {
        let dim = 64;
        let queries: Vec<Vec<half::f16>> = (0..8)
            .map(|qi| {
                (0..dim)
                    .map(|d| half::f16::from_f32((qi * dim + d) as f32 * 0.01))
                    .collect()
            })
            .collect();
        let base: Vec<half::f16> = (0..dim)
            .map(|d| half::f16::from_f32(d as f32 * 0.05 - 1.0))
            .collect();

        let query_refs: Vec<&[half::f16]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = TransposedBatch::from_f16(&query_refs, dim);

        let mut batched_out = vec![0.0f32; SIMD_BATCH_WIDTH];
        batched_l1_f16(&batch, &base, &mut batched_out);

        let pairwise_fn = select_distance_fn_f16(Metric::L1);
        for qi in 0..8 {
            let pairwise = pairwise_fn(&queries[qi], &base);
            assert!(
                (batched_out[qi] - pairwise).abs() < 1.0,
                "L1 f16 mismatch at qi={}: batched={}, pairwise={}",
                qi, batched_out[qi], pairwise
            );
        }
    }

    // -- f64 tests ------------------------------------------------------------

    fn make_f64_vecs(len: usize) -> (Vec<f64>, Vec<f64>) {
        let a: Vec<f64> = (0..len).map(|i| (i as f64) * 0.1).collect();
        let b: Vec<f64> = (0..len).map(|i| (i as f64) * 0.2 - 5.0).collect();
        (a, b)
    }

    #[test]
    fn test_f64_select_returns_working_fn() {
        let (a, b) = make_f64_vecs(8);
        for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
            let f = select_distance_fn_f64(metric);
            let d = f(&a, &b);
            assert!(d.is_finite(), "f64 metric {:?} returned non-finite: {}", metric, d);
        }
    }

    #[test]
    fn test_f64_simd_matches_scalar_l2() {
        let (a, b) = make_f64_vecs(128);
        let scalar = l2_f64_scalar(&a, &b);
        let simd = select_distance_fn_f64(Metric::L2)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-2,
            "f64 L2 mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f64_simd_matches_scalar_cosine() {
        let (a, b) = make_f64_vecs(128);
        let scalar = cosine_f64_scalar(&a, &b);
        let simd = select_distance_fn_f64(Metric::Cosine)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-3,
            "f64 Cosine mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f64_simd_matches_scalar_dot() {
        let (a, b) = make_f64_vecs(128);
        let scalar = dot_product_f64_scalar(&a, &b);
        let simd = select_distance_fn_f64(Metric::DotProduct)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-1,
            "f64 Dot mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f64_simd_matches_scalar_l1() {
        let (a, b) = make_f64_vecs(128);
        let scalar = l1_f64_scalar(&a, &b);
        let simd = select_distance_fn_f64(Metric::L1)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-2,
            "f64 L1 mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_f64_odd_length_vectors() {
        for len in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65] {
            let a: Vec<f64> = (0..len).map(|i| i as f64).collect();
            let b: Vec<f64> = (0..len).map(|i| (i as f64) * 2.0).collect();

            for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
                let f = select_distance_fn_f64(metric);
                let d = f(&a, &b);
                assert!(d.is_finite(), "f64 metric {:?} len={} non-finite", metric, len);
            }
        }
    }

    // -- i8 tests -------------------------------------------------------------

    fn make_i8_vecs(len: usize) -> (Vec<i8>, Vec<i8>) {
        let a: Vec<i8> = (0..len).map(|i| ((i % 127) as i8).wrapping_sub(64)).collect();
        let b: Vec<i8> = (0..len).map(|i| ((i * 3 % 127) as i8).wrapping_sub(32)).collect();
        (a, b)
    }

    #[test]
    fn test_i8_select_returns_working_fn() {
        let (a, b) = make_i8_vecs(8);
        for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
            let f = select_distance_fn_i8(metric);
            let d = f(&a, &b);
            assert!(d.is_finite(), "i8 metric {:?} returned non-finite: {}", metric, d);
        }
    }

    #[test]
    fn test_i8_simd_matches_scalar_l2() {
        let (a, b) = make_i8_vecs(128);
        let scalar = l2_i8_scalar(&a, &b);
        let simd = select_distance_fn_i8(Metric::L2)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1.0,
            "i8 L2 mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_i8_simd_matches_scalar_cosine() {
        let (a, b) = make_i8_vecs(128);
        let scalar = cosine_i8_scalar(&a, &b);
        let simd = select_distance_fn_i8(Metric::Cosine)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-2,
            "i8 Cosine mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_i8_simd_matches_scalar_dot() {
        let (a, b) = make_i8_vecs(128);
        let scalar = dot_product_i8_scalar(&a, &b);
        let simd = select_distance_fn_i8(Metric::DotProduct)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1.0,
            "i8 Dot mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_i8_simd_matches_scalar_l1() {
        let (a, b) = make_i8_vecs(128);
        let scalar = l1_i8_scalar(&a, &b);
        let simd = select_distance_fn_i8(Metric::L1)(&a, &b);
        assert!(
            (scalar - simd).abs() < 1.0,
            "i8 L1 mismatch: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    fn test_i8_odd_length_vectors() {
        for len in [1, 3, 7, 9, 15, 17, 31, 33, 63, 65] {
            let a: Vec<i8> = (0..len).map(|i| ((i % 127) as i8).wrapping_sub(64)).collect();
            let b: Vec<i8> = (0..len).map(|i| ((i * 3 % 127) as i8).wrapping_sub(32)).collect();

            for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
                let f = select_distance_fn_i8(metric);
                let d = f(&a, &b);
                assert!(d.is_finite(), "i8 metric {:?} len={} non-finite", metric, len);
            }
        }
    }
}
