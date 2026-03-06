// Copyright (c) DataStax, Inc.
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
        for (qi, q) in queries.iter().enumerate() {
            for d in 0..dim {
                data[d * SIMD_BATCH_WIDTH + qi] = q[d].to_f32();
            }
        }
        Self { data, dim, count }
    }

    /// Create a transposed batch from f32 query slices.
    pub fn from_f32(queries: &[&[f32]], dim: usize) -> Self {
        assert!(queries.len() <= SIMD_BATCH_WIDTH);
        let count = queries.len();
        let mut data = vec![0.0f32; dim * SIMD_BATCH_WIDTH];
        for (qi, q) in queries.iter().enumerate() {
            for d in 0..dim {
                data[d * SIMD_BATCH_WIDTH + qi] = q[d];
            }
        }
        Self { data, dim, count }
    }

    /// Number of actual queries in this batch (≤ SIMD_BATCH_WIDTH).
    pub fn count(&self) -> usize {
        self.count
    }
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
        Metric::L1 => Some(batched_l1_f16),
        _ => None,
    }
}

/// Select a batched SIMD distance function for f32 vectors, if available.
pub fn select_batched_fn_f32(metric: Metric) -> Option<BatchedDistFnF32> {
    match metric {
        Metric::L2 => Some(batched_l2sq_f32),
        Metric::DotProduct => Some(batched_neg_dot_f32),
        Metric::L1 => Some(batched_l1_f32),
        _ => None,
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
}
