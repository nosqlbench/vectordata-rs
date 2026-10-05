// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Query-batch layouts and the kernels that score one base vector
//! against a whole batch.
//!
//! A batch holds [`BATCH_WIDTH`] (16) queries in dimension-major order,
//! so for each base dimension the kernel broadcasts `base[d]` once and
//! updates all 16 query lanes with one vector operation. Sixteen lanes
//! is one register on AVX-512, two on AVX2 and four on NEON or SSE; the
//! batch width is a property of the layout, not of the instruction set.
//!
//! ## Lane-order invariant
//!
//! The single-batch, dual-batch and packed kernels compute each query
//! lane with the same sequence of operations: one fused multiply-add per
//! dimension, in dimension order, from a zero start, finished by the
//! same per-metric epilogue. A query's distance to a base vector is
//! therefore bit-identical whichever kernel scored it, so how a caller
//! groups queries into batches cannot change its results.

use fearless_simd::{Simd, f32x16, prelude::*};
use fearless_simd_macros::simd;

/// Queries per transposed batch.
pub const BATCH_WIDTH: usize = 16;

/// Up to [`BATCH_WIDTH`] queries in dimension-major (columnar) layout,
/// pre-converted to f32.
///
/// Row `d` holds dimension `d` of every query: `data[d * 16 + qi]`.
/// Lanes past [`count`](Self::count) are zero, with a query norm of 1.0,
/// so the kernels can run all 16 lanes unconditionally; their outputs
/// are meaningless and must not be read.
pub struct TransposedBatch {
    data: Vec<f32>,
    query_norms: [f32; BATCH_WIDTH],
    dim: usize,
    count: usize,
}

impl TransposedBatch {
    /// Transpose f32 queries. Panics if more than [`BATCH_WIDTH`] are
    /// given or a query is shorter than `dim`.
    pub fn from_f32(queries: &[&[f32]], dim: usize) -> Self {
        Self::build(queries.len(), dim, |qi, d| queries[qi][d])
    }

    /// Transpose f16 queries, widening each value to f32 once here so the
    /// kernels never convert on the hot path.
    pub fn from_f16(queries: &[&[half::f16]], dim: usize) -> Self {
        Self::build(queries.len(), dim, |qi, d| queries[qi][d].to_f32())
    }

    fn build(count: usize, dim: usize, value: impl Fn(usize, usize) -> f32) -> Self {
        assert!(count <= BATCH_WIDTH, "a transposed batch holds at most {BATCH_WIDTH} queries, got {count}");
        let mut data = vec![0.0f32; dim * BATCH_WIDTH];
        let mut query_norms = [1.0f32; BATCH_WIDTH];
        for (qi, norm) in query_norms.iter_mut().enumerate().take(count) {
            let mut norm_sq = 0.0f32;
            for d in 0..dim {
                let v = value(qi, d);
                data[d * BATCH_WIDTH + qi] = v;
                norm_sq += v * v;
            }
            *norm = norm_sq.sqrt().max(f32::EPSILON);
        }
        Self { data, query_norms, dim, count }
    }

    /// Number of real queries in the batch.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Dimensionality of the queries.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The L2 norm of each query lane (1.0 for unused lanes), floored at
    /// `f32::EPSILON`.
    pub fn query_norms(&self) -> &[f32; BATCH_WIDTH] {
        &self.query_norms
    }

    /// The transposed data as rows of 16 lanes, one row per dimension.
    pub fn rows(&self) -> &[[f32; BATCH_WIDTH]] {
        self.data.as_chunks::<BATCH_WIDTH>().0
    }
}

/// Every 16-query sub-batch of a query set, interleaved per dimension.
///
/// Layout: `data[d * row_stride + si * 16 + qi]`, where `row_stride` is
/// `n_batches * 16`. All sub-batches' values for one dimension sit in
/// adjacent cache lines, so a scan over the base reads the batch data as
/// one sequential stream rather than one stream per sub-batch.
pub struct PackedBatches {
    data: Vec<f32>,
    dim: usize,
    n_batches: usize,
    counts: Vec<usize>,
    offsets: Vec<usize>,
}

impl PackedBatches {
    /// Pack f32 queries.
    pub fn from_f32(queries: &[&[f32]], dim: usize) -> Self {
        Self::build(queries.len(), dim, |qi, d| queries[qi][d])
    }

    /// Pack f16 queries, widened to f32.
    pub fn from_f16(queries: &[&[half::f16]], dim: usize) -> Self {
        Self::build(queries.len(), dim, |qi, d| queries[qi][d].to_f32())
    }

    fn build(n_queries: usize, dim: usize, value: impl Fn(usize, usize) -> f32) -> Self {
        let n_batches = n_queries.div_ceil(BATCH_WIDTH);
        let stride = n_batches * BATCH_WIDTH;
        let mut data = vec![0.0f32; dim * stride];
        let mut counts = Vec::with_capacity(n_batches);
        let mut offsets = Vec::with_capacity(n_batches);
        for si in 0..n_batches {
            let first = si * BATCH_WIDTH;
            let count = (n_queries - first).min(BATCH_WIDTH);
            counts.push(count);
            offsets.push(first);
            for qi in 0..count {
                for d in 0..dim {
                    data[d * stride + si * BATCH_WIDTH + qi] = value(first + qi, d);
                }
            }
        }
        Self { data, dim, n_batches, counts, offsets }
    }

    /// Number of sub-batches.
    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    /// Number of real queries in sub-batch `si`.
    pub fn count(&self, si: usize) -> usize {
        self.counts[si]
    }

    /// Index of sub-batch `si`'s first query in the original query list.
    pub fn offset(&self, si: usize) -> usize {
        self.offsets[si]
    }

    /// Dimensionality of the queries.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Elements per dimension row (`n_batches * 16`).
    pub fn row_stride(&self) -> usize {
        self.n_batches * BATCH_WIDTH
    }

    /// The packed data as 16-lane chunks: chunk `d * n_batches + si` is
    /// dimension `d` of sub-batch `si`.
    fn chunks(&self) -> &[[f32; BATCH_WIDTH]] {
        self.data.as_chunks::<BATCH_WIDTH>().0
    }
}

// ─── Per-metric lane updates and epilogues ──────────────────────────────
//
// Shared by the single and dual kernels so both apply exactly the same
// operations to a lane (the invariant in the module docs).

#[inline(always)]
fn step_l2sq<S: Simd>(acc: f32x16<S>, b: f32x16<S>, q: f32x16<S>) -> f32x16<S> {
    let diff = b - q;
    diff.mul_add(diff, acc)
}

#[inline(always)]
fn step_dot<S: Simd>(acc: f32x16<S>, b: f32x16<S>, q: f32x16<S>) -> f32x16<S> {
    b.mul_add(q, acc)
}

#[inline(always)]
fn step_l1<S: Simd>(acc: f32x16<S>, b: f32x16<S>, q: f32x16<S>) -> f32x16<S> {
    acc + (b - q).abs()
}

#[inline(always)]
fn negate<S: Simd>(simd: S, acc: f32x16<S>) -> f32x16<S> {
    // `0 - acc`, not `-acc`: an exact-zero dot product must come out as
    // +0.0, as it always has, so stored distances do not change sign.
    f32x16::splat(simd, 0.0) - acc
}

#[inline(always)]
fn cosine_epilogue<S: Simd>(simd: S, dot: f32x16<S>, base_norm: f32, query_norms: &[f32; BATCH_WIDTH]) -> f32x16<S> {
    let denom = f32x16::load_array_ref(simd, query_norms) * base_norm;
    f32x16::splat(simd, 1.0) - dot / denom
}

/// The base vector's L2 norm, floored at `f32::EPSILON`.
#[inline(always)]
fn base_norm<S: Simd>(simd: S, base: &[f32]) -> f32 {
    crate::pairwise::dot_f32(simd, base, base).sqrt().max(f32::EPSILON)
}

macro_rules! single_kernel {
    ($name:ident, $step:ident, $epilogue:expr) => {
        #[simd]
        pub(crate) fn $name<S: Simd>(simd: S, batch: &TransposedBatch, base: &[f32], out: &mut [f32]) {
            let mut acc = f32x16::splat(simd, 0.0);
            for (row, &bv) in batch.rows().iter().zip(base) {
                acc = $step(acc, f32x16::splat(simd, bv), f32x16::load_array_ref(simd, row));
            }
            let result: f32x16<S> = $epilogue(simd, acc, batch, base);
            result.store_slice(&mut out[..BATCH_WIDTH]);
        }
    };
}

macro_rules! dual_kernel {
    ($name:ident, $step:ident, $epilogue:expr) => {
        #[simd]
        pub(crate) fn $name<S: Simd>(
            simd: S,
            a: &TransposedBatch,
            b: &TransposedBatch,
            base: &[f32],
            out: &mut [f32; 2 * BATCH_WIDTH],
        ) {
            let zero = f32x16::splat(simd, 0.0);
            let (mut acc_a, mut acc_b) = (zero, zero);
            for ((ra, rb), &bv) in a.rows().iter().zip(b.rows()).zip(base) {
                let bc = f32x16::splat(simd, bv);
                acc_a = $step(acc_a, bc, f32x16::load_array_ref(simd, ra));
                acc_b = $step(acc_b, bc, f32x16::load_array_ref(simd, rb));
            }
            let (lo, hi) = out.split_at_mut(BATCH_WIDTH);
            let ra: f32x16<S> = $epilogue(simd, acc_a, a, base);
            let rb: f32x16<S> = $epilogue(simd, acc_b, b, base);
            ra.store_slice(lo);
            rb.store_slice(hi);
        }
    };
}

#[inline(always)]
fn ep_identity<S: Simd>(_: S, acc: f32x16<S>, _: &TransposedBatch, _: &[f32]) -> f32x16<S> {
    acc
}

#[inline(always)]
fn ep_negate<S: Simd>(simd: S, acc: f32x16<S>, _: &TransposedBatch, _: &[f32]) -> f32x16<S> {
    negate(simd, acc)
}

#[inline(always)]
fn ep_cosine<S: Simd>(simd: S, acc: f32x16<S>, batch: &TransposedBatch, base: &[f32]) -> f32x16<S> {
    cosine_epilogue(simd, acc, base_norm(simd, base), batch.query_norms())
}

single_kernel!(batch_l2sq, step_l2sq, ep_identity);
single_kernel!(batch_neg_dot, step_dot, ep_negate);
single_kernel!(batch_cosine, step_dot, ep_cosine);
single_kernel!(batch_l1, step_l1, ep_identity);

dual_kernel!(dual_l2sq, step_l2sq, ep_identity);
dual_kernel!(dual_neg_dot, step_dot, ep_negate);
dual_kernel!(dual_cosine, step_dot, ep_cosine);
dual_kernel!(dual_l1, step_l1, ep_identity);

// ─── Packed negated dot product ─────────────────────────────────────────

/// Score one base vector against every sub-batch of `packed` with the
/// negated dot product. `out[si * 16 + qi]` receives sub-batch `si`,
/// lane `qi`; `out` must hold at least `n_batches * 16` values.
///
/// Sub-batches are processed in groups of `group` whose accumulators
/// stay in registers for the whole pass over the base vector: the base
/// vector is re-read once per group, from L1, rather than spilling
/// accumulators on every dimension. Each level's table passes the group
/// size that fits its register file.
#[simd]
pub(crate) fn packed_neg_dot<S: Simd>(simd: S, base: &[f32], packed: &PackedBatches, out: &mut [f32], group: usize) {
    let n = packed.n_batches();
    assert!(out.len() >= n * BATCH_WIDTH, "packed output holds {} values, needs {}", out.len(), n * BATCH_WIDTH);
    let mut first = 0;
    while first < n {
        let g = (n - first).min(group);
        // Each arm fixes the group size at compile time so the
        // accumulator array lives in registers.
        match g {
            1 => packed_group::<S, 1>(simd, base, packed, first, out),
            2 => packed_group::<S, 2>(simd, base, packed, first, out),
            3 => packed_group::<S, 3>(simd, base, packed, first, out),
            4 => packed_group::<S, 4>(simd, base, packed, first, out),
            5 => packed_group::<S, 5>(simd, base, packed, first, out),
            6 => packed_group::<S, 6>(simd, base, packed, first, out),
            7 => packed_group::<S, 7>(simd, base, packed, first, out),
            8 => packed_group::<S, 8>(simd, base, packed, first, out),
            9 => packed_group::<S, 9>(simd, base, packed, first, out),
            10 => packed_group::<S, 10>(simd, base, packed, first, out),
            11 => packed_group::<S, 11>(simd, base, packed, first, out),
            12 => packed_group::<S, 12>(simd, base, packed, first, out),
            13 => packed_group::<S, 13>(simd, base, packed, first, out),
            14 => packed_group::<S, 14>(simd, base, packed, first, out),
            15 => packed_group::<S, 15>(simd, base, packed, first, out),
            _ => packed_group::<S, 16>(simd, base, packed, first, out),
        }
        first += g;
    }
}

#[inline(always)]
fn packed_group<S: Simd, const N: usize>(
    simd: S,
    base: &[f32],
    packed: &PackedBatches,
    first: usize,
    out: &mut [f32],
) {
    let n = packed.n_batches();
    // Checked once here so the per-dimension group slice below needs no
    // bounds check: every row is exactly `n` chunks long.
    assert!(first + N <= n, "packed group {first}..{} exceeds {n} sub-batches", first + N);
    let mut acc = [f32x16::splat(simd, 0.0); N];
    for (row, &bv) in packed.chunks().chunks_exact(n).zip(base) {
        let bc = f32x16::splat(simd, bv);
        for (slot, q) in acc.iter_mut().zip(&row[first..first + N]) {
            *slot = step_dot(*slot, bc, f32x16::load_array_ref(simd, q));
        }
    }
    for (s, slot) in acc.iter().enumerate() {
        let at = (first + s) * BATCH_WIDTH;
        negate(simd, *slot).store_slice(&mut out[at..at + BATCH_WIDTH]);
    }
}
