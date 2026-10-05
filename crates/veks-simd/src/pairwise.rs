// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pairwise distance kernels, generic over the dispatch level.
//!
//! Every reduction keeps four independent accumulators at the level's
//! native vector width, so a kernel is bound by load and FMA throughput
//! rather than by the latency of one dependent chain. The accumulators
//! are folded pairwise (`(a0 + a1) + (a2 + a3)`), then the remaining
//! whole vectors are added into the fold, then the scalar tail.
//!
//! Operands are taken to be the same length; a longer operand's excess
//! is ignored.

use fearless_simd::{Simd, f32x16, prelude::*};
use fearless_simd_macros::simd;

use crate::f16_lanes::F16Lanes;

const UNROLL: usize = 4;

/// Shared loop skeleton for one-accumulator-per-lane reductions over
/// native-width vectors.
///
/// `$vt` is the native vector type (`S::f32s` / `S::f64s`), `$step` the
/// per-chunk update `|acc, va, vb| -> acc`, `$scalar` the per-element
/// tail update `|sum, a, b| -> sum`.
macro_rules! reduce_native {
    ($simd:ident, $vt:ty, $elem:ty, $a:ident, $b:ident, $step:expr, $scalar:expr) => {{
        let n = $a.len().min($b.len());
        let (a, b) = (&$a[..n], &$b[..n]);
        let lanes = <$vt>::LEN;
        let block = lanes * UNROLL;
        let zero = <$vt>::splat($simd, 0.0);
        let mut acc = [zero; UNROLL];
        let blocks = n / block;
        for c in 0..blocks {
            let base = c * block;
            for (u, slot) in acc.iter_mut().enumerate() {
                let o = base + u * lanes;
                let va = <$vt>::from_slice($simd, &a[o..o + lanes]);
                let vb = <$vt>::from_slice($simd, &b[o..o + lanes]);
                *slot = $step(*slot, va, vb);
            }
        }
        let mut folded = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        let mut i = blocks * block;
        while i + lanes <= n {
            let va = <$vt>::from_slice($simd, &a[i..i + lanes]);
            let vb = <$vt>::from_slice($simd, &b[i..i + lanes]);
            folded = $step(folded, va, vb);
            i += lanes;
        }
        let mut sum: $elem = folded.reduce_sum();
        while i < n {
            sum = $scalar(sum, a[i], b[i]);
            i += 1;
        }
        sum
    }};
}

// ─── f32 ────────────────────────────────────────────────────────────────

/// Inner product `a · b`.
#[simd]
pub(crate) fn dot_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    reduce_native!(
        simd, S::f32s, f32, a, b,
        |acc: S::f32s, va: S::f32s, vb: S::f32s| va.mul_add(vb, acc),
        |s: f32, x: f32, y: f32| s + x * y
    )
}

/// Squared Euclidean distance.
#[simd]
pub(crate) fn l2sq_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    reduce_native!(
        simd, S::f32s, f32, a, b,
        |acc: S::f32s, va: S::f32s, vb: S::f32s| { let d = va - vb; d.mul_add(d, acc) },
        |s: f32, x: f32, y: f32| { let d = x - y; s + d * d }
    )
}

/// Manhattan distance.
#[simd]
pub(crate) fn l1_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    reduce_native!(
        simd, S::f32s, f32, a, b,
        |acc: S::f32s, va: S::f32s, vb: S::f32s| acc + (va - vb).abs(),
        |s: f32, x: f32, y: f32| s + (x - y).abs()
    )
}

/// Cosine distance `1 − a·b / (|a|·|b|)`, `1.0` when either norm is zero.
///
/// Two accumulator triples rather than four: three running sums per
/// block already give six independent chains, and four triples would
/// spill the 16 vector registers of AVX2 and SSE.
#[simd]
pub(crate) fn cosine_f32<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let lanes = S::f32s::LEN;
    let block = lanes * 2;
    let zero = S::f32s::splat(simd, 0.0);
    let (mut dot, mut na, mut nb) = ([zero; 2], [zero; 2], [zero; 2]);
    let blocks = n / block;
    for c in 0..blocks {
        let base = c * block;
        for u in 0..2 {
            let o = base + u * lanes;
            let va = S::f32s::from_slice(simd, &a[o..o + lanes]);
            let vb = S::f32s::from_slice(simd, &b[o..o + lanes]);
            dot[u] = va.mul_add(vb, dot[u]);
            na[u] = va.mul_add(va, na[u]);
            nb[u] = vb.mul_add(vb, nb[u]);
        }
    }
    let (mut d, mut x, mut y) = (dot[0] + dot[1], na[0] + na[1], nb[0] + nb[1]);
    let mut i = blocks * block;
    while i + lanes <= n {
        let va = S::f32s::from_slice(simd, &a[i..i + lanes]);
        let vb = S::f32s::from_slice(simd, &b[i..i + lanes]);
        d = va.mul_add(vb, d);
        x = va.mul_add(va, x);
        y = vb.mul_add(vb, y);
        i += lanes;
    }
    let (mut d, mut x, mut y) = (d.reduce_sum(), x.reduce_sum(), y.reduce_sum());
    while i < n {
        d += a[i] * b[i];
        x += a[i] * a[i];
        y += b[i] * b[i];
        i += 1;
    }
    cosine_distance(d, x, y)
}

/// `1 − dot / sqrt(na · nb)`, `1.0` for a zero denominator.
#[inline(always)]
fn cosine_distance(dot: f32, na: f32, nb: f32) -> f32 {
    let denom = (na * nb).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

// ─── f64 ────────────────────────────────────────────────────────────────

/// Inner product of f64 vectors, accumulated in f64.
#[simd]
pub(crate) fn dot_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    reduce_native!(
        simd, S::f64s, f64, a, b,
        |acc: S::f64s, va: S::f64s, vb: S::f64s| va.mul_add(vb, acc),
        |s: f64, x: f64, y: f64| s + x * y
    )
}

/// Squared Euclidean distance of f64 vectors.
#[simd]
pub(crate) fn l2sq_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    reduce_native!(
        simd, S::f64s, f64, a, b,
        |acc: S::f64s, va: S::f64s, vb: S::f64s| { let d = va - vb; d.mul_add(d, acc) },
        |s: f64, x: f64, y: f64| { let d = x - y; s + d * d }
    )
}

/// Manhattan distance of f64 vectors.
#[simd]
pub(crate) fn l1_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    reduce_native!(
        simd, S::f64s, f64, a, b,
        |acc: S::f64s, va: S::f64s, vb: S::f64s| acc + (va - vb).abs(),
        |s: f64, x: f64, y: f64| s + (x - y).abs()
    )
}

/// Cosine distance of f64 vectors, computed in f64.
#[simd]
pub(crate) fn cosine_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let lanes = S::f64s::LEN;
    let zero = S::f64s::splat(simd, 0.0);
    let (mut d, mut x, mut y) = (zero, zero, zero);
    let mut i = 0;
    while i + lanes <= n {
        let va = S::f64s::from_slice(simd, &a[i..i + lanes]);
        let vb = S::f64s::from_slice(simd, &b[i..i + lanes]);
        d = va.mul_add(vb, d);
        x = va.mul_add(va, x);
        y = vb.mul_add(vb, y);
        i += lanes;
    }
    let (mut d, mut x, mut y) = (d.reduce_sum(), x.reduce_sum(), y.reduce_sum());
    while i < n {
        d += a[i] * b[i];
        x += a[i] * a[i];
        y += b[i] * b[i];
        i += 1;
    }
    let denom = (x * y).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - d / denom }
}

// ─── f16 ────────────────────────────────────────────────────────────────
//
// f16 lanes are widened to f32 sixteen at a time and accumulated in
// f32, two accumulators deep.

/// Shared loop skeleton for f16 reductions: decode 16 lanes of each
/// operand, apply `$step`, fold, then a scalar tail through `half`.
macro_rules! reduce_f16 {
    ($simd:ident, $a:ident, $b:ident, $step:expr, $scalar:expr) => {{
        let n = $a.len().min($b.len());
        let a = half::slice::HalfFloatSliceExt::reinterpret_cast(&$a[..n]);
        let b = half::slice::HalfFloatSliceExt::reinterpret_cast(&$b[..n]);
        let (a16, a_tail) = a.as_chunks::<16>();
        let (b16, _) = b.as_chunks::<16>();
        let zero = f32x16::splat($simd, 0.0);
        let mut acc = [zero; 2];
        let pairs = a16.len() / 2;
        for p in 0..pairs {
            for (u, slot) in acc.iter_mut().enumerate() {
                let va = $simd.f16x16_to_f32(&a16[2 * p + u]);
                let vb = $simd.f16x16_to_f32(&b16[2 * p + u]);
                *slot = $step(*slot, va, vb);
            }
        }
        let mut folded = acc[0] + acc[1];
        if a16.len() % 2 == 1 {
            let last = a16.len() - 1;
            let va = $simd.f16x16_to_f32(&a16[last]);
            let vb = $simd.f16x16_to_f32(&b16[last]);
            folded = $step(folded, va, vb);
        }
        let mut sum = folded.reduce_sum();
        let tail_start = n - a_tail.len();
        for i in tail_start..n {
            let x = half::f16::from_bits(a[i]).to_f32();
            let y = half::f16::from_bits(b[i]).to_f32();
            sum = $scalar(sum, x, y);
        }
        sum
    }};
}

/// Inner product of f16 vectors, accumulated in f32.
#[simd]
pub(crate) fn dot_f16<S: F16Lanes>(simd: S, a: &[half::f16], b: &[half::f16]) -> f32 {
    reduce_f16!(
        simd, a, b,
        |acc: f32x16<S>, va: f32x16<S>, vb: f32x16<S>| va.mul_add(vb, acc),
        |s: f32, x: f32, y: f32| s + x * y
    )
}

/// Squared Euclidean distance of f16 vectors, accumulated in f32.
#[simd]
pub(crate) fn l2sq_f16<S: F16Lanes>(simd: S, a: &[half::f16], b: &[half::f16]) -> f32 {
    reduce_f16!(
        simd, a, b,
        |acc: f32x16<S>, va: f32x16<S>, vb: f32x16<S>| { let d = va - vb; d.mul_add(d, acc) },
        |s: f32, x: f32, y: f32| { let d = x - y; s + d * d }
    )
}

/// Manhattan distance of f16 vectors, accumulated in f32.
#[simd]
pub(crate) fn l1_f16<S: F16Lanes>(simd: S, a: &[half::f16], b: &[half::f16]) -> f32 {
    reduce_f16!(
        simd, a, b,
        |acc: f32x16<S>, va: f32x16<S>, vb: f32x16<S>| acc + (va - vb).abs(),
        |s: f32, x: f32, y: f32| s + (x - y).abs()
    )
}

/// Cosine distance of f16 vectors, accumulated in f32.
#[simd]
pub(crate) fn cosine_f16<S: F16Lanes>(simd: S, a: &[half::f16], b: &[half::f16]) -> f32 {
    let n = a.len().min(b.len());
    let a = half::slice::HalfFloatSliceExt::reinterpret_cast(&a[..n]);
    let b = half::slice::HalfFloatSliceExt::reinterpret_cast(&b[..n]);
    let (a16, a_tail) = a.as_chunks::<16>();
    let (b16, _) = b.as_chunks::<16>();
    let zero = f32x16::splat(simd, 0.0);
    let (mut d, mut x, mut y) = (zero, zero, zero);
    for (ca, cb) in a16.iter().zip(b16) {
        let va = simd.f16x16_to_f32(ca);
        let vb = simd.f16x16_to_f32(cb);
        d = va.mul_add(vb, d);
        x = va.mul_add(va, x);
        y = vb.mul_add(vb, y);
    }
    let (mut d, mut x, mut y) = (d.reduce_sum(), x.reduce_sum(), y.reduce_sum());
    for i in (n - a_tail.len())..n {
        let p = half::f16::from_bits(a[i]).to_f32();
        let q = half::f16::from_bits(b[i]).to_f32();
        d += p * q;
        x += p * p;
        y += q * q;
    }
    cosine_distance(d, x, y)
}
