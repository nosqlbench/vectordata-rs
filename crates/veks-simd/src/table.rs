// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-level kernel tables and the dispatch entry points.
//!
//! Each compiled [`SimdLevel`] gets one static [`Kernels`] table of
//! function pointers, monomorphized for that level's token. Dispatch is
//! picking a table: [`Kernels::detected`] does it once per process, and
//! everything after that is a plain indirect call.

use crate::batch::{self, PackedBatches, TransposedBatch};
use crate::level::SimdLevel;
use crate::metric::Metric;
use crate::{convert, pairwise};

/// Pairwise distance over f32 vectors.
pub type DistFnF32 = fn(&[f32], &[f32]) -> f32;
/// Pairwise distance over f16 vectors, accumulated in f32.
pub type DistFnF16 = fn(&[half::f16], &[half::f16]) -> f32;
/// Pairwise distance over f64 vectors, accumulated in f64, returned as f32.
pub type DistFnF64 = fn(&[f64], &[f64]) -> f32;
/// The L2 norm of an f32 vector.
pub type NormFnF32 = fn(&[f32]) -> f32;
/// One base vector against one [`TransposedBatch`]: writes 16 distances
/// to `out[..16]`.
pub type BatchFnF32 = fn(&TransposedBatch, &[f32], &mut [f32]);
/// One base vector against two batches: batch `a` to `out[..16]`, batch
/// `b` to `out[16..]`.
pub type DualBatchFnF32 = fn(&TransposedBatch, &TransposedBatch, &[f32], &mut [f32; 32]);
/// One base vector against every sub-batch of a [`PackedBatches`] with
/// the negated dot product: sub-batch `si`, lane `qi` to `out[si * 16 + qi]`.
pub type PackedNegDotFn = fn(&[f32], &PackedBatches, &mut [f32]);
/// Widen f16 to f32 into `dst[..src.len()]`.
pub type ConvertF16ToF32 = fn(&[half::f16], &mut [f32]);
/// Narrow f32 to f16 (nearest-even) into `dst[..src.len()]`.
pub type ConvertF32ToF16 = fn(&[f32], &mut [half::f16]);
/// Convert little-endian element bytes; returns the bytes written.
pub type ConvertBytes = fn(&[u8], &mut [u8]) -> usize;

/// The kernels compiled for one [`SimdLevel`].
///
/// Obtain the table for this CPU with [`Kernels::detected`], or a
/// specific supported level with [`Kernels::for_level`] (for tests and
/// comparisons). Accessors return plain function pointers; resolve them
/// once, outside the hot loop.
pub struct Kernels {
    level: SimdLevel,
    dist_f32: [DistFnF32; 4],
    dist_f16: [DistFnF16; 4],
    dist_f64: [DistFnF64; 4],
    dot_f32: DistFnF32,
    norm_f32: NormFnF32,
    batch: [BatchFnF32; 4],
    dual: [DualBatchFnF32; 4],
    packed_neg_dot: PackedNegDotFn,
    f16_to_f32: ConvertF16ToF32,
    f32_to_f16: ConvertF32ToF16,
    f16_bytes_to_f32: ConvertBytes,
    f32_bytes_to_f16: ConvertBytes,
}

impl Kernels {
    /// The table for the strongest level this CPU supports. Detection
    /// runs once per process.
    pub fn detected() -> &'static Kernels {
        table_for(SimdLevel::detected())
    }

    /// The table for `level`, or `None` when the level is not compiled
    /// for this target or this CPU cannot run it.
    pub fn for_level(level: SimdLevel) -> Option<&'static Kernels> {
        SimdLevel::supported().contains(&level).then(|| table_for(level))
    }

    /// The level these kernels are compiled for.
    pub fn level(&self) -> SimdLevel {
        self.level
    }

    /// Pairwise distance over f32 vectors.
    pub fn distance_f32(&self, metric: Metric) -> DistFnF32 {
        self.dist_f32[metric.index()]
    }

    /// Pairwise distance over f16 vectors, accumulated in f32.
    pub fn distance_f16(&self, metric: Metric) -> DistFnF16 {
        self.dist_f16[metric.index()]
    }

    /// Pairwise distance over f64 vectors, accumulated in f64.
    pub fn distance_f64(&self, metric: Metric) -> DistFnF64 {
        self.dist_f64[metric.index()]
    }

    /// The plain (not negated) inner product of f32 vectors.
    pub fn dot_f32(&self) -> DistFnF32 {
        self.dot_f32
    }

    /// The L2 norm of an f32 vector.
    pub fn norm_f32(&self) -> NormFnF32 {
        self.norm_f32
    }

    /// One base vector against one transposed batch.
    pub fn batch_f32(&self, metric: Metric) -> BatchFnF32 {
        self.batch[metric.index()]
    }

    /// One base vector against two transposed batches.
    pub fn dual_batch_f32(&self, metric: Metric) -> DualBatchFnF32 {
        self.dual[metric.index()]
    }

    /// One base vector against every sub-batch of a packed query set,
    /// negated dot product.
    pub fn packed_neg_dot_f32(&self) -> PackedNegDotFn {
        self.packed_neg_dot
    }

    /// Bulk f16 → f32.
    pub fn f16_to_f32(&self) -> ConvertF16ToF32 {
        self.f16_to_f32
    }

    /// Bulk f32 → f16, rounding to nearest-even.
    pub fn f32_to_f16(&self) -> ConvertF32ToF16 {
        self.f32_to_f16
    }

    /// Bulk little-endian f16 bytes → little-endian f32 bytes.
    pub fn f16_bytes_to_f32_bytes(&self) -> ConvertBytes {
        self.f16_bytes_to_f32
    }

    /// Bulk little-endian f32 bytes → little-endian f16 bytes.
    pub fn f32_bytes_to_f16_bytes(&self) -> ConvertBytes {
        self.f32_bytes_to_f16
    }
}

/// Generate one level's table.
///
/// `$token` constructs the level's `fearless_simd` token. For levels
/// detected at runtime it is an `unsafe` `assume_supported()`, which is
/// sound because the table is only reachable through [`table_for`],
/// whose callers pass a level from [`SimdLevel::detected`] or check it
/// against [`SimdLevel::supported`] first.
///
/// `$group` is how many 16-lane packed sub-batches keep their
/// accumulators in registers: one register per sub-batch on AVX-512 (32
/// registers), two on AVX2 (16), four on NEON (32) and SSE (16).
macro_rules! level_table {
    ($module:ident, $level:expr, $group:expr, $token:expr) => {
        mod $module {
            use super::*;

            #[inline(always)]
            fn token() -> impl crate::f16_lanes::F16Lanes {
                $token
            }

            fn l2_f32(a: &[f32], b: &[f32]) -> f32 { pairwise::l2sq_f32(token(), a, b) }
            fn cos_f32(a: &[f32], b: &[f32]) -> f32 { pairwise::cosine_f32(token(), a, b) }
            fn ndot_f32(a: &[f32], b: &[f32]) -> f32 { -pairwise::dot_f32(token(), a, b) }
            fn l1_f32(a: &[f32], b: &[f32]) -> f32 { pairwise::l1_f32(token(), a, b) }
            fn dot_f32(a: &[f32], b: &[f32]) -> f32 { pairwise::dot_f32(token(), a, b) }
            fn norm_f32(a: &[f32]) -> f32 { pairwise::dot_f32(token(), a, a).sqrt() }

            fn l2_f16(a: &[half::f16], b: &[half::f16]) -> f32 { pairwise::l2sq_f16(token(), a, b) }
            fn cos_f16(a: &[half::f16], b: &[half::f16]) -> f32 { pairwise::cosine_f16(token(), a, b) }
            fn ndot_f16(a: &[half::f16], b: &[half::f16]) -> f32 { -pairwise::dot_f16(token(), a, b) }
            fn l1_f16(a: &[half::f16], b: &[half::f16]) -> f32 { pairwise::l1_f16(token(), a, b) }

            fn l2_f64(a: &[f64], b: &[f64]) -> f32 { pairwise::l2sq_f64(token(), a, b) as f32 }
            fn cos_f64(a: &[f64], b: &[f64]) -> f32 { pairwise::cosine_f64(token(), a, b) as f32 }
            fn ndot_f64(a: &[f64], b: &[f64]) -> f32 { -pairwise::dot_f64(token(), a, b) as f32 }
            fn l1_f64(a: &[f64], b: &[f64]) -> f32 { pairwise::l1_f64(token(), a, b) as f32 }

            fn b_l2(t: &TransposedBatch, b: &[f32], o: &mut [f32]) { batch::batch_l2sq(token(), t, b, o) }
            fn b_cos(t: &TransposedBatch, b: &[f32], o: &mut [f32]) { batch::batch_cosine(token(), t, b, o) }
            fn b_ndot(t: &TransposedBatch, b: &[f32], o: &mut [f32]) { batch::batch_neg_dot(token(), t, b, o) }
            fn b_l1(t: &TransposedBatch, b: &[f32], o: &mut [f32]) { batch::batch_l1(token(), t, b, o) }

            fn d_l2(x: &TransposedBatch, y: &TransposedBatch, b: &[f32], o: &mut [f32; 32]) { batch::dual_l2sq(token(), x, y, b, o) }
            fn d_cos(x: &TransposedBatch, y: &TransposedBatch, b: &[f32], o: &mut [f32; 32]) { batch::dual_cosine(token(), x, y, b, o) }
            fn d_ndot(x: &TransposedBatch, y: &TransposedBatch, b: &[f32], o: &mut [f32; 32]) { batch::dual_neg_dot(token(), x, y, b, o) }
            fn d_l1(x: &TransposedBatch, y: &TransposedBatch, b: &[f32], o: &mut [f32; 32]) { batch::dual_l1(token(), x, y, b, o) }

            fn packed(b: &[f32], p: &PackedBatches, o: &mut [f32]) { batch::packed_neg_dot(token(), b, p, o, $group) }

            fn h2f(s: &[half::f16], d: &mut [f32]) { convert::f16_to_f32(token(), s, d) }
            fn f2h(s: &[f32], d: &mut [half::f16]) { convert::f32_to_f16(token(), s, d) }
            fn h2f_bytes(s: &[u8], d: &mut [u8]) -> usize { convert::f16_bytes_to_f32_bytes(token(), s, d) }
            fn f2h_bytes(s: &[u8], d: &mut [u8]) -> usize { convert::f32_bytes_to_f16_bytes(token(), s, d) }

            pub(super) static KERNELS: Kernels = Kernels {
                level: $level,
                dist_f32: [l2_f32, cos_f32, ndot_f32, l1_f32],
                dist_f16: [l2_f16, cos_f16, ndot_f16, l1_f16],
                dist_f64: [l2_f64, cos_f64, ndot_f64, l1_f64],
                dot_f32,
                norm_f32,
                batch: [b_l2, b_cos, b_ndot, b_l1],
                dual: [d_l2, d_cos, d_ndot, d_l1],
                packed_neg_dot: packed,
                f16_to_f32: h2f,
                f32_to_f16: f2h,
                f16_bytes_to_f32: h2f_bytes,
                f32_bytes_to_f16: f2h_bytes,
            };
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_tables {
    use super::*;

    // SAFETY (all four): see `level_table!` — reachable only for a level
    // this CPU was detected to support.
    level_table!(avx512, SimdLevel::Avx512, 16, unsafe { fearless_simd::Avx512::assume_supported() });
    level_table!(avx2, SimdLevel::Avx2, 6, unsafe { fearless_simd::Avx2::assume_supported() });
    level_table!(sse4_2, SimdLevel::Sse4_2, 3, unsafe { fearless_simd::Sse4_2::assume_supported() });
    level_table!(sse2, SimdLevel::Sse2, 3, unsafe { fearless_simd::Sse2::assume_supported() });

    pub(super) fn table(level: SimdLevel) -> &'static Kernels {
        match level {
            SimdLevel::Avx512 => &avx512::KERNELS,
            SimdLevel::Avx2 => &avx2::KERNELS,
            SimdLevel::Sse4_2 => &sse4_2::KERNELS,
            _ => &sse2::KERNELS,
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_tables {
    use super::*;

    // SAFETY: NEON is mandatory on aarch64.
    level_table!(neon, SimdLevel::Neon, 6, unsafe { fearless_simd::Neon::assume_supported() });

    pub(super) fn table(_: SimdLevel) -> &'static Kernels {
        &neon::KERNELS
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_tables {
    use super::*;

    // SAFETY: the target is compiled with `simd128`.
    level_table!(wasm128, SimdLevel::Wasm128, 3, unsafe { fearless_simd::WasmSimd128::assume_supported() });

    pub(super) fn table(_: SimdLevel) -> &'static Kernels {
        &wasm128::KERNELS
    }
}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod scalar_tables {
    use super::*;

    level_table!(scalar, SimdLevel::Scalar, 4, fearless_simd::Fallback::new());

    pub(super) fn table(_: SimdLevel) -> &'static Kernels {
        &scalar::KERNELS
    }
}

/// The table for a level the caller has established this CPU supports.
fn table_for(level: SimdLevel) -> &'static Kernels {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86_tables::table(level)
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64_tables::table(level)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm_tables::table(level)
    }
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_tables::table(level)
    }
}
