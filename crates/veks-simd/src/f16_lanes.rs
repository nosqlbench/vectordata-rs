// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Lane-wise f16 ↔ f32 conversion, per dispatch level.
//!
//! x86 levels with F16C (`Avx2`, `Avx512`) use `vcvtph2ps` and
//! `vcvtps2ph`. Every other level widens with integer operations that
//! are exact for every bit pattern, including subnormals, infinities
//! and NaN, and match the `half` crate bit for bit (a NaN comes out
//! quiet, payload kept); it narrows through `half`, which rounds to
//! nearest-even as F16C does. aarch64's own conversion instructions
//! need Rust's unstable `f16` intrinsics, so NEON takes the integer
//! path for widening.

use fearless_simd::{Simd, f32x8, f32x16, prelude::*, u16x8, u32x8};

/// A SIMD level that can convert between f16 and f32 lanes.
pub(crate) trait F16Lanes: Simd {
    /// Widen 8 f16 values (raw IEEE 754 binary16 bits) to f32.
    #[inline(always)]
    fn f16x8_to_f32(self, src: &[u16; 8]) -> f32x8<Self> {
        widen_integer(self, src)
    }

    /// Widen 16 f16 values to f32.
    #[inline(always)]
    fn f16x16_to_f32(self, src: &[u16; 16]) -> f32x16<Self> {
        let (lo, hi) = src.as_chunks::<8>().0.split_at(1);
        self.f16x8_to_f32(&lo[0]).combine(self.f16x8_to_f32(&hi[0]))
    }

    /// Narrow 8 f32 values to f16 bits, rounding to nearest-even.
    #[inline(always)]
    fn f32x8_to_f16(self, src: &[f32; 8]) -> [u16; 8] {
        src.map(|v| half::f16::from_f32(v).to_bits())
    }
}

/// Exact f16 → f32 with integer operations.
///
/// Shifting the 15 magnitude bits up by 13 lines the f16 exponent and
/// mantissa up with f32's; adding 112 to the exponent field rebiases a
/// normal value from f16's bias of 15 to f32's 127. A subnormal f16 is
/// its mantissa times 2⁻²⁴, which an integer-to-float conversion and an
/// exact multiply rebuild as a normal f32. Infinity and NaN keep their
/// mantissa under an all-ones exponent.
///
/// No floating-point operation here sees a subnormal operand: x86
/// handles those with a microcode assist costing tens of cycles per
/// lane, which is what a single float rebias multiply over every lane
/// would hit on f16 subnormals.
#[inline(always)]
fn widen_integer<S: Simd>(simd: S, src: &[u16; 8]) -> f32x8<S> {
    let h = u16x8::from_slice(simd, src);
    let (lo, hi) = simd.widen_u16x8(h);
    let w: u32x8<S> = lo.combine(hi);
    let sign = (w & 0x8000) << 16;
    let magnitude = w & 0x7fff;
    let exponent = w & 0x7c00;
    let shifted = magnitude << 13;
    let normal = shifted + (112u32 << 23);
    // 2^-24, exactly: the weight of a subnormal f16's lowest mantissa bit.
    let subnormal = (magnitude.to_float::<f32x8<S>>() * f32::from_bits(0x3380_0000)).bitcast::<u32x8<S>>();
    let is_subnormal = exponent.simd_eq(0);
    let is_special = exponent.simd_eq(0x7c00);
    // Infinity keeps a zero mantissa; NaN gets the quiet bit.
    let is_nan = (w & 0x03ff).simd_gt(0);
    let quiet = is_nan.select(u32x8::splat(simd, 0x0040_0000), u32x8::splat(simd, 0));
    let special = shifted | 0x7f80_0000 | quiet;
    let finite = is_subnormal.select(subnormal, normal);
    (is_special.select(special, finite) | sign).bitcast::<f32x8<S>>()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::F16Lanes;
    use fearless_simd::{Avx2, Avx512, Sse2, Sse4_2, f32x8, f32x16, prelude::*};

    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    impl F16Lanes for Sse2 {}
    impl F16Lanes for Sse4_2 {}

    /// `vcvtph2ps` on 8 lanes.
    ///
    /// # Safety
    /// The CPU must support `f16c` and `avx`.
    #[inline(always)]
    unsafe fn cvtph8(src: &[u16; 8]) -> __m256 {
        // SAFETY: the caller guarantees the features; the unaligned load
        // reads exactly the 16 bytes of `src`.
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(src.as_ptr().cast())) }
    }

    /// `vcvtps2ph` (round to nearest-even) on 8 lanes.
    ///
    /// # Safety
    /// The CPU must support `f16c` and `avx`.
    #[inline(always)]
    unsafe fn cvtps8(src: &[f32; 8]) -> [u16; 8] {
        let mut out = [0u16; 8];
        // SAFETY: the caller guarantees the features; the load reads the
        // 32 bytes of `src` and the store writes the 16 bytes of `out`.
        unsafe {
            let h = _mm256_cvtps_ph(_mm256_loadu_ps(src.as_ptr()), _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(out.as_mut_ptr().cast(), h);
        }
        out
    }

    impl F16Lanes for Avx2 {
        #[inline(always)]
        fn f16x8_to_f32(self, src: &[u16; 8]) -> f32x8<Self> {
            // SAFETY: an `Avx2` token proves x86-64-v3, which includes
            // `f16c` and `avx`.
            unsafe { cvtph8(src) }.simd_into(self)
        }

        #[inline(always)]
        fn f32x8_to_f16(self, src: &[f32; 8]) -> [u16; 8] {
            // SAFETY: as above.
            unsafe { cvtps8(src) }
        }
    }

    impl F16Lanes for Avx512 {
        #[inline(always)]
        fn f16x8_to_f32(self, src: &[u16; 8]) -> f32x8<Self> {
            // SAFETY: the Ice Lake feature set includes `avx512f`, which
            // implies `f16c` and `avx`.
            unsafe { cvtph8(src) }.simd_into(self)
        }

        #[inline(always)]
        fn f16x16_to_f32(self, src: &[u16; 16]) -> f32x16<Self> {
            // SAFETY: as above; the load reads exactly the 32 bytes of `src`.
            let v = unsafe { _mm512_cvtph_ps(_mm256_loadu_si256(src.as_ptr().cast())) };
            v.simd_into(self)
        }

        #[inline(always)]
        fn f32x8_to_f16(self, src: &[f32; 8]) -> [u16; 8] {
            // SAFETY: as above.
            unsafe { cvtps8(src) }
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl F16Lanes for fearless_simd::Neon {}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl F16Lanes for fearless_simd::WasmSimd128 {}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
impl F16Lanes for fearless_simd::Fallback {}
