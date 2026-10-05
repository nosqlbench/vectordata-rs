// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Bulk f16 ↔ f32 conversion.
//!
//! Typed variants convert between `half::f16` and `f32` slices. Byte
//! variants convert little-endian element bytes, which is what file
//! records hold, and need no particular alignment.

use fearless_simd::{f32x16, prelude::*};
use fearless_simd_macros::simd;

use crate::f16_lanes::F16Lanes;

/// Widen `src` into `dst[..src.len()]`. Panics if `dst` is shorter.
#[simd]
pub(crate) fn f16_to_f32<S: F16Lanes>(simd: S, src: &[half::f16], dst: &mut [f32]) {
    let n = src.len();
    let bits = half::slice::HalfFloatSliceExt::reinterpret_cast(src);
    let dst = &mut dst[..n];
    let (src16, _) = bits.as_chunks::<16>();
    let (dst16, _) = dst.as_chunks_mut::<16>();
    for (s, d) in src16.iter().zip(dst16.iter_mut()) {
        simd.f16x16_to_f32(s).store_array(d);
    }
    for i in src16.len() * 16..n {
        dst[i] = src[i].to_f32();
    }
}

/// Narrow `src` into `dst[..src.len()]`, rounding to nearest-even.
/// Panics if `dst` is shorter.
#[simd]
pub(crate) fn f32_to_f16<S: F16Lanes>(simd: S, src: &[f32], dst: &mut [half::f16]) {
    let n = src.len();
    let dst = half::slice::HalfFloatSliceExt::reinterpret_cast_mut(&mut dst[..n]);
    let (src8, _) = src.as_chunks::<8>();
    let (dst8, _) = dst.as_chunks_mut::<8>();
    for (s, d) in src8.iter().zip(dst8.iter_mut()) {
        *d = simd.f32x8_to_f16(s);
    }
    for i in src8.len() * 8..n {
        dst[i] = half::f16::from_f32(src[i]).to_bits();
    }
}

/// Widen little-endian f16 bytes to little-endian f32 bytes. Converts
/// `src.len() / 2` elements and returns the bytes written. Panics if
/// `dst` is shorter than twice `src`.
#[simd]
pub(crate) fn f16_bytes_to_f32_bytes<S: F16Lanes>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
    let n = src.len() / 2;
    let (src32, _) = src[..n * 2].as_chunks::<32>();
    let (dst64, _) = dst[..n * 4].as_chunks_mut::<64>();
    for (s, d) in src32.iter().zip(dst64.iter_mut()) {
        let mut bits = [0u16; 16];
        for (b, pair) in bits.iter_mut().zip(s.as_chunks::<2>().0) {
            *b = u16::from_le_bytes(*pair);
        }
        let wide: f32x16<S> = simd.f16x16_to_f32(&bits);
        for (out, v) in d.as_chunks_mut::<4>().0.iter_mut().zip(wide.to_array()) {
            *out = v.to_le_bytes();
        }
    }
    for i in src32.len() * 16..n {
        let v = half::f16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32();
        dst[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    n * 4
}

/// Narrow little-endian f32 bytes to little-endian f16 bytes, rounding
/// to nearest-even. Converts `src.len() / 4` elements and returns the
/// bytes written. Panics if `dst` is shorter than half of `src`.
#[simd]
pub(crate) fn f32_bytes_to_f16_bytes<S: F16Lanes>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
    let n = src.len() / 4;
    let (src32, _) = src[..n * 4].as_chunks::<32>();
    let (dst16, _) = dst[..n * 2].as_chunks_mut::<16>();
    for (s, d) in src32.iter().zip(dst16.iter_mut()) {
        let mut vals = [0f32; 8];
        for (v, quad) in vals.iter_mut().zip(s.as_chunks::<4>().0) {
            *v = f32::from_le_bytes(*quad);
        }
        let narrow = simd.f32x8_to_f16(&vals);
        for (out, h) in d.as_chunks_mut::<2>().0.iter_mut().zip(narrow) {
            *out = h.to_le_bytes();
        }
    }
    for i in src32.len() * 8..n {
        let v = f32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]]);
        dst[2 * i..2 * i + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    n * 2
}
