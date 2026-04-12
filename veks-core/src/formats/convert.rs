// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Element-type conversion for vector data (f16↔f32↔f64).
//!
//! When converting between xvec formats with different element sizes
//! (e.g. mvec→fvec), raw bytes must be re-interpreted and widened or
//! narrowed element-by-element. This module provides SIMD-accelerated
//! conversion where available.
//!
//! ## SIMD support
//!
//! On x86_64:
//! - **AVX-512 + F16C**: converts 16 f16→f32 per instruction via `vcvtph2ps`
//! - **AVX2 + F16C**: converts 8 f16→f32 per instruction
//! - **Scalar fallback**: uses `half::f16::to_f32()` per element

/// Convert a record's raw bytes from one element size to another.
///
/// Returns the converted bytes, or `None` if no conversion is needed
/// (same element size) or the conversion is not supported.
///
/// Supported conversions:
/// - 2→4 (f16→f32): SIMD-accelerated
/// - 4→2 (f32→f16): SIMD-accelerated
/// - 4→8 (f32→f64): scalar
/// - 8→4 (f64→f32): scalar
/// - 2→8 (f16→f64): scalar (via f32)
/// - 8→2 (f64→f16): scalar (via f32)
pub fn convert_elements(data: &[u8], from_size: usize, to_size: usize) -> Option<Vec<u8>> {
    if from_size == to_size {
        return None;
    }
    match (from_size, to_size) {
        (2, 4) => Some(f16_to_f32(data)),
        (4, 2) => Some(f32_to_f16(data)),
        (4, 8) => Some(f32_to_f64(data)),
        (8, 4) => Some(f64_to_f32(data)),
        (2, 8) => {
            let f32_bytes = f16_to_f32(data);
            Some(f32_to_f64(&f32_bytes))
        }
        (8, 2) => {
            let f32_bytes = f64_to_f32(data);
            Some(f32_to_f16(&f32_bytes))
        }
        _ => None,
    }
}

/// Convert elements into a pre-allocated output buffer, avoiding
/// per-record heap allocation.
///
/// `out` must be at least `(data.len() / from_size) * to_size` bytes.
/// Returns the number of bytes written to `out`, or `None` if the
/// conversion is not supported.
pub fn convert_elements_into(
    data: &[u8],
    from_size: usize,
    to_size: usize,
    out: &mut [u8],
) -> Option<usize> {
    if from_size == to_size {
        return None;
    }
    match (from_size, to_size) {
        (2, 4) => Some(f16_to_f32_into(data, out)),
        (4, 2) => Some(f32_to_f16_into(data, out)),
        (4, 8) => Some(f32_to_f64_into(data, out)),
        (8, 4) => Some(f64_to_f32_into(data, out)),
        _ => None,
    }
}

/// Convert f16 bytes to f32 bytes into a pre-allocated buffer.
fn f16_to_f32_into(data: &[u8], out: &mut [u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return unsafe { f16_to_f32_avx512_into(data, out) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { f16_to_f32_avx2_into(data, out) };
        }
    }
    f16_to_f32_scalar_into(data, out)
}

fn f16_to_f32_scalar_into(data: &[u8], out: &mut [u8]) -> usize {
    let n = data.len() / 2;
    for i in 0..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        let off = i * 4;
        out[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
    n * 4
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn f16_to_f32_avx2_into(data: &[u8], out: &mut [u8]) -> usize {
    use std::arch::x86_64::*;
    let n = data.len() / 2;
    let chunks = n / 8;
    let src = data.as_ptr() as *const u16;
    let dst = out.as_mut_ptr() as *mut f32;
    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let v_i = _mm_loadu_si128(src.add(offset) as *const __m128i);
            let v_f = _mm256_cvtph_ps(v_i);
            _mm256_storeu_ps(dst.add(offset), v_f);
        }
    }
    let tail_start = chunks * 8;
    for i in tail_start..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        let off = i * 4;
        out[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
    n * 4
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "f16c")]
unsafe fn f16_to_f32_avx512_into(data: &[u8], out: &mut [u8]) -> usize {
    use std::arch::x86_64::*;
    let n = data.len() / 2;
    let chunks = n / 16;
    let src = data.as_ptr() as *const u16;
    let dst = out.as_mut_ptr() as *mut f32;
    for i in 0..chunks {
        let offset = i * 16;
        unsafe {
            let v_i = _mm256_loadu_si256(src.add(offset) as *const __m256i);
            let v_f = _mm512_cvtph_ps(v_i);
            _mm512_storeu_ps(dst.add(offset), v_f);
        }
    }
    let tail_start = chunks * 16;
    for i in tail_start..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        let off = i * 4;
        out[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
    n * 4
}

/// Convert f32 bytes to f16 bytes into a pre-allocated buffer.
fn f32_to_f16_into(data: &[u8], out: &mut [u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return unsafe { f32_to_f16_avx512_into(data, out) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { f32_to_f16_avx2_into(data, out) };
        }
    }
    f32_to_f16_scalar_into(data, out)
}

fn f32_to_f16_scalar_into(data: &[u8], out: &mut [u8]) -> usize {
    let n = data.len() / 4;
    for i in 0..n {
        let off = i * 4;
        let val = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let h = half::f16::from_f32(val);
        let dst = i * 2;
        out[dst..dst + 2].copy_from_slice(&h.to_le_bytes());
    }
    n * 2
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn f32_to_f16_avx2_into(data: &[u8], out: &mut [u8]) -> usize {
    use std::arch::x86_64::*;
    let n = data.len() / 4;
    let src = data.as_ptr() as *const f32;
    let dst = out.as_mut_ptr() as *mut u16;

    let chunks = n / 8;
    for i in 0..chunks {
        unsafe {
            let v = _mm256_loadu_ps(src.add(i * 8));
            let h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(dst.add(i * 8) as *mut __m128i, h);
        }
    }
    for i in (chunks * 8)..n {
        unsafe {
            let val = *src.add(i);
            let h = half::f16::from_f32(val);
            *(dst.add(i)) = h.to_bits();
        }
    }
    n * 2
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "f16c")]
unsafe fn f32_to_f16_avx512_into(data: &[u8], out: &mut [u8]) -> usize {
    use std::arch::x86_64::*;
    let n = data.len() / 4;
    let src = data.as_ptr() as *const f32;
    let dst = out.as_mut_ptr() as *mut u16;

    let chunks = n / 16;
    for i in 0..chunks {
        unsafe {
            let v = _mm512_loadu_ps(src.add(i * 16));
            let h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256(dst.add(i * 16) as *mut __m256i, h);
        }
    }
    let remaining_start = chunks * 16;
    let remaining = n - remaining_start;
    if remaining >= 8 {
        let avx_chunks = remaining / 8;
        for i in 0..avx_chunks {
            let off = remaining_start + i * 8;
            unsafe {
                let v = _mm256_loadu_ps(src.add(off));
                let h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(dst.add(off) as *mut __m128i, h);
            }
        }
        let scalar_start = remaining_start + avx_chunks * 8;
        for i in scalar_start..n {
            unsafe {
                let val = *src.add(i);
                let h = half::f16::from_f32(val);
                *(dst.add(i)) = h.to_bits();
            }
        }
    } else {
        for i in remaining_start..n {
            unsafe {
                let val = *src.add(i);
                let h = half::f16::from_f32(val);
                *(dst.add(i)) = h.to_bits();
            }
        }
    }
    n * 2
}

fn f32_to_f64_into(data: &[u8], out: &mut [u8]) -> usize {
    let n = data.len() / 4;
    for i in 0..n {
        let off = i * 4;
        let val = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let dst = i * 8;
        out[dst..dst + 8].copy_from_slice(&(val as f64).to_le_bytes());
    }
    n * 8
}

fn f64_to_f32_into(data: &[u8], out: &mut [u8]) -> usize {
    let n = data.len() / 8;
    for i in 0..n {
        let off = i * 8;
        let val = f64::from_le_bytes([
            data[off], data[off + 1], data[off + 2], data[off + 3],
            data[off + 4], data[off + 5], data[off + 6], data[off + 7],
        ]);
        let dst = i * 4;
        out[dst..dst + 4].copy_from_slice(&(val as f32).to_le_bytes());
    }
    n * 4
}

/// Convert f16 bytes to f32 bytes, dispatching to the best available SIMD.
fn f16_to_f32(data: &[u8]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return unsafe { f16_to_f32_avx512(data) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { f16_to_f32_avx2(data) };
        }
    }
    f16_to_f32_scalar(data)
}

fn f16_to_f32_scalar(data: &[u8]) -> Vec<u8> {
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn f16_to_f32_avx2(data: &[u8]) -> Vec<u8> {
    use std::arch::x86_64::*;

    let n = data.len() / 2;
    let mut out = vec![0u8; n * 4];
    let chunks = n / 8;
    let src = data.as_ptr() as *const u16;
    let dst = out.as_mut_ptr() as *mut f32;

    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let v_i = _mm_loadu_si128(src.add(offset) as *const __m128i);
            let v_f = _mm256_cvtph_ps(v_i);
            _mm256_storeu_ps(dst.add(offset), v_f);
        }
    }

    // Scalar tail
    let tail_start = chunks * 8;
    for i in tail_start..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        let off = i * 4;
        out[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "f16c")]
unsafe fn f16_to_f32_avx512(data: &[u8]) -> Vec<u8> {
    use std::arch::x86_64::*;

    let n = data.len() / 2;
    let mut out = vec![0u8; n * 4];
    let chunks = n / 16;
    let src = data.as_ptr() as *const u16;
    let dst = out.as_mut_ptr() as *mut f32;

    for i in 0..chunks {
        let offset = i * 16;
        unsafe {
            let v_i = _mm256_loadu_si256(src.add(offset) as *const __m256i);
            let v_f = _mm512_cvtph_ps(v_i);
            _mm512_storeu_ps(dst.add(offset), v_f);
        }
    }

    // Scalar tail
    let tail_start = chunks * 16;
    for i in tail_start..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        let off = i * 4;
        out[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    out
}

/// Convert f32 bytes to f16 bytes.
fn f32_to_f16(data: &[u8]) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            return unsafe { f32_to_f16_avx512(data) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            return unsafe { f32_to_f16_avx2(data) };
        }
    }
    f32_to_f16_scalar(data)
}

fn f32_to_f16_scalar(data: &[u8]) -> Vec<u8> {
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let off = i * 4;
        let val = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let h = half::f16::from_f32(val);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn f32_to_f16_avx2(data: &[u8]) -> Vec<u8> {
    use std::arch::x86_64::*;

    let n = data.len() / 4;
    let mut out = vec![0u8; n * 2];
    let chunks = n / 8;
    let src = data.as_ptr() as *const f32;
    let dst = out.as_mut_ptr() as *mut u16;

    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let v_f = _mm256_loadu_ps(src.add(offset));
            // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 0x00
            let v_h = _mm256_cvtps_ph(v_f, 0x00);
            _mm_storeu_si128(dst.add(offset) as *mut __m128i, v_h);
        }
    }

    let tail_start = chunks * 8;
    for i in tail_start..n {
        let off = i * 4;
        let val = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let h = half::f16::from_f32(val);
        let dst_off = i * 2;
        out[dst_off..dst_off + 2].copy_from_slice(&h.to_le_bytes());
    }

    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "f16c")]
unsafe fn f32_to_f16_avx512(data: &[u8]) -> Vec<u8> {
    use std::arch::x86_64::*;

    let n = data.len() / 4;
    let mut out = vec![0u8; n * 2];
    let chunks = n / 16;
    let src = data.as_ptr() as *const f32;
    let dst = out.as_mut_ptr() as *mut u16;

    for i in 0..chunks {
        let offset = i * 16;
        unsafe {
            let v_f = _mm512_loadu_ps(src.add(offset));
            let v_h = _mm512_cvtps_ph(v_f, 0x00);
            _mm256_storeu_si256(dst.add(offset) as *mut __m256i, v_h);
        }
    }

    let tail_start = chunks * 16;
    for i in tail_start..n {
        let off = i * 4;
        let val = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let h = half::f16::from_f32(val);
        let dst_off = i * 2;
        out[dst_off..dst_off + 2].copy_from_slice(&h.to_le_bytes());
    }

    out
}

/// Convert f32 bytes to f64 bytes (scalar).
fn f32_to_f64(data: &[u8]) -> Vec<u8> {
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n * 8);
    for i in 0..n {
        let off = i * 4;
        let val = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        out.extend_from_slice(&(val as f64).to_le_bytes());
    }
    out
}

/// Convert f64 bytes to f32 bytes (scalar).
fn f64_to_f32(data: &[u8]) -> Vec<u8> {
    let n = data.len() / 8;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let off = i * 8;
        let val = f64::from_le_bytes([
            data[off], data[off + 1], data[off + 2], data[off + 3],
            data[off + 4], data[off + 5], data[off + 6], data[off + 7],
        ]);
        out.extend_from_slice(&(val as f32).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_f32_roundtrip() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0];
        let mut f16_bytes = Vec::new();
        for &v in &values {
            f16_bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }

        let f32_bytes = f16_to_f32(&f16_bytes);
        assert_eq!(f32_bytes.len(), values.len() * 4);

        for (i, &expected) in values.iter().enumerate() {
            let off = i * 4;
            let got = f32::from_le_bytes([
                f32_bytes[off], f32_bytes[off + 1], f32_bytes[off + 2], f32_bytes[off + 3],
            ]);
            assert!(
                (got - expected).abs() < 1e-3,
                "index {}: expected {}, got {}",
                i, expected, got
            );
        }

        // Round-trip back
        let back = f32_to_f16(&f32_bytes);
        assert_eq!(back, f16_bytes);
    }

    #[test]
    fn test_convert_elements_noop() {
        assert!(convert_elements(&[1, 2, 3, 4], 4, 4).is_none());
    }

    #[test]
    fn test_convert_elements_f16_to_f32() {
        let one = half::f16::from_f32(1.0);
        let data = one.to_le_bytes();
        let result = convert_elements(&data, 2, 4).unwrap();
        assert_eq!(result.len(), 4);
        let val = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f32_f64_roundtrip() {
        let values: Vec<f32> = vec![0.0, 1.0, -3.14, 1e10];
        let mut f32_bytes = Vec::new();
        for &v in &values {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let f64_bytes = f32_to_f64(&f32_bytes);
        let back = f64_to_f32(&f64_bytes);

        for (i, &expected) in values.iter().enumerate() {
            let off = i * 4;
            let got = f32::from_le_bytes([back[off], back[off + 1], back[off + 2], back[off + 3]]);
            assert_eq!(got, expected, "index {}", i);
        }
    }
}
