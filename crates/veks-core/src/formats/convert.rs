// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Element-type conversion for vector data (f16↔f32↔f64).
//!
//! When converting between xvec formats with different element sizes
//! (e.g. mvec→fvec), raw bytes must be re-interpreted and widened or
//! narrowed element-by-element. Record bytes are little-endian and carry
//! no alignment guarantee, so every conversion here works on byte slices.
//!
//! ## SIMD support
//!
//! f16↔f32 runs on `veks-simd`'s byte converters, dispatched once per
//! process to the CPU's best level: F16C (`vcvtph2ps` / `vcvtps2ph`) on
//! x86 with AVX2 or AVX-512, an exact integer widening plus `half`'s
//! narrowing elsewhere (NEON, SSE). Results are bit-identical with the
//! `half` crate at every level. f32↔f64 is a plain cast per element.

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
/// - 2→8 (f16→f64): via f32
/// - 8→2 (f64→f16): via f32
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

fn kernels() -> &'static veks_simd::Kernels {
    veks_simd::Kernels::detected()
}

/// Convert f16 bytes to f32 bytes into a pre-allocated buffer.
fn f16_to_f32_into(data: &[u8], out: &mut [u8]) -> usize {
    kernels().f16_bytes_to_f32_bytes()(data, out)
}

/// Convert f32 bytes to f16 bytes into a pre-allocated buffer.
fn f32_to_f16_into(data: &[u8], out: &mut [u8]) -> usize {
    kernels().f32_bytes_to_f16_bytes()(data, out)
}

fn f32_to_f64_into(data: &[u8], out: &mut [u8]) -> usize {
    let n = data.len() / 4;
    for (src, dst) in data.as_chunks::<4>().0.iter().zip(out[..n * 8].as_chunks_mut::<8>().0) {
        *dst = (f32::from_le_bytes(*src) as f64).to_le_bytes();
    }
    n * 8
}

fn f64_to_f32_into(data: &[u8], out: &mut [u8]) -> usize {
    let n = data.len() / 8;
    for (src, dst) in data.as_chunks::<8>().0.iter().zip(out[..n * 4].as_chunks_mut::<4>().0) {
        *dst = (f64::from_le_bytes(*src) as f32).to_le_bytes();
    }
    n * 4
}

/// Convert f16 bytes to f32 bytes.
fn f16_to_f32(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() / 2 * 4];
    f16_to_f32_into(data, &mut out);
    out
}

/// Convert f32 bytes to f16 bytes.
fn f32_to_f16(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() / 4 * 2];
    f32_to_f16_into(data, &mut out);
    out
}

/// Convert f32 bytes to f64 bytes.
fn f32_to_f64(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() / 4 * 8];
    f32_to_f64_into(data, &mut out);
    out
}

/// Convert f64 bytes to f32 bytes.
fn f64_to_f32(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; data.len() / 8 * 4];
    f64_to_f32_into(data, &mut out);
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
        let values: Vec<f32> = vec![0.0, 1.0, -3.25, 1e10];
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

    #[test]
    fn test_convert_elements_into_reports_bytes_written() {
        let src: Vec<u8> = (0..37u16).flat_map(|i| half::f16::from_f32(i as f32 * 0.25).to_le_bytes()).collect();
        let mut out = vec![0u8; 37 * 4];
        assert_eq!(convert_elements_into(&src, 2, 4, &mut out), Some(37 * 4));
        let mut back = vec![0u8; 37 * 2];
        assert_eq!(convert_elements_into(&out, 4, 2, &mut back), Some(37 * 2));
        assert_eq!(back, src);
        let mut wide = vec![0u8; 37 * 8];
        assert_eq!(convert_elements_into(&out, 4, 8, &mut wide), Some(37 * 8));
        assert_eq!(convert_elements_into(&src, 2, 8, &mut wide), None, "2→8 has no in-place form");
    }
}
