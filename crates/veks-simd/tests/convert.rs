// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! f16 ↔ f32 conversion, bit-exact with the `half` crate (SRD
//! acceptance case 4), at every dispatch level.

use veks_simd::{Kernels, SimdLevel};

fn levels() -> Vec<&'static Kernels> {
    SimdLevel::supported().into_iter().filter_map(Kernels::for_level).collect()
}

fn all_f16() -> Vec<half::f16> {
    (0..=u16::MAX).map(half::f16::from_bits).collect()
}

#[test]
fn f16_to_f32_is_exact_for_every_bit_pattern() {
    let src = all_f16();
    for kernels in levels() {
        let mut dst = vec![0.0f32; src.len()];
        kernels.f16_to_f32()(&src, &mut dst);
        for (h, f) in src.iter().zip(&dst) {
            assert_eq!(f.to_bits(), h.to_f32().to_bits(), "{} {:#06x}", kernels.level().name(), h.to_bits());
        }
    }
}

#[test]
fn f16_bytes_to_f32_bytes_is_exact_and_alignment_free() {
    let src = all_f16();
    let mut bytes: Vec<u8> = vec![0xAA];
    for h in &src {
        bytes.extend_from_slice(&h.to_le_bytes());
    }
    // Offset by one byte: the input is deliberately misaligned.
    let input = &bytes[1..];
    for kernels in levels() {
        let mut out = vec![0u8; input.len() * 2 + 1];
        let written = kernels.f16_bytes_to_f32_bytes()(input, &mut out[1..]);
        assert_eq!(written, src.len() * 4);
        for (i, h) in src.iter().enumerate() {
            let got = u32::from_le_bytes(out[1 + 4 * i..5 + 4 * i].try_into().unwrap());
            assert_eq!(got, h.to_f32().to_bits(), "{} {:#06x}", kernels.level().name(), h.to_bits());
        }
    }
}

/// Every f16 value round-tripped, the midpoints between neighbouring
/// f16 values (where nearest-even rounding decides), and the edges.
fn narrowing_inputs() -> Vec<f32> {
    let mut v: Vec<f32> = Vec::new();
    let finite: Vec<f32> = all_f16().iter().map(|h| h.to_f32()).filter(|f| f.is_finite()).collect();
    v.extend(&finite);
    let mut sorted = finite.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup_by(|a, b| a.to_bits() == b.to_bits());
    for w in sorted.windows(2) {
        let mid = ((w[0] as f64 + w[1] as f64) / 2.0) as f32;
        v.push(mid);
        v.push(f32::from_bits(mid.to_bits().wrapping_add(1)));
        v.push(f32::from_bits(mid.to_bits().wrapping_sub(1)));
    }
    v.extend([
        f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN,
        f32::from_bits(0x7f80_0001), f32::from_bits(0x7fbf_ffff),
        f32::MAX, f32::MIN, 65504.0, 65519.99, 65520.0, 1e-8, -1e-8,
        f32::MIN_POSITIVE, f32::from_bits(1), 0.0, -0.0,
    ]);
    v
}

#[test]
fn f32_to_f16_rounds_like_half() {
    let src = narrowing_inputs();
    for kernels in levels() {
        let mut dst = vec![half::f16::ZERO; src.len()];
        kernels.f32_to_f16()(&src, &mut dst);
        for (f, h) in src.iter().zip(&dst) {
            let want = half::f16::from_f32(*f);
            if want.is_nan() {
                assert!(h.is_nan(), "{} {:#010x} must stay NaN", kernels.level().name(), f.to_bits());
            } else {
                assert_eq!(h.to_bits(), want.to_bits(), "{} {:#010x}", kernels.level().name(), f.to_bits());
            }
        }
    }
}

#[test]
fn f32_bytes_to_f16_bytes_rounds_like_half() {
    let src = narrowing_inputs();
    let mut bytes = vec![0x55u8];
    for f in &src {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    let input = &bytes[1..];
    for kernels in levels() {
        let mut out = vec![0u8; src.len() * 2];
        assert_eq!(kernels.f32_bytes_to_f16_bytes()(input, &mut out), src.len() * 2);
        for (i, f) in src.iter().enumerate() {
            let got = half::f16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
            let want = half::f16::from_f32(*f);
            if want.is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{} {:#010x}", kernels.level().name(), f.to_bits());
            }
        }
    }
}
