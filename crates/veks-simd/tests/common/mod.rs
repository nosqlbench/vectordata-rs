// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared fixtures: a deterministic generator and f64 reference kernels.

#![allow(dead_code)]

use veks_simd::Metric;

/// Dimensions every kernel is checked at: every length through two AVX-512
/// vectors plus one (all remainder shapes of every level), and the
/// production widths.
pub const DIMS: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 63, 64, 65, 127, 128, 129, 384, 768, 1536, 4096,
];

/// xorshift64*, for reproducible inputs without a dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    pub fn vec_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit() as f32).collect()
    }

    pub fn vec_f64(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.unit()).collect()
    }

    pub fn vec_f16(&mut self, n: usize) -> Vec<half::f16> {
        (0..n).map(|_| half::f16::from_f64(self.unit())).collect()
    }
}

/// The distance of `metric` in f64, plus a magnitude scale for tolerance:
/// the sum of absolute terms, so error bounds track cancellation.
pub fn reference(metric: Metric, a: &[f64], b: &[f64]) -> (f64, f64) {
    match metric {
        Metric::L2 => {
            let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            (s, s)
        }
        Metric::L1 => {
            let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
            (s, s)
        }
        Metric::DotProduct => {
            let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let scale: f64 = a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum();
            (-d, scale)
        }
        Metric::Cosine => {
            let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|x| x * x).sum();
            let nb: f64 = b.iter().map(|x| x * x).sum();
            let denom = (na * nb).sqrt();
            (if denom == 0.0 { 1.0 } else { 1.0 - d / denom }, 1.0)
        }
    }
}

/// Assert `got` matches the reference within `rel` of its scale.
pub fn assert_close(what: &str, got: f64, (want, scale): (f64, f64), rel: f64) {
    let tol = rel * scale.max(1e-30) + 1e-30;
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (diff {}, tol {tol})",
        (got - want).abs()
    );
}
