// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pairwise kernels against an f64 reference, at every dispatch level
//! this CPU supports (SRD acceptance cases 1 and 5).

mod common;

use common::{DIMS, Rng, assert_close, reference};
use veks_simd::{Kernels, Metric, SimdLevel};

/// f32 accumulation over up to 4096 terms in [-1, 1]: the error grows
/// with the term count, and this bound holds the worst observed case
/// with an order of magnitude to spare.
const F32_REL: f64 = 1e-5;
const F64_REL: f64 = 1e-12;

fn levels() -> Vec<&'static Kernels> {
    SimdLevel::supported()
        .into_iter()
        .map(|l| Kernels::for_level(l).expect("a supported level has a table"))
        .collect()
}

#[test]
fn f32_kernels_match_reference_at_every_level() {
    let mut rng = Rng::new(1);
    for kernels in levels() {
        for &dim in DIMS {
            let a = rng.vec_f32(dim);
            let b = rng.vec_f32(dim);
            let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
            let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
            for metric in Metric::ALL {
                let got = kernels.distance_f32(metric)(&a, &b) as f64;
                let what = format!("{} f32 {metric:?} dim={dim}", kernels.level().name());
                assert_close(&what, got, reference(metric, &a64, &b64), F32_REL);
            }
        }
    }
}

#[test]
fn f16_kernels_match_reference_at_every_level() {
    let mut rng = Rng::new(2);
    for kernels in levels() {
        for &dim in DIMS {
            let a = rng.vec_f16(dim);
            let b = rng.vec_f16(dim);
            // The reference sees exactly the f16 values the kernel sees.
            let a64: Vec<f64> = a.iter().map(|x| x.to_f64()).collect();
            let b64: Vec<f64> = b.iter().map(|x| x.to_f64()).collect();
            for metric in Metric::ALL {
                let got = kernels.distance_f16(metric)(&a, &b) as f64;
                let what = format!("{} f16 {metric:?} dim={dim}", kernels.level().name());
                assert_close(&what, got, reference(metric, &a64, &b64), F32_REL);
            }
        }
    }
}

#[test]
fn f64_kernels_match_reference_at_every_level() {
    let mut rng = Rng::new(3);
    for kernels in levels() {
        for &dim in DIMS {
            let a = rng.vec_f64(dim);
            let b = rng.vec_f64(dim);
            for metric in Metric::ALL {
                // f64 kernels return f32, so the comparison is at f32
                // precision on an f64-accurate value.
                let got = kernels.distance_f64(metric)(&a, &b) as f64;
                let (want, scale) = reference(metric, &a, &b);
                let what = format!("{} f64 {metric:?} dim={dim}", kernels.level().name());
                assert_close(&what, got, (want, scale.max(want.abs())), 1e-6_f64.max(F64_REL));
            }
        }
    }
}

#[test]
fn dot_and_norm_match_reference() {
    let mut rng = Rng::new(4);
    for kernels in levels() {
        for &dim in DIMS {
            let a = rng.vec_f32(dim);
            let b = rng.vec_f32(dim);
            let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
            let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
            let (neg, scale) = reference(Metric::DotProduct, &a64, &b64);
            assert_close("dot", kernels.dot_f32()(&a, &b) as f64, (-neg, scale), F32_REL);
            let norm: f64 = a64.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert_close("norm", kernels.norm_f32()(&a) as f64, (norm, norm), F32_REL);
        }
    }
}

#[test]
fn zero_vectors_have_unit_cosine_distance() {
    for kernels in levels() {
        let z = vec![0.0f32; 37];
        let v = vec![1.0f32; 37];
        assert_eq!(kernels.distance_f32(Metric::Cosine)(&z, &v), 1.0);
        assert_eq!(kernels.distance_f32(Metric::Cosine)(&z, &z), 1.0);
        let zh = vec![half::f16::ZERO; 37];
        assert_eq!(kernels.distance_f16(Metric::Cosine)(&zh, &zh), 1.0);
        let zd = vec![0.0f64; 37];
        assert_eq!(kernels.distance_f64(Metric::Cosine)(&zd, &zd), 1.0);
    }
}

#[test]
fn levels_agree_with_each_other() {
    // Forcing each level gives results within tolerance of the detected
    // one: a level-specific kernel bug cannot hide behind the reference
    // tolerance being loose in one direction.
    let mut rng = Rng::new(5);
    let detected = Kernels::detected();
    for &dim in &[7usize, 128, 384, 1000] {
        let a = rng.vec_f32(dim);
        let b = rng.vec_f32(dim);
        let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        for metric in Metric::ALL {
            let want = detected.distance_f32(metric)(&a, &b) as f64;
            let (_, scale) = reference(metric, &a64, &b64);
            for kernels in levels() {
                let got = kernels.distance_f32(metric)(&a, &b) as f64;
                let what = format!("{} vs {} {metric:?} dim={dim}", kernels.level().name(), detected.level().name());
                assert_close(&what, got, (want, scale), F32_REL);
            }
        }
    }
}
