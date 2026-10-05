// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Batch kernels (SRD acceptance case 2) and the lane-order invariant:
//! single, dual and packed kernels give bit-identical distances for the
//! same query lane.

mod common;

use common::{Rng, assert_close, reference};
use veks_simd::{BATCH_WIDTH, Kernels, Metric, PackedBatches, SimdLevel, TransposedBatch};

fn levels() -> Vec<&'static Kernels> {
    SimdLevel::supported().into_iter().filter_map(Kernels::for_level).collect()
}

fn queries(rng: &mut Rng, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| rng.vec_f32(dim)).collect()
}

#[test]
fn batch_kernels_match_reference_for_every_fill() {
    let mut rng = Rng::new(10);
    for kernels in levels() {
        for &dim in &[1usize, 5, 16, 17, 128, 384] {
            let base = rng.vec_f32(dim);
            let base64: Vec<f64> = base.iter().map(|&x| x as f64).collect();
            for count in [1usize, 7, 15, 16] {
                let qs = queries(&mut rng, count, dim);
                let refs: Vec<&[f32]> = qs.iter().map(|q| q.as_slice()).collect();
                let batch = TransposedBatch::from_f32(&refs, dim);
                for metric in Metric::ALL {
                    let mut out = [0.0f32; BATCH_WIDTH];
                    kernels.batch_f32(metric)(&batch, &base, &mut out);
                    for (qi, q) in qs.iter().enumerate() {
                        let q64: Vec<f64> = q.iter().map(|&x| x as f64).collect();
                        let what = format!("{} batch {metric:?} dim={dim} count={count} lane={qi}", kernels.level().name());
                        assert_close(&what, out[qi] as f64, reference(metric, &base64, &q64), 1e-5);
                    }
                }
            }
        }
    }
}

#[test]
fn single_and_dual_kernels_agree_bitwise() {
    let mut rng = Rng::new(11);
    for kernels in levels() {
        for &dim in &[3usize, 64, 384] {
            let base = rng.vec_f32(dim);
            let qa = queries(&mut rng, 16, dim);
            let qb = queries(&mut rng, 9, dim);
            let ra: Vec<&[f32]> = qa.iter().map(|q| q.as_slice()).collect();
            let rb: Vec<&[f32]> = qb.iter().map(|q| q.as_slice()).collect();
            let a = TransposedBatch::from_f32(&ra, dim);
            let b = TransposedBatch::from_f32(&rb, dim);
            for metric in Metric::ALL {
                let mut single_a = [0.0f32; BATCH_WIDTH];
                let mut single_b = [0.0f32; BATCH_WIDTH];
                let mut dual = [0.0f32; 2 * BATCH_WIDTH];
                kernels.batch_f32(metric)(&a, &base, &mut single_a);
                kernels.batch_f32(metric)(&b, &base, &mut single_b);
                kernels.dual_batch_f32(metric)(&a, &b, &base, &mut dual);
                for qi in 0..a.count() {
                    assert_eq!(single_a[qi].to_bits(), dual[qi].to_bits(), "{metric:?} lane a{qi}");
                }
                for qi in 0..b.count() {
                    assert_eq!(single_b[qi].to_bits(), dual[16 + qi].to_bits(), "{metric:?} lane b{qi}");
                }
            }
        }
    }
}

#[test]
fn packed_kernel_agrees_bitwise_with_single_batches() {
    let mut rng = Rng::new(12);
    for kernels in levels() {
        // 1..=20 sub-batches covers every compile-time group size and the
        // multi-group path on every level (largest group is 16).
        for n_queries in [1usize, 15, 16, 17, 40, 96, 161, 256, 320] {
            for &dim in &[5usize, 384] {
                let base = rng.vec_f32(dim);
                let qs = queries(&mut rng, n_queries, dim);
                let refs: Vec<&[f32]> = qs.iter().map(|q| q.as_slice()).collect();
                let packed = PackedBatches::from_f32(&refs, dim);
                let mut out = vec![f32::NAN; packed.n_batches() * BATCH_WIDTH];
                kernels.packed_neg_dot_f32()(&base, &packed, &mut out);
                for si in 0..packed.n_batches() {
                    let off = packed.offset(si);
                    let batch = TransposedBatch::from_f32(&refs[off..off + packed.count(si)], dim);
                    let mut single = [0.0f32; BATCH_WIDTH];
                    kernels.batch_f32(Metric::DotProduct)(&batch, &base, &mut single);
                    for qi in 0..packed.count(si) {
                        assert_eq!(
                            out[si * BATCH_WIDTH + qi].to_bits(),
                            single[qi].to_bits(),
                            "{} n={n_queries} dim={dim} sub-batch {si} lane {qi}",
                            kernels.level().name()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn exact_zero_dot_is_positive_zero() {
    // Stored distances must not flip to -0.0 for orthogonal pairs.
    for kernels in levels() {
        let base = [1.0f32, 0.0, 0.0, 0.0];
        let q = [0.0f32, 1.0, 0.0, 0.0];
        let batch = TransposedBatch::from_f32(&[&q], 4);
        let mut out = [f32::NAN; BATCH_WIDTH];
        kernels.batch_f32(Metric::DotProduct)(&batch, &base, &mut out);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        let packed = PackedBatches::from_f32(&[&q], 4);
        let mut pout = vec![f32::NAN; BATCH_WIDTH];
        kernels.packed_neg_dot_f32()(&base, &packed, &mut pout);
        assert_eq!(pout[0].to_bits(), 0.0f32.to_bits());
    }
}

#[test]
fn f16_batches_hold_widened_values() {
    let mut rng = Rng::new(13);
    let qs: Vec<Vec<half::f16>> = (0..5).map(|_| rng.vec_f16(33)).collect();
    let refs: Vec<&[half::f16]> = qs.iter().map(|q| q.as_slice()).collect();
    let batch = TransposedBatch::from_f16(&refs, 33);
    for (d, row) in batch.rows().iter().enumerate() {
        for (qi, q) in qs.iter().enumerate() {
            assert_eq!(row[qi], q[d].to_f32());
        }
        assert!(row[5..].iter().all(|&v| v == 0.0), "unused lanes are zero");
    }
    assert_eq!(batch.count(), 5);
    assert!(batch.query_norms()[5..].iter().all(|&n| n == 1.0));
}
