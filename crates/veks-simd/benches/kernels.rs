// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Kernel throughput at every supported level, at the production
//! reference width (dim = 384).
//!
//! ```text
//! cargo bench -p veks-simd --bench kernels
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use veks_simd::{BATCH_WIDTH, Kernels, Metric, PackedBatches, SimdLevel, TransposedBatch};

const DIM: usize = 384;
const BASE: usize = 4096;

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn levels() -> Vec<&'static Kernels> {
    SimdLevel::supported().into_iter().filter_map(Kernels::for_level).collect()
}

fn pairwise(c: &mut Criterion) {
    let base = data(DIM * BASE, 1);
    let query = data(DIM, 2);
    let base16: Vec<half::f16> = base.iter().map(|&v| half::f16::from_f32(v)).collect();
    let query16: Vec<half::f16> = query.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut g = c.benchmark_group("pairwise");
    g.throughput(Throughput::Elements(BASE as u64));
    for kernels in levels() {
        for metric in [Metric::L2, Metric::DotProduct, Metric::Cosine] {
            let f = kernels.distance_f32(metric);
            g.bench_with_input(BenchmarkId::new(format!("f32-{metric:?}"), kernels.level().name()), &(), |b, _| {
                b.iter(|| {
                    let mut s = 0.0f32;
                    for v in base.as_chunks::<DIM>().0 {
                        s += f(black_box(&query), v);
                    }
                    s
                })
            });
        }
        let f = kernels.distance_f16(Metric::L2);
        g.bench_with_input(BenchmarkId::new("f16-L2", kernels.level().name()), &(), |b, _| {
            b.iter(|| {
                let mut s = 0.0f32;
                for v in base16.as_chunks::<DIM>().0 {
                    s += f(black_box(&query16), v);
                }
                s
            })
        });
    }
    g.finish();
}

fn batched(c: &mut Criterion) {
    let base = data(DIM * BASE, 3);
    let qs: Vec<Vec<f32>> = (0..256).map(|i| data(DIM, 100 + i)).collect();
    let refs: Vec<&[f32]> = qs.iter().map(|q| q.as_slice()).collect();
    let single = TransposedBatch::from_f32(&refs[..BATCH_WIDTH], DIM);
    let other = TransposedBatch::from_f32(&refs[BATCH_WIDTH..2 * BATCH_WIDTH], DIM);
    let packed = PackedBatches::from_f32(&refs, DIM);
    let mut g = c.benchmark_group("batched");
    for kernels in levels() {
        let name = kernels.level().name();
        g.throughput(Throughput::Elements((BASE * BATCH_WIDTH) as u64));
        let f = kernels.batch_f32(Metric::L2);
        g.bench_with_input(BenchmarkId::new("batch16-L2", name), &(), |b, _| {
            let mut out = [0.0f32; BATCH_WIDTH];
            b.iter(|| {
                for v in base.as_chunks::<DIM>().0 {
                    f(&single, black_box(v), &mut out);
                }
                out[0]
            })
        });
        g.throughput(Throughput::Elements((BASE * 2 * BATCH_WIDTH) as u64));
        let f = kernels.dual_batch_f32(Metric::L2);
        g.bench_with_input(BenchmarkId::new("dual32-L2", name), &(), |b, _| {
            let mut out = [0.0f32; 2 * BATCH_WIDTH];
            b.iter(|| {
                for v in base.as_chunks::<DIM>().0 {
                    f(&single, &other, black_box(v), &mut out);
                }
                out[0]
            })
        });
        g.throughput(Throughput::Elements((BASE * 256) as u64));
        let f = kernels.packed_neg_dot_f32();
        g.bench_with_input(BenchmarkId::new("packed256-dot", name), &(), |b, _| {
            let mut out = vec![0.0f32; packed.n_batches() * BATCH_WIDTH];
            b.iter(|| {
                for v in base.as_chunks::<DIM>().0 {
                    f(black_box(v), &packed, &mut out);
                }
                out[0]
            })
        });
    }
    g.finish();
}

criterion_group!(benches, pairwise, batched);
criterion_main!(benches);
