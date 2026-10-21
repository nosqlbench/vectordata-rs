// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The readers over vectordata's storage against the same readers over
//! a local mapping, on the same local files
//! (`docs/design/srd-loose-files-overlays-integrity.md`, IO-4).
//!
//! veks-io is the one implementation of every file format; vectordata's
//! readers are veks-io's over its own source. Before the move the two
//! were separate implementations, measured by this bench (SRD §6.2);
//! after it, no case may be slower than the better of the two before.
//! Each benchmark id is `<case>/<impl>`: `vectordata` (over its
//! storage) or `veks-io` (over a mapped file), and `veks-io-stream` for
//! veks-io's streaming reader.
//!
//! The files are written once under `target/bench-data/readers/` and
//! reused while their size matches. Reads are from a warm page cache:
//! this measures the readers, not the disk. Pin to one core so the
//! runs are comparable:
//!
//! ```text
//! taskset -c 2 cargo bench -p vectordata --bench readers
//! ```
//!
//! `READERS_BENCH_MIB` sets each uniform file's size (default 1024).

use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use vectordata::io::{IndexedVvecReader, ScalarReader, VectorReader, VvecReader, XvecReader};

/// Records per bulk read: a reused buffer of this many records.
const BATCH: usize = 4096;
/// Random lookups per iteration.
const RANDOM_READS: usize = 1_000_000;
/// Records in the variable-length file.
const VVEC_RECORDS: usize = 1_000_000;

fn data_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/bench-data/readers");
    std::fs::create_dir_all(&dir).expect("create bench data dir");
    dir
}

fn file_mib() -> u64 {
    std::env::var("READERS_BENCH_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024)
}

/// A deterministic stream of values: the same files on every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Write `path` unless a file of exactly `len` bytes is already there.
fn ensure(path: &Path, len: u64, write: impl FnOnce(&mut BufWriter<File>)) {
    if std::fs::metadata(path).is_ok_and(|m| m.len() == len) {
        return;
    }
    eprintln!("writing {} ({} MiB)", path.display(), len >> 20);
    let mut out = BufWriter::with_capacity(16 << 20, File::create(path).expect("create bench file"));
    write(&mut out);
    out.flush().expect("flush bench file");
    drop(out);
    assert_eq!(std::fs::metadata(path).unwrap().len(), len, "{}", path.display());
}

/// A uniform xvec of `dim`-element records of `width`-byte elements,
/// `file_mib()` long, rounded down to whole records.
fn uniform(name: &str, dim: usize, width: usize, value: impl Fn(&mut Lcg) -> Vec<u8>) -> (PathBuf, usize) {
    let stride = 4 + dim * width;
    let count = (file_mib() << 20) as usize / stride;
    let path = data_dir().join(name);
    ensure(&path, (count * stride) as u64, |out| {
        let mut rng = Lcg(7);
        for _ in 0..count {
            out.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for _ in 0..dim {
                out.write_all(&value(&mut rng)).unwrap();
            }
        }
    });
    (path, count)
}

/// A header-free column of `count` `width`-byte values.
fn column(name: &str, count: usize, width: usize, value: impl Fn(&mut Lcg) -> Vec<u8>) -> PathBuf {
    let path = data_dir().join(name);
    ensure(&path, (count * width) as u64, |out| {
        let mut rng = Lcg(11);
        for _ in 0..count {
            out.write_all(&value(&mut rng)).unwrap();
        }
    });
    path
}

/// A variable-length i32 xvec: dims 1..=64.
fn ragged(name: &str) -> PathBuf {
    let mut rng = Lcg(13);
    let dims: Vec<usize> = (0..VVEC_RECORDS).map(|_| 1 + (rng.next() % 64) as usize).collect();
    let len: usize = dims.iter().map(|d| 4 + d * 4).sum();
    let path = data_dir().join(name);
    ensure(&path, len as u64, |out| {
        let mut rng = Lcg(17);
        for &d in &dims {
            out.write_all(&(d as i32).to_le_bytes()).unwrap();
            for _ in 0..d {
                out.write_all(&(rng.next() as i32).to_le_bytes()).unwrap();
            }
        }
    });
    path
}

fn f32_value(rng: &mut Lcg) -> Vec<u8> {
    ((rng.next() % 2001) as f32 / 1000.0 - 1.0).to_le_bytes().to_vec()
}
fn u8_value(rng: &mut Lcg) -> Vec<u8> {
    vec![rng.next() as u8]
}
fn i32_value(rng: &mut Lcg) -> Vec<u8> {
    (rng.next() as i32).to_le_bytes().to_vec()
}

/// Indices into `0..count`, the same on every run.
fn random_indices(count: usize) -> Vec<usize> {
    let mut rng = Lcg(23);
    (0..RANDOM_READS).map(|_| rng.next() as usize % count).collect()
}

/// Whole-file bulk reads into a reused buffer, per element type.
fn sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_read_into");
    group.sample_size(10).measurement_time(Duration::from_secs(20));
    seq_case::<f32>(&mut group, "f32", uniform("seq.fvecs", 128, 4, f32_value));
    seq_case::<u8>(&mut group, "u8", uniform("seq.bvecs", 128, 1, u8_value));
    seq_case::<i32>(&mut group, "i32", uniform("seq.ivecs", 100, 4, i32_value));
    group.finish();
}

fn seq_case<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ty: &str,
    (path, count): (PathBuf, usize),
) where
    T: vectordata::Element + Default + Copy + veks_io::Element,
{
    group.throughput(Throughput::Bytes(std::fs::metadata(&path).unwrap().len()));

    let vd = XvecReader::<T>::open_path(&path).unwrap();
    let dim = vd.dim();
    let mut buf = vec![T::default(); BATCH * dim];
    group.bench_function(BenchmarkId::new(ty, "vectordata"), |b| {
        b.iter(|| {
            let mut start = 0;
            while start < count {
                start += VectorReader::read_into(&vd, start, &mut buf).unwrap();
                black_box(&buf);
            }
        })
    });

    let vio = veks_io::XvecReader::<T>::open_path(&path).unwrap();
    group.bench_function(BenchmarkId::new(ty, "veks-io"), |b| {
        b.iter(|| {
            let mut start = 0;
            while start < count {
                start += VectorReader::read_into(&vio, start, &mut buf).unwrap();
                black_box(&buf);
            }
        })
    });

    group.bench_function(BenchmarkId::new(ty, "veks-io-stream"), |b| {
        b.iter(|| {
            let mut source = veks_io::open(&path).unwrap();
            while let Some(record) = source.next_record() {
                black_box(&record);
            }
        })
    });

    group.bench_function(BenchmarkId::new(ty, "veks-io-stream-into"), |b| {
        let mut record = Vec::new();
        b.iter(|| {
            let mut source = veks_io::open(&path).unwrap();
            while source.read_record_into(&mut record) {
                black_box(&record);
            }
        })
    });
}

/// One million random record reads from the f32 file: owned (`get`)
/// and borrowed (`get_slice`).
fn random(c: &mut Criterion) {
    let (path, count) = uniform("seq.fvecs", 128, 4, f32_value);
    let indices = random_indices(count);
    let mut group = c.benchmark_group("random_f32");
    group.sample_size(10).measurement_time(Duration::from_secs(15));
    group.throughput(Throughput::Elements(RANDOM_READS as u64));

    let vd = XvecReader::<f32>::open_path(&path).unwrap();
    let vio = veks_io::XvecReader::<f32>::open_path(&path).unwrap();
    group.bench_function(BenchmarkId::new("get", "vectordata"), |b| {
        b.iter(|| {
            for &i in &indices {
                black_box(VectorReader::get(&vd, i).unwrap());
            }
        })
    });
    group.bench_function(BenchmarkId::new("get", "veks-io"), |b| {
        b.iter(|| {
            for &i in &indices {
                black_box(VectorReader::get(&vio, i).unwrap());
            }
        })
    });
    group.bench_function(BenchmarkId::new("get_slice", "vectordata"), |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &i in &indices {
                sum += VectorReader::get_slice(&vd, i).unwrap()[0];
            }
            black_box(sum)
        })
    });
    group.bench_function(BenchmarkId::new("get_slice", "veks-io"), |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &i in &indices {
                sum += VectorReader::get_slice(&vio, i).unwrap()[0];
            }
            black_box(sum)
        })
    });
    group.finish();
}

/// A variable-length file read start to end, and at random (vectordata
/// only: veks-io has no indexed variable-length reader).
fn vvec(c: &mut Criterion) {
    let path = ragged("ragged.ivecs");
    let mut group = c.benchmark_group("vvec_i32");
    group.sample_size(10).measurement_time(Duration::from_secs(15));
    group.throughput(Throughput::Bytes(std::fs::metadata(&path).unwrap().len()));

    let vd = IndexedVvecReader::<i32>::open_path(&path).unwrap();
    group.bench_function(BenchmarkId::new("scan", "vectordata"), |b| {
        b.iter(|| {
            for i in 0..vd.count() {
                black_box(VvecReader::get(&vd, i).unwrap());
            }
        })
    });
    group.bench_function(BenchmarkId::new("scan", "veks-io-stream"), |b| {
        b.iter(|| {
            let mut source = veks_io::open_varlen(&path).unwrap();
            while let Some(record) = source.next_record() {
                black_box(&record);
            }
        })
    });
    let indices = random_indices(vd.count());
    group.throughput(Throughput::Elements(RANDOM_READS as u64));
    group.bench_function(BenchmarkId::new("random_get", "vectordata"), |b| {
        b.iter(|| {
            for &i in &indices {
                black_box(VvecReader::get(&vd, i).unwrap());
            }
        })
    });
    group.finish();
}

/// A header-free column: summed through `as_slice`, and copied out in
/// batches through `read_into`.
fn scalar(c: &mut Criterion) {
    let count = (file_mib() << 20) as usize / 4;
    let mut group = c.benchmark_group("scalar");
    group.sample_size(10).measurement_time(Duration::from_secs(15));
    scalar_case::<i32>(&mut group, "i32", column("col.i32", count, 4, i32_value), |v| v as i64);
    scalar_case::<f32>(&mut group, "f32", column("col.f32", count, 4, f32_value), |v| v as i64);
    scalar_case::<u8>(&mut group, "u8", column("col.u8", count * 4, 1, u8_value), |v| v as i64);
    group.finish();
}

fn scalar_case<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ty: &str,
    path: PathBuf,
    widen: fn(T) -> i64,
) where
    T: vectordata::Element + Default + Copy + veks_io::Element,
{
    group.throughput(Throughput::Bytes(std::fs::metadata(&path).unwrap().len()));
    let vd = ScalarReader::<T>::open_path(&path).unwrap();
    let vio = veks_io::ScalarReader::<T>::open_path(&path).unwrap();
    let count = vd.count();

    group.bench_function(BenchmarkId::new(format!("{ty}/as_slice_sum"), "vectordata"), |b| {
        b.iter(|| black_box(VectorReader::as_slice(&vd).unwrap().iter().map(|&v| widen(v)).sum::<i64>()))
    });
    group.bench_function(BenchmarkId::new(format!("{ty}/as_slice_sum"), "veks-io"), |b| {
        b.iter(|| black_box(VectorReader::as_slice(&vio).unwrap().iter().map(|&v| widen(v)).sum::<i64>()))
    });

    let mut buf = vec![T::default(); BATCH * 64];
    group.bench_function(BenchmarkId::new(format!("{ty}/read_into"), "vectordata"), |b| {
        b.iter(|| {
            let mut start = 0;
            while start < count {
                start += VectorReader::read_into(&vd, start, &mut buf).unwrap();
                black_box(&buf);
            }
        })
    });
    group.bench_function(BenchmarkId::new(format!("{ty}/read_into"), "veks-io"), |b| {
        b.iter(|| {
            let mut start = 0;
            while start < count {
                start += VectorReader::read_into(&vio, start, &mut buf).unwrap();
                black_box(&buf);
            }
        })
    });
}

criterion_group!(benches, sequential, random, vvec, scalar);
criterion_main!(benches);
