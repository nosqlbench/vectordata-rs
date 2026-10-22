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
//!
//! The crate-boundaries SRD (PR-10) adds the paths its moves touch:
//! reads through a dataset view (`view`), the streaming and typed
//! writers (`stream_write`), a whole fetch over loopback into a fresh
//! cache (`remote_fetch`), and reads from that cache (`remote_cached`).
//! Each runs before and after every step that touches a hot path.

use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use vectordata::io::{IndexedVvecReader, ScalarReader, VectorReader, VvecReader, XvecReader};
use vectordata::fetch::{FetchEvent, FetchRequest};
use vectordata::{TestDataGroup, TestDataViewExt};

#[allow(dead_code)]
#[path = "../tests/support/testserver.rs"]
mod testserver;

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
            while let Some(record) = source.next_record().unwrap() {
                black_box(&record);
            }
        })
    });

    group.bench_function(BenchmarkId::new(ty, "veks-io-stream-into"), |b| {
        let mut record = Vec::new();
        b.iter(|| {
            let mut source = veks_io::open(&path).unwrap();
            while source.read_record_into(&mut record).unwrap() {
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
            while let Some(record) = source.next_record().unwrap() {
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
    // A closure, not a `fn` pointer: the sum inlines the widening, so
    // the case measures the borrow and one pass over the column rather
    // than an indirect call per element, whose cost moved with where
    // the linker happened to place the loop (srd-crate-boundaries.md
    // §12.1).
    widen: impl Fn(T) -> i64 + Copy,
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

/// A dataset directory over the sequential f32 file: a `dataset.yaml`
/// naming it, so the reads below go through the dataset API.
fn view_dataset() -> (PathBuf, usize) {
    let (path, count) = uniform("seq.fvecs", 128, 4, f32_value);
    let dir = data_dir().join("view");
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("base.fvecs");
    if !base.exists() {
        std::fs::hard_link(&path, &base).unwrap();
    }
    std::fs::write(dir.join("dataset.yaml"), "name: view\nprofiles:\n  default:\n    base_vectors: base.fvecs\n").unwrap();
    (dir, count)
}

/// Reads through `Arc<dyn TestDataView>` and the reader it hands out:
/// the dispatch every dataset consumer pays.
fn view(c: &mut Criterion) {
    let (dir, count) = view_dataset();
    let view = TestDataGroup::load(dir.to_str().unwrap()).unwrap().profile("default").unwrap();
    let base = view.base_vectors_as::<f32>().unwrap();
    let dim = base.dim();
    let mut group = c.benchmark_group("view");
    group.sample_size(10).measurement_time(Duration::from_secs(20));

    let mut buf = vec![0f32; BATCH * dim];
    group.throughput(Throughput::Bytes((count * (4 + dim * 4)) as u64));
    group.bench_function("seq_read_into", |b| {
        b.iter(|| {
            let mut start = 0;
            while start < count {
                start += base.read_into(start, &mut buf).unwrap();
                black_box(&buf);
            }
        })
    });

    let indices = random_indices(count);
    group.throughput(Throughput::Elements(RANDOM_READS as u64));
    group.bench_function("random_get", |b| {
        b.iter(|| {
            for &i in &indices {
                black_box(base.get(i).unwrap());
            }
        })
    });
    group.bench_function("open", |b| {
        b.iter(|| black_box(view.base_vectors_as::<f32>().unwrap()))
    });
    group.finish();
}

/// Records written through veks-io's streaming sink and the typed
/// writer, 256 MiB of dim-128 f32 each, to a file rewritten in place.
fn stream_write(c: &mut Criterion) {
    let dim = 128;
    let count = (256usize << 20) / (4 + dim * 4);
    let mut rng = Lcg(7);
    let row: Vec<f32> = (0..dim).map(|_| f32::from_le_bytes(f32_value(&mut rng).try_into().unwrap())).collect();
    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = data_dir().join("write.fvecs");
    let mut group = c.benchmark_group("stream_write");
    group.sample_size(10).measurement_time(Duration::from_secs(20));
    group.throughput(Throughput::Bytes((count * (4 + dim * 4)) as u64));
    group.bench_function("sink_write_record", |b| {
        b.iter(|| {
            let mut sink = veks_io::create(&out, dim as u32).unwrap();
            for i in 0..count {
                sink.write_record(i as i64, &bytes);
            }
            sink.finish().unwrap();
        })
    });
    group.bench_function("typed_xvec_writer", |b| {
        b.iter(|| {
            let mut w = vectordata::io::XvecWriter::<f32>::create(&out, dim).unwrap();
            for _ in 0..count {
                w.write(&row).unwrap();
            }
            w.finish().unwrap();
        })
    });
    group.finish();
}

/// The remote dataset the two remote groups read: 64 MiB of dim-128 f32
/// served over loopback.
fn remote_dataset() -> (PathBuf, usize) {
    let (path, count) = uniform_sized("remote.fvecs", 128, 64 << 20);
    let root = data_dir().join("served");
    let dir = root.join("remote");
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("base.fvecs");
    if !base.exists() {
        std::fs::hard_link(&path, &base).unwrap();
    }
    std::fs::write(dir.join("dataset.yaml"), "name: remote\nprofiles:\n  default:\n    base_vectors: base.fvecs\n").unwrap();
    (root, count)
}

/// A uniform f32 file of about `bytes`, written once.
fn uniform_sized(name: &str, dim: usize, bytes: usize) -> (PathBuf, usize) {
    let path = data_dir().join(name);
    let count = bytes / (4 + dim * 4);
    ensure(&path, (count * (4 + dim * 4)) as u64, |w| {
        let mut rng = Lcg(11);
        for _ in 0..count {
            w.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for _ in 0..dim {
                w.write_all(&f32_value(&mut rng)).unwrap();
            }
        }
    });
    (path, count)
}

/// The cache the remote groups fetch into, one per bench process.
fn bench_cache() -> PathBuf {
    let dir = data_dir().join(format!("cache-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    vectordata::settings::override_cache_dir_for_process(dir.clone());
    dir
}

/// A whole facet fetched over loopback into a fresh cache directory,
/// each iteration: the chunk store's transfer, verification-free path
/// (no `.mref` is published), sidecar writes included.
fn remote_fetch(c: &mut Criterion) {
    let (root, _) = remote_dataset();
    let cache = bench_cache();
    let server = testserver::TestServer::start(&root).unwrap();
    let mut group = c.benchmark_group("remote_fetch");
    group.sample_size(10).measurement_time(Duration::from_secs(30));
    group.throughput(Throughput::Bytes(std::fs::metadata(root.join("remote/base.fvecs")).unwrap().len()));
    let mut round = 0u64;
    group.bench_function("whole_facet", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                round += 1;
                // A fresh dataset name is a fresh cache directory.
                let name = format!("remote-{round}");
                let served = root.join(&name);
                if !served.exists() {
                    std::os::unix::fs::symlink(root.join("remote"), &served).unwrap();
                }
                let group = TestDataGroup::load(&format!("{}/{name}/", server.base_url())).unwrap();
                let view = group.profile("default").unwrap();
                let start = std::time::Instant::now();
                view.fetch(&FetchRequest::all(), &mut |_: &FetchEvent<'_>| {}).unwrap();
                total += start.elapsed();
                drop(view);
                let _ = std::fs::remove_dir_all(cache.join(&name));
                let _ = std::fs::remove_file(&served);
            }
            total
        })
    });
    group.finish();
}

/// Reads of a remote facet once it is all in the cache: the promoted
/// mapping under the chunk store.
fn remote_cached(c: &mut Criterion) {
    let (root, count) = remote_dataset();
    let _cache = bench_cache();
    let server = testserver::TestServer::start(&root).unwrap();
    let group_ = TestDataGroup::load(&format!("{}/remote/", server.base_url())).unwrap();
    let view = group_.profile("default").unwrap();
    view.fetch(&FetchRequest::all(), &mut |_: &FetchEvent<'_>| {}).unwrap();
    let base = view.base_vectors_as::<f32>().unwrap();
    let dim = base.dim();
    let mut buf = vec![0f32; BATCH * dim];
    let mut group = c.benchmark_group("remote_cached");
    group.sample_size(10).measurement_time(Duration::from_secs(15));
    group.throughput(Throughput::Bytes((count * (4 + dim * 4)) as u64));
    group.bench_function("seq_read_into", |b| {
        b.iter(|| {
            let mut start = 0;
            while start < count {
                start += base.read_into(start, &mut buf).unwrap();
                black_box(&buf);
            }
        })
    });
    let indices: Vec<usize> = random_indices(count).into_iter().take(200_000).collect();
    group.throughput(Throughput::Elements(indices.len() as u64));
    group.bench_function("random_get", |b| {
        b.iter(|| {
            for &i in &indices {
                black_box(base.get(i).unwrap());
            }
        })
    });
    group.finish();
}

criterion_group!(benches, sequential, random, vvec, scalar, view, stream_write, remote_fetch, remote_cached);
criterion_main!(benches);
