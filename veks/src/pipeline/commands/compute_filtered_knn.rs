// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute predicate-filtered exact K-nearest neighbors.
//!
//! A variant of `compute knn` that restricts each query's candidate set to
//! the base vectors matching a pre-computed predicate. The predicate answer
//! keys are stored in a slab file produced by `generate predicate-keys`:
//! each slab record `i` is a packed `[i32 LE]*` array of base ordinals that
//! satisfy predicate `i`.
//!
//! For query `i`, only the base vectors whose ordinals appear in the
//! predicate-keys record `i` are considered. This models pre-filtered
//! vector search where metadata predicates narrow the candidate set
//! before distance computation.
//!
//! ## Partitioned computation
//!
//! Uses the same base-vector partitioning strategy as `compute knn`:
//!
//! 1. The base vector space is split into contiguous partitions.
//! 2. Each partition is processed sequentially: for every query, the
//!    predicate-key ordinals that fall within the partition's range are
//!    filtered and distance-computed against the mmap'd base vectors.
//! 3. A prefetch thread pages in upcoming partitions for sequential I/O.
//! 4. A background writer flushes completed partition results to cache
//!    while the next partition is being computed.
//! 5. Completed partition pages are released from the page cache.
//! 6. A cross-partition merge combines per-partition top-K results into
//!    the global top-K for each query.
//!
//! This gives the same sequential I/O, bounded memory, and resumability
//! characteristics as unfiltered KNN.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{BufReader, Read as _, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt};
use half;
use crate::ui::{ProgressHandle, UiHandle};
use slabtastic::SlabReader;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::simd_distance::{self, Metric};

/// Pipeline command: compute predicate-filtered exact KNN.
pub struct ComputeFilteredKnnOp;

/// Create a boxed `ComputeFilteredKnnOp` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeFilteredKnnOp)
}

// -- Top-K heap ---------------------------------------------------------------

/// A neighbor candidate with index and distance (max-heap by distance).
#[derive(Clone)]
struct Neighbor {
    index: u32,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Read a predicate-keys slab record as a Vec of i32 base ordinals.
fn read_ordinals(data: &[u8]) -> Vec<i32> {
    let mut cursor = std::io::Cursor::new(data);
    let mut result = Vec::with_capacity(data.len() / 4);
    while let Ok(v) = cursor.read_i32::<LittleEndian>() {
        result.push(v);
    }
    result
}

// -- Partition infrastructure -------------------------------------------------

/// Metadata for a single partition of the base vector space.
struct PartitionMeta {
    start: usize,
    end: usize,
    neighbors_path: PathBuf,
    distances_path: PathBuf,
    cached: bool,
}

/// Build a cache file path for a partition.
fn build_cache_path(
    cache_dir: &Path,
    step_id: &str,
    start: usize,
    end: usize,
    k: usize,
    metric: Metric,
    suffix: &str,
    ext: &str,
) -> PathBuf {
    let metric_str = metric_label(metric);
    cache_dir.join(format!(
        "{}.range_{:06}_{:06}.k{}.{}.fknn.{}.{}",
        step_id, start, end, k, metric_str, suffix, ext,
    ))
}

/// Validate that a cache file exists and has the expected byte size.
fn validate_cache_file(path: &Path, query_count: usize, k: usize, elem_size: usize) -> bool {
    let expected = query_count as u64 * (4 + k as u64 * elem_size as u64);
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() == expected,
        Err(_) => false,
    }
}

fn metric_label(metric: Metric) -> &'static str {
    match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        Metric::DotProduct => "dot_product",
        Metric::L1 => "l1",
    }
}

// -- Write partition cache ----------------------------------------------------

/// Write neighbor indices as ivec (each row: k i32 indices).
fn write_indices(path: &Path, results: &[Vec<Neighbor>], k: usize) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("create {}: {}", path.display(), e))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
    let dim = k as i32;
    for row in results {
        w.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
        for i in 0..k {
            let idx: i32 = if i < row.len() { row[i].index as i32 } else { -1 };
            w.write_all(&idx.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write neighbor distances as fvec (each row: k f32 distances).
fn write_distances(path: &Path, results: &[Vec<Neighbor>], k: usize) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("create {}: {}", path.display(), e))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
    let dim = k as i32;
    for row in results {
        w.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
        for i in 0..k {
            let dist: f32 = if i < row.len() { row[i].distance } else { f32::INFINITY };
            w.write_all(&dist.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write both ivec and fvec for a partition, then validate.
fn write_partition_cache(
    meta: &PartitionMeta,
    results: &[Vec<Neighbor>],
    k: usize,
    query_count: usize,
) -> Result<(), String> {
    write_indices(&meta.neighbors_path, results, k)?;
    write_distances(&meta.distances_path, results, k)?;

    if !validate_cache_file(&meta.neighbors_path, query_count, k, 4)
        || !validate_cache_file(&meta.distances_path, query_count, k, 4)
    {
        return Err(format!(
            "partition [{}, {}) cache files failed size validation after write",
            meta.start, meta.end,
        ));
    }
    Ok(())
}

// -- Cross-partition merge ----------------------------------------------------

/// Merge all partition cache files into final output.
///
/// For each query, reads its row from every partition's cached ivec and fvec,
/// collects all (index, distance) candidates, sorts by distance, takes top-K.
fn merge_partitions(
    partitions: &[PartitionMeta],
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    query_count: usize,
    ui: &UiHandle,
) -> Result<(), String> {
    let mut ivec_readers: Vec<BufReader<std::fs::File>> = Vec::with_capacity(partitions.len());
    let mut fvec_readers: Vec<BufReader<std::fs::File>> = Vec::with_capacity(partitions.len());

    for part in partitions {
        let f = std::fs::File::open(&part.neighbors_path)
            .map_err(|e| format!("open {}: {}", part.neighbors_path.display(), e))?;
        ivec_readers.push(BufReader::with_capacity(1 << 16, f));
        let f = std::fs::File::open(&part.distances_path)
            .map_err(|e| format!("open {}: {}", part.distances_path.display(), e))?;
        fvec_readers.push(BufReader::with_capacity(1 << 16, f));
    }

    let idx_file = std::fs::File::create(indices_path)
        .map_err(|e| format!("create {}: {}", indices_path.display(), e))?;
    let mut idx_writer = std::io::BufWriter::with_capacity(1 << 20, idx_file);

    let mut dist_writer = match distances_path {
        Some(p) => {
            let f = std::fs::File::create(p)
                .map_err(|e| format!("create {}: {}", p.display(), e))?;
            Some(std::io::BufWriter::with_capacity(1 << 20, f))
        }
        None => None,
    };

    let row_bytes = 4 + k * 4;
    let mut ivec_row = vec![0u8; row_bytes];
    let mut fvec_row = vec![0u8; row_bytes];

    let pb = make_query_progress_bar(query_count as u64, ui);

    for _qi in 0..query_count {
        let mut candidates: Vec<Neighbor> = Vec::with_capacity(partitions.len() * k);

        for pi in 0..partitions.len() {
            ivec_readers[pi]
                .read_exact(&mut ivec_row)
                .map_err(|e| format!("read ivec partition: {}", e))?;
            fvec_readers[pi]
                .read_exact(&mut fvec_row)
                .map_err(|e| format!("read fvec partition: {}", e))?;

            for j in 0..k {
                let off = 4 + j * 4;
                let idx = i32::from_le_bytes([
                    ivec_row[off], ivec_row[off + 1], ivec_row[off + 2], ivec_row[off + 3],
                ]);
                let dist = f32::from_le_bytes([
                    fvec_row[off], fvec_row[off + 1], fvec_row[off + 2], fvec_row[off + 3],
                ]);
                if idx >= 0 && dist.is_finite() {
                    candidates.push(Neighbor { index: idx as u32, distance: dist });
                }
            }
        }

        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        candidates.truncate(k);

        let dim = k as i32;
        idx_writer.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
        for j in 0..k {
            let idx: i32 = if j < candidates.len() { candidates[j].index as i32 } else { -1 };
            idx_writer.write_all(&idx.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        if let Some(ref mut dw) = dist_writer {
            dw.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
            for j in 0..k {
                let dist: f32 = if j < candidates.len() { candidates[j].distance } else { f32::INFINITY };
                dw.write_all(&dist.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }

        pb.inc(1);
    }
    pb.finish();

    idx_writer.flush().map_err(|e| e.to_string())?;
    if let Some(ref mut dw) = dist_writer {
        dw.flush().map_err(|e| e.to_string())?;
    }

    Ok(())
}

// -- Filtered top-K -----------------------------------------------------------

/// Find top-K neighbors from a filtered set of base vector ordinals (f32).
fn find_top_k_filtered_f32(
    query: &[f32],
    base_reader: &MmapVectorReader<f32>,
    ordinals: &[i32],
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<Neighbor> {
    let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
    let mut threshold = f32::INFINITY;
    let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(base_reader);

    for &ord in ordinals {
        let idx = ord as usize;
        if idx >= base_count { continue; }
        let dist = dist_fn(query, base_reader.get_slice(idx));
        if dist < threshold {
            heap.push(Neighbor { index: ord as u32, distance: dist });
            if heap.len() > k { heap.pop(); }
            if heap.len() == k { threshold = heap.peek().unwrap().distance; }
        }
    }

    let mut result: Vec<Neighbor> = heap.into_vec();
    result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result
}

/// Find top-K neighbors from a filtered set of base vector ordinals (f16).
fn find_top_k_filtered_f16(
    query: &[half::f16],
    base_reader: &MmapVectorReader<half::f16>,
    ordinals: &[i32],
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
) -> Vec<Neighbor> {
    let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
    let mut threshold = f32::INFINITY;
    let base_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(base_reader);

    for &ord in ordinals {
        let idx = ord as usize;
        if idx >= base_count { continue; }
        let dist = dist_fn(query, base_reader.get_slice(idx));
        if dist < threshold {
            heap.push(Neighbor { index: ord as u32, distance: dist });
            if heap.len() > k { heap.pop(); }
            if heap.len() == k { threshold = heap.peek().unwrap().distance; }
        }
    }

    let mut result: Vec<Neighbor> = heap.into_vec();
    result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result
}

// -- Partition compute --------------------------------------------------------

/// Compute filtered KNN for a single base-vector partition `[start, end)` (f32).
///
/// For each query, reads predicate-key ordinals, filters to those within the
/// partition range, and computes distances only to matching base vectors.
fn compute_partition_filtered_f32(
    query_reader: &MmapVectorReader<f32>,
    query_count: usize,
    base_reader: &Arc<MmapVectorReader<f32>>,
    keys_reader: &SlabReader,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    threads: usize,
    pb: &ProgressHandle,
) -> Vec<Vec<Neighbor>> {
    let mut results: Vec<Vec<Neighbor>> = (0..query_count).map(|_| Vec::new()).collect();

    if threads > 1 && query_count > 1 {
        let effective_threads = std::cmp::min(threads, query_count);
        let chunk_size = (query_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            results.chunks_mut(chunk_size).collect();

        std::thread::scope(|scope| {
            for (ci, chunk) in result_chunks.into_iter().enumerate() {
                let chunk_start = ci * chunk_size;
                let chunk_len = chunk.len();
                let base_ref = Arc::clone(base_reader);

                scope.spawn(move || {
                    for qi in 0..chunk_len {
                        let global_qi = chunk_start + qi;
                        let ordinals = match keys_reader.get(global_qi as i64) {
                            Ok(data) => read_ordinals(&data),
                            Err(_) => Vec::new(),
                        };
                        let filtered: Vec<i32> = ordinals.into_iter()
                            .filter(|&o| (o as usize) >= start && (o as usize) < end)
                            .collect();
                        chunk[qi] = find_top_k_filtered_f32(
                            query_reader.get_slice(global_qi),
                            &base_ref,
                            &filtered,
                            k,
                            dist_fn,
                        );
                        pb.inc(1);
                    }
                });
            }
        });
    } else {
        for qi in 0..query_count {
            let ordinals = match keys_reader.get(qi as i64) {
                Ok(data) => read_ordinals(&data),
                Err(_) => Vec::new(),
            };
            let filtered: Vec<i32> = ordinals.into_iter()
                .filter(|&o| (o as usize) >= start && (o as usize) < end)
                .collect();
            results[qi] = find_top_k_filtered_f32(
                query_reader.get_slice(qi),
                base_reader,
                &filtered,
                k,
                dist_fn,
            );
            pb.inc(1);
        }
    }

    results
}

/// Compute filtered KNN for a single base-vector partition `[start, end)` (f16).
fn compute_partition_filtered_f16(
    query_reader: &MmapVectorReader<half::f16>,
    query_count: usize,
    base_reader: &Arc<MmapVectorReader<half::f16>>,
    keys_reader: &SlabReader,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    threads: usize,
    pb: &ProgressHandle,
) -> Vec<Vec<Neighbor>> {
    let mut results: Vec<Vec<Neighbor>> = (0..query_count).map(|_| Vec::new()).collect();

    if threads > 1 && query_count > 1 {
        let effective_threads = std::cmp::min(threads, query_count);
        let chunk_size = (query_count + effective_threads - 1) / effective_threads;

        let result_chunks: Vec<&mut [Vec<Neighbor>]> =
            results.chunks_mut(chunk_size).collect();

        std::thread::scope(|scope| {
            for (ci, chunk) in result_chunks.into_iter().enumerate() {
                let chunk_start = ci * chunk_size;
                let chunk_len = chunk.len();
                let base_ref = Arc::clone(base_reader);

                scope.spawn(move || {
                    for qi in 0..chunk_len {
                        let global_qi = chunk_start + qi;
                        let ordinals = match keys_reader.get(global_qi as i64) {
                            Ok(data) => read_ordinals(&data),
                            Err(_) => Vec::new(),
                        };
                        let filtered: Vec<i32> = ordinals.into_iter()
                            .filter(|&o| (o as usize) >= start && (o as usize) < end)
                            .collect();
                        chunk[qi] = find_top_k_filtered_f16(
                            query_reader.get_slice(global_qi),
                            &base_ref,
                            &filtered,
                            k,
                            dist_fn,
                        );
                        pb.inc(1);
                    }
                });
            }
        });
    } else {
        for qi in 0..query_count {
            let ordinals = match keys_reader.get(qi as i64) {
                Ok(data) => read_ordinals(&data),
                Err(_) => Vec::new(),
            };
            let filtered: Vec<i32> = ordinals.into_iter()
                .filter(|&o| (o as usize) >= start && (o as usize) < end)
                .collect();
            results[qi] = find_top_k_filtered_f16(
                query_reader.get_slice(qi),
                base_reader,
                &filtered,
                k,
                dist_fn,
            );
            pb.inc(1);
        }
    }

    results
}

// -- Element type detection ---------------------------------------------------

enum ElementType { F32, F16 }

fn detect_element_type(path: &Path) -> ElementType {
    match path.extension().and_then(|e| e.to_str()) {
        Some("hvec") => ElementType::F16,
        _ => ElementType::F32,
    }
}

// -- CommandOp ----------------------------------------------------------------

impl CommandOp for ComputeFilteredKnnOp {
    fn command_path(&self) -> &str {
        "compute filtered-knn"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Brute-force filtered KNN with predicate-key pre-filtering".into(),
            body: format!(r#"# compute filtered-knn

Brute-force filtered KNN with predicate-key pre-filtering.

## Description

A variant of `compute knn` that restricts each query's candidate set to
the base vectors matching a pre-computed predicate. Uses the same
base-vector partitioning, sequential I/O, prefetch, and caching strategy
as `compute knn`.

For large base vector sets, the base space is split into partitions.
Each partition is processed sequentially with a prefetch thread paging in
upcoming partitions. Results are cached per partition. A cross-partition
merge combines per-partition top-K into the global top-K for each query.

## Options

{}

## Notes

- Requires a predicate-keys slab file from the `generate predicate-keys` command.
- Queries with empty predicate-key sets produce sentinel values (-1 indices, infinity distances).
- Thread count of 0 uses the system default (all available cores).
- Partition caching allows incremental and resumable computation.
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Partition results and top-K heaps".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Parallel query processing".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential prefetch for mmap'd base vectors".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let keys_str = match options.require("predicate-keys") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("indices") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };

        let k: usize = match options.require("neighbors") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid neighbors: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let metric_str = options.get("metric").unwrap_or("L2");
        let metric = match Metric::from_str(metric_str) {
            Some(m) => m,
            None => return error_result(
                format!("unknown metric: '{}'. Use L2, COSINE, DOT_PRODUCT, or L1", metric_str),
                start,
            ),
        };

        let threads: usize = options
            .get("threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| ctx.governor.current_or("threads", ctx.threads as u64) as usize);

        let partition_size: usize = options
            .get("partition_size")
            .and_then(|s| s.parse().ok())
            .or_else(|| ctx.governor.current("segmentsize").map(|v| v as usize))
            .unwrap_or(1_000_000);

        let step_id = if ctx.step_id.is_empty() {
            "compute-filtered-knn".to_string()
        } else {
            ctx.step_id.clone()
        };

        let base_path = resolve_path(base_str, &ctx.workspace);
        let query_path = resolve_path(query_str, &ctx.workspace);
        let keys_path = resolve_path(keys_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
        let distances_path = options
            .get("distances")
            .map(|s| resolve_path(s, &ctx.workspace));

        // Create output directories
        for path in std::iter::once(&indices_path).chain(distances_path.iter()) {
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return error_result(format!("failed to create directory: {}", e), start);
                    }
                }
            }
        }

        // Load predicate-keys slab
        let keys_reader = match SlabReader::open(&keys_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open predicate-keys: {}", e), start),
        };

        // Governor checkpoint
        if ctx.governor.checkpoint() {
            ctx.ui.log("  governor: throttle active");
        }

        match detect_element_type(&base_path) {
            ElementType::F32 => {
                let dist_fn = simd_distance::select_distance_fn(metric);
                execute_f32(
                    &base_path, &query_path, &keys_reader, &indices_path,
                    distances_path.as_deref(), k, metric, dist_fn, threads,
                    partition_size, &step_id, ctx, start,
                )
            }
            ElementType::F16 => {
                let dist_fn = simd_distance::select_distance_fn_f16(metric);
                execute_f16(
                    &base_path, &query_path, &keys_reader, &indices_path,
                    distances_path.as_deref(), k, metric, dist_fn, threads,
                    partition_size, &step_id, ctx, start,
                )
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("base", "Path", true, None, "Base vectors file (fvec or hvec)"),
            opt("query", "Path", true, None, "Query vectors file (fvec or hvec)"),
            opt("predicate-keys", "Path", true, None, "Predicate-keys slab from generate predicate-keys"),
            opt("indices", "Path", true, None, "Output neighbor indices (ivec)"),
            opt("distances", "Path", false, None, "Output neighbor distances (fvec)"),
            opt("neighbors", "int", true, None, "Number of nearest neighbors (k)"),
            opt("metric", "enum", false, Some("L2"), "Distance metric: L2, COSINE, DOT_PRODUCT, L1"),
            opt("threads", "int", false, Some("0"), "Thread count (0 = auto)"),
            opt("partition_size", "int", false, Some("1000000"), "Base vectors per partition for cache-backed computation"),
        ]
    }
}

// -- Execute (f32) ------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn execute_f32(
    base_path: &Path,
    query_path: &Path,
    keys_reader: &SlabReader,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    threads: usize,
    partition_size: usize,
    step_id: &str,
    ctx: &mut StreamContext,
    start: Instant,
) -> CommandResult {
    ctx.ui.log(&format!("  opening base vectors: {}", base_path.display()));
    let base_reader = match MmapVectorReader::<f32>::open_fvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => return error_result(format!("open base: {}", e), start),
    };
    ctx.ui.log(&format!("  opening query vectors: {}", query_path.display()));
    let query_reader = match MmapVectorReader::<f32>::open_fvec(query_path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("open query: {}", e), start),
    };

    let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&*base_reader);
    let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
    let keys_count = keys_reader.total_records() as usize;
    let actual_count = query_count.min(keys_count);

    let base_bytes = base_count as u64 * base_reader.entry_size() as u64;
    ctx.ui.log(&format!(
        "filtered KNN: {} queries x {} base vectors (f32), k={}, metric={:?}, threads={}",
        format_count(actual_count), format_count(base_count), k, metric, threads,
    ));
    ctx.ui.log(&format!(
        "  base: {} ({})  keys: {} records",
        base_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(base_bytes),
        format_count(keys_count),
    ));

    execute_with_partitions(
        &query_reader, &base_reader, keys_reader, base_count, actual_count,
        indices_path, distances_path, k, metric, dist_fn, threads, partition_size,
        step_id, ctx, start, compute_partition_filtered_f32,
    )
}

// -- Execute (f16) ------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn execute_f16(
    base_path: &Path,
    query_path: &Path,
    keys_reader: &SlabReader,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    threads: usize,
    partition_size: usize,
    step_id: &str,
    ctx: &mut StreamContext,
    start: Instant,
) -> CommandResult {
    ctx.ui.log(&format!("  opening base vectors: {}", base_path.display()));
    let base_reader = match MmapVectorReader::<half::f16>::open_hvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => return error_result(format!("open base: {}", e), start),
    };
    ctx.ui.log(&format!("  opening query vectors: {}", query_path.display()));
    let query_reader = match MmapVectorReader::<half::f16>::open_hvec(query_path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("open query: {}", e), start),
    };

    let base_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&*base_reader);
    let query_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&query_reader);
    let keys_count = keys_reader.total_records() as usize;
    let actual_count = query_count.min(keys_count);

    let base_bytes = base_count as u64 * base_reader.entry_size() as u64;
    ctx.ui.log(&format!(
        "filtered KNN: {} queries x {} base vectors (f16), k={}, metric={:?}, threads={}",
        format_count(actual_count), format_count(base_count), k, metric, threads,
    ));
    ctx.ui.log(&format!(
        "  base: {} ({})  keys: {} records",
        base_path.file_name().unwrap_or_default().to_string_lossy(),
        format_bytes(base_bytes),
        format_count(keys_count),
    ));

    execute_with_partitions(
        &query_reader, &base_reader, keys_reader, base_count, actual_count,
        indices_path, distances_path, k, metric, dist_fn, threads, partition_size,
        step_id, ctx, start, compute_partition_filtered_f16,
    )
}

// -- Partitioned execution (generic) ------------------------------------------

/// Shared partition/merge logic for both f32 and f16.
///
/// Base vectors are partitioned into contiguous ranges. Each partition is
/// processed sequentially: a prefetch thread warms the page cache, compute
/// threads process all queries against matching ordinals in the partition,
/// and a background writer flushes results to cache.
#[allow(clippy::too_many_arguments)]
fn execute_with_partitions<T: Send + Sync + 'static>(
    query_reader: &MmapVectorReader<T>,
    base_reader: &Arc<MmapVectorReader<T>>,
    keys_reader: &SlabReader,
    base_count: usize,
    query_count: usize,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    metric: Metric,
    dist_fn: fn(&[T], &[T]) -> f32,
    threads: usize,
    partition_size: usize,
    step_id: &str,
    ctx: &mut StreamContext,
    start: Instant,
    compute_fn: fn(
        &MmapVectorReader<T>, usize, &Arc<MmapVectorReader<T>>, &SlabReader,
        usize, usize, usize, fn(&[T], &[T]) -> f32, usize, &ProgressHandle,
    ) -> Vec<Vec<Neighbor>>,
) -> CommandResult {
    // Single-partition fast path
    if base_count <= partition_size {
        let pb = make_query_progress_bar(query_count as u64, &ctx.ui);
        let results = compute_fn(
            query_reader, query_count, base_reader, keys_reader,
            0, base_count, k, dist_fn, threads, &pb,
        );
        pb.finish();

        ctx.ui.log("  writing results...");
        if let Err(e) = write_indices(indices_path, &results, k) {
            return error_result(e, start);
        }
        let mut produced = vec![indices_path.to_path_buf()];
        if let Some(dist_path) = distances_path {
            if let Err(e) = write_distances(dist_path, &results, k) {
                return error_result(e, start);
            }
            produced.push(dist_path.to_path_buf());
        }

        return CommandResult {
            status: Status::Ok,
            message: format!(
                "filtered KNN: {} queries, k={}, metric={:?}, {} base vectors",
                query_count, k, metric, base_count,
            ),
            produced,
            elapsed: start.elapsed(),
        };
    }

    // -- Multi-partition path -------------------------------------------------

    // Memory-aware partition sizing (REQ-RM-09)
    let partition_size = if let Some(mem_ceiling) = ctx.governor.mem_ceiling() {
        let snapshot = super::super::resource::SystemSnapshot::sample();
        let target = (mem_ceiling as f64 * 0.85) as u64;
        let available = if snapshot.rss_bytes < target {
            target - snapshot.rss_bytes
        } else {
            (mem_ceiling as f64 * 0.10) as u64
        };
        let result_bytes = query_count as u64 * k as u64 * 8;
        let heap_bytes = threads as u64 * k as u64 * 8;
        let fixed_overhead = result_bytes + heap_bytes;

        if fixed_overhead > available {
            ctx.ui.log(&format!(
                "  WARNING: result set alone ({}) exceeds available memory ({})",
                format_bytes(fixed_overhead), format_bytes(available),
            ));
        }

        let remaining = available.saturating_sub(fixed_overhead);
        let entry_size = base_reader.entry_size() as u64;
        if entry_size > 0 {
            let max_partition = (remaining / entry_size) as usize;
            if max_partition < partition_size && max_partition >= 1000 {
                ctx.ui.log(&format!(
                    "  memory-aware partition sizing: {} → {} (available: {}, RSS: {}, ceiling: {})",
                    format_count(partition_size), format_count(max_partition),
                    format_bytes(available), format_bytes(snapshot.rss_bytes),
                    format_bytes(mem_ceiling),
                ));
                max_partition
            } else {
                partition_size
            }
        } else {
            partition_size
        }
    } else {
        partition_size
    };

    // Create cache directory
    if !ctx.cache.exists() {
        if let Err(e) = std::fs::create_dir_all(&ctx.cache) {
            return error_result(format!("create cache dir: {}", e), start);
        }
    }

    // Phase 1: Plan partitions and validate cache
    let estimated_partitions = (base_count + partition_size - 1) / partition_size;
    ctx.ui.log(&format!(
        "  planning ~{} partitions (partition_size={})...",
        estimated_partitions, format_count(partition_size),
    ));

    let plan_pb = ctx.ui.bar(estimated_partitions as u64, "validating cache");

    let mut partitions: Vec<PartitionMeta> = Vec::new();
    let mut part_start = 0;
    while part_start < base_count {
        let part_end = std::cmp::min(part_start + partition_size, base_count);
        let neighbors_path = build_cache_path(
            &ctx.cache, step_id, part_start, part_end, k, metric, "neighbors", "ivec",
        );
        let dist_cache_path = build_cache_path(
            &ctx.cache, step_id, part_start, part_end, k, metric, "distances", "fvec",
        );
        let cached = validate_cache_file(&neighbors_path, query_count, k, 4)
            && validate_cache_file(&dist_cache_path, query_count, k, 4);
        partitions.push(PartitionMeta {
            start: part_start,
            end: part_end,
            neighbors_path,
            distances_path: dist_cache_path,
            cached,
        });
        plan_pb.inc(1);
        part_start = part_end;
    }
    plan_pb.finish();

    let num_partitions = partitions.len();
    let cached_count = partitions.iter().filter(|p| p.cached).count();
    let to_compute = num_partitions - cached_count;

    ctx.ui.log(&format!(
        "  {} partitions of {} base vectors ({} cached, {} to compute)",
        num_partitions, format_count(partition_size), cached_count, to_compute,
    ));

    // Phase 2: Compute missing partitions with background writer
    //
    // Unlike unfiltered KNN which sequentially scans every base vector in a
    // partition, filtered KNN only accesses the sparse set of matching
    // ordinals. Prefetching entire partitions would page in data that is
    // never touched, bloating RSS. Instead we rely on demand-paging: the
    // kernel faults in only the pages containing matching ordinals.
    // release_range still frees completed partition pages afterward.

    let part_pb = ctx.ui.bar(num_partitions as u64, "computing partitions");

    let mut prev_writer: Option<std::thread::JoinHandle<Result<(), String>>> = None;

    for (pi, part) in partitions.iter().enumerate() {
        if part.cached {
            part_pb.inc(1);
            continue;
        }

        // Governor checkpoint at partition boundary
        if ctx.governor.checkpoint() {
            log::info!(
                "partition {}/{} — governor throttle active",
                pi + 1, num_partitions,
            );
        }

        let results = compute_fn(
            query_reader, query_count, base_reader, keys_reader,
            part.start, part.end, k, dist_fn, threads,
            &ctx.ui.bar(0, "partition compute"),
        );

        // Wait for previous background writer
        if let Some(handle) = prev_writer.take() {
            match join_writer_with_spinner(handle, "previous", &ctx.ui) {
                Ok(()) => {}
                Err(msg) => return error_result(msg, start),
            }
        }

        // Spawn background writer
        let neighbors_path = part.neighbors_path.clone();
        let distances_path_clone = part.distances_path.clone();
        let part_start_idx = part.start;
        let part_end_idx = part.end;
        let part_query_count = query_count;
        prev_writer = Some(std::thread::spawn(move || {
            let meta = PartitionMeta {
                start: part_start_idx,
                end: part_end_idx,
                neighbors_path,
                distances_path: distances_path_clone,
                cached: false,
            };
            write_partition_cache(&meta, &results, k, part_query_count)
        }));

        // Release completed partition's pages from page cache
        base_reader.release_range(part.start, part.end);

        part_pb.inc(1);
    }

    // Wait for final background writer
    if let Some(handle) = prev_writer.take() {
        match join_writer_with_spinner(handle, "final", &ctx.ui) {
            Ok(()) => {}
            Err(msg) => return error_result(msg, start),
        }
    }
    part_pb.finish();

    // Phase 3: Cross-partition merge
    ctx.ui.log(&format!(
        "  merging {} partitions ({} queries)...",
        num_partitions, format_count(query_count),
    ));

    if let Err(e) = merge_partitions(
        &partitions, indices_path, distances_path, k, query_count, &ctx.ui,
    ) {
        return error_result(format!("merge failed: {}", e), start);
    }

    let mut produced = vec![indices_path.to_path_buf()];
    if let Some(dp) = distances_path {
        produced.push(dp.to_path_buf());
    }

    CommandResult {
        status: Status::Ok,
        message: format!(
            "filtered KNN: {} queries, k={}, metric={:?}, {} base vectors ({} partitions)",
            query_count, k, metric, base_count, num_partitions,
        ),
        produced,
        elapsed: start.elapsed(),
    }
}

// -- Helpers ------------------------------------------------------------------

fn make_query_progress_bar(total: u64, ui: &UiHandle) -> ProgressHandle {
    ui.bar(total, "queries")
}

fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(ch);
    }
    result.chars().rev().collect()
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * 1024 * 1024 * 1024;
    if bytes >= TIB { format!("{:.1} TiB", bytes as f64 / TIB as f64) }
    else if bytes >= GIB { format!("{:.1} GiB", bytes as f64 / GIB as f64) }
    else if bytes >= MIB { format!("{:.1} MiB", bytes as f64 / MIB as f64) }
    else if bytes >= KIB { format!("{:.1} KiB", bytes as f64 / KIB as f64) }
    else { format!("{} B", bytes) }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn join_writer_with_spinner(
    handle: std::thread::JoinHandle<Result<(), String>>,
    label: &str,
    ui: &UiHandle,
) -> Result<(), String> {
    if !handle.is_finished() {
        let sp = ui.spinner(format!("flushing {} cache write...", label));
        let result = handle.join();
        sp.finish();
        match result {
            Ok(inner) => inner,
            Err(_) => Err("partition cache writer thread panicked".to_string()),
        }
    } else {
        match handle.join() {
            Ok(inner) => inner,
            Err(_) => Err("partition cache writer thread panicked".to_string()),
        }
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use slabtastic::{SlabWriter, WriterConfig};

    fn test_ctx(dir: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
        }
    }

    /// Create a small fvec file with `count` vectors of `dim` dimensions.
    fn create_fvec(dir: &std::path::Path, name: &str, dim: usize, count: usize) -> std::path::PathBuf {
        use byteorder::WriteBytesExt;
        let path = dir.join(name);
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        for i in 0..count {
            f.write_i32::<LittleEndian>(dim as i32).unwrap();
            for d in 0..dim {
                f.write_f32::<LittleEndian>((i * dim + d) as f32).unwrap();
            }
        }
        f.flush().unwrap();
        path
    }

    /// Create a predicate-keys slab where each record contains the given ordinals.
    fn create_keys_slab(dir: &std::path::Path, name: &str, keys: &[Vec<i32>]) -> std::path::PathBuf {
        use byteorder::WriteBytesExt;
        let path = dir.join(name);
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        for ordinals in keys {
            let mut buf = Vec::with_capacity(ordinals.len() * 4);
            for &ord in ordinals {
                buf.write_i32::<LittleEndian>(ord).unwrap();
            }
            writer.add_record(&buf).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    #[test]
    fn test_filtered_knn_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let dim = 4;
        let base_path = create_fvec(ws, "base.fvec", dim, 10);
        let query_path = create_fvec(ws, "query.fvec", dim, 3);

        let keys = vec![
            vec![0, 2, 4, 6, 8],
            vec![1, 3, 5, 7, 9],
            (0..10).collect(),
        ];
        let keys_path = create_keys_slab(ws, "keys.slab", &keys);
        let indices_path = ws.join("indices.ivec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("predicate-keys", keys_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("neighbors", "3".to_string());
        opts.set("metric", "L2".to_string());

        let mut op = ComputeFilteredKnnOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        // 3 queries × (4 + 3*4) = 48 bytes
        let meta = std::fs::metadata(&indices_path).unwrap();
        assert_eq!(meta.len(), 48);
    }

    #[test]
    fn test_filtered_knn_empty_filter() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let dim = 4;
        let base_path = create_fvec(ws, "base.fvec", dim, 10);
        let query_path = create_fvec(ws, "query.fvec", dim, 1);

        let keys = vec![vec![]];
        let keys_path = create_keys_slab(ws, "keys.slab", &keys);
        let indices_path = ws.join("indices.ivec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("predicate-keys", keys_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("neighbors", "5".to_string());
        opts.set("metric", "L2".to_string());

        let mut op = ComputeFilteredKnnOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        let data = std::fs::read(&indices_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let dim_val = cursor.read_i32::<LittleEndian>().unwrap();
        assert_eq!(dim_val, 5);
        for _ in 0..5 {
            assert_eq!(cursor.read_i32::<LittleEndian>().unwrap(), -1);
        }
    }

    #[test]
    fn test_filtered_knn_multi_partition() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        // Use cache dir so partition files are written
        std::fs::create_dir_all(ws.join(".cache")).unwrap();
        ctx.step_id = "test-fknn".to_string();

        let dim = 4;
        // 10 base vectors, 2 queries
        let base_path = create_fvec(ws, "base.fvec", dim, 10);
        let query_path = create_fvec(ws, "query.fvec", dim, 2);

        // Query 0 matches ordinals [0, 5, 9]
        // Query 1 matches ordinals [3, 7]
        let keys = vec![
            vec![0, 5, 9],
            vec![3, 7],
        ];
        let keys_path = create_keys_slab(ws, "keys.slab", &keys);
        let indices_path = ws.join("indices.ivec");
        let distances_path = ws.join("distances.fvec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("predicate-keys", keys_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("distances", distances_path.to_string_lossy().to_string());
        opts.set("neighbors", "2".to_string());
        opts.set("metric", "L2".to_string());
        // Force multiple partitions: 3 base vectors per partition → 4 partitions
        opts.set("partition_size", "3".to_string());

        let mut op = ComputeFilteredKnnOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        // 2 queries × (4 + 2*4) = 24 bytes each file
        assert_eq!(std::fs::metadata(&indices_path).unwrap().len(), 24);
        assert_eq!(std::fs::metadata(&distances_path).unwrap().len(), 24);

        // Verify cache files were created (4 partitions × 2 files each)
        let cache_files: Vec<_> = std::fs::read_dir(ws.join(".cache"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("fknn"))
            .collect();
        assert_eq!(cache_files.len(), 8, "expected 8 cache files (4 partitions × ivec + fvec)");

        // Read indices and verify query 0's top-2 neighbors are from [0, 5, 9]
        let data = std::fs::read(&indices_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        // Query 0
        let dim_val = cursor.read_i32::<LittleEndian>().unwrap();
        assert_eq!(dim_val, 2);
        let idx0 = cursor.read_i32::<LittleEndian>().unwrap();
        let idx1 = cursor.read_i32::<LittleEndian>().unwrap();
        assert!(
            [0, 5, 9].contains(&idx0) && [0, 5, 9].contains(&idx1),
            "query 0 results ({}, {}) should be from [0, 5, 9]",
            idx0, idx1,
        );
    }
}
