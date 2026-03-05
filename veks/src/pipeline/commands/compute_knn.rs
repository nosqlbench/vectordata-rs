// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute exact K-nearest neighbors.
//!
//! Computes ground truth KNN for a set of query vectors against a base vector
//! set. Outputs both neighbor indices (ivec) and distances (fvec).
//!
//! Supports both f32 (fvec) and f16 (hvec) input formats. The element type is
//! detected from the file extension of the base vectors path. When hvec files
//! are used, native f16 SIMD distance kernels operate directly on half-precision
//! data without an explicit upcast to f32.
//!
//! Supports distance metrics: L2 (Euclidean), Cosine, DotProduct, L1 (Manhattan).
//!
//! Uses multi-threaded query processing with a max-heap for top-k selection.
//! Base vectors are mmap'd for O(1) random access.
//!
//! For large base vector sets, supports **partitioned computation**: the base
//! space is split into partitions, each partition's per-query top-K results are
//! cached in `.cache/`, and a merge phase combines partition results into the
//! final global top-K. Cached partitions are reusable across runs with different
//! base-vector ranges.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{BufReader, Read as _, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};
use crate::pipeline::simd_distance::{self, Metric};

/// Pipeline command: compute exact KNN ground truth.
pub struct ComputeKnnOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeKnnOp)
}

// -- Top-K heap ---------------------------------------------------------------

/// A neighbor candidate with index and distance, ordered by distance descending
/// (max-heap: worst neighbor at top, easily evicted).
#[derive(Clone, Debug)]
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
        // Max-heap: larger distance = higher priority (gets evicted first)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Find the k nearest neighbors for a query within base vectors `[start, end)`.
///
/// Indices in the returned `Neighbor` values are absolute (i.e. relative to the
/// full base vector file, not the partition).
fn find_top_k_range(
    query: &[f32],
    base_reader: &MmapVectorReader<f32>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<Neighbor> {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for i in start..end {
        let base_vec = match base_reader.get(i) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dist = dist_fn(query, &base_vec);

        if heap.len() < k {
            heap.push(Neighbor {
                index: i as u32,
                distance: dist,
            });
        } else if let Some(worst) = heap.peek() {
            if dist < worst.distance {
                heap.pop();
                heap.push(Neighbor {
                    index: i as u32,
                    distance: dist,
                });
            }
        }
    }

    // Sort by distance ascending (closest first)
    let mut result: Vec<Neighbor> = heap.into_vec();
    result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result
}

// -- Partitioned computation --------------------------------------------------

/// Metadata for a single partition of the base vector space.
struct PartitionMeta {
    start: usize,
    end: usize,
    neighbors_path: PathBuf,
    distances_path: PathBuf,
}

/// Build a cache file path for a partition segment.
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
    let metric_str = match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        Metric::DotProduct => "dot_product",
        Metric::L1 => "l1",
    };
    cache_dir.join(format!(
        "{}.range_{:06}_{:06}.k{}.{}.{}.{}",
        step_id, start, end, k, metric_str, suffix, ext
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

/// Compute KNN for a single partition `[start, end)` across all queries.
fn compute_partition(
    queries: &[Vec<f32>],
    base_reader: &Arc<MmapVectorReader<f32>>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    threads: usize,
) -> Vec<Vec<Neighbor>> {
    let query_count = queries.len();

    if threads > 1 && query_count > 1 {
        use std::sync::Mutex;

        let effective_threads = std::cmp::min(threads, query_count);
        let results = Arc::new(Mutex::new(vec![Vec::new(); query_count]));
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        std::thread::scope(|scope| {
            let chunk_size = (query_count + effective_threads - 1) / effective_threads;
            for chunk_start in (0..query_count).step_by(chunk_size) {
                let chunk_end = std::cmp::min(chunk_start + chunk_size, query_count);
                let base_ref = Arc::clone(base_reader);
                let results_ref = Arc::clone(&results);
                let completed_ref = Arc::clone(&completed);

                scope.spawn(move || {
                    for qi in chunk_start..chunk_end {
                        let neighbors =
                            find_top_k_range(&queries[qi], &base_ref, start, end, k, dist_fn);
                        {
                            let mut lock = results_ref.lock().unwrap();
                            lock[qi] = neighbors;
                        }
                        let done = completed_ref
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            + 1;
                        if done % 10 == 0 || done == query_count {
                            eprint!(
                                "\r  {}/{} queries ({:.0}%)",
                                done,
                                query_count,
                                done as f64 / query_count as f64 * 100.0
                            );
                        }
                    }
                });
            }
        });
        eprintln!();

        Arc::try_unwrap(results).unwrap().into_inner().unwrap()
    } else {
        queries
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let neighbors = find_top_k_range(q, base_reader, start, end, k, dist_fn);
                if (i + 1) % 10 == 0 || i + 1 == query_count {
                    eprint!(
                        "\r  {}/{} queries ({:.0}%)",
                        i + 1,
                        query_count,
                        (i + 1) as f64 / query_count as f64 * 100.0
                    );
                }
                neighbors
            })
            .collect()
    }
}

// -- Element type detection ---------------------------------------------------

/// Detected element type based on file extension.
enum ElementType {
    F32,
    F16,
}

/// Detect the vector element type from the file extension.
fn detect_element_type(path: &Path) -> ElementType {
    match path.extension().and_then(|e| e.to_str()) {
        Some("hvec") => ElementType::F16,
        _ => ElementType::F32,
    }
}

// -- f16 KNN functions --------------------------------------------------------

/// Find the k nearest neighbors for an f16 query within base vectors `[start, end)`.
fn find_top_k_range_f16(
    query: &[half::f16],
    base_reader: &MmapVectorReader<half::f16>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
) -> Vec<Neighbor> {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for i in start..end {
        let base_vec = match base_reader.get(i) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dist = dist_fn(query, &base_vec);

        if heap.len() < k {
            heap.push(Neighbor {
                index: i as u32,
                distance: dist,
            });
        } else if let Some(worst) = heap.peek() {
            if dist < worst.distance {
                heap.pop();
                heap.push(Neighbor {
                    index: i as u32,
                    distance: dist,
                });
            }
        }
    }

    let mut result: Vec<Neighbor> = heap.into_vec();
    result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result
}

/// Compute KNN for a single partition `[start, end)` across all f16 queries.
fn compute_partition_f16(
    queries: &[Vec<half::f16>],
    base_reader: &Arc<MmapVectorReader<half::f16>>,
    start: usize,
    end: usize,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    threads: usize,
) -> Vec<Vec<Neighbor>> {
    let query_count = queries.len();

    if threads > 1 && query_count > 1 {
        use std::sync::Mutex;

        let effective_threads = std::cmp::min(threads, query_count);
        let results = Arc::new(Mutex::new(vec![Vec::new(); query_count]));
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        std::thread::scope(|scope| {
            let chunk_size = (query_count + effective_threads - 1) / effective_threads;
            for chunk_start in (0..query_count).step_by(chunk_size) {
                let chunk_end = std::cmp::min(chunk_start + chunk_size, query_count);
                let base_ref = Arc::clone(base_reader);
                let results_ref = Arc::clone(&results);
                let completed_ref = Arc::clone(&completed);

                scope.spawn(move || {
                    for qi in chunk_start..chunk_end {
                        let neighbors =
                            find_top_k_range_f16(&queries[qi], &base_ref, start, end, k, dist_fn);
                        {
                            let mut lock = results_ref.lock().unwrap();
                            lock[qi] = neighbors;
                        }
                        let done = completed_ref
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            + 1;
                        if done % 10 == 0 || done == query_count {
                            eprint!(
                                "\r  {}/{} queries ({:.0}%)",
                                done,
                                query_count,
                                done as f64 / query_count as f64 * 100.0
                            );
                        }
                    }
                });
            }
        });
        eprintln!();

        Arc::try_unwrap(results).unwrap().into_inner().unwrap()
    } else {
        queries
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let neighbors = find_top_k_range_f16(q, base_reader, start, end, k, dist_fn);
                if (i + 1) % 10 == 0 || i + 1 == query_count {
                    eprint!(
                        "\r  {}/{} queries ({:.0}%)",
                        i + 1,
                        query_count,
                        (i + 1) as f64 / query_count as f64 * 100.0
                    );
                }
                neighbors
            })
            .collect()
    }
}

/// Write partition results to cache files (both ivec and fvec).
fn write_partition_cache(
    meta: &PartitionMeta,
    results: &[Vec<Neighbor>],
    k: usize,
) -> Result<(), String> {
    write_indices(&meta.neighbors_path, results, k)?;
    write_distances(&meta.distances_path, results, k)?;
    Ok(())
}

/// Merge all partition cache files into final output files.
///
/// For each query row, reads that row from every partition's cached ivec and
/// fvec, collects all (index, distance) candidates, sorts by distance, and
/// takes the top-K. Memory usage is O(num_partitions * k) per query.
fn merge_partitions(
    partitions: &[PartitionMeta],
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    query_count: usize,
) -> Result<(), String> {
    // Open all partition files
    let mut ivec_readers: Vec<BufReader<std::fs::File>> = Vec::with_capacity(partitions.len());
    let mut fvec_readers: Vec<BufReader<std::fs::File>> = Vec::with_capacity(partitions.len());

    for part in partitions {
        let ivec_file = std::fs::File::open(&part.neighbors_path)
            .map_err(|e| format!("open {}: {}", part.neighbors_path.display(), e))?;
        ivec_readers.push(BufReader::with_capacity(1 << 16, ivec_file));

        let fvec_file = std::fs::File::open(&part.distances_path)
            .map_err(|e| format!("open {}: {}", part.distances_path.display(), e))?;
        fvec_readers.push(BufReader::with_capacity(1 << 16, fvec_file));
    }

    // Open output files
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

    let row_bytes = 4 + k * 4; // dim header + k elements
    let mut ivec_row = vec![0u8; row_bytes];
    let mut fvec_row = vec![0u8; row_bytes];

    for _qi in 0..query_count {
        // Collect candidates from all partitions
        let mut candidates: Vec<Neighbor> = Vec::with_capacity(partitions.len() * k);

        for (pi, _part) in partitions.iter().enumerate() {
            ivec_readers[pi]
                .read_exact(&mut ivec_row)
                .map_err(|e| format!("read ivec partition: {}", e))?;
            fvec_readers[pi]
                .read_exact(&mut fvec_row)
                .map_err(|e| format!("read fvec partition: {}", e))?;

            // Parse k (index, distance) pairs from this partition's row
            for j in 0..k {
                let idx_offset = 4 + j * 4;
                let idx = i32::from_le_bytes([
                    ivec_row[idx_offset],
                    ivec_row[idx_offset + 1],
                    ivec_row[idx_offset + 2],
                    ivec_row[idx_offset + 3],
                ]);
                let dist = f32::from_le_bytes([
                    fvec_row[idx_offset],
                    fvec_row[idx_offset + 1],
                    fvec_row[idx_offset + 2],
                    fvec_row[idx_offset + 3],
                ]);

                if idx >= 0 && dist.is_finite() {
                    candidates.push(Neighbor {
                        index: idx as u32,
                        distance: dist,
                    });
                }
            }
        }

        // Sort by distance ascending and take top-K
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        candidates.truncate(k);

        // Write merged row to indices output
        let dim = k as i32;
        idx_writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for j in 0..k {
            let idx: i32 = if j < candidates.len() {
                candidates[j].index as i32
            } else {
                -1
            };
            idx_writer
                .write_all(&idx.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }

        // Write merged row to distances output
        if let Some(ref mut dw) = dist_writer {
            dw.write_all(&dim.to_le_bytes())
                .map_err(|e| e.to_string())?;
            for j in 0..k {
                let dist: f32 = if j < candidates.len() {
                    candidates[j].distance
                } else {
                    f32::INFINITY
                };
                dw.write_all(&dist.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
    }

    idx_writer.flush().map_err(|e| e.to_string())?;
    if let Some(ref mut dw) = dist_writer {
        dw.flush().map_err(|e| e.to_string())?;
    }

    Ok(())
}

// -- CommandOp impl -----------------------------------------------------------

impl CommandOp for ComputeKnnOp {
    fn command_path(&self) -> &str {
        "compute knn"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("indices") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
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
            None => {
                return error_result(
                    format!(
                        "unknown metric: '{}'. Use L2, COSINE, DOT_PRODUCT, or L1",
                        metric_str
                    ),
                    start,
                )
            }
        };
        let threads: usize = options
            .get("threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(ctx.threads);

        let partition_size: usize = options
            .get("partition_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000_000);

        let step_id = if ctx.step_id.is_empty() {
            "compute-knn".to_string()
        } else {
            ctx.step_id.clone()
        };

        let base_path = resolve_path(base_str, &ctx.workspace);
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);

        // Optional distances output
        let distances_path = options
            .get("distances")
            .map(|s| resolve_path(s, &ctx.workspace));

        // Create output directories
        for path in std::iter::once(&indices_path).chain(distances_path.iter()) {
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return error_result(
                            format!("failed to create directory: {}", e),
                            start,
                        );
                    }
                }
            }
        }

        // Detect element type from base file extension and dispatch
        match detect_element_type(&base_path) {
            ElementType::F16 => {
                let dist_fn = simd_distance::select_distance_fn_f16(metric);
                execute_f16(
                    &base_path,
                    &query_path,
                    &indices_path,
                    distances_path.as_deref(),
                    k,
                    metric,
                    dist_fn,
                    threads,
                    partition_size,
                    &step_id,
                    ctx,
                    start,
                )
            }
            ElementType::F32 => {
                let dist_fn = simd_distance::select_distance_fn(metric);
                execute_f32(
                    &base_path,
                    &query_path,
                    &indices_path,
                    distances_path.as_deref(),
                    k,
                    metric,
                    dist_fn,
                    threads,
                    partition_size,
                    &step_id,
                    ctx,
                    start,
                )
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Base vectors file (fvec or hvec)".to_string(),
            },
            OptionDesc {
                name: "query".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Query vectors file (fvec or hvec)".to_string(),
            },
            OptionDesc {
                name: "indices".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output neighbor indices (ivec)".to_string(),
            },
            OptionDesc {
                name: "distances".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output neighbor distances (fvec)".to_string(),
            },
            OptionDesc {
                name: "neighbors".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of nearest neighbors (k)".to_string(),
            },
            OptionDesc {
                name: "metric".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("L2".to_string()),
                description: "Distance metric: L2, COSINE, DOT_PRODUCT, L1".to_string(),
            },
            OptionDesc {
                name: "threads".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Thread count (0 = auto)".to_string(),
            },
            OptionDesc {
                name: "partition_size".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1000000".to_string()),
                description: "Base vectors per partition for cache-backed computation".to_string(),
            },
        ]
    }
}

/// Execute KNN computation using f32 vectors.
#[allow(clippy::too_many_arguments)]
fn execute_f32(
    base_path: &Path,
    query_path: &Path,
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
    let base_reader = match MmapVectorReader::<f32>::open_fvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            return error_result(
                format!("failed to open base {}: {}", base_path.display(), e),
                start,
            )
        }
    };
    let query_reader = match MmapVectorReader::<f32>::open_fvec(query_path) {
        Ok(r) => r,
        Err(e) => {
            return error_result(
                format!("failed to open query {}: {}", query_path.display(), e),
                start,
            )
        }
    };

    let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&*base_reader);
    let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
    let base_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&*base_reader);
    let query_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&query_reader);

    if base_dim != query_dim {
        return error_result(
            format!("dimension mismatch: base={}, query={}", base_dim, query_dim),
            start,
        );
    }

    eprintln!(
        "KNN: {} queries x {} base vectors (f32), k={}, metric={:?}, threads={}, simd={}",
        query_count, base_count, k, metric, threads, simd_distance::simd_level()
    );

    let queries: Vec<Vec<f32>> = (0..query_count)
        .map(|i| query_reader.get(i).unwrap())
        .collect();

    execute_with_partitions(
        &queries,
        &base_reader,
        base_count,
        query_count,
        indices_path,
        distances_path,
        k,
        metric,
        dist_fn,
        threads,
        partition_size,
        step_id,
        ctx,
        start,
        compute_partition,
    )
}

/// Execute KNN computation using f16 vectors.
#[allow(clippy::too_many_arguments)]
fn execute_f16(
    base_path: &Path,
    query_path: &Path,
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
    let base_reader = match MmapVectorReader::<half::f16>::open_hvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            return error_result(
                format!("failed to open base {}: {}", base_path.display(), e),
                start,
            )
        }
    };
    let query_reader = match MmapVectorReader::<half::f16>::open_hvec(query_path) {
        Ok(r) => r,
        Err(e) => {
            return error_result(
                format!("failed to open query {}: {}", query_path.display(), e),
                start,
            )
        }
    };

    let base_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&*base_reader);
    let query_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&query_reader);
    let base_dim = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&*base_reader);
    let query_dim = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&query_reader);

    if base_dim != query_dim {
        return error_result(
            format!("dimension mismatch: base={}, query={}", base_dim, query_dim),
            start,
        );
    }

    eprintln!(
        "KNN: {} queries x {} base vectors (f16), k={}, metric={:?}, threads={}, simd={}",
        query_count, base_count, k, metric, threads, simd_distance::simd_level()
    );

    let queries: Vec<Vec<half::f16>> = (0..query_count)
        .map(|i| query_reader.get(i).unwrap())
        .collect();

    execute_with_partitions(
        &queries,
        &base_reader,
        base_count,
        query_count,
        indices_path,
        distances_path,
        k,
        metric,
        dist_fn,
        threads,
        partition_size,
        step_id,
        ctx,
        start,
        compute_partition_f16,
    )
}

/// Shared partition/merge logic for both f32 and f16 paths.
#[allow(clippy::too_many_arguments)]
fn execute_with_partitions<T>(
    queries: &[Vec<T>],
    base_reader: &Arc<MmapVectorReader<T>>,
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
    compute_fn: fn(&[Vec<T>], &Arc<MmapVectorReader<T>>, usize, usize, usize, fn(&[T], &[T]) -> f32, usize) -> Vec<Vec<Neighbor>>,
) -> CommandResult
where
    T: Send + Sync,
{
    // Single-partition fast path
    if base_count <= partition_size {
        let results = compute_fn(queries, base_reader, 0, base_count, k, dist_fn, threads);

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
                "computed KNN: {} queries, k={}, metric={:?}, {} base vectors",
                query_count, k, metric, base_count
            ),
            produced,
            elapsed: start.elapsed(),
        };
    }

    // Partitioned path

    if !ctx.cache.exists() {
        if let Err(e) = std::fs::create_dir_all(&ctx.cache) {
            return error_result(
                format!("failed to create cache directory: {}", e),
                start,
            );
        }
    }

    // Phase 1: Plan partitions
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
        partitions.push(PartitionMeta {
            start: part_start,
            end: part_end,
            neighbors_path,
            distances_path: dist_cache_path,
        });
        part_start = part_end;
    }

    eprintln!(
        "KNN: {} partitions of up to {} base vectors each",
        partitions.len(),
        partition_size
    );

    // Phase 2: Compute missing partitions
    for (pi, part) in partitions.iter().enumerate() {
        let ivec_valid = validate_cache_file(&part.neighbors_path, query_count, k, 4);
        let fvec_valid = validate_cache_file(&part.distances_path, query_count, k, 4);

        if ivec_valid && fvec_valid {
            eprintln!(
                "  partition {}/{} [{}, {}) — cached",
                pi + 1, partitions.len(), part.start, part.end
            );
            continue;
        }

        eprintln!(
            "  partition {}/{} [{}, {}) — computing...",
            pi + 1, partitions.len(), part.start, part.end
        );

        let results = compute_fn(
            queries, base_reader, part.start, part.end, k, dist_fn, threads,
        );

        if let Err(e) = write_partition_cache(part, &results, k) {
            return error_result(
                format!("failed to write partition cache: {}", e),
                start,
            );
        }

        if !validate_cache_file(&part.neighbors_path, query_count, k, 4)
            || !validate_cache_file(&part.distances_path, query_count, k, 4)
        {
            return error_result(
                format!(
                    "partition [{}, {}) cache files failed size validation after write",
                    part.start, part.end
                ),
                start,
            );
        }
    }

    // Phase 3: Merge
    eprintln!("  merging {} partitions...", partitions.len());

    if let Err(e) = merge_partitions(
        &partitions,
        indices_path,
        distances_path,
        k,
        query_count,
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
            "computed KNN: {} queries, k={}, metric={:?}, {} base vectors ({} partitions)",
            query_count, k, metric, base_count, partitions.len()
        ),
        produced,
        elapsed: start.elapsed(),
    }
}

/// Write neighbor indices as ivec (each row: k i32 indices).
fn write_indices(path: &Path, results: &[Vec<Neighbor>], k: usize) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

    let dim = k as i32;
    for row in results {
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for i in 0..k {
            let idx: i32 = if i < row.len() {
                row[i].index as i32
            } else {
                -1 // Padding if fewer than k neighbors
            };
            writer
                .write_all(&idx.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write neighbor distances as fvec (each row: k f32 distances).
fn write_distances(path: &Path, results: &[Vec<Neighbor>], k: usize) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

    let dim = k as i32;
    for row in results {
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        for i in 0..k {
            let dist: f32 = if i < row.len() {
                row[i].distance
            } else {
                f32::INFINITY
            };
            writer
                .write_all(&dist.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
        }
    }

    /// Generate test vectors with a fixed seed.
    fn gen_vectors(ctx: &mut StreamContext, path: &Path, dim: &str, count: &str) {
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", dim);
        opts.set("count", count);
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        let r = gen_op.execute(&opts, ctx);
        assert_eq!(r.status, Status::Ok);
    }

    /// Read raw ivec/fvec row data for comparison.
    fn read_rows(path: &Path, k: usize) -> Vec<Vec<u8>> {
        let data = std::fs::read(path).unwrap();
        let row_bytes = 4 + k * 4;
        data.chunks(row_bytes).map(|c| c.to_vec()).collect()
    }

    #[test]
    fn test_simd_distance_selection() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1] {
            let f = simd_distance::select_distance_fn(metric);
            let d = f(&a, &b);
            assert!(d.is_finite(), "metric {:?} returned non-finite", metric);
        }
    }

    #[test]
    fn test_knn_small() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "50");
        gen_vectors(&mut ctx, &query_path, "4", "5");

        let indices_path = workspace.join("indices.ivec");
        let distances_path = workspace.join("distances.fvec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("distances", distances_path.to_string_lossy().to_string());
        opts.set("neighbors", "3");
        opts.set("metric", "L2");

        let mut knn = ComputeKnnOp;
        let result = knn.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        // indices: 5 rows x (4 + 3*4) = 5 * 16 = 80 bytes
        let idx_size = std::fs::metadata(&indices_path).unwrap().len();
        assert_eq!(idx_size, 5 * (4 + 3 * 4));

        let dist_size = std::fs::metadata(&distances_path).unwrap().len();
        assert_eq!(dist_size, 5 * (4 + 3 * 4));
    }

    #[test]
    fn test_knn_results_sorted_by_distance() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "8", "30");
        gen_vectors(&mut ctx, &query_path, "8", "3");

        let distances_path = workspace.join("distances.fvec");
        let indices_path = workspace.join("indices.ivec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("distances", distances_path.to_string_lossy().to_string());
        opts.set("neighbors", "5");
        opts.set("metric", "L2");

        let mut knn = ComputeKnnOp;
        knn.execute(&opts, &mut ctx);

        // Read distances and verify they're sorted ascending per row
        let data = std::fs::read(&distances_path).unwrap();
        let k = 5;
        let record_size = 4 + k * 4;
        for row in 0..3 {
            let mut prev_dist = f32::NEG_INFINITY;
            for col in 0..k {
                let offset = row * record_size + 4 + col * 4;
                let dist = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                assert!(
                    dist >= prev_dist,
                    "row {} distances not sorted: {} < {}",
                    row,
                    dist,
                    prev_dist
                );
                prev_dist = dist;
            }
        }
    }

    #[test]
    fn test_knn_threaded() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "20");
        gen_vectors(&mut ctx, &query_path, "4", "10");

        // Single-threaded
        let idx1 = workspace.join("idx1.ivec");
        let mut opts1 = Options::new();
        opts1.set("base", base_path.to_string_lossy().to_string());
        opts1.set("query", query_path.to_string_lossy().to_string());
        opts1.set("indices", idx1.to_string_lossy().to_string());
        opts1.set("neighbors", "3");
        opts1.set("threads", "1");
        let mut knn1 = ComputeKnnOp;
        knn1.execute(&opts1, &mut ctx);

        // Multi-threaded
        let idx2 = workspace.join("idx2.ivec");
        let mut opts2 = Options::new();
        opts2.set("base", base_path.to_string_lossy().to_string());
        opts2.set("query", query_path.to_string_lossy().to_string());
        opts2.set("indices", idx2.to_string_lossy().to_string());
        opts2.set("neighbors", "3");
        opts2.set("threads", "4");
        let mut knn2 = ComputeKnnOp;
        ctx.threads = 4;
        knn2.execute(&opts2, &mut ctx);

        let data1 = std::fs::read(&idx1).unwrap();
        let data2 = std::fs::read(&idx2).unwrap();
        assert_eq!(data1, data2, "threaded and single-threaded results differ");
    }

    #[test]
    fn test_metric_parsing() {
        assert_eq!(Metric::from_str("L2"), Some(Metric::L2));
        assert_eq!(Metric::from_str("EUCLIDEAN"), Some(Metric::L2));
        assert_eq!(Metric::from_str("cosine"), Some(Metric::Cosine));
        assert_eq!(Metric::from_str("DOT_PRODUCT"), Some(Metric::DotProduct));
        assert_eq!(Metric::from_str("L1"), Some(Metric::L1));
        assert_eq!(Metric::from_str("invalid"), None);
    }

    #[test]
    fn test_knn_partitioned() {
        // 50 base vectors, 5 queries, partition_size=10 → 5 partitions.
        // Verify partitioned results match single-partition computation.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "50");
        gen_vectors(&mut ctx, &query_path, "4", "5");

        let k = 3;

        // Single-partition reference (partition_size > base_count)
        let ref_idx = workspace.join("ref.ivec");
        let ref_dist = workspace.join("ref.fvec");
        let mut opts_ref = Options::new();
        opts_ref.set("base", base_path.to_string_lossy().to_string());
        opts_ref.set("query", query_path.to_string_lossy().to_string());
        opts_ref.set("indices", ref_idx.to_string_lossy().to_string());
        opts_ref.set("distances", ref_dist.to_string_lossy().to_string());
        opts_ref.set("neighbors", k.to_string());
        opts_ref.set("metric", "L2");
        let mut knn_ref = ComputeKnnOp;
        let r = knn_ref.execute(&opts_ref, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Partitioned computation (partition_size=10 → 5 partitions)
        let part_idx = workspace.join("part.ivec");
        let part_dist = workspace.join("part.fvec");
        let mut opts_part = Options::new();
        opts_part.set("base", base_path.to_string_lossy().to_string());
        opts_part.set("query", query_path.to_string_lossy().to_string());
        opts_part.set("indices", part_idx.to_string_lossy().to_string());
        opts_part.set("distances", part_dist.to_string_lossy().to_string());
        opts_part.set("neighbors", k.to_string());
        opts_part.set("metric", "L2");
        opts_part.set("partition_size", "10");
        ctx.step_id = "test-knn-part".to_string();
        let mut knn_part = ComputeKnnOp;
        let r = knn_part.execute(&opts_part, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Results should be identical
        let ref_idx_data = read_rows(&ref_idx, k);
        let part_idx_data = read_rows(&part_idx, k);
        assert_eq!(ref_idx_data, part_idx_data, "partitioned indices differ from reference");

        let ref_dist_data = read_rows(&ref_dist, k);
        let part_dist_data = read_rows(&part_dist, k);
        assert_eq!(ref_dist_data, part_dist_data, "partitioned distances differ from reference");

        // Verify cache files exist (5 partitions)
        let cache_dir = workspace.join(".cache");
        let cache_files: Vec<_> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(
            cache_files.len(),
            10, // 5 partitions x 2 files (ivec + fvec)
            "expected 10 cache files, found {}",
            cache_files.len()
        );
    }

    #[test]
    fn test_knn_cache_reuse() {
        // Run partitioned KNN twice; verify the second run reuses cache
        // (produces identical results without recomputing).
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "50");
        gen_vectors(&mut ctx, &query_path, "4", "5");

        let k = 3;
        ctx.step_id = "cache-reuse".to_string();

        // First run
        let idx1 = workspace.join("idx1.ivec");
        let dist1 = workspace.join("dist1.fvec");
        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", idx1.to_string_lossy().to_string());
        opts.set("distances", dist1.to_string_lossy().to_string());
        opts.set("neighbors", k.to_string());
        opts.set("metric", "L2");
        opts.set("partition_size", "10");
        let mut knn = ComputeKnnOp;
        let r = knn.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Record cache file mtimes
        let cache_dir = workspace.join(".cache");
        let mut mtimes: Vec<_> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| {
                let meta = e.metadata().unwrap();
                (e.file_name(), meta.modified().unwrap())
            })
            .collect();
        mtimes.sort_by(|a, b| a.0.cmp(&b.0));

        // Brief pause so mtime would differ if rewritten
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Second run with different output paths
        let idx2 = workspace.join("idx2.ivec");
        let dist2 = workspace.join("dist2.fvec");
        let mut opts2 = Options::new();
        opts2.set("base", base_path.to_string_lossy().to_string());
        opts2.set("query", query_path.to_string_lossy().to_string());
        opts2.set("indices", idx2.to_string_lossy().to_string());
        opts2.set("distances", dist2.to_string_lossy().to_string());
        opts2.set("neighbors", k.to_string());
        opts2.set("metric", "L2");
        opts2.set("partition_size", "10");
        let mut knn2 = ComputeKnnOp;
        let r2 = knn2.execute(&opts2, &mut ctx);
        assert_eq!(r2.status, Status::Ok);

        // Cache files should not have been rewritten
        let mut mtimes2: Vec<_> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| {
                let meta = e.metadata().unwrap();
                (e.file_name(), meta.modified().unwrap())
            })
            .collect();
        mtimes2.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(mtimes, mtimes2, "cache files were unexpectedly rewritten");

        // Results should be identical
        let data1 = std::fs::read(&idx1).unwrap();
        let data2 = std::fs::read(&idx2).unwrap();
        assert_eq!(data1, data2, "second run produced different results");
    }

    #[test]
    fn test_knn_cache_validation() {
        // Truncate a cache file and verify it's recomputed.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let mut ctx = test_ctx(workspace);

        let base_path = workspace.join("base.fvec");
        let query_path = workspace.join("query.fvec");
        gen_vectors(&mut ctx, &base_path, "4", "30");
        gen_vectors(&mut ctx, &query_path, "4", "3");

        let k = 2;
        ctx.step_id = "cache-val".to_string();

        // First run
        let idx1 = workspace.join("idx1.ivec");
        let dist1 = workspace.join("dist1.fvec");
        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", idx1.to_string_lossy().to_string());
        opts.set("distances", dist1.to_string_lossy().to_string());
        opts.set("neighbors", k.to_string());
        opts.set("metric", "L2");
        opts.set("partition_size", "10");
        let mut knn = ComputeKnnOp;
        let r = knn.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Truncate one cache file
        let cache_dir = workspace.join(".cache");
        let first_ivec = std::fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(".neighbors.ivec")
            })
            .unwrap();
        let ivec_path = first_ivec.path();
        std::fs::write(&ivec_path, b"truncated").unwrap();

        // Second run — should recompute the truncated partition
        let idx2 = workspace.join("idx2.ivec");
        let dist2 = workspace.join("dist2.fvec");
        let mut opts2 = Options::new();
        opts2.set("base", base_path.to_string_lossy().to_string());
        opts2.set("query", query_path.to_string_lossy().to_string());
        opts2.set("indices", idx2.to_string_lossy().to_string());
        opts2.set("distances", dist2.to_string_lossy().to_string());
        opts2.set("neighbors", k.to_string());
        opts2.set("metric", "L2");
        opts2.set("partition_size", "10");
        let mut knn2 = ComputeKnnOp;
        let r2 = knn2.execute(&opts2, &mut ctx);
        assert_eq!(r2.status, Status::Ok);

        // Results should still be correct (match original)
        let data1 = std::fs::read(&idx1).unwrap();
        let data2 = std::fs::read(&idx2).unwrap();
        assert_eq!(data1, data2, "recomputed results differ from original");
    }
}
