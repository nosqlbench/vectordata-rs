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

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt};
use half;
use indicatif::ProgressStyle;

use crate::pipeline::display::ProgressDisplay;
use slabtastic::SlabReader;
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
the base vectors matching a pre-computed predicate. The predicate answer
keys are stored in a slab file produced by `generate predicate-keys`:
each slab record `i` is a packed `[i32 LE]*` array of base ordinals that
satisfy predicate `i`.

For query `i`, only the base vectors whose ordinals appear in the
predicate-keys record `i` are considered. This models pre-filtered
vector search where metadata predicates narrow the candidate set
before distance computation.

Supports both f32 (fvec) and f16 (hvec) input formats, with element type
auto-detected from the base file extension.

## Options

{}

## Notes

- Requires a predicate-keys slab file from the `generate predicate-keys` command.
- Queries with empty predicate-key sets produce sentinel values (-1 indices, infinity distances).
- Thread count of 0 uses the system default (all available cores).
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Filtered candidate sets and top-K heaps".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Parallel query processing".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential prefetch for mmap'd vectors".into(), adjustable: false },
        ]
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
        let keys_str = match options.require("predicate-keys") {
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
            None => return error_result(
                format!("unknown metric: '{}'. Use L2, COSINE, DOT_PRODUCT, or L1", metric_str),
                start,
            ),
        };

        let threads: usize = options
            .get("threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| ctx.governor.current_or("threads", ctx.threads as u64) as usize);

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

        // Governor checkpoint before filtered KNN processing
        if ctx.governor.checkpoint() {
            ctx.display.log("  governor: throttle active");
        }

        // Dispatch based on element type
        let display = ctx.display.clone();
        match detect_element_type(&base_path) {
            ElementType::F32 => {
                let dist_fn = simd_distance::select_distance_fn(metric);
                execute_filtered_f32(
                    &base_path, &query_path, &keys_reader, &indices_path,
                    distances_path.as_deref(), k, dist_fn, threads, start, &display,
                )
            }
            ElementType::F16 => {
                let dist_fn = simd_distance::select_distance_fn_f16(metric);
                execute_filtered_f16(
                    &base_path, &query_path, &keys_reader, &indices_path,
                    distances_path.as_deref(), k, dist_fn, threads, start, &display,
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
        ]
    }
}

/// Detect whether the file uses f32 or f16 elements based on extension.
enum ElementType {
    F32,
    F16,
}

fn detect_element_type(path: &Path) -> ElementType {
    match path.extension().and_then(|e| e.to_str()) {
        Some("hvec") => ElementType::F16,
        _ => ElementType::F32,
    }
}

/// Execute filtered KNN for f32 base/query vectors.
fn execute_filtered_f32(
    base_path: &Path,
    query_path: &Path,
    keys_reader: &SlabReader,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    threads: usize,
    start: Instant,
    display: &ProgressDisplay,
) -> CommandResult {
    let base_reader = match MmapVectorReader::<f32>::open_fvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => return error_result(format!("failed to open base vectors: {}", e), start),
    };
    let query_reader = match MmapVectorReader::<f32>::open_fvec(query_path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("failed to open query vectors: {}", e), start),
    };

    use vectordata::VectorReader;
    let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
    let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&*base_reader);
    let keys_count = keys_reader.total_records() as usize;
    let actual_count = query_count.min(keys_count);

    display.log(&format!(
        "compute filtered-knn: {} queries, {} predicate-key records, k={}, {} base vectors",
        query_count,
        keys_count,
        k,
        base_count,
    ));

    let pb = display.bar_with_style(
        actual_count as u64,
        ProgressStyle::with_template("  {spinner} [{bar:40}] {pos}/{len} queries ({eta})")
            .unwrap()
            .progress_chars("=> "),
    );

    // Process queries — each query gets its own filtered ordinal set
    let mut all_results: Vec<Vec<Neighbor>> = Vec::with_capacity(actual_count);

    if threads > 1 && actual_count > 1 {
        // Multi-threaded: process chunks of queries in parallel
        let effective_threads = threads.min(actual_count);
        let chunk_size = (actual_count + effective_threads - 1) / effective_threads;

        let mut result_vecs: Vec<Vec<Neighbor>> = (0..actual_count).map(|_| Vec::new()).collect();
        let result_chunks: Vec<&mut [Vec<Neighbor>]> = result_vecs.chunks_mut(chunk_size).collect();
        let base_ref = &base_reader;
        let query_ref = &query_reader;
        let keys_ref = keys_reader;
        let pb_ref = &pb;

        std::thread::scope(|scope| {
            for (ci, chunk) in result_chunks.into_iter().enumerate() {
                let chunk_start = ci * chunk_size;
                scope.spawn(move || {
                    for qi in 0..chunk.len() {
                        let global_qi = chunk_start + qi;
                        let ordinals = match keys_ref.get(global_qi as i64) {
                            Ok(data) => read_ordinals(&data),
                            Err(_) => Vec::new(),
                        };
                        let query_vec = query_ref.get_slice(global_qi);
                        chunk[qi] = find_top_k_filtered_f32(
                            query_vec, base_ref, &ordinals, k, dist_fn,
                        );
                        pb_ref.inc(1);
                    }
                });
            }
        });

        all_results = result_vecs;
    } else {
        for qi in 0..actual_count {
            let ordinals = match keys_reader.get(qi as i64) {
                Ok(data) => read_ordinals(&data),
                Err(_) => Vec::new(),
            };
            let query_vec = query_reader.get_slice(qi);
            let result = find_top_k_filtered_f32(query_vec, &base_reader, &ordinals, k, dist_fn);
            all_results.push(result);
            pb.inc(1);
        }
    }

    pb.finish_and_clear();

    write_results(&all_results, k, indices_path, distances_path, actual_count, start, display)
}

/// Execute filtered KNN for f16 base/query vectors.
fn execute_filtered_f16(
    base_path: &Path,
    query_path: &Path,
    keys_reader: &SlabReader,
    indices_path: &Path,
    distances_path: Option<&Path>,
    k: usize,
    dist_fn: fn(&[half::f16], &[half::f16]) -> f32,
    threads: usize,
    start: Instant,
    display: &ProgressDisplay,
) -> CommandResult {
    let base_reader = match MmapVectorReader::<half::f16>::open_hvec(base_path) {
        Ok(r) => Arc::new(r),
        Err(e) => return error_result(format!("failed to open base vectors: {}", e), start),
    };
    let query_reader = match MmapVectorReader::<half::f16>::open_hvec(query_path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("failed to open query vectors: {}", e), start),
    };

    use vectordata::VectorReader;
    let query_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&query_reader);
    let base_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&*base_reader);
    let keys_count = keys_reader.total_records() as usize;
    let actual_count = query_count.min(keys_count);

    display.log(&format!(
        "compute filtered-knn: {} queries, {} predicate-key records, k={}, {} base vectors (f16)",
        query_count,
        keys_count,
        k,
        base_count,
    ));

    let pb = display.bar_with_style(
        actual_count as u64,
        ProgressStyle::with_template("  {spinner} [{bar:40}] {pos}/{len} queries ({eta})")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut all_results: Vec<Vec<Neighbor>> = Vec::with_capacity(actual_count);

    if threads > 1 && actual_count > 1 {
        let effective_threads = threads.min(actual_count);
        let chunk_size = (actual_count + effective_threads - 1) / effective_threads;

        let mut result_vecs: Vec<Vec<Neighbor>> = (0..actual_count).map(|_| Vec::new()).collect();
        let result_chunks: Vec<&mut [Vec<Neighbor>]> = result_vecs.chunks_mut(chunk_size).collect();
        let base_ref = &base_reader;
        let query_ref = &query_reader;
        let keys_ref = keys_reader;
        let pb_ref = &pb;

        std::thread::scope(|scope| {
            for (ci, chunk) in result_chunks.into_iter().enumerate() {
                let chunk_start = ci * chunk_size;
                scope.spawn(move || {
                    for qi in 0..chunk.len() {
                        let global_qi = chunk_start + qi;
                        let ordinals = match keys_ref.get(global_qi as i64) {
                            Ok(data) => read_ordinals(&data),
                            Err(_) => Vec::new(),
                        };
                        let query_vec = query_ref.get_slice(global_qi);
                        chunk[qi] = find_top_k_filtered_f16(
                            query_vec, base_ref, &ordinals, k, dist_fn,
                        );
                        pb_ref.inc(1);
                    }
                });
            }
        });

        all_results = result_vecs;
    } else {
        for qi in 0..actual_count {
            let ordinals = match keys_reader.get(qi as i64) {
                Ok(data) => read_ordinals(&data),
                Err(_) => Vec::new(),
            };
            let query_vec = query_reader.get_slice(qi);
            let result = find_top_k_filtered_f16(query_vec, &base_reader, &ordinals, k, dist_fn);
            all_results.push(result);
            pb.inc(1);
        }
    }

    pb.finish_and_clear();

    write_results(&all_results, k, indices_path, distances_path, actual_count, start, display)
}

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
    use vectordata::VectorReader;
    let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(base_reader);

    for &ord in ordinals {
        let idx = ord as usize;
        if idx >= base_count {
            continue;
        }
        let base_vec = base_reader.get_slice(idx);
        let dist = dist_fn(query, base_vec);

        if dist < threshold {
            heap.push(Neighbor { index: ord as u32, distance: dist });
            if heap.len() > k {
                heap.pop();
            }
            if heap.len() == k {
                threshold = heap.peek().unwrap().distance;
            }
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
    use vectordata::VectorReader;
    let base_count = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(base_reader);

    for &ord in ordinals {
        let idx = ord as usize;
        if idx >= base_count {
            continue;
        }
        let base_vec = base_reader.get_slice(idx);
        let dist = dist_fn(query, base_vec);

        if dist < threshold {
            heap.push(Neighbor { index: ord as u32, distance: dist });
            if heap.len() > k {
                heap.pop();
            }
            if heap.len() == k {
                threshold = heap.peek().unwrap().distance;
            }
        }
    }

    let mut result: Vec<Neighbor> = heap.into_vec();
    result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result
}

/// Write KNN results (indices and optional distances) to output files.
///
/// Each query produces a fixed-width ivec record `[k: i32 LE][idx_0..idx_k-1: i32 LE]`.
/// If fewer than `k` neighbors were found, remaining slots are filled with `-1`
/// (indices) and `f32::INFINITY` (distances).
fn write_results(
    all_results: &[Vec<Neighbor>],
    k: usize,
    indices_path: &Path,
    distances_path: Option<&Path>,
    query_count: usize,
    start: Instant,
    display: &ProgressDisplay,
) -> CommandResult {
    let mut idx_file = match std::fs::File::create(indices_path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => return error_result(format!("failed to create indices file: {}", e), start),
    };

    let mut dist_file = match distances_path {
        Some(p) => match std::fs::File::create(p) {
            Ok(f) => Some(std::io::BufWriter::new(f)),
            Err(e) => return error_result(format!("failed to create distances file: {}", e), start),
        },
        None => None,
    };

    use byteorder::WriteBytesExt;

    for result in all_results {
        // Write ivec record: [dimension: i32 LE][indices...]
        idx_file.write_i32::<LittleEndian>(k as i32).unwrap();
        for i in 0..k {
            let idx = result.get(i).map_or(-1i32, |n| n.index as i32);
            idx_file.write_i32::<LittleEndian>(idx).unwrap();
        }

        if let Some(ref mut df) = dist_file {
            // Write fvec record: [dimension: i32 LE][distances...]
            df.write_i32::<LittleEndian>(k as i32).unwrap();
            for i in 0..k {
                let dist = result.get(i).map_or(f32::INFINITY, |n| n.distance);
                df.write_f32::<LittleEndian>(dist).unwrap();
            }
        }
    }

    idx_file.flush().unwrap();
    if let Some(ref mut df) = dist_file {
        df.flush().unwrap();
    }

    let mut produced = vec![indices_path.to_path_buf()];
    if let Some(p) = distances_path {
        produced.push(p.to_path_buf());
    }

    let message = format!(
        "filtered KNN complete: {} queries × k={}",
        query_count, k,
    );
    display.log(&format!("compute filtered-knn: {}", message));

    CommandResult {
        status: Status::Ok,
        message,
        produced,
        elapsed: start.elapsed(),
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(path_str);
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
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
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
        // 10 base vectors, 3 queries
        let base_path = create_fvec(ws, "base.fvec", dim, 10);
        let query_path = create_fvec(ws, "query.fvec", dim, 3);

        // Query 0: consider base vectors [0, 2, 4, 6, 8]
        // Query 1: consider base vectors [1, 3, 5, 7, 9]
        // Query 2: consider all [0..9]
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

        // Verify the output file exists and has correct size
        // 3 queries × (4 + 3*4) = 3 × 16 = 48 bytes
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

        // Empty filter — no candidates
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

        // Should produce valid output with -1 sentinel indices
        let data = std::fs::read(&indices_path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let dim_val = cursor.read_i32::<LittleEndian>().unwrap();
        assert_eq!(dim_val, 5);
        // All indices should be -1 since no candidates
        for _ in 0..5 {
            let idx = cursor.read_i32::<LittleEndian>().unwrap();
            assert_eq!(idx, -1);
        }
    }
}
