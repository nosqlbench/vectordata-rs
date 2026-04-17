// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute exact KNN using direct BLAS sgemm.
//!
//! Computes brute-force exact K-nearest neighbors by calling `cblas_sgemm`
//! directly for the distance matrix computation, then selecting top-K per
//! query via a max-heap. This produces identical results to FAISS
//! `IndexFlatIP`/`IndexFlatL2` when both use the same BLAS backend,
//! without requiring the FAISS C++ library or cmake to build.
//!
//! Supports inner product (IP/DOT\_PRODUCT/COSINE) and L2 metrics.
//!
//! Links dynamically against the system BLAS (`libopenblas-dev` or
//! `libmkl-dev`) — no static compilation needed.

use std::collections::BinaryHeap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};
use super::source_window::resolve_source;

// CBLAS FFI — links against system BLAS (OpenBLAS or MKL via libblas.so)
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,     // CblasRowMajor = 101
        transa: i32,    // CblasNoTrans = 111
        transb: i32,    // CblasTrans = 112
        m: i32,         // rows of A (n_query batch)
        n: i32,         // cols of B^T (n_base)
        k: i32,         // shared dim
        alpha: f32,
        a: *const f32,  // query batch (m x k)
        lda: i32,       // k
        b: *const f32,  // base data (n x k, transposed by flag)
        ldb: i32,       // k
        beta: f32,
        c: *mut f32,    // output scores (m x n)
        ldc: i32,       // n
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// A neighbor candidate for the top-k heap.
#[derive(Clone, Copy)]
pub(super) struct Neighbor {
    pub(super) index: u32,
    pub(super) score: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}
impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap by score (worst score at top, gets evicted first).
        // For IP: higher score = better, so we want to evict the LOWEST.
        // BinaryHeap is a max-heap, so we reverse: lower score = higher priority.
        match other.score.partial_cmp(&self.score).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => self.index.cmp(&other.index),
            ord => ord,
        }
    }
}

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: msg.into(),
        elapsed: start.elapsed(),
        produced: vec![],
    }
}

fn resolve_path(s: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::Path::new(s);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

/// Select top-k from a row of scores (higher score = better).
///
/// Uses a min-heap of size k: the heap top is the worst (lowest-score)
/// element. When a new element has a higher score than the heap top,
/// the top is replaced. After processing all elements, the heap
/// contains the k highest-scoring elements.
pub(super) fn topk_indices(scores: &[f32], k: usize) -> Vec<Neighbor> {
    let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
    for (i, &score) in scores.iter().enumerate() {
        heap.push(Neighbor { index: i as u32, score });
        if heap.len() > k {
            heap.pop(); // evict the element with highest Ord priority = lowest score
        }
    }
    // heap.into_sorted_vec() returns elements from lowest to highest Ord priority.
    // Our Ord: lower score = higher priority (gets evicted first).
    // So into_sorted_vec gives: [highest_score, ..., lowest_score]
    // This is already best-first order for IP.
    heap.into_sorted_vec()
}

/// Compute IP or L2 score matrix via cblas_sgemm.
///
/// For IP: scores = queries @ base.T
/// For L2: scores = -(||q||^2 + ||b||^2 - 2*q.b) (negated so higher = better)
///
/// # Safety
/// Calls cblas_sgemm via FFI. Input slices must be valid contiguous f32 data.
pub(super) unsafe fn blas_sgemm_scores(
    query_data: &[f32], n_query: usize,
    base_data: &[f32], n_base: usize,
    dim: usize, use_ip: bool,
    scores: &mut [f32],
) {
    if use_ip {
        unsafe { cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            n_query as i32, n_base as i32, dim as i32,
            1.0, query_data.as_ptr(), dim as i32,
            base_data.as_ptr(), dim as i32,
            0.0, scores.as_mut_ptr(), n_base as i32,
        ) };
    } else {
        unsafe { cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            n_query as i32, n_base as i32, dim as i32,
            -2.0, query_data.as_ptr(), dim as i32,
            base_data.as_ptr(), dim as i32,
            0.0, scores.as_mut_ptr(), n_base as i32,
        ) };
        for qi in 0..n_query {
            let q_slice = &query_data[qi * dim..(qi + 1) * dim];
            let q_norm_sq: f32 = q_slice.iter().map(|v| v * v).sum();
            let row = &mut scores[qi * n_base..(qi + 1) * n_base];
            for (bi, score) in row.iter_mut().enumerate() {
                let b_norm_sq: f32 = base_data[bi * dim..(bi + 1) * dim]
                    .iter().map(|v| v * v).sum();
                *score += q_norm_sq + b_norm_sq;
            }
            for s in row.iter_mut() { *s = -*s; }
        }
    }
}

/// Pipeline command: compute exact KNN via direct BLAS sgemm.
pub struct ComputeKnnBlasOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeKnnBlasOp)
}

impl CommandOp for ComputeKnnBlasOp {
    fn command_path(&self) -> &str {
        "compute knn-blas"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Brute-force exact KNN via BLAS sgemm (knn_utils compatible)".into(),
            body: format!(r#"# compute knn-blas

Brute-force exact K-nearest-neighbor ground truth computation using
direct BLAS `cblas_sgemm` for the distance matrix, with Rust top-K
heap selection.

Produces identical results to FAISS `IndexFlatIP`/`IndexFlatL2` when
both use the same BLAS backend (OpenBLAS or MKL). Does not require
FAISS, cmake, or C++ compilation — only a system BLAS library
(`libopenblas-dev` or `libmkl-dev`).

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<crate::pipeline::command::ResourceDesc> {
        vec![
            crate::pipeline::command::ResourceDesc {
                name: "mem".into(),
                description: "Score matrix (query_batch x n_base floats)".into(),
                adjustable: true,
            },
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
        let use_ip = match metric_str.to_uppercase().as_str() {
            "IP" | "DOT_PRODUCT" | "COSINE" => true,
            "L2" => false,
            other => return error_result(format!("unsupported metric: '{}'", other), start),
        };

        let base_source = match resolve_source(base_str, &ctx.workspace) {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
        let distances_path = options
            .get("distances")
            .map(|s| resolve_path(s, &ctx.workspace));

        for path in std::iter::once(&indices_path).chain(distances_path.iter()) {
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
        }

        // Open readers
        ctx.ui.log(&format!("  opening base vectors: {}", base_path.display()));
        let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open base: {}", e), start),
        };
        ctx.ui.log(&format!("  opening query vectors: {}", query_path.display()));
        let query_reader = match MmapVectorReader::<f32>::open_fvec(&query_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open query: {}", e), start),
        };

        let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&base_reader);
        let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
        let base_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&base_reader);
        let query_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&query_reader);

        let (base_offset, base_n) = match base_source.window {
            Some((ws, we)) => (ws.min(base_count), we.min(base_count).saturating_sub(ws.min(base_count))),
            None => (0, base_count),
        };

        if base_dim != query_dim {
            return error_result(format!("dimension mismatch: base={} query={}", base_dim, query_dim), start);
        }
        let dim = base_dim;

        ctx.ui.log(&format!(
            "  base: {} vectors, query: {} vectors, dim: {}, metric: {} (BLAS sgemm)",
            base_n, query_count, dim, metric_str
        ));

        // Load base vectors into contiguous buffer
        ctx.ui.log("  loading base vectors...");
        let mut base_data: Vec<f32> = Vec::with_capacity(base_n * dim);
        for i in 0..base_n {
            base_data.extend_from_slice(base_reader.get_slice(base_offset + i));
        }

        // Precompute base norms for L2 metric
        let base_norms_sq: Option<Vec<f32>> = if !use_ip {
            Some(
                (0..base_n)
                    .map(|i| {
                        let s = &base_data[i * dim..(i + 1) * dim];
                        s.iter().map(|v| v * v).sum::<f32>()
                    })
                    .collect()
            )
        } else {
            None
        };

        // Process queries in batches to limit score matrix memory.
        // Score matrix: batch_size x base_n floats.
        // Target ~2 GB max for the score matrix.
        let max_score_bytes: usize = 2 * 1024 * 1024 * 1024;
        let batch_size = (max_score_bytes / (base_n * 4)).max(1).min(query_count);

        ctx.ui.log(&format!("  query batch size: {} (score matrix: {:.0} MB per batch)",
            batch_size, (batch_size * base_n * 4) as f64 / (1024.0 * 1024.0)));

        let mut idx_file = match std::fs::File::create(&indices_path) {
            Ok(f) => std::io::BufWriter::with_capacity(1 << 20, f),
            Err(e) => return error_result(format!("create {}: {}", indices_path.display(), e), start),
        };
        let mut dist_file = match &distances_path {
            Some(dp) => match std::fs::File::create(dp) {
                Ok(f) => Some(std::io::BufWriter::with_capacity(1 << 20, f)),
                Err(e) => return error_result(format!("create {}: {}", dp.display(), e), start),
            },
            None => None,
        };

        let dim_bytes = (k as i32).to_le_bytes();
        let mut total_processed = 0usize;

        ctx.ui.log(&format!("  computing KNN: {} queries x {} base x dim {} ...",
            query_count, base_n, dim));

        for batch_start in (0..query_count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(query_count);
            let batch_n = batch_end - batch_start;

            // Load query batch
            let mut query_batch: Vec<f32> = Vec::with_capacity(batch_n * dim);
            for i in batch_start..batch_end {
                query_batch.extend_from_slice(query_reader.get_slice(i));
            }

            // Allocate score matrix
            let mut scores: Vec<f32> = vec![0.0f32; batch_n * base_n];

            // sgemm: scores = query_batch @ base_data.T
            // OpenBLAS sgemm is internally multi-threaded — this is the
            // compute-intensive step and benefits from all CPU cores.
            let sgemm_start = Instant::now();
            if use_ip {
                unsafe {
                    cblas_sgemm(
                        CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                        batch_n as i32, base_n as i32, dim as i32,
                        1.0,
                        query_batch.as_ptr(), dim as i32,
                        base_data.as_ptr(), dim as i32,
                        0.0,
                        scores.as_mut_ptr(), base_n as i32,
                    );
                }
            } else {
                // L2: ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q.b
                unsafe {
                    cblas_sgemm(
                        CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                        batch_n as i32, base_n as i32, dim as i32,
                        -2.0,
                        query_batch.as_ptr(), dim as i32,
                        base_data.as_ptr(), dim as i32,
                        0.0,
                        scores.as_mut_ptr(), base_n as i32,
                    );
                }
                let base_norms = base_norms_sq.as_ref().unwrap();
                for qi in 0..batch_n {
                    let q_slice = &query_batch[qi * dim..(qi + 1) * dim];
                    let q_norm_sq: f32 = q_slice.iter().map(|v| v * v).sum();
                    let row = &mut scores[qi * base_n..(qi + 1) * base_n];
                    for (bi, score) in row.iter_mut().enumerate() {
                        *score += q_norm_sq + base_norms[bi];
                    }
                }
                for s in scores.iter_mut() {
                    *s = -*s;
                }
            }
            let sgemm_secs = sgemm_start.elapsed().as_secs_f64();

            // Parallel top-k selection for each query in batch.
            // Each query's row is independent — parallelize with rayon.
            let topk_start = Instant::now();
            let batch_topk: Vec<Vec<Neighbor>> = {
                use rayon::prelude::*;
                (0..batch_n)
                    .into_par_iter()
                    .map(|qi| {
                        let row = &scores[qi * base_n..(qi + 1) * base_n];
                        topk_indices(row, k)
                    })
                    .collect()
            };
            let topk_secs = topk_start.elapsed().as_secs_f64();

            // Write results sequentially (I/O is fast relative to compute)
            for qi in 0..batch_n {
                let topk = &batch_topk[qi];

                idx_file.write_all(&dim_bytes).map_err(|e| e.to_string()).unwrap();
                for j in 0..k {
                    let idx: i32 = if j < topk.len() {
                        topk[j].index as i32
                    } else {
                        -1
                    };
                    idx_file.write_all(&idx.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                }

                if let Some(ref mut dw) = dist_file {
                    dw.write_all(&dim_bytes).map_err(|e| e.to_string()).unwrap();
                    for j in 0..k {
                        let dist: f32 = if j < topk.len() {
                            if use_ip { topk[j].score } else { -topk[j].score }
                        } else {
                            f32::INFINITY
                        };
                        dw.write_all(&dist.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                    }
                }
            }

            total_processed += batch_n;
            ctx.ui.log(&format!("  batch {}/{}: sgemm {:.1}s, topk {:.1}s ({} queries)",
                total_processed, query_count, sgemm_secs, topk_secs, batch_n));
        }

        idx_file.flush().map_err(|e| e.to_string()).unwrap();
        if let Some(ref mut dw) = dist_file {
            dw.flush().map_err(|e| e.to_string()).unwrap();
        }

        let mut produced = vec![indices_path.clone()];
        if let Some(ref dp) = distances_path {
            produced.push(dp.clone());
        }

        let elapsed = start.elapsed();
        ctx.ui.log(&format!(
            "  BLAS KNN complete: {} queries x {} neighbors in {:.1}s",
            query_count, k, elapsed.as_secs_f64()
        ));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} queries, k={}, {} base vectors, metric={}, engine=BLAS-sgemm",
                query_count, k, base_n, metric_str,
            ),
            elapsed,
            produced,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "base".into(), type_name: "Path".into(), required: true, default: None,
                description: "Base vectors file (fvec only)".into(), role: OptionRole::Input },
            OptionDesc { name: "query".into(), type_name: "Path".into(), required: true, default: None,
                description: "Query vectors file (fvec only)".into(), role: OptionRole::Input },
            OptionDesc { name: "indices".into(), type_name: "Path".into(), required: true, default: None,
                description: "Output neighbor indices (ivec)".into(), role: OptionRole::Output },
            OptionDesc { name: "distances".into(), type_name: "Path".into(), required: false, default: None,
                description: "Output neighbor distances (fvec)".into(), role: OptionRole::Output },
            OptionDesc { name: "neighbors".into(), type_name: "int".into(), required: true, default: None,
                description: "Number of nearest neighbors (k)".into(), role: OptionRole::Config },
            OptionDesc { name: "metric".into(), type_name: "enum".into(), required: false,
                default: Some("L2".into()),
                description: "Distance metric: L2, DOT_PRODUCT, COSINE, IP".into(), role: OptionRole::Config },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query"],
            &["indices", "distances"],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::{Options, Status, StreamContext};
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use vectordata::VectorReader;
    use vectordata::io::MmapVectorReader;

    fn make_ctx(workspace: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(), profile: String::new(), profile_names: vec![],
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"), cache: workspace.join(".cache"),
            defaults: IndexMap::new(), dry_run: false, progress: ProgressLog::new(),
            threads: 1, step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1), estimated_total_steps: 0,
        }
    }

    fn write_fvec(path: &std::path::Path, vectors: &[Vec<f32>]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            f.write_all(&(v.len() as i32).to_le_bytes()).unwrap();
            for &val in v { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    fn read_ivec_rows(path: &std::path::Path) -> Vec<Vec<i32>> {
        let reader = MmapVectorReader::<i32>::open_ivec(path).unwrap();
        let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
        (0..count).map(|i| reader.get_slice(i).to_vec()).collect()
    }

    #[test]
    fn test_knn_blas_ip_known_neighbors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![
            vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.7, 0.7, 0.0],
            vec![0.0, 0.0, 1.0], vec![0.5, 0.5, 0.5],
        ];
        let query = vec![vec![0.9, 0.1, 0.0]];
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("idx.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "3");
        opts.set("metric", "IP");

        let mut op = ComputeKnnBlasOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&tmp.path().join("idx.ivec"));
        // IP: base[0]=0.9, base[2]=0.7, base[4]=0.5
        assert_eq!(rows[0][0], 0);
        assert_eq!(rows[0][1], 2);
        assert_eq!(rows[0][2], 4);
    }

    #[test]
    fn test_knn_blas_l2() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let query = vec![vec![0.1, 0.1]];
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("idx.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "2");
        opts.set("metric", "L2");

        let mut op = ComputeKnnBlasOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&tmp.path().join("idx.ivec"));
        assert_eq!(rows[0][0], 0, "L2 nearest to (0.1,0.1) should be (0,0)");
    }

    #[test]
    fn test_knn_blas_multiple_queries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let query = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("idx.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "1");
        opts.set("metric", "IP");

        let mut op = ComputeKnnBlasOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&tmp.path().join("idx.ivec"));
        assert_eq!(rows[0][0], 0);
        assert_eq!(rows[1][0], 1);
        assert_eq!(rows[2][0], 2);
    }
}
