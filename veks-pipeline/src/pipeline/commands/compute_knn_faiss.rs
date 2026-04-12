// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute exact KNN using FAISS.
//!
//! This command provides the `compute knn-faiss` pipeline step, which
//! computes brute-force exact K-nearest neighbors using Facebook's FAISS
//! library (via the `faiss` Rust crate). It is intended for
//! cross-validation against the knn\_utils Python pipeline, which uses
//! the same FAISS C++ library under the hood.
//!
//! The command accepts the same options as `compute knn` (base, query,
//! indices, distances, neighbors, metric) and produces identical output
//! formats (ivec/fvec). Because FAISS and SimSIMD use different SIMD
//! implementations internally, floating-point results may differ at the
//! ULP level — but for exact brute-force search on normalized vectors
//! the neighbor index sets should be identical (modulo tie-breaking).
//!
//! Only f32 (fvec) inputs are supported, matching knn\_utils which
//! operates exclusively on float32.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use faiss::Index;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use super::source_window::resolve_source;

/// Pipeline command: compute exact KNN ground truth via FAISS.
pub struct ComputeKnnFaissOp;

/// Creates a boxed `ComputeKnnFaissOp` for command registration.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeKnnFaissOp)
}

/// Map our metric names to FAISS metric types.
///
/// FAISS supports `METRIC_INNER_PRODUCT` (0) and `METRIC_L2` (1).
/// For cosine similarity on normalized vectors, inner product is
/// equivalent — matching what knn\_utils does with `--metric ip`.
fn faiss_metric(metric_str: &str) -> Result<faiss::MetricType, String> {
    match metric_str.to_uppercase().as_str() {
        "L2" => Ok(faiss::MetricType::L2),
        "IP" | "DOT_PRODUCT" | "COSINE" => Ok(faiss::MetricType::InnerProduct),
        other => Err(format!(
            "unsupported FAISS metric: '{}'. Use L2, DOT_PRODUCT, COSINE, or IP",
            other
        )),
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

impl CommandOp for ComputeKnnFaissOp {
    fn command_path(&self) -> &str {
        "compute knn-faiss"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Brute-force exact KNN via FAISS (knn_utils compatibility)".into(),
            body: format!(r#"# compute knn-faiss

Brute-force exact K-nearest-neighbor ground truth computation using
Facebook's FAISS library.

## Description

This command is the FAISS-backed equivalent of `compute knn`. It uses
FAISS `IndexFlatL2` or `IndexFlatIP` for exhaustive brute-force search,
matching the behavior of the knn\_utils Python pipeline.

Use this command with `--personality knn_utils` to produce ground truth
that is byte-identical to knn\_utils output, enabling cross-validation
between the two pipelines.

Only f32 (fvec) inputs are supported.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description: "FAISS index holds all base vectors in RAM".into(),
                adjustable: false,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // -- Parse options -----------------------------------------------
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
        let faiss_mt = match faiss_metric(metric_str) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };

        // -- Resolve paths -----------------------------------------------
        let base_source = match resolve_source(base_str, &ctx.workspace) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
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

        // -- Read vectors via mmap ---------------------------------------
        ctx.ui.log(&format!("  opening base vectors: {}", base_path.display()));
        let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => return error_result(
                format!("failed to open base {}: {}", base_path.display(), e),
                start,
            ),
        };
        ctx.ui.log(&format!("  opening query vectors: {}", query_path.display()));
        let query_reader = match MmapVectorReader::<f32>::open_fvec(&query_path) {
            Ok(r) => r,
            Err(e) => return error_result(
                format!("failed to open query {}: {}", query_path.display(), e),
                start,
            ),
        };

        let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&base_reader);
        let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
        let base_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&base_reader);
        let query_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&query_reader);

        // Apply window to base vectors
        let (base_offset, base_n) = match base_source.window {
            Some((ws, we)) => {
                let ws = ws.min(base_count);
                let we = we.min(base_count);
                (ws, we.saturating_sub(ws))
            }
            None => (0, base_count),
        };

        if base_dim != query_dim {
            return error_result(
                format!(
                    "dimension mismatch: base dim={} vs query dim={}",
                    base_dim, query_dim
                ),
                start,
            );
        }

        let d = base_dim as u32;
        ctx.ui.log(&format!(
            "  base: {} vectors, query: {} vectors, dim: {}, metric: {} (FAISS)",
            base_n, query_count, d, metric_str
        ));

        // -- Load base vectors into contiguous buffer --------------------
        ctx.ui.log("  loading base vectors into memory...");
        let mut base_data: Vec<f32> = Vec::with_capacity(base_n * base_dim);
        for i in 0..base_n {
            base_data.extend_from_slice(base_reader.get_slice(base_offset + i));
        }

        // -- Build FAISS index -------------------------------------------
        ctx.ui.log("  building FAISS index...");
        let mut index = match faiss::index::flat::FlatIndex::new(d, faiss_mt) {
            Ok(idx) => idx,
            Err(e) => return error_result(format!("FAISS index creation failed: {}", e), start),
        };
        if let Err(e) = index.add(&base_data) {
            return error_result(format!("FAISS index.add failed: {}", e), start);
        }
        ctx.ui.log(&format!("  FAISS index built, ntotal={}", index.ntotal()));

        // -- Load query vectors ------------------------------------------
        ctx.ui.log("  loading query vectors...");
        let mut query_data: Vec<f32> = Vec::with_capacity(query_count * query_dim);
        for i in 0..query_count {
            query_data.extend_from_slice(query_reader.get_slice(i));
        }

        // -- Search ------------------------------------------------------
        ctx.ui.log(&format!("  searching k={} neighbors for {} queries...", k, query_count));
        let result = match index.search(&query_data, k) {
            Ok(r) => r,
            Err(e) => return error_result(format!("FAISS search failed: {}", e), start),
        };

        // -- Write output ------------------------------------------------
        ctx.ui.log(&format!("  writing indices to {}", indices_path.display()));
        let mut idx_file = match std::fs::File::create(&indices_path) {
            Ok(f) => std::io::BufWriter::new(f),
            Err(e) => return error_result(
                format!("create {}: {}", indices_path.display(), e), start,
            ),
        };
        let mut dist_file = match &distances_path {
            Some(dp) => {
                ctx.ui.log(&format!("  writing distances to {}", dp.display()));
                match std::fs::File::create(dp) {
                    Ok(f) => Some(std::io::BufWriter::new(f)),
                    Err(e) => return error_result(
                        format!("create {}: {}", dp.display(), e), start,
                    ),
                }
            }
            None => None,
        };

        let dim_bytes = (k as i32).to_le_bytes();
        for q in 0..query_count {
            // Write ivec record: [k:i32, idx0:i32, idx1:i32, ...]
            if let Err(e) = idx_file.write_all(&dim_bytes) {
                return error_result(format!("write indices: {}", e), start);
            }
            for j in 0..k {
                let label = result.labels[q * k + j];
                let idx: i32 = match label.get() {
                    Some(v) => v as i32,
                    None => -1,
                };
                if let Err(e) = idx_file.write_all(&idx.to_le_bytes()) {
                    return error_result(format!("write indices: {}", e), start);
                }
            }

            // Write fvec record: [k:f32-as-i32, dist0:f32, dist1:f32, ...]
            if let Some(ref mut dw) = dist_file {
                if let Err(e) = dw.write_all(&dim_bytes) {
                    return error_result(format!("write distances: {}", e), start);
                }
                for j in 0..k {
                    let dist = result.distances[q * k + j];
                    if let Err(e) = dw.write_all(&dist.to_le_bytes()) {
                        return error_result(format!("write distances: {}", e), start);
                    }
                }
            }
        }

        if let Err(e) = idx_file.flush() {
            return error_result(format!("flush indices: {}", e), start);
        }
        if let Some(ref mut dw) = dist_file {
            if let Err(e) = dw.flush() {
                return error_result(format!("flush distances: {}", e), start);
            }
        }

        let mut produced = vec![indices_path.clone()];
        if let Some(ref dp) = distances_path {
            produced.push(dp.clone());
        }

        let elapsed = start.elapsed();
        ctx.ui.log(&format!(
            "  FAISS KNN complete: {} queries x {} neighbors in {:.1}s",
            query_count, k, elapsed.as_secs_f64()
        ));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} queries, k={}, {} base vectors, metric={}, engine=FAISS",
                query_count, k, base_n, metric_str,
            ),
            elapsed,
            produced,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Base vectors file (fvec only)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "query".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Query vectors file (fvec only)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "indices".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output neighbor indices (ivec)".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "distances".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output neighbor distances (fvec)".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "neighbors".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of nearest neighbors (k)".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "metric".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("L2".to_string()),
                description: "Distance metric: L2, DOT_PRODUCT, COSINE, IP".to_string(),
                role: OptionRole::Config,
            },
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
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
        }
    }

    fn write_fvec(path: &std::path::Path, vectors: &[Vec<f32>]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    fn read_ivec_rows(path: &std::path::Path) -> Vec<Vec<i32>> {
        let reader = MmapVectorReader::<i32>::open_ivec(path).unwrap();
        let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
        (0..count).map(|i| reader.get_slice(i).to_vec()).collect()
    }

    /// Compute KNN with inner product on a small synthetic dataset
    /// with known correct neighbors.
    #[test]
    fn test_knn_faiss_ip_known_neighbors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        // 5 base vectors in 3D (unit-ish for IP)
        let base = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.7, 0.7, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let query = vec![
            vec![0.9, 0.1, 0.0],  // closest to base[0] (IP=0.9), then base[2] (0.7)
        ];
        let base_path = tmp.path().join("base.fvec");
        let query_path = tmp.path().join("query.fvec");
        write_fvec(&base_path, &base);
        write_fvec(&query_path, &query);

        let indices_path = tmp.path().join("indices.ivec");
        let distances_path = tmp.path().join("distances.fvec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("distances", distances_path.to_string_lossy().to_string());
        opts.set("neighbors", "3");
        opts.set("metric", "IP");

        let mut op = ComputeKnnFaissOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&indices_path);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 3);

        // IP scores: base[0]=0.9, base[1]=0.1, base[2]=0.7*0.9+0.7*0.1=0.7,
        //            base[3]=0.0, base[4]=0.5*0.9+0.5*0.1=0.5
        // Top 3: base[0] (0.9), base[2] (0.7), base[4] (0.5)
        assert_eq!(rows[0][0], 0, "top-1 should be base[0]");
        assert_eq!(rows[0][1], 2, "top-2 should be base[2]");
        assert_eq!(rows[0][2], 4, "top-3 should be base[4]");
    }

    /// Compute KNN with L2 metric.
    #[test]
    fn test_knn_faiss_l2() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let query = vec![
            vec![0.1, 0.1],  // closest to [0,0]
        ];
        let base_path = tmp.path().join("base.fvec");
        let query_path = tmp.path().join("query.fvec");
        write_fvec(&base_path, &base);
        write_fvec(&query_path, &query);

        let indices_path = tmp.path().join("indices.ivec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("neighbors", "2");
        opts.set("metric", "L2");

        let mut op = ComputeKnnFaissOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&indices_path);
        assert_eq!(rows[0][0], 0, "L2 top-1 should be [0,0]");
    }

    /// Verify output ivec format is correct (dim header + k values per row).
    #[test]
    fn test_knn_faiss_output_format() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![vec![1.0, 2.0]; 10];
        let query = vec![vec![1.0, 2.0]; 3];
        let base_path = tmp.path().join("base.fvec");
        let query_path = tmp.path().join("query.fvec");
        write_fvec(&base_path, &base);
        write_fvec(&query_path, &query);

        let indices_path = tmp.path().join("indices.ivec");
        let distances_path = tmp.path().join("distances.fvec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("distances", distances_path.to_string_lossy().to_string());
        opts.set("neighbors", "5");
        opts.set("metric", "IP");

        let mut op = ComputeKnnFaissOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // 3 queries x (4 + 5*4) = 3 * 24 = 72 bytes for indices
        let idx_size = std::fs::metadata(&indices_path).unwrap().len();
        assert_eq!(idx_size, 3 * (4 + 5 * 4));

        // distances: same layout with f32
        let dist_size = std::fs::metadata(&distances_path).unwrap().len();
        assert_eq!(dist_size, 3 * (4 + 5 * 4));

        // All indices should be in [0, 10)
        let rows = read_ivec_rows(&indices_path);
        for (qi, row) in rows.iter().enumerate() {
            for &idx in row {
                assert!(idx >= 0 && idx < 10,
                    "query {} has out-of-range index {}", qi, idx);
            }
        }
    }

    /// Adversarial: k > base count. FAISS pads with -1.
    #[test]
    fn test_knn_faiss_k_exceeds_base() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let query = vec![vec![1.5, 3.0]];
        let base_path = tmp.path().join("base.fvec");
        let query_path = tmp.path().join("query.fvec");
        write_fvec(&base_path, &base);
        write_fvec(&query_path, &query);

        let indices_path = tmp.path().join("indices.ivec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("neighbors", "5");
        opts.set("metric", "IP");

        let mut op = ComputeKnnFaissOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&indices_path);
        assert_eq!(rows[0].len(), 5);
        // First 2 should be valid indices, rest should be -1
        assert!(rows[0][0] >= 0 && rows[0][0] < 2);
        assert!(rows[0][1] >= 0 && rows[0][1] < 2);
        assert_eq!(rows[0][2], -1, "padding should be -1");
        assert_eq!(rows[0][3], -1);
        assert_eq!(rows[0][4], -1);
    }

    /// Multiple queries with different nearest neighbors.
    #[test]
    fn test_knn_faiss_multiple_queries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![
            vec![1.0, 0.0, 0.0],  // nearest: base[0]
            vec![0.0, 1.0, 0.0],  // nearest: base[1]
            vec![0.0, 0.0, 1.0],  // nearest: base[2]
        ];
        let base_path = tmp.path().join("base.fvec");
        let query_path = tmp.path().join("query.fvec");
        write_fvec(&base_path, &base);
        write_fvec(&query_path, &query);

        let indices_path = tmp.path().join("indices.ivec");

        let mut opts = Options::new();
        opts.set("base", base_path.to_string_lossy().to_string());
        opts.set("query", query_path.to_string_lossy().to_string());
        opts.set("indices", indices_path.to_string_lossy().to_string());
        opts.set("neighbors", "1");
        opts.set("metric", "IP");

        let mut op = ComputeKnnFaissOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&indices_path);
        assert_eq!(rows[0][0], 0, "query 0 nearest should be base 0");
        assert_eq!(rows[1][0], 1, "query 1 nearest should be base 1");
        assert_eq!(rows[2][0], 2, "query 2 nearest should be base 2");
    }
}
