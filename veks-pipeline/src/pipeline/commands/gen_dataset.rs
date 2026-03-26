// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate a complete test dataset.
//!
//! Creates a full dataset directory containing:
//! - `base.fvec` — base vectors
//! - `query.fvec` — query vectors
//! - `indices.ivec` — ground truth nearest-neighbor indices
//! - `distances.fvec` — ground truth distances
//! - `dataset.yaml` — descriptor file
//!
//! Uses SIMD-accelerated distance computation for ground truth KNN.
//!
//! Equivalent to the Java `CMD_generate_dataset` command.

use std::collections::BinaryHeap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::Rng;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;
use crate::pipeline::simd_distance;

/// Pipeline command: generate complete dataset.
pub struct GenerateDatasetOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateDatasetOp)
}

impl CommandOp for GenerateDatasetOp {
    fn command_path(&self) -> &str {
        "generate dataset"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate a complete synthetic dataset".into(),
            body: format!(r#"# generate dataset

Generate a complete synthetic dataset.

## Description

Creates a fully self-contained dataset directory in a single command by
orchestrating multiple generation steps internally. The output directory
contains everything needed for ANN benchmarking:

- `base.fvec` -- the base (corpus) vectors to index
- `query.fvec` -- the query vectors to search with
- `indices.ivec` -- ground truth K-nearest-neighbor indices for each query
- `distances.fvec` -- ground truth distances corresponding to each neighbor
- `dataset.yaml` -- a descriptor file with metadata (dimension, counts,
  distance metric, seed, etc.)

## How it works

1. **Base vectors** are generated as uniformly random f32 vectors in the
   configured `[min, max)` range.
2. **Query vectors** are generated independently from the same distribution
   (they are *not* drawn from the base set).
3. **Ground truth KNN** is computed by brute-force exact search using
   SIMD-accelerated distance functions. For each query, all base vectors
   are scored and the top-K nearest neighbors (by the configured distance
   metric) are retained. Supported metrics include L2, Cosine, DotProduct,
   and L1.

If a `dataset.yaml` already exists in the output directory, the command
will refuse to overwrite unless `force=true`.

## Role in dataset pipelines

This is a convenience command for quickly producing small to mid-sized
test datasets from scratch. It is the fastest way to get a complete
dataset for development and integration testing.

For larger or more realistic datasets, the individual pipeline steps
(`generate vectors`, `generate ivec-shuffle`, `transform fvec-extract`,
and a separate KNN computation) give more control over the generation
process -- for example, allowing self-search splits, model-based vector
generation, or half-precision storage.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let output_dir_str = match options.require("output-dir") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let dimension: usize = match options.require("dimension") {
            Ok(s) => match s.parse() {
                Ok(d) if d > 0 => d,
                _ => return error_result(format!("invalid dimension: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let base_count: usize = options
            .get("base-count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);
        let query_count: usize = options
            .get("query-count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000);
        let k: usize = options
            .get("neighbors")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let distance = options.get("distance").unwrap_or("L2");
        let force = options.get("force").map_or(false, |s| s == "true");
        let seed = rng::parse_seed(options.get("seed"));
        let min_val: f32 = options
            .get("min")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let max_val: f32 = options
            .get("max")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        let model = options.get("model").unwrap_or("synthetic");
        let license = options.get("license").unwrap_or("Apache-2.0");
        let vendor = options.get("vendor").unwrap_or("veks");

        let output_dir = resolve_path(output_dir_str, &ctx.workspace);
        let dataset_yaml = output_dir.join("dataset.yaml");

        // Check existing
        if dataset_yaml.exists() && !force {
            return error_result(
                format!(
                    "dataset.yaml already exists in {}. Use force=true to overwrite.",
                    output_dir.display()
                ),
                start,
            );
        }

        // Create directory
        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            return error_result(format!("failed to create directory: {}", e), start);
        }

        let mut rng_inst = rng::seeded_rng(seed);

        // Generate base vectors
        let base_path = output_dir.join("base.fvec");
        ctx.ui.log(&format!(
            "  Generating {} base vectors (dim={})...",
            base_count, dimension
        ));
        if let Err(e) = generate_fvec(&base_path, base_count, dimension, min_val, max_val, &mut rng_inst) {
            return error_result(format!("base generation failed: {}", e), start);
        }

        // Generate query vectors
        let query_path = output_dir.join("query.fvec");
        ctx.ui.log(&format!(
            "  Generating {} query vectors (dim={})...",
            query_count, dimension
        ));
        if let Err(e) = generate_fvec(&query_path, query_count, dimension, min_val, max_val, &mut rng_inst) {
            return error_result(format!("query generation failed: {}", e), start);
        }

        // Compute ground truth KNN
        let indices_path = output_dir.join("indices.ivec");
        let distances_path = output_dir.join("distances.fvec");
        ctx.ui.log(&format!(
            "  Computing {}-NN ground truth ({} queries × {} base)...",
            k, query_count, base_count
        ));
        let effective_k = k.min(base_count);
        if let Err(e) = compute_ground_truth(
            &base_path,
            &query_path,
            &indices_path,
            &distances_path,
            effective_k,
            distance,
        ) {
            return error_result(format!("ground truth computation failed: {}", e), start);
        }

        // Write dataset.yaml
        let yaml_content = format!(
            "name: generated-dataset\n\
             description: Synthetic dataset generated by veks\n\
             \n\
             attributes:\n\
             \x20 model: {}\n\
             \x20 distance_function: {}\n\
             \x20 dimension: {}\n\
             \x20 base_count: {}\n\
             \x20 query_count: {}\n\
             \x20 neighbors: {}\n\
             \x20 vector_type: float[]\n\
             \x20 format: xvec\n\
             \x20 license: {}\n\
             \x20 vendor: {}\n\
             \x20 generation_seed: {}\n\
             \n\
             profiles:\n\
             \x20 default:\n\
             \x20   base_vectors: base.fvec\n\
             \x20   query_vectors: query.fvec\n\
             \x20   neighbor_indices: indices.ivec\n\
             \x20   neighbor_distances: distances.fvec\n",
            model, distance, dimension, base_count, query_count, effective_k,
            license, vendor, seed,
        );

        if let Err(e) = std::fs::write(&dataset_yaml, &yaml_content) {
            return error_result(format!("failed to write dataset.yaml: {}", e), start);
        }

        ctx.ui.log(&format!("  Dataset created: {}", output_dir.display()));

        // Write verified counts for xvec output files
        for (path, cnt) in [
            (&base_path, base_count),
            (&query_path, query_count),
            (&indices_path, query_count),
            (&distances_path, query_count),
        ] {
            let var_name = format!("verified_count:{}",
                path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &cnt.to_string());
            ctx.defaults.insert(var_name, cnt.to_string());
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "generated dataset: {} base, {} query, k={}, dim={} in {}",
                base_count, query_count, effective_k, dimension, output_dir.display()
            ),
            produced: vec![
                base_path,
                query_path,
                indices_path,
                distances_path,
                dataset_yaml,
            ],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "output-dir".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output directory for dataset files".to_string(),
                        role: OptionRole::Output,
        },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Vector dimensionality".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "base-count".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("10000".to_string()),
                description: "Number of base vectors".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "query-count".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1000".to_string()),
                description: "Number of query vectors".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "neighbors".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("100".to_string()),
                description: "Ground truth neighbor count (k)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "distance".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("L2".to_string()),
                description: "Distance metric: L2, Cosine, DotProduct, L1".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "PRNG seed (default: time-based)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "min".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.0".to_string()),
                description: "Minimum value for uniform distribution".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "max".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("1.0".to_string()),
                description: "Maximum value for uniform distribution".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "force".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Overwrite existing dataset".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "model".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("synthetic".to_string()),
                description: "Model name metadata".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "license".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("Apache-2.0".to_string()),
                description: "License metadata".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "vendor".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("veks".to_string()),
                description: "Vendor metadata".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Generate fvec file with random f32 vectors.
fn generate_fvec(
    path: &Path,
    count: usize,
    dim: usize,
    min: f32,
    max: f32,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    let mut buf = std::io::BufWriter::new(&mut file);

    let dim_bytes = (dim as i32).to_le_bytes();
    let range = max - min;

    for _ in 0..count {
        buf.write_all(&dim_bytes)
            .map_err(|e| format!("write error: {}", e))?;
        for _ in 0..dim {
            let val: f32 = min + rng.random::<f32>() * range;
            buf.write_all(&val.to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
    }

    buf.flush().map_err(|e| format!("flush error: {}", e))?;
    Ok(())
}

/// Compute exact KNN ground truth using SIMD-accelerated distance functions.
fn compute_ground_truth(
    base_path: &Path,
    query_path: &Path,
    indices_path: &Path,
    distances_path: &Path,
    k: usize,
    distance_name: &str,
) -> Result<(), String> {
    let metric = simd_distance::Metric::from_str(distance_name)
        .ok_or_else(|| format!("unknown distance metric: {}", distance_name))?;
    let dist_fn = simd_distance::select_distance_fn(metric);

    let base = MmapVectorReader::<f32>::open_fvec(base_path)
        .map_err(|e| format!("failed to open base {}: {}", base_path.display(), e))?;
    let query = MmapVectorReader::<f32>::open_fvec(query_path)
        .map_err(|e| format!("failed to open query {}: {}", query_path.display(), e))?;

    let base_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&base);
    let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query);

    let mut idx_file = std::io::BufWriter::new(
        std::fs::File::create(indices_path)
            .map_err(|e| format!("failed to create indices: {}", e))?,
    );
    let mut dist_file = std::io::BufWriter::new(
        std::fs::File::create(distances_path)
            .map_err(|e| format!("failed to create distances: {}", e))?,
    );

    let k_bytes = (k as i32).to_le_bytes();

    for qi in 0..query_count {
        let qvec = query.get(qi).map_err(|e| format!("read query {}: {}", qi, e))?;

        // Max-heap for top-k (smallest distances)
        let mut heap: BinaryHeap<DistEntry> = BinaryHeap::new();

        for bi in 0..base_count {
            let bvec = base.get(bi).map_err(|e| format!("read base {}: {}", bi, e))?;
            let d = dist_fn(&qvec, &bvec);

            if heap.len() < k {
                heap.push(DistEntry { dist: d, idx: bi });
            } else if let Some(top) = heap.peek() {
                if d < top.dist {
                    heap.pop();
                    heap.push(DistEntry { dist: d, idx: bi });
                }
            }
        }

        // Sort by distance ascending
        let mut results: Vec<DistEntry> = heap.into_sorted_vec();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));

        // Write indices
        idx_file
            .write_all(&k_bytes)
            .map_err(|e| format!("write error: {}", e))?;
        for r in &results {
            idx_file
                .write_all(&(r.idx as i32).to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
        for _ in results.len()..k {
            idx_file
                .write_all(&(-1i32).to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }

        // Write distances
        dist_file
            .write_all(&k_bytes)
            .map_err(|e| format!("write error: {}", e))?;
        for r in &results {
            dist_file
                .write_all(&r.dist.to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
        for _ in results.len()..k {
            dist_file
                .write_all(&f32::MAX.to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
    }

    idx_file.flush().map_err(|e| format!("flush error: {}", e))?;
    dist_file.flush().map_err(|e| format!("flush error: {}", e))?;

    Ok(())
}

#[derive(Debug, Clone)]
struct DistEntry {
    dist: f32,
    idx: usize,
}

impl PartialEq for DistEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for DistEntry {}

impl PartialOrd for DistEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for max-heap (we want smallest distances at top)
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
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
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_generate_dataset_small() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let out_dir = ws.join("test-dataset");
        let mut opts = Options::new();
        opts.set("output-dir", out_dir.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("base-count", "20");
        opts.set("query-count", "5");
        opts.set("neighbors", "3");
        opts.set("seed", "42");

        let mut op = GenerateDatasetOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);

        // Check all files exist
        assert!(out_dir.join("base.fvec").exists());
        assert!(out_dir.join("query.fvec").exists());
        assert!(out_dir.join("indices.ivec").exists());
        assert!(out_dir.join("distances.fvec").exists());
        assert!(out_dir.join("dataset.yaml").exists());

        // Verify base file dimensions
        let reader = MmapVectorReader::<f32>::open_fvec(&out_dir.join("base.fvec")).unwrap();
        assert_eq!(
            <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader),
            20
        );
        assert_eq!(
            <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader),
            4
        );

        // Verify indices file
        let idx_reader = MmapVectorReader::<i32>::open_ivec(&out_dir.join("indices.ivec")).unwrap();
        assert_eq!(
            <MmapVectorReader<i32> as VectorReader<i32>>::count(&idx_reader),
            5
        );
        assert_eq!(
            <MmapVectorReader<i32> as VectorReader<i32>>::dim(&idx_reader),
            3
        );
    }

    #[test]
    fn test_generate_dataset_no_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let out_dir = ws.join("test-dataset");
        std::fs::create_dir_all(&out_dir).unwrap();
        std::fs::write(out_dir.join("dataset.yaml"), "existing").unwrap();

        let mut opts = Options::new();
        opts.set("output-dir", out_dir.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("base-count", "10");
        opts.set("query-count", "5");
        opts.set("neighbors", "3");

        let mut op = GenerateDatasetOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("already exists"));
    }

    #[test]
    fn test_generate_dataset_force_overwrite() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let out_dir = ws.join("test-dataset");
        std::fs::create_dir_all(&out_dir).unwrap();
        std::fs::write(out_dir.join("dataset.yaml"), "existing").unwrap();

        let mut opts = Options::new();
        opts.set("output-dir", out_dir.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("base-count", "10");
        opts.set("query-count", "5");
        opts.set("neighbors", "3");
        opts.set("force", "true");
        opts.set("seed", "123");

        let mut op = GenerateDatasetOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);
    }

    #[test]
    fn test_dataset_yaml_content() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let out_dir = ws.join("test-dataset");
        let mut opts = Options::new();
        opts.set("output-dir", out_dir.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("base-count", "10");
        opts.set("query-count", "5");
        opts.set("neighbors", "3");
        opts.set("seed", "42");
        opts.set("distance", "COSINE");

        let mut op = GenerateDatasetOp;
        op.execute(&opts, &mut ctx);

        let yaml = std::fs::read_to_string(out_dir.join("dataset.yaml")).unwrap();
        assert!(yaml.contains("distance_function: COSINE"));
        assert!(yaml.contains("dimension: 8"));
        assert!(yaml.contains("base_count: 10"));
        assert!(yaml.contains("query_count: 5"));
        assert!(yaml.contains("neighbors: 3"));
        assert!(yaml.contains("base.fvec"));
        assert!(yaml.contains("query.fvec"));
    }

    #[test]
    fn test_ground_truth_correctness() {
        // Generate a tiny dataset and verify KNN is correct
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        // Write known base vectors
        let base_path = ws.join("base.fvec");
        let query_path = ws.join("query.fvec");
        let idx_path = ws.join("indices.ivec");
        let dist_path = ws.join("distances.fvec");

        // Base: [0,0], [1,0], [10,0]
        write_fvec(&base_path, &[vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 0.0]]);
        // Query: [0.5, 0]
        write_fvec(&query_path, &[vec![0.5, 0.0]]);

        compute_ground_truth(&base_path, &query_path, &idx_path, &dist_path, 2, "L2").unwrap();

        // Nearest to [0.5, 0] should be [0,0] (d=0.5) and [1,0] (d=0.5)
        let reader = MmapVectorReader::<i32>::open_ivec(&idx_path).unwrap();
        let indices = reader.get(0).unwrap();
        // Both base[0] and base[1] are distance 0.5; base[2] is distance 9.5
        assert!(indices.contains(&0) && indices.contains(&1));
    }

    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }
}
