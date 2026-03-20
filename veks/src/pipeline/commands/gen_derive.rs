// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: derive a synthetic dataset from an existing one.
//!
//! Extracts a statistical model (per-dimension distribution parameters) from
//! a source dataset's base vectors, then creates a target directory with a
//! `model.json` and `dataset.yaml` that can be used to generate unlimited
//! synthetic vectors matching the source's statistical profile.
//!
//! Equivalent to the Java `CMD_generate_derive` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: derive a synthetic dataset from an existing one.
pub struct GenerateDeriveOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateDeriveOp)
}

/// Statistical model for an entire vector space (one model per dimension).
#[derive(Debug, Serialize, Deserialize)]
struct VectorSpaceModel {
    dimensions: u32,
    #[serde(rename = "uniqueVectors")]
    unique_vectors: u64,
    #[serde(rename = "scalarModels")]
    scalar_models: Vec<ScalarModelDef>,
}

/// A single-dimension distribution model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ScalarModelDef {
    Normal { mu: f64, sigma: f64 },
    Uniform { lower: f64, upper: f64 },
    Beta { alpha: f64, beta: f64, lower: f64, upper: f64 },
}

impl CommandOp for GenerateDeriveOp {
    fn command_path(&self) -> &str {
        "generate derive"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Derive new facets from existing dataset files".into(),
            body: format!(r#"# generate derive

Derive new facets from existing dataset files.

## Description

Reads the base vectors from an existing dataset and computes per-dimension
distribution parameters, producing a statistical model that captures the
dataset's profile. The output is a target directory containing:

- `model.json` -- a VectorSpaceModel with one scalar model per dimension,
  encoding the fitted distribution type and parameters (mean, variance,
  alpha, beta, bounds, etc.)
- `dataset.yaml` -- a scaffold descriptor for the derived dataset,
  referencing the model and recording the source dataset's provenance

## Model extraction process

For each dimension, the command samples up to `sample` vectors (default
10,000) from the source and computes summary statistics: mean, variance,
min, max, and range. A distribution is fitted using moment-based
heuristics:

- **Beta** is preferred when the data appears bounded (values within
  approximately `[-1.5, 1.5]` with non-trivial variance), as is common
  for normalized embedding vectors. Method-of-moments estimates the
  alpha and beta shape parameters.
- **Uniform** is used when variance is near zero (effectively constant
  dimensions).
- **Normal** is the default fallback for unbounded or wide-ranging data.

The fitted parameters preserve the statistical fingerprint of the original
data -- dimension-wise means, variances, and distributional shapes -- so
that vectors generated from the model exhibit similar distance
distributions and nearest-neighbor structure.

## Role in dataset pipelines

This command is the first half of the **derive-then-generate** workflow
for creating realistic synthetic datasets:

1. **Derive** (`generate derive`) -- run once against a real dataset to
   extract `model.json`.
2. **Generate** (`generate from-model`) -- run as many times as needed to
   produce unlimited synthetic vectors that statistically match the real
   data.

This workflow is valuable when the real dataset is proprietary, too large
to distribute, or subject to licensing restrictions. The model.json is
small (a few KB) and contains no original vector data, only summary
statistics.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Derivation buffers".into(), adjustable: true },
            ResourceDesc { name: "threads".into(), description: "Parallel derivation".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let target_str = match options.require("target") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let target_path = resolve_path(target_str, &ctx.workspace);
        let force = options.get("force").map_or(false, |s| s == "true");
        let sample: usize = options
            .get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000);
        let count: u64 = options
            .get("count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0); // 0 = same as source
        let name = options.get("name").unwrap_or("derived");

        // Find base vectors file in source
        let base_vectors_path = find_base_vectors(&source_path);
        let base_path = match base_vectors_path {
            Some(p) => p,
            None => {
                return error_result(
                    format!(
                        "no base vector file found in {}. Expected .fvec file.",
                        source_path.display()
                    ),
                    start,
                )
            }
        };

        // Open and sample vectors
        let reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open {}: {}", base_path.display(), e),
                    start,
                )
            }
        };

        let total_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);
        let effective_sample = sample.min(total_count);
        let output_count = if count == 0 { total_count as u64 } else { count };

        ctx.ui.log(&format!(
            "Extracting model from {} ({} vectors, dim={}, sampling {})",
            base_path.display(),
            total_count,
            dim,
            effective_sample
        ));

        // Extract per-dimension statistics
        let mut models: Vec<ScalarModelDef> = Vec::with_capacity(dim);
        for d in 0..dim {
            let mut values: Vec<f64> = Vec::with_capacity(effective_sample);
            for i in 0..effective_sample {
                if let Ok(vec) = reader.get(i) {
                    values.push(vec[d] as f64);
                }
            }
            let model = fit_dimension(&values);
            models.push(model);
        }

        // Create target directory
        if target_path.exists() && !force {
            return error_result(
                format!(
                    "{} already exists. Use force=true to overwrite.",
                    target_path.display()
                ),
                start,
            );
        }
        if let Err(e) = std::fs::create_dir_all(&target_path) {
            return error_result(format!("failed to create {}: {}", target_path.display(), e), start);
        }

        // Write model.json
        let vsm = VectorSpaceModel {
            dimensions: dim as u32,
            unique_vectors: output_count,
            scalar_models: models,
        };
        let model_json = match serde_json::to_string_pretty(&vsm) {
            Ok(j) => j,
            Err(e) => return error_result(format!("JSON serialization error: {}", e), start),
        };
        let model_path = target_path.join("model.json");
        if let Err(e) = std::fs::write(&model_path, &model_json) {
            return error_result(format!("failed to write model.json: {}", e), start);
        }

        // Write dataset.yaml
        let dataset_yaml = format!(
            "name: {}\n\
             description: \"Derived from {}\"\n\
             \n\
             attributes:\n\
             \x20 derived_from: \"{}\"\n\
             \x20 generation_mode: virtdata\n\
             \x20 dimensions: {}\n\
             \n\
             profiles:\n\
             \x20 default:\n\
             \x20   base_vectors: base_vectors.fvec\n",
            name,
            source_path.display(),
            source_path.display(),
            dim,
        );
        let yaml_path = target_path.join("dataset.yaml");
        if let Err(e) = std::fs::write(&yaml_path, &dataset_yaml) {
            return error_result(format!("failed to write dataset.yaml: {}", e), start);
        }

        ctx.ui.log(&format!(
            "Derived dataset '{}' with {} dimensions, {} target vectors",
            name, dim, output_count
        ));
        ctx.ui.log(&format!("  model: {}", model_path.display()));
        ctx.ui.log(&format!("  config: {}", yaml_path.display()));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "derived model (dim={}) from {} vectors to {}",
                dim, total_count, target_path.display()
            ),
            produced: vec![model_path, yaml_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source dataset directory or fvec file".to_string(),
            },
            OptionDesc {
                name: "target".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Target directory for derived dataset".to_string(),
            },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("10000".to_string()),
                description: "Max vectors to sample for model extraction".to_string(),
            },
            OptionDesc {
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Target vector count (0 = same as source)".to_string(),
            },
            OptionDesc {
                name: "name".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("derived".to_string()),
                description: "Name for the derived dataset".to_string(),
            },
            OptionDesc {
                name: "force".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Overwrite target if exists".to_string(),
            },
        ]
    }
}

/// Find a base vectors .fvec file in a source path.
///
/// If the path is a .fvec file directly, returns it.
/// Otherwise searches the directory for base_vectors.fvec or any .fvec file.
fn find_base_vectors(source: &Path) -> Option<PathBuf> {
    if source.is_file() {
        if source.extension().and_then(|e| e.to_str()) == Some("fvec") {
            return Some(source.to_path_buf());
        }
        return None;
    }

    // Look for base_vectors.fvec first
    let bv = source.join("base_vectors.fvec");
    if bv.exists() {
        return Some(bv);
    }

    // Fall back to any .fvec file
    if let Ok(entries) = std::fs::read_dir(source) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("fvec") {
                return Some(path);
            }
        }
    }

    None
}

/// Fit a distribution model to a dimension's values.
///
/// Uses moment-based selection: if data appears bounded (within [-1,1] with
/// non-trivial shape), fits a Beta distribution. If data has very low variance,
/// fits Uniform. Otherwise fits Normal.
fn fit_dimension(values: &[f64]) -> ScalarModelDef {
    if values.is_empty() {
        return ScalarModelDef::Normal { mu: 0.0, sigma: 1.0 };
    }

    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    // If range is near zero, use uniform
    if range < 1e-10 {
        return ScalarModelDef::Uniform {
            lower: min - 0.001,
            upper: max + 0.001,
        };
    }

    // Check if data appears bounded (common for normalized vectors)
    let is_bounded = min >= -1.5 && max <= 1.5 && range < 3.0;

    if is_bounded && variance > 1e-6 {
        // Try Beta distribution on normalized data
        let a = min - range * 0.01; // slight padding
        let b = max + range * 0.01;
        let norm_mean = (mean - a) / (b - a);
        let norm_var = variance / ((b - a) * (b - a));

        // Method of moments for Beta
        if norm_mean > 0.0 && norm_mean < 1.0 && norm_var > 0.0 && norm_var < norm_mean * (1.0 - norm_mean) {
            let common = norm_mean * (1.0 - norm_mean) / norm_var - 1.0;
            let alpha = norm_mean * common;
            let beta_param = (1.0 - norm_mean) * common;

            if alpha > 0.1 && beta_param > 0.1 && alpha < 100.0 && beta_param < 100.0 {
                return ScalarModelDef::Beta {
                    alpha,
                    beta: beta_param,
                    lower: a,
                    upper: b,
                };
            }
        }
    }

    // Default: Normal distribution
    ScalarModelDef::Normal {
        mu: mean,
        sigma: std_dev.max(1e-6),
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
    use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
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
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_fit_normal() {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 100.0).collect();
        let model = fit_dimension(&values);
        match model {
            ScalarModelDef::Normal { mu, sigma } => {
                assert!((mu - 0.0).abs() < 0.5);
                assert!(sigma > 0.0);
            }
            _ => panic!("expected Normal model"),
        }
    }

    #[test]
    fn test_fit_bounded() {
        // Data in [-0.5, 0.5] should trigger Beta
        let values: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) - 0.5).collect();
        let model = fit_dimension(&values);
        match model {
            ScalarModelDef::Beta { alpha, beta, lower, upper } => {
                assert!(alpha > 0.0);
                assert!(beta > 0.0);
                assert!(lower < -0.5);
                assert!(upper > 0.5);
            }
            _ => panic!("expected Beta model, got {:?}", model),
        }
    }

    #[test]
    fn test_fit_constant() {
        let values = vec![5.0; 100];
        let model = fit_dimension(&values);
        match model {
            ScalarModelDef::Uniform { lower, upper } => {
                assert!(lower < 5.0);
                assert!(upper > 5.0);
            }
            _ => panic!("expected Uniform for constant data"),
        }
    }

    #[test]
    fn test_derive_from_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate source vectors
        let source_path = ws.join("source.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", source_path.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "50");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        // Derive
        let target_dir = ws.join("derived");
        let mut opts = Options::new();
        opts.set("source", source_path.to_string_lossy().to_string());
        opts.set("target", target_dir.to_string_lossy().to_string());
        opts.set("name", "test-derived");
        let mut op = GenerateDeriveOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "derive failed: {}", result.message);

        // Check outputs
        assert!(target_dir.join("model.json").exists());
        assert!(target_dir.join("dataset.yaml").exists());

        // Validate model.json
        let model_str = std::fs::read_to_string(target_dir.join("model.json")).unwrap();
        let model: VectorSpaceModel = serde_json::from_str(&model_str).unwrap();
        assert_eq!(model.dimensions, 4);
        assert_eq!(model.scalar_models.len(), 4);
    }
}
