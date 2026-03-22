// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate vectors from a statistical model.
//!
//! Loads a `VectorSpaceModel` JSON file (produced by profile analysis or
//! `generate sketch --model-out`) and generates synthetic vectors by sampling
//! each dimension from its fitted distribution via inverse CDF transform.
//!
//! The model JSON format:
//! ```json
//! {
//!   "dimensions": 128,
//!   "unique_vectors": 1000000,
//!   "models": [
//!     { "type": "normal", "mean": 0.1, "std_dev": 0.3 },
//!     { "type": "beta", "alpha": 2.0, "beta": 5.0, "lower": -1.0, "upper": 1.0 },
//!     { "type": "uniform", "lower": -0.5, "upper": 0.5 },
//!     ...
//!   ]
//! }
//! ```
//!
//! Equivalent to the Java `CMD_generate_from_model` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rand_distr::{Beta, Distribution, Normal, Uniform};
use serde::Deserialize;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;

/// Pipeline command: generate vectors from a statistical model.
pub struct GenerateFromModelOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateFromModelOp)
}

/// Vector space model loaded from JSON.
#[derive(Debug, Deserialize)]
struct VectorSpaceModel {
    /// Number of dimensions.
    dimensions: u32,
    /// Number of unique vectors in the model's training set.
    #[serde(default)]
    unique_vectors: Option<u64>,
    /// Per-dimension scalar models.
    models: Vec<ScalarModelDef>,
}

/// Scalar distribution model definition (JSON representation).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ScalarModelDef {
    Normal {
        mean: f64,
        std_dev: f64,
    },
    Beta {
        alpha: f64,
        beta: f64,
        #[serde(default = "default_lower")]
        lower: f64,
        #[serde(default = "default_upper")]
        upper: f64,
    },
    Uniform {
        lower: f64,
        upper: f64,
    },
}

fn default_lower() -> f64 {
    -1.0
}
fn default_upper() -> f64 {
    1.0
}

/// Compiled sampler from a scalar model definition.
enum ModelSampler {
    Normal(Normal<f64>),
    BetaScaled {
        dist: Beta<f64>,
        lower: f64,
        upper: f64,
    },
    Uniform(Uniform<f64>),
}

impl ModelSampler {
    fn from_def(def: &ScalarModelDef) -> Result<Self, String> {
        match def {
            ScalarModelDef::Normal { mean, std_dev } => {
                let dist = Normal::new(*mean, *std_dev)
                    .map_err(|e| format!("invalid normal params: {}", e))?;
                Ok(ModelSampler::Normal(dist))
            }
            ScalarModelDef::Beta {
                alpha,
                beta,
                lower,
                upper,
            } => {
                let dist = Beta::new(*alpha, *beta)
                    .map_err(|e| format!("invalid beta params: {}", e))?;
                Ok(ModelSampler::BetaScaled {
                    dist,
                    lower: *lower,
                    upper: *upper,
                })
            }
            ScalarModelDef::Uniform { lower, upper } => {
                let dist = Uniform::new(*lower, *upper)
                    .map_err(|e| format!("invalid uniform params: {}", e))?;
                Ok(ModelSampler::Uniform(dist))
            }
        }
    }

    fn sample(&self, rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> f64 {
        match self {
            ModelSampler::Normal(dist) => dist.sample(rng),
            ModelSampler::BetaScaled { dist, lower, upper } => {
                let u = dist.sample(rng);
                lower + u * (upper - lower)
            }
            ModelSampler::Uniform(dist) => dist.sample(rng),
        }
    }
}

impl CommandOp for GenerateFromModelOp {
    fn command_path(&self) -> &str {
        "generate from-model"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate vectors using an embedding model".into(),
            body: format!(r#"# generate from-model

Generate vectors using an embedding model.

## Description

Reads a `model.json` file describing per-dimension distribution parameters
and generates synthetic vectors that statistically match a real dataset's
profile. Each dimension is sampled independently from its fitted
distribution (Normal, Beta, or Uniform) via inverse CDF transform.

The model JSON format contains a `dimensions` count and a `models` array
with one entry per dimension. Each entry specifies a distribution type and
its parameters:

- **normal** -- `mean` and `std_dev`
- **beta** -- `alpha`, `beta`, `lower`, `upper` (samples are scaled from
  `[0,1]` to `[lower, upper]`)
- **uniform** -- `lower` and `upper`

The number of entries in the `models` array must exactly match the
`dimensions` field; a mismatch is an error.

## Deterministic generation

When the same `seed` is used with the same model file, the output is
byte-identical across runs. This allows reproducible benchmarks even when
the underlying real dataset cannot be distributed.

## Role in dataset pipelines

This command is the generation half of the derive-then-generate workflow
for creating realistic synthetic datasets:

1. **Derive** -- `generate derive` reads a real dataset and extracts a
   `model.json` capturing per-dimension statistics (mean, variance, shape).
2. **Generate** -- `generate from-model` reads that `model.json` and
   produces an arbitrary number of synthetic vectors whose per-dimension
   distributions match the real data.

The resulting synthetic vectors preserve the statistical fingerprint of
the original dataset -- dimension-wise means, variances, and distributional
shapes -- without containing any of the original data. This is useful for
benchmarking ANN algorithms on realistic workloads when the real dataset
is proprietary or too large to distribute.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let model_str = match options.require("model") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let count: u64 = match options.require("count") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid count: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let seed = rng::parse_seed(options.get("seed"));
        let model_path = resolve_path(model_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Load model
        let model_json = match std::fs::read_to_string(&model_path) {
            Ok(s) => s,
            Err(e) => {
                return error_result(
                    format!("failed to read model {}: {}", model_path.display(), e),
                    start,
                )
            }
        };
        let model: VectorSpaceModel = match serde_json::from_str(&model_json) {
            Ok(m) => m,
            Err(e) => {
                return error_result(
                    format!("failed to parse model {}: {}", model_path.display(), e),
                    start,
                )
            }
        };

        if model.models.len() != model.dimensions as usize {
            return error_result(
                format!(
                    "model declares {} dimensions but has {} scalar models",
                    model.dimensions,
                    model.models.len()
                ),
                start,
            );
        }

        // Compile samplers
        let samplers: Vec<ModelSampler> = match model
            .models
            .iter()
            .enumerate()
            .map(|(i, def)| {
                ModelSampler::from_def(def)
                    .map_err(|e| format!("dimension {}: {}", i, e))
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        // Generate vectors
        let mut data_rng = rng::seeded_rng(seed);

        match generate_from_model_fvec(&output_path, model.dimensions, count, &samplers, &mut data_rng) {
            Ok(()) => {
                let uv_info = model
                    .unique_vectors
                    .map(|n| format!(", model trained on {} unique vectors", n))
                    .unwrap_or_default();
                CommandResult {
                    status: Status::Ok,
                    message: format!(
                        "generated {} vectors (dim={}) from model to {}{}",
                        count, model.dimensions, output_path.display(), uv_info
                    ),
                    produced: vec![output_path],
                    elapsed: start.elapsed(),
                }
            },
            Err(e) => error_result(e, start),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "model".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "VectorSpaceModel JSON file".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output fvec file".to_string(),
                        role: OptionRole::Output,
        },
            OptionDesc {
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of vectors to generate".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Random seed".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Generate vectors from compiled model samplers as fvec.
fn generate_from_model_fvec(
    output: &Path,
    dim: u32,
    count: u64,
    samplers: &[ModelSampler],
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

    for _ in 0..count {
        writer
            .write_all(&(dim as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for sampler in samplers {
            let val = sampler.sample(rng) as f32;
            writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
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
    fn test_from_model_basic() {
        let tmp = tempfile::tempdir().unwrap();

        // Write a simple model
        let model_path = tmp.path().join("model.json");
        let model_json = r#"{
            "dimensions": 4,
            "models": [
                { "type": "normal", "mean": 0.0, "std_dev": 1.0 },
                { "type": "beta", "alpha": 2.0, "beta": 5.0 },
                { "type": "uniform", "lower": -1.0, "upper": 1.0 },
                { "type": "normal", "mean": 0.5, "std_dev": 0.2 }
            ]
        }"#;
        std::fs::write(&model_path, model_json).unwrap();

        let out = tmp.path().join("output.fvec");
        let mut opts = Options::new();
        opts.set("model", model_path.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("count", "50");
        opts.set("seed", "42");

        let mut op = GenerateFromModelOp;
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let size = std::fs::metadata(&out).unwrap().len();
        assert_eq!(size, 50 * (4 + 4 * 4));
    }

    #[test]
    fn test_from_model_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join("model.json");
        let model_json = r#"{
            "dimensions": 3,
            "models": [
                { "type": "normal", "mean": 0.0, "std_dev": 1.0 },
                { "type": "uniform", "lower": 0.0, "upper": 1.0 },
                { "type": "beta", "alpha": 1.0, "beta": 1.0 }
            ]
        }"#;
        std::fs::write(&model_path, model_json).unwrap();

        let out1 = tmp.path().join("a.fvec");
        let out2 = tmp.path().join("b.fvec");

        for out in [&out1, &out2] {
            let mut opts = Options::new();
            opts.set("model", model_path.to_string_lossy().to_string());
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("count", "100");
            opts.set("seed", "42");

            let mut op = GenerateFromModelOp;
            let mut ctx = test_ctx(tmp.path());
            op.execute(&opts, &mut ctx);
        }

        let data1 = std::fs::read(&out1).unwrap();
        let data2 = std::fs::read(&out2).unwrap();
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_from_model_dimension_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join("model.json");
        let model_json = r#"{
            "dimensions": 3,
            "models": [
                { "type": "normal", "mean": 0.0, "std_dev": 1.0 }
            ]
        }"#;
        std::fs::write(&model_path, model_json).unwrap();

        let out = tmp.path().join("output.fvec");
        let mut opts = Options::new();
        opts.set("model", model_path.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("count", "10");

        let mut op = GenerateFromModelOp;
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("dimension"));
    }

    #[test]
    fn test_scalar_model_parsing() {
        let json = r#"{ "type": "normal", "mean": 0.5, "std_dev": 0.1 }"#;
        let def: ScalarModelDef = serde_json::from_str(json).unwrap();
        match def {
            ScalarModelDef::Normal { mean, std_dev } => {
                assert_eq!(mean, 0.5);
                assert_eq!(std_dev, 0.1);
            }
            _ => panic!("expected Normal"),
        }

        let json = r#"{ "type": "beta", "alpha": 2.0, "beta": 3.0, "lower": 0.0, "upper": 1.0 }"#;
        let def: ScalarModelDef = serde_json::from_str(json).unwrap();
        match def {
            ScalarModelDef::Beta { alpha, beta, .. } => {
                assert_eq!(alpha, 2.0);
                assert_eq!(beta, 3.0);
            }
            _ => panic!("expected Beta"),
        }
    }
}
