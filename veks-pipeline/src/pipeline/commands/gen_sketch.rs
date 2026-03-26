// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate sketch vectors with controlled distributions.
//!
//! Generates reference datasets with known, reproducible statistical properties
//! by selecting from predefined distribution mix strategies. Each dimension of
//! a vector is sampled from an independently configured distribution, producing
//! datasets useful for testing profile analysis and similarity search.
//!
//! Distribution mix types:
//! - `bounded` — 50% Normal, 30% Beta, 20% Uniform (default)
//! - `normal` — all dimensions use truncated Normal
//! - `beta` — all dimensions use Beta distribution
//! - `uniform` — all dimensions use Uniform
//! - `mixed` — rotating Normal/Beta/Uniform by dimension index
//!
//! Equivalent to the Java `CMD_generate_sketch` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::Rng;
use rand_distr::{Beta, Distribution, Normal, Uniform};
use rayon::prelude::*;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;

/// Pipeline command: generate sketch vectors.
pub struct GenerateSketchOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateSketchOp)
}

/// Distribution mix strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MixType {
    Bounded,
    Normal,
    Beta,
    Uniform,
    Mixed,
}

impl MixType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bounded" => Some(MixType::Bounded),
            "normal" | "normal_only" => Some(MixType::Normal),
            "beta" | "beta_only" => Some(MixType::Beta),
            "uniform" | "uniform_only" => Some(MixType::Uniform),
            "mixed" => Some(MixType::Mixed),
            _ => None,
        }
    }
}

/// Per-dimension sampler configuration.
enum DimSampler {
    NormalTruncated { dist: Normal<f64>, lower: f64, upper: f64 },
    BetaScaled { dist: Beta<f64>, lower: f64, upper: f64 },
    UniformRange { dist: Uniform<f64> },
}

impl DimSampler {
    fn sample(&self, rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> f64 {
        match self {
            DimSampler::NormalTruncated { dist, lower, upper } => {
                // Rejection sampling for truncated normal
                loop {
                    let val = dist.sample(rng);
                    if val >= *lower && val <= *upper {
                        return val;
                    }
                }
            }
            DimSampler::BetaScaled { dist, lower, upper } => {
                let u = dist.sample(rng); // u in [0, 1]
                lower + u * (upper - lower)
            }
            DimSampler::UniformRange { dist } => dist.sample(rng),
        }
    }
}

impl CommandOp for GenerateSketchOp {
    fn command_path(&self) -> &str {
        "generate sketch"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate a compact sketch of vector data".into(),
            body: format!(r#"# generate sketch

Generate a compact sketch of vector data.

## Description

Generates reference datasets with known, reproducible statistical properties
by selecting from predefined distribution mix strategies. Each dimension of
a vector is sampled from an independently configured distribution, producing
datasets useful for testing profile analysis and similarity search.

Unlike `generate vectors` (which draws all elements from a single uniform
distribution), sketch vectors have per-dimension statistical structure.
This makes them more representative of real embedding spaces, where
different dimensions often have distinct value ranges and shapes.

## Distribution strategies

The `mix` parameter selects how per-dimension samplers are assigned:

- **bounded** (default) -- 50% Normal, 30% Beta, 20% Uniform, assigned by
  dimension index modulo 10. This is a good general-purpose default that
  produces realistic-looking value distributions.
- **normal** -- all dimensions use truncated Normal distributions with
  randomly varied means and standard deviations.
- **beta** -- all dimensions use Beta distributions with random alpha/beta
  shape parameters, scaled to the `[lower, upper]` range.
- **uniform** -- all dimensions use Uniform distributions with slight
  random variation in bounds.
- **mixed** -- rotating Normal/Beta/Uniform by dimension index modulo 3.

For each dimension, the sampler's parameters (mean, std_dev, alpha, beta,
bounds) are drawn from the parameter PRNG seeded by `seed`. A separate
data PRNG (seeded by `seed + 1`) generates the actual vector elements.
This two-PRNG design ensures that adding more vectors does not change
earlier vectors' values.

## Normalization

When `normalize=true` (the default), each vector is L2-normalized after
sampling so that all output vectors lie on the unit hypersphere. This is
standard for cosine-similarity workloads.

## Role in dataset pipelines

Sketch datasets provide a compact, controllable representation for
exploratory analysis and quick similarity checks before running expensive
exact KNN computations. They are also useful for validating that distance
functions, indexing algorithms, and recall metrics behave correctly on
data with known statistical properties.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let dimension: u32 = match options.require("dimension") {
            Ok(s) => match s.parse() {
                Ok(d) if d > 0 => d,
                _ => return error_result(format!("invalid dimension: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let count: u64 = match options.require("count") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid count: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let mix_str = options.get("mix").unwrap_or("bounded");
        let mix = match MixType::from_str(mix_str) {
            Some(m) => m,
            None => {
                return error_result(
                    format!(
                        "unknown mix type: '{}'. Use bounded, normal, beta, uniform, or mixed",
                        mix_str
                    ),
                    start,
                )
            }
        };

        let seed = rng::parse_seed(options.get("seed"));
        let lower: f64 = options.get("lower").and_then(|s| s.parse().ok()).unwrap_or(-1.0);
        let upper: f64 = options.get("upper").and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let normalize = options
            .get("normalize")
            .map(|s| s == "true" || s == "1")
            .unwrap_or(true);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        // Build per-dimension samplers
        let mut param_rng = rng::seeded_rng(seed);
        let samplers: Vec<DimSampler> = (0..dimension)
            .map(|d| build_sampler(d, mix, lower, upper, &mut param_rng))
            .collect();

        // Query governor for thread count and build rayon pool
        let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        let pb = ctx.ui.bar_with_unit(count, "generating sketch vectors", "vectors");

        // Generate vectors
        let mut data_rng = rng::seeded_rng(seed.wrapping_add(1));
        match generate_sketch_fvec(&output_path, dimension, count, &samplers, normalize, &mut data_rng, &pb, pool.as_ref()) {
            Ok(()) => {
                pb.finish();

                // Write verified count for the bound checker
                let var_name = format!("verified_count:{}",
                    output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
                let _ = crate::pipeline::variables::set_and_save(
                    &ctx.workspace, &var_name, &count.to_string());
                ctx.defaults.insert(var_name, count.to_string());

                CommandResult {
                    status: Status::Ok,
                    message: format!(
                        "generated {} sketch vectors (dim={}, mix={:?}) to {}",
                        count, dimension, mix, output_path.display()
                    ),
                    produced: vec![output_path],
                    elapsed: start.elapsed(),
                }
            }
            Err(e) => {
                pb.finish();
                error_result(e, start)
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output fvec file".to_string(),
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
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of vectors".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "mix".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("bounded".to_string()),
                description: "Distribution mix: bounded, normal, beta, uniform, mixed".to_string(),
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
            OptionDesc {
                name: "lower".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("-1.0".to_string()),
                description: "Lower bound for distributions".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "upper".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("1.0".to_string()),
                description: "Upper bound for distributions".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "normalize".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("true".to_string()),
                description: "L2-normalize output vectors".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Build a per-dimension sampler based on the mix strategy.
fn build_sampler(
    dim_idx: u32,
    mix: MixType,
    lower: f64,
    upper: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> DimSampler {
    let range = upper - lower;
    let center = (lower + upper) / 2.0;

    match mix {
        MixType::Normal => {
            // Mean varies within ±25% of center, stddev 10-40% of range
            let mean = center + (rng.random::<f64>() - 0.5) * range * 0.5;
            let std_dev = range * (0.1 + rng.random::<f64>() * 0.3);
            DimSampler::NormalTruncated {
                dist: Normal::new(mean, std_dev).unwrap(),
                lower,
                upper,
            }
        }
        MixType::Beta => {
            let alpha = 1.0 + rng.random::<f64>() * 4.0;
            let beta = 1.0 + rng.random::<f64>() * 4.0;
            DimSampler::BetaScaled {
                dist: Beta::new(alpha, beta).unwrap(),
                lower,
                upper,
            }
        }
        MixType::Uniform => {
            // Slight random variation in bounds
            let margin = range * 0.1;
            let lo = lower + rng.random::<f64>() * margin;
            let hi = upper - rng.random::<f64>() * margin;
            DimSampler::UniformRange {
                dist: Uniform::new(lo, hi).unwrap(),
            }
        }
        MixType::Bounded => {
            // 50% Normal, 30% Beta, 20% Uniform — selected by dimension index
            match dim_idx % 10 {
                0..5 => build_sampler(dim_idx, MixType::Normal, lower, upper, rng),
                5..8 => build_sampler(dim_idx, MixType::Beta, lower, upper, rng),
                _ => build_sampler(dim_idx, MixType::Uniform, lower, upper, rng),
            }
        }
        MixType::Mixed => {
            // Rotating by dimension index
            match dim_idx % 3 {
                0 => build_sampler(dim_idx, MixType::Normal, lower, upper, rng),
                1 => build_sampler(dim_idx, MixType::Beta, lower, upper, rng),
                _ => build_sampler(dim_idx, MixType::Uniform, lower, upper, rng),
            }
        }
    }
}

/// Generate sketch vectors as fvec, optionally L2-normalized.
///
/// Vectors are generated sequentially (RNG is stateful) but normalization of
/// batched chunks is parallelized via rayon when a pool is provided.
fn generate_sketch_fvec(
    output: &Path,
    dim: u32,
    count: u64,
    samplers: &[DimSampler],
    normalize: bool,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &veks_core::ui::ProgressHandle,
    pool: Option<&rayon::ThreadPool>,
) -> Result<(), String> {
    use std::io::Write;
    let mut writer = AtomicWriter::new(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;

    let dim_usize = dim as usize;
    let batch_size = 10_000usize;
    let mut written = 0u64;

    while written < count {
        let batch_count = batch_size.min((count - written) as usize);

        // Sample batch sequentially (RNG is stateful)
        let mut batch: Vec<f32> = vec![0.0; batch_count * dim_usize];
        for v in 0..batch_count {
            let offset = v * dim_usize;
            for (d, sampler) in samplers.iter().enumerate() {
                batch[offset + d] = sampler.sample(rng) as f32;
            }
        }

        // Normalize batch in parallel if requested
        if normalize {
            let mut normalize_fn = || {
                batch
                    .par_chunks_mut(dim_usize)
                    .for_each(|vec_buf| {
                        let norm_sq: f64 = vec_buf.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                        if norm_sq > 0.0 {
                            let inv_norm = 1.0 / norm_sq.sqrt();
                            for v in vec_buf.iter_mut() {
                                *v = (*v as f64 * inv_norm) as f32;
                            }
                        }
                    });
            };
            if let Some(p) = pool {
                p.install(normalize_fn);
            } else {
                normalize_fn();
            }
        }

        // Write batch to fvec
        for v in 0..batch_count {
            let offset = v * dim_usize;
            writer
                .write_all(&(dim as i32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for d in 0..dim_usize {
                writer
                    .write_all(&batch[offset + d].to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }

        written += batch_count as u64;
        pb.set_position(written);
    }

    writer.finish().map_err(|e| e.to_string())?;
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_sketch_bounded() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("sketch.fvec");

        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("dimension", "16");
        opts.set("count", "100");
        opts.set("mix", "bounded");
        opts.set("seed", "42");

        let mut op = GenerateSketchOp;
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let size = std::fs::metadata(&out).unwrap().len();
        assert_eq!(size, 100 * (4 + 16 * 4));
    }

    #[test]
    fn test_sketch_normalized() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("sketch.fvec");

        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "50");
        opts.set("normalize", "true");
        opts.set("seed", "42");

        let mut op = GenerateSketchOp;
        let mut ctx = test_ctx(tmp.path());
        op.execute(&opts, &mut ctx);

        // Read back and check normalization
        let data = std::fs::read(&out).unwrap();
        let record_size = 4 + 8 * 4;
        for r in 0..50 {
            let offset = r * record_size + 4;
            let mut norm_sq = 0.0f64;
            for d in 0..8 {
                let byte_offset = offset + d * 4;
                let val = f32::from_le_bytes([
                    data[byte_offset],
                    data[byte_offset + 1],
                    data[byte_offset + 2],
                    data[byte_offset + 3],
                ]) as f64;
                norm_sq += val * val;
            }
            assert!(
                (norm_sq.sqrt() - 1.0).abs() < 0.01,
                "vector {} not normalized: ||v|| = {}",
                r,
                norm_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_sketch_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let out1 = tmp.path().join("a.fvec");
        let out2 = tmp.path().join("b.fvec");

        for out in [&out1, &out2] {
            let mut opts = Options::new();
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("dimension", "8");
            opts.set("count", "50");
            opts.set("mix", "mixed");
            opts.set("seed", "42");

            let mut op = GenerateSketchOp;
            let mut ctx = test_ctx(tmp.path());
            op.execute(&opts, &mut ctx);
        }

        let data1 = std::fs::read(&out1).unwrap();
        let data2 = std::fs::read(&out2).unwrap();
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_all_mix_types() {
        let tmp = tempfile::tempdir().unwrap();

        for mix in ["bounded", "normal", "beta", "uniform", "mixed"] {
            let out = tmp.path().join(format!("{}.fvec", mix));
            let mut opts = Options::new();
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("dimension", "4");
            opts.set("count", "10");
            opts.set("mix", mix);
            opts.set("seed", "42");

            let mut op = GenerateSketchOp;
            let mut ctx = test_ctx(tmp.path());
            let result = op.execute(&opts, &mut ctx);
            assert_eq!(result.status, Status::Ok, "failed for mix={}", mix);
        }
    }

    #[test]
    fn test_mix_type_parsing() {
        assert_eq!(MixType::from_str("bounded"), Some(MixType::Bounded));
        assert_eq!(MixType::from_str("NORMAL"), Some(MixType::Normal));
        assert_eq!(MixType::from_str("beta_only"), Some(MixType::Beta));
        assert_eq!(MixType::from_str("uniform_only"), Some(MixType::Uniform));
        assert_eq!(MixType::from_str("mixed"), Some(MixType::Mixed));
        assert_eq!(MixType::from_str("invalid"), None);
    }
}
