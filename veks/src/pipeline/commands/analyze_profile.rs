// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: profile a vector dataset's statistical distribution.
//!
//! Analyzes base vectors and builds a per-dimension VectorSpaceModel
//! describing the statistical distribution of each component. Outputs
//! a JSON model file suitable for use with `generate from-model`.
//!
//! Equivalent to the Java `CMD_analyze_profile` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: profile a vector dataset.
pub struct AnalyzeProfileOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeProfileOp)
}

/// Per-dimension fit result for reporting.
#[derive(Debug)]
struct DimFitResult {
    model_type: String,
    ks_d: f64,
}

/// Vector space model for serialization.
#[derive(Debug, Serialize, Deserialize)]
struct VectorSpaceModel {
    dimensions: u32,
    #[serde(rename = "uniqueVectors")]
    unique_vectors: u64,
    #[serde(rename = "scalarModels")]
    scalar_models: Vec<ScalarModelDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ScalarModelDef {
    Normal { mu: f64, sigma: f64 },
    Uniform { lower: f64, upper: f64 },
    Beta { alpha: f64, beta: f64, lower: f64, upper: f64 },
}

impl CommandOp for AnalyzeProfileOp {
    fn command_path(&self) -> &str {
        "analyze profile"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Profile distance computation performance".into(),
            body: format!(
                "# analyze profile\n\nProfile distance computation performance.\n\n## Description\n\nAnalyzes base vectors and builds a per-dimension VectorSpaceModel describing the statistical distribution of each component. Outputs a JSON model file suitable for use with `generate from-model`.\n\n## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Vector data for benchmarking".into(), adjustable: false },
            ResourceDesc { name: "threads".into(), description: "Parallel distance benchmarks".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let sample: usize = options
            .get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000);
        let verbose = options.get("verbose").map_or(false, |s| s == "true");

        let reader = match MmapVectorReader::<f32>::open_fvec(&source_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open {}: {}", source_path.display(), e),
                    start,
                )
            }
        };

        let total_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);
        let effective = sample.min(total_count);

        ctx.ui.log(&format!(
            "Profiling {} ({} vectors, dim={}, sampling {})",
            source_path.display(),
            total_count,
            dim,
            effective
        ));

        let mut models: Vec<ScalarModelDef> = Vec::with_capacity(dim);
        let mut fit_results: Vec<DimFitResult> = Vec::new();

        for d in 0..dim {
            let mut values: Vec<f64> = Vec::with_capacity(effective);
            for i in 0..effective {
                if let Ok(vec) = reader.get(i) {
                    values.push(vec[d] as f64);
                }
            }

            let (model, model_type, ks_d) = fit_and_evaluate(&values);
            if verbose || d < 5 || d == dim - 1 {
                ctx.ui.log(&format!(
                    "  dim[{}]: {} (KS D={:.4})",
                    d, model_type, ks_d
                ));
            } else if d == 5 {
                ctx.ui.log("  ...");
            }

            fit_results.push(DimFitResult {
                model_type: model_type.clone(),
                ks_d,
            });
            models.push(model);
        }

        // Summary
        let normal_count = fit_results.iter().filter(|r| r.model_type == "normal").count();
        let beta_count = fit_results.iter().filter(|r| r.model_type == "beta").count();
        let uniform_count = fit_results.iter().filter(|r| r.model_type == "uniform").count();
        let avg_ks: f64 = fit_results.iter().map(|r| r.ks_d).sum::<f64>() / fit_results.len() as f64;
        let max_ks: f64 = fit_results
            .iter()
            .map(|r| r.ks_d)
            .fold(0.0, f64::max);

        ctx.ui.log("");
        ctx.ui.log("Model summary:");
        ctx.ui.log(&format!("  Normal: {}, Beta: {}, Uniform: {}", normal_count, beta_count, uniform_count));
        ctx.ui.log(&format!("  Avg KS D: {:.4}, Max KS D: {:.4}", avg_ks, max_ks));

        // Write model
        let vsm = VectorSpaceModel {
            dimensions: dim as u32,
            unique_vectors: total_count as u64,
            scalar_models: models,
        };
        let json = match serde_json::to_string_pretty(&vsm) {
            Ok(j) => j,
            Err(e) => return error_result(format!("serialization error: {}", e), start),
        };
        if let Err(e) = std::fs::write(&output_path, &json) {
            return error_result(format!("failed to write {}: {}", output_path.display(), e), start);
        }

        ctx.ui.log(&format!("Wrote model to {}", output_path.display()));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "profiled {} dims from {} vectors, avg KS D={:.4}",
                dim, total_count, avg_ks
            ),
            produced: vec![output_path],
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
                description: "Input fvec file to profile".to_string(),
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output JSON model file".to_string(),
            },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("10000".to_string()),
                description: "Max vectors to sample".to_string(),
            },
            OptionDesc {
                name: "verbose".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show per-dimension details".to_string(),
            },
        ]
    }
}

/// Fit a distribution and compute KS D-statistic.
///
/// Returns (model, type_name, ks_d).
fn fit_and_evaluate(values: &[f64]) -> (ScalarModelDef, String, f64) {
    if values.is_empty() {
        return (
            ScalarModelDef::Normal { mu: 0.0, sigma: 1.0 },
            "normal".to_string(),
            1.0,
        );
    }

    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range < 1e-10 {
        return (
            ScalarModelDef::Uniform {
                lower: min - 0.001,
                upper: max + 0.001,
            },
            "uniform".to_string(),
            0.0,
        );
    }

    // Sort values for KS test
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Try Normal fit
    let normal_ks = ks_test_normal(&sorted, mean, std_dev);

    // Try Beta fit if data looks bounded
    let is_bounded = min >= -1.5 && max <= 1.5;
    let beta_result = if is_bounded && variance > 1e-6 {
        try_beta_fit(&sorted, min, max, mean, variance)
    } else {
        None
    };

    // Select best model
    match beta_result {
        Some((alpha, beta_param, lower, upper, beta_ks)) if beta_ks < normal_ks => (
            ScalarModelDef::Beta {
                alpha,
                beta: beta_param,
                lower,
                upper,
            },
            "beta".to_string(),
            beta_ks,
        ),
        _ => (
            ScalarModelDef::Normal {
                mu: mean,
                sigma: std_dev.max(1e-6),
            },
            "normal".to_string(),
            normal_ks,
        ),
    }
}

/// Kolmogorov-Smirnov D-statistic against Normal(mu, sigma).
fn ks_test_normal(sorted: &[f64], mu: f64, sigma: f64) -> f64 {
    let n = sorted.len() as f64;
    let mut max_d = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let empirical = (i + 1) as f64 / n;
        let z = (x - mu) / sigma;
        let theoretical = normal_cdf(z);
        let d = (empirical - theoretical).abs();
        max_d = max_d.max(d);
    }
    max_d
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327 * (-x * x / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.330274))));
    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Try fitting a Beta distribution. Returns (alpha, beta, lower, upper, ks_d) if valid.
fn try_beta_fit(
    sorted: &[f64],
    min: f64,
    max: f64,
    mean: f64,
    variance: f64,
) -> Option<(f64, f64, f64, f64, f64)> {
    let range = max - min;
    let a = min - range * 0.01;
    let b = max + range * 0.01;
    let scale = b - a;

    let norm_mean = (mean - a) / scale;
    let norm_var = variance / (scale * scale);

    if norm_mean <= 0.0 || norm_mean >= 1.0 || norm_var <= 0.0 {
        return None;
    }
    if norm_var >= norm_mean * (1.0 - norm_mean) {
        return None;
    }

    let common = norm_mean * (1.0 - norm_mean) / norm_var - 1.0;
    let alpha = norm_mean * common;
    let beta_param = (1.0 - norm_mean) * common;

    if alpha <= 0.1 || beta_param <= 0.1 || alpha > 100.0 || beta_param > 100.0 {
        return None;
    }

    // Compute KS D for Beta
    let n = sorted.len() as f64;
    let mut max_d = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let empirical = (i + 1) as f64 / n;
        let normalized = ((x - a) / scale).clamp(0.0, 1.0);
        let theoretical = incomplete_beta_approx(normalized, alpha, beta_param);
        let d = (empirical - theoretical).abs();
        max_d = max_d.max(d);
    }

    Some((alpha, beta_param, a, b, max_d))
}

/// Rough approximation of the regularized incomplete beta function.
///
/// Uses a simple numerical integration for the purpose of KS testing.
fn incomplete_beta_approx(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Simple trapezoidal integration of Beta PDF
    let steps = 200;
    let dx = x / steps as f64;
    let mut sum = 0.0;
    for i in 0..steps {
        let t0 = i as f64 * dx;
        let t1 = (i + 1) as f64 * dx;
        let f0 = beta_pdf(t0, a, b);
        let f1 = beta_pdf(t1, a, b);
        sum += (f0 + f1) / 2.0 * dx;
    }
    sum.clamp(0.0, 1.0)
}

/// Beta PDF (unnormalized, then normalized by beta function).
fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - log_beta(a, b);
    log_pdf.exp()
}

/// Log of the beta function using Stirling's approximation of log-gamma.
fn log_beta(a: f64, b: f64) -> f64 {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

/// Stirling's approximation of log-gamma.
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Lanczos approximation coefficients
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut sum = c[0];
    for (i, &coeff) in c[1..].iter().enumerate() {
        sum += coeff / (x + i as f64 + 1.0);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t).ln() * (x + 0.5) - t + sum.ln()
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
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    #[test]
    fn test_ks_normal_perfect() {
        // Standard normal samples — KS D should be small
        let mut values: Vec<f64> = (0..1000)
            .map(|i| {
                let p = (i as f64 + 0.5) / 1000.0;
                // Inverse CDF approximation: just use uniform for testing
                p * 2.0 - 1.0
            })
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = 0.0;
        let sigma = (1.0f64 / 3.0).sqrt(); // uniform[-1,1] std
        let ks = ks_test_normal(&values, mean, sigma);
        // Should be reasonably small for uniform data
        assert!(ks < 0.5, "KS D too large: {}", ks);
    }

    #[test]
    fn test_profile_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate test vectors
        let fvec = ws.join("test.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", fvec.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "100");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        // Profile
        let model_path = ws.join("model.json");
        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("output", model_path.to_string_lossy().to_string());
        let mut op = AnalyzeProfileOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "profile failed: {}", result.message);

        // Verify model can be deserialized
        let json = std::fs::read_to_string(&model_path).unwrap();
        let model: VectorSpaceModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model.dimensions, 4);
        assert_eq!(model.scalar_models.len(), 4);
    }
}
