// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: verify generated vectors match their statistical model.
//!
//! Loads a model.json and a generated fvec file, then performs per-dimension
//! Kolmogorov-Smirnov tests to verify the vectors match the model's
//! statistical profile within acceptable tolerances.
//!
//! Equivalent to the Java `CMD_analyze_verifyprofiles` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: verify vectors match a model profile.
pub struct AnalyzeVerifyProfilesOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeVerifyProfilesOp)
}

#[derive(Debug, Deserialize)]
struct VectorSpaceModel {
    dimensions: u32,
    #[serde(rename = "scalarModels")]
    scalar_models: Vec<ScalarModelDef>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ScalarModelDef {
    Normal { mu: f64, sigma: f64 },
    Uniform { lower: f64, upper: f64 },
    Beta { alpha: f64, beta: f64, lower: f64, upper: f64 },
}

impl CommandOp for AnalyzeVerifyProfilesOp {
    fn command_path(&self) -> &str {
        "analyze verify-profiles"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify dataset profile facet consistency".into(),
            body: format!(
                "# analyze verify-profiles\n\n\
                Verify dataset profile facet consistency.\n\n\
                ## Description\n\n\
                Loads a VectorSpaceModel JSON file (as produced by `analyze profile`) \
                and a generated fvec file, then performs per-dimension Kolmogorov-Smirnov \
                (KS) tests to verify that the vectors match the model's statistical \
                profile within acceptable tolerances.\n\n\
                ## How It Works\n\n\
                For each dimension, the command:\n\n\
                1. Extracts sampled values from the vector file and sorts them.\n\
                2. Computes the theoretical CDF from the model definition (Normal CDF \
                via Abramowitz-Stegun approximation, Uniform CDF, or Beta CDF via \
                trapezoidal numerical integration).\n\
                3. Walks the sorted values to find the maximum difference between the \
                empirical and theoretical CDFs (the KS D-statistic).\n\
                4. Compares the D-statistic against the configurable threshold to \
                determine pass/fail for that dimension.\n\n\
                The overall result is PASS only if all dimensions are within the \
                threshold. Failed dimensions are always printed regardless of the \
                verbose setting.\n\n\
                ## Role in Dataset Preparation\n\n\
                This command is the counterpart to `analyze profile` and `generate \
                from-model`. After generating synthetic vectors from a statistical \
                model, running verify-profiles confirms that the generated data \
                actually follows the intended distribution. This catches bugs in the \
                random number generation, distribution sampling code, or parameter \
                serialization. It is typically the final validation step before \
                using generated vectors for benchmarking.\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Vector data buffers".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let model_str = match options.require("model") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let vectors_str = match options.require("vectors") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let model_path = resolve_path(model_str, &ctx.workspace);
        let vectors_path = resolve_path(vectors_str, &ctx.workspace);

        let threshold: f64 = options
            .get("threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.05);
        let sample: usize = options
            .get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000);
        let verbose = options.get("verbose").map_or(false, |s| s == "true");

        // Load model
        let content = match std::fs::read_to_string(&model_path) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to read model: {}", e), start),
        };
        let model: VectorSpaceModel = match serde_json::from_str(&content) {
            Ok(m) => m,
            Err(e) => return error_result(format!("failed to parse model: {}", e), start),
        };

        // Open vectors
        let reader = match MmapVectorReader::<f32>::open_fvec(&vectors_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open vectors {}: {}", vectors_path.display(), e), start),
        };

        let vec_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let vec_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);

        if vec_dim != model.dimensions as usize {
            return error_result(
                format!(
                    "dimension mismatch: model={}, vectors={}",
                    model.dimensions, vec_dim
                ),
                start,
            );
        }

        let effective = sample.min(vec_count);
        ctx.ui.log(&format!(
            "Verifying {} vectors (sampling {}) against model ({} dims)",
            vec_count, effective, model.dimensions
        ));

        let mut pass_count = 0;
        let mut fail_count = 0;
        let mut total_ks = 0.0f64;

        for d in 0..vec_dim {
            let mut values: Vec<f64> = Vec::with_capacity(effective);
            for i in 0..effective {
                if let Ok(vec) = reader.get(i) {
                    values.push(vec[d] as f64);
                }
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let ks_d = ks_test(&values, &model.scalar_models[d]);
            total_ks += ks_d;
            let pass = ks_d <= threshold;

            if pass {
                pass_count += 1;
            } else {
                fail_count += 1;
            }

            if verbose || !pass {
                let status = if pass { "ok" } else { "FAIL" };
                ctx.ui.log(&format!(
                    "  dim[{}]: KS D={:.4} (threshold={:.4}) — {}",
                    d, ks_d, threshold, status
                ));
            }
        }

        let avg_ks = total_ks / vec_dim as f64;
        ctx.ui.log("");
        ctx.ui.log(&format!(
            "Results: {}/{} dimensions pass (avg KS D={:.4}, threshold={:.4})",
            pass_count, vec_dim, avg_ks, threshold
        ));

        if fail_count == 0 {
            ctx.ui.log("PASS");
            CommandResult {
                status: Status::Ok,
                message: format!(
                    "all {} dimensions pass, avg KS D={:.4}",
                    vec_dim, avg_ks
                ),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        } else {
            ctx.ui.log(&format!("FAIL ({} dimensions exceeded threshold)", fail_count));
            CommandResult {
                status: Status::Error,
                message: format!(
                    "{}/{} dimensions failed, avg KS D={:.4}",
                    fail_count, vec_dim, avg_ks
                ),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "model".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Model JSON file to verify against".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "vectors".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Vector fvec file to verify".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "threshold".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.05".to_string()),
                description: "KS D-statistic threshold for pass/fail".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("10000".to_string()),
                description: "Max vectors to sample".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "verbose".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show per-dimension results".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Compute KS D-statistic for sorted values against a model CDF.
fn ks_test(sorted: &[f64], model: &ScalarModelDef) -> f64 {
    let n = sorted.len() as f64;
    let mut max_d = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let empirical = (i + 1) as f64 / n;
        let theoretical = model_cdf(x, model);
        max_d = max_d.max((empirical - theoretical).abs());
    }
    max_d
}

/// CDF for a model distribution at point x.
fn model_cdf(x: f64, model: &ScalarModelDef) -> f64 {
    match model {
        ScalarModelDef::Normal { mu, sigma } => {
            let z = (x - mu) / sigma.max(1e-10);
            normal_cdf(z)
        }
        ScalarModelDef::Uniform { lower, upper } => {
            if x <= *lower {
                0.0
            } else if x >= *upper {
                1.0
            } else {
                (x - lower) / (upper - lower)
            }
        }
        ScalarModelDef::Beta { alpha, beta, lower, upper } => {
            if x <= *lower {
                0.0
            } else if x >= *upper {
                1.0
            } else {
                let normalized = (x - lower) / (upper - lower);
                incomplete_beta_approx(normalized.clamp(0.0, 1.0), *alpha, *beta)
            }
        }
    }
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327 * (-x * x / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.330274))));
    if x >= 0.0 { 1.0 - p } else { p }
}

/// Approximate regularized incomplete beta function via trapezoidal integration.
fn incomplete_beta_approx(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
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

fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 { return 0.0; }
    let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - log_beta(a, b);
    log_pdf.exp()
}

fn log_beta(a: f64, b: f64) -> f64 {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut sum = c[0];
    for (i, &coeff) in c[1..].iter().enumerate() {
        sum += coeff / (x + i as f64 + 1.0);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + t.ln() * (x + 0.5) - t + sum.ln()
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
        }
    }

    #[test]
    fn test_verify_uniform_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate random vectors (uniform 0..1, dim=2, n=1000)
        let fvec = ws.join("test.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", fvec.to_string_lossy().to_string());
        gen_opts.set("dimension", "2");
        gen_opts.set("count", "1000");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        // Build a matching model directly. The generator produces
        // uniform(-1, 1) on each dimension by default.
        let model_json = r#"{"dimensions":2,"uniqueVectors":1000,"scalarModels":[{"type":"uniform","lower":-1.0,"upper":1.0},{"type":"uniform","lower":-1.0,"upper":1.0}]}"#;
        let model_path = ws.join("model.json");
        std::fs::write(&model_path, model_json).unwrap();

        let mut opts = Options::new();
        opts.set("model", model_path.to_string_lossy().to_string());
        opts.set("vectors", fvec.to_string_lossy().to_string());
        opts.set("threshold", "0.1");
        let mut op = AnalyzeVerifyProfilesOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "verify failed: {}", result.message);
    }

    #[test]
    fn test_verify_wrong_model_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate uniform 0..1 vectors
        let fvec = ws.join("test.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", fvec.to_string_lossy().to_string());
        gen_opts.set("dimension", "2");
        gen_opts.set("count", "1000");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        // Create a model with very different parameters
        let model_json = r#"{"dimensions":2,"uniqueVectors":1000,"scalarModels":[{"type":"normal","mu":100.0,"sigma":0.001},{"type":"normal","mu":100.0,"sigma":0.001}]}"#;
        let model_path = ws.join("model.json");
        std::fs::write(&model_path, model_json).unwrap();

        let mut opts = Options::new();
        opts.set("model", model_path.to_string_lossy().to_string());
        opts.set("vectors", fvec.to_string_lossy().to_string());
        opts.set("threshold", "0.05");
        let mut op = AnalyzeVerifyProfilesOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }
}
