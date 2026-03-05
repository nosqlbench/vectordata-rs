// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compare two VectorSpaceModel JSON files.
//!
//! Loads two model files and compares them dimension-by-dimension, computing
//! parameter drift percentages and reporting PASS/FAIL based on thresholds.
//!
//! Equivalent to the Java `CMD_analyze_model_diff` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

/// Pipeline command: diff two model.json files.
pub struct AnalyzeModelDiffOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeModelDiffOp)
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

impl ScalarModelDef {
    fn type_name(&self) -> &str {
        match self {
            ScalarModelDef::Normal { .. } => "normal",
            ScalarModelDef::Uniform { .. } => "uniform",
            ScalarModelDef::Beta { .. } => "beta",
        }
    }

    /// Compute normalized drift percentage against another model of the same type.
    fn drift(&self, other: &ScalarModelDef) -> Option<f64> {
        match (self, other) {
            (
                ScalarModelDef::Normal { mu: m1, sigma: s1 },
                ScalarModelDef::Normal { mu: m2, sigma: s2 },
            ) => {
                let mu_drift = param_drift(*m1, *m2);
                let sigma_drift = param_drift(*s1, *s2);
                Some((mu_drift + sigma_drift) / 2.0)
            }
            (
                ScalarModelDef::Uniform { lower: l1, upper: u1 },
                ScalarModelDef::Uniform { lower: l2, upper: u2 },
            ) => {
                let lower_drift = param_drift(*l1, *l2);
                let upper_drift = param_drift(*u1, *u2);
                Some((lower_drift + upper_drift) / 2.0)
            }
            (
                ScalarModelDef::Beta { alpha: a1, beta: b1, lower: l1, upper: u1 },
                ScalarModelDef::Beta { alpha: a2, beta: b2, lower: l2, upper: u2 },
            ) => {
                let ad = param_drift(*a1, *a2);
                let bd = param_drift(*b1, *b2);
                let ld = param_drift(*l1, *l2);
                let ud = param_drift(*u1, *u2);
                Some((ad + bd + ld + ud) / 4.0)
            }
            _ => None, // type mismatch
        }
    }
}

/// Normalized parameter drift as a percentage.
fn param_drift(a: f64, b: f64) -> f64 {
    let denom = a.abs().max(b.abs()).max(1e-10);
    ((a - b).abs() / denom) * 100.0
}

impl CommandOp for AnalyzeModelDiffOp {
    fn command_path(&self) -> &str {
        "analyze model-diff"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let orig_str = match options.require("original") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let comp_str = match options.require("compare") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let orig_path = resolve_path(orig_str, &ctx.workspace);
        let comp_path = resolve_path(comp_str, &ctx.workspace);

        let drift_threshold: f64 = options
            .get("drift-threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let max_drift_threshold: f64 = options
            .get("max-drift-threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0);
        let verbose = options.get("verbose").map_or(false, |s| s == "true");

        let orig = match load_model(&orig_path) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };
        let comp = match load_model(&comp_path) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };

        if orig.dimensions != comp.dimensions {
            return error_result(
                format!(
                    "dimension count mismatch: original={}, compare={}",
                    orig.dimensions, comp.dimensions
                ),
                start,
            );
        }

        let dim_count = orig.scalar_models.len();
        let mut type_matches = 0usize;
        let mut total_drift = 0.0f64;
        let mut max_drift = 0.0f64;
        let mut problem_dims: Vec<usize> = Vec::new();

        eprintln!(
            "{:<6} {:<10} {:<10} {:>10}  {}",
            "Dim", "Original", "Compare", "Drift%", "Status"
        );
        eprintln!("{}", "-".repeat(52));

        for i in 0..dim_count {
            let o = &orig.scalar_models[i];
            let c = &comp.scalar_models[i];
            let types_match = o.type_name() == c.type_name();
            if types_match {
                type_matches += 1;
            }

            let drift = o.drift(c).unwrap_or(100.0);
            total_drift += drift;
            if drift > max_drift {
                max_drift = drift;
            }

            let status = if !types_match {
                problem_dims.push(i);
                "TYPE MISMATCH"
            } else if drift > max_drift_threshold {
                problem_dims.push(i);
                "HIGH DRIFT"
            } else {
                "ok"
            };

            let show = verbose || i < 10 || problem_dims.contains(&i) || i == dim_count - 1;
            if show {
                eprintln!(
                    "{:<6} {:<10} {:<10} {:>9.2}%  {}",
                    i,
                    o.type_name(),
                    c.type_name(),
                    drift,
                    status
                );
            } else if i == 10 {
                eprintln!("  ...");
            }
        }

        let avg_drift = total_drift / dim_count as f64;
        let type_match_pct = (type_matches as f64 / dim_count as f64) * 100.0;

        eprintln!();
        eprintln!("Summary:");
        eprintln!("  Dimensions:    {}", dim_count);
        eprintln!("  Type match:    {:.1}% ({}/{})", type_match_pct, type_matches, dim_count);
        eprintln!("  Avg drift:     {:.2}%", avg_drift);
        eprintln!("  Max drift:     {:.2}%", max_drift);
        eprintln!("  Problem dims:  {}", problem_dims.len());

        let pass = avg_drift <= drift_threshold
            && max_drift <= max_drift_threshold
            && type_match_pct >= 100.0;

        if pass {
            eprintln!("  Result: PASS");
        } else {
            eprintln!("  Result: FAIL");
        }

        CommandResult {
            status: if pass { Status::Ok } else { Status::Error },
            message: format!(
                "avg drift={:.2}%, max drift={:.2}%, type match={:.1}%",
                avg_drift, max_drift, type_match_pct
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "original".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Original model.json file".to_string(),
            },
            OptionDesc {
                name: "compare".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Model.json file to compare against".to_string(),
            },
            OptionDesc {
                name: "drift-threshold".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("1.0".to_string()),
                description: "Max allowed average drift percentage".to_string(),
            },
            OptionDesc {
                name: "max-drift-threshold".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("2.0".to_string()),
                description: "Max allowed single-dimension drift percentage".to_string(),
            },
            OptionDesc {
                name: "verbose".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show all dimensions".to_string(),
            },
        ]
    }
}

fn load_model(path: &Path) -> Result<VectorSpaceModel, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {}", path.display(), e))
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

    #[test]
    fn test_identical_models_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let model = r#"{"dimensions":2,"uniqueVectors":100,"scalarModels":[{"type":"normal","mu":0.0,"sigma":1.0},{"type":"normal","mu":0.5,"sigma":0.3}]}"#;
        std::fs::write(ws.join("a.json"), model).unwrap();
        std::fs::write(ws.join("b.json"), model).unwrap();

        let mut opts = Options::new();
        opts.set("original", ws.join("a.json").to_string_lossy().to_string());
        opts.set("compare", ws.join("b.json").to_string_lossy().to_string());
        let mut op = AnalyzeModelDiffOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_drifted_models_fail() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let a = r#"{"dimensions":1,"uniqueVectors":100,"scalarModels":[{"type":"normal","mu":0.0,"sigma":1.0}]}"#;
        let b = r#"{"dimensions":1,"uniqueVectors":100,"scalarModels":[{"type":"normal","mu":0.5,"sigma":1.5}]}"#;
        std::fs::write(ws.join("a.json"), a).unwrap();
        std::fs::write(ws.join("b.json"), b).unwrap();

        let mut opts = Options::new();
        opts.set("original", ws.join("a.json").to_string_lossy().to_string());
        opts.set("compare", ws.join("b.json").to_string_lossy().to_string());
        let mut op = AnalyzeModelDiffOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }

    #[test]
    fn test_param_drift() {
        assert!((param_drift(1.0, 1.0)).abs() < 1e-10);
        assert!((param_drift(1.0, 1.01) - 1.0).abs() < 0.1);
        assert!((param_drift(0.0, 0.0)).abs() < 1e-10);
    }
}
