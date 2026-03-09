// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compare vector distributions.
//!
//! Performs a two-sample Kolmogorov-Smirnov test per dimension between two
//! fvec files to determine if their distributions match. This is useful for
//! comparing synthetic/generated data against the original dataset.
//!
//! Equivalent to the Java `CMD_analyze_compare` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::io::MmapVectorReader;
use vectordata::io::VectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: K-S distribution comparison.
pub struct AnalyzeCompareOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeCompareOp)
}

/// Result of a K-S test on one dimension.
struct KsResult {
    dimension: usize,
    d_statistic: f64,
    p_value: f64,
    passed: bool,
}

/// Two-sample Kolmogorov-Smirnov test.
///
/// Returns (D-statistic, approximate p-value).
fn ks_test(a: &mut [f64], b: &mut [f64]) -> (f64, f64) {
    a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = a.len() as f64;
    let n2 = b.len() as f64;

    // Merge-walk both sorted arrays to find max CDF difference
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_max = 0.0f64;

    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if b[j] < a[i] {
            j += 1;
        } else {
            // Equal: advance both
            i += 1;
            j += 1;
        }
        let diff = ((i as f64 / n1) - (j as f64 / n2)).abs();
        if diff > d_max {
            d_max = diff;
        }
    }

    // Approximate p-value using the asymptotic Kolmogorov distribution
    let z = d_max * (n1 * n2 / (n1 + n2)).sqrt();
    let p_value = ks_p_value(z);

    (d_max, p_value)
}

/// Approximate p-value from K-S z-statistic using the asymptotic series.
///
/// P ≈ 2 * sum_{k=1..100} (-1)^(k-1) * exp(-2 * k^2 * z^2)
fn ks_p_value(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }

    let mut sum = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * (k as f64) * (k as f64) * z * z).exp();
        sum += term;
        // Early termination when terms become negligible
        if term.abs() < 1e-15 {
            break;
        }
    }

    (2.0 * sum).clamp(0.0, 1.0)
}

impl CommandOp for AnalyzeCompareOp {
    fn command_path(&self) -> &str {
        "analyze compare"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Compare two vector files element-by-element".into(),
            body: format!(
                "# analyze compare\n\nCompare two vector files element-by-element.\n\n## Description\n\nPerforms a two-sample Kolmogorov-Smirnov test per dimension between two fvec files to determine if their distributions match. Useful for comparing synthetic or generated data against the original dataset.\n\n## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Buffers for two vector files".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let original_str = match options.require("original") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let synthetic_str = match options.require("synthetic") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let alpha: f64 = options
            .get("alpha")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.05);
        let sample_size: usize = options
            .get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);
        let verbose = options.get("verbose").map_or(false, |s| s == "true");
        let single_dim: Option<usize> = options.get("dimension").and_then(|s| s.parse().ok());

        let orig_path = resolve_path(original_str, &ctx.workspace);
        let synth_path = resolve_path(synthetic_str, &ctx.workspace);

        // Open both files
        let orig_reader = match MmapVectorReader::<f32>::open_fvec(&orig_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open original {}: {}", orig_path.display(), e),
                    start,
                )
            }
        };
        let synth_reader = match MmapVectorReader::<f32>::open_fvec(&synth_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open synthetic {}: {}", synth_path.display(), e),
                    start,
                )
            }
        };

        let orig_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&orig_reader);
        let synth_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&synth_reader);

        if orig_dim != synth_dim {
            return error_result(
                format!(
                    "dimension mismatch: original={}, synthetic={}",
                    orig_dim, synth_dim
                ),
                start,
            );
        }

        let orig_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&orig_reader);
        let synth_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&synth_reader);

        let orig_sample = sample_size.min(orig_count);
        let synth_sample = sample_size.min(synth_count);

        // Determine which dimensions to test
        let dims: Vec<usize> = match single_dim {
            Some(d) if d < orig_dim => vec![d],
            Some(d) => {
                return error_result(
                    format!("dimension {} out of range (max {})", d, orig_dim - 1),
                    start,
                )
            }
            None => (0..orig_dim).collect(),
        };

        // Pre-read sampled vectors
        let orig_vecs: Vec<Vec<f32>> = (0..orig_sample)
            .filter_map(|i| orig_reader.get(i).ok())
            .collect();
        let synth_vecs: Vec<Vec<f32>> = (0..synth_sample)
            .filter_map(|i| synth_reader.get(i).ok())
            .collect();

        // Run K-S test per dimension
        let mut results = Vec::with_capacity(dims.len());
        let mut fail_count = 0usize;

        for &d in &dims {
            let mut orig_vals: Vec<f64> = orig_vecs.iter().map(|v| v[d] as f64).collect();
            let mut synth_vals: Vec<f64> = synth_vecs.iter().map(|v| v[d] as f64).collect();

            let (d_stat, p_val) = ks_test(&mut orig_vals, &mut synth_vals);
            let passed = p_val >= alpha;
            if !passed {
                fail_count += 1;
            }

            results.push(KsResult {
                dimension: d,
                d_statistic: d_stat,
                p_value: p_val,
                passed,
            });
        }

        // Print results table
        ctx.ui.log(&format!(
            "\nK-S Distribution Comparison (alpha={}, samples: orig={}, synth={})",
            alpha, orig_sample, synth_sample
        ));
        ctx.ui.log(&format!("{:>8}  {:>10}  {:>10}  {:>6}", "Dim", "D-stat", "P-value", "Result"));
        ctx.ui.log(&format!("{}", "-".repeat(42)));

        let show_count = if verbose { results.len() } else { 10.min(results.len()) };
        let mut shown = 0;

        for r in &results {
            if verbose || shown < 10 || !r.passed {
                ctx.ui.log(&format!(
                    "{:>8}  {:>10.6}  {:>10.6}  {:>6}",
                    r.dimension,
                    r.d_statistic,
                    r.p_value,
                    if r.passed { "PASS" } else { "FAIL" }
                ));
                shown += 1;
            }
        }

        if !verbose && results.len() > show_count {
            ctx.ui.log(&format!("  ... {} more dimensions (use verbose=true to see all)", results.len() - shown));
        }

        ctx.ui.log("");
        if fail_count == 0 {
            ctx.ui.log(&format!("DISTRIBUTIONS MATCH ({}/{} dimensions passed)", dims.len(), dims.len()));
        } else {
            ctx.ui.log(&format!(
                "DISTRIBUTIONS DIFFER ({}/{} dimensions failed)",
                fail_count,
                dims.len()
            ));
        }

        let status = if fail_count == 0 {
            Status::Ok
        } else {
            Status::Warning
        };

        CommandResult {
            status,
            message: format!(
                "K-S test: {}/{} dims passed (alpha={})",
                dims.len() - fail_count,
                dims.len(),
                alpha
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
                description: "Original/reference fvec file".to_string(),
            },
            OptionDesc {
                name: "synthetic".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Synthetic/comparison fvec file".to_string(),
            },
            OptionDesc {
                name: "alpha".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.05".to_string()),
                description: "Significance level for K-S test".to_string(),
            },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("10000".to_string()),
                description: "Max vectors to sample from each file".to_string(),
            },
            OptionDesc {
                name: "verbose".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show all dimensions instead of first 10 + failures".to_string(),
            },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Test only this specific dimension".to_string(),
            },
        ]
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
    use std::io::Write;

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

    #[test]
    fn test_ks_identical() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (d, p) = ks_test(&mut a, &mut b);
        assert_eq!(d, 0.0);
        assert!((p - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ks_different() {
        let mut a: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let mut b: Vec<f64> = (0..100).map(|i| 0.5 + i as f64 * 0.01).collect();
        let (d, p) = ks_test(&mut a, &mut b);
        assert!(d > 0.3);
        assert!(p < 0.05);
    }

    #[test]
    fn test_compare_identical_files() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate identical vectors
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) * 0.1, (i as f32) * 0.2])
            .collect();

        let orig = ws.join("original.fvec");
        let synth = ws.join("synthetic.fvec");
        write_fvec(&orig, &vectors);
        write_fvec(&synth, &vectors);

        let mut opts = Options::new();
        opts.set("original", orig.to_string_lossy().to_string());
        opts.set("synthetic", synth.to_string_lossy().to_string());

        let mut op = AnalyzeCompareOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("2/2 dims passed"));
    }

    #[test]
    fn test_compare_different_files() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let orig_vecs: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![(i as f32) * 0.01, (i as f32) * 0.02])
            .collect();
        let synth_vecs: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![50.0 + (i as f32) * 0.01, 100.0 + (i as f32) * 0.02])
            .collect();

        let orig = ws.join("original.fvec");
        let synth = ws.join("synthetic.fvec");
        write_fvec(&orig, &orig_vecs);
        write_fvec(&synth, &synth_vecs);

        let mut opts = Options::new();
        opts.set("original", orig.to_string_lossy().to_string());
        opts.set("synthetic", synth.to_string_lossy().to_string());

        let mut op = AnalyzeCompareOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
    }

    #[test]
    fn test_compare_single_dimension() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) * 0.1, (i as f32) * 0.2])
            .collect();

        let orig = ws.join("original.fvec");
        let synth = ws.join("synthetic.fvec");
        write_fvec(&orig, &vectors);
        write_fvec(&synth, &vectors);

        let mut opts = Options::new();
        opts.set("original", orig.to_string_lossy().to_string());
        opts.set("synthetic", synth.to_string_lossy().to_string());
        opts.set("dimension", "0");

        let mut op = AnalyzeCompareOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("1/1 dims passed"));
    }
}
