// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: fvecs file validation (knn\_utils personality).
//!
//! Replicates the logic of `fvecs_check.py` from the knn\_utils project,
//! including the same underlying library calls:
//!
//! - Per-vector L2 norm via BLAS `cblas_snrm2` (same as `np.linalg.norm`)
//! - Normalization check: `abs(norm - 1) > tol_norm` (default 1e-5)
//! - Zero vector check: `norm < tol_zero` (default 1e-6)
//! - Dimension consistency validation across all vectors
//! - Norm statistics computed in f64 (matching `compute_norm_stats`)
//! - Log-space histogram of norms (matching `compute_histogram_data`)
//! - Report file output
//!
//! This command operates on the raw source file, before any dedup or
//! extraction steps, so it captures the true zero and normalization
//! counts of the original data.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};

// BLAS snrm2: same routine knn_utils calls via np.linalg.norm(vector).
unsafe extern "C" {
    fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
}

/// Compute L2 norm of a float32 slice using BLAS `cblas_snrm2`.
fn blas_snrm2(v: &[f32]) -> f32 {
    unsafe { cblas_snrm2(v.len() as i32, v.as_ptr(), 1) }
}

fn error_result(message: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: message.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve_path(value: &str, workspace: &Path) -> std::path::PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

/// Pipeline command: knn\_utils-compatible fvecs validation.
pub struct AnalyzeFvecsCheckKnnUtilsOp;

/// Creates a boxed `AnalyzeFvecsCheckKnnUtilsOp` for command registration.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeFvecsCheckKnnUtilsOp)
}

impl CommandOp for AnalyzeFvecsCheckKnnUtilsOp {
    fn command_path(&self) -> &str {
        "analyze fvecs-check-knnutils"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Validate fvecs file using knn_utils conventions".into(),
            body: format!(r#"# analyze fvecs-check-knnutils

Validates an fvecs file, replicating the exact logic of `fvecs_check.py`
from the knn\_utils project.

## Checks Performed

- **Dimension consistency**: every vector must have the same dimension
- **Normalization**: per-vector L2 norm via BLAS `cblas_snrm2`, flag
  vectors where `abs(norm - 1) > tol_norm`
- **Zero vectors**: count vectors where `norm < tol_zero`
- **Norm statistics**: min, max, mean, max absolute deviation from 1.0
  (computed in f64, matching `compute_norm_stats`)
- **Histogram**: log-space binned norm distribution (matching
  `compute_histogram_data`)

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "fvecs file to validate".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "tol-norm".into(),
                type_name: "float".into(),
                required: false,
                default: Some("1e-5".into()),
                description: "Normalization tolerance (knn_utils default: 1e-5)".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "tol-zero".into(),
                type_name: "float".into(),
                required: false,
                default: Some("1e-6".into()),
                description: "Zero-vector tolerance (knn_utils default: 1e-6)".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "report".into(),
                type_name: "Path".into(),
                required: false,
                default: None,
                description: "Output report file path (default: <input>_fvecs_check_report.txt)".into(),
                role: OptionRole::Output,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let tol_norm: f64 = match options.parse_or("tol-norm", 1e-5_f64) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let tol_zero: f64 = match options.parse_or("tol-zero", 1e-6_f64) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);

        // Default report path: <input>_fvecs_check_report.txt
        let report_path = match options.get("report") {
            Some(s) => resolve_path(s, &ctx.workspace),
            None => {
                let stem = input_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("vectors");
                input_path.with_file_name(format!("{}_fvecs_check_report.txt", stem))
            }
        };

        let reader = match MmapVectorReader::<f32>::open_fvec(&input_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open: {}", e), start),
        };
        let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);

        if count == 0 || dim == 0 {
            return error_result("no valid vector data found in the file", start);
        }

        ctx.ui.log(&format!(
            "  fvecs-check-knnutils: {} vectors, dim={}, tol_norm={:.0e}, tol_zero={:.0e}",
            count, dim, tol_norm, tol_zero,
        ));

        // Stream through all vectors, collecting norms (matching fvecs_check.py)
        let pb = ctx.ui.bar_with_unit(count as u64, "checking vectors", "vectors");
        let mut norms: Vec<f64> = Vec::with_capacity(count);
        let mut zero_count: u64 = 0;
        let mut unnormalized_count: u64 = 0;
        let mut normalized = true;

        for i in 0..count {
            let slice = reader.get_slice(i);

            // Per-vector L2 norm via BLAS cblas_snrm2
            // Matches: norm_val = np.linalg.norm(vector)
            let norm_val = blas_snrm2(slice) as f64;
            norms.push(norm_val);

            // Matches: if (abs(norm_val - 1) > tol_norm): normalized = False
            if (norm_val - 1.0).abs() > tol_norm {
                normalized = false;
                unnormalized_count += 1;
            }

            // Matches: if norm_val < tol_zero: zero_count += 1
            if norm_val < tol_zero {
                zero_count += 1;
            }

            if (i + 1) % 100_000 == 0 {
                pb.set_position((i + 1) as u64);
            }
        }
        pb.finish();

        // Compute norm statistics in f64 (matching compute_norm_stats)
        let n = norms.len() as f64;
        let min_norm = norms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_norm = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_norm = norms.iter().sum::<f64>() / n;
        let max_abs_dev = norms.iter()
            .map(|&v| (v - 1.0).abs())
            .fold(0.0_f64, f64::max);

        // Compute histogram (matching compute_histogram_data)
        // Exact zeros get their own bin; non-zero norms get log-space bins
        let exact_zeros: u64 = norms.iter().filter(|&&v| v == 0.0).count() as u64;
        let non_zero_norms: Vec<f64> = norms.iter().cloned().filter(|&v| v != 0.0).collect();

        let histogram_lines = if non_zero_norms.is_empty() {
            "All vectors are exact zero; skipping non-zero histogram bins.".to_string()
        } else {
            let nz_min = non_zero_norms.iter().cloned().fold(f64::INFINITY, f64::min);
            let nz_max = non_zero_norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // 101 log-space bin edges (matching np.logspace(..., num=101))
            let num_bins = 100;
            let log_min = nz_min.log10();
            let log_max = nz_max.log10();
            let mut edges: Vec<f64> = Vec::with_capacity(num_bins + 1);
            if (nz_max - nz_min).abs() < f64::EPSILON {
                edges.push(nz_min);
                edges.push(nz_min * 1.01);
            } else {
                for i in 0..=num_bins {
                    let t = i as f64 / num_bins as f64;
                    edges.push(10.0_f64.powf(log_min + t * (log_max - log_min)));
                }
            }

            let mut counts = vec![0u64; edges.len() - 1];
            for &v in &non_zero_norms {
                // Binary search for the bin (matching np.histogram)
                let mut lo = 0;
                let mut hi = edges.len() - 1;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if v >= edges[mid + 1] { lo = mid + 1; } else { hi = mid; }
                }
                // Last bin is inclusive on the right (matching numpy)
                let bin = lo.min(counts.len() - 1);
                counts[bin] += 1;
            }

            let mut lines = Vec::new();
            for (i, &c) in counts.iter().enumerate() {
                if c > 0 {
                    lines.push(format!("{:>18.9e} {:>18.9e} {:10}", edges[i], edges[i + 1], c));
                }
            }
            lines.join("\n")
        };

        // Display results
        ctx.ui.log(&format!("  total embeddings: {}", count));
        ctx.ui.log(&format!("  dimensionality: {}", dim));
        ctx.ui.log(&format!("  zero vectors (< {:.0e}): {}", tol_zero, zero_count));
        ctx.ui.log(&format!("  unnormalized vectors (abs(||v||_2 - 1) > {:.0e}): {}",
            tol_norm, unnormalized_count));
        ctx.ui.log(&format!("  norm max abs deviation from 1.0: {:.9e}", max_abs_dev));
        ctx.ui.log(&format!("  norm mean: {:.9e}", mean_norm));
        ctx.ui.log(&format!("  norm range: [{:.9e}, {:.9e}]", min_norm, max_norm));

        // Write report file (matching fvecs_check.py report format)
        let report_content = format!(
            "Input file: {}\n\
             Total embeddings: {}\n\
             Dimensionality: {}\n\
             Zero vectors (< {:.0e}): {}\n\
             Unnormalized vectors (abs(||v||_2 - 1.0) > {:.0e}): {}\n\
             Norm max abs deviation from 1.0: {:.9e}\n\
             Norm mean: {:.9e}\n\
             \n\
             Histogram summary:\n\
             {:>18} {:>18} {:>10}\n\
             {}{}\n",
            input_path.display(),
            count, dim,
            tol_zero, zero_count,
            tol_norm, unnormalized_count,
            max_abs_dev, mean_norm,
            "Bin start", "Bin end", "Count",
            if exact_zeros > 0 {
                format!("{:>18} {:>18} {:10}\n", "0.0 (exact)", "0.0 (exact)", exact_zeros)
            } else {
                String::new()
            },
            histogram_lines,
        );

        let mut produced = Vec::new();
        if let Some(parent) = report_path.parent() {
            if !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match std::fs::File::create(&report_path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(report_content.as_bytes()) {
                    ctx.ui.log(&format!("  warning: failed to write report: {}", e));
                } else {
                    ctx.ui.log(&format!("  report saved to {}", report_path.display()));
                    produced.push(report_path);
                }
            }
            Err(e) => {
                ctx.ui.log(&format!("  warning: failed to create report file: {}", e));
            }
        }

        // Write pipeline variables
        let vars = [
            ("is_normalized", normalized.to_string()),
            ("zero_count", zero_count.to_string()),
            ("unnormalized_count", unnormalized_count.to_string()),
            ("norm_min", format!("{:.9e}", min_norm)),
            ("norm_max", format!("{:.9e}", max_norm)),
            ("norm_mean", format!("{:.9e}", mean_norm)),
            ("norm_max_abs_deviation", format!("{:.9e}", max_abs_dev)),
        ];
        for (name, value) in &vars {
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, name, value);
            ctx.defaults.insert(name.to_string(), value.clone());
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} vectors, dim={}, normalized={}, zeros={}, unnormalized={}",
                count, dim, normalized, zero_count, unnormalized_count,
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["report"],
        )
    }
}
