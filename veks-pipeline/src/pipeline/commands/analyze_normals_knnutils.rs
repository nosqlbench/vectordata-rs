// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: check vector normalization (knn\_utils personality).
//!
//! Replicates the normalization check from knn\_utils exactly:
//!
//! ```python
//! norms = np.linalg.norm(vecs, axis=1)
//! return np.all(np.abs(norms - 1) < tol)
//! ```
//!
//! - L2 norm computed via BLAS `cblas_snrm2` (the same MKL routine that
//!   numpy calls internally for `np.linalg.norm`), producing **identical**
//!   floating-point results to knn\_utils
//! - Checks **all** vectors (no sampling)
//! - Default tolerance **1e-3** (matching `check_normalization()` in knn\_utils)
//! - Produces a simple boolean `is_normalized` result
//!
//! Also reports basic statistics (zero count, min/max norm) for diagnostics,
//! matching the output of `count_zero_vectors()` and `check_normalization()`
//! from knn\_utils.

use std::path::Path;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

// BLAS snrm2: computes the L2 norm of a float32 vector.
// This is the same routine numpy calls via MKL for np.linalg.norm().
// Linked transitively through FAISS's BLAS dependency (MKL).
unsafe extern "C" {
    fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
}

/// Compute L2 norm of a float32 slice using BLAS `cblas_snrm2`.
///
/// Produces identical results to `np.linalg.norm(vec)` when numpy
/// is backed by the same BLAS (MKL).
fn blas_snrm2(v: &[f32]) -> f32 {
    unsafe { cblas_snrm2(v.len() as i32, v.as_ptr(), 1) }
}

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};

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

/// Pipeline command: knn\_utils-compatible normalization check.
pub struct AnalyzeNormalsKnnUtilsOp;

/// Creates a boxed `AnalyzeNormalsKnnUtilsOp` for command registration.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeNormalsKnnUtilsOp)
}

impl CommandOp for AnalyzeNormalsKnnUtilsOp {
    fn command_path(&self) -> &str {
        "analyze check-normalization-knnutils"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Check normalization using knn_utils conventions (f32, tol=1e-3, all vectors)".into(),
            body: format!(r#"# analyze check-normalization-knnutils

Checks whether all vectors are approximately L2-normalized, replicating
the exact algorithm from knn\_utils `check_normalization()`:

    norms = np.linalg.norm(vecs, axis=1)       # f32 precision
    return np.all(np.abs(norms - 1) < tol)      # tol=1e-3 default

Also counts exact-zero vectors (norm == 0.0), matching knn\_utils
`count_zero_vectors()`.

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
                description: "Vector file to check (fvec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "tolerance".into(),
                type_name: "float".into(),
                required: false,
                default: Some("1e-3".into()),
                description: "Normalization tolerance (knn_utils default: 1e-3)".into(),
                role: OptionRole::Config,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let tolerance: f32 = match options.parse_or("tolerance", 1e-3_f32) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);

        let reader = match MmapVectorReader::<f32>::open_fvec(&input_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open: {}", e), start),
        };
        let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);

        if count == 0 || dim == 0 {
            return error_result("empty vector file", start);
        }

        ctx.ui.log(&format!(
            "  check-normalization-knnutils: {} vectors, dim={}, tol={:.0e} (f32 precision)",
            count, dim, tolerance,
        ));

        // Replicate knn_utils exactly:
        //   norms = np.linalg.norm(vecs, axis=1)  # via MKL cblas_snrm2
        //   zero_count = int(np.sum(norms <= 0.0))
        //   is_normalized = np.all(np.abs(norms - 1) < tol)
        let pb = ctx.ui.bar_with_unit(count as u64, "checking norms", "vectors");
        let mut zero_count: u64 = 0;
        let mut all_normalized = true;
        let mut min_norm: f32 = f32::INFINITY;
        let mut max_norm: f32 = f32::NEG_INFINITY;
        let mut fail_count: u64 = 0;

        for i in 0..count {
            let slice = reader.get_slice(i);

            // L2 norm via BLAS cblas_snrm2 — identical to numpy's
            // np.linalg.norm() when both use MKL.
            let norm = blas_snrm2(slice);

            // count_zero_vectors: norm <= 0.0 (exact zeros)
            if norm <= 0.0 {
                zero_count += 1;
            }

            // check_normalization: abs(norm - 1) < tol
            if (norm - 1.0_f32).abs() >= tolerance {
                all_normalized = false;
                fail_count += 1;
            }

            if norm < min_norm { min_norm = norm; }
            if norm > max_norm { max_norm = norm; }

            if (i + 1) % 100_000 == 0 {
                pb.set_position((i + 1) as u64);
            }
        }
        pb.finish();

        let is_normalized = all_normalized;

        ctx.ui.log(&format!("  zero vectors: {} / {}", zero_count, count));
        ctx.ui.log(&format!(
            "  normalized: {} (tol={:.0e}, {} failed out of {})",
            if is_normalized { "Yes" } else { "No" },
            tolerance, fail_count, count,
        ));
        ctx.ui.log(&format!(
            "  norm range: [{:.8}, {:.8}]",
            min_norm, max_norm,
        ));

        // Write variables matching knn_utils output conventions
        let vars = [
            ("is_normalized", is_normalized.to_string()),
            ("zero_count", zero_count.to_string()),
            ("normalization_tolerance", format!("{:.0e}", tolerance)),
        ];
        for (name, value) in &vars {
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, name, value);
            ctx.defaults.insert(name.to_string(), value.clone());
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "normalized={}, zeros={}, norm_range=[{:.8},{:.8}] ({} vectors, tol={:.0e})",
                is_normalized, zero_count, min_norm, max_norm, count, tolerance,
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &[],
        )
    }
}
