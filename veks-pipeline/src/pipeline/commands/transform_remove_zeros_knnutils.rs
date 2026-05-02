// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: remove zero vectors (knn\_utils personality).
//!
//! Replicates `fvecs_remove_zeros.py` from knn\_utils:
//!
//! ```python
//! norms = np.linalg.norm(arr, axis=1)
//! keep_mask = norms > tol
//! return arr[keep_mask]
//! ```
//!
//! Reads an fvec file, computes the L2 norm of each vector via BLAS
//! `cblas_snrm2`, and writes only vectors with `norm > tolerance` to
//! the output. Default tolerance is 0.0 (exact zeros only), matching
//! `fvecs_remove_zeros.py`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::XvecReader;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};

unsafe extern "C" {
    fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
}

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

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

/// Pipeline command: remove zero/near-zero vectors from fvec file.
pub struct TransformRemoveZerosKnnUtilsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(TransformRemoveZerosKnnUtilsOp)
}

impl CommandOp for TransformRemoveZerosKnnUtilsOp {
    fn command_path(&self) -> &str {
        "transform remove-zeros-knnutils"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_TRANSFORM
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Remove zero/near-zero vectors from fvec file (knn_utils compatible)".into(),
            body: format!(r#"# transform remove-zeros-knnutils

Remove vectors whose L2 norm is at or below a tolerance, replicating
`fvecs_remove_zeros.py` from knn\_utils.

Norm is computed via BLAS `cblas_snrm2` (matching knn\_utils which uses
`np.linalg.norm`). Default tolerance is 0.0 (exact zeros only).

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let tolerance: f32 = match options.parse_or("tolerance", 0.0_f32) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        let reader = match XvecReader::<f32>::open_path(&source_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
        };
        let count = <XvecReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <XvecReader<f32> as VectorReader<f32>>::dim(&reader);

        if count == 0 {
            return error_result("empty source file", start);
        }

        ctx.ui.log(&format!(
            "  remove-zeros-knnutils: {} vectors, dim={}, tolerance={:.0e}",
            count, dim, tolerance,
        ));

        let pb = ctx.ui.bar_with_unit(count as u64, "filtering zeros", "vectors");
        let mut writer = match AtomicWriter::new(&output_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
        };

        let dim_bytes = (dim as i32).to_le_bytes();
        let mut kept = 0usize;
        let mut removed = 0usize;

        for i in 0..count {
            let slice = reader.get_slice(i);
            let norm = blas_snrm2(slice);

            if norm > tolerance {
                writer.write_all(&dim_bytes).map_err(|e| e.to_string()).unwrap();
                let src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        slice.len() * 4,
                    )
                };
                writer.write_all(src_bytes).map_err(|e| e.to_string()).unwrap();
                kept += 1;
            } else {
                removed += 1;
            }

            if (i + 1) % 100_000 == 0 {
                pb.set_position((i + 1) as u64);
            }
        }
        pb.finish();

        if let Err(e) = writer.finish() {
            return error_result(format!("finalize: {}", e), start);
        }

        ctx.ui.log(&format!("  kept: {}, removed: {} (norm <= {:.0e})", kept, removed, tolerance));

        // Write variables
        let set_var = |ctx: &mut StreamContext, name: &str, value: &str| {
            let _ = crate::pipeline::variables::set_and_save(&ctx.workspace, name, value);
            ctx.defaults.insert(name.to_string(), value.to_string());
        };
        set_var(ctx, "zero_count", &removed.to_string());
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        set_var(ctx, &var_name, &kept.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} -> {} kept, {} removed (norm <= {:.0e})",
                count, kept, removed, tolerance,
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
                description: "Source fvec file".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output fvec file (zeros removed)".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "tolerance".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.0".to_string()),
                description: "Remove vectors with norm <= tolerance (default: 0.0, exact zeros)".to_string(),
                role: OptionRole::Config,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output"],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::{Options, Status, StreamContext};
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use vectordata::VectorReader;
    use vectordata::io::XvecReader;

    fn make_ctx(workspace: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    fn write_fvec(path: &std::path::Path, vectors: &[Vec<f32>]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    /// Remove exact zeros (default tolerance=0.0).
    #[test]
    fn test_remove_exact_zeros() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.0],  // zero
            vec![4.0, 5.0, 6.0],
            vec![0.0, 0.0, 0.0],  // zero
            vec![7.0, 8.0, 9.0],
        ];
        let src = tmp.path().join("source.fvec");
        write_fvec(&src, &vectors);

        let out = tmp.path().join("clean.fvec");
        let mut opts = Options::new();
        opts.set("source", src.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());

        let mut op = TransformRemoveZerosKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let reader = XvecReader::<f32>::open_path(&out).unwrap();
        assert_eq!(<XvecReader<f32> as VectorReader<f32>>::count(&reader), 3);
        assert_eq!(reader.get_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(reader.get_slice(1), &[4.0, 5.0, 6.0]);
        assert_eq!(reader.get_slice(2), &[7.0, 8.0, 9.0]);
    }

    /// All-zero file → error (all removed, empty output).
    #[test]
    fn test_remove_zeros_all_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let src = tmp.path().join("source.fvec");
        write_fvec(&src, &vectors);

        let out = tmp.path().join("clean.fvec");
        let mut opts = Options::new();
        opts.set("source", src.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());

        let mut op = TransformRemoveZerosKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let reader = XvecReader::<f32>::open_path(&out).unwrap();
        assert_eq!(<XvecReader<f32> as VectorReader<f32>>::count(&reader), 0);
    }

    /// No zeros → output identical to input.
    #[test]
    fn test_remove_zeros_none_present() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let src = tmp.path().join("source.fvec");
        write_fvec(&src, &vectors);

        let out = tmp.path().join("clean.fvec");
        let mut opts = Options::new();
        opts.set("source", src.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());

        let mut op = TransformRemoveZerosKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let reader = XvecReader::<f32>::open_path(&out).unwrap();
        assert_eq!(<XvecReader<f32> as VectorReader<f32>>::count(&reader), 2);
    }

    /// Custom tolerance removes near-zero vectors.
    #[test]
    fn test_remove_zeros_with_tolerance() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let vectors = vec![
            vec![1.0, 2.0],
            vec![0.0001, 0.0001],  // norm ≈ 0.00014, below 0.001
            vec![3.0, 4.0],
        ];
        let src = tmp.path().join("source.fvec");
        write_fvec(&src, &vectors);

        let out = tmp.path().join("clean.fvec");
        let mut opts = Options::new();
        opts.set("source", src.to_string_lossy().to_string());
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("tolerance", "0.001");

        let mut op = TransformRemoveZerosKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let reader = XvecReader::<f32>::open_path(&out).unwrap();
        assert_eq!(<XvecReader<f32> as VectorReader<f32>>::count(&reader), 2);
    }
}
