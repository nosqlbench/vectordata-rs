// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: detect near-zero vectors by L2-norm threshold.
//!
//! Scans a vector file and identifies vectors whose L2 norm (computed in
//! f64 precision) falls below a configurable threshold (default 1×10⁻⁶).
//! Near-zero vectors cause undefined behavior in cosine similarity and
//! degenerate results after normalization. See SRD §19.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use vectordata::io::MmapVectorReader;
use vectordata::io::VectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use crate::pipeline::atomic_write::safe_create_file;
use crate::pipeline::element_type::ElementType;

/// Default L2-norm threshold for near-zero detection.
const DEFAULT_THRESHOLD: f64 = 1e-6;

/// Pipeline command: detect near-zero vectors.
pub struct AnalyzeZerosOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeZerosOp)
}

impl CommandOp for AnalyzeZerosOp {
    fn command_path(&self) -> &str {
        "analyze zeros"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Find and report near-zero vectors in a vector file".into(),
            body: format!(
                "# analyze zeros\n\n\
                Find and report near-zero vectors in a vector file.\n\n\
                ## Description\n\n\
                Scans a vector file and identifies vectors whose L2 norm \
                (computed entirely in f64 precision) falls below a configurable \
                threshold (default 1×10⁻⁶). Reports the count and percentage \
                of near-zero vectors found.\n\n\
                ## Why Near-Zero Vectors Matter\n\n\
                Near-zero vectors are problematic in several common scenarios:\n\n\
                - **Cosine similarity**: Division by near-zero norm produces \
                numerically unstable results.\n\
                - **Normalization**: Normalizing a near-zero vector amplifies \
                quantization noise to unit scale, producing a meaningless \
                direction vector.\n\
                - **KNN ground truth**: Near-zero vectors distort distance \
                rankings and produce degenerate neighborhoods.\n\n\
                ## Detection Method\n\n\
                Each vector's L2 norm is computed in f64 precision:\n\n\
                    ‖x‖₂ = √(Σ xᵢ²)   [all arithmetic in f64]\n\n\
                A vector is classified as near-zero when ‖x‖₂ < threshold. \
                The comparison is performed in squared space (Σ xᵢ² < τ²) \
                to avoid the square root. See SRD §19 for details.\n\n\
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

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = options.get("output").map(|s| resolve_path(s, &ctx.workspace));

        let threshold: f64 = options.get("threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_THRESHOLD);

        let etype = match ElementType::from_path(&source_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        // Open reader and extract (count, get_f64_fn) based on element type.
        // All components are upcast to f64 for norm computation (SRD §19.2).
        let (count, get_f64): (usize, Box<dyn Fn(usize) -> Vec<f64> + Sync>) = match etype {
            ElementType::F32 => {
                let r = match MmapVectorReader::<f32>::open_fvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f32>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::F16 => {
                let r = match MmapVectorReader::<half::f16>::open_mvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<half::f16>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|v| v.to_f64()).collect()))
            }
            ElementType::F64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default()))
            }
            ElementType::I32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::I16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U8 | ElementType::I8 => {
                let r = match MmapVectorReader::<u8>::open_bvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<u8>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U64 | ElementType::I64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                (fc, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
        };

        let threads = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        // Squared threshold avoids sqrt per vector (SRD §19.2)
        let threshold_sq = threshold * threshold;

        let pb = ctx.ui.bar_with_unit(count as u64, "scanning for near-zero vectors", "vectors");
        let progress = AtomicU64::new(0);
        let pb_ref = &pb;
        let progress_ref = &progress;

        let need_ordinals = output_path.is_some();

        let scan_fn = || {
            if need_ordinals {
                let ordinals: Vec<usize> = (0..count)
                    .into_par_iter()
                    .filter_map(|i| {
                        let components = get_f64(i);
                        let norm_sq: f64 = components.iter().map(|&v| v * v).sum();
                        let is_near_zero = norm_sq < threshold_sq;
                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 100_000 == 0 {
                            pb_ref.set_position(done);
                        }
                        if is_near_zero { Some(i) } else { None }
                    })
                    .collect();
                (ordinals.len(), ordinals)
            } else {
                let zero_count: usize = (0..count)
                    .into_par_iter()
                    .map(|i| {
                        let components = get_f64(i);
                        let norm_sq: f64 = components.iter().map(|&v| v * v).sum();
                        let is_near_zero = norm_sq < threshold_sq;
                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 100_000 == 0 {
                            pb_ref.set_position(done);
                        }
                        if is_near_zero { 1usize } else { 0usize }
                    })
                    .sum();
                (zero_count, vec![])
            }
        };

        let (zero_count, zero_ordinals) = if let Some(p) = pool {
            p.install(scan_fn)
        } else {
            scan_fn()
        };

        pb.finish();

        let produced = if let Some(out) = output_path {
            match write_ordinals_ivec(&out, &zero_ordinals) {
                Ok(()) => vec![out],
                Err(e) => return error_result(e, start),
            }
        } else {
            vec![]
        };

        finish_result(&source_path, zero_count, count, threshold, produced, start)
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        let input_keys: Vec<&str> = vec!["source"];
        let output_keys: Vec<&str> = if options.get("output").is_some() {
            vec!["output"]
        } else {
            vec![]
        };
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &input_keys,
            &output_keys,
        )
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Input vector file (fvec, mvec, dvec, ivec, etc.)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output ivec of near-zero vector ordinals for downstream clean steps"
                    .to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "threshold".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("1e-06".to_string()),
                description: "L2-norm threshold below which a vector is classified as near-zero (SRD §19)"
                    .to_string(),
                role: OptionRole::Config,
            },
        ]
    }
}

/// Write a list of ordinals as a dim=1 ivec file.
fn write_ordinals_ivec(path: &Path, ordinals: &[usize]) -> Result<(), String> {
    ensure_parent(path);
    let mut f = safe_create_file(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    for &ord in ordinals {
        f.write_all(&1i32.to_le_bytes())
            .map_err(|e| format!("write error: {}", e))?;
        f.write_all(&(ord as i32).to_le_bytes())
            .map_err(|e| format!("write error: {}", e))?;
    }
    Ok(())
}

/// Build the final `CommandResult` from zero-count stats.
fn finish_result(
    path: &Path,
    zero_count: usize,
    total: usize,
    threshold: f64,
    produced: Vec<PathBuf>,
    start: Instant,
) -> CommandResult {
    let pct = if total > 0 {
        (zero_count as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    log::info!(
        "{}: {} near-zero vectors (L2 < {:.0e}) out of {} total ({:.2}%)",
        path.display(),
        zero_count,
        threshold,
        total,
        pct
    );

    // Write verified count for the bound checker
    for p in &produced {
        if let Some(fname) = p.file_name().and_then(|n| n.to_str()) {
            let var_name = format!("verified_count:{}", fname);
            let ws = p.parent()
                .and_then(|d| d.parent())
                .unwrap_or(Path::new("."));
            let _ = crate::pipeline::variables::set_and_save(ws, &var_name, &zero_count.to_string());
        }
    }

    let status = if zero_count > 0 {
        Status::Warning
    } else {
        Status::Ok
    };

    CommandResult {
        status,
        message: format!(
            "{} near-zero vectors (L2 < {:.0e}) out of {} ({:.2}%)",
            zero_count, threshold, total, pct
        ),
        produced,
        elapsed: start.elapsed(),
    }
}

/// Ensure the parent directory of a path exists.
fn ensure_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
}

fn resolve_path(s: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(s);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
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
    use indexmap::IndexMap;
    use crate::pipeline::progress::ProgressLog;

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

    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for vec in vectors {
            f.write_all(&(vec.len() as i32).to_le_bytes()).unwrap();
            for &v in vec {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn test_exact_zero_detected() {
        let dir = tempfile::tempdir().unwrap();
        let fvec = dir.path().join("test.fvec");
        write_fvec(&fvec, &[
            vec![1.0, 2.0],
            vec![0.0, 0.0],  // exact zero — L2 norm = 0
            vec![3.0, 4.0],
        ]);
        let output = dir.path().join("zeros.ivec");

        let mut cmd = AnalyzeZerosOp;
        let mut opts = Options::new();
        opts.set("source", fvec.to_str().unwrap());
        opts.set("output", output.to_str().unwrap());

        let mut ctx = test_ctx(dir.path());
        let result = cmd.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("1 near-zero"));
    }

    #[test]
    fn test_near_zero_detected() {
        let dir = tempfile::tempdir().unwrap();
        let fvec = dir.path().join("test.fvec");
        // Vector with L2 norm ≈ 1.41e-08 < 1e-06 threshold
        write_fvec(&fvec, &[
            vec![1.0, 2.0],
            vec![1e-8, 1e-8],  // near-zero
            vec![3.0, 4.0],
        ]);
        let output = dir.path().join("zeros.ivec");

        let mut cmd = AnalyzeZerosOp;
        let mut opts = Options::new();
        opts.set("source", fvec.to_str().unwrap());
        opts.set("output", output.to_str().unwrap());

        let mut ctx = test_ctx(dir.path());
        let result = cmd.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("1 near-zero"));
    }

    #[test]
    fn test_above_threshold_not_detected() {
        let dir = tempfile::tempdir().unwrap();
        let fvec = dir.path().join("test.fvec");
        // Vector with L2 norm = 1e-05 > 1e-06 threshold
        write_fvec(&fvec, &[
            vec![1.0, 2.0],
            vec![1e-5, 0.0],  // above threshold
            vec![3.0, 4.0],
        ]);

        let mut cmd = AnalyzeZerosOp;
        let mut opts = Options::new();
        opts.set("source", fvec.to_str().unwrap());

        let mut ctx = test_ctx(dir.path());
        let result = cmd.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("0 near-zero"));
    }

    #[test]
    fn test_no_zeros() {
        let dir = tempfile::tempdir().unwrap();
        let fvec = dir.path().join("test.fvec");
        write_fvec(&fvec, &[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        let mut cmd = AnalyzeZerosOp;
        let mut opts = Options::new();
        opts.set("source", fvec.to_str().unwrap());

        let mut ctx = test_ctx(dir.path());
        let result = cmd.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Ok);
    }
}
