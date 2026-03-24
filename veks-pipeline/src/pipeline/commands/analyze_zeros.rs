// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: count zero vectors.
//!
//! Scans an fvec or ivec file and counts vectors where all components are zero.
//! Reports the count and percentage.
//!
//! Equivalent to the Java `CMD_count_zeros` / `CMD_analyze_zeros` command.

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
use crate::pipeline::element_type::ElementType;

/// Pipeline command: count all-zero vectors.
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
            summary: "Find and report zero vectors in a vector file".into(),
            body: format!(
                "# analyze zeros\n\n\
                Find and report zero vectors in a vector file.\n\n\
                ## Description\n\n\
                Scans an fvec or ivec file and counts vectors where all components are \
                exactly zero. Reports the count and percentage of zero vectors found. \
                The command performs a sequential scan of every vector in the file, \
                checking each component against zero.\n\n\
                ## Why Zero Vectors Matter\n\n\
                All-zero vectors (and near-zero vectors) are problematic in several \
                common scenarios:\n\n\
                - **Cosine similarity**: Division by zero occurs when computing the \
                L2 norm of a zero vector, producing NaN or infinity results.\n\
                - **Normalization**: Attempting to L2-normalize a zero vector fails.\n\
                - **KNN ground truth**: Zero vectors can distort distance rankings \
                and produce degenerate neighborhoods where many vectors are equidistant.\n\
                - **Benchmark validity**: Datasets with a significant fraction of zero \
                vectors may not represent realistic workloads.\n\n\
                ## Role in Dataset Preparation\n\n\
                This command serves as a data quality check that should be run early \
                in a pipeline, immediately after import or download. If zero vectors \
                are detected, the result status is set to Warning rather than Ok, \
                alerting downstream pipeline steps. Common remediation strategies \
                include filtering out zero vectors or replacing them with small random \
                perturbations.\n\n\
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
        let index_path = options.get("index").map(|s| resolve_path(s, &ctx.workspace));

        let etype = match ElementType::from_path(&source_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        // Open reader and extract (count, get_f64_fn) based on element type.
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
        };

        // When an index is provided, use the fast linear-search path.
        if let Some(ref idx_path) = index_path {
            return count_zeros_indexed(
                &source_path, idx_path, &get_f64, count,
                output_path.as_deref(), start, ctx,
            );
        }

        let threads = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        count_zeros_generic(
            &source_path, &get_f64, count,
            output_path.as_deref(), start, ctx, pool.as_ref(),
        )
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        let mut input_keys: Vec<&str> = vec!["source"];
        if options.get("index").is_some() {
            input_keys.push("index");
        }
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
                description: "Input vector file (fvec or ivec)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "index".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Sorted dedup ordinal index (ivec, dim=1). When provided, \
                    uses binary search at position 0 instead of a full scan."
                    .to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output ivec of zero-vector ordinals for downstream clean steps"
                    .to_string(),
                role: OptionRole::Output,
            },
        ]
    }
}

/// Generic zero-vector scan using a type-erased f64 accessor.
fn count_zeros_generic(
    path: &Path,
    get_f64: &(dyn Fn(usize) -> Vec<f64> + Sync),
    count: usize,
    output: Option<&Path>,
    start: Instant,
    ctx: &mut StreamContext,
    pool: Option<&rayon::ThreadPool>,
) -> CommandResult {
    let pb = ctx.ui.bar_with_unit(count as u64, "scanning for zeros", "vectors");

    let progress = AtomicU64::new(0);
    let pb_ref = &pb;
    let progress_ref = &progress;

    let need_ordinals = output.is_some();

    let scan_fn = || {
        if need_ordinals {
            let ordinals: Vec<usize> = (0..count)
                .into_par_iter()
                .filter_map(|i| {
                    let is_zero = get_f64(i).iter().all(|&v| v == 0.0);
                    let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 100_000 == 0 {
                        pb_ref.set_position(done);
                    }
                    if is_zero { Some(i) } else { None }
                })
                .collect();
            (ordinals.len(), ordinals)
        } else {
            let zero_count: usize = (0..count)
                .into_par_iter()
                .map(|i| {
                    let is_zero = get_f64(i).iter().all(|&v| v == 0.0);
                    let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 100_000 == 0 {
                        pb_ref.set_position(done);
                    }
                    if is_zero { 1usize } else { 0usize }
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

    let produced = if let Some(out) = output {
        match write_ordinals_ivec(out, &zero_ordinals) {
            Ok(()) => vec![out.to_path_buf()],
            Err(e) => return error_result(e, start),
        }
    } else {
        vec![]
    };

    finish_result(path, zero_count, count, produced, start)
}

/// Count zero vectors using a sorted dedup ordinal index.
///
/// The index is a dim=1 ivec file whose ordinals are sorted by lexicographic
/// component order (dimension 0 first, then 1, etc.). A zero vector sorts to
/// position 0, so we read forward from the start until we find a non-zero
/// vector.
fn count_zeros_indexed(
    source_path: &Path,
    index_path: &Path,
    get_f64: &(dyn Fn(usize) -> Vec<f64> + Sync),
    source_count: usize,
    output: Option<&Path>,
    start: Instant,
    ctx: &mut StreamContext,
) -> CommandResult {
    let index = match MmapVectorReader::<i32>::open_ivec(index_path) {
        Ok(r) => r,
        Err(e) => return error_result(
            format!("failed to open index {}: {}", index_path.display(), e),
            start,
        ),
    };

    let index_count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&index);
    let pb = ctx.ui.bar_with_unit(index_count as u64, "index search for zeros", "entries");

    let mut zero_ordinals: Vec<usize> = Vec::new();
    for pos in 0..index_count {
        let ordinal = match index.get(pos) {
            Ok(v) => v[0] as usize,
            Err(e) => return error_result(
                format!("failed to read index at {}: {}", pos, e),
                start,
            ),
        };
        let is_zero = get_f64(ordinal).iter().all(|&v| v == 0.0);
        pb.set_position((pos + 1) as u64);
        if is_zero {
            zero_ordinals.push(ordinal);
        } else {
            // Sorted lexicographically: once we see a non-zero, all remaining
            // entries are also non-zero.
            break;
        }
    }
    pb.finish();

    let zero_count = zero_ordinals.len();
    let produced = if let Some(out) = output {
        match write_ordinals_ivec(out, &zero_ordinals) {
            Ok(()) => vec![out.to_path_buf()],
            Err(e) => return error_result(e, start),
        }
    } else {
        vec![]
    };

    finish_result(source_path, zero_count, source_count, produced, start)
}

/// Write a list of ordinals as a dim=1 ivec file.
fn write_ordinals_ivec(path: &Path, ordinals: &[usize]) -> Result<(), String> {
    ensure_parent(path);
    let mut f = std::fs::File::create(path)
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
    produced: Vec<PathBuf>,
    start: Instant,
) -> CommandResult {
    let pct = if total > 0 {
        (zero_count as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    log::info!(
        "{}: {} zero vectors out of {} total ({:.2}%)",
        path.display(),
        zero_count,
        total,
        pct
    );

    let status = if zero_count > 0 {
        Status::Warning
    } else {
        Status::Ok
    };

    CommandResult {
        status,
        message: format!(
            "{} zero vectors out of {} ({:.2}%)",
            zero_count, total, pct
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
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

    fn write_ivec(path: &Path, vectors: &[Vec<i32>]) {
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
    fn test_zeros_fvec_no_zeros() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        write_fvec(&input, &[vec![1.0, 2.0], vec![3.0, 4.0]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeZerosOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("0 zero vectors"));
    }

    #[test]
    fn test_zeros_fvec_with_zeros() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        write_fvec(
            &input,
            &[vec![1.0, 2.0], vec![0.0, 0.0], vec![3.0, 0.0], vec![0.0, 0.0]],
        );

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeZerosOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("2 zero vectors"));
    }

    #[test]
    fn test_zeros_ivec() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.ivec");
        write_ivec(&input, &[vec![0, 0, 0], vec![1, 2, 3]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeZerosOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("1 zero vectors"));
    }
}
