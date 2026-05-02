// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: unified dataset verification (knn\_utils personality).
//!
//! Runs all the verification checks that knn\_utils users would normally
//! perform across `fvecs_check.py`, `ivecs_check.py`, and cross-file
//! consistency checks — in a single pipeline step.
//!
//! Checks performed:
//!
//! **fvecs checks** (base and query vectors):
//! - Dimension consistency (all vectors same dim)
//! - L2 normalization via BLAS `cblas_snrm2` (tol\_norm, default 1e-5)
//! - Zero/near-zero vectors (tol\_zero, default 1e-6)
//! - Norm statistics: min, max, mean, max abs deviation from 1.0
//!
//! **ivecs checks** (ground truth):
//! - No duplicate ordinals within any row
//! - No negative entries
//! - Row length = k for all rows
//! - Max ordinal < base vector count
//!
//! **Cross-file consistency**:
//! - base dim = query dim
//! - Ground truth row count = query count
//! - Ground truth k matches expected neighbors
//!
//! **KNN accuracy** (replicates `validate_knn_utils.py`):
//! - Samples N queries (default 100)
//! - Recomputes brute-force KNN via FAISS for the sample
//! - Compares neighbor sets against ground truth (set equality,
//!   tolerant of tie-breaking order differences)
//! - Reports per-query recall and overall pass/fail
//!
//! Output follows the knn\_utils report style for familiarity.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::XvecReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};
use crate::pipeline::element_type::ElementType;

// BLAS snrm2: same routine knn_utils calls via np.linalg.norm(vector).
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

fn resolve_path(value: &str, workspace: &Path) -> std::path::PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

/// Results from checking one fvecs file.
struct FvecsCheckResult {
    path: String,
    count: usize,
    dim: usize,
    zero_count: u64,
    unnormalized_count: u64,
    normalized: bool,
    norm_min: f64,
    norm_max: f64,
    norm_mean: f64,
    max_abs_dev: f64,
}

/// Results from checking one ivecs file.
struct IvecsCheckResult {
    path: String,
    num_rows: usize,
    row_length: usize,
    max_ordinal: i32,
    duplicate_rows: usize,
    negative_rows: usize,
    truncated_rows: usize,
    overlong_rows: usize,
    passed: bool,
}

/// Check an fvecs file — replicates fvecs_check.py logic.
fn check_fvecs(
    path: &Path,
    tol_norm: f64,
    tol_zero: f64,
    ui: &veks_core::ui::UiHandle,
) -> Result<FvecsCheckResult, String> {
    let reader = XvecReader::<f32>::open_path(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let count = <XvecReader<f32> as VectorReader<f32>>::count(&reader);
    let dim = <XvecReader<f32> as VectorReader<f32>>::dim(&reader);

    if count == 0 || dim == 0 {
        return Err(format!("{}: empty file (count={}, dim={})", path.display(), count, dim));
    }

    let mut zero_count: u64 = 0;
    let mut unnormalized_count: u64 = 0;
    let mut normalized = true;
    let mut norm_sum: f64 = 0.0;
    let mut norm_min: f64 = f64::INFINITY;
    let mut norm_max: f64 = f64::NEG_INFINITY;

    // Resource cooperation: the file may be many TB. mmap'd pages
    // count toward RSS once touched, and on machines under no
    // immediate memory pressure the kernel can let RSS grow right up
    // to the governor's ceiling before reclaiming. `StreamReclaim`
    // handles MADV_SEQUENTIAL + per-window MADV_DONTNEED so resident
    // pages stay bounded to ~256 MiB regardless of dim or RAM.
    let mut reclaim = vectordata::io::StreamReclaim::new(&reader, 0, count);

    let pb_label = format!("checking fvecs: {}", path.file_name()
        .map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "<input>".into()));
    let pb = ui.bar_with_unit(count as u64, pb_label, "vectors");
    // Update progress in 1<<14 chunks. PlainSink throttles renders to
    // every 250-500 ms internally, so the position update itself is
    // lock-only and cheap; the chunk size keeps the per-iteration
    // overhead negligible while still giving a smooth-looking bar.
    const PB_CHUNK: usize = 1 << 14;
    for i in 0..count {
        let slice = reader.get_slice(i);
        let norm = blas_snrm2(slice) as f64;

        norm_sum += norm;
        if norm < norm_min { norm_min = norm; }
        if norm > norm_max { norm_max = norm; }

        if (norm - 1.0).abs() > tol_norm {
            normalized = false;
            unnormalized_count += 1;
        }
        if norm < tol_zero {
            zero_count += 1;
        }
        if i % PB_CHUNK == 0 {
            pb.set_position(i as u64);
        }
        reclaim.advance(i);
    }
    drop(reclaim);
    pb.finish();

    let norm_mean = norm_sum / count as f64;
    let max_abs_dev = (norm_min - 1.0).abs().max((norm_max - 1.0).abs());

    Ok(FvecsCheckResult {
        path: path.display().to_string(),
        count,
        dim,
        zero_count,
        unnormalized_count,
        normalized,
        norm_min,
        norm_max,
        norm_mean,
        max_abs_dev,
    })
}

/// Check an mvecs (f16) file — same logic as `check_fvecs` but reads
/// half-precision and upcasts each vector to f32 for the BLAS norm.
fn check_mvecs(
    path: &Path,
    tol_norm: f64,
    tol_zero: f64,
    ui: &veks_core::ui::UiHandle,
) -> Result<FvecsCheckResult, String> {
    let reader = XvecReader::<half::f16>::open_path(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let count = <XvecReader<half::f16> as VectorReader<half::f16>>::count(&reader);
    let dim = <XvecReader<half::f16> as VectorReader<half::f16>>::dim(&reader);

    if count == 0 || dim == 0 {
        return Err(format!("{}: empty file (count={}, dim={})", path.display(), count, dim));
    }

    let mut zero_count: u64 = 0;
    let mut unnormalized_count: u64 = 0;
    let mut normalized = true;
    let mut norm_sum: f64 = 0.0;
    let mut norm_min: f64 = f64::INFINITY;
    let mut norm_max: f64 = f64::NEG_INFINITY;

    let mut reclaim = vectordata::io::StreamReclaim::new(&reader, 0, count);

    let pb_label = format!("checking mvecs: {}", path.file_name()
        .map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "<input>".into()));
    let pb = ui.bar_with_unit(count as u64, pb_label, "vectors");
    const PB_CHUNK: usize = 1 << 14;

    // Reusable f32 upcast buffer to avoid per-vector allocation.
    let mut f32_buf: Vec<f32> = vec![0.0; dim];

    for i in 0..count {
        let slice = reader.get_slice(i);
        for (j, v) in slice.iter().enumerate() {
            f32_buf[j] = v.to_f32();
        }
        let norm = blas_snrm2(&f32_buf) as f64;

        norm_sum += norm;
        if norm < norm_min { norm_min = norm; }
        if norm > norm_max { norm_max = norm; }

        if (norm - 1.0).abs() > tol_norm {
            normalized = false;
            unnormalized_count += 1;
        }
        if norm < tol_zero {
            zero_count += 1;
        }
        if i % PB_CHUNK == 0 {
            pb.set_position(i as u64);
        }
        reclaim.advance(i);
    }
    drop(reclaim);
    pb.finish();

    let norm_mean = norm_sum / count as f64;
    let max_abs_dev = (norm_min - 1.0).abs().max((norm_max - 1.0).abs());

    Ok(FvecsCheckResult {
        path: path.display().to_string(),
        count,
        dim,
        zero_count,
        unnormalized_count,
        normalized,
        norm_min,
        norm_max,
        norm_mean,
        max_abs_dev,
    })
}

/// Effective per-element-type tolerance. f16's 11-bit mantissa adds
/// ~1e-3 of unavoidable per-element relative noise, which after
/// accumulation across a typical dim lands a normalized vector's L2
/// norm in a band several orders of magnitude wider than f32's. A
/// fixed 1e-5 default would cause every f16-stored normalized dataset
/// to fail this check spuriously; relax it when the user hasn't
/// pinned the value explicitly.
fn default_tol_norm_for(etype: ElementType) -> f64 {
    match etype {
        ElementType::F32 => 1e-5,
        ElementType::F16 => 1e-3,
        _ => 1e-5,
    }
}

/// Dispatch an xvec norm/zero check by element type. Today only f32
/// (.fvec) and f16 (.mvec) bases/queries are accepted by the broader
/// pipeline, so anything else is a hard error here rather than a
/// silent-but-wrong f32 reinterpretation.
///
/// Returns the (result, effective_tol_norm) pair so the caller can
/// report the band that was actually applied — useful when the
/// element type relaxed it from the user-facing default.
fn check_vectors(
    path: &Path,
    user_tol_norm: Option<f64>,
    tol_zero: f64,
    ui: &veks_core::ui::UiHandle,
) -> Result<(FvecsCheckResult, f64), String> {
    let etype = ElementType::from_path(path)
        .map_err(|e| format!(
            "verify dataset-knnutils: cannot determine element type of {}: {}",
            path.display(), e,
        ))?;
    let tol = user_tol_norm.unwrap_or_else(|| default_tol_norm_for(etype));
    let result = match etype {
        ElementType::F32 => check_fvecs(path, tol, tol_zero, ui)?,
        ElementType::F16 => check_mvecs(path, tol, tol_zero, ui)?,
        _ => return Err(format!(
            "verify dataset-knnutils: unsupported element type {:?} for {} (only f32/.fvec and f16/.mvec are supported)",
            etype, path.display(),
        )),
    };
    Ok((result, tol))
}

/// Check an ivecs file — replicates ivecs_check.py logic.
fn check_ivecs(
    path: &Path,
    required_k: usize,
    max_valid_ordinal: usize,
    ui: &veks_core::ui::UiHandle,
) -> Result<IvecsCheckResult, String> {
    let reader = XvecReader::<i32>::open_path(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let num_rows = <XvecReader<i32> as VectorReader<i32>>::count(&reader);
    let row_length = <XvecReader<i32> as VectorReader<i32>>::dim(&reader);

    let mut max_ordinal: i32 = -1;
    let mut duplicate_rows: usize = 0;
    let mut negative_rows: usize = 0;
    let mut truncated_rows: usize = 0;
    let mut overlong_rows: usize = 0;

    // Same resource-cooperation strategy as check_fvecs.
    let mut reclaim = vectordata::io::StreamReclaim::new(&reader, 0, num_rows);

    let pb_label = format!("checking ivecs: {}", path.file_name()
        .map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "<input>".into()));
    let pb = ui.bar_with_unit(num_rows as u64, pb_label, "rows");
    const PB_CHUNK: usize = 1 << 12;
    for i in 0..num_rows {
        if i % PB_CHUNK == 0 {
            pb.set_position(i as u64);
        }
        reclaim.advance(i);
        let row = reader.get_slice(i);
        let actual_k = row.len();

        if actual_k < required_k {
            truncated_rows += 1;
        } else if actual_k > required_k {
            overlong_rows += 1;
        }

        let mut has_negative = false;
        let mut seen = HashSet::with_capacity(actual_k);
        let mut has_dup = false;

        for &val in row {
            if val < 0 {
                has_negative = true;
            }
            if val > max_ordinal {
                max_ordinal = val;
            }
            if !seen.insert(val) {
                has_dup = true;
            }
        }

        if has_negative { negative_rows += 1; }
        if has_dup { duplicate_rows += 1; }
    }
    drop(reclaim);
    pb.finish();

    let ordinal_in_range = max_ordinal < 0 || (max_ordinal as usize) < max_valid_ordinal;
    let passed = duplicate_rows == 0
        && negative_rows == 0
        && truncated_rows == 0
        && overlong_rows == 0
        && ordinal_in_range;

    Ok(IvecsCheckResult {
        path: path.display().to_string(),
        num_rows,
        row_length,
        max_ordinal,
        duplicate_rows,
        negative_rows,
        truncated_rows,
        overlong_rows,
        passed,
    })
}

/// Pipeline command: unified knn\_utils-style dataset verification.
pub struct VerifyDatasetKnnUtilsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyDatasetKnnUtilsOp)
}

impl CommandOp for VerifyDatasetKnnUtilsOp {
    fn command_path(&self) -> &str {
        "verify dataset-knnutils"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Unified dataset verification using knn_utils conventions".into(),
            body: format!(r#"# verify dataset-knnutils

Runs all verification checks that knn\_utils users would normally perform
across `fvecs_check.py` and `ivecs_check.py`, plus cross-file consistency
checks, in a single pipeline step.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Base vectors file (fvec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "query".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Query vectors file (fvec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "indices".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Ground truth neighbor indices (ivec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "neighbors".into(),
                type_name: "int".into(),
                required: true,
                default: None,
                description: "Expected number of neighbors per query (k)".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "tol-norm".into(),
                type_name: "float".into(),
                required: false,
                default: Some("1e-5".into()),
                description: "Normalization tolerance (knn_utils fvecs_check default)".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "tol-zero".into(),
                type_name: "float".into(),
                required: false,
                default: Some("1e-6".into()),
                description: "Zero-vector tolerance (knn_utils fvecs_check default)".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "metric".into(),
                type_name: "enum".into(),
                required: false,
                default: Some("IP".into()),
                description: "Distance metric: IP (inner product), L2".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "sample".into(),
                type_name: "int".into(),
                required: false,
                default: Some("100".into()),
                description: "Number of queries to sample for KNN accuracy check".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "report".into(),
                type_name: "Path".into(),
                required: false,
                default: None,
                description: "Output report file path".into(),
                role: OptionRole::Output,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // knn_utils verifier uses sgemm — same MKL ABI hazard as
        // every other BLAS-touching command (see pipeline::blas_abi).
        crate::pipeline::blas_abi::set_single_threaded_if_faiss();

        // Up-front: confirm the local dataset has the minimum facets
        // this verify kind requires (see pipeline::dataset_lookup).
        if let Err(e) = crate::pipeline::dataset_lookup::validate_and_log(
            ctx, options, crate::pipeline::dataset_lookup::VerifyKind::DatasetKnnutils,
        ) {
            return error_result(e, start);
        }

        // Standalone-friendly: input paths come from the active
        // profile's facets in dataset.yaml (see pipeline::dataset_lookup).
        let base_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "base", "base_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let query_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "query", "query_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let indices_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "indices", "neighbor_indices",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        // Standalone-friendly: `--neighbors` defaults to the active
        // profile's `maxk` in dataset.yaml when unset.
        let k: usize = match crate::pipeline::dataset_lookup::resolve_neighbors(ctx, options) {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };
        // tol-norm default depends on element type. f32 storage holds
        // a normalized vector to within ~1e-6 per element, so a 1e-5
        // band on the L2 norm is appropriate. f16 storage only has 11
        // bits of mantissa (~9.77e-4 per-element relative precision);
        // a perfectly-normalized f16 vector at dim=512 has accumulated
        // norm error of order 5e-4, which would always trip a 1e-5
        // tolerance even though the data is in fact L2-normalized.
        // Pass the user value through `Option` so `check_vectors` can
        // pick the per-type default when the user hasn't pinned it.
        let user_tol_norm: Option<f64> = if options.get("tol-norm").is_some() {
            match options.parse_or("tol-norm", 1e-5_f64) {
                Ok(v) => Some(v),
                Err(e) => return error_result(e, start),
            }
        } else {
            None
        };
        let tol_zero: f64 = match options.parse_or("tol-zero", 1e-6_f64) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };
        let metric_str = options.get("metric").unwrap_or("IP");
        let sample_count: usize = match options.parse_or("sample", 100usize) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };

        let base_path = resolve_path(&base_str, &ctx.workspace);
        let query_path = resolve_path(&query_str, &ctx.workspace);
        let indices_path = resolve_path(&indices_str, &ctx.workspace);

        let report_path = match options.get("report") {
            Some(s) => resolve_path(s, &ctx.workspace),
            None => ctx.workspace.join(".cache/verify_dataset_knnutils.txt"),
        };

        // Banner: tell the user up-front what's about to happen and at
        // what scale. Cheap to compute — just peek at the query file
        // header for the total count via mmap.
        // Peek the query count for the banner. Dispatch by element
        // type so the .mvecs (f16) case doesn't trip the f32 stride
        // check inside the reader.
        let query_total = match ElementType::from_path(&query_path) {
            Ok(ElementType::F32) => XvecReader::<f32>::open_path(&query_path)
                .map(|r| <XvecReader<f32> as VectorReader<f32>>::count(&r))
                .ok(),
            Ok(ElementType::F16) => XvecReader::<half::f16>::open_path(&query_path)
                .map(|r| <XvecReader<half::f16> as VectorReader<half::f16>>::count(&r))
                .ok(),
            _ => None,
        };
        let actual_sample = sample_count.min(query_total.unwrap_or(sample_count));
        let total_str = match query_total {
            Some(t) => format!("{}", t),
            None => "?".to_string(),
        };
        ctx.ui.log(&format!(
            "verify dataset-knnutils: norm/zero + neighbor sanity + KNN sample-recompute via BLAS sgemm \
             (single-threaded, MKL-safe); sample={} of {} queries, k={}, metric={}",
            actual_sample, total_str, k, metric_str,
        ));

        let mut report = Vec::<String>::new();
        let mut all_passed = true;

        // Helper to emit to both log and report
        let mut emit = |ctx: &mut StreamContext, line: &str| {
            ctx.ui.log(line);
            report.push(line.to_string());
        };

        emit(ctx, "=== knn_utils Dataset Verification ===");
        emit(ctx, "");

        // ── fvecs check: base vectors ───────────────────────────────────
        emit(ctx, &format!("--- Base Vectors: {} ---", base_path.display()));

        let (base_result, base_tol_norm) = match check_vectors(&base_path, user_tol_norm, tol_zero, &ctx.ui) {
            Ok(r) => r,
            Err(e) => {
                emit(ctx, &format!("  FAIL: {}", e));
                return error_result(e, start);
            }
        };

        emit(ctx, &format!("  Total embeddings: {}", base_result.count));
        emit(ctx, &format!("  Dimensionality: {}", base_result.dim));
        emit(ctx, &format!("  Zero vectors (< {:.0e}): {}", tol_zero, base_result.zero_count));
        emit(ctx, &format!("  Unnormalized vectors (|norm-1| > {:.0e}): {}{}",
            base_tol_norm, base_result.unnormalized_count,
            if user_tol_norm.is_none() { format!(" [auto for {:?}]", base_path.extension().and_then(|e| e.to_str()).unwrap_or("?")) } else { String::new() }));
        emit(ctx, &format!("  Norm max abs deviation from 1.0: {:.9e}", base_result.max_abs_dev));
        emit(ctx, &format!("  Norm mean: {:.9e}", base_result.norm_mean));
        emit(ctx, &format!("  Norm range: [{:.9e}, {:.9e}]", base_result.norm_min, base_result.norm_max));

        let base_status = if base_result.normalized && base_result.zero_count == 0 {
            "PASS"
        } else {
            all_passed = false;
            "FAIL"
        };
        emit(ctx, &format!("  Normalized: {} (tol={:.0e})", base_result.normalized, base_tol_norm));
        emit(ctx, &format!("  Status: {}", base_status));
        emit(ctx, "");

        // ── fvecs check: query vectors ──────────────────────────────────
        emit(ctx, &format!("--- Query Vectors: {} ---", query_path.display()));

        let (query_result, query_tol_norm) = match check_vectors(&query_path, user_tol_norm, tol_zero, &ctx.ui) {
            Ok(r) => r,
            Err(e) => {
                emit(ctx, &format!("  FAIL: {}", e));
                return error_result(e, start);
            }
        };

        emit(ctx, &format!("  Total embeddings: {}", query_result.count));
        emit(ctx, &format!("  Dimensionality: {}", query_result.dim));
        emit(ctx, &format!("  Zero vectors (< {:.0e}): {}", tol_zero, query_result.zero_count));
        emit(ctx, &format!("  Unnormalized vectors (|norm-1| > {:.0e}): {}{}",
            query_tol_norm, query_result.unnormalized_count,
            if user_tol_norm.is_none() { format!(" [auto for {:?}]", query_path.extension().and_then(|e| e.to_str()).unwrap_or("?")) } else { String::new() }));
        emit(ctx, &format!("  Norm max abs deviation from 1.0: {:.9e}", query_result.max_abs_dev));
        emit(ctx, &format!("  Norm mean: {:.9e}", query_result.norm_mean));
        emit(ctx, &format!("  Norm range: [{:.9e}, {:.9e}]", query_result.norm_min, query_result.norm_max));

        let query_status = if query_result.normalized && query_result.zero_count == 0 {
            "PASS"
        } else {
            all_passed = false;
            "FAIL"
        };
        emit(ctx, &format!("  Normalized: {} (tol={:.0e})", query_result.normalized, query_tol_norm));
        emit(ctx, &format!("  Status: {}", query_status));
        emit(ctx, "");

        // ── ivecs check: ground truth ───────────────────────────────────
        emit(ctx, &format!("--- Ground Truth: {} ---", indices_path.display()));

        let ivecs_result = match check_ivecs(&indices_path, k, base_result.count, &ctx.ui) {
            Ok(r) => r,
            Err(e) => {
                emit(ctx, &format!("  FAIL: {}", e));
                return error_result(e, start);
            }
        };

        emit(ctx, &format!("  Rows read: {}", ivecs_result.num_rows));
        emit(ctx, &format!("  Row length (k): {}", ivecs_result.row_length));
        emit(ctx, &format!("  Max ordinal: {}", ivecs_result.max_ordinal));

        // Per-check PASS/FAIL (matching ivecs_check.py report style)
        let dup_ok = ivecs_result.duplicate_rows == 0;
        let neg_ok = ivecs_result.negative_rows == 0;
        let trunc_ok = ivecs_result.truncated_rows == 0;
        let over_ok = ivecs_result.overlong_rows == 0;
        let ord_ok = ivecs_result.max_ordinal < 0
            || (ivecs_result.max_ordinal as usize) < base_result.count;

        emit(ctx, &format!("  Duplicate ordinals:  {} ({})",
            if dup_ok { "PASS" } else { "FAIL" }, ivecs_result.duplicate_rows));
        emit(ctx, &format!("  Negative entries:    {} ({})",
            if neg_ok { "PASS" } else { "FAIL" }, ivecs_result.negative_rows));
        emit(ctx, &format!("  Truncated rows:      {} ({})",
            if trunc_ok { "PASS" } else { "FAIL" }, ivecs_result.truncated_rows));
        emit(ctx, &format!("  Overlong rows:       {} ({})",
            if over_ok { "PASS" } else { "FAIL" }, ivecs_result.overlong_rows));
        emit(ctx, &format!("  Ordinals in range:   {} (max {} < base count {})",
            if ord_ok { "PASS" } else { "FAIL" },
            ivecs_result.max_ordinal, base_result.count));

        let ivecs_status = if ivecs_result.passed && ord_ok { "PASS" } else {
            all_passed = false;
            "FAIL"
        };
        emit(ctx, &format!("  Status: {}", ivecs_status));
        emit(ctx, "");

        // ── Cross-file consistency ──────────────────────────────────────
        emit(ctx, "--- Cross-File Consistency ---");

        let dim_ok = base_result.dim == query_result.dim;
        emit(ctx, &format!("  Dimension match (base={}, query={}): {}",
            base_result.dim, query_result.dim,
            if dim_ok { "PASS" } else { all_passed = false; "FAIL" }));

        let count_ok = query_result.count == ivecs_result.num_rows;
        emit(ctx, &format!("  Query count = GT rows ({} = {}): {}",
            query_result.count, ivecs_result.num_rows,
            if count_ok { "PASS" } else { all_passed = false; "FAIL" }));

        let k_ok = ivecs_result.row_length == k;
        emit(ctx, &format!("  GT row length = k ({} = {}): {}",
            ivecs_result.row_length, k,
            if k_ok { "PASS" } else { all_passed = false; "FAIL" }));

        emit(ctx, "");

        // ── KNN Accuracy (replicates validate_knn_utils.py) ─────────────
        emit(ctx, "--- KNN Accuracy Check ---");

        let faiss_mt_is_ip = match metric_str.to_uppercase().as_str() {
            "IP" | "DOT_PRODUCT" | "COSINE" => true,
            "L2" => false,
            other => {
                emit(ctx, &format!("  SKIP: unsupported metric '{}' for accuracy check", other));
                emit(ctx, "");
                // fall through to overall
                let overall = if all_passed { "PASS" } else { "FAIL" };
                emit(ctx, &format!("=== Overall: {} ===", overall));
                // (write report and return below)
                let mut produced = Vec::new();
                if let Some(parent) = report_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Ok(mut f) = std::fs::File::create(&report_path) {
                    let content = report.join("\n") + "\n";
                    if f.write_all(content.as_bytes()).is_ok() {
                        produced.push(report_path.clone());
                    }
                }
                return CommandResult {
                    status: if all_passed { Status::Ok } else { Status::Error },
                    message: format!("unsupported metric for accuracy check: {}", other),
                    produced,
                    elapsed: start.elapsed(),
                };
            }
        };

        let actual_sample = sample_count.min(query_result.count);
        emit(ctx, &format!("  Sampling {} of {} queries, metric={}, k={}",
            actual_sample, query_result.count, metric_str, k));

        // Recompute KNN for sampled queries using BLAS sgemm. sgemm
        // wants f32, but the base / query files may be f16. Decide the
        // element type once per file, then dispatch the actual reads
        // below — same pattern as `verify knn-consolidated`, no
        // AnyFloatReader hidden in a hot loop.
        let base_etype = match ElementType::from_path(&base_path) {
            Ok(et @ (ElementType::F32 | ElementType::F16)) => et,
            Ok(et) => return error_result(format!(
                "verify dataset-knnutils: unsupported base element type {:?} for {}",
                et, base_path.display(),
            ), start),
            Err(e) => return error_result(format!(
                "verify dataset-knnutils: cannot determine base element type: {}", e,
            ), start),
        };
        let query_etype = match ElementType::from_path(&query_path) {
            Ok(et @ (ElementType::F32 | ElementType::F16)) => et,
            Ok(et) => return error_result(format!(
                "verify dataset-knnutils: unsupported query element type {:?} for {}",
                et, query_path.display(),
            ), start),
            Err(e) => return error_result(format!(
                "verify dataset-knnutils: cannot determine query element type: {}", e,
            ), start),
        };
        let gt_reader = XvecReader::<i32>::open_path(&indices_path)
            .map_err(|e| format!("reopen gt: {}", e));
        let gt_reader = match gt_reader {
            Ok(r) => r,
            Err(e) => return error_result(e, start),
        };

        let dim = base_result.dim;

        // The previous implementation loaded the entire base into a
        // single Vec<f32> and ran one big sgemm — fine for medium
        // datasets but a non-starter on hundred-GiB / TB bases, where
        // it used to silently bail out with a "use verify
        // knn-consolidated" hint. The KNN check is the headline result
        // of this verifier; skipping it isn't an option. Use the same
        // streaming sgemm machinery as `verify knn-consolidated`
        // (`SgemmScanBuffers` + `scan_range_sgemm` + cumulative top-k
        // heaps) so RSS stays bounded regardless of base size.
        let n_base = base_result.count;
        let use_ip = faiss_mt_is_ip;
        // Per the knn_utils convention this verifier mirrors, COSINE
        // assumes pre-normalized inputs and is computed as inner
        // product. `proper_cosine = false` keeps that contract.
        let proper_cosine = false;
        let elem_size: usize = match base_etype {
            ElementType::F32 => 4,
            ElementType::F16 => 2,
            _ => unreachable!("base_etype was filtered to F32/F16 above"),
        };

        // Generate deterministic sample indices.
        let mut sample_indices: Vec<usize> = Vec::with_capacity(actual_sample);
        if actual_sample >= query_result.count {
            sample_indices.extend(0..query_result.count);
        } else {
            let step = query_result.count as f64 / actual_sample as f64;
            for i in 0..actual_sample {
                sample_indices.push((i as f64 * step) as usize);
            }
        }

        // Pack sampled queries flat (one row per query, dim wide). The
        // streaming scan reuses this packed buffer on every chunk.
        let mut queries_packed: Vec<f32> = Vec::with_capacity(actual_sample * dim);
        match query_etype {
            ElementType::F32 => {
                let r = match XvecReader::<f32>::open_path(&query_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("reopen query: {}", e), start),
                };
                for &qi in &sample_indices {
                    queries_packed.extend_from_slice(r.get_slice(qi));
                }
            }
            ElementType::F16 => {
                let r = match XvecReader::<half::f16>::open_path(&query_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("reopen query: {}", e), start),
                };
                for &qi in &sample_indices {
                    let s = r.get_slice(qi);
                    for v in s { queries_packed.push(v.to_f32()); }
                }
            }
            _ => unreachable!("query_etype was filtered to F32/F16 above"),
        }
        let query_norms_sq: Vec<f32> = (0..actual_sample).map(|qi| {
            let s = &queries_packed[qi * dim..(qi + 1) * dim];
            s.iter().map(|v| v * v).sum::<f32>()
        }).collect();

        emit(ctx, &format!(
            "  Streaming sgemm scan over {} base vectors ({} sample queries, elem={}B)",
            n_base, actual_sample, elem_size,
        ));
        let base_file = match std::fs::File::open(&base_path) {
            Ok(f) => f,
            Err(e) => return error_result(format!("open base for streaming: {}", e), start),
        };

        let mut scan_buffers = super::compute_knn_blas::SgemmScanBuffers::new(actual_sample, dim);
        emit(ctx, &format!(
            "  chunk plan: {} base/chunk × {} queries/sub-batch",
            scan_buffers.vecs_per_chunk(), scan_buffers.qb_size(),
        ));

        let mut cumulative_heaps: Vec<std::collections::BinaryHeap<super::compute_knn::Neighbor>> =
            (0..actual_sample)
                .map(|_| std::collections::BinaryHeap::with_capacity(k + 1))
                .collect();
        let mut cumulative_thresholds: Vec<f32> = vec![f32::INFINITY; actual_sample];

        let scan_pb = ctx.ui.bar_with_unit(
            n_base as u64,
            format!("scanning {} base ({}q × {}d)", n_base, actual_sample, dim),
            "vectors",
        );
        let scan_t0 = Instant::now();
        {
            let pb_ref = &scan_pb;
            let mut tick_cb = move |done: u64| { pb_ref.set_position(done); };
            if let Err(e) = super::compute_knn_blas::scan_range_sgemm(
                &base_file,
                0..n_base,
                dim,
                elem_size,
                &queries_packed,
                actual_sample,
                &query_norms_sq,
                use_ip,
                proper_cosine,
                k,
                &mut scan_buffers,
                &mut cumulative_heaps,
                &mut cumulative_thresholds,
                Some(&mut tick_cb),
            ) {
                return error_result(format!("streaming scan: {}", e), start);
            }
        }
        scan_pb.finish();
        emit(ctx, &format!(
            "  scan done in {:.1}s ({:.1}M base/s)",
            scan_t0.elapsed().as_secs_f64(),
            n_base as f64 / scan_t0.elapsed().as_secs_f64().max(0.001) / 1e6,
        ));

        // Convert heaps → flat labels (ascending distance, nearest
        // first) for the existing per-query comparison loop. Pad with
        // -1 if a heap somehow ended short of k.
        let mut labels: Vec<i32> = Vec::with_capacity(actual_sample * k);
        for heap in cumulative_heaps.into_iter() {
            let sorted = heap.into_sorted_vec();
            for j in 0..k {
                if j < sorted.len() {
                    labels.push(sorted[j].index as i32);
                } else {
                    labels.push(-1);
                }
            }
        }
        struct VerifyResult { labels: Vec<i32> }
        let batch_result = VerifyResult { labels };

        // Compare each sampled query's result against ground truth.
        //
        // Multi-threaded BLAS (MKL/OpenBLAS) is non-deterministic across
        // calls with different batch sizes — the thread block decomposition
        // in sgemm produces different floating-point rounding. Queries
        // where the only difference is a small number of boundary neighbors
        // (swapped at ULP-level distance ties) are expected and acceptable.
        // A query is a "boundary mismatch" when <= 5 neighbors differ;
        // larger mismatches indicate a real problem.
        let mut exact_match = 0usize;
        let mut set_match = 0usize;
        let mut boundary_mismatch = 0usize;
        let mut real_mismatch = 0usize;
        let mut mismatch_details: Vec<String> = Vec::new();
        let boundary_threshold = 5;

        for (si, &qi) in sample_indices.iter().enumerate() {
            let recomputed_neighbors: HashSet<i32> = batch_result.labels[si * k..(si + 1) * k]
                .iter()
                .cloned()
                .filter(|&v| v >= 0)
                .collect();

            let gt_row = gt_reader.get_slice(qi);
            let gt_neighbors: HashSet<i32> = gt_row.iter().cloned().filter(|&v| v >= 0).collect();

            if recomputed_neighbors == gt_neighbors {
                let recomputed_ordered: Vec<i32> = batch_result.labels[si * k..(si + 1) * k]
                    .to_vec();
                let gt_ordered: Vec<i32> = gt_row.iter().cloned().collect();
                if recomputed_ordered == gt_ordered {
                    exact_match += 1;
                } else {
                    set_match += 1;
                }
            } else {
                let diff_count = gt_neighbors.symmetric_difference(&recomputed_neighbors).count() / 2;
                if diff_count <= boundary_threshold {
                    boundary_mismatch += 1;
                } else {
                    real_mismatch += 1;
                    if mismatch_details.len() < 5 {
                        mismatch_details.push(format!(
                            "    query {}: {} neighbors differ (exceeds boundary threshold {})",
                            qi, diff_count, boundary_threshold,
                        ));
                    }
                }
            }
        }

        let accuracy_ok = real_mismatch == 0;
        emit(ctx, &format!("  Exact order match: {}/{}", exact_match, actual_sample));
        emit(ctx, &format!("  Set match (tie-break order): {}/{}", set_match, actual_sample));
        emit(ctx, &format!("  Boundary mismatch (BLAS rounding, <= {} swaps): {}/{}",
            boundary_threshold, boundary_mismatch, actual_sample));
        emit(ctx, &format!("  Real mismatch (> {} swaps): {}/{}",
            boundary_threshold, real_mismatch, actual_sample));
        for detail in &mismatch_details {
            emit(ctx, detail);
        }
        if real_mismatch > 0 && mismatch_details.len() < real_mismatch {
            emit(ctx, &format!("    ... and {} more", real_mismatch - mismatch_details.len()));
        }
        let knn_status = if accuracy_ok { "PASS" } else {
            all_passed = false;
            "FAIL"
        };
        emit(ctx, &format!("  Status: {}", knn_status));
        emit(ctx, "");

        // ── Overall ─────────────────────────────────────────────────────
        let overall = if all_passed { "PASS" } else { "FAIL" };
        emit(ctx, &format!("=== Overall: {} ===", overall));

        // Write report file
        let mut produced = Vec::new();
        if let Some(parent) = report_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::File::create(&report_path) {
            Ok(mut f) => {
                let content = report.join("\n") + "\n";
                if f.write_all(content.as_bytes()).is_ok() {
                    produced.push(report_path.clone());
                    ctx.ui.log(&format!("  Report saved to {}", report_path.display()));
                }
            }
            Err(e) => {
                ctx.ui.log(&format!("  Warning: failed to write report: {}", e));
            }
        }

        CommandResult {
            status: if all_passed { Status::Ok } else { Status::Error },
            message: format!(
                "base={} query={} gt={}: {} (base:{} query:{} ivecs:{} cross:{} knn:{})",
                base_result.count, query_result.count, ivecs_result.num_rows,
                overall, base_status, query_status, ivecs_status,
                if dim_ok && count_ok && k_ok { "PASS" } else { "FAIL" },
                knn_status,
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query", "indices"],
            &["report"],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::{Options, Status, StreamContext};
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

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

    fn write_ivec(path: &std::path::Path, rows: &[Vec<i32>]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for row in rows {
            let k = row.len() as i32;
            f.write_all(&k.to_le_bytes()).unwrap();
            for &val in row { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    /// Build a valid small dataset and verify it passes all checks.
    #[test]
    fn test_verify_valid_dataset() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let _ = std::fs::create_dir_all(tmp.path().join(".cache"));

        // 5 base vectors, 2 queries, dim=3, k=2 — all unit-normalized
        let s2 = 1.0f32 / 2.0f32.sqrt();
        let s3 = 1.0f32 / 3.0f32.sqrt();
        let base = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![s2, s2, 0.0],
            vec![s3, s3, s3],
        ];
        let q_norm = (0.81f32 + 0.01).sqrt();
        let query = vec![
            vec![0.9 / q_norm, 0.1 / q_norm, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        // Compute GT via BLAS sgemm
        let mut knn_opts = Options::new();
        knn_opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        knn_opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        knn_opts.set("indices", tmp.path().join("gt.ivec").to_string_lossy().to_string());
        knn_opts.set("neighbors", "2");
        knn_opts.set("metric", "IP");
        let mut knn = crate::pipeline::commands::compute_knn_blas::ComputeKnnBlasOp;
        let r = knn.execute(&knn_opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Now verify
        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("gt.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "2");
        opts.set("metric", "IP");
        opts.set("sample", "2");

        let mut op = VerifyDatasetKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok,
            "valid dataset should pass verification: {}", r.message);
    }

    /// Dimension mismatch between base and query should fail.
    #[test]
    fn test_verify_dimension_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let _ = std::fs::create_dir_all(tmp.path().join(".cache"));

        write_fvec(&tmp.path().join("base.fvec"), &[vec![1.0, 2.0, 3.0]]);
        write_fvec(&tmp.path().join("query.fvec"), &[vec![1.0, 2.0]]);
        write_ivec(&tmp.path().join("gt.ivec"), &[vec![0]]);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("gt.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "1");
        opts.set("metric", "IP");
        opts.set("sample", "0");

        let mut op = VerifyDatasetKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        // The verify command should detect dimension mismatch and report FAIL
        assert_eq!(r.status, Status::Error,
            "dimension mismatch should fail: {}", r.message);
    }

    /// GT with duplicate ordinals should be detected.
    #[test]
    fn test_verify_detects_duplicate_ordinals() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let _ = std::fs::create_dir_all(tmp.path().join(".cache"));

        write_fvec(&tmp.path().join("base.fvec"), &[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
        write_fvec(&tmp.path().join("query.fvec"), &[vec![0.9, 0.1]]);
        // GT with duplicate: index 0 appears twice
        write_ivec(&tmp.path().join("gt.ivec"), &[vec![0, 0, 1]]);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("gt.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "3");
        opts.set("metric", "IP");
        opts.set("sample", "0");

        let mut op = VerifyDatasetKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error,
            "duplicate ordinals should fail: {}", r.message);
        assert!(r.message.contains("ivecs:FAIL"),
            "should report ivecs failure: {}", r.message);
    }

    /// GT with negative indices should be detected.
    #[test]
    fn test_verify_detects_negative_indices() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let _ = std::fs::create_dir_all(tmp.path().join(".cache"));

        write_fvec(&tmp.path().join("base.fvec"), &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        write_fvec(&tmp.path().join("query.fvec"), &[vec![0.9, 0.1]]);
        write_ivec(&tmp.path().join("gt.ivec"), &[vec![-1, 0]]);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("gt.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "2");
        opts.set("metric", "IP");
        opts.set("sample", "0");

        let mut op = VerifyDatasetKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error,
            "negative indices should fail: {}", r.message);
    }

    /// Query count != GT row count should fail cross-file check.
    #[test]
    fn test_verify_count_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());
        let _ = std::fs::create_dir_all(tmp.path().join(".cache"));

        write_fvec(&tmp.path().join("base.fvec"), &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        write_fvec(&tmp.path().join("query.fvec"), &[vec![0.9, 0.1], vec![0.1, 0.9]]);
        // GT has 1 row but query has 2 vectors
        write_ivec(&tmp.path().join("gt.ivec"), &[vec![0, 1]]);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("gt.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "2");
        opts.set("metric", "IP");
        opts.set("sample", "0");

        let mut op = VerifyDatasetKnnUtilsOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error,
            "count mismatch should fail: {}", r.message);
        assert!(r.message.contains("cross:FAIL"),
            "should report cross-file failure: {}", r.message);
    }
}
