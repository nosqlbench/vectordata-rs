// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: analyze L2-norm distribution of vectors.
//!
//! Scans all (or sampled) vectors, computes the L2 (Frobenius) norm of
//! each in f64 precision, records the values in an HDR histogram, and
//! displays a percentile summary plus an ASCII plot.

use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use vectordata::VectorReader;
use vectordata::io::XvecReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use crate::pipeline::element_type::ElementType;

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

pub struct AnalyzeNormsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeNormsOp)
}

impl CommandOp for AnalyzeNormsOp {
    fn command_path(&self) -> &str {
        "analyze display-norms"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Analyze L2-norm distribution of vector files".into(),
            body: format!(
                "# analyze norms\n\n\
                Analyze L2-norm distribution of vector files.\n\n\
                ## Description\n\n\
                Scans vectors from the source file and computes the L2 \
                (Frobenius) norm of each vector in f64 precision. The norms \
                are recorded in an HDR histogram (3 significant digits) and \
                displayed as a percentile summary table and ASCII bar chart.\n\n\
                ## Output\n\n\
                - **Percentile table**: p0 (min), p25, p50, p75, p90, p99, \
                  p99.9, p100 (max)\n\
                - **ASCII histogram**: distribution shape across equal-width \
                  bins spanning the min-max range\n\
                - **Summary**: mean, stdev, count, zero count (norms below \
                  threshold)\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "threads".into(), description: "Parallel norm computation".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential mmap prefetch".into(), adjustable: false },
        ]
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Source vector file (fvec, mvec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "sample".into(),
                type_name: "integer".into(),
                required: false,
                default: None,
                description: "Sample this many vectors (default: all)".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "bins".into(),
                type_name: "integer".into(),
                required: false,
                default: Some("50".into()),
                description: "Number of histogram bins for the ASCII chart".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "width".into(),
                type_name: "integer".into(),
                required: false,
                default: Some("60".into()),
                description: "Maximum bar width in characters".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "zero-threshold".into(),
                type_name: "float".into(),
                required: false,
                default: Some("1e-06".into()),
                description: "L2-norm threshold for near-zero classification".into(),
                role: OptionRole::Config,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let source_path = resolve_path(source_str, &ctx.workspace);

        let bins: usize = options.get("bins").and_then(|s| s.parse().ok()).unwrap_or(50);
        let width: usize = options.get("width").and_then(|s| s.parse().ok()).unwrap_or(60);
        let sample: Option<usize> = options.get("sample").and_then(|s| s.parse().ok());
        let zero_threshold: f64 = options.get("zero-threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-6);

        let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;

        // Detect element type and open reader
        let etype = ElementType::from_path(&source_path)
            .unwrap_or(ElementType::F32);

        let (count, dim, norms) = match etype {
            ElementType::F16 => {
                let reader = match XvecReader::<half::f16>::open_path(&source_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
                };
                reader.advise_sequential();
                let count = VectorReader::<half::f16>::count(&reader);
                let dim = VectorReader::<half::f16>::dim(&reader);
                let indices = build_sample_indices(count, sample);
                let norms = compute_norms_f16(&reader, &indices, dim, threads, ctx);
                (count, dim, norms)
            }
            _ => {
                let reader = match XvecReader::<f32>::open_path(&source_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
                };
                reader.advise_sequential();
                let count = VectorReader::<f32>::count(&reader);
                let dim = VectorReader::<f32>::dim(&reader);
                let indices = build_sample_indices(count, sample);
                let norms = compute_norms_f32(&reader, &indices, dim, threads, ctx);
                (count, dim, norms)
            }
        };

        let scanned = norms.len();
        if scanned == 0 {
            return error_result("no vectors to analyze", start);
        }

        // Sort norms for exact percentile computation
        let mut sorted_norms = norms.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let zero_count = sorted_norms.partition_point(|&v| v < zero_threshold);
        let sum: f64 = sorted_norms.iter().sum();
        let sum_sq: f64 = sorted_norms.iter().map(|v| v * v).sum();
        let min_norm = sorted_norms[0];
        let max_norm = sorted_norms[scanned - 1];
        let mean = sum / scanned as f64;
        let variance = (sum_sq / scanned as f64) - mean * mean;
        let stdev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        let pct = |p: f64| -> f64 {
            let idx = ((p / 100.0) * (scanned - 1) as f64) as usize;
            sorted_norms[idx.min(scanned - 1)]
        };

        // Choose display precision based on the range of norms
        let range = max_norm - min_norm;
        let prec = if range < 1e-6 { 10 }
            else if range < 1e-3 { 8 }
            else if range < 1.0 { 6 }
            else { 4 };

        // Percentile summary
        ctx.ui.log(&format!(
            "analyze-norms: {} vectors scanned ({} total, dim={}), {:.1}s",
            scanned, count, dim, start.elapsed().as_secs_f64(),
        ));
        ctx.ui.log("");
        ctx.ui.log("  Percentile Summary (L2 norm):");
        ctx.ui.log(&format!("    p0   (min):  {:.p$}", min_norm, p = prec));
        ctx.ui.log(&format!("    p25:         {:.p$}", pct(25.0), p = prec));
        ctx.ui.log(&format!("    p50 (med):   {:.p$}", pct(50.0), p = prec));
        ctx.ui.log(&format!("    p75:         {:.p$}", pct(75.0), p = prec));
        ctx.ui.log(&format!("    p90:         {:.p$}", pct(90.0), p = prec));
        ctx.ui.log(&format!("    p99:         {:.p$}", pct(99.0), p = prec));
        ctx.ui.log(&format!("    p99.9:       {:.p$}", pct(99.9), p = prec));
        ctx.ui.log(&format!("    p100 (max):  {:.p$}", max_norm, p = prec));
        ctx.ui.log("");
        ctx.ui.log(&format!("    mean:   {:.p$}", mean, p = prec));
        ctx.ui.log(&format!("    stdev:  {:.2e}", stdev));
        ctx.ui.log(&format!("    zeros:  {} (L2 < {:.0e})", zero_count, zero_threshold));
        ctx.ui.log("");

        // ASCII histogram — if zeros are present, show two histograms:
        // one for the full range and one zoomed into the non-zero norms.
        if zero_count > 0 && zero_count < scanned {
            ctx.ui.log(&format!("  Full range ({} vectors including {} zeros):", scanned, zero_count));
            render_histogram(&sorted_norms, bins, width, ctx);
            let non_zero = &sorted_norms[zero_count..];
            ctx.ui.log(&format!("  Non-zero range ({} vectors, norms >= {:.0e}):", non_zero.len(), zero_threshold));
            render_histogram(non_zero, bins, width, ctx);
        } else {
            render_histogram(&sorted_norms, bins, width, ctx);
        }

        let msg = format!(
            "scanned {} vectors: mean_norm={:.6}, stdev={:.6}, zeros={}, range=[{:.6}, {:.6}]",
            scanned, mean, stdev, zero_count, min_norm, max_norm,
        );

        CommandResult {
            status: Status::Ok,
            message: msg,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }
}

/// Build the list of vector indices to scan.
fn build_sample_indices(count: usize, sample: Option<usize>) -> Vec<usize> {
    match sample {
        Some(n) if n < count => {
            // Uniform stride sampling
            let step = count / n;
            (0..n).map(|i| i * step).collect()
        }
        _ => (0..count).collect(),
    }
}

/// Compute L2 norms for f32 vectors in parallel.
fn compute_norms_f32(
    reader: &XvecReader<f32>,
    indices: &[usize],
    dim: usize,
    threads: usize,
    ctx: &mut StreamContext,
) -> Vec<f64> {
    let pb = ctx.ui.bar_with_unit(indices.len() as u64, "computing norms", "vectors");
    let progress = std::sync::atomic::AtomicU64::new(0);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let compute = || {
        indices.par_iter().map(|&idx| {
            let slice = reader.get_slice(idx);
            let mut norm_sq = 0.0f64;
            for d in 0..dim {
                let v = slice[d] as f64;
                norm_sq += v * v;
            }
            let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % 100_000 == 0 { pb.set_position(done); }
            norm_sq.sqrt()
        }).collect::<Vec<f64>>()
    };

    let result = if let Some(ref p) = pool {
        p.install(compute)
    } else {
        compute()
    };
    pb.finish();
    result
}

/// Compute L2 norms for f16 vectors in parallel.
fn compute_norms_f16(
    reader: &XvecReader<half::f16>,
    indices: &[usize],
    dim: usize,
    threads: usize,
    ctx: &mut StreamContext,
) -> Vec<f64> {
    let pb = ctx.ui.bar_with_unit(indices.len() as u64, "computing norms", "vectors");
    let progress = std::sync::atomic::AtomicU64::new(0);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let compute = || {
        indices.par_iter().map(|&idx| {
            let slice = reader.get_slice(idx);
            let mut norm_sq = 0.0f64;
            for d in 0..dim {
                let v = slice[d].to_f64();
                norm_sq += v * v;
            }
            let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % 100_000 == 0 { pb.set_position(done); }
            norm_sq.sqrt()
        }).collect::<Vec<f64>>()
    };

    let result = if let Some(ref p) = pool {
        p.install(compute)
    } else {
        compute()
    };
    pb.finish();
    result
}

/// Render an ASCII histogram of norm values.
fn render_histogram(norms: &[f64], num_bins: usize, max_width: usize, ctx: &mut StreamContext) {
    if norms.is_empty() { return; }

    let min = norms.iter().copied().fold(f64::MAX, f64::min);
    let max = norms.iter().copied().fold(0.0f64, f64::max);

    if (max - min).abs() < f64::EPSILON {
        ctx.ui.log(&format!("  All norms identical: {:.10}", min));
        return;
    }

    let bin_width = (max - min) / num_bins as f64;
    let mut counts = vec![0u64; num_bins];

    for &v in norms {
        let bin = ((v - min) / bin_width) as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);

    // Choose decimal precision based on the range width so that
    // bin edges are visually distinct even for very narrow ranges.
    let range = max - min;
    let precision = if range < 1e-6 { 10 }
        else if range < 1e-3 { 8 }
        else if range < 1.0 { 6 }
        else { 4 };
    let label_width = precision + 6; // sign + digits + decimal point + padding

    ctx.ui.log("  L2 Norm Distribution:");
    ctx.ui.log("");

    for (i, &count) in counts.iter().enumerate() {
        let lo = min + i as f64 * bin_width;
        let hi = lo + bin_width;
        let bar_len = if max_count > 0 {
            (count as f64 / max_count as f64 * max_width as f64) as usize
        } else {
            0
        };
        let bar: String = "█".repeat(bar_len);
        let count_str = if count > 0 { format!(" {}", count) } else { String::new() };
        ctx.ui.log(&format!("  {:>w$.p$} - {:<w$.p$} |{}{}",
            lo, hi, bar, count_str, w = label_width, p = precision));
    }
    ctx.ui.log("");
}
