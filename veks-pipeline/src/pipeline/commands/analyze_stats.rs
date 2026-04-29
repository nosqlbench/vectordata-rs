// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute statistics for vector dimensions.
//!
//! Computes per-dimension statistics (min, max, mean, std_dev, skewness,
//! kurtosis) and optional percentiles for fvec files.
//!
//! Equivalent to the Java `CMD_analyze_stats` command.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::element_type::ElementType;
use super::source_window::resolve_source;

/// Pipeline command: compute vector statistics.
pub struct AnalyzeStatsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeStatsOp)
}

/// Statistics for a single dimension.
#[derive(Debug, Clone)]
pub struct DimensionStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl DimensionStats {
    /// Compute statistics from a slice of f64 values.
    ///
    /// Callers should convert their element type to f64 before calling.
    pub fn compute(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return DimensionStats {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
            };
        }

        // Pass 1: min, max, sum
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }
        let mean = sum / n as f64;

        // Pass 2: variance, skewness, kurtosis (central moments)
        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        let mut m4 = 0.0f64;
        for &v in values {
            let d = v - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }

        let variance = m2 / n as f64;
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 0.0 {
            (m3 / n as f64) / (std_dev * std_dev * std_dev)
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            (m4 / n as f64) / (variance * variance)
        } else {
            0.0
        };

        DimensionStats {
            count: n,
            min,
            max,
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
        }
    }
}

/// Compute percentiles from sorted values using linear interpolation.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl CommandOp for AnalyzeStatsOp {
    fn command_path(&self) -> &str {
        "analyze stats"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Compute per-dimension statistics for vector files".into(),
            body: format!(
                "# analyze stats\n\n\
                Compute per-dimension statistics for vector files.\n\n\
                ## Description\n\n\
                Computes per-dimension statistics (min, max, mean, std_dev, skewness, \
                kurtosis) and optional percentiles for fvec files. For each dimension, \
                a two-pass algorithm is used: the first pass computes min, max, and sum \
                to derive the mean; the second pass computes central moments (variance, \
                skewness, kurtosis) relative to that mean. The L2 norm is available \
                implicitly through the variance output.\n\n\
                ## Analysis Modes\n\n\
                The command supports three analysis modes depending on the options provided:\n\n\
                - **Single dimension** (`dimension=N`): Provides a detailed breakdown \
                for one dimension including percentiles (p1, p5, p25, p50, p75, p95, p99), \
                interquartile range (IQR), outlier fences, and outlier counts.\n\
                - **All dimensions** (`all-dimensions=true`): Prints a summary table with \
                mean, std_dev, min, max, skewness, and kurtosis for every dimension.\n\
                - **Global** (default): Flattens all dimension values into a single \
                distribution and reports aggregate statistics.\n\n\
                ## Role in Dataset Preparation\n\n\
                This command is typically used after importing or generating vector data \
                to validate that values are within expected ranges. It can detect anomalies \
                such as NaN or infinity values, all-zero vectors, unexpectedly large or \
                small magnitudes, and skewed or heavy-tailed distributions that may cause \
                issues with distance computations or quantization. Running this command \
                before and after transformations (e.g., normalization, shuffling) helps \
                confirm that statistical properties are preserved.\n\n\
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

        let dimension: Option<usize> = options.get("dimension").and_then(|s| s.parse().ok());
        let all_dimensions = options.get("all-dimensions").map_or(false, |s| s == "true");
        let sample: Option<usize> = options.get("sample").and_then(|s| s.parse().ok());

        let source = match resolve_source(source_str, &ctx.workspace) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let source_path = source.path.clone();

        let etype = match ElementType::from_path(&source_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        // Open reader and extract (file_count, dim, get_f64_fn) based on element type.
        // We use a boxed closure to erase the reader type while keeping the inner
        // loops monomorphic through inlining.
        let (file_count, dim, get_f64): (usize, usize, Box<dyn Fn(usize) -> Vec<f64> + Sync>) = match etype {
            ElementType::F32 => {
                let r = match MmapVectorReader::<f32>::open_fvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f32>::count(&r);
                let d = VectorReader::<f32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::F16 => {
                let r = match MmapVectorReader::<half::f16>::open_mvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<half::f16>::count(&r);
                let d = VectorReader::<half::f16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|v| v.to_f64()).collect()))
            }
            ElementType::F64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default()))
            }
            ElementType::I32 => {
                match MmapVectorReader::<i32>::open_ivec(&source_path) {
                    Ok(r) => {
                        let fc = VectorReader::<i32>::count(&r);
                        let d = VectorReader::<i32>::dim(&r);
                        (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
                    }
                    Err(vectordata::io::IoError::VariableLengthRecords(_)) => {
                        let r = match vectordata::io::IndexedXvecReader::open_ivec(&source_path) {
                            Ok(r) => r, Err(e) => return error_result(format!("open indexed ivec: {}", e), start),
                        };
                        let rc = r.count();
                        (rc, 0, Box::new(move |i: usize| {
                            r.get_i32(i).unwrap_or_default().iter().map(|&v| v as f64).collect()
                        }) as Box<dyn Fn(usize) -> Vec<f64> + Sync>)
                    }
                    Err(e) => return error_result(format!("open: {}", e), start),
                }
            }
            ElementType::I16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U8 | ElementType::I8 => {
                let r = match MmapVectorReader::<u8>::open_bvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<u8>::count(&r);
                let d = VectorReader::<u8>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U32 => {
                match MmapVectorReader::<i32>::open_ivec(&source_path) {
                    Ok(r) => {
                        let fc = VectorReader::<i32>::count(&r);
                        let d = VectorReader::<i32>::dim(&r);
                        (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
                    }
                    Err(vectordata::io::IoError::VariableLengthRecords(_)) => {
                        let r = match vectordata::io::IndexedXvecReader::open_ivec(&source_path) {
                            Ok(r) => r, Err(e) => return error_result(format!("open indexed ivec: {}", e), start),
                        };
                        let rc = r.count();
                        (rc, 0, Box::new(move |i: usize| {
                            r.get_i32(i).unwrap_or_default().iter().map(|&v| v as f64).collect()
                        }) as Box<dyn Fn(usize) -> Vec<f64> + Sync>)
                    }
                    Err(e) => return error_result(format!("open: {}", e), start),
                }
            }
            ElementType::U64 | ElementType::I64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
        };

        ctx.ui.log(&format!("  source: {} ({}, dim={}, {} records)", source_path.display(), etype, dim, file_count));

        let (base_offset, count) = match source.window {
            Some((ws, we)) => {
                let ws = ws.min(file_count);
                let we = we.min(file_count);
                (ws, we.saturating_sub(ws))
            }
            None => (0, file_count),
        };
        let effective_count = sample.map(|s| s.min(count)).unwrap_or(count);

        let threads = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        if let Some(d) = dimension {
            if d >= dim {
                return error_result(
                    format!("dimension {} out of range (max {})", d, dim - 1),
                    start,
                );
            }

            // Single dimension analysis with percentiles
            let pb = ctx.ui.bar_with_unit(effective_count as u64, "reading dimension values", "vectors");
            let mut values: Vec<f64> = Vec::with_capacity(effective_count);
            for i in 0..effective_count {
                let vec = get_f64(base_offset + i);
                values.push(vec[d]);
                if (i + 1) % 100_000 == 0 { pb.set_position((i + 1) as u64); }
            }
            pb.finish();

            let stats = DimensionStats::compute(&values);

            // Percentiles
            let mut sorted: Vec<f64> = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let p1 = percentile(&sorted, 1.0);
            let p5 = percentile(&sorted, 5.0);
            let p25 = percentile(&sorted, 25.0);
            let p50 = percentile(&sorted, 50.0);
            let p75 = percentile(&sorted, 75.0);
            let p95 = percentile(&sorted, 95.0);
            let p99 = percentile(&sorted, 99.0);

            let iqr = p75 - p25;
            let lower_fence = p25 - 1.5 * iqr;
            let upper_fence = p75 + 1.5 * iqr;
            let outliers = sorted
                .iter()
                .filter(|&&v| v < lower_fence || v > upper_fence)
                .count();

            ctx.ui.log(&format!("Dimension {} statistics ({} vectors):", d, stats.count));
            ctx.ui.log(&format!("  Mean:     {:.6}", stats.mean));
            ctx.ui.log(&format!("  Variance: {:.6}", stats.variance));
            ctx.ui.log(&format!("  StdDev:   {:.6}", stats.std_dev));
            ctx.ui.log(&format!("  Min:      {:.6}", stats.min));
            ctx.ui.log(&format!("  Max:      {:.6}", stats.max));
            ctx.ui.log(&format!("  Skewness: {:.4} ({})", stats.skewness, skewness_label(stats.skewness)));
            ctx.ui.log(&format!("  Kurtosis: {:.4} ({})", stats.kurtosis, kurtosis_label(stats.kurtosis)));
            ctx.ui.log("  Percentiles:");
            ctx.ui.log(&format!(
                "    p1={:.4}  p5={:.4}  p25={:.4}  p50={:.4}  p75={:.4}  p95={:.4}  p99={:.4}",
                p1, p5, p25, p50, p75, p95, p99
            ));
            ctx.ui.log(&format!(
                "  Outliers: {} ({:.1}%), fences: [{:.4}, {:.4}]",
                outliers,
                outliers as f64 / effective_count as f64 * 100.0,
                lower_fence,
                upper_fence
            ));
        } else if all_dimensions {
            // Summary table for all dimensions -- parallelize per-dimension stats
            let pb = ctx.ui.bar_with_unit(dim as u64, "computing per-dimension stats", "vectors");
            let progress = AtomicU64::new(0);
            let pb_ref = &pb;
            let progress_ref = &progress;

            let get_f64_ref = &get_f64;
            let compute_fn = || {
                (0..dim)
                    .into_par_iter()
                    .map(|d| {
                        let mut values: Vec<f64> = Vec::with_capacity(effective_count);
                        for i in 0..effective_count {
                            let vec = get_f64_ref(base_offset + i);
                            values.push(vec[d]);
                        }
                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        pb_ref.set_position(done);
                        (d, DimensionStats::compute(&values))
                    })
                    .collect::<Vec<_>>()
            };

            let mut dim_stats: Vec<(usize, DimensionStats)> = if let Some(ref p) = pool {
                p.install(compute_fn)
            } else {
                compute_fn()
            };
            pb.finish();

            dim_stats.sort_by_key(|(d, _)| *d);

            ctx.ui.log(&format!(
                "{:>5} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}",
                "Dim", "Mean", "StdDev", "Min", "Max", "Skewness", "Kurtosis"
            ));
            ctx.ui.log(&format!("{}", "-".repeat(83)));

            for (d, stats) in &dim_stats {
                ctx.ui.log(&format!(
                    "{:5} {:12.6} {:12.6} {:12.6} {:12.6} {:10.4} {:10.4}",
                    d, stats.mean, stats.std_dev, stats.min, stats.max, stats.skewness, stats.kurtosis
                ));
            }
        } else {
            // Global stats across all dimensions
            let pb = ctx.ui.bar_with_unit(effective_count as u64, "reading vectors", "vectors");
            let mut all_values: Vec<f64> = Vec::with_capacity(effective_count * dim);
            for i in 0..effective_count {
                all_values.extend(get_f64(base_offset + i));
                if (i + 1) % 100_000 == 0 { pb.set_position((i + 1) as u64); }
            }
            pb.finish();

            let stats = DimensionStats::compute(&all_values);

            ctx.ui.log(&format!(
                "Global statistics ({} vectors, {} dims, {} values):",
                effective_count, dim, all_values.len()
            ));
            ctx.ui.log(&format!("  Mean:     {:.6}", stats.mean));
            ctx.ui.log(&format!("  StdDev:   {:.6}", stats.std_dev));
            ctx.ui.log(&format!("  Min:      {:.6}", stats.min));
            ctx.ui.log(&format!("  Max:      {:.6}", stats.max));
            ctx.ui.log(&format!("  Skewness: {:.4}", stats.skewness));
            ctx.ui.log(&format!("  Kurtosis: {:.4}", stats.kurtosis));
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "computed stats for {} (dim={}, {} vectors)",
                source_path.display(),
                dim,
                effective_count
            ),
            produced: vec![],
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
                description: "Input fvec file".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Specific dimension to analyze (0-indexed)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "all-dimensions".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show summary table for all dimensions".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Max vectors to sample (default: all)".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

fn skewness_label(s: f64) -> &'static str {
    if s.abs() < 0.5 {
        "symmetric"
    } else if s > 0.0 {
        "right-skewed"
    } else {
        "left-skewed"
    }
}

fn kurtosis_label(k: f64) -> &'static str {
    if k > 3.5 {
        "leptokurtic/heavy-tailed"
    } else if k < 2.5 {
        "platykurtic/light-tailed"
    } else {
        "mesokurtic/normal-like"
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
    use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
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
    fn test_dimension_stats_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = DimensionStats::compute(&values);
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_dimension_stats_empty() {
        let stats = DimensionStats::compute(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_dimension_stats_constant() {
        let values = vec![5.0; 100];
        let stats = DimensionStats::compute(&values);
        assert!((stats.mean - 5.0).abs() < 1e-6);
        assert!(stats.std_dev < 1e-6);
        assert!(stats.variance < 1e-6);
    }

    #[test]
    fn test_percentile() {
        let sorted: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!((percentile(&sorted, 50.0) - 49.5).abs() < 1e-6);
        assert!((percentile(&sorted, 0.0) - 0.0).abs() < 1e-6);
        assert!((percentile(&sorted, 100.0) - 99.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_command_global() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("vectors.fvec");
        let mut opts = Options::new();
        opts.set("output", source.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "100");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());

        let mut stats_op = AnalyzeStatsOp;
        let result = stats_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_stats_command_single_dim() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("vectors.fvec");
        let mut opts = Options::new();
        opts.set("output", source.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "50");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("dimension", "2");

        let mut stats_op = AnalyzeStatsOp;
        let result = stats_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_stats_invalid_dimension() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("vectors.fvec");
        let mut opts = Options::new();
        opts.set("output", source.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "10");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("dimension", "10"); // out of range

        let mut stats_op = AnalyzeStatsOp;
        let result = stats_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }
}
