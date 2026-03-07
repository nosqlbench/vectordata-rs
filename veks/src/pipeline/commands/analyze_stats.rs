// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute statistics for vector dimensions.
//!
//! Computes per-dimension statistics (min, max, mean, std_dev, skewness,
//! kurtosis) and optional percentiles for fvec files.
//!
//! Equivalent to the Java `CMD_analyze_stats` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

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
    /// Compute statistics from a slice of values.
    pub fn compute(values: &[f32]) -> Self {
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
            let vf = v as f64;
            if vf < min {
                min = vf;
            }
            if vf > max {
                max = vf;
            }
            sum += vf;
        }
        let mean = sum / n as f64;

        // Pass 2: variance, skewness, kurtosis (central moments)
        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        let mut m4 = 0.0f64;
        for &v in values {
            let d = v as f64 - mean;
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

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Compute per-dimension statistics for vector files".into(),
            body: format!(
                "# analyze stats\n\nCompute per-dimension statistics for vector files.\n\n## Description\n\nComputes per-dimension statistics (min, max, mean, std_dev, skewness, kurtosis) and optional percentiles for fvec files. Can analyze a single dimension with detailed percentile breakdown, all dimensions in a summary table, or global statistics across all dimensions.\n\n## Options\n\n{}",
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

        let source_path = resolve_path(source_str, &ctx.workspace);

        let reader = match MmapVectorReader::<f32>::open_fvec(&source_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open {}: {}", source_path.display(), e),
                    start,
                )
            }
        };

        let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);
        let effective_count = sample.map(|s| s.min(count)).unwrap_or(count);

        if let Some(d) = dimension {
            if d >= dim {
                return error_result(
                    format!("dimension {} out of range (max {})", d, dim - 1),
                    start,
                );
            }

            // Single dimension analysis with percentiles
            let mut values: Vec<f32> = Vec::with_capacity(effective_count);
            for i in 0..effective_count {
                let vec = reader.get(i).unwrap_or_default();
                values.push(vec[d]);
            }

            let stats = DimensionStats::compute(&values);

            // Percentiles
            let mut sorted: Vec<f64> = values.iter().map(|&v| v as f64).collect();
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
            // Summary table for all dimensions
            ctx.ui.log(&format!(
                "{:>5} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}",
                "Dim", "Mean", "StdDev", "Min", "Max", "Skewness", "Kurtosis"
            ));
            ctx.ui.log(&format!("{}", "-".repeat(83)));

            for d in 0..dim {
                let mut values: Vec<f32> = Vec::with_capacity(effective_count);
                for i in 0..effective_count {
                    let vec = reader.get(i).unwrap_or_default();
                    values.push(vec[d]);
                }
                let stats = DimensionStats::compute(&values);
                ctx.ui.log(&format!(
                    "{:5} {:12.6} {:12.6} {:12.6} {:12.6} {:10.4} {:10.4}",
                    d, stats.mean, stats.std_dev, stats.min, stats.max, stats.skewness, stats.kurtosis
                ));
            }
        } else {
            // Global stats across all dimensions
            let mut all_values: Vec<f32> = Vec::with_capacity(effective_count * dim);
            for i in 0..effective_count {
                let vec = reader.get(i).unwrap_or_default();
                all_values.extend_from_slice(&vec);
            }
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
            },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Specific dimension to analyze (0-indexed)".to_string(),
            },
            OptionDesc {
                name: "all-dimensions".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show summary table for all dimensions".to_string(),
            },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Max vectors to sample (default: all)".to_string(),
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
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
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
