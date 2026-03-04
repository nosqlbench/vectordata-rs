// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate histogram for vector dimension distributions.
//!
//! Reads an fvec file and produces an ASCII histogram of values for a specific
//! dimension, showing the distribution shape.
//!
//! Equivalent to the Java `CMD_analyze_histogram` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};
use super::analyze_stats::DimensionStats;

/// Pipeline command: histogram visualization.
pub struct AnalyzeHistogramOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeHistogramOp)
}

impl CommandOp for AnalyzeHistogramOp {
    fn command_path(&self) -> &str {
        "analyze histogram"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let dimension: usize = match options.require("dimension") {
            Ok(s) => match s.parse() {
                Ok(d) => d,
                _ => return error_result(format!("invalid dimension: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let bins: usize = options
            .get("bins")
            .and_then(|s| s.parse().ok())
            .unwrap_or(40);
        let width: usize = options
            .get("width")
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
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

        if dimension >= dim {
            return error_result(
                format!("dimension {} out of range (max {})", dimension, dim - 1),
                start,
            );
        }

        let effective_count = sample.map(|s| s.min(count)).unwrap_or(count);

        // Extract values for the dimension
        let mut values: Vec<f32> = Vec::with_capacity(effective_count);
        for i in 0..effective_count {
            let vec = reader.get(i).unwrap_or_default();
            values.push(vec[dimension]);
        }

        let stats = DimensionStats::compute(&values);

        if values.is_empty() {
            return error_result("no values to histogram".to_string(), start);
        }

        // Compute histogram bins
        let min = stats.min;
        let max = stats.max;

        if (max - min).abs() < f64::EPSILON {
            eprintln!(
                "Dimension {}: all {} values = {:.6}",
                dimension, effective_count, min
            );
            return CommandResult {
                status: Status::Ok,
                message: format!("histogram: all values identical ({:.6})", min),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let bin_width = (max - min) / bins as f64;
        let mut bin_counts = vec![0usize; bins];

        for &v in &values {
            let idx = ((v as f64 - min) / bin_width) as usize;
            let idx = idx.min(bins - 1);
            bin_counts[idx] += 1;
        }

        let max_count = *bin_counts.iter().max().unwrap_or(&1);

        // Print header
        eprintln!(
            "Histogram: dimension {} ({} vectors)",
            dimension, effective_count
        );
        eprintln!(
            "  Range: [{:.4}, {:.4}], Mean: {:.4}, StdDev: {:.4}",
            min, max, stats.mean, stats.std_dev
        );
        eprintln!("  {} bins, bin width: {:.6}", bins, bin_width);
        eprintln!();

        // Print histogram
        for (i, &count) in bin_counts.iter().enumerate() {
            let lo = min + i as f64 * bin_width;
            let hi = lo + bin_width;
            let bar_len = if max_count > 0 {
                (count as f64 / max_count as f64 * width as f64) as usize
            } else {
                0
            };
            let bar: String = "\u{2588}".repeat(bar_len);
            eprintln!(
                "  [{:8.4}, {:8.4}) {:6} |{}",
                lo, hi, count, bar
            );
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "histogram for dimension {} of {} ({} bins, {} vectors)",
                dimension,
                source_path.display(),
                bins,
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
                required: true,
                default: None,
                description: "Dimension index to visualize (0-indexed)".to_string(),
            },
            OptionDesc {
                name: "bins".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("40".to_string()),
                description: "Number of histogram bins".to_string(),
            },
            OptionDesc {
                name: "width".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("60".to_string()),
                description: "Width of histogram bars in characters".to_string(),
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
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
        }
    }

    #[test]
    fn test_histogram_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("vectors.fvec");
        let mut opts = Options::new();
        opts.set("output", source.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "200");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("dimension", "0");
        opts.set("bins", "10");

        let mut hist_op = AnalyzeHistogramOp;
        let result = hist_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("10 bins"));
    }

    #[test]
    fn test_histogram_invalid_dimension() {
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
        opts.set("dimension", "10");

        let mut hist_op = AnalyzeHistogramOp;
        let result = hist_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }

    #[test]
    fn test_histogram_with_sample() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("vectors.fvec");
        let mut opts = Options::new();
        opts.set("output", source.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "100");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("dimension", "2");
        opts.set("sample", "50");
        opts.set("bins", "5");

        let mut hist_op = AnalyzeHistogramOp;
        let result = hist_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("50 vectors"));
    }
}
