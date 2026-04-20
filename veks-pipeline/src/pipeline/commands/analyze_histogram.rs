// Copyright (c) Jonathan Shook
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
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::element_type::ElementType;
use super::analyze_stats::DimensionStats;

/// Pipeline command: histogram visualization.
pub struct AnalyzeHistogramOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeHistogramOp)
}

impl CommandOp for AnalyzeHistogramOp {
    fn command_path(&self) -> &str {
        "analyze display-histogram"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate dimension-value histograms for vector files".into(),
            body: format!(
                "# analyze histogram\n\n\
                Generate dimension-value histograms for vector files.\n\n\
                ## Description\n\n\
                Reads an fvec file and builds a frequency distribution for a specific \
                dimension, then renders it as an ASCII histogram with configurable bin \
                count and bar width. The output includes the value range, mean, and \
                standard deviation alongside the bar chart, making it easy to visually \
                assess distribution shape at a glance.\n\n\
                ## How It Works\n\n\
                The command extracts all values for the specified dimension, computes \
                basic statistics via `DimensionStats`, then bins the values into \
                equal-width intervals spanning the min-max range. Each bin is rendered \
                as a row of Unicode block characters whose length is proportional to \
                the bin count relative to the most populated bin.\n\n\
                ## Role in Dataset Preparation\n\n\
                Histograms are useful for spotting quantization artifacts (e.g., values \
                clustering at specific levels after int8 conversion), skewed or \
                multi-modal distributions, and unexpected gaps or spikes in data. \
                Comparing histograms before and after transformations such as \
                normalization or PCA rotation can reveal whether the transformation \
                behaved as expected. The `sample` option allows quick inspection of \
                very large files without reading every vector.\n\n\
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
        let dimension: usize = options.get("dimension")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

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

        let etype = match ElementType::from_path(&source_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        // Open reader and extract (count, dim, get_f64_fn) based on element type.
        let (count, dim, get_f64): (usize, usize, Box<dyn Fn(usize) -> Vec<f64> + Sync>) = match etype {
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

        if dimension >= dim {
            return error_result(
                format!("dimension {} out of range (max {})", dimension, dim - 1),
                start,
            );
        }

        let effective_count = sample.map(|s| s.min(count)).unwrap_or(count);

        // Thread count from governor (used by range-bin parallel accumulation)
        let _threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;

        // Extract values for the dimension with progress bar
        let pb = ctx.ui.bar_with_unit(effective_count as u64, "reading dimension values", "vectors");
        let mut values: Vec<f64> = Vec::with_capacity(effective_count);
        for i in 0..effective_count {
            let vec = get_f64(i);
            values.push(vec[dimension]);
            if (i + 1) % 100_000 == 0 { pb.set_position((i + 1) as u64); }
        }
        pb.finish();

        let stats = DimensionStats::compute(&values);

        if values.is_empty() {
            return error_result("no values to histogram".to_string(), start);
        }

        // Compute histogram bins
        let min = stats.min;
        let max = stats.max;

        if (max - min).abs() < f64::EPSILON {
            ctx.ui.log(&format!(
                "Dimension {}: all {} values = {:.6}",
                dimension, effective_count, min
            ));
            return CommandResult {
                status: Status::Ok,
                message: format!("histogram: all values identical ({:.6})", min),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Single-pass: build range bins AND discrete counts simultaneously.
        // If discrete values stay under the threshold, use them. Otherwise
        // fall back to range bins. No second pass over the data.
        let max_discrete = bins.min(100);
        let bin_width = (max - min) / bins as f64;
        let mut range_bins = vec![0usize; bins];
        let mut discrete_counts: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        let mut discrete_viable = true;

        for &v in &values {
            // Range bin — always accumulated
            let idx = ((v - min) / bin_width) as usize;
            range_bins[idx.min(bins - 1)] += 1;

            // Discrete tracking — accumulated until overflow
            if discrete_viable {
                let rounded = v.round();
                if (v - rounded).abs() > 1e-9 {
                    discrete_viable = false;
                } else {
                    *discrete_counts.entry(rounded as i64).or_insert(0) += 1;
                    if discrete_counts.len() > max_discrete {
                        discrete_viable = false;
                    }
                }
            }
        }

        let use_discrete = discrete_viable && !discrete_counts.is_empty();

        if use_discrete {
            // Discrete histogram: one bar per distinct value
            let max_count = *discrete_counts.values().max().unwrap_or(&1);
            let n_distinct = discrete_counts.len();

            ctx.ui.log(&format!(
                "Histogram: dimension {} ({} vectors, {} distinct values)",
                dimension, effective_count, n_distinct
            ));
            ctx.ui.log(&format!(
                "  Range: [{}, {}], Mean: {:.4}, StdDev: {:.4}",
                discrete_counts.keys().next().unwrap(),
                discrete_counts.keys().last().unwrap(),
                stats.mean, stats.std_dev
            ));
            ctx.ui.log("");

            let label_width = discrete_counts.keys().last()
                .map(|v| format!("{}", v).len())
                .unwrap_or(1)
                .max(discrete_counts.keys().next()
                    .map(|v| format!("{}", v).len())
                    .unwrap_or(1));

            for (&val, &count) in &discrete_counts {
                let bar_len = if max_count > 0 {
                    (count as f64 / max_count as f64 * width as f64) as usize
                } else {
                    0
                };
                let bar: String = "\u{2588}".repeat(bar_len);
                let pct = 100.0 * count as f64 / effective_count as f64;
                ctx.ui.log(&format!(
                    "  {:>w$} {:6} ({:5.1}%) |{}",
                    val, count, pct, bar, w = label_width
                ));
            }
        } else {
            // Range histogram: equal-width bins (already accumulated)
            let max_count = *range_bins.iter().max().unwrap_or(&1);

            ctx.ui.log(&format!(
                "Histogram: dimension {} ({} vectors)",
                dimension, effective_count
            ));
            ctx.ui.log(&format!(
                "  Range: [{:.4}, {:.4}], Mean: {:.4}, StdDev: {:.4}",
                min, max, stats.mean, stats.std_dev
            ));
            ctx.ui.log(&format!("  {} bins, bin width: {:.6}", bins, bin_width));
            ctx.ui.log("");

            for (i, &count) in range_bins.iter().enumerate() {
                let lo = min + i as f64 * bin_width;
                let hi = lo + bin_width;
                let bar_len = if max_count > 0 {
                    (count as f64 / max_count as f64 * width as f64) as usize
                } else {
                    0
                };
                let bar: String = "\u{2588}".repeat(bar_len);
                ctx.ui.log(&format!(
                    "  [{:8.4}, {:8.4}) {:6} |{}",
                    lo, hi, count, bar
                ));
            }
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
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".into()),
                description: "Dimension index to visualize (0-indexed)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "bins".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("40".to_string()),
                description: "Number of histogram bins".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "width".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("60".to_string()),
                description: "Width of histogram bars in characters".to_string(),
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
