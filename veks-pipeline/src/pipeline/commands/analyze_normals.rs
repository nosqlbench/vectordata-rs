// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: measure vector normalization precision.
//!
//! Computes the L2 norm of sampled vectors entirely in f64 precision,
//! then measures the absolute deviation from 1.0 (the "normal epsilon").
//! Produces gaussian distribution statistics of the epsilon values as
//! pipeline variables.

use std::path::Path;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
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

pub struct AnalyzeNormalsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeNormalsOp)
}

impl CommandOp for AnalyzeNormalsOp {
    fn command_path(&self) -> &str {
        "analyze measure-normals"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Measure vector normalization precision at f64".into(),
            body: format!(r#"# analyze measure-normals

Measure the L2 normalization precision of a vector file.

## Description

Samples vectors randomly and computes each vector's L2 norm entirely
in f64 (double) precision. The **normal epsilon** for each vector is
the absolute difference between its computed norm and 1.0:

    epsilon = |norm_f64 - 1.0|

The epsilon values are summarized as a gaussian distribution. The
following variables are written to `variables.yaml`:

- `mean_normal_epsilon` — average deviation from unit norm
- `min_normal_epsilon` — smallest deviation (best-normalized vector)
- `max_normal_epsilon` — largest deviation (worst-normalized vector)
- `stddev_normal_epsilon` — standard deviation of the epsilon distribution
- `median_normal_epsilon` — median deviation
- `is_normalized` — true if mean epsilon < C × ε_mach(element_type) × √dim

The threshold adapts to element precision and dimensionality using the
probabilistic rounding-error bound from Higham & Mary (2019). C = 10
provides headroom above the expected error floor. See SRD §18.3.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "input".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Vector file to analyze (fvec, mvec, dvec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "sample".into(),
                type_name: "int".into(),
                required: false,
                default: Some("10000".into()),
                description: "Number of vectors to sample".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "seed".into(),
                type_name: "int".into(),
                required: false,
                default: Some("42".into()),
                description: "Random seed for sampling".into(),
                role: OptionRole::Config,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let sample_count: usize = match options.parse_or("sample", 10_000usize) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let seed: u64 = match options.parse_or("seed", 42u64) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);

        // Detect element type and open reader
        let etype = match ElementType::from_path(&input_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };
        if !etype.is_float() {
            return error_result(
                format!("normalization analysis requires float vectors, got {:?}", etype),
                start,
            );
        }

        // Open reader and get count/dim
        let (count, dim, get_f64): (usize, usize, Box<dyn Fn(usize) -> Vec<f64>>) = match etype {
            ElementType::F32 => {
                let r = match MmapVectorReader::<f32>::open_fvec(&input_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open: {}", e), start),
                };
                let c = VectorReader::<f32>::count(&r);
                let d = VectorReader::<f32>::dim(&r);
                (c, d, Box::new(move |i| {
                    r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()
                }))
            }
            ElementType::F16 => {
                let r = match MmapVectorReader::<half::f16>::open_mvec(&input_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open: {}", e), start),
                };
                let c = VectorReader::<half::f16>::count(&r);
                let d = VectorReader::<half::f16>::dim(&r);
                (c, d, Box::new(move |i| {
                    r.get(i).unwrap_or_default().iter().map(|v| v.to_f64()).collect()
                }))
            }
            ElementType::F64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&input_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open: {}", e), start),
                };
                let c = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (c, d, Box::new(move |i| {
                    r.get(i).unwrap_or_default()
                }))
            }
            _ => return error_result("unsupported element type for normalization analysis", start),
        };

        if count == 0 || dim == 0 {
            return error_result("empty vector file", start);
        }

        let actual_sample = sample_count.min(count);
        ctx.ui.log(&format!(
            "measure-normals: {} vectors (dim={}), sampling {} at f64 precision",
            count, dim, actual_sample,
        ));

        // Generate random sample indices using LCG
        let mut rng = seed;
        let mut sample_indices: Vec<usize> = Vec::with_capacity(actual_sample);
        for _ in 0..actual_sample {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            sample_indices.push((rng as usize) % count);
        }
        sample_indices.sort();
        sample_indices.dedup();
        let num_samples = sample_indices.len();

        // Compute epsilons entirely in f64
        let pb = ctx.ui.bar_with_unit(num_samples as u64, "computing normals", "vectors");
        let mut epsilons: Vec<f64> = Vec::with_capacity(num_samples);

        for (i, &idx) in sample_indices.iter().enumerate() {
            let vec = get_f64(idx);
            // Compute L2 norm in f64: sqrt(sum(v_i^2))
            let sum_sq: f64 = vec.iter().map(|&v| v * v).sum();
            let norm = sum_sq.sqrt();
            let epsilon = (norm - 1.0_f64).abs();
            epsilons.push(epsilon);
            if (i + 1) % 1000 == 0 {
                pb.set_position((i + 1) as u64);
            }
        }
        pb.finish();

        if epsilons.is_empty() {
            return error_result("no valid samples", start);
        }

        // Compute distribution statistics
        let n = epsilons.len() as f64;
        let mean = epsilons.iter().sum::<f64>() / n;
        let min = epsilons.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = epsilons.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = epsilons.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let stddev = variance.sqrt();

        // Median
        let mut sorted = epsilons.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Precision-aware normalization threshold (SRD §18.3).
        let eps_mach = etype.machine_epsilon().unwrap_or(1e-7);
        let threshold = etype.normalization_threshold(dim)
            .unwrap_or(10.0 * 1e-7 * (dim as f64).sqrt());
        let is_normalized = mean < threshold;

        ctx.ui.log(&format!(
            "  mean_epsilon={:.2e}  stddev={:.2e}  min={:.2e}  max={:.2e}  median={:.2e}",
            mean, stddev, min, max, median,
        ));
        ctx.ui.log(&format!(
            "  threshold={:.2e} (C=10 × eps={:.2e} × √dim={})",
            threshold, eps_mach, dim,
        ));
        ctx.ui.log(&format!(
            "  is_normalized={} (mean_ε={:.2e} {} threshold)",
            is_normalized, mean, if is_normalized { "<" } else { ">=" },
        ));

        // Write variables
        let vars = [
            ("mean_normal_epsilon", format!("{:.15e}", mean)),
            ("min_normal_epsilon", format!("{:.15e}", min)),
            ("max_normal_epsilon", format!("{:.15e}", max)),
            ("stddev_normal_epsilon", format!("{:.15e}", stddev)),
            ("median_normal_epsilon", format!("{:.15e}", median)),
            ("normal_threshold", format!("{:.15e}", threshold)),
            ("is_normalized", is_normalized.to_string()),
        ];
        for (name, value) in &vars {
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, name, value);
            ctx.defaults.insert(name.to_string(), value.clone());
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "normal epsilon: mean={:.2e} stddev={:.2e} ({} samples) is_normalized={}",
                mean, stddev, num_samples, is_normalized,
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }
}
