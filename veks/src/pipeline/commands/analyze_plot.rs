// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: terminal-based vector dimension plotting.
//!
//! Renders histograms and scatter plots using Unicode braille characters
//! (U+2800..U+28FF) for high-resolution terminal output. Each braille
//! character encodes a 2x4 dot matrix, giving 2x horizontal and 4x
//! vertical resolution relative to character cells.
//!
//! Supports multiple series overlay with ANSI 256-color coding,
//! dimension selection, configurable bin counts, and sampling.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::Rng;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;

/// Pipeline command: terminal braille plots.
pub struct AnalyzePlotOp;

/// Create a boxed `AnalyzePlotOp`.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzePlotOp)
}

/// Map an 8-bit dot pattern to the corresponding braille character.
///
/// Braille dot numbering (column-major):
/// ```text
///   0  3
///   1  4
///   2  5
///   6  7
/// ```
/// The offset from U+2800 is computed directly from the 8-bit mask.
fn braille_char(dots: u8) -> char {
    char::from_u32(0x2800 + dots as u32).unwrap_or(' ')
}

/// Cycle through a set of distinct ANSI 256-color codes for series.
fn ansi_color(idx: usize) -> &'static str {
    const COLORS: &[&str] = &[
        "\x1b[38;5;196m", // red
        "\x1b[38;5;46m",  // green
        "\x1b[38;5;33m",  // blue
        "\x1b[38;5;226m", // yellow
        "\x1b[38;5;201m", // magenta
        "\x1b[38;5;51m",  // cyan
        "\x1b[38;5;208m", // orange
        "\x1b[38;5;141m", // purple
    ];
    COLORS[idx % COLORS.len()]
}

const ANSI_RESET: &str = "\x1b[0m";

/// Render a histogram of `values` into a braille grid written to `output`.
///
/// The grid has `width` character columns and `height` character rows.
/// Each character cell covers 2 horizontal dots and 4 vertical dots.
fn render_histogram(
    values: &[f32],
    bins: usize,
    width: usize,
    height: usize,
    label: &str,
    color: &str,
    output: &mut String,
) {
    if values.is_empty() {
        return;
    }

    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = if (max - min).abs() < f32::EPSILON {
        1.0
    } else {
        max - min
    };

    // Bin the values
    let mut counts = vec![0u64; bins];
    for &v in values {
        let idx = ((v - min) / range * (bins as f32 - 1.0))
            .round()
            .max(0.0)
            .min((bins - 1) as f32) as usize;
        counts[idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1).max(&1);

    // Pixel grid: width*2 horizontal dots, height*4 vertical dots
    let px_w = width * 2;
    let px_h = height * 4;
    let mut grid = vec![vec![false; px_w]; px_h];

    // Map bins to pixel columns and counts to pixel rows
    for (bi, &count) in counts.iter().enumerate() {
        let col_start = (bi * px_w) / bins;
        let col_end = ((bi + 1) * px_w) / bins;
        let bar_height = ((count as f64 / max_count as f64) * px_h as f64).round() as usize;

        for col in col_start..col_end {
            for row in 0..bar_height {
                // Bottom-up: row 0 is bottom
                grid[px_h - 1 - row][col] = true;
            }
        }
    }

    // Render label
    output.push_str(&format!("{color}{label}{ANSI_RESET}\n"));

    // Render axis: max count
    output.push_str(&format!("{:>8} |", max_count));
    output.push('\n');

    // Render grid as braille characters
    for cy in 0..height {
        output.push_str("         |");
        for cx in 0..width {
            let mut dots: u8 = 0;
            // Braille dot numbering (column-major):
            // col0: bits 0,1,2,6 -> rows 0,1,2,3
            // col1: bits 3,4,5,7 -> rows 0,1,2,3
            for row_off in 0..4 {
                let gy = cy * 4 + row_off;
                if gy < px_h {
                    let gx0 = cx * 2;
                    let gx1 = cx * 2 + 1;
                    if gx0 < px_w && grid[gy][gx0] {
                        dots |= match row_off {
                            0 => 0x01,
                            1 => 0x02,
                            2 => 0x04,
                            3 => 0x40,
                            _ => 0,
                        };
                    }
                    if gx1 < px_w && grid[gy][gx1] {
                        dots |= match row_off {
                            0 => 0x08,
                            1 => 0x10,
                            2 => 0x20,
                            3 => 0x80,
                            _ => 0,
                        };
                    }
                }
            }
            output.push_str(color);
            output.push(braille_char(dots));
            output.push_str(ANSI_RESET);
        }
        output.push('\n');
    }

    // X-axis
    output.push_str(&format!("       0 +{}\n", "-".repeat(width)));
    output.push_str(&format!(
        "         {:<w$.4}{:>w$.4}\n",
        min,
        max,
        w = width
    ));
}

/// Render a scatter plot of dimension pairs using braille dots.
///
/// `dim_x` and `dim_y` select which dimensions to plot on each axis.
fn render_scatter(
    reader: &MmapVectorReader<f32>,
    count: usize,
    dim_x: usize,
    dim_y: usize,
    width: usize,
    height: usize,
    color: &str,
    output: &mut String,
) {
    let px_w = width * 2;
    let px_h = height * 4;
    let mut grid = vec![vec![false; px_w]; px_h];

    // Collect min/max for both dimensions
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for i in 0..count {
        if let Ok(v) = reader.get(i) {
            let x = v[dim_x];
            let y = v[dim_y];
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }
    }

    let range_x = if (max_x - min_x).abs() < f32::EPSILON { 1.0 } else { max_x - min_x };
    let range_y = if (max_y - min_y).abs() < f32::EPSILON { 1.0 } else { max_y - min_y };

    // Plot points
    for i in 0..count {
        if let Ok(v) = reader.get(i) {
            let px = ((v[dim_x] - min_x) / range_x * (px_w - 1) as f32)
                .round()
                .max(0.0)
                .min((px_w - 1) as f32) as usize;
            let py = ((v[dim_y] - min_y) / range_y * (px_h - 1) as f32)
                .round()
                .max(0.0)
                .min((px_h - 1) as f32) as usize;
            // Invert Y so larger values are at the top
            grid[px_h - 1 - py][px] = true;
        }
    }

    // Render label
    output.push_str(&format!(
        "{color}scatter: dim{dim_x} vs dim{dim_y}{ANSI_RESET}\n"
    ));
    output.push_str(&format!("{:>8.4} |", max_y));
    output.push('\n');

    // Render grid
    for cy in 0..height {
        output.push_str("         |");
        for cx in 0..width {
            let mut dots: u8 = 0;
            for row_off in 0..4 {
                let gy = cy * 4 + row_off;
                if gy < px_h {
                    let gx0 = cx * 2;
                    let gx1 = cx * 2 + 1;
                    if gx0 < px_w && grid[gy][gx0] {
                        dots |= match row_off {
                            0 => 0x01,
                            1 => 0x02,
                            2 => 0x04,
                            3 => 0x40,
                            _ => 0,
                        };
                    }
                    if gx1 < px_w && grid[gy][gx1] {
                        dots |= match row_off {
                            0 => 0x08,
                            1 => 0x10,
                            2 => 0x20,
                            3 => 0x80,
                            _ => 0,
                        };
                    }
                }
            }
            output.push_str(color);
            output.push(braille_char(dots));
            output.push_str(ANSI_RESET);
        }
        output.push('\n');
    }

    output.push_str(&format!("{:>8.4} +{}\n", min_y, "-".repeat(width)));
    output.push_str(&format!(
        "         {:<w$.4}{:>w$.4}\n",
        min_x,
        max_x,
        w = width
    ));
}

impl CommandOp for AnalyzePlotOp {
    fn command_path(&self) -> &str {
        "analyze plot"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate scatter/distribution plots for vectors".into(),
            body: format!(
                "# analyze plot\n\nGenerate scatter/distribution plots for vectors.\n\n## Description\n\nRenders histograms and scatter plots using Unicode braille characters for high-resolution terminal output. Supports multiple series overlay with ANSI 256-color coding, dimension selection, configurable bin counts, and sampling.\n\n## Options\n\n{}",
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

        let plot_type = options.get("type").unwrap_or("histogram");
        let dims_str = options.get("dimensions").unwrap_or("0");
        let width: usize = options
            .get("width")
            .and_then(|s| s.parse().ok())
            .unwrap_or(80);
        let height: usize = options
            .get("height")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let bins: Option<usize> = options.get("bins").and_then(|s| s.parse().ok());
        let sample: usize = options
            .get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000);
        let seed = rng::parse_seed(options.get("seed"));

        // Parse sources (semicolon-separated for multiple series)
        let sources: Vec<&str> = source_str.split(';').collect();

        // Parse dimensions
        let dims: Vec<usize> = dims_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if dims.is_empty() {
            return error_result("no valid dimensions specified".to_string(), start);
        }

        let mut output = String::new();

        for (si, source) in sources.iter().enumerate() {
            let source_path = resolve_path(source.trim(), &ctx.workspace);
            let reader = match MmapVectorReader::<f32>::open_fvec(&source_path) {
                Ok(r) => r,
                Err(e) => {
                    return error_result(
                        format!("failed to open {}: {}", source_path.display(), e),
                        start,
                    );
                }
            };

            let total = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
            let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);
            let effective = sample.min(total);
            let color = ansi_color(si);

            // Build sample indices
            let indices: Vec<usize> = if effective < total {
                let mut rng_inst = rng::seeded_rng(seed);
                let mut idx: Vec<usize> = (0..total).collect();
                for i in 0..effective {
                    let j = rng_inst.random_range(i..total);
                    idx.swap(i, j);
                }
                idx.truncate(effective);
                idx
            } else {
                (0..total).collect()
            };

            match plot_type {
                "histogram" => {
                    for &d in &dims {
                        if d >= dim {
                            return error_result(
                                format!("dimension {} out of range (max {})", d, dim - 1),
                                start,
                            );
                        }
                        let values: Vec<f32> = indices
                            .iter()
                            .filter_map(|&i| reader.get(i).ok().map(|v| v[d]))
                            .collect();

                        let num_bins = bins.unwrap_or_else(|| {
                            // Sturges' rule
                            ((values.len() as f64).log2().ceil() as usize + 1).max(10)
                        });

                        let label = format!(
                            "{} dim={} ({} values)",
                            source_path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy(),
                            d,
                            values.len()
                        );
                        render_histogram(&values, num_bins, width, height, &label, color, &mut output);
                        output.push('\n');
                    }
                }
                "scatter" => {
                    if dims.len() < 2 {
                        return error_result(
                            "scatter plot requires at least 2 dimensions (e.g. dimensions=0,1)"
                                .to_string(),
                            start,
                        );
                    }
                    for pair in dims.chunks(2) {
                        if pair.len() < 2 {
                            break;
                        }
                        let (dx, dy) = (pair[0], pair[1]);
                        if dx >= dim || dy >= dim {
                            return error_result(
                                format!("dimension out of range (max {})", dim - 1),
                                start,
                            );
                        }
                        render_scatter(&reader, effective, dx, dy, width, height, color, &mut output);
                        output.push('\n');
                    }
                }
                other => {
                    return error_result(
                        format!("unknown plot type '{}', use 'histogram' or 'scatter'", other),
                        start,
                    );
                }
            }
        }

        ctx.ui.emit(output);

        CommandResult {
            status: Status::Ok,
            message: format!(
                "plotted {} source(s), type={}",
                sources.len(),
                plot_type
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
                description: "Input fvec file(s), semicolon-separated for multiple series"
                    .to_string(),
            },
            OptionDesc {
                name: "dimensions".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Comma-separated dimension indices to plot".to_string(),
            },
            OptionDesc {
                name: "type".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("histogram".to_string()),
                description: "Plot type: histogram or scatter".to_string(),
            },
            OptionDesc {
                name: "width".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("80".to_string()),
                description: "Plot width in character columns".to_string(),
            },
            OptionDesc {
                name: "height".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("20".to_string()),
                description: "Plot height in character rows".to_string(),
            },
            OptionDesc {
                name: "bins".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Number of histogram bins (default: auto via Sturges' rule)"
                    .to_string(),
            },
            OptionDesc {
                name: "sample".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("10000".to_string()),
                description: "Max vectors to sample".to_string(),
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "PRNG seed for sampling".to_string(),
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
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_braille_char_empty() {
        assert_eq!(braille_char(0), '\u{2800}');
    }

    #[test]
    fn test_braille_char_full() {
        assert_eq!(braille_char(0xFF), '\u{28FF}');
    }

    #[test]
    fn test_histogram_renders_braille() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut output = String::new();
        render_histogram(&values, 10, 20, 5, "test", "", &mut output);
        // Output should contain braille characters
        assert!(output.chars().any(|c| ('\u{2800}'..='\u{28FF}').contains(&c)));
    }

    #[test]
    fn test_plot_command_histogram() {
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
        opts.set("dimensions", "0,3");
        opts.set("width", "40");
        opts.set("height", "10");

        let mut op = AnalyzePlotOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_plot_command_scatter() {
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
        opts.set("type", "scatter");
        opts.set("dimensions", "0,1");
        opts.set("width", "30");
        opts.set("height", "10");

        let mut op = AnalyzePlotOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_plot_invalid_dimension() {
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
        opts.set("dimensions", "10"); // out of range

        let mut op = AnalyzePlotOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }
}
