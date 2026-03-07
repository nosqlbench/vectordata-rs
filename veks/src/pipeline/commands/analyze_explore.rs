// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: interactive REPL for exploring vector files.
//!
//! Opens one or more vector files and provides an interactive command-line
//! interface for querying vectors, computing distances, finding neighbors,
//! and inspecting file contents.
//!
//! Equivalent to the Java `CMD_analyze_explore` command.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: interactive vector file explorer.
pub struct AnalyzeExploreOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeExploreOp)
}

impl CommandOp for AnalyzeExploreOp {
    fn command_path(&self) -> &str {
        "analyze explore"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Interactive exploration of vector file contents".into(),
            body: format!(
                "# analyze explore\n\nInteractive exploration of vector file contents.\n\n## Description\n\nOpens a vector file and provides an interactive REPL for querying vectors, computing distances, finding neighbors, and inspecting file contents. Supports scripted commands via the `commands` option.\n\n## Options\n\n{}",
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
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };

        let source = resolve_path(&source_str, &ctx.workspace);

        if !source.exists() {
            return error_result(format!("file not found: {}", source.display()), start);
        }

        let reader = match MmapVectorReader::<f32>::open_fvec(&source) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open: {}", e), start),
        };

        let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);

        ctx.ui.log(&format!("Exploring: {} ({} vectors, {} dims)", source.display(), count, dim));
        ctx.ui.log("Type 'help' for available commands, 'quit' to exit.");
        ctx.ui.log("");

        // Check if stdin is a TTY for interactive mode
        let is_interactive = atty::is(atty::Stream::Stdin);

        // If running in batch/script mode with no TTY, just print info and return
        if !is_interactive && !options.has("commands") {
            return CommandResult {
                status: Status::Ok,
                message: format!("{} vectors, {} dims", count, dim),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Process commands from the "commands" option (for scripted/test use)
        if let Some(cmds) = options.get("commands") {
            let mut output = Vec::new();
            for line in cmds.split(';') {
                let line = line.trim();
                if line.is_empty() || line == "quit" || line == "exit" {
                    break;
                }
                let result = execute_repl_command(line, &reader, count, dim);
                output.push(result);
            }
            return CommandResult {
                status: Status::Ok,
                message: output.join("\n"),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Interactive REPL loop
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            ctx.ui.emit("explore> ");
            let _ = stdout.flush();

            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break, // EOF
                Err(_) => break,
                _ => {}
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line == "quit" || line == "exit" {
                break;
            }

            let result = execute_repl_command(line, &reader, count, dim);
            ctx.ui.emitln(result);
        }

        CommandResult {
            status: Status::Ok,
            message: "explore session ended".to_string(),
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
                description: "Vector file to explore".to_string(),
            },
            OptionDesc {
                name: "commands".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Semicolon-separated commands for non-interactive use".to_string(),
            },
        ]
    }
}

/// Execute a single REPL command and return the output as a string.
fn execute_repl_command(
    line: &str,
    reader: &MmapVectorReader<f32>,
    count: usize,
    dim: usize,
) -> String {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return String::new();
    }

    match parts[0] {
        "help" | "?" => help_text(),

        "info" => format!("vectors: {}, dimensions: {}", count, dim),

        "get" | "vec" => {
            if parts.len() < 2 {
                return "usage: get <index>".to_string();
            }
            match parts[1].parse::<usize>() {
                Ok(idx) if idx < count => match reader.get(idx) {
                    Ok(vec) => format_vector(idx, vec.as_slice(), dim),
                    Err(e) => format!("error reading vector {}: {}", idx, e),
                },
                Ok(idx) => format!("index {} out of range (0..{})", idx, count),
                Err(_) => "invalid index".to_string(),
            }
        }

        "range" => {
            if parts.len() < 3 {
                return "usage: range <start> <end>".to_string();
            }
            let start = parts[1].parse::<usize>().unwrap_or(0);
            let end = parts[2].parse::<usize>().unwrap_or(0).min(count);
            if start >= end || start >= count {
                return "invalid range".to_string();
            }
            let limit = (end - start).min(20); // cap display
            let mut lines = Vec::new();
            for i in start..start + limit {
                match reader.get(i) {
                    Ok(vec) => lines.push(format_vector(i, vec.as_slice(), dim)),
                    Err(e) => lines.push(format!("[{}] error: {}", i, e)),
                }
            }
            if end - start > limit {
                lines.push(format!("... ({} more)", end - start - limit));
            }
            lines.join("\n")
        }

        "dist" | "distance" => {
            if parts.len() < 4 {
                return "usage: dist <metric> <i> <j>  (metrics: l2, cosine, dot)".to_string();
            }
            let metric = parts[1];
            let i = parts[2].parse::<usize>().unwrap_or(usize::MAX);
            let j = parts[3].parse::<usize>().unwrap_or(usize::MAX);
            if i >= count || j >= count {
                return format!("index out of range (0..{})", count);
            }
            match (reader.get(i), reader.get(j)) {
                (Ok(a), Ok(b)) => {
                    let d = compute_distance(metric, a.as_ref(), b.as_ref());
                    match d {
                        Some(val) => format!("{}([{}], [{}]) = {:.6}", metric, i, j, val),
                        None => format!("unknown metric: {} (use l2, cosine, dot)", metric),
                    }
                }
                _ => "error reading vectors".to_string(),
            }
        }

        "stats" => {
            let sample = if parts.len() > 1 {
                parts[1].parse::<usize>().unwrap_or(1000)
            } else {
                1000
            };
            let effective = sample.min(count);
            compute_stats(reader, effective, dim)
        }

        "norm" | "norms" => {
            if parts.len() < 2 {
                return "usage: norm <index> [count]".to_string();
            }
            let start_idx = parts[1].parse::<usize>().unwrap_or(0);
            let n = if parts.len() > 2 {
                parts[2].parse::<usize>().unwrap_or(1)
            } else {
                1
            };
            let mut lines = Vec::new();
            for i in start_idx..start_idx.saturating_add(n).min(count) {
                if let Ok(vec) = reader.get(i) {
                    let l2: f64 = vec.as_slice().iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    lines.push(format!("[{}] L2 norm = {:.6}", i, l2));
                }
            }
            lines.join("\n")
        }

        "head" => {
            let n = if parts.len() > 1 {
                parts[1].parse::<usize>().unwrap_or(5)
            } else {
                5
            };
            let n = n.min(count);
            let mut lines = Vec::new();
            for i in 0..n {
                match reader.get(i) {
                    Ok(vec) => lines.push(format_vector(i, vec.as_slice(), dim)),
                    Err(e) => lines.push(format!("[{}] error: {}", i, e)),
                }
            }
            lines.join("\n")
        }

        "tail" => {
            let n = if parts.len() > 1 {
                parts[1].parse::<usize>().unwrap_or(5)
            } else {
                5
            };
            let n = n.min(count);
            let start = count - n;
            let mut lines = Vec::new();
            for i in start..count {
                match reader.get(i) {
                    Ok(vec) => lines.push(format_vector(i, vec.as_slice(), dim)),
                    Err(e) => lines.push(format!("[{}] error: {}", i, e)),
                }
            }
            lines.join("\n")
        }

        _ => format!("unknown command: '{}'. Type 'help' for available commands.", parts[0]),
    }
}

fn help_text() -> String {
    [
        "Available commands:",
        "  info              — show file info (count, dims)",
        "  get <index>       — show vector at index",
        "  range <start> <end> — show vectors in range",
        "  head [n]          — show first n vectors (default 5)",
        "  tail [n]          — show last n vectors (default 5)",
        "  dist <metric> <i> <j> — distance between vectors (l2, cosine, dot)",
        "  norm <index> [n]  — show L2 norms",
        "  stats [sample]    — compute per-dimension statistics",
        "  help              — show this help",
        "  quit              — exit",
    ]
    .join("\n")
}

fn format_vector(idx: usize, vec: &[f32], dim: usize) -> String {
    if dim <= 8 {
        let vals: Vec<String> = vec.iter().map(|v| format!("{:.4}", v)).collect();
        format!("[{}] [{}]", idx, vals.join(", "))
    } else {
        let first: Vec<String> = vec[..4].iter().map(|v| format!("{:.4}", v)).collect();
        let last: Vec<String> = vec[dim - 2..].iter().map(|v| format!("{:.4}", v)).collect();
        format!(
            "[{}] [{}, ... , {}] ({} dims)",
            idx,
            first.join(", "),
            last.join(", "),
            dim
        )
    }
}

fn compute_distance(metric: &str, a: &[f32], b: &[f32]) -> Option<f64> {
    match metric {
        "l2" | "euclidean" => {
            let sum: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = (x as f64) - (y as f64);
                    d * d
                })
                .sum();
            Some(sum.sqrt())
        }
        "cosine" => {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                let xf = x as f64;
                let yf = y as f64;
                dot += xf * yf;
                na += xf * xf;
                nb += yf * yf;
            }
            let denom = na.sqrt() * nb.sqrt();
            if denom < 1e-10 {
                Some(1.0)
            } else {
                Some(1.0 - dot / denom)
            }
        }
        "dot" | "inner" => {
            let dot: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x as f64) * (y as f64))
                .sum();
            Some(dot)
        }
        _ => None,
    }
}

fn compute_stats(reader: &MmapVectorReader<f32>, sample: usize, dim: usize) -> String {
    let mut mins = vec![f64::INFINITY; dim];
    let mut maxs = vec![f64::NEG_INFINITY; dim];
    let mut sums = vec![0.0f64; dim];
    let mut sq_sums = vec![0.0f64; dim];

    let mut actual = 0usize;
    for i in 0..sample {
        if let Ok(vec) = reader.get(i) {
            actual += 1;
            for (d, &v) in vec.as_slice().iter().enumerate() {
                let vf = v as f64;
                if vf < mins[d] { mins[d] = vf; }
                if vf > maxs[d] { maxs[d] = vf; }
                sums[d] += vf;
                sq_sums[d] += vf * vf;
            }
        }
    }

    if actual == 0 {
        return "no vectors to analyze".to_string();
    }

    let n = actual as f64;
    let mut lines = vec![format!("Statistics (sample of {}):", actual)];
    let show_dims = dim.min(10);
    for d in 0..show_dims {
        let mean = sums[d] / n;
        let var = (sq_sums[d] / n) - mean * mean;
        let stddev = var.max(0.0).sqrt();
        lines.push(format!(
            "  dim[{}]: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            d, mins[d], maxs[d], mean, stddev
        ));
    }
    if dim > show_dims {
        lines.push(format!("  ... ({} more dimensions)", dim - show_dims));
    }

    lines.join("\n")
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
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
    fn test_explore_info() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate test vectors
        let fvec = ws.join("test.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", fvec.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "100");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("commands", "info;head 3;stats 50");
        let mut op = AnalyzeExploreOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("vectors: 100"));
        assert!(result.message.contains("dimensions: 4"));
    }

    #[test]
    fn test_explore_distance() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let fvec = ws.join("test.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", fvec.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "10");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("commands", "dist l2 0 1;dist cosine 0 0");
        let mut op = AnalyzeExploreOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("l2"));
        // Distance of a vector to itself should be 0
        assert!(result.message.contains("cosine([0], [0]) = 0.000000"));
    }

    #[test]
    fn test_explore_norms() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let fvec = ws.join("test.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", fvec.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "10");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", fvec.to_string_lossy().to_string());
        opts.set("commands", "norm 0 3");
        let mut op = AnalyzeExploreOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("L2 norm"));
    }
}
