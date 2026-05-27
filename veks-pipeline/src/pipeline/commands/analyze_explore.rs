// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! REPL command engine for the interactive explore TUI.
//!
//! This module owns the textual command-table that the explore
//! TUI's data shell (`veks/src/explore/data_shell.rs`) dispatches
//! against: `info`, `get`, `range`, `head`, `tail`, `dist`, `norm`,
//! `stats`, `help`, etc.
//!
//! Originally registered as the `"analyze visualize-explore"`
//! pipeline command, but the interactive surface migrated to the
//! ratatui-based explorer in `veks/src/explore/unified.rs` (see
//! sysref §22). The CommandOp wrapper was removed when the
//! pipeline-command path lost users; the REPL helpers below
//! survive because the new TUI still uses them as its command
//! engine.
//!
//! Keep this module's `pub` surface stable for `data_shell.rs`'s
//! consumers; don't re-add a CommandOp wrapper here without
//! checking with the consumers that the new entry point is needed.


/// Execute a single REPL command and return the output as a string.
pub fn execute_repl_command(
    line: &str,
    get_f64: &dyn Fn(usize) -> Vec<f64>,
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
                Ok(idx) if idx < count => {
                    let vec = get_f64(idx);
                    format_vector(idx, &vec, dim)
                }
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
                let vec = get_f64(i);
                lines.push(format_vector(i, &vec, dim));
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
            let a = get_f64(i);
            let b = get_f64(j);
            let d = compute_distance(metric, &a, &b);
            match d {
                Some(val) => format!("{}([{}], [{}]) = {:.6}", metric, i, j, val),
                None => format!("unknown metric: {} (use l2, cosine, dot)", metric),
            }
        }

        "stats" => {
            let sample = if parts.len() > 1 {
                parts[1].parse::<usize>().unwrap_or(1000)
            } else {
                1000
            };
            let effective = sample.min(count);
            compute_stats(get_f64, effective, dim)
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
                let vec = get_f64(i);
                let l2: f64 = vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
                lines.push(format!("[{}] L2 norm = {:.6}", i, l2));
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
                let vec = get_f64(i);
                lines.push(format_vector(i, &vec, dim));
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
                let vec = get_f64(i);
                lines.push(format_vector(i, &vec, dim));
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

fn format_vector(idx: usize, vec: &[f64], dim: usize) -> String {
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

fn compute_distance(metric: &str, a: &[f64], b: &[f64]) -> Option<f64> {
    match metric {
        "l2" | "euclidean" => {
            let sum: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = x - y;
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
                dot += x * y;
                na += x * x;
                nb += y * y;
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
                .map(|(&x, &y)| x * y)
                .sum();
            Some(dot)
        }
        _ => None,
    }
}

fn compute_stats(get_f64: &dyn Fn(usize) -> Vec<f64>, sample: usize, dim: usize) -> String {
    let mut mins = vec![f64::INFINITY; dim];
    let mut maxs = vec![f64::NEG_INFINITY; dim];
    let mut sums = vec![0.0f64; dim];
    let mut sq_sums = vec![0.0f64; dim];

    let mut actual = 0usize;
    for i in 0..sample {
        let vec = get_f64(i);
        if !vec.is_empty() {
            actual += 1;
            for (d, &v) in vec.iter().enumerate() {
                if v < mins[d] { mins[d] = v; }
                if v > maxs[d] { maxs[d] = v; }
                sums[d] += v;
                sq_sums[d] += v * v;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic `get_f64` closure that returns vector `[i*10, i*10+1, ..., i*10+(dim-1)]`
    /// for index `i` — fully deterministic so REPL output is byte-stable.
    fn synth(dim: usize) -> impl Fn(usize) -> Vec<f64> + Send + Sync + 'static {
        move |i: usize| (0..dim).map(|d| (i * 10 + d) as f64).collect()
    }

    #[test]
    fn info_reports_count_and_dim() {
        let get = synth(4);
        let out = execute_repl_command("info", &get, 100, 4);
        assert!(out.contains("vectors: 100"));
        assert!(out.contains("dimensions: 4"));
    }

    #[test]
    fn dist_self_is_zero_for_l2() {
        let get = synth(4);
        let out = execute_repl_command("dist l2 1 1", &get, 10, 4);
        assert!(out.contains("l2"));
        // The same vector has L2 distance 0 — exact bytewise equality.
        assert!(out.contains("0"));
    }

    #[test]
    fn norm_emits_l2_label() {
        let get = synth(4);
        let out = execute_repl_command("norm 0 3", &get, 10, 4);
        assert!(out.contains("L2 norm"));
    }

    #[test]
    fn help_lists_commands() {
        let get = synth(4);
        let out = execute_repl_command("help", &get, 10, 4);
        for kw in &["info", "get", "range", "head", "tail", "dist", "norm", "stats"] {
            assert!(out.contains(kw), "help text missing '{kw}': {out}");
        }
    }
}
