// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: terminal-based flame graph rendering.
//!
//! Reads folded stack trace files (Brendan Gregg's format, as produced by
//! `perf script | stackcollapse-perf.pl`) and renders a text-based flame
//! graph using Unicode block characters and ANSI colors.
//!
//! Folded format: `function1;function2;function3 count\n` per line.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

/// Pipeline command: text flame graph from folded stack traces.
pub struct AnalyzeFlamegraphOp;

/// Create a boxed `AnalyzeFlamegraphOp`.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeFlamegraphOp)
}

/// A node in the stack trace tree.
#[derive(Debug, Default)]
struct FlameNode {
    /// Samples at this exact frame (self time).
    self_count: u64,
    /// Total samples including children.
    total_count: u64,
    /// Children keyed by frame name.
    children: HashMap<String, FlameNode>,
}

impl FlameNode {
    fn new() -> Self {
        FlameNode::default()
    }

    /// Insert a stack trace (slice of frames, bottom to top) with a sample count.
    fn insert(&mut self, frames: &[&str], count: u64) {
        self.total_count += count;
        if frames.is_empty() {
            self.self_count += count;
            return;
        }
        let child = self
            .children
            .entry(frames[0].to_string())
            .or_insert_with(FlameNode::new);
        child.insert(&frames[1..], count);
    }
}

/// Parse folded stack traces from a reader into a tree.
fn parse_folded_stacks(reader: impl BufRead) -> Result<FlameNode, String> {
    let mut root = FlameNode::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read error at line {}: {}", line_num + 1, e))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Format: "frame1;frame2;frame3 count"
        let (stack, count_str) = match line.rsplit_once(' ') {
            Some((s, c)) => (s, c),
            None => continue, // skip malformed lines
        };

        let count: u64 = match count_str.parse() {
            Ok(c) => c,
            Err(_) => continue, // skip malformed lines
        };

        let frames: Vec<&str> = stack.split(';').collect();
        root.insert(&frames, count);
    }

    Ok(root)
}

/// Choose an ANSI color based on the frame name.
///
/// Uses heuristics similar to Brendan Gregg's flamegraph.pl:
/// - Java/JVM frames: green tones
/// - kernel/system frames: orange/red
/// - native/C/C++: yellow
/// - everything else: warm palette
fn frame_color(frame: &str) -> &'static str {
    if frame.contains("java") || frame.contains("jdk") || frame.contains("javax") {
        "\x1b[38;5;82m" // green
    } else if frame.starts_with('[') || frame.contains("kernel") || frame.contains("vmlinux") {
        "\x1b[38;5;208m" // orange
    } else if frame.contains("::") || frame.starts_with("_Z") {
        "\x1b[38;5;226m" // yellow (C++)
    } else if frame.contains("libc") || frame.contains("libpthread") || frame.contains("syscall")
    {
        "\x1b[38;5;196m" // red (system)
    } else {
        "\x1b[38;5;215m" // warm default
    }
}

const ANSI_RESET: &str = "\x1b[0m";

/// Block characters for fractional column widths.
const BLOCKS: &[char] = &['█', '▓', '▒', '░'];

/// Render a flame graph from the tree to a string.
///
/// Each row represents a stack depth level. Wider bars indicate more samples.
fn render_flamegraph(
    root: &FlameNode,
    width: usize,
    _height: usize,
    min_samples: u64,
    output: &mut String,
) {
    if root.total_count == 0 {
        output.push_str("(no samples)\n");
        return;
    }

    output.push_str(&format!(
        "Flame Graph ({} total samples, width={})\n",
        root.total_count, width
    ));
    output.push_str(&format!("{}\n", "=".repeat(width)));

    // BFS-style rendering: collect spans at each depth level
    // Each span: (start_col, end_col, name, total_count)
    type Span = (usize, usize, String, u64);

    let current_level: Vec<Span> = vec![(0, width, "all".to_string(), root.total_count)];
    let root_total = root.total_count as f64;

    // Render the root level
    render_spans(&current_level, root_total, output);

    // Walk down the tree level by level
    let mut current_nodes: Vec<(usize, usize, &FlameNode)> = vec![(0, width, root)];

    loop {
        let mut next_spans: Vec<Span> = Vec::new();
        let mut next_nodes: Vec<(usize, usize, &FlameNode)> = Vec::new();

        for &(span_start, span_end, ref node) in &current_nodes {
            let span_width = span_end - span_start;
            if span_width == 0 || node.total_count == 0 {
                continue;
            }

            // Sort children by total_count descending for stable layout
            let mut children: Vec<(&String, &FlameNode)> = node.children.iter().collect();
            children.sort_by(|a, b| b.1.total_count.cmp(&a.1.total_count));

            let mut col = span_start;
            for (name, child) in children {
                if child.total_count < min_samples {
                    continue;
                }
                let child_width =
                    ((child.total_count as f64 / node.total_count as f64) * span_width as f64)
                        .round() as usize;
                let child_width = child_width.max(1).min(span_end - col);
                if child_width == 0 || col >= span_end {
                    break;
                }

                next_spans.push((col, col + child_width, name.clone(), child.total_count));
                next_nodes.push((col, col + child_width, child));
                col += child_width;
            }
        }

        if next_spans.is_empty() {
            break;
        }

        render_spans(&next_spans, root_total, output);
        current_nodes = next_nodes;
    }

    output.push_str(&format!("{}\n", "-".repeat(width)));
}

/// Render a single level of flame graph spans.
fn render_spans(spans: &[(usize, usize, String, u64)], root_total: f64, output: &mut String) {
    if spans.is_empty() {
        return;
    }

    // Find the total width from spans
    let max_col = spans.iter().map(|s| s.1).max().unwrap_or(0);
    let mut line = vec![' '; max_col];
    let mut colors: Vec<&str> = vec![ANSI_RESET; max_col];

    for (start, end, name, count) in spans {
        let w = end - start;
        if w == 0 {
            continue;
        }

        let color = frame_color(name);
        let pct = *count as f64 / root_total * 100.0;
        let label = if w > name.len() + 8 {
            format!("{} ({:.1}%)", name, pct)
        } else if w > name.len() + 1 {
            name.clone()
        } else if w > 3 {
            name.chars().take(w - 2).collect::<String>() + ".."
        } else {
            String::new()
        };

        // Fill block chars
        let block = BLOCKS[0];
        for col in *start..*end {
            line[col] = block;
            colors[col] = color;
        }

        // Overlay label
        for (i, ch) in label.chars().enumerate() {
            let col = start + 1 + i; // 1-col padding from left edge
            if col < *end {
                line[col] = ch;
            }
        }
    }

    // Emit the line with per-character coloring
    let mut prev_color = ANSI_RESET;
    for (i, &ch) in line.iter().enumerate() {
        if colors[i] != prev_color {
            output.push_str(colors[i]);
            prev_color = colors[i];
        }
        output.push(ch);
    }
    if prev_color != ANSI_RESET {
        output.push_str(ANSI_RESET);
    }
    output.push('\n');
}

impl CommandOp for AnalyzeFlamegraphOp {
    fn command_path(&self) -> &str {
        "analyze flamegraph"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let width: usize = options
            .get("width")
            .and_then(|s| s.parse().ok())
            .unwrap_or(120);
        let height: usize = options
            .get("height")
            .and_then(|s| s.parse().ok())
            .unwrap_or(40);
        let min_samples: u64 = options
            .get("min-samples")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        let input_path = resolve_path(input_str, &ctx.workspace);

        let file = match std::fs::File::open(&input_path) {
            Ok(f) => f,
            Err(e) => {
                return error_result(
                    format!("failed to open {}: {}", input_path.display(), e),
                    start,
                );
            }
        };

        let reader = std::io::BufReader::new(file);
        let root = match parse_folded_stacks(reader) {
            Ok(r) => r,
            Err(e) => return error_result(e, start),
        };

        let mut output = String::new();
        render_flamegraph(&root, width, height, min_samples, &mut output);
        eprint!("{}", output);

        CommandResult {
            status: Status::Ok,
            message: format!(
                "rendered flame graph from {} ({} total samples)",
                input_path.display(),
                root.total_count
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "input".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Input folded stack trace file".to_string(),
            },
            OptionDesc {
                name: "width".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("120".to_string()),
                description: "Output width in character columns".to_string(),
            },
            OptionDesc {
                name: "height".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("40".to_string()),
                description: "Maximum flame graph height in rows".to_string(),
            },
            OptionDesc {
                name: "min-samples".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("10".to_string()),
                description: "Minimum sample count to display a frame".to_string(),
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
    fn test_parse_folded_stacks() {
        let input = "main;foo;bar 100\nmain;foo;baz 50\nmain;qux 200\n";
        let root = parse_folded_stacks(input.as_bytes()).unwrap();
        assert_eq!(root.total_count, 350);
        assert_eq!(root.children.len(), 1); // "main"
        let main_node = &root.children["main"];
        assert_eq!(main_node.total_count, 350);
        assert_eq!(main_node.children.len(), 2); // "foo" and "qux"
    }

    #[test]
    fn test_parse_empty_input() {
        let root = parse_folded_stacks("".as_bytes()).unwrap();
        assert_eq!(root.total_count, 0);
    }

    #[test]
    fn test_parse_comments_and_blanks() {
        let input = "# comment\n\nmain;foo 10\n";
        let root = parse_folded_stacks(input.as_bytes()).unwrap();
        assert_eq!(root.total_count, 10);
    }

    #[test]
    fn test_render_flamegraph_basic() {
        let input = "main;foo;bar 100\nmain;foo;baz 50\nmain;qux 200\n";
        let root = parse_folded_stacks(input.as_bytes()).unwrap();
        let mut output = String::new();
        render_flamegraph(&root, 80, 20, 1, &mut output);
        // Should contain the frame names
        assert!(output.contains("all"));
        assert!(output.contains("main"));
        // Should contain block characters
        assert!(output.contains('█'));
    }

    #[test]
    fn test_flamegraph_command() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Write a sample folded stacks file
        let input_path = ws.join("stacks.folded");
        std::fs::write(
            &input_path,
            "main;process;compute 500\nmain;process;io_wait 300\nmain;gc 200\n",
        )
        .unwrap();

        let mut opts = Options::new();
        opts.set("input", input_path.to_string_lossy().to_string());
        opts.set("width", "60");
        opts.set("min-samples", "1");

        let mut op = AnalyzeFlamegraphOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("1000 total samples"));
    }

    #[test]
    fn test_flamegraph_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let mut opts = Options::new();
        opts.set("input", "nonexistent.folded");

        let mut op = AnalyzeFlamegraphOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }

    #[test]
    fn test_frame_color_heuristics() {
        assert!(frame_color("java.lang.Thread").contains("82")); // green
        assert!(frame_color("[kernel]").contains("208")); // orange
        assert!(frame_color("std::vector::push_back").contains("226")); // yellow
        assert!(frame_color("libc_start_main").contains("196")); // red
    }
}
