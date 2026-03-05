// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: slice and display vector data.
//!
//! Extracts a subset of vectors (by ordinal range) and a subset of dimensions
//! (by component range) from an xvec file, printing in text, CSV, TSV, or JSON
//! format.
//!
//! Equivalent to the Java `CMD_analyze_slice` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::io::MmapVectorReader;
use vectordata::io::VectorReader;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

/// Pipeline command: slice vector data.
pub struct AnalyzeSliceOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeSliceOp)
}

/// Parse a range string into (start_inclusive, end_exclusive).
///
/// Supported forms:
/// - `n` → `[0, n)`
/// - `m..n` → `[m, n+1)` (inclusive both ends)
/// - `[m,n)` → `[m, n)`
/// - `[m,n]` → `[m, n+1)`
/// - `(m,n)` → `[m+1, n)`
/// - `(m,n]` → `[m+1, n+1)`
/// - `[n]` → `[n, n+1)` (single element)
fn parse_range(s: &str, max: usize) -> Result<(usize, usize), String> {
    let s = s.trim();

    // Single element: [n]
    if s.starts_with('[') && s.ends_with(']') && !s.contains(',') {
        let inner = &s[1..s.len() - 1];
        let n: usize = inner.parse().map_err(|_| format!("invalid range: '{}'", s))?;
        return Ok((n, (n + 1).min(max)));
    }

    // Interval notation: [m,n) or (m,n] etc.
    if (s.starts_with('[') || s.starts_with('(')) && (s.ends_with(')') || s.ends_with(']')) {
        let start_inclusive = s.starts_with('[');
        let end_inclusive = s.ends_with(']');
        let inner = &s[1..s.len() - 1];
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() != 2 {
            return Err(format!("invalid range: '{}'", s));
        }
        let mut start: usize = parts[0].trim().parse().map_err(|_| format!("invalid range: '{}'", s))?;
        let mut end: usize = parts[1].trim().parse().map_err(|_| format!("invalid range: '{}'", s))?;
        if !start_inclusive {
            start += 1;
        }
        if end_inclusive {
            end += 1;
        }
        return Ok((start.min(max), end.min(max)));
    }

    // m..n (inclusive both ends)
    if let Some(idx) = s.find("..") {
        let left: usize = s[..idx].trim().parse().map_err(|_| format!("invalid range: '{}'", s))?;
        let right: usize = s[idx + 2..].trim().parse().map_err(|_| format!("invalid range: '{}'", s))?;
        return Ok((left.min(max), (right + 1).min(max)));
    }

    // Plain number: n → [0, n)
    let n: usize = s.parse().map_err(|_| format!("invalid range: '{}'", s))?;
    Ok((0, n.min(max)))
}

impl CommandOp for AnalyzeSliceOp {
    fn command_path(&self) -> &str {
        "analyze slice"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let format = options.get("format").unwrap_or("text").to_lowercase();
        let max_vectors: usize = options
            .get("max-vectors")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        let source_path = resolve_path(source_str, &ctx.workspace);

        let ext = source_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Dispatch based on format
        match ext.as_str() {
            "fvec" => self.slice_fvec(&source_path, options, &format, max_vectors, start),
            "ivec" => self.slice_ivec(&source_path, options, &format, max_vectors, start),
            _ => error_result(
                format!("unsupported format for slicing: '.{}' (supported: fvec, ivec)", ext),
                start,
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Input vector file (fvec or ivec)".to_string(),
            },
            OptionDesc {
                name: "ordinal-range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Row range: n, m..n, [m,n), [n], etc.".to_string(),
            },
            OptionDesc {
                name: "component-range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Dimension range within each vector".to_string(),
            },
            OptionDesc {
                name: "format".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("text".to_string()),
                description: "Output format: text, csv, tsv, json".to_string(),
            },
            OptionDesc {
                name: "max-vectors".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("100".to_string()),
                description: "Max vectors to display in text mode".to_string(),
            },
        ]
    }
}

impl AnalyzeSliceOp {
    fn slice_fvec(
        &self,
        path: &Path,
        options: &Options,
        format: &str,
        max_vectors: usize,
        start: Instant,
    ) -> CommandResult {
        let reader = match MmapVectorReader::<f32>::open_fvec(path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", path.display(), e), start),
        };

        let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);

        let (ord_start, ord_end) = match options.get("ordinal-range") {
            Some(r) => match parse_range(r, count) {
                Ok(range) => range,
                Err(e) => return error_result(e, start),
            },
            None => (0, count),
        };

        let (comp_start, comp_end) = match options.get("component-range") {
            Some(r) => match parse_range(r, dim) {
                Ok(range) => range,
                Err(e) => return error_result(e, start),
            },
            None => (0, dim),
        };

        let total = ord_end.saturating_sub(ord_start);
        let mut output_lines = Vec::new();

        if format == "json" {
            output_lines.push("[".to_string());
        }

        let display_count = if format == "text" { max_vectors.min(total) } else { total };

        for (i, ord) in (ord_start..ord_end).enumerate() {
            if i >= display_count {
                break;
            }

            let vec = match reader.get(ord) {
                Ok(v) => v,
                Err(e) => {
                    return error_result(format!("read error at ordinal {}: {}", ord, e), start);
                }
            };

            let slice = &vec[comp_start..comp_end.min(vec.len())];
            let line = format_slice_f32(ord, slice, format, i + 1 < display_count);
            output_lines.push(line);
        }

        if format == "text" && total > display_count {
            output_lines.push(format!("... {} more vectors", total - display_count));
        }

        if format == "json" {
            output_lines.push("]".to_string());
        }

        for line in &output_lines {
            eprintln!("{}", line);
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "sliced {} vectors (ordinals {}..{}, components {}..{})",
                total.min(display_count),
                ord_start,
                ord_end,
                comp_start,
                comp_end
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn slice_ivec(
        &self,
        path: &Path,
        options: &Options,
        format: &str,
        max_vectors: usize,
        start: Instant,
    ) -> CommandResult {
        let reader = match MmapVectorReader::<i32>::open_ivec(path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", path.display(), e), start),
        };

        let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
        let dim = <MmapVectorReader<i32> as VectorReader<i32>>::dim(&reader);

        let (ord_start, ord_end) = match options.get("ordinal-range") {
            Some(r) => match parse_range(r, count) {
                Ok(range) => range,
                Err(e) => return error_result(e, start),
            },
            None => (0, count),
        };

        let (comp_start, comp_end) = match options.get("component-range") {
            Some(r) => match parse_range(r, dim) {
                Ok(range) => range,
                Err(e) => return error_result(e, start),
            },
            None => (0, dim),
        };

        let total = ord_end.saturating_sub(ord_start);
        let mut output_lines = Vec::new();

        if format == "json" {
            output_lines.push("[".to_string());
        }

        let display_count = if format == "text" { max_vectors.min(total) } else { total };

        for (i, ord) in (ord_start..ord_end).enumerate() {
            if i >= display_count {
                break;
            }

            let vec = match reader.get(ord) {
                Ok(v) => v,
                Err(e) => {
                    return error_result(format!("read error at ordinal {}: {}", ord, e), start);
                }
            };

            let slice = &vec[comp_start..comp_end.min(vec.len())];
            let line = format_slice_i32(ord, slice, format, i + 1 < display_count);
            output_lines.push(line);
        }

        if format == "text" && total > display_count {
            output_lines.push(format!("... {} more vectors", total - display_count));
        }

        if format == "json" {
            output_lines.push("]".to_string());
        }

        for line in &output_lines {
            eprintln!("{}", line);
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "sliced {} vectors (ordinals {}..{}, components {}..{})",
                total.min(display_count),
                ord_start,
                ord_end,
                comp_start,
                comp_end
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }
}

fn format_slice_f32(ordinal: usize, slice: &[f32], format: &str, has_more: bool) -> String {
    match format {
        "csv" => {
            let vals: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
            format!("{},{}", ordinal, vals.join(","))
        }
        "tsv" => {
            let vals: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
            format!("{}\t{}", ordinal, vals.join("\t"))
        }
        "json" => {
            let vals: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
            let comma = if has_more { "," } else { "" };
            format!("  {{\"ordinal\": {}, \"values\": [{}]}}{}", ordinal, vals.join(", "), comma)
        }
        _ => {
            // text
            let vals: Vec<String> = slice.iter().map(|v| format!("{:.6}", v)).collect();
            format!("[{}] {}", ordinal, vals.join(" "))
        }
    }
}

fn format_slice_i32(ordinal: usize, slice: &[i32], format: &str, has_more: bool) -> String {
    match format {
        "csv" => {
            let vals: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
            format!("{},{}", ordinal, vals.join(","))
        }
        "tsv" => {
            let vals: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
            format!("{}\t{}", ordinal, vals.join("\t"))
        }
        "json" => {
            let vals: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
            let comma = if has_more { "," } else { "" };
            format!("  {{\"ordinal\": {}, \"values\": [{}]}}{}", ordinal, vals.join(", "), comma)
        }
        _ => {
            let vals: Vec<String> = slice.iter().map(|v| format!("{}", v)).collect();
            format!("[{}] {}", ordinal, vals.join(" "))
        }
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
    use std::io::Write;

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

    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    fn write_ivec(path: &Path, vectors: &[Vec<i32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            let dim = v.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn test_parse_range_plain() {
        assert_eq!(parse_range("10", 100).unwrap(), (0, 10));
        assert_eq!(parse_range("200", 100).unwrap(), (0, 100));
    }

    #[test]
    fn test_parse_range_dotdot() {
        assert_eq!(parse_range("2..5", 100).unwrap(), (2, 6));
    }

    #[test]
    fn test_parse_range_interval() {
        assert_eq!(parse_range("[2,5)", 100).unwrap(), (2, 5));
        assert_eq!(parse_range("[2,5]", 100).unwrap(), (2, 6));
        assert_eq!(parse_range("(2,5)", 100).unwrap(), (3, 5));
        assert_eq!(parse_range("(2,5]", 100).unwrap(), (3, 6));
    }

    #[test]
    fn test_parse_range_single() {
        assert_eq!(parse_range("[3]", 100).unwrap(), (3, 4));
    }

    #[test]
    fn test_slice_fvec_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        write_fvec(
            &input,
            &[
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
        );

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());
        opts.set("ordinal-range", "[1,2]");
        opts.set("component-range", "[0,1]");

        let mut op = AnalyzeSliceOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("2 vectors"));
    }

    #[test]
    fn test_slice_ivec_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.ivec");
        write_ivec(&input, &[vec![10, 20, 30], vec![40, 50, 60]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());

        let mut op = AnalyzeSliceOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("2 vectors"));
    }

    #[test]
    fn test_slice_json_format() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("test.fvec");
        write_fvec(&input, &[vec![1.0, 2.0], vec![3.0, 4.0]]);

        let mut opts = Options::new();
        opts.set("source", input.to_string_lossy().to_string());
        opts.set("format", "json");

        let mut op = AnalyzeSliceOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }
}
