// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: select and display a single vector by ordinal.
//!
//! Retrieves a vector at a specific 0-based index from a vector file and
//! prints its type and values.
//!
//! Equivalent to the Java `CMD_analyze_select` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: select a vector by ordinal.
pub struct AnalyzeSelectOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeSelectOp)
}

impl CommandOp for AnalyzeSelectOp {
    fn command_path(&self) -> &str {
        "analyze select"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Select and print specific vectors by ordinal".into(),
            body: format!(
                "# analyze select\n\nSelect and print specific vectors by ordinal.\n\n## Description\n\nRetrieves a vector at a specific 0-based index from a vector file and prints its type and values. Supports fvec and ivec formats with text, CSV, and JSON output.\n\n## Options\n\n{}",
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

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let ordinal: usize = match options.require("ordinal") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid ordinal: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let format = options.get("format").unwrap_or("text");
        let input_path = resolve_path(input_str, &ctx.workspace);

        let ext = input_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext {
            "fvec" => select_fvec(&input_path, ordinal, format, start),
            "ivec" => select_ivec(&input_path, ordinal, format, start),
            _ => error_result(
                format!("unsupported file extension: '{}'. Use fvec or ivec", ext),
                start,
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "input".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Vector file to read from".to_string(),
            },
            OptionDesc {
                name: "ordinal".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "0-based index of the vector to retrieve".to_string(),
            },
            OptionDesc {
                name: "format".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("text".to_string()),
                description: "Output format: text, csv, json".to_string(),
            },
        ]
    }
}

fn select_fvec(path: &Path, ordinal: usize, format: &str, start: Instant) -> CommandResult {
    let reader = match MmapVectorReader::<f32>::open_fvec(path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("failed to open {}: {}", path.display(), e), start),
    };

    let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
    if ordinal >= count {
        return error_result(
            format!("ordinal {} out of range (file has {} vectors)", ordinal, count),
            start,
        );
    }

    let vec = match reader.get(ordinal) {
        Ok(v) => v,
        Err(e) => return error_result(format!("read error: {}", e), start),
    };

    let output = format_vector("float[]", &vec, format, ordinal);
    eprintln!("{}", output);

    CommandResult {
        status: Status::Ok,
        message: format!("selected float vector at ordinal {}", ordinal),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn select_ivec(path: &Path, ordinal: usize, format: &str, start: Instant) -> CommandResult {
    let reader = match MmapVectorReader::<i32>::open_ivec(path) {
        Ok(r) => r,
        Err(e) => return error_result(format!("failed to open {}: {}", path.display(), e), start),
    };

    let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
    if ordinal >= count {
        return error_result(
            format!("ordinal {} out of range (file has {} vectors)", ordinal, count),
            start,
        );
    }

    let vec = match reader.get(ordinal) {
        Ok(v) => v,
        Err(e) => return error_result(format!("read error: {}", e), start),
    };

    let output = format_vector("int[]", &vec, format, ordinal);
    eprintln!("{}", output);

    CommandResult {
        status: Status::Ok,
        message: format!("selected int vector at ordinal {}", ordinal),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn format_vector<T: std::fmt::Display>(type_name: &str, values: &[T], format: &str, ordinal: usize) -> String {
    match format {
        "json" => {
            let vals: Vec<String> = values.iter().map(|v| format!("{}", v)).collect();
            format!(
                "{{\"ordinal\":{},\"type\":\"{}\",\"dim\":{},\"values\":[{}]}}",
                ordinal,
                type_name,
                values.len(),
                vals.join(",")
            )
        }
        "csv" => {
            let vals: Vec<String> = values.iter().map(|v| format!("{}", v)).collect();
            vals.join(",")
        }
        _ => {
            // text format
            let vals: Vec<String> = values.iter().map(|v| format!("{}", v)).collect();
            format!(
                "Vector Data:\n  type: {}\n  ordinal: {}\n  dim: {}\n  values: [{}]",
                type_name,
                ordinal,
                values.len(),
                vals.join(", ")
            )
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
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
        }
    }

    #[test]
    fn test_select_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate vectors
        let path = ws.join("test.fvec");
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "10");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        // Select vector at ordinal 5
        let mut opts = Options::new();
        opts.set("input", path.to_string_lossy().to_string());
        opts.set("ordinal", "5");
        let mut op = AnalyzeSelectOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_select_out_of_range() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let path = ws.join("test.fvec");
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "5");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("input", path.to_string_lossy().to_string());
        opts.set("ordinal", "10");
        let mut op = AnalyzeSelectOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("out of range"));
    }

    #[test]
    fn test_select_json_format() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let path = ws.join("test.fvec");
        let mut opts = Options::new();
        opts.set("output", path.to_string_lossy().to_string());
        opts.set("dimension", "3");
        opts.set("count", "5");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("input", path.to_string_lossy().to_string());
        opts.set("ordinal", "0");
        opts.set("format", "json");
        let mut op = AnalyzeSelectOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_format_vector_text() {
        let values = vec![1.0f32, 2.5, 3.7];
        let output = format_vector("float[]", &values, "text", 0);
        assert!(output.contains("float[]"));
        assert!(output.contains("dim: 3"));
    }

    #[test]
    fn test_format_vector_csv() {
        let values = vec![1, 2, 3];
        let output = format_vector("int[]", &values, "csv", 0);
        assert_eq!(output, "1,2,3");
    }
}
