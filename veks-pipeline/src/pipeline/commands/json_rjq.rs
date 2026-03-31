// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: JSON/JSONL transformation using jaq (pure Rust jq).
//!
//! Applies full jq expressions to JSONL (JSON Lines) files via the `jaq`
//! crate — a pure Rust implementation of the jq language. This gives access
//! to nearly the complete jq expression language including pipes, array
//! operations, object construction, conditionals, `map`, `select`,
//! `group_by`, `keys`, `length`, etc., with no C library dependencies.
//!
//! Each line of the input is parsed as a JSON value, the compiled jq program
//! is applied, and all results are written to the output.
//!
//! This is the Rust equivalent of the Java `CMD_jjq` command — named `rjq`
//! to distinguish it from the Java `jjq`.

use std::io::{BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::time::Instant;

use jaq_interpret::{Ctx, FilterT, ParseCtx, RcIter, Val};

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: JSON/JSONL transformation via jaq.
pub struct JsonRjqOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(JsonRjqOp)
}

/// Compile a jq expression using jaq, including the standard library.
fn compile_jq(expr: &str) -> Result<jaq_interpret::Filter, String> {
    // Create context with no global variables
    let mut defs = ParseCtx::new(Vec::new());

    // Load jaq standard library definitions
    defs.insert_natives(jaq_core::core());
    defs.insert_defs(jaq_std::std());

    // Parse the expression
    let (parsed, errs) = jaq_parse::parse(expr, jaq_parse::main());
    if !errs.is_empty() {
        return Err(format!(
            "parse errors in '{}': {}",
            expr,
            errs.len()
        ));
    }
    let parsed = parsed.ok_or_else(|| format!("failed to parse expression: '{}'", expr))?;

    // Compile
    let filter = defs.compile(parsed);
    if !defs.errs.is_empty() {
        return Err(format!(
            "compile errors in '{}': {}",
            expr,
            defs.errs.len()
        ));
    }

    Ok(filter)
}

/// Convert a serde_json::Value to a jaq Val.
fn json_to_val(v: serde_json::Value) -> Val {
    Val::from(v)
}

/// Convert a jaq Val to a serde_json::Value.
fn val_to_json(v: Val) -> serde_json::Value {
    serde_json::Value::from(v)
}

impl CommandOp for JsonRjqOp {
    fn command_path(&self) -> &str {
        "query records"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Query slab records with jq-like expressions".into(),
            body: format!(
                r#"# query records

Query slab records with jq-like expressions.

## Description

Applies full jq expressions to JSONL files via the `jaq` crate -- a
pure Rust implementation of the jq language. Unlike `json jjq` which
supports only a simplified subset, `rjq` provides nearly the complete
jq expression language including pipes, array operations, object
construction, conditionals, `map`, `select`, `group_by`, `keys`,
`length`, and more.

## How It Works

The command compiles the jq expression using the `jaq` parser and
standard library (including `jaq_core` and `jaq_std`), then opens the
input JSONL file and processes it line by line. Each line is parsed as
a `serde_json::Value`, converted to a `jaq` `Val`, and run through
the compiled filter. A single input value may produce zero, one, or
many output values (for example, `.[]` iterates array elements).
All output values are serialized as JSON lines. Parse errors and
evaluation errors are logged (up to 5) and counted in the status.

## Data Preparation Role

`json rjq` is the full-power JSON transformation tool in the pipeline,
used when `json jjq`'s simplified expressions are insufficient.
Typical use cases include restructuring complex metadata records,
computing derived fields, grouping records for aggregation, and
performing conditional transformations. Because it uses a pure Rust jq
implementation with no C dependencies, it works on any platform without
requiring an external `jq` binary. This makes it suitable for
automated pipeline execution in containerized or restricted
environments.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Slab record iteration buffers".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Read-ahead buffer size".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let expr_str = options.get("expression").unwrap_or(".");
        let output_str = options.get("output");

        let input_path = resolve_path(input_str, &ctx.workspace);

        // Compile jq program
        let filter = match compile_jq(expr_str) {
            Ok(f) => f,
            Err(e) => return error_result(e, start),
        };

        // Open input
        let file = match std::fs::File::open(&input_path) {
            Ok(f) => f,
            Err(e) => {
                return error_result(
                    format!("failed to open {}: {}", input_path.display(), e),
                    start,
                )
            }
        };
        let reader = std::io::BufReader::new(file);

        // Open output
        let output_path = output_str.map(|s| resolve_path(s, &ctx.workspace));
        let mut writer: Box<dyn IoWrite> = match &output_path {
            Some(p) if p.to_string_lossy() != "stdout" && p.to_string_lossy() != "null" => {
                if let Some(parent) = p.parent() {
                    if !parent.exists() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                }
                match std::fs::File::create(p) {
                    Ok(f) => Box::new(std::io::BufWriter::new(f)),
                    Err(e) => {
                        return error_result(
                            format!("failed to create {}: {}", p.display(), e),
                            start,
                        )
                    }
                }
            }
            Some(p) if p.to_string_lossy() == "null" => Box::new(std::io::sink()),
            _ => Box::new(std::io::BufWriter::new(std::io::stderr())),
        };

        let mut line_count = 0usize;
        let mut output_count = 0usize;
        let mut error_count = 0usize;

        // Process JSONL line by line
        let inputs = RcIter::new(core::iter::empty());

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            line_count += 1;

            // Parse JSON
            let json_val: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    error_count += 1;
                    if error_count <= 5 {
                        ctx.ui.log(&format!("  Warning: line {}: parse error: {}", line_count, e));
                    }
                    continue;
                }
            };

            // Run filter
            let val = json_to_val(json_val);
            let mut out = filter.run((Ctx::new([], &inputs), val));

            while let Some(result) = out.next() {
                match result {
                    Ok(v) => {
                        let json = val_to_json(v);
                        let _ = serde_json::to_writer(&mut writer, &json);
                        let _ = writeln!(writer);
                        output_count += 1;
                    }
                    Err(e) => {
                        error_count += 1;
                        if error_count <= 5 {
                            ctx.ui.log(&format!("  Warning: line {}: eval error: {:?}", line_count, e));
                        }
                    }
                }
            }
        }

        let _ = writer.flush();

        if error_count > 5 {
            ctx.ui.log(&format!("  ... and {} more errors", error_count - 5));
        }

        let status = if error_count > 0 && output_count == 0 {
            Status::Error
        } else if error_count > 0 {
            Status::Warning
        } else {
            Status::Ok
        };

        CommandResult {
            status,
            message: format!(
                "processed {} lines, output {} results{} (expr: {})",
                line_count,
                output_count,
                if error_count > 0 {
                    format!(", {} errors", error_count)
                } else {
                    String::new()
                },
                expr_str
            ),
            produced: output_path.into_iter().collect(),
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
                description: "Input JSONL file".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "expression".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some(".".to_string()),
                description: "jq expression (full jq syntax via jaq)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output file (default: stderr, 'null' discards)".to_string(),
                        role: OptionRole::Output,
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_compile_jq_identity() {
        let filter = compile_jq(".").unwrap();
        let inputs = RcIter::new(core::iter::empty());
        let val = json_to_val(serde_json::json!({"a": 1}));
        let mut out = filter.run((Ctx::new([], &inputs), val));
        let result = out.next().unwrap().unwrap();
        let json = val_to_json(result);
        assert_eq!(json, serde_json::json!({"a": 1}));
    }

    #[test]
    fn test_compile_jq_field() {
        let filter = compile_jq(".name").unwrap();
        let inputs = RcIter::new(core::iter::empty());
        let val = json_to_val(serde_json::json!({"name": "Alice", "age": 30}));
        let mut out = filter.run((Ctx::new([], &inputs), val));
        let result = out.next().unwrap().unwrap();
        let json = val_to_json(result);
        assert_eq!(json, serde_json::json!("Alice"));
    }

    #[test]
    fn test_compile_jq_invalid() {
        assert!(compile_jq("invalid syntax [[[").is_err());
    }

    #[test]
    fn test_rjq_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(&input, "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n").unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());

        let mut op = JsonRjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "rjq failed: {}", result.message);
        assert!(result.message.contains("3 lines"));

        let content = std::fs::read_to_string(&output).unwrap();
        assert_eq!(content.trim().lines().count(), 3);
    }

    #[test]
    fn test_rjq_field_extract() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(
            &input,
            "{\"name\":\"Alice\",\"age\":30}\n{\"name\":\"Bob\",\"age\":25}\n",
        )
        .unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("expression", ".name");

        let mut op = JsonRjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("Alice"));
        assert!(content.contains("Bob"));
    }

    #[test]
    fn test_rjq_select_filter() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(
            &input,
            "{\"name\":\"Alice\",\"age\":30}\n{\"name\":\"Bob\",\"age\":15}\n{\"name\":\"Charlie\",\"age\":25}\n",
        )
        .unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("expression", "select(.age >= 25)");

        let mut op = JsonRjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("Alice"));
        assert!(content.contains("Charlie"));
        assert!(!content.contains("Bob"));
    }

    #[test]
    fn test_rjq_pipe_expression() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(
            &input,
            "{\"items\":[1,2,3]}\n{\"items\":[4,5,6]}\n",
        )
        .unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("expression", ".items | length");

        let mut op = JsonRjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("3"));
    }

    #[test]
    fn test_rjq_object_construction() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(
            &input,
            "{\"first\":\"Alice\",\"last\":\"Smith\",\"age\":30}\n",
        )
        .unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("expression", "{name: .first, surname: .last}");

        let mut op = JsonRjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("Alice"));
        assert!(content.contains("Smith"));
    }

    #[test]
    fn test_rjq_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(&input, "{\"b\":2,\"a\":1,\"c\":3}\n").unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("expression", "keys");

        let mut op = JsonRjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("a"));
        assert!(content.contains("b"));
        assert!(content.contains("c"));
    }

    #[test]
    fn test_rjq_invalid_expression() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(&input, "{\"a\":1}\n").unwrap();

        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("expression", "invalid syntax [[[");

        let mut op = JsonRjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }
}
