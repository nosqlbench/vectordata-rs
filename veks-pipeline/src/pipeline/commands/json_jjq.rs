// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: JSON/JSONL transformation.
//!
//! Applies simple JQ-like expressions to JSONL (JSON Lines) files. Each line
//! of the input is parsed as a JSON object, the expression is applied, and the
//! result is written to the output.
//!
//! Supported expressions:
//! - `.` — identity (pass through)
//! - `.field` — extract field
//! - `.field1.field2` — nested field access
//! - `.field1, .field2` — select multiple fields (output as object)
//! - `select(.field == value)` — filter records
//! - `length` — count records
//!
//! This is a simplified subset of JQ, not a full implementation.
//! Equivalent to the Java `CMD_jjq` command (simplified).

use std::io::{BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: JSON/JSONL transformation.
pub struct JsonJjqOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(JsonJjqOp)
}

/// Parsed JQ expression.
#[derive(Debug, Clone)]
enum JqExpr {
    /// `.` — identity
    Identity,
    /// `.field` or `.field1.field2` — field access path
    FieldPath(Vec<String>),
    /// `.field1, .field2` — select multiple fields
    MultiField(Vec<Vec<String>>),
    /// `select(.field op value)` — filter
    Select {
        path: Vec<String>,
        op: SelectOp,
        value: Value,
    },
    /// `length` — count records
    Length,
}

#[derive(Debug, Clone)]
enum SelectOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
}

impl JqExpr {
    fn parse(expr: &str) -> Result<Self, String> {
        let expr = expr.trim();

        if expr == "." {
            return Ok(JqExpr::Identity);
        }

        if expr == "length" {
            return Ok(JqExpr::Length);
        }

        // select(.field op value)
        if expr.starts_with("select(") && expr.ends_with(')') {
            let inner = &expr[7..expr.len() - 1].trim();
            return parse_select(inner);
        }

        // Multiple fields: .field1, .field2
        if expr.contains(',') {
            let parts: Vec<Vec<String>> = expr
                .split(',')
                .map(|p| parse_field_path(p.trim()))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(JqExpr::MultiField(parts));
        }

        // Single field path: .field or .field1.field2
        if expr.starts_with('.') {
            let path = parse_field_path(expr)?;
            return Ok(JqExpr::FieldPath(path));
        }

        Err(format!("unsupported expression: '{}'", expr))
    }

    /// Apply expression to a JSON value.
    fn apply(&self, value: &Value) -> Option<Value> {
        match self {
            JqExpr::Identity => Some(value.clone()),
            JqExpr::FieldPath(path) => resolve_path(value, path),
            JqExpr::MultiField(paths) => {
                let mut obj = serde_json::Map::new();
                for path in paths {
                    if let Some(v) = resolve_path(value, path) {
                        let key = path.last().cloned().unwrap_or_default();
                        obj.insert(key, v);
                    }
                }
                Some(Value::Object(obj))
            }
            JqExpr::Select { path, op, value: expected } => {
                let actual = resolve_path(value, path)?;
                let matches = match op {
                    SelectOp::Eq => actual == *expected,
                    SelectOp::Ne => actual != *expected,
                    SelectOp::Gt => compare_values(&actual, expected) == Some(std::cmp::Ordering::Greater),
                    SelectOp::Lt => compare_values(&actual, expected) == Some(std::cmp::Ordering::Less),
                    SelectOp::Gte => {
                        let c = compare_values(&actual, expected);
                        c == Some(std::cmp::Ordering::Greater) || c == Some(std::cmp::Ordering::Equal)
                    }
                    SelectOp::Lte => {
                        let c = compare_values(&actual, expected);
                        c == Some(std::cmp::Ordering::Less) || c == Some(std::cmp::Ordering::Equal)
                    }
                };
                if matches {
                    Some(value.clone())
                } else {
                    None
                }
            }
            JqExpr::Length => None, // Handled at the stream level
        }
    }
}

fn parse_field_path(s: &str) -> Result<Vec<String>, String> {
    let s = s.trim();
    if !s.starts_with('.') {
        return Err(format!("field path must start with '.': '{}'", s));
    }
    let path: Vec<String> = s[1..]
        .split('.')
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect();
    if path.is_empty() {
        return Err(format!("empty field path: '{}'", s));
    }
    Ok(path)
}

fn resolve_path(value: &Value, path: &[String]) -> Option<Value> {
    let mut current = value;
    for key in path {
        current = current.get(key.as_str())?;
    }
    Some(current.clone())
}

fn parse_select(inner: &str) -> Result<JqExpr, String> {
    // Parse: .field op value
    let ops = ["!=", ">=", "<=", "==", ">", "<"];
    for op_str in &ops {
        if let Some(pos) = inner.find(op_str) {
            let field_part = inner[..pos].trim();
            let value_part = inner[pos + op_str.len()..].trim();

            let path = parse_field_path(field_part)?;
            let op = match *op_str {
                "==" => SelectOp::Eq,
                "!=" => SelectOp::Ne,
                ">" => SelectOp::Gt,
                "<" => SelectOp::Lt,
                ">=" => SelectOp::Gte,
                "<=" => SelectOp::Lte,
                _ => unreachable!(),
            };

            let value: Value = serde_json::from_str(value_part)
                .map_err(|e| format!("invalid select value '{}': {}", value_part, e))?;

            return Ok(JqExpr::Select { path, op, value });
        }
    }
    Err(format!("unsupported select expression: '{}'", inner))
}

fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let af = a.as_f64()?;
            let bf = b.as_f64()?;
            af.partial_cmp(&bf)
        }
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

impl CommandOp for JsonJjqOp {
    fn command_path(&self) -> &str {
        "query json"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Query JSON files with jq-like expressions".into(),
            body: format!(
                r#"# query json

Query JSON files with jq-like expressions.

## Description

Applies simple JQ-like expressions to JSONL (JSON Lines) files. Each
line of the input is parsed as a JSON object, the expression is
applied, and the result is written to the output. Supports field
extraction, nested access, multi-field selection, filtering with
`select()`, and `length` counting.

## Supported Expressions

- `.` -- identity (pass through the entire object)
- `.field` -- extract a single field
- `.field1.field2` -- nested field access
- `.field1, .field2` -- select multiple fields (output as object)
- `select(.field == value)` -- filter records (supports ==, !=, >, <, >=, <=)
- `length` -- count the number of records

## How It Works

The command opens the input JSONL file and processes it line by line.
Each line is parsed as a `serde_json::Value`, then the compiled
expression is applied. For most expressions the result is written as
a JSON line to the output. For `select()` expressions, lines that do
not match the predicate are silently dropped. For `length`, the entire
file is scanned and only the final count is emitted. Output defaults
to stderr; specify an output path to write to a file, or use `null`
to discard results (useful with `length`).

## Data Preparation Role

`json jjq` is a lightweight built-in alternative to external `jq` for
transforming metadata during pipeline execution. Common uses include
extracting specific fields from parquet-exported JSONL files before
slab import, filtering records by attribute value, and counting records
to set pipeline variables. Because it is built into veks, it avoids
the need for an external `jq` binary on the system.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "JSON parse buffers".into(), adjustable: false },
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

        let input_path = resolve_file_path(input_str, &ctx.workspace);

        let expr = match JqExpr::parse(expr_str) {
            Ok(e) => e,
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
        let output_path = output_str.map(|s| resolve_file_path(s, &ctx.workspace));
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

        // Handle `length` specially
        if matches!(expr, JqExpr::Length) {
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                if line.trim().is_empty() {
                    continue;
                }
                if serde_json::from_str::<Value>(&line).is_ok() {
                    line_count += 1;
                }
            }
            let _ = writeln!(writer, "{}", line_count);
            return CommandResult {
                status: Status::Ok,
                message: format!("counted {} records", line_count),
                produced: output_path.into_iter().collect(),
                elapsed: start.elapsed(),
            };
        }

        // Process JSONL line by line
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            line_count += 1;

            let value: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    ctx.ui.log(&format!("  Warning: line {}: parse error: {}", line_count, e));
                    continue;
                }
            };

            if let Some(result) = expr.apply(&value) {
                let _ = serde_json::to_writer(&mut writer, &result);
                let _ = writeln!(writer);
                output_count += 1;
            }
        }

        let _ = writer.flush();

        CommandResult {
            status: Status::Ok,
            message: format!(
                "processed {} lines, output {} records (expr: {})",
                line_count, output_count, expr_str
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
                description: "JQ expression (e.g. '.field', 'select(.x == 1)')".to_string(),
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

fn resolve_file_path(path_str: &str, workspace: &Path) -> PathBuf {
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
    fn test_parse_identity() {
        let expr = JqExpr::parse(".").unwrap();
        assert!(matches!(expr, JqExpr::Identity));
    }

    #[test]
    fn test_parse_field_path() {
        let expr = JqExpr::parse(".name").unwrap();
        match expr {
            JqExpr::FieldPath(path) => assert_eq!(path, vec!["name"]),
            _ => panic!("expected FieldPath"),
        }
    }

    #[test]
    fn test_parse_nested_path() {
        let expr = JqExpr::parse(".data.value").unwrap();
        match expr {
            JqExpr::FieldPath(path) => assert_eq!(path, vec!["data", "value"]),
            _ => panic!("expected FieldPath"),
        }
    }

    #[test]
    fn test_parse_multi_field() {
        let expr = JqExpr::parse(".name, .age").unwrap();
        assert!(matches!(expr, JqExpr::MultiField(_)));
    }

    #[test]
    fn test_parse_select() {
        let expr = JqExpr::parse("select(.age > 18)").unwrap();
        assert!(matches!(expr, JqExpr::Select { .. }));
    }

    #[test]
    fn test_apply_identity() {
        let expr = JqExpr::Identity;
        let val: Value = serde_json::json!({"a": 1});
        assert_eq!(expr.apply(&val), Some(val));
    }

    #[test]
    fn test_apply_field_path() {
        let expr = JqExpr::FieldPath(vec!["name".to_string()]);
        let val: Value = serde_json::json!({"name": "Alice", "age": 30});
        assert_eq!(expr.apply(&val), Some(Value::String("Alice".to_string())));
    }

    #[test]
    fn test_apply_select() {
        let expr = JqExpr::Select {
            path: vec!["age".to_string()],
            op: SelectOp::Gt,
            value: serde_json::json!(18),
        };

        let young: Value = serde_json::json!({"name": "Kid", "age": 10});
        assert_eq!(expr.apply(&young), None);

        let adult: Value = serde_json::json!({"name": "Adult", "age": 25});
        assert!(expr.apply(&adult).is_some());
    }

    #[test]
    fn test_jjq_identity_command() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Write JSONL input
        let input = ws.join("input.jsonl");
        std::fs::write(
            &input,
            "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n",
        )
        .unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());

        let mut op = JsonJjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("3 lines"));

        let output_content = std::fs::read_to_string(&output).unwrap();
        assert_eq!(output_content.lines().count(), 3);
    }

    #[test]
    fn test_jjq_field_extract() {
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

        let mut op = JsonJjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("Alice"));
        assert!(content.contains("Bob"));
    }

    #[test]
    fn test_jjq_select_filter() {
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

        let mut op = JsonJjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("2 records")); // Alice and Charlie

        let content = std::fs::read_to_string(&output).unwrap();
        assert_eq!(content.lines().count(), 2);
    }

    #[test]
    fn test_jjq_length() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let input = ws.join("input.jsonl");
        std::fs::write(&input, "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n").unwrap();

        let output = ws.join("output.jsonl");
        let mut opts = Options::new();
        opts.set("input", input.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("expression", "length");

        let mut op = JsonJjqOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("3 records"));
    }
}
