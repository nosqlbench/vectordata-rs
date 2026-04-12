// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate `dataset.jsonl` from `dataset.log`.
//!
//! Converts the human-readable `dataset.log` provenance log into
//! JSON Lines format, with one JSON object per log entry. This
//! provides machine-readable access to the full preparation history.

use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options,
    Status, StreamContext, render_options_table,
};

pub struct GenDatasetLogJsonlOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenDatasetLogJsonlOp)
}

impl CommandOp for GenDatasetLogJsonlOp {
    fn command_path(&self) -> &str {
        "generate dataset-log-jsonl"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate dataset.jsonl from dataset.log".into(),
            body: format!(
                "# generate dataset-log-jsonl\n\n\
                 Convert the dataset provenance log to JSON Lines format.\n\n\
                 ## Description\n\n\
                 Parses `dataset.log` from the workspace and writes `dataset.jsonl` \
                 alongside it. Each log entry becomes a JSON object with fields for \
                 the event type, step ID, command, elapsed time, and message. \
                 Session headers and summary blocks are also captured.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![]
    }

    fn execute(&mut self, _options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let log_path = ctx.workspace.join("dataset.log");
        let jsonl_path = ctx.workspace.join("dataset.jsonl");

        if !log_path.exists() {
            return CommandResult {
                status: Status::Ok,
                message: "no dataset.log to convert".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let content = match std::fs::read_to_string(&log_path) {
            Ok(s) => s,
            Err(e) => return CommandResult {
                status: Status::Error,
                message: format!("failed to read {}: {}", log_path.display(), e),
                produced: vec![],
                elapsed: start.elapsed(),
            },
        };

        let mut entries: Vec<serde_json::Value> = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }

            // Session header: === veks prepare bootstrap 2026-04-03 18:28:37 ===
            if trimmed.starts_with("=== ") && trimmed.ends_with(" ===") {
                let inner = &trimmed[4..trimmed.len() - 4];
                entries.push(serde_json::json!({
                    "type": "session",
                    "message": inner,
                }));
                continue;
            }

            // Step start: [start] step-id — command
            if let Some(rest) = trimmed.strip_prefix("[start] ") {
                let (step_id, command) = if let Some((s, c)) = rest.split_once(" — ") {
                    (s.trim(), c.trim())
                } else {
                    (rest.trim(), "")
                };
                entries.push(serde_json::json!({
                    "type": "start",
                    "step": step_id,
                    "command": command,
                }));
                continue;
            }

            // Step ok: [ok] step-id (1.2s): message
            if let Some(rest) = trimmed.strip_prefix("[ok] ") {
                let (step_id, elapsed, message) = parse_result_line(rest);
                entries.push(serde_json::json!({
                    "type": "ok",
                    "step": step_id,
                    "elapsed_s": elapsed,
                    "message": message,
                }));
                continue;
            }

            // Step error: [error] step-id (1.2s): message
            if let Some(rest) = trimmed.strip_prefix("[error] ") {
                let (step_id, elapsed, message) = parse_result_line(rest);
                entries.push(serde_json::json!({
                    "type": "error",
                    "step": step_id,
                    "elapsed_s": elapsed,
                    "message": message,
                }));
                continue;
            }

            // Step skipped: [skip] step-id: message
            if let Some(rest) = trimmed.strip_prefix("[skip] ") {
                let (step_id, message) = if let Some((s, m)) = rest.split_once(": ") {
                    (s.trim(), m.trim())
                } else {
                    (rest.trim(), "")
                };
                entries.push(serde_json::json!({
                    "type": "skip",
                    "step": step_id,
                    "message": message,
                }));
                continue;
            }

            // Summary line: summary: N executed (Xs), M skipped, P total
            if trimmed.starts_with("summary:") {
                entries.push(serde_json::json!({
                    "type": "summary",
                    "message": trimmed,
                }));
                continue;
            }

            // Data flow lines: key: value (indented)
            if trimmed.starts_with("  ") && trimmed.contains(':') {
                // Contextual detail — attach to last entry or emit standalone
                if let Some((key, value)) = trimmed.trim().split_once(':') {
                    entries.push(serde_json::json!({
                        "type": "detail",
                        "key": key.trim(),
                        "value": value.trim(),
                    }));
                }
                continue;
            }

            // Section headers (e.g., "Inputs:", "data flow:")
            if trimmed.ends_with(':') && !trimmed.contains(' ') || trimmed == "data flow:" {
                entries.push(serde_json::json!({
                    "type": "section",
                    "name": trimmed.trim_end_matches(':'),
                }));
                continue;
            }

            // Anything else — raw line
            entries.push(serde_json::json!({
                "type": "line",
                "text": trimmed,
            }));
        }

        // Write JSONL
        let mut output = String::new();
        for entry in &entries {
            output.push_str(&serde_json::to_string(entry).unwrap_or_default());
            output.push('\n');
        }

        if let Err(e) = std::fs::write(&jsonl_path, &output) {
            return CommandResult {
                status: Status::Error,
                message: format!("failed to write {}: {}", jsonl_path.display(), e),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        ctx.ui.log(&format!(
            "  wrote dataset.jsonl: {} entries",
            entries.len(),
        ));

        CommandResult {
            status: Status::Ok,
            message: format!("wrote {} entries to {}", entries.len(), jsonl_path.display()),
            produced: vec![jsonl_path],
            elapsed: start.elapsed(),
        }
    }
}

/// Parse a result line like "step-id (1.2s): message"
/// Returns (step_id, elapsed_seconds, message)
fn parse_result_line(rest: &str) -> (&str, f64, &str) {
    // Format: step-id (Xs): message
    if let Some(paren_start) = rest.find('(') {
        let step_id = rest[..paren_start].trim();
        if let Some(paren_end) = rest[paren_start..].find(')') {
            let elapsed_str = &rest[paren_start + 1..paren_start + paren_end];
            let elapsed = elapsed_str.trim_end_matches('s').parse::<f64>().unwrap_or(0.0);
            let after_paren = &rest[paren_start + paren_end + 1..];
            let message = after_paren.strip_prefix(": ").unwrap_or(after_paren).trim();
            return (step_id, elapsed, message);
        }
    }
    (rest.trim(), 0.0, "")
}
