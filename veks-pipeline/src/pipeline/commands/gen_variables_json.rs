// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate `variables.json` from `variables.yaml`.
//!
//! Writes a JSON-format copy of the pipeline variables alongside the
//! YAML source. This provides a JSON-accessible copy for clients and
//! tools that prefer JSON over YAML (dashboards, CI scripts, etc.).

use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options,
    Status, StreamContext, render_options_table,
};

pub struct GenVariablesJsonOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenVariablesJsonOp)
}

impl CommandOp for GenVariablesJsonOp {
    fn command_path(&self) -> &str {
        "generate variables-json"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate variables.json from variables.yaml".into(),
            body: format!(
                "# generate variables-json\n\n\
                 Generate a JSON copy of the pipeline variables.\n\n\
                 ## Description\n\n\
                 Reads `variables.yaml` from the workspace and writes `variables.json` \
                 alongside it. The JSON file contains the same key-value pairs as the \
                 YAML source. Numeric strings are preserved as strings for exact \
                 round-trip fidelity.\n\n\
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

        let yaml_path = ctx.workspace.join("variables.yaml");
        let json_path = ctx.workspace.join("variables.json");

        // Load from variables.yaml, overlaying ctx.defaults for freshness
        let mut vars: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();

        match super::super::variables::load(&ctx.workspace) {
            Ok(file_vars) => {
                for (k, v) in file_vars {
                    vars.insert(k, v);
                }
            }
            Err(e) => {
                return CommandResult {
                    status: Status::Error,
                    message: format!("failed to load variables.yaml: {}", e),
                    produced: vec![],
                    elapsed: start.elapsed(),
                };
            }
        }

        // Overlay ctx.defaults (may have newer values from the current run)
        for (k, v) in &ctx.defaults {
            vars.insert(k.clone(), v.clone());
        }

        if vars.is_empty() && !yaml_path.exists() {
            return CommandResult {
                status: Status::Ok,
                message: "no variables to write".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Build a JSON object preserving insertion order
        let json_obj: serde_json::Value = serde_json::Value::Object(
            vars.iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect()
        );

        let json_str = match serde_json::to_string_pretty(&json_obj) {
            Ok(s) => s,
            Err(e) => {
                return CommandResult {
                    status: Status::Error,
                    message: format!("JSON serialization failed: {}", e),
                    produced: vec![],
                    elapsed: start.elapsed(),
                };
            }
        };

        if let Err(e) = std::fs::write(&json_path, &json_str) {
            return CommandResult {
                status: Status::Error,
                message: format!("failed to write {}: {}", json_path.display(), e),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        ctx.ui.log(&format!(
            "  wrote variables.json: {} variables",
            vars.len(),
        ));

        CommandResult {
            status: Status::Ok,
            message: format!("wrote {} variables to {}", vars.len(), json_path.display()),
            produced: vec![json_path],
            elapsed: start.elapsed(),
        }
    }
}
