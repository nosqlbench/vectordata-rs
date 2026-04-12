// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate `dataset.json` from `dataset.yaml`.
//!
//! Writes a JSON-format copy of the dataset configuration alongside
//! the YAML source. This provides a JSON-accessible copy for clients
//! that prefer JSON over YAML. The file is only written when missing
//! or when `dataset.yaml` is newer.

use std::time::Instant;

use vectordata::dataset::DatasetConfig;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, Options,
    Status, StreamContext, render_options_table,
};

pub struct DatasetJsonOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(DatasetJsonOp)
}

impl CommandOp for DatasetJsonOp {
    fn command_path(&self) -> &str {
        "generate dataset-json"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate dataset.json from dataset.yaml".into(),
            body: format!(
                "# generate dataset-json\n\n\
                 Generate a JSON copy of the dataset configuration.\n\n\
                 ## Description\n\n\
                 Reads `dataset.yaml` from the workspace and writes `dataset.json` \
                 alongside it. The JSON file is only written when missing or when \
                 `dataset.yaml` has a newer modification time. This provides a \
                 JSON-accessible copy for clients that prefer JSON over YAML.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, _options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let yaml_path = ctx.workspace.join("dataset.yaml");
        let json_path = ctx.workspace.join("dataset.json");

        if !yaml_path.exists() {
            return CommandResult {
                status: Status::Error,
                message: format!("dataset.yaml not found in {}", ctx.workspace.display()),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let needs_update = if !json_path.exists() {
            true
        } else {
            match (std::fs::metadata(&yaml_path), std::fs::metadata(&json_path)) {
                (Ok(ym), Ok(jm)) => ym.modified().ok() > jm.modified().ok(),
                _ => true,
            }
        };

        if !needs_update {
            return CommandResult {
                status: Status::Ok,
                message: "dataset.json is up to date".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let config = match DatasetConfig::load(&yaml_path) {
            Ok(c) => c,
            Err(e) => return CommandResult {
                status: Status::Error,
                message: format!("failed to load dataset.yaml: {}", e),
                produced: vec![],
                elapsed: start.elapsed(),
            },
        };

        let json = match serde_json::to_string_pretty(&config) {
            Ok(j) => j,
            Err(e) => return CommandResult {
                status: Status::Error,
                message: format!("JSON serialization failed: {}", e),
                produced: vec![],
                elapsed: start.elapsed(),
            },
        };

        if let Err(e) = std::fs::write(&json_path, &json) {
            return CommandResult {
                status: Status::Error,
                message: format!("failed to write {}: {}", json_path.display(), e),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        ctx.ui.log(&format!("  wrote {}", json_path.display()));

        CommandResult {
            status: Status::Ok,
            message: "wrote dataset.json".into(),
            produced: vec![json_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![]
    }

    fn project_artifacts(&self, step_id: &str, _options: &Options) -> ArtifactManifest {
        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs: vec![],
            outputs: vec!["dataset.json".to_string()],
            intermediates: vec![],
        }
    }
}
