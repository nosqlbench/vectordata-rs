// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: require another pipeline to complete successfully.
//!
//! Runs an external pipeline YAML file to completion before the current
//! pipeline continues. This enables dependency chains across pipeline
//! files — for example, a download pipeline that feeds multiple dataset
//! preparation pipelines.

use std::path::PathBuf;
use std::time::Instant;

use crate::pipeline::command::{
    ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, ResourceDesc, Status, StreamContext, render_options_table,
};
use crate::pipeline::progress::ProgressLog;

/// Check if a step is complete in the progress log.
fn step_complete(progress: &ProgressLog, step_id: &str) -> bool {
    progress.get_step(step_id)
        .is_some_and(|r| r.status == Status::Ok)
}

pub struct RequirePipelineOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(RequirePipelineOp)
}

impl CommandOp for RequirePipelineOp {
    fn command_path(&self) -> &str {
        "pipeline require"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Require another pipeline to complete before continuing".into(),
            body: format!(
                r#"# pipeline require

Require another pipeline YAML file to complete successfully.

## Description

Runs the specified pipeline file to completion using the same pipeline
runner. If the required pipeline succeeds (all steps complete), this
step succeeds. If it fails, this step fails and the current pipeline
stops.

This enables dependency chains across pipeline files. For example, a
`download.yaml` pipeline that fetches source data can be required by
multiple `dataset.yaml` pipelines that process different subsets of
that data.

The required pipeline runs in its own workspace (the directory
containing the required YAML file), with its own progress log and
variable state. It does not share state with the parent pipeline.

If the required pipeline's steps are already complete (from a previous
run), the command returns immediately without re-running anything.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let file_str = match options.require("file") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };

        let profile = options.get("profile").unwrap_or("all").to_string();

        // Resolve relative to the current workspace
        let pipeline_path = if std::path::Path::new(&file_str).is_absolute() {
            PathBuf::from(&file_str)
        } else {
            ctx.workspace.join(&file_str)
        };

        if !pipeline_path.exists() {
            return error_result(
                format!("required pipeline not found: {}", pipeline_path.display()),
                start,
            );
        }

        ctx.ui.log(&format!("require: running {}", file_str));

        // Load the required pipeline's config
        let config = match vectordata::dataset::config::DatasetConfig::load(&pipeline_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("failed to load {}: {}", file_str, e), start),
        };

        let upstream = match &config.upstream {
            Some(u) => u,
            None => {
                return CommandResult {
                    status: Status::Ok,
                    message: format!("{}: no upstream steps", file_str),
                    produced: vec![],
                    elapsed: start.elapsed(),
                };
            }
        };

        // Collect step IDs
        let step_ids: Vec<String> = upstream.steps.iter().flatten()
            .filter_map(|s| s.id.clone().or_else(|| Some(s.run.clone())))
            .collect();

        // Check if all steps are already complete
        let progress_path = ProgressLog::path_for_dataset(&pipeline_path);
        let progress = match ProgressLog::load(&progress_path) {
            Ok((p, _)) => p,
            Err(e) => {
                if progress_path.exists() {
                    ctx.ui.log(&format!(
                        "Warning: could not load progress log '{}': {} — treating as empty",
                        progress_path.display(), e
                    ));
                }
                ProgressLog::new()
            }
        };

        let all_complete = step_ids.iter().all(|id| step_complete(&progress, id));

        if all_complete {
            ctx.ui.log(&format!("require: {} — all steps already complete", file_str));
            return CommandResult {
                status: Status::Ok,
                message: format!("{}: all steps complete", file_str),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Run the required pipeline
        let run_args = crate::pipeline::RunArgs {
            dataset: Some(pipeline_path.clone()),
            recursive: false,
            profile,
            dry_run: ctx.dry_run,
            clean: false,
            clean_last: false,
            reset: false,
            overrides: vec![],
            threads: ctx.threads,
            emit_yaml: false,
            resources: None,
            governor: "maximize".to_string(),
            status_interval: 250,
            output: "auto".to_string(),
        };

        if let Err(e) = crate::pipeline::run_pipeline(run_args) {
            return error_result(format!("required pipeline failed: {}", e), start);
        }

        // Re-check progress after run
        let progress = match ProgressLog::load(&progress_path) {
            Ok((p, _)) => p,
            Err(e) => {
                ctx.ui.log(&format!(
                    "Warning: could not load progress log after run: {} — treating as empty", e
                ));
                ProgressLog::new()
            }
        };
        let all_complete = step_ids.iter().all(|id| step_complete(&progress, id));

        if all_complete {
            CommandResult {
                status: Status::Ok,
                message: format!("{}: completed successfully", file_str),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        } else {
            let incomplete: Vec<&str> = step_ids.iter()
                .filter(|id| !step_complete(&progress, id))
                .map(|s| s.as_str())
                .collect();
            error_result(
                format!("{}: {} step(s) incomplete: {}",
                    file_str, incomplete.len(), incomplete.join(", ")),
                start,
            )
        }
    }

    fn check_artifact(&self, _output: &std::path::Path, options: &Options) -> ArtifactState {
        let file_str = match options.get("file") {
            Some(s) => s.to_string(),
            None => return ArtifactState::Absent,
        };

        let pipeline_path = PathBuf::from(&file_str);
        if !pipeline_path.exists() {
            return ArtifactState::Absent;
        }

        let config = match vectordata::dataset::config::DatasetConfig::load(&pipeline_path) {
            Ok(c) => c,
            Err(_) => return ArtifactState::Absent,
        };

        let progress_path = ProgressLog::path_for_dataset(&pipeline_path);
        let progress = match ProgressLog::load(&progress_path) {
            Ok((p, _)) => p,
            Err(_) => return ArtifactState::Absent,
        };

        match &config.upstream {
            Some(upstream) => {
                let step_ids: Vec<String> = upstream.steps.iter().flatten()
                    .filter_map(|s| s.id.clone().or_else(|| Some(s.run.clone())))
                    .collect();
                let all_complete = step_ids.iter().all(|id| step_complete(&progress, id));
                if all_complete {
                    ArtifactState::Complete
                } else {
                    ArtifactState::PartialResumable
                }
            }
            None => ArtifactState::Complete,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Path to the required pipeline YAML file".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "profile".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("all".to_string()),
                description: "Profile to run in the required pipeline".to_string(),
                role: OptionRole::Config,
            },
        ]
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
