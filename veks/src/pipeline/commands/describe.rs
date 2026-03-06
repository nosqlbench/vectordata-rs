// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline wrapper for the analyze describe operation.
//!
//! Runs `analyze describe` on a source file and captures the output as a
//! command result. This is primarily useful for validation steps in pipelines.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::formats::VecFormat;
use crate::formats::reader;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: describe/analyze a vector file.
pub struct AnalyzeDescribeOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeDescribeOp)
}

impl CommandOp for AnalyzeDescribeOp {
    fn command_path(&self) -> &str {
        "analyze describe"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Describe vector file format and dimensions".into(),
            body: format!(
                "# analyze describe\n\nDescribe vector file format and dimensions.\n\n## Description\n\nProbes a vector file and reports its format, dimensionality, element size, and record count. Useful for validation steps in pipelines.\n\n## Options\n\n{}",
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

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);

        let format = if let Some(fmt_str) = options.get("format") {
            match VecFormat::from_extension(fmt_str) {
                Some(f) => f,
                None => return error_result(format!("unknown format: '{}'", fmt_str), start),
            }
        } else {
            match VecFormat::detect(&source_path) {
                Some(f) => f,
                None => {
                    return error_result(
                        format!(
                            "cannot detect format for '{}', set 'format' option",
                            source_path.display()
                        ),
                        start,
                    )
                }
            }
        };

        // Probe the source to get metadata
        let meta = match reader::probe_source(&source_path, format) {
            Ok(m) => m,
            Err(e) => return error_result(format!("failed to probe source: {}", e), start),
        };

        let message = format!(
            "{}: format={}, dim={}, elem_size={}, records={}",
            source_path.display(),
            format.name(),
            meta.dimension,
            meta.element_size,
            meta.record_count
                .map_or("unknown".to_string(), |n| n.to_string()),
        );

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "File or directory to describe".to_string(),
            },
            OptionDesc {
                name: "format".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Format override (auto-detected if omitted)".to_string(),
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
