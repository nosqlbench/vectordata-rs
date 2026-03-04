// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline wrapper for the convert file operation.
//!
//! Extracts options from the pipeline `Options` map and delegates to the
//! existing format conversion logic.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::formats::VecFormat;
use crate::formats::reader;
use crate::formats::writer::{self, SinkConfig};
use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

/// Pipeline command: convert a vector file between formats.
pub struct ConvertFileOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ConvertFileOp)
}

impl CommandOp for ConvertFileOp {
    fn command_path(&self) -> &str {
        "convert file"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let to_format_str = match options.require("to") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        let to_format = match VecFormat::from_extension(to_format_str) {
            Some(f) => f,
            None => return error_result(format!("unknown output format: '{}'", to_format_str), start),
        };

        let source_format = if let Some(fmt_str) = options.get("from") {
            match VecFormat::from_extension(fmt_str) {
                Some(f) => f,
                None => return error_result(format!("unknown source format: '{}'", fmt_str), start),
            }
        } else {
            match VecFormat::detect(&source_path) {
                Some(f) => f,
                None => {
                    return error_result(
                        format!(
                            "cannot detect source format for '{}', set 'from' option",
                            source_path.display()
                        ),
                        start,
                    )
                }
            }
        };

        let slab_page_size: Option<u32> = options
            .get("slab_page_size")
            .and_then(|s| s.parse().ok());
        let slab_namespace: u8 = options
            .get("slab_namespace")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        // Open source
        let mut source = match reader::open_source(&source_path, source_format, 0) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open source: {}", e), start),
        };

        let dimension = source.dimension();

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(
                        format!("failed to create directory: {}", e),
                        start,
                    );
                }
            }
        }

        // Open sink
        let sink_config = SinkConfig {
            dimension,
            source_format,
            slab_page_size,
            slab_namespace,
        };
        let mut sink = match writer::open_sink(&output_path, to_format, &sink_config) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open sink: {}", e), start),
        };

        // Convert records
        let mut count: u64 = 0;
        while let Some(data) = source.next_record() {
            sink.write_record(count as i64, &data);
            count += 1;
        }

        if let Err(e) = sink.finish() {
            return error_result(format!("failed to finalize output: {}", e), start);
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "converted {} records ({} -> {}) to {}",
                count,
                source_format.name(),
                to_format.name(),
                output_path.display()
            ),
            produced: vec![output_path],
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
                description: "Source file or directory".to_string(),
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output file path".to_string(),
            },
            OptionDesc {
                name: "to".to_string(),
                type_name: "enum".to_string(),
                required: true,
                default: None,
                description: "Output format (fvec, ivec, slab, etc.)".to_string(),
            },
            OptionDesc {
                name: "from".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Source format override (auto-detected if omitted)".to_string(),
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
