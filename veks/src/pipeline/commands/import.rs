// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline wrapper for the import facet operation.
//!
//! Extracts options from the pipeline `Options` map, constructs the
//! appropriate arguments, and delegates to the existing import logic.

use std::path::{Path, PathBuf};
use std::time::Instant;

use indicatif::{ProgressState, ProgressStyle};

use crate::formats::VecFormat;
use crate::formats::reader;
use crate::formats::writer::{self, SinkConfig};
use crate::import::Facet;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: import a single facet from source to output.
pub struct ImportFacetOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ImportFacetOp)
}

impl CommandOp for ImportFacetOp {
    fn command_path(&self) -> &str {
        "import facet"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Import a data facet from source format to output".into(),
            body: format!(
                "# import facet\n\n\
                 Import a data facet from source format to output.\n\n\
                 ## Description\n\n\
                 Extracts options from the pipeline options map, constructs the \
                 appropriate arguments, and delegates to the existing import logic. \
                 Supports format auto-detection and configurable threading.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "threads".into(), description: "Parallel source reading".into(), adjustable: true },
            ResourceDesc { name: "iothreads".into(), description: "Concurrent I/O operations".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Read-ahead buffer size".into(), adjustable: false },
        ]
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
        let facet_str = match options.require("facet") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let facet = match Facet::from_key(facet_str) {
            Some(f) => f,
            None => return error_result(format!("unknown facet: '{}'", facet_str), start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let format_override = options.get("format");
        let threads: usize = options
            .get("threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| ctx.governor.current_or("threads", ctx.threads as u64) as usize);
        let slab_page_size: Option<u32> = options
            .get("slab_page_size")
            .and_then(|s| s.parse().ok());
        let slab_namespace: u8 = options
            .get("slab_namespace")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let max_count: Option<u64> = options
            .get("count")
            .and_then(|s| s.parse().ok());

        // Resolve source format
        let source_format = if let Some(fmt_str) = format_override {
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

        // Open source
        let mut source = match reader::open_source_for_facet(&source_path, source_format, facet, threads, max_count) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open source: {}", e), start),
        };

        let dimension = source.dimension();
        let element_size = source.element_size();
        let target_format = facet.preferred_format(element_size);

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(
                        format!("failed to create directory {}: {}", parent.display(), e),
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
        let mut sink = match writer::open_sink(&output_path, target_format, &sink_config) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open sink: {}", e), start),
        };

        // Progress bar
        let record_count = source.record_count();
        let effective_total = match (max_count, record_count) {
            (Some(mc), Some(rc)) => Some(std::cmp::min(mc, rc)),
            (Some(mc), None) => Some(mc),
            (None, Some(rc)) => Some(rc),
            (None, None) => None,
        };
        let pb = if let Some(total) = effective_total {
            ctx.display.bar_with_style(
                total,
                ProgressStyle::default_bar()
                    .template("  [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) records — {rps} — ETA {eta}")
                    .expect("invalid template")
                    .with_key("rps", format_rps)
                    .progress_chars("=>-"),
            )
        } else {
            ctx.display.spinner("importing records")
        };

        // Governor checkpoint before import processing
        if ctx.governor.checkpoint() {
            ctx.display.log("  governor: throttle active");
        }

        // Import records
        let mut count: u64 = 0;
        while let Some(data) = source.next_record() {
            if let Some(mc) = max_count {
                if count >= mc {
                    break;
                }
            }
            sink.write_record(count as i64, &data);
            count += 1;
            pb.inc(1);
        }

        pb.finish_and_clear();

        if let Err(e) = sink.finish() {
            return error_result(format!("failed to finalize output: {}", e), start);
        }

        CommandResult {
            status: Status::Ok,
            message: format!("imported {} records to {}", count, output_path.display()),
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
                name: "facet".to_string(),
                type_name: "enum".to_string(),
                required: true,
                default: None,
                description: "Facet type (e.g. base_vectors, query_vectors)".to_string(),
            },
            OptionDesc {
                name: "format".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Source format override (auto-detected if omitted)".to_string(),
            },
            OptionDesc {
                name: "threads".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Number of loader threads (0 = auto)".to_string(),
            },
            OptionDesc {
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Maximum number of records to import (all if omitted)".to_string(),
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

/// Format records/sec for the progress bar.
fn format_rps(state: &ProgressState, w: &mut dyn std::fmt::Write) {
    let rps = state.per_sec();
    if rps < 100.0 {
        write!(w, "{:.1}/s", rps).unwrap();
    } else {
        let whole = rps as u64;
        let s = whole.to_string();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        for (i, ch) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(ch);
        }
        let formatted: String = result.chars().rev().collect();
        write!(w, "{}/s", formatted).unwrap();
    }
}
