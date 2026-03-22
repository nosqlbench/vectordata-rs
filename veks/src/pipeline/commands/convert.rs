// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline wrapper for the convert file operation.
//!
//! Extracts options from the pipeline `Options` map and delegates to the
//! existing format conversion logic.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;


use crate::formats::VecFormat;
use crate::formats::convert::convert_elements_into;
use crate::formats::reader;
use crate::formats::writer::{self, SinkConfig};
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status,
    StreamContext, render_options_table,
};

/// Number of records to buffer in the read-ahead channel.
///
/// Allows the reader thread to stay ahead of the convert+write thread,
/// hiding I/O latency behind conversion work.
const READ_AHEAD_BUFSIZE: usize = 4096;

/// Pipeline command: convert a vector file between formats.
pub struct ConvertFileOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ConvertFileOp)
}

impl CommandOp for ConvertFileOp {
    fn command_path(&self) -> &str {
        "convert file"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Convert a vector file between formats".into(),
            body: format!(
                r#"# convert file

Convert a vector file between formats.

## Description

The convert file command transforms a vector file from one format to another.
It supports all recognized vector formats including fvec, ivec, bvec, npy, mvec,
slab, and xvec. Both the source and target formats can be specified explicitly,
or the source format can be auto-detected from the file extension.

When the source and target formats use different element sizes, the command
performs element widening or narrowing automatically. For example, converting
from fvec (f32, 4 bytes per element) to a half-precision format (f16, 2 bytes
per element) narrows each element, while going from bvec (u8, 1 byte) to fvec
widens them. The conversion is applied per-element across every dimension of
every record.

Internally, the command uses a read-ahead pipeline architecture: a background
reader thread fills a bounded channel with source records while the main thread
handles element conversion and writes to the output sink. This design hides I/O
latency behind conversion work, keeping throughput high even for large files.
The read-ahead buffer holds up to 4096 records.

Common format pairs include:

- npy to fvec or ivec -- converting numpy exports to native vector formats
- fvec to mvec -- migrating to memory-mapped layout
- bvec to fvec -- widening byte vectors to float vectors
- any format to slab -- converting to paged slab storage

## Data Preparation Role

The convert command is typically used as a standalone operation rather than as
part of a multi-step pipeline. It is most useful for format migration: when
source data arrives in one format but downstream tools or benchmarks require
another. In pipeline contexts, the import command usually handles format
selection automatically, so explicit conversion is only needed for special cases
such as changing element precision after an initial import.

## Examples

Convert a numpy file to fvec format:

    convert file --source vectors.npy --output vectors.fvec --to fvec

Convert with explicit source format override:

    convert file --source data.bin --output data.fvec --from bvec --to fvec

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
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
        ctx.ui.log(&format!(
            "  converting {} ({}) -> {} ({})",
            source_path.display(),
            source_format.name(),
            output_path.display(),
            to_format.name(),
        ));
        ctx.ui.log("  opening source...");
        let threads = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
        let mut source = match reader::open_source(&source_path, source_format, threads, None) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open source {}: {}", source_path.display(), e), start),
        };

        let dimension = source.dimension();
        let src_element_size = source.element_size();
        let dst_element_size = to_format.element_size();
        let record_count = source.record_count();

        // Determine if element conversion is needed
        let needs_conversion = dst_element_size > 0
            && src_element_size > 0
            && src_element_size != dst_element_size;

        ctx.ui.log(&format!(
            "  source: dimension={}, element_size={}, records={}",
            dimension,
            src_element_size,
            record_count.map_or("unknown".to_string(), |n| n.to_string()),
        ));
        if needs_conversion {
            ctx.ui.log(&format!(
                "  element conversion: {} ({} bytes) -> {} ({} bytes)",
                source_format.data_type_name(),
                src_element_size,
                to_format.data_type_name(),
                dst_element_size,
            ));
        }

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

        // Open sink — when converting element sizes, tell the sink about the
        // target format so it computes record sizes from the output element size.
        let sink_config = SinkConfig {
            dimension,
            source_format: if needs_conversion { to_format } else { source_format },
            slab_page_size,
            slab_namespace,
        };
        let mut sink = match writer::open_sink(&output_path, to_format, &sink_config) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open sink {}: {}", output_path.display(), e), start),
        };

        // Progress bar
        let pb = if let Some(total) = record_count {
            ctx.ui.bar(total, "converting records")
        } else {
            ctx.ui.spinner("converting records")
        };

        // Read-ahead pipeline: a background thread reads records into a
        // bounded channel while the main thread converts and writes them.
        // This hides I/O latency behind conversion/write work.
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(READ_AHEAD_BUFSIZE);

        let reader_handle = std::thread::Builder::new()
            .name("convert-reader".into())
            .spawn(move || {
                while let Some(data) = source.next_record() {
                    if tx.send(data).is_err() {
                        break; // receiver dropped
                    }
                }
            })
            .expect("failed to spawn reader thread");

        // Pre-allocate conversion buffer (reused across all records)
        let dst_record_bytes = dimension as usize * dst_element_size;
        let mut conv_buf = if needs_conversion {
            vec![0u8; dst_record_bytes]
        } else {
            Vec::new()
        };

        // Governor checkpoint before conversion
        if ctx.governor.checkpoint() {
            log::info!("governor: throttle active");
        }

        // Convert + write on the main thread
        let mut count: u64 = 0;
        for data in rx {
            if needs_conversion {
                if let Some(_n) = convert_elements_into(
                    &data, src_element_size, dst_element_size, &mut conv_buf,
                ) {
                    sink.write_record(count as i64, &conv_buf);
                } else {
                    sink.write_record(count as i64, &data);
                }
            } else {
                sink.write_record(count as i64, &data);
            };
            count += 1;
            pb.inc(1);
        }

        reader_handle.join().expect("reader thread panicked");
        pb.finish();

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
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output file path".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "to".to_string(),
                type_name: "enum".to_string(),
                required: true,
                default: None,
                description: "Output format (fvec, ivec, slab, etc.)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "from".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Source format override (auto-detected if omitted)".to_string(),
                role: OptionRole::Config,
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

