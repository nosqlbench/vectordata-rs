// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline wrapper for the import operation.
//!
//! Extracts options from the pipeline `Options` map, constructs the
//! appropriate arguments, and delegates to the existing import logic.

use std::path::{Path, PathBuf};
use std::time::Instant;


use veks_core::formats::VecFormat;
use veks_core::formats::reader;
use veks_core::formats::writer::{self, SinkConfig};
use veks_core::formats::facet::Facet;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

/// Pipeline command: import a single facet from source to output.
pub struct ImportFacetOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ImportFacetOp)
}

impl CommandOp for ImportFacetOp {
    fn command_path(&self) -> &str {
        "transform convert"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Import a data facet from source format to output".into(),
            body: format!(
                r#"# transform convert

Import a data facet from source format to output.

## Description

The import command is the primary entry point for getting external vector data
into the pipeline. It reads a directory (or single file) of source vectors and
writes a single consolidated output file in the pipeline's internal format.

Source files may be in any supported format: npy, fvec, ivec, bvec, parquet, or
other recognized extensions. When the `format` option is omitted, the command
auto-detects the source format by inspecting file extensions in the source path.
If the source is a directory, all files with a matching extension are discovered
and read in sorted order.

During import, each source format is converted to the appropriate internal
representation. Numpy `.npy` files are converted to xvec layout, parquet rows
become slab records, and native xvec/fvec files are read directly. The output
format is chosen automatically based on the facet type and the element size of
the source data.

The `facet` option labels this import with a dataset role such as `base_vectors`,
`query_vectors`, or `ground_truth`. This label determines how the data is used
in downstream pipeline steps and in the final dataset configuration.

Import is multithreaded: the `threads` option controls how many source reader
threads run in parallel. The pipeline governor may also adjust thread counts
dynamically based on observed throughput. For large imports spanning hundreds of
source files, a progress bar updates per-record so you can monitor throughput.

## Data Preparation Role

Import is the first step in any dataset preparation pipeline. It transforms raw
source data from its original distribution format into a consolidated internal
file that subsequent pipeline commands (extract, shuffle, partition) can operate
on efficiently. Without an import step, no other pipeline command has data to
work with.

## Notes

- For directories with many small files, increasing `threads` can significantly
  improve throughput by overlapping I/O across files.
- The `count` option is useful for creating smaller test datasets from large
  sources without copying the full data.
- Throughput statistics are logged at debug level every 5 seconds, including
  records/sec and MB/s. If sustained throughput is low, the governor may
  automatically request additional I/O bandwidth.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "threads".into(), description: "Parallel source reading".into(), adjustable: true },
            ResourceDesc { name: "iothreads".into(), description: "Concurrent I/O operations".into(), adjustable: true },
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

        // Probe source metadata using the facet-aware probe, which dispatches
        // metadata parquet to the MNode reader instead of the vector reader.
        let probe = match reader::probe_source_for_facet(&source_path, source_format, facet) {
            Ok(m) => m,
            Err(e) => return error_result(format!("failed to probe source {}: {}", source_path.display(), e), start),
        };
        let target_format = facet.preferred_format(probe.element_size);

        // Symlink fast path: if the source is already in the target format,
        // is a single file (not a directory), and no count limit is specified,
        // create a symlink instead of copying bytes.
        if source_format == target_format
            && max_count.is_none()
            && source_path.is_file()
        {
            // Ensure output directory exists
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

            // Remove existing output if present (could be a stale symlink or file)
            if output_path.exists() || output_path.symlink_metadata().is_ok() {
                let _ = std::fs::remove_file(&output_path);
            }

            // Create relative symlink for portability
            let link_dir = output_path.parent().unwrap_or(std::path::Path::new("."));
            let rel_source = veks_core::paths::relative_path(link_dir, &source_path);
            match std::os::unix::fs::symlink(&rel_source, &output_path) {
                Ok(()) => {
                    let record_str = probe.record_count
                        .map_or("unknown".to_string(), |n| n.to_string());
                    return CommandResult {
                        status: Status::Ok,
                        message: format!(
                            "symlinked {} → {} ({} records, format already {})",
                            output_path.display(), rel_source.display(), record_str, source_format.name(),
                        ),
                        produced: vec![output_path],
                        elapsed: start.elapsed(),
                    };
                }
                Err(e) => {
                    ctx.ui.log(&format!(
                        "  symlink failed ({}), falling back to copy", e
                    ));
                    // Fall through to normal import
                }
            }
        }

        // Open source
        let mut source = match reader::open_source_for_facet(&source_path, source_format, facet, threads, max_count) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open source {}: {}", source_path.display(), e), start),
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
            Err(e) => return error_result(format!("failed to open sink {}: {}", output_path.display(), e), start),
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
            ctx.ui.bar_with_unit(total, "importing records", "records")
        } else {
            ctx.ui.spinner("importing records")
        };

        // Governor checkpoint before import processing
        if ctx.governor.checkpoint() {
            log::info!("governor: throttle active");
        }

        // Import records with time-based throughput tracking and demand signaling.
        //
        // Every checkpoint_secs, we measure sustained throughput and offer
        // demand for more I/O bandwidth if the rate has been stable and
        // suggests headroom. We require several consecutive low-throughput
        // samples before signaling demand to avoid reacting to transient dips.
        let mut count: u64 = 0;
        let mut interval_bytes: u64 = 0;
        let mut interval_records: u64 = 0;
        let mut checkpoint_time = Instant::now();
        let checkpoint_secs = 5.0_f64;

        // Demand hysteresis: require consecutive_low samples before offering
        let mut consecutive_low: u32 = 0;
        let demand_threshold: u32 = 3; // 3 × 5s = 15s sustained before demand

        while let Some(data) = source.next_record() {
            if let Some(mc) = max_count {
                if count >= mc {
                    break;
                }
            }
            interval_bytes += data.len() as u64;
            interval_records += 1;
            sink.write_record(count as i64, &data);
            count += 1;
            pb.inc(1);

            let elapsed_secs = checkpoint_time.elapsed().as_secs_f64();
            if elapsed_secs >= checkpoint_secs {
                let throughput_mbps = (interval_bytes as f64 / (1024.0 * 1024.0)) / elapsed_secs;
                let records_per_sec = interval_records as f64 / elapsed_secs;

                // Heuristic: if throughput is under 500 MB/s and we have
                // I/O headroom (queue depth below saturation), more source
                // reader threads would likely help.
                let saturation_depth = ctx.governor.io_saturation_depth();
                let current_threads = threads as u64;

                if throughput_mbps < 500.0 {
                    consecutive_low += 1;
                } else {
                    consecutive_low = 0;
                }

                if consecutive_low >= demand_threshold {
                    let could_use = (current_threads * 2).max(8).min(64);
                    ctx.governor.offer_demand("threads", current_threads, could_use);
                }

                log::debug!(
                    "import: {:.0} rec/s, {:.1} MB/s, {} threads, io_sat={}{}",
                    records_per_sec, throughput_mbps, threads, saturation_depth,
                    if consecutive_low >= demand_threshold { " [demand offered]" } else { "" },
                );

                // Reset for next interval
                interval_bytes = 0;
                interval_records = 0;
                checkpoint_time = Instant::now();

                // Governor throttle check
                if ctx.governor.checkpoint() {
                    log::info!("governor: throttle active at record {}", count);
                }
            }
        }

        pb.finish();

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
                name: "facet".to_string(),
                type_name: "enum".to_string(),
                required: true,
                default: None,
                description: "Facet type (e.g. base_vectors, query_vectors)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "format".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Source format override (auto-detected if omitted)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "threads".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Number of loader threads (0 = auto)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Maximum number of records to import (all if omitted)".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output"],
        )
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

