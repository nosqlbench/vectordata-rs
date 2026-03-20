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
                r#"# analyze describe

Describe vector file format and dimensions.

## Description

The analyze describe command inspects a vector file and reports key metadata
about its structure. It probes the file without reading all records, making it
fast even for very large files.

The reported fields include:

- **format** -- the detected or overridden vector format (fvec, ivec, bvec,
  npy, slab, xvec, etc.)
- **dimension** -- the number of elements per vector record
- **element size** -- the byte width of each element (e.g. 4 for f32, 2 for
  f16, 1 for u8)
- **record count** -- the total number of vector records in the file, when
  determinable from the file size and format header
- **file size** -- derived from the underlying file metadata

When the `format` option is omitted, the command auto-detects the format from
the file extension or header bytes. You can override this with an explicit
format string if the file has a non-standard extension.

## Data Preparation Role

Describe serves as a debugging and validation tool within dataset preparation
pipelines. After an import or extract step produces an output file, a describe
step can confirm that the file has the expected format, dimension, element size,
and record count. This catches problems early -- such as a dimension mismatch or
a truncated file -- before downstream steps consume the data.

It is also useful outside of pipelines for quick inspection of any vector file
on disk, answering questions like "how many vectors are in this file?" or "what
element type does this dataset use?"

## Notes

- The probe reads only the file header and metadata; it does not scan all
  records, so it completes in constant time regardless of file size.
- For slab files, the record count is derived from the slab page index rather
  than the raw file size.
- If the file is corrupt or truncated, the probe will report an error rather
  than returning partial metadata.

## Options

{}"#,
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
