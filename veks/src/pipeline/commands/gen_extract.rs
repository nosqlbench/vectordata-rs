// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline commands: fvec-extract, ivec-extract, and hvec-extract.
//!
//! `fvec-extract`: Extracts vectors from an fvec file using indices from an
//! ivec file. Each index in the ivec is a 1-dimensional record whose value
//! selects a vector from the fvec source.
//!
//! `ivec-extract`: Extracts a range of records from an ivec file.
//!
//! `hvec-extract`: Extracts a range of records from an hvec (half-precision
//! float16) file.
//!
//! All support range specifications in the format `[start,end)` or `start..end`
//! to select a subset of the source file.
//!
//! Equivalent to Java `CMD_generate_fvecExtract` and `CMD_generate_ivecExtract`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

// ---- fvec-extract -----------------------------------------------------------

/// Pipeline command: extract fvec vectors by ivec indices.
pub struct GenerateFvecExtractOp;

pub fn fvec_factory() -> Box<dyn CommandOp> {
    Box::new(GenerateFvecExtractOp)
}

impl CommandOp for GenerateFvecExtractOp {
    fn command_path(&self) -> &str {
        "generate fvec-extract"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Extract a subset of vectors from an fvec file".into(),
            body: format!(r#"# generate fvec-extract

Extract a subset of vectors from an fvec file.

## Description

Extracts vectors from an fvec file using indices from an ivec file. Each
index in the ivec is a 1-dimensional record whose value selects a vector
from the fvec source. Supports range specifications to select a subset.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let ivec_str = match options.require("ivec-file") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let fvec_str = match options.require("fvec-file") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let ivec_path = resolve_path(ivec_str, &ctx.workspace);
        let fvec_path = resolve_path(fvec_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Parse range
        let range = match options.get("range") {
            Some(r) => match parse_range(r) {
                Ok(rng) => rng,
                Err(e) => return error_result(format!("invalid range '{}': {}", r, e), start),
            },
            None => Range { start: 0, end: None },
        };

        // Open the ivec file (index array)
        let ivec_reader = match MmapVectorReader::<i32>::open_ivec(&ivec_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open ivec file {}: {}", ivec_path.display(), e),
                    start,
                )
            }
        };

        // Open the fvec file (source vectors)
        let fvec_reader = match MmapVectorReader::<f32>::open_fvec(&fvec_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open fvec file {}: {}", fvec_path.display(), e),
                    start,
                )
            }
        };

        let ivec_count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&ivec_reader);
        let fvec_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&fvec_reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&fvec_reader) as u32;
        let range_start = range.start;
        let range_end = range.end.unwrap_or(ivec_count);

        if range_start >= ivec_count {
            return error_result(
                format!(
                    "range start {} exceeds ivec count {}",
                    range_start, ivec_count
                ),
                start,
            );
        }
        let effective_end = std::cmp::min(range_end, ivec_count);

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        // Extract vectors
        use std::io::Write;
        let file = match std::fs::File::create(&output_path) {
            Ok(f) => f,
            Err(e) => {
                return error_result(
                    format!("failed to create {}: {}", output_path.display(), e),
                    start,
                )
            }
        };
        let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

        let mut count: u64 = 0;
        for i in range_start..effective_end {
            // Read the index from the ivec file (1-dimensional: first element is the index)
            let index_vec = match ivec_reader.get(i) {
                Ok(v) => v,
                Err(e) => {
                    return error_result(
                        format!("failed to read ivec[{}]: {}", i, e),
                        start,
                    )
                }
            };
            let index = index_vec[0] as usize;

            if index >= fvec_count {
                return error_result(
                    format!(
                        "index {} at ivec[{}] exceeds fvec count {}",
                        index, i, fvec_count
                    ),
                    start,
                );
            }

            // Read the vector from the fvec file
            let vector = match fvec_reader.get(index) {
                Ok(v) => v,
                Err(e) => {
                    return error_result(
                        format!("failed to read fvec[{}]: {}", index, e),
                        start,
                    )
                }
            };

            // Write as fvec record
            writer
                .write_all(&(dim as i32).to_le_bytes())
                .map_err(|e| e.to_string())
                .unwrap();
            let slice: &[f32] = vector.as_ref();
            for &val in slice {
                writer
                    .write_all(&val.to_le_bytes())
                    .map_err(|e| e.to_string())
                    .unwrap();
            }
            count += 1;
        }

        if let Err(e) = writer.flush() {
            return error_result(format!("failed to flush output: {}", e), start);
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "extracted {} vectors (range [{}..{})) to {}",
                count,
                range_start,
                effective_end,
                output_path.display()
            ),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "ivec-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Ivec file containing indices".to_string(),
            },
            OptionDesc {
                name: "fvec-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Fvec file containing source vectors".to_string(),
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output fvec file".to_string(),
            },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Index range: [start,end) or start..end".to_string(),
            },
        ]
    }
}

// ---- ivec-extract -----------------------------------------------------------

/// Pipeline command: extract records from an ivec file.
///
/// Supports two modes:
/// - **Index-based**: provide `index-file` (an ivec containing indices); each
///   index selects a record from the source ivec. `range` selects which
///   index-file entries to use.
/// - **Range-based**: omit `index-file`; `range` selects a contiguous range
///   of ivec records directly.
pub struct GenerateIvecExtractOp;

pub fn ivec_factory() -> Box<dyn CommandOp> {
    Box::new(GenerateIvecExtractOp)
}

impl CommandOp for GenerateIvecExtractOp {
    fn command_path(&self) -> &str {
        "generate ivec-extract"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Extract a subset of vectors from an ivec file".into(),
            body: format!(r#"# generate ivec-extract

Extract a subset of vectors from an ivec file.

## Description

Extracts records from an ivec file. Two modes:

**Index-based** (with `index-file`): Each index in the index ivec is a
1-dimensional record whose value selects a record from the source ivec.
The `range` parameter controls which entries of the index file to use.

**Range-based** (without `index-file`): Extracts a contiguous range of records
from the ivec file directly.

Supports range specifications in the format `[start,end)` or `start..end`.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let ivec_str = match options.require("ivec-file") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let ivec_path = resolve_path(ivec_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let index_path = options.get("index-file").map(|s| resolve_path(s, &ctx.workspace));

        let range = match options.get("range") {
            Some(r) => match parse_range(r) {
                Ok(rng) => rng,
                Err(e) => return error_result(format!("invalid range '{}': {}", r, e), start),
            },
            None => Range { start: 0, end: None },
        };

        // Open the source ivec file
        let ivec_reader = match MmapVectorReader::<i32>::open_ivec(&ivec_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open ivec file {}: {}", ivec_path.display(), e),
                    start,
                )
            }
        };

        let ivec_count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&ivec_reader);
        let dim = <MmapVectorReader<i32> as VectorReader<i32>>::dim(&ivec_reader) as u32;

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        use std::io::Write;
        let file = match std::fs::File::create(&output_path) {
            Ok(f) => f,
            Err(e) => {
                return error_result(
                    format!("failed to create {}: {}", output_path.display(), e),
                    start,
                )
            }
        };
        let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

        if let Some(ref idx_p) = index_path {
            // Index-based extraction: read indices from index-file, look up records in source ivec
            let idx_reader = match MmapVectorReader::<i32>::open_ivec(idx_p) {
                Ok(r) => r,
                Err(e) => {
                    return error_result(
                        format!("failed to open index file {}: {}", idx_p.display(), e),
                        start,
                    )
                }
            };

            let idx_count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&idx_reader);
            let range_start = range.start;
            let range_end = range.end.unwrap_or(idx_count);

            if range_start >= idx_count {
                return error_result(
                    format!("range start {} exceeds index-file count {}", range_start, idx_count),
                    start,
                );
            }
            let effective_end = std::cmp::min(range_end, idx_count);

            let mut count: u64 = 0;
            for i in range_start..effective_end {
                let index_vec = match idx_reader.get(i) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_result(
                            format!("failed to read index-file[{}]: {}", i, e),
                            start,
                        )
                    }
                };
                let index = index_vec[0] as usize;

                if index >= ivec_count {
                    return error_result(
                        format!(
                            "index {} at index-file[{}] exceeds ivec count {}",
                            index, i, ivec_count
                        ),
                        start,
                    );
                }

                let vec = match ivec_reader.get(index) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_result(
                            format!("failed to read ivec[{}]: {}", index, e),
                            start,
                        )
                    }
                };

                writer.write_all(&(dim as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                let slice: &[i32] = vec.as_ref();
                for &val in slice {
                    writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                }
                count += 1;
            }

            if let Err(e) = writer.flush() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            CommandResult {
                status: Status::Ok,
                message: format!(
                    "extracted {} ivec records by index (range [{}..{})) to {}",
                    count, range_start, effective_end, output_path.display()
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            }
        } else {
            // Range-based extraction: contiguous range from ivec
            let range_start = range.start;
            let range_end = range.end.unwrap_or(ivec_count);
            let effective_end = std::cmp::min(range_end, ivec_count);

            if range_start >= ivec_count {
                return error_result(
                    format!(
                        "range start {} exceeds ivec count {}",
                        range_start, ivec_count
                    ),
                    start,
                );
            }

            let mut count: u64 = 0;
            for i in range_start..effective_end {
                let vec = match ivec_reader.get(i) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_result(
                            format!("failed to read ivec[{}]: {}", i, e),
                            start,
                        )
                    }
                };

                writer.write_all(&(dim as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                let slice: &[i32] = vec.as_ref();
                for &val in slice {
                    writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                }
                count += 1;
            }

            if let Err(e) = writer.flush() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            CommandResult {
                status: Status::Ok,
                message: format!(
                    "extracted {} ivec records (range [{}..{})) to {}",
                    count, range_start, effective_end, output_path.display()
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "ivec-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source ivec file".to_string(),
            },
            OptionDesc {
                name: "index-file".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Ivec file containing indices (enables index-based extraction)".to_string(),
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output ivec file".to_string(),
            },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Range: [start,end) or start..end. Applies to index-file entries (index mode) or ivec records (range mode)".to_string(),
            },
        ]
    }
}

// ---- hvec-extract -----------------------------------------------------------

/// Pipeline command: extract records from an hvec (f16) file.
///
/// Supports two modes:
/// - **Index-based**: provide `ivec-file` containing indices; each index selects
///   a vector from the hvec source. `range` selects which ivec entries to use.
/// - **Range-based**: omit `ivec-file`; `range` selects a contiguous range of
///   hvec records directly.
pub struct GenerateHvecExtractOp;

pub fn hvec_factory() -> Box<dyn CommandOp> {
    Box::new(GenerateHvecExtractOp)
}

impl CommandOp for GenerateHvecExtractOp {
    fn command_path(&self) -> &str {
        "generate hvec-extract"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Extract a subset of vectors from an hvec file".into(),
            body: format!(r#"# generate hvec-extract

Extract a subset of vectors from an hvec file.

## Description

Extracts vectors from an hvec (half-precision float16) file. Two modes:

**Index-based** (with `ivec-file`): Each index in the ivec is a 1-dimensional
record whose value selects a vector from the hvec source. The `range` parameter
controls which entries of the ivec to use.

**Range-based** (without `ivec-file`): Extracts a contiguous range of records
from the hvec file directly.

Supports range specifications in the format `[start,end)` or `start..end`.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let hvec_str = match options.require("hvec-file") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let hvec_path = resolve_path(hvec_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let ivec_path = options.get("ivec-file").map(|s| resolve_path(s, &ctx.workspace));

        let range = match options.get("range") {
            Some(r) => match parse_range(r) {
                Ok(rng) => rng,
                Err(e) => return error_result(format!("invalid range '{}': {}", r, e), start),
            },
            None => Range { start: 0, end: None },
        };

        // Open the hvec file
        let hvec_reader = match MmapVectorReader::<half::f16>::open_hvec(&hvec_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open hvec file {}: {}", hvec_path.display(), e),
                    start,
                )
            }
        };

        let hvec_count =
            <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&hvec_reader);
        let dim =
            <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&hvec_reader) as u32;

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        use std::io::Write;
        let file = match std::fs::File::create(&output_path) {
            Ok(f) => f,
            Err(e) => {
                return error_result(
                    format!("failed to create {}: {}", output_path.display(), e),
                    start,
                )
            }
        };
        let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

        if let Some(ref ivec_p) = ivec_path {
            // Index-based extraction: read indices from ivec, look up vectors in hvec
            let ivec_reader = match MmapVectorReader::<i32>::open_ivec(ivec_p) {
                Ok(r) => r,
                Err(e) => {
                    return error_result(
                        format!("failed to open ivec file {}: {}", ivec_p.display(), e),
                        start,
                    )
                }
            };

            let ivec_count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&ivec_reader);
            let range_start = range.start;
            let range_end = range.end.unwrap_or(ivec_count);

            if range_start >= ivec_count {
                return error_result(
                    format!("range start {} exceeds ivec count {}", range_start, ivec_count),
                    start,
                );
            }
            let effective_end = std::cmp::min(range_end, ivec_count);

            let mut count: u64 = 0;
            for i in range_start..effective_end {
                let index_vec = match ivec_reader.get(i) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_result(
                            format!("failed to read ivec[{}]: {}", i, e),
                            start,
                        )
                    }
                };
                let index = index_vec[0] as usize;

                if index >= hvec_count {
                    return error_result(
                        format!(
                            "index {} at ivec[{}] exceeds hvec count {}",
                            index, i, hvec_count
                        ),
                        start,
                    );
                }

                let vector = match hvec_reader.get(index) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_result(
                            format!("failed to read hvec[{}]: {}", index, e),
                            start,
                        )
                    }
                };

                writer.write_all(&(dim as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                let slice: &[half::f16] = vector.as_ref();
                for &val in slice {
                    writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                }
                count += 1;
            }

            if let Err(e) = writer.flush() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            CommandResult {
                status: Status::Ok,
                message: format!(
                    "extracted {} hvec vectors by index (ivec range [{}..{})) to {}",
                    count, range_start, effective_end, output_path.display()
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            }
        } else {
            // Range-based extraction: contiguous range from hvec
            let range_start = range.start;
            let range_end = range.end.unwrap_or(hvec_count);
            let effective_end = std::cmp::min(range_end, hvec_count);

            if range_start >= hvec_count {
                return error_result(
                    format!(
                        "range start {} exceeds hvec count {}",
                        range_start, hvec_count
                    ),
                    start,
                );
            }

            let mut count: u64 = 0;
            for i in range_start..effective_end {
                let vec = match hvec_reader.get(i) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_result(
                            format!("failed to read hvec[{}]: {}", i, e),
                            start,
                        )
                    }
                };

                writer.write_all(&(dim as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                let slice: &[half::f16] = vec.as_ref();
                for &val in slice {
                    writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                }
                count += 1;
            }

            if let Err(e) = writer.flush() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            CommandResult {
                status: Status::Ok,
                message: format!(
                    "extracted {} hvec records (range [{}..{})) to {}",
                    count, range_start, effective_end, output_path.display()
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "hvec-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source hvec file".to_string(),
            },
            OptionDesc {
                name: "ivec-file".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Ivec file containing indices (enables index-based extraction)".to_string(),
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output hvec file".to_string(),
            },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Range: [start,end) or start..end. Applies to ivec entries (index mode) or hvec records (range mode)".to_string(),
            },
        ]
    }
}

// ---- Range parsing ----------------------------------------------------------

/// Parsed range with inclusive start and exclusive end.
struct Range {
    start: usize,
    end: Option<usize>,
}

/// Parse a number with optional unit suffix, returning usize.
fn parse_range_number(s: &str) -> Result<usize, String> {
    let v = dataset::source::parse_number_with_suffix(s)?;
    usize::try_from(v).map_err(|_| format!("value too large for usize: {}", v))
}

/// Parse a range specification.
///
/// Supported formats:
/// - `[start,end)` — inclusive start, exclusive end (Java interval notation)
/// - `start..end` — Rust-style exclusive end
/// - `start` — from start to end of file
///
/// Symbolic open-ended ranges (unit suffixes supported in all positions):
/// - `[10k..]` — from 10,000 to end of file
/// - `[..10k)` — first 10k elements
/// - `[..10k]` — first 10,001 elements (inclusive end)
/// - `(10k..]` — from 10,001 to end of file (exclusive start)
fn parse_range(s: &str) -> Result<Range, String> {
    let s = s.trim();

    // Detect bracket types for inclusive/exclusive semantics
    let left_exclusive = s.starts_with('(');
    let right_inclusive = s.ends_with(']');
    let has_left_bracket = s.starts_with('[') || s.starts_with('(');
    let has_right_bracket = s.ends_with(')') || s.ends_with(']');

    // Strip brackets
    let inner = if has_left_bracket { &s[1..] } else { s };
    let inner = if has_right_bracket {
        &inner[..inner.len() - 1]
    } else {
        inner
    };
    let inner = inner.trim();

    // Try comma separator first (Java interval notation), then '..'
    let sep = if inner.contains(',') {
        ","
    } else if inner.contains("..") {
        ".."
    } else {
        // Single value: start only (or shorthand for 0..N)
        let val = parse_range_number(inner)?;
        return Ok(Range {
            start: val,
            end: None,
        });
    };

    let (left, right) = inner.split_once(sep).unwrap();
    let left = left.trim();
    let right = right.trim();

    let mut start = if left.is_empty() {
        0
    } else {
        parse_range_number(left)?
    };
    let mut end = if right.is_empty() {
        None
    } else {
        Some(parse_range_number(right)?)
    };

    // Exclusive start '(' → skip one more element
    if left_exclusive && !left.is_empty() {
        start = start.checked_add(1).ok_or("start overflow")?;
    }
    // Inclusive end ']' → include the boundary element
    if right_inclusive {
        if let Some(e) = end {
            end = Some(e.checked_add(1).ok_or("end overflow")?);
        }
    }

    Ok(Range { start, end })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_range_java_interval() {
        let r = parse_range("[100000,1000000)").unwrap();
        assert_eq!(r.start, 100000);
        assert_eq!(r.end, Some(1000000));
    }

    #[test]
    fn test_parse_range_rust_style() {
        let r = parse_range("0..100").unwrap();
        assert_eq!(r.start, 0);
        assert_eq!(r.end, Some(100));
    }

    #[test]
    fn test_parse_range_start_only() {
        let r = parse_range("500").unwrap();
        assert_eq!(r.start, 500);
        assert_eq!(r.end, None);
    }

    #[test]
    fn test_parse_range_with_spaces() {
        let r = parse_range("[ 100 , 200 )").unwrap();
        assert_eq!(r.start, 100);
        assert_eq!(r.end, Some(200));
    }

    #[test]
    fn test_parse_range_invalid() {
        assert!(parse_range("[abc,100)").is_err());
    }

    #[test]
    fn test_parse_range_with_suffixes() {
        let r = parse_range("[0,10K)").unwrap();
        assert_eq!(r.start, 0);
        assert_eq!(r.end, Some(10_000));
    }

    #[test]
    fn test_parse_range_open_right() {
        let r = parse_range("[10k..]").unwrap();
        assert_eq!(r.start, 10_000);
        assert_eq!(r.end, None);
    }

    #[test]
    fn test_parse_range_open_left_exclusive() {
        let r = parse_range("[..10k)").unwrap();
        assert_eq!(r.start, 0);
        assert_eq!(r.end, Some(10_000));
    }

    #[test]
    fn test_parse_range_open_left_inclusive() {
        let r = parse_range("[..10k]").unwrap();
        assert_eq!(r.start, 0);
        assert_eq!(r.end, Some(10_001));
    }

    #[test]
    fn test_parse_range_exclusive_start() {
        let r = parse_range("(10k..]").unwrap();
        assert_eq!(r.start, 10_001);
        assert_eq!(r.end, None);
    }

    #[test]
    fn test_parse_range_all() {
        let r = parse_range("[..]").unwrap();
        assert_eq!(r.start, 0);
        assert_eq!(r.end, None);
    }

    #[test]
    fn test_hvec_extract_range() {
        use crate::pipeline::command::StreamContext;
        use crate::pipeline::progress::ProgressLog;
        use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
        use indexmap::IndexMap;

        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        let mut ctx = StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
        };

        // Generate 50 f16 vectors of dimension 8
        let hvec_path = workspace.join("source.hvec");
        let mut opts = Options::new();
        opts.set("output", hvec_path.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "50");
        opts.set("seed", "42");
        opts.set("type", "f16");
        let mut gen_op = GenerateVectorsOp;
        let r = gen_op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Extract first 10 records
        let out_path = workspace.join("extracted.hvec");
        let mut opts = Options::new();
        opts.set("hvec-file", hvec_path.to_string_lossy().to_string());
        opts.set("output", out_path.to_string_lossy().to_string());
        opts.set("range", "[0,10)");
        let mut ext = GenerateHvecExtractOp;
        let r = ext.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // 10 records of dim 8 (f16=2 bytes): 10 * (4 + 8*2) = 200 bytes
        let size = std::fs::metadata(&out_path).unwrap().len();
        assert_eq!(size, 10 * (4 + 8 * 2));

        // Verify extracted vectors match originals
        let orig = MmapVectorReader::<half::f16>::open_hvec(&hvec_path).unwrap();
        let extracted = MmapVectorReader::<half::f16>::open_hvec(&out_path).unwrap();
        assert_eq!(
            <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&extracted),
            10
        );
        for i in 0..10 {
            let o = orig.get(i).unwrap();
            let e = extracted.get(i).unwrap();
            let o_slice: &[half::f16] = o.as_ref();
            let e_slice: &[half::f16] = e.as_ref();
            assert_eq!(o_slice, e_slice);
        }
    }

    #[test]
    fn test_hvec_extract_by_index() {
        use crate::pipeline::command::StreamContext;
        use crate::pipeline::progress::ProgressLog;
        use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
        use crate::pipeline::commands::gen_shuffle::GenerateIvecShuffleOp;
        use indexmap::IndexMap;

        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        let mut ctx = StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
        };

        // Generate 20 f16 vectors of dimension 8
        let hvec_path = workspace.join("source.hvec");
        let mut opts = Options::new();
        opts.set("output", hvec_path.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "20");
        opts.set("seed", "42");
        opts.set("type", "f16");
        let mut gen_op = GenerateVectorsOp;
        let r = gen_op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Generate shuffle of 20 elements
        let ivec_path = workspace.join("shuffle.ivec");
        let mut opts = Options::new();
        opts.set("output", ivec_path.to_string_lossy().to_string());
        opts.set("interval", "20");
        opts.set("seed", "99");
        let mut shuf_op = GenerateIvecShuffleOp;
        let r = shuf_op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Extract first 5 vectors using shuffle indices
        let out_path = workspace.join("extracted.hvec");
        let mut opts = Options::new();
        opts.set("hvec-file", hvec_path.to_string_lossy().to_string());
        opts.set("ivec-file", ivec_path.to_string_lossy().to_string());
        opts.set("output", out_path.to_string_lossy().to_string());
        opts.set("range", "[0,5)");
        let mut ext = GenerateHvecExtractOp;
        let r = ext.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // 5 records of dim 8 (f16=2 bytes): 5 * (4 + 8*2) = 100 bytes
        let size = std::fs::metadata(&out_path).unwrap().len();
        assert_eq!(size, 5 * (4 + 8 * 2));

        // Verify extracted vectors match shuffled originals
        let orig = MmapVectorReader::<half::f16>::open_hvec(&hvec_path).unwrap();
        let extracted = MmapVectorReader::<half::f16>::open_hvec(&out_path).unwrap();
        let shuffle = MmapVectorReader::<i32>::open_ivec(&ivec_path).unwrap();
        assert_eq!(
            <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&extracted),
            5
        );
        for i in 0..5 {
            let shuf_idx = shuffle.get(i).unwrap()[0] as usize;
            let o = orig.get(shuf_idx).unwrap();
            let e = extracted.get(i).unwrap();
            let o_slice: &[half::f16] = o.as_ref();
            let e_slice: &[half::f16] = e.as_ref();
            assert_eq!(o_slice, e_slice, "mismatch at extracted[{}] (orig[{}])", i, shuf_idx);
        }
    }

    #[test]
    fn test_fvec_extract_roundtrip() {
        use crate::pipeline::command::StreamContext;
        use crate::pipeline::progress::ProgressLog;
        use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
        use crate::pipeline::commands::gen_shuffle::GenerateIvecShuffleOp;
        use indexmap::IndexMap;

        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();

        let mut ctx = StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
        };

        // Generate 20 vectors of dimension 4
        let fvec_path = workspace.join("source.fvec");
        let mut opts = Options::new();
        opts.set("output", fvec_path.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "20");
        opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        let r = gen_op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Generate shuffle of 20 elements
        let ivec_path = workspace.join("shuffle.ivec");
        let mut opts = Options::new();
        opts.set("output", ivec_path.to_string_lossy().to_string());
        opts.set("interval", "20");
        opts.set("seed", "99");
        let mut shuf_op = GenerateIvecShuffleOp;
        let r = shuf_op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Extract first 10 vectors using the shuffle indices
        let out_path = workspace.join("extracted.fvec");
        let mut opts = Options::new();
        opts.set("ivec-file", ivec_path.to_string_lossy().to_string());
        opts.set("fvec-file", fvec_path.to_string_lossy().to_string());
        opts.set("output", out_path.to_string_lossy().to_string());
        opts.set("range", "[0,10)");
        let mut ext = GenerateFvecExtractOp;
        let r = ext.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Output should have 10 records of dim 4: 10 * (4 + 4*4) = 200 bytes
        let size = std::fs::metadata(&out_path).unwrap().len();
        assert_eq!(size, 10 * (4 + 4 * 4));
    }
}
