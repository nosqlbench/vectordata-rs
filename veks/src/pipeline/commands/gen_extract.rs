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

/// Pipeline command: extract a range of records from an ivec file.
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

Extracts a range of records from an ivec file. Supports range specifications
in the format `[start,end)` or `start..end` to select a subset of the source file.

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

        let range = match options.get("range") {
            Some(r) => match parse_range(r) {
                Ok(rng) => rng,
                Err(e) => return error_result(format!("invalid range '{}': {}", r, e), start),
            },
            None => Range { start: 0, end: None },
        };

        // Open the ivec file
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

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        // Extract records
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
            let vec = match ivec_reader.get(i) {
                Ok(v) => v,
                Err(e) => {
                    return error_result(
                        format!("failed to read ivec[{}]: {}", i, e),
                        start,
                    )
                }
            };

            writer
                .write_all(&(dim as i32).to_le_bytes())
                .map_err(|e| e.to_string())
                .unwrap();
            let slice: &[i32] = vec.as_ref();
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
                "extracted {} ivec records (range [{}..{})) to {}",
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
                description: "Source ivec file".to_string(),
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
                description: "Record range: [start,end) or start..end".to_string(),
            },
        ]
    }
}

// ---- hvec-extract -----------------------------------------------------------

/// Pipeline command: extract a range of records from an hvec (f16) file.
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

Extracts a range of records from an hvec (half-precision float16) file.
Supports range specifications in the format `[start,end)` or `start..end`
to select a subset of the source file.

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

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        // Extract records
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
            let vec = match hvec_reader.get(i) {
                Ok(v) => v,
                Err(e) => {
                    return error_result(
                        format!("failed to read hvec[{}]: {}", i, e),
                        start,
                    )
                }
            };

            writer
                .write_all(&(dim as i32).to_le_bytes())
                .map_err(|e| e.to_string())
                .unwrap();
            let slice: &[half::f16] = vec.as_ref();
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
                "extracted {} hvec records (range [{}..{})) to {}",
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
                name: "hvec-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source hvec file".to_string(),
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
                description: "Record range: [start,end) or start..end".to_string(),
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

/// Parse a range specification.
///
/// Supported formats:
/// - `[start,end)` — inclusive start, exclusive end (Java interval notation)
/// - `start..end` — Rust-style exclusive end
/// - `start` — from start to end of file
fn parse_range(s: &str) -> Result<Range, String> {
    let s = s.trim();

    // Java interval notation: [start,end)
    if s.starts_with('[') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() != 2 {
            return Err("expected [start,end)".to_string());
        }
        let start: usize = parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("invalid start: '{}'", parts[0].trim()))?;
        let end: usize = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("invalid end: '{}'", parts[1].trim()))?;
        return Ok(Range {
            start,
            end: Some(end),
        });
    }

    // Rust-style: start..end
    if let Some((left, right)) = s.split_once("..") {
        let start: usize = left
            .trim()
            .parse()
            .map_err(|_| format!("invalid start: '{}'", left.trim()))?;
        let end: usize = right
            .trim()
            .parse()
            .map_err(|_| format!("invalid end: '{}'", right.trim()))?;
        return Ok(Range {
            start,
            end: Some(end),
        });
    }

    // Single value: start only
    let start: usize = s
        .parse()
        .map_err(|_| format!("invalid range: '{}'", s))?;
    Ok(Range { start, end: None })
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
        assert!(parse_range("[100,)").is_err());
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
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
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

    // Integration tests for extract require actual fvec/ivec files on disk.
    // The generate commands produce these, so end-to-end testing is done
    // via pipeline integration tests.

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
            workspace: workspace.to_path_buf(),
            scratch: workspace.join(".scratch"),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
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
