// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate an ivec shuffle permutation.
//!
//! Generates a deterministic Fisher-Yates shuffle of integers `[0, interval)`,
//! writing each shuffled value as a 1-dimensional ivec record. The output can
//! be used as an index array for downstream extraction steps.
//!
//! Equivalent to the Java `CMD_generate_ivecShuffle` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;

/// Pipeline command: generate ivec-shuffle permutation.
pub struct GenerateIvecShuffleOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateIvecShuffleOp)
}

impl CommandOp for GenerateIvecShuffleOp {
    fn command_path(&self) -> &str {
        "generate ivec-shuffle"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate a random ordinal shuffle mapping".into(),
            body: format!(r#"# generate ivec-shuffle

Generate a random ordinal shuffle mapping.

## Description

Generates a deterministic Fisher-Yates shuffle of integers `[0, interval)`,
writing each shuffled value as a 1-dimensional ivec record. The output can
be used as an index array for downstream extraction steps.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let interval: usize = match options.require("interval") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid interval: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let seed = rng::parse_seed(options.get("seed"));
        let output_path = resolve_path(output_str, &ctx.workspace);

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

        // Build the identity sequence [0, 1, 2, ..., interval-1]
        let mut values: Vec<i32> = (0..interval as i32).collect();

        // Fisher-Yates shuffle with the seeded PRNG
        let mut rng_inst = rng::seeded_rng(seed);
        rng::fisher_yates_shuffle(&mut values, &mut rng_inst);

        // Write as 1-dimensional ivec records
        match write_ivec_1d(&output_path, &values) {
            Ok(()) => CommandResult {
                status: Status::Ok,
                message: format!(
                    "generated shuffle permutation of {} elements to {}",
                    interval,
                    output_path.display()
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            },
            Err(e) => error_result(e, start),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output ivec file path".to_string(),
            },
            OptionDesc {
                name: "interval".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of elements to shuffle (0 to interval-1)".to_string(),
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Random seed for reproducibility".to_string(),
            },
        ]
    }
}

/// Write a vector of i32 values as 1-dimensional ivec records.
///
/// Each record: `[dim=1: i32 LE][value: i32 LE]` = 8 bytes.
fn write_ivec_1d(output: &Path, values: &[i32]) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

    let dim: i32 = 1;
    for &value in values {
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
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
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
        }
    }

    #[test]
    fn test_shuffle_output_size() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("shuffle.ivec");

        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("interval", "100");
        opts.set("seed", "42");

        let mut op = GenerateIvecShuffleOp;
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Ok);
        // Each record: 4 (dim) + 4 (value) = 8 bytes, 100 records
        let size = std::fs::metadata(&out).unwrap().len();
        assert_eq!(size, 100 * 8);
    }

    #[test]
    fn test_shuffle_is_permutation() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("shuffle.ivec");

        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("interval", "50");
        opts.set("seed", "99");

        let mut op = GenerateIvecShuffleOp;
        let mut ctx = test_ctx(tmp.path());
        op.execute(&opts, &mut ctx);

        // Read back and verify it's a permutation
        let data = std::fs::read(&out).unwrap();
        let mut values: Vec<i32> = Vec::new();
        for i in 0..50 {
            let offset = i * 8 + 4; // skip dim header
            let val = i32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            values.push(val);
        }
        let mut sorted = values.clone();
        sorted.sort();
        let expected: Vec<i32> = (0..50).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_shuffle_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let out1 = tmp.path().join("a.ivec");
        let out2 = tmp.path().join("b.ivec");

        for out in [&out1, &out2] {
            let mut opts = Options::new();
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("interval", "1000");
            opts.set("seed", "42");

            let mut op = GenerateIvecShuffleOp;
            let mut ctx = test_ctx(tmp.path());
            op.execute(&opts, &mut ctx);
        }

        let data1 = std::fs::read(&out1).unwrap();
        let data2 = std::fs::read(&out2).unwrap();
        assert_eq!(data1, data2);
    }
}
