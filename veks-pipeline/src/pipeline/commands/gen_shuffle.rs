// Copyright (c) Jonathan Shook
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

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use crate::pipeline::rng;

/// Pipeline command: generate ivec-shuffle permutation.
pub struct GenerateIvecShuffleOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateIvecShuffleOp)
}

impl CommandOp for GenerateIvecShuffleOp {
    fn command_path(&self) -> &str {
        "generate shuffle"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate a random ordinal shuffle mapping".into(),
            body: format!(r#"# generate shuffle

Generate a random ordinal shuffle mapping.

## Description

Creates a random permutation of the integers `[0, interval)` using the
Fisher-Yates (Knuth) shuffle algorithm and writes the result as a
1-dimensional ivec file. Each record in the output contains a single i32
value: record `i` holds the shuffled ordinal that was originally at
position `i`. The output file is exactly `interval * 8` bytes (4 bytes
for the dimension header `1` plus 4 bytes for the value, per record).

## Deterministic generation

Given the same `seed` and `interval`, the shuffle is fully deterministic
and produces a byte-identical output file. This property is critical for
reproducible train/test splits -- every downstream artifact derived from
the shuffle will be stable across runs.

## Role in dataset pipelines

The shuffle ivec is the foundation of **self-search dataset construction**.
In a self-search dataset the query vectors are drawn from the same corpus
as the base vectors; the corpus must be split into disjoint query and base
sets with no overlap. The standard pattern is:

1. **Shuffle** -- `generate ivec-shuffle` produces a random permutation of
   all corpus ordinals.
2. **Extract queries** -- an extract command (e.g. `transform fvec-extract`
   or `transform mvec-extract`) reads the first K entries of the shuffle
   ivec and copies the corresponding source vectors into `query_vectors`.
3. **Extract base** -- a second extract command reads the remaining entries
   (from K to the end) and copies them into `base_vectors`.

Because the shuffle is a true permutation, every corpus vector appears in
exactly one of the two output sets. The `range` option on extract commands
controls which portion of the shuffle ivec to consume -- `[0,K)` for
queries and `[K,N)` for the base set.

The same shuffle ivec is also applied to metadata slabs (via
`transform slab-extract`) so that ordinal correspondence is maintained:
`base_metadata.slab[i]` describes `base_vectors[i]`.

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
        let ordinals_path = options.get("ordinals").map(|s| resolve_path(s, &ctx.workspace));
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

        // Build the value sequence to shuffle.
        // When an ordinals file is provided (e.g., clean_ordinals.ivec),
        // shuffle those actual ordinals so the output contains source-
        // level ordinals. Otherwise, generate [0, interval).
        let mut values: Vec<i32> = if let Some(ref ord_path) = ordinals_path {
            ctx.ui.log(&format!("  loading ordinals from {}", ord_path.display()));
            let data = match std::fs::read(ord_path) {
                Ok(d) => d,
                Err(e) => return error_result(format!("failed to read ordinals: {}", e), start),
            };
            // Parse dim=1 ivec records
            let mut ords = Vec::with_capacity(interval);
            let mut pos = 0;
            while pos + 8 <= data.len() && ords.len() < interval {
                // skip dim header (always 1)
                pos += 4;
                let val = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                ords.push(val);
                pos += 4;
            }
            if ords.len() != interval {
                ctx.ui.log(&format!("  warning: ordinals file has {} records, expected {}",
                    ords.len(), interval));
            }
            ords
        } else {
            (0..interval as i32).collect()
        };

        // Fisher-Yates shuffle with the seeded PRNG, with progress
        let pb = ctx.ui.bar(interval as u64, "shuffle");
        let mut rng_inst = rng::seeded_rng(seed);
        {
            use rand::Rng;
            let batch = 10_000.max(interval / 1000);
            for i in (1..interval).rev() {
                let j = rng_inst.random_range(0..=i);
                values.swap(i, j);
                if i % batch == 0 {
                    pb.set_position((interval - i) as u64);
                }
            }
            pb.set_position(interval as u64);
        }
        pb.finish();

        // Write as 1-dimensional ivec records
        let write_pb = ctx.ui.bar(values.len() as u64, "write");
        match write_ivec_1d(&output_path, &values, &write_pb) {
            Ok(()) => {
                // Write verified count for the bound checker
                let var_name = format!("verified_count:{}",
                    output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
                let _ = crate::pipeline::variables::set_and_save(
                    &ctx.workspace, &var_name, &interval.to_string());
                ctx.defaults.insert(var_name, interval.to_string());

                CommandResult {
                    status: Status::Ok,
                    message: format!(
                        "generated shuffle permutation of {} elements to {}",
                        interval,
                        output_path.display()
                    ),
                    produced: vec![output_path],
                    elapsed: start.elapsed(),
                }
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
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "interval".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of elements to shuffle (0 to interval-1)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Random seed for reproducibility".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "ordinals".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Input ordinals ivec to shuffle (default: generate [0,interval))".to_string(),
                role: OptionRole::Input,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &[],
            &["output"],
        )
    }
}

/// Write a vector of i32 values as 1-dimensional ivec records.
///
/// Each record: `[dim=1: i32 LE][value: i32 LE]` = 8 bytes.
fn write_ivec_1d(output: &Path, values: &[i32], pb: &veks_core::ui::ProgressHandle) -> Result<(), String> {
    use std::io::Write;
    let mut writer = AtomicWriter::new(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;

    let dim: i32 = 1;
    let batch = 10_000.max(values.len() / 1000);
    for (i, &value) in values.iter().enumerate() {
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|e| e.to_string())?;
        if (i + 1) % batch == 0 {
            pb.inc(batch as u64);
        }
    }
    pb.set_position(values.len() as u64);
    writer.finish().map_err(|e| e.to_string())?;
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
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
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
