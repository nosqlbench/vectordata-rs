// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate random vectors.
//!
//! Generates vectors with independently sampled elements from a uniform
//! distribution. Supports f32, i32, f64, u8 (byte), f16 (half), and i16
//! element types. Output format is determined by the element type.
//!
//! Equivalent to the Java `CMD_generate_vectors` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::Rng;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;

/// Pipeline command: generate random vectors.
pub struct GenerateVectorsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateVectorsOp)
}

impl CommandOp for GenerateVectorsOp {
    fn command_path(&self) -> &str {
        "generate vectors"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate synthetic random vectors".into(),
            body: format!(r#"# generate vectors

Generate synthetic random vectors.

## Description

Produces vectors whose elements are independently sampled from a uniform
distribution over a configurable range. Each generated vector is written as
a single record in the appropriate xvec binary format, determined by the
chosen element type:

- `f32` / `float[]` -- fvec (4-byte IEEE 754 floats)
- `i32` / `int[]` -- ivec (4-byte signed integers)
- `f64` / `double[]` -- dvec (8-byte IEEE 754 doubles)
- `u8` / `byte[]` -- bvec (single-byte unsigned integers, full 0-255 range)
- `f16` / `half` -- mvec (2-byte IEEE 754 half-precision floats)
- `i16` / `short[]` -- svec (2-byte signed integers)

Every element within a vector is drawn independently -- there is no
correlation between dimensions. For float and double types, elements are
uniformly distributed in `[min, max)`. For integer types, elements are
uniformly distributed in `[int-min, int-max]`.

## Deterministic generation

When a `seed` value is provided, the PRNG is initialized deterministically
so that repeated runs with the same seed, dimension, count, and type
produce byte-identical output files. This is critical for reproducible
benchmarking pipelines where upstream artifacts must be stable.

## Role in dataset pipelines

This command is the simplest way to create test datasets from scratch. It
is commonly used to fill in vector facets during synthetic dataset
generation -- for example, producing a `base_vectors.fvec` or
`query_vectors.fvec` for a new dataset layout.

Because the element values are uniformly random, the resulting vectors have
no meaningful cluster structure. For datasets that need statistically
realistic vector distributions, use `generate from-model` instead, which
samples each dimension from a fitted distribution model derived from real
data.

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
        let dimension: u32 = match options.require("dimension") {
            Ok(s) => match s.parse() {
                Ok(d) if d > 0 => d,
                _ => return error_result(format!("invalid dimension: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let count: u64 = match options.require("count") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid count: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let elem_type = options.get("type").unwrap_or("float[]");
        let seed = rng::parse_seed(options.get("seed"));
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Parse min/max ranges
        let float_min: f64 = options
            .get("min")
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1.0);
        let float_max: f64 = options
            .get("max")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let int_min: i32 = options
            .get("int-min")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let int_max: i32 = options
            .get("int-max")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

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

        let mut rng_inst = rng::seeded_rng(seed);
        let pb = ctx.ui.bar(count, "generating");

        let result = match elem_type {
            "float[]" | "f32" => {
                generate_xvec_f32(&output_path, dimension, count, float_min as f32, float_max as f32, &mut rng_inst, &pb)
            }
            "int[]" | "i32" => {
                generate_xvec_i32(&output_path, dimension, count, int_min, int_max, &mut rng_inst, &pb)
            }
            "double[]" | "f64" => {
                generate_xvec_f64(&output_path, dimension, count, float_min, float_max, &mut rng_inst, &pb)
            }
            "byte[]" | "u8" => {
                generate_xvec_u8(&output_path, dimension, count, &mut rng_inst, &pb)
            }
            "half" | "f16" => {
                generate_xvec_f16(&output_path, dimension, count, float_min as f32, float_max as f32, &mut rng_inst, &pb)
            }
            "short[]" | "i16" => {
                generate_xvec_i16(&output_path, dimension, count, int_min as i16, int_max as i16, &mut rng_inst, &pb)
            }
            _ => Err(format!("unsupported element type: '{}'. Use float[], int[], double[], byte[], half, or short[]", elem_type)),
        };

        match result {
            Ok(()) => CommandResult {
                status: Status::Ok,
                message: format!(
                    "generated {} {}x{} vectors to {}",
                    count,
                    dimension,
                    elem_type,
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
                description: "Output file path".to_string(),
            },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Vector dimensionality".to_string(),
            },
            OptionDesc {
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of vectors to generate".to_string(),
            },
            OptionDesc {
                name: "type".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("float[]".to_string()),
                description: "Element type: float[], int[], double[], byte[], half, short[]".to_string(),
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Random seed for reproducibility".to_string(),
            },
            OptionDesc {
                name: "min".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("-1.0".to_string()),
                description: "Minimum value for float/double elements".to_string(),
            },
            OptionDesc {
                name: "max".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("1.0".to_string()),
                description: "Maximum value for float/double elements".to_string(),
            },
        ]
    }
}

/// Batch size for progress updates in generate loops.
const GEN_BATCH: u64 = 10_000;

/// Generate f32 vectors in fvec format.
fn generate_xvec_f32(
    output: &Path,
    dim: u32,
    count: u64,
    min: f32,
    max: f32,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &crate::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    let range = max - min;

    for i in 0..count {
        writer
            .write_all(&(dim as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for _ in 0..dim {
            let val: f32 = min + rng.random::<f32>() * range;
            writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        if (i + 1) % GEN_BATCH == 0 { pb.set_position(i + 1); }
    }
    pb.set_position(count);
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Generate i32 vectors in ivec format.
fn generate_xvec_i32(
    output: &Path,
    dim: u32,
    count: u64,
    min: i32,
    max: i32,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &crate::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    let range = (max - min + 1) as u32;

    for i in 0..count {
        writer
            .write_all(&(dim as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for _ in 0..dim {
            let val: i32 = min + (rng.random::<u32>() % range) as i32;
            writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        if (i + 1) % GEN_BATCH == 0 { pb.set_position(i + 1); }
    }
    pb.set_position(count);
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Generate f64 vectors in dvec format.
fn generate_xvec_f64(
    output: &Path,
    dim: u32,
    count: u64,
    min: f64,
    max: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &crate::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    let range = max - min;

    for i in 0..count {
        writer
            .write_all(&(dim as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for _ in 0..dim {
            let val: f64 = min + rng.random::<f64>() * range;
            writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        if (i + 1) % GEN_BATCH == 0 { pb.set_position(i + 1); }
    }
    pb.set_position(count);
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Generate u8 vectors in bvec format.
fn generate_xvec_u8(
    output: &Path,
    dim: u32,
    count: u64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &crate::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

    for i in 0..count {
        writer
            .write_all(&(dim as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for _ in 0..dim {
            let val: u8 = rng.random();
            writer.write_all(&[val]).map_err(|e| e.to_string())?;
        }
        if (i + 1) % GEN_BATCH == 0 { pb.set_position(i + 1); }
    }
    pb.set_position(count);
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Generate f16 vectors in mvec format.
fn generate_xvec_f16(
    output: &Path,
    dim: u32,
    count: u64,
    min: f32,
    max: f32,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &crate::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    let range = max - min;

    for i in 0..count {
        writer
            .write_all(&(dim as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for _ in 0..dim {
            let val_f32: f32 = min + rng.random::<f32>() * range;
            let val_f16 = half::f16::from_f32(val_f32);
            writer
                .write_all(&val_f16.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        if (i + 1) % GEN_BATCH == 0 { pb.set_position(i + 1); }
    }
    pb.set_position(count);
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Generate i16 vectors in svec format.
fn generate_xvec_i16(
    output: &Path,
    dim: u32,
    count: u64,
    min: i16,
    max: i16,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &crate::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    let range = (max as i32 - min as i32 + 1) as u32;

    for i in 0..count {
        writer
            .write_all(&(dim as i32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for _ in 0..dim {
            let val: i16 = min + (rng.random::<u32>() % range) as i16;
            writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        if (i + 1) % GEN_BATCH == 0 { pb.set_position(i + 1); }
    }
    pb.set_position(count);
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
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_generate_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("test.fvec");

        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "10");
        opts.set("type", "float[]");
        opts.set("seed", "42");

        let mut op = GenerateVectorsOp;
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Ok);
        assert!(out.exists());

        // fvec: each record = 4 bytes (dim) + 4 * 4 bytes (elements) = 20 bytes
        let size = std::fs::metadata(&out).unwrap().len();
        assert_eq!(size, 10 * (4 + 4 * 4));
    }

    #[test]
    fn test_generate_ivec() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("test.ivec");

        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("dimension", "3");
        opts.set("count", "5");
        opts.set("type", "int[]");
        opts.set("seed", "99");

        let mut op = GenerateVectorsOp;
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Ok);
        // ivec: each record = 4 + 3 * 4 = 16 bytes
        let size = std::fs::metadata(&out).unwrap().len();
        assert_eq!(size, 5 * (4 + 3 * 4));
    }

    #[test]
    fn test_generate_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let out1 = tmp.path().join("a.fvec");
        let out2 = tmp.path().join("b.fvec");

        for out in [&out1, &out2] {
            let mut opts = Options::new();
            opts.set("output", out.to_string_lossy().to_string());
            opts.set("dimension", "8");
            opts.set("count", "100");
            opts.set("seed", "42");

            let mut op = GenerateVectorsOp;
            let mut ctx = test_ctx(tmp.path());
            op.execute(&opts, &mut ctx);
        }

        let data1 = std::fs::read(&out1).unwrap();
        let data2 = std::fs::read(&out2).unwrap();
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_generate_bvec() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("test.bvec");

        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("dimension", "16");
        opts.set("count", "3");
        opts.set("type", "byte[]");

        let mut op = GenerateVectorsOp;
        let mut ctx = test_ctx(tmp.path());
        let result = op.execute(&opts, &mut ctx);

        assert_eq!(result.status, Status::Ok);
        // bvec: each record = 4 + 16 * 1 = 20 bytes
        let size = std::fs::metadata(&out).unwrap().len();
        assert_eq!(size, 3 * (4 + 16));
    }
}
