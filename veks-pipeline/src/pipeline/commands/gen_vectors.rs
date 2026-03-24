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
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;

/// Parse a u64 with optional ISO suffixes (K, M, G/B, T, KiB, MiB, GiB, TiB).
fn parse_suffixed_u64(s: &str) -> Result<u64, String> {
    vectordata::dataset::source::parse_number_with_suffix(s)
}

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

        let output_str = match options.get("output") {
            Some(s) => s.to_string(),
            None => match options.get("append") {
                Some(s) => s.to_string(),
                None => return error_result("either 'output' or 'append' is required".into(), start),
            }
        };
        let dimension: u32 = match options.require("dimension") {
            Ok(s) => match parse_suffixed_u64(s) {
                Ok(d) if d > 0 && d <= u32::MAX as u64 => d as u32,
                Ok(_) => return error_result(format!("dimension out of range: '{}'", s), start),
                Err(e) => return error_result(format!("invalid dimension '{}': {}", s, e), start),
            },
            Err(e) => return error_result(e, start),
        };
        let count: u64 = match options.require("count") {
            Ok(s) => match parse_suffixed_u64(s) {
                Ok(n) if n > 0 => n,
                Ok(_) => return error_result(format!("count must be > 0: '{}'", s), start),
                Err(e) => return error_result(format!("invalid count '{}': {}", s, e), start),
            },
            Err(e) => return error_result(e, start),
        };

        let seed = rng::parse_seed(options.get("seed"));
        let output_path = resolve_path(&output_str, &ctx.workspace);

        // Infer element type from output extension if --type not specified
        let elem_type = if let Some(t) = options.get("type") {
            t
        } else {
            match output_path.extension().and_then(|e| e.to_str()) {
                Some("fvec") => "f32",
                Some("mvec") => "f16",
                Some("dvec") => "f64",
                Some("ivec") => "i32",
                Some("svec") => "i16",
                Some("bvec") => "u8",
                _ => "float[]", // default fallback
            }
        };

        // Validate that the output extension matches the element type
        let expected_ext = match elem_type {
            "float[]" | "f32" => "fvec",
            "half" | "f16" => "mvec",
            "double[]" | "f64" => "dvec",
            "int[]" | "i32" => "ivec",
            "short[]" | "i16" => "svec",
            "byte[]" | "u8" => "bvec",
            _ => "",
        };
        if !expected_ext.is_empty() {
            if let Some(ext) = output_path.extension().and_then(|e| e.to_str()) {
                if ext != expected_ext {
                    return error_result(
                        format!(
                            "output extension '.{}' does not match element type '{}' (expected '.{}'). \
                             Use --type {} or rename to .{}",
                            ext, elem_type, expected_ext, elem_type, expected_ext,
                        ),
                        start,
                    );
                }
            }
        }

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

        // Injection ratios for test data generation
        let zeros_ratio: f64 = options
            .get("zeros-ratio")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let duplicates_ratio: f64 = options
            .get("duplicates-ratio")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let append_path = options.get("append").map(|s| resolve_path(s, &ctx.workspace));

        // When appending, detect dimension from existing file
        let (effective_dim, open_mode) = if let Some(ref ap) = append_path {
            if !ap.exists() {
                return error_result(format!("append target does not exist: {}", ap.display()), start);
            }
            let meta = std::fs::metadata(ap).map_err(|e| format!("{}", e));
            let file_size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            if file_size < 4 {
                return error_result(format!("append target too small to read dimension: {}", ap.display()), start);
            }
            let data = std::fs::read(ap)
                .map_err(|e| format!("read append target: {}", e));
            let existing_dim = match data {
                Ok(d) => i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as u32,
                Err(e) => return error_result(e, start),
            };
            if dimension != existing_dim {
                ctx.ui.log(&format!(
                    "  NOTE: --dimension {} overridden by append target dimension {}",
                    dimension, existing_dim
                ));
            }
            (existing_dim, true)
        } else {
            (dimension, false)
        };

        // Create output directory (or prepare append)
        let actual_output = if let Some(ref ap) = append_path {
            ap.clone()
        } else {
            if let Some(parent) = output_path.parent() {
                if !parent.exists() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return error_result(format!("failed to create directory: {}", e), start);
                    }
                }
            }
            output_path.clone()
        };

        let mut rng_inst = rng::seeded_rng(seed);

        // Compute injection counts
        let zeros_count = (count as f64 * zeros_ratio).round() as u64;
        let dups_count = (count as f64 * duplicates_ratio).round() as u64;
        let normal_count = count.saturating_sub(zeros_count).saturating_sub(dups_count);

        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let record_bytes = 4 + effective_dim as u64 * match elem_type {
            "double[]" | "f64" => 8u64, "float[]" | "f32" | "int[]" | "i32" => 4,
            "half" | "f16" | "short[]" | "i16" => 2, "byte[]" | "u8" => 1, _ => 4,
        };
        let total_bytes = count * record_bytes;
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

        ctx.ui.log(&format!(
            "generate vectors: {} x dim={} ({}) → {:.1} MB, {} threads",
            count, effective_dim, elem_type, total_mb, threads,
        ));
        if zeros_count > 0 || dups_count > 0 {
            ctx.ui.log(&format!(
                "  injection: {} normal + {} zeros + {} duplicates",
                normal_count, zeros_count, dups_count,
            ));
        }
        ctx.ui.log(&format!("  output: {}", actual_output.display()));

        let pb = ctx.ui.bar_with_unit(count, "generating", "vectors");

        let fmin = float_min as f32;
        let fmax = float_max as f32;
        let frange = fmax - fmin;
        let drange = float_max - float_min;
        let d = effective_dim;

        // Single unified path: parallel chunked generation with optional
        // zero/duplicate injection. When zeros_count and dups_count are 0,
        // the alias table produces all-Normal dispositions — no overhead.
        let es: usize = match elem_type {
            "float[]" | "f32" | "int[]" | "i32" => 4,
            "half" | "f16" | "short[]" | "i16" => 2,
            "double[]" | "f64" => 8,
            "byte[]" | "u8" => 1,
            _ => return error_result(format!("unsupported element type: '{}'", elem_type), start),
        };
        let result = generate_xvec_with_injection(
            &actual_output, d, es, elem_type, count, zeros_count, dups_count, open_mode,
            fmin, frange, float_min, drange, int_min, int_max,
            &mut rng_inst, &pb,
        );

        let mode_str = if open_mode { "appended" } else { "generated" };
        match result {
            Ok(()) => CommandResult {
                status: Status::Ok,
                message: format!(
                    "{} {} {}x{} vectors to {}{}",
                    mode_str, count, effective_dim, elem_type, actual_output.display(),
                    if zeros_count > 0 || dups_count > 0 {
                        format!(" ({} zeros, {} duplicates)", zeros_count, dups_count)
                    } else {
                        String::new()
                    }
                ),
                produced: vec![actual_output],
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
                required: false,
                default: None,
                description: "Output file path (or use --append)".to_string(),
                        role: OptionRole::Output,
        },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Vector dimensionality".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of vectors to generate".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "type".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("float[]".to_string()),
                description: "Element type: float[], int[], double[], byte[], half, short[]".to_string(),
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
                name: "min".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("-1.0".to_string()),
                description: "Minimum value for float/double elements".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "max".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("1.0".to_string()),
                description: "Maximum value for float/double elements".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "zeros-ratio".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.0".to_string()),
                description: "Fraction of vectors to replace with zero vectors (0.0-1.0)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "duplicates-ratio".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.0".to_string()),
                description: "Fraction of vectors to replace with duplicates of earlier vectors (0.0-1.0)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "append".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Append to an existing vector file (matches its dimensionality)".to_string(),
                        role: OptionRole::Input,
        },
        ]
    }
}

/// Batch size for progress updates in generate loops.
const GEN_BATCH: u64 = 10_000;

/// Vector disposition: what kind of vector to write at each position.
#[derive(Clone, Copy, PartialEq)]
enum VecDisposition {
    Normal,
    Zero,
    Duplicate,
}

/// Alias table for O(1) per-vector disposition sampling (Vose's alias method).
///
/// Given probabilities for Normal, Zero, and Duplicate, this builds a table
/// that allows sampling the disposition in O(1) time with a single uniform
/// random draw.
struct AliasTable {
    prob: [f64; 3],
    alias: [VecDisposition; 3],
    labels: [VecDisposition; 3],
}

impl AliasTable {
    fn new(normal_ratio: f64, zero_ratio: f64, dup_ratio: f64) -> Self {
        let labels = [VecDisposition::Normal, VecDisposition::Zero, VecDisposition::Duplicate];
        let n = 3.0;
        let mut scaled = [normal_ratio * n, zero_ratio * n, dup_ratio * n];
        let mut prob = [0.0f64; 3];
        let mut alias = [VecDisposition::Normal; 3];

        let mut small = Vec::new();
        let mut large = Vec::new();
        for i in 0..3 {
            if scaled[i] < 1.0 { small.push(i); } else { large.push(i); }
        }

        while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
            prob[s] = scaled[s];
            alias[s] = labels[l];
            scaled[l] = (scaled[l] + scaled[s]) - 1.0;
            if scaled[l] < 1.0 { small.push(l); } else { large.push(l); }
        }
        for &i in large.iter().chain(small.iter()) {
            prob[i] = 1.0;
        }

        AliasTable { prob, alias, labels }
    }

    /// O(1) sample: one uniform random draw → disposition.
    fn sample(&self, rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> VecDisposition {
        use rand::Rng;
        let u: f64 = rng.random();
        let scaled = u * 3.0;
        let i = (scaled as usize).min(2);
        let frac = scaled - i as f64;
        if frac < self.prob[i] { self.labels[i] } else { self.alias[i] }
    }

    /// All normal (no injection).
    fn all_normal() -> Self {
        AliasTable {
            prob: [1.0, 1.0, 1.0],
            alias: [VecDisposition::Normal; 3],
            labels: [VecDisposition::Normal, VecDisposition::Normal, VecDisposition::Normal],
        }
    }
}

/// Generic generation with zero/duplicate injection, append support,
/// and parallel chunk generation.
///
/// Splits the work into chunks, generates each chunk's byte buffer in
/// parallel using per-thread RNGs (seeded deterministically from the
/// main RNG), then writes all chunks sequentially to maintain ordering.
fn generate_xvec_with_injection(
    output: &Path,
    dim: u32,
    elem_size: usize,
    elem_type: &str,
    count: u64,
    zeros_count: u64,
    dups_count: u64,
    append: bool,
    float_min: f32,
    float_range: f32,
    float_min_f64: f64,
    float_range_f64: f64,
    int_min: i32,
    int_max: i32,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &veks_core::ui::ProgressHandle,
) -> Result<(), String> {
    use std::io::Write;
    use rand::{Rng, SeedableRng};

    let record_data_size = dim as usize * elem_size;
    let record_total_size = 4 + record_data_size; // dim header + data
    let dim_header = (dim as i32).to_le_bytes();
    let zero_bytes = vec![0u8; record_data_size];

    let normal_ratio = if count > 0 {
        1.0 - (zeros_count as f64 / count as f64) - (dups_count as f64 / count as f64)
    } else { 1.0 };
    let zero_ratio = if count > 0 { zeros_count as f64 / count as f64 } else { 0.0 };
    let dup_ratio = if count > 0 { dups_count as f64 / count as f64 } else { 0.0 };

    // Single unified path: split into chunks, generate in parallel, write
    // sequentially. Progress is reported per sub-batch within each chunk
    // for smooth status updates.
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let chunk_size = ((count as usize + threads - 1) / threads).max(1);
    let num_chunks = (count as usize + chunk_size - 1) / chunk_size;

    let chunk_seeds: Vec<u64> = (0..num_chunks).map(|_| rng.random()).collect();
    let alias_table = AliasTable::new(normal_ratio.max(0.0), zero_ratio, dup_ratio);

    // Progress reporting interval: ~10K vectors per update
    let progress_interval = 10_000usize;

    let chunks: Vec<Vec<u8>> = {
        use rayon::prelude::*;
        chunk_seeds.into_par_iter().enumerate().map(|(ci, seed)| {
            let start_idx = ci * chunk_size;
            let end_idx = std::cmp::min(start_idx + chunk_size, count as usize);
            let chunk_len = end_idx - start_idx;

            let mut chunk_rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
            let mut buf = Vec::with_capacity(chunk_len * record_total_size);
            let mut records_written = 0usize;

            for j in 0..chunk_len {
                let disp = alias_table.sample(&mut chunk_rng);
                buf.extend_from_slice(&dim_header);
                match disp {
                    VecDisposition::Zero => buf.extend_from_slice(&zero_bytes),
                    VecDisposition::Duplicate if records_written > 0 => {
                        let src_idx = chunk_rng.random_range(0..records_written);
                        let src_offset = src_idx * record_total_size + 4;
                        buf.extend_from_slice(&buf[src_offset..src_offset + record_data_size].to_vec());
                    }
                    VecDisposition::Normal | VecDisposition::Duplicate => {
                        for _ in 0..dim {
                            match elem_type {
                                "f32" | "float[]" => {
                                    let v: f32 = float_min + chunk_rng.random::<f32>() * float_range;
                                    buf.extend_from_slice(&v.to_le_bytes());
                                }
                                "f16" | "half" => {
                                    let v = half::f16::from_f32(float_min + chunk_rng.random::<f32>() * float_range);
                                    buf.extend_from_slice(&v.to_le_bytes());
                                }
                                "f64" | "double[]" => {
                                    let v: f64 = float_min_f64 + chunk_rng.random::<f64>() * float_range_f64;
                                    buf.extend_from_slice(&v.to_le_bytes());
                                }
                                "i32" | "int[]" => {
                                    let v: i32 = chunk_rng.random_range(int_min..=int_max);
                                    buf.extend_from_slice(&v.to_le_bytes());
                                }
                                "i16" | "short[]" => {
                                    let v: i16 = chunk_rng.random_range(int_min as i16..=int_max as i16);
                                    buf.extend_from_slice(&v.to_le_bytes());
                                }
                                "u8" | "byte[]" => {
                                    buf.push(chunk_rng.random::<u8>());
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                records_written += 1;
                // Report progress every ~10K vectors for smooth updates
                if (j + 1) % progress_interval == 0 {
                    pb.inc(progress_interval as u64);
                }
            }
            // Report remaining vectors in this chunk
            let remaining = chunk_len % progress_interval;
            if remaining > 0 {
                pb.inc(remaining as u64);
            }
            buf
        }).collect()
    };

    // Write all chunks sequentially
    let file = if append {
        std::fs::OpenOptions::new().append(true).open(output)
    } else {
        std::fs::File::create(output)
    }.map_err(|e| format!("failed to open {}: {}", output.display(), e))?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 22, file);
    for chunk in &chunks {
        writer.write_all(chunk).map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;

    pb.set_position(count);
    Ok(())
}

/// Generate f32 vectors in fvec format.
fn generate_xvec_f32(
    output: &Path,
    dim: u32,
    count: u64,
    min: f32,
    max: f32,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    pb: &veks_core::ui::ProgressHandle,
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
    pb: &veks_core::ui::ProgressHandle,
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
    pb: &veks_core::ui::ProgressHandle,
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
    pb: &veks_core::ui::ProgressHandle,
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
    pb: &veks_core::ui::ProgressHandle,
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
    pb: &veks_core::ui::ProgressHandle,
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
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
