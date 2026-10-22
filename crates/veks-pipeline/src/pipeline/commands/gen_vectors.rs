// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate random vectors.
//!
//! Generates vectors with independently sampled elements from a uniform
//! distribution. Supports f32, i32, f64, u8 (byte), f16 (half), and i16
//! element types. Output format is determined by the element type.
//!
//! Equivalent to the Java `CMD_generate_vectors` command.

use crate::pipeline::element_type::ElementTypeExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::Rng;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::element_type::ElementType;
use crate::pipeline::rng;

/// Parse a u64 with optional ISO suffixes (K, M, G/B, T, KiB, MiB, GiB, TiB).
fn parse_suffixed_u64(s: &str) -> Result<u64, String> {
    vectordata::dataset::source::parse_number_with_suffix(s)
}

/// Pipeline command: generate random vectors.
pub struct GenerateVectorsOp;

/// Creates a boxed [`GenerateVectorsOp`]; registered as the `generate vectors` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateVectorsOp)
}

impl CommandOp for GenerateVectorsOp {
    fn command_path(&self) -> &str {
        "generate vectors"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

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
        let inferred;
        let elem_type = if let Some(t) = options.get("type") {
            t
        } else {
            match ElementType::from_path(&output_path) {
                Ok(et) => { inferred = et.to_string(); &inferred }
                Err(_) => "float[]", // default fallback
            }
        };

        // Validate that the output extension matches the element type.
        // Every spelling of the type's uniform-vector format is accepted
        // (`.fvec`, `.fvecs`, `.f32vecs`, …): the comparison goes through
        // the format enum, which canonicalizes them all.
        let gen_type = generated_element_type(elem_type);
        if let Some(et) = gen_type
            && let Some(expected_fmt) = veks_io::VecFormat::of(veks_io::Container::Vec, et)
            && let Some(ext) = output_path.extension().and_then(|e| e.to_str())
        {
            let actual_fmt = veks_io::VecFormat::from_extension(ext);
            let expected_ext = expected_fmt.preferred_extension();
            if actual_fmt != Some(expected_fmt) {
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
            let existing = match veks_io::probe(ap) {
                Ok(meta) => meta,
                Err(e) => return error_result(format!("read append target: {}", e), start),
            };
            // Records of another element type would corrupt the target.
            if let (Some(have), Some(want)) = (existing.element_type, gen_type)
                && have != want
            {
                return error_result(
                    format!("append target {} holds {} elements, not '{}'", ap.display(), have, elem_type),
                    start,
                );
            }
            let existing_dim = existing.dimension;
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
            if let Some(parent) = output_path.parent()
                && !parent.exists()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                return error_result(format!("failed to create directory: {}", e), start);
            }
            output_path.clone()
        };

        let mut rng_inst = rng::seeded_rng(seed);

        // Compute injection counts
        let zeros_count = (count as f64 * zeros_ratio).round() as u64;
        let dups_count = (count as f64 * duplicates_ratio).round() as u64;
        let normal_count = count.saturating_sub(zeros_count).saturating_sub(dups_count);

        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let Some(gen_type) = gen_type else {
            return error_result(format!("unsupported element type: '{}'", elem_type), start);
        };
        let record_bytes = match i32::try_from(effective_dim)
            .map_err(|_| format!("dimension {} is not 1 to {}", effective_dim, i32::MAX))
            .and_then(|d| veks_io::reader::xvec_record_bytes(d, gen_type).map_err(|e| e.to_string()))
        {
            Ok(b) => b as u64,
            Err(e) => return error_result(e, start),
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
        // The type is dispatched once here; each generator is monomorphic.
        macro_rules! generate {
            ($t:ty, $sample:expr) => {
                generate_xvec_with_injection::<$t>(
                    &actual_output, d, count, zeros_count, dups_count, open_mode,
                    &mut rng_inst, &pb, $sample,
                )
            };
        }
        use veks_io::ElementType as Et;
        let result = match gen_type {
            Et::F32 => generate!(f32, move |r: &mut Xoshiro| fmin + r.random::<f32>() * frange),
            Et::F16 => generate!(veks_io::f16, move |r: &mut Xoshiro| {
                veks_io::f16::from_f32(fmin + r.random::<f32>() * frange)
            }),
            Et::F64 => generate!(f64, move |r: &mut Xoshiro| float_min + r.random::<f64>() * drange),
            Et::I32 => generate!(i32, move |r: &mut Xoshiro| r.random_range(int_min..=int_max)),
            Et::I16 => generate!(i16, move |r: &mut Xoshiro| {
                r.random_range(int_min as i16..=int_max as i16)
            }),
            Et::U8 => generate!(u8, move |r: &mut Xoshiro| r.random::<u8>()),
            _ => return error_result(format!("unsupported element type: '{}'", elem_type), start),
        };

        let mode_str = if open_mode { "appended" } else { "generated" };
        match result {
            Ok(()) => {
                // Write verified count for the bound checker
                let var_name = format!("verified_count:{}",
                    actual_output.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
                let _ = crate::pipeline::variables::set_and_save(
                    &ctx.workspace, &var_name, &count.to_string());
                ctx.defaults.insert(var_name, count.to_string());

                CommandResult {
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
                }
            }
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
                extended_description: None,
                        role: OptionRole::Output,
        },
            OptionDesc {
                name: "dimension".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Vector dimensionality".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "count".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Number of vectors to generate".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "type".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("float[]".to_string()),
                description: "Element type: float[], int[], double[], byte[], half, short[]".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Random seed for reproducibility".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "min".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("-1.0".to_string()),
                description: "Minimum value for float/double elements".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "max".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("1.0".to_string()),
                description: "Maximum value for float/double elements".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "zeros-ratio".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.0".to_string()),
                description: "Fraction of vectors to replace with zero vectors (0.0-1.0)".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "duplicates-ratio".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.0".to_string()),
                description: "Fraction of vectors to replace with duplicates of earlier vectors (0.0-1.0)".to_string(),
                extended_description: None,
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "append".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Append to an existing vector file (matches its dimensionality)".to_string(),
                extended_description: None,
                        role: OptionRole::Input,
        },
        ]
    }
}

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

/// The generators' RNG.
type Xoshiro = rand_xoshiro::Xoshiro256PlusPlus;

/// Generic generation with zero/duplicate injection, append support,
/// and parallel chunk generation.
///
/// Splits the work into chunks, generates each chunk's rows of `T` in
/// parallel using per-thread RNGs (seeded deterministically from the
/// main RNG), then writes all chunks sequentially to maintain ordering.
/// `sample` draws one element. The rows go through
/// [`veks_io::XvecWriter`], which creates the file or, for an append,
/// continues it after its last whole record.
fn generate_xvec_with_injection<T: veks_io::Element>(
    output: &Path,
    dim: u32,
    count: u64,
    zeros_count: u64,
    dups_count: u64,
    append: bool,
    rng: &mut Xoshiro,
    pb: &veks_core::ui::ProgressHandle,
    sample: impl Fn(&mut Xoshiro) -> T + Sync,
) -> Result<(), String> {
    use rand::{Rng, SeedableRng};

    let dim = dim as usize;
    // Every element type's zero is the all-zero encoding.
    let zero = T::from_le_bytes(&[0u8; 16]);

    let normal_ratio = if count > 0 {
        1.0 - (zeros_count as f64 / count as f64) - (dups_count as f64 / count as f64)
    } else { 1.0 };
    let zero_ratio = if count > 0 { zeros_count as f64 / count as f64 } else { 0.0 };
    let dup_ratio = if count > 0 { dups_count as f64 / count as f64 } else { 0.0 };

    // Single unified path: split into chunks, generate in parallel, write
    // sequentially. Progress is reported per sub-batch within each chunk
    // for smooth status updates.
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let chunk_size = (count as usize).div_ceil(threads).max(1);
    let num_chunks = (count as usize).div_ceil(chunk_size);

    let chunk_seeds: Vec<u64> = (0..num_chunks).map(|_| rng.random()).collect();
    let alias_table = AliasTable::new(normal_ratio.max(0.0), zero_ratio, dup_ratio);

    // Progress reporting interval: ~10K vectors per update
    let progress_interval = 10_000usize;

    let chunks: Vec<Vec<T>> = {
        use rayon::prelude::*;
        chunk_seeds.into_par_iter().enumerate().map(|(ci, seed)| {
            let start_idx = ci * chunk_size;
            let end_idx = std::cmp::min(start_idx + chunk_size, count as usize);
            let chunk_len = end_idx - start_idx;

            let mut chunk_rng = Xoshiro::seed_from_u64(seed);
            let mut buf: Vec<T> = Vec::with_capacity(chunk_len * dim);

            for j in 0..chunk_len {
                let disp = alias_table.sample(&mut chunk_rng);
                match disp {
                    VecDisposition::Zero => buf.resize(buf.len() + dim, zero),
                    // `j` rows have been generated when row `j` starts.
                    VecDisposition::Duplicate if j > 0 => {
                        let src_idx = chunk_rng.random_range(0..j);
                        buf.extend_from_within(src_idx * dim..src_idx * dim + dim);
                    }
                    VecDisposition::Normal | VecDisposition::Duplicate => {
                        for _ in 0..dim {
                            buf.push(sample(&mut chunk_rng));
                        }
                    }
                }
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
    let mut writer = if append {
        veks_io::XvecWriter::<T>::append(output, dim)
    } else {
        veks_io::XvecWriter::<T>::create(output, dim)
    }
    .map_err(|e| format!("failed to open {}: {}", output.display(), e))?;
    for chunk in &chunks {
        writer.write_packed(chunk).map_err(|e| e.to_string())?;
    }
    writer.finish().map_err(|e| e.to_string())?;

    pb.set_position(count);
    Ok(())
}

/// The element type a `type` option generates, by its generator spelling
/// (`float[]`, `f32`, …); `None` for a spelling the generators lack.
fn generated_element_type(elem_type: &str) -> Option<veks_io::ElementType> {
    use veks_io::ElementType;
    match elem_type {
        "float[]" | "f32" => Some(ElementType::F32),
        "half" | "f16" => Some(ElementType::F16),
        "double[]" | "f64" => Some(ElementType::F64),
        "int[]" | "i32" => Some(ElementType::I32),
        "short[]" | "i16" => Some(ElementType::I16),
        "byte[]" | "u8" => Some(ElementType::U8),
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            attributes: Vec::new(),
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
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
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
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

    /// **An append continues only a whole file**: rows appended after a
    /// partial record would be misread from it on, so a target with
    /// bytes past its last whole record is refused and left unchanged.
    #[test]
    fn an_append_to_a_cut_file_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("base.fvecs");
        let mut opts = Options::new();
        opts.set("output", out.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "10");
        let mut ctx = test_ctx(tmp.path());
        assert_eq!(GenerateVectorsOp.execute(&opts, &mut ctx).status, Status::Ok);

        let mut bytes = std::fs::read(&out).unwrap();
        bytes.extend([4, 0, 0]);
        std::fs::write(&out, &bytes).unwrap();
        let mut opts = Options::new();
        opts.set("append", out.to_string_lossy().to_string());
        opts.set("dimension", "4");
        opts.set("count", "5");
        let result = GenerateVectorsOp.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error, "{}", result.message);
        assert_eq!(std::fs::read(&out).unwrap(), bytes, "the target is left as it was");
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
