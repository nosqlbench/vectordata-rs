// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline commands: fvec-extract, ivec-extract, mvec-extract, and slab-extract.
//!
//! `fvec-extract`: Extracts vectors from an fvec file using indices from an
//! ivec file. Each index in the ivec is a 1-dimensional record whose value
//! selects a vector from the fvec source.
//!
//! `ivec-extract`: Extracts a range of records from an ivec file.
//!
//! `mvec-extract`: Extracts a range of records from an mvec (half-precision
//! float16) file.
//!
//! `slab-extract`: Extracts and reorders records from a slab file using indices
//! from an ivec file, or extracts a contiguous range. Used to maintain ordinal
//! correspondence between metadata slabs and shuffled/partitioned vector files.
//!
//! All support range specifications in the format `[start,end)` or `start..end`
//! to select a subset of the source file.
//!
//! Equivalent to Java `CMD_generate_fvecExtract` and `CMD_generate_ivecExtract`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

// ---- fvec-extract -----------------------------------------------------------

/// Pipeline command: extract fvec vectors by ivec indices.
/// Internal delegate for `transform extract` when source is `.fvec`.
pub(super) struct GenerateFvecExtractOp;

impl CommandOp for GenerateFvecExtractOp {
    fn command_path(&self) -> &str {
        "transform extract"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Extract a subset of vectors from an fvec file".into(),
            body: format!(r#"# transform extract

Extract a subset of vectors from an fvec file.

## Description

Extracts f32 vectors from a source fvec file using index-based lookup.
Each entry in the ivec file is a 1-dimensional record whose value is the
ordinal of a vector to copy from the source fvec. The `range` parameter
selects which entries of the ivec to use (not which source records to
read). For example, `range=[0,1000)` reads the first 1000 ivec entries
and copies the source vectors they reference.

## Partitioned-pass extraction

When the source file is too large to random-access efficiently, the
extraction uses a partitioned-pass algorithm. The requested indices are
sorted and bucketed into contiguous partitions of the source file. Each
partition is read sequentially in a single pass, and the matching vectors
are collected. Multiple passes over the source may be needed if the
indices span the entire file, but each pass reads sequentially, which is
far more efficient than scattered random reads on large datasets.

## Role in dataset pipelines

This is the workhorse of dataset splitting. After `generate ivec-shuffle`
produces a random permutation, `fvec-extract` applies it to produce
disjoint query and base vector sets:

- Query set: `fvec-extract` with `range=[0,K)` on the shuffle ivec
- Base set: `fvec-extract` with `range=[K,N)` on the shuffle ivec

The same shuffle + range pattern is applied to metadata slabs via
`slab-extract` so that ordinal correspondence is preserved across all
facets of the dataset.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
            ResourceDesc { name: "mem".into(), description: "Memory budget for write buffering".into(), adjustable: true },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let fvec_str = match options.require("fvec-file") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let fvec_path = resolve_path(fvec_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let ivec_path = options.get("ivec-file").map(|s| resolve_path(s, &ctx.workspace));

        // Parse range
        let range = match options.get("range") {
            Some(r) => match parse_range(r) {
                Ok(rng) => rng,
                Err(e) => return error_result(format!("invalid range '{}': {}", r, e), start),
            },
            None => Range { start: 0, end: None },
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

        // Validate extension — catch mismatched formats early
        if let Some(ext) = fvec_path.extension().and_then(|e| e.to_str()) {
            if ext != "fvec" {
                return error_result(
                    format!("fvec-extract expects a .fvec file, got .{} — use the appropriate *-extract command for this format", ext),
                    start,
                );
            }
        }

        let fvec_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&fvec_reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&fvec_reader) as u32;

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        let normalize = options.get("normalize").map(|s| s == "true").unwrap_or(false);
        if normalize {
            ctx.ui.log("  L2 normalization enabled during extraction");
        }

        if let Some(ref ivec_p) = ivec_path {
            // Index-based extraction via ivec indirection
            let ivec_reader = match MmapVectorReader::<i32>::open_ivec(ivec_p) {
                Ok(r) => r,
                Err(e) => return error_result(
                    format!("failed to open ivec file {}: {}", ivec_p.display(), e), start,
                ),
            };
            let ivec_count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&ivec_reader);
            let range_start = range.start;
            let range_end = range.end.unwrap_or(ivec_count);
            if range_start >= ivec_count {
                return error_result(
                    format!("range start {} exceeds ivec count {}", range_start, ivec_count), start,
                );
            }
            let effective_end = std::cmp::min(range_end, ivec_count);
            let result = sorted_index_extract_fvec(
                &fvec_reader, fvec_count, dim,
                &ivec_reader, range_start, effective_end,
                &output_path, normalize, ctx, start,
            );
            match result {
                Ok(msg) => {
                    // Write verified count for the bound checker
                    let extracted = effective_end - range_start;
                    let var_name = format!("verified_count:{}",
                        output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
                    let _ = crate::pipeline::variables::set_and_save(
                        &ctx.workspace, &var_name, &extracted.to_string());
                    ctx.defaults.insert(var_name, extracted.to_string());

                    CommandResult {
                        status: Status::Ok,
                        message: msg,
                        produced: vec![output_path],
                        elapsed: start.elapsed(),
                    }
                },
                Err(e) => error_result(e, start),
            }
        } else {
            // Range-based extraction: identity function over contiguous range
            use std::io::Write;
            let range_start = range.start;
            let range_end = range.end.unwrap_or(fvec_count);
            let effective_end = std::cmp::min(range_end, fvec_count);
            if range_start >= fvec_count {
                return error_result(
                    format!("range start {} exceeds fvec count {}", range_start, fvec_count), start,
                );
            }

            let mut writer = match AtomicWriter::new(&output_path) {
                Ok(w) => w,
                Err(e) => return error_result(format!("failed to create {}: {}", output_path.display(), e), start),
            };
            let pb = ctx.ui.bar((effective_end - range_start) as u64, "extracting fvec");
            for i in range_start..effective_end {
                let vec = match fvec_reader.get(i) {
                    Ok(v) => v,
                    Err(e) => return error_result(format!("read error at {}: {}", i, e), start),
                };
                let output_vec = if normalize {
                    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if norm > 0.0 { vec.iter().map(|v| v / norm).collect() } else { vec }
                } else {
                    vec
                };
                writer.write_all(&(dim as i32).to_le_bytes()).unwrap_or(());
                for &v in &output_vec {
                    writer.write_all(&v.to_le_bytes()).unwrap_or(());
                }
                pb.inc(1);
            }
            pb.finish();
            if let Err(e) = writer.finish() {
                return error_result(format!("failed to finalize {}: {}", output_path.display(), e), start);
            }

            // Write verified count for the bound checker
            let extracted = effective_end - range_start;
            let var_name = format!("verified_count:{}",
                output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &extracted.to_string());
            ctx.defaults.insert(var_name, extracted.to_string());

            CommandResult {
                status: Status::Ok,
                message: format!("extracted {} vectors to {}", extracted, output_path.display()),
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
                required: false,
                default: None,
                description: "Ivec file containing indices (identity if omitted)".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "fvec-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Fvec file containing source vectors".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output fvec file".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Index range: [start,end) or start..end".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "normalize".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "L2-normalize vectors during extraction".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["fvec-file", "ivec-file"],
            &["output"],
        )
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
/// Internal delegate for `transform extract` when source is `.ivec`.
pub(super) struct GenerateIvecExtractOp;

impl CommandOp for GenerateIvecExtractOp {
    fn command_path(&self) -> &str {
        "transform extract"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Extract a subset of vectors from an ivec file".into(),
            body: format!(r#"# transform extract

Extract a subset of vectors from an ivec file.

## Description

Extracts records from an ivec file. Two modes are supported:

**Index-based** (with `index-file`): Each entry in the index ivec is a
1-dimensional record whose value is the ordinal of a record to copy from
the source ivec. The `range` parameter controls which entries of the
index file to use -- it selects a window into the index array, not the
source file. For example, `range=[0,1000)` uses the first 1000 index
entries.

**Range-based** (without `index-file`): Extracts a contiguous slice of
records from the source ivec directly. The `range` parameter specifies
the half-open interval of source ordinals to copy.

Ranges are specified as `[start,end)` or `start..end`.

## Partitioned-pass extraction (index mode)

For index-based extraction on datasets larger than RAM, the algorithm
buckets the requested indices into partitions aligned to the source file
layout. Each partition is read sequentially in a single pass, collecting
the matching records. This avoids scattered random I/O and keeps memory
usage bounded regardless of dataset size.

## Role in dataset pipelines

After `generate ivec-shuffle` produces a random permutation, ivec-extract
can split an existing ivec (e.g. ground truth neighbor indices) into query
and base portions using the same shuffle + range pattern applied to
vector files. This ensures all facets of a dataset remain aligned by
ordinal.

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

            let pb = ctx.ui.bar((effective_end - range_start) as u64, "extracting ivec by index");
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
                if count % 10_000 == 0 { pb.set_position(count); }
            }
            pb.finish();

            if let Err(e) = writer.flush() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            // Write verified count for the bound checker
            let var_name = format!("verified_count:{}",
                output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &count.to_string());
            ctx.defaults.insert(var_name, count.to_string());

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

            let pb = ctx.ui.bar((effective_end - range_start) as u64, "extracting ivec");
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
                if count % 10_000 == 0 { pb.set_position(count); }
            }
            pb.finish();

            if let Err(e) = writer.flush() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            // Write verified count for the bound checker
            let var_name = format!("verified_count:{}",
                output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &count.to_string());
            ctx.defaults.insert(var_name, count.to_string());

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
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "index-file".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Ivec file containing indices (enables index-based extraction)".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output ivec file".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Range: [start,end) or start..end. Applies to index-file entries (index mode) or ivec records (range mode)".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["ivec-file"],
            &["output"],
        )
    }
}

// ---- mvec-extract -----------------------------------------------------------

/// Pipeline command: extract records from an mvec (f16) file.
///
/// Supports two modes:
/// - **Index-based**: provide `ivec-file` containing indices; each index selects
///   a vector from the mvec source. `range` selects which ivec entries to use.
/// - **Range-based**: omit `ivec-file`; `range` selects a contiguous range of
///   mvec records directly.
/// Internal delegate for `transform extract` when source is `.mvec`.
pub(super) struct GenerateMvecExtractOp;

impl CommandOp for GenerateMvecExtractOp {
    fn command_path(&self) -> &str {
        "transform extract"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Extract a subset of vectors from an mvec file".into(),
            body: format!(r#"# transform extract

Extract a subset of vectors from an mvec file.

## Description

Extracts vectors from an mvec (half-precision float16) file. Two modes
are supported:

**Index-based** (with `ivec-file`): Each entry in the ivec is a
1-dimensional record whose value is the ordinal of a vector to copy from
the mvec source. The `range` parameter controls which entries of the ivec
to use -- it selects a window into the index array, not the source file.
For example, `range=[K,N)` reads ivec entries K through N-1 and copies
the corresponding source vectors.

**Range-based** (without `ivec-file`): Extracts a contiguous slice of
records from the mvec file directly.

Ranges are specified as `[start,end)` or `start..end`.

## Partitioned-pass extraction (index mode)

For index-based extraction on large datasets, the algorithm buckets the
requested indices into partitions of the source file and reads each
partition sequentially. This avoids scattered random I/O and keeps memory
usage bounded. Multiple sequential passes over the source may be needed if
the requested indices span the entire file, but each pass is a linear scan
rather than random access.

## Role in dataset pipelines

This is typically the primary extract command for datasets stored in
half-precision format. After `generate ivec-shuffle` produces a random
permutation, mvec-extract splits the corpus vectors into disjoint query
and base sets:

- Query set: `mvec-extract` with `range=[0,K)` on the shuffle ivec
- Base set: `mvec-extract` with `range=[K,N)` on the shuffle ivec

The same shuffle + range is also applied to metadata via `slab-extract`
so that `base_metadata.slab[i]` corresponds to `base_vectors.mvec[i]`.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
            ResourceDesc { name: "mem".into(), description: "Memory budget for write buffering".into(), adjustable: true },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let mvec_str = match options.require("mvec-file") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let mvec_path = resolve_path(mvec_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let ivec_path = options.get("ivec-file").map(|s| resolve_path(s, &ctx.workspace));

        let range = match options.get("range") {
            Some(r) => match parse_range(r) {
                Ok(rng) => rng,
                Err(e) => return error_result(format!("invalid range '{}': {}", r, e), start),
            },
            None => Range { start: 0, end: None },
        };

        // Validate extension
        if let Some(ext) = mvec_path.extension().and_then(|e| e.to_str()) {
            if ext != "mvec" {
                return error_result(
                    format!("mvec-extract expects a .mvec file, got .{} — use the appropriate *-extract command for this format", ext),
                    start,
                );
            }
        }

        // Open the mvec file
        let mvec_reader = match MmapVectorReader::<half::f16>::open_mvec(&mvec_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open mvec file {}: {}", mvec_path.display(), e),
                    start,
                )
            }
        };

        let mvec_count =
            <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&mvec_reader);
        let dim =
            <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&mvec_reader) as u32;

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        let normalize = options.get("normalize").map(|s| s == "true").unwrap_or(false);
        if normalize {
            ctx.ui.log("  L2 normalization enabled during extraction");
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
            // Index-based extraction: read indices from ivec, look up vectors in mvec.
            //
            // Optimization: sort indices by source position to convert random I/O
            // into sequential I/O, then write each vector at its correct output
            // offset. Processes in memory-bounded chunks (up to half system RAM)
            // so this works on systems with limited memory.
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
            let result = sorted_index_extract_mvec(
                &mvec_reader, mvec_count, dim,
                &ivec_reader, range_start, effective_end,
                &output_path, normalize, ctx, start,
            );
            match result {
                Ok(msg) => {
                    // Write verified count for the bound checker
                    let extracted = effective_end - range_start;
                    let var_name = format!("verified_count:{}",
                        output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
                    let _ = crate::pipeline::variables::set_and_save(
                        &ctx.workspace, &var_name, &extracted.to_string());
                    ctx.defaults.insert(var_name, extracted.to_string());

                    CommandResult {
                        status: Status::Ok,
                        message: msg,
                        produced: vec![output_path],
                        elapsed: start.elapsed(),
                    }
                },
                Err(e) => error_result(e, start),
            }
        } else {
            // Range-based extraction: contiguous range from mvec
            let range_start = range.start;
            let range_end = range.end.unwrap_or(mvec_count);
            let effective_end = std::cmp::min(range_end, mvec_count);

            if range_start >= mvec_count {
                return error_result(
                    format!(
                        "range start {} exceeds mvec count {}",
                        range_start, mvec_count
                    ),
                    start,
                );
            }

            let pb = ctx.ui.bar((effective_end - range_start) as u64, "extracting mvec");
            let mut count: u64 = 0;
            for i in range_start..effective_end {
                let vec = match mvec_reader.get(i) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_result(
                            format!("failed to read mvec[{}]: {}", i, e),
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
                if count % 10_000 == 0 { pb.set_position(count); }
            }
            pb.finish();

            if let Err(e) = writer.flush() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            // Write verified count for the bound checker
            let var_name = format!("verified_count:{}",
                output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &count.to_string());
            ctx.defaults.insert(var_name, count.to_string());

            CommandResult {
                status: Status::Ok,
                message: format!(
                    "extracted {} mvec records (range [{}..{})) to {}",
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
                name: "mvec-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source mvec file".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "ivec-file".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Ivec file containing indices (enables index-based extraction)".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output mvec file".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Range: [start,end) or start..end. Applies to ivec entries (index mode) or mvec records (range mode)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "normalize".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "L2-normalize vectors during extraction".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["mvec-file", "ivec-file"],
            &["output"],
        )
    }
}

// ---- slab-extract (internal delegate for `transform extract`) ---------------

/// Internal delegate: extract and reorder slab records by ivec indices.
///
/// Not registered as a standalone command — invoked by `TransformExtractOp`
/// when the source file has a `.slab` extension.
pub(super) struct GenerateSlabExtractOp;

impl CommandOp for GenerateSlabExtractOp {
    fn command_path(&self) -> &str {
        "transform extract" // delegate — same path as parent
    }

    fn command_doc(&self) -> CommandDoc {
        CommandDoc {
            summary: "Extract and reorder records from a slab file (internal delegate)".into(),
            body: "Internal delegate for `transform extract` when source is .slab. \
                   See `transform extract` for documentation.".into(),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Partition buffer memory".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let slab_str = match options.require("slab-file") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let slab_path = resolve_path(slab_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let ivec_path = options.get("ivec-file").map(|s| resolve_path(s, &ctx.workspace));

        let range = match options.get("range") {
            Some(r) => match parse_range(r) {
                Ok(rng) => rng,
                Err(e) => return error_result(format!("invalid range '{}': {}", r, e), start),
            },
            None => Range { start: 0, end: None },
        };

        // Open the source slab
        let reader = match slabtastic::SlabReader::open(&slab_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open slab file {}: {}", slab_path.display(), e),
                    start,
                )
            }
        };

        let slab_count = reader.total_records() as usize;

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        let page_size: u32 = options
            .get("page-size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);

        if let Some(ref ivec_p) = ivec_path {
            // Index-based extraction: partitioned-pass approach for sequential I/O
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

            match sorted_index_extract_slab(
                &reader, slab_count, &ivec_reader,
                range_start, effective_end, &output_path, page_size,
                &slab_path, ivec_p,
                ctx, start,
            ) {
                Ok(msg) => CommandResult {
                    status: Status::Ok,
                    message: msg,
                    produced: vec![output_path],
                    elapsed: start.elapsed(),
                },
                Err(msg) => error_result(msg, start),
            }
        } else {
            // Range-based extraction: contiguous range from slab (already sequential)
            let range_start = range.start;
            let range_end = range.end.unwrap_or(slab_count);
            let effective_end = std::cmp::min(range_end, slab_count);

            if range_start >= slab_count {
                return error_result(
                    format!(
                        "range start {} exceeds slab record count {}",
                        range_start, slab_count
                    ),
                    start,
                );
            }

            let config = match slabtastic::WriterConfig::new(512, page_size, u32::MAX, false) {
                Ok(c) => c,
                Err(e) => return error_result(format!("invalid writer config: {}", e), start),
            };
            let mut writer = match slabtastic::SlabWriter::new(&output_path, config) {
                Ok(w) => w,
                Err(e) => return error_result(format!("failed to create {}: {}", output_path.display(), e), start),
            };

            let pb = ctx.ui.bar((effective_end - range_start) as u64, "extracting slab");
            let mut count: u64 = 0;
            for i in range_start..effective_end {
                let data = match reader.get_ref(i as i64) {
                    Ok(d) => d,
                    Err(e) => {
                        return error_result(
                            format!("failed to read slab record {}: {}", i, e),
                            start,
                        )
                    }
                };

                if let Err(e) = writer.add_record(data) {
                    return error_result(format!("write error at record {}: {}", count, e), start);
                }
                count += 1;
                if count % 10_000 == 0 { pb.set_position(count); }
            }
            pb.finish();

            if let Err(e) = writer.finish() {
                return error_result(format!("failed to finalize output: {}", e), start);
            }

            CommandResult {
                status: Status::Ok,
                message: format!(
                    "extracted {} slab records (range [{}..{})) to {}",
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
                name: "slab-file".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source slab file".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "ivec-file".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Ivec file containing indices (enables index-based extraction)".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output slab file".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Range: [start,end) or start..end. Applies to ivec entries (index mode) or slab records (range mode)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "page-size".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("65536".to_string()),
                description: "Preferred page size for output slab".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["slab-file", "ivec-file"],
            &["output"],
        )
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
    let v = vectordata::dataset::source::parse_number_with_suffix(s)?;
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

/// Return half of system RAM in bytes, clamping to at least 256 MiB.
fn half_system_ram() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) } as u64;
        let total = page_size * pages;
        (total / 2).max(256 * 1024 * 1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Assume 4 GiB on non-Linux
        2u64 * 1024 * 1024 * 1024
    }
}

/// Partitioned-pass extraction for mvec files.
///
/// For each output partition:
/// 1. Scan the ivec (just integers) to find entries whose output position
///    falls in this partition. Build a read plan: `(source_idx, local_out_pos)`.
/// 2. Sort the read plan by source_idx — this is the sequential read order.
/// 3. Walk the read plan, reading source records in order (sequential I/O).
///    Each record is placed directly into the partition buffer at its
///    transpose position.
/// 4. Write the entire partition buffer contiguously.
///
/// Both reads and writes are sequential. The ivec scan per pass is cheap
/// (integer range check only). Source data is only read for records in the
/// current partition.
fn sorted_index_extract_mvec(
    mvec_reader: &MmapVectorReader<half::f16>,
    mvec_count: usize,
    dim: u32,
    ivec_reader: &MmapVectorReader<i32>,
    range_start: usize,
    effective_end: usize,
    output_path: &Path,
    normalize: bool,
    ctx: &mut StreamContext,
    _start: Instant,
) -> Result<String, String> {
    use std::io::Write;

    let extract_count = effective_end - range_start;
    let record_bytes = 4 + (dim as usize) * 2; // dim header + f16 values

    // Governor-controlled thread pool for parallel phases
    let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();
    ctx.ui.log(&format!("mvec-extract: using {} threads", threads));

    // Determine partition count from memory budget
    let default_mem = half_system_ram();
    let mem_budget = ctx.governor.offer_demand("mem", 0, default_mem);
    let records_per_partition = (mem_budget as usize / record_bytes).max(1);
    let raw_partitions = (extract_count + records_per_partition - 1) / records_per_partition;
    let num_partitions = raw_partitions.max(2);
    let partition_size = (extract_count + num_partitions - 1) / num_partitions;

    ctx.ui.log(&format!(
        "mvec-extract: {} vectors, {} bytes/record, {} passes (budget: {:.1} GiB, {:.0} MB/partition)",
        extract_count, record_bytes, num_partitions,
        mem_budget as f64 / (1024.0 * 1024.0 * 1024.0),
        (partition_size * record_bytes) as f64 / (1024.0 * 1024.0),
    ));

    // Pre-allocate output file
    let total_bytes = (extract_count as u64) * (record_bytes as u64);
    {
        let f = std::fs::File::create(output_path)
            .map_err(|e| format!("failed to create output: {}", e))?;
        f.set_len(total_bytes)
            .map_err(|e| format!("failed to set output size: {}", e))?;
    }

    let mut out_file = std::fs::OpenOptions::new()
        .write(true)
        .open(output_path)
        .map_err(|e| format!("failed to open output {}: {}", output_path.display(), e))?;

    let dim_bytes = (dim as i32).to_le_bytes();
    let pass_label = |p: usize| -> String {
        if num_partitions > 1 { format!(" (pass {}/{})", p + 1, num_partitions) } else { String::new() }
    };

    // Pre-allocate buffers outside the loop to avoid huge alloc/dealloc between passes
    let max_part_len = partition_size; // first partition is the largest or equal
    let mut read_plan: Vec<(usize, usize)> = Vec::with_capacity(max_part_len);
    let mut part_buf: Vec<u8> = vec![0u8; max_part_len * record_bytes];

    for pass in 0..num_partitions {
        let part_start = pass * partition_size;
        let part_end = std::cmp::min(part_start + partition_size, extract_count);
        let part_len = part_end - part_start;

        // Step 1: Scan ivec for this partition's entries only.
        // Output positions are sequential, so we jump directly to the
        // partition's global range instead of iterating all entries.
        let global_start = range_start + part_start;
        let global_end = range_start + part_end;
        let scan_pb = ctx.ui.bar_with_unit(part_len as u64, &format!("scanning indices{}", pass_label(pass)), "vectors");
        read_plan.clear();
        for (local_pos, i) in (global_start..global_end).enumerate() {
            let index_vec = ivec_reader.get(i)
                .map_err(|e| format!("failed to read ivec[{}]: {}", i, e))?;
            let source_idx = index_vec[0] as usize;
            if source_idx >= mvec_count {
                return Err(format!("index {} at ivec[{}] exceeds mvec count {}", source_idx, i, mvec_count));
            }
            read_plan.push((source_idx, local_pos));
            if (local_pos + 1) % 100_000 == 0 { scan_pb.set_position((local_pos + 1) as u64); }
        }
        scan_pb.finish();

        // Step 2: Sort by source position — parallel bucket + sort.
        // Distribute into buckets in parallel, sort each bucket in
        // parallel, then flatten with prefix-sum offsets.
        let num_buckets = 256usize;
        let bucket_range = (mvec_count / num_buckets).max(1);

        let dist_pb = ctx.ui.bar_with_unit(read_plan.len() as u64, &format!("bucketing {} entries{}", read_plan.len(), pass_label(pass)), "vectors");
        let thread_buckets: Vec<Vec<Vec<(usize, usize)>>>;
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};
            let progress = AtomicU64::new(0);
            let bucket_fn = || {
                read_plan.par_chunks(64 * 1024).map(|chunk| {
                    let mut local: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_buckets];
                    for &entry in chunk {
                        let b = (entry.0 / bucket_range).min(num_buckets - 1);
                        local[b].push(entry);
                    }
                    let done = progress.fetch_add(chunk.len() as u64, Ordering::Relaxed) + chunk.len() as u64;
                    dist_pb.set_position(done);
                    local
                }).collect()
            };
            thread_buckets = if let Some(ref p) = pool {
                p.install(bucket_fn)
            } else {
                bucket_fn()
            };
        }
        dist_pb.finish();

        // Merge thread-local buckets into unified buckets
        let merge_pb = ctx.ui.spinner(&format!("merging {} thread buckets{}", thread_buckets.len(), pass_label(pass)));
        let mut buckets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_buckets];
        for tb in &thread_buckets {
            for (i, b) in tb.iter().enumerate() {
                buckets[i].reserve(b.len());
            }
        }
        for tb in thread_buckets {
            for (i, b) in tb.into_iter().enumerate() {
                buckets[i].extend(b);
            }
        }
        merge_pb.finish();

        let sort_pb = ctx.ui.bar_with_unit(read_plan.len() as u64, &format!("sorting {} entries by source position{}", read_plan.len(), pass_label(pass)), "vectors");
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};
            let progress = AtomicU64::new(0);
            let mut sort_fn = || {
                buckets.par_iter_mut().for_each(|bucket| {
                    bucket.sort_unstable_by_key(|&(src, _)| src);
                    let done = progress.fetch_add(bucket.len() as u64, Ordering::Relaxed) + bucket.len() as u64;
                    sort_pb.set_position(done);
                });
            };
            if let Some(ref p) = pool {
                p.install(sort_fn);
            } else {
                sort_fn();
            }
        }
        sort_pb.finish();

        // Flatten sorted buckets into read_plan using prefix-sum offsets
        let flatten_pb = ctx.ui.spinner(&format!("flattening read plan{}", pass_label(pass)));
        let total_entries = buckets.iter().map(|b| b.len()).sum::<usize>();
        read_plan.clear();
        read_plan.resize(total_entries, (0, 0));
        let mut offsets: Vec<usize> = Vec::with_capacity(num_buckets);
        let mut off = 0usize;
        for bucket in &buckets {
            offsets.push(off);
            off += bucket.len();
        }
        {
            use rayon::prelude::*;
            let flatten_fn = || {
                buckets.into_par_iter().enumerate().for_each(|(i, bucket)| {
                    let start = offsets[i];
                    let dest = unsafe {
                        std::slice::from_raw_parts_mut(
                            (read_plan.as_ptr() as *mut (usize, usize)).add(start),
                            bucket.len(),
                        )
                    };
                    dest.copy_from_slice(&bucket);
                });
            };
            if let Some(ref p) = pool {
                p.install(flatten_fn);
            } else {
                flatten_fn();
            }
        }

        flatten_pb.finish();

        // Step 3: Read source data in parallel, placing each record at its
        // transpose position in the buffer. Each local_pos is unique, so
        // concurrent writes to part_buf are safe (disjoint regions).
        let read_pb = ctx.ui.bar_with_unit(read_plan.len() as u64, &format!("reading+transposing mvec{}", pass_label(pass)), "vectors");
        let part_buf_len = part_len * record_bytes;
        part_buf[..part_buf_len].fill(0);
        mvec_reader.advise_sequential();

        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};

            let progress = AtomicU64::new(0);
            let shared_buf = SharedBuf::new(&mut part_buf);

            let read_fn = || {
                read_plan.par_iter().try_for_each(|&(source_idx, local_pos)| {
                    let vector = mvec_reader.get(source_idx)
                        .map_err(|e| format!("failed to read mvec[{}]: {}", source_idx, e))?;

                    let buf_offset = local_pos * record_bytes;
                    let dest = unsafe { shared_buf.slice_mut(buf_offset, record_bytes) };
                    dest[..4].copy_from_slice(&dim_bytes);
                    let slice: &[half::f16] = vector.as_ref();
                    let src_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 2)
                    };
                    dest[4..].copy_from_slice(src_bytes);

                    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 100_000 == 0 {
                        read_pb.set_position(done);
                    }
                    Ok(())
                })
            };
            let result: Result<(), String> = if let Some(ref p) = pool {
                p.install(read_fn)
            } else {
                read_fn()
            };
            result?;
        }
        read_pb.finish();

        // Optional L2 normalization of vectors in the output buffer
        if normalize {
            let buf_used = part_len * record_bytes;
            let dim_usize = dim as usize;
            let stride = 4 + dim_usize * 2; // dim header + dim f16s
            for offset in (0..buf_used).step_by(stride) {
                let values_start = offset + 4;
                let mut norm_sq = 0.0f64;
                for d in 0..dim_usize {
                    let pos = values_start + d * 2;
                    let v = half::f16::from_le_bytes([part_buf[pos], part_buf[pos + 1]]);
                    let vf = v.to_f64();
                    norm_sq += vf * vf;
                }
                if norm_sq > 0.0 {
                    let inv_norm = 1.0 / norm_sq.sqrt();
                    for d in 0..dim_usize {
                        let pos = values_start + d * 2;
                        let v = half::f16::from_le_bytes([part_buf[pos], part_buf[pos + 1]]);
                        let normalized = half::f16::from_f64(v.to_f64() * inv_norm);
                        part_buf[pos..pos + 2].copy_from_slice(&normalized.to_le_bytes());
                    }
                }
            }
        }

        // Step 4: Write partition in chunks with progress, including sync
        let total_write_bytes = part_len * record_bytes;
        let write_mb = total_write_bytes as f64 / (1024.0 * 1024.0);
        // Progress covers writes + sync; reserve 5% of bar for the sync phase
        let sync_reserve = total_write_bytes as u64 / 20;
        let write_pb = ctx.ui.bar_with_unit(total_write_bytes as u64 + sync_reserve, &format!("writing+syncing {:.0} MB{}", write_mb, pass_label(pass)), "bytes");
        let file_offset = (part_start as u64) * (record_bytes as u64);
        use std::io::Seek;
        out_file.seek(std::io::SeekFrom::Start(file_offset))
            .map_err(|e| format!("seek failed: {}", e))?;
        let write_chunk = 8 * 1024 * 1024; // 8 MiB per write call
        let mut written: usize = 0;
        while written < total_write_bytes {
            let end = std::cmp::min(written + write_chunk, total_write_bytes);
            out_file.write_all(&part_buf[written..end])
                .map_err(|e| format!("write failed: {}", e))?;
            written = end;
            write_pb.set_position(written as u64);
        }
        out_file.sync_data()
            .map_err(|e| format!("sync failed: {}", e))?;
        write_pb.set_position(total_write_bytes as u64 + sync_reserve);
        write_pb.finish();

        log::debug!(
            "mvec-extract: pass {}/{}, wrote {} records ({:.1} MB)",
            pass + 1, num_partitions, read_plan.len(), write_mb,
        );

        ctx.governor.checkpoint();
    }
    out_file.sync_all().map_err(|e| format!("sync failed: {}", e))?;

    Ok(format!(
        "extracted {} mvec vectors by index (ivec range [{}..{}), {} passes, {} threads) to {}",
        extract_count, range_start, effective_end, num_partitions, threads, output_path.display()
    ))
}

/// Partitioned-pass extraction for fvec files.
///
/// Same algorithm as mvec: for each output partition, scan ivec to build
/// a read plan, sort by source position, read sequentially, write contiguously.
fn sorted_index_extract_fvec(
    fvec_reader: &MmapVectorReader<f32>,
    fvec_count: usize,
    dim: u32,
    ivec_reader: &MmapVectorReader<i32>,
    range_start: usize,
    effective_end: usize,
    output_path: &Path,
    normalize: bool,
    ctx: &mut StreamContext,
    _start: Instant,
) -> Result<String, String> {
    use std::io::Write;

    let extract_count = effective_end - range_start;
    let record_bytes = 4 + (dim as usize) * 4; // dim header + f32 values

    // Governor-controlled thread pool for parallel phases
    let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();
    ctx.ui.log(&format!("fvec-extract: using {} threads", threads));

    // Partition sizing
    let default_mem = half_system_ram();
    let mem_budget = ctx.governor.offer_demand("mem", 0, default_mem);
    let records_per_partition = (mem_budget as usize / record_bytes).max(1);
    let raw_partitions = (extract_count + records_per_partition - 1) / records_per_partition;
    let num_partitions = raw_partitions.max(2);
    let partition_size = (extract_count + num_partitions - 1) / num_partitions;

    ctx.ui.log(&format!(
        "fvec-extract: {} vectors, {} bytes/record, {} passes (budget: {:.1} GiB, {:.0} MB/partition)",
        extract_count, record_bytes, num_partitions,
        mem_budget as f64 / (1024.0 * 1024.0 * 1024.0),
        (partition_size * record_bytes) as f64 / (1024.0 * 1024.0),
    ));

    // Pre-allocate output
    let total_bytes = (extract_count as u64) * (record_bytes as u64);
    {
        let f = std::fs::File::create(output_path)
            .map_err(|e| format!("failed to create output: {}", e))?;
        f.set_len(total_bytes)
            .map_err(|e| format!("failed to set output size: {}", e))?;
    }

    let mut out_file = std::fs::OpenOptions::new()
        .write(true)
        .open(output_path)
        .map_err(|e| format!("failed to open output {}: {}", output_path.display(), e))?;

    let dim_bytes = (dim as i32).to_le_bytes();
    let pass_label = |p: usize| -> String {
        if num_partitions > 1 { format!(" (pass {}/{})", p + 1, num_partitions) } else { String::new() }
    };

    // Pre-allocate buffers outside the loop to avoid huge alloc/dealloc between passes
    let max_part_len = partition_size;
    let mut read_plan: Vec<(usize, usize)> = Vec::with_capacity(max_part_len);
    let mut part_buf: Vec<u8> = vec![0u8; max_part_len * record_bytes];

    for pass in 0..num_partitions {
        let part_start = pass * partition_size;
        let part_end = std::cmp::min(part_start + partition_size, extract_count);
        let part_len = part_end - part_start;

        // Step 1: Scan ivec for this partition's entries only.
        let global_start = range_start + part_start;
        let global_end = range_start + part_end;
        let scan_pb = ctx.ui.bar_with_unit(part_len as u64, &format!("scanning indices{}", pass_label(pass)), "vectors");
        read_plan.clear();
        for (local_pos, i) in (global_start..global_end).enumerate() {
            let index_vec = ivec_reader.get(i)
                .map_err(|e| format!("failed to read ivec[{}]: {}", i, e))?;
            let source_idx = index_vec[0] as usize;
            if source_idx >= fvec_count {
                return Err(format!("index {} at ivec[{}] exceeds fvec count {}", source_idx, i, fvec_count));
            }
            read_plan.push((source_idx, local_pos));
            if (local_pos + 1) % 100_000 == 0 { scan_pb.set_position((local_pos + 1) as u64); }
        }
        scan_pb.finish();

        // Step 2: Sort by source position — parallel bucket + sort.
        let num_buckets = 256usize;
        let bucket_range = (fvec_count / num_buckets).max(1);

        let dist_pb = ctx.ui.bar_with_unit(read_plan.len() as u64, &format!("bucketing {} entries{}", read_plan.len(), pass_label(pass)), "vectors");
        let thread_buckets: Vec<Vec<Vec<(usize, usize)>>>;
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};
            let progress = AtomicU64::new(0);
            let bucket_fn = || {
                read_plan.par_chunks(64 * 1024).map(|chunk| {
                    let mut local: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_buckets];
                    for &entry in chunk {
                        let b = (entry.0 / bucket_range).min(num_buckets - 1);
                        local[b].push(entry);
                    }
                    let done = progress.fetch_add(chunk.len() as u64, Ordering::Relaxed) + chunk.len() as u64;
                    dist_pb.set_position(done);
                    local
                }).collect()
            };
            thread_buckets = if let Some(ref p) = pool {
                p.install(bucket_fn)
            } else {
                bucket_fn()
            };
        }
        dist_pb.finish();

        let merge_pb = ctx.ui.spinner(&format!("merging {} thread buckets{}", thread_buckets.len(), pass_label(pass)));
        let mut buckets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_buckets];
        for tb in &thread_buckets {
            for (i, b) in tb.iter().enumerate() {
                buckets[i].reserve(b.len());
            }
        }
        for tb in thread_buckets {
            for (i, b) in tb.into_iter().enumerate() {
                buckets[i].extend(b);
            }
        }
        merge_pb.finish();

        let sort_pb = ctx.ui.bar_with_unit(read_plan.len() as u64, &format!("sorting {} entries by source position{}", read_plan.len(), pass_label(pass)), "vectors");
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};
            let progress = AtomicU64::new(0);
            let mut sort_fn = || {
                buckets.par_iter_mut().for_each(|bucket| {
                    bucket.sort_unstable_by_key(|&(src, _)| src);
                    let done = progress.fetch_add(bucket.len() as u64, Ordering::Relaxed) + bucket.len() as u64;
                    sort_pb.set_position(done);
                });
            };
            if let Some(ref p) = pool {
                p.install(sort_fn);
            } else {
                sort_fn();
            }
        }
        sort_pb.finish();

        let flatten_pb = ctx.ui.spinner(&format!("flattening read plan{}", pass_label(pass)));
        let total_entries = buckets.iter().map(|b| b.len()).sum::<usize>();
        read_plan.clear();
        read_plan.resize(total_entries, (0, 0));
        let mut offsets: Vec<usize> = Vec::with_capacity(num_buckets);
        let mut off = 0usize;
        for bucket in &buckets {
            offsets.push(off);
            off += bucket.len();
        }
        {
            use rayon::prelude::*;
            let flatten_fn = || {
                buckets.into_par_iter().enumerate().for_each(|(i, bucket)| {
                    let start = offsets[i];
                    let dest = unsafe {
                        std::slice::from_raw_parts_mut(
                            (read_plan.as_ptr() as *mut (usize, usize)).add(start),
                            bucket.len(),
                        )
                    };
                    dest.copy_from_slice(&bucket);
                });
            };
            if let Some(ref p) = pool {
                p.install(flatten_fn);
            } else {
                flatten_fn();
            }
        }
        flatten_pb.finish();

        // Step 3: Read source data in parallel with transpose placement
        let read_pb = ctx.ui.bar_with_unit(read_plan.len() as u64, &format!("reading+transposing fvec{}", pass_label(pass)), "vectors");
        let part_buf_len = part_len * record_bytes;
        part_buf[..part_buf_len].fill(0);
        fvec_reader.advise_sequential();

        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};

            let progress = AtomicU64::new(0);
            let shared_buf = SharedBuf::new(&mut part_buf);

            let read_fn = || {
                read_plan.par_iter().try_for_each(|&(source_idx, local_pos)| {
                    let vector = fvec_reader.get(source_idx)
                        .map_err(|e| format!("failed to read fvec[{}]: {}", source_idx, e))?;

                    let buf_offset = local_pos * record_bytes;
                    let dest = unsafe { shared_buf.slice_mut(buf_offset, record_bytes) };
                    dest[..4].copy_from_slice(&dim_bytes);
                    let slice: &[f32] = vector.as_ref();
                    let src_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
                    };
                    dest[4..].copy_from_slice(src_bytes);

                    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 100_000 == 0 {
                        read_pb.set_position(done);
                    }
                    Ok(())
                })
            };
            let result: Result<(), String> = if let Some(ref p) = pool {
                p.install(read_fn)
            } else {
                read_fn()
            };
            result?;
        }
        read_pb.finish();

        // Optional L2 normalization of vectors in the output buffer
        if normalize {
            let buf_used = part_len * record_bytes;
            let dim_usize = dim as usize;
            let stride = 4 + dim_usize * 4; // dim header + dim f32s
            for offset in (0..buf_used).step_by(stride) {
                let values_start = offset + 4;
                let mut norm_sq = 0.0f64;
                for d in 0..dim_usize {
                    let pos = values_start + d * 4;
                    let v = f32::from_le_bytes([
                        part_buf[pos], part_buf[pos + 1],
                        part_buf[pos + 2], part_buf[pos + 3],
                    ]);
                    norm_sq += (v as f64) * (v as f64);
                }
                if norm_sq > 0.0 {
                    let inv_norm = (1.0 / norm_sq.sqrt()) as f32;
                    for d in 0..dim_usize {
                        let pos = values_start + d * 4;
                        let v = f32::from_le_bytes([
                            part_buf[pos], part_buf[pos + 1],
                            part_buf[pos + 2], part_buf[pos + 3],
                        ]);
                        let normalized = v * inv_norm;
                        part_buf[pos..pos + 4].copy_from_slice(&normalized.to_le_bytes());
                    }
                }
            }
        }

        // Step 4: Write partition in chunks with progress, including sync
        let total_write_bytes = part_len * record_bytes;
        let write_mb = total_write_bytes as f64 / (1024.0 * 1024.0);
        let sync_reserve = total_write_bytes as u64 / 20;
        let write_pb = ctx.ui.bar_with_unit(total_write_bytes as u64 + sync_reserve, &format!("writing+syncing {:.0} MB{}", write_mb, pass_label(pass)), "bytes");
        let file_offset = (part_start as u64) * (record_bytes as u64);
        use std::io::Seek;
        out_file.seek(std::io::SeekFrom::Start(file_offset))
            .map_err(|e| format!("seek failed: {}", e))?;
        let write_chunk = 8 * 1024 * 1024; // 8 MiB per write call
        let mut written: usize = 0;
        while written < total_write_bytes {
            let end = std::cmp::min(written + write_chunk, total_write_bytes);
            out_file.write_all(&part_buf[written..end])
                .map_err(|e| format!("write failed: {}", e))?;
            written = end;
            write_pb.set_position(written as u64);
        }
        out_file.sync_data()
            .map_err(|e| format!("sync failed: {}", e))?;
        write_pb.set_position(total_write_bytes as u64 + sync_reserve);
        write_pb.finish();

        log::debug!(
            "fvec-extract: pass {}/{}, wrote {} records ({:.1} MB)",
            pass + 1, num_partitions, read_plan.len(), write_mb,
        );

        ctx.governor.checkpoint();
    }

    out_file.sync_all().map_err(|e| format!("sync failed: {}", e))?;

    Ok(format!(
        "extracted {} fvec vectors (range [{}..{}), {} passes, {} threads) to {}",
        extract_count, range_start, effective_end, num_partitions, threads, output_path.display()
    ))
}

/// Metadata for slab-extract resume validation.
///
/// Persisted as `slab-extract.meta.json` in the cache directory so that a
/// subsequent run can detect whether the parameters match and reuse
/// per-partition cache files.
#[derive(serde::Serialize, serde::Deserialize)]
struct SlabExtractMeta {
    slab_path: String,
    ivec_path: String,
    range_start: usize,
    effective_end: usize,
    num_partitions: usize,
    page_size: u32,
}

/// Partitioned-pass extraction for slab files.
///
/// Same principle as xvec extraction: partition by output position, scan ivec
/// per partition, sort by source ordinal for sequential slab reads, then write
/// in output order. Since slab records are variable-length, we collect
/// `(local_out_pos, data)` pairs and sort by output position before writing.
///
/// Supports per-partition cache/resume: each partition is written to a cache
/// file under `ctx.cache`. On re-run with matching parameters, partitions
/// whose cache file already exists are replayed from cache instead of being
/// recomputed.
fn sorted_index_extract_slab(
    reader: &slabtastic::SlabReader,
    slab_count: usize,
    ivec_reader: &MmapVectorReader<i32>,
    range_start: usize,
    effective_end: usize,
    output_path: &Path,
    page_size: u32,
    slab_path: &Path,
    ivec_path: &Path,
    ctx: &mut StreamContext,
    _start: Instant,
) -> Result<String, String> {
    let extract_count = effective_end - range_start;

    // Estimate average record size from a sample for partitioning.
    // Sample up to 1000 records evenly spaced across the slab.
    let sample_count = std::cmp::min(1000, slab_count);
    let mut total_sample_bytes: u64 = 0;
    for i in 0..sample_count {
        let ordinal = (i * slab_count / sample_count) as i64;
        if let Ok(data) = reader.get_ref(ordinal) {
            total_sample_bytes += data.len() as u64;
        }
    }
    let avg_record_bytes = if sample_count > 0 {
        (total_sample_bytes / sample_count as u64).max(64) as usize
    } else {
        256 // fallback
    };

    // Partition sizing from memory budget
    let default_mem = half_system_ram();
    let mem_budget = ctx.governor.offer_demand("mem", 0, default_mem);
    let records_per_partition = (mem_budget as usize / avg_record_bytes).max(1);
    let raw_partitions = (extract_count + records_per_partition - 1) / records_per_partition;
    let num_partitions = raw_partitions.max(2);
    let partition_size = (extract_count + num_partitions - 1) / num_partitions;

    ctx.ui.log(&format!(
        "slab-extract: {} records, ~{} bytes/record avg, {} passes (budget: {:.1} GiB)",
        extract_count, avg_record_bytes, num_partitions,
        mem_budget as f64 / (1024.0 * 1024.0 * 1024.0),
    ));

    // ---- cache / resume validation ----
    let cache_dir = ctx.cache.join("slab-extract");
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        return Err(format!("failed to create cache dir {}: {}", cache_dir.display(), e));
    }

    let meta_path = cache_dir.join("slab-extract.meta.json");
    let current_meta = SlabExtractMeta {
        slab_path: slab_path.to_string_lossy().into_owned(),
        ivec_path: ivec_path.to_string_lossy().into_owned(),
        range_start,
        effective_end,
        num_partitions,
        page_size,
    };

    let can_resume = if meta_path.exists() {
        match std::fs::read_to_string(&meta_path) {
            Ok(content) => match serde_json::from_str::<SlabExtractMeta>(&content) {
                Ok(prev) => {
                    prev.slab_path == current_meta.slab_path
                        && prev.ivec_path == current_meta.ivec_path
                        && prev.range_start == current_meta.range_start
                        && prev.effective_end == current_meta.effective_end
                        && prev.num_partitions == current_meta.num_partitions
                        && prev.page_size == current_meta.page_size
                }
                Err(_) => false,
            },
            Err(_) => false,
        }
    } else {
        false
    };

    if !can_resume {
        // Parameters changed — invalidate all existing partition caches.
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("slab-extract.part_") && name_str.ends_with(".cache") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        // Write fresh metadata so future runs can resume.
        if let Ok(json) = serde_json::to_string_pretty(&current_meta) {
            let _ = std::fs::write(&meta_path, json);
        }
    }

    let config = match slabtastic::WriterConfig::new(512, page_size, u32::MAX, false) {
        Ok(c) => c,
        Err(e) => return Err(format!("invalid writer config: {}", e)),
    };
    let mut writer = match slabtastic::SlabWriter::new(output_path, config) {
        Ok(w) => w,
        Err(e) => return Err(format!("failed to create {}: {}", output_path.display(), e)),
    };

    let pass_label = |p: usize| -> String {
        if num_partitions > 1 { format!(" (pass {}/{})", p + 1, num_partitions) } else { String::new() }
    };

    let mut read_plan: Vec<(usize, usize)> = Vec::with_capacity(partition_size);
    let mut resumed_count: usize = 0;

    for pass in 0..num_partitions {
        let part_start = pass * partition_size;
        let part_end = std::cmp::min(part_start + partition_size, extract_count);
        let part_len = part_end - part_start;

        // ---- per-partition cache check ----
        let cache_path = cache_dir.join(format!("slab-extract.part_{:04}.cache", pass));
        if can_resume && cache_path.exists() {
            let usable = (|| -> Option<bool> {
                let file_meta = std::fs::metadata(&cache_path).ok()?;
                if file_meta.len() == 0 { return None; }
                let cache_reader = slabtastic::SlabReader::open(&cache_path).ok()?;
                let cached_count = cache_reader.total_records() as usize;
                if cached_count != part_len { return None; }
                Some(true)
            })();

            if usable == Some(true) {
                // Replay cached partition into the output writer.
                let cache_reader = slabtastic::SlabReader::open(&cache_path)
                    .map_err(|e| format!("failed to reopen cache {}: {}", cache_path.display(), e))?;
                let cached_count = cache_reader.total_records() as usize;
                let replay_pb = ctx.ui.bar_with_unit(
                    cached_count as u64,
                    &format!("resuming from cache{}", pass_label(pass)),
                    "records",
                );
                for i in 0..cached_count {
                    let data = cache_reader.get_ref(i as i64)
                        .map_err(|e| format!(
                            "failed to read cache record {}: {}", i, e
                        ))?;
                    writer.add_record(data)
                        .map_err(|e| format!(
                            "write error replaying cache record {}: {}", i, e
                        ))?;
                    if (i + 1) % 10_000 == 0 {
                        replay_pb.set_position((i + 1) as u64);
                    }
                }
                replay_pb.finish();
                log::debug!(
                    "slab-extract: pass {}/{}, resumed {} records from cache",
                    pass + 1, num_partitions, cached_count,
                );
                resumed_count += 1;
                ctx.governor.checkpoint();
                continue;
            }
            // Cache file is unusable — remove it and recompute.
            let _ = std::fs::remove_file(&cache_path);
        }

        // Step 1: Scan ivec for this partition's entries
        let global_start = range_start + part_start;
        let global_end = range_start + part_end;
        let scan_pb = ctx.ui.bar(part_len as u64, &format!("scanning indices{}", pass_label(pass)));
        read_plan.clear();
        for (local_pos, i) in (global_start..global_end).enumerate() {
            let index_vec = ivec_reader.get(i)
                .map_err(|e| format!("failed to read ivec[{}]: {}", i, e))?;
            let source_idx = index_vec[0] as usize;
            if source_idx >= slab_count {
                return Err(format!("index {} at ivec[{}] exceeds slab count {}", source_idx, i, slab_count));
            }
            read_plan.push((source_idx, local_pos));
            if (local_pos + 1) % 100_000 == 0 { scan_pb.set_position((local_pos + 1) as u64); }
        }
        scan_pb.finish();

        // Step 2: Sort by source ordinal — parallel bucketed sort
        let num_buckets = 256usize;
        let bucket_range = (slab_count / num_buckets).max(1);

        let dist_pb = ctx.ui.bar(read_plan.len() as u64, &format!("bucketing {} entries{}", read_plan.len(), pass_label(pass)));
        let thread_buckets: Vec<Vec<Vec<(usize, usize)>>>;
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};
            let progress = AtomicU64::new(0);
            thread_buckets = read_plan.par_chunks(64 * 1024).map(|chunk| {
                let mut local: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_buckets];
                for &entry in chunk {
                    let b = (entry.0 / bucket_range).min(num_buckets - 1);
                    local[b].push(entry);
                }
                let done = progress.fetch_add(chunk.len() as u64, Ordering::Relaxed) + chunk.len() as u64;
                dist_pb.set_position(done);
                local
            }).collect();
        }
        dist_pb.finish();

        let mut buckets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_buckets];
        for tb in &thread_buckets {
            for (i, b) in tb.iter().enumerate() {
                buckets[i].reserve(b.len());
            }
        }
        for tb in thread_buckets {
            for (i, b) in tb.into_iter().enumerate() {
                buckets[i].extend(b);
            }
        }

        let sort_pb = ctx.ui.bar(read_plan.len() as u64, &format!("sorting {} entries{}", read_plan.len(), pass_label(pass)));
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};
            let progress = AtomicU64::new(0);
            buckets.par_iter_mut().for_each(|bucket| {
                bucket.sort_unstable_by_key(|&(src, _)| src);
                let done = progress.fetch_add(bucket.len() as u64, Ordering::Relaxed) + bucket.len() as u64;
                sort_pb.set_position(done);
            });
        }
        sort_pb.finish();

        let total_entries = buckets.iter().map(|b| b.len()).sum::<usize>();
        let merge_sp = ctx.ui.spinner(&format!("merging {} entries into read plan{}", total_entries, pass_label(pass)));
        read_plan.clear();
        read_plan.resize(total_entries, (0, 0));
        let mut offsets: Vec<usize> = Vec::with_capacity(num_buckets);
        let mut off = 0usize;
        for bucket in &buckets {
            offsets.push(off);
            off += bucket.len();
        }
        {
            use rayon::prelude::*;
            buckets.into_par_iter().enumerate().for_each(|(i, bucket)| {
                let start = offsets[i];
                let dest = unsafe {
                    std::slice::from_raw_parts_mut(
                        (read_plan.as_ptr() as *mut (usize, usize)).add(start),
                        bucket.len(),
                    )
                };
                dest.copy_from_slice(&bucket);
            });
        }
        merge_sp.finish();

        // Step 3: Read slab records in source order, collecting (local_pos, data)
        let read_pb = ctx.ui.bar_with_unit(read_plan.len() as u64, &format!("reading slab records{}", pass_label(pass)), "records");
        let mut records: Vec<(usize, Vec<u8>)> = Vec::with_capacity(part_len);
        for (i, &(source_idx, local_pos)) in read_plan.iter().enumerate() {
            let data = reader.get_ref(source_idx as i64)
                .map_err(|e| format!("failed to read slab[{}]: {}", source_idx, e))?;
            records.push((local_pos, data.to_vec()));
            if (i + 1) % 10_000 == 0 { read_pb.set_position((i + 1) as u64); }
        }
        read_pb.finish();

        // Step 4: Sort by output position and write sequentially
        let sort_out_sp = ctx.ui.spinner(&format!("sorting by output position{}", pass_label(pass)));
        records.sort_unstable_by_key(|(pos, _)| *pos);
        sort_out_sp.finish();

        let write_pb = ctx.ui.bar_with_unit(records.len() as u64, &format!("writing slab records{}", pass_label(pass)), "records");
        for (i, (_, data)) in records.iter().enumerate() {
            writer.add_record(data)
                .map_err(|e| format!("write error at record {}: {}", part_start + i, e))?;
            if (i + 1) % 10_000 == 0 { write_pb.set_position((i + 1) as u64); }
        }
        write_pb.finish();

        // Step 5: Write partition records to cache file for future resume.
        {
            let cache_config = slabtastic::WriterConfig::new(512, page_size, u32::MAX, false)
                .map_err(|e| format!("invalid cache writer config: {}", e))?;
            let mut cache_writer = slabtastic::SlabWriter::new(&cache_path, cache_config)
                .map_err(|e| format!("failed to create cache file {}: {}", cache_path.display(), e))?;
            for (_, data) in &records {
                cache_writer.add_record(data)
                    .map_err(|e| format!("cache write error: {}", e))?;
            }
            cache_writer.finish()
                .map_err(|e| format!("failed to finalize cache file: {}", e))?;
        }

        log::debug!(
            "slab-extract: pass {}/{}, wrote {} records",
            pass + 1, num_partitions, records.len(),
        );

        ctx.governor.checkpoint();
    }

    if resumed_count > 0 {
        ctx.ui.log(&format!(
            "slab-extract: resumed {} of {} partitions from cache",
            resumed_count, num_partitions,
        ));
    }

    let finalize_sp = ctx.ui.spinner("finalizing slab output");
    writer.finish()
        .map_err(|e| format!("failed to finalize output: {}", e))?;
    finalize_sp.finish();

    Ok(format!(
        "extracted {} slab records by index (ivec range [{}..{}), {} passes, {} resumed) to {}",
        extract_count, range_start, effective_end, num_partitions, resumed_count, output_path.display()
    ))
}

/// A shareable handle to a mutable byte buffer for concurrent disjoint writes.
///
/// SAFETY: The caller must ensure that concurrent writes via this handle
/// target disjoint byte ranges (no two threads write overlapping regions).
/// Stores the pointer as `usize` so references to this struct are `Send + Sync`.
#[derive(Clone, Copy)]
struct SharedBuf {
    addr: usize,
}

impl SharedBuf {
    fn new(buf: &mut [u8]) -> Self {
        SharedBuf { addr: buf.as_mut_ptr() as usize }
    }

    /// Get a mutable slice at the given byte offset.
    ///
    /// SAFETY: caller must ensure no other thread is writing to the same range.
    unsafe fn slice_mut(&self, offset: usize, len: usize) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut((self.addr + offset) as *mut u8, len) }
    }
}

unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

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

// ---- generic extract (auto-detect format) -----------------------------------

/// Pipeline command: extract records from any xvec or slab file.
///
/// Detects the source format from the file extension and delegates to the
/// appropriate format-specific extraction logic.
pub struct TransformExtractOp;

pub fn extract_factory() -> Box<dyn CommandOp> {
    Box::new(TransformExtractOp)
}

impl CommandOp for TransformExtractOp {
    fn command_path(&self) -> &str {
        "transform extract"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Extract records from vector or slab files (auto-detects format)".into(),
            body: format!(
                "# transform extract\n\n\
                Extract records from any supported file format.\n\n\
                ## Description\n\n\
                Auto-detects the source file format from its extension and delegates \
                to the appropriate format-specific extractor.\n\n\
                Supported formats:\n\
                - Vector files: `.fvec` (f32), `.mvec` (f16), `.ivec` (i32), `.dvec`, `.svec`, `.bvec`\n\
                - Container files: `.slab` (record-oriented binary — may contain vectors, \
                metadata, predicate keys, or any structured data)\n\n\
                Two extraction modes:\n\
                - **Index-based** (with `ivec-file`): select and reorder records by ordinal position\n\
                - **Range-based** (without `ivec-file`): extract a contiguous slice\n\n\
                The `normalize` option applies L2-normalization (vector formats only). \
                The `page-size` option controls slab output page size (slab format only).\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
            ResourceDesc { name: "mem".into(), description: "Memory budget for write buffering".into(), adjustable: true },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let source_path = resolve_path(source_str, &ctx.workspace);

        let ext = source_path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        // Build options for the delegate command, mapping generic names
        // to format-specific names.
        let mut delegate_opts = Options::new();
        match ext {
            "fvec" => {
                delegate_opts.set("fvec-file", source_str);
            }
            "mvec" => {
                delegate_opts.set("mvec-file", source_str);
            }
            "ivec" => {
                delegate_opts.set("ivec-file", source_str);
            }
            "slab" => {
                delegate_opts.set("slab-file", source_str);
            }
            _ => {
                return error_result(
                    format!("unsupported format '.{}' for extract — expected fvec, mvec, ivec, dvec, svec, bvec, or slab", ext),
                    start,
                );
            }
        }

        // Pass through common options
        if let Some(v) = options.get("ivec-file") { delegate_opts.set("ivec-file", v); }
        if let Some(v) = options.get("index-file") { delegate_opts.set("index-file", v); }
        if let Some(v) = options.get("output") { delegate_opts.set("output", v); }
        if let Some(v) = options.get("range") { delegate_opts.set("range", v); }
        if let Some(v) = options.get("normalize") { delegate_opts.set("normalize", v); }
        if let Some(v) = options.get("page-size") { delegate_opts.set("page-size", v); }

        // Delegate to format-specific command
        let mut cmd: Box<dyn CommandOp> = match ext {
            "fvec" => Box::new(GenerateFvecExtractOp),
            "mvec" => Box::new(GenerateMvecExtractOp),
            "ivec" => Box::new(GenerateIvecExtractOp),
            "slab" => Box::new(GenerateSlabExtractOp),
            _ => unreachable!(),
        };

        cmd.execute(&delegate_opts, ctx)
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source file (format auto-detected from extension)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "ivec-file".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Index file for indirect extraction (identity if omitted)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output file".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Record range: [start,end) or start..end".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "normalize".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "L2-normalize vectors during extraction (fvec/mvec only)".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "page-size".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("65536".to_string()),
                description: "Preferred page size for output slab (slab format only)".to_string(),
                role: OptionRole::Config,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        use crate::pipeline::command::is_cache_path;

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut intermediates = Vec::new();

        if let Some(v) = options.get("source") {
            inputs.push(v.to_string());
        }
        if let Some(v) = options.get("ivec-file") {
            inputs.push(v.to_string());
        }
        if let Some(v) = options.get("output") {
            if is_cache_path(v) {
                intermediates.push(v.to_string());
            } else {
                outputs.push(v.to_string());
            }
        }

        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs,
            outputs,
            intermediates,
        }
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
    fn test_mvec_extract_range() {
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        };

        // Generate 50 f16 vectors of dimension 8
        let mvec_path = workspace.join("source.mvec");
        let mut opts = Options::new();
        opts.set("output", mvec_path.to_string_lossy().to_string());
        opts.set("dimension", "8");
        opts.set("count", "50");
        opts.set("seed", "42");
        opts.set("type", "f16");
        let mut gen_op = GenerateVectorsOp;
        let r = gen_op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Extract first 10 records
        let out_path = workspace.join("extracted.mvec");
        let mut opts = Options::new();
        opts.set("mvec-file", mvec_path.to_string_lossy().to_string());
        opts.set("output", out_path.to_string_lossy().to_string());
        opts.set("range", "[0,10)");
        let mut ext = GenerateMvecExtractOp;
        let r = ext.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // 10 records of dim 8 (f16=2 bytes): 10 * (4 + 8*2) = 200 bytes
        let size = std::fs::metadata(&out_path).unwrap().len();
        assert_eq!(size, 10 * (4 + 8 * 2));

        // Verify extracted vectors match originals
        let orig = MmapVectorReader::<half::f16>::open_mvec(&mvec_path).unwrap();
        let extracted = MmapVectorReader::<half::f16>::open_mvec(&out_path).unwrap();
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
    fn test_mvec_extract_by_index() {
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        };

        // Generate 20 f16 vectors of dimension 8
        let mvec_path = workspace.join("source.mvec");
        let mut opts = Options::new();
        opts.set("output", mvec_path.to_string_lossy().to_string());
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
        let out_path = workspace.join("extracted.mvec");
        let mut opts = Options::new();
        opts.set("mvec-file", mvec_path.to_string_lossy().to_string());
        opts.set("ivec-file", ivec_path.to_string_lossy().to_string());
        opts.set("output", out_path.to_string_lossy().to_string());
        opts.set("range", "[0,5)");
        let mut ext = GenerateMvecExtractOp;
        let r = ext.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // 5 records of dim 8 (f16=2 bytes): 5 * (4 + 8*2) = 100 bytes
        let size = std::fs::metadata(&out_path).unwrap().len();
        assert_eq!(size, 5 * (4 + 8 * 2));

        // Verify extracted vectors match shuffled originals
        let orig = MmapVectorReader::<half::f16>::open_mvec(&mvec_path).unwrap();
        let extracted = MmapVectorReader::<half::f16>::open_mvec(&out_path).unwrap();
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
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
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
