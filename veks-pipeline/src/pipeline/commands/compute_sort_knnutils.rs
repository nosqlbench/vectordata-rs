// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: sort and deduplicate vectors (knn\_utils personality).
//!
//! Replicates the sort+dedup algorithm from knn\_utils' `fvecs_deduplicator.py`:
//!
//! - **Full lexicographic sort** on all vector components (matching Python's
//!   tuple comparison), not the prefix-based approximate sort used by the
//!   native `compute sort`.
//! - **Consecutive duplicate removal**: after sorting, adjacent identical
//!   vectors are collapsed to one, keeping the first occurrence.
//!
//! The native `compute sort` uses a prefix of N leading components for the
//! primary sort and only reads full vectors when prefixes collide. This is
//! faster but can produce a different sorted order when two vectors share
//! the same N-component prefix but differ in later components. For the
//! knn\_utils personality, exact sort-order parity is required so that
//! downstream shuffle produces the same permutation.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};

fn error_result(message: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: message.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

/// Pipeline command: full-lexicographic sort + dedup (knn\_utils compatible).
pub struct ComputeSortKnnUtilsOp;

/// Creates a boxed `ComputeSortKnnUtilsOp` for command registration.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeSortKnnUtilsOp)
}

impl CommandOp for ComputeSortKnnUtilsOp {
    fn command_path(&self) -> &str {
        "compute sort-knnutils"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Sort and deduplicate vectors using knn_utils full lexicographic order".into(),
            body: format!(r#"# compute sort-knnutils

Sort and deduplicate vectors using full lexicographic comparison on all
components, matching knn\_utils `fvecs_deduplicator.py`.

Unlike the native `compute sort` which uses a prefix-based sort, this
command compares all vector components during sorting. This ensures the
sorted order is identical to Python's `sorted(vectors, key=lambda v: tuple(v))`,
which is necessary for the knn\_utils personality pipeline to produce
byte-identical shuffle results.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let duplicates_str = match options.get("duplicates") {
            Some(s) => s.to_string(),
            None => {
                let source_path = resolve_path(source_str, &ctx.workspace);
                let stem = source_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("vectors");
                let dir = source_path.parent().unwrap_or(Path::new("."));
                dir.join(format!("{}_duplicates.ivec", stem)).to_string_lossy().to_string()
            }
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);
        let duplicates_path = resolve_path(&duplicates_str, &ctx.workspace);

        // Create output directories
        for p in [&output_path, &duplicates_path] {
            if let Some(parent) = p.parent() {
                if !parent.exists() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
        }

        // Open source vectors
        let reader = match MmapVectorReader::<f32>::open_fvec(&source_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
        };
        let count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&reader);
        let dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&reader);

        if count == 0 {
            return error_result("empty source file", start);
        }

        ctx.ui.log(&format!(
            "sort-knnutils: {} vectors, dim={}, full lexicographic sort + dedup",
            count, dim,
        ));

        // Phase 1: Build ordinal array and sort by full vector content.
        // This matches knn_utils' `chunk.sort(key=lambda x: x[0])` where
        // x[0] is the float tuple — Python tuple comparison is lexicographic
        // on all components.
        let sort_start = Instant::now();
        let sort_pb = ctx.ui.spinner(&format!("sorting {} vectors (full lexicographic)", count));

        let mut ordinals: Vec<u32> = (0..count as u32).collect();

        // Parallel sort using rayon
        use rayon::prelude::*;
        let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        let sort_fn = |ords: &mut Vec<u32>| {
            ords.par_sort_unstable_by(|&a, &b| {
                let va = reader.get_slice(a as usize);
                let vb = reader.get_slice(b as usize);
                // Full lexicographic comparison on all components
                for i in 0..dim {
                    match va[i].partial_cmp(&vb[i]) {
                        Some(std::cmp::Ordering::Equal) => continue,
                        Some(ord) => return ord,
                        None => {
                            // Handle NaN: treat NaN as greater than all values
                            let a_nan = va[i].is_nan();
                            let b_nan = vb[i].is_nan();
                            if a_nan && b_nan { continue; }
                            if a_nan { return std::cmp::Ordering::Greater; }
                            return std::cmp::Ordering::Less;
                        }
                    }
                }
                // Fully equal vectors: stable tiebreak by ordinal
                a.cmp(&b)
            });
        };

        if let Some(ref p) = pool {
            p.install(|| sort_fn(&mut ordinals));
        } else {
            sort_fn(&mut ordinals);
        }
        sort_pb.finish();

        let sort_secs = sort_start.elapsed().as_secs_f64();
        ctx.ui.log(&format!("  sorted in {:.1}s ({} threads)", sort_secs, threads));

        // Phase 2: Deduplicate by scanning consecutive pairs.
        // Matches knn_utils: keep first occurrence, mark rest as duplicates.
        let dedup_start = Instant::now();
        let dedup_pb = ctx.ui.bar_with_unit(count as u64, "dedup", "vectors");

        let mut unique_ordinals: Vec<u32> = Vec::with_capacity(count);
        let mut dup_ordinals: Vec<u32> = Vec::new();

        unique_ordinals.push(ordinals[0]);
        dedup_pb.inc(1);

        for i in 1..ordinals.len() {
            let prev = reader.get_slice(ordinals[i - 1] as usize);
            let curr = reader.get_slice(ordinals[i] as usize);

            if prev == curr {
                dup_ordinals.push(ordinals[i]);
            } else {
                unique_ordinals.push(ordinals[i]);
            }

            if (i + 1) % 100_000 == 0 {
                dedup_pb.set_position((i + 1) as u64);
            }
        }
        dedup_pb.finish();

        let dedup_secs = dedup_start.elapsed().as_secs_f64();
        let unique_count = unique_ordinals.len();
        let dup_count = dup_ordinals.len();
        ctx.ui.log(&format!(
            "  dedup: {:.1}s, {} unique, {} duplicates ({:.2}%)",
            dedup_secs, unique_count, dup_count,
            if count > 0 { 100.0 * dup_count as f64 / count as f64 } else { 0.0 },
        ));

        // Phase 3: Write output ivec (sorted unique ordinals)
        let write_start = Instant::now();
        let write_pb = ctx.ui.bar(unique_count as u64, "write sorted");

        let mut writer = match AtomicWriter::new(&output_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
        };
        let dim_1: i32 = 1;
        let batch = 10_000.max(unique_count / 1000);
        for (i, &ord) in unique_ordinals.iter().enumerate() {
            writer.write_all(&dim_1.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            writer.write_all(&(ord as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            if (i + 1) % batch == 0 {
                write_pb.inc(batch as u64);
            }
        }
        write_pb.set_position(unique_count as u64);
        writer.finish().map_err(|e| e.to_string()).unwrap();
        write_pb.finish();

        // Write duplicates ivec
        let dup_pb = ctx.ui.bar(dup_count as u64, "write dups");
        let mut dup_writer = match AtomicWriter::new(&duplicates_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("create {}: {}", duplicates_path.display(), e), start),
        };
        for (i, &ord) in dup_ordinals.iter().enumerate() {
            dup_writer.write_all(&dim_1.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            dup_writer.write_all(&(ord as i32).to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            if (i + 1) % batch == 0 {
                dup_pb.inc(batch as u64);
            }
        }
        dup_pb.set_position(dup_count as u64);
        dup_writer.finish().map_err(|e| e.to_string()).unwrap();
        dup_pb.finish();

        let write_secs = write_start.elapsed().as_secs_f64();
        ctx.ui.log(&format!("  wrote in {:.1}s", write_secs));

        // Write verified counts
        let set_var = |ctx: &mut StreamContext, name: &str, value: &str| {
            let _ = crate::pipeline::variables::set_and_save(&ctx.workspace, name, value);
            ctx.defaults.insert(name.to_string(), value.to_string());
        };
        let out_var = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        set_var(ctx, &out_var, &unique_count.to_string());
        let dup_var = format!("verified_count:{}",
            duplicates_path.file_name().and_then(|n| n.to_str()).unwrap_or("dups"));
        set_var(ctx, &dup_var, &dup_count.to_string());

        let total_secs = start.elapsed().as_secs_f64();
        let produced = vec![output_path, duplicates_path];

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} vectors -> {} unique, {} dup ({:.2}%) elided, {:.1}s (full lexicographic sort)",
                count, unique_count, dup_count,
                if count > 0 { 100.0 * dup_count as f64 / count as f64 } else { 0.0 },
                total_secs,
            ),
            produced,
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
                description: "Source fvec file".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output sorted unique ordinals (ivec)".to_string(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "duplicates".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output duplicate ordinals (ivec)".to_string(),
                role: OptionRole::Output,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output", "duplicates"],
        )
    }
}
