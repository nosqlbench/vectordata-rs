// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: produce a clean ordinal index.
//!
//! Takes a sorted ordinal index (from `compute dedup`) and exclusion lists
//! (duplicate ordinals, zero-vector ordinals) and writes a new index
//! containing only the clean ordinals. This clean index becomes the
//! upstream artifact for shuffle and extraction.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use byteorder::{LittleEndian, WriteBytesExt};
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

/// Pipeline command: filter ordinals to produce a clean index.
pub struct CleanOrdinalsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(CleanOrdinalsOp)
}

impl CommandOp for CleanOrdinalsOp {
    fn command_path(&self) -> &str {
        "transform ordinals"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Filter ordinal index by excluding duplicates and zeros".into(),
            body: format!(r#"# transform ordinals

Filter an ordinal index by excluding duplicate and zero-vector ordinals.

## Description

Reads a sorted ordinal index (typically from `compute dedup`) and one or
more exclusion lists (duplicate ordinals, zero-vector ordinals). Writes a
new ordinal index containing only the entries not present in any exclusion
list.

The output is the **clean ordinal set** — the set of original vector
ordinals that are unique, non-zero, and suitable for downstream processing.
This index is used as the upstream artifact for `generate ivec-shuffle`
and the extract commands, ensuring that duplicates and degenerate vectors
are excluded from the final dataset.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Exclusion set memory".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Load exclusion sets
        let mut exclude: HashSet<u32> = HashSet::new();
        let mut exclude_sources: Vec<String> = Vec::new();

        if let Some(dups_str) = options.get("duplicates") {
            let dups_path = resolve_path(dups_str, &ctx.workspace);
            if dups_path.exists() {
                match load_ordinals_into_set(&dups_path, &mut exclude) {
                    Ok(count) => exclude_sources.push(format!("{} duplicates", count)),
                    Err(e) => return error_result(e, start),
                }
            }
        }

        if let Some(zeros_str) = options.get("zeros") {
            let zeros_path = resolve_path(zeros_str, &ctx.workspace);
            if zeros_path.exists() {
                match load_ordinals_into_set(&zeros_path, &mut exclude) {
                    Ok(count) => exclude_sources.push(format!("{} zeros", count)),
                    Err(e) => return error_result(e, start),
                }
            }
        }

        ctx.ui.log(&format!(
            "Clean ordinals: excluding {} ordinals ({})",
            exclude.len(),
            if exclude_sources.is_empty() { "none".to_string() } else { exclude_sources.join(", ") },
        ));

        // If nothing to exclude, just copy the source
        if exclude.is_empty() {
            if source_path != output_path {
                ensure_parent(&output_path);
                if let Err(e) = std::fs::copy(&source_path, &output_path) {
                    return error_result(format!("failed to copy: {}", e), start);
                }
            }
            let count = match ivec_record_count(&source_path) {
                Ok(c) => c,
                Err(e) => return error_result(e, start),
            };
            return CommandResult {
                status: Status::Ok,
                message: format!("{} ordinals (no exclusions needed)", count),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            };
        }

        // Read source ordinals, filter, write output
        let reader = match MmapVectorReader::<i32>::open_ivec(&source_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open {}: {}", source_path.display(), e), start),
        };

        let total = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
        let mut writer = match AtomicWriter::new(&output_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("failed to create {}: {}", output_path.display(), e), start),
        };

        let pb = ctx.ui.bar_with_unit(total as u64, "filtering ordinals", "ordinals");
        let mut kept = 0u64;
        let mut excluded = 0u64;

        // Bound mmap RSS during a full sequential scan.
        let mut reclaim = vectordata::io::StreamReclaim::new(&reader, 0, total);

        for i in 0..total {
            let vec = match reader.get(i) {
                Ok(v) => v,
                Err(e) => return error_result(format!("read error at {}: {}", i, e), start),
            };
            reclaim.advance(i);
            let ordinal = vec[0] as u32;

            if exclude.contains(&ordinal) {
                excluded += 1;
            } else {
                if let Err(e) = writer.write_i32::<LittleEndian>(1)
                    .and_then(|_| writer.write_i32::<LittleEndian>(ordinal as i32))
                {
                    return error_result(format!("write error: {}", e), start);
                }
                kept += 1;
            }

            if (i + 1) % 100_000 == 0 {
                pb.set_position((i + 1) as u64);
            }
        }
        pb.finish();

        if let Err(e) = writer.finish() {
            return error_result(format!("failed to finalize {}: {}", output_path.display(), e), start);
        }

        // Write verified count for the bound checker
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &kept.to_string());
        ctx.defaults.insert(var_name, kept.to_string());

        let msg = format!(
            "{} kept, {} excluded (of {} total)",
            kept, excluded, total,
        );
        ctx.ui.log(&msg);

        CommandResult {
            status: Status::Ok,
            message: msg,
            produced: vec![output_path],
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
                description: "Input sorted ordinal index (from compute dedup)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "duplicates".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Duplicate ordinals to exclude (ivec)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "zeros".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Zero-vector ordinals to exclude (ivec)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output clean ordinal index (ivec)".to_string(),
                role: OptionRole::Output,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source", "duplicates", "zeros"],
            &["output"],
        )
    }
}

/// Load ordinals from a dim=1 ivec file into a HashSet.
fn load_ordinals_into_set(path: &Path, set: &mut HashSet<u32>) -> Result<usize, String> {
    let reader = MmapVectorReader::<i32>::open_ivec(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
    for i in 0..count {
        let vec = reader.get(i)
            .map_err(|e| format!("failed to read ordinal at {}: {}", i, e))?;
        set.insert(vec[0] as u32);
    }
    Ok(count)
}

/// Count records in a dim=1 ivec file. Returns 0 for empty files.
fn ivec_record_count(path: &Path) -> Result<usize, String> {
    let reader = MmapVectorReader::<i32>::open_ivec(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    Ok(<MmapVectorReader<i32> as VectorReader<i32>>::count(&reader))
}

fn ensure_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}
