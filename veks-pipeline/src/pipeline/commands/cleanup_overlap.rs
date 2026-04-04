// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: remove overlapping query vectors from the query set.
//!
//! Hashes every base vector, then rewrites the query file excluding any
//! vectors whose hash matches a base vector. The original query file is
//! replaced atomically. Reports the number of vectors removed.
//!
//! This is the fix counterpart to `analyze overlap`, which only detects
//! the problem. When overlap count is zero the query file is untouched.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use crate::pipeline::element_type::ElementType;

pub struct CleanupOverlapOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(CleanupOverlapOp)
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(path_str);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

impl CommandOp for CleanupOverlapOp {
    fn command_path(&self) -> &str {
        "cleanup overlap"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Remove query vectors that overlap with the base set".into(),
            body: format!(
                "# cleanup overlap\n\n\
                 Remove query vectors that also appear in the base set.\n\n\
                 ## Description\n\n\
                 Hashes every base vector, then rewrites the query file \
                 excluding any vectors whose hash matches. When no overlaps \
                 are found the query file is left untouched.\n\n\
                 ## Why\n\n\
                 Overlapping vectors make KNN ground truth unreliable — a \
                 query that exists in the base set has itself as nearest \
                 neighbor at distance 0.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        // Output path: when provided, cleaned queries are written here
        // instead of overwriting the input. This preserves the directional
        // pipeline model — upstream outputs are never modified.
        let output_str = options.get("output");

        let base_path = resolve_path(base_str, &ctx.workspace);
        let query_path = resolve_path(query_str, &ctx.workspace);
        let output_path = output_str.map(|s| resolve_path(s, &ctx.workspace));

        let base_etype = match ElementType::from_path(&base_path) {
            Ok(e) => e,
            Err(e) => return error_result(format!("base: {}", e), start),
        };
        let query_etype = match ElementType::from_path(&query_path) {
            Ok(e) => e,
            Err(e) => return error_result(format!("query: {}", e), start),
        };

        if base_etype != query_etype {
            return error_result(
                format!("element type mismatch: base={}, query={}", base_etype, query_etype),
                start,
            );
        }

        let elem_size = base_etype.element_size();

        let write_path = output_path.as_ref().unwrap_or(&query_path);
        let result = fix_overlap(&base_path, &query_path, write_path, elem_size, &ctx.ui);

        let (base_count, original_query_count, kept, removed) = match result {
            Ok(r) => r,
            Err(msg) => return error_result(msg, start),
        };

        // Save metrics as variables
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "overlap_count", &removed.to_string(),
        );
        let overlap_frac = if original_query_count > 0 {
            removed as f64 / original_query_count as f64
        } else {
            0.0
        };
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "overlap_fraction", &format!("{:.6}", overlap_frac),
        );
        ctx.defaults.insert("overlap_count".into(), removed.to_string());
        ctx.defaults.insert("overlap_fraction".into(), format!("{:.6}", overlap_frac));

        let message = format!(
            "base={} query={} removed={} kept={}",
            base_count, original_query_count, removed, kept,
        );

        if removed > 0 {
            ctx.ui.log(&format!(
                "  Removed {} overlapping query vectors ({} → {})",
                removed, original_query_count, kept,
            ));
        } else {
            ctx.ui.log(&format!(
                "  No overlaps — base and query sets are disjoint ({} base, {} query)",
                base_count, original_query_count,
            ));
        }

        CommandResult {
            status: Status::Ok,
            message,
            produced: if output_path.is_some() { vec![write_path.clone()] } else { vec![] },
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Base vectors file (read-only)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "query".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Query vectors file (read-only input)".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output path for cleaned query vectors (if omitted, query file is rewritten in place)".to_string(),
                role: OptionRole::Output,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query"],
            &["query"],
        )
    }
}

/// Hash the raw bytes of a vector for fast set membership (FNV-1a).
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for chunk in bytes.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let word = u64::from_le_bytes(buf);
        h ^= word;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Read a file fully into memory.
fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path)
        .map_err(|e| format!("read {}: {}", path.display(), e))
}

/// Detect and remove overlapping query vectors.
///
/// `write_path` is where the cleaned query file is written. It may be the
/// same as `query_path` (legacy in-place mode) or a separate output file
/// (directional pipeline mode).
///
/// Returns `(base_count, original_query_count, kept_count, removed_count)`.
fn fix_overlap(
    base_path: &Path,
    query_path: &Path,
    write_path: &Path,
    elem_size: usize,
    ui: &veks_core::ui::UiHandle,
) -> Result<(usize, usize, usize, usize), String> {
    let base_data = read_file(base_path)?;
    let query_data = read_file(query_path)?;

    if base_data.len() < 4 {
        return Err("base file too small to contain a dimension header".into());
    }
    if query_data.len() < 4 {
        return Err("query file too small to contain a dimension header".into());
    }

    let base_dim = u32::from_le_bytes(base_data[..4].try_into().unwrap()) as usize;
    let query_dim = u32::from_le_bytes(query_data[..4].try_into().unwrap()) as usize;

    if base_dim != query_dim {
        return Err(format!("dimension mismatch: base={}, query={}", base_dim, query_dim));
    }

    let record_size = 4 + base_dim * elem_size; // 4-byte dim header + data
    let base_count = base_data.len() / record_size;
    let query_count = query_data.len() / record_size;

    // Phase 1: hash all base vectors
    ui.log(&format!("  hashing {} base vectors (dim={})", base_count, base_dim));
    let pb = ui.bar(base_count as u64, "hashing base vectors");
    let mut base_hashes = HashSet::with_capacity(base_count);
    for i in 0..base_count {
        let offset = i * record_size;
        let values = &base_data[offset + 4..offset + record_size];
        base_hashes.insert(hash_bytes(values));
        if (i + 1) % 100_000 == 0 { pb.set_position((i + 1) as u64); }
    }
    pb.finish();

    // Phase 2: scan query vectors for overlaps
    ui.log(&format!("  checking {} query vectors for overlap", query_count));
    let pb = ui.bar(query_count as u64, "checking query vectors");
    let mut overlap_indices: Vec<bool> = Vec::with_capacity(query_count);
    let mut overlap_count = 0usize;
    for i in 0..query_count {
        let offset = i * record_size;
        let values = &query_data[offset + 4..offset + record_size];
        let is_overlap = base_hashes.contains(&hash_bytes(values));
        if is_overlap { overlap_count += 1; }
        overlap_indices.push(is_overlap);
        if (i + 1) % 100_000 == 0 { pb.set_position((i + 1) as u64); }
    }
    pb.finish();

    if overlap_count == 0 {
        // No overlaps — if writing to a separate output, copy the input
        // so downstream steps have the expected file.
        if write_path != query_path {
            crate::pipeline::atomic_write::guard_against_symlink_write(write_path)
                .map_err(|e| format!("symlink guard {}: {}", write_path.display(), e))?;
            std::fs::copy(query_path, write_path)
                .map_err(|e| format!("copy {} → {}: {}", query_path.display(), write_path.display(), e))?;
        }
        return Ok((base_count, query_count, query_count, 0));
    }

    // Phase 3: rewrite query file excluding overlapping vectors
    let kept = query_count - overlap_count;
    ui.log(&format!("  rewriting query file: {} → {} vectors", query_count, kept));
    let pb = ui.bar(query_count as u64, "rewriting queries");

    let mut writer = AtomicWriter::new(write_path)
        .map_err(|e| format!("create output {}: {}", write_path.display(), e))?;

    for i in 0..query_count {
        if !overlap_indices[i] {
            let offset = i * record_size;
            let record = &query_data[offset..offset + record_size];
            writer.write_all(record)
                .map_err(|e| format!("write error: {}", e))?;
        }
        if (i + 1) % 100_000 == 0 { pb.set_position((i + 1) as u64); }
    }
    pb.finish();

    writer.finish()
        .map_err(|e| format!("finalize output: {}", e))?;

    Ok((base_count, query_count, kept, overlap_count))
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}
