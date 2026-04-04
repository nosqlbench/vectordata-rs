// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: clean/repair xvec files.
//!
//! Detects and removes:
//! - Trailing bytes (truncated/incomplete records)
//! - Zero vectors (all dimensions = 0)
//! - Duplicate vectors
//!
//! Writes a cleaned output file and reports what was removed.
//!
//! Equivalent to the Java `CMD_cleanfvec` command (which was a stub — this
//! is a complete implementation).

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: clean/repair fvec files.
pub struct CleanupCleanfvecOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(CleanupCleanfvecOp)
}

/// Result of cleaning a file.
struct CleanResult {
    original_count: usize,
    written_count: usize,
    trailing_bytes_removed: u64,
    zero_vectors_removed: usize,
    duplicate_vectors_removed: usize,
}

impl CommandOp for CleanupCleanfvecOp {
    fn command_path(&self) -> &str {
        "cleanup cleanfvec"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Clean and repair a malformed fvec file".into(),
            body: format!(
                r#"# cleanup cleanfvec

Clean and repair a malformed fvec file.

## Description

Detects and removes trailing bytes, zero vectors, and duplicate
vectors from an fvec (float32) vector file. Writes a cleaned output
file and reports exactly what was removed: trailing byte count, zero
vectors removed, and duplicate vectors removed.

## How It Works

The command reads the source fvec file, determines the dimension from
the first record's 4-byte header, and computes the per-vector byte
size. It then iterates over all complete vectors in the file, applying
the enabled filters in order:

1. **Trailing bytes**: Any bytes beyond the last complete vector are
   silently discarded (they indicate truncation or corruption).
2. **Zero vectors** (enabled by default): Vectors where every
   component is zero are removed, as they are typically artifacts of
   failed data extraction.
3. **Duplicate vectors** (disabled by default): Vectors with identical
   content (detected via FNV-1a hashing) are deduplicated, keeping
   only the first occurrence.

Vectors that pass all filters are written to the output file.

## Data Preparation Role

`cleanup cleanfvec` is a data hygiene step applied after downloading
or extracting vector files. Malformed fvec files are common when
downloading from external sources -- partial HTTP transfers leave
trailing bytes, embedding extraction failures produce zero vectors,
and data pipeline bugs can introduce duplicates. Running this command
before index building ensures that the vector file contains only valid,
unique vectors, preventing wasted storage and incorrect KNN results.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Read/write buffers for vector repair".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
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

        let remove_zeros = options.get("remove-zeros").map_or(true, |s| s != "false");
        let remove_dupes = options.get("remove-duplicates").map_or(false, |s| s == "true");

        let source_path = resolve_path(source_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        let data = match std::fs::read(&source_path) {
            Ok(d) => d,
            Err(e) => {
                return error_result(
                    format!("failed to read {}: {}", source_path.display(), e),
                    start,
                )
            }
        };

        if data.len() < 4 {
            return error_result(
                format!("file too small ({} bytes)", data.len()),
                start,
            );
        }

        // Read dimension from first record
        let dim = i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if dim == 0 {
            return error_result("dimension is 0".to_string(), start);
        }

        let bytes_per_element = 4usize; // fvec = f32
        let bytes_per_vector = 4 + dim * bytes_per_element;
        let complete_vectors = data.len() / bytes_per_vector;
        let trailing = data.len() as u64 - (complete_vectors * bytes_per_vector) as u64;

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("failed to create directory: {}", e), start);
                }
            }
        }

        let mut writer = match AtomicWriter::new(&output_path) {
            Ok(w) => w,
            Err(e) => {
                return error_result(
                    format!("failed to create {}: {}", output_path.display(), e),
                    start,
                )
            }
        };

        let mut written = 0usize;
        let mut zeros_removed = 0usize;
        let mut dupes_removed = 0usize;
        let mut seen_hashes: HashSet<u64> = HashSet::new();

        // Governor checkpoint before processing loop
        if ctx.governor.checkpoint() {
            ctx.ui.log("  governor: throttle active");
        }

        let threads = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        // Precompute hashes in parallel when dedup is enabled
        let hashes: Vec<u64> = if remove_dupes {
            let hash_pb = ctx.ui.bar_with_unit(complete_vectors as u64, "hashing vectors", "vectors");
            let progress = AtomicU64::new(0);
            let progress_ref = &progress;
            let hash_pb_ref = &hash_pb;

            let hash_fn = || {
                (0..complete_vectors)
                    .into_par_iter()
                    .map(|i| {
                        let offset = i * bytes_per_vector;
                        let values_bytes = &data[offset + 4..offset + bytes_per_vector];
                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 100_000 == 0 {
                            hash_pb_ref.set_position(done);
                        }
                        simple_hash(values_bytes)
                    })
                    .collect::<Vec<_>>()
            };

            let result = if let Some(ref p) = pool {
                p.install(hash_fn)
            } else {
                hash_fn()
            };
            hash_pb.finish();
            result
        } else {
            Vec::new()
        };

        let pb = ctx.ui.bar_with_unit(complete_vectors as u64, "cleaning", "vectors");
        for i in 0..complete_vectors {
            let offset = i * bytes_per_vector;
            let record = &data[offset..offset + bytes_per_vector];
            let values_bytes = &record[4..]; // skip dim header

            // Check for zero vector
            if remove_zeros {
                let is_zero = values_bytes.iter().all(|&b| b == 0);
                if is_zero {
                    zeros_removed += 1;
                    if (i + 1) % 10_000 == 0 { pb.set_position((i + 1) as u64); }
                    continue;
                }
            }

            // Check for duplicate using precomputed hash
            if remove_dupes {
                if !seen_hashes.insert(hashes[i]) {
                    dupes_removed += 1;
                    if (i + 1) % 10_000 == 0 { pb.set_position((i + 1) as u64); }
                    continue;
                }
            }

            writer.write_all(record).map_err(|e| e.to_string()).unwrap();
            written += 1;
            if (i + 1) % 10_000 == 0 { pb.set_position((i + 1) as u64); }
        }
        pb.finish();

        writer.finish().map_err(|e| format!("failed to finalize {}: {}", output_path.display(), e)).unwrap();

        let result = CleanResult {
            original_count: complete_vectors,
            written_count: written,
            trailing_bytes_removed: trailing,
            zero_vectors_removed: zeros_removed,
            duplicate_vectors_removed: dupes_removed,
        };

        let removed_total = result.original_count - result.written_count;
        ctx.ui.log(&format!(
            "Clean: {} -> {} vectors ({} removed)",
            result.original_count, result.written_count, removed_total
        ));
        if result.trailing_bytes_removed > 0 {
            ctx.ui.log(&format!("  Trailing bytes removed: {}", result.trailing_bytes_removed));
        }
        if result.zero_vectors_removed > 0 {
            ctx.ui.log(&format!("  Zero vectors removed: {}", result.zero_vectors_removed));
        }
        if result.duplicate_vectors_removed > 0 {
            ctx.ui.log(&format!("  Duplicate vectors removed: {}", result.duplicate_vectors_removed));
        }

        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &result.written_count.to_string());
        ctx.defaults.insert(var_name, result.written_count.to_string());

        CommandResult {
            status: Status::Ok,
            message: format!(
                "cleaned {} -> {} vectors (dim={}, {} removed, {} trailing bytes)",
                result.original_count,
                result.written_count,
                dim,
                removed_total,
                result.trailing_bytes_removed
            ),
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
                description: "Input fvec file to clean".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output cleaned fvec file".to_string(),
                        role: OptionRole::Output,
        },
            OptionDesc {
                name: "remove-zeros".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("true".to_string()),
                description: "Remove zero vectors".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "remove-duplicates".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Remove duplicate vectors (hash-based)".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Simple non-cryptographic hash for duplicate detection.
fn simple_hash(bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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

    /// Write test fvec from f32 vectors.
    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for vec in vectors {
            let dim = vec.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for &v in vec {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    fn read_fvec(path: &Path) -> Vec<Vec<f32>> {
        let data = std::fs::read(path).unwrap();
        let mut result = Vec::new();
        let mut offset = 0;
        while offset + 4 <= data.len() {
            let dim = i32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]) as usize;
            offset += 4;
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                let v = f32::from_le_bytes([
                    data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                ]);
                vec.push(v);
                offset += 4;
            }
            result.push(vec);
        }
        result
    }

    #[test]
    fn test_clean_removes_trailing_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        let source = ws.join("input.fvec");
        {
            let mut f = std::fs::File::create(&source).unwrap();
            // 2 complete vectors (dim=2) + 5 trailing bytes
            for _ in 0..2 {
                f.write_all(&2i32.to_le_bytes()).unwrap();
                f.write_all(&1.0f32.to_le_bytes()).unwrap();
                f.write_all(&2.0f32.to_le_bytes()).unwrap();
            }
            f.write_all(&[0, 0, 0, 0, 0]).unwrap();
        }

        let output = ws.join("output.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("remove-zeros", "false");

        let mut op = CleanupCleanfvecOp;
        let mut ctx = test_ctx(ws);
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("trailing bytes"));

        let vecs = read_fvec(&output);
        assert_eq!(vecs.len(), 2);
    }

    #[test]
    fn test_clean_removes_zero_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        let source = ws.join("input.fvec");
        write_fvec(
            &source,
            &[
                vec![1.0, 2.0],
                vec![0.0, 0.0], // zero vector
                vec![3.0, 4.0],
            ],
        );

        let output = ws.join("output.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());

        let mut op = CleanupCleanfvecOp;
        let mut ctx = test_ctx(ws);
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let vecs = read_fvec(&output);
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![1.0, 2.0]);
        assert_eq!(vecs[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_clean_removes_duplicates() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        let source = ws.join("input.fvec");
        write_fvec(
            &source,
            &[
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![1.0, 2.0], // duplicate
                vec![5.0, 6.0],
            ],
        );

        let output = ws.join("output.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("remove-zeros", "false");
        opts.set("remove-duplicates", "true");

        let mut op = CleanupCleanfvecOp;
        let mut ctx = test_ctx(ws);
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let vecs = read_fvec(&output);
        assert_eq!(vecs.len(), 3);
    }

    #[test]
    fn test_clean_no_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        let source = ws.join("input.fvec");
        write_fvec(&source, &[vec![1.0, 2.0], vec![3.0, 4.0]]);

        let output = ws.join("output.fvec");
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("output", output.to_string_lossy().to_string());

        let mut op = CleanupCleanfvecOp;
        let mut ctx = test_ctx(ws);
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("0 removed"));
    }

    #[test]
    fn test_fnv_hash() {
        let a = simple_hash(&[1, 2, 3, 4]);
        let b = simple_hash(&[1, 2, 3, 4]);
        let c = simple_hash(&[4, 3, 2, 1]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
