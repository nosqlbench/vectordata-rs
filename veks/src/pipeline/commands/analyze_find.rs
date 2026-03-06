// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: find a vector from one file in another.
//!
//! Reads a vector at a given index from a source file, then searches for it
//! in a target file using either binary search (if sorted) or exhaustive scan.
//!
//! Equivalent to the Java `CMD_analyze_find` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: find a specific vector in a target file.
pub struct AnalyzeFindOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeFindOp)
}

impl CommandOp for AnalyzeFindOp {
    fn command_path(&self) -> &str {
        "analyze find"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Search for vectors matching a value pattern".into(),
            body: format!(
                "# analyze find\n\nSearch for vectors matching a value pattern.\n\n## Description\n\nReads a vector at a given index from a source file, then searches for it in a target file using either binary search (if sorted) or exhaustive scan.\n\n## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Vector data buffers".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let target_str = match options.require("target") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let index: usize = match options.require("index") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid index: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let target_path = resolve_path(target_str, &ctx.workspace);

        // Open source and get the needle vector
        let source_reader = match MmapVectorReader::<f32>::open_fvec(&source_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open source: {}", e),
                    start,
                )
            }
        };

        let source_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&source_reader);
        if index >= source_count {
            return error_result(
                format!(
                    "index {} out of range (source has {} vectors)",
                    index, source_count
                ),
                start,
            );
        }

        let needle = match source_reader.get(index) {
            Ok(v) => v.to_vec(),
            Err(e) => return error_result(format!("failed to read source[{}]: {}", index, e), start),
        };
        let dim = needle.len();

        ctx.display.log(&format!(
            "Searching for source[{}] (dim={}) in {}",
            index,
            dim,
            target_path.display()
        ));

        // Open target
        let target_reader = match MmapVectorReader::<f32>::open_fvec(&target_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open target: {}", e),
                    start,
                )
            }
        };

        let target_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&target_reader);
        let target_dim = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&target_reader);

        if dim != target_dim {
            return error_result(
                format!(
                    "dimension mismatch: source dim={}, target dim={}",
                    dim, target_dim
                ),
                start,
            );
        }

        // Check if target is sorted (sample first and last vectors)
        let is_sorted = check_sorted(&target_reader, target_count);

        if is_sorted {
            ctx.display.log("Target appears sorted — using binary search");
            match binary_search(&target_reader, &needle, target_count) {
                Some(found_idx) => {
                    ctx.display.log(&format!("FOUND at index {}", found_idx));
                    CommandResult {
                        status: Status::Ok,
                        message: format!("found at index {}", found_idx),
                        produced: vec![],
                        elapsed: start.elapsed(),
                    }
                }
                None => {
                    ctx.display.log("NOT FOUND (binary search)");
                    CommandResult {
                        status: Status::Warning,
                        message: "not found".to_string(),
                        produced: vec![],
                        elapsed: start.elapsed(),
                    }
                }
            }
        } else {
            ctx.display.log(&format!(
                "Target is unsorted — exhaustive scan ({} vectors)",
                target_count
            ));
            let result = exhaustive_scan(&target_reader, &needle, target_count);
            match result {
                Some((found_idx, exact)) => {
                    if exact {
                        ctx.display.log(&format!("FOUND exact match at index {}", found_idx));
                        CommandResult {
                            status: Status::Ok,
                            message: format!("found exact match at index {}", found_idx),
                            produced: vec![],
                            elapsed: start.elapsed(),
                        }
                    } else {
                        ctx.display.log(&format!("Closest match at index {} (not exact)", found_idx));
                        CommandResult {
                            status: Status::Warning,
                            message: format!("closest match at index {} (not exact)", found_idx),
                            produced: vec![],
                            elapsed: start.elapsed(),
                        }
                    }
                }
                None => {
                    ctx.display.log("NOT FOUND");
                    CommandResult {
                        status: Status::Warning,
                        message: "not found".to_string(),
                        produced: vec![],
                        elapsed: start.elapsed(),
                    }
                }
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Source file containing the vector to find".to_string(),
            },
            OptionDesc {
                name: "target".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Target file to search in".to_string(),
            },
            OptionDesc {
                name: "index".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "0-based index of vector in source file".to_string(),
            },
        ]
    }
}

/// Check if a file appears lexicographically sorted by sampling.
fn check_sorted(reader: &MmapVectorReader<f32>, count: usize) -> bool {
    if count <= 1 {
        return true;
    }
    let sample = 100.min(count - 1);
    for i in 0..sample {
        let a = match reader.get(i) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let b = match reader.get(i + 1) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if compare_vectors(&a, &b) == std::cmp::Ordering::Greater {
            return false;
        }
    }
    // Also check last few
    if count > sample + 2 {
        for i in (count - sample)..count - 1 {
            let a = match reader.get(i) {
                Ok(v) => v,
                Err(_) => return false,
            };
            let b = match reader.get(i + 1) {
                Ok(v) => v,
                Err(_) => return false,
            };
            if compare_vectors(&a, &b) == std::cmp::Ordering::Greater {
                return false;
            }
        }
    }
    true
}

/// Lexicographic comparison of two float vectors.
fn compare_vectors(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
    for i in 0..a.len().min(b.len()) {
        match a[i].partial_cmp(&b[i]) {
            Some(std::cmp::Ordering::Equal) => continue,
            Some(ord) => return ord,
            None => {
                // NaN handling: treat NaN as greater
                if a[i].is_nan() && !b[i].is_nan() {
                    return std::cmp::Ordering::Greater;
                }
                if !a[i].is_nan() && b[i].is_nan() {
                    return std::cmp::Ordering::Less;
                }
                continue;
            }
        }
    }
    a.len().cmp(&b.len())
}

/// Binary search for a vector in a sorted file.
fn binary_search(
    reader: &MmapVectorReader<f32>,
    needle: &[f32],
    count: usize,
) -> Option<usize> {
    let mut low = 0usize;
    let mut high = count;
    while low < high {
        let mid = low + (high - low) / 2;
        let vec = reader.get(mid).ok()?;
        match compare_vectors(&vec, needle) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
        }
    }
    None
}

/// Exhaustive scan returning (index, is_exact_match).
fn exhaustive_scan(
    reader: &MmapVectorReader<f32>,
    needle: &[f32],
    count: usize,
) -> Option<(usize, bool)> {
    let mut best_idx = 0;
    let mut best_matching = 0usize;

    for i in 0..count {
        let vec = match reader.get(i) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if vec.as_slice() == needle {
            return Some((i, true));
        }
        let matching = vec
            .iter()
            .zip(needle.iter())
            .filter(|(a, b)| (*a - *b).abs() < f32::EPSILON)
            .count();
        if matching > best_matching {
            best_matching = matching;
            best_idx = i;
        }
    }

    if best_matching > 0 {
        Some((best_idx, false))
    } else if count > 0 {
        Some((0, false))
    } else {
        None
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
    use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
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
    fn test_find_exact_match() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate vectors (same file used as both source and target)
        let path = ws.join("test.fvec");
        let mut gen_opts = Options::new();
        gen_opts.set("output", path.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "20");
        gen_opts.set("seed", "42");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        // Find vector 5 in same file — should find exact match
        let mut opts = Options::new();
        opts.set("source", path.to_string_lossy().to_string());
        opts.set("target", path.to_string_lossy().to_string());
        opts.set("index", "5");
        let mut op = AnalyzeFindOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "find failed: {}", result.message);
        assert!(result.message.contains("found"));
    }

    #[test]
    fn test_find_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Two files with different seeds — unlikely to match
        let src = ws.join("src.fvec");
        let tgt = ws.join("tgt.fvec");

        let mut gen_opts = Options::new();
        gen_opts.set("output", src.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "5");
        gen_opts.set("seed", "1");
        let mut gen_op = GenerateVectorsOp;
        gen_op.execute(&gen_opts, &mut ctx);

        let mut gen_opts = Options::new();
        gen_opts.set("output", tgt.to_string_lossy().to_string());
        gen_opts.set("dimension", "4");
        gen_opts.set("count", "5");
        gen_opts.set("seed", "999");
        gen_op.execute(&gen_opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", src.to_string_lossy().to_string());
        opts.set("target", tgt.to_string_lossy().to_string());
        opts.set("index", "0");
        let mut op = AnalyzeFindOp;
        let result = op.execute(&opts, &mut ctx);
        // Should be warning (closest match, not exact)
        assert_ne!(result.status, Status::Error);
    }

    #[test]
    fn test_compare_vectors() {
        assert_eq!(compare_vectors(&[1.0, 2.0], &[1.0, 2.0]), std::cmp::Ordering::Equal);
        assert_eq!(compare_vectors(&[1.0, 2.0], &[1.0, 3.0]), std::cmp::Ordering::Less);
        assert_eq!(compare_vectors(&[2.0, 0.0], &[1.0, 9.0]), std::cmp::Ordering::Greater);
    }
}
