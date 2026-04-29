// Copyright (c) Jonathan Shook
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
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::element_type::ElementType;

/// Pipeline command: find a specific vector in a target file.
pub struct AnalyzeFindOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeFindOp)
}

impl CommandOp for AnalyzeFindOp {
    fn command_path(&self) -> &str {
        "analyze find"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Search for vectors matching a value pattern".into(),
            body: format!(
                "# analyze find\n\n\
                Search for vectors matching a value pattern.\n\n\
                ## Description\n\n\
                Reads a vector at a given index from a source file, then searches for \
                an exact or close match in a target file. The command automatically \
                detects whether the target file is lexicographically sorted by sampling \
                the first and last 100 adjacent pairs. If sorted, it uses binary search \
                for O(log n) lookup; otherwise, it falls back to an exhaustive linear \
                scan.\n\n\
                ## Search Patterns\n\n\
                The current search pattern is \"find by example\": you specify a source \
                file and an index within it, and the command extracts that vector as the \
                search needle. The target file is then searched for:\n\n\
                - **Exact match**: All components are bitwise identical.\n\
                - **Closest match**: If no exact match exists, the vector with the most \
                component-wise matches (within float epsilon) is reported as an \
                approximate result.\n\n\
                Lexicographic comparison is used for binary search, with NaN values \
                sorted to the end.\n\n\
                ## Role in Dataset Preparation\n\n\
                This command is useful for verifying that a specific vector from one \
                file (e.g., the original dataset) appears in another file (e.g., a \
                shuffled or filtered version). It can also diagnose duplication issues \
                or confirm that a particular record survived a filtering step.\n\n\
                ## Options\n\n{}",
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

        // Open source file with element-type dispatch
        let src_etype = match ElementType::from_path(&source_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };
        let (source_count, _src_dim, src_get): (usize, usize, Box<dyn Fn(usize) -> Vec<f64> + Sync>) = match src_etype {
            ElementType::F32 => {
                let r = match MmapVectorReader::<f32>::open_fvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<f32>::count(&r);
                let d = VectorReader::<f32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::F16 => {
                let r = match MmapVectorReader::<half::f16>::open_mvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<half::f16>::count(&r);
                let d = VectorReader::<half::f16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|v| v.to_f64()).collect()))
            }
            ElementType::F64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default()))
            }
            ElementType::I32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                let d = VectorReader::<i32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::I16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U8 | ElementType::I8 => {
                let r = match MmapVectorReader::<u8>::open_bvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<u8>::count(&r);
                let d = VectorReader::<u8>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                let d = VectorReader::<i32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U64 | ElementType::I64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&source_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open source: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
        };

        if index >= source_count {
            return error_result(
                format!(
                    "index {} out of range (source has {} vectors)",
                    index, source_count
                ),
                start,
            );
        }

        let needle = src_get(index);
        let dim = needle.len();

        ctx.ui.log(&format!(
            "Searching for source[{}] (dim={}) in {}",
            index,
            dim,
            target_path.display()
        ));

        // Open target file with element-type dispatch
        let tgt_etype = match ElementType::from_path(&target_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };
        let (target_count, target_dim, tgt_get): (usize, usize, Box<dyn Fn(usize) -> Vec<f64> + Sync>) = match tgt_etype {
            ElementType::F32 => {
                let r = match MmapVectorReader::<f32>::open_fvec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<f32>::count(&r);
                let d = VectorReader::<f32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::F16 => {
                let r = match MmapVectorReader::<half::f16>::open_mvec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<half::f16>::count(&r);
                let d = VectorReader::<half::f16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|v| v.to_f64()).collect()))
            }
            ElementType::F64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default()))
            }
            ElementType::I32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                let d = VectorReader::<i32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::I16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U8 | ElementType::I8 => {
                let r = match MmapVectorReader::<u8>::open_bvec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<u8>::count(&r);
                let d = VectorReader::<u8>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U16 => {
                let r = match MmapVectorReader::<i16>::open_svec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<i16>::count(&r);
                let d = VectorReader::<i16>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U32 => {
                let r = match MmapVectorReader::<i32>::open_ivec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<i32>::count(&r);
                let d = VectorReader::<i32>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
            ElementType::U64 | ElementType::I64 => {
                let r = match MmapVectorReader::<f64>::open_dvec(&target_path) {
                    Ok(r) => r, Err(e) => return error_result(format!("open target: {}", e), start),
                };
                let fc = VectorReader::<f64>::count(&r);
                let d = VectorReader::<f64>::dim(&r);
                (fc, d, Box::new(move |i| r.get(i).unwrap_or_default().iter().map(|&v| v as f64).collect()))
            }
        };

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
        let is_sorted = check_sorted(&*tgt_get, target_count);

        if is_sorted {
            ctx.ui.log("Target appears sorted — using binary search");
            match binary_search(&*tgt_get, &needle, target_count) {
                Some(found_idx) => {
                    ctx.ui.log(&format!("FOUND at index {}", found_idx));
                    CommandResult {
                        status: Status::Ok,
                        message: format!("found at index {}", found_idx),
                        produced: vec![],
                        elapsed: start.elapsed(),
                    }
                }
                None => {
                    ctx.ui.log("NOT FOUND (binary search)");
                    CommandResult {
                        status: Status::Warning,
                        message: "not found".to_string(),
                        produced: vec![],
                        elapsed: start.elapsed(),
                    }
                }
            }
        } else {
            ctx.ui.log(&format!(
                "Target is unsorted — exhaustive scan ({} vectors)",
                target_count
            ));
            let result = exhaustive_scan(&*tgt_get, &needle, target_count);
            match result {
                Some((found_idx, exact)) => {
                    if exact {
                        ctx.ui.log(&format!("FOUND exact match at index {}", found_idx));
                        CommandResult {
                            status: Status::Ok,
                            message: format!("found exact match at index {}", found_idx),
                            produced: vec![],
                            elapsed: start.elapsed(),
                        }
                    } else {
                        ctx.ui.log(&format!("Closest match at index {} (not exact)", found_idx));
                        CommandResult {
                            status: Status::Warning,
                            message: format!("closest match at index {} (not exact)", found_idx),
                            produced: vec![],
                            elapsed: start.elapsed(),
                        }
                    }
                }
                None => {
                    ctx.ui.log("NOT FOUND");
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
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "target".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Target file to search in".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "index".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "0-based index of vector in source file".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Check if a file appears lexicographically sorted by sampling.
fn check_sorted(get_f64: &dyn Fn(usize) -> Vec<f64>, count: usize) -> bool {
    if count <= 1 {
        return true;
    }
    let sample = 100.min(count - 1);
    for i in 0..sample {
        let a = get_f64(i);
        let b = get_f64(i + 1);
        if compare_vectors(&a, &b) == std::cmp::Ordering::Greater {
            return false;
        }
    }
    // Also check last few
    if count > sample + 2 {
        for i in (count - sample)..count - 1 {
            let a = get_f64(i);
            let b = get_f64(i + 1);
            if compare_vectors(&a, &b) == std::cmp::Ordering::Greater {
                return false;
            }
        }
    }
    true
}

/// Lexicographic comparison of two float vectors.
fn compare_vectors(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
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
    get_f64: &dyn Fn(usize) -> Vec<f64>,
    needle: &[f64],
    count: usize,
) -> Option<usize> {
    let mut low = 0usize;
    let mut high = count;
    while low < high {
        let mid = low + (high - low) / 2;
        let vec = get_f64(mid);
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
    get_f64: &dyn Fn(usize) -> Vec<f64>,
    needle: &[f64],
    count: usize,
) -> Option<(usize, bool)> {
    let mut best_idx = 0;
    let mut best_matching = 0usize;

    for i in 0..count {
        let vec = get_f64(i);
        if vec.is_empty() {
            continue;
        }
        if vec.as_slice() == needle {
            return Some((i, true));
        }
        let matching = vec
            .iter()
            .zip(needle.iter())
            .filter(|(a, b)| (*a - *b).abs() < f64::EPSILON)
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
        assert_eq!(compare_vectors(&[1.0f64, 2.0], &[1.0, 2.0]), std::cmp::Ordering::Equal);
        assert_eq!(compare_vectors(&[1.0f64, 2.0], &[1.0, 3.0]), std::cmp::Ordering::Less);
        assert_eq!(compare_vectors(&[2.0f64, 0.0], &[1.0, 9.0]), std::cmp::Ordering::Greater);
    }
}
