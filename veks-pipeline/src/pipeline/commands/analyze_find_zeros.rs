// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: brute-force scan for near-zero vectors.
//!
//! Scans every vector in a source file, computes the L2 norm in f64
//! precision, and reports all vectors below the zero threshold.
//! Multi-threaded with madvise(SEQUENTIAL) for maximum I/O throughput.

use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use crate::pipeline::element_type::ElementType;

fn error_result(message: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: message.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve_path(value: &str, workspace: &Path) -> std::path::PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

pub struct AnalyzeFindZerosOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeFindZerosOp)
}

impl CommandOp for AnalyzeFindZerosOp {
    fn command_path(&self) -> &str {
        "analyze find-zeros"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Brute-force scan for near-zero vectors".into(),
            body: format!(
                "# analyze find-zeros\n\n\
                Scan every vector in a source file and list all near-zero vectors.\n\n\
                ## Description\n\n\
                Reads every vector from the source file, computes the L2 norm in \
                f64 precision, and reports each vector whose norm falls below the \
                zero threshold. For each zero vector found, prints the ordinal \
                index, exact L2 norm, and a sample of leading components.\n\n\
                The scan is multi-threaded (rayon) with `madvise(SEQUENTIAL)` for \
                maximum I/O throughput on large files.\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "threads".into(), description: "Parallel scan".into(), adjustable: true },
        ]
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".into(),
                type_name: "Path".into(),
                required: false,
                default: None,
                description: "Source vector file or directory (defaults to '.' with --recursive)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "threshold".into(),
                type_name: "float".into(),
                required: false,
                default: Some("1e-06".into()),
                description: "L2-norm threshold for near-zero classification".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "limit".into(),
                type_name: "integer".into(),
                required: false,
                default: None,
                description: "Stop after finding this many zeros (default: report all)".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "components".into(),
                type_name: "integer".into(),
                required: false,
                default: Some("8".into()),
                description: "Number of leading components to display per zero vector".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "recursive".into(),
                type_name: "bool".into(),
                required: false,
                default: Some("false".into()),
                description: "Recursively scan all vector files under the source directory".into(),
                role: OptionRole::Config,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let recursive = options.get("recursive").map(|s| s == "true").unwrap_or(false);
        let source_str = options.get("source").unwrap_or(if recursive { "." } else { "" });
        if source_str.is_empty() {
            return error_result("'source' is required (or use --recursive to scan current directory)", start);
        }
        let source_path = resolve_path(source_str, &ctx.workspace);
        let threshold: f64 = options.get("threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-6);
        let threshold_sq = threshold * threshold;
        let limit: Option<usize> = options.get("limit").and_then(|s| s.parse().ok());
        let show_components: usize = options.get("components")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let recursive = options.get("recursive").map(|s| s == "true").unwrap_or(false);

        let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;

        // Recursive mode: scan directory for all vector files
        if recursive || source_path.is_dir() {
            let dir = if source_path.is_dir() { &source_path } else { source_path.parent().unwrap_or(Path::new(".")) };
            return scan_directory(dir, threshold, threshold_sq, limit, threads, ctx, start);
        }

        let etype = ElementType::from_path(&source_path)
            .unwrap_or(ElementType::F32);

        match etype {
            ElementType::F16 => {
                let reader = match MmapVectorReader::<half::f16>::open_mvec(&source_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
                };
                reader.advise_sequential();
                let count = VectorReader::<half::f16>::count(&reader);
                let dim = VectorReader::<half::f16>::dim(&reader);
                ctx.ui.log(&format!(
                    "find-zeros: scanning {} f16 vectors (dim={}, threshold={:.0e}, {} threads)",
                    count, dim, threshold, threads,
                ));
                let zeros = scan_zeros_f16(&reader, count, dim, threshold_sq, limit, threads, ctx);
                report_zeros_f16(&zeros, &reader, dim, show_components, threshold, count, start, ctx)
            }
            _ => {
                let reader = match MmapVectorReader::<f32>::open_fvec(&source_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
                };
                reader.advise_sequential();
                let count = VectorReader::<f32>::count(&reader);
                let dim = VectorReader::<f32>::dim(&reader);
                ctx.ui.log(&format!(
                    "find-zeros: scanning {} f32 vectors (dim={}, threshold={:.0e}, {} threads)",
                    count, dim, threshold, threads,
                ));
                let zeros = scan_zeros_f32(&reader, count, dim, threshold_sq, limit, threads, ctx);
                report_zeros_f32(&zeros, &reader, dim, show_components, threshold, count, start, ctx)
            }
        }
    }
}

/// A detected near-zero vector.
struct ZeroHit {
    ordinal: usize,
    norm: f64,
}

fn scan_zeros_f32(
    reader: &MmapVectorReader<f32>,
    count: usize,
    dim: usize,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
    ctx: &mut StreamContext,
) -> Vec<ZeroHit> {
    let pb = ctx.ui.bar_with_unit(count as u64, "scanning", "vectors");
    let progress = std::sync::atomic::AtomicU64::new(0);
    let found = std::sync::atomic::AtomicUsize::new(0);
    let limit_reached = std::sync::atomic::AtomicBool::new(false);

    let chunk_size = (count / (threads * 4)).max(10_000);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let compute = || {
        (0..count).into_par_iter()
            .chunks(chunk_size)
            .flat_map(|chunk| {
                let mut local_zeros = Vec::new();
                for i in chunk {
                    if limit_reached.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let slice = reader.get_slice(i);
                    let mut norm_sq = 0.0f64;
                    for d in 0..dim {
                        let v = slice[d] as f64;
                        norm_sq += v * v;
                    }
                    if norm_sq < threshold_sq {
                        let norm = norm_sq.sqrt();
                        local_zeros.push(ZeroHit { ordinal: i, norm });
                        let total = found.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if let Some(lim) = limit {
                            if total >= lim {
                                limit_reached.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                    let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if done % 100_000 == 0 { pb.set_position(done); }
                }
                local_zeros
            })
            .collect::<Vec<_>>()
    };

    let mut zeros = if let Some(ref p) = pool {
        p.install(compute)
    } else {
        compute()
    };
    pb.finish();

    zeros.sort_by_key(|z| z.ordinal);
    if let Some(lim) = limit {
        zeros.truncate(lim);
    }
    zeros
}

fn scan_zeros_f16(
    reader: &MmapVectorReader<half::f16>,
    count: usize,
    dim: usize,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
    ctx: &mut StreamContext,
) -> Vec<ZeroHit> {
    let pb = ctx.ui.bar_with_unit(count as u64, "scanning", "vectors");
    let progress = std::sync::atomic::AtomicU64::new(0);
    let found = std::sync::atomic::AtomicUsize::new(0);
    let limit_reached = std::sync::atomic::AtomicBool::new(false);

    let chunk_size = (count / (threads * 4)).max(10_000);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let compute = || {
        (0..count).into_par_iter()
            .chunks(chunk_size)
            .flat_map(|chunk| {
                let mut local_zeros = Vec::new();
                for i in chunk {
                    if limit_reached.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let slice = reader.get_slice(i);
                    let mut norm_sq = 0.0f64;
                    for d in 0..dim {
                        let v = slice[d].to_f64();
                        norm_sq += v * v;
                    }
                    if norm_sq < threshold_sq {
                        let norm = norm_sq.sqrt();
                        local_zeros.push(ZeroHit { ordinal: i, norm });
                        let total = found.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if let Some(lim) = limit {
                            if total >= lim {
                                limit_reached.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                    let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if done % 100_000 == 0 { pb.set_position(done); }
                }
                local_zeros
            })
            .collect::<Vec<_>>()
    };

    let mut zeros = if let Some(ref p) = pool {
        p.install(compute)
    } else {
        compute()
    };
    pb.finish();

    zeros.sort_by_key(|z| z.ordinal);
    if let Some(lim) = limit {
        zeros.truncate(lim);
    }
    zeros
}

fn report_zeros_f32(
    zeros: &[ZeroHit],
    reader: &MmapVectorReader<f32>,
    dim: usize,
    show_components: usize,
    threshold: f64,
    count: usize,
    start: Instant,
    ctx: &mut StreamContext,
) -> CommandResult {
    ctx.ui.log("");
    ctx.ui.log(&format!(
        "  {} near-zero vectors found (L2 < {:.0e}) out of {}",
        zeros.len(), threshold, count,
    ));
    ctx.ui.log("");

    if !zeros.is_empty() {
        let show = show_components.min(dim);
        ctx.ui.log(&format!(
            "  {:>10}  {:>14}  components[0..{}]",
            "ordinal", "L2 norm", show,
        ));
        ctx.ui.log(&format!(
            "  {:>10}  {:>14}  {}",
            "-------", "---------", "-------------------",
        ));

        for z in zeros {
            let slice = reader.get_slice(z.ordinal);
            let components: Vec<String> = slice[..show].iter()
                .map(|v| format!("{:.6}", v))
                .collect();
            let is_exact = slice.iter().all(|v| v.to_bits() == 0);
            ctx.ui.log(&format!(
                "  {:>10}  {:>14.8e}  [{}]{}",
                z.ordinal, z.norm,
                components.join(", "),
                if is_exact { "  (exact zero)" } else { "" },
            ));
        }
        ctx.ui.log("");

        // Summary: group by exact vs near-zero
        let exact_count = zeros.iter().filter(|z| {
            let slice = reader.get_slice(z.ordinal);
            slice.iter().all(|v| v.to_bits() == 0)
        }).count();
        let near_count = zeros.len() - exact_count;
        ctx.ui.log(&format!(
            "  summary: {} exact zero, {} near-zero (norm > 0 but < {:.0e})",
            exact_count, near_count, threshold,
        ));
    }

    CommandResult {
        status: Status::Ok,
        message: format!("{} near-zero vectors in {} (threshold {:.0e})",
            zeros.len(), count, threshold),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn report_zeros_f16(
    zeros: &[ZeroHit],
    reader: &MmapVectorReader<half::f16>,
    dim: usize,
    show_components: usize,
    threshold: f64,
    count: usize,
    start: Instant,
    ctx: &mut StreamContext,
) -> CommandResult {
    ctx.ui.log("");
    ctx.ui.log(&format!(
        "  {} near-zero vectors found (L2 < {:.0e}) out of {}",
        zeros.len(), threshold, count,
    ));
    ctx.ui.log("");

    if !zeros.is_empty() {
        let show = show_components.min(dim);
        ctx.ui.log(&format!(
            "  {:>10}  {:>14}  components[0..{}]",
            "ordinal", "L2 norm", show,
        ));
        ctx.ui.log(&format!(
            "  {:>10}  {:>14}  {}",
            "-------", "---------", "-------------------",
        ));

        for z in zeros {
            let slice = reader.get_slice(z.ordinal);
            let components: Vec<String> = slice[..show].iter()
                .map(|v| format!("{:.6}", v.to_f32()))
                .collect();
            let is_exact = slice.iter().all(|v| v.to_bits() == 0);
            ctx.ui.log(&format!(
                "  {:>10}  {:>14.8e}  [{}]{}",
                z.ordinal, z.norm,
                components.join(", "),
                if is_exact { "  (exact zero)" } else { "" },
            ));
        }
        ctx.ui.log("");

        let exact_count = zeros.iter().filter(|z| {
            let slice = reader.get_slice(z.ordinal);
            slice.iter().all(|v| v.to_bits() == 0)
        }).count();
        let near_count = zeros.len() - exact_count;
        ctx.ui.log(&format!(
            "  summary: {} exact zero, {} near-zero (norm > 0 but < {:.0e})",
            exact_count, near_count, threshold,
        ));
    }

    CommandResult {
        status: Status::Ok,
        message: format!("{} near-zero vectors in {} (threshold {:.0e})",
            zeros.len(), count, threshold),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Recursive directory scanning
// ═══════════════════════════════════════════════════════════════════════════

/// Supported vector file extensions for recursive scanning.
const VECTOR_EXTENSIONS: &[&str] = &["fvec", "fvecs", "mvec", "dvec"];

/// Recursively scan a directory for vector files and report zero counts.
fn scan_directory(
    dir: &Path,
    threshold: f64,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
    ctx: &mut StreamContext,
    start: Instant,
) -> CommandResult {
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    collect_vector_files(dir, &mut files);
    files.sort();

    if files.is_empty() {
        ctx.ui.log(&format!("  no vector files found under {}", dir.display()));
        return CommandResult {
            status: Status::Ok,
            message: "no vector files found".into(),
            produced: vec![],
            elapsed: start.elapsed(),
        };
    }

    ctx.ui.log(&format!(
        "find-zeros: scanning {} vector files under {} (threshold={:.0e}, {} threads)",
        files.len(), dir.display(), threshold, threads,
    ));
    ctx.ui.log("");
    ctx.ui.log(&format!(
        "  {:>10}  {:>10}  {:>10}  {:>6}  {:>8}  {:>12}  {}",
        "vectors", "zeros", "exact", "dim", "time", "vectors/s", "path",
    ));
    ctx.ui.log(&format!(
        "  {:>10}  {:>10}  {:>10}  {:>6}  {:>8}  {:>12}  {}",
        "-------", "-----", "-----", "---", "----", "---------", "----",
    ));

    let mut total_vectors: u64 = 0;
    let mut total_zeros: u64 = 0;
    let mut total_exact: u64 = 0;
    let mut files_with_zeros = 0usize;

    for file_path in &files {
        let rel = file_path.strip_prefix(dir).unwrap_or(file_path);
        let etype = ElementType::from_path(file_path).unwrap_or(ElementType::F32);

        let file_start = Instant::now();
        let result = match etype {
            ElementType::F16 => scan_file_concise_f16(file_path, threshold_sq, limit, threads),
            _ => scan_file_concise_f32(file_path, threshold_sq, limit, threads),
        };
        let file_elapsed = file_start.elapsed();

        match result {
            Ok((count, dim, zero_count, exact_count)) => {
                total_vectors += count as u64;
                total_zeros += zero_count as u64;
                total_exact += exact_count as u64;
                if zero_count > 0 { files_with_zeros += 1; }
                let secs = file_elapsed.as_secs_f64();
                let rate = if secs > 0.0 { count as f64 / secs } else { 0.0 };
                let time_str = if secs >= 60.0 {
                    format!("{:.0}m{:.0}s", secs / 60.0, secs % 60.0)
                } else {
                    format!("{:.1}s", secs)
                };
                let rate_str = if rate >= 1_000_000.0 {
                    format!("{:.1}M/s", rate / 1_000_000.0)
                } else if rate >= 1_000.0 {
                    format!("{:.1}K/s", rate / 1_000.0)
                } else {
                    format!("{:.0}/s", rate)
                };
                ctx.ui.log(&format!(
                    "  {:>10}  {:>10}  {:>10}  {:>6}  {:>8}  {:>12}  {}",
                    count, zero_count, exact_count, dim, time_str, rate_str, rel.display(),
                ));
            }
            Err(e) => {
                ctx.ui.log(&format!("  {:>10}  {:>10}  {:>10}  {:>6}  {:>8}  {:>12}  {} ({})",
                    "?", "?", "?", "?", "?", "?", rel.display(), e));
            }
        }
    }

    let total_secs = start.elapsed().as_secs_f64();
    let total_rate = if total_secs > 0.0 { total_vectors as f64 / total_secs } else { 0.0 };
    let total_rate_str = if total_rate >= 1_000_000.0 {
        format!("{:.1}M/s", total_rate / 1_000_000.0)
    } else if total_rate >= 1_000.0 {
        format!("{:.1}K/s", total_rate / 1_000.0)
    } else {
        format!("{:.0}/s", total_rate)
    };
    ctx.ui.log("");
    ctx.ui.log(&format!(
        "  total: {} vectors, {} zeros ({} exact) across {} files ({} with zeros), {:.1}s ({})",
        total_vectors, total_zeros, total_exact, files.len(), files_with_zeros,
        total_secs, total_rate_str,
    ));

    CommandResult {
        status: Status::Ok,
        message: format!(
            "{} zeros in {} vectors across {} files",
            total_zeros, total_vectors, files.len(),
        ),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

/// Collect all vector files recursively, skipping hidden dirs and caches.
fn collect_vector_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if name_str.starts_with('.') || name_str == "target" || name_str == "node_modules" {
                continue;
            }
            collect_vector_files(&path, files);
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if VECTOR_EXTENSIONS.contains(&ext) {
                files.push(path);
            }
        }
    }
}

/// Scan a single f32 file, return (count, dim, zero_count, exact_zero_count).
fn scan_file_concise_f32(
    path: &Path,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let reader = MmapVectorReader::<f32>::open_fvec(path)
        .map_err(|e| format!("{}", e))?;
    reader.advise_sequential();
    let count = VectorReader::<f32>::count(&reader);
    let dim = VectorReader::<f32>::dim(&reader);
    let chunk_size = (count / (threads * 4)).max(10_000);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let compute = || {
        (0..count).into_par_iter()
            .chunks(chunk_size)
            .map(|chunk| {
                let mut zeros = 0usize;
                let mut exact = 0usize;
                for i in chunk {
                    let slice = reader.get_slice(i);
                    let mut norm_sq = 0.0f64;
                    for d in 0..dim {
                        let v = slice[d] as f64;
                        norm_sq += v * v;
                    }
                    if norm_sq < threshold_sq {
                        zeros += 1;
                        if slice.iter().all(|v| v.to_bits() == 0) {
                            exact += 1;
                        }
                        if let Some(lim) = limit {
                            if zeros >= lim { break; }
                        }
                    }
                }
                (zeros, exact)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    };

    let (zc, ec) = if let Some(ref p) = pool { p.install(compute) } else { compute() };
    Ok((count, dim, zc, ec))
}

/// Scan a single f16 file, return (count, dim, zero_count, exact_zero_count).
fn scan_file_concise_f16(
    path: &Path,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let reader = MmapVectorReader::<half::f16>::open_mvec(path)
        .map_err(|e| format!("{}", e))?;
    reader.advise_sequential();
    let count = VectorReader::<half::f16>::count(&reader);
    let dim = VectorReader::<half::f16>::dim(&reader);
    let chunk_size = (count / (threads * 4)).max(10_000);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let compute = || {
        (0..count).into_par_iter()
            .chunks(chunk_size)
            .map(|chunk| {
                let mut zeros = 0usize;
                let mut exact = 0usize;
                for i in chunk {
                    let slice = reader.get_slice(i);
                    let mut norm_sq = 0.0f64;
                    for d in 0..dim {
                        let v = slice[d].to_f64();
                        norm_sq += v * v;
                    }
                    if norm_sq < threshold_sq {
                        zeros += 1;
                        if slice.iter().all(|v| v.to_bits() == 0) {
                            exact += 1;
                        }
                        if let Some(lim) = limit {
                            if zeros >= lim { break; }
                        }
                    }
                }
                (zeros, exact)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    };

    let (zc, ec) = if let Some(ref p) = pool { p.install(compute) } else { compute() };
    Ok((count, dim, zc, ec))
}
