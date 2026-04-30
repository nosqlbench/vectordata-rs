// Copyright (c) Jonathan Shook
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
use vectordata::io::XvecReader;

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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

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
                // Mmap reader used only for metadata + the per-hit
                // component display in the report phase. Full scan
                // uses pread streaming to keep RSS bounded.
                let reader = match XvecReader::<half::f16>::open_path(&source_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
                };
                let count = VectorReader::<half::f16>::count(&reader);
                let dim = VectorReader::<half::f16>::dim(&reader);
                ctx.ui.log(&format!(
                    "find-zeros: scanning {} f16 vectors (dim={}, threshold={:.0e}, {} threads)",
                    count, dim, threshold, threads,
                ));
                let zeros = scan_zeros_f16(&source_path, count, dim, threshold_sq, limit, threads, ctx);
                report_zeros_f16(&zeros, &reader, dim, show_components, threshold, count, start, ctx)
            }
            _ => {
                // Open the mmap reader only for metadata (count, dim)
                // and for the per-hit component display in the report
                // phase — which touches ≤ `limit` vectors. The actual
                // full scan uses pread streaming so RSS stays bounded
                // regardless of source file size.
                let reader = match XvecReader::<f32>::open_path(&source_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(format!("open {}: {}", source_path.display(), e), start),
                };
                let count = VectorReader::<f32>::count(&reader);
                let dim = VectorReader::<f32>::dim(&reader);
                ctx.ui.log(&format!(
                    "find-zeros: scanning {} f32 vectors (dim={}, threshold={:.0e}, {} threads)",
                    count, dim, threshold, threads,
                ));
                let zeros = scan_zeros_f32(&source_path, count, dim, threshold_sq, limit, threads, ctx);
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

/// Target chunk size for streaming pread. 64 MiB balances syscall
/// overhead against keeping resident memory small and bounded.
const SCAN_CHUNK_BYTES: usize = 64 * 1024 * 1024;

fn scan_zeros_f32(
    path: &std::path::Path,
    count: usize,
    dim: usize,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
    ctx: &mut StreamContext,
) -> Vec<ZeroHit> {
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    let pb = ctx.ui.bar_with_unit(count as u64, "scanning", "vectors");

    if count == 0 || dim == 0 {
        pb.finish();
        return Vec::new();
    }

    // Workers each own a contiguous slab of chunks and do their own
    // pread → scan → pread → scan loop. No central reader, no rayon
    // fork-join between chunks, no idle-disk-while-we-scan bubble.
    // On NVMe, concurrent prefix prereads from disjoint file offsets
    // happily saturate the device; the old shape (one pread, then
    // parallel scan) left ~50% of wall time with the disk idle.
    //
    // Per-chunk size is kept modest (4 MiB) instead of the old 64 MiB
    // so threads don't all pin on the same few giant reads — smaller
    // chunks give the kernel scheduler and the NVMe queue finer-grained
    // work to interleave. Aggregate in-flight memory stays small
    // (`threads * chunk_bytes`).
    const PER_WORKER_CHUNK_BYTES: usize = 4 * 1024 * 1024;
    let entry_size = 4 + dim * 4;
    let vecs_per_chunk = (PER_WORKER_CHUNK_BYTES / entry_size).max(1);

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            pb.finish();
            log::error!("open {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    // Kernel hint: we'll read once, sequentially, never come back.
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        }
    }

    let found = AtomicUsize::new(0);
    let limit_reached = AtomicBool::new(false);
    let progress = AtomicUsize::new(0);
    let all_zeros: Mutex<Vec<ZeroHit>> = Mutex::new(Vec::new());

    let threads = threads.max(1).min((count + vecs_per_chunk - 1) / vecs_per_chunk);
    let file_ref = &file;
    let found_ref = &found;
    let limit_reached_ref = &limit_reached;
    let progress_ref = &progress;
    let all_zeros_ref = &all_zeros;
    let pb_ref = &pb;

    std::thread::scope(|scope| {
        // Contiguous per-worker stride: worker `w` owns vector indices
        // `[w * stride_vecs, (w+1) * stride_vecs)` (last worker gets
        // the remainder). Contiguous > round-robin here because
        // sequential prefetch inside each worker's range is worth
        // more than even load balance at the chunk level — scan work
        // per vector is roughly constant.
        let stride_vecs = (count + threads - 1) / threads;
        for w in 0..threads {
            let start_vec = w * stride_vecs;
            if start_vec >= count { break; }
            let end_vec = ((w + 1) * stride_vecs).min(count);

            scope.spawn(move || {
                // One reusable per-worker buffer. `Vec<f32>` for alignment;
                // the pread path reinterprets it as bytes.
                let chunk_cap_bytes = vecs_per_chunk * entry_size;
                let mut buf: Vec<f32> = vec![0.0; (chunk_cap_bytes + 3) / 4];

                let mut v = start_vec;
                while v < end_vec {
                    if limit_reached_ref.load(Ordering::Relaxed) { break; }

                    let n_vecs = vecs_per_chunk.min(end_vec - v);
                    let chunk_bytes = n_vecs * entry_size;
                    let byte_off = (v as u64) * (entry_size as u64);

                    let bytes_mut: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            buf.as_mut_ptr() as *mut u8,
                            buf.len() * 4,
                        )
                    };
                    if let Err(e) = file_ref.read_exact_at(&mut bytes_mut[..chunk_bytes], byte_off) {
                        log::error!("pread at vec {}: {}", v, e);
                        return;
                    }

                    let mut local_hits: Vec<ZeroHit> = Vec::new();
                    for local_i in 0..n_vecs {
                        // Each entry = 4-byte dim header + dim*4 bytes.
                        // In f32-units: header at `local_i*(1+dim)`, payload
                        // starts at `local_i*(1+dim)+1`.
                        let payload_off = local_i * (1 + dim) + 1;
                        let slice: &[f32] = &buf[payload_off..payload_off + dim];
                        // Accumulate in f64 to match the previous impl's
                        // threshold semantics; the extra precision
                        // matters at the near-zero boundary.
                        let mut norm_sq = 0.0f64;
                        for &x in slice {
                            let xf = x as f64;
                            norm_sq += xf * xf;
                        }
                        if norm_sq < threshold_sq {
                            let norm = norm_sq.sqrt();
                            let total = found_ref.fetch_add(1, Ordering::Relaxed) + 1;
                            if let Some(lim) = limit {
                                if total >= lim {
                                    limit_reached_ref.store(true, Ordering::Relaxed);
                                }
                            }
                            local_hits.push(ZeroHit { ordinal: v + local_i, norm });
                        }
                    }

                    // Drop the pages we just finished with — a one-shot
                    // scan never needs them again, and keeping 1+ TB
                    // of source resident only evicts hot pages.
                    #[cfg(target_os = "linux")]
                    {
                        use std::os::unix::io::AsRawFd;
                        unsafe {
                            libc::posix_fadvise(
                                file_ref.as_raw_fd(),
                                byte_off as libc::off_t,
                                chunk_bytes as libc::off_t,
                                libc::POSIX_FADV_DONTNEED,
                            );
                        }
                    }

                    if !local_hits.is_empty() {
                        all_zeros_ref.lock().unwrap().extend(local_hits);
                    }

                    let done = progress_ref.fetch_add(n_vecs, Ordering::Relaxed) + n_vecs;
                    pb_ref.set_position(done as u64);

                    v += n_vecs;
                }
            });
        }
    });
    pb.finish();

    let mut zeros = all_zeros.into_inner().unwrap_or_default();
    zeros.sort_by_key(|z| z.ordinal);
    if let Some(lim) = limit {
        zeros.truncate(lim);
    }
    zeros
}

fn scan_zeros_f16(
    path: &std::path::Path,
    count: usize,
    dim: usize,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
    ctx: &mut StreamContext,
) -> Vec<ZeroHit> {
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    let pb = ctx.ui.bar_with_unit(count as u64, "scanning", "vectors");
    let found = std::sync::atomic::AtomicUsize::new(0);
    let limit_reached = std::sync::atomic::AtomicBool::new(false);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    // Guard against degenerate inputs — see scan_zeros_f32 for rationale.
    if count == 0 || dim == 0 {
        pb.finish();
        return Vec::new();
    }
    // mvec entry: 4-byte dim header + dim × f16 (2 bytes each).
    // Backing buffer as `Vec<u16>` so it's 2-byte aligned — the
    // header is 4 bytes (2 u16s), then `dim` u16 payload values
    // reinterpretable as `half::f16` via `#[repr(transparent)]`.
    let entry_size = 4 + dim * 2;
    let vecs_per_chunk = (SCAN_CHUNK_BYTES / entry_size).max(1);
    let chunk_capacity_bytes = vecs_per_chunk * entry_size;
    let chunk_capacity_u16 = (chunk_capacity_bytes + 1) / 2;

    let mut chunk_buf: Vec<u16> = vec![0u16; chunk_capacity_u16];

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            pb.finish();
            log::error!("open {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    let mut all_zeros: Vec<ZeroHit> = Vec::new();
    let mut offset_vec: usize = 0;

    while offset_vec < count {
        if limit_reached.load(std::sync::atomic::Ordering::Relaxed) { break; }

        let n_vecs = vecs_per_chunk.min(count - offset_vec);
        let chunk_bytes_now = n_vecs * entry_size;
        let byte_off = (offset_vec as u64) * (entry_size as u64);

        let bytes_mut: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                chunk_buf.as_mut_ptr() as *mut u8,
                chunk_buf.len() * 2,
            )
        };
        if let Err(e) = file.read_exact_at(&mut bytes_mut[..chunk_bytes_now], byte_off) {
            log::error!("pread at vec {}: {}", offset_vec, e);
            break;
        }

        let base_vec_idx = offset_vec;
        let chunk_view: &[u16] = &chunk_buf;
        // Each entry in the u16 chunk view: 2 u16s of header, then
        // `dim` u16 payload. Header u16-offset per vector: local_i * (2 + dim).
        let compute = || {
            (0..n_vecs).into_par_iter()
                .filter_map(|local_i| {
                    if limit_reached.load(std::sync::atomic::Ordering::Relaxed) {
                        return None;
                    }
                    let payload_off = local_i * (2 + dim) + 2;
                    let payload_u16 = &chunk_view[payload_off..payload_off + dim];
                    // Reinterpret u16 → half::f16 (#[repr(transparent)]).
                    let slice: &[half::f16] = unsafe {
                        std::slice::from_raw_parts(
                            payload_u16.as_ptr() as *const half::f16,
                            dim,
                        )
                    };
                    let mut norm_sq = 0.0f64;
                    for v in slice {
                        let vf = v.to_f64();
                        norm_sq += vf * vf;
                    }
                    if norm_sq < threshold_sq {
                        let norm = norm_sq.sqrt();
                        let total = found.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if let Some(lim) = limit {
                            if total >= lim {
                                limit_reached.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        Some(ZeroHit { ordinal: base_vec_idx + local_i, norm })
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };

        let chunk_zeros = if let Some(ref p) = pool {
            p.install(compute)
        } else {
            compute()
        };
        all_zeros.extend(chunk_zeros);

        offset_vec += n_vecs;
        pb.set_position(offset_vec as u64);
    }
    pb.finish();
    let mut zeros = all_zeros;

    zeros.sort_by_key(|z| z.ordinal);
    if let Some(lim) = limit {
        zeros.truncate(lim);
    }
    zeros
}

fn report_zeros_f32(
    zeros: &[ZeroHit],
    reader: &XvecReader<f32>,
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

    // Set pipeline variables
    let zero_count = zeros.len();
    let _ = crate::pipeline::variables::set_and_save(
        &ctx.workspace, "zero_count", &zero_count.to_string());
    ctx.defaults.insert("zero_count".into(), zero_count.to_string());
    let _ = crate::pipeline::variables::set_and_save(
        &ctx.workspace, "source_zero_count", &zero_count.to_string());
    ctx.defaults.insert("source_zero_count".into(), zero_count.to_string());
    // Mirror the count into `dataset.yaml` as the boolean attribute the
    // publish-readiness check requires. Running `analyze find-zeros`
    // standalone now satisfies `veks check dataset-attributes` without
    // a full pipeline replay; the attribute write does not invalidate
    // any compute-step fingerprints.
    let _ = crate::pipeline::variables::set_dataset_attribute(
        &ctx.workspace, "is_zero_vector_free",
        if zero_count == 0 { "true" } else { "false" });

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
    reader: &XvecReader<half::f16>,
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

    // Set pipeline variables
    let zc = zeros.len();
    let _ = crate::pipeline::variables::set_and_save(
        &ctx.workspace, "zero_count", &zc.to_string());
    ctx.defaults.insert("zero_count".into(), zc.to_string());
    let _ = crate::pipeline::variables::set_and_save(
        &ctx.workspace, "source_zero_count", &zc.to_string());
    ctx.defaults.insert("source_zero_count".into(), zc.to_string());
    let _ = crate::pipeline::variables::set_dataset_attribute(
        &ctx.workspace, "is_zero_vector_free",
        if zc == 0 { "true" } else { "false" });

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

    // Set pipeline variables so dataset attributes can be populated
    let _ = crate::pipeline::variables::set_and_save(
        &ctx.workspace, "zero_count", &total_zeros.to_string());
    ctx.defaults.insert("zero_count".into(), total_zeros.to_string());
    let _ = crate::pipeline::variables::set_and_save(
        &ctx.workspace, "source_zero_count", &total_zeros.to_string());
    ctx.defaults.insert("source_zero_count".into(), total_zeros.to_string());
    let _ = crate::pipeline::variables::set_dataset_attribute(
        &ctx.workspace, "is_zero_vector_free",
        if total_zeros == 0 { "true" } else { "false" });

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
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    // Metadata via mmap (touches only the dim header). Full scan via
    // pread streaming so RSS stays bounded on terabyte-class sources.
    let reader = XvecReader::<f32>::open_path(path)
        .map_err(|e| format!("{}", e))?;
    let count = VectorReader::<f32>::count(&reader);
    let dim = VectorReader::<f32>::dim(&reader);
    drop(reader);

    if count == 0 || dim == 0 {
        return Ok((count, dim, 0, 0));
    }
    let entry_size = 4 + dim * 4;
    let vecs_per_chunk = (SCAN_CHUNK_BYTES / entry_size).max(1);
    let chunk_capacity_f32 = (vecs_per_chunk * entry_size + 3) / 4;
    let mut chunk_buf: Vec<f32> = vec![0.0; chunk_capacity_f32];

    let file = File::open(path).map_err(|e| format!("{}", e))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let limit_atomic = std::sync::atomic::AtomicUsize::new(0);
    let mut total_zeros = 0usize;
    let mut total_exact = 0usize;
    let mut offset_vec = 0usize;

    while offset_vec < count {
        if let Some(lim) = limit {
            if total_zeros >= lim { break; }
        }
        let n_vecs = vecs_per_chunk.min(count - offset_vec);
        let chunk_bytes_now = n_vecs * entry_size;
        let byte_off = (offset_vec as u64) * (entry_size as u64);

        let bytes_mut: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                chunk_buf.as_mut_ptr() as *mut u8,
                chunk_buf.len() * 4,
            )
        };
        file.read_exact_at(&mut bytes_mut[..chunk_bytes_now], byte_off)
            .map_err(|e| format!("pread at vec {}: {}", offset_vec, e))?;

        let chunk_view: &[f32] = &chunk_buf;
        let compute = || {
            (0..n_vecs).into_par_iter()
                .map(|local_i| {
                    let payload_off = local_i * (1 + dim) + 1;
                    let slice = &chunk_view[payload_off..payload_off + dim];
                    let mut norm_sq = 0.0f64;
                    for &v in slice {
                        let vf = v as f64;
                        norm_sq += vf * vf;
                    }
                    if norm_sq < threshold_sq {
                        let exact = slice.iter().all(|v| v.to_bits() == 0);
                        (1usize, if exact { 1usize } else { 0usize })
                    } else {
                        (0, 0)
                    }
                })
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        };

        let (zc, ec) = if let Some(ref p) = pool { p.install(compute) } else { compute() };
        total_zeros += zc;
        total_exact += ec;
        let _ = limit_atomic.fetch_add(zc, std::sync::atomic::Ordering::Relaxed);
        offset_vec += n_vecs;
    }

    Ok((count, dim, total_zeros, total_exact))
}

/// Scan a single f16 file, return (count, dim, zero_count, exact_zero_count).
fn scan_file_concise_f16(
    path: &Path,
    threshold_sq: f64,
    limit: Option<usize>,
    threads: usize,
) -> Result<(usize, usize, usize, usize), String> {
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    let reader = XvecReader::<half::f16>::open_path(path)
        .map_err(|e| format!("{}", e))?;
    let count = VectorReader::<half::f16>::count(&reader);
    let dim = VectorReader::<half::f16>::dim(&reader);
    drop(reader);

    if count == 0 || dim == 0 {
        return Ok((count, dim, 0, 0));
    }
    let entry_size = 4 + dim * 2;
    let vecs_per_chunk = (SCAN_CHUNK_BYTES / entry_size).max(1);
    let chunk_capacity_u16 = (vecs_per_chunk * entry_size + 1) / 2;
    let mut chunk_buf: Vec<u16> = vec![0u16; chunk_capacity_u16];

    let file = File::open(path).map_err(|e| format!("{}", e))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok();

    let mut total_zeros = 0usize;
    let mut total_exact = 0usize;
    let mut offset_vec = 0usize;

    while offset_vec < count {
        if let Some(lim) = limit {
            if total_zeros >= lim { break; }
        }
        let n_vecs = vecs_per_chunk.min(count - offset_vec);
        let chunk_bytes_now = n_vecs * entry_size;
        let byte_off = (offset_vec as u64) * (entry_size as u64);

        let bytes_mut: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                chunk_buf.as_mut_ptr() as *mut u8,
                chunk_buf.len() * 2,
            )
        };
        file.read_exact_at(&mut bytes_mut[..chunk_bytes_now], byte_off)
            .map_err(|e| format!("pread at vec {}: {}", offset_vec, e))?;

        let chunk_view: &[u16] = &chunk_buf;
        let compute = || {
            (0..n_vecs).into_par_iter()
                .map(|local_i| {
                    let payload_off = local_i * (2 + dim) + 2;
                    let payload_u16 = &chunk_view[payload_off..payload_off + dim];
                    let slice: &[half::f16] = unsafe {
                        std::slice::from_raw_parts(
                            payload_u16.as_ptr() as *const half::f16,
                            dim,
                        )
                    };
                    let mut norm_sq = 0.0f64;
                    for v in slice {
                        let vf = v.to_f64();
                        norm_sq += vf * vf;
                    }
                    if norm_sq < threshold_sq {
                        let exact = slice.iter().all(|v| v.to_bits() == 0);
                        (1usize, if exact { 1usize } else { 0usize })
                    } else {
                        (0, 0)
                    }
                })
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        };

        let (zc, ec) = if let Some(ref p) = pool { p.install(compute) } else { compute() };
        total_zeros += zc;
        total_exact += ec;
        offset_vec += n_vecs;
    }

    Ok((count, dim, total_zeros, total_exact))
}
