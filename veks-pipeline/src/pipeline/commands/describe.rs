// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline wrapper for the analyze describe operation.
//!
//! Runs `analyze describe` on a source file and captures the output as a
//! command result. This is primarily useful for validation steps in pipelines.

use std::path::{Path, PathBuf};
use std::time::Instant;

use veks_core::formats::VecFormat;
use veks_core::formats::reader;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: describe/analyze a vector file.
pub struct AnalyzeDescribeOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeDescribeOp)
}

impl CommandOp for AnalyzeDescribeOp {
    fn command_path(&self) -> &str {
        "analyze describe"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Describe vector file format and dimensions".into(),
            body: format!(
                r#"# analyze describe

Describe vector file format and dimensions.

## Description

The analyze describe command inspects a vector file and reports key metadata
about its structure. It probes the file without reading all records, making it
fast even for very large files.

The reported fields include:

- **format** -- the detected or overridden vector format (fvec, ivec, bvec,
  npy, slab, xvec, etc.)
- **dimension** -- the number of elements per vector record
- **element size** -- the byte width of each element (e.g. 4 for f32, 2 for
  f16, 1 for u8)
- **record count** -- the total number of vector records in the file, when
  determinable from the file size and format header
- **file size** -- derived from the underlying file metadata

When the `format` option is omitted, the command auto-detects the format from
the file extension or header bytes. You can override this with an explicit
format string if the file has a non-standard extension.

## Data Preparation Role

Describe serves as a debugging and validation tool within dataset preparation
pipelines. After an import or extract step produces an output file, a describe
step can confirm that the file has the expected format, dimension, element size,
and record count. This catches problems early -- such as a dimension mismatch or
a truncated file -- before downstream steps consume the data.

It is also useful outside of pipelines for quick inspection of any vector file
on disk, answering questions like "how many vectors are in this file?" or "what
element type does this dataset use?"

## Notes

- The probe reads only the file header and metadata; it does not scan all
  records, so it completes in constant time regardless of file size.
- For slab files, the record count is derived from the slab page index rather
  than the raw file size.
- If the file is corrupt or truncated, the probe will report an error rather
  than returning partial metadata.

## Options

{}"#,
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

        let source_path = resolve_path(source_str, &ctx.workspace);

        let format = if let Some(fmt_str) = options.get("format") {
            match VecFormat::from_extension(fmt_str) {
                Some(f) => f,
                None => return error_result(format!("unknown format: '{}'", fmt_str), start),
            }
        } else {
            match VecFormat::detect(&source_path) {
                Some(f) => f,
                None => {
                    return error_result(
                        format!(
                            "cannot detect format for '{}', set 'format' option",
                            source_path.display()
                        ),
                        start,
                    )
                }
            }
        };

        let full_scan = options.get("full-scan").map(|s| s != "false").unwrap_or(false);
        let sparse_scan = options.get("sparse-scan").map(|s| s != "false").unwrap_or(false);
        if sparse_scan && full_scan {
            return error_result(
                "--sparse-scan and --full-scan are mutually exclusive".into(),
                start,
            );
        }

        // Three paths from here on:
        //   - directory + sparse:  walk + stat all; probe FIRST file only;
        //                          estimate cardinality from amortized
        //                          per-record bytes (size_first / rows_first).
        //   - directory + default: walk + stat all; probe whole dir for an
        //                          exact record count from per-file metadata.
        //   - file:                probe just the file.
        let (meta, dir_agg, estimate) = if source_path.is_dir() {
            let agg = match aggregate_directory(&source_path, format, ctx) {
                Ok(a) => a,
                Err(e) => return error_result(format!("scan directory: {}", e), start),
            };
            if sparse_scan {
                let m = match reader::probe_source(&agg.first_file, format) {
                    Ok(m) => m,
                    Err(e) => return error_result(
                        format!("probe first file: {}", e),
                        start,
                    ),
                };
                let est = m.record_count.and_then(|first_rows| {
                    if first_rows == 0 || agg.first_file_bytes == 0 {
                        return None;
                    }
                    let bpr = agg.first_file_bytes as f64 / first_rows as f64;
                    if bpr <= 0.0 { return None; }
                    Some(EstimatedRecords {
                        rows: (agg.total_bytes as f64 / bpr).round() as u64,
                        amortized_bpr: bpr,
                        first_file_rows: first_rows,
                        first_file_bytes: agg.first_file_bytes,
                    })
                });
                (m, Some(agg), est)
            } else {
                let m = match reader::probe_source(&source_path, format) {
                    Ok(m) => m,
                    Err(e) => return error_result(
                        format!("failed to probe source: {}", e),
                        start,
                    ),
                };
                (m, Some(agg), None)
            }
        } else {
            let m = match reader::probe_source(&source_path, format) {
                Ok(m) => m,
                Err(e) => return error_result(
                    format!("failed to probe source: {}", e),
                    start,
                ),
            };
            (m, None, None)
        };

        let file_size = match &dir_agg {
            Some(a) => Some(a.total_bytes),
            None => std::fs::metadata(&source_path).map(|m| m.len()).ok(),
        };
        let records_str = if let Some(est) = &estimate {
            format!("~{} (estimated)", format_count(est.rows))
        } else {
            meta.record_count
                .map_or("unknown".to_string(), |n| format_count(n))
        };
        let record_bytes = 4 + meta.dimension as usize * meta.element_size;

        let display_path = source_path
            .canonicalize()
            .unwrap_or_else(|_| source_path.clone());
        let path_label = if dir_agg.is_some() { "Directory:  " } else { "File:       " };
        let size_label = if dir_agg.is_some() { "Total size: " } else { "File size:  " };
        let mut message = format!(
            "{}{}\n\
             Format:     {}\n\
             Dimensions: {}\n\
             Element:    {} bytes ({})\n\
             Records:    {}",
            path_label,
            display_path.display(),
            format.name(),
            meta.dimension,
            meta.element_size,
            element_type_label(meta.element_size),
            records_str,
        );
        if let Some(a) = &dir_agg {
            message.push_str(&format!("\nFiles:      {}", a.file_count));
        }
        if record_bytes > 0 {
            message.push_str(&format!("\nRecord size:{} bytes", format!(" {}", record_bytes)));
        }
        if let Some(size) = file_size {
            message.push_str(&format!("\n{}{}", size_label, format_bytes(size)));
        }

        if let Some(est) = &estimate {
            message.push_str(&format!(
                "\n\nEstimate basis (--sparse-scan):\n\
                 \x20 First file:        {} rows in {}\n\
                 \x20 Amortized bytes/record: {:.2}\n\
                 \x20 Projected total:   ~{} rows ({} / {:.2})",
                format_count(est.first_file_rows),
                format_bytes(est.first_file_bytes),
                est.amortized_bpr,
                format_count(est.rows),
                format_bytes(file_size.unwrap_or(0)),
                est.amortized_bpr,
            ));
        }

        // For xvec formats, report whether records are uniform or variable length
        if format.is_xvec() && dir_agg.is_none() {
            if let Some(size) = file_size {
                let stride = record_bytes as u64;
                if stride > 0 && size % stride == 0 {
                    let count = size / stride;
                    message.push_str(&format!(
                        "\nStructure:   uniform (all records dim={}, {} records)", meta.dimension, count));
                } else {
                    message.push_str("\nStructure:   variable-length (records have different dimensions)");
                }
            }
        }

        // Hint on directories: tell the caller about the alternative scan
        // modes available (sparse for speed, full-scan for per-record verify).
        if dir_agg.is_some() && !full_scan {
            let hint = if sparse_scan {
                "\n\nNote: estimate from first-file amortized rate. \
                 Drop --sparse-scan for an exact count from per-file metadata, \
                 or pass --full-scan to walk every record."
            } else {
                "\n\nNote: per-file metadata only. \
                 Pass --sparse-scan for a faster first-file-only estimate, \
                 or --full-scan to walk every record."
            };
            message.push_str(hint);
        }

        // --full-scan: walk all records and build a histogram of record lengths
        if full_scan && format.is_xvec() && dir_agg.is_none() {
            if let Some(size) = file_size {
                match scan_xvec_records(&source_path, format, size, ctx) {
                    Ok(scan) => {
                        message.push_str(&format!("\n\n── Record length scan ({} records) ──", scan.total_records));
                        if scan.distinct_dims.len() == 1 {
                            message.push_str(&format!("\n  All records: dim={}", scan.distinct_dims[0].0));
                        } else {
                            message.push_str(&format!("\n  {} distinct dimensions:", scan.distinct_dims.len()));
                            message.push_str(&format!("\n  {:>10}  {:>10}  {:>8}", "dim", "count", "pct"));
                            message.push_str(&format!("\n  {:>10}  {:>10}  {:>8}", "───", "─────", "───"));
                            for (dim, count) in &scan.distinct_dims {
                                let pct = 100.0 * *count as f64 / scan.total_records.max(1) as f64;
                                message.push_str(&format!("\n  {:>10}  {:>10}  {:>7.2}%", dim, count, pct));
                            }
                            message.push_str(&format!("\n  min dim: {}  max dim: {}  median dim: {}",
                                scan.min_dim, scan.max_dim, scan.median_dim));
                            message.push_str(&format!("\n  mean dim: {:.1}  stddev: {:.1}",
                                scan.mean_dim, scan.stddev_dim));
                        }
                    }
                    Err(e) => {
                        message.push_str(&format!("\n  scan error: {}", e));
                    }
                }
            }
        }

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![],
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
                description: "File or directory to describe".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "format".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Format override (auto-detected if omitted)".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "full-scan".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Walk every record (xvec dim histogram); without it, only file/footer metadata is read".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "sparse-scan".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Estimate cardinality from the first file only (stat all files, probe first); fastest mode for huge corpora".to_string(),
                role: OptionRole::Config,
            },
        ]
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

fn element_type_label(elem_size: usize) -> &'static str {
    match elem_size {
        1 => "u8/i8",
        2 => "f16/i16",
        4 => "f32/i32",
        8 => "f64/i64",
        _ => "unknown",
    }
}

struct DirAggregate {
    file_count: usize,
    total_bytes: u64,
    /// First file in sorted order — used as the representative sample for
    /// `--sparse-scan` cardinality estimation.
    first_file: PathBuf,
    /// On-disk bytes of the first file (cached so callers don't re-stat it).
    first_file_bytes: u64,
}

struct EstimatedRecords {
    rows: u64,
    /// Bytes of file storage attributable to one record on average — derived
    /// from the first file (`first_file_bytes / first_file_rows`). This
    /// captures format overhead (parquet footer, column metadata, dictionary
    /// pages, etc.) along with the per-record payload, so multiplying by the
    /// total directory size yields a usable estimate in O(1) read I/O.
    amortized_bpr: f64,
    first_file_rows: u64,
    first_file_bytes: u64,
}

/// Walk a directory and aggregate the on-disk size of every file whose
/// extension matches the chosen format. Stat-only — no file contents are
/// read here, so this stays cheap on huge corpora. Progress is reported
/// in file-count units via `ctx.ui`.
fn aggregate_directory(
    dir: &Path,
    format: VecFormat,
    ctx: &mut StreamContext,
) -> Result<DirAggregate, String> {
    let ext = format.preferred_extension().to_ascii_lowercase();
    // Some formats accept multiple extensions (e.g. .fvec/.fvecs); match
    // by VecFormat::from_extension on the candidate to stay consistent
    // with detection logic elsewhere.
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read dir {}: {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file() && match p.extension().and_then(|s| s.to_str()) {
                Some(e) => {
                    let lower = e.to_ascii_lowercase();
                    lower == ext || VecFormat::from_extension(&lower) == Some(format)
                }
                None => false,
            }
        })
        .collect();
    // Stable sort so "first file" is deterministic across runs.
    entries.sort();

    if entries.is_empty() {
        return Err(format!("no .{} files under {}", ext, dir.display()));
    }

    let first_file = entries[0].clone();
    let pb = ctx.ui.bar_with_unit(entries.len() as u64, "scanning", "files");
    let mut total_bytes: u64 = 0;
    let mut first_file_bytes: u64 = 0;
    for (i, p) in entries.iter().enumerate() {
        match std::fs::metadata(p) {
            Ok(m) => {
                let len = m.len();
                if i == 0 { first_file_bytes = len; }
                total_bytes += len;
            }
            Err(e) => {
                pb.finish();
                return Err(format!("stat {}: {}", p.display(), e));
            }
        }
        pb.inc(1);
    }
    pb.finish();

    Ok(DirAggregate {
        file_count: entries.len(),
        total_bytes,
        first_file,
        first_file_bytes,
    })
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{} ({:.2}B)", n, n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{} ({:.2}M)", n, n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{} ({:.1}K)", n, n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

struct ScanResult {
    total_records: u64,
    distinct_dims: Vec<(i32, u64)>,  // (dim, count) sorted by count descending
    min_dim: i32,
    max_dim: i32,
    median_dim: i32,
    mean_dim: f64,
    stddev_dim: f64,
}

/// Walk all records in an xvec file, counting dimensions.
fn scan_xvec_records(
    path: &Path,
    format: VecFormat,
    file_size: u64,
    ctx: &mut StreamContext,
) -> Result<ScanResult, String> {
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom};

    let elem_size = format.element_size() as u64;
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("open: {}", e))?;
    let mut dim_buf = [0u8; 4];

    let mut dim_counts: HashMap<i32, u64> = HashMap::new();
    let mut all_dims: Vec<i32> = Vec::new();
    let mut offset: u64 = 0;
    let mut record_idx: u64 = 0;

    // Progress bar
    let pb = ctx.ui.bar(file_size, "scanning records");

    loop {
        if offset >= file_size { break; }
        if offset + 4 > file_size {
            return Err(format!("truncated dim header at offset {}", offset));
        }

        f.seek(SeekFrom::Start(offset))
            .map_err(|e| format!("seek record {}: {}", record_idx, e))?;
        f.read_exact(&mut dim_buf)
            .map_err(|e| format!("read dim record {}: {}", record_idx, e))?;
        let dim = i32::from_le_bytes(dim_buf);

        if dim <= 0 {
            return Err(format!("invalid dim {} at record {} (offset {})", dim, record_idx, offset));
        }

        let record_size = 4 + dim as u64 * elem_size;
        if offset + record_size > file_size {
            return Err(format!("record {} truncated at offset {}", record_idx, offset));
        }

        *dim_counts.entry(dim).or_insert(0) += 1;
        all_dims.push(dim);

        offset += record_size;
        record_idx += 1;

        if record_idx % 10_000 == 0 {
            pb.set_position(offset);
        }
    }
    pb.finish();

    if all_dims.is_empty() {
        return Ok(ScanResult {
            total_records: 0,
            distinct_dims: vec![],
            min_dim: 0, max_dim: 0, median_dim: 0,
            mean_dim: 0.0, stddev_dim: 0.0,
        });
    }

    all_dims.sort_unstable();
    let min_dim = all_dims[0];
    let max_dim = *all_dims.last().unwrap();
    let median_dim = all_dims[all_dims.len() / 2];
    let mean_dim = all_dims.iter().map(|&d| d as f64).sum::<f64>() / all_dims.len() as f64;
    let variance = all_dims.iter()
        .map(|&d| { let diff = d as f64 - mean_dim; diff * diff })
        .sum::<f64>() / all_dims.len() as f64;
    let stddev_dim = variance.sqrt();

    let mut distinct: Vec<(i32, u64)> = dim_counts.into_iter().collect();
    distinct.sort_by(|a, b| b.1.cmp(&a.1)); // sort by count descending

    Ok(ScanResult {
        total_records: record_idx,
        distinct_dims: distinct,
        min_dim, max_dim, median_dim,
        mean_dim, stddev_dim,
    })
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
