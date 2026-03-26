// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline wrapper for the convert file operation.
//!
//! Extracts options from the pipeline `Options` map and delegates to the
//! existing format conversion logic.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;


use veks_core::formats::VecFormat;
use veks_core::formats::convert::convert_elements_into;
use veks_core::formats::reader;
use veks_core::formats::writer::{self, SinkConfig};
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

/// Number of records to buffer in the read-ahead channel.
///
/// Allows the reader thread to stay ahead of the convert+write thread,
/// hiding I/O latency behind conversion work.
const READ_AHEAD_BUFSIZE: usize = 4096;

/// Pipeline command: convert a vector file between formats.
pub struct ConvertFileOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ConvertFileOp)
}

impl CommandOp for ConvertFileOp {
    fn command_path(&self) -> &str {
        "transform convert"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Convert a vector file between formats".into(),
            body: format!(
                r#"# convert file

Convert a vector file between formats.

## Description

The convert file command transforms a vector file from one format to another.
It supports all recognized vector formats including fvec, ivec, bvec, npy, mvec,
slab, and xvec. Both the source and target formats can be specified explicitly,
or the source format can be auto-detected from the file extension.

When the source and target formats use different element sizes, the command
performs element widening or narrowing automatically. For example, converting
from fvec (f32, 4 bytes per element) to a half-precision format (f16, 2 bytes
per element) narrows each element, while going from bvec (u8, 1 byte) to fvec
widens them. The conversion is applied per-element across every dimension of
every record.

Internally, the command uses a read-ahead pipeline architecture: a background
reader thread fills a bounded channel with source records while the main thread
handles element conversion and writes to the output sink. This design hides I/O
latency behind conversion work, keeping throughput high even for large files.
The read-ahead buffer holds up to 4096 records.

Common format pairs include:

- npy to fvec or ivec -- converting numpy exports to native vector formats
- fvec to mvec -- migrating to memory-mapped layout
- bvec to fvec -- widening byte vectors to float vectors
- any format to slab -- converting to paged slab storage

## Data Preparation Role

The convert command is typically used as a standalone operation rather than as
part of a multi-step pipeline. It is most useful for format migration: when
source data arrives in one format but downstream tools or benchmarks require
another. In pipeline contexts, the import command usually handles format
selection automatically, so explicit conversion is only needed for special cases
such as changing element precision after an initial import.

## Examples

Convert a numpy file to fvec format:

    convert file --source vectors.npy --output vectors.fvec --to fvec

Convert with explicit source format override:

    convert file --source data.bin --output data.fvec --from bvec --to fvec

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "iothreads".into(), description: "Concurrent I/O operations".into(), adjustable: false },
            ResourceDesc { name: "readahead".into(), description: "Read-ahead buffer size".into(), adjustable: false },
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

        let to_format = if let Some(to_str) = options.get("to") {
            match VecFormat::from_extension(to_str) {
                Some(f) => f,
                None => return error_result(format!("unknown output format: '{}'", to_str), start),
            }
        } else {
            // Infer from output file extension
            match VecFormat::detect(&output_path) {
                Some(f) => f,
                None => return error_result(
                    format!("cannot infer output format from '{}', set 'to' option", output_path.display()),
                    start,
                ),
            }
        };

        let source_format = if let Some(fmt_str) = options.get("from") {
            match VecFormat::from_extension(fmt_str) {
                Some(f) => f,
                None => return error_result(format!("unknown source format: '{}'", fmt_str), start),
            }
        } else {
            match VecFormat::detect(&source_path) {
                Some(f) => f,
                None => {
                    return error_result(
                        format!(
                            "cannot detect source format for '{}', set 'from' option",
                            source_path.display()
                        ),
                        start,
                    )
                }
            }
        };

        let explicit_limit: Option<u64> = match options.parse_opt("limit") {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let fraction: Option<f64> = match options.parse_opt::<f64>("fraction") {
            Ok(v) => v.filter(|&f| f > 0.0 && f < 1.0),
            Err(e) => return error_result(e, start),
        };

        let slab_page_size: Option<u32> = match options.parse_opt("slab_page_size") {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let slab_namespace: u8 = match options.parse_or("slab_namespace", 1u8) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };

        // Open source
        ctx.ui.log(&format!(
            "  converting {} ({}) -> {} ({})",
            source_path.display(),
            source_format.name(),
            output_path.display(),
            to_format.name(),
        ));
        ctx.ui.log("  opening source...");
        let threads = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
        // Use facet-aware reader: metadata_content parquet uses the MNode
        // reader which handles arbitrary scalar columns, not just list-of-float.
        let facet = options.get("facet")
            .and_then(|f| veks_core::formats::facet::Facet::from_key(f)
                .or_else(|| veks_core::formats::facet::Facet::from_alias(f)));
        let source_result = match facet {
            Some(f) => reader::open_source_for_facet(&source_path, source_format, f, threads, None),
            None => reader::open_source(&source_path, source_format, threads, None),
        };
        let mut source = match source_result {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open source {}: {}", source_path.display(), e), start),
        };

        let dimension = source.dimension();
        let src_element_size = source.element_size();
        let dst_element_size = to_format.element_size();
        // Compute effective limit from explicit limit or fraction.
        let limit = match (explicit_limit, fraction, source.record_count()) {
            (Some(lim), _, _) => Some(lim),
            (None, Some(frac), Some(total)) => {
                let needed = ((total as f64) * frac).ceil() as u64;
                ctx.ui.log(&format!("  fraction {:.0}%: converting {} of {} records",
                    frac * 100.0, needed, total));
                Some(needed.min(total))
            }
            _ => None,
        };
        let record_count = match (source.record_count(), limit) {
            (Some(total), Some(lim)) => Some(total.min(lim)),
            (count, _) => count,
        };

        // Determine if element conversion is needed
        let needs_conversion = dst_element_size > 0
            && src_element_size > 0
            && src_element_size != dst_element_size;

        ctx.ui.log(&format!(
            "  source: dimension={}, element_size={}, records={}",
            dimension,
            src_element_size,
            record_count.map_or("unknown".to_string(), |n| n.to_string()),
        ));
        if needs_conversion {
            ctx.ui.log(&format!(
                "  element conversion: {} ({} bytes) -> {} ({} bytes)",
                source_format.data_type_name(),
                src_element_size,
                to_format.data_type_name(),
                dst_element_size,
            ));
        }

        // Create output directory
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(
                        format!("failed to create directory: {}", e),
                        start,
                    );
                }
            }
        }

        // ── Fast path: parallel mmap conversion for npy→xvec ──
        //
        // When the source is a directory of npy files and the output is an
        // xvec format with known record count, we can:
        // 1. Scan npy headers to get per-file row counts and offsets
        // 2. Pre-allocate the output file via mmap
        // 3. Load + convert + write files in parallel (each file writes to
        //    its computed offset in the mmap)
        let mut count: u64;
        let is_npy = source_format == VecFormat::Npy;
        let is_xvec = to_format.is_xvec();
        let is_dir = source_path.is_dir();
        let has_count = record_count.is_some();
        let use_mmap_parallel = is_npy && is_xvec && is_dir && has_count;

        log::debug!(
            "mmap eligibility: npy={} xvec={} dir={} count={} → {}",
            is_npy, is_xvec, is_dir, has_count,
            if use_mmap_parallel { "PARALLEL" } else { "sequential" }
        );

        log::debug!("convert path: {}",
            if use_mmap_parallel { "PARALLEL MMAP" } else { "sequential" });

        if use_mmap_parallel {
            drop(source); // release the sequential reader

            let total = record_count.unwrap();
            let record_stride = 4 + dimension as usize * dst_element_size;
            let total_bytes = total * record_stride as u64;
            let rayon_threads = rayon::current_num_threads();

            ctx.ui.log(&format!(
                "  mode: parallel mmap ({} threads, {} records, {} → {})",
                rayon_threads,
                total,
                source_format.data_type_name(),
                to_format.data_type_name(),
            ));
            ctx.ui.log(&format!(
                "  output: {} ({} bytes, stride {})",
                veks_core::paths::rel_display(&output_path),
                total_bytes,
                record_stride,
            ));

            // Scan npy manifest for per-file offsets
            ctx.ui.log("  scanning npy headers...");
            let (mut manifest, _dim, _total) = match veks_core::formats::reader::npy::scan_npy_manifest(&source_path) {
                Ok(m) => m,
                Err(e) => return error_result(format!("failed to scan npy manifest: {}", e), start),
            };

            // When limited, trim manifest to only the files covering `total` records.
            // Truncate the last included file's row count if it extends past the limit.
            if let Some(lim) = limit {
                let mut kept = 0;
                for (i, entry) in manifest.iter_mut().enumerate() {
                    if entry.offset >= lim {
                        manifest.truncate(i);
                        break;
                    }
                    let end = entry.offset + entry.rows;
                    if end > lim {
                        entry.rows = lim - entry.offset;
                    }
                    kept += entry.rows;
                }
                if kept < lim && !manifest.is_empty() {
                    // Already covers enough
                }
                ctx.ui.log(&format!("  limit: {} records, using {} of {} npy files",
                    total, manifest.len(), _total));
            }
            log::debug!("{} npy files, {} total records", manifest.len(), total);

            // Pre-allocate output file
            log::debug!("pre-allocating {} ...", format_size(total_bytes));
            let alloc_start = Instant::now();
            let mmap_writer = match veks_core::formats::writer::mmap_xvec::MmapXvecWriter::create(
                &output_path, dimension, dst_element_size, total,
            ) {
                Ok(w) => w,
                Err(e) => return error_result(format!("failed to create mmap writer: {}", e), start),
            };
            log::debug!("pre-allocation done ({:.1}s)",
                alloc_start.elapsed().as_secs_f64());
            let shared_writer = veks_core::formats::writer::mmap_xvec::SharedMmapWriter::new(mmap_writer);

            let pb = ctx.ui.bar(total, "converting records");
            let pb_id = pb.id();

            // Two-stage pipeline: loaders → chunked writers with bounded buffer.
            //
            // Stage 1 — I/O loaders (few threads) read npy files in roughly
            // sequential order, keeping I/O streaming-friendly. Each loaded
            // file is wrapped in Arc and split into row-range chunks.
            //
            // Stage 2 — CPU writer threads (more) pull individual chunks from
            // a bounded channel. Multiple writers process different chunks of
            // the *same* file concurrently, giving intra-file parallelism.
            // The bounded channel provides backpressure: loaders block when
            // the buffer is full, preventing unbounded memory growth.

            /// Rows per chunk for intra-file parallelism. Balances per-chunk
            /// overhead against keeping all writer threads busy. 8192 rows at
            /// 768-dim f16 ≈ 12 MiB per chunk.
            const CHUNK_ROWS: usize = 8192;

            let dtype = match veks_core::formats::reader::npy::scan_npy_headers(&source_path, None) {
                Ok(s) => s.dtype,
                Err(e) => return error_result(e, start),
            };

            // I/O loaders: few threads, streaming access pattern
            let io_threads = ctx.governor.request("iothreads", 4).max(1) as usize;
            // CPU writers: more threads for SIMD conversion + mmap writes
            let hw_threads = std::thread::available_parallelism()
                .map(|n| n.get()).unwrap_or(8) as u64;
            let cpu_threads = ctx.governor.request("threads", hw_threads / 2).max(2) as usize;
            // Buffer: enough chunks to keep all writers busy plus headroom
            let buffer_depth = (cpu_threads * 3).max(8);

            ctx.ui.log(&format!(
                "  pipeline: {} I/O loaders → [{}×{}row chunks] → {} writers",
                io_threads, buffer_depth, CHUNK_ROWS, cpu_threads));

            /// A loaded npy file shared between chunk work items.
            struct LoadedFile {
                offset: u64,
                rows: u64,
                array: veks_core::formats::reader::npy::NpyArrayData,
            }

            /// A chunk of rows within a loaded file. Multiple chunks reference
            /// the same Arc<LoadedFile>, enabling intra-file parallelism.
            struct WriteChunk {
                file: std::sync::Arc<LoadedFile>,
                start_row: usize,   // inclusive, within file
                end_row: usize,     // exclusive, within file
            }

            let (chunk_tx, chunk_rx) = std::sync::mpsc::sync_channel::<WriteChunk>(buffer_depth);
            let error_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

            // Work queue for loaders (files in order for sequential I/O)
            let work_queue = std::sync::Arc::new(std::sync::Mutex::new(
                manifest.into_iter().collect::<std::collections::VecDeque<_>>()
            ));

            // Spawn loader threads
            let mut loader_handles = Vec::with_capacity(io_threads);
            for _ in 0..io_threads {
                let queue = work_queue.clone();
                let tx = chunk_tx.clone();
                let err = error_flag.clone();
                let dt = dtype;

                loader_handles.push(std::thread::Builder::new()
                    .name("npy-loader".into())
                    .spawn(move || {
                        loop {
                            if err.load(std::sync::atomic::Ordering::Relaxed) { break; }
                            let entry = match queue.lock().unwrap().pop_front() {
                                Some(e) => e,
                                None => break,
                            };
                            match veks_core::formats::reader::npy::load_npy(&entry.path, dt) {
                                Ok(array) => {
                                    let file = std::sync::Arc::new(LoadedFile {
                                        offset: entry.offset,
                                        rows: entry.rows,
                                        array,
                                    });
                                    // Split into chunks for intra-file parallelism
                                    let total_rows = file.rows as usize;
                                    let mut start = 0;
                                    while start < total_rows {
                                        let end = (start + CHUNK_ROWS).min(total_rows);
                                        if tx.send(WriteChunk {
                                            file: file.clone(),
                                            start_row: start,
                                            end_row: end,
                                        }).is_err() {
                                            return; // writers gone
                                        }
                                        start = end;
                                    }
                                }
                                Err(e) => {
                                    err.store(true, std::sync::atomic::Ordering::Relaxed);
                                    log::error!("npy load failed: {}: {}", entry.path.display(), e);
                                    break;
                                }
                            }
                        }
                    }).expect("failed to spawn loader"));
            }
            drop(chunk_tx); // main thread's clone — channel closes when all loaders finish

            // Spawn writer threads that pull individual chunks
            let chunk_rx = std::sync::Arc::new(std::sync::Mutex::new(chunk_rx));
            let mut writer_handles = Vec::with_capacity(cpu_threads);

            for _ in 0..cpu_threads {
                let rx = chunk_rx.clone();
                let writer = shared_writer.clone_handle();
                let err = error_flag.clone();
                let ui = ctx.ui.clone();
                let pb = pb_id;

                writer_handles.push(std::thread::Builder::new()
                    .name("mmap-writer".into())
                    .spawn(move || {
                        let mut conv_buf = vec![0u8; dimension as usize * dst_element_size];
                        loop {
                            if err.load(std::sync::atomic::Ordering::Relaxed) { break; }
                            let chunk = match rx.lock().unwrap().recv() {
                                Ok(c) => c,
                                Err(_) => break, // channel closed
                            };

                            let file_offset = chunk.file.offset;
                            for row in chunk.start_row..chunk.end_row {
                                let data = chunk.file.array.row_bytes(row);
                                let ordinal = file_offset + row as u64;
                                if needs_conversion {
                                    if convert_elements_into(&data, src_element_size, dst_element_size, &mut conv_buf).is_some() {
                                        writer.write_record_at(ordinal, &conv_buf);
                                    } else {
                                        writer.write_record_at(ordinal, &data);
                                    }
                                } else {
                                    writer.write_record_at(ordinal, &data);
                                }
                            }

                            // Advise writeback for the chunk's byte range
                            let chunk_start_ordinal = (file_offset + chunk.start_row as u64) as usize;
                            let chunk_rows = chunk.end_row - chunk.start_row;
                            let byte_offset = chunk_start_ordinal * writer.record_stride();
                            let byte_len = chunk_rows * writer.record_stride();
                            writer.advise_writeback(byte_offset, byte_len);

                            ui.inc_by_id(pb, chunk_rows as u64);
                        }
                    }).expect("failed to spawn writer"));
            }

            // Wait for completion
            for h in loader_handles { let _ = h.join(); }
            for h in writer_handles { let _ = h.join(); }

            pb.finish();

            if error_flag.load(std::sync::atomic::Ordering::Relaxed) {
                return error_result("one or more npy files failed to convert".to_string(), start);
            }

            if let Err(e) = shared_writer.finish() {
                return error_result(format!("failed to flush output: {}", e), start);
            }

            count = total;

        // ── Sequential path (all other format combinations) ──
        } else {

        ctx.ui.log("  mode: sequential (read-ahead pipeline)");

        // Open sink
        let sink_config = SinkConfig {
            dimension,
            source_format: if needs_conversion { to_format } else { source_format },
            slab_page_size,
            slab_namespace,
        };
        let mut sink = match writer::open_sink(&output_path, to_format, &sink_config) {
            Ok(s) => s,
            Err(e) => return error_result(format!("failed to open sink {}: {}", output_path.display(), e), start),
        };

        let pb = if let Some(total) = record_count {
            ctx.ui.bar(total, "converting records")
        } else {
            ctx.ui.spinner("converting records")
        };

        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(READ_AHEAD_BUFSIZE);

        let reader_handle = std::thread::Builder::new()
            .name("convert-reader".into())
            .spawn(move || {
                while let Some(data) = source.next_record() {
                    if tx.send(data).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn reader thread");

        let dst_record_bytes = dimension as usize * dst_element_size;
        let mut conv_buf = if needs_conversion {
            vec![0u8; dst_record_bytes]
        } else {
            Vec::new()
        };

        if ctx.governor.checkpoint() {
            log::info!("governor: throttle active");
        }

        count = 0;
        for data in rx {
            if needs_conversion {
                if let Some(_n) = convert_elements_into(
                    &data, src_element_size, dst_element_size, &mut conv_buf,
                ) {
                    sink.write_record(count as i64, &conv_buf);
                } else {
                    sink.write_record(count as i64, &data);
                }
            } else {
                sink.write_record(count as i64, &data);
            };
            count += 1;
            pb.inc(1);
            if let Some(lim) = limit {
                if count >= lim { break; }
            }
        }

        reader_handle.join().expect("reader thread panicked");
        pb.finish();

        if let Err(e) = sink.finish() {
            return error_result(format!("failed to finalize output: {}", e), start);
        }

        } // end sequential path

        // Post-conversion verification: check file size alignment
        if to_format.is_xvec() && output_path.exists() {
            let record_stride = 4 + dimension as usize * dst_element_size;
            let file_size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
            let expected_size = count * record_stride as u64;

            if file_size != expected_size {
                let actual_records = file_size / record_stride as u64;
                let remainder = file_size % record_stride as u64;
                return error_result(format!(
                    "output size mismatch: expected {} bytes ({} records × {} stride) \
                     but got {} bytes ({} complete records, {} trailing bytes). \
                     The conversion may have been interrupted.",
                    expected_size, count, record_stride,
                    file_size, actual_records, remainder,
                ), start);
            }

            // Store verified record count in variables.yaml so the bound
            // checker can detect files interrupted at a record boundary.
            let var_name = format!("verified_count:{}",
                output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &count.to_string());
            ctx.defaults.insert(var_name, count.to_string());
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "converted {} records ({} -> {}) to {}",
                count,
                source_format.name(),
                to_format.name(),
                output_path.display()
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
                description: "Source file or directory".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output file path".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "to".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: None,
                description: "Output format (fvec, ivec, slab, etc.) — inferred from output extension if omitted".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "from".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Source format override (auto-detected if omitted)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "limit".to_string(),
                type_name: "integer".to_string(),
                required: false,
                default: None,
                description: "Maximum number of records to convert (omit for all)".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output"],
        )
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

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 { format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0) }
    else if bytes >= 1_048_576 { format!("{:.1} MiB", bytes as f64 / 1_048_576.0) }
    else if bytes >= 1024 { format!("{:.1} KiB", bytes as f64 / 1024.0) }
    else { format!("{} B", bytes) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

