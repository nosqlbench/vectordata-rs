// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: bulk download files from a URL template.
//!
//! Downloads files by expanding token placeholders in a URL template and
//! fetching each expanded URL into a local output directory. Supports
//! concurrency, retries, and resume of partial downloads.

mod config;
mod download;
mod expand;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;


use config::{StatusFile, parse_token_spec};
use download::{download_file, head_content_length_retry};
use expand::expand_tokens;
use crate::pipeline::command::{
    ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

/// Pipeline command: bulk download files from a URL template.
pub struct FetchBulkdlOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(FetchBulkdlOp)
}

/// Result sent from a download worker back to the main thread.
enum DownloadResult {
    /// File successfully downloaded or skipped. Includes filename and bytes.
    Ok(String, u64),
    /// All retry attempts failed.
    Failed,
}

impl CommandOp for FetchBulkdlOp {
    fn command_path(&self) -> &str {
        "download bulk"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Bulk download files from a URL template".into(),
            body: format!(
                r#"# download bulk

Bulk download files from a URL template.

## Description

Downloads files by expanding token placeholders in a URL template and
fetching each expanded URL into a local output directory. Supports
concurrency control, retries, and resume of partial downloads.

Token placeholders use `${{name}}` syntax in the URL. Each token has a
range specification like `[0..409]` (inclusive). Multiple tokens produce
the cartesian product of all ranges.

## How It Works

The command parses the token specifications (e.g. `number=[0..409]`)
into value ranges, then computes the cartesian product of all token
ranges to generate the full list of (URL, filename) pairs. A status
file (`.down-rs-status.json`) in the output directory tracks which
files have already been downloaded, enabling resume across runs.
Pending files are downloaded in batches with configurable concurrency.
Each download supports retries on failure, and a HEAD request checks
whether a local file already has the correct size before downloading.

## Data Preparation Role

`fetch bulkdl` is used to acquire source data files that are hosted
as numbered sequences on web servers rather than in structured
repositories. This is common for large-scale datasets like LAION
where files are split into hundreds of numbered shards. The URL
template with token expansion generates all the necessary download
URLs without requiring them to be listed individually. Combined with
the resume-tracking status file and configurable concurrency, this
command can reliably download hundreds of gigabytes across multiple
pipeline runs, picking up where it left off after interruptions.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "iothreads".into(),
                description: "Concurrent download connections".into(),
                adjustable: false,
            },
            ResourceDesc {
                name: "network".into(),
                description: "Network bandwidth".into(),
                adjustable: false,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let baseurl = match options.require("baseurl") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };

        let output_dir = match options.require("output") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };

        let tries: u32 = options
            .get("tries")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        let concurrency: usize = options
            .get("concurrency")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        // Parse token specifications from options.
        // Tokens are specified as `tokens: "name=[start..end]"` or
        // `tokens: "name=[start..end],name2=[start..end]"`.
        let mut token_map = std::collections::HashMap::new();
        if let Some(tokens_str) = options.get("tokens") {
            for part in tokens_str.split(',') {
                let part = part.trim();
                if let Some(eq_pos) = part.find('=') {
                    let name = part[..eq_pos].trim().to_string();
                    let spec_str = part[eq_pos + 1..].trim();
                    match parse_token_spec(spec_str) {
                        Some(spec) => {
                            token_map.insert(name, spec);
                        }
                        None => {
                            return error_result(
                                format!("invalid token spec '{}', expected format like [0..409]", spec_str),
                                start,
                            );
                        }
                    }
                } else {
                    return error_result(
                        format!("invalid token entry '{}', expected name=[start..end]", part),
                        start,
                    );
                }
            }
        }

        // Ensure output directory exists
        if let Err(e) = fs::create_dir_all(&output_dir) {
            return error_result(
                format!("failed to create directory {}: {}", output_dir.display(), e),
                start,
            );
        }

        // Expand URL template into (url, filename) pairs
        let all_files = expand_tokens(&baseurl, &token_map);
        let total = all_files.len();

        // Load status file for resume tracking
        let status_path = output_dir.join(".down-rs-status.json");
        let status = Arc::new(Mutex::new(StatusFile::load(&status_path)));

        // Filter out already completed files.
        // Two checks: (1) status file from previous runs, and (2) local
        // file size matches a quick HEAD check. This avoids sending files
        // to workers that are already fully downloaded but weren't in the
        // status file (e.g., first run after a manual download).
        let mut pending: Vec<(String, String)> = Vec::new();
        let mut pending_bytes: u64 = 0;
        let mut skipped_status = 0usize;
        let mut skipped_size = 0usize;
        let verify_pb = ctx.ui.bar_with_unit(total as u64, "verifying", "files");
        {
            let mut st = status.lock().unwrap();
            for (idx, (url, filename)) in all_files.into_iter().enumerate() {
                verify_pb.set_position(idx as u64 + 1);
                if st.completed.contains(&filename) {
                    skipped_status += 1;
                    continue;
                }
                let dest = output_dir.join(&filename);
                if dest.exists() {
                    if let Ok(meta) = fs::metadata(&dest) {
                        let local_len = meta.len();
                        if local_len > 0 {
                            verify_pb.set_message(format!("checking {}", filename));
                            match head_content_length_retry(&url, 10) {
                                Ok(remote_len) if local_len == remote_len => {
                                    st.completed.push(filename.clone());
                                    skipped_size += 1;
                                    continue;
                                }
                                Ok(remote_len) => {
                                    ctx.ui.log(&format!("size mismatch {}: local={} remote={}, re-downloading",
                                        filename, local_len, remote_len));
                                    pending_bytes += remote_len;
                                }
                                Err(e) => {
                                    ctx.ui.log(&format!("WARNING: could not verify {}: {}", filename, e));
                                }
                            }
                        }
                    }
                } else {
                    verify_pb.set_message(format!("sizing {}", filename));
                    if let Ok(remote_len) = head_content_length_retry(&url, 3) {
                        pending_bytes += remote_len;
                    }
                }
                pending.push((url, filename));
            }
            if skipped_size > 0 {
                let _ = st.save(&status_path);
            }
        }

        verify_pb.finish();

        let already_done = skipped_status + skipped_size;
        ctx.ui.log(&format!(
            "bulkdl: {} total, {} from status, {} size-verified, {} to download",
            total, skipped_status, skipped_size, pending.len()
        ));

        if pending.is_empty() {
            return CommandResult {
                status: Status::Ok,
                message: format!("all {} files already downloaded", total),
                produced: vec![output_dir],
                elapsed: start.elapsed(),
            };
        }

        let pb = if pending_bytes > 0 {
            ctx.ui.bar_with_unit(pending_bytes, "downloading", "bytes")
        } else {
            ctx.ui.bar_with_unit(pending.len() as u64, "downloading", "files")
        };
        let byte_bar = pending_bytes > 0;
        let pb_id = pb.id();

        // Channel-based work stealing: a bounded work queue feeds N
        // persistent worker threads. Each worker pulls the next job as
        // soon as it finishes, keeping all slots saturated instead of
        // blocking at batch boundaries.
        let (tx, rx) = std::sync::mpsc::sync_channel::<(String, String)>(concurrency * 2);
        let rx = Arc::new(Mutex::new(rx));

        // Completed filenames are collected via a result channel so the
        // status file is flushed periodically from the main thread
        // instead of under contention from every worker.
        let (result_tx, result_rx) = std::sync::mpsc::channel::<DownloadResult>();

        // Spawn persistent worker threads
        let mut workers = Vec::with_capacity(concurrency);
        for worker_id in 0..concurrency {
            let rx = Arc::clone(&rx);
            let result_tx = result_tx.clone();
            let output_dir = output_dir.clone();
            let ui = ctx.ui.clone();
            let w_pb_id = pb_id;
            let w_byte_bar = byte_bar;

            workers.push(std::thread::spawn(move || {
                loop {
                    let (url, filename) = match rx.lock().unwrap().recv() {
                        Ok(item) => item,
                        Err(_) => break, // channel closed, all work done
                    };

                    let dest = output_dir.join(&filename);

                    // Check if file already exists with correct size via HEAD (with retries)
                    if dest.exists() {
                        if let Ok(meta) = fs::metadata(&dest) {
                            if meta.len() > 0 {
                                if let Ok(remote_len) = head_content_length_retry(&url, 10) {
                                    if meta.len() == remote_len {
                                        ui.log(&format!("[w{}] skip {} ({})",
                                            worker_id, filename, format_size(remote_len)));
                                        let _ = result_tx.send(DownloadResult::Ok(filename, 0));
                                        continue;
                                    } else {
                                        ui.log(&format!("[w{}] size mismatch {}: local={} remote={}",
                                            worker_id, filename, meta.len(), remote_len));
                                    }
                                }
                            }
                        }
                    }

                    // Download with retries
                    let mut last_err = String::new();
                    let mut succeeded = false;
                    for attempt in 1..=tries {
                        if attempt == 1 {
                            ui.log(&format!("[w{}] {}", worker_id, filename));
                        } else {
                            ui.log(&format!("[w{}] {} (retry, tries {}/{})",
                                worker_id, filename, attempt, tries));
                        }
                        let t0 = Instant::now();
                        // Track bytes for live progress bar updates
                        let prev_reported = std::sync::atomic::AtomicU64::new(0);
                        let ui_ref = &ui;
                        let on_bytes = |total_written: u64| {
                            if w_byte_bar {
                                let prev = prev_reported.load(std::sync::atomic::Ordering::Relaxed);
                                let delta = total_written.saturating_sub(prev);
                                if delta > 0 {
                                    ui_ref.inc_by_id(w_pb_id, delta);
                                    prev_reported.store(total_written, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                        };
                        match download_file(&url, &dest, &on_bytes) {
                            Ok(()) => {
                                let file_size = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                                let secs = t0.elapsed().as_secs_f64();
                                let mbps = if secs > 0.0 { file_size as f64 / 1_048_576.0 / secs } else { 0.0 };
                                ui.log(&format!("[w{}] done {} ({} @ {:.1} MiB/s)",
                                    worker_id, filename, format_size(file_size), mbps));
                                // Don't send file_size to result — bytes already incremented live
                                let _ = result_tx.send(DownloadResult::Ok(filename.clone(), file_size));
                                succeeded = true;
                                break;
                            }
                            Err(e) => {
                                last_err = e.clone();
                                ui.log(&format!("[w{}] FAIL {}/{} {}: {}",
                                    worker_id, attempt, tries, filename, e));
                            }
                        }
                    }
                    if !succeeded {
                        ui.log(&format!("[w{}] FAILED {} after {} attempts: {}",
                            worker_id, filename, tries, last_err));
                        let _ = result_tx.send(DownloadResult::Failed);
                    }
                }
            }));
        }
        // Drop the sender clone held by main so workers see channel close
        drop(result_tx);

        // Feed work items into the channel. This blocks when the channel
        // is full (bounded), providing natural backpressure.
        let feeder = {
            let pending = pending;
            std::thread::spawn(move || {
                for item in pending {
                    if tx.send(item).is_err() {
                        break; // workers gone
                    }
                }
                // tx drops here, closing the channel
            })
        };

        // Collect results on the main thread, track bandwidth, and flush
        // the status file periodically.
        let flush_interval = concurrency.max(10);
        let mut downloaded_total = 0u32;
        let mut downloaded_bytes = 0u64;
        let mut failed_total = 0u32;
        let mut since_last_flush = 0usize;
        let bandwidth_start = Instant::now();

        for result in result_rx {
            match result {
                DownloadResult::Ok(filename, bytes) => {
                    downloaded_total += 1;
                    downloaded_bytes += bytes;
                    if byte_bar {
                        // Bytes already incremented live by the worker's on_bytes callback.
                        ctx.ui.set_message_by_id(pb_id, format!(
                            "{} files", downloaded_total,
                        ));
                    } else {
                        ctx.ui.inc_by_id(pb_id, 1);
                    }
                    let mut st = status.lock().unwrap();
                    st.completed.push(filename);
                    since_last_flush += 1;
                    if since_last_flush >= flush_interval {
                        let _ = st.save(&status_path);
                        since_last_flush = 0;
                    }
                }
                DownloadResult::Failed => {
                    failed_total += 1;
                    if !byte_bar {
                        ctx.ui.inc_by_id(pb_id, 1);
                    }
                }
            }
        }

        // Final flush
        {
            let st = status.lock().unwrap();
            let _ = st.save(&status_path);
        }

        // Wait for feeder and workers to finish
        let _ = feeder.join();
        for w in workers {
            let _ = w.join();
        }

        pb.finish();

        let status_result = if failed_total > 0 {
            Status::Error
        } else {
            Status::Ok
        };

        let total_elapsed = bandwidth_start.elapsed().as_secs_f64().max(0.001);
        let avg_mbps = (downloaded_bytes as f64 / 1_048_576.0) / total_elapsed;

        CommandResult {
            status: status_result,
            message: format!(
                "{} downloaded ({:.1} MiB, {:.1} MiB/s), {} skipped, {} failed (of {} total)",
                downloaded_total, downloaded_bytes as f64 / 1_048_576.0, avg_mbps,
                already_done, failed_total, total
            ),
            produced: vec![output_dir],
            elapsed: start.elapsed(),
        }
    }

    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        // The output is a directory. Check the status file to see if all
        // expected files are recorded as completed. This avoids any network
        // I/O — the status file is the authoritative record of what has been
        // downloaded.
        if !output.exists() {
            return ArtifactState::Absent;
        }
        if !output.is_dir() {
            // Not a directory — defer to default check
            return crate::pipeline::bound::check_artifact_default(output, options);
        }

        let status_path = output.join(".down-rs-status.json");
        if !status_path.exists() {
            return ArtifactState::PartialResumable;
        }

        // Parse token specs from options to compute expected file count
        let baseurl = match options.get("baseurl") {
            Some(u) => u.to_string(),
            None => return ArtifactState::Complete, // can't verify without URL
        };

        let mut token_map = std::collections::HashMap::new();
        if let Some(tokens_str) = options.get("tokens") {
            for part in tokens_str.split(',') {
                let part = part.trim();
                if let Some(eq_pos) = part.find('=') {
                    let name = part[..eq_pos].trim().to_string();
                    let spec_str = part[eq_pos + 1..].trim();
                    if let Some(spec) = parse_token_spec(spec_str) {
                        token_map.insert(name, spec);
                    }
                }
            }
        }

        let expected_files = expand_tokens(&baseurl, &token_map);
        let expected_count = expected_files.len();

        let status = StatusFile::load(&status_path);
        let completed_count = status.completed.len();

        if completed_count >= expected_count {
            ArtifactState::Complete
        } else {
            ArtifactState::PartialResumable
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "baseurl".to_string(),
                type_name: "String".to_string(),
                required: true,
                default: None,
                description: "URL template with ${token} placeholders".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output directory for downloaded files".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "tokens".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Token ranges, e.g. \"number=[0..409]\" or \"x=[0..9],y=[0..9]\"".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "tries".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("3".to_string()),
                description: "Number of download attempts per file".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "concurrency".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("4".to_string()),
                description: "Maximum concurrent downloads".to_string(),
                role: OptionRole::Config,
        },
        ]
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 { format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0) }
    else if bytes >= 1_048_576 { format!("{:.1} MiB", bytes as f64 / 1_048_576.0) }
    else if bytes >= 1024 { format!("{:.1} KiB", bytes as f64 / 1024.0) }
    else { format!("{} B", bytes) }
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

    #[test]
    fn test_describe_options() {
        let op = FetchBulkdlOp;
        let opts = op.describe_options();
        assert!(opts.iter().any(|o| o.name == "baseurl" && o.required));
        assert!(opts.iter().any(|o| o.name == "output" && o.required));
        assert!(opts.iter().any(|o| o.name == "tokens" && !o.required));
        assert!(opts.iter().any(|o| o.name == "concurrency" && !o.required));
    }

    #[test]
    fn test_command_path() {
        let op = FetchBulkdlOp;
        assert_eq!(op.command_path(), "download bulk");
    }
}
