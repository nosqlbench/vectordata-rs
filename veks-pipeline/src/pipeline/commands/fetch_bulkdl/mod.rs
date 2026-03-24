// Copyright (c) DataStax, Inc.
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

use log::{error, info, warn};

use config::{StatusFile, parse_token_spec};
use download::{download_file, head_content_length};
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
    /// File successfully downloaded or skipped (already correct size).
    Ok(String),
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

        // Filter out already completed files
        let pending: Vec<(String, String)> = {
            let st = status.lock().unwrap();
            all_files
                .into_iter()
                .filter(|(_, filename)| !st.completed.contains(filename))
                .collect()
        };

        let already_done = total - pending.len();
        ctx.ui.log(&format!(
            "bulkdl: {} total files, {} already completed, {} to download",
            total, already_done, pending.len()
        ));

        if pending.is_empty() {
            return CommandResult {
                status: Status::Ok,
                message: format!("all {} files already downloaded", total),
                produced: vec![output_dir],
                elapsed: start.elapsed(),
            };
        }

        let pb = ctx.ui.bar_with_unit(total as u64, "downloading", "files");
        pb.set_position(already_done as u64);
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
        for _ in 0..concurrency {
            let rx = Arc::clone(&rx);
            let result_tx = result_tx.clone();
            let output_dir = output_dir.clone();
            let ui = ctx.ui.clone();

            workers.push(std::thread::spawn(move || {
                loop {
                    let (url, filename) = match rx.lock().unwrap().recv() {
                        Ok(item) => item,
                        Err(_) => break, // channel closed, all work done
                    };

                    let dest = output_dir.join(&filename);

                    // Check if file already exists with correct size via HEAD
                    if dest.exists() {
                        if let Some(remote_len) = head_content_length(&url) {
                            if let Ok(meta) = fs::metadata(&dest) {
                                if meta.len() == remote_len {
                                    info!("Skipping {} (size matches)", filename);
                                    ui.inc_by_id(pb_id, 1);
                                    let _ = result_tx.send(DownloadResult::Ok(filename));
                                    continue;
                                }
                            }
                        }
                    }

                    // Download with retries
                    let mut last_err = String::new();
                    let mut succeeded = false;
                    for attempt in 1..=tries {
                        info!("Downloading {} (attempt {}/{})", url, attempt, tries);
                        match download_file(&url, &dest) {
                            Ok(()) => {
                                info!("Completed: {}", filename);
                                ui.inc_by_id(pb_id, 1);
                                let _ = result_tx.send(DownloadResult::Ok(filename.clone()));
                                succeeded = true;
                                break;
                            }
                            Err(e) => {
                                last_err = e.clone();
                                warn!(
                                    "Attempt {}/{} failed for {}: {}",
                                    attempt, tries, url, e
                                );
                            }
                        }
                    }
                    if !succeeded {
                        error!("All {} attempts failed for {}: {}", tries, url, last_err);
                        ui.inc_by_id(pb_id, 1);
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

        // Collect results on the main thread and flush the status file
        // periodically (every flush_interval completions).
        let flush_interval = concurrency.max(10);
        let mut downloaded_total = 0u32;
        let mut failed_total = 0u32;
        let mut since_last_flush = 0usize;

        for result in result_rx {
            match result {
                DownloadResult::Ok(filename) => {
                    downloaded_total += 1;
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

        CommandResult {
            status: status_result,
            message: format!(
                "{} downloaded, {} skipped, {} failed (of {} total)",
                downloaded_total, already_done, failed_total, total
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
            // No status file — either never run or status was deleted.
            // Check if directory has any files as a fallback.
            let has_files = std::fs::read_dir(output)
                .ok()
                .map(|entries| entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false)
                        && e.file_name().to_string_lossy() != ".down-rs-status.json"))
                .unwrap_or(false);
            return if has_files { ArtifactState::Complete } else { ArtifactState::Partial };
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
            ArtifactState::Partial
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
