// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: bulk download files into a local directory.
//!
//! Two modes, dispatched on the required `mode` option:
//!
//! - `mode: template` — Expand `${token}` placeholders in `baseurl` against
//!   one or more named ranges (e.g. `tokens: "n=[0..409]"`) to generate the
//!   list of HTTP URLs to download.
//! - `mode: cos` — Enumerate objects under `bucket`/`prefix` on an
//!   S3-compatible endpoint (IBM COS in particular) using `ListObjectsV2`,
//!   signing each request with AWS SigV4. Credentials are read from the
//!   standard AWS environment variables.
//!
//! Both modes share the same worker pool, retry policy, status-file resume
//! tracking, and progress bar. The mode option determines only how the list
//! of files is produced; the download mechanics below are mode-aware only at
//! the per-request level.

mod config;
mod cos;
mod download;
mod expand;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;


use config::{StatusFile, parse_token_spec};
use cos::{CosContext, CosCredentials};
use download::{download_file, download_signed_file, head_content_length_retry};
use expand::expand_tokens;
use reqwest::blocking::Client;
use crate::pipeline::command::{
    ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

/// Pipeline command: bulk download files (template URLs or COS prefix).
pub struct FetchBulkdlOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(FetchBulkdlOp)
}

/// Which enumeration strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Expand `${token}` placeholders against integer ranges.
    Template,
    /// List a bucket prefix on an S3-compatible endpoint with SigV4.
    Cos,
}

impl Mode {
    fn parse(s: &str) -> Result<Mode, String> {
        match s {
            "template" => Ok(Mode::Template),
            "cos" => Ok(Mode::Cos),
            other => Err(format!(
                "unknown mode '{}', expected 'template' or 'cos'", other
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Mode::Template => "template",
            Mode::Cos => "cos",
        }
    }

    /// Options that belong to this mode (other modes' options must not be
    /// set, to keep configurations unambiguous).
    fn owned_options(self) -> &'static [&'static str] {
        match self {
            Mode::Template => &["baseurl", "tokens"],
            Mode::Cos => &["endpoint", "bucket", "prefix", "region"],
        }
    }
}

/// Per-file download job. Variants carry the mode-specific data needed by a
/// worker to issue a single GET request without referring back to global
/// configuration.
enum FetchJob {
    /// Plain HTTP download. `expected_size` comes from a HEAD probe; it may
    /// be absent if the server did not answer HEAD.
    Http {
        url: String,
        filename: String,
        expected_size: Option<u64>,
    },
    /// COS download. `key` is the absolute object key in the bucket;
    /// `filename` is the local relative path under the output directory
    /// (typically `key` with the prefix stripped). `expected_size` comes from
    /// the `ListObjectsV2` response and is always known.
    Cos {
        key: String,
        filename: String,
        expected_size: u64,
        ctx: Arc<CosContext>,
        client: Arc<Client>,
    },
}

impl FetchJob {
    fn filename(&self) -> &str {
        match self {
            FetchJob::Http { filename, .. } | FetchJob::Cos { filename, .. } => filename,
        }
    }

    fn expected_size(&self) -> Option<u64> {
        match self {
            FetchJob::Http { expected_size, .. } => *expected_size,
            FetchJob::Cos { expected_size, .. } => Some(*expected_size),
        }
    }

    /// Display name for log lines (URL for HTTP, key for COS).
    fn display(&self) -> &str {
        match self {
            FetchJob::Http { url, .. } => url,
            FetchJob::Cos { key, .. } => key,
        }
    }

    /// Verify the local file's size matches the expected size, if known.
    /// In COS mode the size comes from the listing (no extra HTTP call).
    /// In HTTP mode this re-issues a HEAD with retries.
    fn check_existing_size(&self, local_len: u64) -> Result<bool, String> {
        match self {
            FetchJob::Http { url, .. } => {
                let remote = head_content_length_retry(url, 10)?;
                Ok(remote == local_len)
            }
            FetchJob::Cos { expected_size, .. } => Ok(*expected_size == local_len),
        }
    }

    fn perform(&self, dest: &Path, on_bytes: &dyn Fn(u64)) -> Result<(), String> {
        // Ensure parent directory exists (COS keys may contain subpaths).
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!("failed to create directory {}: {}", parent.display(), e)
                })?;
            }
        }

        match self {
            FetchJob::Http { url, .. } => download_file(url, dest, on_bytes),
            FetchJob::Cos { key, expected_size, ctx, client, .. } => {
                // Sign at the moment of the GET — SigV4 signatures are valid
                // for 15 minutes from x-amz-date.
                let (url, headers) = ctx.signed_get(key);
                download_signed_file(client, &url, headers, dest, *expected_size, on_bytes)
            }
        }
    }
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
            summary: "Bulk download files (template URLs or S3-compatible bucket prefix)".into(),
            body: format!(
                r#"# download bulk

Bulk download files into a local directory using one of two enumeration
strategies, selected by the required `mode` option.

## Modes

### `mode: template`

Expands `${{token}}` placeholders in `baseurl` against integer ranges and
downloads each expanded URL via plain HTTPS. Use for datasets published as
numbered shards.

Required options: `baseurl`. Optional: `tokens` (defaults to no expansion).

### `mode: cos`

Enumerates every object under `bucket`/`prefix` on an S3-compatible endpoint
(IBM Cloud Object Storage in particular) via `ListObjectsV2`, signing every
request with AWS Signature Version 4. Credentials are read from the standard
AWS environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_SESSION_TOKEN`), so the same `auth.sh` files used with the AWS CLI work
unchanged.

Required options: `endpoint`, `bucket`. Optional: `prefix` (default empty),
`region` (default `us-south`).

## Common Behavior

Both modes share the same downloader internals: a status file
(`.down-rs-status.json`) tracks which files have already been downloaded and
enables resume across runs; pending files are downloaded in batches with
configurable concurrency; each download supports retries on failure; and a
size check verifies any pre-existing local file before re-downloading.

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

        // 1. Parse mode and validate that only the right options are set.
        let mode = match options.require("mode").and_then(Mode::parse) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };
        if let Err(e) = validate_mode_options(mode, options) {
            return error_result(e, start);
        }

        let output_dir = match options.require("output") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };

        let tries: u32 = options.parse_or("tries", 3u32).unwrap_or(3);
        let concurrency: usize = options.parse_or("concurrency", 4usize).unwrap_or(4);

        // 2. Ensure output directory exists before any other I/O.
        if let Err(e) = fs::create_dir_all(&output_dir) {
            return error_result(
                format!("failed to create directory {}: {}", output_dir.display(), e),
                start,
            );
        }

        // 3. Enumerate the candidate files according to the mode.
        let candidates = match mode {
            Mode::Template => match enumerate_template(options) {
                Ok(c) => c,
                Err(e) => return error_result(e, start),
            },
            Mode::Cos => match enumerate_cos(options, ctx) {
                Ok(c) => c,
                Err(e) => return error_result(e, start),
            },
        };

        let total = candidates.len();
        let status_path = output_dir.join(".down-rs-status.json");
        let status = Arc::new(Mutex::new(StatusFile::load(&status_path)));

        // 4. Pre-verify pass: skip files that the status file already
        //    records as complete, or whose local size already matches the
        //    expected size. The check method is mode-specific: HEAD for HTTP,
        //    direct size compare for COS (size came from the listing).
        let mut pending: Vec<FetchJob> = Vec::new();
        let mut pending_bytes: u64 = 0;
        let mut skipped_status = 0usize;
        let mut skipped_size = 0usize;
        let verify_pb = ctx.ui.bar_with_unit(total as u64, "verifying", "files");
        {
            let mut st = status.lock().unwrap();
            for (idx, job) in candidates.into_iter().enumerate() {
                verify_pb.set_position(idx as u64 + 1);
                let filename = job.filename().to_string();
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
                            match job.check_existing_size(local_len) {
                                Ok(true) => {
                                    st.completed.push(filename);
                                    skipped_size += 1;
                                    continue;
                                }
                                Ok(false) => {
                                    ctx.ui.log(&format!(
                                        "size mismatch {}: local={}, re-downloading", filename, local_len
                                    ));
                                    if let Some(sz) = job.expected_size() {
                                        pending_bytes += sz;
                                    }
                                }
                                Err(e) => {
                                    ctx.ui.log(&format!(
                                        "WARNING: could not verify {}: {}", filename, e
                                    ));
                                }
                            }
                        }
                    }
                } else if let Some(sz) = job.expected_size() {
                    pending_bytes += sz;
                }
                pending.push(job);
            }
            if skipped_size > 0 {
                let _ = st.save(&status_path);
            }
        }
        verify_pb.finish();

        let already_done = skipped_status + skipped_size;
        ctx.ui.log(&format!(
            "bulkdl ({}): {} total, {} from status, {} size-verified, {} to download",
            mode.name(), total, skipped_status, skipped_size, pending.len()
        ));

        if pending.is_empty() {
            return CommandResult {
                status: Status::Ok,
                message: format!("all {} files already downloaded", total),
                produced: vec![output_dir],
                elapsed: start.elapsed(),
            };
        }

        // 5. Download pending jobs through a fixed-size worker pool.
        let total_pending = pending.len();
        let pb = if pending_bytes > 0 {
            ctx.ui.bar_with_unit(pending_bytes, "downloading", "bytes")
        } else {
            ctx.ui.bar_with_unit(total_pending as u64, "downloading", "files")
        };
        let byte_bar = pending_bytes > 0;
        let pb_id = pb.id();

        // Shared atomic counters drive the bar's message line. Workers bump
        // `started` when they begin a download; the result loop bumps
        // `completed`/`failed`. A single dedicated ticker thread reads these
        // atomics every 250ms and emits one UI message — no per-byte or
        // per-file message updates, so the work to maintain the counter
        // display is bounded regardless of throughput or file size.
        let started_files = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let completed_files = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let failed_files = Arc::new(std::sync::atomic::AtomicU32::new(0));
        update_progress_message(
            &ctx.ui, pb_id,
            &started_files, &completed_files, &failed_files,
            total_pending,
        );

        let ticker_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ticker = {
            let stop = Arc::clone(&ticker_stop);
            let started = Arc::clone(&started_files);
            let completed = Arc::clone(&completed_files);
            let failed = Arc::clone(&failed_files);
            let ui = ctx.ui.clone();
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    update_progress_message(
                        &ui, pb_id, &started, &completed, &failed, total_pending,
                    );
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            })
        };

        let (tx, rx) = std::sync::mpsc::sync_channel::<FetchJob>(concurrency * 2);
        let rx = Arc::new(Mutex::new(rx));
        let (result_tx, result_rx) = std::sync::mpsc::channel::<DownloadResult>();

        let mut workers = Vec::with_capacity(concurrency);
        for worker_id in 0..concurrency {
            let rx = Arc::clone(&rx);
            let result_tx = result_tx.clone();
            let output_dir = output_dir.clone();
            let ui = ctx.ui.clone();
            let w_pb_id = pb_id;
            let w_byte_bar = byte_bar;
            let w_started = Arc::clone(&started_files);

            workers.push(std::thread::spawn(move || {
                loop {
                    let job = match rx.lock().unwrap().recv() {
                        Ok(item) => item,
                        Err(_) => break, // channel closed, all work done
                    };

                    let filename = job.filename().to_string();
                    let display = job.display().to_string();
                    let dest = output_dir.join(&filename);

                    // Re-verify in worker too (covers the race where the file
                    // was finished by another process between pre-verify and
                    // the worker pickup).
                    if dest.exists() {
                        if let Ok(meta) = fs::metadata(&dest) {
                            if meta.len() > 0 {
                                if let Ok(true) = job.check_existing_size(meta.len()) {
                                    ui.log(&format!(
                                        "[w{}] skip {} ({})",
                                        worker_id, filename, format_size(meta.len())
                                    ));
                                    let _ = result_tx.send(DownloadResult::Ok(filename, 0));
                                    continue;
                                }
                            }
                        }
                    }

                    // Mark this file as in-flight. The ticker thread picks
                    // up the change on its next 250ms tick.
                    w_started.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    let mut last_err = String::new();
                    let mut succeeded = false;
                    for attempt in 1..=tries {
                        if attempt == 1 {
                            ui.log(&format!("[w{}] {}", worker_id, display));
                        } else {
                            ui.log(&format!(
                                "[w{}] {} (retry, tries {}/{})",
                                worker_id, display, attempt, tries
                            ));
                        }
                        let t0 = Instant::now();
                        let prev_reported = std::sync::atomic::AtomicU64::new(0);
                        let ui_ref = &ui;
                        let on_bytes = |total_written: u64| {
                            if w_byte_bar {
                                let prev = prev_reported.load(std::sync::atomic::Ordering::Relaxed);
                                let delta = total_written.saturating_sub(prev);
                                if delta > 0 {
                                    ui_ref.inc_by_id(w_pb_id, delta);
                                    prev_reported.store(
                                        total_written, std::sync::atomic::Ordering::Relaxed
                                    );
                                }
                            }
                        };
                        match job.perform(&dest, &on_bytes) {
                            Ok(()) => {
                                let file_size = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                                let secs = t0.elapsed().as_secs_f64();
                                let mbps = if secs > 0.0 {
                                    file_size as f64 / 1_048_576.0 / secs
                                } else {
                                    0.0
                                };
                                ui.log(&format!(
                                    "[w{}] done {} ({} @ {:.1} MiB/s)",
                                    worker_id, filename, format_size(file_size), mbps
                                ));
                                let _ = result_tx.send(DownloadResult::Ok(filename.clone(), file_size));
                                succeeded = true;
                                break;
                            }
                            Err(e) => {
                                last_err = e.clone();
                                ui.log(&format!(
                                    "[w{}] FAIL {}/{} {}: {}",
                                    worker_id, attempt, tries, filename, e
                                ));
                            }
                        }
                    }
                    if !succeeded {
                        ui.log(&format!(
                            "[w{}] FAILED {} after {} attempts: {}",
                            worker_id, filename, tries, last_err
                        ));
                        let _ = result_tx.send(DownloadResult::Failed);
                    }
                }
            }));
        }
        drop(result_tx);

        let feeder = std::thread::spawn(move || {
            for job in pending {
                if tx.send(job).is_err() {
                    break;
                }
            }
        });

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
                    completed_files.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if !byte_bar {
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
                    failed_files.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if !byte_bar {
                        ctx.ui.inc_by_id(pb_id, 1);
                    }
                }
            }
        }

        // Result channel closed → all work is done. Stop the ticker, then
        // emit one final message so the bar shows the terminal counts.
        ticker_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = ticker.join();
        update_progress_message(
            &ctx.ui, pb_id,
            &started_files, &completed_files, &failed_files, total_pending,
        );

        {
            let st = status.lock().unwrap();
            let _ = st.save(&status_path);
        }

        let _ = feeder.join();
        for w in workers {
            let _ = w.join();
        }

        pb.finish();

        let status_result = if failed_total > 0 { Status::Error } else { Status::Ok };
        let total_elapsed = bandwidth_start.elapsed().as_secs_f64().max(0.001);
        let avg_mbps = (downloaded_bytes as f64 / 1_048_576.0) / total_elapsed;

        CommandResult {
            status: status_result,
            message: format!(
                "{} downloaded ({:.1} MiB, {:.1} MiB/s), {} skipped, {} failed (of {} total)",
                downloaded_total,
                downloaded_bytes as f64 / 1_048_576.0,
                avg_mbps,
                already_done,
                failed_total,
                total
            ),
            produced: vec![output_dir],
            elapsed: start.elapsed(),
        }
    }

    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        if !output.exists() {
            return ArtifactState::Absent;
        }
        if !output.is_dir() {
            return crate::pipeline::bound::check_artifact_default(output, options);
        }

        let status_path = output.join(".down-rs-status.json");
        if !status_path.exists() {
            return ArtifactState::PartialResumable;
        }

        // For template mode we can compute the expected file count locally
        // from the token ranges. For COS mode the only authoritative count
        // requires a network listing, so we conservatively report
        // PartialResumable and let execute() short-circuit when the status
        // file already covers everything.
        let mode = options
            .get("mode")
            .and_then(|s| Mode::parse(s).ok())
            .unwrap_or(Mode::Template);
        match mode {
            Mode::Cos => ArtifactState::PartialResumable,
            Mode::Template => {
                let baseurl = match options.get("baseurl") {
                    Some(u) => u.to_string(),
                    None => return ArtifactState::Complete,
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
                let expected_count = expand_tokens(&baseurl, &token_map).len();
                let completed_count = StatusFile::load(&status_path).completed.len();
                if completed_count >= expected_count {
                    ArtifactState::Complete
                } else {
                    ArtifactState::PartialResumable
                }
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "mode".to_string(),
                type_name: "enum".to_string(),
                required: true,
                default: None,
                description: "Enumeration mode: 'template' or 'cos'".to_string(),
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
            // template mode
            OptionDesc {
                name: "baseurl".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "[template] URL template with ${token} placeholders".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "tokens".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "[template] Token ranges, e.g. \"number=[0..409]\"".to_string(),
                role: OptionRole::Config,
            },
            // cos mode
            OptionDesc {
                name: "endpoint".to_string(),
                type_name: "URL".to_string(),
                required: false,
                default: None,
                description: "[cos] S3-compatible endpoint URL (e.g. https://s3.us-south.cloud-object-storage.appdomain.cloud)".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "bucket".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "[cos] Bucket name".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "prefix".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some(String::new()),
                description: "[cos] Object key prefix to enumerate".to_string(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "region".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("us-south".to_string()),
                description: "[cos] SigV4 region (default: us-south)".to_string(),
                role: OptionRole::Config,
            },
            // common
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

/// Reject options that belong to a different mode than the one selected.
/// This keeps configuration files unambiguous: a `mode: cos` step that also
/// has `baseurl` set is almost always a copy-paste mistake.
fn validate_mode_options(mode: Mode, options: &Options) -> Result<(), String> {
    let owned: std::collections::HashSet<&str> = mode.owned_options().iter().copied().collect();
    let other_mode = match mode {
        Mode::Template => Mode::Cos,
        Mode::Cos => Mode::Template,
    };
    for foreign in other_mode.owned_options() {
        if options.has(foreign) && !owned.contains(foreign) {
            return Err(format!(
                "option '{}' belongs to mode '{}', not '{}'",
                foreign, other_mode.name(), mode.name(),
            ));
        }
    }
    Ok(())
}

/// Parse template-mode options and expand `${token}` placeholders into the
/// list of (URL, filename) pairs, then probe each URL's `Content-Length` so
/// the byte-level progress bar has a denominator.
fn enumerate_template(options: &Options) -> Result<Vec<FetchJob>, String> {
    let baseurl = options.require("baseurl")?.to_string();

    let mut token_map = std::collections::HashMap::new();
    if let Some(tokens_str) = options.get("tokens") {
        for part in tokens_str.split(',') {
            let part = part.trim();
            if let Some(eq_pos) = part.find('=') {
                let name = part[..eq_pos].trim().to_string();
                let spec_str = part[eq_pos + 1..].trim();
                match parse_token_spec(spec_str) {
                    Some(spec) => { token_map.insert(name, spec); }
                    None => {
                        return Err(format!(
                            "invalid token spec '{}', expected format like [0..409]", spec_str
                        ));
                    }
                }
            } else {
                return Err(format!(
                    "invalid token entry '{}', expected name=[start..end]", part
                ));
            }
        }
    }

    let pairs = expand_tokens(&baseurl, &token_map);
    Ok(pairs
        .into_iter()
        .map(|(url, filename)| FetchJob::Http {
            url,
            filename,
            // expected_size is filled in lazily below via HEAD probes.
            expected_size: None,
        })
        .collect())
}

/// Parse cos-mode options, build the [`CosContext`], and list every object
/// under `prefix`. The local filename for each key is the key with `prefix`
/// stripped — preserving any subdirectory structure beneath the prefix.
fn enumerate_cos(options: &Options, ctx: &mut StreamContext) -> Result<Vec<FetchJob>, String> {
    let endpoint = options.require("endpoint")?.to_string();
    let bucket = options.require("bucket")?.to_string();
    let prefix = options.get("prefix").unwrap_or("").to_string();
    let region = options.get("region").unwrap_or("us-south").to_string();

    let credentials = CosCredentials::from_env()?;
    let cos_ctx = Arc::new(CosContext::new(endpoint.clone(), bucket.clone(), region, credentials)?);
    let client = Arc::new(Client::new());

    ctx.ui.log(&format!(
        "bulkdl (cos): listing s3://{}/{} via {}", bucket, prefix, endpoint
    ));
    let listed = cos_ctx.list_prefix(&client, &prefix)?;
    ctx.ui.log(&format!("bulkdl (cos): found {} object(s) under prefix", listed.len()));

    let mut jobs = Vec::with_capacity(listed.len());
    for (key, size) in listed {
        // Drop directory-marker entries (zero-byte keys ending in '/').
        if key.ends_with('/') && size == 0 {
            continue;
        }
        let filename = strip_prefix(&key, &prefix);
        if filename.is_empty() {
            // Object key is exactly the prefix — skip.
            continue;
        }
        jobs.push(FetchJob::Cos {
            key,
            filename,
            expected_size: size,
            ctx: Arc::clone(&cos_ctx),
            client: Arc::clone(&client),
        });
    }
    Ok(jobs)
}

/// Strip `prefix` from `key`. If `key` does not start with `prefix` (e.g. the
/// listing returned an unrelated key — shouldn't happen for ListObjectsV2 but
/// is harmless to handle), fall back to the full key.
fn strip_prefix(key: &str, prefix: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else if let Some(rest) = key.strip_prefix(prefix) {
        rest.to_string()
    } else {
        key.to_string()
    }
}

/// Update the bar's message line with file-count progress: `done`, currently
/// `active` (started but not yet finished), and `total`. The `active` count
/// is what makes the message informative at startup and during long
/// downloads — without it, the file count would appear frozen at `0 done`
/// while every worker is busy on a multi-GB file.
///
/// Called only at file-boundary transitions (worker pickup, completion,
/// failure), not in the per-byte callback — the message text only changes
/// at those transitions, so per-byte updates would just allocate format
/// strings and queue UI events for no visible effect.
fn update_progress_message(
    ui: &veks_core::ui::UiHandle,
    pb_id: veks_core::ui::ProgressId,
    started: &std::sync::atomic::AtomicU32,
    completed: &std::sync::atomic::AtomicU32,
    failed: &std::sync::atomic::AtomicU32,
    total: usize,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let done = completed.load(Relaxed);
    let fail = failed.load(Relaxed);
    let in_flight = started
        .load(Relaxed)
        .saturating_sub(done)
        .saturating_sub(fail);
    let msg = if fail > 0 {
        format!(
            "{} done + {} active / {} files ({} failed)",
            done, in_flight, total, fail
        )
    } else {
        format!("{} done + {} active / {} files", done, in_flight, total)
    };
    ui.set_message_by_id(pb_id, msg);
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
    fn describe_options_advertises_mode_required() {
        let op = FetchBulkdlOp;
        let opts = op.describe_options();
        assert!(opts.iter().any(|o| o.name == "mode" && o.required));
        assert!(opts.iter().any(|o| o.name == "output" && o.required));
        assert!(opts.iter().any(|o| o.name == "baseurl" && !o.required));
        assert!(opts.iter().any(|o| o.name == "endpoint" && !o.required));
        assert!(opts.iter().any(|o| o.name == "bucket" && !o.required));
    }

    #[test]
    fn command_path_unchanged() {
        let op = FetchBulkdlOp;
        assert_eq!(op.command_path(), "download bulk");
    }

    #[test]
    fn mode_parse_accepts_template_and_cos() {
        assert_eq!(Mode::parse("template").unwrap(), Mode::Template);
        assert_eq!(Mode::parse("cos").unwrap(), Mode::Cos);
        assert!(Mode::parse("s3").is_err());
        assert!(Mode::parse("").is_err());
    }

    #[test]
    fn validate_mode_options_rejects_cross_mode_options() {
        let mut opts = Options::new();
        opts.set("mode", "cos");
        opts.set("baseurl", "https://example.com");
        let err = validate_mode_options(Mode::Cos, &opts).unwrap_err();
        assert!(err.contains("baseurl"));
        assert!(err.contains("template"));

        let mut opts = Options::new();
        opts.set("mode", "template");
        opts.set("bucket", "b");
        let err = validate_mode_options(Mode::Template, &opts).unwrap_err();
        assert!(err.contains("bucket"));
        assert!(err.contains("cos"));
    }

    #[test]
    fn validate_mode_options_accepts_clean_configs() {
        let mut opts = Options::new();
        opts.set("mode", "cos");
        opts.set("endpoint", "https://example.com");
        opts.set("bucket", "b");
        opts.set("prefix", "p/");
        assert!(validate_mode_options(Mode::Cos, &opts).is_ok());

        let mut opts = Options::new();
        opts.set("mode", "template");
        opts.set("baseurl", "https://example.com/${i}");
        opts.set("tokens", "i=[0..3]");
        assert!(validate_mode_options(Mode::Template, &opts).is_ok());
    }

    #[test]
    fn strip_prefix_trims_subpaths() {
        assert_eq!(
            strip_prefix("bulk-v0.8-1B/file1.bin", "bulk-v0.8-1B/"),
            "file1.bin"
        );
        assert_eq!(
            strip_prefix("bulk-v0.8-1B/sub/file2.bin", "bulk-v0.8-1B/"),
            "sub/file2.bin"
        );
        assert_eq!(strip_prefix("file.bin", ""), "file.bin");
    }
}
