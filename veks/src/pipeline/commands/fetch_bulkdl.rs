// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: bulk download files from a URL template.
//!
//! Downloads files by expanding token placeholders in a URL template and
//! fetching each expanded URL into a local output directory. Supports
//! concurrency, retries, and resume of partial downloads.
//!
//! Wraps the same logic as the standalone `veks bulkdl` CLI subcommand.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use log::{error, info, warn};

use crate::bulkdl::config::{StatusFile, parse_token_spec};
use crate::bulkdl::download::{download_file, head_content_length};
use crate::bulkdl::expand::expand_tokens;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: bulk download files from a URL template.
pub struct FetchBulkdlOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(FetchBulkdlOp)
}

impl CommandOp for FetchBulkdlOp {
    fn command_path(&self) -> &str {
        "fetch bulkdl"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Bulk download files from a URL template".into(),
            body: format!(
                r#"# fetch bulkdl

Bulk download files from a URL template.

## Description

Downloads files by expanding token placeholders in a URL template and
fetching each expanded URL into a local output directory. Supports
concurrency control, retries, and resume of partial downloads.

Token placeholders use `${{name}}` syntax in the URL. Each token has a
range specification like `[0..409]` (inclusive). Multiple tokens produce
the cartesian product of all ranges.

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

        let pb = ctx.ui.bar(total as u64, "downloading");
        pb.set_position(already_done as u64);
        let pb_id = pb.id();

        // Download files with concurrency via batched threads
        let mut downloaded_total = 0u32;
        let mut failed_total = 0u32;

        for batch in pending.chunks(concurrency) {
            let handles: Vec<_> = batch
                .iter()
                .map(|(url, filename)| {
                    let url = url.clone();
                    let filename = filename.clone();
                    let status = Arc::clone(&status);
                    let status_path = status_path.clone();
                    let output_dir = output_dir.clone();
                    let ui = ctx.ui.clone();

                    std::thread::spawn(move || {
                        let dest = output_dir.join(&filename);

                        // Check if file already exists with correct size via HEAD
                        if dest.exists() {
                            if let Some(remote_len) = head_content_length(&url) {
                                if let Ok(meta) = fs::metadata(&dest) {
                                    if meta.len() == remote_len {
                                        info!("Skipping {} (size matches)", filename);
                                        let mut st = status.lock().unwrap();
                                        st.completed.push(filename.clone());
                                        let _ = st.save(&status_path);
                                        ui.inc_by_id(pb_id, 1);
                                        return (true, false);
                                    }
                                }
                            }
                        }

                        // Download with retries
                        let mut last_err = String::new();
                        for attempt in 1..=tries {
                            info!("Downloading {} (attempt {}/{})", url, attempt, tries);
                            match download_file(&url, &dest) {
                                Ok(()) => {
                                    info!("Completed: {}", filename);
                                    let mut st = status.lock().unwrap();
                                    st.completed.push(filename.clone());
                                    let _ = st.save(&status_path);
                                    ui.inc_by_id(pb_id, 1);
                                    return (true, false);
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
                        error!("All {} attempts failed for {}: {}", tries, url, last_err);
                        ui.inc_by_id(pb_id, 1);
                        (false, true)
                    })
                })
                .collect();

            for handle in handles {
                match handle.join() {
                    Ok((true, _)) => downloaded_total += 1,
                    Ok((false, true)) => failed_total += 1,
                    Ok(_) => {}
                    Err(e) => {
                        error!("Download thread panicked: {:?}", e);
                        failed_total += 1;
                    }
                }
            }
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

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "baseurl".to_string(),
                type_name: "String".to_string(),
                required: true,
                default: None,
                description: "URL template with ${token} placeholders".to_string(),
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output directory for downloaded files".to_string(),
            },
            OptionDesc {
                name: "tokens".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Token ranges, e.g. \"number=[0..409]\" or \"x=[0..9],y=[0..9]\"".to_string(),
            },
            OptionDesc {
                name: "tries".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("3".to_string()),
                description: "Number of download attempts per file".to_string(),
            },
            OptionDesc {
                name: "concurrency".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("4".to_string()),
                description: "Maximum concurrent downloads".to_string(),
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
        assert_eq!(op.command_path(), "fetch bulkdl");
    }
}
