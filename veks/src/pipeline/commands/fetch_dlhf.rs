// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: download files from HuggingFace Hub.
//!
//! Given a HuggingFace repository identifier (e.g. `username/dataset-name`),
//! lists and downloads matching files into a local output directory using the
//! HuggingFace HTTP API.
//!
//! Equivalent to the Java `CMD_fetch_dlhf` / `bulkdl` command.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status,
    StreamContext, render_options_table,
};

/// Pipeline command: download from HuggingFace Hub.
pub struct FetchDlhfOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(FetchDlhfOp)
}

impl CommandOp for FetchDlhfOp {
    fn command_path(&self) -> &str {
        "fetch dlhf"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Download files from HuggingFace Hub".into(),
            body: format!(
                r#"# fetch dlhf

Download files from HuggingFace Hub.

## Description

Given a HuggingFace repository identifier (e.g. `username/dataset-name`),
lists and downloads matching files into a local output directory using
the HuggingFace HTTP API. Supports glob filtering, revision/branch
selection, repository type (dataset or model), and automatic resume of
previously completed downloads.

## How It Works

The command first queries the HuggingFace tree API
(`https://huggingface.co/api/<type>s/<repo>/tree/<revision>`) to
enumerate all files in the repository. The response is filtered by
the glob pattern to select only matching files. For each matching
file, the command checks whether a local file of the same name and
size already exists; if so, the download is skipped. Otherwise, the
file is downloaded from the HuggingFace resolve endpoint. If the
`HF_TOKEN` environment variable is set, it is included as a Bearer
token for authenticated access to private or gated repositories.

## Data Preparation Role

`fetch dlhf` is the primary way to acquire source datasets from
HuggingFace Hub at the beginning of a dataset preparation pipeline.
It downloads parquet files, vector files, metadata, and any other
artifacts published in a HuggingFace dataset repository. The resume
capability ensures that interrupted downloads can be continued without
re-downloading completed files, which is critical for large datasets
that may take hours to transfer. Downloaded files then feed into
subsequent pipeline steps like parquet-to-JSONL conversion, slab
import, and vector file assembly.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "iothreads".into(), description: "Concurrent download connections".into(), adjustable: false },
            ResourceDesc { name: "mem".into(), description: "Download buffers".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let repo = match options.require("repo") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };

        let output_dir = options
            .get("output")
            .map(|s| resolve_path(s, &ctx.workspace))
            .unwrap_or_else(|| ctx.workspace.join(repo.replace('/', "_")));

        let pattern = options.get("pattern").unwrap_or("*");
        let revision = options.get("revision").unwrap_or("main");
        let repo_type = options.get("type").unwrap_or("dataset");

        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            return error_result(format!("failed to create output dir: {}", e), start);
        }

        ctx.ui.log(&format!("Fetching file list from HuggingFace: {}", repo));

        // List files via the HuggingFace API
        let api_url = format!(
            "https://huggingface.co/api/{}s/{}/tree/{}",
            repo_type, repo, revision
        );

        let file_list = match list_hf_files(&api_url) {
            Ok(files) => files,
            Err(e) => return error_result(format!("failed to list files: {}", e), start),
        };

        // Filter files by pattern
        let matching: Vec<&HfFileEntry> = file_list
            .iter()
            .filter(|f| f.file_type == "file" && glob_match(pattern, &f.rfilename))
            .collect();

        ctx.ui.log(&format!(
            "Found {} file(s) matching '{}' (of {} total)",
            matching.len(),
            pattern,
            file_list.len()
        ));

        if matching.is_empty() {
            return CommandResult {
                status: Status::Warning,
                message: "no matching files found".to_string(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let mut downloaded = 0u32;
        let mut skipped = 0u32;
        let mut failed = 0u32;

        for entry in &matching {
            let dest = output_dir.join(&entry.rfilename);

            // Skip if already exists with matching size
            if dest.exists() {
                if let Ok(meta) = std::fs::metadata(&dest) {
                    if meta.len() == entry.size {
                        ctx.ui.log(&format!(
                            "  {} — already downloaded ({} bytes)",
                            entry.rfilename, entry.size
                        ));
                        skipped += 1;
                        continue;
                    }
                }
            }

            let download_url = format!(
                "https://huggingface.co/{}s/{}/resolve/{}/{}",
                repo_type, repo, revision, entry.rfilename
            );

            ctx.ui.log(&format!(
                "  {} — downloading ({} bytes)...",
                entry.rfilename, entry.size
            ));

            match download_file(&download_url, &dest) {
                Ok(size) => {
                    ctx.ui.log(&format!("  {} — done ({} bytes)", entry.rfilename, size));
                    downloaded += 1;
                }
                Err(e) => {
                    ctx.ui.log(&format!("  {} — FAILED: {}", entry.rfilename, e));
                    failed += 1;
                }
            }
        }

        ctx.ui.log("");
        ctx.ui.log(&format!(
            "HuggingFace download: {} downloaded, {} skipped, {} failed",
            downloaded, skipped, failed
        ));

        let status = if failed > 0 {
            Status::Error
        } else {
            Status::Ok
        };

        CommandResult {
            status,
            message: format!(
                "{} downloaded, {} skipped, {} failed",
                downloaded, skipped, failed
            ),
            produced: vec![output_dir],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "repo".to_string(),
                type_name: "String".to_string(),
                required: true,
                default: None,
                description: "HuggingFace repository (e.g. user/dataset-name)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output directory for downloaded files".to_string(),
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "pattern".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("*".to_string()),
                description: "Glob pattern for file matching".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "revision".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("main".to_string()),
                description: "Branch or revision to download from".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "type".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("dataset".to_string()),
                description: "Repository type: dataset or model".to_string(),
                role: OptionRole::Config,
        },
        ]
    }
}

#[derive(Debug)]
struct HfFileEntry {
    rfilename: String,
    size: u64,
    file_type: String,
}

/// List files in a HuggingFace repo via the API.
fn list_hf_files(api_url: &str) -> Result<Vec<HfFileEntry>, String> {
    let mut response_data = Vec::new();

    let mut easy = curl::easy::Easy::new();
    easy.url(api_url)
        .map_err(|e| format!("invalid URL: {}", e))?;
    easy.follow_location(true).ok();
    easy.fail_on_error(true).ok();
    easy.useragent("vecs-rs/1.0").ok();

    // Check for HF_TOKEN env var for authenticated access
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let mut headers = curl::easy::List::new();
        headers
            .append(&format!("Authorization: Bearer {}", token))
            .ok();
        easy.http_headers(headers).ok();
    }

    {
        let mut transfer = easy.transfer();
        transfer
            .write_function(|data| {
                response_data.extend_from_slice(data);
                Ok(data.len())
            })
            .map_err(|e| format!("transfer setup error: {}", e))?;
        transfer
            .perform()
            .map_err(|e| format!("API request failed: {}", e))?;
    }

    let json_str =
        String::from_utf8(response_data).map_err(|e| format!("invalid UTF-8: {}", e))?;

    // Parse the JSON array of file entries
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&json_str).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut files = Vec::new();
    for entry in entries {
        let rfilename = entry
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let size = entry
            .get("size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let file_type = entry
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("file")
            .to_string();

        if !rfilename.is_empty() {
            files.push(HfFileEntry {
                rfilename,
                size,
                file_type,
            });
        }
    }

    Ok(files)
}

/// Download a URL to a local file.
fn download_file(url: &str, dest: &Path) -> Result<u64, String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create dir: {}", e))?;
    }

    let mut file =
        std::fs::File::create(dest).map_err(|e| format!("failed to create {}: {}", dest.display(), e))?;

    let mut easy = curl::easy::Easy::new();
    easy.url(url).map_err(|e| format!("invalid URL: {}", e))?;
    easy.follow_location(true).ok();
    easy.fail_on_error(true).ok();
    easy.useragent("vecs-rs/1.0").ok();

    if let Ok(token) = std::env::var("HF_TOKEN") {
        let mut headers = curl::easy::List::new();
        headers
            .append(&format!("Authorization: Bearer {}", token))
            .ok();
        easy.http_headers(headers).ok();
    }

    let mut transfer = easy.transfer();
    transfer
        .write_function(|data| {
            file.write_all(data).map_or(Ok(0), |()| Ok(data.len()))
        })
        .map_err(|e| format!("transfer setup error: {}", e))?;
    transfer
        .perform()
        .map_err(|e| format!("download failed: {}", e))?;
    drop(transfer);

    let size = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    Ok(size)
}

/// Simple glob pattern matching supporting `*` and `?` wildcards.
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    glob_match_inner(&pat, &txt, 0, 0)
}

fn glob_match_inner(pattern: &[char], text: &[char], mut pi: usize, mut ti: usize) -> bool {
    let mut star_pi = usize::MAX;
    let mut star_ti = usize::MAX;

    while ti < text.len() {
        if pi < pattern.len() && (pattern[pi] == '?' || pattern[pi] == text[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == '*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == '*' {
        pi += 1;
    }

    pi == pattern.len()
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
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
    fn test_glob_match() {
        assert!(glob_match("*", "anything.txt"));
        assert!(glob_match("*.fvec", "base.fvec"));
        assert!(!glob_match("*.fvec", "base.ivec"));
        assert!(glob_match("base.*", "base.fvec"));
        assert!(glob_match("*.f???", "data.fvec"));
        assert!(glob_match("data*", "data_vectors.fvec"));
    }

    #[test]
    fn test_describe_options() {
        let op = FetchDlhfOp;
        let opts = op.describe_options();
        assert!(opts.iter().any(|o| o.name == "repo" && o.required));
        assert!(opts.iter().any(|o| o.name == "pattern" && !o.required));
    }
}
