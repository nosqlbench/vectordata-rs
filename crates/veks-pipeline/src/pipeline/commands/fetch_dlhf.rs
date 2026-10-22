// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: download files from HuggingFace Hub.
//!
//! Given a HuggingFace repository identifier (e.g. `username/dataset-name`),
//! lists and downloads matching files into a local output directory using the
//! HuggingFace HTTP API.
//!
//! Equivalent to the Java `CMD_fetch_dlhf` / `bulkdl` command.
//!
//! Requests go through the process's transport
//! (`vectordata::transport::shared`): its client pool, TLS trust, offline
//! gate and user agent. Their authorisation is the command's own — the
//! `HF_TOKEN` bearer, or none — and never vectordata's read credentials,
//! which belong to dataset servers, not to Hugging Face
//! (`docs/design/srd-crate-boundaries.md` TR-9).

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::transport::{Request, Transport};

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status,
    StreamContext, render_options_table,
};

/// Pipeline command: download from HuggingFace Hub.
pub struct FetchDlhfOp;

/// Creates a boxed [`FetchDlhfOp`]; registered as the `download huggingface` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(FetchDlhfOp)
}

impl CommandOp for FetchDlhfOp {
    fn command_path(&self) -> &str {
        "download huggingface"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_DOWNLOAD
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Download files from HuggingFace Hub".into(),
            body: format!(
                r#"# download huggingface

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

        let transport = vectordata::transport::shared();
        let token = hf_token(std::env::var("HF_TOKEN").ok());
        let file_list = match list_hf_files(transport, &api_url, token.as_deref()) {
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
            if dest.exists()
                && let Ok(meta) = std::fs::metadata(&dest)
                    && meta.len() == entry.size {
                        ctx.ui.log(&format!(
                            "  {} — already downloaded ({} bytes)",
                            entry.rfilename, entry.size
                        ));
                        skipped += 1;
                        continue;
                    }

            let download_url = format!(
                "https://huggingface.co/{}s/{}/resolve/{}/{}",
                repo_type, repo, revision, entry.rfilename
            );

            ctx.ui.log(&format!(
                "  {} — downloading ({} bytes)...",
                entry.rfilename, entry.size
            ));

            match download_file(transport, &download_url, &dest, token.as_deref()) {
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
                extended_description: None,
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Output directory for downloaded files".to_string(),
                extended_description: None,
                role: OptionRole::Output,
        },
            OptionDesc {
                name: "pattern".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("*".to_string()),
                description: "Glob pattern for file matching".to_string(),
                extended_description: None,
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "revision".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("main".to_string()),
                description: "Branch or revision to download from".to_string(),
                extended_description: None,
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "type".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("dataset".to_string()),
                description: "Repository type: dataset or model".to_string(),
                extended_description: None,
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

/// The Hugging Face token from the `HF_TOKEN` value: `None` when unset
/// or blank, so a blank variable sends no `Authorization` at all.
fn hf_token(env_value: Option<String>) -> Option<String> {
    env_value.map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

/// List files in a HuggingFace repo via the API, authorised by `token`
/// alone (anonymous without one).
fn list_hf_files(transport: &Transport, api_url: &str, token: Option<&str>) -> Result<Vec<HfFileEntry>, String> {
    let response = transport
        .send(Request::get(api_url).bearer_or_anonymous(token))
        .map_err(|e| format!("API request failed: {}", e))?;

    if !response.is_success() {
        return Err(format!("HTTP {} from {}", response.status(), api_url));
    }

    let json_str = response.text()
        .map_err(|e| format!("invalid response: {}", e))?;

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

/// Download a URL to a local file, streaming the body to disk,
/// authorised by `token` alone (anonymous without one).
fn download_file(transport: &Transport, url: &str, dest: &Path, token: Option<&str>) -> Result<u64, String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create dir: {}", e))?;
    }

    let mut file =
        std::fs::File::create(dest).map_err(|e| format!("failed to create {}: {}", dest.display(), e))?;

    let response = transport
        .send(Request::get(url).bearer_or_anonymous(token))
        .map_err(|e| format!("download failed: {}", e))?;

    if !response.is_success() {
        return Err(format!("HTTP {} from {}", response.status(), url));
    }

    response.copy_to(&mut file)
        .map_err(|e| format!("write error: {}", e))?;

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

    /// Read credentials that would authorise every URL: what
    /// `$VECTORDATA_TOKEN` is to the process's transport.
    struct Everywhere;

    impl vectordata::transport::Credentials for Everywhere {
        fn read_token(&self, _url: &str) -> Option<String> {
            Some("vectordata-secret".to_string())
        }
    }

    /// A Hugging Face stand-in: the tree API lists `a.bin` (3 bytes) and
    /// the resolve endpoint serves it. Returns its base URL and the
    /// `Authorization` header of every request it got (`None` when absent).
    fn hub() -> (String, std::sync::Arc<std::sync::Mutex<Vec<Option<String>>>>) {
        use std::io::{BufRead, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = std::sync::Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                let _ = reader.read_line(&mut line);
                let mut auth = None;
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).is_err() || h.trim().is_empty() {
                        break;
                    }
                    if let Some((k, v)) = h.split_once(':')
                        && k.eq_ignore_ascii_case("authorization")
                    {
                        auth = Some(v.trim().to_string());
                    }
                }
                log.lock().unwrap().push(auth);
                let body = if line.contains("/api/") { r#"[{"path":"a.bin","size":3,"type":"file"}]"# } else { "abc" };
                let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        (base, seen)
    }

    /// **A Hugging Face request carries `HF_TOKEN` or nothing — never a
    /// vectordata read token**, even from a transport whose credentials
    /// would authorise every URL (TR-9).
    #[test]
    fn hub_requests_carry_only_their_own_token() {
        let transport = Transport::with_credentials(
            vectordata::transport::TransportConfig::default(),
            std::sync::Arc::new(Everywhere),
        );
        let dir = tempfile::tempdir().unwrap();
        for token in [Some("hf-token"), None] {
            let (base, seen) = hub();
            let files = list_hf_files(&transport, &format!("{base}/api/datasets/r/tree/main"), token).unwrap();
            assert_eq!(files.len(), 1);
            let dest = dir.path().join("a.bin");
            assert_eq!(download_file(&transport, &format!("{base}/datasets/r/resolve/main/a.bin"), &dest, token).unwrap(), 3);
            let expected = token.map(|t| format!("Bearer {t}"));
            assert_eq!(*seen.lock().unwrap(), vec![expected.clone(), expected], "token {token:?}");
        }
    }

    /// Offline mode refuses both requests before the network.
    #[test]
    fn offline_mode_refuses_the_hub() {
        let offline = Transport::new(vectordata::transport::TransportConfig {
            offline: true,
            ..Default::default()
        });
        let (base, seen) = hub();
        let listed = list_hf_files(&offline, &format!("{base}/api/datasets/r/tree/main"), None).unwrap_err();
        assert!(listed.contains("offline mode is on"), "{listed}");
        let dir = tempfile::tempdir().unwrap();
        let fetched = download_file(&offline, &format!("{base}/datasets/r/resolve/main/a.bin"), &dir.path().join("a"), None)
            .unwrap_err();
        assert!(fetched.contains("offline mode is on"), "{fetched}");
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn a_blank_hf_token_sends_none() {
        assert_eq!(hf_token(None), None);
        assert_eq!(hf_token(Some("  ".into())), None);
        assert_eq!(hf_token(Some(" t ".into())).as_deref(), Some("t"));
    }

    #[test]
    fn test_describe_options() {
        let op = FetchDlhfOp;
        let opts = op.describe_options();
        assert!(opts.iter().any(|o| o.name == "repo" && o.required));
        assert!(opts.iter().any(|o| o.name == "pattern" && !o.required));
    }
}
