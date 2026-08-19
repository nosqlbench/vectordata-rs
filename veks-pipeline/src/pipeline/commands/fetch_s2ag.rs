// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: download S2AG Datasets API bulk files.
//!
//! The [Semantic Scholar Datasets API](https://api.semanticscholar.org/api-docs/datasets)
//! serves dataset releases as lists of *signed, expiring* bulk-file URLs,
//! negotiated per request with an `x-api-key` header. That flow doesn't fit
//! `download bulk`: per-file signed URLs are inexpressible in template mode,
//! signed URLs sign the GET method (so template mode's HEAD re-verify 403s),
//! and no header injection exists. This command owns the whole negotiation:
//!
//! 1. `GET {api-base}/release/{release}/dataset/{dataset}` (with `x-api-key`
//!    from the `S2_API_KEY` environment variable when set) to obtain the
//!    file list — re-negotiated on every run, so URL expiry only matters
//!    within a single run.
//! 2. Deterministic file selection over URL basenames in lexical order
//!    (`files: first:N` or a glob) — the selection rule is a step option and
//!    therefore a provenance axis.
//! 3. Signed GETs (query-string auth; no HEAD probe) with per-file retries,
//!    a worker pool, and status-file resume (`.s2ag-status.json`).
//!
//! The API key itself is never an option value: option values land in
//! provenance sidecars and `dataset.log`, and secrets must not. The key is
//! resolved from the `api-key-file` option (a file *reference* — only the
//! path can enter provenance) or, failing that, from `S2_API_KEY`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use super::fetch_bulkdl::config::StatusFile;
use super::fetch_bulkdl::download::download_file;
use super::fetch_bulkdl::expand::url_to_filename;
use crate::pipeline::command::{
    ArtifactManifest, ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, ResourceDesc, Status, StreamContext, render_options_table,
};

/// Pipeline command: download S2AG Datasets API bulk files.
pub struct FetchS2agOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(FetchS2agOp)
}

/// Default API base; overridable (e.g. for tests) via the `api-base` option.
const DEFAULT_API_BASE: &str = "https://api.semanticscholar.org/datasets/v1";

/// Name of the resume-status file written into the output directory.
const STATUS_FILENAME: &str = ".s2ag-status.json";

/// Environment variable carrying the Semantic Scholar API key.
const API_KEY_ENV: &str = "S2_API_KEY";

/// Deterministic file selection over lexically-sorted URL basenames.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FileSelector {
    /// The first N files.
    FirstN(usize),
    /// Basenames matching a glob (`all` ≡ `*`).
    Glob(String),
}

impl FileSelector {
    fn parse(s: &str) -> Result<Self, String> {
        if let Some(n) = s.strip_prefix("first:") {
            return match n.parse::<usize>() {
                Ok(n) if n > 0 => Ok(Self::FirstN(n)),
                _ => Err(format!("invalid files selector '{}': first:N needs N > 0", s)),
            };
        }
        if s == "all" {
            return Ok(Self::Glob("*".to_string()));
        }
        if s.is_empty() {
            return Err("empty files selector".to_string());
        }
        Ok(Self::Glob(s.to_string()))
    }

    /// The number of files this selector will pick, when knowable offline.
    fn expected_count(&self) -> Option<usize> {
        match self {
            Self::FirstN(n) => Some(*n),
            Self::Glob(_) => None,
        }
    }
}

/// Resolve the API key from an environment value (pure; unit-testable
/// without process-env mutation).
fn resolve_api_key(env_value: Option<String>) -> Option<String> {
    env_value.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// Parse the contents of an `api-key-file`: either a raw single-line key,
/// or a small YAML-style map carrying an `S2-API-KEY:` entry (hyphen or
/// underscore form, case-insensitive). Pure; error messages never echo the
/// file contents.
fn parse_key_file(contents: &str) -> Result<String, String> {
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            if k.eq_ignore_ascii_case("S2-API-KEY") || k.eq_ignore_ascii_case("S2_API_KEY") {
                return validate_key(v.trim());
            }
        }
    }
    let lines: Vec<&str> = contents
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if let [only] = lines.as_slice()
        && !only.contains(':')
    {
        return validate_key(only);
    }
    Err(
        "api-key-file must hold a single-line key or a YAML map with an 'S2-API-KEY' entry"
            .to_string(),
    )
}

/// Shape-check a candidate key without ever echoing it.
fn validate_key(key: &str) -> Result<String, String> {
    if key.is_empty() {
        return Err("api-key-file entry is empty".to_string());
    }
    if key.chars().any(|c| c.is_whitespace()) {
        return Err("api key contains whitespace — wrong file referenced?".to_string());
    }
    Ok(key.to_string())
}

/// Validate the `release` option: an explicit, dated release id is required
/// so the recorded provenance identifies the actual data. `latest` would
/// resolve to a different release over time while the recorded option value
/// stayed constant, silently breaking reproducibility.
fn validate_release(release: &str) -> Result<(), String> {
    if release.is_empty() {
        return Err("release must not be empty".to_string());
    }
    if release.eq_ignore_ascii_case("latest") {
        return Err(
            "release 'latest' is not allowed: pin an explicit release id (e.g. 2026-08-12) \
             so provenance identifies the data actually downloaded"
                .to_string(),
        );
    }
    Ok(())
}

impl CommandOp for FetchS2agOp {
    fn command_path(&self) -> &str {
        "download s2ag"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_DOWNLOAD
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Download S2AG Datasets API bulk files".into(),
            body: format!(
                r#"# download s2ag

Download S2AG Datasets API bulk files.

## Description

Negotiates the file list for a pinned Semantic Scholar dataset release
(`{base}/release/<release>/dataset/<dataset>`) and downloads the selected
bulk files into the output directory. The API returns signed, expiring
URLs; they are re-negotiated on every run, and files are fetched with the
signed GET exactly as issued (query-string auth, no HEAD probe). Files are
stored under their URL basename with the signature query stripped, exactly
as served (typically gzipped JSONL) — no decompression is performed.

## Authentication

The key is sent as the `x-api-key` header on the file-list request. It is
resolved from `api-key-file` when set (a single-line file, or a YAML map
with an `S2-API-KEY` entry), otherwise from the `S2_API_KEY` environment
variable. Keys are free (api.semanticscholar.org); without one the API
refuses the file-list request. Either way the key *contents* never land in
provenance records, `dataset.log`, or error messages — at most the
key-file path does.

## Determinism and resume

`release` must be an explicit release id (`latest` is rejected — provenance
must identify the data actually downloaded). File selection is applied to
URL basenames in lexical order: `files: first:N` takes the first N,
`files: <glob>` matches basenames, `files: all` takes everything. Completed
files are recorded in `{status}` in the output directory and skipped on
subsequent runs.

## Options

{opts}"#,
                base = DEFAULT_API_BASE,
                status = STATUS_FILENAME,
                opts = render_options_table(&options)
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
                name: "mem".into(),
                description: "Download buffers".into(),
                adjustable: false,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let release = match options.require("release") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        if let Err(e) = validate_release(&release) {
            return error_result(e, start);
        }
        let dataset = options.get("dataset-name").unwrap_or("s2orc").to_string();
        let selector = match FileSelector::parse(options.get("files").unwrap_or("first:1")) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_dir = match options.require("output") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let tries: u32 = match options.parse_or("tries", 3) {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };
        let concurrency: usize = match options.parse_or("concurrency", 4) {
            Ok(n) if n > 0 => n,
            Ok(_) => return error_result("concurrency must be > 0".to_string(), start),
            Err(e) => return error_result(e, start),
        };
        let api_base = options
            .get("api-base")
            .unwrap_or(DEFAULT_API_BASE)
            .trim_end_matches('/')
            .to_string();

        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            return error_result(format!("failed to create output dir: {}", e), start);
        }

        // Key precedence: explicit api-key-file option, else the environment.
        // Only the file *path* can enter provenance; the key contents never
        // appear in options, logs, or error messages.
        let (api_key, key_source) = match options.get("api-key-file") {
            Some(p) => {
                let path = resolve_path(p, &ctx.workspace);
                let contents = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(e) => {
                        return error_result(
                            format!("failed to read api-key-file {}: {}", path.display(), e),
                            start,
                        );
                    }
                };
                match parse_key_file(&contents) {
                    Ok(key) => (Some(key), "file"),
                    Err(e) => {
                        return error_result(
                            format!("api-key-file {}: {}", path.display(), e),
                            start,
                        );
                    }
                }
            }
            None => match resolve_api_key(std::env::var(API_KEY_ENV).ok()) {
                Some(key) => (Some(key), "env"),
                None => (None, "unset"),
            },
        };
        let list_url = format!("{}/release/{}/dataset/{}", api_base, release, dataset);
        ctx.ui.log(&format!(
            "negotiating file list: {} (x-api-key: {})",
            list_url, key_source
        ));

        let urls = match fetch_file_list(&list_url, api_key.as_deref(), tries) {
            Ok(u) => u,
            Err(e) => return error_result(e, start),
        };
        let selected = select_files(&urls, &selector);
        ctx.ui.log(&format!(
            "selected {} of {} file(s) with selector '{}'",
            selected.len(),
            urls.len(),
            options.get("files").unwrap_or("first:1")
        ));
        if selected.is_empty() {
            return CommandResult {
                status: Status::Warning,
                message: format!("no files matched selector for {}/{}", release, dataset),
                produced: vec![output_dir],
                elapsed: start.elapsed(),
            };
        }

        // Resume: skip files the status file already records as complete.
        let status_path = output_dir.join(STATUS_FILENAME);
        let status = Mutex::new(StatusFile::load(&status_path));
        let mut jobs: VecDeque<(String, String)> = VecDeque::new();
        let mut skipped = 0u32;
        {
            let status = status.lock().expect("status lock");
            for (filename, url) in &selected {
                if status.completed.contains(filename) {
                    skipped += 1;
                } else {
                    jobs.push_back((filename.clone(), url.clone()));
                }
            }
        }
        let total_jobs = jobs.len();
        let pb = ctx.ui.bar_with_unit(total_jobs as u64, "download", "files");

        let queue = Mutex::new(jobs);
        let failures = Mutex::new(Vec::<String>::new());
        let done = std::sync::atomic::AtomicU64::new(0);
        let workers = concurrency.min(total_jobs.max(1));
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let job = queue.lock().expect("queue lock").pop_front();
                        let Some((filename, url)) = job else { break };
                        let dest = output_dir.join(&filename);
                        let mut last_err = String::new();
                        let mut ok = false;
                        for _attempt in 0..tries.max(1) {
                            match download_file(&url, &dest, &|_| {}) {
                                Ok(()) => {
                                    ok = true;
                                    break;
                                }
                                Err(e) => last_err = e,
                            }
                        }
                        if ok {
                            let mut status = status.lock().expect("status lock");
                            if !status.completed.contains(&filename) {
                                status.completed.push(filename);
                            }
                            let _ = status.save(&status_path);
                        } else {
                            failures
                                .lock()
                                .expect("failures lock")
                                .push(format!("{}: {}", filename, last_err));
                        }
                        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        pb.set_position(n);
                    }
                });
            }
        });
        pb.finish();

        let failures = failures.into_inner().expect("failures lock");
        for f in &failures {
            ctx.ui.log(&format!("  FAILED {}", f));
        }
        let downloaded = total_jobs - failures.len();
        let message = format!(
            "{} downloaded, {} skipped (already complete), {} failed for {}/{}",
            downloaded,
            skipped,
            failures.len(),
            release,
            dataset
        );
        ctx.ui.log(&message);

        CommandResult {
            status: if failures.is_empty() { Status::Ok } else { Status::Error },
            message,
            produced: vec![output_dir],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "release".to_string(),
                type_name: "String".to_string(),
                required: true,
                default: None,
                description: "Explicit release id (e.g. 2026-08-12); 'latest' is rejected"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "dataset-name".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("s2orc".to_string()),
                description: "S2AG dataset name (e.g. s2orc, papers, abstracts, tldrs)".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "files".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("first:1".to_string()),
                description: "Selection over lexically-sorted basenames: first:N, a glob, or all"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output directory for downloaded shard files".to_string(),
                extended_description: None,
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "tries".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("3".to_string()),
                description: "Download attempts per file".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "concurrency".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("4".to_string()),
                description: "Parallel download workers".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "api-key-file".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "File holding the API key (single line, or YAML with an S2-API-KEY entry); overrides S2_API_KEY"
                    .to_string(),
                extended_description: Some(
                    "Only the file path can enter provenance records and dataset.log; \
                     the key contents never do. Rotating the key in place does not \
                     invalidate downloaded artifacts."
                        .to_string(),
                ),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "api-base".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some(DEFAULT_API_BASE.to_string()),
                description: "Datasets API base URL".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }

    /// Offline freshness: with a `first:N` selector, the status file tells
    /// whether all N files completed; glob selectors need the API to know
    /// the expected set, so they always resume.
    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        if !output.exists() {
            return ArtifactState::Absent;
        }
        let status = StatusFile::load(&output.join(STATUS_FILENAME));
        let selector = match FileSelector::parse(options.get("files").unwrap_or("first:1")) {
            Ok(s) => s,
            Err(_) => return ArtifactState::PartialResumable,
        };
        match selector.expected_count() {
            Some(n) if status.completed.len() >= n => ArtifactState::Complete,
            _ => ArtifactState::PartialResumable,
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &[],
            &["output"],
        )
    }
}

/// Fetch and parse the release file list: `{"files": ["<signed url>", ...]}`.
///
/// Throttling (429) and server errors (5xx) are retried up to `tries` times
/// with exponential backoff (2s, 4s, 8s, capped at 16s) — the API meters
/// requests per second, so a fresh key or a shared-IP burst can 429
/// transiently. Auth and client errors fail fast.
fn fetch_file_list(list_url: &str, api_key: Option<&str>, tries: u32) -> Result<Vec<String>, String> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("veks/0.14")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let attempts = tries.max(1);
    let mut last_err = String::new();
    let mut body = None;
    for attempt in 0..attempts {
        if attempt > 0 {
            let backoff = (2u64 << (attempt - 1).min(3)).min(16);
            std::thread::sleep(std::time::Duration::from_secs(backoff));
        }
        let mut request = client.get(list_url);
        if let Some(key) = api_key {
            request = request.header("x-api-key", key);
        }
        let response = match request.send() {
            Ok(r) => r,
            Err(e) => {
                last_err = format!("file-list request failed: {}", e);
                continue;
            }
        };
        let status = response.status();
        if status.is_success() {
            body = Some(
                response
                    .text()
                    .map_err(|e| format!("invalid response: {}", e))?,
            );
            break;
        }
        let code = status.as_u16();
        if code == 429 || code >= 500 {
            last_err = format!("HTTP {} from {}", code, list_url);
            continue;
        }
        let hint = if (code == 401 || code == 403) && api_key.is_none() {
            format!(" (no API key was sent — set {})", API_KEY_ENV)
        } else {
            String::new()
        };
        return Err(format!("HTTP {} from {}{}", code, list_url, hint));
    }
    let Some(body) = body else {
        return Err(format!("{} (after {} attempt(s))", last_err, attempts));
    };
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("JSON parse error: {}", e))?;
    let files = json
        .get("files")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("no 'files' array in response from {}", list_url))?;
    Ok(files
        .iter()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect())
}

/// Apply the selector over URLs sorted by basename (signature query
/// stripped). Returns (basename, url) pairs in basename order.
fn select_files(urls: &[String], selector: &FileSelector) -> Vec<(String, String)> {
    let mut named: Vec<(String, String)> = urls
        .iter()
        .map(|u| (url_to_filename(u), u.clone()))
        .collect();
    named.sort();
    match selector {
        FileSelector::FirstN(n) => named.into_iter().take(*n).collect(),
        FileSelector::Glob(pattern) => named
            .into_iter()
            .filter(|(name, _)| glob_match(pattern, name))
            .collect(),
    }
}

/// Simple glob matching supporting `*` and `?` (same as `download
/// huggingface`).
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_pi, mut star_ti) = (usize::MAX, usize::MAX);
    while ti < txt.len() {
        if pi < pat.len() && (pat[pi] == '?' || pat[pi] == txt[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pat.len() && pat[pi] == '*' {
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
    while pi < pat.len() && pat[pi] == '*' {
        pi += 1;
    }
    pi == pat.len()
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
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;

    // ── pure-logic tests ─────────────────────────────────────────────────

    #[test]
    fn selector_parsing() {
        assert_eq!(FileSelector::parse("first:3").unwrap(), FileSelector::FirstN(3));
        assert_eq!(FileSelector::parse("all").unwrap(), FileSelector::Glob("*".into()));
        assert_eq!(
            FileSelector::parse("part-0*.gz").unwrap(),
            FileSelector::Glob("part-0*.gz".into())
        );
        assert!(FileSelector::parse("first:0").is_err());
        assert!(FileSelector::parse("first:x").is_err());
        assert!(FileSelector::parse("").is_err());
    }

    #[test]
    fn selection_sorts_by_basename_and_strips_signatures() {
        let urls = vec![
            "https://host/x/part-b.gz?sig=zzz".to_string(),
            "https://host/y/part-a.gz?sig=aaa".to_string(),
            "https://host/z/part-c.gz?sig=mmm".to_string(),
        ];
        let first2 = select_files(&urls, &FileSelector::FirstN(2));
        assert_eq!(
            first2.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["part-a.gz", "part-b.gz"]
        );
        assert!(first2[0].1.contains("sig=aaa"));
        let globbed = select_files(&urls, &FileSelector::Glob("part-c*".into()));
        assert_eq!(globbed.len(), 1);
        assert_eq!(globbed[0].0, "part-c.gz");
    }

    #[test]
    fn api_key_resolution_is_pure() {
        assert_eq!(resolve_api_key(None), None);
        assert_eq!(resolve_api_key(Some("".into())), None);
        assert_eq!(resolve_api_key(Some("  ".into())), None);
        assert_eq!(resolve_api_key(Some(" k123 ".into())), Some("k123".into()));
    }

    #[test]
    fn key_file_parsing_accepts_both_shapes_and_rejects_garbage() {
        // YAML-map form (the shape delivered as keys.yaml), hyphen/underscore,
        // any case, comments and blank lines tolerated.
        assert_eq!(parse_key_file("S2-API-KEY: abc123DEF\n").unwrap(), "abc123DEF");
        assert_eq!(parse_key_file("# note\n\ns2_api_key:  k9\n").unwrap(), "k9");
        // Raw single-line form.
        assert_eq!(parse_key_file("rawkey42\n").unwrap(), "rawkey42");
        assert_eq!(parse_key_file("# comment\nrawkey42\n").unwrap(), "rawkey42");
        // Rejections — and no error message may echo file contents.
        for bad in ["", "  \n", "other-key: v\nmore: w\n", "two words\n", "a\nb\n"] {
            let err = parse_key_file(bad).unwrap_err();
            assert!(!err.contains("words") && !err.contains("other-key"), "{}", err);
        }
        assert!(parse_key_file("S2-API-KEY:   \n").is_err());
    }

    #[test]
    fn latest_release_is_rejected() {
        assert!(validate_release("latest").is_err());
        assert!(validate_release("LATEST").is_err());
        assert!(validate_release("").is_err());
        assert!(validate_release("2026-08-12").is_ok());
    }

    // ── mock Datasets API server ─────────────────────────────────────────

    /// Minimal blocking HTTP server: serves the release file-list JSON at
    /// `/datasets/v1/release/<r>/dataset/<d>` (requiring `x-api-key` when
    /// configured) and shard bytes at `/files/<name>` (requiring the signed
    /// query `?sig=ok`).
    struct MockApi {
        port: u16,
        require_key: Option<String>,
        shards: Vec<(String, Vec<u8>)>,
        fail_first_gets: usize,
        /// 429 the first N file-list requests (throttle simulation).
        fail_first_lists: usize,
        gets: std::sync::atomic::AtomicUsize,
        lists: std::sync::atomic::AtomicUsize,
    }

    impl MockApi {
        fn start(
            require_key: Option<String>,
            shards: Vec<(String, Vec<u8>)>,
            fail_first_gets: usize,
            fail_first_lists: usize,
        ) -> Arc<Self> {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let api = Arc::new(MockApi {
                port,
                require_key,
                shards,
                fail_first_gets,
                fail_first_lists,
                gets: std::sync::atomic::AtomicUsize::new(0),
                lists: std::sync::atomic::AtomicUsize::new(0),
            });
            let api2 = Arc::clone(&api);
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { break };
                    let api = Arc::clone(&api2);
                    std::thread::spawn(move || api.handle(stream));
                }
            });
            api
        }

        fn base_url(&self) -> String {
            format!("http://127.0.0.1:{}/datasets/v1", self.port)
        }

        fn handle(&self, mut stream: TcpStream) {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).is_err() {
                return;
            }
            let target = request_line.split_whitespace().nth(1).unwrap_or("/").to_string();
            let mut api_key_header = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                    break;
                }
                let lower = line.to_lowercase();
                if let Some(v) = lower.strip_prefix("x-api-key:") {
                    api_key_header = Some(v.trim().to_string());
                }
            }

            let respond = |stream: &mut TcpStream, status: &str, body: &[u8]| {
                let _ = write!(
                    stream,
                    "HTTP/1.1 {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    status,
                    body.len()
                );
                let _ = stream.write_all(body);
            };

            if target.starts_with("/datasets/v1/release/") {
                let n = self.lists.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n < self.fail_first_lists {
                    respond(&mut stream, "429 Too Many Requests", b"{}");
                    return;
                }
                if let Some(required) = &self.require_key
                    && api_key_header.as_deref() != Some(required.as_str())
                {
                    respond(&mut stream, "403 Forbidden", b"{}");
                    return;
                }
                let files: Vec<String> = self
                    .shards
                    .iter()
                    .map(|(name, _)| {
                        format!("http://127.0.0.1:{}/files/{}?sig=ok", self.port, name)
                    })
                    .collect();
                let body = serde_json::json!({ "files": files }).to_string();
                respond(&mut stream, "200 OK", body.as_bytes());
            } else if let Some(rest) = target.strip_prefix("/files/") {
                let (name, query) = rest.split_once('?').unwrap_or((rest, ""));
                if query != "sig=ok" {
                    respond(&mut stream, "403 Forbidden", b"bad signature");
                    return;
                }
                let n = self.gets.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n < self.fail_first_gets {
                    respond(&mut stream, "500 Internal Server Error", b"flaky");
                    return;
                }
                match self.shards.iter().find(|(s, _)| s == name) {
                    Some((_, bytes)) => respond(&mut stream, "200 OK", bytes),
                    None => respond(&mut stream, "404 Not Found", b"absent"),
                }
            } else {
                respond(&mut stream, "404 Not Found", b"absent");
            }
        }
    }

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    fn run(dir: &Path, api: &MockApi, extra: &[(&str, &str)]) -> (CommandResult, PathBuf) {
        let output = dir.join("s2orc");
        let mut opts = Options::new();
        opts.set("release", "2026-08-12");
        opts.set("dataset-name", "s2orc");
        opts.set("output", output.to_string_lossy().to_string());
        opts.set("api-base", api.base_url());
        opts.set("concurrency", "1");
        for (k, v) in extra {
            opts.set(*k, *v);
        }
        let mut ctx = test_ctx(dir);
        (FetchS2agOp.execute(&opts, &mut ctx), output)
    }

    fn shard(name: &str, len: usize) -> (String, Vec<u8>) {
        (name.to_string(), vec![b'x'; len])
    }

    #[test]
    fn downloads_first_n_in_basename_order() {
        let tmp = tempfile::tempdir().unwrap();
        let api = MockApi::start(
            None,
            vec![shard("part-c.gz", 30), shard("part-a.gz", 10), shard("part-b.gz", 20)],
            0,
            0,
        );
        let (result, output) = run(tmp.path(), &api, &[("files", "first:2")]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        assert_eq!(std::fs::metadata(output.join("part-a.gz")).unwrap().len(), 10);
        assert_eq!(std::fs::metadata(output.join("part-b.gz")).unwrap().len(), 20);
        assert!(!output.join("part-c.gz").exists());
        // Signed query stripped from the local filename.
        assert!(!output.join("part-a.gz?sig=ok").exists());
    }

    #[test]
    fn resume_skips_completed_files() {
        let tmp = tempfile::tempdir().unwrap();
        let api = MockApi::start(None, vec![shard("part-a.gz", 10), shard("part-b.gz", 20)], 0, 0);
        let (r1, output) = run(tmp.path(), &api, &[("files", "all")]);
        assert_eq!(r1.status, Status::Ok, "{}", r1.message);
        assert!(r1.message.starts_with("2 downloaded, 0 skipped"));
        let (r2, _) = run(tmp.path(), &api, &[("files", "all")]);
        assert_eq!(r2.status, Status::Ok);
        assert!(r2.message.starts_with("0 downloaded, 2 skipped"), "{}", r2.message);
        let status = StatusFile::load(&output.join(STATUS_FILENAME));
        assert_eq!(status.completed.len(), 2);
    }

    #[test]
    fn retries_recover_from_transient_failures() {
        let tmp = tempfile::tempdir().unwrap();
        // First two GETs 500; with tries=3 the single selected file succeeds.
        let api = MockApi::start(None, vec![shard("part-a.gz", 10)], 2, 0);
        let (result, output) = run(tmp.path(), &api, &[("files", "first:1"), ("tries", "3")]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        assert_eq!(std::fs::metadata(output.join("part-a.gz")).unwrap().len(), 10);
    }

    #[test]
    fn missing_api_key_yields_hinted_error() {
        let tmp = tempfile::tempdir().unwrap();
        let api = MockApi::start(Some("sekrit".to_string()), vec![shard("part-a.gz", 10)], 0, 0);
        // The test environment must not carry a real S2_API_KEY; the hint
        // only appears when no key was sent.
        if std::env::var(API_KEY_ENV).is_ok() {
            return;
        }
        let (result, _) = run(tmp.path(), &api, &[]);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("403"), "{}", result.message);
        assert!(result.message.contains(API_KEY_ENV), "{}", result.message);
    }

    #[test]
    fn api_key_file_authenticates_file_list() {
        let tmp = tempfile::tempdir().unwrap();
        let api = MockApi::start(
            Some("sekrit-from-file".to_string()),
            vec![shard("part-a.gz", 10)],
            0,
            0,
        );
        std::fs::write(tmp.path().join("keys.yaml"), "S2-API-KEY: sekrit-from-file\n").unwrap();
        // Relative path resolves against the workspace, like every other
        // path option.
        let (result, output) = run(tmp.path(), &api, &[("api-key-file", "keys.yaml")]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        assert_eq!(std::fs::metadata(output.join("part-a.gz")).unwrap().len(), 10);

        // A dangling reference fails cleanly, naming the path only.
        let (result, _) = run(tmp.path(), &api, &[("api-key-file", "absent.yaml")]);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("absent.yaml"), "{}", result.message);
        assert!(!result.message.contains("sekrit"), "{}", result.message);
    }

    #[test]
    fn check_artifact_reports_first_n_completion_offline() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("s2orc");
        std::fs::create_dir_all(&output).unwrap();
        let mut opts = Options::new();
        opts.set("files", "first:2");
        let op = FetchS2agOp;
        assert_eq!(op.check_artifact(&output, &opts), ArtifactState::PartialResumable);
        StatusFile { completed: vec!["a.gz".into(), "b.gz".into()] }
            .save(&output.join(STATUS_FILENAME))
            .unwrap();
        assert_eq!(op.check_artifact(&output, &opts), ArtifactState::Complete);
        let mut glob_opts = Options::new();
        glob_opts.set("files", "all");
        assert_eq!(op.check_artifact(&output, &glob_opts), ArtifactState::PartialResumable);
        assert_eq!(
            op.check_artifact(&tmp.path().join("absent"), &opts),
            ArtifactState::Absent
        );
    }

    #[test]
    fn file_list_negotiation_retries_on_throttle() {
        let tmp = tempfile::tempdir().unwrap();
        // First list request 429s; with tries=3 the negotiation must recover.
        let api = MockApi::start(None, vec![shard("part-a.gz", 10)], 0, 1);
        let (result, output) = run(tmp.path(), &api, &[("files", "first:1"), ("tries", "3")]);
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        assert_eq!(std::fs::metadata(output.join("part-a.gz")).unwrap().len(), 10);
    }

    #[test]
    fn file_list_throttle_exhausts_tries_with_clear_error() {
        let tmp = tempfile::tempdir().unwrap();
        let api = MockApi::start(None, vec![shard("part-a.gz", 10)], 0, 99);
        let (result, _) = run(tmp.path(), &api, &[("files", "first:1"), ("tries", "2")]);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("429"), "{}", result.message);
        assert!(result.message.contains("attempt"), "{}", result.message);
    }

    #[test]
    fn empty_glob_selection_warns() {
        let tmp = tempfile::tempdir().unwrap();
        let api = MockApi::start(None, vec![shard("part-a.gz", 10)], 0, 0);
        let (result, _) = run(tmp.path(), &api, &[("files", "nomatch-*")]);
        assert_eq!(result.status, Status::Warning);
    }
}
