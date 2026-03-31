// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Publish transport abstraction.
//!
//! Each supported URL scheme (`s3://`, etc.) has a corresponding transport
//! implementation that knows how to synchronize a local directory to the
//! remote destination.
//!
//! The sync uses an **include-only** strategy: exclude everything by default,
//! then explicitly include only the files that `enumerate_publishable_files`
//! identified. This ensures the sync matches the enumeration exactly —
//! one code path decides what's publishable.

use std::path::Path;
use std::process::Command;

/// Options passed through to the transport layer.
pub struct SyncOptions<'a> {
    pub dry_run: bool,
    pub delete: bool,
    pub size_only: bool,
    /// Relative paths of files to include (from `enumerate_publishable_files`).
    pub include_files: &'a [String],
    pub profile: Option<&'a str>,
    pub endpoint_url: Option<&'a str>,
}

/// A publish transport that can sync a local directory to a remote URL.
pub trait PublishTransport {
    /// Synchronize `directory` to `url`, returning a process exit code.
    fn sync(&self, directory: &Path, url: &str, opts: &SyncOptions) -> i32;
}

/// S3 transport via the AWS CLI (`aws s3 sync`).
pub struct S3Transport;

impl PublishTransport for S3Transport {
    fn sync(&self, directory: &Path, url: &str, opts: &SyncOptions) -> i32 {
        let mut cmd = Command::new("aws");
        cmd.arg("s3").arg("sync");
        cmd.arg(directory.to_string_lossy().as_ref());
        cmd.arg(url);

        // Exclude everything by default, then include only publishable files.
        // AWS S3 sync applies include/exclude filters to scope which files
        // are synced. --delete is NOT used here because the filters also
        // scope deletion — remote-only files outside the include set would
        // be invisible to --delete and never removed.
        cmd.arg("--exclude").arg("*");
        for path in opts.include_files {
            cmd.arg("--include").arg(path);
        }
        if opts.dry_run { cmd.arg("--dryrun"); }
        if opts.size_only { cmd.arg("--size-only"); }

        // Log the command summary for debugging
        eprintln!("  aws s3 sync {} {} ({} include patterns, exclude=*{}{})",
            directory.display(), url,
            opts.include_files.len(),
            if opts.dry_run { ", dryrun" } else { "" },
            if opts.delete { ", delete" } else { "" },
        );
        // Show first/last few includes for verification
        if !opts.include_files.is_empty() {
            for p in opts.include_files.iter().take(3) {
                eprintln!("    --include {}", p);
            }
            if opts.include_files.len() > 6 {
                eprintln!("    ... ({} more)", opts.include_files.len() - 6);
            }
            for p in opts.include_files.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev() {
                eprintln!("    --include {}", p);
            }
        }

        if let Some(profile) = opts.profile {
            cmd.arg("--profile").arg(profile);
        }
        if let Some(endpoint) = opts.endpoint_url {
            cmd.arg("--endpoint-url").arg(endpoint);
        }

        cmd.stdin(std::process::Stdio::inherit());
        cmd.stdout(std::process::Stdio::inherit());
        cmd.stderr(std::process::Stdio::inherit());

        match cmd.status() {
            Ok(status) if !status.success() => {
                let code = status.code().unwrap_or(1);
                eprintln!();
                eprintln!("aws s3 sync exited with code {}", code);
                return 1;
            }
            Err(e) => {
                eprintln!("Error: failed to execute 'aws' CLI: {}", e);
                eprintln!("Ensure the AWS CLI is installed and on your PATH.");
                return 2;
            }
            _ => {}
        }

        // Force-copy infrastructure files (dataset.yaml, catalog.json, etc.)
        // that aws s3 sync may skip when the remote copy has the same size.
        // These are small, frequently updated files that must always reflect
        // the latest local state.
        for path in opts.include_files {
            let name = path.rsplit('/').next().unwrap_or(path);
            if veks_core::filters::is_infrastructure_file(name) {
                let src = directory.join(path);
                let dst = format!("{}/{}", url.trim_end_matches('/'), path);
                if opts.dry_run {
                    eprintln!("(dryrun) force-copy: {} -> {}", path, dst);
                } else {
                    let mut cp = Command::new("aws");
                    cp.arg("s3").arg("cp");
                    cp.arg(src.to_string_lossy().as_ref());
                    cp.arg(&dst);
                    if let Some(profile) = opts.profile {
                        cp.arg("--profile").arg(profile);
                    }
                    if let Some(endpoint) = opts.endpoint_url {
                        cp.arg("--endpoint-url").arg(endpoint);
                    }
                    cp.stdout(std::process::Stdio::inherit());
                    cp.stderr(std::process::Stdio::inherit());
                    let _ = cp.status();
                }
            }
        }

        // Phase 2: delete remote files not in the include set.
        // aws s3 sync --delete with include filters can't see remote-only
        // files outside the filter, so we list remote keys and explicitly
        // delete any not in the publishable set.
        if opts.delete {
            let include_set: std::collections::HashSet<&str> = opts.include_files
                .iter()
                .map(|s| s.as_str())
                .collect();

            // Ensure URL has trailing slash for prefix listing
            let list_url = if url.ends_with('/') {
                url.to_string()
            } else {
                format!("{}/", url)
            };

            let mut list_cmd = Command::new("aws");
            list_cmd.arg("s3").arg("ls").arg(&list_url).arg("--recursive");
            if let Some(profile) = opts.profile {
                list_cmd.arg("--profile").arg(profile);
            }
            if let Some(endpoint) = opts.endpoint_url {
                list_cmd.arg("--endpoint-url").arg(endpoint);
            }
            list_cmd.stderr(std::process::Stdio::inherit());

            let list_output = match list_cmd.output() {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("Error listing remote keys: {}", e);
                    return 1;
                }
            };

            let stdout = String::from_utf8_lossy(&list_output.stdout);
            let mut to_delete: Vec<String> = Vec::new();
            for line in stdout.lines() {
                // aws s3 ls --recursive format: "2024-01-01 12:00:00  12345 path/to/file"
                let parts: Vec<&str> = line.splitn(4, char::is_whitespace).collect();
                if parts.len() >= 4 {
                    let key = parts[3].trim();
                    if !key.is_empty() && !include_set.contains(key) {
                        to_delete.push(key.to_string());
                    }
                }
            }

            if !to_delete.is_empty() {
                eprintln!("{} remote object(s) to delete", to_delete.len());
                for key in &to_delete {
                    let full_key = format!("{}/{}", url.trim_end_matches('/'), key);
                    if opts.dry_run {
                        println!("(dryrun) delete: {}", key);
                    } else {
                        let mut rm_cmd = Command::new("aws");
                        rm_cmd.arg("s3").arg("rm").arg(&full_key);
                        if let Some(profile) = opts.profile {
                            rm_cmd.arg("--profile").arg(profile);
                        }
                        if let Some(endpoint) = opts.endpoint_url {
                            rm_cmd.arg("--endpoint-url").arg(endpoint);
                        }
                        rm_cmd.stdout(std::process::Stdio::inherit());
                        rm_cmd.stderr(std::process::Stdio::inherit());
                        let _ = rm_cmd.status();
                        println!("  deleted: {}", key);
                    }
                }
            }
        }

        println!();
        println!("Publish complete: {} -> {}",
            crate::check::rel_display(directory), url);
        0
    }
}

/// Select the appropriate transport for a URL scheme.
pub fn transport_for_scheme(scheme: &str) -> Result<Box<dyn PublishTransport>, String> {
    match scheme {
        "s3" => Ok(Box::new(S3Transport)),
        other => Err(format!(
            "no transport implementation for scheme '{}' (supported: s3)",
            other,
        )),
    }
}
