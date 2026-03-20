// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Publish transport abstraction.
//!
//! Each supported URL scheme (`s3://`, etc.) has a corresponding transport
//! implementation that knows how to synchronize a local directory to the
//! remote destination.

use std::path::Path;
use std::process::Command;

/// Options passed through to the transport layer.
pub struct SyncOptions<'a> {
    pub dry_run: bool,
    pub delete: bool,
    pub size_only: bool,
    pub exclude: &'a [String],
    pub include: &'a [String],
    pub profile: Option<&'a str>,
    pub endpoint_url: Option<&'a str>,
    pub default_excludes: &'a [&'a str],
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

        for pattern in opts.default_excludes {
            cmd.arg("--exclude").arg(pattern);
        }
        for pattern in opts.exclude {
            cmd.arg("--exclude").arg(pattern);
        }
        for pattern in opts.include {
            cmd.arg("--include").arg(pattern);
        }

        if opts.delete { cmd.arg("--delete"); }
        if opts.dry_run { cmd.arg("--dryrun"); }
        if opts.size_only { cmd.arg("--size-only"); }

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
            Ok(status) => {
                if status.success() {
                    println!();
                    println!("Publish complete.");
                    0
                } else {
                    let code = status.code().unwrap_or(1);
                    eprintln!();
                    eprintln!("aws s3 sync exited with code {}", code);
                    1
                }
            }
            Err(e) => {
                eprintln!("Error: failed to execute 'aws' CLI: {}", e);
                eprintln!("Ensure the AWS CLI is installed and on your PATH.");
                2
            }
        }
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
