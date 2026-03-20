// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dataset publishing to S3 (`veks publish`).
//!
//! Synchronizes the publishable content of a local dataset directory to
//! a remote destination using the transport specified by the `.publish_url`
//! file. Currently supports S3 (`s3://`) via the AWS CLI.

mod transport;

use std::path::{Path, PathBuf};

use clap::Args;

use transport::{SyncOptions, transport_for_scheme};

/// Arguments for `veks publish`.
#[derive(Args)]
pub struct PublishArgs {
    /// Target directory to publish (default: current directory)
    #[arg(default_value = ".")]
    pub directory: PathBuf,

    /// Skip confirmation prompt
    #[arg(short = 'y')]
    pub yes: bool,

    /// Show what would be uploaded without transferring
    #[arg(long)]
    pub dry_run: bool,

    /// Remove remote objects that no longer exist locally
    #[arg(long)]
    pub delete: bool,

    /// Number of parallel upload streams
    #[arg(long, default_value = "4")]
    pub concurrency: u32,

    /// Additional glob patterns to exclude
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Additional glob patterns to force-include (overrides excludes)
    #[arg(long)]
    pub include: Vec<String>,

    /// Skip based on size only, ignoring timestamps
    #[arg(long)]
    pub size_only: bool,

    /// AWS profile name for credentials
    #[arg(long)]
    pub profile: Option<String>,

    /// Custom S3 endpoint (for S3-compatible stores)
    #[arg(long)]
    pub endpoint_url: Option<String>,

    /// Skip pre-flight checks
    #[arg(long)]
    pub no_check: bool,
}

/// Default exclusion patterns for publishing.
///
/// All hidden files and directories (dot-prefixed) are categorically
/// excluded — they are local workspace state that must never appear
/// in the published dataset. The `.*` and `*/.*` patterns handle
/// top-level and nested hidden entries respectively.
const DEFAULT_EXCLUDES: &[&str] = &[
    ".*",
    "*/.*",
    "*.tmp",
    "*.partial",
    "__pycache__/*",
    "*.pyc",
];

/// Entry point for `veks publish`.
pub fn run(args: PublishArgs) {
    let directory = resolve_directory(&args.directory);

    if !directory.is_dir() {
        eprintln!("Error: '{}' is not a directory", directory.display());
        std::process::exit(2);
    }

    // Locate .publish_url file
    let publish_file = crate::check::publish_url::find_publish_file(&directory);
    let publish_file = match publish_file {
        Some(p) => p,
        None => {
            eprintln!("Error: no .publish_url file found in '{}' or parent directories", directory.display());
            eprintln!();
            eprintln!("Create one with:");
            eprintln!("  echo 's3://bucket-name/prefix/' > {}/.publish_url", directory.display());
            std::process::exit(2);
        }
    };

    let content = std::fs::read_to_string(&publish_file).unwrap_or_else(|e| {
        eprintln!("Error: failed to read {}: {}", publish_file.display(), e);
        std::process::exit(2);
    });

    let parsed = match crate::check::publish_url::parse_publish_url(&content) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: invalid .publish_url at {}: {}", publish_file.display(), e);
            std::process::exit(2);
        }
    };
    let s3_url = parsed.url;

    // Run pre-flight checks unless --no-check
    if !args.no_check && !args.dry_run {
        let check_ok = run_preflight_checks(&directory);
        if !check_ok {
            eprintln!();
            eprintln!("Pre-flight checks failed. Run 'veks check' for details,");
            eprintln!("or use '--no-check' to override.");
            std::process::exit(1);
        }
    }

    // Enumerate publishable files for the summary
    let publishable = enumerate_publishable_files(&directory);
    let total_size: u64 = publishable.iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();

    // Present summary and get confirmation
    println!("Publish summary:");
    println!("  Source:      {}", directory.display());
    println!("  Destination: {}", s3_url);
    println!("  Files:       {} to sync, {} total",
        publishable.len(),
        format_size(total_size),
    );
    if args.delete {
        println!("  Delete:      enabled (remote-only objects will be removed)");
    } else {
        println!("  Delete:      disabled");
    }
    println!("  Excludes:    {}", DEFAULT_EXCLUDES.join(", "));
    if !args.exclude.is_empty() {
        println!("  Extra excl:  {}", args.exclude.join(", "));
    }
    println!();

    if args.dry_run {
        println!("(dry run — no changes will be made)");
        println!();
    } else if !args.yes {
        // Interactive confirmation
        if !atty::is(atty::Stream::Stdin) {
            eprintln!("Error: stdin is not a TTY. Use -y to skip confirmation.");
            std::process::exit(3);
        }

        eprint!("Proceed? Type YES to confirm: ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap_or(0);
        if input.trim().to_uppercase() != "YES" {
            eprintln!("Aborted.");
            std::process::exit(3);
        }
        println!();
    }

    // Select transport based on URL scheme
    let transport = match transport_for_scheme(&parsed.scheme) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(2);
        }
    };

    let sync_opts = SyncOptions {
        dry_run: args.dry_run,
        delete: args.delete,
        size_only: args.size_only,
        exclude: &args.exclude,
        include: &args.include,
        profile: args.profile.as_deref(),
        endpoint_url: args.endpoint_url.as_deref(),
        default_excludes: DEFAULT_EXCLUDES,
    };

    let exit_code = transport.sync(&directory, &s3_url, &sync_opts);
    std::process::exit(exit_code);
}

/// Run all pre-flight checks before publishing.
///
/// Runs the same checks as `veks check --check-all`: pipelines, publish,
/// merkle, integrity, and catalogs. Returns `true` only if every check passes.
fn run_preflight_checks(directory: &Path) -> bool {
    let dataset_files = crate::check::discover_datasets(directory);
    let publishable = crate::publish::enumerate_publishable_files(directory);
    let merkle_threshold = 100_000_000u64; // 100 MB default

    let checks: Vec<crate::check::CheckResult> = vec![
        crate::check::pipelines::check(directory, &dataset_files),
        crate::check::pipelines::check_coverage(directory, &dataset_files, &publishable, merkle_threshold),
        crate::check::publish_url::check(directory, &dataset_files),
        crate::check::merkle::check(directory, &publishable, merkle_threshold),
        crate::check::integrity::check(directory, &publishable),
        crate::check::catalogs::check(directory, &dataset_files),
    ];

    let mut all_ok = true;
    for result in &checks {
        if !result.passed {
            eprintln!("Pre-flight check failed: {}", result.name);
            for msg in &result.messages {
                eprintln!("  {}", msg);
            }
            all_ok = false;
        }
    }

    all_ok
}

/// Enumerate all publishable files under a directory (applying default exclusions).
pub fn enumerate_publishable_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    enumerate_recursive(root, root, &mut files);
    files.sort();
    files
}

fn enumerate_recursive(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            // All hidden directories are local state — never publish
            if name_str.starts_with('.') || name_str == "__pycache__" {
                continue;
            }
            enumerate_recursive(root, &path, files);
        } else {
            // All hidden files are local state — never publish
            if name_str.starts_with('.')
                || name_str.ends_with(".tmp")
                || name_str.ends_with(".partial")
                || name_str.ends_with(".pyc")
            {
                continue;
            }
            files.push(path);
        }
    }
}

fn resolve_directory(dir: &Path) -> PathBuf {
    if dir.is_absolute() {
        dir.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(dir)
    }
}

/// Format a byte count as a human-readable string.
fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
