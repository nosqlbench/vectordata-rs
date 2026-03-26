// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Cache garbage collection — removes orphaned files from `.cache/`
//! that are no longer referenced by the current pipeline configuration.
//!
//! Orphaned files accumulate when:
//! - The pipeline is re-bootstrapped with different options (e.g., fraction)
//! - Cache key strategies change (file size fingerprinting)
//! - Partition sizes change between runs
//!
//! The GC identifies "live" files by:
//! 1. Reading the progress log to find recorded output files
//! 2. Scanning for files matching current cache key patterns
//! 3. Preserving the progress log itself, variables.yaml, and run.log
//! 4. Everything else in `.cache/` is considered orphaned

use std::collections::HashSet;
use std::path::Path;

/// Known file patterns that should always be preserved.
const PRESERVED_NAMES: &[&str] = &[
    ".upstream.progress.yaml",
    "run.log",
    "meta.json",
];

/// Known file prefixes for intermediate artifacts that the pipeline
/// produces and references by name (not by cache key).
const PRESERVED_PREFIXES: &[&str] = &[
    "all_vectors.",
    "sorted_ordinals.",
    "dedup_duplicates.",
    "dedup_report.",
    "zero_ordinals.",
    "clean_ordinals.",
    "shuffle.",
    "metadata_all.",
    "metadata_survey.",
    "subset_vectors.",
];

pub fn run(path: &Path, dry_run: bool) {
    let dataset_dir = if path.join("dataset.yaml").exists() {
        path.to_path_buf()
    } else if path.file_name().map(|n| n == "dataset.yaml").unwrap_or(false) {
        path.parent().unwrap_or(Path::new(".")).to_path_buf()
    } else {
        eprintln!("Error: no dataset.yaml found at {}", path.display());
        std::process::exit(1);
    };

    let cache_dir = dataset_dir.join(".cache");
    if !cache_dir.exists() {
        println!("No .cache directory found — nothing to clean.");
        return;
    }

    // Build the set of live files
    let mut live: HashSet<String> = HashSet::new();

    // Always preserve known infrastructure files
    for name in PRESERVED_NAMES {
        live.insert(name.to_string());
    }

    // Preserve files matching known intermediate artifact prefixes
    // (these are referenced by pipeline steps via ${cache}/name)
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if PRESERVED_PREFIXES.iter().any(|p| name.starts_with(p)) {
                live.insert(name);
            }
        }
    }

    // Read progress log to find recorded output files
    let progress_path = cache_dir.join(".upstream.progress.yaml");
    if progress_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&progress_path) {
            // Parse output paths from the YAML
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("path:") {
                    let path_val = trimmed.strip_prefix("path:").unwrap().trim().trim_matches('\'').trim_matches('"');
                    // Extract just the filename if it's a cache-relative path
                    if path_val.starts_with(".cache/") {
                        if let Some(name) = path_val.strip_prefix(".cache/") {
                            live.insert(name.to_string());
                        }
                    }
                }
            }
        }
    }

    // Scan .cache/ for all files and directories
    let mut orphaned: Vec<(String, u64, bool)> = Vec::new(); // (name, size, is_dir)
    let mut total_orphaned_bytes: u64 = 0;
    let mut total_live: usize = 0;

    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            let meta = entry.metadata().ok();
            let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);

            if live.contains(&name) {
                total_live += 1;
                continue;
            }

            // Directories like dedup_runs/ may contain stale run files
            let size = if is_dir {
                dir_size(&entry.path())
            } else {
                meta.map(|m| m.len()).unwrap_or(0)
            };

            orphaned.push((name, size, is_dir));
            total_orphaned_bytes += size;
        }
    }

    if orphaned.is_empty() {
        println!("No orphaned files found in .cache/ ({} live files).", total_live);
        return;
    }

    // Sort by size descending
    orphaned.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Orphaned files in .cache/ ({} live, {} orphaned, {} total):",
        total_live, orphaned.len(), format_size(total_orphaned_bytes));

    for (name, size, is_dir) in &orphaned {
        let kind = if *is_dir { "dir " } else { "file" };
        println!("  {} {:>10}  {}", kind, format_size(*size), name);
    }

    if dry_run {
        println!("\nDry run — no files deleted. Run without --dry-run to remove.");
        return;
    }

    // Delete orphaned files
    let mut deleted = 0u64;
    for (name, size, is_dir) in &orphaned {
        let path = cache_dir.join(name);
        let result = if *is_dir {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match result {
            Ok(()) => deleted += size,
            Err(e) => eprintln!("  warning: failed to remove {}: {}", name, e),
        }
    }

    println!("\nRemoved {} orphaned files ({})", orphaned.len(), format_size(deleted));
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.filter_map(|e| e.ok()) {
            if let Ok(meta) = entry.metadata() {
                if meta.is_dir() {
                    total += dir_size(&entry.path());
                } else {
                    total += meta.len();
                }
            }
        }
    }
    total
}

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
