// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets drop-cache` — remove cached dataset directories.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Run the drop-cache command.
///
/// When `datasets` is empty, discovers all cached datasets and prompts
/// the user to confirm each one (unless `yes` is true).
/// When `datasets` is non-empty, each entry is matched as an exact name
/// or glob pattern against cached dataset directory names.
pub fn run(
    datasets: &[String],
    cache_dir: Option<&Path>,
    yes: bool,
    verbose: bool,
) {
    let cache_dir = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| crate::pipeline::commands::config::configured_cache_dir());

    if !cache_dir.exists() {
        println!("Cache directory does not exist: {}", cache_dir.display());
        return;
    }

    // Discover all cached datasets (directories containing dataset.yaml
    // or .mrkl files).
    let all_cached = discover_cached_datasets(&cache_dir);
    if all_cached.is_empty() {
        println!("No cached datasets found in {}", cache_dir.display());
        return;
    }

    // Resolve which datasets to drop.
    let targets: Vec<&PathBuf> = if datasets.is_empty() {
        // No filter — offer all cached datasets.
        all_cached.iter().collect()
    } else {
        // Match each argument as exact name or glob.
        let mut matched: Vec<&PathBuf> = Vec::new();
        for pattern in datasets {
            let mut found = false;
            for ds_path in &all_cached {
                let name = ds_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                if matches_pattern(name, pattern) {
                    if !matched.contains(&ds_path) {
                        matched.push(ds_path);
                    }
                    found = true;
                }
            }
            if !found {
                eprintln!("Warning: no cached dataset matches '{}'", pattern);
            }
        }
        matched
    };

    if targets.is_empty() {
        println!("No matching cached datasets found.");
        return;
    }

    // Sort by name for consistent display.
    let mut targets = targets;
    targets.sort_by_key(|p| p.file_name().unwrap_or_default().to_os_string());

    // Compute sizes.
    println!("Cache directory: {}", cache_dir.display());
    println!();

    let mut entries: Vec<(&PathBuf, String, u64, usize, Vec<(String, u64)>)> = Vec::new();
    for ds_path in &targets {
        let name = ds_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        let mut files = Vec::new();
        scan_directory(ds_path, ds_path, &mut files);
        let file_count = files.len();
        let total_size: u64 = files.iter().map(|(_, s)| *s).sum();
        entries.push((ds_path, name, total_size, file_count, files));
    }

    if yes {
        // Non-interactive: drop all targets.
        for (ds_path, name, total_size, file_count, files) in &entries {
            if verbose {
                println!("  {}/ ({}, {} files):", name, format_size(*total_size), file_count);
                for (rel, sz) in files {
                    println!("    {:<50} {:>12}", rel, format_size(*sz));
                }
            }
            print!("  Dropping {} ({}, {} files)... ", name, format_size(*total_size), file_count);
            io::stdout().flush().ok();
            match std::fs::remove_dir_all(ds_path) {
                Ok(()) => println!("done"),
                Err(e) => println!("ERROR: {}", e),
            }
        }
    } else {
        // Interactive: prompt for each dataset.
        println!("Cached datasets to drop:\n");
        for (i, (_ds_path, name, total_size, file_count, files)) in entries.iter().enumerate() {
            println!("  {:>3}. {:<30} {:>6} files  {:>12}",
                i + 1, name, file_count, format_size(*total_size));
            if verbose {
                for (rel, sz) in files {
                    println!("       {:<46} {:>12}", rel, format_size(*sz));
                }
            }
        }
        let total_bytes: u64 = entries.iter().map(|(_, _, sz, _, _)| *sz).sum();
        println!();
        println!("  Total: {} dataset(s), {}", entries.len(), format_size(total_bytes));
        println!();

        let answer = if entries.len() == 1 {
            prompt(&format!("Drop {}? [y/N] ", entries[0].1))
        } else {
            prompt("Drop all listed datasets? [y/N/select] ")
        };
        let answer = answer.trim().to_lowercase();

        match answer.as_str() {
            "y" | "yes" => {
                for (ds_path, name, total_size, file_count, _) in &entries {
                    print!("  Dropping {} ({}, {} files)... ", name, format_size(*total_size), file_count);
                    io::stdout().flush().ok();
                    match std::fs::remove_dir_all(ds_path) {
                        Ok(()) => println!("done"),
                        Err(e) => println!("ERROR: {}", e),
                    }
                }
            }
            "s" | "select" => {
                // Per-dataset confirmation.
                for (ds_path, name, total_size, file_count, _) in &entries {
                    let ans = prompt(&format!("  Drop {} ({}, {} files)? [y/N] ",
                        name, format_size(*total_size), file_count));
                    if ans.trim().eq_ignore_ascii_case("y") {
                        match std::fs::remove_dir_all(ds_path) {
                            Ok(()) => println!("    dropped"),
                            Err(e) => println!("    ERROR: {}", e),
                        }
                    } else {
                        println!("    skipped");
                    }
                }
            }
            _ => {
                println!("Cancelled.");
            }
        }
    }
}

/// Discover cached dataset directories.
fn discover_cached_datasets(cache_dir: &Path) -> Vec<PathBuf> {
    let mut datasets = Vec::new();
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let has_yaml = path.join("dataset.yaml").exists();
                let has_mrkl = has_mrkl_files(&path);
                if has_yaml || has_mrkl {
                    datasets.push(path);
                }
            }
        }
    }
    datasets.sort();
    datasets
}

/// Check if a name matches a pattern (exact, glob, or substring).
fn matches_pattern(name: &str, pattern: &str) -> bool {
    if name == pattern {
        return true;
    }
    // Glob matching.
    if pattern.contains('*') || pattern.contains('?') {
        return glob_match(pattern, name);
    }
    // Substring.
    name.contains(pattern)
}

/// Simple glob matcher supporting `*` and `?`.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<_> = pattern.chars().collect();
    let t: Vec<_> = text.chars().collect();
    glob_match_inner(&p, &t, 0, 0)
}

fn glob_match_inner(pattern: &[char], text: &[char], pi: usize, ti: usize) -> bool {
    if pi == pattern.len() {
        return ti == text.len();
    }
    match pattern[pi] {
        '*' => {
            // Match zero or more characters.
            for i in ti..=text.len() {
                if glob_match_inner(pattern, text, pi + 1, i) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if ti < text.len() {
                glob_match_inner(pattern, text, pi + 1, ti + 1)
            } else {
                false
            }
        }
        c => {
            if ti < text.len() && text[ti] == c {
                glob_match_inner(pattern, text, pi + 1, ti + 1)
            } else {
                false
            }
        }
    }
}

fn has_mrkl_files(dir: &Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|ext| ext == "mrkl") { return true; }
            if p.is_dir() && has_mrkl_files(&p) { return true; }
        }
    }
    false
}

/// Collect all files under `dir` with their relative paths and sizes.
fn scan_directory(root: &Path, dir: &Path, out: &mut Vec<(String, u64)>) {
    let mut entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().collect(),
        Err(_) => return,
    };
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        if p.is_dir() {
            scan_directory(root, &p, out);
        } else {
            let rel = p.strip_prefix(root)
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|_| p.to_string_lossy().to_string());
            let sz = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            out.push((rel, sz));
        }
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_099_511_627_776 {
        format!("{:.1} TiB", bytes as f64 / 1_099_511_627_776.0)
    } else if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn prompt(msg: &str) -> String {
    print!("{}", msg);
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    input
}
