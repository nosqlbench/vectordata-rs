// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets cache` — list locally cached datasets.

use std::path::Path;

pub fn run(cache_dir: Option<&Path>, verbose: bool) {
    let cache_dir = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| crate::pipeline::commands::config::configured_cache_dir());

    if !cache_dir.exists() {
        println!("Cache directory does not exist: {}", crate::check::rel_display(&cache_dir));
        return;
    }

    println!("Cache directory: {}", crate::check::rel_display(&cache_dir));
    println!();

    let mut datasets = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("dataset.yaml").exists() {
                datasets.push(path);
            }
        }
    }

    datasets.sort();

    if datasets.is_empty() {
        println!("No datasets found in cache.");
    } else {
        println!("{:<30} {:>6} {:>12}", "Dataset", "Files", "Size");
        println!("{}", "-".repeat(52));

        for ds_dir in &datasets {
            let name = ds_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");
            let (file_count, total_size) = scan_directory(ds_dir);

            println!(
                "{:<30} {:>6} {:>12}",
                name,
                file_count,
                format_size(total_size)
            );

            if verbose {
                if let Ok(files) = std::fs::read_dir(ds_dir) {
                    for f in files.flatten() {
                        let fp = f.path();
                        if fp.is_file()
                            && fp.file_name().and_then(|n| n.to_str()) != Some("dataset.yaml")
                        {
                            let fname =
                                fp.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                            let fsize =
                                std::fs::metadata(&fp).map(|m| m.len()).unwrap_or(0);
                            println!("  {:<28} {:>12}", fname, format_size(fsize));
                        }
                    }
                }
            }
        }
    }

    println!();
    println!("{} dataset(s) cached", datasets.len());
}

/// Show cache status for all cached datasets.
pub fn run_cache_status_all(
    verbose: bool,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
) {
    let cache_dir = crate::pipeline::commands::config::configured_cache_dir();
    println!("Cache directory: {}", crate::check::rel_display(&cache_dir));
    println!();

    if !cache_dir.is_dir() {
        println!("  No cache directory found.");
        return;
    }

    let mut found = false;
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        let mut dirs: Vec<_> = entries.flatten()
            .filter(|e| e.path().is_dir())
            .collect();
        dirs.sort_by_key(|e| e.file_name());

        for entry in dirs {
            let name = entry.file_name().to_string_lossy().to_string();
            let ds_path = entry.path();
            // Only show directories that have .mrkl files or dataset.yaml
            let has_mrkl = has_mrkl_files(&ds_path);
            let has_yaml = ds_path.join("dataset.yaml").exists();
            if has_mrkl || has_yaml {
                if found { println!(); }
                found = true;
                run_cache_status(&name, verbose, configdir, extra_catalogs, at);
            }
        }
    }

    if !found {
        println!("  No cached datasets found.");
    }
}

fn has_mrkl_files(dir: &std::path::Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|ext| ext == "mrkl") { return true; }
            if p.is_dir() && has_mrkl_files(&p) { return true; }
        }
    }
    false
}

/// Show detailed cache status for a specific dataset across all profiles.
pub fn run_cache_status(
    dataset_name: &str,
    verbose: bool,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
) {
    use vectordata::merkle::MerkleState;
    use crate::catalog::resolver::Catalog;
    use crate::catalog::sources::CatalogSources;

    let cache_dir = crate::pipeline::commands::config::configured_cache_dir();
    let ds_cache = cache_dir.join(dataset_name);

    // Load catalog to get profile info
    let mut sources = CatalogSources::new();
    if !at.is_empty() {
        sources = sources.add_catalogs(at);
    } else {
        sources = sources.configure(configdir);
        if !extra_catalogs.is_empty() {
            sources = sources.add_catalogs(extra_catalogs);
        }
    }
    let catalog = Catalog::of(&sources);
    let entry = catalog.find_exact(dataset_name);

    println!("Cache status: {}", dataset_name);
    println!("  Cache dir: {}", crate::check::rel_display(&ds_cache));
    println!();

    if !ds_cache.is_dir() {
        println!("  Not cached (directory does not exist)");
        return;
    }

    // Scan all .mrkl files recursively
    let mut total_valid: u64 = 0;
    let mut total_chunks: u64 = 0;
    let mut total_cached_bytes: u64 = 0;
    let mut total_content_bytes: u64 = 0;
    let mut facet_rows: Vec<(String, u32, u32, u64, u64, bool, String)> = Vec::new();

    scan_mrkl_status(&ds_cache, &ds_cache, &mut facet_rows);

    if facet_rows.is_empty() {
        // No .mrkl files — show raw file listing
        println!("  No merkle state — files downloaded without verification:");
        println!();
        print_file_tree(&ds_cache, &ds_cache, 2);
        let (file_count, dir_size) = scan_directory_recursive(&ds_cache);
        println!();
        println!("  {} file(s), {} total", file_count, format_size(dir_size));
        return;
    }

    // Header
    println!("  {:<40} {:>8} {:>8} {:>6} {:>12} {:>12}",
        "FILE", "VALID", "TOTAL", "%", "CACHED", "CONTENT");
    println!("  {}", "-".repeat(88));

    for (path, valid, total_c, cached_bytes, content_bytes, complete, rle) in &facet_rows {
        let pct = if *total_c > 0 { 100.0 * *valid as f64 / *total_c as f64 } else { 0.0 };
        let status = if *complete { "OK" } else { "" };
        println!("  {:<40} {:>8} {:>8} {:>5.0}% {:>12} {:>12} {}",
            path, valid, total_c, pct,
            format_size(*cached_bytes), format_size(*content_bytes), status);
        if verbose && !rle.is_empty() && !*complete {
            println!("    coverage: {}", rle);
        }
        total_valid += *valid as u64;
        total_chunks += *total_c as u64;
        total_cached_bytes += cached_bytes;
        total_content_bytes += content_bytes;
    }

    println!("  {}", "-".repeat(88));
    let overall_pct = if total_chunks > 0 { 100.0 * total_valid as f64 / total_chunks as f64 } else { 0.0 };
    println!("  {:<40} {:>8} {:>8} {:>5.0}% {:>12} {:>12}",
        "TOTAL", total_valid, total_chunks, overall_pct,
        format_size(total_cached_bytes), format_size(total_content_bytes));

    // Profile summary if catalog is available
    if let Some(entry) = entry {
        println!();
        println!("  Profiles:");
        for pname in entry.profile_names() {
            if let Some(profile) = entry.layout.profiles.profile(pname) {
                let mut pv = 0u32;
                let mut pt = 0u32;
                for (_facet, view) in &profile.views {
                    let source = &view.source.path;
                    if source.is_empty() { continue; }
                    let clean = if let Some(b) = source.find(|c: char| c == '[' || c == '(') {
                        &source[..b]
                    } else { source.as_str() };
                    let mrkl = ds_cache.join(format!("{}.mrkl", clean));
                    if let Ok(state) = MerkleState::load(&mrkl) {
                        pv += state.valid_count();
                        pt += state.shape().total_chunks;
                    }
                }
                if pt > 0 {
                    let ppct = 100.0 * pv as f64 / pt as f64;
                    let label = if pv == pt { "complete" } else { "" };
                    println!("    {:<20} {:>5.0}% ({}/{} chunks) {}",
                        pname, ppct, pv, pt, label);
                } else {
                    println!("    {:<20} not cached", pname);
                }
            }
        }
    }
}

/// Rows: (path, valid_chunks, total_chunks, cached_bytes, content_bytes, complete, rle_string)
fn scan_mrkl_status(
    base: &Path, dir: &Path,
    rows: &mut Vec<(String, u32, u32, u64, u64, bool, String)>,
) {
    use vectordata::merkle::MerkleState;

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_mrkl_status(base, &path, rows);
        } else if path.extension().is_some_and(|e| e == "mrkl") {
            if let Ok(state) = MerkleState::load(&path) {
                let valid = state.valid_count();
                let total = state.shape().total_chunks;
                let content_size = state.shape().total_content_size;
                let chunk_size = state.shape().chunk_size;
                let cached_bytes = valid as u64 * chunk_size;
                let complete = state.is_complete();
                let rle = chunk_coverage_rle(&state);
                let rel = path.strip_prefix(base)
                    .map(|r| r.to_string_lossy().to_string())
                    .unwrap_or_else(|_| path.to_string_lossy().to_string());
                rows.push((rel, valid, total, cached_bytes, content_size, complete, rle));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
}

/// Build an RLE representation of chunk coverage.
///
/// Output format: `0-99:ok 100-149:miss 150-200:ok`
/// Consecutive runs of the same state are collapsed.
fn chunk_coverage_rle(state: &vectordata::merkle::MerkleState) -> String {
    let total = state.shape().total_chunks;
    if total == 0 { return String::new(); }

    let mut parts = Vec::new();
    let mut run_start = 0u32;
    let mut run_valid = state.is_valid(0);

    for i in 1..total {
        let v = state.is_valid(i);
        if v != run_valid {
            let label = if run_valid { "ok" } else { "miss" };
            if run_start == i - 1 {
                parts.push(format!("{}:{}", run_start, label));
            } else {
                parts.push(format!("{}-{}:{}", run_start, i - 1, label));
            }
            run_start = i;
            run_valid = v;
        }
    }
    // Final run
    let label = if run_valid { "ok" } else { "miss" };
    if run_start == total - 1 {
        parts.push(format!("{}:{}", run_start, label));
    } else {
        parts.push(format!("{}-{}:{}", run_start, total - 1, label));
    }

    parts.join(" ")
}

/// Collect all files with relative paths and sizes, then print aligned.
fn print_file_tree(base: &Path, _dir: &Path, _indent: usize) {
    let mut files: Vec<(String, u64)> = Vec::new();
    collect_files_recursive(base, base, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));

    if files.is_empty() { return; }

    let max_path = files.iter().map(|(p, _)| p.len()).max().unwrap_or(20);
    let col_w = max_path.max(20) + 2;

    for (rel, size) in &files {
        println!("    {:<width$} {:>10}", rel, format_size(*size), width = col_w);
    }
}

fn collect_files_recursive(base: &Path, dir: &Path, files: &mut Vec<(String, u64)>) {
    let mut entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(e) => e.flatten().collect(),
        Err(_) => return,
    };
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(base, &path, files);
        } else {
            let rel = path.strip_prefix(base)
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            files.push((rel, size));
        }
    }
}

fn scan_directory_recursive(dir: &Path) -> (usize, u64) {
    let mut count = 0;
    let mut size = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                count += 1;
                size += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            } else if path.is_dir() {
                let (c, s) = scan_directory_recursive(&path);
                count += c;
                size += s;
            }
        }
    }
    (count, size)
}

/// Print a tree view of the cache directory with sizes.
pub fn print_cache_tree(dir: &Path) {
    print_tree_recursive(dir, "    ", true);
}

fn print_tree_recursive(dir: &Path, prefix: &str, is_root: bool) {
    let mut entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(e) => e.flatten().collect(),
        Err(_) => return,
    };
    entries.sort_by_key(|e| e.file_name());

    if is_root {
        let name = dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(".");
        println!("{}{}/ ", prefix, name);
    }

    let count = entries.len();
    for (i, entry) in entries.iter().enumerate() {
        let is_last = i == count - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let child_prefix = if is_last {
            format!("{}    ", prefix)
        } else {
            format!("{}│   ", prefix)
        };
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path();

        if path.is_dir() {
            let (fc, sz) = scan_directory_recursive(&path);
            println!("{}{}{}/  ({} files, {})",
                prefix, connector, name, fc, format_size(sz));
            print_tree_recursive(&path, &child_prefix, false);
        } else {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!("{}{}{}  {}", prefix, connector, name, format_size(size));
        }
    }
}

fn scan_directory(dir: &Path) -> (usize, u64) {
    let mut count = 0;
    let mut size = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().is_file() {
                count += 1;
                size += std::fs::metadata(entry.path())
                    .map(|m| m.len())
                    .unwrap_or(0);
            }
        }
    }
    (count, size)
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_values() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1048576), "1.0 MB");
    }

    #[test]
    fn scan_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let (count, size) = scan_directory(tmp.path());
        assert_eq!(count, 0);
        assert_eq!(size, 0);
    }
}
