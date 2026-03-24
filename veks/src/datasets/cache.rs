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
