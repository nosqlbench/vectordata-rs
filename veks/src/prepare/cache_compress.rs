// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks cache compress` — retroactively compress eligible cache files.
//!
//! Walks a cache directory tree, identifies files that are only accessed
//! sequentially (sorted runs, KNN partition caches, predicate segment
//! caches, etc.), compresses each one with gzip, and deletes the original
//! uncompressed file.
//!
//! Files that require random-access mmap (vectors, shuffle indices, ordinals,
//! JSON metadata) are skipped.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Glob-style patterns for files eligible for compression.
///
/// These are sequential-only artifacts that the pipeline loads entirely
/// into memory via `load_gz()` when it needs them.
const ELIGIBLE_PATTERNS: &[&str] = &[
    // dedup sorted runs
    "run_",           // dedup_runs/run_*.bin
    ".neighbors.ivec", // KNN partition caches
    ".distances.fvec", // KNN partition caches
    ".predkeys.slab",  // predicate segment caches
    "compute-knn.part_", // KNN partition cache files
];

/// File names and extensions that must NOT be compressed (random-access / mmap).
const SKIP_NAMES: &[&str] = &[
    "all_vectors.mvec",
    "all_vectors.hvec", // legacy name before hvec→mvec rename
    "shuffle.ivecs",
    "dedup_ordinals.ivec",
    "clean_ordinals.ivec",
    "meta.json",
];

const SKIP_EXTENSIONS: &[&str] = &["json", "gz"];

/// Returns `true` if the file name matches an eligible compression pattern.
fn is_eligible(file_name: &str) -> bool {
    // Never compress files that are explicitly skipped.
    if SKIP_NAMES.iter().any(|s| file_name == *s) {
        return false;
    }
    if let Some(ext) = Path::new(file_name).extension().and_then(|e| e.to_str()) {
        if SKIP_EXTENSIONS.iter().any(|s| *s == ext) {
            return false;
        }
    }

    ELIGIBLE_PATTERNS.iter().any(|pat| file_name.contains(pat))
}

/// Recursively collect all files under `dir`.
fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_files(&path, &mut *out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

/// Run the cache-compress command.
///
/// Walks `cache_dir`, compresses eligible files via
/// [`crate::pipeline::gz_cache::save_gz`], removes the originals, and
/// prints a summary of files compressed and total space saved.
///
/// When `dry_run` is `true`, reports what *would* be compressed without
/// modifying any files.
pub fn run(cache_dir: &Path, dry_run: bool) {
    if !cache_dir.exists() {
        eprintln!("Cache directory does not exist: {}", crate::check::rel_display(&cache_dir.to_path_buf()));
        std::process::exit(1);
    }

    let mut all_files = Vec::new();
    walk_files(cache_dir, &mut all_files);
    all_files.sort();

    let eligible: Vec<&PathBuf> = all_files
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(is_eligible)
                .unwrap_or(false)
        })
        .collect();

    if eligible.is_empty() {
        println!("No eligible files found in {}", crate::check::rel_display(&cache_dir.to_path_buf()));
        return;
    }

    println!(
        "{} eligible file(s) in {}",
        eligible.len(),
        crate::check::rel_display(&cache_dir.to_path_buf())
    );
    if dry_run {
        println!("(dry run — no files will be modified)");
    }
    println!();

    let total = eligible.len();

    if dry_run {
        let mut total_original: u64 = 0;
        for (i, path) in eligible.iter().enumerate() {
            let rel = path.strip_prefix(cache_dir).unwrap_or(path);
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            println!("[{}/{}] would compress: {} ({})", i + 1, total, rel.display(), format_size(size));
            total_original += size;
        }
        println!("\n--- cache compress summary (dry run) ---");
        println!("Files:         {}", total);
        println!("Original size: {}", format_size(total_original));
        return;
    }

    // Parallel compression with rayon
    use rayon::prelude::*;

    let compressed_count = AtomicU64::new(0);
    let a_total_original = AtomicU64::new(0);
    let a_total_compressed = AtomicU64::new(0);
    let a_errors = AtomicU64::new(0);
    let done = AtomicU64::new(0);

    eligible.par_iter().for_each(|path| {
        let rel = path.strip_prefix(cache_dir).unwrap_or(path);
        let original_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let original_mtime = std::fs::metadata(path.as_path()).ok().and_then(|m| m.modified().ok());

        let data = match std::fs::read(path.as_path()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ERROR reading {}: {}", rel.display(), e);
                a_errors.fetch_add(1, Ordering::Relaxed);
                done.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        if let Err(e) = crate::pipeline::gz_cache::save_gz(path, &data) {
            eprintln!("ERROR compressing {}: {}", rel.display(), e);
            a_errors.fetch_add(1, Ordering::Relaxed);
            done.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let gz_path = crate::pipeline::gz_cache::gz_path(path);
        let compressed_size = std::fs::metadata(&gz_path).map(|m| m.len()).unwrap_or(0);

        if let Some(mtime) = original_mtime {
            let _ = filetime::set_file_mtime(&gz_path, filetime::FileTime::from_system_time(mtime));
        }

        let _ = std::fs::remove_file(path.as_path());

        a_total_original.fetch_add(original_size, Ordering::Relaxed);
        a_total_compressed.fetch_add(compressed_size, Ordering::Relaxed);
        compressed_count.fetch_add(1, Ordering::Relaxed);
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 10 == 0 || n as usize == total {
            eprint!("\r[{}/{}] compressed...       ", n, total);
        }
    });

    let cc = compressed_count.load(Ordering::Relaxed);
    let to = a_total_original.load(Ordering::Relaxed);
    let tc = a_total_compressed.load(Ordering::Relaxed);
    let errs = a_errors.load(Ordering::Relaxed);

    eprint!("\r{}\r", " ".repeat(60));
    println!("\n--- cache compress summary ---");
    println!("Files compressed: {}", cc);
    if errs > 0 { println!("Errors:           {}", errs); }
    println!("Original size:    {}", format_size(to));
    println!("Compressed size:  {}", format_size(tc));
    if to > 0 {
        let saved = to.saturating_sub(tc);
        println!("Space saved:      {} ({:.1}%)", format_size(saved), (saved as f64 / to as f64) * 100.0);
    }
}

/// Run the cache-uncompress command.
///
/// Walks `cache_dir`, finds `.gz` files that match eligible patterns,
/// decompresses each one, writes the original file, deletes the `.gz`,
/// and preserves the mtime.
pub fn run_uncompress(cache_dir: &Path, dry_run: bool) {
    if !cache_dir.exists() {
        eprintln!("Cache directory does not exist: {}", crate::check::rel_display(&cache_dir.to_path_buf()));
        std::process::exit(1);
    }

    let mut all_files = Vec::new();
    walk_files(cache_dir, &mut all_files);
    all_files.sort();

    // Find .gz files whose base name matches an eligible pattern
    let gz_files: Vec<&PathBuf> = all_files
        .iter()
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.ends_with(".gz") { return false; }
            let base = &name[..name.len() - 3]; // strip .gz
            is_eligible(base)
        })
        .collect();

    if gz_files.is_empty() {
        println!("No compressed cache files found in {}", crate::check::rel_display(&cache_dir.to_path_buf()));
        return;
    }

    println!("{} compressed file(s) in {}", gz_files.len(), crate::check::rel_display(&cache_dir.to_path_buf()));
    if dry_run { println!("(dry run — no files will be modified)"); }
    println!();

    let total = gz_files.len();

    if dry_run {
        let mut total_compressed: u64 = 0;
        for (i, gz_path) in gz_files.iter().enumerate() {
            let rel = gz_path.strip_prefix(cache_dir).unwrap_or(gz_path);
            let compressed_size = std::fs::metadata(gz_path.as_path()).map(|m| m.len()).unwrap_or(0);
            println!("[{}/{}] would uncompress: {} ({})", i + 1, total, rel.display(), format_size(compressed_size));
            total_compressed += compressed_size;
        }
        println!("\n--- cache uncompress summary (dry run) ---");
        println!("Files:           {}", total);
        println!("Compressed size: {}", format_size(total_compressed));
        return;
    }

    // Parallel decompression with rayon
    use rayon::prelude::*;

    let uncompressed_count = AtomicU64::new(0);
    let a_total_compressed = AtomicU64::new(0);
    let a_total_uncompressed = AtomicU64::new(0);
    let a_errors = AtomicU64::new(0);
    let done = AtomicU64::new(0);

    gz_files.par_iter().for_each(|gz_path| {
        let rel = gz_path.strip_prefix(cache_dir).unwrap_or(gz_path);
        let compressed_size = std::fs::metadata(gz_path.as_path()).map(|m| m.len()).unwrap_or(0);

        // Capture mtime from .gz for preservation
        let gz_mtime = std::fs::metadata(gz_path.as_path()).ok().and_then(|m| m.modified().ok());

        // Derive the original path by stripping .gz
        let original_path = gz_path.with_extension("");

        let data = match crate::pipeline::gz_cache::load_gz(&original_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ERROR decompressing {}: {}", rel.display(), e);
                a_errors.fetch_add(1, Ordering::Relaxed);
                done.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        if let Err(e) = std::fs::write(&original_path, &data) {
            eprintln!("ERROR writing {}: {}", original_path.display(), e);
            a_errors.fetch_add(1, Ordering::Relaxed);
            done.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Preserve mtime
        if let Some(mtime) = gz_mtime {
            let _ = filetime::set_file_mtime(&original_path, filetime::FileTime::from_system_time(mtime));
        }

        // Remove .gz
        if let Err(e) = std::fs::remove_file(gz_path.as_path()) {
            eprintln!("WARNING: uncompressed ok but failed to remove {}: {}", rel.display(), e);
        }

        a_total_compressed.fetch_add(compressed_size, Ordering::Relaxed);
        a_total_uncompressed.fetch_add(data.len() as u64, Ordering::Relaxed);
        uncompressed_count.fetch_add(1, Ordering::Relaxed);
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 10 == 0 || n as usize == total {
            eprint!("\r[{}/{}] uncompressed...       ", n, total);
        }
    });

    let uc = uncompressed_count.load(Ordering::Relaxed);
    let tc = a_total_compressed.load(Ordering::Relaxed);
    let tu = a_total_uncompressed.load(Ordering::Relaxed);
    let errs = a_errors.load(Ordering::Relaxed);

    eprint!("\r{}\r", " ".repeat(60));
    println!("\n--- cache uncompress summary ---");
    println!("Files uncompressed: {}", uc);
    if errs > 0 { println!("Errors:             {}", errs); }
    println!("Compressed size:    {}", format_size(tc));
    println!("Uncompressed size:  {}", format_size(tu));
}

/// Human-readable size formatting (matches `cache.rs` style).
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
    fn eligible_patterns() {
        assert!(is_eligible("run_00042.bin"));
        assert!(is_eligible("part0.neighbors.ivec"));
        assert!(is_eligible("part3.distances.fvec"));
        assert!(is_eligible("seg7.predkeys.slab"));
        assert!(is_eligible("compute-knn.part_0012.cache"));
    }

    #[test]
    fn skip_patterns() {
        assert!(!is_eligible("all_vectors.mvec"));
        assert!(!is_eligible("shuffle.ivecs"));
        assert!(!is_eligible("dedup_ordinals.ivec"));
        assert!(!is_eligible("clean_ordinals.ivec"));
        assert!(!is_eligible("meta.json"));
        assert!(!is_eligible("something.json"));
        assert!(!is_eligible("already.gz"));
        assert!(!is_eligible("run_00042.bin.gz"));
    }

    #[test]
    fn dry_run_no_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let dedup_dir = tmp.path().join("dedup_runs");
        std::fs::create_dir_all(&dedup_dir).unwrap();

        let f = dedup_dir.join("run_0001.bin");
        std::fs::write(&f, b"test data for compression").unwrap();

        run(tmp.path(), true);

        // Original file still exists, no .gz created.
        assert!(f.exists());
        assert!(!f.with_extension("bin.gz").exists());
    }

    #[test]
    fn compress_eligible_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dedup_dir = tmp.path().join("dedup_runs");
        std::fs::create_dir_all(&dedup_dir).unwrap();

        let f = dedup_dir.join("run_0001.bin");
        let data = vec![42u8; 4096];
        std::fs::write(&f, &data).unwrap();

        run(tmp.path(), false);

        // Original removed, .gz exists.
        assert!(!f.exists(), "original should be deleted");
        let gz = dedup_dir.join("run_0001.bin.gz");
        assert!(gz.exists(), ".gz file should exist");

        // Verify roundtrip.
        let loaded = crate::pipeline::gz_cache::load_gz(&f).unwrap();
        assert_eq!(loaded, data);
    }
}
