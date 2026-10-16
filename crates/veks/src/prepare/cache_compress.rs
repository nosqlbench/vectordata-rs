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
    if SKIP_NAMES.contains(&file_name) {
        return false;
    }
    if let Some(ext) = Path::new(file_name).extension().and_then(|e| e.to_str())
        && SKIP_EXTENSIONS.contains(&ext) {
            return false;
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
        eprintln!("Cache directory does not exist: {}", crate::check::rel_display(cache_dir));
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
        println!("No eligible files found in {}", crate::check::rel_display(cache_dir));
        return;
    }

    println!(
        "{} eligible file(s) in {}",
        eligible.len(),
        crate::check::rel_display(cache_dir)
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

    let sizes: Vec<u64> = eligible.iter().map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).collect();
    let outcome = for_each_streaming(&eligible, &sizes, "compress", |path, read| {
        let original_mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
        let compressed_size = crate::pipeline::gz_cache::compress_file(path, read)?;
        let gz_path = crate::pipeline::gz_cache::gz_path(path);
        if let Some(mtime) = original_mtime {
            let _ = filetime::set_file_mtime(&gz_path, filetime::FileTime::from_system_time(mtime));
        }
        let _ = std::fs::remove_file(path);
        Ok(compressed_size)
    }, cache_dir);

    println!("\n--- cache compress summary ---");
    println!("Files compressed: {}", outcome.done);
    if outcome.errors > 0 { println!("Errors:           {}", outcome.errors); }
    println!("Original size:    {}", format_size(outcome.read));
    println!("Compressed size:  {}", format_size(outcome.written));
    if outcome.read > 0 {
        let saved = outcome.read.saturating_sub(outcome.written);
        println!("Space saved:      {} ({:.1}%)", format_size(saved), (saved as f64 / outcome.read as f64) * 100.0);
    }
}

/// What [`for_each_streaming`] did: files finished and failed, and the
/// bytes read from and written to disk across them.
struct StreamOutcome {
    done: u64,
    errors: u64,
    read: u64,
    written: u64,
}

/// Run `op` over `files` on the rayon pool, each one a streaming copy
/// that adds the bytes it reads to the counter it is given and returns
/// the size it wrote, so the memory a file costs does not grow with it.
/// The files, bytes and the file last finished are reported on stderr as
/// the work goes: redrawn in place on a terminal, a line every 30 seconds
/// otherwise.
fn for_each_streaming(
    files: &[&PathBuf],
    sizes: &[u64],
    verb: &str,
    op: impl Fn(&Path, &AtomicU64) -> Result<u64, String> + Sync,
    root: &Path,
) -> StreamOutcome {
    use rayon::prelude::*;
    use std::io::IsTerminal;

    let total = files.len();
    let total_bytes: u64 = sizes.iter().sum();
    let read = AtomicU64::new(0);
    let written = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let errors = AtomicU64::new(0);
    let last = std::sync::Mutex::new(String::new());
    let finished = std::sync::atomic::AtomicBool::new(false);
    let tty = std::io::stderr().is_terminal();

    let line = || {
        format!(
            "{verb}: {}/{} files, {}/{} — {}",
            done.load(Ordering::Relaxed) + errors.load(Ordering::Relaxed),
            total,
            format_size(read.load(Ordering::Relaxed)),
            format_size(total_bytes),
            last.lock().expect("last file"),
        )
    };
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut since_line = std::time::Instant::now();
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(250));
                if tty {
                    eprint!("\r  {}\x1b[K", line());
                } else if since_line.elapsed() >= std::time::Duration::from_secs(30) {
                    eprintln!("  {}", line());
                    since_line = std::time::Instant::now();
                }
            }
        });
        files.par_iter().for_each(|path| {
            let rel = path.strip_prefix(root).unwrap_or(path);
            match op(path, &read) {
                Ok(size) => {
                    written.fetch_add(size, Ordering::Relaxed);
                    done.fetch_add(1, Ordering::Relaxed);
                    *last.lock().expect("last file") = rel.display().to_string();
                }
                Err(e) => {
                    eprintln!("\nERROR {verb}ing {}: {e}", rel.display());
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        finished.store(true, Ordering::Relaxed);
    });
    if tty {
        eprintln!("\r  {}\x1b[K", line());
    } else {
        eprintln!("  {}", line());
    }
    StreamOutcome {
        done: done.into_inner(),
        errors: errors.into_inner(),
        read: read.into_inner(),
        written: written.into_inner(),
    }
}

/// Run the cache-uncompress command.
///
/// Walks `cache_dir`, finds `.gz` files that match eligible patterns,
/// decompresses each one, writes the original file, deletes the `.gz`,
/// and preserves the mtime.
pub fn run_uncompress(cache_dir: &Path, dry_run: bool) {
    if !cache_dir.exists() {
        eprintln!("Cache directory does not exist: {}", crate::check::rel_display(cache_dir));
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
        println!("No compressed cache files found in {}", crate::check::rel_display(cache_dir));
        return;
    }

    println!("{} compressed file(s) in {}", gz_files.len(), crate::check::rel_display(cache_dir));
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

    let sizes: Vec<u64> = gz_files.iter().map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).collect();
    let outcome = for_each_streaming(&gz_files, &sizes, "uncompress", |gz_path, read| {
        let gz_mtime = std::fs::metadata(gz_path).ok().and_then(|m| m.modified().ok());
        let original_path = gz_path.with_extension("");
        let size = crate::pipeline::gz_cache::decompress_file(&original_path, read)?;
        if let Some(mtime) = gz_mtime {
            let _ = filetime::set_file_mtime(&original_path, filetime::FileTime::from_system_time(mtime));
        }
        if let Err(e) = std::fs::remove_file(gz_path) {
            eprintln!("\nWARNING: uncompressed ok but failed to remove {}: {}", gz_path.display(), e);
        }
        Ok(size)
    }, cache_dir);

    println!("\n--- cache uncompress summary ---");
    println!("Files uncompressed: {}", outcome.done);
    if outcome.errors > 0 { println!("Errors:             {}", outcome.errors); }
    println!("Compressed size:    {}", format_size(outcome.read));
    println!("Uncompressed size:  {}", format_size(outcome.written));
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

    /// **Compress and uncompress give back each file byte for byte, with
    /// its modification time**, which is how the pipeline judges a cache
    /// file against the step that wrote it.
    #[test]
    fn uncompress_restores_what_compress_took() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("keys.ab.seg_0000000000_0000000010.predkeys.slab");
        let data: Vec<u8> = (0..1u32 << 20).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        std::fs::write(&f, &data).unwrap();
        let older = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        std::fs::File::options().write(true).open(&f).unwrap().set_modified(older).unwrap();

        run(tmp.path(), false);
        assert!(!f.exists(), "the original was kept");
        run_uncompress(tmp.path(), false);

        assert_eq!(std::fs::read(&f).unwrap(), data);
        assert_eq!(std::fs::metadata(&f).unwrap().modified().unwrap(), older);
        assert!(!crate::pipeline::gz_cache::gz_path(&f).exists(), "the .gz was kept");
    }
}
