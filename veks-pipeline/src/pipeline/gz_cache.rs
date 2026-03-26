// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Gzip-compressed cache I/O for intermediate pipeline artifacts.
//!
//! Provides in-memory compress-then-save and load-then-decompress for
//! cache artifacts that are only accessed in streaming mode and fit in
//! memory as individual segments (sorted runs, partition caches, etc.).
//!
//! Files are stored with a `.gz` extension. The data is compressed in
//! memory before writing and decompressed in memory after reading,
//! avoiding streaming gzip I/O complexity.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

/// Configurable compression level. Defaults to maximum (9).
/// Cache files are written once and read many times, so we spend
/// more CPU up front to minimize I/O on every subsequent read.
/// Set to 0 to disable compression entirely (raw writes).
static LEVEL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(9);

/// Set the compression level (0-9). Call before any save operations.
/// Level 0 disables compression: files are written raw (no .gz wrapper).
pub fn set_compression_level(level: u32) {
    LEVEL.store(level.min(9), std::sync::atomic::Ordering::Relaxed);
}

/// Returns the current compression level.
pub fn compression_enabled() -> bool {
    LEVEL.load(std::sync::atomic::Ordering::Relaxed) > 0
}

fn compression_level() -> Compression {
    Compression::new(LEVEL.load(std::sync::atomic::Ordering::Relaxed))
}

/// Save data to cache. Compresses as `.gz` when compression level > 0,
/// writes raw otherwise. Atomic (write to tmp, rename).
pub fn save_gz(path: &Path, data: &[u8]) -> Result<(), String> {
    if !compression_enabled() {
        return save_raw(path, data);
    }

    let gz_path = gz_path(path);
    if let Some(parent) = gz_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create dir for {}: {}", gz_path.display(), e))?;
    }

    let mut encoder = GzEncoder::new(Vec::with_capacity(data.len() / 2), compression_level());
    encoder.write_all(data)
        .map_err(|e| format!("gzip compress error: {}", e))?;
    let compressed = encoder.finish()
        .map_err(|e| format!("gzip finish error: {}", e))?;

    let tmp_path = gz_path.with_extension("gz.tmp");
    std::fs::write(&tmp_path, &compressed)
        .map_err(|e| format!("failed to write {}: {}", tmp_path.display(), e))?;
    std::fs::rename(&tmp_path, &gz_path)
        .map_err(|e| format!("failed to rename {}: {}", tmp_path.display(), e))?;

    Ok(())
}

/// Save data as a raw (uncompressed) file. Atomic (write to tmp, rename).
fn save_raw(path: &Path, data: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create dir for {}: {}", path.display(), e))?;
    }
    let tmp_path = path.with_extension("bin.tmp");
    std::fs::write(&tmp_path, data)
        .map_err(|e| format!("failed to write {}: {}", tmp_path.display(), e))?;
    std::fs::rename(&tmp_path, path)
        .map_err(|e| format!("failed to rename {}: {}", tmp_path.display(), e))?;
    Ok(())
}

/// Load cached data. Tries `.gz` first, then raw file.
pub fn load_gz(path: &Path) -> Result<Vec<u8>, String> {
    let gz = gz_path(path);
    if gz.exists() {
        let compressed = std::fs::read(&gz)
            .map_err(|e| format!("failed to read {}: {}", gz.display(), e))?;
        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut data = Vec::with_capacity(compressed.len() * 3);
        decoder.read_to_end(&mut data)
            .map_err(|e| format!("gzip decompress error: {}", e))?;
        return Ok(data);
    }

    // Fall back to raw file
    if path.exists() {
        return std::fs::read(path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e));
    }

    Err(format!("cache file not found: {} (checked .gz and raw)", path.display()))
}

/// Check if a cached file exists (either `.gz` or raw).
pub fn gz_exists(path: &Path) -> bool {
    gz_path(path).exists() || path.exists()
}

/// Get the `.gz` path for a given cache path.
///
/// If the path already ends in `.gz`, returns it unchanged.
/// Otherwise appends `.gz`.
pub fn gz_path(path: &Path) -> PathBuf {
    if path.extension().map(|e| e == "gz").unwrap_or(false) {
        path.to_path_buf()
    } else {
        let mut p = path.as_os_str().to_owned();
        p.push(".gz");
        PathBuf::from(p)
    }
}

/// Get the uncompressed size hint from a `.gz` file (last 4 bytes = ISIZE).
///
/// This is the original uncompressed size modulo 2^32, useful for
/// pre-allocating the decompression buffer for files under 4 GiB.
pub fn gz_uncompressed_size_hint(path: &Path) -> Option<u32> {
    let gz = gz_path(path);
    let data = std::fs::read(&gz).ok()?;
    if data.len() < 4 { return None; }
    let last4 = &data[data.len() - 4..];
    Some(u32::from_le_bytes([last4[0], last4[1], last4[2], last4[3]]))
}

/// Compression statistics for logging.
pub struct GzStats {
    pub original_size: u64,
    pub compressed_size: u64,
}

impl GzStats {
    pub fn ratio(&self) -> f64 {
        if self.compressed_size == 0 { 0.0 }
        else { self.original_size as f64 / self.compressed_size as f64 }
    }

    pub fn savings_pct(&self) -> f64 {
        if self.original_size == 0 { 0.0 }
        else { (1.0 - self.compressed_size as f64 / self.original_size as f64) * 100.0 }
    }
}

/// Compress data in memory and write to `.gz`, returning stats.
pub fn save_gz_with_stats(path: &Path, data: &[u8]) -> Result<GzStats, String> {
    let original_size = data.len() as u64;
    save_gz(path, data)?;
    let compressed_size = std::fs::metadata(&gz_path(path))
        .map(|m| m.len())
        .unwrap_or(0);
    Ok(GzStats { original_size, compressed_size })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        let data = b"hello world this is test data for compression roundtrip";

        save_gz(&path, data).unwrap();
        assert!(gz_exists(&path));
        assert!(gz_path(&path).exists());

        let loaded = load_gz(&path).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn roundtrip_large() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");
        // 1 MB of patterned data (compresses well)
        let data: Vec<u8> = (0..1_000_000u32)
            .flat_map(|i| i.to_le_bytes())
            .collect();

        let stats = save_gz_with_stats(&path, &data).unwrap();
        assert!(stats.ratio() > 1.5, "expected good compression, got ratio {:.1}", stats.ratio());

        let loaded = load_gz(&path).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn gz_path_idempotent() {
        let p = PathBuf::from("test.bin");
        assert_eq!(gz_path(&p), PathBuf::from("test.bin.gz"));
        assert_eq!(gz_path(&gz_path(&p)), PathBuf::from("test.bin.gz"));
    }

    #[test]
    fn empty_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        save_gz(&path, &[]).unwrap();
        let loaded = load_gz(&path).unwrap();
        assert!(loaded.is_empty());
    }
}
