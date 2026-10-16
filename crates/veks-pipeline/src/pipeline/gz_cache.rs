// Copyright (c) Jonathan Shook
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

/// Buffer size for the streaming copies: a few of these per thread is
/// all the memory a file of any size costs.
const STREAM_BUF: usize = 1 << 20;

/// A reader that adds every byte it yields to `counter`, so a caller can
/// report progress through a copy it does not drive itself.
struct Counting<'a, R> {
    inner: R,
    counter: &'a std::sync::atomic::AtomicU64,
}

impl<R: Read> Read for Counting<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

/// Compress the file at `path` into its `.gz` beside it at the current
/// level, streaming, so the memory it costs does not grow with the file.
/// The `.gz` is written to a temporary file and renamed into place, so
/// it is either whole or absent; `path` itself is left alone. Every byte
/// read from `path` is added to `progress`. Returns the `.gz` size.
pub fn compress_file(path: &Path, progress: &std::sync::atomic::AtomicU64) -> Result<u64, String> {
    let gz = gz_path(path);
    let tmp = gz.with_extension("gz.tmp");
    let copy = || -> std::io::Result<u64> {
        let src = std::fs::File::open(path)?;
        let mut reader = Counting { inner: std::io::BufReader::with_capacity(STREAM_BUF, src), counter: progress };
        let out = std::io::BufWriter::with_capacity(STREAM_BUF, std::fs::File::create(&tmp)?);
        let mut encoder = GzEncoder::new(out, compression_level());
        std::io::copy(&mut reader, &mut encoder)?;
        let file = encoder.finish()?.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(file.metadata()?.len())
    };
    match copy() {
        Ok(size) => {
            std::fs::rename(&tmp, &gz).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("failed to rename {}: {e}", tmp.display())
            })?;
            Ok(size)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(format!("failed to compress {}: {e}", path.display()))
        }
    }
}

/// Decompress the `.gz` of `path` into `path`, streaming, so the memory
/// it costs does not grow with the file. `path` is written to a
/// temporary file and renamed into place, so it is either whole or
/// absent; the `.gz` is left alone. Every byte read from the `.gz` is
/// added to `progress`. Returns the decompressed size.
pub fn decompress_file(path: &Path, progress: &std::sync::atomic::AtomicU64) -> Result<u64, String> {
    let gz = gz_path(path);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let copy = || -> std::io::Result<u64> {
        let src = std::fs::File::open(&gz)?;
        let counted = Counting { inner: std::io::BufReader::with_capacity(STREAM_BUF, src), counter: progress };
        let mut decoder = GzDecoder::new(counted);
        let mut out = std::io::BufWriter::with_capacity(STREAM_BUF, std::fs::File::create(&tmp)?);
        let size = std::io::copy(&mut decoder, &mut out)?;
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        Ok(size)
    };
    match copy() {
        Ok(size) => {
            std::fs::rename(&tmp, path).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("failed to rename {}: {e}", tmp.display())
            })?;
            Ok(size)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(format!("failed to decompress {}: {e}", gz.display()))
        }
    }
}

/// Compression statistics for logging.
pub struct GzStats {
    /// Uncompressed size in bytes.
    pub original_size: u64,
    /// Size of the `.gz` file in bytes.
    pub compressed_size: u64,
}

impl GzStats {
    /// Compression ratio, `original / compressed` (0 when nothing was written).
    pub fn ratio(&self) -> f64 {
        if self.compressed_size == 0 { 0.0 }
        else { self.original_size as f64 / self.compressed_size as f64 }
    }

    /// Space saved as a percentage of the original size (0 for an empty
    /// original).
    pub fn savings_pct(&self) -> f64 {
        if self.original_size == 0 { 0.0 }
        else { (1.0 - self.compressed_size as f64 / self.original_size as f64) * 100.0 }
    }
}

/// The path [`save_gz`] actually writes for `path`: the `.gz` sidecar when
/// compression is enabled, otherwise the raw `path` itself. Pure (takes the
/// mode as a parameter) so the branch selection is unit-testable without
/// mutating the process-global compression level.
fn written_path(path: &Path, compressed: bool) -> PathBuf {
    if compressed { gz_path(path) } else { path.to_path_buf() }
}

/// Save `data` via [`save_gz`] and return its size stats (original vs.
/// on-disk).
///
/// Stats are read from the file that was *actually* written — the `.gz`
/// sidecar when compression is enabled, or the raw `path` when it's
/// disabled (level 0). A failure to stat that file is a genuine error (the
/// write just succeeded), so it is propagated rather than silently reported
/// as `compressed_size = 0` — the latter made [`GzStats::ratio`] /
/// [`GzStats::savings_pct`] report bogus values (and previously mis-read the
/// `.gz` path even in raw mode, where it never reflects the real size).
pub fn save_gz_with_stats(path: &Path, data: &[u8]) -> Result<GzStats, String> {
    let original_size = data.len() as u64;
    save_gz(path, data)?;
    let written = written_path(path, compression_enabled());
    let compressed_size = std::fs::metadata(&written)
        .map_err(|e| format!("failed to stat written cache file {}: {}", written.display(), e))?
        .len();
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

    /// `save_gz_with_stats` must stat the file that was actually written:
    /// the `.gz` sidecar when compression is on, the raw path when it's off.
    /// Pure check — no global compression-level mutation (which would race
    /// with other tests in this binary).
    #[test]
    fn written_path_tracks_compression_mode() {
        let p = Path::new("seg.bin");
        assert_eq!(written_path(p, true), PathBuf::from("seg.bin.gz"));
        assert_eq!(written_path(p, false), PathBuf::from("seg.bin"));
        // An already-`.gz` path is returned unchanged in compressed mode.
        let g = Path::new("seg.bin.gz");
        assert_eq!(written_path(g, true), PathBuf::from("seg.bin.gz"));
    }

    /// Compressed-mode stats (the default) report a real non-zero size and a
    /// sane ratio — and `save_gz_with_stats` now surfaces a stat failure as
    /// an error instead of silently reporting zero bytes.
    #[test]
    fn stats_report_real_compressed_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.bin");
        let data: Vec<u8> = (0..100_000u32).flat_map(|i| i.to_le_bytes()).collect();

        let stats = save_gz_with_stats(&path, &data).unwrap();
        // The on-disk size matches the actual `.gz` file (default level 9).
        let on_disk = std::fs::metadata(gz_path(&path)).unwrap().len();
        assert_eq!(stats.compressed_size, on_disk);
        assert!(stats.compressed_size > 0, "compressed size must be measured, not zeroed");
        assert!(stats.ratio() > 1.0 && stats.savings_pct() > 0.0);
    }

    /// Bytes no generator would compress well, so the `.gz` grows as
    /// fast as its input is read.
    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect()
    }

    /// **A file compresses to its `.gz` and back byte for byte**, the
    /// original left in place, nothing temporary left behind, and every
    /// byte read counted.
    #[test]
    fn compress_and_decompress_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.predkeys.slab");
        let data = noise(3 << 20, 7);
        std::fs::write(&path, &data).unwrap();

        let read = std::sync::atomic::AtomicU64::new(0);
        let gz_size = compress_file(&path, &read).unwrap();
        assert_eq!(read.load(std::sync::atomic::Ordering::Relaxed), data.len() as u64);
        assert_eq!(std::fs::metadata(gz_path(&path)).unwrap().len(), gz_size);
        assert_eq!(std::fs::read(&path).unwrap(), data, "compress_file touched the original");

        std::fs::remove_file(&path).unwrap();
        let read = std::sync::atomic::AtomicU64::new(0);
        assert_eq!(decompress_file(&path, &read).unwrap(), data.len() as u64);
        assert_eq!(read.load(std::sync::atomic::Ordering::Relaxed), gz_size);
        assert_eq!(std::fs::read(&path).unwrap(), data);
        let names: Vec<String> = std::fs::read_dir(dir.path()).unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        assert!(names.iter().all(|n| !n.ends_with(".tmp")), "a temporary file was left: {names:?}");
    }

    /// **Compression streams: output is written while input is still
    /// arriving.** The input is a pipe its writer holds open until the
    /// `.gz` has started to grow, which a compressor that reads its
    /// whole input before writing never does — that is how
    /// `veks prepare cache-compress` held every file it compressed in
    /// memory at once and was killed at 765 GB.
    #[cfg(unix)]
    #[test]
    fn compress_file_writes_before_its_input_ends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.predkeys.slab");
        let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0, "mkfifo failed");
        let tmp = gz_path(&path).with_extension("gz.tmp");

        let writer = {
            let path = path.clone();
            let tmp = tmp.clone();
            std::thread::spawn(move || -> bool {
                let mut pipe = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
                let mut seed = 1;
                // Keep feeding until the output shows up, then close.
                while std::time::Instant::now() < deadline {
                    if pipe.write_all(&noise(1 << 20, seed)).is_err() {
                        return false;
                    }
                    seed += 1;
                    if std::fs::metadata(&tmp).map(|m| m.len() > 0).unwrap_or(false) {
                        return true;
                    }
                }
                false
            })
        };
        let read = std::sync::atomic::AtomicU64::new(0);
        let compressed = compress_file(&path, &read);
        let streamed = writer.join().unwrap();
        assert!(streamed, "no output was written while the input was still open");
        assert!(compressed.is_ok(), "{compressed:?}");
    }
}
