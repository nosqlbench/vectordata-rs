// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Npy directory reader — reads directories of `.npy` files containing 2D
//! float arrays and emits raw LE vector bytes.
//!
//! Supports f16 (`<f2`), f32 (`<f4`), and f64 (`<f8`) element types.
//! A configurable pool of background workers loads npy files in parallel
//! while an adaptive prefetch gate controls how far ahead they may read.
//! Results are reassembled in strict file-sorted order by the consumer so
//! that row ordering is deterministic regardless of which worker finishes
//! first.

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;

use super::VecSource;

/// Detected element type from the npy descriptor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    F16, // <f2
    F32, // <f4
    F64, // <f8
}

impl NpyDtype {
    fn from_descr(descr: &str) -> Option<Self> {
        match descr {
            "<f2" => Some(Self::F16),
            "<f4" => Some(Self::F32),
            "<f8" => Some(Self::F64),
            _ => None,
        }
    }

    fn element_size(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Parsed npy file header (dtype, shape) without loading data
#[derive(Debug)]
struct NpyHeader {
    dtype: NpyDtype,
    rows: usize,
    cols: usize,
}

/// Maximum number of loaded files held in memory at once.
///
/// This bounds memory pressure independently of the thread count. Workers
/// beyond this limit simply wait for a slot — the large thread pool still
/// ensures one is always ready to start loading the moment a slot frees.
const MAX_IN_FLIGHT: usize = 8;

/// Maximum prefetch depth the adaptive algorithm will grow to
const MAX_PREFETCH: usize = 16;

/// Shared state between the consumer and loader pool for adaptive backpressure.
///
/// Workers block when total in-flight items reach `allowed`. When the consumer
/// finds itself waiting (a "short"), it bumps `allowed` and wakes workers.
struct PrefetchGate {
    /// How many un-consumed items all workers may have in flight combined
    allowed: AtomicUsize,
    /// How many items workers have sent but the consumer hasn't consumed
    in_flight: Mutex<usize>,
    /// Wakes workers when in_flight drops or allowed increases
    can_send: Condvar,
}

/// A loaded npy file tagged with its position in the sorted file list so the
/// consumer can reassemble results in order.
struct IndexedFile {
    /// Position in the original sorted file list
    index: usize,
    /// Loaded array data
    data: NpyArrayData,
}

/// Shared work queue: workers atomically claim the next file index to load.
struct WorkQueue {
    /// Next file index to hand out
    next: AtomicUsize,
    /// Total number of files
    total: usize,
    /// All file paths (indexed by position)
    files: Vec<PathBuf>,
    /// Element type for all files
    dtype: NpyDtype,
}

impl WorkQueue {
    /// Claim the next file to load. Returns `None` when all files are claimed.
    fn next_job(&self) -> Option<(usize, &Path)> {
        let idx = self.next.fetch_add(1, Ordering::Relaxed);
        if idx < self.total {
            Some((idx, &self.files[idx]))
        } else {
            None
        }
    }
}

/// Loaded npy array data, polymorphic over element type
pub enum NpyArrayData {
    F16 { data: Vec<u8>, rows: usize, cols: usize },
    F32(Array2<f32>),
    F64(Array2<f64>),
}

// NpyArrayData is Send — Vec<u8> and Array2<T> are Send
unsafe impl Send for NpyArrayData {}

impl NpyArrayData {
    fn nrows(&self) -> usize {
        match self {
            Self::F16 { rows, .. } => *rows,
            Self::F32(a) => a.nrows(),
            Self::F64(a) => a.nrows(),
        }
    }

    /// Extract row as raw LE bytes.
    ///
    /// Uses zero-copy slicing where possible: F16 data is already raw bytes,
    /// and F32/F64 arrays in standard layout on little-endian platforms have
    /// their elements contiguous in wire format. A single `to_vec()` memcpy
    /// at the trait boundary produces the owned `Vec<u8>`.
    pub fn row_bytes(&self, row: usize) -> Vec<u8> {
        match self {
            Self::F16 { data, cols, .. } => {
                let start = row * cols * 2;
                let end = start + cols * 2;
                data[start..end].to_vec()
            }
            Self::F32(a) => {
                // On LE platforms with standard layout, the f32 slice is
                // already in wire format — cast to &[u8] and copy once.
                if cfg!(target_endian = "little") {
                    if let Some(slice) = a.as_slice() {
                        let cols = a.ncols();
                        let start = row * cols;
                        let end = start + cols;
                        let f32_row = &slice[start..end];
                        let byte_slice = unsafe {
                            std::slice::from_raw_parts(
                                f32_row.as_ptr() as *const u8,
                                f32_row.len() * 4,
                            )
                        };
                        return byte_slice.to_vec();
                    }
                }
                // Fallback: non-LE or non-contiguous layout
                a.row(row).iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            Self::F64(a) => {
                if cfg!(target_endian = "little") {
                    if let Some(slice) = a.as_slice() {
                        let cols = a.ncols();
                        let start = row * cols;
                        let end = start + cols;
                        let f64_row = &slice[start..end];
                        let byte_slice = unsafe {
                            std::slice::from_raw_parts(
                                f64_row.as_ptr() as *const u8,
                                f64_row.len() * 8,
                            )
                        };
                        return byte_slice.to_vec();
                    }
                }
                a.row(row).iter().flat_map(|f| f.to_le_bytes()).collect()
            }
        }
    }
}

/// Result of scanning npy file headers (no data loaded)
pub struct NpyScanResult {
    pub files: Vec<std::path::PathBuf>,
    pub dtype: NpyDtype,
    pub dimension: u32,
    pub total_rows: u64,
}

/// Per-file manifest entry for parallel conversion.
pub struct NpyFileManifest {
    pub path: std::path::PathBuf,
    pub rows: u64,
    /// Cumulative row offset (first record ordinal in this file).
    pub offset: u64,
}

/// Scan npy files and return a manifest with per-file row counts and offsets.
///
/// Used by the parallel mmap converter to compute write positions.
pub fn scan_npy_manifest(path: &Path) -> Result<(Vec<NpyFileManifest>, u32, u64), String> {
    let scan = scan_npy_headers(path, None)?;
    let mut manifest = Vec::with_capacity(scan.files.len());
    let mut offset = 0u64;
    for file in &scan.files {
        let header = read_npy_header(file)?;
        let rows = header.rows as u64;
        manifest.push(NpyFileManifest {
            path: file.clone(),
            rows,
            offset,
        });
        offset += rows;
    }
    Ok((manifest, scan.dimension, scan.total_rows))
}

/// Scan a directory of npy files: collect sorted paths, validate headers,
/// and compute total row count. Only reads headers (a few hundred bytes each).
pub fn scan_npy_headers(path: &Path, max_count: Option<u64>) -> Result<NpyScanResult, String> {
    let files: Vec<_> = if path.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(".npy")
            })
            .map(|e| e.path())
            .collect();
        entries.sort();
        entries
    } else {
        vec![path.to_path_buf()]
    };

    if files.is_empty() {
        return Err(format!("No .npy files found in {}", path.display()));
    }

    let first_header = read_npy_header(&files[0])?;
    let dtype = first_header.dtype;
    let dimension = first_header.cols as u32;
    let mut total_rows = first_header.rows as u64;

    // When max_count is set, only scan enough shards to cover the limit
    let mut scanned_files = 1usize; // first file already scanned
    let have_enough = max_count.map_or(false, |mc| total_rows >= mc);

    if !have_enough {
        for f in &files[1..] {
            let header = read_npy_header(f)?;
            if header.dtype != dtype {
                return Err(format!(
                    "Dtype mismatch in {}: expected {:?}, got {:?}",
                    f.display(),
                    dtype,
                    header.dtype
                ));
            }
            if header.cols as u32 != dimension {
                return Err(format!(
                    "Dimension mismatch in {}: expected {}, got {}",
                    f.display(),
                    dimension,
                    header.cols
                ));
            }
            total_rows += header.rows as u64;
            scanned_files += 1;
            if let Some(mc) = max_count {
                if total_rows >= mc {
                    break;
                }
            }
        }
    }

    // Only include the files we actually need
    let files = if scanned_files < files.len() {
        files[..scanned_files].to_vec()
    } else {
        files
    };

    Ok(NpyScanResult { files, dtype, dimension, total_rows })
}

/// Reads a directory of `.npy` files containing 2D float arrays.
///
/// Supports f16 (`<f2`), f32 (`<f4`), and f64 (`<f8`) element types.
/// A pool of workers loads files in parallel with adaptive backpressure.
/// Results are reassembled in strict file-sorted order using a reorder
/// buffer, so row ordering is deterministic regardless of which worker
/// finishes first.
pub struct NpyDirReader {
    /// Channel receiving loaded files from the worker pool
    receiver: mpsc::Receiver<IndexedFile>,
    /// Shared prefetch gate for adaptive backpressure
    gate: Arc<PrefetchGate>,
    /// Reorder buffer: files that arrived out of order, keyed by file index
    reorder_buf: HashMap<usize, NpyArrayData>,
    /// The file index we need next to maintain sorted order
    next_file_index: usize,
    /// Total number of files being loaded
    total_files: usize,
    /// Currently-consumed array data
    current_array: Option<NpyArrayData>,
    /// Row index within the current array
    current_row: usize,
    /// Vector dimension (columns per row)
    dimension: u32,
    /// Total record count across all files
    record_count: Option<u64>,
    /// Element type
    dtype: NpyDtype,
    /// Worker thread handles
    _workers: Vec<thread::JoinHandle<()>>,
}

impl NpyDirReader {
    /// Probe a directory of npy files for metadata without starting a loader.
    ///
    /// Only reads file headers — no data is loaded, no background threads
    /// are spawned. Used for fail-fast validation in the import probe phase.
    pub fn probe(path: &Path) -> Result<super::SourceMeta, String> {
        let scan = scan_npy_headers(path, None)?;
        log::info!(
            "    {} npy file(s), dtype {:?}, {} bytes/element, dimension {}",
            scan.files.len(),
            scan.dtype,
            scan.dtype.element_size(),
            scan.dimension
        );
        Ok(super::SourceMeta {
            dimension: scan.dimension,
            element_size: scan.dtype.element_size(),
            record_count: Some(scan.total_rows),
        })
    }

    /// Open a directory of `.npy` files as a vector source.
    ///
    /// Reads file headers to determine dtype, dimension, and total row count,
    /// then spawns a pool of worker threads that load files in parallel with
    /// adaptive backpressure. Results are reassembled in file-sorted order.
    ///
    /// `threads` controls the number of loader threads. Pass `0` to
    /// auto-detect: hardware thread count, capped at 64.
    pub fn open(path: &Path, threads: usize, max_count: Option<u64>) -> Result<Box<dyn VecSource>, String> {
        let scan = scan_npy_headers(path, max_count)?;

        log::info!(
            "    {} npy file(s), dtype {:?}, {} bytes/element, dimension {}",
            scan.files.len(),
            scan.dtype,
            scan.dtype.element_size(),
            scan.dimension
        );

        let NpyScanResult { files, dtype, dimension, total_rows } = scan;

        let total_files = files.len();
        let default_threads = {
            let hw = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            hw.max(1).min(64)
        };
        let num_workers = if threads == 0 { default_threads } else { threads }
            .min(total_files);

        let initial_in_flight = MAX_IN_FLIGHT.min(total_files);
        log::info!(
            "npy: {} file(s), {} total rows, {} loader thread(s), {} max in-flight",
            total_files, total_rows, num_workers, initial_in_flight
        );

        // Shared work queue and prefetch gate
        let queue = Arc::new(WorkQueue {
            next: AtomicUsize::new(0),
            total: total_files,
            files,
            dtype,
        });
        let gate = Arc::new(PrefetchGate {
            allowed: AtomicUsize::new(MAX_IN_FLIGHT.min(total_files)),
            in_flight: Mutex::new(0),
            can_send: Condvar::new(),
        });

        // Spawn worker pool
        let (tx, rx) = mpsc::channel::<IndexedFile>();
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let queue = Arc::clone(&queue);
            let gate = Arc::clone(&gate);
            let tx = tx.clone();
            workers.push(thread::spawn(move || {
                loop {
                    // Backpressure FIRST: acquire a slot before claiming a file.
                    // This ensures files are claimed in strict index order,
                    // preventing priority inversion where out-of-order files
                    // consume all in-flight slots while the needed file's
                    // worker is stuck behind backpressure.
                    {
                        let mut in_flight = gate.in_flight.lock().unwrap();
                        while *in_flight >= gate.allowed.load(Ordering::Relaxed) {
                            in_flight = gate.can_send.wait(in_flight).unwrap();
                        }
                        *in_flight += 1;
                    }

                    // Now claim the next file — guaranteed to be the lowest
                    // unclaimed index since all workers gate before claiming.
                    let Some((idx, path)) = queue.next_job() else {
                        // No more files — release the slot we just acquired
                        let mut in_flight = gate.in_flight.lock().unwrap();
                        *in_flight -= 1;
                        gate.can_send.notify_one();
                        break;
                    };

                    match load_npy(path, queue.dtype) {
                        Ok(array) => {
                            if tx.send(IndexedFile { index: idx, data: array }).is_err() {
                                break; // consumer dropped
                            }
                        }
                        Err(e) => {
                            log::info!("Warning: failed to load {}: {}", path.display(), e);
                            let mut in_flight = gate.in_flight.lock().unwrap();
                            *in_flight -= 1;
                            gate.can_send.notify_one();
                        }
                    }
                }
            }));
        }
        // Drop the sender held by open() so the channel closes when all workers finish
        drop(tx);

        Ok(Box::new(NpyDirReader {
            receiver: rx,
            gate,
            reorder_buf: HashMap::new(),
            next_file_index: 0,
            total_files,
            current_array: None,
            current_row: 0,
            dimension,
            record_count: Some(total_rows),
            dtype,
            _workers: workers,
        }))
    }

    /// Release one in-flight slot and wake one worker.
    fn release_in_flight(&self) {
        let mut in_flight = self.gate.in_flight.lock().unwrap();
        *in_flight = in_flight.saturating_sub(1);
        self.gate.can_send.notify_one();
    }

    /// Try to advance to the next file in sorted order.
    ///
    /// First checks the reorder buffer for the next expected file index. If
    /// not buffered, receives from the channel — buffering any out-of-order
    /// results — until the needed file arrives or the channel closes.
    fn advance_to_next_file(&mut self) -> bool {
        if self.next_file_index >= self.total_files {
            return false;
        }

        // Release the previous file's in-flight slot
        if self.current_array.is_some() {
            self.release_in_flight();
            self.current_array = None;
        }

        // Check reorder buffer first
        if let Some(data) = self.reorder_buf.remove(&self.next_file_index) {
            self.current_array = Some(data);
            self.current_row = 0;
            self.next_file_index += 1;
            return true;
        }

        // Try non-blocking first to detect "shorts"
        match self.receiver.try_recv() {
            Ok(loaded) => {
                if self.accept_loaded(loaded) {
                    return true;
                }
                // Out of order — stashed, fall through to block-receive
            }
            Err(mpsc::TryRecvError::Empty) => {
                // Consumer outran the workers — bump prefetch depth
                let old = self.gate.allowed.load(Ordering::Relaxed);
                if old < MAX_PREFETCH {
                    self.gate.allowed.store(old + 1, Ordering::Relaxed);
                    self.gate.can_send.notify_all();
                    log::info!(
                        "npy: prefetch short — increased depth to {}",
                        old + 1
                    );
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => return false,
        }

        // Block-receive until we get the file we need, checking the reorder
        // buffer after each stash in case the needed file was already buffered
        // from the try_recv above.
        loop {
            if let Some(data) = self.reorder_buf.remove(&self.next_file_index) {
                self.current_array = Some(data);
                self.current_row = 0;
                self.next_file_index += 1;
                return true;
            }
            match self.receiver.recv() {
                Ok(loaded) => {
                    if self.accept_loaded(loaded) {
                        return true;
                    }
                }
                Err(_) => return false,
            }
        }
    }

    /// Process a received `IndexedFile`: if it's the one we need, install it
    /// as current; otherwise stash it in the reorder buffer and return false.
    fn accept_loaded(&mut self, loaded: IndexedFile) -> bool {
        if loaded.index == self.next_file_index {
            self.current_array = Some(loaded.data);
            self.current_row = 0;
            self.next_file_index += 1;
            true
        } else {
            self.reorder_buf.insert(loaded.index, loaded.data);
            false
        }
    }
}

impl VecSource for NpyDirReader {
    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn element_size(&self) -> usize {
        self.dtype.element_size()
    }

    fn record_count(&self) -> Option<u64> {
        self.record_count
    }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        loop {
            if let Some(ref array) = self.current_array {
                if self.current_row < array.nrows() {
                    let bytes = array.row_bytes(self.current_row);
                    self.current_row += 1;
                    return Some(bytes);
                }
            }

            // Need next file in sorted order
            if !self.advance_to_next_file() {
                return None;
            }
        }
    }
}

/// Read only the npy file header to extract dtype and shape, without loading data
fn read_npy_header(path: &Path) -> Result<NpyHeader, String> {
    let mut file = BufReader::new(
        fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?,
    );

    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|e| format!("Failed to read npy magic in {}: {}", path.display(), e))?;
    if &magic != b"\x93NUMPY" {
        return Err(format!("{} is not a valid npy file", path.display()));
    }

    let major = file.read_u8().map_err(|e| format!("Read error in {}: {}", path.display(), e))?;
    let _minor = file.read_u8().map_err(|e| format!("Read error in {}: {}", path.display(), e))?;

    let header_len = if major >= 2 {
        file.read_u32::<LittleEndian>()
            .map_err(|e| format!("Read error in {}: {}", path.display(), e))? as usize
    } else {
        file.read_u16::<LittleEndian>()
            .map_err(|e| format!("Read error in {}: {}", path.display(), e))? as usize
    };

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header in {}: {}", path.display(), e))?;
    let header = String::from_utf8_lossy(&header_bytes);

    let descr = extract_descr(&header)
        .ok_or_else(|| format!("Could not parse 'descr' from npy header in {}", path.display()))?;
    let dtype = NpyDtype::from_descr(&descr).ok_or_else(|| {
        format!(
            "Unsupported npy dtype '{}' in {}. Supported: <f2, <f4, <f8",
            descr,
            path.display()
        )
    })?;

    let (rows, cols) = extract_shape(&header).ok_or_else(|| {
        format!("Could not parse shape from npy header in {}", path.display())
    })?;

    Ok(NpyHeader { dtype, rows, cols })
}

/// Extract the descr value from a npy header string
fn extract_descr(header: &str) -> Option<String> {
    let descr_idx = header.find("'descr'")?;
    let after = &header[descr_idx..];
    let colon_idx = after.find(':')?;
    let after_colon = &after[colon_idx + 1..];
    let quote_start = after_colon.find('\'')?;
    let rest = &after_colon[quote_start + 1..];
    let quote_end = rest.find('\'')?;
    Some(rest[..quote_end].to_string())
}

/// Extract the shape tuple from a npy header string
fn extract_shape(header: &str) -> Option<(usize, usize)> {
    let shape_idx = header.find("'shape'")?;
    let after = &header[shape_idx..];
    let paren_start = after.find('(')?;
    let paren_end = after.find(')')?;
    let inside = &after[paren_start + 1..paren_end];
    let parts: Vec<&str> = inside.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if parts.len() == 2 {
        let rows = parts[0].parse().ok()?;
        let cols = parts[1].parse().ok()?;
        Some((rows, cols))
    } else {
        None
    }
}

/// Load an npy file with the given dtype
pub fn load_npy(path: &Path, dtype: NpyDtype) -> Result<NpyArrayData, String> {
    match dtype {
        NpyDtype::F32 => {
            let file = fs::File::open(path)
                .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
            let reader = BufReader::new(file);
            let array = Array2::<f32>::read_npy(reader)
                .map_err(|e| format!("Failed to read npy {}: {}", path.display(), e))?;
            Ok(NpyArrayData::F32(array))
        }
        NpyDtype::F64 => {
            let file = fs::File::open(path)
                .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
            let reader = BufReader::new(file);
            let array = Array2::<f64>::read_npy(reader)
                .map_err(|e| format!("Failed to read npy {}: {}", path.display(), e))?;
            Ok(NpyArrayData::F64(array))
        }
        NpyDtype::F16 => load_npy_f16_raw(path),
    }
}

/// Load an f16 npy file by reading raw bytes (ndarray-npy doesn't support f16)
pub(crate) fn load_npy_f16_raw(path: &Path) -> Result<NpyArrayData, String> {
    let header = read_npy_header(path)?;

    let mut file = BufReader::new(
        fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?,
    );

    // Skip past the header to the data
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic).map_err(|e| format!("Read error: {}", e))?;
    let major = file.read_u8().map_err(|e| format!("Read error: {}", e))?;
    let _minor = file.read_u8().map_err(|e| format!("Read error: {}", e))?;
    let header_len = if major >= 2 {
        file.read_u32::<LittleEndian>().map_err(|e| format!("Read error: {}", e))? as usize
    } else {
        file.read_u16::<LittleEndian>().map_err(|e| format!("Read error: {}", e))? as usize
    };
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|e| format!("Read error: {}", e))?;

    // Read the raw data
    let data_len = header.rows * header.cols * 2;
    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)
        .map_err(|e| format!("Failed to read f16 data from {}: {}", path.display(), e))?;

    Ok(NpyArrayData::F16 { data, rows: header.rows, cols: header.cols })
}
