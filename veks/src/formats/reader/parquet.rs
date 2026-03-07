// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Parquet vector reader — reads directories of `.parquet` files containing
//! list-of-float vector columns and emits raw LE vector bytes.
//!
//! A configurable pool of background workers loads parquet files in parallel
//! while an adaptive prefetch gate controls how far ahead they may read.
//! Results are reassembled in strict file-sorted order by the consumer so
//! that row ordering is deterministic regardless of which worker finishes
//! first.
//!
//! On little-endian platforms, vector rows are extracted via zero-copy from
//! the Arrow `Float32Array` backing buffer, avoiding per-element conversion.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use arrow::array::{Array, Float32Array, ListArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{SourceMeta, VecSource};

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

/// A loaded file tagged with its position in the sorted file list so the
/// consumer can reassemble results in order.
struct IndexedFile {
    /// Position in the original sorted file list
    index: usize,
    /// All record batches from this file
    batches: Vec<RecordBatch>,
}

/// Shared work queue: workers atomically claim the next file index to load.
struct WorkQueue {
    /// Next file index to hand out
    next: AtomicUsize,
    /// Total number of files
    total: usize,
    /// All file paths (indexed by position)
    files: Vec<PathBuf>,
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

/// Reads a directory of `.parquet` files containing vector data.
///
/// Expects each parquet file to have a column containing a list/array of floats.
/// The first list-of-float column found is used as the vector data.
///
/// A pool of workers loads files in parallel. Each worker pulls the next
/// unloaded file from a shared [`WorkQueue`], loads it, and sends the result
/// (tagged with its file index) through a channel. The consumer reassembles
/// results in sorted order using a small reorder buffer.
pub struct ParquetDirReader {
    /// Channel receiving loaded files from the worker pool
    receiver: mpsc::Receiver<IndexedFile>,
    /// Shared prefetch gate for adaptive backpressure
    gate: Arc<PrefetchGate>,
    /// Reorder buffer: files that arrived out of order, keyed by file index
    reorder_buf: HashMap<usize, Vec<RecordBatch>>,
    /// The file index we need next to maintain sorted order
    next_file_index: usize,
    /// Total number of files being loaded
    total_files: usize,
    /// Batches from the currently-consumed file
    current_batches: Vec<RecordBatch>,
    /// Index into `current_batches`
    current_batch_idx: usize,
    /// Row index within the current batch
    current_row: usize,
    /// Index of the vector column in the parquet schema
    vector_col_idx: usize,
    /// Vector dimension (number of elements per record)
    dimension: u32,
    /// Total record count across all files
    record_count: Option<u64>,
    /// Worker thread handles
    _workers: Vec<thread::JoinHandle<()>>,
}

/// Metadata from probing a parquet source.
///
/// Unlike `SourceMeta`, this always succeeds regardless of whether the
/// parquet files contain vector columns. It reports the schema, total
/// row count, and optionally the vector-specific metadata if a
/// list-of-float column is present.
pub struct ParquetProbeMeta {
    /// Human-readable Arrow schema
    pub schema: String,
    /// Total row count across all parquet files
    pub row_count: u64,
    /// Number of parquet files
    pub file_count: usize,
    /// Vector-specific metadata, if a list-of-float column was found
    pub vector_meta: Option<SourceMeta>,
}

impl ParquetDirReader {
    /// Probe a parquet source for metadata without requiring a vector column.
    ///
    /// Reads the Arrow schema from the first file, counts total rows across
    /// all files, and optionally extracts vector dimension if a list-of-float
    /// column exists.
    pub fn probe(path: &Path) -> Result<ParquetProbeMeta, String> {
        let files = collect_parquet_files(path)?;

        // Open first file for schema
        let first_file = fs::File::open(&files[0])
            .map_err(|e| format!("Failed to open {}: {}", files[0].display(), e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(first_file)
            .map_err(|e| format!("Failed to read parquet {}: {}", files[0].display(), e))?;

        let schema = builder.schema().clone();
        let schema_str = format_schema(&schema);

        // Check for vector column
        let vector_col = find_vector_column(&schema);

        // Get vector dimension if present (requires reading first batch)
        let (vector_meta, first_file_rows) = if let Some(col_idx) = vector_col {
            let reader = builder
                .build()
                .map_err(|e| format!("Failed to build parquet reader: {}", e))?;
            let batches: Vec<_> = reader
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to read batches: {}", e))?;

            let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
            let dim = batches
                .first()
                .map(|b| get_vector_dimension(b, col_idx))
                .transpose()?;

            let meta = dim.map(|d| SourceMeta {
                dimension: d,
                element_size: 4,
                record_count: None, // filled in below with total
            });
            (meta, rows)
        } else {
            // No vector column — still count rows from parquet metadata
            let first_file2 = fs::File::open(&files[0])
                .map_err(|e| format!("Failed to open {}: {}", files[0].display(), e))?;
            let pq_reader = parquet::file::reader::SerializedFileReader::new(first_file2)
                .map_err(|e| format!("Failed to read parquet metadata: {}", e))?;
            use parquet::file::reader::FileReader;
            let rows = pq_reader.metadata().file_metadata().num_rows() as u64;
            (None, rows)
        };

        // Count rows in remaining files
        let mut total_rows = first_file_rows;
        for f in &files[1..] {
            let file = fs::File::open(f)
                .map_err(|e| format!("Failed to open {}: {}", f.display(), e))?;
            let pq_reader = parquet::file::reader::SerializedFileReader::new(file)
                .map_err(|e| format!("Failed to read parquet metadata: {}", e))?;
            use parquet::file::reader::FileReader;
            total_rows += pq_reader.metadata().file_metadata().num_rows() as u64;
        }

        // Fill in total record count on vector meta
        let vector_meta = vector_meta.map(|mut m| {
            m.record_count = Some(total_rows);
            m
        });

        Ok(ParquetProbeMeta {
            schema: schema_str,
            row_count: total_rows,
            file_count: files.len(),
            vector_meta,
        })
    }

    /// Open a directory of `.parquet` files as a vector source.
    ///
    /// Reads the schema from the first file, determines the vector column
    /// and dimension, counts total rows, then spawns a pool of worker threads
    /// that load files in parallel with adaptive backpressure. Results are
    /// reassembled in file-sorted order.
    ///
    /// `threads` controls the number of loader threads. Pass `0` to
    /// auto-detect: hardware thread count, capped at 64.
    pub fn open(path: &Path, threads: usize) -> Result<Box<dyn VecSource>, String> {
        let files = collect_parquet_files(path)?;

        // Open first file to determine schema and dimension
        let first_file = fs::File::open(&files[0])
            .map_err(|e| format!("Failed to open {}: {}", files[0].display(), e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(first_file)
            .map_err(|e| format!("Failed to read parquet {}: {}", files[0].display(), e))?;

        let schema = builder.schema().clone();

        // Find the first list-of-float column
        let vector_col_idx = find_vector_column(&schema).ok_or_else(|| {
            "No list-of-float column found in parquet schema".to_string()
        })?;

        // Read first file's batches to determine dimension
        let reader = builder
            .build()
            .map_err(|e| format!("Failed to build parquet reader: {}", e))?;
        let first_batches: Vec<_> = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to read batches: {}", e))?;

        let dimension = if let Some(batch) = first_batches.first() {
            get_vector_dimension(batch, vector_col_idx)?
        } else {
            return Err("Empty parquet file".to_string());
        };

        // Count total rows across all files
        let mut total_rows: u64 = first_batches.iter().map(|b| b.num_rows() as u64).sum();
        for f in &files[1..] {
            let file = fs::File::open(f)
                .map_err(|e| format!("Failed to open {}: {}", f.display(), e))?;
            let metadata = parquet::file::reader::SerializedFileReader::new(file)
                .map_err(|e| format!("Failed to read parquet metadata: {}", e))?;
            use parquet::file::reader::FileReader;
            total_rows += metadata.metadata().file_metadata().num_rows() as u64;
        }

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
            "parquet: {} file(s), {} total rows, dim {}, {} loader thread(s), {} max in-flight",
            total_files, total_rows, dimension, num_workers, initial_in_flight
        );

        // Build the work queue starting at file index 1 — we already loaded
        // file 0 above and will inject its batches directly into the reader.
        let queue = Arc::new(WorkQueue {
            next: AtomicUsize::new(1),
            total: total_files,
            files,
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

                    match load_parquet_file(path) {
                        Ok(batches) => {
                            if tx.send(IndexedFile { index: idx, batches }).is_err() {
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

        Ok(Box::new(ParquetDirReader {
            receiver: rx,
            gate,
            reorder_buf: HashMap::new(),
            next_file_index: 0,
            total_files,
            current_batches: first_batches,
            current_batch_idx: 0,
            current_row: 0,
            vector_col_idx,
            dimension,
            record_count: Some(total_rows),
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
        if !self.current_batches.is_empty() {
            self.release_in_flight();
            self.current_batches.clear();
        }

        // Check reorder buffer first — the file's in-flight slot will be
        // released at the top of the NEXT advance_to_next_file call when
        // this file finishes being consumed.
        if let Some(batches) = self.reorder_buf.remove(&self.next_file_index) {
            self.current_batches = batches;
            self.current_batch_idx = 0;
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
                        "parquet: prefetch short — increased depth to {}",
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
            // Check reorder buffer — the file's in-flight slot will be
            // released at the top of the NEXT advance_to_next_file call.
            if let Some(batches) = self.reorder_buf.remove(&self.next_file_index) {
                self.current_batches = batches;
                self.current_batch_idx = 0;
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
            self.current_batches = loaded.batches;
            self.current_batch_idx = 0;
            self.current_row = 0;
            self.next_file_index += 1;
            true
        } else {
            self.reorder_buf.insert(loaded.index, loaded.batches);
            false
        }
    }
}

impl VecSource for ParquetDirReader {
    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn element_size(&self) -> usize {
        4 // parquet vector columns are f32
    }

    fn record_count(&self) -> Option<u64> {
        self.record_count
    }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        loop {
            // Try current batch
            if self.current_batch_idx < self.current_batches.len() {
                let batch = &self.current_batches[self.current_batch_idx];
                if self.current_row < batch.num_rows() {
                    let col = batch.column(self.vector_col_idx);
                    let list_array = col.as_any().downcast_ref::<ListArray>()?;
                    let values = list_array.value(self.current_row);
                    let float_array = values.as_any().downcast_ref::<Float32Array>()?;

                    let bytes = extract_f32_row_bytes(float_array);

                    self.current_row += 1;
                    return Some(bytes);
                }
                // Next batch in this file
                self.current_batch_idx += 1;
                self.current_row = 0;
                continue;
            }

            // Need next file in sorted order
            if !self.advance_to_next_file() {
                return None;
            }
        }
    }
}

/// Extract raw LE bytes from a `Float32Array` row.
///
/// On little-endian platforms, the Arrow backing buffer is already contiguous
/// LE bytes — we cast directly to `&[u8]` and copy once. On big-endian
/// platforms, falls back to per-element `to_le_bytes()`.
fn extract_f32_row_bytes(float_array: &Float32Array) -> Vec<u8> {
    if cfg!(target_endian = "little") {
        // Float32Array::values() returns a ScalarBuffer<f32> whose inner
        // Buffer is contiguous LE f32 values. Cast to &[u8] and copy once.
        let values = float_array.values();
        let byte_len = values.len() * 4;
        let byte_ptr = values.as_ptr() as *const u8;
        // SAFETY: ScalarBuffer<f32> is a contiguous allocation of f32 values.
        // On LE platforms, the in-memory layout matches the wire format.
        // We only read `values.len() * 4` bytes, which is within the
        // allocation.
        let byte_slice = unsafe { std::slice::from_raw_parts(byte_ptr, byte_len) };
        byte_slice.to_vec()
    } else {
        float_array
            .values()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }
}

/// Collect parquet files from a path (single file or directory), sorted.
fn collect_parquet_files(path: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let files: Vec<_> = if path.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".parquet"))
            .map(|e| e.path())
            .collect();
        entries.sort();
        entries
    } else {
        vec![path.to_path_buf()]
    };

    if files.is_empty() {
        Err(format!("No .parquet files found in {}", path.display()))
    } else {
        Ok(files)
    }
}

/// Load all record batches from a single parquet file.
fn load_parquet_file(path: &Path) -> Result<Vec<RecordBatch>, String> {
    let file = fs::File::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to read parquet {}: {}", path.display(), e))?;
    let reader = builder
        .build()
        .map_err(|e| format!("Failed to build parquet reader for {}: {}", path.display(), e))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to read batches from {}: {}", path.display(), e))
}

/// Format an Arrow schema as a human-readable string.
fn format_schema(schema: &Arc<arrow::datatypes::Schema>) -> String {
    let mut lines = Vec::new();
    for field in schema.fields() {
        lines.push(format!(
            "    {} : {} (nullable={})",
            field.name(),
            field.data_type(),
            field.is_nullable()
        ));
    }
    lines.join("\n")
}

/// Find the first column in the schema that is a list of floats
fn find_vector_column(schema: &Arc<arrow::datatypes::Schema>) -> Option<usize> {
    use arrow::datatypes::DataType;
    for (i, field) in schema.fields().iter().enumerate() {
        if let DataType::List(inner) = field.data_type() {
            if matches!(inner.data_type(), DataType::Float32) {
                return Some(i);
            }
        }
        // Also check FixedSizeList
        if let DataType::FixedSizeList(inner, _) = field.data_type() {
            if matches!(inner.data_type(), DataType::Float32) {
                return Some(i);
            }
        }
    }
    None
}

/// Get vector dimension from the first row of a batch
fn get_vector_dimension(
    batch: &arrow::record_batch::RecordBatch,
    col_idx: usize,
) -> Result<u32, String> {
    use arrow::datatypes::DataType;
    let col = batch.column(col_idx);

    // Check if it's a FixedSizeList (dimension is in the type)
    if let DataType::FixedSizeList(_, size) = col.data_type() {
        return Ok(*size as u32);
    }

    // For List, check the first row
    let list_array = col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or("Column is not a ListArray")?;
    if list_array.is_empty() {
        return Err("Empty batch, cannot determine dimension".to_string());
    }
    Ok(list_array.value(0).len() as u32)
}
