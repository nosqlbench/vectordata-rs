// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Parquet metadata reader — reads scalar parquet columns and emits MNode wire
//! format bytes via [`CompiledMnodeWriter`].
//!
//! Unlike [`ParquetDirReader`](super::parquet::ParquetDirReader), which requires
//! a list-of-float vector column, this reader handles parquet files with
//! arbitrary scalar columns (strings, ints, timestamps, etc.) and converts each
//! row to an unframed MNode payload.
//!
//! A configurable pool of background workers loads parquet files in
//! parallel while an adaptive prefetch gate controls how far ahead they may
//! read. Results are reassembled in strict file-sorted order by the consumer
//! so that row ordering is deterministic regardless of which worker finishes
//! first.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{SourceMeta, VecSource};
use crate::formats::mnode::parquet_compiler::CompiledMnodeWriter;

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

/// Reads a directory of `.parquet` files and emits each row as an unframed
/// MNode payload (raw wire bytes).
///
/// A pool of workers loads files in parallel. Each worker
/// pulls the next unloaded file from a shared [`WorkQueue`], loads it, and
/// sends the result (tagged with its file index) through a channel. The
/// consumer reassembles results in sorted order using a small reorder buffer.
pub struct ParquetMnodeReader {
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
    /// Compiled plan for converting Arrow rows to MNode bytes
    writer: CompiledMnodeWriter,
    /// Reusable buffer for MNode serialization
    buf: Vec<u8>,
    /// Total row count across all files
    record_count: u64,
    /// Worker thread handles
    _workers: Vec<thread::JoinHandle<()>>,
}

impl ParquetMnodeReader {
    /// Probe a parquet source for metadata without starting a full reader.
    ///
    /// Reads the schema from the first file, compiles an MNode writer to
    /// validate that all columns are supported, and counts total rows across
    /// all files via parquet metadata.
    pub fn probe(path: &Path) -> Result<SourceMeta, String> {
        let files = collect_parquet_files(path)?;
        log::info!("    {} parquet file(s), reading metadata...", files.len());
        let total_rows = count_total_rows(&files)?;

        // Validate schema by compiling a writer from the first file
        let schema = read_schema(&files[0])?;
        CompiledMnodeWriter::compile(&schema)?;

        Ok(SourceMeta {
            dimension: 0,
            element_size: 1,
            record_count: Some(total_rows),
        })
    }

    /// Open a parquet source as a metadata (MNode) reader.
    ///
    /// Reads the schema from the first file, compiles a
    /// [`CompiledMnodeWriter`], counts total rows, then spawns a pool of
    /// worker threads that load files in parallel with adaptive
    /// backpressure. Results are reassembled in file-sorted order.
    ///
    /// `threads` controls the number of loader threads. Pass `0` to
    /// auto-detect: hardware thread count, capped at 64.
    pub fn open(path: &Path, threads: usize) -> Result<Box<dyn VecSource>, String> {
        let files = collect_parquet_files(path)?;
        let total_rows = count_total_rows(&files)?;

        // Compile the MNode writer from the first file's schema
        let schema = read_schema(&files[0])?;
        let writer = CompiledMnodeWriter::compile(&schema)?;

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
            "parquet-mnode: {} file(s), {} total rows, {} loader thread(s), {} max in-flight",
            total_files, total_rows, num_workers, initial_in_flight
        );

        // Shared work queue and prefetch gate
        let queue = Arc::new(WorkQueue {
            next: AtomicUsize::new(0),
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

        Ok(Box::new(ParquetMnodeReader {
            receiver: rx,
            gate,
            reorder_buf: HashMap::new(),
            next_file_index: 0,
            total_files,
            current_batches: Vec::new(),
            current_batch_idx: 0,
            current_row: 0,
            writer,
            buf: Vec::new(),
            record_count: total_rows,
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
                        "parquet-mnode: prefetch short — increased depth to {}",
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

impl VecSource for ParquetMnodeReader {
    /// Returns 0 — metadata records have no fixed vector dimension.
    fn dimension(&self) -> u32 {
        0
    }

    /// Returns 1 — metadata records are variable-length byte payloads.
    ///
    /// A value of 1 avoids multiply-by-zero in slab's record byte length
    /// calculation and is semantically "byte-oriented".
    fn element_size(&self) -> usize {
        1
    }

    fn record_count(&self) -> Option<u64> {
        Some(self.record_count)
    }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        loop {
            // Try current batch
            if self.current_batch_idx < self.current_batches.len() {
                let batch = &self.current_batches[self.current_batch_idx];
                if self.current_row < batch.num_rows() {
                    self.buf.clear();
                    self.writer.write_row(batch, self.current_row, &mut self.buf);
                    self.current_row += 1;
                    return Some(self.buf.clone());
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

/// Read the Arrow schema from a parquet file without loading data.
fn read_schema(path: &Path) -> Result<Schema, String> {
    let file = fs::File::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to read parquet {}: {}", path.display(), e))?;
    Ok(builder.schema().as_ref().clone())
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

/// Collect `.parquet` files from a path (single file or directory), sorted.
fn collect_parquet_files(path: &Path) -> Result<Vec<PathBuf>, String> {
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

/// Count total rows across all parquet files using file-level metadata.
///
/// For directories with more than 10 files, prints periodic progress to
/// stderr so the user can see that work is happening.
fn count_total_rows(files: &[PathBuf]) -> Result<u64, String> {
    let mut total = 0u64;
    let n = files.len();
    // Report every ~10% but at least every 20 files, and always for >10 files
    let report_interval = if n > 10 { (n / 10).max(1).min(20) } else { n + 1 };
    if n > 10 {
        log::info!("    counting rows across {} parquet files...", n);
    }
    for (i, f) in files.iter().enumerate() {
        let file = fs::File::open(f)
            .map_err(|e| format!("Failed to open {}: {}", f.display(), e))?;
        let pq_reader = parquet::file::reader::SerializedFileReader::new(file)
            .map_err(|e| format!("Failed to read parquet metadata for {}: {}", f.display(), e))?;
        use parquet::file::reader::FileReader;
        total += pq_reader.metadata().file_metadata().num_rows() as u64;
        if (i + 1) % report_interval == 0 {
            log::info!("    scanned {}/{} files, {} rows so far", i + 1, n, total);
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mnode::MNode;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Write a simple two-column parquet file for testing.
    fn write_test_parquet(dir: &Path, filename: &str, names: &[&str], ages: &[i64]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(names.to_vec())),
                Arc::new(Int64Array::from(ages.to_vec())),
            ],
        )
        .unwrap();

        let path = dir.join(filename);
        let file = fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn test_probe_returns_correct_metadata() {
        let tmp = TempDir::new().unwrap();
        write_test_parquet(tmp.path(), "part-0.parquet", &["alice", "bob"], &[30, 25]);
        write_test_parquet(tmp.path(), "part-1.parquet", &["carol"], &[40]);

        let meta = ParquetMnodeReader::probe(tmp.path()).unwrap();
        assert_eq!(meta.dimension, 0);
        assert_eq!(meta.element_size, 1);
        assert_eq!(meta.record_count, Some(3));
    }

    #[test]
    fn test_open_reads_all_rows_as_mnode() {
        let tmp = TempDir::new().unwrap();
        write_test_parquet(tmp.path(), "part-0.parquet", &["alice", "bob"], &[30, 25]);
        write_test_parquet(tmp.path(), "part-1.parquet", &["carol"], &[40]);

        let mut reader = ParquetMnodeReader::open(tmp.path(), 0).unwrap();
        assert_eq!(reader.dimension(), 0);
        assert_eq!(reader.element_size(), 1);
        assert_eq!(reader.record_count(), Some(3));

        let mut records = Vec::new();
        while let Some(data) = reader.next_record() {
            let node = MNode::from_bytes(&data).unwrap();
            records.push(node);
        }
        assert_eq!(records.len(), 3);

        use crate::formats::mnode::MValue;
        assert_eq!(records[0].fields["name"], MValue::Text("alice".into()));
        assert_eq!(records[0].fields["age"], MValue::Int(30));
        assert_eq!(records[2].fields["name"], MValue::Text("carol".into()));
        assert_eq!(records[2].fields["age"], MValue::Int(40));
    }

    #[test]
    fn test_single_file_mode() {
        let tmp = TempDir::new().unwrap();
        write_test_parquet(tmp.path(), "data.parquet", &["x"], &[99]);

        let file_path = tmp.path().join("data.parquet");
        let mut reader = ParquetMnodeReader::open(&file_path, 0).unwrap();
        assert_eq!(reader.record_count(), Some(1));

        let data = reader.next_record().unwrap();
        let node = MNode::from_bytes(&data).unwrap();
        use crate::formats::mnode::MValue;
        assert_eq!(node.fields["name"], MValue::Text("x".into()));
        assert_eq!(node.fields["age"], MValue::Int(99));

        assert!(reader.next_record().is_none());
    }

    #[test]
    fn test_many_files_ordering_preserved() {
        let tmp = TempDir::new().unwrap();
        // Create more files than worker threads to exercise the pool +
        // reorder buffer
        for i in 0..12 {
            write_test_parquet(
                tmp.path(),
                &format!("part-{:02}.parquet", i),
                &[&format!("name-{}", i)],
                &[i as i64],
            );
        }

        let mut reader = ParquetMnodeReader::open(tmp.path(), 0).unwrap();
        assert_eq!(reader.record_count(), Some(12));

        let mut records = Vec::new();
        while let Some(data) = reader.next_record() {
            let node = MNode::from_bytes(&data).unwrap();
            records.push(node);
        }
        assert_eq!(records.len(), 12);

        // Verify strict file-sorted ordering despite parallel loading
        use crate::formats::mnode::MValue;
        for (i, node) in records.iter().enumerate() {
            assert_eq!(
                node.fields["name"],
                MValue::Text(format!("name-{}", i)),
                "ordering broken at position {}",
                i
            );
            assert_eq!(node.fields["age"], MValue::Int(i as i64));
        }
    }

    #[test]
    fn test_multi_row_multi_file_parallel() {
        let tmp = TempDir::new().unwrap();
        // Multiple rows per file × many files to stress the thread pool
        for i in 0..8 {
            let names: Vec<&str> = vec!["a", "b", "c"];
            let ages: Vec<i64> = vec![i * 10, i * 10 + 1, i * 10 + 2];
            write_test_parquet(
                tmp.path(),
                &format!("part-{:02}.parquet", i),
                &names,
                &ages,
            );
        }

        let mut reader = ParquetMnodeReader::open(tmp.path(), 0).unwrap();
        assert_eq!(reader.record_count(), Some(24));

        let mut count = 0;
        let mut prev_file_age_base: Option<i64> = None;
        while let Some(data) = reader.next_record() {
            let node = MNode::from_bytes(&data).unwrap();
            use crate::formats::mnode::MValue;
            if let MValue::Int(age) = &node.fields["age"] {
                let file_base = (*age / 10) * 10;
                // Within a file, ages should be monotonically increasing
                // Across files, the file base should be non-decreasing
                if let Some(prev) = prev_file_age_base {
                    assert!(
                        file_base >= prev,
                        "file ordering broken: prev base {}, current base {}",
                        prev,
                        file_base
                    );
                }
                prev_file_age_base = Some(file_base);
            }
            count += 1;
        }
        assert_eq!(count, 24);
    }
}
