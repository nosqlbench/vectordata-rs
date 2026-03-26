// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Memory-mapped xvec writer for concurrent random-access writes.
//!
//! Pre-allocates the output file to the known total size, memory-maps it,
//! and allows multiple threads to write records at computed byte offsets
//! without coordination. Each record is `4 bytes (dim as i32) + dim * element_size`.
//!
//! This is safe because each thread writes to a disjoint byte range
//! determined by the record ordinal.

use std::path::Path;
use std::sync::Arc;

use memmap2::MmapMut;

/// A concurrent xvec writer backed by a memory-mapped file.
///
/// Multiple threads can call `write_record_at` simultaneously for
/// different ordinals — each write targets a non-overlapping byte range.
pub struct MmapXvecWriter {
    mmap: MmapMut,
    /// Bytes per record: 4 (dim header) + dim * element_size.
    record_stride: usize,
    /// Dimension written into each record header.
    dimension: u32,
    /// Total records the file was sized for.
    total_records: u64,
}

/// Thread-safe handle for concurrent writes.
///
/// Wraps the writer in an `Arc` so multiple threads can hold references.
/// Safety: writes to disjoint byte ranges are safe with `MmapMut` because
/// each thread writes to a unique ordinal range.
pub struct SharedMmapWriter {
    inner: Arc<MmapXvecWriter>,
}

// MmapMut is Send but not Sync. We guarantee disjoint access by ordinal,
// so sharing across threads is safe.
unsafe impl Sync for SharedMmapWriter {}

impl MmapXvecWriter {
    /// Create a new mmap writer.
    ///
    /// Pre-allocates the file to `total_records * record_stride` bytes
    /// and memory-maps it for random-access writes. Advises the kernel
    /// for random access patterns and eager writeback.
    pub fn create(
        path: &Path,
        dimension: u32,
        element_size: usize,
        total_records: u64,
    ) -> Result<Self, String> {
        let record_stride = 4 + dimension as usize * element_size;
        let total_bytes = total_records * record_stride as u64;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create directory: {}", e))?;
        }

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;

        file.set_len(total_bytes)
            .map_err(|e| format!("failed to pre-allocate {} bytes: {}", total_bytes, e))?;

        // Advise kernel: pre-allocate on disk (avoid delayed allocation)
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fallocate(file.as_raw_fd(), 0, total_bytes as libc::off_t);
            }
        }

        let mmap = unsafe {
            MmapMut::map_mut(&file)
                .map_err(|e| format!("failed to mmap {}: {}", path.display(), e))?
        };

        // Advise kernel: random access pattern (not sequential), and
        // that the region will be written soon
        let _ = mmap.advise(memmap2::Advice::Random);
        #[cfg(target_os = "linux")]
        {
            let _ = mmap.advise(memmap2::Advice::WillNeed);
        }

        Ok(MmapXvecWriter {
            mmap,
            record_stride,
            dimension,
            total_records,
        })
    }

    /// Write a record at the given ordinal position.
    ///
    /// `ordinal` is the zero-based record index. `data` is raw element
    /// bytes (without the dimension prefix — the prefix is written
    /// automatically).
    ///
    /// # Safety
    /// Callers must ensure no two threads write the same ordinal
    /// concurrently. Disjoint ordinals are safe to write in parallel.
    pub fn write_record_at(&self, ordinal: u64, data: &[u8]) {
        debug_assert!(ordinal < self.total_records,
            "ordinal {} >= total {}", ordinal, self.total_records);

        let offset = ordinal as usize * self.record_stride;
        let end = offset + self.record_stride;
        debug_assert!(end <= self.mmap.len());

        // Get a mutable pointer to the mmap region for this record.
        // Safety: each ordinal maps to a disjoint byte range, and we
        // require callers to not write the same ordinal concurrently.
        let ptr = self.mmap.as_ptr() as *mut u8;
        let slice = unsafe {
            std::slice::from_raw_parts_mut(ptr.add(offset), self.record_stride)
        };

        // Write dimension header (little-endian i32)
        slice[0..4].copy_from_slice(&(self.dimension as i32).to_le_bytes());
        // Write element data
        let data_len = data.len().min(self.record_stride - 4);
        slice[4..4 + data_len].copy_from_slice(&data[..data_len]);
    }

    /// Advise the kernel to begin writeback for a byte range.
    ///
    /// Call this periodically (e.g., after writing a batch of records)
    /// to avoid a large flush at the end. The kernel writes back dirty
    /// pages asynchronously.
    pub fn advise_writeback(&self, offset: usize, len: usize) {
        #[cfg(target_os = "linux")]
        {
            // MS_ASYNC tells the kernel to schedule writeback without waiting
            let ptr = self.mmap.as_ptr() as *const libc::c_void;
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let aligned_offset = offset & !(page_size - 1);
            let aligned_len = ((offset + len) - aligned_offset + page_size - 1) & !(page_size - 1);
            unsafe {
                libc::msync(
                    ptr.add(aligned_offset) as *mut libc::c_void,
                    aligned_len,
                    libc::MS_ASYNC,
                );
            }
        }
        let _ = (offset, len); // suppress unused on non-Linux
    }

    /// Flush the mmap to disk synchronously.
    pub fn finish(self) -> Result<(), String> {
        self.mmap.flush()
            .map_err(|e| format!("failed to flush mmap: {}", e))
    }

    /// Total number of records this writer was sized for.
    pub fn total_records(&self) -> u64 {
        self.total_records
    }

    /// Bytes per record (including the 4-byte dimension header).
    pub fn record_stride(&self) -> usize {
        self.record_stride
    }
}

impl SharedMmapWriter {
    /// Create a shared writer from an `MmapXvecWriter`.
    pub fn new(writer: MmapXvecWriter) -> Self {
        SharedMmapWriter { inner: Arc::new(writer) }
    }

    /// Write a record at the given ordinal. Thread-safe for disjoint ordinals.
    pub fn write_record_at(&self, ordinal: u64, data: &[u8]) {
        self.inner.write_record_at(ordinal, data);
    }

    /// Consume all handles and flush to disk.
    ///
    /// Returns an error if other `Arc` references still exist.
    pub fn finish(self) -> Result<(), String> {
        Arc::try_unwrap(self.inner)
            .map_err(|_| "cannot finish: other references still held".to_string())?
            .finish()
    }

    /// Advise the kernel to begin async writeback for a byte range.
    pub fn advise_writeback(&self, offset: usize, len: usize) {
        self.inner.advise_writeback(offset, len);
    }

    /// Clone the shared handle for another thread.
    pub fn clone_handle(&self) -> Self {
        SharedMmapWriter { inner: Arc::clone(&self.inner) }
    }

    pub fn total_records(&self) -> u64 {
        self.inner.total_records()
    }

    pub fn record_stride(&self) -> usize {
        self.inner.record_stride()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read back an xvec file and return (dimension, records) where each
    /// record is a Vec<u8> of raw element bytes.
    fn read_xvec(path: &Path) -> (u32, Vec<Vec<u8>>) {
        let data = std::fs::read(path).unwrap();
        let mut records = Vec::new();
        let mut pos = 0;
        let mut dim = 0u32;
        while pos + 4 <= data.len() {
            let d = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as u32;
            if dim == 0 { dim = d; } else { assert_eq!(dim, d); }
            // Infer element size from stride: first record tells us
            let record_bytes = if records.is_empty() {
                // We don't know element size yet — scan forward to next dim header
                // For test purposes, assume f32 (4 bytes per element)
                d as usize * 4
            } else {
                d as usize * 4
            };
            if pos + 4 + record_bytes > data.len() { break; }
            records.push(data[pos+4..pos+4+record_bytes].to_vec());
            pos += 4 + record_bytes;
        }
        (dim, records)
    }

    #[test]
    fn test_single_thread_write_and_readback() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.fvec");

        let dim = 3u32;
        let elem_size = 4; // f32
        let total = 5u64;

        let writer = MmapXvecWriter::create(&path, dim, elem_size, total).unwrap();

        for i in 0..total {
            let val = (i + 1) as f32;
            let data: Vec<u8> = (0..dim).flat_map(|_| val.to_le_bytes()).collect();
            writer.write_record_at(i, &data);
        }
        writer.finish().unwrap();

        let (read_dim, records) = read_xvec(&path);
        assert_eq!(read_dim, dim);
        assert_eq!(records.len(), total as usize);

        for (i, rec) in records.iter().enumerate() {
            let expected_val = (i + 1) as f32;
            for chunk in rec.chunks(4) {
                let val = f32::from_le_bytes(chunk.try_into().unwrap());
                assert_eq!(val, expected_val);
            }
        }
    }

    #[test]
    fn test_concurrent_writes_disjoint_ordinals() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("concurrent.fvec");

        let dim = 4u32;
        let elem_size = 4;
        let total = 1000u64;

        let writer = MmapXvecWriter::create(&path, dim, elem_size, total).unwrap();
        let shared = SharedMmapWriter::new(writer);

        // Spawn threads that each write a disjoint range
        let num_threads = 8;
        let chunk_size = total / num_threads as u64;
        let mut handles = Vec::new();

        for t in 0..num_threads {
            let w = shared.clone_handle();
            handles.push(std::thread::spawn(move || {
                let start = t as u64 * chunk_size;
                let end = if t == num_threads - 1 { total } else { start + chunk_size };
                for ordinal in start..end {
                    // Each record: all elements = ordinal as f32
                    let val = ordinal as f32;
                    let data: Vec<u8> = (0..dim).flat_map(|_| val.to_le_bytes()).collect();
                    w.write_record_at(ordinal, &data);
                }
            }));
        }

        for h in handles { h.join().unwrap(); }
        shared.finish().unwrap();

        // Verify every record
        let (read_dim, records) = read_xvec(&path);
        assert_eq!(read_dim, dim);
        assert_eq!(records.len(), total as usize);

        for (i, rec) in records.iter().enumerate() {
            let expected = i as f32;
            for chunk in rec.chunks(4) {
                let val = f32::from_le_bytes(chunk.try_into().unwrap());
                assert_eq!(val, expected, "record {} mismatch", i);
            }
        }
    }

    #[test]
    fn test_concurrent_writes_interleaved_ordinals() {
        // Threads write interleaved ordinals (thread 0 writes 0,8,16,...
        // thread 1 writes 1,9,17,... etc.) — stress test for cache line
        // contention on adjacent records.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("interleaved.fvec");

        let dim = 8u32;
        let elem_size = 4;
        let total = 2000u64;
        let num_threads = 16usize;

        let writer = MmapXvecWriter::create(&path, dim, elem_size, total).unwrap();
        let shared = SharedMmapWriter::new(writer);

        let mut handles = Vec::new();
        for t in 0..num_threads {
            let w = shared.clone_handle();
            handles.push(std::thread::spawn(move || {
                let mut ordinal = t as u64;
                while ordinal < total {
                    let val = ordinal as f32;
                    let data: Vec<u8> = (0..dim).flat_map(|_| val.to_le_bytes()).collect();
                    w.write_record_at(ordinal, &data);
                    ordinal += num_threads as u64;
                }
            }));
        }

        for h in handles { h.join().unwrap(); }
        shared.finish().unwrap();

        let (read_dim, records) = read_xvec(&path);
        assert_eq!(read_dim, dim);
        assert_eq!(records.len(), total as usize);

        for (i, rec) in records.iter().enumerate() {
            let expected = i as f32;
            for chunk in rec.chunks(4) {
                let val = f32::from_le_bytes(chunk.try_into().unwrap());
                assert_eq!(val, expected, "record {} mismatch", i);
            }
        }
    }

    #[test]
    fn test_large_dimension_records() {
        // Test with high-dimensional vectors (768-d, like BERT embeddings)
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("large_dim.fvec");

        let dim = 768u32;
        let elem_size = 4;
        let total = 100u64;

        let writer = MmapXvecWriter::create(&path, dim, elem_size, total).unwrap();
        let shared = SharedMmapWriter::new(writer);

        let mut handles = Vec::new();
        for t in 0..4 {
            let w = shared.clone_handle();
            handles.push(std::thread::spawn(move || {
                let start = t * 25;
                for ordinal in start..start + 25 {
                    // Fill with ordinal-dependent pattern
                    let data: Vec<u8> = (0..dim)
                        .flat_map(|d| ((ordinal as f32) + (d as f32) * 0.001).to_le_bytes())
                        .collect();
                    w.write_record_at(ordinal, &data);
                }
            }));
        }

        for h in handles { h.join().unwrap(); }
        shared.finish().unwrap();

        // Verify first and last record
        let (read_dim, records) = read_xvec(&path);
        assert_eq!(read_dim, dim);
        assert_eq!(records.len(), total as usize);

        // Check record 0: values should be 0.0, 0.001, 0.002, ...
        let rec0 = &records[0];
        let v0 = f32::from_le_bytes(rec0[0..4].try_into().unwrap());
        assert!((v0 - 0.0).abs() < 1e-6);
        let v1 = f32::from_le_bytes(rec0[4..8].try_into().unwrap());
        assert!((v1 - 0.001).abs() < 1e-6);

        // Check record 99
        let rec99 = &records[99];
        let v99_0 = f32::from_le_bytes(rec99[0..4].try_into().unwrap());
        assert!((v99_0 - 99.0).abs() < 1e-4);
    }

    #[test]
    fn test_file_size_matches_expected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("size_check.mvec");

        let dim = 128u32;
        let elem_size = 2; // f16
        let total = 500u64;
        let expected_size = total * (4 + dim as u64 * elem_size as u64);

        let writer = MmapXvecWriter::create(&path, dim, elem_size, total).unwrap();
        writer.finish().unwrap();

        let actual_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(actual_size, expected_size);
    }

    #[test]
    fn test_advise_writeback_does_not_crash() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("advise.fvec");

        let dim = 4u32;
        let elem_size = 4;
        let total = 100u64;

        let writer = MmapXvecWriter::create(&path, dim, elem_size, total).unwrap();
        let stride = writer.record_stride();

        for i in 0..total {
            let data: Vec<u8> = vec![0u8; dim as usize * elem_size];
            writer.write_record_at(i, &data);
        }

        // Advise writeback on various ranges — should not panic
        writer.advise_writeback(0, stride * 50);
        writer.advise_writeback(stride * 50, stride * 50);
        writer.advise_writeback(0, stride * total as usize);

        writer.finish().unwrap();
    }
}
