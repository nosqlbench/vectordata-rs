// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Typed vector views for the data access layer.
//!
//! [`TypedVectorView`] provides format-agnostic access to vector data,
//! returning f64 or f32 values regardless of the underlying storage format.
//! Two implementations cover the two access paths:
//!
//! - [`LocalVectorView`] — zero-copy mmap access to local files
//! - [`CachedVectorView`] — on-demand merkle-verified access via [`CachedChannel`]
//!
//! Both support windowed access (arbitrary record ranges within a file).

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::io::{IoError, MmapVectorReader, VectorReader};
use crate::cache::CachedChannel;

/// Convert IoError to io::Error for the view API.
fn to_io(e: IoError) -> io::Error {
    match e {
        IoError::Io(inner) => inner,
        other => io::Error::new(io::ErrorKind::Other, other.to_string()),
    }
}

/// Format-agnostic, indexed access to vector data.
///
/// Returns vectors as owned `Vec<f64>` or `Vec<f32>`, converting from
/// the underlying storage format on the fly. Supports windowed access:
/// `count()` returns the window size, `get(0)` returns the first vector
/// in the window.
pub trait TypedVectorView: Send + Sync {
    /// Number of vectors accessible (window size, not file size).
    fn count(&self) -> usize;
    /// Dimensionality of each vector.
    fn dim(&self) -> usize;
    /// Read a single vector as f64.
    fn get_f64(&self, index: usize) -> Option<Vec<f64>>;
    /// Read a single vector as f32.
    fn get_f32(&self, index: usize) -> Option<Vec<f32>>;
    /// Read `count` consecutive vectors starting at `start` as f64.
    ///
    /// The default implementation simply loops `get_f64`, which is
    /// correct but slow on cached/remote backends — each call pays the
    /// chunk-validity check, mutex acquisition, and file-seek overhead.
    /// Backends that can satisfy a contiguous range from a single
    /// underlying read (e.g. one HTTP chunk fetch covering many vectors,
    /// or a single mmap slice) should override this method.
    fn get_f64_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f64>>> {
        (0..count).map(|i| self.get_f64(start + i)).collect()
    }
    /// Read `count` consecutive vectors starting at `start` as f32.
    /// See [`get_f64_range`](Self::get_f64_range) for rationale.
    fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        (0..count).map(|i| self.get_f32(start + i)).collect()
    }
    /// Prebuffer a range of vectors for fast subsequent access.
    fn prebuffer(&self, start: usize, end: usize) -> io::Result<()>;
    /// Returns true if all data is locally cached (mmap fast path).
    fn is_local(&self) -> bool;
    /// Cache download statistics for remote-backed views.
    /// Returns `None` for local views.
    fn cache_stats(&self) -> Option<CacheStats> { None }
}

/// Download/cache statistics for a remote-backed vector view.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of chunks that have been downloaded and verified.
    pub valid_chunks: u32,
    /// Total number of chunks in the file.
    pub total_chunks: u32,
    /// Total content size in bytes.
    pub content_size: u64,
    /// Whether all chunks are cached.
    pub is_complete: bool,
    /// Number of chunks currently being fetched.
    pub in_flight: usize,
    /// Bytes per chunk (for computing cached bytes).
    pub chunk_size: u64,
}

/// Composite view of a dataset profile's facets.
pub trait DatasetView: Send + Sync {
    fn base_vectors(&self) -> Option<Box<dyn TypedVectorView>>;
    fn query_vectors(&self) -> Option<Box<dyn TypedVectorView>>;
    fn neighbor_indices(&self) -> Option<Box<dyn TypedVectorView>>;
    fn neighbor_distances(&self) -> Option<Box<dyn TypedVectorView>>;
    fn prebuffer_all(&self) -> io::Result<()>;
}

// ---------------------------------------------------------------------------
// LocalVectorView — mmap-backed, zero-copy
// ---------------------------------------------------------------------------

/// Element type of the underlying xvec file.
#[derive(Debug, Clone, Copy)]
pub enum VecElementType {
    F32,
    F16,
    F64,
    I32,
    I16,
    U8,
}

impl VecElementType {
    /// Detect element type from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "fvec" | "fvecs" => Some(VecElementType::F32),
            "mvec" | "mvecs" => Some(VecElementType::F16),
            "dvec" | "dvecs" => Some(VecElementType::F64),
            "ivec" | "ivecs" => Some(VecElementType::I32),
            "svec" | "svecs" => Some(VecElementType::I16),
            "bvec" | "bvecs" => Some(VecElementType::U8),
            _ => None,
        }
    }

    /// Bytes per element.
    pub fn element_size(&self) -> usize {
        match self {
            VecElementType::F32 => 4,
            VecElementType::F16 => 2,
            VecElementType::F64 => 8,
            VecElementType::I32 => 4,
            VecElementType::I16 => 2,
            VecElementType::U8 => 1,
        }
    }
}

/// Local mmap-backed vector view. Supports all xvec formats.
enum LocalReader {
    F32(MmapVectorReader<f32>),
    F16(MmapVectorReader<half::f16>),
    F64(MmapVectorReader<f64>),
    I32(MmapVectorReader<i32>),
}

/// Local mmap-backed vector view with optional windowing.
pub struct LocalVectorView {
    reader: LocalReader,
    window_start: usize,
    window_count: usize,
    dim: usize,
}

impl LocalVectorView {
    /// Open a local vector file with optional window.
    ///
    /// `window` is `(start_index, count)`. If `None`, the full file is used.
    pub fn open(path: &Path, window: Option<(usize, usize)>) -> io::Result<Self> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let etype = VecElementType::from_extension(ext)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported format: .{}", ext),
            ))?;

        let (reader, total_count, dim) = match etype {
            VecElementType::F32 => {
                let r = MmapVectorReader::<f32>::open_fvec(path).map_err(to_io)?;
                let c = <MmapVectorReader<f32> as VectorReader<f32>>::count(&r);
                let d = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&r);
                (LocalReader::F32(r), c, d)
            }
            VecElementType::F16 => {
                let r = MmapVectorReader::<half::f16>::open_mvec(path).map_err(to_io)?;
                let c = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::count(&r);
                let d = <MmapVectorReader<half::f16> as VectorReader<half::f16>>::dim(&r);
                (LocalReader::F16(r), c, d)
            }
            VecElementType::F64 => {
                let r = MmapVectorReader::<f64>::open_dvec(path).map_err(to_io)?;
                let c = <MmapVectorReader<f64> as VectorReader<f64>>::count(&r);
                let d = <MmapVectorReader<f64> as VectorReader<f64>>::dim(&r);
                (LocalReader::F64(r), c, d)
            }
            VecElementType::I32 => {
                let r = MmapVectorReader::<i32>::open_ivec(path).map_err(to_io)?;
                let c = <MmapVectorReader<i32> as VectorReader<i32>>::count(&r);
                let d = <MmapVectorReader<i32> as VectorReader<i32>>::dim(&r);
                (LocalReader::I32(r), c, d)
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("format .{} not yet supported for views", ext),
                ));
            }
        };

        let (window_start, window_count) = match window {
            Some((start, count)) => {
                let effective_count = count.min(total_count.saturating_sub(start));
                (start, effective_count)
            }
            None => (0, total_count),
        };

        Ok(LocalVectorView { reader, window_start, window_count, dim })
    }
}

impl TypedVectorView for LocalVectorView {
    fn count(&self) -> usize { self.window_count }
    fn dim(&self) -> usize { self.dim }

    fn get_f64(&self, index: usize) -> Option<Vec<f64>> {
        if index >= self.window_count { return None; }
        let file_index = self.window_start + index;
        match &self.reader {
            LocalReader::F32(r) => r.get(file_index).ok().map(|v| v.iter().map(|&x| x as f64).collect()),
            LocalReader::F16(r) => r.get(file_index).ok().map(|v| v.iter().map(|x| x.to_f64()).collect()),
            LocalReader::F64(r) => r.get(file_index).ok(),
            LocalReader::I32(r) => r.get(file_index).ok().map(|v| v.iter().map(|&x| x as f64).collect()),
        }
    }

    fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        if index >= self.window_count { return None; }
        let file_index = self.window_start + index;
        match &self.reader {
            LocalReader::F32(r) => r.get(file_index).ok(),
            LocalReader::F16(r) => r.get(file_index).ok().map(|v| v.iter().map(|x| x.to_f32()).collect()),
            LocalReader::F64(r) => r.get(file_index).ok().map(|v| v.iter().map(|&x| x as f32).collect()),
            LocalReader::I32(r) => r.get(file_index).ok().map(|v| v.iter().map(|&x| x as f32).collect()),
        }
    }

    fn prebuffer(&self, _start: usize, _end: usize) -> io::Result<()> {
        Ok(()) // Already mmap'd — nothing to do
    }

    fn is_local(&self) -> bool { true }
}

// ---------------------------------------------------------------------------
// CachedVectorView — on-demand merkle-verified access via CachedChannel
// ---------------------------------------------------------------------------

/// Remote/cached vector view backed by CachedChannel.
///
/// Reads individual vectors by computing byte offsets into the xvec file
/// layout and reading through the CachedChannel (which handles on-demand
/// chunk fetching and merkle verification).
///
/// After `prebuffer()`, promotes to a MmapVectorReader for zero-copy access.
pub struct CachedVectorView {
    channel: CachedChannel,
    cache_path: PathBuf,
    dim: usize,
    entry_size: usize,
    element_type: VecElementType,
    window_start: usize,
    window_count: usize,
    #[allow(dead_code)]
    total_count: usize,
    /// After prebuffer, switch to mmap for zero-copy reads.
    promoted: Mutex<Option<LocalVectorView>>,
}

impl CachedVectorView {
    /// Create a cached vector view.
    ///
    /// Reads the dimension from the first 4 bytes of the cached/remote file
    /// to compute the entry size and total count.
    pub fn new(
        channel: CachedChannel,
        cache_path: PathBuf,
        element_type: VecElementType,
        window: Option<(usize, usize)>,
    ) -> io::Result<Self> {
        // Read dimension from the first 4 bytes (xvec header)
        let dim_bytes = channel.read(0, 4)?;
        let dim = i32::from_le_bytes([dim_bytes[0], dim_bytes[1], dim_bytes[2], dim_bytes[3]]) as usize;

        let entry_size = 4 + dim * element_type.element_size();
        let total_content = channel.content_size();
        let total_count = total_content as usize / entry_size;

        let (window_start, window_count) = match window {
            Some((start, count)) => {
                let effective = count.min(total_count.saturating_sub(start));
                (start, effective)
            }
            None => (0, total_count),
        };

        Ok(CachedVectorView {
            channel,
            cache_path,
            dim,
            entry_size,
            element_type,
            window_start,
            window_count,
            total_count,
            promoted: Mutex::new(None),
        })
    }

    /// Read a single vector's raw bytes from the channel.
    fn read_vector_bytes(&self, file_index: usize) -> Option<Vec<u8>> {
        let offset = (file_index * self.entry_size) as u64;
        // Skip the 4-byte dimension header, read element data only
        let data_offset = offset + 4;
        let data_len = (self.dim * self.element_type.element_size()) as u64;
        self.channel.read(data_offset, data_len).ok()
    }

    /// Convert raw bytes to f64 vector based on element type.
    fn bytes_to_f64(&self, bytes: &[u8]) -> Vec<f64> {
        match self.element_type {
            VecElementType::F32 => {
                bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect()
            }
            VecElementType::F16 => {
                bytes.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(bits).to_f64()
                    })
                    .collect()
            }
            VecElementType::F64 => {
                bytes.chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect()
            }
            VecElementType::I32 => {
                bytes.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect()
            }
            VecElementType::I16 => {
                bytes.chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
                    .collect()
            }
            VecElementType::U8 => {
                bytes.iter().map(|&b| b as f64).collect()
            }
        }
    }

    /// Convert raw bytes to f32 vector.
    fn bytes_to_f32(&self, bytes: &[u8]) -> Vec<f32> {
        match self.element_type {
            VecElementType::F32 => {
                bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            VecElementType::F16 => {
                bytes.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect()
            }
            VecElementType::F64 => {
                bytes.chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                    .collect()
            }
            VecElementType::I32 => {
                bytes.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()
            }
            _ => {
                // Fall back to f64 → f32
                self.bytes_to_f64(bytes).iter().map(|&v| v as f32).collect()
            }
        }
    }
}

impl TypedVectorView for CachedVectorView {
    fn count(&self) -> usize { self.window_count }
    fn dim(&self) -> usize { self.dim }

    fn get_f64(&self, index: usize) -> Option<Vec<f64>> {
        if index >= self.window_count { return None; }

        // Check for promoted mmap reader
        if let Ok(guard) = self.promoted.lock() {
            if let Some(ref local) = *guard {
                return local.get_f64(index);
            }
        }

        let file_index = self.window_start + index;
        let bytes = self.read_vector_bytes(file_index)?;
        Some(self.bytes_to_f64(&bytes))
    }

    fn get_f32(&self, index: usize) -> Option<Vec<f32>> {
        if index >= self.window_count { return None; }

        if let Ok(guard) = self.promoted.lock() {
            if let Some(ref local) = *guard {
                return local.get_f32(index);
            }
        }

        let file_index = self.window_start + index;
        let bytes = self.read_vector_bytes(file_index)?;
        Some(self.bytes_to_f32(&bytes))
    }

    /// Batched range read. The default trait impl makes one channel
    /// read per vector, which on a remote-backed view means up to one
    /// HTTP chunk fetch per vector when the visible window crosses
    /// chunk boundaries — a serial latency wall for any interactive
    /// scrolling. This override issues a single contiguous read for
    /// the whole range, so all overlapping chunks are fetched in
    /// parallel by `fetch_chunks_parallel` and decoded once.
    fn get_f64_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f64>>> {
        if count == 0 { return Vec::new(); }
        let effective = count.min(self.window_count.saturating_sub(start));
        if effective == 0 { return Vec::new(); }

        // Promoted mmap path: defer to the local reader, which has its
        // own (cheap) per-index path.
        if let Ok(guard) = self.promoted.lock() {
            if let Some(ref local) = *guard {
                return (0..effective).map(|i| local.get_f64(start + i)).collect();
            }
        }

        let bytes_per_vec = self.dim * self.element_type.element_size();
        let file_index = self.window_start + start;
        let entry_offset = (file_index * self.entry_size) as u64;
        // The xvec layout is [u32 dim header | data | u32 dim header | data ...]
        // for each entry. A contiguous range covers `count * entry_size`
        // bytes including the per-entry headers — we just step over
        // each header when slicing the resulting buffer.
        let total_len = (effective * self.entry_size) as u64;
        let buf = match self.channel.read(entry_offset, total_len) {
            Ok(b) => b,
            Err(_) => return vec![None; effective],
        };

        let mut out = Vec::with_capacity(effective);
        for i in 0..effective {
            let entry_start = i * self.entry_size + 4; // skip dim header
            let entry_end = entry_start + bytes_per_vec;
            if entry_end > buf.len() {
                out.push(None);
                continue;
            }
            out.push(Some(self.bytes_to_f64(&buf[entry_start..entry_end])));
        }
        out
    }

    fn get_f32_range(&self, start: usize, count: usize) -> Vec<Option<Vec<f32>>> {
        if count == 0 { return Vec::new(); }
        let effective = count.min(self.window_count.saturating_sub(start));
        if effective == 0 { return Vec::new(); }

        if let Ok(guard) = self.promoted.lock() {
            if let Some(ref local) = *guard {
                return (0..effective).map(|i| local.get_f32(start + i)).collect();
            }
        }

        let bytes_per_vec = self.dim * self.element_type.element_size();
        let file_index = self.window_start + start;
        let entry_offset = (file_index * self.entry_size) as u64;
        let total_len = (effective * self.entry_size) as u64;
        let buf = match self.channel.read(entry_offset, total_len) {
            Ok(b) => b,
            Err(_) => return vec![None; effective],
        };

        let mut out = Vec::with_capacity(effective);
        for i in 0..effective {
            let entry_start = i * self.entry_size + 4;
            let entry_end = entry_start + bytes_per_vec;
            if entry_end > buf.len() {
                out.push(None);
                continue;
            }
            out.push(Some(self.bytes_to_f32(&buf[entry_start..entry_end])));
        }
        out
    }

    fn prebuffer(&self, start: usize, end: usize) -> io::Result<()> {
        let file_start = self.window_start + start;
        let file_end = self.window_start + end.min(self.window_count);

        let byte_start = (file_start * self.entry_size) as u64;
        let byte_end = (file_end * self.entry_size) as u64;

        // Ensure all chunks in the byte range are cached
        let shape = self.channel.reference().shape();
        let first_chunk = (byte_start / shape.chunk_size) as u32;
        let last_chunk = ((byte_end.saturating_sub(1)) / shape.chunk_size) as u32;
        self.channel.ensure_range(first_chunk, last_chunk)?;

        // If fully cached, promote to mmap
        if self.channel.is_complete() {
            let window = Some((self.window_start, self.window_count));
            if let Ok(local) = LocalVectorView::open(&self.cache_path, window) {
                let mut guard = self.promoted.lock().unwrap();
                *guard = Some(local);
            }
        }

        Ok(())
    }

    fn is_local(&self) -> bool {
        self.promoted.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    fn cache_stats(&self) -> Option<CacheStats> {
        let shape = self.channel.reference().shape();
        Some(CacheStats {
            valid_chunks: self.channel.valid_count(),
            total_chunks: self.channel.total_chunks(),
            content_size: self.channel.content_size(),
            is_complete: self.channel.is_complete(),
            in_flight: self.channel.in_flight_count(),
            chunk_size: shape.chunk_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fvec(path: &Path, dim: u32, count: u32) {
        let mut f = std::fs::File::create(path).unwrap();
        for i in 0..count {
            f.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for j in 0..dim {
                let val = (i * dim + j) as f32;
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn test_local_view_fvec() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.fvec");
        write_fvec(&path, 4, 100);

        let view = LocalVectorView::open(&path, None).unwrap();
        assert_eq!(view.count(), 100);
        assert_eq!(view.dim(), 4);
        assert!(view.is_local());

        let v = view.get_f64(0).unwrap();
        assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0]);

        let v = view.get_f32(99).unwrap();
        assert_eq!(v, vec![396.0, 397.0, 398.0, 399.0]);

        assert!(view.get_f64(100).is_none());
    }

    #[test]
    fn test_local_view_windowed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.fvec");
        write_fvec(&path, 4, 100);

        let view = LocalVectorView::open(&path, Some((10, 20))).unwrap();
        assert_eq!(view.count(), 20);
        assert_eq!(view.dim(), 4);

        // Index 0 of the view is index 10 of the file
        let v = view.get_f64(0).unwrap();
        assert_eq!(v, vec![40.0, 41.0, 42.0, 43.0]); // 10*4, 10*4+1, ...

        assert!(view.get_f64(20).is_none());
    }

    #[test]
    fn test_local_view_window_clamped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.fvec");
        write_fvec(&path, 4, 100);

        // Window extends beyond file
        let view = LocalVectorView::open(&path, Some((90, 50))).unwrap();
        assert_eq!(view.count(), 10); // clamped to 100 - 90
    }

    /// In-memory transport for the cached-view range-read test.
    struct MemTransport(Vec<u8>);
    impl crate::transport::ChunkedTransport for MemTransport {
        fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
            let s = start as usize;
            let e = s + len as usize;
            Ok(self.0[s..e].to_vec())
        }
        fn content_length(&self) -> io::Result<u64> { Ok(self.0.len() as u64) }
        fn supports_range(&self) -> bool { true }
    }

    fn fvec_buffer(dim: u32, count: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        for i in 0..count {
            buf.extend_from_slice(&(dim as i32).to_le_bytes());
            for j in 0..dim {
                let val = (i * dim + j) as f32;
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn test_cached_view_get_f64_range_matches_per_index() {
        // Build a small fvec, wrap it in CachedChannel via MemTransport,
        // then verify the optimized batch read returns the same vectors
        // a per-index loop would. Uses a chunk_size that forces the
        // range to span multiple chunks so the batched single-channel
        // read exercises the multi-chunk fetch path.
        let dim = 4u32;
        let count = 50u32;
        let buf = fvec_buffer(dim, count);
        let chunk_size = 64u64; // ~3 vectors per chunk
        let mref = crate::merkle::MerkleRef::from_content(&buf, chunk_size);
        let dir = tempfile::tempdir().unwrap();
        let transport = Box::new(MemTransport(buf.clone()));
        let channel = CachedChannel::open(transport, mref, dir.path(), "test.fvec").unwrap();
        let view = CachedVectorView::new(
            channel,
            dir.path().join("test.fvec"),
            VecElementType::F32,
            None,
        ).unwrap();

        let want: Vec<Vec<f64>> = (0..10)
            .map(|i| view.get_f64(i).unwrap())
            .collect();
        let got = view.get_f64_range(0, 10);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let g = g.as_ref().unwrap_or_else(|| panic!("missing {i}"));
            assert_eq!(g, w, "row {i}");
        }
        // First vector should be [0, 1, 2, 3] per fvec_buffer.
        assert_eq!(got[0].as_ref().unwrap(), &vec![0.0, 1.0, 2.0, 3.0]);
        // Range that runs past the end is clamped, not panicked.
        let tail = view.get_f64_range(48, 10);
        assert_eq!(tail.len(), 2);
    }
}
