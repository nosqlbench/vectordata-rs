// Copyright (c) nosqlbench contributors
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
}
