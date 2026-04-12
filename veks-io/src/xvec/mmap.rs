// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Memory-mapped xvec reader for zero-copy random access.
//!
//! Provides typed random-access readers for f32 (fvec), i32 (ivec),
//! f16 (mvec), f64 (dvec), u8 (bvec), and i16 (svec) formats.

use std::marker::PhantomData;
use std::path::Path;

use memmap2::Mmap;

/// Error type for mmap I/O operations.
#[derive(Debug)]
pub enum MmapError {
    /// File could not be opened.
    Open(std::io::Error),
    /// File could not be memory-mapped.
    Mmap(std::io::Error),
    /// File is empty or has invalid format.
    InvalidFormat(String),
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::Open(e) => write!(f, "open: {}", e),
            MmapError::Mmap(e) => write!(f, "mmap: {}", e),
            MmapError::InvalidFormat(s) => write!(f, "invalid format: {}", s),
        }
    }
}

impl std::error::Error for MmapError {}

/// Memory-mapped vector reader for zero-copy random access.
///
/// Element type `T` determines how the raw bytes are interpreted:
/// - `f32` for fvec files
/// - `i32` for ivec files
/// - `half::f16` for mvec files
/// - `f64` for dvec files
/// - `u8` for bvec files (note: bvec stores 4-byte groups)
/// - `i16` for svec files
pub struct MmapReader<T> {
    mmap: Mmap,
    dim: usize,
    count: usize,
    entry_bytes: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy + 'static> MmapReader<T> {
    /// Open a file and memory-map it.
    fn open_impl(path: &Path, elem_size: usize) -> Result<Self, MmapError> {
        let file = std::fs::File::open(path).map_err(MmapError::Open)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(MmapError::Mmap)?;

        if mmap.len() < 4 {
            return Err(MmapError::InvalidFormat("file too small".into()));
        }

        let dim = i32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
        if dim == 0 {
            return Err(MmapError::InvalidFormat("zero dimension".into()));
        }

        let entry_bytes = 4 + dim * elem_size; // 4-byte dim header + data
        let count = mmap.len() / entry_bytes;

        if mmap.len() % entry_bytes != 0 {
            return Err(MmapError::InvalidFormat(format!(
                "file size {} is not a multiple of record size {}",
                mmap.len(), entry_bytes
            )));
        }

        Ok(Self { mmap, dim, count, entry_bytes, _phantom: PhantomData })
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize { self.dim }

    /// Total number of vectors.
    pub fn count(&self) -> usize { self.count }

    /// Bytes per record (including 4-byte dimension header).
    pub fn entry_bytes(&self) -> usize { self.entry_bytes }

    /// Zero-copy slice access to vector at `index`.
    ///
    /// Returns a slice of `T` with `dim` elements. Panics if index is
    /// out of bounds.
    pub fn get_slice(&self, index: usize) -> &[T] {
        assert!(index < self.count, "index {} out of bounds (count={})", index, self.count);
        let offset = index * self.entry_bytes + 4; // skip dim header
        let ptr = self.mmap[offset..].as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.dim) }
    }

    /// Hint the kernel to use sequential read-ahead.
    pub fn advise_sequential(&self) {
        #[cfg(target_os = "linux")]
        unsafe {
            libc::posix_madvise(
                self.mmap.as_ptr() as *mut libc::c_void,
                self.mmap.len(),
                libc::POSIX_MADV_SEQUENTIAL,
            );
        }
    }
}

// ── Type-specific constructors ─────────────────────────────────────────

impl MmapReader<f32> {
    /// Open an fvec file (float32 vectors).
    pub fn open_fvec(path: &Path) -> Result<Self, MmapError> {
        Self::open_impl(path, 4)
    }
}

impl MmapReader<i32> {
    /// Open an ivec file (int32 vectors).
    pub fn open_ivec(path: &Path) -> Result<Self, MmapError> {
        Self::open_impl(path, 4)
    }
}

impl MmapReader<half::f16> {
    /// Open an mvec file (float16 vectors).
    pub fn open_mvec(path: &Path) -> Result<Self, MmapError> {
        Self::open_impl(path, 2)
    }
}

impl MmapReader<f64> {
    /// Open a dvec file (float64 vectors).
    pub fn open_dvec(path: &Path) -> Result<Self, MmapError> {
        Self::open_impl(path, 8)
    }
}

impl MmapReader<u8> {
    /// Open a bvec file (uint8 vectors, 1 byte per element).
    pub fn open_bvec(path: &Path) -> Result<Self, MmapError> {
        Self::open_impl(path, 1)
    }
}

impl MmapReader<i16> {
    /// Open an svec file (int16 vectors).
    pub fn open_svec(path: &Path) -> Result<Self, MmapError> {
        Self::open_impl(path, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            f.write_all(&(v.len() as i32).to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    fn write_ivec(path: &Path, vectors: &[Vec<i32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            f.write_all(&(v.len() as i32).to_le_bytes()).unwrap();
            for &val in v {
                f.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn test_fvec_mmap_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.fvec");
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        write_fvec(&path, &data);

        let reader = MmapReader::<f32>::open_fvec(&path).unwrap();
        assert_eq!(reader.dim(), 3);
        assert_eq!(reader.count(), 2);
        assert_eq!(reader.get_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(reader.get_slice(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_ivec_mmap() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.ivec");
        let data = vec![vec![10, 20], vec![30, 40]];
        write_ivec(&path, &data);

        let reader = MmapReader::<i32>::open_ivec(&path).unwrap();
        assert_eq!(reader.dim(), 2);
        assert_eq!(reader.count(), 2);
        assert_eq!(reader.get_slice(0), &[10, 20]);
        assert_eq!(reader.get_slice(1), &[30, 40]);
    }

    #[test]
    fn test_empty_file_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.fvec");
        std::fs::File::create(&path).unwrap();
        assert!(MmapReader::<f32>::open_fvec(&path).is_err());
    }
}
