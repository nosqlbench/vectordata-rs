// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Memory-mapped reader for scalar formats.
//!
//! Zero-copy random access to flat packed arrays.

use std::marker::PhantomData;
use std::path::Path;

use memmap2::Mmap;

/// Error type for mmap operations.
#[derive(Debug)]
pub enum MmapError {
    Io(std::io::Error),
    InvalidSize { file_size: u64, element_size: usize },
    EmptyFile,
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::Io(e) => write!(f, "I/O error: {}", e),
            MmapError::InvalidSize { file_size, element_size } => {
                write!(f, "file size {} not divisible by element size {}", file_size, element_size)
            }
            MmapError::EmptyFile => write!(f, "empty file"),
        }
    }
}

impl std::error::Error for MmapError {}

/// Zero-copy random-access reader for scalar files.
///
/// Each element is `size_of::<T>()` bytes. No header, no dimension prefix.
/// Ordinal N is at byte offset `N * size_of::<T>()`.
pub struct ScalarMmapReader<T> {
    mmap: Mmap,
    count: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy + 'static> ScalarMmapReader<T> {
    fn open_impl(path: &Path) -> Result<Self, MmapError> {
        let file = std::fs::File::open(path).map_err(MmapError::Io)?;
        let meta = file.metadata().map_err(MmapError::Io)?;
        let file_size = meta.len();
        if file_size == 0 {
            return Err(MmapError::EmptyFile);
        }
        let elem_size = std::mem::size_of::<T>();
        if file_size as usize % elem_size != 0 {
            return Err(MmapError::InvalidSize { file_size, element_size: elem_size });
        }
        let count = file_size as usize / elem_size;
        let mmap = unsafe { Mmap::map(&file).map_err(MmapError::Io)? };

        // Advise sequential read for initial scan
        #[cfg(unix)]
        unsafe {
            libc::posix_madvise(
                mmap.as_ptr() as *mut _,
                mmap.len(),
                libc::POSIX_MADV_SEQUENTIAL,
            );
        }

        Ok(ScalarMmapReader { mmap, count, _phantom: PhantomData })
    }

    /// Number of elements in the file.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get element at the given ordinal.
    ///
    /// # Panics
    /// Panics if `index >= count()`.
    pub fn get(&self, index: usize) -> T {
        assert!(index < self.count, "index {} out of range (count {})", index, self.count);
        let elem_size = std::mem::size_of::<T>();
        let offset = index * elem_size;
        let ptr = self.mmap[offset..offset + elem_size].as_ptr() as *const T;
        unsafe { std::ptr::read_unaligned(ptr) }
    }

    /// Get element as a single-element slice (for API compatibility with vector readers).
    pub fn get_slice(&self, index: usize) -> &[T] {
        assert!(index < self.count, "index {} out of range (count {})", index, self.count);
        let elem_size = std::mem::size_of::<T>();
        let offset = index * elem_size;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap[offset..].as_ptr() as *const T,
                1,
            )
        }
    }
}

// Typed constructors
impl ScalarMmapReader<u8> {
    /// Open a `.u8` scalar file.
    pub fn open_u8(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}
impl ScalarMmapReader<i8> {
    /// Open a `.i8` scalar file.
    pub fn open_i8(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}
impl ScalarMmapReader<u16> {
    /// Open a `.u16` scalar file.
    pub fn open_u16(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}
impl ScalarMmapReader<i16> {
    /// Open a `.i16` scalar file.
    pub fn open_i16(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}
impl ScalarMmapReader<u32> {
    /// Open a `.u32` scalar file.
    pub fn open_u32(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}
impl ScalarMmapReader<i32> {
    /// Open a `.i32` scalar file.
    pub fn open_i32(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}
impl ScalarMmapReader<u64> {
    /// Open a `.u64` scalar file.
    pub fn open_u64(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}
impl ScalarMmapReader<i64> {
    /// Open a `.i64` scalar file.
    pub fn open_i64(path: &Path) -> Result<Self, MmapError> { Self::open_impl(path) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_tmp() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    #[test]
    fn test_u8_mmap_roundtrip() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        let values: Vec<u8> = (0..=255).collect();
        std::fs::write(&path, &values).unwrap();

        let reader = ScalarMmapReader::<u8>::open_u8(&path).unwrap();
        assert_eq!(reader.count(), 256);
        for i in 0..256 {
            assert_eq!(reader.get(i), i as u8);
        }
    }

    #[test]
    fn test_i32_mmap_roundtrip() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        for v in [-100i32, -1, 0, 1, 100, i32::MAX, i32::MIN] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let reader = ScalarMmapReader::<i32>::open_i32(&path).unwrap();
        assert_eq!(reader.count(), 7);
        assert_eq!(reader.get(0), -100);
        assert_eq!(reader.get(2), 0);
        assert_eq!(reader.get(5), i32::MAX);
        assert_eq!(reader.get(6), i32::MIN);
    }

    #[test]
    fn test_u64_mmap() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u64");
        let mut f = std::fs::File::create(&path).unwrap();
        for v in [0u64, 1, u64::MAX] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let reader = ScalarMmapReader::<u64>::open_u64(&path).unwrap();
        assert_eq!(reader.count(), 3);
        assert_eq!(reader.get(2), u64::MAX);
    }

    #[test]
    fn test_invalid_size_rejected() {
        let tmp = make_tmp();
        let path = tmp.path().join("bad.i32");
        // 5 bytes is not divisible by 4
        std::fs::write(&path, &[0u8; 5]).unwrap();
        assert!(ScalarMmapReader::<i32>::open_i32(&path).is_err());
    }

    #[test]
    fn test_empty_file_rejected() {
        let tmp = make_tmp();
        let path = tmp.path().join("empty.u8");
        std::fs::write(&path, &[]).unwrap();
        assert!(ScalarMmapReader::<u8>::open_u8(&path).is_err());
    }

    #[test]
    fn test_get_slice_returns_single() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u16");
        let mut f = std::fs::File::create(&path).unwrap();
        for v in [10u16, 20, 30] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let reader = ScalarMmapReader::<u16>::open_u16(&path).unwrap();
        assert_eq!(reader.get_slice(1), &[20u16]);
    }
}
