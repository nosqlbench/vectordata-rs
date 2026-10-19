// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Memory-mapped reader for header-free columns: zero-copy random
//! access, and the whole column as one slice.

use std::marker::PhantomData;
use std::path::Path;

use memmap2::Mmap;

use crate::element::{Container, Element, aligned_for, decode_into, require_native};
pub use crate::element::MmapError;

/// Zero-copy random-access reader for a column file.
///
/// Each element is [`T::SIZE`](Element::SIZE) bytes, little-endian,
/// one after another with no header: ordinal `n` is at byte offset
/// `n * SIZE`. `T` is the type the extension names (`.u8`, `.i32`,
/// `.f32`, `.bf16`, …); [`open`](Self::open) refuses any other.
pub struct ScalarMmapReader<T> {
    mmap: Mmap,
    count: usize,
    _phantom: PhantomData<T>,
}

impl<T: Element> ScalarMmapReader<T> {
    /// Open and memory-map a column file whose extension names `T`.
    /// The file must be non-empty and a whole number of elements.
    pub fn open(path: &Path) -> Result<Self, MmapError> {
        require_native::<T>(path, Container::Scalar)?;
        Self::open_impl(path)
    }

    fn open_impl(path: &Path) -> Result<Self, MmapError> {
        let file = std::fs::File::open(path).map_err(MmapError::Open)?;
        let file_size = file.metadata().map_err(MmapError::Open)?.len();
        if file_size == 0 {
            return Err(MmapError::InvalidFormat(format!("{}: empty file", path.display())));
        }
        let file_size = usize::try_from(file_size)
            .map_err(|_| MmapError::InvalidFormat(format!("{}: too large to map", path.display())))?;
        if !file_size.is_multiple_of(T::SIZE) {
            return Err(MmapError::InvalidFormat(format!(
                "{}: file size {file_size} is not a whole number of {}-byte elements",
                path.display(),
                T::SIZE
            )));
        }
        let count = file_size / T::SIZE;
        // SAFETY: the mapping is read-only, and a file changed underneath
        // a mapping is the documented hazard of every mmap reader here.
        let mmap = unsafe { Mmap::map(&file).map_err(MmapError::Mmap)? };

        // Advise sequential read for initial scan
        #[cfg(unix)]
        // SAFETY: the pointer and length are the live mapping's.
        unsafe {
            libc::posix_madvise(mmap.as_ptr() as *mut _, mmap.len(), libc::POSIX_MADV_SEQUENTIAL);
        }

        Ok(ScalarMmapReader { mmap, count, _phantom: PhantomData })
    }

    /// Number of elements in the file.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The element at `index`. `None` when `index` is out of bounds.
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.count {
            return None;
        }
        Some(T::from_le_bytes(&self.mmap[index * T::SIZE..(index + 1) * T::SIZE]))
    }

    /// The element at `index` as a one-element slice borrowed from the
    /// mapping, for callers that treat a column as vectors of dimension
    /// one. `None` when `index` is out of bounds or the element is not
    /// on a `T` boundary.
    pub fn get_slice(&self, index: usize) -> Option<&[T]> {
        if index >= self.count {
            return None;
        }
        self.as_slice().map(|all| &all[index..index + 1])
    }

    /// The whole column, borrowed from the mapping: `count()` elements
    /// with no copy. `None` when the mapping does not start on a `T`
    /// boundary, which a page-aligned mapping of a file always does.
    pub fn as_slice(&self) -> Option<&[T]> {
        if !aligned_for::<T>(&self.mmap) {
            return None;
        }
        // SAFETY: the mapping is `count * T::SIZE` bytes on a `T`
        // boundary, and every `Element` is a plain number.
        Some(unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const T, self.count) })
    }

    /// Read elements from `start` into `out`. Returns how many were
    /// read: as many as fit, or as many as remain.
    pub fn read_into(&self, start: usize, out: &mut [T]) -> Result<usize, MmapError> {
        if start > self.count {
            return Err(MmapError::OutOfBounds { index: start, count: self.count });
        }
        let n = out.len().min(self.count - start);
        decode_into(&self.mmap[start * T::SIZE..(start + n) * T::SIZE], &mut out[..n]);
        Ok(n)
    }
}

impl<T: Element> std::fmt::Debug for ScalarMmapReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarMmapReader")
            .field("element", &T::TYPE)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

// ── Type-specific constructors ─────────────────────────────────────────
//
// Each is `ScalarMmapReader::<T>::open` with the type spelled in the
// name. They predate the element model.

macro_rules! deprecated_open {
    ($($t:ty, $name:ident, $ext:literal, $note:literal;)*) => {$(
        impl ScalarMmapReader<$t> {
            #[doc = concat!("Open a `.", $ext, "` column; `ScalarMmapReader::<", $ext, ">::open` with the type in the name.")]
            #[deprecated(note = $note)]
            pub fn $name(path: &Path) -> Result<Self, MmapError> {
                Self::open(path)
            }
        }
    )*};
}

deprecated_open! {
    u8, open_u8, "u8", "use `ScalarMmapReader::<u8>::open`";
    i8, open_i8, "i8", "use `ScalarMmapReader::<i8>::open`";
    u16, open_u16, "u16", "use `ScalarMmapReader::<u16>::open`";
    i16, open_i16, "i16", "use `ScalarMmapReader::<i16>::open`";
    u32, open_u32, "u32", "use `ScalarMmapReader::<u32>::open`";
    i32, open_i32, "i32", "use `ScalarMmapReader::<i32>::open`";
    u64, open_u64, "u64", "use `ScalarMmapReader::<u64>::open`";
    i64, open_i64, "i64", "use `ScalarMmapReader::<i64>::open`";
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
    fn a_u8_column_reads_back() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        let values: Vec<u8> = (0..=255).collect();
        std::fs::write(&path, &values).unwrap();

        let reader = ScalarMmapReader::<u8>::open(&path).unwrap();
        assert_eq!(reader.count(), 256);
        for i in 0..256 {
            assert_eq!(reader.get(i), Some(i as u8));
        }
        assert_eq!(reader.get(256), None);
        assert_eq!(reader.as_slice(), Some(&values[..]));
    }

    #[test]
    fn an_i32_column_reads_back() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut f = std::fs::File::create(&path).unwrap();
        let values = [-100i32, -1, 0, 1, 100, i32::MAX, i32::MIN];
        for v in values {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let reader = ScalarMmapReader::<i32>::open(&path).unwrap();
        assert_eq!(reader.count(), 7);
        assert_eq!(reader.get(0), Some(-100));
        assert_eq!(reader.get(2), Some(0));
        assert_eq!(reader.get(5), Some(i32::MAX));
        assert_eq!(reader.get(6), Some(i32::MIN));
        assert_eq!(reader.as_slice(), Some(&values[..]));
        let mut out = [0i32; 3];
        assert_eq!(reader.read_into(5, &mut out).unwrap(), 2);
        assert_eq!(&out[..2], &[i32::MAX, i32::MIN]);
        assert!(matches!(reader.read_into(8, &mut out), Err(MmapError::OutOfBounds { index: 8, count: 7 })));
    }

    #[test]
    fn a_u64_column_reads_back() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u64");
        let mut f = std::fs::File::create(&path).unwrap();
        for v in [0u64, 1, u64::MAX] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let reader = ScalarMmapReader::<u64>::open(&path).unwrap();
        assert_eq!(reader.count(), 3);
        assert_eq!(reader.get(2), Some(u64::MAX));
    }

    #[test]
    fn a_size_that_is_not_whole_elements_is_refused() {
        let tmp = make_tmp();
        let path = tmp.path().join("bad.i32");
        // 5 bytes is not divisible by 4
        std::fs::write(&path, [0u8; 5]).unwrap();
        assert!(matches!(ScalarMmapReader::<i32>::open(&path), Err(MmapError::InvalidFormat(_))));
    }

    #[test]
    fn an_empty_column_is_refused() {
        let tmp = make_tmp();
        let path = tmp.path().join("empty.u8");
        std::fs::write(&path, []).unwrap();
        assert!(matches!(ScalarMmapReader::<u8>::open(&path), Err(MmapError::InvalidFormat(_))));
    }

    #[test]
    fn get_slice_is_one_element() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u16");
        let mut f = std::fs::File::create(&path).unwrap();
        for v in [10u16, 20, 30] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);

        let reader = ScalarMmapReader::<u16>::open(&path).unwrap();
        assert_eq!(reader.get_slice(1), Some(&[20u16][..]));
        assert_eq!(reader.get_slice(3), None);
    }

    #[test]
    fn a_column_is_not_opened_at_another_type() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u16");
        std::fs::write(&path, [0u8; 8]).unwrap();
        assert!(matches!(
            ScalarMmapReader::<i16>::open(&path),
            Err(MmapError::ElementMismatch { file: crate::ElementType::U16, requested: crate::ElementType::I16 })
        ));
        assert!(matches!(ScalarMmapReader::<u32>::open(&path), Err(MmapError::ElementMismatch { .. })));
    }
}
