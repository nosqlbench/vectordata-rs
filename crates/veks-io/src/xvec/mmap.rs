// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Memory-mapped xvec reader for zero-copy random access, at any
//! [`Element`] type the file's extension names.

use std::marker::PhantomData;
use std::path::Path;

use memmap2::Mmap;

use crate::element::{Container, Element, aligned_for, decode_into, require_native};
pub use crate::element::MmapError;

/// Memory-mapped vector reader for zero-copy random access.
///
/// `T` is the element type the file holds, which its extension names:
/// `f32` for `.fvecs`, `u8` for `.bvecs`, [`bf16`](crate::bf16) for
/// `.bf16vecs`, and so on through every [`ElementType`]. [`open`]
/// refuses a file whose extension names another type.
///
/// [`ElementType`]: crate::ElementType
/// [`open`]: Self::open
pub struct MmapReader<T> {
    mmap: Mmap,
    dim: usize,
    count: usize,
    entry_bytes: usize,
    _phantom: PhantomData<T>,
}

impl<T: Element> MmapReader<T> {
    /// Open and memory-map a uniform vector file whose extension names
    /// `T`.
    ///
    /// The file must be a whole number of records, each a positive
    /// little-endian `i32` dimension followed by that many elements;
    /// the first record's dimension is taken as every record's.
    ///
    /// # Examples
    ///
    /// An integer file (`.u32vecs` holds `u32`), read zero-copy:
    ///
    /// ```
    /// use veks_io::MmapReader;
    /// # use std::io::Write;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let path = dir.path().join("ids.u32vecs");
    /// # let mut f = std::fs::File::create(&path).unwrap();
    /// # for row in [[1u32, 2], [u32::MAX, 0]] {
    /// #     f.write_all(&2i32.to_le_bytes()).unwrap();
    /// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
    /// # }
    /// # drop(f);
    /// let ids = MmapReader::<u32>::open(&path).unwrap();
    /// assert_eq!((ids.dim(), ids.count()), (2, 2));
    /// assert_eq!(ids.get_slice(1).unwrap(), &[u32::MAX, 0]); // zero-copy
    /// ```
    ///
    /// A `bf16` file (`.bf16vecs`), the same way; the elements come
    /// back as [`bf16`](crate::bf16) and widen to `f32` exactly:
    ///
    /// ```
    /// use veks_io::{MmapReader, bf16};
    /// # use std::io::Write;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let path = dir.path().join("base.bf16vecs");
    /// # let mut f = std::fs::File::create(&path).unwrap();
    /// # for row in [[1.5f32, -2.0], [0.0, 64.0]] {
    /// #     f.write_all(&2i32.to_le_bytes()).unwrap();
    /// #     for v in row { f.write_all(&bf16::from_f32(v).to_le_bytes()).unwrap(); }
    /// # }
    /// # drop(f);
    /// let base = MmapReader::<bf16>::open(&path).unwrap();
    /// let row: &[bf16] = base.get_slice(0).unwrap(); // zero-copy
    /// let widened: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
    /// assert_eq!(widened, [1.5, -2.0]);
    /// ```
    ///
    /// An 8-byte element type (`f64` in `.dvecs`): a record's payload
    /// sits 4 bytes after a multiple of the record size, and every
    /// `f64` record size is 4 more than a multiple of 8, so even
    /// records are not on an 8-byte boundary. [`get_slice`](Self::get_slice)
    /// is `None` there and [`get`](Self::get) still reads (a decoded copy):
    ///
    /// ```
    /// use veks_io::MmapReader;
    /// # use std::io::Write;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let path = dir.path().join("precise.dvecs");
    /// # let mut f = std::fs::File::create(&path).unwrap();
    /// # for row in [[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]] {
    /// #     f.write_all(&3i32.to_le_bytes()).unwrap();
    /// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
    /// # }
    /// # drop(f);
    /// let precise = MmapReader::<f64>::open(&path).unwrap();
    /// assert!(precise.get_slice(0).is_none());           // unaligned: refused, not reinterpreted
    /// assert_eq!(precise.get(0).unwrap(), [1.0, 2.0, 3.0]); // decoded copy
    /// assert_eq!(precise.get_slice(1).unwrap(), &[4.0, 5.0, 6.0]); // the odd record is aligned
    /// ```
    pub fn open(path: &Path) -> Result<Self, MmapError> {
        require_native::<T>(path, Container::Vec)?;
        Self::open_impl(path)
    }

    fn open_impl(path: &Path) -> Result<Self, MmapError> {
        let file = std::fs::File::open(path).map_err(MmapError::Open)?;
        // SAFETY: the mapping is read-only, and a file changed underneath
        // a mapping is the documented hazard of every mmap reader here.
        let mmap = unsafe { Mmap::map(&file) }.map_err(MmapError::Mmap)?;

        if mmap.len() < 4 {
            return Err(MmapError::InvalidFormat(format!(
                "{}: {} bytes is too short for a record header",
                path.display(),
                mmap.len()
            )));
        }

        let dim = i32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
        if dim <= 0 {
            return Err(MmapError::InvalidFormat(format!(
                "{}: dimension {dim} is not positive",
                path.display()
            )));
        }
        let dim = dim as usize;
        let entry_bytes = dim
            .checked_mul(T::SIZE)
            .and_then(|payload| payload.checked_add(4))
            .ok_or_else(|| {
                MmapError::InvalidFormat(format!("{}: a record of dimension {dim} overflows", path.display()))
            })?;

        if !mmap.len().is_multiple_of(entry_bytes) {
            return Err(MmapError::InvalidFormat(format!(
                "{}: file size {} is not a whole number of {entry_bytes}-byte records",
                path.display(),
                mmap.len()
            )));
        }
        let count = mmap.len() / entry_bytes;

        Ok(Self { mmap, dim, count, entry_bytes, _phantom: PhantomData })
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total number of vectors.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Bytes per record, including the 4-byte dimension header.
    pub fn entry_bytes(&self) -> usize {
        self.entry_bytes
    }

    /// The bytes of record `index`'s elements, after its header.
    fn payload(&self, index: usize) -> Option<&[u8]> {
        if index >= self.count {
            return None;
        }
        let start = index * self.entry_bytes + 4;
        Some(&self.mmap[start..start + self.dim * T::SIZE])
    }

    /// The vector at `index`, borrowed from the mapping: `dim` elements
    /// with no copy.
    ///
    /// `None` when `index` is out of bounds, and when the record does
    /// not start on a `T` boundary — every record's payload sits 4
    /// bytes after a multiple of the record size, so an 8-byte element
    /// type is aligned only when the record size is a multiple of 8.
    /// Read such a record with [`get`](Self::get) or
    /// [`read_into`](Self::read_into).
    pub fn get_slice(&self, index: usize) -> Option<&[T]> {
        let bytes = self.payload(index)?;
        if !aligned_for::<T>(bytes) {
            return None;
        }
        // SAFETY: `bytes` is `dim * T::SIZE` bytes of the mapping, on a
        // `T` boundary, and every `Element` is a plain number.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, self.dim) })
    }

    /// The vector at `index`, decoded into a new `Vec`. `None` when
    /// `index` is out of bounds.
    pub fn get(&self, index: usize) -> Option<Vec<T>> {
        let bytes = self.payload(index)?;
        Some(bytes.chunks_exact(T::SIZE).map(T::from_le_bytes).collect())
    }

    /// Read whole records from `start` into `out`, which must hold a
    /// whole number of `dim`-element records. Returns how many records
    /// were read: as many as fit, or as many as remain.
    ///
    /// # Example
    ///
    /// Three `i16` records in an `.svecs` file; a buffer for two reads
    /// records 1 and 2 as one decoded copy, and from the last record
    /// only what remains:
    ///
    /// ```
    /// use veks_io::MmapReader;
    /// # use std::io::Write;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let path = dir.path().join("short.svecs");
    /// # let mut f = std::fs::File::create(&path).unwrap();
    /// # for row in [[1i16, -1], [200, -200], [i16::MAX, i16::MIN]] {
    /// #     f.write_all(&2i32.to_le_bytes()).unwrap();
    /// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
    /// # }
    /// # drop(f);
    /// let short = MmapReader::<i16>::open(&path).unwrap();
    /// let mut out = vec![0i16; short.dim() * 2];
    /// assert_eq!(short.read_into(1, &mut out).unwrap(), 2);
    /// assert_eq!(out, [200, -200, i16::MAX, i16::MIN]);
    /// assert_eq!(short.read_into(2, &mut out).unwrap(), 1); // only one record remains
    /// assert_eq!(&out[..2], &[i16::MAX, i16::MIN]);
    /// ```
    pub fn read_into(&self, start: usize, out: &mut [T]) -> Result<usize, MmapError> {
        if !out.len().is_multiple_of(self.dim) {
            return Err(MmapError::InvalidFormat(format!(
                "a buffer of {} elements is not a whole number of {}-element records",
                out.len(),
                self.dim
            )));
        }
        if start > self.count {
            return Err(MmapError::OutOfBounds { index: start, count: self.count });
        }
        let records = (out.len() / self.dim).min(self.count - start);
        for (i, chunk) in out.chunks_exact_mut(self.dim).take(records).enumerate() {
            let bytes = self.payload(start + i).expect("within count");
            decode_into(bytes, chunk);
        }
        Ok(records)
    }

    /// Hint the kernel to use sequential read-ahead.
    pub fn advise_sequential(&self) {
        #[cfg(target_os = "linux")]
        // SAFETY: the pointer and length are the live mapping's.
        unsafe {
            libc::posix_madvise(
                self.mmap.as_ptr() as *mut libc::c_void,
                self.mmap.len(),
                libc::POSIX_MADV_SEQUENTIAL,
            );
        }
    }
}

impl<T: Element> std::fmt::Debug for MmapReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapReader")
            .field("element", &T::TYPE)
            .field("dim", &self.dim)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

// ── Type-specific constructors ─────────────────────────────────────────
//
// Each is `MmapReader::<T>::open` with the type spelled in the name.
// They predate the element model, which makes the one generic open
// enough.

impl MmapReader<f32> {
    /// Open an `.fvecs` file; `MmapReader::<f32>::open` with the type in the name.
    #[deprecated(note = "use `MmapReader::<f32>::open`")]
    pub fn open_fvec(path: &Path) -> Result<Self, MmapError> {
        Self::open(path)
    }
}

impl MmapReader<i32> {
    /// Open an `.ivecs` file; `MmapReader::<i32>::open` with the type in the name.
    #[deprecated(note = "use `MmapReader::<i32>::open`")]
    pub fn open_ivec(path: &Path) -> Result<Self, MmapError> {
        Self::open(path)
    }
}

impl MmapReader<half::f16> {
    /// Open an `.mvecs` file; `MmapReader::<f16>::open` with the type in the name.
    #[deprecated(note = "use `MmapReader::<f16>::open`")]
    pub fn open_mvec(path: &Path) -> Result<Self, MmapError> {
        Self::open(path)
    }
}

impl MmapReader<f64> {
    /// Open a `.dvecs` file; `MmapReader::<f64>::open` with the type in the name.
    #[deprecated(note = "use `MmapReader::<f64>::open`")]
    pub fn open_dvec(path: &Path) -> Result<Self, MmapError> {
        Self::open(path)
    }
}

impl MmapReader<u8> {
    /// Open a `.bvecs` file (one byte per element); `MmapReader::<u8>::open` with the type in the name.
    #[deprecated(note = "use `MmapReader::<u8>::open`")]
    pub fn open_bvec(path: &Path) -> Result<Self, MmapError> {
        Self::open(path)
    }
}

impl MmapReader<i16> {
    /// Open an `.svecs` file; `MmapReader::<i16>::open` with the type in the name.
    #[deprecated(note = "use `MmapReader::<i16>::open`")]
    pub fn open_svec(path: &Path) -> Result<Self, MmapError> {
        Self::open(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_records<T: Element>(path: &Path, vectors: &[Vec<T>]) {
        let mut f = std::fs::File::create(path).unwrap();
        let mut buf = vec![0u8; T::SIZE];
        for v in vectors {
            f.write_all(&(v.len() as i32).to_le_bytes()).unwrap();
            for &val in v {
                val.write_le_bytes(&mut buf);
                f.write_all(&buf).unwrap();
            }
        }
    }

    #[test]
    fn an_fvec_reads_back_through_every_access_path() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.fvec");
        let data = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        write_records(&path, &data);

        let reader = MmapReader::<f32>::open(&path).unwrap();
        assert_eq!(reader.dim(), 3);
        assert_eq!(reader.count(), 2);
        assert_eq!(reader.get_slice(0), Some(&[1.0f32, 2.0, 3.0][..]));
        assert_eq!(reader.get_slice(1), Some(&[4.0f32, 5.0, 6.0][..]));
        assert_eq!(reader.get_slice(2), None);
        assert_eq!(reader.get(1), Some(vec![4.0, 5.0, 6.0]));
        assert_eq!(reader.get(2), None);
        let mut out = [0f32; 6];
        assert_eq!(reader.read_into(0, &mut out).unwrap(), 2);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(reader.read_into(1, &mut out).unwrap(), 1);
        assert_eq!(&out[..3], &[4.0, 5.0, 6.0]);
        assert_eq!(reader.read_into(2, &mut out).unwrap(), 0);
        assert!(matches!(reader.read_into(3, &mut out), Err(MmapError::OutOfBounds { index: 3, count: 2 })));
        assert!(matches!(reader.read_into(0, &mut out[..4]), Err(MmapError::InvalidFormat(_))));
    }

    #[test]
    fn an_ivec_reads_as_i32() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.ivec");
        write_records(&path, &[vec![10i32, 20], vec![30, 40]]);

        let reader = MmapReader::<i32>::open(&path).unwrap();
        assert_eq!(reader.dim(), 2);
        assert_eq!(reader.count(), 2);
        assert_eq!(reader.get_slice(0), Some(&[10, 20][..]));
        assert_eq!(reader.get_slice(1), Some(&[30, 40][..]));
    }

    #[test]
    fn an_empty_file_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.fvec");
        std::fs::File::create(&path).unwrap();
        assert!(matches!(MmapReader::<f32>::open(&path), Err(MmapError::InvalidFormat(_))));
    }

    #[test]
    fn a_file_without_an_extension_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("vectors");
        write_records(&path, &[vec![1.0f32]]);
        assert!(matches!(MmapReader::<f32>::open(&path), Err(MmapError::InvalidFormat(_))));
    }
}
