// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Typed writers: rows of `T` into the file layout the extension names,
//! the writing counterparts of [`XvecReader`](crate::XvecReader),
//! [`ScalarReader`](crate::ScalarReader) and
//! [`IndexedVvecReader`](crate::IndexedVvecReader).
//!
//! Each refuses an extension that names another element type or
//! layout, checks every row it is given, and returns every I/O error:
//! a write that fails is an error from that call or from `finish`,
//! never a short file. Call `finish`; dropping a writer flushes what it
//! can and reports nothing.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use veks_datatypes::{Container, Element};

use crate::reader::{IoError, ext_of, is_vvec_ext, require_native};

/// Buffer between a writer and its file.
const WRITE_BUF_SIZE: usize = 4 << 20;

/// The elements of `row` as their little-endian bytes: the row itself
/// on a little-endian host, an encoded copy elsewhere.
fn le_bytes<T: Element>(row: &[T]) -> std::borrow::Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: every `Element` is a plain number whose in-memory form
        // on this host is its little-endian encoding.
        std::borrow::Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(row.as_ptr() as *const u8, std::mem::size_of_val(row))
        })
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = vec![0u8; row.len() * T::SIZE];
        for (v, slot) in row.iter().zip(out.chunks_exact_mut(T::SIZE)) {
            v.write_le_bytes(slot);
        }
        std::borrow::Cow::Owned(out)
    }
}

/// Check that `path` names a `container` file of `T`, and create it.
fn create_for<T: Element>(path: &Path, container: Container, reader: &str) -> Result<BufWriter<File>, IoError> {
    let name = path.to_string_lossy();
    let (file, found) = require_native::<T>(&name)?;
    let found = if found == Container::Vec && container == Container::Vvec && is_vvec_ext(ext_of(&name)) {
        // The bare `ivec(s)` has named variable-length files from the
        // first datasets on, and the vvec reader opens them.
        Container::Vvec
    } else {
        found
    };
    if found != container {
        return Err(IoError::InvalidFormat(format!(
            "'{}' names a {} file of {file}; write it with {reader}",
            ext_of(&name),
            match found {
                Container::Scalar => "column",
                Container::Vvec => "variable-length",
                _ => "uniform vector",
            }
        )));
    }
    crate::xvec::writer::refuse_symlink(path).map_err(IoError::InvalidFormat)?;
    Ok(BufWriter::with_capacity(WRITE_BUF_SIZE, File::create(path)?))
}

/// Rows of `T`, every one `dim` elements, into a uniform vector file
/// (`.fvecs`, `.bvecs`, `.bf16vecs`, …) whose extension names `T`.
///
/// ```
/// use veks_io::{IoError, VectorReader, XvecReader, XvecWriter};
/// # let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("base.fvecs");
/// let mut writer = XvecWriter::<f32>::create(&path, 3).unwrap();
/// writer.write(&[1.0, 2.0, 3.0]).unwrap();
/// writer.write_all([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
/// assert!(matches!(writer.write(&[1.0]), Err(IoError::InvalidFormat(_))), "a row of the wrong length");
/// assert_eq!(writer.finish().unwrap(), 3);
///
/// let reader = XvecReader::<f32>::open_path(&path).unwrap();
/// assert_eq!(reader.get(2).unwrap(), vec![7.0, 8.0, 9.0]);
/// // An extension names its type: f32 rows are not a .bvecs file.
/// assert!(matches!(XvecWriter::<f32>::create(&dir.path().join("codes.bvecs"), 3), Err(IoError::ElementMismatch { .. })));
/// ```
pub struct XvecWriter<T: Element> {
    out: BufWriter<File>,
    path: PathBuf,
    dim: usize,
    header: [u8; 4],
    records: u64,
    _t: PhantomData<T>,
}

impl<T: Element> XvecWriter<T> {
    /// Create `path` for rows of `dim` elements. `dim` must be 1 to
    /// `i32::MAX`, and the extension must name `T` in the uniform
    /// layout.
    pub fn create(path: &Path, dim: usize) -> Result<Self, IoError> {
        let header = i32::try_from(dim)
            .ok()
            .filter(|&d| d > 0)
            .ok_or_else(|| IoError::InvalidFormat(format!("dimension {dim} is not 1 to {}", i32::MAX)))?
            .to_le_bytes();
        let out = create_for::<T>(path, Container::Vec, "XvecWriter")?;
        Ok(Self { out, path: path.to_path_buf(), dim, header, records: 0, _t: PhantomData })
    }

    /// Write one row, which must be `dim` elements.
    pub fn write(&mut self, row: &[T]) -> Result<(), IoError> {
        if row.len() != self.dim {
            return Err(IoError::InvalidFormat(format!(
                "a row of {} elements in a file of dimension {} ({})",
                row.len(),
                self.dim,
                self.path.display()
            )));
        }
        self.out.write_all(&self.header)?;
        self.out.write_all(&le_bytes(row))?;
        self.records += 1;
        Ok(())
    }

    /// Write every row, in order.
    pub fn write_all<R: AsRef<[T]>>(&mut self, rows: impl IntoIterator<Item = R>) -> Result<(), IoError> {
        rows.into_iter().try_for_each(|row| self.write(row.as_ref()))
    }

    /// Write `rows.len() / dim` rows packed one after another, as a
    /// [`VectorReader::read_into`](crate::VectorReader::read_into)
    /// buffer holds them. The length must be a whole number of rows.
    pub fn write_packed(&mut self, rows: &[T]) -> Result<(), IoError> {
        if !rows.len().is_multiple_of(self.dim) {
            return Err(IoError::InvalidFormat(format!(
                "{} elements is not a whole number of {}-element rows",
                rows.len(),
                self.dim
            )));
        }
        rows.chunks_exact(self.dim).try_for_each(|row| self.write(row))
    }

    /// Rows written so far.
    pub fn records(&self) -> u64 {
        self.records
    }

    /// Flush the file and return how many rows it holds.
    pub fn finish(mut self) -> Result<u64, IoError> {
        self.out.flush()?;
        Ok(self.records)
    }
}

/// Values of `T` packed with no header into a column (`.u8`, `.i32`,
/// `.f32`, `.bf16`, …) whose extension names `T`.
///
/// ```
/// use veks_io::{ScalarReader, ScalarWriter, VectorReader};
/// # let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("hash.u64");
/// let mut writer = ScalarWriter::<u64>::create(&path).unwrap();
/// writer.write(&[u64::MAX, 1]).unwrap();
/// writer.write(&[7]).unwrap();
/// assert_eq!(writer.finish().unwrap(), 3);
/// assert_eq!(ScalarReader::<u64>::open_path(&path).unwrap().as_slice(), Some(&[u64::MAX, 1, 7][..]));
/// ```
pub struct ScalarWriter<T: Element> {
    out: BufWriter<File>,
    values: u64,
    _t: PhantomData<T>,
}

impl<T: Element> ScalarWriter<T> {
    /// Create `path`, whose extension must name `T` in the column
    /// layout.
    pub fn create(path: &Path) -> Result<Self, IoError> {
        Ok(Self { out: create_for::<T>(path, Container::Scalar, "ScalarWriter")?, values: 0, _t: PhantomData })
    }

    /// Append `values` to the column.
    pub fn write(&mut self, values: &[T]) -> Result<(), IoError> {
        self.out.write_all(&le_bytes(values))?;
        self.values += values.len() as u64;
        Ok(())
    }

    /// Values written so far.
    pub fn values(&self) -> u64 {
        self.values
    }

    /// Flush the file and return how many values it holds.
    pub fn finish(mut self) -> Result<u64, IoError> {
        self.out.flush()?;
        Ok(self.values)
    }
}

/// Rows of `T` of any length into a variable-length file (`.ivvecs`,
/// `.fvvecs`, …) whose extension names `T`, with its `IDXFOR__` offset
/// index written beside it on [`finish`](Self::finish), so the first
/// reader needs no walk.
///
/// ```
/// use veks_io::{IndexedVvecReader, VvecReader, VvecWriter};
/// # let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("matches.i32vvecs");
/// let mut writer = VvecWriter::<i32>::create(&path).unwrap();
/// writer.write_all([&[1, 2, 3][..], &[], &[-7]]).unwrap();
/// assert_eq!(writer.finish().unwrap(), 3);
/// assert!(dir.path().join("IDXFOR__matches.i32vvecs.i32").is_file());
///
/// let reader = IndexedVvecReader::<i32>::open_path(&path).unwrap();
/// assert_eq!((reader.dim_at(0).unwrap(), reader.dim_at(1).unwrap()), (3, 0));
/// assert_eq!(VvecReader::get(&reader, 2).unwrap(), vec![-7]);
/// ```
pub struct VvecWriter<T: Element> {
    out: BufWriter<File>,
    path: PathBuf,
    offsets: Vec<u64>,
    bytes: u64,
    _t: PhantomData<T>,
}

impl<T: Element> VvecWriter<T> {
    /// Create `path`, whose extension must name `T` in the
    /// variable-length layout. Any stale `IDXFOR__` index beside it is
    /// removed, so no reader pairs the new file with the old index.
    pub fn create(path: &Path) -> Result<Self, IoError> {
        let out = create_for::<T>(path, Container::Vvec, "VvecWriter")?;
        crate::reader::remove_vvec_index(path);
        Ok(Self { out, path: path.to_path_buf(), offsets: Vec::new(), bytes: 0, _t: PhantomData })
    }

    /// Write one row, of any length up to `i32::MAX` elements.
    pub fn write(&mut self, row: &[T]) -> Result<(), IoError> {
        let dim = i32::try_from(row.len())
            .map_err(|_| IoError::InvalidFormat(format!("a row of {} elements is too long", row.len())))?;
        self.out.write_all(&dim.to_le_bytes())?;
        self.out.write_all(&le_bytes(row))?;
        self.offsets.push(self.bytes);
        self.bytes += 4 + std::mem::size_of_val(row) as u64;
        Ok(())
    }

    /// Write every row, in order.
    pub fn write_all<R: AsRef<[T]>>(&mut self, rows: impl IntoIterator<Item = R>) -> Result<(), IoError> {
        rows.into_iter().try_for_each(|row| self.write(row.as_ref()))
    }

    /// Rows written so far.
    pub fn records(&self) -> u64 {
        self.offsets.len() as u64
    }

    /// Flush the file, write its offset index beside it, and return how
    /// many rows it holds.
    pub fn finish(mut self) -> Result<u64, IoError> {
        self.out.flush()?;
        drop(self.out);
        crate::reader::write_offset_index(&self.path, &self.offsets, self.bytes)?;
        Ok(self.offsets.len() as u64)
    }
}
