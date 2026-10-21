// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The column reader: values packed with no header.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::path::Path;

use veks_datatypes::{Container, Element, ElementType};

use super::{IoError, VectorReader, aligned_for, copy_elements, ext_of, records_to_read, require_native};
use crate::source::{ByteSource, MappedFile};

/// A header-free column: `T` after `T`, nothing else. The reader for a
/// scalar file (`.u8`, `.i32`, `.f32`, `.bf16`, …), with `dim() == 1`
/// and the whole column borrowable as one slice.
///
/// Like [`XvecReader`](super::XvecReader), it reads a file at its own
/// type only.
///
/// ```
/// use veks_io::{ScalarReader, VectorReader};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("scale.f32")).unwrap();
/// # for x in [0.5f32, 0.25, 2.0] { f.write_all(&x.to_le_bytes()).unwrap(); }
/// # drop(f);
/// # let mut g = std::fs::File::create(dir.path().join("hash.u64")).unwrap();
/// # for x in [u64::MAX, 1] { g.write_all(&x.to_le_bytes()).unwrap(); }
/// # drop(g);
/// let scale = ScalarReader::<f32>::open_path(&dir.path().join("scale.f32")).unwrap();
/// assert_eq!((scale.count(), scale.dim()), (3, 1));
/// assert_eq!(scale.as_slice(), Some(&[0.5f32, 0.25, 2.0][..]));     // the whole column, zero-copy
/// let hash = ScalarReader::<u64>::open_path(&dir.path().join("hash.u64")).unwrap();
/// assert_eq!(hash.get(0).unwrap(), vec![u64::MAX]);
/// ```
pub struct ScalarReader<T, S: ByteSource = MappedFile> {
    source: S,
    count: usize,
    _phantom: PhantomData<T>,
}

impl<T: Element, S: ByteSource> ScalarReader<T, S> {
    /// Open the column `source` names, whose extension names `T` in
    /// the column layout.
    pub fn open(source: &str) -> Result<Self, IoError> {
        Self::require_column(source)?;
        Self::from_source(S::open(source)?)
    }

    /// Open a local column file.
    pub fn open_path(path: &Path) -> Result<Self, IoError> {
        Self::require_column(&path.to_string_lossy())?;
        Self::from_source(S::open_path(path)?)
    }

    fn require_column(source: &str) -> Result<(), IoError> {
        match require_native::<T>(source)? {
            (_, Container::Scalar) => Ok(()),
            (file, _) => Err(IoError::InvalidFormat(format!(
                "'{}' holds vectors of {file}, not a column; read it with XvecReader",
                ext_of(source)
            ))),
        }
    }

    /// Read the bytes `source` holds as a column of `T`. The caller has
    /// established that they are one: the extension check is
    /// [`open`](Self::open)'s. A size that is not a whole number of
    /// elements is refused.
    pub fn from_source(source: S) -> Result<Self, IoError> {
        let total = source.total_size();
        if !total.is_multiple_of(T::SIZE as u64) {
            return Err(IoError::InvalidFormat(format!(
                "a column of {total} bytes is not a whole number of {}-byte {} elements",
                T::SIZE,
                T::TYPE
            )));
        }
        Ok(Self { source, count: (total / T::SIZE as u64) as usize, _phantom: PhantomData })
    }

    /// Number of elements.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The source the reader reads.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// The bytes of elements `start..start + n`, borrowed when mapped,
    /// read otherwise.
    fn bytes(&self, start: usize, n: usize) -> Result<Cow<'_, [u8]>, IoError> {
        let (offset, len) = ((start * T::SIZE) as u64, (n * T::SIZE) as u64);
        Ok(match self.source.mmap_slice(offset, len) {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(self.source.read_bytes(offset, len)?),
        })
    }
}

impl<T: Element, S: ByteSource> std::fmt::Debug for ScalarReader<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarReader")
            .field("element", &T::TYPE)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl<T: Element, S: ByteSource> VectorReader<T> for ScalarReader<T, S> {
    fn dim(&self) -> usize {
        1
    }
    fn count(&self) -> usize {
        self.count
    }
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }
        Ok(vec![T::from_le_bytes(&self.bytes(index, 1)?)])
    }
    fn get_slice(&self, index: usize) -> Option<&[T]> {
        self.as_slice()?.get(index..index + 1)
    }
    fn read_into(&self, start: usize, out: &mut [T]) -> Result<usize, IoError> {
        let n = records_to_read(start, out.len(), 1, self.count)?;
        copy_elements(&self.bytes(start, n)?, &mut out[..n]);
        Ok(n)
    }
    fn precache(&self) -> std::io::Result<()> {
        self.source.precache()
    }
    fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
    fn as_slice(&self) -> Option<&[T]> {
        let bytes = self.source.mmap_slice(0, (self.count * T::SIZE) as u64)?;
        if !aligned_for::<T>(bytes) {
            return None;
        }
        // SAFETY: `bytes` is `count * T::SIZE` bytes of the mapping on a
        // `T` boundary, and every `Element` is a plain number.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, self.count) })
    }
    fn native_element_type(&self) -> Option<ElementType> {
        Some(T::TYPE)
    }
}
