// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Streaming xvec reader.
//!
//! Reads xvec files record by record out of a read-only mapping,
//! releasing the pages behind the cursor as it goes, so a scan of a
//! file larger than memory keeps a bounded resident set.

use std::fs::File;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::format::VecFormat;
use crate::source::{MappedFile, mapping};
use crate::traits::{SourceMeta, VecSource};

/// Bytes scanned between releases of the pages behind the cursor:
/// large enough that the `madvise` costs nothing, small enough that
/// the resident set stays bounded.
pub(crate) const RELEASE_WINDOW: u64 = 256 << 20;

/// The shape of an xvec file from its first header and its size: the
/// dimension, the record size, and the record count. The file must be
/// a whole number of records of a positive dimension; a short file is
/// refused here rather than read up to where it stops, because a tail
/// that is silently dropped is a record a consumer never sees.
fn shape(path: &Path, dim: i32, file_size: u64, elem_size: usize) -> Result<(u32, usize, u64), String> {
    if dim <= 0 {
        return Err(format!("{}: dimension {dim} is not positive", path.display()));
    }
    let record_bytes = (dim as usize)
        .checked_mul(elem_size)
        .and_then(|payload| payload.checked_add(4))
        .ok_or_else(|| format!("{}: a record of dimension {dim} overflows", path.display()))?;
    if !file_size.is_multiple_of(record_bytes as u64) {
        return Err(format!(
            "{}: file size {file_size} is not a whole number of {record_bytes}-byte records",
            path.display()
        ));
    }
    Ok((dim as u32, record_bytes, file_size / record_bytes as u64))
}

/// Open an xvec file as a streaming [`VecSource`].
///
/// # Example
///
/// A round trip at `u64`: written through
/// [`xvec::writer::open`](crate::xvec::writer::open), probed, then
/// streamed back one copied record at a time.
///
/// ```
/// use veks_io::{VecFormat, xvec};
/// # let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("ids.u64vecs");
///
/// let mut writer = xvec::writer::open(&path, 2).unwrap();
/// for row in [[1u64, 2], [u64::MAX, 0]] {
///     let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
///     writer.write_record(0, &bytes);
/// }
/// writer.finish().unwrap();
///
/// let meta = xvec::reader::probe(&path, VecFormat::U64vec).unwrap();
/// assert_eq!((meta.dimension, meta.element_size, meta.record_count), (2, 8, Some(2)));
///
/// let mut reader = xvec::reader::open(&path, VecFormat::U64vec).unwrap();
/// let first = reader.next_record().unwrap();
/// assert_eq!(u64::from_le_bytes(first[8..16].try_into().unwrap()), 2);
/// let second = reader.next_record().unwrap();
/// assert_eq!(u64::from_le_bytes(second[0..8].try_into().unwrap()), u64::MAX);
/// assert!(reader.next_record().is_none());
/// ```
pub fn open(path: &Path, format: VecFormat) -> Result<Box<dyn VecSource>, String> {
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} is not an xvec format", format));
    }
    let file = MappedFile::map(path).map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let bytes = file.bytes();
    if bytes.len() < 4 {
        return Err(format!("read dim: {} bytes is too short for a record header", bytes.len()));
    }
    let dim = i32::from_le_bytes(bytes[..4].try_into().expect("four bytes"));
    let (dim, record_bytes, total_records) = shape(path, dim, bytes.len() as u64, elem_size)?;
    mapping::advise_sequential_bytes(bytes);
    Ok(Box::new(XvecReader { file, dim, record_bytes, total_records, records_read: 0, released: 0 }))
}

/// Probe an xvec file for metadata without reading all data.
///
/// # Example
///
/// The element size is the format's (`u16` is 2 bytes); the count is
/// the file size over the record size, and a file that is not a whole
/// number of records is refused rather than truncated:
///
/// ```
/// use veks_io::{VecFormat, xvec};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("counts.u16vecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # for row in [[1u16, 2, 3], [4, 5, 6]] {
/// #     f.write_all(&3i32.to_le_bytes()).unwrap();
/// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
/// # }
/// # drop(f);
/// let meta = xvec::reader::probe(&path, VecFormat::U16vec).unwrap();
/// assert_eq!((meta.dimension, meta.element_size, meta.record_count), (3, 2, Some(2)));
///
/// // the same bytes read as a 1-byte type are not a whole number of 7-byte records
/// let err = xvec::reader::probe(&path, VecFormat::Bvec).unwrap_err();
/// assert!(err.contains("not a whole number"), "{err}");
/// ```
pub fn probe(path: &Path, format: VecFormat) -> Result<SourceMeta, String> {
    let file = File::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let file_size = file.metadata()
        .map_err(|e| format!("metadata: {}", e))?
        .len();
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} is not an xvec format", format));
    }

    let mut reader = file;
    let dim = reader.read_i32::<LittleEndian>()
        .map_err(|e| format!("read dim: {}", e))?;
    let (dim, _, count) = shape(path, dim, file_size, elem_size)?;

    Ok(SourceMeta {
        dimension: dim,
        element_size: elem_size,
        record_count: Some(count),
    })
}

struct XvecReader {
    file: MappedFile,
    dim: u32,
    record_bytes: usize,
    total_records: u64,
    records_read: u64,
    /// Bytes behind the cursor already released.
    released: u64,
}

impl XvecReader {
    /// The next record's element bytes, borrowed from the mapping, and
    /// the cursor moved past it. `None` at the end, and at a record
    /// whose dimension is not the file's.
    fn next_slice(&mut self) -> Option<&[u8]> {
        if self.records_read >= self.total_records {
            return None;
        }
        let at = self.records_read as usize * self.record_bytes;
        let bytes = self.file.bytes();
        let dim = i32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
        if dim as u32 != self.dim {
            return None; // dimension mismatch
        }
        self.records_read += 1;
        let end = (at + self.record_bytes) as u64;
        if end - self.released >= RELEASE_WINDOW {
            mapping::release_range_bytes(bytes, self.released, end);
            self.released = end;
        }
        Some(&self.file.bytes()[at + 4..at + self.record_bytes])
    }
}

impl VecSource for XvecReader {
    fn dimension(&self) -> u32 { self.dim }
    fn element_size(&self) -> usize { (self.record_bytes - 4) / self.dim as usize }
    fn record_count(&self) -> Option<u64> { Some(self.total_records) }

    fn next_record(&mut self) -> Option<Vec<u8>> {
        self.next_slice().map(<[u8]>::to_vec)
    }

    fn read_record_into(&mut self, buf: &mut Vec<u8>) -> bool {
        match self.next_slice() {
            Some(record) => {
                buf.clear();
                buf.extend_from_slice(record);
                true
            }
            None => false,
        }
    }
}
