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

/// Open an xvec file, a directory of them, or a series (`<stem>__NNNN`)
/// as one streaming [`VecSource`], each file checked for the same
/// dimension and a whole number of records.
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
    if elem_size == 0 || !format.is_uniform_xvec() {
        return Err(format!("{} is not a uniform xvec format", format));
    }
    let mut parts = Vec::new();
    let mut dim = None;
    for file in files_of(path, format)? {
        let mapped = MappedFile::map(&file).map_err(|e| format!("failed to open {}: {}", file.display(), e))?;
        let bytes = mapped.bytes();
        if bytes.len() < 4 {
            return Err(format!("{}: {} bytes is too short for a record header", file.display(), bytes.len()));
        }
        let first = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let (d, record_bytes, records) = shape(&file, first, bytes.len() as u64, elem_size)?;
        match dim {
            None => dim = Some((d, record_bytes)),
            Some((want, _)) if want != d => {
                return Err(format!("{}: dimension {d} where the files before it hold {want}", file.display()));
            }
            Some(_) => {}
        }
        mapping::advise_sequential_bytes(bytes);
        parts.push(Part { file: mapped, records });
    }
    let (dim, record_bytes) = dim.ok_or_else(|| format!("no {format} files at {}", path.display()))?;
    let total_records = parts.iter().map(|p| p.records).sum();
    Ok(Box::new(XvecReader { parts, part: 0, dim, record_bytes, total_records, read_in_part: 0, released: 0 }))
}

/// The files `path` names for `format`: the file itself; the files of
/// that format in a directory, in name order (the order a series'
/// `__NNNN` names sort in); or, when `path` itself is absent, the
/// shards of the series it is the unsharded name of.
fn files_of(path: &Path, format: VecFormat) -> Result<Vec<std::path::PathBuf>, String> {
    if path.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(path)
            .map_err(|e| format!("read directory {}: {e}", path.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()).and_then(VecFormat::from_extension) == Some(format))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no {format} files in {}", path.display()));
        }
        return Ok(files);
    }
    if !path.exists() {
        let shards = crate::series::discover_shards(path);
        if !shards.is_empty() {
            return Ok(shards);
        }
    }
    Ok(vec![path.to_path_buf()])
}

/// Probe an xvec file, directory or series for its dimension, element
/// size and record count, reading one header per file.
///
/// ```
/// use veks_io::{VecFormat, xvec};
/// # let dir = tempfile::tempdir().unwrap();
/// for (name, rows) in [("b__0000.fvecs", 3), ("b__0001.fvecs", 2)] {
///     let mut w = xvec::writer::open(&dir.path().join(name), 4).unwrap();
///     for _ in 0..rows {
///         w.write_record(0, &[0u8; 16]);
///     }
///     w.finish().unwrap();
/// }
/// let meta = xvec::reader::probe(dir.path(), VecFormat::Fvec).unwrap();
/// assert_eq!((meta.dimension, meta.element_size, meta.record_count), (4, 4, Some(5)));
/// ```
pub fn probe(path: &Path, format: VecFormat) -> Result<SourceMeta, String> {
    let elem_size = format.element_size();
    if elem_size == 0 {
        return Err(format!("{} is not an xvec format", format));
    }
    let mut dim = None;
    let mut count = 0u64;
    for file in files_of(path, format)? {
        let mut f = File::open(&file).map_err(|e| format!("failed to open {}: {}", file.display(), e))?;
        let file_size = f.metadata().map_err(|e| format!("metadata: {}", e))?.len();
        let first = f.read_i32::<LittleEndian>().map_err(|e| format!("read dim: {}", e))?;
        let (d, _, records) = shape(&file, first, file_size, elem_size)?;
        if dim.is_some_and(|want| want != d) {
            return Err(format!("{}: dimension {d} where the files before it hold {}", file.display(), dim.unwrap_or(d)));
        }
        dim = Some(d);
        count += records;
    }
    Ok(SourceMeta {
        dimension: dim.ok_or_else(|| format!("no {format} files at {}", path.display()))?,
        element_size: elem_size,
        record_count: Some(count),
    })
}

/// One mapped file of the stream.
struct Part {
    file: MappedFile,
    records: u64,
}

struct XvecReader {
    parts: Vec<Part>,
    /// The part being read.
    part: usize,
    dim: u32,
    record_bytes: usize,
    total_records: u64,
    /// Records read from the current part.
    read_in_part: u64,
    /// Bytes of the current part whose pages have been released.
    released: u64,
}

impl XvecReader {
    fn next_slice(&mut self) -> Option<&[u8]> {
        while self.part < self.parts.len() && self.read_in_part >= self.parts[self.part].records {
            // Done with a part: release what is left of it before the next.
            let bytes = self.parts[self.part].file.bytes();
            mapping::release_range_bytes(bytes, self.released, bytes.len() as u64);
            self.part += 1;
            self.read_in_part = 0;
            self.released = 0;
        }
        let part = self.parts.get(self.part)?;
        let at = self.read_in_part as usize * self.record_bytes;
        let bytes = part.file.bytes();
        let dim = i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        if dim as u32 != self.dim {
            return None; // dimension mismatch
        }
        self.read_in_part += 1;
        let end = (at + self.record_bytes) as u64;
        if end - self.released >= RELEASE_WINDOW {
            mapping::release_range_bytes(bytes, self.released, end);
            self.released = end;
        }
        Some(&self.parts[self.part].file.bytes()[at + 4..at + self.record_bytes])
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
