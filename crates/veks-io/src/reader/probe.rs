// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! What a file is, before it is opened as one: an xvec source's shape
//! from its first header, tolerating a source that is partial, sparse
//! or truncated ([`probe_source`]), and a variable-length series' record
//! offsets across its files ([`load_series_offsets`]).

use veks_datatypes::ElementType;

use super::{IoError, load_offsets};
use crate::source::ByteSource;

/// The shape of an xvec source, as its first header and its size say.
///
/// A whole file has [`trailing_bytes`](Self::trailing_bytes) zero; a
/// download cut off mid-record, or a file with garbage past its last
/// record, does not, and [`records`](Self::records) counts only the
/// whole ones. [`complete`](Self::complete) says whether every byte is
/// here, so a caller tells a partial download from a truncated file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XvecProbe {
    /// Elements per record, from the first header.
    pub dim: usize,
    /// Bytes per record: the header and `dim` elements.
    pub record_bytes: u64,
    /// Whole records in the source.
    pub records: u64,
    /// Bytes past the last whole record.
    pub trailing_bytes: u64,
    /// Whether every byte of the source is here.
    pub complete: bool,
}

/// Read `source`'s first header and size as uniform `element` records.
///
/// Reads four bytes — one chunk of a remote source — and nothing else,
/// so it describes a file that is not whole rather than refusing it: a
/// half-downloaded base, a sparse cache file, an artifact a crash cut
/// short. A header that is not a positive dimension is
/// [`IoError::InvalidFormat`].
///
/// ```
/// use veks_io::reader::probe_source;
/// use veks_io::{ElementType, MappedFile};
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("cut.fvecs");
/// // Two dim-2 f32 records, then three bytes of a third.
/// let mut bytes = Vec::new();
/// for _ in 0..2 {
///     bytes.extend(2i32.to_le_bytes());
///     bytes.extend([0u8; 8]);
/// }
/// bytes.extend([2, 0, 0]);
/// std::fs::write(&path, &bytes).unwrap();
///
/// let probe = probe_source(&MappedFile::map(&path).unwrap(), ElementType::F32).unwrap();
/// assert_eq!((probe.dim, probe.records, probe.trailing_bytes), (2, 2, 3));
/// assert!(probe.complete, "a local file is all here: the tail is garbage, not a gap");
/// ```
pub fn probe_source<S: ByteSource>(source: &S, element: ElementType) -> Result<XvecProbe, IoError> {
    let size = source.total_size();
    if size < 4 {
        return Err(IoError::InvalidFormat(format!("{size} bytes is too short for a record header")));
    }
    let header = match source.mmap_slice(0, 4) {
        Some(h) => [h[0], h[1], h[2], h[3]],
        None => {
            let h = source.read_bytes(0, 4)?;
            [h[0], h[1], h[2], h[3]]
        }
    };
    let dim = i32::from_le_bytes(header);
    let record_bytes = crate::reader::xvec_record_bytes(dim, element)? as u64;
    Ok(XvecProbe {
        dim: dim as usize,
        record_bytes,
        records: size / record_bytes,
        trailing_bytes: size % record_bytes,
        complete: source.is_complete(),
    })
}

/// The record offsets of a variable-length series: each file's offsets
/// ([`load_offsets`], so each file's `IDXFOR__` index is found or built
/// as for one file), shifted by the bytes of the files before it, so
/// record `i` of the series begins at offset `i` of the result in the
/// series' joined byte space.
///
/// ```
/// use veks_io::reader::load_series_offsets;
/// use veks_io::writer::VvecWriter;
/// use veks_io::MappedFile;
/// # let dir = tempfile::tempdir().unwrap();
/// let mut parts = Vec::new();
/// for (name, rows) in [("m__0000.ivvecs", vec![vec![1i32], vec![2, 3]]), ("m__0001.ivvecs", vec![vec![4, 5, 6]])] {
///     let path = dir.path().join(name);
///     let mut w = VvecWriter::<i32>::create(&path).unwrap();
///     w.write_all(rows).unwrap();
///     w.finish().unwrap();
///     parts.push((path.to_string_lossy().into_owned(), MappedFile::map(&path).unwrap()));
/// }
/// // 8 bytes then 12 in the first file; the second file starts at 20.
/// assert_eq!(load_series_offsets(&parts, 4).unwrap(), vec![0, 8, 20]);
/// ```
pub fn load_series_offsets<S: ByteSource>(parts: &[(String, S)], elem_size: usize) -> Result<Vec<u64>, IoError> {
    let mut out = Vec::new();
    let mut base = 0u64;
    for (name, source) in parts {
        out.extend(load_offsets(name, source, elem_size)?.into_iter().map(|o| base + o));
        base += source.total_size();
    }
    Ok(out)
}
