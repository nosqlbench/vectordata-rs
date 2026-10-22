// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The variable-length reader, and the `IDXFOR__` offset index that
//! lets it read a record by ordinal.

use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};
use veks_datatypes::Element;

use super::{IoError, VvecReader, ext_of, is_vvec_ext, require_native};
use crate::source::{ByteSource, MappedFile, OpenSource};

/// A file whose records vary in length: each a little-endian `i32`
/// dimension and that many elements of `T`. Random access goes through
/// an offset index, the `IDXFOR__<name>.<i32|i64>` sidecar.
///
/// On open the index is read from beside the file when one is there and
/// is current, else fetched where the source publishes one, else built
/// by walking the file once and kept beside it for next time. A remote
/// file with none of those is refused ([`IoError::OffsetIndexUnavailable`])
/// rather than downloaded whole behind the call.
///
/// ```
/// use veks_io::{IndexedVvecReader, VvecReader};
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let mut f = std::fs::File::create(dir.path().join("labels.u8vvecs")).unwrap();
/// # for v in [vec![9u8, 8, 7], vec![], vec![255]] { f.write_all(&(v.len() as i32).to_le_bytes()).unwrap(); f.write_all(&v).unwrap(); }
/// # drop(f);
/// let r = IndexedVvecReader::<u8>::open_path(&dir.path().join("labels.u8vvecs")).unwrap(); // builds IDXFOR__labels.u8vvecs.i32 beside it
/// assert_eq!(r.count(), 3);
/// assert_eq!((r.dim_at(0).unwrap(), r.dim_at(1).unwrap(), r.dim_at(2).unwrap()), (3, 0, 1));
/// assert_eq!(VvecReader::get(&r, 2).unwrap(), vec![255u8]);
/// assert!(dir.path().join("IDXFOR__labels.u8vvecs.i32").is_file());
/// ```
pub struct IndexedVvecReader<T, S: ByteSource = MappedFile> {
    source: S,
    offsets: Vec<u64>,
    elem_size: usize,
    _phantom: PhantomData<T>,
}

impl<T, S: ByteSource> std::fmt::Debug for IndexedVvecReader<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedVvecReader")
            .field("records", &self.offsets.len())
            .field("elem_size", &self.elem_size)
            .field("source", &self.source)
            .finish()
    }
}

impl<T: Element, S: ByteSource> IndexedVvecReader<T, S> {
    /// Open a local variable-length file whose extension names `T`.
    pub fn open_path(path: &Path) -> Result<Self, IoError>
    where
        S: OpenSource,
    {
        let name = path.to_string_lossy();
        Self::validate_source(&name)?;
        Self::from_source(S::open_path(path)?, &name)
    }

    /// Open the variable-length file `source` names.
    pub fn open(source: &str) -> Result<Self, IoError>
    where
        S: OpenSource,
    {
        Self::validate_source(source)?;
        Self::from_source(S::open(source)?, source)
    }

    /// Check that `source` names a variable-length file of `T`, before
    /// any bytes are read.
    pub fn validate_source(source: &str) -> Result<(), IoError> {
        let ext = ext_of(source);
        require_native::<T>(source)?;
        if !is_vvec_ext(ext) {
            return Err(IoError::InvalidFormat(format!(
                "extension '.{ext}' is not a variable-length format; use XvecReader for uniform-stride files"
            )));
        }
        Ok(())
    }

    /// Read `source`, the bytes of the file `source_name` names, as
    /// records of `T`. `source_name` locates the `IDXFOR__` sidecar:
    /// for one file of a series, it is that file's own name, as each
    /// file carries its own index in its own ordinals.
    pub fn from_source(source: S, source_name: &str) -> Result<Self, IoError> {
        Self::from_source_sized(source, source_name, T::SIZE)
    }

    /// [`from_source`](Self::from_source) for a file whose elements are
    /// `elem_size` bytes wide, read as raw records (`T = u8`) by a
    /// caller that decodes them itself.
    pub fn from_source_sized(source: S, source_name: &str, elem_size: usize) -> Result<Self, IoError> {
        let offsets = load_offsets(source_name, &source, elem_size)?;
        Ok(Self { source, offsets, elem_size, _phantom: PhantomData })
    }

    /// Records in the file.
    pub fn count(&self) -> usize {
        self.offsets.len()
    }

    /// Elements in record `index`.
    pub fn dim_at(&self, index: usize) -> Result<usize, IoError> {
        let offset = *self.offsets.get(index).ok_or(IoError::OutOfBounds(index))?;
        let header = match self.source.mmap_slice(offset, 4) {
            Some(h) => LittleEndian::read_i32(h),
            None => {
                let bytes = self.source.read_bytes(offset, 4)?;
                if bytes.len() < 4 {
                    return Err(IoError::InvalidFormat("short dim header".into()));
                }
                LittleEndian::read_i32(&bytes)
            }
        };
        Ok(header as usize)
    }

    /// Record `index`'s elements as raw little-endian bytes, without the
    /// dimension header.
    pub fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        if let Some(raw) = self.get_raw(index) {
            return Ok(raw.to_vec());
        }
        let offset = *self.offsets.get(index).ok_or(IoError::OutOfBounds(index))?;
        let dim = self.dim_at(index)?;
        Ok(self.source.read_bytes(offset + 4, (dim * self.elem_size) as u64)?)
    }

    /// Record `index`'s elements borrowed, when the bytes are mapped.
    pub fn get_raw(&self, index: usize) -> Option<&[u8]> {
        let offset = *self.offsets.get(index)?;
        let dim = LittleEndian::read_i32(self.source.mmap_slice(offset, 4)?) as usize;
        self.source.mmap_slice(offset + 4, (dim * self.elem_size) as u64)
    }

    /// Low-level: bring this one file here whole.
    pub fn precache(&self) -> std::io::Result<()> {
        self.source.precache()
    }
    /// Whether every byte is here.
    pub fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
}

impl<S: ByteSource> IndexedVvecReader<i32, S> {
    /// Record `index` as `Vec<i32>`: the trait's `get`.
    pub fn get_i32(&self, index: usize) -> Result<Vec<i32>, IoError> {
        VvecReader::get(self, index)
    }
}

impl<T: Element, S: ByteSource> VvecReader<T> for IndexedVvecReader<T, S> {
    fn count(&self) -> usize {
        self.offsets.len()
    }
    fn dim_at(&self, index: usize) -> Result<usize, IoError> {
        IndexedVvecReader::dim_at(self, index)
    }
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError> {
        IndexedVvecReader::get_bytes(self, index)
    }
    fn get(&self, index: usize) -> Result<Vec<T>, IoError> {
        match self.get_raw(index) {
            Some(raw) => Ok(super::decode_vec(raw)),
            None => Ok(super::decode_vec(&IndexedVvecReader::get_bytes(self, index)?)),
        }
    }
    fn precache(&self) -> std::io::Result<()> {
        self.source.precache()
    }
    fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
}

/// The record start offsets of the variable-length file `source_name`
/// names, whose bytes `source` holds, without the element type.
///
/// The offsets turn an ordinal range into a byte range, and nothing
/// about them depends on `T`: a caller that windows a variable-length
/// file needs them without a typed reader. This is the lookup
/// [`IndexedVvecReader`] makes, in the same order, so a windowed
/// prefetch and a later read agree where record `i` begins:
///
/// 1. a sidecar beside the local file or the local copy;
/// 2. for a remote file, the sidecar published beside it, kept beside
///    the copy once fetched;
/// 3. a walk of the file, when every byte is here, kept beside it.
///
/// A remote file with no sidecar and no complete copy is
/// [`IoError::OffsetIndexUnavailable`]: only a download of every byte
/// would find its records, and no open does that behind the caller.
pub fn load_offsets<S: ByteSource>(source_name: &str, source: &S, elem_size: usize) -> Result<Vec<u64>, IoError> {
    let total_size = source.total_size();
    // The sidecar lives beside the file the source actually opened,
    // which is not necessarily what the caller named: a `dataset.yaml`
    // facet is written relative to the dataset root, so resolving it
    // against the process CWD both misses a published sidecar and
    // writes the rebuilt one wherever the process happens to be.
    let local = source.local_path().or_else(|| (!source.is_remote()).then(|| PathBuf::from(source_name)));
    if let Some(data_path) = &local {
        // A local file's mtime says whether an older sidecar is stale; a
        // copy's moves every time a chunk lands while its content stays
        // fixed, so for one the check would throw away good indexes.
        if let Some(cached) = load_local_index(&index_path_for(data_path, total_size), data_path, !source.is_remote())? {
            return Ok(cached);
        }
    }
    if source.is_remote() {
        let data_name = source_name.rsplit('/').next().unwrap_or(source_name);
        // The width this file's size calls for first — the rule
        // `index_path_for` writes by — so the common case is one request.
        let expected = index_ext_for(total_size);
        let other = if expected == "i32" { "i64" } else { "i32" };
        for ext in [expected, other] {
            if let Some(bytes) = source.fetch_sibling(source_name, &format!("IDXFOR__{data_name}.{ext}")) {
                let offsets = parse_index_bytes(&bytes, ext, total_size);
                if let Some(cp) = &local {
                    let _ = write_index(&index_path_for(cp, total_size), &offsets, total_size);
                }
                return Ok(offsets);
            }
        }
        if !source.is_complete() {
            return Err(IoError::OffsetIndexUnavailable { url: source_name.to_string(), bytes: total_size });
        }
    }
    let offsets = match source.mmap_slice(0, total_size) {
        Some(bytes) => walk_offsets(total_size, |offset| Ok(LittleEndian::read_i32(&bytes[offset as usize..])), elem_size)?,
        None => walk_offsets(total_size, |offset| Ok(LittleEndian::read_i32(&source.read_bytes(offset, 4)?)), elem_size)?,
    };
    // Paid once per file: a read-only directory only costs the rebuild.
    if let Some(cp) = &local {
        let _ = write_index(&index_path_for(cp, total_size), &offsets, total_size);
    }
    Ok(offsets)
}

/// Walk a variable-length file of `file_size` bytes, reading each
/// record's dimension with `dim_at`.
fn walk_offsets(
    file_size: u64,
    mut dim_at: impl FnMut(u64) -> Result<i32, IoError>,
    elem_size: usize,
) -> Result<Vec<u64>, IoError> {
    let mut offsets = Vec::new();
    let mut offset: u64 = 0;
    while offset + 4 <= file_size {
        offsets.push(offset);
        let dim = dim_at(offset)?;
        if dim < 0 {
            return Err(IoError::InvalidFormat(format!("negative dimension {dim} at offset {offset}")));
        }
        offset += 4 + dim as u64 * elem_size as u64;
    }
    if offset != file_size {
        return Err(IoError::InvalidFormat(format!(
            "file does not end at a record boundary: {} bytes remaining at offset {offset}",
            file_size - offset
        )));
    }
    Ok(offsets)
}

/// Parse an `IDXFOR__` sidecar into **record start offsets**.
///
/// Two layouts are in circulation next to real data. This crate's own
/// [`write_index`] persists `N` starts; nbdatatools publishes `N + 1`
/// entries whose last is the payload size — an end sentinel that makes
/// every record's extent readable as `offsets[i + 1] - offsets[i]`
/// without a special case for the final one.
///
/// No record can *start* at the payload size, so a trailing entry equal
/// to `payload_size` is unambiguously the sentinel and is dropped here.
/// Both layouts therefore yield `N` starts.
pub(crate) fn parse_index_bytes(bytes: &[u8], ext: &str, payload_size: u64) -> Vec<u64> {
    let mut offsets: Vec<u64> = match ext {
        "i32" => bytes.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as u64).collect(),
        "i64" => bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c) as u64).collect(),
        _ => Vec::new(),
    };
    if offsets.last() == Some(&payload_size) {
        offsets.pop();
    }
    offsets
}

/// The sidecar entry width a payload of `file_size` bytes calls for:
/// `i32` while every offset still fits one, `i64` beyond that.
fn index_ext_for(file_size: u64) -> &'static str {
    if file_size <= i32::MAX as u64 { "i32" } else { "i64" }
}

fn index_path_for(data_path: &Path, file_size: u64) -> PathBuf {
    let parent = data_path.parent().unwrap_or(Path::new("."));
    let name = data_path.file_name().and_then(|n| n.to_str()).unwrap_or("data");
    parent.join(format!("IDXFOR__{name}.{}", index_ext_for(file_size)))
}

/// Load an offset sidecar beside `data_path`.
///
/// `data_can_change` says whether the data file's mtime means anything:
/// for a local file it does — an index older than its data was built
/// from a previous version — but a cache copy's mtime moves every time
/// a chunk lands while its content stays fixed. A copy whose upstream
/// changed is caught by revalidation, not here.
fn load_local_index(index_path: &Path, data_path: &Path, data_can_change: bool) -> Result<Option<Vec<u64>>, IoError> {
    if !index_path.is_file() {
        return Ok(None);
    }
    let data_meta = std::fs::metadata(data_path)?;
    if data_can_change {
        let data_mtime = data_meta.modified().map_err(|e| IoError::InvalidFormat(format!("mtime: {e}")))?;
        let index_mtime =
            std::fs::metadata(index_path)?.modified().map_err(|e| IoError::InvalidFormat(format!("mtime: {e}")))?;
        if index_mtime < data_mtime {
            return Ok(None);
        }
    }
    let data = std::fs::read(index_path)?;
    let ext = index_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    // A whole number of offsets, or the file is not one: a sidecar left
    // short by an older build, a truncated copy, or a partial download
    // sends us back to the walk rather than yield a reader that cannot
    // see the records the tail describes.
    let width = if ext == "i64" { 8 } else { 4 };
    if data.is_empty() || !data.len().is_multiple_of(width) {
        return Ok(None);
    }
    Ok(Some(parse_index_bytes(&data, ext, data_meta.len())))
}

/// Write the offset sidecar **atomically**: to a sibling temporary,
/// then renamed, so a reader opening it mid-write sees the previous
/// index or the whole new one, never a short file that would hide the
/// records at the tail. Two readers opening one file at once both walk
/// and both write; this is what makes that safe.
fn write_index(index_path: &Path, offsets: &[u64], file_size: u64) -> Result<(), IoError> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Unique per writer: the pid separates processes, the counter
    // separates threads within one.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = index_path.parent().unwrap_or(Path::new("."));
    let stem = index_path.file_name().and_then(|n| n.to_str()).unwrap_or("IDXFOR__");
    let tmp = dir.join(format!(".{stem}.{}.{}.tmp", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));

    let write = || -> Result<(), IoError> {
        let mut f = std::io::BufWriter::new(File::create(&tmp)?);
        if file_size <= i32::MAX as u64 {
            for &o in offsets {
                f.write_all(&(o as i32).to_le_bytes())?;
            }
        } else {
            for &o in offsets {
                f.write_all(&(o as i64).to_le_bytes())?;
            }
        }
        f.flush()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, index_path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

/// Write the `IDXFOR__` index of the variable-length file `data_path`,
/// `file_size` bytes whose records start at `offsets`, beside it.
pub(crate) fn write_offset_index(data_path: &Path, offsets: &[u64], file_size: u64) -> Result<(), IoError> {
    write_index(&index_path_for(data_path, file_size), offsets, file_size)
}

/// Remove any `IDXFOR__` index beside a data file, before rewriting it,
/// so a stale index is not read. The next open rebuilds it.
pub fn remove_vvec_index(data_path: &Path) {
    let parent = data_path.parent().unwrap_or(Path::new("."));
    let name = data_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.is_empty() {
        return;
    }
    for ext in ["i32", "i64"] {
        let idx = parent.join(format!("IDXFOR__{name}.{ext}"));
        if idx.exists() {
            let _ = std::fs::remove_file(&idx);
        }
    }
}

#[cfg(test)]
mod sidecar_layouts {
    use super::*;

    /// Starts-only — what this crate's own `write_index` persists.
    #[test]
    fn a_starts_only_sidecar_keeps_every_entry() {
        // Three records of 4 bytes each in a 12-byte payload.
        let bytes: Vec<u8> = [0i32, 4, 8].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(parse_index_bytes(&bytes, "i32", 12), vec![0, 4, 8]);
    }

    /// Sentinel — what nbdatatools publishes. The trailing entry is the
    /// payload size, which lets a consumer take every record's extent
    /// as `offsets[i + 1] - offsets[i]`. It is not a record.
    #[test]
    fn an_end_sentinel_is_not_a_record() {
        let bytes: Vec<u8> = [0i32, 4, 8, 12]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            parse_index_bytes(&bytes, "i32", 12),
            vec![0, 4, 8],
            "a final entry equal to the payload size is the sentinel; kept, \
             it reads as a fourth record starting at EOF"
        );
    }

    /// The same, at the wider entry width.
    #[test]
    fn the_sentinel_is_dropped_at_i64_width_too() {
        let bytes: Vec<u8> = [0i64, 20, 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(parse_index_bytes(&bytes, "i64", 40), vec![0, 20]);
    }

    /// Only the *last* entry can be the sentinel. An interior offset
    /// equal to the payload size would be a corrupt index, not a
    /// layout, and silently dropping it would hide that.
    #[test]
    fn only_a_trailing_entry_is_treated_as_the_sentinel() {
        let bytes: Vec<u8> = [0i32, 12, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(parse_index_bytes(&bytes, "i32", 12), vec![0, 12, 4]);
    }

    /// A record needs at least a 4-byte header, so no record can start
    /// at the payload size — which is what makes the sentinel test
    /// unambiguous rather than a heuristic.
    #[test]
    fn the_sentinel_value_is_never_a_legal_record_start() {
        let dims: [i32; 3] = [2, 5, 1];
        let mut at = 0u64;
        let starts: Vec<u64> = dims
            .iter()
            .map(|&d| {
                let s = at;
                at += 4 + d as u64 * 4;
                s
            })
            .collect();
        assert!(
            starts.iter().all(|&s| s < at),
            "every record start is strictly inside the payload"
        );
    }

    /// The width a payload calls for decides both the sidecar's name
    /// and which remote sidecar is probed first. One rule, so the two
    /// cannot drift apart and cost a guaranteed 404 on every open.
    #[test]
    fn the_entry_width_follows_the_payload_size() {
        assert_eq!(index_ext_for(0), "i32");
        assert_eq!(index_ext_for(i32::MAX as u64), "i32");
        assert_eq!(index_ext_for(i32::MAX as u64 + 1), "i64");
        let p = std::path::Path::new("/d/x.ivvec");
        assert_eq!(
            index_path_for(p, 1024).file_name().unwrap(),
            "IDXFOR__x.ivvec.i32"
        );
        assert_eq!(
            index_path_for(p, i32::MAX as u64 + 1).file_name().unwrap(),
            "IDXFOR__x.ivvec.i64"
        );
    }
}

#[cfg(test)]
mod index_atomicity {
    use super::*;

    /// **A half-written sidecar must never be read as a whole one.**
    ///
    /// `write_index` used to truncate the target in place and fill it
    /// offset by offset, so a concurrent reader saw a short file and
    /// parsed fewer offsets than the data had records — a reader that
    /// silently could not see the tail of its own file. It surfaced as
    /// a one-in-six flake in a test that opened one vvec from eight
    /// threads.
    #[test]
    fn many_writers_never_leave_a_readable_partial_index() {
        let tmp = tempfile::tempdir().unwrap();
        let index = tmp.path().join("IDXFOR__x.ivvec.i32");
        let offsets: Vec<u64> = (0..4096).map(|i| i * 12).collect();

        // Writers racing on one path, readers observing it throughout.
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let index = index.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut bad = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        if let Ok(bytes) = std::fs::read(&index)
                            && !bytes.is_empty()
                            && bytes.len() != offsets_len_bytes()
                        {
                            bad += 1;
                        }
                    }
                    bad
                })
            })
            .collect();

        let writers: Vec<_> = (0..4)
            .map(|_| {
                let index = index.clone();
                let offsets = offsets.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        write_index(&index, &offsets, 1 << 20).unwrap();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);

        let partial: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
        assert_eq!(
            partial, 0,
            "readers observed {partial} partial sidecars; the swap is not atomic"
        );

        // And no temporaries are left lying about.
        let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temporaries left behind: {leftovers:?}"
        );
    }

    fn offsets_len_bytes() -> usize {
        4096 * 4
    }
}
