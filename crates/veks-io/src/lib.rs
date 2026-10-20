// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! # veks-io — Standalone Vector Data I/O
//!
//! Read and write vector data in the xvec family (fvec, ivec, mvec,
//! dvec, bvec, svec, and the typed spellings up to bf16vecs), the
//! header-free scalar columns of every [`ElementType`], and
//! (optionally) npy, parquet, and slab formats.
//!
//! The element model — which types a file may hold, what an extension
//! says, and what a reader may do when asked for another type — is
//! `veks-datatypes`, re-exported here ([`ElementType`], [`Element`],
//! [`ElementConversion`], [`Container`], [`struct@bf16`],
//! [`struct@f16`]). The readers
//! in this crate read a file at its own type only; the
//! [`ElementMismatch`](MmapError::ElementMismatch) error is what a
//! wrong type gets.
//!
//! ## Quick Start
//!
//! The streaming API moves raw little-endian bytes; the element type
//! is whatever the extension names, here `f32` for `.fvecs`.
//!
//! ```
//! use veks_io::{open, create, probe};
//! # let dir = tempfile::tempdir().unwrap();
//! let path = dir.path().join("dataset.fvecs");
//!
//! // Write records: each is dim × 4 bytes of f32, no header
//! let mut writer = create(&path, 3).unwrap();
//! for row in [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]] {
//!     let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
//!     writer.write_record(0, &bytes);
//! }
//! writer.finish().unwrap();
//!
//! // Probe a file for metadata (reads one header, not the data)
//! let meta = probe(&path).unwrap();
//! assert_eq!((meta.dimension, meta.element_size, meta.record_count), (3, 4, Some(2)));
//!
//! // Stream-read all records: each is a fresh Vec<u8> of element bytes
//! let mut reader = open(&path).unwrap();
//! let mut sum = 0.0f32;
//! while let Some(record) = reader.next_record() {
//!     for chunk in record.chunks_exact(4) {
//!         sum += f32::from_le_bytes(chunk.try_into().unwrap());
//!     }
//! }
//! assert_eq!(sum, 21.0);
//! ```
//!
//! ## Zero-Copy Mmap Access
//!
//! For random access on xvec files, use the typed mmap reader at the
//! file's own type. `get_slice` borrows from the mapping (zero-copy);
//! `read_into` decodes into a caller's buffer (one copy).
//!
//! ```
//! use veks_io::MmapReader;
//! # use std::io::Write;
//! # let dir = tempfile::tempdir().unwrap();
//! # let path = dir.path().join("base.fvecs");
//! # let mut f = std::fs::File::create(&path).unwrap();
//! # for row in [[0.5f32, 1.5], [2.5, 3.5], [4.5, 5.5]] {
//! #     f.write_all(&2i32.to_le_bytes()).unwrap();
//! #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
//! # }
//! # drop(f);
//!
//! let reader = MmapReader::<f32>::open(&path).unwrap();
//! let vec: &[f32] = reader.get_slice(1).unwrap(); // zero-copy
//! assert_eq!(vec, &[2.5, 3.5]);
//!
//! let mut rows = vec![0f32; reader.dim() * 2];
//! let read = reader.read_into(1, &mut rows).unwrap(); // decoded copy of records 1 and 2
//! assert_eq!(read, 2);
//! assert_eq!(rows, [2.5, 3.5, 4.5, 5.5]);
//! ```
//!
//! A header-free column is one slice, at the type its extension names
//! (`.f32` here; `.u8`, `.i64`, `.bf16` and the rest work the same way):
//!
//! ```
//! use veks_io::scalar::mmap::ScalarMmapReader;
//! # let dir = tempfile::tempdir().unwrap();
//! # let path = dir.path().join("quant_scale.f32");
//! # let bytes: Vec<u8> = [0.25f32, 0.5, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
//! # std::fs::write(&path, bytes).unwrap();
//!
//! let scale = ScalarMmapReader::<f32>::open(&path).unwrap();
//! let all: &[f32] = scale.as_slice().unwrap(); // zero-copy, the whole column
//! assert_eq!(all, &[0.25, 0.5, 1.0]);
//! ```
//!
//! ## Feature Flags
//!
//! - `npy` — NumPy `.npy` directory reading (adds ndarray dependency)
//! - `parquet` — Apache Parquet directory reading (adds arrow/parquet dependencies)
//! - `slab` — Slab format read/write (adds slabtastic dependency)

#![warn(missing_docs)]

pub mod element;
pub mod format;
pub mod scalar;
pub mod traits;
pub mod xvec;

pub use element::{Container, Element, ElementConversion, ElementKind, ElementType, FloatFormat, MmapError, bf16, f16};
pub use format::VecFormat;
pub use traits::{SourceMeta, VecSink, VecSource};
pub use traits::{VarlenRecord, VarlenSink, VarlenSource};
pub use xvec::mmap::MmapReader;

use std::path::Path;

/// Open a vector file for streaming reading.
///
/// Format is auto-detected from the file extension. Returns a boxed
/// [`VecSource`] that yields records as raw bytes.
///
/// # Example
///
/// An `.i8vecs` file streams as records of `i8` bytes; the reader
/// copies each record out of its buffer, and the caller decodes.
///
/// ```
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("codes.i8vecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # for row in [[-1i8, 2, -3], [4, -5, 6]] {
/// #     f.write_all(&3i32.to_le_bytes()).unwrap();
/// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
/// # }
/// # drop(f);
/// let mut reader = veks_io::open(&path).unwrap();
/// assert_eq!((reader.dimension(), reader.element_size()), (3, 1));
/// let mut rows: Vec<Vec<i8>> = Vec::new();
/// while let Some(record) = reader.next_record() {
///     rows.push(record.iter().map(|&b| i8::from_le_bytes([b])).collect());
/// }
/// assert_eq!(rows, [[-1, 2, -3], [4, -5, 6]]);
/// ```
pub fn open(path: impl AsRef<Path>) -> Result<Box<dyn VecSource>, String> {
    let path = path.as_ref();
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("cannot detect format for {}", path.display()))?;
    open_format(path, format)
}

/// Open a vector file with an explicit format.
pub fn open_format(path: &Path, format: VecFormat) -> Result<Box<dyn VecSource>, String> {
    if format.is_xvec() {
        return xvec::reader::open(path, format);
    }
    if format.is_scalar() {
        return scalar::reader::open(path, format);
    }
    match format {
        #[cfg(feature = "npy")]
        VecFormat::Npy => Err("npy reading not yet implemented in veks-io".into()),
        #[cfg(feature = "parquet")]
        VecFormat::Parquet => Err("parquet reading not yet implemented in veks-io".into()),
        #[cfg(feature = "slab")]
        VecFormat::Slab => Err("slab reading not yet implemented in veks-io".into()),
        _ => Err(format!("{} format is not supported", format)),
    }
}

/// Create a vector file for sequential writing.
///
/// Format is auto-detected from the file extension. Returns a boxed
/// [`VecSink`] for writing records.
///
/// # Example
///
/// The extension picks the element width the writer expects per
/// record: `.fvecs` is `f32`, so a record of dimension 128 is
/// `128 * 4` bytes.
///
/// ```
/// # let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("output.fvecs");
/// let mut writer = veks_io::create(&path, 128).unwrap();
/// writer.write_record(0, &vec![0u8; 128 * 4]); // 128 f32s, all zero
/// writer.finish().unwrap();
///
/// let meta = veks_io::probe(&path).unwrap();
/// assert_eq!((meta.dimension, meta.record_count), (128, Some(1)));
/// assert_eq!(std::fs::metadata(&path).unwrap().len(), 4 + 128 * 4);
/// ```
pub fn create(path: impl AsRef<Path>, dimension: u32) -> Result<Box<dyn VecSink>, String> {
    let path = path.as_ref();
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("cannot detect format for {}", path.display()))?;
    create_format(path, format, dimension)
}

/// Create a vector file with an explicit format.
pub fn create_format(path: &Path, format: VecFormat, dimension: u32) -> Result<Box<dyn VecSink>, String> {
    if !format.is_writable() {
        return Err(format!("{} is not a writable format", format));
    }
    if format.is_xvec() {
        return xvec::writer::open(path, dimension);
    }
    if format.is_scalar() {
        return scalar::writer::open(path, format);
    }
    Err(format!("{} writing not yet implemented in veks-io", format))
}

/// Probe a vector file for metadata without reading all data.
///
/// Returns dimension, element size, and record count.
///
/// # Example
///
/// One header read and a size check; the element size comes from the
/// extension, here 8 bytes for the `f64` of a `.dvecs` file.
///
/// ```
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("precise.dvecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # for row in [[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]] {
/// #     f.write_all(&2i32.to_le_bytes()).unwrap();
/// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
/// # }
/// # drop(f);
/// let meta = veks_io::probe(&path).unwrap();
/// assert_eq!(meta.dimension, 2);
/// assert_eq!(meta.element_size, 8); // f64
/// assert_eq!(meta.record_count, Some(3));
/// ```
pub fn probe(path: impl AsRef<Path>) -> Result<SourceMeta, String> {
    let path = path.as_ref();
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("cannot detect format for {}", path.display()))?;
    probe_format(path, format)
}

/// Open a vector file for variable-length streaming reading.
///
/// Unlike [`open`], this reader handles files where records may have
/// different dimensions. Each record carries its own dimension.
///
/// # Example
///
/// A `.u16vecs` file whose records have dimensions 2 and 3; each
/// record comes back with its own dimension and `u16` bytes.
///
/// ```
/// # use std::io::Write;
/// # let dir = tempfile::tempdir().unwrap();
/// # let path = dir.path().join("ragged.u16vecs");
/// # let mut f = std::fs::File::create(&path).unwrap();
/// # for row in [vec![10u16, 20], vec![30, 40, 50]] {
/// #     f.write_all(&(row.len() as i32).to_le_bytes()).unwrap();
/// #     for v in row { f.write_all(&v.to_le_bytes()).unwrap(); }
/// # }
/// # drop(f);
/// let mut reader = veks_io::open_varlen(&path).unwrap();
/// assert_eq!(reader.element_size(), 2); // u16
/// let mut dims = Vec::new();
/// while let Some(record) = reader.next_record() {
///     assert_eq!(record.data.len(), record.dimension as usize * 2);
///     dims.push(record.dimension);
/// }
/// assert_eq!(dims, [2, 3]);
/// ```
pub fn open_varlen(path: impl AsRef<Path>) -> Result<Box<dyn VarlenSource>, String> {
    let path = path.as_ref();
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("cannot detect format for {}", path.display()))?;
    if !format.is_xvec() {
        return Err(format!("varlen reading only supported for xvec formats, got {}", format));
    }
    xvec::varlen::open_reader(path, format)
}

/// Create a vector file for variable-length writing.
///
/// Unlike [`create`], this writer accepts a per-record dimension,
/// allowing records with different dimensions in the same file.
///
/// # Example
///
/// Two `i64` records of different dimension into an `.i64vecs` file,
/// read back through [`open_varlen`].
///
/// ```
/// # let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("ragged.i64vecs");
/// let mut writer = veks_io::create_varlen(&path).unwrap();
/// let short: Vec<u8> = [7i64, -8].iter().flat_map(|v| v.to_le_bytes()).collect();
/// let long: Vec<u8> = [i64::MIN, 0, i64::MAX].iter().flat_map(|v| v.to_le_bytes()).collect();
/// writer.write_record(2, &short); // dim=2
/// writer.write_record(3, &long);  // dim=3
/// writer.finish().unwrap();
///
/// let mut reader = veks_io::open_varlen(&path).unwrap();
/// let first = reader.next_record().unwrap();
/// assert_eq!(first.dimension, 2);
/// assert_eq!(i64::from_le_bytes(first.data[8..16].try_into().unwrap()), -8);
/// assert_eq!(reader.next_record().unwrap().dimension, 3);
/// ```
pub fn create_varlen(path: impl AsRef<Path>) -> Result<Box<dyn VarlenSink>, String> {
    let path = path.as_ref();
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("cannot detect format for {}", path.display()))?;
    if !format.is_xvec() {
        return Err(format!("varlen writing only supported for xvec formats, got {}", format));
    }
    xvec::varlen::open_writer(path)
}

/// Probe with an explicit format.
pub fn probe_format(path: &Path, format: VecFormat) -> Result<SourceMeta, String> {
    if format.is_xvec() {
        return xvec::reader::probe(path, format);
    }
    if format.is_scalar() {
        return scalar::reader::probe(path, format);
    }
    Err(format!("{} probing not yet implemented in veks-io", format))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            f.write_all(&(v.len() as i32).to_le_bytes()).unwrap();
            for &val in v { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    #[test]
    fn test_open_and_read_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.fvec");
        write_fvec(&path, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 3);
        assert_eq!(reader.element_size(), 4);
        assert_eq!(reader.record_count(), Some(2));

        let r0 = reader.next_record().unwrap();
        assert_eq!(r0.len(), 12); // 3 × 4 bytes
        let v0 = f32::from_le_bytes(r0[0..4].try_into().unwrap());
        assert_eq!(v0, 1.0);

        let r1 = reader.next_record().unwrap();
        assert_eq!(r1.len(), 12);

        assert!(reader.next_record().is_none());
    }

    #[test]
    fn test_create_and_write_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("output.fvec");

        let mut writer = create(&path, 3).unwrap();
        let data: Vec<u8> = [1.0f32, 2.0, 3.0].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_record(0, &data);
        writer.finish().unwrap();

        // Read back
        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 3);
        assert_eq!(reader.record_count(), Some(1));
        let record = reader.next_record().unwrap();
        assert_eq!(record, data);
    }

    #[test]
    fn test_probe_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.fvec");
        write_fvec(&path, &[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);

        let meta = probe(&path).unwrap();
        assert_eq!(meta.dimension, 2);
        assert_eq!(meta.element_size, 4);
        assert_eq!(meta.record_count, Some(3));
    }

    #[test]
    fn test_roundtrip_ivec() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.ivec");

        let mut writer = create(&path, 2).unwrap();
        let data: Vec<u8> = [10i32, 20].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_record(0, &data);
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 2);
        let record = reader.next_record().unwrap();
        let v0 = i32::from_le_bytes(record[0..4].try_into().unwrap());
        let v1 = i32::from_le_bytes(record[4..8].try_into().unwrap());
        assert_eq!((v0, v1), (10, 20));
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(VecFormat::detect(Path::new("data.fvec")), Some(VecFormat::Fvec));
        assert_eq!(VecFormat::detect(Path::new("data.ivecs")), Some(VecFormat::Ivec));
        assert_eq!(VecFormat::detect(Path::new("data.mvec")), Some(VecFormat::Mvec));
        assert_eq!(VecFormat::detect(Path::new("data.unknown")), None);
    }

    #[test]
    fn test_unknown_format_error() {
        assert!(open(Path::new("data.txt")).is_err());
        assert!(create(Path::new("data.txt"), 10).is_err());
    }

    /// Create a project-local temp directory under `target/tmp`.
    fn make_tmp() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    // ── mvec (f16) roundtrip ────────────────────────────────────────

    #[test]
    fn roundtrip_mvec() {
        let tmp = make_tmp();
        let path = tmp.path().join("test.mvec");

        let dim = 4u32;
        let mut writer = create(&path, dim).unwrap();
        // Write two f16 vectors
        let v0: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let v1: Vec<u8> = [5.0f32, 6.0, 7.0, 8.0].iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        writer.write_record(0, &v0);
        writer.write_record(1, &v1);
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 4);
        assert_eq!(reader.element_size(), 2);
        assert_eq!(reader.record_count(), Some(2));

        let r0 = reader.next_record().unwrap();
        assert_eq!(r0, v0);
        let r1 = reader.next_record().unwrap();
        assert_eq!(r1, v1);
        assert!(reader.next_record().is_none());
    }

    // ── dvec (f64) roundtrip ────────────────────────────────────────

    #[test]
    fn roundtrip_dvec() {
        let tmp = make_tmp();
        let path = tmp.path().join("test.dvec");

        let mut writer = create(&path, 2).unwrap();
        let v0: Vec<u8> = [1.0f64, 2.0].iter()
            .flat_map(|v| v.to_le_bytes()).collect();
        writer.write_record(0, &v0);
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 2);
        assert_eq!(reader.element_size(), 8);
        let r0 = reader.next_record().unwrap();
        let d0 = f64::from_le_bytes(r0[0..8].try_into().unwrap());
        let d1 = f64::from_le_bytes(r0[8..16].try_into().unwrap());
        assert_eq!((d0, d1), (1.0, 2.0));
    }

    // ── bvec (u8) roundtrip ─────────────────────────────────────────

    #[test]
    fn roundtrip_bvec() {
        let tmp = make_tmp();
        let path = tmp.path().join("test.bvec");

        let mut writer = create(&path, 5).unwrap();
        // bvec: 1 byte per element (u8 vectors)
        let v0: Vec<u8> = vec![10, 20, 30, 40, 50];
        writer.write_record(0, &v0);
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 5);
        assert_eq!(reader.element_size(), 1);
        let r0 = reader.next_record().unwrap();
        assert_eq!(r0, v0);
    }

    // ── svec (i16) roundtrip ────────────────────────────────────────

    #[test]
    fn roundtrip_svec() {
        let tmp = make_tmp();
        let path = tmp.path().join("test.svec");

        let mut writer = create(&path, 3).unwrap();
        let v0: Vec<u8> = [100i16, -200, 300].iter()
            .flat_map(|v| v.to_le_bytes()).collect();
        writer.write_record(0, &v0);
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 3);
        assert_eq!(reader.element_size(), 2);
        let r0 = reader.next_record().unwrap();
        let s0 = i16::from_le_bytes(r0[0..2].try_into().unwrap());
        let s1 = i16::from_le_bytes(r0[2..4].try_into().unwrap());
        let s2 = i16::from_le_bytes(r0[4..6].try_into().unwrap());
        assert_eq!((s0, s1, s2), (100, -200, 300));
    }

    // ── write then mmap read ────────────────────────────────────────

    #[test]
    fn write_then_mmap_read_fvec() {
        let tmp = make_tmp();
        let path = tmp.path().join("mmap_test.fvec");

        let mut writer = create(&path, 3).unwrap();
        for i in 0..100 {
            let data: Vec<u8> = [i as f32, (i * 2) as f32, (i * 3) as f32]
                .iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_record(i, &data);
        }
        writer.finish().unwrap();

        let reader = MmapReader::<f32>::open(&path).unwrap();
        assert_eq!(reader.count(), 100);
        assert_eq!(reader.dim(), 3);

        for i in 0..100 {
            let slice = reader.get_slice(i).unwrap();
            assert_eq!(slice[0], i as f32);
            assert_eq!(slice[1], (i * 2) as f32);
            assert_eq!(slice[2], (i * 3) as f32);
        }
    }

    // ── dim=1 edge case ─────────────────────────────────────────────

    #[test]
    fn roundtrip_dim_one() {
        let tmp = make_tmp();
        let path = tmp.path().join("dim1.fvec");

        let mut writer = create(&path, 1).unwrap();
        for i in 0..5 {
            let data = (i as f32).to_le_bytes().to_vec();
            writer.write_record(i, &data);
        }
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 1);
        assert_eq!(reader.record_count(), Some(5));
        for i in 0..5 {
            let r = reader.next_record().unwrap();
            let v = f32::from_le_bytes(r[0..4].try_into().unwrap());
            assert_eq!(v, i as f32);
        }
        assert!(reader.next_record().is_none());
    }

    // ── large dimension ─────────────────────────────────────────────

    #[test]
    fn roundtrip_large_dim() {
        let tmp = make_tmp();
        let path = tmp.path().join("large_dim.fvec");
        let dim = 2048u32;

        let mut writer = create(&path, dim).unwrap();
        let data: Vec<u8> = (0..dim).flat_map(|i| (i as f32).to_le_bytes()).collect();
        writer.write_record(0, &data);
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), dim);
        let r = reader.next_record().unwrap();
        assert_eq!(r.len(), dim as usize * 4);
        let v0 = f32::from_le_bytes(r[0..4].try_into().unwrap());
        let vlast = f32::from_le_bytes(r[(dim as usize - 1) * 4..dim as usize * 4].try_into().unwrap());
        assert_eq!(v0, 0.0);
        assert_eq!(vlast, (dim - 1) as f32);
    }

    // ── many records streaming ───────────────────────────────────────

    #[test]
    fn stream_many_records() {
        let tmp = make_tmp();
        let path = tmp.path().join("many.fvec");

        let count = 10_000;
        let dim = 4u32;
        let mut writer = create(&path, dim).unwrap();
        for i in 0..count {
            let data: Vec<u8> = (0..dim).flat_map(|d| ((i * dim + d) as f32).to_le_bytes()).collect();
            writer.write_record(i as i64, &data);
        }
        writer.finish().unwrap();

        let meta = probe(&path).unwrap();
        assert_eq!(meta.dimension, dim);
        assert_eq!(meta.record_count, Some(count as u64));

        let mut reader = open(&path).unwrap();
        let mut read_count = 0u32;
        while reader.next_record().is_some() {
            read_count += 1;
        }
        assert_eq!(read_count, count);
    }

    // ── truncated file ──────────────────────────────────────────────

    /// A file that is not a whole number of records is refused at open,
    /// by the streaming reader as by the mmap reader: a tail silently
    /// dropped is a record a consumer never sees.
    #[test]
    fn a_truncated_file_is_refused() {
        let tmp = make_tmp();
        let path = tmp.path().join("truncated.fvec");

        // Write one complete record then truncate mid-record
        {
            let mut f = std::fs::File::create(&path).unwrap();
            // Record 0: complete (dim=2, two f32s)
            f.write_all(&2i32.to_le_bytes()).unwrap();
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
            f.write_all(&2.0f32.to_le_bytes()).unwrap();
            // Record 1: truncated (dim header only, no data)
            f.write_all(&2i32.to_le_bytes()).unwrap();
            // Missing: the two f32 values
        }

        let err = open(&path).err().expect("a short file does not open");
        assert!(err.contains("not a whole number"), "{err}");
        assert!(probe(&path).is_err());
    }

    // ── corrupted dimension header ──────────────────────────────────

    #[test]
    fn corrupted_dim_mid_file() {
        let tmp = make_tmp();
        let path = tmp.path().join("bad_dim.fvec");

        {
            let mut f = std::fs::File::create(&path).unwrap();
            // Record 0: dim=3
            f.write_all(&3i32.to_le_bytes()).unwrap();
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
            f.write_all(&2.0f32.to_le_bytes()).unwrap();
            f.write_all(&3.0f32.to_le_bytes()).unwrap();
            // Record 1: header says dim=5 (mismatch!) over a record-sized
            // payload, so the file is still a whole number of records and
            // only the header is wrong.
            f.write_all(&5i32.to_le_bytes()).unwrap();
            f.write_all(&4.0f32.to_le_bytes()).unwrap();
            f.write_all(&5.0f32.to_le_bytes()).unwrap();
            f.write_all(&6.0f32.to_le_bytes()).unwrap();
        }

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 3);
        let r0 = reader.next_record();
        assert!(r0.is_some(), "first record should be readable");
        // Second record has wrong dim — reader should stop
        let r1 = reader.next_record();
        assert!(r1.is_none(), "dimension mismatch should stop iteration");
    }

    // ── mmap on truncated file ──────────────────────────────────────

    #[test]
    fn mmap_truncated_file() {
        let tmp = make_tmp();
        let path = tmp.path().join("trunc.fvec");

        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&3i32.to_le_bytes()).unwrap();
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
            // Missing 2 more f32s — file size not a multiple of record size
        }

        let result = MmapReader::<f32>::open(&path);
        assert!(result.is_err(), "truncated file should fail mmap open");
    }

    // ── cross-format compatibility with veks-pipeline writes ────────

    #[test]
    fn veks_pipeline_compatible_fvec() {
        // Write an fvec the way veks-pipeline does (manual dim+data writes)
        let tmp = make_tmp();
        let path = tmp.path().join("pipeline_compat.fvec");

        {
            let mut f = std::fs::File::create(&path).unwrap();
            for i in 0..3 {
                f.write_all(&4i32.to_le_bytes()).unwrap(); // dim=4
                for d in 0..4 {
                    let v = (i * 10 + d) as f32;
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }

        // Verify veks-io can read it
        let meta = probe(&path).unwrap();
        assert_eq!(meta.dimension, 4);
        assert_eq!(meta.record_count, Some(3));

        let mut reader = open(&path).unwrap();
        let r0 = reader.next_record().unwrap();
        let v00 = f32::from_le_bytes(r0[0..4].try_into().unwrap());
        assert_eq!(v00, 0.0);

        // Also verify mmap
        let mmap = MmapReader::<f32>::open(&path).unwrap();
        assert_eq!(mmap.count(), 3);
        assert_eq!(mmap.get_slice(0), Some(&[0.0f32, 1.0, 2.0, 3.0][..]));
        assert_eq!(mmap.get_slice(2), Some(&[20.0f32, 21.0, 22.0, 23.0][..]));
    }

    // ── probe on all xvec formats ───────────────────────────────────

    #[test]
    fn probe_all_xvec_formats() {
        let tmp = make_tmp();

        for (ext, elem_size) in [("fvec", 4), ("ivec", 4), ("dvec", 8), ("mvec", 2), ("svec", 2), ("bvec", 1), ("bf16vecs", 2)] {
            let path = tmp.path().join(format!("test.{}", ext));
            let dim = 5u32;
            let mut writer = create(&path, dim).unwrap();
            let data = vec![0u8; dim as usize * elem_size];
            writer.write_record(0, &data);
            writer.write_record(1, &data);
            writer.finish().unwrap();

            let meta = probe(&path).unwrap();
            assert_eq!(meta.dimension, dim, "dim mismatch for {}", ext);
            assert_eq!(meta.element_size, elem_size, "elem_size mismatch for {}", ext);
            assert_eq!(meta.record_count, Some(2), "count mismatch for {}", ext);
        }
    }

    // ── nonexistent file ────────────────────────────────────────────

    #[test]
    fn open_nonexistent_file() {
        assert!(open(Path::new("/nonexistent/path/data.fvec")).is_err());
        assert!(probe(Path::new("/nonexistent/path/data.fvec")).is_err());
    }

    // ── write to read-only path ─────────────────────────────────────

    #[test]
    fn create_unwritable_path() {
        assert!(create(Path::new("/proc/data.fvec"), 10).is_err());
    }

    // ── scalar format roundtrips ────────────────────────────────────

    #[test]
    fn scalar_u8_roundtrip() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u8");
        let mut w = create(&path, 1).unwrap();
        for v in [0u8, 127, 255] {
            w.write_record(0, &[v]);
        }
        w.finish().unwrap();

        let meta = probe(&path).unwrap();
        assert_eq!(meta.dimension, 1);
        assert_eq!(meta.element_size, 1);
        assert_eq!(meta.record_count, Some(3));

        let mut r = open(&path).unwrap();
        assert_eq!(r.next_record().unwrap(), vec![0u8]);
        assert_eq!(r.next_record().unwrap(), vec![127u8]);
        assert_eq!(r.next_record().unwrap(), vec![255u8]);
        assert!(r.next_record().is_none());
    }

    #[test]
    fn scalar_i32_roundtrip() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i32");
        let mut w = create(&path, 1).unwrap();
        for v in [-1i32, 0, 42, i32::MAX, i32::MIN] {
            w.write_record(0, &v.to_le_bytes());
        }
        w.finish().unwrap();

        let meta = probe(&path).unwrap();
        assert_eq!(meta.dimension, 1);
        assert_eq!(meta.element_size, 4);
        assert_eq!(meta.record_count, Some(5));

        let mut r = open(&path).unwrap();
        let v0 = r.next_record().unwrap();
        assert_eq!(i32::from_le_bytes(v0[..4].try_into().unwrap()), -1);
        let v3 = { r.next_record(); r.next_record(); r.next_record().unwrap() };
        assert_eq!(i32::from_le_bytes(v3[..4].try_into().unwrap()), i32::MAX);
    }

    #[test]
    fn scalar_u64_roundtrip() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u64");
        let mut w = create(&path, 1).unwrap();
        for v in [0u64, 1, u64::MAX] {
            w.write_record(0, &v.to_le_bytes());
        }
        w.finish().unwrap();

        let meta = probe(&path).unwrap();
        assert_eq!(meta.record_count, Some(3));
        assert_eq!(meta.element_size, 8);
    }

    #[test]
    fn scalar_all_types_probe() {
        let tmp = make_tmp();
        for (ext, elem_size) in [
            ("u8", 1), ("i8", 1), ("u16", 2), ("i16", 2),
            ("u32", 4), ("i32", 4), ("u64", 8), ("i64", 8),
            ("f16", 2), ("bf16", 2), ("f32", 4), ("f64", 8),
        ] {
            let path = tmp.path().join(format!("data.{}", ext));
            let data = vec![0u8; elem_size * 10]; // 10 records
            std::fs::write(&path, &data).unwrap();

            let meta = probe(&path).unwrap();
            assert_eq!(meta.dimension, 1, "dim for {}", ext);
            assert_eq!(meta.element_size, elem_size, "elem_size for {}", ext);
            assert_eq!(meta.record_count, Some(10), "count for {}", ext);
        }
    }

    #[test]
    fn scalar_invalid_file_size_rejected() {
        let tmp = make_tmp();
        let path = tmp.path().join("bad.i32");
        std::fs::write(&path, [0u8; 5]).unwrap(); // 5 bytes, not divisible by 4
        assert!(open(&path).is_err());
        assert!(probe(&path).is_err());
    }

    // ── new vector format roundtrips ────────────────────────────────

    #[test]
    fn roundtrip_i8vec() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i8vec");
        let mut w = create(&path, 3).unwrap();
        let v: Vec<u8> = [-1i8, 0, 127].iter().map(|v| *v as u8).collect();
        w.write_record(0, &v);
        w.finish().unwrap();

        let mut r = open(&path).unwrap();
        assert_eq!(r.dimension(), 3);
        assert_eq!(r.element_size(), 1);
        let rec = r.next_record().unwrap();
        assert_eq!(rec[0] as i8, -1);
        assert_eq!(rec[1] as i8, 0);
        assert_eq!(rec[2] as i8, 127);
    }

    #[test]
    fn roundtrip_u16vec() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u16vec");
        let mut w = create(&path, 2).unwrap();
        let v: Vec<u8> = [1000u16, 65535].iter()
            .flat_map(|v| v.to_le_bytes()).collect();
        w.write_record(0, &v);
        w.finish().unwrap();

        let r = open(&path).unwrap();
        assert_eq!(r.dimension(), 2);
        assert_eq!(r.element_size(), 2);
    }

    #[test]
    fn roundtrip_u32vec() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u32vec");
        let mut w = create(&path, 2).unwrap();
        let v: Vec<u8> = [0u32, u32::MAX].iter()
            .flat_map(|v| v.to_le_bytes()).collect();
        w.write_record(0, &v);
        w.finish().unwrap();

        let r = open(&path).unwrap();
        assert_eq!(r.dimension(), 2);
        assert_eq!(r.element_size(), 4);
    }

    #[test]
    fn roundtrip_i64vec() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.i64vec");
        let mut w = create(&path, 1).unwrap();
        let v = i64::MIN.to_le_bytes().to_vec();
        w.write_record(0, &v);
        w.finish().unwrap();

        let mut r = open(&path).unwrap();
        assert_eq!(r.dimension(), 1);
        assert_eq!(r.element_size(), 8);
        let rec = r.next_record().unwrap();
        assert_eq!(i64::from_le_bytes(rec[..8].try_into().unwrap()), i64::MIN);
    }

    #[test]
    fn roundtrip_u64vec() {
        let tmp = make_tmp();
        let path = tmp.path().join("data.u64vec");
        let mut w = create(&path, 1).unwrap();
        let v = u64::MAX.to_le_bytes().to_vec();
        w.write_record(0, &v);
        w.finish().unwrap();

        let mut r = open(&path).unwrap();
        assert_eq!(r.element_size(), 8);
        let rec = r.next_record().unwrap();
        assert_eq!(u64::from_le_bytes(rec[..8].try_into().unwrap()), u64::MAX);
    }
}
