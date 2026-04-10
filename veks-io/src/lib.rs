// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! # veks-io — Standalone Vector Data I/O
//!
//! Read and write vector data in fvec, ivec, mvec, dvec, bvec, svec,
//! and (optionally) npy, parquet, and slab formats.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use veks_io::{open, create, probe};
//!
//! // Probe a file for metadata
//! let meta = probe("dataset.fvec").unwrap();
//! println!("dim={}, records={:?}", meta.dimension, meta.record_count);
//!
//! // Stream-read all records
//! let mut reader = open("dataset.fvec").unwrap();
//! while let Some(record) = reader.next_record() {
//!     // record is Vec<u8> — raw element bytes
//! }
//!
//! // Write records
//! let mut writer = create("output.fvec", 128).unwrap();
//! writer.write_record(0, &vec![0u8; 128 * 4]);
//! writer.finish().unwrap();
//! ```
//!
//! ## Zero-Copy Mmap Access
//!
//! For random-access on xvec files, use the typed mmap readers:
//!
//! ```rust,no_run
//! use veks_io::xvec::mmap::MmapReader;
//!
//! let reader = MmapReader::<f32>::open_fvec("base.fvec".as_ref()).unwrap();
//! let vec = reader.get_slice(42); // &[f32], zero-copy
//! ```
//!
//! ## Feature Flags
//!
//! - `npy` — NumPy `.npy` directory reading (adds ndarray dependency)
//! - `parquet` — Apache Parquet directory reading (adds arrow/parquet dependencies)
//! - `slab` — Slab format read/write (adds slabtastic dependency)

pub mod format;
pub mod traits;
pub mod xvec;

pub use format::VecFormat;
pub use traits::{SourceMeta, VecSink, VecSource};
pub use xvec::mmap::MmapReader;

use std::path::Path;

/// Open a vector file for streaming reading.
///
/// Format is auto-detected from the file extension. Returns a boxed
/// [`VecSource`] that yields records as raw bytes.
///
/// # Example
///
/// ```rust,no_run
/// let mut reader = veks_io::open("data.fvec").unwrap();
/// while let Some(record) = reader.next_record() {
///     // process record bytes
/// }
/// ```
pub fn open(path: impl AsRef<Path>) -> Result<Box<dyn VecSource>, String> {
    let path = path.as_ref();
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("cannot detect format for {}", path.display()))?;
    open_format(path, format)
}

/// Open a vector file with an explicit format.
pub fn open_format(path: &Path, format: VecFormat) -> Result<Box<dyn VecSource>, String> {
    match format {
        VecFormat::Fvec | VecFormat::Ivec | VecFormat::Bvec
        | VecFormat::Dvec | VecFormat::Mvec | VecFormat::Svec => {
            xvec::reader::open(path, format)
        }
        #[cfg(feature = "npy")]
        VecFormat::Npy => {
            Err("npy reading not yet implemented in veks-io".into())
        }
        #[cfg(feature = "parquet")]
        VecFormat::Parquet => {
            Err("parquet reading not yet implemented in veks-io".into())
        }
        #[cfg(feature = "slab")]
        VecFormat::Slab => {
            Err("slab reading not yet implemented in veks-io".into())
        }
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
/// ```rust,no_run
/// let mut writer = veks_io::create("output.fvec", 128).unwrap();
/// writer.write_record(0, &vec![0u8; 128 * 4]); // 128 f32s
/// writer.finish().unwrap();
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
    match format {
        VecFormat::Fvec | VecFormat::Ivec | VecFormat::Bvec
        | VecFormat::Dvec | VecFormat::Mvec | VecFormat::Svec => {
            xvec::writer::open(path, dimension)
        }
        _ => Err(format!("{} writing not yet implemented in veks-io", format)),
    }
}

/// Probe a vector file for metadata without reading all data.
///
/// Returns dimension, element size, and record count.
pub fn probe(path: impl AsRef<Path>) -> Result<SourceMeta, String> {
    let path = path.as_ref();
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("cannot detect format for {}", path.display()))?;
    probe_format(path, format)
}

/// Probe with an explicit format.
pub fn probe_format(path: &Path, format: VecFormat) -> Result<SourceMeta, String> {
    match format {
        VecFormat::Fvec | VecFormat::Ivec | VecFormat::Bvec
        | VecFormat::Dvec | VecFormat::Mvec | VecFormat::Svec => {
            xvec::reader::probe(path, format)
        }
        _ => Err(format!("{} probing not yet implemented in veks-io", format)),
    }
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

        let mut writer = create(&path, 3).unwrap();
        // bvec element_size is 4 (stores u8 in 4-byte groups per the format)
        // but the raw data written is 3*4=12 bytes
        let v0: Vec<u8> = [10u32, 20, 30].iter()
            .flat_map(|v| v.to_le_bytes()).collect();
        writer.write_record(0, &v0);
        writer.finish().unwrap();

        let mut reader = open(&path).unwrap();
        assert_eq!(reader.dimension(), 3);
        assert_eq!(reader.element_size(), 4);
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

        let reader = MmapReader::<f32>::open_fvec(&path).unwrap();
        assert_eq!(reader.count(), 100);
        assert_eq!(reader.dim(), 3);

        for i in 0..100 {
            let slice = reader.get_slice(i);
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

    #[test]
    fn truncated_file_stops_gracefully() {
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

        let mut reader = open(&path).unwrap();
        // First record should succeed
        let r0 = reader.next_record();
        assert!(r0.is_some(), "first record should be readable");
        // Second record is truncated — should return None, not panic
        let r1 = reader.next_record();
        assert!(r1.is_none(), "truncated record should return None");
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
            // Record 1: dim=5 (mismatch!)
            f.write_all(&5i32.to_le_bytes()).unwrap();
            f.write_all(&4.0f32.to_le_bytes()).unwrap();
            f.write_all(&5.0f32.to_le_bytes()).unwrap();
            f.write_all(&6.0f32.to_le_bytes()).unwrap();
            f.write_all(&7.0f32.to_le_bytes()).unwrap();
            f.write_all(&8.0f32.to_le_bytes()).unwrap();
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

        let result = MmapReader::<f32>::open_fvec(&path);
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
        let mmap = MmapReader::<f32>::open_fvec(&path).unwrap();
        assert_eq!(mmap.count(), 3);
        assert_eq!(mmap.get_slice(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(mmap.get_slice(2), &[20.0, 21.0, 22.0, 23.0]);
    }

    // ── probe on all xvec formats ───────────────────────────────────

    #[test]
    fn probe_all_xvec_formats() {
        let tmp = make_tmp();

        for (ext, elem_size) in [("fvec", 4), ("ivec", 4), ("dvec", 8), ("mvec", 2), ("svec", 2), ("bvec", 4)] {
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
}
