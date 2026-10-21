// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The techniques the readers kept because they measured faster
//! (`docs/design/srd-loose-files-overlays-integrity.md`, IO-4), each
//! held by a count that does not depend on timing:
//!
//! - a zero-copy read asks its source for the mapping once, then
//!   indexes it (1.3× on random `get_slice`);
//! - `get` copies a record in bulk rather than decoding element by
//!   element (2.4× on random `get`).

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use veks_io::{ByteSource, ElementType, IoError, MappedFile, VectorReader, XvecReader};

fn write_fvecs(path: &Path, rows: usize, dim: usize) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in 0..rows {
        f.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for k in 0..dim {
            f.write_all(&((r * 1000 + k) as f32).to_le_bytes()).unwrap();
        }
    }
}

/// A local file that counts how often a reader asks where its mapping
/// is.
#[derive(Clone, Debug)]
struct CountingSource {
    inner: MappedFile,
    asks: Arc<AtomicUsize>,
}

impl ByteSource for CountingSource {
    type Progress = ();

    fn open(source: &str) -> Result<Self, IoError> {
        Self::open_path(Path::new(source))
    }
    fn open_path(path: &Path) -> Result<Self, IoError> {
        Ok(Self { inner: MappedFile::open_path(path)?, asks: Arc::new(AtomicUsize::new(0)) })
    }
    fn total_size(&self) -> u64 {
        self.inner.total_size()
    }
    fn read_bytes(&self, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
        self.inner.read_bytes(offset, len)
    }
    fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        self.asks.fetch_add(1, Ordering::Relaxed);
        self.inner.mmap_slice(offset, len)
    }
    fn mmap_base(&self) -> Option<*const u8> {
        self.asks.fetch_add(1, Ordering::Relaxed);
        self.inner.mmap_base()
    }
}

/// **A read asks the source for its mapping once.** A mapping never
/// moves for the life of its source, so after the first answer the
/// reader keeps the address and every later read is a pointer add —
/// the checked and unchecked zero-copy forms, and the copying `get`.
#[test]
fn a_read_asks_the_source_for_its_mapping_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("base.fvecs");
    write_fvecs(&path, 1000, 8);
    let reader = XvecReader::<f32, CountingSource>::open_path(&path).unwrap();
    let asks = reader.source().asks.clone();
    asks.store(0, Ordering::Relaxed);

    let mut sum = 0.0f64;
    for i in 0..1000 {
        sum += VectorReader::get_slice(&reader, i).unwrap()[0] as f64;
        sum += reader.get_slice(i)[1] as f64;
        sum += VectorReader::get(&reader, i).unwrap()[2] as f64;
    }
    assert_eq!(sum, (0..1000).map(|r| (r * 3000 + 3) as f64).sum::<f64>());
    assert!(
        asks.load(Ordering::Relaxed) <= 1,
        "3000 reads asked the source {} times",
        asks.load(Ordering::Relaxed)
    );
}

/// How many times [`CountedF32::from_le_bytes`] has run.
static DECODES: AtomicUsize = AtomicUsize::new(0);

/// An `f32` that counts its decodes: the element type a reader reads
/// an `.fvecs` file as, here to see how `get` builds its record.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CountedF32(f32);

impl veks_io::Element for CountedF32 {
    const TYPE: ElementType = ElementType::F32;
    const SIZE: usize = 4;

    fn from_le_bytes(bytes: &[u8]) -> Self {
        DECODES.fetch_add(1, Ordering::Relaxed);
        CountedF32(f32::from_le_bytes(bytes[..4].try_into().unwrap()))
    }
    fn write_le_bytes(self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.0.to_le_bytes());
    }
    fn to_value(self) -> veks_datatypes::Value {
        self.0.to_value()
    }
    fn from_value(value: veks_datatypes::Value) -> Option<Self> {
        f32::from_value(value).map(CountedF32)
    }
    fn as_f64(self) -> f64 {
        self.0 as f64
    }
}


/// **`get` copies a record, it does not decode it element by
/// element.** On a little-endian host an element's bytes are its
/// value, so the record is one allocation and one copy; a per-element
/// decode depends on the compiler inlining it to be fast, and the bench
/// measured it not doing so.
#[test]
#[cfg(target_endian = "little")]
fn get_copies_a_record_without_decoding_each_element() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("base.fvecs");
    write_fvecs(&path, 100, 64);
    let reader = XvecReader::<CountedF32>::open_path(&path).unwrap();
    let before = DECODES.load(Ordering::Relaxed);
    for i in 0..100 {
        let row = VectorReader::get(&reader, i).unwrap();
        assert_eq!(row.len(), 64);
        assert_eq!(row[63], CountedF32((i * 1000 + 63) as f32));
    }
    let decoded = DECODES.load(Ordering::Relaxed) - before;
    assert_eq!(decoded, 0, "100 records of 64 elements decoded {decoded} elements one by one");
}
