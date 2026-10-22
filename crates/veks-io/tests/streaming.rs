// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The streaming xvec reader and writer, merged from veks-core's into
//! veks-io's (`docs/design/srd-crate-boundaries.md` FL-2, FL-3): a
//! directory or series read as one stream, a symlink at the target
//! refused, and writeback over a file
//! larger than the writeback interval.

use std::path::Path;

use veks_io::{VecFormat, VecSource, xvec};

/// `rows` records of `dim` f32 values, record `r` holding `base + r`.
fn write_fvecs(path: &Path, dim: u32, rows: u32, base: f32) {
    let mut w = xvec::writer::open(path, dim).unwrap();
    for r in 0..rows {
        let bytes: Vec<u8> = (0..dim).flat_map(|_| (base + r as f32).to_le_bytes()).collect();
        w.write_record(r as i64, &bytes);
    }
    w.finish().unwrap();
}

/// Every record of `source`, as its first f32.
fn firsts(mut source: Box<dyn VecSource>) -> Vec<f32> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    while source.read_record_into(&mut buf) {
        out.push(f32::from_le_bytes(buf[..4].try_into().unwrap()));
    }
    out
}

/// **A directory of xvec files reads as one stream**, in name order,
/// its count the sum of the files' (veks-core's reader did this; the
/// merged one must).
#[test]
fn a_directory_of_shards_reads_as_one_stream() {
    let dir = tempfile::tempdir().unwrap();
    write_fvecs(&dir.path().join("b__0000.fvecs"), 3, 4, 0.0);
    write_fvecs(&dir.path().join("b__0001.fvecs"), 3, 2, 100.0);
    std::fs::write(dir.path().join("notes.txt"), "not a shard").unwrap();
    let meta = xvec::reader::probe(dir.path(), VecFormat::Fvec).unwrap();
    assert_eq!((meta.dimension, meta.record_count), (3, Some(6)));
    let source = xvec::reader::open(dir.path(), VecFormat::Fvec).unwrap();
    assert_eq!(source.record_count(), Some(6));
    assert_eq!(firsts(source), vec![0.0, 1.0, 2.0, 3.0, 100.0, 101.0]);
}

/// **A series opens by its unsharded name**, the name a facet declares
/// when nothing is written at it.
#[test]
fn a_series_opens_by_its_unsharded_name() {
    let dir = tempfile::tempdir().unwrap();
    write_fvecs(&dir.path().join("base__0000.fvecs"), 2, 3, 0.0);
    write_fvecs(&dir.path().join("base__0001.fvecs"), 2, 1, 10.0);
    let source = xvec::reader::open(&dir.path().join("base.fvecs"), VecFormat::Fvec).unwrap();
    assert_eq!(firsts(source), vec![0.0, 1.0, 2.0, 10.0]);
}

/// **Files that disagree on dimension are not one stream.**
#[test]
fn shards_of_another_dimension_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    write_fvecs(&dir.path().join("a.fvecs"), 3, 2, 0.0);
    write_fvecs(&dir.path().join("b.fvecs"), 4, 2, 0.0);
    let err = xvec::reader::open(dir.path(), VecFormat::Fvec).err().expect("refused");
    assert!(err.contains("dimension 4 where the files before it hold 3"), "{err}");
}

/// **A symlink at the target is an error**, from every writer: the
/// file it links to — often source data — is left as it was, and so is
/// the link.
#[test]
fn a_symlink_at_the_target_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.fvecs");
    write_fvecs(&source, 2, 5, 7.0);
    let before = std::fs::read(&source).unwrap();
    for name in ["out.fvecs", "out.ivvecs", "out.u8"] {
        let target = dir.path().join(name);
        std::os::unix::fs::symlink(&source, &target).unwrap();
        let errors = [
            xvec::writer::open(&target, 2).err(),
            veks_io::scalar::writer::open(&target, VecFormat::ScalarU8).err(),
            xvec::varlen::open_writer(&target).err(),
        ];
        for err in errors {
            let err = err.expect("a write through a symlink is refused");
            assert!(err.contains("is a symlink") && err.contains("source.fvecs"), "{err}");
        }
        assert!(target.is_symlink(), "the link is left in place");
    }
    assert!(veks_io::XvecWriter::<f32>::create(&dir.path().join("out.fvecs"), 2).is_err());
    assert_eq!(std::fs::read(&source).unwrap(), before, "the linked source is untouched");
}

/// **A write past the writeback interval is whole**: records written
/// across several writeback cycles read back exactly.
#[test]
fn a_file_written_across_writeback_cycles_reads_back_whole() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.fvecs");
    // 70 MiB of dim-256 records: past one 64 MiB cycle.
    let rows = (70u32 << 20) / (4 + 256 * 4);
    write_fvecs(&path, 256, rows, 0.0);
    let got = firsts(xvec::reader::open(&path, VecFormat::Fvec).unwrap());
    assert_eq!(got.len(), rows as usize);
    assert!(got.iter().enumerate().all(|(i, v)| *v == i as f32));
}
