// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The typed writers over every element type, read back by the readers
//! (`docs/design/srd-loose-files-overlays-integrity.md`, LS-3), and the
//! byte-level writers' errors, which they used to discard.

use veks_io::{
    Container, Element, ElementType, IndexedVvecReader, IoError, ScalarReader, ScalarWriter, VectorReader,
    VvecReader, VvecWriter, XvecReader, XvecWriter,
};

/// A value of every type: small, so it fits them all.
fn sample<T: Element>(i: usize) -> T {
    T::from_value(veks_datatypes::Value::Int((i % 100) as i128)).expect("fits every type")
}

fn bits<T: Element>(values: &[T]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * T::SIZE];
    for (v, slot) in values.iter().zip(out.chunks_exact_mut(T::SIZE)) {
        v.write_le_bytes(slot);
    }
    out
}

/// **Every element type round-trips through the three writers**, read
/// back bit for bit by the reader of its layout; the vvec index written
/// on finish is the one the reader uses.
#[test]
fn every_type_round_trips_through_the_typed_writers() {
    for &ty in ElementType::ALL {
        veks_datatypes::dispatch_element!(ty, T => {
            let dir = tempfile::tempdir().unwrap();
            let rows: Vec<Vec<T>> = (0..5).map(|r| (0..3).map(|k| sample::<T>(r * 3 + k)).collect()).collect();

            let vectors = dir.path().join(format!("v.{}", ty.extension(Container::Vec)));
            let mut w = XvecWriter::<T>::create(&vectors, 3).unwrap();
            w.write_all(&rows).unwrap();
            assert_eq!(w.finish().unwrap(), 5, "{ty}");
            let r = XvecReader::<T>::open_path(&vectors).unwrap();
            assert_eq!((r.count(), r.dim()), (5, 3), "{ty}");
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(bits(&r.get(i).unwrap()), bits(row), "{ty} row {i}");
            }

            let column = dir.path().join(format!("c.{}", ty.extension(Container::Scalar)));
            let flat: Vec<T> = rows.concat();
            let mut w = ScalarWriter::<T>::create(&column).unwrap();
            w.write(&flat[..7]).unwrap();
            w.write(&flat[7..]).unwrap();
            assert_eq!(w.finish().unwrap(), 15, "{ty}");
            assert_eq!(bits(ScalarReader::<T>::open_path(&column).unwrap().as_slice().unwrap()), bits(&flat), "{ty}");

            let ragged = dir.path().join(format!("r.{}", ty.extension(Container::Vvec)));
            let mut w = VvecWriter::<T>::create(&ragged).unwrap();
            for (i, row) in rows.iter().enumerate() {
                w.write(&row[..i % 4]).unwrap();
            }
            assert_eq!(w.finish().unwrap(), 5, "{ty}");
            let r = IndexedVvecReader::<T>::open_path(&ragged).unwrap();
            assert_eq!(r.count(), 5, "{ty}");
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(bits(&VvecReader::get(&r, i).unwrap()), bits(&row[..i % 4]), "{ty} ragged row {i}");
            }
        });
    }
}

/// **A writer refuses what its file cannot hold**: another element
/// type, another layout, a row of the wrong length, a dimension that is
/// not positive.
#[test]
fn a_writer_refuses_what_its_file_cannot_hold() {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    assert!(matches!(
        XvecWriter::<f32>::create(&p("codes.bvecs"), 4),
        Err(IoError::ElementMismatch { file: ElementType::U8, requested: ElementType::F32, .. })
    ));
    let err = XvecWriter::<u8>::create(&p("codes.u8"), 4).err().unwrap();
    assert!(err.to_string().contains("names a column file of u8; write it with XvecWriter"), "{err}");
    let err = ScalarWriter::<f32>::create(&p("base.fvecs")).err().unwrap();
    assert!(err.to_string().contains("names a uniform vector file of f32; write it with ScalarWriter"), "{err}");
    let err = VvecWriter::<f32>::create(&p("base.fvecs")).err().unwrap();
    assert!(err.to_string().contains("write it with VvecWriter"), "{err}");
    assert!(matches!(XvecWriter::<f32>::create(&p("zero.fvecs"), 0), Err(IoError::InvalidFormat(_))));

    let mut w = XvecWriter::<f32>::create(&p("base.fvecs"), 2).unwrap();
    assert!(matches!(w.write(&[1.0, 2.0, 3.0]), Err(IoError::InvalidFormat(_))));
    assert!(matches!(w.write_packed(&[1.0, 2.0, 3.0]), Err(IoError::InvalidFormat(_))));
    w.write_packed(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(w.finish().unwrap(), 2, "the refused rows were not written");

    // The bare `ivecs` has named variable-length files from the first
    // datasets on, and the vvec reader opens them: so does the writer.
    let mut w = VvecWriter::<i32>::create(&p("legacy.ivecs")).unwrap();
    w.write(&[1, 2]).unwrap();
    assert_eq!(w.finish().unwrap(), 1);
}

/// **A vvec writer replaces a stale index**: rewriting a file leaves no
/// index from the previous version for a reader to pair with it.
#[test]
fn a_rewritten_vvec_file_is_read_by_its_own_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.ivvecs");
    let mut w = VvecWriter::<i32>::create(&path).unwrap();
    w.write_all([&[1, 2, 3][..], &[4]]).unwrap();
    w.finish().unwrap();
    let mut w = VvecWriter::<i32>::create(&path).unwrap();
    w.write_all([&[9][..], &[8, 7], &[6, 5, 4]]).unwrap();
    w.finish().unwrap();
    let r = IndexedVvecReader::<i32>::open_path(&path).unwrap();
    assert_eq!(r.count(), 3);
    assert_eq!(VvecReader::get(&r, 2).unwrap(), vec![6, 5, 4]);
}

/// The file-size limit the child runs under: every write past it fails
/// (`EFBIG`), as a write to a full disk does (`ENOSPC`). A limit, not a
/// symlink to `/dev/full`: the writers replace a symlink at their
/// target rather than write through it.
#[cfg(target_os = "linux")]
const FILE_SIZE_LIMIT: u64 = 1 << 20;

/// **A full disk is an error, never a short file**, from the typed
/// writers and from the byte-level ones. The byte-level writers
/// discarded every write error, so a batch larger than their buffer —
/// written past it, straight to the file — failed silently and `finish`
/// reported success.
///
/// Runs [`writes_past_the_file_size_limit_fail`] in a child of this
/// test binary with `RLIMIT_FSIZE` set and `SIGXFSZ` ignored, so the
/// limit touches nothing else in this process.
#[test]
#[cfg(target_os = "linux")]
fn a_full_disk_is_an_error_from_every_writer() {
    use std::os::unix::process::CommandExt;
    let mut child = std::process::Command::new(std::env::current_exe().unwrap());
    child.args(["writes_past_the_file_size_limit_fail", "--exact", "--ignored", "--nocapture"]);
    // SAFETY: runs in the forked child before exec, and calls only
    // async-signal-safe functions.
    unsafe {
        child.pre_exec(|| {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
            let limit = libc::rlimit { rlim_cur: FILE_SIZE_LIMIT, rlim_max: FILE_SIZE_LIMIT };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let out = child.output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success() && text.contains("1 passed"), "{text}{}", String::from_utf8_lossy(&out.stderr));
}

/// The body of [`a_full_disk_is_an_error_from_every_writer`], run only
/// in its file-size-limited child.
#[test]
#[cfg(target_os = "linux")]
#[ignore = "run by a_full_disk_is_an_error_from_every_writer under a file-size limit"]
fn writes_past_the_file_size_limit_fail() {
    let dir = tempfile::tempdir().unwrap();

    let mut typed = XvecWriter::<f32>::create(&dir.path().join("typed.fvecs"), 1024).unwrap();
    let row = vec![1.0f32; 1024];
    let wrote: Result<(), IoError> = (0..8192).try_for_each(|_| typed.write(&row));
    assert!(wrote.is_err() || typed.finish().is_err(), "32 MiB past the limit reported success");

    let mut sink = veks_io::xvec::writer::open(&dir.path().join("bytes.fvecs"), 1024).unwrap();
    let batch = vec![0u8; 4096 * 5000]; // 20 MiB: more than the 16 MiB buffer
    sink.write_records_fixed_dim(0, &batch, 5000, 4096);
    let err = sink.finish().expect_err("a batch written past the buffer and the limit reported success");
    assert!(err.contains("too large") || err.contains("space"), "{err}");

    let mut column = veks_io::scalar::writer::open(&dir.path().join("col.u8"), veks_io::VecFormat::ScalarU8).unwrap();
    for _ in 0..(5 << 20) {
        column.write_record(0, &[7]);
    }
    assert!(column.finish().is_err(), "5 MiB past the limit reported success");
}
