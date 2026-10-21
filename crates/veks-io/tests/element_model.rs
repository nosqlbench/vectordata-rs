// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The element model in veks-io's readers (SRD element access, §14.3
//! step 2): the guards that failed on the code before it, and the
//! behaviour of the generic readers.

use std::io::Write;
use std::path::Path;

use veks_io::{Container, Element, ElementType, IoError, ScalarReader, VectorReader, XvecReader, bf16, f16};

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// The error the streaming reader gives for `path`.
fn stream_error(path: &Path) -> String {
    veks_io::open(path).err().expect("the streaming reader refuses the file")
}

fn write_records(path: &Path, dim: i32, records: &[Vec<u8>]) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in records {
        f.write_all(&dim.to_le_bytes()).unwrap();
        f.write_all(r).unwrap();
    }
}

fn bytes_of<T: Element>(values: &[T]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * T::SIZE];
    for (v, slot) in values.iter().zip(out.chunks_exact_mut(T::SIZE)) {
        v.write_le_bytes(slot);
    }
    out
}

/// A sample of every type: the edges and one value in between.
fn sample<T: Element>(i: usize) -> T {
    let value = match i % 5 {
        0 => veks_datatypes::Value::Int(0),
        1 => veks_datatypes::Value::Int(1),
        2 => veks_datatypes::Value::Int(100),
        3 => veks_datatypes::Value::Int(if T::TYPE.is_integer() && T::TYPE.kind() != veks_datatypes::ElementKind::Unsigned { -7 } else { 7 }),
        _ => veks_datatypes::Value::Int(127),
    };
    T::from_value(value).expect("fits every type")
}

// ── Guards (each failed on the code before this step) ──────────────────

#[test]
fn a_negative_dimension_is_rejected() {
    let dir = tmp();
    let path = dir.path().join("neg.fvec");
    write_records(&path, -4, &[vec![0u8; 16]]);
    let err = XvecReader::<f32>::open_path(&path).unwrap_err();
    assert!(err.to_string().contains("dimension -4 is not positive"), "{err}");
    assert!(stream_error(&path).contains("not positive"));
}

#[test]
fn a_truncated_file_is_rejected_by_the_streaming_reader() {
    let dir = tmp();
    let path = dir.path().join("short.fvec");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&2i32.to_le_bytes()).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&2.0f32.to_le_bytes()).unwrap();
    f.write_all(&2i32.to_le_bytes()).unwrap(); // a second header and nothing after it
    drop(f);
    let err = stream_error(&path);
    assert!(err.contains("file size 16 is not a whole number of 12-byte records"), "{err}");
    assert!(veks_io::probe(&path).is_err());
}

#[test]
fn a_misaligned_f64_record_has_no_slice() {
    let dir = tmp();
    let path = dir.path().join("one.dvec");
    write_records(&path, 1, &[1.5f64.to_le_bytes().to_vec(), 2.5f64.to_le_bytes().to_vec()]);
    let reader = XvecReader::<f64>::open_path(&path).unwrap();
    // Record 0's payload starts 4 bytes into a page-aligned mapping: not
    // an f64 boundary, so there is no slice to borrow — but the values
    // are still there to decode.
    assert_eq!(VectorReader::get_slice(&reader, 0), None);
    assert_eq!(reader.get(0).unwrap(), vec![1.5]);
    let mut out = [0f64; 2];
    assert_eq!(reader.read_into(0, &mut out).unwrap(), 2);
    assert_eq!(out, [1.5, 2.5]);
}

#[test]
fn a_bvecs_file_does_not_open_as_f32() {
    let dir = tmp();
    let path = dir.path().join("codes.bvecs");
    // 17 records of 16 bytes: 340 bytes, which also divides by the 68-byte
    // stride an f32 reader derives from the same header, so nothing but
    // the extension can tell the two apart.
    let records: Vec<Vec<u8>> = (0..17).map(|i| vec![i as u8; 16]).collect();
    write_records(&path, 16, &records);
    assert!(matches!(
        XvecReader::<f32>::open_path(&path),
        Err(IoError::ElementMismatch { file: ElementType::U8, requested: ElementType::F32, .. })
    ));
    let reader = XvecReader::<u8>::open_path(&path).unwrap();
    assert_eq!(reader.count(), 17);
    assert_eq!(VectorReader::get_slice(&reader, 16), Some(&[16u8; 16][..]));
}

// ── The generic readers ────────────────────────────────────────────────

fn mmap_round_trip<T: Element + PartialEq + std::fmt::Debug>(dir: &Path) {
    let dim = 4;
    let records: Vec<Vec<T>> = (0..3).map(|r| (0..dim).map(|d| sample::<T>(r * dim + d)).collect()).collect();
    let path = dir.join(format!("vectors.{}", T::TYPE.extension(Container::Vec)));
    write_records(&path, dim as i32, &records.iter().map(|r| bytes_of(r)).collect::<Vec<_>>());

    let reader = XvecReader::<T>::open_path(&path).unwrap();
    assert_eq!((reader.dim(), reader.count()), (dim, 3), "{}", T::TYPE);
    for (i, want) in records.iter().enumerate() {
        assert_eq!(&reader.get(i).unwrap(), want, "{} get({i})", T::TYPE);
        // The payload sits 4 bytes after a multiple of the record size:
        // on a boundary for types up to 4 bytes wide, and for 8 only in
        // the records whose offset the record size brings back to one.
        match VectorReader::get_slice(&reader, i) {
            Some(slice) => assert_eq!(slice, &want[..], "{} get_slice({i})", T::TYPE),
            None => assert_eq!(T::SIZE, 8, "{} get_slice({i}) should be borrowable", T::TYPE),
        }
        // The unchecked hot-path form agrees wherever the record is
        // aligned.
        if T::SIZE < 8 {
            assert_eq!(reader.get_slice(i), &want[..], "{} unchecked get_slice({i})", T::TYPE);
        }
    }
    let mut out = vec![sample::<T>(0); dim * 3];
    assert_eq!(reader.read_into(0, &mut out).unwrap(), 3);
    assert_eq!(out, records.concat(), "{} read_into", T::TYPE);
    assert_eq!(reader.read_into(2, &mut out).unwrap(), 1);
    assert_eq!(&out[..dim], &records[2][..]);
}

#[test]
fn every_type_round_trips_through_mmap() {
    let dir = tmp();
    mmap_round_trip::<u8>(dir.path());
    mmap_round_trip::<i8>(dir.path());
    mmap_round_trip::<u16>(dir.path());
    mmap_round_trip::<i16>(dir.path());
    mmap_round_trip::<u32>(dir.path());
    mmap_round_trip::<i32>(dir.path());
    mmap_round_trip::<u64>(dir.path());
    mmap_round_trip::<i64>(dir.path());
    mmap_round_trip::<f16>(dir.path());
    mmap_round_trip::<bf16>(dir.path());
    mmap_round_trip::<f32>(dir.path());
    mmap_round_trip::<f64>(dir.path());
}

fn column_round_trip<T: Element + PartialEq + std::fmt::Debug>(dir: &Path) {
    let values: Vec<T> = (0..9).map(sample::<T>).collect();
    let path = dir.join(format!("column.{}", T::TYPE.extension(Container::Scalar)));
    std::fs::write(&path, bytes_of(&values)).unwrap();

    let reader = ScalarReader::<T>::open_path(&path).unwrap();
    assert_eq!(reader.count(), 9, "{}", T::TYPE);
    assert_eq!(reader.as_slice(), Some(&values[..]), "{} as_slice", T::TYPE);
    for (i, want) in values.iter().enumerate() {
        assert_eq!(reader.get(i).unwrap(), vec![*want], "{} get({i})", T::TYPE);
        assert_eq!(reader.get_slice(i), Some(std::slice::from_ref(want)), "{} get_slice({i})", T::TYPE);
    }
    let mut out = vec![sample::<T>(0); 4];
    assert_eq!(reader.read_into(7, &mut out).unwrap(), 2);
    assert_eq!(&out[..2], &values[7..]);
}

#[test]
fn a_scalar_column_is_one_slice() {
    let dir = tmp();
    column_round_trip::<u8>(dir.path());
    column_round_trip::<i8>(dir.path());
    column_round_trip::<u16>(dir.path());
    column_round_trip::<i16>(dir.path());
    column_round_trip::<u32>(dir.path());
    column_round_trip::<i32>(dir.path());
    column_round_trip::<u64>(dir.path());
    column_round_trip::<i64>(dir.path());
    column_round_trip::<f16>(dir.path());
    column_round_trip::<bf16>(dir.path());
    column_round_trip::<f32>(dir.path());
    column_round_trip::<f64>(dir.path());
}

#[test]
fn bf16_vectors_round_trip_through_the_writer() {
    let dir = tmp();
    let path = dir.path().join("base.bf16vecs");
    let rows: Vec<Vec<bf16>> = vec![
        vec![bf16::from_f32(1.0), bf16::from_f32(-2.5), bf16::from_f32(3.0e30)],
        vec![bf16::from_f32(0.0), bf16::from_f32(0.1), bf16::from_f32(-1.0)],
    ];
    let mut writer = veks_io::create(&path, 3).unwrap();
    for (i, row) in rows.iter().enumerate() {
        writer.write_record(i as i64, &bytes_of(row));
    }
    writer.finish().unwrap();

    let meta = veks_io::probe(&path).unwrap();
    assert_eq!((meta.dimension, meta.element_size, meta.record_count), (3, 2, Some(2)));
    let mut stream = veks_io::open(&path).unwrap();
    assert_eq!(stream.next_record().unwrap(), bytes_of(&rows[0]));

    let reader = XvecReader::<bf16>::open_path(&path).unwrap();
    assert_eq!(VectorReader::get_slice(&reader, 1), Some(&rows[1][..]));
    assert_eq!(reader.get_slice(0)[2].to_f32(), bf16::from_f32(3.0e30).to_f32());
    // The same bytes are not f16: a different format, refused.
    assert!(matches!(
        XvecReader::<f16>::open_path(&path),
        Err(IoError::ElementMismatch { file: ElementType::BF16, requested: ElementType::F16, .. })
    ));
    assert!(matches!(XvecReader::<u16>::open_path(&path), Err(IoError::ElementMismatch { .. })));
}

#[test]
fn f32_columns_round_trip_through_the_scalar_writer() {
    let dir = tmp();
    let path = dir.path().join("quant_scale.f32");
    let scales = [0.5f32, 0.25, 1.0e-3, 7.0];
    let mut writer = veks_io::create(&path, 1).unwrap();
    for (i, s) in scales.iter().enumerate() {
        writer.write_record(i as i64, &s.to_le_bytes());
    }
    writer.finish().unwrap();

    let meta = veks_io::probe(&path).unwrap();
    assert_eq!((meta.dimension, meta.element_size, meta.record_count), (1, 4, Some(4)));
    let reader = ScalarReader::<f32>::open_path(&path).unwrap();
    assert_eq!(reader.as_slice(), Some(&scales[..]));
    assert!(matches!(ScalarReader::<i32>::open_path(&path), Err(IoError::ElementMismatch { .. })));
}

#[test]
#[allow(deprecated)]
fn the_deprecated_constructors_open_the_same_file() {
    let dir = tmp();
    macro_rules! same_as_open {
        ($t:ty, $ctor:ident, $ext:literal) => {{
            let path = dir.path().join(concat!("base.", $ext));
            let rows: Vec<Vec<$t>> = (0..2).map(|r| (0..2).map(|d| sample::<$t>(r * 2 + d)).collect()).collect();
            write_records(&path, 2, &rows.iter().map(|r| bytes_of(r)).collect::<Vec<_>>());
            let by_name = XvecReader::<$t>::$ctor(&path).unwrap();
            let by_type = XvecReader::<$t>::open_path(&path).unwrap();
            assert_eq!((by_name.count(), by_name.dim(), by_name.entry_size()), (by_type.count(), by_type.dim(), by_type.entry_size()));
            assert_eq!(by_name.get(1).unwrap(), by_type.get(1).unwrap());
            // The earlier name is the same type.
            let _: &veks_io::MmapReader<$t> = &by_name;
        }};
    }
    same_as_open!(f32, open_fvec, "fvec");
    same_as_open!(i32, open_ivec, "ivec");
    same_as_open!(f16, open_mvec, "mvec");
    same_as_open!(f64, open_dvec, "dvec");
    same_as_open!(u8, open_bvec, "bvec");
    same_as_open!(i16, open_svec, "svec");

    macro_rules! same_column {
        ($t:ty, $ctor:ident, $ext:literal) => {{
            let column = dir.path().join(concat!("ids.", $ext));
            let values: Vec<$t> = (0..3).map(sample::<$t>).collect();
            std::fs::write(&column, bytes_of(&values)).unwrap();
            assert_eq!(ScalarReader::<$t>::$ctor(&column).unwrap().as_slice(), Some(&values[..]));
            let _: veks_io::scalar::mmap::ScalarMmapReader<$t> = ScalarReader::<$t>::open_path(&column).unwrap();
        }};
    }
    same_column!(u8, open_u8, "u8");
    same_column!(i8, open_i8, "i8");
    same_column!(u16, open_u16, "u16");
    same_column!(i16, open_i16, "i16");
    same_column!(u32, open_u32, "u32");
    same_column!(i32, open_i32, "i32");
    same_column!(u64, open_u64, "u64");
    same_column!(i64, open_i64, "i64");
}

#[test]
fn errors_say_what_happened() {
    let dir = tmp();
    let missing = XvecReader::<f32>::open_path(&dir.path().join("missing.fvec")).unwrap_err();
    assert!(matches!(&missing, IoError::Io(e) if e.kind() == std::io::ErrorKind::NotFound), "{missing}");

    let path = dir.path().join("two.fvec");
    write_records(&path, 1, &[bytes_of(&[1.0f32]), bytes_of(&[2.0f32])]);
    let reader = XvecReader::<f32>::open_path(&path).unwrap();
    let past = reader.read_into(3, &mut [0f32; 1]).unwrap_err();
    assert_eq!(past.to_string(), "Index out of bounds: 3");
    assert!(matches!(past, IoError::OutOfBounds(3)));

    let wrong = XvecReader::<i32>::open_path(&path).unwrap_err();
    assert_eq!(wrong.to_string(), "the file holds f32 elements and cannot be read as i32: no conversion between the two");

    let varlen = dir.path().join("rows.fvvecs");
    write_records(&varlen, 1, &[bytes_of(&[1.0f32])]);
    let err = XvecReader::<f32>::open_path(&varlen).unwrap_err();
    assert!(err.to_string().contains("holds variable-length records of f32; read it with IndexedVvecReader"), "{err}");

    let unknown = dir.path().join("rows.xyz");
    write_records(&unknown, 1, &[bytes_of(&[1.0f32])]);
    let err = XvecReader::<f32>::open_path(&unknown).unwrap_err();
    assert!(err.to_string().contains("cannot infer element type from extension '.xyz'"), "{err}");
}

#[test]
fn readers_describe_themselves() {
    let dir = tmp();
    let path = dir.path().join("two.bf16vecs");
    write_records(&path, 3, &[bytes_of(&[bf16::from_f32(1.0); 3]), bytes_of(&[bf16::from_f32(2.0); 3])]);
    let reader = XvecReader::<bf16>::open_path(&path).unwrap();
    reader.advise_sequential();
    let shown = format!("{reader:?}");
    assert!(shown.contains("element: BF16") && shown.contains("dim: 3") && shown.contains("count: 2"), "{shown}");

    let column = dir.path().join("ids.u64");
    std::fs::write(&column, bytes_of(&[7u64; 4])).unwrap();
    let shown = format!("{:?}", ScalarReader::<u64>::open_path(&column).unwrap());
    assert!(shown.contains("element: U64") && shown.contains("count: 4"), "{shown}");
}

#[test]
fn the_streaming_readers_take_only_their_own_layout() {
    let dir = tmp();
    let vectors = dir.path().join("base.fvec");
    write_records(&vectors, 1, &[bytes_of(&[1.0f32])]);
    assert!(veks_io::xvec::reader::open(&vectors, veks_io::VecFormat::ScalarU8).is_err());
    assert!(veks_io::xvec::reader::probe(&vectors, veks_io::VecFormat::Slab).is_err());
    let column = dir.path().join("ids.u8");
    std::fs::write(&column, [1u8, 2]).unwrap();
    assert!(veks_io::scalar::reader::open(&column, veks_io::VecFormat::Slab).is_err());
    assert!(veks_io::scalar::reader::probe(&column, veks_io::VecFormat::Npy).is_err());
    assert!(veks_io::scalar::writer::open(&column, veks_io::VecFormat::Parquet).is_err());
}

#[test]
fn a_column_does_not_open_as_vectors_and_vice_versa() {
    let dir = tmp();
    let column = dir.path().join("codes.u8");
    std::fs::write(&column, [1u8, 2, 3, 4]).unwrap();
    let err = XvecReader::<u8>::open_path(&column).unwrap_err();
    assert!(err.to_string().contains("is a column of u8, not vectors"), "{err}");

    let vectors = dir.path().join("codes.bvecs");
    write_records(&vectors, 2, &[vec![1, 2]]);
    let err = ScalarReader::<u8>::open_path(&vectors).unwrap_err();
    assert!(err.to_string().contains("holds vectors of u8, not a column"), "{err}");
}

/// **An empty file is zero records**, for vectors and columns alike:
/// an empty facet is a facet, and a reader of it says so rather than
/// refusing it. The vectors' dimension is undefined, there being no
/// record to read it from.
#[test]
fn an_empty_file_is_zero_records() {
    let dir = tmp();
    let vectors = dir.path().join("empty.fvec");
    std::fs::File::create(&vectors).unwrap();
    let reader = XvecReader::<f32>::open_path(&vectors).unwrap();
    assert_eq!((reader.count(), reader.dim()), (0, veks_io::DIM_UNDEFINED));
    assert!(matches!(reader.get(0), Err(IoError::OutOfBounds(0))));

    let column = dir.path().join("empty.u8");
    std::fs::write(&column, []).unwrap();
    let reader = ScalarReader::<u8>::open_path(&column).unwrap();
    assert_eq!(reader.count(), 0);
}

// ── The writers, every type (SRD element access, §13.10) ───────────────

/// **The scalar writer writes a column of every element type**, read
/// back through `ScalarReader<T>` as one slice equal to what was
/// written; and the xvec writer writes records of every type, read
/// back value for value — the 16- and 32-bit unsigned types included,
/// which were only ever probed for size.
#[test]
fn the_writers_write_every_element_type() {
    use veks_io::VecFormat;
    for &ty in ElementType::ALL {
        veks_datatypes::dispatch_element!(ty, T => {
            let dir = tmp();
            let values: Vec<T> = (0..9).map(sample::<T>).collect();
            let elem = bytes_of(&values);

            let column = dir.path().join(format!("c.{}", ty.extension(Container::Scalar)));
            let format = VecFormat::of(Container::Scalar, ty).unwrap_or_else(|| panic!("{ty} has a scalar format"));
            let mut sink = veks_io::create_format(&column, format, 1).unwrap_or_else(|e| panic!("{ty}: {e}"));
            for i in 0..values.len() {
                sink.write_record(i as i64, &elem[i * T::SIZE..(i + 1) * T::SIZE]);
            }
            sink.finish().unwrap();
            let reader = ScalarReader::<T>::open_path(&column).unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!(reader.count(), values.len(), "{ty}");
            assert_eq!(bytes_of(reader.as_slice().unwrap()), elem, "{ty}: the column read back");
            let meta = veks_io::probe(&column).unwrap();
            assert_eq!((meta.element_size, meta.record_count), (T::SIZE, Some(values.len() as u64)), "{ty}: probed");

            let dim = 3;
            let vectors = dir.path().join(format!("v.{}", ty.extension(Container::Vec)));
            let format = VecFormat::of(Container::Vec, ty).unwrap();
            let mut sink = veks_io::create_format(&vectors, format, dim as u32).unwrap_or_else(|e| panic!("{ty}: {e}"));
            for (i, record) in elem.chunks_exact(T::SIZE * dim).enumerate() {
                sink.write_record(i as i64, record);
            }
            sink.finish().unwrap();
            let reader = XvecReader::<T>::open_path(&vectors).unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!((reader.count(), reader.dim()), (3, dim), "{ty}");
            for i in 0..3 {
                assert_eq!(bytes_of(&reader.get(i).unwrap()), elem[i * dim * T::SIZE..(i + 1) * dim * T::SIZE], "{ty}: record {i}");
            }
            let mut streamed = veks_io::open(&vectors).unwrap();
            let first = streamed.next_record().unwrap();
            assert_eq!(first, elem[..dim * T::SIZE], "{ty}: the streaming reader agrees");
        });
    }
}
