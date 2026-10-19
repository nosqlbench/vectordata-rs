// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The element model in vectordata's raw readers (SRD element access,
//! §14.3 step 3): the guards that failed on the code before it, and
//! the behaviour of the shared model's tables as this crate uses them.

use std::io::Write;
use std::path::Path;

use vectordata::dataset::facet::FacetFormat;
use vectordata::io::{IndexedVvecReader, IoError, XvecReader, open_vec};
use vectordata::typed_access::{Container, ElementType, TypedReader};

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn write_records(path: &Path, dim: i32, records: &[Vec<u8>]) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in records {
        f.write_all(&dim.to_le_bytes()).unwrap();
        f.write_all(r).unwrap();
    }
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn mismatch(err: &IoError) -> Option<(ElementType, ElementType)> {
    match err {
        IoError::ElementMismatch { file, requested, .. } => Some((*file, *requested)),
        _ => None,
    }
}

// ── Guards (each failed on the code before this step) ──────────────────

#[test]
fn an_ivec_does_not_open_as_f32_by_path() {
    let dir = tmp();
    let path = dir.path().join("ids.ivec");
    write_records(&path, 2, &[vec![1, 0, 0, 0, 2, 0, 0, 0]]);
    let err = XvecReader::<f32>::open_path(&path).expect_err("an i32 file is not an f32 file");
    assert_eq!(mismatch(&err), Some((ElementType::I32, ElementType::F32)));
}

#[test]
fn an_fvec_does_not_open_as_i32() {
    let dir = tmp();
    let path = dir.path().join("base.fvec");
    write_records(&path, 2, &[f32s(&[1.0, 2.0])]);
    let err = XvecReader::<i32>::open(path.to_str().unwrap()).expect_err("an f32 file is not an i32 file");
    assert_eq!(mismatch(&err), Some((ElementType::F32, ElementType::I32)));
}

#[test]
fn a_dvec_opened_as_f32_is_a_mismatch_not_a_short_file() {
    let dir = tmp();
    let path = dir.path().join("base.dvec");
    // 5 records of 4 f64: 180 bytes, which is also 9 records of the
    // 20-byte stride an f32 reader derives from the same header.
    let records: Vec<Vec<u8>> = (0..5).map(|r| (0..4).flat_map(|d| ((r * 4 + d) as f64).to_le_bytes()).collect()).collect();
    write_records(&path, 4, &records);
    let err = XvecReader::<f32>::open_path(&path).expect_err("an f64 file is not an f32 file");
    assert_eq!(mismatch(&err), Some((ElementType::F64, ElementType::F32)), "{err}");
    assert!(!err.to_string().contains("Variable-length"), "the wrong reason: {err}");
    let reader = XvecReader::<f64>::open_path(&path).unwrap();
    assert_eq!(vectordata::VectorReader::count(&reader), 5);
}

#[test]
fn a_u32vvec_does_not_open_as_i32() {
    let dir = tmp();
    let path = dir.path().join("rows.u32vvec");
    write_records(&path, 2, &[vec![1, 0, 0, 0, 2, 0, 0, 0]]);
    let err = IndexedVvecReader::<i32>::open_path(&path).expect_err("a u32 file is not an i32 file");
    assert_eq!(mismatch(&err), Some((ElementType::U32, ElementType::I32)));
    assert!(IndexedVvecReader::<u32>::open_path(&path).is_ok());
}

#[test]
fn a_float_file_is_refused_by_the_typed_reader() {
    let dir = tmp();
    let path = dir.path().join("base.fvec");
    write_records(&path, 2, &[f32s(&[1.5, 2.5])]);
    let err = TypedReader::<i32>::open(&path).expect_err("an f32 file read as i32 is not zeros");
    assert!(err.to_string().contains("cannot open f32 file as i32"), "{err}");
    // At its own type, and widened to f64, the values are the values.
    let native = TypedReader::<f32>::open(&path).unwrap();
    assert_eq!(native.get_record(0).unwrap(), vec![1.5, 2.5]);
    let wide = TypedReader::<f64>::open(&path).unwrap();
    assert_eq!(wide.get_record(0).unwrap(), vec![1.5, 2.5]);
}

#[test]
fn bf16_and_the_float_columns_are_recognised_extensions() {
    assert_eq!(ElementType::from_extension("bf16vecs"), Some(ElementType::BF16));
    assert_eq!(ElementType::from_extension("bf16"), Some(ElementType::BF16));
    assert_eq!(ElementType::from_extension("f32"), Some(ElementType::F32));
    assert_eq!(FacetFormat::from_extension("f32"), Some(FacetFormat::FloatScalar));
    assert_eq!(FacetFormat::from_extension("bf16"), Some(FacetFormat::FloatScalar));
    assert_eq!(FacetFormat::from_extension("bf16vecs"), Some(FacetFormat::FloatXvec));
    assert_eq!(FacetFormat::from_extension("i32"), Some(FacetFormat::ScalarPacked), "integer columns keep their format");
}

// ── Behaviour ──────────────────────────────────────────────────────────

#[test]
fn open_vec_refuses_a_column() {
    let dir = tmp();
    let path = dir.path().join("scale.f32");
    std::fs::write(&path, f32s(&[0.5, 0.25])).unwrap();
    let err = match open_vec::<f32>(path.to_str().unwrap()) {
        Err(e) => e,
        Ok(reader) => panic!("a column is not vectors, but opened with count {}", reader.count()),
    };
    assert!(matches!(err, IoError::InvalidFormat(_)), "{err}");
    assert!(err.to_string().contains("column"), "{err}");
    // A column reads through the typed reader at its own type.
    let column = TypedReader::<f32>::open(&path).unwrap();
    assert_eq!((column.count(), column.dim()), (2, 1));
    assert_eq!(column.get_value(1).unwrap(), 0.25);
}

#[test]
fn a_float_file_opens_as_itself_only_through_the_raw_readers() {
    let dir = tmp();
    let path = dir.path().join("base.mvec");
    let halves: Vec<u8> = [1.5f32, -2.0].iter().flat_map(|v| vectordata::typed_access::f16::from_f32(*v).to_le_bytes()).collect();
    write_records(&path, 2, &[halves]);
    // f16 widens to f32 by the table, but a raw reader does not convert:
    // it says so and names the view as the way to widen.
    let err = XvecReader::<f32>::open_path(&path).expect_err("a raw reader reads a file at its own type");
    assert_eq!(mismatch(&err), Some((ElementType::F16, ElementType::F32)));
    assert!(err.to_string().contains("dataset view"), "{err}");
    let native = XvecReader::<vectordata::typed_access::f16>::open_path(&path).unwrap();
    assert_eq!(vectordata::VectorReader::get(&native, 0).unwrap()[1].to_f32(), -2.0);
}

#[test]
fn the_facet_format_lists_derive_from_the_element_table() {
    for &ty in ElementType::ALL {
        for &container in Container::ALL {
            let spelled = ty.extension(container);
            let formats: Vec<_> = FacetFormat::ALL.iter().filter(|f| f.extensions().contains(&spelled)).collect();
            match FacetFormat::of(container, ty) {
                Some(want) => assert_eq!(formats, vec![&want], "{spelled} lands in exactly its format"),
                None => assert!(formats.is_empty(), "{spelled} has no facet format"),
            }
            for alias in ty.aliases(container) {
                assert_eq!(FacetFormat::from_extension(alias), FacetFormat::of(container, ty), "{alias}");
            }
        }
    }
    // Canonical spellings lead their type's run, so a writer that picks
    // the first spelling of a type picks the typed one.
    assert_eq!(FacetFormat::FloatXvec.extensions()[0], "f16vecs");
    assert!(FacetFormat::FloatXvec.extensions().contains(&"bf16vecs"));
    assert!(FacetFormat::IntegerVarXvec.extensions().contains(&"ivvecs"));
}

#[test]
fn the_mismatch_error_names_both_types_and_the_reason() {
    let dir = tmp();
    let path = dir.path().join("codes.bvecs");
    write_records(&path, 2, &[vec![1, 2]]);
    let err = XvecReader::<i8>::open_path(&path).expect_err("a u8 file is not an i8 file");
    assert_eq!(
        err.to_string(),
        "the file holds u8 elements and cannot be read as i8: a raw reader reads a file at its own type; open it through a dataset view to check each value"
    );
    let err = XvecReader::<f64>::open_path(&path).expect_err("a u8 file is not an f64 file");
    assert!(err.to_string().ends_with("open it through a dataset view to widen"), "{err}");
    let err = XvecReader::<u8>::open_path(&dir.path().join("codes.xyz")).expect_err("unknown");
    assert!(err.to_string().contains("cannot infer element type from extension '.xyz'"), "{err}");
}
