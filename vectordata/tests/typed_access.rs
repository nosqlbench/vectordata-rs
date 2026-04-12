// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for typed data access.
//!
//! Tests the full access path: write files in various formats → create
//! dataset.yaml → load via TestDataGroup → access facets via TypedReader
//! with native, widening, and cross-sign conversions.

mod support;

use std::io::Write;
use std::path::Path;

use vectordata::typed_access::{ElementType, TypedReader};
use vectordata::TestDataView; // trait import for facet_element_type()

fn make_tmp() -> tempfile::TempDir {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

// ── File writers for each format ───────────────────────────────────

fn write_scalar_u8(path: &Path, values: &[u8]) {
    std::fs::write(path, values).unwrap();
}

fn write_scalar_i8(path: &Path, values: &[i8]) {
    let bytes: Vec<u8> = values.iter().map(|v| *v as u8).collect();
    std::fs::write(path, &bytes).unwrap();
}

fn write_scalar_u16(path: &Path, values: &[u16]) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values { f.write_all(&v.to_le_bytes()).unwrap(); }
}

fn write_scalar_i16(path: &Path, values: &[i16]) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values { f.write_all(&v.to_le_bytes()).unwrap(); }
}

fn write_scalar_u32(path: &Path, values: &[u32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values { f.write_all(&v.to_le_bytes()).unwrap(); }
}

fn write_scalar_i32(path: &Path, values: &[i32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values { f.write_all(&v.to_le_bytes()).unwrap(); }
}

fn write_scalar_u64(path: &Path, values: &[u64]) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values { f.write_all(&v.to_le_bytes()).unwrap(); }
}

fn write_scalar_i64(path: &Path, values: &[i64]) {
    let mut f = std::fs::File::create(path).unwrap();
    for v in values { f.write_all(&v.to_le_bytes()).unwrap(); }
}

fn write_ivec(path: &Path, dim: u32, records: &[Vec<i32>]) {
    let mut f = std::fs::File::create(path).unwrap();
    for rec in records {
        f.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for v in rec { f.write_all(&v.to_le_bytes()).unwrap(); }
    }
}

fn write_fvec(path: &Path, dim: u32, records: &[Vec<f32>]) {
    let mut f = std::fs::File::create(path).unwrap();
    for rec in records {
        f.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for v in rec { f.write_all(&v.to_le_bytes()).unwrap(); }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scalar format tests — standalone TypedReader
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scalar_u8_native_access() {
    let tmp = make_tmp();
    let path = tmp.path().join("meta.u8");
    write_scalar_u8(&path, &[0, 42, 127, 200, 255]);

    let r = TypedReader::<u8>::open(&path).unwrap();
    assert_eq!(r.native_type(), ElementType::U8);
    assert!(r.is_native());
    assert_eq!(r.count(), 5);
    assert_eq!(r.dim(), 1);
    assert_eq!(r.get_native(0), 0);
    assert_eq!(r.get_native(2), 127);
    assert_eq!(r.get_native(4), 255);
}

#[test]
fn scalar_i8_native_access() {
    let tmp = make_tmp();
    let path = tmp.path().join("meta.i8");
    write_scalar_i8(&path, &[-128, -1, 0, 1, 127]);

    let r = TypedReader::<i8>::open(&path).unwrap();
    assert_eq!(r.count(), 5);
    assert_eq!(r.get_value(0).unwrap(), -128i8);
    assert_eq!(r.get_value(1).unwrap(), -1i8);
    assert_eq!(r.get_value(4).unwrap(), 127i8);
}

#[test]
fn scalar_u16_native_access() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u16");
    write_scalar_u16(&path, &[0, 1000, 65535]);

    let r = TypedReader::<u16>::open(&path).unwrap();
    assert_eq!(r.count(), 3);
    assert_eq!(r.get_value(2).unwrap(), 65535u16);
}

#[test]
fn scalar_i32_native_access() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i32");
    write_scalar_i32(&path, &[i32::MIN, -1, 0, 1, i32::MAX]);

    let r = TypedReader::<i32>::open(&path).unwrap();
    assert_eq!(r.count(), 5);
    assert_eq!(r.get_native(0), i32::MIN);
    assert_eq!(r.get_native(4), i32::MAX);
}

#[test]
fn scalar_u64_native_access() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u64");
    write_scalar_u64(&path, &[0, u64::MAX]);

    let r = TypedReader::<u64>::open(&path).unwrap();
    assert_eq!(r.count(), 2);
    assert_eq!(r.get_value(1).unwrap(), u64::MAX);
}

// ═══════════════════════════════════════════════════════════════════
// Widening conversions — always succeed
// ═══════════════════════════════════════════════════════════════════

#[test]
fn widen_u8_to_i16() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u8");
    write_scalar_u8(&path, &[0, 255]);

    let r = TypedReader::<i16>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), 0i16);
    assert_eq!(r.get_value(1).unwrap(), 255i16);
}

#[test]
fn widen_u8_to_i32() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u8");
    write_scalar_u8(&path, &[255]);

    let r = TypedReader::<i32>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), 255i32);
}

#[test]
fn widen_u8_to_u64() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u8");
    write_scalar_u8(&path, &[255]);

    let r = TypedReader::<u64>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), 255u64);
}

#[test]
fn widen_i8_to_i64() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i8");
    write_scalar_i8(&path, &[-128, 127]);

    let r = TypedReader::<i64>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), -128i64);
    assert_eq!(r.get_value(1).unwrap(), 127i64);
}

#[test]
fn widen_u16_to_u32() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u16");
    write_scalar_u16(&path, &[65535]);

    let r = TypedReader::<u32>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), 65535u32);
}

#[test]
fn widen_i16_to_i32() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i16");
    write_scalar_i16(&path, &[-32768, 32767]);

    let r = TypedReader::<i32>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), -32768i32);
    assert_eq!(r.get_value(1).unwrap(), 32767i32);
}

#[test]
fn widen_i32_to_i64() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i32");
    write_scalar_i32(&path, &[i32::MIN, i32::MAX]);

    let r = TypedReader::<i64>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), i32::MIN as i64);
    assert_eq!(r.get_value(1).unwrap(), i32::MAX as i64);
}

// ═══════════════════════════════════════════════════════════════════
// Cross-sign same width — checked per value
// ═══════════════════════════════════════════════════════════════════

#[test]
fn cross_sign_u8_to_i8_ok() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u8");
    write_scalar_u8(&path, &[0, 42, 127]);

    let r = TypedReader::<i8>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), 0i8);
    assert_eq!(r.get_value(1).unwrap(), 42i8);
    assert_eq!(r.get_value(2).unwrap(), 127i8);
}

#[test]
fn cross_sign_u8_to_i8_overflow() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u8");
    write_scalar_u8(&path, &[128]);

    let r = TypedReader::<i8>::open(&path).unwrap();
    let err = r.get_value(0).unwrap_err();
    assert!(matches!(err, vectordata::typed_access::TypedAccessError::ValueOverflow { .. }));
}

#[test]
fn cross_sign_i8_to_u8_ok() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i8");
    write_scalar_i8(&path, &[0, 127]);

    let r = TypedReader::<u8>::open(&path).unwrap();
    assert_eq!(r.get_value(0).unwrap(), 0u8);
    assert_eq!(r.get_value(1).unwrap(), 127u8);
}

#[test]
fn cross_sign_i8_to_u8_negative_fails() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i8");
    write_scalar_i8(&path, &[-1]);

    let r = TypedReader::<u8>::open(&path).unwrap();
    assert!(r.get_value(0).is_err());
}

#[test]
fn cross_sign_u32_to_i32_ok() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u32");
    write_scalar_u32(&path, &[0, 42, i32::MAX as u32]);

    let r = TypedReader::<i32>::open(&path).unwrap();
    assert_eq!(r.get_value(2).unwrap(), i32::MAX);
}

#[test]
fn cross_sign_u32_to_i32_overflow() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u32");
    write_scalar_u32(&path, &[u32::MAX]);

    let r = TypedReader::<i32>::open(&path).unwrap();
    assert!(r.get_value(0).is_err());
}

#[test]
fn cross_sign_i32_to_u32_negative_fails() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i32");
    write_scalar_i32(&path, &[-1]);

    let r = TypedReader::<u32>::open(&path).unwrap();
    assert!(r.get_value(0).is_err());
}

// ═══════════════════════════════════════════════════════════════════
// Narrowing — rejected at open time
// ═══════════════════════════════════════════════════════════════════

#[test]
fn narrowing_i32_to_u8_rejected() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i32");
    write_scalar_i32(&path, &[42]);
    assert!(TypedReader::<u8>::open(&path).is_err());
}

#[test]
fn narrowing_i32_to_i16_rejected() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i32");
    write_scalar_i32(&path, &[42]);
    assert!(TypedReader::<i16>::open(&path).is_err());
}

#[test]
fn narrowing_u64_to_u32_rejected() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.u64");
    write_scalar_u64(&path, &[42]);
    assert!(TypedReader::<u32>::open(&path).is_err());
}

#[test]
fn narrowing_i64_to_i8_rejected() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.i64");
    write_scalar_i64(&path, &[42]);
    assert!(TypedReader::<i8>::open(&path).is_err());
}

// ═══════════════════════════════════════════════════════════════════
// Vector (xvec) format tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn ivec_native_record_access() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.ivec");
    write_ivec(&path, 3, &[vec![10, 20, 30], vec![40, 50, 60]]);

    let r = TypedReader::<i32>::open(&path).unwrap();
    assert_eq!(r.count(), 2);
    assert_eq!(r.dim(), 3);
    assert_eq!(r.get_record(0).unwrap(), vec![10, 20, 30]);
    assert_eq!(r.get_record(1).unwrap(), vec![40, 50, 60]);
}

#[test]
fn ivec_as_i64_widening() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.ivec");
    write_ivec(&path, 2, &[vec![-1, i32::MAX]]);

    let r = TypedReader::<i64>::open(&path).unwrap();
    let rec = r.get_record(0).unwrap();
    assert_eq!(rec, vec![-1i64, i32::MAX as i64]);
}

#[test]
fn ivec_as_u32_cross_sign_ok() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.ivec");
    write_ivec(&path, 2, &[vec![0, 100]]);

    let r = TypedReader::<u32>::open(&path).unwrap();
    let rec = r.get_record(0).unwrap();
    assert_eq!(rec, vec![0u32, 100]);
}

#[test]
fn ivec_as_u32_negative_fails() {
    let tmp = make_tmp();
    let path = tmp.path().join("data.ivec");
    write_ivec(&path, 1, &[vec![-1]]);

    let r = TypedReader::<u32>::open(&path).unwrap();
    assert!(r.get_record(0).is_err());
}

// ═══════════════════════════════════════════════════════════════════
// Dataset integration — TestDataGroup + TestDataView
// ═══════════════════════════════════════════════════════════════════

#[test]
fn dataset_facet_element_type() {
    let tmp = make_tmp();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();

    // Write metadata as u8 and predicates as u8
    write_scalar_u8(&ds.join("profiles/default/metadata_content.u8"), &[0, 1, 2, 3, 4]);
    write_scalar_u8(&ds.join("profiles/default/predicates.u8"), &[1, 3]);
    write_ivec(&ds.join("profiles/default/metadata_indices.ivec"), 2, &[vec![1, 3], vec![0, 2]]);
    write_fvec(&ds.join("profiles/default/base_vectors.fvec"), 4, &[
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
    ]);

    let yaml = r#"
name: typed-test
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    metadata_content: profiles/default/metadata_content.u8
    metadata_predicates: profiles/default/predicates.u8
    predicate_results: profiles/default/metadata_indices.ivec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();

    // Interrogate types
    let meta_type = view.facet_element_type("metadata_content").unwrap();
    assert_eq!(meta_type, ElementType::U8);

    let pred_type = view.facet_element_type("metadata_predicates").unwrap();
    assert_eq!(pred_type, ElementType::U8);

    let results_type = view.facet_element_type("predicate_results").unwrap();
    assert_eq!(results_type, ElementType::I32);

    let base_type = view.facet_element_type("base_vectors").unwrap();
    assert_eq!(base_type, ElementType::F32);
}

#[test]
fn dataset_open_facet_typed_native() {
    let tmp = make_tmp();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();

    write_scalar_u8(&ds.join("profiles/default/meta.u8"), &[7, 3, 10, 0, 5]);

    let yaml = r#"
name: native-test
profiles:
  default:
    metadata_content: profiles/default/meta.u8
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    // GenericTestDataView has open_facet_typed
    let view = group.generic_view("default").unwrap();

    let reader = view.open_facet_typed::<u8>("metadata_content").unwrap();
    assert!(reader.is_native());
    assert_eq!(reader.count(), 5);
    assert_eq!(reader.get_native(0), 7);
    assert_eq!(reader.get_native(3), 0);
}

#[test]
fn dataset_open_facet_typed_widening() {
    let tmp = make_tmp();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();

    write_scalar_u8(&ds.join("profiles/default/meta.u8"), &[255, 0, 42]);

    let yaml = r#"
name: widen-test
profiles:
  default:
    metadata_content: profiles/default/meta.u8
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.generic_view("default").unwrap();

    // Open u8 file as i32 — widening
    let reader = view.open_facet_typed::<i32>("metadata_content").unwrap();
    assert!(!reader.is_native());
    assert_eq!(reader.get_value(0).unwrap(), 255i32);
    assert_eq!(reader.get_value(2).unwrap(), 42i32);
}

#[test]
fn dataset_runtime_dispatch() {
    let tmp = make_tmp();
    let ds = tmp.path().join("ds");
    std::fs::create_dir_all(ds.join("profiles/default")).unwrap();

    write_scalar_u8(&ds.join("profiles/default/meta.u8"), &[10, 20, 30]);

    let yaml = r#"
name: dispatch-test
profiles:
  default:
    metadata_content: profiles/default/meta.u8
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.generic_view("default").unwrap();

    // Runtime dispatch: check type, then open with native arms
    let etype = view.facet_element_type("metadata_content").unwrap();
    let sum: i64 = match etype {
        ElementType::U8 => {
            let r = view.open_facet_typed::<u8>("metadata_content").unwrap();
            (0..r.count()).map(|i| r.get_native(i) as i64).sum()
        }
        ElementType::I32 => {
            let r = view.open_facet_typed::<i32>("metadata_content").unwrap();
            (0..r.count()).map(|i| r.get_native(i) as i64).sum()
        }
        _ => panic!("unexpected type"),
    };
    assert_eq!(sum, 60); // 10 + 20 + 30
}

// ═══════════════════════════════════════════════════════════════════
// Mixed format dataset — multiple facet types
// ═══════════════════════════════════════════════════════════════════

#[test]
fn dataset_mixed_formats() {
    let tmp = make_tmp();
    let ds = tmp.path().join("ds");
    let prof = ds.join("profiles/default");
    std::fs::create_dir_all(&prof).unwrap();

    // Base vectors: f32
    write_fvec(&prof.join("base.fvec"), 2, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
    // Metadata: u8
    write_scalar_u8(&prof.join("meta.u8"), &[5, 10]);
    // Predicates: u8
    write_scalar_u8(&prof.join("pred.u8"), &[5]);
    // Results: ivec (ordinal lists)
    write_ivec(&prof.join("results.ivec"), 1, &[vec![0]]); // predicate 0 matches record 0

    let yaml = r#"
name: mixed-test
profiles:
  default:
    base_vectors: profiles/default/base.fvec
    metadata_content: profiles/default/meta.u8
    metadata_predicates: profiles/default/pred.u8
    predicate_results: profiles/default/results.ivec
"#;
    std::fs::write(ds.join("dataset.yaml"), yaml).unwrap();

    let group = vectordata::TestDataGroup::load(ds.to_str().unwrap()).unwrap();
    let view = group.generic_view("default").unwrap();

    // Each facet has a different type
    assert_eq!(view.facet_element_type("base_vectors").unwrap(), ElementType::F32);
    assert_eq!(view.facet_element_type("metadata_content").unwrap(), ElementType::U8);
    assert_eq!(view.facet_element_type("metadata_predicates").unwrap(), ElementType::U8);
    assert_eq!(view.facet_element_type("predicate_results").unwrap(), ElementType::I32);

    // Access metadata as native u8
    let meta = view.open_facet_typed::<u8>("metadata_content").unwrap();
    assert_eq!(meta.get_native(0), 5);
    assert_eq!(meta.get_native(1), 10);

    // Access predicates as wider i32
    let pred = view.open_facet_typed::<i32>("metadata_predicates").unwrap();
    assert_eq!(pred.get_value(0).unwrap(), 5i32);

    // Access results as native i32
    let results = view.open_facet_typed::<i32>("predicate_results").unwrap();
    assert_eq!(results.dim(), 1);
    assert_eq!(results.get_record(0).unwrap(), vec![0i32]);
}
