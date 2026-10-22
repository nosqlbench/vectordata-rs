// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The generic facet API (SRD element access, §14.3 step 4): a facet
//! read at its own type, widened, checked, or refused, by the one
//! conversion table — through the view, the extension trait, and the
//! erased open.

mod support;

use std::path::Path;
use std::sync::Arc;

use support::fixtures::{write_column, write_dataset_yaml, write_mref, write_vvec, write_xvec};
use support::testserver::TestServer;
use vectordata::io::{AnyVectorReader, IoError};
use vectordata::typed_access::{ElementConversion, ElementType, bf16, f16};
use vectordata::{Error, TestDataGroup, TestDataView, TestDataViewExt};

/// The error of a typed open that must fail.
fn refused<T>(r: Result<T, IoError>) -> IoError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("the open was expected to be refused"),
    }
}

/// The `default` profile of a dataset in `dir` declaring `facets`,
/// whose files the test has written.
fn dataset_in(dir: &tempfile::TempDir, facets: &[(&str, &str)]) -> Arc<dyn TestDataView> {
    write_dataset_yaml(dir.path(), "generic", facets);
    load(dir.path())
}

fn load(dir: &Path) -> Arc<dyn TestDataView> {
    TestDataGroup::load(dir.to_str().unwrap()).unwrap().profile("default").unwrap()
}

fn mismatch(err: &Error) -> Option<(ElementType, ElementType)> {
    match err {
        Error::VectorIo(IoError::ElementMismatch { file, requested, .. }) => Some((*file, *requested)),
        _ => None,
    }
}

fn overflow(err: &IoError) -> Option<usize> {
    match err {
        IoError::ValueOverflow { index, .. } => Some(*index),
        _ => None,
    }
}

// ── Guards ─────────────────────────────────────────────────────────────

/// An f16 base read as f32: widened by the table, with the right count
/// and the exact values. Before this step it was refused outright; on
/// 3.0.4 it was reinterpreted (count 2 of garbage).
#[test]
fn an_f16_facet_widens_to_f32_with_the_right_count() {
    let dir = tempfile::tempdir().unwrap();
    write_xvec::<f16>(&dir.path().join("base.mvec"), 3, 2, |i, d| f16::from_f32((i * 2 + d) as f32 * 0.5));
    write_dataset_yaml(dir.path(), "halves", &[("base_vectors", "base.mvec")]);
    let view = load(dir.path());

    let wide = view.base_vectors_as::<f32>().unwrap();
    assert_eq!((wide.count(), wide.dim()), (3, 2));
    assert_eq!(wide.get(2).unwrap(), vec![2.0, 2.5]);
    assert_eq!(wide.conversion(), ElementConversion::Widening);
    assert_eq!(wide.get_slice(0), None, "nothing of f32 exists to borrow");
    assert_eq!(wide.native_element_type(), Some(ElementType::F16));
    let mut all = [0f32; 6];
    assert_eq!(wide.read_into(0, &mut all).unwrap(), 3);
    assert_eq!(all, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
}

/// `facet_as::<f32>` refuses an index facet, not reinterpreting it as
/// 3.0.4's `facet()` did.
#[test]
fn facet_refuses_an_index_facet() {
    let dir = tempfile::tempdir().unwrap();
    write_xvec::<i32>(&dir.path().join("gt.ivec"), 2, 3, |i, d| (i * 3 + d) as i32);
    write_dataset_yaml(dir.path(), "ids", &[("neighbor_indices", "gt.ivec")]);
    let view = load(dir.path());
    let err = view.facet_as::<f32>("neighbor_indices").err().expect("i32 does not widen to f32");
    assert_eq!(mismatch(&err), Some((ElementType::I32, ElementType::F32)));
    assert_eq!(view.facet_as::<i64>("neighbor_indices").unwrap().get(1).unwrap(), vec![3, 4, 5]);
}

// ── Native ─────────────────────────────────────────────────────────────

#[test]
fn a_u8_facet_reads_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let codes: Vec<u8> = (0..8).map(|i| (i * 31 % 256) as u8).collect();
    write_column(&dir.path().join("codes.u8"), &codes);
    write_xvec::<u8>(&dir.path().join("base.bvecs"), 4, 3, |i, d| (i * 3 + d) as u8);
    write_dataset_yaml(dir.path(), "codes", &[("base_vectors", "base.bvecs"), ("metadata_content", "codes.u8")]);
    let view = load(dir.path());

    let column = view.facet_as::<u8>("metadata_content").unwrap();
    assert_eq!(column.conversion(), ElementConversion::Native);
    assert_eq!(column.native_element_type(), Some(ElementType::U8));
    assert_eq!((column.dim(), column.count()), (1, 8));
    assert_eq!(column.as_slice(), Some(&codes[..]), "a column borrows whole");
    for (i, c) in codes.iter().enumerate() {
        assert_eq!(column.get_slice(i), Some(std::slice::from_ref(c)));
    }
    let mut out = [0u8; 3];
    assert_eq!(column.read_into(5, &mut out).unwrap(), 3);
    assert_eq!(out, codes[5..8]);

    let base = view.base_vectors_as::<u8>().unwrap();
    assert_eq!(base.get_slice(3), Some(&[9u8, 10, 11][..]), "a record borrows from the mapping");
    assert_eq!(base.as_slice(), None, "xvec headers interleave the records");
}

#[test]
fn a_bf16_facet_reads_in_place_and_widens_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let value = |i: usize, d: usize| bf16::from_f32((i * 4 + d) as f32 * 0.1 - 1.0);
    write_xvec::<bf16>(&dir.path().join("base.bf16vecs"), 5, 4, value);
    write_dataset_yaml(dir.path(), "brains", &[("base_vectors", "base.bf16vecs")]);
    let view = load(dir.path());

    let native = view.base_vectors_as::<bf16>().unwrap();
    assert_eq!(native.conversion(), ElementConversion::Native);
    assert_eq!(native.get_slice(4), Some(&[value(4, 0), value(4, 1), value(4, 2), value(4, 3)][..]));

    let wide = view.base_vectors_as::<f32>().unwrap();
    assert_eq!(wide.conversion(), ElementConversion::Widening);
    for i in 0..5 {
        let want: Vec<f32> = (0..4).map(|d| value(i, d).to_f32()).collect();
        assert_eq!(wide.get(i).unwrap(), want, "record {i}");
    }
    let err = view.base_vectors_as::<f16>().err().expect("bf16 is not f16");
    assert_eq!(mismatch(&err), Some((ElementType::BF16, ElementType::F16)));
    let err = view.base_vectors_as::<u16>().err().expect("bf16 is not u16");
    assert_eq!(mismatch(&err), Some((ElementType::BF16, ElementType::U16)));
}

#[test]
fn an_i32_index_facet_reads_in_place_as_i32() {
    let dir = tempfile::tempdir().unwrap();
    write_xvec::<i32>(&dir.path().join("gt.ivecs"), 3, 2, |i, d| (i * 2 + d) as i32);
    write_dataset_yaml(dir.path(), "ids", &[("neighbor_indices", "gt.ivecs")]);
    let view = load(dir.path());
    let ids = view.neighbor_indices_as::<i32>().unwrap();
    assert_eq!(ids.conversion(), ElementConversion::Native);
    assert_eq!(ids.get_slice(2), Some(&[4, 5][..]));
}

// ── Converting ─────────────────────────────────────────────────────────

#[test]
fn a_u8_facet_widens_to_u16_i32_and_f32() {
    let dir = tempfile::tempdir().unwrap();
    write_xvec::<u8>(&dir.path().join("base.bvecs"), 3, 2, |i, d| (200 + i * 2 + d) as u8);
    write_dataset_yaml(dir.path(), "codes", &[("base_vectors", "base.bvecs")]);
    let view = load(dir.path());
    let as_u16 = view.base_vectors_as::<u16>().unwrap();
    let as_i32 = view.base_vectors_as::<i32>().unwrap();
    let as_f32 = view.base_vectors_as::<f32>().unwrap();
    assert_eq!(as_u16.get(2).unwrap(), vec![204, 205]);
    assert_eq!(as_i32.get(2).unwrap(), vec![204, 205]);
    assert_eq!(as_f32.get(2).unwrap(), vec![204.0, 205.0]);
    for conversion in [as_u16.conversion(), as_i32.conversion(), as_f32.conversion()] {
        assert_eq!(conversion, ElementConversion::Widening);
    }
    assert_eq!(as_f32.get_slice(0), None);
    let mut rows = [0i32; 6];
    assert_eq!(as_i32.read_into(0, &mut rows).unwrap(), 3);
    assert_eq!(rows, [200, 201, 202, 203, 204, 205]);
}

#[test]
fn a_u8_facet_as_i8_reports_the_first_overflow_by_ordinal() {
    let dir = tempfile::tempdir().unwrap();
    // Record 2 holds 200, which is not an i8.
    write_xvec::<u8>(&dir.path().join("base.bvecs"), 4, 2, |i, d| if i == 2 && d == 1 { 200 } else { (i + d) as u8 });
    write_dataset_yaml(dir.path(), "codes", &[("base_vectors", "base.bvecs")]);
    let view = load(dir.path());
    let signed = view.base_vectors_as::<i8>().unwrap();
    assert_eq!(signed.conversion(), ElementConversion::CrossSign);
    assert_eq!(signed.get(1).unwrap(), vec![1, 2]);
    assert_eq!(overflow(&signed.get(2).unwrap_err()), Some(2));
    let mut rows = [0i8; 8];
    assert_eq!(overflow(&signed.read_into(0, &mut rows).unwrap_err()), Some(2), "read_into names the facet ordinal");
    assert_eq!(signed.read_into(0, &mut rows[..4]).unwrap(), 2, "the records before it read fine");
}

#[test]
fn an_i32_index_facet_as_u32_fails_on_a_negative_sentinel() {
    let dir = tempfile::tempdir().unwrap();
    write_xvec::<i32>(&dir.path().join("gt.ivecs"), 3, 2, |i, d| if i == 1 && d == 1 { -1 } else { (i * 2 + d) as i32 });
    write_dataset_yaml(dir.path(), "ids", &[("neighbor_indices", "gt.ivecs")]);
    let view = load(dir.path());
    let unsigned = view.neighbor_indices_as::<u32>().unwrap();
    assert_eq!(unsigned.get(0).unwrap(), vec![0, 1]);
    assert_eq!(overflow(&unsigned.get(1).unwrap_err()), Some(1));
    let wide = view.neighbor_indices_as::<i64>().unwrap();
    assert_eq!(wide.get(1).unwrap(), vec![2, -1]);
}

// ── Refused ────────────────────────────────────────────────────────────

#[test]
fn an_f32_facet_as_u8_is_a_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    write_xvec::<f32>(&dir.path().join("base.fvecs"), 2, 2, |i, d| (i + d) as f32);
    write_dataset_yaml(dir.path(), "floats", &[("base_vectors", "base.fvecs")]);
    let view = load(dir.path());
    let err = view.base_vectors_as::<u8>().err().expect("a float never becomes an integer");
    assert_eq!(mismatch(&err), Some((ElementType::F32, ElementType::U8)));
    assert!(err.to_string().contains("no conversion between the two"), "{err}");
    let err = view.base_vectors_as::<f16>().err().expect("f32 does not narrow");
    assert_eq!(mismatch(&err), Some((ElementType::F32, ElementType::F16)));
}

// ── Columns and series ─────────────────────────────────────────────────

fn write_u32_column(path: &Path, values: std::ops::Range<u32>) {
    write_column(path, &values.collect::<Vec<_>>());
}

#[test]
fn a_scalar_series_reads_through_facet_as() {
    let dir = tempfile::tempdir().unwrap();
    write_u32_column(&dir.path().join("layout__0000.u32"), 0..100);
    write_u32_column(&dir.path().join("layout__0001.u32"), 100..175);
    std::fs::write(
        dir.path().join("dataset.yaml"),
        "format_version: 2\nname: scalars\nprofiles:\n  default:\n    metadata_layout:\n      \
         source: layout__NNNN.u32\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 175\n",
    )
    .unwrap();
    let view = load(dir.path());
    let column = view.facet_as::<u32>("metadata_layout").unwrap();
    assert_eq!((column.count(), column.dim()), (175, 1));
    assert_eq!(column.as_slice(), None, "a series has no one mapping to borrow");
    for o in [0usize, 1, 99, 100, 101, 174] {
        assert_eq!(column.get(o).unwrap(), vec![o as u32], "value at ordinal {o}");
    }
    let mut across = [0u32; 10];
    assert_eq!(column.read_into(95, &mut across).unwrap(), 10);
    assert_eq!(across, [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]);
    assert!(column.get(175).is_err());
    let wide = view.facet_as::<u64>("metadata_layout").unwrap();
    assert_eq!(wide.get(174).unwrap(), vec![174]);
}

#[test]
fn a_sliced_scalar_series_maps_ordinal_10() {
    let dir = tempfile::tempdir().unwrap();
    write_u32_column(&dir.path().join("corpus.u32"), 0..1000);
    std::fs::write(
        dir.path().join("dataset.yaml"),
        "format_version: 2\nname: sliced\nprofiles:\n  default:\n    metadata_layout:\n      source:\n        \
         - corpus.u32[0..10]=10\n        - corpus.u32[990..1000]=10\n      record_count: 20\n",
    )
    .unwrap();
    let view = load(dir.path());
    let column = view.facet_as::<u32>("metadata_layout").unwrap();
    assert_eq!(column.count(), 20);
    assert_eq!(column.get(9).unwrap(), vec![9]);
    assert_eq!(column.get(10).unwrap(), vec![990]);
    assert_eq!(column.get(19).unwrap(), vec![999]);
    assert!(column.get(20).is_err());
}

#[test]
fn a_window_over_a_column_is_a_sub_slice() {
    let dir = tempfile::tempdir().unwrap();
    let codes: Vec<u8> = (10..20).collect();
    write_column(&dir.path().join("codes.u8"), &codes);
    write_dataset_yaml(dir.path(), "window", &[("metadata_content", "codes.u8[2..6)")]);
    let view = load(dir.path());
    let column = view.facet_as::<u8>("metadata_content").unwrap();
    assert_eq!(column.count(), 4);
    assert_eq!(column.as_slice(), Some(&codes[2..6]));
    assert_eq!(column.get(0).unwrap(), vec![12]);
    assert_eq!(column.get_slice(3), Some(&[15u8][..]));
    assert_eq!(column.get_slice(4), None);
}

// ── Erased ─────────────────────────────────────────────────────────────

#[test]
fn open_facet_native_names_the_files_type() {
    let dir = tempfile::tempdir().unwrap();
    write_column(&dir.path().join("codes.u8"), &[1u8, 2, 3]);
    let view = dataset_in(&dir, &[("metadata_content", "codes.u8")]);
    let any = view.open_facet_native("metadata_content").unwrap();
    assert!(matches!(any, AnyVectorReader::U8(_)));
    assert_eq!((any.element_type(), any.count(), any.dim()), (ElementType::U8, 3, 1));
    let err = refused(view.open_facet_native("metadata_content").unwrap().into_typed::<i32>());
    assert!(matches!(err, IoError::ElementMismatch { file: ElementType::U8, requested: ElementType::I32, .. }));
    let typed = view.open_facet_native("metadata_content").unwrap().into_typed::<u8>().unwrap();
    assert_eq!(typed.as_slice(), Some(&[1u8, 2, 3][..]));
    let converted = view.open_facet_as_type("metadata_content", ElementType::I32).unwrap();
    assert!(matches!(converted, AnyVectorReader::I32(_)));
    assert_eq!(converted.into_typed::<i32>().unwrap().get(2).unwrap(), vec![3]);
}

/// A function over a `dyn` view calls the typed readers with only the
/// extension trait in scope.
fn first_record<T: vectordata::Element>(view: &Arc<dyn TestDataView>, facet: &str) -> Vec<T> {
    view.facet_as::<T>(facet).unwrap().get(0).unwrap()
}

#[test]
fn the_extension_trait_works_on_a_dyn_view() {
    let dir = tempfile::tempdir().unwrap();
    write_xvec::<f32>(&dir.path().join("base.fvecs"), 2, 2, |i, d| (i + d) as f32 + 0.5);
    write_dataset_yaml(dir.path(), "dyn", &[("base_vectors", "base.fvecs")]);
    let view: Arc<dyn TestDataView> = load(dir.path());
    assert_eq!(first_record::<f32>(&view, "base_vectors"), vec![0.5, 1.5]);
    assert_eq!(first_record::<f64>(&view, "base_vectors"), vec![0.5, 1.5]);
}

#[test]
fn open_vec_as_widens_a_raw_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("base.mvec");
    write_xvec::<f16>(&path, 2, 2, |i, d| f16::from_f32((i + d) as f32));
    let wide = vectordata::io::open_vec_as::<f32>(path.to_str().unwrap()).unwrap();
    assert_eq!(wide.get(1).unwrap(), vec![1.0, 2.0]);
    assert_eq!(wide.conversion(), ElementConversion::Widening);
    let native = vectordata::io::open_vec_native(path.to_str().unwrap()).unwrap();
    assert_eq!(native.element_type(), ElementType::F16);
    let column = dir.path().join("scale.f32");
    write_column(&column, &[0.5f32, 0.25]);
    let any = vectordata::io::open_vec_native(column.to_str().unwrap()).unwrap();
    assert_eq!((any.element_type(), any.dim(), any.count()), (ElementType::F32, 1, 2));
    assert_eq!(any.into_typed::<f32>().unwrap().as_slice(), Some(&[0.5f32, 0.25][..]));
    let err = refused(vectordata::io::open_vec_as::<u8>(path.to_str().unwrap()));
    assert!(matches!(err, IoError::ElementMismatch { file: ElementType::F16, requested: ElementType::U8, .. }));
}

// ── Remote ─────────────────────────────────────────────────────────────

static TEST_CACHE_DIR: std::sync::LazyLock<tempfile::TempDir> =
    std::sync::LazyLock::new(|| tempfile::tempdir().unwrap());

/// A column served over HTTP has no mapping to borrow until a fetch
/// brings it whole into the cache; its values read either way.
#[test]
fn a_remote_column_has_no_slice_until_fetched() {
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let dir = tempfile::tempdir().unwrap();
    let codes: Vec<u8> = (0..64).map(|i| (i * 7 % 251) as u8).collect();
    write_column(&dir.path().join("codes.u8"), &codes);
    write_mref(&dir.path().join("codes.u8"));
    write_dataset_yaml(dir.path(), "remote-codes", &[("metadata_content", "codes.u8")]);
    let server = TestServer::start(dir.path()).unwrap();
    let view = TestDataGroup::load(&server.base_url()).unwrap().profile("default").unwrap();

    let column = view.facet_as::<u8>("metadata_content").unwrap();
    assert_eq!(column.count(), 64);
    assert_eq!(column.as_slice(), None, "nothing is mapped before a fetch");
    assert_eq!(column.get(9).unwrap(), vec![codes[9]], "a value still reads, through the transport");
    assert_eq!(column.conversion(), ElementConversion::Native);

    let mut silent = vectordata::fetch::Silent;
    view.fetch(&vectordata::fetch::FetchRequest::all(), &mut silent).unwrap();
    let fetched = view.facet_as::<u8>("metadata_content").unwrap();
    assert_eq!(fetched.as_slice(), Some(&codes[..]), "fetched whole, the column maps");
}

// ── Every typed reader, and the variable-length one ────────────────────

#[test]
fn every_facet_reader_has_its_typed_form() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write_xvec::<f32>(&d.join("base.fvecs"), 4, 2, |i, dd| (i * 2 + dd) as f32);
    write_xvec::<f32>(&d.join("query.fvecs"), 2, 2, |i, dd| (i + dd) as f32 + 0.5);
    for (name, bump) in [("gt", 0), ("pre", 10), ("post", 20)] {
        write_xvec::<i32>(&d.join(format!("{name}.ivecs")), 2, 2, move |i, dd| (bump + i * 2 + dd) as i32);
        write_xvec::<f32>(&d.join(format!("{name}_d.fvecs")), 2, 2, move |i, dd| (bump + i * 2 + dd) as f32 * 0.5);
    }
    write_vvec::<i32>(&d.join("results.ivvecs"), &[vec![0, 1, 2], vec![3]]);
    let view = dataset_in(&dir, &[
        ("base_vectors", "base.fvecs"), ("query_vectors", "query.fvecs"),
        ("neighbor_indices", "gt.ivecs"), ("neighbor_distances", "gt_d.fvecs"),
        ("prefiltered_neighbor_indices", "pre.ivecs"), ("prefiltered_neighbor_distances", "pre_d.fvecs"),
        ("postfiltered_neighbor_indices", "post.ivecs"), ("postfiltered_neighbor_distances", "post_d.fvecs"),
        ("metadata_results", "results.ivvecs"),
    ]);
    assert_eq!(view.query_vectors_as::<f32>().unwrap().get(1).unwrap(), vec![1.5, 2.5]);
    assert_eq!(view.neighbor_distances_as::<f32>().unwrap().get(1).unwrap(), vec![1.0, 1.5]);
    assert_eq!(view.prefiltered_neighbor_indices_as::<i32>().unwrap().get(0).unwrap(), vec![10, 11]);
    assert_eq!(view.prefiltered_neighbor_distances_as::<f64>().unwrap().get(0).unwrap(), vec![5.0, 5.5]);
    assert_eq!(view.postfiltered_neighbor_indices_as::<i64>().unwrap().get(1).unwrap(), vec![22, 23]);
    assert_eq!(view.postfiltered_neighbor_distances_as::<f32>().unwrap().get(1).unwrap(), vec![11.0, 11.5]);

    let results = view.metadata_results_as::<i32>().unwrap();
    assert_eq!((results.count(), results.dim_at(0).unwrap()), (2, 3));
    assert_eq!(results.get(1).unwrap(), vec![3]);
    let err = refused(view.open_vvec_facet_native("predicate_results").unwrap().into_typed::<u32>());
    assert!(matches!(err, IoError::ElementMismatch { file: ElementType::I32, requested: ElementType::U32, .. }), "a vvec facet is never converted");
    let any = view.open_vvec_facet_native("metadata_indices").unwrap();
    assert_eq!((any.element_type(), any.count()), (ElementType::I32, 2));
    assert!(format!("{any:?}").contains("element: I32"));
}

// ── The readers themselves ─────────────────────────────────────────────

#[test]
fn the_column_reader_opens_only_columns_of_its_type() {
    use vectordata::io::ScalarReader;
    use vectordata::VectorReader;
    let dir = tempfile::tempdir().unwrap();
    let column = dir.path().join("ids.i32");
    write_column(&column, &[7i32, -1, 9]);
    let reader = ScalarReader::<i32>::open_path(&column).unwrap();
    assert_eq!(reader.count(), 3);
    assert_eq!(reader.get(1).unwrap(), vec![-1]);
    assert!(matches!(reader.get(3), Err(IoError::OutOfBounds(3))));
    assert!(reader.is_complete());
    reader.precache().unwrap();
    assert!(format!("{reader:?}").contains("element: I32"), "{reader:?}");

    let vectors = dir.path().join("base.bvecs");
    write_xvec::<u8>(&vectors, 1, 2, |_, dd| dd as u8);
    let err = refused(ScalarReader::<u8>::open_path(&vectors));
    assert!(err.to_string().contains("holds vectors of u8, not a column"), "{err}");

    let ragged = dir.path().join("short.i32");
    std::fs::write(&ragged, [0u8; 5]).unwrap();
    let err = refused(ScalarReader::<i32>::open(ragged.to_str().unwrap()));
    assert!(err.to_string().contains("not a whole number of 4-byte i32 elements"), "{err}");
}

#[test]
fn a_converting_reader_takes_only_a_converting_pair() {
    use vectordata::io::{ConvertingReader, ScalarReader};
    use vectordata::VectorReader;
    let dir = tempfile::tempdir().unwrap();
    let column = dir.path().join("codes.u8");
    write_column(&column, &[1u8, 2, 250]);
    let native: Arc<dyn vectordata::VectorReader<u8>> = Arc::new(ScalarReader::<u8>::open_path(&column).unwrap());

    let err = refused(ConvertingReader::<u8, u8>::new(native.clone()));
    assert!(err.to_string().contains("already holds this type"), "{err}");
    let scales = dir.path().join("scale.f32");
    write_column(&scales, &[0.5f32, 2.0]);
    let floats: Arc<dyn vectordata::VectorReader<f32>> = Arc::new(ScalarReader::<f32>::open_path(&scales).unwrap());
    let err = refused(ConvertingReader::<f32, u8>::new(floats));
    assert!(err.to_string().contains("no conversion between the two"), "{err}");

    let wide = ConvertingReader::<u8, i16>::new(native.clone()).unwrap();
    assert_eq!(wide.conversion(), ElementConversion::Widening);
    assert_eq!(wide.native_element_type(), Some(ElementType::U8));
    assert!(wide.is_complete());
    wide.precache().unwrap();
    assert_eq!(wide.as_slice(), None);
    assert!(format!("{wide:?}").contains("from: U8") && format!("{wide:?}").contains("to: I16"), "{wide:?}");
    let any = vectordata::io::open_vec_native(column.to_str().unwrap()).unwrap();
    assert!(format!("{any:?}").contains("element: U8"), "{any:?}");
}
