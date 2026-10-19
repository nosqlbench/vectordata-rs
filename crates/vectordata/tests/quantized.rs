// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pre-quantized corpora (SRD element access, §14.3 step 5): the codes
//! and their three parameter columns resolve as one dataset, checked
//! against each other and against the scheme.

mod support;

use std::sync::Arc;

use support::fixtures::{write_column, write_quantized_profile};
use vectordata::dataset::{QuantGranularity, QuantScheme};
use vectordata::io::IoError;
use vectordata::{ElementType, Error, TestDataGroup, TestDataView, TestDataViewExt};

fn load(dir: &std::path::Path) -> Arc<dyn TestDataView> {
    TestDataGroup::load(dir.to_str().unwrap()).unwrap().profile("default").unwrap()
}

#[test]
fn a_quantized_profile_opens_as_one_base() {
    let dir = tempfile::tempdir().unwrap();
    let (codes, scale, zero_point, sum) = write_quantized_profile(dir.path(), 6, 4);
    let view = load(dir.path());

    let q = view.quantized_base::<u8>().unwrap();
    assert_eq!((q.codes.count(), q.codes.dim()), (6, 4));
    for (i, row) in codes.iter().enumerate() {
        assert_eq!(q.codes.get_slice(i), Some(&row[..]), "codes borrow from the mapping");
    }
    assert_eq!(q.scale.as_slice(), Some(&scale[..]), "the scale column is one slice");
    assert_eq!(q.zero_point.as_slice(), Some(&zero_point[..]));
    assert_eq!(q.sum.as_slice(), Some(&sum[..]));
    assert_eq!(q.scheme.scheme, QuantScheme::Affine);
    assert_eq!(q.scheme.granularity, QuantGranularity::PerVector);
    assert_eq!((q.scheme.codes, q.scheme.range), (ElementType::U8, [0, 255]));
    // The manifest lists the three columns as standard facets.
    let manifest = view.facet_manifest();
    for key in ["quant_scale", "quant_zero_point", "quant_sum"] {
        assert!(manifest.contains_key(key), "{key} is in the manifest");
    }
}

#[test]
fn quantized_base_at_the_wrong_code_type_is_a_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    write_quantized_profile(dir.path(), 3, 2);
    let view = load(dir.path());
    match view.quantized_base::<i8>() {
        Err(Error::VectorIo(IoError::ElementMismatch { file, requested, reason })) => {
            assert_eq!((file, requested), (ElementType::U8, ElementType::I8));
            assert!(reason.contains("quantization attribute names the type"), "{reason}");
        }
        other => panic!("expected a mismatch, got {:?}", other.map(|q| q.codes.count())),
    }
}

#[test]
fn a_short_quant_sum_is_a_cardinality_error() {
    let dir = tempfile::tempdir().unwrap();
    let (_, _, _, sum) = write_quantized_profile(dir.path(), 5, 3);
    write_column(&dir.path().join("quant_sum.i32"), &sum[..4]);
    let view = load(dir.path());
    match view.quantized_base::<u8>() {
        Err(Error::FacetCardinality { facet, against, expected, found }) => {
            assert_eq!((facet.as_str(), against.as_str(), expected, found), ("quant_sum", "base_vectors", 5, 4));
        }
        other => panic!("expected a cardinality error, got {:?}", other.map(|q| q.sum.count())),
    }
    let err = view.quantized_base::<u8>().err().unwrap();
    assert_eq!(
        err.to_string(),
        "facet 'quant_sum' holds 4 records where 5 were expected, one per record of 'base_vectors'"
    );
}

#[test]
fn without_the_attribute_quantized_base_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    write_quantized_profile(dir.path(), 3, 2);
    std::fs::write(
        dir.path().join("dataset.yaml"),
        "name: bare\nprofiles:\n  default:\n    base_vectors: base_vectors.bvecs\n    quant_scale: quant_scale.f32\n    \
         quant_zero_point: quant_zero_point.i32\n    quant_sum: quant_sum.i32\n",
    )
    .unwrap();
    let view = load(dir.path());
    assert!(view.quantization().is_none());
    let err = view.quantized_base::<u8>().err().unwrap();
    assert!(err.to_string().contains("no quantization attribute"), "{err}");
    // The columns still read on their own, as any facet does.
    assert_eq!(view.facet_as::<f32>("quant_scale").unwrap().count(), 3);
}

#[test]
fn the_profile_attribute_wins_over_the_datasets() {
    let dir = tempfile::tempdir().unwrap();
    write_quantized_profile(dir.path(), 3, 2);
    std::fs::write(
        dir.path().join("dataset.yaml"),
        "name: both\nattributes:\n  quantization:\n    scheme: affine\n    granularity: per-vector\n    codes: u8\n    \
         range: [0, 255]\nprofiles:\n  default:\n    attributes:\n      quantization:\n        scheme: symmetric\n        \
         granularity: per-vector\n        codes: u8\n        range: [0, 127]\n    base_vectors: base_vectors.bvecs\n    \
         quant_scale: quant_scale.f32\n    quant_zero_point: quant_zero_point.i32\n    quant_sum: quant_sum.i32\n",
    )
    .unwrap();
    let view = load(dir.path());
    let q = view.quantized_base::<u8>().unwrap();
    assert_eq!((q.scheme.scheme, q.scheme.range), (QuantScheme::Symmetric, [0, 127]));
}

#[test]
fn a_malformed_attribute_is_an_error_not_a_scheme() {
    let dir = tempfile::tempdir().unwrap();
    write_quantized_profile(dir.path(), 3, 2);
    std::fs::write(
        dir.path().join("dataset.yaml"),
        "name: bad\nattributes:\n  quantization:\n    scheme: affine\n    granularity: per-block\n    codes: u8\n    \
         range: [0, 255]\nprofiles:\n  default:\n    base_vectors: base_vectors.bvecs\n    quant_scale: quant_scale.f32\n    \
         quant_zero_point: quant_zero_point.i32\n    quant_sum: quant_sum.i32\n",
    )
    .unwrap();
    let view = load(dir.path());
    assert!(matches!(view.quantization(), Some(Err(ref e)) if e.contains("block size")));
    let err = view.quantized_base::<u8>().err().unwrap();
    assert!(err.to_string().contains("does not parse"), "{err}");
}
