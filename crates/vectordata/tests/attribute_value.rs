// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **The attribute value type is vectordata's to name.** A dataset's
//! attributes, an overlay's values and the descriptor builders are
//! `serde_yaml` values; `vectordata::AttributeValue` names the type and
//! `vectordata::serde_yaml` the crate at the version vectordata uses,
//! so a consumer that writes a helper over attributes needs no
//! `serde_yaml` dependency of its own.

use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
use vectordata::serde_yaml::Mapping;
use vectordata::{AttributeValue, Location};

/// A consumer's helper over a namespaced attribute.
fn namespaced(entry: &CatalogEntry, key: &str) -> Option<AttributeValue> {
    entry.attribute(key)
}

#[test]
fn an_attribute_is_read_as_vectordatas_attribute_value() {
    let mut nested = Mapping::new();
    nested.insert("centroids".into(), 64.into());
    let description = DatasetConfig::builder("demo")
        .attribute("fvec.ncentroids", 1024)
        .attribute("fvec.layout", AttributeValue::Mapping(nested.clone()))
        .profile("default", |p| p.facet(StandardFacet::BaseVectors, "base.fvec"))
        .build()
        .unwrap();
    let entry = CatalogEntry::builder("demo")
        .home(&Location::parse("https://example.com/datasets/demo/").unwrap())
        .dataset(description)
        .build()
        .unwrap();
    assert_eq!(namespaced(&entry, "fvec.ncentroids"), Some(AttributeValue::from(1024)));
    assert_eq!(namespaced(&entry, "fvec.layout"), Some(AttributeValue::Mapping(nested)));
    assert_eq!(namespaced(&entry, "fvec.missing"), None);
}
