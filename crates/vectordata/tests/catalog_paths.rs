// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **The catalog contracts resolve at every path they had in 4.0.2.**
//! `CatalogSource` and `CatalogOrigin` moved to veks-apicore in 4.1.0;
//! `vectordata::catalog` re-exported them, but
//! `vectordata::catalog::source` imported them privately, so code
//! naming them there stopped compiling.

use vectordata::catalog::source::{CatalogOrigin, CatalogSource};
use vectordata::catalog::{CatalogBuilder, CatalogSources};

/// A program's built catalog, reached through the `source` paths.
fn origin_of(source: &dyn CatalogSource) -> CatalogOrigin {
    source.origin()
}

#[test]
fn the_source_paths_name_the_catalog_contracts() {
    let built = CatalogBuilder::named("mine").build();
    match origin_of(&built) {
        CatalogOrigin::Registered { label } => assert!(label.ends_with("mine"), "{label}"),
        other => panic!("a built catalog is registered: {other:?}"),
    }
    // The same items as the `catalog` paths, not copies of them.
    let _: &dyn vectordata::catalog::CatalogSource = &built;
    let _ = CatalogSources::new().with(built);
}
