// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Register catalogs a program supplies, beside (or instead of) the
//! configured ones, and open datasets from them by name.
//!
//! ```text
//! cargo run -p vectordata --example register_catalog
//! ```
//!
//! Two catalogs are built here, over a small dataset this example writes
//! to a temporary directory:
//!
//! - one from a dataset description and an entry built in code
//!   (`DatasetConfig::builder`, `CatalogEntry::builder`);
//! - one from catalog content the program holds, as it might ship one
//!   with `include_str!`, resolved against a base location.
//!
//! Both join a `CatalogSources` set with `with`, where they behave as a
//! configured catalog does: their datasets list, open, and report which
//! catalog they came from, and a name in two catalogs is ambiguous. Add
//! `.configure_default()` to the set to include the user's own
//! catalogs; `vectordata::explore::run_with_sources` opens the explorer
//! over the same set.

use std::path::Path;

use vectordata::catalog::{Catalog, CatalogBuilder, CatalogSources};
use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
use vectordata::{Location, TestDataViewExt};

/// A shipped catalog in the `knn_entries.yaml` shape: `<dataset>:<profile>`
/// keys, facet paths relative to the catalog's base.
const SHIPPED: &str = "\
listed:default:
  base: base.fvec
  query: query.fvec
  gt: gt.ivec
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    write_dataset(dir.path())?;
    let home = Location::from_path(dir.path());

    // A dataset described in code, and the catalog entry that lists it.
    let description = DatasetConfig::builder("demo")
        .distance_function("L2")
        .profile("default", |p| {
            p.facet(StandardFacet::BaseVectors, "base.fvec")
                .facet(StandardFacet::QueryVectors, "query.fvec")
                .facet(StandardFacet::NeighborIndices, "gt.ivec")
        })
        .build()?;
    let entry = CatalogEntry::builder("demo").home(&home).dataset(description).build()?;
    let built = CatalogBuilder::named("in-code").entry(entry).build();

    // Catalog content the program holds, read as a catalog file is.
    let shipped = CatalogBuilder::named("shipped").base(home.clone()).content(SHIPPED).build();

    let sources = CatalogSources::new().with(built).with(shipped);
    let catalog = Catalog::of(&sources);
    // Loading problems are values; a program decides what to show.
    for d in catalog.diagnostics() {
        eprintln!("{d}");
    }

    for name in ["demo", "listed"] {
        let entry = catalog.lookup(name)?;
        let view = catalog.open_profile(name, "default")?;
        let base = view.base_vectors_as::<f32>()?;
        let gt = view.neighbor_indices_as::<i32>()?;
        println!(
            "{name} (catalog '{}'): {} base vectors, dim {}; vector 1 = {:?}; nearest to query 0: {:?}",
            entry.catalog_name().unwrap_or("?"),
            base.count(),
            base.dim(),
            base.get(1)?,
            gt.get(0)?,
        );
    }
    Ok(())
}

/// Write a four-vector, two-dimensional dataset: base and query vectors
/// (`.fvec`) and each query's nearest base vectors (`.ivec`).
fn write_dataset(dir: &Path) -> std::io::Result<()> {
    let record = |values: &[[u8; 4]]| -> Vec<u8> {
        let mut out = (values.len() as i32).to_le_bytes().to_vec();
        values.iter().for_each(|v| out.extend(v));
        out
    };
    let vectors = |rows: &[[f32; 2]]| -> Vec<u8> {
        rows.iter().flat_map(|r| record(&[r[0].to_le_bytes(), r[1].to_le_bytes()])).collect()
    };
    std::fs::write(dir.join("base.fvec"), vectors(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))?;
    std::fs::write(dir.join("query.fvec"), vectors(&[[0.9, 0.1]]))?;
    std::fs::write(dir.join("gt.ivec"), record(&[1i32.to_le_bytes(), 0i32.to_le_bytes()]))
}
