// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Open a dataset from a spec a user typed, and describe what it holds.
//!
//! ```text
//! cargo run -p vectordata --example open_by_spec -- my-dataset:default
//! cargo run -p vectordata --example open_by_spec -- ./path/to/dataset
//! ```
//!
//! A spec is `<head>[:<selector>]`: a catalog name, a local directory or
//! `dataset.yaml`, or a URL, optionally followed by a profile selector.
//! `Catalog::open_spec` knows the grammar — URL ports and Windows drive
//! letters included — so the spec is never split by hand.

use vectordata::catalog::{Catalog, CatalogSources};

fn main() -> vectordata::Result<()> {
    let spec = std::env::args().nth(1).unwrap_or_else(|| "my-dataset:default".into());

    let catalog = Catalog::of(&CatalogSources::new().configure_default());
    // Loading problems are values; a program decides what to show.
    for d in catalog.diagnostics() {
        eprintln!("{d}");
    }

    let selection = catalog.open_spec(&spec)?;
    println!("{} — profiles {}", selection.dataset(), selection.profiles().join(", "));
    for (profile, view) in selection.views()? {
        let mut facets: Vec<_> = view.facet_manifest().into_iter().collect();
        facets.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, desc) in facets {
            println!(
                "  {profile}:{name} ({})",
                desc.source_type.as_deref().unwrap_or("?")
            );
        }
    }
    Ok(())
}
