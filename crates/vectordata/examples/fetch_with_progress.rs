// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fetch chosen facets of a profile into the local cache, with the same
//! progress meter `vectordata datasets precache` draws, then read from
//! the cache.
//!
//! ```text
//! cargo run -p vectordata --example fetch_with_progress -- my-dataset:default base_vectors query_vectors
//! ```
//!
//! With no facet names, every facet of the profile is fetched. The plan
//! is made and its cost printed before anything downloads.

use vectordata::catalog::{Catalog, CatalogSources};
use vectordata::fetch::{FetchRequest, TextMeter};

fn main() -> vectordata::Result<()> {
    let mut args = std::env::args().skip(1);
    let spec = args.next().unwrap_or_else(|| "my-dataset:default".into());
    let facets: Vec<String> = args.collect();

    let catalog = Catalog::of(&CatalogSources::new().configure_default());
    let view = catalog.open_spec(&spec)?.view()?;

    let request = if facets.is_empty() {
        FetchRequest::all()
    } else {
        FetchRequest::facets(facets)
    };
    let mut meter = TextMeter::stderr("Fetch");

    // Two steps, so the cost is known before anything moves.
    let plan = view.plan_fetch(&request, &mut meter)?;
    eprintln!("{} facet(s), {} bytes to fetch", plan.facets().len(), plan.bytes_to_fetch());
    let report = plan.execute(&mut meter)?;

    for row in &report.facets {
        println!("{}: {} bytes in {} range(s)", row.id, row.bytes_fetched, row.ranges_fetched);
    }
    // Everything fetched is now local: reads are zero-copy.
    if report.facet("base_vectors").is_some() {
        let base = view.base_vectors()?;
        println!("base_vectors: {} × {}", base.count(), base.dim());
    }
    Ok(())
}
