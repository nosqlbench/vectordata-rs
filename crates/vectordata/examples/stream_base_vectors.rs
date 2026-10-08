// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Stream a profile's base vectors in order, a window at a time, with
//! the next window fetching in the background while the current one is
//! read.
//!
//! ```text
//! cargo run -p vectordata --example stream_base_vectors -- my-dataset:default 100000
//! ```
//!
//! The second argument is the window size in records. Reads that
//! overtake the background fetch are not wrong, only slower: they fetch
//! the chunk they need themselves, and the prefetch skips what is
//! already resident.

use vectordata::catalog::{Catalog, CatalogSources};
use vectordata::dataset::source::{DSInterval, DSWindow};
use vectordata::fetch::{FetchRequest, Silent};
use vectordata::WholeFacetFallback;

fn window(start: u64, end: u64) -> DSWindow {
    DSWindow(vec![DSInterval { min_incl: start, max_excl: end }])
}

fn main() -> vectordata::Result<()> {
    let mut args = std::env::args().skip(1);
    let spec = args.next().unwrap_or_else(|| "my-dataset:default".into());
    let step: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(100_000);

    let catalog = Catalog::of(&CatalogSources::new().configure_default());
    let view = catalog.open_spec(&spec)?.view()?;
    let base = view.base_vectors()?;
    let count = base.count() as u64;

    // The first window, fetched before reading starts.
    view.fetch(
        &FetchRequest::facets(["base_vectors"]).window(window(0, step.min(count))),
        &mut Silent,
    )?;

    let mut sum = 0f64;
    let mut start = 0;
    while start < count {
        let end = (start + step).min(count);
        // Warm the next window while this one is read.
        let next = (end < count).then(|| {
            view.prefetch_in_background(
                "base_vectors",
                &window(end, (end + step).min(count)),
                WholeFacetFallback::Refuse,
            )
        });
        for i in start..end {
            let v = base.get(i as usize)?;
            sum += v.iter().map(|x| *x as f64).sum::<f64>();
        }
        if let Some(handle) = next {
            handle?.join()?;
        }
        start = end;
    }
    println!("{count} vectors, element sum {sum:.3}");
    Ok(())
}
