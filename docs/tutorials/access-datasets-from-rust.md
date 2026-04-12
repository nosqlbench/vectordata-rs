<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Accessing Datasets from Rust

Load and read vector datasets using the `vectordata` crate.

## Setup

```toml
[dependencies]
vectordata = "0.17"
```

## Load a dataset

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

// From a local directory
let group = TestDataGroup::load("./my-dataset/")?;

// From an HTTP URL
let group = TestDataGroup::load("https://example.com/datasets/my-dataset/")?;

// Access a profile
let view = group.profile("default").unwrap();
```

## Read uniform vectors

```rust
let base = view.base_vectors()?;       // Arc<dyn VectorReader<f32>>
let query = view.query_vectors()?;
let gt = view.neighbor_indices()?;     // Arc<dyn VectorReader<i32>>

println!("{} vectors, dim={}", base.count(), base.dim());
let v: Vec<f32> = base.get(42)?;
let neighbors: Vec<i32> = gt.get(0)?;
```

## Read variable-length vectors

Predicate results have a different number of matching ordinals per
predicate:

```rust
let mi = view.metadata_indices()?;     // Arc<dyn VvecReader<i32>>
println!("{} records", mi.count());

let matching = mi.get(0)?;             // Vec<i32>, variable length
let dim = mi.dim_at(0)?;              // dimension of this record
```

## Read typed scalars

```rust
let gview = group.generic_view("default").unwrap();

let meta = gview.open_facet_typed::<u8>("metadata_content")?;
let pred = gview.open_facet_typed::<u8>("metadata_predicates")?;

for i in 0..10 {
    println!("base[{}] label={}, query[{}] filter={}",
        i, meta.get_native(i), i, pred.get_native(i));
}
```

## Open files directly (without a dataset)

```rust
use vectordata::io::{open_vec, open_vvec, VectorReader, VvecReader};

// Uniform vectors
let reader = open_vec::<f32>("base.fvec")?;
let reader = open_vec::<i32>("https://example.com/neighbors.ivec")?;

// Variable-length vectors
let reader = open_vvec::<i32>("metadata_indices.ivvec")?;
```

## Discover datasets from catalogs

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

let sources = CatalogSources::new().configure_default();
let catalog = Catalog::of(&sources);

for entry in catalog.datasets() {
    println!("{} (profiles: {})", entry.name, entry.profile_names().join(", "));
}

if let Some(entry) = catalog.find_exact("my-dataset") {
    println!("found: {}", entry.name);
}
```

## Parallel access

All readers are `Send + Sync`:

```rust
use rayon::prelude::*;

let base = view.base_vectors()?;
let norms: Vec<f64> = (0..base.count())
    .into_par_iter()
    .map(|i| {
        let v = base.get(i).unwrap();
        v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
    })
    .collect();
```

## Check dataset attributes

```rust
let dist = group.attribute("distance_function")
    .and_then(|v| v.as_str())
    .unwrap_or("unknown");

let zero_free = group.attribute("is_zero_vector_free")
    .and_then(|v| v.as_bool())
    .unwrap_or(false);
```
