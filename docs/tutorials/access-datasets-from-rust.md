<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Accessing datasets from Rust

The `vectordata` crate gives you typed, named access to vector
search datasets. There is one prescribed entry path for everything
this tutorial covers — **catalog → profile → reader**:

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

let catalog = Catalog::of(&CatalogSources::new().configure_default());
let view    = catalog.open_profile("my-dataset", "default")?;
let base    = view.base_vectors()?;       // Arc<dyn VectorReader<f32>>
let v: Vec<f32> = base.get(42)?;
```

You never construct URLs, never name a transport, and never decide
whether a remote dataset should be cached or read directly. The
crate picks the right backing storage (local mmap, merkle-cached
HTTP with auto-promotion to mmap, or direct HTTP RANGE) from the
catalog entry alone.

The rest of this tutorial walks the same pattern through every
common task. The bottom of the page covers low-level file access
for the rare cases where you don't have a catalog.

## Setup

```toml
[dependencies]
vectordata = "0.25"
```

## 1. Open a profile by name

The catalog returns an `Arc<dyn TestDataView>` — a thread-safe
handle you can pass around freely.

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;
use vectordata::TestDataView;
use std::sync::Arc;

let catalog = Catalog::of(&CatalogSources::new().configure_default());
let view: Arc<dyn TestDataView> = catalog.open_profile("my-dataset", "default")?;

println!("distance: {:?}", view.distance_function());
println!("base_count: {:?}", view.base_count());
```

## 2. Read uniform vectors

Standard facets return readers parameterised on the natural
element type:

```rust
let base  = view.base_vectors()?;        // Arc<dyn VectorReader<f32>>
let query = view.query_vectors()?;
let gt    = view.neighbor_indices()?;    // Arc<dyn VectorReader<i32>>
let dist  = view.neighbor_distances()?;

println!("{} vectors, dim={}", base.count(), base.dim());

let v: Vec<f32> = base.get(42)?;
let neighbors: Vec<i32> = gt.get(0)?;
```

## 3. Read variable-length vectors

Predicate results have a different number of matching ordinals per
predicate, so they're served as a `VvecReader`:

```rust
use vectordata::VvecReader;

let mi = view.metadata_indices()?;       // Arc<dyn VvecReader<i32>>
println!("{} records", mi.count());

let matching: Vec<i32> = mi.get(0)?;     // length varies per record
let dim: usize = mi.dim_at(0)?;
```

## 4. Read typed scalars and metadata

Scalar metadata (`.u8`, `.i32`, etc.) is opened via
`open_facet_typed`, which dispatches on the facet's native element
type and gives you a `TypedReader<T>` with widening conversions:

```rust
use vectordata::{open_facet_typed, TypedReader};

let meta: TypedReader<u8>  = open_facet_typed(&*view, "metadata_content")?;
let pred: TypedReader<u8>  = open_facet_typed(&*view, "metadata_predicates")?;

for i in 0..5 {
    println!("base[{}] label={}, query[{}] filter={}",
        i, meta.get_native(i), i, pred.get_native(i));
}

// Widening: open a u8 file as i32 — every value succeeds
let widened: TypedReader<i32> = open_facet_typed(&*view, "metadata_content")?;
let v: i32 = widened.get_value(42)?;
```

Same-width cross-sign (e.g., `u8` ↔ `i8`) is allowed and checked
per value; narrowing is rejected at open time.

## 5. Discover what's available

```rust
let group = catalog.open("my-dataset")?;

for profile in group.profile_names() {
    let pview = group.profile(&profile).unwrap();
    let manifest = pview.facet_manifest();
    println!("profile {profile}: {} facets", manifest.len());
    for (facet, desc) in &manifest {
        println!("  {facet} ({})", desc.source_type.as_deref().unwrap_or("?"));
    }
}
```

## 6. Prebuffer for offline / zero-copy access

Remote facets are downloaded on demand by the merkle-cache layer.
Call `prebuffer_all` to drive every facet to fully-resident state
up front; subsequent reads come from a memory-mapped local cache
file with no per-read overhead.

```rust
view.prebuffer_all()?;
// every facet is now mmap-backed and zero-copy
```

With per-facet progress:

```rust
view.prebuffer_all_with_progress(&mut |facet, p| {
    let pct = if p.total_chunks > 0 {
        100.0 * p.verified_chunks as f64 / p.total_chunks as f64
    } else { 0.0 };
    eprintln!("  {facet}: {:.0}% ({}/{} chunks, {:.1} MiB)",
        pct, p.verified_chunks, p.total_chunks,
        p.total_bytes as f64 / 1_048_576.0);
})?;
```

For a single facet:

```rust
use vectordata::CacheStats;

let storage = view.open_facet_storage("base_vectors")?;
storage.prebuffer()?;
assert!(storage.is_complete());

if let Some(cs): Option<CacheStats> = storage.cache_stats() {
    println!("{}/{} chunks ({} bytes)",
        cs.valid_chunks, cs.total_chunks, cs.content_size);
}
```

`open_facet_storage` returns `None` for purely local datasets
because there's nothing to cache; `cache_stats` returns `None` for
local storage and for direct-HTTP (no `.mref` published).

## 7. Read in parallel

All readers are `Send + Sync` and the inner `Arc<Storage>` is
shared across clones, so a single `prebuffer_all` makes every
reader on every thread zero-copy:

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

## 8. Hot-path zero-copy reads

The trait method `VectorReader::get_slice(index) -> Option<&[T]>`
is the safe form: it returns `None` when the underlying storage
isn't mmap-backed (e.g., direct HTTP, or cached-remote that hasn't
been prebuffered yet).

For the inner loop of a KNN scan or a normalisation pass, the
concrete `XvecReader<T>` exposes an inherent `get_slice(index) -> &[T]`
that **panics** if the storage is not mmap-backed and **does not
bounds-check** — the caller is responsible for staying within
`count()`. Reach for this only when you've already prebuffered (or
are reading a local file) and the call site is in a hot loop:

```rust
use vectordata::{XvecReader, VectorReader};

// Direct concrete-type access
let base = XvecReader::<f32>::open_path(std::path::Path::new("base.fvec"))?;
base.advise_sequential();                // madvise hint
for i in 0..base.count() {
    let v: &[f32] = base.get_slice(i);   // zero-copy, no bounds check
    // ... work ...
}
```

`XvecReader` also exposes `prefetch_range`, `release_range`, and
`prefetch_pages` for streaming-scan RSS control.

---

## Low-level file access (without a catalog)

When you don't have a catalog — testing, debugging, processing a
file dropped on disk — open files directly. Same dispatch story:
local paths mmap, URLs go through the cache when a `.mref` is
published, fall back to direct HTTP otherwise.

```rust
use vectordata::io::{open_vec, open_vvec, VectorReader, VvecReader};

let r = open_vec::<f32>("base.fvec")?;
let r = open_vec::<i32>("https://example.com/neighbors.ivec")?;

let r = open_vvec::<i32>("metadata_indices.ivvec")?;
```

For typed scalar access without a `TestDataView`:

```rust
use vectordata::typed_access::{ElementType, TypedReader};

let r = TypedReader::<u8>::open("metadata.u8")?;
let r = TypedReader::<i32>::open_url(
    url::Url::parse("https://example.com/metadata.u8")?,
    ElementType::U8,
)?;
let r = TypedReader::<i64>::open_auto("https://host/metadata.i32",
    ElementType::I32)?;
```

---

## Where to go next

- [API reference](../sysref/02-api.md) — full method-by-method reference.
- [Catalogs](../sysref/03-catalogs.md) — publishing datasets and
  managing catalog sources.
- [Storage / transport factoring](../design/storage_transport_factoring.md)
  — the design behind the unified open dispatch.
