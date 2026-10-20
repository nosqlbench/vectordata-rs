<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Accessing datasets from Rust

The `vectordata` crate gives you typed, named access to vector
search datasets. There is one prescribed entry path for everything
this tutorial covers — **catalog → profile → reader**:

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;
use vectordata::TestDataViewExt;

let catalog = Catalog::of(&CatalogSources::new().configure_default());
let view    = catalog.open_profile("my-dataset", "default")?;
let base    = view.base_vectors_as::<f32>()?;   // Arc<dyn VectorReader<f32>>
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

println!("distance: {:?}", view.distance_function()); // as written
println!("typed: {:?}", view.distance());              // Option<DistanceFunction>
println!("base_count: {:?}", view.base_count());
```

## 2. Read uniform vectors

Every facet opens at the element type you ask for, through
`TestDataViewExt` (import it): natively when the file holds that
type, widened when the conversion is lossless (a `bf16` or `u8` base
as `f32`), refused otherwise (an `f32` base as `u8`) — never a
reinterpretation of the bytes.

```rust
use vectordata::TestDataViewExt;

let base  = view.base_vectors_as::<f32>()?;        // Arc<dyn VectorReader<f32>>
let query = view.query_vectors_as::<f32>()?;
let gt    = view.neighbor_indices_as::<i32>()?;    // Arc<dyn VectorReader<i32>>
let dist  = view.neighbor_distances_as::<f32>()?;

println!("{} vectors, dim={}", base.count(), base.dim());

let v: Vec<f32> = base.get(42)?;
let neighbors: Vec<i32> = gt.get(0)?;

// What the file holds, and whether this reader converts
println!("{:?} read as {:?}", base.native_element_type(), base.conversion());
```

A quantized corpus reads as its codes — `base_vectors_as::<u8>()` —
and `quantized_base::<u8>()` opens the codes with their per-vector
scale, zero point and code sum as one thing (API reference §2.7).

## 3. Read variable-length vectors

Predicate results have a different number of matching ordinals per
predicate, so they're served as a `VvecReader`:

```rust
use vectordata::VvecReader;

let mi = view.metadata_results_as::<i32>()?;   // Arc<dyn VvecReader<i32>>
println!("{} records", mi.count());

let matching: Vec<i32> = mi.get(0)?;     // length varies per record
let dim: usize = mi.dim_at(0)?;
```

## 4. Read typed scalars and metadata

Scalar metadata (`.u8`, `.i32`, `.f32`, …) is a header-free column.
`facet_as::<T>(name)` opens it like any other facet; once the file is
resident, `as_slice()` is the whole column, borrowed from the mapping:

```rust
let meta = view.facet_as::<u8>("metadata_content")?;
let pred = view.facet_as::<u8>("metadata_predicates")?;

let labels  = meta.as_slice().expect("local and fetched");
let filters = pred.as_slice().expect("local and fetched");
for i in 0..5 {
    println!("base[{}] label={}, query[{}] filter={}", i, labels[i], i, filters[i]);
}

// Widening: open a u8 column as i32 — every value succeeds
let widened = view.facet_as::<i32>("metadata_content")?;
let v: i32 = widened.get(42)?[0];           // as_slice() is None: nothing to borrow at i32
```

Same-width cross-sign (`u8` ↔ `i8`) is allowed and checked per value
(`IoError::ValueOverflow` at the first that does not fit); narrowing,
and any conversion that could lose precision, is refused at open with
`IoError::ElementMismatch`. The table is in the API reference, §2.7.

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

## 6. Fetch for offline / zero-copy access

Remote facets are downloaded on demand, a chunk at a time, as reads
touch them. For a scan, fetch first: `fetch` plans every facet, skips
what is already resident, checks the cache has room, downloads in
parallel, and reports progress. Afterwards reads come from a
memory-mapped local cache file with no per-read overhead.

```rust
use vectordata::fetch::{FetchRequest, Silent, TextMeter};

// Everything the profile declares, quietly:
view.fetch(&FetchRequest::all(), &mut Silent)?;

// Or chosen facets, with the progress meter the CLI draws:
view.fetch(
    &FetchRequest::facets(["base_vectors", "query_vectors"]),
    &mut TextMeter::stderr("Fetch"),
)?;
```

To see the cost before anything moves, plan first:

```rust
let plan = view.plan_fetch(&FetchRequest::all(), &mut Silent)?;
println!("{} bytes to fetch", plan.bytes_to_fetch());
let report = plan.execute(&mut TextMeter::stderr("Fetch"))?;
```

For your own progress display, pass a closure:

```rust
use vectordata::fetch::FetchEvent;

view.fetch(&FetchRequest::all(), &mut |e: &FetchEvent<'_>| {
    if let FetchEvent::Progress { facet, bytes, total, .. } = e {
        eprintln!("  {facet}: {bytes}/{total} bytes");
    }
})?;
```

Per-facet cache state, without fetching:

```rust
use vectordata::CacheStats;

let storage = view.open_facet_storage("base_vectors")?;
if let Some(cs) = storage.cache_stats() {
    println!("{}/{} chunks ({} bytes)",
        cs.valid_chunks, cs.total_chunks, cs.content_size);
}

```

`cache_stats()` returns `None` for purely local datasets — there is no
cache to report on. `is_local()` is true for local datasets from the
start and for cached-remote once promoted.

Read the data through the facet readers (`base_vectors_as`,
`facet_as`), never by opening cache files yourself: a cache
file is pre-sized and sparse until complete, which chunks are valid is
recorded separately, and a sharded facet spans several files. The
readers handle all of that and are zero-copy once the facet is fetched
(`VectorReader::get_slice`). `FacetStorage::cache_path()` exists for
diagnostics — showing where a facet is cached — not for access.

## 6a. Mixing local and remote in one dataset

A `dataset.yaml` may declare some facets with relative paths and
others with absolute HTTP URLs. The view layer handles each facet
independently — the relative ones resolve against the dataset's
base location, the absolute ones pass through unchanged:

```yaml
# dataset.yaml on local disk
profiles:
  default:
    base_vectors: profiles/base/base_vectors.fvecs        # local
    query_vectors: https://hosted.example.com/q.fvecs     # remote
    metadata_content: profiles/base/metadata_content.u8  # local
```

```rust
let group = vectordata::TestDataGroup::load("./my-dataset/")?;
let view = group.profile("default").unwrap();

// base_vectors → Storage::Mmap (local, no cache, no copy)
let base = view.base_vectors_as::<f32>()?;

// query_vectors → Storage::Cached (downloads to cache_dir on first
// read, promotes to mmap once complete)
let query = view.query_vectors_as::<f32>()?;

// fetch is per-facet aware: local facets cost nothing, remote facets
// download.
view.fetch(&vectordata::fetch::FetchRequest::all(), &mut vectordata::fetch::Silent)?;
```

The same logic applies in reverse — a remote `dataset.yaml` can
point at facets in a different bucket via absolute URLs, and the
catalog layer resolves each facet via `Storage::open(source)` with
no special-casing.

## 7. Read in parallel

All readers are `Send + Sync` and the inner `Arc<Storage>` is
shared across clones, so a single `fetch` makes every
reader on every thread zero-copy:

```rust
use rayon::prelude::*;

let base = view.base_vectors_as::<f32>()?;
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
let base = XvecReader::<f32>::open_path(std::path::Path::new("base.fvecs"))?;
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

let r = open_vec::<f32>("base.fvecs")?;
let r = open_vec::<i32>("https://example.com/neighbors.ivecs")?;

let r = open_vvec::<i32>("metadata_indices.ivvecs")?;
```

The raw opens are native-only: `open_vec::<f32>("x.ivecs")` is an
`ElementMismatch`, not a reinterpretation. For a file whose type you
do not know, or at a type the file does not hold:

```rust
use vectordata::io::{open_vec_as, open_vec_native, ScalarReader};

let any = open_vec_native("metadata.u8")?;                // AnyVectorReader, at its own type
println!("{} of {}", any.count(), any.element_type());
let r = open_vec_as::<i64>("https://host/metadata.i32")?; // widened; the type comes from the extension
let r = ScalarReader::<u8>::open_path("metadata.u8")?;    // a column, natively, with as_slice()
```

---

## Where to go next

- [API reference](../../../docs/sysref/02-api.md) — full method-by-method reference.
- [Catalogs](../../../docs/sysref/03-catalogs.md) — publishing datasets and
  managing catalog sources.
- [Storage / transport factoring](../../../docs/design/storage_transport_factoring.md)
  — the design behind the unified open dispatch.
