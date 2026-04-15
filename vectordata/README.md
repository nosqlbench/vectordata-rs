# vectordata

Typed access to vector search benchmark datasets by name.

Find datasets in configured catalogs, discover profiles and facets,
and read vectors — all without constructing URLs or managing files.

## Quick Start

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

// Load catalogs from ~/.config/vectordata/catalogs.yaml
let catalog = Catalog::of(&CatalogSources::new().configure_default());

// Find and open a dataset by name
let group = catalog.open("my-dataset")?;

// Discover what's available
for name in group.profile_names() {
    let view = group.profile(&name).unwrap();
    for (facet, desc) in view.facet_manifest() {
        println!("  {}: {} ({})", name, facet, desc.source_type.as_deref().unwrap_or("?"));
    }
}

// Read vectors
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;          // Arc<dyn VectorReader<f32>>
let gt   = view.neighbor_indices()?;      // Arc<dyn VectorReader<i32>>
let mi   = view.metadata_indices()?;      // Arc<dyn VvecReader<i32>>

println!("{} vectors, dim={}", base.count(), base.dim());
let v: Vec<f32> = base.get(42)?;
```

## Typed Ordinal Access

For metadata and scalar facets, use typed readers with automatic
widening (e.g., read a `.u8` file as `i32`):

```rust
use vectordata::typed_access::TypedReader;

let view = catalog.open_profile("my-dataset", "default")?;
let meta: TypedReader<i32> = view.open_facet_typed("metadata_content")?;
let label: i32 = meta.get_value(42)?;
```

## Caching

Remote data is automatically cached locally with merkle verification.
Downloaded chunks are persisted to `~/.cache/vectordata/` (configurable
via `~/.config/vectordata/settings.yaml`). Fully cached files switch
to mmap for zero-copy reads.

Prebuffer an entire dataset for offline use:

```rust
let group = catalog.open("my-dataset")?;
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;
// Data is now local — subsequent reads are zero-copy mmap
```

## Features

- **Catalog discovery** — find datasets by name from configured sources
- **Profile + facet model** — datasets have named profiles, each with typed facets
- **Merkle-cached HTTP** — download on first access, verify, cache, switch to mmap
- **All element types** — f32, f64, f16, i32, i16, i8, u8, u16, u32, u64, i64
- **Variable-length records** — `.ivvec` with `IDXFOR__` offset index
- **Typed access** — `TypedReader<T>` with widening, cross-sign, native zero-copy
- **Thread-safe** — all readers are `Send + Sync`
- **knn_entries.yaml** — automatic fallback for jvector-compatible datasets

## Documentation

- [API Guide](../docs/sysref/02-api.md) — full consumer reference
- [Data Model](../docs/sysref/01-data-model.md) — file formats, facets, profiles
- [Catalogs](../docs/sysref/03-catalogs.md) — discovery, publishing, caching
