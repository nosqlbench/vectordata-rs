# vectordata

Consumer-facing Rust library for vector search datasets.

Load, read, and verify ANN benchmark datasets from local files or
remote catalogs with a single API call.

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;
use vectordata::io::{open_vec, open_vvec, VectorReader, VvecReader};

// Load from URL or local path
let group = TestDataGroup::load("https://example.com/datasets/sift1m/")?;
let view = group.profile("default").unwrap();

// Uniform vectors
let base = view.base_vectors()?;         // Arc<dyn VectorReader<f32>>
let gt = view.neighbor_indices()?;       // Arc<dyn VectorReader<i32>>

// Variable-length vectors
let mi = view.metadata_indices()?;       // Arc<dyn VvecReader<i32>>

// Direct file access (local or remote)
let reader = open_vec::<f32>("base.fvec")?;
let reader = open_vvec::<i32>("metadata_indices.ivvec")?;
```

## Features

- **Unified API** — `open_vec<T>()` and `open_vvec<T>()` handle local
  mmap and HTTP Range requests transparently
- **All element types** — f32, f64, f16, i32, i16, i8, u8, u16, u32, u64, i64
- **Variable-length records** — `.ivvec` files with `IDXFOR__` offset
  index for O(1) random access
- **Catalog discovery** — search configured catalogs by dataset name
- **Merkle integrity** — chunk-level SHA-256 verification on download
- **Prebuffering** — download and cache datasets for offline use
- **Thread-safe** — all readers are `Send + Sync`

## Documentation

- [API Guide](../docs/sysref/02-api.md) — full consumer reference
- [Data Model](../docs/sysref/01-data-model.md) — file formats, facets, dataset.yaml
- [Catalogs](../docs/sysref/03-catalogs.md) — discovery, publishing, caching
