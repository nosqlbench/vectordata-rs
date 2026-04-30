# vectordata

Typed access to vector search benchmark datasets by name.

Find datasets in configured catalogs, discover profiles and facets,
and read vectors — without constructing URLs, choosing transports,
or managing files. Datasets carry ground truth that has been
numerically cross-verified against FAISS and the Python `knn_utils`
reference (numpy + FAISS) — see the
[KNN engine conformance section](../docs/sysref/12-knn-utils-verification.md#127-cross-engine-conformance-testing).

## The prescribed pattern

This is the entry path the crate is designed around. Use this for
everything unless you genuinely have no catalog to start from.

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;
use vectordata::{open_facet_typed, TestDataView, TypedReader};

// 1. Resolve the catalog (~/.config/vectordata/catalogs.yaml)
let catalog = Catalog::of(&CatalogSources::new().configure_default());

// 2. Open a profile by name. Returns Arc<dyn TestDataView>.
let view = catalog.open_profile("my-dataset", "default")?;

// 3. Read facets through the trait — uniform, variable-length, typed
let base  = view.base_vectors()?;                              // Arc<dyn VectorReader<f32>>
let gt    = view.neighbor_indices()?;                          // Arc<dyn VectorReader<i32>>
let mi    = view.metadata_indices()?;                          // Arc<dyn VvecReader<i32>>
let meta: TypedReader<u8> = open_facet_typed(&*view, "metadata_content")?;

println!("{} vectors, dim={}", base.count(), base.dim());
let v: Vec<f32> = base.get(42)?;
let label: u8   = meta.get_native(42);

// 4. (Optional) prebuffer the whole profile so subsequent reads are
//    zero-copy mmap with no per-read overhead. No-op for local data
//    and for direct-HTTP data.
view.prebuffer_all()?;
```

The crate picks the right transport (local mmap, merkle-cached
HTTP with auto-promotion to mmap, or direct HTTP RANGE) for you
based on the catalog entry. There is no public type or function in
the crate that lets a caller bypass the cache or pick the slow
direct-HTTP path on a URL that has a published `.mref` — see the
[Storage / transport factoring](../docs/design/storage_transport_factoring.md)
design note.

For the full walk-through, see the
[Accessing datasets from Rust](../docs/tutorials/access-datasets-from-rust.md)
tutorial.

## Discover profiles and facets

```rust
let group = catalog.open("my-dataset")?;
for name in group.profile_names() {
    let view = group.profile(&name).unwrap();
    for (facet, desc) in view.facet_manifest() {
        println!("  {name}: {facet} ({})", desc.source_type.as_deref().unwrap_or("?"));
    }
}
```

## Prebuffer with progress

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

For per-facet cache state without prebuffering:

```rust
use vectordata::CacheStats;

let storage = view.open_facet_storage("base_vectors")?;
if let Some(cs): Option<CacheStats> = storage.cache_stats() {
    println!("{}/{} chunks", cs.valid_chunks, cs.total_chunks);
}
```

## Configuration

Cache location is set in `~/.config/vectordata/settings.yaml` (or
`$VECTORDATA_HOME/settings.yaml` for tests):

```yaml
cache_dir: /mnt/fast-storage/vectordata-cache
protect_settings: true
```

Or via the CLI:

```bash
veks datasets config set-cache /mnt/fast-storage/vectordata-cache
veks datasets config show
```

Resolution order:
1. Per-call override (e.g. `--cache-dir` CLI flag).
2. `cache_dir:` from settings.yaml.
3. `$HOME/.cache/vectordata/` fallback.

## Direct file access (no catalog)

Reach for these only when you don't have a catalog (tests,
debugging, ad-hoc tooling). The same auto-dispatch applies — local
paths mmap; URLs go through the cache when `.mref` is published.

```rust
use vectordata::io::{open_vec, open_vvec, VectorReader, VvecReader};

let r = open_vec::<f32>("base_vectors.fvec")?;
let r = open_vec::<f32>("https://example.com/dataset/base.fvec")?;
let r = open_vvec::<i32>("metadata_indices.ivvec")?;
```

## Features

- **Catalog discovery** — find datasets by name from configured sources.
- **Profile + facet model** — datasets have named profiles, each
  with typed facets (uniform vectors, variable-length records,
  scalar metadata, predicate result lists).
- **Single-source-of-truth transport dispatch** — `Storage` chooses
  local mmap vs merkle-cached vs direct HTTP from the source string;
  consumers never see transport types.
- **Merkle-cached HTTP** — download on first access, verify against
  SHA-256 chunk hashes, persist to disk, switch to mmap once
  complete.
- **All element types** — `f32`, `f64`, `f16`, `i32`, `i16`, `i8`,
  `u8`, `u16`, `u32`, `u64`, `i64`.
- **Variable-length records** — `.ivvec` etc. with `IDXFOR__`
  offset index, fetched as a sibling for remote datasets.
- **Typed access** — `TypedReader<T>` with widening, cross-sign
  checks, and zero-copy native reads.
- **Thread-safe** — every reader is `Send + Sync`; cloning the
  outer `Arc` shares the promoted-mmap state across threads.
- **knn_entries.yaml fallback** — automatic for jvector-compatible
  datasets without a `dataset.yaml`.

## Documentation

- [API reference](../docs/sysref/02-api.md) — full consumer reference.
- [Tutorial: Accessing datasets from Rust](../docs/tutorials/access-datasets-from-rust.md)
  — end-to-end walk-through of the prescribed pattern.
- [Data Model](../docs/sysref/01-data-model.md) — file formats,
  facets, profiles.
- [Catalogs](../docs/sysref/03-catalogs.md) — discovery, publishing,
  caching.
- [Storage / transport factoring](../docs/design/storage_transport_factoring.md)
  — the design behind the unified-dispatch reader API.
