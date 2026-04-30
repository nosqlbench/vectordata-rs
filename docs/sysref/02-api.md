# 2. API

The `vectordata` crate provides typed, unified access to vector
datasets regardless of storage location (local file or HTTP) and
record structure (uniform or variable-length). This document is the
definitive reference for external consumers.

The transport choice — local mmap, merkle-cached HTTP with
auto-promotion to mmap, or direct HTTP RANGE — is chosen for you
based on the source string. There is no public type or function in
the crate that lets a caller bypass the cache or pick the slow
direct-HTTP path on a URL that has a published `.mref`. See
[Storage / transport factoring](../design/storage_transport_factoring.md)
for the underlying design.

---

## 2.1 Quick Start

Add the dependency:

```toml
[dependencies]
vectordata = "0.25"
```

### Find and use a dataset by name

The primary access path: catalog → dataset → profile → facet →
vectors. No URLs or paths — just names:

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

// Load configured catalogs (~/.config/vectordata/catalogs.yaml)
let catalog = Catalog::of(&CatalogSources::new().configure_default());

// Open a dataset by name → get vectors in two calls
let group = catalog.open("my-dataset")?;
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;
println!("{} vectors, dim={}", base.count(), base.dim());
let v: Vec<f32> = base.get(42)?;

// Or even shorter — open a profile directly
let view = catalog.open_profile("my-dataset", "default")?;
let gt = view.neighbor_indices()?;
```

### Discover available profiles and facets

```rust
let group = catalog.open("my-dataset")?;
for name in group.profile_names() {
    let view = group.profile(&name).unwrap();
    let manifest = view.facet_manifest();
    for (facet_name, desc) in &manifest {
        println!("  {} ({})", facet_name,
            desc.source_type.as_deref().unwrap_or("?"));
    }
}
```

### Low-level file access

For direct file access without catalogs — typically only useful for
testing, debugging, or building tools on top of the library:

```rust
use vectordata::io::{open_vec, open_vvec, VectorReader, VvecReader};

// Uniform vectors — local or remote, same call
let reader = open_vec::<f32>("base_vectors.fvec")?;
let reader = open_vec::<f32>("https://example.com/dataset/base.fvec")?;
println!("{} vectors, dim={}", reader.count(), reader.dim());
let v: Vec<f32> = reader.get(42)?;

// Variable-length vectors (vvec)
let reader = open_vvec::<i32>("metadata_indices.ivvec")?;
println!("{} records", reader.count());
let record: Vec<i32> = reader.get(0)?;
let dim: usize = reader.dim_at(0)?;
```

Load a dataset by URL or path:

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://example.com/dataset/")?;
let view = group.profile("default").unwrap();

let base = view.base_vectors()?;           // Arc<dyn VectorReader<f32>>
let gt   = view.neighbor_indices()?;       // Arc<dyn VectorReader<i32>>
let mi   = view.metadata_indices()?;       // Arc<dyn VvecReader<i32>>
```

`TestDataGroup::load` falls back to `knn_entries.yaml`
(jvector-compatible format) when `dataset.yaml` is not present.

---

## 2.2 Catalog-Based Dataset Discovery

### Catalog configuration

Catalogs are configured in `~/.config/vectordata/catalogs.yaml`:

```yaml
# Each entry is an HTTP URL or local path pointing to a directory
# containing a catalog.json (which indexes all datasets under it)
- https://vectordata-datasets.s3.amazonaws.com/production/
- https://internal-bucket.s3.us-east-1.amazonaws.com/testing/
- /mnt/data/local-datasets/
```

Manage catalogs via the CLI:

```bash
veks datasets config add-catalog https://example.com/datasets/
veks datasets config list-catalogs
veks datasets config remove-catalog 2
```

### Discovering datasets

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

let sources = CatalogSources::new().configure_default();
let catalog = Catalog::of(&sources);

// List all available datasets
for entry in catalog.datasets() {
    println!("{} (profiles: {})",
        entry.name,
        entry.profile_names().join(", "));
}

// Find by exact name (case-insensitive)
if let Some(entry) = catalog.find_exact("my-dataset") {
    println!("found: {}, path: {}", entry.name, entry.path);
}

// Find by glob pattern
let matches = catalog.match_glob("my-vectors*");
for entry in matches {
    println!("  {}", entry.name);
}
```

### Adding catalogs programmatically

```rust
let sources = CatalogSources::new()
    .add_catalogs(&[
        "https://my-bucket.s3.amazonaws.com/datasets/".into(),
        "/local/path/to/datasets/".into(),
    ]);
let catalog = Catalog::of(&sources);
```

---

## 2.3 File Extension Scheme

See [SRD §22](22-vector-file-extensions.md) for the full
specification.

### Summary

| Suffix | Structure | Example |
|--------|-----------|---------|
| `.<type>` | Scalar (flat packed, no header) | `.u8`, `.i32`, `.f64` |
| `.<type>vec` | Uniform vector (fixed dimension per record) | `.fvec`, `.ivec`, `.u8vec` |
| `.<type>vvec` | Variable-length vector (per-record dimension) | `.ivvec`, `.fvvec`, `.u8vvec` |

Legacy aliases: `.fvec`=`.f32vec`, `.ivec`=`.i32vec`,
`.bvec`=`.u8vec`, `.svec`=`.i16vec`, `.mvec`=`.f16vec`,
`.dvec`=`.f64vec`.

### Record layout

**Uniform (`vec`)** — `[dim:i32 | elem₀ | elem₁ | … | elem_{dim-1}]`
— all records share the same dimension. Random access by stride
arithmetic.

**Variable (`vvec`)** — same per-record layout, but each record may
have a different dimension. Random access requires an offset index
file (`IDXFOR__<name>.<i32|i64>`), built automatically on first
local open and fetched as a sibling URL on remote open.

---

## 2.4 Unified Open Functions

### `open_vec<T>(path_or_url) → Box<dyn VectorReader<T>>`

Opens a uniform vector file for typed random access. Local paths
mmap; URLs go through the merkle-cached path when a `.mref` is
published, falling back silently to direct HTTP RANGE otherwise.

```rust
use vectordata::io::{open_vec, VectorReader};

let r = open_vec::<f32>("data/base.fvec")?;             // local mmap
let r = open_vec::<i32>("https://host/neighbors.ivec")?; // cached or direct HTTP

println!("count={}, dim={}", r.count(), r.dim());
let vec: Vec<f32> = r.get(0)?;
```

**Supported types:** `f32`, `f64`, `half::f16`, `i32`, `i16`, `i8`,
`u8`, `u16`, `u32`, `u64`, `i64`.

The type parameter `T` must match the file's element width. A
mismatch (e.g., `open_vec::<f32>("data.dvec")` where `.dvec` is
8-byte f64) returns an error at open time.

### `open_vvec<T>(path_or_url) → Box<dyn VvecReader<T>>`

Opens a variable-length vector file. Requires a companion
`IDXFOR__<name>.<i32|i64>` offset index file:

- For local files, built and persisted on first open.
- For remote files, fetched via HTTP from the same URL prefix; if
  absent, the reader walks the file via the channel to rebuild the
  index (slow first time, correct).

```rust
use vectordata::io::{open_vvec, VvecReader};

let r = open_vvec::<i32>("metadata_indices.ivvec")?;
println!("{} records", r.count());
let record: Vec<i32> = r.get(42)?;
let dim: usize = r.dim_at(42)?;
```

---

## 2.5 Traits and concrete types

### `VectorReader<T>` — uniform vectors

```rust
pub trait VectorReader<T>: Send + Sync {
    fn dim(&self) -> usize;
    fn count(&self) -> usize;
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;

    // bounds-checked zero-copy slice; None when storage is not mmap-backed
    fn get_slice(&self, index: usize) -> Option<&[T]> { None }

    // drive underlying storage to fully resident; no-op for local/direct-HTTP
    fn prebuffer(&self) -> std::io::Result<()> { Ok(()) }

    // true for local; true for cached once every chunk is verified; false for direct-HTTP
    fn is_complete(&self) -> bool { true }
}
```

The canonical concrete implementation is `XvecReader<T>`. It also
exposes hot-path inherent methods that the `dyn VectorReader<T>`
surface can't: an unchecked `get_slice(index) -> &[T]` (panics if
the storage is not mmap-backed; intended for KNN inner loops), plus
`advise_sequential` / `advise_random`, `prefetch_range`,
`release_range`, and `prefetch_pages` (madvise hints; no-op when
not mmap-backed).

### `VvecReader<T>` — variable-length vectors

```rust
pub trait VvecReader<T: VvecElement>: Send + Sync {
    fn count(&self) -> usize;
    fn dim_at(&self, index: usize) -> Result<usize, IoError>;
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError>;
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;        // default impl
    fn prebuffer(&self) -> std::io::Result<()>;
    fn is_complete(&self) -> bool;
}
```

The canonical concrete implementation is `IndexedVvecReader<T>`.
`get_raw(index) -> Option<&[u8]>` provides a zero-copy slice when
the storage is mmap-backed.

### `VvecElement` — byte decoding

```rust
pub trait VvecElement: Copy + Send + Sync + 'static {
    const ELEM_SIZE: usize;
    fn from_le_bytes(bytes: &[u8]) -> Self;
}
```

Implemented for: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`,
`i64`, `f32`, `f64`, `half::f16`.

---

## 2.6 Dataset Access via TestDataGroup

### Loading a dataset

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

// From a local directory containing dataset.yaml
let group = TestDataGroup::load("./my-dataset/")?;

// From an HTTP URL
let group = TestDataGroup::load("https://host/datasets/my-dataset/")?;

// Top-level dataset attributes
let dist = group.attribute("distance_function");  // Option<&Value>
```

### Profile access

```rust
// Trait object — the universal handle. Use this for almost everything.
let view: Arc<dyn TestDataView> = group.profile("default").unwrap();

// Concrete type — only needed for the typed open_facet_typed::<T> method
// when you don't want to use the free function. See §2.7.
let gview: GenericTestDataView = group.generic_view("default").unwrap();
```

### Source resolution

A facet's source string in `dataset.yaml` may be:

- A **relative path or URL fragment** — joined onto the dataset's
  base location (the directory of `dataset.yaml` for local datasets,
  the URL prefix for remote ones).
- An **absolute HTTP URL** (`http://…` or `https://…`) — passed
  through unchanged regardless of where the `dataset.yaml` lives.
  This means a local `dataset.yaml` can declare facets hosted on
  a remote server, and a remote `dataset.yaml` can pull facets
  from a different bucket — the dispatch is per-facet.
- An **absolute local path** — used as-is on local filesystems.

For each facet, the resolved source string is then handed to
`Storage::open(source)`, which picks the transport variant
(local mmap, merkle-cached, direct HTTP) without further input from
the caller.

### Standard facet methods

All methods on `TestDataView` return readers that work identically
for local and remote datasets:

```rust
// Uniform vector facets → Arc<dyn VectorReader<T>>
let base:  Arc<dyn VectorReader<f32>> = view.base_vectors()?;
let query: Arc<dyn VectorReader<f32>> = view.query_vectors()?;
let gt:    Arc<dyn VectorReader<i32>> = view.neighbor_indices()?;
let dist:  Arc<dyn VectorReader<f32>> = view.neighbor_distances()?;
let fki:   Arc<dyn VectorReader<i32>> = view.filtered_neighbor_indices()?;
let fkd:   Arc<dyn VectorReader<f32>> = view.filtered_neighbor_distances()?;

// Variable-length facet → Arc<dyn VvecReader<i32>>
let mi: Arc<dyn VvecReader<i32>> = view.metadata_indices()?;

// Metadata configuration (declared facets, no data access)
let meta_cfg: Option<&FacetConfig> = view.metadata_content();
let pred_cfg: Option<&FacetConfig> = view.metadata_predicates();

// Facet discovery
let manifest: HashMap<String, FacetDescriptor> = view.facet_manifest();

// Element type interrogation
let etype = view.facet_element_type("metadata_content")?;  // ElementType::U8

// Resolved source path/URL for any declared facet
let src: Option<String> = view.facet_source("metadata_content");

// Prebuffer / cache control
view.prebuffer_all()?;
let storage = view.open_facet_storage("base_vectors")?;    // FacetStorage
```

---

## 2.7 Typed scalar access

Metadata (M) and predicates (P) are scalar files — flat-packed
arrays with no per-record header. Read them as type-checked values
via `TypedReader<T>`, with widening conversions and runtime
overflow checks.

### From a `TestDataView` (catalog or dyn handle)

The free function `open_facet_typed::<T>` works against any
`&dyn TestDataView`:

```rust
use vectordata::{open_facet_typed, TypedReader};

let view = catalog.open_profile("my-dataset", "default")?;

// Open with native type — zero-copy when storage is mmap-backed
let r: TypedReader<u8> = open_facet_typed(&*view, "metadata_content")?;
let val: u8 = r.get_native(42);

// Open with a wider type — widening always succeeds
let r: TypedReader<i32> = open_facet_typed(&*view, "metadata_content")?;
let val: i32 = r.get_value(42)?;
```

### From a known-concrete `GenericTestDataView`

The same call exists as a method on `GenericTestDataView`:

```rust
let gview = group.generic_view("default").unwrap();
let r: TypedReader<u8> = gview.open_facet_typed::<u8>("metadata_content")?;
```

### Direct file open

```rust
use vectordata::typed_access::{ElementType, TypedReader};

// Local path, native type from extension
let r = TypedReader::<u8>::open("metadata.u8")?;

// Remote URL, native type explicit (cache-first; falls back to direct HTTP)
let r = TypedReader::<i32>::open_url(
    url::Url::parse("https://host/metadata.u8")?,
    ElementType::U8,
)?;

// Path-or-URL string, dispatched automatically
let r = TypedReader::<i64>::open_auto("https://host/metadata.i32",
    ElementType::I32)?;
```

### Width compatibility

| target T | native type | result |
|---|---|---|
| same width, same sign | exact match | zero-copy `get_native` works |
| same width, cross sign | e.g. `u8` ↔ `i8` | checked per value via `get_value`, fails on overflow |
| wider | e.g. `i32` from `u8` | widening; `get_value` always succeeds |
| narrower | e.g. `u8` from `i32` | rejected at open time (`Narrowing` error) |

---

## 2.8 Facet Reference

| Facet code | YAML key | Trait method | Reader type | Typical format |
|-----------|----------|-------------|-------------|---------------|
| B | `base_vectors` | `base_vectors()` | `VectorReader<f32>` | `.fvec` |
| Q | `query_vectors` | `query_vectors()` | `VectorReader<f32>` | `.fvec` |
| G | `neighbor_indices` | `neighbor_indices()` | `VectorReader<i32>` | `.ivec` |
| D | `neighbor_distances` | `neighbor_distances()` | `VectorReader<f32>` | `.fvec` |
| M | `metadata_content` | `metadata_content()` | config + `open_facet_typed` | `.u8`, `.slab` |
| P | `metadata_predicates` | `metadata_predicates()` | config + `open_facet_typed` | `.u8`, `.slab` |
| R | `metadata_indices` | `metadata_indices()` | `VvecReader<i32>` | `.ivvec` |
| F (indices) | `filtered_neighbor_indices` | `filtered_neighbor_indices()` | `VectorReader<i32>` | `.ivec` |
| F (distances) | `filtered_neighbor_distances` | `filtered_neighbor_distances()` | `VectorReader<f32>` | `.fvec` |

For M and P facets, use `open_facet_typed::<T>(view, name)` for
typed data access (see §2.7).

---

## 2.9 Dataset YAML profile schema

A `dataset.yaml` profile section maps canonical facet names to file
paths. All paths are relative to the dataset directory:

```yaml
profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
    neighbor_indices: profiles/default/neighbor_indices.ivec
    neighbor_distances: profiles/default/neighbor_distances.fvec
    metadata_content: profiles/base/metadata_content.u8
    metadata_predicates: profiles/base/predicates.u8
    metadata_indices: profiles/default/metadata_indices.ivvec
    filtered_neighbor_indices: profiles/default/filtered_neighbor_indices.ivec
    filtered_neighbor_distances: profiles/default/filtered_neighbor_distances.fvec
```

The `metadata_indices` key maps to the `predicate_results` field
internally (serde alias).

### Partition profiles

Partition profiles are marked with `partition: true`. They have
independent base vectors (not windowed from default) and do NOT
inherit views from the default profile:

```yaml
  label-0:
    maxk: 100
    base_count: 82993
    partition: true
    base_vectors: profiles/label-0/base_vectors.fvec
    query_vectors: profiles/label-0/query_vectors.fvec
    neighbor_indices: profiles/label-0/neighbor_indices.ivec
    neighbor_distances: profiles/label-0/neighbor_distances.fvec
```

### `knn_entries.yaml` fallback

When `dataset.yaml` is not found, `TestDataGroup::load` falls back
to `knn_entries.yaml` (jvector-compatible format):

```yaml
_defaults:
  base_url: https://example.com/data

"my-dataset:default":
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  gt:    profiles/base/neighbor_indices.ivec
```

The `knn_entries` module can also be used directly:

```rust
use vectordata::knn_entries::KnnEntries;

let entries = KnnEntries::load("knn_entries.yaml")?;
println!("datasets: {:?}", entries.dataset_names());
let config = entries.to_config();  // → DatasetConfig
```

---

## 2.10 Offset index files

Variable-length vector files (`.ivvec`, `.fvvec`, etc.) require a
companion offset index for random access:

```
data/metadata_indices.ivvec              # variable-length data
data/IDXFOR__metadata_indices.ivvec.i32  # offset index (< 2 GB data)
data/IDXFOR__metadata_indices.ivvec.i64  # offset index (≥ 2 GB data)
```

The index is a flat-packed array of byte offsets (one per record).
It is created automatically:

- On first local open by `IndexedVvecReader::open` /
  `IndexedVvecReader::open_path`.
- By the `generate vvec-index` pipeline step before publishing.
- By `evaluate-predicates` immediately after writing vvec output.

For remote access, the reader fetches the sidecar from the same URL
prefix. If the sidecar is absent, the reader walks the file
through the storage layer to rebuild the index (slow first time but
correct; the rebuilt index is not persisted to the remote source).

---

## 2.11 Error handling

```rust
use vectordata::io::IoError;

match open_vec::<f32>("data.fvec") {
    Ok(reader) => { /* use reader */ }
    Err(IoError::Io(e)) => eprintln!("I/O error: {e}"),
    Err(IoError::Http(e)) => eprintln!("HTTP error: {e}"),
    Err(IoError::InvalidFormat(msg)) => eprintln!("bad format: {msg}"),
    Err(IoError::OutOfBounds(idx)) => eprintln!("index {idx} out of range"),
    Err(IoError::VariableLengthRecords(msg)) => {
        // Wrong shape — file has variable-length records; use open_vvec
        eprintln!("use open_vvec for this file: {msg}");
    }
}
```

`VariableLengthRecords` is returned by `open_vec` when the file
turns out to have variable-length records (its size is not a
multiple of the implied stride). The caller should switch to
`open_vvec`.

---

## 2.12 Prebuffering and caching

### Cache location

Remote downloads are cached under `vectordata::settings::cache_dir()`,
the single source of truth for cache resolution shared with the
`veks-pipeline` crate. Resolution order:

1. `--cache-dir` CLI flag (per-command override).
2. `cache_dir:` entry in `~/.config/vectordata/settings.yaml`
   (or `$VECTORDATA_HOME/settings.yaml` for tests).

If neither is set, every API that needs the cache returns
`vectordata::settings::SettingsError::NotConfigured`. Print the
error directly — its `Display` impl includes the `veks` CLI
command and the manual `mkdir`+`echo` sequence the user can paste
to fix it. There is no silent fallback to `$HOME/.cache/vectordata/`.

Configure via the CLI:

```bash
veks datasets config set-cache /mnt/fast-storage/vectordata-cache
veks datasets config show
```

The directory layout under the resolved root is
`<host>:<port>/<url-path-prefix>/<filename>`, with a sibling
`<filename>.mrkl` carrying merkle state.

### Prebuffering datasets

Prebuffering downloads every facet of a profile so subsequent reads
are zero-copy mmap with no further HTTP requests.

#### From the CLI

```bash
veks datasets prebuffer --dataset my-dataset
veks datasets prebuffer --dataset my-dataset:default
veks datasets prebuffer --dataset my-dataset --at https://host/datasets/
veks datasets prebuffer --dataset my-dataset --cache-dir /tmp/vd-cache
```

#### From Rust

```rust
use vectordata::TestDataView;

let view = catalog.open_profile("my-dataset", "default")?;

// Walks every facet declared in the profile, drives each underlying
// storage to fully-resident state. Local-mmap and direct-HTTP
// facets no-op silently.
view.prebuffer_all()?;

// With per-facet progress
view.prebuffer_all_with_progress(&mut |facet, p| {
    let pct = if p.total_chunks > 0 {
        100.0 * p.verified_chunks as f64 / p.total_chunks as f64
    } else { 0.0 };
    eprintln!("  {facet}: {:.0}% ({}/{} chunks, {:.1} MiB)",
        pct, p.verified_chunks, p.total_chunks,
        p.total_bytes as f64 / 1_048_576.0);
})?;
```

`PrebufferProgress` exposes `verified_chunks`, `total_chunks`,
`verified_bytes`, `total_bytes`.

### Per-facet cache stats

`view.open_facet_storage(name)` returns a `FacetStorage` handle —
opaque from the outside but knowing whether the underlying transport
is cached.

| Method | local mmap | cached-remote | direct HTTP (no `.mref`) |
|---|---|---|---|
| `is_local()` | `true` | `true` once promoted | `false` |
| `is_complete()` | `true` | `true` once every chunk verified | `false` (always) |
| `cache_stats()` | `None` | `Some(CacheStats)` | `None` |
| `cache_path()` | `None` | path to cache file | `None` |
| `prebuffer()` | no-op | downloads + verifies | no-op |

```rust
use vectordata::CacheStats;

let storage = view.open_facet_storage("base_vectors")?;

if let Some(cs): Option<CacheStats> = storage.cache_stats() {
    let pct = 100.0 * cs.valid_chunks as f64 / cs.total_chunks as f64;
    println!("cached: {:.0}% ({}/{} chunks, {} bytes total)",
        pct, cs.valid_chunks, cs.total_chunks, cs.content_size);
}

// Drive this single facet to full-resident state (without touching
// the rest of the profile).
storage.prebuffer()?;
assert!(storage.is_complete());
assert!(storage.is_local());

// After prebuffer, the bytes are mmap-promoted; cache_path() points
// to the local cache file so external tools (mmap, hashing, etc.)
// can address it directly.
if let Some(path) = storage.cache_path() {
    println!("cached file landed at: {}", path.display());
}
```

### How chunked verification works

1. **`.mref` file** (published) — precomputed Merkle tree with
   SHA-256 hashes for all fixed-size data chunks (1 MiB by
   default).
2. **`.mrkl` file** (local) — tracks which chunks have been
   downloaded and verified. Persists across restarts; the merkle
   reference is embedded so `.mref` is needed only on first open.
3. **Read path** — check local cache → fetch chunk if missing →
   SHA-256 hash → compare against the merkle leaf → write to
   cache → update `.mrkl` → return bytes.
4. **Promotion** — once every chunk is verified, the cached
   storage flips into a `Mmap` view of the cache file. Subsequent
   reads (via `read_bytes` / `mmap_slice` / `get_slice`) are
   zero-copy with no per-read overhead.
5. **Fallback** — if the URL has no `.mref`, the cache layer is
   bypassed and every read is a direct HTTP RANGE request. Slow
   but correct; `is_complete()` always returns `false` so callers
   can distinguish.

### Listing cached datasets

```bash
veks datasets cache
```

---

## 2.13 Thread Safety

All reader types are `Send + Sync`:

- `VectorReader<T>` and `VvecReader<T>` traits both require
  `Send + Sync`.
- `XvecReader<T>`, `IndexedVvecReader<T>`, and `TypedReader<T>` are
  built on `Arc<Storage>` (each variant — `Mmap`, `Http`, `Cached`
  — is internally thread-safe). Cloning the outer `Arc` shares the
  underlying storage and any promoted-mmap state, so two clones see
  the same `is_complete()` and the same zero-copy slices.
- `Arc<dyn VectorReader<T>>` and `Arc<dyn VvecReader<T>>` returned
  by `TestDataView` methods are safe to share across threads.

Typical parallel access pattern:

```rust
use std::sync::Arc;
use rayon::prelude::*;
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://host/dataset/")?;
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;          // Arc<dyn VectorReader<f32>>

let results: Vec<f64> = (0..base.count())
    .into_par_iter()
    .map(|i| {
        let v = base.get(i).unwrap();
        v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
    })
    .collect();
```

---

## 2.14 Profiles

A dataset can have multiple profiles representing different subsets
or configurations of the same data:

- **`default`** — the full dataset, always present.
- **Sized profiles** — subsets like `10K`, `100K`, `1M` with a
  `base_count` that limits the number of base vectors.

### Accessing profiles

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("./my-dataset/")?;

if let Some(view) = group.profile("default") {
    let manifest = view.facet_manifest();
    println!("default profile has {} facets", manifest.len());
}

if let Some(view) = group.profile("100K") {
    let base = view.base_vectors()?;
    println!("100K profile: {} vectors", base.count());
}
```

### Profile YAML structure

```yaml
strata:
  - "mul:100K..1M/2"

profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
    neighbor_indices: profiles/default/neighbor_indices.ivec
    metadata_content: profiles/base/metadata_content.u8
    metadata_indices: profiles/default/metadata_indices.ivvec

  100K:
    base_count: 100000
    maxk: 100
    base_vectors: "profiles/base/base_vectors.fvec[0..100000)"
    query_vectors: profiles/base/query_vectors.fvec
    neighbor_indices: profiles/100K/neighbor_indices.ivec
```

Sized profiles share base/query data — `base_vectors` is windowed
via the `[lo..hi)` suffix rather than copied — but have their own
computed KNN, filtered results, and metadata indices. The
pipeline's `per_profile` mechanism generates these automatically;
see §1.7 for the full list of generator strategies (`decade`,
`mul:`, `fib:`, `linear:`, …).

---

## 2.15 Dataset attributes and metadata

### Required attributes

Every published dataset declares these attributes in
`dataset.yaml`:

```yaml
attributes:
  distance_function: L2          # L2, COSINE, or DOT_PRODUCT
  is_zero_vector_free: true      # no zero vectors in base data
  is_duplicate_vector_free: true # no duplicate vectors in base data
```

These are set automatically by the pipeline after scanning or dedup.

### Accessing attributes

```rust
use vectordata::TestDataGroup;

let group = TestDataGroup::load("./dataset/")?;

let dist = group.attribute("distance_function")
    .and_then(|v| v.as_str())
    .unwrap_or("unknown");

let zero_free = group.attribute("is_zero_vector_free")
    .and_then(|v| v.as_bool())
    .unwrap_or(false);

// Distance function is also exposed on the view
let view = group.profile("default").unwrap();
if let Some(df) = view.distance_function() {
    println!("metric: {df}");
}
```

### Metadata facets

```rust
use vectordata::open_facet_typed;

let view = catalog.open_profile("my-dataset", "default")?;

// Metadata content (e.g., labels as u8)
let meta = open_facet_typed::<u8>(&*view, "metadata_content")?;
println!("{} metadata records", meta.count());
for i in 0..5 {
    println!("  base[{}] label = {}", i, meta.get_native(i));
}

// Predicates (e.g., equality filters as u8)
let pred = open_facet_typed::<u8>(&*view, "metadata_predicates")?;

// Predicate results (variable-length ordinal lists)
let mi = view.metadata_indices()?;
for qi in 0..5 {
    let matching = mi.get(qi)?;
    println!("  query[{qi}] (field_0 == {}): {} matching base vectors",
        pred.get_native(qi), matching.len());
}
```

### Element type interrogation

```rust
use vectordata::typed_access::ElementType;
use vectordata::open_facet_typed;

let view = catalog.open_profile("my-dataset", "default")?;

let etype = view.facet_element_type("metadata_content")?;
match etype {
    ElementType::U8 => {
        let r = open_facet_typed::<u8>(&*view, "metadata_content")?;
        let val: u8 = r.get_native(0);
    }
    ElementType::I32 => {
        let r = open_facet_typed::<i32>(&*view, "metadata_content")?;
        let val: i32 = r.get_native(0);
    }
    _ => {
        // Widen to i64 for any integer type
        let r = open_facet_typed::<i64>(&*view, "metadata_content")?;
        let val: i64 = r.get_value(0)?;
    }
}
```
