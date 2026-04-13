# 2. API

The `vectordata` crate provides typed, unified access to vector datasets
regardless of storage location (local filesystem or HTTP) and record
structure (uniform or variable-length). This document is the definitive
reference for external consumers.

---

## 2.1 Quick Start

Add the dependency:

```toml
[dependencies]
vectordata = "0.18"
```

### Find and use a dataset by name

The primary access path: catalog → dataset → profile → facet → vectors.
No URLs or paths — just names:

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
// List profiles
let group = catalog.open("my-dataset")?;
for name in group.profile_names() {
    let view = group.profile(&name).unwrap();

    // List facets on this profile
    let manifest = view.facet_manifest();
    for (facet_name, desc) in &manifest {
        println!("  {} ({})", facet_name, desc.source_type.as_deref().unwrap_or("?"));
    }
}
```

### Open raw vector files directly

For low-level access without catalogs:

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
let record: Vec<i32> = reader.get(0)?;        // variable length per record
let dim: usize = reader.dim_at(0)?;           // dimension of record 0
```

Load a dataset by URL:

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://example.com/dataset/")?;
let view = group.profile("default").unwrap();

let base = view.base_vectors()?;           // Arc<dyn VectorReader<f32>>
let gt = view.neighbor_indices()?;         // Arc<dyn VectorReader<i32>>
let mi = view.metadata_indices()?;         // Arc<dyn VvecReader<i32>>
```

Load from `knn_entries.yaml` (jvector-compatible format — automatic
fallback when `dataset.yaml` is not found):

```rust
// Works for both local directories and HTTP URLs.
// If dataset.yaml is absent but knn_entries.yaml exists, it's used.
let group = TestDataGroup::load("/path/to/dataset/")?;
let group = TestDataGroup::load("https://example.com/dataset/")?;

// knn_entries entries become profiles: "name:profile" → profile
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;
```

Find and load a dataset from configured catalogs:

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

// Load catalogs from ~/.config/vectordata/catalogs.yaml
let sources = CatalogSources::new().configure_default();
let catalog = Catalog::of(&sources);

// Find a dataset by name
let entry = catalog.find_exact("my-dataset").expect("dataset not found");
println!("found: {} at {}", entry.name, entry.path);
println!("profiles: {:?}", entry.profile_names());

// Load it — the entry's path is relative to its catalog URL
// Use TestDataGroup::load with the catalog base URL + entry path
```

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

// Load from ~/.config/vectordata/catalogs.yaml
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
    println!("found: {}", entry.name);
    println!("path: {}", entry.path);
    println!("views: {:?}", entry.view_names());
}

// Find by glob pattern
let matches = catalog.match_glob("my-vectors*");
for entry in matches {
    println!("  {}", entry.name);
}
```

### Loading a discovered dataset

`CatalogEntry` provides the dataset's `path` relative to the catalog
root. Construct the full URL and pass it to `TestDataGroup::load`:

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

let sources = CatalogSources::new().configure_default();
let catalog = Catalog::of(&sources);
let entry = catalog.find_exact("my-dataset").expect("not found");

// The catalog source URL + entry path gives the dataset URL
// For a catalog at https://host/datasets/ with entry path "my-dataset",
// the dataset URL is https://host/datasets/my-dataset/
let group = TestDataGroup::load("https://host/datasets/my-dataset/")?;
let view = group.profile("default").unwrap();

// Now use the standard facet methods
let base = view.base_vectors()?;
let gt = view.neighbor_indices()?;
let mi = view.metadata_indices()?;
```

### Adding catalogs programmatically

```rust
// Use specific catalog URLs (overrides ~/.config)
let sources = CatalogSources::new()
    .add_catalogs(&[
        "https://my-bucket.s3.amazonaws.com/datasets/".into(),
        "/local/path/to/datasets/".into(),
    ]);
let catalog = Catalog::of(&sources);
```

---

## 2.3 File Extension Scheme

See [SRD §22](22-vector-file-extensions.md) for the full specification.

### Summary

| Suffix | Structure | Example |
|--------|-----------|---------|
| `.<type>` | Scalar (flat packed, no header) | `.u8`, `.i32`, `.f64` |
| `.<type>vec` | Uniform vector (fixed dimension per record) | `.fvec`, `.ivec`, `.u8vec` |
| `.<type>vvec` | Variable-length vector (per-record dimension) | `.ivvec`, `.fvvec`, `.u8vvec` |

Legacy aliases: `.fvec`=`.f32vec`, `.ivec`=`.i32vec`, `.bvec`=`.u8vec`,
`.svec`=`.i16vec`, `.mvec`=`.f16vec`, `.dvec`=`.f64vec`.

### Record layout

**Uniform (`vec`):** `[dim:i32 | elem₀ | elem₁ | ... | elem_{dim-1}]` —
all records share the same dimension. Random access by stride arithmetic.

**Variable (`vvec`):** Same per-record layout, but each record may have
a different dimension. Random access requires an offset index file
(`IDXFOR__<name>.<i32|i64>`).

---

## 2.4 Unified Open Functions

### `open_vec<T>(path_or_url) → Box<dyn VectorReader<T>>`

Opens a uniform vector file for typed random access. Transparently
handles local files (memory-mapped) and HTTP URLs (Range requests).

```rust
use vectordata::io::{open_vec, VectorReader};

// Local file
let r = open_vec::<f32>("data/base.fvec")?;

// Remote URL
let r = open_vec::<i32>("https://host/neighbors.ivec")?;

// Access
println!("count={}, dim={}", r.count(), r.dim());
let vec: Vec<f32> = r.get(0)?;
```

**Supported types:** `f32`, `f64`, `half::f16`, `i32`, `i16`, `i8`,
`u8`, `u16`, `u32`, `u64`, `i64`.

The type parameter `T` must match the file's element size. A mismatch
(e.g., `open_vec::<f32>("data.dvec")` where dvec is 8-byte f64) returns
an error at open time.

### `open_vvec<T>(path_or_url) → Box<dyn VvecReader<T>>`

Opens a variable-length vector file for typed random access. Requires
a companion `IDXFOR__<name>.<i32|i64>` offset index file (created
automatically by the pipeline or by `IndexedXvecReader::open`).

```rust
use vectordata::io::{open_vvec, VvecReader};

let r = open_vvec::<i32>("metadata_indices.ivvec")?;

println!("{} records", r.count());
let record: Vec<i32> = r.get(42)?;
let dim: usize = r.dim_at(42)?;    // each record has its own dimension
```

For HTTP access, the index file must be served alongside the data file.
The reader fetches it automatically.

---

## 2.5 Traits

### `VectorReader<T>` — Uniform Vectors

```rust
pub trait VectorReader<T>: Send + Sync {
    fn dim(&self) -> usize;
    fn count(&self) -> usize;
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;
}
```

All records have the same dimension. `get(index)` returns a `Vec<T>`
of length `dim()`.

### `VvecReader<T>` — Variable-Length Vectors

```rust
pub trait VvecReader<T: VvecElement>: Send + Sync {
    fn count(&self) -> usize;
    fn dim_at(&self, index: usize) -> Result<usize, IoError>;
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError>;
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;  // default impl
}
```

Each record may have a different dimension. `dim_at(index)` returns the
dimension of a specific record. `get(index)` returns a `Vec<T>` whose
length equals `dim_at(index)`.

### `VvecElement` — Byte Decoding

```rust
pub trait VvecElement: Copy + Send + Sync + 'static {
    const ELEM_SIZE: usize;
    fn from_le_bytes(bytes: &[u8]) -> Self;
}
```

Implemented for: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
`f32`, `f64`, `half::f16`.

---

## 2.6 Dataset Access via TestDataGroup

### Loading a dataset

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

// From a local directory containing dataset.yaml
let group = TestDataGroup::load("./my-dataset/")?;

// From an HTTP URL (fetches dataset.yaml, then Range-requests for data)
let group = TestDataGroup::load("https://host/datasets/my-dataset/")?;

// Access dataset-level attributes
let dist = group.attribute("distance_function");  // Option<&Value>
```

### Profile access

```rust
// Get a profile view (trait object)
let view: Arc<dyn TestDataView> = group.profile("default").unwrap();

// Get the concrete type for typed facet access
let gview: GenericTestDataView = group.generic_view("default").unwrap();
```

### Standard facet methods

All methods on `TestDataView` return readers that work identically
for local and remote datasets:

```rust
// Uniform vector facets → Arc<dyn VectorReader<T>>
let base: Arc<dyn VectorReader<f32>> = view.base_vectors()?;
let query: Arc<dyn VectorReader<f32>> = view.query_vectors()?;
let gt: Arc<dyn VectorReader<i32>> = view.neighbor_indices()?;
let dist: Arc<dyn VectorReader<f32>> = view.neighbor_distances()?;
let fki: Arc<dyn VectorReader<i32>> = view.filtered_neighbor_indices()?;
let fkd: Arc<dyn VectorReader<f32>> = view.filtered_neighbor_distances()?;

// Variable-length facet → Arc<dyn VvecReader<i32>>
let mi: Arc<dyn VvecReader<i32>> = view.metadata_indices()?;

// Metadata config (for path resolution, not data access)
let meta_cfg: Option<&FacetConfig> = view.metadata_content();
let pred_cfg: Option<&FacetConfig> = view.metadata_predicates();

// Facet discovery
let manifest: HashMap<String, FacetDescriptor> = view.facet_manifest();

// Element type interrogation
let etype = view.facet_element_type("metadata_content")?;  // ElementType::U8
```

### Typed scalar access (local only)

For scalar facets (`.u8`, `.i32`, etc.) with compile-time type safety
and runtime width validation:

```rust
let gview = group.generic_view("default").unwrap();

// Open with native type (zero-copy mmap access)
let r = gview.open_facet_typed::<u8>("metadata_content")?;
let val: u8 = r.get_native(42);

// Open with wider type (widening, always succeeds)
let r = gview.open_facet_typed::<i32>("metadata_content")?;
let val: i32 = r.get_value(42)?;  // u8 → i32

// Narrowing is rejected at open time
let err = gview.open_facet_typed::<u8>("some_i32_facet");  // Err(Narrowing)
```

---

## 2.7 Dataset YAML Profile Schema

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

Filter partition profiles from the profile list:

```rust
for name in group.profile_names() {
    let view = group.profile(&name).unwrap();
    // Partition profiles have "label-" prefix by convention
    // but are identified by the partition: true field in dataset.yaml
}
```

### knn_entries.yaml fallback

When `dataset.yaml` is not found, `TestDataGroup::load` falls back
to `knn_entries.yaml` (jvector-compatible format). Each entry maps
`"name:profile"` to base/query/gt paths:

```yaml
_defaults:
  base_url: https://example.com/data

"my-dataset:default":
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  gt: profiles/base/neighbor_indices.ivec
```

The `knn_entries` module can also be used directly:

```rust
use vectordata::knn_entries::KnnEntries;

let entries = KnnEntries::load("knn_entries.yaml")?;
println!("datasets: {:?}", entries.dataset_names());
let config = entries.to_config();  // → DatasetConfig
```

---

## 2.8 Facet Reference

| Facet code | YAML key | Trait method | Reader type | Typical format |
|-----------|----------|-------------|-------------|---------------|
| B | `base_vectors` | `base_vectors()` | `VectorReader<f32>` | `.fvec` |
| Q | `query_vectors` | `query_vectors()` | `VectorReader<f32>` | `.fvec` |
| G | `neighbor_indices` | `neighbor_indices()` | `VectorReader<i32>` | `.ivec` |
| D | `neighbor_distances` | `neighbor_distances()` | `VectorReader<f32>` | `.fvec` |
| M | `metadata_content` | `metadata_content()` | config only | `.u8`, `.slab` |
| P | `metadata_predicates` | `metadata_predicates()` | config only | `.u8`, `.slab` |
| R | `metadata_indices` | `metadata_indices()` | `VvecReader<i32>` | `.ivvec` |
| F (indices) | `filtered_neighbor_indices` | `filtered_neighbor_indices()` | `VectorReader<i32>` | `.ivec` |
| F (distances) | `filtered_neighbor_distances` | `filtered_neighbor_distances()` | `VectorReader<f32>` | `.fvec` |

For M and P facets, use `generic_view().open_facet_typed::<T>(name)` for
typed data access.

---

## 2.9 Complete Example: Verify Filtered KNN

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;
use vectordata::io::VvecReader;

fn verify_filtered_knn(dataset_url: &str) -> anyhow::Result<()> {
    let group = TestDataGroup::load(dataset_url)?;
    let view = group.profile("default").unwrap();
    let gview = group.generic_view("default").unwrap();

    // Load all facets
    let query = view.query_vectors()?;
    let fki = view.filtered_neighbor_indices()?;
    let mi = view.metadata_indices()?;
    let meta = gview.open_facet_typed::<u8>("metadata_content")?;
    let pred = gview.open_facet_typed::<u8>("metadata_predicates")?;

    println!("{} queries, {} base vectors", query.count(), meta.count());

    // For each query, verify filtered neighbors pass the predicate
    for qi in 0..query.count().min(10) {
        let pred_val = pred.get_native(qi);
        let neighbors = fki.get(qi)?;
        let matching = mi.get(qi)?;

        println!("query {}: predicate=field_0=={}, {} neighbors, {} matching base vectors",
            qi, pred_val, neighbors.len(), matching.len());

        // Every filtered neighbor should have the right metadata value
        for &ord in &neighbors {
            if ord < 0 { continue; } // sentinel for unfilled slots
            let meta_val = meta.get_native(ord as usize);
            assert_eq!(meta_val, pred_val,
                "query {}: neighbor {} has meta={}, expected {}",
                qi, ord, meta_val, pred_val);
        }
    }

    println!("verified!");
    Ok(())
}
```

---

## 2.10 Offset Index Files

Variable-length vector files (`.ivvec`, `.fvvec`, etc.) require a
companion offset index for random access:

```
data/metadata_indices.ivvec              # variable-length data
data/IDXFOR__metadata_indices.ivvec.i32  # offset index (< 2GB data)
data/IDXFOR__metadata_indices.ivvec.i64  # offset index (≥ 2GB data)
```

The index is a flat-packed array of byte offsets (one per record).
It is created automatically:

- By `IndexedXvecReader::open()` on first local access
- By the `generate vvec-index` pipeline step before publishing
- By `evaluate-predicates` immediately after writing vvec output

For remote access, the index file must be served alongside the data
file. `HttpIndexedXvecReader` (used internally by `open_vvec`) fetches
the index via HTTP, then uses it for Range requests to the data file.

---

## 2.11 Error Handling

```rust
use vectordata::io::IoError;

match open_vec::<f32>("data.fvec") {
    Ok(reader) => { /* use reader */ }
    Err(IoError::Io(e)) => eprintln!("I/O error: {}", e),
    Err(IoError::Http(e)) => eprintln!("HTTP error: {}", e),
    Err(IoError::InvalidFormat(msg)) => eprintln!("bad format: {}", msg),
    Err(IoError::OutOfBounds(idx)) => eprintln!("index {} out of range", idx),
    Err(IoError::VariableLengthRecords(msg)) => {
        // File has variable-length records — use open_vvec instead
        eprintln!("use open_vvec for this file: {}", msg);
    }
}
```

`VariableLengthRecords` is returned by `open_vec` (which expects
uniform records) when the file has variable-length records. The caller
should switch to `open_vvec`.

---

## 2.12 Prebuffering and Caching

### Cache location

Dataset facets downloaded from remote catalogs are cached locally.
The default cache directory is `~/.cache/vectordata/`. Each dataset
gets a subdirectory containing its cached facet files.

Configure the cache location in `~/.config/vectordata/settings.yaml`:

```yaml
cache_dir: /mnt/fast-storage/vectordata-cache
```

Or via the CLI:

```bash
# Set the cache directory
veks datasets config set-cache /mnt/fast-storage/vectordata-cache

# View current settings
veks datasets config show
```

The resolution order is:
1. `--cache-dir` CLI flag (per-command override)
2. `cache_dir` in `~/.config/vectordata/settings.yaml`
3. `$HOME/.cache/vectordata/` (default fallback)

### Prebuffering datasets

Prebuffering downloads all facets for a dataset profile into the local
cache so they can be accessed offline without further HTTP requests:

```bash
# Prebuffer from configured catalogs (looks up by name)
veks datasets prebuffer --dataset my-dataset

# Prebuffer a specific profile
veks datasets prebuffer --dataset my-dataset:default

# Prebuffer from a specific catalog URL
veks datasets prebuffer --dataset my-dataset --at https://host/datasets/

# Prebuffer with a custom cache directory
veks datasets prebuffer --dataset my-dataset --cache-dir /tmp/vd-cache
```

The prebuffer command:
1. Resolves the dataset via the catalog chain
2. Opens each facet view (base vectors, queries, GT, metadata, etc.)
3. Downloads all chunks, verified against merkle hashes (`.mref` files)
4. Stores the data in `<cache_dir>/<dataset_name>/`
5. Skips facets that are already fully cached

### Progress tracking during prebuffer

The `prebuffer_with_progress` API provides real-time download tracking
via `DownloadProgress` — a thread-safe atomic counter struct:

```rust
use vectordata::cache::CachedDataset;
use vectordata::transport::DownloadProgress;

let cached = CachedDataset::open("https://example.com/dataset/")?;
cached.prebuffer_with_progress(|progress: &DownloadProgress| {
    let pct = if progress.total_bytes() > 0 {
        100.0 * progress.downloaded_bytes() as f64 / progress.total_bytes() as f64
    } else { 0.0 };
    eprintln!("{:.1}% ({}/{} chunks)",
        pct, progress.completed_chunks(), progress.total_chunks());
});
```

`DownloadProgress` fields:
- `total_bytes()` / `downloaded_bytes()` — byte-level progress
- `total_chunks()` / `completed_chunks()` — chunk-level progress
- `fraction()` — convenience `0.0..1.0` ratio
- `is_failed()` — true if any chunk failed permanently
- `is_complete()` — true when all chunks are downloaded

### Listing cached datasets

```bash
veks datasets cache
```

Output:

```
Cache directory: /home/user/.cache/vectordata

Dataset                        Files         Size
----------------------------------------------------
my-dataset                            12      523.4 MiB
other-dataset                          8       48.2 MiB
```

### How caching works

When `TestDataGroup::load` opens a remote dataset, subsequent
`VectorReader::get()` calls issue HTTP Range requests for individual
records. This is efficient for random access but slow for sequential
scans.

Prebuffering downloads the full files up front. Once cached, the
`CachedChannel` layer serves reads from disk instead of HTTP. The
cache is merkle-verified: each chunk's hash is checked against the
`.mref` file to detect corruption or incomplete downloads.

Cache files are stored as:

```
~/.cache/vectordata/
├── my-dataset/
│   ├── dataset.yaml
│   ├── profiles/
│   │   ├── base/
│   │   │   ├── base_vectors.fvec
│   │   │   ├── base_vectors.fvec.mrkl    # local merkle state
│   │   │   ├── query_vectors.fvec
│   │   │   └── ...
│   │   └── default/
│   │       ├── neighbor_indices.ivec
│   │       ├── metadata_indices.ivvec
│   │       ├── IDXFOR__metadata_indices.ivvec.i32
│   │       └── ...
```

The `.mrkl` files track which chunks have been downloaded and verified.
They are local state, not published.

### How chunked verification works

1. **`.mref` file** (published) — precomputed Merkle tree with SHA-256
   hashes for all fixed-size data chunks (typically 64 KB)
2. **`.mrkl` file** (local) — bitfield tracking which chunks have been
   downloaded and verified
3. **Read path**: check local cache → fetch chunk if missing →
   SHA-256 hash → compare against Merkle leaf → save to cache →
   update `.mrkl` → return bytes
4. **Prebuffer**: eagerly downloads all unverified chunks; after
   completion, backend switches to direct local-file I/O with no
   per-read overhead

---

## 2.13 Thread Safety

All reader types are `Send + Sync`:

- `VectorReader<T>` trait requires `Send + Sync`
- `VvecReader<T>` trait requires `Send + Sync`
- `MmapVectorReader<T>` — backed by `Mmap` which is `Sync`
- `IndexedXvecReader` — backed by `Mmap` + `Vec<u64>`, both `Sync`
- `HttpVectorReader<T>` — uses `reqwest::blocking::Client` which is `Sync`
- `Arc<dyn VectorReader<T>>` and `Arc<dyn VvecReader<T>>` — returned
  by `TestDataView` methods, safe to share across threads

Typical parallel access pattern:

```rust
use std::sync::Arc;
use rayon::prelude::*;
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://host/dataset/")?;
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;  // Arc<dyn VectorReader<f32>>

// Safe to share across threads
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

- **`default`** — the full dataset, always present
- **Sized profiles** — subsets like `10K`, `100K`, `1M` with a
  `base_count` that limits the number of base vectors

### Accessing profiles

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("./my-dataset/")?;

// List available profiles (via facet manifest on each)
if let Some(view) = group.profile("default") {
    let manifest = view.facet_manifest();
    println!("default profile has {} facets", manifest.len());
}

// Access a sized profile
if let Some(view) = group.profile("100K") {
    let base = view.base_vectors()?;
    println!("100K profile: {} vectors", base.count());
}

// Concrete view for typed access
let gview = group.generic_view("default").unwrap();
let etype = gview.facet_element_type("metadata_content")?;
```

### Profile YAML structure

```yaml
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
    base_vectors: profiles/base/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
    neighbor_indices: profiles/100K/neighbor_indices.ivec
```

Sized profiles share base/query source data but have their own
computed KNN, filtered results, and metadata indices. The pipeline's
`per_profile` mechanism generates these automatically.

---

## 2.15 Dataset Attributes and Metadata

### Required attributes

Every published dataset must declare these attributes in
`dataset.yaml`:

```yaml
attributes:
  distance_function: L2          # L2, COSINE, or DOT_PRODUCT
  is_zero_vector_free: true      # no zero vectors in base data
  is_duplicate_vector_free: true  # no duplicate vectors in base data
```

These are set automatically by the pipeline after scanning or dedup.

### Accessing attributes

```rust
use vectordata::TestDataGroup;

let group = TestDataGroup::load("./dataset/")?;

// Top-level attributes
let dist = group.attribute("distance_function")
    .and_then(|v| v.as_str())
    .unwrap_or("unknown");
println!("distance function: {}", dist);

let zero_free = group.attribute("is_zero_vector_free")
    .and_then(|v| v.as_bool())
    .unwrap_or(false);
println!("zero-free: {}", zero_free);

// Distance function is also available on the view
let view = group.profile("default").unwrap();
if let Some(df) = view.distance_function() {
    println!("metric: {}", df);
}
```

### Metadata facets

Metadata (M) and predicates (P) are scalar files — flat-packed arrays
with no per-record header. Access them via `TypedReader`:

```rust
let gview = group.generic_view("default").unwrap();

// Metadata content (e.g., labels as u8)
let meta = gview.open_facet_typed::<u8>("metadata_content")?;
println!("{} metadata records", meta.count());
for i in 0..5 {
    println!("  base[{}] label = {}", i, meta.get_native(i));
}

// Predicates (e.g., equality filters as u8)
let pred = gview.open_facet_typed::<u8>("metadata_predicates")?;
println!("{} predicates", pred.count());
for i in 0..5 {
    println!("  query[{}] filter = field_0 == {}", i, pred.get_native(i));
}

// Predicate results (variable-length ordinal lists)
let mi = view.metadata_indices()?;
for qi in 0..5 {
    let matching = mi.get(qi)?;
    let pred_val = pred.get_native(qi);
    println!("  query[{}] (field_0 == {}): {} matching base vectors",
        qi, pred_val, matching.len());
}
```

### Element type interrogation

Before opening a facet, you can query its native element type:

```rust
use vectordata::typed_access::ElementType;

let gview = group.generic_view("default").unwrap();

let etype = gview.facet_element_type("metadata_content")?;
match etype {
    ElementType::U8 => {
        let r = gview.open_facet_typed::<u8>("metadata_content")?;
        // zero-copy native access
        let val: u8 = r.get_native(0);
    }
    ElementType::I32 => {
        let r = gview.open_facet_typed::<i32>("metadata_content")?;
        let val: i32 = r.get_native(0);
    }
    other => {
        // Widen to i64 for any integer type
        let r = gview.open_facet_typed::<i64>("metadata_content")?;
        let val: i64 = r.get_value(0)?;
    }
}
```
