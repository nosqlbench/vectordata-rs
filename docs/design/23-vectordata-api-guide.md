# 23 — vectordata API Guide

The `vectordata` crate provides typed, unified access to vector datasets
regardless of storage location (local filesystem or HTTP) and record
structure (uniform or variable-length). This document is the definitive
reference for external consumers.

---

## 23.1 Quick Start

Add the dependency:

```toml
[dependencies]
vectordata = "0.17"
```

Open and read vectors:

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

Load a dataset by catalog:

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://example.com/dataset/")?;
let view = group.profile("default").unwrap();

let base = view.base_vectors()?;           // Arc<dyn VectorReader<f32>>
let gt = view.neighbor_indices()?;         // Arc<dyn VectorReader<i32>>
let mi = view.metadata_indices()?;         // Arc<dyn VvecReader<i32>>
```

---

## 23.2 File Extension Scheme

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

## 23.3 Unified Open Functions

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

## 23.4 Traits

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

## 23.5 Dataset Access via TestDataGroup

### Loading a dataset

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

// From a local directory containing dataset.yaml
let group = TestDataGroup::load("./my-dataset/")?;

// From an HTTP URL (fetches dataset.yaml, then Range-requests for data)
let group = TestDataGroup::load("https://host/datasets/sift1m/")?;

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

## 23.6 Dataset YAML Profile Schema

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

---

## 23.7 Facet Reference

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

## 23.8 Complete Example: Verify Filtered KNN

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

## 23.9 Offset Index Files

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

## 23.10 Error Handling

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
