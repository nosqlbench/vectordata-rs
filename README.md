# vectordata-rs

High-performance Rust toolkit for vector search datasets.

Load, analyze, and benchmark ANN datasets from anywhere — local files
or remote catalogs — with a single API call.

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

// Load a dataset from a catalog URL
let group = TestDataGroup::load("https://example.com/datasets/sift1m/")?;
let view = group.profile("default").unwrap();

// Access vectors — works identically for local and remote
let base = view.base_vectors()?;              // 1M float vectors
let gt = view.neighbor_indices()?;            // ground truth KNN
let mi = view.metadata_indices()?;            // predicate results (variable-length)

println!("{} base vectors, dim={}", base.count(), base.dim());
let nearest = gt.get(0)?;                     // neighbors of query 0
let matching = mi.get(0)?;                    // base ordinals matching predicate 0
```

---

## Workspace

### [vectordata](./vectordata/) — Dataset Access Library

The consumer-facing crate. One `open_vec` or `open_vvec` call handles
local mmap, remote HTTP Range requests, uniform or variable-length
records, and all element types (f32, i32, u8, f16, etc.).

See the [API Guide](./docs/design/23-vectordata-api-guide.md) for the
full reference.

### [veks](./veks/) — CLI and Pipeline Engine

```bash
# Configure a dataset catalog
veks datasets config add-catalog https://example.com/datasets/

# List available datasets
veks datasets list

# Download and cache a dataset locally
veks datasets prebuffer --dataset sift1m

# Inspect a dataset
veks analyze describe --source base_vectors.fvec
veks analyze explain-filtered-knn --ordinal 42

# Build a new dataset from source data
veks bootstrap -i

# Run the full pipeline (KNN, metadata, filtered search, verification)
veks run dataset.yaml
```

### [veks-pipeline](./veks-pipeline/) — Pipeline Commands

50+ commands for vector data processing: KNN computation, metadata
synthesis, predicate evaluation, filtered search, verification,
merkle hashing, and catalog generation.

### [slabtastic](./slabtastic/) — Random-Access Storage

Page-aligned storage engine for variable-length records. O(1) ordinal
lookup, append-friendly, supports files up to 2^63 bytes.

### [veks-anode](./veks-anode/) — Wire Formats

Binary codecs for structured metadata (MNode), predicate trees (PNode),
and the unified ANode wrapper.

---

## File Formats

| Extension | Structure | Description |
|-----------|-----------|-------------|
| `.fvec` | uniform vector | float32 vectors (dim header + data) |
| `.ivec` | uniform vector | int32 vectors |
| `.ivvec` | variable-length vector | int32 records with per-record dimension |
| `.u8` | scalar | flat-packed unsigned bytes |
| `.slab` | page-indexed | variable-length binary records |

All `vec` formats use a 4-byte little-endian dimension header per
record. `vvec` formats allow each record to have a different dimension.
See [SRD 22](./docs/design/22-vector-file-extensions.md) for the
complete extension scheme.

---

## Dataset Facets (BQGDMPRF)

A complete predicated search dataset includes:

| Code | Facet | Format |
|------|-------|--------|
| B | Base vectors | `.fvec` |
| Q | Query vectors | `.fvec` |
| G | Ground truth indices | `.ivec` |
| D | Ground truth distances | `.fvec` |
| M | Metadata labels | `.u8` |
| P | Predicate filters | `.u8` |
| R | Predicate result ordinals | `.ivvec` |
| F | Filtered KNN results | `.ivec` + `.fvec` |

---

## Quick Start

```bash
# Build
cargo build --release

# Run all tests (1000+)
cargo test

# Generate a fully synthetic dataset from scratch
cd veks/tests/fixtures/synthetic-1k
veks run dataset.yaml
```

---

## Documentation

- [API Guide](./docs/design/23-vectordata-api-guide.md) — How to use the vectordata library
- [Extension Scheme](./docs/design/22-vector-file-extensions.md) — vec/vvec/scalar file formats
- [Typed Access](./docs/design/21-typed-data-access.md) — TypedReader with width negotiation
- [SRD](./docs/design/README.md) — System Requirements and design documents

---

## License

Apache-2.0
