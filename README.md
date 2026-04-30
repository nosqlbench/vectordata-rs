# vectordata-rs

A Rust toolkit for building, sharing, and consuming vector search
benchmark datasets.

Whether you're evaluating an ANN index, running filtered search
experiments, or publishing reproducible benchmarks — vectordata-rs
handles the data plumbing so you can focus on the search.

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;
use vectordata::{open_facet_typed, TestDataView, TypedReader};

// Catalog → profile → reader. The prescribed entry path.
let catalog = Catalog::of(&CatalogSources::new().configure_default());
let view    = catalog.open_profile("sift1m", "default")?;

let base = view.base_vectors()?;       // Arc<dyn VectorReader<f32>>  — 1M vectors
let gt   = view.neighbor_indices()?;   // Arc<dyn VectorReader<i32>>  — exact KNN
let mi   = view.metadata_indices()?;   // Arc<dyn VvecReader<i32>>    — predicate matches

let nearest:  Vec<i32> = gt.get(0)?;
let matching: Vec<i32> = mi.get(0)?;

// Typed scalar metadata via the same view handle
let label: TypedReader<u8> = open_facet_typed(&*view, "metadata_content")?;
let v: u8 = label.get_native(42);

// Make the whole profile zero-copy mmap for hot loops
view.prebuffer_all()?;
```

You never construct URLs, never name a transport, and never decide
whether a remote dataset should be cached. The crate picks the
right backing storage (local mmap, merkle-cached HTTP with
auto-promotion to mmap, or direct HTTP fallback) from the catalog
entry alone. Uniform and variable-length facets share the same
`view.X()` shape; typed scalars come through `open_facet_typed`.

This is the prescribed pattern — full walk-through in
[Accessing datasets from Rust](./docs/tutorials/access-datasets-from-rust.md).

## Verified against FAISS and numpy

Ground truth produced by vectordata-rs is held to a numerical
parity guarantee against the Python `knn_utils` reference (FAISS +
numpy). Four independent KNN engines — SimSIMD (`knn-metal`), pure
`std::arch` (`knn-stdarch`), BLAS sgemm (`knn-blas`), and FAISS
(`knn-faiss`) — are cross-verified at the unit-test level (asserting
*zero* differing neighbors on deterministic fixtures) and re-checked
end-to-end before every dataset is published. See the
[KNN engine conformance section](./docs/sysref/12-knn-utils-verification.md#127-cross-engine-conformance-testing)
for observed numbers and the two degenerate regimes where a small
boundary tolerance is justified.

**See it live in 5 seconds** — `veks pipeline verify engine-parity`
runs every available engine on the same fixture and prints a
side-by-side neighbor table plus a pair-wise classification:

```sh
cargo install --features knnutils,faiss --path veks
veks pipeline verify engine-parity --synthetic \
  --dim 32 --base-count 500 --query-count 20 --neighbors 5
```

Engines you didn't compile show up as `skipped: feature not enabled`.
Run the in-tree conformance suite as well:

```sh
cargo test -p veks-pipeline --lib pipeline::commands::compute_knn
cargo test -p veks-pipeline --features knnutils,faiss \
  --lib pipeline::commands::compute_knn
cargo test -p veks-pipeline --features knnutils \
  --lib pipeline::commands::verify_dataset_knnutils
```

---

## What's in the box

**[vectordata](./vectordata/)** — the access library. Add it as a
dependency and read any dataset from anywhere.
([Tutorial](./docs/tutorials/access-datasets-from-rust.md) ·
[API Reference](./docs/sysref/02-api.md))

**[veks](./veks/)** — the CLI. Bootstrap new datasets, run processing
pipelines, analyze data, publish to catalogs.
([Commands](./docs/sysref/05-commands.md) |
[Pipeline](./docs/sysref/04-pipeline.md) |
[Import](./docs/sysref/07-import.md))

**[veks-pipeline](./veks-pipeline/)** — 50+ pipeline commands for KNN
computation, metadata synthesis, predicate evaluation, filtered search,
and verification.

**[slabtastic](./slabtastic/)** — page-aligned storage engine for
variable-length records.

**[veks-anode](./veks-anode/)** — binary codecs for structured metadata
and predicate trees.

---

## Working with datasets

### Discover

```bash
veks datasets config add-catalog https://example.com/datasets/
veks datasets list
veks datasets probe --dataset sift1m
```

### Download

```bash
veks datasets prebuffer --dataset sift1m
```

### Analyze

```bash
veks analyze describe --source base_vectors.fvec
veks analyze explain-predicates --ordinal 42
veks analyze explain-filtered-knn --ordinal 42
```

### Build your own

```bash
veks bootstrap -i                    # interactive wizard
veks run dataset.yaml                # run the full pipeline
veks check                           # verify everything
veks publish                         # push to catalog
```

---

## Dataset anatomy

A dataset is a set of typed facets describing a vector search benchmark:

| Code | Facet | What it is |
|------|-------|-----------|
| **B** | Base vectors | The collection you search through |
| **Q** | Query vectors | What you're searching for |
| **G** | Ground truth | The correct nearest neighbors |
| **D** | Distances | How far each neighbor is |
| **M** | Metadata | Per-vector labels (for filtered search) |
| **P** | Predicates | Per-query filters (for filtered search) |
| **R** | Predicate results | Which base vectors pass each filter |
| **F** | Filtered KNN | Nearest neighbors after filtering |
| **O** | Oracle partitions | Per-label base vectors + partitioned KNN |

Files come in three flavors: **scalar** (`.u8`, `.i32` — flat arrays),
**uniform vector** (`.fvec`, `.ivec` — fixed dimension per record), and
**variable-length vector** (`.ivvec` — each record has its own
dimension).

[Data Model Reference](./docs/sysref/01-data-model.md)

---

## Getting started

```bash
cargo build --release
cargo test                           # 1000+ tests

# Try it: generate a complete synthetic dataset from scratch
cd veks/tests/fixtures/synthetic-1k
veks run dataset.yaml                # 18 steps, ~0.4 seconds
veks check                           # all green
```

---

## Documentation

### [System Reference](./docs/sysref/README.md)

| | |
|---|---|
| [Data Model](./docs/sysref/01-data-model.md) | File formats, facets, dataset.yaml, directory layout |
| [API](./docs/sysref/02-api.md) | vectordata library: loading, reading, catalogs, caching, thread safety |
| [Catalogs](./docs/sysref/03-catalogs.md) | Dataset discovery, publishing, prebuffering, merkle integrity |
| [Pipeline](./docs/sysref/04-pipeline.md) | DAG engine, step execution, profiles, variables |
| [Commands](./docs/sysref/05-commands.md) | All 50+ pipeline and analysis commands |
| [Processing](./docs/sysref/06-processing.md) | Dedup, normalization, zero detection, KNN, predicates |
| [Import](./docs/sysref/07-import.md) | Bootstrap wizard, source detection, pipeline generation |
| [Architecture](./docs/sysref/08-architecture.md) | CommandOp trait, resource governance, UI eventing, swimlane |
| [Algorithms](./docs/sysref/09-algorithms.md) | Normalization, vector generation, model extraction, numerical methods |
| [ANode Codec](./docs/sysref/10-anode-codec.md) | MNode/PNode wire formats, 13 vernacular renderers, codec architecture |
| [Completions](./docs/sysref/11-completions.md) | Dynamic shell completion engine, value providers, bash integration |
| [knn_utils](./docs/sysref/12-knn-utils-verification.md) | Cross-verification against Python knn_utils reference |

---

## License

Apache-2.0
