# vectordata-rs

A Rust toolkit for **building, sharing, and consuming** vector-search
benchmark datasets — for evaluating an ANN index, running filtered-search
experiments, or publishing reproducible benchmarks, with the data
plumbing handled for you.

## Which tool do you need?

- **Download & read datasets** → **`vectordata`** — the access CLI + Rust library.
- **Host & publish datasets** → **`vecd`** — the server (a private gateway with auth + TLS).
- **Build, process & verify datasets** → **`veks`** — the pipeline + analysis CLI.

Grab a prebuilt binary from [**Releases**](https://github.com/nosqlbench/vectordata-rs/releases),
or install from source with cargo:

```bash
cargo install --git https://github.com/nosqlbench/vectordata-rs vectordata   # download + read
cargo install --git https://github.com/nosqlbench/vectordata-rs vecd         # host + publish
cargo install --git https://github.com/nosqlbench/vectordata-rs veks         # build + process
```

Most people start by **consuming** datasets: find one in a catalog and
read its vectors. There are two ways to do that, both over the same
catalog → profile → facet model. (Building and serving your own come
further down.)

## 1. Find & download datasets — the CLI

The quickest entry point is `vectordata explore`: with no arguments it
pops a **catalog picker**, lets you browse the datasets in your
configured catalogs, and opens a TUI (norms, distances, eigenvalues, PCA)
against any of them — fetching data on demand.

```bash
vectordata config catalog add https://example.com/datasets/   # register a catalog (once)

vectordata explore                    # easiest: pick + visualize interactively
vectordata datasets list              # what's available
vectordata datasets describe my-dataset   # profiles, facets, metric
vectordata datasets precache my-dataset   # download + verify into the local cache
```

Full walk-through: [Find and fetch datasets with the CLI](./vectordata/docs/find-and-fetch-datasets.md).

## 2. Read datasets — the Rust API

The same flow programmatically: resolve a catalog, open a profile by
name, read its facets through typed readers.

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;
use vectordata::{open_facet_typed, TestDataView, TypedReader};

// Catalog → profile → reader. The prescribed entry path.
let catalog = Catalog::of(&CatalogSources::new().configure_default());
let view    = catalog.open_profile("myset", "default")?;

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
[Accessing datasets from Rust](./vectordata/docs/access-datasets-from-rust.md).

## What's in the box

**[vectordata](./vectordata/)** — the access library. Add it as a
dependency and read any dataset from anywhere.
([Tutorial](./vectordata/docs/access-datasets-from-rust.md) ·
[API Reference](./docs/sysref/02-api.md))

**[veks](./veks/)** — the CLI. Bootstrap new datasets, run processing
pipelines, analyze data, publish to catalogs.
([Commands](./docs/sysref/05-commands.md) |
[Pipeline](./docs/sysref/04-pipeline.md) |
[Import](./docs/sysref/07-import.md))

**[vecd](./vecd/)** — the server. Self-host a private gateway that publishes
datasets over HTTP with authentication, per-namespace access control, and
versioned atomic uploads.
([Intro & quickstart](./docs/guides/vecd-intro.md) ·
[End-to-end tutorial](./docs/tutorials/vecd-end-to-end/) ·
[Design](./docs/design/vecd-daemon.md))

**[veks-pipeline](./veks-pipeline/)** — 50+ pipeline commands for KNN
computation, metadata synthesis, predicate evaluation, filtered search,
and verification.

**[slabtastic](./slabtastic/)** — page-aligned storage engine for
variable-length records.

**[veks-anode](./veks-anode/)** — binary codecs for structured metadata
and predicate trees.

---

## Build, analyze, and serve your own

Finding and reading datasets (above) is the consume side. The rest of the
toolkit lets you **make** and **host** them.

### Build & analyze — the `veks` CLI

```bash
veks bootstrap -i                    # interactive wizard
veks run dataset.yaml                # run the full pipeline (KNN, metadata, predicates, …)
veks check                           # verify everything
veks publish                         # push to a catalog

veks analyze describe --source base_vectors.fvecs
veks analyze explain-filtered-knn --ordinal 42
```

### Reproducible & cross-verified — the FAISS/numpy parity guarantee

This is what makes a published benchmark *reproducible*. The ground truth `veks`
produces is held to a numerical parity guarantee against the Python `knn_utils`
reference (FAISS + numpy). Four independent KNN engines — SimSIMD (`knn-metal`),
pure `std::arch` (`knn-stdarch`), BLAS sgemm (`knn-blas`), and FAISS
(`knn-faiss`) — are cross-verified at the unit-test level (asserting *zero*
differing neighbors on deterministic fixtures) and re-checked end-to-end before
every dataset is published. See the
[KNN engine conformance section](./docs/sysref/12-knn-utils-verification.md#127-cross-engine-conformance-testing)
for observed numbers and the two degenerate regimes where a small boundary
tolerance is justified.

See it live in 5 seconds — every available engine on one fixture, side by side:

```bash
cargo install --features knnutils,faiss --path veks
veks pipeline verify engine-parity --use-synthetic \
  --dim 32 --base-count 500 --query-count 20 --neighbors 5
```

Engines you didn't compile show up as `skipped: feature not enabled`. The
in-tree conformance suite runs the same checks:

```bash
cargo test -p veks-pipeline --features knnutils,faiss \
  --lib pipeline::commands::compute_knn
cargo test -p veks-pipeline --features knnutils \
  --lib pipeline::commands::verify_dataset_knnutils
```

### Serve — the `vecd` server

Self-host a private gateway that publishes datasets over (optionally
TLS-secured) HTTP, with authentication, per-namespace access control, and
versioned atomic uploads. A vectordata client then `config catalog add`s
your endpoint and reads from it exactly like any other catalog.

```bash
vecd tls generate                                          # self-signed cert → HTTPS (optional)
vecd ns add datasets --owner me --backend-config store --active
vecd start
vectordata datasets push ./my-dataset --to datasets       # publish; then `datasets list` finds it
```

Full walk-through: the [end-to-end vecd tutorial](./docs/tutorials/vecd-end-to-end/).

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
**uniform vector** (`.fvecs`, `.ivecs` — fixed dimension per record), and
**variable-length vector** (`.ivvecs` — each record has its own
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
