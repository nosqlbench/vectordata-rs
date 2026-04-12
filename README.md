# vectordata-rs

A Rust toolkit for building, sharing, and consuming vector search
benchmark datasets.

Whether you're evaluating an ANN index, running filtered search
experiments, or publishing reproducible benchmarks — vectordata-rs
handles the data plumbing so you can focus on the search.

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://example.com/datasets/sift1m/")?;
let view = group.profile("default").unwrap();

let base = view.base_vectors()?;           // 1M float vectors
let gt = view.neighbor_indices()?;         // exact KNN ground truth
let mi = view.metadata_indices()?;         // predicate filter results

let nearest = gt.get(0)?;                  // query 0's nearest neighbors
let matching = mi.get(0)?;                 // base vectors matching predicate 0
```

Local files and remote catalogs use the same API. Uniform vectors
and variable-length records use the same API. You never pick an
implementation — just call `open_vec` or `open_vvec` and go.

---

## What's in the box

**[vectordata](./vectordata/)** — the access library. Add it as a
dependency and read any dataset from anywhere.
([API Reference](./docs/sysref/02-api.md))

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
