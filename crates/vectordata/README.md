# vectordata

Find, download, and read vector-search benchmark datasets by name —
from the terminal or from Rust. You name a dataset; the crate resolves
it through your catalogs, fetches what you ask for into a verified local
cache with progress, and serves vectors, ground truth and metadata from
there.

Datasets carry ground truth (exact and filtered KNN) that has been
numerically cross-verified against FAISS and the Python `knn_utils`
reference (numpy + FAISS) — see the
[KNN engine conformance section](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/sysref/12-knn-utils-verification.md#127-cross-engine-conformance-testing).

## From the command line

```bash
# Register a catalog once (an HTTP URL or a local directory).
vectordata config catalog add https://example.com/datasets/

vectordata explore                          # browse and visualize interactively
vectordata datasets list                    # what's available
vectordata datasets describe my-dataset     # profiles, facets, metric
vectordata datasets precache my-dataset:default   # fetch and verify into the cache
vectordata cache list                       # what's cached
```

Walk-through: [Find and fetch datasets with the CLI](https://github.com/nosqlbench/vectordata-rs/blob/main/crates/vectordata/docs/find-and-fetch-datasets.md).

## From Rust

```rust
use vectordata::catalog::{Catalog, CatalogSources};
use vectordata::fetch::{FetchRequest, TextMeter};

let catalog = Catalog::of(&CatalogSources::new().configure_default());
let view = catalog.open_spec("my-dataset:default")?.view()?;

// Fetch what you need, with the same progress meter the CLI draws.
view.fetch(
    &FetchRequest::facets(["base_vectors", "query_vectors"]),
    &mut TextMeter::stderr("Fetch"),
)?;

let base = view.base_vectors()?;
let v: Vec<f32> = base.get(42)?;
```

Every `vectordata` command is a thin layer over a library call; the
crate documentation names the call for each.

## Where to look

- **[Common tasks](https://docs.rs/vectordata/latest/vectordata/#common-tasks)**
  — the index: one call per task, with examples, layers, traps and
  feature flags.
- **`AGENTS.md`** (in this crate, and rendered as
  [`vectordata::_agents`](https://docs.rs/vectordata/latest/vectordata/_agents/index.html))
  — the same index for coding agents, plus a map from every CLI command
  to its library call and a paragraph to paste into your own project's
  agent instructions.
- [Examples](https://github.com/nosqlbench/vectordata-rs/tree/main/crates/vectordata/examples)
  — open by spec, fetch with progress, stream base vectors.
- [Tutorial: Accessing datasets from Rust](https://github.com/nosqlbench/vectordata-rs/blob/main/crates/vectordata/docs/access-datasets-from-rust.md)
- [API reference](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/sysref/02-api.md),
  [data model](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/sysref/01-data-model.md),
  [catalogs](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/sysref/03-catalogs.md)

## Configuration

The cache location is set in `~/.config/vectordata/settings.yaml` (or
`$VECTORDATA_HOME/settings.yaml`), or with
`vectordata config set cache <dir>`. There is no silent fallback: an
unconfigured cache is an error whose message prints the setup commands.

License: Apache-2.0
