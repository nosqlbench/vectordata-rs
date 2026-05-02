# veks

A CLI for bulk processing of vector datasets used in approximate nearest neighbor (ANN) benchmarking. Veks handles the full lifecycle — downloading source data, converting between formats, importing into canonical layouts, computing ground truth, and analyzing results.

## Verified against FAISS and numpy

Ground truth produced by `veks` is held to a numerical parity
guarantee against the Python `knn_utils` reference (FAISS + numpy).
Four KNN engines (`knn-metal` / SimSIMD, `knn-stdarch`, `knn-blas`,
`knn-faiss`) are cross-verified at the unit-test level and the
`verify dataset-knnutils` pipeline command re-runs FAISS on a
sampled subset of every published dataset to catch regressions.
See the [conformance section](../docs/sysref/12-knn-utils-verification.md#127-cross-engine-conformance-testing)
for the comparison model and a per-test breakdown.

**Live demo** — runs every available engine on the same input and
prints a side-by-side comparison:

```sh
veks pipeline verify engine-parity --synthetic \
  --dim 32 --base-count 500 --query-count 20 --neighbors 5
```

In-tree test runs:

```sh
cargo test -p veks-pipeline --lib pipeline::commands::compute_knn
cargo test -p veks-pipeline --features knnutils,faiss \
  --lib pipeline::commands::compute_knn
cargo test -p veks-pipeline --features knnutils \
  --lib pipeline::commands::verify_dataset_knnutils
```

## Installation

```sh
cargo install --path veks
```

## Commands

The CLI is organized into a few top-level categories. Most commands
are also reachable directly without the category prefix (`veks run`
is shorthand for `veks prepare run`).

| Category | What it does |
|----------|--------------|
| `veks datasets` | Browse, search, and cache datasets from configured catalogs |
| `veks prepare`  | Bootstrap, import, stratify, run pipelines, publish |
| `veks pipeline` | Run a single pipeline command directly (the building blocks) |
| `veks interact` | Interactive TUIs for exploring vector data |

### datasets

Discover and cache datasets from configured catalogs.

```sh
veks datasets list
veks datasets prebuffer --dataset myset
veks datasets cache-status --dataset myset
veks datasets probe https://example.com/datasets/myset
veks datasets drop-cache --dataset myset
veks datasets config show           # cache directory + mounts
```

### prepare

End-to-end dataset preparation. The wizard infers facets, generates
a `dataset.yaml` with a complete pipeline DAG, and stratifies into
multi-scale sized profiles when the dataset is large enough.

```sh
veks bootstrap -i                    # interactive wizard
veks bootstrap -y                    # accept defaults
veks run                             # execute pipeline
veks run --dry-run
veks run --clean
veks run --explain-staleness         # show why each step would re-run
veks run --provenance version-aware  # ignore minor/patch binary bumps
veks check                           # pre-flight readiness checks
veks publish                         # push to S3 / catalog
veks stratify --strata "decade"      # add sized profiles after the fact
veks cache-gc                        # remove orphaned cache files
veks cache-compress                  # gzip eligible cache artifacts
```

Sized profiles live under the root-level `strata:` block in
`dataset.yaml`. Available generator strategies:

| Strategy        | Effect |
|-----------------|--------|
| `<size>`            | Single literal size (e.g. `100k`) |
| `step:<lo>..<hi>/<step>` | Arithmetic range with explicit step |
| `parts:<lo>..<hi>/<n>`   | Range divided into `n` equal segments |
| `mul:<lo>..<hi>/<factor>` | Geometric progression by `factor` (upper bound optional) |
| `fib:<lo>`          | Fibonacci progression starting at `lo`, capped at `base_count` |
| `linear:<lo>/<step>` | Open-ended arithmetic, capped at `base_count` |
| `decade`            | 100k, 200k, … 900k, 1m, 2m, … one detent per decimal click |

See [docs/sysref/01-data-model.md](../docs/sysref/01-data-model.md)
for the full strata spec.

### interact

Interactive TUIs for poking at vector data — open by name from a
catalog or by file path.

| Command | What it does |
|---------|--------------|
| `interact explore` | Norms, distances, eigenvalues, PCA in one TUI; tab-switchable views |
| `interact shell`   | REPL: `info`, `get`, `range`, `head`, `tail`, `dist`, `norm`, `stats` |
| `interact values`  | Scrollable raw-values grid: ordinals × dimensions, decimal-aligned, with 24-bit-color heatmap, sig-digit control, L2-norm column, and L2-normalized view toggle |

```sh
veks interact explore --dataset myset
veks interact values  --source ./data/base.fvec --start 0 --digits 4
veks interact shell   --source ./data/base.fvec "info; range 0 5"
```

See the [Explore tutorial](../docs/tutorials/explore-vector-data.md)
for the full keybinding sheet (vim hjkl/HJKL navigation, palette/curve
cycling, color-blind-safe palettes, Turbo / Spectrum scatter palettes).

### pipeline

Run an individual pipeline command directly, with full shell
tab-completion. Use this when scripting outside the DAG runner.

```sh
veks pipeline analyze stats --source=test.fvec
veks pipeline compute knn --base=base.fvec --queries=query.fvec --k=100
veks pipeline generate vectors --dimension=128 --count=10000 --output=base.fvec
```

Pipeline command groups (run `veks pipeline <group> --help` for the
list of subcommands in each):

| Group | What it does |
|-------|--------------|
| `analyze`   | Inspect / characterize / verify vector data and datasets |
| `catalog`   | Build catalog index files (`generate`, `stats`) |
| `cleanup`   | Remove overlapping query vectors |
| `compute`   | Brute-force KNN engines (stdarch / BLAS / metal), filtered KNN, partition profiles, predicate evaluation |
| `download`  | Bulk file downloader, HuggingFace fetcher |
| `generate`  | Synthetic data — vectors, datasets, metadata, predicates, shuffles, sketches |
| `merkle`    | Create / verify / diff / spoil merkle trees |
| `pipeline`  | DAG primitives (`require`) |
| `query`     | jq-like queries against JSON or slab records |
| `state`     | Pipeline variable management (`set`, `clear`) |
| `transform` | Convert formats, extract records, normalize / filter for knn_utils parity |
| `verify`    | Cross-engine, knn_utils-parity, predicate, and partition verification |

The full subcommand list is dense and changes often — `veks pipeline
--help` is the source of truth.

### completions

Generate shell completions for bash, zsh, fish, elvish, or PowerShell.

```sh
veks completions bash > ~/.local/share/bash-completion/completions/veks
```

## Supported formats

| Extension | Type | Element |
|-----------|------|---------|
| `.fvec` | float vectors | 4 bytes (f32) |
| `.ivec` | integer vectors | 4 bytes (i32) |
| `.bvec` | byte vectors | 4 bytes (u8 padded) |
| `.dvec` | double vectors | 8 bytes (f64) |
| `.mvec` | half-float vectors | 2 bytes (f16) |
| `.svec` | short vectors | 2 bytes (i16) |
| `.npy` | NumPy arrays | varies |
| `.parquet` | Apache Parquet | varies |
| `.slab` | slabtastic | variable-length records |

## License

Apache-2.0
