# veks

A CLI for bulk processing of vector datasets used in approximate nearest neighbor (ANN) benchmarking. Veks handles the full lifecycle — downloading source data, converting between formats, importing into canonical layouts, computing ground truth, and analyzing results.

## Installation

```sh
cargo install --path veks
```

## Commands

### analyze

Inspect and characterize vector data files.

```sh
veks analyze describe base.fvec
```

### bulkdl

Bulk file downloader driven by a YAML config with token expansion and parallel downloads.

```sh
veks bulkdl downloads.yaml --concurrency 4
```

### convert

Convert vector data between formats (fvec, ivec, bvec, dvec, mvec, svec, npy, parquet, slab).

```sh
veks convert --source base.npy --target base.fvec
```

### import

Import source data into preferred internal format by facet type. Automatically selects the output format based on what the data represents (vectors → xvec, indices → ivec, metadata → slab).

```sh
veks import --dataset ./my-dataset --facet base_vectors --source raw/base.npy
```

### run

Execute a multi-step pipeline defined in `dataset.yaml`. Pipelines are DAG-ordered with skip-if-fresh semantics, variable interpolation, and dry-run support.

```sh
veks run dataset.yaml
veks run dataset.yaml --dry-run
veks run dataset.yaml --clean
```

### pipeline

Execute individual pipeline commands directly, with full shell tab-completion.

```sh
veks pipeline analyze stats --source=test.fvec
veks pipeline compute knn --base=base.fvec --queries=query.fvec --k=100
veks pipeline generate vectors --dimension=128 --count=10000 --output=base.fvec
```

Available command groups:

| Group | Commands |
|-------|----------|
| **analyze** | check-endian, compare, describe, explore, find, flamegraph, histogram, model-diff, plot, profile, select, slice, stats, verify-knn, verify-profiles, zeros |
| **catalog** | generate |
| **cleanup** | cleanfvec |
| **compute** | knn, sort |
| **config** | init, list-mounts, show |
| **convert** | file |
| **datasets** | cache, curlify, list, plan, prebuffer |
| **fetch** | dlhf |
| **generate** | dataset, derive, from-model, fvec-extract, ivec-extract, ivec-shuffle, predicated, sketch, vectors |
| **import** | facet |
| **info** | compute, file |
| **json** | jjq, rjq |
| **merkle** | create, diff, path, spoilbits, spoilchunks, summary, treeview, verify |
| **slab** | analyze, append, check, explain, export, get, import, inspect, namespaces, rewrite |

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
