<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Getting Started with Veks — From Install to Publish

This tutorial walks through the full lifecycle: installing veks,
browsing remote datasets, preparing a new dataset from source files,
and publishing it to S3.

---

## Part 1: Installation

### Prerequisites

On a fresh Ubuntu/Debian system, ensure you have a C compiler and
OpenSSL development headers:

```shell
sudo apt update
sudo apt install gcc pkg-config libssl-dev
```

### Install Rust

If you don't already have Rust installed:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts, then load the environment:

```shell
. "$HOME/.cargo/env"
```

### Install Veks

```shell
cargo install veks
```

The first install compiles from source and takes a few minutes — veks
has dependencies for many vector formats, SIMD acceleration, TUI
rendering, and concurrent I/O. Subsequent `cargo install veks` runs
are incremental and much faster.

### Enable Tab Completion

Activate tab completion for your current session:

```shell
eval "$(veks completions)"
```

To make it permanent, add it to your shell profile:

```shell
echo 'eval "$(veks completions)"' >> ~/.bashrc
```

Veks auto-detects your shell (bash, zsh, fish). Tab completion
provides context-aware suggestions for commands, options, dataset
names, profile names, and filter values.

---

## Part 2: Browsing Datasets

### Add a Catalog

Catalogs are remote indexes that tell veks what datasets are available.
Add one:

```shell
veks datasets config add-catalog https://vectordata-testdata.s3.us-east-1.amazonaws.com/askmeforthis/
```

Veks verifies the catalog is reachable and reports how many datasets
it contains.

### List Available Datasets

```shell
veks datasets list
```

This shows all datasets from your configured catalogs with their
profiles and distance metrics. Use filters to narrow the list:

```shell
veks datasets list --matching-name "img*" --with-metric COSINE
```

### Explore a Remote Dataset

Open the interactive vector space explorer on a remote dataset —
veks streams the data on demand:

```shell
veks interact explore --dataset img-search --profile default
```

Navigate between views (norms, distances, eigenvalues, PCA) with
Tab or F1-F8. Press `?` for the full key reference.

---

## Part 3: Preparing a Dataset for Publishing

This section covers taking raw source files (base vectors, query
vectors, ground truth) and producing a publish-ready dataset.

### Gather Source Files

Start with a clean directory containing your source files. Use clear
names so the bootstrap wizard can identify each file's role:

```
my-dataset/
├── _base_vectors.npy       # base corpus vectors
├── _query_vectors.npy      # query vectors
├── _ground_truth.ivec      # pre-computed neighbor indices
└── _ground_truth_dist.fvec # pre-computed neighbor distances (optional)
```

Underscore-prefixed names (`_base_vectors`) are treated as source
files and excluded from publishing.

### Run the Bootstrap Wizard

```shell
cd my-dataset
veks prepare bootstrap --interactive
```

The wizard asks about:
- **Dataset name** — used in catalogs and remote URLs
- **Distance metric** — L2, Cosine, or DotProduct (describes how the
  embeddings were trained)
- **L2 normalization** — whether to normalize vectors (converts L2
  distance ranking to Cosine-equivalent)
- **Ground truth** — if you have pre-computed KNN files, the wizard
  detects them and skips KNN computation
- **Sized profiles** — generates windowed sub-profiles at multiple
  scales (e.g., 1M, 2M, 4M, 10M) for benchmarking at different sizes

The wizard creates `dataset.yaml` with a complete pipeline definition.

### Run the Pipeline

```shell
veks prepare run
```

This executes the pipeline defined in `dataset.yaml`: converting
formats, shuffling, deduplicating, splitting into base/query sets,
computing KNN ground truth (if not pre-provided), generating merkle
trees, and building catalog indexes.

Progress is shown in the terminal. The pipeline is resumable — if
interrupted, re-run `veks prepare run` to pick up where it left off.

### Add Sized Profiles (Optional)

If you want to add windowed profiles after the initial run:

```shell
veks prepare stratify
veks prepare run
```

Stratification creates sub-profiles (e.g., 1M, 4M, 10M base vectors)
and the second run computes KNN ground truth for each.

### Verify Readiness

```shell
veks prepare check
```

This runs all pre-flight checks: pipeline completeness, merkle
coverage, catalog freshness, and data integrity. Fix any reported
issues before publishing.

---

## Part 4: Publishing to S3

### Configure the Publish Target

Tell veks where this dataset should be published. The `.publish_url`
file is placed at the root of your publish tree (which may be above
the dataset directory):

```shell
echo 's3://my-bucket/datasets/' > .publish_url
```

### Mark the Dataset for Publishing

The `.publish` sentinel file opts a dataset into the publish set:

```shell
touch .publish
```

Datasets without `.publish` are local-only and silently excluded
from publishing and catalog completeness checks.

### Generate Catalogs

```shell
veks prepare catalog generate .
```

This scans for publishable datasets and writes `catalog.json` and
`catalog.yaml` at every directory level from each dataset up to the
publish root.

### Final Check

```shell
veks prepare check
```

Verify everything is green — all pipeline steps complete, merkle
files present, catalogs fresh.

### Preview the Upload

```shell
veks prepare publish --dry-run
```

Review the publish summary: source, destination, dataset list with
file counts and sizes. No data is transferred.

### Publish

```shell
veks prepare publish
```

The summary is shown and you're prompted to type `YES` to confirm.
Veks syncs only the publishable files (exclude-all, include-only
strategy) using `aws s3 sync`.

After publishing, anyone who adds your catalog URL can browse and
access the dataset:

```shell
veks datasets config add-catalog https://my-bucket.s3.amazonaws.com/datasets/
veks datasets list
veks interact explore --dataset my-dataset
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `cargo install veks` |
| Tab completion | `eval "$(veks completions)"` |
| Add catalog | `veks datasets config add-catalog <URL>` |
| List datasets | `veks datasets list` |
| Explore interactively | `veks interact explore --dataset <name>` |
| Bootstrap new dataset | `veks prepare bootstrap --interactive` |
| Run pipeline | `veks prepare run` |
| Add sized profiles | `veks prepare stratify && veks prepare run` |
| Pre-flight check | `veks prepare check` |
| Generate catalogs | `veks prepare catalog generate .` |
| Publish | `veks prepare publish` |
| Pipeline command help | `veks help <command>` |
