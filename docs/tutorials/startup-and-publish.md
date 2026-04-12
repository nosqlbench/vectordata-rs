<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Getting Started — From Install to Publish

This tutorial walks through the full lifecycle: installing veks,
browsing remote datasets, preparing a new dataset from source files,
and publishing it.

---

## Part 1: Installation

### Prerequisites

```shell
sudo apt update
sudo apt install gcc pkg-config libssl-dev
```

### Install Rust and Veks

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
cargo install veks
```

### Enable Tab Completion

```shell
eval "$(veks completions)"
echo 'eval "$(veks completions)"' >> ~/.bashrc   # make permanent
```

Tab completion provides context-aware suggestions for commands,
options, dataset names, and catalog indexes.

---

## Part 2: Browsing Datasets

### Add a Catalog

```shell
veks datasets config add-catalog https://example.com/datasets/
```

### List and Explore

```shell
veks datasets list
veks datasets list --matching-name "sift*"
veks datasets probe --dataset my-dataset
```

### Prebuffer for Offline Use

```shell
veks datasets prebuffer --dataset my-dataset
```

Downloads all facets into `~/.cache/vectordata/my-dataset/`, verified
against merkle hashes. Configure the cache location:

```shell
veks datasets config set-cache /mnt/fast-storage/vd-cache
```

---

## Part 3: Preparing a Dataset

### Gather Source Files

Start with a directory containing your source files:

```
my-dataset/
├── _base_vectors.fvec       # base corpus (underscore = source, not published)
├── _query_vectors.fvec      # query vectors
└── _groundtruth.ivecs       # optional pre-computed ground truth
```

### Run the Bootstrap Wizard

```shell
cd my-dataset
veks bootstrap -i
```

The wizard:

1. **Detects files** — assigns roles by filename keywords (base,
   query, gt, metadata, etc.)
2. **Shows facets** — which facets will be produced (BQGDMPRF)
3. **Configures** — metric (L2/Cosine/DotProduct), normalization,
   metadata synthesis, predicate selectivity
4. **Renames sources** — offers to underscore-prefix detected and
   unassigned files
5. **Generates** — creates `dataset.yaml` with the full pipeline

### Run the Pipeline

```shell
veks run dataset.yaml
```

The pipeline processes in phases:

| Phase | What happens |
|-------|-------------|
| 0 | Vector preparation (dedup, normalize, extract), KNN computation |
| 1 | Metadata/predicate evaluation, SQLite verification |
| 2 | Filtered KNN computation |
| — | Finalize: dataset.json, vvec indices, merkle trees, catalog |

The pipeline is resumable — re-run to pick up where it left off.
Use `--clean` to start fresh (preserves source symlinks).

### Analyze Results

```shell
veks analyze describe --source profiles/base/base_vectors.fvec
veks analyze explain-predicates --ordinal 42
veks analyze explain-filtered-knn --ordinal 42
```

### Verify Readiness

```shell
veks check
```

Runs all pre-flight checks: pipeline completeness, merkle coverage,
catalog freshness, data integrity, required attributes.

---

## Part 4: Publishing

### Configure the Publish Target

```shell
echo 's3://my-bucket/datasets/' > .publish_url
touch .publish
```

### Publish

```shell
veks publish --dry-run    # preview
veks publish              # upload (prompts for confirmation)
```

After publishing, others can access the dataset:

```shell
veks datasets config add-catalog https://my-bucket.s3.amazonaws.com/datasets/
veks datasets list
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `cargo install veks` |
| Tab completion | `eval "$(veks completions)"` |
| Add catalog | `veks datasets config add-catalog <URL>` |
| List datasets | `veks datasets list` |
| Probe dataset | `veks datasets probe --dataset <name>` |
| Prebuffer | `veks datasets prebuffer --dataset <name>` |
| Bootstrap | `veks bootstrap -i` |
| Run pipeline | `veks run dataset.yaml` |
| Clean run | `veks run dataset.yaml --clean` |
| Dry run | `veks run dataset.yaml --dry-run` |
| Pre-flight check | `veks check` |
| Publish | `veks publish` |
| Explain predicates | `veks analyze explain-predicates --ordinal 0` |
| Explain filtered KNN | `veks analyze explain-filtered-knn --ordinal 0` |
