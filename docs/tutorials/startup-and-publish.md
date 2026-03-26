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

This section covers taking raw source files (base vectors, metadata)
and producing a publish-ready dataset with stratified profiles.

### Gather Source Files

Start with a clean directory containing your source files. Use clear
names so the bootstrap wizard can identify each file's role:

```
my-dataset/
├── _base_vectors/          # npy directory with base corpus vectors
├── _metadata/              # parquet directory with metadata
└── (optional: _gt.ivec)    # pre-computed neighbor indices
```

Underscore-prefixed names (`_base_vectors`) are treated as source
files and excluded from publishing.

### Run the Bootstrap Wizard

```shell
cd my-dataset
veks bootstrap --interactive
```

The wizard walks through these steps:

1. **File detection** — scans the directory for recognized data files
   and assigns roles (base vectors, metadata, etc.)

2. **Facet selection** — shows which dataset facets will be produced
   based on detected inputs, using SRD §2.8 implication rules:
   ```
   [x] B  base vectors
   [x] Q  query vectors
   [x] G  KNN ground-truth indices
   [x] D  KNN ground-truth distances
   [x] M  metadata
   [x] P  predicates
   [x] R  predicate results
   [x] F  filtered KNN ground-truth
   ```
   Accept the defaults or type a custom facet string (e.g., `BQGD`
   to skip metadata).

3. **Base data fraction** — percentage of source vectors to use.
   This is immutable after the first run (see SRD §3.13). Use 100%
   for production, lower for fast iteration during development.

4. **Distance metric** — L2, Cosine, or DotProduct. The wizard
   probes your vectors to detect normalization:
   - If vectors are L2-normalized, defaults to DotProduct (fastest
     kernel, no norm computation needed)
   - If not normalized and metric is Cosine/DotProduct, recommends
     L2-normalization during extraction

5. **Normalization** — when the metric requires it (Cosine/DotProduct
   on unnormalized data), the wizard recommends normalizing. This
   enables the 16-wide AVX-512 batched kernel for KNN computation.
   Without normalization, the full cosine kernel with norm division
   is used (correct but slightly slower).

6. **Predicate selectivity** — when metadata is present, controls
   how selective the synthesized predicates are (0.0001 = very hard,
   0.1 = easy).

7. **Sized profiles** — generates stratified profiles at multiple
   scales using `${base_count}` as the upper bound. The profiles
   are resolved at run time after the pipeline determines the actual
   base vector count:
   ```
   Sized profile spec: mul:1m..${base_count}/2
   ```
   This produces profiles at 1M, 2M, 4M, 8M, ... up to the base count.

8. **Summary** — shows a colored artifact lineage view mapping inputs
   to facets:
   ```
   _base_img_emb (npy dir)    ──convert──► B  base vectors    [convert]
   _base_img_emb (npy dir)    ───split───► Q  query vectors   [self-search]
   B x Q brute-force          ──compute──► G  KNN indices      [compute]
   _metadata (parquet dir)    ──convert──► M  metadata         [convert]
   ```

The wizard creates `dataset.yaml` with a complete pipeline definition.

### Run the Pipeline

```shell
veks run
```

This executes the pipeline defined in `dataset.yaml`. The pipeline
runs in phases optimized for I/O cache coherence:

| Phase | Steps | I/O pattern |
|-------|-------|-------------|
| 0 | compute-knn (all profiles) | Base vectors sequential scan |
| 1 | evaluate-predicates (all profiles) | Metadata I/O |
| 2 | compute-filtered-knn (all profiles) | Base + predicate indices |
| 3 | verify-knn (consolidated) | Light sampling, single scan |
| 4 | verify-predicates (consolidated) | Light sampling |
| — | generate-catalog, generate-merkle | Final indexing |

Within each phase, profiles process smallest to largest. KNN cache
segments from smaller profiles are automatically reused by larger
profiles (e.g., the 1M profile's partition [0,1M) is reused by the
2M, 4M, 8M, ... profiles).

The pipeline is resumable — if interrupted, re-run `veks run` to pick
up where it left off. Per-step fingerprints detect configuration
changes and re-execute only affected steps.

### Output Modes

```shell
veks run                     # auto-detect: TUI if terminal, basic otherwise
veks run --output tui        # rich TUI with progress bars and throughput charts
veks run --output basic      # plain-text progress on stderr (good for logging)
veks run --output batch      # log-only, no console output (for CI/scripts)
```

### Stratification

Sized profiles can be added at any time — either at bootstrap (via
the wizard) or after the first run:

```shell
# Option A: included in bootstrap (recommended)
veks bootstrap --interactive   # wizard asks about sized profiles

# Option B: add after first run
veks prepare stratify          # adds profiles based on actual base count
veks run                       # computes per-profile KNN (cached segments reused)
```

Adding profiles never invalidates core pipeline steps. Profile
expansion is dynamic — only per-profile steps are added.

### Verify Readiness

```shell
veks check
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

### Final Check

```shell
veks check
```

Verify everything is green — all pipeline steps complete, merkle
files present, catalogs fresh.

### Preview the Upload

```shell
veks publish --dry-run
```

Review the publish summary: source, destination, dataset list with
file counts and sizes. No data is transferred.

### Publish

```shell
veks publish
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
| Bootstrap new dataset | `veks bootstrap --interactive` |
| Bootstrap (auto, no prompts) | `veks bootstrap --auto` |
| Run pipeline | `veks run` |
| Run with basic output | `veks run --output basic` |
| Dry-run (show plan) | `veks run --dry-run` |
| Add sized profiles | `veks prepare stratify` |
| Pre-flight check | `veks check` |
| Publish (dry-run) | `veks publish --dry-run` |
| Publish | `veks publish` |
| Pipeline command help | `veks help <command>` |
