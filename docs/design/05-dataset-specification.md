<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 05 — Dataset Specification

## 5.1 dataset.yaml Schema

The dataset.yaml file is the canonical descriptor for a vector dataset.
It defines provenance, upstream preparation steps, and output profiles.

### Top-level fields

```yaml
name: <string>           # Dataset identifier
description: <string>    # Human-readable description

upstream:                 # Optional: data preparation pipeline
  steps: [...]

profiles:                 # Named facet mappings
  default:
    base_vectors: <path>
    query_vectors: <path>
    ...
```

### Step definition

```yaml
- id: <string>           # Unique step identifier (optional, derived from run)
  run: <string>          # Command path, e.g., "import facet" (required)
  description: <string>  # Human-readable description of this step (optional)
  after: [<string>, ...] # Dependency step IDs (optional)
  profiles: [<string>, ...]  # Profile gate (optional, see below)
  on_partial: restart|resume  # What to do with partial artifacts (optional)

  # All remaining fields are command-specific options:
  source: <path>
  output: <path>
  facet: <facet_key>
  from: <format>
  count: <int>
  metric: <enum>
  neighbors: <int>
  # ... etc.
```

The `description` field is purely informational. When present, it is printed
to stderr before the step executes, providing context in pipeline logs.

### Profile-gated steps

Steps may include a `profiles` field listing which profiles they belong to.
When running with `--profile <name>`:

- Steps with **no** `profiles` field are **shared** — they always run.
- Steps with a `profiles` list run **only** when the active profile is in the list.

This enables a single `dataset.yaml` to define both the full pipeline and
smaller subset pipelines (e.g., 10M from a 400M corpus) that share common
import/shuffle steps but produce profile-specific outputs.

```yaml
upstream:
  steps:
    - id: import-all          # shared: runs for every profile
      run: import facet
      output: all_vectors.hvec

    - id: extract-base        # default only
      run: generate hvec-extract
      profiles: [default]
      output: base_vectors.hvec
      range: "[10000,${vector_count})"

    - id: extract-base-10M    # 10M only
      run: generate hvec-extract
      profiles: [10M]
      output: 10M/base_vectors.hvec
      range: "[10000,10010000)"
```

Running `veks run dataset.yaml` (defaults to `--profile default`) executes
shared steps plus default-gated steps. Running `veks run dataset.yaml --profile 10M`
executes shared steps plus 10M-gated steps.

The `profile_name` and `profile_dir` variables are automatically injected
into the defaults for use in step options (`${profile_name}`, `${profile_dir}`).

### Variable interpolation

Step options support `${variable}` references:

| Variable | Resolved to |
|----------|-------------|
| `${scratch}` | Workspace `.scratch/` directory |
| `${cache}` | Workspace `.cache/` directory |

Custom variables can be provided via the `defaults` mechanism.

## 5.2 Facet Keys

Recognized facet keys and their preferred output formats:

| Facet key | Preferred format | Description |
|-----------|-----------------|-------------|
| `base_vectors` | fvec/hvec | Corpus vectors |
| `query_vectors` | fvec/hvec | Query vectors |
| `neighbor_indices` | ivec | Ground-truth neighbor indices |
| `neighbor_distances` | fvec | Ground-truth neighbor distances |
| `metadata_content` | slab (MNode) | Per-vector metadata records |
| `metadata_predicates` | slab (PNode) | Per-query predicate trees |
| `metadata_layout` | slab | Metadata field schema |
| `predicate_results` | slab | Predicate evaluation bitmaps |
| `filtered_neighbor_indices` | ivec | Filtered ground-truth indices |
| `filtered_neighbor_distances` | fvec | Filtered ground-truth distances |

## 5.3 Profiles

A profile is a named mapping from facet keys to file paths. The `default`
profile is conventional.

```yaml
profiles:
  default:
    base_vectors: base_vectors.hvec
    query_vectors: query_vectors.hvec
    neighbor_indices: neighbor_indices.ivec
    neighbor_distances: neighbor_distances.fvec
    metadata_content: metadata_content.slab
```

Profiles enable multiple slices of the same dataset (e.g., different query
subsets, different predicate sets) to coexist with different ground truths.

## 5.4 Upstream Pipeline Patterns

### Basic KNN dataset

```yaml
upstream:
  steps:
    - id: import-base
      run: import facet
      facet: base_vectors
      source: raw/base.npy
      from: npy
      output: base_vectors.fvec

    - id: import-query
      run: import facet
      facet: query_vectors
      source: raw/queries.npy
      from: npy
      output: query_vectors.fvec

    - id: compute-knn
      run: compute knn
      after: [import-base, import-query]
      base: base_vectors.fvec
      query: query_vectors.fvec
      indices: neighbor_indices.ivec
      distances: neighbor_distances.fvec
      neighbors: 100
      metric: L2
```

### With format conversion

```yaml
    - id: upcast-base
      run: convert file
      after: [import-base]
      source: base_vectors.hvec
      output: ${scratch}/upcast-base.fvec
      to: fvec
```

### Predicated dataset (full pipeline)

```yaml
    - id: import-metadata
      run: import facet
      facet: metadata_content
      source: raw/metadata
      from: parquet
      output: metadata_content.slab

    - id: analyze-metadata
      run: slab survey
      after: [import-metadata]
      input: metadata_content.slab
      output: metadata_survey.json
      samples: 10000
      max-distinct: 100

    - id: generate-predicates
      run: generate predicates
      after: [analyze-metadata]
      input: metadata_content.slab
      survey: metadata_survey.json
      output: predicates.slab
      count: 10000
      selectivity: 0.0001
      seed: 42

    - id: compute-predicates
      run: generate predicate-keys
      after: [import-metadata, generate-predicates]
      input: metadata_content.slab
      predicates: predicates.slab
      output: predicate_keys.slab

    - id: compute-filtered-knn
      run: compute filtered-knn
      after: [import-base, import-query, compute-predicates]
      base: base_vectors.hvec
      query: query_vectors_10k.hvec
      predicate-keys: predicate_keys.slab
      indices: filtered_neighbor_indices.ivec
      distances: filtered_neighbor_distances.fvec
      neighbors: 100
      metric: L2
```

## 5.5 Skip-if-Fresh Semantics

The pipeline automatically skips steps whose outputs are already complete:

1. **Progress log check**: If `.upstream.progress.yaml` records the step
   as OK with matching options and output sizes, skip immediately.

2. **Artifact check**: If the progress log has no record but the output
   file exists and passes format-specific completeness checks, record as
   complete and skip.

3. **Partial handling**: If an output exists but is incomplete:
   - `on_partial: restart` (default) — delete and re-run
   - `on_partial: resume` — pass to command for continuation

This enables idempotent re-runs and crash recovery.

## 5.6 Bulk Download Configuration

Separate from dataset.yaml, bulk download configs drive the `veks bulkdl`
command:

```yaml
datasets:
  - name: text_emb
    baseurl: 'https://deploy.laion.ai/.../text_emb_${number}.npy'
    tokens:
      number: [0..409]
    savedir: embeddings/text_emb/
    tries: 5
    concurrency: 5
```

Token expansion generates URLs from templates. Existing files are skipped
via HEAD probe size matching. Status files enable complete skip without
network probes.

## 5.7 Workspace Layout Convention

```
dataset-name/
├── dataset.yaml                    # Dataset descriptor
├── .upstream.progress.yaml         # Pipeline progress log
├── .scratch/                       # Temporary files (disposable after success)
├── .cache/                         # Persistent intermediates (NOT disposable)
│   ├── compute-knn.part_*.cache    # KNN partition caches
│   └── compute-predicates.seg_*    # Predicate-key segment caches
├── base_vectors.hvec               # Output facets
├── query_vectors.hvec
├── neighbor_indices.ivec
├── neighbor_distances.fvec
├── metadata_content.slab
├── metadata_survey.json
├── predicates.slab
├── predicate_keys.slab
├── filtered_neighbor_indices.ivec
└── filtered_neighbor_distances.fvec
```

### Scratch vs cache semantics

**Scratch** (`${scratch}`) holds truly temporary data — intermediate
format conversions, working copies, etc. Scratch is cleaned after a
successful pipeline run.

**Cache** (`${cache}`) holds expensive-to-compute intermediates that are
not part of the final hosted dataset but remain valuable in-situ. Examples
include KNN partition results and predicate-key segment evaluations, each
of which may represent hours of computation over hundreds of GB of input
data. Cache data is not carried into hosted views, but future pipeline
re-executions (to fill in missing pieces or extend the dataset) can skip
already-cached segments. Deleting cache forces full recomputation of all
intermediate stages and should be treated as a significant operational
decision, not routine cleanup.
