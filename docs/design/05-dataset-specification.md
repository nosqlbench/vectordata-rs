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
  run: <string>          # Command path, e.g., "import" (required)
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
      run: import
      output: all_vectors.hvec

    - id: extract-base        # default only
      run: transform hvec-extract
      profiles: [default]
      output: base_vectors.hvec
      range: "[10000,${vector_count})"

    - id: extract-base-10M    # 10M only
      run: transform hvec-extract
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

A profile is a named mapping from facet keys to data views. The `default`
profile is conventional. Profiles support a rich sugar syntax for concise
specification.

### Profile group behavior

- **Default inheritance**: All non-default profiles automatically inherit
  views and `maxk` from the `default` profile. Explicitly declared views
  in a child profile override inherited ones.
- **`base_count`** is never inherited — it must be explicitly set per profile.
- Custom (non-standard) facet keys are preserved and inherited like
  standard facets.

### Facet key aliases

Profile view keys can use shorthand aliases that resolve to canonical names:

| Alias | Canonical name |
|-------|---------------|
| `base`, `train` | `base_vectors` |
| `query`, `queries`, `test` | `query_vectors` |
| `indices`, `neighbors`, `ground_truth`, `gt` | `neighbor_indices` |
| `distances` | `neighbor_distances` |
| `content`, `meta_content`, `meta_base` | `metadata_content` |
| `meta_predicates` | `metadata_predicates` |
| `meta_results`, `predicate_results` | `metadata_results` |
| `layout`, `meta_layout` | `metadata_layout` |
| `filtered_indices`, `filtered_gt`, `filtered_ground_truth` | `filtered_neighbor_indices` |
| `filtered_distances`, `filtered_neighbors` | `filtered_neighbor_distances` |

Any key that does not match a standard facet or alias is treated as a
custom facet and preserved as-is.

### View sugar forms

Each profile view value is a map with a `source` key identifying the file,
plus optional `namespace` and `window` keys:

**Canonical map form**:
```yaml
base_vectors:
  source: base.fvec

base_vectors:
  source: base.fvec
  window: "0..1M"

metadata_content:
  source: metadata.slab
  namespace: content
  window: "0..10K"
```

The `source` key may also be written as `path`, and `namespace` as `ns`.

**Sugar: bare string** — a plain string value is shorthand for `{ source: <string> }`:
```yaml
# These two are equivalent:
base_vectors: base.fvec
base_vectors:
  source: base.fvec
```

**Sugar: inline window** — bracket or paren notation on the string embeds
the window directly:
```yaml
# These are equivalent:
base_vectors: "base.fvec[0..1M]"
base_vectors:
  source: base.fvec
  window: "0..1M"
```

Paren notation also works: `"base.fvec(0..1000)"`.

**Sugar: inline namespace** — colon-delimited namespace:
```yaml
# These are equivalent:
metadata_content: "metadata.slab:content"
metadata_content:
  source: metadata.slab
  namespace: content
```

**Combined sugar** — namespace + window:
```yaml
metadata_content: "metadata.slab:content:[0..10K]"
```

### Window syntax

Windows specify record ranges as half-open intervals `[min, max)`.

**Range forms**:

| Form | Meaning |
|------|---------|
| `0..1000` | `[0, 1000)` |
| `[0..1000)` | `[0, 1000)` — explicit exclusive end |
| `[0..1000]` | `[0, 1001)` — inclusive end |
| `(5..10)` | `[6, 10)` — exclusive start |
| `[10k..]` | `[10000, MAX)` — open-ended right |
| `[..10k)` | `[0, 10000)` — open-ended left, exclusive end |
| `[..10k]` | `[0, 10001)` — open-ended left, inclusive end |
| `[..]` | `[0, MAX)` — all elements |
| `1M` | `[0, 1000000)` — single number shorthand |

**Number suffixes**:

| Suffix | Multiplier |
|--------|-----------|
| `K`, `k` | ×1,000 |
| `M`, `m` | ×1,000,000 |
| `B`, `b`, `G`, `g` | ×1,000,000,000 |
| `T`, `t` | ×1,000,000,000,000 |
| `KB`, `MB`, `GB`, `TB` | SI decimal (same as above) |
| `KiB`, `MiB`, `GiB`, `TiB` | IEC binary (1024-based) |

Underscore separators are allowed: `1_000_000`.

**Multi-interval windows** (comma-separated or YAML list):
```yaml
window: "[0..1K, 2K..3K]"
# or as YAML list:
window:
  - "0..1K"
  - "2K..3K"
```

**YAML number shorthand** — a bare number is treated as `[0, N)`:
```yaml
window: 1000000
```

### Profile fields

| Field | Type | Description |
|-------|------|-------------|
| `maxk` | int | Maximum k for KNN queries (inherited from default) |
| `base_count` | int/suffixed | Base vector count for sized profiles (not inherited). Accepts suffix notation: `1M`, `10M` |
| *(view keys)* | string or map | Data views, keyed by facet name or alias |

### Example: canonical and sugared forms

The canonical (map) form and sugared (string) forms can be mixed freely:

```yaml
profiles:
  default:
    maxk: 100
    base_vectors:                     # canonical map form
      source: base_vectors.fvec
    query_vectors: query_vectors.fvec # sugar: bare string
    indices: gnd/idx.ivecs            # sugar: alias "indices" → neighbor_indices
    distances: gnd/dis.fvecs          # sugar: alias "distances" → neighbor_distances
    model_profile: model.json         # custom facet (preserved as-is)
  1M:
    base_count: 1M
    base_vectors:                     # canonical map form with window
      source: base_vectors.fvec
      window: "0..1M"
    neighbor_indices: gnd/idx_1M.ivecs
    neighbor_distances: gnd/dis_1M.fvecs
    # query_vectors, model_profile inherited from default
```

The same 1M profile using sugar throughout:

```yaml
  1M:
    base_count: 1M
    base: "base_vectors.fvec[0..1M]"  # alias + inline window
    indices: gnd/idx_1M.ivecs
    distances: gnd/dis_1M.fvecs
```

Profiles enable multiple slices of the same dataset (e.g., different query
subsets, different predicate sets) to coexist with different ground truths.

### Sized profiles with windows

When multiple profiles share the same base vectors file but use different
counts, sized profiles can reference the default profile's file with a
**window** instead of extracting separate copies. This saves disk space and
enables partial downloads via merkle-based transfer.

```yaml
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    query_vectors: profiles/default/query_vectors.fvec
    neighbor_indices: profiles/default/neighbor_indices.ivec
  1m:
    base_count: 1000000
    base_vectors: "profiles/default/base_vectors.fvec[0..1000000)"
    query_vectors: profiles/default/query_vectors.fvec
    neighbor_indices: profiles/1m/neighbor_indices.ivec
  10m:
    base_count: 10000000
    base_vectors: "profiles/default/base_vectors.fvec[0..10000000)"
    query_vectors: profiles/default/query_vectors.fvec
    neighbor_indices: profiles/10m/neighbor_indices.ivec
```

The window syntax `[0..1000000)` specifies a half-open record range. Suffix
notation is supported: `[0..1M)`, `[0..10M)`. All commands that read vector
files (`compute knn`, `compute filtered-knn`, `analyze verify-knn`,
`analyze stats`, etc.) honor window specifications in source paths.

When `per_profile` steps are expanded, sized profiles automatically receive
window-based views referencing the default profile's files rather than
creating separate extracted copies.

## 5.4 Upstream Pipeline Patterns

### Basic KNN dataset

```yaml
upstream:
  steps:
    - id: import-base
      run: import
      facet: base_vectors
      source: raw/base.npy
      from: npy
      output: base_vectors.fvec

    - id: import-query
      run: import
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
      run: import
      facet: metadata_content
      source: raw/metadata
      from: parquet
      output: metadata_content.slab

    - id: analyze-metadata
      run: survey
      after: [import-metadata]
      input: metadata_content.slab
      output: metadata_survey.json
      samples: 10000
      max-distinct: 100

    - id: synthesize-predicates
      run: synthesize predicates
      after: [analyze-metadata]
      input: metadata_content.slab
      survey: metadata_survey.json
      output: predicates.slab
      count: 10000
      selectivity: 0.0001
      seed: 42

    - id: compute-predicates
      run: evaluate predicates
      after: [import-metadata, synthesize-predicates]
      input: metadata_content.slab
      predicates: predicates.slab
      output: metadata_indices.slab

    - id: compute-filtered-knn
      run: compute filtered-knn
      after: [import-base, import-query, compute-predicates]
      base: base_vectors.hvec
      query: query_vectors_10k.hvec
      metadata-indices: metadata_indices.slab
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
│   └── compute-predicates.seg_*    # Predicate evaluation segment caches
├── base_vectors.hvec               # Output facets
├── query_vectors.hvec
├── neighbor_indices.ivec
├── neighbor_distances.fvec
├── metadata_content.slab
├── metadata_survey.json
├── predicates.slab
├── metadata_indices.slab
├── filtered_neighbor_indices.ivec
└── filtered_neighbor_distances.fvec
```

### Scratch vs cache semantics

**Scratch** (`${scratch}`) holds truly temporary data — intermediate
format conversions, working copies, etc. Scratch is cleaned after a
successful pipeline run.

**Cache** (`${cache}`) holds expensive-to-compute intermediates that are
not part of the final hosted dataset but remain valuable in-situ. Examples
include KNN partition results and predicate evaluation segments, each
of which may represent hours of computation over hundreds of GB of input
data. Cache data is not carried into hosted views, but future pipeline
re-executions (to fill in missing pieces or extend the dataset) can skip
already-cached segments. Deleting cache forces full recomputation of all
intermediate stages and should be treated as a significant operational
decision, not routine cleanup.
