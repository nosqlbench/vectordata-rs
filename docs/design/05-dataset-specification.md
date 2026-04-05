<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 05 — Dataset Specification

This document is the normative reference for implementing dataset access in any
language or system. It specifies the `dataset.yaml` manifest format, the profile
resolution algorithm, the data file formats, and the protocol for local and
remote dataset access. A conforming implementation must be able to load a
`dataset.yaml`, resolve any named profile, and read the data files referenced by
that profile's views — whether those files are on a local filesystem or served
over HTTP.

---

## 5.1 dataset.yaml Schema

The `dataset.yaml` file is the canonical descriptor for a vector dataset.
It defines provenance, upstream preparation steps, and output profiles.

### Top-level fields

```yaml
name: <string>           # Dataset identifier (required)
description: <string>    # Human-readable description (optional)

upstream:                 # Optional: data preparation pipeline
  steps: [...]

profiles:                 # Named facet mappings (required for access)
  default:
    base_vectors: <path>
    query_vectors: <path>
    ...
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Dataset identifier |
| `description` | string | No | Human-readable description |
| `upstream` | object | No | Pipeline configuration for producing data files |
| `profiles` | map | Yes | Named profiles mapping facet keys to data views |

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
      output: all_vectors.mvec

    - id: extract-base        # default only
      run: transform mvec-extract
      profiles: [default]
      output: base_vectors.mvec
      range: "[10000,${vector_count})"

    - id: extract-base-10M    # 10M only
      run: transform mvec-extract
      profiles: [10M]
      output: 10M/base_vectors.mvec
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
| `base_vectors` | fvec/mvec | Corpus vectors |
| `query_vectors` | fvec/mvec | Query vectors |
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

### Sized profile expansion (`sized` key)

Instead of declaring each sized profile individually, use the `sized` key
with a list of size specifications:

```yaml
profiles:
  default:
    maxk: 100
    query_vectors: query_vectors.mvec
  sized: [10m, 20m, 100m..400m/100m]
```

This expands to profiles named `10m`, `20m`, `100m`, `200m`, `300m`, `400m`,
each with `base_count` set and inheriting from `default`. Profiles are sorted
smallest to largest.

The `sized` key is reserved — it is not a profile name. It may appear
alongside explicit profile definitions and the `default` profile.

#### Sized entry forms

Each entry in the `sized` list is a string parsed into one or more
`(profile_name, base_count)` pairs. Five forms are supported:

**1. Simple value** — a single count:
```yaml
sized: [10m, 20m, 50m]
```
Each value becomes one profile. `10m` → profile named `"10m"` with
`base_count: 10000000`.

**2. Linear range with step size** (`start..end/step`):
```yaml
sized: ["100m..400m/100m"]
```
The divisor has a **unit suffix** (like `m`, `k`), so it is interpreted as
an absolute step size. Produces a profile at each step from start to end
inclusive: `100m, 200m, 300m, 400m`. If end is not exactly reachable, the
last value that does not exceed end is included.

Another example: `"10m..25m/10m"` → `10m, 20m` (30m would exceed 25m).

**3. Linear range with count** (`start..end/N`):
```yaml
sized: ["0m..400m/10"]
```
The divisor is a **bare integer** (no unit suffix), so it is interpreted as
the number of equal divisions. The range `[start, end]` is divided into N
equal parts, and profiles are created at each division boundary (excluding
start if start is 0). Formula: `value[i] = start + (range × i) / N` for
`i = 1..N`.

Produces: `40m, 80m, 120m, 160m, 200m, 240m, 280m, 320m, 360m, 400m`.

With nonzero start: `"100m..400m/3"` → `200m, 300m, 400m` (3 divisions
of the 300m range).

**The critical distinction**: `100m..400m/10m` (step of 10 million)
produces 31 profiles (100m, 110m, 120m, ..., 400m), while
`100m..400m/10` (10 divisions) produces 10 profiles (130m, 160m, ...,
400m). The presence or absence of a unit suffix on the divisor determines
which interpretation is used.

**4. Fibonacci series** (`fib:start..end`):
```yaml
sized: ["fib:1m..400m"]
```
Uses `start` as the base unit. Generates profiles at fibonacci multiples
of that unit within the range: `fib(n) × start` for each fibonacci
number where the result falls within `[start, end]`.

With base unit 1m, the fibonacci sequence 1, 1, 2, 3, 5, 8, 13, 21, 34,
55, 89, 144, 233, 377 produces: `1m, 1m, 2m, 3m, 5m, 8m, 13m, 21m,
34m, 55m, 89m, 144m, 233m, 377m`. Start must be > 0.

**5. Geometric (multiply-by-factor) series** (`mul:start..end/factor` or `mul:start/factor`):
```yaml
sized: ["mul:1m..100m/2"]
```
Each successive value is `floor(prev × factor)`. The factor must be > 1.0
and can be fractional. Produces: `1m, 2m, 4m, 8m, 16m, 32m, 64m`.

With a fractional factor: `"mul:10m..100m/1.5"` → `10m, 15m, 22500k,
33750k, 50625k, 75937500` (values are `floor(prev × 1.5)`, named with
the largest clean suffix).

**Implicit upper bound form** (`mul:start/factor`): When the `..end` is
omitted, the upper bound is the actual base vector count of the dataset.
This is the preferred form for early stratification:
```yaml
sized: ["mul:1m/2"]
```
Generates profiles `1m, 2m, 4m, 8m, ...` stopping before the actual
base count. A safety cap of 100 profiles prevents runaway expansion.

#### Base count requirement

Profile expansion **requires** the actual base vector count to be known.
Every generator checks each candidate value against this bound and stops
producing profiles when a value would equal or exceed the base count
(such profiles are redundant with the default profile). **Post-hoc
filtering of oversized profiles is not allowed** — the invariant is
enforced at generation time.

Entries that use the implicit upper bound form (`mul:start/factor`) are
automatically deferred during YAML deserialization and expanded later
when `base_count` becomes available (after core pipeline stages produce
`variables.yaml`). Entries with explicit ranges (`mul:1m..100m/2`,
`fib:1m..400m`, linear ranges) are also capped at `base_count` during
expansion.

If `base_count` is not available when expansion is attempted, deferred
entries remain unresolved and a warning is emitted. The pipeline must
produce `base_count` before per-profile steps can be generated.

This design means that verify-knn and other downstream steps can rely on
the invariant that every sized profile has `base_count < actual_base_count`.
No downstream code needs to validate or filter profiles — the invariant
is established at the single point of profile generation.

**Conflict with explicit profiles**: If a sized expansion produces a
profile name that conflicts with an explicitly defined profile, this is
a deserialization error. For example, defining both `sized: [10m]` and
an explicit `10m:` profile in the same `profiles:` block is invalid.

#### Profile naming

Profile names are derived from the `base_count` value using the largest
clean suffix: values divisible by 1B get `b` suffix, by 1M get `m`, by
1K get `k`, otherwise the raw number is used. Examples: `10000000` →
`"10m"`, `1500000` → `"1500k"`, `100000` → `"100k"`, `1234` → `"1234"`.

#### Variable references in sized specs (deprecated)

> **Note**: Variable references in sized specs (e.g., `${base_count}`)
> are deprecated. The preferred approach is `mul:1m/2`, which defers
> expansion until `base_count` is known and generates only valid profiles.
> The variable reference mechanism is retained for backward compatibility
> but should not be used in new configurations.

#### Structured sized form with facet templates

Instead of a simple list, `sized` can be a map with explicit facet
templates that control how views are constructed for each expanded
profile:

```yaml
profiles:
  default:
    maxk: 100
    query_vectors: query_vectors.mvec
  sized:
    ranges: ["10m", "20m"]
    facets:
      base_vectors: "base_vectors.mvec:${range}"
      metadata_content: "metadata.slab:${range}"
      neighbor_indices: "profiles/${profile}/neighbor_indices.ivec"
```

Template variables:
- `${profile}` — the expanded profile name (e.g., `"10m"`)
- `${range}` — a window spec `[0..base_count]` for that profile

This form is useful when different facets require different access
patterns — some windowed into a shared file, others stored per-profile.

#### Expansion ordering and inheritance

- All sized entries are expanded, then sorted by `base_count` ascending.
- Each expanded profile inherits views and `maxk` from the `default`
  profile (same rules as §5.3 Profile group behavior).
- `base_count` is set from the expanded value and is never inherited.
- Sized profiles coexist with explicitly declared profiles. Explicit
  profiles are not affected by the sized expansion.

#### Complete example

```yaml
profiles:
  default:
    maxk: 100
    base_vectors: base_vectors.mvec
    query_vectors: query_vectors.mvec
    neighbor_indices: neighbor_indices.ivec

  sized: [10m, 20m, 100m..300m/100m]

  custom-queries:
    query_vectors: alt_queries.mvec
    neighbor_indices: alt_indices.ivec
```

Expands to 7 profiles: `default`, `10m`, `20m`, `100m`, `200m`, `300m`,
`custom-queries`. The sized profiles inherit all views from default and
have `base_count` set. The explicit `custom-queries` profile is
unaffected by sized expansion.

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
      source: base_vectors.mvec
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
      run: compute predicates
      after: [import-metadata, synthesize-predicates]
      input: metadata_content.slab
      predicates: predicates.slab
      output: metadata_indices.slab

    - id: compute-filtered-knn
      run: compute filtered-knn
      after: [import-base, import-query, compute-predicates]
      base: base_vectors.mvec
      query: query_vectors_10k.mvec
      metadata-indices: metadata_indices.slab
      indices: filtered_neighbor_indices.ivec
      distances: filtered_neighbor_distances.fvec
      neighbors: 100
      metric: L2
```

## 5.5 Skip-if-Fresh Semantics

The pipeline automatically skips steps whose outputs are already complete:

1. **Progress log check**: If `.cache/.upstream.progress.yaml` records the step
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
├── .scratch/                       # Temporary files (disposable after success)
├── .cache/                         # Persistent intermediates (NOT disposable)
│   ├── .upstream.progress.yaml     # Pipeline progress log
│   ├── .governor.log               # Resource governor log (JSON-line)
│   ├── compute-knn.part_*.cache    # KNN partition caches
│   ├── compute-predicates.seg_*    # Predicate evaluation segment caches
│   ├── dedup_ordinals.ivec         # Sorted ordinal index (from compute dedup)
│   ├── dedup_duplicates.ivec       # Duplicate vector ordinals
│   ├── dedup_report.json           # Dedup statistics (total, unique, dup counts)
│   ├── dedup_runs/                 # Intermediate sorted run files (resumable)
│   │   ├── run_0000.bin
│   │   ├── run_0001.bin
│   │   └── meta.json               # Run parameters for resume validation
│   └── clean_ordinals.ivec         # Clean index (dedup + zero exclusion applied)
├── base_vectors.mvec               # Output facets
├── query_vectors.mvec
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

### Compressed cache files

Cache artifacts that are only accessed in sequential/streaming mode or
that fit in memory as individual segments may be stored as `.gz` files.
Compressed cache files are created by the `--compress-cache` option on
pipeline commands that support it, or retroactively by `veks datasets
cache-compress`.

**Eligible for compression** (sequential or in-memory segment access):
- Dedup sorted runs (`dedup_runs/run_*.bin.gz`)
- KNN partition caches (`*.neighbors.ivec.gz`, `*.distances.fvec.gz`)
- Predicate segment caches (`*.predkeys.slab.gz`)

**Not eligible** (require mmap random access from disk):
- `all_vectors.mvec` — random access by ordinal
- `shuffle.ivec` — random access by index
- `dedup_ordinals.ivec` — binary search
- `clean_ordinals.ivec` — random access
- Any final output file referenced by profile views

Files are compressed in memory before writing and decompressed in memory
after reading, using the `flate2` crate (gzip format). The `.gz` suffix
is appended to the original filename.

---

## 5.8 Implementor's Guide: Dataset Discovery and Access

This section specifies the exact algorithm that any client system must
follow to load a dataset, resolve a profile, and access data files. It is
the normative reference for implementing dataset access in any language.

### 5.8.1 Dataset discovery

A dataset is identified by a **root location** — either a local filesystem
directory or an HTTP(S) base URL. The manifest is always named
`dataset.yaml` within that root.

**Local filesystem**:
```
root = /path/to/dataset-name/
manifest = root + "dataset.yaml"
```

**Remote HTTP(S)**:
```
root = https://host/path/to/dataset-name/
manifest = root + "dataset.yaml"
```

If the caller provides a path ending in `.yaml` or `.yml`, treat it as the
manifest directly and derive the root as its parent directory/URL.

If the caller provides a bare directory path or URL (no `.yaml` suffix),
append `dataset.yaml` to locate the manifest. For URLs, ensure the path
ends with `/` before appending.

**Loading**:
1. Read the manifest content (local `read_to_string`, or HTTP GET).
2. Parse as YAML into the top-level schema (§5.1).
3. Resolve the `profiles` section using the profile resolution algorithm
   (§5.8.3).

### 5.8.2 Resolved view model

After parsing and profile resolution, each view resolves to this canonical
form:

```
ResolvedView {
    source_path: String,          // relative path to data file
    namespace:   Option<String>,  // slab namespace (if applicable)
    window:      List<Interval>,  // half-open record ranges; empty = all
}

Interval {
    min_incl: u64,                // inclusive lower bound
    max_excl: u64,                // exclusive upper bound
}
```

The `source_path` is always relative to the dataset root. To access the
file:
- **Local**: join the root directory path with `source_path`.
- **Remote**: resolve `source_path` against the root URL (standard URL
  resolution).

### 5.8.3 Profile resolution algorithm

Given a requested profile name (default: `"default"`), resolve a complete
set of views as follows:

```
function resolve_profile(profiles_map, requested_name) -> ResolvedProfile:

    1. ALIAS NORMALIZATION
       For every profile in profiles_map:
         For every view key in the profile:
           If the key matches a known alias (§5.3 Facet key aliases):
             Replace the key with the canonical facet name.
           Else:
             Preserve the key as-is (custom facet).

    2. SIZED EXPANSION (if "sized" key exists in profiles_map)
       Parse the sized value (list or structured map form, see §5.3).
       For each entry string, determine the expansion form:
         a. Prefix "fib:" → fibonacci series (§5.3 form 4)
         b. Prefix "mul:" → geometric series (§5.3 form 5)
         c. Contains "/" with suffixed divisor → linear step (§5.3 form 2)
         d. Contains "/" with bare-integer divisor → linear count (§5.3 form 3)
         e. Otherwise → simple value (§5.3 form 1)
       Expand each entry to (name, base_count) pairs.
       If structured form with "facets" templates: interpolate ${profile}
       and ${range} into each template per expanded profile.
       Sort all expanded pairs by base_count ascending.
       For each pair, create a profile with base_count set.
       Add all expanded profiles to profiles_map.
       Remove the "sized" key.

    3. SUGAR DESUGARING
       For every view value in every profile:
         If the value is a bare string:
           Parse using the sugar grammar (§5.3 View sugar forms):
             a. Check for bracket/paren window suffix → extract window
             b. Check for colon-delimited namespace → extract namespace
             c. Remaining text = source path
           Convert to canonical form: { source, namespace?, window? }
         If the value is a map:
           Accept "source" or "path" for the file path.
           Accept "namespace" or "ns" for the namespace.
           Parse "window" if present (string, number, or list form).

    4. INHERITANCE
       Let default_profile = profiles_map["default"] (may not exist).
       Let target = profiles_map[requested_name].
       If target is not "default" AND default_profile exists:
         For each view in default_profile:
           If target does NOT have a view with the same key:
             Copy the view into target (inherited).
         If target.maxk is not set AND default_profile.maxk is set:
           Copy default_profile.maxk into target.
         NOTE: base_count is NEVER inherited.

    5. WINDOW RESOLUTION
       For each view in target:
         If the view has a view-level window override:
           Use the view-level window (takes precedence).
         Else if the source has an inline window:
           Use the source window.
         Else:
           Window is empty (meaning: all records).

    6. Return the resolved profile with all views in canonical form.
```

**Ordering**: Steps 1-3 happen during YAML deserialization. Step 4 happens
as a post-deserialization pass. Step 5 is evaluated at access time.

### 5.8.4 Data file formats

All data files use simple binary layouts designed for O(1) random access
by record ordinal. The format is determined by file extension.

#### xvec family (fixed-dimension vectors)

Each record is:
```
[dimension: i32 LE][element_0 .. element_{dim-1}: T]
```

| Extension | Element type | Element size | Record size |
|-----------|-------------|--------------|-------------|
| `.fvec` | f32 (IEEE 754) | 4 bytes | `4 + dim × 4` |
| `.ivec` | i32 (signed) | 4 bytes | `4 + dim × 4` |
| `.bvec` | u8 | 1 byte | `4 + dim × 1` |
| `.dvec` | f64 (IEEE 754) | 8 bytes | `4 + dim × 8` |
| `.mvec` | f16 (IEEE 754) | 2 bytes | `4 + dim × 2` |
| `.svec` | i16 (signed) | 2 bytes | `4 + dim × 2` |

**Key properties**:
- The dimension prefix is repeated before every record (not just once at
  file start). Every record is self-describing.
- All multi-byte values are little-endian.
- Record count = `file_size / record_size`. If `file_size` is not evenly
  divisible, the file is corrupt.
- To read record at ordinal N: seek to `N × record_size`, read
  `record_size` bytes, extract dimension from first 4 bytes, then read
  `dim` elements.

**Cardinality and dimensionality semantics**:

A **zero-byte xvec file is valid** — it represents an empty record set
with cardinality 0. This is a defined result, not an error condition.
Pipeline steps that produce empty outputs (e.g., `analyze zeros` when
no zero vectors exist, or `clean-ordinals` when no records survive
filtering) write 0-byte files as their formal outputs.

When cardinality is 0, **dimensionality is undefined** — there are no
records from which to derive a dimension. Implementations must use a
sentinel value (e.g., `usize::MAX` or `DIM_UNDEFINED`) rather than 0,
because 0 could be confused with a degenerate zero-dimensional vector.
Callers must check `count > 0` before relying on `dim`. Attempting to
read a record from an empty file is an error.

**Dimension detection**: Read the first 4 bytes of the file as `i32 LE`.
This gives the dimension. Compute `record_size = 4 + dim × element_size`.
If the file is 0 bytes, skip dimension detection and report
count = 0, dim = undefined.
Compute `count = file_size / record_size`.

#### Slab format (.slab)

Page-aligned record container for variable-length records (MNode, PNode,
bitmaps, etc.). See document 02 for the full physical layout. Key points
for implementors:

- Records are addressed by sequential ordinal (0-based).
- The PagesPage (last page in file) contains the page index.
- Binary-search the page index to find which page holds ordinal N.
- Each page has a directory of `(offset, length)` entries for its records.
- Optional namespace support: a slab file may contain multiple independent
  record streams, each addressed by a string namespace.

#### MNode / PNode binary formats

MNode (metadata) and PNode (predicate) are self-describing binary records
stored inside slab files. See document 02 for wire format details.
Implementations that only need vector and index access can treat slab
facets as opaque and skip MNode/PNode decoding.

### 5.8.5 Windowed access

When a view specifies a window (list of half-open intervals), the
implementation must restrict record access to only the specified ranges.

**For xvec files**: Map the logical ordinal to the physical ordinal within
the windowed ranges. For a single interval `[min, max)`:
- Logical ordinal 0 maps to physical ordinal `min`.
- Logical ordinal `N` maps to physical ordinal `min + N`.
- Total logical count = `max - min`.
- Seek to `(min + N) × record_size` in the file.

**For multi-interval windows**: The logical ordinal space is the
concatenation of all intervals in order. To resolve logical ordinal `N`:
1. Walk intervals in order, accumulating counts.
2. When the cumulative count exceeds `N`, the current interval contains
   the target. Physical ordinal = `interval.min + (N - prior_count)`.
3. Total logical count = sum of `(max - min)` across all intervals.

**For slab files**: Windows restrict which ordinals are accessible.
Apply the same logical-to-physical mapping when requesting records.

### 5.8.6 Remote access protocol

Remote datasets are served as static files over HTTP(S). No special server
is required — any static file server or object store (S3, GCS, etc.)
that supports Range requests is sufficient.

#### Manifest retrieval

```
GET {root_url}/dataset.yaml
```

Returns the full YAML manifest. Parse identically to a local file.

#### Data file access

For each view's `source_path`, the data file URL is:
```
file_url = root_url + source_path
```

(Standard URL resolution. If `source_path` is relative, it resolves
against `root_url`.)

#### Dimension and count detection (xvec)

To determine dimension and count without downloading the full file:

1. **HEAD request** → get `Content-Length` header for total file size.
2. **Range request for header**:
   ```
   GET {file_url}
   Range: bytes=0-3
   ```
   Returns the first 4 bytes (dimension as `i32 LE`).
3. Compute `record_size = 4 + dim × element_size`.
4. Compute `count = content_length / record_size`.

#### Single-record fetch (xvec)

To read record at ordinal N:
```
byte_offset = N × record_size
GET {file_url}
Range: bytes={byte_offset}-{byte_offset + record_size - 1}
```

Implementations should batch nearby records into larger Range requests
when sequential access patterns are detected.

#### Connection pooling

HTTP clients must reuse connections (HTTP/1.1 keep-alive or HTTP/2
multiplexing). Creating a new TCP+TLS connection per record is not
acceptable for performance.

#### Integrity verification (Merkle)

For each data file, a companion `.mref` file may exist at `{file_url}.mref`
containing a Merkle tree of SHA-256 chunk hashes. See document 09 for the
Merkle wire format. When `.mref` files are present, implementations should:

1. Fetch `{file_url}.mref` and parse the Merkle reference tree.
2. Download data in chunk-sized units matching the tree's `chunk_size`.
3. Verify each chunk's SHA-256 hash against the corresponding leaf hash.
4. Track verification state locally in a `.mrkl` file for crash recovery.

This is optional for read-only clients that trust the transport layer
but required for caching clients that persist data locally.

### 5.8.7 Facet manifest and discovery

A conforming implementation should expose a **facet manifest** that lists
all facets in a resolved profile without materializing data. For each
facet, the manifest reports:

| Field | Description |
|-------|-------------|
| `name` | Canonical facet key (after alias resolution) |
| `source_path` | File path from the view |
| `source_type` | Inferred format (from file extension: `fvec`, `ivec`, `mvec`, `slab`, etc.) |
| `is_standard` | Whether this matches one of the 10 standard facets |

This enables "discover then load" patterns where a client enumerates
available facets before deciding which to materialize.

### 5.8.8 Implementation checklist

A conforming dataset access implementation must support:

| Capability | Required | Notes |
|------------|----------|-------|
| Parse `dataset.yaml` | Yes | Full schema per §5.1 |
| Alias resolution | Yes | All aliases per §5.3 |
| Sugar desugaring (bare string, inline window, inline namespace) | Yes | All forms per §5.3 |
| Profile inheritance from `default` | Yes | Views + `maxk`, not `base_count` |
| Window parsing (all range forms + suffixes) | Yes | All forms per §5.3 |
| Windowed record access | Yes | Single and multi-interval |
| xvec format reading (fvec, ivec at minimum) | Yes | mvec, bvec, dvec, svec recommended |
| Custom facet preservation | Yes | Non-standard keys pass through |
| Remote manifest fetch (HTTP GET) | Recommended | Required for remote datasets |
| Remote data access (HTTP Range) | Recommended | Required for remote datasets |
| Slab format reading | Optional | Required only for metadata facets |
| MNode/PNode decoding | Optional | Required only for predicated datasets |
| Merkle verification | Optional | Required for cached/offline access |
| Sized profile expansion | Optional | Required for `sized` key support |
| Facet manifest (discovery API) | Recommended | Enables generic tooling |

### 5.8.9 Reference implementations

| Language | Module | Scope |
|----------|--------|-------|
| Rust | `vectordata::dataset` module | YAML parsing, profiles, views, aliases, windows, sized expansion |
| Rust | `vectordata` crate | VectorReader trait, mmap/HTTP backends, facet manifest, Merkle |
| Java | `datatools-vectordata` | Full access layer with Merkle, caching, transport |

### 5.8.10 Worked example: resolving a profile

Given this `dataset.yaml`:

```yaml
name: example-768
description: 768-dim float vectors with metadata

profiles:
  default:
    maxk: 100
    base: base_vectors.fvec
    query: query_vectors.fvec
    indices: gnd/idx.ivec
    distances: gnd/dis.fvec
    metadata_content: "metadata.slab:content"
    model_config: model.json

  1m:
    base_count: 1M
    base: "base_vectors.fvec[0..1M]"
    indices: gnd/idx_1m.ivec
    distances: gnd/dis_1m.fvec
```

**Resolving profile `"1m"`**:

Step 1 — Alias normalization:
```
"base"      → "base_vectors"
"query"     → "query_vectors"
"indices"   → "neighbor_indices"
"distances" → "neighbor_distances"
"model_config" → preserved (custom facet)
```

Step 3 — Sugar desugaring:
```
default.base_vectors:      { source: "base_vectors.fvec" }
default.query_vectors:     { source: "query_vectors.fvec" }
default.neighbor_indices:  { source: "gnd/idx.ivec" }
default.neighbor_distances: { source: "gnd/dis.fvec" }
default.metadata_content:  { source: "metadata.slab", namespace: "content" }
default.model_config:      { source: "model.json" }

1m.base_vectors:           { source: "base_vectors.fvec", window: [0..1000000) }
1m.neighbor_indices:       { source: "gnd/idx_1m.ivec" }
1m.neighbor_distances:     { source: "gnd/dis_1m.fvec" }
```

Step 4 — Inheritance (1m inherits from default):
```
1m.base_vectors:            { source: "base_vectors.fvec", window: [0..1000000) }  — declared
1m.query_vectors:           { source: "query_vectors.fvec" }                        — inherited
1m.neighbor_indices:        { source: "gnd/idx_1m.ivec" }                           — declared (overrides)
1m.neighbor_distances:      { source: "gnd/dis_1m.fvec" }                           — declared (overrides)
1m.metadata_content:        { source: "metadata.slab", namespace: "content" }       — inherited
1m.model_config:            { source: "model.json" }                                — inherited (custom)
1m.maxk:                    100                                                     — inherited
1m.base_count:              1000000                                                 — declared (never inherited)
```

**Reading `base_vectors` for profile `1m`** (local):
```
file = /path/to/example-768/base_vectors.fvec
dim  = read_i32_le(file, offset=0)        → 768
record_size = 4 + 768 × 4                 → 3076 bytes
window = [0, 1000000)
logical_count = 1000000
To read logical ordinal 500:
  physical ordinal = 0 + 500 = 500
  seek to 500 × 3076 = 1538000
  read 3076 bytes
```

**Reading `base_vectors` for profile `1m`** (remote):
```
url = https://host/example-768/base_vectors.fvec

# Detect dimension
HEAD url → Content-Length: 307600000076  (hypothetical)
GET url, Range: bytes=0-3 → dim = 768
record_size = 3076

# Read logical ordinal 500 (window [0, 1000000))
byte_offset = 500 × 3076 = 1538000
GET url, Range: bytes=1538000-1540075
→ parse: skip first 4 bytes (dim), read 768 × f32 LE
```
