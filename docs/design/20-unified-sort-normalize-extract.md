<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 20 — Unified Sort–Deduplicate–Extract Pipeline

This document specifies the data preparation flow that combines
deduplication, L2-normalization, near-zero detection, and extraction
into an efficient pipeline.

---

## 20.1 Motivation

Preparing a vector dataset for benchmarking requires several
transformations:

1. **Deduplication** — identify and remove exact-duplicate vectors
2. **L2-normalization** — scale all vectors to unit length
3. **Near-zero detection** — identify vectors with negligible magnitude
4. **Extraction** — write the clean, normalized subset to the output file
5. **Shuffle** (self-search) — randomize ordinal order for train/test split

The design below minimizes redundant I/O and computation by combining
normalization and zero detection into the extraction step, where each
vector is already in memory for writing.

---

## 20.2 Overview

```
Pass 1: Sort + Deduplicate
══════════════════════════

Source file (raw, any order):
┌───────────────────────────────────────────────────────────┐
│ v₀  v₁  v₂  v₃  v₄  v₅  v₆  v₇  v₈  v₉ ...           │
└───────────────────────────────────────────────────────────┘
                    ↓ read prefix components in segments

For each segment:
┌───────────────────────────────────────────────────────────┐
│ 1. Read prefix components (first 10 dims) + ordinal       │
│ 2. Lexicographic sort by prefix → sorted run file         │
└───────────────────────────────────────────────────────────┘
                    ↓ k-way merge sorted runs

K-way merge:
┌───────────────────────────────────────────────────────────┐
│ 1. Merge sorted runs by prefix key                        │
│ 2. Within prefix collision groups: read full vectors to    │
│    detect exact duplicates via bitwise comparison          │
│ 3. Emit sorted_ordinals.ivec + dedup_duplicates.ivec      │
└───────────────────────────────────────────────────────────┘

Artifacts produced:
  • sorted_ordinals.ivec    — canonical sorted order (deduped)
  • dedup_duplicates.ivec   — ordinals of exact duplicates


Pass 2: Extract + Normalize + Zero-Detect
═════════════════════════════════════════

Inputs:
  • Raw source vectors (via mmap)
  • shuffle.ivec (ordinal permutation for output order)
  • dedup_duplicates.ivec (ordinals to skip)

For each vector read during extraction:
┌───────────────────────────────────────────────────────────┐
│ 1. Read vector from source                                │
│ 2. Compute L2 norm in f64:                                │
│      norm_sq = Σ (xᵢ as f64)²                             │
│ 3. If norm_sq < (1×10⁻⁶)²:                                │
│      → skip (near-zero), record in zero_ordinals          │
│ 4. Normalize: inv_norm = (1/√norm_sq) as f32              │
│      xᵢ ← xᵢ × inv_norm                                   │
│ 5. Write normalized vector to output                      │
└───────────────────────────────────────────────────────────┘
```

**Key insight**: The extraction step reads every vector into memory to
write it to the output file. Since the vector is already in a CPU
register/cache line, computing the L2 norm (one multiply-accumulate
per component) and normalizing (one multiply per component) adds
negligible overhead. There is no separate norm scan pass — normalization
and zero detection happen exactly once, during extraction.

---

## 20.3 Pass 1: Sort + Deduplicate

### Segment Processing

The external merge sort reads prefix components (first 10 dimensions)
from the source file in memory-sized segments. Each segment produces
a sorted run file containing `(ordinal, prefix[0..10])` records.
**Full vectors are not read during this phase** — only the prefix
components needed for sorting.

### Merge Phase

The k-way merge produces `sorted_ordinals.ivec`. When two records
have identical prefixes, the merge reads the full vectors from the
source (via mmap random access) for bitwise equality comparison.
This short-circuits the vast majority of comparisons to prefix-only.

### Artifacts

- `sorted_ordinals.ivec` — deduplicated ordinals in sorted order
- `dedup_duplicates.ivec` — ordinals of exact duplicates (elided
  from the sorted output when `elide=true`)

---

## 20.4 Pass 2: Extract + Normalize + Zero-Detect

Extraction reads raw vectors from the source file and writes them
to the output, applying L2-normalization and near-zero filtering
inline. Every vector is processed exactly once.

### Normalization

For each vector read during extraction:

```
norm_sq: f64 = Σ (xᵢ as f64)²

if norm_sq < (1×10⁻⁶)²:
    record ordinal in zero_ordinals
    skip (do not write)
else:
    inv_norm = (1.0f64 / √norm_sq) as f32
    for each component: xᵢ ← xᵢ × inv_norm
    write normalized vector to output
```

All arithmetic uses f64 precision (SRD §18.2). The `inv_norm` is
computed in f64 and cast to f32 for the per-component multiply.

### Zero Detection

Near-zero vectors (L2 norm < 1×10⁻⁶) are detected inline during
normalization. They are:
- Not written to the output file
- Recorded in `zero_ordinals.ivec` for audit
- Counted in `zero_count` variable

### Shuffled Extraction (Self-Search)

For self-search datasets, the output order follows `shuffle.ivec`
(PRNG permutation). The partitioned-buffer strategy processes output
in memory-sized chunks:

1. **Partition** the output into memory-sized chunks
2. For each output partition:
   a. **Build read plan**: which source ordinals map to this partition
   b. **Sort by source position** for sequential mmap reads
   c. **Read, normalize, and write**: each vector is normalized
      as it is placed in the output buffer
   d. **Flush** the buffer to the output file

---

## 20.5 Required Variables

After a complete pipeline run, `variables.yaml` (and its JSON copy
`variables.json`) **must** contain the following variables. These are
the canonical dataset characterization metrics. Any tool that reads
prepared datasets can rely on their presence.

### Dataset identity

| Variable | Set by | Description |
|----------|--------|-------------|
| `dataset_name` | pipeline init | Dataset name from dataset.yaml |
| `dim` | extract-base | Vector dimensionality |
| `distance_function` | pipeline init | Distance metric (COSINE, L2, DOT_PRODUCT) |

### Counts

| Variable | Set by | Description |
|----------|--------|-------------|
| `vector_count` | count-vectors | Total vectors in source file (dirty) |
| `query_count` | extract-base | Number of query vectors |
| `duplicate_count` | prepare-vectors | Exact duplicates detected |
| `zero_count` | extract-base | Near-zero vectors filtered (L2 < τ) |
| `clean_count` | prepare-vectors | Vectors after dedup, before zero removal |
| `base_count` | count-base | Final base vector count (clean) |
| `extract_input_count` | extract-base | Vectors read during extraction |
| `extract_output_count` | extract-base | Vectors written (after zero filter) |

### Normalization quality

| Variable | Set by | Description |
|----------|--------|-------------|
| `is_normalized` | extract-base | Always `true` — output is normalized |
| `source_was_normalized` | extract-base | Whether source was already normalized |
| `mean_normal_epsilon` | extract-base | Pre-normalization mean &#124;norm − 1&#124; |
| `min_normal_epsilon` | extract-base | Pre-normalization min &#124;norm − 1&#124; |
| `max_normal_epsilon` | extract-base | Pre-normalization max &#124;norm − 1&#124; |
| `normal_threshold` | extract-base | Element-type normalization threshold |

### KNN quality

| Variable | Set by | Description |
|----------|--------|-------------|
| `knn_queries_with_ties` | compute-knn | Queries with boundary ties at rank k |
| `knn_tied_neighbors` | compute-knn | Extra tied neighbors beyond k |

### Provenance chain

The count variables satisfy the invariant:

```
vector_count
  − duplicate_count
  − zero_count
  = base_count
```

### Output formats

The pipeline generates three representations of the variables:

- `variables.yaml` — canonical source, updated incrementally by each step
- `variables.json` — JSON copy, generated by `generate variables-json`
- `dataset.jsonl` — JSON Lines provenance log, generated by `generate dataset-log-jsonl`

All three are generated before merkle hashing and catalog generation,
ensuring the catalog reflects the final variable state.

---

## 20.6 Precision Guarantees

All L2-norm computations use **f64** (IEEE 754 binary64) precision:

| Operation | Precision | Location |
|-----------|-----------|----------|
| Norm computation for zero detection | f64 | Extract (in-memory) |
| Norm computation for normalization | f64 | Extract (in-memory) |
| `inv_norm = 1/√(norm_sq)` | f64, then cast to f32 | Extract |
| `xᵢ × inv_norm` | f32 × f32 → f32 | Extract |

This matches SRD §18.2: "All arithmetic is performed in f64 regardless
of the storage element type."

---

## 20.7 DAG Structure

The pipeline separates sort+dedup from extract+normalize. Zero
detection and normalization are NOT part of `prepare-vectors` —
they happen during extraction where each vector is already in memory
for writing. This eliminates the separate `find-zeros`, `count-zeros`,
`filter-ordinals`, `count-clean`, and `measure-normals` steps.

### Query Generation Strategies

There are three strategies for producing query vectors, determined
by the input source type and the facets provided:

**Strategy 1: Non-HDF5 with B+Q provided**

When the user provides both base and query vectors as separate native
files (fvec, mvec, etc.), the pipeline **combines** them into a single
dataset before processing:

```
┌──────────────┐   ┌───────────────┐
│ base vectors │ + │ query vectors │ ──→ combined source
└──────────────┘   └───────────────┘
                         │
                    count-vectors
                         │
                    prepare-vectors  (sort + dedup)
                         │
                    generate-shuffle
                         │
                         ├──→ extract-queries  (normalize + zero-detect)
                         └──→ extract-base     (normalize + zero-detect)
```

By combining first and then deduplicating, any vectors that appear
in both the base and query sets are naturally deduplicated. The
shuffle produces a clean train/test split from the combined set
with no overlap by construction.

**Strategy 2: HDF5 with B+Q provided**

HDF5 datasets have pre-defined internal datasets (e.g., `train`,
`test`). The base and query data are processed independently:

```
Base path (hdf5#train):              Query path (hdf5#test):
     │                                    │
prepare-vectors                     extract-queries
(sort + dedup)                      (normalize + zero-detect)
     │                                    │
extract-base                              │
(normalize + zero-detect)                 │
     │                                    │
     └──────────────┬─────────────────────┘
                    │
               compute-knn ──→ verify-knn
```

**Strategy 3: Non-HDF5 with B only (self-search)**

When only base vectors are provided, queries are extracted from the
base set via PRNG shuffle:

```
count-vectors
     │
prepare-vectors  (sort + dedup)
     │
generate-shuffle
     │
     ├──→ extract-queries  (normalize + zero-detect)
     └──→ extract-base     (normalize + zero-detect)
```

---

### Superset DAG

The pipeline is a superset graph where every slot always exists. Slots
that require no work collapse to Identity (alias to an upstream
artifact) and emit no pipeline step.

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUPERSET DAG                                │
│                                                                 │
│  fetch-vectors ─→ import-vectors ─→ all-vectors                 │
│       [I]              [I]             [T]                      │
│                                         │                       │
│                                    combine-bq                   │
│                                      [I/M]                      │
│                                         │                       │
│                                  convert-precision              │
│                                      [I/M]                      │
│                                         │                       │
│                                    count-vectors                │
│                                         │                       │
│                                  prepare-vectors ──→ count-dups │
│                                  (sort + dedup)      [M]        │
│                                      [I/M]                      │
│                                         │                       │
│                                  generate-shuffle               │
│                                      [I/M]                      │
│                                    ┌────┴────┐                  │
│                                    │         │                  │
│                             extract-queries  extract-base       │
│                             (normalize +     (normalize +       │
│                              zero-detect)     zero-detect)      │
│                                [I/M]           [M]              │
│                                    │         │                  │
│                               convert-qprec  count-base         │
│                                  [I/M]         │                │
│                                    │           │                │
│                                    └────┬──────┘                │
│                                         │                       │
│                                    compute-knn                  │
│                                      [M]                        │
│                                         │                       │
│                                    verify-knn                   │
│                                      [M]                        │
│                                         │                       │
│                                  gen-dataset-json               │
│                                         │                       │
│                                   gen-merkle                    │
│                                         │                       │
│                                   gen-catalog                   │
│                                                                 │
│  Legend: [I] = Identity (collapsed)                             │
│          [M] = Materialized (step emitted)                      │
│          [T] = Terminal (always resolves)                       │
│        [I/M] = Depends on strategy / inputs                     │
└─────────────────────────────────────────────────────────────────┘
```
