<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 20 — Unified Sort–Normalize–Extract Pipeline

This document specifies the optimized data preparation flow that
combines deduplication, L2-normalization, near-zero detection, and
extraction into a minimal number of I/O passes over the vector data.

---

## 20.1 Motivation

Preparing a vector dataset for benchmarking requires several
transformations:

1. **Deduplication** — identify and remove exact-duplicate vectors
2. **L2-normalization** — scale all vectors to unit length
3. **Near-zero detection** — identify vectors with negligible magnitude
4. **Extraction** — write the clean, normalized subset to the output file
5. **Shuffle** (self-search) — randomize ordinal order for train/test split

Naively, each transformation requires a full pass over the data. For a
500 GB dataset, five passes = 2.5 TB of I/O. The design below reduces
this to **two passes**: one for sort+normalize+detect, one for extract.

---

## 20.2 Overview

```
Pass 1: Sort + Normalize + Detect
══════════════════════════════════

Source file (raw, any order):
┌───────────────────────────────────────────────────────────┐
│ v₀  v₁  v₂  v₃  v₄  v₅  v₆  v₇  v₈  v₉ ...           │
└───────────────────────────────────────────────────────────┘
                    ↓ read in segments (memory-sized)

For each segment in memory:
┌───────────────────────────────────────────────────────────┐
│ 1. Lexicographic sort (produces sorted ordinal index)     │
│ 2. Detect exact duplicates (byproduct of sorted order)    │
│ 3. Compute L2 norm (f64) for each vector:                 │
│    • If ‖x‖₂ < 1×10⁻⁶ → record ordinal in zeros.ivec   │
│    • Normalize in place: x̂ᵢ = xᵢ / ‖x‖₂               │
│ 4. Write sorted+normalized segment to run file            │
└───────────────────────────────────────────────────────────┘
                    ↓ merge sorted runs

Artifacts produced:
  • sorted_ordinals.ivec    — canonical sorted order
  • dedup_duplicates.ivec   — ordinals of exact duplicates
  • zero_ordinals.ivec      — ordinals of near-zero vectors
  • Sorted+normalized run files (in .cache/)

Key insight: while each segment is in RAM for sorting, the L2
norm computation and normalization are essentially FREE — the data
is already in L1/L2 cache from the sort comparisons.


Pass 2: Extract (with elision)
══════════════════════════════

Inputs:
  • Sorted+normalized data (from pass 1 run files, or source if
    re-reading with ordinal index)
  • exclude = duplicates ∪ zeros (sorted, merged)
  • shuffle.ivec (optional, for self-search)

┌───────────────────────────────────────────────────────────┐
│ Sorted+normalized data:                                   │
│ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                │
│ │s₀│s₁│s₂│s₃│s₄│s₅│s₆│s₇│s₈│s₉│..│..│..│               │
│ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                │
│      ╳        ╳                 ╳    ← excluded           │
│                                                           │
│ Contiguous runs between exclusions:                       │
│ ┌──┐   ┌──┬──┐  ┌──┬──┬──┬──┐   ┌──┬──┐                 │
│ │s₀│   │s₂│s₃│  │s₅│s₆│s₇│s₈│   │..│..│                 │
│ └──┘   └──┴──┘  └──┴──┴──┴──┘   └──┴──┘                 │
│                                                           │
│ Each run is a contiguous memcpy — no per-vector overhead  │
└───────────────────────────────────────────────────────────┘
```

---

## 20.3 Pass 1: Sort + Normalize + Detect

### Segment Processing

The external merge sort reads the source file in memory-sized segments
(governed by `ResourceGovernor`). For each segment:

1. **Read** N vectors into an in-memory buffer
2. **Sort** lexicographically by component values (parallel radix or
   comparison sort, same as current `compute sort`)
3. **Dedup** — scan adjacent pairs in sorted order for exact equality.
   Record duplicate ordinals in `dedup_duplicates.ivec`
4. **Normalize + zero-detect** — for each vector in the segment:

   ```
   norm_sq: f64 = Σ (xᵢ as f64)²
   
   if norm_sq < (1×10⁻⁶)²:
       record ordinal in zero_ordinals
   else:
       inv_norm = (1.0f64 / √norm_sq) as f32
       for each component: xᵢ ← xᵢ × inv_norm
   ```

5. **Write** the sorted+normalized segment as a run file

All arithmetic for norm computation uses f64 precision (SRD §18.2).
The normalization overwrites the in-memory buffer before writing,
so the run files contain normalized data.

### Why In-Segment Is Free

The sort step loads each vector's components into CPU cache for
comparison. Immediately after sorting, the data is still cache-hot.
Computing the L2 norm (one multiply-accumulate per component) and
normalizing (one multiply per component) adds ~2d FLOPs per vector
where d is the dimension. For d = 1024, this is ~2K FLOPs — a few
microseconds per vector, dominated by the sort's O(N log N)
comparison overhead. The normalization pass is compute-bound on
already-cached data, adding negligible wall time.

### Merge Phase

After all segments are sorted and written, the standard k-way merge
produces `sorted_ordinals.ivec`. The run files on disk contain the
sorted+normalized vector data. The merge phase operates only on
ordinals (not vector data), so it does not re-read the vectors.

---

## 20.4 Exclusion Set Construction

After pass 1, three index files exist:

```
sorted_ordinals.ivec     — all ordinals in sorted order
dedup_duplicates.ivec    — ordinals of duplicate vectors
zero_ordinals.ivec       — ordinals of near-zero vectors (L2 < τ)
```

The exclusion set is the union:

```
exclude = dedup_duplicates ∪ zero_ordinals
```

For extraction, this set is sorted and stored as a compact sorted
array for O(1) membership testing via binary search or merge scan.

---

## 20.5 Pass 2: Extract with Elision

### Non-Shuffled Datasets (Separate Query File)

For datasets with a separate query file, the output base vectors
preserve the sorted order (minus excluded entries). Extraction
scans the sorted+normalized data sequentially:

```
write_cursor = 0
for ordinal in sorted_ordinals:
    if ordinal ∈ exclude:
        continue  // skip this vector
    output[write_cursor] = normalized_data[ordinal]
    write_cursor += 1
```

Because both the sorted data and the exclusion set are in sorted
order, this degenerates to writing contiguous runs between
exclusion points. Each run is a single `memcpy`:

```
Sorted data:  [A B C _ D E F G _ _ H I J K ...]
                      ╳               ╳ ╳
Output:       [A B C | D E F G | H I J K ...]
               run 1   run 2    run 3
```

The `_` entries are excluded. Runs between them are written as
contiguous blocks, minimizing syscall overhead and maximizing
sequential I/O throughput.

### Shuffled Datasets (Self-Search)

For self-search datasets, the output must be in shuffle order (PRNG
permutation). The shuffle covers the full dataset including
excluded entries:

```
shuffle.ivec = PRNG permutation of [0..N)
```

Extraction uses the partitioned-buffer strategy (SRD §12):

1. **Partition** the output into memory-sized chunks
2. For each output partition:
   a. **Build read plan**: which source ordinals map to output
      positions in this partition (via shuffle inverse)
   b. **Sort read plan by source position** for sequential reads
   c. **Sequential scan** of sorted+normalized data, writing each
      non-excluded vector to its output position in the buffer
   d. **Flush** the buffer to the output file

The sort-by-source-position step ensures the source mmap is read
sequentially. The writes to the in-memory buffer are random but
cache-friendly (buffer fits in RAM). This is the same binning
strategy used by the current extraction logic.

### Query Extraction

Query vectors are extracted from the shuffle permutation's first
`query_count` entries, using the same read plan + buffer strategy.
Near-zero queries are also filtered during extraction.

---

## 20.6 Provenance Metrics

Every processing step records metrics for human-readable audit trails
in `variables.yaml` and `dataset.log`:

| Variable | Set by | Description |
|----------|--------|-------------|
| `vector_count` | count-vectors | Total vectors in source file (or combined B+Q) |
| `duplicate_count` | prepare-vectors | Exact duplicates detected |
| `zero_count` | prepare-vectors | Near-zero vectors detected (L2 < τ) |
| `mean_normal_epsilon` | prepare-vectors | Mean deviation of L2 norms from 1.0 (f64) |
| `max_normal_epsilon` | prepare-vectors | Worst-case norm deviation |
| `is_normalized` | prepare-vectors | Whether mean epsilon < precision threshold (§18) |
| `base_count` | count-base | Final base vector count |

The provenance chain is verifiable:

```
vector_count
  − duplicate_count
  − zero_count
  = base_count
```

Additionally, `zero_ordinals.ivec` is written as a secondary artifact
for post-hoc verification of which specific vectors were filtered.

---

## 20.7 Precision Guarantees

All L2-norm computations use **f64** (IEEE 754 binary64) precision,
regardless of the storage element type:

| Operation | Precision | Location |
|-----------|-----------|----------|
| Norm computation for zero detection | f64 | Sort pass (in-memory) |
| Norm computation for normalization | f64 | Sort pass (in-memory) |
| `inv_norm = 1/√(norm_sq)` | f64, then cast to f32 | Sort pass |
| `xᵢ × inv_norm` | f32 × f32 → f32 | Sort pass |
| Norm validation (prepare-vectors (normalization stats)) | f64 | Post-extract verification |

This matches SRD §18.2: "All arithmetic is performed in f64 regardless
of the storage element type."

---

## 20.8 DAG Structure

The unified pipeline eliminates `find-zeros`, `count-zeros`,
`filter-ordinals`, and `count-clean` as separate steps. Their work
is absorbed into `prepare-vectors` (dedup + normalize + zero-detect)
and `extract-base` (elide excluded ordinals directly).

### Query Generation Strategies

There are three strategies for producing query vectors, determined
by the input source type and the facets provided:

**Strategy 1: Non-HDF5 with B+Q provided**

When the user provides both base and query vectors as separate native
files (fvec, mvec, etc.), the pipeline **combines** them into a single
dataset before processing:

```
┌─────────────┐   ┌──────────────┐
│ base vectors │ + │ query vectors │ ──→ combined source
└─────────────┘   └──────────────┘
                         │
                    count-vectors
                         │
                    prepare-vectors  (dedup + normalize + zero-detect)
                         │
                    generate-shuffle
                         │
                    ├──→ extract-queries  (first query_count from shuffle)
                    └──→ extract-base     (remainder from shuffle)
```

By combining first and then deduplicating, any vectors that appear
in both the base and query sets are naturally deduplicated. The
shuffle produces a clean train/test split from the combined set
with no overlap by construction. **No separate overlap removal step
is needed.**

**Strategy 2: HDF5 with B+Q provided**

HDF5 datasets have pre-defined internal datasets (e.g., `train`,
`test`). The base and query data are processed independently — no
intermixing:

```
Base path (hdf5#train):              Query path (hdf5#test):
     │                                    │
prepare-vectors                     normalize + zero-detect
(dedup + normalize                  (no dedup — queries are
 + zero-detect)                      typically small & unique)
     │                                    │
extract-base                        extract-queries
     │                                    │
     └──────────────┬─────────────────────┘
                    │
               compute-knn ──→ verify-knn
```

For HDF5 queries, only normalization (L2 in f64) and near-zero
filtering are applied. No deduplication — HDF5 test sets are
typically curated and small enough that duplicates are not a concern.

**Strategy 3: Non-HDF5 with B only (self-search)**

When only base vectors are provided, queries are extracted from the
base set via PRNG shuffle:

```
count-vectors
     │
prepare-vectors  (dedup + normalize + zero-detect)
     │
generate-shuffle
     │
├──→ extract-queries  (first query_count from shuffle)
└──→ extract-base     (remainder from shuffle)
```

This is the standard self-search path. The shuffle guarantees
disjoint base and query sets by construction — the first
`query_count` ordinals become queries, the remainder become base.

---

### Superset DAG

The pipeline is a superset graph where every slot always exists. Slots
that require no work collapse to Identity (alias to an upstream
artifact) and emit no pipeline step. The superset contains all slots
for all three strategies; strategy selection determines which slots
collapse.

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUPERSET DAG                                │
│                                                                 │
│  fetch-vectors ─→ import-vectors ─→ all-vectors                │
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
│                                  prepare-vectors ──→ count-dups  │
│                                  (sort + dedup +     [M]        │
│                                   normalize +                   │
│                                   zero-detect +                 │
│                                   norm-stats)                   │
│                                      [I/M]                      │
│                                         │                       │
│                                  generate-shuffle               │
│                                      [I/M]                      │
│                                    ┌────┴────┐                  │
│                                    │         │                  │
│                             extract-queries  extract-base       │
│                                [I/M]           [M]              │
│                                    │         │                  │
│                               convert-qprec  count-base        │
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
│          [T] = Terminal (always resolves)                        │
│        [I/M] = Depends on strategy / inputs                     │
└─────────────────────────────────────────────────────────────────┘
```

### Identity Collapse Rules

| Slot | Collapses to Identity when | Effect |
|------|---------------------------|--------|
| `fetch-vectors` | No URL provided | No download step |
| `import-vectors` | Source is native xvec | Symlink, no conversion |
| `combine-bq` | HDF5 source, or B-only input | No concatenation; base used directly |
| `convert-precision` | No `--base-convert-format` | No precision conversion |
| `prepare-vectors` | `--no-dedup` AND `--no-normalize` | No sort, no normalization, no zero-detect, no norm stats. Source ordinals used as-is |
| `generate-shuffle` | No Q facet, or HDF5 with separate query | No shuffle; base and query are independent |
| `extract-queries` | HDF5 source (queries converted separately) | Query conversion handled by `convert-queries` instead |
| `convert-qprec` | No query precision conversion | No conversion step |

### Strategy-Specific Collapse Patterns

**Strategy 1: Non-HDF5 B+Q (combined)**

```
Materialized: combine-bq, prepare-vectors, generate-shuffle,
              extract-queries, extract-base, compute-knn, verify-knn,
              count-*, gen-*
Identity:     fetch, import (if native), convert-precision (if not needed)
```

**Strategy 2: HDF5 B+Q (independent)**

```
Materialized: import-vectors, prepare-vectors, extract-base,
              convert-queries (with normalize+zero-filter),
              compute-knn, verify-knn, count-*, gen-*
Identity:     combine-bq, generate-shuffle, extract-queries
```

**Strategy 3: Non-HDF5 B only (self-search)**

```
Materialized: prepare-vectors, generate-shuffle, extract-queries,
              extract-base, compute-knn, verify-knn, count-*, gen-*
Identity:     combine-bq, fetch (if no URL), import (if native)
```

---

### Step Responsibilities

| Step | Inputs | Outputs | Work |
|------|--------|---------|------|
| `count-vectors` | source file (or combined B+Q) | `vector_count` | Record count |
| `prepare-vectors` | working vectors | `sorted_ordinals.ivec`, `dedup_duplicates.ivec`, `zero_ordinals.ivec`, sorted+normalized run files, normalization statistics | External merge sort, dedup detection, L2-normalize in-place (f64), near-zero detection (L2 < τ), normalization quality measurement. See §20.3 |
| `count-duplicates` | `dedup_duplicates.ivec` | `duplicate_count` | Record count |
| `generate-shuffle` | `sorted_ordinals.ivec`, exclusion set | `shuffle.ivec` | PRNG permutation over clean ordinals |
| `extract-queries` | sorted+normalized data, shuffle | `query_vectors.fvec` | First `query_count` shuffled ordinals, elide excluded |
| `extract-base` | sorted+normalized data, shuffle or sorted, exclusion set | `base_vectors.fvec` | Remainder (shuffle) or full set (non-shuffle), elide excluded. See §20.5 |
| `count-base` | `base_vectors.fvec` | `base_count` | Record count |
| `convert-queries` | HDF5 query source | `query_vectors.fvec` | Import + normalize (f64) + zero-filter. HDF5 Strategy 2 only |
| `compute-knn` | `base_vectors.fvec`, `query_vectors.fvec` | `neighbor_indices.ivec`, `neighbor_distances.fvec` | Brute-force exact KNN |
| `verify-knn` | same as compute-knn + GT files | verification report | Recompute sample queries, compare to GT |

### Step Count Comparison

| Pipeline | Before (§12) | After (§20) |
|----------|-------------|-------------|
| Self-search (B only) | 18 | 11 |
| Combined B+Q | 16 | 11 |
| HDF5 B+Q | 16 | 10 |

Steps eliminated:
- `find-zeros` + `count-zeros` → absorbed into `prepare-vectors`
- `filter-ordinals` + `count-clean` → eliminated; exclusion set
  applied directly by `extract-base` and `extract-queries`
- `remove-query-duplicates` → eliminated; Strategy 1 combines
  inputs before dedup, Strategy 2 processes independently,
  Strategy 3 uses shuffle disjointness

---

## 20.9 Relationship to Other SRD Sections

- **§12** — Dataset Import Flowchart: the slot model and identity
  collapse rules remain unchanged. The `zero_check` slot becomes
  Identity (no separate step); zero detection is a byproduct of sort.
- **§18** — Normalization Analysis: the f64 precision guarantees
  apply to the in-sort normalization pass.
- **§19** — Zero Vector Detection: the L2-norm threshold (τ = 1×10⁻⁶)
  and squared comparison apply during the sort pass.
