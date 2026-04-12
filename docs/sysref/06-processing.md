# 6. Data Processing

---

## 6.1 Deduplication

Sort-based deduplication using bitwise vector equality:

1. **Sort** — Radix sort on vector bytes, producing `sorted_ordinals.ivec`
2. **Compare** — Adjacent vectors in sorted order are compared bitwise
3. **Mark** — Duplicate ordinals written to `dedup_duplicates.ivec`
4. **Filter** — Extract steps skip marked ordinals

The unified `prepare-vectors` step (`compute sort`) handles sort +
dedup + zero detection + norm measurement in a single pass.

---

## 6.2 L2 Normalization

Vectors are optionally normalized to unit length (L2 norm = 1.0):

```
v_normalized = v / ||v||₂
```

Precision analysis reports:
- `norm_mean`, `norm_min`, `norm_max` — distribution of output norms
- `norm_max_abs_deviation` — worst-case deviation from 1.0
- Machine epsilon thresholds: f16 (~1e-3), f32 (~1e-7), f64 (~1e-16)

When pre-computed ground truth is provided with the vectors,
normalization defaults to OFF (the GT was computed on unnormalized data).

---

## 6.3 Zero Vector Detection

Near-zero vectors (L2 norm below threshold) are detected and flagged:

- **Threshold**: configurable, default 1e-6
- **Exact zeros**: all components are bitwise zero
- **Near-zeros**: norm > 0 but below threshold

Zero vectors cause division-by-zero during normalization and degenerate
nearest-neighbor results. The pipeline sets `zero_count` and
`source_zero_count` variables, which drive the `is_zero_vector_free`
dataset attribute.

When `prepare-vectors` is skipped (Identity base), a standalone
`analyze find-zeros` step scans the source vectors.

---

## 6.4 Metadata Synthesis

### Simple-int-eq mode

Generates random integer metadata and equality predicates:

1. **Metadata** (`generate metadata`): Random integers in `[min, max]`,
   one per base vector, written as flat-packed scalar (e.g., `.u8`)
2. **Predicates** (`generate predicates`): Random integers in the same
   range, one per query, written as scalar
3. **Evaluation** (`compute evaluate-predicates`): For each predicate
   value, finds all base ordinals with matching metadata. Output is
   variable-length `.ivvec` — each record lists the matching ordinals

### Selectivity

With range `[0, max]`, expected selectivity = `1 / (max + 1)`.
For `max = 12`: ~7.7% of base vectors match each predicate.

---

## 6.5 Predicate Evaluation

The `evaluate-predicates` command:

1. Reads metadata (scalar) and predicates (scalar)
2. Builds a hash map index: `value → [ordinals]`
3. For each predicate, looks up matching ordinals
4. Writes results as `.ivvec` (variable-length records)
5. Builds the `IDXFOR__` offset index for the output

Time complexity: O(M) to build index + O(P) to evaluate.
Space: O(M) for the hash map.

### Verification

`verify predicates-sqlite` loads M, P, R into SQLite and independently
evaluates every predicate via SQL, comparing against the stored results.

---

## 6.6 KNN Computation

Brute-force exact k-nearest-neighbor search:

1. For each query, compute distance to every base vector
2. Maintain a max-heap of size k
3. Write indices (`.ivec`) and distances (`.fvec`)

Metrics: L2, Cosine, DotProduct. Multi-threaded with configurable
thread count.

### Filtered KNN

Predicate-filtered KNN pre-filters the candidate set:

1. Load predicate results (`.ivvec`) for the query's predicate
2. Compute distances only to matching base vectors
3. Top-k from the filtered set

### Tie-break handling

When multiple base vectors at the k-th boundary have identical
distances (duplicate vectors), verification counts these as passes
rather than mismatches.
