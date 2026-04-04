<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 19 — Zero Vector Detection and L2-Norm Pipeline Integration

This document specifies the near-zero vector detection algorithm and its
integration with the always-on L2 normalization pipeline.

---

## 19.1 Motivation

Exact-zero vectors produce undefined results for cosine similarity
(division by zero norm) and degenerate results for dot product (distance
always zero). Near-zero vectors — those with extremely small but non-zero
components — suffer the same problems: normalization amplifies noise to
unit scale, producing meaningless unit vectors, and distance computations
lose numerical stability.

Previous versions detected zero vectors by exact byte comparison
(`∀ᵢ xᵢ = 0.0`). This missed near-zero vectors such as
`[1×10⁻⁸, 0, 0, …, 0]` (L2 norm = 1×10⁻⁸) that cause the same
numerical issues after normalization.

---

## 19.2 Detection Algorithm

A vector **x** ∈ ℝᵈ is classified as **near-zero** when:

```
‖x‖₂  =  √(Σᵢ xᵢ²)  <  τ
```

where τ = 1×10⁻⁶ is the default threshold.

### Implementation

To avoid computing the square root, the comparison is performed in
squared space:

```
Σᵢ xᵢ²  <  τ²
```

All arithmetic is performed in **f64** (IEEE 754 binary64) regardless of
the storage element type, consistent with the normalization precision
requirement (SRD §18.2). Each component `xᵢ` is upcast to f64 before
squaring:

```rust
let norm_sq: f64 = (0..dim).map(|d| {
    let v = component[d] as f64;
    v * v
}).sum();
let is_zero = norm_sq < threshold * threshold;
```

### Threshold

The default threshold τ = 1×10⁻⁶ is chosen to be:

- **Below meaningful signal.** At f32 precision (~7 decimal digits),
  vectors with L2 norm < 10⁻⁶ carry at most 1 significant digit of
  directional information. Normalizing such a vector amplifies
  quantization noise to unit scale.
- **Above floating-point underflow.** The threshold is well above the
  f32 minimum normal (≈1.2×10⁻³⁸), so detection is not confused by
  denormalized values.

The threshold is configurable via the `threshold` pipeline step option
for datasets with unusual scale characteristics.

---

## 19.3 DAG Integration

### Dependency Structure

The `find-zeros` step scans the raw source vectors directly. It does
**not** depend on the sorted ordinal index produced by `sort-and-dedup`.
This enables parallel execution:

```
count-vectors ──→ sort-and-dedup ──→ count-duplicates ──┐
             │                                          │
             └──→ find-zeros ──→ count-zeros ───────────┤
                                                        ↓
                                              filter-ordinals → count-clean → extract-base → …
```

`filter-ordinals` waits for both `sort-and-dedup` and `find-zeros`
before combining their exclusion sets (duplicate ordinals + near-zero
ordinals) into `clean_ordinals.ivec`.

### Previous Design (Removed)

The previous implementation used a binary search optimization: since
all-zero vectors sort lexicographically first, `find-zeros` could scan
forward from position 0 in the sorted index and break at the first
non-zero vector. This optimization is incompatible with L2-norm
detection because near-zero vectors do not necessarily sort to the
beginning (e.g., `[-1×10⁻⁸, 0, …]` sorts before `[0, 0, …]` which
sorts before `[1×10⁻⁸, 0, …]`).

### Parallelism

The full-scan approach uses rayon parallel iteration over all vectors.
Each vector's norm is computed independently — the workload is
embarrassingly parallel with no shared mutable state.

---

## 19.4 Interaction with Normalization

### Always-On Normalization

L2 normalization is enabled by default for all datasets (SRD §18).
The `extract-base` step normalizes every extracted vector:

```
x̂ᵢ = xᵢ / ‖x‖₂
```

Normalization also computes L2 norms, but this is a **separate pass**
over a **different subset** of vectors:

| Step | Vectors processed | Purpose |
|------|-------------------|---------|
| `find-zeros` | All raw source vectors | Identify near-zero vectors for exclusion |
| `extract-base` | Clean ordinals only (post-dedup, post-zero-exclusion) | Normalize and write output |

Caching norms between these steps is not beneficial because:

1. The subsets are different (all vectors vs. clean subset).
2. The I/O cost of writing/reading a norm cache file (~4 bytes × N
   vectors) often exceeds the compute cost of recomputing norms.
3. The norm computation is O(d) per vector — negligible compared to
   the I/O of reading the vector data itself.

### Precision

Both steps compute norms in f64 precision:

- `find-zeros`: f64 norm for threshold comparison (detection accuracy)
- `extract-base`: f64 norm for computing `inv_norm`, then cast to f32
  for the normalized output (SRD §18.2 precision guarantee)
- `measure-normals`: f64 norm for validation (SRD §18.2)

---

## 19.5 Pipeline Step Specification

### `analyze zeros`

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `source` | Path | yes | — | Source vector file (fvec, mvec, dvec) |
| `output` | Path | yes | — | Output ivec file of near-zero ordinals |
| `threshold` | float | no | `1e-06` | L2-norm threshold below which a vector is classified as near-zero |

**Output:** An ivec file with dimension 1, containing the ordinals of
all vectors whose L2 norm (computed in f64) is below the threshold.

**Status:**
- `Ok` — no near-zero vectors found
- `Warning` — near-zero vectors detected and recorded

**Variables written:**
- `zero_count` (via downstream `state set` step)

---

## 19.6 Rationale for Threshold Choice

| Threshold | Effect |
|-----------|--------|
| 1×10⁻³ | Too aggressive — removes vectors with meaningful small magnitudes |
| **1×10⁻⁶** | **Default — below meaningful f32 signal, catches degenerate vectors** |
| 1×10⁻¹⁰ | Too lenient — misses vectors that produce unstable normalized results |
| 0 (exact) | Previous behavior — misses all near-zero vectors |

The threshold is not dimension-dependent. A vector with ‖x‖₂ < 10⁻⁶ in
any dimensionality carries insufficient directional information for
meaningful similarity computation.
