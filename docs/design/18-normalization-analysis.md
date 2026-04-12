<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 18 — Normalization Analysis

This document specifies the normalization measurement algorithm used by
`analyze measure-normals` and the precision-aware heuristic that determines
whether a vector dataset is considered "normalized."

---

## 18.1 Purpose

Many ANN indexes and distance functions (cosine similarity, dot product)
assume or benefit from unit-normalized input vectors. Detecting whether a
dataset is normalized — and with what precision — allows the pipeline to:

1. Select appropriate distance functions automatically.
2. Flag datasets where re-normalization would improve quality.
3. Record the measurement as a machine-readable dataset attribute
   (`is_normalized`) and as distribution statistics in `variables.yaml`.

---

## 18.2 Normal Epsilon

The **normal epsilon** of a vector **x** is defined as:

```
ε(x)  =  | ‖x‖₂ − 1 |
        =  | √(Σ xᵢ²) − 1 |
```

All arithmetic is performed in f64 (IEEE 754 binary64) regardless of the
storage element type. This avoids conflating the precision of the
measurement with the precision of the data.

### Sampling

Given a vector file of *N* vectors, the command samples *S* vectors
(default 10 000, configurable via the `sample` option) using a
deterministic xorshift PRNG seeded by the `seed` option (default 42).
After deduplication, the sampled indices are sorted for sequential I/O.

### Distribution Summary

The epsilon values are summarized as five statistics written to
`variables.yaml`:

| Variable                  | Meaning                                |
|---------------------------|----------------------------------------|
| `mean_normal_epsilon`     | Arithmetic mean of ε across samples    |
| `min_normal_epsilon`      | Smallest ε (best-normalized vector)    |
| `max_normal_epsilon`      | Largest ε (worst-normalized vector)    |
| `stddev_normal_epsilon`   | Sample standard deviation              |
| `median_normal_epsilon`   | Median ε                               |

---

## 18.3 Precision-Aware Normalization Threshold

### 18.3.1 Problem

A flat threshold (e.g., `mean_ε < 1e-6`) does not account for:

- **Element precision**: f16 vectors carry ~3.3 significant decimal digits;
  f32 ~7.2; f64 ~15.9. A dataset stored in f16 cannot achieve the same
  epsilon as one stored in f64.
- **Dimensionality**: Accumulating a sum of *n* squared terms incurs
  rounding error that grows with *n*.

A principled threshold must account for both.

### 18.3.2 Theoretical Foundation

The error analysis rests on two well-established results from numerical
linear algebra.

#### Deterministic Bound (Higham 2002)

For the computed inner product `fl(xᵀy)` of two *n*-vectors using
recursive summation in precision *u* (unit roundoff), the forward error
satisfies:

```
|fl(xᵀy) − xᵀy|  ≤  γₙ · Σ|xᵢ yᵢ|
```

where γₙ = n·u / (1 − n·u) ≈ n·u for practical *n*. When x = y (squared
norm computation), the right-hand side becomes γₙ · ‖x‖₂². For a unit
vector, ‖x‖₂² = 1, so the **relative error in the squared norm** is
bounded by approximately n·u.

The square root introduces one additional rounding of at most *u*, giving
a relative error in the computed norm of approximately (n/2)·u + u — still
**O(n · u)** in the worst case.

> **Reference:** N. J. Higham, *Accuracy and Stability of Numerical
> Algorithms*, 2nd ed., SIAM, 2002. Theorem 3.1 (inner product bound);
> §21.6 (Euclidean norm); §2.8 (statistical rounding-error model).

#### Probabilistic Bound (Higham & Mary 2019)

The worst-case O(n · u) bound assumes all rounding errors accumulate in
the same direction, which is extremely unlikely for real data. Under a
probabilistic model where individual rounding errors are independent
random variables bounded by *u*, the accumulated error for a sum of *n*
terms satisfies, with high probability:

```
|fl(xᵀy) − xᵀy|  ≤  c · √n · u · Σ|xᵢ yᵢ|
```

where *c* is a small constant (O(√(log n)) for failure probability 1/n).

This **√n** factor replaces the *n* factor of the deterministic bound and
matches empirical observations: rounding errors partially cancel like a
random walk.

> **Reference:** N. J. Higham and T. Mary, "A New Approach to
> Probabilistic Rounding Error Analysis," *SIAM J. Matrix Anal. Appl.*,
> vol. 40, no. 4, pp. 1302–1328, 2019. DOI: 10.1137/18M1226312.

### 18.3.3 Applying the Bound to Normalization Detection

For a unit vector stored at element precision *p* (f16, f32, or f64) and
measured in f64:

1. **Quantization error**: Converting from precision *p* to f64 introduces
   per-element error bounded by ε_mach(*p*) / 2. For a unit vector, this
   perturbs the squared norm by at most O(n · ε_mach(*p*)).

2. **Accumulation error**: The f64 inner product accumulates rounding error
   of O(√n · ε_mach(f64)) probabilistically. Since ε_mach(f64) ≪
   ε_mach(*p*) for f16 and f32, the quantization error dominates.

3. **Combined**: The expected epsilon for a correctly normalized vector is:

```
E[ε]  ≈  √n · ε_mach(p)
```

where ε_mach(*p*) is the machine epsilon of the **storage** type (the
dominant error source), and the √n factor reflects probabilistic error
accumulation across *n* dimensions.

### 18.3.4 Threshold Formula

The `is_normalized` determination uses:

```
is_normalized  ⟺  mean_ε  <  C · ε_mach(element_type) · √dim
```

where:

- **C = 10** — provides ~10× headroom above the expected error floor,
  accommodating datasets where normalization was performed in the storage
  precision rather than f64.
- **ε_mach(element_type)** — the machine epsilon of the storage format:
  - f16: 9.77 × 10⁻⁴
  - f32: 1.19 × 10⁻⁷
  - f64: 2.22 × 10⁻¹⁶
- **dim** — the vector dimensionality.

### 18.3.5 Expected Thresholds

| Element | dim | ε_mach        | Threshold (C=10)  |
|---------|-----|---------------|--------------------|
| f16     | 128 | 9.77 × 10⁻⁴  | 1.11 × 10⁻¹       |
| f16     | 768 | 9.77 × 10⁻⁴  | 2.71 × 10⁻¹       |
| f16     | 1536| 9.77 × 10⁻⁴  | 3.83 × 10⁻¹       |
| f32     | 128 | 1.19 × 10⁻⁷  | 1.35 × 10⁻⁵       |
| f32     | 768 | 1.19 × 10⁻⁷  | 3.30 × 10⁻⁵       |
| f32     | 1536| 1.19 × 10⁻⁷  | 4.67 × 10⁻⁵       |
| f64     | 128 | 2.22 × 10⁻¹⁶ | 2.51 × 10⁻¹⁴      |
| f64     | 768 | 2.22 × 10⁻¹⁶ | 6.16 × 10⁻¹⁴      |
| f64     | 1536| 2.22 × 10⁻¹⁶ | 8.71 × 10⁻¹⁴      |

### 18.3.6 Logging

When the threshold is computed, the command logs:

```
  threshold={:.2e} (C=10 × eps={:.2e} × √dim={})
  is_normalized={} (mean_ε={:.2e} {} threshold)
```

This makes the decision transparent and auditable in pipeline output.

---

## 18.4 Pipeline Integration

The `analyze measure-normals` step is inserted into the normative
bootstrap pipeline immediately after the `extract-base` step. Its outputs
are:

1. **Variables** in `variables.yaml` — the five epsilon statistics plus
   `is_normalized`.
2. **Dataset attribute** `is_normalized` — synced to `dataset.yaml` via
   the variable-to-attribute mechanism.

Downstream steps (e.g., distance function selection) may read
`is_normalized` to adapt behavior.

---

## 18.5 References

1. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
   ed., SIAM, Philadelphia, 2002. ISBN 978-0-89871-521-7.
   — Theorem 3.1 (inner product error), §2.8 (probabilistic model),
   §21.6 (Euclidean norm).

2. N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding
   Error Analysis," *SIAM J. Matrix Anal. Appl.*, vol. 40, no. 4,
   pp. 1302–1328, 2019. DOI: 10.1137/18M1226312.
   — Rigorous proof that √n replaces n in probabilistic setting.

3. D. Goldberg, "What Every Computer Scientist Should Know About
   Floating-Point Arithmetic," *ACM Comput. Surv.*, vol. 23, no. 1,
   pp. 5–48, 1991. DOI: 10.1145/103162.103163.
   — Tutorial treatment of accumulation error in summation.
