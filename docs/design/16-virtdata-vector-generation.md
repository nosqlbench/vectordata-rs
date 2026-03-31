# 16 — Virtdata: Deterministic Vector Generation from Statistical Models

This document captures the requirements and algorithms from the Java `datatools-virtdata`
module in `nbdatatools`, for reimplementation as a `veks` subcommand.

## Purpose

Generate M-dimensional float vectors deterministically from ordinal indices using
per-dimension statistical distribution models. Given the same ordinal and model, the
same vector is always produced — enabling reproducible, parallel, on-the-fly synthetic
data without materializing to disk.

## Use Cases

- **Synthetic dataset generation**: Produce arbitrarily large vector datasets from compact
  model files (~1 KB JSON describes billions of vectors).
- **Ground truth computation**: Deterministic generation enables exact KNN computation
  without storing base vectors.
- **Distribution-preserving data**: Generated vectors match the statistical properties
  (per-dimension distributions) of real datasets from which the model was extracted.

## Input: VectorSpaceModel (JSON)

A model file describes N unique vectors of M dimensions, where each dimension has an
independent scalar distribution.

### Uniform Gaussian Format

All dimensions share identical parameters:

```json
{
  "unique_vectors": 1000000,
  "dimensions": 128,
  "mean": 0.0,
  "std_dev": 1.0,
  "lower_bound": -1.0,
  "upper_bound": 1.0
}
```

Fields:
- `unique_vectors` (u64, required): Number of unique vectors N.
- `dimensions` (usize, required): Dimensionality M.
- `mean` (f64, default 0.0): Gaussian mean.
- `std_dev` (f64, default 1.0): Gaussian standard deviation.
- `lower_bound` (f64, optional): Truncation lower bound.
- `upper_bound` (f64, optional): Truncation upper bound.

### Per-Dimension Heterogeneous Format

Each dimension specifies its own distribution:

```json
{
  "unique_vectors": 1000000,
  "components": [
    {"type": "normal", "mean": 0.0, "std_dev": 1.0},
    {"type": "uniform", "lower": -1.0, "upper": 1.0},
    {"type": "beta", "alpha": 2.0, "beta": 5.0},
    {"type": "gamma", "shape": 2.0, "scale": 1.0},
    {"type": "student_t", "nu": 5.0},
    {"type": "inverse_gamma", "shape": 3.0, "scale": 2.0},
    {"type": "beta_prime", "alpha": 2.0, "beta": 4.0},
    {"type": "pearson_iv", "m": 2.0, "nu": 0.5, "a": 1.0, "lambda": 0.0},
    {
      "type": "composite",
      "components": [
        {"type": "normal", "mean": -1.0, "std_dev": 0.5},
        {"type": "normal", "mean": 1.0, "std_dev": 0.5}
      ],
      "weights": [0.7, 0.3]
    },
    {
      "type": "empirical",
      "bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
      "cdf": [0.0, 0.1, 0.3, 0.7, 1.0],
      "min": -1.0,
      "max": 1.0
    }
  ]
}
```

The number of entries in `components` equals the dimensionality M.

---

## Generation Pipeline

For each vector at ordinal `o` (0 ≤ o < N):

```
for d in 0..M:
    u = stratified_sample(o, d, N)      // → (0, 1)
    vector[d] = inverse_cdf(u, model[d]) // → f32
```

### Stage 1: Ordinal Normalization

```
normalized = ordinal % N
if normalized < 0: normalized += N
```

Handles negative ordinals and values ≥ N by wrapping.

### Stage 2: Stratified Sampling (ordinal, dimension → unit interval)

Maps `(ordinal, dimension)` to a value u ∈ (ε, 1−ε) where ε = 1e-10.

#### Algorithm

1. **Seed mixing** (SplitMix64 avalanche):
   ```
   seed = ordinal * 0xBF58476D1CE4E5B9 + dimension * 0x9E3779B97F4A7C15
   seed ^= seed >> 30
   seed *= 0xBF58476D1CE4E5B9
   seed ^= seed >> 27
   seed *= 0x94D049BB133111EB
   seed ^= seed >> 31
   ```

2. **RNG**: Initialize PCG-XSH-RR-32 with the mixed seed.

3. **Sample**: `u = rng.next_f64()`, clamped to `[1e-10, 1 - 1e-10]`.

#### Constants

| Name | Value | Provenance |
|------|-------|------------|
| ORDINAL_PRIME | `0xBF58476D1CE4E5B9` | Murmur3 / SplitMix64 mixing constant |
| DIMENSION_PRIME | `0x9E3779B97F4A7C15` | Golden ratio: (φ−1) × 2^64 |
| SplitMix multiplier 2 | `0x94D049BB133111EB` | SplitMix64 second mixing constant |
| EPSILON | `1e-10` | Avoids infinities in inverse CDF |

#### Properties

- O(1) per sample, no global state, parallelizable by ordinal.
- PCG-XSH-RR-32 passes TestU01 BigCrush.
- Different dimensions produce uncorrelated values for same ordinal.

### Stage 3: Inverse CDF Transform

Each dimension's scalar model defines a distribution. The sampler maps u ∈ (0,1) to a
sample from that distribution via the inverse CDF (quantile function).

All distribution parameters are bound at construction time. The `sample(u)` hot path
takes only the unit-interval value.

---

## Scalar Distribution Samplers

### Normal (type: `"normal"`)

**Parameters**: `mean` (μ), `std_dev` (σ), optional `lower_bound`, `upper_bound`.

**Unbounded**: `x = μ + σ · Φ⁻¹(u)`

**Truncated** (when bounds present): Proper inverse transform, not clamping:
```
a = (lower - μ) / σ
b = (upper - μ) / σ
Φ_a = Φ(a)           // standard normal CDF at a
Z = Φ(b) - Φ_a       // probability mass in [lower, upper]
u' = Φ_a + u · Z     // rescale u to truncated range
x = μ + σ · Φ⁻¹(u')
```

**Inverse Normal CDF** (Abramowitz & Stegun 26.2.23):
```
t = sqrt(-2 · ln(p))     // p = min(u, 1-u)
x = t - (c₀ + c₁·t + c₂·t²) / (1 + d₁·t + d₂·t² + d₃·t³)
// negate if u < 0.5
```

Coefficients:
- c₀ = 2.515517, c₁ = 0.802853, c₂ = 0.010328
- d₁ = 1.432788, d₂ = 0.189269, d₃ = 0.001308
- Accuracy: ~4.5 × 10⁻⁴ relative error.

**Forward Normal CDF** (for truncation setup, Abramowitz & Stegun 7.1.26):
```
Φ(x) = 0.5 · (1 + erf(x / sqrt(2)))

erf(x) = 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵) · exp(-x²)
where t = 1 / (1 + 0.3275911 · |x|)
```
Coefficients: a₁=0.254829592, a₂=−0.284496736, a₃=1.421413741, a₄=−1.453152027, a₅=1.061405429.
Accuracy: |error| < 1.5 × 10⁻⁷.

### Uniform (type: `"uniform"`)

**Parameters**: `lower`, `upper`.

```
x = lower + u · (upper - lower)
```

### Beta (type: `"beta"`)

**Parameters**: `alpha` (α), `beta` (β), `lower` (default 0), `upper` (default 1).

Newton-Raphson on the regularized incomplete beta function I_x(α, β):

1. Clamp u to [1e-10, 1−1e-10].
2. Initial guess: normal approximation when α,β ≥ 1, else `x₀ = clamp(u, 0.01, 0.99)`.
3. Iterate (max 50):
   ```
   cdf = I_x(α, β)
   pdf = x^(α-1) · (1-x)^(β-1) / B(α,β)
   if |cdf - u| < 1e-12: break
   if pdf < 1e-100: break
   x -= (cdf - u) / pdf
   x = clamp(x, 1e-10, 1-1e-10)
   ```
4. Scale: `result = lower + (upper - lower) · x`.

**Regularized Incomplete Beta** I_x(a,b): Continued fraction (Lentz algorithm, max 200 iterations, convergence |δ−1| < 1e-14). Use `I_x` directly when `x < (a+1)/(a+b+2)`, otherwise `1 − I_{1-x}(b,a)`.

### Gamma (type: `"gamma"`)

**Parameters**: `shape` (k), `scale` (θ), `location` (shift, default 0).

Newton-Raphson on the lower regularized incomplete gamma P(k, x):

1. Clamp u to [1e-10, 1−1e-10].
2. Initial guess: Wilson-Hilferty approximation for k ≥ 1, else `x₀ = max(0.01, u^(1/k) · k)`.
3. Iterate (max 50):
   ```
   cdf = P(k, x)
   pdf = exp((k-1)·ln(x) - x - lnΓ(k))
   if |cdf - u| < 1e-12: break
   x = max(1e-10, x - (cdf - u) / pdf)
   ```
4. Scale: `result = location + scale · x`.

**Lower Regularized Incomplete Gamma P(a,x)**:
- x < a+1: series expansion (max 200 terms, convergence |δ| < |sum|·1e-14).
- x ≥ a+1: `1 − Q(a,x)` via continued fraction (Lentz, max 200, |δ−1| < 1e-14).

### Student's t (type: `"student_t"`)

**Parameters**: `nu` (ν, degrees of freedom), `mu` (location, default 0), `sigma` (scale, default 1).

Uses the relationship between t-distribution and beta function:

1. Split: if u < 0.5 then negative=true, q=2u; else negative=false, q=2(1−u).
2. Compute w = InverseBetaCDF(q, ν/2, 1/2) using Newton-Raphson (same as Beta sampler).
3. Transform: `t = sqrt(ν · (1-w) / w)`.
4. Apply sign: if negative then t = −t.
5. Scale: `result = mu + sigma · t`.

### Inverse Gamma (type: `"inverse_gamma"`)

**Parameters**: `shape` (α), `scale` (β).

Transform of gamma: `InverseGamma⁻¹(u) = β / Gamma⁻¹(1−u; α, 1)`.

1. Clamp u to [1e-10, 1−1e-10].
2. `gamma_quantile = InverseGammaCDF(1 - u, shape)` (same solver as Gamma).
3. `result = scale / gamma_quantile`.

### Beta Prime (type: `"beta_prime"`)

**Parameters**: `alpha` (α), `beta` (β), `scale` (σ, default 1).

Transform of beta: `BetaPrime⁻¹(u) = σ · x/(1−x)` where x = Beta⁻¹(u).

**Critical**: Conservative u-bounds `[2e-7, 1−2e-4]` because x/(1−x) explodes as x→1.

Uses hybrid Newton-Raphson + bisection (max 100 iterations):
- Damped Newton step (factor 0.5).
- Bisection bounds updated continuously.
- Convergence: |error| < 1e-12 or |Δx| < 1e-15.

### Pearson Type IV (type: `"pearson_iv"`)

**Parameters**: `m` (shape), `nu` (ν, skewness), `a` (scale), `lambda` (λ, location).

**PDF**:
```
f(x) = k · [1 + ((x−λ)/a)²]^(−m) · exp(−ν · arctan((x−λ)/a))
```

Normalization constant (log domain):
```
log_k = lnΓ(m) - 0.5·ln(π) - lnΓ(m - 0.5)
if ν ≠ 0: log_k -= 0.5·ln(1 + (ν/2)²/m²)
```

Newton-Raphson with numerically integrated CDF (max 100 iterations):
- CDF via adaptive Simpson's rule (tolerance 1e-10, max depth 20).
- Integration lower bound: λ − 20σ where σ = a/√(2m−1).
- Damped Newton step clamped to ±2a.
- Convergence: |error| < 1e-10 or |Δx| < 1e-12.

### Empirical (type: `"empirical"`)

**Parameters**: `bins` (edges), `cdf` (cumulative probabilities), `min`, `max`.

Binary search + linear interpolation:
1. Binary search to find bin index where `cdf[i] ≤ u < cdf[i+1]`.
2. Interpolate: `t = (u − cdf[i]) / (cdf[i+1] − cdf[i])`.
3. `result = bins[i] + t · (bins[i+1] − bins[i])`.

O(log n) per sample.

### Composite / Mixture (type: `"composite"`)

**Parameters**: `components` (ScalarModel[]), `weights` (f64[], sum to 1).

1. Compute cumulative weights at construction.
2. Per sample: linear scan to find component where `u < cumulative[i]`.
3. Rescale u to component's interval: `u' = (u − cumulative[i-1]) / weight[i]`.
4. Delegate: `result = component_samplers[i].sample(u')`.

Supports recursive nesting (components can be Composite).

---

## LERP Optimization (Optional)

Any sampler can be wrapped with a precomputed lookup table for O(1) sampling:

**Construction** (one-time, per dimension):
```
for i in 0..table_size:
    u = (i + 0.5) / table_size    // midpoint of interval
    table[i] = delegate.sample(u)
```

**Sampling** (O(1)):
```
idx = u * (table_size - 1)
lo = floor(idx)
hi = min(lo + 1, table_size - 1)
t = idx - lo
result = table[lo] + t · (table[hi] - table[lo])
```

Default table size: 1024 (8 KB per dimension). Minimum: 16.
Accuracy: error < 0.1% for most distributions with 1024 points.

---

## L2 Normalization (Optional)

Post-generation wrapper that normalizes vectors to unit length:

```
norm = sqrt(sum(v[d]² for d in 0..M))
if norm > 1e-10:
    for d in 0..M: v[d] /= norm
// zero/near-zero vectors left unchanged
```

Use case: similarity search workloads requiring unit vectors (cosine distance).

---

## Shared Numerical Utilities

### Log Gamma (Lanczos Approximation)

Used by Beta, Gamma, Student-t, Inverse Gamma, Beta Prime, Pearson IV samplers.

```
y = x
tmp = x + 5.5
tmp -= (x + 0.5) · ln(tmp)
ser = 1.000000000190015
ser += 76.18009172947146 / (y + 1)
ser += -86.50532032941677 / (y + 2)
ser += 24.01409824083091 / (y + 3)
ser += -1.231739572450155 / (y + 4)
ser += 0.1208650973866179e-2 / (y + 5)
ser += -0.5395239384953e-5 / (y + 6)
result = -tmp + ln(2.5066282746310005 · ser / x)
```

Accuracy: 6-7 significant digits.

### Continued Fraction (Modified Lentz Algorithm)

Used for regularized incomplete beta and incomplete gamma functions.

```
c = 1, d = 1/b₀, h = d
for m in 1..=200:
    (a_m, b_m) = problem_specific_terms(m)
    d = a_m · d + b_m
    if |d| < 1e-30: d = 1e-30
    c = b_m + a_m / c
    if |c| < 1e-30: c = 1e-30
    d = 1/d
    delta = d · c
    h *= delta
    if |delta - 1| < 1e-14: break
result = h
```

---

## Convergence Criteria Summary

| Algorithm | Max Iterations | Error Tolerance | Guard |
|-----------|---------------|-----------------|-------|
| Normal inverse CDF | Analytical | N/A | — |
| Beta Newton-Raphson | 50 | |err| < 1e-12 | pdf < 1e-100 |
| Gamma Newton-Raphson | 50 | |err| < 1e-12 | pdf < 1e-100 |
| Student-t (via beta) | 50 | |err| < 1e-12 | pdf < 1e-100 |
| Inverse Gamma (via gamma) | 50 | |err| < 1e-12 | pdf < 1e-100 |
| Beta Prime (hybrid) | 100 | |err| < 1e-12 | |Δx| < 1e-15 |
| Pearson IV (Newton + quadrature) | 100 | |err| < 1e-10 | |Δx| < 1e-12 |
| Continued fraction (Lentz) | 200 | |δ−1| < 1e-14 | — |
| Gamma series | 200 | |δ| < |sum|·1e-14 | — |
| Adaptive Simpson (Pearson IV CDF) | depth 20 | 1e-10 | — |

---

## SIMD Optimization (Optional, for all-normal models)

When all M dimensions use Normal distributions, the inverse CDF can be vectorized:

1. Detect `all_normal` at initialization.
2. Extract `means[M]` and `std_devs[M]` arrays.
3. Process LANES dimensions at a time using SIMD:
   - Vectorized seed mixing and PCG sampling.
   - Vectorized log (range reduction + Remez polynomial).
   - Vectorized sqrt (fast inverse sqrt + 2 Newton-Raphson iterations).
   - Vectorized Abramowitz-Stegun rational approximation.
4. Scalar tail for remaining `M % LANES` dimensions.

### Vectorized Log Constants (Remez minimax for log(1+x) on [0,1])

- L0 = 0.9999999999999998
- L1 = −0.4999999999532199
- L2 = 0.33333332916609
- L3 = −0.24999845065233
- L4 = 0.20012028055456
- LN2 = 0.6931471805599453

### Vectorized Sqrt Magic Constant

- `0x5FE6EB50C7B537A9` (Lomont's double-precision fast inverse sqrt constant)
- Two Newton-Raphson refinement iterations: `y = y · (1.5 − 0.5·x·y·y)`

---

## Proposed CLI Interface

```
veks generate virtdata --model model.json --count 1000000 --output base_vectors.fvec
veks generate virtdata --model model.json --start 0 --count 1000 --format json
veks generate virtdata --model model.json --count 1000000 --lerp --lerp-table-size 2048
veks generate virtdata --model model.json --count 1000000 --normalize-l2
```

Options:
- `--model <path>`: VectorSpaceModel JSON file (required).
- `--count <n>`: Number of vectors to generate.
- `--start <ordinal>`: Starting ordinal (default 0).
- `--output <path>`: Output file (format inferred from extension).
- `--format`: Override output format (fvec, json, csv).
- `--lerp`: Enable LERP lookup table optimization.
- `--lerp-table-size <n>`: Table size (default 1024, min 16).
- `--normalize-l2`: Normalize output vectors to unit length.
- `--batch-size <n>`: Vectors per batch for progress reporting.

---

## Verification Criteria

A correct implementation must satisfy:

1. **Determinism**: `generate(ordinal, model)` always produces the identical vector.
2. **Statistical accuracy**: For each dimension, the empirical distribution of generated
   values (over many ordinals) must match the model's specified distribution within
   KS-test tolerance.
3. **Cross-implementation compatibility**: Given identical model JSON and ordinals, the
   Rust implementation should produce bit-identical f32 vectors to the Java reference
   (since the algorithms and constants are fully specified).
4. **Independence**: Different dimensions for the same ordinal are uncorrelated.
5. **Performance**: Single-threaded throughput should exceed 1M vectors/sec for 128-dim
   all-normal models on modern hardware.
