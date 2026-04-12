# 9. Algorithms and Analysis

Detailed specifications for statistical and numerical algorithms.

---

## 9.1 Normalization Analysis

### Normal epsilon

The normal epsilon of a vector **x** measures deviation from unit length:

```
epsilon(x) = | ||x||_2 - 1 |
```

All arithmetic in f64 regardless of storage type.

### Precision-aware threshold

A flat threshold doesn't account for element precision or dimensionality.
The threshold formula uses the probabilistic rounding error bound
(Higham & Mary 2019):

```
is_normalized  <=>  mean_epsilon < C * epsilon_mach(type) * sqrt(dim)
```

Where:
- **C = 10** — headroom above expected error floor
- **epsilon_mach**: f16 = 9.77e-4, f32 = 1.19e-7, f64 = 2.22e-16
- **dim** — vector dimensionality

| Element | dim=128 | dim=768 | dim=1536 |
|---------|---------|---------|----------|
| f16 | 1.11e-1 | 2.71e-1 | 3.83e-1 |
| f32 | 1.35e-5 | 3.30e-5 | 4.67e-5 |
| f64 | 2.51e-14 | 6.16e-14 | 8.71e-14 |

### Theoretical foundation

**Deterministic bound** (Higham 2002, Theorem 3.1): For recursive
summation in precision u, forward error satisfies
`|fl(x^T y) - x^T y| <= gamma_n * sum|x_i y_i|`
where `gamma_n = n*u / (1 - n*u)`.

**Probabilistic bound** (Higham & Mary 2019): Under independent
rounding errors, sqrt(n) replaces n — matching empirical observation
of random walk error cancellation.

### Pipeline integration

The `analyze measure-normals` step produces five epsilon statistics
(`mean`, `min`, `max`, `stddev`, `median`) plus `is_normalized` in
`variables.yaml`, synced to the `dataset.yaml` attributes.

---

## 9.2 Statistical Vector Generation (Virtdata)

Deterministic M-dimensional vector generation from statistical
distribution models. Given the same ordinal and model, the same
vector is always produced.

### Generation pipeline

```
ordinal → normalize → stratified sample (u ∈ (0,1)) → inverse CDF → element value
```

1. **Ordinal normalization**: wrap negative/overflow ordinals
2. **Stratified sampling**: SplitMix64 seed mixing + PCG-XSH-RR-32 RNG
   maps `(ordinal, dimension)` to `u ∈ (1e-10, 1-1e-10)`
3. **Inverse CDF**: per-dimension distribution model maps u to sample

### VectorSpaceModel (JSON)

```json
{
  "unique_vectors": 1000000,
  "dimensions": 128,
  "models": [
    {"type": "normal", "mean": 0.5, "std_dev": 0.1},
    {"type": "beta", "alpha": 2.0, "beta": 5.0}
  ]
}
```

Uniform format (all dimensions identical) or per-dimension
heterogeneous format.

### Scalar distribution samplers

| Distribution | Inverse CDF method | Parameters |
|-------------|-------------------|------------|
| Normal | Abramowitz & Stegun 26.2.23 | mean, std_dev, [bounds] |
| Uniform | Linear scaling | lower, upper |
| Beta (Pearson I) | Newton-Raphson on regularized incomplete beta | alpha, beta, [bounds] |
| Gamma (Pearson III) | Newton-Raphson on regularized incomplete gamma | shape, scale, location |
| Student's t (Pearson VII) | Via beta function relationship | nu, mu, sigma |
| Inverse Gamma (Pearson V) | Transform of gamma | shape, scale |
| Beta Prime (Pearson VI) | Hybrid Newton-Raphson + bisection | alpha, beta, scale |
| Pearson IV | Adaptive Simpson numerical integration | m, nu, a, lambda |
| Empirical | Binary search + linear interpolation in CDF | bins, cdf_values |
| Composite | Linear scan to component, rescale, delegate | sub_models, weights |

### LERP optimization

Optional precomputed lookup table (default 1024 points) for O(1)
sampling, trading ~4 KB memory per dimension for constant-time access.

### Verification criteria

1. **Determinism**: same ordinal + model = identical f32 vector
2. **Statistical accuracy**: empirical distribution matches model (KS test)
3. **Cross-implementation**: bit-identical f32 to Java reference
4. **Independence**: different dimensions uncorrelated
5. **Performance**: > 1M vectors/sec for 128-dim all-normal

---

## 9.3 Statistical Model Extraction (Vshapes)

Extract compact statistical models from real vector datasets.
A `VectorSpaceModel` JSON file (~1-10 KB) captures per-dimension
distribution characteristics sufficient to regenerate statistically
equivalent synthetic vectors.

### Dimension statistics

Computed via two-pass algorithm or parallel Welford/Chan's:

```
count, min, max, mean, variance, skewness, kurtosis
```

Chan's algorithm combines statistics from parallel chunks:

```
delta = mean_B - mean_A
combined_mean = (n_A * mean_A + n_B * mean_B) / (n_A + n_B)
combined_M2 = M2_A + M2_B + delta^2 * n_A * n_B / (n_A + n_B)
```

Higher moments (M3, M4) follow analogous formulas.

### Pearson distribution classification

Given beta_1 = skewness^2 and beta_2 = kurtosis:

| Condition | Classification |
|-----------|---------------|
| symmetric, beta_2 < 3 | Beta (Type I) |
| symmetric, beta_2 = 3 | Normal (Type 0) |
| symmetric, beta_2 > 3 | Student's t (Type VII) |
| kappa < 0 | Beta (Type I) |
| kappa = 0 | Gamma (Type III) |
| 0 < kappa < 1 | Pearson IV |
| kappa = 1 | Inverse Gamma (Type V) |
| kappa > 1 | Beta Prime (Type VI) |

Where kappa = discriminant computed from beta_1 and beta_2.

### Model fitting and selection

**BestFitSelector** — two-stage selection with simplicity bias:

1. Fit all candidate distributions, compute KS D-statistic
2. Select simplest model within 30% relative margin of best fit

Complexity ordering: Normal > Uniform > Beta > Gamma > Student-t >
Inverse Gamma > Beta Prime > Pearson IV > Composite > Empirical.

### Multimodal detection

Histogram-based mode detection with Gaussian kernel smoothing:
- Sturges' rule binning with adaptive scaling
- Peak finding via local maxima
- Prominence filtering (default 0.03 threshold)

When multiple modes detected: EM clustering (Gaussian Mixture Model)
→ segment data → fit each segment → assemble composite model.

### Streaming architecture

```
DataSource → TransposedChunkDataSource → AnalyzerHarness
                                              ├── StreamingDimensionAccumulator (Welford online)
                                              ├── StreamingHistogram (adaptive bounds)
                                              └── StreamingModelExtractor (per-dimension locks)
```

O(1) memory per dimension. Thread-safe via per-dimension locking.

### Statistical validation

| Test | Criterion |
|------|-----------|
| KS D-statistic | D < critical value at significance level |
| Mean | within 1% of sigma |
| Variance | within 5% relative |
| Skewness | within 0.15 |
| Kurtosis | within 0.5 |
| Q-Q correlation | r > 0.995 on 100 quantile points |

### Shared numerical utilities

- **Log Gamma**: Lanczos approximation (6-7 significant digits)
- **Regularized incomplete beta**: Continued fraction via Lentz (max 100 iterations)
- **Regularized incomplete gamma**: Series for x < a+1, continued fraction for x >= a+1
- **Continued fraction**: Modified Lentz algorithm (max 200 iterations, tolerance 1e-12)

---

## 9.4 References

1. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
   2nd ed., SIAM, 2002

2. N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding
   Error Analysis," *SIAM J. Matrix Anal. Appl.*, 40(4), 2019

3. D. Goldberg, "What Every Computer Scientist Should Know About
   Floating-Point Arithmetic," *ACM Comput. Surv.*, 23(1), 1991

4. M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*,
   Dover, 1964

5. K. Pearson, "Contributions to the Mathematical Theory of Evolution,"
   *Phil. Trans. Royal Soc.*, 1895
