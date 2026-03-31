# 17 — Vshapes: Statistical Model Extraction from Real Vector Data

This document captures the requirements and algorithms from the Java `datatools-vshapes`
module in `nbdatatools`, for reimplementation in the `vectordata-rs` project.

**Companion document**: [16-virtdata-vector-generation.md](16-virtdata-vector-generation.md)
covers the generation/sampling side that **consumes** VectorSpaceModel JSON. This document
covers the **analysis** side — how models are extracted from real vector data.

---

## Purpose

Analyze real vector datasets (e.g., embedding files in `.fvec` format) to extract compact
statistical models that capture the per-dimension distribution characteristics. The output
is a `VectorSpaceModel` JSON file (see doc 16 for the schema) that can regenerate
statistically equivalent synthetic vectors.

## Use Cases

- **Model extraction**: Given a large vector dataset (millions of vectors, hundreds of
  dimensions), produce a small JSON model (~1-10 KB) that describes the statistical
  shape of each dimension.
- **Distribution fitting**: Automatically select the best distribution type per dimension
  from the Pearson family (Normal, Beta, Gamma, Student-t, etc.) plus empirical and
  composite (mixture) models.
- **Streaming analysis**: Process datasets that don't fit in memory by streaming chunks
  and accumulating statistics incrementally.
- **Quality validation**: Verify that extracted models faithfully reproduce the original
  distribution using KS tests, moment comparison, and Q-Q correlation.

---

## 1. Scalar Model Types and Parameters

Each dimension is modeled independently as a `ScalarModel`. The following types are
supported, corresponding to the Pearson distribution classification system.

### 1.1 NormalScalarModel (Pearson Type 0)

**JSON type**: `"normal"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Mean | `mean` | f64 | yes | — | any real |
| Std Dev | `std_dev` | f64 | yes | — | > 0 |
| Lower Bound | `lower_bound` | f64 | no | -Infinity | < upper |
| Upper Bound | `upper_bound` | f64 | no | +Infinity | > lower |

**Truncation**: When both bounds are present, the model represents a truncated normal.
The CDF for the truncated case is:

```
F(x) = (Phi((x - mu) / sigma) - Phi((lower - mu) / sigma))
     / (Phi((upper - mu) / sigma) - Phi((lower - mu) / sigma))
```

Guard: if the denominator `< 1e-15`, fall back to linear interpolation `(x - lower) / (upper - lower)`.

**Fitting**: MLE — `mu = sample_mean`, `sigma = sample_std_dev` (population formula).

**Truncation detection heuristic**: detect truncation when:
1. Observed range < 50% of expected range (6 sigma), OR
2. More than 2% of data is within 1% of both the min and max boundaries.

### 1.2 UniformScalarModel

**JSON type**: `"uniform"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Lower | `lower` | f64 | yes | — | < upper |
| Upper | `upper` | f64 | yes | — | > lower |

**Derived**:
- Mean = `(lower + upper) / 2`
- Std Dev = `(upper - lower) / sqrt(12)`
- CDF: `(x - lower) / (upper - lower)`, clamped to [0, 1]

**Fitting**: `lower = min(data)`, `upper = max(data)`.

### 1.3 BetaScalarModel (Pearson Type I / II)

**JSON type**: `"beta"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Alpha | `alpha` | f64 | yes | — | > 0 |
| Beta | `beta` | f64 | yes | — | > 0 |
| Lower | `lower` | f64 | no | 0.0 | < upper |
| Upper | `upper` | f64 | no | 1.0 | > lower |

**Moments** (standard [0,1] interval, scale for custom [lower, upper]):
- Mean = `alpha / (alpha + beta)`
- Variance = `alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))`
- Skewness = `2 * (beta - alpha) * sqrt(alpha + beta + 1) / ((alpha + beta + 2) * sqrt(alpha * beta))`
- Kurtosis (standard) = `3 * (alpha + beta + 1) * (2*(alpha+beta)^2 + alpha*beta*(alpha+beta-6)) / (alpha*beta*(alpha+beta+2)*(alpha+beta+3))`

**CDF**: Regularized incomplete beta function `I_x(alpha, beta)` via continued fraction (see Section 7).

**Type I vs II**: Type II is the symmetric case where `alpha = beta`.

**Effectively uniform**: When `|alpha - 1| < 0.1` and `|beta - 1| < 0.1`, the beta is
functionally identical to uniform. The selector may alias Beta(1,1) to Uniform in
non-strict mode.

### 1.4 GammaScalarModel (Pearson Type III)

**JSON type**: `"gamma"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Shape | `shape` | f64 | yes | — | > 0 |
| Scale | `scale` | f64 | yes | — | > 0 |
| Location | `location` | f64 | no | 0.0 | any real |

**Moments**:
- Mean = `shape * scale + location`
- Variance = `shape * scale^2`
- Skewness = `2 / sqrt(shape)`
- Kurtosis (standard) = `3 + 6/shape`

**CDF**: Regularized lower incomplete gamma `P(shape, (x - location) / scale)` (see Section 7).

**Method of moments fitting**: `shape = mean^2 / variance`, `scale = variance / mean`.

### 1.5 StudentTScalarModel (Pearson Type VII)

**JSON type**: `"student_t"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Degrees of Freedom | `nu` | f64 | yes | — | > 0 |
| Location | `mu` | f64 | no | 0.0 | any real |
| Scale | `sigma` | f64 | no | 1.0 | > 0 |

**Moments** (when defined):
- Mean = `mu` (for nu > 1; undefined otherwise)
- Variance = `sigma^2 * nu / (nu - 2)` (for nu > 2; infinite for 1 < nu <= 2)
- Skewness = 0 (for nu > 3)
- Excess Kurtosis = `6 / (nu - 4)` (for nu > 4)

**CDF**: Via relationship to incomplete beta:
```
z = nu / (nu + t^2)
beta_cdf = I_z(nu/2, 0.5)
CDF = (t > 0) ? 1 - beta_cdf/2 : beta_cdf/2
```

**Estimation from excess kurtosis**: `nu = 4 + 6 / excess_kurtosis` (requires excess kurtosis > 0).

### 1.6 InverseGammaScalarModel (Pearson Type V)

**JSON type**: `"inverse_gamma"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Shape | `shape` | f64 | yes | — | > 0 |
| Scale | `scale` | f64 | yes | — | > 0 |

**Moments**:
- Mean = `scale / (shape - 1)` (for shape > 1)
- Variance = `scale^2 / ((shape - 1)^2 * (shape - 2))` (for shape > 2)
- Mode = `scale / (shape + 1)`
- Skewness = `4 * sqrt(shape - 2) / (shape - 3)` (for shape > 3)

**CDF**: `F(x) = 1 - P(shape, scale / x)` where P is regularized lower incomplete gamma.

**Method of moments**: `shape = 2 + mean^2 / variance`, `scale = mean * (shape - 1)`.

### 1.7 BetaPrimeScalarModel (Pearson Type VI)

**JSON type**: `"beta_prime"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Alpha | `alpha` | f64 | yes | — | > 0 |
| Beta | `beta` | f64 | yes | — | > 0 |
| Scale | `scale` | f64 | no | 1.0 | > 0 |

**Relationship**: If X ~ Beta(alpha, beta) then X/(1-X) ~ BetaPrime(alpha, beta).

**CDF**: `F(x) = I_{x/(1+x)}(alpha, beta)` — transform to [0,1] then use regularized incomplete beta.

**Moments**:
- Mean = `scale * alpha / (beta - 1)` (for beta > 1)
- Mode = `scale * (alpha - 1) / (beta + 1)` (for alpha >= 1)

### 1.8 PearsonIVScalarModel (Pearson Type IV)

**JSON type**: `"pearson_iv"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Shape | `m` | f64 | yes | — | > 0.5 |
| Skewness param | `nu` | f64 | yes | — | any real |
| Scale | `a` | f64 | yes | — | > 0 |
| Location | `lambda` | f64 | yes | — | any real |

**PDF**:
```
f(x) = k * [1 + ((x - lambda) / a)^2]^(-m) * exp(-nu * arctan((x - lambda) / a))
```

**Normalization constant** (log domain):
```
log_k = ln_gamma(m) - 0.5 * ln(pi) - ln_gamma(m - 0.5)
if nu != 0: log_k -= 0.5 * ln(1 + (nu / 2)^2 / m^2)
```

**CDF**: Numerical integration of PDF (trapezoidal rule with 1000 steps over `[lambda - 10*a, x]`).

**Mean** = `lambda + a * nu / (2*m - 2)` (for m > 1).

**From moments**:
```
beta1 = skewness^2
beta2 = kurtosis  (standard, not excess)
denom1 = 2*beta2 - 3*beta1 - 6
r = 6 * (beta2 - beta1 - 1) / denom1
m = r / 2
nu = -skewness * sqrt(m)
a = std_dev * sqrt((2*m - 2)^2 / ((2*m - 1)*(2*m - 2) - nu^2))
lambda = mean - a * nu / (2*m - 2)
```

### 1.9 EmpiricalScalarModel

**JSON type**: `"empirical"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Bin Edges | `bins` | f64[] | yes | — | length >= 2, monotonically increasing |
| CDF Values | `cdf` | f64[] | yes | — | same length as bins, [0, 1], monotonic |
| Mean | `mean` | f64 | yes | — | — |
| Std Dev | `std_dev` | f64 | yes | — | — |
| Min | `min` | f64 | yes | — | = bins[0] |
| Max | `max` | f64 | yes | — | = bins[last] |

**CDF**: Linear interpolation within bins. Binary search for bin, then:
```
t = (u - cdf[i]) / (cdf[i+1] - cdf[i])
x = bins[i] + t * (bins[i+1] - bins[i])
```

**Construction from data** (`fromData`):
1. Determine bin count via Sturges' rule: `ceil(log2(n)) + 1`, clamped to [10, 1000].
2. Compute uniform bin edges from `[min, max]`.
3. Histogram counting, then cumulative normalization to CDF.

**Construction from streaming histogram** (`fromHistogram`):
Accepts `long[]` counts, min, max, mean, stdDev. Converts to bin edges + CDF.

### 1.10 CompositeScalarModel

**JSON type**: `"composite"`

| Parameter | JSON key | Type | Required | Default | Constraints |
|-----------|----------|------|----------|---------|-------------|
| Components | `sub_models` | ComponentConfig[] | yes | — | non-empty |
| Weights | `weights` | f64[] | yes | — | same length, non-negative, normalized to sum=1 |

**CDF**: `F(x) = sum(weights[i] * components[i].cdf(x))`, clamped to [0, 1].

**Mixture moments** (law of total expectation / law of total variance):
```
mean     = sum(w_i * mu_i)
variance = sum(w_i * (sigma_i^2 + (mu_i - mean)^2))
```

Third and fourth central moments use the full binomial expansion:
```
third_moment = sum(w_i * (gamma_i*sigma_i^3 + 3*(mu_i - mean)*sigma_i^2 + (mu_i - mean)^3))
fourth_moment = sum(w_i * ((kurt_i+3)*sigma_i^4 + 4*(mu_i-mean)*gamma_i*sigma_i^3
                          + 6*(mu_i-mean)^2*sigma_i^2 + (mu_i-mean)^4))
```

**Simple composites**: A composite with exactly one component is functionally equivalent
to the underlying model. On serialization, simple composites are unwrapped to the inner model.

**Canonical form**: Components sorted by characteristic location (mean/mode), with
model reduction (e.g., Beta(1,1) -> Uniform, merge equivalent uniforms).

---

## 2. VectorSpaceModel Structure

The `VectorSpaceModel` is a second-order tensor: a collection of M `ScalarModel` instances,
one per dimension, plus metadata.

**Fields**:
- `unique_vectors` (u64): Number of unique vectors N. Must be positive.
- `dimensions` (usize): Dimensionality M. Equals the length of `scalarModels`.
- `scalarModels` (ScalarModel[M]): Per-dimension distribution model.

**Isomorphic detection**: When all M scalar models have the same type (e.g., all Normal),
the model is "isomorphic" and may use vectorized/SIMD sampling strategies.

### 2.1 JSON Serialization (VectorSpaceModelConfig)

Two JSON representations are supported:

**Compact uniform format** (all dimensions share identical normal parameters):
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

**Per-dimension heterogeneous format** (each dimension has its own model):
```json
{
  "unique_vectors": 1000000,
  "components": [
    {"type": "normal", "mean": 0.0, "std_dev": 1.0},
    {"type": "uniform", "lower": 0.0, "upper": 1.0},
    {"type": "beta", "alpha": 2.0, "beta": 5.0, "lower": 0.0, "upper": 1.0},
    {"type": "gamma", "shape": 2.0, "scale": 1.0, "shift": 0.0},
    {"type": "student_t", "nu": 5.0, "mu": 0.0, "sigma": 1.0},
    {"type": "inverse_gamma", "shape": 3.0, "scale": 2.0},
    {"type": "beta_prime", "alpha": 2.0, "beta": 4.0, "scale": 1.0},
    {"type": "pearson_iv", "m": 2.0, "nu": 0.5, "a": 1.0, "lambda": 0.0},
    {"type": "empirical", "bins": [...], "mean": 0.0, "std_dev": 1.0, "min": -1.0, "max": 1.0},
    {
      "type": "composite",
      "sub_models": [
        {"type": "normal", "mean": -1.0, "std_dev": 0.5},
        {"type": "normal", "mean": 1.0, "std_dev": 0.5}
      ],
      "weights": [0.7, 0.3]
    }
  ]
}
```

**Serialization rule**: When saving, if all dimensions are identical normal models,
use the compact format. Otherwise, use per-dimension format.

**Type adapter**: A Gson `TypeAdapterFactory` (`ScalarModelTypeAdapterFactory`) handles
polymorphic serialization/deserialization based on the `"type"` field.

---

## 3. Dimension Statistics Computation

Statistics are computed per-dimension as the first step of model fitting.

### 3.1 DimensionStatistics Record

| Field | Type | Description |
|-------|------|-------------|
| dimension | int | Dimension index |
| count | i64 | Number of observations |
| min | f64 | Minimum observed value |
| max | f64 | Maximum observed value |
| mean | f64 | Arithmetic mean |
| variance | f64 | Population variance |
| skewness | f64 | Third standardized moment |
| kurtosis | f64 | Fourth standardized moment (standard, NOT excess) |

**Note on kurtosis convention**: The Java code uses **standard kurtosis** (beta_2),
where Normal = 3.0. Excess kurtosis = standard kurtosis - 3.

### 3.2 Two-Pass Batch Algorithm

**Pass 1** — min, max, sum:
```
for each value v:
    min = min(min, v)
    max = max(max, v)
    sum += v
mean = sum / count
```

**Pass 2** — central moments:
```
m2 = m3 = m4 = 0
for each value v:
    diff = v - mean
    diff2 = diff * diff
    m2 += diff2
    m3 += diff2 * diff
    m4 += diff2 * diff2

variance = m2 / count
std_dev = sqrt(variance)
skewness = (m3 / count) / std_dev^3     (if std_dev > 0, else 0)
kurtosis = (m4 / count) / variance^2    (if std_dev > 0, else 3)
```

### 3.3 Combining Statistics (Parallel Welford / Chan's Algorithm)

Two `DimensionStatistics` A and B can be algebraically combined:

```
n_AB = n_A + n_B
delta = mean_B - mean_A
mean_AB = mean_A + delta * n_B / n_AB

M2_A = variance_A * n_A
M2_B = variance_B * n_B
M2_AB = M2_A + M2_B + delta^2 * n_A * n_B / n_AB
variance_AB = M2_AB / n_AB

M3_A = skewness_A * stdDev_A^3 * n_A
M3_B = skewness_B * stdDev_B^3 * n_B
M3_AB = M3_A + M3_B
      + delta^3 * n_A * n_B * (n_A - n_B) / n_AB^2
      + 3 * delta * (n_A * M2_B - n_B * M2_A) / n_AB

M4_A = kurtosis_A * variance_A^2 * n_A
M4_B = kurtosis_B * variance_B^2 * n_B
M4_AB = M4_A + M4_B
      + delta^4 * n_A * n_B * (n_A^2 - n_A*n_B + n_B^2) / n_AB^3
      + 6 * delta^2 * (n_A^2 * M2_B + n_B^2 * M2_A) / n_AB^2
      + 4 * delta * (n_A * M3_B - n_B * M3_A) / n_AB
```

This enables parallel and chunked computation with exact algebraic equivalence to
single-pass computation. Properties: associative, approximately commutative (up to
floating-point precision).

---

## 4. Pearson Distribution Classification

The Pearson system classifies distributions based on skewness and kurtosis into
types that determine which parametric model family to try first.

### 4.1 Classifier Inputs

- `beta_1 = skewness^2` (squared skewness)
- `beta_2 = kurtosis` (standard kurtosis, NOT excess)

### 4.2 Classification Algorithm

**Step 1: Symmetric check** (beta_1 < SKEWNESS_TOLERANCE^2, i.e., |skewness| < 0.1):

| Condition | Type |
|-----------|------|
| |beta_2 - 3| < 0.2 | Type 0 (Normal) |
| beta_2 < 3 | Type II (Symmetric Beta) |
| beta_2 > 3 | Type VII (Student's t) |

**Step 2: Asymmetric classification** — compute discriminant kappa:

```
kappa = beta_1 * (beta_2 + 3)^2 / [4 * (2*beta_2 - 3*beta_1 - 6) * (4*beta_2 - 3*beta_1)]
```

Edge case: if `|denominator| < 1e-10`, classify as Type III (Gamma).

| kappa range | Type |
|-------------|------|
| kappa < -0.05 | Type I (Beta) |
| |kappa| <= 0.05 | Type III (Gamma) |
| 0.05 < kappa < 0.95 | Type IV (Pearson IV) |
| |kappa - 1| <= 0.05 | Type V (Inverse Gamma) |
| kappa > 1.05 | Type VI (Beta Prime) |

### 4.3 Classification Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| SKEWNESS_TOLERANCE | 0.1 | Threshold for considering distribution symmetric |
| KURTOSIS_TOLERANCE | 0.2 | Threshold for beta_2 ~ 3 (normal-like) |
| KAPPA_TOLERANCE | 0.05 | Threshold for kappa boundary comparisons |

---

## 5. Model Fitting and Selection

### 5.1 ComponentModelFitter Interface

Each distribution type has a fitter that:
1. Estimates parameters from `DimensionStatistics` and raw data.
2. Computes a goodness-of-fit score (KS D-statistic).
3. Returns a `FitResult(model, goodness_of_fit, model_type)`.

### 5.2 Goodness-of-Fit: Kolmogorov-Smirnov D-Statistic

All parametric fitters use the same scoring method for fair comparison:

```
Sort values ascending.
max_D = 0
for i in 0..n:
    empirical_cdf      = (i + 1) / n
    empirical_cdf_prev = i / n
    model_cdf = model.cdf(sorted[i])
    d1 = |empirical_cdf - model_cdf|
    d2 = |empirical_cdf_prev - model_cdf|
    max_D = max(max_D, d1, d2)
return max_D
```

Lower D-statistic = better fit. Range: [0, 1].

### 5.3 Normal Fitter Kurtosis Adjustment

The Normal fitter applies a kurtosis-based score adjustment to improve discrimination:

- **Symmetric data with kurtosis near 3.0**: bonus up to -40% of KS score.
  ```
  normalDistance = |kurtosis - 3.0|
  symmetry = 1.0 - |skewness| / 0.3
  adjustment = -0.40 * ksScore * (1 - normalDistance/0.5) * symmetry
  ```
- **Uniform-like kurtosis (< 2.2)**: penalty up to +25% of KS score.
- Otherwise: no adjustment.

### 5.4 BestFitSelector: Multi-Model Competition

The selector runs all registered fitters and picks the winner:

**Two-stage selection**:

1. **Raw best**: lowest penalized score (empirical models get +`empirical_penalty` added).

2. **Simplicity bias**: among models within a relative threshold of the best score,
   prefer the simpler one.
   - Default multiplier: 0.30 (30% relative margin).
   - Strict mode multiplier: 0.0 (disabled).
   - A model qualifies if `score <= rawBest * (1 + multiplier)`.

**Model complexity ordering** (lower = simpler, preferred):

| Rank | Type | Rationale |
|------|------|-----------|
| 1 | normal | 2 params, highly stable |
| 2 | uniform | 2 params |
| 3 | beta | 4 params, can oscillate with normal |
| 4 | gamma | 3+ params, unbounded tail |
| 5 | student_t | Heavy tails |
| 6 | inverse_gamma | — |
| 7 | beta_prime | — |
| 8 | pearson_iv | — |
| 9 | composite | Mixture model |
| 10 | empirical | Fallback, penalized separately |

**Beta-to-Uniform aliasing** (non-strict mode only): If Beta is selected with
alpha ~ beta ~ 1 (tolerance 0.1), replace with Uniform. Disabled in strict mode
for round-trip testing.

### 5.5 Selector Presets

| Preset | Fitters | Use Case |
|--------|---------|----------|
| `defaultSelector` | Normal, Uniform, Empirical | Quick basic fitting |
| `boundedDataSelector` | Normal, Beta, Uniform | Data in [-1,1] or [0,1] |
| `pearsonSelector` | Normal, Beta, Gamma, Student-t, Uniform | Full parametric |
| `fullPearsonSelector` | All 8 parametric types + Uniform | Complete Pearson family |
| `normalizedPearsonSelector` | Normal[-1,1], Uniform[-1,1], Empirical | L2-normalized vectors |
| `pearsonMultimodalSelector` | Full Pearson + Composite(EM) + Empirical | Multimodal data |
| `adaptiveCompositeSelector` | Full Pearson + Composite(EM,4) + Empirical | Adaptive extraction |

The `normalizedPearsonSelector` intentionally excludes Beta to prevent round-trip
instability (Normal and high-alpha Beta are indistinguishable for bell-shaped bounded data).

---

## 6. Multimodal Detection and Composite Fitting

### 6.1 Mode Detection (ModeDetector)

**Algorithm**: Histogram-based with Gaussian kernel smoothing.

1. **Bin count**: Sturges' rule `ceil(log2(n)) + 1`, with adaptive scaling:
   `max(bins_per_mode * max_modes, sturges_bins)` where `bins_per_mode = 5`.

2. **Histogram construction**: Uniform bins from [min, max].

3. **Gaussian kernel smoothing**: Convolve with kernel of bandwidth (default 3.0 bins).

4. **Peak finding**: Local maxima where `smoothed[i] > smoothed[i-1]` and `smoothed[i] > smoothed[i+1]`.

5. **Prominence filtering**: Peak prominence must exceed `threshold * max_height` (default threshold 0.03).
   Prominence is the height above the highest valley on either side.

6. **Adaptive resolution refinement** (for maxModes > 4): iteratively increase bin count
   when peaks appear merged (too wide, asymmetric, or have shoulders).

**Constants**:

| Constant | Value | Purpose |
|----------|-------|---------|
| DEFAULT_MAX_MODES | 10 | Maximum modes to detect |
| DEFAULT_PROMINENCE_THRESHOLD | 0.03 | Minimum peak prominence (fraction of max) |
| DEFAULT_SMOOTHING_BANDWIDTH | 3.0 | Gaussian kernel bandwidth in bins |
| DIP_MULTIMODAL_THRESHOLD | 0.05 | Hartigan's dip test threshold |
| MIN_BINS_PER_MODE | 5 | Minimum histogram resolution per mode |

### 6.2 EM Clustering (EMClusterer)

Expectation-Maximization for Gaussian Mixture Models.

**Initialization**: From detected peak locations. Equal weights, standard deviation
estimated as `(data_max - data_min) / (2 * num_components)`.

**E-step** (compute responsibilities):
```
for each data point i:
    for each component k:
        densities[k] = weights[k] * gaussian_pdf(x_i, means[k], stddevs[k])
    sum_density = sum(densities)
    if sum_density < 1e-300: sum_density = 1e-300
    responsibilities[i][k] = densities[k] / sum_density
    log_likelihood += ln(max(sum_density, 1e-300))
```

**M-step** (update parameters):
```
for each component k:
    N_k = sum_i(responsibilities[i][k])
    means[k] = sum_i(responsibilities[i][k] * x_i) / N_k
    variance_k = sum_i(responsibilities[i][k] * (x_i - means[k])^2) / N_k
    stddevs[k] = sqrt(max(variance_k, 1e-10))
    weights[k] = N_k / total_data_points
// Normalize weights to sum to 1
```

**Convergence criteria**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_ITERATIONS | 50 | Maximum EM iterations |
| CONVERGENCE_THRESHOLD | 1e-6 | Change in log-likelihood for convergence |
| MIN_VARIANCE | 1e-10 | Floor for component variance |
| LOG_EPSILON | 1e-300 | Floor for log(0) avoidance |

### 6.3 Composite Model Fitting (CompositeModelFitter)

Combines mode detection, EM clustering, and per-component fitting:

1. **Detect modes** using ModeDetector.
2. If only 1 mode detected, decline to fit (return poor score).
3. **Run EM** with detected peak locations as initialization.
4. **Segment data** by maximum responsibility (hard assignment from soft clustering).
5. **Fit each segment** independently using a component BestFitSelector.
6. **Assemble** CompositeScalarModel from fitted components + EM weights.
7. **Score** the composite model using KS D-statistic against full data.

**Clustering strategies**: `EM` (default, preferred) or `NEAREST_MODE` (simpler hard clustering).

---

## 7. Shared Numerical Utilities

### 7.1 Log Gamma (Lanczos Approximation)

Used by Beta, Gamma, Student-t, Inverse Gamma, Beta Prime, Pearson IV.

```
y = x
tmp = x + 5.5
tmp -= (x + 0.5) * ln(tmp)
ser = 1.000000000190015
ser += 76.18009172947146   / (y + 1)
ser += -86.50532032941677  / (y + 2)
ser += 24.01409824083091   / (y + 3)
ser += -1.231739572450155  / (y + 4)
ser += 0.1208650973866179e-2 / (y + 5)
ser += -0.5395239384953e-5   / (y + 6)
result = -tmp + ln(2.5066282746310005 * ser / x)
```

Accuracy: 6-7 significant digits.

### 7.2 Regularized Incomplete Beta Function

`I_x(a, b)` via continued fraction (Lentz algorithm):

```
bt = exp(lnGamma(a+b) - lnGamma(a) - lnGamma(b) + a*ln(x) + b*ln(1-x))

if x < (a+1)/(a+b+2):
    return bt * betaCF(x, a, b) / a
else:
    return 1.0 - bt * betaCF(1-x, b, a) / b
```

**betaContinuedFraction** (max 100 iterations, convergence |delta - 1| < 1e-10):
```
qab = a + b; qap = a + 1; qam = a - 1
c = 1; d = 1 - qab*x/qap
if |d| < 1e-30: d = 1e-30
d = 1/d; h = d

for m = 1..100:
    m2 = 2*m
    aa = m*(b-m)*x / ((qam+m2)*(a+m2))
    d = 1 + aa*d;  if |d| < 1e-30: d = 1e-30
    c = 1 + aa/c;  if |c| < 1e-30: c = 1e-30
    d = 1/d; h *= d*c

    aa = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
    d = 1 + aa*d;  if |d| < 1e-30: d = 1e-30
    c = 1 + aa/c;  if |c| < 1e-30: c = 1e-30
    d = 1/d; del = d*c; h *= del
    if |del - 1| < 1e-10: break
return h
```

### 7.3 Regularized Incomplete Gamma Function

`P(a, x)` — lower regularized incomplete gamma:

**Series expansion** (for x < a + 1, max 100 terms, convergence |term| < |sum| * 1e-10):
```
sum = 1/a; term = sum
for n = 1..100:
    term *= x / (a + n)
    sum += term
    if |term| < |sum| * 1e-10: break
return sum * exp(-x + a*ln(x) - lnGamma(a))
```

**Continued fraction** (for x >= a + 1, max 100 iterations, convergence |delta - 1| < 1e-10):
```
b = x + 1 - a
c = 1 / 1e-30
d = 1 / b; h = d
for i = 1..100:
    an = -i * (i - a)
    b += 2
    d = an*d + b;  if |d| < 1e-30: d = 1e-30
    c = b + an/c;  if |c| < 1e-30: c = 1e-30
    d = 1/d; del = d*c; h *= del
    if |del - 1| < 1e-10: break
return exp(-x + a*ln(x) - lnGamma(a)) * h
```

Asymptotic guard: if `x > a + 100`, return 1.0.

---

## 8. SIMD Optimizations in Analysis

The Java implementation provides SIMD-optimized dimension statistics computation as a
multi-release JAR override (Java 25+ with Panama Vector API). The Rust implementation
should use equivalent SIMD intrinsics.

### 8.1 Architecture

Runtime detection via `ComputeMode`:

| Mode | Vector Width | Float Lanes | Double Lanes | Requirements |
|------|-------------|-------------|--------------|--------------|
| PANAMA_AVX512F | 512-bit | 16 | 8 | Java 25+, AVX-512F CPU |
| PANAMA_AVX2 | 256-bit | 8 | 4 | Java 25+, AVX2 CPU |
| PANAMA_SSE | 128-bit | 4 | 2 | Java 25+, SSE CPU |
| SCALAR | — | 1 | 1 | Fallback |

CPU detection reads `/proc/cpuinfo` flags on Linux. `ComputeModeSpecies` provides
cached `VectorSpecies<Float>` and `VectorSpecies<Double>` for the detected mode.

### 8.2 SIMD Statistics Algorithm

**First pass** (min, max, sum) — 8-way loop unrolling:
```
vMin = broadcast(+Infinity)
vMax = broadcast(-Infinity)
vSum = broadcast(0.0)

for each group of 8 SIMD vectors (8 * LANES elements):
    v0..v7 = load 8 vectors from contiguous memory
    vMin = vMin.min(v0).min(v1)...min(v7)
    vMax = vMax.max(v0).max(v1)...max(v7)
    vSum = vSum.add(v0).add(v1)...add(v7)

// Reduce lanes
min = vMin.reduceLanes(MIN)
max = vMax.reduceLanes(MAX)
sum = vSum.reduceLanes(ADD)
// Scalar tail for remaining elements
```

**Second pass** (moments) — FMA-optimized:
```
vMean = broadcast(mean)
vM2 = vM3 = vM4 = broadcast(0.0)

for each SIMD vector:
    v = load(values, offset)
    diff = v - vMean
    diff2 = diff * diff
    vM2 = vM2 + diff2
    vM3 = diff2.fma(diff, vM3)     // m3 += diff^2 * diff
    vM4 = diff2.fma(diff2, vM4)    // m4 += diff^2 * diff^2
```

The 8-way unrolling reduces `reduceLanes()` overhead by doing more work per reduction.

### 8.3 Rust Implementation Notes

In Rust, use `std::simd` (portable SIMD) or `core::arch` intrinsics:
- `f64x4` (AVX2) or `f64x8` (AVX-512) for double-precision moment computation.
- `f32x8` (AVX2) or `f32x16` (AVX-512) for first-pass min/max/sum.
- Feature-gate with `#[cfg(target_feature = "avx2")]` and `#[cfg(target_feature = "avx512f")]`.
- Use `_mm256_fmadd_pd` / `_mm512_fmadd_pd` for fused multiply-add in moment computation.

---

## 9. Streaming / Chunked Analysis Architecture

### 9.1 DataSource Interface

Provides chunked access to vector data regardless of backing storage:

```
trait DataSource {
    fn shape(&self) -> DataspaceShape;       // (cardinality, dimensionality, layout)
    fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = Vec<Vec<f32>>>;
}
```

`DataspaceShape` contains:
- `cardinality`: total vector count (u64)
- `dimensionality`: number of dimensions per vector (usize)
- `layout`: ROW_MAJOR or COLUMNAR

### 9.2 TransposedChunkDataSource

Optimized data source that reads `.fvec` files and yields chunks in **column-major** format:
```
chunk[dimension_index][vector_index]
```

This layout is critical for cache-efficient per-dimension analysis.

**File format** (`.fvec`): Each vector is stored as a 4-byte little-endian int (dimension count)
followed by `dimension * 4` bytes of little-endian f32 values.

**Chunk size calculation** (`ChunkSizeCalculator`):
- Budget = available_heap * budget_fraction (default 0.6)
- bytes_per_vector = dimensions * 4 (f32) * overhead_factor (default 1.2)
- chunk_size = floor(budget / bytes_per_vector)
- Clamped to [1, total_vectors]

### 9.3 StreamingAnalyzer Interface

```
trait StreamingAnalyzer<M> {
    fn analyzer_type(&self) -> &str;
    fn initialize(&mut self, shape: DataspaceShape);
    fn accept(&mut self, chunk: &[&[f32]], start_index: u64);  // columnar chunk
    fn complete(self) -> M;
}
```

**Lifecycle**: `initialize` -> `accept` (repeated) -> `complete`.

**Thread safety**: `accept` may be called concurrently from multiple threads.

### 9.4 AnalyzerHarness

Orchestrates multiple analyzers over a data source:

1. Initialize all analyzers with shape.
2. For each chunk from data source:
   a. Load chunk (I/O phase).
   b. Convert to columnar format if source is row-major.
   c. Submit chunk to all analyzers concurrently (via thread pool).
   d. Wait for all to complete the chunk.
   e. Report progress.
   f. Check for early stopping (convergence).
3. Call `complete()` on all analyzers to collect results.

**Error handling**: Fail-fast mode (default) aborts all on first error. Non-fail-fast
continues other analyzers.

**Early stopping**: StreamingModelExtractor can signal convergence to stop processing
before all chunks are consumed.

### 9.5 StreamingDimensionAccumulator

Per-dimension accumulator using Welford's online algorithm:

```
for each new value x:
    count++
    delta = x - mean
    deltaN = delta / count
    deltaN2 = deltaN^2
    term1 = delta * deltaN * (count - 1)

    // Update M4 before M3, M3 before M2 (uses old values)
    m4 += term1 * deltaN2 * (count^2 - 3*count + 3)
        + 6 * deltaN2 * m2
        - 4 * deltaN * m3

    m3 += term1 * deltaN * (count - 2)
        - 3 * deltaN * m2

    m2 += term1

    mean += deltaN
```

**Critical ordering**: M4 must be updated before M3, and M3 before M2, because each
update depends on the **old** values of the lower moments.

**Properties**:
- O(1) memory per dimension.
- Numerically stable (no catastrophic cancellation).
- Exact equivalence to two-pass batch computation.
- Parallelizable via `combine()` using Chan's formulas (same as DimensionStatistics.combine).

### 9.6 StreamingHistogram

Fixed-bin histogram with adaptive bounds for incremental shape analysis:

- **Adaptive bounds**: Expand when new min/max values exceed current bounds, redistributing
  existing counts to new bins. Adds 10% margin to avoid immediate re-expansion.
- **Mode detection**: Moving-average smoothing (window=3), local maxima finding, prominence
  filtering (default threshold 0.1 of max count).
- **Gap detection**: Valley-to-peak contrast analysis. A significant gap has contrast ratio
  (valley_height / lower_peak_height) < 0.4. Uses wider smoothing (window=5) for cleaner
  peak/valley identification.
- Default bin count: 100.

### 9.7 StreamingModelExtractor

The primary streaming analyzer. Lifecycle:

1. **initialize**: Allocate per-dimension accumulators, histograms, and per-dimension locks.
2. **accept**: For each chunk (columnar format), update accumulators and histograms
   per dimension, with per-dimension locking for thread safety.
3. **complete**: For each dimension, convert accumulated statistics to DimensionStatistics,
   run BestFitSelector to pick best model, assemble into VectorSpaceModel.

---

## 10. Statistical Validation

### 10.1 Two-Sample Kolmogorov-Smirnov Test

Compares empirical CDFs of original and synthetic samples:

```
D = max|F1(x) - F2(x)|
critical_value = c(alpha) * sqrt((n1 + n2) / (n1 * n2))
pass = D < critical_value
```

Critical coefficients:
| alpha | c(alpha) |
|-------|----------|
| 0.01 | 1.63 |
| 0.05 | 1.36 |
| 0.10 | 1.22 |

**Note**: Use `(i64) n1 * n2` to avoid integer overflow for large samples.

### 10.2 Moment Comparison

Compare first four moments between original and synthetic data:

| Moment | Default Tolerance | Formula |
|--------|-------------------|---------|
| Mean | 1% of original sigma | `|mean_orig - mean_synth| < 0.01 * sigma_orig` |
| Variance | 5% relative | `|var_orig - var_synth| / var_orig < 0.05` |
| Skewness | 0.15 absolute | `|skew_orig - skew_synth| < 0.15` |
| Kurtosis | 0.5 absolute | `|kurt_orig - kurt_synth| < 0.5` |

### 10.3 Q-Q Correlation

Compute Pearson correlation between quantiles of original and synthetic samples:

1. Sort both samples.
2. Compute quantiles at 100 evenly-spaced points `(i + 0.5) / 100`.
3. Compute Pearson correlation coefficient.
4. Pass threshold: r > 0.995.

### 10.4 Comprehensive Dimension Accuracy

A dimension passes if ALL of:
- KS test passes (D < critical value at alpha = 0.05)
- All four moment comparisons pass
- Q-Q correlation > 0.995

---

## Proposed CLI Interface

```
veks analyze --input base_vectors.fvec --output model.json
veks analyze --input base_vectors.fvec --output model.json --selector pearson
veks analyze --input base_vectors.fvec --output model.json --selector full-pearson
veks analyze --input base_vectors.fvec --output model.json --selector multimodal --max-modes 6
veks analyze --input base_vectors.fvec --output model.json --chunk-size 50000
veks analyze --input base_vectors.fvec --output model.json --memory-budget 2G
veks analyze --input base_vectors.fvec --output model.json --validate --validate-count 10000
```

Options:
- `--input <path>`: Input vector file (.fvec format) (required).
- `--output <path>`: Output model JSON file (required).
- `--selector <preset>`: Fitting strategy preset (default, bounded, pearson, full-pearson,
  normalized, multimodal).
- `--max-modes <n>`: Maximum mixture components for multimodal selector (default 4, max 10).
- `--chunk-size <n>`: Vectors per processing chunk (auto-calculated if omitted).
- `--memory-budget <size>`: Memory budget for chunking (e.g., "2G", "512M").
- `--unique-vectors <n>`: Override unique vector count in output model (default: input count).
- `--validate`: After extraction, generate synthetic vectors and run statistical tests.
- `--validate-count <n>`: Number of synthetic vectors for validation (default 10000).
- `--strict`: Disable model simplification (Beta-to-Uniform aliasing, simplicity bias).
- `--progress`: Show progress during analysis.

---

## Verification Criteria

A correct reimplementation must satisfy:

1. **Model equivalence**: Given the same input vector data, the extracted model parameters
   must be statistically equivalent (within tolerance of numerical precision differences
   between Java and Rust floating-point).
2. **Round-trip fidelity**: Vectors generated from the extracted model (via doc 16's
   generation pipeline) must pass the statistical validation suite (KS test, moments,
   Q-Q correlation) when compared against the original data.
3. **JSON compatibility**: Model JSON produced by the Rust implementation must be
   loadable by the Java implementation and vice versa.
4. **Streaming correctness**: Statistics accumulated via streaming (chunked Welford's) must
   be algebraically equivalent to batch computation.
5. **Performance**: Single-threaded throughput should process at least 100K 128-dim vectors
   per second on modern hardware. SIMD-enabled builds should achieve 4-8x improvement.
HEREDOC_EOF
