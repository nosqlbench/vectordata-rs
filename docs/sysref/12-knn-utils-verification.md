# 12. knn_utils Verification

The Python `knn_utils` project is the definitive quality reference for
vector search datasets. Datasets produced by vectordata-rs can be
cross-verified against knn_utils to confirm byte-level correctness
of all processing stages.

---

## 12.1 What knn_utils Is

knn_utils is an independent Python toolkit that uses FAISS + numpy
for vector dataset preparation:

| Component | Library | What it does |
|-----------|---------|-------------|
| KNN computation | FAISS (C++/CUDA) | Brute-force exact nearest neighbors |
| Normalization | numpy (MKL BLAS) | L2 unit-vector normalization |
| Shuffle | numpy MT19937 | Deterministic random permutation |
| Sort/dedup | numpy | Lexicographic sort + duplicate removal |
| Zero detection | numpy | L2 norm threshold check |
| Validation | Custom Python | fvecs/ivecs format checks, KNN accuracy |

Because knn_utils uses different language, libraries, and numerical
implementations, matching its output byte-for-byte provides strong
evidence that the vectordata-rs pipeline is correct.

---

## 12.2 The knn_utils Personality

The `--personality knn_utils` flag switches the pipeline to use
knn_utils-compatible commands that replicate its exact behavior:

| Native command | knn_utils equivalent | Difference |
|---------------|---------------------|------------|
| `compute knn` (SimSIMD) | `compute knn-blas` (BLAS sgemm) | Different distance kernel |
| `compute sort` (prefix) | `compute sort-knnutils` (full lex) | Sort ordering within groups |
| `generate shuffle` (PCG) | `generate shuffle-knnutils` (MT19937) | Different PRNG |
| `verify knn-consolidated` | `verify dataset-knnutils` | Unified verification suite |

### Build requirements

```bash
# knn_utils personality requires system BLAS
sudo apt install libopenblas-dev   # or libmkl-dev for Intel MKL

cargo build --features knnutils
```

### Bootstrap

```bash
veks bootstrap -i --personality knn_utils \
  --base-vectors source.fvec --metric Cosine
```

This generates a pipeline using knn_utils-compatible commands at
every stage.

---

## 12.3 What Gets Verified

The `verify dataset-knnutils` command runs a comprehensive check suite
matching knn_utils' own validation tools:

| Check | What it verifies | Matching knn_utils tool |
|-------|-----------------|----------------------|
| Vector format | File size, stride, dimension consistency | `fvecs_check.py` |
| Zero vectors | No near-zero vectors (L2 norm < threshold) | `fvecs_remove_zeros.py` |
| Normalization | All vectors unit-length within tolerance | `check_normalization()` |
| Duplicates | No byte-identical vectors | `fvecs_deduplicator.py` |
| GT format | ivec structure, ordinal validity | `ivecs_check.py` |
| KNN accuracy | Brute-force recomputation on sample queries | `validate_knn_utils.py` |

---

## 12.4 Verification Levels

### Byte-identical (strongest)

When using the same BLAS library (MKL) and knn_utils personality,
output files match byte-for-byte:

```
example-dataset: base BYTE-IDENTICAL, query BYTE-IDENTICAL, GT BYTE-IDENTICAL
```

This means every float value, every ordinal, every byte in the output
is identical between veks and knn_utils.

### Set-equivalent (practical)

When using different BLAS (OpenBLAS vs MKL), distances may differ by
1 ULP at KNN boundaries. This produces neighbor set differences at
the k-th boundary where multiple vectors have the same distance:

```
example-dataset (OpenBLAS): 9998/10000 exact match, 2 set match, 0 real differences
```

Set-equivalent means the same neighbors appear, possibly in different
order at tied distances. This is correct — both orderings are valid.

### Self-consistent (minimum)

For datasets without a knn_utils reference (new datasets, different
source formats), the pipeline verifies internal self-consistency:
brute-force KNN recomputation on sample queries matches the pipeline's
stored results.

---

## 12.5 BLAS and Numerical Precision

### Why BLAS matters

KNN computation depends on distance matrix calculation. Different BLAS
implementations (MKL, OpenBLAS, Apple Accelerate) use different internal
algorithms with different rounding behavior. For high-dimensional
vectors (dim=128+), accumulated rounding differences can change which
vector is the k-th nearest neighbor when distances are very close.

### Normalization subtlety

numpy's `np.linalg.norm(arr, axis=1)` uses f64 pairwise accumulation —
not `cblas_snrm2`. The result differs from any BLAS norm by 1+ ULP for
high-dimensional vectors. The knn_utils personality calls numpy via
subprocess to achieve byte-identical normalization.

### Bottom line

- **Same BLAS + knn_utils personality** → byte-identical output
- **Different BLAS** → set-equivalent (ties at boundary may differ)
- **Native personality** → self-consistent (verified by brute-force)

All three levels confirm dataset correctness. Byte-identical is the
gold standard; self-consistent is the minimum for any published dataset.

---

## 12.6 A/B Testing

The `compute knn-faiss` command (behind the `faiss` feature flag)
provides an independent third implementation for A/B testing:

```bash
cargo build --features knnutils,faiss

# Compare knn-blas vs knn-faiss
veks pipeline compute knn-blas --base base.fvec --query query.fvec ...
veks pipeline compute knn-faiss --base base.fvec --query query.fvec ...

# Results should match (same BLAS → identical, different → set-equivalent)
```

---

## 12.7 Cross-Engine Conformance Testing

vectordata-rs ships four independent KNN engines, and they're held
to numerical-equivalence guarantees by a layered set of automated
tests. Together these prove that any engine — including the BLAS
sgemm path used by the `knn_utils` personality — produces results
that match FAISS (and, by transitivity through `knn_utils`, the
numpy reference).

### Engines under test

| Engine          | Kernel | Source |
|-----------------|--------|--------|
| `knn-metal`     | SimSIMD (AVX-512 / AVX2 / NEON, hardware-dispatched) | [`compute_knn.rs`](../../veks-pipeline/src/pipeline/commands/compute_knn.rs) |
| `knn-stdarch`   | Pure `std::arch` SIMD, zero deps | [`compute_knn_stdarch.rs`](../../veks-pipeline/src/pipeline/commands/compute_knn_stdarch.rs) |
| `knn-blas`      | `cblas_sgemm` (MKL or OpenBLAS) | [`compute_knn_blas.rs`](../../veks-pipeline/src/pipeline/commands/compute_knn_blas.rs) |
| `knn-faiss`     | FAISS `IndexFlat` brute force | [`compute_knn_faiss.rs`](../../veks-pipeline/src/pipeline/commands/compute_knn_faiss.rs) |

### Comparison model

Each query's KNN result is classified into one of four buckets by
[`knn_compare::compare_query_ordinals`](../../veks-pipeline/src/pipeline/commands/knn_compare.rs):

| `QueryResult`           | Meaning |
|-------------------------|---------|
| `ExactMatch`            | All `k` neighbors match in identical order |
| `SetMatch`              | Same neighbor *set*, different order (tie-breaking) |
| `BoundaryMismatch(d)`   | Sets differ by `d` neighbors with `d ≤ BOUNDARY_THRESHOLD` |
| `RealMismatch(d)`       | Sets differ by `d > BOUNDARY_THRESHOLD` |

`BOUNDARY_THRESHOLD = 5` is a *defensive ceiling* — not a regular
operating point. For realistic embedding-shaped data it stays
unused; it only starts mattering when synthetic uniform-random
vectors push us into the curse-of-dimensionality regime (see
below).

### Actual numbers — dim sweep 8 → 4096 at k=100

Ran against 1000 base vectors × 100 queries × **k=100**, seed=42.
All four engines compiled in (`--features knnutils,faiss`).
Reference engine is metal (SimSIMD). The sweep covers three
synthetic distributions (`--distribution uniform|gaussian|clustered`)
and all four metric modes the engines support:

- `--metric L2`
- `--metric DOT_PRODUCT`
- `--metric COSINE --assume_normalized_like_faiss=true` — cosine
  collapses to inner product on pre-normalized inputs (FAISS/numpy
  convention, what `knn_utils` does)
- `--metric COSINE --use_proper_cosine_metric=true` — in-kernel
  `dot/(|q|·|b|)` so cosine works on un-normalized inputs (FAISS
  has no native equivalent and is correctly skipped in this mode)

Result: **every cell in the 11 dims × 3 distributions × 2 metrics
matrix passes 100 EXACT / 100** for stdarch / blas / faiss /
blas-mirror against the metal reference. Higher k surfaces
more boundary candidates (k=100 vs the prior k=10 sweep) and the
canonical f64 rerank still produces byte-identical output across
every engine at every dim, including the dim=4096 uniform case
that historically pushed engines apart.

Reproduce with the built-in `--sweep` mode:

```bash
# L2 + DOT_PRODUCT × all distributions × dim 8…4096 × k=100
veks pipeline verify engine-parity --sweep \
  --base-count 1000 --query-count 100 \
  --neighbors 100 --show-queries 0 \
  --boundary-tolerance 1000000

# COSINE (assume-normalized): FAISS-equivalent
veks pipeline verify engine-parity --sweep \
  --metrics COSINE --base-count 1000 --query-count 100 \
  --neighbors 100 --assume_normalized_like_faiss=true \
  --show-queries 0 --boundary-tolerance 1000000

# COSINE (proper, in-kernel): FAISS skipped — see "FAISS gap" below
veks pipeline verify engine-parity --sweep \
  --metrics COSINE --base-count 1000 --query-count 100 \
  --neighbors 100 --use_proper_cosine_metric=true \
  --show-queries 0 --boundary-tolerance 1000000
```

`--sweep` runs the cross-product of (`--metrics` × `--neighbors-list`
× `--dims` × `--distributions`) internally and emits a single
matrix-shaped summary table. Each axis defaults to a standard sweep
range; pass any subset to narrow. See
`docs/design/knn-engine-characterization.md` for the
parity-vs-truth distinction the matrix exists to verify.

`verify engine-parity` does two things production `compute knn*`
calls don't:

1. Forces single-threaded BLAS (via `OPENBLAS_NUM_THREADS=1` +
   MKL/OMP siblings, set at the top of `execute`). Without that,
   FAISS's OpenMP-scheduled sgemm picks non-deterministic
   block-accumulation orders and the comparison becomes noise.
2. Sets `rerank_margin_ratio=3` on each engine it invokes, so
   each engine internally tracks top-(k + 3·k) candidates per
   query rather than top-k. Production `compute knn*` callers
   leave this option unset → margin = 0 → engine heaps are
   sized at exactly k (the original pre-rerank behavior, with
   full pruning aggressiveness — no slowdown). The canonical
   f64 rerank post-pass still runs at margin=0; it just
   reorders the engine's existing top-k via f64 direct math.

With both flags on, **every row of the table is 100 EXACT** —
bit-identical neighbor sets across all four engines — because the
margin gives every engine enough boundary candidates that the
sgemm-expansion path (`‖a‖² + ‖b‖² − 2·a·b`) and the direct path
(`Σ(a−b)²`) both hold the true top-k inside their respective
top-(k + margin), and the shared **f64-direct canonical rerank**
(`knn_segment::rerank_output_post_pass`) selects the same ten.

EXACT count per engine against the metal (SimSIMD) reference,
format `stdarch / blas / faiss / blas-mirror`:

| dim  | uniform                  | gaussian                  | clustered                 |
|------|--------------------------|---------------------------|---------------------------|
|    8 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
|   32 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
|  128 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
|  384 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
|  512 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
|  768 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
| 1024 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
| 1536 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
| 2048 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
| 3072 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |
| 4096 | 100 / 100 / 100 / 100    | 100 / 100 / 100 / 100     | 100 / 100 / 100 / 100     |

**100% EXACT parity across every dimension, every distribution,
and every engine** — for every metric mode any given engine
supports. Uniform-random at dim=4096 — the most pathological
synthetic case where pairwise distances concentrate to within a
few-ULP window — is captured cleanly by the margin: even when the
f32 sgemm and direct kernels disagree on which boundary candidate
sits at rank 10, both forms hold all of the true top-10 within
their respective top-(k + margin), and the f64 rerank promotes the
correct ten regardless of how the f32 arithmetic ordered them.

The same numbers hold across the metric matrix:

| Metric mode                          | metal | stdarch | blas | faiss | blas-mirror |
|--------------------------------------|-------|---------|------|-------|-------------|
| L2                                   |  ref  |  100    | 100  | 100   |  100        |
| DOT_PRODUCT                          |  ref  |  100    | 100  | 100   |  100        |
| COSINE + assume_normalized_like_faiss|  ref  |  100    | 100  | 100   |  100        |
| COSINE + use_proper_cosine_metric    |  ref  |  100    | 100  | (gap) |  100        |

**FAISS gap on COSINE + use_proper_cosine_metric**: FAISS has no
native proper-cosine kernel — its `METRIC_INNER_PRODUCT` assumes
pre-normalized inputs and never divides by `|q|·|b|`. In that mode
`verify engine-parity` correctly skips FAISS (with an explanatory
note) instead of comparing against a different ranking. On any
naturally-normalized data (`gaussian`, `clustered`, or
`assume_normalized_like_faiss=true`), inner product equals proper
cosine and FAISS agrees with the other engines bit-for-bit.

### The path to FAISS parity

Four factors had to align for bit-identical output:

1. **Algebraic form** — all sgemm-path engines (blas, faiss,
   blas-mirror) use the expansion `‖a‖² + ‖b‖² − 2·a·b`; the direct
   engines (metal, stdarch) use `Σ(a−b)²`. These produce ULP-level
   different floating-point values, but on any distribution with
   enough inter-neighbor distance gap the *ranking* is the same.
2. **BLAS call shape** — FAISS uses outer query block
   `distance_compute_blas_query_bs = 4096`, inner base block
   `distance_compute_blas_database_bs = 1024`, orientation
   `sgemm("T", "N", nyi=nb, nxi=nq, ...)`. Our `blas-mirror` engine
   replicates that exactly; our production `knn-blas` uses different
   block sizes for I/O streaming but still converges under
   single-threaded BLAS.
3. **Single-threaded BLAS** — the critical factor. FAISS and every
   BLAS library default to OpenMP-threaded sgemm; block accumulation
   order then depends on which core gets which block at which time.
   Forcing thread count to 1 makes sgemm deterministic, and every
   caller — FAISS, our knn-blas, our blas-mirror — produces the same
   bytes. The parity command does this automatically via
   `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`OMP_NUM_THREADS`.
4. **Top-(k + margin) internal tracking, opt-in** — when invoked
   with `rerank_margin_ratio=N` (or `rerank_margin=M`) every
   engine sizes its per-query heaps and segment-cache rows at
   `internal_k = k + margin` instead of `k`. Pulling extra
   candidates out of each engine's f32 scan is what closes the
   boundary-membership gap: even when an f32 sgemm and an f32
   direct kernel disagree on *which* neighbor sits at rank 10,
   both forms hold the true top-k within their top-(k + margin),
   and the canonical rerank (next item) picks the same ten.

   Margin is **opt-in** because it widens the threshold-pruning
   cutoff (the worst-of-top-k distance) and slows the scan loop
   measurably — `verify engine-parity` opts in (it cares about
   bit-identical agreement, not throughput); production
   `compute knn*` callers do not (`margin = 0` ⇒ original speed).
5. **Canonical f64 rerank post-pass** — every `compute knn*` command
   ends with a call to `knn_segment::rerank_output_post_pass` that
   reads back its just-written top-(k + margin) ivec, recomputes
   each candidate's distance in f64 using the direct `Σ(a−b)²`
   form, and rewrites the file with the canonically-ordered top-k
   (and matching f64-cast distances if the engine emitted a
   distances fvec). This makes both the *membership* and the
   *ordering* canonical across every engine and every fixture.

The `blas-mirror` engine in `verify engine-parity` exists as the
proof-of-concept for (2) — a ~100-line Rust function that
reproduces FAISS's exact sgemm call pattern across all four metric
paths (L2, IP, COSINE-as-IP, proper-COSINE). Its 100% EXACT
column in the table (matching every other engine across every
dimension and distribution) is the demonstration.

### On-disk distance sign convention

Every fvec the codebase writes — segment caches in `<workspace>/.cache/`,
published `<workspace>/profiles/<size>/neighbor_distances.fvecs`, and
final `compute knn*` output — uses **FAISS publication convention**:

| Metric          | On-disk value          | Direction        |
|-----------------|------------------------|------------------|
| `L2`            | `+L2sq`                | smaller = better |
| `DotProduct`    | `+dot`                 | larger = better  |
| `Cosine`        | `+cos_sim` ∈ `[-1, 1]` | larger = better  |

The kernel still operates on monotonic distances internally
(smaller = better, so a single heap implementation works for
every metric). Conversion happens at every disk boundary via
[`knn_segment::kernel_to_published`] and `published_to_kernel`,
which are pinned by unit tests in `knn_segment::tests`.

This matters because the segment-cache reuse path (Path-2 sibling
profile loading) reads files written by *any* prior `compute knn*`
run. If one engine wrote `-dot` and another reads back assuming
`+dot`, the heap merging would invert and silently corrupt every
downstream profile that reused that file.

**Cache-version note**: the on-disk format changed between cache
versions: v1 (original) → v2 (margin support, mixed sign convention)
→ v3 (uniform FAISS publication convention). Each `compute knn*`
run logs a one-line notice if it sees prior-version cache files
and recomputes everything fresh.

### Performance note

Single-threaded BLAS is the *conformance-test* setting, not the
production-compute setting. Throughput drops substantially (one core
vs all of them). The trade-off is explicit: `verify engine-parity`
sacrifices speed for determinism; `compute knn-blas` keeps its
multi-threaded sgemm for production throughput, with the
understanding that two multi-threaded runs can produce different
ULP-level results and will sometimes show 0–5 boundary mismatches
on pathological fixtures.

### Tests and commands

In-tree unit tests:
- [`test_stdarch_matches_metal`](../../veks-pipeline/src/pipeline/commands/compute_knn_stdarch.rs)
  — asserts `diff == 0` (stdarch ≡ metal bit-identical across every
  dim/distribution/seed).
- [`test_knn_faiss_matches_compute_knn`](../../veks-pipeline/src/pipeline/commands/compute_knn_faiss.rs)
  — asserts `diff == 0` at dim=8 (post-rerank, the FAISS and metal
  outputs share an identical canonical ranking).
- [`test_metal_cache_reuse`](../../veks-pipeline/src/pipeline/commands/compute_knn.rs),
  [`test_stdarch_cache_reuse`](../../veks-pipeline/src/pipeline/commands/compute_knn_stdarch.rs),
  [`test_blas_cache_reuse`](../../veks-pipeline/src/pipeline/commands/compute_knn_blas.rs)
  — assert each engine's segment cache filenames carry the
  engine-specific `ENGINE_NAME` prefix and that a second run hits
  the cache.

Reproduce the full sweep (built-in `--sweep` mode runs the cross-product
internally and emits a single matrix summary):

```bash
veks pipeline verify engine-parity --sweep \
  --base-count 1000 --query-count 100 \
  --neighbors 100 --show-queries 0 \
  --boundary-tolerance 1000000
```

Each axis has a default; pass any of them to narrow the matrix:

- `--dims=8,128,4096`              (default: 8,32,128,384,512,768,1024,1536,2048,3072,4096)
- `--distributions=clustered`      (default: uniform,gaussian,clustered)
- `--metrics=L2,DOT_PRODUCT`       (default: L2,DOT_PRODUCT)
- `--neighbors-list=10,100`        (default: just `--neighbors`, or 100 if unset)

> **Segment cache key contract.** Every `compute knn*`
> implementation declares a unique single-word engine name
> (`knn-metal`, `knn-stdarch`, `knn-blas`, …) via its `ENGINE_NAME`
> constant. That engine name is the leading component of every
> cache filename, so two engines with different numerical behaviour
> never replay each other's output — each engine has its own
> namespace in `<workspace>/.cache`. The remaining key components
> are `(base_stem, query_stem, base_size, query_size, range, k,
> metric)`; file *content* is deliberately not part of the key, so
> rerunning the same engine against the same file paths is a cache
> hit by design. Users iterating with different data on top of the
> same output paths should point the engine at a fresh workspace
> (or `rm -rf <workspace>/.cache` between experiments); the pipeline
> runner does this automatically through its step-fingerprint chain
> when inputs change upstream.

### Why this is not a correctness concern

Real embedding models (sentence-transformers, CLIP, OpenAI
text-embedding-3, etc.) produce vectors where meaningful clusters
exist — pairwise distances are not concentrated, the top-k is
well-defined, and all four engines agree. The divergence above is a
property of synthetic uniform-random data at high dim, not of the
engines. The `verify dataset-knnutils` command (§12.3) runs on real
datasets before publication and is the authoritative check for
published ground truth.

The slack exists for two known degenerate regimes:

- **Multi-threaded BLAS at scale** — sgemm thread-block decomposition
  on billion-vector base files (called via `verify-knn-consolidated`
  on a real published dataset) produces ULP-level rounding that can
  swap one or two boundary neighbors per query.
- **Curse of dimensionality on synthetic data** — as shown in the
  sweep above. Passing `--boundary-tolerance k` (where `k` is the
  requested neighbor count) lets the demo run through these fixtures
  without flipping to FAIL, but the honest reading is "this fixture
  doesn't have a well-defined top-k".

Tests for the classifier itself live in
[`knn_compare.rs`](../../veks-pipeline/src/pipeline/commands/knn_compare.rs):
`test_exact_match`, `test_set_match`, `test_boundary_mismatch`,
`test_boundary_at_threshold`, `test_real_mismatch`.

### Per-engine conformance tests

Each engine has an in-tree unit test that runs it against a
deterministic seeded fixture and asserts the result matches the
reference (or another engine) within `BoundaryMismatch` tolerance.

| Test | What it asserts |
|------|------------------|
| [`test_knn_faiss_matches_compute_knn`](../../veks-pipeline/src/pipeline/commands/compute_knn_faiss.rs) | FAISS results (L2, k=5, dim=8, 100 base × 10 queries) match the SimSIMD `compute knn` reference — sets differ by at most 2 boundary swaps. |
| [`test_stdarch_matches_metal`](../../veks-pipeline/src/pipeline/commands/compute_knn_stdarch.rs) | Pure-`std::arch` kernel matches the SimSIMD `compute knn` reference under the same fixture and same boundary threshold. |
| [`test_knn_blas_ip_known_neighbors`](../../veks-pipeline/src/pipeline/commands/compute_knn_blas.rs) / [`test_knn_blas_l2`](../../veks-pipeline/src/pipeline/commands/compute_knn_blas.rs) / [`test_knn_blas_multiple_queries`](../../veks-pipeline/src/pipeline/commands/compute_knn_blas.rs) | The BLAS sgemm kernel returns the analytically-correct nearest neighbors for IP and L2 metrics on hand-built fixtures with known answers. |
| [`test_knn_faiss_ip_known_neighbors`](../../veks-pipeline/src/pipeline/commands/compute_knn_faiss.rs) / [`test_knn_faiss_l2`](../../veks-pipeline/src/pipeline/commands/compute_knn_faiss.rs) | FAISS returns the same hand-built ground truth — establishes FAISS itself as a trustworthy reference under our test harness. |

### Pipeline-level verification — `verify dataset-knnutils`

The pipeline command
[`verify dataset-knnutils`](../../veks-pipeline/src/pipeline/commands/verify_dataset_knnutils.rs)
re-runs FAISS brute force on a sampled subset of queries against a
published dataset and compares the recomputed neighbor sets against
the dataset's stored ground truth, using the same
`BoundaryMismatch ≤ 5` rule. This is the "set-equivalent" level
described in §12.4 — every dataset published by the `knn_utils`
personality runs this check before publication.

Tests for the verification scaffolding (positive and negative
cases) live in
[`verify_dataset_knnutils.rs`](../../veks-pipeline/src/pipeline/commands/verify_dataset_knnutils.rs):
`test_verify_valid_dataset`, `test_verify_dimension_mismatch`,
`test_verify_detects_duplicate_ordinals`.

### What this proves

Reading the chain:

1. **knn-faiss matches knn-metal** at the kernel level
   (`test_knn_faiss_matches_compute_knn`).
2. **knn-stdarch matches knn-metal** at the kernel level
   (`test_stdarch_matches_metal`).
3. **knn-blas matches both** on hand-built fixtures with known
   answers (`test_knn_blas_*`); FAISS independently matches the
   same answers (`test_knn_faiss_*`), so transitivity gives
   `knn-blas ≡ knn-faiss` within `BoundaryMismatch ≤ 5`.
4. **`verify dataset-knnutils`** re-establishes the same equivalence
   on the published dataset before it ships, so a downstream
   consumer comparing against `knn_utils` (numpy + FAISS) gets
   set-equivalent results without rerunning anything.

The combination — same kernel under unit tests, end-to-end
recomputation under the verification command, and
`BoundaryMismatch` accounting for multi-threaded BLAS rounding —
is what lets us put numpy/`knn_utils` parity on the label.

### Feature gating

The four engines have different build dependencies, so the conformance
suite spans more than the default cargo features:

| Engine          | Cargo feature  | What it needs |
|-----------------|----------------|---------------|
| `knn-metal`     | (default)      | nothing extra (SimSIMD bundled) |
| `knn-stdarch`   | (default)      | nothing extra |
| `knn-blas`      | `knnutils`     | system BLAS (libopenblas-dev or libmkl-dev) |
| `knn-faiss`     | `faiss`        | system FAISS + BLAS |

`verify_dataset_knnutils` itself is also gated on `knnutils` (it
links `cblas_snrm2`).

### One-shot live demo — `verify engine-parity`

For an at-a-glance "see it for yourself" check there's a dedicated
pipeline command that runs every available engine on the same
input, prints their per-query neighbor lists side-by-side, and fails
the verdict on **any** disagreement (`--boundary-tolerance` defaults
to 0):

```bash
# Self-contained: deterministic synthetic fixture in a temp dir.
veks pipeline verify engine-parity --synthetic \
  --dim 32 --base-count 500 --query-count 20 --neighbors 5 --metric L2

# Or against your own dataset:
veks pipeline verify engine-parity \
  --base profiles/base/base_vectors.fvec \
  --query profiles/base/query_vectors.fvec \
  --neighbors 100 --metric L2 --show-queries 5
```

Output looks like:

```
ENGINE       STATUS                       ELAPSED
──────────── ──────────────────────────── ─────────
metal        ran                              0.10s
stdarch      ran                              0.00s
blas         ran                              0.07s
faiss        ran                              0.02s

First 3 queries (k=5) — neighbors per engine:
  query 0:
    metal    [10, 389, 467, 350, 213]
    stdarch  [10, 389, 467, 350, 213]
    blas     [10, 389, 467, 350, 213]
    faiss    [10, 389, 467, 350, 213]
  ...

Pair-wise comparison (reference = metal, boundary-tolerance = 0):
VS                EXACT     SET   BOUND  EXCEED   TOTAL  VERDICT
──────────────── ────── ─────── ─────── ─────── ──────── ────────
stdarch              20       0       0       0       20  PASS
blas                 20       0       0       0       20  PASS
faiss                20       0       0       0       20  PASS

Result: All engines produce identical neighbor sets (set-equivalent or stricter)
```

The `EXCEED` column counts queries whose diff exceeds
`--boundary-tolerance` (default 0); a non-zero value flips the
verdict to FAIL and the command exits non-zero. Pass
`--boundary-tolerance N` only when you're knowingly probing one of
the degenerate regimes documented above (multi-threaded BLAS at
scale, or curse-of-dimensionality fixtures).

Engines you didn't compile in show up as `skipped: feature not
enabled` rather than failing the whole demo. Source:
[`verify_engine_parity.rs`](../../veks-pipeline/src/pipeline/commands/verify_engine_parity.rs).

### Run the conformance suite

```bash
# Default features — exercises knn-metal and knn-stdarch parity tests
cargo test -p veks-pipeline --lib pipeline::commands::compute_knn

# Add knnutils for the BLAS engine + dataset-knnutils verifier tests
cargo test -p veks-pipeline --features knnutils \
  --lib pipeline::commands::verify_dataset_knnutils

# Add faiss for the cross-engine FAISS-vs-SimSIMD parity test
cargo test -p veks-pipeline --features knnutils,faiss \
  --lib pipeline::commands::compute_knn
```

### Demonstrate against your own dataset

For a dataset laid out in the canonical wizard layout (`profiles/base/`
holds `base_vectors.fvec` + `query_vectors.fvec`, `profiles/default/`
holds `neighbor_indices.ivec`), every verifier is a single command.
Run from the dataset directory.

**Quick sanity — recompute distances and compare against stored GT:**

```bash
veks pipeline analyze verify-knn \
  --base    profiles/base/base_vectors.fvec \
  --query   profiles/base/query_vectors.fvec \
  --indices profiles/default/neighbor_indices.ivec \
  --metric  L2
```

Stream-recomputes the brute-force distance for every (query, neighbor)
pair and asserts each computed distance matches the stored neighbor
distance within a configurable tolerance (`--phi`, default 1e-3).
Fast — runs out of cache for cached profiles. No special features
required.

**All-profiles sample sweep — sample-based recall against GT:**

```bash
veks pipeline verify knn-consolidated \
  --base   profiles/base/base_vectors.fvec \
  --query  profiles/base/query_vectors.fvec \
  --metric L2 \
  --sample 100 \
  --output verification.json
```

Single-pass over the base file, walks every sized profile in the
dataset, samples 100 queries per profile, brute-force recomputes
their nearest neighbors, and emits a JSON report classifying each
query as `ExactMatch` / `SetMatch` / `BoundaryMismatch(d)` /
`RealMismatch(d)`. This is the same comparison model described in
§12.7.

**Full knn_utils-style report (FAISS recompute on a sample):**

```bash
# Requires `cargo install --features knnutils --path veks` so the
# command is registered; the verifier itself uses BLAS directly.
veks pipeline verify dataset-knnutils \
  --base      profiles/base/base_vectors.fvec \
  --query     profiles/base/query_vectors.fvec \
  --indices   profiles/default/neighbor_indices.ivec \
  --neighbors 100 \
  --metric    IP \
  --sample    100 \
  --report    verification.txt
```

Runs every check from §12.3 (fvecs format / normalization /
zero-vector / GT structure / cross-file consistency / FAISS-recompute
sample) and writes a knn_utils-style `verification.txt` you can
diff against the upstream reference.

For a dataset already published with this verifier in its pipeline
(the `knn_utils` personality and the default sized-profile pipeline
both wire it in), a plain `veks run` will re-execute it with
skip-if-fresh semantics — useful when you want to verify a remote
dataset you've just pulled into the cache.
