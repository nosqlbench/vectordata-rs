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
