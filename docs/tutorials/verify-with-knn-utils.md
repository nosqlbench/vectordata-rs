<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Cross-Verifying a Dataset Against knn_utils

The Python `knn_utils` project is the definitive quality reference
for vector search datasets. This tutorial shows how to verify a
dataset produced by vectordata-rs against knn_utils to confirm
correctness at every processing stage.

---

## Prerequisites

- A built veks binary with the `knnutils` feature:
  ```bash
  sudo apt install libopenblas-dev
  cargo build --release --features knnutils
  ```
- A source dataset (e.g., sift-128-euclidean from ann-benchmarks)
- Python 3 with `knn_utils` installed (for the reference run)

---

## Step 1: Build the dataset with knn_utils personality

The `--personality knn_utils` flag tells the bootstrap wizard to use
knn_utils-compatible commands at every pipeline stage:

```bash
mkdir sift128e && cd sift128e
cp /path/to/sift-128-euclidean.hdf5 _source.hdf5

veks bootstrap -i --personality knn_utils
```

The wizard detects the HDF5 source and prompts for dataset paths
(`train`, `test`, `neighbors`). Accept the defaults.

```bash
veks run dataset.yaml
```

The pipeline uses:
- `compute sort-knnutils` — lexicographic sort matching Python
- `generate shuffle-knnutils` — MT19937 PRNG matching numpy
- `transform normalize-knnutils` — numpy subprocess for byte-identical norms
- `compute knn-blas` — BLAS sgemm matching FAISS's distance matrix

---

## Step 2: Run the knn_utils verification

```bash
veks pipeline verify dataset-knnutils \
  --base profiles/base/base_vectors.fvec \
  --query profiles/base/query_vectors.fvec \
  --indices profiles/default/neighbor_indices.ivec \
  --neighbors 100 \
  --metric IP \
  --sample 1000
```

This runs the full knn_utils verification suite:

```
=== knn_utils Dataset Verification ===

--- Base vectors (fvecs_check) ---
  file: profiles/base/base_vectors.fvec
  vectors: 985462, dim: 128
  zero vectors: 0 (threshold 1e-6)
  normalization: PASS (max deviation 1.19e-7, tolerance 1e-5)
  duplicates: 0
  PASS

--- Query vectors (fvecs_check) ---
  file: profiles/base/query_vectors.fvec
  vectors: 10000, dim: 128
  PASS

--- Ground truth (ivecs_check) ---
  file: profiles/default/neighbor_indices.ivec
  queries: 10000, k: 100
  all ordinals in range [0, 985462)
  PASS

--- KNN accuracy (brute-force sample) ---
  sampled 1000 of 10000 queries
  exact match: 1000/1000
  PASS

=== ALL CHECKS PASSED ===
```

---

## Step 3: Compare against a Python knn_utils reference

If you have a dataset previously produced by the Python knn_utils,
you can compare byte-for-byte:

```bash
# Compare base vectors
cmp veks_output/profiles/base/base_vectors.fvec knn_utils_output/base_vectors.fvecs
# (no output = identical)

# Compare ground truth
cmp veks_output/profiles/default/neighbor_indices.ivec knn_utils_output/gt.ivecs
```

With the same BLAS (MKL) and knn_utils personality, these should
produce no output (byte-identical).

### If differences appear

Small differences at KNN boundaries are expected when using different
BLAS implementations (OpenBLAS vs MKL):

```bash
# Check neighbor set overlap
veks pipeline analyze compare-files \
  --source veks_output/profiles/default/neighbor_indices.ivec \
  --reference knn_utils_output/gt.ivecs
```

Typically 99.95%+ identical neighbor sets. Differences occur only
where multiple base vectors have the same distance to a query —
both orderings are valid.

---

## Step 4: Verify a native-personality dataset

For datasets built with the native personality (default, no
`--personality knn_utils`), use the standard verification:

```bash
veks run dataset.yaml
# Pipeline automatically runs:
#   verify-knn (brute-force sample recomputation)
#   verify-predicates-sqlite (SQLite oracle for predicate results)
#   verify-filtered-knn (filtered brute-force recomputation)
```

The native pipeline uses different algorithms (SimSIMD distances,
PCG PRNG, prefix sort) but produces self-consistent results verified
by independent brute-force recomputation at each stage.

To additionally cross-verify against knn_utils:

```bash
veks pipeline verify dataset-knnutils \
  --base profiles/base/base_vectors.fvec \
  --query profiles/base/query_vectors.fvec \
  --indices profiles/default/neighbor_indices.ivec \
  --neighbors 100 \
  --metric L2 \
  --sample 1000
```

This confirms the native pipeline's output passes all knn_utils
validation checks (format, zeros, normalization, duplicates, KNN
accuracy).

---

## Verification levels

| Level | What it means | When to use |
|-------|--------------|-------------|
| **Byte-identical** | Every byte matches knn_utils output | Same BLAS + knn_utils personality |
| **Set-equivalent** | Same neighbors, possibly different order at ties | Different BLAS |
| **Self-consistent** | Pipeline's own brute-force confirms results | Native personality, any dataset |

All three levels confirm correctness. Published datasets should pass
at least self-consistent verification. Byte-identical against a
knn_utils reference is the gold standard.

---

## Quick reference

```bash
# Build with knn_utils support
cargo build --release --features knnutils

# Bootstrap with knn_utils personality
veks bootstrap -i --personality knn_utils

# Run pipeline
veks run dataset.yaml

# Verify against knn_utils standards
veks pipeline verify dataset-knnutils \
  --base base.fvec --query query.fvec --indices gt.ivec \
  --neighbors 100 --metric IP --sample 1000

# A/B test (requires --features knnutils,faiss)
veks pipeline compute knn-faiss \
  --base base.fvec --query query.fvec \
  --indices gt_faiss.ivec --distances gt_faiss.fvec \
  --neighbors 100 --metric IP
```
