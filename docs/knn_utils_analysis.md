# knn_utils Analysis & veks Personality System

## Overview

This document catalogs the Python `knn_utils` project (linked at
`links/knn_utils`), maps its dependencies to veks equivalents, and
documents the `--personality knn_utils` pipeline system that reproduces
knn\_utils behavior with byte-identical results.

---

## 1. Library Inventory & Native Provenance

### Python Libraries → Upstream C/C++

| Python Library   | Version  | Upstream C/C++ Library                  | Role in knn\_utils                                         |
|------------------|----------|-----------------------------------------|------------------------------------------------------------|
| **faiss-gpu**    | 1.9.0    | Facebook FAISS (C++) + CUDA 12.1.1      | Exact brute-force KNN: `IndexFlatL2`, `IndexFlatIP`; multi-GPU sharding |
| **numpy**        | 1.26.4   | Intel MKL 2023.1.0 (BLAS/LAPACK)        | L2 norms, array reshape, masking, histograms, binary I/O   |
| **h5py**         | (pinned) | HDF5 C library (libhdf5)                | Lazy-loading vector datasets from HDF5 files                |
| **pyarrow**      | (pinned) | Apache Arrow/Parquet (C++)              | Reading Parquet-format files                                |

### veks Equivalents

| knn\_utils Dependency | veks Approach | Feature Flag |
|-----------------------|---------------|-------------|
| FAISS (KNN) | Direct BLAS `cblas_sgemm` + Rust top-k heap | `knnutils` |
| FAISS (A/B testing) | `faiss` crate with vendored FAISS 1.9.0 | `faiss` |
| numpy (normalization) | numpy subprocess call (byte-identical) | `knnutils` |
| numpy (norms) | BLAS `cblas_snrm2` / `cblas_sdot` | `knnutils` |
| numpy (shuffle) | Rust MT19937 (`rand_mt`) with `rk_interval` | `knnutils` |
| h5py (HDF5 I/O) | **Removed** — pre-convert to fvec format | — |
| Arrow/Parquet | `arrow-rs` + `parquet` (pure Rust) | always |

---

## 2. Feature Flags

```
cargo build                        # vanilla — no knnutils, no BLAS needed
cargo build --features knnutils    # knn_utils personality (needs system libopenblas-dev)
cargo build --features faiss       # adds FAISS for A/B testing (needs cmake + g++)
```

| Feature | Build Requirements | What It Enables |
|---------|-------------------|-----------------|
| (none) | vanilla cargo | Native SimSIMD pipeline only |
| `knnutils` | `libopenblas-dev` (apt) | knn\_utils personality commands, BLAS sgemm KNN |
| `faiss` | cmake + g++ + BLAS | `compute knn-faiss` for A/B verification |

---

## 3. knn\_utils Personality Commands

All behind `#[cfg(feature = "knnutils")]`:

| Command | Purpose | Matching knn\_utils Tool |
|---------|---------|------------------------|
| `compute knn-blas` | Brute-force KNN via BLAS sgemm | `knn_utils.py` (FAISS search) |
| `compute sort-knnutils` | Full lexicographic sort + dedup | `fvecs_deduplicator.py` |
| `generate shuffle-knnutils` | MT19937 shuffle with rk\_interval | `np.random.shuffle` / `fvecs_shuffle.py` |
| `transform normalize-knnutils` | Normalization via numpy (byte-identical) | `fvecs_normalize.py` / `knn_utils.py --normalize` |
| `transform remove-zeros-knnutils` | Zero vector removal (BLAS snrm2) | `fvecs_remove_zeros.py` |
| `analyze fvecs-check-knnutils` | fvecs validation | `fvecs_check.py` |
| `analyze check-normalization-knnutils` | Normalization check (tol 1e-3) | `check_normalization()` |
| `verify dataset-knnutils` | Unified verification + KNN accuracy | `fvecs_check.py` + `ivecs_check.py` + `validate_knn_utils.py` |

Behind `#[cfg(feature = "faiss")]` only:

| Command | Purpose |
|---------|---------|
| `compute knn-faiss` | FAISS-based KNN for A/B testing vs knn-blas |

### `--personality knn_utils` Bootstrap

```bash
veks prepare bootstrap --personality knn_utils --base-vectors source.fvec --metric Cosine
```

Maps native commands to knn\_utils equivalents:
- `compute sort` → `compute sort-knnutils`
- `generate shuffle` → `generate shuffle-knnutils`
- `compute knn` → `compute knn-blas`
- `verify knn-consolidated` → `verify dataset-knnutils`

---

## 4. Key Implementation Details

### KNN Computation (`compute knn-blas`)

Direct BLAS `cblas_sgemm` for the distance matrix, with Rust top-k
heap selection. No FAISS library needed.

- **IP metric**: `scores = query @ base.T` (single sgemm call)
- **L2 metric**: `||q - b||^2 = ||q||^2 + ||b||^2 - 2*q.b` (sgemm + precomputed norms)
- **Batching**: queries processed in batches to limit score matrix memory (~2 GB max)
- **Parallelism**: OpenBLAS multi-threaded sgemm + rayon parallel top-k selection
- **Performance**: 10.2s for 985k×10k×128 (3.3× faster than FAISS)

### Normalization (`transform normalize-knnutils`)

Calls numpy via subprocess — the only way to guarantee byte-identical
results. numpy's internal pairwise reduction (compiled C with f64
accumulators and unspecified block structure) cannot be reliably
replicated in Rust.

Key findings:
- `np.linalg.norm(arr, axis=1)` uses f64 pairwise accumulation, NOT `cblas_snrm2` or `cblas_sdot`
- Results differ by 1+ ULP from any BLAS norm routine for high-dimensional vectors
- `arr / norms` uses f32 division, NOT multiply-by-reciprocal
- Division by `1.0f32` in Rust can produce 1-ULP differences from numpy

### Shuffle (`generate shuffle-knnutils`)

Replicates numpy's `np.random.seed(N); np.random.shuffle(arr)`:
- **PRNG**: MT19937 via `rand_mt` crate
- **Bounded random**: `rk_interval` (bit-mask rejection sampling from numpy's `randomkit.c`)
- **PRNG chaining**: `prng-state-out`/`prng-state-in` for base→query shuffle continuity
- Verified bit-identical to numpy for 100k+ elements

### Sort/Dedup (`compute sort-knnutils`)

Full lexicographic comparison on all vector components, matching
Python's `chunk.sort(key=lambda x: tuple(x[0]))`.

The native `compute sort` uses prefix-based comparison (first N
components) with full comparison only within prefix collision groups.
Both produce the same unique set, but may order non-identical vectors
differently within prefix groups. The native sort was also fixed to
fully sort within prefix groups.

---

## 5. BLAS Dependency and Licensing

The knn\_utils personality commands call BLAS routines (`cblas_sgemm`,
`cblas_snrm2`, `cblas_sdot`) through whatever system BLAS is linked.

knn\_utils uses Intel MKL (`mkl=2023.1.0`). The `libmkl-dev` Debian
package is freely distributed by Intel via apt under the Intel
Simplified Software License (ISSL).

**With MKL**: `compute knn-blas` output matches Python knn\_utils on
the same machine (verified byte-identical for sift128e HDF5 dataset).

**With OpenBLAS**: 99.95% identical neighbor sets; ~5 queries per 10k
swap a single neighbor at the k=100 boundary (ULP-level BLAS rounding).

**Build-time**: cmake `FindBLAS` discovers the system BLAS via
`libblas.so` (configured via `update-alternatives`).

---

## 6. Verified Results

| Dataset | Source | Base | Query | GT | Engine |
|---------|--------|------|-------|----|--------|
| **sift128e** | HDF5 | BYTE-IDENTICAL | BYTE-IDENTICAL | BYTE-IDENTICAL | knn-blas (OpenBLAS) |
| **ada0021m** | non-HDF5 (zeros, unnormalized) | Self-consistent PASS | Self-consistent PASS | Self-consistent PASS | knn-blas |

### A/B Testing (knn-blas vs knn-faiss)

sift128e (985k×10k×128, k=10): 9998/10000 exact match, 2 set match, 0 real differences.

---

## 7. Native Pipeline Fixes (from this analysis)

1. **fvec extract ordering**: Transpose mode preserves ivec ordering for shuffled indices
2. **source\_zero\_count**: Accumulates across query+base extracts including duplicate zeros
3. **compute sort prefix groups**: Vectors within prefix collision groups now fully sorted
4. **HDF5 dependency removed**: Format detection retained, I/O requires pre-conversion
