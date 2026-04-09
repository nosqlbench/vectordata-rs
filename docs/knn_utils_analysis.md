# knn_utils Analysis & Rust Ecosystem Mapping

## Overview

This document analyzes the Python `knn_utils` project (linked at `links/knn_utils`)
to catalog its dependencies, trace their native/C/C++ provenance, and map them to
equivalent Rust crates — particularly those that wrap the **same upstream libraries**.
It also documents algorithmic conventions for ensuring veks maintains equal or higher
rigor in its pipeline steps.

---

## 1. Library Inventory & Native Provenance

### Python Libraries → Upstream C/C++

| Python Library   | Version  | Upstream C/C++ Library                  | Role in knn\_utils                                         |
|------------------|----------|-----------------------------------------|------------------------------------------------------------|
| **faiss-gpu**    | 1.9.0    | Facebook FAISS (C++) + CUDA 12.1.1      | Exact brute-force KNN: `IndexFlatL2`, `IndexFlatIP`; multi-GPU sharding |
| **numpy**        | 1.26.4   | Intel MKL 2023.1.0 (BLAS/LAPACK)        | L2 norms, array reshape, masking, histograms, binary I/O   |
| **h5py**         | (pinned) | HDF5 C library (libhdf5)                | Lazy-loading vector datasets from HDF5 files                |
| **pyarrow**      | (pinned) | Apache Arrow/Parquet (C++)              | Reading Parquet-format files                                |

**Environment:** Python 3.12.9, CUDA 12.4 toolkit, Intel MKL 2023.1.0, conda-pinned
via `environment.yml`.

### Rust Crate Equivalents (Same Upstream)

| Upstream C/C++        | Python Wrapper   | Rust Crate                        | Same upstream? | veks status       |
|-----------------------|------------------|-----------------------------------|----------------|-------------------|
| Facebook FAISS (C++)  | `faiss`          | `faiss` (via `faiss-sys`)         | Yes            | **Not present**   |
| Intel MKL (BLAS)      | `numpy`          | `intel-mkl-src` + `ndarray-linalg`| Yes            | **Not present**   |
| HDF5 C library        | `h5py`           | `hdf5` (via `hdf5-sys`)          | Yes            | Already in veks-core |
| Apache Arrow/Parquet  | `pyarrow`        | `arrow` + `parquet` (arrow-rs)   | Format-compat* | Already in veks   |

\* `arrow-rs` / `parquet` are pure-Rust implementations of the same file format,
not C++ bindings. They read/write identical files but do not link the C++ Arrow library.

---

## 2. Algorithmic Conventions in knn\_utils

### KNN Computation
- **Brute-force exact search only** — no approximate nearest neighbors
- **L2 (Euclidean):** `faiss.IndexFlatL2` — computes squared L2 distances
- **Inner Product:** `faiss.IndexFlatIP` — for pre-normalized vectors (cosine proxy)
- GPU-accelerated with multi-GPU sharding via `GpuMultipleClonerOptions`

### Normalization
- L2 norm per vector: `np.linalg.norm(vecs, axis=1, keepdims=True)`
- Zero-norm protection: norms of 0.0 replaced with 1.0 before division
- Tolerance check: vector is "normalized" if `|norm - 1.0| < 1e-3`

### Vector Formats
- **fvecs:** `[dim:i32, f32×dim]` per record, little-endian, contiguous, no padding
- **ivecs:** same layout with `i32` payloads (ground truth neighbor indices)
- **HDF5:** standard datasets, lazy-loaded

### Deduplication
- External merge sort (3-stage threaded pipeline: reader → sorter → writer)
- Heap-based k-way merge of sorted chunks
- Streaming consecutive-duplicate detection with histogram reporting
- Pre-sorted fast path: single-pass comparison of consecutive vectors

### Sorting
- Lexicographic on float tuples; raw-bytes fast path with numeric fallback
- External sort: configurable chunk size (default 200k vectors)
- Complexity: O(n log n) per chunk, O(k log k) for k-way merge

### Shuffling
- Fisher-Yates on in-RAM index array, then random-access reads
- Seed 42 for reproducibility

### Numerical Precision
- float32 throughout; float64 only for norm histogram stability
- Integer indices: int32

### Processing Pipeline Order
1. Load base and query vectors
2. Count exact zero vectors (eps=0.0)
3. Optionally remove zeros
4. Check if already normalized
5. Optionally shuffle (seed=42)
6. Optionally truncate
7. Optionally normalize (in-place)
8. Write processed vectors if any preprocessing applied
9. Build FAISS index from base vectors
10. Perform k-NN search
11. Write output indices to .ivecs

### Test Coverage
- L2 and cosine metric correctness validated against brute-force reference
  (`np.argsort(np.linalg.norm(...))`)
- fvecs format validation (dimension consistency, truncation detection)
- ivecs ground truth validation (duplicates, negatives, row-length mismatches)
- **Gaps:** no tests for shuffle, dedup, normalization, sorting, endian swap, HDF5 conversion

---

## 3. Q&A: Integration Details

### What does "MKL-backed" mean?

**MKL** is Intel's **Math Kernel Library** — a proprietary, highly optimized
implementation of the BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear
Algebra PACKage) standard APIs. When numpy is "MKL-backed," it means that operations
like matrix multiply, dot product, and vector norms are not running in pure Python or
even generic C — they dispatch to Intel's hand-tuned assembly routines that exploit
specific CPU microarchitectural features (AVX-512, cache hierarchy, prefetch patterns,
etc.).

In knn\_utils, `environment.yml` pins `mkl = 2023.1.0` and `intel-openmp`, meaning
every `np.linalg.norm()`, `np.dot()`, and `np.matmul()` call ultimately runs MKL
code. This matters because:

1. **Numerical behavior is deterministic and tied to MKL's implementation** — a
   different BLAS (OpenBLAS, Apple Accelerate) may produce subtly different
   floating-point results due to different summation orders or FMA usage.
2. **Performance baselines assume MKL** — benchmarks against knn\_utils assume
   MKL-grade throughput for norm computations.

For veks, this is mostly informational. Our KNN distance computation is
SIMD-accelerated via SimSIMD (hand-written AVX-512/AVX2 kernels), and our norm
computations in the pipeline are explicit Rust code — we do not route through a BLAS
library at all. This is actually a **higher level of control** since we know exactly
which instruction sequences execute and can guarantee deterministic results. Adding
MKL via `intel-mkl-src` would only matter if we needed to call general-purpose linear
algebra routines (matrix factorization, eigendecomposition, etc.), which we currently
do not.

**Bottom line:** We do not need MKL. Our SimSIMD-based distance kernels and explicit
Rust norm code are already more controlled than numpy+MKL for our use case.

### What is the relationship with ndarray-linalg — do we just add it to dependencies?

`ndarray-linalg` is a Rust crate that adds linear algebra operations (matrix inverse,
SVD, eigendecomposition, solve, norms) to the `ndarray` crate. It does this by
linking to a BLAS/LAPACK backend at compile time. The relationship is:

```
ndarray          — n-dimensional array container (we already use this)
    │
    └── ndarray-linalg  — linear algebra trait impls for ndarray
            │
            └── one of:
                ├── intel-mkl-src   — links Intel MKL
                ├── openblas-src    — links OpenBLAS
                └── netlib-src      — links reference LAPACK (slow)
```

Adding it is not just "add a dependency" — it requires choosing and configuring a
BLAS backend. This means:

1. **`intel-mkl-src`** — Requires MKL to be installed on the build machine (or
   downloads a large binary). Produces MKL-identical results to knn\_utils/numpy.
2. **`openblas-src`** — Easier to build, but different numerical behavior than MKL.
3. **`netlib-src`** — Reference implementation, slow but portable.

**For veks, we do not currently need `ndarray-linalg`.** Our linear algebra needs are:
- **L2 norms:** Implemented directly in SIMD distance kernels
- **Dot products:** Implemented directly in SIMD distance kernels
- **Normalization:** Simple division by L2 norm, done in explicit Rust

We would only need `ndarray-linalg` if we added operations like matrix factorization,
least-squares solving, or PCA — none of which are in scope. The `ndarray` crate alone
(which we already depend on) is sufficient for array manipulation.

---

## 4. Personality System Design

### Concept

A **personality** is a named set of alternative command implementations that the
pipeline uses instead of the defaults. When a user runs
`veks prepare bootstrap --personality knn_utils`, the generated `dataset.yaml`
references knn\_utils-compatible command names instead of the native ones. Both
personalities produce the same artifact types (fvec, ivec, etc.) and are
interchangeable at the pipeline level — only the underlying library and algorithm
differ.

### Personalities Defined

| Personality      | KNN Engine   | Distance Kernels | Sort/Dedup | Normalization |
|------------------|-------------|-------------------|------------|---------------|
| `native_simd`    | SimSIMD transposed-batch SIMD | SimSIMD (AVX-512/AVX2) | Rust `compute sort` | Explicit Rust SIMD |
| `knn_utils`      | FAISS (`IndexFlatL2`/`IndexFlatIP`) via `faiss` crate | FAISS internal | Rust `compute sort` (same)* | numpy-style via FAISS preprocessing* |

\* Not every step needs a knn\_utils variant. Only steps where the underlying library
differs need alternative implementations. Sort/dedup and shuffle are pure algorithmic
operations where our Rust implementation is already equivalent or superior to the
Python threaded pipeline in knn\_utils.

### Which Commands Get Personality Variants

The key question is: which pipeline steps use libraries that differ between
knn\_utils and our native implementation? Only those need alternative commands.

| Pipeline Step | native\_simd Command | knn\_utils Command | Why variant needed? |
|---------------|---------------------|--------------------|---------------------|
| KNN ground truth | `compute knn` (SimSIMD) | `compute knn-faiss` | Different distance engine (SimSIMD vs FAISS C++) |
| Filtered KNN | `compute filtered-knn` (SimSIMD) | `compute filtered-knn-faiss` | Same reason |
| Sort/dedup | `compute sort` | `compute sort` (same) | Pure Rust algorithmic — no library difference |
| Shuffle | `generate shuffle` | `generate shuffle` (same) | Fisher-Yates is Fisher-Yates |
| Extract | `transform extract` | `transform extract` (same) | Format I/O — already uses same libhdf5 |
| Convert | `transform convert` | `transform convert` (same) | Already uses same upstream libraries |
| Verify KNN | `verify knn-consolidated` | `verify knn-consolidated` (same) | Verification is library-agnostic |

**Result:** Only `compute knn` and `compute filtered-knn` need knn\_utils variants.
Everything else already uses the same upstream libraries or is a pure algorithm.

### Implementation Architecture

#### New Command Modules

```
veks-pipeline/src/pipeline/commands/
    compute_knn.rs              ← existing (SimSIMD)
    compute_knn_faiss.rs        ← NEW (FAISS via faiss crate)
    compute_filtered_knn.rs     ← existing (SimSIMD)
    compute_filtered_knn_faiss.rs ← NEW (FAISS via faiss crate)
```

#### Registration (commands/mod.rs)

```rust
// ── compute (native_simd personality) ───────────────────────────
registry.register("compute knn", compute_knn::factory);
registry.register("compute filtered-knn", compute_filtered_knn::factory);

// ── compute (knn_utils personality) ─────────────────────────────
#[cfg(feature = "faiss")]
registry.register("compute knn-faiss", compute_knn_faiss::factory);
#[cfg(feature = "faiss")]
registry.register("compute filtered-knn-faiss", compute_filtered_knn_faiss::factory);
```

#### Cargo Feature Flag (veks-pipeline/Cargo.toml)

```toml
[features]
default = []
faiss = ["dep:faiss"]

[dependencies]
faiss = { version = "0.13", optional = true }
```

GPU support as a separate additive feature:

```toml
[features]
default = []
faiss = ["dep:faiss"]
faiss-gpu = ["faiss", "faiss/gpu"]
```

#### Personality → Command Mapping in Bootstrap

In `veks/src/prepare/import.rs`, the `emit_steps()` function hardcodes command
names like `"compute knn"`. The personality system maps these at generation time:

```rust
/// Command name mapping for each personality.
fn knn_command(personality: &str) -> &'static str {
    match personality {
        "knn_utils" => "compute knn-faiss",
        _ => "compute knn",  // native_simd (default)
    }
}

fn filtered_knn_command(personality: &str) -> &'static str {
    match personality {
        "knn_utils" => "compute filtered-knn-faiss",
        _ => "compute filtered-knn",
    }
}
```

Then in `emit_steps()`:

```rust
// Before:
//   run: "compute knn".into(),
// After:
//   run: knn_command(&args.personality).into(),

steps.push(Step {
    id: "compute-knn".into(),
    run: knn_command(&args.personality).into(),
    description: Some("Compute brute-force exact KNN ground truth".into()),
    // ... rest unchanged
});
```

#### CLI Surface

```
veks prepare bootstrap \
    --name cohere-msmarco \
    --base-vectors source.fvec \
    --metric Cosine \
    --personality knn_utils     # ← NEW FLAG (default: native_simd)
```

The personality is also recorded in `dataset.yaml` metadata so that `veks run`
can report which personality was used:

```yaml
metadata:
  personality: knn_utils
```

#### The Generated dataset.yaml Difference

With `--personality native_simd` (default):
```yaml
- id: compute-knn
  run: compute knn
  base: profiles/base/base_vectors.fvec[0..${base_count})
  query: profiles/base/query_vectors.fvec
  indices: neighbor_indices.ivec
  distances: neighbor_distances.fvec
  neighbors: 100
  metric: Cosine
```

With `--personality knn_utils`:
```yaml
- id: compute-knn
  run: compute knn-faiss
  base: profiles/base/base_vectors.fvec[0..${base_count})
  query: profiles/base/query_vectors.fvec
  indices: neighbor_indices.ivec
  distances: neighbor_distances.fvec
  neighbors: 100
  metric: Cosine
```

The step ID stays the same (`compute-knn`), the artifacts stay the same, only the
`run:` target changes. This means:
- Downstream steps (`verify knn-consolidated`, etc.) work unchanged
- Artifact paths are identical — results are directly comparable
- You can bootstrap two datasets from the same source with different personalities
  and diff the output ivecs

### A/B Testing Workflow

```bash
# Generate two datasets from the same source, different personalities
veks prepare bootstrap --name cohere-native --personality native_simd \
    --base-vectors /data/cohere/base.fvec --metric Cosine

veks prepare bootstrap --name cohere-faiss --personality knn_utils \
    --base-vectors /data/cohere/base.fvec --metric Cosine

# Run both
veks run cohere-native/
veks run cohere-faiss/

# Compare ground truth outputs
veks analyze compare-files \
    cohere-native/profiles/default/neighbor_indices.ivec \
    cohere-faiss/profiles/default/neighbor_indices.ivec
```

### Rust FAISS Crate Details

The `faiss` crate (crates.io) provides safe Rust bindings via `faiss-sys` (C API):

```rust
use faiss::{IndexFlatL2, IndexFlatIP, Index};

// Build index — mirrors knn_utils build_index()
let mut index = match metric {
    Metric::L2 => IndexFlatL2::new(dimension as u32)?,
    Metric::Cosine | Metric::DotProduct => IndexFlatIP::new(dimension as u32)?,
};
index.add(&base_vectors)?;

// Search — mirrors knn_utils main() search call
let result = index.search(&query_vectors, k)?;
// result.labels: Vec<i64>  — neighbor indices (same as knn_utils output)
// result.distances: Vec<f32> — neighbor distances
```

GPU support (matching knn\_utils `faiss-gpu` with CUDA 12.1.1):

```rust
#[cfg(feature = "faiss-gpu")]
{
    let gpu_res = faiss::GpuResources::new()?;
    let gpu_index = faiss::index_cpu_to_gpu(&gpu_res, 0, &index)?;
    let result = gpu_index.search(&query_vectors, k)?;
}
```

### Build Considerations

- FAISS requires `libfaiss` to be installed (or built from source via `faiss-sys`)
- GPU support requires CUDA toolkit (12.x recommended to match knn\_utils)
- The `faiss-sys` build can be slow (compiles FAISS C++ from source)
- The `faiss` cargo feature keeps FAISS out of default builds — users who don't
  need cross-validation never pay the build cost
- `cargo build --features faiss` enables the knn\_utils personality
- `cargo build --features faiss-gpu` adds GPU support on top

---

## 5. Summary: What veks Already Has vs. Gaps

| Capability             | knn\_utils Approach           | veks Approach                          | Gap?  |
|------------------------|-------------------------------|----------------------------------------|-------|
| KNN computation        | FAISS brute-force (GPU)       | SimSIMD transposed-batch SIMD (CPU)    | Personality system bridges this |
| Distance metrics       | L2, Inner Product             | L2, Cosine, DotProduct, L1            | veks has **more** metrics |
| Normalization          | numpy L2 norm + MKL           | Explicit Rust SIMD                     | No gap — veks is more controlled |
| HDF5 I/O               | h5py (libhdf5)               | hdf5 crate (same libhdf5)             | Same upstream |
| Parquet I/O             | pyarrow (C++)                | arrow-rs + parquet (pure Rust)         | Format-compatible |
| External sort/dedup    | Python threaded pipeline      | Rust pipeline commands                 | Same algorithm, better implementation |
| Shuffling              | Fisher-Yates in RAM           | Rust `generate shuffle`                | Same algorithm |
| f16 support            | Not supported                 | Native f16 SIMD kernels               | veks is **ahead** |
| Memory partitioning    | Not supported                 | Partition + cache + merge              | veks is **ahead** |
| Tie-breaking           | FAISS default (unspecified)   | Deterministic (lower index wins)       | veks is **more rigorous** |

### Personality Coverage

| Pipeline Step         | Needs personality variant? | Reason |
|-----------------------|---------------------------|--------|
| `compute knn`         | **Yes** → `compute knn-faiss` | Different distance engine |
| `compute filtered-knn`| **Yes** → `compute filtered-knn-faiss` | Different distance engine |
| `compute sort`        | No | Pure algorithmic, no library difference |
| `generate shuffle`    | No | Same algorithm |
| `transform extract`   | No | Same I/O libraries (libhdf5) |
| `transform convert`   | No | Same I/O libraries |
| `verify *`            | No | Verification is library-agnostic |
| `state *`             | No | Variable management, no math |
| `generate *`          | No | No library-dependent math |
