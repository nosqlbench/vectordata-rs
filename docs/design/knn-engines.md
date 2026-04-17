# KNN Engine Comparison: knn-metal vs knn-faiss vs knn-stdarch

## Overview

veks provides three brute-force exact KNN engines for ground truth computation:

- **knn-metal** (`compute knn-metal`) — Custom Rust implementation using SimSIMD
  for hardware-dispatched SIMD (AVX-512, AVX2, NEON).
- **knn-stdarch** (`compute knn-stdarch`) — Pure `std::arch` implementation with
  zero external SIMD dependencies. Uses hand-rolled AVX-512/AVX2 kernels.
- **knn-faiss** (`compute knn-faiss`, also the default `compute knn`) — Facebook's
  FAISS library via the `faiss` Rust crate, using FlatIndex for exhaustive search.

knn-metal and knn-stdarch produce byte-identical results. FAISS produces
equivalent results (within floating-point tolerance) with the same
tie-breaking strategy: lower ordinal wins at equal distance.

## FAISS Batch Size Limitation

**Critical:** The `faiss` Rust crate (v0.13, static build) silently produces
corrupt results (zero distances, wrong indices) when a single `index.search()`
call has `n_queries * dim > 65536`. The FAISS C++ library itself does not have
this limitation — Python FAISS works correctly at any size. The bug is in the
Rust `faiss-sys` FFI bindings.

**Symptoms:** All distances return 0.0, indices are small sequential values
(0, 1, 3, 4...) regardless of the actual nearest neighbors.

**Workaround:** All FAISS search calls in veks are chunked to keep
`batch_size * dim <= 65536`. This is enforced automatically.

**Affected configurations:**

| Dimension | Max queries per batch |
|-----------|---------------------|
| 128 | 512 |
| 256 | 256 |
| 512 | 128 |
| 768 | 85 |
| 1024 | 64 |
| 2048 | 32 |

## Performance Characteristics

On a 128-core machine with sift1m (1M base x 10K queries x dim=128 x k=100):

| Aspect | knn-metal | knn-faiss |
|--------|-----------|-----------|
| Throughput | ~5B dist/s | ~0.2B dist/s |
| Threading | 128 threads x query batches | OpenMP within BLAS |
| SIMD kernel | 16-wide transposed AVX-512 FMA | BLAS sgemm |
| Top-k selection | Fused in inner loop | Separate pass over distance matrix |
| Memory model | mmap streaming, no copy | Bulk copy into FAISS index |
| Distance matrix | Never materialized | Full N x M in RAM |

knn-metal is approximately 20-25x faster than knn-faiss for exhaustive
brute-force KNN on this hardware.

## Why knn-metal Is Faster

### 1. Threading model

knn-metal partitions queries across all available cores using `std::thread::scope`.
Each thread independently scans base vectors and maintains its own top-k heaps.
For 10K queries on 128 threads, each thread handles ~78 queries simultaneously,
and all threads scan base vectors in parallel.

FAISS uses OpenMP internally, but its FlatIndex brute-force search parallelizes
*within* a single query batch — it divides the base vectors across threads for
distance computation. The parallelism is at the base-vector level, not the query
level. This means FAISS can't overlap the heap maintenance across queries the
way knn-metal does.

### 2. Transposed batch SIMD kernel

knn-metal's key optimization is the **transposed batch kernel**. It groups 16
queries into a `TransposedBatch` with dimension-major layout
(`data[d * 16 + qi]`), then for each base vector:

1. Loads one base vector dimension value
2. Broadcasts it to all 16 SIMD lanes
3. Computes 16 distances with a single AVX-512 FMA instruction

This means one sequential pass over each base vector computes distances to 16
queries simultaneously. For 10K queries, that's 625 passes over the base
vectors instead of 10,000. The data-to-compute ratio is 16x better than
pairwise computation.

FAISS FlatIndex performs a matrix multiply (queries x base^T) via BLAS/MKL.
While BLAS is highly optimized for GEMM, it produces a full N x M distance
matrix that then needs a separate scan for top-k extraction. This uses more
memory bandwidth and doesn't fuse the top-k selection with distance
computation.

### 3. Memory access pattern

knn-metal reads base vectors directly from mmap'd files. The kernel iterates
base vectors sequentially, which triggers hardware prefetching and requires no
data copying. The OS page cache provides the buffering.

FAISS requires all base vectors copied into a contiguous `Vec<f32>` buffer
before building the FlatIndex. For sift1m, that's ~490 MiB of allocation and
copying just for setup, before any search begins. FAISS internally may copy
again into its own layout.

### 4. Top-k fusion

knn-metal fuses distance computation with top-k heap maintenance in the inner
loop. As each distance is computed, it's immediately compared against the
current k-th threshold. If below threshold, it's inserted into the binary
heap. Most distances (all but the nearest k) are discarded with a single
comparison — no memory write.

FAISS computes ALL distances first (matrix multiply), then performs a separate
top-k extraction pass. The full distance matrix for 10K queries x 1M base =
40 GB of f32 values must be written to memory and then read back. This is
pure memory bandwidth waste for the brute-force case.

### 5. Cache-friendly partitioning

knn-metal auto-partitions large base vector sets to fit in L3 cache. Each
partition's base vectors stay resident in cache while all query batches are
processed against them. The partition size is automatically tuned to ~50% of
available RAM.

FAISS FlatIndex has no such partitioning — it relies on BLAS to manage cache
locality, which works well for the matrix multiply kernel but poorly for the
subsequent top-k extraction scan over the full distance matrix.

## Why knn-faiss Is the Default

Despite being slower for brute-force search, FAISS is the default `compute knn`
engine because:

1. **Cross-validation** — FAISS is the reference implementation used by the
   knn_utils Python pipeline and most published benchmarks. Using the same
   engine for ground truth computation ensures compatibility.

2. **Correctness confidence** — FAISS is widely deployed, extensively tested,
   and its brute-force FlatIndex is simple and well-understood. Using it as
   the default reduces the risk of subtle bugs in ground truth.

3. **Deterministic ordering** — knn-metal and knn-faiss use the same
   tie-breaking strategy (lower ordinal wins), but produce slightly different
   distance values due to different SIMD instruction ordering. Using FAISS
   consistently avoids boundary mismatches when cross-checking.

For performance-critical ground truth computation (large datasets, many
profiles), switch to knn-metal:

```yaml
# In dataset.yaml pipeline steps:
- id: compute-knn
  run: compute knn-metal    # instead of: compute knn
```

## Tie-Breaking and Boundary Mismatches

Both engines use the same deterministic tie-breaking strategy:

- **Heap eviction**: max-heap ordered by (distance desc, index desc). At equal
  distance, the higher index has higher priority in the max-heap and is evicted
  first. Lower index is retained.
- **Output sort**: results sorted by (distance asc, index asc). At equal
  distance, lower index appears first.

This means both engines produce the same output given the same distance values.

However, **the distance values themselves differ at the ULP level** due to
different SIMD implementations (SimSIMD vs FAISS/BLAS). The same vector pair
can produce distances that differ by 1 ULP (unit in the last place) due to
different FMA instruction ordering. At the k-th boundary where many base
vectors have nearly identical distances to a query, a 1-ULP difference can
change which neighbor is the k-th nearest.

This is observable as "boundary mismatches" when cross-checking:

```
$ veks verify knn-faiss
  mismatches (111 total):
    query 197: 1 neighbors differ (boundary)
    query 262: 1 neighbors differ (boundary)
    ...
```

All 111 mismatches involve exactly 1 neighbor swap at the k-th boundary.
Verifying at k-1 confirms they vanish:

```
$ veks verify knn-faiss --at-k 99
  PASS: 10000/10000 queries — 10000 exact, 0 set, 0 boundary, 0 fail
```

These are floating-point rounding artifacts, not correctness bugs. Both result
sets are valid ground truth — they differ only in which of two equidistant
vectors occupies the last position.

## Verification Architecture

The verification commands mirror the engine split:

| Command | Engine | Usage |
|---------|--------|-------|
| `verify knn-groundtruth` | FAISS (default) | Per-profile pipeline step |
| `verify knn-groundtruth-metal` | SimSIMD | Explicit SimSIMD verification |
| `verify knn-consolidated` | FAISS (default) | Multi-profile single-pass |
| `verify knn-consolidated-metal` | SimSIMD | Explicit SimSIMD consolidated |
| `verify knn-faiss` | FAISS | Post-hoc standalone verification |

The consolidated FAISS verifier loads base vectors once and shares them across
all sized profiles. Partition profiles are verified in parallel using
`std::thread::scope`, each with its own FAISS index on a separate thread.
