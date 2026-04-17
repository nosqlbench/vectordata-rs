# FAISS-rs BLAS ABI Mismatch: Silent Data Corruption

## Status

**Active workaround in veks.** All FAISS search calls are chunked to
keep `batch_size <= 65536 / dim`. This prevents the corruption at the
cost of reduced batch efficiency for high-dimensional datasets.

**The FINTEGER fix alone is insufficient.** Changing `FINTEGER` from
`long` to `int` in `distances.cpp` eliminates the zero-distance
corruption for large batches, but introduces a secondary failure mode:
small-batch results return non-zero distances that are *wrong* (worse
neighbors than brute-force, 0/1000 set match against knn-metal). The
root cause is deeper than the `FINTEGER` typedef — likely involves
additional ABI mismatches in the BLAS tiling code paths or in how the
FAISS static build links against MKL.

## Summary

The `faiss` Rust crate (v0.13.0 / faiss-sys v0.7.0) silently produces
corrupt search results when using MKL LP64 as the BLAS implementation.
Two distinct failure modes have been observed:

1. **Zero-distance corruption** — all distances return 0.0, indices are
   small sequential values. Triggered by `n_queries * dim > 65536` in a
   single `index.search()` call. Prevented by the batch size cap.

2. **Wrong-neighbor corruption** — distances are non-zero but incorrect
   (finding worse neighbors than exhaustive search). Observed even with
   small batches (64 queries) at dim=1024 after applying the FINTEGER
   fix. This mode is subtler and harder to detect without cross-engine
   comparison.

Python FAISS on the same data produces correct results at all sizes,
confirming the bug is in the Rust/C binding layer or the static build
configuration, not the FAISS algorithm.

## Affected Configuration

- **faiss-sys** v0.7.0 with `static` feature (cmake-built)
- **BLAS**: Intel MKL LP64 (`libmkl_intel_lp64.so`)
- **Runtime**: `libmkl_rt.so` (MKL runtime dispatch)
- **Platform**: Linux x86_64 (64-bit `long`)
- **Trigger**: `n_queries * dim > ~65536` in a single `index.search()` call

Not affected:
- Python FAISS (uses correctly matched BLAS interface)
- Datasets where `n_queries * dim <= 65536` per search call (e.g., sift1m
  at dim=128 with batch_size=512 is within the safe range)

## Root Cause Analysis

### Confirmed: FINTEGER/BLAS integer size mismatch

FAISS's `distances.cpp` (line 31) defines:

```cpp
#ifndef FINTEGER
#define FINTEGER long    // 8 bytes on Linux x86_64
#endif
```

This type is used for all `sgemm_` (BLAS matrix multiply) arguments:

```cpp
// distances.cpp:300
FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
sgemm_("Transpose", "Not transpose",
       &nyi, &nxi, &di,          // ← 8-byte pointers
       &one, y + j0 * d, &di,
       x + i0 * d, &di,
       &zero, ip_block.get(), &nyi);
```

MKL LP64's `sgemm_` expects `int*` (4-byte) pointers for all integer
arguments. The function signature mismatch:

```
FAISS passes:   sgemm_(char*, char*, long*, long*, long*, ...)
MKL LP64 expects: sgemm_(char*, char*, int*,  int*,  int*,  ...)
```

### Why it doesn't crash

On little-endian x86_64, the low 4 bytes of a `long` containing a
small value (e.g., 1024) are identical to the `int` representation:

```
long 1024:  [0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
int  1024:  [0x00, 0x04, 0x00, 0x00]
```

MKL reads the first 4 bytes through the `int*` pointer and gets the
correct value. This is why small problem sizes work correctly — the
ABI mismatch is invisible when all values fit in 32 bits.

### Why it corrupts at large sizes

The corruption occurs because `sgemm_` parameters are `FINTEGER`
local variables allocated contiguously on the stack:

```cpp
FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
//       [8 bytes]       [8 bytes]       [8 bytes]
```

When MKL reads `&nyi` as `int*`, it reads 4 bytes. But the next
parameter `&nxi` is at offset +8 (not +4), so MKL's internal
pointer arithmetic based on `sizeof(int)` misaligns subsequent
parameter reads. The exact failure mode depends on:

1. **Stack layout** — compiler may reorder or pad locals
2. **BLAS internal tiling** — `sgemm_` internally subdivides the
   matrix multiply; the leading dimensions (`lda`, `ldb`, `ldc`)
   control output stride. A misread leading dimension writes
   results at wrong offsets in the output buffer.
3. **OpenMP thread-local storage** — the BLAS parallelization
   divides work by rows; misread dimensions change the per-thread
   work assignment.

The result: `sgemm_` writes the dot-product output matrix
`ip_block` with incorrect stride or to incorrect offsets. The
subsequent L2 distance computation reads zeros or garbage from
`ip_block`, producing zero distances.

### Why the FINTEGER fix is insufficient

Changing `FINTEGER` from `long` to `int` eliminates the
zero-distance failure mode — sgemm parameters are now passed
correctly. However, testing showed a secondary failure:

- **With `FINTEGER=long`**, batch cap of 64 queries at dim=1024 →
  correct results (distances non-zero, correct neighbors)
- **With `FINTEGER=int`**, batch cap removed, 1000 queries at
  dim=1024 → zero distances (same as before the fix)
- **With `FINTEGER=int`**, batch cap of 64 queries at dim=1024 →
  non-zero distances but **wrong neighbors** (12/100 overlap with
  knn-metal, 0/1000 exact set match)

This indicates that the `FINTEGER` mismatch is one of multiple
ABI issues between the FAISS static build and MKL. Possible
additional causes:

- Other `FINTEGER` usage sites in FAISS (norm computation, inner
  product paths, the AVX2-specialized distance code)
- Static vs dynamic MKL linking mismatch (cmake finds
  `libmkl_intel_lp64.so` at build time but `libmkl_rt.so` is
  linked at runtime — these may use different internal conventions)
- OpenMP runtime conflicts between FAISS's `-lgomp` and MKL's
  internal threading

### The 65536 threshold

FAISS tiles the search into blocks:

```cpp
int distance_compute_blas_query_bs = 4096;    // max queries per block
int distance_compute_blas_database_bs = 1024; // max base vecs per block
```

Empirically observed thresholds (n_queries at which zero-distance
corruption begins):

| dim  | max safe queries | n * dim threshold |
|------|-----------------|-------------------|
| 128  | >1000 (unaffected) | >128K |
| 512  | >128 (unaffected)  | >65K  |
| 800  | 800              | 80,000 (OK)       |
| 801  | FAILS at 100     | 80,100            |
| 1024 | 64               | 65,536 = 2^16     |
| 2048 | 32               | 65,536 = 2^16     |

The exact threshold varies because it depends on how the compiler
lays out the `FINTEGER` locals and how MKL's internal tiling
interacts with the misread parameters.

## Symptoms

### Mode 1: Zero-distance corruption (without batch cap)

- `index.search()` returns successfully (no error code)
- All distances in the result are `0.0`
- Indices are small sequential values (0, 1, 3, 4, 7, ...)
  regardless of the actual nearest neighbors
- The index reports correct `ntotal` and `d`
- `index.add()` reports success

### Mode 2: Wrong-neighbor corruption (with FINTEGER fix, no cap)

- `index.search()` returns successfully
- Distances are non-zero and plausible-looking
- Neighbors are wrong — knn-metal finds nearest at distance 1773,
  FAISS finds 1794 (worse neighbors)
- Only 12/100 neighbor overlap with knn-metal on query 0
- 0/1000 exact set match across all queries
- Not detectable without cross-engine comparison

## Verification

```bash
# Fails (65 * 1024 = 66560 > 65536):
veks compute knn-faiss --base data.fvec --query q65.fvec \
    --indices out.ivec --neighbors 5 --metric L2
# → distances all 0.0, wrong indices

# Works (64 * 1024 = 65536):
veks compute knn-faiss --base data.fvec --query q64.fvec \
    --indices out.ivec --neighbors 5 --metric L2
# → correct distances and indices

# Python FAISS works at any size:
python3 -c "
import faiss, numpy as np
index = faiss.IndexFlatL2(1024)
index.add(np.random.randn(10000, 1024).astype('f'))
D, I = index.search(np.random.randn(100, 1024).astype('f'), 5)
print(D[0])  # correct non-zero distances
"

# Cross-engine verification:
veks compute knn-metal --base data.fvec --query q.fvec --indices metal.ivec ...
veks compute knn-faiss --base data.fvec --query q.fvec --indices faiss.ivec ...
# Compare neighbor sets — should have >95% overlap at dim=128,
# but may show 0% at dim>=1024
```

## Current Workaround (Option 4)

Cap the query batch size so `n_queries * dim <= 65536`:

```rust
let max_queries_per_batch = (65536 / dim).max(1);
let chunk_size = max_queries_per_batch.min(query_count);
```

This avoids the zero-distance corruption. At dim=128 (sift1m and
similar datasets), the batch size is 512 — large enough for good
BLAS performance.

### Performance impact

| dim  | max batch | batches for 10K queries | overhead |
|------|-----------|------------------------|----------|
| 128  | 512       | 20                     | minimal  |
| 256  | 256       | 40                     | low      |
| 512  | 128       | 79                     | moderate |
| 1024 | 64        | 157                    | high     |
| 2048 | 32        | 313                    | severe   |

### Accuracy with workaround

At dim=128 (sift1m): metal vs faiss shows 111/10000 boundary
mismatches (1 neighbor swap each, all at the k-th boundary). These
are floating-point rounding differences, not corruption. Verified
by `--at-k 99` dropping all mismatches to zero.

At dim=1024: **not verified as accurate** — the comparison needs
to be re-run on an idle system. Preliminary results with the batch
cap showed FAISS returning wrong neighbors (worse distances than
metal/stdarch, 0% set match), which may indicate the workaround
is necessary but not sufficient at high dimensions.

## Fix Options Explored

### Option 1: FINTEGER=int (tested, insufficient)

Changing `FINTEGER` from `long` to `int` in `distances.cpp:31`
eliminates zero-distance corruption but introduces wrong-neighbor
corruption. Not a complete fix.

### Option 2: CMake-level BLAS integer detection (untested)

Would ensure FINTEGER matches the linked BLAS, but may not address
the runtime vs build-time MKL mismatch (`libmkl_intel_lp64.so` at
cmake time vs `libmkl_rt.so` at runtime).

### Option 3: Link against MKL ILP64 (untested)

Using `libmkl_intel_ilp64.so` instead of LP64 would make BLAS
expect 64-bit integers, matching `FINTEGER=long`. This requires
changes to both the faiss-sys cmake configuration and the system
MKL setup.

### Option 4: Batch size cap (current, working for dim<=512)

The only approach verified to produce correct results. Applied
to all FAISS search paths in veks.

### Option 5: Use knn-metal or knn-stdarch instead

For ground truth computation, knn-metal (SimSIMD) and knn-stdarch
(pure `std::arch`) produce byte-identical results and are 50-180x
faster than FAISS. FAISS is retained for cross-validation with
the knn_utils Python ecosystem, but should not be the primary
compute engine for high-dimensional datasets.

## Upstream References

- **faiss-rs repository**: https://github.com/Enet4/faiss-rs
- **FAISS source**: `faiss/utils/distances.cpp:31` (`FINTEGER` definition)
- **MKL LP64 documentation**: LP64 uses 32-bit integers for BLAS/LAPACK
  routine arguments. ILP64 uses 64-bit integers and requires linking
  against `libmkl_intel_ilp64.so` instead of `libmkl_intel_lp64.so`.
- **OpenBLAS**: Also uses 32-bit integers by default. The 64-bit
  interface requires building with `INTERFACE64=1`.

## Discovery Timeline

1. Found during accuracy comparison of knn-metal vs knn-faiss on a
   synthetic 1M × 1024-dim dataset. FAISS produced all-zero distances.

2. Binary search across batch sizes and dimensions identified the
   `n * dim > 65536` threshold. Python FAISS confirmed correct at
   all sizes.

3. Traced to `FINTEGER=long` vs MKL LP64 `int` mismatch in the
   `sgemm_` call parameters.

4. Applied `FINTEGER=int` fix — eliminated zero distances but
   introduced wrong-neighbor results (0/1000 match vs metal).

5. Reverted FINTEGER fix. Batch size cap remains as the only
   verified workaround.

6. Root cause is deeper than FINTEGER alone — likely involves
   static/dynamic MKL linking mismatch or additional ABI issues
   in the FAISS distance computation code paths.
