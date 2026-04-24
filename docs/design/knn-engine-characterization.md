# KNN Engine Characterization (post-fix baseline)

This is the current state of cross-kernel agreement and per-engine
performance after all the fixes in this branch land. Supersedes the
earlier engine-parity discussion that conflated three different notions
of agreement and didn't account for the faiss-sys MKL bug's process-wide
reach.

## Fixture used for all numbers below

| | |
|---|---|
| base vectors | 100,000 |
| query vectors | 10,000 |
| dim | 384 |
| metric | DOT_PRODUCT |
| k | 100 |
| host | 128-thread x86_64 with AVX-512 |
| BLAS link | faiss-sys static MKL (LP64) |

This shape is in the same ballpark as the user's production pipeline
(384-dim embeddings, 10K queries) so the numbers should be
representative of what you actually see in the field.

## Per-engine performance baseline

| Engine | Wall clock | Hot-path rate | Speed vs stdarch | Notes |
|---|---|---|---|---|
| **stdarch** | 0.66 s | 1.71 B dist/s | 1.0× (baseline) | pure `std::arch` AVX-512 SIMD, no BLAS |
| **metal** | 0.45 s | 2.4 B dist/s | 1.5× faster | SimSIMD AVX-512 (slightly tighter kernel) |
| **blas** | 0.54 s | 2.20 B dist/s | 1.2× faster | sgemm via MKL, **forced single-threaded** for the ABI bug |
| **faiss** | 37.6 s | 27 M dist/s | **57× slower** | sgemm via MKL, **single-threaded + batch-capped** |

The "hot-path rate" column reports the segment-compute throughput from
the engine's own log line, which excludes setup and the canonical
rerank post-pass (~50 ms with current code).

## Why FAISS is so much slower than blas, even though they share MKL

Two compounding penalties from the bug workaround stack:

1. **Single-threaded MKL** (`OPENBLAS_NUM_THREADS=1` etc., set at the
   top of every BLAS-touching `execute()` — see `pipeline::blas_abi`).
   On a 128-thread box this alone is a ~10× slowdown vs. multi-threaded
   sgemm. We accept it because multi-threaded MKL silently corrupts
   sgemm output on this faiss-sys static MKL build (see
   `docs/design/faiss-blas-abi-bug.md`).

2. **FAISS batch-size cap**: faiss-sys returns wrong neighbors when
   `n_queries × dim > 65536` in a single `index.search()` call.
   Workaround: split queries into batches of `65536 / dim` queries
   each. At dim=384 that's 170 queries per batch → 59 sequential FAISS
   calls for the 10K-query workload. Per-batch overhead (re-entering
   the FAISS C++ layer, re-allocating result buffers) dominates.

`compute knn-blas` only pays penalty (1) — it manages its own batching
through our shared `scan_range_sgemm`. FAISS pays both. That's why
FAISS is now ~70× slower than blas despite calling the same underlying
sgemm symbol.

This is *exactly the right cost shape* for the new policy:
- **Compute**: use stdarch (or metal). No BLAS dependency, no MKL
  poisoning, full-throughput SIMD. Fast.
- **Verify**: use FAISS as an *independent* implementation to
  cross-check, accepting that it'll be slow. Verification only runs
  on a sample of queries (default 100 to a few thousand), so 50× slower
  on the verify path is workload-appropriate.

## Cross-kernel correctness

Two distinct correctness questions, both passing now.

### Recall against f64 brute-force truth

Each engine's top-k vs the f64-recomputed top-k:

| Engine | recall@100 mean | recall@100 min | f64-truth-set agreement |
|---|---|---|---|
| stdarch | 1.0000 | 0.99 | identical (modulo 1 tied-neighbor query) |
| metal | 1.0000 | 0.99 | identical |
| blas | 1.0000 | 0.99 | identical |
| faiss | 1.0000 | 0.99 | identical |

The 0.99 minimum is one specific query whose 100th neighbor has a
distance tied with the 101st; any deterministic tiebreaker picks one
of them. Not a real disagreement.

### Order agreement (FAISS-congruent canonical ranking)

After the canonical f64 rerank post-pass, each engine's output is
**byte-identical** to every other engine's output for this fixture:

| Engine pair | top-100 order match |
|---|---|
| stdarch ↔ metal | 100% |
| stdarch ↔ blas | 100% |
| stdarch ↔ faiss | 100% |

This is the property that matters for downstream consumers:
- **recall@k for varying k** is well-defined regardless of which
  engine produced the ground truth (the order at the boundary is
  canonical).
- **k-position-sensitive measures** (NDCG, MRR, etc.) compute the
  same value across engines.
- **Bit-comparison against published `knn_utils` / FAISS reference
  datasets** matches.

Without the canonical rerank, the engines would disagree on the
*order* within their top-k even when their *sets* match — that's the
property the rerank exists to enforce.

## What the canonical rerank actually costs now

| Workload | Old rerank cost | New rerank cost |
|---|---|---|
| 10k queries × k=100 × dim=384 | ~2.0 s | ~50 ms |

Two anti-patterns, fixed:

1. **`F: FnMut(u32) -> Option<Vec<f32>>`** with caller `.to_vec()` on
   an mmap slice → 1M allocations × 1.5KB each per pipeline step.
   Now: closure returns `Option<&'a [f32]>`, zero copies.

2. **Unbuffered `f.write_all(&[u8; 4])`** in a tight loop → 1M tiny
   syscalls per output file. Now: `BufWriter::with_capacity(1 << 20)`,
   coalesced into a handful of large writes.

Net: rerank is ~5% overhead on the compute step, comfortably justified
by the order-stability property it provides.

## Default policy now wired into bootstrap

| Phase | Default command | Why |
|---|---|---|
| Compute KNN | `compute knn` (= stdarch) | Fast, BLAS-independent, 100% recall vs truth, identical to FAISS after canonical rerank |
| Verify KNN | `verify knn-faiss` (when faiss feature on) | Independent implementation, cross-validates against the same library `knn_utils` Python uses |
| Verify KNN (no faiss) | `verify knn-consolidated` | Falls back to our own sgemm scan with the same MKL safety net |

`compute knn-blas` and `compute knn-faiss` remain registered for users
who specifically want them (regression testing against numpy /
knn_utils, A/B comparisons), but neither is a default. Both have the
single-threaded BLAS safety net applied internally so they can't
silently produce wrong results.

## What changed from earlier characterizations

The earlier "100% EXACT across the board" claim was about
`verify engine-parity`, which:
- Forces single-threaded BLAS (so the MKL bug doesn't fire)
- Sets `rerank_margin_ratio=3` (so engines can recover boundary
  candidates the f32 kernel mis-ranks)
- Compares engines against each other (with metal as reference), not
  against f64 truth

That property still holds for `verify engine-parity`. What's *new*
in this baseline:

- **Production default agrees with FAISS too**, in *both* SET and
  ORDER — without needing the margin opt-in. The canonical rerank
  ensures order, and stdarch's f32 direct-kernel path doesn't drift
  from f64 truth at the dimensions / distributions real workloads use.
  Margin only matters for the adversarial uniform-random high-dim
  fixture in the parity test.

- **The earlier 2-second rerank cost is gone**. Production speed is
  back to the e1aa117 baseline (~1s wall-clock for the test fixture)
  with the rerank still running.

- **`compute knn-blas` is no longer the bootstrap default**. It still
  works correctly (with the single-threaded BLAS safety net), but
  given that single-threaded sgemm throws away most of BLAS's
  throughput advantage, stdarch is both faster and produces
  bit-identical canonical output. blas remains useful for explicit
  numpy / knn_utils byte-for-byte comparison.
