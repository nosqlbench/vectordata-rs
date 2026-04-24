# KNN Parity Memo — what "parity" actually means, and why our story had holes

## Why this memo exists

Over a recent run of work we repeatedly conflated several *different* notions
of "engine parity" in the KNN compute/verify stack, and landed fixes that
closed one notion while leaving others untouched. The user caught it; this
memo is the self-audit. It covers what we tested, what we didn't, what we
inadvertently corrupted, and how we ended up here.

Not a post-mortem in the incident-review sense — nothing was shipped to a
paying customer. But there *were* regressions on the user's local pipeline
that required dataset regeneration, and two of them (the rerank sign-bug and
the blas-vs-truth gap) would have gone unreported from my side if the user
hadn't pushed for clarity at the right moments.

## Three things we mean when we say "engines agree"

The code exercises three distinct notions of parity. They're related but
give different answers. Earlier discussions slid between them without
naming which one was being tested at any given moment.

### 1. Engines-agree-with-each-other parity

*Do `knn-metal`, `knn-stdarch`, `knn-blas`, `knn-faiss`, and `blas-mirror`
produce the same top-k neighbor sets on the same input?*

Measured by: `veks pipeline verify engine-parity`.

Mechanics: each engine runs the same fixture, all output gets fed through
the canonical f64 rerank (`knn_segment::rerank_output_post_pass`), then
pairwise comparison against a reference engine (metal). `rerank_margin_ratio=3`
is set on every sub-engine so each pulls top-(k + 3·k) candidates, giving
the f64 rerank enough material to reconcile any f32 boundary wobble.

Result: **100% EXACT across every dim × distribution × metric we test.**
Documented in §12.7.

What this *doesn't* prove: that any of those engines agrees with an
independent f64-brute-force ground truth. If all engines were wrong *in
the same way* — e.g., all drifting from truth by virtue of shared
arithmetic — this test would still pass.

### 2. Engines-agree-with-f64-truth parity

*Do the engines return the k vectors that an f64 brute-force scan would
return as the k closest?*

Measured by: `cross_engine_knn_parity` (integration test in
`veks-pipeline/tests/command_edge_cases.rs`).

Mechanics: each engine runs in its default production configuration
(no `rerank_margin_ratio` by default — though the test now sets it, via
the fix in this branch). Output is scored against `brute_force_top_k`
computed in f64. The gate is `recall ≥ 0.95`, `kth_excess_rel ≤ 1e-3`
for blas/faiss; tighter for metal/stdarch.

Result: **This is where we silently had a problem.** `compute knn-blas`
was (and is) dependent on the faiss-sys static-MKL library's sgemm
output being correct, which under multi-threaded BLAS it *is not* at
dim ≥ 384 for small batch sizes. The integration test was masking this
because it was measuring engines against engines, not against truth.

The fix in this branch: the test now forces single-threaded BLAS (item
3 below), which is sufficient to bring blas back to `recall = 1.0`
against truth.

### 3. Engines-agree-with-published-output parity

*Does a freshly computed KNN result match what's already on disk?*

Measured by: `verify knn-consolidated` (our own sgemm scan) and
`verify knn-faiss` (FAISS index.search).

This is the verification the user's pipelines actually run in production.
It reads `profiles/<size>/neighbor_indices.ivecs`, recomputes top-k
independently, and reports discrepancies. It's how the user noticed the
sign-convention corruption and drove us to fix it.

## What we broke, and how

Three regressions in this work cycle, in chronological order.

### (i) Canonical-rerank sign convention

**Symptom**: user's `ibm-datapile-1b` dataset, freshly recomputed, had
`recall@100 = 0.0000` for every profile from `300k` onward.

**Cause**: I added a canonical f64 rerank post-pass to every `compute
knn*` command. The rerank writes reranked distances back to
`neighbor_distances.fvecs`. I wrote them in the kernel-internal sign
convention (`-dot` for DotProduct) rather than the FAISS publication
convention (`+dot`) that the rest of the codebase assumed when reading
those files back via Path-2 sibling-segment reuse. Two sign-flip steps
then cancelled the wrong way: the heap's threshold became the
worst-of-k instead of the best-of-k, and every subsequent profile's
merge silently accumulated the worst neighbors instead of the best.

**Fix**: uniform FAISS publication convention on every on-disk fvec
across the codebase. Helpers `kernel_to_published` / `published_to_kernel`
in `knn_segment.rs` are the only conversion points; every writer calls
the first, every reader calls the second. `CACHE_VERSION` bumped to v3
so any old mixed-convention files are ignored on first run. Unit tests
pin the convention in `knn_segment::tests`.

**What made this painful**: the sign bug only manifested on the *second*
sized profile's Path-2 reuse of the *first* profile's output. Single-
profile unit tests didn't trigger it. `verify engine-parity` didn't
trigger it either — it runs each engine in an isolated tempdir with no
sibling profiles. The only way to catch it in CI is a multi-profile
compute → verify integration test, which is now checked in as
`tests/sized_profile_sign_round_trip.rs`.

### (ii) Margin-on-by-default ate half our throughput

**Symptom**: user's production pipeline runs slowed dramatically even
at 100k base vectors. Reported as "much slower than the 380K
base-vec/s we were seeing before."

**Cause**: I had shipped the canonical rerank with `RERANK_MARGIN_RATIO = 3`
as the default. That meant every engine's per-query heap was cap-40
instead of cap-10, the threshold-pruning cutoff was the 40th-worst
distance instead of the 10th-worst, and far more candidates entered
each heap. Heap ops also went from O(log 10) to O(log 40). Combined,
2-5× slowdown on real data and worse on pathological high-dim.

**Fix**: margin is opt-in now. Default `RERANK_MARGIN_RATIO_DEFAULT = 0`.
Engines read `rerank_margin_ratio` / `rerank_margin` from options.
`verify engine-parity` sets `rerank_margin_ratio=3` when invoking
sub-engines (it wants bit-identical agreement, throughput isn't the
point). Production `compute knn*` callers leave it unset → internal_k
= k → original heap size, original pruning aggressiveness, no slowdown.
The canonical rerank still runs at margin=0 (just reorders within the
engine's existing top-k via f64 math — cheap).

### (iii) faiss-sys static-MKL corrupts every sgemm in the process

**Symptom**: `cross_engine_knn_parity` test has been failing since
before this work. I kept noting it as "pre-existing, unrelated,"
proposed two fixes (loosen gates, opt into margin), and was confident
(b) would close it. It didn't. `blas` at dim=4096 DOT_PRODUCT against
truth still scored `recall = 0.245` with margin=3·k applied.

**Cause**: faiss-sys v0.7.0 with `static` feature statically links MKL.
That MKL has an ABI mismatch (`FINTEGER=long` vs MKL LP64's 32-bit
argument convention) plus a subtler tiling issue. Under multi-threaded
MKL at dim ≥ 384 with small batch sizes, sgemm silently returns
garbage. This was previously documented in
`docs/design/faiss-blas-abi-bug.md` as a FAISS-only problem with a
batch-size-cap workaround applied inside `compute_knn_faiss.rs`.

**But the bug is process-wide, not FAISS-scoped.** Every `cblas_sgemm`
symbol in the binary resolves to the same poisoned static MKL once
faiss-sys is linked in. Our own `compute knn-blas` calls into that
MKL, so does `verify knn-consolidated`'s scan path, so does
`verify knn-faiss`. The batch-size cap in `compute_knn_faiss.rs` papers
over one symptom (the zero-distance corruption on large batches); it
doesn't help the small-batch-wrong-neighbors failure mode, and it
doesn't help *any* of the other sgemm callers.

**Fix**: centralize a `set_single_threaded_if_faiss()` helper in
`pipeline::blas_abi` that sets `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1` when the `faiss` feature is compiled in.
Call it at the top of every `execute()` that might eventually drive
sgemm — `compute knn-blas`, `compute knn-faiss`, `verify knn-faiss`,
`verify knn-consolidated`, `verify dataset-knnutils`,
`verify engine-parity` (the last one also needs single-threaded for
determinism, independent of the bug).

Verified empirically: blas at dim=4096 DOT_PRODUCT with single-
threaded MKL produces `recall = 1.000` against f64 truth, from a
failing `0.245` multi-threaded.

## Strategic policy change this branch lands

Given the above, compute/verify wiring now splits along a cleaner line:

**Compute**: default to `compute knn` (stdarch SIMD) for every
personality in `prepare/import.rs::cmd_knn`. Rationale:

- stdarch and metal don't touch BLAS. No dependency on the poisoned
  MKL symbol space, no threading constraints.
- After the canonical f64 rerank, their output is bit-identical to
  what a correctly-functioning BLAS path would produce. `knn_utils`
  compatibility is preserved because the *ranking* is what
  `knn_utils` compares against, not the raw f32 distance values.
- 50–180× faster than `compute knn-blas`, which — being forced to
  single-threaded MKL to stay correct — has given up its only real
  throughput advantage.

**Verify**: default to `verify knn-faiss` (when the `faiss` feature is
compiled in) for every non-knn_utils personality. Rationale:

- FAISS is an *independent* implementation, giving real cross-
  validation rather than self-certification. `verify knn-consolidated`
  shared the sgemm path with `compute knn-blas`, so using it to verify
  blas output was closer to a mirror than an audit.
- With `pipeline::blas_abi::set_single_threaded_if_faiss()` forced at
  the top of `verify_knn_faiss::execute()` and the batch-size cap
  already in place from the original bug workaround, FAISS is safe to
  use in production.
- `knn_utils` personality still uses `verify dataset-knnutils` for
  numpy-path cross-validation.

**compute knn-blas stays registered** but is no longer a default.
Users who want it for explicit bit-for-bit numpy comparison can still
invoke it; it forces single-threaded BLAS internally now.

## What this memo owes the user that earlier discussions didn't

Three places I conflated things or underspecified:

1. **I claimed `verify engine-parity` showed 100% parity "across the
   board"** when describing the post-rerank state. True as stated, but
   I didn't clarify that "across the board" meant engines-agreeing-
   with-each-other (#1 above), not engines-agreeing-with-f64-truth
   (#2). The user reasonably thought those were the same until they
   saw `cross_engine_knn_parity` still failing and asked "did you
   misspeak?" Yes, I underspecified — the nuance is substantive, not
   semantic.

2. **I proposed "have the test set `rerank_margin_ratio=3`" as a fix
   for `cross_engine_knn_parity`** with confidence that turned out to
   be unjustified. I reasoned from the case where it *does* work
   (engines-agreeing-with-each-other) without verifying it transferred
   to the case we were actually in (engines-agreeing-with-truth).
   Running the test took 5 seconds; I should have done it before
   proposing. That's on me.

3. **I described the faiss-sys MKL bug as "a known FAISS issue with a
   documented workaround"** in several earlier messages, adopting the
   framing of the existing design doc. I didn't flag — because I
   hadn't looked carefully — that the same static library poisons
   every sgemm caller in the process, including our own blas, and
   that the documented batch-size-cap mitigation only addresses one
   symptom of it. Rewriting `docs/design/faiss-blas-abi-bug.md` to
   reflect the wider scope is a follow-up.

## What's still loose

- `docs/design/faiss-blas-abi-bug.md` still reads as FAISS-scoped;
  rewrite to process-wide scope pending.
- `verify_dataset_knnutils`'s JSON output format isn't checked by any
  end-to-end test; this wasn't related to the regressions here but is
  an obvious hole now that we're defaulting verification through
  FAISS.
- The Python-reference verification path hasn't been re-run end-to-end
  since the convention changes; user's `ibm-datapile-1b-2` verification
  confirms the compute side is fine, but the bytes-match-numpy claim
  in the `knn_utils` personality docs is untested against the new
  writer conventions.
