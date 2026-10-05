# SRD — Native SIMD kernels by default

**Status:** implemented (crate `veks-simd`); §8 records what
implementation settled
**Scope:** a new pure-Rust kernel crate; `simd_distance`, the four
`compute knn` engines and their dispatch, `veks-core`'s element
conversion, the `explore` TUI's analytics, the `*_knnutils` norm
commands, the workspace feature matrix, and the release CI matrix.

## 1. Problem

The project's vector arithmetic runs through four implementations, and
none of them is both portable and fast:

- **simsimd** (C, compiled by `cc`) serves pairwise dot, l2sq and cosine
  for f32, f16 and f64. It is a hard dependency of `veks-pipeline` and
  `veks`, and of `vectordata` through its default `explore` feature.
- **Hand-written `std::arch`** in `veks-pipeline/src/pipeline/simd_distance.rs`
  and `compute_knn_stdarch.rs` serves L1, the 16-query transposed batch
  kernels, the dual-batch and packed/tiled dot kernels, and all of
  `knn-stdarch`. Every one of these is `#[cfg(target_arch = "x86_64")]`.
  Most of the batch kernels are AVX-512 only.
- **System BLAS** (`cblas_sgemm`, `cblas_snrm2` through hand-declared
  `extern "C"` blocks and a `build.rs` link directive) serves `knn-blas`,
  the blas-mirror in `verify engine-parity`, `verify knn-consolidated`'s
  sgemm scan, and four `*_knnutils` norm commands. It is unix only.
- **FAISS** (`faiss-sys`, C++ with statically linked MKL) serves
  `knn-faiss`, and its BLAS ABI defect forces every other BLAS user in
  the process to one thread (`faiss-blas-abi-bug.md`).

Three consequences follow.

**SK-1.** An upstream crate that depends on `vectordata` with `cli` and
`explore` needs a C toolchain for simsimd. The requirement here is that
it does not: the default feature set of every library crate in this
workspace builds from Rust sources alone as far as SIMD and linear
algebra are concerned. Native-code engines remain available, but only
as features someone enables on purpose.

**SK-2.** On aarch64, the throughput-defining kernels are scalar. The
transposed batch kernels that let `knn-metal` make one pass over the
base per 16 queries, the f16 converters, and every `knn-stdarch` kernel
fall back to scalar loops. Only simsimd's pairwise kernels are
vectorized there. On x86_64 without AVX-512 the batch kernels are scalar
too. The release matrix ships two aarch64 targets
(`aarch64-unknown-linux-musl`, `aarch64-apple-darwin`). The requirement
is that every shipped target gets vectorized kernels for every metric,
element type and batch shape the engines use.

**SK-3.** The released `veks` binary is built `--no-default-features`
because system BLAS has no clean story on Windows MSVC, so released
builds carry no `knn-blas` and no `knnutils` commands. The requirement
is that matrix-multiply KNN and the norm commands work on every release
target without system BLAS.

## 2. What the kernels must cover

This list is the contract the native crate must meet. It is derived from
what the engines and commands call today, not from what a general SIMD
library offers.

**SK-4. Pairwise distance.** Metrics dot (returned negated, so smaller is
closer), l2sq (no square root), cosine, and L1. Element types f32, f16
and f64. f16 is converted to f32 lane by lane and accumulated in f32; f64
accumulates in f64. All four metrics for all three types.
`select_distance_fn_i8` has no callers and is not carried forward (§7).

**SK-5. Cosine modes.** Both `CosineMode`s: input already normalized (the
kernel is dot), and computed (dot over the product of norms, with query
norms precomputed and the base norm computed in the same pass).

**SK-6. Transposed query batches.** The `TransposedBatch` layout
`data[d * W + qi]`: for each base dimension, broadcast `base[d]` and FMA
it against `W` queries at once, for l2sq, neg-dot, cosine and L1, over
f32 base vectors. `W = 16` at every level: sixteen f32 lanes are one
AVX-512 register, two AVX2 registers, four NEON or SSE registers (§8).
f16 base vectors are widened to f32 once per base vector and scored with
the f32 kernels, which is what the engines already do.

**SK-7. Dual batch and packed batches.** Two batches per base vector
(l2sq, neg-dot, cosine, L1), and the `PackedBatches` layout for dot: all
sub-batches for one dimension in adjacent cache lines, scored in one
pass per base vector with the accumulators held in registers (SK-26).

**SK-8. Matrix multiply.** Row-major Q·Bᵀ in f32 for query sub-batches
against a base chunk, with L2 expanded as |q|² + |b|² − 2q·b from
precomputed squared norms. That is the `knn-blas` shape, the blas-mirror
shape, and `verify knn-consolidated`'s scan.

**SK-9. Conversion and norms.** Bulk f16→f32 and f32→f16 (round to
nearest even), replacing both `convert_f16_to_f32_bulk` and the
`veks-core::formats::convert` kernels. f32 L2 norm, replacing
`cblas_snrm2`. f32↔f64 stays as it is.

**SK-10. Out of scope.** Top-k heaps, the f64 rerank
(`knn_segment::rerank_topk_f64`), stats accumulators, sort/dedup and
k-means orchestration stay scalar Rust with rayon. They are not where
time goes, and the f64 rerank is scalar on purpose: it is the reference
that makes engines byte-identical. bf16, binary/Hamming, integer and
quantized kernels are not required, because no storage format or command
uses them.

## 3. Design

**SK-11. One kernel crate, below `vectordata`.** A new workspace crate
(working name `veks-simd`, §7) owns every kernel in §2. It depends on
`fearless_simd` and nothing native. `vectordata` (for `explore`),
`veks-core` (for conversion) and `veks-pipeline` (for everything else)
depend on it. `simd_distance.rs` becomes a thin adapter over it or is
removed. `compute_knn_stdarch.rs` keeps its engine (the streaming pread
I/O design) but loses its private kernels.

The crate sits below `vectordata` because `explore` needs the pairwise
kernels and `vectordata` must not depend on the pipeline. `veks-anode`
already sets the precedent for a `veks-*` crate under `vectordata`.

**SK-12. `fearless_simd` as the kernel layer.** Chosen because it is the
only candidate that meets all of:

- stable Rust;
- runtime dispatch built in (`#[simd]`), with the dispatch policy owned
  by the binary rather than the library;
- AVX2, AVX-512, NEON and wasm levels;
- generic over vector width, so one transposed-batch kernel compiles to
  16 f32 lanes on AVX-512, 8 on AVX2, and 4×4 on NEON;
- raw intrinsics can be mixed into a dispatched function, which is how
  F16C and NEON `fcvt` conversion get used inside the batch kernels;
- AVX-512 used only on Ice Lake and later, so no frequency throttling on
  early AVX-512 parts;
- v1.0 with a security policy, and far less `unsafe` internally than the
  alternatives.

Its one documented gap, trigonometry, is not on the §2 list.

**SK-13. Pure-Rust matrix multiply.** The default `knn-blas` engine,
the blas-mirror and the consolidated sgemm scan use the `gemm` crate
(the engine under faer). It is pure Rust, dispatches at runtime through
pulp, threads with rayon, and has f16 kernels that compute in f32. Its
AVX-512 kernels are behind its `x86-v4` feature, which this workspace
enables. Full faer is not taken: nothing here needs decompositions.

Different summation order from MKL or OpenBLAS is acceptable. The engines
already disagree in the last bits before the f64 rerank, and the rerank
is what makes them byte-identical (SK-19).

**SK-14. Accumulator discipline.** Pairwise reductions use at least four
independent accumulators per lane group. `knn-stdarch` is limited by FMA
latency today because each of its kernels uses one
(commands/mod.rs:186-200). Batch kernels already carry `W` independent
accumulators by construction and keep that shape.

**SK-15. Dispatch is decided once.** Level selection happens at engine
setup, never per vector. The selected function pointers or level token
go into the per-thread state the engines already build, in line with the
existing `select_*` functions. A kernel call on the hot path never
re-checks CPU features.

**SK-16. Reported level is the dispatched level.** `info compute` and
every engine's startup line report the level the kernel crate actually
selected (for example `avx512`, `avx2+fma`, `neon`, `scalar`), taken from
the crate, not from compile-time `cfg!(target_feature = ...)`.
`info_compute.rs:177-188` reports the compile-time baseline today, which
is SSE2 on a release build regardless of the CPU.

## 4. Features

**SK-17. Default features are native only.**

| Crate | Default | Opt-in |
|---|---|---|
| kernel crate | native kernels | `simsimd`: simsimd as an alternative pairwise backend |
| `vectordata` | `cli`, `explore` on native kernels | — (explore has no backend choice) |
| `veks-pipeline` / `veks` | native kernels, `gemm`-based `knn-blas`, `knnutils` commands on native norms | `blas-system`: `cblas_sgemm` / `cblas_snrm2` against the system library (unix); `faiss`; `simsimd`; `embed`, `embed-cuda` |
| root `vectordata-rs` | mirrors `veks` | mirrors `veks` |

The `knnutils` feature stops implying system BLAS. It keeps gating what
it is for: the numpy-parity personality, `rand_mt`, and the commands that
mirror knn_utils. System BLAS moves to `blas-system`, so the
`cfg(all(feature = "knnutils", unix))` gates become
`cfg(all(feature = "blas-system", unix))`, and the `build.rs` link
directive moves with them. `veks` declares simsimd today and never uses
it; that dependency is removed.

**SK-18. Engine identity under backends.** The command names
(`compute knn`, `knn-stdarch`, `knn-blas`, `knn-faiss`) and their
semantics stay. What changes is which backend serves them:

- `compute knn` (metal) uses native kernels by default and simsimd's
  pairwise kernels when built with `simsimd` and asked for it.
- `knn-blas` uses `gemm` by default and the system library when built
  with `blas-system` and asked for it.

The backend in use is recorded where the engine is recorded today, in
the step's provenance and its startup line, so two runs with different
backends never look like the same computation. How the backend is chosen
when both are compiled in is open (§7).

## 5. What must not change

**SK-19. Results.** For every engine, metric and element type in the
parity sweep (`12-knn-utils-verification.md`, dims 8 through 4096),
output after the f64 rerank is byte-identical to what the current
implementation produces. Before the rerank, distances agree within the
tolerances the parity tests already use: f32 within the existing
relative bound, f16 within the `element_type.rs` epsilon.

**SK-20. Throughput on x86_64 AVX-512.** On the reference shape from
`knn-engine-characterization.md` (100K base × 10K queries × dim 384, DOT,
k = 100, the 128-thread AVX-512 host), the native `compute knn` sustains
at least the current 2.4 B distances/s, and the native `knn-blas` at
least the current single-threaded-MKL 2.2 B/s. The native `knn-stdarch`
must not regress from 1.71 B/s; SK-14 is expected to raise it.

**SK-21. Memory and I/O behavior.** Partitioning, madvise, streaming
pread, segment caching and the per-thread batch sizes stay as they are.
This SRD changes arithmetic, not data movement.

## 6. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | every pairwise kernel × metric × {f32, f16, f64}, dims 1–33, 127, 128, 384, 1536, 4096, against an f64 scalar reference | within SK-19's tolerance; remainder lanes covered |
| 2 | each batch kernel (SK-6, SK-7), batch partially filled, every metric | per-query results equal to the pairwise kernel's |
| 3 | `gemm` scan vs pairwise neg-dot and expanded L2 | within SK-19's tolerance |
| 4 | f16→f32 over all 65,536 f16 bit patterns; f32→f16 over their round trips, the rounding midpoints between them, and NaN/Inf/subnormal/overflow values | bit-exact with `half`'s conversion |
| 5 | each dispatch level available on the host, forced | identical results across levels within tolerance |
| 6 | level selection on a host with AVX-512 / AVX2 / NEON | selects that level, never scalar (SK-16) |
| 7 | parity sweep, all engines, native defaults | byte-identical after rerank to the stored current outputs (SK-19) |
| 8 | manifests of `vectordata` (default + `cli`), `veks-core`, `veks-pipeline`, `veks` | simsimd, faiss and system BLAS are optional and off by default |
| 9 | `cargo tree -e normal,build` for `vectordata` default + `cli` | no SIMD or BLAS crate that builds native code |
| 10 | release CI on all four targets | builds with `knnutils` on, runs case 1 and case 7 natively, including on the two aarch64 runners |
| 11 | reference-shape benchmarks | meet SK-20; aarch64 numbers recorded as a new baseline |

**SK-22.** Case 6 is the guard against the failure this design is most
exposed to: a dispatch mistake that silently selects scalar. It passes
everything else and shows up only as "the machine was slow". It must be
a deterministic test, in line with the hot-path guard practice: assert
the selected level's identity, not a timing.

Cases 8 and 9 hold SK-1 in place. They belong next to the packaging
tests in `src/main.rs`, which already read the manifests to keep the
install story honest.

## 7. Open

**Whether `knn-stdarch` folds into `compute knn`.** It keeps its name
for now (§8); with the shared kernels it differs from `compute knn` only
in its I/O strategy. Decide from characterization on production shapes.

**Not SIMD, but in the way of SK-1.** `vectordata` with `cli` also builds
C through `aws-lc-sys` (cmake), pulled in by reqwest's default rustls
crypto provider. Removing it means choosing `ring` (still `cc`, no cmake)
or a RustCrypto-based rustls provider (pure Rust, less mature for TLS).
That decision belongs to the transport layer, not this SRD. It is listed
so that "no native code by default" is not claimed while it remains.

**Stale engine docs.** `knn-engines.md` names faiss as the default and
`knn-engine-characterization.md` names stdarch. The code routes
`compute knn` to metal, and a `veks/Cargo.toml` comment says bootstrap
emits `knn-blas`. Correct these as part of SK-18, so they describe the
backends as built.

## 8. Settled during implementation

**SK-23. Crate and backend surface.** The crate is `veks-simd`. A
command that offers a backend takes a `backend` option, mirrored as a
CLI flag and a `dataset.yaml` step option like every other option:

| Commands | Values | Default |
|---|---|---|
| `compute knn` | `native`, `simsimd` | `native` |
| `compute knn-blas`, `verify knn-consolidated`, `verify dataset-knnutils` | `gemm`, `system` | `gemm` |

An unavailable value (`simsimd` without the feature, `system` without
`blas-system` or off unix) is an error naming the missing feature, never
a silent fallback. Segment caches are keyed by backend: `knn-metal` and
`knn-metal-simsimd`; `knn-blas-gemm` and `knn-blas` (the system backend
keeps the historical name, which every cache written before `gemm`
existed carries). The cache claims follow the step's backend, so cache
GC keeps the right namespace live. `verify engine-parity` runs the
variants as engines of their own (`metal-simsimd`, `blas-system`,
`blas-mirror-system`) and reports a variant that is not compiled in as
skipped. `knn-stdarch` remains a named engine.

**SK-24. Batch width is a layout constant.** Sixteen lanes at every
level. A per-level width would have rewritten every engine's batch
bookkeeping to gain nothing: on AVX2 and NEON the 16-lane accumulator is
two and four registers whose dependency chains are independent, which
is the ILP SK-14 asks for.

**SK-25. Lane-order invariant.** The single-batch, dual-batch and packed
kernels compute each query lane with the same operations in the same
order, so a query's distance is bit-identical whichever kernel scored
it. `compute knn` therefore no longer pads its sub-batches to an even
count to keep every query on the dual kernel. A `veks-simd` test pins
the invariant bitwise across every level and every packed group size.

**SK-26. Packed groups per register file.** The packed neg-dot kernel
keeps one accumulator per sub-batch in registers for a whole pass over
the base vector, in groups sized to the level: 16 sub-batches on
AVX-512, 6 on AVX2 and NEON, 3 on SSE. A query set wider than one group
re-reads the base vector (from L1) per group instead of spilling.

**SK-27. f16 lanes.** One per-level trait converts 8 or 16 lanes: F16C
(`vcvtph2ps`, `vcvtps2ph`) on the AVX2 and AVX-512 levels, and an exact
integer widening everywhere else, including NEON, whose own conversion
instructions need Rust's unstable `f16` intrinsics. The integer path
rebiases normal values with an integer add and rebuilds subnormals from
an exact integer-to-float conversion, so no float operation ever sees a
subnormal operand: the first version's single float rebias multiply hit
x86 microcode assists on f16 subnormals and ran 2.8× slower. Narrowing
off F16C goes through `half`. Both directions are bit-exact with `half`
for every f16 bit pattern at every level.

**SK-28. Removed rather than ported.** The i8 kernels (no callers;
`supports_simd_distance` no longer claims i8), the f16 batch kernels
(only ever used as a capability check; see SK-6), the tiled multi-batch
dot kernel (superseded by the packed kernel, no callers), and
`verify knn-groundtruth`'s pairwise fallbacks (every metric now has a
batch kernel at every level). `veks` no longer depends on simsimd at
all; `vectordata`'s explore uses the native kernels unconditionally.

**SK-29. AVX-512 means Ice Lake.** `fearless_simd` selects its AVX-512
level only for the Ice Lake feature set. Skylake-SP and Cascade Lake
hosts, on which the hand-written kernels used bare `avx512f`, now run
the AVX2 tables. That is the throttling policy SK-12 chose; whether it
costs throughput on those parts is measured there, not assumed.

