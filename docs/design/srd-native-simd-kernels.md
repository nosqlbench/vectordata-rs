# SRD — Native SIMD kernels by default

**Status:** proposed
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
it against `W` queries at once, for l2sq, neg-dot, cosine and L1, f32 and
f16. Today `W = SIMD_BATCH_WIDTH = 16`, which is one AVX-512 register of
f32. The native crate exposes one width for the engines to lay batches
out by, chosen per dispatch level (§3, SK-11), not a hard-coded 16.

**SK-7. Dual batch and packed tiles.** Two batches per base vector
(l2sq, neg-dot, cosine, L1), and the `PackedBatches` /
`tiled_neg_dot_all_batches_f32` layout for dot: all sub-batches for one
dimension in adjacent cache lines, `DIM_TILE = 64`, with the working set
sized to L1 as the current comment budgets it.

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
| `vectordata` | `cli`, `explore` on native kernels | `simsimd` (forwarded) |
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

**Crate name.** `veks-simd` follows the `veks-*` library convention
already used under `vectordata`. `vectordata-kernels` would name the
consumer rather than the toolkit. Settle before the crate is published,
since the name becomes a published dependency of `vectordata`.

**Backend selection when several are compiled in.** Either a per-engine
flag (`--backend native|simsimd`, `--backend gemm|system`) mirrored as a
step option in `dataset.yaml`, or separate command names. The flag keeps
the command surface flat, and the CLI/YAML mirror rule applies to it
either way.

**Whether `knn-stdarch` survives.** Once its kernels are the shared
native ones, it differs from `compute knn` only in its I/O strategy
(streaming pread and a shared base segment, versus mmap partitions). If
that difference is worth keeping, it stays as a named engine. If not, it
folds into `compute knn` as an I/O mode. The characterization numbers
after SK-14 should decide it.

**f16 conversion inside dispatched kernels.** `fearless_simd`'s
`kernel!` macro mixes intrinsics into a dispatched function but does not
annotate generic functions. The f16 batch kernels need F16C or NEON
`fcvt` inside a width-generic body. Prototype this first. If the macro
cannot express it, the fallback is a per-level non-generic f16 kernel
for the batch shapes only, generated by `macro_rules!`.

**The i8 kernels.** `select_distance_fn_i8` has no callers.
`ElementType::supports_simd_distance` lists I8 anyway. Either remove the
claim with the kernels, or give i8 a native kernel and a caller. Nothing
in the current formats needs it.

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
