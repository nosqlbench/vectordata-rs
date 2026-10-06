# veks-simd

Pure-Rust SIMD kernels for vector distance computation, with the
instruction set chosen at runtime. It is the kernel layer under
[`vectordata`](https://crates.io/crates/vectordata)'s `explore` TUI and
the [`veks`](https://crates.io/crates/veks) KNN engines, and nothing in
it builds native code.

- **Pairwise distances** for `f32`, `f16` and `f64`: squared L2, cosine
  distance, negated dot product and L1, plus the plain dot product and
  the L2 norm. Reductions keep four independent accumulators.
- **Query batches**: 16 queries in dimension-major layout
  (`TransposedBatch`), and every 16-query sub-batch of a thread's query
  set interleaved per dimension (`PackedBatches`), scored against one
  base vector per pass. The single, dual and packed kernels compute each
  query lane identically, so results do not depend on how queries were
  batched.
- **f16 ↔ f32 bulk conversion**, bit-exact with the
  [`half`](https://crates.io/crates/half) crate at every level, over typed
  slices or unaligned little-endian bytes.

Dispatch covers AVX-512 (Ice Lake class), AVX2 + FMA, SSE4.2 and SSE2 on
x86, NEON on aarch64, and wasm `simd128`. It is decided once per process;
after that every kernel is a plain function-pointer call.

```rust
use veks_simd::{Kernels, Metric};

let kernels = Kernels::detected();
let l2 = kernels.distance_f32(Metric::L2);
assert_eq!(l2(&[1.0, 2.0, 3.0], &[1.0, 2.0, 5.0]), 4.0);
println!("dispatched to {}", kernels.level().name());
```

Distances are "smaller is closer": L2 is squared, cosine is `1 − cos θ`,
and the dot product is negated, so one max-heap serves every metric.

## Features

- `simsimd` (off by default) — simsimd's C pairwise kernels behind the
  same function-pointer types, as an independent implementation for
  parity runs. Needs a C compiler.

## Design

The requirements, the settled decisions and the measured throughput are
in the
[native SIMD kernels SRD](https://github.com/nosqlbench/vectordata-rs/blob/main/docs/design/srd-native-simd-kernels.md).

License: Apache-2.0
