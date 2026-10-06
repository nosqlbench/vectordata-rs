// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust SIMD kernels for vector distance computation.
//!
//! This crate is the one place in the workspace where vector arithmetic
//! is vectorized. It covers what the KNN engines, verifiers and the
//! `explore` TUI actually compute:
//!
//! - **Pairwise distances** for `f32`, `f16` and `f64` under four
//!   metrics ([`Metric`]): squared L2, cosine distance, negated dot
//!   product, and L1. Plus the positive dot product and the L2 norm.
//! - **Transposed query batches** ([`TransposedBatch`], 16 queries in
//!   dimension-major layout) and **packed batches** ([`PackedBatches`],
//!   every 16-query sub-batch of a thread interleaved per dimension), with
//!   kernels that score one base vector against all of them in one pass.
//! - **f16 ↔ f32 bulk conversion**, bit-exact with the `half` crate.
//!
//! Everything is built on [`fearless_simd`], so nothing here compiles
//! native code. The instruction set is chosen at runtime, once: each
//! supported [`SimdLevel`] has a static table of monomorphized function
//! pointers ([`Kernels`]), and [`Kernels::detected`] hands out the table
//! for the best level the CPU supports. A kernel call never re-checks
//! CPU features.
//!
//! ```
//! use veks_simd::{Kernels, Metric};
//!
//! let kernels = Kernels::detected();
//! let l2 = kernels.distance_f32(Metric::L2);
//! assert_eq!(l2(&[1.0, 2.0, 3.0], &[1.0, 2.0, 5.0]), 4.0);
//! println!("dispatched to {}", kernels.level().name());
//! ```
//!
//! ## Distance conventions
//!
//! Every distance is "smaller is closer", so one max-heap serves all
//! metrics: L2 is squared (no square root), cosine is `1 − cos θ`, and
//! the dot product is negated. [`Kernels::dot_f32`] is the exception,
//! returning the plain inner product for callers that maximize it.
//!
//! ## Alternative backend
//!
//! With the `simsimd` feature, the `simsimd_backend` module exposes simsimd's
//! pairwise kernels behind the same function-pointer types, for parity
//! runs that want to compare against it. Native kernels are always the
//! default.

#![warn(missing_docs)]

mod batch;
mod convert;
mod f16_lanes;
mod level;
mod metric;
mod pairwise;
mod table;

#[cfg(feature = "simsimd")]
pub mod simsimd_backend;

pub use batch::{BATCH_WIDTH, PackedBatches, TransposedBatch};
pub use level::SimdLevel;
pub use metric::Metric;
pub use table::{
    BatchFnF32, ConvertBytes, ConvertF16ToF32, ConvertF32ToF16, DistFnF16, DistFnF32,
    DistFnF64, DualBatchFnF32, Kernels, NormFnF32, PackedNegDotFn,
};
