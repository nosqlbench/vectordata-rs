// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vendored streaming sketches for the metadata survey.
//!
//! Four sketches cover the survey's bounded-memory needs:
//!
//! - [`reservoir::Reservoir`]: Algorithm-R reservoir sampling — uniform
//!   random sample of fixed size from a stream of unknown length.
//! - [`misra_gries::MisraGries`]: heavy-hitter detection — top-K
//!   frequent items with bounded-error frequency estimates.
//! - [`hll::HyperLogLog`]: cardinality estimation — distinct-count
//!   estimate with constant memory and ~σ = 1.04/√m relative error.
//! - [`kll::KllSketch`]: quantile estimation — rank approximation
//!   with ε ≈ 1.0/k error using O(k · log(n/k)) memory.
//!
//! These are vendored rather than pulled from crates so that the JSON
//! snapshot format is stable across crate-version churn and remains
//! under this workspace's serialization-versioning policy.
//!
//! Every sketch implements the minimal [`Sketch`] trait so the
//! orchestrator can poll memory usage uniformly. The per-sketch APIs
//! are otherwise free-form (a different item type, a different
//! summary shape) and not forced through a common observe/finalize
//! shape.

pub mod hll;
pub mod kll;
pub mod misra_gries;
pub mod reservoir;

pub use hll::HyperLogLog;
pub use kll::KllSketch;
pub use misra_gries::MisraGries;
pub use reservoir::Reservoir;

/// Common surface for bounded-memory streaming sketches.
///
/// The orchestrator uses this to enforce the aggregate memory budget
/// — each sketch reports its current footprint, and downscaling
/// decisions consult [`memory_bytes`](Self::memory_bytes) when
/// choosing what to shrink. Sketches that hand back stable snapshots
/// for the survey report do so through type-specific accessors on
/// the concrete type, not through this trait.
pub trait Sketch {
    /// Approximate current memory footprint in bytes. Implementations
    /// return their best estimate; precision matters more for the
    /// large sketches (KLL, HLL) than the small ones.
    fn memory_bytes(&self) -> usize;
}
