// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Executable models of the SPLAT family.
//!
//! The documents in `docs/gsplat/` make claims: that every mapped record
//! is read exactly once, that reads ascend within a pass, that resident
//! memory stays inside the budget, that container touches follow
//! `A(P) = P · (1 − exp(−w / P))`. Prose cannot fail a test. This crate
//! exists so those claims can.
//!
//! Nothing here performs real I/O. A [`Geometry`](model::Geometry)
//! describes a store the way the cost model describes one — a record
//! count, a record size, a container size — and every access an
//! algorithm makes is recorded as a virtual operation in a
//! [`Trace`](model::Trace). Costs are then *computed* from the trace
//! rather than measured from a device, which makes them exact,
//! reproducible, and comparable against the formulas.
//!
//! Three things can be checked this way:
//!
//! - **Correctness.** A rewrite is correct when the sink ends up holding
//!   `map` — since a record's virtual payload is its own source ordinal,
//!   `sink[i] == map[i]` is the whole of `output[i] = source[map[i]]`.
//! - **Invariants.** [`check`] turns each documented invariant into an
//!   assertion over the trace.
//! - **Cost.** [`study`] sweeps parameters and prints measured cost
//!   beside predicted cost, so a wrong formula shows up as a column that
//!   does not line up.
//!
//! Scope: flat, single-space rewrites — the [gsplat](../../docs/gsplat)
//! core. Structured and multi-space variants are modelled in the same
//! terms but are not implemented here yet; [`model::Geometry`] carries
//! the container notion they will need.

pub mod algo;
pub mod cache;
pub mod check;
pub mod device;
pub mod io;
pub mod model;
pub mod price;
pub mod regime;
pub mod study;

pub use algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather, naive::NaiveScatter};
pub use model::{Geometry, Map, Metrics, Trace};
