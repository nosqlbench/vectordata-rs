// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Rewrites under test. Each one applies the same map to the same
//! virtualized store and reports what it did, so their traces are
//! directly comparable.

pub mod gsplat;
pub mod naive;
pub mod staged;

use crate::model::{Geometry, Map, Sink, Trace};

/// A rewrite applies `map` to a store of `geometry`, filling a sink.
pub trait Rewrite {
    /// Name used in study output.
    fn name(&self) -> &'static str;

    /// Run the rewrite, returning what was produced and what it cost.
    ///
    /// `budget_bytes` is the memory the rewrite may hold resident. A
    /// rewrite that does not use memory ignores it.
    fn run(&self, geometry: Geometry, map: &Map, budget_bytes: u64) -> (Sink, Trace);
}

/// Run a rewrite and check it produced what the map called for.
///
/// Correctness needs no payload comparison: a record's virtual content
/// *is* its source ordinal, so `sink[i] == map[i]` states exactly
/// `output[i] = source[map[i]]`.
pub fn run_verified(
    algo: &dyn Rewrite,
    geometry: Geometry,
    map: &Map,
    budget_bytes: u64,
) -> Result<Trace, String> {
    let (sink, trace) = algo.run(geometry, map, budget_bytes);
    if sink.unfilled() != 0 {
        return Err(format!(
            "{}: {} output slots never written",
            algo.name(),
            sink.unfilled()
        ));
    }
    if !sink.matches(map) {
        return Err(format!("{}: output does not match the map", algo.name()));
    }
    Ok(trace)
}
