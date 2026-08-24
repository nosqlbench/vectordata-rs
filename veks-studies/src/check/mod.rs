// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The documented invariants, as checks over a trace.
//!
//! `docs/gsplat/README.md` lists four: single read and single write,
//! monotone access, bounded memory, determinism. Each one below is that
//! claim restated as something that can fail, and named so a failure
//! reads as the invariant it broke.

use crate::model::{Map, Trace};
use std::collections::HashMap;

/// An invariant that did not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub invariant: &'static str,
    pub detail: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.invariant, self.detail)
    }
}

/// Check every invariant that applies to a gsplat-shaped rewrite.
///
/// The naive baselines deliberately violate some of these — that is what
/// makes them the baselines — so this is meant for gsplat traces.
pub fn check_all(trace: &Trace, map: &Map, budget_bytes: u64) -> Vec<Violation> {
    let mut v = Vec::new();
    v.extend(single_read(trace, map));
    v.extend(monotone_access(trace));
    v.extend(single_write(trace, map));
    v.extend(bounded_memory(trace, budget_bytes));
    v
}

/// **Single read.** Every mapped source record is read exactly once
/// across all passes.
pub fn single_read(trace: &Trace, map: &Map) -> Vec<Violation> {
    let mut counts: HashMap<u64, u64> = HashMap::new();
    for reads in trace.reads_per_pass() {
        for ordinal in reads {
            *counts.entry(ordinal).or_default() += 1;
        }
    }

    let mut out = Vec::new();
    for &wanted in &map.0 {
        match counts.get(&wanted).copied().unwrap_or(0) {
            1 => {}
            0 => out.push(Violation {
                invariant: "single read",
                detail: format!("source {wanted} is mapped but never read"),
            }),
            n => out.push(Violation {
                invariant: "single read",
                detail: format!("source {wanted} read {n} times"),
            }),
        }
        if out.len() > 8 {
            break;
        }
    }
    out
}

/// **Monotone access.** Within a pass, source reads ascend by ordinal —
/// which, given the monotonicity premise, means ascending by address.
pub fn monotone_access(trace: &Trace) -> Vec<Violation> {
    let mut out = Vec::new();
    for (pass, reads) in trace.reads_per_pass().into_iter().enumerate() {
        for pair in reads.windows(2) {
            if pair[1] < pair[0] {
                out.push(Violation {
                    invariant: "monotone access",
                    detail: format!(
                        "pass {pass} reads {} after {} — a backward step",
                        pair[1], pair[0]
                    ),
                });
                break;
            }
        }
    }
    out
}

/// **Single write.** Every output byte is written exactly once, and each
/// pass writes one contiguous range.
pub fn single_write(trace: &Trace, map: &Map) -> Vec<Violation> {
    use crate::model::Op;

    let mut covered = vec![0u32; map.0.len()];
    let mut out = Vec::new();

    for (pass, ops) in trace.passes().into_iter().enumerate() {
        let ranges: Vec<(u64, u64)> = ops
            .iter()
            .filter_map(|op| match op {
                Op::WriteRange {
                    first_slot,
                    records,
                } => Some((*first_slot, *records)),
                _ => None,
            })
            .collect();
        if ranges.len() > 1 {
            out.push(Violation {
                invariant: "single write",
                detail: format!("pass {pass} wrote {} ranges, not one", ranges.len()),
            });
        }
        for (first, count) in ranges {
            for slot in first..first + count {
                if let Some(c) = covered.get_mut(slot as usize) {
                    *c += 1;
                }
            }
        }
    }

    for (slot, &times) in covered.iter().enumerate() {
        if times != 1 {
            out.push(Violation {
                invariant: "single write",
                detail: format!("output slot {slot} written {times} times"),
            });
            break;
        }
    }
    out
}

/// **Bounded memory.** Resident state never exceeds the budget, and
/// never scales with the size of the store.
pub fn bounded_memory(trace: &Trace, budget_bytes: u64) -> Vec<Violation> {
    // The plan rides alongside the buffer rather than inside it, so the
    // ceiling is the budget plus the plan for one segment.
    let g = trace.geometry;
    let per_segment = g.records_per_segment(budget_bytes);
    let ceiling = budget_bytes + per_segment * 16;
    if trace.peak_resident_bytes > ceiling {
        return vec![Violation {
            invariant: "bounded memory",
            detail: format!(
                "peak resident {} exceeds budget {} (+ plan, ceiling {})",
                trace.peak_resident_bytes, budget_bytes, ceiling
            ),
        }];
    }
    Vec::new()
}

/// **Determinism.** Two runs of the same rewrite over the same inputs
/// produce identical output and identical costs.
pub fn deterministic(
    algo: &dyn crate::algo::Rewrite,
    geometry: crate::model::Geometry,
    map: &Map,
    budget_bytes: u64,
) -> Vec<Violation> {
    let (sink_a, trace_a) = algo.run(geometry, map, budget_bytes);
    let (sink_b, trace_b) = algo.run(geometry, map, budget_bytes);
    let mut out = Vec::new();
    if sink_a.slots != sink_b.slots {
        out.push(Violation {
            invariant: "determinism",
            detail: "two runs produced different output".into(),
        });
    }
    if trace_a.metrics() != trace_b.metrics() {
        out.push(Violation {
            invariant: "determinism",
            detail: "two runs produced different costs".into(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::Rewrite;
    use crate::algo::{gsplat::Gsplat, naive::NaiveGather};
    use crate::model::Geometry;

    fn geom() -> Geometry {
        Geometry::new(400, 100, 1_000)
    }

    #[test]
    fn gsplat_holds_every_invariant() {
        let g = geom();
        for seed in 0..4 {
            let map = Map::shuffled(400, seed);
            for budget in [1_000, 4_000, 40_000] {
                let (_, trace) = Gsplat::new().run(g, &map, budget);
                let v = check_all(&trace, &map, budget);
                assert!(v.is_empty(), "seed {seed} budget {budget}: {v:?}");
            }
        }
    }

    #[test]
    fn the_checks_can_actually_fail() {
        // The point of a checker is that it detects the thing. Naive
        // gather reads in map order, so a shuffled map makes it step
        // backwards — the invariant gsplat exists to establish.
        let g = geom();
        let map = Map::shuffled(400, 5);
        let (_, trace) = NaiveGather.run(g, &map, 0);
        let v = monotone_access(&trace);
        assert!(
            !v.is_empty(),
            "scattered reads must trip the monotone check"
        );
        assert_eq!(v[0].invariant, "monotone access");
    }

    #[test]
    fn naive_gather_still_reads_each_record_once() {
        // It violates ordering, not economy: the read count is the same,
        // which is the point the cost model makes about operation counts.
        let g = geom();
        let map = Map::shuffled(400, 6);
        let (_, trace) = NaiveGather.run(g, &map, 0);
        assert!(single_read(&trace, &map).is_empty());
    }

    #[test]
    fn gsplat_is_deterministic() {
        let g = geom();
        let map = Map::shuffled(400, 9);
        assert!(deterministic(&Gsplat::new(), g, &map, 4_000).is_empty());
    }
}
