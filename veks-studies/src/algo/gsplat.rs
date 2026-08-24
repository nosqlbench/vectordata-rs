// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! gsplat itself: Segment, Plan, Linearize, Assemble, Transfer.
//!
//! The step boundaries here are the ones the documents describe, kept
//! deliberately literal so that a change in the specification and a
//! change in this file are the same size. Sizing follows
//! `docs/gsplat/01-segment.md`; the window shortcut in Plan follows
//! `02-plan.md`; the sort key in Linearize follows `03-linearize.md`.

use super::Rewrite;
use crate::model::{Geometry, Map, Op, Sink, Trace};

/// The published algorithm, parameterized only by whether the
/// already-sorted fast path is allowed.
pub struct Gsplat {
    /// When set, a segment whose plan is already ascending skips the
    /// sort and the transpose, streaming instead — the fvec fast path.
    pub detect_sorted: bool,
}

impl Default for Gsplat {
    fn default() -> Self {
        Gsplat { detect_sorted: true }
    }
}

impl Gsplat {
    pub fn new() -> Self {
        Self::default()
    }

    /// Without the fast path, so every segment pays the full transpose.
    pub fn always_transpose() -> Self {
        Gsplat { detect_sorted: false }
    }
}

impl Rewrite for Gsplat {
    fn name(&self) -> &'static str {
        "gsplat"
    }

    fn run(&self, geometry: Geometry, map: &Map, budget_bytes: u64) -> (Sink, Trace) {
        let output_count = map.len();
        let mut sink = Sink::new(output_count);
        let mut trace = Trace::new(geometry);

        // ── S · Segment ──────────────────────────────────────────────
        // Divide the OUTPUT ordinal space, never the input. The floor of
        // two segments keeps the buffer at or below half the output.
        let passes = geometry.passes(output_count, budget_bytes);
        let segment_size = output_count.div_ceil(passes);

        // Resident state: the segment buffer, plus the plan for the pass.
        // Both are claimed as virtual bytes, not allocated.
        let buffer_bytes = segment_size * geometry.record_bytes;
        let plan_bytes = segment_size * PLAN_ENTRY_BYTES;
        trace.claim_resident(buffer_bytes + plan_bytes);

        let mut plan: Vec<(u64, u64)> = Vec::with_capacity(segment_size as usize);

        for pass in 0..passes {
            let start = pass * segment_size;
            if start >= output_count {
                // A floor-of-two segmentation can leave the last pass
                // empty when the output is tiny; it still counts as a
                // pass, and it does nothing.
                trace.push(Op::PassStart { pass });
                continue;
            }
            let end = (start + segment_size).min(output_count);
            let len = end - start;
            trace.push(Op::PassStart { pass });

            // ── P · Plan ─────────────────────────────────────────────
            // The map is destination-ordered, so this segment's entries
            // are the contiguous window at those positions: a seek, not
            // a filter. Each entry is reversed into (source, local).
            trace.push(Op::ReadMap { from: start, count: len });
            plan.clear();
            for slot in start..end {
                let source = map.0[slot as usize];
                assert!(
                    source < geometry.records,
                    "map entry {source} at slot {slot} exceeds the source"
                );
                plan.push((source, slot - start));
            }

            // ── L · Linearize ────────────────────────────────────────
            // Sort by source ordinal, which under the family's
            // monotonicity premise is sorting by address.
            let already_sorted = plan.windows(2).all(|w| w[0].0 <= w[1].0);
            let streaming = self.detect_sorted && already_sorted;
            if !already_sorted {
                plan.sort_unstable_by_key(|(source, _)| *source);
            }

            // ── A · Assemble ─────────────────────────────────────────
            // Read ascending; scatter into the buffer at the local slot.
            // The scatter is the transpose, and it happens in memory.
            for &(source, local) in &plan {
                trace.push(Op::ReadRecord { ordinal: source });
                if !streaming {
                    trace.push(Op::Scatter { local });
                }
                sink.slots[(start + local) as usize] = Some(source);
            }

            // ── T · Transfer ─────────────────────────────────────────
            // One contiguous range at the segment's position, then a
            // barrier so this pass's writeback does not land inside the
            // next pass's reads.
            trace.push(Op::WriteRange { first_slot: start, records: len });
            trace.push(Op::Barrier);
        }

        (sink, trace)
    }
}

/// Bytes per plan entry: a source ordinal and a segment-local slot.
/// `02-plan.md` quotes 16 bytes per entry.
const PLAN_ENTRY_BYTES: u64 = 16;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::run_verified;

    fn geom(records: u64) -> Geometry {
        Geometry::new(records, 100, 1_000)
    }

    #[test]
    fn segment_sizing_follows_the_documented_formula() {
        let g = geom(1_000);
        // 100-byte records, 10_000-byte budget → 100 records per segment
        // → 10 passes.
        let (_, trace) = Gsplat::new().run(g, &Map::shuffled(1_000, 1), 10_000);
        assert_eq!(trace.metrics().passes, 10);
        // Halving the budget doubles the passes.
        let (_, trace) = Gsplat::new().run(g, &Map::shuffled(1_000, 1), 5_000);
        assert_eq!(trace.metrics().passes, 20);
    }

    #[test]
    fn the_identity_map_takes_the_streaming_fast_path() {
        let g = geom(500);
        let map = Map::identity(500);
        let (_, fast) = Gsplat::new().run(g, &map, 10_000);
        let (_, slow) = Gsplat::always_transpose().run(g, &map, 10_000);
        assert_eq!(fast.metrics().scatters, 0, "sorted plans skip the transpose");
        assert_eq!(slow.metrics().scatters, 500);
        // Both are still correct.
        assert!(run_verified(&Gsplat::new(), g, &map, 10_000).is_ok());
    }

    #[test]
    fn a_rotation_is_sorted_within_most_segments() {
        // A rotation is ascending except where it wraps, so most
        // segments take the fast path and one does not.
        let g = geom(1_000);
        let map = Map::rotated(1_000, 250);
        let (_, trace) = Gsplat::new().run(g, &map, 10_000);
        let m = trace.metrics();
        assert!(m.scatters > 0, "the wrapping segment transposes");
        assert!(
            m.scatters < 1_000,
            "but most segments stream: {} scatters",
            m.scatters
        );
    }

    #[test]
    fn every_map_shape_produces_the_output_the_map_calls_for() {
        let g = geom(600);
        for map in [
            Map::identity(600),
            Map::shuffled(600, 11),
            Map::rotated(600, 97),
            Map::reversed(600),
        ] {
            for budget in [1_000, 6_000, 60_000, 600_000] {
                run_verified(&Gsplat::new(), g, &map, budget)
                    .unwrap_or_else(|e| panic!("budget {budget}: {e}"));
            }
        }
    }
}
