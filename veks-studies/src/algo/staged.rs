// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! gsplat with **spill**: the form that holds when memory is a small
//! fraction of the payload.
//!
//! [`crate::algo::gsplat::Gsplat`] orders its reads but keeps no scratch,
//! so it sweeps the source once for every destination segment. That is
//! fine while the segment count is small and ruinous when it is not: at a
//! terabyte with 32 GiB of memory it is thirty-odd sweeps, and the source
//! is read thirty times over. The ordering was never the problem — the
//! **re-reading** was.
//!
//! Staging removes it. One sweep of the source routes each record into
//! the bucket its destination falls in; each open bucket owns a
//! container-sized write buffer, so its spill writes are large and
//! sequential. A second sweep loads one bucket at a time — a bucket is
//! sized to fit in memory by construction — permutes it there, and emits
//! it as one contiguous range. Nothing is read twice, nothing is written
//! randomly, and neither side ever seeks.
//!
//! ```text
//!   source ─ascending─▶ [ bucket buffers ] ─sequential─▶ spill
//!   spill  ─sequential─▶ [ segment in RAM ] ─contiguous─▶ output
//! ```
//!
//! # Fan-out, and why one stage is nearly always enough
//!
//! A bucket is only worth having if its writes are big enough not to
//! seek, so each one costs a container-sized buffer and memory divided by
//! that is the **fan-out** `f`: how many buckets a single sweep can keep
//! open. With a 32 GiB budget and 128 KiB containers, `f` is a quarter of
//! a million. A terabyte of kilobyte records needs 32 destination
//! segments, a petabyte needs 32,768 — both under `f`, both one stage.
//!
//! When the segment count does exceed `f`, the split recurses: a coarse
//! distribution into `f` groups, then each group distributed again, for
//! `ceil(log_f(segments))` stages. Growth is logarithmic in how badly
//! memory is oversubscribed, which is why the approach does not have a
//! scale at which it stops working.
//!
//! This is the external distribution sort of Aggarwal & Vitter's I/O
//! model (*Communications of the ACM* 31(9), 1988), whose bound for a
//! permutation is `Θ((N/B)·log_{M/B}(N/B))` I/Os — the `log_f` here is
//! that logarithm, and `f` is their `M/B`.

use super::Rewrite;
use crate::model::{Geometry, Map, Op, Sink, Trace};

/// The staged rewrite, parameterized by how much of the budget goes to
/// bucket buffers rather than to the final segment.
#[derive(Debug, Clone, Copy, Default)]
pub struct StagedSplat {
    /// Bytes per open bucket buffer, or `None` for one container.
    ///
    /// A container is the natural default: it is the unit the tier emits
    /// whole, so a buffer any smaller buys no extra sequentiality while a
    /// larger one costs fan-out for nothing. Overriding it is how the
    /// studies show the trade between spill-write size and stage count.
    pub bucket_buffer_bytes: Option<u64>,
}

impl StagedSplat {
    pub fn new() -> Self {
        Self::default()
    }

    /// With an explicit bucket buffer size instead of one container.
    pub fn with_buffer(bucket_buffer_bytes: u64) -> Self {
        StagedSplat {
            bucket_buffer_bytes: Some(bucket_buffer_bytes.max(1)),
        }
    }

    /// Bytes each open bucket buffers before it flushes.
    pub fn buffer_bytes(&self, geometry: Geometry) -> u64 {
        self.bucket_buffer_bytes
            .unwrap_or(geometry.container_bytes)
            .max(geometry.record_bytes)
    }

    /// How many buckets one stage can keep open on `budget_bytes`.
    pub fn fanout(&self, geometry: Geometry, budget_bytes: u64) -> u64 {
        (budget_bytes / self.buffer_bytes(geometry)).max(2)
    }
}

/// A run of records being carried through the stages together: the
/// destination range it covers, and the `(source, destination)` pairs
/// currently living in it.
struct Run {
    /// First destination slot this run covers.
    first_slot: u64,
    /// Destination slots covered.
    slots: u64,
    /// `(source ordinal, destination slot)`, in the order the run holds
    /// them — source-ascending after the first stage.
    entries: Vec<(u64, u64)>,
}

impl Rewrite for StagedSplat {
    fn name(&self) -> &'static str {
        "gsplat-staged"
    }

    fn run(&self, geometry: Geometry, map: &Map, budget_bytes: u64) -> (Sink, Trace) {
        let output_count = map.len();
        let mut sink = Sink::new(output_count);
        let mut trace = Trace::new(geometry);

        // ── S · Segment ──────────────────────────────────────────────
        // A segment is what memory can hold and permute at once; the
        // number of them is the only thing the stage count depends on.
        let per_segment = geometry.records_per_segment(budget_bytes);
        let buffer_bytes = self.buffer_bytes(geometry);
        let fanout = self.fanout(geometry, budget_bytes);

        // Groups start at the whole destination space and are divided by
        // `fanout` each stage until a group is one segment wide.
        let mut group_slots = output_count.max(1);
        let mut stages = 0u64;
        while group_slots > per_segment {
            group_slots = group_slots.div_ceil(fanout);
            stages += 1;
        }

        // ── P · Plan ─────────────────────────────────────────────────
        // The map is destination-ordered, so one sequential read of it
        // gives every (source, destination) pair. Unlike the re-scan
        // form, the map is read once for the whole rewrite rather than
        // once per segment.
        trace.push(Op::PassStart { pass: 0 });
        trace.push(Op::ReadMap {
            from: 0,
            count: output_count,
        });

        // ── L · Linearize ────────────────────────────────────────────
        // Order the pairs by source ordinal, which under the
        // monotonicity premise is ordering by address: the first stage's
        // reads then ascend through the source exactly once.
        //
        // No table is resident for this. The premise the whole family
        // rests on is that the map is a **closed transform** — the
        // destination of a source ordinal is computed, not looked up —
        // so a sweep in source order needs nothing but a counter. The
        // vector below is the simulation materializing a function it is
        // cheaper to store than to re-evaluate; it is not state the
        // algorithm claims, and the resident budget below does not
        // include it.
        let mut entries: Vec<(u64, u64)> = (0..output_count)
            .map(|slot| (map.0[slot as usize], slot))
            .collect();
        entries.sort_unstable_by_key(|(source, _)| *source);

        // Resident: the fan-out buffers during distribution, one segment
        // during transfer. Neither scales with the size of the store.
        trace.claim_resident(per_segment.min(output_count).max(1) * geometry.record_bytes);

        let mut runs = vec![Run {
            first_slot: 0,
            slots: output_count,
            entries,
        }];

        // ── A · Assemble, stage by stage ─────────────────────────────
        // Every distribution stage reads its input straight through and
        // appends to bucket buffers. The very first stage's input is the
        // source itself; later stages read spill written by the stage
        // before.
        for stage in 0..stages {
            let width = {
                let mut w = output_count.max(1);
                for _ in 0..=stage {
                    w = w.div_ceil(fanout);
                }
                w.max(per_segment.min(output_count).max(1))
            };
            trace.push(Op::PassStart { pass: stage });
            trace.claim_resident(fanout.min(output_count.max(1)) * buffer_bytes);

            let mut next: Vec<Run> = Vec::new();
            let mut bucket_id = 0u64;
            for run in runs {
                if stage == 0 {
                    // The source, ascending. This is the sweep the whole
                    // design exists to make possible.
                    for &(source, _) in &run.entries {
                        trace.push(Op::ReadRecord { ordinal: source });
                    }
                } else {
                    // A spill stream, read straight through.
                    trace.push(Op::SpillRead {
                        bucket: run.first_slot / width.max(1),
                        records: run.entries.len() as u64,
                    });
                }

                // Route into buckets. Each bucket keeps its records in
                // arrival order, which is source-ascending, so a later
                // stage still reads its input in address order.
                let buckets = run.slots.div_ceil(width).max(1);
                let mut parts: Vec<Vec<(u64, u64)>> = vec![Vec::new(); buckets as usize];
                for (source, slot) in run.entries {
                    let b = ((slot - run.first_slot) / width).min(buckets - 1);
                    parts[b as usize].push((source, slot));
                }

                for (b, entries) in parts.into_iter().enumerate() {
                    let first_slot = run.first_slot + b as u64 * width;
                    if entries.is_empty() {
                        bucket_id += 1;
                        continue;
                    }
                    // The buffer is flushed whole, so the spill is a
                    // sequence of container-sized appends and the tail.
                    emit_spill_runs(
                        &mut trace,
                        bucket_id,
                        entries.len() as u64,
                        buffer_bytes / geometry.record_bytes.max(1),
                    );
                    next.push(Run {
                        first_slot,
                        slots: width.min(run.first_slot + run.slots - first_slot),
                        entries,
                    });
                    bucket_id += 1;
                }
            }
            runs = next;
        }

        // ── T · Transfer ─────────────────────────────────────────────
        // Each run is now at most one segment wide, so it is loaded,
        // permuted in memory, and emitted as one contiguous range.
        trace.push(Op::PassStart { pass: stages });
        trace.claim_resident(per_segment.min(output_count) * geometry.record_bytes);
        for (index, run) in runs.iter().enumerate() {
            if run.entries.is_empty() {
                continue;
            }
            if stages == 0 {
                // Everything fit; there was never anything to spill.
                for &(source, _) in &run.entries {
                    trace.push(Op::ReadRecord { ordinal: source });
                }
            } else {
                trace.push(Op::SpillRead {
                    bucket: index as u64,
                    records: run.entries.len() as u64,
                });
            }
            for &(source, slot) in &run.entries {
                trace.push(Op::Scatter {
                    local: slot - run.first_slot,
                });
                sink.slots[slot as usize] = Some(source);
            }
            let (first, last) = span_of(&run.entries);
            trace.push(Op::WriteRange {
                first_slot: first,
                records: last - first + 1,
            });
        }
        trace.push(Op::Barrier);

        (sink, trace)
    }
}

/// Append `records` to a bucket as whole-buffer flushes plus a tail, so
/// the spill stream is a sequence of large sequential writes rather than
/// one implausibly contiguous extent.
fn emit_spill_runs(trace: &mut Trace, bucket: u64, records: u64, per_flush: u64) {
    let per_flush = per_flush.max(1);
    let mut remaining = records;
    while remaining > 0 {
        let run = remaining.min(per_flush);
        trace.push(Op::SpillWrite {
            bucket,
            records: run,
        });
        remaining -= run;
    }
}

/// The destination range a run covers, which is contiguous because the
/// distribution splits the destination space by position.
fn span_of(entries: &[(u64, u64)]) -> (u64, u64) {
    let mut first = u64::MAX;
    let mut last = 0;
    for &(_, slot) in entries {
        first = first.min(slot);
        last = last.max(slot);
    }
    (first, last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::{Rewrite, gsplat::Gsplat, run_verified};

    fn geom(records: u64) -> Geometry {
        Geometry::new(records, 100, 1_000)
    }

    /// Correctness first: whatever the staging does, the output has to be
    /// what the map called for.
    #[test]
    fn every_map_shape_and_budget_produces_the_mapped_output() {
        let g = geom(600);
        for map in [
            Map::identity(600),
            Map::shuffled(600, 11),
            Map::rotated(600, 97),
            Map::reversed(600),
        ] {
            for budget in [1_000u64, 6_000, 60_000, 600_000] {
                for buffer in [200u64, 1_000, 10_000] {
                    let algo = StagedSplat::with_buffer(buffer);
                    run_verified(&algo, g, &map, budget)
                        .unwrap_or_else(|e| panic!("budget {budget} buffer {buffer}: {e}"));
                }
            }
        }
    }

    /// **The property that matters.** The source is read exactly once, no
    /// matter how many segments the budget forces — which is the whole
    /// difference from the re-scan form.
    #[test]
    fn the_source_is_read_exactly_once_at_any_budget() {
        let g = geom(10_000);
        let map = Map::shuffled(10_000, 0xA11CE);
        for budget in [10_000u64, 50_000, 200_000, 1_000_000] {
            let m = StagedSplat::new().run(g, &map, budget).1.metrics();
            assert_eq!(
                m.record_reads, 10_000,
                "budget {budget}: the source is swept once, not once per segment"
            );
            assert_eq!(m.backward_steps, 0, "and the sweep never goes backwards");
        }
    }

    /// Against the re-scan form at the same budget: identical ordering
    /// discipline, but the re-scan reads the source once per segment and
    /// the staged one does not.
    #[test]
    fn staging_removes_the_re_read_the_rescan_form_pays() {
        let g = geom(10_000);
        let map = Map::shuffled(10_000, 7);
        let budget = 100_000; // 1000 records → 10 segments
        let rescan = Gsplat::new().run(g, &map, budget).1.metrics();
        let staged = StagedSplat::new().run(g, &map, budget).1.metrics();

        assert_eq!(rescan.passes, 10);
        // Both request the same records — the difference is not in what
        // they ask for but in how many times the tier has to visit a
        // container to serve it. The re-scan revisits every container
        // once per segment; the staged sweep visits each one once.
        assert_eq!(rescan.record_reads, staged.record_reads);
        assert!(
            rescan.container_touches > staged.container_touches * 5,
            "the re-scan touches {} containers against {}",
            rescan.container_touches,
            staged.container_touches
        );
        assert_eq!(
            staged.container_touches,
            g.container_count(),
            "one visit per container is the floor, and staging reaches it"
        );
        assert!(
            rescan.amplification() > 5.0 * staged.amplification(),
            "which is amplification {:.1}x against {:.1}x",
            rescan.amplification(),
            staged.amplification()
        );
    }

    /// The spill is what pays for it: the payload crosses the scratch
    /// extent once each way, and it carries a destination ordinal with it.
    #[test]
    fn the_price_of_staging_is_one_round_trip_through_spill() {
        let g = geom(10_000);
        let map = Map::shuffled(10_000, 3);
        let m = StagedSplat::new().run(g, &map, 100_000).1.metrics();
        assert_eq!(m.records_spilled, 10_000);
        assert_eq!(m.records_unspilled, 10_000);
        assert_eq!(
            m.spill_bytes(),
            2 * 10_000 * (100 + crate::model::trace::SPILL_TAG_BYTES)
        );
    }

    /// When everything fits, there is nothing to stage and nothing is
    /// spilled — the algorithm degenerates to a single ordered pass.
    #[test]
    fn a_budget_that_holds_everything_spills_nothing() {
        let g = geom(500);
        let m = StagedSplat::new()
            .run(g, &Map::shuffled(500, 5), 500 * 100)
            .1
            .metrics();
        assert_eq!(m.records_spilled, 0);
        assert_eq!(m.record_reads, 500);
        assert_eq!(m.write_ranges, 1, "one contiguous output range");
    }

    /// **Stage count is logarithmic in the segment count.** Squeezing the
    /// bucket buffers reduces the fan-out and eventually forces a second
    /// stage, but it takes a large squeeze to do it.
    #[test]
    fn stage_count_grows_logarithmically_with_the_fan_out_squeezed() {
        let g = geom(100_000);
        let map = Map::shuffled(100_000, 0xF00);
        let budget = 100_000; // 1000 records per segment → 100 segments

        // A generous buffer: fan-out 10, so 100 segments needs two stages.
        let wide = StagedSplat::with_buffer(10_000)
            .run(g, &map, budget)
            .1
            .metrics();
        // A tight buffer: fan-out 1000, one stage covers all 100.
        let narrow = StagedSplat::with_buffer(100)
            .run(g, &map, budget)
            .1
            .metrics();

        assert!(
            wide.records_spilled > narrow.records_spilled,
            "more stages spill the payload more times: {} against {}",
            wide.records_spilled,
            narrow.records_spilled
        );
        assert_eq!(
            narrow.records_spilled, 100_000,
            "one stage spills the payload exactly once"
        );
        assert_eq!(
            wide.records_spilled, 200_000,
            "two stages spill it twice — the logarithm, made of whole passes"
        );
    }

    /// Writes stay contiguous and cover the output exactly once, which is
    /// the invariant the checker enforces for every rewrite here.
    #[test]
    fn the_output_is_written_once_in_contiguous_ranges() {
        let g = geom(5_000);
        let map = Map::shuffled(5_000, 9);
        let trace = StagedSplat::new().run(g, &map, 50_000).1;
        let m = trace.metrics();
        assert_eq!(m.records_written, 5_000);
        assert!(
            m.write_ranges <= 5_000 / 500 + 1,
            "one range per segment, not one per record: {}",
            m.write_ranges
        );
        assert!(crate::check::single_write(&trace, &map).is_empty());
    }
}
