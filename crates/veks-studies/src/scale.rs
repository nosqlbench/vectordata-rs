// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The problem gsplat actually solves, at the scale it solves it.
//!
//! Everything else in this crate runs at 20,000 to 200,000 records —
//! a payload of tens of megabytes, which finishes in seconds by any
//! method and never presses against a ceiling. That is the regime where
//! gsplat does not matter yet, and measuring only there answers the wrong
//! question.
//!
//! The problem is a **terabyte-scale permutation**: a very large body of
//! records carrying a monotonic ordinal, a closed transform map over
//! those ordinals, memory enough for only a fraction of the data, and a
//! storage tier on which random access is ruinous. At a billion ordinals
//! the interesting constraints are not the ones the small runs surface:
//!
//! - **Command count**, not seek time. A gather issues one device request
//!   per record because nothing lets them coalesce. gsplat issues the
//!   same number of *record reads* but ascending, so the tier merges them
//!   into container-sized fetches — `N` commands become `N/w`. Where the
//!   controller's command rate is the ceiling, that ratio is the whole
//!   speedup, and amplification never enters it.
//! - **Amplification on both sides.** A scatter writes randomly, and a
//!   random write to a partial block is a read-modify-write: the block is
//!   fetched, altered, and written back, so the bytes move twice and a
//!   read appears on the write path.
//! - **Readahead working against you.** On a scattered stream it either
//!   disengages, buying nothing, or misfires and spends bandwidth on
//!   pages nobody wanted.
//!
//! A trace of a billion operations is 24 GB, so this module prices
//! strategies analytically instead — and every cost function here is
//! checked against the discrete-event simulator at a scale where both can
//! run, so the extrapolation rests on something.
//!
//! What each study reports is not only a time but **which ceiling
//! produced it**. A configuration where gsplat wins because commands ran
//! out is a different claim from one where it wins because bytes did, and
//! collapsing them into a speedup ratio hides the thing worth knowing.

use crate::device::DeviceModel;
use crate::io::hw::HostModel;
use crate::queueing::{Demand, Resource};

/// The rewrite to be performed.
#[derive(Debug, Clone, Copy)]
pub struct Workload {
    /// Ordinals to rewrite. The real cases are 10^8 and up.
    pub records: u64,
    pub record_bytes: u64,
    /// Memory available to hold reordered output before it is flushed.
    pub budget_bytes: u64,
    /// The unit the tier fetches and emits whole when access is ordered.
    pub container_bytes: u64,
    /// Minimum addressable transfer — page or sector.
    pub block_bytes: u64,
    /// Requests kept outstanding.
    pub depth: f64,
}

impl Workload {
    pub fn payload_bytes(&self) -> u64 {
        self.records.saturating_mul(self.record_bytes)
    }

    /// Records per container.
    pub fn w(&self) -> u64 {
        (self.container_bytes / self.record_bytes).max(1)
    }

    /// RAM-sized segments the destination has to be cut into. This is the
    /// quantity the whole problem turns on: it is how many times the
    /// payload exceeds what memory can hold at once.
    pub fn segments(&self) -> u64 {
        let per_segment = (self.budget_bytes / self.record_bytes).max(1);
        self.records.div_ceil(per_segment).max(1)
    }

    /// Passes, floored at two as the re-scan model floors it.
    pub fn passes(&self) -> u64 {
        self.segments().max(2)
    }

    /// **Fan-out**: how many destination buckets can be kept open at once.
    ///
    /// A bucket is only useful if its spill writes are large enough to be
    /// sequential, so each open bucket costs a container-sized write
    /// buffer, and memory divided by that is how many can run together.
    /// This is the number that decides whether the rewrite needs one
    /// distribution stage or several.
    pub fn fanout(&self) -> u64 {
        (self.budget_bytes / self.container_bytes).max(2)
    }

    /// Distribution stages needed to cut the payload down to RAM-sized
    /// pieces: `ceil(log_f(segments))`.
    ///
    /// One stage handles `f` buckets, two handle `f²`, and so on. With a
    /// 32 GiB budget and 128 KiB containers `f` is a quarter of a million,
    /// so a terabyte needs exactly one — **the staged rewrite is a
    /// constant number of sequential passes, and the constant is two.**
    pub fn stages(&self) -> u64 {
        let segments = self.segments();
        if segments <= 1 {
            return 0;
        }
        let f = self.fanout() as f64;
        ((segments as f64).ln() / f.ln()).ceil().max(1.0) as u64
    }

    /// Read amplification for an ordered rewrite.
    pub fn amplification(&self) -> f64 {
        let p = self.passes() as f64;
        p * (1.0 - (-(self.w() as f64) / p).exp())
    }

    /// Bytes a single random record read actually moves, after the tier
    /// rounds it up to whole blocks.
    fn random_read_bytes(&self) -> u64 {
        self.record_bytes.div_ceil(self.block_bytes) * self.block_bytes
    }
}

/// How the rewrite is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Walk the output in order, read each record where it lies. Random
    /// reads, sequential writes.
    Gather,
    /// Walk the source in order, write each record where it belongs.
    /// Sequential reads, random writes — and a random write to a partial
    /// block costs a read-modify-write.
    Scatter,
    /// Order the reads but keep no spill: sweep the source once per
    /// destination segment and take what belongs to that segment.
    ///
    /// Neither side is random, so commands coalesce — but the source is
    /// read `A(P)` times over, and `A(P)` grows with the segment count.
    /// This is ordering without staging, and it is the arm that loses
    /// once memory is a small fraction of the payload.
    OrderedScan,
    /// Segment, plan, linearize, assemble, transfer — **with spill**.
    ///
    /// One sequential pass distributes records into RAM-sized buckets,
    /// each buffered a container at a time so its writes are sequential
    /// too; a second pass loads each bucket, permutes it in memory, and
    /// writes it out. Neither side is ever random and neither side is
    /// ever re-read, so the cost is a fixed small number of sequential
    /// passes no matter how far the payload exceeds memory.
    Splat,
}

impl Strategy {
    pub fn label(self) -> &'static str {
        match self {
            Strategy::Gather => "naive gather",
            Strategy::Scatter => "naive scatter",
            Strategy::OrderedScan => "ordered rescan",
            Strategy::Splat => "gsplat staged",
        }
    }

    /// Whether the strategy's device access is ascending, so the tier can
    /// coalesce it and the streaming bound applies.
    pub fn is_ordered(self) -> bool {
        matches!(self, Strategy::OrderedScan | Strategy::Splat)
    }

    pub const ALL: [Strategy; 4] = [
        Strategy::Gather,
        Strategy::Scatter,
        Strategy::OrderedScan,
        Strategy::Splat,
    ];
}

/// What one strategy asks of the machine, per record and in total.
#[derive(Debug, Clone, Copy)]
pub struct Cost {
    pub strategy: Strategy,
    pub passes: u64,
    pub read_commands: u64,
    pub write_commands: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// Seconds of each resource per record transformed.
    pub demand: Demand,
    pub seconds: f64,
    /// Transform rate achieved at the workload's concurrency.
    pub records_per_second: f64,
}

impl Cost {
    pub fn commands(&self) -> u64 {
        self.read_commands + self.write_commands
    }

    /// Resources at or near full utilization. More than one can be, and
    /// when more than one is, no amount of tuning will help.
    pub fn pegged(&self) -> Vec<Resource> {
        self.demand.pegged(0.05)
    }
}

/// Turn a strategy's request profile into per-record service demands.
///
/// This is where the mechanism lives. A gather places one command per
/// record on the controller; an ordered pass places one per container.
/// The bytes go the other way. Everything the studies show follows from
/// those two facts meeting a device's ceilings.
#[allow(clippy::too_many_arguments)]
fn demands(
    workload: &Workload,
    device: &DeviceModel,
    host: &HostModel,
    read_commands: u64,
    bytes_read: u64,
    write_commands: u64,
    bytes_written: u64,
    ordered: bool,
    passes: u64,
) -> Demand {
    let records = workload.records.max(1) as f64;
    let commands = (read_commands + write_commands) as f64;
    let bytes = (bytes_read + bytes_written) as f64;

    // Positioning is what a request pays before any byte moves. A
    // scattered reader pays it once per command; an ordered one does not,
    // because consecutive fetches are contiguous — it pays once per pass
    // and streams the rest. That is the streaming bound, and leaving it
    // out charges an ordered terabyte rewrite a seek per container and
    // turns two days into fourteen.
    let scattered_positioning =
        commands / records * device.latency_s / device.native_queue_depth.max(1.0);
    let streamed_positioning = passes.max(1) as f64 / records * device.latency_s;

    Demand {
        controller: commands / records / device.iops_ceiling,
        bandwidth: bytes / records / device.sequential_bandwidth(),
        host_cpu: commands / records / host.ceiling_iops(),
        media: if ordered {
            scattered_positioning.min(streamed_positioning)
        } else {
            scattered_positioning
        },
    }
}

/// Price one strategy.
pub fn cost(
    strategy: Strategy,
    workload: &Workload,
    device: &DeviceModel,
    host: &HostModel,
) -> Cost {
    let payload = workload.payload_bytes();
    let container = workload.container_bytes;

    // Each arm is the strategy's request profile: how many commands land
    // on the device, how many bytes cross, and how big a typical request
    // is. Everything downstream is those numbers meeting the ceilings.
    let (read_commands, bytes_read, write_commands, bytes_written, passes) = match strategy {
        Strategy::Gather => (
            // One command per record: scattered reads cannot merge.
            workload.records,
            workload.records * workload.random_read_bytes(),
            // The output is walked in order, so writes do merge.
            payload.div_ceil(container),
            payload,
            1,
        ),
        Strategy::Scatter => {
            // Reads stream. Writes are scattered, and a partial-block
            // write must fetch its block first — the bytes cross twice
            // and a read lands on the write path.
            let partial = workload.record_bytes < workload.block_bytes;
            let touched = workload.records * workload.random_read_bytes();
            (
                payload.div_ceil(container) + if partial { workload.records } else { 0 },
                payload + if partial { touched } else { 0 },
                workload.records,
                touched,
                1,
            )
        }
        Strategy::OrderedScan => {
            // Each sweep ascends, so the tier coalesces record reads
            // into container fetches. A sweep costs whichever is
            // cheaper: fetching the containers it needs, or streaming
            // the extent and discarding the rest.
            let p = workload.passes();
            let containers = payload.div_ceil(container);
            let touched = (workload.amplification() * containers as f64) as u64;
            let commands = touched.min(p * containers).max(containers);
            (commands, commands * container, containers, payload, p)
        }
        Strategy::Splat => {
            // Distribution stages plus the final permute-and-emit
            // stage. Every one of them reads the payload once
            // sequentially and writes it once sequentially, at
            // container granularity on both sides — nothing is ever
            // re-read, so the segment count enters only through the
            // stage count, and that is logarithmic in it.
            let sweeps = workload.stages() + 1;
            let containers = payload.div_ceil(container);
            (
                sweeps * containers,
                sweeps * payload,
                sweeps * containers,
                sweeps * payload,
                sweeps,
            )
        }
    };

    let demand = demands(
        workload,
        device,
        host,
        read_commands,
        bytes_read,
        write_commands,
        bytes_written,
        strategy.is_ordered(),
        passes,
    );

    Cost {
        strategy,
        passes,
        read_commands,
        write_commands,
        bytes_read,
        bytes_written,
        demand,
        seconds: demand.completion_seconds(workload.records, workload.depth),
        records_per_second: demand.transform_rate_at(workload.depth),
    }
}

/// Price every strategy for one workload.
pub fn compare(workload: &Workload, device: &DeviceModel, host: &HostModel) -> [Cost; 4] {
    Strategy::ALL.map(|s| cost(s, workload, device, host))
}

/// The cost of one named strategy out of a comparison.
pub fn pick(costs: &[Cost; 4], strategy: Strategy) -> &Cost {
    costs
        .iter()
        .find(|c| c.strategy == strategy)
        .expect("every strategy is priced")
}

/// The best of the two naive strategies.
pub fn best_naive(costs: &[Cost; 4]) -> &Cost {
    let gather = pick(costs, Strategy::Gather);
    let scatter = pick(costs, Strategy::Scatter);
    if gather.seconds <= scatter.seconds {
        gather
    } else {
        scatter
    }
}

/// How much better the staged rewrite is than the best naive strategy.
pub fn advantage(workload: &Workload, device: &DeviceModel, host: &HostModel) -> f64 {
    let costs = compare(workload, device, host);
    best_naive(&costs).seconds / pick(&costs, Strategy::Splat).seconds.max(1e-12)
}

pub fn human_time(seconds: f64) -> String {
    if seconds >= 86_400.0 {
        format!("{:.1}d", seconds / 86_400.0)
    } else if seconds >= 3_600.0 {
        format!("{:.1}h", seconds / 3_600.0)
    } else if seconds >= 60.0 {
        format!("{:.1}m", seconds / 60.0)
    } else {
        format!("{seconds:.1}s")
    }
}

pub fn human_count(n: u64) -> String {
    let v = n as f64;
    if v >= 1e9 {
        format!("{:.1}B", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.1}k", v / 1e3)
    } else {
        format!("{n}")
    }
}

pub fn human_bytes(n: u64) -> String {
    let v = n as f64;
    if v >= (1u64 << 40) as f64 {
        format!("{:.2}T", v / (1u64 << 40) as f64)
    } else if v >= (1u64 << 30) as f64 {
        format!("{:.1}G", v / (1u64 << 30) as f64)
    } else if v >= (1u64 << 20) as f64 {
        format!("{:.0}M", v / (1u64 << 20) as f64)
    } else {
        // Container sizes live down here, and rounding 16 KiB to "0M"
        // makes the fan-out table unreadable.
        format!("{:.0}K", v / 1024.0)
    }
}

/// Render a comparison of all three strategies.
pub fn render_comparison(
    title: &str,
    workload: &Workload,
    device: &DeviceModel,
    host: &HostModel,
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n  {title}\n  {} records x {} B = {} payload, budget {}, {} on {}",
        human_count(workload.records),
        workload.record_bytes,
        human_bytes(workload.payload_bytes()),
        human_bytes(workload.budget_bytes),
        format_args!(
            "segments={} w={} fanout={} stages={}",
            workload.segments(),
            workload.w(),
            human_count(workload.fanout()),
            workload.stages()
        ),
        device.name
    );
    let _ = writeln!(
        s,
        "\n  {:<14} {:>9} {:>8} {:>8} {:>9} {:>10}   utilization",
        "strategy", "commands", "read", "written", "time", "records/s"
    );
    for c in compare(workload, device, host) {
        let mut utilization = String::new();
        for (resource, u) in c.demand.utilizations(c.records_per_second) {
            if u >= 0.05 {
                let _ = write!(utilization, "{} {:.0}%  ", resource.label(), u * 100.0);
            }
        }
        let _ = writeln!(
            s,
            "  {:<14} {:>9} {:>8} {:>8} {:>9} {:>10}   {}",
            c.strategy.label(),
            human_count(c.commands()),
            human_bytes(c.bytes_read),
            human_bytes(c.bytes_written),
            human_time(c.seconds),
            human_count(c.records_per_second as u64),
            utilization.trim_end()
        );
    }
    s
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;
    use crate::device::{ALL_MODELS, NVME_CONSUMER_MODEL, NVME_MODERN_MODEL, SPINNING_SATA_MODEL};

    fn terabyte(record_bytes: u64, budget_bytes: u64) -> Workload {
        Workload {
            records: (1u64 << 40) / record_bytes,
            record_bytes,
            budget_bytes,
            container_bytes: 128 * 1024,
            block_bytes: 4_096,
            depth: 32.0,
        }
    }

    /// **The problem, stated as a test.** A terabyte permutation of
    /// kilobyte records with 32 GiB of memory, on a seek-bound disk: the
    /// naive strategies take days and the staged one takes hours.
    #[test]
    fn a_terabyte_permutation_on_a_disk_takes_days_naively_and_hours_staged() {
        let w = terabyte(1_024, 32 << 30);
        let costs = compare(&w, &SPINNING_SATA_MODEL, &HostModel::DEFAULT);
        let naive = best_naive(&costs).seconds;
        let staged = pick(&costs, Strategy::Splat).seconds;

        assert!(
            naive > 2.0 * 86_400.0,
            "the naive strategies should take days, got {}",
            human_time(naive)
        );
        assert!(
            staged < 86_400.0,
            "the staged rewrite should come in under a day, got {}",
            human_time(staged)
        );
        assert!(naive / staged > 10.0);
    }

    /// **Why staging is the thing that makes it work.**
    ///
    /// Both ordered arms avoid random access, but only one of them avoids
    /// re-reading. The re-scan sweeps the source once per destination
    /// segment; the staged rewrite spills into buckets and touches the
    /// payload a fixed number of times. When memory is a small fraction of
    /// the payload — which is the stated problem — that difference is the
    /// difference between finishing and not.
    #[test]
    fn staging_is_what_separates_gsplat_from_merely_ordering_the_reads() {
        let w = terabyte(1_024, 8 << 30);
        assert!(
            w.segments() > 100,
            "memory is a small fraction of the payload"
        );

        let costs = compare(&w, &NVME_CONSUMER_MODEL, &HostModel::DEFAULT);
        let rescan = pick(&costs, Strategy::OrderedScan);
        let staged = pick(&costs, Strategy::Splat);

        assert!(
            rescan.bytes_read > 50 * w.payload_bytes(),
            "the re-scan reads the payload tens of times over: {}",
            human_bytes(rescan.bytes_read)
        );
        assert!(
            staged.bytes_read <= 3 * w.payload_bytes(),
            "the staged rewrite reads it a small fixed number of times: {}",
            human_bytes(staged.bytes_read)
        );
        assert!(
            staged.seconds < rescan.seconds / 10.0,
            "staged {} against re-scan {}",
            human_time(staged.seconds),
            human_time(rescan.seconds)
        );
    }

    /// **The fan-out result.** One distribution stage can cut the payload
    /// into as many buckets as memory has container-sized buffers. With a
    /// 32 GiB budget and 128 KiB containers that is a quarter of a million
    /// buckets, so a terabyte — and a petabyte — needs exactly one, and
    /// the staged rewrite is a constant four sequential passes over the
    /// data regardless of how far it exceeds memory.
    #[test]
    fn one_distribution_stage_covers_a_quarter_million_segments() {
        let w = terabyte(1_024, 32 << 30);
        assert!(
            w.fanout() > 200_000,
            "fan-out is memory over container size"
        );
        assert_eq!(w.stages(), 1, "a terabyte needs one distribution stage");

        let petabyte = Workload {
            records: (1u64 << 50) / 1_024,
            ..w
        };
        assert_eq!(
            petabyte.stages(),
            1,
            "and so does a petabyte: {} segments against a fan-out of {}",
            human_count(petabyte.segments()),
            human_count(petabyte.fanout())
        );

        // Two stages only when the segment count exceeds the fan-out
        // squared is wrong; it is when it exceeds the fan-out at all.
        let cramped = Workload {
            budget_bytes: 8 << 20,
            ..w
        };
        assert!(
            cramped.stages() >= 2,
            "a tiny budget forces more than one stage: fan-out {} against {} segments",
            human_count(cramped.fanout()),
            human_count(cramped.segments())
        );
    }

    /// Passes are logarithmic in the segment count, so completion is
    /// essentially flat in how badly memory is oversubscribed — the shape
    /// that makes the approach hold at any scale.
    #[test]
    fn staged_cost_is_logarithmic_in_how_far_memory_is_exceeded() {
        let mut previous = 0u64;
        for budget in [64u64 << 30, 8 << 30, 1 << 30, 128 << 20] {
            let w = terabyte(1_024, budget);
            let staged = cost(
                Strategy::Splat,
                &w,
                &NVME_CONSUMER_MODEL,
                &HostModel::DEFAULT,
            );
            assert!(
                staged.passes >= previous,
                "shrinking memory can only add passes"
            );
            assert!(
                staged.passes <= 4,
                "a 512x range of budgets should stay within four passes, got {} at {}",
                staged.passes,
                human_bytes(budget)
            );
            previous = staged.passes;
        }
    }

    /// **Where the advantage comes from at scale: command count.**
    ///
    /// A gather issues one device command per record because nothing lets
    /// them coalesce. An ordered pass issues one per container. At a
    /// billion ordinals that is the difference between a billion commands
    /// and a few tens of millions, and on a device whose controller is
    /// the ceiling it is the entire speedup.
    #[test]
    fn ordering_collapses_the_command_count() {
        let w = terabyte(1_024, 32 << 30);
        let costs = compare(&w, &NVME_CONSUMER_MODEL, &HostModel::DEFAULT);
        let gather = pick(&costs, Strategy::Gather);
        let staged = pick(&costs, Strategy::Splat);

        assert_eq!(gather.read_commands, w.records, "a gather cannot coalesce");
        // Every read is a whole container, so the reduction is exactly the
        // records per container divided by the number of passes.
        let reduction = gather.read_commands as f64 / staged.read_commands as f64;
        let expected = w.w() as f64 / staged.passes as f64;
        assert!(
            (reduction - expected).abs() / expected < 0.01,
            "the reduction is w/passes = {expected:.1}x, measured {reduction:.1}x"
        );
        assert!(
            reduction > 32.0,
            "at 128 records per container and two passes that is 64x, got {reduction:.1}x"
        );
        // A billion scattered reads peg the controller. Whether bytes peg
        // too is a separate question with its own answer — the point is
        // that the controller has no headroom left.
        assert!(
            gather.pegged().contains(&Resource::Controller),
            "a billion scattered reads should exhaust command processing; pegged: {:?}",
            gather
                .pegged()
                .iter()
                .map(|r| r.label())
                .collect::<Vec<_>>()
        );
    }

    /// The other side of the problem: a scatter's random writes to
    /// partial blocks cost a read-modify-write, so the bytes cross twice
    /// and a read appears on the write path.
    #[test]
    fn scattered_writes_to_partial_blocks_move_the_bytes_twice() {
        let small = terabyte(512, 32 << 30);
        let scatter = cost(
            Strategy::Scatter,
            &small,
            &NVME_CONSUMER_MODEL,
            &HostModel::DEFAULT,
        );
        assert!(
            scatter.bytes_written > small.payload_bytes() * 4,
            "a 512 B record in a 4 KiB block writes 8x its own size"
        );
        assert!(
            scatter.bytes_read > small.payload_bytes(),
            "and the write path has to read the blocks first"
        );

        // With records at or above the block size there is nothing to
        // read-modify-write, and the penalty disappears.
        let large = terabyte(8_192, 32 << 30);
        let clean = cost(
            Strategy::Scatter,
            &large,
            &NVME_CONSUMER_MODEL,
            &HostModel::DEFAULT,
        );
        assert_eq!(clean.bytes_read, large.payload_bytes());
    }

    /// **The necessity regime.** Small records, so the ordinal count is
    /// enormous and every random access is mostly padding: two billion
    /// ordinals of 512 B in a terabyte, with memory at 1.5% of it. Here
    /// staging is not an optimisation to be weighed — the naive
    /// strategies do not finish in a usable time on any device modelled.
    #[test]
    fn at_scale_the_naive_strategies_are_not_merely_slower() {
        let w = terabyte(512, 16 << 30);
        for device in ALL_MODELS {
            let costs = compare(&w, device, &HostModel::DEFAULT);
            let naive = best_naive(&costs).seconds;
            let staged = pick(&costs, Strategy::Splat).seconds;
            assert!(
                naive > 3_600.0,
                "{}: naive should take at least an hour, got {}",
                device.name,
                human_time(naive)
            );
            assert!(
                naive / staged > 3.0,
                "{}: staging should be worth at least 3x, got {:.1}x ({} against {})",
                device.name,
                naive / staged,
                human_time(naive),
                human_time(staged)
            );
        }
    }

    /// **And where it is not.** The honest boundary, stated as sharply as
    /// the win: when the record is the block, a random read fetches no
    /// padding and the ordinal count for a given payload is at its
    /// smallest. On flash, whose command rate is high enough to absorb
    /// what is left, the naive gather is then *competitive with or better
    /// than* the staged rewrite — because staging still has to move the
    /// payload through scratch, and a gather that wastes nothing has
    /// nothing left to save.
    #[test]
    fn staging_loses_when_the_record_is_the_block_and_the_device_is_fast() {
        let w = terabyte(4_096, 32 << 30);
        let costs = compare(&w, &NVME_MODERN_MODEL, &HostModel::cores(8));
        let gather = pick(&costs, Strategy::Gather);
        let staged = pick(&costs, Strategy::Splat);

        assert_eq!(
            gather.bytes_read,
            w.payload_bytes(),
            "a 4 KiB record in a 4 KiB block reads no padding"
        );
        assert!(
            gather.seconds < staged.seconds,
            "the gather should win here: {} against {}",
            human_time(gather.seconds),
            human_time(staged.seconds)
        );
        assert!(
            gather.pegged().contains(&Resource::Bandwidth),
            "and it wins because bandwidth, not command count, is what bounds it"
        );

        // Shrink the record and the boundary is crossed: the same device,
        // the same payload, eight times the ordinals, and the gather is
        // now paying for padding and for commands at once.
        let small = terabyte(512, 32 << 30);
        let small_costs = compare(&small, &NVME_MODERN_MODEL, &HostModel::cores(8));
        assert!(
            pick(&small_costs, Strategy::Splat).seconds
                < pick(&small_costs, Strategy::Gather).seconds,
            "at 512 B the staged rewrite should win"
        );
        assert!(
            pick(&small_costs, Strategy::Gather)
                .pegged()
                .contains(&Resource::Controller),
            "and what changed is that the controller ran out"
        );
    }

    /// The advantage is not uniform, and the shape is the point. It grows
    /// as records shrink, because a smaller record means more ordinals
    /// for the same payload and more commands a gather cannot avoid.
    #[test]
    fn the_advantage_grows_as_records_shrink() {
        let mut previous = 0.0;
        for record_bytes in [16_384u64, 4_096, 1_024, 256] {
            let w = terabyte(record_bytes, 32 << 30);
            let gain = advantage(&w, &NVME_CONSUMER_MODEL, &HostModel::DEFAULT);
            assert!(
                gain > previous,
                "smaller records should widen the advantage: {record_bytes} B gave \
                 {gain:.1}x against {previous:.1}x"
            );
            previous = gain;
        }
    }

    /// And it decays as memory grows, because the pass count falls toward
    /// its floor and there is less left to save.
    #[test]
    fn the_advantage_is_bounded_by_the_pass_floor() {
        let plenty = Workload {
            budget_bytes: (1u64 << 40) / 2,
            ..terabyte(1_024, 32 << 30)
        };
        assert_eq!(
            plenty.passes(),
            2,
            "a huge budget still floors at two passes"
        );
        let gain = advantage(&plenty, &NVME_CONSUMER_MODEL, &HostModel::DEFAULT);
        assert!(gain > 1.0, "ordering should still win at the floor");
    }

    /// A modern drive moves the boundary but does not remove it: the
    /// command rate is higher, so the ordinal count at which a gather
    /// becomes impractical is higher too.
    #[test]
    fn a_faster_device_moves_the_boundary_rather_than_removing_it() {
        let w = terabyte(1_024, 32 << 30);
        let old = advantage(&w, &NVME_CONSUMER_MODEL, &HostModel::DEFAULT);
        let new = advantage(&w, &NVME_MODERN_MODEL, &HostModel::cores(8));
        assert!(new < old, "a faster device should narrow the advantage");
        assert!(new > 1.0, "but not close it at a terabyte");
    }

    /// The analytic model has to agree with the discrete-event simulator
    /// where both can run, or the extrapolation to a terabyte rests on
    /// nothing.
    #[test]
    fn the_analytic_model_agrees_with_the_simulator_at_traceable_scale() {
        use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};
        use crate::cache::CacheConfig;
        use crate::io::hw::SPINNING_SATA_HW;
        use crate::model::{Geometry, Map};
        use crate::price;

        let records = 100_000u64;
        let record_bytes = 4_096u64;
        let geometry = Geometry {
            records,
            record_bytes,
            container_bytes: 128 * 1024,
        };
        let budget = geometry.payload_bytes() / 8;
        let map = Map::shuffled(records, 0x5CA1E);
        let cache = Some(CacheConfig::single_page(128 * 1024));

        let simulated_ordered = price::simulate_io(
            &Gsplat::new().run(geometry, &map, budget).1,
            &SPINNING_SATA_HW,
            cache,
            32,
        )
        .elapsed_s;
        let simulated_gather = price::simulate_io(
            &NaiveGather.run(geometry, &map, budget).1,
            &SPINNING_SATA_HW,
            cache,
            32,
        )
        .elapsed_s;

        let w = Workload {
            records,
            record_bytes,
            budget_bytes: budget,
            container_bytes: 128 * 1024,
            block_bytes: 4_096,
            depth: 32.0,
        };
        let analytic = compare(&w, &SPINNING_SATA_MODEL, &HostModel::DEFAULT);

        // The two paths share no code, so agreement on the *ratio* is
        // what matters: the analytic model is used for the advantage at
        // scales the simulator cannot reach.
        let simulated_gain = simulated_gather / simulated_ordered;
        let analytic_gain = pick(&analytic, Strategy::Gather).seconds
            / pick(&analytic, Strategy::OrderedScan).seconds;
        assert!(
            (simulated_gain / analytic_gain).clamp(0.25, 4.0) == simulated_gain / analytic_gain,
            "analytic and simulated advantage should agree within 4x: \
             {analytic_gain:.1}x analytic against {simulated_gain:.1}x simulated"
        );
    }

    /// **The staged arm, proved in the small.** The analytic price for
    /// [`Strategy::Splat`] is not a hopeful formula — there is a working
    /// implementation whose trace goes through the same cache and the
    /// same discrete-event device path as everything else, and it has to
    /// agree with the closed form at a scale where both can run.
    #[test]
    fn the_staged_arm_agrees_with_its_implementation_at_traceable_scale() {
        use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather, staged::StagedSplat};
        use crate::cache::CacheConfig;
        use crate::io::hw::SPINNING_SATA_HW;
        use crate::model::{Geometry, Map};
        use crate::price;

        let records = 200_000u64;
        let record_bytes = 512u64;
        let geometry = Geometry {
            records,
            record_bytes,
            container_bytes: 128 * 1024,
        };
        // Memory at an eighth of the payload — eight segments, which is
        // the shape the terabyte case has, just smaller.
        let budget = geometry.payload_bytes() / 8;
        let map = Map::shuffled(records, 0x5EED);
        let cache = Some(CacheConfig::single_page(128 * 1024));

        let run =
            |t: &crate::model::Trace| price::simulate_io(t, &SPINNING_SATA_HW, cache, 32).elapsed_s;
        let staged = run(&StagedSplat::new().run(geometry, &map, budget).1);
        let rescan = run(&Gsplat::new().run(geometry, &map, budget).1);
        let gather = run(&NaiveGather.run(geometry, &map, budget).1);

        // The ordering the simulator produces is the claim: staging beats
        // re-scanning, and re-scanning beats a random gather on a device
        // where positioning dominates.
        assert!(
            staged < rescan,
            "staged {staged:.2}s should beat the re-scan {rescan:.2}s"
        );
        assert!(
            rescan < gather,
            "re-scan {rescan:.2}s should beat the gather {gather:.2}s"
        );

        let w = Workload {
            records,
            record_bytes,
            budget_bytes: budget,
            container_bytes: 128 * 1024,
            block_bytes: 4_096,
            depth: 32.0,
        };
        let analytic = compare(&w, &SPINNING_SATA_MODEL, &HostModel::DEFAULT);
        let analytic_gain =
            pick(&analytic, Strategy::Gather).seconds / pick(&analytic, Strategy::Splat).seconds;
        let simulated_gain = gather / staged;
        assert!(
            (simulated_gain / analytic_gain).clamp(0.2, 5.0) == simulated_gain / analytic_gain,
            "analytic and simulated staging advantage should agree within 5x: \
             {analytic_gain:.1}x analytic against {simulated_gain:.1}x simulated"
        );
    }

    /// And the implementation keeps the invariants it claims while doing
    /// it — including the one the re-scan form cannot: spill balance.
    #[test]
    fn the_staged_implementation_keeps_its_invariants() {
        use crate::algo::{Rewrite, staged::StagedSplat};
        use crate::model::{Geometry, Map};

        let geometry = Geometry::new(50_000, 512, 65_536);
        let map = Map::shuffled(50_000, 0xB0A7);
        for divisor in [2u64, 8, 32, 128] {
            let budget = geometry.payload_bytes() / divisor;
            let trace = StagedSplat::new().run(geometry, &map, budget).1;
            let violations = crate::check::check_staged(&trace, &map, budget);
            assert!(
                violations.is_empty(),
                "budget 1/{divisor}: {}",
                violations
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
    }
}

#[cfg(all(test, feature = "heavy-tests"))]
mod crossover {
    use super::*;
    use crate::device::{ALL_MODELS_WITH_MODERN, NVME_CONSUMER_MODEL};

    fn at_segments(segments: u64) -> Workload {
        let record_bytes = 1_024u64;
        let records = (1u64 << 40) / record_bytes;
        Workload {
            records,
            record_bytes,
            budget_bytes: records.div_ceil(segments) * record_bytes,
            container_bytes: 128 * 1024,
            block_bytes: 4_096,
            depth: 32.0,
        }
    }

    /// **Where staging starts to pay against merely ordering.**
    ///
    /// A staged rewrite moves the payload four times — read and write in
    /// the distribution stage, read and write in the transfer stage —
    /// whatever the segment count. A re-scan moves it `A(P) + 1` times.
    /// So the two are level at `A(P) = 3`, and since `A(P) → P` while
    /// `P ≪ w`, that lands at **three or four segments**: below it the
    /// re-scan is cheaper because it never touches scratch, and above it
    /// the re-scan's cost keeps climbing while the staged form's does
    /// not.
    ///
    /// `docs/gsplat/cost-model.md` says the two-level variant is
    /// "unnecessary while they fit in a handful of passes". This puts a
    /// number on "a handful", and the number is small.
    #[test]
    fn staging_overtakes_the_rescan_at_about_four_segments() {
        let mut crossover = None;
        for segments in 2..=16u64 {
            let w = at_segments(segments);
            let costs = compare(&w, &NVME_CONSUMER_MODEL, &HostModel::cores(8));
            let rescan = pick(&costs, Strategy::OrderedScan).seconds;
            let staged = pick(&costs, Strategy::Splat).seconds;
            if staged < rescan && crossover.is_none() {
                crossover = Some(segments);
            }
        }
        let segments = crossover.expect("staging should overtake somewhere in 2..=16");
        assert!(
            (3..=6).contains(&segments),
            "the crossover should be a handful of segments, got {segments}"
        );
        println!("  staging overtakes the re-scan at {segments} segments");

        // Below it the re-scan really is cheaper — worth asserting,
        // because a claim that staging always wins would be false.
        let small = at_segments(2);
        let small_costs = compare(&small, &NVME_CONSUMER_MODEL, &HostModel::cores(8));
        assert!(
            pick(&small_costs, Strategy::OrderedScan).seconds
                < pick(&small_costs, Strategy::Splat).seconds,
            "at two segments the re-scan avoids the scratch round trip and wins"
        );
    }

    /// And the crossover is a property of the *algorithms*, not of the
    /// device: both arms are sequential there, so it lands in the same
    /// place on every device modelled.
    #[test]
    fn the_crossover_is_the_same_on_every_device() {
        let first = |device: &crate::device::DeviceModel| {
            (2..=16u64).find(|&segments| {
                let w = at_segments(segments);
                let costs = compare(&w, device, &HostModel::cores(8));
                pick(&costs, Strategy::Splat).seconds < pick(&costs, Strategy::OrderedScan).seconds
            })
        };
        let reference = first(&NVME_CONSUMER_MODEL);
        for device in ALL_MODELS_WITH_MODERN {
            assert_eq!(
                first(device),
                reference,
                "{} disagrees about the crossover",
                device.name
            );
        }
    }
}
