// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **The manifold**: the region of parameter space in which a staged
//! rewrite stops being an optimisation and becomes the only way to
//! finish.
//!
//! [`crate::scale`] prices strategies and [`crate::queueing`] bounds
//! them. This module walks the axes those two are parameterized by and
//! reports where the answer changes, because the useful claim about
//! gsplat is not "it is faster" — it is *here is the boundary, here is
//! which resource produced it, and here is what happens on each side*.
//!
//! # The axes, and what each one moves
//!
//! | Axis | What it changes | Why it matters |
//! |---|---|---|
//! | ordinal count `N` | commands, at fixed payload | a gather issues one command per ordinal and nothing coalesces them |
//! | memory `M` | segments `= payload/M` | segments drive the re-scan's amplification and the staged form's stage count |
//! | record size `R` | padding per random access, and `N` at fixed payload | at `R < block` every random read fetches waste |
//! | container `W` | records per command `w`, and fan-out `M/W` | `w` is the coalescing gain; `M/W` is how many buckets one stage can hold |
//! | device class | which ceiling binds | the boundary is at a different `R` on a disk than on an SSD |
//! | issue depth `n` | where the system sits against `n*` | below the knee nothing is saturated and the analysis is about concurrency, not capacity |
//!
//! # What the studies keep finding
//!
//! Three results recur across the grids below, and they are the whole
//! shape of the manifold:
//!
//! 1. **Staging, not ordering, is what scales.** Ordering the reads
//!    without a spill extent — [`Strategy::OrderedScan`] — sweeps the
//!    source once per segment, so its cost rises with `payload/M`. On
//!    flash it frequently loses to a naive gather outright. What removes
//!    the re-read is the bucket spill, and that is the step worth
//!    defending.
//! 2. **The boundary is drawn by the record size, not the payload.**
//!    When a record fills a block there is no padding to waste and the
//!    ordinal count for a given payload is at its smallest; a fast
//!    device's controller can absorb what remains and the gather wins.
//!    Shrink the record and both terms move against the gather at once —
//!    more ordinals *and* more waste per ordinal.
//! 3. **Both ceilings are real and they are different ceilings.** A
//!    gather on small records pegs the *controller*; the same gather on
//!    block-sized records pegs *bandwidth*; a staged rewrite always pegs
//!    bandwidth because that is what it converted the problem into.
//!    Which one is pegged decides whether a faster link or a deeper
//!    queue would help, and a single speedup number cannot say.
//!
//! Device figures come from the corpus named in
//! [the crate bibliography](crate#sources). The external-memory bound the
//! staged form realizes is Aggarwal & Vitter's
//! `Θ((N/B)·log_{M/B}(N/B))`, *Communications of the ACM* 31(9), 1988.

use crate::device::{
    ALL_MODELS_WITH_MODERN, DeviceModel, NVME_CONSUMER_MODEL, NVME_MODERN_MODEL,
    SPINNING_SATA_MODEL,
};
use crate::io::hw::HostModel;
use crate::queueing;
use crate::scale::{
    Cost, Strategy, Workload, best_naive, compare, human_bytes, human_count, human_time, pick,
    render_comparison,
};
use std::fmt::Write as _;

/// A named study over the parameter space.
pub struct Study {
    pub name: &'static str,
    pub headline: &'static str,
    run: fn() -> String,
}

impl Study {
    pub fn render(&self) -> String {
        (self.run)()
    }
}

/// Every study, in the order they build on each other.
pub const ALL: &[Study] = &[
    Study {
        name: "scale",
        headline: "ordinal count, from a million to a billion",
        run: scale_study,
    },
    Study {
        name: "memory",
        headline: "memory as a fraction of the payload",
        run: memory_study,
    },
    Study {
        name: "record",
        headline: "record size, and where the boundary actually is",
        run: record_study,
    },
    Study {
        name: "strategies",
        headline: "all four strategies side by side, per device",
        run: strategy_study,
    },
    Study {
        name: "pegged",
        headline: "which resources saturate, across devices and records",
        run: pegged_study,
    },
    Study {
        name: "bounds",
        headline: "operational bounds: D_max, the knee, and utilization",
        run: bounds_study,
    },
    Study {
        name: "depth",
        headline: "issue depth against the knee",
        run: depth_study,
    },
    Study {
        name: "fanout",
        headline: "container size, fan-out, and the stage count",
        run: fanout_study,
    },
    Study {
        name: "readahead",
        headline: "why block readahead makes a scattered rewrite worse",
        run: readahead_study,
    },
    Study {
        name: "scheduler",
        headline: "what the Linux block scheduler costs the rewrite",
        run: scheduler_study,
    },
    Study {
        name: "writeback",
        headline: "dirty-page pacing, and why a buffered scatter cannot be saved",
        run: writeback_study,
    },
    Study {
        name: "corners",
        headline: "corner cases, with each delta attributed to one factor",
        run: corners_study,
    },
    Study {
        name: "frontier",
        headline: "the boundary itself: record size where staging starts to win",
        run: frontier_study,
    },
];

/// Look a study up by name.
pub fn find(name: &str) -> Option<&'static Study> {
    ALL.iter().find(|s| s.name == name)
}

/// Run every study.
pub fn render_all() -> String {
    ALL.iter().map(|s| s.render()).collect()
}

/// A terabyte of `record_bytes` records with `budget_bytes` of memory —
/// the shape the whole module is about.
pub fn terabyte(record_bytes: u64, budget_bytes: u64) -> Workload {
    Workload {
        records: (1u64 << 40) / record_bytes,
        record_bytes,
        budget_bytes,
        container_bytes: 128 * 1024,
        block_bytes: 4_096,
        depth: 32.0,
    }
}

fn heading(s: &mut String, title: &str, subtitle: &str) {
    let _ = writeln!(s, "\n{title}");
    let _ = writeln!(s, "{}", "─".repeat(title.len()));
    let _ = writeln!(s, "{subtitle}");
}

fn pegged_list(cost: &Cost) -> String {
    let names: Vec<&str> = cost.pegged().iter().map(|r| r.label()).collect();
    if names.is_empty() {
        "—".to_string()
    } else {
        names.join(" + ")
    }
}

// ── 1 · scale ────────────────────────────────────────────────────────

fn scale_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 1 — ordinal count",
        "1 KiB records, 32 GiB budget, depth 32, on a seek-bound disk.\n\
         Payload grows with the ordinal count, so this is the axis along\n\
         which a workable job becomes an unworkable one.",
    );
    for records in [1_000_000u64, 10_000_000, 100_000_000, 1_000_000_000] {
        let w = Workload {
            records,
            ..terabyte(1_024, 32 << 30)
        };
        s.push_str(&render_comparison(
            &format!("N = {}", human_count(records)),
            &w,
            &SPINNING_SATA_MODEL,
            &HostModel::DEFAULT,
        ));
    }
    let _ = writeln!(
        s,
        "\n  Both costs are linear in N — the naive one at a positioning\n  \
           time per record, the staged one at a streaming time per byte —\n  \
           so the *ratio* barely moves down the page. What moves is the\n  \
           absolute time: the same advantage that is academic at a million\n  \
           ordinals is the difference between a shift and a quarter at a\n  \
           billion. The staged column steps once, where the payload\n  \
           outgrows memory and buys a second pass."
    );
    s
}

// ── 2 · memory ───────────────────────────────────────────────────────

fn memory_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 2 — memory as a fraction of the payload",
        "1 TiB of 1 KiB records on a consumer NVMe. Memory does two\n\
         different things: it divides the re-scan's sweep count, and it\n\
         multiplies the staged form's fan-out. Only one of those is a\n\
         strong lever.",
    );
    let _ = writeln!(
        s,
        "\n  {:>9} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>8}",
        "budget", "segments", "A(P)", "fan-out", "stages", "rescan", "staged", "gain"
    );
    for budget in [
        1u64 << 30,
        4 << 30,
        16 << 30,
        64 << 30,
        256 << 30,
        512 << 30,
    ] {
        let w = terabyte(1_024, budget);
        let costs = compare(&w, &NVME_CONSUMER_MODEL, &HostModel::DEFAULT);
        let staged = pick(&costs, Strategy::Splat);
        let _ = writeln!(
            s,
            "  {:>9} {:>9} {:>8.1} {:>8} {:>9} {:>9} {:>9} {:>7.1}x",
            human_bytes(budget),
            human_count(w.segments()),
            w.amplification(),
            human_count(w.fanout()),
            w.stages(),
            human_time(pick(&costs, Strategy::OrderedScan).seconds),
            human_time(staged.seconds),
            best_naive(&costs).seconds / staged.seconds
        );
    }
    let _ = writeln!(
        s,
        "\n  The re-scan column moves by orders of magnitude across this\n  \
           range; the staged column does not move at all until the fan-out\n  \
           falls below the segment count. **Staging converts memory from a\n  \
           throughput parameter into a correctness-of-scale parameter** —\n  \
           you need enough to hold one bucket buffer per segment, and past\n  \
           that more memory buys nothing."
    );
    s
}

// ── 3 · record size ──────────────────────────────────────────────────

fn record_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 3 — record size at a fixed 1 TiB payload",
        "32 GiB budget. Shrinking the record does two things to a naive\n\
         gather at once: it multiplies the ordinal count, and — below the\n\
         4 KiB block — it makes every random read mostly padding.",
    );
    let _ = writeln!(
        s,
        "\n  {:>7} {:>9} {:>6} {:>7} {:>9} {:>9} {:>8}  gather pegs",
        "record", "ordinals", "w", "waste", "gather", "staged", "gain"
    );
    for record_bytes in [128u64, 256, 512, 1_024, 4_096, 16_384, 65_536] {
        let w = terabyte(record_bytes, 32 << 30);
        let costs = compare(&w, &NVME_CONSUMER_MODEL, &HostModel::DEFAULT);
        let gather = pick(&costs, Strategy::Gather);
        let staged = pick(&costs, Strategy::Splat);
        let _ = writeln!(
            s,
            "  {:>7} {:>9} {:>6} {:>6.1}x {:>9} {:>9} {:>7.1}x  {}",
            record_bytes,
            human_count(w.records),
            w.w(),
            gather.bytes_read as f64 / w.payload_bytes() as f64,
            human_time(gather.seconds),
            human_time(staged.seconds),
            gather.seconds / staged.seconds,
            pegged_list(gather)
        );
    }
    let _ = writeln!(
        s,
        "\n  Read the `gather pegs` column down the page: it changes from\n  \
           bandwidth to controller as the record shrinks. That is the same\n  \
           boundary the gain column crosses, seen from the resource side."
    );
    s
}

// ── 4 · strategies ───────────────────────────────────────────────────

fn strategy_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 4 — all four strategies, per device",
        "1 TiB of 512 B records, 32 GiB budget. Both naive sides are\n\
         priced: a gather's reads are random, a scatter's writes are, and\n\
         a partial-block write costs a read-modify-write on top.",
    );
    for device in ALL_MODELS_WITH_MODERN {
        s.push_str(&render_comparison(
            device.name,
            &terabyte(512, 32 << 30),
            device,
            &HostModel::DEFAULT,
        ));
    }
    let _ = writeln!(
        s,
        "\n  The scatter is never the answer — it pays the gather's\n  \
           positioning cost and then pays again to fetch every block it\n  \
           only partially overwrites. It is here so that \"avoid random\n  \
           writes\" is a measured claim rather than an assumed one."
    );
    s
}

// ── 5 · pegged resources ─────────────────────────────────────────────

fn pegged_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 5 — which resources saturate",
        "1 TiB, 32 GiB budget, 8 host cores. A run can exhaust command\n\
         processing and bandwidth at once; naming only the larger of the\n\
         two hides that there is no headroom anywhere.",
    );
    let _ = writeln!(
        s,
        "\n  {:<15} {:>7} {:<15} {:>9} {:>9}  pegged at >=95%",
        "device", "record", "strategy", "time", "X (rec/s)"
    );
    for device in ALL_MODELS_WITH_MODERN {
        for record_bytes in [512u64, 4_096] {
            let w = terabyte(record_bytes, 32 << 30);
            for c in compare(&w, device, &HostModel::cores(8)) {
                let _ = writeln!(
                    s,
                    "  {:<15} {:>7} {:<15} {:>9} {:>9}  {}",
                    device.name,
                    record_bytes,
                    c.strategy.label(),
                    human_time(c.seconds),
                    human_count(c.records_per_second as u64),
                    pegged_list(&c)
                );
            }
        }
    }
    let _ = writeln!(
        s,
        "\n  A staged rewrite always pegs bandwidth, because bandwidth is\n  \
           what it converted the problem into. That is the trade stated as\n  \
           a measurement: command demand falls by w, byte demand rises by\n  \
           the passes, and the rewrite wins wherever the first term was\n  \
           the binding one."
    );
    s
}

// ── 6 · operational bounds ───────────────────────────────────────────

fn bounds_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 6 — operational bounds",
        "1 TiB of 512 B records, 32 GiB budget, 8 cores, on a consumer\n\
         NVMe. Service demand per record decides everything: the maximum\n\
         transform rate is 1/D_max, the knee is D_total/D_max, and the\n\
         completion time is N/X(n).",
    );
    let w = terabyte(512, 32 << 30);
    for c in compare(&w, &NVME_CONSUMER_MODEL, &HostModel::cores(8)) {
        s.push_str(&queueing::render(
            c.strategy.label(),
            &c.demand,
            w.records,
            w.depth,
        ));
    }
    let _ = writeln!(
        s,
        "\n  X(n) <= min(n/D_total, 1/D_max) and R(n) >= max(D_total, n·D_max)\n  \
           are Denning & Buzen's operational bounds, and they need no\n  \
           distributional assumption at all — only that the demands are\n  \
           what they are. The transform rate a strategy can reach is\n  \
           therefore a property of its request profile, not of the\n  \
           scheduler that issues it."
    );
    s
}

// ── 7 · issue depth ──────────────────────────────────────────────────

fn depth_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 7 — issue depth against the knee",
        "1 TiB of 512 B records on a modern NVMe. Below n* concurrency\n\
         buys throughput; above it concurrency buys only queueing delay.",
    );
    let w = terabyte(512, 32 << 30);
    let _ = writeln!(
        s,
        "\n  {:<15} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "strategy", "n*", "n=1", "n=8", "n=n*", "n=1024"
    );
    for c in compare(&w, &NVME_MODERN_MODEL, &HostModel::cores(8)) {
        let knee = c.demand.saturation_concurrency();
        let _ = writeln!(
            s,
            "  {:<15} {:>8.1} {:>10} {:>10} {:>10} {:>10}",
            c.strategy.label(),
            knee,
            human_time(c.demand.completion_seconds(w.records, 1.0)),
            human_time(c.demand.completion_seconds(w.records, 8.0)),
            human_time(c.demand.completion_seconds(w.records, knee)),
            human_time(c.demand.completion_seconds(w.records, 1024.0)),
        );
    }
    let _ = writeln!(
        s,
        "\n  The n=n* and n=1024 columns are identical by construction, and\n  \
           that identity is the practical point: past the knee there is\n  \
           nothing left to win by issuing deeper, and every extra request\n  \
           in flight is latency added to some record's residence time."
    );
    s
}

// ── 8 · fan-out ──────────────────────────────────────────────────────

fn fanout_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 8 — container size, fan-out, and stages",
        "1 TiB of 512 B records, 4 GiB budget. A bucket buffer has to be\n\
         big enough that its spill writes are sequential; making it bigger\n\
         costs fan-out, and losing fan-out costs a whole extra pass over\n\
         the payload.",
    );
    let _ = writeln!(
        s,
        "\n  {:>9} {:>6} {:>10} {:>9} {:>8} {:>10} {:>10}",
        "container", "w", "segments", "fan-out", "stages", "read", "staged"
    );
    for container_bytes in [
        16u64 << 10,
        64 << 10,
        128 << 10,
        1 << 20,
        16 << 20,
        256 << 20,
    ] {
        let w = Workload {
            container_bytes,
            ..terabyte(512, 4 << 30)
        };
        let staged = crate::scale::cost(
            Strategy::Splat,
            &w,
            &NVME_CONSUMER_MODEL,
            &HostModel::DEFAULT,
        );
        let _ = writeln!(
            s,
            "  {:>9} {:>6} {:>10} {:>9} {:>8} {:>10} {:>10}",
            human_bytes(container_bytes),
            w.w(),
            human_count(w.segments()),
            human_count(w.fanout()),
            w.stages(),
            human_bytes(staged.bytes_read),
            human_time(staged.seconds)
        );
    }
    let _ = writeln!(
        s,
        "\n  Stages are ceil(log_f(segments)) with f = M/W, so the cost of a\n  \
           bigger container is a step function, not a slope. Everything\n  \
           left of the step is free; the first configuration past it pays a\n  \
           full extra read and write of the payload."
    );
    s
}

// ── 9 · readahead ────────────────────────────────────────────────────

fn readahead_study() -> String {
    use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather, staged::StagedSplat};
    use crate::cache::CacheConfig;
    use crate::io::{Readahead, hw};
    use crate::model::{Geometry, Map};
    use crate::price;

    let mut s = String::new();
    heading(
        &mut s,
        "Study 9 — readahead on a scattered rewrite",
        "This one is simulated rather than priced: 200,000 records of\n\
         512 B through the discrete-event storage path, with the kernel's\n\
         readahead on and off. Readahead does not merely fail to help a\n\
         scattered reader — it taxes one.",
    );
    let _ = writeln!(
        s,
        "\n  A miss the kernel does not recognize as sequential starts a new\n  \
           region and is seeded at get_init_ra_size(): the request size is\n  \
           rounded to a power of two and multiplied, so a 4 KiB fault\n  \
           fetches 16 KiB against a 128 KiB ceiling. Three of those four\n  \
           pages are speculative and a scattered stream uses none of them.\n  \
           Source: linux/mm/readahead.c. The documented remedy is\n  \
           POSIX_FADV_RANDOM, which sets ra_pages to zero."
    );

    let geometry = Geometry {
        records: 200_000,
        record_bytes: 512,
        container_bytes: 128 * 1024,
    };
    let map = Map::shuffled(geometry.records, 0x4EAD);
    let budget = geometry.payload_bytes() / 8;
    let cache = CacheConfig::new(geometry.payload_bytes() / 8, 4_096);

    let traces: Vec<(&str, crate::model::Trace)> = vec![
        ("naive gather", NaiveGather.run(geometry, &map, budget).1),
        (
            "ordered rescan",
            Gsplat::new().run(geometry, &map, budget).1,
        ),
        (
            "gsplat staged",
            StagedSplat::new().run(geometry, &map, budget).1,
        ),
    ];

    let _ = writeln!(
        s,
        "\n  {:<15} {:>10} {:>10} {:>8} {:>12} {:>12}",
        "strategy", "RA on", "RA off", "tax", "bytes on", "bytes off"
    );
    for (name, trace) in &traces {
        let at = |ra: Readahead| {
            crate::io::run(
                &hw::NVME_CONSUMER_HW,
                &mut crate::io::sched::Noop::default(),
                &mut crate::io::Recorded::new(price::accesses_of(trace)),
                crate::io::RunConfig {
                    readahead: ra,
                    cache: Some(cache),
                    ..crate::io::RunConfig::buffered(
                        32,
                        geometry.payload_bytes() * 2 + price::spill_extent_bytes(trace),
                        cache,
                    )
                },
            )
        };
        let on = at(Readahead::DEFAULT);
        let off = at(Readahead::OFF);
        let _ = writeln!(
            s,
            "  {:<15} {:>10} {:>10} {:>7.2}x {:>12} {:>12}",
            name,
            human_time(on.elapsed_s),
            human_time(off.elapsed_s),
            on.elapsed_s / off.elapsed_s,
            human_bytes(on.bytes_transferred),
            human_bytes(off.bytes_transferred)
        );
    }
    let _ = writeln!(
        s,
        "\n  The tax column is the whole point of the study: above 1.0 the\n  \
           kernel's guessing is costing time. An ordered reader is on the\n  \
           other side of the same mechanism — its prefetches are used, so\n  \
           readahead is the thing making its passes cheap. **The same\n  \
           kernel feature is a subsidy for one access pattern and a tax on\n  \
           the other**, which is a second, independent reason ordering is\n  \
           worth arranging."
    );
    s
}

// ── 10 · schedulers ──────────────────────────────────────────────────

/// The three rewrites, traced at a scale the simulator can run.
fn traced_rewrites(
    geometry: crate::model::Geometry,
    map: &crate::model::Map,
    budget: u64,
) -> Vec<(&'static str, crate::model::Trace)> {
    use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather, staged::StagedSplat};
    vec![
        ("naive gather", NaiveGather.run(geometry, map, budget).1),
        ("ordered rescan", Gsplat::new().run(geometry, map, budget).1),
        (
            "gsplat staged",
            StagedSplat::new().run(geometry, map, budget).1,
        ),
    ]
}

fn scheduler_study() -> String {
    use crate::io::sched::LinuxScheduler;
    use crate::io::{self, RandomAccess, RunConfig, hw};
    use crate::model::{Geometry, Map};
    use crate::price;

    let mut s = String::new();
    heading(
        &mut s,
        "Study 10 — the block scheduler",
        "Which scheduler is in force is configuration a benchmark usually\n\
         forgets to state. It can cost 60% of a modern NVMe's throughput —\n\
         and cost a gsplat rewrite nothing at all. Both halves are here.",
    );
    let _ = writeln!(
        s,
        "\n  Ren, Doekemeijer, Tehrany & Trivedi (ICPE '24) measured all four\n  \
           on Samsung NVMe: against a 785.7 KIOPS device, none and kyber\n  \
           reach it, mq-deadline peaks at 569.2 (0.72x) and bfq at 315.3\n  \
           (0.40x). The cause is lock contention, not policy — up to 78.0%\n  \
           of cycles for bfq. In exchange kyber and bfq deliver up to 99.3%\n  \
           lower P99 under interference."
    );

    // (a) Saturation: drive the device hard enough to find the ceilings.
    let span = 512u64 << 20;
    let _ = writeln!(
        s,
        "\n  (a) At saturation — 4 KiB random reads, depth 512, 16 cores.\n"
    );
    let _ = writeln!(
        s,
        "  {:<14} {:>12} {:>12} {:>10} {:>10}  what bounds it",
        "scheduler", "achieved", "published", "error", "lock busy"
    );
    for scheduler in LinuxScheduler::ALL {
        let stats = io::run_under(
            &hw::NVME_MODERN_HW,
            scheduler,
            &mut RandomAccess::new(span, 4_096, 200_000, 0x5A7),
            RunConfig {
                host: hw::HostModel::cores(16),
                ..RunConfig::direct(512, span)
            },
        );
        let achieved = stats.requests_completed as f64 / stats.elapsed_s;
        let published = scheduler.measured_ceiling_iops();
        // For `none` and `kyber` the published figure is the *device's*
        // ceiling on the paper's Samsung drive, not a property of the
        // scheduler. This crate's modern-NVMe model is a faster drive, so
        // exceeding it there is expected and says nothing either way.
        let scheduler_bound = scheduler.dispatch_cost_s() > 0.0;
        let _ = writeln!(
            s,
            "  {:<14} {:>11.0}k {:>11.0}k {:>9} {:>9.0}%  {}",
            scheduler.name(),
            achieved / 1e3,
            published / 1e3,
            if scheduler_bound {
                format!("{:.1}%", (achieved - published) / published * 100.0)
            } else {
                "n/a".to_string()
            },
            stats.scheduler_utilization() * 100.0,
            if scheduler_bound {
                "the scheduler's lock"
            } else {
                "the device (this model's drive is faster than the paper's)"
            }
        );
    }

    // (b) The rewrites, at the command rate they actually offer.
    let geometry = Geometry {
        records: 60_000,
        record_bytes: 512,
        container_bytes: 128 * 1024,
    };
    let map = Map::shuffled(geometry.records, 0x5C4E);
    let budget = geometry.payload_bytes() / 8;
    let traces = traced_rewrites(geometry, &map, budget);

    let _ = writeln!(
        s,
        "\n  (b) The rewrites themselves — 60,000 records of 512 B, depth 128.\n"
    );
    let _ = writeln!(
        s,
        "  {:<15} {:>10} {:>10} {:>11} {:>10} {:>12}",
        "strategy", "none", "mq-deadline", "kyber", "bfq", "offered IOPS"
    );
    for (name, trace) in &traces {
        let _ = write!(s, "  {name:<15}");
        let mut offered = 0.0;
        for scheduler in LinuxScheduler::ALL {
            let stats = io::run_under(
                &hw::NVME_MODERN_HW,
                scheduler,
                &mut io::Recorded::new(price::accesses_of(trace)),
                RunConfig {
                    host: hw::HostModel::cores(8),
                    ..RunConfig::direct(
                        128,
                        geometry.payload_bytes() * 2 + price::spill_extent_bytes(trace),
                    )
                },
            );
            if scheduler == LinuxScheduler::None {
                offered = stats.requests_completed as f64 / stats.elapsed_s;
            }
            let _ = write!(s, " {:>10}", human_time(stats.elapsed_s));
        }
        let _ = writeln!(s, " {:>11.0}k", offered / 1e3);
    }

    let _ = writeln!(
        s,
        "\n  Read the two halves together. The ceilings in (a) are real and\n  \
           the model reproduces them, but in (b) nothing moves — because\n  \
           none of these rewrites offers a command rate anywhere near\n  \
           315 KIOPS. **A scheduler can only tax commands you issue.** A\n  \
           gsplat rewrite issues few large ones and is therefore immune to\n  \
           a configuration choice nobody remembers to make; a naive gather\n  \
           at a billion ordinals is not, because its command rate is the\n  \
           thing it is trying to sustain and the scheduler is one more\n  \
           ceiling in its way."
    );
    s
}

// ── 11 · writeback ───────────────────────────────────────────────────

fn writeback_study() -> String {
    use crate::cache::CacheConfig;
    use crate::io::{self, RunConfig, Writeback, hw, sched};
    use crate::model::{Geometry, Map};
    use crate::price;

    let mut s = String::new();
    heading(
        &mut s,
        "Study 11 — dirty-page pacing",
        "The same rewrites, buffered. A write does not reach the device;\n\
         it dirties a page and returns, and what happens next is decided\n\
         by dirty_background_ratio, dirty_ratio, and how well the pages\n\
         coalesce on the way out.",
    );
    let _ = writeln!(
        s,
        "\n  Linux defaults: the flusher wakes at 10% of memory dirty, the\n  \
           writer is put to sleep in balance_dirty_pages at 20%, pages\n  \
           expire at 30 s, and no single pause exceeds 200 ms. The pacing\n  \
           is IO-less — the writer sleeps rather than submitting writeback\n  \
           itself (Wu Fengguang, 2011), so the flusher keeps the ordering.\n  \
           Sources: Documentation/admin-guide/sysctl/vm.rst; LWN 456904."
    );

    let geometry = Geometry {
        records: 60_000,
        record_bytes: 512,
        container_bytes: 128 * 1024,
    };
    let map = Map::shuffled(geometry.records, 0xD147);
    let budget = geometry.payload_bytes() / 8;
    let ram = geometry.payload_bytes() / 4;
    let cache = CacheConfig::new(ram, 4_096);
    let traces = traced_rewrites(geometry, &map, budget);

    let _ = writeln!(
        s,
        "\n  {:<15} {:>9} {:>9} {:>10} {:>10} {:>11} {:>10}",
        "strategy", "buffered", "direct", "throttled", "peak dirty", "flusher wb", "evict wb"
    );
    for (name, trace) in &traces {
        let span = geometry.payload_bytes() * 2 + price::spill_extent_bytes(trace);
        let at = |writeback: Writeback| {
            io::run(
                &hw::SATA_SSD_HW,
                &mut sched::Noop::default(),
                &mut io::Recorded::new(price::accesses_of(trace)),
                RunConfig {
                    writeback,
                    ..RunConfig::buffered(64, span, cache)
                },
            )
        };
        let buffered = at(Writeback::DEFAULT);
        let direct = at(Writeback::OFF);
        let _ = writeln!(
            s,
            "  {name:<15} {:>9} {:>9} {:>10} {:>10} {:>11} {:>10}",
            human_time(buffered.elapsed_s),
            human_time(direct.elapsed_s),
            human_time(buffered.writeback_throttled_s),
            human_bytes(buffered.peak_dirty_bytes),
            human_count(buffered.flusher_writebacks),
            human_count(buffered.eviction_writebacks),
        );
    }
    let _ = writeln!(
        s,
        "\n  The `evict wb` column is the one to read. A dirty page whose\n  \
           frame is needed has to be written before the frame can be\n  \
           reused — on the allocation path, in LRU order, one page at a\n  \
           time — and all three of these rewrites write their *output* in\n  \
           order, so none of them is a scattered writer. What puts the\n  \
           gather's pages on that path is its **reads**: they land all\n  \
           over the source, each claiming a frame, and the frames they\n  \
           claim are the ones holding output the flusher has not written\n  \
           yet.\n\n  \
           So scattered reads cost twice — once on the device, and again\n  \
           inside the page cache, where they convert someone else's\n  \
           coalesced writeback into page-at-a-time cleaning. A bigger\n  \
           cache does eventually fix it, at roughly half the payload; but\n  \
           a cache that size means the rewrite fitted in memory and there\n  \
           was never a problem to solve."
    );
    s
}

// ── 12 · corner cases, differentially ────────────────────────────────

/// One named configuration at the edge of the parameter space.
struct Corner {
    name: &'static str,
    /// What makes it a corner.
    note: &'static str,
    workload: Workload,
    device: &'static DeviceModel,
}

/// The baseline every differential is measured against: a terabyte of
/// kilobyte records with memory at 3% of it, on a 2016 consumer NVMe.
/// Nothing about it is extreme, which is the point — the corners are
/// what happens when one thing about it is.
fn baseline_corner() -> Corner {
    Corner {
        name: "baseline",
        note: "1 TiB, 1 KiB records, 32 GiB, 128 KiB containers",
        workload: terabyte(1_024, 32 << 30),
        device: &NVME_CONSUMER_MODEL,
    }
}

/// One-factor-at-a-time perturbations of the baseline. Each changes
/// exactly one thing, so the delta it produces is attributable.
fn corners() -> Vec<Corner> {
    let base = baseline_corner().workload;
    vec![
        baseline_corner(),
        Corner {
            name: "fits in memory",
            note: "budget = payload; segments = 1",
            workload: Workload {
                budget_bytes: base.payload_bytes(),
                ..base
            },
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "memory starved",
            note: "budget 1 GiB; 1000 segments",
            workload: Workload {
                budget_bytes: 1 << 30,
                ..base
            },
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "sub-block records",
            note: "128 B records; 8.6 G ordinals, 32x read waste",
            workload: terabyte(128, 32 << 30),
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "record = block",
            note: "4 KiB records; no padding, fewest ordinals",
            workload: terabyte(4_096, 32 << 30),
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "huge records",
            note: "64 KiB records; 16 M ordinals",
            workload: terabyte(65_536, 32 << 30),
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "seek-bound device",
            note: "spinning SATA; positioning dominates",
            workload: base,
            device: &SPINNING_SATA_MODEL,
        },
        Corner {
            name: "fastest device",
            note: "modern NVMe; command rate and bandwidth both high",
            workload: base,
            device: &NVME_MODERN_MODEL,
        },
        Corner {
            name: "container = block",
            note: "4 KiB containers; w = 4, no coalescing left",
            workload: Workload {
                container_bytes: 4_096,
                ..base
            },
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "fan-out squeezed",
            note: "256 MiB containers on a 1 GiB budget: f = 4, so 1000 \
                   segments need five stages",
            workload: Workload {
                container_bytes: 256 << 20,
                budget_bytes: 1 << 30,
                ..base
            },
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "depth 1",
            note: "one request in flight; concurrency-limited",
            workload: Workload { depth: 1.0, ..base },
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "depth 4096",
            note: "far past the knee; capacity-limited",
            workload: Workload {
                depth: 4096.0,
                ..base
            },
            device: &NVME_CONSUMER_MODEL,
        },
        Corner {
            name: "worst case",
            note: "128 B records, 1 GiB budget, seek-bound disk",
            workload: Workload {
                budget_bytes: 1 << 30,
                ..terabyte(128, 32 << 30)
            },
            device: &SPINNING_SATA_MODEL,
        },
        Corner {
            name: "best case for naive",
            note: "64 KiB records, generous memory, fastest device",
            workload: Workload {
                budget_bytes: 512 << 30,
                ..terabyte(65_536, 32 << 30)
            },
            device: &NVME_MODERN_MODEL,
        },
    ]
}

fn corners_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 12 — corner cases, differentially",
        "The single-axis walks above show shape. This shows *attribution*:\n\
         one factor changed at a time from a fixed baseline, so each delta\n\
         belongs to exactly one parameter.",
    );

    let host = HostModel::cores(8);
    let base = baseline_corner();
    let base_costs = compare(&base.workload, base.device, &host);
    let base_naive = best_naive(&base_costs).seconds;
    let base_staged = pick(&base_costs, Strategy::Splat).seconds;
    let base_gain = base_naive / base_staged;

    // (a) Absolute: what each corner costs and who wins there.
    let _ = writeln!(s, "\n  (a) Absolute — every corner, all four strategies.\n");
    let _ = writeln!(
        s,
        "  {:<20} {:>9} {:>9} {:>9} {:>9} {:>8}  winner",
        "corner", "gather", "scatter", "rescan", "staged", "gain"
    );
    for corner in corners() {
        let costs = compare(&corner.workload, corner.device, &host);
        let naive = best_naive(&costs).seconds;
        let staged = pick(&costs, Strategy::Splat).seconds;
        let best = costs
            .iter()
            .min_by(|a, b| a.seconds.partial_cmp(&b.seconds).unwrap())
            .expect("four strategies");
        let _ = writeln!(
            s,
            "  {:<20} {:>9} {:>9} {:>9} {:>9} {:>7.1}x  {}",
            corner.name,
            human_time(pick(&costs, Strategy::Gather).seconds),
            human_time(pick(&costs, Strategy::Scatter).seconds),
            human_time(pick(&costs, Strategy::OrderedScan).seconds),
            human_time(staged),
            naive / staged,
            best.strategy.label()
        );
    }

    // (b) Differential: the delta each single factor produced.
    let _ = writeln!(
        s,
        "\n  (b) Differential — change against the baseline, one factor at a time.\n"
    );
    let _ = writeln!(
        s,
        "  {:<20} {:>9} {:>7} {:>9} {:>9} {:>8} {:>8}  naive pegs",
        "corner", "segments", "stages", "d naive", "d staged", "gain", "d gain"
    );
    for corner in corners() {
        let costs = compare(&corner.workload, corner.device, &host);
        let naive = best_naive(&costs).seconds;
        let staged = pick(&costs, Strategy::Splat).seconds;
        let gain = naive / staged;
        let _ = writeln!(
            s,
            "  {:<20} {:>9} {:>7} {:>9} {:>9} {:>7.1}x {:>8}  {}",
            corner.name,
            human_count(corner.workload.segments()),
            corner.workload.stages(),
            ratio(naive / base_naive),
            ratio(staged / base_staged),
            gain,
            ratio(gain / base_gain),
            pegged_list(best_naive(&costs))
        );
    }

    let _ = writeln!(s, "\n  What each corner changes:\n");
    for corner in corners() {
        let _ = writeln!(s, "    {:<20} {}", corner.name, corner.note);
    }

    let _ = writeln!(
        s,
        "\n  Read column `d gain` as \"how much more (or less) staging is\n  \
           worth here than at the baseline\". Three groups fall out:\n\n  \
           * Factors that make staging matter more — smaller records, a\n    \
             seek-bound device, less memory. All three raise the naive\n    \
             cost without touching the staged one.\n  \
           * Factors that make it matter less — a record that fills a\n    \
             block, a device whose command rate is high enough to absorb\n    \
             the ordinals, memory enough to hold the payload. Two of\n    \
             those three describe a job that was never in trouble.\n  \
           * Factors that move both together — issue depth, container\n    \
             size below the point where fan-out collapses. These change\n    \
             absolute times and leave the decision alone."
    );
    s
}

/// A multiplicative change, rendered so the direction is legible at a
/// glance: `3.2x` for an increase, `/3.2` for a decrease, `—` for no
/// meaningful change.
fn ratio(r: f64) -> String {
    if !r.is_finite() {
        return "—".to_string();
    }
    if (r - 1.0).abs() < 0.02 {
        "—".to_string()
    } else if r >= 1.0 {
        format!("{r:.1}x")
    } else {
        format!("/{:.1}", 1.0 / r)
    }
}

// ── 13 · the frontier ────────────────────────────────────────────────

fn frontier_study() -> String {
    let mut s = String::new();
    heading(
        &mut s,
        "Study 13 — the frontier",
        "The boundary itself. For each device and memory budget, the\n\
         largest record size at which the staged rewrite still beats the\n\
         best naive strategy on a 1 TiB payload — and what the naive\n\
         strategy was pegged on at that point.",
    );
    let sizes = [
        64u64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 65_536,
    ];
    let _ = writeln!(
        s,
        "\n  {:<15} {:>9} {:>12} {:>9} {:>9}  naive pegs at the boundary",
        "device", "budget", "boundary R", "naive", "staged"
    );
    for device in ALL_MODELS_WITH_MODERN {
        for budget in [8u64 << 30, 32 << 30, 128 << 30] {
            let mut boundary = None;
            for &record_bytes in sizes.iter().rev() {
                let w = terabyte(record_bytes, budget);
                let costs = compare(&w, device, &HostModel::cores(8));
                let naive = best_naive(&costs);
                let staged = pick(&costs, Strategy::Splat);
                if staged.seconds < naive.seconds {
                    boundary = Some((
                        record_bytes,
                        naive.seconds,
                        staged.seconds,
                        pegged_list(naive),
                    ));
                    break;
                }
            }
            match boundary {
                Some((r, naive, staged, pegs)) => {
                    let _ = writeln!(
                        s,
                        "  {:<15} {:>9} {:>12} {:>9} {:>9}  {}",
                        device.name,
                        human_bytes(budget),
                        format!("{r} B"),
                        human_time(naive),
                        human_time(staged),
                        pegs
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        "  {:<15} {:>9} {:>12}",
                        device.name,
                        human_bytes(budget),
                        "none"
                    );
                }
            }
        }
    }
    let _ = writeln!(
        s,
        "\n  The boundary moves with the device and barely at all with the\n  \
           budget, which is the summary of the whole manifold: **what\n  \
           decides whether you need gsplat is the ratio of record size to\n  \
           block size and the device's command rate — not how much memory\n  \
           you have.** Memory decides how many stages it takes, not\n  \
           whether it is worth doing."
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::SATA_SSD_MODEL;
    use crate::queueing::Resource;

    /// Every study runs and produces output. This is the smoke test that
    /// keeps the CLI honest when the models underneath change.
    #[test]
    fn every_study_renders() {
        for study in ALL {
            let out = study.render();
            assert!(
                out.len() > 200,
                "{} produced only {} bytes",
                study.name,
                out.len()
            );
            assert!(!out.contains("NaN"), "{} produced a NaN", study.name);
            assert!(!out.contains("inf"), "{} produced an infinity", study.name);
        }
    }

    #[test]
    fn studies_are_addressable_by_name() {
        for study in ALL {
            assert!(find(study.name).is_some());
        }
        assert!(find("no-such-study").is_none());
    }

    /// **The manifold's first claim, and it is not the obvious one.**
    ///
    /// Both costs are linear in the ordinal count at fixed memory — the
    /// naive one because it issues a command per record, the staged one
    /// because it moves a payload proportional to the record count — so
    /// the *ratio* between them does not grow with `N`. It is set by how
    /// a device's per-command positioning time compares to its per-byte
    /// streaming time, and it steps down only when the segment count
    /// crosses the fan-out and buys another stage.
    ///
    /// What changes with `N` is the absolute time, and that is the whole
    /// argument: the same 250x advantage is a curiosity at a million
    /// ordinals and the difference between a shift and a quarter at a
    /// billion.
    #[test]
    fn the_advantage_is_flat_in_the_ordinal_count_but_the_absolute_time_is_not() {
        let mut gains = Vec::new();
        let mut naive_times = Vec::new();
        for records in [1_000_000u64, 10_000_000, 100_000_000, 1_000_000_000] {
            let w = Workload {
                records,
                ..terabyte(1_024, 32 << 30)
            };
            let costs = compare(&w, &SPINNING_SATA_MODEL, &HostModel::DEFAULT);
            let naive = best_naive(&costs).seconds;
            gains.push(naive / pick(&costs, Strategy::Splat).seconds);
            naive_times.push(naive);
        }

        let low = gains.iter().cloned().fold(f64::INFINITY, f64::min);
        let high = gains.iter().cloned().fold(0.0, f64::max);
        assert!(
            high / low < 4.0,
            "the ratio should be roughly flat across three decades of N: {gains:?}"
        );
        assert!(low > 20.0, "and large throughout: {gains:?}");

        // The absolute times are what move.
        assert!(
            naive_times[0] < 2.0 * 3_600.0,
            "a million ordinals is an hour naively: {}",
            human_time(naive_times[0])
        );
        assert!(
            naive_times[3] > 30.0 * 86_400.0,
            "a billion is a month or more: {}",
            human_time(naive_times[3])
        );
        assert!(
            (naive_times[3] / naive_times[0] / 1000.0 - 1.0).abs() < 0.1,
            "and the growth is linear in N, not worse"
        );
    }

    /// The staged cost steps rather than slopes: it is flat while the
    /// stage count is, and takes a discrete jump when another stage is
    /// bought. That is `ceil(log_f(segments))` showing through.
    #[test]
    fn the_staged_cost_steps_with_the_stage_count() {
        let per_record = |records: u64| {
            let w = Workload {
                records,
                ..terabyte(1_024, 32 << 30)
            };
            let costs = compare(&w, &SPINNING_SATA_MODEL, &HostModel::DEFAULT);
            (
                w.stages(),
                pick(&costs, Strategy::Splat).seconds / records as f64,
            )
        };

        let (small_stages, small) = per_record(1_000_000);
        let (mid_stages, mid) = per_record(10_000_000);
        let (big_stages, big) = per_record(1_000_000_000);

        assert_eq!(small_stages, 0, "a gigabyte fits in a 32 GiB budget");
        assert_eq!(mid_stages, 0);
        assert_eq!(big_stages, 1, "a terabyte needs one distribution stage");
        assert!(
            (mid / small - 1.0).abs() < 0.01,
            "no new stage means no new cost per record"
        );
        assert!(
            (big / mid - 2.0).abs() < 0.2,
            "one new stage doubles it: {:.3} us against {:.3} us per record",
            big * 1e6,
            mid * 1e6
        );
    }

    /// **The second claim.** The frontier is set by the record size and
    /// the device, and is nearly independent of the memory budget —
    /// which is the opposite of what the re-scan form would predict.
    #[test]
    fn the_frontier_moves_with_the_device_and_not_with_the_budget() {
        let boundary = |device: &DeviceModel, budget: u64| -> u64 {
            let mut found = 0;
            for r in [64u64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384] {
                let w = terabyte(r, budget);
                let costs = compare(&w, device, &HostModel::cores(8));
                if pick(&costs, Strategy::Splat).seconds < best_naive(&costs).seconds {
                    found = r;
                }
            }
            found
        };

        for device in ALL_MODELS_WITH_MODERN {
            let small = boundary(device, 8 << 30);
            let large = boundary(device, 128 << 30);
            assert_eq!(
                small, large,
                "{}: a 16x budget change moved the boundary from {small} B to {large} B",
                device.name
            );
        }

        assert!(
            boundary(&SPINNING_SATA_MODEL, 32 << 30) > boundary(&NVME_MODERN_MODEL, 32 << 30),
            "a seek-bound disk should need staging at far larger records \
             than a modern NVMe does"
        );
    }

    /// **The third claim.** More than one resource can be the problem,
    /// and which one it is changes along the record axis rather than
    /// being a property of the strategy.
    #[test]
    fn the_binding_resource_changes_along_the_record_axis() {
        let small = terabyte(512, 32 << 30);
        let large = terabyte(16_384, 32 << 30);
        let gather = |w: &Workload| {
            pick(
                &compare(w, &NVME_MODERN_MODEL, &HostModel::cores(8)),
                Strategy::Gather,
            )
            .pegged()
        };
        assert!(
            gather(&small).contains(&Resource::Controller),
            "sub-block records exhaust command processing"
        );
        assert!(
            gather(&large).contains(&Resource::Bandwidth),
            "large records exhaust bandwidth instead"
        );
        assert!(!gather(&large).contains(&Resource::Controller));
    }

    /// A staged rewrite converts the problem into a bandwidth problem, so
    /// bandwidth is what it pegs — on every device modelled.
    #[test]
    fn the_staged_rewrite_is_always_bandwidth_bound() {
        for device in ALL_MODELS_WITH_MODERN {
            for record_bytes in [512u64, 4_096, 65_536] {
                let w = terabyte(record_bytes, 32 << 30);
                let costs = compare(&w, device, &HostModel::cores(8));
                let staged = pick(&costs, Strategy::Splat);
                assert!(
                    staged.pegged().contains(&Resource::Bandwidth),
                    "{} at {record_bytes} B: pegged {}",
                    device.name,
                    pegged_list(staged)
                );
            }
        }
    }

    /// **The corner set has to straddle the decision.** A study whose
    /// every configuration favours the same strategy is an advertisement,
    /// not a study. This asserts the corners actually bracket the
    /// boundary — some of them are won by a naive gather.
    #[test]
    fn the_corner_set_contains_configurations_that_naive_wins() {
        let host = HostModel::cores(8);
        let mut staged_wins = 0;
        let mut naive_wins = 0;
        for corner in corners() {
            let costs = compare(&corner.workload, corner.device, &host);
            if pick(&costs, Strategy::Splat).seconds < best_naive(&costs).seconds {
                staged_wins += 1;
            } else {
                naive_wins += 1;
            }
        }
        assert!(
            staged_wins >= 8,
            "only {staged_wins} corners favour staging"
        );
        assert!(
            naive_wins >= 2,
            "a corner set with no naive wins is not bracketing anything"
        );
    }

    /// **Squeezing the fan-out is the one way to make staging expensive.**
    ///
    /// Everything else that hurts a naive rewrite leaves the staged one
    /// alone. This is the exception, and it is worth knowing because it
    /// is entirely a configuration choice: a container so large that
    /// memory holds few buffers forces `ceil(log_f(segments))` upward, and
    /// each stage is a whole extra pass over the payload in both
    /// directions.
    #[test]
    fn only_a_collapsed_fan_out_makes_the_staged_form_expensive() {
        let host = HostModel::cores(8);
        let base = terabyte(1_024, 32 << 30);
        let baseline = pick(
            &compare(&base, &NVME_CONSUMER_MODEL, &host),
            Strategy::Splat,
        )
        .seconds;

        // Memory, record size and device all leave the staged cost alone
        // or improve it; none of them multiplies it.
        for altered in [
            Workload {
                budget_bytes: 1 << 30,
                ..base
            },
            terabyte(128, 32 << 30),
            terabyte(65_536, 32 << 30),
        ] {
            let staged = pick(
                &compare(&altered, &NVME_CONSUMER_MODEL, &host),
                Strategy::Splat,
            )
            .seconds;
            assert!(
                staged <= baseline * 1.05,
                "staged cost should not rise: {} against {}",
                human_time(staged),
                human_time(baseline)
            );
        }

        // A collapsed fan-out does multiply it, and by the stage count.
        let squeezed = Workload {
            container_bytes: 256 << 20,
            budget_bytes: 1 << 30,
            ..base
        };
        assert_eq!(
            squeezed.fanout(),
            4,
            "a 1 GiB budget holds four 256 MiB buffers"
        );
        assert_eq!(
            squeezed.stages(),
            5,
            "and 1000 segments then need five stages"
        );
        let staged = pick(
            &compare(&squeezed, &NVME_CONSUMER_MODEL, &host),
            Strategy::Splat,
        )
        .seconds;
        assert!(
            staged > baseline * 2.0,
            "five stages against one should cost about threefold: {} against {}",
            human_time(staged),
            human_time(baseline)
        );
    }

    /// **A scattered reader evicts the writer's dirty pages.**
    ///
    /// Both of these rewrites write their output in order, so neither is
    /// a scattered *writer*. What separates them is the read side: a
    /// naive gather's reads land all over the source, each one claiming
    /// a frame, and the frames they claim are the ones holding output
    /// pages the flusher has not written yet. Those pages then go out one
    /// at a time on the allocation path instead of as extents.
    ///
    /// This is read/write interference inside the page cache rather than
    /// on the device, and it is a second, independent reason scattered
    /// reads cost more than the reads themselves.
    #[test]
    fn a_scattered_reader_forces_the_writers_pages_out_by_eviction() {
        use crate::algo::{Rewrite, naive::NaiveGather, staged::StagedSplat};
        use crate::cache::CacheConfig;
        use crate::io::{self, RunConfig, hw, sched};
        use crate::model::{Geometry, Map};
        use crate::price;

        let geometry = Geometry {
            records: 60_000,
            record_bytes: 512,
            container_bytes: 128 * 1024,
        };
        let map = Map::shuffled(geometry.records, 0xB0FF);
        let budget = geometry.payload_bytes() / 8;
        let cache = CacheConfig::new(geometry.payload_bytes() / 4, 4_096);

        let run = |trace: &crate::model::Trace| {
            let span = geometry.payload_bytes() * 2 + price::spill_extent_bytes(trace);
            io::run(
                &hw::SATA_SSD_HW,
                &mut sched::Noop::default(),
                &mut io::Recorded::new(price::accesses_of(trace)),
                RunConfig::buffered(64, span, cache),
            )
        };

        let scattered = run(&NaiveGather.run(geometry, &map, budget).1);
        let staged = run(&StagedSplat::new().run(geometry, &map, budget).1);

        assert!(
            scattered.eviction_writebacks > scattered.flusher_writebacks,
            "the gather's output pages go out the expensive way: \
             {} by eviction against {} by the flusher",
            scattered.eviction_writebacks,
            scattered.flusher_writebacks
        );
        assert!(
            staged.flusher_writebacks > staged.eviction_writebacks * 10,
            "a staged rewrite's go out the cheap way: \
             {} by the flusher against {} by eviction",
            staged.flusher_writebacks,
            staged.eviction_writebacks
        );
    }

    /// **And "add RAM" only works once RAM is most of the payload.**
    ///
    /// The reflex answer to eviction writebacks is a bigger page cache,
    /// and it does eventually work — but the boundary is at a cache
    /// comparable to the payload, which is precisely the regime the whole
    /// problem is defined to exclude. Below it, multiplying memory leaves
    /// the pages going out the same way; above it the rewrite would have
    /// fit in memory and there was nothing to solve.
    ///
    /// Asserting where the boundary *is* is more useful than asserting
    /// that memory never helps, which is not true.
    #[test]
    fn more_memory_helps_only_once_the_cache_approaches_the_payload() {
        use crate::algo::{Rewrite, naive::NaiveGather};
        use crate::cache::CacheConfig;
        use crate::io::{self, RunConfig, hw, sched};
        use crate::model::{Geometry, Map};
        use crate::price;

        let geometry = Geometry {
            records: 60_000,
            record_bytes: 512,
            container_bytes: 128 * 1024,
        };
        let map = Map::shuffled(geometry.records, 0xB166);
        let budget = geometry.payload_bytes() / 8;
        let trace = NaiveGather.run(geometry, &map, budget).1;
        let span = geometry.payload_bytes() * 2 + price::spill_extent_bytes(&trace);

        let at = |divisor: u64| {
            io::run(
                &hw::SATA_SSD_HW,
                &mut sched::Noop::default(),
                &mut io::Recorded::new(price::accesses_of(&trace)),
                RunConfig::buffered(
                    64,
                    span,
                    CacheConfig::new(geometry.payload_bytes() / divisor, 4_096),
                ),
            )
        };
        // A cache at a sixteenth, an eighth and a quarter of the payload
        // — the realistic band — all keep the pages on the eviction path.
        for divisor in [16u64, 8, 4] {
            let stats = at(divisor);
            assert!(
                stats.eviction_writebacks > stats.flusher_writebacks,
                "cache at 1/{divisor} of payload: {} by eviction against \
                 {} by the flusher",
                stats.eviction_writebacks,
                stats.flusher_writebacks
            );
        }

        // At half the payload the flusher keeps ahead and the eviction
        // path empties — the boundary, stated rather than assumed.
        let generous = at(2);
        assert_eq!(
            generous.eviction_writebacks, 0,
            "with the cache at half the payload nothing is evicted dirty"
        );
        assert!(generous.flusher_writebacks > 0);
    }

    /// The SATA SSD and consumer NVMe are in the corpus for a reason:
    /// they sit on opposite sides of the boundary for a common shape.
    #[test]
    fn the_devices_disagree_about_a_kilobyte_record() {
        let w = terabyte(1_024, 32 << 30);
        let advantage = |d: &DeviceModel| {
            let costs = compare(&w, d, &HostModel::cores(8));
            best_naive(&costs).seconds / pick(&costs, Strategy::Splat).seconds
        };
        assert!(advantage(&SPINNING_SATA_MODEL) > advantage(&SATA_SSD_MODEL));
        assert!(advantage(&SATA_SSD_MODEL) > advantage(&NVME_MODERN_MODEL));
    }
}
