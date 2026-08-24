// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Turning a trace into a predicted time against a measured device.
//!
//! [`crate::model::Metrics`] counts operations. A [`Regime`] says what
//! operations cost. Together they give a number that can be argued with.
//!
//! The read side is derived from the trace rather than assumed per
//! algorithm, which matters because it is the mechanism the documents
//! actually claim: **monotone access is what permits container-sized
//! fetches.** An algorithm whose trace records no backward steps can have
//! its reads coalesced into whole containers, so it pays the container
//! block price; one that jumps around cannot, so it pays per record at
//! the record block price. [`read_plan`] reads that distinction straight
//! off `backward_steps`, so an algorithm that lost monotonicity would be
//! repriced automatically instead of keeping a favourable label.
//!
//! **Two honest limits.** First, the perfscripts sweep has no random-write
//! workload, so writes are priced sequentially for every algorithm. That
//! understates scatter-style writers, whose cost here is a lower bound —
//! stated rather than hidden, and acceptable because they lose even at
//! that bound. [`Priced::write_ranges`] carries the count that would drive
//! the correction if the measurement existed.
//!
//! Second, reads and writes are priced additively, as if they did not
//! interfere. The regime's own contention sweep shows when that is fair:
//! under a rate cap the jobs split a near-constant bandwidth pool, so
//! adding the terms is about right; uncapped, a sequential writer starves
//! a concurrent random reader by more than two orders of magnitude and
//! the additive model is worthless. See
//! [`Regime::starvation_ratio`](crate::regime::Regime::starvation_ratio).
//! Additive pricing is therefore a statement about a *governed* pipeline.

use crate::model::Metrics;
use crate::regime::Regime;

/// How an algorithm's reads land on the device, derived from its trace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadPlan {
    pub ops: u64,
    pub block_bytes: u64,
    /// Whether ascending access permitted container-sized fetches.
    pub coalesced: bool,
}

/// Derive the read plan from what the trace actually did.
pub fn read_plan(m: &Metrics) -> ReadPlan {
    if m.backward_steps == 0 {
        ReadPlan {
            ops: m.container_touches,
            block_bytes: m.geometry.container_bytes,
            coalesced: true,
        }
    } else {
        ReadPlan {
            ops: m.record_reads,
            block_bytes: m.geometry.record_bytes,
            coalesced: false,
        }
    }
}

/// A trace priced against a device.
#[derive(Debug, Clone)]
pub struct Priced {
    pub algo: &'static str,
    pub regime: &'static str,
    pub plan: ReadPlan,
    pub read_seconds: f64,
    /// Sequential-write cost. A lower bound for scattered writers.
    pub write_seconds: f64,
    /// Non-contiguous write runs, for judging how loose that bound is.
    pub write_ranges: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

impl Priced {
    pub fn seconds(&self) -> f64 {
        self.read_seconds + self.write_seconds
    }

    /// Effective read throughput — the number the device curve is really
    /// deciding.
    pub fn read_mib_per_s(&self) -> f64 {
        if self.read_seconds == 0.0 {
            return 0.0;
        }
        self.bytes_read as f64 / self.read_seconds / (1024.0 * 1024.0)
    }
}

/// Price one traced algorithm against one device.
///
/// An ordered reader gets the **streaming bound**: since its accesses
/// ascend, it can always choose to read the source straight through and
/// discard what it does not need, so no pass costs more than a sequential
/// scan. A reader that jumps has no such option — it cannot know what to
/// skip — and pays per access.
pub fn price(algo: &'static str, m: &Metrics, regime: &Regime) -> Priced {
    let plan = read_plan(m);
    let bytes_read = plan.ops * plan.block_bytes;
    let bytes_written = m.bytes_written();

    let seeking = regime.random_read_seconds(plan.ops, plan.block_bytes);
    let read_seconds = if plan.coalesced {
        let passes = m.passes.max(1);
        let scan = regime.sequential_read_seconds(m.geometry.payload_bytes());
        seeking.min(scan * passes as f64)
    } else {
        seeking
    };

    Priced {
        algo,
        regime: regime.name,
        plan,
        read_seconds,
        write_seconds: regime.sequential_write_seconds(bytes_written),
        write_ranges: m.write_ranges,
        bytes_read,
        bytes_written,
    }
}

/// A cost derived end to end by simulation, with no table lookup and no
/// assumed container.
///
/// The chain is: the algorithm emits a trace; the trace is replayed
/// through an LRU page cache of a stated size and page granularity; the
/// misses that survive are handed to a device model whose parameters were
/// validated against measured sweeps. Every step is a mechanism rather
/// than a fitted constant, so a wrong assumption anywhere shows up as a
/// number that disagrees with the measurements.
///
/// It also removes the container from the model's vocabulary. The read
/// block size is the *page* size, and how many records come along with
/// each fault is whatever the geometry and the page size imply.
#[derive(Debug, Clone, Copy)]
pub struct Simulated {
    pub algo: &'static str,
    pub device: &'static str,
    pub cache: crate::cache::CacheStats,
    pub ram_bytes: u64,
    pub read_seconds: f64,
    pub write_seconds: f64,
    pub device_read_bytes: u64,
}

impl Simulated {
    pub fn seconds(&self) -> f64 {
        self.read_seconds + self.write_seconds
    }

    /// Device traffic divided by useful payload — amplification as
    /// simulated rather than as predicted by formula.
    pub fn amplification(&self, payload_bytes: u64) -> f64 {
        if payload_bytes == 0 {
            0.0
        } else {
            self.device_read_bytes as f64 / payload_bytes as f64
        }
    }
}

/// Price a trace by simulating the cache and the device.
pub fn simulate(
    algo: &'static str,
    trace: &crate::model::Trace,
    model: &crate::device::DeviceModel,
    cache: crate::cache::CacheConfig,
) -> Simulated {
    let stats = crate::cache::replay(trace, cache);
    let device_read_bytes = stats.read_bytes_from_device();
    let metrics = trace.metrics();
    let bytes_written = metrics.bytes_written();

    // Same streaming bound as the table-driven path: an ascending reader
    // can stream instead of seeking, so no pass costs more than a scan.
    let seeking = model.random_read_seconds(stats.read_misses, stats.page_bytes);
    let read_seconds = if metrics.backward_steps == 0 {
        let scan = model.sequential_seconds(metrics.geometry.payload_bytes());
        seeking.min(scan * metrics.passes.max(1) as f64)
    } else {
        seeking
    };

    Simulated {
        algo,
        device: model.name,
        cache: stats,
        ram_bytes: cache.ram_bytes,
        read_seconds,
        // Output is written in order, so it costs the model's rate at a
        // full page — the best case the device offers.
        write_seconds: model.sequential_seconds(bytes_written),
        device_read_bytes,
    }
}

/// The memory budget at which ordering starts to pay, for one device.
///
/// Ordering is not free: it buys large sequential-ish fetches at the cost
/// of reading each container once per pass. Whether that trade wins
/// depends on how many passes the budget forces, so "is gsplat better"
/// has no answer without a budget and a device. This finds the smallest
/// budget, among those offered, where it is.
pub fn crossover_budget(
    geometry: crate::model::Geometry,
    map: &crate::model::Map,
    regime: &Regime,
    budgets: &[u64],
) -> Option<u64> {
    use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};

    budgets.iter().copied().find(|&budget| {
        let g = Gsplat::new().run(geometry, map, budget).1.metrics();
        let n = NaiveGather.run(geometry, map, budget).1.metrics();
        price("gsplat", &g, regime).seconds() < price("naive-gather", &n, regime).seconds()
    })
}

/// Render a regime-by-budget comparison as a table.
pub fn render_regime_sweep(
    geometry: crate::model::Geometry,
    map: &crate::model::Map,
    regime: &Regime,
    budgets: &[u64],
) -> String {
    use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};
    use std::fmt::Write as _;

    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n{} — {} ({} records × {} B, {} B containers)",
        regime.name,
        regime.device,
        geometry.records,
        geometry.record_bytes,
        geometry.container_bytes
    );
    let _ = writeln!(
        s,
        "  seq read {:.0} MB/s · random {} B {:.0} MB/s · efficient block {:?}",
        regime.seq_read.mib_per_s(),
        geometry.record_bytes,
        regime
            .random_at(geometry.record_bytes)
            .bandwidth
            .mib_per_s(),
        regime.efficient_block(0.95)
    );
    let _ = writeln!(
        s,
        "\n  {:>6}  {:>6}  {:>8}  {:>10}  {:>10}  {:>8}",
        "budget", "passes", "amp", "gsplat s", "naive s", "speedup"
    );

    for &budget in budgets {
        let g = Gsplat::new().run(geometry, map, budget).1.metrics();
        let n = NaiveGather.run(geometry, map, budget).1.metrics();
        let pg = price("gsplat", &g, regime);
        let pn = price("naive-gather", &n, regime);
        let _ = writeln!(
            s,
            "  {:>5}%  {:>6}  {:>8.2}  {:>10.3}  {:>10.3}  {:>7.2}×",
            budget * 100 / geometry.payload_bytes().max(1),
            g.passes,
            g.amplification(),
            pg.seconds(),
            pn.seconds(),
            pn.seconds() / pg.seconds()
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};
    use crate::model::{Geometry, Map};
    use crate::regime::{ALL, NVME_CONSUMER, SATA_SSD, SPINNING_SATA};

    /// A geometry in the shape of a real vector dataset: 100k records of
    /// 1536-dimensional f32 vectors, stored 64 KiB at a time.
    fn vectors() -> Geometry {
        Geometry {
            records: 100_000,
            record_bytes: 1_536 * 4,
            container_bytes: 65_536,
        }
    }

    fn traced(geometry: Geometry) -> (Metrics, Metrics) {
        let map = Map::shuffled(geometry.records, 0xC0FFEE);
        let budget = geometry.record_bytes * 4_096;
        let g = Gsplat::new().run(geometry, &map, budget).1.metrics();
        let n = NaiveGather.run(geometry, &map, budget).1.metrics();
        (g, n)
    }

    /// The pricing split follows from monotonicity, not from a label.
    #[test]
    fn coalescing_is_earned_by_ascending_access() {
        let (g, n) = traced(vectors());
        assert!(
            read_plan(&g).coalesced,
            "gsplat ascends, so it fetches containers"
        );
        assert!(
            !read_plan(&n).coalesced,
            "naive gather jumps, so it fetches records"
        );
        assert_eq!(read_plan(&g).block_bytes, 65_536);
        assert_eq!(read_plan(&n).block_bytes, 6_144);
    }

    /// Budgets from 1% to 100% of the payload.
    fn budget_ladder(g: Geometry) -> Vec<u64> {
        let payload = g.payload_bytes();
        (0..=20).map(|i| payload * (1 + i * 5) / 100).collect()
    }

    /// Whether ordering pays is a question about the memory budget, and
    /// each device answers it differently. The spinning disk pays for
    /// ordering at the smallest budget it is offered; flash demands a
    /// far larger one before the trade turns favourable.
    ///
    /// This is the single most important thing the measured regimes
    /// added: without them, "gsplat beats naive gather" reads as
    /// unconditional. It is not.
    #[test]
    fn the_budget_at_which_ordering_pays_depends_on_the_device() {
        let geo = vectors();
        let map = Map::shuffled(geo.records, 0xC0FFEE);
        let ladder = budget_ladder(geo);

        let hdd = crossover_budget(geo, &map, &SPINNING_SATA, &ladder)
            .expect("ordering must pay somewhere on a spinning disk");
        let nvme = crossover_budget(geo, &map, &NVME_CONSUMER, &ladder)
            .expect("ordering must pay somewhere on NVMe");

        assert!(
            hdd < nvme,
            "the disk should need less memory to justify ordering: {} vs {} bytes",
            hdd,
            nvme
        );
    }

    /// Records small relative to the device's efficient block — the case
    /// ordering exists for.
    fn small_records() -> Geometry {
        Geometry {
            records: 2_000_000,
            record_bytes: 128,
            container_bytes: 65_536,
        }
    }

    /// On flash, a starved budget makes ordering a liability: the pass
    /// count drives amplification past anything the larger block size can
    /// recover.
    ///
    /// The spinning disk is deliberately excluded, and the next test says
    /// why.
    #[test]
    fn on_flash_ordering_loses_when_the_budget_forces_many_passes() {
        let geo = vectors();
        let map = Map::shuffled(geo.records, 0xC0FFEE);
        let starved = geo.payload_bytes() / 25;

        for regime in [&SATA_SSD, &NVME_CONSUMER] {
            let g = Gsplat::new().run(geo, &map, starved).1.metrics();
            let n = NaiveGather.run(geo, &map, starved).1.metrics();
            assert!(
                g.passes >= 20,
                "this budget should force many passes, got {}",
                g.passes
            );
            assert!(
                price("gsplat", &g, regime).seconds() > price("naive-gather", &n, regime).seconds(),
                "{}: ordering should lose at {} passes",
                regime.name,
                g.passes
            );
        }
    }

    /// **The spinning disk never punishes ordering**, even at 25 passes
    /// reading eight times the payload. Its random rate is flat in
    /// operations per second from 512 B to 16 KiB, so what it charges for
    /// is arrivals, not bytes — and ordering always issues fewer, larger
    /// arrivals. Any statement of the form "too many passes and ordering
    /// stops paying" is a statement about flash.
    #[test]
    fn a_seek_bound_disk_rewards_ordering_at_any_pass_count() {
        let geo = vectors();
        let map = Map::shuffled(geo.records, 0xC0FFEE);

        for divisor in [2u64, 10, 25, 50] {
            let budget = geo.payload_bytes() / divisor;
            let g = Gsplat::new().run(geo, &map, budget).1.metrics();
            let n = NaiveGather.run(geo, &map, budget).1.metrics();
            let pg = price("gsplat", &g, &SPINNING_SATA);
            let pn = price("naive-gather", &n, &SPINNING_SATA);
            assert!(
                pg.bytes_read > pn.bytes_read,
                "ordering should be reading far more bytes at {} passes",
                g.passes
            );
            assert!(
                pg.seconds() < pn.seconds(),
                "yet still win at {} passes ({:.0}s vs {:.0}s)",
                g.passes,
                pg.seconds(),
                pn.seconds()
            );
        }
    }

    /// **Ordering has nothing to sell when the record is already an
    /// efficient block.** At 6 KiB, both flash devices serve random reads
    /// at over half their sequential rate, so there is at most a 2×
    /// headroom to win and even a two-pass run spends more than that.
    ///
    /// This is a real limit of the technique, not a tuning problem, and
    /// it is invisible without measured curves — the amplification
    /// formula alone would happily recommend ordering here.
    #[test]
    fn ordering_cannot_pay_on_flash_when_records_are_already_large() {
        let geo = vectors();
        let map = Map::shuffled(geo.records, 0xC0FFEE);
        let ample = geo.payload_bytes() / 2;
        let g = Gsplat::new().run(geo, &map, ample).1.metrics();
        let n = NaiveGather.run(geo, &map, ample).1.metrics();
        assert_eq!(g.passes, 2, "the most favourable multi-pass case there is");

        let headroom = SATA_SSD.random_penalty(geo.record_bytes);
        assert!(
            headroom < 2.0,
            "only {headroom:.2}× to win at this record size"
        );
        assert!(
            price("gsplat", &g, &SATA_SSD).seconds()
                > price("naive-gather", &n, &SATA_SSD).seconds(),
            "so two passes cannot pay for themselves"
        );
    }

    /// Given small records and enough memory to keep the pass count low,
    /// ordering wins everywhere — and by far the largest margin on the
    /// device with the harshest random-access penalty.
    #[test]
    fn ordering_wins_everywhere_on_small_records_once_passes_are_few() {
        let geo = small_records();
        let map = Map::shuffled(geo.records, 0xC0FFEE);
        let ample = geo.payload_bytes() / 2;

        let g = Gsplat::new().run(geo, &map, ample).1.metrics();
        let n = NaiveGather.run(geo, &map, ample).1.metrics();
        assert_eq!(g.passes, 2, "expected a two-pass run, got {}", g.passes);

        let mut speedups = Vec::new();
        for regime in ALL {
            let pg = price("gsplat", &g, regime);
            let pn = price("naive-gather", &n, regime);
            let speedup = pn.seconds() / pg.seconds();
            assert!(
                speedup > 10.0,
                "{}: ordering should win big, got {speedup:.2}×",
                regime.name
            );
            speedups.push((regime.name, speedup));
        }

        let hdd = speedups
            .iter()
            .find(|(n, _)| *n == "spinning-sata")
            .unwrap()
            .1;
        let nvme = speedups
            .iter()
            .find(|(n, _)| *n == "nvme-consumer")
            .unwrap()
            .1;
        assert!(
            hdd > nvme * 2.0,
            "the disk should gain far more from ordering than NVMe: {hdd:.1}× vs {nvme:.1}×"
        );
    }

    /// Two independent paths to the same number: the table-driven price,
    /// which looks costs up in the measured sweep, and the fully
    /// simulated one, which replays the trace through a page cache and a
    /// parametric device. They share no code and no constants, so
    /// agreement between them is evidence and not tautology.
    #[test]
    fn the_table_and_the_simulation_agree_on_cost() {
        use crate::cache::CacheConfig;
        use crate::device::paired;

        let geo = vectors();
        let map = Map::shuffled(geo.records, 0xC0FFEE);
        let budget = geo.payload_bytes() / 4;
        let (_, trace) = Gsplat::new().run(geo, &map, budget);
        let metrics = trace.metrics();

        for (model, regime) in paired() {
            let table = price("gsplat", &metrics, regime);
            // One page of retention at container granularity is precisely
            // what `container_touches` counts, so this is the
            // configuration in which the two paths describe the same run.
            let sim = simulate(
                "gsplat",
                &trace,
                &model,
                CacheConfig::single_page(geo.container_bytes),
            );

            let ratio = sim.read_seconds / table.read_seconds;
            assert!(
                (0.7..1.45).contains(&ratio),
                "{}: simulated {:.3}s vs table {:.3}s ({ratio:.2}×)",
                model.name,
                sim.read_seconds,
                table.read_seconds
            );
        }
    }

    /// Page size cuts both ways, and which way depends entirely on
    /// whether access is ordered. A large page is a gift to the algorithm
    /// that reads ascending — every fault delivers records it is about to
    /// want — and a tax on the one that jumps, which discards almost all
    /// of what it faulted in.
    ///
    /// This is the clearest statement of what ordering actually buys, and
    /// it needs no device constants at all: it is visible in the byte
    /// counts.
    #[test]
    fn page_size_rewards_ordered_access_and_punishes_random_access() {
        use crate::cache::{CacheConfig, replay};

        let geo = Geometry {
            records: 200_000,
            record_bytes: 512,
            container_bytes: 65_536,
        };
        let map = Map::shuffled(geo.records, 5);
        let budget = geo.payload_bytes() / 2;
        let (_, ordered) = Gsplat::new().run(geo, &map, budget);
        let (_, random) = NaiveGather.run(geo, &map, budget);

        let small = 4_096;
        let large = 65_536;
        let ram = geo.payload_bytes() / 16;

        let ordered_small = replay(&ordered, CacheConfig::new(ram, small));
        let ordered_large = replay(&ordered, CacheConfig::new(ram, large));
        let random_small = replay(&random, CacheConfig::new(ram, small));
        let random_large = replay(&random, CacheConfig::new(ram, large));

        // The ordered reader faults fewer times *and* moves no more than
        // proportionally more bytes, because the extra bytes are records
        // it goes on to use.
        assert!(ordered_large.read_misses < ordered_small.read_misses / 4);

        // The random reader also faults less often, but every fault now
        // drags in 128 records to use one.
        let random_growth = random_large.read_bytes_from_device() as f64
            / random_small.read_bytes_from_device() as f64;
        let ordered_growth = ordered_large.read_bytes_from_device() as f64
            / ordered_small.read_bytes_from_device() as f64;
        assert!(
            random_growth > ordered_growth * 2.0,
            "growing the page cost the random reader {random_growth:.2}× the traffic \
             and the ordered reader only {ordered_growth:.2}×"
        );
    }

    /// RAM is not a dial that trades smoothly against time here. Below
    /// the whole source it buys nothing, because each pass is a cyclic
    /// scan; at the whole source the repeat passes vanish at once.
    #[test]
    fn ram_buys_nothing_until_it_holds_the_whole_source() {
        use crate::cache::CacheConfig;
        use crate::device::NVME_CONSUMER_MODEL;

        let geo = Geometry {
            records: 40_000,
            record_bytes: 512,
            container_bytes: 65_536,
        };
        let map = Map::shuffled(geo.records, 11);
        let payload = geo.payload_bytes();
        let (_, trace) = Gsplat::new().run(geo, &map, payload / 6);

        let at = |ram: u64| {
            simulate(
                "gsplat",
                &trace,
                &NVME_CONSUMER_MODEL,
                CacheConfig::new(ram, 4_096),
            )
            .read_seconds
        };

        let quarter = at(payload / 4);
        let half = at(payload / 2);
        let whole = at(payload * 2);

        assert!(
            (quarter - half).abs() / quarter < 0.01,
            "doubling the cache to half the source changes nothing: {quarter:.3}s vs {half:.3}s"
        );
        // Holding the whole source does help — but by less than the miss
        // counts imply, because the streaming bound had already capped
        // what a starved cache could cost. RAM competes with sequential
        // bandwidth, not with the seek storm the miss count suggests.
        assert!(
            whole < quarter * 0.8,
            "holding the source must help: {whole:.4}s vs {quarter:.4}s"
        );
        assert!(
            whole > quarter * 0.4,
            "but not dramatically, since streaming already bounded the alternative"
        );
    }

    /// Reading more bytes can still take less time — but only when the
    /// device charges enough per operation relative to the record size.
    #[test]
    fn gsplat_reads_more_bytes_and_still_finishes_sooner() {
        let geo = small_records();
        let map = Map::shuffled(geo.records, 0xC0FFEE);
        let ample = geo.payload_bytes() / 2;
        let g = Gsplat::new().run(geo, &map, ample).1.metrics();
        let n = NaiveGather.run(geo, &map, ample).1.metrics();

        for regime in ALL {
            let pg = price("gsplat", &g, regime);
            let pn = price("naive-gather", &n, regime);
            assert!(pg.bytes_read > pn.bytes_read, "{}: more bytes", regime.name);
            assert!(
                pg.read_seconds < pn.read_seconds,
                "{}: yet sooner ({:.2}s vs {:.2}s)",
                regime.name,
                pg.read_seconds,
                pn.read_seconds
            );
        }
    }
}

#[cfg(test)]
mod dump {
    use super::*;
    use crate::model::{Geometry, Map};
    use crate::regime::ALL;

    #[test]
    #[ignore = "diagnostic dump, not an assertion"]
    fn print_matrix() {
        for geo in [
            Geometry {
                records: 100_000,
                record_bytes: 6_144,
                container_bytes: 65_536,
            },
            Geometry {
                records: 2_000_000,
                record_bytes: 128,
                container_bytes: 65_536,
            },
        ] {
            let map = Map::shuffled(geo.records, 0xC0FFEE);
            let payload = geo.payload_bytes();
            let budgets: Vec<u64> = [1u64, 2, 5, 10, 25, 50]
                .iter()
                .map(|p| payload * p / 100)
                .collect();
            for r in ALL {
                print!("{}", render_regime_sweep(geo, &map, r, &budgets));
            }
        }
    }
}
