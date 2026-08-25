// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Validation against measurement, scored the way the literature scores
//! it.
//!
//! A simulator that reproduces one curve is anecdote. This module runs
//! the full cross-product — three devices, every block size from 512 B to
//! 1 MiB, throughput and latency — and reports mean absolute percentage
//! error per metric, which is what published storage simulators report.
//!
//! ## What counts as good
//!
//! The bar, from the papers that state one:
//!
//! | Simulator | Reported accuracy |
//! |---|---|
//! | [MQSim](https://www.usenix.org/conference/fast18/presentation/tavakkol) (FAST '18) | within **6–18%** of four real SSDs |
//! | [SimpleSSD](https://arxiv.org/pdf/1705.06419) | up to **28%** throughput, **36%** latency |
//! | [Generative black-box models](https://arxiv.org/pdf/2307.02073) | **4–10%** IOPS, **3–16%** latency |
//!
//! Those numbers are not all measuring the same thing, and it matters.
//! MQSim's 6–18% is a *range across devices* of its throughput agreement;
//! SimpleSSD's figures are worst-case deviations. Comparing a mean
//! absolute percentage error against a worst-case bound would flatter
//! this model, so [`Scorecard`] reports both MAPE and worst case and the
//! comparison below uses like for like.
//!
//! ## What is and is not a prediction here
//!
//! Being explicit, because "validated" is easy to overclaim:
//!
//! - **Throughput is a prediction.** The device parameters were fitted to
//!   the random-read curve, so agreement there is a fit, not a test. But
//!   sequential throughput, the contention sweep, and the latency
//!   distribution all fall out of parameters chosen without reference to
//!   them.
//! - **Latency shape is partly calibrated.** The NAND page-type spread in
//!   [`crate::io::hw::ReadVariation`] and the disk's
//!   `rotational_awareness` were both fitted against measured
//!   percentiles. Those are calibrated outputs. The *means* they produce
//!   are not — read variation is mean-preserving by construction.
//! - **The write path is validated only sequentially.** The corpus has a
//!   sequential-write job for every device — 1 MiB blocks at
//!   `iodepth=10`, with throughput and latency — and
//!   [`score_sequential_write`] checks against it. There is no
//!   random-write workload anywhere in the corpus, so garbage collection
//!   and write amplification remain asserted rather than measured.

use crate::io::{self, hw::Hardware};
use crate::regime::{self, Regime};

/// One metric's agreement with measurement.
#[derive(Debug, Clone, Copy, Default)]
pub struct Score {
    pub name: &'static str,
    pub samples: usize,
    /// Mean absolute percentage error.
    pub mape: f64,
    /// Worst absolute percentage error.
    pub worst: f64,
    /// Mean signed error, so systematic bias is visible.
    pub bias: f64,
}

impl Score {
    fn from(name: &'static str, pairs: &[(f64, f64)]) -> Score {
        let usable: Vec<(f64, f64)> = pairs
            .iter()
            .copied()
            .filter(|(_, measured)| *measured > 0.0)
            .collect();
        if usable.is_empty() {
            return Score {
                name,
                ..Score::default()
            };
        }
        let errors: Vec<f64> = usable
            .iter()
            .map(|(sim, measured)| (sim - measured) / measured)
            .collect();
        Score {
            name,
            samples: usable.len(),
            mape: errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len() as f64,
            worst: errors.iter().map(|e| e.abs()).fold(0.0, f64::max),
            bias: errors.iter().sum::<f64>() / errors.len() as f64,
        }
    }
}

/// Every metric, for one device or for the whole set.
#[derive(Debug, Clone)]
pub struct Scorecard {
    pub subject: String,
    pub throughput: Score,
    pub mean_latency: Score,
    pub p50: Score,
    pub p95: Score,
    pub p99: Score,
    pub tail_ratio: Score,
}

impl Scorecard {
    /// The worst MAPE across metrics — the number to quote when a single
    /// figure is wanted, because quoting the best one is not validation.
    pub fn worst_mape(&self) -> f64 {
        [
            self.throughput.mape,
            self.mean_latency.mape,
            self.p50.mape,
            self.p95.mape,
            self.p99.mape,
        ]
        .into_iter()
        .fold(0.0, f64::max)
    }

    pub fn scores(&self) -> [Score; 6] {
        [
            self.throughput,
            self.mean_latency,
            self.p50,
            self.p95,
            self.p99,
            self.tail_ratio,
        ]
    }
}

/// How many requests to simulate for a point. Slow devices need fewer to
/// reach steady state, and simulating a spinning disk for as many
/// requests as an NVMe drive is simply wasted work.
fn sample_count(hardware: &Hardware, block_bytes: u64) -> u64 {
    let base = if hardware.name == "spinning-sata" {
        3_000
    } else {
        8_000
    };
    if block_bytes >= 1 << 20 {
        base / 10
    } else {
        base
    }
}

/// Validate one device against its measured sweeps.
pub fn score_device(hardware: &Hardware, regime: &Regime) -> Scorecard {
    let latency = regime::measured_latency(hardware.name);

    let mut throughput = Vec::new();
    let mut mean_lat = Vec::new();
    let mut p50 = Vec::new();
    let mut p95 = Vec::new();
    let mut p99 = Vec::new();
    let mut tails = Vec::new();

    for point in regime
        .random_read
        .iter()
        .filter(|p| p.block_bytes <= 1 << 20)
    {
        let n = sample_count(hardware, point.block_bytes);
        let result = io::fio_like_detailed(hardware, point.block_bytes, n);
        throughput.push((result.total.iops(), point.iops as f64));

        if let Some(m) = latency.iter().find(|l| l.block_bytes == point.block_bytes) {
            let s = result.latency.summary().micros();
            mean_lat.push((s.mean, m.mean_us));
            p50.push((s.p50, m.p50_us));
            p95.push((s.p95, m.p95_us));
            p99.push((s.p99, m.p99_us));
            tails.push((s.tail_ratio(), m.tail_ratio()));
        }
    }

    Scorecard {
        subject: hardware.name.to_string(),
        throughput: Score::from("throughput", &throughput),
        mean_latency: Score::from("mean latency", &mean_lat),
        p50: Score::from("p50", &p50),
        p95: Score::from("p95", &p95),
        p99: Score::from("p99", &p99),
        tail_ratio: Score::from("p99/p50", &tails),
    }
}

/// Sequential-write agreement, which the corpus does cover.
///
/// fio's `seqwrite` job is 1 MiB blocks at `iodepth=10` and reports both
/// bandwidth and latency, so the write path is not wholly unchecked —
/// only its random and garbage-collecting behaviour is.
pub fn score_sequential_write() -> Scorecard {
    let mut throughput = Vec::new();
    let mut mean_lat = Vec::new();

    // Measured `lat` means from the seqwrite runs, microseconds.
    let measured_latency_us: &[(&str, f64)] = &[
        ("spinning-sata", 52_460.0),
        ("sata-ssd", 19_502.8),
        ("nvme-consumer", 10_962.3),
    ];

    for (hardware, regime) in io::hw::HISTORICAL_HARDWARE.iter().zip(regime::ALL.iter()) {
        const SPAN: u64 = 5 * 1024 * 1024 * 1024;
        let n = if hardware.name == "spinning-sata" {
            400
        } else {
            2_000
        };
        let mut scheduler = io::sched::Noop::default();
        let mut issuer = io::SequentialAccess::new(SPAN, 1 << 20, n, true);
        let result = io::run_streams(
            hardware,
            &mut scheduler,
            &mut [io::Stream::new("main", &mut issuer, 10)],
            io::RunConfig::direct(10, SPAN),
        );

        throughput.push((
            result.total.throughput(),
            regime.seq_write.bytes_per_s() as f64,
        ));
        if let Some((_, measured)) = measured_latency_us
            .iter()
            .find(|(n, _)| *n == hardware.name)
        {
            mean_lat.push((result.latency.summary().micros().mean, *measured));
        }
    }

    Scorecard {
        subject: "sequential write".to_string(),
        throughput: Score::from("write throughput", &throughput),
        mean_latency: Score::from("write latency", &mean_lat),
        p50: Score::default(),
        p95: Score::default(),
        p99: Score::default(),
        tail_ratio: Score::default(),
    }
}

/// Validate every device with measured data.
pub fn score_all() -> Vec<Scorecard> {
    io::hw::HISTORICAL_HARDWARE
        .iter()
        .zip(regime::ALL.iter())
        .map(|(hardware, regime)| score_device(hardware, regime))
        .collect()
}

/// Aggregate across devices, which is the figure comparable to a
/// simulator paper's headline.
pub fn overall(cards: &[Scorecard]) -> Scorecard {
    let combine = |pick: fn(&Scorecard) -> Score, name: &'static str| -> Score {
        let total: usize = cards.iter().map(|c| pick(c).samples).sum();
        if total == 0 {
            return Score {
                name,
                ..Score::default()
            };
        }
        Score {
            name,
            samples: total,
            mape: cards
                .iter()
                .map(|c| pick(c).mape * pick(c).samples as f64)
                .sum::<f64>()
                / total as f64,
            worst: cards.iter().map(|c| pick(c).worst).fold(0.0, f64::max),
            bias: cards
                .iter()
                .map(|c| pick(c).bias * pick(c).samples as f64)
                .sum::<f64>()
                / total as f64,
        }
    };

    Scorecard {
        subject: "all devices".to_string(),
        throughput: combine(|c| c.throughput, "throughput"),
        mean_latency: combine(|c| c.mean_latency, "mean latency"),
        p50: combine(|c| c.p50, "p50"),
        p95: combine(|c| c.p95, "p95"),
        p99: combine(|c| c.p99, "p99"),
        tail_ratio: combine(|c| c.tail_ratio, "p99/p50"),
    }
}

/// Render the scorecards and the comparison against published accuracy.
pub fn render(cards: &[Scorecard]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();

    let _ = writeln!(
        s,
        "\n  {:<16} {:<14} {:>8} {:>9} {:>9} {:>9}",
        "device", "metric", "samples", "MAPE", "worst", "bias"
    );
    for card in cards {
        for score in card.scores() {
            if score.samples == 0 {
                continue;
            }
            let _ = writeln!(
                s,
                "  {:<16} {:<14} {:>8} {:>8.1}% {:>8.1}% {:>+8.1}%",
                card.subject,
                score.name,
                score.samples,
                score.mape * 100.0,
                score.worst * 100.0,
                score.bias * 100.0
            );
        }
    }

    let total = overall(cards);
    let _ = writeln!(s, "\n  Aggregate across devices:\n");
    for score in total.scores() {
        if score.samples == 0 {
            continue;
        }
        let _ = writeln!(
            s,
            "  {:<31} {:>8} {:>8.1}% {:>8.1}% {:>+8.1}%",
            score.name,
            score.samples,
            score.mape * 100.0,
            score.worst * 100.0,
            score.bias * 100.0
        );
    }

    let _ = writeln!(
        s,
        "\n  Published bars, like for like:\n\n  \
         {:<44} {:>10}\n  {:<44} {:>10}\n  {:<44} {:>10}\n  {:<44} {:>9.1}%\n  {:<44} {:>9.1}%",
        "MQSim (FAST '18), throughput vs 4 real SSDs",
        "6-18%",
        "SimpleSSD, worst-case throughput",
        "28%",
        "SimpleSSD, worst-case latency",
        "36%",
        "this model, throughput MAPE",
        total.throughput.mape * 100.0,
        "this model, worst-case throughput",
        total.throughput.worst * 100.0,
    );
    let _ = writeln!(
        s,
        "  {:<44} {:>9.1}%\n  {:<44} {:>9.1}%",
        "this model, mean-latency MAPE",
        total.mean_latency.mape * 100.0,
        "this model, worst-case p99",
        total.p99.worst * 100.0,
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Throughput agreement at or inside the band MQSim reports.**
    ///
    /// MQSim's headline is that it lands within 6–18% of four real SSDs.
    /// This is the comparable claim, made across three devices and every
    /// block size from 512 B to 1 MiB rather than at a handful of points.
    #[test]
    fn throughput_agreement_matches_the_published_band() {
        let cards = score_all();
        let total = overall(&cards);
        assert!(
            total.throughput.mape < 0.05,
            "throughput MAPE {:.1}% should sit inside MQSim's 6-18% band\n{}",
            total.throughput.mape * 100.0,
            render(&cards)
        );
        assert!(
            total.throughput.worst < 0.18,
            "worst-case throughput {:.1}% should beat SimpleSSD's 28%",
            total.throughput.worst * 100.0
        );
    }

    /// Latency is the harder half, and the one a throughput-only model
    /// cannot claim at all.
    #[test]
    fn latency_agreement_beats_the_published_worst_case() {
        let cards = score_all();
        let total = overall(&cards);
        assert!(
            total.mean_latency.mape < 0.06,
            "mean-latency MAPE {:.1}%\n{}",
            total.mean_latency.mape * 100.0,
            render(&cards)
        );
        assert!(
            total.p99.worst < 0.36,
            "worst-case p99 {:.1}% should beat SimpleSSD's 36% latency figure",
            total.p99.worst * 100.0
        );
    }

    /// **The write path, which used to be unvalidated.** The corpus has
    /// a sequential-write job for every device, and reproducing all three
    /// took two mechanisms that a read-only check would never have
    /// demanded: programming charged by the byte rather than per request,
    /// and a write-path bandwidth ceiling separate from the read one.
    #[test]
    fn sequential_write_is_reproduced() {
        let card = score_sequential_write();
        // The disk and the SATA SSD land within 1%. The NVMe drive sits
        // about 12% low since its die count was corrected downward to the
        // controller's actual reach: at 64 units, eight concurrent 1 MiB
        // writes exhaust the parallelism before the write-path ceiling is
        // reached. That is the model being consistent with a sourced
        // parameter rather than a tuned one, and it is left standing.
        assert!(
            card.throughput.mape < 0.06,
            "write throughput MAPE {:.1}%",
            card.throughput.mape * 100.0
        );
        assert!(
            card.mean_latency.mape < 0.07,
            "write latency MAPE {:.1}%",
            card.mean_latency.mape * 100.0
        );
    }

    /// A disk must have no separate write path. Giving it one even
    /// slightly below its media rate desynchronises sequential writes
    /// from the platter and halves its throughput — the head finishes
    /// each block a fraction after the next sector has passed.
    #[test]
    fn a_disk_writes_at_its_media_rate() {
        let hw = io::hw::SPINNING_SATA_HW;
        assert!(
            !hw.write_bandwidth.is_finite(),
            "a disk should carry no write ceiling at all"
        );
        assert_eq!(hw.program_rate_per_die, 0.0, "and no program phase");
    }

    /// The median is the part of a distribution a mean-matching model can
    /// still get badly wrong, so it gets its own bar.
    #[test]
    fn the_median_is_reproduced_not_merely_the_mean() {
        let cards = score_all();
        let total = overall(&cards);
        assert!(
            total.p50.mape < 0.10,
            "p50 MAPE {:.1}%\n{}",
            total.p50.mape * 100.0,
            render(&cards)
        );
    }

    /// No metric should be carried by a single device: if one is fitted
    /// well and another badly, the aggregate hides it.
    #[test]
    fn no_device_is_carrying_the_aggregate() {
        for card in score_all() {
            assert!(
                card.throughput.mape < 0.09,
                "{}: throughput MAPE {:.1}%",
                card.subject,
                card.throughput.mape * 100.0
            );
            assert!(
                card.mean_latency.mape < 0.12,
                "{}: mean-latency MAPE {:.1}%",
                card.subject,
                card.mean_latency.mape * 100.0
            );
        }
    }

    /// Systematic bias is worse than scatter: a model that is wrong in
    /// one direction everywhere has a term missing, where one that
    /// scatters is merely imprecise.
    #[test]
    fn no_metric_is_systematically_biased() {
        let cards = score_all();
        let total = overall(&cards);
        for score in total.scores() {
            if score.samples == 0 {
                continue;
            }
            assert!(
                score.bias.abs() < 0.15,
                "{} carries a {:+.1}% systematic bias\n{}",
                score.name,
                score.bias * 100.0,
                render(&cards)
            );
        }
    }
}

#[cfg(test)]
mod report {
    use super::*;

    #[test]
    #[ignore = "diagnostic report"]
    fn print_validation_report() {
        print!("{}", render(&score_all()));
    }
}

#[cfg(test)]
mod per_block {
    use super::*;

    #[test]
    #[ignore = "diagnostic"]
    fn print_per_block_errors() {
        for (hardware, regime) in io::hw::HISTORICAL_HARDWARE
            .iter()
            .zip(regime::ALL.iter())
            .filter(|(h, _)| h.name != "spinning-sata")
        {
            println!("\n### {}", hardware.name);
            println!(
                "  {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
                "block", "iops%", "mean%", "p50%", "p95%", "p99%"
            );
            let latency = regime::measured_latency(hardware.name);
            for point in regime
                .random_read
                .iter()
                .filter(|p| p.block_bytes <= 1 << 20)
            {
                let Some(m) = latency.iter().find(|l| l.block_bytes == point.block_bytes) else {
                    continue;
                };
                let r = io::fio_like_detailed(hardware, point.block_bytes, 8_000);
                let s = r.latency.summary().micros();
                let e = |sim: f64, meas: f64| (sim - meas) / meas * 100.0;
                println!(
                    "  {:>8} {:>+7.1} {:>+7.1} {:>+7.1} {:>+7.1} {:>+7.1}",
                    point.block_bytes,
                    e(r.total.iops(), point.iops as f64),
                    e(s.mean, m.mean_us),
                    e(s.p50, m.p50_us),
                    e(s.p95, m.p95_us),
                    e(s.p99, m.p99_us)
                );
            }
        }
    }
}

#[cfg(test)]
mod bus_sweep {
    use super::*;
    use crate::regime::NVME_CONSUMER;

    #[test]
    #[ignore = "diagnostic"]
    fn sweep_nvme_bus_rate() {
        println!(
            "\n  {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "bus MB/s", "iops", "mean", "p50", "p95", "p99"
        );
        for bus in [1_500.0f64, 1_600.0, 1_700.0, 1_750.0, 1_800.0, 1_900.0] {
            let hardware = io::hw::Hardware {
                bus_rate: bus * 1e6,
                ..io::hw::NVME_CONSUMER_HW
            };
            let card = score_device(&hardware, &NVME_CONSUMER);
            println!(
                "  {:>8.0} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}%",
                bus,
                card.throughput.mape * 100.0,
                card.mean_latency.mape * 100.0,
                card.p50.mape * 100.0,
                card.p95.mape * 100.0,
                card.p99.mape * 100.0
            );
        }
    }
}

#[cfg(test)]
mod spread_sweep {
    use super::*;

    #[test]
    #[ignore = "diagnostic"]
    fn sweep_transfer_share_spread() {
        println!(
            "\n  {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "spread", "iops", "p50", "p95", "p99", "p99 bias"
        );
        for spread in [0.0f64, 0.10, 0.18, 0.25, 0.35, 0.45] {
            let cards: Vec<Scorecard> = io::hw::HISTORICAL_HARDWARE
                .iter()
                .zip(regime::ALL.iter())
                .map(|(h, r)| {
                    let hardware = io::hw::Hardware {
                        transfer_share_spread: if h.dies > 1 { spread } else { 0.0 },
                        ..*h
                    };
                    score_device(&hardware, r)
                })
                .collect();
            let t = overall(&cards);
            println!(
                "  {:>7.2} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>+8.1}%",
                spread,
                t.throughput.mape * 100.0,
                t.p50.mape * 100.0,
                t.p95.mape * 100.0,
                t.p99.mape * 100.0,
                t.p99.bias * 100.0
            );
        }
    }
}

#[cfg(test)]
mod write_report {
    use super::*;

    #[test]
    #[ignore = "diagnostic"]
    fn print_sequential_write() {
        let targets: &[(&str, f64, f64)] = &[
            ("spinning-sata", 195_143.0 * 1024.0, 52_460.0),
            ("sata-ssd", 524_907.0 * 1024.0, 19_502.8),
            ("nvme-consumer", 933_819.0 * 1024.0, 10_962.3),
        ];
        for (hardware, (_, bw, lat)) in io::hw::HISTORICAL_HARDWARE.iter().zip(targets) {
            const SPAN: u64 = 5 * 1024 * 1024 * 1024;
            let n = if hardware.name == "spinning-sata" {
                400
            } else {
                2_000
            };
            let mut sched = io::sched::Noop::default();
            let mut issuer = io::SequentialAccess::new(SPAN, 1 << 20, n, true);
            let r = io::run_streams(
                hardware,
                &mut sched,
                &mut [io::Stream::new("main", &mut issuer, 10)],
                io::RunConfig::direct(10, SPAN),
            );
            println!(
                "  {:<16} {:>7.0} vs {:>7.0} MB/s ({:+.0}%)   {:>8.0} vs {:>8.0} us ({:+.0}%)",
                hardware.name,
                r.total.throughput() / 1e6,
                bw / 1e6,
                (r.total.throughput() - bw) / bw * 100.0,
                r.latency.summary().micros().mean,
                lat,
                (r.latency.summary().micros().mean - lat) / lat * 100.0
            );
        }
        let card = score_sequential_write();
        for score in [card.throughput, card.mean_latency] {
            println!(
                "  {:<18} {:>3} samples  MAPE {:>6.1}%  worst {:>6.1}%  bias {:>+6.1}%",
                score.name,
                score.samples,
                score.mape * 100.0,
                score.worst * 100.0,
                score.bias * 100.0
            );
        }
    }
}

#[cfg(test)]
mod program_sweep {
    use super::*;

    #[test]
    #[ignore = "diagnostic"]
    fn sweep_program_rate() {
        let targets: &[(&str, f64)] = &[
            ("spinning-sata", 195_143.0 * 1024.0),
            ("sata-ssd", 524_907.0 * 1024.0),
            ("nvme-consumer", 933_819.0 * 1024.0),
        ];
        for hardware in io::hw::HISTORICAL_HARDWARE {
            let target = targets.iter().find(|(n, _)| *n == hardware.name).unwrap().1;
            println!("\n### {} target {:.0} MB/s", hardware.name, target / 1e6);
            for rate in [3.0f64, 5.0, 7.0, 9.0, 12.0, 16.0, 22.0] {
                let hw2 = io::hw::Hardware {
                    program_rate_per_die: rate * 1e6,
                    ..*hardware
                };
                const SPAN: u64 = 5 * 1024 * 1024 * 1024;
                let n = if hardware.name == "spinning-sata" {
                    400
                } else {
                    1_500
                };
                let mut sched = io::sched::Noop::default();
                let mut issuer = io::SequentialAccess::new(SPAN, 1 << 20, n, true);
                let r = io::run_streams(
                    &hw2,
                    &mut sched,
                    &mut [io::Stream::new("main", &mut issuer, 10)],
                    io::RunConfig::direct(10, SPAN),
                );
                println!(
                    "  {:>6.0} MB/s/die -> {:>7.0} MB/s  ({:+.0}%)",
                    rate,
                    r.total.throughput() / 1e6,
                    (r.total.throughput() - target) / target * 100.0
                );
            }
        }
    }
}
