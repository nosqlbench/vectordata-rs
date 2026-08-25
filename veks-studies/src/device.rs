// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A forward-simulating device model, checked against the measurements.
//!
//! A regime is a table of numbers. A table can be interpolated but it
//! cannot be reasoned with, and it says nothing about block sizes,
//! queue depths, or devices nobody measured. This module holds the
//! *generative* counterpart: a small parametric model that predicts what
//! a device will do, and a test that makes it reproduce the perfscripts
//! curves it claims to explain.
//!
//! That check is the point. Pricing a simulated trace against a lookup
//! table proves nothing about the model — the table was measured, not
//! derived. But if a four-parameter model can regenerate an entire
//! 512 B–16 MiB random-read sweep across three unrelated devices, then
//! the mechanism it encodes is the mechanism those devices have.
//!
//! The mechanism is deliberately plain. Each operation occupies a queue
//! slot for
//!
//! ```text
//! t(b) = latency + b / transfer_rate
//! ```
//!
//! and `queue_depth` slots run concurrently, subject to two ceilings —
//! one on operations per second (command processing) and one on bytes per
//! second (the link). Everything the sweeps show follows from those four
//! numbers: the flat small-block IOPS plateau is the operation ceiling,
//! the flat large-block bandwidth plateau is the byte ceiling, and the
//! knee between them is where `b / transfer_rate` overtakes `latency`.
//!
//! A spinning disk is the same model with `queue_depth = 1` and a
//! latency three orders of magnitude larger — which is why its random
//! curve is flat in IOPS rather than flat in bandwidth, and why ordering
//! is worth so much more there.
//!
//! Parameters here are fitted to the perfscripts sweeps; the modern
//! device is calibrated to published figures instead. Both are named in
//! [the crate bibliography](crate#sources).
//!
//! Where the model and the measurement disagree, [`FitReport`] says so
//! rather than the parameters being bent until they agree. See
//! [`NVME_CONSUMER_MODEL`] for the one place that matters.

use crate::regime::Regime;

/// Which constraint is setting the throughput at a given block size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    /// Not enough operations in flight to keep the device busy.
    Concurrency,
    /// The controller cannot accept requests any faster.
    CommandRate,
    /// The link cannot carry bytes any faster.
    Bandwidth,
}

/// A parametric device.
#[derive(Debug, Clone, Copy)]
pub struct DeviceModel {
    pub name: &'static str,
    /// Fixed per-operation cost, seconds.
    pub latency_s: f64,
    /// Bytes per second a single outstanding operation transfers.
    pub transfer_bytes_per_s: f64,
    /// Effective concurrency achieved at the depth the sweep was run at.
    ///
    /// This is *not* the device's native queue capability — see
    /// [`DeviceModel::native_queue_depth`]. It is how many operations
    /// were actually in flight and productive during the measurement,
    /// which for a spinning disk is close to one however deep the queue
    /// is offered, because there is one head.
    pub queue_depth: f64,
    /// The deepest concurrency the device can turn into throughput.
    /// Offering more than this changes nothing.
    pub native_queue_depth: f64,
    /// Command-rate ceiling, operations per second. The controller's
    /// limit on how many requests it can process, independent of size.
    pub iops_ceiling: f64,
    /// Link ceiling, bytes per second. The interconnect's limit,
    /// independent of request count.
    pub bw_ceiling: f64,
}

impl DeviceModel {
    /// How long one operation of `block_bytes` occupies its queue slot.
    pub fn service_time_s(&self, block_bytes: u64) -> f64 {
        self.latency_s + block_bytes as f64 / self.transfer_bytes_per_s
    }

    /// Predicted operations per second at this block size, at the
    /// concurrency the model was measured at.
    ///
    /// Three limits apply at once and the smallest wins: concurrency
    /// divided by service time (Little's law), the controller's command
    /// rate, and the link's byte rate. Which one binds is the whole
    /// shape of the curve.
    pub fn iops(&self, block_bytes: u64) -> f64 {
        self.iops_at_depth(block_bytes, self.queue_depth)
    }

    /// The same prediction at an arbitrary offered concurrency.
    ///
    /// Offering more than [`Self::native_queue_depth`] buys nothing, and
    /// neither ceiling moves — a deeper queue can only help until one of
    /// them binds.
    ///
    /// **Validation limit:** the perfscripts sweeps were all run at
    /// `iodepth=10` and there is no queue-depth sweep in that data, so
    /// predictions away from that depth exercise the model's structure
    /// without a measurement behind them.
    pub fn iops_at_depth(&self, block_bytes: u64, offered_depth: f64) -> f64 {
        let effective = offered_depth.min(self.native_queue_depth);
        let by_concurrency = effective / self.service_time_s(block_bytes);
        let by_bandwidth = self.bw_ceiling / block_bytes as f64;
        by_concurrency.min(self.iops_ceiling).min(by_bandwidth)
    }

    /// Which of the three limits is binding at this block size.
    pub fn binding_limit(&self, block_bytes: u64) -> Limit {
        let by_concurrency = self.queue_depth / self.service_time_s(block_bytes);
        let by_bandwidth = self.bw_ceiling / block_bytes as f64;
        if by_concurrency <= self.iops_ceiling && by_concurrency <= by_bandwidth {
            Limit::Concurrency
        } else if self.iops_ceiling <= by_bandwidth {
            Limit::CommandRate
        } else {
            Limit::Bandwidth
        }
    }

    /// Predicted bytes per second at this block size.
    pub fn bandwidth(&self, block_bytes: u64) -> f64 {
        self.iops(block_bytes) * block_bytes as f64
    }

    /// Seconds to perform `count` random operations of `block_bytes`.
    pub fn random_read_seconds(&self, count: u64, block_bytes: u64) -> f64 {
        count as f64 / self.iops(block_bytes)
    }

    /// Throughput when reads are contiguous, so no operation pays the
    /// access latency: the model's asymptote as block size grows, which
    /// is whichever ceiling binds first.
    ///
    /// This is a different quantity from `bandwidth(b)` at large `b`, and
    /// confusing the two is expensive. `bandwidth(128 KiB)` on a spinning
    /// disk is 30 MB/s, because each of those 128 KiB reads pays a seek;
    /// reading the same bytes in order costs 195 MB/s. A model that
    /// prices an ordered scan with the random figure overstates it
    /// sixfold.
    pub fn sequential_bandwidth(&self) -> f64 {
        self.bw_ceiling
            .min(self.transfer_bytes_per_s * self.queue_depth)
    }

    pub fn sequential_seconds(&self, bytes: u64) -> f64 {
        bytes as f64 / self.sequential_bandwidth()
    }

    /// Compare the model against every point of a measured sweep.
    pub fn fit(&self, regime: &Regime) -> FitReport {
        let points = regime
            .random_read
            .iter()
            .map(|p| {
                let predicted = self.iops(p.block_bytes);
                let measured = p.iops as f64;
                FitPoint {
                    block_bytes: p.block_bytes,
                    predicted_iops: predicted,
                    measured_iops: measured,
                    relative_error: (predicted - measured) / measured,
                }
            })
            .collect();
        FitReport {
            model: self.name,
            device: regime.device,
            points,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FitPoint {
    pub block_bytes: u64,
    pub predicted_iops: f64,
    pub measured_iops: f64,
    /// Signed, so systematic bias is visible rather than averaged away.
    pub relative_error: f64,
}

#[derive(Debug, Clone)]
pub struct FitReport {
    pub model: &'static str,
    pub device: &'static str,
    pub points: Vec<FitPoint>,
}

impl FitReport {
    /// Worst absolute relative error among blocks up to `limit`.
    pub fn worst_error_up_to(&self, limit: u64) -> f64 {
        self.points
            .iter()
            .filter(|p| p.block_bytes <= limit)
            .map(|p| p.relative_error.abs())
            .fold(0.0, f64::max)
    }

    pub fn median_error(&self) -> f64 {
        let mut errs: Vec<f64> = self.points.iter().map(|p| p.relative_error.abs()).collect();
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        errs[errs.len() / 2]
    }

    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "\n{} vs {}", self.model, self.device);
        let _ = writeln!(
            s,
            "  {:>9}  {:>12}  {:>12}  {:>8}",
            "block", "predicted", "measured", "error"
        );
        for p in &self.points {
            let _ = writeln!(
                s,
                "  {:>9}  {:>12.0}  {:>12.0}  {:>7.1}%",
                p.block_bytes,
                p.predicted_iops,
                p.measured_iops,
                p.relative_error * 100.0
            );
        }
        s
    }
}

/// One head, one seek at a time. `queue_depth = 1` is not a
/// simplification — it is the physical fact that makes this device what
/// it is, and the reason its random IOPS barely move from 512 B to
/// 16 KiB: at those sizes the transfer is lost in the seek.
///
/// The transfer rate here was not chosen; it falls out of the slope
/// between the 4 KiB and 16 MiB points, and lands within 2% of the
/// device's separately measured sequential read rate.
pub const SPINNING_SATA_MODEL: DeviceModel = DeviceModel {
    name: "spinning-sata",
    latency_s: 3.74e-3,
    transfer_bytes_per_s: 195.0e6,
    queue_depth: 1.0,
    // A disk's controller is electronic and fast; what limits a disk is
    // the head, which is modelled separately as positioning. This ceiling
    // never binds in `iops()` — concurrency over service time binds first
    // at 266 — but it is read directly as a command-processing rate by
    // the service-demand model, where leaving it at a positioning-derived
    // 300 charged ordered reads a mechanical cost they do not pay.
    // SATA NCQ accepts 32, and the drive does use it to reorder seeks —
    // but reordering shortens seeks, it does not add heads, so offered
    // depth converts to throughput only weakly. Modelled as no useful
    // gain past what the sweep already measured.
    native_queue_depth: 1.0,
    iops_ceiling: 100_000.0,
    bw_ceiling: 201.0e6,
};

/// Flat below 4 KiB because the controller runs out of commands, flat
/// above 256 KiB because SATA runs out of link. The whole curve is the
/// interval between those two ceilings.
pub const SATA_SSD_MODEL: DeviceModel = DeviceModel {
    name: "sata-ssd",
    latency_s: 32.0e-6,
    transfer_bytes_per_s: 57.0e6,
    queue_depth: 10.0,
    // SATA NCQ depth.
    native_queue_depth: 32.0,
    iops_ceiling: 80_000.0,
    bw_ceiling: 568.0e6,
};

/// **Known divergence.** The measured curve is not monotonic: bandwidth
/// peaks at 128 KiB (1667 MB/s) and then *falls* to a 1340 MB/s plateau
/// for every larger block. No single byte ceiling reproduces both, so
/// this model is fitted to the peak and overpredicts by roughly a third
/// from 256 KiB up. The parameters are left alone rather than split into
/// a piecewise ceiling that would hide the disagreement; tests bound the
/// error below 128 KiB and assert the divergence above it, so it cannot
/// be forgotten.
pub const NVME_CONSUMER_MODEL: DeviceModel = DeviceModel {
    name: "nvme-consumer",
    latency_s: 62.0e-6,
    transfer_bytes_per_s: 190.0e6,
    queue_depth: 10.0,
    // NVMe queues are far deeper than anything this model needs; the
    // command-rate ceiling binds long before the queue does.
    native_queue_depth: 256.0,
    iops_ceiling: 124_000.0,
    bw_ceiling: 1_750.0e6,
};

/// The closed-form counterpart of [`crate::io::hw::NVME_MODERN_HW`],
/// calibrated to the same published figures.
///
/// Its reference concurrency is 128 rather than 10, because that is the
/// regime the figures describe: the ICPE '24 testbed needed many cores
/// to reach its aggregate, and the MQSSD measurements that pin the
/// random-to-sequential ratio are quoted at k=128.
pub const NVME_MODERN_MODEL: DeviceModel = DeviceModel {
    name: "nvme-modern",
    latency_s: 68.0e-6,
    transfer_bytes_per_s: 400.0e6,
    queue_depth: 128.0,
    native_queue_depth: 1_024.0,
    iops_ceiling: 1_200_000.0,
    bw_ceiling: 7_000.0e6,
};

pub const ALL_MODELS: &[DeviceModel] = &[SPINNING_SATA_MODEL, SATA_SSD_MODEL, NVME_CONSUMER_MODEL];

/// The measured corpus plus the projected modern drive.
///
/// [`ALL_MODELS`] is deliberately only what was swept, so validation
/// never scores itself against a projection. The studies want the modern
/// drive as well, because the interesting question about a boundary is
/// which way newer hardware moves it — and the answer is only meaningful
/// if the projection is labelled as one.
pub const ALL_MODELS_WITH_MODERN: &[DeviceModel] = &[
    SPINNING_SATA_MODEL,
    SATA_SSD_MODEL,
    NVME_CONSUMER_MODEL,
    NVME_MODERN_MODEL,
];

/// Pair each model with the sweep it claims to explain.
pub fn paired() -> Vec<(DeviceModel, &'static Regime)> {
    use crate::regime::{NVME_CONSUMER, SATA_SSD, SPINNING_SATA};
    vec![
        (SPINNING_SATA_MODEL, &SPINNING_SATA),
        (SATA_SSD_MODEL, &SATA_SSD),
        (NVME_CONSUMER_MODEL, &NVME_CONSUMER),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The central claim: four parameters regenerate three measured
    /// sweeps across the block sizes that matter for record layout.
    #[test]
    fn the_models_reproduce_the_measured_sweeps() {
        for (model, regime) in paired() {
            let report = model.fit(regime);
            let worst = report.worst_error_up_to(131_072);
            assert!(
                worst < 0.15,
                "{} misses its own measurements by {:.0}% somewhere at or below 128 KiB\n{}",
                model.name,
                worst * 100.0,
                report.render()
            );
        }
    }

    /// Achieved accuracy across the whole sweep, stated as measured
    /// rather than as hoped. The two devices with well-behaved curves fit
    /// to within a couple of percent; the NVMe drive fits to about ten,
    /// because no monotone model reproduces a curve that peaks and then
    /// falls back.
    #[test]
    fn the_typical_error_is_small_across_the_full_sweep() {
        for (model, regime) in paired() {
            let median = model.fit(regime).median_error();
            let allowed = if model.queue_depth > 1.0 && model.bw_ceiling > 1.0e9 {
                0.12
            } else {
                0.03
            };
            assert!(
                median < allowed,
                "{}: median error {:.1}%, allowed {:.0}%",
                model.name,
                median * 100.0,
                allowed * 100.0
            );
        }
    }

    /// The divergence documented on [`NVME_CONSUMER_MODEL`], asserted so
    /// that it stays known. If this test starts failing, the measured
    /// curve or the parameters changed and the doc comment is stale.
    #[test]
    fn the_nvme_model_overpredicts_above_the_bandwidth_peak() {
        use crate::regime::NVME_CONSUMER;
        let report = NVME_CONSUMER_MODEL.fit(&NVME_CONSUMER);
        let large: Vec<&FitPoint> = report
            .points
            .iter()
            .filter(|p| p.block_bytes >= 262_144)
            .collect();

        assert!(
            large.iter().all(|p| p.relative_error > 0.15),
            "every block at or above 256 KiB should be overpredicted"
        );
        assert!(
            large.iter().all(|p| p.relative_error < 0.45),
            "…but by a third, not by an order of magnitude"
        );
    }

    /// The fitted transfer rate is not a free parameter dressed up as
    /// physics: on the spinning disk it has to agree with the separately
    /// measured sequential rate, and it does.
    #[test]
    fn the_fitted_transfer_rate_matches_measured_sequential_throughput() {
        use crate::regime::SPINNING_SATA;
        let measured = SPINNING_SATA.seq_read.bytes_per_s() as f64;
        let fitted = SPINNING_SATA_MODEL.transfer_bytes_per_s;
        assert!(
            (fitted - measured).abs() / measured < 0.05,
            "fitted {fitted:.0} B/s vs measured sequential {measured:.0} B/s"
        );
    }

    /// The same fit, stated as a mechanism: on flash the aggregate of all
    /// queue slots is what saturates the link, so per-slot transfer rate
    /// times queue depth should land near the measured ceiling.
    #[test]
    fn queue_slots_in_aggregate_explain_the_link_ceiling() {
        for (model, regime) in paired().into_iter().filter(|(m, _)| m.queue_depth > 1.0) {
            let aggregate = model.transfer_bytes_per_s * model.queue_depth;
            let measured_peak = regime
                .random_read
                .iter()
                .map(|p| p.bandwidth.bytes_per_s())
                .max()
                .unwrap() as f64;
            let ratio = aggregate / measured_peak;
            assert!(
                (0.6..1.6).contains(&ratio),
                "{}: aggregate slot throughput {aggregate:.0} vs peak {measured_peak:.0}",
                model.name
            );
        }
    }

    /// A model is only useful if it extrapolates. These block sizes were
    /// never measured, and the predictions must still be ordered sanely.
    #[test]
    fn predictions_between_measured_points_stay_monotone() {
        for model in ALL_MODELS {
            let mut previous_bw = 0.0;
            for block in [3_000u64, 6_144, 12_288, 24_576, 49_152] {
                let bw = model.bandwidth(block);
                assert!(
                    bw >= previous_bw,
                    "{}: bandwidth fell from {previous_bw:.0} to {bw:.0} at {block} B",
                    model.name
                );
                previous_bw = bw;
            }
        }
    }
}

#[cfg(test)]
mod dump {
    use super::*;

    #[test]
    #[ignore = "diagnostic dump, not an assertion"]
    fn print_fits() {
        for (model, regime) in paired() {
            let r = model.fit(regime);
            print!("{}", r.render());
            println!(
                "  median {:.1}% · worst<=128K {:.1}%",
                r.median_error() * 100.0,
                r.worst_error_up_to(131_072) * 100.0
            );
        }
    }
}

#[cfg(test)]
mod concurrency {
    use super::*;

    /// Both device ceilings are modelled, and which one binds moves with
    /// block size. Small requests exhaust the controller's command rate;
    /// large ones exhaust the link. On a spinning disk neither is ever
    /// reached, because a single head cannot generate enough of either.
    #[test]
    fn each_ceiling_binds_where_it_should() {
        assert_eq!(SATA_SSD_MODEL.binding_limit(512), Limit::CommandRate);
        assert_eq!(SATA_SSD_MODEL.binding_limit(1 << 20), Limit::Bandwidth);
        assert_eq!(NVME_CONSUMER_MODEL.binding_limit(512), Limit::CommandRate);
        assert_eq!(NVME_CONSUMER_MODEL.binding_limit(1 << 20), Limit::Bandwidth);

        for block in [512u64, 4_096, 65_536, 1 << 20, 1 << 24] {
            assert_eq!(
                SPINNING_SATA_MODEL.binding_limit(block),
                Limit::Concurrency,
                "one head is always the binding constraint at {block} B"
            );
        }
    }

    /// A deeper queue helps until a ceiling stops it, and then it does
    /// nothing at all. Both halves matter: the first is why concurrency
    /// is worth configuring, the second is why it is not worth
    /// configuring past the knee.
    #[test]
    fn deeper_queues_help_until_a_ceiling_binds() {
        let block = 32_768;
        let shallow = NVME_CONSUMER_MODEL.iops_at_depth(block, 1.0);
        let measured = NVME_CONSUMER_MODEL.iops_at_depth(block, 10.0);
        let deep = NVME_CONSUMER_MODEL.iops_at_depth(block, 64.0);
        let absurd = NVME_CONSUMER_MODEL.iops_at_depth(block, 100_000.0);

        assert!(measured > shallow * 5.0, "ten slots should beat one");
        assert!(deep > measured, "and more should still help at 32 KiB");
        assert_eq!(
            deep, absurd,
            "until a ceiling binds, after which nothing does"
        );
    }

    /// Offering a spinning disk more concurrency changes nothing, because
    /// its native useful depth is one. This is the modelled reason
    /// ordering matters so much more there: you cannot buy your way out
    /// of seeks with parallelism.
    #[test]
    fn concurrency_cannot_rescue_a_single_head() {
        let one = SPINNING_SATA_MODEL.iops_at_depth(4_096, 1.0);
        let many = SPINNING_SATA_MODEL.iops_at_depth(4_096, 64.0);
        assert_eq!(one, many);
        assert!(one < 300.0);
    }

    /// The bandwidth ceiling is a real constraint and not decoration: at
    /// large blocks the model's throughput must sit at it rather than
    /// climbing with block size.
    #[test]
    fn the_bandwidth_ceiling_actually_caps_throughput() {
        for model in [SATA_SSD_MODEL, NVME_CONSUMER_MODEL] {
            let big = model.bandwidth(1 << 22);
            assert!(
                (big - model.bw_ceiling).abs() / model.bw_ceiling < 0.01,
                "{}: {big:.0} B/s should sit at the {:.0} B/s ceiling",
                model.name,
                model.bw_ceiling
            );
        }
    }
}

impl DeviceModel {
    /// **The random-access penalty at a record size**: how much slower it
    /// is to fetch records where they lie than to stream the extent that
    /// contains them.
    ///
    /// This single number decides whether ordering is worth anything on a
    /// device, and it is a property of the *record size*, not of the
    /// storage class. A 128-byte record on the 2016 NVMe drive carries a
    /// penalty of 110; a 4 KiB record on the same drive carries 3.6.
    pub fn random_penalty(&self, record_bytes: u64) -> f64 {
        self.random_penalty_at_depth(record_bytes, self.queue_depth)
    }

    /// **The penalty is also a function of concurrency**, and strongly.
    ///
    /// Random access loses to streaming because each request pays an
    /// access latency that a stream does not. Concurrency hides latency:
    /// with `k` requests outstanding the latency is amortised `k` ways,
    /// so the penalty falls as `k` rises and vanishes once a ceiling
    /// binds instead.
    ///
    /// The effect is not small. The
    /// [MQSSD measurements](https://arxiv.org/abs/2507.06349) report a
    /// random-to-sequential read ratio of **1.3–1.5× at k=128** on a
    /// current drive, against 38–57× for writes at k=1. Any statement
    /// about whether ordering pays that does not name a concurrency is
    /// underdetermined — and the perfscripts corpus was captured at
    /// `iodepth=10`, so every figure derived from it is a statement about
    /// that depth.
    pub fn random_penalty_at_depth(&self, record_bytes: u64, depth: f64) -> f64 {
        let random = self.iops_at_depth(record_bytes, depth) * record_bytes as f64;
        if random <= 0.0 {
            f64::INFINITY
        } else {
            self.sequential_bandwidth() / random
        }
    }

    /// **The line.** Ordering pays exactly when the pass count is below
    /// the random-access penalty.
    ///
    /// A gather costs `N / IOPS(R)`. An ordered rewrite costs `P` scans,
    /// or `P · N·R / BW_seq`. Setting them equal:
    ///
    /// ```text
    ///     N / IOPS(R)  =  P · N·R / BW_seq
    ///     BW_seq / (R · IOPS(R))  =  P
    ///     penalty(R)  =  P
    /// ```
    ///
    /// `N` cancels, so the crossover does not depend on how much data
    /// there is — only on the record size and how many passes the budget
    /// forces.
    pub fn ordering_pays(&self, record_bytes: u64, passes: u64) -> bool {
        (passes as f64) < self.random_penalty(record_bytes)
    }

    /// The same test at a stated concurrency.
    pub fn ordering_pays_at_depth(&self, record_bytes: u64, passes: u64, depth: f64) -> bool {
        (passes as f64) < self.random_penalty_at_depth(record_bytes, depth)
    }

    /// The same line expressed as memory, which is the parameter anyone
    /// actually controls.
    ///
    /// Since `P ≈ payload / M`, ordering pays once
    /// `M > payload / penalty(R)`. Below that budget an ordered rewrite
    /// re-reads the source more times than the seeks it saves are worth.
    pub fn min_budget_for_ordering(&self, payload_bytes: u64, record_bytes: u64) -> u64 {
        self.min_budget_for_ordering_at_depth(payload_bytes, record_bytes, self.queue_depth)
    }

    /// The same line at a stated concurrency. Raising concurrency raises
    /// the budget ordering needs in order to pay — and past some depth
    /// the requirement exceeds the payload, meaning ordering has nothing
    /// left to offer at any budget.
    pub fn min_budget_for_ordering_at_depth(
        &self,
        payload_bytes: u64,
        record_bytes: u64,
        depth: f64,
    ) -> u64 {
        let penalty = self.random_penalty_at_depth(record_bytes, depth);
        if !penalty.is_finite() || penalty <= 1.0 {
            return u64::MAX;
        }
        (payload_bytes as f64 / penalty).ceil() as u64
    }
}

#[cfg(test)]
mod crossover {
    use super::*;

    /// The rule and the priced worked examples must agree. If they ever
    /// diverge, one of them is wrong — which is the point of having both.
    #[test]
    fn the_rule_predicts_what_the_priced_examples_do() {
        use crate::study::WorkedExample;

        let cases = [
            ("B, M = 8 GiB", 100_000_000u64, 1_540u64, 8u64 << 30),
            ("B, M = 32 GiB", 100_000_000, 1_540, 32 << 30),
            ("C, M = 32 GiB", 450_000_000, 4_100, 32 << 30),
            ("C, M = 600 GiB", 450_000_000, 4_100, 600 << 30),
            ("small records", 2_000_000_000, 128, 8 << 30),
        ];

        for (label, records, record_bytes, budget) in cases {
            let ex = WorkedExample {
                label: "",
                records,
                record_bytes,
                container_bytes: 128 * 1024,
                budget_bytes: budget,
            };
            for model in ALL_MODELS {
                let predicted = model.ordering_pays(record_bytes, ex.passes());
                let priced = ex.gsplat_seconds(model) < ex.naive_seconds(model);
                assert_eq!(
                    predicted,
                    priced,
                    "{label} on {}: rule says {predicted}, pricing says {priced} \
                     (P = {}, penalty = {:.1})",
                    model.name,
                    ex.passes(),
                    model.random_penalty(record_bytes)
                );
            }
        }
    }

    /// **Ordering is not a spinning-disk technique.** It pays on NVMe
    /// too; the earlier examples simply sat below the line. Given a
    /// budget above `payload / penalty`, the same rewrite that lost turns
    /// into a win on the same drive.
    #[test]
    fn ordering_pays_on_nvme_once_the_budget_clears_the_line() {
        use crate::study::WorkedExample;

        let records = 100_000_000u64;
        let record_bytes = 1_540u64;
        let payload = records * record_bytes;
        let line = NVME_CONSUMER_MODEL.min_budget_for_ordering(payload, record_bytes);

        assert!(
            line < payload / 5,
            "the line should be a fraction of the payload"
        );

        let below = WorkedExample {
            label: "",
            records,
            record_bytes,
            container_bytes: 128 * 1024,
            budget_bytes: line / 2,
        };
        let above = WorkedExample {
            budget_bytes: line * 2,
            ..below
        };

        assert!(
            below.gsplat_seconds(&NVME_CONSUMER_MODEL) > below.naive_seconds(&NVME_CONSUMER_MODEL),
            "below the line ordering must lose"
        );
        assert!(
            above.gsplat_seconds(&NVME_CONSUMER_MODEL) < above.naive_seconds(&NVME_CONSUMER_MODEL),
            "above it must win"
        );
    }

    /// How far the line moves with record size — three orders of
    /// magnitude of required memory across a 32× range of record size, on
    /// one device.
    #[test]
    fn the_line_moves_enormously_with_record_size() {
        let payload = 1u64 << 40;
        let tiny = NVME_CONSUMER_MODEL.min_budget_for_ordering(payload, 128);
        let large = NVME_CONSUMER_MODEL.min_budget_for_ordering(payload, 4_096);
        assert!(
            large > tiny * 25,
            "128 B needs {tiny} bytes of budget, 4 KiB needs {large}"
        );
    }

    /// And how far it moves with the device: the same rewrite that needs
    /// a large budget to pay on flash pays on a spinning disk at almost
    /// any budget.
    #[test]
    fn the_line_moves_enormously_with_the_device() {
        let payload = 1u64 << 40;
        let disk = SPINNING_SATA_MODEL.min_budget_for_ordering(payload, 4_096);
        let nvme = NVME_CONSUMER_MODEL.min_budget_for_ordering(payload, 4_096);
        assert!(
            nvme > disk * 20,
            "disk needs {disk} bytes, NVMe needs {nvme}"
        );
    }
}

/// Render the crossover line for a range of record sizes.
pub fn render_crossover_table(payload_bytes: u64) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n  Ordering pays when passes < penalty(R), i.e. when M > payload / penalty(R).\n  \
         Payload {:.0} GiB.\n",
        payload_bytes as f64 / (1u64 << 30) as f64
    );
    let _ = writeln!(
        s,
        "  {:>8}  {:>28}  {:>28}  {:>28}",
        "record", "spinning-sata", "sata-ssd", "nvme-consumer"
    );
    let _ = writeln!(
        s,
        "  {:>8}  {:>13} {:>14}  {:>13} {:>14}  {:>13} {:>14}",
        "", "penalty", "min budget", "penalty", "min budget", "penalty", "min budget"
    );

    for r in [128u64, 512, 1_540, 4_096, 16_384, 65_536] {
        let mut row = format!("  {r:>8}");
        for model in ALL_MODELS {
            let penalty = model.random_penalty(r);
            let budget = model.min_budget_for_ordering(payload_bytes, r);
            let budget_str = if budget == u64::MAX {
                "never pays".to_string()
            } else if budget >= (1 << 30) {
                format!("{:.1} GiB", budget as f64 / (1u64 << 30) as f64)
            } else {
                format!("{:.0} MiB", budget as f64 / (1u64 << 20) as f64)
            };
            let _ = write!(row, "  {penalty:>13.1} {budget_str:>14}");
        }
        let _ = writeln!(s, "{row}");
    }
    s
}

/// Render how the crossover moves with concurrency.
pub fn render_concurrency_crossover(payload_bytes: u64, record_bytes: u64) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n  Record {} B, payload {:.0} GiB. Ordering pays while P < penalty.\n",
        record_bytes,
        payload_bytes as f64 / (1u64 << 30) as f64
    );
    let _ = writeln!(
        s,
        "  {:>7}  {:>26}  {:>26}",
        "depth", "nvme-consumer (2016)", "nvme-modern"
    );
    let _ = writeln!(
        s,
        "  {:>7}  {:>11} {:>14}  {:>11} {:>14}",
        "", "penalty", "min budget", "penalty", "min budget"
    );
    for depth in [1.0f64, 4.0, 10.0, 32.0, 64.0, 128.0, 256.0] {
        let mut row = format!("  {depth:>7.0}");
        for model in [NVME_CONSUMER_MODEL, NVME_MODERN_MODEL] {
            let penalty = model.random_penalty_at_depth(record_bytes, depth);
            let budget = model.min_budget_for_ordering_at_depth(payload_bytes, record_bytes, depth);
            let text = if budget == u64::MAX || budget >= payload_bytes {
                "never pays".to_string()
            } else if budget >= (1 << 30) {
                format!("{:.1} GiB", budget as f64 / (1u64 << 30) as f64)
            } else {
                format!("{:.0} MiB", budget as f64 / (1u64 << 20) as f64)
            };
            let _ = write!(row, "  {penalty:>11.1} {text:>14}");
        }
        let _ = writeln!(s, "{row}");
    }
    s
}

#[cfg(test)]
mod concurrency_crossover {
    use super::*;

    /// **Concurrency erodes the case for ordering.** Each outstanding
    /// request hides another request's latency, so the gap random access
    /// has to make up shrinks as the queue deepens.
    #[test]
    fn the_penalty_falls_as_concurrency_rises() {
        for model in [NVME_CONSUMER_MODEL, NVME_MODERN_MODEL, SATA_SSD_MODEL] {
            let shallow = model.random_penalty_at_depth(4_096, 1.0);
            let deep = model.random_penalty_at_depth(4_096, 128.0);
            assert!(
                deep < shallow / 4.0,
                "{}: penalty {shallow:.1}× at k=1 should fall far by k=128, got {deep:.1}×",
                model.name
            );
        }
    }

    /// The modern drive at high concurrency lands inside the 1.3–1.5×
    /// random-to-sequential band the MQSSD measurements report. That band
    /// is the calibration target, so this test is what keeps the modern
    /// regime honest.
    #[test]
    fn the_modern_drive_matches_the_published_random_sequential_ratio() {
        let ratio = NVME_MODERN_MODEL.random_penalty_at_depth(4_096, 128.0);
        assert!(
            (1.3..=1.5).contains(&ratio),
            "expected the published 1.3–1.5× band, got {ratio:.2}×"
        );
    }

    /// **The conclusion that changes.** At `iodepth=10` on a 2016 drive,
    /// ordering a 4 KiB-record rewrite pays given a budget of about a
    /// quarter of the payload. At realistic concurrency on a current
    /// drive it pays at no budget at all, because the penalty it would
    /// have to beat is under two.
    #[test]
    fn ordering_stops_paying_on_a_modern_drive_at_realistic_concurrency() {
        let payload = 1u64 << 40;

        let old = NVME_CONSUMER_MODEL.min_budget_for_ordering_at_depth(payload, 4_096, 10.0);
        assert!(
            old < payload / 3,
            "the 2016 drive at QD10 needs {old} bytes"
        );

        // On the modern drive the penalty drops below two, and a rewrite
        // always makes at least two passes. So no budget wins: the floor
        // of the pass count is already above the penalty it must beat.
        let penalty = NVME_MODERN_MODEL.random_penalty_at_depth(4_096, 128.0);
        assert!(
            penalty < 2.0,
            "penalty {penalty:.2}× is still beatable by a two-pass run"
        );
        assert!(!NVME_MODERN_MODEL.ordering_pays_at_depth(4_096, 2, 128.0));

        let modern = NVME_MODERN_MODEL.min_budget_for_ordering_at_depth(payload, 4_096, 128.0);
        assert!(
            modern > payload / 2,
            "and the nominal budget exceeds half the payload, at which point the \
             rewrite is nearly an in-memory sort anyway: {modern}"
        );
    }

    /// Small records still justify ordering everywhere, on every device
    /// and at every depth measured. The technique did not stop working;
    /// its domain narrowed to where records are small relative to what
    /// the device serves efficiently.
    #[test]
    fn small_records_still_justify_ordering_at_any_depth() {
        for depth in [1.0f64, 10.0, 128.0] {
            for model in ALL_MODELS.iter().chain([&NVME_MODERN_MODEL]) {
                let penalty = model.random_penalty_at_depth(128, depth);
                assert!(
                    penalty > 8.0,
                    "{} at k={depth}: 128 B records should still carry a large penalty, got {penalty:.1}×",
                    model.name
                );
            }
        }
    }
}
