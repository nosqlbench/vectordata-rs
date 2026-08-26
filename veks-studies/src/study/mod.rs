// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Parameter sweeps that put the published formulas on trial.
//!
//! Each study runs the simulator across a range of inputs and reports
//! measured cost beside predicted cost. A formula that is wrong shows up
//! as a column that does not line up, which is the whole point: the
//! documents assert `A(P) = P · (1 − exp(−w / P))` and nothing but a
//! measurement can contradict them.
//!
//! The analysis these sweeps test extends the external-memory model of
//! [Aggarwal & Vitter (CACM 31(9), 1988)](https://dl.acm.org/doi/10.1145/48529.48535),
//! which established the sorting and permutation I/O bounds this family
//! of algorithms lives inside. Device figures come from the corpus named
//! in [the crate bibliography](crate#sources).

use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};
use crate::model::{Geometry, Map, Metrics};

/// One row of an amplification sweep.
#[derive(Debug, Clone, Copy)]
pub struct AmplificationRow {
    pub budget_bytes: u64,
    pub passes: u64,
    pub measured: f64,
    pub predicted: f64,
}

impl AmplificationRow {
    /// Relative error of the prediction against the measurement.
    pub fn relative_error(&self) -> f64 {
        if self.measured == 0.0 {
            return 0.0;
        }
        (self.predicted - self.measured).abs() / self.measured
    }
}

/// Sweep the memory budget over a random permutation, measuring the read
/// amplification gsplat actually incurs and comparing it against the
/// formula.
///
/// The formula is derived for a uniform random permutation, so that is
/// what is used; a structured map would legitimately disagree with it.
pub fn amplification_sweep(
    geometry: Geometry,
    seed: u64,
    budgets: &[u64],
) -> Vec<AmplificationRow> {
    let map = Map::shuffled(geometry.records, seed);
    let algo = Gsplat::new();
    budgets
        .iter()
        .map(|&budget_bytes| {
            let (_, trace) = algo.run(geometry, &map, budget_bytes);
            let m = trace.metrics();
            AmplificationRow {
                budget_bytes,
                passes: m.passes,
                measured: m.amplification(),
                predicted: m.predicted_amplification(),
            }
        })
        .collect()
}

/// A head-to-head of the rewrites at one operating point.
#[derive(Debug, Clone, Copy)]
pub struct Comparison {
    pub name: &'static str,
    pub metrics: Metrics,
}

/// Run every rewrite over the same inputs so their traces can be
/// compared directly.
pub fn compare_all(geometry: Geometry, map: &Map, budget_bytes: u64) -> Vec<Comparison> {
    let algos: Vec<Box<dyn Rewrite>> = vec![Box::new(NaiveGather), Box::new(Gsplat::new())];
    algos
        .into_iter()
        .map(|a| {
            let (_, trace) = a.run(geometry, map, budget_bytes);
            Comparison {
                name: a.name(),
                metrics: trace.metrics(),
            }
        })
        .collect()
}

/// Render an amplification sweep as a table.
pub fn render_amplification(geometry: Geometry, rows: &[AmplificationRow]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "records {}  ·  R {}B  ·  W {}B  ·  w {}  ·  live {:.2} MiB\n",
        geometry.records,
        geometry.record_bytes,
        geometry.container_bytes,
        geometry.records_per_container(),
        geometry.payload_bytes() as f64 / (1024.0 * 1024.0),
    ));
    s.push_str("\n  budget        P   measured A   predicted A   error\n");
    s.push_str("  ----------  ---   ----------   -----------   -----\n");
    for r in rows {
        s.push_str(&format!(
            "  {:>9}  {:>4}   {:>10.2}   {:>11.2}   {:>4.1}%\n",
            human_bytes(r.budget_bytes),
            r.passes,
            r.measured,
            r.predicted,
            r.relative_error() * 100.0,
        ));
    }
    s
}

/// Render a rewrite comparison as a table.
pub fn render_comparison(rows: &[Comparison]) -> String {
    let mut s = String::new();
    s.push_str("\n  rewrite         passes   reads   touches   backward   bytes read\n");
    s.push_str("  -------------   ------   -----   -------   --------   ----------\n");
    for r in rows {
        s.push_str(&format!(
            "  {:<13}   {:>6}   {:>5}   {:>7}   {:>8}   {:>10}\n",
            r.name,
            r.metrics.passes,
            r.metrics.record_reads,
            r.metrics.container_touches,
            r.metrics.backward_steps,
            human_bytes(r.metrics.bytes_read()),
        ));
    }
    s
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [(u64, &str); 4] = [
        (1024 * 1024 * 1024, "GiB"),
        (1024 * 1024, "MiB"),
        (1024, "KiB"),
        (1, "B"),
    ];
    for (scale, unit) in UNITS {
        if bytes >= scale {
            let v = bytes as f64 / scale as f64;
            return if v >= 100.0 {
                format!("{v:.0} {unit}")
            } else {
                format!("{v:.1} {unit}")
            };
        }
    }
    format!("{bytes} B")
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;

    /// The headline check: does the published amplification formula
    /// describe what the algorithm actually does?
    ///
    /// The formula assumes a uniform random permutation and treats
    /// container occupancy as independent, so exact agreement is not
    /// expected — but a formula that is *structurally* right should
    /// track the measurement closely across the whole range, including
    /// through the dense-to-sparse crossover at `P = w`.
    #[test]
    fn the_amplification_formula_tracks_measurement() {
        // w = 16, so the crossover sits at P = 16 and the sweep spans
        // both regimes.
        let g = Geometry::new(20_000, 64, 1_024);
        assert_eq!(g.records_per_container(), 16);

        let live = g.payload_bytes();
        let budgets: Vec<u64> = [2u64, 4, 8, 16, 32, 64, 128]
            .iter()
            .map(|p| live / p)
            .collect();

        let rows = amplification_sweep(g, 1234, &budgets);
        for r in &rows {
            assert!(
                r.relative_error() < 0.10,
                "P={} measured {:.3} vs predicted {:.3} — {:.1}% off",
                r.passes,
                r.measured,
                r.predicted,
                r.relative_error() * 100.0
            );
        }
    }

    /// The dense regime's claim: below the crossover, amplification is
    /// approximately the pass count, because every container is needed
    /// on every pass.
    #[test]
    fn the_dense_regime_amplifies_by_about_the_pass_count() {
        let g = Geometry::new(20_000, 64, 1_024); // w = 16
        let live = g.payload_bytes();
        let rows = amplification_sweep(g, 99, &[live / 2, live / 4]);
        for r in rows {
            assert!(
                (r.measured - r.passes as f64).abs() / (r.passes as f64) < 0.05,
                "P={} should amplify by about P, measured {:.2}",
                r.passes,
                r.measured
            );
        }
    }

    /// The sparse regime's claim: far above the crossover, amplification
    /// saturates near `w` rather than continuing to grow with `P`.
    #[test]
    fn the_sparse_regime_saturates_near_w() {
        let g = Geometry::new(20_000, 64, 1_024); // w = 16
        let live = g.payload_bytes();
        let rows = amplification_sweep(g, 7, &[live / 256, live / 512]);
        let w = g.records_per_container() as f64;
        for r in rows {
            assert!(
                r.measured > w * 0.75 && r.measured <= w * 1.01,
                "P={} should saturate near w={w}, measured {:.2}",
                r.passes,
                r.measured
            );
        }
    }

    /// The cost model's central asymmetry: gsplat and naive issue the
    /// same number of record reads, and differ in ordering.
    #[test]
    fn gsplat_and_naive_issue_the_same_read_count() {
        let g = Geometry::new(4_000, 64, 1_024);
        let map = Map::shuffled(4_000, 3);
        let rows = compare_all(g, &map, g.payload_bytes() / 8);
        let naive = rows.iter().find(|r| r.name == "naive-gather").unwrap();
        let splat = rows.iter().find(|r| r.name == "gsplat").unwrap();

        assert_eq!(
            naive.metrics.record_reads, splat.metrics.record_reads,
            "the documents claim read counts are equal"
        );
        assert!(
            naive.metrics.backward_steps > 0 && splat.metrics.backward_steps == 0,
            "the difference is ordering, not count"
        );
        assert!(
            naive.metrics.container_touches > splat.metrics.container_touches,
            "scattered reads re-enter containers: naive {} vs gsplat {}",
            naive.metrics.container_touches,
            splat.metrics.container_touches
        );
    }
}

/// A worked example at a scale too large to trace operation by operation.
///
/// Everything here is analytic, using the amplification formula the
/// traced simulations confirmed and the device models the measured
/// sweeps confirmed. That combination is what makes it legitimate to
/// quote figures for a 2 TB rewrite without simulating 450 million reads:
/// both halves were validated at sizes where full simulation was possible.
#[derive(Debug, Clone, Copy)]
pub struct WorkedExample {
    pub label: &'static str,
    pub records: u64,
    pub record_bytes: u64,
    pub container_bytes: u64,
    pub budget_bytes: u64,
}

impl WorkedExample {
    pub fn passes(&self) -> u64 {
        let per_segment = (self.budget_bytes / self.record_bytes).max(1);
        self.records.div_ceil(per_segment).max(2)
    }

    pub fn records_per_container(&self) -> u64 {
        (self.container_bytes / self.record_bytes).max(1)
    }

    pub fn amplification(&self) -> f64 {
        let p = self.passes() as f64;
        let w = self.records_per_container() as f64;
        p * (1.0 - (-w / p).exp())
    }

    pub fn payload_bytes(&self) -> u64 {
        self.records * self.record_bytes
    }

    /// Seconds for a gather that reads each record where it lies.
    pub fn naive_seconds(&self, model: &crate::device::DeviceModel) -> f64 {
        model.random_read_seconds(self.records, self.record_bytes)
            + model.sequential_seconds(self.payload_bytes())
    }

    /// Seconds for an ordered rewrite reading whole containers.
    ///
    /// **A pass is never more expensive than a full sequential scan.**
    /// Because a pass visits containers in ascending order, a reader that
    /// finds itself wanting most of them can simply stream the source and
    /// discard what it does not need, paying the sequential rate rather
    /// than a seek per container. That option is always available, so it
    /// caps the cost — and in the dense regime, where each pass touches
    /// nearly every container, it is the option that wins.
    ///
    /// Leaving this bound out is what made an earlier version of this
    /// model quote 27 hours for a rewrite that streams in four: it priced
    /// a sequential scan as though it were a million independent seeks.
    pub fn gsplat_seconds(&self, model: &crate::device::DeviceModel) -> f64 {
        let containers = self.payload_bytes().div_ceil(self.container_bytes);
        let touches = (self.amplification() * containers as f64).round() as u64;
        let passes = self.passes();

        let streaming = model.sequential_seconds(self.payload_bytes());
        let seeking = model.random_read_seconds(touches / passes.max(1), self.container_bytes);
        let per_pass = streaming.min(seeking);

        per_pass * passes as f64 + model.sequential_seconds(self.payload_bytes())
    }

    /// Whether each pass is cheaper streamed than seeked — true in the
    /// dense regime, false once passes touch only a sparse subset.
    pub fn passes_stream(&self, model: &crate::device::DeviceModel) -> bool {
        let containers = self.payload_bytes().div_ceil(self.container_bytes);
        let touches = (self.amplification() * containers as f64).round() as u64;
        let streaming = self.payload_bytes() as f64 / model.bandwidth(self.container_bytes);
        let seeking =
            model.random_read_seconds(touches / self.passes().max(1), self.container_bytes);
        streaming <= seeking
    }

    pub fn gsplat_bytes_read(&self) -> u64 {
        (self.amplification() * self.payload_bytes() as f64) as u64
    }
}

fn human_seconds(s: f64) -> String {
    if s < 90.0 {
        format!("{s:.0} s")
    } else if s < 5_400.0 {
        format!("{:.0} min", s / 60.0)
    } else if s < 172_800.0 {
        format!("{:.1} h", s / 3_600.0)
    } else {
        format!("{:.1} days", s / 86_400.0)
    }
}

/// Render worked examples against every validated device model.
pub fn render_worked_examples(examples: &[WorkedExample]) -> String {
    use std::fmt::Write as _;
    let models = crate::device::ALL_MODELS;
    let mut s = String::new();

    for ex in examples {
        let _ = writeln!(
            s,
            "\n{} — {} records × {} B = {}, budget {}",
            ex.label,
            ex.records,
            ex.record_bytes,
            human_bytes(ex.payload_bytes()),
            human_bytes(ex.budget_bytes)
        );
        let _ = writeln!(
            s,
            "  P = {}   w = {}   A = {:.1}   ({})",
            ex.passes(),
            ex.records_per_container(),
            ex.amplification(),
            if ex.passes() <= ex.records_per_container() {
                "dense"
            } else {
                "sparse"
            }
        );
        let _ = writeln!(
            s,
            "\n  {:<16} {:>12} {:>12} {:>12} {:>12}",
            "", "bytes read", models[0].name, models[1].name, models[2].name
        );
        let _ = writeln!(
            s,
            "  {:<16} {:>12} {:>12} {:>12} {:>12}",
            "naive gather",
            human_bytes(ex.payload_bytes()),
            human_seconds(ex.naive_seconds(&models[0])),
            human_seconds(ex.naive_seconds(&models[1])),
            human_seconds(ex.naive_seconds(&models[2]))
        );
        let _ = writeln!(
            s,
            "  {:<16} {:>12} {:>12} {:>12} {:>12}",
            "gsplat",
            human_bytes(ex.gsplat_bytes_read()),
            human_seconds(ex.gsplat_seconds(&models[0])),
            human_seconds(ex.gsplat_seconds(&models[1])),
            human_seconds(ex.gsplat_seconds(&models[2]))
        );
        let _ = writeln!(
            s,
            "  {:<16} {:>12} {:>12.1} {:>12.1} {:>12.1}",
            "speedup",
            "",
            ex.naive_seconds(&models[0]) / ex.gsplat_seconds(&models[0]),
            ex.naive_seconds(&models[1]) / ex.gsplat_seconds(&models[1]),
            ex.naive_seconds(&models[2]) / ex.gsplat_seconds(&models[2])
        );
    }
    s
}
