// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Parameter sweeps that put the published formulas on trial.
//!
//! Each study runs the simulator across a range of inputs and reports
//! measured cost beside predicted cost. A formula that is wrong shows up
//! as a column that does not line up, which is the whole point: the
//! documents assert `A(P) = P · (1 − exp(−w / P))` and nothing but a
//! measurement can contradict them.

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
pub fn amplification_sweep(geometry: Geometry, seed: u64, budgets: &[u64]) -> Vec<AmplificationRow> {
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
            Comparison { name: a.name(), metrics: trace.metrics() }
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

#[cfg(test)]
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
