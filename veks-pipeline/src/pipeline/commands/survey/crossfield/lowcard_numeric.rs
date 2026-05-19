// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Variance-explained analysis for low-card categorical × numeric
//! pairs: how much of the numeric field's variance is accounted for
//! by membership in the categorical group?
//!
//! Reports the η² ("eta-squared") statistic from one-way ANOVA:
//! `η² = SS_between / SS_total ∈ [0, 1]`. 0 means group membership
//! tells you nothing about the numeric value; 1 means it tells you
//! everything.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::{PairAnalyzer, PairAnalyzerKind, PairReport};
use crate::pipeline::commands::survey::measure::MeasureCtx;
use crate::pipeline::commands::survey::measures::cardinality::canonical_distinct_key;
use crate::pipeline::commands::survey::measures::numeric::mvalue_as_f64;

pub struct LowCardNumericAnalyzer {
    /// `categorical_key → (count, running_mean, m2)` per group.
    groups: IndexMap<String, GroupStats>,
    n: u64,
    /// Pooled grand mean / m2 (Welford).
    grand_mean: f64,
    grand_m2: f64,
    /// True if categorical is on the A side; false if on the B side.
    /// Set by the orchestrator at construction (via `with_category_side`)
    /// so the analyzer knows which arg of `observe_pair` to read.
    category_on_a: bool,
}

struct GroupStats {
    n: u64,
    mean: f64,
    m2: f64,
}

impl LowCardNumericAnalyzer {
    pub fn new(category_on_a: bool) -> Self {
        LowCardNumericAnalyzer {
            groups: IndexMap::new(),
            n: 0,
            grand_mean: 0.0,
            grand_m2: 0.0,
            category_on_a,
        }
    }
}

impl PairAnalyzer for LowCardNumericAnalyzer {
    fn observe_pair(&mut self, a: &MValue, b: &MValue, _ctx: &MeasureCtx) {
        let (cat_value, num_value) = if self.category_on_a {
            (a, b)
        } else {
            (b, a)
        };
        let Some(x) = mvalue_as_f64(num_value) else { return };
        if x.is_nan() { return }
        let key = canonical_distinct_key(cat_value);
        // Grand-mean Welford
        self.n += 1;
        let dx = x - self.grand_mean;
        self.grand_mean += dx / self.n as f64;
        self.grand_m2 += dx * (x - self.grand_mean);
        // Per-group Welford
        let g = self.groups.entry(key).or_insert(GroupStats {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        });
        g.n += 1;
        let dg = x - g.mean;
        g.mean += dg / g.n as f64;
        g.m2 += dg * (x - g.mean);
    }

    fn finalize(self: Box<Self>) -> PairReport {
        if self.n < 2 || self.groups.len() < 2 || self.grand_m2 == 0.0 {
            return PairReport::LowCardNumeric(LowCardNumericReport {
                n: self.n,
                groups: self.groups.len() as u32,
                eta_squared: None,
                ss_total: self.grand_m2,
                ss_between: 0.0,
                ss_within: self.grand_m2,
                group_means: IndexMap::new(),
            });
        }
        // SS_between = Σ_g n_g · (mean_g - grand_mean)²
        let mut ss_between = 0f64;
        let mut group_means = IndexMap::with_capacity(self.groups.len());
        for (key, g) in &self.groups {
            let diff = g.mean - self.grand_mean;
            ss_between += g.n as f64 * diff * diff;
            group_means.insert(key.clone(), g.mean);
        }
        let ss_total = self.grand_m2;
        let ss_within = (ss_total - ss_between).max(0.0);
        let eta_squared = Some((ss_between / ss_total).clamp(0.0, 1.0));
        PairReport::LowCardNumeric(LowCardNumericReport {
            n: self.n,
            groups: self.groups.len() as u32,
            eta_squared,
            ss_total,
            ss_between,
            ss_within,
            group_means,
        })
    }

    fn kind(&self) -> PairAnalyzerKind {
        PairAnalyzerKind::LowCardNumeric
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LowCardNumericReport {
    pub n: u64,
    pub groups: u32,
    /// η² ∈ [0, 1]. 0 = category explains no variance; 1 = perfect.
    /// `None` when n < 2 or fewer than 2 groups or zero total
    /// variance.
    pub eta_squared: Option<f64>,
    pub ss_total: f64,
    pub ss_between: f64,
    pub ss_within: f64,
    pub group_means: IndexMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    /// Numeric value is constant per group, different across groups
    /// → η² = 1.
    #[test]
    fn perfect_explanation() {
        let mut a = LowCardNumericAnalyzer::new(true);
        for i in 0..100 {
            let cat = i % 3;
            let value = (cat as i64) * 100;
            a.observe_pair(&MValue::Int(cat as i64), &MValue::Int(value), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::LowCardNumeric(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!((r.eta_squared.unwrap() - 1.0).abs() < 1e-9, "η² = {:?}", r.eta_squared);
    }

    /// Numeric distribution identical per group → η² = 0.
    #[test]
    fn no_explanation() {
        let mut a = LowCardNumericAnalyzer::new(true);
        // 4 groups, each gets the same 0..100 distribution.
        for cat in 0..4 {
            for v in 0..100 {
                a.observe_pair(&MValue::Int(cat as i64), &MValue::Int(v as i64), &ctx());
            }
        }
        let r = match Box::new(a).finalize() {
            PairReport::LowCardNumeric(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!(r.eta_squared.unwrap() < 1e-9, "η² should be ~0, got {:?}", r.eta_squared);
    }

    /// Single group → undefined (NaN).
    #[test]
    fn single_group_undefined() {
        let mut a = LowCardNumericAnalyzer::new(true);
        for i in 0..50 {
            a.observe_pair(&MValue::Text("only".into()), &MValue::Int(i), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::LowCardNumeric(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!(r.eta_squared.is_none());
    }

    /// Category-on-B mode works symmetrically.
    #[test]
    fn category_on_b_side() {
        let mut a = LowCardNumericAnalyzer::new(false);
        for i in 0..60 {
            let cat = i % 2;
            let value = cat as i64 * 50;
            a.observe_pair(&MValue::Int(value), &MValue::Int(cat as i64), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::LowCardNumeric(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!((r.eta_squared.unwrap() - 1.0).abs() < 1e-9);
    }
}
