// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pearson correlation between two numeric fields.
//!
//! Uses the numerically-stable Welford online covariance update
//! (Welford 1962, extended for pairs). For each record where both
//! fields are present we run a single update step; absent records
//! are ignored — Pearson r is defined on the joint observation set.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::{PairAnalyzer, PairAnalyzerKind, PairReport};
use crate::pipeline::commands::survey::measure::MeasureCtx;
use crate::pipeline::commands::survey::measures::numeric::mvalue_as_f64;

pub struct NumericCorrelationAnalyzer {
    n: u64,
    mean_a: f64,
    mean_b: f64,
    m2_a: f64,
    m2_b: f64,
    /// Comoment between A and B (centred running sum).
    c_ab: f64,
}

impl Default for NumericCorrelationAnalyzer {
    fn default() -> Self { Self::new() }
}

impl NumericCorrelationAnalyzer {
    pub fn new() -> Self {
        NumericCorrelationAnalyzer {
            n: 0,
            mean_a: 0.0,
            mean_b: 0.0,
            m2_a: 0.0,
            m2_b: 0.0,
            c_ab: 0.0,
        }
    }
}

impl PairAnalyzer for NumericCorrelationAnalyzer {
    fn observe_pair(&mut self, a: &MValue, b: &MValue, _ctx: &MeasureCtx) {
        let (Some(xa), Some(xb)) = (mvalue_as_f64(a), mvalue_as_f64(b)) else {
            return;
        };
        if xa.is_nan() || xb.is_nan() { return; }
        self.n += 1;
        let n = self.n as f64;
        let dx = xa - self.mean_a;
        let dy = xb - self.mean_b;
        self.mean_a += dx / n;
        self.mean_b += dy / n;
        let dy_after = xb - self.mean_b;
        self.m2_a += dx * (xa - self.mean_a);
        self.m2_b += dy * dy_after;
        self.c_ab += dx * dy_after;
    }

    fn finalize(self: Box<Self>) -> PairReport {
        let (pearson_r, slope, intercept) = if self.n < 2 || self.m2_a == 0.0 || self.m2_b == 0.0 {
            (None, None, None)
        } else {
            let r = self.c_ab / (self.m2_a.sqrt() * self.m2_b.sqrt());
            let s = self.c_ab / self.m2_a;
            let intc = self.mean_b - s * self.mean_a;
            (Some(r), Some(s), Some(intc))
        };
        PairReport::NumericCorrelation(NumericCorrelationReport {
            n: self.n,
            pearson_r,
            mean_a: self.mean_a,
            mean_b: self.mean_b,
            slope,
            intercept,
        })
    }

    fn kind(&self) -> PairAnalyzerKind {
        PairAnalyzerKind::NumericCorrelation
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericCorrelationReport {
    /// Number of paired observations contributing to the estimate.
    pub n: u64,
    /// Pearson correlation coefficient. `None` if n < 2 or either
    /// variance was zero.
    pub pearson_r: Option<f64>,
    pub mean_a: f64,
    pub mean_b: f64,
    /// Slope of the OLS regression line `B = slope · A + intercept`.
    pub slope: Option<f64>,
    pub intercept: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn perfect_positive_correlation() {
        let mut a = NumericCorrelationAnalyzer::new();
        // y = 2 · x + 3
        for x in 0..100 {
            let y = 2 * x + 3;
            a.observe_pair(&MValue::Int(x as i64), &MValue::Int(y as i64), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::NumericCorrelation(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!((r.pearson_r.unwrap() - 1.0).abs() < 1e-9, "r = {:?}", r.pearson_r);
        assert!((r.slope.unwrap() - 2.0).abs() < 1e-9);
        assert!((r.intercept.unwrap() - 3.0).abs() < 1e-9);
        assert_eq!(r.n, 100);
    }

    #[test]
    fn perfect_negative_correlation() {
        let mut a = NumericCorrelationAnalyzer::new();
        for x in 0..50 {
            a.observe_pair(&MValue::Int(x as i64), &MValue::Int(-(x as i64)), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::NumericCorrelation(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!((r.pearson_r.unwrap() + 1.0).abs() < 1e-9, "r = {:?}", r.pearson_r);
        assert!((r.slope.unwrap() + 1.0).abs() < 1e-9);
    }

    #[test]
    fn no_correlation_random_pairs() {
        let mut a = NumericCorrelationAnalyzer::new();
        let mut state: u64 = 42;
        for _ in 0..2_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (state >> 11) as f64 / (1u64 << 53) as f64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = (state >> 11) as f64 / (1u64 << 53) as f64;
            a.observe_pair(&MValue::Float(x), &MValue::Float(y), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::NumericCorrelation(r) => r,
            _ => panic!("wrong kind"),
        };
        // Two independent uniform streams → r near 0. Loose tolerance.
        assert!(r.pearson_r.unwrap().abs() < 0.1, "r = {:?} should be near 0", r.pearson_r);
    }

    #[test]
    fn ignores_non_numeric_pairs() {
        let mut a = NumericCorrelationAnalyzer::new();
        a.observe_pair(&MValue::Text("oops".into()), &MValue::Int(1), &ctx());
        a.observe_pair(&MValue::Int(1), &MValue::Text("oops".into()), &ctx());
        let r = match Box::new(a).finalize() {
            PairReport::NumericCorrelation(r) => r,
            _ => panic!("wrong kind"),
        };
        assert_eq!(r.n, 0);
        assert!(r.pearson_r.is_none());
    }

    #[test]
    fn constant_field_gives_none() {
        let mut a = NumericCorrelationAnalyzer::new();
        for x in 0..50 {
            a.observe_pair(&MValue::Int(5), &MValue::Int(x as i64), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::NumericCorrelation(r) => r,
            _ => panic!("wrong kind"),
        };
        // One field is constant → variance zero → r undefined.
        assert!(r.pearson_r.is_none());
    }
}
