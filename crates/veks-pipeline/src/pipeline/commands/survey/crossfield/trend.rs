// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Trend analyzer — Pearson correlation of a numeric field's
//! values against the record index. Used to flag fields that drift
//! systematically with stream position (clock-driven IDs, growing
//! counts, etc.).
//!
//! Implemented as a special-cased pair analyzer: the `b` value
//! passed to `observe_pair` is the record index encoded as
//! `MValue::Int(record_index)` by the orchestrator. The "pair" is
//! really `(field_value, record_index)`.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::{PairAnalyzer, PairAnalyzerKind, PairReport};
use crate::pipeline::commands::survey::measure::MeasureCtx;
use crate::pipeline::commands::survey::measures::numeric::mvalue_as_f64;

pub struct TrendAnalyzer {
    n: u64,
    mean_v: f64,
    mean_i: f64,
    m2_v: f64,
    m2_i: f64,
    c: f64,
}

impl Default for TrendAnalyzer {
    fn default() -> Self { Self::new() }
}

impl TrendAnalyzer {
    pub fn new() -> Self {
        TrendAnalyzer {
            n: 0,
            mean_v: 0.0,
            mean_i: 0.0,
            m2_v: 0.0,
            m2_i: 0.0,
            c: 0.0,
        }
    }
}

impl PairAnalyzer for TrendAnalyzer {
    fn observe_pair(&mut self, a: &MValue, _b: &MValue, ctx: &MeasureCtx) {
        let Some(v) = mvalue_as_f64(a) else { return };
        if v.is_nan() { return }
        // The orchestrator drives Trend by calling
        // `observe_pair(value, /* unused */, ctx)` and we read the
        // record index out of ctx.
        let i = ctx.record_index as f64;
        self.n += 1;
        let n = self.n as f64;
        let dv = v - self.mean_v;
        let di = i - self.mean_i;
        self.mean_v += dv / n;
        self.mean_i += di / n;
        let di_after = i - self.mean_i;
        self.m2_v += dv * (v - self.mean_v);
        self.m2_i += di * di_after;
        self.c += dv * di_after;
    }

    fn finalize(self: Box<Self>) -> PairReport {
        let (r, slope) = if self.n < 2 || self.m2_v == 0.0 || self.m2_i == 0.0 {
            (None, None)
        } else {
            let rr = self.c / (self.m2_v.sqrt() * self.m2_i.sqrt());
            let s = self.c / self.m2_i;
            (Some(rr), Some(s))
        };
        PairReport::Trend(TrendReport {
            n: self.n,
            pearson_r_with_index: r,
            slope_per_record: slope,
        })
    }

    fn kind(&self) -> PairAnalyzerKind {
        PairAnalyzerKind::Trend
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrendReport {
    pub n: u64,
    /// Pearson r between the field's values and the record index.
    /// Near ±1 → strong monotone drift with stream position. `None`
    /// when n < 2 or the field is constant.
    pub pearson_r_with_index: Option<f64>,
    /// OLS slope (`value` per unit `record_index`).
    pub slope_per_record: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(i: u64) -> MeasureCtx<'static> {
        MeasureCtx { record_index: i, semantic_type: None }
    }

    /// Monotone-increasing value → r ≈ 1.
    #[test]
    fn monotone_value_strong_trend() {
        let mut a = TrendAnalyzer::new();
        for i in 0..100u64 {
            a.observe_pair(&MValue::Int(i as i64), &MValue::Null, &ctx(i));
        }
        let r = match Box::new(a).finalize() {
            PairReport::Trend(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!((r.pearson_r_with_index.unwrap() - 1.0).abs() < 1e-9);
        assert!((r.slope_per_record.unwrap() - 1.0).abs() < 1e-9);
    }

    /// Random walk → r near 0.
    #[test]
    fn random_value_no_trend() {
        let mut a = TrendAnalyzer::new();
        let mut state: u64 = 7;
        for i in 0..2_000u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (state >> 11) as f64 / (1u64 << 53) as f64;
            a.observe_pair(&MValue::Float(v), &MValue::Null, &ctx(i));
        }
        let r = match Box::new(a).finalize() {
            PairReport::Trend(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!(r.pearson_r_with_index.unwrap().abs() < 0.1, "r = {:?}", r.pearson_r_with_index);
    }

    /// Constant value → r undefined.
    #[test]
    fn constant_value_nan() {
        let mut a = TrendAnalyzer::new();
        for i in 0..50u64 {
            a.observe_pair(&MValue::Int(7), &MValue::Null, &ctx(i));
        }
        let r = match Box::new(a).finalize() {
            PairReport::Trend(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!(r.pearson_r_with_index.is_none());
    }
}
