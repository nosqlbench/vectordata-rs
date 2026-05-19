// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Co-presence analyzer.
//!
//! Tracks per-record presence overlap between two fields, reports
//! both conditional probabilities and the Jaccard coefficient.
//! Cheapest of the cross-field analyzers — runs against every
//! eligible pair regardless of regime, since presence information
//! is useful even when richer statistics aren't.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::{PairAnalyzer, PairAnalyzerKind, PairReport};
use crate::pipeline::commands::survey::measure::MeasureCtx;

pub struct CopresenceAnalyzer {
    both: u64,
    a_only: u64,
    b_only: u64,
    neither: u64,
}

impl Default for CopresenceAnalyzer {
    fn default() -> Self { Self::new() }
}

impl CopresenceAnalyzer {
    pub fn new() -> Self {
        CopresenceAnalyzer {
            both: 0,
            a_only: 0,
            b_only: 0,
            neither: 0,
        }
    }
}

impl PairAnalyzer for CopresenceAnalyzer {
    fn observe_pair(&mut self, _a: &MValue, _b: &MValue, _ctx: &MeasureCtx) {
        self.both += 1;
    }

    fn observe_missing(&mut self, a_present: bool, b_present: bool, _ctx: &MeasureCtx) {
        match (a_present, b_present) {
            (true, false) => self.a_only += 1,
            (false, true) => self.b_only += 1,
            (false, false) => self.neither += 1,
            // `(true, true)` is handled by `observe_pair`.
            (true, true) => {}
        }
    }

    fn finalize(self: Box<Self>) -> PairReport {
        let union = self.both + self.a_only + self.b_only;
        let total = union + self.neither;
        let jaccard = if union == 0 { 0.0 } else { self.both as f64 / union as f64 };
        let a_total = self.both + self.a_only;
        let b_total = self.both + self.b_only;
        let p_a_given_b = if b_total == 0 { 0.0 } else { self.both as f64 / b_total as f64 };
        let p_b_given_a = if a_total == 0 { 0.0 } else { self.both as f64 / a_total as f64 };
        PairReport::Copresence(CopresenceReport {
            records: total,
            both: self.both,
            a_only: self.a_only,
            b_only: self.b_only,
            neither: self.neither,
            jaccard,
            p_a_given_b,
            p_b_given_a,
        })
    }

    fn kind(&self) -> PairAnalyzerKind {
        PairAnalyzerKind::Copresence
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopresenceReport {
    /// Total records observed (sum of the four cells).
    pub records: u64,
    /// Both fields present in the record.
    pub both: u64,
    /// Only A present, B absent.
    pub a_only: u64,
    /// Only B present, A absent.
    pub b_only: u64,
    /// Both absent.
    pub neither: u64,
    /// `|A ∩ B| / |A ∪ B|`.
    pub jaccard: f64,
    /// `P(A | B)`.
    pub p_a_given_b: f64,
    /// `P(B | A)`.
    pub p_b_given_a: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn jaccard_perfect_overlap() {
        let mut a = CopresenceAnalyzer::new();
        for _ in 0..10 {
            a.observe_pair(&MValue::Int(1), &MValue::Int(1), &ctx());
        }
        let r = match Box::new(a).finalize() {
            PairReport::Copresence(r) => r,
            _ => panic!("wrong kind"),
        };
        assert_eq!(r.jaccard, 1.0);
        assert_eq!(r.p_a_given_b, 1.0);
        assert_eq!(r.p_b_given_a, 1.0);
        assert_eq!(r.records, 10);
    }

    #[test]
    fn no_overlap() {
        let mut a = CopresenceAnalyzer::new();
        for _ in 0..5 { a.observe_missing(true, false, &ctx()); }
        for _ in 0..5 { a.observe_missing(false, true, &ctx()); }
        let r = match Box::new(a).finalize() {
            PairReport::Copresence(r) => r,
            _ => panic!("wrong kind"),
        };
        assert_eq!(r.both, 0);
        assert_eq!(r.jaccard, 0.0);
        assert_eq!(r.p_a_given_b, 0.0);
        assert_eq!(r.p_b_given_a, 0.0);
    }

    #[test]
    fn conditional_probabilities() {
        let mut a = CopresenceAnalyzer::new();
        // 3 both, 7 A-only, 2 B-only, 88 neither.
        for _ in 0..3 { a.observe_pair(&MValue::Int(1), &MValue::Int(1), &ctx()); }
        for _ in 0..7 { a.observe_missing(true, false, &ctx()); }
        for _ in 0..2 { a.observe_missing(false, true, &ctx()); }
        for _ in 0..88 { a.observe_missing(false, false, &ctx()); }
        let r = match Box::new(a).finalize() {
            PairReport::Copresence(r) => r,
            _ => panic!("wrong kind"),
        };
        // |A|=10, |B|=5, |A∩B|=3, |A∪B|=12, total=100.
        assert_eq!(r.records, 100);
        assert!((r.jaccard - 0.25).abs() < 1e-9); // 3/12
        assert!((r.p_a_given_b - 0.6).abs() < 1e-9); // 3/5
        assert!((r.p_b_given_a - 0.3).abs() < 1e-9); // 3/10
    }
}
