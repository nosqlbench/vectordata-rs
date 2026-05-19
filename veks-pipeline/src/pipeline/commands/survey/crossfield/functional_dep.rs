// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Approximate functional dependency detection.
//!
//! For each value of A, tracks how many distinct B values appear in
//! records sharing that A. If every (or nearly every) A maps to a
//! single B, A "functionally determines" B (`A → B`).
//!
//! The report carries the **support** — the fraction of records
//! where the A→B mapping holds — and the count of distinct A
//! values participating. Bounded memory: the per-A distinct-B set
//! is capped at `distinct_cap`; on overflow that A is reported as
//! "non-deterministic" but counts are preserved.

use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::{PairAnalyzer, PairAnalyzerKind, PairReport};
use crate::pipeline::commands::survey::measure::MeasureCtx;
use crate::pipeline::commands::survey::measures::cardinality::canonical_distinct_key;

pub struct FunctionalDependencyAnalyzer {
    /// `A_value → set of observed B_values (bounded)`. Once a set
    /// exceeds `distinct_cap`, it's promoted to None (= overflow).
    map: IndexMap<String, Option<IndexSet<String>>>,
    distinct_cap: usize,
    n: u64,
    /// Counts how many records had the *dominant* B for their A.
    /// Computed at finalize time using the mode of each per-A set.
    /// During observation we just maintain (a, b) → count.
    pair_counts: IndexMap<(String, String), u64>,
    per_a_counts: IndexMap<String, u64>,
}

impl FunctionalDependencyAnalyzer {
    pub fn new() -> Self {
        Self::with_cap(256)
    }
    pub fn with_cap(distinct_cap: usize) -> Self {
        FunctionalDependencyAnalyzer {
            map: IndexMap::new(),
            distinct_cap,
            n: 0,
            pair_counts: IndexMap::new(),
            per_a_counts: IndexMap::new(),
        }
    }
}

impl Default for FunctionalDependencyAnalyzer {
    fn default() -> Self { Self::new() }
}

impl PairAnalyzer for FunctionalDependencyAnalyzer {
    fn observe_pair(&mut self, a: &MValue, b: &MValue, _ctx: &MeasureCtx) {
        let ka = canonical_distinct_key(a);
        let kb = canonical_distinct_key(b);
        self.n += 1;
        *self.per_a_counts.entry(ka.clone()).or_insert(0) += 1;
        *self.pair_counts.entry((ka.clone(), kb.clone())).or_insert(0) += 1;
        let entry = self.map.entry(ka).or_insert_with(|| Some(IndexSet::new()));
        if let Some(set) = entry {
            if set.len() < self.distinct_cap {
                set.insert(kb);
            } else if !set.contains(&kb) {
                *entry = None; // overflow
            }
        }
    }

    fn finalize(self: Box<Self>) -> PairReport {
        let n = self.n;
        let distinct_a = self.map.len() as u32;
        // For each A value, find the mode of B and accumulate
        // "consistent" counts.
        let mut consistent_records: u64 = 0;
        let mut deterministic_a: u32 = 0;
        for (a, b_set) in &self.map {
            let total_a = *self.per_a_counts.get(a).unwrap_or(&0);
            if total_a == 0 { continue; }
            // Find the most-frequent B for this A.
            let mut best_count = 0u64;
            for ((aa, _bb), c) in &self.pair_counts {
                if aa == a && *c > best_count {
                    best_count = *c;
                }
            }
            consistent_records += best_count;
            if let Some(set) = b_set {
                if set.len() == 1 {
                    deterministic_a += 1;
                }
            }
        }
        let support = if n == 0 { 0.0 } else { consistent_records as f64 / n as f64 };
        PairReport::FunctionalDependency(FunctionalDependencyReport {
            n,
            distinct_a,
            deterministic_a,
            support,
        })
    }

    fn kind(&self) -> PairAnalyzerKind {
        PairAnalyzerKind::FunctionalDependency
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionalDependencyReport {
    pub n: u64,
    /// Distinct A values observed.
    pub distinct_a: u32,
    /// Number of A values for which every record had the same B
    /// (true functional dependency on that subset).
    pub deterministic_a: u32,
    /// Fraction of records that map A to A's most-common B.
    /// `1.0` means A → B is perfectly deterministic across the
    /// whole sample.
    pub support: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn perfect_fd() {
        let mut a = FunctionalDependencyAnalyzer::new();
        // Country code → currency: 5 countries, each maps to one
        // currency, observed 20 times each.
        let mapping: &[(&str, &str)] = &[
            ("US", "USD"), ("GB", "GBP"), ("DE", "EUR"), ("JP", "JPY"), ("BR", "BRL"),
        ];
        for (cc, cur) in mapping {
            for _ in 0..20 {
                a.observe_pair(&MValue::Text((*cc).into()), &MValue::Text((*cur).into()), &ctx());
            }
        }
        let r = match Box::new(a).finalize() {
            PairReport::FunctionalDependency(r) => r,
            _ => panic!("wrong kind"),
        };
        assert_eq!(r.distinct_a, 5);
        assert_eq!(r.deterministic_a, 5);
        assert!((r.support - 1.0).abs() < 1e-9);
    }

    /// Imperfect FD: support < 1.
    #[test]
    fn imperfect_fd() {
        let mut a = FunctionalDependencyAnalyzer::new();
        // US → USD 19 times, US → EUR once.
        for _ in 0..19 {
            a.observe_pair(&MValue::Text("US".into()), &MValue::Text("USD".into()), &ctx());
        }
        a.observe_pair(&MValue::Text("US".into()), &MValue::Text("EUR".into()), &ctx());
        let r = match Box::new(a).finalize() {
            PairReport::FunctionalDependency(r) => r,
            _ => panic!("wrong kind"),
        };
        // 19/20 = 0.95 support, distinct_a = 1, deterministic_a = 0
        // (since US maps to 2 distinct currencies, not just one).
        assert_eq!(r.distinct_a, 1);
        assert_eq!(r.deterministic_a, 0);
        assert!((r.support - 0.95).abs() < 1e-9);
    }

    /// No relationship: every A maps to every B uniformly.
    #[test]
    fn no_relationship() {
        let mut a = FunctionalDependencyAnalyzer::new();
        for ai in 0..4 {
            for bi in 0..4 {
                a.observe_pair(&MValue::Int(ai), &MValue::Int(bi), &ctx());
            }
        }
        let r = match Box::new(a).finalize() {
            PairReport::FunctionalDependency(r) => r,
            _ => panic!("wrong kind"),
        };
        assert_eq!(r.deterministic_a, 0);
        // Each A maps to 4 distinct Bs equally; mode-of-B has
        // count 1 per A. Support = 4 modes × 1 / 16 records = 0.25.
        assert!((r.support - 0.25).abs() < 1e-9);
    }
}
