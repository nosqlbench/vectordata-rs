// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Categorical association: builds a contingency table over paired
//! observations, then reports χ², Cramér's V, and mutual
//! information (in bits).
//!
//! Only instantiated for pairs where both fields' cardinality
//! regime is `Constant` / `Binary` / `LowCard` per §13.7 — the
//! contingency table would be intractable for high-cardinality
//! fields.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::{PairAnalyzer, PairAnalyzerKind, PairReport};
use crate::pipeline::commands::survey::measure::MeasureCtx;
use crate::pipeline::commands::survey::measures::cardinality::canonical_distinct_key;

pub struct CategoricalAssociationAnalyzer {
    /// Contingency table: a_value → (b_value → count).
    table: IndexMap<String, IndexMap<String, u64>>,
    /// Row totals (per A value).
    row_totals: IndexMap<String, u64>,
    /// Column totals (per B value).
    col_totals: IndexMap<String, u64>,
    /// Grand total.
    n: u64,
    /// Safety valve: stop adding new combinations after this many
    /// distinct cells.
    cell_cap: usize,
    overflowed: bool,
}

impl CategoricalAssociationAnalyzer {
    pub fn new() -> Self {
        Self::with_cap(10_000)
    }
    pub fn with_cap(cell_cap: usize) -> Self {
        CategoricalAssociationAnalyzer {
            table: IndexMap::new(),
            row_totals: IndexMap::new(),
            col_totals: IndexMap::new(),
            n: 0,
            cell_cap,
            overflowed: false,
        }
    }

    fn cell_count(&self) -> usize {
        self.table.values().map(|row| row.len()).sum()
    }
}

impl Default for CategoricalAssociationAnalyzer {
    fn default() -> Self { Self::new() }
}

impl PairAnalyzer for CategoricalAssociationAnalyzer {
    fn observe_pair(&mut self, a: &MValue, b: &MValue, _ctx: &MeasureCtx) {
        let ka = canonical_distinct_key(a);
        let kb = canonical_distinct_key(b);
        self.n += 1;
        *self.row_totals.entry(ka.clone()).or_insert(0) += 1;
        *self.col_totals.entry(kb.clone()).or_insert(0) += 1;
        // Cell count must be sampled before borrowing the row mutably;
        // borrow checker can't see that `self.table.entry` only
        // touches the row corresponding to `ka`.
        let current_cells = self.cell_count();
        let row = self.table.entry(ka).or_default();
        if let Some(c) = row.get_mut(&kb) {
            *c += 1;
        } else if current_cells < self.cell_cap {
            row.insert(kb, 1);
        } else {
            self.overflowed = true;
        }
    }

    fn finalize(self: Box<Self>) -> PairReport {
        let n = self.n;
        if n == 0 {
            return PairReport::CategoricalAssociation(CategoricalAssociationReport {
                n: 0,
                rows: 0,
                cols: 0,
                chi_squared: 0.0,
                cramers_v: 0.0,
                mutual_information_bits: 0.0,
                overflowed: false,
            });
        }
        let nf = n as f64;
        let mut chi2 = 0f64;
        let mut mi = 0f64;
        // χ² must visit every (a, b) cell in the cartesian product of
        // row and column dimensions — including cells with observed
        // count zero, since the (0 - expected)² / expected term is
        // non-negligible for sparse tables. Mutual information only
        // needs the non-zero observed cells (p_ab · log(...) is 0
        // when p_ab is 0).
        for (a_key, &row_total_u) in &self.row_totals {
            let row_total = row_total_u as f64;
            if row_total == 0.0 { continue; }
            for (b_key, &col_total_u) in &self.col_totals {
                let col_total = col_total_u as f64;
                if col_total == 0.0 { continue; }
                let expected = row_total * col_total / nf;
                let observed = self
                    .table
                    .get(a_key)
                    .and_then(|row| row.get(b_key))
                    .copied()
                    .unwrap_or(0) as f64;
                let diff = observed - expected;
                if expected > 0.0 {
                    chi2 += diff * diff / expected;
                }
                let p_ab = observed / nf;
                if p_ab > 0.0 {
                    let p_a = row_total / nf;
                    let p_b = col_total / nf;
                    mi += p_ab * (p_ab / (p_a * p_b)).log2();
                }
            }
        }
        let rows = self.row_totals.len();
        let cols = self.col_totals.len();
        let cramers_v = if rows <= 1 || cols <= 1 {
            0.0
        } else {
            let k = rows.min(cols).saturating_sub(1) as f64;
            (chi2 / (nf * k)).sqrt()
        };
        PairReport::CategoricalAssociation(CategoricalAssociationReport {
            n,
            rows: rows as u32,
            cols: cols as u32,
            chi_squared: chi2,
            cramers_v,
            mutual_information_bits: mi.max(0.0),
            overflowed: self.overflowed,
        })
    }

    fn kind(&self) -> PairAnalyzerKind {
        PairAnalyzerKind::CategoricalAssociation
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalAssociationReport {
    pub n: u64,
    pub rows: u32,
    pub cols: u32,
    pub chi_squared: f64,
    /// Cramér's V in `[0, 1]`. 0 = independent, 1 = perfectly
    /// associated.
    pub cramers_v: f64,
    /// Mutual information in bits.
    pub mutual_information_bits: f64,
    /// True if the contingency table exceeded its cell cap.
    pub overflowed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    /// Perfect functional dependency: A=x ↔ B=fa(x). Cramér's V
    /// approaches 1; MI approaches log2(K) where K is the row count.
    #[test]
    fn perfect_association() {
        let mut a = CategoricalAssociationAnalyzer::new();
        // 4 categories on both sides; 1:1 mapping.
        for i in 0..400 {
            let cat = i % 4;
            a.observe_pair(
                &MValue::Text(format!("a-{}", cat)),
                &MValue::Text(format!("b-{}", cat)),
                &ctx(),
            );
        }
        let r = match Box::new(a).finalize() {
            PairReport::CategoricalAssociation(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!((r.cramers_v - 1.0).abs() < 1e-6, "V = {}", r.cramers_v);
        // I(A;B) for a 4-category perfect map: log2(4) = 2 bits.
        assert!((r.mutual_information_bits - 2.0).abs() < 1e-9);
    }

    /// Independent fields: V near 0, MI near 0.
    #[test]
    fn independent_fields() {
        let mut a = CategoricalAssociationAnalyzer::new();
        // 4 A-categories × 4 B-categories, each combination
        // observed equally often → fully independent table.
        for ai in 0..4 {
            for bi in 0..4 {
                for _ in 0..50 {
                    a.observe_pair(
                        &MValue::Int(ai),
                        &MValue::Int(bi),
                        &ctx(),
                    );
                }
            }
        }
        let r = match Box::new(a).finalize() {
            PairReport::CategoricalAssociation(r) => r,
            _ => panic!("wrong kind"),
        };
        assert!(r.cramers_v < 0.02, "V should be near 0, got {}", r.cramers_v);
        assert!(r.mutual_information_bits < 0.01, "MI should be near 0, got {}", r.mutual_information_bits);
    }

    /// Empty stream is handled cleanly.
    #[test]
    fn empty_stream() {
        let r = match Box::new(CategoricalAssociationAnalyzer::new()).finalize() {
            PairReport::CategoricalAssociation(r) => r,
            _ => panic!("wrong kind"),
        };
        assert_eq!(r.n, 0);
        assert_eq!(r.cramers_v, 0.0);
        assert_eq!(r.mutual_information_bits, 0.0);
    }

    /// Single-category-on-one-side fields: V = 0 (no association
    /// possible when there's no variation).
    #[test]
    fn one_side_constant() {
        let mut a = CategoricalAssociationAnalyzer::new();
        for i in 0..100 {
            a.observe_pair(
                &MValue::Text("same".into()),
                &MValue::Int(i % 4),
                &ctx(),
            );
        }
        let r = match Box::new(a).finalize() {
            PairReport::CategoricalAssociation(r) => r,
            _ => panic!("wrong kind"),
        };
        assert_eq!(r.cramers_v, 0.0);
    }
}
