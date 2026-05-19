// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cross-field pair analyzers.
//!
//! Each analyzer consumes observations from **two** named fields
//! and finalizes into a [`PairReport`]. The orchestrator selects
//! which analyzers to instantiate for each candidate pair based on
//! the two fields' `SemanticType` verdicts and a budgeted
//! eligibility ranking (`max_pair_analyses`, sysref §13.7).
//!
//! The trait is intentionally small — `observe_pair` and
//! `observe_missing` give the analyzer the per-record signal it
//! needs without forcing every implementation through the same
//! aggregation logic.
//!
//! Submodules implement specific analyzer families. Each family's
//! results are folded into a section of the survey report's
//! `cross_field` block per §13.8.

pub mod categorical_assoc;
pub mod copresence;
pub mod functional_dep;
pub mod lowcard_numeric;
pub mod numeric_corr;
pub mod trend;

pub use categorical_assoc::{CategoricalAssociationAnalyzer, CategoricalAssociationReport};
pub use copresence::{CopresenceAnalyzer, CopresenceReport};
pub use functional_dep::{FunctionalDependencyAnalyzer, FunctionalDependencyReport};
pub use lowcard_numeric::{LowCardNumericAnalyzer, LowCardNumericReport};
pub use numeric_corr::{NumericCorrelationAnalyzer, NumericCorrelationReport};
pub use trend::{TrendAnalyzer, TrendReport};

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::measure::MeasureCtx;
use super::template::FieldTemplate;
use super::types::{CardinalityRegime, SemanticType};
#[cfg(test)]
use super::types::NumberKind;

/// Identifier for a concrete pair-analyzer implementation. The
/// `as_str` form is used as the stable key in the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PairAnalyzerKind {
    /// Pearson r over numeric × numeric.
    NumericCorrelation,
    /// χ², Cramér's V, mutual information over categorical × categorical.
    CategoricalAssociation,
    /// One-way ANOVA η² over numeric grouped by categorical.
    LowCardNumeric,
    /// Pearson r between value and record index (trend).
    Trend,
    /// Co-presence rates (Jaccard / conditional).
    Copresence,
    /// Approximate functional dependency probe.
    FunctionalDependency,
}

impl PairAnalyzerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PairAnalyzerKind::NumericCorrelation => "NumericCorrelation",
            PairAnalyzerKind::CategoricalAssociation => "CategoricalAssociation",
            PairAnalyzerKind::LowCardNumeric => "LowCardNumeric",
            PairAnalyzerKind::Trend => "Trend",
            PairAnalyzerKind::Copresence => "Copresence",
            PairAnalyzerKind::FunctionalDependency => "FunctionalDependency",
        }
    }
}

/// Cross-field observation surface.
///
/// `observe_pair` fires when both fields are present in the same
/// record. `observe_missing` fires whenever the record was
/// dispatched and is the analyzer's signal for "this row exists";
/// presence flags `a_present` / `b_present` carry the per-field
/// presence. Co-presence and functional-dependency analyzers rely
/// on this for unbiased counting.
pub trait PairAnalyzer: Send {
    fn observe_pair(&mut self, a: &MValue, b: &MValue, ctx: &MeasureCtx);
    fn observe_missing(&mut self, _a_present: bool, _b_present: bool, _ctx: &MeasureCtx) {}
    fn finalize(self: Box<Self>) -> PairReport;
    fn kind(&self) -> PairAnalyzerKind;
}

/// Output of a finalized [`PairAnalyzer`]. Serialized per-pair
/// inside the survey's `cross_field` block (§13.8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PairReport {
    NumericCorrelation(NumericCorrelationReport),
    CategoricalAssociation(CategoricalAssociationReport),
    LowCardNumeric(LowCardNumericReport),
    Trend(TrendReport),
    Copresence(CopresenceReport),
    FunctionalDependency(FunctionalDependencyReport),
}

impl PairReport {
    pub fn kind(&self) -> PairAnalyzerKind {
        match self {
            PairReport::NumericCorrelation(_) => PairAnalyzerKind::NumericCorrelation,
            PairReport::CategoricalAssociation(_) => PairAnalyzerKind::CategoricalAssociation,
            PairReport::LowCardNumeric(_) => PairAnalyzerKind::LowCardNumeric,
            PairReport::Trend(_) => PairAnalyzerKind::Trend,
            PairReport::Copresence(_) => PairAnalyzerKind::Copresence,
            PairReport::FunctionalDependency(_) => PairAnalyzerKind::FunctionalDependency,
        }
    }
}

// ---------------------------------------------------------------------------
// Survey-report-shaped per-family entries
// ---------------------------------------------------------------------------
//
// Each entry pairs the field names with the analyzer's report so
// the survey's `cross_field` block can serialize as flat lists per
// family, as documented in §13.8.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericCorrelationEntry {
    pub a: String,
    pub b: String,
    #[serde(flatten)]
    pub data: NumericCorrelationReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalAssociationEntry {
    pub a: String,
    pub b: String,
    #[serde(flatten)]
    pub data: CategoricalAssociationReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopresenceEntry {
    pub a: String,
    pub b: String,
    #[serde(flatten)]
    pub data: CopresenceReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LowCardNumericEntry {
    pub a: String,
    pub b: String,
    #[serde(flatten)]
    pub data: LowCardNumericReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrendEntry {
    pub field: String,
    #[serde(flatten)]
    pub data: TrendReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionalDependencyEntry {
    pub lhs: String,
    pub rhs: String,
    #[serde(flatten)]
    pub data: FunctionalDependencyReport,
}

// ---------------------------------------------------------------------------
// Eligibility planning
// ---------------------------------------------------------------------------

/// One scheduled pair analyzer: the field names, the analyzer
/// kind, and the eligibility rank that earned its spot in the
/// budget.
#[derive(Debug, Clone)]
pub struct PairPlanEntry {
    pub a: String,
    pub b: String,
    pub kind: PairAnalyzerKind,
    pub rank: f64,
}

/// Decide which analyzers to run for which pairs, capped at
/// `max_analyses` total. Pairs are ranked by
/// `min(uniqueness(A), uniqueness(B)) × copresence_estimate(A,B)`;
/// the planner defers to higher-uniqueness pairs first because they
/// carry more discriminative power.
///
/// Co-presence is **always** scheduled for every eligible pair (and
/// is the cheapest analyzer); other analyzers gate on the
/// (SemanticType, regime) of both fields per §13.7.
pub fn plan_pair_analyzers(
    templates: &indexmap::IndexMap<String, FieldTemplate>,
    max_analyses: usize,
) -> Vec<PairPlanEntry> {
    let field_names: Vec<&String> = templates.keys().collect();
    let mut entries: Vec<PairPlanEntry> = Vec::new();

    for i in 0..field_names.len() {
        for j in (i + 1)..field_names.len() {
            let a = field_names[i];
            let b = field_names[j];
            let ta = templates.get(a).unwrap();
            let tb = templates.get(b).unwrap();
            // Unstable fields are excluded entirely (§13.7 table
            // row).
            if matches!(ta.semantic_type, Some(SemanticType::Unstable)) {
                continue;
            }
            if matches!(tb.semantic_type, Some(SemanticType::Unstable)) {
                continue;
            }
            if ta.semantic_type.is_none() || tb.semantic_type.is_none() {
                continue;
            }
            let uniq_a = uniqueness_score(&ta.cardinality_regime);
            let uniq_b = uniqueness_score(&tb.cardinality_regime);
            // Copresence-only score for ranking; we cannot observe
            // true co-presence at plan time, so we proxy with
            // present_fraction × present_fraction.
            let rank = uniq_a.min(uniq_b);
            // Always queue Copresence — it's free.
            entries.push(PairPlanEntry {
                a: a.clone(),
                b: b.clone(),
                kind: PairAnalyzerKind::Copresence,
                rank,
            });
            // Numeric × numeric → Pearson r.
            if is_numeric(ta) && is_numeric(tb) {
                entries.push(PairPlanEntry {
                    a: a.clone(),
                    b: b.clone(),
                    kind: PairAnalyzerKind::NumericCorrelation,
                    rank,
                });
            }
            // LowCard × LowCard → categorical association.
            if is_low_card(ta) && is_low_card(tb) {
                entries.push(PairPlanEntry {
                    a: a.clone(),
                    b: b.clone(),
                    kind: PairAnalyzerKind::CategoricalAssociation,
                    rank,
                });
            }
            // LowCard × Numeric → LowCardNumeric η².
            if (is_low_card(ta) && is_numeric(tb)) || (is_numeric(ta) && is_low_card(tb)) {
                entries.push(PairPlanEntry {
                    a: a.clone(),
                    b: b.clone(),
                    kind: PairAnalyzerKind::LowCardNumeric,
                    rank,
                });
            }
            // LowCard × Any → approximate functional dependency.
            // Schedule both directions so an asymmetric FD (e.g.
            // country → currency holds, but currency → country
            // doesn't) surfaces in the report.
            if is_low_card(ta) {
                entries.push(PairPlanEntry {
                    a: a.clone(),
                    b: b.clone(),
                    kind: PairAnalyzerKind::FunctionalDependency,
                    rank,
                });
            }
            if is_low_card(tb) && (a.as_str() != b.as_str()) {
                entries.push(PairPlanEntry {
                    a: b.clone(),
                    b: a.clone(),
                    kind: PairAnalyzerKind::FunctionalDependency,
                    rank,
                });
            }
        }
    }

    // Trend is unary-shaped (numeric vs record index) but still
    // crosses two "axes" — we treat the second axis as a sentinel
    // field name "__record_index__".
    for (name, t) in templates.iter() {
        if matches!(t.semantic_type, Some(SemanticType::Unstable))
            || t.semantic_type.is_none()
            || !is_numeric(t)
        {
            continue;
        }
        entries.push(PairPlanEntry {
            a: name.clone(),
            b: "__record_index__".into(),
            kind: PairAnalyzerKind::Trend,
            rank: uniqueness_score(&t.cardinality_regime),
        });
    }

    // Stable sort: keep Copresence first, then higher-rank entries
    // ahead of lower-rank ones.
    entries.sort_by(|x, y| {
        // Prefer Copresence — it's cheap and survives any prune.
        let cx = matches!(x.kind, PairAnalyzerKind::Copresence);
        let cy = matches!(y.kind, PairAnalyzerKind::Copresence);
        cy.cmp(&cx).then_with(|| {
            y.rank.partial_cmp(&x.rank).unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    entries.truncate(max_analyses);
    entries
}

/// Score in `[0, 1]` indicating how distinctive a field is.
fn uniqueness_score(regime: &CardinalityRegime) -> f64 {
    match regime {
        CardinalityRegime::Constant => 0.0,
        CardinalityRegime::Binary => 0.1,
        CardinalityRegime::LowCard { exact_distinct } => {
            (*exact_distinct as f64).ln_1p() / 8.0
        }
        CardinalityRegime::MidCard { hll_estimate_at_pass1 } => {
            (hll_estimate_at_pass1.ln_1p() / 16.0).clamp(0.0, 0.9)
        }
        CardinalityRegime::HighCardOrUnique { uniqueness_ratio } => {
            uniqueness_ratio.clamp(0.5, 1.0)
        }
        CardinalityRegime::Unknown => 0.0,
    }
}

fn is_numeric(t: &FieldTemplate) -> bool {
    matches!(
        t.semantic_type,
        Some(SemanticType::Number(_)) | Some(SemanticType::Boolean)
    )
}

fn is_low_card(t: &FieldTemplate) -> bool {
    matches!(
        t.cardinality_regime,
        CardinalityRegime::Constant
            | CardinalityRegime::Binary
            | CardinalityRegime::LowCard { .. }
    ) && t.semantic_type.is_some()
        && !matches!(t.semantic_type, Some(SemanticType::Unstable))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::commands::survey::types::{NumericWidth, WireEncoding, WireEncodingKind};
    use crate::pipeline::commands::survey::measure::MeasureKind;

    fn numeric_template() -> FieldTemplate {
        FieldTemplate {
            wire_encoding: WireEncoding {
                kind: WireEncodingKind::Numeric,
                storage_width: Some(NumericWidth::I32),
                narrowest_width: Some(NumericWidth::I8),
                tag_histogram: indexmap::IndexMap::new(),
                mixed: false,
            },
            semantic_type: Some(SemanticType::Number(NumberKind::Integer {
                signed: false,
                bit_width_hint: NumericWidth::I8,
            })),
            semantic_confidence: 1.0,
            cardinality_regime: CardinalityRegime::MidCard { hll_estimate_at_pass1: 1000.0 },
            measures: vec![MeasureKind::ExactMoments],
            probe_tallies: Vec::new(),
        }
    }

    fn low_card_template() -> FieldTemplate {
        FieldTemplate {
            wire_encoding: WireEncoding {
                kind: WireEncodingKind::Textual,
                storage_width: None,
                narrowest_width: None,
                tag_histogram: indexmap::IndexMap::new(),
                mixed: false,
            },
            semantic_type: Some(SemanticType::Categorical(super::super::types::CategoricalKind::Enum)),
            semantic_confidence: 1.0,
            cardinality_regime: CardinalityRegime::LowCard { exact_distinct: 5 },
            measures: vec![],
            probe_tallies: Vec::new(),
        }
    }

    fn unstable_template() -> FieldTemplate {
        FieldTemplate {
            wire_encoding: WireEncoding {
                kind: WireEncodingKind::Null,
                storage_width: None,
                narrowest_width: None,
                tag_histogram: indexmap::IndexMap::new(),
                mixed: true,
            },
            semantic_type: Some(SemanticType::Unstable),
            semantic_confidence: 0.0,
            cardinality_regime: CardinalityRegime::Unknown,
            measures: vec![],
            probe_tallies: Vec::new(),
        }
    }

    #[test]
    fn plan_numeric_pair_includes_pearson_and_copresence() {
        let mut t = indexmap::IndexMap::new();
        t.insert("a".to_string(), numeric_template());
        t.insert("b".to_string(), numeric_template());
        let plan = plan_pair_analyzers(&t, 32);
        // Expect Copresence + NumericCorrelation + Trend(×2).
        let kinds: Vec<PairAnalyzerKind> = plan.iter().map(|p| p.kind).collect();
        assert!(kinds.contains(&PairAnalyzerKind::Copresence));
        assert!(kinds.contains(&PairAnalyzerKind::NumericCorrelation));
        // Two Trend entries (one per numeric field × record_index).
        assert_eq!(kinds.iter().filter(|k| **k == PairAnalyzerKind::Trend).count(), 2);
    }

    #[test]
    fn plan_lowcard_pair_includes_categorical_assoc_and_fd() {
        let mut t = indexmap::IndexMap::new();
        t.insert("country".to_string(), low_card_template());
        t.insert("currency".to_string(), low_card_template());
        let plan = plan_pair_analyzers(&t, 32);
        let kinds: Vec<PairAnalyzerKind> = plan.iter().map(|p| p.kind).collect();
        assert!(kinds.contains(&PairAnalyzerKind::CategoricalAssociation));
        assert!(kinds.contains(&PairAnalyzerKind::FunctionalDependency));
    }

    #[test]
    fn plan_excludes_unstable_fields() {
        let mut t = indexmap::IndexMap::new();
        t.insert("good".to_string(), numeric_template());
        t.insert("bad".to_string(), unstable_template());
        let plan = plan_pair_analyzers(&t, 32);
        // Only Trend(good) should be in the plan — every pair
        // involving `bad` is excluded.
        for p in &plan {
            assert_ne!(p.a, "bad");
            assert_ne!(p.b, "bad");
        }
    }

    #[test]
    fn plan_caps_at_max_analyses() {
        let mut t = indexmap::IndexMap::new();
        for i in 0..20 {
            t.insert(format!("f{}", i), numeric_template());
        }
        let plan = plan_pair_analyzers(&t, 16);
        assert_eq!(plan.len(), 16);
    }

    #[test]
    fn plan_lowcard_numeric_cross() {
        let mut t = indexmap::IndexMap::new();
        t.insert("group".to_string(), low_card_template());
        t.insert("value".to_string(), numeric_template());
        let plan = plan_pair_analyzers(&t, 32);
        let kinds: Vec<PairAnalyzerKind> = plan.iter().map(|p| p.kind).collect();
        assert!(kinds.contains(&PairAnalyzerKind::LowCardNumeric));
    }
}
