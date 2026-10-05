// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Survey orchestrator: two sampled passes and an optional exhaustive
//! census.
//!
//! Drives the full survey pipeline end-to-end:
//!
//! 1. Open the slab and resolve sampling parameters.
//! 2. **Pass 1** — iterate sampled MNode records; per field, advance
//!    an [`ExplorationProbe`].
//! 3. Synthesize a [`FieldTemplate`] per field from the probes.
//! 4. **Pass 2** — re-iterate the same sample; per field,
//!    instantiate the measure suite chosen by its template and
//!    dispatch each observation.
//! 5. Finalize every measure and assemble the per-field profiles.
//! 6. **Pass 3** — when anything is declared, the census: one
//!    exhaustive pass counting declared fields, hierarchies and pairs
//!    exactly over every record (`census.rs`). Its results replace the
//!    sampled cardinality verdicts of the fields it covers.
//! 7. Finalize cross-field analyzers and assemble the [`SurveyReport`].

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::ui::UiHandle;

use super::crossfield::{
    plan_pair_analyzers, CategoricalAssociationAnalyzer, CategoricalAssociationEntry,
    CopresenceAnalyzer, CopresenceEntry, FunctionalDependencyAnalyzer,
    FunctionalDependencyEntry, LowCardNumericAnalyzer, LowCardNumericEntry,
    NumericCorrelationAnalyzer, NumericCorrelationEntry, PairAnalyzer, PairAnalyzerKind,
    PairPlanEntry, PairReport, TrendAnalyzer, TrendEntry,
};
use super::census::{
    run_census_pass, CensusConfig, CensusFieldPlan, CensusInfo, CensusPlan, DroppedField,
    FieldCensusResult, HierarchyCensusReport, PairCensusReport,
};
use super::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport, PresenceReport,
};
use super::measures::{
    BitWidthMeasure, ByteEntropyMeasure, ByteOrCharLengthRangeMeasure, CharClassMix,
    DiscreteIndicatorMeasure, EpochPlausibilityMeasure, ExactExtrema, ExactFrequencyTable,
    ExactLengthMoments, ExactMoments, HeavyHittersMeasure, HistogramFromQuantilesMeasure,
    HyperLogLogMeasure, LabelsetHeavyHittersMeasure, LengthQuantiles, MonotonicityMeasure,
    PatternSkeletonMeasure, PresenceMeasure, ProbeAttemptMeasure, QuantileSketchMeasure,
    ReservoirSample, TemporalRangeMeasure, TrigramHeavyHittersMeasure, TypeStabilityMeasure,
    WireEncodingHistogramMeasure, DEFAULT_LABELSET_TOP_K, DEFAULT_TRIGRAM_TOP_K,
};
use super::progress::{ProgressDriver, SurveyProgress};
use super::template::{ExplorationProbe, FieldTemplate, TemplateConfig};
use super::types::{CardinalityRegime, SemanticType, WireEncoding, WireEncodingKind};

use crate::pipeline::commands::slab::{open_slab_with_ui, sample_page_indices};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Operator-facing configuration. Mirrors the CLI flags / YAML keys
/// documented in sysref §13.9.
#[derive(Debug, Clone)]
pub struct SurveyConfig {
    /// Maximum records sampled across the two passes (the same sample
    /// is reused for Pass 1 and Pass 2).
    pub samples: usize,
    /// Capacity for the bounded distinct tracker during Pass 1.
    pub distinct_cap: u32,
    /// Cardinality ≤ this → `LowCard` regime + exact frequency.
    pub low_card_threshold: u32,
    /// Cardinality between low/mid thresholds → `MidCard` regime.
    pub mid_card_threshold: u32,
    /// Per-field reservoir size.
    pub reservoir_size: usize,
    /// Reservoir seed. Reproducible surveys want a stable seed.
    pub reservoir_seed: u64,
    /// HLL precision (register count = 2^p).
    pub hll_precision: u8,
    /// Misra-Gries top-K capacity.
    pub top_k: usize,
    /// KLL quantile sketch parameter (rank error ≈ 1.0 / k).
    pub quantile_k: usize,
    /// Maximum number of cross-field pair analyzers to schedule.
    pub max_pair_analyses: usize,
    /// Semantic-probe match threshold (sysref §13.3.3). A probe must
    /// match at this rate or higher across the field's reservoir to
    /// commit its verdict; otherwise the encoding-only floor wins.
    pub semantic_confidence: f64,
    /// Pass 3 census declarations (sysref §13.4, Pass 3).
    pub census: CensusConfig,
}

impl Default for SurveyConfig {
    fn default() -> Self {
        SurveyConfig {
            samples: 100_000,
            distinct_cap: 4_096,
            low_card_threshold: 64,
            mid_card_threshold: 4_096,
            reservoir_size: 1_024,
            reservoir_seed: 0xC011EC70,
            hll_precision: 12,
            top_k: 64,
            quantile_k: 1000,
            max_pair_analyses: 1_024,
            semantic_confidence: 0.95,
            census: CensusConfig::default(),
        }
    }
}

impl SurveyConfig {
    fn template_config(&self) -> TemplateConfig {
        TemplateConfig {
            low_card_threshold: self.low_card_threshold,
            mid_card_threshold: self.mid_card_threshold,
            distinct_cap: self.distinct_cap,
            reservoir_size: self.reservoir_size,
            reservoir_seed: self.reservoir_seed,
            semantic_confidence: self.semantic_confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Top-level survey report — serialized to `survey.json` per §13.8.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurveyReport {
    pub schema_version: u32,
    pub produced_by: String,
    pub source: SourceInfo,
    pub fields: IndexMap<String, FieldProfile>,
    pub cross_field: CrossFieldReport,
    /// Verified, counted trees for each declared hierarchy (Pass 3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hierarchies: Vec<HierarchyCensusReport>,
    /// Exact joint tables for each declared pair (Pass 3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pair_census: Vec<PairCensusReport>,
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub path: String,
    pub format: String,
    pub total_records: u64,
    pub sampled_records: u64,
    pub sampling: SamplingInfo,
    /// Present when the census pass ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub census: Option<CensusInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingInfo {
    pub mode: String,
    pub page_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldProfile {
    pub wire_encoding: WireEncoding,
    pub semantic_type: Option<SemanticType>,
    pub semantic_confidence: f64,
    pub cardinality_regime: CardinalityRegime,
    pub presence: PresenceReport,
    /// True when the census pass counted this field: `presence` and
    /// the cardinality regime are then exact over every record, and
    /// the sampled cardinality measures have been replaced by the
    /// census tables.
    #[serde(default)]
    pub censused: bool,
    /// Open map keyed by [`MeasureKind::as_str`].
    ///
    /// `#[serde(untagged)]` on [`MeasureReport`] is correct for
    /// serialization (the outer map key carries the
    /// discriminator), but ambiguous on the way back in — reports
    /// with overlapping field shapes (e.g. `ExactExtremaReport` is
    /// a subset of `QuantileSketchReport`) would be mis-typed by
    /// serde's first-fit variant matching. The custom
    /// `deserialize_with` routes each value to the right variant
    /// using the map key.
    #[serde(deserialize_with = "deserialize_measures_by_key")]
    pub measures: IndexMap<String, MeasureReport>,
}

/// Deserialize the per-field measures map by routing each value
/// through the report type the outer key names. Keeps the JSON
/// wire format unchanged (a flat map of `MeasureKind → report`)
/// while avoiding the variant-ambiguity bug that
/// `#[serde(untagged)]` introduces when multiple reports share
/// field shapes.
fn deserialize_measures_by_key<'de, D>(
    d: D,
) -> Result<IndexMap<String, MeasureReport>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    use super::measure::{MeasureKind, MeasureReport};
    use super::measures::cardinality::{
        ExactFrequencyTableReport, HeavyHittersReport, HyperLogLogReport,
    };
    use super::measures::labelset::LabelsetHeavyHittersReport;
    use super::measures::numeric::{
        BitWidthReport, DiscreteIndicatorReport, HistogramFromQuantilesReport,
        MonotonicityReport, QuantileSketchReport,
    };
    use super::measures::opaque::{
        ByteOrCharLengthRangeReport, ProbeAttemptReport, WireEncodingHistogramReport,
    };
    use super::measures::temporal::{EpochPlausibilityReport, TemporalRangeReport};
    use super::measures::textual::{
        CharClassMixReport, ExactLengthMomentsReport, LengthQuantilesReport,
        PatternSkeletonReport,
    };
    use super::measures::trigram::TrigramHeavyHittersReport;
    use super::measure::{
        ExactExtremaReport, ExactMomentsReport, PresenceReport,
        ReservoirSampleReport, TypeStabilityReport,
    };
    use super::measures::bytes::ByteEntropyReport;

    let raw: IndexMap<String, serde_json::Value> = IndexMap::deserialize(d)?;
    let mut out: IndexMap<String, MeasureReport> = IndexMap::with_capacity(raw.len());
    for (key, value) in raw {
        let report = match key.as_str() {
            s if s == MeasureKind::Presence.as_str() => {
                MeasureReport::Presence(serde_json::from_value::<PresenceReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::TypeStability.as_str() => {
                MeasureReport::TypeStability(serde_json::from_value::<TypeStabilityReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ReservoirSample.as_str() => {
                MeasureReport::ReservoirSample(serde_json::from_value::<ReservoirSampleReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ExactMoments.as_str() => {
                MeasureReport::ExactMoments(serde_json::from_value::<ExactMomentsReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ExactExtrema.as_str() => {
                MeasureReport::ExactExtrema(serde_json::from_value::<ExactExtremaReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::QuantileSketch.as_str() => {
                MeasureReport::QuantileSketch(serde_json::from_value::<QuantileSketchReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::BitWidth.as_str() => {
                MeasureReport::BitWidth(serde_json::from_value::<BitWidthReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::HistogramFromQuantiles.as_str() => {
                MeasureReport::HistogramFromQuantiles(serde_json::from_value::<HistogramFromQuantilesReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::Monotonicity.as_str() => {
                MeasureReport::Monotonicity(serde_json::from_value::<MonotonicityReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::DiscreteIndicator.as_str() => {
                MeasureReport::DiscreteIndicator(serde_json::from_value::<DiscreteIndicatorReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::HyperLogLog.as_str() => {
                MeasureReport::HyperLogLog(serde_json::from_value::<HyperLogLogReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::HeavyHitters.as_str() => {
                MeasureReport::HeavyHitters(serde_json::from_value::<HeavyHittersReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ExactFrequencyTable.as_str() => {
                MeasureReport::ExactFrequencyTable(serde_json::from_value::<ExactFrequencyTableReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ExactLengthMoments.as_str() => {
                MeasureReport::ExactLengthMoments(serde_json::from_value::<ExactLengthMomentsReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::LengthQuantiles.as_str() => {
                MeasureReport::LengthQuantiles(serde_json::from_value::<LengthQuantilesReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::CharClassMix.as_str() => {
                MeasureReport::CharClassMix(serde_json::from_value::<CharClassMixReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::PatternSkeleton.as_str() => {
                MeasureReport::PatternSkeleton(serde_json::from_value::<PatternSkeletonReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::TrigramHeavyHitters.as_str() => {
                MeasureReport::TrigramHeavyHitters(serde_json::from_value::<TrigramHeavyHittersReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::LabelsetHeavyHitters.as_str() => {
                MeasureReport::LabelsetHeavyHitters(serde_json::from_value::<LabelsetHeavyHittersReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::TemporalRange.as_str() => {
                MeasureReport::TemporalRange(serde_json::from_value::<TemporalRangeReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::EpochPlausibility.as_str() => {
                MeasureReport::EpochPlausibility(serde_json::from_value::<EpochPlausibilityReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ByteEntropy.as_str() => {
                MeasureReport::ByteEntropy(serde_json::from_value::<ByteEntropyReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::WireEncodingHistogram.as_str() => {
                MeasureReport::WireEncodingHistogram(serde_json::from_value::<WireEncodingHistogramReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ByteOrCharLengthRange.as_str() => {
                MeasureReport::ByteOrCharLengthRange(serde_json::from_value::<ByteOrCharLengthRangeReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ProbeAttempt.as_str() => {
                MeasureReport::ProbeAttempt(serde_json::from_value::<ProbeAttemptReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ExactValueCensus.as_str() => {
                MeasureReport::ExactValueCensus(serde_json::from_value::<super::census::ExactValueCensusReport>(value).map_err(D::Error::custom)?)
            }
            s if s == MeasureKind::ExactIntegerHistogram.as_str() => {
                MeasureReport::ExactIntegerHistogram(serde_json::from_value::<super::census::ExactIntegerHistogramReport>(value).map_err(D::Error::custom)?)
            }
            _ => continue, // unknown kind — silently drop so older readers can ignore new measures
        };
        out.insert(key, report);
    }
    Ok(out)
}

/// Cross-field analysis results (§13.7).
///
/// Each family is reported as a list of per-pair entries — flat,
/// flexible, and trivial to slice in downstream tooling. Empty
/// families serialize as absent fields rather than as empty arrays
/// so a sparse report stays compact.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossFieldReport {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub numeric_correlation: Vec<super::crossfield::NumericCorrelationEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub categorical_association: Vec<super::crossfield::CategoricalAssociationEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub copresence: Vec<super::crossfield::CopresenceEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lowcard_numeric: Vec<super::crossfield::LowCardNumericEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trend: Vec<super::crossfield::TrendEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub functional_dependencies: Vec<super::crossfield::FunctionalDependencyEntry>,
    /// Number of pair-analyses scheduled by Pass 1.
    #[serde(default)]
    pub planned: u32,
    /// Number that ran (less than `planned` if some pairs had zero
    /// joint observations).
    #[serde(default)]
    pub executed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Warning {
    pub severity: String,
    pub field: Option<String>,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Run a survey end-to-end against a slab file and return the
/// structured report. The caller is responsible for serializing
/// the report to disk.
///
/// `ui` is optional. When `None`, the driver runs headless (no
/// progress bars, no log lines). Tests pass `None`; the
/// `SurveyOp` command-op pulls it from the pipeline context.
pub fn survey(
    path: &Path,
    config: &SurveyConfig,
    ui: Option<&UiHandle>,
) -> Result<SurveyReport, String> {
    let started = Instant::now();
    let reader = open_slab_with_ui(path, ui).map_err(|e| {
        format!("failed to open {}: {}", path.display(), e)
    })?;

    let page_entries = reader.page_entries();
    let total_pages = page_entries.len();
    let total_records = reader.total_records() as usize;

    // Determine the page sample set (deterministic stride; same as
    // the legacy survey's sampling).
    let desired_pages = if total_records == 0 {
        0
    } else {
        let avg_per_page = total_records as f64 / total_pages.max(1) as f64;
        ((config.samples as f64 / avg_per_page).ceil() as usize)
            .max(1)
            .min(total_pages)
    };
    let sample_pages = sample_page_indices(total_pages, desired_pages);

    if let Some(u) = ui {
        u.log(&format!(
            "survey: sampling {} of {} pages ({} total records, target {} samples)",
            sample_pages.len(), total_pages, total_records, config.samples,
        ));
    }

    // ── Pass 1 ──────────────────────────────────────────────────────
    let mut probes: IndexMap<String, ExplorationProbe> = IndexMap::new();
    let mut record_field_buf: HashSet<String> = HashSet::new();
    let mut sampled = 0usize;
    let mut non_mnode_count = 0u64;
    let mut decode_errors = 0u64;
    let tcfg = config.template_config();

    'pass1: for &page_idx in &sample_pages {
        let entry = &page_entries[page_idx];
        let page = match reader.read_data_page(entry) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for i in 0..page.record_count() {
            if sampled >= config.samples {
                break 'pass1;
            }
            let bytes = match page.get_record(i) {
                Some(b) => b,
                None => continue,
            };
            match anode::decode(bytes) {
                Ok(ANode::MNode(mnode)) => {
                    record_field_buf.clear();
                    for (name, value) in &mnode.fields {
                        record_field_buf.insert(name.clone());
                        let probe = probes
                            .entry(name.clone())
                            .or_insert_with(|| ExplorationProbe::new(tcfg.clone()));
                        probe.observe(value);
                    }
                    sampled += 1;
                }
                Ok(ANode::PNode(_)) => non_mnode_count += 1,
                Err(_) => decode_errors += 1,
            }
        }
    }

    if let Some(u) = ui {
        u.log(&format!(
            "survey: Pass 1 complete in {:.1}s — {} records, {} fields, {} non-MNode, {} decode errors",
            started.elapsed().as_secs_f64(), sampled, probes.len(), non_mnode_count, decode_errors,
        ));
    }

    // ── Template synthesis ──────────────────────────────────────────
    let templates: IndexMap<String, FieldTemplate> = probes
        .into_iter()
        .map(|(name, probe)| {
            let t = probe.finalize(&name);
            (name, t)
        })
        .collect();

    let unstable_count = templates
        .values()
        .filter(|t| matches!(t.semantic_type, Some(SemanticType::Unstable)))
        .count();

    if let Some(u) = ui {
        u.log(&format!(
            "survey: classified {} fields ({} Unstable). Pass 2 starting.",
            templates.len(), unstable_count,
        ));
    }

    // ── Pass 2 ──────────────────────────────────────────────────────
    // Instantiate the measure suite per field from its template.
    let expected_tags_by_field: IndexMap<String, Vec<&'static str>> = templates
        .iter()
        .map(|(name, t)| (name.clone(), expected_tags_from(t)))
        .collect();

    let mut field_measures: IndexMap<String, Vec<Box<dyn Measure>>> = IndexMap::new();
    for (name, template) in &templates {
        let mut bundle: Vec<Box<dyn Measure>> = Vec::with_capacity(template.measures.len());
        for kind in &template.measures {
            if let Some(m) = instantiate_measure(*kind, expected_tags_by_field.get(name).map(|v| v.as_slice()).unwrap_or(&[]), config, template) {
                bundle.push(m);
            }
        }
        field_measures.insert(name.clone(), bundle);
    }

    // ── Plan cross-field analyzers ──────────────────────────────────
    let pair_plan = plan_pair_analyzers(&templates, config.max_pair_analyses);
    let planned_pairs = pair_plan.len() as u32;
    let mut pair_analyzers: Vec<(PairPlanEntry, Box<dyn PairAnalyzer>)> = pair_plan
        .into_iter()
        .filter_map(|p| {
            let a = instantiate_pair_analyzer(&p, &templates)?;
            Some((p, a))
        })
        .collect();
    if let Some(u) = ui {
        u.log(&format!(
            "survey: planned {} pair analyzers ({} after gating)",
            planned_pairs, pair_analyzers.len()
        ));
    }

    let mut pass2_sampled = 0usize;
    'pass2: for &page_idx in &sample_pages {
        let entry = &page_entries[page_idx];
        let page = match reader.read_data_page(entry) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for i in 0..page.record_count() {
            if pass2_sampled >= config.samples {
                break 'pass2;
            }
            let bytes = match page.get_record(i) {
                Some(b) => b,
                None => continue,
            };
            let mnode = match anode::decode(bytes) {
                Ok(ANode::MNode(m)) => m,
                Ok(ANode::PNode(_)) => continue,
                Err(_) => continue,
            };
            dispatch_record(&mnode, pass2_sampled as u64, &templates, &mut field_measures);
            dispatch_pairs(&mnode, pass2_sampled as u64, &templates, &mut pair_analyzers);
            pass2_sampled += 1;
        }
    }

    if let Some(u) = ui {
        u.log(&format!(
            "survey: Pass 2 complete in {:.1}s — {} records dispatched",
            started.elapsed().as_secs_f64(), pass2_sampled,
        ));
    }

    // ── Finalize ────────────────────────────────────────────────────
    let mut fields = IndexMap::new();
    for (name, template) in templates {
        let measures = field_measures.swap_remove(&name).unwrap_or_default();
        let mut measure_reports: IndexMap<String, MeasureReport> = IndexMap::new();
        let mut presence: Option<PresenceReport> = None;
        for m in measures {
            let kind = m.kind();
            let report = m.finalize();
            if let (MeasureKind::Presence, MeasureReport::Presence(p)) = (kind, &report) {
                presence = Some(p.clone());
            }
            measure_reports.insert(kind.as_str().to_string(), report);
        }
        // Presence is hoisted to the top of the field profile per §13.8.
        // Remove the duplicate from the measures map; the JSON is
        // operator-readable enough without the same data twice.
        measure_reports.swap_remove(MeasureKind::Presence.as_str());
        let presence = presence.unwrap_or(PresenceReport {
            present: 0,
            null_count: 0,
            absent_in_record: 0,
        });
        fields.insert(
            name,
            FieldProfile {
                wire_encoding: template.wire_encoding,
                semantic_type: template.semantic_type,
                semantic_confidence: template.semantic_confidence,
                cardinality_regime: template.cardinality_regime,
                presence,
                censused: false,
                measures: measure_reports,
            },
        );
    }

    let mut warnings = Vec::new();
    if decode_errors > 0 {
        warnings.push(Warning {
            severity: "warning".into(),
            field: None,
            message: format!(
                "{} records failed ANode decode and were skipped", decode_errors
            ),
        });
    }
    if non_mnode_count > 0 {
        warnings.push(Warning {
            severity: "info".into(),
            field: None,
            message: format!(
                "{} non-MNode records (e.g. PNode) were skipped — survey only profiles MNodes",
                non_mnode_count,
            ),
        });
    }

    // ── Pass 3: census ──────────────────────────────────────────────
    // Exhaustive and exact over every record, for what was declared.
    // Runs after the profiles exist because `auto` selects fields by
    // their Pass 1 regime, and merges into them because the census
    // supersedes every sampled cardinality verdict it covers.
    let (census_info, hierarchies, pair_census) = if config.census.is_noop() {
        (None, Vec::new(), Vec::new())
    } else {
        let plan = plan_census(&config.census, &fields)?;
        if let Some(u) = ui {
            u.log(&format!(
                "survey: Pass 3 (census) over all {} records — {} fields, {} hierarchies, {} pairs",
                total_records, plan.fields.len(), plan.hierarchies.len(), plan.pairs.len(),
            ));
        }
        let mut driver = ProgressDriver::new(
            Arc::new(SurveyProgress::new()),
            ui.cloned(),
            CENSUS_LOG_EVERY_PAGES,
        );
        let outcome = run_census_pass(
            &reader,
            &page_entries,
            total_records as u64,
            &plan,
            config.census.threads,
            &mut driver,
        )?;
        let mut kept = Vec::new();
        let mut dropped = Vec::new();
        for result in outcome.fields {
            let FieldCensusResult {
                name, present, nulls, absent, value, histogram, dropped: reasons, ..
            } = result;
            let Some(profile) = fields.get_mut(&name) else { continue };
            let Some(value) = value else {
                // One entry per field, whatever combination of its
                // accumulators overflowed.
                dropped.push(DroppedField { field: name.clone(), reason: reasons.join("; ") });
                continue;
            };
            // The census supersedes every sampled cardinality measure
            // it covers; leaving them beside it would offer a reader
            // two answers, one of them an estimate.
            profile.measures.swap_remove(MeasureKind::ExactFrequencyTable.as_str());
            profile.measures.swap_remove(MeasureKind::HeavyHitters.as_str());
            profile.measures.swap_remove(MeasureKind::HyperLogLog.as_str());
            profile.cardinality_regime =
                CardinalityRegime::Censused { exact_distinct: value.distinct };
            profile.presence = PresenceReport {
                present: present + nulls,
                null_count: nulls,
                absent_in_record: absent,
            };
            profile.censused = true;
            profile.measures.insert(
                MeasureKind::ExactValueCensus.as_str().to_string(),
                MeasureReport::ExactValueCensus(value),
            );
            if let Some(h) = histogram {
                profile.measures.insert(
                    MeasureKind::ExactIntegerHistogram.as_str().to_string(),
                    MeasureReport::ExactIntegerHistogram(h),
                );
            }
            for reason in reasons {
                warnings.push(Warning {
                    severity: "warning".into(),
                    field: Some(name.clone()),
                    message: format!("census: {}", reason),
                });
            }
            kept.push(name);
        }
        for d in &dropped {
            warnings.push(Warning {
                severity: "warning".into(),
                field: Some(d.field.clone()),
                message: format!("census: field left the census — {}", d.reason),
            });
        }
        if outcome.decode_errors > 0 {
            warnings.push(Warning {
                severity: "warning".into(),
                field: None,
                message: format!(
                    "census: {} records failed ANode decode and were skipped",
                    outcome.decode_errors
                ),
            });
        }
        if let Some(u) = ui {
            u.log(&format!(
                "survey: Pass 3 complete in {:.1}s — {} records, {} fields censused, {} dropped",
                started.elapsed().as_secs_f64(), outcome.records, kept.len(), dropped.len(),
            ));
        }
        (
            Some(CensusInfo {
                records: outcome.records,
                auto: config.census.auto,
                fields: kept,
                dropped,
            }),
            outcome.hierarchies,
            outcome.pairs,
        )
    };

    // ── Finalize cross-field analyzers ──────────────────────────────
    let cross_field = collect_pair_reports(pair_analyzers, planned_pairs);

    Ok(SurveyReport {
        schema_version: 2,
        produced_by: "veks-pipeline analyze survey".into(),
        source: SourceInfo {
            path: path.display().to_string(),
            format: "slab".into(),
            total_records: total_records as u64,
            sampled_records: sampled as u64,
            sampling: SamplingInfo {
                mode: "page_stride".into(),
                page_count: sample_pages.len() as u64,
            },
            census: census_info,
        },
        fields,
        cross_field,
        hierarchies,
        pair_census,
        warnings,
    })
}

// ---------------------------------------------------------------------------
// Pass-3 planning
// ---------------------------------------------------------------------------

/// Pages between census milestone log lines. Slab pages of a metadata
/// facet hold a few hundred records each, so this is a line every few
/// million records — a few dozen lines over half a billion.
const CENSUS_LOG_EVERY_PAGES: u32 = 16_384;

/// Resolve the census declarations against the surveyed fields.
///
/// `auto` takes every field whose Pass 1 regime already shows it to be
/// enumerable; a listed field is taken regardless of regime and must
/// exist. Hierarchy and pair fields must exist too: a declaration
/// naming a field the records do not carry is a configuration error,
/// not an empty table.
fn plan_census(
    config: &CensusConfig,
    fields: &IndexMap<String, FieldProfile>,
) -> Result<CensusPlan, String> {
    let mut planned: Vec<CensusFieldPlan> = Vec::new();
    if config.auto {
        for (name, profile) in fields {
            let enumerable = matches!(
                profile.cardinality_regime,
                CardinalityRegime::Constant
                    | CardinalityRegime::Binary
                    | CardinalityRegime::LowCard { .. }
                    | CardinalityRegime::MidCard { .. }
            );
            let typed = profile
                .semantic_type
                .as_ref()
                .is_some_and(|t| !matches!(t, SemanticType::Unstable));
            if enumerable && typed {
                planned.push(CensusFieldPlan {
                    name: name.clone(),
                    listed: false,
                    integer: is_integer_encoded(profile),
                });
            }
        }
    }
    for name in &config.listed {
        let profile = fields.get(name).ok_or_else(|| {
            format!("census: listed field `{}` is not present in the surveyed records", name)
        })?;
        if let Some(existing) = planned.iter_mut().find(|f| &f.name == name) {
            existing.listed = true;
            continue;
        }
        planned.push(CensusFieldPlan {
            name: name.clone(),
            listed: true,
            integer: is_integer_encoded(profile),
        });
    }
    for levels in &config.hierarchies {
        for field in levels {
            if !fields.contains_key(field) {
                return Err(format!(
                    "hierarchy {}: field `{}` is not present in the surveyed records",
                    levels.join(">"),
                    field
                ));
            }
        }
    }
    for (a, b) in &config.pairs {
        for field in [a, b] {
            if !fields.contains_key(field) {
                return Err(format!(
                    "census-pair {}:{}: field `{}` is not present in the surveyed records",
                    a, b, field
                ));
            }
        }
    }
    Ok(CensusPlan {
        fields: planned,
        hierarchies: config.hierarchies.clone(),
        pairs: config.pairs.clone(),
        cap: config.cap,
        pair_cells_cap: config.pair_cells_cap,
    })
}

/// A field every one of whose Pass 1 tags is an integer variant gets a
/// dense histogram beside its value table.
fn is_integer_encoded(profile: &FieldProfile) -> bool {
    matches!(profile.wire_encoding.kind, WireEncodingKind::Numeric)
        && !profile.wire_encoding.tag_histogram.is_empty()
        && profile
            .wire_encoding
            .tag_histogram
            .keys()
            .all(|tag| matches!(tag.as_str(), "Int" | "Int32" | "Short" | "EnumOrd"))
}

// ---------------------------------------------------------------------------
// Pass-2 dispatch
// ---------------------------------------------------------------------------

fn dispatch_record(
    mnode: &MNode,
    record_index: u64,
    templates: &IndexMap<String, FieldTemplate>,
    field_measures: &mut IndexMap<String, Vec<Box<dyn Measure>>>,
) {
    // Walk each registered field. Fields present in the record get
    // `observe`; fields not present get `observe_missing`.
    let mut present: HashSet<&str> = HashSet::with_capacity(mnode.fields.len());
    for (name, _) in &mnode.fields {
        present.insert(name.as_str());
    }
    for (name, _template) in templates {
        let ctx = MeasureCtx {
            record_index,
            semantic_type: templates.get(name).and_then(|t| t.semantic_type.as_ref()),
        };
        let measures = match field_measures.get_mut(name) {
            Some(m) => m,
            None => continue,
        };
        if let Some(value) = mnode.fields.get(name.as_str()) {
            for m in measures.iter_mut() {
                m.observe(value, &ctx);
            }
            let _ = present.remove(name.as_str());
        } else {
            for m in measures.iter_mut() {
                m.observe_missing(&ctx);
            }
        }
    }
    // Any field present in the record but absent from templates is a
    // Pass 2 surprise — log via TypeStabilityMeasure on a sentinel?
    // For step 5 we drop these silently; step 7's surprise reporting
    // owns this concern.
    drop(present);
}

// ---------------------------------------------------------------------------
// Measure-suite instantiation
// ---------------------------------------------------------------------------

fn expected_tags_from(template: &FieldTemplate) -> Vec<&'static str> {
    template
        .wire_encoding
        .tag_histogram
        .keys()
        .filter_map(|k| static_tag_name(k))
        .collect()
}

/// Map a runtime tag string (from `{:?}` on `MValue::tag()`) to its
/// `&'static str` form for the type-stability measure. Returns
/// `None` for unrecognized tags — TypeStabilityMeasure will then
/// treat the tag as a surprise even when it was seen in Pass 1,
/// which is conservative.
fn static_tag_name(s: &str) -> Option<&'static str> {
    // The Debug form is the bare variant name (no surrounding
    // syntax) when the enum variant has no payload. Match the
    // common cases by direct equality with the static strings.
    static TAGS: &[&str] = &[
        "Text", "Int", "Float", "Bool", "Bytes", "Null", "EnumStr",
        "EnumOrd", "List", "Map", "Ascii", "Int32", "Short", "Float32",
        "Half", "Millis", "Nanos", "Date", "Time", "DateTime",
        "UuidV1", "UuidV7", "Ulid", "Array", "Set", "TypedMap",
    ];
    TAGS.iter().copied().find(|t| *t == s)
}

// ---------------------------------------------------------------------------
// Cross-field dispatch helpers
// ---------------------------------------------------------------------------

const RECORD_INDEX_SENTINEL: &str = "__record_index__";

fn instantiate_pair_analyzer(
    plan: &PairPlanEntry,
    templates: &IndexMap<String, FieldTemplate>,
) -> Option<Box<dyn PairAnalyzer>> {
    match plan.kind {
        PairAnalyzerKind::Copresence => Some(Box::new(CopresenceAnalyzer::new())),
        PairAnalyzerKind::NumericCorrelation => Some(Box::new(NumericCorrelationAnalyzer::new())),
        PairAnalyzerKind::CategoricalAssociation => {
            Some(Box::new(CategoricalAssociationAnalyzer::new()))
        }
        PairAnalyzerKind::LowCardNumeric => {
            // Decide which side carries the categorical based on the
            // templates' cardinality regimes. Falls back to category-on-a
            // when both are LowCard.
            let a_t = templates.get(&plan.a)?;
            let category_on_a = is_low_cardinality(a_t);
            Some(Box::new(LowCardNumericAnalyzer::new(category_on_a)))
        }
        PairAnalyzerKind::Trend => Some(Box::new(TrendAnalyzer::new())),
        PairAnalyzerKind::FunctionalDependency => {
            Some(Box::new(FunctionalDependencyAnalyzer::new()))
        }
    }
}

fn is_low_cardinality(t: &FieldTemplate) -> bool {
    matches!(
        t.cardinality_regime,
        super::types::CardinalityRegime::Constant
            | super::types::CardinalityRegime::Binary
            | super::types::CardinalityRegime::LowCard { .. }
    )
}

fn dispatch_pairs(
    mnode: &MNode,
    record_index: u64,
    templates: &IndexMap<String, FieldTemplate>,
    pair_analyzers: &mut [(PairPlanEntry, Box<dyn PairAnalyzer>)],
) {
    let ctx = MeasureCtx { record_index, semantic_type: None };
    for (plan, analyzer) in pair_analyzers.iter_mut() {
        // Trend is unary-shaped: B side is the synthetic record-index field.
        if plan.b == RECORD_INDEX_SENTINEL {
            if let Some(a_value) = mnode.fields.get(plan.a.as_str()) {
                let ctx = MeasureCtx {
                    record_index,
                    semantic_type: templates.get(&plan.a).and_then(|t| t.semantic_type.as_ref()),
                };
                analyzer.observe_pair(a_value, &MValue::Null, &ctx);
            }
            continue;
        }
        let a_value = mnode.fields.get(plan.a.as_str());
        let b_value = mnode.fields.get(plan.b.as_str());
        match (a_value, b_value) {
            (Some(av), Some(bv)) => analyzer.observe_pair(av, bv, &ctx),
            (a, b) => analyzer.observe_missing(a.is_some(), b.is_some(), &ctx),
        }
    }
}

fn collect_pair_reports(
    pair_analyzers: Vec<(PairPlanEntry, Box<dyn PairAnalyzer>)>,
    planned: u32,
) -> CrossFieldReport {
    let mut out = CrossFieldReport { planned, ..Default::default() };
    for (plan, analyzer) in pair_analyzers {
        let report = analyzer.finalize();
        out.executed += 1;
        match report {
            PairReport::Copresence(r) => out.copresence.push(CopresenceEntry {
                a: plan.a, b: plan.b, data: r,
            }),
            PairReport::NumericCorrelation(r) => out.numeric_correlation.push(NumericCorrelationEntry {
                a: plan.a, b: plan.b, data: r,
            }),
            PairReport::CategoricalAssociation(r) => out.categorical_association.push(CategoricalAssociationEntry {
                a: plan.a, b: plan.b, data: r,
            }),
            PairReport::LowCardNumeric(r) => out.lowcard_numeric.push(LowCardNumericEntry {
                a: plan.a, b: plan.b, data: r,
            }),
            PairReport::Trend(r) => out.trend.push(TrendEntry {
                field: plan.a, data: r,
            }),
            PairReport::FunctionalDependency(r) => out.functional_dependencies.push(FunctionalDependencyEntry {
                lhs: plan.a, rhs: plan.b, data: r,
            }),
        }
    }
    out
}

fn instantiate_measure(
    kind: MeasureKind,
    expected_tags: &[&'static str],
    config: &SurveyConfig,
    template: &FieldTemplate,
) -> Option<Box<dyn Measure>> {
    match kind {
        MeasureKind::Presence => Some(Box::new(PresenceMeasure::new())),
        MeasureKind::TypeStability => {
            Some(Box::new(TypeStabilityMeasure::new(expected_tags.iter().copied())))
        }
        MeasureKind::ReservoirSample => Some(Box::new(ReservoirSample::new(
            config.reservoir_size,
            config.reservoir_seed,
        ))),
        MeasureKind::ExactMoments => Some(Box::new(ExactMoments::new())),
        MeasureKind::ExactExtrema => Some(Box::new(ExactExtrema::new())),
        MeasureKind::HyperLogLog => Some(Box::new(HyperLogLogMeasure::new(config.hll_precision))),
        MeasureKind::HeavyHitters => Some(Box::new(HeavyHittersMeasure::new(config.top_k))),
        MeasureKind::ExactFrequencyTable => {
            Some(Box::new(ExactFrequencyTable::new(config.low_card_threshold as usize)))
        }
        MeasureKind::ExactLengthMoments => Some(Box::new(ExactLengthMoments::new())),
        MeasureKind::LengthQuantiles => {
            Some(Box::new(LengthQuantiles::new(config.quantile_k, config.reservoir_seed)))
        }
        MeasureKind::CharClassMix => Some(Box::new(CharClassMix::new())),
        MeasureKind::PatternSkeleton => Some(Box::new(PatternSkeletonMeasure::new(config.top_k))),
        MeasureKind::TrigramHeavyHitters => {
            Some(Box::new(TrigramHeavyHittersMeasure::new(DEFAULT_TRIGRAM_TOP_K)))
        }
        MeasureKind::LabelsetHeavyHitters => {
            Some(Box::new(LabelsetHeavyHittersMeasure::new(DEFAULT_LABELSET_TOP_K)))
        }
        MeasureKind::QuantileSketch => {
            Some(Box::new(QuantileSketchMeasure::new(config.quantile_k, config.reservoir_seed)))
        }
        MeasureKind::BitWidth => Some(Box::new(BitWidthMeasure::new())),
        MeasureKind::HistogramFromQuantiles => {
            Some(Box::new(HistogramFromQuantilesMeasure::new(32)))
        }
        MeasureKind::Monotonicity => Some(Box::new(MonotonicityMeasure::new(1_000))),
        MeasureKind::DiscreteIndicator => Some(Box::new(DiscreteIndicatorMeasure::new())),
        MeasureKind::TemporalRange => Some(Box::new(TemporalRangeMeasure::new())),
        MeasureKind::EpochPlausibility => Some(Box::new(EpochPlausibilityMeasure::new())),
        MeasureKind::ByteEntropy => Some(Box::new(ByteEntropyMeasure::new())),
        MeasureKind::WireEncodingHistogram => Some(Box::new(WireEncodingHistogramMeasure::new())),
        MeasureKind::ByteOrCharLengthRange => Some(Box::new(ByteOrCharLengthRangeMeasure::new())),
        MeasureKind::ProbeAttempt => Some(Box::new(ProbeAttemptMeasure::from_tallies(
            template.probe_tallies.clone(),
        ))),
        // Census measures are produced by Pass 3 (`census.rs`), never
        // instantiated as Pass 2 observers.
        MeasureKind::ExactValueCensus | MeasureKind::ExactIntegerHistogram => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use slabtastic::{SlabWriter, WriterConfig};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use veks_core::formats::anode;
    use veks_core::formats::mnode::{MNode, MValue};

    fn tmp_path(name: &str) -> PathBuf {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        // Leak the tempdir so the path survives — the survey driver
        // re-opens the file and we don't want the dir to drop early.
        std::mem::forget(dir);
        path
    }

    fn write_slab(path: &Path, records: &[HashMap<&str, MValue>]) {
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = SlabWriter::new(path, config).unwrap();
        for rec in records {
            let mut fields = indexmap::IndexMap::new();
            for (k, v) in rec {
                fields.insert((*k).to_string(), v.clone());
            }
            let mnode = MNode { fields };
            let bytes = anode::encode(&anode::ANode::MNode(mnode));
            w.add_record(&bytes).unwrap();
        }
        w.finish().unwrap();
    }

    fn census_config(auto: bool, listed: &[&str], cap: usize) -> SurveyConfig {
        SurveyConfig {
            census: CensusConfig {
                auto,
                listed: listed.iter().map(|s| s.to_string()).collect(),
                cap,
                ..CensusConfig::default()
            },
            ..SurveyConfig::default()
        }
    }

    /// `auto` censuses a low-cardinality field exactly: the sampled
    /// frequency table is replaced, the regime becomes `Censused`,
    /// and presence covers every record.
    #[test]
    fn census_auto_counts_low_card_field_exactly() {
        let path = tmp_path("census_auto.slab");
        let countries = ["US", "GB", "DE", "FR", "JP"];
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..300 {
            let mut r = HashMap::new();
            r.insert("country", MValue::Text(countries[i % 5].into()));
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let f = r.fields.get("country").unwrap();
        assert!(f.censused);
        assert_eq!(f.cardinality_regime, CardinalityRegime::Censused { exact_distinct: 5 });
        assert_eq!(f.presence.present, 300);
        assert!(!f.measures.contains_key("ExactFrequencyTable"));
        match f.measures.get("ExactValueCensus") {
            Some(MeasureReport::ExactValueCensus(c)) => {
                assert_eq!(c.population, 300);
                assert_eq!(c.missing, 0);
                assert_eq!(c.distinct, 5);
                assert!(c.counts.values().all(|n| *n == 60), "{:?}", c.counts);
            }
            other => panic!("expected ExactValueCensus, got {:?}", other),
        }
        let info = r.source.census.as_ref().expect("census info");
        assert_eq!(info.records, 300);
        assert!(info.auto);
        assert_eq!(info.fields, vec!["country".to_string()]);
        assert!(info.dropped.is_empty());
        assert_eq!(r.schema_version, 2);
    }

    /// `census: none` is exactly the two-pass survey.
    #[test]
    fn census_none_leaves_report_unchanged() {
        let path = tmp_path("census_none.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..50 {
            let mut r = HashMap::new();
            r.insert("k", MValue::Text(format!("v{}", i % 3)));
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &census_config(false, &[], 10), None).unwrap();
        let f = r.fields.get("k").unwrap();
        assert!(!f.censused);
        assert!(f.measures.contains_key("ExactFrequencyTable"));
        assert!(matches!(f.cardinality_regime, CardinalityRegime::LowCard { .. }));
        assert!(r.source.census.is_none());
        assert!(r.hierarchies.is_empty());
        assert!(r.pair_census.is_empty());
    }

    /// The census counts every record even when the sampled passes
    /// saw a handful of them.
    #[test]
    fn census_counts_everything_when_sample_is_small() {
        let path = tmp_path("census_small_sample.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..500 {
            let mut r = HashMap::new();
            r.insert(
                "k",
                MValue::Text(if i % 10 == 0 { "rare".into() } else { "common".into() }),
            );
            records.push(r);
        }
        write_slab(&path, &records);
        let cfg = SurveyConfig { samples: 10, ..SurveyConfig::default() };
        let r = survey(&path, &cfg, None).unwrap();
        assert_eq!(r.source.sampled_records, 10);
        let f = r.fields.get("k").unwrap();
        match f.measures.get("ExactValueCensus") {
            Some(MeasureReport::ExactValueCensus(c)) => {
                assert_eq!(c.population, 500);
                assert_eq!(c.counts.get("Text(\"rare\")"), Some(&50));
                assert_eq!(c.counts.get("Text(\"common\")"), Some(&450));
            }
            other => panic!("expected ExactValueCensus, got {:?}", other),
        }
        assert_eq!(f.presence.present, 500);
    }

    /// A listed field over the cap is an error: a truncated table
    /// presented as exact would be a wrong selectivity.
    #[test]
    fn census_listed_field_over_cap_errors() {
        let path = tmp_path("census_listed_cap.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..100 {
            let mut r = HashMap::new();
            r.insert("uid", MValue::Text(format!("u{}", i)));
            records.push(r);
        }
        write_slab(&path, &records);
        let err = survey(&path, &census_config(false, &["uid"], 10), None).unwrap_err();
        assert!(err.contains("uid") && err.contains("census-cap"), "{}", err);
    }

    /// An `auto` field over the cap leaves the census with a warning
    /// and keeps its sampled measures.
    #[test]
    fn census_auto_field_over_cap_is_dropped_with_warning() {
        let path = tmp_path("census_auto_cap.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..100 {
            let mut r = HashMap::new();
            r.insert("uid", MValue::Text(format!("u{}", i)));
            r.insert("k", MValue::Text(format!("v{}", i % 3)));
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &census_config(true, &[], 10), None).unwrap();
        let uid = r.fields.get("uid").unwrap();
        assert!(!uid.censused);
        assert!(!uid.measures.contains_key("ExactValueCensus"));
        assert!(
            uid.measures.contains_key("HeavyHitters") || uid.measures.contains_key("HyperLogLog"),
            "sampled measures stay: {:?}",
            uid.measures.keys().collect::<Vec<_>>()
        );
        let info = r.source.census.as_ref().unwrap();
        assert_eq!(info.fields, vec!["k".to_string()]);
        assert_eq!(info.dropped.len(), 1);
        assert_eq!(info.dropped[0].field, "uid");
        assert!(r.warnings.iter().any(|w| w.field.as_deref() == Some("uid") && w.message.contains("census")));
    }

    /// Integer fields get a dense histogram beside the value table.
    #[test]
    fn census_integer_field_gets_dense_histogram() {
        let path = tmp_path("census_int.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..120 {
            let mut r = HashMap::new();
            r.insert("year", MValue::Int32(2000 + (i % 12)));
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let f = r.fields.get("year").unwrap();
        assert!(f.censused);
        match f.measures.get("ExactIntegerHistogram") {
            Some(MeasureReport::ExactIntegerHistogram(h)) => {
                assert_eq!((h.min, h.max), (2000, 2011));
                assert_eq!(h.counts.len(), 12);
                assert!(h.counts.iter().all(|n| *n == 10));
                assert_eq!(h.population, 120);
            }
            other => panic!("expected ExactIntegerHistogram, got {:?}", other),
        }
    }

    /// Hierarchy and pair declarations produce a verified tree and a
    /// dense joint table whose margins agree with the field census,
    /// and both survive a JSON round trip.
    #[test]
    fn census_hierarchy_and_pair_end_to_end() {
        let path = tmp_path("census_hier_pair.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..240 {
            let l1 = if i % 3 == 0 { "a" } else { "b" };
            let l2 = format!("{}{}", l1, i % 2);
            let mut r = HashMap::new();
            r.insert("l1", MValue::Text(l1.into()));
            r.insert("l2", MValue::Text(l2));
            r.insert("year", MValue::Int(2000 + (i % 4) as i64));
            records.push(r);
        }
        write_slab(&path, &records);
        let cfg = SurveyConfig {
            census: CensusConfig {
                hierarchies: vec![vec!["l1".into(), "l2".into()]],
                pairs: vec![("l2".into(), "year".into())],
                ..CensusConfig::default()
            },
            ..SurveyConfig::default()
        };
        let r = survey(&path, &cfg, None).unwrap();
        assert_eq!(r.hierarchies.len(), 1);
        let h = &r.hierarchies[0];
        assert_eq!(h.fields, vec!["l1".to_string(), "l2".to_string()]);
        assert_eq!(h.population, 240);
        assert_eq!(h.incomplete, 0);
        assert_eq!(h.level_sizes, vec![2, 4]);
        assert_eq!(h.nodes[0].value, "Text(\"b\")");
        assert_eq!(h.nodes[0].count, 160);
        assert_eq!(h.nodes[0].children.iter().map(|c| c.count).sum::<u64>(), 160);
        assert_eq!(h.nodes[1].count, 80);
        // Every node's count equals the field census for that value.
        let l2 = match r.fields.get("l2").unwrap().measures.get("ExactValueCensus") {
            Some(MeasureReport::ExactValueCensus(c)) => c.counts.clone(),
            other => panic!("{:?}", other),
        };
        for node in &h.nodes {
            for child in &node.children {
                assert_eq!(l2.get(&child.value), Some(&child.count), "{}", child.value);
            }
        }
        assert_eq!(r.pair_census.len(), 1);
        let p = &r.pair_census[0];
        assert_eq!((p.a.as_str(), p.b.as_str()), ("l2", "year"));
        assert_eq!(p.population, 240);
        assert_eq!(p.a_values.len(), 4);
        assert_eq!(p.b_values.len(), 4);
        for (i, row) in p.counts.iter().enumerate() {
            assert_eq!(
                row.iter().sum::<u64>(),
                *l2.get(&p.a_values[i]).unwrap(),
                "row {}",
                p.a_values[i]
            );
        }
        let s = serde_json::to_string(&r).unwrap();
        let back: SurveyReport = serde_json::from_str(&s).unwrap();
        assert_eq!(back.hierarchies, r.hierarchies);
        assert_eq!(back.pair_census, r.pair_census);
        let l2_back = back.fields.get("l2").unwrap();
        assert_eq!(l2_back.cardinality_regime, CardinalityRegime::Censused { exact_distinct: 4 });
        assert!(l2_back.censused);
        assert!(l2_back.measures.contains_key("ExactValueCensus"));
    }

    /// A value with two parents fails the survey: a declared
    /// hierarchy is an invariant of the data that produced it.
    #[test]
    fn census_hierarchy_violation_fails_survey() {
        let path = tmp_path("census_hier_bad.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..40 {
            let mut r = HashMap::new();
            r.insert("l1", MValue::Text(if i % 2 == 0 { "a".into() } else { "b".into() }));
            r.insert("l2", MValue::Text("shared".into()));
            records.push(r);
        }
        write_slab(&path, &records);
        let cfg = SurveyConfig {
            census: CensusConfig {
                hierarchies: vec![vec!["l1".into(), "l2".into()]],
                ..CensusConfig::default()
            },
            ..SurveyConfig::default()
        };
        let err = survey(&path, &cfg, None).unwrap_err();
        assert!(err.contains("nesting violated"), "{}", err);
    }

    /// Decoding is parallel and counting is sequential, so the report
    /// is byte-identical whatever the thread count.
    #[test]
    fn census_is_identical_across_thread_counts() {
        let path = tmp_path("census_threads.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..3_000 {
            let l1 = if i % 7 == 0 { "a" } else { "b" };
            let mut r = HashMap::new();
            r.insert("l1", MValue::Text(l1.into()));
            r.insert("l2", MValue::Text(format!("{}{}", l1, i % 5)));
            r.insert("k", MValue::Int(i % 13));
            records.push(r);
        }
        write_slab(&path, &records);
        let run = |threads: usize| {
            let cfg = SurveyConfig {
                census: CensusConfig {
                    hierarchies: vec![vec!["l1".into(), "l2".into()]],
                    pairs: vec![("l2".into(), "k".into())],
                    threads,
                    ..CensusConfig::default()
                },
                ..SurveyConfig::default()
            };
            serde_json::to_string(&survey(&path, &cfg, None).unwrap()).unwrap()
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert_eq!(one, run(0));
        let back: SurveyReport = serde_json::from_str(&one).unwrap();
        assert_eq!(back.source.census.as_ref().unwrap().records, 3_000);
        assert!(back.source.sampling.page_count < 3_000, "fixture spans several pages");
    }

    /// A declaration naming a field the records do not carry is a
    /// configuration error, not an empty table.
    #[test]
    fn census_unknown_field_in_declaration_errors() {
        let path = tmp_path("census_unknown.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..10 {
            let mut r = HashMap::new();
            r.insert("k", MValue::Int(i));
            records.push(r);
        }
        write_slab(&path, &records);
        let err = survey(&path, &census_config(true, &["nope"], 100), None).unwrap_err();
        assert!(err.contains("nope"), "{}", err);
        let cfg = SurveyConfig {
            census: CensusConfig {
                pairs: vec![("k".into(), "nope".into())],
                ..CensusConfig::default()
            },
            ..SurveyConfig::default()
        };
        assert!(survey(&path, &cfg, None).unwrap_err().contains("nope"));
    }

    /// Empty slab survey: produces a report with no fields, zero
    /// records, no warnings.
    #[test]
    fn empty_slab_runs_clean() {
        let path = tmp_path("empty.slab");
        write_slab(&path, &[]);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        assert_eq!(r.source.total_records, 0);
        assert_eq!(r.source.sampled_records, 0);
        assert!(r.fields.is_empty());
        assert!(r.warnings.is_empty());
        assert_eq!(r.schema_version, 2);
    }

    /// A single-field integer column: classified as Number(Integer),
    /// LowCard regime, with ExactMoments and ExactExtrema populated.
    #[test]
    fn integer_field_full_pipeline() {
        let path = tmp_path("ints.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..50 {
            let mut r = HashMap::new();
            r.insert("count", MValue::Int32(i));
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let f = r.fields.get("count").expect("count field present");
        let st = f.semantic_type.as_ref().expect("verdict expected");
        match st {
            SemanticType::Number(_) => {}
            other => panic!("expected Number, got {:?}", other),
        }
        // Presence: 50 present, 0 null, 0 absent.
        assert_eq!(f.presence.present, 50);
        assert_eq!(f.presence.null_count, 0);
        assert_eq!(f.presence.absent_in_record, 0);
        // Measures: ExactMoments and ExactExtrema should be present.
        // (Presence is hoisted to top of profile, NOT in measures map.)
        assert!(f.measures.contains_key("ExactMoments"));
        assert!(f.measures.contains_key("ExactExtrema"));
        assert!(!f.measures.contains_key("Presence"));
        // Min/max correct.
        match f.measures.get("ExactExtrema") {
            Some(MeasureReport::ExactExtrema(e)) => {
                assert_eq!(e.min, Some(0.0));
                assert_eq!(e.max, Some(49.0));
            }
            other => panic!("wrong ExactExtrema report: {:?}", other),
        }
    }

    /// Field present in some records but not others: absent count
    /// matches the missing-record count.
    #[test]
    fn sparse_field_tracks_absence() {
        let path = tmp_path("sparse.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..10 {
            let mut r = HashMap::new();
            r.insert("always", MValue::Int(i));
            if i % 2 == 0 {
                r.insert("sometimes", MValue::Int(i * 10));
            }
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let always = r.fields.get("always").expect("always field");
        let sometimes = r.fields.get("sometimes").expect("sometimes field");
        // "always" was in every record; "sometimes" in half.
        assert_eq!(always.presence.present, 10);
        assert_eq!(always.presence.absent_in_record, 0);
        assert_eq!(sometimes.presence.present, 5);
        assert_eq!(sometimes.presence.absent_in_record, 5);
    }

    /// Mixed wire encodings on one field → SemanticType::Unstable.
    #[test]
    fn mixed_encoding_lands_unstable() {
        let path = tmp_path("mixed.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..10 {
            let mut r = HashMap::new();
            if i % 2 == 0 {
                r.insert("messy", MValue::Int(i));
            } else {
                r.insert("messy", MValue::Text(format!("v{}", i)));
            }
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let f = r.fields.get("messy").unwrap();
        assert_eq!(f.semantic_type, Some(SemanticType::Unstable));
        assert!(f.wire_encoding.mixed);
        // Only the universal opaque-set measures should be present.
        assert!(!f.measures.contains_key("ExactMoments"));
        assert!(!f.measures.contains_key("ExactExtrema"));
    }

    /// Null-only field stays in Unknown.
    #[test]
    fn null_only_field_stays_unknown() {
        let path = tmp_path("nullonly.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for _ in 0..20 {
            let mut r = HashMap::new();
            r.insert("nothing", MValue::Null);
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let f = r.fields.get("nothing").unwrap();
        assert!(f.semantic_type.is_none());
        assert_eq!(f.presence.null_count, 20);
    }

    /// Survey output is JSON-serializable and round-trips back to
    /// an equal `SurveyReport`.
    #[test]
    fn report_json_roundtrip() {
        let path = tmp_path("roundtrip.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..30 {
            let mut r = HashMap::new();
            r.insert("x", MValue::Int(i));
            r.insert("name", MValue::Text(format!("item-{}", i)));
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let s = serde_json::to_string(&r).unwrap();
        let _back: SurveyReport = serde_json::from_str(&s).unwrap();
    }

    /// End-to-end cross-field check: two perfectly-correlated numeric
    /// fields should surface Pearson r ≈ 1.0 in the cross_field block.
    #[test]
    fn cross_field_numeric_correlation_end_to_end() {
        let path = tmp_path("crossfield_numcorr.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        for i in 0..200 {
            let mut r = HashMap::new();
            r.insert("x", MValue::Int(i));
            r.insert("y", MValue::Int(2 * i + 3));
            records.push(r);
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        assert_eq!(r.cross_field.numeric_correlation.len(), 1, "expected one numeric pair");
        let entry = &r.cross_field.numeric_correlation[0];
        assert!((entry.data.pearson_r.unwrap() - 1.0).abs() < 1e-9, "r = {:?}", entry.data.pearson_r);
        // Trend: both x and y are monotone over record index → 2 entries.
        assert_eq!(r.cross_field.trend.len(), 2);
        for trend in &r.cross_field.trend {
            assert!((trend.data.pearson_r_with_index.unwrap() - 1.0).abs() < 1e-9);
        }
    }

    /// End-to-end functional-dependency check: country → currency is
    /// perfect, so the FD report's support should be 1.0.
    #[test]
    fn cross_field_functional_dependency_end_to_end() {
        let path = tmp_path("crossfield_fd.slab");
        let mut records: Vec<HashMap<&str, MValue>> = Vec::new();
        let mapping: &[(&str, &str)] = &[("US","USD"),("GB","GBP"),("DE","EUR"),("FR","EUR")];
        for _ in 0..10 {
            for (cc, cur) in mapping {
                let mut r = HashMap::new();
                r.insert("country", MValue::Text((*cc).into()));
                r.insert("currency", MValue::Text((*cur).into()));
                records.push(r);
            }
        }
        write_slab(&path, &records);
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        // Both fields LowCard ⇒ a CategoricalAssociation entry + a
        // FunctionalDependency entry both fire.
        assert!(!r.cross_field.functional_dependencies.is_empty(), "expected FD entries: {:?}", r.cross_field);
        assert!(!r.cross_field.categorical_association.is_empty(), "expected categorical_assoc entries");
        // At least one FD entry should hit perfect support.
        let max_support = r.cross_field
            .functional_dependencies
            .iter()
            .map(|e| e.data.support)
            .fold(0.0_f64, f64::max);
        assert!((max_support - 1.0).abs() < 1e-9);
    }

    /// Decode errors and PNode records produce warnings.
    #[test]
    fn pnode_record_produces_info_warning() {
        let path = tmp_path("with_pnode.slab");
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = SlabWriter::new(&path, config).unwrap();
        for i in 0..5 {
            let mut fields = indexmap::IndexMap::new();
            fields.insert("x".into(), MValue::Int(i));
            let mnode = MNode { fields };
            w.add_record(&anode::encode(&anode::ANode::MNode(mnode))).unwrap();
        }
        // Add one PNode record.
        use veks_core::formats::pnode::{Comparand, FieldRef, OpType, PNode, PredicateNode};
        let pnode = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("x".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(1)],
        });
        w.add_record(&anode::encode(&anode::ANode::PNode(pnode))).unwrap();
        w.finish().unwrap();
        let r = survey(&path, &SurveyConfig::default(), None).unwrap();
        let has_pnode_warning = r.warnings.iter().any(|w| w.message.contains("non-MNode"));
        assert!(has_pnode_warning, "expected non-MNode warning in {:?}", r.warnings);
    }
}
