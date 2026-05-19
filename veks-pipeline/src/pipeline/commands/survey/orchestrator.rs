// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Two-pass survey orchestrator.
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
//! 5. Finalize every measure and assemble the [`SurveyReport`].
//!
//! Cross-field analyzers and findings rendering land in subsequent
//! build-plan steps; the orchestrator emits the structured JSON
//! contract from §13.8 with `cross_field` empty.

use std::collections::HashSet;
use std::path::Path;
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
use super::template::{ExplorationProbe, FieldTemplate, TemplateConfig};
use super::types::{CardinalityRegime, SemanticType, WireEncoding};

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
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub path: String,
    pub format: String,
    pub total_records: u64,
    pub sampled_records: u64,
    pub sampling: SamplingInfo,
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
            match (kind, &report) {
                (MeasureKind::Presence, MeasureReport::Presence(p)) => {
                    presence = Some(p.clone());
                }
                _ => {}
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

    // ── Finalize cross-field analyzers ──────────────────────────────
    let cross_field = collect_pair_reports(pair_analyzers, planned_pairs);

    Ok(SurveyReport {
        schema_version: 1,
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
        },
        fields,
        cross_field,
        warnings,
    })
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
    let mut out = CrossFieldReport::default();
    out.planned = planned;
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
        assert_eq!(r.schema_version, 1);
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
