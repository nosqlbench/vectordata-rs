// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Operator-readable findings renderer.
//!
//! Consumes a [`SurveyReport`] and emits **both** a Markdown summary
//! and a machine-readable JSON form. The two are produced from a
//! single curated-findings tree so they cannot drift, and they share
//! the same `json_path` pointers back into the structured
//! `survey.json`.
//!
//! See `docs/sysref/13-metadata-survey.md` §13.11.4 for the design
//! and section list.

use serde::{Deserialize, Serialize};

use super::measure::MeasureReport;
use super::orchestrator::{FieldProfile, SurveyReport};
use super::types::{IdentifierKind, NumberKind, SemanticType};

/// Severity tag for a single finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Notable,
    Warning,
    Error,
}

impl Severity {
    fn tag(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Notable => "notable",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

/// One curated finding. Both renderers walk the same list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub section: String,
    pub severity: Severity,
    /// Field this finding is anchored to (`None` for section-level
    /// observations like the overview).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    pub title: String,
    pub body: String,
    /// Up to a few reservoir samples making the finding concrete.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub samples: Vec<serde_json::Value>,
    /// JSON pointer back into `survey.json` (no leading `$.`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_path: Option<String>,
}

/// JSON envelope for `survey.findings.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingsReport {
    pub schema_version: u32,
    pub produced_by: String,
    pub source: FindingsSource,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingsSource {
    pub path: String,
    pub survey_json: String,
}

/// Knobs for the findings renderer.
#[derive(Debug, Clone)]
pub struct FindingsConfig {
    /// Minimum severity to include.
    pub min_severity: Severity,
    /// Top-K cross-field entries per family.
    pub cross_field_top_k: usize,
    /// Maximum samples to embed per finding.
    pub max_samples_per_finding: usize,
    /// Source path identifier for the `survey_json` field (defaults
    /// to `"survey.json"`).
    pub survey_json_filename: String,
    /// `|Pearson r|` and `|Cramér's V|` threshold for surfacing as
    /// a `notable` cross-field highlight.
    pub correlation_threshold: f64,
    /// Null-rate threshold for surfacing a quality warning.
    pub null_rate_warning: f64,
}

impl Default for FindingsConfig {
    fn default() -> Self {
        FindingsConfig {
            min_severity: Severity::Info,
            cross_field_top_k: 5,
            max_samples_per_finding: 5,
            survey_json_filename: "survey.json".into(),
            correlation_threshold: 0.5,
            null_rate_warning: 0.1,
        }
    }
}

/// Render both output forms. Returns `(markdown, json_report)`.
pub fn render_findings(
    report: &SurveyReport,
    config: &FindingsConfig,
) -> (String, FindingsReport) {
    let raw = collect_findings(report, config);
    let filtered: Vec<Finding> = raw
        .into_iter()
        .filter(|f| f.severity >= config.min_severity)
        .collect();
    let md = render_markdown(report, &filtered);
    let json = FindingsReport {
        schema_version: 1,
        produced_by: "veks-pipeline analyze survey (findings)".into(),
        source: FindingsSource {
            path: report.source.path.clone(),
            survey_json: config.survey_json_filename.clone(),
        },
        findings: filtered,
    };
    (md, json)
}

// ---------------------------------------------------------------------------
// Curation
// ---------------------------------------------------------------------------

fn collect_findings(report: &SurveyReport, cfg: &FindingsConfig) -> Vec<Finding> {
    let mut out = Vec::new();
    overview_findings(report, &mut out);
    schema_findings(report, &mut out);
    unstable_findings(report, cfg, &mut out);
    partition_candidate_findings(report, cfg, &mut out);
    predicate_candidate_findings(report, &mut out);
    identifier_findings(report, &mut out);
    cross_field_findings(report, cfg, &mut out);
    quality_findings(report, cfg, &mut out);
    out
}

fn overview_findings(report: &SurveyReport, out: &mut Vec<Finding>) {
    out.push(Finding {
        section: "Overview".into(),
        severity: Severity::Info,
        field: None,
        title: format!(
            "Surveyed {} of {} records across {} fields",
            report.source.sampled_records,
            report.source.total_records,
            report.fields.len(),
        ),
        body: format!(
            "Sampling: {} ({} pages). Cross-field analyzers: {} planned, {} executed.",
            report.source.sampling.mode,
            report.source.sampling.page_count,
            report.cross_field.planned,
            report.cross_field.executed,
        ),
        samples: vec![],
        json_path: Some("source".into()),
    });
    for w in &report.warnings {
        let severity = match w.severity.as_str() {
            "error" => Severity::Error,
            "warning" => Severity::Warning,
            "notable" => Severity::Notable,
            _ => Severity::Info,
        };
        out.push(Finding {
            section: "Overview".into(),
            severity,
            field: w.field.clone(),
            title: "Survey warning".into(),
            body: w.message.clone(),
            samples: vec![],
            json_path: Some("warnings".into()),
        });
    }
}

fn schema_findings(report: &SurveyReport, out: &mut Vec<Finding>) {
    use std::collections::BTreeMap;
    let mut by_semantic: BTreeMap<String, u32> = BTreeMap::new();
    for f in report.fields.values() {
        let key = semantic_label(&f.semantic_type);
        *by_semantic.entry(key).or_insert(0) += 1;
    }
    let body = by_semantic
        .iter()
        .map(|(k, v)| format!("- {}: {}", k, v))
        .collect::<Vec<_>>()
        .join("\n");
    out.push(Finding {
        section: "Schema at a glance".into(),
        severity: Severity::Info,
        field: None,
        title: format!("{} field(s) classified", report.fields.len()),
        body,
        samples: vec![],
        json_path: Some("fields".into()),
    });
}

fn unstable_findings(report: &SurveyReport, cfg: &FindingsConfig, out: &mut Vec<Finding>) {
    for (name, profile) in &report.fields {
        if matches!(profile.semantic_type, Some(SemanticType::Unstable)) {
            let samples = reservoir_samples(profile, cfg.max_samples_per_finding);
            let encodings = profile
                .wire_encoding
                .tag_histogram
                .iter()
                .map(|(k, v)| format!("{} ({:.0}%)", k, v * 100.0))
                .collect::<Vec<_>>()
                .join(", ");
            out.push(Finding {
                section: "Unstable fields".into(),
                severity: Severity::Warning,
                field: Some(name.clone()),
                title: format!("Field `{}` is unstable", name),
                body: format!(
                    "Pass 1 could not commit to a SemanticType. Observed wire encodings: {}.",
                    encodings,
                ),
                samples,
                json_path: Some(format!("fields.{}", name)),
            });
        }
    }
}

fn partition_candidate_findings(report: &SurveyReport, cfg: &FindingsConfig, out: &mut Vec<Finding>) {
    for (name, profile) in &report.fields {
        let suitable = matches!(
            profile.cardinality_regime,
            super::types::CardinalityRegime::Binary
                | super::types::CardinalityRegime::LowCard { .. }
        ) && profile.semantic_type.is_some()
            && !matches!(profile.semantic_type, Some(SemanticType::Unstable));
        if !suitable { continue; }
        let distinct = match profile.cardinality_regime {
            super::types::CardinalityRegime::LowCard { exact_distinct } => exact_distinct,
            super::types::CardinalityRegime::Binary => 2,
            _ => 0,
        };
        let samples = reservoir_samples(profile, cfg.max_samples_per_finding);
        out.push(Finding {
            section: "Partition-candidate fields".into(),
            severity: Severity::Notable,
            field: Some(name.clone()),
            title: format!("`{}` is a partition-key candidate", name),
            body: format!(
                "Cardinality regime is {:?} ({} distinct). Suitable for partitioning datasets and as a constraining predicate.",
                profile.cardinality_regime, distinct,
            ),
            samples,
            json_path: Some(format!("fields.{}.cardinality_regime", name)),
        });
    }
}

fn predicate_candidate_findings(report: &SurveyReport, out: &mut Vec<Finding>) {
    for (name, profile) in &report.fields {
        let is_numeric = matches!(
            profile.semantic_type,
            Some(SemanticType::Number(_)) | Some(SemanticType::Boolean) | Some(SemanticType::Temporal(_))
        );
        if !is_numeric { continue; }
        // Must have a quantile sketch to support selectivity-targeted ranges.
        let q = match profile.measures.get("QuantileSketch") {
            Some(MeasureReport::QuantileSketch(r)) => r,
            _ => continue,
        };
        let p50 = q.quantiles.iter().find(|(k, _)| k == "p50").map(|(_, v)| *v);
        let p99 = q.quantiles.iter().find(|(k, _)| k == "p99").map(|(_, v)| *v);
        out.push(Finding {
            section: "Predicate-candidate fields".into(),
            severity: Severity::Notable,
            field: Some(name.clone()),
            title: format!("`{}` supports selectivity-targeted range predicates", name),
            body: format!(
                "Numeric quantile sketch available (k={}, n={}). p50 ≈ {:.4}, p99 ≈ {:.4}. min={:?}, max={:?}.",
                q.k, q.count,
                p50.unwrap_or(f64::NAN),
                p99.unwrap_or(f64::NAN),
                q.min, q.max,
            ),
            samples: vec![],
            json_path: Some(format!("fields.{}.measures.QuantileSketch", name)),
        });
    }
}

fn identifier_findings(report: &SurveyReport, out: &mut Vec<Finding>) {
    for (name, profile) in &report.fields {
        if let Some(SemanticType::Identifier(kind)) = &profile.semantic_type {
            let label = match kind {
                IdentifierKind::Uuid => "UUID",
                IdentifierKind::Ulid => "ULID",
                IdentifierKind::Sequential => "sequential integer ID",
                IdentifierKind::HashLike => "hash-like identifier",
                IdentifierKind::Composite { .. } => "composite identifier",
                IdentifierKind::Opaque => "opaque identifier",
            };
            out.push(Finding {
                section: "Identifier fields".into(),
                severity: Severity::Notable,
                field: Some(name.clone()),
                title: format!("`{}` is a {}", name, label),
                body: "Useful for join planning and synthetic ID generation.".into(),
                samples: vec![],
                json_path: Some(format!("fields.{}.semantic_type", name)),
            });
        }
    }
}

fn cross_field_findings(report: &SurveyReport, cfg: &FindingsConfig, out: &mut Vec<Finding>) {
    // Top numeric correlations by |r|.
    let mut nc: Vec<_> = report
        .cross_field
        .numeric_correlation
        .iter()
        .filter(|e| e.data.pearson_r.is_some())
        .collect();
    nc.sort_by(|a, b| b.data.pearson_r.unwrap().abs().partial_cmp(&a.data.pearson_r.unwrap().abs()).unwrap_or(std::cmp::Ordering::Equal));
    for entry in nc.iter().take(cfg.cross_field_top_k) {
        let r = entry.data.pearson_r.unwrap();
        if r.abs() < cfg.correlation_threshold { continue; }
        out.push(Finding {
            section: "Cross-field highlights".into(),
            severity: Severity::Notable,
            field: None,
            title: format!("Strong numeric correlation: `{}` ↔ `{}` (r={:.3})", entry.a, entry.b, r),
            body: format!(
                "Pearson r = {:.4} over {} paired observations. OLS slope = {:.4}, intercept = {:.4}.",
                r, entry.data.n,
                entry.data.slope.unwrap_or(f64::NAN),
                entry.data.intercept.unwrap_or(f64::NAN),
            ),
            samples: vec![],
            json_path: Some(format!("cross_field.numeric_correlation[{}, {}]", entry.a, entry.b)),
        });
    }
    // Top categorical associations by Cramér's V.
    let mut ca: Vec<_> = report.cross_field.categorical_association.iter().collect();
    ca.sort_by(|a, b| b.data.cramers_v.partial_cmp(&a.data.cramers_v).unwrap_or(std::cmp::Ordering::Equal));
    for entry in ca.iter().take(cfg.cross_field_top_k) {
        if entry.data.cramers_v < cfg.correlation_threshold { continue; }
        out.push(Finding {
            section: "Cross-field highlights".into(),
            severity: Severity::Notable,
            field: None,
            title: format!("Categorical association: `{}` ↔ `{}` (V={:.3})", entry.a, entry.b, entry.data.cramers_v),
            body: format!(
                "Cramér's V = {:.4}, mutual information = {:.4} bits, χ² = {:.2} over {} records.",
                entry.data.cramers_v,
                entry.data.mutual_information_bits,
                entry.data.chi_squared,
                entry.data.n,
            ),
            samples: vec![],
            json_path: Some(format!("cross_field.categorical_association[{}, {}]", entry.a, entry.b)),
        });
    }
    // Top functional dependencies by support.
    let mut fd: Vec<_> = report.cross_field.functional_dependencies.iter().collect();
    fd.sort_by(|a, b| b.data.support.partial_cmp(&a.data.support).unwrap_or(std::cmp::Ordering::Equal));
    for entry in fd.iter().take(cfg.cross_field_top_k) {
        if entry.data.support < 0.9 { continue; }
        out.push(Finding {
            section: "Cross-field highlights".into(),
            severity: Severity::Notable,
            field: None,
            title: format!("Candidate functional dependency: `{}` → `{}` (support={:.1}%)", entry.lhs, entry.rhs, entry.data.support * 100.0),
            body: format!(
                "{} of {} distinct A values map to a single B; total support {:.4} over {} records.",
                entry.data.deterministic_a, entry.data.distinct_a, entry.data.support, entry.data.n,
            ),
            samples: vec![],
            json_path: Some(format!("cross_field.functional_dependencies[{} → {}]", entry.lhs, entry.rhs)),
        });
    }
    // Strong trend with record index.
    let mut trends: Vec<_> = report.cross_field.trend.iter().filter(|t| t.data.pearson_r_with_index.is_some()).collect();
    trends.sort_by(|a, b| b.data.pearson_r_with_index.unwrap().abs().partial_cmp(&a.data.pearson_r_with_index.unwrap().abs()).unwrap_or(std::cmp::Ordering::Equal));
    for entry in trends.iter().take(cfg.cross_field_top_k) {
        let r = entry.data.pearson_r_with_index.unwrap();
        if r.abs() < cfg.correlation_threshold { continue; }
        out.push(Finding {
            section: "Cross-field highlights".into(),
            severity: Severity::Notable,
            field: Some(entry.field.clone()),
            title: format!("`{}` drifts with record order (r={:.3})", entry.field, r),
            body: format!(
                "Pearson r against record index = {:.4}; OLS slope per record = {:.6}.",
                r, entry.data.slope_per_record.unwrap_or(f64::NAN),
            ),
            samples: vec![],
            json_path: Some(format!("cross_field.trend[{}]", entry.field)),
        });
    }
}

fn quality_findings(report: &SurveyReport, cfg: &FindingsConfig, out: &mut Vec<Finding>) {
    for (name, profile) in &report.fields {
        let total = profile.presence.present + profile.presence.absent_in_record;
        if total == 0 { continue; }
        let null_rate = profile.presence.null_count as f64 / total as f64;
        let absent_rate = profile.presence.absent_in_record as f64 / total as f64;
        if null_rate >= cfg.null_rate_warning {
            out.push(Finding {
                section: "Data-quality flags".into(),
                severity: Severity::Warning,
                field: Some(name.clone()),
                title: format!("`{}` has {:.1}% null values", name, null_rate * 100.0),
                body: format!(
                    "{} null observations / {} present / {} absent over {} records.",
                    profile.presence.null_count, profile.presence.present, profile.presence.absent_in_record, total,
                ),
                samples: vec![],
                json_path: Some(format!("fields.{}.presence", name)),
            });
        }
        if absent_rate >= cfg.null_rate_warning {
            out.push(Finding {
                section: "Data-quality flags".into(),
                severity: Severity::Info,
                field: Some(name.clone()),
                title: format!("`{}` is absent from {:.1}% of records", name, absent_rate * 100.0),
                body: "Sparse fields can still be valid; flagged for visibility.".into(),
                samples: vec![],
                json_path: Some(format!("fields.{}.presence", name)),
            });
        }
        if let Some(MeasureReport::TypeStability(t)) = profile.measures.get("TypeStability")
            && t.surprise_count > 0 {
                out.push(Finding {
                    section: "Pass 1 vs Pass 2 surprises".into(),
                    severity: Severity::Warning,
                    field: Some(name.clone()),
                    title: format!("`{}` saw {} unexpected tag(s) in Pass 2", name, t.surprise_count),
                    body: format!("New tags: {}", t.surprise_tags.join(", ")),
                    samples: vec![],
                    json_path: Some(format!("fields.{}.measures.TypeStability", name)),
                });
            }
    }
}

fn reservoir_samples(profile: &FieldProfile, max: usize) -> Vec<serde_json::Value> {
    match profile.measures.get("ReservoirSample") {
        Some(MeasureReport::ReservoirSample(r)) => {
            r.items.iter().take(max).cloned().collect()
        }
        _ => vec![],
    }
}

fn semantic_label(st: &Option<SemanticType>) -> String {
    match st {
        None => "Unknown (no observations)".into(),
        Some(SemanticType::Number(NumberKind::Integer { .. })) => "Number/Integer".into(),
        Some(SemanticType::Number(NumberKind::Decimal { .. })) => "Number/Decimal".into(),
        Some(SemanticType::Number(NumberKind::Floating)) => "Number/Floating".into(),
        Some(SemanticType::Boolean) => "Boolean".into(),
        Some(SemanticType::Temporal(_)) => "Temporal".into(),
        Some(SemanticType::Identifier(_)) => "Identifier".into(),
        Some(SemanticType::Categorical(_)) => "Categorical".into(),
        Some(SemanticType::Structured(_)) => "Structured".into(),
        Some(SemanticType::FreeText) => "FreeText".into(),
        Some(SemanticType::Binary(_)) => "Binary".into(),
        Some(SemanticType::Unstable) => "Unstable".into(),
    }
}

// ---------------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------------

fn render_markdown(report: &SurveyReport, findings: &[Finding]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "# Survey findings");
    let _ = writeln!(s);
    let _ = writeln!(s, "Source: `{}`", report.source.path);
    let _ = writeln!(s, "Records: {} / {} sampled", report.source.sampled_records, report.source.total_records);
    let _ = writeln!(s, "Schema version: {}", report.schema_version);
    let _ = writeln!(s);
    let mut current_section = String::new();
    for f in findings {
        if f.section != current_section {
            current_section = f.section.clone();
            let _ = writeln!(s);
            let _ = writeln!(s, "## {}", current_section);
            let _ = writeln!(s);
        }
        let _ = writeln!(s, "- **[{}]** {}", f.severity.tag(), f.title);
        for line in f.body.lines() {
            let _ = writeln!(s, "  {}", line);
        }
        if !f.samples.is_empty() {
            let samples = f
                .samples
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(s, "  Samples: {}", samples);
        }
        if let Some(p) = &f.json_path {
            let _ = writeln!(s, "  <sub>↳ json:`{}`</sub>", p);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::commands::survey::orchestrator::{
        CrossFieldReport, FieldProfile, SamplingInfo, SourceInfo, SurveyReport,
    };
    use crate::pipeline::commands::survey::measure::{
        ExactExtremaReport, PresenceReport, ReservoirSampleReport, TypeStabilityReport,
    };
    use crate::pipeline::commands::survey::types::{
        CardinalityRegime, NumberKind, NumericWidth, WireEncoding, WireEncodingKind,
    };
    use indexmap::IndexMap;

    fn empty_report() -> SurveyReport {
        SurveyReport {
            schema_version: 1,
            produced_by: "test".into(),
            source: SourceInfo {
                path: "test.slab".into(),
                format: "slab".into(),
                total_records: 0,
                sampled_records: 0,
                sampling: SamplingInfo { mode: "page_stride".into(), page_count: 0 },
            },
            fields: IndexMap::new(),
            cross_field: CrossFieldReport::default(),
            warnings: vec![],
        }
    }

    fn integer_profile(name: &str, n: u64) -> (String, FieldProfile) {
        let mut measures: IndexMap<String, MeasureReport> = IndexMap::new();
        measures.insert("ExactExtrema".into(), MeasureReport::ExactExtrema(ExactExtremaReport {
            min: Some(0.0),
            max: Some(99.0),
        }));
        let profile = FieldProfile {
            wire_encoding: WireEncoding {
                kind: WireEncodingKind::Numeric,
                storage_width: Some(NumericWidth::I32),
                narrowest_width: Some(NumericWidth::I8),
                tag_histogram: IndexMap::new(),
                mixed: false,
            },
            semantic_type: Some(SemanticType::Number(NumberKind::Integer {
                signed: false,
                bit_width_hint: NumericWidth::I8,
            })),
            semantic_confidence: 1.0,
            cardinality_regime: CardinalityRegime::LowCard { exact_distinct: 8 },
            presence: PresenceReport {
                present: n,
                null_count: 0,
                absent_in_record: 0,
            },
            measures,
        };
        (name.into(), profile)
    }

    #[test]
    fn empty_report_renders_clean_overview() {
        let r = empty_report();
        let (md, json) = render_findings(&r, &FindingsConfig::default());
        assert!(md.contains("# Survey findings"));
        assert!(md.contains("Overview"));
        assert!(json.findings.iter().any(|f| f.section == "Overview"));
    }

    #[test]
    fn partition_candidate_surfaces_for_low_card_field() {
        let mut r = empty_report();
        r.source.sampled_records = 100;
        let (name, profile) = integer_profile("flag", 100);
        r.fields.insert(name, profile);
        let (md, json) = render_findings(&r, &FindingsConfig::default());
        assert!(md.contains("partition-key candidate"));
        let f = json
            .findings
            .iter()
            .find(|f| f.section == "Partition-candidate fields")
            .expect("expected partition finding");
        assert_eq!(f.severity, Severity::Notable);
        assert_eq!(f.field.as_deref(), Some("flag"));
    }

    #[test]
    fn unstable_field_surfaces_as_warning() {
        let mut r = empty_report();
        let mut measures: IndexMap<String, MeasureReport> = IndexMap::new();
        measures.insert("ReservoirSample".into(), MeasureReport::ReservoirSample(ReservoirSampleReport {
            items: vec![serde_json::Value::String("oops".into()), serde_json::Value::from(42)],
            observed: 2,
        }));
        let mut tag_hist = IndexMap::new();
        tag_hist.insert("Text".into(), 0.7);
        tag_hist.insert("Int".into(), 0.3);
        let profile = FieldProfile {
            wire_encoding: WireEncoding {
                kind: WireEncodingKind::Null,
                storage_width: None,
                narrowest_width: None,
                tag_histogram: tag_hist,
                mixed: true,
            },
            semantic_type: Some(SemanticType::Unstable),
            semantic_confidence: 0.0,
            cardinality_regime: CardinalityRegime::Unknown,
            presence: PresenceReport { present: 10, null_count: 0, absent_in_record: 0 },
            measures,
        };
        r.fields.insert("messy".into(), profile);
        let (md, json) = render_findings(&r, &FindingsConfig::default());
        assert!(md.contains("unstable"));
        let f = json.findings.iter().find(|f| f.section == "Unstable fields").unwrap();
        assert_eq!(f.severity, Severity::Warning);
        assert!(!f.samples.is_empty(), "expected reservoir samples on unstable finding");
    }

    #[test]
    fn high_null_rate_surfaces_quality_warning() {
        let mut r = empty_report();
        let (name, mut profile) = integer_profile("sparse", 100);
        profile.presence.null_count = 25;
        profile.presence.present = 75;
        r.fields.insert(name, profile);
        let (_, json) = render_findings(&r, &FindingsConfig::default());
        let f = json.findings.iter().find(|f| f.section == "Data-quality flags").unwrap();
        assert_eq!(f.severity, Severity::Warning);
        assert!(f.title.contains("null"));
    }

    #[test]
    fn severity_filter_drops_lower_severity() {
        let mut r = empty_report();
        let (name, profile) = integer_profile("flag", 100);
        r.fields.insert(name, profile);
        let cfg = FindingsConfig { min_severity: Severity::Warning, ..Default::default() };
        let (_, json) = render_findings(&r, &cfg);
        // Partition candidate is Notable; should be filtered.
        assert!(json.findings.iter().all(|f| f.severity >= Severity::Warning));
    }

    #[test]
    fn type_stability_surprise_surfaces_warning() {
        let mut r = empty_report();
        let (name, mut profile) = integer_profile("xs", 100);
        profile.measures.insert("TypeStability".into(), MeasureReport::TypeStability(TypeStabilityReport {
            surprise_count: 5,
            surprise_tags: vec!["Text".into(), "Bool".into()],
        }));
        r.fields.insert(name, profile);
        let (_, json) = render_findings(&r, &FindingsConfig::default());
        let surprise = json.findings.iter().find(|f| f.section == "Pass 1 vs Pass 2 surprises").unwrap();
        assert_eq!(surprise.severity, Severity::Warning);
        assert!(surprise.title.contains("5 unexpected tag"));
    }

    #[test]
    fn findings_json_roundtrips() {
        let r = empty_report();
        let (_, json) = render_findings(&r, &FindingsConfig::default());
        let s = serde_json::to_string(&json).unwrap();
        let _back: FindingsReport = serde_json::from_str(&s).unwrap();
    }
}
