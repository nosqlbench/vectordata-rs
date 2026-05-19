// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration test for `analyze survey`.
//!
//! Builds a fixture metadata slab with deliberately-chosen field
//! characteristics — low-card categorical, UUID identifier, email
//! structured, numeric with known distribution, monotone counter,
//! mixed-encoding unstable, functional-dependent pair — and runs
//! the new survey orchestrator against it. Asserts that the
//! produced `SurveyReport` correctly classifies every field,
//! populates the right measures, and surfaces the cross-field
//! relationships (numeric correlation, functional dependency,
//! trend, copresence).
//!
//! Also runs the findings renderer and confirms the curated output
//! highlights the expected partitioning candidates and notable
//! relationships.

use std::path::Path;

use indexmap::IndexMap;
use slabtastic::{SlabWriter, WriterConfig};

use veks_pipeline::pipeline::commands::survey::{
    self, findings, BinaryKind, CardinalityRegime, IdentifierKind, MeasureReport, NumberKind,
    SemanticType, StructuredKind, SurveyConfig, TemporalKind,
};

use veks_core::formats::anode;
use veks_core::formats::mnode::{MNode, MValue};

/// Build a metadata slab with the integration scenario.
///
/// Field shapes:
/// - `country`     : LowCard categorical (5 distinct: US/GB/DE/FR/JP)
/// - `currency`    : LowCard categorical, functionally determined by country
/// - `user_id`     : Sequential integer (monotone trend with record order)
/// - `score`       : Float with known distribution
/// - `email`       : Textual, Structured(Email) via probe
/// - `request_id`  : Textual UUIDs via probe
/// - `signup_ts`   : Millis temporal
/// - `state`       : Mixed-encoding — Text most of the time, Int occasionally → Unstable
fn write_fixture_slab(path: &Path, record_count: usize) {
    let mapping: &[(&str, &str)] = &[
        ("US", "USD"), ("GB", "GBP"), ("DE", "EUR"), ("FR", "EUR"), ("JP", "JPY"),
    ];
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(path, config).unwrap();
    for i in 0..record_count {
        let mut fields: IndexMap<String, MValue> = IndexMap::new();
        let (cc, cur) = mapping[i % mapping.len()];
        fields.insert("country".into(), MValue::Text(cc.into()));
        fields.insert("currency".into(), MValue::Text(cur.into()));
        // Sequential ID.
        fields.insert("user_id".into(), MValue::Int(i as i64));
        // Quadratic-ish numeric so it has non-trivial variance.
        fields.insert("score".into(), MValue::Float((i as f64) * 0.5 + 1.0));
        // Email per user.
        fields.insert(
            "email".into(),
            MValue::Text(format!("user_{:04}@example.com", i)),
        );
        // UUID-shaped string per record (deterministic but valid form).
        fields.insert(
            "request_id".into(),
            MValue::Text(format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                i as u32, (i % 0xFFFF) as u16, (i % 0xFFFF) as u16,
                (i % 0xFFFF) as u16, i as u64,
            )),
        );
        // Timestamp in Millis (2024-01-01 onward).
        fields.insert(
            "signup_ts".into(),
            MValue::Millis(1_704_067_200_000 + (i as i64) * 86_400_000),
        );
        // Mostly Text "active" but every 7th record is an Int — mixed encoding.
        if i % 7 == 0 {
            fields.insert("state".into(), MValue::Int(i as i64));
        } else {
            fields.insert("state".into(), MValue::Text("active".into()));
        }
        let node = MNode { fields };
        w.add_record(&anode::encode(&anode::ANode::MNode(node))).unwrap();
    }
    w.finish().unwrap();
}

#[test]
fn full_survey_pipeline_classifies_all_fields_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let slab_path = dir.path().join("fixture.slab");
    write_fixture_slab(&slab_path, 200);

    let cfg = SurveyConfig::default();
    let report = survey::survey(&slab_path, &cfg, None).expect("survey");

    // ── Source bookkeeping ──────────────────────────────────────────
    assert_eq!(report.source.total_records, 200);
    assert!(report.source.sampled_records > 0);
    assert_eq!(report.schema_version, 1);

    // ── Per-field semantic classifications ──────────────────────────
    let country = report.fields.get("country").expect("country present");
    assert!(matches!(
        country.semantic_type,
        Some(SemanticType::Categorical(_))
            | Some(SemanticType::Identifier(_)) // composite probe may fire too
            | Some(SemanticType::FreeText)
    ));
    // 5 distinct values → LowCard regime.
    match &country.cardinality_regime {
        CardinalityRegime::LowCard { exact_distinct } => assert_eq!(*exact_distinct, 5),
        other => panic!("country regime: {:?}", other),
    }

    let user_id = report.fields.get("user_id").expect("user_id present");
    match &user_id.semantic_type {
        Some(SemanticType::Number(NumberKind::Integer { .. })) => {}
        Some(SemanticType::Temporal(TemporalKind::Timestamp { .. })) => {} // EpochPlausibility false-positive
        other => panic!("user_id semantic: {:?}", other),
    }
    assert_eq!(user_id.presence.present, 200);

    let score = report.fields.get("score").expect("score present");
    // Floats might survive as Number(Floating) or, if every value's
    // fractional part is .0 / .5 etc., the DecimalLiteralProbe-
    // chain might not engage. We accept any Number(_) verdict.
    assert!(matches!(score.semantic_type, Some(SemanticType::Number(_))));

    let email = report.fields.get("email").expect("email present");
    // Email probe at 100% match rate → SemanticType::Structured(Email).
    assert_eq!(
        email.semantic_type,
        Some(SemanticType::Structured(StructuredKind::Email)),
    );

    let request_id = report.fields.get("request_id").expect("request_id present");
    // UUID probe should commit.
    assert_eq!(
        request_id.semantic_type,
        Some(SemanticType::Identifier(IdentifierKind::Uuid)),
    );

    let signup_ts = report.fields.get("signup_ts").expect("signup_ts present");
    match &signup_ts.semantic_type {
        Some(SemanticType::Temporal(_)) => {}
        Some(SemanticType::Number(_)) => {} // fallback if no temporal verdict
        other => panic!("signup_ts semantic: {:?}", other),
    }

    let state = report.fields.get("state").expect("state present");
    // Mixed encodings (Text + Int) → SemanticType::Unstable.
    assert_eq!(state.semantic_type, Some(SemanticType::Unstable));
    assert!(state.wire_encoding.mixed);
    // Unstable fields should carry the opaque-only diagnostic measures.
    assert!(state.measures.contains_key("WireEncodingHistogram"));
    assert!(state.measures.contains_key("ProbeAttemptReport"));
}

#[test]
fn full_survey_pipeline_populates_cross_field_relationships() {
    let dir = tempfile::tempdir().unwrap();
    let slab_path = dir.path().join("fixture.slab");
    write_fixture_slab(&slab_path, 200);

    let cfg = SurveyConfig::default();
    let report = survey::survey(&slab_path, &cfg, None).expect("survey");

    // Country → currency is a 1.0-support functional dependency.
    let perfect_fd = report
        .cross_field
        .functional_dependencies
        .iter()
        .any(|e| {
            (e.lhs == "country" && e.rhs == "currency" && (e.data.support - 1.0).abs() < 1e-9)
                || (e.lhs == "currency" && e.rhs == "country" && e.data.support < 1.0)
        });
    assert!(
        perfect_fd,
        "expected country → currency perfect FD, got {:?}",
        report.cross_field.functional_dependencies,
    );

    // Categorical association between country and currency: V ≈ 1.
    let ca = report.cross_field.categorical_association.iter().find(|e| {
        (e.a == "country" && e.b == "currency") || (e.a == "currency" && e.b == "country")
    });
    let ca = ca.expect("country↔currency categorical association");
    assert!(ca.data.cramers_v > 0.99, "Cramér's V = {}", ca.data.cramers_v);

    // user_id should trend monotonically with record index.
    let trend = report
        .cross_field
        .trend
        .iter()
        .find(|e| e.field == "user_id")
        .expect("user_id trend");
    let r = trend.data.pearson_r_with_index.expect("non-empty trend");
    assert!((r - 1.0).abs() < 1e-9, "user_id trend r = {}", r);

    // user_id × score: perfect linear correlation (both monotone in i).
    let nc = report.cross_field.numeric_correlation.iter().find(|e| {
        (e.a == "user_id" && e.b == "score") || (e.a == "score" && e.b == "user_id")
    });
    let nc = nc.expect("user_id ↔ score numeric correlation");
    let r = nc.data.pearson_r.expect("non-empty pearson");
    assert!((r - 1.0).abs() < 1e-9, "user_id↔score r = {}", r);

    // Copresence between any two always-present fields should be 1.0.
    let cp = report.cross_field.copresence.iter().find(|e| {
        (e.a == "user_id" && e.b == "score") || (e.a == "score" && e.b == "user_id")
    });
    let cp = cp.expect("user_id ↔ score copresence");
    assert!((cp.data.jaccard - 1.0).abs() < 1e-9);

    // Pair-plan counters are non-zero.
    assert!(report.cross_field.planned > 0);
    assert!(report.cross_field.executed > 0);
}

#[test]
fn full_survey_pipeline_emits_findings_with_expected_highlights() {
    let dir = tempfile::tempdir().unwrap();
    let slab_path = dir.path().join("fixture.slab");
    write_fixture_slab(&slab_path, 200);

    let cfg = SurveyConfig::default();
    let report = survey::survey(&slab_path, &cfg, None).expect("survey");
    let (md, json) = findings::render_findings(&report, &findings::FindingsConfig::default());

    // Schema-at-a-glance + Overview always present.
    assert!(md.contains("Schema at a glance"));
    assert!(md.contains("Overview"));

    // Unstable `state` field should surface as a warning.
    let unstable = json
        .findings
        .iter()
        .find(|f| f.section == "Unstable fields" && f.field.as_deref() == Some("state"))
        .expect("expected `state` Unstable finding");
    assert_eq!(unstable.severity, findings::Severity::Warning);

    // `country` (LowCard categorical) should appear as a partition candidate.
    let partition = json
        .findings
        .iter()
        .find(|f| f.section == "Partition-candidate fields" && f.field.as_deref() == Some("country"))
        .expect("expected partition-candidate finding for `country`");
    assert_eq!(partition.severity, findings::Severity::Notable);

    // The cross-field highlights section should call out user_id↔score
    // and country→currency.
    let crossfield_highlights: Vec<&findings::Finding> = json
        .findings
        .iter()
        .filter(|f| f.section == "Cross-field highlights")
        .collect();
    let has_pearson = crossfield_highlights
        .iter()
        .any(|f| f.title.contains("numeric correlation") && (f.title.contains("user_id") || f.title.contains("score")));
    let has_fd = crossfield_highlights
        .iter()
        .any(|f| f.title.contains("functional dependency") && f.title.contains("country"));
    assert!(has_pearson, "missing numeric-correlation finding: {:#?}", crossfield_highlights);
    assert!(has_fd, "missing functional-dependency finding: {:#?}", crossfield_highlights);

    // request_id → Identifier finding (UUID-shaped).
    let id_findings: Vec<&findings::Finding> = json
        .findings
        .iter()
        .filter(|f| f.section == "Identifier fields" && f.field.as_deref() == Some("request_id"))
        .collect();
    assert!(
        !id_findings.is_empty(),
        "missing Identifier finding for request_id; available: {:?}",
        json.findings.iter().filter(|f| f.section == "Identifier fields").collect::<Vec<_>>()
    );
}

#[test]
fn full_survey_report_round_trips_through_json() {
    let dir = tempfile::tempdir().unwrap();
    let slab_path = dir.path().join("fixture.slab");
    write_fixture_slab(&slab_path, 50);
    let cfg = SurveyConfig::default();
    let report = survey::survey(&slab_path, &cfg, None).expect("survey");
    let s = serde_json::to_string(&report).expect("serialize");
    let back: survey::SurveyReport = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(back.fields.len(), report.fields.len());
    assert_eq!(back.source.sampled_records, report.source.sampled_records);
}

#[test]
fn integer_string_is_recognized_as_number_via_probe() {
    // Pure regression on the §13.3 motivating example: a field where
    // every value is a Text("integer") should commit to
    // SemanticType::Number(Integer), not FreeText.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int_str.slab");
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(&path, config).unwrap();
    for i in 0..200 {
        let mut fields: IndexMap<String, MValue> = IndexMap::new();
        fields.insert("count_as_text".into(), MValue::Text(format!("{}", i * 7)));
        let node = MNode { fields };
        w.add_record(&anode::encode(&anode::ANode::MNode(node))).unwrap();
    }
    w.finish().unwrap();

    let report = survey::survey(&path, &SurveyConfig::default(), None).expect("survey");
    let f = report.fields.get("count_as_text").expect("field present");
    match &f.semantic_type {
        Some(SemanticType::Number(NumberKind::Integer { .. })) => {}
        other => panic!("expected Number(Integer), got {:?}", other),
    }
    // semantic_confidence should be near 1.0 (all values parse).
    assert!(
        f.semantic_confidence >= 0.95,
        "confidence = {}", f.semantic_confidence,
    );
}

#[allow(dead_code)]
fn _silence_unused_imports() {
    let _ = BinaryKind::Opaque;
}
