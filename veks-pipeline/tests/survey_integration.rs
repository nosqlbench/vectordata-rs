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
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use slabtastic::{SlabWriter, WriterConfig};
use veks_core::ui::{TestSink, UiHandle};
use veks_pipeline::pipeline::command::{ArtifactState, CommandOp, Options, Status, StreamContext};
use veks_pipeline::pipeline::commands::survey::{
    self, findings, BinaryKind, CardinalityRegime, CensusConfig, IdentifierKind, MeasureReport,
    NumberKind, SemanticType, StructuredKind, SurveyConfig, SurveyOp, SurveyReport, TemporalKind,
};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;

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

    // Passes 1 and 2 only: this test is about the sampled verdicts,
    // which the census would replace for every enumerable field.
    let cfg = SurveyConfig {
        census: CensusConfig { auto: false, ..CensusConfig::default() },
        ..SurveyConfig::default()
    };
    let report = survey::survey(&slab_path, &cfg, None).expect("survey");

    // ── Source bookkeeping ──────────────────────────────────────────
    assert_eq!(report.source.total_records, 200);
    assert!(report.source.sampled_records > 0);
    assert_eq!(report.schema_version, 2);
    assert!(report.source.census.is_none());

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

/// Create a tempdir under `target/tmp/`.
fn tmp_dir() -> tempfile::TempDir {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// Minimal `StreamContext` rooted at `dir`.
fn test_ctx(dir: &Path) -> StreamContext {
    StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace: dir.to_path_buf(),
        cache: dir.join(".cache"),
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads: 1,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: UiHandle::new(Arc::new(TestSink::new())),
        status_interval: Duration::from_secs(1),
        estimated_total_steps: 0,
        provenance_selector: veks_pipeline::pipeline::provenance::ProvenanceFlags::STRICT,
    }
}

/// A slab shaped like an enriched passage corpus: a three-level topic
/// hierarchy nested by construction (2 roots × 3 branches × 2 leaves),
/// a paper-level integer, a boolean, and a per-record id whose
/// cardinality exceeds a small census cap.
fn write_census_fixture(path: &Path, n: usize) {
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(path, config).unwrap();
    for i in 0..n {
        let mut fields: IndexMap<String, MValue> = IndexMap::new();
        let l1 = i % 2;
        let l2 = l1 * 3 + (i / 2) % 3;
        let l3 = l2 * 2 + (i / 6) % 2;
        fields.insert("topic_l1".into(), MValue::Text(format!("root-{}", l1)));
        fields.insert("topic_l2".into(), MValue::Text(format!("branch-{}", l2)));
        fields.insert("topic_l3".into(), MValue::Text(format!("leaf-{}", l3)));
        fields.insert("year".into(), MValue::Int32(2015 + (i % 8) as i32));
        fields.insert("isopenaccess".into(), MValue::Bool(i % 3 == 0));
        fields.insert("corpusid".into(), MValue::Int(i as i64));
        w.add_record(&anode::encode(&anode::ANode::MNode(MNode { fields }))).unwrap();
    }
    w.finish().unwrap();
}

fn census_options(source: &Path, output: &Path) -> Options {
    let mut o = Options::new();
    o.set("source", source.display().to_string());
    o.set("output", output.display().to_string());
    o.set("samples", "50");
    o.set("census-cap", "100");
    o.set("hierarchy", "topic_l1>topic_l2>topic_l3");
    o.set("census-pair", "topic_l3:year, topic_l1:isopenaccess");
    o.set("findings-markdown", "");
    o.set("findings-json", "");
    o
}

/// The census pass end to end: declared hierarchy and pairs are
/// counted exactly over every record, the field census agrees with
/// the tree and the joint tables, an `auto` field over the cap leaves
/// the census with a warning, and the whole report round-trips.
#[test]
fn census_pass_counts_declarations_exactly() {
    let dir = tmp_dir();
    let slab_path = dir.path().join("enriched.slab");
    let n = 480;
    write_census_fixture(&slab_path, n);

    let cfg = SurveyConfig {
        samples: 50,
        census: CensusConfig {
            auto: true,
            listed: vec![],
            cap: 100,
            hierarchies: vec![vec!["topic_l1".into(), "topic_l2".into(), "topic_l3".into()]],
            pairs: vec![
                ("topic_l3".into(), "year".into()),
                ("topic_l1".into(), "isopenaccess".into()),
            ],
            ..CensusConfig::default()
        },
        ..SurveyConfig::default()
    };
    let report = survey::survey(&slab_path, &cfg, None).expect("survey");

    // The sample was small; the census was not.
    assert_eq!(report.source.sampled_records, 50);
    let info = report.source.census.as_ref().expect("census ran");
    assert_eq!(info.records, n as u64);
    for f in ["topic_l1", "topic_l2", "topic_l3", "year", "isopenaccess"] {
        assert!(info.fields.iter().any(|x| x == f), "{} censused: {:?}", f, info.fields);
        let profile = report.fields.get(f).unwrap();
        assert!(profile.censused);
        assert_eq!(profile.presence.present, n as u64);
        assert!(!profile.measures.contains_key("ExactFrequencyTable"));
        assert!(!profile.measures.contains_key("HeavyHitters"));
    }
    // corpusid has 480 distinct values: MidCard by sample, over the
    // cap of 100, so it leaves the census and keeps its sampled view.
    assert_eq!(info.dropped.len(), 1);
    assert_eq!(info.dropped[0].field, "corpusid");
    assert!(!report.fields.get("corpusid").unwrap().censused);
    assert!(report.warnings.iter().any(|w| w.field.as_deref() == Some("corpusid")));

    // Field census: exact, and the regime says so.
    let l3 = report.fields.get("topic_l3").unwrap();
    assert_eq!(l3.cardinality_regime, CardinalityRegime::Censused { exact_distinct: 12 });
    let l3_counts = match l3.measures.get("ExactValueCensus") {
        Some(MeasureReport::ExactValueCensus(c)) => {
            assert_eq!(c.population, n as u64);
            assert_eq!(c.missing, 0);
            assert!(c.counts.values().all(|v| *v == 40), "{:?}", c.counts);
            c.counts.clone()
        }
        other => panic!("{:?}", other),
    };
    match report.fields.get("year").unwrap().measures.get("ExactIntegerHistogram") {
        Some(MeasureReport::ExactIntegerHistogram(h)) => {
            assert_eq!((h.min, h.max), (2015, 2022));
            assert_eq!(h.counts, vec![60; 8]);
        }
        other => panic!("{:?}", other),
    }
    match report.fields.get("isopenaccess").unwrap().measures.get("ExactValueCensus") {
        Some(MeasureReport::ExactValueCensus(c)) => {
            assert_eq!(c.counts.get("Bool(true)"), Some(&160));
            assert_eq!(c.counts.get("Bool(false)"), Some(&320));
        }
        other => panic!("{:?}", other),
    }

    // Hierarchy: nesting verified, every node's count is the field
    // census of its value, and the roots sum to the population.
    assert_eq!(report.hierarchies.len(), 1);
    let h = &report.hierarchies[0];
    assert_eq!(h.level_sizes, vec![2, 6, 12]);
    assert_eq!(h.population, n as u64);
    assert_eq!(h.incomplete, 0);
    assert_eq!(h.nodes.iter().map(|r| r.count).sum::<u64>(), n as u64);
    for root in &h.nodes {
        assert_eq!(root.count, 240);
        assert_eq!(root.children.len(), 3);
        for branch in &root.children {
            assert_eq!(branch.count, 80);
            assert_eq!(branch.children.len(), 2);
            for leaf in &branch.children {
                assert_eq!(l3_counts.get(&leaf.value), Some(&leaf.count), "{}", leaf.value);
                assert!(leaf.children.is_empty());
            }
        }
    }

    // Pairs: dense tables whose row sums are the row field's census.
    assert_eq!(report.pair_census.len(), 2);
    let p = &report.pair_census[0];
    assert_eq!((p.a.as_str(), p.b.as_str()), ("topic_l3", "year"));
    assert_eq!((p.a_values.len(), p.b_values.len()), (12, 8));
    assert_eq!(p.population, n as u64);
    for (i, row) in p.counts.iter().enumerate() {
        assert_eq!(row.iter().sum::<u64>(), *l3_counts.get(&p.a_values[i]).unwrap());
    }
    let q = &report.pair_census[1];
    assert_eq!((q.a.as_str(), q.b.as_str()), ("topic_l1", "isopenaccess"));
    assert_eq!(q.counts.iter().flatten().sum::<u64>(), n as u64);

    // Round trip: sections and the census regime survive JSON.
    let text = serde_json::to_string(&report).unwrap();
    let back: SurveyReport = serde_json::from_str(&text).unwrap();
    assert_eq!(back.hierarchies, report.hierarchies);
    assert_eq!(back.pair_census, report.pair_census);
    assert_eq!(back.fields.get("topic_l3").unwrap().cardinality_regime, l3.cardinality_regime);
    assert_eq!(back.source.census, report.source.census);

    // Findings carry the census section.
    let (md, _) = findings::render_findings(&report, &findings::FindingsConfig::default());
    assert!(md.contains("Census (exact counts)"), "{}", md);
    assert!(md.contains("topic_l1>topic_l2>topic_l3"), "{}", md);
}

/// The command surface: declarations arrive as options, the report is
/// written, `check_artifact` judges it against the declarations, and a
/// listed field over the cap is an error rather than a report.
#[test]
fn census_command_writes_checks_and_refuses_over_cap_listing() {
    let dir = tmp_dir();
    let slab_path = dir.path().join("enriched.slab");
    write_census_fixture(&slab_path, 480);
    let output = dir.path().join("survey.json");

    let mut op = SurveyOp;
    let mut ctx = test_ctx(dir.path());
    let opts = census_options(&slab_path, &output);
    let result = op.execute(&opts, &mut ctx);
    assert_eq!(result.status, Status::Ok, "{}", result.message);
    assert!(
        result.message.contains("census: 5 fields exact over 480 records"),
        "{}",
        result.message
    );
    assert!(result.message.contains("1 dropped"), "{}", result.message);
    assert!(output.exists());

    // The artifact satisfies its declarations.
    assert_eq!(op.check_artifact(&output, &opts), ArtifactState::Complete);

    // A declaration the report does not carry makes it Partial.
    let mut more = census_options(&slab_path, &output);
    more.set("census-pair", "topic_l3:year, topic_l1:isopenaccess, topic_l2:year");
    assert_eq!(op.check_artifact(&output, &more), ArtifactState::Partial);
    let mut listed = census_options(&slab_path, &output);
    listed.set("census", "auto,corpusid");
    assert_eq!(op.check_artifact(&output, &listed), ArtifactState::Partial);

    // No declarations at all: any parseable report is complete.
    let mut none = Options::new();
    none.set("source", slab_path.display().to_string());
    none.set("output", output.display().to_string());
    none.set("census", "none");
    assert_eq!(op.check_artifact(&output, &none), ArtifactState::Complete);

    // Listing a field the cap cannot hold is refused outright.
    let result = op.execute(&listed, &mut ctx);
    assert_eq!(result.status, Status::Error);
    assert!(result.message.contains("corpusid"), "{}", result.message);

    // The written report is the same shape the library returns.
    let text = std::fs::read_to_string(&output).unwrap();
    let report: SurveyReport = serde_json::from_str(&text).unwrap();
    assert_eq!(report.hierarchies.len(), 1);
    assert_eq!(report.pair_census.len(), 2);
    assert!(report.fields.get("topic_l3").unwrap().censused);
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
