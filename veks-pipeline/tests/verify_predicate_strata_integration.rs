// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `verify predicate-strata` holds a stratified predicate set's claims
//! against the answer keys: exact counts at the census profile, bands
//! at sized profiles above the threshold, one record per query, and
//! every `query_in_filter` label re-derived from the queries' own rows
//! (TS-167, TS-168).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use slabtastic::{SlabReader, SlabWriter, WriterConfig};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::pnode::PNode;
use veks_core::formats::pnode::eval::evaluate;
use veks_core::ui::{TestSink, UiHandle};
use veks_pipeline::pipeline::command::{CommandOp, Options, Status, StreamContext};
use veks_pipeline::pipeline::commands::gen_predicates::GenPredicatesOp;
use veks_pipeline::pipeline::commands::survey::{self, CensusConfig, SurveyConfig};
use veks_pipeline::pipeline::commands::verify_predicate_strata::{StrataReport, VerifyPredicateStrataOp};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;

const N: usize = 6000;
const COUNT: usize = 120;
const BUCKETS: i64 = 4096;

fn tmp_dir() -> tempfile::TempDir {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

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

/// Deterministic rows with a few censusable fields.
fn write_rows(path: &Path) -> Vec<MNode> {
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(path, config).unwrap();
    let classes = ["introduction", "methods", "results", "other"];
    let topics = ["physics", "biology", "chemistry"];
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        let mut fields = IndexMap::new();
        fields.insert("topic_l1".to_string(), MValue::Text(topics[(i * 7) % 3].into()));
        fields.insert("section_class".to_string(), MValue::Text(classes[(i * 31) % 4].into()));
        fields.insert("year".to_string(), MValue::Int32(2010 + ((i * 13) % 12) as i32));
        fields.insert("isopenaccess".to_string(), MValue::Bool(i % 3 == 0));
        fields.insert("citation_percentile".to_string(), MValue::Short(((i * 37) % 100) as i16));
        fields.insert("passage_position".to_string(), MValue::Short(((i * 11) % 100) as i16));
        fields.insert("word_count".to_string(), MValue::Short((60 + (i * 17) % 171) as i16));
        // A hash-like bucket, so every prefix of the rows is uniform over it.
        fields.insert("sample_bucket".to_string(), MValue::Int32(((i as u64).wrapping_mul(2_654_435_761) % BUCKETS as u64) as i32));
        let node = MNode { fields };
        w.add_record(&anode::encode(&ANode::MNode(node.clone()))).unwrap();
        rows.push(node);
    }
    w.finish().unwrap();
    rows
}

fn write_survey(dir: &Path, slab: &Path) {
    let cfg = SurveyConfig {
        samples: 500,
        census: CensusConfig {
            auto: true,
            listed: vec![],
            cap: 1000,
            hierarchies: vec![vec!["topic_l1".into()]],
            pairs: vec![("topic_l1".into(), "year".into())],
            ..CensusConfig::default()
        },
        ..SurveyConfig::default()
    };
    let report = survey::survey(slab, &cfg, None).expect("survey");
    std::fs::write(dir.join("survey.json"), serde_json::to_string(&report).unwrap()).unwrap();
}

fn write_mnodes(path: &Path, rows: &[MNode]) {
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(path, config).unwrap();
    for r in rows {
        w.add_record(&anode::encode(&ANode::MNode(r.clone()))).unwrap();
    }
    w.finish().unwrap();
}

fn predicates(path: &Path) -> Vec<PNode> {
    let reader = SlabReader::open(path).unwrap();
    let mut out = Vec::new();
    for entry in reader.page_entries() {
        let page = reader.read_data_page(&entry).unwrap();
        for i in 0..page.record_count() {
            out.push(PNode::from_bytes_named(page.get_record(i).unwrap()).unwrap());
        }
    }
    out
}

/// The answer key `compute evaluate-predicates` would write over the
/// first `n` rows: per predicate, the packed ordinals of matching rows.
/// `drop` removes one ordinal from the first non-empty record, to plant
/// a mismatch.
fn write_results(path: &Path, preds: &[PNode], rows: &[MNode], n: usize, drop: bool) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(path, config).unwrap();
    let mut dropped = false;
    for p in preds {
        let mut ordinals: Vec<i32> = rows[..n]
            .iter()
            .enumerate()
            .filter(|(_, r)| evaluate(p, r))
            .map(|(i, _)| i as i32)
            .collect();
        // The planted mismatch goes on a censused predicate: a control
        // predicate's count is by construction and is band-checked.
        if drop && !dropped && !ordinals.is_empty() && !p.to_string().contains("sample_bucket") {
            ordinals.pop();
            dropped = true;
        }
        let mut bytes = Vec::with_capacity(ordinals.len() * 4);
        for o in ordinals {
            bytes.extend_from_slice(&o.to_le_bytes());
        }
        w.add_record(&bytes).unwrap();
    }
    w.finish().unwrap();
}

/// Generate a stratified set with query rows, then lay out the answer
/// keys of a census profile and a sized profile.
fn build(dir: &Path) -> (Vec<PNode>, Vec<MNode>, Vec<MNode>) {
    let slab = dir.join("metadata_content.slab");
    let rows = write_rows(&slab);
    write_survey(dir, &slab);
    std::fs::create_dir_all(dir.join("profiles/base")).unwrap();
    let query_rows: Vec<MNode> = (0..COUNT).map(|i| rows[(i * 97) % N].clone()).collect();
    write_mnodes(&dir.join("profiles/base/query_metadata.slab"), &query_rows);
    let mut o = Options::new();
    o.set("strategy", "stratified");
    o.set("survey", "survey.json");
    o.set("output", "profiles/base/predicates.slab");
    o.set("decades", "1e-1..1e-2");
    o.set("per-cell", "1,1");
    o.set("buckets", BUCKETS.to_string());
    o.set("base-count", N.to_string());
    o.set("min-matches", "5");
    o.set("reliability-threshold", "1000");
    o.set("seed", "42");
    o.set("count", COUNT.to_string());
    o.set("query-metadata", "profiles/base/query_metadata.slab");
    let mut op = GenPredicatesOp;
    let r = op.execute(&o, &mut test_ctx(dir));
    assert_eq!(r.status, Status::Ok, "{}", r.message);
    let preds = predicates(&dir.join("profiles/base/predicates.slab"));
    assert_eq!(preds.len(), COUNT);
    write_results(&dir.join("profiles/default/metadata_results.slab"), &preds, &rows, N, false);
    write_results(&dir.join("profiles/3000/metadata_results.slab"), &preds, &rows, 3000, false);
    (preds, rows, query_rows)
}

fn verify(dir: &Path, extra: &[(&str, &str)]) -> (Status, String, StrataReport) {
    let mut o = Options::new();
    o.set("predicates", "profiles/base/predicates.slab");
    o.set("reliability-threshold", "1000");
    o.set("output", ".cache/verify_predicate_strata.json");
    for (k, v) in extra {
        o.set(*k, *v);
    }
    let mut op = VerifyPredicateStrataOp;
    let r = op.execute(&o, &mut test_ctx(dir));
    let report: StrataReport = serde_json::from_str(
        &std::fs::read_to_string(dir.join(".cache/verify_predicate_strata.json")).unwrap(),
    )
    .unwrap();
    (r.status, r.message, report)
}

#[test]
fn claims_hold_at_the_census_profile_and_in_band_at_a_sized_one() {
    let dir = tmp_dir();
    build(dir.path());
    let (status, message, report) = verify(
        dir.path(),
        &[("query-metadata", "profiles/base/query_metadata.slab")],
    );
    assert_eq!(status, Status::Ok, "{}", message);
    assert_eq!(report.predicates, COUNT);
    assert_eq!(report.census_population, N as u64);
    assert_eq!(report.violations, 0);
    assert_eq!(report.label_checks, Some(COUNT));
    assert_eq!(report.label_disagreements, Some(0));
    assert_eq!(report.profiles.len(), 2);
    let sized = report.profiles.iter().find(|p| p.profile == "3000").unwrap();
    let census = report.profiles.iter().find(|p| p.profile == "default").unwrap();
    assert!(!sized.census_profile && sized.above_threshold);
    assert!(census.census_profile);
    assert_eq!((census.exact_mismatches, sized.band_violations, sized.zero_matches), (0, 0, 0));
    for (family, f) in &census.per_family {
        // Censused families are exact; the control family's count is by
        // construction and lands within its band.
        let tolerance = if family == "control" { 1e-3 } else { 1e-12 };
        assert!(
            (f.mean_claimed_selectivity - f.mean_realised_selectivity).abs() < tolerance,
            "{family}: {:?}",
            f
        );
    }
}

#[test]
fn a_single_missing_ordinal_at_the_census_profile_fails() {
    let dir = tmp_dir();
    let (preds, rows, _) = build(dir.path());
    write_results(&dir.path().join("profiles/default/metadata_results.slab"), &preds, &rows, N, true);
    let (status, message, report) = verify(dir.path(), &[]);
    assert_eq!(status, Status::Error, "{}", message);
    assert!(message.contains("exact mismatch"), "{}", message);
    let census = report.profiles.iter().find(|p| p.profile == "default").unwrap();
    assert_eq!(census.exact_mismatches, 1);
    assert!(census.first_violations[0].contains("expected"));
}

#[test]
fn labels_are_re_derived_from_the_queries_own_rows() {
    let dir = tmp_dir();
    let (_, _, query_rows) = build(dir.path());
    // Rows shifted by one are somebody else's: most labels must disagree.
    let mut shifted = query_rows.clone();
    shifted.rotate_left(1);
    write_mnodes(&dir.path().join("profiles/base/other_rows.slab"), &shifted);
    let (status, message, report) = verify(
        dir.path(),
        &[("query-metadata", "profiles/base/other_rows.slab")],
    );
    assert_eq!(status, Status::Error, "{}", message);
    assert!(message.contains("query_in_filter"), "{}", message);
    assert!(report.label_disagreements.unwrap() > 0);
}

#[test]
fn a_predicate_count_other_than_the_query_count_fails() {
    let dir = tmp_dir();
    build(dir.path());
    // A query facet with a different count: 8-dim fvecs, 119 records.
    let mut bytes = Vec::new();
    for _ in 0..119 {
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
    }
    std::fs::write(dir.path().join("profiles/base/query_vectors.fvecs"), bytes).unwrap();
    let (status, message, _) = verify(dir.path(), &[("queries", "profiles/base/query_vectors.fvecs")]);
    assert_eq!(status, Status::Error, "{}", message);
    assert!(message.contains("119"), "{}", message);
}
