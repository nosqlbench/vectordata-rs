// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `generate predicates --strategy stratified`.
//!
//! Builds an enriched metadata slab with every column the design
//! derives — a two-level topic hierarchy, `section_class`, `year`,
//! `isopenaccess`, `citation_percentile`, `passage_position`,
//! `word_count` and a uniform `sample_bucket` — surveys it with the
//! census declarations the tessera pipeline uses, draws a stratified
//! predicate set from the survey, and then checks the strongest
//! property the design promises: every predicate's expected count in
//! the `generation` namespace equals its exact match count over the
//! slab, and every `families` record's selectivity lies in its cell's
//! band. Also checks the namespaces agree, the artifact check, and
//! that the draw is byte-identical across runs.

use std::collections::HashMap;
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
use veks_pipeline::pipeline::command::{ArtifactState, CommandOp, Options, Status, StreamContext};
use veks_pipeline::pipeline::commands::gen_predicates::GenPredicatesOp;
use veks_pipeline::pipeline::commands::gen_predicates_stratified::GenerationReport;
use veks_pipeline::pipeline::commands::survey::{self, CensusConfig, SurveyConfig};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;

const N: usize = 6_000;
const BUCKETS: i64 = 1_000;

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
        threads: 2,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: UiHandle::new(Arc::new(TestSink::new())),
        status_interval: Duration::from_secs(1),
        estimated_total_steps: 0,
        provenance_selector: veks_pipeline::pipeline::provenance::ProvenanceFlags::STRICT,
    }
}

/// An enriched slab with skewed but deterministic distributions, so
/// candidates land in several decades.
fn write_enriched_slab(path: &Path) -> Vec<MNode> {
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(path, config).unwrap();
    let l1 = ["physics", "biology"];
    let l2 = [
        "optics-lasers",
        "plasma",
        "cosmology",
        "genomics",
        "ecology",
        "neuroscience",
    ];
    let classes = ["introduction", "methods", "results", "other"];
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        // Skewed leaf sizes: leaf k gets a share proportional to 2^k.
        let leaf = {
            let r = (i * 7919) % 63;
            if r < 1 {
                0
            } else if r < 3 {
                1
            } else if r < 7 {
                2
            } else if r < 15 {
                3
            } else if r < 31 {
                4
            } else {
                5
            }
        };
        let mut fields = IndexMap::new();
        fields.insert("topic_l1".to_string(), MValue::Text(l1[leaf / 3].into()));
        fields.insert("topic_l2".to_string(), MValue::Text(l2[leaf].into()));
        fields.insert(
            "section_class".to_string(),
            MValue::Text(classes[(i * 31) % 4].into()),
        );
        fields.insert(
            "year".to_string(),
            MValue::Int32(2010 + ((i * 13) % 12) as i32),
        );
        fields.insert("isopenaccess".to_string(), MValue::Bool(i % 3 == 0));
        fields.insert(
            "citation_percentile".to_string(),
            MValue::Short(((i * 37) % 100) as i16),
        );
        fields.insert(
            "passage_position".to_string(),
            MValue::Short(((i * 11) % 100) as i16),
        );
        fields.insert(
            "word_count".to_string(),
            MValue::Short((60 + (i * 17) % 171) as i16),
        );
        fields.insert(
            "sample_bucket".to_string(),
            MValue::Int32((i as i64 % BUCKETS) as i32),
        );
        let node = MNode { fields };
        w.add_record(&anode::encode(&ANode::MNode(node.clone())))
            .unwrap();
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
            hierarchies: vec![vec!["topic_l1".into(), "topic_l2".into()]],
            pairs: vec![
                ("topic_l2".into(), "citation_percentile".into()),
                ("topic_l1".into(), "isopenaccess".into()),
                ("topic_l2".into(), "year".into()),
            ],
            ..CensusConfig::default()
        },
        ..SurveyConfig::default()
    };
    let report = survey::survey(slab, &cfg, None).expect("survey");
    std::fs::write(
        dir.join("survey.json"),
        serde_json::to_string(&report).unwrap(),
    )
    .unwrap();
}

fn options() -> Options {
    let mut o = Options::new();
    o.set("strategy", "stratified");
    o.set("survey", "survey.json");
    o.set("output", "profiles/base/predicates.slab");
    o.set("decades", "1e-1..1e-3");
    o.set("per-cell", "4,6,8");
    o.set("buckets", BUCKETS.to_string());
    o.set("base-count", N.to_string());
    o.set("min-matches", "5");
    o.set("reliability-threshold", "1000");
    o.set("seed", "42");
    o
}

/// Decode every MNode of a namespace.
fn namespace_records(path: &Path, ns: Option<&str>) -> Vec<MNode> {
    let reader = SlabReader::open_namespace(path, ns).unwrap();
    let mut out = Vec::new();
    for entry in reader.page_entries() {
        let page = reader.read_data_page(&entry).unwrap();
        for i in 0..page.record_count() {
            if let Ok(ANode::MNode(m)) = anode::decode(page.get_record(i).unwrap()) {
                out.push(m);
            }
        }
    }
    out
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

fn text(m: &MNode, k: &str) -> String {
    match m.fields.get(k) {
        Some(MValue::Text(t)) => t.clone(),
        other => panic!("{} = {:?}", k, other),
    }
}

fn int(m: &MNode, k: &str) -> i64 {
    match m.fields.get(k) {
        Some(MValue::Int(v)) => *v,
        other => panic!("{} = {:?}", k, other),
    }
}

fn float(m: &MNode, k: &str) -> f64 {
    match m.fields.get(k) {
        Some(MValue::Float(v)) => *v,
        other => panic!("{} = {:?}", k, other),
    }
}

#[test]
fn stratified_predicates_have_exactly_the_selectivity_they_claim() {
    let dir = tmp_dir();
    let slab = dir.path().join("metadata_content.slab");
    let rows = write_enriched_slab(&slab);
    write_survey(dir.path(), &slab);
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();

    let mut op = GenPredicatesOp;
    let mut ctx = test_ctx(dir.path());
    let r = op.execute(&options(), &mut ctx);
    assert_eq!(r.status, Status::Ok, "{}", r.message);
    let out = dir.path().join("profiles/base/predicates.slab");

    let preds = predicates(&out);
    let families = namespace_records(&out, Some("families"));
    let generation = namespace_records(&out, Some("generation"));
    assert!(!preds.is_empty());
    assert_eq!(
        families.len(),
        preds.len(),
        "one families record per predicate"
    );
    assert_eq!(
        generation.len(),
        preds.len(),
        "one generation record per predicate"
    );
    assert_eq!(
        namespace_records(&out, Some("schema")).len()
            + namespace_records(&out, Some("survey")).len(),
        0,
        "schema and survey are JSON, not MNodes"
    );
    assert_eq!(
        SlabReader::open_namespace(&out, Some("survey"))
            .unwrap()
            .total_records(),
        1
    );

    // Every predicate: exact count over the slab equals the expected
    // count the generator recorded, and the selectivity sits in the
    // half-decade band of the cell it filled.
    let mut per_family: HashMap<String, usize> = HashMap::new();
    for (i, p) in preds.iter().enumerate() {
        let realised = rows.iter().filter(|m| evaluate(p, m)).count() as i64;
        let expected = int(&generation[i], "expected_count");
        let family = text(&families[i], "family");
        assert_eq!(
            realised, expected,
            "predicate {} `{}` ({}) claims {} matches, has {}",
            i, p, family, expected, realised
        );
        let sel = float(&families[i], "selectivity");
        assert!(
            (sel - realised as f64 / N as f64).abs() < 1e-9,
            "{}: {} vs {}",
            p,
            sel,
            realised
        );
        let cell = text(&generation[i], "cell");
        let decade: i32 = cell.split(":1e").nth(1).unwrap().parse().unwrap();
        let lo = 10f64.powi(decade) / 10f64.sqrt();
        let hi = 10f64.powi(decade) * 10f64.sqrt();
        assert!(
            sel >= lo && sel < hi,
            "{}: selectivity {} outside band of {}",
            p,
            sel,
            cell
        );
        assert!(cell.starts_with(&family));
        assert_eq!(text(&generation[i], "vernacular"), p.to_string());
        *per_family.entry(family.clone()).or_insert(0) += 1;
        if family == "topical" {
            let level = int(&families[i], "topic_level");
            assert!(level == 1 || level == 2);
            let s = p.to_string();
            assert!(s.contains("topic_l1") || s.contains("topic_l2"), "{}", s);
        } else {
            assert!(!p.to_string().contains("topic_l"), "{} in {}", p, family);
        }
        if family != "control" {
            assert!(
                !p.to_string().contains("sample_bucket"),
                "the hash is the control family's alone: {}",
                p
            );
        }
    }
    for f in ["topical", "structural", "bibliographic", "control"] {
        assert!(
            per_family.get(f).copied().unwrap_or(0) > 0,
            "family {} empty: {:?}",
            f,
            per_family
        );
    }
    // Control: every cell full, since the hash fills any cell on demand.
    assert_eq!(per_family["control"], 4 + 6 + 8);

    // The report: one cell per family × decade, shortfalls consistent.
    let report: GenerationReport = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("profiles/base/predicates.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report.cells.len(), 12);
    assert_eq!(report.predicates, preds.len());
    assert_eq!(report.decades, vec![-1, -2, -3]);
    assert_eq!(report.per_cell_targets, vec![4, 6, 8]);
    assert_eq!(
        report.cells.iter().map(|c| c.drawn).sum::<usize>(),
        preds.len()
    );
    assert!(
        report
            .cells
            .iter()
            .all(|c| c.drawn + c.shortfall == c.target)
    );
    assert_eq!(report.floors.len(), 3);
    assert!(
        report.floors[0].reliable_at_base_count,
        "1e-1 at N=6000 clears M=5"
    );
    assert!(report.query_count.is_none());

    // Artifact check.
    let op = GenPredicatesOp;
    assert_eq!(op.check_artifact(&out, &options()), ArtifactState::Complete);
    assert_eq!(
        op.check_artifact(&dir.path().join("profiles/base/nowhere.slab"), &options()),
        ArtifactState::Absent
    );
    // The verified count the runner reads was set.
    assert_eq!(
        ctx.defaults
            .get("verified_count:predicates.slab")
            .map(String::as_str),
        Some(preds.len().to_string().as_str())
    );
}

#[test]
fn stratified_draw_is_reproducible_and_seed_sensitive() {
    let dir = tmp_dir();
    let slab = dir.path().join("metadata_content.slab");
    write_enriched_slab(&slab);
    write_survey(dir.path(), &slab);
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
    let out = dir.path().join("profiles/base/predicates.slab");
    let run = |seed: &str| -> Vec<String> {
        let mut op = GenPredicatesOp;
        let mut ctx = test_ctx(dir.path());
        let mut o = options();
        o.set("seed", seed);
        let r = op.execute(&o, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        predicates(&out).iter().map(|p| p.to_string()).collect()
    };
    let a = run("42");
    let b = run("42");
    let c = run("43");
    assert_eq!(a, b, "same seed, same predicate set, byte for byte");
    assert_ne!(a, c, "a different seed draws differently");
}

#[test]
fn stratified_refuses_a_survey_without_the_census() {
    let dir = tmp_dir();
    let slab = dir.path().join("metadata_content.slab");
    write_enriched_slab(&slab);
    let cfg = SurveyConfig {
        census: CensusConfig {
            auto: false,
            ..CensusConfig::default()
        },
        ..SurveyConfig::default()
    };
    let report = survey::survey(&slab, &cfg, None).unwrap();
    std::fs::write(
        dir.path().join("survey.json"),
        serde_json::to_string(&report).unwrap(),
    )
    .unwrap();
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
    let mut op = GenPredicatesOp;
    let mut ctx = test_ctx(dir.path());
    let r = op.execute(&options(), &mut ctx);
    assert_eq!(r.status, Status::Error);
    assert!(r.message.contains("census"), "{}", r.message);
}
