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
use std::path::{Path, PathBuf};
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
use veks_pipeline::pipeline::commands::gen_predicates_stratified::{Family, GenerationReport};
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
    o.set("count", COUNT.to_string());
    o
}

/// Records written: one per query ordinal (TS-156). With no queries
/// given, `count` says how many.
const COUNT: usize = 300;

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
    assert_eq!(preds.len(), COUNT, "one predicate per query ordinal");
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
        assert!(int(&families[i], "predicate") >= 0);
        *per_family.entry(family.clone()).or_insert(0) += 1;
        if family == "topical" {
            let level = int(&families[i], "topic_level");
            assert!(level == 1 || level == 2);
            let s = p.to_string();
            assert!(s.contains("topic_l1") || s.contains("topic_l2"), "{}", s);
            assert!(s.contains(&text(&families[i], "topic")), "{} names {}", s, text(&families[i], "topic"));
            assert!(families[i].fields.get("query_placement").is_none(), "no queries, no placement");
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
    // The report: one cell per family × decade; the slots are the
    // family's quarter of the count split 4:6:8; every slot is either
    // filled by its cell or backfilled from the control family, so the
    // records add up to the count exactly.
    let report: GenerationReport = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("profiles/base/predicates.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report.cells.len(), 12);
    assert_eq!(report.predicates, preds.len());
    assert_eq!(report.decades, vec![-1, -2, -3]);
    assert_eq!(report.per_cell, vec!["4", "6", "8"]);
    assert_eq!(report.slots_per_cell, [[17, 25, 33]; 4].concat());
    assert_eq!(report.slots_per_cell.iter().sum::<usize>(), COUNT);
    assert!(report.cells.iter().all(|c| c.filled + c.shortfall == c.target));
    assert!(report.cells.iter().all(|c| c.drawn <= c.filled));
    assert_eq!(
        report.cells.iter().map(|c| c.filled).sum::<usize>() + report.backfilled,
        COUNT
    );
    assert_eq!(
        report.cells.iter().map(|c| c.shortfall).sum::<usize>(),
        report.backfilled
    );
    // Control cells never fall short: the hash fills any cell on demand.
    assert!(report.cells.iter().filter(|c| c.family == Family::Control).all(|c| c.shortfall == 0));
    assert!(report.distinct_predicates >= report.cells.iter().map(|c| c.drawn).sum::<usize>());
    let backfilled = generation
        .iter()
        .filter(|g| matches!(g.fields.get("backfill"), Some(MValue::Bool(true))))
        .count();
    assert_eq!(backfilled, report.backfilled);
    // A distinct predicate repeats only where its cell's pool was
    // smaller than its slots.
    let mut by_id: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, f) in families.iter().enumerate() {
        by_id.entry(int(f, "predicate")).or_default().push(i);
    }
    assert_eq!(by_id.len(), report.distinct_predicates);
    for idx in by_id.values() {
        if matches!(generation[idx[0]].fields.get("backfill"), Some(MValue::Bool(true))) {
            continue;
        }
        let cell = text(&generation[idx[0]], "cell");
        let c = report.cells.iter().find(|c| format!("{}:1e{}", c.family.as_str(), c.decade) == cell).unwrap();
        assert!(idx.len() == 1 || c.drawn < c.target, "{} repeated {} times in a cell with {} distinct for {} slots", preds[idx[0]], idx.len(), c.drawn, c.target);
    }
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

/// A two-level topic model whose leaves are one-hot directions, so a
/// query built from a leaf's centroid descends to that leaf without
/// doubt. Levels [2, 3]: parents `physics`, `biology`; six leaves named
/// as the fixture's `topic_l2` values, in the same order.
fn write_topic_model(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    use veks_pipeline::pipeline::commands::compute_topics::{LevelReport, SampleOrder, TopicModelReport};
    const DIM: usize = 8;
    // Level 1: e0, e1. Level 2 under parent p: e_p + e_{2+k} for k in 0..3.
    let mut centroids: Vec<[f32; DIM]> = Vec::new();
    for p in 0..2 {
        let mut c = [0f32; DIM];
        c[p] = 1.0;
        centroids.push(c);
    }
    for p in 0..2 {
        for k in 0..3 {
            let mut c = [0f32; DIM];
            c[p] = 1.0;
            c[2 + k] = 1.0;
            centroids.push(c);
        }
    }
    let write_fvecs = |path: &Path, rows: &[[f32; DIM]]| {
        let mut bytes = Vec::with_capacity(rows.len() * (4 + DIM * 4));
        for r in rows {
            bytes.extend_from_slice(&(DIM as u32).to_le_bytes());
            for v in r {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    };
    let centroids_path = dir.join("profiles/base/topic_centroids.fvecs");
    write_fvecs(&centroids_path, &centroids);
    let model = TopicModelReport {
        schema_version: 1,
        dim: DIM,
        levels: vec![2, 3],
        total_centroids: 8,
        sample: "synthetic".into(),
        sample_size: 0,
        sample_order: SampleOrder::Prefix,
        seed: 0,
        iterations: 0,
        tolerance: 0.0,
        normalize: true,
        kernel: "test".into(),
        fit_seconds: 0.0,
        per_level: vec![
            LevelReport { branching: 2, clusters: 2, empty: 0, runs: 1, converged: 1, max_final_movement: 0.0, repairs: 0 },
            LevelReport { branching: 3, clusters: 6, empty: 0, runs: 2, converged: 2, max_final_movement: 0.0, repairs: 0 },
        ],
        assignment: None,
    };
    let model_path = dir.join("profiles/base/topic_centroids.json");
    std::fs::write(&model_path, serde_json::to_string_pretty(&model).unwrap()).unwrap();
    // Labels: level 1 codes 0,1; level 2 codes 0..6 in leaf order.
    let labels_path = dir.join("profiles/base/topic_labels.slab");
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(&labels_path, config).unwrap();
    let l1 = ["physics", "biology"];
    let l2 = ["optics-lasers", "plasma", "cosmology", "genomics", "ecology", "neuroscience"];
    let mut add = |level: i64, code: i64, label: &str| {
        let mut fields = IndexMap::new();
        fields.insert("level".to_string(), MValue::Int(level));
        fields.insert("code".to_string(), MValue::Int(code));
        fields.insert("label".to_string(), MValue::Text(label.to_string()));
        fields.insert("terms".to_string(), MValue::Text(label.to_string()));
        fields.insert("sample_size".to_string(), MValue::Int(100));
        fields.insert("positional".to_string(), MValue::Bool(false));
        w.add_record(&anode::encode(&ANode::MNode(MNode { fields }))).unwrap();
    };
    for (i, l) in l1.iter().enumerate() {
        add(1, i as i64, l);
    }
    for (i, l) in l2.iter().enumerate() {
        add(2, i as i64, l);
    }
    w.finish().unwrap();
    // Queries: 60, query i sits in leaf i % 6 (its centroid, slightly
    // perturbed so no two are identical).
    let mut queries: Vec<[f32; DIM]> = Vec::new();
    for i in 0..60 {
        let mut q = centroids[2 + (i % 6)];
        q[7] = 0.01 * (i as f32 / 60.0);
        queries.push(q);
    }
    let queries_path = dir.join("profiles/base/query_vectors.fvecs");
    write_fvecs(&queries_path, &queries);
    (centroids_path, model_path, labels_path)
}

/// The queries' own metadata rows in query order (TS-165): query i is
/// given base row `i * 97 % N`, so its section, year and topic labels
/// are real rows of the fixture.
fn write_query_metadata(dir: &Path, rows: &[MNode], count: usize) -> Vec<MNode> {
    let path = dir.join("profiles/base/query_metadata.slab");
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(&path, config).unwrap();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let row = rows[(i * 97) % rows.len()].clone();
        w.add_record(&anode::encode(&ANode::MNode(row.clone()))).unwrap();
        out.push(row);
    }
    w.finish().unwrap();
    out
}

/// With the queries given, record i is query i's predicate and every
/// topical pair's placement is what the query's own descent says:
/// an in-topic pair's query lies in the predicate's topic, an
/// out-of-topic pair's does not, and the query's own topic at that
/// level is recorded beside it (TS-19, TS-157).
#[test]
fn placement_is_decided_from_each_querys_own_descent() {
    let dir = tmp_dir();
    let slab = dir.path().join("metadata_content.slab");
    let rows = write_enriched_slab(&slab);
    write_survey(dir.path(), &slab);
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
    write_topic_model(dir.path());
    let query_rows = write_query_metadata(dir.path(), &rows, 60);
    let mut o = options();
    o.0.shift_remove("count");
    o.set("queries", "profiles/base/query_vectors.fvecs");
    o.set("centroids", "profiles/base/topic_centroids.fvecs");
    o.set("model", "profiles/base/topic_centroids.json");
    o.set("labels", "profiles/base/topic_labels.slab");
    o.set("query-placement", "mixed");
    o.set("query-metadata", "profiles/base/query_metadata.slab");
    let mut op = GenPredicatesOp;
    let mut ctx = test_ctx(dir.path());
    let r = op.execute(&o, &mut ctx);
    assert_eq!(r.status, Status::Ok, "{}", r.message);
    let out = dir.path().join("profiles/base/predicates.slab");
    let preds = predicates(&out);
    let families = namespace_records(&out, Some("families"));
    assert_eq!(preds.len(), 60, "one predicate per query");
    let l1 = ["physics", "biology"];
    let l2 = ["optics-lasers", "plasma", "cosmology", "genomics", "ecology", "neuroscience"];
    let (mut ins, mut outs) = (0, 0);
    for (q, f) in families.iter().enumerate() {
        if text(f, "family") != "topical" {
            assert!(f.fields.get("query_placement").is_none());
            continue;
        }
        let leaf = q % 6;
        let level = int(f, "topic_level") as usize;
        let query_label = if level == 1 { l1[leaf / 3] } else { l2[leaf] };
        assert_eq!(text(f, "query_topic"), query_label, "query {q} at level {level}");
        let topic = text(f, "topic");
        match text(f, "query_placement").as_str() {
            "in-topic" => {
                assert_eq!(topic, query_label, "query {q}: {}", preds[q]);
                ins += 1;
            }
            "out-of-topic" => {
                assert_ne!(topic, query_label, "query {q}: {}", preds[q]);
                outs += 1;
            }
            other => panic!("query {q}: placement {other}"),
        }
    }
    assert!(ins > 0 && outs > 0, "mixed placement gives both kinds: {ins} in, {outs} out");
    // Every pair is labelled against its query's own row (TS-166), for
    // every family, and the label is what evaluating the predicate on
    // that row says.
    let (mut in_filter, mut out_of_filter) = (0, 0);
    for (q, f) in families.iter().enumerate() {
        let recorded = match f.fields.get("query_in_filter") {
            Some(MValue::Bool(b)) => *b,
            other => panic!("query {q}: query_in_filter = {other:?}"),
        };
        assert_eq!(recorded, evaluate(&preds[q], &query_rows[q]), "query {q}: {}", preds[q]);
        if recorded { in_filter += 1 } else { out_of_filter += 1 }
    }
    let report: GenerationReport = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("profiles/base/predicates.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report.in_filter, Some(in_filter));
    assert_eq!(report.out_of_filter, Some(out_of_filter));
    assert_eq!(report.cells.iter().map(|c| c.in_filter + c.out_of_filter).sum::<usize>(), 60);
    assert_eq!(report.query_count, Some(60));
    assert_eq!(report.predicates, 60);
    assert_eq!(report.cells.iter().map(|c| c.in_topic).sum::<usize>(), ins);
    assert_eq!(report.cells.iter().map(|c| c.out_of_topic).sum::<usize>(), outs);
    // The artifact check reads the expected count from the queries.
    let op = GenPredicatesOp;
    assert_eq!(op.check_artifact(&out, &o), ArtifactState::Complete);
    // A count that disagrees with the queries is refused.
    let mut bad = o.clone();
    bad.set("count", "59");
    let mut op = GenPredicatesOp;
    let r = op.execute(&bad, &mut test_ctx(dir.path()));
    assert_eq!(r.status, Status::Error);
    assert!(r.message.contains("59"), "{}", r.message);
}

/// Query metadata with a row count other than the query count is a
/// misalignment, and refused.
#[test]
fn query_metadata_must_have_one_row_per_query() {
    let dir = tmp_dir();
    let slab = dir.path().join("metadata_content.slab");
    let rows = write_enriched_slab(&slab);
    write_survey(dir.path(), &slab);
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
    write_query_metadata(dir.path(), &rows, 59);
    let mut o = options();
    o.set("query-metadata", "profiles/base/query_metadata.slab");
    let mut op = GenPredicatesOp;
    let r = op.execute(&o, &mut test_ctx(dir.path()));
    assert_eq!(r.status, Status::Error);
    assert!(r.message.contains("59") && r.message.contains("300"), "{}", r.message);
}

/// Without queries and without `count` there is nothing to pair with.
#[test]
fn stratified_needs_queries_or_a_count() {
    let dir = tmp_dir();
    let slab = dir.path().join("metadata_content.slab");
    write_enriched_slab(&slab);
    write_survey(dir.path(), &slab);
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
    let mut o = options();
    o.0.shift_remove("count");
    let mut op = GenPredicatesOp;
    let r = op.execute(&o, &mut test_ctx(dir.path()));
    assert_eq!(r.status, Status::Error);
    assert!(r.message.contains("count"), "{}", r.message);
}
