// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end: the topic-stratified predicate pipeline on a planted
//! corpus, through the real binary (TS-120).
//!
//! A corpus of 200 passages in 50 papers is planted around six leaf
//! directions with matching vocabulary and metadata. The dataset is
//! bootstrapped by `veks prepare bootstrap` and then extended exactly
//! as tessera's definition is: topic fit and labels, enrichment, the
//! query metadata facet, the margin adjunct, a censused survey, the
//! stratified predicate set with per-pair placement and labels,
//! evaluation, filtered ground truth, and the strata verification.
//! Every artifact the design registers is then checked.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{BooleanBuilder, Int32Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_yaml::Value;
use slabtastic::SlabReader;
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::passage_table::{ParentRow, ParentTableWriter, PassageRow, PassageTableWriter};

const ROWS: usize = 200;
const PER_PAPER: usize = 4;
const DIM: usize = 8;
const QUERIES: usize = 10;

fn veks_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_veks"));
    if !path.exists() {
        path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/debug/veks");
    }
    path
}

fn make_tempdir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// Leaf of row `i`: every passage of a paper shares its paper's leaf.
fn leaf_of(i: usize) -> usize {
    (i / PER_PAPER) % 6
}

/// Planted vectors: leaf `l` sits at e_{l/3} + e_{2 + l%3}, with a
/// small deterministic wobble so no two rows are identical.
fn write_vectors(path: &Path) {
    let mut bytes = Vec::with_capacity(ROWS * (4 + DIM * 4));
    for i in 0..ROWS {
        let l = leaf_of(i);
        let mut v = [0f32; DIM];
        v[l / 3] = 1.0;
        v[2 + l % 3] = 1.0;
        v[5] = 0.03 * ((i * 7 % 11) as f32 - 5.0);
        v[6] = 0.03 * ((i * 13 % 7) as f32 - 3.0);
        bytes.extend_from_slice(&(DIM as u32).to_le_bytes());
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

const HEADINGS: [&str; 4] = ["Introduction", "Methods", "Results", "Discussion"];
const VOCAB: [&str; 6] = [
    "laser optics photon",
    "plasma fusion tokamak",
    "galaxy cosmology redshift",
    "genome sequencing allele",
    "habitat species ecosystem",
    "neuron synapse cortex",
];

fn paper_of(i: usize) -> i64 {
    (i / PER_PAPER) as i64
}

/// Metadata in the source shape tessera has: one row per passage,
/// paper-level fields repeated across a paper's passages.
fn write_metadata(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("corpusid", DataType::Int64, false),
        Field::new("section", DataType::Utf8, false),
        Field::new("year", DataType::Int32, false),
        Field::new("citationcount", DataType::Int64, false),
        Field::new("isopenaccess", DataType::Boolean, false),
        Field::new("field", DataType::Utf8, false),
        Field::new("venue", DataType::Utf8, false),
    ]));
    let mut corpusid = Int64Builder::new();
    let mut section = StringBuilder::new();
    let mut year = Int32Builder::new();
    let mut cits = Int64Builder::new();
    let mut oa = BooleanBuilder::new();
    let mut field = StringBuilder::new();
    let mut venue = StringBuilder::new();
    for i in 0..ROWS {
        let p = paper_of(i);
        corpusid.append_value(p);
        section.append_value(HEADINGS[i % PER_PAPER]);
        year.append_value(2010 + (p % 12) as i32);
        cits.append_value((p * 7) % 50);
        oa.append_value(p % 3 == 0);
        field.append_value(if leaf_of(i) < 3 { "Physics" } else { "Biology" });
        venue.append_value(format!("venue-{}", p % 5));
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(corpusid.finish()),
            Arc::new(section.finish()),
            Arc::new(year.finish()),
            Arc::new(cits.finish()),
            Arc::new(oa.finish()),
            Arc::new(field.finish()),
            Arc::new(venue.finish()),
        ],
    )
    .unwrap();
    let mut w = ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

/// Passages with leaf-specific vocabulary, and the parent table.
fn write_passages_and_parents(dir: &Path) {
    let mut passages = PassageTableWriter::create(&dir.join("passages.parquet")).unwrap();
    let mut parents = ParentTableWriter::create(&dir.join("parents.parquet")).unwrap();
    for i in 0..ROWS {
        let words = 40 + (i * 17) % 60;
        let text = (0..words)
            .map(|w| {
                if w % 3 == 0 {
                    VOCAB[leaf_of(i)].split(' ').nth(w % 3).unwrap().to_string()
                } else {
                    format!("w{}", w % 17)
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        passages
            .push(&PassageRow {
                corpusid: paper_of(i),
                section: HEADINGS[i % PER_PAPER].to_string(),
                ordinal: (i % PER_PAPER) as i32,
                char_start: 0,
                char_end: text.len() as i64,
                text,
            })
            .unwrap();
        if i % PER_PAPER == 0 {
            parents
                .push(&ParentRow {
                    corpusid: paper_of(i),
                    passage_count: PER_PAPER as i32,
                    row_start: i as i64,
                })
                .unwrap();
        }
    }
    passages.finish().unwrap();
    parents.finish().unwrap();
}

fn bootstrap(src: &Path, dataset: &Path) {
    let out = Command::new(veks_bin())
        .args(["prepare", "bootstrap", "--name", "e2e-topics", "--force"])
        .arg("--output").arg(dataset)
        .arg("--base-vectors").arg(src.join("base.fvecs"))
        .arg("--metadata").arg(src.join("metadata.parquet"))
        .args(["--self-search", "--query-count", &QUERIES.to_string(), "--neighbors", "5", "--seed", "42",
               "--no-dedup", "--no-zero-check", "--round-digits", "10"])
        .output()
        .expect("failed to execute veks prepare bootstrap");
    assert!(out.status.success(), "bootstrap failed:\n{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}

fn step(pairs: &[(&str, Value)]) -> Value {
    let mut m = serde_yaml::Mapping::new();
    for (k, v) in pairs {
        m.insert(Value::String(k.to_string()), v.clone());
    }
    Value::Mapping(m)
}

fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

fn list(items: &[&str]) -> Value {
    Value::Sequence(items.iter().map(|i| s(i)).collect())
}

/// Extend the bootstrapped definition the way tessera's is extended.
fn extend_definition(dataset_yaml: &Path, src_rel: &str) {
    let text = std::fs::read_to_string(dataset_yaml).unwrap();
    let mut doc: Value = serde_yaml::from_str(&text).unwrap();
    let steps = doc["upstream"]["steps"].as_sequence_mut().unwrap();
    let idx = |steps: &Vec<Value>, id: &str| steps.iter().position(|st| st["id"] == id).unwrap_or_else(|| panic!("no step {id}"));
    let passages = format!("{src_rel}/passages.parquet");
    let parents = format!("{src_rel}/parents.parquet");
    let metadata = format!("{src_rel}/metadata.parquet");
    let source_vectors = steps[idx(steps, "extract-base")]["source"].as_str().unwrap().to_string();

    let new_steps = vec![
        step(&[("id", s("compute-topics")), ("run", s("compute topics")), ("after", list(&["extract-base"])),
              ("base", s(&source_vectors)), ("sample", s("profiles/base/base_vectors.fvecs")), ("sample-order", s("prefix")),
              ("sample-size", Value::Number(190.into())), ("levels", s("2,3")), ("seed", s("${seed}")),
              ("centroids", s("profiles/base/topic_centroids.fvecs")), ("output", s("${cache}/topic_assign.u16vecs")),
              ("margin", s("${cache}/topic_margin_all.mvecs"))]),
        step(&[("id", s("compute-topic-labels")), ("run", s("compute topic-labels")), ("after", list(&["compute-topics"])),
              ("passages", s(&passages)), ("assignments", s("${cache}/topic_assign.u16vecs")),
              ("model", s("profiles/base/topic_centroids.json")), ("row-groups", Value::Number(1.into())),
              ("sample-per-cluster", Value::Number(100.into())), ("min-sample", Value::Number(2.into())),
              ("top-terms", Value::Number(2.into())), ("seed", s("${seed}")), ("output", s("profiles/base/topic_labels.slab"))]),
        step(&[("id", s("enrich-metadata")), ("run", s("transform enrich-metadata")), ("after", list(&["compute-topic-labels"])),
              ("metadata", s(&metadata)), ("passages", s(&passages)), ("parents", s(&parents)),
              ("assignments", s("${cache}/topic_assign.u16vecs")), ("labels", s("profiles/base/topic_labels.slab")),
              ("seed", s("${seed}")), ("buckets", Value::Number(4096.into())),
              ("output", s("${cache}/metadata_enriched.parquet")), ("section-map-out", s("profiles/base/section_class_map.slab"))]),
    ];
    let at = idx(steps, "count-base") + 1;
    for (k, st) in new_steps.into_iter().enumerate() {
        steps.insert(at + k, st);
    }
    // convert-metadata reads the enriched table.
    let i = idx(steps, "convert-metadata");
    steps[i]["source"] = s("${cache}/metadata_enriched.parquet");
    steps[i]["after"] = list(&["enrich-metadata"]);
    // The query metadata facet and the margin adjunct, after extract-metadata.
    let i = idx(steps, "extract-metadata") + 1;
    steps.insert(i, step(&[("id", s("extract-query-metadata")), ("run", s("transform extract")),
        ("after", list(&["convert-metadata", "generate-shuffle"])), ("source", s("${cache}/metadata_all.slab")),
        ("ivec-file", s("${cache}/shuffle.ivecs")), ("output", s("profiles/base/query_metadata.slab")),
        ("range", s("[0,${query_count})"))]));
    steps.insert(i + 1, step(&[("id", s("extract-topic-margin")), ("run", s("transform extract")),
        ("after", list(&["compute-topics", "generate-shuffle"])), ("source", s("${cache}/topic_margin_all.mvecs")),
        ("ivec-file", s("${cache}/shuffle.ivecs")), ("output", s("profiles/base/topic_margin.mvecs")),
        ("range", s("[${query_count},${vector_count})"))]));
    // The survey censuses the base facet.
    let i = idx(steps, "survey-metadata");
    steps[i]["after"] = list(&["extract-metadata"]);
    steps[i]["source"] = s("profiles/base/metadata_content.slab");
    steps[i]["samples"] = Value::Number(1000.into());
    steps[i]["census"] = s("auto,topic_l2");
    steps[i]["hierarchy"] = s("topic_l1>topic_l2");
    steps[i]["census-pair"] = s("topic_l1:year,topic_l2:isopenaccess,topic_l1:citation_percentile");
    // The stratified predicate set, one per query, placed and labelled.
    let i = idx(steps, "generate-predicates");
    let m = steps[i].as_mapping_mut().unwrap();
    for k in ["count", "source", "selectivity"] {
        m.remove(Value::String(k.into()));
    }
    for (k, v) in [
        ("after", list(&["survey-metadata", "count-base", "compute-topic-labels", "extract-query-metadata"])),
        ("strategy", s("stratified")), ("base-count", s("${base_count}")), ("decades", s("1e-1..1e-2")),
        ("per-cell", s("1,rest")), ("buckets", Value::Number(4096.into())), ("min-matches", Value::Number(2.into())),
        ("reliability-threshold", Value::Number(100.into())), ("queries", s("profiles/base/query_vectors.fvecs")),
        ("centroids", s("profiles/base/topic_centroids.fvecs")), ("model", s("profiles/base/topic_centroids.json")),
        ("labels", s("profiles/base/topic_labels.slab")), ("query-metadata", s("profiles/base/query_metadata.slab")),
    ] {
        m.insert(Value::String(k.into()), v);
    }
    // The strata verification gates the publish steps.
    let i = idx(steps, "verify-predicates") + 1;
    steps.insert(i, step(&[("id", s("verify-predicate-strata")), ("run", s("verify predicate-strata")),
        ("after", list(&["evaluate-predicates"])), ("predicates", s("profiles/base/predicates.slab")),
        ("queries", s("profiles/base/query_vectors.fvecs")), ("query-metadata", s("profiles/base/query_metadata.slab")),
        ("reliability-threshold", Value::Number(100.into())), ("output", s("${cache}/verify_predicate_strata.json"))]));
    let i = idx(steps, "generate-dataset-json");
    let after = steps[i]["after"].as_sequence_mut().unwrap();
    after.push(s("verify-predicate-strata"));
    std::fs::write(dataset_yaml, serde_yaml::to_string(&doc).unwrap()).unwrap();
}

fn run_pipeline(dataset_yaml: &Path) -> (bool, String) {
    let output = Command::new(veks_bin())
        .arg("run").args(["--output", "batch", "--threads", "2"]).arg(dataset_yaml)
        .output()
        .expect("failed to execute veks");
    let text = format!("{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

fn records(path: &Path, namespace: Option<&str>) -> Vec<Vec<u8>> {
    let reader = match namespace {
        Some(ns) => SlabReader::open_namespace(path, Some(ns)).unwrap(),
        None => SlabReader::open(path).unwrap(),
    };
    let mut out = Vec::new();
    for entry in reader.page_entries() {
        let page = reader.read_data_page(&entry).unwrap();
        for i in 0..page.record_count() {
            out.push(page.get_record(i).unwrap().to_vec());
        }
    }
    out
}

fn mnodes(path: &Path, namespace: Option<&str>) -> Vec<MNode> {
    records(path, namespace)
        .iter()
        .map(|b| match anode::decode(b) {
            Ok(ANode::MNode(m)) => m,
            other => panic!("{}: not an MNode: {:?}", path.display(), other.map(|_| ())),
        })
        .collect()
}

#[test]
fn e2e_topic_stratified_predicates() {
    let tmp = make_tempdir();
    let root = tmp.path().to_path_buf();
    // `E2E_KEEP=1` leaves the dataset behind for inspection.
    if std::env::var_os("E2E_KEEP").is_some() {
        eprintln!("keeping {}", root.display());
        std::mem::forget(tmp);
    }
    let src = root.join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_vectors(&src.join("base.fvecs"));
    write_metadata(&src.join("metadata.parquet"));
    write_passages_and_parents(&src);
    let dataset = root.join("dataset");
    bootstrap(&src, &dataset);
    let dataset_yaml = dataset.join("dataset.yaml");
    extend_definition(&dataset_yaml, "../src");

    let (ok, output) = run_pipeline(&dataset_yaml);
    assert!(ok, "pipeline failed:\n{}", output);

    // Retained adjuncts of the design (TS-84, TS-86).
    for f in [
        "profiles/base/topic_centroids.fvecs", "profiles/base/topic_centroids.json", "profiles/base/topic_labels.slab",
        "profiles/base/topic_margin.mvecs", "profiles/base/section_class_map.slab", "profiles/base/query_metadata.slab",
        "profiles/base/predicates.slab", "profiles/base/predicates.json", "profiles/default/metadata_results.slab",
        "profiles/default/prefiltered_neighbor_indices.ivec", "profiles/default/postfiltered_neighbor_indices.ivec",
        ".cache/verify_predicate_strata.json",
    ] {
        assert!(dataset.join(f).exists(), "{f} not produced\n{output}");
    }

    // The enriched M facet carries every derived column (TS-120).
    let base_rows = mnodes(&dataset.join("profiles/base/metadata_content.slab"), None);
    assert_eq!(base_rows.len(), ROWS - QUERIES);
    for col in ["corpusid", "section", "year", "citationcount", "isopenaccess", "field", "venue",
                "topic_l1", "topic_l2", "section_class", "citation_percentile", "passage_position", "word_count", "sample_bucket"] {
        assert!(base_rows[0].fields.contains_key(col), "missing {col}: {:?}", base_rows[0].fields.keys().collect::<Vec<_>>());
    }
    assert!(matches!(base_rows[0].fields.get("citation_percentile"), Some(MValue::Short(_))));
    assert!(matches!(base_rows[0].fields.get("sample_bucket"), Some(MValue::Int32(_))));

    // The queries' own rows: one per query, same columns (TS-165).
    let query_rows = mnodes(&dataset.join("profiles/base/query_metadata.slab"), None);
    assert_eq!(query_rows.len(), QUERIES);
    assert!(query_rows[0].fields.contains_key("topic_l2"));

    // One predicate per query, both namespaces, every pair labelled (TS-156, TS-166).
    let preds = dataset.join("profiles/base/predicates.slab");
    assert_eq!(records(&preds, None).len(), QUERIES);
    let families = mnodes(&preds, Some("families"));
    let generation = mnodes(&preds, Some("generation"));
    assert_eq!(families.len(), QUERIES);
    assert_eq!(generation.len(), QUERIES);
    let mut seen_families = std::collections::HashSet::new();
    for f in &families {
        assert!(matches!(f.fields.get("query_in_filter"), Some(MValue::Bool(_))), "{:?}", f.fields);
        let family = match f.fields.get("family") { Some(MValue::Text(t)) => t.clone(), other => panic!("{other:?}") };
        if family == "topical" {
            assert!(matches!(f.fields.get("query_placement"), Some(MValue::Text(_))));
            assert!(matches!(f.fields.get("query_topic"), Some(MValue::Text(_))));
        }
        seen_families.insert(family);
    }
    assert!(seen_families.len() >= 3, "families drawn: {seen_families:?}");

    // The answer keys: one results record per query; filtered facets sized to the queries.
    assert_eq!(records(&dataset.join("profiles/default/metadata_results.slab"), None).len(), QUERIES);

    // The strata verification passed with the labels re-derived.
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dataset.join(".cache/verify_predicate_strata.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["violations"], 0, "{report}");
    assert_eq!(report["predicates"], QUERIES);
    assert_eq!(report["label_checks"], QUERIES);
    assert_eq!(report["label_disagreements"], 0);

    // The dataset's own checks pass (merkle included, since the run finished).
    let check = Command::new(veks_bin()).args(["check"]).arg(&dataset).output().unwrap();
    let text = String::from_utf8_lossy(&check.stdout).to_string() + &String::from_utf8_lossy(&check.stderr);
    assert!(check.status.success(), "veks check failed:\n{text}");
}
