// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `transform enrich-metadata`.
//!
//! Builds a small corpus the way the passage pipeline would — a
//! metadata table with paper-level fields repeated per passage, the
//! passage and parent tables, a two-level topic assignment and a label
//! slab — with the metadata written in row groups that do not line up
//! with the passage table's, runs the command, and checks every
//! derived column row by row against what was planted. Also checks the
//! positional fallback without labels, the artifact check, and that
//! the run is byte-identical across thread counts.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{BooleanBuilder, Int32Builder, Int64Builder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use indexmap::IndexMap;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use slabtastic::{SlabWriter, WriterConfig};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::passage_table::{
    ParentRow, ParentTableWriter, PassageRow, PassageTableWriter, read_row_range,
};
use veks_core::ui::{TestSink, UiHandle};
use veks_pipeline::pipeline::command::{ArtifactState, CommandOp, Options, Status, StreamContext};
use veks_pipeline::pipeline::commands::transform_enrich_metadata::{
    DEFAULT_BUCKETS, EnrichMetadataOp, EnrichReport, sample_bucket,
};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;

fn tmp_dir() -> tempfile::TempDir {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

fn test_ctx(dir: &Path, threads: usize) -> StreamContext {
    StreamContext {
        dataset_name: String::new(),
        profile: String::new(),
        profile_names: vec![],
        workspace: dir.to_path_buf(),
        cache: dir.join(".cache"),
        defaults: IndexMap::new(),
        dry_run: false,
        progress: ProgressLog::new(),
        threads,
        step_id: String::new(),
        governor: ResourceGovernor::default_governor(),
        ui: UiHandle::new(Arc::new(TestSink::new())),
        status_interval: Duration::from_secs(1),
        estimated_total_steps: 0,
        provenance_selector: veks_pipeline::pipeline::provenance::ProvenanceFlags::STRICT,
    }
}

/// One planted passage.
#[derive(Clone)]
struct Planted {
    corpusid: i64,
    /// The upstream `ordinal`: section-local, restarting at zero in
    /// every section, as the real passage table has it (TS-174).
    ordinal: i32,
    /// The passage's index within its paper.
    index_in_paper: i32,
    passage_count: i32,
    year: i32,
    citations: i64,
    heading: &'static str,
    words: usize,
    l1: u16,
    l2: u16,
}

const HEADINGS: [&str; 5] = [
    "Introduction",
    "2. Materials and Methods",
    "Results and Discussion",
    "",
    "Acknowledgements",
];

/// `papers` papers of varying length, written in paper order.
fn plan(papers: usize) -> Vec<Planted> {
    let mut out = Vec::new();
    for p in 0..papers {
        let count = 1 + (p % 5) as i32; // 1..5 passages
        let year = 2018 + (p % 3) as i32;
        let citations = ((p * 7) % 11) as i64; // repeated values → ties
        for o in 0..count {
            let leaf = ((p + o as usize) % 6) as u16;
            out.push(Planted {
                corpusid: 1000 + p as i64,
                // Two passages per section: the section-local ordinal
                // repeats within a paper.
                ordinal: o % 2,
                index_in_paper: o,
                passage_count: count,
                year,
                citations,
                heading: HEADINGS[(p + o as usize) % HEADINGS.len()],
                words: 3 + ((p + o as usize) % 9),
                l1: leaf / 3,
                l2: leaf,
            });
        }
    }
    out
}

fn write_metadata(path: &Path, rows: &[Planted], group_rows: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("corpusid", DataType::Int64, false),
        Field::new("section", DataType::Utf8, false),
        Field::new("year", DataType::Int32, false),
        Field::new("citationcount", DataType::Int64, false),
        Field::new("isopenaccess", DataType::Boolean, false),
        Field::new("field", DataType::Utf8, false),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_size(group_rows)
        .build();
    let mut w = ArrowWriter::try_new(
        std::fs::File::create(path).unwrap(),
        schema.clone(),
        Some(props),
    )
    .unwrap();
    let mut corpusid = Int64Builder::new();
    let mut section = StringBuilder::new();
    let mut year = Int32Builder::new();
    let mut cits = Int64Builder::new();
    let mut oa = BooleanBuilder::new();
    let mut field = StringBuilder::new();
    for r in rows {
        corpusid.append_value(r.corpusid);
        section.append_value(r.heading);
        year.append_value(r.year);
        cits.append_value(r.citations);
        oa.append_value(r.corpusid % 2 == 0);
        field.append_value(if r.l1 == 0 { "Physics" } else { "Biology" });
    }
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(corpusid.finish()),
            Arc::new(section.finish()),
            Arc::new(year.finish()),
            Arc::new(cits.finish()),
            Arc::new(oa.finish()),
            Arc::new(field.finish()),
        ],
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn write_passages_and_parents(dir: &Path, rows: &[Planted]) {
    let mut passages = PassageTableWriter::create(&dir.join("passages.parquet")).unwrap();
    let mut parents = ParentTableWriter::create(&dir.join("parents.parquet")).unwrap();
    let mut last_paper: Option<i64> = None;
    for (i, r) in rows.iter().enumerate() {
        let text = (0..r.words)
            .map(|w| format!("w{}", w))
            .collect::<Vec<_>>()
            .join(" ");
        passages
            .push(&PassageRow {
                corpusid: r.corpusid,
                section: r.heading.to_string(),
                ordinal: r.ordinal,
                char_start: 0,
                char_end: text.len() as i64,
                text,
            })
            .unwrap();
        if last_paper != Some(r.corpusid) {
            parents
                .push(&ParentRow {
                    corpusid: r.corpusid,
                    passage_count: r.passage_count,
                    row_start: i as i64,
                })
                .unwrap();
            last_paper = Some(r.corpusid);
        }
    }
    passages.finish().unwrap();
    parents.finish().unwrap();
}

fn write_assignments(path: &Path, rows: &[Planted]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for r in rows {
        f.write_all(&2i32.to_le_bytes()).unwrap();
        f.write_all(&r.l1.to_le_bytes()).unwrap();
        f.write_all(&r.l2.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn label_of(level: usize, code: usize) -> String {
    match level {
        1 => ["physics", "biology"][code].to_string(),
        _ => [
            "optics-lasers",
            "plasma",
            "cosmology",
            "genomics",
            "ecology",
            "neuroscience",
        ][code]
            .to_string(),
    }
}

fn write_labels(path: &Path) {
    let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
    let mut w = SlabWriter::new(path, config).unwrap();
    for (level, n) in [(1usize, 2usize), (2, 6)] {
        for code in 0..n {
            let mut fields = IndexMap::new();
            fields.insert("level".to_string(), MValue::Int(level as i64));
            fields.insert("code".to_string(), MValue::Int(code as i64));
            fields.insert("label".to_string(), MValue::Text(label_of(level, code)));
            w.add_record(&anode::encode(&ANode::MNode(MNode { fields })))
                .unwrap();
        }
    }
    w.finish().unwrap();
}

fn options(with_labels: bool) -> Options {
    let mut o = Options::new();
    o.set("metadata", "metadata.parquet");
    o.set("passages", "passages.parquet");
    o.set("parents", "parents.parquet");
    o.set("assignments", "topic_assign.u16vecs");
    if with_labels {
        o.set("labels", "topic_labels.slab");
    }
    o.set("output", ".cache/metadata_enriched.parquet");
    o.set("seed", "42");
    o
}

fn fixture(dir: &Path, papers: usize, metadata_group_rows: usize) -> Vec<Planted> {
    let rows = plan(papers);
    write_metadata(&dir.join("metadata.parquet"), &rows, metadata_group_rows);
    write_passages_and_parents(dir, &rows);
    write_assignments(&dir.join("topic_assign.u16vecs"), &rows);
    write_labels(&dir.join("topic_labels.slab"));
    std::fs::create_dir_all(dir.join(".cache")).unwrap();
    rows
}

/// Every string of a utf8 column, by name, across batches.
fn strings(batches: &[RecordBatch], name: &str) -> Vec<String> {
    use arrow::array::{Array, StringArray};
    let mut out = Vec::new();
    for b in batches {
        let col = b.column(b.schema().index_of(name).unwrap());
        let a = col.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..a.len() {
            out.push(a.value(i).to_string());
        }
    }
    out
}

fn i16s(batches: &[RecordBatch], name: &str) -> Vec<i16> {
    use arrow::array::{Array, Int16Array};
    let mut out = Vec::new();
    for b in batches {
        let col = b.column(b.schema().index_of(name).unwrap());
        let a = col.as_any().downcast_ref::<Int16Array>().unwrap();
        for i in 0..a.len() {
            out.push(a.value(i));
        }
    }
    out
}

fn i32s(batches: &[RecordBatch], name: &str) -> Vec<i32> {
    use arrow::array::{Array, Int32Array};
    let mut out = Vec::new();
    for b in batches {
        let col = b.column(b.schema().index_of(name).unwrap());
        let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..a.len() {
            out.push(a.value(i));
        }
    }
    out
}

#[test]
fn enrichment_derives_every_column_from_what_was_planted() {
    let dir = tmp_dir();
    // 700 rows in metadata groups of 250 against one passage row
    // group: the ranges cross boundaries.
    let rows = fixture(dir.path(), 233, 250);
    let n = rows.len();

    let mut op = EnrichMetadataOp;
    let mut ctx = test_ctx(dir.path(), 4);
    let r = op.execute(&options(true), &mut ctx);
    assert_eq!(r.status, Status::Ok, "{}", r.message);
    assert!(
        r.message
            .contains(&format!("{} rows enriched with 7 columns", n)),
        "{}",
        r.message
    );

    let out = dir.path().join(".cache/metadata_enriched.parquet");
    let batches = read_row_range(&out, 0, n as u64).unwrap();
    let schema = batches[0].schema();
    for name in [
        "corpusid",
        "section",
        "year",
        "citationcount",
        "isopenaccess",
        "field",
    ] {
        assert!(schema.index_of(name).is_ok(), "source column {} kept", name);
    }
    let l1 = strings(&batches, "topic_l1");
    let l2 = strings(&batches, "topic_l2");
    let classes = strings(&batches, "section_class");
    let pct = i16s(&batches, "citation_percentile");
    let pos = i16s(&batches, "passage_position");
    let words = i16s(&batches, "word_count");
    let buckets = i32s(&batches, "sample_bucket");
    let original_field = strings(&batches, "field");

    // Citation percentiles: recompute the per-year midpoint ranks over
    // papers from the plan.
    let mut per_year: std::collections::HashMap<i32, Vec<i64>> = Default::default();
    let mut seen = std::collections::HashSet::new();
    for r in &rows {
        if seen.insert(r.corpusid) {
            per_year.entry(r.year).or_default().push(r.citations);
        }
    }
    let expected_pct = |year: i32, c: i64| -> i16 {
        let v = &per_year[&year];
        let below = v.iter().filter(|x| **x < c).count() as f64;
        let ties = v.iter().filter(|x| **x == c).count() as f64;
        ((100.0 * (below + (ties - 1.0) / 2.0) / v.len() as f64).floor() as i64).clamp(0, 99) as i16
    };

    for (i, r) in rows.iter().enumerate() {
        assert_eq!(l1[i], label_of(1, r.l1 as usize), "row {}", i);
        assert_eq!(l2[i], label_of(2, r.l2 as usize), "row {}", i);
        let expected_class = match r.heading {
            "Introduction" => "introduction",
            "2. Materials and Methods" => "methods",
            "Results and Discussion" => "discussion",
            _ => "other",
        };
        assert_eq!(
            classes[i], expected_class,
            "row {} heading {:?}",
            i, r.heading
        );
        assert_eq!(pct[i], expected_pct(r.year, r.citations), "row {}", i);
        assert_eq!(
            pos[i],
            ((100 * r.index_in_paper as i64) / r.passage_count as i64).min(99) as i16,
            "row {}: position is within the paper, not the section",
            i
        );
        assert_eq!(words[i], r.words as i16, "row {}", i);
        assert_eq!(
            buckets[i],
            sample_bucket(42, r.corpusid, i as u64, DEFAULT_BUCKETS) as i32,
            "row {}: keyed on the source row, not the section-local ordinal",
            i
        );
        assert_eq!(
            original_field[i],
            if r.l1 == 0 { "Physics" } else { "Biology" }
        );
    }

    // The heading table beside the output, and the report.
    let map = dir.path().join(".cache/section_class_map.slab");
    assert!(map.exists());
    let report: EnrichReport = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join(".cache/metadata_enriched.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report.rows, n as u64);
    assert_eq!(report.papers, 233);
    assert_eq!(report.years, 3);
    assert_eq!(report.distinct_headings, 5);
    assert!(!report.positional_labels);
    assert!(
        report.headings_other_share > 0.3 && report.headings_other_share < 0.5,
        "{}",
        report.headings_other_share
    );

    // Artifact check: Complete; Partial when the assignments would
    // demand a column the output lacks; Absent when missing.
    let op = EnrichMetadataOp;
    assert_eq!(
        op.check_artifact(&out, &options(true)),
        ArtifactState::Complete
    );
    let mut three = plan(5);
    for r in three.iter_mut() {
        r.l1 = 0;
    }
    let deeper = dir.path().join("deeper.u16vecs");
    {
        let mut f = std::fs::File::create(&deeper).unwrap();
        for _ in 0..n {
            f.write_all(&3i32.to_le_bytes()).unwrap();
            f.write_all(&[0u8; 6]).unwrap();
        }
    }
    let mut o = options(true);
    o.set("assignments", "deeper.u16vecs");
    assert_eq!(op.check_artifact(&out, &o), ArtifactState::Partial);
    assert_eq!(
        op.check_artifact(&dir.path().join(".cache/nowhere.parquet"), &options(true)),
        ArtifactState::Absent
    );
}

#[test]
fn enrichment_without_labels_writes_positional_labels() {
    let dir = tmp_dir();
    let rows = fixture(dir.path(), 40, 1000);
    let mut op = EnrichMetadataOp;
    let mut ctx = test_ctx(dir.path(), 2);
    let r = op.execute(&options(false), &mut ctx);
    assert_eq!(r.status, Status::Ok, "{}", r.message);
    let batches = read_row_range(
        &dir.path().join(".cache/metadata_enriched.parquet"),
        0,
        rows.len() as u64,
    )
    .unwrap();
    let l2 = strings(&batches, "topic_l2");
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(l2[i], format!("l2-{:05}", r.l2));
    }
    let report: EnrichReport = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join(".cache/metadata_enriched.json")).unwrap(),
    )
    .unwrap();
    assert!(report.positional_labels);
}

#[test]
fn enrichment_is_identical_across_thread_counts() {
    let mut outputs = Vec::new();
    for threads in [1usize, 7] {
        let dir = tmp_dir();
        fixture(dir.path(), 300, 128);
        let mut op = EnrichMetadataOp;
        let mut ctx = test_ctx(dir.path(), threads);
        let r = op.execute(&options(true), &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        outputs.push(std::fs::read(dir.path().join(".cache/metadata_enriched.parquet")).unwrap());
    }
    assert_eq!(outputs[0], outputs[1]);
}
