// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `compute topic-labels`.
//!
//! Builds a passage table whose texts are drawn from planted
//! vocabularies — two families of three leaves, each leaf with words
//! of its own on top of its family's — plus the matching assignments
//! and model report, runs the command, and checks that every cluster's
//! label is built from its own vocabulary, that labels are unique per
//! level, that the artifact check judges the slab against the model,
//! and that the run is byte-identical across thread counts.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use veks_core::formats::passage_table::{PassageRow, PassageTableWriter};
use veks_core::ui::{TestSink, UiHandle};
use veks_pipeline::pipeline::command::{ArtifactState, CommandOp, Options, Status, StreamContext};
use veks_pipeline::pipeline::commands::compute_topic_labels::{
    ComputeTopicLabelsOp, TopicLabelsReport,
};
use veks_pipeline::pipeline::commands::compute_topics::{
    LevelReport, SampleOrder, TopicModelReport,
};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;

const FAMILY_WORDS: [[&str; 3]; 2] = [
    ["solar", "photovoltaic", "irradiance"],
    ["neural", "network", "gradient"],
];
const LEAF_WORDS: [[&str; 2]; 6] = [
    ["inverter", "panel"],
    ["battery", "storage"],
    ["grid", "transmission"],
    ["convolution", "kernel"],
    ["recurrent", "sequence"],
    ["attention", "transformer"],
];
const FILLER: [&str; 8] = [
    "the", "analysis", "results", "also", "using", "data", "method", "values",
];

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

/// Passages, assignments and a model report for `per` rows of each of
/// the six planted leaves, interleaved. Returns each row's leaf.
fn write_fixture(dir: &Path, per: usize) -> Vec<usize> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
    let mut passages = PassageTableWriter::create(&dir.join("passages.parquet")).unwrap();
    let mut assign =
        std::io::BufWriter::new(std::fs::File::create(dir.join("topic_assign.u16vecs")).unwrap());
    let mut leaves = Vec::new();
    for i in 0..per * 6 {
        let leaf = i % 6;
        let fam = leaf / 3;
        let mut words: Vec<&str> = Vec::new();
        for _ in 0..8 {
            words.push(FAMILY_WORDS[fam][rng.random_range(0..3)]);
        }
        for _ in 0..8 {
            words.push(LEAF_WORDS[leaf][rng.random_range(0..2)]);
        }
        for _ in 0..4 {
            words.push(FILLER[rng.random_range(0..FILLER.len())]);
        }
        // Shuffle the words so bigrams vary.
        for k in (1..words.len()).rev() {
            let j = rng.random_range(0..=k);
            words.swap(k, j);
        }
        let text = words.join(" ");
        passages
            .push(&PassageRow {
                corpusid: (i / 3) as i64,
                section: "body".into(),
                ordinal: (i % 3) as i32,
                char_start: 0,
                char_end: text.chars().count() as i64,
                text,
            })
            .unwrap();
        assign.write_all(&2i32.to_le_bytes()).unwrap();
        assign.write_all(&(fam as u16).to_le_bytes()).unwrap();
        assign.write_all(&(leaf as u16).to_le_bytes()).unwrap();
        leaves.push(leaf);
    }
    passages.finish().unwrap();
    assign.flush().unwrap();
    write_model(dir, &[2, 3]);
    leaves
}

fn write_model(dir: &Path, levels: &[usize]) {
    let report = TopicModelReport {
        schema_version: 1,
        dim: 8,
        levels: levels.to_vec(),
        total_centroids: levels
            .iter()
            .scan(1, |n, k| {
                *n *= k;
                Some(*n)
            })
            .sum(),
        sample: "base_vectors.fvecs".into(),
        sample_size: 0,
        sample_order: SampleOrder::Prefix,
        seed: 42,
        iterations: 50,
        tolerance: 1e-4,
        normalize: true,
        kernel: "test".into(),
        fit_seconds: 0.0,
        per_level: levels
            .iter()
            .map(|k| LevelReport {
                branching: *k,
                clusters: *k,
                empty: 0,
                runs: 1,
                converged: 1,
                max_final_movement: 0.0,
                repairs: 0,
            })
            .collect(),
        assignment: None,
    };
    std::fs::write(
        dir.join("topic_centroids.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
}

fn options() -> Options {
    let mut o = Options::new();
    o.set("passages", "passages.parquet");
    o.set("assignments", "topic_assign.u16vecs");
    o.set("model", "topic_centroids.json");
    o.set("output", "profiles/base/topic_labels.slab");
    o.set("sample-per-cluster", "50");
    o.set("min-sample", "5");
    o.set("top-terms", "3");
    o.set("seed", "42");
    o
}

fn run(dir: &Path, threads: usize) -> (Status, String) {
    let mut op = ComputeTopicLabelsOp;
    let mut ctx = test_ctx(dir, threads);
    let r = op.execute(&options(), &mut ctx);
    (r.status, r.message)
}

#[test]
fn topic_labels_name_clusters_from_their_own_vocabulary() {
    let dir = tmp_dir();
    let _leaves = write_fixture(dir.path(), 100);
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();

    let (status, message) = run(dir.path(), 4);
    assert_eq!(status, Status::Ok, "{}", message);
    assert!(
        message.contains("8 clusters labelled over 2 levels"),
        "{}",
        message
    );

    let labels = veks_pipeline::pipeline::commands::compute_topic_labels::read_labels(
        &dir.path().join("profiles/base/topic_labels.slab"),
    )
    .unwrap();
    assert_eq!(labels.len(), 8);
    // Level 1: family words. Level 2: the leaf's own words.
    for (level, code, label) in &labels {
        let words: Vec<&str> = label.split('-').collect();
        match level {
            1 => assert!(
                words.iter().any(|w| FAMILY_WORDS[*code].contains(w)),
                "family {} label `{}` lacks a family word",
                code,
                label
            ),
            2 => assert!(
                words.iter().any(|w| LEAF_WORDS[*code].contains(w)),
                "leaf {} label `{}` lacks a leaf word",
                code,
                label
            ),
            other => panic!("unexpected level {}", other),
        }
        assert!(
            !words.iter().any(|w| FILLER.contains(w)),
            "filler in `{}`",
            label
        );
    }
    let level2: std::collections::HashSet<&String> = labels
        .iter()
        .filter(|(l, _, _)| *l == 2)
        .map(|(_, _, s)| s)
        .collect();
    assert_eq!(level2.len(), 6, "leaf labels unique");
    // Records are in level, then code, order.
    let order: Vec<(usize, usize)> = labels.iter().map(|(l, c, _)| (*l, *c)).collect();
    assert_eq!(
        order,
        vec![
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5)
        ]
    );

    let report: TopicLabelsReport = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("profiles/base/topic_labels.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report.levels, vec![2, 3]);
    assert_eq!(report.per_level.len(), 2);
    assert!(
        report.per_level.iter().all(|l| l.positional == 0),
        "{:?}",
        report.per_level
    );
    assert_eq!(report.per_level[1].sample_max, 50, "cap honoured");
    assert!(report.docs_tokenized <= 600);

    // The artifact check judges the slab against the model.
    let op = ComputeTopicLabelsOp;
    let output = dir.path().join("profiles/base/topic_labels.slab");
    assert_eq!(
        op.check_artifact(&output, &options()),
        ArtifactState::Complete
    );
    write_model(dir.path(), &[2, 4]);
    assert_eq!(
        op.check_artifact(&output, &options()),
        ArtifactState::Partial
    );
    write_model(dir.path(), &[2, 3]);
    assert_eq!(
        op.check_artifact(&output, &options()),
        ArtifactState::Complete
    );
    assert_eq!(
        op.check_artifact(&dir.path().join("profiles/base/nowhere.slab"), &options()),
        ArtifactState::Absent
    );
}

/// Thin clusters get positional labels rather than labels fitted to
/// a handful of passages.
#[test]
fn topic_labels_fall_back_to_positional_below_min_sample() {
    let dir = tmp_dir();
    write_fixture(dir.path(), 4);
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
    let mut op = ComputeTopicLabelsOp;
    let mut ctx = test_ctx(dir.path(), 2);
    let mut o = options();
    o.set("min-sample", "10");
    let r = op.execute(&o, &mut ctx);
    assert_eq!(r.status, Status::Ok, "{}", r.message);
    let labels = veks_pipeline::pipeline::commands::compute_topic_labels::read_labels(
        &dir.path().join("profiles/base/topic_labels.slab"),
    )
    .unwrap();
    for (level, code, label) in labels.iter().filter(|(l, _, _)| *l == 2) {
        assert_eq!(label, &format!("l{}-{:05}", level, code));
    }
    // Level 1 sees 12 rows per family, above the threshold.
    assert!(
        labels
            .iter()
            .filter(|(l, _, _)| *l == 1)
            .all(|(_, _, s)| !s.starts_with("l1-"))
    );
}

#[test]
fn topic_labels_are_identical_across_thread_counts() {
    let mut slabs = Vec::new();
    for threads in [1usize, 5] {
        let dir = tmp_dir();
        write_fixture(dir.path(), 60);
        std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
        let (status, message) = run(dir.path(), threads);
        assert_eq!(status, Status::Ok, "{}", message);
        slabs.push(std::fs::read(dir.path().join("profiles/base/topic_labels.slab")).unwrap());
    }
    assert_eq!(slabs[0], slabs[1]);
}
