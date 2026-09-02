// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `compute topics`.
//!
//! Builds an fvecs base with a planted two-level hierarchy in corpus
//! order — two families of three sub-topics, rows grouped by family so
//! a prefix would be one corner of the corpus — runs the command, and
//! checks every output against what was planted: the centroid facet
//! holds Σ levels records, every assignment nests under its parent and
//! recovers the planted leaf, margins order correctly, the model report
//! describes the fit, `check_artifact` judges the outputs against the
//! configuration, and the whole run is byte-identical across thread
//! counts.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use half::f16;
use indexmap::IndexMap;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use vectordata::io::{VectorReader, XvecReader};
use veks_core::ui::{TestSink, UiHandle};
use veks_pipeline::pipeline::command::{ArtifactState, CommandOp, Options, Status, StreamContext};
use veks_pipeline::pipeline::commands::compute_topics::{ComputeTopicsOp, TopicModelReport};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::resource::ResourceGovernor;

/// Create a tempdir under `target/tmp/`.
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

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// `families × subs` planted leaves, `per` rows each, written in corpus
/// order (all of family 0, then family 1). Returns each row's leaf.
fn write_planted_base(
    path: &Path,
    families: usize,
    subs: usize,
    per: usize,
    dim: usize,
) -> Vec<usize> {
    use std::io::Write;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(77);
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut leaves = Vec::new();
    for fam in 0..families {
        for i in 0..per * subs {
            let sub = i % subs;
            let mut v = vec![0.0f32; dim];
            v[fam] = 1.0;
            v[families + fam * subs + sub] = 0.6;
            for x in v.iter_mut() {
                *x += 0.05 * (rng.random::<f32>() - 0.5);
            }
            normalize(&mut v);
            file.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for x in &v {
                file.write_all(&x.to_le_bytes()).unwrap();
            }
            leaves.push(fam * subs + sub);
        }
    }
    file.flush().unwrap();
    leaves
}

fn options(dir: &Path) -> Options {
    let mut o = Options::new();
    o.set("base", "base_all.fvecs");
    o.set("centroids", "profiles/base/topic_centroids.fvecs");
    o.set("output", ".cache/topic_assign.u16vecs");
    o.set("margin", ".cache/topic_margin_all.mvecs");
    o.set("levels", "2,3");
    o.set("sample-size", "600");
    o.set("sample-order", "strided");
    o.set("seed", "42");
    o.set("iterations", "50");
    o.set("tolerance", "1e-6");
    let _ = dir;
    o
}

fn run(dir: &Path, threads: usize) -> (Status, String) {
    let mut op = ComputeTopicsOp;
    let mut ctx = test_ctx(dir, threads);
    let r = op.execute(&options(dir), &mut ctx);
    (r.status, r.message)
}

#[test]
fn compute_topics_fits_assigns_and_checks() {
    let dir = tmp_dir();
    let dim = 16;
    let leaves = write_planted_base(&dir.path().join("base_all.fvecs"), 2, 3, 500, dim);
    let n = leaves.len();
    std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();

    let (status, message) = run(dir.path(), 4);
    assert_eq!(status, Status::Ok, "{}", message);
    assert!(message.contains("8 centroids over 2 levels"), "{}", message);

    // Centroids: Σ levels records of the base's dimension.
    let centroids =
        XvecReader::<f32>::open_path(&dir.path().join("profiles/base/topic_centroids.fvecs"))
            .unwrap();
    assert_eq!(centroids.count(), 8);
    assert_eq!(centroids.dim(), dim);
    for i in 0..8 {
        let c = centroids.get(i).unwrap();
        let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "centroid {} norm {}", i, norm);
    }

    // Assignments: one record per base vector, one code per level,
    // leaf nested under parent, planted leaves recovered exactly.
    let assign =
        XvecReader::<u16>::open_path(&dir.path().join(".cache/topic_assign.u16vecs")).unwrap();
    assert_eq!(assign.count(), n);
    assert_eq!(assign.dim(), 2);
    let mut leaf_by_planted: std::collections::HashMap<usize, u16> = Default::default();
    for (i, planted) in leaves.iter().enumerate() {
        let codes = assign.get(i).unwrap();
        assert_eq!(codes[0] as usize, codes[1] as usize / 3, "row {} nests", i);
        match leaf_by_planted.get(planted) {
            Some(prev) => assert_eq!(
                *prev, codes[1],
                "planted leaf {} split at row {}",
                planted, i
            ),
            None => {
                assert!(
                    !leaf_by_planted.values().any(|v| *v == codes[1]),
                    "leaf {} holds two planted groups",
                    codes[1]
                );
                leaf_by_planted.insert(*planted, codes[1]);
            }
        }
    }
    assert_eq!(leaf_by_planted.len(), 6);

    // Margins: dim 2, chosen leaf at least as close as its sibling.
    let margin =
        XvecReader::<f16>::open_path(&dir.path().join(".cache/topic_margin_all.mvecs")).unwrap();
    assert_eq!(margin.count(), n);
    assert_eq!(margin.dim(), 2);
    for i in (0..n).step_by(97) {
        let m = margin.get(i).unwrap();
        let (best, runner) = (m[0].to_f32(), m[1].to_f32());
        assert!(
            best >= 0.0 && best <= runner + 1e-3,
            "row {}: {} vs {}",
            i,
            best,
            runner
        );
    }

    // Model report beside the centroids.
    let report: TopicModelReport = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("profiles/base/topic_centroids.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report.levels, vec![2, 3]);
    assert_eq!(report.total_centroids, 8);
    assert_eq!(report.dim, dim);
    assert_eq!(report.sample_size, 600);
    assert!(
        report.per_level.iter().all(|l| l.empty == 0),
        "{:?}",
        report.per_level
    );
    assert_eq!(report.assignment.as_ref().unwrap().records, n as u64);
    assert!(report.assignment.as_ref().unwrap().margin_written);

    // check_artifact: Complete against the run's options, Partial
    // against a different configuration or a damaged output.
    let op = ComputeTopicsOp;
    let output = dir.path().join(".cache/topic_assign.u16vecs");
    assert_eq!(
        op.check_artifact(&output, &options(dir.path())),
        ArtifactState::Complete
    );
    let mut other = options(dir.path());
    other.set("levels", "2,4");
    assert_eq!(op.check_artifact(&output, &other), ArtifactState::Partial);
    let mut no_margin = options(dir.path());
    no_margin.set("margin", ".cache/absent.mvecs");
    assert_eq!(
        op.check_artifact(&output, &no_margin),
        ArtifactState::Partial
    );
    let model_path = dir.path().join("profiles/base/topic_centroids.json");
    let saved = std::fs::read(&model_path).unwrap();
    std::fs::remove_file(&model_path).unwrap();
    assert_eq!(
        op.check_artifact(&output, &options(dir.path())),
        ArtifactState::Partial
    );
    std::fs::write(&model_path, &saved).unwrap();
    assert_eq!(
        op.check_artifact(&output, &options(dir.path())),
        ArtifactState::Complete
    );
    assert_eq!(
        op.check_artifact(
            &dir.path().join(".cache/nowhere.u16vecs"),
            &options(dir.path())
        ),
        ArtifactState::Absent
    );
}

/// The same base and configuration on 1 and 6 threads produce
/// byte-identical centroids, assignments and margins.
#[test]
fn compute_topics_is_identical_across_thread_counts() {
    let outputs = [
        "profiles/base/topic_centroids.fvecs",
        ".cache/topic_assign.u16vecs",
        ".cache/topic_margin_all.mvecs",
    ];
    let mut runs: Vec<Vec<Vec<u8>>> = Vec::new();
    for threads in [1usize, 6] {
        let dir = tmp_dir();
        write_planted_base(&dir.path().join("base_all.fvecs"), 2, 3, 400, 12);
        std::fs::create_dir_all(dir.path().join("profiles/base")).unwrap();
        let (status, message) = run(dir.path(), threads);
        assert_eq!(status, Status::Ok, "{}", message);
        runs.push(
            outputs
                .iter()
                .map(|o| std::fs::read(dir.path().join(o)).unwrap())
                .collect(),
        );
    }
    for (i, o) in outputs.iter().enumerate() {
        assert_eq!(
            runs[0][i], runs[1][i],
            "{} differs between thread counts",
            o
        );
    }
}
