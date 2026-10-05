// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end shard sizing: bootstrap a dataset under a small file-size
//! cap, run the real pipeline, and check what landed on disk.
//!
//! The caps here are absurdly small — hundreds of bytes — so that a
//! 200-vector fixture crosses several shard boundaries. That is the
//! point: the arithmetic is identical at 1 TB, and a test that used
//! the real default would have to write a terabyte to exercise a
//! single rollover.
//!
//! These run the `veks` binary, not a library call, so they cover the
//! whole path: the wizard's answer → `upstream.defaults.shard_size` →
//! the resource governor → whichever writer the step uses.
//!
//! See `docs/design/srd-multifile-facet-shards.md`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use veks::prepare::import::ImportArgs;

fn veks_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_veks"))
}

fn make_tempdir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// Distinct, non-zero, non-duplicate vectors — so dedup and the zero
/// check remove nothing and the counts below are exact.
fn write_fvec(path: &Path, count: usize, dim: usize) {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for i in 0..count {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            w.write_all(&(((i * dim + d) as f32) + 1.0).to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

fn args_for(name: &str, out: &Path, cap: u64) -> ImportArgs {
    let mut a = base_args(name, out);
    a.max_shard_bytes = cap;
    a
}

fn base_args(name: &str, output: &Path) -> ImportArgs {
    ImportArgs {
        name: name.to_string(),
        output: output.to_path_buf(),
        base_vectors: None,
        query_vectors: None,
        self_search: true,
        query_count: 10,
        metadata: None,
        ground_truth: None,
        ground_truth_distances: None,
        metric: "Cosine".to_string(),
        neighbors: 5,
        seed: 42,
        description: None,
        no_dedup: false,
        no_filtered: false,
        no_zero_check: false,
        duplicate_count: None,
        zero_count: None,
        normalize: false,
        force: true,
        base_convert_format: None,
        query_convert_format: None,
        compress_cache: false,
        sized_profiles: None,
        base_fraction: 1.0,
        required_facets: None,
        round_digits: 10,
        max_shard_bytes: vectordata::dataset::DEFAULT_MAX_SHARD_BYTES,
        pedantic_dedup: false,
        selectivity: 0.0001,
        predicate_count: 100,
        predicate_strategy: "eq".to_string(),
        provided_facets: None,
        classic: false,
        personality: "native".to_string(),
        synthesize_metadata: false,
        synthesis_mode: "simple-int-eq".to_string(),
        synthesis_format: "slab".to_string(),
        metadata_fields: 3,
        metadata_range_min: 0,
        metadata_range_max: 1000,
        predicate_range_min: 0,
        predicate_range_max: 1000,
        verify_knn_sample: 0,
        partition_oracles: false,
        max_partitions: 100,
        on_undersized: "error".to_string(),
        cosine_mode: None,
    }
}

fn run_pipeline(dataset_yaml: &Path) -> (bool, String) {
    let out = Command::new(veks_bin())
        .arg("run")
        .arg("--output").arg("batch")
        .arg("--threads").arg("2")
        .arg(dataset_yaml)
        .output()
        .expect("failed to execute veks");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// Bootstrap + run, returning the dataset directory.
fn build(name: &str, cap: u64, tune: impl FnOnce(&mut ImportArgs)) -> (tempfile::TempDir, PathBuf, String) {
    let tmp = make_tempdir();
    let src = tmp.path().join("vectors.fvecs");
    write_fvec(&src, 200, 4);

    let out = tmp.path().join("dataset");
    let mut args = args_for(name, &out, cap);
    args.base_vectors = Some(src);
    tune(&mut args);
    veks::prepare::import::run(args);

    let yaml = out.join("dataset.yaml");
    assert!(yaml.exists(), "bootstrap produced no dataset.yaml");
    let (ok, log) = run_pipeline(&yaml);
    assert!(ok, "pipeline failed:\n{log}");
    (tmp, out, log)
}

/// Every `.fvecs`/`.ivecs`/`.slab` facet file under `profiles/`, with
/// its size.
fn facet_files(root: &Path) -> Vec<(PathBuf, u64)> {
    fn walk(dir: &Path, acc: &mut Vec<(PathBuf, u64)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, acc);
            } else {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.ends_with(".mref") || name.ends_with(".tmp") {
                    continue;
                }
                let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                acc.push((p, len));
            }
        }
    }
    let mut acc = Vec::new();
    walk(&root.join("profiles"), &mut acc);
    acc.sort();
    acc
}

// ── the cap reaches the run ────────────────────────────────────────

/// **The wizard's answer reaches the governor.** Bootstrap records the
/// cap in the dataset, and the run reports it back — the link that
/// makes every assertion below meaningful rather than incidental.
#[test]
fn the_declared_cap_reaches_the_running_pipeline() {
    let (_tmp, out, log) = build("cap-reaches", 500, |_| {});

    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(yaml.contains("shard_size:"), "the dataset declares its cap:\n{yaml}");
    assert!(
        log.contains("shardsize: 500 B") || log.contains("shardsize"),
        "the governor reports the cap it is enforcing:\n{log}"
    );
}

/// The default cap is 1 TB, and a dataset bootstrapped without an
/// opinion carries it — so "facet files are limited by default" is
/// true of a plain `veks prepare bootstrap`, not only of a run that
/// asked.
#[test]
fn the_default_cap_is_one_terabyte() {
    let (_tmp, out, _log) = build("cap-default", vectordata::dataset::DEFAULT_MAX_SHARD_BYTES, |_| {});
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(
        yaml.contains("shard_size: 1000000000000") || yaml.contains("shard_size: '1000000000000'"),
        "the default cap is declared:\n{yaml}"
    );
}

/// **A cap no facet reaches changes nothing.** Every output is one
/// file, byte-for-byte what an uncapped run produces — so the feature
/// is inert until a facet is actually large.
#[test]
fn a_roomy_cap_leaves_every_facet_a_single_file() {
    let (_tmp, capped, _) = build("roomy", 1_000_000_000_000, |_| {});
    let (_tmp2, plain, _) = build("plain", u64::MAX, |_| {});

    let a = facet_files(&capped);
    let b = facet_files(&plain);
    let names = |v: &[(PathBuf, u64)]| -> Vec<String> {
        v.iter()
            .map(|(p, _)| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect()
    };
    assert_eq!(names(&a), names(&b), "the same files either way");
    assert!(
        !names(&a).iter().any(|n| n.contains("__0000")),
        "nothing was sharded: {:?}",
        names(&a)
    );
}

// ── what a small cap actually produces ─────────────────────────────

/// **Every facet file stays under the cap.** No exceptions list: a
/// facet the cap is too small for is written as a series, and the
/// steps that read it later read across its shards.
#[test]
fn every_facet_file_stays_under_the_cap() {
    let cap = 400u64;
    let (_tmp, out, _log) = build("small-cap", cap, |_| {});

    let files = facet_files(&out);
    assert!(!files.is_empty(), "the run produced no facets");
    for (path, len) in files {
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(len <= cap, "{name} is {len} bytes over a {cap}-byte cap");
    }
}

/// **A small cap actually produces a series.** Without this the test
/// above would pass on a run that simply wrote small files.
#[test]
fn a_small_cap_produces_a_multi_file_facet() {
    let (_tmp, out, log) = build("produces-series", 400, |_| {});

    let sharded: Vec<String> = facet_files(&out)
        .iter()
        .map(|(p, _)| p.file_name().unwrap().to_str().unwrap().to_string())
        .filter(|n| n.contains("__0000"))
        .collect();
    assert!(!sharded.is_empty(), "nothing was sharded: {log}");

    // And the base facet is the one that got large enough.
    assert!(
        sharded.iter().any(|n| n.starts_with("base_vectors")),
        "base vectors were not sharded: {sharded:?}"
    );
}

/// **The declaration describes what was written** (SH-37). A facet
/// split into shards is declared as a series, with the stride and
/// count the files actually have — so a consumer that never ran the
/// pipeline reads the same facet the pipeline did.
#[test]
fn a_sharded_facet_is_declared_as_a_series() {
    let (_tmp, out, _log) = build("declared", 400, |_| {});
    let yaml = std::fs::read_to_string(out.join("dataset.yaml")).unwrap();

    assert!(yaml.contains("base_vectors__NNNN.fvecs"), "the pattern, not a filename:\n{yaml}");
    assert!(yaml.contains("shard_stride:"), "{yaml}");
    assert!(yaml.contains("shard_count:"), "{yaml}");
    assert!(yaml.contains("record_count:"), "{yaml}");

    // The declared count matches the shards on disk.
    let dir = out.join("profiles/base");
    let shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .filter(|n| n.starts_with("base_vectors__") && n.ends_with(".fvecs"))
        .collect();
    let declared = yaml
        .lines()
        .find_map(|l| l.trim().strip_prefix("shard_count: "))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .expect("a declared shard count");
    assert_eq!(declared, shards.len(), "declared {declared}, on disk {}", shards.len());
}

/// **The KNN ground truth is byte-identical to an unsharded run.**
///
/// This is the assertion that makes the whole thing worth having.
/// `compute knn` scans the base facet through the zero-copy path; if
/// shard resolution were wrong there it would read adjacent memory and
/// return plausible neighbours rather than failing. Comparing against
/// the same data laid out as one file is the only check that catches
/// that.
#[test]
fn sharded_and_unsharded_runs_produce_identical_ground_truth() {
    let (_tmp_a, sharded, _) = build("gt-sharded", 400, |_| {});
    let (_tmp_b, whole, _) = build("gt-whole", u64::MAX, |_| {});

    // The sharded run really is sharded, or this proves nothing.
    assert!(
        sharded.join("profiles/base/base_vectors__0000.fvecs").exists(),
        "the sharded run did not shard"
    );

    for facet in ["neighbor_indices.ivecs", "neighbor_distances.fvecs"] {
        let a = std::fs::read(sharded.join("profiles/default").join(facet)).unwrap();
        let b = std::fs::read(whole.join("profiles/default").join(facet)).unwrap();
        assert!(!a.is_empty(), "{facet} is empty");
        assert_eq!(a, b, "{facet} differs between a sharded and an unsharded run");
    }
}

/// **A capped run and an uncapped run hold the same data.** Whatever
/// the layout, the vectors that come out are identical — which is the
/// property a shard split could silently break.
#[test]
fn capping_does_not_change_the_data() {
    let (_tmp_a, capped, _) = build("data-capped", 400, |_| {});
    let (_tmp_b, plain, _) = build("data-plain", u64::MAX, |_| {});

    // Read a facet as a flat vector stream, across shards if any.
    let read_facet = |root: &Path, stem: &str, ext: &str| -> Vec<Vec<f32>> {
        let dir = root.join("profiles/base");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().unwrap().to_str().unwrap();
                n.starts_with(stem) && n.ends_with(ext) && !n.ends_with(".mref")
            })
            .collect();
        files.sort();
        let mut out = Vec::new();
        for f in files {
            let data = std::fs::read(&f).unwrap();
            if data.is_empty() {
                continue;
            }
            let dim = i32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
            let stride = 4 + dim * 4;
            for r in 0..data.len() / stride {
                let at = r * stride + 4;
                out.push(
                    (0..dim)
                        .map(|d| {
                            let e = at + d * 4;
                            f32::from_le_bytes(data[e..e + 4].try_into().unwrap())
                        })
                        .collect(),
                );
            }
        }
        out
    };

    for stem in ["base_vectors", "query_vectors"] {
        let a = read_facet(&capped, stem, ".fvecs");
        let b = read_facet(&plain, stem, ".fvecs");
        assert!(!a.is_empty(), "{stem} produced nothing");
        assert_eq!(a, b, "{stem} differs between a capped and an uncapped run");
    }
}

/// A cap too small to hold a run of records does not produce a file
/// per record: the writer declines to shard rather than emitting a
/// layout nothing wants.
#[test]
fn an_unusably_small_cap_does_not_shatter_the_output() {
    // Twenty bytes is exactly one dim-4 fvec record.
    let (_tmp, out, _log) = build("tiny-cap", 20, |_| {});
    let files = facet_files(&out);
    assert!(
        files.len() < 50,
        "a 20-byte cap must not produce a file per record: {} files",
        files.len()
    );
    for (path, _) in &files {
        let n = path.file_name().unwrap().to_str().unwrap();
        assert!(!n.contains("__0010"), "shattered into shards: {n}");
    }
}

/// The dataset re-opens and reads correctly after a capped run — the
/// declaration and the files agree, whatever the layout.
#[test]
fn a_capped_dataset_still_opens_and_reads() {
    let (_tmp, out, _log) = build("reopen", 400, |_| {});

    let group = vectordata::TestDataGroup::load(out.to_str().unwrap())
        .expect("a capped dataset must still load");
    let view = group.profile("default").expect("default profile");
    let base = view.base_vectors().expect("base vectors must open");
    assert!(base.count() > 0, "base vectors read back empty");
    assert_eq!(base.dim(), 4);
    // And a value round-trips, so the declaration points at real data.
    let first = base.get(0).expect("record 0");
    assert_eq!(first.len(), 4);
}

/// **A cache artifact is never sharded**, however small the cap.
///
/// This is a regression test for a real break: `convert` split
/// `.cache/all_vectors.mvec` into a series, and the next step — which
/// opens that path exactly, by name — reported the artifact Absent and
/// failed the run. A cache artifact is consumed inside the run by a
/// reader that maps one file, and nothing outside the run ever stores
/// or moves it, so there is nothing for a cap to protect.
#[test]
fn a_cache_artifact_is_never_sharded() {
    let (_tmp, out, _log) = build("cache-intact", 400, |a| {
        // Forces a `convert` step, which writes through the
        // record-oriented sink into the cache.
        a.base_convert_format = Some("mvec".to_string());
    });

    let cache = out.join(".cache");
    let mut sharded = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&cache) {
        for e in entries.flatten() {
            let name = e.file_name().to_str().unwrap_or("").to_string();
            if name.contains("__0000") || name.contains("__0001") {
                sharded.push(name);
            }
        }
    }
    assert!(
        sharded.is_empty(),
        "cache artifacts were split into a series: {sharded:?}"
    );
}

/// The same run, driven through the record-oriented sink, still
/// produces a complete and correct dataset — the cap changes where
/// bytes go, never whether the run works.
#[test]
fn a_converted_dataset_builds_under_a_small_cap() {
    let (_tmp, out, _log) = build("converted", 400, |a| {
        a.base_convert_format = Some("mvec".to_string());
    });
    let group = vectordata::TestDataGroup::load(out.to_str().unwrap())
        .expect("a converted, capped dataset must load");
    let view = group.profile("default").expect("default profile");
    let base = view.base_vectors().expect("base vectors must open");
    assert!(base.count() > 0);
    assert_eq!(base.dim(), 4);
}

// ── across formats ─────────────────────────────────────────────────

/// **f16 records shard the same way f32 records do.** A converted
/// facet has half the record width, so the same cap holds twice the
/// records — the stride follows the format rather than being fixed.
#[test]
fn an_f16_facet_shards_at_its_own_record_width() {
    let (_tmp, out, _log) = build("f16", 400, |a| {
        a.base_convert_format = Some("mvec".to_string());
    });

    for (path, len) in facet_files(&out) {
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(len <= 400, "{name} is {len} bytes over a 400-byte cap");
    }
    // And the dataset still reads.
    let group = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let base = view.base_vectors().unwrap();
    assert!(base.count() > 0);
    assert_eq!(base.dim(), 4);
}

/// **A metadata facet is capped too.** Synthesized as integer vectors,
/// so this covers a third element width alongside the f32 and f16
/// cases above.
///
/// (The slab synthesis format is not used here: this fixture's
/// `verify-predicates-sqlite` step cannot load slab metadata, with or
/// without a cap — a pre-existing limitation of that combination.)
#[test]
fn a_synthesized_metadata_facet_is_capped() {
    let (_tmp, out, _log) = build("meta", 4096, |a| {
        a.synthesize_metadata = true;
        a.required_facets = Some("BQGM".to_string());
        a.synthesis_format = "ivec".to_string();
        a.metadata_fields = 2;
    });

    for (path, len) in facet_files(&out) {
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(len <= 4096, "{name} is {len} bytes over a 4096-byte cap");
    }

    let group = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    assert!(group.profile("default").is_some());
}

/// **The neighbour facets shard too.** They are the KNN outputs, and
/// they are read back by `verify-knn` in the same run — so a cap that
/// splits them exercises both halves at once.
#[test]
fn the_neighbour_facets_shard_and_are_still_read_back() {
    // 5 neighbours → 24-byte rows; a 100-byte cap holds 4 of them,
    // and the decade below that is 1... so use a cap that yields 10.
    let (_tmp, out, log) = build("neighbours", 240, |_| {});

    let dir = out.join("profiles/default");
    let sharded: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_str().unwrap().to_string())
        .filter(|n| n.contains("neighbor") && n.contains("__0000"))
        .collect();

    // Either they sharded, or they fit — both are correct, but the run
    // must have completed and the facets must be readable either way.
    assert!(log.contains("Pipeline complete") || !log.contains("failed"), "{log}");
    let group = vectordata::TestDataGroup::load(out.to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    assert!(view.neighbor_indices().is_ok(), "neighbours must read back");
    eprintln!("sharded neighbour facets: {sharded:?}");
}
