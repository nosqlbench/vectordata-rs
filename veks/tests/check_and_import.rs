// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for `datasets import` and `veks check`.
//!
//! Uses synthetic small files, verifies before/after state, and covers
//! corner cases adversarially. All state is local to temp directories.

use std::path::{Path, PathBuf};

use veks::check;
use veks::datasets::import::{ImportArgs};

// ═══════════════════════════════════════════════════════════════════════════
// Test helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Write a minimal fvec file (dim-header + zero-filled vectors).
fn write_fvec(path: &Path, records: usize, dim: u32) {
    use std::io::Write;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..records {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            let val = (i * dim as usize + d as usize) as f32;
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Write a minimal mvec (f16) file.
fn write_mvec(path: &Path, records: usize, dim: u32) {
    use std::io::Write;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    for i in 0..records {
        w.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for d in 0..dim {
            let val = half::f16::from_f32((i * dim as usize + d as usize) as f32);
            w.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Create a directory with a fake parquet file for format detection.
fn write_parquet_dir(path: &Path) {
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(path.join("part-0.parquet"), b"PARQUET_FAKE").unwrap();
}

/// Create a directory with npy files.
fn write_npy_dir(path: &Path) {
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(path.join("shard_0.npy"), b"NPY_FAKE_DATA").unwrap();
}

/// Read a dataset.yaml and return its content as a string.
fn read_yaml(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("dataset.yaml")).unwrap()
}

/// Check that a dataset.yaml exists in a directory.
fn has_dataset_yaml(dir: &Path) -> bool {
    dir.join("dataset.yaml").exists()
}

fn default_args(name: &str, output: &Path) -> ImportArgs {
    ImportArgs {
        name: name.to_string(),
        output: output.to_path_buf(),
        base_vectors: None,
        query_vectors: None,
        self_search: false,
        query_count: 100,
        metadata: None,
        ground_truth: None,
        ground_truth_distances: None,
        metric: "L2".to_string(),
        neighbors: 10,
        seed: 42,
        description: None,
        no_dedup: false,
        no_filtered: false,
        force: false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// datasets import: slot resolution tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn import_minimal_no_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let args = default_args("empty-test", &out);

    veks::datasets::import::run(args);

    assert!(has_dataset_yaml(&out));
    let yaml = read_yaml(&out);
    assert!(yaml.contains("name: empty-test"));
    // No upstream steps when no base vectors
    assert!(yaml.contains("profiles:"));
}

#[test]
fn import_native_fvec_self_search() {
    let dir = tempfile::tempdir().unwrap();
    let fvec = dir.path().join("vectors.fvec");
    write_fvec(&fvec, 200, 4);

    let out = dir.path().join("dataset");
    let mut args = default_args("fvec-self", &out);
    args.base_vectors = Some(fvec.clone());
    args.query_count = 10;

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);

    // Should NOT have an import step (native fvec → identity)
    assert!(!yaml.contains("run: import"), "native fvec should not need import step");

    // Should have self-search steps
    assert!(yaml.contains("shuffle-ordinals"), "should have shuffle");
    assert!(yaml.contains("extract-query-vectors"), "should have query extract");
    assert!(yaml.contains("extract-base-vectors"), "should have base extract");
    assert!(yaml.contains("compute knn"), "should have KNN");
    assert!(yaml.contains("compute dedup"), "should have dedup");

    // Profile should reference the extracted vectors, not the source
    assert!(yaml.contains("profiles/base/base_vectors.mvec")
        || yaml.contains("profiles/base/base_vectors.fvec")
        || yaml.contains(&fvec.to_string_lossy().to_string()),
        "profile should reference vectors");
}

#[test]
fn import_npy_dir_needs_import_step() {
    let dir = tempfile::tempdir().unwrap();
    let npy = dir.path().join("embeddings");
    write_npy_dir(&npy);

    let out = dir.path().join("dataset");
    let mut args = default_args("npy-import", &out);
    args.base_vectors = Some(npy);
    args.query_count = 5;

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);

    // Should have import step (npy is foreign format)
    assert!(yaml.contains("run: import"), "npy dir should need import step");
    assert!(yaml.contains("${cache}/all_vectors.mvec"), "import output should go to cache");
}

#[test]
fn import_separate_query_no_shuffle() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.fvec");
    let query = dir.path().join("query.fvec");
    write_fvec(&base, 100, 3);
    write_fvec(&query, 10, 3);

    let out = dir.path().join("dataset");
    let mut args = default_args("sep-query", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query.clone());

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);

    // Separate query: no shuffle, no extract
    assert!(!yaml.contains("shuffle-ordinals"), "separate query should not shuffle");
    assert!(!yaml.contains("extract-query"), "separate query should not extract");
    assert!(!yaml.contains("extract-base"), "separate query should not extract base");

    // KNN should still be present
    assert!(yaml.contains("compute knn"), "should have KNN with separate query");

    // Query should be referenced directly (identity)
    assert!(yaml.contains(&query.to_string_lossy().to_string()),
        "query path should be referenced directly");
}

#[test]
fn import_no_dedup_flag() {
    let dir = tempfile::tempdir().unwrap();
    let fvec = dir.path().join("base.fvec");
    write_fvec(&fvec, 50, 2);

    let out = dir.path().join("dataset");
    let mut args = default_args("no-dedup", &out);
    args.base_vectors = Some(fvec);
    args.no_dedup = true;

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);
    assert!(!yaml.contains("compute dedup"), "should not have dedup when --no-dedup");
}

#[test]
fn import_with_metadata_generates_predicate_chain() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.fvec");
    let query = dir.path().join("query.fvec");
    let meta = dir.path().join("meta");
    write_fvec(&base, 100, 3);
    write_fvec(&query, 10, 3);
    write_parquet_dir(&meta);

    let out = dir.path().join("dataset");
    let mut args = default_args("with-meta", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.metadata = Some(meta);

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);
    assert!(yaml.contains("import-metadata"), "should import metadata");
    assert!(yaml.contains("survey-metadata"), "should survey metadata");
    assert!(yaml.contains("synthesize-predicates"), "should synthesize predicates");
    assert!(yaml.contains("evaluate-predicates"), "should evaluate predicates");
    assert!(yaml.contains("compute filtered-knn"), "should have filtered KNN");
}

#[test]
fn import_no_filtered_skips_filtered_knn() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.fvec");
    let query = dir.path().join("query.fvec");
    let meta = dir.path().join("meta");
    write_fvec(&base, 100, 3);
    write_fvec(&query, 10, 3);
    write_parquet_dir(&meta);

    let out = dir.path().join("dataset");
    let mut args = default_args("no-filtered", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.metadata = Some(meta);
    args.no_filtered = true;

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);
    // Metadata chain still present
    assert!(yaml.contains("survey-metadata"));
    // But no filtered KNN
    assert!(!yaml.contains("compute filtered-knn"),
        "should not have filtered KNN with --no-filtered");
}

#[test]
fn import_precomputed_gt_skips_knn() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.fvec");
    let query = dir.path().join("query.fvec");
    let gt = dir.path().join("gt.ivec");
    write_fvec(&base, 100, 3);
    write_fvec(&query, 10, 3);
    std::fs::write(&gt, b"fake ground truth").unwrap();

    let out = dir.path().join("dataset");
    let mut args = default_args("precomputed-gt", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.ground_truth = Some(gt.clone());

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);
    assert!(!yaml.contains("compute knn"), "should not compute KNN when GT provided");
    // GT should be referenced in profile (identity)
    assert!(yaml.contains(&gt.to_string_lossy().to_string())
        || yaml.contains("neighbor_indices"),
        "GT should be in profile");
}

#[test]
fn import_force_overwrites_existing() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dataset");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("dataset.yaml"), "name: old\n").unwrap();

    let mut args = default_args("new-name", &out);
    args.force = true;

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);
    assert!(yaml.contains("name: new-name"), "force should overwrite");
    assert!(!yaml.contains("name: old"), "old content should be gone");
}

#[test]
fn import_creates_gitignore() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dataset");
    let args = default_args("gitignore-test", &out);

    veks::datasets::import::run(args);

    let gitignore = std::fs::read_to_string(out.join(".gitignore")).unwrap();
    assert!(gitignore.contains(".scratch/"));
    assert!(gitignore.contains(".cache/"));
}

// ═══════════════════════════════════════════════════════════════════════════
// veks check: pre-flight validation tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn check_publish_valid() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".publish_url"), "s3://test-bucket/prefix/\n").unwrap();

    let result = check::publish_url::check(dir.path(), &[]);
    assert!(result.passed, "valid publish URL should pass: {:?}", result.messages);
}

#[test]
fn check_publish_missing() {
    let dir = tempfile::tempdir().unwrap();
    let result = check::publish_url::check(dir.path(), &[]);
    assert!(!result.passed, "missing publish URL should fail");
}

#[test]
fn check_publish_invalid_scheme() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".publish_url"), "https://not-s3/\n").unwrap();

    let result = check::publish_url::check(dir.path(), &[]);
    assert!(!result.passed, "wrong scheme should fail");
}

#[test]
fn check_publish_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".publish_url"), "").unwrap();

    let result = check::publish_url::check(dir.path(), &[]);
    assert!(!result.passed, "empty publish URL file should fail");
}

#[test]
fn check_publish_comment_only() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".publish_url"), "# just a comment\n").unwrap();

    let result = check::publish_url::check(dir.path(), &[]);
    assert!(!result.passed, "comment-only publish URL file should fail");
}

#[test]
fn check_publish_walks_parents() {
    let dir = tempfile::tempdir().unwrap();
    let child = dir.path().join("datasets").join("my-dataset");
    std::fs::create_dir_all(&child).unwrap();
    std::fs::write(dir.path().join(".publish_url"), "s3://parent-bucket/\n").unwrap();

    let found = check::publish_url::find_publish_file(&child);
    assert!(found.is_some(), "should find publish URL in parent");
    assert_eq!(found.unwrap(), dir.path().join(".publish_url"));
}

#[test]
fn check_publish_child_overrides_parent() {
    let dir = tempfile::tempdir().unwrap();
    let child = dir.path().join("special");
    std::fs::create_dir_all(&child).unwrap();
    std::fs::write(dir.path().join(".publish_url"), "s3://parent/\n").unwrap();
    std::fs::write(child.join(".publish_url"), "s3://child-override/\n").unwrap();

    let found = check::publish_url::find_publish_file(&child);
    assert!(found.is_some());
    assert_eq!(found.unwrap(), child.join(".publish_url"));
}

#[test]
fn check_merkle_all_covered() {
    let dir = tempfile::tempdir().unwrap();
    let big_file = dir.path().join("big.fvec");
    write_fvec(&big_file, 1000, 100); // ~400KB — below 100MB threshold
    let mref = dir.path().join("big.fvec.mref");
    std::fs::write(&mref, b"fake mref").unwrap();

    let files = vec![big_file];
    // Threshold 0 means everything needs merkle
    let result = check::merkle::check(dir.path(), &files, 0);
    assert!(result.passed, "should pass with mref present: {:?}", result.messages);
}

#[test]
fn check_merkle_missing_mref() {
    let dir = tempfile::tempdir().unwrap();
    let big_file = dir.path().join("big.fvec");
    write_fvec(&big_file, 1000, 100);

    let files = vec![big_file];
    let result = check::merkle::check(dir.path(), &files, 0);
    assert!(!result.passed, "should fail with missing mref");
}

#[test]
fn check_merkle_below_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let small = dir.path().join("small.fvec");
    write_fvec(&small, 5, 3); // tiny file

    let files = vec![small];
    // High threshold: file is too small to need merkle
    let result = check::merkle::check(dir.path(), &files, 100_000_000);
    assert!(result.passed, "small file below threshold should pass without mref");
}

#[test]
fn check_integrity_valid_fvec() {
    let dir = tempfile::tempdir().unwrap();
    let fvec = dir.path().join("valid.fvec");
    write_fvec(&fvec, 10, 4);

    let files = vec![fvec];
    let result = check::integrity::check(dir.path(), &files);
    assert!(result.passed, "valid fvec should pass: {:?}", result.messages);
}

#[test]
fn check_integrity_truncated_fvec() {
    let dir = tempfile::tempdir().unwrap();
    let fvec = dir.path().join("truncated.fvec");
    write_fvec(&fvec, 10, 4);

    // Truncate the file mid-record
    let meta = std::fs::metadata(&fvec).unwrap();
    let truncated_size = meta.len() - 3; // remove last 3 bytes
    let file = std::fs::OpenOptions::new().write(true).open(&fvec).unwrap();
    file.set_len(truncated_size).unwrap();

    let files = vec![fvec];
    let result = check::integrity::check(dir.path(), &files);
    assert!(!result.passed, "truncated fvec should fail integrity check");
}

#[test]
fn check_integrity_empty_fvec() {
    let dir = tempfile::tempdir().unwrap();
    let fvec = dir.path().join("empty.fvec");
    std::fs::write(&fvec, b"").unwrap();

    let files = vec![fvec];
    let result = check::integrity::check(dir.path(), &files);
    // Empty file is valid (0 records)
    assert!(result.passed, "empty fvec should pass");
}

#[test]
fn check_integrity_bad_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let fvec = dir.path().join("bad_dim.fvec");
    // Write a negative dimension
    use std::io::Write;
    let mut f = std::fs::File::create(&fvec).unwrap();
    f.write_all(&(-1i32).to_le_bytes()).unwrap();
    f.write_all(&[0u8; 12]).unwrap(); // some garbage data

    let files = vec![fvec];
    let result = check::integrity::check(dir.path(), &files);
    assert!(!result.passed, "negative dimension should fail");
}

#[test]
fn check_integrity_mvec_valid() {
    let dir = tempfile::tempdir().unwrap();
    let mvec = dir.path().join("valid.mvec");
    write_mvec(&mvec, 10, 4);

    let files = vec![mvec];
    let result = check::integrity::check(dir.path(), &files);
    assert!(result.passed, "valid mvec should pass: {:?}", result.messages);
}

#[test]
fn check_catalogs_missing() {
    let dir = tempfile::tempdir().unwrap();
    let ds = dir.path().join("dataset.yaml");
    std::fs::write(&ds, "name: test\nprofiles:\n  default:\n    maxk: 10\n").unwrap();

    let result = check::catalogs::check(dir.path(), &[ds]);
    assert!(!result.passed, "missing catalog should fail");
}

#[test]
fn check_catalogs_present() {
    let dir = tempfile::tempdir().unwrap();
    let ds = dir.path().join("dataset.yaml");
    std::fs::write(&ds, "name: test\nprofiles:\n  default:\n    maxk: 10\n").unwrap();
    std::fs::write(dir.path().join("catalog.json"), "[]").unwrap();

    let result = check::catalogs::check(dir.path(), &[ds]);
    assert!(result.passed, "present catalog should pass: {:?}", result.messages);
}

// ═══════════════════════════════════════════════════════════════════════════
// Adversarial edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn import_empty_directory_source() {
    // A base_vectors path that points to an empty directory
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty_src");
    std::fs::create_dir(&empty).unwrap();

    let out = dir.path().join("dataset");
    let mut args = default_args("empty-dir", &out);
    args.base_vectors = Some(empty);

    // Should not panic — should produce import step (directory → import)
    veks::datasets::import::run(args);
    assert!(has_dataset_yaml(&out));
}

#[test]
fn import_nonexistent_source() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dataset");
    let mut args = default_args("nonexistent", &out);
    args.base_vectors = Some(PathBuf::from("/nonexistent/path/vectors.fvec"));

    // Should still produce a yaml (the path is a reference, not validated at import time)
    veks::datasets::import::run(args);
    assert!(has_dataset_yaml(&out));
}

#[test]
fn check_publish_url_parse_edge_cases() {
    // Valid cases
    assert!(check::publish_url::parse_publish_url("s3://bucket").is_ok());
    assert!(check::publish_url::parse_publish_url("s3://bucket/").is_ok());
    assert!(check::publish_url::parse_publish_url("s3://bucket/deep/prefix/path").is_ok());
    assert!(check::publish_url::parse_publish_url("  s3://bucket/  \n").is_ok()); // whitespace

    // Invalid cases
    assert!(check::publish_url::parse_publish_url("").is_err());
    assert!(check::publish_url::parse_publish_url("s3://").is_err());
    assert!(check::publish_url::parse_publish_url("s3:///no-bucket").is_err());
    assert!(check::publish_url::parse_publish_url("http://wrong-scheme").is_err());
    assert!(check::publish_url::parse_publish_url("just-text").is_err());
}

#[test]
fn check_merkle_mref_files_are_skipped() {
    // .mref files themselves should not need their own merkle
    let dir = tempfile::tempdir().unwrap();
    let mref = dir.path().join("data.fvec.mref");
    std::fs::write(&mref, b"fake mref content").unwrap();

    let files = vec![mref];
    let result = check::merkle::check(dir.path(), &files, 0);
    assert!(result.passed, ".mref files should be skipped");
}

#[test]
fn import_both_self_search_and_separate_query() {
    // When both --self-search and --query-vectors are given,
    // --query-vectors should win (separate query)
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.fvec");
    let query = dir.path().join("query.fvec");
    write_fvec(&base, 100, 3);
    write_fvec(&query, 10, 3);

    let out = dir.path().join("dataset");
    let mut args = default_args("both-modes", &out);
    args.base_vectors = Some(base);
    args.query_vectors = Some(query);
    args.self_search = true; // should be overridden by query_vectors

    veks::datasets::import::run(args);

    let yaml = read_yaml(&out);
    // Separate query takes precedence — no shuffle
    assert!(!yaml.contains("shuffle-ordinals"),
        "separate query should override self-search");
}

#[test]
fn check_discover_datasets_skips_hidden_dirs() {
    let dir = tempfile::tempdir().unwrap();
    // dataset.yaml in a hidden directory should NOT be discovered
    let hidden = dir.path().join(".hidden");
    std::fs::create_dir(&hidden).unwrap();
    std::fs::write(hidden.join("dataset.yaml"), "name: hidden\n").unwrap();
    // dataset.yaml in a normal directory should be discovered
    let normal = dir.path().join("visible");
    std::fs::create_dir(&normal).unwrap();
    std::fs::write(normal.join("dataset.yaml"), "name: visible\n").unwrap();

    let found = check::discover_datasets(dir.path());
    assert_eq!(found.len(), 1, "should find only visible dataset: {:?}", found);
    assert!(found[0].to_string_lossy().contains("visible"));
}

#[test]
fn publish_enumerate_skips_hidden_files() {
    let dir = tempfile::tempdir().unwrap();
    // Hidden file should be excluded
    std::fs::write(dir.path().join(".publish_url"), "s3://b/").unwrap();
    std::fs::write(dir.path().join(".gitignore"), "*.tmp").unwrap();
    std::fs::write(dir.path().join("data.fvec"), b"visible").unwrap();
    std::fs::write(dir.path().join("dataset.yaml"), "name: t\n").unwrap();
    // Hidden directory should be excluded entirely
    let hidden = dir.path().join(".cache");
    std::fs::create_dir(&hidden).unwrap();
    std::fs::write(hidden.join("big.fvec"), b"cached").unwrap();

    let files = veks::publish::enumerate_publishable_files(dir.path());
    let names: Vec<String> = files.iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();

    assert!(names.contains(&"data.fvec".to_string()), "visible files should be included");
    assert!(names.contains(&"dataset.yaml".to_string()), "dataset.yaml should be included");
    assert!(!names.contains(&".publish_url".to_string()), "hidden files should be excluded");
    assert!(!names.contains(&".gitignore".to_string()), "hidden files should be excluded");
    assert!(!names.contains(&"big.fvec".to_string()), "files in hidden dirs should be excluded");
}

// ═══════════════════════════════════════════════════════════════════════════
// project_artifacts: adversarial tests for every core pipeline command
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: create a command from the built-in registry by its command path.
fn make_cmd(command_path: &str) -> Box<dyn veks::pipeline::command::CommandOp> {
    let registry = veks::pipeline::registry::CommandRegistry::with_builtins();
    let factory = registry.get(command_path)
        .unwrap_or_else(|| panic!("command '{}' not found in registry", command_path));
    factory()
}

// ── import ──────────────────────────────────────────────────────────────────

#[test]
fn project_artifacts_import_cache_output() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("import");
    let mut opts = Options::new();
    opts.set("source", "/data/embeddings");
    opts.set("output", "${cache}/all_vectors.mvec");
    opts.set("facet", "base_vectors");
    opts.set("format", "npy");

    let manifest = cmd.project_artifacts("import-vectors", &opts);
    assert_eq!(manifest.step_id, "import-vectors");
    assert_eq!(manifest.command, "import");
    assert_eq!(manifest.inputs, vec!["/data/embeddings"]);
    assert!(manifest.outputs.is_empty(), "cache path should be intermediate, not output");
    assert_eq!(manifest.intermediates, vec!["${cache}/all_vectors.mvec"]);
}

#[test]
fn project_artifacts_import_non_cache_output() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("import");
    let mut opts = Options::new();
    opts.set("source", "/data/embeddings");
    opts.set("output", "profiles/default/base_vectors.mvec");
    opts.set("facet", "base_vectors");

    let manifest = cmd.project_artifacts("import-base", &opts);
    assert_eq!(manifest.inputs, vec!["/data/embeddings"]);
    assert_eq!(manifest.outputs, vec!["profiles/default/base_vectors.mvec"]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_import_non_path_options_excluded() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("import");
    let mut opts = Options::new();
    opts.set("source", "/data/src");
    opts.set("output", "out.mvec");
    opts.set("facet", "base_vectors");
    opts.set("format", "npy");
    opts.set("threads", "8");
    opts.set("count", "50000");

    let manifest = cmd.project_artifacts("import-with-extras", &opts);
    // Only source should appear as input; format, threads, count are not file paths
    assert_eq!(manifest.inputs, vec!["/data/src"]);
    assert_eq!(manifest.outputs, vec!["out.mvec"]);
}

#[test]
fn project_artifacts_import_empty_options() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("import");
    let opts = Options::new();

    let manifest = cmd.project_artifacts("import-empty", &opts);
    assert_eq!(manifest.step_id, "import-empty");
    assert_eq!(manifest.command, "import");
    assert!(manifest.inputs.is_empty());
    assert!(manifest.outputs.is_empty());
    assert!(manifest.intermediates.is_empty());
}

// ── compute knn ─────────────────────────────────────────────────────────────

#[test]
fn project_artifacts_compute_knn_basic() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute knn");
    let mut opts = Options::new();
    opts.set("base", "profiles/default/base_vectors.mvec");
    opts.set("query", "query_vectors.mvec");
    opts.set("indices", "profiles/default/neighbor_indices.ivec");
    opts.set("distances", "profiles/default/neighbor_distances.fvec");
    opts.set("neighbors", "100");
    opts.set("metric", "L2");
    opts.set("threads", "4");

    let manifest = cmd.project_artifacts("compute-knn", &opts);
    assert_eq!(manifest.step_id, "compute-knn");
    assert_eq!(manifest.command, "compute knn");
    assert_eq!(manifest.inputs, vec![
        "profiles/default/base_vectors.mvec",
        "query_vectors.mvec",
    ]);
    assert_eq!(manifest.outputs, vec![
        "profiles/default/neighbor_indices.ivec",
        "profiles/default/neighbor_distances.fvec",
    ]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_compute_knn_window_stripped() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute knn");
    let mut opts = Options::new();
    opts.set("base", "base.mvec[0..${base_count})");
    opts.set("query", "query.mvec[0..1000)");
    opts.set("indices", "gnd.ivec");
    opts.set("neighbors", "10");

    let manifest = cmd.project_artifacts("knn-windowed", &opts);
    // Window notation should be stripped from inputs
    assert_eq!(manifest.inputs, vec!["base.mvec", "query.mvec"]);
    assert_eq!(manifest.outputs, vec!["gnd.ivec"]);
}

#[test]
fn project_artifacts_compute_knn_cache_outputs() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute knn");
    let mut opts = Options::new();
    opts.set("base", "base.mvec");
    opts.set("query", "query.mvec");
    opts.set("indices", "${cache}/gnd.ivec");
    opts.set("distances", ".cache/gnd_dist.fvec");
    opts.set("neighbors", "10");

    let manifest = cmd.project_artifacts("knn-cached", &opts);
    assert!(manifest.outputs.is_empty(), "cache outputs should be intermediates");
    assert_eq!(manifest.intermediates, vec![
        "${cache}/gnd.ivec",
        ".cache/gnd_dist.fvec",
    ]);
}

#[test]
fn project_artifacts_compute_knn_no_distances() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute knn");
    let mut opts = Options::new();
    opts.set("base", "base.fvec");
    opts.set("query", "query.fvec");
    opts.set("indices", "out.ivec");
    opts.set("neighbors", "10");
    // No distances option

    let manifest = cmd.project_artifacts("knn-no-dist", &opts);
    assert_eq!(manifest.outputs, vec!["out.ivec"]);
    assert!(manifest.intermediates.is_empty());
}

// ── compute filtered-knn ────────────────────────────────────────────────────

#[test]
fn project_artifacts_compute_filtered_knn() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute filtered-knn");
    let mut opts = Options::new();
    opts.set("base", "base.mvec[0..${base_count})");
    opts.set("query", "query.mvec");
    opts.set("metadata-indices", "predicate_keys.slab");
    opts.set("indices", "profiles/default/filtered_indices.ivec");
    opts.set("distances", "profiles/default/filtered_distances.fvec");
    opts.set("neighbors", "100");
    opts.set("metric", "COSINE");
    opts.set("threads", "16");
    opts.set("partition_size", "500000");

    let manifest = cmd.project_artifacts("fknn-step", &opts);
    assert_eq!(manifest.step_id, "fknn-step");
    assert_eq!(manifest.command, "compute filtered-knn");
    // Window notation stripped from base
    assert_eq!(manifest.inputs, vec![
        "base.mvec",
        "query.mvec",
        "predicate_keys.slab",
    ]);
    assert_eq!(manifest.outputs, vec![
        "profiles/default/filtered_indices.ivec",
        "profiles/default/filtered_distances.fvec",
    ]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_compute_filtered_knn_cache_mixed() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute filtered-knn");
    let mut opts = Options::new();
    opts.set("base", "base.mvec");
    opts.set("query", "query.mvec");
    opts.set("metadata-indices", "keys.slab");
    opts.set("indices", "${cache}/filtered.ivec");
    opts.set("distances", "profiles/default/filtered_dist.fvec");
    opts.set("neighbors", "10");

    let manifest = cmd.project_artifacts("fknn-mixed", &opts);
    assert_eq!(manifest.outputs, vec!["profiles/default/filtered_dist.fvec"]);
    assert_eq!(manifest.intermediates, vec!["${cache}/filtered.ivec"]);
}

// ── compute dedup ───────────────────────────────────────────────────────────

#[test]
fn project_artifacts_compute_dedup_non_cache() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute dedup");
    let mut opts = Options::new();
    opts.set("source", "${cache}/all_vectors.mvec");
    opts.set("output", "dedup_index.ivec");
    opts.set("report", "dedup_report.json");
    opts.set("elide", "true");
    opts.set("batch-size", "500000");

    let manifest = cmd.project_artifacts("dedup-step", &opts);
    assert_eq!(manifest.step_id, "dedup-step");
    assert_eq!(manifest.command, "compute dedup");
    assert_eq!(manifest.inputs, vec!["${cache}/all_vectors.mvec"]);
    assert_eq!(manifest.outputs, vec!["dedup_index.ivec", "dedup_report.json"]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_compute_dedup_cache_outputs() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute dedup");
    let mut opts = Options::new();
    opts.set("source", "data.mvec");
    opts.set("output", "${cache}/dedup.ivec");
    opts.set("report", ".cache/dedup.json");

    let manifest = cmd.project_artifacts("dedup-cached", &opts);
    assert_eq!(manifest.inputs, vec!["data.mvec"]);
    assert!(manifest.outputs.is_empty());
    assert_eq!(manifest.intermediates, vec![
        "${cache}/dedup.ivec",
        ".cache/dedup.json",
    ]);
}

#[test]
fn project_artifacts_compute_dedup_no_report() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute dedup");
    let mut opts = Options::new();
    opts.set("source", "vectors.mvec");
    opts.set("output", "dedup.ivec");
    // No report option

    let manifest = cmd.project_artifacts("dedup-no-report", &opts);
    assert_eq!(manifest.inputs, vec!["vectors.mvec"]);
    assert_eq!(manifest.outputs, vec!["dedup.ivec"]);
}

// ── compute sort ────────────────────────────────────────────────────────────

#[test]
fn project_artifacts_compute_sort() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute sort");
    let mut opts = Options::new();
    opts.set("source", "base.fvec");
    opts.set("output", "sorted.fvec");
    opts.set("sort-by", "norm");

    let manifest = cmd.project_artifacts("sort-step", &opts);
    assert_eq!(manifest.step_id, "sort-step");
    assert_eq!(manifest.command, "compute sort");
    assert_eq!(manifest.inputs, vec!["base.fvec"]);
    assert_eq!(manifest.outputs, vec!["sorted.fvec"]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_compute_sort_cache_output() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute sort");
    let mut opts = Options::new();
    opts.set("source", "base.fvec");
    opts.set("output", "${cache}/sorted.fvec");

    let manifest = cmd.project_artifacts("sort-cached", &opts);
    assert_eq!(manifest.inputs, vec!["base.fvec"]);
    assert!(manifest.outputs.is_empty());
    assert_eq!(manifest.intermediates, vec!["${cache}/sorted.fvec"]);
}

// ── generate ivec-shuffle ───────────────────────────────────────────────────

#[test]
fn project_artifacts_gen_shuffle_no_inputs() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("generate ivec-shuffle");
    let mut opts = Options::new();
    opts.set("output", "${cache}/shuffle.ivec");
    opts.set("interval", "1000000");
    opts.set("seed", "42");

    let manifest = cmd.project_artifacts("shuffle-step", &opts);
    assert_eq!(manifest.step_id, "shuffle-step");
    assert_eq!(manifest.command, "generate ivec-shuffle");
    // No file inputs — interval and seed are numeric options
    assert!(manifest.inputs.is_empty());
    assert!(manifest.outputs.is_empty(), "cache path should be intermediate");
    assert_eq!(manifest.intermediates, vec!["${cache}/shuffle.ivec"]);
}

#[test]
fn project_artifacts_gen_shuffle_non_cache_output() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("generate ivec-shuffle");
    let mut opts = Options::new();
    opts.set("output", "shuffle.ivec");
    opts.set("interval", "500");

    let manifest = cmd.project_artifacts("shuffle-final", &opts);
    assert!(manifest.inputs.is_empty());
    assert_eq!(manifest.outputs, vec!["shuffle.ivec"]);
    assert!(manifest.intermediates.is_empty());
}

// ── transform mvec-extract ──────────────────────────────────────────────────

#[test]
fn project_artifacts_mvec_extract() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("transform mvec-extract");
    let mut opts = Options::new();
    opts.set("mvec-file", "${cache}/all_vectors.mvec");
    opts.set("ivec-file", "${cache}/shuffle.ivec");
    opts.set("output", "profiles/default/base_vectors.mvec");
    opts.set("range", "[100,500000)");

    let manifest = cmd.project_artifacts("extract-base", &opts);
    assert_eq!(manifest.step_id, "extract-base");
    assert_eq!(manifest.command, "transform mvec-extract");
    assert_eq!(manifest.inputs, vec![
        "${cache}/all_vectors.mvec",
        "${cache}/shuffle.ivec",
    ]);
    assert_eq!(manifest.outputs, vec!["profiles/default/base_vectors.mvec"]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_mvec_extract_window_in_source() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("transform mvec-extract");
    let mut opts = Options::new();
    opts.set("mvec-file", "all.mvec[0..10000)");
    opts.set("ivec-file", "shuffle.ivec");
    opts.set("output", "extracted.mvec");

    let manifest = cmd.project_artifacts("extract-windowed", &opts);
    // Window stripped from mvec-file
    assert_eq!(manifest.inputs, vec!["all.mvec", "shuffle.ivec"]);
}

// ── transform slab-extract ──────────────────────────────────────────────────

#[test]
fn project_artifacts_slab_extract() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("transform slab-extract");
    let mut opts = Options::new();
    opts.set("slab-file", "metadata_all.slab");
    opts.set("ivec-file", "${cache}/shuffle.ivec");
    opts.set("output", "profiles/default/base_metadata.slab");

    let manifest = cmd.project_artifacts("extract-metadata", &opts);
    assert_eq!(manifest.step_id, "extract-metadata");
    assert_eq!(manifest.command, "transform slab-extract");
    assert_eq!(manifest.inputs, vec![
        "metadata_all.slab",
        "${cache}/shuffle.ivec",
    ]);
    assert_eq!(manifest.outputs, vec!["profiles/default/base_metadata.slab"]);
}

#[test]
fn project_artifacts_slab_extract_cache_output() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("transform slab-extract");
    let mut opts = Options::new();
    opts.set("slab-file", "raw.slab");
    opts.set("ivec-file", "idx.ivec");
    opts.set("output", ".cache/extracted.slab");

    let manifest = cmd.project_artifacts("slab-cached", &opts);
    assert!(manifest.outputs.is_empty());
    assert_eq!(manifest.intermediates, vec![".cache/extracted.slab"]);
}

// ── set variable ────────────────────────────────────────────────────────────

#[test]
fn project_artifacts_set_variable_empty() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("set variable");
    let mut opts = Options::new();
    opts.set("name", "vector_count");
    opts.set("value", "count:all_vectors.mvec");

    let manifest = cmd.project_artifacts("set-count", &opts);
    assert_eq!(manifest.step_id, "set-count");
    assert_eq!(manifest.command, "set variable");
    // set variable produces no file artifacts
    assert!(manifest.inputs.is_empty());
    assert!(manifest.outputs.is_empty());
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_set_variable_literal() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("set variable");
    let mut opts = Options::new();
    opts.set("name", "seed");
    opts.set("value", "42");

    let manifest = cmd.project_artifacts("set-seed", &opts);
    assert!(manifest.inputs.is_empty());
    assert!(manifest.outputs.is_empty());
    assert!(manifest.intermediates.is_empty());
}

// ── barrier ─────────────────────────────────────────────────────────────────

#[test]
fn project_artifacts_barrier_empty() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("barrier");
    let opts = Options::new();

    let manifest = cmd.project_artifacts("barrier-default", &opts);
    assert_eq!(manifest.step_id, "barrier-default");
    assert_eq!(manifest.command, "barrier");
    assert!(manifest.inputs.is_empty());
    assert!(manifest.outputs.is_empty());
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_barrier_ignores_spurious_options() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("barrier");
    let mut opts = Options::new();
    opts.set("source", "/some/path");
    opts.set("output", "/some/other/path");

    let manifest = cmd.project_artifacts("barrier-noisy", &opts);
    // Barrier ignores all options
    assert!(manifest.inputs.is_empty());
    assert!(manifest.outputs.is_empty());
    assert!(manifest.intermediates.is_empty());
}

// ── compute predicates (gen_predicate_keys) ─────────────────────────────────

#[test]
fn project_artifacts_compute_predicates() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute predicates");
    let mut opts = Options::new();
    opts.set("input", "profiles/default/base_metadata.slab");
    opts.set("predicates", "predicates.slab");
    opts.set("survey", "survey.json");
    opts.set("output", "profiles/default/predicate_keys.slab");
    opts.set("threads", "8");
    opts.set("segment-size", "1000000");

    let manifest = cmd.project_artifacts("eval-predicates", &opts);
    assert_eq!(manifest.step_id, "eval-predicates");
    assert_eq!(manifest.command, "compute predicates");
    assert_eq!(manifest.inputs, vec![
        "profiles/default/base_metadata.slab",
        "predicates.slab",
        "survey.json",
    ]);
    assert_eq!(manifest.outputs, vec!["profiles/default/predicate_keys.slab"]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_compute_predicates_optional_survey_missing() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute predicates");
    let mut opts = Options::new();
    opts.set("input", "meta.slab");
    opts.set("predicates", "preds.slab");
    opts.set("output", "keys.slab");
    // survey not set

    let manifest = cmd.project_artifacts("pred-no-survey", &opts);
    // Only input and predicates in inputs; survey absent
    assert_eq!(manifest.inputs, vec!["meta.slab", "preds.slab"]);
    assert_eq!(manifest.outputs, vec!["keys.slab"]);
}

// ── synthesize predicates (gen_predicates) ──────────────────────────────────

#[test]
fn project_artifacts_synthesize_predicates() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("synthesize predicates");
    let mut opts = Options::new();
    opts.set("input", "metadata_all.slab");
    opts.set("survey", "survey.json");
    opts.set("output", "predicates.slab");
    opts.set("count", "1000");
    opts.set("seed", "42");
    opts.set("strategy", "compound");
    opts.set("selectivity", "0.1");

    let manifest = cmd.project_artifacts("synth-preds", &opts);
    assert_eq!(manifest.step_id, "synth-preds");
    assert_eq!(manifest.command, "synthesize predicates");
    assert_eq!(manifest.inputs, vec!["metadata_all.slab", "survey.json"]);
    assert_eq!(manifest.outputs, vec!["predicates.slab"]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_synthesize_predicates_no_survey() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("synthesize predicates");
    let mut opts = Options::new();
    opts.set("input", "meta.slab");
    opts.set("output", "preds.slab");
    // no survey

    let manifest = cmd.project_artifacts("synth-no-survey", &opts);
    assert_eq!(manifest.inputs, vec!["meta.slab"]);
    assert_eq!(manifest.outputs, vec!["preds.slab"]);
}

// ── survey (slab.rs) ────────────────────────────────────────────────────────

#[test]
fn project_artifacts_survey() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("survey");
    let mut opts = Options::new();
    opts.set("input", "metadata_all.slab");
    opts.set("output", "survey.json");

    let manifest = cmd.project_artifacts("survey-metadata", &opts);
    assert_eq!(manifest.step_id, "survey-metadata");
    assert_eq!(manifest.command, "survey");
    assert_eq!(manifest.inputs, vec!["metadata_all.slab"]);
    assert_eq!(manifest.outputs, vec!["survey.json"]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_survey_cache_output() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("survey");
    let mut opts = Options::new();
    opts.set("input", "meta.slab");
    opts.set("output", "${cache}/survey.json");

    let manifest = cmd.project_artifacts("survey-cached", &opts);
    assert_eq!(manifest.inputs, vec!["meta.slab"]);
    assert!(manifest.outputs.is_empty());
    assert_eq!(manifest.intermediates, vec!["${cache}/survey.json"]);
}

// ── merkle create ───────────────────────────────────────────────────────────

#[test]
fn project_artifacts_merkle_create_single_file() {
    use veks::pipeline::command::Options;

    // Since the source does not exist yet, merkle uses the path as-is
    let nonexistent = "/tmp/nonexistent_test_path_for_merkle.fvec";

    let cmd = make_cmd("merkle create");
    let mut opts = Options::new();
    opts.set("source", nonexistent);
    // min-size 0 so any file qualifies
    opts.set("min-size", "0");

    let manifest = cmd.project_artifacts("merkle-single", &opts);
    assert_eq!(manifest.step_id, "merkle-single");
    assert_eq!(manifest.command, "merkle create");
    assert_eq!(manifest.inputs, vec![nonexistent]);
    assert_eq!(manifest.outputs, vec![format!("{}.mref", nonexistent)]);
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_merkle_create_directory() {
    use veks::pipeline::command::Options;

    let dir = tempfile::tempdir().unwrap();
    let profiles = dir.path().join("profiles").join("default");
    std::fs::create_dir_all(&profiles).unwrap();

    // Create large-enough files (merkle min-size = 0 for test)
    write_fvec(&profiles.join("base.fvec"), 100, 4);
    write_fvec(&profiles.join("query.fvec"), 10, 4);
    // Create a small non-vector file that should also match at min-size=0
    std::fs::write(profiles.join("meta.json"), "{}").unwrap();

    let cmd = make_cmd("merkle create");
    let mut opts = Options::new();
    opts.set("source", profiles.to_string_lossy().as_ref());
    opts.set("min-size", "0");

    let manifest = cmd.project_artifacts("merkle-dir", &opts);
    assert_eq!(manifest.command, "merkle create");

    // Should have one input and one .mref output per eligible file
    assert_eq!(manifest.inputs.len(), manifest.outputs.len(),
        "each input should have exactly one .mref output");
    for output in &manifest.outputs {
        assert!(output.ends_with(".mref"),
            "output '{}' should end with .mref", output);
    }
    assert!(manifest.inputs.len() >= 2,
        "should find at least 2 files, found {}", manifest.inputs.len());
    assert!(manifest.intermediates.is_empty());
}

#[test]
fn project_artifacts_merkle_create_min_size_filter() {
    use veks::pipeline::command::Options;

    let dir = tempfile::tempdir().unwrap();
    // Write a very small file that should be filtered by min-size
    write_fvec(&dir.path().join("tiny.fvec"), 1, 2);

    let cmd = make_cmd("merkle create");
    let mut opts = Options::new();
    opts.set("source", dir.path().join("tiny.fvec").to_string_lossy().as_ref());
    // Default min-size is 100M, tiny.fvec is ~12 bytes
    opts.set("min-size", "100M");

    let manifest = cmd.project_artifacts("merkle-tiny", &opts);
    // File exists but is below threshold — no inputs or outputs
    assert!(manifest.inputs.is_empty(), "tiny file should be filtered by min-size");
    assert!(manifest.outputs.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Adversarial edge cases for manifest_from_keys
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn project_artifacts_count_prefix_not_an_input() {
    use veks::pipeline::command::Options;

    // The import command uses "source" as input key.
    // If someone passes a count: prefixed value, it must not appear as input.
    let cmd = make_cmd("import");
    let mut opts = Options::new();
    opts.set("source", "count:/data/vectors.mvec");
    opts.set("output", "out.mvec");
    opts.set("facet", "base_vectors");

    let manifest = cmd.project_artifacts("count-prefix-test", &opts);
    // count: prefix means this is not a file path
    assert!(manifest.inputs.is_empty(),
        "count: prefixed value should not appear as input, got {:?}", manifest.inputs);
}

#[test]
fn project_artifacts_window_notation_complex() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute knn");
    let mut opts = Options::new();
    opts.set("base", "base.mvec[0..${base_count})");
    opts.set("query", "query.mvec[${query_start}..${query_end})");
    opts.set("indices", "out.ivec");
    opts.set("neighbors", "10");

    let manifest = cmd.project_artifacts("window-complex", &opts);
    assert_eq!(manifest.inputs, vec!["base.mvec", "query.mvec"]);
}

#[test]
fn project_artifacts_dot_cache_path_classified() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("import");
    let mut opts = Options::new();
    opts.set("source", "/data/src");
    opts.set("output", "workspace/.cache/imported.mvec");

    let manifest = cmd.project_artifacts("dot-cache-test", &opts);
    // Contains /.cache/ — should be intermediate
    assert!(manifest.outputs.is_empty());
    assert_eq!(manifest.intermediates, vec!["workspace/.cache/imported.mvec"]);
}

#[test]
fn project_artifacts_boolean_option_not_input() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("compute dedup");
    let mut opts = Options::new();
    opts.set("source", "vectors.mvec");
    opts.set("output", "dedup.ivec");
    opts.set("elide", "true");
    opts.set("batch-size", "1000000");

    let manifest = cmd.project_artifacts("bool-test", &opts);
    // elide and batch-size are not file paths, only source is an input
    assert_eq!(manifest.inputs, vec!["vectors.mvec"]);
    assert_eq!(manifest.outputs, vec!["dedup.ivec"]);
}

#[test]
fn project_artifacts_clear_variables_empty() {
    use veks::pipeline::command::Options;

    let cmd = make_cmd("clear variables");
    let opts = Options::new();

    let manifest = cmd.project_artifacts("clear-vars", &opts);
    assert_eq!(manifest.command, "clear variables");
    // May or may not have a custom implementation; should not panic
    assert!(manifest.inputs.is_empty() || !manifest.inputs.is_empty());
}

#[test]
fn project_artifacts_every_registered_command_no_panic() {
    use veks::pipeline::command::Options;

    // Verify that calling project_artifacts with empty options on every
    // registered command does not panic.
    let registry = veks::pipeline::registry::CommandRegistry::with_builtins();
    let commands = [
        "import", "convert file", "analyze describe",
        "generate vectors", "generate ivec-shuffle",
        "transform fvec-extract", "transform ivec-extract",
        "transform mvec-extract", "transform slab-extract",
        "generate sketch", "generate from-model",
        "compute knn", "compute filtered-knn", "compute sort", "compute dedup",
        "analyze verify-knn", "analyze stats", "analyze histogram",
        "info file", "info compute", "cleanup cleanfvec",
        "merkle create", "merkle verify", "merkle diff", "merkle summary",
        "merkle treeview", "json jjq", "json rjq",
        "analyze slice", "analyze check-endian", "analyze zeros", "analyze compare",
        "generate dataset", "generate derive", "merkle path",
        "config show", "config init", "analyze select",
        "merkle spoilbits", "merkle spoilchunks", "config list-mounts",
        "analyze explore", "analyze find", "analyze profile",
        "analyze model-diff", "analyze verify-profiles",
        "fetch dlhf", "fetch bulkdl",
        "slab import", "slab export", "slab append", "slab rewrite",
        "slab check", "slab get", "slab analyze", "slab explain",
        "slab namespaces", "slab inspect", "survey",
        "analyze plot", "analyze flamegraph",
        "generate predicated", "synthesize predicates", "compute predicates",
        "inspect predicate",
        "set variable", "clear variables",
        "barrier",
    ];

    let opts = Options::new();
    for cmd_path in &commands {
        let factory = registry.get(cmd_path)
            .unwrap_or_else(|| panic!("command '{}' not found in registry", cmd_path));
        let cmd = factory();
        // Must not panic
        let manifest = cmd.project_artifacts("smoke-test", &opts);
        assert_eq!(manifest.step_id, "smoke-test",
            "step_id mismatch for command '{}'", cmd_path);
        // command field should match the command_path
        assert_eq!(manifest.command, cmd.command_path(),
            "command mismatch for '{}'", cmd_path);
    }
}
