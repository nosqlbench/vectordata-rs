// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end slice of the S2OA passage pilot
//! (`docs/design/scaleup_pvs/s2oa-passage-pilot-plan.md`):
//!
//! synthetic s2orc shards → `veks pipeline generate passages` →
//! aligned synthetic vectors → `veks pipeline verify alignment` →
//! `veks prepare bootstrap` (BQGD, Cosine, self-search) → `veks run` →
//! `veks check --check-integrity`
//!
//! Everything runs through the real binary (the CLI↔yaml option mirror is
//! part of what this exercises); numerical verification of the chunked
//! output goes through the veks-core passage-table reader.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use veks_core::formats::passage_table::{read_parents, read_passages};

fn veks_bin() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_veks") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/debug/veks")
}

fn make_tempdir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).expect("create target/tmp");
    tempfile::tempdir_in(&base).expect("create tempdir")
}

/// One synthetic s2orc record: `sections` sections × `paras` paragraphs of
/// `words` words each, with annotation arrays JSON-encoded as strings the
/// way s2orc ships them.
fn synth_record(corpusid: i64, sections: usize, paras: usize, words: usize) -> String {
    let mut text = String::new();
    let mut header_spans = Vec::new();
    let mut para_spans = Vec::new();
    for s in 0..sections {
        let header = format!("Section{}", s);
        let start = text.chars().count();
        text.push_str(&header);
        header_spans
            .push(serde_json::json!({"start": start, "end": start + header.chars().count()}));
        text.push('\n');
        for p in 0..paras {
            let body = (0..words)
                .map(|w| format!("c{}s{}p{}w{}", corpusid, s, p, w))
                .collect::<Vec<_>>()
                .join(" ");
            let start = text.chars().count();
            text.push_str(&body);
            para_spans
                .push(serde_json::json!({"start": start, "end": start + body.chars().count()}));
            text.push('\n');
        }
    }
    serde_json::json!({
        "corpusid": corpusid,
        "content": {
            "text": text,
            "annotations": {
                "paragraph": serde_json::Value::Array(para_spans).to_string(),
                "sectionheader": serde_json::Value::Array(header_spans).to_string(),
            }
        }
    })
    .to_string()
}

fn write_gz_shard(path: &Path, records: &[String]) {
    let file = std::fs::File::create(path).expect("create shard");
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    for record in records {
        writeln!(gz, "{}", record).expect("write record");
    }
    gz.finish().expect("finish gz");
}

/// Write a C-order `<f4` npy of shape [rows, dim] with distinct, nonzero,
/// deterministic rows (so dedup/zero-scan remove nothing).
fn write_npy(path: &Path, rows: usize, dim: usize) {
    let header_body = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows, dim
    );
    let unpadded = 10 + header_body.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let header = format!("{}{}\n", header_body, " ".repeat(padding));
    let mut f = std::fs::File::create(path).expect("create npy");
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for r in 0..rows {
        for d in 0..dim {
            let v = 1.0f32 + ((r * 31 + d * 7) % 97) as f32 + (r as f32) * 0.001;
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

fn run_veks(args: &[&str], cwd: &Path) -> std::process::Output {
    Command::new(veks_bin())
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("spawn veks")
}

fn assert_success(output: &std::process::Output, what: &str) {
    assert!(
        output.status.success(),
        "{} failed\nstdout:\n{}\nstderr:\n{}",
        what,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn passages_pilot_end_to_end() {
    let tmp = make_tempdir();
    let work = tmp.path();

    // ── Fixture: 25 docs across two gz shards, ids interleaved ──────────
    let shards = work.join("shards");
    std::fs::create_dir_all(&shards).unwrap();
    let (mut recs_a, mut recs_b) = (Vec::new(), Vec::new());
    for id in 0..25i64 {
        // 2 sections × 2 paragraphs × 4 words: paragraphs pack pairwise
        // (8 ≤ max-words 8) → exactly 2 passages per document.
        let record = synth_record(id, 2, 2, 4);
        if id % 2 == 0 {
            recs_a.push(record);
        } else {
            recs_b.push(record);
        }
    }
    // One no-body record that must not count against doc-limit.
    recs_b.push(serde_json::json!({"corpusid": 500, "content": {"text": ""}}).to_string());
    write_gz_shard(&shards.join("part-a.jsonl.gz"), &recs_a);
    write_gz_shard(&shards.join("part-b.jsonl.gz"), &recs_b);

    // ── Stage: generate passages (doc-limit 20 = ids 0..19) ─────────────
    let passages = work.join("upstream/passages.parquet");
    let parents = work.join("upstream/parents.parquet");
    let out = run_veks(
        &[
            "pipeline",
            "generate",
            "passages",
            "--source",
            shards.to_str().unwrap(),
            "--output",
            passages.to_str().unwrap(),
            "--doc-limit",
            "20",
            "--min-words",
            "3",
            "--target-words",
            "6",
            "--max-words",
            "8",
        ],
        work,
    );
    assert_success(&out, "generate passages");

    let passage_rows = read_passages(&passages).expect("read passages");
    let parent_rows = read_parents(&parents).expect("read parents");
    assert_eq!(parent_rows.len(), 20, "doc-limit selects 20 parents");
    assert_eq!(
        parent_rows.iter().map(|p| p.corpusid).collect::<Vec<_>>(),
        (0..20i64).collect::<Vec<_>>(),
        "lowest corpusids in ascending order"
    );
    assert_eq!(passage_rows.len(), 40, "2 passages per parent");
    // Parent blocks tile the row space exactly.
    let mut row = 0i64;
    for parent in &parent_rows {
        assert_eq!(parent.row_start, row);
        for passage in &passage_rows[row as usize..(row + parent.passage_count as i64) as usize] {
            assert_eq!(passage.corpusid, parent.corpusid);
        }
        row += parent.passage_count as i64;
    }
    assert_eq!(row, passage_rows.len() as i64);

    // ── Stage: embed contract (synthetic) + verify alignment ────────────
    let vectors = work.join("upstream/base_all.npy");
    write_npy(&vectors, passage_rows.len(), 8);
    let out = run_veks(
        &[
            "pipeline",
            "verify",
            "alignment",
            "--source",
            vectors.to_str().unwrap(),
            "--reference",
            passages.to_str().unwrap(),
            "--dim",
            "8",
        ],
        work,
    );
    assert_success(&out, "verify alignment");

    // Negative control: a truncated vectors artifact must fail the gate.
    let truncated = work.join("upstream/truncated.npy");
    write_npy(&truncated, passage_rows.len() - 1, 8);
    let out = run_veks(
        &[
            "pipeline",
            "verify",
            "alignment",
            "--source",
            truncated.to_str().unwrap(),
            "--reference",
            passages.to_str().unwrap(),
        ],
        work,
    );
    assert!(
        !out.status.success(),
        "misaligned vectors must fail verify alignment\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );

    // ── Stage: bootstrap (BQGD, Cosine, self-search) + run + check ──────
    let dataset = work.join("dataset");
    let out = run_veks(
        &[
            "prepare",
            "bootstrap",
            "--name",
            "passages-pilot-e2e",
            "--output",
            dataset.to_str().unwrap(),
            "--base-vectors",
            vectors.to_str().unwrap(),
            "--self-search",
            "--query-count",
            "10",
            "--metric",
            "Cosine",
            "--neighbors",
            "5",
            "--seed",
            "42",
            "--required-facets",
            "BQGD",
            "--force",
        ],
        work,
    );
    assert_success(&out, "prepare bootstrap");
    let dataset_yaml = dataset.join("dataset.yaml");
    assert!(dataset_yaml.exists(), "bootstrap wrote dataset.yaml");

    let out = run_veks(
        &[
            "run",
            "--output",
            "batch",
            "--threads",
            "2",
            dataset_yaml.to_str().unwrap(),
        ],
        work,
    );
    assert_success(&out, "veks run");

    let out = run_veks(
        &["check", dataset.to_str().unwrap(), "--check-integrity"],
        work,
    );
    assert_success(&out, "veks check --check-integrity");

    // ── Numerical spot checks on the produced facets ────────────────────
    // 40 vectors − 10 self-search queries = 30 base vectors (all rows are
    // distinct and nonzero, so cleaning removes none).
    let base = dataset.join("profiles/base/base_vectors.fvecs");
    assert!(base.exists(), "base facet exists at {}", base.display());
    let bytes = std::fs::read(&base).unwrap();
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(dim, 8);
    let stride = 4 + 8 * 4;
    assert_eq!(bytes.len() % stride, 0, "whole fvecs records");
    assert_eq!(bytes.len() / stride, 30, "40 passages − 10 queries");
}

#[test]
fn generate_passages_emit_yaml_mirrors_options() {
    // The CLI↔yaml congruence rule: an ad hoc invocation with --emit-yaml
    // prints a paste-ready step whose keys are exactly the option names.
    let tmp = make_tempdir();
    let out = run_veks(
        &[
            "pipeline",
            "generate",
            "passages",
            "--source",
            "shards",
            "--output",
            "upstream/passages.parquet",
            "--doc-limit",
            "1000",
            "--emit-yaml",
        ],
        tmp.path(),
    );
    assert_success(&out, "generate passages --emit-yaml");
    let yaml = String::from_utf8_lossy(&out.stdout);
    assert!(yaml.contains("run: generate passages"), "yaml:\n{}", yaml);
    assert!(yaml.contains("source: shards"), "yaml:\n{}", yaml);
    assert!(yaml.contains("doc-limit: '1000'") || yaml.contains("doc-limit: 1000"), "yaml:\n{}", yaml);
}
