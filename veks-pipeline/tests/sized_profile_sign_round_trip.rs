// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration test: multiple sized profiles compute → verify
//! through the production code path that exercises Path-2 sibling segment
//! reuse for IP-style metrics.
//!
//! This is the regression net for the sign-convention bug we hit:
//! `compute knn-blas` writes per-profile `neighbor_distances.fvecs`, then a
//! larger profile loads its smaller sibling's fvec via `scan_cached_segments`
//! Path-2 reuse. If either side gets the FAISS publication-vs-kernel sign
//! convention wrong, the larger profile's heap merge silently corrupts (it
//! accumulates worst-of-k instead of best-of-k) and recall collapses to 0
//! at the third profile rung onward.
//!
//! Running 3+ sized profiles for `DOT_PRODUCT` is the minimum fixture that
//! triggers Path-2 reuse + a metric where the sign actually flips. Verify
//! must report `recall@k = 1.0` for every profile, including the largest.

#![cfg(feature = "knnutils")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use veks_core::ui::{TestSink, UiHandle};
use veks_pipeline::pipeline::command::{Options, Status, StreamContext};
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::registry::CommandRegistry;
use veks_pipeline::pipeline::resource::ResourceGovernor;

fn test_ctx(dir: &Path) -> StreamContext {
    StreamContext {
        dataset_name: "sign-rtt".into(),
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

/// Write an fvec file with `count` deterministic gaussian-normalized vectors
/// (xorshift PRNG seeded for repeatability).
fn write_normalized_fvec(path: &Path, count: usize, dim: usize, seed: u64) {
    use std::io::Write;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    let mut rng = seed.max(1);
    let mut next = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        // Map xorshift to roughly N(0, 1) via Box-Muller-ish: use two uniforms.
        let u1 = (rng as f32 / u64::MAX as f32).clamp(1e-9, 1.0);
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let u2 = rng as f32 / u64::MAX as f32;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    };
    for _ in 0..count {
        let dim_i32 = dim as i32;
        w.write_all(&dim_i32.to_le_bytes()).unwrap();
        let raw: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(f32::MIN_POSITIVE);
        for v in &raw {
            w.write_all(&(v / norm).to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
}

/// Run `compute knn-blas` against a windowed slice of the base file, writing
/// to the profile's standard `profiles/<name>/neighbor_*.{ivecs,fvecs}` paths
/// — exactly the layout the verifier and the next-larger profile expect.
fn run_compute_blas(
    workspace: &Path,
    base_path: &Path,
    query_path: &Path,
    profile_name: &str,
    base_count: usize,
    k: usize,
    metric: &str,
) {
    let prof_dir = workspace.join(format!("profiles/{}", profile_name));
    std::fs::create_dir_all(&prof_dir).unwrap();
    let indices_path = prof_dir.join("neighbor_indices.ivecs");
    let distances_path = prof_dir.join("neighbor_distances.fvecs");

    let mut ctx = test_ctx(workspace);
    let registry = CommandRegistry::with_builtins();
    let mut op = registry.get("compute knn-blas")
        .expect("compute knn-blas should be registered with knnutils feature")();
    let mut opts = Options::new();
    opts.set("base", format!("{}[0..{})", base_path.display(), base_count));
    opts.set("query", query_path.to_string_lossy().to_string());
    opts.set("indices", indices_path.to_string_lossy().to_string());
    opts.set("distances", distances_path.to_string_lossy().to_string());
    opts.set("neighbors", k.to_string());
    opts.set("metric", metric.to_string());

    let r = op.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok,
        "compute knn-blas failed for profile {}: {}", profile_name, r.message);
}

/// Multi-profile compute → verify cycle for DOT_PRODUCT, the metric where
/// the sign convention actually flips between heap-internal (`-dot`) and
/// on-disk publication (`+dot`).
///
/// Four sized profiles ensure Path-2 sibling reuse fires: profile 100 has
/// no sibling, profile 200 reuses 100, profile 500 reuses 200, profile
/// 1000 reuses 500. Any sign-handling regression silently breaks the
/// merge for the larger profiles → recall < 1.0 → test fails.
#[test]
fn dot_product_multi_profile_compute_verify_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    let base_path = workspace.join("base.fvec");
    let query_path = workspace.join("query.fvec");

    let dim = 64;
    let n_base = 1000;
    let n_query = 30;
    let k = 10;

    write_normalized_fvec(&base_path, n_base, dim, 42);
    write_normalized_fvec(&query_path, n_query, dim, 137);

    // Compute profiles in ascending size order. Each later profile
    // discovers and reuses its smaller sibling's published fvec via
    // Path-2; that's the load-bearing path for the sign-convention
    // round-trip we're guarding.
    for &(name, bc) in &[("100", 100usize), ("200", 200), ("500", 500), ("default", 1000)] {
        run_compute_blas(workspace, &base_path, &query_path,
                          name, bc, k, "DOT_PRODUCT");
    }

    // Minimal dataset.yaml so verify can discover the profiles.
    let yaml = format!(
        "name: sign-rtt\n\
         distance_function: DOT_PRODUCT\n\
         profiles:\n\
         \x20 default:\n\
         \x20   base_vectors: base.fvec\n\
         \x20   query_vectors: query.fvec\n\
         \x20   neighbor_indices: profiles/default/neighbor_indices.ivecs\n\
         \x20   neighbor_distances: profiles/default/neighbor_distances.fvecs\n\
         \x20   base_count: {}\n\
         \x20 100:\n\
         \x20   base_vectors: base.fvec[0..100)\n\
         \x20   query_vectors: query.fvec\n\
         \x20   neighbor_indices: profiles/100/neighbor_indices.ivecs\n\
         \x20   neighbor_distances: profiles/100/neighbor_distances.fvecs\n\
         \x20   base_count: 100\n\
         \x20 200:\n\
         \x20   base_vectors: base.fvec[0..200)\n\
         \x20   query_vectors: query.fvec\n\
         \x20   neighbor_indices: profiles/200/neighbor_indices.ivecs\n\
         \x20   neighbor_distances: profiles/200/neighbor_distances.fvecs\n\
         \x20   base_count: 200\n\
         \x20 500:\n\
         \x20   base_vectors: base.fvec[0..500)\n\
         \x20   query_vectors: query.fvec\n\
         \x20   neighbor_indices: profiles/500/neighbor_indices.ivecs\n\
         \x20   neighbor_distances: profiles/500/neighbor_distances.fvecs\n\
         \x20   base_count: 500\n",
        n_base);
    std::fs::write(workspace.join("dataset.yaml"), yaml).unwrap();

    // Run verify-knn-consolidated. It does its own scan via sgemm and
    // compares against the published `neighbor_indices.ivecs` per
    // profile. Any sign flip in the engine output OR in Path-2 reuse
    // shows up here as recall < 1.0.
    let report_path = workspace.join("verify-report.json");
    let mut ctx = test_ctx(workspace);
    let registry = CommandRegistry::with_builtins();
    let mut op = registry.get("verify knn-consolidated").unwrap()();
    let mut opts = Options::new();
    opts.set("base", base_path.to_string_lossy().to_string());
    opts.set("query", query_path.to_string_lossy().to_string());
    opts.set("metric", "DOT_PRODUCT");
    opts.set("sample", n_query.to_string());
    opts.set("output", report_path.to_string_lossy().to_string());
    let r = op.execute(&opts, &mut ctx);
    assert_eq!(r.status, Status::Ok,
        "verify knn-consolidated failed: {}", r.message);

    // Parse report and assert every profile got recall@k = 1.0. The
    // exact-match requirement is intentional — even one boundary swap
    // would indicate a numerical drift the rerank should be hiding,
    // and at this fixture size (1k base × 30 query × dim=64) the f64
    // canonical rerank should be deterministically perfect.
    let report_str = std::fs::read_to_string(&report_path)
        .expect("verify should write a JSON report");
    let report: serde_json::Value = serde_json::from_str(&report_str)
        .expect("report is valid JSON");
    let profiles = report["profiles"].as_array()
        .expect("report has a 'profiles' array");
    assert!(!profiles.is_empty(), "report should list at least one profile");
    for prof in profiles {
        let name = prof["name"].as_str().unwrap_or("?");
        let pass = prof["pass"].as_u64().unwrap_or(0);
        let fail = prof["fail"].as_u64().unwrap_or(u64::MAX);
        assert_eq!(fail, 0,
            "profile {} had {} failed queries — sign convention or Path-2 \
             reuse regression?", name, fail);
        assert!(pass > 0, "profile {} reported 0 passes", name);
    }
}
