// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for sized-profile publishing.
//!
//! Drives the full pipeline on the standard e2e fixture with several
//! sized profiles, serves the resulting dataset directory over HTTP,
//! then exercises the public `vectordata::TestDataGroup` client API
//! against it. Explicitly asserts the things that silently regressed
//! in prior iterations:
//!
//! 1. Each sized profile is materialised as a concrete entry in the
//!    published `dataset.yaml` (clients that don't know the compact
//!    `sized:` spec still see them).
//! 2. Each sized profile carries its OWN `neighbor_indices` and
//!    `neighbor_distances` view paths pointing at
//!    `profiles/{name}/...` — not inherited from the default profile.
//! 3. The per-profile ground-truth files actually exist on disk.
//! 4. They are fetchable over HTTP through
//!    `TestDataGroup::load(http_url)` → `profile(name).neighbor_indices()`.
//! 5. Every neighbour index is in-range for the profile's `base_count`
//!    — the "sized-range top-K mistakenly derived from the full-range
//!    top-K" bug that burned us earlier.
//!
//! Data is small (the shared 200-vector dim-4 fixture) so this test
//! completes in a few seconds of pipeline work plus the HTTP probing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use veks::prepare::import::ImportArgs;

mod support;
use support::testserver::TestServer;

// ─── Shared test helpers (miniature copies of the e2e_pipeline.rs
// fixtures; duplicated deliberately because integration tests can't
// share a helper module across files without extra ceremony). ────────

fn veks_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_veks"));
    if !path.exists() {
        path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../target/debug/veks");
    }
    path
}

fn make_tempdir() -> tempfile::TempDir {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

fn fixtures() -> PathBuf {
    // Matches e2e_pipeline.rs::fixtures — fixture generated on first run
    // by the `e2e_fixtures` build script under `target/tmp/e2e-fixtures/`.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/tmp/e2e-fixtures")
}

fn ensure_base_fixture() -> PathBuf {
    // The fixture is materialised lazily by e2e_pipeline.rs. If that
    // test binary hasn't run yet, write one inline here using the same
    // known-duplicates / known-zero recipe so this test is self-
    // sufficient.
    let p = fixtures().join("base.fvecs");
    if !p.exists() {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let f = std::fs::File::create(&p).unwrap();
        let mut w = std::io::BufWriter::new(f);
        let dim: usize = 4;
        let count: usize = 200;
        for i in 0..count {
            w.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for d in 0..dim {
                // Same pattern as e2e_pipeline.rs::write_test_fvec:
                // vector 1 duplicates vector 0, vector 2 is zero.
                let val: f32 = if i == 2 {
                    0.0
                } else if i == 1 || i == 0 {
                    (d + 1) as f32
                } else {
                    (i * dim + d + 1) as f32
                };
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }
    p
}

fn default_args(name: &str, output: &Path) -> ImportArgs {
    ImportArgs {
        name: name.to_string(),
        output: output.to_path_buf(),
        base_vectors: None,
        query_vectors: None,
        self_search: false,
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
        pedantic_dedup: false,
        selectivity: 0.0001,
        predicate_count: 10000,
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
    let output = Command::new(veks_bin())
        .arg("run")
        .arg("--output").arg("batch")
        .arg("--threads").arg("2")
        .arg(dataset_yaml)
        .output()
        .expect("failed to execute veks");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.status.success() {
        eprintln!("=== PIPELINE STDOUT ===\n{}", stdout);
        eprintln!("=== PIPELINE STDERR ===\n{}", stderr);
    }
    (output.status.success(), format!("{}\n{}", stdout, stderr))
}

// ─── The test ────────────────────────────────────────────────────────

/// Full coverage for sized-profile publishing over HTTP.
#[test]
fn e2e_sized_profiles_published_and_fetchable_over_http() {
    let tmp = make_tempdir();
    let fvec_src = ensure_base_fixture();
    let fvec = tmp.path().join("base.fvecs");
    std::fs::copy(&fvec_src, &fvec).unwrap();

    let out = tmp.path().join("out");
    let mut args = default_args("http-sized", &out);
    args.base_vectors = Some(fvec);
    args.self_search = true;
    // Three concrete counts. All below `clean_count` (≈ 188 after
    // dedup + zero removal on the 200-vector fixture), so no clamping
    // or skipping at expansion time. Bare-integer specs produce
    // profiles named by their count via `format_count_with_suffix`:
    // for these values the output is the literal decimal string.
    args.sized_profiles = Some("25,50,100".to_string());

    veks::prepare::import::run(args);

    let yaml_path = out.join("dataset.yaml");
    let (ok, log) = run_pipeline(&yaml_path);
    assert!(ok, "pipeline failed:\n{}", log);

    // ── (1) + (2): concrete sized entries with their own GT paths ────
    let yaml = std::fs::read_to_string(&yaml_path).unwrap();
    for sized in &["25", "50", "100"] {
        let entry_marker = format!("\n  {}:\n", sized);
        assert!(
            yaml.contains(&entry_marker),
            "expected concrete profile entry `{}:` in published dataset.yaml, got:\n{}",
            sized, yaml,
        );
        let idx_path = format!("profiles/{}/neighbor_indices.ivecs", sized);
        let dist_path = format!("profiles/{}/neighbor_distances.fvecs", sized);
        assert!(
            yaml.contains(&idx_path),
            "expected per-profile `{}` in published dataset.yaml — sized \
             profiles must carry their OWN GT paths, not inherit default's. got:\n{}",
            idx_path, yaml,
        );
        assert!(
            yaml.contains(&dist_path),
            "expected per-profile `{}` in published dataset.yaml. got:\n{}",
            dist_path, yaml,
        );
    }
    // The compact generator spec is preserved at root-level as
    // `strata:` (migrated out from `profiles.sized:`) — idempotent
    // re-processing needs the generator source, and root-level keeps
    // the profile map clean of non-profile entries.
    assert!(
        yaml.contains("\nstrata:"),
        "the root-level `strata:` spec must stay in dataset.yaml for round-trip \
         re-processing, got:\n{}",
        yaml,
    );
    // And `sized:` must NOT live inside the profiles map any more —
    // having it there forces every consumer of `profiles:` to skip it.
    let profiles_section = yaml.split("\nprofiles:\n").nth(1).unwrap_or("");
    assert!(
        !profiles_section.contains("  sized:"),
        "`sized:` must not sit under `profiles:` — it moved to root-level `strata:`, got:\n{}",
        yaml,
    );

    // ── (3): GT files actually exist on disk ─────────────────────────
    for profile in &["default", "25", "50", "100"] {
        let idx = out.join(format!("profiles/{}/neighbor_indices.ivecs", profile));
        let dist = out.join(format!("profiles/{}/neighbor_distances.fvecs", profile));
        assert!(idx.exists(), "missing {:?}", idx);
        assert!(dist.exists(), "missing {:?}", dist);
    }

    // ── (3b): the published yaml carries the windowed base_vectors
    // reference for every sized profile. The documented sub-ordinal
    // suffix model — `base_vectors.window: "0..N"` — is what lets
    // clients open the shared base file with automatic clipping;
    // dropping it (as we did briefly) silently let consumers iterate
    // the FULL base instead of the sized prefix. Pin the wire shape.
    //
    // Parse the yaml and check the structured profile.base_vectors
    // window field directly — substring matching is too easy to
    // satisfy by accident (e.g., `base_count: 25` contains `25`).
    {
        use vectordata::dataset::DatasetConfig;
        let cfg = DatasetConfig::load(&yaml_path)
            .expect("re-parse published dataset.yaml");
        for (sized, bc) in [("25", 25u64), ("50", 50u64), ("100", 100u64)] {
            let p = cfg.profiles.profile(sized)
                .unwrap_or_else(|| panic!("profile '{}' missing from re-loaded config", sized));
            let bv = p.view("base_vectors")
                .unwrap_or_else(|| panic!("profile '{}' must inherit base_vectors", sized));
            // `effective_window` is the canonical accessor — DSView
            // splits storage between `view.window` (set by an explicit
            // `window:` field in YAML) and `view.source.window` (set by
            // a `path[0..N)` source-string suffix). Either is valid;
            // `effective_window` returns whichever is set, preferring
            // the outer view-level window.
            let window = bv.effective_window();
            assert!(
                !window.is_empty(),
                "profile '{}' base_vectors must carry a [0..{}) window — sub-ordinal \
                 suffix model is the contract that lets clients clip without honoring \
                 base_count manually. View was: source.path={}, source.window={:?}, view.window={:?}",
                sized, bc, bv.source.path, bv.source.window, bv.window,
            );
            let iv = &window.0[0];
            assert_eq!(
                (iv.min_incl, iv.max_excl), (0, bc),
                "profile '{}' base_vectors window mismatch", sized,
            );
        }
    }

    // ── (4) + (5): fetch + verify through the client API over HTTP ──
    let server = TestServer::start(&out).expect("start HTTP test server");
    let group = vectordata::TestDataGroup::load(&server.base_url())
        .expect("TestDataGroup::load over HTTP");

    let names: std::collections::HashSet<String> =
        group.profile_names().into_iter().collect();
    for expected in &["default", "25", "50", "100"] {
        assert!(
            names.contains(*expected),
            "profile '{}' missing from HTTP-served group; have: {:?}",
            expected, names,
        );
    }

    // Per-sized-profile GT fetch + range validation.
    let k = 5usize;           // matches default_args.neighbors
    let q_count = 10usize;    // matches default_args.query_count
    for (sized, base_count) in [("25", 25u64), ("50", 50u64), ("100", 100u64)] {
        let view = group.profile(sized)
            .unwrap_or_else(|| panic!("profile '{}' not found over HTTP", sized));

        let indices = view.neighbor_indices()
            .unwrap_or_else(|e| panic!("open neighbor_indices for {}: {:?}", sized, e));
        assert_eq!(
            indices.count(), q_count,
            "profile {}: expected {} query rows, got {}",
            sized, q_count, indices.count(),
        );
        assert_eq!(
            indices.dim(), k,
            "profile {}: expected k={}, got {}",
            sized, k, indices.dim(),
        );

        for qi in 0..indices.count() {
            let row = indices.get(qi)
                .unwrap_or_else(|e| panic!("fetch row {} of profile {}: {:?}", qi, sized, e));
            assert_eq!(row.len(), k);
            for &idx in &row {
                assert!(
                    idx >= 0,
                    "profile {}: negative neighbour index {} in query row {}",
                    sized, idx, qi,
                );
                assert!(
                    (idx as u64) < base_count,
                    "profile {}: neighbour index {} >= base_count {} (query row {})",
                    sized, idx, base_count, qi,
                );
            }
        }

        // Distances come back over HTTP too and have the same shape.
        let dists = view.neighbor_distances()
            .unwrap_or_else(|e| panic!("open neighbor_distances for {}: {:?}", sized, e));
        assert_eq!(dists.count(), q_count);
        assert_eq!(dists.dim(), k);

        // ── Client-side window enforcement ─────────────────────────
        // The whole point of the sub-ordinal suffix model is that
        // `view.base_vectors()` returns a reader clipped to the
        // profile's window — consumers shouldn't have to remember
        // `view.base_count()` and clamp manually. Verify both:
        //   - count() reports the windowed length (base_count).
        //   - get(N) for N == base_count is rejected.
        //   - get(N) for N inside the window returns the same data
        //     as default's underlying file at that ordinal (proves
        //     it's a windowed view of the SAME file, not a copy).
        let base = view.base_vectors()
            .unwrap_or_else(|e| panic!("open base_vectors for {}: {:?}", sized, e));
        assert_eq!(
            base.count() as u64, base_count,
            "profile {}: base_vectors.count() must report the window length ({}), got {}",
            sized, base_count, base.count(),
        );
        // Out-of-window read must error.
        assert!(
            base.get(base_count as usize).is_err(),
            "profile {}: base_vectors.get({}) must be out-of-range (window len {})",
            sized, base_count, base_count,
        );
        // In-window reads must succeed and match default's file.
        let default_view = group.profile("default").expect("default profile");
        let default_base = default_view.base_vectors().expect("default base_vectors");
        for &probe in &[0usize, 1, (base_count as usize / 2)] {
            if probe >= base_count as usize { continue; }
            let from_sized = base.get(probe)
                .unwrap_or_else(|e| panic!("profile {}: get({}) failed: {:?}", sized, probe, e));
            let from_default = default_base.get(probe)
                .unwrap_or_else(|e| panic!("default: get({}) failed: {:?}", probe, e));
            assert_eq!(
                from_sized, from_default,
                "profile {}: windowed read at ordinal {} must equal default's read at the same ordinal — \
                 windowed view should reference, not copy",
                sized, probe,
            );
        }
        // And `view.base_count()` must agree with the windowed reader.
        assert_eq!(
            view.base_count(), Some(base_count),
            "profile {}: base_count() metadata must equal windowed reader length",
            sized,
        );
    }

    // Sanity: the default profile's GT is also fetchable and has the
    // full base (post-zero, post-dedup) in its address space.
    let default_view = group.profile("default")
        .expect("default profile over HTTP");
    let default_idx = default_view.neighbor_indices()
        .expect("open default neighbor_indices");
    assert_eq!(default_idx.count(), q_count);
    assert_eq!(default_idx.dim(), k);
}
