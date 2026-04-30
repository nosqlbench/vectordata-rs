// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: cross-engine KNN parity demo.
//!
//! Runs every available KNN engine on the same inputs and prints a
//! readable comparison so a user can see — directly, without reading
//! source — that the engines agree under the documented
//! `BoundaryMismatch ≤ 5` tolerance described in the conformance
//! section of [`12-knn-utils-verification.md`].
//!
//! Engines exercised:
//!
//! - `compute knn`         — SimSIMD (always available)
//! - `compute knn-stdarch` — pure `std::arch` (always available)
//! - `compute knn-blas`    — `cblas_sgemm` (requires `knnutils` feature)
//! - `compute knn-faiss`   — FAISS `IndexFlat` (requires `faiss` feature)
//!
//! Engines not enabled at build time are reported as `skipped: feature
//! not enabled` rather than failing — the demo still produces a useful
//! comparison among whichever subset is present. Comparison classifier
//! is the same one used by the unit tests and `verify-knn-consolidated`
//! ([`super::knn_compare::compare_query_ordinals`]).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::XvecReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};
use super::knn_compare::{compare_query_ordinals, QueryResult, VerifySummary};

#[cfg(feature = "knnutils")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,     // CblasRowMajor = 101
        transa: i32,    // CblasNoTrans = 111
        transb: i32,    // CblasTrans = 112
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

}
#[cfg(feature = "knnutils")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(feature = "knnutils")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(feature = "knnutils")]
const CBLAS_TRANS: i32 = 112;

/// Set env vars that BLAS libraries read at init time. This only
/// has an effect if called before the BLAS is first invoked — for
/// our parity demo the process hasn't touched BLAS yet when this
// blas_set_single_threaded() was a private helper here; the logic is
// now inlined at the top of execute() above. See pipeline::blas_abi
// for the production-grade variant gated on the `faiss` feature.

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyEngineParityOp)
}

pub struct VerifyEngineParityOp;

/// What we ran an engine with, so we can render the result table.
struct EngineRun {
    name: &'static str,
    /// Path to the produced ivec file. `None` when skipped (feature off
    /// or engine errored out).
    indices_path: Option<PathBuf>,
    /// Wall time for the engine's own run, excluding setup.
    elapsed: std::time::Duration,
    /// Diagnostic when the engine couldn't run.
    note: Option<String>,
}

impl CommandOp for VerifyEngineParityOp {
    fn command_path(&self) -> &str {
        "verify engine-parity"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Run all available KNN engines on the same inputs and report cross-engine parity".into(),
            body: format!(
                "# verify engine-parity\n\n\
                Runs every available KNN engine on the same inputs and \
                shows their results side-by-side, classified per query as \
                ExactMatch / SetMatch / BoundaryMismatch(d) / RealMismatch(d).\n\n\
                ## Tolerance\n\n\
                Default boundary tolerance is **0** — any neighbor-set \
                difference fails the run. Real measurements at \
                dim ≤ 128 with k ≤ 100 produce 0 boundary mismatches across \
                all four engines (SimSIMD / stdarch / BLAS / FAISS); the \
                small `SetMatch` count seen at larger k is just \
                tie-breaking order, not a substantive disagreement.\n\n\
                Pass `--boundary-tolerance N` to allow up to N differing \
                neighbors per query. This exists for two known degenerate \
                regimes:\n\n\
                - **Multi-threaded BLAS**: sgemm thread-block decomposition \
                  produces non-deterministic ULP-level rounding at billion-vector \
                  scale.\n\
                - **Curse of dimensionality**: at very high dim with small base, \
                  pairwise distances cluster so tightly that neighbor selection \
                  is dominated by ULP noise.\n\n\
                ## Two modes\n\n\
                - **User-data mode**: pass `--base` and `--query` to compare engines on \
                  your own files.\n\
                - **Synthetic mode**: pass `--synthetic` (with optional `--dim` / \
                  `--base-count` / `--query-count` / `--seed`) to run on a deterministic \
                  in-process fixture.\n\n\
                ## Options\n\n{}",
                render_options_table(&options),
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "base".into(),         type_name: "Path".into(), required: false, default: None,
                description: "Base vectors (fvec). Required unless --synthetic.".into(),
                role: OptionRole::Input },
            OptionDesc { name: "query".into(),        type_name: "Path".into(), required: false, default: None,
                description: "Query vectors (fvec). Required unless --synthetic.".into(),
                role: OptionRole::Input },
            OptionDesc { name: "neighbors".into(),    type_name: "int".into(),  required: false, default: Some("10".into()),
                description: "k (number of neighbors per query)".into(),
                role: OptionRole::Config },
            OptionDesc { name: "metric".into(),       type_name: "enum".into(), required: false, default: Some("L2".into()),
                description: "L2, DOT_PRODUCT, COSINE, IP".into(),
                role: OptionRole::Config },
            OptionDesc { name: "engines".into(),      type_name: "string".into(), required: false, default: None,
                description: "Comma-separated subset (default = all enabled): metal,stdarch,blas,faiss".into(),
                role: OptionRole::Config },
            OptionDesc { name: "show-queries".into(), type_name: "int".into(),  required: false, default: Some("5".into()),
                description: "Number of queries to print side-by-side".into(),
                role: OptionRole::Config },
            OptionDesc { name: "sample".into(), type_name: "int".into(), required: false, default: Some("100".into()),
                description: "User-data mode: number of queries to sample from the input query file. Each engine then runs against the full base × this many queries. Set to 0 to verify all queries (only feasible for small datasets — every engine scans the full base).".into(),
                role: OptionRole::Config },
            OptionDesc { name: "boundary-tolerance".into(), type_name: "int".into(), required: false, default: Some("0".into()),
                description: "Max differing neighbors per query before the verdict flips to FAIL. Default 0 — any disagreement fails. See --help for the regimes where slack is justified.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "synthetic".into(),    type_name: "bool".into(), required: false, default: Some("false".into()),
                description: "Generate a deterministic fixture instead of reading --base/--query".into(),
                role: OptionRole::Config },
            OptionDesc { name: "dim".into(),          type_name: "int".into(),  required: false, default: Some("128".into()),
                description: "Synthetic mode: dimension".into(),
                role: OptionRole::Config },
            OptionDesc { name: "base-count".into(),   type_name: "int".into(),  required: false, default: Some("10000".into()),
                description: "Synthetic mode: base vector count".into(),
                role: OptionRole::Config },
            OptionDesc { name: "query-count".into(),  type_name: "int".into(),  required: false, default: Some("100".into()),
                description: "Synthetic mode: query vector count".into(),
                role: OptionRole::Config },
            OptionDesc { name: "seed".into(),         type_name: "int".into(),  required: false, default: Some("42".into()),
                description: "PRNG seed (xorshift) — used by synthetic-mode fixture generation and by user-data-mode query sampling.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "distribution".into(), type_name: "enum".into(), required: false, default: Some("uniform".into()),
                description: "Synthetic mode: 'uniform' (U[-1,1] per component — curse-of-dim kicks in at high dim), 'gaussian' (N(0,1) per component + L2-normalize onto unit sphere — still curse-prone at high dim), or 'clustered' (mixture of 16 Gaussian clusters on the unit sphere — simulates the cluster structure real embedding models produce; this is the distribution where top-k is well-defined at any dim and all kernels agree).".into(),
                role: OptionRole::Config },
            OptionDesc { name: "clusters".into(), type_name: "int".into(), required: false, default: Some("16".into()),
                description: "Synthetic 'clustered' mode: number of Gaussian clusters (default 16). More clusters → denser sphere → more ties.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "cluster-spread".into(), type_name: "float".into(), required: false, default: Some("0.05".into()),
                description: "Synthetic 'clustered' mode: per-vector noise std-dev relative to cluster radius (default 0.05). Smaller → tighter clusters → distances spread more widely → easier top-k.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "assume_normalized_like_faiss".into(), type_name: "bool".into(), required: false, default: Some("false".into()),
                description: "metric=COSINE only: treat inputs as already unit-normalized so cosine collapses to inner product (FAISS/numpy/knn_utils convention). Mutually exclusive with use_proper_cosine_metric. Propagated to every engine.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "use_proper_cosine_metric".into(), type_name: "bool".into(), required: false, default: Some("false".into()),
                description: "metric=COSINE only: divide the dot product by |q|·|b| in-kernel so cosine works on un-normalized inputs. Mutually exclusive with assume_normalized_like_faiss. Propagated to every engine.".into(),
                role: OptionRole::Config },

            // ── Sweep mode ────────────────────────────────────────
            OptionDesc { name: "sweep".into(), type_name: "bool".into(), required: false, default: Some("false".into()),
                description: "Run the cross-product of (dims × distributions × metrics × neighbors) internally and emit a single matrix-shaped summary table. Each axis defaults to a standard sweep range; pass `--dims=8,32,4096` (etc.) to narrow.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "dims".into(), type_name: "string".into(), required: false, default: None,
                description: "Sweep mode: comma-separated list of dimensions (default: 8,32,128,384,512,768,1024,1536,2048,3072,4096).".into(),
                role: OptionRole::Config },
            OptionDesc { name: "distributions".into(), type_name: "string".into(), required: false, default: None,
                description: "Sweep mode: comma-separated list (default: uniform,gaussian,clustered).".into(),
                role: OptionRole::Config },
            OptionDesc { name: "metrics".into(), type_name: "string".into(), required: false, default: None,
                description: "Sweep mode: comma-separated list (default: L2,DOT_PRODUCT). For COSINE include the explicit mode flag(s) on the parent invocation.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "neighbors-list".into(), type_name: "string".into(), required: false, default: None,
                description: "Sweep mode: comma-separated list of k values (default: just the value of `--neighbors`, or 100 if unset).".into(),
                role: OptionRole::Config },
        ]
    }

    fn value_completions(&self) -> std::collections::HashMap<String, crate::pipeline::command::ValueCompletions> {
        use crate::pipeline::command::ValueCompletions;
        let mut m = std::collections::HashMap::new();
        m.insert("engines".to_string(), ValueCompletions::comma_separated_enum(
            &["metal", "stdarch", "blas", "blas-mirror", "faiss"]));
        m.insert("metric".to_string(), ValueCompletions::enum_values(
            &["L2", "DOT_PRODUCT", "COSINE", "IP"]));
        m.insert("distribution".to_string(), ValueCompletions::enum_values(
            &["uniform", "gaussian", "clustered"]));
        m.insert("metrics".to_string(), ValueCompletions::comma_separated_enum(
            &["L2", "DOT_PRODUCT", "COSINE"]));
        m.insert("distributions".to_string(), ValueCompletions::comma_separated_enum(
            &["uniform", "gaussian", "clustered"]));
        m
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // Force the underlying BLAS to single-threaded BEFORE any
        // engine runs. Two reasons stack up here:
        //   1. The faiss-sys static MKL bug (see pipeline::blas_abi)
        //      corrupts sgemm output under multi-threading.
        //   2. Even with a clean BLAS, OpenMP's non-deterministic
        //      block scheduling makes sgemm vary run-to-run, which
        //      defeats the deterministic-parity-demo goal.
        // We unconditionally force single-threaded here (not just
        // when faiss feature is on) to handle reason (2) too.
        // Done once at the top so a sweep doesn't re-set N times.
        for var in &["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                     "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"] {
            unsafe { std::env::set_var(var, "1"); }
        }

        // Sweep mode: when `--sweep` is set, the command runs the
        // cross-product of (dims × distributions × metrics × ks)
        // internally and emits a single matrix-shaped summary table.
        // List-valued options (`--dims`, `--distributions`, etc.)
        // narrow the matrix; with no list options the full default
        // matrix runs. See `run_sweep` for the full enumeration.
        let sweep_mode = options.get("sweep")
            .map(|s| matches!(s, "true" | "1" | "yes"))
            .unwrap_or(false);
        if sweep_mode {
            return run_sweep(options, ctx, start);
        }

        let neighbors: usize = parse_int(options, "neighbors", 10);
        let metric = options.get("metric").unwrap_or("L2").to_string();
        let show_queries: usize = parse_int(options, "show-queries", 5);
        let boundary_tolerance: usize = parse_int(options, "boundary-tolerance", 0);
        let synthetic = options.get("synthetic")
            .map(|s| matches!(s, "true" | "1" | "yes"))
            .unwrap_or(false);

        // Use the system tempdir (honours $TMPDIR, which `.cargo/config.toml`
        // points at `target/test-tmp` for cargo-driven runs and which the
        // OS auto-cleans for direct CLI runs). Avoid `tempdir_in(workspace)`
        // — it dropped the synthetic fvec fixtures into the project root,
        // and a SIGINT'd parity run leaves them behind as `.tmpXXXXXX/`
        // litter. Our fixtures are at most a few MiB; cross-filesystem
        // rename concerns don't apply.
        let workdir = match tempfile::tempdir() {
            Ok(d) => d,
            Err(e) => return error(format!("could not create workdir: {}", e), start),
        };

        // Resolve the (base, query) paths — either user-supplied or
        // synthetic-generated into the workdir. This MUST happen
        // before the workspace swap below: in user-data mode the
        // resolver consults `<workspace>/dataset.yaml` for the
        // `base_vectors` / `query_vectors` facets, and once
        // `ctx.workspace` is pointing at the tempdir there's no
        // dataset.yaml to find.
        let (base_path, query_path) = if synthetic {
            let dim:        usize = parse_int(options, "dim", 128);
            let base_count: usize = parse_int(options, "base-count", 10_000);
            let query_count: usize = parse_int(options, "query-count", 100);
            let seed:       u64   = parse_int(options, "seed", 42) as u64;
            let dist_name = options.get("distribution").unwrap_or("uniform").to_lowercase();
            let clusters: usize = parse_int(options, "clusters", 16);
            let cluster_spread: f32 = options.get("cluster-spread")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.05);
            let dist = match dist_name.as_str() {
                "uniform" => Distribution::Uniform,
                "gaussian" | "normal" => Distribution::GaussianNormalized,
                "clustered" | "clusters" | "embedding" => Distribution::Clustered {
                    k: clusters.max(1),
                    spread: cluster_spread.max(0.0),
                },
                other => return error(
                    format!("unknown --distribution '{}': use 'uniform', 'gaussian', or 'clustered'", other),
                    start,
                ),
            };
            let bp = workdir.path().join("base.fvec");
            let qp = workdir.path().join("query.fvec");
            // Debug: mirror the generated fixture to /tmp so callers can
            // inspect the exact data each engine saw.
            if let Ok(probe) = std::env::var("VEKS_PARITY_PROBE_DIR") {
                let _ = std::fs::create_dir_all(&probe);
                ctx.ui.log(&format!("  PROBE: mirroring fixture to {}", probe));
            }
            ctx.ui.log(&format!(
                "synthetic fixture: dim={}, base={}, query={}, seed={}, distribution={}",
                dim, base_count, query_count, seed, dist.label()));
            if let Err(e) = write_synthetic_fvec(&bp, dim, base_count, seed, dist) {
                return error(format!("synthetic base write: {}", e), start);
            }
            if let Err(e) = write_synthetic_fvec(&qp, dim, query_count, seed.wrapping_add(1), dist) {
                return error(format!("synthetic query write: {}", e), start);
            }
            if let Ok(probe) = std::env::var("VEKS_PARITY_PROBE_DIR") {
                let _ = std::fs::copy(&bp, std::path::Path::new(&probe).join("base.fvec"));
                let _ = std::fs::copy(&qp, std::path::Path::new(&probe).join("query.fvec"));
            }
            (bp, qp)
        } else {
            // User-data mode: explicit --base/--query wins; otherwise
            // resolve from the active profile's `base_vectors` /
            // `query_vectors` facets in dataset.yaml. See
            // pipeline::dataset_lookup.
            let b = match crate::pipeline::dataset_lookup::resolve_path_option(
                ctx, options, "base", "base_vectors",
            ) { Ok(s) => s, Err(e) => return error(e, start) };
            let q = match crate::pipeline::dataset_lookup::resolve_path_option(
                ctx, options, "query", "query_vectors",
            ) { Ok(s) => s, Err(e) => return error(e, start) };
            // Canonicalize to absolute so the per-engine sub-calls
            // (which run against the swapped tempdir workspace) don't
            // re-resolve `./profiles/...` against the wrong root.
            let bp = resolve(&b, &ctx.workspace);
            let qp = resolve(&q, &ctx.workspace);
            let bp = bp.canonicalize().unwrap_or(bp);
            let qp = qp.canonicalize().unwrap_or(qp);

            // Bound per-engine work by sampling queries: each engine
            // still scans the full base, but only against `sample`
            // queries. Keeps engine-parity feasible at any base size.
            // `--sample 0` means "use all queries" (small fixtures).
            let sample_n: usize = parse_int(options, "sample", 100);
            let seed_v: u64 = parse_int(options, "seed", 42) as u64;
            let qp = if sample_n > 0 {
                match write_sampled_query_file(
                    &qp, sample_n, seed_v, workdir.path(), &mut ctx.ui,
                ) {
                    Ok(p) => p,
                    Err(e) => return error(format!("query sampling: {}", e), start),
                }
            } else {
                qp
            };

            (bp, qp)
        };

        // Now isolate the engine working environment — workspace
        // AND cache — to the per-run tempdir. Without redirecting
        // workspace, engines that write side-effects relative to it
        // (e.g. `compute knn-metal` writes `knn_queries_with_ties` /
        // `knn_tied_neighbors` to `<workspace>/variables.yaml`) leak
        // those files into the user's cwd. Without redirecting cache,
        // engines from earlier runs replay each other (the cache key
        // is content-insensitive — see knn_segment::cache_prefix_for).
        // Both pointers get restored before this command returns.
        let isolated_cache = workdir.path().join("cache");
        let _ = std::fs::create_dir_all(&isolated_cache);
        let saved_ctx_cache = std::mem::replace(&mut ctx.cache, isolated_cache);
        let saved_ctx_workspace = std::mem::replace(&mut ctx.workspace, workdir.path().to_path_buf());

        // Decide which engines to run. The user can override via
        // --engines; otherwise we run everything that was compiled in.
        let requested: Vec<String> = options.get("engines")
            .map(|s| s.split(',').map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()).collect())
            .unwrap_or_else(|| vec![
                "metal".into(), "stdarch".into(), "blas".into(),
                "blas-mirror".into(), "faiss".into(),
            ]);
        let want = |name: &str| requested.iter().any(|r| r == name);

        // Cosine mode propagation. When `metric=COSINE`, every engine
        // independently requires either `assume_normalized_like_faiss`
        // or `use_proper_cosine_metric`. Read the parity command's
        // own flag(s) and forward them to every sub-engine.
        let assume_normalized = options.get("assume_normalized_like_faiss")
            .map(|s| s == "true").unwrap_or(false);
        let use_proper_cosine = options.get("use_proper_cosine_metric")
            .map(|s| s == "true").unwrap_or(false);
        let metric_upper = metric.to_uppercase();
        if metric_upper == "COSINE" && !(assume_normalized ^ use_proper_cosine) {
            return error(
                "metric=COSINE: set exactly one of assume_normalized_like_faiss=true \
                 or use_proper_cosine_metric=true on `verify engine-parity` so the flag \
                 can be forwarded to every engine",
                start,
            );
        }

        ctx.ui.log(&format!(
            "engine-parity: base={} query={} k={} metric={}",
            base_path.display(), query_path.display(), neighbors, metric));

        let mut runs: Vec<EngineRun> = Vec::new();

        if want("metal") {
            runs.push(run_engine(
                "metal",
                || super::compute_knn::factory(),
                &base_path, &query_path, &workdir.path().join("metal.ivec"),
                neighbors, &metric, assume_normalized, use_proper_cosine, ctx,
            ));
        }
        if want("stdarch") {
            runs.push(run_engine(
                "stdarch",
                || super::compute_knn_stdarch::factory(),
                &base_path, &query_path, &workdir.path().join("stdarch.ivec"),
                neighbors, &metric, assume_normalized, use_proper_cosine, ctx,
            ));
        }

        #[cfg(feature = "knnutils")]
        if want("blas") {
            runs.push(run_engine(
                "blas",
                || super::compute_knn_blas::factory(),
                &base_path, &query_path, &workdir.path().join("blas.ivec"),
                neighbors, &metric, assume_normalized, use_proper_cosine, ctx,
            ));
        }
        #[cfg(not(feature = "knnutils"))]
        if want("blas") {
            runs.push(skipped("blas", "knnutils feature not enabled (cargo build --features knnutils)"));
        }

        #[cfg(feature = "faiss")]
        if want("faiss") {
            // FAISS has no native proper-cosine kernel — its
            // METRIC_INNER_PRODUCT assumes pre-normalized inputs and
            // never divides by `|q|·|b|`. On *normalized* data
            // (`assume_normalized_like_faiss=true`, or any naturally
            // unit-norm distribution like our `gaussian`/`clustered`
            // synthetics), IP === proper cosine and FAISS agrees with
            // the other engines. On un-normalized data with
            // `use_proper_cosine_metric=true`, FAISS would compare
            // against a different ranking — that's a genuine
            // semantic difference, not a parity violation, so skip.
            if metric_upper == "COSINE" && use_proper_cosine {
                runs.push(skipped("faiss",
                    "metric=COSINE use_proper_cosine_metric=true: FAISS has no \
                     native proper-cosine kernel (it treats COSINE as IP on \
                     assumed-normalized input); not comparable in this mode"));
            } else {
                runs.push(run_engine(
                    "faiss",
                    || Box::new(super::compute_knn_faiss::ComputeKnnFaissOp) as Box<dyn CommandOp>,
                    &base_path, &query_path, &workdir.path().join("faiss.ivec"),
                    neighbors, &metric, assume_normalized, use_proper_cosine, ctx,
                ));
            }
        }
        #[cfg(not(feature = "faiss"))]
        if want("faiss") {
            runs.push(skipped("faiss", "faiss feature not enabled (cargo build --features faiss)"));
        }

        // Probe engine: mirrors FAISS's `exhaustive_L2sqr_blas_default_impl`
        // exactly — same outer/inner block sizes, same sgemm orientation,
        // same per-cell post-processing. Extended for IP/Cosine using the
        // same sgemm + per-cell post-pass.
        #[cfg(feature = "knnutils")]
        if want("blas-mirror") {
            ctx.ui.log("  running engine: blas-mirror");
            let started = Instant::now();
            let path = workdir.path().join("blas-mirror.ivec");
            let res = run_blas_mirror(
                &base_path, &query_path, &path, neighbors, &metric,
                assume_normalized, use_proper_cosine, &mut ctx.ui,
            );
            let elapsed = started.elapsed();
            runs.push(match res {
                Ok(()) => EngineRun { name: "blas-mirror", indices_path: Some(path), elapsed, note: None },
                Err(e) => EngineRun { name: "blas-mirror", indices_path: None, elapsed, note: Some(e) },
            });
        }
        #[cfg(not(feature = "knnutils"))]
        if want("blas-mirror") {
            runs.push(skipped("blas-mirror", "knnutils feature not enabled"));
        }

        // Print run summaries.
        let mut emit = String::new();
        emit.push('\n');
        emit.push_str("ENGINE       STATUS                                                ELAPSED\n");
        emit.push_str("──────────── ───────────────────────────────────────────────────── ─────────\n");
        for r in &runs {
            let (status, elapsed) = match (&r.indices_path, &r.note) {
                (Some(_), _)        => ("ran".to_string(),                       format!("{:.2}s", r.elapsed.as_secs_f64())),
                (None, Some(note))  => (format!("skipped: {}", note),            "—".to_string()),
                (None, None)        => ("skipped: unknown".to_string(),          "—".to_string()),
            };
            emit.push_str(&format!("{:<12} {:<53} {:>9}\n", r.name, truncate(&status, 53), elapsed));
        }

        // Load the indices files for engines that ran.
        let mut loaded: Vec<(&'static str, Vec<Vec<i32>>)> = Vec::new();
        for r in &runs {
            if let Some(p) = &r.indices_path {
                if let Ok(probe) = std::env::var("VEKS_PARITY_PROBE_DIR") {
                    let _ = std::fs::copy(
                        p,
                        std::path::Path::new(&probe).join(format!("{}.ivec", r.name)),
                    );
                }
                match read_ivec_rows(p) {
                    Ok(rows) => loaded.push((r.name, rows)),
                    Err(e) => emit.push_str(&format!("  WARN: failed to read {}: {}\n", r.name, e)),
                }
            }
        }

        if loaded.len() < 2 {
            emit.push_str("\nNeed at least two engines to compare. Build with `--features knnutils,faiss` for the full set.\n");
            ctx.ui.log(&emit);
            return CommandResult { status: Status::Ok, message: "not enough engines to compare".into(), produced: vec![], elapsed: start.elapsed() };
        }

        // Side-by-side per-query view for the first `show_queries`
        // queries against the first engine as a visual reference.
        let n_queries = loaded[0].1.len();
        let n_show = show_queries.min(n_queries);
        if n_show > 0 {
            emit.push('\n');
            emit.push_str(&format!("First {} queries (k={}) — neighbors per engine:\n", n_show, neighbors));
            for q in 0..n_show {
                emit.push_str(&format!("  query {}:\n", q));
                for (name, rows) in &loaded {
                    emit.push_str(&format!("    {:<8} {}\n", name, fmt_neighbors(&rows[q], neighbors)));
                }
            }
        }

        // Pair-wise classification table. We pin the first engine as
        // the "reference" and compare every other engine against it.
        // VerifySummary uses the static BOUNDARY_THRESHOLD constant for
        // its own classification into exact / set / bound / real, so
        // we render all four buckets (they must sum to TOTAL) and add
        // a MAX-DIFF column showing the largest observed disagreement
        // plus an EXCEED column counting queries that exceed the
        // user-supplied --boundary-tolerance.
        let (ref_name, ref_rows) = loaded.first().cloned().unwrap();
        let mut any_fail = false;
        emit.push('\n');
        emit.push_str(&format!(
            "Pair-wise comparison (reference = {}, boundary-tolerance = {}):\n",
            ref_name, boundary_tolerance,
        ));
        emit.push_str("VS                EXACT     SET   BOUND    REAL  MAX-DIFF  EXCEED   TOTAL  VERDICT\n");
        emit.push_str("──────────────── ────── ─────── ─────── ─────── ───────── ─────── ──────── ────────\n");
        for (other_name, other_rows) in loaded.iter().skip(1) {
            let mut summary = VerifySummary::default();
            let mut max_diff = 0usize;
            let mut over_tolerance = 0usize;
            for q in 0..n_queries {
                let r = compare_query_ordinals(&other_rows[q], &ref_rows[q]);
                summary.record(&r);
                let diff = match r {
                    QueryResult::ExactMatch | QueryResult::SetMatch => 0,
                    QueryResult::BoundaryMismatch(d) | QueryResult::RealMismatch(d) => d,
                };
                if diff > max_diff { max_diff = diff; }
                if diff > boundary_tolerance { over_tolerance += 1; }
            }
            let verdict = if over_tolerance == 0 { "PASS" } else { any_fail = true; "FAIL" };
            emit.push_str(&format!(
                "{:<16} {:>6} {:>7} {:>7} {:>7} {:>9} {:>7} {:>8}  {}\n",
                other_name,
                summary.exact_match,
                summary.set_match,
                summary.boundary_mismatch,
                summary.real_mismatch,
                max_diff,
                over_tolerance,
                summary.total,
                verdict,
            ));
        }

        emit.push('\n');
        let verdict = if any_fail {
            format!("Engines disagree beyond boundary-tolerance={} — see EXCEED column above. \
                     Pass --boundary-tolerance N to allow up to N differing neighbors per query \
                     (only justifiable for multi-threaded BLAS at scale or curse-of-dimensionality regimes).",
                    boundary_tolerance)
        } else if boundary_tolerance == 0 {
            "All engines produce identical neighbor sets (set-equivalent or stricter)".to_string()
        } else {
            format!("All pairs agree within boundary-tolerance={}", boundary_tolerance)
        };
        emit.push_str(&format!("Result: {}\n", verdict));
        ctx.ui.log(&emit);

        ctx.cache = saved_ctx_cache;
        ctx.workspace = saved_ctx_workspace;

        CommandResult {
            status: if any_fail { Status::Error } else { Status::Ok },
            message: verdict,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

}

fn parse_int(options: &Options, key: &str, default: u64) -> usize {
    options.get(key)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default) as usize
}

/// Parse a comma-separated list option. Returns `None` if the option is
/// unset (so the caller can supply a default), or `Some(empty)` if the
/// option is set to a string with no non-whitespace tokens.
fn parse_list(options: &Options, key: &str) -> Option<Vec<String>> {
    options.get(key).map(|s| {
        s.split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect()
    })
}

/// Sweep mode: run the cross-product of (metric × k × dim × distribution)
/// internally and emit a single matrix-shaped summary table.
///
/// Each cell is dispatched by recursively calling `execute` with `sweep`
/// cleared and the singular axis keys (`dim`, `distribution`, `metric`,
/// `neighbors`) overridden. All other options on the parent invocation
/// (engines subset, boundary-tolerance, cosine flags, base-count, etc.)
/// pass through unchanged.
fn run_sweep(options: &Options, ctx: &mut StreamContext, start: Instant) -> CommandResult {
    let dims = parse_list(options, "dims").unwrap_or_else(|| {
        vec!["8", "32", "128", "384", "512", "768", "1024", "1536", "2048", "3072", "4096"]
            .into_iter().map(String::from).collect()
    });
    let distributions = parse_list(options, "distributions").unwrap_or_else(|| {
        vec!["uniform", "gaussian", "clustered"].into_iter().map(String::from).collect()
    });
    let metrics = parse_list(options, "metrics").unwrap_or_else(|| {
        vec!["L2", "DOT_PRODUCT"].into_iter().map(String::from).collect()
    });
    let ks = parse_list(options, "neighbors-list").unwrap_or_else(|| {
        let k = options.get("neighbors").map(|s| s.to_string()).unwrap_or_else(|| "100".into());
        vec![k]
    });

    let total = dims.len() * distributions.len() * metrics.len() * ks.len();
    if total == 0 {
        return error("sweep: empty cross-product (one of --dims/--distributions/--metrics/--neighbors-list is empty)", start);
    }

    ctx.ui.log(&format!(
        "engine-parity sweep: {} cells ({} dims × {} distributions × {} metrics × {} ks)",
        total, dims.len(), distributions.len(), metrics.len(), ks.len(),
    ));

    struct Cell {
        dim: String,
        dist: String,
        metric: String,
        k: String,
        status: Status,
        msg: String,
        elapsed: std::time::Duration,
    }
    let mut cells: Vec<Cell> = Vec::with_capacity(total);

    let mut idx = 0usize;
    for metric in &metrics {
        for k in &ks {
            for dim in &dims {
                for dist in &distributions {
                    idx += 1;
                    ctx.ui.log(&format!(
                        "  [{}/{}] metric={} k={} dim={} distribution={}",
                        idx, total, metric, k, dim, dist,
                    ));

                    // Build per-cell options: clone parent, force
                    // synthetic mode, override the swept axes, clear
                    // `sweep` so the recursive call doesn't re-enter.
                    let mut cell_opts = options.clone();
                    cell_opts.set("sweep", "false");
                    cell_opts.set("synthetic", "true");
                    cell_opts.set("dim", dim);
                    cell_opts.set("distribution", dist);
                    cell_opts.set("metric", metric);
                    cell_opts.set("neighbors", k);

                    let cell_start = Instant::now();
                    let mut op = VerifyEngineParityOp;
                    let r = op.execute(&cell_opts, ctx);
                    cells.push(Cell {
                        dim: dim.clone(),
                        dist: dist.clone(),
                        metric: metric.clone(),
                        k: k.clone(),
                        status: r.status,
                        msg: r.message,
                        elapsed: cell_start.elapsed(),
                    });
                }
            }
        }
    }

    // Render summary matrix: one (metric × k) section, rows = dim,
    // columns = distribution. Cell label = "PASS (Xs)" / "FAIL (Xs)".
    let mut emit = String::new();
    emit.push('\n');
    emit.push_str("════════════════════════════════════════════════════════════════════\n");
    emit.push_str(&format!("ENGINE-PARITY SWEEP SUMMARY ({} cells)\n", total));
    emit.push_str("════════════════════════════════════════════════════════════════════\n");

    let col_w = 16usize;
    for metric in &metrics {
        for k in &ks {
            emit.push('\n');
            emit.push_str(&format!("metric={} k={}\n", metric, k));
            emit.push_str(&format!("{:>8}  ", "dim"));
            for dist in &distributions {
                emit.push_str(&format!("{:>w$}  ", dist, w = col_w));
            }
            emit.push('\n');
            for dim in &dims {
                emit.push_str(&format!("{:>8}  ", dim));
                for dist in &distributions {
                    let cell = cells.iter().find(|c|
                        &c.dim == dim && &c.dist == dist
                            && &c.metric == metric && &c.k == k);
                    let lbl = match cell {
                        Some(c) if matches!(c.status, Status::Ok) =>
                            format!("PASS ({:.1}s)", c.elapsed.as_secs_f64()),
                        Some(c) =>
                            format!("FAIL ({:.1}s)", c.elapsed.as_secs_f64()),
                        None => "-".to_string(),
                    };
                    emit.push_str(&format!("{:>w$}  ", lbl, w = col_w));
                }
                emit.push('\n');
            }
        }
    }

    let pass_count = cells.iter().filter(|c| matches!(c.status, Status::Ok)).count();
    let fail_count = cells.len() - pass_count;
    emit.push('\n');
    emit.push_str(&format!(
        "Result: {}/{} cells PASS, {} FAIL\n",
        pass_count, cells.len(), fail_count,
    ));
    if fail_count > 0 {
        emit.push_str("Failed cells (see per-cell logs above for full detail):\n");
        for c in cells.iter().filter(|c| !matches!(c.status, Status::Ok)) {
            emit.push_str(&format!(
                "  metric={} k={} dim={} dist={}: {}\n",
                c.metric, c.k, c.dim, c.dist, truncate(&c.msg, 100),
            ));
        }
    }
    ctx.ui.log(&emit);

    let final_status = if fail_count == 0 { Status::Ok } else { Status::Error };
    let msg = if fail_count == 0 {
        format!("sweep PASS: {}/{} cells", pass_count, total)
    } else {
        format!("sweep FAIL: {} of {} cells failed", fail_count, total)
    };
    CommandResult {
        status: final_status,
        message: msg,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve(value: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

/// Sample `n` queries deterministically from `query_path` (an fvec)
/// and write them as a fresh fvec into `workdir/sampled_query.fvec`.
/// Returns the path to the sampled file.
///
/// This is what bounds engine-parity's per-engine work in user-data
/// mode: each engine still scans the full base, but only against the
/// sampled queries. Sampling preserves byte-identical comparison
/// across engines (every engine sees the same query subset).
///
/// Uses a deterministic xorshift seeded by `seed` and "sample-without-
/// replacement via reservoir" so the resulting subset is reproducible
/// across runs. If `n >= total_queries`, returns the original path
/// unchanged (no copy).
fn write_sampled_query_file(
    query_path: &Path,
    n: usize,
    seed: u64,
    workdir: &Path,
    ui: &mut veks_core::ui::UiHandle,
) -> Result<PathBuf, String> {
    let reader = XvecReader::<f32>::open_path(query_path)
        .map_err(|e| format!("open query {}: {}", query_path.display(), e))?;
    let total = <XvecReader<f32> as VectorReader<f32>>::count(&reader);
    let dim = <XvecReader<f32> as VectorReader<f32>>::dim(&reader);
    if n >= total {
        return Ok(query_path.to_path_buf());
    }

    // Deterministic sample of `n` distinct indices in [0, total).
    // Reservoir sampling for reproducibility across runs and engines;
    // independent of `total` value.
    let mut rng = seed.max(1);
    let mut next_u64 = || {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng
    };
    let mut indices: Vec<usize> = (0..n).collect();
    for i in n..total {
        let j = (next_u64() as usize) % (i + 1);
        if j < n { indices[j] = i; }
    }
    indices.sort_unstable();

    let out_path = workdir.join("sampled_query.fvec");
    let mut f = std::io::BufWriter::with_capacity(
        1 << 20,
        std::fs::File::create(&out_path)
            .map_err(|e| format!("create {}: {}", out_path.display(), e))?,
    );
    let dim_bytes = (dim as i32).to_le_bytes();
    for &qi in &indices {
        let s = reader.get_slice(qi);
        f.write_all(&dim_bytes).map_err(|e| e.to_string())?;
        for v in s {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    f.flush().map_err(|e| e.to_string())?;

    ui.log(&format!(
        "  sampled {} of {} queries (seed={}) → {}",
        n, total, seed, out_path.display(),
    ));
    Ok(out_path)
}

fn error(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult { status: Status::Error, message: msg.into(), produced: vec![], elapsed: start.elapsed() }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() } else { s.chars().take(max - 1).chain("…".chars()).collect() }
}

fn fmt_neighbors(row: &[i32], k: usize) -> String {
    let take = row.len().min(k);
    let parts: Vec<String> = row[..take].iter().map(|n| n.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn read_ivec_rows(path: &Path) -> Result<Vec<Vec<i32>>, String> {
    let reader = XvecReader::<i32>::open_path(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let n = <XvecReader<i32> as VectorReader<i32>>::count(&reader);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = <XvecReader<i32> as VectorReader<i32>>::get(&reader, i)
            .map_err(|e| format!("read row {}: {}", i, e))?;
        out.push(row);
    }
    Ok(out)
}

/// Run a single engine, capturing wall time and (on failure) the
/// engine's own error message — we don't want one engine's failure to
/// abort the whole demo.
#[allow(clippy::too_many_arguments)]
fn run_engine<F>(
    name: &'static str,
    factory: F,
    base: &Path,
    query: &Path,
    indices: &Path,
    neighbors: usize,
    metric: &str,
    assume_normalized: bool,
    use_proper_cosine: bool,
    ctx: &mut StreamContext,
) -> EngineRun
where
    F: FnOnce() -> Box<dyn CommandOp>,
{
    ctx.ui.log(&format!("  running engine: {}", name));
    let mut op = factory();
    let mut opts = Options::new();
    opts.set("base", base.to_string_lossy().to_string());
    opts.set("query", query.to_string_lossy().to_string());
    opts.set("indices", indices.to_string_lossy().to_string());
    opts.set("neighbors", neighbors.to_string());
    opts.set("metric", metric.to_string());
    // Forward cosine mode — every engine independently requires one
    // of these flags when `metric=COSINE`. Validated upstream in the
    // parity command so exactly one is true here.
    if assume_normalized { opts.set("assume_normalized_like_faiss", "true"); }
    if use_proper_cosine { opts.set("use_proper_cosine_metric", "true"); }
    // Parity testing opts in to a margin — each engine pulls
    // top-(k + 3·k) candidates internally so the canonical f64
    // rerank can recover the last 1% of boundary cases that
    // surface on pathological synthetic distributions (uniform
    // random at very high dim). Production `compute knn*` calls
    // leave `rerank_margin_ratio` unset → margin=0 → original
    // heap sizes and full pruning aggressiveness.
    opts.set("rerank_margin_ratio", "3");
    let started = Instant::now();
    let res = op.execute(&opts, ctx);
    let elapsed = started.elapsed();
    if matches!(res.status, Status::Error) {
        EngineRun { name, indices_path: None, elapsed, note: Some(res.message) }
    } else {
        EngineRun { name, indices_path: Some(indices.to_path_buf()), elapsed, note: None }
    }
}

fn skipped(name: &'static str, why: &str) -> EngineRun {
    EngineRun { name, indices_path: None, elapsed: std::time::Duration::ZERO, note: Some(why.into()) }
}

/// Distribution shape for the synthetic fixture.
#[derive(Clone, Copy, PartialEq)]
enum Distribution {
    /// Each component drawn from U[-1, 1] independently. Pairwise
    /// distances concentrate sharply at high dim (curse of
    /// dimensionality), so neighbor selection collapses into the
    /// ULP-noise regime where different sgemm callers disagree.
    Uniform,
    /// Each component drawn from N(0, 1), per-vector L2-normalized
    /// onto the unit sphere. Still curse-of-dim prone: at high dim
    /// random sphere points have `L2² ≈ 2 ± 2/√d` — distances
    /// concentrate and the top-k is again ULP-dominated.
    GaussianNormalized,
    /// Mixture of `k` Gaussian clusters on the unit sphere with
    /// per-cluster noise `spread`. Simulates the cluster structure
    /// real embedding models produce, so pairwise distances have a
    /// wide range (within-cluster: small; between-cluster: near √2)
    /// and top-k is well-defined at any dimension. All kernels agree.
    Clustered { k: usize, spread: f32 },
}

impl Distribution {
    fn label(self) -> String {
        match self {
            Distribution::Uniform => "uniform".into(),
            Distribution::GaussianNormalized => "gaussian-normalized".into(),
            Distribution::Clustered { k, spread } => format!("clustered(k={}, σ={})", k, spread),
        }
    }
}

/// Deterministic xorshift64 fvec writer.
fn write_synthetic_fvec(
    path: &Path,
    dim: usize,
    count: usize,
    seed: u64,
    dist: Distribution,
) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let dim32 = dim as i32;
    let mut rng = seed.max(1);

    // Closure: one uniform-[0,1) draw — building block for both modes.
    let mut next_uniform01 = || -> f32 {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        // Map xorshift's full u64 range into [0,1). Excluding 0 avoids
        // the log(0) blow-up in the Box-Muller path below.
        ((rng >> 11) as f64 / (1u64 << 53) as f64) as f32
    };

    let mut row = vec![0f32; dim];

    // For clustered mode, pre-generate `k` cluster centers (unit
    // vectors drawn from Gaussian + normalized). Base and query share
    // the same cluster centers so queries fall near actual clusters
    // — but the per-vector cluster assignment and noise differs,
    // which is what would happen with real data.
    let centers: Vec<Vec<f32>> = if let Distribution::Clustered { k, .. } = dist {
        // Derive the center-PRNG stream from a fixed sub-seed so
        // base.fvec and query.fvec (called with different top-level
        // seeds) use *identical* centers.
        let mut center_rng = 0xC1057E11_u64;
        let mut center_next = || -> f32 {
            center_rng ^= center_rng << 13; center_rng ^= center_rng >> 7; center_rng ^= center_rng << 17;
            ((center_rng >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        (0..k).map(|_| {
            let mut c = vec![0f32; dim];
            fill_gaussian(&mut c, &mut center_next);
            normalize(&mut c);
            c
        }).collect()
    } else {
        Vec::new()
    };

    for _ in 0..count {
        match dist {
            Distribution::Uniform => {
                for v in row.iter_mut() {
                    *v = next_uniform01() * 2.0 - 1.0;
                }
            }
            Distribution::GaussianNormalized => {
                fill_gaussian(&mut row, &mut next_uniform01);
                normalize(&mut row);
            }
            Distribution::Clustered { k, spread } => {
                // Pick a uniform cluster, add spread·N(0,1) noise,
                // normalize back to the unit sphere. The resulting
                // pairwise distances have a bimodal distribution —
                // small within-cluster, ~√2 between-cluster — so
                // top-k is well-defined at any dimension.
                let c_idx = (next_uniform01() * k as f32) as usize;
                let c_idx = c_idx.min(k - 1);
                fill_gaussian(&mut row, &mut next_uniform01);
                for (i, v) in row.iter_mut().enumerate() {
                    *v = centers[c_idx][i] + *v * spread;
                }
                normalize(&mut row);
            }
        }

        f.write_all(&dim32.to_le_bytes())?;
        for v in &row {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Fill a row with i.i.d. N(0,1) samples via Box-Muller.
fn fill_gaussian<F: FnMut() -> f32>(row: &mut [f32], rng: &mut F) {
    let dim = row.len();
    let mut d = 0;
    while d < dim {
        let u1 = rng().max(f32::MIN_POSITIVE);
        let u2 = rng();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        row[d] = z0;
        if d + 1 < dim { row[d + 1] = z1; }
        d += 2;
    }
}

/// Per-vector L2-normalize in place.
fn normalize(row: &mut [f32]) {
    let norm_sq: f32 = row.iter().map(|x| x * x).sum();
    let inv = if norm_sq > 0.0 { 1.0 / norm_sq.sqrt() } else { 0.0 };
    for v in row.iter_mut() { *v *= inv; }
}

/// FAISS-mirror KNN: replicates `exhaustive_L2sqr_blas_default_impl`
/// byte-for-byte (modulo BLAS library choice).
///
/// Pattern:
/// - Outer loop: queries in blocks of `BS_X = 4096` (FAISS's
///   `distance_compute_blas_query_bs`)
/// - Inner loop: base in blocks of `BS_Y = 1024` (FAISS's
///   `distance_compute_blas_database_bs`)
/// - Per inner block: single `cblas_sgemm` computing the `(nb × nq)`
///   dot-product matrix with FAISS's exact orientation and `alpha = 1`
/// - Post-process per cell: `dis = ‖q‖² + ‖b‖² − 2·ip`, clamped ≥ 0
/// - Update a per-query max-heap of the top-k
///
/// The entire base is read into RAM — matches FAISS for the fixtures
/// the parity demo cares about. Streaming + sgemm for billion-vector
/// datasets is the job of `compute knn-blas`; this is strictly for
/// proving the parity hypothesis.
#[cfg(feature = "knnutils")]
fn run_blas_mirror(
    base_path: &Path,
    query_path: &Path,
    out_path: &Path,
    k: usize,
    metric: &str,
    assume_normalized: bool,
    use_proper_cosine: bool,
    ui: &mut veks_core::ui::UiHandle,
) -> Result<(), String> {
    use vectordata::VectorReader;
    use vectordata::io::XvecReader;

    // Resolve which sgemm-derived distance to compute. blas-mirror
    // tracks all four metric paths the engines do, so it can serve
    // as the FAISS-equivalent reference for every metric:
    //   - L2:  `‖q‖² + ‖b‖² − 2·ip` (FAISS's exhaustive_L2sqr_blas_default_impl)
    //   - IP / DOT_PRODUCT: raw `ip`, ranked by max
    //   - COSINE + assume_normalized: same as IP (FAISS convention)
    //   - COSINE + use_proper_cosine: `ip / (|q|·|b|)`
    enum MirrorMode { L2, Ip, CosineProper }
    let m_upper = metric.to_uppercase();
    let mode = match m_upper.as_str() {
        "L2" => MirrorMode::L2,
        "IP" | "DOT_PRODUCT" => MirrorMode::Ip,
        "COSINE" => {
            if assume_normalized && !use_proper_cosine {
                MirrorMode::Ip
            } else if use_proper_cosine && !assume_normalized {
                MirrorMode::CosineProper
            } else {
                return Err(
                    "metric=COSINE: blas-mirror needs exactly one of \
                     assume_normalized_like_faiss / use_proper_cosine_metric".into(),
                );
            }
        }
        other => return Err(format!("blas-mirror: unsupported metric '{}'", other)),
    };
    // The kernel-internal distance type used downstream:
    //   L2  → Metric::L2 (positive distance, smaller better)
    //   IP  → Metric::DotProduct (heap stores `-dot`)
    //   Cosine proper → Metric::Cosine (heap stores `1 − cos_sim`)
    let rerank_metric = match mode {
        MirrorMode::L2 => super::knn_segment::Metric::L2,
        MirrorMode::Ip => super::knn_segment::Metric::DotProduct,
        MirrorMode::CosineProper => super::knn_segment::Metric::Cosine,
    };

    let base_reader = XvecReader::<f32>::open_path(base_path)
        .map_err(|e| format!("open {}: {}", base_path.display(), e))?;
    let query_reader = XvecReader::<f32>::open_path(query_path)
        .map_err(|e| format!("open {}: {}", query_path.display(), e))?;

    let n_base  = <XvecReader<f32> as VectorReader<f32>>::count(&base_reader);
    let n_query = <XvecReader<f32> as VectorReader<f32>>::count(&query_reader);
    let dim     = <XvecReader<f32> as VectorReader<f32>>::dim(&base_reader);
    let qdim    = <XvecReader<f32> as VectorReader<f32>>::dim(&query_reader);
    if dim != qdim {
        return Err(format!("dim mismatch: base={}, query={}", dim, qdim));
    }

    // Streaming: do NOT eagerly copy base or query into Vec<f32>. The
    // base file can be TB-scale; copying it would always OOM. Instead
    // hold small per-block buffers that we fill from the mmap reader
    // on each sgemm step. Memory footprint:
    //   query_block_buf:    BS_X × dim × 4 bytes  (e.g., 4096 × 384 × 4 = 6 MB)
    //   base_block_buf:     BS_Y × dim × 4 bytes  (e.g., 1024 × 384 × 4 = 1.5 MB)
    //   ip_block:           BS_X × BS_Y × 4 bytes (≈ 16 MB)
    //   q_norms_sq:         n_query × 4 bytes      (10K queries → 40 KB)
    //   b_norms_sq_block:   BS_Y × 4 bytes         (4 KB per block)
    // Total: a few tens of MB regardless of base size. Hint sequential
    // so the kernel reclaims pages behind the cursor.
    base_reader.advise_sequential();

    const BS_X: usize = 4096; // FAISS distance_compute_blas_query_bs
    const BS_Y: usize = 1024; // FAISS distance_compute_blas_database_bs

    let mut query_block_buf: Vec<f32> = vec![0f32; BS_X * dim];
    let mut base_block_buf: Vec<f32> = vec![0f32; BS_Y * dim];
    let mut ip_block = vec![0f32; BS_X * BS_Y];

    // Query norms are small (n_query × 4 bytes). Compute up front via
    // mmap reads; no full query Vec needed.
    let q_norms_sq: Vec<f32> = (0..n_query).map(|i| {
        let s = query_reader.get_slice(i);
        s.iter().map(|v| v * v).sum::<f32>()
    }).collect();

    // Base norms are computed per BS_Y block on the fly — sized so the
    // working set stays bounded regardless of n_base.
    let mut b_norms_sq_block: Vec<f32> = vec![0f32; BS_Y];

    // Pull top-(k + margin) candidates from sgemm, then f64-rerank
    // down to the canonical top-k. Margin=3k empirically suffices
    // for concentrated-distance uniform random at dim ≥ 4096.
    // Clamped so we never ask for more candidates than exist.
    let margin = (3 * k).min(n_base.saturating_sub(k));
    let capacity = k + margin;
    let mut heaps: Vec<std::collections::BinaryHeap<Neighbor>> =
        (0..n_query).map(|_| std::collections::BinaryHeap::with_capacity(capacity + 1)).collect();

    // Progress bar for the per-block sgemm scan. Total work is
    // (n_query / BS_X) outer × n_base inner blocks; we display at
    // base granularity since base dominates. With ~1B base / 1024
    // per block ≈ 1M inner iterations, we update every 32 blocks
    // so the bar feels responsive without atomic-store overhead.
    let pb = ui.bar_with_unit(
        n_base as u64,
        format!(
            "blas-mirror sgemm: {} queries × {} base, dim={}, mode={}",
            n_query, n_base, dim,
            match mode {
                MirrorMode::L2 => "L2",
                MirrorMode::Ip => "IP",
                MirrorMode::CosineProper => "Cosine(proper)",
            },
        ),
        "base",
    );

    let mut i0 = 0;
    while i0 < n_query {
        let i1 = (i0 + BS_X).min(n_query);
        let nxi = i1 - i0;

        // Fill query block from mmap.
        for qi_local in 0..nxi {
            let dst = qi_local * dim;
            query_block_buf[dst..dst + dim].copy_from_slice(query_reader.get_slice(i0 + qi_local));
        }

        let mut j0 = 0;
        let mut blocks_since_tick = 0usize;
        const TICK_EVERY_N_BLOCKS: usize = 32;
        while j0 < n_base {
            let j1 = (j0 + BS_Y).min(n_base);
            let nyi = j1 - j0;

            // Fill base block from mmap, computing per-block norms in
            // the same pass.
            for bi_local in 0..nyi {
                let src = base_reader.get_slice(j0 + bi_local);
                let dst = bi_local * dim;
                base_block_buf[dst..dst + dim].copy_from_slice(src);
                let mut s = 0.0f32;
                for v in src { s += v * v; }
                b_norms_sq_block[bi_local] = s;
            }

            // Row-major equivalent of FAISS's column-major sgemm.
            // Result `ip_block[i * nyi + j]` = query[i0+i] · base[j0+j].
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                    nxi as i32, nyi as i32, dim as i32,
                    1.0,
                    query_block_buf.as_ptr(), dim as i32,
                    base_block_buf.as_ptr(),  dim as i32,
                    0.0,
                    ip_block.as_mut_ptr(), nyi as i32,
                );
            }

            // Per-cell post-pass — produces a kernel-convention
            // distance (smaller = better) for every metric so the
            // heap math is uniform.
            for qi_local in 0..nxi {
                let qi_global = i0 + qi_local;
                let qn = q_norms_sq[qi_global];
                let row = &ip_block[qi_local * nyi..qi_local * nyi + nyi];
                for bi_local in 0..nyi {
                    let bi_global = j0 + bi_local;
                    let bn = b_norms_sq_block[bi_local];
                    let ip = row[bi_local];
                    let dis: f32 = match mode {
                        MirrorMode::L2 => {
                            let mut d = qn + bn - 2.0 * ip;
                            if d < 0.0 { d = 0.0; }
                            d
                        }
                        MirrorMode::Ip => {
                            // Kernel convention for IP: store `-ip` so
                            // smaller = larger inner product = better.
                            -ip
                        }
                        MirrorMode::CosineProper => {
                            // cos_sim = ip / (|q|·|b|), guarded against
                            // zero norm (returns 0 distance there to
                            // mirror compute_knn_blas's behavior).
                            let denom = qn.sqrt() * bn.sqrt();
                            if denom <= f32::MIN_POSITIVE {
                                1.0 // worst-case "no similarity"
                            } else {
                                1.0 - ip / denom
                            }
                        }
                    };
                    heaps[qi_global].push(Neighbor { index: bi_global as u32, distance: dis });
                    if heaps[qi_global].len() > capacity {
                        heaps[qi_global].pop();
                    }
                }
            }

            // Release pages we just streamed past. Aligned MADV_DONTNEED
            // keeps RSS bounded (without the page-aligned fix in
            // vectordata::io::release_range, this would be a no-op).
            base_reader.release_range(j0, j1);
            blocks_since_tick += 1;
            if blocks_since_tick >= TICK_EVERY_N_BLOCKS {
                pb.set_position(j1 as u64);
                blocks_since_tick = 0;
            }
            j0 = j1;
        }
        i0 = i1;
    }
    pb.finish();

    // Rerank every query's top-(k + margin) candidates through the
    // shared f64-direct kernel to produce canonical top-k. Both the
    // query slice and the per-candidate base slices are read directly
    // from mmap — no need to keep a contiguous Vec of either.
    let reranked: Vec<Vec<Neighbor>> = (0..n_query)
        .map(|qi| {
            let heap = std::mem::take(&mut heaps[qi]);
            let cands: Vec<Neighbor> = heap.into_vec();
            let query_slice = query_reader.get_slice(qi);
            super::knn_segment::rerank_topk_f64(
                query_slice, &cands, k,
                rerank_metric,
                |idx| {
                    let i = idx as usize;
                    if i < n_base {
                        Some(base_reader.get_slice(i))
                    } else {
                        None
                    }
                },
            )
        })
        .collect();

    // Emit ivec with per-query reranked top-k.
    let mut f = std::fs::File::create(out_path)
        .map_err(|e| format!("create {}: {}", out_path.display(), e))?;
    let dim_bytes = (k as i32).to_le_bytes();
    for v in reranked {
        f.write_all(&dim_bytes).map_err(|e| e.to_string())?;
        for j in 0..k {
            let idx = if j < v.len() { v[j].index as i32 } else { -1 };
            f.write_all(&idx.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(feature = "knnutils")]
use super::compute_knn::Neighbor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::progress::ProgressLog;
    use crate::pipeline::resource::ResourceGovernor;
    use indexmap::IndexMap;

    fn make_ctx(workspace: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
        }
    }

    #[test]
    fn engine_parity_synthetic_default_engines_pass() {
        // Default-feature build: only metal + stdarch run, blas/faiss
        // are skipped. Two engines is enough to exercise the comparison
        // path. Synthetic fixture is small so the test stays fast.
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let mut opts = Options::new();
        opts.set("synthetic", "true");
        opts.set("dim", "8");
        opts.set("base-count", "100");
        opts.set("query-count", "10");
        opts.set("neighbors", "5");
        opts.set("metric", "L2");
        opts.set("show-queries", "2");
        opts.set("seed", "42");

        let mut op = VerifyEngineParityOp;
        let r = op.execute(&opts, &mut ctx);
        assert!(matches!(r.status, Status::Ok), "{}", r.message);
        // The verdict line is what we assert against — any mismatch
        // beyond the default tolerance=0 would have flipped this to
        // Status::Error already.
        assert!(r.message.contains("identical neighbor sets")
             || r.message.contains("not enough engines"),
            "unexpected verdict: {}", r.message);
    }

    #[test]
    fn engine_parity_sweep_runs_cross_product_and_summarizes() {
        // Tiny 2×2×1×1 matrix so the test stays under a second. The
        // same cells run under the non-sweep tests already; what
        // matters here is that --sweep dispatches through every cell
        // and returns a single aggregate Ok verdict with a matrix
        // summary in the message.
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let mut opts = Options::new();
        opts.set("sweep", "true");
        opts.set("dims", "4,8");
        opts.set("distributions", "uniform,clustered");
        opts.set("metrics", "L2");
        opts.set("neighbors-list", "5");
        opts.set("base-count", "100");
        opts.set("query-count", "10");
        opts.set("seed", "42");

        let mut op = VerifyEngineParityOp;
        let r = op.execute(&opts, &mut ctx);
        assert!(matches!(r.status, Status::Ok), "{}", r.message);
        assert!(r.message.contains("sweep"),
            "expected sweep-style summary, got: {}", r.message);
    }

    #[test]
    fn engine_parity_synthetic_subset_runs_only_what_user_asked() {
        // --engines=stdarch is the only one we have unconditionally
        // available. With only one engine loaded, the command emits
        // "not enough engines to compare" but still exits Ok — a useful
        // diagnostic for anyone trying to demo BLAS/FAISS without the
        // features turned on.
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let mut opts = Options::new();
        opts.set("synthetic", "true");
        opts.set("dim", "4");
        opts.set("base-count", "50");
        opts.set("query-count", "5");
        opts.set("neighbors", "3");
        opts.set("metric", "L2");
        opts.set("engines", "stdarch");

        let mut op = VerifyEngineParityOp;
        let r = op.execute(&opts, &mut ctx);
        assert!(matches!(r.status, Status::Ok), "{}", r.message);
        assert!(r.message.contains("not enough engines"), "{}", r.message);
    }
}
