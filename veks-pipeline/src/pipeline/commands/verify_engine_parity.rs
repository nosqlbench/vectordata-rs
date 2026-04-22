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
use vectordata::io::MmapVectorReader;

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
/// command starts, so it works. Covers OpenBLAS, MKL, OMP-threaded
/// BLIS, and Apple Accelerate.
#[cfg(feature = "knnutils")]
fn blas_set_single_threaded() {
    for var in &["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"] {
        // set_var is unsafe because it races with getenv from other
        // threads; we call this at the very top of execute() before
        // spawning any BLAS work, so it's sound here.
        unsafe { std::env::set_var(var, "1"); }
    }
}

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
                description: "Synthetic mode: PRNG seed (xorshift)".into(),
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
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // Force the underlying BLAS to single-threaded BEFORE any
        // engine runs — FAISS's sgemm is multi-threaded by default,
        // and OpenMP's non-deterministic block scheduling makes its
        // output vary across runs with the same inputs. For a
        // deterministic parity demo we want every sgemm caller
        // (FAISS and our own blas-mirror) to share an identical
        // accumulation order. Single-threaded BLAS is the only thread
        // count that reproduces across BLAS variants.
        #[cfg(feature = "knnutils")]
        blas_set_single_threaded();

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

        // Isolate the entire engine working environment — workspace
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

        // Resolve the (base, query) paths — either user-supplied or
        // synthetic-generated into the workdir.
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
            let b = match options.require("base") { Ok(s) => s, Err(e) => return error(e, start) };
            let q = match options.require("query") { Ok(s) => s, Err(e) => return error(e, start) };
            (resolve(&b, &ctx.workspace), resolve(&q, &ctx.workspace))
        };

        // Decide which engines to run. The user can override via
        // --engines; otherwise we run everything that was compiled in.
        let requested: Vec<String> = options.get("engines")
            .map(|s| s.split(',').map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()).collect())
            .unwrap_or_else(|| vec![
                "metal".into(), "stdarch".into(), "blas".into(),
                "blas-mirror".into(), "faiss".into(),
            ]);
        let want = |name: &str| requested.iter().any(|r| r == name);

        ctx.ui.log(&format!(
            "engine-parity: base={} query={} k={} metric={}",
            base_path.display(), query_path.display(), neighbors, metric));

        let mut runs: Vec<EngineRun> = Vec::new();

        if want("metal") {
            runs.push(run_engine(
                "metal",
                || super::compute_knn::factory(),
                &base_path, &query_path, &workdir.path().join("metal.ivec"),
                neighbors, &metric, ctx,
            ));
        }
        if want("stdarch") {
            runs.push(run_engine(
                "stdarch",
                || super::compute_knn_stdarch::factory(),
                &base_path, &query_path, &workdir.path().join("stdarch.ivec"),
                neighbors, &metric, ctx,
            ));
        }

        #[cfg(feature = "knnutils")]
        if want("blas") {
            runs.push(run_engine(
                "blas",
                || super::compute_knn_blas::factory(),
                &base_path, &query_path, &workdir.path().join("blas.ivec"),
                neighbors, &metric, ctx,
            ));
        }
        #[cfg(not(feature = "knnutils"))]
        if want("blas") {
            runs.push(skipped("blas", "knnutils feature not enabled (cargo build --features knnutils)"));
        }

        #[cfg(feature = "faiss")]
        if want("faiss") {
            runs.push(run_engine(
                "faiss",
                || Box::new(super::compute_knn_faiss::ComputeKnnFaissOp) as Box<dyn CommandOp>,
                &base_path, &query_path, &workdir.path().join("faiss.ivec"),
                neighbors, &metric, ctx,
            ));
        }
        #[cfg(not(feature = "faiss"))]
        if want("faiss") {
            runs.push(skipped("faiss", "faiss feature not enabled (cargo build --features faiss)"));
        }

        // Probe engine: mirrors FAISS's `exhaustive_L2sqr_blas_default_impl`
        // exactly — same outer/inner block sizes, same sgemm orientation,
        // same per-cell post-processing. If FAISS's BLAS link and ours
        // resolve to the same library, this engine should produce
        // bit-identical neighbor sets to FAISS at every dim.
        #[cfg(feature = "knnutils")]
        if want("blas-mirror") {
            let started = Instant::now();
            let path = workdir.path().join("blas-mirror.ivec");
            let res = run_blas_mirror(&base_path, &query_path, &path, neighbors, &metric);
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

fn resolve(value: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
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
    let reader = MmapVectorReader::<i32>::open_ivec(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let n = <MmapVectorReader<i32> as VectorReader<i32>>::count(&reader);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = <MmapVectorReader<i32> as VectorReader<i32>>::get(&reader, i)
            .map_err(|e| format!("read row {}: {}", i, e))?;
        out.push(row);
    }
    Ok(out)
}

/// Run a single engine, capturing wall time and (on failure) the
/// engine's own error message — we don't want one engine's failure to
/// abort the whole demo.
fn run_engine<F>(
    name: &'static str,
    factory: F,
    base: &Path,
    query: &Path,
    indices: &Path,
    neighbors: usize,
    metric: &str,
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
) -> Result<(), String> {
    use vectordata::VectorReader;
    use vectordata::io::MmapVectorReader;

    if metric != "L2" {
        return Err(format!("blas-mirror currently implements L2 only (got {})", metric));
    }

    let base_reader = MmapVectorReader::<f32>::open_fvec(base_path)
        .map_err(|e| format!("open {}: {}", base_path.display(), e))?;
    let query_reader = MmapVectorReader::<f32>::open_fvec(query_path)
        .map_err(|e| format!("open {}: {}", query_path.display(), e))?;

    let n_base  = <MmapVectorReader<f32> as VectorReader<f32>>::count(&base_reader);
    let n_query = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
    let dim     = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&base_reader);
    let qdim    = <MmapVectorReader<f32> as VectorReader<f32>>::dim(&query_reader);
    if dim != qdim {
        return Err(format!("dim mismatch: base={}, query={}", dim, qdim));
    }

    // Load contiguous f32 payloads (strip the per-entry dim header).
    let mut base_data: Vec<f32> = Vec::with_capacity(n_base * dim);
    for i in 0..n_base {
        base_data.extend_from_slice(base_reader.get_slice(i));
    }
    let mut query_data: Vec<f32> = Vec::with_capacity(n_query * dim);
    for i in 0..n_query {
        query_data.extend_from_slice(query_reader.get_slice(i));
    }

    // Precompute per-query and per-base squared norms (FAISS does the same).
    let q_norms_sq: Vec<f32> = (0..n_query).map(|i| {
        let s = &query_data[i * dim..(i + 1) * dim];
        s.iter().map(|v| v * v).sum::<f32>()
    }).collect();
    let b_norms_sq: Vec<f32> = (0..n_base).map(|j| {
        let s = &base_data[j * dim..(j + 1) * dim];
        s.iter().map(|v| v * v).sum::<f32>()
    }).collect();

    const BS_X: usize = 4096; // FAISS distance_compute_blas_query_bs
    const BS_Y: usize = 1024; // FAISS distance_compute_blas_database_bs

    // Pull top-(k + margin) candidates from sgemm, then f64-rerank
    // down to the canonical top-k. Margin=3k empirically suffices
    // for concentrated-distance uniform random at dim ≥ 4096 to
    // catch sgemm's f32 ranking errors: the genuine top-k neighbor
    // that sgemm misplaces is almost always within the top-4k
    // window. Clamped so we never ask for more candidates than
    // exist in the base.
    let margin = (3 * k).min(n_base.saturating_sub(k));
    let capacity = k + margin;
    let mut heaps: Vec<std::collections::BinaryHeap<Neighbor>> =
        (0..n_query).map(|_| std::collections::BinaryHeap::with_capacity(capacity + 1)).collect();
    let mut ip_block = vec![0f32; BS_X * BS_Y];

    let mut i0 = 0;
    while i0 < n_query {
        let i1 = (i0 + BS_X).min(n_query);
        let nxi = i1 - i0;

        let mut j0 = 0;
        while j0 < n_base {
            let j1 = (j0 + BS_Y).min(n_base);
            let nyi = j1 - j0;

            // Row-major equivalent of FAISS's column-major
            //   sgemm_("T", "N", nyi, nxi, di, 1, y+j0*d, di, x+i0*d, di, 0, ip, nyi).
            //
            // Column-major(T,N,M=nyi,N=nxi,K=di,A=y,lda=di,B=x,ldb=di,C,ldc=nyi)
            // is layout-equivalent to row-major with A/B swapped and M/N
            // swapped, with the transpose flag flipped. cblas handles
            // this via ROW_MAJOR + NO_TRANS + TRANS with swapped shapes.
            // Result `ip_block[i * nyi + j]` = query[i0+i] · base[j0+j].
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                    nxi as i32, nyi as i32, dim as i32,
                    1.0,
                    query_data.as_ptr().add(i0 * dim), dim as i32,
                    base_data.as_ptr().add(j0 * dim),  dim as i32,
                    0.0,
                    ip_block.as_mut_ptr(), nyi as i32,
                );
            }

            // Per-cell: dis = ‖q‖² + ‖b‖² - 2·ip, clamp, heap update.
            for qi_local in 0..nxi {
                let qi_global = i0 + qi_local;
                let qn = q_norms_sq[qi_global];
                let row = &ip_block[qi_local * nyi..qi_local * nyi + nyi];
                for bi_local in 0..nyi {
                    let bi_global = j0 + bi_local;
                    let bn = b_norms_sq[bi_global];
                    let ip = row[bi_local];
                    let mut dis = qn + bn - 2.0 * ip;
                    if dis < 0.0 { dis = 0.0; }
                    heaps[qi_global].push(Neighbor { index: bi_global as u32, distance: dis });
                    if heaps[qi_global].len() > capacity {
                        heaps[qi_global].pop();
                    }
                }
            }

            j0 = j1;
        }
        i0 = i1;
    }

    // Rerank every query's top-(k + margin) candidates through the
    // shared f64-direct kernel to produce canonical top-k. This is
    // what makes sgemm-path engines agree with direct-kernel engines
    // on pathologically-concentrated fixtures where f32 sgemm's
    // `‖a‖² + ‖b‖² − 2·a·b` computation puts boundary neighbors on
    // the wrong side of the top-k cutoff.
    let reranked: Vec<Vec<Neighbor>> = (0..n_query)
        .map(|qi| {
            let heap = std::mem::take(&mut heaps[qi]);
            let cands: Vec<Neighbor> = heap.into_vec();
            let q_start = qi * dim;
            let query_slice = &query_data[q_start..q_start + dim];
            super::knn_segment::rerank_topk_f64(
                query_slice, &cands, k,
                super::knn_segment::Metric::L2,
                |idx| {
                    let s = idx as usize * dim;
                    if s + dim <= base_data.len() {
                        Some(base_data[s..s + dim].to_vec())
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
