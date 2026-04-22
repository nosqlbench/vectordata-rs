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
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let neighbors: usize = parse_int(options, "neighbors", 10);
        let metric = options.get("metric").unwrap_or("L2").to_string();
        let show_queries: usize = parse_int(options, "show-queries", 5);
        let boundary_tolerance: usize = parse_int(options, "boundary-tolerance", 0);
        let synthetic = options.get("synthetic")
            .map(|s| matches!(s, "true" | "1" | "yes"))
            .unwrap_or(false);

        let workdir = match tempfile::tempdir_in(&ctx.workspace) {
            Ok(d) => d,
            Err(e) => return error(format!("could not create workdir: {}", e), start),
        };

        // Resolve the (base, query) paths — either user-supplied or
        // synthetic-generated into the workdir.
        let (base_path, query_path) = if synthetic {
            let dim:        usize = parse_int(options, "dim", 128);
            let base_count: usize = parse_int(options, "base-count", 10_000);
            let query_count: usize = parse_int(options, "query-count", 100);
            let seed:       u64   = parse_int(options, "seed", 42) as u64;
            let bp = workdir.path().join("base.fvec");
            let qp = workdir.path().join("query.fvec");
            ctx.ui.log(&format!(
                "synthetic fixture: dim={}, base={}, query={}, seed={}",
                dim, base_count, query_count, seed));
            if let Err(e) = write_synthetic_fvec(&bp, dim, base_count, seed) {
                return error(format!("synthetic base write: {}", e), start);
            }
            if let Err(e) = write_synthetic_fvec(&qp, dim, query_count, seed.wrapping_add(1)) {
                return error(format!("synthetic query write: {}", e), start);
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
            .unwrap_or_else(|| vec!["metal".into(), "stdarch".into(), "blas".into(), "faiss".into()]);
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
        // its own classification, so we re-classify here against the
        // user-supplied tolerance: anything beyond `boundary_tolerance`
        // differing neighbors flips to FAIL, regardless of which bucket
        // it landed in.
        let (ref_name, ref_rows) = loaded.first().cloned().unwrap();
        let mut any_fail = false;
        emit.push('\n');
        emit.push_str(&format!(
            "Pair-wise comparison (reference = {}, boundary-tolerance = {}):\n",
            ref_name, boundary_tolerance,
        ));
        emit.push_str("VS                EXACT     SET   BOUND  EXCEED   TOTAL  VERDICT\n");
        emit.push_str("──────────────── ────── ─────── ─────── ─────── ──────── ────────\n");
        for (other_name, other_rows) in loaded.iter().skip(1) {
            let mut summary = VerifySummary::default();
            let mut over_tolerance = 0usize; // queries with diff > boundary_tolerance
            for q in 0..n_queries {
                let r = compare_query_ordinals(&other_rows[q], &ref_rows[q]);
                summary.record(&r);
                let diff = match r {
                    QueryResult::ExactMatch | QueryResult::SetMatch => 0,
                    QueryResult::BoundaryMismatch(d) | QueryResult::RealMismatch(d) => d,
                };
                if diff > boundary_tolerance {
                    over_tolerance += 1;
                }
            }
            let verdict = if over_tolerance == 0 { "PASS" } else { any_fail = true; "FAIL" };
            emit.push_str(&format!(
                "{:<16} {:>6} {:>7} {:>7} {:>7} {:>8}  {}\n",
                other_name,
                summary.exact_match,
                summary.set_match,
                summary.boundary_mismatch,
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

/// Deterministic xorshift64 fvec writer. Vectors are uniform in
/// [-1, 1] — same family as the in-tree parity tests use.
fn write_synthetic_fvec(path: &Path, dim: usize, count: usize, seed: u64) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let dim32 = dim as i32;
    let mut rng = seed.max(1);
    for _ in 0..count {
        f.write_all(&dim32.to_le_bytes())?;
        for _ in 0..dim {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let v: f32 = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

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
