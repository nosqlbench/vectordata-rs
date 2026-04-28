// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: verify KNN ground truth using FAISS brute-force search.
//!
//! Recomputes exact KNN via FAISS FlatIndex and compares against stored
//! ground truth (neighbor_indices). When run inside a dataset.yaml
//! directory without explicit paths, automatically iterates over all
//! profiles that have base_vectors, query_vectors, and neighbor_indices.
//!
//! This is a post-hoc verification tool — not wired into the pipeline DAG.

use std::path::{Path, PathBuf};
use std::time::Instant;

use faiss::Index;
use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use super::knn_compare::{self, VerifySummary};

pub struct VerifyKnnFaissOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyKnnFaissOp)
}

fn faiss_metric(metric_str: &str) -> Result<faiss::MetricType, String> {
    match metric_str.to_uppercase().as_str() {
        "L2" => Ok(faiss::MetricType::L2),
        "IP" | "DOT_PRODUCT" | "COSINE" => Ok(faiss::MetricType::InnerProduct),
        other => Err(format!(
            "unsupported FAISS metric: '{}'. Use L2, IP, DOT_PRODUCT, or COSINE", other)),
    }
}

impl CommandOp for VerifyKnnFaissOp {
    fn command_path(&self) -> &str {
        "verify knn-faiss"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify KNN ground truth via FAISS brute-force recomputation".into(),
            body: format!(
                r#"# verify knn-faiss

Verify KNN ground truth by recomputing exact nearest neighbors using
FAISS and comparing against stored results.

## Description

Builds a FAISS FlatIndex from base vectors, searches for k-nearest
neighbors of each query (or a random sample), and compares the result
sets against the stored neighbor_indices ground truth.

When run inside a directory with `dataset.yaml` and no explicit paths,
automatically discovers and verifies all profiles that have
base_vectors, query_vectors, and neighbor_indices.

Comparison uses the canonical set-based model: exact order differences
from tie-breaking are acceptable, and up to {} boundary neighbor
swaps (from BLAS/SIMD rounding) are tolerated.

## Options

{}"#,
                knn_compare::BOUNDARY_THRESHOLD,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description: "Base vectors loaded into FAISS index".into(),
                adjustable: false,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // faiss-sys static MKL + multi-threading = silent sgemm
        // corruption (see pipeline::blas_abi). This verifier relies
        // on `index.search()` being correct; force single-threaded.
        crate::pipeline::blas_abi::set_single_threaded_if_faiss();

        let metric_str = options.get("metric").unwrap_or("L2");
        let faiss_mt = match faiss_metric(metric_str) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };
        let sample: usize = options.get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0); // 0 = all queries
        let at_k: usize = options.get("at-k")
            .or_else(|| options.get("at_k"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0); // 0 = use full k from GT
        let limit: usize = options.get("limit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        // Accept "indices" as alias for "ground-truth" (pipeline compatibility)
        let has_explicit = options.get("base").is_some()
            && (options.get("ground-truth").is_some() || options.get("indices").is_some());

        // If explicit paths are provided, verify just those.
        if has_explicit {
            return self.verify_explicit(options, ctx, faiss_mt, metric_str, sample, at_k, limit, start);
        }

        // Otherwise, auto-discover from dataset.yaml.
        let ds_path = ctx.workspace.join("dataset.yaml");
        let config = match vectordata::dataset::DatasetConfig::load(&ds_path) {
            Ok(c) => c,
            Err(e) => return error_result(
                format!("no explicit paths and no dataset.yaml: {}", e), start),
        };

        let metric_from_config = config.distance_function()
            .unwrap_or("L2");
        let effective_metric = if options.get("metric").is_some() {
            metric_str
        } else {
            metric_from_config
        };
        let effective_faiss_mt = match faiss_metric(effective_metric) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };

        // Collect profiles with base, query, and GT.
        let mut profiles: Vec<(String, PathBuf, PathBuf, PathBuf, Option<u64>)> = Vec::new();
        for (name, profile) in &config.profiles.profiles {
            let base = profile.view("base_vectors")
                .map(|v| ctx.workspace.join(v.path()));
            let query = profile.view("query_vectors")
                .map(|v| ctx.workspace.join(v.path()));
            let gt = profile.view("neighbor_indices")
                .map(|v| ctx.workspace.join(v.path()));

            if let (Some(b), Some(q), Some(g)) = (base, query, gt) {
                if b.exists() && q.exists() && g.exists() {
                    profiles.push((name.clone(), b, q, g, profile.base_count));
                }
            }
        }

        if profiles.is_empty() {
            return error_result(
                "no profiles with base_vectors, query_vectors, and neighbor_indices found",
                start);
        }

        // Size order — sized profile names ("100k", "1m", …) sort by
        // their parsed value, not their leading digit string.
        profiles.sort_by(|a, b| {
            vectordata::dataset::profile::profile_sort_by_size(&a.0, a.4, &b.0, b.4)
        });

        ctx.ui.log(&format!(
            "verify knn-faiss-consolidated: KNN ground truth across {} profile{} via FAISS \
             IndexFlat oracle (single-threaded MKL, batch-capped); sample={}, metric={}",
            profiles.len(),
            if profiles.len() == 1 { "" } else { "s" },
            sample, effective_metric,
        ));

        let mut overall = VerifySummary::default();
        let mut all_pass = true;

        for (name, base_path, query_path, gt_path, _base_count) in &profiles {
            ctx.ui.log(&format!("\n── profile: {} ──────────────────────────────", name));

            match verify_profile(
                base_path, query_path, gt_path,
                effective_faiss_mt, effective_metric,
                sample, at_k, limit, ctx,
            ) {
                Ok(summary) => {
                    let pass = summary.real_mismatch == 0;
                    if !pass { all_pass = false; }
                    let status = if pass { "PASS" } else { "FAIL" };
                    ctx.ui.log(&format!(
                        "  {}: {}/{} queries — {} exact, {} set, {} boundary, {} fail",
                        status, summary.total, summary.total,
                        summary.exact_match, summary.set_match,
                        summary.boundary_mismatch, summary.real_mismatch));
                    overall.total += summary.total;
                    overall.exact_match += summary.exact_match;
                    overall.set_match += summary.set_match;
                    overall.boundary_mismatch += summary.boundary_mismatch;
                    overall.real_mismatch += summary.real_mismatch;
                }
                Err(e) => {
                    ctx.ui.log(&format!("  ERROR: {}", e));
                    all_pass = false;
                }
            }
        }

        ctx.ui.log(&format!(
            "\noverall: {} profiles, {}/{} queries — {} exact, {} set, {} boundary, {} fail",
            profiles.len(), overall.total, overall.total,
            overall.exact_match, overall.set_match,
            overall.boundary_mismatch, overall.real_mismatch));

        // Write JSON report if `output` option was supplied. Pipeline
        // runners declare this path as a post-execution artifact, so
        // skipping the write causes the step to be marked as failing
        // even when verification itself succeeded.
        let mut produced: Vec<PathBuf> = Vec::new();
        if let Some(out_str) = options.get("output") {
            let out_path = resolve_path(out_str, &ctx.workspace);
            let report = serde_json::json!({
                "type": "knn-faiss",
                "engine": "FAISS",
                "metric": effective_metric,
                "profiles_verified": profiles.len(),
                "queries_total": overall.total,
                "exact_match": overall.exact_match,
                "set_match": overall.set_match,
                "boundary_mismatch": overall.boundary_mismatch,
                "real_mismatch": overall.real_mismatch,
                "pass": all_pass,
            });
            if let Some(parent) = out_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap_or_default()) {
                Ok(()) => produced.push(out_path),
                Err(e) => ctx.ui.log(&format!("  WARNING: write report: {}", e)),
            }
        }

        CommandResult {
            status: if all_pass { Status::Ok } else { Status::Error },
            message: format!(
                "FAISS verification: {} profiles, {}/{} pass, {} fail",
                profiles.len(),
                overall.total - overall.real_mismatch,
                overall.total,
                overall.real_mismatch),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("base", "Path", false, None,
                "Base vectors (fvec). Omit to auto-discover from dataset.yaml", OptionRole::Input),
            opt("query", "Path", false, None,
                "Query vectors (fvec). Omit to auto-discover from dataset.yaml", OptionRole::Input),
            opt("ground-truth", "Path", false, None,
                "Ground truth indices (ivec). Omit to auto-discover from dataset.yaml", OptionRole::Input),
            opt("indices", "Path", false, None,
                "Alias for ground-truth (pipeline compatibility)", OptionRole::Input),
            opt("distances", "Path", false, None,
                "Distances file (accepted for compatibility, not used for verification)", OptionRole::Input),
            opt("metric", "string", false, Some("L2"),
                "Distance metric: L2, IP, DOT_PRODUCT, COSINE", OptionRole::Config),
            opt("normalized", "bool", false, Some("false"),
                "Whether vectors are normalized (accepted for compatibility)", OptionRole::Config),
            opt("neighbors", "int", false, None,
                "Accepted for compatibility (k is read from GT file)", OptionRole::Config),
            opt("sample", "int", false, Some("0"),
                "Number of queries to sample (0 = all)", OptionRole::Config),
            opt("seed", "int", false, Some("42"),
                "Random seed for sampling", OptionRole::Config),
            opt("at-k", "int", false, Some("0"),
                "Compare only the top at-k neighbors (0 = full k)", OptionRole::Config),
            opt("output", "Path", false, None,
                "Output JSON report (pipeline compatibility)", OptionRole::Output),
            opt("limit", "int", false, Some("10"),
                "Max mismatch details to display", OptionRole::Config),
        ]
    }
}

impl VerifyKnnFaissOp {
    /// Verify with explicit base/query/gt paths.
    fn verify_explicit(
        &self,
        options: &Options,
        ctx: &mut StreamContext,
        faiss_mt: faiss::MetricType,
        metric_str: &str,
        sample: usize,
        at_k: usize,
        limit: usize,
        start: Instant,
    ) -> CommandResult {
        // Standalone-friendly: input paths come from the active
        // profile's facets in dataset.yaml.
        let base_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "base", "base_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let query_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "query", "query_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let base_path = resolve_path(&base_str, &ctx.workspace);
        // Either `--ground-truth` or `--indices` is accepted; both map
        // to the `neighbor_indices` facet on standalone fallback.
        let gt_str = match options.get("ground-truth").or_else(|| options.get("indices")) {
            Some(s) => s.to_string(),
            None => match crate::pipeline::dataset_lookup::resolve_path_option(
                ctx, options, "ground-truth", "neighbor_indices",
            ) {
                Ok(s) => s,
                Err(_) => return error_result(
                    "required option 'ground-truth' (or 'indices') not set", start),
            },
        };
        let query_path = resolve_path(&query_str, &ctx.workspace);
        let gt_path = resolve_path(&gt_str, &ctx.workspace);

        ctx.ui.log(&format!(
            "verify knn-faiss: KNN ground truth via FAISS IndexFlat oracle \
             (single-threaded MKL, batch-capped); metric={}, at_k={}, sample={}, limit={}",
            metric_str, at_k, sample, limit,
        ));

        let output_path = options.get("output")
            .map(|s| resolve_path(s, &ctx.workspace));

        match verify_profile(&base_path, &query_path, &gt_path, faiss_mt, metric_str, sample, at_k, limit, ctx) {
            Ok(summary) => {
                let pass = summary.real_mismatch == 0;
                let mut produced = vec![];

                // Write JSON report if output path was provided
                if let Some(ref op) = output_path {
                    let report = serde_json::json!({
                        "queries": summary.total,
                        "exact_match": summary.exact_match,
                        "set_match": summary.set_match,
                        "boundary_mismatch": summary.boundary_mismatch,
                        "real_mismatch": summary.real_mismatch,
                        "pass": pass,
                        "engine": "FAISS",
                        "metric": metric_str,
                    });
                    if let Some(parent) = op.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(op, serde_json::to_string_pretty(&report).unwrap_or_default());
                    produced.push(op.clone());
                }

                CommandResult {
                    status: if pass { Status::Ok } else { Status::Error },
                    message: format!(
                        "FAISS verification: {}/{} pass, {} exact, {} set, {} boundary, {} fail",
                        summary.total - summary.real_mismatch, summary.total,
                        summary.exact_match, summary.set_match,
                        summary.boundary_mismatch, summary.real_mismatch),
                    produced,
                    elapsed: start.elapsed(),
                }
            }
            Err(e) => error_result(e, start),
        }
    }
}

/// Verify one profile's KNN ground truth against FAISS recomputation.
fn verify_profile(
    base_path: &Path,
    query_path: &Path,
    gt_path: &Path,
    faiss_mt: faiss::MetricType,
    metric_str: &str,
    sample: usize,
    at_k: usize,
    limit: usize,
    ctx: &mut StreamContext,
) -> Result<VerifySummary, String> {
    // Open readers.
    let base_reader = MmapVectorReader::<f32>::open_fvec(base_path)
        .map_err(|e| format!("open base {}: {}", base_path.display(), e))?;
    let query_reader = MmapVectorReader::<f32>::open_fvec(query_path)
        .map_err(|e| format!("open query {}: {}", query_path.display(), e))?;
    let gt_reader = MmapVectorReader::<i32>::open_ivec(gt_path)
        .map_err(|e| format!("open GT {}: {}", gt_path.display(), e))?;

    let base_count = VectorReader::<f32>::count(&base_reader);
    let query_count = VectorReader::<f32>::count(&query_reader);
    let dim = VectorReader::<f32>::dim(&base_reader);
    let gt_count = VectorReader::<i32>::count(&gt_reader);
    let k = VectorReader::<i32>::dim(&gt_reader);

    if VectorReader::<f32>::dim(&query_reader) != dim {
        return Err(format!("dim mismatch: base={} query={}",
            dim, VectorReader::<f32>::dim(&query_reader)));
    }
    // Use the smaller of GT and query counts — GT may have more entries
    // if per-label query extraction happened after KNN was computed.
    let verify_count = query_count.min(gt_count);
    if gt_count != query_count {
        ctx.ui.log(&format!("  note: GT has {} entries but query file has {} vectors — verifying {}",
            gt_count, query_count, verify_count));
    }

    // at_k truncates comparison to the first at_k neighbors.
    // FAISS still searches for the full k so boundary neighbors are
    // computed, but only the top at_k are compared against GT.
    let compare_k = if at_k > 0 && at_k < k { at_k } else { k };

    if compare_k < k {
        ctx.ui.log(&format!("  base: {} vectors, query: {} vectors, dim={}, k={}, at_k={}, metric={}",
            base_count, query_count, dim, k, compare_k, metric_str));
    } else {
        ctx.ui.log(&format!("  base: {} vectors, query: {} vectors, dim={}, k={}, metric={}",
            base_count, query_count, dim, k, metric_str));
    }

    // Cap sample size to stay within FAISS safe batch zone.
    // faiss-sys has a BLAS ABI bug where nq * dim > 65536 produces
    // corrupt results. Cap the verify sample to stay well within this.
    let faiss_safe_max = (65536 / dim).max(1);
    let effective_sample = if sample > 0 {
        sample.min(faiss_safe_max)
    } else {
        verify_count.min(faiss_safe_max)
    };
    if effective_sample < verify_count {
        ctx.ui.log(&format!("  FAISS safe mode: sampling {} of {} queries (dim={}, max safe batch={})",
            effective_sample, verify_count, dim, faiss_safe_max));
    }

    // Determine which queries to verify.
    let query_indices: Vec<usize> = if effective_sample < verify_count {
        use std::collections::BTreeSet;
        let mut rng = simple_rng(42);
        let mut selected = BTreeSet::new();
        while selected.len() < effective_sample {
            selected.insert(rng.next() as usize % verify_count);
        }
        selected.into_iter().collect()
    } else {
        (0..verify_count).collect()
    };

    let n_verify = query_indices.len();
    ctx.ui.log(&format!("  verifying {} of {} queries...", n_verify, query_count));

    // FAISS uses OpenMP for parallelism — thread count is controlled
    // by OMP_NUM_THREADS (defaults to all cores).

    // Load base vectors into contiguous buffer for FAISS.
    let t0 = Instant::now();
    ctx.ui.log(&format!("  loading {} base vectors ({:.1} MiB)...",
        base_count, (base_count * dim * 4) as f64 / 1_048_576.0));
    let mut base_data: Vec<f32> = Vec::with_capacity(base_count * dim);
    for i in 0..base_count {
        base_data.extend_from_slice(base_reader.get_slice(i));
    }
    ctx.ui.log(&format!("  base vectors loaded in {:.1}s", t0.elapsed().as_secs_f64()));

    // Build FAISS index.
    let t1 = Instant::now();
    ctx.ui.log("  building FAISS FlatIndex...");
    let mut index = faiss::index::flat::FlatIndex::new(dim as u32, faiss_mt)
        .map_err(|e| format!("FAISS index creation: {}", e))?;
    index.add(&base_data)
        .map_err(|e| format!("FAISS index.add: {}", e))?;
    ctx.ui.log(&format!("  FAISS index ready: ntotal={}, built in {:.1}s",
        index.ntotal(), t1.elapsed().as_secs_f64()));

    // Load query vectors for batch search.
    let t2 = Instant::now();
    ctx.ui.log(&format!("  loading {} query vectors...", n_verify));
    let mut query_data: Vec<f32> = Vec::with_capacity(n_verify * dim);
    for &qi in &query_indices {
        query_data.extend_from_slice(query_reader.get_slice(qi));
    }

    // Chunked FAISS search — faiss-sys v0.13 corrupts results when
    // nq * dim > ~65536 in a single call.
    let max_batch = (65536 / dim).max(1);
    let evals = n_verify as f64 * base_count as f64;
    ctx.ui.log(&format!("  FAISS search: {} queries × {} base, k={}...",
        n_verify, base_count, k));

    let mut all_labels: Vec<faiss::index::Idx> = Vec::with_capacity(n_verify * k);
    for batch_start in (0..n_verify).step_by(max_batch) {
        let batch_end = (batch_start + max_batch).min(n_verify);
        let batch_data = &query_data[batch_start * dim..batch_end * dim];
        let result = index.search(batch_data, k)
            .map_err(|e| format!("FAISS search: {}", e))?;
        all_labels.extend_from_slice(&result.labels);
    }

    let search_secs = t2.elapsed().as_secs_f64();
    ctx.ui.log(&format!("  search complete in {:.2}s ({:.1}B dist/s)",
        search_secs, evals / search_secs / 1e9));

    // Compare results against ground truth.
    let mut summary = VerifySummary::default();
    let mut mismatch_details: Vec<(usize, usize)> = Vec::new();

    for (ri, &qi) in query_indices.iter().enumerate() {
        let offset = ri * k;
        let faiss_indices: Vec<i32> = all_labels[offset..offset + compare_k].iter()
            .map(|l| l.get().map(|v| v as i32).unwrap_or(-1))
            .collect();

        let gt_slice = gt_reader.get_slice(qi);
        let gt_indices = &gt_slice[..compare_k.min(gt_slice.len())];

        let cmp = knn_compare::compare_query_ordinals(&faiss_indices, gt_indices);
        summary.total += 1;
        match cmp {
            knn_compare::QueryResult::ExactMatch => summary.exact_match += 1,
            knn_compare::QueryResult::SetMatch => summary.set_match += 1,
            knn_compare::QueryResult::BoundaryMismatch(n) => {
                summary.boundary_mismatch += 1;
                mismatch_details.push((qi, n));
            }
            knn_compare::QueryResult::RealMismatch(n) => {
                summary.real_mismatch += 1;
                mismatch_details.push((qi, n));
            }
        }
    }

    // Show mismatch details.
    if !mismatch_details.is_empty() {
        let show = mismatch_details.len().min(limit);
        ctx.ui.log(&format!("  mismatches ({} total, showing {}):",
            mismatch_details.len(), show));
        for &(qi, diff) in mismatch_details.iter().take(show) {
            let kind = if diff <= knn_compare::BOUNDARY_THRESHOLD { "boundary" } else { "REAL" };
            ctx.ui.log(&format!("    query {:>6}: {} neighbors differ ({})", qi, diff, kind));
        }
        if mismatch_details.len() > show {
            ctx.ui.log(&format!("    ... ({} more)", mismatch_details.len() - show));
        }
    }

    Ok(summary)
}

/// Minimal deterministic PRNG for query sampling.
struct SimpleRng(u64);

fn simple_rng(seed: u64) -> SimpleRng {
    SimpleRng(seed)
}

impl SimpleRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Consolidated FAISS verifier — single pass across all sized profiles
// ═══════════════════════════════════════════════════════════════════════

pub struct VerifyKnnFaissConsolidatedOp;

pub fn consolidated_factory() -> Box<dyn CommandOp> {
    Box::new(VerifyKnnFaissConsolidatedOp)
}

impl CommandOp for VerifyKnnFaissConsolidatedOp {
    fn command_path(&self) -> &str {
        "verify knn-consolidated"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "FAISS-based consolidated KNN verification across all profiles".into(),
            body: format!(
                "Verifies KNN ground truth for all sized profiles using FAISS.\n\n\
                 Loads base and query vectors once, sharing them across all sized \
                 profiles. Profiles are sorted by base_count ascending — the FAISS \
                 index is rebuilt for each profile size using a prefix of the shared \
                 base data. Partition profiles are verified independently in parallel \
                 with their own base vectors. The sample and seed options control \
                 query sampling. The metric and normalized options control distance \
                 computation. Results are written to the output JSON report.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("base", "Path", true, None, "Base vectors file (shared across sized profiles)", OptionRole::Input),
            opt("query", "Path", true, None, "Query vectors file", OptionRole::Input),
            opt("metric", "String", false, Some("L2"), "Distance metric", OptionRole::Config),
            opt("normalized", "bool", false, Some("false"), "Vectors are L2-normalized", OptionRole::Config),
            opt("sample", "int", false, Some("100"), "Number of queries to sample per profile", OptionRole::Config),
            opt("seed", "int", false, Some("42"), "Random seed for sampling", OptionRole::Config),
            opt("output", "Path", true, None, "Output JSON report", OptionRole::Output),
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "FAISS index + base vectors".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // Up-front: confirm the local dataset has the minimum facets
        // this verify kind requires (see pipeline::dataset_lookup).
        if let Err(e) = crate::pipeline::dataset_lookup::validate_and_log(
            ctx, options, crate::pipeline::dataset_lookup::VerifyKind::KnnFaissConsolidated,
        ) {
            return error_result(e, start);
        }

        // Standalone-friendly: input paths come from the active
        // profile's facets in dataset.yaml.
        let base_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "base", "base_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let query_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "query", "query_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let output_str = match options.require("output") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };

        let metric_str = options.get("metric").unwrap_or("L2");
        let faiss_mt = match faiss_metric(metric_str) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };

        let sample_count: usize = options.get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let seed: u64 = options.get("seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);

        let base_path = resolve_path(&base_str, &ctx.workspace);
        let query_path = resolve_path(&query_str, &ctx.workspace);
        let output_path = resolve_path(&output_str, &ctx.workspace);

        // Load dataset config for profile discovery
        let ds_path = ctx.workspace.join("dataset.yaml");
        let config = match vectordata::dataset::DatasetConfig::load_and_resolve(&ds_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("load dataset.yaml: {}", e), start),
        };

        // Collect sized profiles (shared base/query) sorted by base_count
        let mut sized_profiles: Vec<(String, usize, PathBuf)> = Vec::new();
        // Collect partition profiles (independent base/query)
        let mut partition_profiles: Vec<(String, PathBuf, PathBuf, PathBuf)> = Vec::new();

        for (name, profile) in &config.profiles.profiles {
            let gt_path = profile.view("neighbor_indices")
                .map(|v| ctx.workspace.join(v.path()));
            let gt = match gt_path {
                Some(p) if p.exists() => p,
                _ => continue,
            };

            if profile.partition {
                let bp = profile.view("base_vectors")
                    .map(|v| ctx.workspace.join(v.path()));
                let qp = profile.view("query_vectors")
                    .map(|v| ctx.workspace.join(v.path()));
                if let (Some(b), Some(q)) = (bp, qp) {
                    if b.exists() && q.exists() {
                        partition_profiles.push((name.clone(), b, q, gt));
                    }
                }
            } else {
                let bc = profile.base_count
                    .map(|c| c as usize)
                    .unwrap_or(usize::MAX);
                sized_profiles.push((name.clone(), bc, gt));
            }
        }

        sized_profiles.sort_by_key(|(_, bc, _)| *bc);

        let total_profiles = sized_profiles.len() + partition_profiles.len();
        ctx.ui.log(&format!(
            "verify knn-consolidated (FAISS): {} profiles ({} sized, {} partition), metric={}",
            total_profiles, sized_profiles.len(), partition_profiles.len(), metric_str));

        // Open shared base and query readers
        let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open base: {}", e), start),
        };
        let query_reader = match MmapVectorReader::<f32>::open_fvec(&query_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open query: {}", e), start),
        };

        let full_base_count = VectorReader::<f32>::count(&base_reader);
        let query_count = VectorReader::<f32>::count(&query_reader);
        let dim = VectorReader::<f32>::dim(&base_reader);

        // Load ALL base vectors once
        let t0 = Instant::now();
        ctx.ui.log(&format!("  loading {} base vectors ({:.1} MiB)...",
            full_base_count, (full_base_count * dim * 4) as f64 / 1_048_576.0));
        let mut base_data: Vec<f32> = Vec::with_capacity(full_base_count * dim);
        for i in 0..full_base_count {
            base_data.extend_from_slice(base_reader.get_slice(i));
        }
        ctx.ui.log(&format!("  loaded in {:.1}s", t0.elapsed().as_secs_f64()));

        // Cap sample to FAISS safe batch size
        let faiss_safe_max = (65536 / dim).max(1);
        let effective_sample = if sample_count > 0 {
            sample_count.min(faiss_safe_max)
        } else {
            query_count.min(faiss_safe_max)
        };
        if effective_sample < query_count {
            ctx.ui.log(&format!("  FAISS safe mode: sampling {} of {} queries (max safe={})",
                effective_sample, query_count, faiss_safe_max));
        }

        // Select sample queries (shared across all profiles)
        let sample_indices: Vec<usize> = if effective_sample < query_count {
            let mut rng = simple_rng(seed);
            let mut selected = std::collections::BTreeSet::new();
            while selected.len() < effective_sample {
                selected.insert(rng.next() as usize % query_count);
            }
            selected.into_iter().collect()
        } else {
            (0..query_count).collect()
        };

        // Load sample query vectors
        let n_sample = sample_indices.len();
        let mut query_data: Vec<f32> = Vec::with_capacity(n_sample * dim);
        for &qi in &sample_indices {
            query_data.extend_from_slice(query_reader.get_slice(qi));
        }

        ctx.ui.log(&format!("  {} sample queries selected", n_sample));

        let mut overall = VerifySummary::default();
        let mut all_pass = true;
        let mut profile_results: Vec<serde_json::Value> = Vec::new();

        // ── Sized profiles: share base data, rebuild index per size ────
        let mut _prev_base_count = 0usize;
        for (name, bc, gt_path) in &sized_profiles {
            let effective_bc = if *bc == usize::MAX { full_base_count } else { (*bc).min(full_base_count) };

            ctx.ui.log(&format!("\n  profile '{}' (base_count={})", name,
                if *bc == usize::MAX { "full".to_string() } else { format!("{}", effective_bc) }));

            // Build FAISS index for this profile's base_count
            // Only rebuild if base_count changed from previous profile
            let t1 = Instant::now();
            let base_slice = &base_data[..effective_bc * dim];
            let mut index = match faiss::index::flat::FlatIndex::new(dim as u32, faiss_mt) {
                Ok(idx) => idx,
                Err(e) => {
                    ctx.ui.log(&format!("    ERROR: FAISS index: {}", e));
                    all_pass = false;
                    continue;
                }
            };
            if let Err(e) = index.add(base_slice) {
                ctx.ui.log(&format!("    ERROR: FAISS add: {}", e));
                all_pass = false;
                continue;
            }

            // Open GT
            let gt_reader = match MmapVectorReader::<i32>::open_ivec(gt_path) {
                Ok(r) => r,
                Err(e) => {
                    ctx.ui.log(&format!("    ERROR: open GT: {}", e));
                    all_pass = false;
                    continue;
                }
            };
            let k = VectorReader::<i32>::dim(&gt_reader);

            // Chunked FAISS search — cap batch size to avoid faiss-sys corruption
            let chunk_size = (65536 / dim).max(1).min(n_sample);
            let pb = ctx.ui.bar(n_sample as u64, &format!("verify {}", name));

            let mut summary = VerifySummary::default();
            for chunk_start in (0..n_sample).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(n_sample);
                let chunk_data = &query_data[chunk_start * dim..chunk_end * dim];

                let result = match index.search(chunk_data, k) {
                    Ok(r) => r,
                    Err(e) => {
                        ctx.ui.log(&format!("    ERROR: search: {}", e));
                        all_pass = false;
                        break;
                    }
                };

                for ri in 0..(chunk_end - chunk_start) {
                    let qi = sample_indices[chunk_start + ri];
                    let offset = ri * k;
                    let faiss_indices: Vec<i32> = result.labels[offset..offset + k].iter()
                        .map(|l| l.get().map(|v| v as i32).unwrap_or(-1))
                        .collect();

                    if qi < VectorReader::<i32>::count(&gt_reader) {
                        let gt_indices = gt_reader.get_slice(qi);
                        let cmp = knn_compare::compare_query_ordinals(&faiss_indices, gt_indices);
                        summary.total += 1;
                        match cmp {
                            knn_compare::QueryResult::ExactMatch => summary.exact_match += 1,
                            knn_compare::QueryResult::SetMatch => summary.set_match += 1,
                            knn_compare::QueryResult::BoundaryMismatch(_) => summary.boundary_mismatch += 1,
                            knn_compare::QueryResult::RealMismatch(_) => summary.real_mismatch += 1,
                        }
                    }
                }
                pb.set_position(chunk_end as u64);
            }
            pb.finish();

            let pass = summary.real_mismatch == 0;
            if !pass { all_pass = false; }
            let build_ms = t1.elapsed().as_millis();
            ctx.ui.log(&format!("    {}: {}/{} queries — {} exact, {} set, {} boundary, {} fail ({} ms)",
                if pass { "PASS" } else { "FAIL" },
                summary.total, summary.total,
                summary.exact_match, summary.set_match,
                summary.boundary_mismatch, summary.real_mismatch, build_ms));

            profile_results.push(serde_json::json!({
                "profile": name,
                "base_count": effective_bc,
                "queries": summary.total,
                "exact": summary.exact_match,
                "set_match": summary.set_match,
                "boundary": summary.boundary_mismatch,
                "fail": summary.real_mismatch,
                "pass": pass,
            }));

            overall.total += summary.total;
            overall.exact_match += summary.exact_match;
            overall.set_match += summary.set_match;
            overall.boundary_mismatch += summary.boundary_mismatch;
            overall.real_mismatch += summary.real_mismatch;
            _prev_base_count = effective_bc;
        }

        // ── Partition profiles: independent base/query ────────────────
        // Run all partition verifications in parallel — each has its own
        // base vectors, query vectors, and GT. No shared state.
        if !partition_profiles.is_empty() {
            ctx.ui.log(&format!("\n  verifying {} partition profiles in parallel...",
                partition_profiles.len()));
            let partition_pb = ctx.ui.bar(partition_profiles.len() as u64, "partitions");

            let results: Vec<(String, Result<VerifySummary, String>)> = std::thread::scope(|s| {
                let handles: Vec<_> = partition_profiles.iter().map(|(name, bp, qp, gp)| {
                    let name = name.clone();
                    let bp = bp.clone();
                    let qp = qp.clone();
                    let gp = gp.clone();
                    s.spawn(move || {
                        let base_reader = match MmapVectorReader::<f32>::open_fvec(&bp) {
                            Ok(r) => r, Err(e) => return (name, Err(format!("open base: {}", e))),
                        };
                        let query_reader = match MmapVectorReader::<f32>::open_fvec(&qp) {
                            Ok(r) => r, Err(e) => return (name, Err(format!("open query: {}", e))),
                        };
                        let gt_reader = match MmapVectorReader::<i32>::open_ivec(&gp) {
                            Ok(r) => r, Err(e) => return (name, Err(format!("open GT: {}", e))),
                        };

                        let bc = VectorReader::<f32>::count(&base_reader);
                        let qc = VectorReader::<f32>::count(&query_reader);
                        let d = VectorReader::<f32>::dim(&base_reader);
                        let gc = VectorReader::<i32>::count(&gt_reader);
                        let k = VectorReader::<i32>::dim(&gt_reader);
                        let faiss_safe = (65536 / d).max(1);
                        let vc = qc.min(gc).min(faiss_safe);

                        // Load base
                        let mut base_data = Vec::with_capacity(bc * d);
                        for i in 0..bc { base_data.extend_from_slice(base_reader.get_slice(i)); }

                        // Build index
                        let mut index = match faiss::index::flat::FlatIndex::new(d as u32, faiss_mt) {
                            Ok(idx) => idx, Err(e) => return (name, Err(format!("FAISS: {}", e))),
                        };
                        if let Err(e) = index.add(&base_data) {
                            return (name, Err(format!("FAISS add: {}", e)));
                        }
                        drop(base_data);

                        // Load queries + chunked search (faiss-sys batch size limit)
                        let mut qdata = Vec::with_capacity(vc * d);
                        for i in 0..vc { qdata.extend_from_slice(query_reader.get_slice(i)); }

                        let max_batch = (65536 / d).max(1);
                        let mut all_labels: Vec<faiss::index::Idx> = Vec::with_capacity(vc * k);
                        for bs in (0..vc).step_by(max_batch) {
                            let be = (bs + max_batch).min(vc);
                            let bd = &qdata[bs * d..be * d];
                            match index.search(bd, k) {
                                Ok(r) => all_labels.extend_from_slice(&r.labels),
                                Err(e) => return (name, Err(format!("search: {}", e))),
                            }
                        }

                        // Compare
                        let mut summary = VerifySummary::default();
                        for qi in 0..vc {
                            let offset = qi * k;
                            let fi: Vec<i32> = all_labels[offset..offset + k].iter()
                                .map(|l| l.get().map(|v| v as i32).unwrap_or(-1)).collect();
                            let gt = gt_reader.get_slice(qi);
                            summary.total += 1;
                            match knn_compare::compare_query_ordinals(&fi, gt) {
                                knn_compare::QueryResult::ExactMatch => summary.exact_match += 1,
                                knn_compare::QueryResult::SetMatch => summary.set_match += 1,
                                knn_compare::QueryResult::BoundaryMismatch(_) => summary.boundary_mismatch += 1,
                                knn_compare::QueryResult::RealMismatch(_) => summary.real_mismatch += 1,
                            }
                        }
                        (name, Ok(summary))
                    })
                }).collect();

                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            for (name, result) in results {
                partition_pb.inc(1);
                match result {
                    Ok(summary) => {
                        let pass = summary.real_mismatch == 0;
                        if !pass { all_pass = false; }
                        ctx.ui.log(&format!("    {} {}: {}/{} — {} exact, {} set, {} boundary, {} fail",
                            name,
                            if pass { "PASS" } else { "FAIL" },
                            summary.total, summary.total,
                            summary.exact_match, summary.set_match,
                            summary.boundary_mismatch, summary.real_mismatch));

                        profile_results.push(serde_json::json!({
                            "profile": name,
                            "partition": true,
                            "queries": summary.total,
                            "exact": summary.exact_match,
                            "set_match": summary.set_match,
                            "boundary": summary.boundary_mismatch,
                            "fail": summary.real_mismatch,
                            "pass": pass,
                        }));

                        overall.total += summary.total;
                        overall.exact_match += summary.exact_match;
                        overall.set_match += summary.set_match;
                        overall.boundary_mismatch += summary.boundary_mismatch;
                        overall.real_mismatch += summary.real_mismatch;
                    }
                    Err(e) => {
                        ctx.ui.log(&format!("    {} ERROR: {}", name, e));
                        all_pass = false;
                    }
                }
            }
            partition_pb.finish();
        }

        // Write JSON report
        let report = serde_json::json!({
            "profiles": profile_results,
            "total_queries": overall.total,
            "exact_match": overall.exact_match,
            "set_match": overall.set_match,
            "boundary_mismatch": overall.boundary_mismatch,
            "real_mismatch": overall.real_mismatch,
            "pass": all_pass,
            "engine": "FAISS",
            "metric": metric_str,
        });
        if let Some(parent) = output_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(&output_path, serde_json::to_string_pretty(&report).unwrap_or_default()) {
            ctx.ui.log(&format!("  WARNING: write report: {}", e));
        }

        ctx.ui.log(&format!(
            "\noverall: {} profiles, {}/{} pass, {} boundary, {} fail, engine=FAISS",
            total_profiles,
            overall.total - overall.real_mismatch, overall.total,
            overall.boundary_mismatch, overall.real_mismatch));

        CommandResult {
            status: if all_pass { Status::Ok } else { Status::Error },
            message: format!(
                "verified {} profiles: {}/{} pass, {} fail (FAISS, k=sampled, metric={})",
                total_profiles,
                overall.total - overall.real_mismatch,
                overall.total,
                overall.real_mismatch,
                metric_str),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }
}

fn resolve_path(s: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(s);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: msg.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        role,
    }
}
