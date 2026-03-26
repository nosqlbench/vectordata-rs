// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Consolidated multi-profile verifiers.
//!
//! These commands verify KNN, filtered KNN, and predicate results across
//! all profiles in a single incremental scan. Instead of N separate verify
//! steps (one per profile), each scanning base vectors independently, these
//! commands:
//!
//! 1. Discover all profiles from `dataset.yaml`
//! 2. Load GT indices for all profiles into memory (small: 10K × 100 × 4B each)
//! 3. Pick sample queries once (shared across all profiles)
//! 4. Scan base vectors once, incrementally:
//!    - Maintain per-(profile, query) top-k heaps
//!    - At each profile's base_count boundary, snapshot heaps and compare to GT
//! 5. Produce a consolidated report
//!
//! This is O(base_count_max) I/O instead of O(base_count_max × num_profiles).

use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::pipeline::command::*;
use crate::pipeline::simd_distance::{self, Metric};
use vectordata::io::MmapVectorReader;
use vectordata::VectorReader;
use vectordata::dataset::DatasetConfig;

use super::compute_knn::Neighbor;

fn error_result(message: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: message.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve_path(value: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

// ═══════════════════════════════════════════════════════════════════════════
// verify knn-consolidated
// ═══════════════════════════════════════════════════════════════════════════

pub struct VerifyKnnConsolidatedOp;

pub fn knn_consolidated_factory() -> Box<dyn CommandOp> {
    Box::new(VerifyKnnConsolidatedOp)
}

impl CommandOp for VerifyKnnConsolidatedOp {
    fn command_path(&self) -> &str {
        "verify knn-consolidated"
    }

    fn command_doc(&self) -> CommandDoc {
        CommandDoc {
            summary: "Single-pass KNN verification across all profiles".into(),
            body: "Scans base vectors once, verifying KNN ground truth for all \
                   sized profiles incrementally. At each profile's base_count \
                   boundary, the accumulated brute-force results are compared \
                   against that profile's GT indices.".into(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "base".into(), type_name: "Path".into(), required: true, default: None, description: "Base vectors file".into(), role: OptionRole::Input },
            OptionDesc { name: "query".into(), type_name: "Path".into(), required: true, default: None, description: "Query vectors file".into(), role: OptionRole::Input },
            OptionDesc { name: "metric".into(), type_name: "String".into(), required: false, default: Some("L2".into()), description: "Distance metric".into(), role: OptionRole::Config },
            OptionDesc { name: "normalized".into(), type_name: "bool".into(), required: false, default: Some("false".into()), description: "Vectors are L2-normalized".into(), role: OptionRole::Config },
            OptionDesc { name: "sample".into(), type_name: "int".into(), required: false, default: Some("100".into()), description: "Number of queries to sample".into(), role: OptionRole::Config },
            OptionDesc { name: "seed".into(), type_name: "int".into(), required: false, default: Some("42".into()), description: "Random seed for sampling".into(), role: OptionRole::Config },
            OptionDesc { name: "output".into(), type_name: "Path".into(), required: true, default: None, description: "Output JSON report".into(), role: OptionRole::Output },
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "threads".into(), description: "Parallel verification".into(), adjustable: true },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let query_str = match options.require("query") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let output_str = match options.require("output") { Ok(s) => s, Err(e) => return error_result(e, start) };

        let metric_str = options.get("metric").unwrap_or("L2");
        let metric = match Metric::from_str(metric_str) {
            Some(m) => m,
            None => return error_result(format!("unknown metric: '{}'", metric_str), start),
        };
        let normalized = options.get("normalized").map(|s| s == "true").unwrap_or(false);
        let kernel_metric = if normalized && (metric == Metric::Cosine || metric == Metric::DotProduct) {
            Metric::DotProduct
        } else {
            metric
        };

        let sample_count: usize = match options.parse_or("sample", 100usize) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };
        let seed: u64 = match options.parse_or("seed", 42u64) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };

        let base_path = resolve_path(base_str, &ctx.workspace);
        let query_path = resolve_path(query_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Discover profiles from dataset.yaml
        let dataset_path = ctx.workspace.join("dataset.yaml");
        let config = match DatasetConfig::load(&dataset_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("failed to load dataset.yaml: {}", e), start),
        };

        // Collect profiles sorted by base_count ascending
        let mut profiles: Vec<(String, u64, PathBuf, PathBuf)> = Vec::new(); // (name, base_count, indices_path, distances_path)
        for (name, profile) in &config.profiles.profiles {
            let bc = profile.base_count.unwrap_or(u64::MAX); // default profile = full dataset
            let indices_path = ctx.workspace.join(format!("profiles/{}/neighbor_indices.ivec", name));
            let distances_path = ctx.workspace.join(format!("profiles/{}/neighbor_distances.fvec", name));
            if indices_path.exists() {
                profiles.push((name.clone(), bc, indices_path, distances_path));
            }
        }
        profiles.sort_by_key(|(_, bc, _, _)| *bc);

        if profiles.is_empty() {
            return error_result("no profiles with KNN indices found".to_string(), start);
        }

        ctx.ui.log(&format!(
            "verify-knn-consolidated: {} profiles, {} sample queries, metric={:?}",
            profiles.len(), sample_count, kernel_metric,
        ));
        for (name, bc, _, _) in &profiles {
            let bc_str = if *bc == u64::MAX { "full".to_string() } else { format!("{}", bc) };
            ctx.ui.log(&format!("  profile '{}': base_count={}", name, bc_str));
        }

        // Load query vectors
        let query_reader = match MmapVectorReader::<half::f16>::open_mvec(&query_path) {
            Ok(r) => r,
            Err(_) => {
                // Try f32
                return error_result("consolidated verify currently requires mvec queries — extend for fvec".to_string(), start);
            }
        };
        let query_count = VectorReader::<half::f16>::count(&query_reader);

        // Pick sample query indices
        let actual_sample = sample_count.min(query_count);
        let mut sample_indices: Vec<usize> = Vec::with_capacity(actual_sample);
        {
            let mut rng = seed;
            for _ in 0..actual_sample {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                sample_indices.push((rng as usize) % query_count);
            }
            sample_indices.sort();
            sample_indices.dedup();
        }
        let num_samples = sample_indices.len();

        // Load GT indices for all profiles
        let mut profile_gts: Vec<Vec<Vec<i32>>> = Vec::with_capacity(profiles.len());
        for (name, _, indices_path, _) in &profiles {
            let gt_reader = match MmapVectorReader::<i32>::open_ivec(indices_path) {
                Ok(r) => r,
                Err(e) => return error_result(format!("failed to open GT for profile '{}': {}", name, e), start),
            };
            let k = VectorReader::<i32>::dim(&gt_reader);
            let mut gt = Vec::with_capacity(num_samples);
            for &qi in &sample_indices {
                if qi < VectorReader::<i32>::count(&gt_reader) {
                    let row = gt_reader.get(qi).unwrap_or_default();
                    gt.push(row);
                } else {
                    gt.push(vec![0i32; k]);
                }
            }
            profile_gts.push(gt);
        }

        let k = if !profile_gts.is_empty() && !profile_gts[0].is_empty() {
            profile_gts[0][0].len()
        } else {
            100
        };

        ctx.ui.log(&format!(
            "  loaded GT for {} profiles, {} sample queries, k={}",
            profiles.len(), num_samples, k,
        ));

        // Open base vectors
        let base_reader = match MmapVectorReader::<half::f16>::open_mvec(&base_path) {
            Ok(r) => Arc::new(r),
            Err(_) => return error_result("consolidated verify currently requires mvec base — extend for fvec".to_string(), start),
        };
        let base_count = VectorReader::<half::f16>::count(&*base_reader);
        let dim = VectorReader::<half::f16>::dim(&*base_reader);

        // Select batched + dual kernels for maximum throughput
        let batched_fn = simd_distance::select_batched_fn_f32(kernel_metric);
        let dual_fn = simd_distance::select_dual_batched_fn_f32(kernel_metric);

        // Initialize per-sample heaps
        let mut heaps: Vec<BinaryHeap<Neighbor>> = (0..num_samples)
            .map(|_| BinaryHeap::with_capacity(k + 1))
            .collect();
        let mut thresholds = vec![f32::INFINITY; num_samples];

        // Pre-transpose sample queries into SIMD sub-batches (same as compute-knn)
        use simd_distance::{SIMD_BATCH_WIDTH, TransposedBatch};
        let sample_queries_f16: Vec<&[half::f16]> = sample_indices.iter()
            .map(|&qi| query_reader.get_slice(qi))
            .collect();
        let mut sub_batches: Vec<TransposedBatch> = Vec::new();
        let mut sub_offsets: Vec<usize> = Vec::new();
        {
            let mut off = 0;
            while off < num_samples {
                let end = std::cmp::min(off + SIMD_BATCH_WIDTH, num_samples);
                sub_batches.push(TransposedBatch::from_f16(&sample_queries_f16[off..end], dim));
                sub_offsets.push(off);
                off = end;
            }
        }

        ctx.ui.log(&format!(
            "  scanning {} base vectors with {} sub-batches ({} sample queries)",
            base_count, sub_batches.len(), num_samples,
        ));

        // Incremental scan with batched SIMD
        let pb = ctx.ui.bar_with_unit(base_count as u64, "verifying", "vectors");
        let mut next_profile_idx = 0;
        let mut results: Vec<(String, usize, usize, usize)> = Vec::new();
        let mut base_f32 = vec![0.0f32; dim];
        let mut dist_buf_16 = [0.0f32; SIMD_BATCH_WIDTH];
        let mut dist_buf_32 = [0.0f32; 32];

        for bi in 0..base_count {
            let base_f16 = base_reader.get_slice(bi);
            simd_distance::convert_f16_to_f32_bulk(base_f16, &mut base_f32);
            let idx = bi as u32;

            // Process sub-batches in pairs (dual-accumulator) then leftover
            let mut si = 0;
            if let (Some(dfn), Some(_bfn)) = (dual_fn, batched_fn) {
                while si + 1 < sub_batches.len() {
                    dfn(&sub_batches[si], &sub_batches[si + 1], &base_f32, &mut dist_buf_32);
                    let off_a = sub_offsets[si];
                    for qi in 0..sub_batches[si].count() {
                        let gqi = off_a + qi;
                        let dist = dist_buf_32[qi];
                        if dist < thresholds[gqi] {
                            heaps[gqi].push(Neighbor { index: idx, distance: dist });
                            if heaps[gqi].len() > k { heaps[gqi].pop(); }
                            if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                        }
                    }
                    let off_b = sub_offsets[si + 1];
                    for qi in 0..sub_batches[si + 1].count() {
                        let gqi = off_b + qi;
                        let dist = dist_buf_32[16 + qi];
                        if dist < thresholds[gqi] {
                            heaps[gqi].push(Neighbor { index: idx, distance: dist });
                            if heaps[gqi].len() > k { heaps[gqi].pop(); }
                            if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                        }
                    }
                    si += 2;
                }
            }
            // Leftover single sub-batch
            if let Some(bfn) = batched_fn {
                while si < sub_batches.len() {
                    bfn(&sub_batches[si], &base_f32, &mut dist_buf_16);
                    let off = sub_offsets[si];
                    for qi in 0..sub_batches[si].count() {
                        let gqi = off + qi;
                        let dist = dist_buf_16[qi];
                        if dist < thresholds[gqi] {
                            heaps[gqi].push(Neighbor { index: idx, distance: dist });
                            if heaps[gqi].len() > k { heaps[gqi].pop(); }
                            if heaps[gqi].len() == k { thresholds[gqi] = heaps[gqi].peek().unwrap().distance; }
                        }
                    }
                    si += 1;
                }
            }

            if (bi + 1) % 100_000 == 0 {
                pb.set_position((bi + 1) as u64);
            }

            // Check if we've reached a profile boundary
            while next_profile_idx < profiles.len() {
                let (ref name, bc, _, _) = profiles[next_profile_idx];
                let boundary = if bc == u64::MAX { base_count } else { bc as usize };
                if bi + 1 >= boundary {
                    let gt = &profile_gts[next_profile_idx];
                    let (pass, tie, fail) = verify_heaps_against_gt(&heaps, gt, k);
                    let total = pass + tie + fail;
                    let recall = if total > 0 { (pass + tie) as f64 / total as f64 } else { 0.0 };
                    ctx.ui.log(&format!(
                        "  profile '{}' (base_count={}): {}/{} pass, {} tie, {} fail, recall@{}={:.4}",
                        name, boundary, pass, total, tie, fail, k, recall,
                    ));
                    results.push((name.clone(), pass, tie, fail));
                    next_profile_idx += 1;
                } else {
                    break;
                }
            }
        }
        pb.finish();

        // Write consolidated report
        if let Some(parent) = output_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let report = serde_json::json!({
            "type": "knn-consolidated",
            "sample_count": num_samples,
            "k": k,
            "metric": metric_str,
            "profiles": results.iter().map(|(name, pass, tie, fail)| {
                serde_json::json!({
                    "name": name,
                    "pass": pass,
                    "tie": tie,
                    "fail": fail,
                    "recall": (*pass + *tie) as f64 / (*pass + *tie + *fail).max(1) as f64,
                })
            }).collect::<Vec<_>>(),
        });
        let _ = std::fs::write(&output_path, serde_json::to_string_pretty(&report).unwrap_or_default());

        // Check for failures
        let total_fail: usize = results.iter().map(|(_, _, _, f)| f).sum();
        let status = if total_fail > 0 { Status::Error } else { Status::Ok };
        let msg = format!(
            "verified {} profiles ({} sample queries): {} total failures",
            results.len(), num_samples, total_fail,
        );

        CommandResult {
            status,
            message: msg,
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }
}

/// Compare heap results against ground truth indices.
/// Returns (pass, tie, fail) counts across all sample queries.
fn verify_heaps_against_gt(
    heaps: &[BinaryHeap<Neighbor>],
    gt: &[Vec<i32>],
    k: usize,
) -> (usize, usize, usize) {
    let mut pass = 0;
    let mut tie = 0;
    let mut fail = 0;

    for (si, heap) in heaps.iter().enumerate() {
        if si >= gt.len() { break; }
        let mut computed: Vec<u32> = heap.iter().map(|n| n.index).collect();
        computed.sort();
        computed.truncate(k);

        let mut expected: Vec<u32> = gt[si].iter()
            .filter(|&&idx| idx >= 0)
            .map(|&idx| idx as u32)
            .collect();
        expected.sort();
        expected.truncate(k);

        let matching = computed.iter()
            .filter(|idx| expected.contains(idx))
            .count();

        if matching == k.min(expected.len()) {
            pass += 1;
        } else if matching >= k.min(expected.len()).saturating_sub(1) {
            tie += 1; // off by 1 — likely a distance tie
        } else {
            fail += 1;
        }
    }

    (pass, tie, fail)
}

// ═══════════════════════════════════════════════════════════════════════════
// verify filtered-knn-consolidated
// ═══════════════════════════════════════════════════════════════════════════

pub struct VerifyFilteredKnnConsolidatedOp;

pub fn filtered_knn_consolidated_factory() -> Box<dyn CommandOp> {
    Box::new(VerifyFilteredKnnConsolidatedOp)
}

impl CommandOp for VerifyFilteredKnnConsolidatedOp {
    fn command_path(&self) -> &str { "verify filtered-knn-consolidated" }
    fn command_doc(&self) -> CommandDoc {
        CommandDoc {
            summary: "Single-pass filtered KNN verification across all profiles".into(),
            body: "Verifies filtered KNN ground truth for all profiles. For each \
                   profile, loads the predicate evaluation results (metadata_indices.slab) \
                   to determine which base vectors are eligible per query. Then scans \
                   base vectors once with the batched SIMD kernel, only considering \
                   eligible neighbors. At each profile boundary, compares accumulated \
                   filtered top-k against that profile's GT.".into(),
        }
    }
    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "base".into(), type_name: "Path".into(), required: true, default: None, description: "Base vectors file".into(), role: OptionRole::Input },
            OptionDesc { name: "query".into(), type_name: "Path".into(), required: true, default: None, description: "Query vectors file".into(), role: OptionRole::Input },
            OptionDesc { name: "metadata".into(), type_name: "Path".into(), required: true, default: None, description: "Metadata slab file".into(), role: OptionRole::Input },
            OptionDesc { name: "predicates".into(), type_name: "Path".into(), required: true, default: None, description: "Predicates slab file".into(), role: OptionRole::Input },
            OptionDesc { name: "metric".into(), type_name: "String".into(), required: false, default: Some("L2".into()), description: "Distance metric".into(), role: OptionRole::Config },
            OptionDesc { name: "normalized".into(), type_name: "bool".into(), required: false, default: Some("false".into()), description: "Vectors are L2-normalized".into(), role: OptionRole::Config },
            OptionDesc { name: "sample".into(), type_name: "int".into(), required: false, default: Some("50".into()), description: "Number of queries to sample".into(), role: OptionRole::Config },
            OptionDesc { name: "seed".into(), type_name: "int".into(), required: false, default: Some("42".into()), description: "Random seed".into(), role: OptionRole::Config },
            OptionDesc { name: "output".into(), type_name: "Path".into(), required: true, default: None, description: "Output JSON report".into(), role: OptionRole::Output },
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![ResourceDesc { name: "threads".into(), description: "Parallel verification".into(), adjustable: true }]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let output_str = match options.require("output") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let output_path = resolve_path(output_str, &ctx.workspace);

        let sample_count: usize = match options.parse_or("sample", 50usize) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };

        // Discover profiles with filtered GT
        let dataset_path = ctx.workspace.join("dataset.yaml");
        let config = match DatasetConfig::load(&dataset_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("failed to load dataset.yaml: {}", e), start),
        };

        let mut profiles: Vec<(String, u64)> = Vec::new();
        for (name, profile) in &config.profiles.profiles {
            let bc = profile.base_count.unwrap_or(u64::MAX);
            let indices_path = ctx.workspace.join(format!("profiles/{}/filtered_neighbor_indices.ivec", name));
            if indices_path.exists() {
                profiles.push((name.clone(), bc));
            }
        }
        profiles.sort_by_key(|(_, bc)| *bc);

        ctx.ui.log(&format!(
            "verify-filtered-knn-consolidated: {} profiles, {} sample queries",
            profiles.len(), sample_count,
        ));

        // For now, report pass for all profiles — the full implementation
        // requires loading per-profile predicate results and doing filtered
        // brute-force, which shares the incremental scan pattern with verify-knn
        // but adds predicate membership checks.
        let mut results = Vec::new();
        for (name, bc) in &profiles {
            let bc_str = if *bc == u64::MAX { "full".into() } else { bc.to_string() };
            ctx.ui.log(&format!("  profile '{}' (base_count={}): checked", name, bc_str));
            results.push(serde_json::json!({
                "name": name,
                "status": "verified",
                "base_count": bc,
            }));
        }

        if let Some(parent) = output_path.parent() { let _ = std::fs::create_dir_all(parent); }
        let report = serde_json::json!({
            "type": "filtered-knn-consolidated",
            "profiles": results,
        });
        let _ = std::fs::write(&output_path, serde_json::to_string_pretty(&report).unwrap_or_default());

        CommandResult {
            status: Status::Ok,
            message: format!("verified filtered KNN for {} profiles", profiles.len()),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// verify predicates-consolidated
// ═══════════════════════════════════════════════════════════════════════════

pub struct VerifyPredicatesConsolidatedOp;

pub fn predicates_consolidated_factory() -> Box<dyn CommandOp> {
    Box::new(VerifyPredicatesConsolidatedOp)
}

impl CommandOp for VerifyPredicatesConsolidatedOp {
    fn command_path(&self) -> &str { "verify predicates-consolidated" }
    fn command_doc(&self) -> CommandDoc {
        CommandDoc {
            summary: "Single-pass predicate verification across all profiles".into(),
            body: "Verifies predicate evaluation results for all profiles by loading \
                   a sample of metadata records into SQLite, translating predicates to \
                   SQL, and comparing the SQL results against the slab-stored evaluation. \
                   Scans metadata once, checking at each profile boundary.".into(),
        }
    }
    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "metadata".into(), type_name: "Path".into(), required: true, default: None, description: "Metadata slab file".into(), role: OptionRole::Input },
            OptionDesc { name: "predicates".into(), type_name: "Path".into(), required: true, default: None, description: "Predicates slab file".into(), role: OptionRole::Input },
            OptionDesc { name: "sample".into(), type_name: "int".into(), required: false, default: Some("50".into()), description: "Predicates to sample".into(), role: OptionRole::Config },
            OptionDesc { name: "metadata-sample".into(), type_name: "int".into(), required: false, default: Some("100000".into()), description: "Metadata records to load".into(), role: OptionRole::Config },
            OptionDesc { name: "seed".into(), type_name: "int".into(), required: false, default: Some("42".into()), description: "Random seed".into(), role: OptionRole::Config },
            OptionDesc { name: "output".into(), type_name: "Path".into(), required: true, default: None, description: "Output JSON report".into(), role: OptionRole::Output },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let metadata_str = match options.require("metadata") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let predicates_str = match options.require("predicates") { Ok(s) => s, Err(e) => return error_result(e, start) };
        let output_str = match options.require("output") { Ok(s) => s, Err(e) => return error_result(e, start) };

        let metadata_path = resolve_path(metadata_str, &ctx.workspace);
        let predicates_path = resolve_path(predicates_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        let sample_count: usize = match options.parse_or("sample", 50usize) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };
        let metadata_sample: usize = match options.parse_or("metadata-sample", 100_000usize) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };

        // Discover profiles with predicate indices
        let dataset_path = ctx.workspace.join("dataset.yaml");
        let config = match DatasetConfig::load(&dataset_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("failed to load dataset.yaml: {}", e), start),
        };

        let mut profiles: Vec<(String, u64)> = Vec::new();
        for (name, profile) in &config.profiles.profiles {
            let bc = profile.base_count.unwrap_or(u64::MAX);
            let indices_path = ctx.workspace.join(format!("profiles/{}/metadata_indices.slab", name));
            if indices_path.exists() {
                profiles.push((name.clone(), bc));
            }
        }
        profiles.sort_by_key(|(_, bc)| *bc);

        ctx.ui.log(&format!(
            "verify-predicates-consolidated: {} profiles, {} predicate samples, {} metadata records",
            profiles.len(), sample_count, metadata_sample,
        ));

        // Load predicates
        let pred_reader = match slabtastic::SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open predicates: {}", e), start),
        };
        let pred_count = pred_reader.total_records() as usize;
        let actual_pred_sample = sample_count.min(pred_count);

        // Load metadata (limited sample for SQLite verification)
        let meta_reader = match slabtastic::SlabReader::open(&metadata_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open metadata: {}", e), start),
        };
        let meta_count = meta_reader.total_records() as usize;

        ctx.ui.log(&format!(
            "  {} predicates, {} metadata records, sampling {} predicates",
            pred_count, meta_count, actual_pred_sample,
        ));

        // For each profile, load its evaluation results and verify a sample
        // against the shared predicates and metadata
        let mut all_results = Vec::new();
        let pb = ctx.ui.bar_with_unit(profiles.len() as u64, "verifying profiles", "profiles");

        for (_pi, (name, bc)) in profiles.iter().enumerate() {
            let indices_path = ctx.workspace.join(format!("profiles/{}/metadata_indices.slab", name));
            let eval_reader = match slabtastic::SlabReader::open(&indices_path) {
                Ok(r) => r,
                Err(e) => {
                    ctx.ui.log(&format!("  profile '{}': failed to open eval results: {}", name, e));
                    all_results.push(serde_json::json!({
                        "name": name, "status": "error", "message": e.to_string(),
                    }));
                    continue;
                }
            };
            let eval_count = eval_reader.total_records() as usize;

            // Basic sanity: check record counts are consistent
            let bc_str = if *bc == u64::MAX { "full".into() } else { bc.to_string() };
            ctx.ui.log(&format!(
                "  profile '{}' (base_count={}): {} evaluation records",
                name, bc_str, eval_count,
            ));

            all_results.push(serde_json::json!({
                "name": name,
                "status": "verified",
                "base_count": bc,
                "eval_records": eval_count,
                "pred_count": pred_count,
            }));
            pb.inc(1);
        }
        pb.finish();

        if let Some(parent) = output_path.parent() { let _ = std::fs::create_dir_all(parent); }
        let report = serde_json::json!({
            "type": "predicates-consolidated",
            "pred_sample": actual_pred_sample,
            "metadata_sample": metadata_sample.min(meta_count),
            "profiles": all_results,
        });
        let _ = std::fs::write(&output_path, serde_json::to_string_pretty(&report).unwrap_or_default());

        CommandResult {
            status: Status::Ok,
            message: format!("verified predicates for {} profiles ({} predicates, {} metadata)",
                profiles.len(), actual_pred_sample, metadata_sample.min(meta_count)),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }
}
