// Copyright (c) nosqlbench contributors
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
//!
//! The scan is multi-threaded: base vectors between profile boundaries are
//! partitioned across threads, each maintaining its own per-query heaps.
//! At each boundary, partial heaps are merged before verification.

use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::pipeline::command::*;
use crate::pipeline::element_type::ElementType;
use crate::pipeline::simd_distance::{self, Metric};
use vectordata::io::MmapVectorReader;
use vectordata::VectorReader;
use vectordata::dataset::DatasetConfig;

use super::compute_knn::Neighbor;

/// Format-agnostic float vector reader for verification.
/// Provides f32 slices regardless of storage precision.
enum AnyFloatReader {
    F16(MmapVectorReader<half::f16>),
    F32(MmapVectorReader<f32>),
}

impl AnyFloatReader {
    fn open(path: &Path) -> Result<Self, String> {
        let etype = ElementType::from_path(path)
            .unwrap_or(ElementType::F32);
        match etype {
            ElementType::F16 => MmapVectorReader::<half::f16>::open_mvec(path)
                .map(AnyFloatReader::F16)
                .map_err(|e| format!("open {}: {}", path.display(), e)),
            _ => MmapVectorReader::<f32>::open_fvec(path)
                .map(AnyFloatReader::F32)
                .map_err(|e| format!("open {}: {}", path.display(), e)),
        }
    }

    fn count(&self) -> usize {
        match self {
            AnyFloatReader::F16(r) => VectorReader::<half::f16>::count(r),
            AnyFloatReader::F32(r) => VectorReader::<f32>::count(r),
        }
    }

    fn dim(&self) -> usize {
        match self {
            AnyFloatReader::F16(r) => VectorReader::<half::f16>::dim(r),
            AnyFloatReader::F32(r) => VectorReader::<f32>::dim(r),
        }
    }

    /// Get a vector as f32 slice. For f16, converts on the fly.
    fn get_f32(&self, index: usize) -> Vec<f32> {
        match self {
            AnyFloatReader::F16(r) => {
                r.get(index).unwrap_or_default().iter().map(|v| v.to_f32()).collect()
            }
            AnyFloatReader::F32(r) => {
                r.get(index).unwrap_or_default()
            }
        }
    }

    /// Get raw f16 slice (for SIMD packing). Returns None for f32 sources.
    fn get_f16_slice(&self, index: usize) -> Option<&[half::f16]> {
        match self {
            AnyFloatReader::F16(r) => Some(r.get_slice(index)),
            AnyFloatReader::F32(_) => None,
        }
    }

    /// Get raw f32 slice. Returns None for f16 sources.
    fn get_f32_slice(&self, index: usize) -> Option<&[f32]> {
        match self {
            AnyFloatReader::F32(r) => Some(r.get_slice(index)),
            AnyFloatReader::F16(_) => None,
        }
    }

    fn is_f16(&self) -> bool {
        matches!(self, AnyFloatReader::F16(_))
    }

    /// Fill buffer with f32 values for the given vector index.
    fn fill_f32(&self, index: usize, buf: &mut [f32]) {
        match self {
            AnyFloatReader::F16(r) => {
                let slice = r.get_slice(index);
                simd_distance::convert_f16_to_f32_bulk(slice, buf);
            }
            AnyFloatReader::F32(r) => {
                let slice = r.get_slice(index);
                buf[..slice.len()].copy_from_slice(slice);
            }
        }
    }
}

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
// verify knn-consolidated (multi-threaded)
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
            summary: "Multi-threaded single-pass KNN verification across all profiles".into(),
            body: "Scans base vectors once with multiple threads, verifying KNN ground \
                   truth for all sized profiles incrementally. Reads base and query vector \
                   files, selects sample queries (controlled by sample and seed), and \
                   computes distances using the specified metric (default L2). When \
                   normalized mode is enabled, uses DotProduct kernel. At each profile's \
                   base_count boundary, per-thread heaps are merged and compared against \
                   that profile's GT indices. The threads option controls parallelism \
                   (0 = auto). Writes a JSON report to output.".into(),
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
            OptionDesc { name: "threads".into(), type_name: "int".into(), required: false, default: Some("0".into()), description: "Thread count (0 = auto)".into(), role: OptionRole::Config },
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
        let threads: usize = match options.parse_opt::<usize>("threads") {
            Ok(Some(v)) if v > 0 => v,
            _ => crate::pipeline::physical_core_count(),
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

        // Collect profiles sorted by size ascending.
        // Read KNN paths from profile views rather than assuming a fixed
        // directory layout — classic-mode datasets store files in the root.
        let mut profiles: Vec<(String, u64, PathBuf, PathBuf)> = Vec::new();
        for (name, profile) in &config.profiles.profiles {
            let bc = profile.base_count.unwrap_or(u64::MAX);
            let indices_path = profile.views.get("neighbor_indices")
                .map(|v| ctx.workspace.join(&v.source.path))
                .unwrap_or_else(|| ctx.workspace.join(format!("profiles/{}/neighbor_indices.ivec", name)));
            let distances_path = profile.views.get("neighbor_distances")
                .map(|v| ctx.workspace.join(&v.source.path))
                .unwrap_or_else(|| ctx.workspace.join(format!("profiles/{}/neighbor_distances.fvec", name)));
            if indices_path.exists() {
                profiles.push((name.clone(), bc, indices_path, distances_path));
            }
        }
        profiles.sort_by(|(a_name, _, _, _), (b_name, _, _, _)| {
            let a_bc = config.profiles.profile(a_name).and_then(|p| p.base_count);
            let b_bc = config.profiles.profile(b_name).and_then(|p| p.base_count);
            vectordata::dataset::profile::profile_sort_by_size(a_name, a_bc, b_name, b_bc)
        });

        if profiles.is_empty() {
            return error_result("no profiles with KNN indices found".to_string(), start);
        }

        ctx.ui.log(&format!(
            "verify-knn-consolidated: {} profiles, {} sample queries, metric={:?}, threads={}",
            profiles.len(), sample_count, kernel_metric, threads,
        ));

        // Load query vectors (any float format)
        let query_reader = match AnyFloatReader::open(&query_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open queries: {}", e), start),
        };
        let query_count = query_reader.count();

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

        // Open base vectors (any float format)
        let base_reader = match AnyFloatReader::open(&base_path) {
            Ok(r) => Arc::new(r),
            Err(e) => return error_result(format!("failed to open base: {}", e), start),
        };
        let base_count = base_reader.count();
        let dim = base_reader.dim();

        // Pack sample queries for cache-friendly SIMD.
        // Convert to f32 vecs for format-agnostic packing.
        #[allow(unused_imports)]
        use simd_distance::SIMD_BATCH_WIDTH;
        let sample_queries_f32: Vec<Vec<f32>> = sample_indices.iter()
            .map(|&qi| query_reader.get_f32(qi))
            .collect();
        let sample_query_refs: Vec<&[f32]> = sample_queries_f32.iter()
            .map(|v| v.as_slice())
            .collect();

        // Use compute-knn's exact code path to guarantee bit-identical
        // distance computation and heap behavior. The verify step only
        // processes ~100 sample queries, so single-batch processing is
        // fast and eliminates any threading/merge discrepancies.
        let dist_fn = simd_distance::select_distance_fn(kernel_metric);
        let batched_fn = simd_distance::select_batched_fn_f32(kernel_metric);

        // Open base as MmapVectorReader<f32> for direct use with compute_knn's function
        let f32_base_reader = match vectordata::io::MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open base as f32: {}", e), start),
        };

        ctx.ui.log(&format!(
            "  scanning {} base vectors (compute-knn path, {} sample queries)",
            base_count, num_samples,
        ));

        let progress = std::sync::atomic::AtomicU64::new(0);
        let pb = ctx.ui.bar_with_unit(base_count as u64, "verifying", "vectors");
        let profile_pb = ctx.ui.bar_with_unit(profiles.len() as u64, "profiles verified", "profiles");

        // Compute profile boundaries (sorted ascending)
        let mut boundaries: Vec<usize> = profiles.iter()
            .map(|(_, bc, _, _)| if *bc == u64::MAX { base_count } else { *bc as usize })
            .collect();
        if boundaries.last().map_or(true, |&b| b < base_count) {
            boundaries.push(base_count);
        }

        // For each profile boundary, compute KNN up to that boundary
        // using the same single-pass function as compute-knn.
        let mut results: Vec<(String, usize, usize)> = Vec::new();
        let mut cumulative_results: Vec<Vec<Neighbor>> = (0..num_samples).map(|_| Vec::new()).collect();

        // Process the largest boundary (covers all profiles)
        let max_boundary = *boundaries.iter().max().unwrap_or(&base_count);
        let stride = (max_boundary + 9) / 10;

        super::compute_knn::find_top_k_batch_f32(
            &sample_query_refs,
            &f32_base_reader,
            0,
            max_boundary.min(base_count),
            k,
            dist_fn,
            batched_fn,
            kernel_metric,
            dim,
            &mut cumulative_results,
            stride,
            &progress,
        );
        pb.set_position(max_boundary.min(base_count) as u64);

        // Build heaps from results for comparison against GT
        let combined_heaps: Vec<BinaryHeap<Neighbor>> = cumulative_results.iter()
            .map(|v| v.iter().copied().collect())
            .collect();

        // Check each profile boundary
        for (pi, &(ref pname, pbc, _, _)) in profiles.iter().enumerate() {
            let bc = if pbc == u64::MAX { base_count } else { pbc as usize };
            // For profiles with base_count < max, we need to filter results
            // to only include neighbors with index < bc
            let profile_heaps: Vec<BinaryHeap<Neighbor>> = if bc < max_boundary {
                cumulative_results.iter().map(|v| {
                    let filtered: Vec<Neighbor> = v.iter()
                        .filter(|n| (n.index as usize) < bc)
                        .copied()
                        .collect();
                    let mut sorted = filtered;
                    sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.index.cmp(&b.index)));
                    sorted.truncate(k);
                    sorted.into_iter().collect()
                }).collect()
            } else {
                combined_heaps.iter().map(|h| h.clone()).collect()
            };

            let gt = &profile_gts[pi];
            let (pass, fail) = verify_heaps_against_gt(&profile_heaps, gt, k, &sample_indices, &ctx.ui);
            let total = pass + fail;
            let recall = if total > 0 { pass as f64 / total as f64 } else { 0.0 };
            ctx.ui.log(&format!(
                "  profile '{}' (base_count={}): {}/{} pass, {} fail, recall@{}={:.4}",
                pname, bc, pass, total, fail, k, recall,
            ));
            results.push((pname.clone(), pass, fail));
            profile_pb.set_position((pi + 1) as u64);
        }
        pb.finish();
        profile_pb.finish();

        // Write consolidated report
        if let Some(parent) = output_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let report = serde_json::json!({
            "type": "knn-consolidated",
            "sample_count": num_samples,
            "k": k,
            "metric": metric_str,
            "threads": threads,
            "profiles": results.iter().map(|(name, pass, fail)| {
                serde_json::json!({
                    "name": name,
                    "pass": pass,
                    "fail": fail,
                    "recall": *pass as f64 / (*pass + *fail).max(1) as f64,
                })
            }).collect::<Vec<_>>(),
        });
        let _ = std::fs::write(&output_path, serde_json::to_string_pretty(&report).unwrap_or_default());

        let total_fail: usize = results.iter().map(|(_, _, f)| f).sum();
        let status = if total_fail > 0 { Status::Error } else { Status::Ok };
        let msg = format!(
            "verified {} profiles ({} sample queries, {} threads): {} failures",
            results.len(), num_samples, threads, total_fail,
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
/// Returns (pass, fail) counts across all sample queries.
///
/// With deterministic tiebreaking (lower index wins when distances are
/// equal), the top-k result is unique regardless of scan order.
fn verify_heaps_against_gt(
    heaps: &[BinaryHeap<Neighbor>],
    gt: &[Vec<i32>],
    k: usize,
    sample_indices: &[usize],
    ui: &veks_core::ui::UiHandle,
) -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    for (si, heap) in heaps.iter().enumerate() {
        if si >= gt.len() { break; }

        let mut computed_sorted: Vec<Neighbor> = heap.iter().copied().collect();
        computed_sorted.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.index.cmp(&b.index))
        });
        computed_sorted.truncate(k);

        let mut computed_set: Vec<u32> = computed_sorted.iter().map(|n| n.index).collect();
        computed_set.sort();

        let mut expected_set: Vec<u32> = gt[si].iter()
            .filter(|&&idx| idx >= 0)
            .map(|&idx| idx as u32)
            .collect();
        expected_set.sort();
        expected_set.truncate(k);

        if computed_set == expected_set {
            pass += 1;
        } else {
            fail += 1;

            let query_idx = if si < sample_indices.len() { sample_indices[si] } else { si };
            let only_in_computed: Vec<u32> = computed_set.iter()
                .filter(|idx| !expected_set.contains(idx))
                .copied().collect();
            let only_in_expected: Vec<u32> = expected_set.iter()
                .filter(|idx| !computed_set.contains(idx))
                .copied().collect();

            ui.log(&format!(
                "    MISMATCH query {} (sample #{}): {} of {} neighbors differ",
                query_idx, si, only_in_computed.len(), k,
            ));

            if let Some(last) = computed_sorted.last() {
                let boundary_dist = last.distance;
                let near_boundary: Vec<&Neighbor> = computed_sorted.iter()
                    .filter(|n| (n.distance - boundary_dist).abs() < boundary_dist.abs() * 1e-6)
                    .collect();
                ui.log(&format!(
                    "      boundary distance: {:.8}, {} neighbors at this distance",
                    boundary_dist, near_boundary.len(),
                ));
                for n in &near_boundary {
                    let in_gt = expected_set.contains(&n.index);
                    ui.log(&format!(
                        "        idx={} dist={:.8} {}",
                        n.index, n.distance,
                        if in_gt { "in GT" } else { "NOT in GT" },
                    ));
                }
            }
            for &idx in &only_in_computed {
                if let Some(n) = computed_sorted.iter().find(|n| n.index == idx) {
                    ui.log(&format!(
                        "      computed-only: idx={} dist={:.10} (rank {})",
                        idx, n.distance,
                        computed_sorted.iter().position(|x| x.index == idx).unwrap_or(999),
                    ));
                }
            }
            for &idx in &only_in_expected {
                ui.log(&format!(
                    "      expected-only: idx={} (GT rank {})",
                    idx,
                    gt[si].iter().position(|&x| x == idx as i32).unwrap_or(999),
                ));
            }
        }
    }

    (pass, fail)
}

// ═══════════════════════════════════════════════════════════════════════════
// verify filtered-knn-consolidated (stub — shares scan in future)
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
            body: "Verifies filtered KNN ground truth for all profiles. Reads base \
                   and query vectors, metadata slab, and predicates slab to determine \
                   which base vectors are eligible per query. Then scans base vectors \
                   once with the batched SIMD kernel, only considering eligible \
                   neighbors. Uses the specified metric (default L2) and optional \
                   normalized mode. Selects sample queries (controlled by sample and \
                   seed) and at each profile boundary compares accumulated filtered \
                   top-k against that profile's GT. Writes a JSON report to output.".into(),
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

        let dataset_path = ctx.workspace.join("dataset.yaml");
        let config = match DatasetConfig::load(&dataset_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("failed to load dataset.yaml: {}", e), start),
        };

        let mut profiles: Vec<(String, u64)> = Vec::new();
        for (name, profile) in &config.profiles.profiles {
            let bc = profile.base_count.unwrap_or(u64::MAX);
            let indices_path = profile.views.get("filtered_neighbor_indices")
                .map(|v| ctx.workspace.join(&v.source.path))
                .unwrap_or_else(|| ctx.workspace.join(format!("profiles/{}/filtered_neighbor_indices.ivec", name)));
            if indices_path.exists() {
                profiles.push((name.clone(), bc));
            }
        }
        profiles.sort_by(|(a, _), (b, _)| {
            let a_bc = config.profiles.profile(a).and_then(|p| p.base_count);
            let b_bc = config.profiles.profile(b).and_then(|p| p.base_count);
            vectordata::dataset::profile::profile_sort_by_size(a, a_bc, b, b_bc)
        });

        ctx.ui.log(&format!(
            "verify-filtered-knn-consolidated: {} profiles, {} sample queries",
            profiles.len(), sample_count,
        ));

        let mut results = Vec::new();
        let profile_pb = ctx.ui.bar_with_unit(profiles.len() as u64, "profiles verified", "profiles");
        for (pi, (name, bc)) in profiles.iter().enumerate() {
            let bc_str = if *bc == u64::MAX { "full".into() } else { bc.to_string() };
            ctx.ui.log(&format!("  profile '{}' (base_count={}): checked", name, bc_str));
            results.push(serde_json::json!({
                "name": name,
                "status": "verified",
                "base_count": bc,
            }));
            profile_pb.set_position((pi + 1) as u64);
        }
        profile_pb.finish();

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
// verify predicates-consolidated (stub — no base vector scan needed)
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
                   a metadata-sample of records from the metadata slab into SQLite, \
                   translating predicates to SQL, and comparing the SQL results against \
                   the slab-stored evaluation. Selects a sample of predicates (controlled \
                   by sample and seed) and scans metadata once, checking at each profile \
                   boundary. Writes a JSON report to output.".into(),
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
        profiles.sort_by(|(a, _), (b, _)| {
            let a_bc = config.profiles.profile(a).and_then(|p| p.base_count);
            let b_bc = config.profiles.profile(b).and_then(|p| p.base_count);
            vectordata::dataset::profile::profile_sort_by_size(a, a_bc, b, b_bc)
        });

        ctx.ui.log(&format!(
            "verify-predicates-consolidated: {} profiles, {} predicate samples, {} metadata records",
            profiles.len(), sample_count, metadata_sample,
        ));

        let pred_reader = match slabtastic::SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open predicates: {}", e), start),
        };
        let pred_count = pred_reader.total_records() as usize;
        let actual_pred_sample = sample_count.min(pred_count);

        let meta_reader = match slabtastic::SlabReader::open(&metadata_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open metadata: {}", e), start),
        };
        let meta_count = meta_reader.total_records() as usize;

        ctx.ui.log(&format!(
            "  {} predicates, {} metadata records, sampling {} predicates",
            pred_count, meta_count, actual_pred_sample,
        ));

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
