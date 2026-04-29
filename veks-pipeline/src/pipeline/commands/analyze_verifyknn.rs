// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: verify KNN ground truth correctness.
//!
//! Loads base vectors, query vectors, and precomputed neighbor indices, then
//! recomputes distances for a range of queries and verifies the provided
//! neighborhoods match the true nearest neighbors.
//!
//! Supports distance metrics: L2, Cosine, DotProduct, L1.
//!
//! Equivalent to the Java `CMD_analyze_verifyknn` command.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use super::source_window::resolve_source;

/// Pipeline command: verify KNN results.
pub struct AnalyzeVerifyKnnOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeVerifyKnnOp)
}

/// Distance metric (shared with compute_knn but self-contained here).
#[derive(Debug, Clone, Copy)]
enum DistanceMetric {
    L2,
    Cosine,
    DotProduct,
    L1,
}

impl DistanceMetric {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "L2" | "EUCLIDEAN" => Some(DistanceMetric::L2),
            "COSINE" => Some(DistanceMetric::Cosine),
            "DOT_PRODUCT" | "DOTPRODUCT" | "DOT" => Some(DistanceMetric::DotProduct),
            "L1" | "MANHATTAN" => Some(DistanceMetric::L1),
            _ => None,
        }
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::L2 => {
                let mut sum = 0.0f64;
                for i in 0..a.len() {
                    let d = (a[i] - b[i]) as f64;
                    sum += d * d;
                }
                sum.sqrt() as f32
            }
            DistanceMetric::Cosine => {
                let mut dot = 0.0f64;
                let mut na = 0.0f64;
                let mut nb = 0.0f64;
                for i in 0..a.len() {
                    let ai = a[i] as f64;
                    let bi = b[i] as f64;
                    dot += ai * bi;
                    na += ai * ai;
                    nb += bi * bi;
                }
                let denom = (na * nb).sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    (1.0 - dot / denom) as f32
                }
            }
            DistanceMetric::DotProduct => {
                let mut dot = 0.0f64;
                for i in 0..a.len() {
                    dot += (a[i] as f64) * (b[i] as f64);
                }
                -dot as f32
            }
            DistanceMetric::L1 => {
                let mut sum = 0.0f64;
                for i in 0..a.len() {
                    sum += ((a[i] - b[i]) as f64).abs();
                }
                sum as f32
            }
        }
    }
}

#[derive(Clone)]
struct Neighbor {
    index: u32,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Find true top-k nearest neighbors by brute force.
fn find_true_top_k(
    query: &[f32],
    base_reader: &MmapVectorReader<f32>,
    base_offset: usize,
    base_end: usize,
    k: usize,
    metric: DistanceMetric,
    inner_bar: Option<&veks_core::ui::ProgressHandle>,
) -> Vec<Neighbor> {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    // Tick the inner bar in chunks. Once per base vector would dominate
    // run time on a billion-vector scan; once per ~1M strikes a balance
    // between meaningful progress signal and negligible overhead, since
    // the bar's own renderer is throttled to ~250–500ms anyway.
    const TICK_EVERY: usize = 1_000_000;
    let mut since_tick = 0usize;

    for i in base_offset..base_end {
        let base_vec = match base_reader.get(i) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dist = metric.distance(query, &base_vec);

        // Store window-relative index
        let window_idx = (i - base_offset) as u32;
        if heap.len() < k {
            heap.push(Neighbor {
                index: window_idx,
                distance: dist,
            });
        } else if let Some(worst) = heap.peek() {
            if dist < worst.distance {
                heap.pop();
                heap.push(Neighbor {
                    index: window_idx,
                    distance: dist,
                });
            }
        }

        since_tick += 1;
        if since_tick >= TICK_EVERY {
            if let Some(bar) = inner_bar {
                bar.inc(since_tick as u64);
            }
            since_tick = 0;
        }
    }
    if let Some(bar) = inner_bar {
        if since_tick > 0 { bar.inc(since_tick as u64); }
    }

    let mut result: Vec<Neighbor> = heap.into_vec();
    result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result
}

/// Read neighbor indices from an ivec file for a specific query.
fn read_ivec_row(reader: &MmapVectorReader<i32>, row: usize) -> Result<Vec<i32>, String> {
    reader
        .get(row)
        .map_err(|e| format!("failed to read indices row {}: {}", row, e))
}

impl CommandOp for AnalyzeVerifyKnnOp {
    fn command_path(&self) -> &str {
        "analyze verify-knn"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify KNN results by recomputing distances".into(),
            body: format!(
                "# analyze verify-knn\n\n\
                Verify KNN results by recomputing distances.\n\n\
                ## Description\n\n\
                Loads base vectors, query vectors, and precomputed neighbor indices, then \
                recomputes distances for a configurable range of queries using brute-force \
                exhaustive search. For each query, the true top-k nearest neighbors are \
                computed from scratch and compared against the stored ground-truth indices. \
                Supports L2 (Euclidean), Cosine, DotProduct, and L1 (Manhattan) distance \
                metrics.\n\n\
                ## How It Works\n\n\
                For each query vector, the command performs a full scan of the base vector \
                set (or a windowed subset) to find the exact k nearest neighbors. It then \
                compares the resulting index set against the precomputed indices from the \
                ivec file. If the sets do not match, it checks whether the discrepancy is \
                caused by distance ties (multiple vectors at the same distance from the \
                query) using the configurable `phi` tolerance. Queries that differ only \
                due to ties are counted as passes.\n\n\
                ## Role in Dataset Preparation\n\n\
                This command is critical for catching data corruption, byte-order \
                (endianness) issues, shuffle bugs, or off-by-one errors that would \
                produce incorrect ground-truth files and therefore invalid benchmark \
                results. It should be run after computing KNN ground truth and again \
                after any transformation that reorders or modifies the base vectors. \
                The `range` option allows verifying a subset of queries to save time \
                on very large datasets while still providing confidence in correctness.\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "Base vector mmap and distance recomputation".into(), adjustable: false },
            ResourceDesc { name: "threads".into(), description: "Parallel distance recomputation".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("indices") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let metric_str = options.get("metric").unwrap_or("L2");
        let metric = match DistanceMetric::from_str(metric_str) {
            Some(m) => m,
            None => {
                return error_result(
                    format!("unknown metric: '{}'. Use L2, COSINE, DOT_PRODUCT, or L1", metric_str),
                    start,
                )
            }
        };

        // Tolerance for floating-point comparison
        let phi: f32 = options
            .get("phi")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.001);

        // Range of queries to verify (default: all)
        let range_str = options.get("range");

        let base_source = match resolve_source(base_str, &ctx.workspace) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);

        // Open readers
        let base_reader = match MmapVectorReader::<f32>::open_fvec(&base_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open base {}: {}", base_path.display(), e),
                    start,
                )
            }
        };

        // Apply window to base vectors
        let file_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&base_reader);
        let (base_offset, base_end) = match base_source.window {
            Some((ws, we)) => {
                let ws = ws.min(file_count);
                let we = we.min(file_count);
                (ws, we)
            }
            None => (0, file_count),
        };
        let query_reader = match MmapVectorReader::<f32>::open_fvec(&query_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open query {}: {}", query_path.display(), e),
                    start,
                )
            }
        };
        let indices_reader = match MmapVectorReader::<i32>::open_ivec(&indices_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open indices {}: {}", indices_path.display(), e),
                    start,
                )
            }
        };

        let query_count = <MmapVectorReader<f32> as VectorReader<f32>>::count(&query_reader);
        let indices_count = <MmapVectorReader<i32> as VectorReader<i32>>::count(&indices_reader);

        if indices_count != query_count {
            return error_result(
                format!(
                    "query count ({}) != indices count ({})",
                    query_count, indices_count
                ),
                start,
            );
        }

        // Auto-detect k from indices dimension
        let k = <MmapVectorReader<i32> as VectorReader<i32>>::dim(&indices_reader);

        // Parse range
        let (range_start, range_end) = match range_str {
            Some(s) => match parse_range(s, query_count) {
                Ok(r) => r,
                Err(e) => return error_result(e, start),
            },
            None => (0, query_count),
        };

        let n_queries = range_end - range_start;
        let n_base = base_end - base_offset;
        ctx.ui.log(&format!(
            "Verify KNN: queries [{}, {}), k={}, metric={:?}, phi={}",
            range_start, range_end, k, metric, phi
        ));
        let total_distances = n_queries as u64 * n_base as u64;
        let total_label = if total_distances >= 1_000_000_000_000 {
            format!("{:.1}T", total_distances as f64 / 1e12)
        } else if total_distances >= 1_000_000_000 {
            format!("{:.1}G", total_distances as f64 / 1e9)
        } else if total_distances >= 1_000_000 {
            format!("{:.1}M", total_distances as f64 / 1e6)
        } else {
            total_distances.to_string()
        };
        ctx.ui.log(&format!(
            "  scanning {} base vector(s) per query, {} query/scan(s) total \
             — total distance computations: {}",
            n_base, n_queries, total_label,
        ));

        // Outer progress bar: queries verified. The label includes
        // pass/fail running totals so even slow scans (billion-vector
        // base files take minutes per query) give a continuously
        // updating signal of "how far through" + "what's the verdict
        // looking like so far". Without this, the user sees only the
        // header line until the entire verification finishes —
        // potentially many hours for large base × query combos.
        let pb = ctx.ui.bar_with_unit(n_queries as u64, "verifying queries", "queries");
        // Inner progress bar: base vectors scanned for the *current*
        // query. At billion-vector scale a single brute-force scan
        // takes minutes, so without this users see no feedback between
        // outer-bar ticks. The inner bar gets its position reset to 0
        // before each query.
        let inner_bar = ctx.ui.bar_with_unit(
            n_base as u64,
            format!("scanning base for query {}", range_start),
            "base vectors",
        );
        let mut pass_count = 0usize;
        let mut fail_count = 0usize;
        let verify_start = std::time::Instant::now();

        for qi in range_start..range_end {
            // Periodic governor checkpoint every 1000 queries
            if (qi - range_start) % 1000 == 0 {
                if ctx.governor.checkpoint() {
                    ctx.ui.log("  governor: throttle active");
                }
            }

            let query_vec = match query_reader.get(qi) {
                Ok(v) => v,
                Err(e) => {
                    return error_result(format!("failed to read query {}: {}", qi, e), start)
                }
            };

            // Read provided indices
            let provided_indices = match read_ivec_row(&indices_reader, qi) {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            };

            // Reset the inner bar for this query and update its label
            // so the user can see which query is currently scanning.
            inner_bar.set_position(0);
            inner_bar.set_message(format!("scanning base for query {}", qi));

            // Compute true neighbors (within window)
            let true_neighbors = find_true_top_k(
                &query_vec, &base_reader, base_offset, base_end, k, metric,
                Some(&inner_bar),
            );

            // Compare: check overlap of index sets
            let true_set: std::collections::HashSet<u32> =
                true_neighbors.iter().map(|n| n.index).collect();
            let _provided_set: std::collections::HashSet<i32> =
                provided_indices.iter().copied().collect();

            let matching = provided_indices
                .iter()
                .filter(|&&idx| idx >= 0 && true_set.contains(&(idx as u32)))
                .count();

            let is_pass = matching == k;

            if is_pass {
                pass_count += 1;
            } else {
                fail_count += 1;
                // Check if it's a distance-tie situation
                let worst_true_dist = true_neighbors.last().map(|n| n.distance).unwrap_or(0.0);
                let mut tie_adjusted_matching = matching;
                for &idx in &provided_indices {
                    if idx >= 0 && !true_set.contains(&(idx as u32)) {
                        // This index wasn't in true set — check if its distance
                        // is within phi of the worst true neighbor
                        let abs_idx = idx as usize + base_offset;
                        let base_vec = base_reader.get(abs_idx).unwrap_or_default();
                        let dist = metric.distance(&query_vec, &base_vec);
                        if (dist - worst_true_dist).abs() <= phi {
                            tie_adjusted_matching += 1;
                        }
                    }
                }
                if tie_adjusted_matching == k {
                    // Ties explain the difference
                    fail_count -= 1;
                    pass_count += 1;
                } else {
                    ctx.ui.log(&format!(
                        "  FAIL query {}: {}/{} matching ({} with ties)",
                        qi, matching, k, tie_adjusted_matching
                    ));
                }
            }

            // Bump the bar after every query — the per-query brute-
            // force scan can take minutes on a billion-vector base,
            // so even one update is meaningful. The bar's own
            // throttling keeps redraw overhead negligible.
            pb.inc(1);

            // Periodic running summary (every 100 queries OR on the
            // last query) for the batch-mode log scrollback. The bar
            // covers the live UI; this gives anyone tail-ing
            // dataset.log a stable progress trail.
            if (qi - range_start + 1) % 100 == 0 || qi + 1 == range_end {
                let done = (qi - range_start + 1) as u64;
                let elapsed = verify_start.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 { done as f64 / elapsed } else { 0.0 };
                let eta = if rate > 0.0 {
                    let remaining = (n_queries as u64).saturating_sub(done);
                    format!("{:.0}s", remaining as f64 / rate)
                } else {
                    "?".to_string()
                };
                ctx.ui.log(&format!(
                    "  {}/{} queries verified — {} pass, {} fail ({:.1} q/s, eta {})",
                    done, n_queries, pass_count, fail_count, rate, eta,
                ));
            }
        }
        pb.finish();

        let total = pass_count + fail_count;
        let status = if fail_count == 0 {
            Status::Ok
        } else {
            Status::Error
        };

        CommandResult {
            status,
            message: format!(
                "verified {}/{} queries: {} pass, {} fail (k={}, metric={:?})",
                total,
                range_end - range_start,
                pass_count,
                fail_count,
                k,
                metric
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Base vectors file (fvec)".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "query".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Query vectors file (fvec)".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "indices".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Precomputed neighbor indices (ivec)".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "metric".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("L2".to_string()),
                description: "Distance metric: L2, COSINE, DOT_PRODUCT, L1".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "phi".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("0.001".to_string()),
                description: "Floating-point tolerance for distance comparison".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Query range to verify (e.g. '0..100')".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Parse a range string like "0..100" or "[0,100)".
fn parse_range(s: &str, max: usize) -> Result<(usize, usize), String> {
    if let Some((a, b)) = s.split_once("..") {
        let start: usize = a.trim().parse().map_err(|_| format!("invalid range start: '{}'", a))?;
        let end: usize = if b.trim().is_empty() {
            max
        } else {
            b.trim().parse().map_err(|_| format!("invalid range end: '{}'", b))?
        };
        if start >= end || end > max {
            return Err(format!("range {}..{} out of bounds (max {})", start, end, max));
        }
        Ok((start, end))
    } else {
        Err(format!("invalid range format: '{}'. Use 'start..end'", s))
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::commands::compute_knn::ComputeKnnOp;
    use crate::pipeline::commands::gen_vectors::GenerateVectorsOp;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
        }
    }

    #[test]
    fn test_verify_knn_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Generate base and query vectors
        let base = ws.join("base.fvec");
        let query = ws.join("query.fvec");
        for (path, count) in [(&base, "30"), (&query, "5")] {
            let mut opts = Options::new();
            opts.set("output", path.to_string_lossy().to_string());
            opts.set("dimension", "4");
            opts.set("count", count);
            opts.set("seed", "42");
            let mut gen_op = GenerateVectorsOp;
            gen_op.execute(&opts, &mut ctx);
        }

        // Compute KNN
        let indices = ws.join("indices.ivec");
        let mut opts = Options::new();
        opts.set("base", base.to_string_lossy().to_string());
        opts.set("query", query.to_string_lossy().to_string());
        opts.set("indices", indices.to_string_lossy().to_string());
        opts.set("neighbors", "3");
        opts.set("metric", "L2");
        let mut knn = ComputeKnnOp;
        let r = knn.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        // Verify — should pass since we just computed the correct answer
        let mut opts = Options::new();
        opts.set("base", base.to_string_lossy().to_string());
        opts.set("query", query.to_string_lossy().to_string());
        opts.set("indices", indices.to_string_lossy().to_string());
        opts.set("metric", "L2");

        let mut verify = AnalyzeVerifyKnnOp;
        let result = verify.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "verify failed: {}", result.message);
    }

    #[test]
    fn test_verify_knn_with_range() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let base = ws.join("base.fvec");
        let query = ws.join("query.fvec");
        for (path, count) in [(&base, "20"), (&query, "10")] {
            let mut opts = Options::new();
            opts.set("output", path.to_string_lossy().to_string());
            opts.set("dimension", "4");
            opts.set("count", count);
            opts.set("seed", "42");
            let mut gen_op = GenerateVectorsOp;
            gen_op.execute(&opts, &mut ctx);
        }

        let indices = ws.join("indices.ivec");
        let mut opts = Options::new();
        opts.set("base", base.to_string_lossy().to_string());
        opts.set("query", query.to_string_lossy().to_string());
        opts.set("indices", indices.to_string_lossy().to_string());
        opts.set("neighbors", "3");
        let mut knn = ComputeKnnOp;
        knn.execute(&opts, &mut ctx);

        // Verify only queries 2..5
        let mut opts = Options::new();
        opts.set("base", base.to_string_lossy().to_string());
        opts.set("query", query.to_string_lossy().to_string());
        opts.set("indices", indices.to_string_lossy().to_string());
        opts.set("range", "2..5");

        let mut verify = AnalyzeVerifyKnnOp;
        let result = verify.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("3/3"));
    }

    #[test]
    fn test_parse_range_valid() {
        assert_eq!(parse_range("0..10", 100).unwrap(), (0, 10));
        assert_eq!(parse_range("5..50", 100).unwrap(), (5, 50));
        assert_eq!(parse_range("0..", 100).unwrap(), (0, 100));
    }

    #[test]
    fn test_parse_range_invalid() {
        assert!(parse_range("50..10", 100).is_err());
        assert!(parse_range("0..200", 100).is_err());
        assert!(parse_range("abc", 100).is_err());
    }
}
