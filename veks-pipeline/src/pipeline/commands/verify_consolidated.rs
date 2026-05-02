// Copyright (c) Jonathan Shook
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
use std::time::Instant;

use crate::pipeline::command::*;
use crate::pipeline::element_type::ElementType;
use crate::pipeline::simd_distance;
// `Metric` is only referenced by the bare name inside the
// knnutils-gated `VerifyKnnConsolidatedOp::execute` impl below.
#[cfg(feature = "knnutils")]
use crate::pipeline::simd_distance::Metric;
use vectordata::io::XvecReader;
use vectordata::VectorReader;
use vectordata::dataset::DatasetConfig;

use super::compute_knn::Neighbor;

/// Type-erased holder for a base reader, kept alive only so its
/// `madvise(SEQUENTIAL)` hint persists across the streaming pread
/// scan. Reads themselves go through a separate `std::fs::File` so
/// we never name the typed reader on the hot path. The variants
/// match the type-dispatch arms in
/// `VerifyKnnConsolidatedOp::execute` — adding a new base element
/// type here means adding a new arm there.
pub(super) enum BaseKeepalive {
    F32(XvecReader<f32>),
    F16(XvecReader<half::f16>),
}

/// Format-agnostic float vector reader for verification.
/// Provides f32 slices regardless of storage precision.
pub(super) enum AnyFloatReader {
    F16(XvecReader<half::f16>),
    F32(XvecReader<f32>),
}

impl AnyFloatReader {
    pub(super) fn open(path: &Path) -> Result<Self, String> {
        let etype = ElementType::from_path(path)
            .unwrap_or(ElementType::F32);
        match etype {
            ElementType::F16 => XvecReader::<half::f16>::open_path(path)
                .map(AnyFloatReader::F16)
                .map_err(|e| format!("open {}: {}", path.display(), e)),
            _ => XvecReader::<f32>::open_path(path)
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

    pub(super) fn dim(&self) -> usize {
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
    pub(super) fn fill_f32(&self, index: usize, buf: &mut [f32]) {
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

// verify knn-consolidated uses the same sgemm kernel as compute knn-blas
// so distances are bit-identical and verify can't false-positive on ULP
// boundary tie-swaps. That kernel is feature-gated on `knnutils`
// (system BLAS link); the command is therefore only registered when
// knnutils is compiled in. Non-knnutils builds can still use the other
// verify commands (filtered / predicates).
#[cfg(feature = "knnutils")]
pub struct VerifyKnnConsolidatedOp;

#[cfg(feature = "knnutils")]
pub fn knn_consolidated_factory() -> Box<dyn CommandOp> {
    Box::new(VerifyKnnConsolidatedOp)
}

#[cfg(feature = "knnutils")]
impl CommandOp for VerifyKnnConsolidatedOp {
    fn command_path(&self) -> &str {
        "verify knn-consolidated"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

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
            ResourceDesc { name: "mem".into(), description: "GT indices and heap memory".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // Verifier drives `scan_range_sgemm` which calls cblas_sgemm
        // directly — poisoned by faiss-sys static MKL under multi-
        // threading (see pipeline::blas_abi). Force single-threaded.
        crate::pipeline::blas_abi::set_single_threaded_if_faiss();

        // Up-front: confirm the local dataset has the minimum facets
        // this verify kind requires (see pipeline::dataset_lookup).
        if let Err(e) = crate::pipeline::dataset_lookup::validate_and_log(
            ctx, options, crate::pipeline::dataset_lookup::VerifyKind::KnnConsolidated,
        ) {
            return error_result(e, start);
        }

        // Standalone-friendly: missing input options are resolved from
        // the dataset's profile/facet view (see pipeline::dataset_lookup).
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
        let metric = match Metric::from_str(metric_str) {
            Some(m) => m,
            None => return error_result(format!("unknown metric: '{}'", metric_str), start),
        };
        // L1 is not supported by the sgemm kernel (no matrix-form
        // equivalent). knn-blas can't compute L1 either, so there
        // wouldn't be any L1 output to verify; fail fast with a
        // precise reason.
        if matches!(metric, Metric::L1) {
            return error_result(
                "verify knn-consolidated: L1 is not supported by the sgemm verification kernel. \
                 Use metric=L2, COSINE, or DOT_PRODUCT.".to_string(),
                start,
            );
        }
        // Resolve the cosine strategy. Mirrors compute-knn-blas's
        // logic so verify and compute see the same arithmetic.
        let seg_metric = match metric {
            Metric::L2 => super::knn_segment::Metric::L2,
            Metric::Cosine => super::knn_segment::Metric::Cosine,
            Metric::DotProduct => super::knn_segment::Metric::DotProduct,
            Metric::L1 => unreachable!("L1 rejected above"),
        };
        let cosine_mode = match super::knn_segment::resolve_cosine_mode_for(
            matches!(seg_metric, super::knn_segment::Metric::Cosine),
            options,
        ) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };
        let kernel_seg_metric = match (seg_metric, cosine_mode) {
            (super::knn_segment::Metric::Cosine,
             Some(super::knn_segment::CosineMode::AssumeNormalized)) =>
                super::knn_segment::Metric::DotProduct,
            _ => seg_metric,
        };
        let use_ip = matches!(kernel_seg_metric,
            super::knn_segment::Metric::DotProduct
            | super::knn_segment::Metric::Cosine);
        let proper_cosine = matches!(cosine_mode,
            Some(super::knn_segment::CosineMode::ProperMetric));
        // `kernel_metric` kept around for the summary log only.
        let kernel_metric = metric;

        let sample_count: usize = match options.parse_or("sample", 100usize) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };
        let seed: u64 = match options.parse_or("seed", 42u64) {
            Ok(v) => v, Err(e) => return error_result(e, start),
        };
        let threads: usize = match options.parse_opt::<usize>("threads") {
            Ok(Some(v)) if v > 0 => v,
            _ => ctx.governor.current_or("threads", ctx.threads as u64) as usize,
        };

        // base_str may carry an inline window (e.g., `base.fvec[0..N)`).
        // Strip the window via source_window::resolve_source so opening
        // the file doesn't blow up on the trailing range syntax —
        // matching what compute-knn-blas does.
        let base_source = match super::source_window::resolve_source(&base_str, &ctx.workspace) {
            Ok(s) => s,
            Err(e) => return error_result(format!("parse base source: {}", e), start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(&query_str, &ctx.workspace);
        let output_path = resolve_path(&output_str, &ctx.workspace);

        // Discover profiles from dataset.yaml
        let dataset_path = ctx.workspace.join("dataset.yaml");
        let config = match DatasetConfig::load_and_resolve(&dataset_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("failed to load dataset.yaml: {}", e), start),
        };

        // Collect profiles sorted by size ascending.
        // Read KNN paths from profile views rather than assuming a fixed
        // directory layout — classic-mode datasets store files in the root.
        // Uses load_and_resolve to expand deferred sized profiles.
        let mut profiles: Vec<(String, u64, PathBuf, PathBuf)> = Vec::new();
        for (name, profile) in &config.profiles.profiles {
            // Skip partition profiles — they have their own verify-knn-partition
            // step. Partition profiles have independent base vectors (not windowed
            // from default) and must NOT be verified against the full base set.
            if profile.partition {
                continue;
            }

            let bc = profile.base_count.unwrap_or(u64::MAX);
            let indices_path = profile.views.get("neighbor_indices")
                .map(|v| ctx.workspace.join(&v.source.path))
                .unwrap_or_else(|| ctx.workspace.join(format!("profiles/{}/neighbor_indices.ivecs", name)));
            // We collect the distances_path for completeness but the
            // verifier only reads `neighbor_indices.ivecs` — GT
            // comparison is index-set based, the verifier recomputes
            // distances itself via `scan_range_sgemm`. If a future
            // change starts reading these fvec values, remember they
            // are in FAISS publication convention (see
            // [`knn_segment::published_to_kernel`]) and need conversion
            // before being compared to kernel-internal heap distances.
            let distances_path = profile.views.get("neighbor_distances")
                .map(|v| ctx.workspace.join(&v.source.path))
                .unwrap_or_else(|| ctx.workspace.join(format!("profiles/{}/neighbor_distances.fvecs", name)));
            if indices_path.exists() {
                profiles.push((name.clone(), bc, indices_path, distances_path));
            }
        }

        // Also discover profiles from the profiles/ directory that may
        // not be in the config (e.g., sized profiles with deferred expansion).
        let profiles_dir = ctx.workspace.join("profiles");
        if profiles_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&profiles_dir) {
                for entry in entries.flatten() {
                    if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) { continue; }
                    let pname = entry.file_name().to_string_lossy().to_string();
                    if profiles.iter().any(|(n, _, _, _)| n == &pname) { continue; }
                    let indices_path = entry.path().join("neighbor_indices.ivecs");
                    let distances_path = entry.path().join("neighbor_distances.fvecs");
                    if indices_path.exists() {
                        // Parse base_count from the directory name (sized profile names encode the count)
                        let bc = vectordata::dataset::source::parse_number_with_suffix(&pname)
                            .unwrap_or(u64::MAX);
                        profiles.push((pname, bc, indices_path, distances_path));
                    }
                }
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

        // Open base vectors with element-type dispatch. Matches the
        // pattern in `compute_knn.rs::execute_{f32,f16,f64}`: detect
        // the on-disk type once, monomorphize per type, run the
        // specialized arm. The verify scan is sgemm-based (BLAS only
        // knows f32) so the f16 arm just upcasts at unpack time —
        // see `pread_and_unpack`'s `elem_size` parameter. For a
        // dedicated f16 SIMD path (cblas_hgemm or stdarch f16),
        // it'd add a third arm here, not collapse into a runtime
        // enum like AnyFloatReader.
        let base_etype = match ElementType::from_path(&base_path) {
            Ok(t) => t,
            Err(e) => return error_result(format!("base extension: {e}"), start),
        };
        let base_keepalive: BaseKeepalive;
        let file_count: usize;
        let dim: usize;
        let elem_size: usize;
        match base_etype {
            ElementType::F32 => {
                let r = match vectordata::io::XvecReader::<f32>::open_path(&base_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(
                        format!("failed to open base as f32: {}", e), start),
                };
                file_count = VectorReader::<f32>::count(&r);
                dim = VectorReader::<f32>::dim(&r);
                r.advise_sequential();
                elem_size = 4;
                base_keepalive = BaseKeepalive::F32(r);
            }
            ElementType::F16 => {
                let r = match vectordata::io::XvecReader::<half::f16>::open_path(&base_path) {
                    Ok(r) => r,
                    Err(e) => return error_result(
                        format!("failed to open base as f16: {}", e), start),
                };
                file_count = VectorReader::<half::f16>::count(&r);
                dim = VectorReader::<half::f16>::dim(&r);
                r.advise_sequential();
                elem_size = 2;
                base_keepalive = BaseKeepalive::F16(r);
            }
            other => return error_result(
                format!(
                    "verify knn-consolidated: base element type {:?} not supported \
                     (only f32 / .fvec and f16 / .mvec are wired up)",
                    other,
                ),
                start,
            ),
        }
        // Effective base count = inline window OR `range` option,
        // clamped to the file. Compute-knn uses the same resolution
        // (`source_window::resolve_window`), so verify and compute see
        // identical base bounds when --base-fraction or sized-profile
        // injection trims the file.
        let effective_window = super::source_window::resolve_window(
            base_source.window, options.get("range"),
        );
        let (base_offset, base_count) = match effective_window {
            Some((ws, we)) => {
                let s = ws.min(file_count);
                (s, we.min(file_count).saturating_sub(s))
            }
            None => (0, file_count),
        };
        // (advise_sequential already called in the type-dispatch arm above)

        // Filter out profiles whose declared base_count exceeds the actual
        // dataset size. These are oversized profiles that were created at
        // parse time before base_count was known (e.g., `sized: [1m]` with
        // only 188 base vectors). Clamping them to base_count would make
        // them identical to the default profile, so skip with a warning.
        profiles.retain(|(name, bc, _, _)| {
            if *bc != u64::MAX && (*bc as usize) > base_count {
                ctx.ui.log(&format!(
                    "  skipping profile '{}': declared base_count {} exceeds actual {} base vectors",
                    name, bc, base_count,
                ));
                false
            } else {
                true
            }
        });

        // Load query vectors (any float format)
        let query_reader = match AnyFloatReader::open(&query_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("failed to open queries: {}", e), start),
        };
        let query_count = query_reader.count();

        // Enforce a sampling floor: at least 100 queries OR 10% of the
        // query set, whichever is greater. Verification with a tiny
        // sample doesn't catch broken results — a single passing
        // query was the previous "verified" signal even on totally
        // wrong output.
        let configured_sample = sample_count;
        let floor = std::cmp::max(100, query_count / 10);
        let target_sample = if configured_sample == 0 || configured_sample >= query_count {
            query_count
        } else {
            configured_sample.max(floor).min(query_count)
        };
        if target_sample > configured_sample && configured_sample != 0 {
            ctx.ui.log(&format!(
                "  raised sample from {} to {} (floor: max(100, 10% of {} queries))",
                configured_sample, target_sample, query_count,
            ));
        }

        // Sample WITHOUT replacement using a real PRNG.
        // The previous inline xorshift was wrong on two axes:
        // (1) seed = 0 produces a degenerate fixed-point (0 ^ (0<<n) = 0
        //     for any n), so every iteration emitted index 0 → after
        //     `dedup` you got num_samples = 1 regardless of the configured
        //     value. The user's pipeline runs with seed=0 (no shuffle),
        //     so verify silently sampled a single query.
        // (2) Push-then-dedup gives sampling with replacement → fewer
        //     than the configured number of unique indices even with a
        //     working RNG.
        // `rand::seq::index::sample` is the standard fix: deterministic
        // for a given seed, samples without replacement, returns
        // exactly the requested count (capped at the population).
        use rand::SeedableRng;
        use rand::seq::index::sample as sample_indices_fn;
        let sample_indices: Vec<usize> = if target_sample >= query_count {
            (0..query_count).collect()
        } else {
            // Pick a non-zero seed so any caller-supplied 0 still
            // gives a useful spread. Mix with a constant so the
            // user-visible seed semantics ("seed for the data
            // pipeline") aren't conflated with the verify sampler.
            let effective_seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(effective_seed);
            let mut idx: Vec<usize> = sample_indices_fn(&mut rng, query_count, target_sample)
                .into_iter()
                .collect();
            idx.sort();
            idx
        };
        let num_samples = sample_indices.len();

        let pct = if query_count > 0 { 100.0 * num_samples as f64 / query_count as f64 } else { 0.0 };
        ctx.ui.log(&format!(
            "verify knn-consolidated: KNN ground truth across {} profile{} via cblas_sgemm \
             (single-threaded, MKL-safe); sample={} of {} queries ({:.1}%), metric={:?}, threads={}",
            profiles.len(),
            if profiles.len() == 1 { "" } else { "s" },
            num_samples, query_count, pct, kernel_metric, threads,
        ));

        // Load GT indices for valid profiles only
        let mut profile_gts: Vec<Vec<Vec<i32>>> = Vec::with_capacity(profiles.len());
        for (name, _, indices_path, _) in &profiles {
            let gt_reader = match XvecReader::<i32>::open_path(indices_path) {
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

        // ── Pack sample queries flat for sgemm ────────────────────
        // Row-major `queries_packed[qi*dim..qi*dim+dim]`. Computed
        // once; fed to every boundary scan.
        let mut queries_packed: Vec<f32> = Vec::with_capacity(num_samples * dim);
        for &qi in &sample_indices {
            queries_packed.extend_from_slice(&query_reader.get_f32(qi));
        }
        let query_norms_sq: Vec<f32> = (0..num_samples).map(|qi| {
            let s = &queries_packed[qi * dim..(qi + 1) * dim];
            s.iter().map(|v| v * v).sum::<f32>()
        }).collect();

        // ── Open base for streaming pread ─────────────────────────
        // The typed reader (`base_keepalive`) is held only to keep
        // the mmap-based sequential-readahead advice live across
        // the scan; the sgemm scan reads via pread into heap
        // buffers so RSS stays bounded. `elem_size` (4 for f32 /
        // 2 for f16) controls how `pread_and_unpack` decodes
        // each record into the f32 sgemm packed buffer.
        let base_file = match std::fs::File::open(&base_path) {
            Ok(f) => f,
            Err(e) => return error_result(format!("open base for streaming: {}", e), start),
        };

        // ── Allocate scan buffers ONCE, reused across boundaries ──
        let mut scan_buffers = super::compute_knn_blas::SgemmScanBuffers::new(num_samples, dim);
        let vecs_per_chunk = scan_buffers.vecs_per_chunk();
        let qb_size = scan_buffers.qb_size();
        ctx.ui.log(&format!(
            "  chunk plan: {} base/chunk × {} queries/sub-batch",
            vecs_per_chunk, qb_size,
        ));
        ctx.ui.log(&format!(
            "  scanning {} base vectors ({} sample queries, metric={:?}, engine=BLAS sgemm)",
            base_count, num_samples, kernel_metric,
        ));

        // ── Boundary grouping ─────────────────────────────────────
        let mut boundary_profiles: Vec<(usize, Vec<usize>)> = Vec::new();
        {
            let mut boundary_map: std::collections::BTreeMap<usize, Vec<usize>> =
                std::collections::BTreeMap::new();
            for (pi, &(_, pbc, _, _)) in profiles.iter().enumerate() {
                let bc = if pbc == u64::MAX { base_count } else { (pbc as usize).min(base_count) };
                boundary_map.entry(bc).or_default().push(pi);
            }
            for (bc, pis) in boundary_map {
                boundary_profiles.push((bc, pis));
            }
        }

        let mut results: Vec<(String, usize, usize)> = vec![
            (String::new(), 0, 0); profiles.len()
        ];
        // Running per-query top-k for [0, prev_boundary). Maintained
        // in-place across boundaries so each scan extends the range
        // rather than rebuilding from scratch.
        let mut cumulative_heaps: Vec<BinaryHeap<Neighbor>> =
            (0..num_samples).map(|_| BinaryHeap::with_capacity(k + 1)).collect();
        let mut cumulative_thresholds: Vec<f32> = vec![f32::INFINITY; num_samples];
        // Drop unused: we no longer have rayon workers at this level.
        let _ = (threads, &base_keepalive);
        let scan_pb = ctx.ui.bar_with_unit(
            base_count as u64,
            &format!("scanning {} base ({}q × {}d)", base_count, num_samples, dim),
            "vectors",
        );
        let boundary_pb = ctx.ui.bar_with_unit(
            boundary_profiles.len() as u64,
            "boundaries",
            "boundaries",
        );
        let profile_pb = ctx.ui.bar_with_unit(profiles.len() as u64, "profiles verified", "profiles");
        let mut prev_boundary = 0usize;
        let mut profiles_verified = 0usize;

        // Scan base vectors in segments between profile boundaries.
        // `scan_range_sgemm` reuses the exact kernel + buffer layout
        // that `compute knn-blas` used to produce the output we're
        // verifying — so distances agree to the last bit of f32 and
        // tie-ordering cannot drift between compute and verify.
        for (boundary_idx, (boundary, profile_indices)) in boundary_profiles.iter().enumerate() {
            let seg_start = prev_boundary;
            let seg_end = *boundary;

            if seg_end > seg_start {
                let phase_start = std::time::Instant::now();
                let pb_ref = &scan_pb;
                let prev = prev_boundary;
                let mut tick_cb = move |done_in_seg: u64| {
                    pb_ref.set_position((prev as u64) + done_in_seg);
                };
                // Boundaries (`seg_start`, `seg_end`) are in
                // window-relative space ([0..base_count)). Translate
                // to absolute file indices by adding `base_offset` so
                // pread reads the right bytes when the source has a
                // non-zero window start.
                if let Err(e) = super::compute_knn_blas::scan_range_sgemm(
                    &base_file,
                    (base_offset + seg_start)..(base_offset + seg_end),
                    dim,
                    elem_size,
                    &queries_packed,
                    num_samples,
                    &query_norms_sq,
                    use_ip,
                    proper_cosine,
                    k,
                    &mut scan_buffers,
                    &mut cumulative_heaps,
                    &mut cumulative_thresholds,
                    Some(&mut tick_cb),
                ) {
                    return error_result(
                        format!("scan [{}..{}): {}", seg_start, seg_end, e),
                        start,
                    );
                }
                scan_pb.set_position(seg_end as u64);
                let scan_elapsed = phase_start.elapsed();
                ctx.ui.log(&format!(
                    "  scan [{}..{}): {:.1}s ({:.1}M base/s)",
                    seg_start, seg_end,
                    scan_elapsed.as_secs_f64(),
                    (seg_end - seg_start) as f64 / scan_elapsed.as_secs_f64() / 1e6,
                ));
            }

            prev_boundary = seg_end;

            // Verify all profiles at this boundary. Build a snapshot of
            // the running heaps — `verify_heaps_against_gt` consumes
            // heaps by reference (sorted internally) and doesn't mutate
            // the originals.
            let verify_start = std::time::Instant::now();
            let verify_heaps: Vec<BinaryHeap<Neighbor>> = cumulative_heaps.iter()
                .map(|h| h.iter().copied().collect())
                .collect();

            for &pi in profile_indices {
                let (ref pname, _, _, _) = profiles[pi];
                let gt = &profile_gts[pi];
                let (pass, fail) = verify_heaps_against_gt(&verify_heaps, gt, k, &sample_indices, &ctx.ui);
                let total = pass + fail;
                let recall = if total > 0 { pass as f64 / total as f64 } else { 0.0 };
                ctx.ui.log(&format!(
                    "  profile '{}' (base_count={}): {}/{} pass, {} fail, recall@{}={:.4}",
                    pname, seg_end, pass, total, fail, k, recall,
                ));
                results[pi] = (pname.clone(), pass, fail);
                profiles_verified += 1;
                profile_pb.set_position(profiles_verified as u64);
            }
            let verify_elapsed = verify_start.elapsed();
            ctx.ui.log(&format!("  verify phase: {:.1}s", verify_elapsed.as_secs_f64()));
            boundary_pb.set_position((boundary_idx + 1) as u64);
        }
        scan_pb.finish();
        boundary_pb.finish();
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

        // Use canonical comparison logic
        {
            use super::knn_compare;

            let computed_ordinals: Vec<i32> = computed_sorted.iter()
                .map(|n| n.index as i32)
                .collect();
            let expected_ordinals: Vec<i32> = expected_set.iter()
                .map(|&idx| idx as i32)
                .collect();

            let result = knn_compare::compare_query_ordinals(&computed_ordinals, &expected_ordinals);

            if result.is_acceptable() {
                pass += 1;
                if let knn_compare::QueryResult::BoundaryMismatch(n) = &result {
                    if pass <= 3 {
                        let query_idx = if si < sample_indices.len() { sample_indices[si] } else { si };
                        ui.log(&format!(
                            "    tie-break query {} (sample #{}): {} boundary swaps (acceptable)",
                            query_idx, si, n,
                        ));
                    }
                }
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
                    let epsilon = boundary_dist.abs() * 1e-6_f32;
                    let near_boundary: Vec<&Neighbor> = computed_sorted.iter()
                        .filter(|n| (n.distance - boundary_dist).abs() < epsilon)
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

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }
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

        // Belt-and-suspenders against the faiss-sys static-MKL bug
        // (see pipeline::blas_abi). Filtered-knn verification may
        // dispatch to sgemm-backed inner kernels.
        crate::pipeline::blas_abi::set_single_threaded_if_faiss();

        // Up-front: confirm the local dataset has the minimum facets
        // this verify kind requires (see pipeline::dataset_lookup).
        if let Err(e) = crate::pipeline::dataset_lookup::validate_and_log(
            ctx, options, crate::pipeline::dataset_lookup::VerifyKind::FilteredKnnConsolidated,
        ) {
            return error_result(e, start);
        }

        let output_str = match options.require("output") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let output_path = resolve_path(&output_str, &ctx.workspace);

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
            // Skip partition profiles
            if profile.partition { continue; }

            let bc = profile.base_count.unwrap_or(u64::MAX);
            let indices_path = profile.views.get("filtered_neighbor_indices")
                .map(|v| ctx.workspace.join(&v.source.path))
                .unwrap_or_else(|| ctx.workspace.join(format!("profiles/{}/filtered_neighbor_indices.ivecs", name)));
            if indices_path.exists() {
                profiles.push((name.clone(), bc));
            }
        }
        profiles.sort_by(|(a, _), (b, _)| {
            let a_bc = config.profiles.profile(a).and_then(|p| p.base_count);
            let b_bc = config.profiles.profile(b).and_then(|p| p.base_count);
            vectordata::dataset::profile::profile_sort_by_size(a, a_bc, b, b_bc)
        });

        // Load base and query vectors (standalone-friendly: see dataset_lookup).
        let base_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "base", "base_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let query_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "query", "query_vectors",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let base_path = resolve_path(&base_str, &ctx.workspace);
        let query_path = resolve_path(&query_str, &ctx.workspace);

        let metric_str = options.get("metric").unwrap_or("L2");
        let metric = crate::pipeline::simd_distance::Metric::from_str(metric_str)
            .unwrap_or(crate::pipeline::simd_distance::Metric::L2);
        let dist_fn = crate::pipeline::simd_distance::select_distance_fn(metric);

        let base_reader = match vectordata::io::XvecReader::<f32>::open_path(&base_path) {
            Ok(r) => r, Err(e) => return error_result(format!("open base: {}", e), start),
        };
        let query_reader = match vectordata::io::XvecReader::<f32>::open_path(&query_path) {
            Ok(r) => r, Err(e) => return error_result(format!("open query: {}", e), start),
        };

        let base_count = vectordata::VectorReader::<f32>::count(&base_reader);
        let query_count = vectordata::VectorReader::<f32>::count(&query_reader);

        let actual_sample = sample_count.min(query_count);
        ctx.ui.log(&format!(
            "verify filtered-knn-consolidated: filtered KNN ground truth across {} profile{} via cblas_sgemm \
             (single-threaded, MKL-safe); sample={} of {} queries, base={}",
            profiles.len(),
            if profiles.len() == 1 { "" } else { "s" },
            actual_sample, query_count, base_count,
        ));
        let step = if query_count <= actual_sample { 1 } else { query_count / actual_sample };
        let sample_indices: Vec<usize> = (0..actual_sample).map(|i| i * step).collect();

        let mut results = Vec::new();
        for (name, bc) in &profiles {
            let bc_val = if *bc == u64::MAX { base_count } else { *bc as usize };

            // Load stored filtered GT for this profile
            let gt_path = ctx.workspace.join(format!("profiles/{}/filtered_neighbor_indices.ivecs", name));
            if !gt_path.exists() {
                results.push(serde_json::json!({
                    "name": name, "status": "skip", "message": "no filtered GT file",
                }));
                continue;
            }

            // Load predicate results — try ivvec (canonical), ivec (legacy), slab
            let pred_indices_candidates = [
                format!("profiles/{}/metadata_indices.ivvec", name),
                format!("profiles/{}/metadata_indices.ivec", name),
                format!("profiles/{}/metadata_indices.slab", name),
            ];
            let pred_indices = {
                let mut loaded = None;
                let mut last_err = String::new();
                for candidate in &pred_indices_candidates {
                    let path = ctx.workspace.join(candidate);
                    if path.exists() {
                        match crate::pipeline::commands::compute_filtered_knn::PredicateIndices::open(&path) {
                            Ok(r) => { loaded = Some(r); break; }
                            Err(e) => { last_err = e; }
                        }
                    }
                }
                match loaded {
                    Some(r) => r,
                    None => {
                        results.push(serde_json::json!({
                            "name": name, "status": "error",
                            "message": if last_err.is_empty() { "no metadata_indices file found".into() } else { last_err },
                        }));
                        continue;
                    }
                }
            };

            // Load stored filtered GT as ivec
            let gt_data = match std::fs::read(&gt_path) {
                Ok(d) => d, Err(e) => {
                    results.push(serde_json::json!({
                        "name": name, "status": "error", "message": format!("{}", e),
                    }));
                    continue;
                }
            };

            // Parse GT ivec records
            let mut gt_records: Vec<Vec<i32>> = Vec::new();
            {
                let mut offset = 0;
                while offset + 4 <= gt_data.len() {
                    let dim = i32::from_le_bytes(gt_data[offset..offset+4].try_into().unwrap()) as usize;
                    offset += 4;
                    if offset + dim * 4 > gt_data.len() { break; }
                    let mut vals = Vec::with_capacity(dim);
                    for i in 0..dim {
                        let fo = offset + i * 4;
                        vals.push(i32::from_le_bytes(gt_data[fo..fo+4].try_into().unwrap()));
                    }
                    gt_records.push(vals);
                    offset += dim * 4;
                }
            }

            let k = if gt_records.is_empty() { 100 } else { gt_records[0].len() };
            let mut pass = 0usize;
            let mut fail = 0usize;

            let pb = ctx.ui.bar(actual_sample as u64, &format!("verify filtered-knn '{}'", name));
            for (si, &qi) in sample_indices.iter().enumerate() {
                if qi >= gt_records.len() || qi >= pred_indices.count() { continue; }

                // Get matching ordinals for this query's predicate
                let matching = match pred_indices.get_ordinals(qi) {
                    Ok(m) => m,
                    Err(_) => { fail += 1; continue; }
                };

                // Brute-force filtered KNN: compute distance from query to each matching base
                let qvec = query_reader.get_slice(qi);
                let mut dists: Vec<(f32, i32)> = Vec::with_capacity(matching.len());
                for &ord in &matching {
                    let ord_u = ord as usize;
                    if ord_u >= bc_val { continue; }
                    let bvec = base_reader.get_slice(ord_u);
                    let d = dist_fn(qvec, bvec);
                    dists.push((d, ord));
                }
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let expected: Vec<i32> = dists.iter().take(k).map(|(_, idx)| *idx).collect();

                // Compare against stored GT
                let stored = &gt_records[qi];
                let is_exact = expected == *stored;
                let recall = if is_exact {
                    1.0
                } else {
                    let stored_set: std::collections::HashSet<i32> = stored.iter().cloned().collect();
                    let expected_set: std::collections::HashSet<i32> = expected.iter().cloned().collect();
                    stored_set.intersection(&expected_set).count() as f64 / k as f64
                };

                // Format top-5 indices for display
                let fmt_top = |v: &[i32], n: usize| -> String {
                    let take: Vec<String> = v.iter().take(n).map(|x| x.to_string()).collect();
                    if v.len() > n { format!("[{}, ...]", take.join(", ")) }
                    else { format!("[{}]", take.join(", ")) }
                };

                if recall >= 0.99 {
                    pass += 1;
                    // Log exemplars: first 3 passing queries show full verification chain
                    if pass <= 3 {
                        let nearest_dist = dists.first().map(|(d, _)| *d).unwrap_or(0.0);
                        let kth_dist = dists.get(k.saturating_sub(1)).map(|(d, _)| *d).unwrap_or(0.0);
                        ctx.ui.log(&format!("  exemplar query {}:", qi));
                        ctx.ui.log(&format!("    predicate R[{}] → {} matching ordinals",
                            qi, matching.len()));
                        ctx.ui.log(&format!("    R ordinals (first 5): {}",
                            fmt_top(&matching, 5)));
                        ctx.ui.log(&format!("    brute-force scanned {} eligible base vectors",
                            dists.len()));
                        ctx.ui.log(&format!("    computed top-{}: nearest={:.6} k-th={:.6}",
                            k, nearest_dist, kth_dist));
                        ctx.ui.log(&format!("    computed indices: {}",
                            fmt_top(&expected, 5)));
                        ctx.ui.log(&format!("    stored   indices: {}",
                            fmt_top(stored, 5)));
                        ctx.ui.log(&format!("    recall={:.4} ✓", recall));
                    }
                } else {
                    fail += 1;
                    if fail <= 5 {
                        let stored_set: std::collections::HashSet<i32> = stored.iter().cloned().collect();
                        let expected_set: std::collections::HashSet<i32> = expected.iter().cloned().collect();
                        let common = stored_set.intersection(&expected_set).count();
                        let in_expected_only = expected_set.difference(&stored_set).count();
                        let in_stored_only = stored_set.difference(&expected_set).count();
                        ctx.ui.log(&format!("  MISMATCH query {}:", qi));
                        ctx.ui.log(&format!("    R[{}] → {} matching ordinals", qi, matching.len()));
                        ctx.ui.log(&format!("    computed indices: {}", fmt_top(&expected, 5)));
                        ctx.ui.log(&format!("    stored   indices: {}", fmt_top(stored, 5)));
                        ctx.ui.log(&format!("    intersection: {} common, {} computed-only, {} stored-only",
                            common, in_expected_only, in_stored_only));
                        ctx.ui.log(&format!("    recall={:.4} ✗", recall));
                    }
                }
                if (si + 1) % 10 == 0 { pb.set_position((si + 1) as u64); }
            }
            pb.finish();

            let bc_str = if *bc == u64::MAX { "full".into() } else { bc.to_string() };
            ctx.ui.log(&format!("  profile '{}' (base_count={}): {}/{} pass, {} fail (k={})",
                name, bc_str, pass, pass + fail, fail, k));
            results.push(serde_json::json!({
                "name": name,
                "status": if fail == 0 { "pass" } else { "fail" },
                "pass": pass,
                "fail": fail,
                "sample": actual_sample,
                "base_count": bc,
                "k": k,
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
// verify predicates-consolidated (stub — no base vector scan needed)
// ═══════════════════════════════════════════════════════════════════════════

pub struct VerifyPredicatesConsolidatedOp;

pub fn predicates_consolidated_factory() -> Box<dyn CommandOp> {
    Box::new(VerifyPredicatesConsolidatedOp)
}

impl CommandOp for VerifyPredicatesConsolidatedOp {
    fn command_path(&self) -> &str { "verify predicates-consolidated" }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }
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

        // Belt-and-suspenders against the faiss-sys static-MKL bug
        // (see pipeline::blas_abi).
        crate::pipeline::blas_abi::set_single_threaded_if_faiss();

        // Up-front: confirm the local dataset has the minimum facets
        // this verify kind requires (see pipeline::dataset_lookup).
        if let Err(e) = crate::pipeline::dataset_lookup::validate_and_log(
            ctx, options, crate::pipeline::dataset_lookup::VerifyKind::PredicatesConsolidated,
        ) {
            return error_result(e, start);
        }

        // Standalone-friendly: input paths come from the dataset's
        // profile/facet view (see pipeline::dataset_lookup).
        let metadata_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "metadata", "metadata_content",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let predicates_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "predicates", "metadata_predicates",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let output_str = match options.require("output") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };

        let metadata_path = resolve_path(&metadata_str, &ctx.workspace);
        let predicates_path = resolve_path(&predicates_str, &ctx.workspace);
        let output_path = resolve_path(&output_str, &ctx.workspace);

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
            let has_indices = ["ivvec", "ivec", "slab"].iter().any(|ext| {
                ctx.workspace.join(format!("profiles/{}/metadata_indices.{}", name, ext)).exists()
            });
            if has_indices {
                profiles.push((name.clone(), bc));
            }
        }
        profiles.sort_by(|(a, _), (b, _)| {
            let a_bc = config.profiles.profile(a).and_then(|p| p.base_count);
            let b_bc = config.profiles.profile(b).and_then(|p| p.base_count);
            vectordata::dataset::profile::profile_sort_by_size(a, a_bc, b, b_bc)
        });

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
            "verify predicates-consolidated: predicate evaluation across {} profile{} via in-memory \
             scan; sample={} of {} predicates × {} metadata records (cap {})",
            profiles.len(),
            if profiles.len() == 1 { "" } else { "s" },
            actual_pred_sample, pred_count, meta_count, metadata_sample,
        ));

        let mut all_results = Vec::new();
        let pb = ctx.ui.bar_with_unit(profiles.len() as u64, "verifying profiles", "profiles");

        for (_pi, (name, bc)) in profiles.iter().enumerate() {
            // Try multiple extensions for metadata indices
            let indices_candidates = ["ivvec", "ivec", "slab"];
            let indices_path = indices_candidates.iter()
                .map(|ext| ctx.workspace.join(format!("profiles/{}/metadata_indices.{}", name, ext)))
                .find(|p| p.exists())
                .unwrap_or_else(|| ctx.workspace.join(format!("profiles/{}/metadata_indices.slab", name)));

            let indices_ext = indices_path.extension().and_then(|e| e.to_str()).unwrap_or("slab");
            let eval_count: usize;
            let _eval_get: Box<dyn Fn(usize) -> Result<Vec<i32>, String>>;

            if indices_ext == "slab" {
                let reader = match slabtastic::SlabReader::open(&indices_path) {
                    Ok(r) => r,
                    Err(e) => {
                        ctx.ui.log(&format!("  profile '{}': failed to open eval results: {}", name, e));
                        all_results.push(serde_json::json!({
                            "name": name, "status": "error", "message": e.to_string(),
                        }));
                        continue;
                    }
                };
                eval_count = reader.total_records() as usize;
                _eval_get = Box::new(move |i: usize| {
                    let data = reader.get(i as i64).map_err(|e| format!("{}", e))?;
                    Ok(data.chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect())
                });
            } else {
                // ivvec or ivec — use IndexedVvecReader
                let reader = match vectordata::io::IndexedVvecReader::<i32>::open_path(&indices_path) {
                    Ok(r) => r,
                    Err(e) => {
                        ctx.ui.log(&format!("  profile '{}': failed to open eval results: {}", name, e));
                        all_results.push(serde_json::json!({
                            "name": name, "status": "error", "message": e.to_string(),
                        }));
                        continue;
                    }
                };
                eval_count = reader.count();
                _eval_get = Box::new(move |i: usize| {
                    reader.get_i32(i).map_err(|e| format!("{}", e))
                });
            };

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
