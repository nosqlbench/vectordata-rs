// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute exact KNN using direct BLAS sgemm.
//!
//! Numpy/knn_utils parity at the kernel level: `cblas_sgemm` is the same
//! routine `numpy.matmul` dispatches to under MKL/OpenBLAS and the same
//! routine FAISS `IndexFlatIP`/`IndexFlatL2` wraps. Results are bit-
//! identical on CPU when compiled against the same BLAS backend as the
//! Python reference.
//!
//! Architecturally this command is a peer of `compute knn-stdarch`: it
//! reuses the shared segment-cache infrastructure ([`super::knn_segment`])
//! for cross-profile reuse and mid-run resumability, and the shared
//! streaming-pread I/O pattern so no base bytes enter process address
//! space during compute. The only difference is the per-chunk distance
//! kernel — sgemm (BLAS matmul) instead of a per-vector SIMD loop.
//!
//! Links dynamically against the system BLAS (`libopenblas-dev` /
//! `libmkl-dev`) — no static compilation, no FAISS dependency.

use std::collections::BinaryHeap;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::io::XvecReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, Status, StreamContext, render_options_table,
};
use super::compute_knn::Neighbor;
use super::knn_segment::{
    CosineMode, Metric, build_cache_path, cache_prefix_for,
    load_segment_cache, merge_segment_into_heaps, resolve_cosine_mode,
    scan_cached_segments, write_segment_cache,
};
use super::source_window::resolve_source;

/// Engine identifier in the segment cache. Distinct from other engines
/// because different backends compute ULP-different f32 values for the
/// same inputs, so their caches are not interchangeable.
const ENGINE_NAME: &str = "knn-blas";

/// I/O chunk size target. Same default as knn-stdarch — 64 MiB of raw
/// fvec bytes per chunk. Actual `vecs_per_chunk` is derived from
/// `entry_size` at execute time and may be further clamped so the
/// sgemm score matrix fits the `SCORE_MATRIX_BUDGET_BYTES` RAM budget.
const STREAM_CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// Score-matrix RAM budget. The score matrix is the dominant memory
/// cost per sgemm call. `vecs_per_chunk` is sized so that
/// `QUERY_SUBBATCH_SIZE × vecs_per_chunk × 4 bytes ≤ budget`, and
/// queries are then processed in sub-batches within each base chunk.
/// Sub-batching lets `vecs_per_chunk` grow (longer sgemm inner work
/// between cache-blocking restarts) while keeping the score matrix
/// bounded.
const SCORE_MATRIX_BUDGET_BYTES: usize = 1024 * 1024 * 1024;

/// Query sub-batch size: number of queries processed per sgemm call.
/// Smaller values shrink the score matrix (lets `vecs_per_chunk`
/// grow). Larger values amortize sgemm startup overhead. 2048 is a
/// sweet spot on modern x86 with AVX-512 / OpenBLAS.
const QUERY_SUBBATCH_SIZE: usize = 2048;

// ─── BLAS FFI ────────────────────────────────────────────────────────────
// Links dynamically against whatever `libblas.so` / `libopenblas.so` /
// `libmkl_rt.so` the system provides. The `-l blas` linker flag is
// handled by `build.rs` / `Cargo.toml` under the `knnutils` feature.
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,     // CblasRowMajor = 101
        transa: i32,    // CblasNoTrans = 111
        transb: i32,    // CblasTrans = 112
        m: i32,         // rows of A (n_query)
        n: i32,         // cols of B^T (n_base)
        k: i32,         // shared dim
        alpha: f32,
        a: *const f32,  // query batch (m × k)
        lda: i32,
        b: *const f32,  // base data (n × k, transposed by flag)
        ldb: i32,
        beta: f32,
        c: *mut f32,    // output scores (m × n)
        ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

// ═══════════════════════════════════════════════════════════════════════
// Streaming + compute helpers
// ═══════════════════════════════════════════════════════════════════════

/// Allocate a heap buffer (as `Vec<u8>`) large enough to hold a chunk's
/// raw fvec bytes.
fn alloc_raw_buf(bytes: usize) -> Vec<u8> {
    vec![0u8; bytes]
}

/// Allocate a heap buffer for dim-packed f32 payload (no headers).
/// `n` is the maximum vector count the chunk will hold; `dim` is the
/// per-vector element count.
fn alloc_packed_buf(n: usize, dim: usize) -> Vec<f32> {
    vec![0.0f32; n * dim]
}

/// `pread` `n_vecs` entries starting at `byte_off` into `raw`, then
/// header-strip into `packed`. `raw` must be ≥ `n_vecs * entry_size`
/// bytes, `packed` must be ≥ `n_vecs * dim` f32s.
///
/// `elem_size` selects the on-disk element width: 4 for f32 (`.fvec`)
/// — fast byte copy — or 2 for f16 (`.mvec`) — convert each pair of
/// bytes to `half::f16` then upcast to `f32` into the packed buffer.
/// The sgemm kernel itself only knows f32, so the upcast happens
/// here at unpack time (cost is negligible vs the sgemm).
fn pread_and_unpack(
    file: &File,
    byte_off: u64,
    n_vecs: usize,
    entry_size: usize,
    dim: usize,
    elem_size: usize,
    raw: &mut [u8],
    packed: &mut [f32],
) -> std::io::Result<()> {
    let n_bytes = n_vecs * entry_size;
    file.read_exact_at(&mut raw[..n_bytes], byte_off)?;
    // Unpack: for each entry, skip the 4-byte dim header, then
    // either copy or convert dim elements into the packed buffer.
    match elem_size {
        4 => {
            // f32 fast path — verbatim byte copy.
            for i in 0..n_vecs {
                let src_off = i * entry_size + 4;
                let dst_off = i * dim;
                let src = &raw[src_off..src_off + dim * 4];
                let dst_bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        packed[dst_off..].as_mut_ptr() as *mut u8,
                        dim * 4,
                    )
                };
                dst_bytes.copy_from_slice(src);
            }
        }
        2 => {
            // f16 path — decode each pair of bytes into half::f16,
            // then upcast to f32. We use the SIMD bulk converter so
            // wide vectors don't bottleneck on per-element decode.
            for i in 0..n_vecs {
                let src_off = i * entry_size + 4;
                let dst_off = i * dim;
                let src = &raw[src_off..src_off + dim * 2];
                // Reinterpret the LE bytes as half::f16 — sound on
                // any host where f16 is laid out as 2 LE bytes
                // (every supported target).
                let f16_slice: &[half::f16] = unsafe {
                    std::slice::from_raw_parts(src.as_ptr() as *const half::f16, dim)
                };
                crate::pipeline::simd_distance::convert_f16_to_f32_bulk(
                    f16_slice,
                    &mut packed[dst_off..dst_off + dim],
                );
            }
        }
        other => return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("pread_and_unpack: unsupported element size {other} (only 4 and 2 supported by the sgemm scan path)"),
        )),
    }
    Ok(())
}

/// Run `cblas_sgemm` for one chunk. Writes `n_query × chunk_n` scores
/// into `scores` (row-major). `use_ip` picks IP (pure matmul) vs L2
/// (|q|² + |b|² − 2·q·b). Caller is responsible for `query_norms_sq`
/// and `base_norms_sq` for the L2 path.
#[allow(clippy::too_many_arguments)]
fn sgemm_chunk(
    queries_packed: &[f32], n_query: usize,
    chunk_packed: &[f32], chunk_n: usize,
    dim: usize,
    use_ip: bool,
    query_norms_sq: Option<&[f32]>,
    base_norms_sq: Option<&[f32]>,
    scores: &mut [f32],
) {
    if use_ip {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                n_query as i32, chunk_n as i32, dim as i32,
                1.0,
                queries_packed.as_ptr(), dim as i32,
                chunk_packed.as_ptr(), dim as i32,
                0.0,
                scores.as_mut_ptr(), chunk_n as i32,
            );
        }
    } else {
        // L2: start with -2·q·b via sgemm, then add |q|² + |b|².
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                n_query as i32, chunk_n as i32, dim as i32,
                -2.0,
                queries_packed.as_ptr(), dim as i32,
                chunk_packed.as_ptr(), dim as i32,
                0.0,
                scores.as_mut_ptr(), chunk_n as i32,
            );
        }
        let q_norms = query_norms_sq.expect("L2 path requires query_norms_sq");
        let b_norms = base_norms_sq.expect("L2 path requires base_norms_sq");
        for qi in 0..n_query {
            let qn = q_norms[qi];
            let row = &mut scores[qi * chunk_n..(qi + 1) * chunk_n];
            for (bi, s) in row.iter_mut().enumerate() {
                *s += qn + b_norms[bi];
            }
        }
    }
}

/// Threshold-aware top-k selection over one query's chunk score row.
///
/// `row` contains `chunk_n` raw sgemm scores (IP: higher is better;
/// L2: `|q-b|²`, smaller is better). `as_distance` converts to the
/// canonical "distance" space where smaller is always better and the
/// max-heap's top is the worst-in-top-k:
///   - IP:     distance = -score
///   - L2:     distance = score
///
/// `initial_threshold` seeds the pruning cutoff from the global
/// per-query heap accumulated so far, so a well-filled heap skips the
/// majority of the chunk via one compare per base.
///
/// `chunk_first_idx` is the absolute base index of `row[0]`, added to
/// the loop counter to tag each `Neighbor` with its file-level index.
fn topk_from_scores_into(
    row: &[f32],
    k: usize,
    use_ip: bool,
    chunk_first_idx: usize,
    initial_threshold: f32,
    heap: &mut BinaryHeap<Neighbor>,
) -> f32 {
    let mut threshold = initial_threshold;
    let push = |index: u32, distance: f32, heap: &mut BinaryHeap<Neighbor>, thr: &mut f32| {
        if distance < *thr {
            heap.push(Neighbor { index, distance });
            if heap.len() > k {
                heap.pop();
                *thr = heap.peek().unwrap().distance;
            } else if heap.len() == k {
                *thr = heap.peek().unwrap().distance;
            }
        }
    };
    if use_ip {
        for (bi, &s) in row.iter().enumerate() {
            let d = -s;
            push((chunk_first_idx + bi) as u32, d, heap, &mut threshold);
        }
    } else {
        for (bi, &s) in row.iter().enumerate() {
            // sgemm produces L2sq directly (after norm adjustment);
            // distance == L2sq (smaller is better).
            push((chunk_first_idx + bi) as u32, s, heap, &mut threshold);
        }
    }
    threshold
}

// ─── Compatibility shims used by verify_dataset_knnutils ──────────────
// These preserve the pre-refactor API for the verifier so it doesn't
// need its own sgemm + top-k path. Both delegate to the new
// `sgemm_chunk` / `topk_from_scores_into` helpers.

/// Compute the score matrix `n_query × n_base` via sgemm, with the
/// "higher score = better" convention (the L2 path negates after the
/// norm adjustment so the caller can run a uniform top-k).
///
/// # Safety
/// Calls `cblas_sgemm` via FFI. Slices must be valid contiguous f32.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn blas_sgemm_scores(
    query_data: &[f32], n_query: usize,
    base_data: &[f32], n_base: usize,
    dim: usize, use_ip: bool,
    scores: &mut [f32],
) {
    let q_norms_sq: Vec<f32>;
    let b_norms_sq: Vec<f32>;
    let (q_n_opt, b_n_opt): (Option<&[f32]>, Option<&[f32]>) = if use_ip {
        (None, None)
    } else {
        q_norms_sq = (0..n_query).map(|qi| {
            let s = &query_data[qi * dim..(qi + 1) * dim];
            s.iter().map(|v| v * v).sum::<f32>()
        }).collect();
        b_norms_sq = (0..n_base).map(|bi| {
            let s = &base_data[bi * dim..(bi + 1) * dim];
            s.iter().map(|v| v * v).sum::<f32>()
        }).collect();
        (Some(q_norms_sq.as_slice()), Some(b_norms_sq.as_slice()))
    };
    sgemm_chunk(
        query_data, n_query,
        base_data, n_base,
        dim, use_ip,
        q_n_opt, b_n_opt,
        scores,
    );
    if !use_ip {
        // Negate L2sq so the caller's top-k can treat all paths
        // uniformly as "higher = better".
        for s in scores.iter_mut() {
            *s = -*s;
        }
    }
}

/// Top-k indices for one row of "higher = better" scores. Sorted
/// ascending by distance (best first). Returns
/// `compute_knn::Neighbor` whose `.distance == -score`.
pub(super) fn topk_indices(scores: &[f32], k: usize) -> Vec<Neighbor> {
    let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
    let _ = topk_from_scores_into(scores, k, /*use_ip=*/true, 0, f32::INFINITY, &mut heap);
    let mut v: Vec<Neighbor> = heap.into_vec();
    v.sort_by(|a, b| {
        a.distance.partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    });
    v
}

// ═══════════════════════════════════════════════════════════════════════
// Shared streaming scan (compute-knn-blas + verify-knn-consolidated)
// ═══════════════════════════════════════════════════════════════════════
//
// Both compute and verify drive the same math: sgemm for distances,
// sub-batched for score-matrix bounding, rayon-parallel top-k, merge
// into caller-owned heaps. By factoring through this function, verify
// produces *bit-identical* distances to compute — no kernel-diversity
// mismatch at the ULP boundary, no false positives on 1B-scale runs.
//
// Caller is responsible for:
//   - Opening `base_file` and knowing `dim` / `entry_size`.
//   - Packing queries contiguously (`queries_packed[qi*dim..qi*dim+dim]`)
//     and precomputing their squared L2 norms.
//   - Allocating reusable buffers via [`SgemmScanBuffers::new`] once at
//     the top of the enclosing execute() and passing by `&mut` so no
//     allocation happens per call.
//   - Seeding `heaps` and `thresholds` (identical length = query_count).
//     Pre-existing entries are preserved; new candidates are merged with
//     threshold-aware pruning.

/// Reusable buffer set for streaming sgemm scans. Allocate once at the
/// top of an `execute()` run and pass the same mut reference into every
/// [`scan_range_sgemm`] call to avoid per-segment allocation.
#[allow(dead_code)]
pub(super) struct SgemmScanBuffers {
    raw_a: Vec<u8>,
    raw_b: Vec<u8>,
    packed_a: Vec<f32>,
    packed_b: Vec<f32>,
    score_matrix: Vec<f32>,
    base_norms_sq: Vec<f32>,
    vecs_per_chunk: usize,
    qb_size: usize,
}

impl SgemmScanBuffers {
    /// Allocate buffers sized for the given shape. Picks `vecs_per_chunk`
    /// so that the score matrix (`qb_size × vecs_per_chunk × 4`) fits
    /// [`SCORE_MATRIX_BUDGET_BYTES`] and the raw chunk (`vecs_per_chunk
    /// × entry_size`) fits [`STREAM_CHUNK_BYTES`].
    pub(super) fn new(query_count: usize, dim: usize) -> Self {
        let entry_size = 4 + dim * 4;
        let qb_size = QUERY_SUBBATCH_SIZE.min(query_count.max(1));
        let vecs_per_chunk_score = (SCORE_MATRIX_BUDGET_BYTES / (qb_size * 4).max(1)).max(1);
        let vecs_per_chunk_io = (STREAM_CHUNK_BYTES / entry_size.max(1)).max(1);
        let vecs_per_chunk = vecs_per_chunk_score.min(vecs_per_chunk_io);
        let raw_chunk_bytes = vecs_per_chunk * entry_size;
        let _packed_chunk_f32 = vecs_per_chunk * dim;
        let score_matrix_len = qb_size * vecs_per_chunk;
        Self {
            raw_a: alloc_raw_buf(raw_chunk_bytes),
            raw_b: alloc_raw_buf(raw_chunk_bytes),
            packed_a: alloc_packed_buf(vecs_per_chunk, dim),
            packed_b: alloc_packed_buf(vecs_per_chunk, dim),
            score_matrix: vec![0.0f32; score_matrix_len],
            base_norms_sq: Vec::new(), // sized lazily
            vecs_per_chunk,
            qb_size,
        }
    }

    pub(super) fn vecs_per_chunk(&self) -> usize { self.vecs_per_chunk }
    pub(super) fn qb_size(&self) -> usize { self.qb_size }
}

/// Scan `base_range` against `queries_packed`, accumulating top-k
/// neighbors into caller-owned `heaps` / `thresholds` with threshold
/// pruning. Uses double-buffered pread streaming + sgemm + sub-batched
/// top-k — the same math as `compute knn-blas`.
///
/// `use_ip` selects the kernel: `true` for IP / DOT / cosine-assume-
/// normalized; `false` for L2 (norm-adjusted sgemm).
/// `proper_cosine` additionally divides scores by `|q| × |b|` after
/// sgemm when computing cosine on non-normalized inputs.
#[allow(clippy::too_many_arguments)]
pub(super) fn scan_range_sgemm(
    base_file: &File,
    base_range: std::ops::Range<usize>,
    dim: usize,
    // On-disk element width of base records: 4 for `.fvec` / `.fvecs`,
    // 2 for `.mvec` / `.mvecs`. The sgemm kernel itself runs in f32;
    // f16 records are upcast at unpack time.
    elem_size: usize,
    queries_packed: &[f32],
    query_count: usize,
    query_norms_sq: &[f32],
    use_ip: bool,
    proper_cosine: bool,
    k: usize,
    buffers: &mut SgemmScanBuffers,
    heaps: &mut [BinaryHeap<Neighbor>],
    thresholds: &mut [f32],
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> Result<(), String> {
    let entry_size = 4 + dim * elem_size;
    let range_start = base_range.start;
    let range_end = base_range.end;
    let range_len = range_end.saturating_sub(range_start);
    if range_len == 0 { return Ok(()); }

    let vecs_per_chunk = buffers.vecs_per_chunk;
    let qb_size = buffers.qb_size;
    let n_chunks = (range_len + vecs_per_chunk - 1) / vecs_per_chunk;

    let need_base_norms = !use_ip || proper_cosine;
    if need_base_norms {
        buffers.base_norms_sq.resize(vecs_per_chunk, 0.0);
    }

    let chunk_bounds = |c: usize| -> (usize, usize) {
        let s = range_start + c * vecs_per_chunk;
        let e = (s + vecs_per_chunk).min(range_end);
        (s, e - s)
    };

    // Destructure buffers so the borrow checker lets us hand `raw_a`/
    // `packed_a` to one scope thread and `raw_b`/`packed_b` to another.
    let SgemmScanBuffers { raw_a, raw_b, packed_a, packed_b, score_matrix, base_norms_sq, .. } = buffers;

    // Prime buf_a with chunk 0.
    let (c0_first, c0_n) = chunk_bounds(0);
    if c0_n == 0 { return Ok(()); }
    let c0_off = (c0_first as u64) * (entry_size as u64);
    pread_and_unpack(base_file, c0_off, c0_n, entry_size, dim, elem_size, raw_a, packed_a)
        .map_err(|e| format!("pread chunk 0 @{}: {}", c0_first, e))?;

    let io_err: Arc<std::sync::Mutex<Option<String>>> =
        Arc::new(std::sync::Mutex::new(None));

    for chunk_i in 0..n_chunks {
        let is_last = chunk_i + 1 == n_chunks;
        let (cur_first, cur_n) = chunk_bounds(chunk_i);
        let (next_first, next_n) = if is_last { (0, 0) } else { chunk_bounds(chunk_i + 1) };
        let next_off = (next_first as u64) * (entry_size as u64);

        std::thread::scope(|scope| {
            // Prefetch next chunk in the idle buffer.
            if !is_last && next_n > 0 {
                let rb: &mut Vec<u8> = raw_b;
                let pb_next: &mut Vec<f32> = packed_b;
                let bf = base_file;
                let err_slot = Arc::clone(&io_err);
                scope.spawn(move || {
                    if let Err(e) = pread_and_unpack(
                        bf, next_off, next_n, entry_size, dim, elem_size, rb, pb_next,
                    ) {
                        *err_slot.lock().unwrap() =
                            Some(format!("pread chunk@{}: {}", next_first, e));
                    }
                });
            }

            // Compute task: sub-batched sgemm + top-k + merge into heaps.
            let packed_cur: &[f32] = &packed_a[..cur_n * dim];
            let q_packed = queries_packed;
            let q_norms_ref = query_norms_sq;
            let heaps_ref: &mut [BinaryHeap<Neighbor>] = heaps;
            let thr_ref: &mut [f32] = thresholds;
            let score_matrix_ref: &mut Vec<f32> = score_matrix;
            let b_norms_ref: Option<&mut Vec<f32>> =
                if need_base_norms { Some(base_norms_sq) } else { None };
            scope.spawn(move || {
                let b_norms_slice: Option<&[f32]> = if need_base_norms {
                    let bn = b_norms_ref.unwrap();
                    for (bi, n) in bn[..cur_n].iter_mut().enumerate() {
                        let s = &packed_cur[bi * dim..(bi + 1) * dim];
                        *n = s.iter().map(|v| v * v).sum::<f32>();
                    }
                    Some(&bn[..cur_n])
                } else {
                    None
                };
                let b_norms_for_sgemm = if use_ip { None } else { b_norms_slice };

                for qb_start in (0..query_count).step_by(qb_size) {
                    let qb_end = (qb_start + qb_size).min(query_count);
                    let qb_n = qb_end - qb_start;

                    let queries_slice = &q_packed[qb_start * dim..qb_end * dim];
                    let q_norms_slice = &q_norms_ref[qb_start..qb_end];
                    let scores_slice = &mut score_matrix_ref[..qb_n * cur_n];

                    sgemm_chunk(
                        queries_slice, qb_n,
                        packed_cur, cur_n,
                        dim,
                        use_ip,
                        if use_ip { None } else { Some(q_norms_slice) },
                        b_norms_for_sgemm,
                        scores_slice,
                    );

                    if proper_cosine {
                        let bn = b_norms_slice.unwrap();
                        for qi in 0..qb_n {
                            let q_mag = q_norms_slice[qi].sqrt().max(f32::MIN_POSITIVE);
                            let row = &mut scores_slice[qi * cur_n..(qi + 1) * cur_n];
                            for (bi, s) in row.iter_mut().enumerate() {
                                let b_mag = bn[bi].sqrt().max(f32::MIN_POSITIVE);
                                *s /= q_mag * b_mag;
                            }
                        }
                    }

                    use rayon::prelude::*;
                    let row_results: Vec<BinaryHeap<Neighbor>> = (0..qb_n)
                        .into_par_iter()
                        .map(|qi_local| {
                            let qi_global = qb_start + qi_local;
                            let row = &scores_slice[qi_local * cur_n..(qi_local + 1) * cur_n];
                            let mut heap: BinaryHeap<Neighbor> =
                                BinaryHeap::with_capacity(k + 1);
                            let _ = topk_from_scores_into(
                                row, k, use_ip, cur_first,
                                thr_ref[qi_global], &mut heap,
                            );
                            heap
                        })
                        .collect();
                    for (qi_local, cand_heap) in row_results.into_iter().enumerate() {
                        let qi_global = qb_start + qi_local;
                        for n in cand_heap.into_iter() {
                            if n.distance < thr_ref[qi_global] {
                                heaps_ref[qi_global].push(n);
                                if heaps_ref[qi_global].len() > k {
                                    heaps_ref[qi_global].pop();
                                    thr_ref[qi_global] = heaps_ref[qi_global].peek().unwrap().distance;
                                } else if heaps_ref[qi_global].len() == k {
                                    thr_ref[qi_global] = heaps_ref[qi_global].peek().unwrap().distance;
                                }
                            }
                        }
                    }
                }
            });
        });

        if let Some(msg) = io_err.lock().unwrap().take() {
            return Err(msg);
        }

        if !is_last {
            std::mem::swap(raw_a, raw_b);
            std::mem::swap(packed_a, packed_b);
        }
        if let Some(ref mut cb) = progress {
            cb((cur_first + cur_n - range_start) as u64);
        }
    }
    Ok(())
}

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: msg.into(),
        elapsed: start.elapsed(),
        produced: vec![],
    }
}

fn resolve_path(s: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::Path::new(s);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

pub struct ComputeKnnBlasOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeKnnBlasOp)
}

impl CommandOp for ComputeKnnBlasOp {
    fn command_path(&self) -> &str {
        "compute knn-blas"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_SECONDARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Brute-force exact KNN via BLAS sgemm (knn_utils compatible)".into(),
            body: format!(r#"# compute knn-blas

Brute-force exact K-nearest-neighbor ground truth computation using
direct BLAS `cblas_sgemm` for the distance matrix, with Rust top-K
heap selection.

## Numerical parity

Produces bit-identical CPU results to:
  - `numpy.matmul` / `np.linalg.norm`-based KNN (both dispatch to the
    same `cblas_sgemm` routine under MKL/OpenBLAS).
  - FAISS `IndexFlatIP` / `IndexFlatL2` when both link the same BLAS
    backend.

## Architecture

Reuses the shared segment-cache infrastructure:
  - Cross-profile cache reuse — smaller sibling profiles' published
    outputs drop in as `[0..N)` segments.
  - Mid-run resumability — each segment's per-query top-K is cached
    in `<workspace>/.cache/knn-blas.v3.*` files.
  - Streaming pread — base vectors never enter the mmap address
    space during compute; I/O is double-buffered against sgemm
    compute.

## Options

{}
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<crate::pipeline::command::ResourceDesc> {
        vec![
            crate::pipeline::command::ResourceDesc {
                name: "mem".into(),
                description: "Score matrix + chunk buffers per segment".into(),
                adjustable: true,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // faiss-sys static MKL is poisoned process-wide when the
        // `faiss` feature is on (see pipeline::blas_abi). sgemm goes
        // wrong under multi-threaded MKL at dim≥384. Force single-
        // threaded before any sgemm call.
        crate::pipeline::blas_abi::set_single_threaded_if_faiss();

        let base_str = match options.require("base") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("indices") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let k: usize = match options.require("neighbors") {
            Ok(s) => match s.parse() {
                Ok(n) if n > 0 => n,
                _ => return error_result(format!("invalid neighbors: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };
        let metric_str = options.get("metric").unwrap_or("L2");
        let metric = match Metric::from_str(metric_str) {
            Some(m) => m,
            None => return error_result(format!("unsupported metric: '{}'", metric_str), start),
        };
        let cosine_mode = match resolve_cosine_mode(metric, options) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };
        // Kernel-metric selection:
        //   L2                       → L2 kernel (sgemm + norm adjust)
        //   DOT_PRODUCT              → IP kernel (raw sgemm)
        //   COSINE + AssumeNormalized→ IP kernel (pre-normalized → dot is cos_sim)
        //   COSINE + ProperMetric    → IP kernel + post-divide by |q| × |b|
        let kernel_metric = match metric {
            Metric::L2 => Metric::L2,
            Metric::DotProduct => Metric::DotProduct,
            Metric::Cosine => match cosine_mode.unwrap() {
                // Collapse to DotProduct so cache keys share with a
                // direct DOT run on the same normalized data — this
                // is exactly what knn_utils does via faiss.
                CosineMode::AssumeNormalized => Metric::DotProduct,
                // Keep Cosine so cache keys stay distinct from a
                // DOT run (distances differ: cos_sim vs raw dot).
                CosineMode::ProperMetric => Metric::Cosine,
            },
        };
        let use_ip = matches!(kernel_metric, Metric::DotProduct | Metric::Cosine);

        let base_source = match resolve_source(base_str, &ctx.workspace) {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let base_path = base_source.path.clone();
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
        let distances_path = options
            .get("distances")
            .map(|s| resolve_path(s, &ctx.workspace));

        for path in std::iter::once(&indices_path).chain(distances_path.iter()) {
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
        }

        // Metadata via mmap; compute uses pread (no base bytes in RSS).
        let base_reader = match XvecReader::<f32>::open_path(&base_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open base: {}", e), start),
        };
        let query_reader = match XvecReader::<f32>::open_path(&query_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open query: {}", e), start),
        };
        let base_file = match File::open(&base_path) {
            Ok(f) => Arc::new(f),
            Err(e) => return error_result(format!("open base for streaming: {}", e), start),
        };

        let base_count = <XvecReader<f32> as VectorReader<f32>>::count(&base_reader);
        let query_count = <XvecReader<f32> as VectorReader<f32>>::count(&query_reader);
        let base_dim = <XvecReader<f32> as VectorReader<f32>>::dim(&base_reader);
        let query_dim = <XvecReader<f32> as VectorReader<f32>>::dim(&query_reader);

        if base_dim != query_dim {
            return error_result(
                format!("dimension mismatch: base={} query={}", base_dim, query_dim),
                start,
            );
        }
        let dim = base_dim;
        let entry_size = 4 + dim * 4;

        // Resolve the effective base window from BOTH the inline path
        // syntax (`base.fvec[0..N]`) and the injected `range` option
        // that sized-profile expansion puts on every per-profile
        // step. Sized profiles inject `range: "[0,base_end)"`; without
        // reading that option here, every sized profile would silently
        // scan the full base.
        let effective_window = super::source_window::resolve_window(
            base_source.window, options.get("range"),
        );
        let (base_offset, base_n) = match effective_window {
            Some((ws, we)) => {
                let s = ws.min(base_count);
                (s, we.min(base_count).saturating_sub(s))
            }
            None => (0, base_count),
        };
        let base_start = base_offset;
        let base_end = base_offset + base_n;

        // Margin is opt-in: default 0 ⇒ `internal_k == k` ⇒ original
        // heap sizes and pruning aggressiveness, no slowdown.
        // `verify engine-parity` sets `rerank_margin_ratio=3` to
        // recover the last 1% boundary cases on pathological
        // synthetic distributions.
        let margin = super::knn_segment::rerank_margin_from(options, k);
        let internal_k = super::knn_segment::internal_k(k, margin).min(base_n);

        ctx.ui.log(&format!(
            "KNN-blas: {} queries × {} base, dim={}, k={} (internal {}), metric={} (engine=BLAS sgemm)",
            query_count, base_n, dim, k, internal_k, metric_str,
        ));

        // ── Segment planning (shared with knn-stdarch) ──────────────
        let segment_size = {
            let explicit: Option<usize> = options.get("partition_size")
                .and_then(|s| s.parse().ok())
                .filter(|&n: &usize| n > 0);
            if let Some(n) = explicit {
                n.min(base_n)
            } else {
                // Target: ~50% of system RAM for the segment. This is a
                // high-water mark the greedy cache matcher will stop at;
                // individual chunks within the segment are sized much
                // smaller by the score-matrix budget.
                let total_ram: u64 = std::fs::read_to_string("/proc/meminfo").ok()
                    .and_then(|s| s.lines()
                        .find(|l| l.starts_with("MemTotal:"))
                        .and_then(|l| l.split_whitespace().nth(1))
                        .and_then(|n| n.parse::<u64>().ok())
                        .map(|kb| kb * 1024))
                    .unwrap_or(8 * 1024 * 1024 * 1024);
                let budget = (total_ram / 2) as usize;
                let entry_bytes = dim * 4;
                let auto = (budget / entry_bytes.max(1)).max(10_000);
                auto.min(base_n)
            }
        };
        let nominal_segments = (base_n + segment_size - 1) / segment_size;
        ctx.ui.log(&format!(
            "  {} base-vector segments @ {} vectors each (nominal, before cache reuse)",
            nominal_segments, segment_size,
        ));

        if !ctx.cache.exists() {
            let _ = std::fs::create_dir_all(&ctx.cache);
        }
        let cache_prefix = cache_prefix_for(&base_path, &query_path);
        let cached_segments = scan_cached_segments(
            &ctx.cache, ENGINE_NAME, &cache_prefix, internal_k, kernel_metric, query_count,
            &ctx.workspace, base_end, &ctx.ui,
        );

        struct PlannedSegment {
            start: usize,
            end: usize,
            ivec_path: PathBuf,
            fvec_path: PathBuf,
            cached: bool,
        }

        let mut plan: Vec<PlannedSegment> = Vec::new();
        let mut pos = base_start;
        while pos < base_end {
            let best = cached_segments.iter()
                .filter(|s| s.start == pos && s.end <= base_end && s.end > pos)
                .max_by_key(|s| s.end - s.start);
            if let Some(seg) = best {
                plan.push(PlannedSegment {
                    start: seg.start, end: seg.end,
                    ivec_path: seg.ivec_path.clone(),
                    fvec_path: seg.fvec_path.clone(),
                    cached: true,
                });
                pos = seg.end;
            } else {
                let pe = (pos + segment_size).min(base_end);
                plan.push(PlannedSegment {
                    start: pos, end: pe,
                    ivec_path: build_cache_path(&ctx.cache, ENGINE_NAME, &cache_prefix, pos, pe, internal_k, kernel_metric, "neighbors", "ivec"),
                    fvec_path: build_cache_path(&ctx.cache, ENGINE_NAME, &cache_prefix, pos, pe, internal_k, kernel_metric, "distances", "fvec"),
                    cached: false,
                });
                pos = pe;
            }
        }

        let n_segments = plan.len();
        let cached_count = plan.iter().filter(|p| p.cached).count();
        let to_compute = n_segments - cached_count;
        let cached_bases: usize = plan.iter()
            .filter(|p| p.cached)
            .map(|p| p.end - p.start)
            .sum();

        // Global per-query heaps, persisted across segments. Sized at
        // internal_k so margin candidates survive into the rerank.
        let mut all_heaps: Vec<BinaryHeap<Neighbor>> = (0..query_count)
            .map(|_| BinaryHeap::with_capacity(internal_k + 1))
            .collect();
        let mut all_thresholds: Vec<f32> = vec![f32::INFINITY; query_count];

        // Progress bar sized to the COMPUTE-ONLY workload (freshly
        // computed base vectors, not cache-replayed ones). Cached
        // segment replay advances nothing and inflates neither the
        // rate nor the ETA — replay is reported separately via its
        // own log line. Tracking only real compute gives a faithful
        // "14.5M rec/s" = actual sgemm throughput, and "eta Ns" = time
        // left on real work.
        let compute_bases: usize = plan.iter()
            .filter(|p| !p.cached)
            .map(|p| p.end - p.start)
            .sum();
        let pb = ctx.ui.bar(compute_bases as u64, "KNN-blas");
        let mut compute_progress: u64 = 0;

        if cached_count > 0 {
            let base_pct = 100.0 * cached_bases as f64 / base_n as f64;
            ctx.ui.log(&format!(
                "  ┌─── segment cache: {}/{} segments reusable ({} of {} base vectors, {:.0}%) ───",
                cached_count, n_segments, cached_bases, base_n, base_pct,
            ));
            ctx.ui.log(&format!(
                "  │  will REPLAY {} segment(s) from disk, COMPUTE {} fresh",
                cached_count, to_compute,
            ));
            ctx.ui.log(&format!("  │  cache dir: {}", ctx.cache.display()));
            ctx.ui.log("  └──────────────────────────────────────────────────────");

            let replay_start = Instant::now();
            let mut replayed_bytes: u64 = 0;
            for (seg_idx, p) in plan.iter().enumerate() {
                if !p.cached { continue; }
                let seg = match load_segment_cache(&p.ivec_path, &p.fvec_path, internal_k, query_count, kernel_metric) {
                    Ok(s) => s,
                    Err(e) => {
                        return error_result(
                            format!("failed to replay segment {} [{}..{}): {}", seg_idx, p.start, p.end, e),
                            start,
                        );
                    }
                };
                merge_segment_into_heaps(&seg, &mut all_heaps, &mut all_thresholds, internal_k);
                let ivec_sz = std::fs::metadata(&p.ivec_path).map(|m| m.len()).unwrap_or(0);
                let fvec_sz = std::fs::metadata(&p.fvec_path).map(|m| m.len()).unwrap_or(0);
                replayed_bytes += ivec_sz + fvec_sz;
                ctx.ui.log(&format!(
                    "  ▸ [seg {:>3}/{}] REUSED range [{}..{}) from {}",
                    seg_idx + 1, n_segments, p.start, p.end,
                    p.ivec_path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                ));
                // Intentionally not advancing the progress bar here —
                // replay isn't work, and counting it inflates rec/s
                // and collapses the ETA.
                let _ = base_start;
            }
            let replay_secs = replay_start.elapsed().as_secs_f64();
            ctx.ui.log(&format!(
                "  ✓ replay complete: {} segment(s), {:.1} MiB read from cache in {:.1}s",
                cached_count,
                replayed_bytes as f64 / (1024.0 * 1024.0),
                replay_secs,
            ));
        } else {
            ctx.ui.log(&format!(
                "  segment cache: 0 reusable — {} segment(s) will be computed fresh (cache will be populated for future profiles)",
                n_segments,
            ));
        }

        // ── Queries: load once, packed contiguously ────────────────
        // Queries are small (typically 10K × dim × 4 = tens of MB) and
        // touched once per segment; always-resident is fine.
        let mut queries_packed: Vec<f32> = Vec::with_capacity(query_count * dim);
        for i in 0..query_count {
            queries_packed.extend_from_slice(query_reader.get_slice(i));
        }
        // Query norms² for the L2 path; cheap to keep around even if
        // unused by IP/DOT.
        let query_norms_sq: Vec<f32> = (0..query_count).map(|qi| {
            let s = &queries_packed[qi * dim..(qi + 1) * dim];
            s.iter().map(|v| v * v).sum::<f32>()
        }).collect();

        // Allocate the shared streaming-scan buffers ONCE for the full
        // KNN run. `SgemmScanBuffers::new` picks the same chunk +
        // sub-batch shape that the in-line code did before. The same
        // `&mut SgemmScanBuffers` is passed to every per-segment
        // `scan_range_sgemm` call below — no per-segment allocation.
        let mut scan_buffers = SgemmScanBuffers::new(query_count, dim);
        let vecs_per_chunk = scan_buffers.vecs_per_chunk();
        let qb_size = scan_buffers.qb_size();
        let proper_cosine = matches!(cosine_mode, Some(CosineMode::ProperMetric));
        let n_query_subbatches = (query_count + qb_size - 1) / qb_size;
        ctx.ui.log(&format!(
            "  chunk plan: {} base/chunk × {} queries/sub-batch ({} sub-batches), \
             raw={} MiB, packed={} MiB, score={} MiB",
            vecs_per_chunk, qb_size, n_query_subbatches,
            (vecs_per_chunk * entry_size) / (1024 * 1024),
            (vecs_per_chunk * dim * 4) / (1024 * 1024),
            (qb_size * vecs_per_chunk * 4) / (1024 * 1024),
        ));

        let mut computed_count = 0usize;
        for seg_idx in 0..n_segments {
            let p = &plan[seg_idx];
            if p.cached { continue; }

            let seg_start = p.start;
            let seg_end = p.end;
            let seg_len = seg_end - seg_start;
            let n_chunks = (seg_len + vecs_per_chunk - 1) / vecs_per_chunk;
            let seg_compute_start = Instant::now();
            ctx.ui.log(&format!(
                "  ▶ [seg {:>3}/{}] COMPUTE range [{}..{}) ({} base × {} queries, {} chunk(s) ≤ {} vecs)",
                seg_idx + 1, n_segments,
                seg_start, seg_end,
                seg_len, query_count,
                n_chunks, vecs_per_chunk,
            ));

            // Per-segment heaps — seeded from global thresholds so
            // early pruning kicks in from the first chunk. Segment
            // output is the delta this segment contributes; we merge
            // into global heaps at segment end.
            let mut seg_heaps: Vec<BinaryHeap<Neighbor>> = (0..query_count)
                .map(|_| BinaryHeap::with_capacity(internal_k + 1))
                .collect();
            let mut seg_thresholds: Vec<f32> = all_thresholds.clone();

            // Stream the segment via the shared scan helper. Identical
            // math as the in-line code that lived here before — now
            // factored so verify-knn can call the same kernel and get
            // bit-identical distances. Progress callback advances the
            // global compute_progress counter as each chunk completes.
            let pb_for_cb = &pb;
            let prev_compute_progress = compute_progress;
            let mut tick_cb = move |done_in_segment: u64| {
                pb_for_cb.set_position(prev_compute_progress + done_in_segment);
            };

            if let Err(e) = scan_range_sgemm(
                &base_file,
                seg_start..seg_end,
                dim,
                // compute knn-blas hardcodes f32 base for now —
                // verify-knn-consolidated dispatches on the
                // file's element type, but this path doesn't yet.
                // When f16 base support is added here, replace 4
                // with the dispatched elem_size.
                4,
                &queries_packed,
                query_count,
                &query_norms_sq,
                use_ip,
                proper_cosine,
                internal_k,
                &mut scan_buffers,
                &mut seg_heaps,
                &mut seg_thresholds,
                Some(&mut tick_cb),
            ) {
                return error_result(format!("seg {}: {}", seg_idx, e), start);
            }
            compute_progress = compute_progress.saturating_add(seg_len as u64);

            // Collect per-segment heaps into absolute-indexed per_query
            // for cache write and global merge.
            let mut per_query: Vec<Vec<Neighbor>> = Vec::with_capacity(query_count);
            for heap in seg_heaps.into_iter() {
                per_query.push(heap.into_vec());
            }

            if let Err(e) = write_segment_cache(
                &plan[seg_idx].ivec_path,
                &plan[seg_idx].fvec_path,
                &per_query, internal_k, kernel_metric,
            ) {
                ctx.ui.log(&format!(
                    "  warning: segment {} cache write failed: {}",
                    seg_idx, e,
                ));
            }

            merge_segment_into_heaps(&per_query, &mut all_heaps, &mut all_thresholds, internal_k);

            computed_count += 1;
            let seg_secs = seg_compute_start.elapsed().as_secs_f64();
            let seg_evals = seg_len as f64 * query_count as f64;
            let seg_rate = if seg_secs > 0.0 { seg_evals / seg_secs / 1e9 } else { 0.0 };
            ctx.ui.log(&format!(
                "  ✓ [seg {:>3}/{}] done in {:.1}s ({:.2}B dist/s) — cache written",
                seg_idx + 1, n_segments, seg_secs, seg_rate,
            ));
        }
        pb.finish();

        if cached_count > 0 {
            ctx.ui.log(&format!(
                "  segments: {} reused from cache + {} computed = {} total (saved ~{:.0}% of base scans)",
                cached_count, computed_count, n_segments,
                100.0 * cached_count as f64 / n_segments as f64,
            ));
        } else {
            ctx.ui.log(&format!(
                "  segments: {} computed (cache populated for future runs)",
                computed_count,
            ));
        }

        let compute_secs = start.elapsed().as_secs_f64();
        let evals = query_count as f64 * base_n as f64;
        ctx.ui.log(&format!(
            "  compute: {:.2}s ({:.1}B dist/s)",
            compute_secs, evals / compute_secs / 1e9,
        ));

        // ── Final output ───────────────────────────────────────────
        // Convert heaps to distance-sorted Neighbor vectors.
        let all_results: Vec<Vec<Neighbor>> = all_heaps.into_iter().map(|heap| {
            let mut v: Vec<Neighbor> = heap.into_vec();
            v.sort_by(|a, b| {
                a.distance.partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.index.cmp(&b.index))
            });
            v
        }).collect();

        let mut idx_file = match std::fs::File::create(&indices_path) {
            Ok(f) => std::io::BufWriter::with_capacity(1 << 20, f),
            Err(e) => return error_result(format!("create {}: {}", indices_path.display(), e), start),
        };
        let mut dist_file = match &distances_path {
            Some(dp) => match std::fs::File::create(dp) {
                Ok(f) => Some(std::io::BufWriter::with_capacity(1 << 20, f)),
                Err(e) => return error_result(format!("create {}: {}", dp.display(), e), start),
            },
            None => None,
        };
        // Write top-internal_k per query; the canonical f64-direct
        // rerank below collapses each row to top-k.
        let dim_bytes = (internal_k as i32).to_le_bytes();
        for row in &all_results {
            idx_file.write_all(&dim_bytes).map_err(|e| e.to_string()).unwrap();
            for j in 0..internal_k {
                let idx: i32 = if j < row.len() { row[j].index as i32 } else { -1 };
                idx_file.write_all(&idx.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
            }
            if let Some(ref mut dw) = dist_file {
                dw.write_all(&dim_bytes).map_err(|e| e.to_string()).unwrap();
                for j in 0..internal_k {
                    // Single sign convention everywhere on disk:
                    // FAISS publication (`+L2sq`, `+dot`, `+cos_sim`).
                    // The heap holds kernel-internal distances; convert
                    // via the shared helper so segment caches, cached
                    // sibling profiles, and final output all agree.
                    let dist: f32 = if j < row.len() {
                        super::knn_segment::kernel_to_published(row[j].distance, kernel_metric)
                    } else {
                        f32::INFINITY
                    };
                    dw.write_all(&dist.to_le_bytes()).map_err(|e| e.to_string()).unwrap();
                }
            }
        }
        let _ = use_ip;
        idx_file.flush().map_err(|e| e.to_string()).unwrap();
        if let Some(ref mut dw) = dist_file {
            dw.flush().map_err(|e| e.to_string()).unwrap();
        }
        drop(idx_file);
        drop(dist_file);

        let mut produced = vec![indices_path.clone()];
        if let Some(ref dp) = distances_path {
            produced.push(dp.clone());
        }

        // Canonical f64-direct rerank — same post-pass every other
        // KNN engine runs at execute end, so on-disk output across
        // all engines is identical (canonical) for the same fixture.
        if let Err(e) = super::knn_segment::rerank_output_post_pass(
            indices_path.as_path(),
            distances_path.as_deref(),
            &base_reader,
            &query_reader,
            kernel_metric, k, base_offset,
            &ctx.ui,
        ) {
            ctx.ui.log(&format!("  rerank post-pass skipped: {}", e));
        }

        let elapsed = start.elapsed();
        ctx.ui.log(&format!(
            "  BLAS KNN complete: {} queries × {} neighbors in {:.1}s",
            query_count, k, elapsed.as_secs_f64(),
        ));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} queries, k={}, {} base vectors, metric={}, engine=BLAS-sgemm",
                query_count, k, base_n, metric_str,
            ),
            elapsed,
            produced,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "base".into(), type_name: "Path".into(), required: true, default: None,
                description: "Base vectors file (fvec only)".into(), role: OptionRole::Input },
            OptionDesc { name: "query".into(), type_name: "Path".into(), required: true, default: None,
                description: "Query vectors file (fvec only)".into(), role: OptionRole::Input },
            OptionDesc { name: "indices".into(), type_name: "Path".into(), required: true, default: None,
                description: "Output neighbor indices (ivec)".into(), role: OptionRole::Output },
            OptionDesc { name: "distances".into(), type_name: "Path".into(), required: false, default: None,
                description: "Output neighbor distances (fvec)".into(), role: OptionRole::Output },
            OptionDesc { name: "neighbors".into(), type_name: "int".into(), required: true, default: None,
                description: "Number of nearest neighbors (k)".into(), role: OptionRole::Config },
            OptionDesc { name: "metric".into(), type_name: "enum".into(), required: false,
                default: Some("L2".into()),
                description: "Distance metric: L2, DOT_PRODUCT, COSINE, IP".into(), role: OptionRole::Config },
            OptionDesc { name: "assume_normalized_like_faiss".into(), type_name: "bool".into(), required: false,
                default: Some("false".into()),
                description: "For COSINE metric: treat inputs as pre-normalized and evaluate cosine as inner product (FAISS / numpy / knn_utils convention). Exactly one of this and use_proper_cosine_metric must be set when metric=COSINE.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "use_proper_cosine_metric".into(), type_name: "bool".into(), required: false,
                default: Some("false".into()),
                description: "For COSINE metric: compute cosine in-kernel as dot / (|q| × |b|). Correct for arbitrary inputs; costs extra norm work. Exactly one of this and assume_normalized_like_faiss must be set when metric=COSINE.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "normalized".into(), type_name: "bool".into(), required: false,
                default: Some("false".into()),
                description: "Deprecated alias for assume_normalized_like_faiss; kept for back-compat with pre-existing dataset.yaml files.".into(),
                role: OptionRole::Config },
            OptionDesc { name: "partition_size".into(), type_name: "int".into(), required: false,
                default: None,
                description: "Override auto-sized segment length (for testing or finer cache granularity)".into(),
                role: OptionRole::Config },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query"],
            &["indices", "distances"],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::{Options, Status, StreamContext};
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use vectordata::VectorReader;
    use vectordata::io::XvecReader;

    fn make_ctx(workspace: &std::path::Path) -> StreamContext {
        let cache = workspace.join(".cache");
        std::fs::create_dir_all(&cache).unwrap();
        StreamContext {
            dataset_name: String::new(), profile: String::new(), profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache,
            defaults: IndexMap::new(), dry_run: false, progress: ProgressLog::new(),
            threads: 1, step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1), estimated_total_steps: 0,
        }
    }

    fn write_fvec(path: &std::path::Path, vectors: &[Vec<f32>]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            f.write_all(&(v.len() as i32).to_le_bytes()).unwrap();
            for &val in v { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    fn read_ivec_rows(path: &std::path::Path) -> Vec<Vec<i32>> {
        let reader = XvecReader::<i32>::open_path(path).unwrap();
        let count = <XvecReader<i32> as VectorReader<i32>>::count(&reader);
        (0..count).map(|i| reader.get_slice(i).to_vec()).collect()
    }

    #[test]
    fn test_knn_blas_ip_known_neighbors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![
            vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.7, 0.7, 0.0],
            vec![0.0, 0.0, 1.0], vec![0.5, 0.5, 0.5],
        ];
        let query = vec![vec![0.9, 0.1, 0.0]];
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("idx.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "3");
        opts.set("metric", "IP");

        let mut op = ComputeKnnBlasOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&tmp.path().join("idx.ivec"));
        // IP: base[0]=0.9, base[2]=0.7, base[4]=0.5
        assert_eq!(rows[0][0], 0);
        assert_eq!(rows[0][1], 2);
        assert_eq!(rows[0][2], 4);
    }

    #[test]
    fn test_knn_blas_l2() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let query = vec![vec![0.1, 0.1]];
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("idx.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "2");
        opts.set("metric", "L2");

        let mut op = ComputeKnnBlasOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&tmp.path().join("idx.ivec"));
        assert_eq!(rows[0][0], 0, "L2 nearest to (0.1,0.1) should be (0,0)");
    }

    #[test]
    fn test_knn_blas_multiple_queries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let base = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let query = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("idx.ivec").to_string_lossy().to_string());
        opts.set("neighbors", "1");
        opts.set("metric", "IP");

        let mut op = ComputeKnnBlasOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok);

        let rows = read_ivec_rows(&tmp.path().join("idx.ivec"));
        assert_eq!(rows[0][0], 0);
        assert_eq!(rows[1][0], 1);
        assert_eq!(rows[2][0], 2);
    }

    /// A second run with every segment cached must produce identical
    /// output without re-running sgemm. Mirrors the stdarch replay
    /// regression test.
    #[test]
    fn blas_replay_identical_with_full_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        // Deterministic inputs.
        let base: Vec<Vec<f32>> = (0..40)
            .map(|i| (0..16).map(|j| ((i * 16 + j) as f32) * 0.01).collect())
            .collect();
        let query: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..16).map(|j| ((i * 3 + j) as f32) * 0.02).collect())
            .collect();
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        let mut opts = Options::new();
        opts.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts.set("indices", tmp.path().join("idx.ivec").to_string_lossy().to_string());
        opts.set("distances", tmp.path().join("dist.fvec").to_string_lossy().to_string());
        opts.set("neighbors", "5");
        opts.set("metric", "L2");
        opts.set("partition_size", "10"); // force 4 segments

        let mut op = ComputeKnnBlasOp;
        let r1 = op.execute(&opts, &mut ctx);
        assert_eq!(r1.status, Status::Ok);
        let rows1 = read_ivec_rows(&tmp.path().join("idx.ivec"));

        // Second run: should replay all cached segments.
        let r2 = op.execute(&opts, &mut ctx);
        assert_eq!(r2.status, Status::Ok);
        let rows2 = read_ivec_rows(&tmp.path().join("idx.ivec"));

        assert_eq!(rows1, rows2, "replay from cache must produce identical indices");
    }

    /// Cross-profile reuse: a larger profile (base[0..60)) should
    /// replay the smaller profile's (base[0..30)) cached segments
    /// instead of recomputing them. Verified via mtime preservation
    /// (reused cache files keep their original mtime; recomputed
    /// files get a fresh one). Plus a baseline-equality check so we
    /// know reuse doesn't distort the answers.
    #[test]
    fn blas_larger_profile_reuses_smaller_profile_segments() {
        let tmp = tempfile::tempdir().unwrap();

        // xorshift PRNG → enough variety that A's [0..30) and
        // B's [0..60) genuinely differ in nearest-neighbor selection.
        let mut rng = 123u64;
        let mut nxt = || -> f32 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let base: Vec<Vec<f32>> = (0..60).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..3).map(|_| (0..4).map(|_| nxt()).collect()).collect();
        write_fvec(&tmp.path().join("b.fvec"), &base);
        write_fvec(&tmp.path().join("q.fvec"), &query);

        let partition_size = 10usize;
        let k = 2usize;
        let cache_dir = tmp.path().join(".cache");

        // ── Profile A: base[0..30) → 3 cached segments
        {
            let mut ctx = make_ctx(tmp.path());
            let mut opts = Options::new();
            opts.set("base", "b.fvec[0..30)");
            opts.set("query", "q.fvec");
            opts.set("indices", "a.ivec");
            opts.set("distances", "a-d.fvec");
            opts.set("neighbors", &k.to_string());
            opts.set("metric", "L2");
            opts.set("partition_size", &partition_size.to_string());
            let mut op = ComputeKnnBlasOp;
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(r.status, Status::Ok, "profile A failed: {}", r.message);
        }

        let files_after_a: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-blas.v3.")).unwrap_or(false))
            .collect();
        assert_eq!(files_after_a.len(), 6,
            "profile A should have written 6 cache files (3 segments × ivec+fvec), got {}",
            files_after_a.len());

        // Force a detectable mtime gap so we can prove A's files
        // weren't rewritten by B.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let a_mtimes: std::collections::HashMap<std::path::PathBuf, std::time::SystemTime> =
            files_after_a.iter().map(|p| {
                (p.clone(), std::fs::metadata(p).unwrap().modified().unwrap())
            }).collect();

        // ── Profile B: base[0..60). Plan should be:
        //    - [0..10), [10..20), [20..30): REUSED from A
        //    - [30..40), [40..50), [50..60): computed fresh
        {
            let mut ctx = make_ctx(tmp.path());
            let mut opts = Options::new();
            opts.set("base", "b.fvec[0..60)");
            opts.set("query", "q.fvec");
            opts.set("indices", "b.ivec");
            opts.set("distances", "b-d.fvec");
            opts.set("neighbors", &k.to_string());
            opts.set("metric", "L2");
            opts.set("partition_size", &partition_size.to_string());
            let mut op = ComputeKnnBlasOp;
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(r.status, Status::Ok, "profile B failed: {}", r.message);
        }

        // Reuse proof #1: A's mtime preserved.
        for (path, orig_mtime) in &a_mtimes {
            let new_mtime = std::fs::metadata(path).unwrap().modified().unwrap();
            assert_eq!(*orig_mtime, new_mtime,
                "profile A's cache file {} was rewritten (should have been reused)",
                path.display());
        }

        // Reuse proof #2: B added 3 more segments × 2 files = 6 new
        // files; total = 12.
        let files_after_b: Vec<_> = std::fs::read_dir(&cache_dir).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.file_name().and_then(|s| s.to_str())
                .map(|n| n.starts_with("knn-blas.v3.")).unwrap_or(false))
            .collect();
        assert_eq!(files_after_b.len(), 12,
            "expected 12 cache files (6 from A + 6 fresh from B), got {}",
            files_after_b.len());

        // Correctness: B's result must match a fresh uncached run.
        let rows_b = read_ivec_rows(&tmp.path().join("b.ivec"));
        let baseline_tmp = tempfile::tempdir().unwrap();
        write_fvec(&baseline_tmp.path().join("b.fvec"), &base);
        write_fvec(&baseline_tmp.path().join("q.fvec"), &query);
        let mut bctx = make_ctx(baseline_tmp.path());
        let mut bopts = Options::new();
        bopts.set("base", "b.fvec[0..60)");
        bopts.set("query", "q.fvec");
        bopts.set("indices", "baseline.ivec");
        bopts.set("distances", "baseline-d.fvec");
        bopts.set("neighbors", &k.to_string());
        bopts.set("metric", "L2");
        bopts.set("partition_size", &partition_size.to_string());
        let mut bop = ComputeKnnBlasOp;
        let br = bop.execute(&bopts, &mut bctx);
        assert_eq!(br.status, Status::Ok);
        let rows_base = read_ivec_rows(&baseline_tmp.path().join("baseline.ivec"));
        assert_eq!(rows_b, rows_base,
            "cross-profile reuse produced different neighbors than fresh compute");
    }

    /// Cache-reuse regression: running knn-blas twice against the
    /// same inputs, with different output paths, must hit the
    /// segment cache on the second run — no rewrites of cache
    /// files, byte-identical output. Pairs with the equivalent
    /// tests in `compute_knn.rs` (metal) and `compute_knn_stdarch.rs`
    /// (stdarch). All three engines share the per-engine namespace
    /// contract: `<workspace>/.cache/<engine_name>....`.
    #[test]
    fn test_blas_cache_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_ctx(tmp.path());

        let mut rng = 42u64;
        let mut next_f32 = || -> f32 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let base: Vec<Vec<f32>> = (0..50).map(|_| (0..8).map(|_| next_f32()).collect()).collect();
        let query: Vec<Vec<f32>> = (0..5).map(|_| (0..8).map(|_| next_f32()).collect()).collect();
        write_fvec(&tmp.path().join("base.fvec"), &base);
        write_fvec(&tmp.path().join("query.fvec"), &query);

        // First run.
        let mut opts1 = Options::new();
        opts1.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts1.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts1.set("indices", tmp.path().join("out1.ivec").to_string_lossy().to_string());
        opts1.set("neighbors", "3"); opts1.set("metric", "L2");
        let mut op1 = ComputeKnnBlasOp;
        let r1 = op1.execute(&opts1, &mut ctx);
        assert_eq!(r1.status, Status::Ok, "{}", r1.message);

        let cache_dir = tmp.path().join(".cache");
        let snapshot = |dir: &std::path::Path| -> Vec<(std::ffi::OsString, std::time::SystemTime)> {
            let mut v: Vec<_> = std::fs::read_dir(dir).unwrap()
                .filter_map(|e| e.ok())
                .map(|e| (e.file_name(), e.metadata().unwrap().modified().unwrap()))
                .collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        let mtimes_before = snapshot(&cache_dir);
        assert!(!mtimes_before.is_empty(), "first run must populate the cache");
        for (name, _) in &mtimes_before {
            let s = name.to_string_lossy();
            assert!(s.starts_with(super::ENGINE_NAME),
                "cache file {} not in {} namespace", s, super::ENGINE_NAME);
        }

        std::thread::sleep(std::time::Duration::from_millis(50));

        // Second run, identical inputs, different output path.
        let mut opts2 = Options::new();
        opts2.set("base", tmp.path().join("base.fvec").to_string_lossy().to_string());
        opts2.set("query", tmp.path().join("query.fvec").to_string_lossy().to_string());
        opts2.set("indices", tmp.path().join("out2.ivec").to_string_lossy().to_string());
        opts2.set("neighbors", "3"); opts2.set("metric", "L2");
        let mut op2 = ComputeKnnBlasOp;
        let r2 = op2.execute(&opts2, &mut ctx);
        assert_eq!(r2.status, Status::Ok, "{}", r2.message);

        let mtimes_after = snapshot(&cache_dir);
        assert_eq!(mtimes_before, mtimes_after,
            "knn-blas cache files were rewritten on identical-input second run");

        let out1 = std::fs::read(tmp.path().join("out1.ivec")).unwrap();
        let out2 = std::fs::read(tmp.path().join("out2.ivec")).unwrap();
        assert_eq!(out1, out2, "second run produced different output");
    }
}
