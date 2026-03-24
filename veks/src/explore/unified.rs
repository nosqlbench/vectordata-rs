// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks interact explore` — unified vector space analysis TUI.
//!
//! Single TUI session with tab-switchable views covering norms, distances,
//! eigenvalue structure, and PCA projection. All computation runs on
//! background threads; the TUI thread never blocks.
//!
//! The background pipeline shares a single vector buffer:
//! 1. Read vectors → compute norms (free during read)
//! 2. Compute pairwise distances (from buffered vectors)
//! 3. Compute eigenvalues + eigenvectors (power iteration on centered buffer)
//! 4. Project onto top PCs

use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Instant;

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    text::Span,
    widgets::{Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset, GraphType, Paragraph, canvas::{Canvas, Points}},
    Terminal,
};
use simsimd::SpatialSimilarity;

use super::shared::{
    UnifiedReader, SampleMode, sample_indices, clump_size_for_dim,
    install_abort_handler, normalize,
};
use vectordata::dataset::view::CacheStats;

// ---------------------------------------------------------------------------
// View definitions
// ---------------------------------------------------------------------------

/// Per-view theory and interpretation descriptions shown with the / key.
const VIEW_INFO: &[&str] = &[
    // 0: Norms histogram
    concat!(
        "L2 Norm Distribution\n\n",
        "The L2 norm (Euclidean length) of each vector reveals fundamental properties:\n\n",
        "  Normalized vectors: All norms cluster tightly around 1.0 (std < 0.01).\n",
        "  This means the model was trained with cosine similarity or the vectors\n",
        "  were post-processed. L2 and cosine metrics produce identical rankings.\n\n",
        "  Unnormalized vectors: Norms vary — magnitude carries semantic meaning.\n",
        "  Common in recommendation models where norm encodes item popularity.\n",
        "  L2 distance is sensitive to magnitude; cosine ignores it.\n\n",
        "  Outlier norms: Vectors far from the mean may be degenerate (zero vectors,\n",
        "  padding, or encoding errors). Check with 'veks analyze zeros'.\n\n",
        "Computation:\n",
        "  norm(v) = sqrt(dot(v, v)) computed via SIMD (simsimd).\n",
        "  Stats use Welford's online algorithm for numerically stable\n",
        "  incremental mean and variance.\n",
        "  NORMALIZED verdict: std < 0.01 AND |mean - 1.0| < 0.01.\n\n",
        "Use b/B to adjust bin count for finer or coarser resolution.",
    ),
    // 1: Sorted norms
    concat!(
        "Sorted Norm Curve\n\n",
        "Vectors sorted by L2 norm from smallest to largest.\n\n",
        "  Flat plateau at 1.0: Vectors are L2-normalized.\n",
        "  Smooth monotonic curve: Natural magnitude distribution.\n",
        "  Sharp steps or plateaus: Discrete clusters of magnitudes,\n",
        "  possibly from different data sources or embedding versions.\n",
        "  Outliers at extremes: Degenerate vectors (near-zero or extreme).",
    ),
    // 2: Distance histogram
    concat!(
        "Pairwise Distance Distribution\n\n",
        "Histogram of L2 distances between random pairs of sampled vectors.\n\n",
        "  Tight distribution (low CV): The curse of dimensionality — distances\n",
        "  concentrate as dimensionality increases. This makes nearest-neighbor\n",
        "  search harder because near and far neighbors are barely distinguishable.\n\n",
        "  Relative contrast = (max-min)/min: Values below 2.0 indicate severe\n",
        "  distance concentration. ANN indices will struggle with recall.\n\n",
        "  CV (coefficient of variation) = std/mean: Below 0.1 is concentrated.\n\n",
        "  Bimodal distribution: May indicate natural clusters in the data.\n",
        "  Uniform spread: Data fills the space without strong structure.\n\n",
        "Computation:\n",
        "  Pairs (i,j) sampled uniformly at random from the vector buffer.\n",
        "  L2 distance = sqrt(sum((v_i[d] - v_j[d])^2)) via SIMD l2sq.\n",
        "  Default: sample*50 pairs (capped at full pairwise for small samples).\n",
        "  Contrast = (max - min) / min.  CV = std / mean.",
    ),
    // 3: Sorted distances
    concat!(
        "Sorted Distance Curve\n\n",
        "Pairwise distances sorted ascending. The shape reveals structure:\n\n",
        "  Steep initial rise then plateau: Most pairs are roughly equidistant\n",
        "  (concentrated distances). The plateau height is the 'typical' distance.\n\n",
        "  Smooth concave curve: Good spread — distances are well-distributed.\n",
        "  ANN indices can exploit the distance structure for efficient search.\n\n",
        "  Steps or kinks: Discrete distance shells, common in quantized or\n",
        "  clustered data.",
    ),
    // 4: Scree plot
    concat!(
        "Scree Plot (Eigenvalue Decay)\n\n",
        "Eigenvalues from PCA, largest to smallest. Each eigenvalue represents\n",
        "the variance captured by that principal component.\n\n",
        "  Sharp elbow: The data lives on a low-dimensional manifold.\n",
        "  Components after the elbow are noise. Product quantization (PQ)\n",
        "  and dimensionality reduction will work well.\n\n",
        "  Gradual decay: Variance is spread across many dimensions.\n",
        "  The data is inherently high-dimensional. PQ needs more sub-quantizers.\n\n",
        "  Power-law decay (straight line on log plot): Common in natural data.\n",
        "  The slope indicates the effective dimensionality.\n\n",
        "Computation:\n",
        "  Eigenvalues computed via power iteration on the centered data matrix.\n",
        "  For each component k: iterate v ← (X^T X / n) v with deflation\n",
        "  against previously found eigenvectors. 15 iterations for the first 3\n",
        "  components, 5 for the rest. Eigenvalue = E[dot(Xv, Xv)] = v^T Sigma v.\n",
        "  Uses simsimd SIMD dot products and rayon parallelism.",
    ),
    // 5: Cumulative variance
    concat!(
        "Cumulative Variance Explained\n\n",
        "Running sum of eigenvalues as a percentage of total variance.\n",
        "The gray line marks 95%.\n\n",
        "  95% elbow at k=20 in 1024-dim space: Only 20 effective dimensions.\n",
        "  The remaining 1004 dimensions are noise. Highly compressible.\n\n",
        "  95% elbow at k=500: The data genuinely uses half its dimensions.\n",
        "  Compression ratios will be modest.\n\n",
        "Computation:\n",
        "  cumvar(k) = sum(lambda_1..lambda_k) / sum(all lambda) * 100%\n",
        "  Effective rank = exp(H), H = -sum(p_i * ln(p_i)), p_i = lambda_i / sum.\n",
        "  Intrinsic dim = count of eigenvalues > 1% of lambda_1.\n",
        "  95% elbow = smallest k where cumvar(k) >= 95%.",
    ),
    // 6: Log decay
    concat!(
        "Log Eigenvalue Decay\n\n",
        "Same eigenvalues as the scree plot, but on a log scale (ln).\n\n",
        "  Straight line: Power-law decay, lambda_k ~ k^(-alpha).\n",
        "  The slope alpha determines effective dimensionality.\n",
        "  alpha > 2: low-dimensional, very compressible.\n",
        "  alpha < 1: high-dimensional, compression is hard.\n\n",
        "  Curved (concave up): Faster-than-power-law decay. Excellent for PQ.\n",
        "  Curved (concave down): Slower decay. More dimensions matter.\n\n",
        "  Plateau at bottom: Noise floor. Eigenvalues below this are\n",
        "  dominated by sampling noise, not real structure.",
    ),
    // 7: PCA scatter
    concat!(
        "PCA Scatter Plot (4D)\n\n",
        "Vectors projected onto principal components. Spatial axes show PC1-3,\n",
        "color (blue→red) shows PC4. Press c/C to reassign which PCs map\n",
        "to which display axes.\n\n",
        "  Clusters: Natural groupings visible as colored blobs.\n",
        "  Color gradients across clusters reveal higher-dimensional structure.\n\n",
        "  Uniform cloud: No obvious low-dimensional structure.\n",
        "  The data fills the available space evenly.\n\n",
        "  Elongated shapes: Strong directional variance along certain PCs.\n",
        "  The longest axis is PC1 (highest variance).\n\n",
        "  Rotation (←→↑↓ a/d w/s) explores different 3D projections\n",
        "  of the same high-dimensional data. Some rotations reveal\n",
        "  structure that is invisible from the default viewing angle.",
    ),
];

const VIEW_NAMES: &[&str] = &[
    "1:Norms",
    "2:NormCurve",
    "3:Distances",
    "4:DistCurve",
    "5:Scree",
    "6:CumVar",
    "7:LogDecay",
    "8:PCA",
];
const NUM_VIEWS: usize = 8;

// ---------------------------------------------------------------------------
// Background messages
// ---------------------------------------------------------------------------

/// Phase 1: vectors + norms from reader thread.
struct ReadBatch {
    vectors: Vec<f32>,
    norms: Vec<f64>,
    count: usize,
    cache_stats: Option<CacheStats>,
}

/// Phase 2: pairwise distances.
struct DistBatch {
    distances: Vec<f32>,
}

/// Phase 3: eigenvalue result.
struct EigenMsg {
    ki: usize,
    eigenvalue: f64,
    eigenvector: Vec<f64>,
    elapsed_ms: f64,
}

/// Phase 4: projected points.
struct ProjectionMsg {
    points: Vec<[f64; 5]>,
}

// ---------------------------------------------------------------------------
// Online stats
// ---------------------------------------------------------------------------

struct WelfordStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl WelfordStats {
    fn new() -> Self { WelfordStats { count: 0, mean: 0.0, m2: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY } }
    fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        if x < self.min { self.min = x; }
        if x > self.max { self.max = x; }
    }
    fn std_dev(&self) -> f64 { if self.count < 2 { 0.0 } else { (self.m2 / self.count as f64).sqrt() } }
    fn is_normalized(&self) -> bool { self.count > 10 && self.std_dev() < 0.01 && (self.mean - 1.0).abs() < 0.01 }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(super) fn run_interactive_explore(
    source: &str, sample_size: usize, seed: u64, sample_mode: SampleMode,
) {
    eprintln!("Opening {}...", source);
    let reader = UnifiedReader::open(source);
    let total = reader.count();
    let dim = reader.dim();

    if total == 0 || dim == 0 {
        eprintln!("Error: no vector data found in '{}'", source);
        std::process::exit(1);
    }

    let filename = if super::shared::is_local_source(source) {
        super::shared::resolve_source(source)
            .file_name().unwrap_or_default().to_string_lossy().to_string()
    } else {
        source.to_string()
    };

    let mut current_sample = sample_size.min(total);
    let is_remote = reader.is_remote();
    let clump = clump_size_for_dim(dim, is_remote);

    let abort_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    install_abort_handler(abort_flag.clone());

    if let Err(e) = enable_raw_mode() {
        eprintln!("Error: {}", e); std::process::exit(1);
    }
    let mut stdout = std::io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen) {
        let _ = disable_raw_mode(); eprintln!("Error: {}", e); std::process::exit(1);
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    // ── View state (persists across sample size restarts) ──
    let mut view_mode: usize = 0;
    let mut num_bins: usize = 0; // 0 = auto (fill available width)
    let mut show_help = false;
    let mut show_info = false; // per-view theory description (/ key)
    let mut info_scroll: u16 = 0; // scroll offset for info view
    let mut rot_y: f64 = 0.0;  // ←→ arrows
    let mut rot_x: f64 = 0.0;  // ↑↓ arrows
    let mut rot_z: f64 = 0.0;  // a/d keys
    let mut rot_w: f64 = 0.0;  // w/s keys
    let mut pc_axes: [usize; 5] = [0, 1, 2, 3, 4];

    // Outer loop: restarts computation when sample size changes
    loop {
    let indices = sample_indices(total, current_sample, seed, sample_mode, clump);

    // ── Phase 1: Read vectors + compute norms on background thread ──
    let (read_tx, read_rx) = mpsc::channel::<ReadBatch>();
    {
        let source_owned = source.to_string();
        let indices_clone = indices.clone();
        let dim_copy = dim;
        std::thread::spawn(move || {
            let bg_reader = UnifiedReader::open(&source_owned);
            let batch_size = 500;
            let mut batch_vecs: Vec<f32> = Vec::with_capacity(batch_size * dim_copy);
            let mut batch_norms: Vec<f64> = Vec::with_capacity(batch_size);
            let mut batch_count = 0;

            for &idx in &indices_clone {
                if let Some(v) = bg_reader.get_f32(idx) {
                    let norm_sq = <f32 as SpatialSimilarity>::dot(&v, &v).unwrap_or(0.0);
                    batch_norms.push((norm_sq as f64).sqrt());
                    batch_vecs.extend_from_slice(&v);
                    batch_count += 1;
                }
                if batch_count >= batch_size {
                    let stats = bg_reader.cache_stats();
                    let _ = read_tx.send(ReadBatch {
                        vectors: std::mem::take(&mut batch_vecs),
                        norms: std::mem::take(&mut batch_norms),
                        count: batch_count,
                        cache_stats: stats,
                    });
                    batch_vecs = Vec::with_capacity(batch_size * dim_copy);
                    batch_norms = Vec::with_capacity(batch_size);
                    batch_count = 0;
                }
            }
            if batch_count > 0 {
                let stats = bg_reader.cache_stats();
                let _ = read_tx.send(ReadBatch {
                    vectors: batch_vecs, norms: batch_norms, count: batch_count, cache_stats: stats,
                });
            }
        });
    }

    // ── Computation state (reset on each restart) ──
    let compute_start = Instant::now();

    // Phase 1 state
    let mut vector_buf: Vec<f32> = Vec::with_capacity(current_sample * dim);
    let mut vectors_loaded: usize = 0;
    let mut all_norms: Vec<f64> = Vec::with_capacity(current_sample);
    let mut norm_stats = WelfordStats::new();
    let mut last_cache_stats: Option<CacheStats> = None;
    let mut phase1_done = false;

    // Sorted caches for curve views (avoid re-sorting every frame)
    let mut sorted_norms: Vec<f64> = Vec::new();
    let mut sorted_norms_len: usize = 0;
    let mut sorted_dists: Vec<f64> = Vec::new();
    let mut sorted_dists_len: usize = 0;

    // Phase 2 state (distances)
    let mut all_dists: Vec<f32> = Vec::new();
    let mut dist_stats = WelfordStats::new();
    let mut dist_rx: Option<mpsc::Receiver<DistBatch>> = None;
    let mut phase2_done = false;
    let total_pairs = (current_sample * 50).min(current_sample.saturating_mul(current_sample.saturating_sub(1)) / 2);

    // Phase 3 state (eigenvalues)
    let mut eigenvalues: Vec<f64> = Vec::new();
    let mut eigenvectors: Vec<Vec<f64>> = Vec::new();
    let mut eigen_rx: Option<mpsc::Receiver<EigenMsg>> = None;
    let mut phase3_done = false;
    let num_eigenvalues = 10usize.min(dim);
    let mut avg_eigen_ms = 0.0f64;

    // Phase 4 state (projection)
    let mut projected: Vec<[f64; 5]> = Vec::new();
    let mut proj_rx: Option<mpsc::Receiver<ProjectionMsg>> = None;
    let mut phase4_done = false;

    // PCA rotation (persists — moved before outer loop)

    // Overall phase tracking
    let mut status_msg;
    let mut final_elapsed: Option<f64> = None;
    let mut restart = false;

    loop {
        if abort_flag.load(Ordering::Relaxed) {
            let _ = disable_raw_mode();
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
            eprintln!("Aborted.");
            std::process::exit(130);
        }

        // ── Drain Phase 1 (read + norms) ──
        if !phase1_done {
            loop {
                match read_rx.try_recv() {
                    Ok(batch) => {
                        vector_buf.extend_from_slice(&batch.vectors);
                        for &n in &batch.norms { norm_stats.update(n); }
                        all_norms.extend_from_slice(&batch.norms);
                        vectors_loaded += batch.count;
                        if let Some(cs) = batch.cache_stats { last_cache_stats = Some(cs); }
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        phase1_done = true;
                        // Kick off Phase 2 (distances) + Phase 3 (eigenvalues) in parallel
                        if vectors_loaded > 1 {
                            // Phase 2: distances
                            let buf = std::sync::Arc::new(vector_buf.clone());
                            let dim_c = dim;
                            let n = vectors_loaded;
                            let pairs = total_pairs;
                            let (dtx, drx) = mpsc::channel::<DistBatch>();
                            dist_rx = Some(drx);
                            let buf2 = buf.clone();
                            std::thread::spawn(move || {
                                let dist_fn = crate::pipeline::simd_distance::select_distance_fn(
                                    crate::pipeline::simd_distance::Metric::L2);
                                let mut rng = crate::pipeline::rng::seeded_rng(42);
                                use rand::Rng;
                                let bs = 1000;
                                let mut batch = Vec::with_capacity(bs);
                                for _ in 0..pairs {
                                    let i = rng.random_range(0..n);
                                    let j = rng.random_range(0..n - 1);
                                    let j = if j >= i { j + 1 } else { j };
                                    let d = dist_fn(&buf2[i*dim_c..(i+1)*dim_c], &buf2[j*dim_c..(j+1)*dim_c]).sqrt();
                                    batch.push(d);
                                    if batch.len() >= bs {
                                        let _ = dtx.send(DistBatch { distances: std::mem::take(&mut batch) });
                                        batch = Vec::with_capacity(bs);
                                    }
                                }
                                if !batch.is_empty() { let _ = dtx.send(DistBatch { distances: batch }); }
                            });

                            // Phase 3: eigenvalues (center + power iteration)
                            let mut centered_buf = vector_buf.clone();
                            let actual_n = vectors_loaded;
                            let dim_c = dim;
                            let k = num_eigenvalues;
                            // Compute mean and center
                            let mut mean_f32 = vec![0.0f32; dim_c];
                            for vi in 0..actual_n {
                                let off = vi * dim_c;
                                for d in 0..dim_c { mean_f32[d] += centered_buf[off + d]; }
                            }
                            for d in 0..dim_c { mean_f32[d] /= actual_n as f32; }
                            for vi in 0..actual_n {
                                let off = vi * dim_c;
                                for d in 0..dim_c { centered_buf[off + d] -= mean_f32[d]; }
                            }

                            let (etx, erx) = mpsc::channel::<EigenMsg>();
                            eigen_rx = Some(erx);
                            std::thread::spawn(move || {
                                let mut evecs: Vec<Vec<f64>> = Vec::new();
                                for ki in 0..k {
                                    let iters = if ki < 3 { 15 } else { 5 };
                                    let mut v: Vec<f64> = (0..dim_c).map(|d| ((d * 7 + 13) % 97) as f64 - 48.0).collect();
                                    normalize(&mut v);
                                    let t0 = Instant::now();
                                    for _ in 0..iters {
                                        let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                                        let mut new_v = vec![0.0f64; dim_c];
                                        for vi in 0..actual_n {
                                            let off = vi * dim_c;
                                            let c = &centered_buf[off..off + dim_c];
                                            let dot = <f32 as SpatialSimilarity>::dot(c, &v_f32).unwrap_or(0.0) as f64;
                                            for d in 0..dim_c { new_v[d] += c[d] as f64 * dot; }
                                        }
                                        for d in 0..dim_c { new_v[d] /= actual_n as f64; }
                                        for prev in &evecs {
                                            let proj: f64 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                                            for d in 0..dim_c { new_v[d] -= proj * prev[d]; }
                                        }
                                        normalize(&mut new_v);
                                        v = new_v;
                                    }
                                    // Eigenvalue
                                    let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                                    let mut ev_sum = 0.0f64;
                                    for vi in 0..actual_n {
                                        let off = vi * dim_c;
                                        let c = &centered_buf[off..off + dim_c];
                                        let dot = <f32 as SpatialSimilarity>::dot(c, &v_f32).unwrap_or(0.0) as f64;
                                        ev_sum += dot * dot;
                                    }
                                    let eigenvalue = ev_sum / actual_n as f64;
                                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                                    evecs.push(v.clone());
                                    let _ = etx.send(EigenMsg { ki, eigenvalue, eigenvector: v, elapsed_ms });
                                }
                            });
                        } else {
                            phase2_done = true;
                            phase3_done = true;
                            phase4_done = true;
                        }
                        break;
                    }
                }
            }
        }

        // ── Drain Phase 2 (distances) ──
        if phase1_done && !phase2_done {
            if let Some(ref rx) = dist_rx {
                loop {
                    match rx.try_recv() {
                        Ok(batch) => {
                            for &d in &batch.distances { dist_stats.update(d as f64); }
                            all_dists.extend_from_slice(&batch.distances);
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => { phase2_done = true; break; }
                    }
                }
            }
        }

        // ── Drain Phase 3 (eigenvalues) ──
        if phase1_done && !phase3_done {
            if let Some(ref rx) = eigen_rx {
                loop {
                    match rx.try_recv() {
                        Ok(msg) => {
                            eigenvalues.push(msg.eigenvalue);
                            eigenvectors.push(msg.eigenvector);
                            avg_eigen_ms = if eigenvalues.len() > 1 {
                                (avg_eigen_ms * (eigenvalues.len() - 1) as f64 + msg.elapsed_ms) / eigenvalues.len() as f64
                            } else { msg.elapsed_ms };
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            phase3_done = true;
                            // Kick off Phase 4 (projection) if we have eigenvectors
                            if eigenvectors.len() >= 3 {
                                let buf = std::sync::Arc::new(vector_buf.clone());
                                let dim_c = dim;
                                let n = vectors_loaded;
                                let evecs = eigenvectors.clone();
                                // Need to re-center (buf is raw, not centered)
                                let mut mean_f32 = vec![0.0f32; dim_c];
                                for vi in 0..n {
                                    let off = vi * dim_c;
                                    for d in 0..dim_c { mean_f32[d] += buf[off + d]; }
                                }
                                for d in 0..dim_c { mean_f32[d] /= n as f32; }

                                let (ptx, prx) = mpsc::channel::<ProjectionMsg>();
                                proj_rx = Some(prx);
                                std::thread::spawn(move || {
                                    use rayon::prelude::*;
                                    let ev_f32: Vec<Vec<f32>> = (0..5.min(evecs.len()))
                                        .map(|i| evecs[i].iter().map(|&x| x as f32).collect())
                                        .collect();
                                    let zero = vec![0.0f32; dim_c];
                                    let chunk_size = 4096.max(n / rayon::current_num_threads().max(1));
                                    let pts: Vec<[f64; 5]> = (0..n)
                                        .collect::<Vec<_>>()
                                        .par_chunks(chunk_size)
                                        .flat_map(|idx_chunk| {
                                            let mut centered = vec![0.0f32; dim_c];
                                            idx_chunk.iter().map(|&vi| {
                                                let off = vi * dim_c;
                                                for d in 0..dim_c { centered[d] = buf[off + d] - mean_f32[d]; }
                                                let mut pcs = [0.0f64; 5];
                                                for k in 0..5 {
                                                    let ev = if k < ev_f32.len() { &ev_f32[k] } else { &zero };
                                                    pcs[k] = <f32 as SpatialSimilarity>::dot(&centered, ev).unwrap_or(0.0) as f64;
                                                }
                                                pcs
                                            }).collect::<Vec<_>>()
                                        }).collect();
                                    let _ = ptx.send(ProjectionMsg { points: pts });
                                });
                            } else {
                                phase4_done = true;
                            }
                            break;
                        }
                    }
                }
            }
        }

        // ── Drain Phase 4 (projection) ──
        if phase3_done && !phase4_done {
            if let Some(ref rx) = proj_rx {
                match rx.try_recv() {
                    Ok(msg) => { projected = msg.points; phase4_done = true; }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => { phase4_done = true; }
                }
            }
        }

        let elapsed = compute_start.elapsed().as_secs_f64();

        // ── Update sorted caches if data changed ──
        if all_norms.len() != sorted_norms_len {
            sorted_norms = all_norms.clone();
            sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_norms_len = all_norms.len();
        }
        if all_dists.len() != sorted_dists_len {
            sorted_dists = all_dists.iter().map(|&d| d as f64).collect();
            sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_dists_len = all_dists.len();
        }

        // ── Status ──
        status_msg = if !phase1_done {
            let rate = if elapsed > 0.0 { vectors_loaded as f64 / elapsed } else { 0.0 };
            let ci = last_cache_stats.as_ref().map(|cs| {
                let pct = if cs.total_chunks > 0 { 100.0 * cs.valid_chunks as f64 / cs.total_chunks as f64 } else { 0.0 };
                format!(" | cache {:.0}%", pct)
            }).unwrap_or_default();
            format!("Reading: {}/{} ({:.0}/s){}", vectors_loaded, current_sample, rate, ci)
        } else if !phase2_done || !phase3_done {
            let d_status = if phase2_done { format!("dists:{}", all_dists.len()) }
                else { format!("dists:{}/{}", all_dists.len(), total_pairs) };
            let e_status = if phase3_done { format!("eigen:{}", eigenvalues.len()) }
                else {
                    let rem = num_eigenvalues.saturating_sub(eigenvalues.len());
                    let eta = rem as f64 * avg_eigen_ms / 1000.0;
                    format!("eigen:{}/{} eta:{:.0}s", eigenvalues.len(), num_eigenvalues, eta)
                };
            format!("Computing: {} | {}", d_status, e_status)
        } else if !phase4_done {
            "Projecting...".into()
        } else {
            let t = *final_elapsed.get_or_insert(elapsed);
            format!("Ready — {} vectors, {:.1}s total", vectors_loaded, t)
        };

        // ── Stats line for current view ──
        let stats_line = match view_mode {
            0 | 1 => { // Norms
                if norm_stats.count > 0 {
                    let verdict = if norm_stats.is_normalized() { "NORMALIZED" } else { "not normalized" };
                    format!("mean={:.4} std={:.4} min={:.4} max={:.4} {}", norm_stats.mean, norm_stats.std_dev(), norm_stats.min, norm_stats.max, verdict)
                } else { String::new() }
            }
            2 | 3 => { // Distances
                if dist_stats.count > 0 {
                    let contrast = if dist_stats.min > 0.0 { (dist_stats.max - dist_stats.min) / dist_stats.min } else { f64::INFINITY };
                    let cv = if dist_stats.mean > 0.0 { dist_stats.std_dev() / dist_stats.mean } else { 0.0 };
                    format!("mean={:.4} std={:.4} contrast={:.1} CV={:.3}", dist_stats.mean, dist_stats.std_dev(), contrast, cv)
                } else { String::new() }
            }
            4..=6 => { // Eigenvalue views
                if !eigenvalues.is_empty() {
                    let total_ev: f64 = eigenvalues.iter().sum();
                    let mut entropy = 0.0f64;
                    for &ev in &eigenvalues { if ev > 0.0 { let p = ev / total_ev; entropy -= p * p.ln(); } }
                    let eff_rank = entropy.exp();
                    let threshold = eigenvalues[0] * 0.01;
                    let intrinsic = eigenvalues.iter().filter(|&&v| v > threshold).count();
                    let mut cum = 0.0;
                    let elbow = eigenvalues.iter().position(|&v| { cum += v; 100.0 * cum / total_ev >= 95.0 }).map(|i| i + 1).unwrap_or(eigenvalues.len());
                    format!("eff_rank={:.1} intrinsic_dim={} 95%-elbow={}", eff_rank, intrinsic, elbow)
                } else { String::new() }
            }
            7 => { // PCA view
                if !eigenvalues.is_empty() {
                    let total_ev: f64 = eigenvalues.iter().sum();
                    let pcts: Vec<f64> = eigenvalues.iter().take(5).map(|v| 100.0 * v / total_ev).collect();
                    format!("PC1:{:.1}% PC2:{:.1}% PC3:{:.1}% PC4:{:.1}% PC5:{:.1}%",
                        pcts.get(0).unwrap_or(&0.0), pcts.get(1).unwrap_or(&0.0),
                        pcts.get(2).unwrap_or(&0.0), pcts.get(3).unwrap_or(&0.0),
                        pcts.get(4).unwrap_or(&0.0))
                } else { String::new() }
            }
            _ => String::new(),
        };

        // ── Render ──
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),  // title
                    Constraint::Min(8),    // main view
                    Constraint::Length(1),  // stats
                    Constraint::Length(1),  // view selector
                    Constraint::Length(1),  // view-specific controls
                    Constraint::Length(1),  // status
                ]).split(frame.area());

            let title = format!(" Explore: {} — {} dims — {}/{} vectors", filename, dim, vectors_loaded, current_sample);
            frame.render_widget(Paragraph::new(Span::styled(title, Style::default().fg(Color::Cyan))), chunks[0]);

            // Main view (or help/info overlay)
            if show_info {
                use ratatui::text::Line;
                let info_text = VIEW_INFO.get(view_mode).unwrap_or(&"");
                let lines: Vec<Line> = info_text.lines()
                    .map(|l| Line::from(Span::styled(l, Style::default().fg(Color::White))))
                    .collect();
                let title = format!(" {} — Theory & Interpretation (↑↓ scroll, / to dismiss) ", VIEW_NAMES[view_mode]);
                frame.render_widget(
                    Paragraph::new(lines)
                        .block(Block::default().borders(Borders::ALL).title(title))
                        .scroll((info_scroll, 0)),
                    chunks[1],
                );
            } else if show_help {
                use ratatui::text::Line;
                let help = vec![
                    Line::from(Span::styled(" Keyboard Shortcuts", Style::default().fg(Color::Cyan))),
                    Line::from(""),
                    Line::from(" 1-8 / Tab / PgUp/PgDn / F1-F8  Switch view"),
                    Line::from(" b / B                           Fewer / more histogram bins"),
                    Line::from(" ←→↑↓                           Rotate PCA (Y/X axes)"),
                    Line::from(" a / d                           Rotate PCA (Z axis)"),
                    Line::from(" w / s                           Rotate PCA (W tilt)"),
                    Line::from(" c / C                           Slide component window (PC1-4 → PC2-5 → ...)"),
                    Line::from(" x / X                           Rotate axis roles (X→Y→Z→color cycle)"),
                    Line::from(" r                               Reset all rotations"),
                    Line::from(" Space                           Double sample size"),
                    Line::from(" + / =                           Increase sample by 50%"),
                    Line::from(" ?                               Toggle this help"),
                    Line::from(" q / Esc / Ctrl-C                Quit"),
                ];
                frame.render_widget(
                    Paragraph::new(help).block(Block::default().borders(Borders::ALL).title(" Help ")),
                    chunks[1],
                );
            } else { match view_mode {
                0 => render_histogram(frame, chunks[1], &all_norms, &norm_stats, num_bins, " Norm Distribution ", Color::Green),
                1 => render_presorted_curve(frame, chunks[1], &sorted_norms, " Sorted Norms ", Color::Green),
                2 => {
                    let dists_f64: Vec<f64> = all_dists.iter().map(|&d| d as f64).collect();
                    render_histogram(frame, chunks[1], &dists_f64, &dist_stats, num_bins, " Distance Distribution (L2) ", Color::Cyan);
                }
                3 => {
                    render_presorted_curve(frame, chunks[1], &sorted_dists, " Sorted Distances ", Color::Cyan);
                }
                4 => render_scree(frame, chunks[1], &eigenvalues),
                5 => render_cumulative(frame, chunks[1], &eigenvalues),
                6 => render_log_decay(frame, chunks[1], &eigenvalues),
                7 => render_pca_scatter(frame, chunks[1], &projected, rot_y, rot_x, rot_z, rot_w, &pc_axes, 4),
                _ => {}
            } } // close match + else

            frame.render_widget(Paragraph::new(Span::styled(&stats_line, Style::default().fg(Color::White))), chunks[2]);

            // View selector line
            let vi: String = VIEW_NAMES.iter().enumerate()
                .map(|(i, n)| if i == view_mode { format!("[{}]", n) } else { n.to_string() })
                .collect::<Vec<_>>().join(" ");
            frame.render_widget(Paragraph::new(Span::styled(
                format!(" {} | Tab/PgDn/PgUp | F1-F8 | /: info | ?: help | q quit", vi),
                Style::default().fg(Color::DarkGray))), chunks[3]);

            // View-specific controls line
            let view_controls: String = match view_mode {
                0 | 2 => " b/B: fewer/more bins | +/Space: increase sample".into(),
                1 | 3 => " +/Space: increase sample".into(),
                4..=6 => String::new(),
                7 => format!(" ←→↑↓ a/d w/s: rotate | c/C: slide PCs | x/X: swap axes [X:PC{} Y:PC{} Z:PC{} color:PC{}] | r +/Space",
                    pc_axes[0]+1, pc_axes[1]+1, pc_axes[2]+1, pc_axes[3]+1),
                _ => String::new(),
            };
            frame.render_widget(Paragraph::new(Span::styled(
                &view_controls, Style::default().fg(Color::DarkGray))), chunks[4]);

            let sc = if phase4_done { Color::Green } else { Color::Yellow };
            frame.render_widget(Paragraph::new(Span::styled(&status_msg, Style::default().fg(sc))), chunks[5]);
        }).unwrap();

        // ── Events ──
        let poll_ms = if phase4_done { 100 } else { 10 };
        if event::poll(std::time::Duration::from_millis(poll_ms)).unwrap() {
            match event::read().unwrap() {
                Event::Key(key) => {
                    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) { break; }
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('?') => { show_help = !show_help; show_info = false; }
                        KeyCode::Char('/') => { show_info = !show_info; show_help = false; info_scroll = 0; }
                        // Arrow keys scroll info when info panel is open
                        KeyCode::Up if show_info => { info_scroll = info_scroll.saturating_sub(1); }
                        KeyCode::Down if show_info => { info_scroll += 1; }
                        KeyCode::Tab => { view_mode = (view_mode + 1) % NUM_VIEWS; }
                        KeyCode::BackTab => { view_mode = (view_mode + NUM_VIEWS - 1) % NUM_VIEWS; }
                        KeyCode::Char('1') => view_mode = 0,
                        KeyCode::Char('2') => view_mode = 1,
                        KeyCode::Char('3') => view_mode = 2,
                        KeyCode::Char('4') => view_mode = 3,
                        KeyCode::Char('5') => view_mode = 4,
                        KeyCode::Char('6') => view_mode = 5,
                        KeyCode::Char('7') => view_mode = 6,
                        KeyCode::Char('8') => view_mode = 7,
                        KeyCode::Char('0') => view_mode = 9,
                        KeyCode::Char('b') => {
                            if num_bins == 0 { num_bins = 100; } // switch from auto to manual
                            num_bins = num_bins.saturating_sub(5).max(10);
                        }
                        KeyCode::Char('B') => {
                            if num_bins == 0 { num_bins = 100; }
                            num_bins = (num_bins + 5).min(500);
                        }
                        // Rotation axes (PCA scatter views only)
                        KeyCode::Left if view_mode == 7 => { rot_y -= 0.1; }
                        KeyCode::Right if view_mode == 7 => { rot_y += 0.1; }
                        KeyCode::Up if view_mode == 7 => { rot_x -= 0.1; }
                        KeyCode::Down if view_mode == 7 => { rot_x += 0.1; }
                        KeyCode::Char('a') if view_mode == 7 => { rot_z -= 0.1; }
                        KeyCode::Char('d') if view_mode == 7 => { rot_z += 0.1; }
                        KeyCode::Char('w') if view_mode == 7 => { rot_w -= 0.1; }
                        KeyCode::Char('s') if view_mode == 7 => { rot_w += 0.1; }
                        KeyCode::Char('r') => { rot_y = 0.0; rot_x = 0.0; rot_z = 0.0; rot_w = 0.0; }
                        // c/C: slide the component window (PC1-4 → PC2-5 → PC3-6...)
                        KeyCode::Char('c') if view_mode == 7 => {
                            let max_pc = eigenvalues.len().max(5);
                            for a in pc_axes.iter_mut() { *a = (*a + 1) % max_pc; }
                        }
                        KeyCode::Char('C') if view_mode == 7 => {
                            let max_pc = eigenvalues.len().max(5);
                            for a in pc_axes.iter_mut() { *a = (*a + max_pc - 1) % max_pc; }
                        }
                        // x/X: rotate axis assignment within the current set
                        // x: X→Y→Z→color cycle (rotate display roles forward)
                        // X: reverse
                        KeyCode::Char('x') if view_mode == 7 => {
                            let tmp = pc_axes[0];
                            pc_axes[0] = pc_axes[1];
                            pc_axes[1] = pc_axes[2];
                            pc_axes[2] = pc_axes[3];
                            pc_axes[3] = tmp;
                        }
                        KeyCode::Char('X') if view_mode == 7 => {
                            let tmp = pc_axes[3];
                            pc_axes[3] = pc_axes[2];
                            pc_axes[2] = pc_axes[1];
                            pc_axes[1] = pc_axes[0];
                            pc_axes[0] = tmp;
                        }
                        // Page Up/Down for view switching
                        KeyCode::PageDown => { view_mode = (view_mode + 1) % NUM_VIEWS; }
                        KeyCode::PageUp => { view_mode = (view_mode + NUM_VIEWS - 1) % NUM_VIEWS; }
                        // Function keys F1-F10 as view shortcuts
                        KeyCode::F(n) if (1..=10).contains(&n) => { view_mode = (n as usize - 1) % NUM_VIEWS; }
                        // Sample size expansion
                        KeyCode::Char(' ') if current_sample < total => {
                            current_sample = (current_sample * 2).min(total);
                            restart = true;
                            break;
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') if current_sample < total => {
                            current_sample = (current_sample * 3 / 2).min(total);
                            restart = true;
                            break;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    } // end inner loop

    if !restart { break; }

    } // end 'compute loop

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

fn render_histogram(
    frame: &mut ratatui::Frame, area: ratatui::layout::Rect,
    values: &[f64], stats: &WelfordStats, num_bins: usize, title: &str, color: Color,
) {
    if values.is_empty() {
        frame.render_widget(Paragraph::new("  Waiting for data...").block(Block::default().borders(Borders::ALL).title(title)), area);
        return;
    }
    // Auto bins: use available width minus borders (2 chars)
    let effective_bins = if num_bins == 0 {
        (area.width as usize).saturating_sub(2).max(10)
    } else {
        num_bins
    };
    let (lo, hi) = (stats.min, stats.max);
    let range = (hi - lo).max(1e-10);
    let bw = range / effective_bins as f64;
    let mut bins = vec![0u64; effective_bins];
    for &v in values {
        let bi = ((v - lo) / bw).floor() as usize;
        bins[bi.min(effective_bins - 1)] += 1;
    }
    let max_c = *bins.iter().max().unwrap_or(&1);
    let label_interval = (effective_bins / 5).max(1);
    let precision = if range < 0.01 { 6 } else if range < 1.0 { 4 } else if range < 100.0 { 2 } else { 0 };
    let bars: Vec<Bar> = bins.iter().enumerate().map(|(i, &c)| {
        let label = if i % label_interval == 0 {
            format!("{:.*}", precision, lo + i as f64 * bw)
        } else {
            String::new()
        };
        Bar::default().value(c).label(label.into()).style(Style::default().fg(color))
    }).collect();
    let chart = BarChart::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .data(BarGroup::default().bars(&bars)).bar_gap(0).max(max_c);
    frame.render_widget(chart, area);
}

/// Render a pre-sorted curve (data must already be sorted ascending).
fn render_presorted_curve(
    frame: &mut ratatui::Frame, area: ratatui::layout::Rect,
    sorted: &[f64], title: &str, color: Color,
) {
    if sorted.len() < 2 {
        frame.render_widget(Paragraph::new("  Waiting for data...").block(Block::default().borders(Borders::ALL).title(title)), area);
        return;
    }
    let n = sorted.len();
    // Downsample to at most 2x the available width to keep rendering fast
    let max_pts = (area.width as usize * 2).max(100);
    let step = (n / max_pts).max(1);
    let pts: Vec<(f64, f64)> = sorted.iter().enumerate()
        .step_by(step)
        .map(|(i, &v)| (i as f64 / n as f64, v))
        .collect();
    let pad = (sorted[n-1] - sorted[0]).max(0.01) * 0.05;
    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .x_bounds([0.0, 1.0]).y_bounds([sorted[0] - pad, sorted[n-1] + pad])
        .paint(move |ctx| { ctx.draw(&Points { coords: &pts, color }); });
    frame.render_widget(canvas, area);
}

fn render_sorted_curve(
    frame: &mut ratatui::Frame, area: ratatui::layout::Rect,
    values: &[f64], title: &str, color: Color,
) {
    if values.len() < 2 {
        frame.render_widget(Paragraph::new("  Waiting for data...").block(Block::default().borders(Borders::ALL).title(title)), area);
        return;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let pts: Vec<(f64, f64)> = sorted.iter().enumerate().map(|(i, &v)| (i as f64 / n as f64, v)).collect();
    let pad = (sorted[n-1] - sorted[0]).max(0.01) * 0.05;
    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .x_bounds([0.0, 1.0]).y_bounds([sorted[0] - pad, sorted[n-1] + pad])
        .paint(move |ctx| { ctx.draw(&Points { coords: &pts, color }); });
    frame.render_widget(canvas, area);
}

fn render_scree(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, eigenvalues: &[f64]) {
    if eigenvalues.len() < 2 {
        frame.render_widget(Paragraph::new("  Computing eigenvalues...").block(Block::default().borders(Borders::ALL).title(" Scree Plot ")), area);
        return;
    }
    let data: Vec<(f64, f64)> = eigenvalues.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect();
    let x_max = (eigenvalues.len() - 1) as f64;
    let y_max = eigenvalues[0] * 1.1;
    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Yellow))
        .data(&data);
    let chart = Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL).title(" Scree Plot (Eigenvalue Decay) "))
        .x_axis(Axis::default().bounds([0.0, x_max]).labels(vec![
            Span::raw("0"), Span::raw(format!("{}", eigenvalues.len() - 1)),
        ]))
        .y_axis(Axis::default().bounds([0.0, y_max]).labels(vec![
            Span::raw("0"), Span::raw(format!("{:.2}", y_max)),
        ]));
    frame.render_widget(chart, area);
}

fn render_cumulative(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, eigenvalues: &[f64]) {
    if eigenvalues.is_empty() {
        frame.render_widget(Paragraph::new("  Computing...").block(Block::default().borders(Borders::ALL).title(" Cumulative Variance ")), area);
        return;
    }
    let total: f64 = eigenvalues.iter().sum();
    let mut cum = 0.0;
    let data: Vec<(f64, f64)> = eigenvalues.iter().enumerate().map(|(i, &v)| { cum += v; (i as f64, 100.0 * cum / total) }).collect();
    let threshold: Vec<(f64, f64)> = vec![(0.0, 95.0), ((eigenvalues.len() - 1) as f64, 95.0)];
    let x_max = (eigenvalues.len().max(1) - 1) as f64;
    let ds_data = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Green))
        .data(&data);
    let ds_thresh = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::DarkGray))
        .data(&threshold);
    let chart = Chart::new(vec![ds_thresh, ds_data])
        .block(Block::default().borders(Borders::ALL).title(" Cumulative Variance (%) — 95% line "))
        .x_axis(Axis::default().bounds([0.0, x_max]).labels(vec![
            Span::raw("0"), Span::raw(format!("{}", eigenvalues.len() - 1)),
        ]))
        .y_axis(Axis::default().bounds([0.0, 105.0]).labels(vec![
            Span::raw("0%"), Span::raw("50%"), Span::raw("100%"),
        ]));
    frame.render_widget(chart, area);
}

fn render_log_decay(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, eigenvalues: &[f64]) {
    if eigenvalues.len() < 2 {
        frame.render_widget(Paragraph::new("  Computing...").block(Block::default().borders(Borders::ALL).title(" Log Eigenvalue Decay ")), area);
        return;
    }
    let data: Vec<(f64, f64)> = eigenvalues.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, &v)| (i as f64, v.ln())).collect();
    let y_min = data.last().map(|p| p.1).unwrap_or(0.0) - 0.5;
    let y_max = data.first().map(|p| p.1).unwrap_or(1.0) + 0.5;
    let x_max = (eigenvalues.len() - 1) as f64;
    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Magenta))
        .data(&data);
    let chart = Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL).title(" Log Eigenvalue Decay "))
        .x_axis(Axis::default().bounds([0.0, x_max]).labels(vec![
            Span::raw("0"), Span::raw(format!("{}", eigenvalues.len() - 1)),
        ]))
        .y_axis(Axis::default().bounds([y_min, y_max]).labels(vec![
            Span::raw(format!("{:.1}", y_min)), Span::raw(format!("{:.1}", y_max)),
        ]));
    frame.render_widget(chart, area);
}

fn render_pca_scatter(
    frame: &mut ratatui::Frame, area: ratatui::layout::Rect,
    projected: &[[f64; 5]], rot_y: f64, rot_x: f64, rot_z: f64, rot_w: f64,
    pc_axes: &[usize; 5], dims: usize,
) {
    if projected.is_empty() {
        let title = format!(" {}D Scatter — computing... ", dims);
        frame.render_widget(Paragraph::new("  Projecting...").block(Block::default().borders(Borders::ALL).title(title)), area);
        return;
    }

    let (sy, cy) = rot_y.sin_cos();
    let (sx, cx) = rot_x.sin_cos();
    let (sz, cz) = rot_z.sin_cos();
    let (sw, cw) = rot_w.sin_cos();

    // Map PC axes: pc_axes[0..5] selects which of the projected PCs to use
    let get_pc = |p: &[f64; 5], axis: usize| -> f64 {
        let idx = pc_axes[axis];
        if idx < 5 { p[idx] } else { 0.0 }
    };

    // Apply 4 rotation planes: Y, X, Z, W
    let rotated: Vec<(f64, f64, f64, f64)> = projected.iter().map(|p| {
        let (x, y, z) = (get_pc(p, 0), get_pc(p, 1), get_pc(p, 2));
        let x1 = x * cy + z * sy;
        let y1 = y;
        let z1 = -x * sy + z * cy;
        let x2 = x1;
        let y2 = y1 * cx - z1 * sx;
        let z2 = y1 * sx + z1 * cx;
        let x3 = x2 * cz - y2 * sz;
        let y3 = x2 * sz + y2 * cz;
        let x4 = x3 * cw - z2 * sw;
        let y4 = y3;
        (x4, y4, get_pc(p, 3), get_pc(p, 4))
    }).collect();

    let x_min = rotated.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = rotated.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = rotated.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = rotated.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let x_pad = (x_max - x_min).max(0.01) * 0.02;
    let y_pad = (y_max - y_min).max(0.01) * 0.02;

    if dims == 3 {
        let pts: Vec<(f64, f64)> = rotated.iter().map(|p| (p.0, p.1)).collect();
        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).title(" 3D PCA Scatter — ←→↑↓ rotate "))
            .x_bounds([x_min - x_pad, x_max + x_pad]).y_bounds([y_min - y_pad, y_max + y_pad])
            .paint(move |ctx| { ctx.draw(&Points { coords: &pts, color: Color::Green }); });
        frame.render_widget(canvas, area);
    } else {
        // 4D/5D: color by PC4
        let pc4_min = rotated.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
        let pc4_max = rotated.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);
        let pc4_range = (pc4_max - pc4_min).max(1e-10);
        let colors = [Color::Blue, Color::Cyan, Color::Green, Color::Yellow, Color::Red];
        let mut bins: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 5];
        for p in &rotated {
            let t = ((p.2 - pc4_min) / pc4_range).clamp(0.0, 0.9999);
            bins[(t * 5.0) as usize].push((p.0, p.1));
        }
        let title = if dims == 4 { " 4D Scatter — PC4→color " } else { " 5D Scatter — PC4→color PC5→brightness " };
        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).title(title))
            .x_bounds([x_min - x_pad, x_max + x_pad]).y_bounds([y_min - y_pad, y_max + y_pad])
            .paint(move |ctx| {
                for (i, bin) in bins.iter().enumerate() {
                    if !bin.is_empty() { ctx.draw(&Points { coords: bin, color: colors[i] }); }
                }
            });
        frame.render_widget(canvas, area);
    }
}
