// Copyright (c) nosqlbench contributors
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

/// Map from display order to VIEW_INFO_RAW index.
/// VIEW_INFO_RAW preserves the original authoring order; this mapping
/// reorders them for display (most impressive first).
const VIEW_INFO_ORDER: [usize; NUM_VIEWS] = [
    7, // V_PCA -> PCA scatter (was index 7)
    8, // V_DIMDIST -> Dimension distribution (was index 8)
    5, // V_LOADINGS -> PCA Loadings (was index 5)
    4, // V_EIGEN -> Eigenvalue analysis (was index 4)
    0, // V_NORMS -> Norms histogram (was index 0)
    1, // V_NORMCURVE -> Sorted norms (was index 1)
    2, // V_DISTS -> Distance histogram (was index 2)
    3, // V_DISTCURVE -> Sorted distances (was index 3)
    6, // V_VARBARS -> Variance bars (was index 6)
];

/// Per-view theory and interpretation descriptions (original authoring order).
const VIEW_INFO_RAW: &[&str] = &[
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
        "Use ↑/↓ arrows to adjust bin count for finer or coarser resolution.",
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
    // 4: Eigen (Scree / CumVar / LogDecay — toggled with m)
    concat!(
        "Eigenvalue Analysis (press m to cycle: Scree / CumVar / LogDecay)\n\n",
        "Three views of the same eigenvalue data, toggled with 'm':\n\n",
        "SCREE PLOT — eigenvalues largest to smallest:\n",
        "  Sharp elbow: low-dimensional manifold, PQ/compression works well.\n",
        "  Gradual decay: high-dimensional, PQ needs more sub-quantizers.\n\n",
        "CUMULATIVE VARIANCE — running sum as % of total (95% threshold):\n",
        "  95% at k=20 in 1024-dim: only 20 effective dimensions.\n",
        "  95% at k=500: data genuinely uses half its dimensions.\n\n",
        "LOG DECAY — ln(eigenvalue) vs component index:\n",
        "  Straight line: power-law decay, slope = effective dimensionality.\n",
        "  Plateau at bottom: noise floor.\n\n",
        "  Power-law decay (straight line on log plot): Common in natural data.\n",
        "  The slope indicates the effective dimensionality.\n\n",
        "Computation:\n",
        "  Eigenvalues computed via power iteration on the centered data matrix.\n",
        "  For each component k: iterate v ← (X^T X / n) v with deflation\n",
        "  against previously found eigenvectors. 15 iterations for the first 3\n",
        "  components, 5 for the rest. Eigenvalue = E[dot(Xv, Xv)] = v^T Sigma v.\n",
        "  Uses simsimd SIMD dot products and rayon parallelism.",
    ),
    // 5: PCA Loadings heatmap
    concat!(
        "PCA Loadings Heatmap\n\n",
        "A 2D grid showing how each original dimension contributes to each\n",
        "principal component. Columns are PCs, rows are original dimensions\n",
        "(grouped into bands when dimensionality is high).\n\n",
        "  Bright cells: Strong loading — this dimension heavily influences this PC.\n",
        "  Dark cells: Near-zero loading — this dimension is irrelevant to this PC.\n\n",
        "  Interpretation:\n",
        "  - If PC1 loads heavily on dims 0-50 and PC2 on dims 200-300,\n",
        "    these dimension ranges capture independent variance axes.\n",
        "  - Uniform loading across all dims means no single dimension\n",
        "    dominates — the PC captures a global pattern.\n",
        "  - Sparse loadings (few bright cells) suggest the data has\n",
        "    structure aligned with specific input features.\n\n",
        "  Color scale: magnitude of loading weight (absolute value).\n",
        "  Blue (low) → Cyan → Yellow → Red (high).\n\n",
        "  For high-dimensional data (>100 dims), dimensions are grouped\n",
        "  into bands and the max absolute loading in each band is shown.",
    ),
    // 6: Variance bars
    concat!(
        "Variance Explained per Principal Component\n\n",
        "Bar chart showing each PC's share of total variance.\n\n",
        "  Dominant first bar: PC1 captures most variance — data has a\n",
        "  strong primary axis. Common in text embeddings where the first\n",
        "  component often captures average document length or frequency.\n\n",
        "  Gradual decline: Variance is distributed across many PCs.\n",
        "  No single direction dominates. The data is genuinely high-dimensional.\n\n",
        "  Sharp drop after k bars: Only k dimensions carry signal.\n",
        "  Remaining PCs are noise. Product quantization with k sub-quantizers\n",
        "  should capture most of the distance structure.\n\n",
        "  Color coding:\n",
        "    Yellow — top 3 components (typically shown in PCA scatter)\n",
        "    Green — components explaining > 5% of variance\n",
        "    Gray — minor components (< 5% each)\n\n",
        "This view complements the Scree plot: the scree shows absolute\n",
        "eigenvalues (decay shape), while this shows relative contribution\n",
        "as percentages, making it easy to see how much each PC matters.",
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
    // 8: Dimension distribution
    concat!(
        "Per-Dimension Value Distribution\n\n",
        "Histogram of values for a single dimension across all sampled vectors.\n",
        "Navigate dimensions with ←/→ arrow keys.\n\n",
        "  Gaussian shape: This dimension's values follow a normal distribution,\n",
        "  typical for well-trained embedding models.\n\n",
        "  Bimodal or multimodal: Natural clusters along this axis.\n",
        "  Certain dimensions encode categorical information.\n\n",
        "  Spike at zero: Many vectors have zero in this dimension.\n",
        "  Common in sparse or ReLU-activated embeddings.\n\n",
        "  Uniform spread: No strong structure in this dimension.\n\n",
        "  Compare across dimensions to find which carry the most information\n",
        "  (wide spread) vs which are near-constant (tight spike).",
    ),
];

// View index constants — ordered for visual impact (most impressive first).
const V_PCA: usize = 0;
const V_DIMDIST: usize = 1;
const V_LOADINGS: usize = 2;
const V_EIGEN: usize = 3;
const V_NORMS: usize = 4;
const V_NORMCURVE: usize = 5;
const V_DISTS: usize = 6;
const V_DISTCURVE: usize = 7;
const V_VARBARS: usize = 8;

const VIEW_NAMES: &[&str] = &[
    "F1:PCA",
    "F2:DimDist",
    "F3:Loadings",
    "F4:Eigen",
    "F5:Norms",
    "F6:NormCurve",
    "F7:Distances",
    "F8:DistCurve",
    "F9:VarBars",
];
const NUM_VIEWS: usize = 9;

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
    /// Check if tracked norms indicate normalized vectors (SRD §18.3).
    ///
    /// Uses the f32 machine epsilon as a conservative default — covers the
    /// overwhelmingly common case and errs on the strict side for f16.
    fn is_normalized(&self, dim: usize) -> bool {
        if self.count < 10 { return false; }
        let threshold = 10.0 * 1.19e-7_f64 * (dim as f64).sqrt();
        let mean_eps = (self.mean - 1.0).abs();
        let max_eps = (self.min - 1.0).abs().max((self.max - 1.0).abs());
        mean_eps < threshold && max_eps < threshold * 5.0
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// How the explore session ended.
pub(super) enum ExploreExit {
    /// User pressed q or Ctrl-C — hard quit.
    Quit,
    /// User pressed Esc when idle — return to previous screen.
    Back,
}

pub(super) fn run_interactive_explore(
    source: &str, sample_size: usize, seed: u64, sample_mode: SampleMode,
) -> ExploreExit {
    eprintln!("Opening {}...", source);
    let reader = UnifiedReader::open(source);
    let total = reader.count();
    let dim = reader.dim();

    if total == 0 || dim == 0 {
        eprintln!("Error: no vector data found in '{}' (count={}, dim={})", source, total, dim);
        eprintln!("  This may mean the profile doesn't have base_vectors, or the data isn't cached yet.");
        eprintln!("  Try: veks datasets prebuffer --dataset {} --profile {}",
            source.split(':').next().unwrap_or(source),
            source.split(':').nth(1).unwrap_or("default"));
        // Brief pause so user can read the message before TUI takes over
        std::thread::sleep(std::time::Duration::from_secs(2));
        return ExploreExit::Back;
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
    let mut selected_dim: usize = 0; // for DimDist view
    let mut pc_axes: [usize; 5] = [0, 1, 2, 3, 4];
    let mut loadings_bar_mode: bool = true; // true=bar chart (default), false=heatmap
    let mut loadings_band_size: usize = 0;   // 0 = auto-fit to terminal height
    let mut loadings_scroll: usize = 0;      // vertical scroll offset (band index)
    let mut eigen_sub_mode: usize = 0;       // 0=scree, 1=cumvar, 2=log decay

    // ── Computation state (persists across restarts — old data stays visible) ──
    let mut compute_start;
    let mut vector_buf: Vec<f32> = Vec::new();
    let mut vectors_loaded: usize;
    let mut all_norms: Vec<f64> = Vec::new();
    let mut norm_stats;
    let mut last_cache_stats: Option<CacheStats> = None;
    let mut phase1_done;
    let mut sorted_norms: Vec<f64> = Vec::new();
    let mut sorted_norms_len: usize;
    let mut sorted_dists: Vec<f64> = Vec::new();
    let mut sorted_dists_len: usize;
    let mut all_dists: Vec<f32> = Vec::new();
    let mut dist_stats;
    let mut dist_rx: Option<mpsc::Receiver<DistBatch>>;
    let mut phase2_done;
    let mut total_pairs;
    let mut eigenvalues: Vec<f64> = Vec::new();
    let mut eigenvectors: Vec<Vec<f64>> = Vec::new();
    let mut eigen_rx: Option<mpsc::Receiver<EigenMsg>>;
    let mut phase3_done;
    let eigen_target = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(10usize.min(dim)));
    let mut avg_eigen_ms;
    let mut projected: Vec<[f64; 5]> = Vec::new();
    let mut proj_rx: Option<mpsc::Receiver<ProjectionMsg>>;
    let mut phase4_done;
    let mut status_msg;
    let mut final_elapsed: Option<f64>;
    let mut frame_count: usize = 0;
    let mut restart;
    let mut exit_reason = ExploreExit::Quit;
    let mut last_esc: Option<std::time::Instant> = None;
    // read_rx lives across restarts — reassigned each iteration
    let mut read_rx: mpsc::Receiver<ReadBatch>;

    // Outer loop: restarts computation when sample size changes.
    // Old data stays visible until new data arrives.
    loop {
    let indices = sample_indices(total, current_sample, seed, sample_mode, clump);

    // ── Phase 1: Read vectors + compute norms on background thread ──
    let (read_tx, new_read_rx) = mpsc::channel::<ReadBatch>();
    read_rx = new_read_rx;
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

    // Reset computation phases — but keep old rendered data visible.
    // The pending_clear flag defers clearing until the first new batch
    // arrives, preventing a blank frame between restarts.
    compute_start = Instant::now();
    let mut pending_clear = true;
    vectors_loaded = 0;
    norm_stats = WelfordStats::new();
    phase1_done = false;
    sorted_norms_len = 0;
    sorted_dists_len = 0;
    dist_stats = WelfordStats::new();
    dist_rx = None;
    phase2_done = false;
    total_pairs = (current_sample * 50).min(current_sample.saturating_mul(current_sample.saturating_sub(1)) / 2);
    eigen_rx = None;
    phase3_done = false;
    eigen_target.store(10usize.min(dim), std::sync::atomic::Ordering::Relaxed);
    avg_eigen_ms = 0.0;
    proj_rx = None;
    phase4_done = false;
    final_elapsed = None;
    restart = false;

    loop {
        if abort_flag.load(Ordering::Relaxed) {
            let _ = disable_raw_mode();
            let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
            eprintln!("Aborted.");
            std::process::exit(130);
        }

        // ── Drain Phase 1 (read + norms) ──
        // Limit batches per frame so the render updates at least once per second.
        if !phase1_done {
            let mut batches_this_frame = 0usize;
            loop {
                if batches_this_frame >= 4 { break; } // yield to render after ~2K vectors
                match read_rx.try_recv() {
                    Ok(batch) => {
                        // On first new batch, clear stale data from previous sample
                        if pending_clear {
                            pending_clear = false;
                            vector_buf.clear();
                            all_norms.clear();
                            all_dists.clear();
                            sorted_norms.clear();
                            sorted_dists.clear();
                            sorted_norms_len = 0;
                            sorted_dists_len = 0;
                            eigenvalues.clear();
                            eigenvectors.clear();
                            projected.clear();
                        }
                        vector_buf.extend_from_slice(&batch.vectors);
                        for &n in &batch.norms { norm_stats.update(n); }
                        all_norms.extend_from_slice(&batch.norms);
                        vectors_loaded += batch.count;
                        if let Some(cs) = batch.cache_stats { last_cache_stats = Some(cs); }
                        batches_this_frame += 1;
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
                            let eigen_target_clone = eigen_target.clone();
                            std::thread::spawn(move || {
                                let mut evecs: Vec<Vec<f64>> = Vec::new();
                                let mut ki = 0usize;
                                loop {
                                    let target = eigen_target_clone.load(std::sync::atomic::Ordering::Relaxed);
                                    if ki >= target || ki >= dim_c { break; }
                                    let iters = if ki < 3 { 15 } else { 5 };
                                    let mut v: Vec<f64> = (0..dim_c).map(|d| ((d * 7 + ki * 31 + 13) % 97) as f64 - 48.0).collect();
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
                                    if etx.send(EigenMsg { ki, eigenvalue, eigenvector: v, elapsed_ms }).is_err() {
                                        break; // receiver dropped
                                    }
                                    ki += 1;
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
                let mut batches = 0usize;
                loop {
                    if batches >= 4 { break; }
                    match rx.try_recv() {
                        Ok(batch) => {
                            for &d in &batch.distances { dist_stats.update(d as f64); }
                            all_dists.extend_from_slice(&batch.distances);
                            batches += 1;
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

        // Auto-deepen eigenvalues when loadings view is active and we have < 30 PCs.
        // The eigen thread checks the target atomically, so bumping it while
        // the thread is still running causes it to compute more.
        if view_mode == V_LOADINGS && eigenvalues.len() < 30 && eigenvalues.len() < dim {
            let desired = 30usize.min(dim);
            let current_target = eigen_target.load(std::sync::atomic::Ordering::Relaxed);
            if desired > current_target {
                eigen_target.store(desired, std::sync::atomic::Ordering::Relaxed);
                // If the thread already finished (phase3_done), un-mark it
                // so we keep draining. The thread may have exited though —
                // that's OK, we accept what we got.
                if phase3_done {
                    phase3_done = false;
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
        frame_count += 1;
        let all_done = phase1_done && phase2_done && phase3_done && phase4_done;
        let spinner = if all_done { " " } else {
            const SPIN: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            SPIN[frame_count % SPIN.len()]
        };
        status_msg = if !phase1_done {
            let rate = if elapsed > 0.0 { vectors_loaded as f64 / elapsed } else { 0.0 };
            let ci = last_cache_stats.as_ref().map(|cs| {
                let pct = if cs.total_chunks > 0 { 100.0 * cs.valid_chunks as f64 / cs.total_chunks as f64 } else { 0.0 };
                format!(" | cache {:.0}%", pct)
            }).unwrap_or_default();
            format!("{} Reading: {}/{} ({:.0}/s){}", spinner, vectors_loaded, current_sample, rate, ci)
        } else if !phase2_done || !phase3_done {
            let d_status = if phase2_done { format!("dists:{}", all_dists.len()) }
                else { format!("dists:{}/{}", all_dists.len(), total_pairs) };
            let et = eigen_target.load(std::sync::atomic::Ordering::Relaxed);
            let e_status = if phase3_done { format!("eigen:{}", eigenvalues.len()) }
                else {
                    let rem = et.saturating_sub(eigenvalues.len());
                    let eta = rem as f64 * avg_eigen_ms / 1000.0;
                    format!("eigen:{}/{} eta:{:.0}s", eigenvalues.len(), et, eta)
                };
            format!("{} {} | {}", spinner, d_status, e_status)
        } else if !phase4_done {
            format!("{} Projecting...", spinner)
        } else {
            let t = *final_elapsed.get_or_insert(elapsed);
            format!("  Ready — {} vectors, {:.1}s total", vectors_loaded, t)
        };

        // ── Stats line for current view ──
        let stats_line = match view_mode {
            V_PCA => {
                if !eigenvalues.is_empty() {
                    let total_ev: f64 = eigenvalues.iter().sum();
                    let pcts: Vec<f64> = eigenvalues.iter().take(5).map(|v| 100.0 * v / total_ev).collect();
                    format!("PC1:{:.1}% PC2:{:.1}% PC3:{:.1}% PC4:{:.1}% PC5:{:.1}%",
                        pcts.get(0).unwrap_or(&0.0), pcts.get(1).unwrap_or(&0.0),
                        pcts.get(2).unwrap_or(&0.0), pcts.get(3).unwrap_or(&0.0),
                        pcts.get(4).unwrap_or(&0.0))
                } else { String::new() }
            }
            V_DIMDIST => {
                let n = vectors_loaded;
                if n > 0 && selected_dim < dim {
                    let mut ds = WelfordStats::new();
                    for i in 0..n {
                        let idx = i * dim + selected_dim;
                        if idx < vector_buf.len() { ds.update(vector_buf[idx] as f64); }
                    }
                    format!("dim[{}]: mean={:.6} std={:.6} min={:.6} max={:.6}",
                        selected_dim, ds.mean, ds.std_dev(), ds.min, ds.max)
                } else { String::new() }
            }
            V_LOADINGS | V_EIGEN | V_VARBARS => {
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
            V_NORMS | V_NORMCURVE => {
                if norm_stats.count > 0 {
                    let verdict = if norm_stats.is_normalized(dim) { "NORMALIZED" } else { "not normalized" };
                    format!("mean={:.4} std={:.4} min={:.4} max={:.4} {}", norm_stats.mean, norm_stats.std_dev(), norm_stats.min, norm_stats.max, verdict)
                } else { String::new() }
            }
            V_DISTS | V_DISTCURVE => {
                if dist_stats.count > 0 {
                    let contrast = if dist_stats.min > 0.0 { (dist_stats.max - dist_stats.min) / dist_stats.min } else { f64::INFINITY };
                    let cv = if dist_stats.mean > 0.0 { dist_stats.std_dev() / dist_stats.mean } else { 0.0 };
                    format!("mean={:.4} std={:.4} contrast={:.1} CV={:.3}", dist_stats.mean, dist_stats.std_dev(), contrast, cv)
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
                let raw_idx = VIEW_INFO_ORDER.get(view_mode).copied().unwrap_or(0);
                let info_text = VIEW_INFO_RAW.get(raw_idx).unwrap_or(&"");
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
                let mut help = vec![
                    Line::from(Span::styled(
                        format!(" Keyboard Shortcuts — {}", VIEW_NAMES[view_mode]),
                        Style::default().fg(Color::Cyan))),
                    Line::from(""),
                    Line::from(" Navigation"),
                    Line::from("   F1-F9 / Tab / PgDn/PgUp        Switch view"),
                    Line::from("   /                               View theory & interpretation"),
                    Line::from("   ?                               Toggle this help"),
                    Line::from(""),
                    Line::from(" Sample"),
                    Line::from("   Space                           Double sample size"),
                    Line::from("   + / =                           Increase sample by 50%"),
                    Line::from("   r                               Reset sample size + rotations"),
                    Line::from("   Esc                             Stop processing / quit"),
                    Line::from("   q / Ctrl-C                      Quit"),
                ];
                // View-specific keys
                let view_keys: Vec<Line> = match view_mode {
                    V_PCA => vec![
                        Line::from(""),
                        Line::from(" PCA Scatter"),
                        Line::from("   ←→↑↓                           Rotate (Y/X axes)"),
                        Line::from("   a / d                           Rotate (Z axis)"),
                        Line::from("   w / s                           Rotate (W tilt)"),
                        Line::from("   c / C                           Slide PC window (PC1-4 → PC2-5 → ...)"),
                        Line::from("   x / X                           Rotate axis roles (X→Y→Z→color cycle)"),
                    ],
                    V_DIMDIST => vec![
                        Line::from(""),
                        Line::from(" Dimension Distribution"),
                        Line::from("   ←/→                             Navigate dimensions"),
                        Line::from("   Home / End                      First / last dimension"),
                        Line::from("   ↑/↓                             More / fewer bins"),
                    ],
                    V_LOADINGS => vec![
                        Line::from(""),
                        Line::from(" PCA Loadings"),
                        Line::from("   ↑/↓                             Increase / decrease resolution"),
                        Line::from("   Home / End                      Top / bottom (scroll)"),
                        Line::from("   m                               Toggle bar chart / heatmap"),
                    ],
                    V_EIGEN => vec![
                        Line::from(""),
                        Line::from(" Eigenvalue View"),
                        Line::from("   ↑/↓                             Cycle: Scree / CumVar / LogDecay"),
                        Line::from("   m                               Cycle mode (same)"),
                    ],
                    V_NORMS | V_DISTS => vec![
                        Line::from(""),
                        Line::from(" Histogram"),
                        Line::from("   ↑/↓                             More / fewer bins"),
                    ],
                    _ => vec![],
                };
                help.extend(view_keys);
                frame.render_widget(
                    Paragraph::new(help).block(Block::default().borders(Borders::ALL).title(" Help ")),
                    chunks[1],
                );
            } else { match view_mode {
                V_PCA => render_pca_scatter(frame, chunks[1], &projected, rot_y, rot_x, rot_z, rot_w, &pc_axes, 4),
                V_DIMDIST => {
                    let n = vectors_loaded;
                    if n > 0 && dim > 0 && selected_dim < dim {
                        let mut dim_values: Vec<f64> = Vec::with_capacity(n);
                        for i in 0..n {
                            let idx = i * dim + selected_dim;
                            if idx < vector_buf.len() {
                                dim_values.push(vector_buf[idx] as f64);
                            }
                        }
                        let mut dim_stats = WelfordStats::new();
                        for &v in &dim_values { dim_stats.update(v); }
                        let title = format!(" Dimension {} of {} ({} vectors) ", selected_dim, dim, n);
                        render_histogram(frame, chunks[1], &dim_values, &dim_stats, num_bins, &title, Color::Magenta);
                    }
                }
                V_LOADINGS => render_loadings(frame, chunks[1], &eigenvectors, &eigenvalues, loadings_bar_mode, loadings_band_size, &mut loadings_scroll),
                V_EIGEN => match eigen_sub_mode {
                    0 => render_scree(frame, chunks[1], &eigenvalues),
                    1 => render_cumulative(frame, chunks[1], &eigenvalues),
                    _ => render_log_decay(frame, chunks[1], &eigenvalues),
                },
                V_NORMS => render_histogram(frame, chunks[1], &all_norms, &norm_stats, num_bins, " Norm Distribution ", Color::Green),
                V_NORMCURVE => render_presorted_curve(frame, chunks[1], &sorted_norms, " Sorted Norms ", Color::Green),
                V_DISTS => {
                    let dists_f64: Vec<f64> = all_dists.iter().map(|&d| d as f64).collect();
                    render_histogram(frame, chunks[1], &dists_f64, &dist_stats, num_bins, " Distance Distribution (L2) ", Color::Cyan);
                }
                V_DISTCURVE => {
                    render_presorted_curve(frame, chunks[1], &sorted_dists, " Sorted Distances ", Color::Cyan);
                }
                V_VARBARS => render_variance_bars(frame, chunks[1], &eigenvalues),
                _ => {}
            } } // close match + else

            frame.render_widget(Paragraph::new(Span::styled(&stats_line, Style::default().fg(Color::White))), chunks[2]);

            // View selector line
            let vi: String = VIEW_NAMES.iter().enumerate()
                .map(|(i, n)| if i == view_mode { format!("[{}]", n) } else { n.to_string() })
                .collect::<Vec<_>>().join(" ");
            frame.render_widget(Paragraph::new(Span::styled(
                format!(" {} | Tab/PgDn/PgUp | /: info | ?: help | q quit", vi),
                Style::default().fg(Color::DarkGray))), chunks[3]);

            // View-specific controls line
            let view_controls: String = match view_mode {
                V_PCA => format!(" ←→↑↓ a/d w/s: rotate | c/C: slide PCs | x/X: swap axes [X:PC{} Y:PC{} Z:PC{} color:PC{}] | r +/Space",
                    pc_axes[0]+1, pc_axes[1]+1, pc_axes[2]+1, pc_axes[3]+1),
                V_DIMDIST => format!(" ←/→: dimension ({}/{}) | Home/End: first/last | ↑↓: bins | b/B: bins",
                    selected_dim, dim),
                V_LOADINGS => format!(" ↑↓: resolution | m: toggle {} | Home/End: scroll ({} dims/row)",
                    if loadings_bar_mode { "heatmap" } else { "bar chart" },
                    if loadings_band_size == 0 { "auto".to_string() } else { loadings_band_size.to_string() }),
                V_EIGEN => {
                    let mode_name = ["Scree", "CumVar", "LogDecay"][eigen_sub_mode];
                    format!(" ↑↓/m: cycle mode ({}) | +/Space: increase sample", mode_name)
                }
                V_NORMS | V_DISTS => " ↑↓: fewer/more bins | +/Space: increase sample".into(),
                V_NORMCURVE | V_DISTCURVE => " +/Space: increase sample".into(),
                V_VARBARS => String::new(),
                _ => String::new(),
            };
            frame.render_widget(Paragraph::new(Span::styled(
                &view_controls, Style::default().fg(Color::DarkGray))), chunks[4]);

            let sc = if all_done { Color::Green } else { Color::Yellow };
            frame.render_widget(Paragraph::new(Span::styled(&status_msg, Style::default().fg(sc))), chunks[5]);
        }).unwrap();

        // ── Events ──
        let poll_ms = if all_done { 100 } else { 10 };
        if event::poll(std::time::Duration::from_millis(poll_ms)).unwrap() {
            match event::read().unwrap() {
                Event::Key(key) => {
                    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) { break; }
                    match key.code {
                        KeyCode::Char('q') => {
                            exit_reason = ExploreExit::Quit;
                            break;
                        }
                        KeyCode::Esc => {
                            if !all_done {
                                // First Esc: stop processing
                                phase1_done = true;
                                phase2_done = true;
                                phase3_done = true;
                                phase4_done = true;
                                dist_rx = None;
                                eigen_rx = None;
                                proj_rx = None;
                                last_esc = None;
                            } else {
                                // Double-tap Esc to exit
                                let now = std::time::Instant::now();
                                if let Some(prev) = last_esc {
                                    if now.duration_since(prev).as_millis() < 500 {
                                        exit_reason = ExploreExit::Back;
                                        break;
                                    }
                                }
                                last_esc = Some(now);
                            }
                        }
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
                        // Loadings resolution moved to Up/Down arrows below
                        KeyCode::Char('m') if view_mode == V_EIGEN => {
                            eigen_sub_mode = (eigen_sub_mode + 1) % 3;
                        }
                        KeyCode::Char('m') if view_mode == V_LOADINGS => {
                            loadings_bar_mode = !loadings_bar_mode;
                        }
                        // b/B kept as aliases for bin adjustment in histogram views
                        KeyCode::Char('b') if matches!(view_mode, V_DIMDIST | V_NORMS | V_DISTS) => {
                            if num_bins == 0 { num_bins = 100; }
                            num_bins = num_bins.saturating_sub(5).max(10);
                        }
                        KeyCode::Char('B') if matches!(view_mode, V_DIMDIST | V_NORMS | V_DISTS) => {
                            if num_bins == 0 { num_bins = 100; }
                            num_bins = (num_bins + 5).min(500);
                        }
                        // Rotation axes (PCA scatter views only)
                        KeyCode::Left if view_mode == V_PCA => { rot_y -= 0.1; }
                        KeyCode::Right if view_mode == V_PCA => { rot_y += 0.1; }
                        KeyCode::Up if view_mode == V_PCA => { rot_x -= 0.1; }
                        KeyCode::Down if view_mode == V_PCA => { rot_x += 0.1; }
                        KeyCode::Left if view_mode == V_DIMDIST => { selected_dim = selected_dim.saturating_sub(1); }
                        KeyCode::Right if view_mode == V_DIMDIST => { if selected_dim + 1 < dim { selected_dim += 1; } }
                        KeyCode::Home if view_mode == V_DIMDIST => { selected_dim = 0; }
                        KeyCode::End if view_mode == V_DIMDIST => { selected_dim = dim.saturating_sub(1); }
                        KeyCode::Up if view_mode == V_LOADINGS => {
                            // Increase resolution (fewer dims per row)
                            if loadings_band_size == 0 {
                                let th = crossterm::terminal::size().map(|(_, h)| h as usize).unwrap_or(40);
                                loadings_band_size = (dim as f64 / th.saturating_sub(8) as f64).ceil() as usize;
                                loadings_band_size = loadings_band_size.next_power_of_two();
                            }
                            loadings_band_size = (loadings_band_size / 2).max(1);
                        }
                        KeyCode::Down if view_mode == V_LOADINGS => {
                            // Decrease resolution (more dims per row)
                            if loadings_band_size == 0 {
                                let th = crossterm::terminal::size().map(|(_, h)| h as usize).unwrap_or(40);
                                loadings_band_size = (dim as f64 / th.saturating_sub(8) as f64).ceil() as usize;
                                loadings_band_size = loadings_band_size.next_power_of_two();
                            }
                            loadings_band_size = (loadings_band_size * 2).min(dim);
                        }
                        // Histogram views: Up/Down adjusts bin count
                        KeyCode::Up if matches!(view_mode, V_DIMDIST | V_NORMS | V_DISTS) => {
                            if num_bins == 0 { num_bins = 100; }
                            num_bins = (num_bins + 5).min(500);
                        }
                        KeyCode::Down if matches!(view_mode, V_DIMDIST | V_NORMS | V_DISTS) => {
                            if num_bins == 0 { num_bins = 100; }
                            num_bins = num_bins.saturating_sub(5).max(10);
                        }
                        // Eigen: Up/Down cycles sub-mode
                        KeyCode::Up if view_mode == V_EIGEN => {
                            eigen_sub_mode = if eigen_sub_mode == 0 { 2 } else { eigen_sub_mode - 1 };
                        }
                        KeyCode::Down if view_mode == V_EIGEN => {
                            eigen_sub_mode = (eigen_sub_mode + 1) % 3;
                        }
                        KeyCode::Home if view_mode == V_LOADINGS => { loadings_scroll = 0; }
                        KeyCode::End if view_mode == V_LOADINGS => { loadings_scroll = usize::MAX; } // clamped in render
                        KeyCode::Char('a') if view_mode == V_PCA => { rot_z -= 0.1; }
                        KeyCode::Char('d') if view_mode == V_PCA => { rot_z += 0.1; }
                        KeyCode::Char('w') if view_mode == V_PCA => { rot_w -= 0.1; }
                        KeyCode::Char('s') if view_mode == V_PCA => { rot_w += 0.1; }
                        KeyCode::Char('r') => {
                            rot_y = 0.0; rot_x = 0.0; rot_z = 0.0; rot_w = 0.0;
                            if current_sample != sample_size.min(total) {
                                current_sample = sample_size.min(total);
                                restart = true;
                                break;
                            }
                        }
                        // c/C: slide the component window (PC1-4 → PC2-5 → PC3-6...)
                        KeyCode::Char('c') if view_mode == V_PCA => {
                            let max_pc = eigenvalues.len().max(5);
                            for a in pc_axes.iter_mut() { *a = (*a + 1) % max_pc; }
                        }
                        KeyCode::Char('C') if view_mode == V_PCA => {
                            let max_pc = eigenvalues.len().max(5);
                            for a in pc_axes.iter_mut() { *a = (*a + max_pc - 1) % max_pc; }
                        }
                        // x/X: rotate axis assignment within the current set
                        // x: X→Y→Z→color cycle (rotate display roles forward)
                        // X: reverse
                        KeyCode::Char('x') if view_mode == V_PCA => {
                            let tmp = pc_axes[0];
                            pc_axes[0] = pc_axes[1];
                            pc_axes[1] = pc_axes[2];
                            pc_axes[2] = pc_axes[3];
                            pc_axes[3] = tmp;
                        }
                        KeyCode::Char('X') if view_mode == V_PCA => {
                            let tmp = pc_axes[3];
                            pc_axes[3] = pc_axes[2];
                            pc_axes[2] = pc_axes[1];
                            pc_axes[1] = pc_axes[0];
                            pc_axes[0] = tmp;
                        }
                        // Page Up/Down for view switching
                        KeyCode::PageDown => { view_mode = (view_mode + 1) % NUM_VIEWS; }
                        KeyCode::PageUp => { view_mode = (view_mode + NUM_VIEWS - 1) % NUM_VIEWS; }
                        // Function keys F1-F8 as view shortcuts
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
                Event::Resize(_, _) => {
                    // Terminal resized — reset auto-sizing so the next
                    // frame recalculates layouts for the new dimensions.
                    if num_bins == 0 { /* auto bins recalculated each frame */ }
                    if loadings_band_size == 0 { /* auto band recalculated each frame */ }
                    sorted_norms_len = 0; // force re-downsample
                    sorted_dists_len = 0;
                }
                _ => {}
            }
        }
    } // end inner loop

    if !restart { break; }

    } // end 'compute loop

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    exit_reason
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

fn render_histogram(
    frame: &mut ratatui::Frame, area: ratatui::layout::Rect,
    values: &[f64], stats: &WelfordStats, num_bins: usize, title: &str, color: Color,
) {
    if values.is_empty() {
        frame.render_widget(Block::default().borders(Borders::ALL).title(title), area);
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
        frame.render_widget(Block::default().borders(Borders::ALL).title(title), area);
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
        frame.render_widget(Block::default().borders(Borders::ALL).title(title), area);
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
        frame.render_widget(Block::default().borders(Borders::ALL).title(" Scree Plot "), area);
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
        frame.render_widget(Block::default().borders(Borders::ALL).title(" Cumulative Variance "), area);
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
        frame.render_widget(Block::default().borders(Borders::ALL).title(" Log Eigenvalue Decay "), area);
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

fn render_loadings(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    eigenvectors: &[Vec<f64>],
    eigenvalues: &[f64],
    bar_mode: bool,
    manual_band_size: usize,
    scroll: &mut usize,
) {
    use ratatui::text::{Line, Span};

    if eigenvectors.is_empty() {
        frame.render_widget(
            Block::default().borders(Borders::ALL).title(" PCA Loadings "),
            area,
        );
        return;
    }

    let num_pcs = eigenvectors.len();
    let dim = eigenvectors[0].len();
    let inner_h = area.height.saturating_sub(2) as usize;
    let inner_w = area.width.saturating_sub(2) as usize;
    if inner_h < 3 || inner_w < 10 { return; }

    // Reserve rows: 1 header + 1 legend (heatmap) or 1 header (bars)
    let legend_rows = if bar_mode { 0 } else { 1 };
    let data_rows = inner_h.saturating_sub(1 + legend_rows);
    let band_size = if manual_band_size > 0 {
        manual_band_size.max(1).min(dim)
    } else {
        // Auto-fit: exactly fill the available vertical space
        (dim as f64 / data_rows as f64).ceil() as usize
    }.max(1);
    let num_bands = (dim + band_size - 1) / band_size;

    let total_ev: f64 = eigenvalues.iter().sum();

    let global_max = eigenvectors.iter()
        .flat_map(|v| v.iter())
        .map(|x| x.abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);

    // Helper: compute max absolute loading for a band in a PC
    let band_loading = |pc: usize, d_start: usize, d_end: usize| -> f64 {
        (d_start..d_end)
            .filter_map(|d| eigenvectors.get(pc).and_then(|v| v.get(d)))
            .map(|x| x.abs())
            .fold(0.0f64, f64::max)
    };

    let label_w = if dim >= 1000 { 10 } else { 8 };

    // Clamp scroll to valid range
    let max_scroll = num_bands.saturating_sub(data_rows);
    if *scroll > max_scroll { *scroll = max_scroll; }

    let band_label = |d_start: usize, d_end: usize| -> String {
        if band_size == 1 {
            format!("{:<w$}", format!("d{}", d_start), w = label_w)
        } else {
            format!("{:<w$}", format!("d{}-{}", d_start, d_end - 1), w = label_w)
        }
    };

    if bar_mode {
        let bar_w_per_pc = ((inner_w - label_w) / num_pcs.max(1)).max(4);
        let max_pcs = ((inner_w - label_w) / bar_w_per_pc).min(num_pcs);
        let bar_fill = bar_w_per_pc.saturating_sub(1);

        // Header
        let mut header_spans = vec![Span::styled(
            format!("{:<w$}", "", w = label_w),
            Style::default().fg(Color::DarkGray),
        )];
        for pc in 0..max_pcs {
            let pct = if total_ev > 0.0 { 100.0 * eigenvalues.get(pc).unwrap_or(&0.0) / total_ev } else { 0.0 };
            header_spans.push(Span::styled(
                format!("{:<w$}", format!("PC{} {:.0}%", pc + 1, pct), w = bar_w_per_pc),
                Style::default().fg(Color::Yellow),
            ));
        }

        let mut lines = vec![Line::from(header_spans)];
        for band in *scroll..(*scroll + data_rows).min(num_bands) {
            let d_start = band * band_size;
            let d_end = (d_start + band_size).min(dim);
            let mut spans = vec![Span::styled(band_label(d_start, d_end), Style::default().fg(Color::DarkGray))];

            for pc in 0..max_pcs {
                let intensity = band_loading(pc, d_start, d_end) / global_max;
                let filled = (intensity * bar_fill as f64).round() as usize;
                let color = loading_color(intensity);

                // Show numeric value if bar is wide enough (>= 6 chars)
                if bar_fill >= 6 {
                    let val_str = format!("{:.2}", intensity);
                    let val_len = val_str.len();
                    if filled >= val_len + 1 {
                        let bar = format!("{}{}{} ",
                            "█".repeat(filled - val_len),
                            val_str,
                            " ".repeat(bar_fill.saturating_sub(filled)));
                        spans.push(Span::styled(bar, Style::default().fg(color)));
                    } else {
                        let bar = format!("{}{} ",
                            "█".repeat(filled),
                            " ".repeat(bar_fill.saturating_sub(filled)));
                        spans.push(Span::styled(bar, Style::default().fg(color)));
                    }
                } else {
                    let bar = format!("{}{} ",
                        "█".repeat(filled),
                        " ".repeat(bar_fill.saturating_sub(filled)));
                    spans.push(Span::styled(bar, Style::default().fg(color)));
                }
            }
            lines.push(Line::from(spans));
        }

        let title = format!(" PCA Loadings (bars) — {} dims x {} PCs (band={}) ", dim, max_pcs, band_size);
        frame.render_widget(Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(title)), area);
    } else {
        // Heatmap mode — scale to fill the available width with no gaps.
        let data_w = inner_w.saturating_sub(label_w);
        let max_pcs = num_pcs.min(data_w); // at least 1 char per PC
        let pc_col_w = if max_pcs > 0 { data_w / max_pcs } else { 1 };
        let max_pcs = if pc_col_w > 0 { (data_w / pc_col_w).min(num_pcs) } else { 0 };

        // Header with PC variance percentages
        let mut header_spans = vec![Span::styled(
            format!("{:<w$}", "dim", w = label_w),
            Style::default().fg(Color::DarkGray),
        )];
        for pc in 0..max_pcs {
            let pct = if total_ev > 0.0 { 100.0 * eigenvalues.get(pc).unwrap_or(&0.0) / total_ev } else { 0.0 };
            header_spans.push(Span::styled(
                format!("{:>w$}", format!("{:.0}%", pct), w = pc_col_w),
                Style::default().fg(Color::Yellow),
            ));
        }

        let mut lines = vec![Line::from(header_spans)];
        for band in *scroll..(*scroll + data_rows).min(num_bands) {
            let d_start = band * band_size;
            let d_end = (d_start + band_size).min(dim);
            let mut spans = vec![Span::styled(band_label(d_start, d_end), Style::default().fg(Color::DarkGray))];

            for pc in 0..max_pcs {
                let intensity = band_loading(pc, d_start, d_end) / global_max;
                let color = loading_color(intensity);
                let block = loading_char(intensity);
                let fill: String = std::iter::repeat(block).take(pc_col_w).collect();
                spans.push(Span::styled(fill, Style::default().fg(color)));
            }
            lines.push(Line::from(spans));
        }

        // Legend row
        let legend = Line::from(vec![
            Span::styled(format!("{:<w$}", "", w = label_w), Style::default()),
            Span::styled("· ", Style::default().fg(Color::DarkGray)),
            Span::styled("<0.1 ", Style::default().fg(Color::DarkGray)),
            Span::styled("░ ", Style::default().fg(Color::Blue)),
            Span::styled("<0.3 ", Style::default().fg(Color::Blue)),
            Span::styled("▒ ", Style::default().fg(Color::Cyan)),
            Span::styled("<0.5 ", Style::default().fg(Color::Cyan)),
            Span::styled("▓ ", Style::default().fg(Color::Yellow)),
            Span::styled("<0.7 ", Style::default().fg(Color::Yellow)),
            Span::styled("█ ", Style::default().fg(Color::Red)),
            Span::styled(">=0.7", Style::default().fg(Color::Red)),
        ]);
        lines.push(legend);

        let title = format!(" PCA Loadings — {} dims x {} PCs (band={}) ", dim, max_pcs, band_size);
        frame.render_widget(Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(title)), area);
    }
}

/// Map loading intensity [0,1] to a color ramp: Blue → Cyan → Yellow → Red.
fn loading_color(t: f64) -> Color {
    if t < 0.25 {
        Color::DarkGray
    } else if t < 0.5 {
        Color::Blue
    } else if t < 0.7 {
        Color::Cyan
    } else if t < 0.85 {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Map loading intensity to a block character for density.
fn loading_char(t: f64) -> &'static str {
    if t < 0.1 { "·" }
    else if t < 0.3 { "░" }
    else if t < 0.5 { "▒" }
    else if t < 0.7 { "▓" }
    else { "█" }
}

fn render_variance_bars(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, eigenvalues: &[f64]) {
    use ratatui::widgets::{BarChart, Bar, BarGroup};

    if eigenvalues.is_empty() {
        frame.render_widget(
            Block::default().borders(Borders::ALL).title(" Variance Explained per PC "),
            area,
        );
        return;
    }

    let total: f64 = eigenvalues.iter().sum();
    if total == 0.0 { return; }

    let bars: Vec<Bar> = eigenvalues.iter().enumerate().map(|(i, &v)| {
        let pct = 100.0 * v / total;
        Bar::default()
            .value(pct as u64)
            .label(format!("PC{}", i + 1).into())
            .style(Style::default().fg(if i < 3 {
                Color::Yellow
            } else if pct > 5.0 {
                Color::Green
            } else {
                Color::DarkGray
            }))
            .text_value(format!("{:.1}%", pct))
    }).collect();

    let bar_group = BarGroup::default().bars(&bars);
    let chart = BarChart::default()
        .block(Block::default().borders(Borders::ALL)
            .title(" Variance Explained per Principal Component "))
        .data(bar_group)
        .bar_width(6)
        .bar_gap(1)
        .value_style(Style::default().fg(Color::White));
    frame.render_widget(chart, area);
}

fn render_pca_scatter(
    frame: &mut ratatui::Frame, area: ratatui::layout::Rect,
    projected: &[[f64; 5]], rot_y: f64, rot_x: f64, rot_z: f64, rot_w: f64,
    pc_axes: &[usize; 5], dims: usize,
) {
    if projected.is_empty() {
        let title = format!(" {}D Scatter ", dims);
        frame.render_widget(Block::default().borders(Borders::ALL).title(title), area);
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
