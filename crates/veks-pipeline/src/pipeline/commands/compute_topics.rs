// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `compute topics` — a hierarchical spherical k-means topic model over
//! a vector facet, and the assignment of every vector to it.
//!
//! The model is fitted on a sample and then every vector of the base is
//! assigned by descent: an argmax over the top level, then over the
//! chosen cluster's children, and so on. Assignment is a pure function
//! of the vector and the fitted centroids, so it is deterministic and
//! reproducible from the published centroids alone.
//!
//! Three properties are load-bearing and are stated where they are
//! enforced:
//!
//! - **Spherical.** Vectors are unit-normalised, so cosine is the inner
//!   product and every argmax here is an argmax of a dot product.
//!   Centroids are re-normalised after each update; without that they
//!   drift off the sphere and the argmax stops corresponding to cosine.
//! - **Deterministic under threading.** Forming a centroid is a
//!   floating-point reduction whose result depends on summation order.
//!   Every reduction here runs over fixed-size chunks of rows and folds
//!   the chunk partials in chunk order, so the fit is bit-identical for
//!   any thread count.
//! - **Greedy descent, acknowledged.** Descending 10 → 30 → 33 costs 73
//!   inner products per vector instead of 10,000 and is therefore not
//!   a flat argmax over the leaf centroids. The margin output — the
//!   distance to the chosen leaf and to its best sibling — is what makes
//!   that discrepancy measurable.
//!
//! See the topic-stratified predicate SRD, §9.1–9.2 and §10.1.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use half::f16;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use vectordata::io::{StreamReclaim, XvecReader};

use crate::pipeline::command::{
    ArtifactManifest, ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, ResourceDesc, Status, StreamContext, render_options_table,
};
use crate::pipeline::shard_write::{FacetWriter, cap_for_output};

use super::compute_knn_stdarch::{DistFn, select_dot_fn};
use super::source_window::resolve_path;

/// `compute topics`.
pub struct ComputeTopicsOp;

/// Factory used by the pipeline command registry.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeTopicsOp)
}

/// Rows per parallel chunk in every reduction. Fixed, never derived
/// from the thread count, which is what makes the fit deterministic.
const CHUNK_ROWS: usize = 4096;

/// Vectors per batch of the assignment pass.
const ASSIGN_BATCH: usize = 16_384;

/// Rounds of empty-cluster repair per iteration.
const REPAIR_ROUNDS: usize = 3;

/// Report schema version, written beside the centroids.
const MODEL_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the sample rows are chosen from the sample facet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleOrder {
    /// The first `sample-size` rows. Right when the facet is already a
    /// uniform shuffle of the population, as `profiles/base/*` is.
    Prefix,
    /// `sample-size` rows spaced evenly over the facet. Right when the
    /// facet is in corpus order, where a prefix would be one corner of
    /// the corpus.
    Strided,
}

impl SampleOrder {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "prefix" => Ok(SampleOrder::Prefix),
            "strided" => Ok(SampleOrder::Strided),
            other => Err(format!(
                "sample-order: expected `prefix` or `strided`, got `{}`",
                other
            )),
        }
    }
}

/// Everything the fit needs, parsed from the options.
#[derive(Debug, Clone, PartialEq)]
pub struct FitConfig {
    /// Branching per level, outermost first.
    pub levels: Vec<usize>,
    /// Iteration cap per k-means run.
    pub iterations: usize,
    /// Mean centroid movement (in cosine distance) below which a run
    /// has converged.
    pub tolerance: f32,
    /// Seed for k-means++ and sampling.
    pub seed: u64,
}

impl FitConfig {
    /// Parse `levels` as `10,30,33`.
    pub fn parse_levels(spec: &str) -> Result<Vec<usize>, String> {
        let mut out = Vec::new();
        for raw in spec.split(',') {
            let t = raw.trim();
            if t.is_empty() {
                continue;
            }
            let k: usize = t
                .parse()
                .map_err(|_| format!("levels: `{}` is not a positive integer", t))?;
            if k == 0 {
                return Err("levels: every level needs at least one cluster".into());
            }
            out.push(k);
        }
        if out.is_empty() {
            return Err("levels: at least one level is required".into());
        }
        let mut total: usize = 1;
        for k in &out {
            total = total
                .checked_mul(*k)
                .ok_or_else(|| "levels: cluster count overflows".to_string())?;
        }
        if total > u16::MAX as usize {
            return Err(format!(
                "levels: {} leaf clusters exceed the {} an assignment code can name",
                total,
                u16::MAX
            ));
        }
        Ok(out)
    }

    /// Clusters at each level: the running product of branchings.
    pub fn clusters_per_level(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.levels.len());
        let mut n = 1;
        for k in &self.levels {
            n *= k;
            out.push(n);
        }
        out
    }

    /// Centroids across all levels.
    pub fn total_centroids(&self) -> usize {
        self.clusters_per_level().iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// One fitted level: `clusters` centroids of `dim`, row-major, where
/// cluster `c` at level `l > 0` is child `c % branching` of cluster
/// `c / branching` at level `l - 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelModel {
    pub branching: usize,
    pub clusters: usize,
    pub centroids: Vec<f32>,
    /// A cluster no sample row landed in even after repair. Kept in
    /// the layout so codes stay positional, never chosen by descent.
    pub empty: Vec<bool>,
    pub runs: Vec<RunStats>,
}

impl LevelModel {
    fn centroid(&self, c: usize, dim: usize) -> &[f32] {
        &self.centroids[c * dim..(c + 1) * dim]
    }
}

/// The fitted hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct TopicModel {
    pub dim: usize,
    pub levels: Vec<LevelModel>,
}

impl TopicModel {
    /// Assign one unit vector by greedy descent. Writes the code at
    /// every level into `codes` and returns the leaf margin: cosine
    /// distance to the chosen leaf and to its best non-empty sibling.
    pub(crate) fn descend(&self, v: &[f32], dot: DistFn, codes: &mut [u16]) -> (f32, f32) {
        let mut parent = 0usize;
        let mut margin = (0.0f32, 0.0f32);
        for (l, level) in self.levels.iter().enumerate() {
            let (start, end) = if l == 0 {
                (0, level.clusters)
            } else {
                (parent * level.branching, (parent + 1) * level.branching)
            };
            let mut best = usize::MAX;
            let mut best_dot = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for c in start..end {
                if level.empty[c] {
                    continue;
                }
                let d = dot(v, level.centroid(c, self.dim));
                if d > best_dot {
                    second = best_dot;
                    best_dot = d;
                    best = c;
                } else if d > second {
                    second = d;
                }
            }
            debug_assert!(best != usize::MAX, "a parent always has a non-empty child");
            codes[l] = best as u16;
            parent = best;
            let runner_up = if second.is_finite() { second } else { best_dot };
            margin = (1.0 - best_dot, 1.0 - runner_up);
        }
        margin
    }
}

/// What one k-means run did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunStats {
    /// Rows the run fitted.
    pub rows: u64,
    /// Iterations executed before the cap or convergence.
    pub iterations: u32,
    /// Mean centroid movement at the last iteration.
    pub final_movement: f32,
    /// Empty-cluster repairs performed.
    pub repairs: u32,
    /// Clusters left empty.
    pub empty: u32,
}

/// The per-level summary written to the model report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelReport {
    pub branching: usize,
    pub clusters: usize,
    pub empty: usize,
    pub runs: usize,
    pub converged: usize,
    pub max_final_movement: f32,
    pub repairs: u32,
}

/// The report written beside the centroids: what was fitted, from
/// what, with which seed, and how each level converged. Together with
/// the centroid file this is what a third party needs to reproduce
/// the labelling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicModelReport {
    pub schema_version: u32,
    pub dim: usize,
    pub levels: Vec<usize>,
    pub total_centroids: usize,
    pub sample: String,
    pub sample_size: usize,
    pub sample_order: SampleOrder,
    pub seed: u64,
    pub iterations: usize,
    pub tolerance: f32,
    pub normalize: bool,
    pub kernel: String,
    pub fit_seconds: f64,
    pub per_level: Vec<LevelReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignment: Option<AssignmentReport>,
}

/// What the assignment pass did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssignmentReport {
    pub base: String,
    pub records: u64,
    pub seconds: f64,
    pub margin_written: bool,
}

// ---------------------------------------------------------------------------
// k-means
// ---------------------------------------------------------------------------

#[inline]
fn row(sample: &[f32], dim: usize, i: u32) -> &[f32] {
    let i = i as usize;
    &sample[i * dim..(i + 1) * dim]
}

/// Scale `v` to unit length in place. A zero vector is left as it is:
/// its dot with everything is zero and it lands in the first live
/// cluster, which is as good a place as any.
fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// A seed for one k-means run, distinct per level and parent so runs
/// are independent but reproducible.
fn run_seed(seed: u64, level: usize, parent: usize) -> u64 {
    // splitmix64 over the tuple.
    let mut z = seed
        .wrapping_add((level as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((parent as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// k-means++ seeding over `rows`: the first centroid uniformly, each
/// next one with probability proportional to its cosine distance to
/// the nearest centroid chosen so far.
fn kmeans_plus_plus(
    sample: &[f32],
    dim: usize,
    rows: &[u32],
    k: usize,
    rng: &mut Xoshiro256PlusPlus,
    dot: DistFn,
) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * dim);
    let first = rows[rng.random_range(0..rows.len())];
    centroids.extend_from_slice(row(sample, dim, first));
    let mut min_d: Vec<f32> = vec![f32::INFINITY; rows.len()];
    for chosen in 0..k {
        let c = &centroids[chosen * dim..(chosen + 1) * dim];
        // Refresh the nearest-chosen distance in fixed chunks; each
        // row is independent, so this is deterministic.
        min_d
            .par_chunks_mut(CHUNK_ROWS)
            .zip(rows.par_chunks(CHUNK_ROWS))
            .for_each(|(d, r)| {
                for (di, &ri) in d.iter_mut().zip(r) {
                    let dist = (1.0 - dot(row(sample, dim, ri), c)).max(0.0);
                    if dist < *di {
                        *di = dist;
                    }
                }
            });
        if chosen + 1 == k {
            break;
        }
        let total: f64 = min_d.iter().map(|d| *d as f64).sum();
        let pick = if total <= 0.0 {
            // Every remaining row coincides with a chosen centroid;
            // any row will do, and the first is the deterministic one.
            0
        } else {
            let mut target = rng.random::<f64>() * total;
            let mut pick = rows.len() - 1;
            for (i, d) in min_d.iter().enumerate() {
                target -= *d as f64;
                if target <= 0.0 {
                    pick = i;
                    break;
                }
            }
            pick
        };
        centroids.extend_from_slice(row(sample, dim, rows[pick]));
    }
    centroids
}

/// Nearest live centroid of `v`, by inner product; ties go to the
/// lowest index.
#[inline]
fn nearest(v: &[f32], centroids: &[f32], dim: usize, k: usize, live: &[bool], dot: DistFn) -> u32 {
    let mut best = u32::MAX;
    let mut best_dot = f32::NEG_INFINITY;
    for c in 0..k {
        if !live[c] {
            continue;
        }
        let d = dot(v, &centroids[c * dim..(c + 1) * dim]);
        if d > best_dot {
            best_dot = d;
            best = c as u32;
        }
    }
    best
}

/// One spherical k-means run over `rows`. Returns the centroids
/// (`k × dim`, unit length), which clusters are empty, each row's
/// cluster, and the run's statistics.
fn kmeans(
    sample: &[f32],
    dim: usize,
    rows: &[u32],
    k: usize,
    cfg: &FitConfig,
    seed: u64,
    dot: DistFn,
) -> (Vec<f32>, Vec<bool>, Vec<u32>, RunStats) {
    let n = rows.len();
    let mut stats = RunStats {
        rows: n as u64,
        iterations: 0,
        final_movement: 0.0,
        repairs: 0,
        empty: 0,
    };
    // Degenerate: fewer rows than clusters. Every row is its own
    // centroid and the rest stay empty; nothing to iterate.
    if n <= k {
        let mut centroids = vec![0.0f32; k * dim];
        let mut empty = vec![true; k];
        let mut assign = Vec::with_capacity(n);
        for (c, &r) in rows.iter().enumerate() {
            centroids[c * dim..(c + 1) * dim].copy_from_slice(row(sample, dim, r));
            empty[c] = false;
            assign.push(c as u32);
        }
        stats.empty = empty.iter().filter(|e| **e).count() as u32;
        return (centroids, empty, assign, stats);
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut centroids = kmeans_plus_plus(sample, dim, rows, k, &mut rng, dot);
    let mut live = vec![true; k];
    let mut assign: Vec<u32> = vec![0; n];
    let mut counts: Vec<u64> = vec![0; k];
    let mut sums: Vec<f64> = vec![0.0; k * dim];
    let repair_singletons = n >= 4 * k;

    for it in 0..cfg.iterations {
        stats.iterations = it as u32 + 1;

        // Assignment step: independent per row.
        assign
            .par_chunks_mut(CHUNK_ROWS)
            .zip(rows.par_chunks(CHUNK_ROWS))
            .for_each(|(a, r)| {
                for (ai, &ri) in a.iter_mut().zip(r) {
                    *ai = nearest(row(sample, dim, ri), &centroids, dim, k, &live, dot);
                }
            });

        // Update step: per-chunk partial sums in f64, folded in chunk
        // order. Deterministic for any thread count.
        let partials: Vec<(Vec<f64>, Vec<u64>)> = rows
            .par_chunks(CHUNK_ROWS)
            .zip(assign.par_chunks(CHUNK_ROWS))
            .map(|(r, a)| {
                let mut s = vec![0.0f64; k * dim];
                let mut c = vec![0u64; k];
                for (&ri, &ai) in r.iter().zip(a) {
                    let ai = ai as usize;
                    c[ai] += 1;
                    let v = row(sample, dim, ri);
                    let dst = &mut s[ai * dim..(ai + 1) * dim];
                    for (d, x) in dst.iter_mut().zip(v) {
                        *d += *x as f64;
                    }
                }
                (s, c)
            })
            .collect();
        sums.iter_mut().for_each(|s| *s = 0.0);
        counts.iter_mut().for_each(|c| *c = 0);
        for (s, c) in &partials {
            for (dst, x) in sums.iter_mut().zip(s) {
                *dst += x;
            }
            for (dst, x) in counts.iter_mut().zip(c) {
                *dst += x;
            }
        }

        // Repair: an empty (or, with rows to spare, single-member)
        // cluster takes the farthest member of the largest cluster.
        for _ in 0..REPAIR_ROUNDS {
            let needy = (0..k).find(|&c| counts[c] == 0 || (repair_singletons && counts[c] == 1));
            let Some(needy) = needy else { break };
            let largest = (0..k)
                .max_by(|&a, &b| counts[a].cmp(&counts[b]).then(b.cmp(&a)))
                .expect("k > 0");
            if largest == needy || counts[largest] < 2 {
                break;
            }
            let lc = &centroids[largest * dim..(largest + 1) * dim];
            // Farthest member of the largest cluster, first on ties.
            let mut far: Option<(usize, f32)> = None;
            for (i, &ai) in assign.iter().enumerate() {
                if ai as usize != largest {
                    continue;
                }
                let d = dot(row(sample, dim, rows[i]), lc);
                if far.is_none_or(|(_, best)| d < best) {
                    far = Some((i, d));
                }
            }
            let Some((i, _)) = far else { break };
            let v = row(sample, dim, rows[i]);
            // Move row `i`: out of the largest cluster's sums, into the
            // needy cluster as its (new) sole member.
            for (d, x) in sums[largest * dim..(largest + 1) * dim].iter_mut().zip(v) {
                *d -= *x as f64;
            }
            counts[largest] -= 1;
            // Whatever the needy cluster held moves back out too, so
            // its centroid becomes this row alone.
            let prior = counts[needy];
            if prior > 0 {
                for j in 0..assign.len() {
                    if assign[j] as usize == needy {
                        let w = row(sample, dim, rows[j]);
                        for (d, x) in sums[needy * dim..(needy + 1) * dim].iter_mut().zip(w) {
                            *d -= *x as f64;
                        }
                        // The displaced member rejoins the largest.
                        for (d, x) in sums[largest * dim..(largest + 1) * dim].iter_mut().zip(w) {
                            *d += *x as f64;
                        }
                        counts[largest] += 1;
                        assign[j] = largest as u32;
                    }
                }
                counts[needy] = 0;
            }
            for (d, x) in sums[needy * dim..(needy + 1) * dim].iter_mut().zip(v) {
                *d = *x as f64;
            }
            counts[needy] = 1;
            assign[i] = needy as u32;
            live[needy] = true;
            stats.repairs += 1;
        }

        // New centroids: means, re-normalised onto the sphere.
        let mut movement = 0.0f64;
        let mut moved = 0usize;
        for c in 0..k {
            if counts[c] == 0 {
                live[c] = false;
                continue;
            }
            live[c] = true;
            let inv = 1.0 / counts[c] as f64;
            let mut fresh: Vec<f32> = sums[c * dim..(c + 1) * dim]
                .iter()
                .map(|s| (s * inv) as f32)
                .collect();
            normalize_in_place(&mut fresh);
            let old = &centroids[c * dim..(c + 1) * dim];
            movement += (1.0 - dot(old, &fresh) as f64).max(0.0);
            moved += 1;
            centroids[c * dim..(c + 1) * dim].copy_from_slice(&fresh);
        }
        let mean_movement = if moved == 0 {
            0.0
        } else {
            movement / moved as f64
        };
        stats.final_movement = mean_movement as f32;
        if mean_movement < cfg.tolerance as f64 {
            break;
        }
    }

    // Final assignment against the last centroids, so the codes and
    // the centroids agree exactly.
    assign
        .par_chunks_mut(CHUNK_ROWS)
        .zip(rows.par_chunks(CHUNK_ROWS))
        .for_each(|(a, r)| {
            for (ai, &ri) in a.iter_mut().zip(r) {
                *ai = nearest(row(sample, dim, ri), &centroids, dim, k, &live, dot);
            }
        });
    let empty: Vec<bool> = live.iter().map(|l| !l).collect();
    stats.empty = empty.iter().filter(|e| **e).count() as u32;
    (centroids, empty, assign, stats)
}

/// Fit the whole hierarchy over `sample` (`n × dim`, unit rows).
///
/// Level 0 is one run over every row; each further level runs once
/// per parent cluster over that cluster's rows, placing its children
/// at `parent × branching + j`.
pub fn fit_hierarchy(
    sample: &[f32],
    dim: usize,
    cfg: &FitConfig,
    dot: DistFn,
    mut progress: impl FnMut(usize, usize, usize),
) -> TopicModel {
    let n = sample.len() / dim;
    let mut levels: Vec<LevelModel> = Vec::with_capacity(cfg.levels.len());
    // Rows grouped by their cluster at the previous level; one group
    // (all rows) before level 0.
    let mut groups: Vec<Vec<u32>> = vec![(0..n as u32).collect()];
    for (l, &k) in cfg.levels.iter().enumerate() {
        let clusters = groups.len() * k;
        let mut centroids = vec![0.0f32; clusters * dim];
        let mut empty = vec![true; clusters];
        let mut runs = Vec::with_capacity(groups.len());
        let mut next_groups: Vec<Vec<u32>> = (0..clusters).map(|_| Vec::new()).collect();
        for (parent, rows) in groups.iter().enumerate() {
            progress(l, parent, groups.len());
            let (c, e, assign, stats) = kmeans(
                sample,
                dim,
                rows,
                k,
                cfg,
                run_seed(cfg.seed, l, parent),
                dot,
            );
            centroids[parent * k * dim..(parent + 1) * k * dim].copy_from_slice(&c);
            empty[parent * k..(parent + 1) * k].copy_from_slice(&e);
            for (&r, &a) in rows.iter().zip(&assign) {
                next_groups[parent * k + a as usize].push(r);
            }
            runs.push(stats);
        }
        levels.push(LevelModel {
            branching: k,
            clusters,
            centroids,
            empty,
            runs,
        });
        groups = next_groups;
    }
    TopicModel { dim, levels }
}

fn level_reports(model: &TopicModel, tolerance: f32) -> Vec<LevelReport> {
    model
        .levels
        .iter()
        .map(|l| LevelReport {
            branching: l.branching,
            clusters: l.clusters,
            empty: l.empty.iter().filter(|e| **e).count(),
            runs: l.runs.len(),
            converged: l
                .runs
                .iter()
                .filter(|r| r.final_movement < tolerance)
                .count(),
            max_final_movement: l.runs.iter().map(|r| r.final_movement).fold(0.0, f32::max),
            repairs: l.runs.iter().map(|r| r.repairs).sum(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Sample loading and assignment
// ---------------------------------------------------------------------------

/// Row indices of the sample: a prefix, or evenly spaced.
fn sample_indices(count: usize, size: usize, order: SampleOrder) -> Vec<usize> {
    let size = size.min(count);
    match order {
        SampleOrder::Prefix => (0..size).collect(),
        SampleOrder::Strided => {
            if size == 0 {
                return Vec::new();
            }
            let stride = count as f64 / size as f64;
            (0..size)
                .map(|i| ((i as f64 * stride) as usize).min(count - 1))
                .collect()
        }
    }
}

/// Load the sample rows into one contiguous, unit-normalised matrix.
fn load_sample(
    reader: &XvecReader<f32>,
    indices: &[usize],
    normalize: bool,
    mut progress: impl FnMut(usize),
) -> Vec<f32> {
    let dim = reader.dim();
    let mut out = vec![0.0f32; indices.len() * dim];
    for (chunk_no, (dst, src)) in out
        .chunks_mut(CHUNK_ROWS * dim)
        .zip(indices.chunks(CHUNK_ROWS))
        .enumerate()
    {
        if let (Some(&first), Some(&last)) = (src.first(), src.last()) {
            reader.prefetch_range(first, last + 1);
        }
        for (d, &i) in dst.chunks_mut(dim).zip(src) {
            d.copy_from_slice(reader.get_slice(i));
            if normalize {
                normalize_in_place(d);
            }
        }
        progress((chunk_no + 1) * CHUNK_ROWS);
    }
    out
}

/// Assign every vector of `reader` in order, writing codes and
/// margins as they go. Returns the record count.
#[allow(clippy::too_many_arguments)]
fn assign_all(
    reader: &XvecReader<f32>,
    model: &TopicModel,
    normalize: bool,
    dot: DistFn,
    codes_out: &mut FacetWriter,
    margin_out: Option<&mut FacetWriter>,
    mut progress: impl FnMut(u64),
) -> Result<u64, String> {
    let total = reader.count();
    let depth = model.levels.len();
    let mut reclaim = StreamReclaim::new(reader, 0, total);
    let mut codes: Vec<u16> = vec![0; ASSIGN_BATCH * depth];
    let mut margins: Vec<f32> = vec![0.0; ASSIGN_BATCH * 2];
    let mut codes_bytes: Vec<u8> = Vec::with_capacity(ASSIGN_BATCH * (4 + 2 * depth));
    let mut margin_bytes: Vec<u8> = Vec::with_capacity(ASSIGN_BATCH * 8);
    let mut margin_out = margin_out;
    let depth_header = (depth as i32).to_le_bytes();
    let two_header = 2i32.to_le_bytes();

    let mut start = 0usize;
    while start < total {
        let end = (start + ASSIGN_BATCH).min(total);
        let n = end - start;
        // Fetch the batch after this one while this one computes.
        if end < total {
            reader.prefetch_range(end, (end + ASSIGN_BATCH).min(total));
        }
        codes[..n * depth]
            .par_chunks_mut(depth)
            .zip(margins[..n * 2].par_chunks_mut(2))
            .enumerate()
            .for_each(|(j, (c, m))| {
                let v = reader.get_slice(start + j);
                let (best, runner) = if normalize {
                    let mut buf = v.to_vec();
                    normalize_in_place(&mut buf);
                    model.descend(&buf, dot, c)
                } else {
                    model.descend(v, dot, c)
                };
                m[0] = best;
                m[1] = runner;
            });
        codes_bytes.clear();
        for c in codes[..n * depth].chunks(depth) {
            codes_bytes.extend_from_slice(&depth_header);
            for code in c {
                codes_bytes.extend_from_slice(&code.to_le_bytes());
            }
        }
        codes_out
            .write_all(&codes_bytes)
            .map_err(|e| format!("failed to write assignments: {}", e))?;
        if let Some(w) = margin_out.as_deref_mut() {
            margin_bytes.clear();
            for m in margins[..n * 2].chunks(2) {
                margin_bytes.extend_from_slice(&two_header);
                margin_bytes.extend_from_slice(&f16::from_f32(m[0]).to_le_bytes());
                margin_bytes.extend_from_slice(&f16::from_f32(m[1]).to_le_bytes());
            }
            w.write_all(&margin_bytes)
                .map_err(|e| format!("failed to write margins: {}", e))?;
        }
        reclaim.advance(end);
        progress(end as u64);
        start = end;
    }
    Ok(total as u64)
}

/// Write the centroids of every level, in level order, as an fvecs
/// facet.
fn write_centroids(model: &TopicModel, writer: &mut FacetWriter) -> Result<(), String> {
    let header = (model.dim as i32).to_le_bytes();
    let mut buf: Vec<u8> = Vec::with_capacity(4 + model.dim * 4);
    for level in &model.levels {
        for c in 0..level.clusters {
            buf.clear();
            buf.extend_from_slice(&header);
            for x in level.centroid(c, model.dim) {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            writer
                .write_all(&buf)
                .map_err(|e| format!("failed to write centroids: {}", e))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(
    name: &str,
    type_name: &str,
    required: bool,
    default: Option<&str>,
    desc: &str,
    role: OptionRole,
) -> OptionDesc {
    OptionDesc {
        name: name.into(),
        type_name: type_name.into(),
        required,
        default: default.map(str::to_string),
        description: desc.into(),
        extended_description: None,
        role,
    }
}

/// The workspace an output was resolved against: the runner hands
/// `check_artifact` the resolved output path and workspace-relative
/// options, so the workspace is the output with its own option's
/// components stripped. An absolute option, or one that does not end
/// the path, leaves the output's directory as the best guess.
pub(crate) fn workspace_of(output: &Path, output_option: Option<&str>) -> PathBuf {
    if let Some(opt) = output_option {
        let rel = Path::new(opt);
        if rel.is_relative() && output.ends_with(rel) {
            let mut ws = output.to_path_buf();
            for _ in rel.components() {
                ws.pop();
            }
            return ws;
        }
    }
    output
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The model report's path: beside the centroids, with a `.json`
/// extension, unless `model` names it.
fn model_path(options: &Options, centroids: &Path, workspace: &Path) -> PathBuf {
    match options.get("model") {
        Some(s) => resolve_path(s, workspace),
        None => centroids.with_extension("json"),
    }
}

impl CommandOp for ComputeTopicsOp {
    fn command_path(&self) -> &str {
        "compute topics"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Fit a hierarchical spherical k-means topic model and assign every vector"
                .into(),
            body: format!(
                r#"# compute topics

Fits a hierarchical spherical k-means model on a sample of a vector
facet — `levels` clusters at the top, that many children under each,
and so on — then assigns every vector of `base` by greedy descent and
writes one code per level per vector, plus the cosine distances to the
chosen leaf and its best sibling.

Vectors are unit-normalised, so every argmax is an inner product and
centroids are re-normalised after each update. Every reduction runs
over fixed-size chunks folded in order, so the fit is identical for any
thread count. Descent is greedy: a vector near a top-level boundary
can land in a leaf that is not its global nearest; the margin output
is what makes that measurable.

Outputs: `centroids` (fvecs, all levels in order — the model), `output`
(u16vecs, one record per base vector with one code per level), an
optional `margin` (mvecs, dim 2, f16), and `model` (JSON: levels,
seed, sample, convergence — what a third party needs to reproduce the
labelling).

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt(
                "base",
                "Path",
                true,
                None,
                "Vector facet to assign, in the order the assignments must follow; accepts a sharded series",
                OptionRole::Input,
            ),
            opt(
                "sample",
                "Path",
                false,
                None,
                "Vector facet to fit on (default: `base`); a shuffled facet with `sample-order: prefix` is the cheap uniform sample",
                OptionRole::Input,
            ),
            opt(
                "sample-size",
                "int",
                false,
                Some("5000000"),
                "Rows fitted",
                OptionRole::Config,
            ),
            opt(
                "sample-order",
                "string",
                false,
                Some("strided"),
                "`prefix` (first rows; right for a shuffled facet) or `strided` (evenly spaced; right for corpus order)",
                OptionRole::Config,
            ),
            opt(
                "levels",
                "string",
                false,
                Some("10,30,33"),
                "Branching per level, outermost first",
                OptionRole::Config,
            ),
            opt(
                "iterations",
                "int",
                false,
                Some("50"),
                "Iteration cap per k-means run",
                OptionRole::Config,
            ),
            opt(
                "tolerance",
                "float",
                false,
                Some("1e-4"),
                "Mean centroid movement (cosine distance) below which a run has converged",
                OptionRole::Config,
            ),
            opt(
                "seed",
                "int",
                false,
                Some("42"),
                "Seed for k-means++ and sampling",
                OptionRole::Config,
            ),
            opt(
                "normalize",
                "bool",
                false,
                Some("true"),
                "Unit-normalise vectors before fitting and assigning",
                OptionRole::Config,
            ),
            opt(
                "centroids",
                "Path",
                true,
                None,
                "Centroid facet, fvecs, all levels in order",
                OptionRole::Output,
            ),
            opt(
                "output",
                "Path",
                true,
                None,
                "Assignments, u16vecs: one record per base vector, one code per level",
                OptionRole::Output,
            ),
            opt(
                "margin",
                "Path",
                false,
                None,
                "Leaf margin facet, mvecs dim 2: distance to the chosen leaf and to its best sibling; omit to skip",
                OptionRole::Output,
            ),
            opt(
                "model",
                "Path",
                false,
                None,
                "Model report JSON (default: beside `centroids` with a .json extension)",
                OptionRole::Output,
            ),
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description: "The fitting sample, held in memory as f32".into(),
                adjustable: false,
            },
            ResourceDesc {
                name: "threads".into(),
                description: "Parallel rows in fitting and assignment".into(),
                adjustable: true,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let base_str = match options.require("base") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let centroids_str = match options.require("centroids") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let levels = match FitConfig::parse_levels(options.get("levels").unwrap_or("10,30,33")) {
            Ok(l) => l,
            Err(e) => return error_result(e, start),
        };
        let cfg = FitConfig {
            levels,
            iterations: match options.parse_or::<usize>("iterations", 50) {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            },
            tolerance: match options.parse_or::<f32>("tolerance", 1e-4) {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            },
            seed: match options.parse_or::<u64>("seed", 42) {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            },
        };
        let sample_size = match options.parse_or::<usize>("sample-size", 5_000_000) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let sample_order =
            match SampleOrder::parse(options.get("sample-order").unwrap_or("strided")) {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            };
        let normalize = options
            .get("normalize")
            .map(|s| s != "false")
            .unwrap_or(true);

        let base_path = resolve_path(&base_str, &ctx.workspace);
        let sample_str = options
            .get("sample")
            .map(|s| s.to_string())
            .unwrap_or_else(|| base_str.clone());
        let sample_path = resolve_path(&sample_str, &ctx.workspace);
        let centroids_path = resolve_path(&centroids_str, &ctx.workspace);
        let output_path = resolve_path(&output_str, &ctx.workspace);
        let margin_path = options
            .get("margin")
            .map(|s| resolve_path(s, &ctx.workspace));
        let model_path = model_path(options, &centroids_path, &ctx.workspace);

        let base = match XvecReader::<f32>::open_path(&base_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open base {}: {}", base_path.display(), e),
                    start,
                );
            }
        };
        let sample_reader = match XvecReader::<f32>::open_path(&sample_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("failed to open sample {}: {}", sample_path.display(), e),
                    start,
                );
            }
        };
        let dim = base.dim();
        if dim == 0 || base.count() == 0 {
            return error_result(format!("base {} is empty", base_path.display()), start);
        }
        if sample_reader.dim() != dim {
            return error_result(
                format!(
                    "sample dim {} differs from base dim {}",
                    sample_reader.dim(),
                    dim
                ),
                start,
            );
        }
        let indices = sample_indices(sample_reader.count(), sample_size, sample_order);
        if indices.len() < cfg.levels[0] {
            return error_result(
                format!(
                    "sample of {} rows cannot seed {} top-level clusters",
                    indices.len(),
                    cfg.levels[0]
                ),
                start,
            );
        }

        // Resources: the sample matrix is the memory; the governor's
        // threads drive every parallel region.
        let needed = (indices.len() * dim * 4 + cfg.total_centroids() * dim * 4) as u64;
        let granted = ctx.governor.request("mem", needed);
        if granted < needed {
            return error_result(
                format!(
                    "the fitting sample needs {} MiB but the mem budget grants {} MiB; lower sample-size or raise the budget",
                    needed >> 20,
                    granted >> 20
                ),
                start,
            );
        }
        let threads = {
            let t = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
            if t == 0 {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            } else {
                t
            }
        };
        let pool = match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Ok(p) => p,
            Err(e) => return error_result(format!("failed to build thread pool: {}", e), start),
        };
        let (dot, kernel) = select_dot_fn();
        ctx.ui.log(&format!(
            "topics: fitting levels {:?} on {} of {} rows from {} ({:?}), dim {}, {} threads, {} kernel",
            cfg.levels, indices.len(), sample_reader.count(),
            sample_path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            sample_order, dim, threads, kernel,
        ));

        // Load the sample.
        let pb = ctx
            .ui
            .bar_with_unit(indices.len() as u64, "loading sample", "vec");
        let sample = load_sample(&sample_reader, &indices, normalize, |done| {
            pb.set_position(done.min(indices.len()) as u64)
        });
        pb.finish();
        drop(sample_reader);

        // Fit.
        let fit_start = Instant::now();
        let runs_total: usize = {
            let mut n = 1;
            let mut t = 0;
            for k in &cfg.levels {
                t += n;
                n *= k;
            }
            t
        };
        let pb = ctx
            .ui
            .bar_with_unit(runs_total as u64, "fitting topics", "runs");
        let mut done_runs = 0u64;
        let model = pool.install(|| {
            fit_hierarchy(&sample, dim, &cfg, dot, |_level, _parent, _of| {
                done_runs += 1;
                pb.set_position(done_runs);
            })
        });
        pb.finish();
        drop(sample);
        let fit_seconds = fit_start.elapsed().as_secs_f64();
        let per_level = level_reports(&model, cfg.tolerance);
        for (l, r) in per_level.iter().enumerate() {
            ctx.ui.log(&format!(
                "topics: level {} — {} clusters, {} empty, {}/{} runs converged, max movement {:.2e}, {} repairs",
                l + 1, r.clusters, r.empty, r.converged, r.runs, r.max_final_movement, r.repairs,
            ));
        }

        // Centroids.
        let mut produced = Vec::new();
        {
            let cap = cap_for_output(&ctx.governor, &ctx.workspace, &centroids_path);
            let mut w = match FacetWriter::open(&centroids_path, 4 + dim as u64 * 4, cap) {
                Ok(w) => w,
                Err(e) => {
                    return error_result(
                        format!("failed to create {}: {}", centroids_path.display(), e),
                        start,
                    );
                }
            };
            if let Err(e) = write_centroids(&model, &mut w) {
                return error_result(e, start);
            }
            match w.finish() {
                Ok(o) => produced.extend(o.files),
                Err(e) => {
                    return error_result(
                        format!("failed to finish {}: {}", centroids_path.display(), e),
                        start,
                    );
                }
            }
        }

        // Assignment.
        let depth = cfg.levels.len();
        let cap = cap_for_output(&ctx.governor, &ctx.workspace, &output_path);
        let mut codes_w = match FacetWriter::open(&output_path, 4 + 2 * depth as u64, cap) {
            Ok(w) => w,
            Err(e) => {
                return error_result(
                    format!("failed to create {}: {}", output_path.display(), e),
                    start,
                );
            }
        };
        let mut margin_w = match &margin_path {
            Some(p) => {
                let cap = cap_for_output(&ctx.governor, &ctx.workspace, p);
                match FacetWriter::open(p, 8, cap) {
                    Ok(w) => Some(w),
                    Err(e) => {
                        return error_result(
                            format!("failed to create {}: {}", p.display(), e),
                            start,
                        );
                    }
                }
            }
            None => None,
        };
        let assign_start = Instant::now();
        let pb = ctx
            .ui
            .bar_with_unit(base.count() as u64, "assigning topics", "vec");
        let records = match pool.install(|| {
            assign_all(
                &base,
                &model,
                normalize,
                dot,
                &mut codes_w,
                margin_w.as_mut(),
                |done| pb.set_position(done),
            )
        }) {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };
        pb.finish();
        match codes_w.finish() {
            Ok(o) => produced.extend(o.files),
            Err(e) => {
                return error_result(
                    format!("failed to finish {}: {}", output_path.display(), e),
                    start,
                );
            }
        }
        if let Some(w) = margin_w {
            match w.finish() {
                Ok(o) => produced.extend(o.files),
                Err(e) => return error_result(format!("failed to finish margin: {}", e), start),
            }
        }

        // Model report.
        let report = TopicModelReport {
            schema_version: MODEL_SCHEMA_VERSION,
            dim,
            levels: cfg.levels.clone(),
            total_centroids: cfg.total_centroids(),
            sample: sample_str,
            sample_size: indices.len(),
            sample_order,
            seed: cfg.seed,
            iterations: cfg.iterations,
            tolerance: cfg.tolerance,
            normalize,
            kernel: kernel.to_string(),
            fit_seconds,
            per_level,
            assignment: Some(AssignmentReport {
                base: base_str,
                records,
                seconds: assign_start.elapsed().as_secs_f64(),
                margin_written: margin_path.is_some(),
            }),
        };
        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                if let Some(parent) = model_path.parent()
                    && !parent.exists()
                    && let Err(e) = std::fs::create_dir_all(parent)
                {
                    return error_result(
                        format!("failed to create {}: {}", parent.display(), e),
                        start,
                    );
                }
                if let Err(e) = std::fs::write(&model_path, json) {
                    return error_result(
                        format!("failed to write {}: {}", model_path.display(), e),
                        start,
                    );
                }
                produced.push(model_path);
            }
            Err(e) => {
                return error_result(format!("model report serialisation failed: {}", e), start);
            }
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} centroids over {} levels fitted on {} rows in {:.1}s; {} vectors assigned in {:.1}s",
                cfg.total_centroids(),
                cfg.levels.len(),
                indices.len(),
                fit_seconds,
                records,
                assign_start.elapsed().as_secs_f64(),
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    /// Complete when every output agrees with the base and the
    /// configuration: the centroid facet holds Σ levels records of the
    /// base's dimension, the assignment facet holds one record per base
    /// vector with one code per level, the margin (if declared) holds
    /// one dim-2 record per base vector, and the model report parses
    /// with the configured levels. Either alone is insufficient:
    /// centroids without assignments is a fit that never finished, and
    /// assignments without centroids cannot be reproduced or extended.
    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        if !output.exists() && vectordata::dataset::discover_shards(output).is_empty() {
            return ArtifactState::Absent;
        }
        let workspace = workspace_of(output, options.get("output"));
        let Ok(levels) = FitConfig::parse_levels(options.get("levels").unwrap_or("10,30,33"))
        else {
            return ArtifactState::Partial;
        };
        let cfg = FitConfig {
            levels,
            iterations: 0,
            tolerance: 0.0,
            seed: 0,
        };
        let resolve = |key: &str| -> Option<PathBuf> {
            options.get(key).map(|s| {
                let p = PathBuf::from(s);
                if p.is_absolute() {
                    p
                } else {
                    workspace.join(p)
                }
            })
        };
        let Some(base_path) = resolve("base") else {
            return ArtifactState::Partial;
        };
        let Ok(base) = XvecReader::<f32>::open_path(&base_path) else {
            return ArtifactState::Unknown("base facet cannot be opened".into());
        };
        let (count, dim) = (base.count(), base.dim());

        // Assignments: `output`.
        match XvecReader::<u16>::open_path(output) {
            Ok(r) if r.count() == count && r.dim() == cfg.levels.len() => {}
            _ => return ArtifactState::Partial,
        }
        // Centroids.
        let Some(centroids_path) = resolve("centroids") else {
            return ArtifactState::Partial;
        };
        match XvecReader::<f32>::open_path(&centroids_path) {
            Ok(r) if r.count() == cfg.total_centroids() && r.dim() == dim => {}
            _ => return ArtifactState::Partial,
        }
        // Margin, when declared.
        if let Some(margin_path) = resolve("margin") {
            match XvecReader::<f16>::open_path(&margin_path) {
                Ok(r) if r.count() == count && r.dim() == 2 => {}
                _ => return ArtifactState::Partial,
            }
        }
        // Model report.
        let model_path = model_path(options, &centroids_path, &workspace);
        let Ok(text) = std::fs::read_to_string(&model_path) else {
            return ArtifactState::Partial;
        };
        match serde_json::from_str::<TopicModelReport>(&text) {
            Ok(r) if r.levels == cfg.levels && r.dim == dim => ArtifactState::Complete,
            _ => ArtifactState::Partial,
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        let mut manifest = crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["base", "sample"],
            &["centroids", "output", "margin", "model"],
        );
        if options.get("model").is_none()
            && let Some(c) = options.get("centroids")
        {
            let p = PathBuf::from(c).with_extension("json");
            manifest.outputs.push(p.to_string_lossy().to_string());
        }
        manifest
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(levels: &[usize], seed: u64) -> FitConfig {
        FitConfig {
            levels: levels.to_vec(),
            iterations: 50,
            tolerance: 1e-6,
            seed,
        }
    }

    /// `groups` planted unit directions in `dim`, `per` noisy members
    /// each, interleaved so no prefix is one group. Returns the matrix
    /// and each row's planted group.
    fn planted(
        groups: usize,
        per: usize,
        dim: usize,
        noise: f32,
        seed: u64,
    ) -> (Vec<f32>, Vec<usize>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut dirs: Vec<Vec<f32>> = Vec::new();
        for g in 0..groups {
            // Orthogonal-ish planted directions: one-hot plus a little
            // spread, so groups are well separated.
            let mut d = vec![0.0f32; dim];
            d[g % dim] = 1.0;
            d[(g * 7 + 3) % dim] += 0.3;
            normalize_in_place(&mut d);
            dirs.push(d);
        }
        let mut rows = Vec::with_capacity(groups * per * dim);
        let mut labels = Vec::with_capacity(groups * per);
        for i in 0..per {
            for g in 0..groups {
                let mut v: Vec<f32> = dirs[g]
                    .iter()
                    .map(|x| x + noise * (rng.random::<f32>() - 0.5))
                    .collect();
                normalize_in_place(&mut v);
                rows.extend_from_slice(&v);
                labels.push(g);
                let _ = i;
            }
        }
        (rows, labels)
    }

    /// Rows with the same planted group share a cluster, and rows
    /// with different groups do not.
    fn assert_partition_matches(assign: &[u32], labels: &[usize]) {
        let mut seen: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
        for (a, l) in assign.iter().zip(labels) {
            match seen.get(l) {
                Some(prev) => assert_eq!(prev, a, "group {} split across clusters", l),
                None => {
                    assert!(
                        !seen.values().any(|v| v == a),
                        "cluster {} holds two planted groups",
                        a
                    );
                    seen.insert(*l, *a);
                }
            }
        }
    }

    #[test]
    fn parse_levels_forms() {
        assert_eq!(
            FitConfig::parse_levels("10,30,33").unwrap(),
            vec![10, 30, 33]
        );
        assert_eq!(FitConfig::parse_levels(" 4 , 2 ").unwrap(), vec![4, 2]);
        assert!(FitConfig::parse_levels("").is_err());
        assert!(FitConfig::parse_levels("0").is_err());
        assert!(FitConfig::parse_levels("x").is_err());
        assert!(
            FitConfig::parse_levels("300,300").is_err(),
            "90,000 leaves exceed u16"
        );
        let c = cfg(&[10, 30, 33], 0);
        assert_eq!(c.clusters_per_level(), vec![10, 300, 9_900]);
        assert_eq!(c.total_centroids(), 10_210);
    }

    #[test]
    fn kmeans_recovers_planted_clusters() {
        let dim = 16;
        let (rows, labels) = planted(4, 200, dim, 0.2, 1);
        let idx: Vec<u32> = (0..labels.len() as u32).collect();
        let (dot, _) = select_dot_fn();
        let (centroids, empty, assign, stats) = kmeans(&rows, dim, &idx, 4, &cfg(&[4], 7), 99, dot);
        assert!(empty.iter().all(|e| !e), "no empty cluster: {:?}", stats);
        assert_partition_matches(&assign, &labels);
        // Centroids are unit length.
        for c in centroids.chunks(dim) {
            let n = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "norm {}", n);
        }
        assert!(stats.final_movement < 1e-6 || stats.iterations == 50);
    }

    /// Bit-identical centroids and assignments with 1, 3 and 8
    /// threads: the reductions are chunked and folded in order.
    #[test]
    fn fit_is_identical_across_thread_counts() {
        let dim = 12;
        let (rows, _) = planted(6, 300, dim, 0.35, 3);
        let (dot, _) = select_dot_fn();
        let fit = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| fit_hierarchy(&rows, dim, &cfg(&[3, 2], 5), dot, |_, _, _| {}))
        };
        let one = fit(1);
        let three = fit(3);
        let eight = fit(8);
        assert_eq!(one, three);
        assert_eq!(one, eight);
        assert_eq!(one.levels[0].clusters, 3);
        assert_eq!(one.levels[1].clusters, 6);
    }

    /// Two real directions, three clusters asked for: the third is not
    /// left empty but split from the largest.
    #[test]
    fn empty_cluster_is_repaired_from_the_largest() {
        let dim = 8;
        let (rows, _) = planted(2, 150, dim, 0.4, 11);
        let idx: Vec<u32> = (0..300).collect();
        let (dot, _) = select_dot_fn();
        let (_, empty, assign, stats) = kmeans(&rows, dim, &idx, 3, &cfg(&[3], 2), 5, dot);
        assert!(
            empty.iter().all(|e| !e),
            "repair left an empty cluster: {:?}",
            stats
        );
        let mut counts = [0usize; 3];
        for a in &assign {
            counts[*a as usize] += 1;
        }
        assert!(counts.iter().all(|c| *c >= 2), "{:?}", counts);
    }

    /// Fewer rows than clusters: each row is its own centroid and the
    /// rest are flagged empty rather than invented.
    #[test]
    fn degenerate_run_flags_empties() {
        let dim = 4;
        let rows = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (dot, _) = select_dot_fn();
        let (c, empty, assign, stats) = kmeans(&rows, dim, &[0, 1], 5, &cfg(&[5], 1), 1, dot);
        assert_eq!(empty, vec![false, false, true, true, true]);
        assert_eq!(assign, vec![0, 1]);
        assert_eq!(stats.empty, 3);
        assert_eq!(&c[..4], &[1.0, 0.0, 0.0, 0.0]);
    }

    /// A 2 × 3 planted hierarchy: the fit's own leaf assignment and
    /// descent agree, every leaf is a planted group, and the margin is
    /// smaller for the chosen leaf than for its sibling.
    #[test]
    fn hierarchy_fit_and_descent_agree() {
        let dim = 24;
        // Six groups arranged as two families of three: family
        // directions plus a smaller within-family offset.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut rows = Vec::new();
        let mut leaf_of = Vec::new();
        for i in 0..240 {
            let fam = i % 2;
            let sub = (i / 2) % 3;
            let mut v = vec![0.0f32; dim];
            v[fam] = 1.0;
            v[4 + fam * 3 + sub] = 0.6;
            for x in v.iter_mut() {
                *x += 0.05 * (rng.random::<f32>() - 0.5);
            }
            normalize_in_place(&mut v);
            rows.extend_from_slice(&v);
            leaf_of.push(fam * 3 + sub);
        }
        let (dot, _) = select_dot_fn();
        let model = fit_hierarchy(&rows, dim, &cfg(&[2, 3], 4), dot, |_, _, _| {});
        assert_eq!(model.levels[1].clusters, 6);
        assert!(model.levels.iter().all(|l| l.empty.iter().all(|e| !e)));
        let mut codes = [0u16; 2];
        let mut leaf_by_group: std::collections::HashMap<usize, u16> = Default::default();
        for (i, planted_leaf) in leaf_of.iter().enumerate() {
            let v = &rows[i * dim..(i + 1) * dim];
            let (best, runner) = model.descend(v, dot, &mut codes);
            assert!(
                best <= runner,
                "chosen leaf must be at least as close as its sibling"
            );
            assert_eq!(
                codes[0] as usize,
                codes[1] as usize / 3,
                "leaf code nests under its parent"
            );
            match leaf_by_group.get(planted_leaf) {
                Some(prev) => assert_eq!(*prev, codes[1], "planted group {} split", planted_leaf),
                None => {
                    leaf_by_group.insert(*planted_leaf, codes[1]);
                }
            }
        }
        assert_eq!(leaf_by_group.len(), 6);
    }

    #[test]
    fn sample_indices_prefix_and_strided() {
        assert_eq!(sample_indices(10, 3, SampleOrder::Prefix), vec![0, 1, 2]);
        assert_eq!(
            sample_indices(10, 4, SampleOrder::Strided),
            vec![0, 2, 5, 7]
        );
        assert_eq!(sample_indices(3, 10, SampleOrder::Strided), vec![0, 1, 2]);
        assert!(sample_indices(0, 5, SampleOrder::Strided).is_empty());
        assert_eq!(SampleOrder::parse("Prefix").unwrap(), SampleOrder::Prefix);
        assert!(SampleOrder::parse("random").is_err());
    }

    #[test]
    fn workspace_is_recovered_from_the_resolved_output() {
        let out = Path::new("/data/ds/.cache/topic_assign.u16vecs");
        assert_eq!(
            workspace_of(out, Some(".cache/topic_assign.u16vecs")),
            PathBuf::from("/data/ds")
        );
        assert_eq!(
            workspace_of(out, Some("/elsewhere/x.u16vecs")),
            PathBuf::from("/data/ds/.cache")
        );
        assert_eq!(workspace_of(out, None), PathBuf::from("/data/ds/.cache"));
    }

    #[test]
    fn run_seeds_differ_per_level_and_parent() {
        let a = run_seed(42, 0, 0);
        let b = run_seed(42, 1, 0);
        let c = run_seed(42, 1, 1);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(a, run_seed(42, 0, 0));
    }
}
