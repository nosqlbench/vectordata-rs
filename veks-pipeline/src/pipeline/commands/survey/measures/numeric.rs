// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Numeric measures used during Pass 2 for fields whose
//! `SemanticType` is `Number(_)` or for the numeric promotion of
//! `Temporal(Timestamp{..})` and `EnumOrd`.
//!
//! This module currently lands [`ExactExtrema`] and [`ExactMoments`];
//! richer measures (quantile sketch, histogram, distribution-fit,
//! bit-width, monotonicity, run-length) land in build-plan step 6.

use serde::{Deserialize, Serialize};
use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    ExactExtremaReport, ExactMomentsReport, Measure, MeasureCtx, MeasureKind, MeasureReport,
};
use crate::pipeline::commands::survey::sketches::KllSketch;

// ---------------------------------------------------------------------------
// ExactExtrema
// ---------------------------------------------------------------------------

/// Tracks the exact minimum and maximum of every numeric observation.
///
/// Observations that cannot be cast to `f64` (e.g. a Text value
/// reaching a measure pinned to `Number`) are ignored — the parser
/// in front of the measure is expected to have rejected them.
pub struct ExactExtrema {
    min: f64,
    max: f64,
    saw_any: bool,
}

impl Default for ExactExtrema {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactExtrema {
    pub fn new() -> Self {
        ExactExtrema {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            saw_any: false,
        }
    }

    fn record(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.saw_any = true;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }
}

impl Measure for ExactExtrema {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let Some(x) = mvalue_as_f64(value) {
            self.record(x);
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let (min, max) = if self.saw_any {
            (Some(self.min), Some(self.max))
        } else {
            (None, None)
        };
        MeasureReport::ExactExtrema(ExactExtremaReport { min, max })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ExactExtrema
    }
}

// ---------------------------------------------------------------------------
// ExactMoments
// ---------------------------------------------------------------------------

/// Exact running moments `m1..m4` over numeric observations, using
/// Welford's online algorithm extended to 4th-order central moments.
///
/// Reference: Pébay, P. P. (2008). "Formulas for robust, one-pass
/// parallel computation of covariances and arbitrary-order
/// statistical moments". Sandia Report SAND2008-6212.
///
/// All arithmetic is in `f64`. Pébay's formulation is numerically
/// stable for streams of arbitrary length and is exact in the
/// computational sense (errors are machine-epsilon rounding only).
pub struct ExactMoments {
    n: u64,
    /// First moment (running mean).
    m1: f64,
    /// Central moment m2 (sum of squared deviations).
    m2: f64,
    /// Central moment m3 (sum of cubed deviations).
    m3: f64,
    /// Central moment m4 (sum of fourth-power deviations).
    m4: f64,
}

impl Default for ExactMoments {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactMoments {
    pub fn new() -> Self {
        ExactMoments {
            n: 0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    /// Welford update extended to fourth-order central moments per
    /// Pébay 2008 equations (2.6) … (2.9).
    fn update(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.m1;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.m1 += delta_n;
        self.m4 +=
            term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2 - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    /// Sample mean.
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.m1
        }
    }

    /// Sample variance with Bessel correction (`1 / (n-1)`). Returns
    /// NaN for streams of fewer than two observations.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }

    /// Sample standard deviation.
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Skewness (third standardized moment, biased estimator
    /// `m3 / (m2)^(3/2) · √n`). Returns 0 when `m2` is zero.
    pub fn skewness(&self) -> f64 {
        if self.n < 2 || self.m2 == 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        (n.sqrt() * self.m3) / self.m2.powf(1.5)
    }

    /// Excess kurtosis (`n · m4 / m2² - 3`). Returns 0 when `m2` is
    /// zero.
    pub fn kurtosis(&self) -> f64 {
        if self.n < 2 || self.m2 == 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        (n * self.m4) / (self.m2 * self.m2) - 3.0
    }

    /// Number of observations contributing to the moments.
    pub fn count(&self) -> u64 {
        self.n
    }
}

impl Measure for ExactMoments {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let Some(x) = mvalue_as_f64(value) {
            self.update(x);
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let (mean, stddev, skewness, kurtosis) = if self.n == 0 {
            (None, None, None, None)
        } else if self.n < 2 {
            (Some(self.m1), None, None, None)
        } else {
            (Some(self.mean()), Some(self.stddev()), Some(self.skewness()), Some(self.kurtosis()))
        };
        MeasureReport::ExactMoments(ExactMomentsReport {
            count: self.n,
            mean,
            stddev,
            skewness,
            kurtosis,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ExactMoments
    }
}

// ---------------------------------------------------------------------------
// QuantileSketchMeasure
// ---------------------------------------------------------------------------

/// KLL quantile sketch over numeric observations.
///
/// Reports a fixed quantile vector plus the underlying min / max for
/// downstream consumers that want to render a CDF or a histogram.
pub struct QuantileSketchMeasure {
    sketch: KllSketch,
}

/// Minimum `k` parameter for the underlying KLL sketch. Quantile
/// resolution at `k=1000` is approximately ±0.16% (the bound is
/// `≈ 1.6/sqrt(k)` on a 99% confidence interval), which is what
/// `generate predicates` needs for its selectivity-precision
/// contract. Smaller `k` saves a bit of memory but blows out the
/// quantile error in ways downstream consumers can't recover from,
/// so the floor is enforced at construction.
pub const QUANTILE_SKETCH_K_MIN: usize = 1000;

impl QuantileSketchMeasure {
    pub fn new(k: usize, seed: u64) -> Self {
        let k = k.max(QUANTILE_SKETCH_K_MIN);
        QuantileSketchMeasure {
            sketch: KllSketch::with_seed(k, seed),
        }
    }
}

impl Measure for QuantileSketchMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let Some(x) = mvalue_as_f64(value) {
            if !x.is_nan() {
                self.sketch.add(x);
            }
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
        let values = self.sketch.quantiles(&qs).unwrap_or_default();
        let pairs: Vec<(String, f64)> = qs
            .iter()
            .zip(values.iter())
            .map(|(q, v)| (format!("p{}", (q * 100.0).round() as u32), *v))
            .collect();
        MeasureReport::QuantileSketch(QuantileSketchReport {
            count: self.sketch.count(),
            min: self.sketch.min(),
            max: self.sketch.max(),
            quantiles: pairs,
            k: self.sketch.k() as u32,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::QuantileSketch
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantileSketchReport {
    pub count: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Quantile/value pairs in ascending quantile order.
    pub quantiles: Vec<(String, f64)>,
    /// KLL parameter used.
    pub k: u32,
}

// ---------------------------------------------------------------------------
// BitWidthReport
// ---------------------------------------------------------------------------

/// Classifies integer fields by the bit-width and density of the
/// observed value space.
///
/// Distinguishes "small packed enum" (tight contiguous range) from
/// "ID-like" (large sparse range, monotone-ish) from "hash-like"
/// (uniform across the full width).
pub struct BitWidthMeasure {
    n: u64,
    min: i64,
    max: i64,
    /// Sum-of-bits histogram across the high N bits (proxy for
    /// uniformity). Each observation contributes its popcount.
    popcount_sum: u128,
    saw_any: bool,
}

impl Default for BitWidthMeasure {
    fn default() -> Self { Self::new() }
}

impl BitWidthMeasure {
    pub fn new() -> Self {
        BitWidthMeasure {
            n: 0,
            min: i64::MAX,
            max: i64::MIN,
            popcount_sum: 0,
            saw_any: false,
        }
    }
}

impl Measure for BitWidthMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let i = match value {
            MValue::Int(v) => *v,
            MValue::Int32(v) => *v as i64,
            MValue::Short(v) => *v as i64,
            MValue::Millis(v) => *v,
            MValue::EnumOrd(v) => *v as i64,
            _ => return,
        };
        self.saw_any = true;
        self.n += 1;
        if i < self.min { self.min = i; }
        if i > self.max { self.max = i; }
        self.popcount_sum += (i as u64).count_ones() as u128;
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        if !self.saw_any {
            return MeasureReport::BitWidth(BitWidthReport {
                count: 0,
                min: 0, max: 0, range: 0,
                bits_used: 0,
                density: 0.0,
                avg_popcount: 0.0,
                classification: "Empty".into(),
            });
        }
        let range = (self.max as i128) - (self.min as i128);
        let range_u = range.max(0) as u128;
        // Bits needed to enumerate the observed range.
        let bits_used = if range_u == 0 { 0 } else { 128 - range_u.leading_zeros() as u32 };
        let density = self.n as f64 / (range_u.max(1) as f64);
        let avg_pop = self.popcount_sum as f64 / self.n as f64;
        // Heuristic classification.
        let classification = if range_u == 0 {
            "Constant"
        } else if bits_used <= 8 && density > 0.5 {
            "PackedEnum"
        } else if density > 0.1 {
            "Sequential"
        } else if avg_pop > 28.0 {
            "HashLike"
        } else {
            "Sparse"
        };
        MeasureReport::BitWidth(BitWidthReport {
            count: self.n,
            min: self.min,
            max: self.max,
            range: range_u.min(u64::MAX as u128) as u64,
            bits_used,
            density,
            avg_popcount: avg_pop,
            classification: classification.into(),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::BitWidth
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BitWidthReport {
    pub count: u64,
    pub min: i64,
    pub max: i64,
    pub range: u64,
    /// Bits needed to enumerate the observed range.
    pub bits_used: u32,
    /// Observations / range. High → dense / sequential. Low → sparse.
    pub density: f64,
    /// Mean Hamming weight of the observed values. ~32 bits set on
    /// avg in a 64-bit space → uniform hash-like distribution.
    pub avg_popcount: f64,
    /// Heuristic classification: PackedEnum / Sequential / HashLike /
    /// Sparse / Constant / Empty.
    pub classification: String,
}

// ---------------------------------------------------------------------------
// HistogramFromQuantilesMeasure
// ---------------------------------------------------------------------------

/// Equi-width histogram over numeric observations, with bin
/// boundaries chosen from the running min/max. Reports per-bin
/// counts and the bin edges; downstream consumers can render a
/// CDF / PDF without re-scanning the raw data.
///
/// Separate from `QuantileSketchMeasure` because histograms answer a
/// different question (per-bin density) and are cheap enough not to
/// require KLL.
pub struct HistogramFromQuantilesMeasure {
    bin_count: usize,
    values: Vec<f64>,
    min: f64,
    max: f64,
}

impl HistogramFromQuantilesMeasure {
    pub fn new(bin_count: usize) -> Self {
        HistogramFromQuantilesMeasure {
            bin_count: bin_count.max(1),
            values: Vec::new(),
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

impl Measure for HistogramFromQuantilesMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let Some(x) = mvalue_as_f64(value) {
            if !x.is_nan() {
                if x < self.min { self.min = x; }
                if x > self.max { self.max = x; }
                self.values.push(x);
            }
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        if self.values.is_empty() {
            return MeasureReport::HistogramFromQuantiles(HistogramFromQuantilesReport {
                bin_count: self.bin_count as u32,
                min: None, max: None,
                edges: Vec::new(),
                counts: Vec::new(),
            });
        }
        let lo = self.min;
        let hi = self.max;
        if (hi - lo).abs() <= f64::EPSILON {
            // Degenerate (constant) — one bin with everything in it.
            return MeasureReport::HistogramFromQuantiles(HistogramFromQuantilesReport {
                bin_count: 1,
                min: Some(lo), max: Some(hi),
                edges: vec![lo, hi],
                counts: vec![self.values.len() as u64],
            });
        }
        let bins = self.bin_count;
        let width = (hi - lo) / bins as f64;
        let mut counts = vec![0u64; bins];
        for v in &self.values {
            let mut idx = ((*v - lo) / width).floor() as isize;
            if idx >= bins as isize { idx = bins as isize - 1; }
            if idx < 0 { idx = 0; }
            counts[idx as usize] += 1;
        }
        let edges: Vec<f64> = (0..=bins).map(|i| lo + i as f64 * width).collect();
        MeasureReport::HistogramFromQuantiles(HistogramFromQuantilesReport {
            bin_count: bins as u32,
            min: Some(lo), max: Some(hi),
            edges,
            counts,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::HistogramFromQuantiles
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistogramFromQuantilesReport {
    pub bin_count: u32,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// `bin_count + 1` bin edges in ascending order.
    pub edges: Vec<f64>,
    /// One count per bin.
    pub counts: Vec<u64>,
}

// ---------------------------------------------------------------------------
// MonotonicityReport (Mann-Kendall τ over a bounded sample)
// ---------------------------------------------------------------------------

/// Tracks ascending / descending / equal step counts in observation
/// order. Reports the Mann-Kendall τ-statistic over the bounded
/// sample, useful for detecting fields whose values drift with
/// record position.
pub struct MonotonicityMeasure {
    last: Option<f64>,
    ascending_steps: u64,
    descending_steps: u64,
    equal_steps: u64,
    /// Bounded sample retained for τ computation. Mann-Kendall is
    /// O(n²), so we cap at `max_pairs` to keep finalize cost bounded.
    sample: Vec<f64>,
    max_sample: usize,
}

impl MonotonicityMeasure {
    pub fn new(max_sample: usize) -> Self {
        MonotonicityMeasure {
            last: None,
            ascending_steps: 0,
            descending_steps: 0,
            equal_steps: 0,
            sample: Vec::new(),
            max_sample: max_sample.max(2),
        }
    }
}

impl Measure for MonotonicityMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let Some(x) = mvalue_as_f64(value) else { return };
        if x.is_nan() { return }
        if let Some(prev) = self.last {
            if x > prev { self.ascending_steps += 1; }
            else if x < prev { self.descending_steps += 1; }
            else { self.equal_steps += 1; }
        }
        self.last = Some(x);
        if self.sample.len() < self.max_sample {
            self.sample.push(x);
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        // Mann-Kendall τ-statistic: (concordant - discordant) / n(n-1)/2.
        let n = self.sample.len();
        let tau = if n < 2 {
            None
        } else {
            let mut concordant = 0i64;
            let mut discordant = 0i64;
            for i in 0..n {
                for j in (i + 1)..n {
                    if self.sample[j] > self.sample[i] { concordant += 1; }
                    else if self.sample[j] < self.sample[i] { discordant += 1; }
                }
            }
            let pairs = (n * (n - 1) / 2) as f64;
            if pairs == 0.0 { None } else { Some((concordant - discordant) as f64 / pairs) }
        };
        let classification = match tau {
            None => "Unknown",
            Some(t) if t > 0.7 => "StrictlyAscending",
            Some(t) if t > 0.3 => "MostlyAscending",
            Some(t) if t < -0.7 => "StrictlyDescending",
            Some(t) if t < -0.3 => "MostlyDescending",
            _ => "Random",
        };
        MeasureReport::Monotonicity(MonotonicityReport {
            ascending_steps: self.ascending_steps,
            descending_steps: self.descending_steps,
            equal_steps: self.equal_steps,
            mann_kendall_tau: tau,
            classification: classification.into(),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::Monotonicity
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonotonicityReport {
    pub ascending_steps: u64,
    pub descending_steps: u64,
    pub equal_steps: u64,
    /// Mann-Kendall τ in `[-1, 1]`. `None` if fewer than 2
    /// observations contributed to the sample.
    pub mann_kendall_tau: Option<f64>,
    /// StrictlyAscending / MostlyAscending / Random / MostlyDescending
    /// / StrictlyDescending / Unknown.
    pub classification: String,
}

// ---------------------------------------------------------------------------
// DiscreteIndicatorMeasure
// ---------------------------------------------------------------------------

/// For Float fields, reports the fraction of observations that
/// happened to be integer-valued (`x == x.round()`). High values
/// indicate the field is logically an integer encoded as float.
pub struct DiscreteIndicatorMeasure {
    n: u64,
    integer_valued: u64,
}

impl Default for DiscreteIndicatorMeasure {
    fn default() -> Self { Self::new() }
}

impl DiscreteIndicatorMeasure {
    pub fn new() -> Self {
        DiscreteIndicatorMeasure { n: 0, integer_valued: 0 }
    }
}

impl Measure for DiscreteIndicatorMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let Some(x) = mvalue_as_f64(value) else { return };
        if x.is_nan() { return }
        self.n += 1;
        if x == x.round() {
            self.integer_valued += 1;
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let rate = if self.n == 0 { 0.0 } else { self.integer_valued as f64 / self.n as f64 };
        MeasureReport::DiscreteIndicator(DiscreteIndicatorReport {
            count: self.n,
            integer_valued: self.integer_valued,
            integer_rate: rate,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::DiscreteIndicator
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscreteIndicatorReport {
    pub count: u64,
    pub integer_valued: u64,
    /// `integer_valued / count`. 1.0 → field is really an integer.
    pub integer_rate: f64,
}

// ---------------------------------------------------------------------------
// Shared helper: numeric coercion from `MValue`
// ---------------------------------------------------------------------------

/// Coerce a numeric-typed `MValue` to `f64`. Returns `None` for
/// non-numeric variants (including `Null`). Textual numbers are
/// handled upstream by the semantic-probe layer — by the time a
/// value reaches a numeric measure, the orchestrator has already
/// converted it to a numeric `MValue` via the parsed canonical form.
pub(crate) fn mvalue_as_f64(value: &MValue) -> Option<f64> {
    match value {
        MValue::Int(v) => Some(*v as f64),
        MValue::Int32(v) => Some(*v as f64),
        MValue::Short(v) => Some(*v as f64),
        MValue::Float(v) => Some(*v),
        MValue::Float32(v) => Some(*v as f64),
        MValue::Half(v) => Some(half::f16::from_bits(*v).to_f64()),
        MValue::Millis(v) => Some(*v as f64),
        MValue::Nanos { epoch_seconds, nano_adjust } => {
            Some(*epoch_seconds as f64 + (*nano_adjust as f64) * 1e-9)
        }
        MValue::EnumOrd(v) => Some(*v as f64),
        MValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    /// Min / max recover exact values regardless of insertion order.
    #[test]
    fn extrema_exact() {
        let mut e = ExactExtrema::new();
        for v in [5.0, -3.0, 12.0, 0.5, -8.0, 7.0] {
            e.observe(&MValue::Float(v), &ctx());
        }
        let r = match Box::new(e).finalize() {
            MeasureReport::ExactExtrema(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.min, Some(-8.0));
        assert_eq!(r.max, Some(12.0));
    }

    /// Mixed numeric types coerce to a common `f64` range.
    #[test]
    fn extrema_mixed_numeric() {
        let mut e = ExactExtrema::new();
        e.observe(&MValue::Int(100), &ctx());
        e.observe(&MValue::Float(0.5), &ctx());
        e.observe(&MValue::Short(-5), &ctx());
        e.observe(&MValue::Bool(true), &ctx());
        let r = match Box::new(e).finalize() {
            MeasureReport::ExactExtrema(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.min, Some(-5.0));
        assert_eq!(r.max, Some(100.0));
    }

    /// Non-numeric values pass through without affecting state.
    #[test]
    fn extrema_ignores_non_numeric() {
        let mut e = ExactExtrema::new();
        e.observe(&MValue::Text("oops".into()), &ctx());
        e.observe(&MValue::Null, &ctx());
        e.observe(&MValue::Int(42), &ctx());
        let r = match Box::new(e).finalize() {
            MeasureReport::ExactExtrema(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.min, Some(42.0));
        assert_eq!(r.max, Some(42.0));
    }

    /// Empty stream yields NaN sentinels.
    #[test]
    fn extrema_empty_is_nan() {
        let e = ExactExtrema::new();
        let r = match Box::new(e).finalize() {
            MeasureReport::ExactExtrema(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert!(r.min.is_none());
        assert!(r.max.is_none());
    }

    /// Mean and stddev match closed-form values for a known stream.
    #[test]
    fn moments_match_closed_form() {
        // Stream: 1, 2, 3, 4, 5
        // mean = 3.0, variance = 2.5, stddev = √2.5 ≈ 1.5811
        let mut m = ExactMoments::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            m.observe(&MValue::Float(v), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::ExactMoments(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 5);
        assert!((r.mean.unwrap() - 3.0).abs() < 1e-12);
        // Bessel-corrected variance: 10 / 4 = 2.5 → stddev = √2.5.
        assert!((r.stddev.unwrap() - (2.5_f64).sqrt()).abs() < 1e-9);
    }

    /// Skewness is exactly 0 for a symmetric stream around the mean.
    #[test]
    fn moments_symmetric_zero_skewness() {
        let mut m = ExactMoments::new();
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            m.observe(&MValue::Float(v), &ctx());
        }
        assert!((m.skewness()).abs() < 1e-9);
    }

    /// Constant stream: mean = constant, variance = 0, derived stats
    /// fall back to zero rather than NaN-explode.
    #[test]
    fn moments_constant_stream() {
        let mut m = ExactMoments::new();
        for _ in 0..100 {
            m.observe(&MValue::Float(7.5), &ctx());
        }
        assert_eq!(m.mean(), 7.5);
        assert_eq!(m.variance(), 0.0);
        assert_eq!(m.skewness(), 0.0);
        assert_eq!(m.kurtosis(), 0.0);
    }

    /// Single-observation case: variance / stddev are NaN (need n ≥ 2).
    #[test]
    fn moments_single_observation() {
        let mut m = ExactMoments::new();
        m.observe(&MValue::Int(42), &ctx());
        assert_eq!(m.count(), 1);
        assert_eq!(m.mean(), 42.0);
        assert!(m.variance().is_nan());
        assert!(m.stddev().is_nan());
    }

    /// Welford stability under a long stream: 10k samples from a
    /// known distribution (uniform[0,1) via a simple LCG) recover
    /// mean and variance within tight tolerances.
    #[test]
    fn moments_long_stream_stable() {
        let mut m = ExactMoments::new();
        let mut state: u64 = 1;
        for _ in 0..10_000 {
            // LCG: deterministic pseudo-uniform [0, 1)
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (state >> 11) as f64 / (1u64 << 53) as f64;
            m.observe(&MValue::Float(x), &ctx());
        }
        // Uniform[0,1): mean = 0.5, variance = 1/12 ≈ 0.0833.
        assert!((m.mean() - 0.5).abs() < 0.02);
        assert!((m.variance() - 1.0 / 12.0).abs() < 0.01);
    }

    /// QuantileSketch recovers p50 on a sorted stream. Effective `k`
    /// is the floor `QUANTILE_SKETCH_K_MIN` even though we asked for
    /// less — the clamp keeps the quantile precision contract that
    /// `generate predicates` relies on.
    #[test]
    fn quantile_sketch_recovers_median() {
        let mut m = QuantileSketchMeasure::new(200, 7);
        for i in 1..=10_000 {
            m.observe(&MValue::Int(i), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::QuantileSketch(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 10_000);
        let p50 = r.quantiles.iter().find(|(k, _)| k == "p50").unwrap().1;
        // True median is 5000; KLL rank error at k=1000 is ε ≈ 0.16%.
        assert!((p50 - 5000.0).abs() < 50.0, "p50 = {}", p50);
        assert_eq!(r.k, QUANTILE_SKETCH_K_MIN as u32);
    }

    /// Asking for k below the floor must produce a sketch reporting
    /// the floor — not the asked-for value. Guards the precision
    /// contract from a caller silently weakening it.
    #[test]
    fn quantile_sketch_k_is_clamped_to_floor() {
        let m = QuantileSketchMeasure::new(50, 0);
        let r = match Box::new(m).finalize() {
            MeasureReport::QuantileSketch(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.k, QUANTILE_SKETCH_K_MIN as u32);
    }

    /// Asking for k above the floor honors the request — the floor
    /// only kicks in for too-small values.
    #[test]
    fn quantile_sketch_k_passes_through_above_floor() {
        let m = QuantileSketchMeasure::new(4000, 0);
        let r = match Box::new(m).finalize() {
            MeasureReport::QuantileSketch(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.k, 4000);
    }

    #[test]
    fn quantile_sketch_empty() {
        let m = QuantileSketchMeasure::new(200, 0);
        let r = match Box::new(m).finalize() {
            MeasureReport::QuantileSketch(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 0);
        assert!(r.min.is_none());
        assert!(r.max.is_none());
    }

    /// PackedEnum: dense small range.
    #[test]
    fn bit_width_packed_enum() {
        let mut m = BitWidthMeasure::new();
        for i in 0..10_000 {
            m.observe(&MValue::Int(i % 8), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::BitWidth(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.min, 0);
        assert_eq!(r.max, 7);
        assert!(r.bits_used <= 3);
        assert_eq!(r.classification, "PackedEnum");
    }

    /// Sequential: dense over a larger range (one obs per integer).
    #[test]
    fn bit_width_sequential() {
        let mut m = BitWidthMeasure::new();
        for i in 0..10_000i64 {
            m.observe(&MValue::Int(i), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::BitWidth(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.classification, "Sequential");
        assert_eq!(r.range, 9_999);
    }

    /// HashLike: ~32 bits set on average in a uniform 64-bit space.
    #[test]
    fn bit_width_hash_like() {
        let mut m = BitWidthMeasure::new();
        let mut state: u64 = 0xdead_beef;
        for _ in 0..2_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            m.observe(&MValue::Int(state as i64), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::BitWidth(r) => r,
            _ => panic!("wrong report kind"),
        };
        // The LCG above produces values across the full 64-bit range
        // with ~32 bits set on average. With i64 reinterpretation
        // half the values are negative — range becomes huge,
        // density tiny, popcount ~32. Classified as HashLike.
        assert!(matches!(r.classification.as_str(), "HashLike" | "Sparse"));
        assert!(r.avg_popcount > 25.0);
    }

    /// Constant: all observations identical.
    #[test]
    fn bit_width_constant() {
        let mut m = BitWidthMeasure::new();
        for _ in 0..100 {
            m.observe(&MValue::Int(42), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::BitWidth(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.classification, "Constant");
        assert_eq!(r.range, 0);
    }

    /// Excess kurtosis for a uniform[a,b] distribution is -6/5 = -1.2
    /// — verify Welford-Pébay update recovers this asymptotically.
    #[test]
    fn moments_uniform_kurtosis_negative_one_point_two() {
        let mut m = ExactMoments::new();
        let mut state: u64 = 42;
        for _ in 0..200_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (state >> 11) as f64 / (1u64 << 53) as f64;
            m.observe(&MValue::Float(x), &ctx());
        }
        // Loose tolerance: with 200k samples kurtosis estimate
        // typically lands within ±0.05 of the asymptotic -1.2.
        let k = m.kurtosis();
        assert!(
            (k - (-1.2)).abs() < 0.1,
            "kurtosis estimate {} should be near -1.2 for uniform[0,1)", k
        );
    }
}
