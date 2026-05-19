// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cardinality and frequency measures.
//!
//! - [`HyperLogLogMeasure`] — sketch-based distinct-count estimate.
//! - [`HeavyHittersMeasure`] — top-K frequent values with Misra–Gries
//!   bounds.
//! - [`ExactFrequencyTable`] — full distinct-value frequency table,
//!   exact mode/entropy/Gini. Gated by the `low_card_threshold`;
//!   only instantiated when Pass 1 confirmed `cardinality ≤
//!   low_card_threshold`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport,
};
use crate::pipeline::commands::survey::sketches::{HyperLogLog, MisraGries};

// ---------------------------------------------------------------------------
// HyperLogLogMeasure
// ---------------------------------------------------------------------------

/// Adapter that wraps a [`HyperLogLog`] sketch behind the `Measure`
/// trait.
pub struct HyperLogLogMeasure {
    inner: HyperLogLog,
}

impl HyperLogLogMeasure {
    /// `precision` is the HLL `p` parameter (register count = 2^p).
    /// Default in the survey config is 12 (~4 KB, σ ≈ 1.6%).
    pub fn new(precision: u8) -> Self {
        HyperLogLogMeasure {
            inner: HyperLogLog::new(precision),
        }
    }
}

impl Measure for HyperLogLogMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let key = canonical_distinct_key(value);
        self.inner.add(&key);
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let estimate = self.inner.estimate();
        let stderr = self.inner.relative_stderr();
        MeasureReport::HyperLogLog(HyperLogLogReport {
            cardinality_estimate: estimate,
            stderr,
            observations: self.inner.observed(),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::HyperLogLog
    }
}

/// Report shape for [`HyperLogLogMeasure::finalize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperLogLogReport {
    /// Estimated distinct count.
    pub cardinality_estimate: f64,
    /// Theoretical relative standard error of the estimate.
    pub stderr: f64,
    /// Total observations that fed the sketch.
    pub observations: u64,
}

// ---------------------------------------------------------------------------
// HeavyHittersMeasure
// ---------------------------------------------------------------------------

/// Top-K frequent items via [`MisraGries`].
pub struct HeavyHittersMeasure {
    inner: MisraGries<String>,
    top_k: usize,
}

impl HeavyHittersMeasure {
    pub fn new(top_k: usize) -> Self {
        HeavyHittersMeasure {
            inner: MisraGries::new(top_k),
            top_k,
        }
    }
}

impl Measure for HeavyHittersMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let key = canonical_distinct_key(value);
        self.inner.add(key);
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let entries = self.inner.top_k();
        let observed = self.inner.seen();
        let error_bound = self.inner.error_bound();
        MeasureReport::HeavyHitters(HeavyHittersReport {
            top_k: self.top_k,
            items: entries.into_iter().map(|(k, v)| HeavyHitterEntry {
                value: k,
                count_lower_bound: v,
            }).collect(),
            observations: observed,
            error_bound,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::HeavyHitters
    }
}

/// Report shape for [`HeavyHittersMeasure::finalize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeavyHittersReport {
    /// Configured K (capacity).
    pub top_k: usize,
    /// Items in descending order of estimated frequency.
    pub items: Vec<HeavyHitterEntry>,
    /// Total observations that fed the sketch.
    pub observations: u64,
    /// Maximum possible undercount on any reported `count_lower_bound`:
    /// the true count is ≤ `count_lower_bound + error_bound`.
    pub error_bound: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeavyHitterEntry {
    pub value: String,
    pub count_lower_bound: u64,
}

// ---------------------------------------------------------------------------
// ExactFrequencyTable
// ---------------------------------------------------------------------------

/// Full distinct-value frequency table. Instantiated only when Pass
/// 1 confirmed cardinality is small (≤ `low_card_threshold`).
///
/// Reports exact mode, Shannon entropy, and the Gini coefficient
/// over the distinct-value distribution.
pub struct ExactFrequencyTable {
    counts: IndexMap<String, u64>,
    total: u64,
    /// Upper bound on tracked distinct values. If exceeded, the
    /// measure switches into "overflow" mode and stops adding new
    /// keys (preserving counts on those already tracked). This is a
    /// safety valve for the case where Pass 1's cardinality verdict
    /// was wrong; it should not normally trigger.
    cap: usize,
    overflowed: bool,
}

impl ExactFrequencyTable {
    pub fn new(cap: usize) -> Self {
        ExactFrequencyTable {
            counts: IndexMap::with_capacity(cap.min(1024)),
            total: 0,
            cap,
            overflowed: false,
        }
    }
}

impl Measure for ExactFrequencyTable {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let key = canonical_distinct_key(value);
        self.total += 1;
        if let Some(c) = self.counts.get_mut(&key) {
            *c += 1;
        } else if self.counts.len() < self.cap {
            self.counts.insert(key, 1);
        } else {
            self.overflowed = true;
            // Stop tracking new keys — preserves the invariant on
            // those already counted.
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let total = self.total;
        let total_tracked: u64 = self.counts.values().sum();
        // Shannon entropy (in bits, log base 2) and Gini, computed
        // over the distinct-value distribution.
        let (entropy, gini) = if total_tracked == 0 {
            (0.0, 0.0)
        } else {
            let mut h = 0f64;
            let mut sorted: Vec<u64> = self.counts.values().copied().collect();
            sorted.sort_unstable();
            // Entropy
            for &c in &sorted {
                if c == 0 { continue }
                let p = c as f64 / total_tracked as f64;
                h -= p * p.log2();
            }
            // Gini: 1 - Σ p_i^2 (Gini-Simpson form). Reports
            // "probability that two random draws are different".
            let mut g = 1.0;
            for &c in &sorted {
                let p = c as f64 / total_tracked as f64;
                g -= p * p;
            }
            (h, g.max(0.0))
        };
        let mode = self
            .counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| k.clone());
        MeasureReport::ExactFrequencyTable(ExactFrequencyTableReport {
            distinct_count: self.counts.len() as u32,
            total_observed: total,
            overflowed: self.overflowed,
            counts: self.counts,
            mode,
            entropy_bits: entropy,
            gini_simpson: gini,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ExactFrequencyTable
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactFrequencyTableReport {
    pub distinct_count: u32,
    pub total_observed: u64,
    /// True if the measure ran past its configured capacity. Should
    /// be `false` for fields where Pass 1's `LowCard` verdict was
    /// correct; `true` is a signal to re-run with a higher
    /// `--low-card-threshold` or to widen the template.
    pub overflowed: bool,
    /// Frequency table keyed by value-as-string.
    pub counts: IndexMap<String, u64>,
    /// Most-frequent value (or `None` if no observations).
    pub mode: Option<String>,
    /// Shannon entropy in bits.
    pub entropy_bits: f64,
    /// Gini-Simpson coefficient (1 - Σ p²).
    pub gini_simpson: f64,
}

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

/// Canonical key used by every cardinality measure so the same
/// value produces the same key across HLL / HeavyHitters /
/// ExactFrequencyTable. Renders MValues with a stable string
/// representation (mirroring the bounded-distinct tracker in
/// `ExplorationProbe`).
///
/// Long strings are truncated at the largest UTF-8 char boundary
/// `<= 256` bytes so the cap doesn't split a multi-byte character.
pub(crate) fn canonical_distinct_key(value: &MValue) -> String {
    let s = format!("{:?}", value);
    if s.len() <= 256 { return s; }
    let mut cut = 256;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s[..cut].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    /// Long strings containing multi-byte UTF-8 chars near the
    /// 256-byte cap must truncate cleanly without panicking on a
    /// char-boundary slice.
    #[test]
    fn canonical_distinct_key_truncates_on_char_boundary() {
        // Build a string whose 256th byte falls inside a 3-byte char.
        // 254 ASCII bytes + '…' (3 bytes) puts the ellipsis at bytes
        // 254..257, so a naive s[..256] would split it.
        let mut payload = "x".repeat(254);
        payload.push('…');
        payload.push_str(" tail");
        let v = MValue::Text(payload);
        let key = canonical_distinct_key(&v);
        // No panic, and the truncated string must end on a valid char
        // boundary (Rust's str::from_utf8 will reject anything else).
        assert!(key.len() <= 260, "key length: {}", key.len());
        let _ = std::str::from_utf8(key.as_bytes()).expect("valid utf8");
    }

    /// HLL reports an estimate near the true distinct count.
    #[test]
    fn hll_estimates_distinct_count() {
        let mut m = HyperLogLogMeasure::new(12);
        for i in 0..10_000 {
            m.observe(&MValue::Int(i), &ctx());
        }
        match Box::new(m).finalize() {
            MeasureReport::HyperLogLog(r) => {
                let band = 4.0 * r.stderr * 10_000.0;
                assert!(
                    (r.cardinality_estimate - 10_000.0).abs() < band,
                    "estimate {} should be near 10000 (band ±{})",
                    r.cardinality_estimate, band,
                );
                assert_eq!(r.observations, 10_000);
            }
            _ => panic!("wrong report kind"),
        }
    }

    /// Heavy hitters retains a dominant value.
    #[test]
    fn heavy_hitters_keeps_dominant() {
        let mut m = HeavyHittersMeasure::new(4);
        for _ in 0..200 {
            m.observe(&MValue::Text("popular".into()), &ctx());
        }
        for i in 0..100 {
            m.observe(&MValue::Text(format!("rare-{}", i)), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::HeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        // 300 observations, k=4 → threshold = 300/5 = 60. "popular"
        // at 200 must be retained.
        assert!(
            r.items.iter().any(|e| e.value.contains("popular")),
            "dominant value missing from {:?}", r.items
        );
        // Top entry is the dominant one.
        assert!(r.items[0].value.contains("popular"));
        assert_eq!(r.observations, 300);
    }

    /// Exact frequency table over a small low-cardinality stream.
    #[test]
    fn exact_frequency_table_low_card() {
        let mut m = ExactFrequencyTable::new(100);
        for i in 0..1_000 {
            m.observe(&MValue::Int(i % 5), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::ExactFrequencyTable(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.distinct_count, 5);
        assert_eq!(r.total_observed, 1_000);
        assert!(!r.overflowed);
        // Uniform 5-bin distribution: entropy = log2(5) ≈ 2.3219.
        assert!((r.entropy_bits - 5.0_f64.log2()).abs() < 1e-9);
        // Gini-Simpson for uniform-K: 1 - K · (1/K)² = 1 - 1/K = 0.8.
        assert!((r.gini_simpson - 0.8).abs() < 1e-9);
        // Mode exists.
        assert!(r.mode.is_some());
    }

    /// Cap exceeded → overflowed flag set, counts on existing keys
    /// preserved.
    #[test]
    fn exact_frequency_table_overflow_safety_valve() {
        let mut m = ExactFrequencyTable::new(4);
        // 10 distinct items.
        for i in 0..10 {
            m.observe(&MValue::Int(i), &ctx());
        }
        // Re-observe two of the originals so their counts grow.
        for _ in 0..5 {
            m.observe(&MValue::Int(0), &ctx());
            m.observe(&MValue::Int(1), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::ExactFrequencyTable(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert!(r.overflowed);
        assert_eq!(r.distinct_count, 4);
        // Items 0 and 1 should have count 6 (1 initial + 5 repeats).
        let c0 = r.counts.get("Int(0)").copied().unwrap_or(0);
        let c1 = r.counts.get("Int(1)").copied().unwrap_or(0);
        assert_eq!(c0, 6);
        assert_eq!(c1, 6);
    }

    /// Constant stream: entropy 0, Gini 0, mode is the single value.
    #[test]
    fn constant_stream_zero_entropy() {
        let mut m = ExactFrequencyTable::new(100);
        for _ in 0..50 {
            m.observe(&MValue::Bool(true), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::ExactFrequencyTable(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.distinct_count, 1);
        assert_eq!(r.entropy_bits, 0.0);
        assert_eq!(r.gini_simpson, 0.0);
        assert_eq!(r.mode.as_deref(), Some("Bool(true)"));
    }
}
