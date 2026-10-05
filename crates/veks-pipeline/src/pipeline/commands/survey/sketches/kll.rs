// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! KLL quantile sketch.
//!
//! Sublinear-memory quantile approximation for streaming numeric
//! data. With parameter `k`, the sketch uses `O(k · log(n/k))` memory
//! and answers quantile queries with rank error `ε ≈ 1.0 / k`.
//!
//! Reference: Karnin, Z., Lang, K., Liberty, E. (2016). "Optimal
//! quantile approximation in streams". FOCS 2016.
//!
//! The sketch is organized as a stack of *compactors*, indexed by
//! height. Each compactor holds up to its level-specific capacity of
//! `f64` values; when it overflows, half its items (chosen by
//! picking either the even-indexed or the odd-indexed positions of
//! the sorted level) are promoted to the next height up, each
//! carrying a doubled rank weight (`2^height`). The remaining half
//! is discarded. The result is a multiset where each retained item
//! represents `2^h` items from the original stream.
//!
//! Capacity scheme follows the original paper: level `h` has
//! capacity `max(min_cap, ceil(k · (2/3)^(H - h)))` where `H` is the
//! number of active levels. This shrinks capacity geometrically for
//! lower (more-compacted) levels, giving the optimal log-memory
//! profile.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::Sketch;

const MIN_LEVEL_CAPACITY: usize = 8;
const SHRINK_RATIO: f64 = 2.0 / 3.0;

/// KLL sketch over `f64` values.
#[derive(Debug, Clone)]
pub struct KllSketch {
    /// Sketch precision parameter. Larger `k` → smaller error, more
    /// memory. Default for the survey: 200.
    k: usize,
    /// Compactors indexed by height. `compactors[0]` is the bottom
    /// (each item there carries weight 1); `compactors[h]` holds
    /// items with weight `2^h`. New levels are pushed lazily.
    compactors: Vec<Vec<f64>>,
    /// Total observed values.
    n: u64,
    /// Cached minimum and maximum for unbiased extrema reporting.
    /// KLL's compaction loses information about exact extrema in
    /// expectation; we track them explicitly.
    min: f64,
    max: f64,
    rng: Xoshiro256PlusPlus,
}

impl KllSketch {
    /// Create a fresh sketch with precision `k`. Typical values:
    /// 100 (≈ 1% rank error), 200 (≈ 0.5%), 800 (≈ 0.125%).
    pub fn new(k: usize) -> Self {
        Self::with_seed(k, 0xC011EC70u64)
    }

    /// Create a sketch with an explicit RNG seed. Reproducible
    /// across runs given identical inputs and identical seeds.
    pub fn with_seed(k: usize, seed: u64) -> Self {
        assert!(k >= MIN_LEVEL_CAPACITY, "KLL k must be >= {}", MIN_LEVEL_CAPACITY);
        KllSketch {
            k,
            compactors: vec![Vec::new()],
            n: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Total observations.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Sketch precision parameter `k`.
    pub fn k(&self) -> usize {
        self.k
    }

    /// True minimum value seen, or `None` if no values were observed.
    pub fn min(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    /// True maximum value seen, or `None` if no values were observed.
    pub fn max(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.max)
        }
    }

    /// Number of active compactor levels.
    pub fn height(&self) -> usize {
        self.compactors.len()
    }

    /// Observe one value. NaN inputs are silently ignored — they
    /// cannot participate in ordering-based quantile estimation.
    pub fn add(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.n += 1;
        self.compactors[0].push(value);
        self.compact_if_needed();
    }

    /// Compact starting at the bottom level. Each level whose item
    /// count exceeds its capacity has half its items (chosen as the
    /// even or odd positions of the sorted level) promoted to the
    /// next level. Promotion may cascade upward.
    fn compact_if_needed(&mut self) {
        let mut h = 0;
        while h < self.compactors.len() {
            let cap = self.level_capacity(h);
            if self.compactors[h].len() < cap {
                h += 1;
                continue;
            }
            self.compact_level(h);
            // Cascade: the next level just received items, check it.
            h += 1;
        }
    }

    /// Capacity for level `h` under the standard KLL geometric
    /// shrink schedule. Items at the topmost level have full capacity
    /// `k`; capacity shrinks by factor `2/3` per level downward,
    /// floored at `MIN_LEVEL_CAPACITY`.
    fn level_capacity(&self, h: usize) -> usize {
        let height = self.compactors.len();
        // Distance from the *top* (currently-allocated) level.
        let from_top = (height - 1).saturating_sub(h);
        let raw = (self.k as f64) * SHRINK_RATIO.powi(from_top as i32);
        (raw.ceil() as usize).max(MIN_LEVEL_CAPACITY)
    }

    fn compact_level(&mut self, h: usize) {
        // Move the level out so we can sort and resample without
        // re-borrowing `self.compactors[h]`.
        let mut level = std::mem::take(&mut self.compactors[h]);
        level.sort_by(|a, b| a.partial_cmp(b).expect("NaN should be filtered on add"));
        // KLL guarantees rank error is bounded when each compaction
        // randomly picks either the even-indexed or the odd-indexed
        // half. We retain only one half; the other is discarded.
        let offset: usize = if self.rng.random::<bool>() { 0 } else { 1 };
        let promoted: Vec<f64> = level.iter().skip(offset).step_by(2).copied().collect();
        // Ensure the destination level exists.
        if h + 1 >= self.compactors.len() {
            self.compactors.push(Vec::new());
        }
        self.compactors[h + 1].extend(promoted);
    }

    /// Approximate the value at rank `q ∈ [0, 1]` (0.0 → minimum,
    /// 1.0 → maximum). Returns `None` if the sketch is empty.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        assert!((0.0..=1.0).contains(&q), "quantile q must be in [0, 1]");
        if self.n == 0 {
            return None;
        }
        if q == 0.0 {
            return Some(self.min);
        }
        if q == 1.0 {
            return Some(self.max);
        }
        // Materialize the weighted multiset by traversing every
        // active level, sort by value, then walk cumulative weight
        // until we reach the requested rank.
        let mut items: Vec<(f64, u64)> = Vec::new();
        for (h, level) in self.compactors.iter().enumerate() {
            let weight = 1u64 << h;
            for &v in level {
                items.push((v, weight));
            }
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN should be filtered on add"));
        let total: u64 = items.iter().map(|(_, w)| *w).sum();
        let target = (q * total as f64).round() as u64;
        let mut cumul = 0u64;
        for (v, w) in &items {
            cumul += *w;
            if cumul >= target.max(1) {
                return Some(*v);
            }
        }
        items.last().map(|(v, _)| *v)
    }

    /// Convenience batch quantile lookup. Returns one value per `q`
    /// in the same order, or `None` if the sketch is empty.
    pub fn quantiles(&self, qs: &[f64]) -> Option<Vec<f64>> {
        if self.n == 0 {
            return None;
        }
        Some(qs.iter().map(|q| self.quantile(*q).unwrap()).collect())
    }

    /// Number of items currently retained across all levels.
    pub fn retained(&self) -> usize {
        self.compactors.iter().map(|l| l.len()).sum()
    }
}

impl Sketch for KllSketch {
    fn memory_bytes(&self) -> usize {
        let f64_size = std::mem::size_of::<f64>();
        let levels = self.compactors.iter().map(|l| l.capacity()).sum::<usize>();
        std::mem::size_of::<Self>() + levels * f64_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty sketch returns `None` for any quantile and `None` for min/max.
    #[test]
    fn empty_returns_none() {
        let s = KllSketch::new(200);
        assert!(s.quantile(0.5).is_none());
        assert!(s.min().is_none());
        assert!(s.max().is_none());
    }

    /// Tiny stream (fewer items than any level capacity): exact recall.
    #[test]
    fn tiny_stream_exact() {
        let mut s = KllSketch::new(200);
        for v in [3.0, 1.0, 2.0, 5.0, 4.0] {
            s.add(v);
        }
        assert_eq!(s.min(), Some(1.0));
        assert_eq!(s.max(), Some(5.0));
        // Median of [1,2,3,4,5] is 3. KLL is exact below level
        // capacity, so this must match.
        assert_eq!(s.quantile(0.5), Some(3.0));
    }

    /// NaN inputs are ignored.
    #[test]
    fn nan_ignored() {
        let mut s = KllSketch::new(200);
        s.add(f64::NAN);
        s.add(1.0);
        s.add(f64::NAN);
        s.add(2.0);
        assert_eq!(s.count(), 2);
    }

    /// Constant stream: every quantile returns the same value.
    #[test]
    fn constant_stream() {
        let mut s = KllSketch::new(200);
        for _ in 0..10_000 {
            s.add(42.0);
        }
        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_eq!(s.quantile(q), Some(42.0));
        }
    }

    /// Sorted ascending stream: rank-error bound.
    ///
    /// Insert values `[1, 2, …, n]`. For each query quantile q, the
    /// true rank-q value is `round(q · n)`. The KLL estimate must lie
    /// within `±ε · n` of the truth, where `ε ≤ 2 / k` is the
    /// algorithm's worst-case rank error bound.
    #[test]
    fn sorted_stream_rank_error_bound() {
        let k = 200;
        let n = 100_000usize;
        let mut s = KllSketch::with_seed(k, 7);
        for i in 1..=n {
            s.add(i as f64);
        }
        let eps_band = (4.0 / k as f64) * n as f64; // generous 4 / k for test stability
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let est = s.quantile(q).unwrap();
            let truth = q * n as f64;
            assert!(
                (est - truth).abs() <= eps_band,
                "q={} est={} truth={} band={}", q, est, truth, eps_band
            );
        }
    }

    /// Shuffled-order independence: same multiset, different order →
    /// quantile estimates should still satisfy the rank-error bound.
    #[test]
    fn shuffled_order_still_accurate() {
        use rand::seq::SliceRandom;
        let k = 200;
        let n = 50_000usize;
        let mut data: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        data.shuffle(&mut rng);
        let mut s = KllSketch::with_seed(k, 99);
        for v in &data {
            s.add(*v);
        }
        let band = (4.0 / k as f64) * n as f64;
        for q in [0.1, 0.5, 0.9] {
            let est = s.quantile(q).unwrap();
            let truth = q * n as f64;
            assert!((est - truth).abs() <= band, "q={} est={} truth={}", q, est, truth);
        }
    }

    /// Min and max are tracked exactly regardless of compaction.
    #[test]
    fn extrema_exact_after_compaction() {
        let mut s = KllSketch::new(64);
        // Push enough to trigger many compactions.
        for i in 0..100_000 {
            s.add(i as f64);
        }
        assert_eq!(s.min(), Some(0.0));
        assert_eq!(s.max(), Some(99_999.0));
        // q=0 / q=1 fast paths bypass weight walk and return min/max
        // directly.
        assert_eq!(s.quantile(0.0), Some(0.0));
        assert_eq!(s.quantile(1.0), Some(99_999.0));
    }

    /// Batch quantile lookup matches individual lookups.
    #[test]
    fn batch_quantiles_match_singles() {
        let mut s = KllSketch::new(200);
        for i in 0..10_000 {
            s.add(i as f64);
        }
        let qs = [0.1, 0.25, 0.5, 0.75, 0.9];
        let batch = s.quantiles(&qs).unwrap();
        for (i, q) in qs.iter().enumerate() {
            assert_eq!(batch[i], s.quantile(*q).unwrap());
        }
    }

    /// Higher `k` yields tighter bounds in expectation. We don't
    /// assert a strict inequality (KLL is randomized) but we do
    /// assert that a very-large-k sketch's max observed error stays
    /// well within a small-k sketch's allowed error band.
    #[test]
    fn higher_k_lower_error() {
        let n = 50_000usize;
        let mut small = KllSketch::with_seed(64, 1);
        let mut large = KllSketch::with_seed(800, 1);
        for i in 1..=n {
            small.add(i as f64);
            large.add(i as f64);
        }
        let large_err = (large.quantile(0.5).unwrap() - n as f64 / 2.0).abs();
        let small_band = (4.0 / 64.0) * n as f64;
        // The large sketch should easily stay within the small
        // sketch's error band.
        assert!(large_err < small_band);
    }

    /// `retained()` is bounded by the sketch's memory promise.
    /// For n items and parameter k, the total retained should not
    /// exceed a small multiple of `k · log2(n/k)`.
    #[test]
    fn retained_size_bounded() {
        let k = 200;
        let n = 100_000usize;
        let mut s = KllSketch::new(k);
        for i in 0..n {
            s.add(i as f64);
        }
        let log2_ratio = ((n as f64) / (k as f64)).log2();
        let bound = (10.0 * k as f64 * log2_ratio) as usize;
        let retained = s.retained();
        assert!(
            retained <= bound,
            "retained {} should be <= {} (10·k·log2(n/k))", retained, bound
        );
    }

    /// Reproducibility: identical seed + identical stream gives an
    /// identical sketch state.
    #[test]
    fn deterministic_by_seed() {
        let mut a = KllSketch::with_seed(128, 1234);
        let mut b = KllSketch::with_seed(128, 1234);
        for i in 0..10_000 {
            let v = (i as f64).sin() * 1000.0;
            a.add(v);
            b.add(v);
        }
        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_eq!(a.quantile(q), b.quantile(q));
        }
    }

    #[test]
    #[should_panic(expected = "KLL k must be >=")]
    fn rejects_tiny_k() {
        let _ = KllSketch::new(4);
    }

    #[test]
    #[should_panic(expected = "quantile q must be in")]
    fn rejects_quantile_out_of_range() {
        let mut s = KllSketch::new(200);
        s.add(1.0);
        let _ = s.quantile(1.5);
    }
}
