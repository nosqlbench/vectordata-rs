// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Latency distributions, because a mean is not a validation.
//!
//! Throughput is one number and it hides everything. Two models can
//! agree on operations per second while disagreeing completely about what
//! any individual request experienced — one serving every request in the
//! mean time, the other serving most instantly and a few catastrophically
//! late. The measurements this crate calibrates against report full
//! percentile breakdowns, and the storage-simulation literature validates
//! against latency as well as throughput
//! ([MQSim](https://www.usenix.org/conference/fast18/presentation/tavakkol),
//! [SimpleSSD](https://arxiv.org/pdf/1705.06419)), so not using them was
//! leaving the harder half of the check on the table.
//!
//! Storage latencies span seven orders of magnitude — tens of
//! microseconds on flash to hundreds of milliseconds on a queued disk —
//! so buckets are logarithmic: each octave is divided into
//! [`SUB_BUCKETS`] parts, giving a bounded relative error everywhere
//! rather than absolute precision somewhere and none elsewhere.

/// Divisions per power of two. 32 bounds relative error at about 1.6%,
/// comfortably finer than the agreement anyone claims for a storage
/// model.
pub const SUB_BUCKETS: usize = 32;

/// Smallest latency resolved, in seconds.
const MIN_SECONDS: f64 = 100e-9;

/// Octaves covered above [`MIN_SECONDS`] — 100 ns to about 100 s.
const OCTAVES: usize = 30;

/// A logarithmic latency histogram.
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    buckets: Vec<u64>,
    count: u64,
    sum: f64,
    sum_squares: f64,
    min: f64,
    max: f64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    pub fn new() -> Self {
        LatencyHistogram {
            buckets: vec![0; OCTAVES * SUB_BUCKETS],
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            min: f64::INFINITY,
            max: 0.0,
        }
    }

    fn index_of(seconds: f64) -> usize {
        if seconds <= MIN_SECONDS {
            return 0;
        }
        let ratio = seconds / MIN_SECONDS;
        let octave = ratio.log2();
        let idx = (octave * SUB_BUCKETS as f64) as usize;
        idx.min(OCTAVES * SUB_BUCKETS - 1)
    }

    /// Representative value for a bucket — its geometric midpoint, so the
    /// error is symmetric in the log domain the buckets live in.
    fn value_of(index: usize) -> f64 {
        let octave = (index as f64 + 0.5) / SUB_BUCKETS as f64;
        MIN_SECONDS * octave.exp2()
    }

    pub fn record(&mut self, seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        self.buckets[Self::index_of(seconds)] += 1;
        self.count += 1;
        self.sum += seconds;
        self.sum_squares += seconds * seconds;
        self.min = self.min.min(seconds);
        self.max = self.max.max(seconds);
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    pub fn stdev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.mean();
        ((self.sum_squares / n) - mean * mean).max(0.0).sqrt()
    }

    pub fn min(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.min }
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    /// The value below which `p` percent of samples fall.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let target = (p / 100.0 * self.count as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (i, &n) in self.buckets.iter().enumerate() {
            seen += n;
            if seen >= target {
                return Self::value_of(i);
            }
        }
        self.max
    }

    /// Merge another histogram into this one.
    pub fn absorb(&mut self, other: &LatencyHistogram) {
        for (a, b) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            *a += b;
        }
        self.count += other.count;
        self.sum += other.sum;
        self.sum_squares += other.sum_squares;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// The percentiles fio reports, for direct comparison.
    pub fn summary(&self) -> LatencySummary {
        LatencySummary {
            count: self.count,
            mean: self.mean(),
            stdev: self.stdev(),
            min: self.min(),
            p50: self.percentile(50.0),
            p95: self.percentile(95.0),
            p99: self.percentile(99.0),
            p999: self.percentile(99.9),
            max: self.max(),
        }
    }
}

/// A latency distribution in the shape fio prints one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LatencySummary {
    pub count: u64,
    pub mean: f64,
    pub stdev: f64,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub max: f64,
}

impl LatencySummary {
    /// How heavy the tail is. A model that gets the mean right and this
    /// wrong is not modelling the same system.
    pub fn tail_ratio(&self) -> f64 {
        if self.p50 <= 0.0 {
            0.0
        } else {
            self.p99 / self.p50
        }
    }

    pub fn micros(&self) -> LatencySummary {
        LatencySummary {
            count: self.count,
            mean: self.mean * 1e6,
            stdev: self.stdev * 1e6,
            min: self.min * 1e6,
            p50: self.p50 * 1e6,
            p95: self.p95 * 1e6,
            p99: self.p99 * 1e6,
            p999: self.p999 * 1e6,
            max: self.max * 1e6,
        }
    }
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_a_uniform_sample_are_where_they_should_be() {
        let mut h = LatencyHistogram::new();
        for i in 1..=10_000 {
            h.record(i as f64 * 1e-6);
        }
        // Bucket resolution is ~1.6%, so allow a few percent.
        assert!((h.percentile(50.0) - 5.0e-3).abs() / 5.0e-3 < 0.05);
        assert!((h.percentile(99.0) - 9.9e-3).abs() / 9.9e-3 < 0.05);
        assert!((h.mean() - 5.0005e-3).abs() / 5.0e-3 < 0.01);
    }

    #[test]
    fn the_histogram_spans_the_range_storage_actually_uses() {
        let mut h = LatencyHistogram::new();
        // 20 µs on flash through 600 ms on a queued disk.
        for v in [20e-6, 80e-6, 1e-3, 40e-3, 600e-3] {
            h.record(v);
        }
        assert_eq!(h.count(), 5);
        assert!((h.min() - 20e-6).abs() < 1e-9);
        assert!((h.max() - 600e-3).abs() < 1e-9);
        // Each value must land in its own bucket, not saturate an end.
        assert!(h.percentile(1.0) < 30e-6);
        assert!(h.percentile(99.0) > 400e-3);
    }

    #[test]
    fn a_constant_latency_has_no_spread() {
        let mut h = LatencyHistogram::new();
        for _ in 0..1_000 {
            h.record(604e-6);
        }
        let s = h.summary();
        assert_eq!(
            s.p50, s.p99,
            "identical samples must give identical percentiles"
        );
        assert!(s.stdev < 1e-9);
        assert!((s.tail_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn merging_preserves_the_distribution() {
        let mut a = LatencyHistogram::new();
        let mut b = LatencyHistogram::new();
        let mut whole = LatencyHistogram::new();
        for i in 1..=500 {
            a.record(i as f64 * 1e-6);
            whole.record(i as f64 * 1e-6);
        }
        for i in 501..=1_000 {
            b.record(i as f64 * 1e-6);
            whole.record(i as f64 * 1e-6);
        }
        a.absorb(&b);
        assert_eq!(a.count(), whole.count());
        assert!((a.percentile(90.0) - whole.percentile(90.0)).abs() < 1e-9);
    }
}
