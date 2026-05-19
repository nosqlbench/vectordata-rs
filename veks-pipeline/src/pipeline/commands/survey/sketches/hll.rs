// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! HyperLogLog cardinality estimation.
//!
//! Tracks the approximate count of distinct items in a stream using
//! constant memory. With precision `p`, the sketch uses `2^p` 8-bit
//! registers (≈ 4 KB at `p = 12`) and produces estimates with
//! relative standard error `σ ≈ 1.04 / √(2^p)` (≈ 1.6% at `p = 12`).
//!
//! Reference: Flajolet, Fusy, Gandouet, Meunier (2007). "HyperLogLog:
//! the analysis of a near-optimal cardinality estimation algorithm".
//! With the bias / small-range corrections from Heule, Nunkesser,
//! Hall (2013), "HyperLogLog in Practice".
//!
//! Hashing: items are hashed with the xxh3 64-bit hash via
//! [`std::hash::Hash`]. The high `p` bits select the register; the
//! remaining bits feed the leading-zero estimator.

use std::hash::{Hash, Hasher};

use super::Sketch;

const MIN_PRECISION: u8 = 4;
const MAX_PRECISION: u8 = 18;

/// HyperLogLog distinct-count estimator.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    /// Precision: number of leading hash bits used as the register
    /// index. `2^precision` registers in total.
    precision: u8,
    /// Per-register maximum (`leading zero count + 1`) over hashes
    /// routed to that register. Stored as u8 because `64 - precision`
    /// fits comfortably.
    registers: Vec<u8>,
    /// Salt mixed into every hash. Distinct sketches can use distinct
    /// salts to avoid systematic register coincidences, though for
    /// the survey use case a single fixed salt suffices.
    seed: u64,
    /// Total `add` calls. Not strictly required for estimation but
    /// surfaced in snapshots for sanity-checking.
    observed: u64,
}

impl HyperLogLog {
    /// Create an HLL sketch with the given precision. Valid range is
    /// `[MIN_PRECISION, MAX_PRECISION] = [4, 18]`. Default
    /// recommendation for the survey is `p = 12`.
    pub fn new(precision: u8) -> Self {
        Self::with_seed(precision, 0)
    }

    /// Create an HLL sketch with a custom hash salt. Tests rely on
    /// this to construct sketches with predictable behavior; the
    /// survey orchestrator uses the default salt.
    pub fn with_seed(precision: u8, seed: u64) -> Self {
        assert!(
            (MIN_PRECISION..=MAX_PRECISION).contains(&precision),
            "HLL precision must be in [{}, {}]",
            MIN_PRECISION,
            MAX_PRECISION
        );
        let m = 1usize << precision;
        HyperLogLog {
            precision,
            registers: vec![0u8; m],
            seed,
            observed: 0,
        }
    }

    /// Number of registers (`m = 2^precision`).
    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    /// Precision parameter (`p`).
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// Number of `add` calls, regardless of whether they affected a
    /// register.
    pub fn observed(&self) -> u64 {
        self.observed
    }

    /// Observe one item.
    pub fn add<H: Hash + ?Sized>(&mut self, item: &H) {
        self.observed += 1;
        let h = self.hash_with_seed(item);
        let idx = (h >> (64 - self.precision)) as usize;
        // The remaining (64 - p) bits feed leading-zero counting.
        // Shift the index bits out and count zeros from the top.
        let remaining = h << self.precision;
        // `leading_zeros` on the shifted remainder counts zeros
        // within the (64 - p)-bit window. The "+ 1" matches the
        // canonical ρ(x) definition.
        let rank = if remaining == 0 {
            64u32 - self.precision as u32 + 1
        } else {
            remaining.leading_zeros() + 1
        };
        let rank = rank as u8;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Current cardinality estimate.
    ///
    /// Uses the standard harmonic-mean estimator with the small-range
    /// linear-counting correction from Heule et al. The large-range
    /// correction is omitted because at 64-bit hashes the
    /// asymptotic-overflow regime is unreachable for any realistic
    /// stream.
    pub fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let alpha = alpha_m(self.registers.len());
        let mut sum = 0f64;
        let mut zeros = 0usize;
        for &r in &self.registers {
            sum += 2f64.powi(-(r as i32));
            if r == 0 {
                zeros += 1;
            }
        }
        let raw = alpha * m * m / sum;
        // Small-range correction: when more than ~m/3 registers are
        // empty, linear counting is more accurate. The exact
        // threshold from Heule et al is 2.5 · m; the linear-counting
        // result is `m · ln(m / zeros)`.
        if raw <= 2.5 * m {
            if zeros > 0 {
                return m * (m / zeros as f64).ln();
            }
        }
        raw
    }

    /// Theoretical relative standard error: `1.04 / √m`.
    pub fn relative_stderr(&self) -> f64 {
        1.04 / (self.registers.len() as f64).sqrt()
    }

    /// Borrow the raw register array.
    pub fn registers(&self) -> &[u8] {
        &self.registers
    }

    fn hash_with_seed<H: Hash + ?Sized>(&self, item: &H) -> u64 {
        // SipHasher13 isn't directly exposed; use the default
        // hasher and XOR in the salt. For survey scale this is
        // sufficient — the registers' statistical behavior depends
        // only on hash uniformity, which both Rust's default and
        // xxh3 satisfy.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish()
    }
}

impl Sketch for HyperLogLog {
    fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.registers.len()
    }
}

/// HLL bias-correction constant α_m. Values for `m ∈ {16, 32, 64}` are
/// the originals from Flajolet et al.; for `m ≥ 128` the asymptotic
/// formula is used.
fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty sketch reports cardinality 0 (linear-counting path with
    /// every register empty → `m · ln(m/m) = 0`).
    #[test]
    fn empty_estimates_zero() {
        let h = HyperLogLog::new(10);
        assert_eq!(h.estimate(), 0.0);
    }

    /// One distinct item: linear-counting branch returns
    /// `m · ln(m / (m-1))`. For p=10 (m=1024) this is ~1.0005, well
    /// within 5% of the true count of 1.
    #[test]
    fn small_cardinality_linear_branch() {
        let mut h = HyperLogLog::new(10);
        h.add("only");
        let est = h.estimate();
        assert!(
            (est - 1.0).abs() < 0.5,
            "estimate {} should be close to 1", est
        );
    }

    /// Inserting the same item many times must still estimate ~1
    /// because HLL deduplicates by hash → register state is fixed
    /// after the first observation that updates a register.
    #[test]
    fn duplicates_collapse() {
        let mut h = HyperLogLog::new(10);
        for _ in 0..10_000 {
            h.add("same");
        }
        let est = h.estimate();
        assert!(est < 5.0, "duplicates should estimate near 1, got {}", est);
    }

    /// Insert N distinct items; verify the estimate is within the
    /// theoretical 4σ band. With p=12 (m=4096), σ ≈ 1.04/√4096 ≈ 1.625%.
    /// 4σ ≈ 6.5%, comfortably wide.
    #[test]
    fn medium_cardinality_within_4sigma() {
        let mut h = HyperLogLog::new(12);
        let n = 50_000usize;
        for i in 0..n {
            h.add(&format!("user-{}", i));
        }
        let est = h.estimate();
        let stderr = h.relative_stderr();
        let band = 4.0 * stderr * n as f64;
        assert!(
            (est - n as f64).abs() < band,
            "estimate {} should be within ±{} of {}", est, band, n
        );
    }

    /// Independent sketches with different seeds give different
    /// estimates but both within the expected error band.
    #[test]
    fn estimates_consistent_across_seeds() {
        let n = 20_000usize;
        let mut a = HyperLogLog::with_seed(12, 1);
        let mut b = HyperLogLog::with_seed(12, 2);
        for i in 0..n {
            let s = format!("k-{}", i);
            a.add(&s);
            b.add(&s);
        }
        let stderr = a.relative_stderr();
        let band = 4.0 * stderr * n as f64;
        assert!((a.estimate() - n as f64).abs() < band);
        assert!((b.estimate() - n as f64).abs() < band);
    }

    /// Higher precision yields lower theoretical error.
    #[test]
    fn higher_precision_lower_stderr() {
        let small = HyperLogLog::new(8);
        let large = HyperLogLog::new(14);
        assert!(large.relative_stderr() < small.relative_stderr());
    }

    /// Memory footprint scales with `2^precision`.
    #[test]
    fn memory_scales_with_precision() {
        let a = HyperLogLog::new(8);
        let b = HyperLogLog::new(14);
        assert!(b.memory_bytes() > a.memory_bytes() * 50);
    }

    /// Bias-correction constants match the published HLL paper
    /// exactly for the small-m branches, and the asymptotic formula
    /// for m=128 converges toward 0.7213 from below as expected.
    #[test]
    fn alpha_constants_match_paper() {
        assert_eq!(alpha_m(16), 0.673);
        assert_eq!(alpha_m(32), 0.697);
        assert_eq!(alpha_m(64), 0.709);
        // α_m(m) = 0.7213 / (1 + 1.079/m). For m=128 that is
        // 0.7213 / 1.00843 ≈ 0.7152.
        let a128 = alpha_m(128);
        assert!(
            (0.714..0.716).contains(&a128),
            "alpha_m(128) = {} should be near 0.7152", a128
        );
        // Asymptote: alpha grows toward 0.7213 as m → ∞.
        let a_large = alpha_m(1 << 18);
        assert!(a_large > 0.72 && a_large < 0.7213);
    }

    #[test]
    #[should_panic(expected = "HLL precision must be in")]
    fn precision_below_minimum_panics() {
        let _ = HyperLogLog::new(2);
    }

    #[test]
    #[should_panic(expected = "HLL precision must be in")]
    fn precision_above_maximum_panics() {
        let _ = HyperLogLog::new(20);
    }
}
