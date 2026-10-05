// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Choosing how many records go in a shard.
//!
//! A facet that would be written as one enormous file is split into a
//! series instead, capped at a maximum file size. The cap exists
//! because filesystems and transfer tools still carry 2 TB limits;
//! 1 TB is the default because it is a round number well under that,
//! with room for the estimate to be wrong.
//!
//! The calculation is the inverse of
//! [`VecFormat::expected_file_size`](../../../veks_core/formats/enum.VecFormat.html#method.expected_file_size):
//! that answers "how many bytes will `n` records take", and this
//! answers "how many records fit in `b` bytes". It is deliberately
//! *not* the exact inverse, in two ways.
//!
//! **The stride is a power of ten** (SH-2's ordinal algebra is
//! `shard(o) = o / stride`, `local(o) = o % stride`). A stride of
//! 100,000,000 makes ordinal 715,000,000 shard 7, local 15,000,000 —
//! readable by a person looking at a filename and an ordinal. An exact
//! stride of 715,896,331 makes that arithmetic unreadable and buys
//! nothing: the cap is a ceiling, not a target.
//!
//! **A format without a fixed stride is sampled, not computed.** A
//! slab record and a vvec record carry their own length, so there is
//! no record size to divide by — only a measurement, which the sample
//! supplies and a margin protects.
//!
//! See `docs/design/srd-multifile-facet-shards.md`.

use std::fmt;

/// The default cap on one shard file: 1 TB.
///
/// Decimal, not binary — the limits this avoids are quoted in decimal
/// terabytes, and a `TiB` here would put the answer on the wrong side
/// of a 2 TB ceiling by 10%.
pub const DEFAULT_MAX_SHARD_BYTES: u64 = 1_000_000_000_000;

/// The margin applied to a sampled record size.
///
/// A sample measures the records it saw, and the mean of the whole
/// facet may be larger. Doubling the per-record budget means the
/// projection stays under the cap as long as the true mean is no more
/// than twice what the sample found.
pub const SAMPLE_MARGIN: u64 = 2;

/// What a per-record byte budget was derived from.
///
/// Kept in the plan rather than collapsed to a number, so a report can
/// say *why* a stride is what it is — and so a sampled plan is
/// visibly an estimate rather than an arithmetic fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordSize {
    /// Every record is exactly this many bytes.
    ///
    /// For a uniform xvec that is `4 + dim * element_size`; for a
    /// packed scalar it is the element width.
    Fixed(u64),
    /// Records vary, and this is what a sample measured.
    ///
    /// `mean` drives the budget, because total file size is
    /// `records * mean` — the largest record seen affects *whether one
    /// record fits*, not how many of them do.
    Sampled {
        /// Mean record length across the sample, in bytes.
        mean: u64,
        /// The largest record the sample saw.
        max: u64,
        /// How many records were measured.
        sampled: u64,
    },
}

impl RecordSize {
    /// The per-record byte budget this basis implies.
    ///
    /// A fixed size is used as-is: there is nothing to be wrong about.
    /// A sampled size carries [`SAMPLE_MARGIN`], and never budgets less
    /// than the largest record seen — a shard has to be able to hold
    /// one whole record whatever the mean says.
    pub fn budget(self) -> u64 {
        match self {
            Self::Fixed(n) => n,
            Self::Sampled { mean, max, .. } => (mean * SAMPLE_MARGIN).max(max),
        }
    }

    /// Whether this basis is a measurement rather than an identity.
    pub fn is_estimate(self) -> bool {
        matches!(self, Self::Sampled { .. })
    }
}

impl fmt::Display for RecordSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(n) => write!(f, "{n} bytes per record"),
            Self::Sampled { mean, max, sampled } => write!(
                f,
                "~{mean} bytes per record (mean of {sampled} sampled, largest {max})"
            ),
        }
    }
}

/// A shard stride and the reasoning that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardPlan {
    /// Ordinals per shard — always a power of ten.
    pub stride: u64,
    /// The per-record budget the stride was divided from.
    pub record_bytes: u64,
    /// The cap this plan respects.
    pub max_bytes: u64,
    /// What the per-record budget came from.
    pub basis: RecordSize,
}

impl ShardPlan {
    /// The size a full shard is projected to reach.
    ///
    /// At or under [`Self::max_bytes`] by construction. For a sampled
    /// plan it is a projection; for a fixed one it is exact.
    pub fn projected_bytes(&self) -> u64 {
        self.stride * self.record_bytes
    }

    /// How much of the cap a full shard is projected to use, 0.0–1.0.
    ///
    /// Flooring to a power of ten leaves this anywhere from just over
    /// 0.1 to 1.0. That is the cost of a readable stride, and it is
    /// reported rather than hidden so an operator can raise the cap if
    /// the utilization bothers them.
    pub fn utilization(&self) -> f64 {
        if self.max_bytes == 0 {
            return 0.0;
        }
        self.projected_bytes() as f64 / self.max_bytes as f64
    }

    /// How many shards `records` ordinals would occupy under this plan.
    pub fn shards_for(&self, records: u64) -> u64 {
        records.div_ceil(self.stride)
    }
}

impl fmt::Display for ShardPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} records per shard ({}, {} projected per file)",
            self.stride,
            self.basis,
            crate::datasets::filter::format_bytes_approx(self.projected_bytes()),
        )
    }
}

/// The `upstream.defaults` key a dataset declares its shard cap under.
///
/// The YAML mirror of `--resources shardsize:`. Named once here so the
/// writer that emits it, the reader that seeds a governor from it, and
/// the wizard that prompts for it cannot spell it differently.
pub const SHARD_SIZE_KEY: &str = "shard_size";

/// How a facet's shard stride is decided.
///
/// One parameter rather than a stride *and* a cap, so "both given" is
/// not a state a caller can reach and no code has to decide which of
/// two answers wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sharding {
    /// One file per facet, whatever its size.
    #[default]
    Whole,
    /// A stride the caller named, in records per shard.
    ///
    /// Used verbatim: an operator who states a stride has a reason,
    /// and second-guessing it against a cap would make the flag mean
    /// something other than what it says.
    Stride(u64),
    /// Whatever stride keeps one shard under this many bytes.
    MaxBytes(u64),
}

impl Sharding {
    /// The cap this asks for, if it is expressed as one.
    pub fn max_bytes(self) -> Option<u64> {
        match self {
            Self::MaxBytes(b) => Some(b),
            Self::Whole | Self::Stride(_) => None,
        }
    }

    /// The stride to use for records of a known fixed size.
    ///
    /// `None` means "write this facet whole" — either because nothing
    /// was asked for, or because the cap is roomy enough that a stride
    /// would name a series of one.
    pub fn stride_for_fixed(self, record_bytes: u64) -> Option<u64> {
        match self {
            Self::Whole => None,
            Self::Stride(n) => Some(n),
            Self::MaxBytes(b) => plan_fixed(b, record_bytes).map(|p| p.stride),
        }
    }

    /// The stride to use given a measured record size.
    ///
    /// An explicit stride still wins: the measurement exists to answer
    /// a cap, and a caller who named a stride did not ask a question.
    pub fn stride_for_sampled(self, basis: RecordSize) -> Option<u64> {
        match self {
            Self::Whole => None,
            Self::Stride(n) => Some(n),
            Self::MaxBytes(b) => plan(b, basis).map(|p| p.stride),
        }
    }

    /// Whether this asks for any sharding at all.
    pub fn is_requested(self) -> bool {
        !matches!(self, Self::Whole)
    }

    /// Resolve the two command-line surfaces into one choice.
    ///
    /// Both accept the suffixes
    /// [`parse_number_with_suffix`](crate::dataset::source::parse_number_with_suffix)
    /// reads, so `--shard-stride 100M` and `--max-shard-bytes 1TB`
    /// both work. Naming both is refused rather than resolved by a
    /// precedence rule nobody would remember.
    pub fn from_flags(stride: Option<&str>, max_bytes: Option<&str>) -> Result<Self, String> {
        let parse = |v: &str, flag: &str| {
            crate::dataset::source::parse_number_with_suffix(v)
                .map_err(|e| format!("--{flag}: {e}"))
        };
        match (stride, max_bytes) {
            (Some(_), Some(_)) => Err(
                "--shard-stride and --max-shard-bytes both set a shard size; give one. \
                 A stride names the records per file directly; a cap asks for whatever \
                 stride keeps a file under a size."
                    .to_string(),
            ),
            (Some(v), None) => Ok(Self::Stride(parse(v, "shard-stride")?)),
            (None, Some(v)) => Ok(Self::MaxBytes(parse(v, "max-shard-bytes")?)),
            (None, None) => Ok(Self::Whole),
        }
    }
}

/// The largest power of ten less than or equal to `n`.
///
/// `0` for `n == 0`: no positive stride fits, and returning 1 would
/// claim one record per shard.
pub fn floor_to_decade(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut decade = 1u64;
    // Stop before overflowing rather than after: the next decade past
    // 10^19 is not representable.
    while decade <= n / 10 {
        decade *= 10;
    }
    decade
}

/// Plan a shard stride for records of the given size under `max_bytes`.
///
/// `None` when not even ten records fit in the cap — a stride of one
/// would name a file per record, which is a misconfiguration rather
/// than a layout, and the caller should say so rather than emit it.
pub fn plan(max_bytes: u64, basis: RecordSize) -> Option<ShardPlan> {
    let record_bytes = basis.budget();
    if record_bytes == 0 || max_bytes == 0 {
        return None;
    }
    let stride = floor_to_decade(max_bytes / record_bytes);
    // A stride below ten means the cap cannot hold a meaningful run.
    (stride >= 10).then_some(ShardPlan {
        stride,
        record_bytes,
        max_bytes,
        basis,
    })
}

/// Plan a stride for a facet whose records are a fixed width.
pub fn plan_fixed(max_bytes: u64, record_bytes: u64) -> Option<ShardPlan> {
    plan(max_bytes, RecordSize::Fixed(record_bytes))
}

/// The fixed record size of a uniform xvec: a 4-byte dimension header
/// followed by `dim` elements.
pub fn xvec_record_bytes(dim: u64, element_bytes: u64) -> u64 {
    4 + dim * element_bytes
}

/// Summarize measured record lengths as a sampling basis.
///
/// `None` for an empty sample: nothing was measured, so there is
/// nothing to project from, and inventing a mean of zero would produce
/// an unbounded stride.
pub fn sample_of(lengths: &[u64]) -> Option<RecordSize> {
    if lengths.is_empty() {
        return None;
    }
    let total: u64 = lengths.iter().sum();
    Some(RecordSize::Sampled {
        // Round the mean up: a fractional byte per record is real
        // bytes across a hundred million of them.
        mean: total.div_ceil(lengths.len() as u64),
        max: lengths.iter().copied().max().unwrap_or(0),
        sampled: lengths.len() as u64,
    })
}

/// How many records to measure when sampling a variable-length facet.
///
/// Enough that the mean is stable across the kinds of skew real
/// metadata has, few enough that sampling a remote facet costs a
/// handful of pages rather than the file.
pub const DEFAULT_SAMPLE_RECORDS: u64 = 1_000;

/// Measure record lengths at evenly spaced ordinals across a facet.
///
/// **Spread, not prefix.** The first thousand records of a facet are
/// frequently unrepresentative — sorted by a key, written by one
/// producer, or holding a header-ish run — and a prefix sample would
/// inherit that skew silently. Even spacing costs the same number of
/// reads and cannot be fooled by ordering.
///
/// `len_of` is called with ascending ordinals, so a reader that walks
/// forward stays forward. `None` when the facet is empty.
pub fn sample_spread<E, F>(
    total: u64,
    target: u64,
    mut len_of: F,
) -> Result<Option<RecordSize>, E>
where
    F: FnMut(u64) -> Result<u64, E>,
{
    if total == 0 || target == 0 {
        return Ok(None);
    }
    let count = target.min(total);
    let mut lengths = Vec::with_capacity(count as usize);
    for i in 0..count {
        // Spaced across the whole range, first and last included when
        // the sample is smaller than the facet.
        let ordinal = if count == 1 { 0 } else { i * (total - 1) / (count - 1) };
        lengths.push(len_of(ordinal)?);
    }
    Ok(sample_of(&lengths))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The floor is a power of ten at or below its input, and it is
    /// *exact* at every decade boundary — the place an off-by-one
    /// would put a whole extra shard in every series.
    #[test]
    fn the_decade_floor_lands_on_powers_of_ten() {
        assert_eq!(floor_to_decade(0), 0);
        assert_eq!(floor_to_decade(1), 1);
        assert_eq!(floor_to_decade(9), 1);
        assert_eq!(floor_to_decade(10), 10, "a decade is its own floor");
        assert_eq!(floor_to_decade(11), 10);
        assert_eq!(floor_to_decade(99), 10);
        assert_eq!(floor_to_decade(100), 100);
        assert_eq!(floor_to_decade(715_896_331), 100_000_000);
        assert_eq!(floor_to_decade(1_000_000_000), 1_000_000_000);
        assert_eq!(floor_to_decade(999_999_999), 100_000_000);
    }

    /// Every decade boundary, mechanically: the boundary itself floors
    /// to itself, and one below floors to the previous decade.
    #[test]
    fn every_decade_boundary_is_exact() {
        let mut decade = 1u64;
        let mut previous = 1u64;
        while decade <= u64::MAX / 10 {
            assert_eq!(floor_to_decade(decade), decade, "at {decade}");
            if decade > 1 {
                assert_eq!(floor_to_decade(decade - 1), previous, "just below {decade}");
            }
            previous = decade;
            decade *= 10;
        }
    }

    /// The floor never overflows or loops on the largest inputs, where
    /// the next decade is not representable.
    #[test]
    fn the_floor_terminates_at_the_top_of_the_range() {
        assert_eq!(floor_to_decade(u64::MAX), 10_000_000_000_000_000_000);
        assert_eq!(floor_to_decade(10_000_000_000_000_000_000), 10_000_000_000_000_000_000);
    }

    /// **A plan never projects past its cap.** This is the property the
    /// whole feature exists for, so it is checked across a wide sweep
    /// of record sizes rather than at one convenient value.
    #[test]
    fn a_plan_never_projects_past_its_cap() {
        let cap = DEFAULT_MAX_SHARD_BYTES;
        for record in [1u64, 2, 3, 5, 129, 1_540, 4_096, 6_004, 1_000_000, 999_999_999] {
            let p = plan_fixed(cap, record).unwrap_or_else(|| panic!("record {record}"));
            assert!(
                p.projected_bytes() <= cap,
                "record {record}: {} projected over a {cap} cap",
                p.projected_bytes()
            );
            assert_eq!(p.stride, floor_to_decade(cap / record));
            assert!(p.utilization() > 0.0 && p.utilization() <= 1.0);
        }
    }

    /// A 384-dimension f32 facet — the shape that motivated the cap.
    #[test]
    fn a_384_dimension_float_facet_plans_a_readable_stride() {
        let record = xvec_record_bytes(384, 4);
        assert_eq!(record, 1540);
        let p = plan_fixed(DEFAULT_MAX_SHARD_BYTES, record).unwrap();

        // 1 TB / 1540 is ~649 million; the decade below is 100 million.
        assert_eq!(p.stride, 100_000_000);
        assert_eq!(p.projected_bytes(), 154_000_000_000);
        assert!(!p.basis.is_estimate());
        // Which is what a readable stride costs: about 15% of the cap.
        assert!((p.utilization() - 0.154).abs() < 0.001);
    }

    /// A cap that cannot hold ten records is refused rather than
    /// answered with a stride of one, which would name a file per
    /// record.
    #[test]
    fn a_cap_too_small_for_a_run_is_refused() {
        assert!(plan_fixed(1_000, 1_000).is_none(), "one record per file");
        assert!(plan_fixed(1_000, 101).is_none(), "nine records per file");
        assert!(plan_fixed(1_000, 100).is_some(), "ten fits");
        assert!(plan_fixed(0, 100).is_none());
        assert!(plan_fixed(1_000, 0).is_none());
    }

    /// A sampled basis carries the 2× margin, so the projection sits
    /// at half the cap and stays under it even if the true mean is
    /// double what the sample saw.
    #[test]
    fn a_sampled_basis_budgets_twice_the_mean() {
        let basis = sample_of(&[100, 100, 100, 100]).unwrap();
        assert_eq!(basis, RecordSize::Sampled { mean: 100, max: 100, sampled: 4 });
        assert_eq!(basis.budget(), 200, "the margin is on the mean");
        assert!(basis.is_estimate());

        let p = plan(1_000_000_000, basis).unwrap();
        assert_eq!(p.stride, floor_to_decade(1_000_000_000 / 200));
        // Were the true mean twice the sample's, the shard would land
        // exactly at the cap rather than over it.
        assert!(p.stride * 100 * 2 <= 1_000_000_000);
    }

    /// A record larger than twice the mean still fits in a shard: the
    /// budget never drops below the largest record seen.
    #[test]
    fn the_budget_holds_the_largest_record_seen() {
        // A long tail: mean 30, but one record of 900.
        let mut lengths = vec![10u64; 99];
        lengths.push(900);
        let basis = sample_of(&lengths).unwrap();
        let RecordSize::Sampled { mean, max, sampled } = basis else {
            panic!("sampled");
        };
        assert_eq!((mean, max, sampled), (19, 900, 100));
        assert_eq!(basis.budget(), 900, "not 38 — one record has to fit");
    }

    /// The mean rounds up. A fraction of a byte per record is real
    /// bytes across a hundred million of them, and rounding down would
    /// put the projection over the cap.
    #[test]
    fn the_sampled_mean_rounds_up() {
        let basis = sample_of(&[10, 11]).unwrap();
        assert!(matches!(basis, RecordSize::Sampled { mean: 11, .. }), "{basis:?}");
    }

    /// An empty sample measures nothing and says so, rather than
    /// reporting a mean of zero and an unbounded stride.
    #[test]
    fn an_empty_sample_is_not_a_basis() {
        assert!(sample_of(&[]).is_none());
    }

    /// The plan answers how many shards a facet needs, with a short
    /// last shard rather than a dropped remainder.
    #[test]
    fn shard_counts_round_up_for_the_last_shard() {
        let p = plan_fixed(DEFAULT_MAX_SHARD_BYTES, 1540).unwrap();
        assert_eq!(p.stride, 100_000_000);
        assert_eq!(p.shards_for(0), 0);
        assert_eq!(p.shards_for(1), 1);
        assert_eq!(p.shards_for(100_000_000), 1);
        assert_eq!(p.shards_for(100_000_001), 2, "the remainder is a shard");
        assert_eq!(p.shards_for(1_000_000_000), 10);
    }

    /// The plan says what it is in one line, distinguishing a measured
    /// basis from a computed one.
    #[test]
    fn a_plan_reports_its_basis() {
        let fixed = plan_fixed(DEFAULT_MAX_SHARD_BYTES, 1540).unwrap().to_string();
        assert!(fixed.contains("100000000 records per shard"), "{fixed}");
        assert!(fixed.contains("1540 bytes per record"), "{fixed}");

        let sampled = plan(DEFAULT_MAX_SHARD_BYTES, sample_of(&[500, 700]).unwrap())
            .unwrap()
            .to_string();
        assert!(sampled.contains('~'), "an estimate reads as one: {sampled}");
        assert!(sampled.contains("2 sampled"), "{sampled}");
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::*;

    fn spread(total: u64, target: u64, lengths: &[u64]) -> Vec<u64> {
        let mut seen = Vec::new();
        let _ = sample_spread::<(), _>(total, target, |o| {
            seen.push(o);
            Ok(lengths[o as usize])
        });
        seen
    }

    /// **The sample spans the facet**, first ordinal to last, rather
    /// than taking a prefix — which a facet sorted by any key would
    /// make unrepresentative.
    #[test]
    fn a_sample_spans_the_facet() {
        let lengths = vec![1u64; 1000];
        let seen = spread(1000, 5, &lengths);
        assert_eq!(seen, [0, 249, 499, 749, 999]);
        assert_eq!(*seen.last().unwrap(), 999, "the last record is measured");
    }

    /// Ordinals ascend, so a reader that walks forward stays forward.
    #[test]
    fn sampled_ordinals_ascend() {
        let lengths = vec![1u64; 10_000];
        let seen = spread(10_000, 97, &lengths);
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");
        assert_eq!(seen.len(), 97);
    }

    /// A facet smaller than the target is measured whole, once each.
    #[test]
    fn a_small_facet_is_measured_entirely() {
        let lengths = vec![7u64; 4];
        let seen = spread(4, 1000, &lengths);
        assert_eq!(seen, [0, 1, 2, 3]);
    }

    /// A prefix sample would be fooled by ordering; a spread one is
    /// not. Here the first half is tiny and the second half large.
    #[test]
    fn a_spread_sample_is_not_fooled_by_ordering() {
        let mut lengths = vec![10u64; 500];
        lengths.extend(std::iter::repeat_n(1_010u64, 500));

        let basis = sample_spread::<(), _>(1000, 100, |o| Ok(lengths[o as usize]))
            .unwrap()
            .unwrap();
        let RecordSize::Sampled { mean, .. } = basis else { panic!("sampled") };
        // The true mean is 510. A prefix of 100 would have said 10.
        assert!((500..=520).contains(&mean), "spread sample found {mean}");
    }

    /// An empty facet yields no basis, and the length function is
    /// never called for an ordinal that does not exist.
    #[test]
    fn an_empty_facet_is_never_probed() {
        let mut called = false;
        let got = sample_spread::<(), _>(0, 100, |_| {
            called = true;
            Ok(1)
        })
        .unwrap();
        assert!(got.is_none());
        assert!(!called, "no ordinal exists to measure");
    }

    /// A read failure surfaces rather than being smoothed into a
    /// smaller sample — a partial measurement would silently change
    /// the mean.
    #[test]
    fn a_failed_measurement_is_not_swallowed() {
        let got = sample_spread::<&str, _>(100, 10, |o| {
            if o > 20 { Err("read failed") } else { Ok(5) }
        });
        assert_eq!(got, Err("read failed"));
    }

    /// End to end: a variable facet with a known mean plans a stride
    /// that keeps the projected file under the cap.
    #[test]
    fn a_sampled_facet_plans_under_its_cap() {
        // 200-byte records, mean exactly 200.
        let basis = sample_spread::<(), _>(1_000_000, 1_000, |_| Ok(200)).unwrap().unwrap();
        let cap = DEFAULT_MAX_SHARD_BYTES;
        let p = plan(cap, basis).unwrap();

        assert_eq!(p.record_bytes, 400, "200 with the 2x margin");
        assert_eq!(p.stride, 1_000_000_000);
        assert!(p.projected_bytes() <= cap);
        // Even if the real mean were double the sample's, it fits.
        assert!(p.stride * 200 * 2 <= cap);
    }
}

#[cfg(test)]
mod sharding_tests {
    use super::*;

    /// **An explicit stride is used verbatim.** An operator who names
    /// one has a reason, and second-guessing it against a cap would
    /// make the flag mean something other than what it says.
    #[test]
    fn an_explicit_stride_is_not_second_guessed() {
        let s = Sharding::Stride(7);
        assert_eq!(s.stride_for_fixed(1_000_000_000), Some(7));
        assert_eq!(s.stride_for_sampled(RecordSize::Fixed(1_000_000_000)), Some(7));
        assert_eq!(s.max_bytes(), None, "a stride is not a cap");
        assert!(s.is_requested());
    }

    /// A cap resolves to whatever decade stride fits under it.
    #[test]
    fn a_cap_resolves_to_a_decade_stride() {
        let s = Sharding::MaxBytes(DEFAULT_MAX_SHARD_BYTES);
        assert_eq!(s.stride_for_fixed(1540), Some(100_000_000));
        assert_eq!(s.max_bytes(), Some(DEFAULT_MAX_SHARD_BYTES));

        // And through a measurement, with the margin applied.
        let basis = sample_of(&[200, 200]).unwrap();
        assert_eq!(s.stride_for_sampled(basis), Some(1_000_000_000));
    }

    /// The default asks for nothing, so a caller that passes it
    /// behaves exactly as before sharding existed.
    #[test]
    fn whole_is_the_default_and_asks_for_nothing() {
        let s = Sharding::default();
        assert_eq!(s, Sharding::Whole);
        assert!(!s.is_requested());
        assert_eq!(s.stride_for_fixed(1540), None);
        assert_eq!(s.stride_for_sampled(RecordSize::Fixed(1540)), None);
    }

    /// A cap too small for a run resolves to nothing, so a caller can
    /// refuse rather than emit a file per handful of records.
    #[test]
    fn a_cap_too_small_resolves_to_nothing() {
        assert_eq!(Sharding::MaxBytes(1_000).stride_for_fixed(1_000), None);
    }

    /// **Both flags is refused, not resolved.** A precedence rule
    /// between "this many records" and "this many bytes" is one nobody
    /// would remember, and getting it wrong is silent.
    #[test]
    fn naming_both_surfaces_is_refused() {
        let err = Sharding::from_flags(Some("1M"), Some("1TB")).unwrap_err();
        assert!(err.contains("--shard-stride"), "{err}");
        assert!(err.contains("--max-shard-bytes"), "{err}");
    }

    /// Each surface parses the suffixes the rest of this crate reads,
    /// and neither is `Whole`.
    #[test]
    fn each_surface_parses_its_own_units() {
        assert_eq!(Sharding::from_flags(Some("1M"), None), Ok(Sharding::Stride(1_000_000)));
        assert_eq!(Sharding::from_flags(Some("500"), None), Ok(Sharding::Stride(500)));
        assert_eq!(
            Sharding::from_flags(None, Some("1TB")),
            Ok(Sharding::MaxBytes(1_000_000_000_000))
        );
        assert_eq!(
            Sharding::from_flags(None, Some("2TiB")),
            Ok(Sharding::MaxBytes(2 << 40))
        );
        assert_eq!(Sharding::from_flags(None, None), Ok(Sharding::Whole));
    }

    /// A bad value names the flag it came from, so an operator who
    /// passed two knows which to fix.
    #[test]
    fn a_bad_value_names_its_own_flag() {
        let err = Sharding::from_flags(Some("lots"), None).unwrap_err();
        assert!(err.starts_with("--shard-stride:"), "{err}");
        let err = Sharding::from_flags(None, Some("huge")).unwrap_err();
        assert!(err.starts_with("--max-shard-bytes:"), "{err}");
    }
}
