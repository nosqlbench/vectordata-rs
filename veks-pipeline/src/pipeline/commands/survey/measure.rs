// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Measure trait, kind discriminator, observation context, and the
//! serialized report shape.
//!
//! A [`Measure`] consumes a stream of `(MValue, MeasureCtx)`
//! observations during Pass 2 and finalizes into a
//! [`MeasureReport`]. The report's outer JSON layout is an open map
//! keyed by [`MeasureKind`]'s discriminant string — adding new
//! measures does not require bumping the survey schema version
//! (§13.8).
//!
//! Pass 1 does **not** instantiate `Measure` implementations; it
//! uses [`super::ExplorationProbe`] instead. The `Measure` trait is
//! the Pass 2 surface.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::types::SemanticType;

/// Identifier for a concrete measure implementation. The string form
/// (via [`as_str`](Self::as_str)) is what appears as the key in the
/// survey JSON's `measures` map and is the stable identifier used
/// in per-field measure-override configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasureKind {
    // ── universal ─────────────────────────────────────────────
    /// Present / absent / null counts.
    Presence,
    /// Tracks whether Pass 2 saw a `MValue` tag absent from Pass 1.
    TypeStability,
    /// Bounded reservoir of representative values for the report.
    ReservoirSample,

    // ── numeric ───────────────────────────────────────────────
    /// Moments m1..m4 → mean, variance, stddev, skewness, kurtosis.
    ExactMoments,
    /// Exact min and max.
    ExactExtrema,
    /// KLL quantile sketch over numeric observations.
    QuantileSketch,
    /// Bit-width / density / popcount classification for integer fields.
    BitWidth,
    /// Equi-width histogram derived from observed min/max.
    HistogramFromQuantiles,
    /// Mann-Kendall τ trend statistic over observation order.
    Monotonicity,
    /// Fraction of integer-valued observations (for Float fields).
    DiscreteIndicator,

    // ── cardinality / frequency ───────────────────────────────
    /// HyperLogLog cardinality estimate.
    HyperLogLog,
    /// Misra-Gries top-K heavy hitters.
    HeavyHitters,
    /// Exact frequency table for low-cardinality fields.
    ExactFrequencyTable,

    // ── textual ───────────────────────────────────────────────
    /// Exact moments over text length (bytes + chars).
    ExactLengthMoments,
    /// KLL quantile sketch over text length.
    LengthQuantiles,
    /// Per-character-class fraction mix (alpha/digit/punct/whitespace/other).
    CharClassMix,
    /// Top-K regex-skeleton patterns (e.g. `9+@A+.A+` for emails).
    PatternSkeleton,
    /// Top-K character-trigram heavy hitters — drives calibrated
    /// `MATCHES` predicate synthesis.
    TrigramHeavyHitters,
    /// Top-K per-label heavy hitters for labelset-shaped fields.
    LabelsetHeavyHitters,

    // ── temporal ─────────────────────────────────────────────
    /// Min / max of Millis / Nanos / Int-as-epoch values, normalized
    /// to seconds since 1970.
    TemporalRange,
    /// Fraction of integer observations that fall in the plausible
    /// epoch-seconds / epoch-millis ranges.
    EpochPlausibility,

    // ── bytes ────────────────────────────────────────────────
    /// Shannon entropy of byte-value distribution.
    ByteEntropy,

    // ── unstable / opaque ─────────────────────────────────────
    /// MValue tag-name → fraction histogram.
    WireEncodingHistogram,
    /// Min / max length in bytes and chars without quantiles.
    ByteOrCharLengthRange,
    /// Per-semantic-probe match rates (diagnostic for Unstable fields).
    ProbeAttempt,
}

impl MeasureKind {
    /// String form used as the JSON map key.
    pub fn as_str(self) -> &'static str {
        match self {
            MeasureKind::Presence => "Presence",
            MeasureKind::TypeStability => "TypeStability",
            MeasureKind::ReservoirSample => "ReservoirSample",
            MeasureKind::ExactMoments => "ExactMoments",
            MeasureKind::ExactExtrema => "ExactExtrema",
            MeasureKind::QuantileSketch => "QuantileSketch",
            MeasureKind::BitWidth => "BitWidth",
            MeasureKind::HistogramFromQuantiles => "HistogramFromQuantiles",
            MeasureKind::Monotonicity => "Monotonicity",
            MeasureKind::DiscreteIndicator => "DiscreteIndicator",
            MeasureKind::HyperLogLog => "HyperLogLog",
            MeasureKind::HeavyHitters => "HeavyHitters",
            MeasureKind::ExactFrequencyTable => "ExactFrequencyTable",
            MeasureKind::ExactLengthMoments => "ExactLengthMoments",
            MeasureKind::LengthQuantiles => "LengthQuantiles",
            MeasureKind::CharClassMix => "CharClassMix",
            MeasureKind::PatternSkeleton => "PatternSkeleton",
            MeasureKind::TrigramHeavyHitters => "TrigramHeavyHitters",
            MeasureKind::LabelsetHeavyHitters => "LabelsetHeavyHitters",
            MeasureKind::TemporalRange => "TemporalRange",
            MeasureKind::EpochPlausibility => "EpochPlausibility",
            MeasureKind::ByteEntropy => "ByteEntropy",
            MeasureKind::WireEncodingHistogram => "WireEncodingHistogram",
            MeasureKind::ByteOrCharLengthRange => "ByteOrCharLengthRange",
            MeasureKind::ProbeAttempt => "ProbeAttemptReport",
        }
    }
}

/// Context passed to a measure on every observation.
///
/// Carries the current record index (for position-correlated
/// measures) and the field's verdict from Pass 1 (so a measure
/// written for `Numeric` can rely on the parser having accepted the
/// value).
#[derive(Debug, Clone)]
pub struct MeasureCtx<'a> {
    /// Zero-based record index in the order observations are issued
    /// to the orchestrator.
    pub record_index: u64,
    /// The field's `SemanticType` verdict from Pass 1, or `None`
    /// when the measure runs before Pass 1's verdict is committed
    /// (e.g. `ReservoirSample` during exploration). Measures that
    /// need the verdict but find it `None` should treat the input
    /// conservatively.
    pub semantic_type: Option<&'a SemanticType>,
}

/// Trait implemented by every Pass 2 measure.
///
/// Implementations are owned by the orchestrator's per-field state
/// and are dispatched dynamically via `Box<dyn Measure>` so that the
/// per-field measure list can be heterogeneous and chosen at
/// template-synthesis time.
pub trait Measure: Send {
    /// Observe one value for the field this measure is bound to.
    fn observe(&mut self, value: &MValue, ctx: &MeasureCtx);
    /// Notify the measure that the record under inspection does not
    /// contain this field at all (a "structural null", distinct from
    /// an `MValue::Null`). The default is a no-op; only measures
    /// that need to distinguish absent-from-record from present-but-null
    /// override it (`PresenceMeasure`).
    fn observe_missing(&mut self, _ctx: &MeasureCtx) {}
    /// Consume the measure and emit its report.
    fn finalize(self: Box<Self>) -> MeasureReport;
    /// Measure kind, used as the JSON map key.
    fn kind(&self) -> MeasureKind;
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// Output of a finalized [`Measure`]. Serialized as one of the
/// variants below; the survey's `fields[name].measures` map keys
/// each report by the corresponding [`MeasureKind`]'s `as_str()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MeasureReport {
    /// `MeasureKind::Presence`.
    Presence(PresenceReport),
    /// `MeasureKind::TypeStability`.
    TypeStability(TypeStabilityReport),
    /// `MeasureKind::ReservoirSample`.
    ReservoirSample(ReservoirSampleReport),
    /// `MeasureKind::ExactMoments`.
    ExactMoments(ExactMomentsReport),
    /// `MeasureKind::ExactExtrema`.
    ExactExtrema(ExactExtremaReport),
    /// `MeasureKind::QuantileSketch`.
    QuantileSketch(super::measures::numeric::QuantileSketchReport),
    /// `MeasureKind::BitWidth`.
    BitWidth(super::measures::numeric::BitWidthReport),
    /// `MeasureKind::HistogramFromQuantiles`.
    HistogramFromQuantiles(super::measures::numeric::HistogramFromQuantilesReport),
    /// `MeasureKind::Monotonicity`.
    Monotonicity(super::measures::numeric::MonotonicityReport),
    /// `MeasureKind::DiscreteIndicator`.
    DiscreteIndicator(super::measures::numeric::DiscreteIndicatorReport),
    /// `MeasureKind::HyperLogLog`.
    HyperLogLog(super::measures::cardinality::HyperLogLogReport),
    /// `MeasureKind::HeavyHitters`.
    HeavyHitters(super::measures::cardinality::HeavyHittersReport),
    /// `MeasureKind::ExactFrequencyTable`.
    ExactFrequencyTable(super::measures::cardinality::ExactFrequencyTableReport),
    /// `MeasureKind::ExactLengthMoments`.
    ExactLengthMoments(super::measures::textual::ExactLengthMomentsReport),
    /// `MeasureKind::LengthQuantiles`.
    LengthQuantiles(super::measures::textual::LengthQuantilesReport),
    /// `MeasureKind::CharClassMix`.
    CharClassMix(super::measures::textual::CharClassMixReport),
    /// `MeasureKind::PatternSkeleton`.
    PatternSkeleton(super::measures::textual::PatternSkeletonReport),
    /// `MeasureKind::TrigramHeavyHitters`.
    TrigramHeavyHitters(super::measures::trigram::TrigramHeavyHittersReport),
    /// `MeasureKind::LabelsetHeavyHitters`.
    LabelsetHeavyHitters(super::measures::labelset::LabelsetHeavyHittersReport),
    /// `MeasureKind::TemporalRange`.
    TemporalRange(super::measures::temporal::TemporalRangeReport),
    /// `MeasureKind::EpochPlausibility`.
    EpochPlausibility(super::measures::temporal::EpochPlausibilityReport),
    /// `MeasureKind::ByteEntropy`.
    ByteEntropy(super::measures::bytes::ByteEntropyReport),
    /// `MeasureKind::WireEncodingHistogram`.
    WireEncodingHistogram(super::measures::opaque::WireEncodingHistogramReport),
    /// `MeasureKind::ByteOrCharLengthRange`.
    ByteOrCharLengthRange(super::measures::opaque::ByteOrCharLengthRangeReport),
    /// `MeasureKind::ProbeAttempt`.
    ProbeAttempt(super::measures::opaque::ProbeAttemptReport),
}

/// Per-field presence summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresenceReport {
    /// Number of records that contained this field (any value, incl. null).
    pub present: u64,
    /// Number of records that contained this field as `MValue::Null`.
    pub null_count: u64,
    /// Number of records that did NOT contain this field at all.
    pub absent_in_record: u64,
}

/// Pass-2 observations of `MValue` tags absent from Pass 1's
/// histogram. Surfaces stream-shift / sampling-bias warnings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeStabilityReport {
    /// Count of Pass 2 observations whose tag was not seen in Pass 1.
    pub surprise_count: u64,
    /// Distinct tags observed in Pass 2 but missing from Pass 1.
    pub surprise_tags: Vec<String>,
}

/// Bounded reservoir of representative values, serialized as JSON.
///
/// Each value is rendered with its `Display`-style canonical form
/// (numbers as numbers, strings as strings, bytes as hex) so the
/// JSON remains human-readable without lossy round-trips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReservoirSampleReport {
    /// Sample of values.
    pub items: Vec<serde_json::Value>,
    /// Total observations the reservoir was drawn from.
    pub observed: u64,
}

/// Numeric moments (m1..m4) and derived stats.
///
/// `mean` is `None` only when no observations were seen.
/// `stddev` / `skewness` / `kurtosis` require `count >= 2` and are
/// `None` for shorter streams. Constant-stream stddev is reported
/// as `Some(0.0)`, not `None` — the value is defined and zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactMomentsReport {
    /// Number of observations contributing to the moments.
    pub count: u64,
    /// Arithmetic mean.
    pub mean: Option<f64>,
    /// Sample standard deviation (Bessel-corrected: `1/(n-1)`).
    pub stddev: Option<f64>,
    /// Skewness (third standardized moment).
    pub skewness: Option<f64>,
    /// Excess kurtosis (fourth standardized moment minus 3).
    pub kurtosis: Option<f64>,
}

/// Exact min / max. `None` when no numeric observations were seen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactExtremaReport {
    /// Minimum observed.
    pub min: Option<f64>,
    /// Maximum observed.
    pub max: Option<f64>,
}

impl MeasureReport {
    /// The [`MeasureKind`] this report corresponds to.
    pub fn kind(&self) -> MeasureKind {
        match self {
            MeasureReport::Presence(_) => MeasureKind::Presence,
            MeasureReport::TypeStability(_) => MeasureKind::TypeStability,
            MeasureReport::ReservoirSample(_) => MeasureKind::ReservoirSample,
            MeasureReport::ExactMoments(_) => MeasureKind::ExactMoments,
            MeasureReport::ExactExtrema(_) => MeasureKind::ExactExtrema,
            MeasureReport::QuantileSketch(_) => MeasureKind::QuantileSketch,
            MeasureReport::BitWidth(_) => MeasureKind::BitWidth,
            MeasureReport::HistogramFromQuantiles(_) => MeasureKind::HistogramFromQuantiles,
            MeasureReport::Monotonicity(_) => MeasureKind::Monotonicity,
            MeasureReport::DiscreteIndicator(_) => MeasureKind::DiscreteIndicator,
            MeasureReport::HyperLogLog(_) => MeasureKind::HyperLogLog,
            MeasureReport::HeavyHitters(_) => MeasureKind::HeavyHitters,
            MeasureReport::ExactFrequencyTable(_) => MeasureKind::ExactFrequencyTable,
            MeasureReport::ExactLengthMoments(_) => MeasureKind::ExactLengthMoments,
            MeasureReport::LengthQuantiles(_) => MeasureKind::LengthQuantiles,
            MeasureReport::CharClassMix(_) => MeasureKind::CharClassMix,
            MeasureReport::PatternSkeleton(_) => MeasureKind::PatternSkeleton,
            MeasureReport::TrigramHeavyHitters(_) => MeasureKind::TrigramHeavyHitters,
            MeasureReport::LabelsetHeavyHitters(_) => MeasureKind::LabelsetHeavyHitters,
            MeasureReport::TemporalRange(_) => MeasureKind::TemporalRange,
            MeasureReport::EpochPlausibility(_) => MeasureKind::EpochPlausibility,
            MeasureReport::ByteEntropy(_) => MeasureKind::ByteEntropy,
            MeasureReport::WireEncodingHistogram(_) => MeasureKind::WireEncodingHistogram,
            MeasureReport::ByteOrCharLengthRange(_) => MeasureKind::ByteOrCharLengthRange,
            MeasureReport::ProbeAttempt(_) => MeasureKind::ProbeAttempt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_as_str_is_pascal_case() {
        // The JSON map keys in §13.8's example use PascalCase
        // ("ExactMoments", "ExactExtrema", "ReservoirSample", …).
        // The string form must match exactly.
        assert_eq!(MeasureKind::Presence.as_str(), "Presence");
        assert_eq!(MeasureKind::TypeStability.as_str(), "TypeStability");
        assert_eq!(MeasureKind::ReservoirSample.as_str(), "ReservoirSample");
        assert_eq!(MeasureKind::ExactMoments.as_str(), "ExactMoments");
        assert_eq!(MeasureKind::ExactExtrema.as_str(), "ExactExtrema");
    }

    #[test]
    fn kind_strings_match_report_kinds() {
        let r = MeasureReport::ExactExtrema(ExactExtremaReport { min: Some(0.0), max: Some(1.0) });
        assert_eq!(r.kind(), MeasureKind::ExactExtrema);
    }

    #[test]
    fn report_roundtrip_through_json() {
        let r = MeasureReport::ExactMoments(ExactMomentsReport {
            count: 100,
            mean: Some(0.5),
            stddev: Some(0.1),
            skewness: Some(0.0),
            kurtosis: Some(-1.2),
        });
        let s = serde_json::to_string(&r).unwrap();
        let back: MeasureReport = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
