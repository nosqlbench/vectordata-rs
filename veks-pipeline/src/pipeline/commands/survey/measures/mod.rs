// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Measure implementations for the metadata survey.
//!
//! Each submodule groups measures by what they operate on. The
//! universal measures run against every field's `SemanticType`; the
//! type-specific ones run only when Pass 1 committed to a matching
//! verdict.
//!
//! Subsequent build-plan steps (per `docs/sysref/13-metadata-survey.md`
//! §13.13) land additional measure families one type at a time.

pub mod bytes;
pub mod cardinality;
pub mod labelset;
pub mod numeric;
pub mod opaque;
pub mod temporal;
pub mod textual;
pub mod trigram;
pub mod universal;

pub use bytes::{ByteEntropyMeasure, ByteEntropyReport};
pub use opaque::{
    ByteOrCharLengthRangeMeasure, ByteOrCharLengthRangeReport, ProbeAttemptEntry,
    ProbeAttemptMeasure, ProbeAttemptReport, WireEncodingHistogramMeasure,
    WireEncodingHistogramReport,
};
pub use temporal::{
    EpochPlausibilityMeasure, EpochPlausibilityReport, TemporalRangeMeasure, TemporalRangeReport,
};

pub use cardinality::{
    ExactFrequencyTable, ExactFrequencyTableReport, HeavyHitterEntry, HeavyHittersMeasure,
    HeavyHittersReport, HyperLogLogMeasure, HyperLogLogReport,
};
pub use numeric::{
    BitWidthMeasure, BitWidthReport, DiscreteIndicatorMeasure, DiscreteIndicatorReport,
    ExactExtrema, ExactMoments, HistogramFromQuantilesMeasure, HistogramFromQuantilesReport,
    MonotonicityMeasure, MonotonicityReport, QuantileSketchMeasure, QuantileSketchReport,
};
pub use textual::{
    CharClassMix, CharClassMixReport, ExactLengthMoments, ExactLengthMomentsReport, LengthQuantiles,
    LengthQuantilesReport, LenSummary, PatternSkeletonMeasure, PatternSkeletonReport,
};
pub use labelset::{
    LabelsetEntry, LabelsetHeavyHittersMeasure, LabelsetHeavyHittersReport,
    DEFAULT_LABELSET_TOP_K,
};
pub use trigram::{
    TrigramEntry, TrigramHeavyHittersMeasure, TrigramHeavyHittersReport, DEFAULT_TRIGRAM_TOP_K,
};
pub use universal::{PresenceMeasure, ReservoirSample, TypeStabilityMeasure};
