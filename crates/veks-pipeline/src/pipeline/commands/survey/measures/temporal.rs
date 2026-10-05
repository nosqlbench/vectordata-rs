// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Temporal measures.
//!
//! - [`TemporalRangeMeasure`] — min/max over Millis / Nanos values
//!   plus any Int values that look like epoch timestamps.
//! - [`EpochPlausibilityMeasure`] — for Int fields, the fraction of
//!   observations that fall in the plausible-epoch ranges for
//!   seconds-since-1970 and millis-since-1970, useful for flagging
//!   "this field is stored as Int but reads as a timestamp".
//!
//! Both measures normalize to **seconds-since-1970** (`f64`) for
//! reporting so an operator sees a single comparable scale.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport,
};

// Year 2000 .. 2100 in seconds since 1970, used by the plausibility
// detector. Anything in this window is "plausibly a Unix timestamp".
const EPOCH_SEC_MIN: i64 = 946_684_800; // 2000-01-01
const EPOCH_SEC_MAX: i64 = 4_102_444_800; // 2100-01-01
const EPOCH_MS_MIN: i64 = EPOCH_SEC_MIN * 1_000;
const EPOCH_MS_MAX: i64 = EPOCH_SEC_MAX * 1_000;

// ---------------------------------------------------------------------------
// TemporalRangeMeasure
// ---------------------------------------------------------------------------

pub struct TemporalRangeMeasure {
    count: u64,
    min_secs: f64,
    max_secs: f64,
    granularity_hint: GranularityHint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GranularityHint {
    Unknown,
    Seconds,
    Millis,
    Nanos,
}

impl Default for TemporalRangeMeasure {
    fn default() -> Self { Self::new() }
}

impl TemporalRangeMeasure {
    pub fn new() -> Self {
        TemporalRangeMeasure {
            count: 0,
            min_secs: f64::INFINITY,
            max_secs: f64::NEG_INFINITY,
            granularity_hint: GranularityHint::Unknown,
        }
    }

    fn record(&mut self, secs: f64, granularity: GranularityHint) {
        if !secs.is_finite() { return; }
        self.count += 1;
        if secs < self.min_secs { self.min_secs = secs; }
        if secs > self.max_secs { self.max_secs = secs; }
        // First non-Unknown wins. If subsequent observations
        // disagree we coarsen toward the broader granularity
        // (Seconds < Millis < Nanos).
        self.granularity_hint = match (self.granularity_hint, granularity) {
            (GranularityHint::Unknown, g) => g,
            (a, GranularityHint::Unknown) => a,
            (a, b) if a == b => a,
            // Mixed → keep the finer-grained one.
            _ => GranularityHint::Nanos,
        };
    }
}

impl Measure for TemporalRangeMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        match value {
            MValue::Millis(ms) => self.record(*ms as f64 / 1_000.0, GranularityHint::Millis),
            MValue::Nanos { epoch_seconds, nano_adjust } => {
                let s = *epoch_seconds as f64 + (*nano_adjust as f64) * 1e-9;
                self.record(s, GranularityHint::Nanos);
            }
            MValue::Int(v) => {
                // Plausibility-promoted integer: if it looks like a
                // Unix-seconds or Unix-millis timestamp, record it.
                let i = *v;
                if (EPOCH_SEC_MIN..EPOCH_SEC_MAX).contains(&i) {
                    self.record(i as f64, GranularityHint::Seconds);
                } else if (EPOCH_MS_MIN..EPOCH_MS_MAX).contains(&i) {
                    self.record(i as f64 / 1_000.0, GranularityHint::Millis);
                }
            }
            _ => {}
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let (min, max) = if self.count == 0 {
            (None, None)
        } else {
            (Some(self.min_secs), Some(self.max_secs))
        };
        let granularity = match self.granularity_hint {
            GranularityHint::Unknown => "unknown",
            GranularityHint::Seconds => "seconds",
            GranularityHint::Millis => "millis",
            GranularityHint::Nanos => "nanos",
        };
        MeasureReport::TemporalRange(TemporalRangeReport {
            count: self.count,
            min_epoch_seconds: min,
            max_epoch_seconds: max,
            granularity: granularity.into(),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::TemporalRange
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalRangeReport {
    pub count: u64,
    pub min_epoch_seconds: Option<f64>,
    pub max_epoch_seconds: Option<f64>,
    /// Inferred granularity: `seconds` / `millis` / `nanos` /
    /// `unknown`.
    pub granularity: String,
}

// ---------------------------------------------------------------------------
// EpochPlausibilityMeasure
// ---------------------------------------------------------------------------

pub struct EpochPlausibilityMeasure {
    n: u64,
    looks_like_seconds: u64,
    looks_like_millis: u64,
    looks_like_neither: u64,
}

impl Default for EpochPlausibilityMeasure {
    fn default() -> Self { Self::new() }
}

impl EpochPlausibilityMeasure {
    pub fn new() -> Self {
        EpochPlausibilityMeasure {
            n: 0,
            looks_like_seconds: 0,
            looks_like_millis: 0,
            looks_like_neither: 0,
        }
    }
}

impl Measure for EpochPlausibilityMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let i = match value {
            MValue::Int(v) => *v,
            MValue::Int32(v) => *v as i64,
            MValue::Millis(v) => *v,
            _ => return,
        };
        self.n += 1;
        if (EPOCH_SEC_MIN..EPOCH_SEC_MAX).contains(&i) {
            self.looks_like_seconds += 1;
        } else if (EPOCH_MS_MIN..EPOCH_MS_MAX).contains(&i) {
            self.looks_like_millis += 1;
        } else {
            self.looks_like_neither += 1;
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let n = self.n.max(1) as f64;
        MeasureReport::EpochPlausibility(EpochPlausibilityReport {
            count: self.n,
            seconds_match_rate: self.looks_like_seconds as f64 / n,
            millis_match_rate: self.looks_like_millis as f64 / n,
            neither_rate: self.looks_like_neither as f64 / n,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::EpochPlausibility
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochPlausibilityReport {
    pub count: u64,
    /// Fraction of integer observations in the seconds-since-1970
    /// plausible range (2000–2100).
    pub seconds_match_rate: f64,
    /// Fraction of integer observations in the milliseconds-since-1970
    /// plausible range (2000–2100).
    pub millis_match_rate: f64,
    /// Fraction in neither range.
    pub neither_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn temporal_range_from_millis() {
        let mut m = TemporalRangeMeasure::new();
        m.observe(&MValue::Millis(1_700_000_000_000), &ctx());
        m.observe(&MValue::Millis(1_750_000_000_000), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TemporalRange(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 2);
        assert!((r.min_epoch_seconds.unwrap() - 1_700_000_000.0).abs() < 1.0);
        assert!((r.max_epoch_seconds.unwrap() - 1_750_000_000.0).abs() < 1.0);
        assert_eq!(r.granularity, "millis");
    }

    #[test]
    fn temporal_range_picks_up_int_epoch_seconds() {
        let mut m = TemporalRangeMeasure::new();
        // A handful of plausible-seconds integers.
        for s in [1_700_000_000i64, 1_700_000_100, 1_700_000_200] {
            m.observe(&MValue::Int(s), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::TemporalRange(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 3);
        assert_eq!(r.granularity, "seconds");
    }

    #[test]
    fn temporal_range_ignores_non_temporal() {
        let mut m = TemporalRangeMeasure::new();
        m.observe(&MValue::Text("hi".into()), &ctx());
        m.observe(&MValue::Float(3.25), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TemporalRange(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 0);
    }

    #[test]
    fn epoch_plausibility_seconds_dominant() {
        let mut m = EpochPlausibilityMeasure::new();
        for s in 1_700_000_000i64..1_700_000_100 {
            m.observe(&MValue::Int(s), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::EpochPlausibility(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert!((r.seconds_match_rate - 1.0).abs() < 1e-9);
        assert_eq!(r.millis_match_rate, 0.0);
    }

    #[test]
    fn epoch_plausibility_mixed() {
        let mut m = EpochPlausibilityMeasure::new();
        m.observe(&MValue::Int(1_700_000_000), &ctx());   // seconds
        m.observe(&MValue::Int(1_700_000_000_000), &ctx()); // millis
        m.observe(&MValue::Int(42), &ctx());                // neither
        let r = match Box::new(m).finalize() {
            MeasureReport::EpochPlausibility(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert!((r.seconds_match_rate - 1.0 / 3.0).abs() < 1e-9);
        assert!((r.millis_match_rate - 1.0 / 3.0).abs() < 1e-9);
        assert!((r.neither_rate - 1.0 / 3.0).abs() < 1e-9);
    }
}
