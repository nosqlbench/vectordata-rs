// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Opaque-value measures for `SemanticType::Unstable` fields per
//! §13.5.8.
//!
//! These are the only measures (alongside `Presence`, `TypeStability`,
//! `ReservoirSample`) the orchestrator runs against a field whose
//! semantic verdict could not be committed. They surface the *shape*
//! of the unstable data — which wire encodings were seen, how big
//! the values are — without committing to any interpretation that
//! the survey deliberately declined to make.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport,
};

// ---------------------------------------------------------------------------
// WireEncodingHistogram
// ---------------------------------------------------------------------------

/// Per-MValue-tag observation histogram. The dominant signal on
/// Unstable fields: an operator scanning the report wants to know
/// *why* the field was unstable, and the answer is usually "mostly
/// Text but ~3% Int".
pub struct WireEncodingHistogramMeasure {
    counts: IndexMap<String, u64>,
    total: u64,
}

impl Default for WireEncodingHistogramMeasure {
    fn default() -> Self { Self::new() }
}

impl WireEncodingHistogramMeasure {
    pub fn new() -> Self {
        WireEncodingHistogramMeasure {
            counts: IndexMap::new(),
            total: 0,
        }
    }
}

impl Measure for WireEncodingHistogramMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let tag = format!("{:?}", value.tag());
        *self.counts.entry(tag).or_insert(0) += 1;
        self.total += 1;
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let mut fractions = IndexMap::with_capacity(self.counts.len());
        let total = self.total.max(1) as f64;
        let mut entries: Vec<(String, u64)> = self.counts.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, c) in entries {
            fractions.insert(k, c as f64 / total);
        }
        MeasureReport::WireEncodingHistogram(WireEncodingHistogramReport {
            total: self.total,
            fractions,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::WireEncodingHistogram
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireEncodingHistogramReport {
    pub total: u64,
    /// MValue tag name → fraction of observations. Sorted by
    /// descending fraction.
    pub fractions: IndexMap<String, f64>,
}

// ---------------------------------------------------------------------------
// ByteOrCharLengthRange
// ---------------------------------------------------------------------------

/// Encoding-agnostic length range: bytes for binary, characters for
/// textual, "n/a" otherwise. Just min/max/count — no quantiles or
/// distribution shape, since committing to those would require the
/// semantic interpretation Unstable fields lack.
pub struct ByteOrCharLengthRangeMeasure {
    bytes: Option<(u64, u64)>,
    chars: Option<(u64, u64)>,
    byte_count: u64,
    char_count: u64,
}

impl Default for ByteOrCharLengthRangeMeasure {
    fn default() -> Self { Self::new() }
}

impl ByteOrCharLengthRangeMeasure {
    pub fn new() -> Self {
        ByteOrCharLengthRangeMeasure {
            bytes: None,
            chars: None,
            byte_count: 0,
            char_count: 0,
        }
    }
}

impl Measure for ByteOrCharLengthRangeMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        match value {
            MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s)
            | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => {
                let cs = s.chars().count() as u64;
                self.char_count += 1;
                self.chars = Some(match self.chars {
                    None => (cs, cs),
                    Some((mn, mx)) => (mn.min(cs), mx.max(cs)),
                });
                // String byte length is also useful — count it under
                // the bytes axis too, since the byte form is what
                // hits disk.
                let bs = s.len() as u64;
                self.byte_count += 1;
                self.bytes = Some(match self.bytes {
                    None => (bs, bs),
                    Some((mn, mx)) => (mn.min(bs), mx.max(bs)),
                });
            }
            MValue::Bytes(b) => {
                let bs = b.len() as u64;
                self.byte_count += 1;
                self.bytes = Some(match self.bytes {
                    None => (bs, bs),
                    Some((mn, mx)) => (mn.min(bs), mx.max(bs)),
                });
            }
            _ => {}
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        MeasureReport::ByteOrCharLengthRange(ByteOrCharLengthRangeReport {
            bytes_count: self.byte_count,
            bytes_min: self.bytes.map(|(mn, _)| mn),
            bytes_max: self.bytes.map(|(_, mx)| mx),
            chars_count: self.char_count,
            chars_min: self.chars.map(|(mn, _)| mn),
            chars_max: self.chars.map(|(_, mx)| mx),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ByteOrCharLengthRange
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ByteOrCharLengthRangeReport {
    pub bytes_count: u64,
    pub bytes_min: Option<u64>,
    pub bytes_max: Option<u64>,
    pub chars_count: u64,
    pub chars_min: Option<u64>,
    pub chars_max: Option<u64>,
}

// ---------------------------------------------------------------------------
// ProbeAttemptReport
// ---------------------------------------------------------------------------

/// Per-semantic-probe match rates for fields where no probe cleared
/// the confidence threshold. Diagnostic — surfaces "why didn't this
/// field commit to a SemanticType, and which probe came closest?"
///
/// Unlike most measures, this one is **not** updated during Pass 2:
/// the tallies are captured during Pass 1 inside `ExplorationProbe`
/// and the orchestrator hands them in at construction. We model it
/// as a Measure for uniform reporting; `observe` is a no-op.
pub struct ProbeAttemptMeasure {
    tallies: Vec<crate::pipeline::commands::survey::probes::ProbeTally>,
}

impl ProbeAttemptMeasure {
    pub fn from_tallies(
        tallies: Vec<crate::pipeline::commands::survey::probes::ProbeTally>,
    ) -> Self {
        ProbeAttemptMeasure { tallies }
    }
}

impl Measure for ProbeAttemptMeasure {
    fn observe(&mut self, _value: &MValue, _ctx: &MeasureCtx) {}

    fn finalize(self: Box<Self>) -> MeasureReport {
        // Sort by descending match rate so the most-relevant
        // candidate appears first in the report.
        let mut entries = self.tallies;
        entries.sort_by(|a, b| {
            b.match_rate().partial_cmp(&a.match_rate()).unwrap_or(std::cmp::Ordering::Equal)
        });
        MeasureReport::ProbeAttempt(ProbeAttemptReport {
            attempts: entries
                .into_iter()
                .map(|t| ProbeAttemptEntry {
                    kind: t.kind,
                    matches: t.matches,
                    samples: t.samples,
                    match_rate: if t.samples == 0 { 0.0 } else { t.matches as f64 / t.samples as f64 },
                })
                .collect(),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ProbeAttempt
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeAttemptReport {
    pub attempts: Vec<ProbeAttemptEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeAttemptEntry {
    pub kind: String,
    pub matches: u64,
    pub samples: u64,
    pub match_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn histogram_sorts_descending() {
        let mut m = WireEncodingHistogramMeasure::new();
        for _ in 0..7 { m.observe(&MValue::Text("a".into()), &ctx()); }
        for _ in 0..3 { m.observe(&MValue::Int(1), &ctx()); }
        for _ in 0..1 { m.observe(&MValue::Bool(true), &ctx()); }
        let r = match Box::new(m).finalize() {
            MeasureReport::WireEncodingHistogram(r) => r,
            _ => panic!("wrong report kind"),
        };
        let keys: Vec<&str> = r.fractions.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["Text", "Int", "Bool"]);
        assert!((r.fractions.get("Text").copied().unwrap() - 7.0 / 11.0).abs() < 1e-9);
        assert_eq!(r.total, 11);
    }

    #[test]
    fn length_range_text_and_bytes() {
        let mut m = ByteOrCharLengthRangeMeasure::new();
        m.observe(&MValue::Text("hi".into()), &ctx());
        m.observe(&MValue::Text("hello".into()), &ctx());
        m.observe(&MValue::Bytes(vec![0; 1024]), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::ByteOrCharLengthRange(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.bytes_count, 3);
        assert_eq!(r.bytes_min, Some(2));
        assert_eq!(r.bytes_max, Some(1024));
        assert_eq!(r.chars_count, 2);
        assert_eq!(r.chars_min, Some(2));
        assert_eq!(r.chars_max, Some(5));
    }

    #[test]
    fn ignores_non_text_non_bytes() {
        let mut m = ByteOrCharLengthRangeMeasure::new();
        m.observe(&MValue::Int(42), &ctx());
        m.observe(&MValue::Null, &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::ByteOrCharLengthRange(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.bytes_count, 0);
        assert_eq!(r.chars_count, 0);
        assert!(r.bytes_min.is_none());
    }
}
