// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Universal measures: run against every field regardless of its
//! `SemanticType` verdict (including `Unstable`).

use std::collections::HashSet;

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    ExactExtremaReport, ExactMomentsReport, Measure, MeasureCtx, MeasureKind, MeasureReport,
    PresenceReport, ReservoirSampleReport, TypeStabilityReport,
};
use crate::pipeline::commands::survey::sketches::Reservoir;

// ---------------------------------------------------------------------------
// PresenceMeasure
// ---------------------------------------------------------------------------

/// Counts presence, null, and absence for one field.
///
/// The orchestrator calls [`Measure::observe`] for every record that
/// carries this field and [`Measure::observe_missing`] for every
/// record that does not — the latter distinguishes a "structural
/// null" (field absent from the record entirely) from an explicit
/// `MValue::Null` value.
pub struct PresenceMeasure {
    present: u64,
    null_count: u64,
    absent_in_record: u64,
}

impl Default for PresenceMeasure {
    fn default() -> Self {
        Self::new()
    }
}

impl PresenceMeasure {
    pub fn new() -> Self {
        PresenceMeasure {
            present: 0,
            null_count: 0,
            absent_in_record: 0,
        }
    }
}

impl Measure for PresenceMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        self.present += 1;
        if matches!(value, MValue::Null) {
            self.null_count += 1;
        }
    }

    fn observe_missing(&mut self, _ctx: &MeasureCtx) {
        self.absent_in_record += 1;
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        MeasureReport::Presence(PresenceReport {
            present: self.present,
            null_count: self.null_count,
            absent_in_record: self.absent_in_record,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::Presence
    }
}

// ---------------------------------------------------------------------------
// TypeStabilityMeasure
// ---------------------------------------------------------------------------

/// Reports any Pass 2 observation whose `MValue` tag was absent
/// from Pass 1's tag histogram.
///
/// The orchestrator instantiates this measure for every field, with
/// `expected_tags` populated from the Pass 1 verdict. New tags seen
/// in Pass 2 surface as warnings in the findings report.
pub struct TypeStabilityMeasure {
    expected: HashSet<&'static str>,
    surprise_count: u64,
    surprise_tags: HashSet<String>,
}

impl TypeStabilityMeasure {
    pub fn new(expected_tags: impl IntoIterator<Item = &'static str>) -> Self {
        TypeStabilityMeasure {
            expected: expected_tags.into_iter().collect(),
            surprise_count: 0,
            surprise_tags: HashSet::new(),
        }
    }
}

impl Measure for TypeStabilityMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let tag = format!("{:?}", value.tag());
        // `tag` here is e.g. "Int", "Text", … — a stable
        // round-trippable string.
        if !self.expected.contains(tag.as_str()) {
            self.surprise_count += 1;
            self.surprise_tags.insert(tag);
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let mut tags: Vec<String> = self.surprise_tags.into_iter().collect();
        tags.sort();
        MeasureReport::TypeStability(TypeStabilityReport {
            surprise_count: self.surprise_count,
            surprise_tags: tags,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::TypeStability
    }
}

// ---------------------------------------------------------------------------
// ReservoirSample
// ---------------------------------------------------------------------------

/// Maintains a bounded reservoir of representative values for the
/// report's "examples" section.
///
/// Values are rendered as `serde_json::Value` for the report so the
/// JSON output stays human-readable.
pub struct ReservoirSample {
    inner: Reservoir<serde_json::Value>,
}

impl ReservoirSample {
    pub fn new(capacity: usize, seed: u64) -> Self {
        ReservoirSample {
            inner: Reservoir::new(capacity, seed),
        }
    }

    /// Borrow the underlying reservoir. Used by the orchestrator's
    /// downscale policy to call `shrink_to`.
    pub fn reservoir_mut(&mut self) -> &mut Reservoir<serde_json::Value> {
        &mut self.inner
    }
}

impl Measure for ReservoirSample {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        self.inner.add(mvalue_to_json(value));
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let observed = self.inner.seen();
        MeasureReport::ReservoirSample(ReservoirSampleReport {
            items: self.inner.into_items(),
            observed,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ReservoirSample
    }
}

/// Convert an `MValue` into a JSON-friendly form for inclusion in
/// the reservoir / findings report. Bytes render as a hex string;
/// UUIDs render as standard 8-4-4-4-12 hex; collections render
/// recursively. Lossy by design — the goal is operator readability,
/// not a perfect round-trip.
pub(crate) fn mvalue_to_json(value: &MValue) -> serde_json::Value {
    use serde_json::Value;
    match value {
        MValue::Null => Value::Null,
        MValue::Bool(b) => Value::Bool(*b),
        MValue::Int(i) => Value::from(*i),
        MValue::Int32(i) => Value::from(*i),
        MValue::Short(i) => Value::from(*i),
        MValue::Float(f) => Value::from(*f),
        MValue::Float32(f) => Value::from(*f as f64),
        MValue::Half(h) => Value::from(half::f16::from_bits(*h).to_f64()),
        MValue::Millis(m) => Value::from(*m),
        MValue::Nanos { epoch_seconds, nano_adjust } => Value::from(format!(
            "{}.{:09}", epoch_seconds, nano_adjust
        )),
        MValue::EnumOrd(o) => Value::from(*o),
        MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s)
        | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => Value::String(s.clone()),
        MValue::Bytes(b) => Value::String(hex_encode(b)),
        MValue::UuidV1(u) | MValue::UuidV7(u) | MValue::Ulid(u) => Value::String(format_uuid(u)),
        MValue::List(items) | MValue::Set(items) => {
            Value::Array(items.iter().map(mvalue_to_json).collect())
        }
        MValue::Array(_, items) => Value::Array(items.iter().map(mvalue_to_json).collect()),
        MValue::Map(node) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in &node.fields {
                obj.insert(k.clone(), mvalue_to_json(v));
            }
            Value::Object(obj)
        }
        MValue::TypedMap(entries) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in entries {
                let key = match mvalue_to_json(k) {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                obj.insert(key, mvalue_to_json(v));
            }
            Value::Object(obj)
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    let h = hex_encode(bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32]
    )
}

// Brought into scope so universal.rs can re-use the structs defined
// in `super::super::measure`. (Compile-time only; not used in
// runtime logic.)
#[allow(dead_code)]
fn _silence_unused() {
    let _ = std::mem::size_of::<ExactExtremaReport>();
    let _ = std::mem::size_of::<ExactMomentsReport>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::commands::survey::measure::Measure;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn presence_counts_present_null_and_absent() {
        let mut m = PresenceMeasure::new();
        m.observe(&MValue::Int(1), &ctx());
        m.observe(&MValue::Null, &ctx());
        m.observe(&MValue::Int(2), &ctx());
        m.observe_missing(&ctx());
        m.observe_missing(&ctx());
        match Box::new(m).finalize() {
            MeasureReport::Presence(r) => {
                assert_eq!(r.present, 3);
                assert_eq!(r.null_count, 1);
                assert_eq!(r.absent_in_record, 2);
            }
            _ => panic!("wrong report kind"),
        }
    }

    #[test]
    fn type_stability_detects_unexpected_tags() {
        // Expecting only Int; observing Text triggers a surprise.
        let mut m = TypeStabilityMeasure::new(["Int"]);
        m.observe(&MValue::Int(1), &ctx());
        m.observe(&MValue::Int(2), &ctx());
        m.observe(&MValue::Text("oops".into()), &ctx());
        m.observe(&MValue::Float(1.5), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TypeStability(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.surprise_count, 2);
        // Sorted for determinism.
        assert_eq!(r.surprise_tags, vec!["Float".to_string(), "Text".to_string()]);
    }

    #[test]
    fn type_stability_clean_field_no_surprises() {
        let mut m = TypeStabilityMeasure::new(["Text"]);
        for s in ["a", "b", "c"] {
            m.observe(&MValue::Text(s.into()), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::TypeStability(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.surprise_count, 0);
        assert!(r.surprise_tags.is_empty());
    }

    #[test]
    fn reservoir_renders_canonical_json_forms() {
        let mut m = ReservoirSample::new(8, 42);
        m.observe(&MValue::Int(42), &ctx());
        m.observe(&MValue::Text("hello".into()), &ctx());
        m.observe(&MValue::Bool(true), &ctx());
        m.observe(&MValue::Bytes(vec![0xde, 0xad, 0xbe, 0xef]), &ctx());
        m.observe(&MValue::UuidV7([
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
            0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
        ]), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::ReservoirSample(r) => r,
            _ => panic!("wrong report kind"),
        };
        // All 5 items fit in capacity 8.
        assert_eq!(r.items.len(), 5);
        assert_eq!(r.observed, 5);
        // Bytes render as hex.
        let any_hex = r.items.iter().any(|v| v.as_str() == Some("deadbeef"));
        assert!(any_hex, "expected deadbeef in {:?}", r.items);
        // UUID renders as 8-4-4-4-12.
        let any_uuid = r.items.iter().any(|v| {
            v.as_str() == Some("01234567-89ab-cdef-fedc-ba9876543210")
        });
        assert!(any_uuid, "expected UUID in {:?}", r.items);
    }

    #[test]
    fn reservoir_sub_sample_under_pressure() {
        let mut m = ReservoirSample::new(16, 1);
        for i in 0..1_000 {
            m.observe(&MValue::Int(i), &ctx());
        }
        m.reservoir_mut().shrink_to(4);
        let r = match Box::new(m).finalize() {
            MeasureReport::ReservoirSample(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.items.len(), 4);
        assert_eq!(r.observed, 1_000);
    }
}
