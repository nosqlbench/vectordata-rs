// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Core type model for the metadata survey.
//!
//! Two independent classification axes track every field:
//!
//! - [`WireEncoding`] — how the bytes arrived (the `MValue` tag),
//!   no interpretation.
//! - [`SemanticType`] — what the value means, determined by Pass 1
//!   semantic probes.
//!
//! See `docs/sysref/13-metadata-survey.md` §13.3 for the full rationale.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

// ---------------------------------------------------------------------------
// Wire encoding (storage-faithful classification)
// ---------------------------------------------------------------------------

/// Coarse encoding family.
///
/// Distinct from the detailed `MValue` tag set, this groups every
/// observed encoding into a small number of buckets so that
/// downstream measure selection can match on broad families without
/// enumerating 26 enum variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum WireEncodingKind {
    /// Bool / EnumOrd-as-boolean.
    Bool,
    /// Any signed integer or epoch numeric type.
    Numeric,
    /// Text / Ascii / EnumStr / Date / Time / DateTime (string-based).
    Textual,
    /// Bytes.
    Bytes,
    /// Direct identifier (`UuidV1`, `UuidV7`, `Ulid`).
    Identifier,
    /// List / Map / Array / Set / TypedMap / Nested record. The
    /// survey does not recurse into containers in this version —
    /// they are recorded as observed but trigger no further analysis.
    Collection,
    /// Null. Appears in the histogram even though it is not a
    /// "field type" per se.
    Null,
}

/// Width hint for numeric wire encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum NumericWidth {
    /// 8-bit signed/unsigned.
    I8,
    /// 16-bit signed/unsigned.
    I16,
    /// 32-bit signed/unsigned.
    I32,
    /// 64-bit signed/unsigned.
    I64,
    /// IEEE 754 half-precision.
    F16,
    /// IEEE 754 single-precision.
    F32,
    /// IEEE 754 double-precision.
    F64,
}

/// Full wire-encoding verdict for a field.
///
/// Tracks both the broad family (`kind`) and refinements: the precise
/// `MValue` tag(s) observed and, for numerics, the narrowest width
/// that covers the observed value range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireEncoding {
    /// Broad family selected for this field.
    pub kind: WireEncodingKind,
    /// Storage width — the actual width used by the wire type (e.g.
    /// `Int` is `I64` even when values fit in `I8`). `None` for
    /// non-numeric kinds.
    pub storage_width: Option<NumericWidth>,
    /// Narrowest width that covers every observed value. May be
    /// smaller than `storage_width`. `None` for non-numeric kinds.
    pub narrowest_width: Option<NumericWidth>,
    /// Tag histogram (variant name → fraction). Always present;
    /// shows the distribution of `MValue` tags seen for this field
    /// during Pass 1.
    pub tag_histogram: indexmap::IndexMap<String, f64>,
    /// True iff more than one mutually-incompatible kind was
    /// observed (sets the field to `SemanticType::Unstable`).
    pub mixed: bool,
}

impl WireEncoding {
    /// Map an `MValue` tag to its `WireEncodingKind`.
    pub fn kind_for(value: &MValue) -> WireEncodingKind {
        use veks_core::formats::mnode::MValue::*;
        match value {
            Bool(_) => WireEncodingKind::Bool,
            Int(_) | Int32(_) | Short(_) | Float(_) | Float32(_) | Half(_) | Millis(_) | Nanos { .. } | EnumOrd(_) => {
                WireEncodingKind::Numeric
            }
            Text(_) | Ascii(_) | EnumStr(_) | Date(_) | Time(_) | DateTime(_) => WireEncodingKind::Textual,
            Bytes(_) => WireEncodingKind::Bytes,
            UuidV1(_) | UuidV7(_) | Ulid(_) => WireEncodingKind::Identifier,
            List(_) | Map(_) | Array(_, _) | Set(_) | TypedMap(_) => WireEncodingKind::Collection,
            Null => WireEncodingKind::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// Semantic type (meaning, determined by Pass 1 probes)
// ---------------------------------------------------------------------------

/// What a field's values *mean*, regardless of how they're encoded.
///
/// Determined by Pass 1 semantic probes (§13.3.3). A textual
/// `"2143"` and an `Int 2143` share `SemanticType::Number(NumberKind::Integer{..})`
/// and are analyzed together with the same numeric measure suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "PascalCase")]
pub enum SemanticType {
    /// Numeric meaning.
    Number(NumberKind),
    /// Boolean meaning (encoded either as `Bool`, integer 0/1, or
    /// stringy `"true"`/`"false"`).
    Boolean,
    /// Temporal meaning.
    Temporal(TemporalKind),
    /// Identifier meaning.
    Identifier(IdentifierKind),
    /// Categorical (small / bounded vocabulary).
    Categorical(CategoricalKind),
    /// Structured value with a known parse format.
    Structured(StructuredKind),
    /// Free text (no detectable structural pattern).
    FreeText,
    /// Binary blob.
    Binary(BinaryKind),
    /// No verdict — only opaque-value measures run (§13.5.8).
    Unstable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "subkind", rename_all = "PascalCase")]
pub enum NumberKind {
    /// Integer value.
    Integer {
        /// True if every observed value was non-negative.
        signed: bool,
        /// Smallest width that covers the observed range.
        bit_width_hint: NumericWidth,
    },
    /// Decimal (fixed-point) value.
    Decimal { precision_hint: u8, scale_hint: u8 },
    /// Floating-point value.
    Floating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum TemporalKind {
    /// Calendar date.
    Date,
    /// Time of day.
    Time,
    /// Date-time (with or without timezone).
    DateTime {
        /// True iff every observation carried a timezone offset.
        has_timezone: bool,
    },
    /// Timestamp at a specific granularity.
    Timestamp { granularity: TimestampGranularity },
    /// Duration.
    Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TimestampGranularity {
    /// Seconds since epoch.
    Seconds,
    /// Milliseconds since epoch.
    Millis,
    /// Microseconds since epoch.
    Micros,
    /// Nanoseconds since epoch.
    Nanos,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "subkind", rename_all = "PascalCase")]
pub enum IdentifierKind {
    /// RFC 4122 UUID.
    Uuid,
    /// ULID.
    Ulid,
    /// Densely packed integer identifier.
    Sequential,
    /// Fixed-width hex / base64 hash-like identifier.
    HashLike,
    /// Composite identifier (prefix + body).
    Composite { prefix: Option<String> },
    /// Identifier-shaped but no parser matched.
    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CategoricalKind {
    /// Closed small set.
    Enum,
    /// Bounded but open set (mid-cardinality).
    OpenSet,
    /// Comma-separated list of short labels (1–3 whitespace
    /// tokens per chunk, ≥2 chunks per value). Predicate
    /// generators target labels, not whole-string values.
    Labelset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum StructuredKind {
    /// Email address.
    Email,
    /// URL.
    Url,
    /// IPv4 address.
    Ipv4,
    /// IPv6 address.
    Ipv6,
    /// E.164-style phone number.
    PhoneNumber,
    /// Lat/lng geocode.
    Geocode,
    /// Currency-formatted amount.
    Currency,
    /// JSON literal (recursive content not surveyed in this version).
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum BinaryKind {
    /// High entropy — likely compressed or random.
    Compressed,
    /// Magic-byte prefix matched a known format.
    Magic,
    /// No detectable structure.
    Opaque,
}

// ---------------------------------------------------------------------------
// Cardinality regime (decided at end of Pass 1)
// ---------------------------------------------------------------------------

/// Coarse cardinality bucket for a field.
///
/// Determines which cardinality-related measures Pass 2 instantiates.
/// `Constant` and `Binary` get exact frequency tables; `LowCard` gets
/// exact frequencies; `MidCard` gets HLL + heavy hitters; `HighCard`
/// gets HLL alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "PascalCase")]
pub enum CardinalityRegime {
    /// All non-null observations were the same value.
    Constant,
    /// Exactly two distinct values.
    Binary,
    /// Cardinality ≤ `low_card_threshold` and fully enumerated.
    LowCard { exact_distinct: u32 },
    /// Cardinality between `low_card_threshold` and `mid_card_threshold`.
    MidCard { hll_estimate_at_pass1: f64 },
    /// Cardinality above `mid_card_threshold` (or essentially unique).
    HighCardOrUnique { uniqueness_ratio: f64 },
    /// Cardinality unknown (Unstable field; nothing meaningful tracked).
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use veks_core::formats::mnode::MValue;

    #[test]
    fn wire_encoding_kind_groups_numeric_variants() {
        for v in [
            MValue::Int(0),
            MValue::Int32(0),
            MValue::Short(0),
            MValue::Float(0.0),
            MValue::Float32(0.0),
            MValue::Half(0),
            MValue::Millis(0),
            MValue::Nanos { epoch_seconds: 0, nano_adjust: 0 },
            MValue::EnumOrd(0),
        ] {
            assert_eq!(WireEncoding::kind_for(&v), WireEncodingKind::Numeric);
        }
    }

    #[test]
    fn wire_encoding_kind_groups_textual_variants() {
        for v in [
            MValue::Text(String::new()),
            MValue::Ascii(String::new()),
            MValue::EnumStr(String::new()),
            MValue::Date(String::new()),
            MValue::Time(String::new()),
            MValue::DateTime(String::new()),
        ] {
            assert_eq!(WireEncoding::kind_for(&v), WireEncodingKind::Textual);
        }
    }

    #[test]
    fn wire_encoding_kind_distinguishes_identifiers() {
        assert_eq!(WireEncoding::kind_for(&MValue::UuidV1([0; 16])), WireEncodingKind::Identifier);
        assert_eq!(WireEncoding::kind_for(&MValue::UuidV7([0; 16])), WireEncodingKind::Identifier);
        assert_eq!(WireEncoding::kind_for(&MValue::Ulid([0; 16])), WireEncodingKind::Identifier);
    }

    #[test]
    fn wire_encoding_kind_handles_collections() {
        assert_eq!(WireEncoding::kind_for(&MValue::List(vec![])), WireEncodingKind::Collection);
        assert_eq!(
            WireEncoding::kind_for(&MValue::Map(veks_core::formats::mnode::MNode {
                fields: indexmap::IndexMap::new(),
            })),
            WireEncodingKind::Collection
        );
    }

    /// Round-trip a sample `SemanticType` through JSON to verify the
    /// tagged-enum representation is what the §13.8 schema specifies.
    #[test]
    fn semantic_type_json_roundtrip() {
        let st = SemanticType::Number(NumberKind::Integer {
            signed: true,
            bit_width_hint: NumericWidth::I32,
        });
        let s = serde_json::to_string(&st).unwrap();
        // Outer tag "kind", inner tag "subkind" per the §13.8 example.
        assert!(s.contains("\"kind\":\"Number\""), "got: {}", s);
        assert!(s.contains("\"subkind\":\"Integer\""), "got: {}", s);
        let back: SemanticType = serde_json::from_str(&s).unwrap();
        assert_eq!(back, st);
    }

    /// Cardinality regimes serialize with `kind` discriminator.
    #[test]
    fn cardinality_regime_json_shape() {
        let r = CardinalityRegime::LowCard { exact_distinct: 7 };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"kind\":\"LowCard\""), "got: {}", s);
        assert!(s.contains("\"exact_distinct\":7"), "got: {}", s);
    }
}
