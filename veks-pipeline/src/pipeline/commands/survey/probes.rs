// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Semantic probes for Pass 1 type classification.
//!
//! A [`SemanticProbe`] inspects an `MValue` and returns `Some(verdict)`
//! when the value matches its pattern. Probes turn a wire-level
//! observation (e.g. `MValue::Text("2143")` or `MValue::Int(1700000000)`)
//! into a richer [`SemanticType`] like `Number(Integer)` or
//! `Temporal(Timestamp)`.
//!
//! The orchestrator runs every applicable probe against every value
//! in the field's reservoir, tracks per-probe match rates, and
//! commits to the highest-rate probe above the field's confidence
//! threshold. Unmatched fields fall through to the encoding-only
//! verdict assigned in [`super::template::classify_semantic`].
//!
//! See sysref §13.3.3 for the full probe-table specification.

use veks_core::formats::mnode::MValue;

use super::types::{
    BinaryKind, CategoricalKind, IdentifierKind, NumberKind, NumericWidth, SemanticType,
    StructuredKind, TemporalKind, TimestampGranularity,
};

/// Trait implemented by every Pass 1 semantic probe.
///
/// Probes are pure — no internal state, no allocation outside the
/// per-invocation parse — so a single registry instance is shared
/// across every field. Match rates are aggregated by the caller.
pub trait SemanticProbe: Sync + Send {
    /// Short kind identifier ("IntegerLiteralProbe", "EmailProbe", …).
    /// Used as the stable key in `ProbeAttemptReport` and in CLI
    /// per-field `--force-semantic-type` overrides.
    fn kind(&self) -> &'static str;

    /// Try to interpret `value` as the probe's semantic type. Return
    /// `Some(verdict)` on a match, `None` otherwise. The verdict
    /// carries the inferred subkind (e.g. integer width, identifier
    /// composite prefix).
    fn try_accept(&self, value: &MValue) -> Option<SemanticType>;
}

// ---------------------------------------------------------------------------
// Registry of default probes
// ---------------------------------------------------------------------------

/// Probes are tried in approximate cost order — direct-tag probes
/// first (free), then short-prefix probes, then full regex /
/// parse-required probes. The orchestrator picks the highest-rate
/// match, so order only affects tie-breaking when two probes match
/// equally often.
pub fn default_probes() -> Vec<Box<dyn SemanticProbe>> {
    vec![
        // Direct-tag probes (zero-cost for non-matching encodings)
        Box::new(DirectUuidProbe),
        Box::new(DirectBoolProbe),
        Box::new(MagicByteProbe),
        // Numeric-encoded plausibility
        Box::new(EpochMillisPlausibility),
        Box::new(EpochSecondsPlausibility),
        // Textual literals
        Box::new(IntegerLiteralProbe),
        Box::new(DecimalLiteralProbe),
        Box::new(FloatLiteralProbe),
        Box::new(BooleanLiteralProbe),
        // Textual structured formats
        Box::new(UuidStringProbe),
        Box::new(Iso8601DateProbe),
        Box::new(Iso8601DateTimeProbe),
        Box::new(EmailProbe),
        Box::new(UrlProbe),
        Box::new(Ipv4Probe),
        Box::new(Ipv6Probe),
        Box::new(PhoneNumberProbe),
        Box::new(GeocodeProbe),
        Box::new(CurrencyProbe),
        Box::new(HexFixedWidthProbe),
        Box::new(JsonProbe),
        Box::new(CompositeIdentifierProbe),
        Box::new(LabelsetProbe),
    ]
}

// ---------------------------------------------------------------------------
// LabelsetProbe — comma-separated short-token labelsets
// ---------------------------------------------------------------------------

/// Detects values shaped like `"a, b c, d e f"` — `,`-separated
/// chunks whose stem count per chunk is in `1..=3`. Used to drive
/// per-label heavy-hitter analysis (see
/// [`LabelsetHeavyHittersMeasure`](super::measures::labelset)).
///
/// Rule:
///   - Tokenize on `,`. Whitespace is split UTF-8-safely via
///     `split_whitespace`, so multi-byte chars never get
///     mid-codepoint-treated as ASCII.
///   - Require ≥2 chunks (one comma minimum) AND every chunk has
///     1–3 whitespace-delimited tokens.
///
/// A single value classifies a `try_accept` call; the orchestrator
/// aggregates match rate across the sample and commits the
/// verdict at `semantic_confidence` (default 0.95).
pub struct LabelsetProbe;
impl SemanticProbe for LabelsetProbe {
    fn kind(&self) -> &'static str { "LabelsetProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = match v {
            MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s) => s.as_str(),
            _ => return None,
        };
        if !is_labelset_shape(s) { return None; }
        Some(SemanticType::Categorical(CategoricalKind::Labelset))
    }
}

/// Decide whether `s` matches the labelset shape:
/// `,`-separated chunks where every chunk has 1..=3 whitespace
/// tokens AND there are ≥2 chunks. Used by both the Pass 1 probe
/// and the Pass 2 measure to gate per-record label extraction.
pub(crate) fn is_labelset_shape(s: &str) -> bool {
    // Reject empty / whitespace-only strings.
    if s.trim().is_empty() { return false; }
    // We need at least one comma → ≥2 chunks.
    if !s.contains(',') { return false; }
    let mut chunks = 0usize;
    for chunk in s.split(',') {
        let tokens = chunk.split_whitespace().count();
        if !(1..=3).contains(&tokens) {
            return false;
        }
        chunks += 1;
    }
    chunks >= 2
}

// ---------------------------------------------------------------------------
// Direct-tag probes (zero-cost)
// ---------------------------------------------------------------------------

pub struct DirectUuidProbe;
impl SemanticProbe for DirectUuidProbe {
    fn kind(&self) -> &'static str { "DirectUuidProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        match v {
            MValue::UuidV1(_) | MValue::UuidV7(_) | MValue::Ulid(_) => {
                Some(SemanticType::Identifier(IdentifierKind::Uuid))
            }
            _ => None,
        }
    }
}

pub struct DirectBoolProbe;
impl SemanticProbe for DirectBoolProbe {
    fn kind(&self) -> &'static str { "DirectBoolProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        match v {
            MValue::Bool(_) => Some(SemanticType::Boolean),
            _ => None,
        }
    }
}

pub struct MagicByteProbe;
impl SemanticProbe for MagicByteProbe {
    fn kind(&self) -> &'static str { "MagicByteProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let bytes = match v {
            MValue::Bytes(b) => b,
            _ => return None,
        };
        if bytes.len() < 4 {
            return Some(SemanticType::Binary(BinaryKind::Opaque));
        }
        let magic = matches!(
            &bytes[..bytes.len().min(8)],
            // PNG
            &[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, ..] |
            // gzip
            &[0x1f, 0x8b, ..] |
            // Parquet
            &[b'P', b'A', b'R', b'1', ..] |
            // JPEG SOI + APP0/APP1
            &[0xff, 0xd8, 0xff, ..] |
            // ZIP
            &[b'P', b'K', 0x03, 0x04, ..] |
            // Zstd
            &[0x28, 0xb5, 0x2f, 0xfd, ..]
        );
        if magic {
            Some(SemanticType::Binary(BinaryKind::Magic))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric-encoded plausibility probes
// ---------------------------------------------------------------------------

// Year-2000 .. year-2100 epoch ranges (inclusive lower, exclusive upper).
const EPOCH_SEC_MIN: i64 = 946_684_800;
const EPOCH_SEC_MAX: i64 = 4_102_444_800;
const EPOCH_MS_MIN: i64 = EPOCH_SEC_MIN * 1_000;
const EPOCH_MS_MAX: i64 = EPOCH_SEC_MAX * 1_000;

pub struct EpochSecondsPlausibility;
impl SemanticProbe for EpochSecondsPlausibility {
    fn kind(&self) -> &'static str { "EpochSecondsPlausibility" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let i = match v {
            MValue::Int(x) => *x,
            MValue::Int32(x) => *x as i64,
            _ => return None,
        };
        if (EPOCH_SEC_MIN..EPOCH_SEC_MAX).contains(&i) {
            Some(SemanticType::Temporal(TemporalKind::Timestamp {
                granularity: TimestampGranularity::Seconds,
            }))
        } else {
            None
        }
    }
}

pub struct EpochMillisPlausibility;
impl SemanticProbe for EpochMillisPlausibility {
    fn kind(&self) -> &'static str { "EpochMillisPlausibility" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let i = match v {
            MValue::Int(x) => *x,
            MValue::Millis(x) => *x,
            _ => return None,
        };
        if (EPOCH_MS_MIN..EPOCH_MS_MAX).contains(&i) {
            Some(SemanticType::Temporal(TemporalKind::Timestamp {
                granularity: TimestampGranularity::Millis,
            }))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Textual literal probes
// ---------------------------------------------------------------------------

fn as_text(v: &MValue) -> Option<&str> {
    match v {
        MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s)
        | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => Some(s.as_str()),
        _ => None,
    }
}

pub struct IntegerLiteralProbe;
impl SemanticProbe for IntegerLiteralProbe {
    fn kind(&self) -> &'static str { "IntegerLiteralProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        if s.is_empty() { return None; }
        let parsed: i64 = s.parse().ok()?;
        let signed = parsed < 0;
        let bit_width_hint = narrowest_int_width(parsed);
        Some(SemanticType::Number(NumberKind::Integer { signed, bit_width_hint }))
    }
}

fn narrowest_int_width(v: i64) -> NumericWidth {
    if v >= 0 {
        if v <= u8::MAX as i64 { NumericWidth::I8 }
        else if v <= u16::MAX as i64 { NumericWidth::I16 }
        else if v <= u32::MAX as i64 { NumericWidth::I32 }
        else { NumericWidth::I64 }
    } else if v >= i8::MIN as i64 && v <= i8::MAX as i64 { NumericWidth::I8 }
    else if v >= i16::MIN as i64 && v <= i16::MAX as i64 { NumericWidth::I16 }
    else if v >= i32::MIN as i64 && v <= i32::MAX as i64 { NumericWidth::I32 }
    else { NumericWidth::I64 }
}

pub struct DecimalLiteralProbe;
impl SemanticProbe for DecimalLiteralProbe {
    fn kind(&self) -> &'static str { "DecimalLiteralProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        // Must contain a decimal point and parse cleanly into f64.
        // Float-literal (scientific notation) is delegated to FloatLiteralProbe.
        if !s.contains('.') || s.contains(['e', 'E']) { return None; }
        let _: f64 = s.parse().ok()?;
        let (int_part, frac_part) = s.split_once('.').unwrap();
        let int_digits = int_part.trim_start_matches('-').len() as u8;
        let scale = frac_part.len() as u8;
        let precision = int_digits + scale;
        Some(SemanticType::Number(NumberKind::Decimal {
            precision_hint: precision,
            scale_hint: scale,
        }))
    }
}

pub struct FloatLiteralProbe;
impl SemanticProbe for FloatLiteralProbe {
    fn kind(&self) -> &'static str { "FloatLiteralProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        if !s.contains(['e', 'E']) { return None; }
        let _: f64 = s.parse().ok()?;
        Some(SemanticType::Number(NumberKind::Floating))
    }
}

pub struct BooleanLiteralProbe;
impl SemanticProbe for BooleanLiteralProbe {
    fn kind(&self) -> &'static str { "BooleanLiteralProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        let lower = s.to_ascii_lowercase();
        if matches!(lower.as_str(),
            "true" | "false" | "t" | "f" | "yes" | "no" | "y" | "n" | "1" | "0")
        {
            Some(SemanticType::Boolean)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Textual structured-format probes
// ---------------------------------------------------------------------------

pub struct UuidStringProbe;
impl SemanticProbe for UuidStringProbe {
    fn kind(&self) -> &'static str { "UuidStringProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        if s.len() != 36 { return None; }
        let bytes = s.as_bytes();
        if bytes[8] != b'-' || bytes[13] != b'-' || bytes[18] != b'-' || bytes[23] != b'-' {
            return None;
        }
        if bytes.iter().enumerate().any(|(i, &b)| {
            !matches!(i, 8 | 13 | 18 | 23) && !b.is_ascii_hexdigit()
        }) {
            return None;
        }
        Some(SemanticType::Identifier(IdentifierKind::Uuid))
    }
}

pub struct Iso8601DateProbe;
impl SemanticProbe for Iso8601DateProbe {
    fn kind(&self) -> &'static str { "Iso8601DateProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        // YYYY-MM-DD exactly.
        if s.len() != 10 { return None; }
        let bytes = s.as_bytes();
        if bytes[4] != b'-' || bytes[7] != b'-' { return None; }
        if !bytes[0..4].iter().all(|b| b.is_ascii_digit())
            || !bytes[5..7].iter().all(|b| b.is_ascii_digit())
            || !bytes[8..10].iter().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        Some(SemanticType::Temporal(TemporalKind::Date))
    }
}

pub struct Iso8601DateTimeProbe;
impl SemanticProbe for Iso8601DateTimeProbe {
    fn kind(&self) -> &'static str { "Iso8601DateTimeProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        // Minimum: YYYY-MM-DDTHH:MM:SS (19 chars).
        if s.len() < 19 { return None; }
        let bytes = s.as_bytes();
        if bytes[4] != b'-' || bytes[7] != b'-' { return None; }
        if !(bytes[10] == b'T' || bytes[10] == b' ') { return None; }
        if bytes[13] != b':' || bytes[16] != b':' { return None; }
        let has_timezone = s.ends_with('Z')
            || (s.len() >= 25
                && matches!(bytes[19], b'.' | b'+' | b'-' | b'Z')
                && (s.contains('+') || s.contains('Z') || (s.len() > 22 && s[20..].contains('-'))));
        Some(SemanticType::Temporal(TemporalKind::DateTime { has_timezone }))
    }
}

pub struct EmailProbe;
impl SemanticProbe for EmailProbe {
    fn kind(&self) -> &'static str { "EmailProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        if s.len() < 3 || s.len() > 254 { return None; }
        let (local, domain) = s.split_once('@')?;
        if local.is_empty() || domain.is_empty() { return None; }
        if !domain.contains('.') { return None; }
        if domain.starts_with('.') || domain.ends_with('.') { return None; }
        // Local-part character class — letters, digits, ._%+-
        if !local.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'%' | b'+' | b'-')) {
            return None;
        }
        // Domain — letters, digits, dots, hyphens.
        if !domain.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-')) {
            return None;
        }
        Some(SemanticType::Structured(StructuredKind::Email))
    }
}

pub struct UrlProbe;
impl SemanticProbe for UrlProbe {
    fn kind(&self) -> &'static str { "UrlProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        if s.len() < 8 { return None; }
        let lower = s.to_ascii_lowercase();
        if !(lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("s3://") || lower.starts_with("file://")) {
            return None;
        }
        let after_scheme = &s[s.find("://")? + 3..];
        if after_scheme.is_empty() { return None; }
        Some(SemanticType::Structured(StructuredKind::Url))
    }
}

pub struct Ipv4Probe;
impl SemanticProbe for Ipv4Probe {
    fn kind(&self) -> &'static str { "Ipv4Probe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 { return None; }
        for p in &parts {
            let n: u32 = p.parse().ok()?;
            if n > 255 { return None; }
        }
        Some(SemanticType::Structured(StructuredKind::Ipv4))
    }
}

pub struct Ipv6Probe;
impl SemanticProbe for Ipv6Probe {
    fn kind(&self) -> &'static str { "Ipv6Probe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        if !s.contains(':') { return None; }
        // Permissive: relies on std parser.
        let _: std::net::Ipv6Addr = s.parse().ok()?;
        Some(SemanticType::Structured(StructuredKind::Ipv6))
    }
}

pub struct PhoneNumberProbe;
impl SemanticProbe for PhoneNumberProbe {
    fn kind(&self) -> &'static str { "PhoneNumberProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        // E.164 style: optional + followed by 7..15 digits, allowing
        // spaces/dashes/parens between groups.
        let digits_only: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
        if !(7..=15).contains(&digits_only.len()) { return None; }
        // Require at least one of: leading '+', parens-grouped, hyphen-separated.
        let has_phone_punct = s.contains('+') || s.contains('(') || s.contains('-');
        if !has_phone_punct { return None; }
        // No alpha allowed.
        if s.bytes().any(|b| b.is_ascii_alphabetic()) { return None; }
        Some(SemanticType::Structured(StructuredKind::PhoneNumber))
    }
}

pub struct GeocodeProbe;
impl SemanticProbe for GeocodeProbe {
    fn kind(&self) -> &'static str { "GeocodeProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        let (a, b) = s.split_once(',')?;
        let lat: f64 = a.trim().parse().ok()?;
        let lng: f64 = b.trim().parse().ok()?;
        if !(-90.0..=90.0).contains(&lat) { return None; }
        if !(-180.0..=180.0).contains(&lng) { return None; }
        Some(SemanticType::Structured(StructuredKind::Geocode))
    }
}

pub struct CurrencyProbe;
impl SemanticProbe for CurrencyProbe {
    fn kind(&self) -> &'static str { "CurrencyProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        // Leading currency symbol or trailing ISO code + amount.
        let amount_text = if let Some(rest) = s
            .strip_prefix('$')
            .or_else(|| s.strip_prefix('€'))
            .or_else(|| s.strip_prefix('£'))
            .or_else(|| s.strip_prefix('¥'))
        {
            rest.trim()
        } else {
            return None;
        };
        // Optional thousands separators; one or two decimal digits.
        let cleaned = amount_text.replace(',', "");
        let _: f64 = cleaned.parse().ok()?;
        Some(SemanticType::Structured(StructuredKind::Currency))
    }
}

pub struct HexFixedWidthProbe;
impl SemanticProbe for HexFixedWidthProbe {
    fn kind(&self) -> &'static str { "HexFixedWidthProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
        // Reject too-short to be considered a hash-like identifier.
        if s.len() < 8 { return None; }
        if !s.bytes().all(|b| b.is_ascii_hexdigit()) { return None; }
        Some(SemanticType::Identifier(IdentifierKind::HashLike))
    }
}

pub struct JsonProbe;
impl SemanticProbe for JsonProbe {
    fn kind(&self) -> &'static str { "JsonProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        if !(s.starts_with('{') || s.starts_with('[')) { return None; }
        if serde_json::from_str::<serde_json::Value>(s).is_err() { return None; }
        Some(SemanticType::Structured(StructuredKind::Json))
    }
}

pub struct CompositeIdentifierProbe;
impl SemanticProbe for CompositeIdentifierProbe {
    fn kind(&self) -> &'static str { "CompositeIdentifierProbe" }
    fn try_accept(&self, v: &MValue) -> Option<SemanticType> {
        let s = as_text(v)?.trim();
        // Pattern: leading ASCII alpha prefix, an underscore or dash,
        // then a numeric or alphanumeric body.
        // Example matches: "USR_00123", "ORD-A12345".
        let (prefix, body) = s.split_once(['_', '-'])?;
        if prefix.len() < 2 || body.is_empty() { return None; }
        if !prefix.bytes().all(|b| b.is_ascii_alphabetic()) { return None; }
        if !body.bytes().all(|b| b.is_ascii_alphanumeric()) { return None; }
        // Body should have a fair share of digits — otherwise it's
        // probably free text with a punctuation collision.
        let digit_count = body.bytes().filter(|b| b.is_ascii_digit()).count();
        if digit_count * 2 < body.len() { return None; }
        Some(SemanticType::Identifier(IdentifierKind::Composite {
            prefix: Some(prefix.to_string()),
        }))
    }
}

// ---------------------------------------------------------------------------
// Match-rate aggregation
// ---------------------------------------------------------------------------

/// Per-(probe, field) tally collected during Pass 1. Drives both the
/// semantic-type commit decision and the `ProbeAttemptReport`
/// measure emitted for Unstable fields.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProbeTally {
    pub kind: String,
    pub matches: u64,
    pub samples: u64,
}

impl ProbeTally {
    pub fn match_rate(&self) -> f64 {
        if self.samples == 0 { 0.0 } else { self.matches as f64 / self.samples as f64 }
    }
}

/// Run every probe in `probes` against every value in `samples` and
/// return per-probe tallies plus the best-matching probe (if any
/// cleared `threshold`).
pub fn run_probes(
    probes: &[Box<dyn SemanticProbe>],
    samples: &[MValue],
    threshold: f64,
) -> (Vec<ProbeTally>, Option<(String, SemanticType, f64)>) {
    let mut tallies: Vec<ProbeTally> = probes
        .iter()
        .map(|p| ProbeTally {
            kind: p.kind().to_string(),
            matches: 0,
            samples: samples.len() as u64,
        })
        .collect();
    // Track each probe's most-recent verdict so we can return it
    // when the probe wins. We keep the *first* verdict it gives;
    // probes are deterministic per-input so this is fine.
    let mut first_verdict: Vec<Option<SemanticType>> = vec![None; probes.len()];
    for value in samples {
        for (i, p) in probes.iter().enumerate() {
            if let Some(v) = p.try_accept(value) {
                tallies[i].matches += 1;
                if first_verdict[i].is_none() {
                    first_verdict[i] = Some(v);
                }
            }
        }
    }
    let best = tallies
        .iter()
        .enumerate()
        .filter(|(_, t)| t.match_rate() >= threshold)
        .max_by(|(_, a), (_, b)| {
            a.match_rate().partial_cmp(&b.match_rate()).unwrap_or(std::cmp::Ordering::Equal)
        })
        .and_then(|(i, t)| first_verdict[i].clone().map(|v| (t.kind.clone(), v, t.match_rate())));
    (tallies, best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probes_accept(probe: impl SemanticProbe, value: MValue) -> Option<SemanticType> {
        probe.try_accept(&value)
    }

    #[test]
    fn integer_literal_recovers_width() {
        match probes_accept(IntegerLiteralProbe, MValue::Text("42".into())).unwrap() {
            SemanticType::Number(NumberKind::Integer { signed, bit_width_hint }) => {
                assert!(!signed);
                assert_eq!(bit_width_hint, NumericWidth::I8);
            }
            other => panic!("got {:?}", other),
        }
        match probes_accept(IntegerLiteralProbe, MValue::Text("-50000".into())).unwrap() {
            SemanticType::Number(NumberKind::Integer { signed, bit_width_hint }) => {
                assert!(signed);
                assert_eq!(bit_width_hint, NumericWidth::I32);
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn integer_literal_rejects_garbage() {
        assert!(probes_accept(IntegerLiteralProbe, MValue::Text("hello".into())).is_none());
        assert!(probes_accept(IntegerLiteralProbe, MValue::Text("1.5".into())).is_none());
        assert!(probes_accept(IntegerLiteralProbe, MValue::Int(42)).is_none());
    }

    #[test]
    fn float_literal_only_matches_scientific() {
        assert!(probes_accept(FloatLiteralProbe, MValue::Text("1.5e3".into())).is_some());
        assert!(probes_accept(FloatLiteralProbe, MValue::Text("1.5".into())).is_none());
    }

    #[test]
    fn decimal_literal_extracts_scale() {
        match probes_accept(DecimalLiteralProbe, MValue::Text("123.45".into())).unwrap() {
            SemanticType::Number(NumberKind::Decimal { precision_hint, scale_hint }) => {
                assert_eq!(scale_hint, 2);
                assert_eq!(precision_hint, 5);
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn uuid_string_matches_canonical() {
        let uuid = "01234567-89ab-cdef-fedc-ba9876543210";
        assert_eq!(
            probes_accept(UuidStringProbe, MValue::Text(uuid.into())),
            Some(SemanticType::Identifier(IdentifierKind::Uuid)),
        );
        assert!(probes_accept(UuidStringProbe, MValue::Text("not-a-uuid".into())).is_none());
    }

    #[test]
    fn email_probe() {
        for ok in &["alice@example.com", "bob.smith+filter@company.io", "c1@d.io"] {
            assert!(
                probes_accept(EmailProbe, MValue::Text((*ok).into())).is_some(),
                "should accept '{}'", ok,
            );
        }
        for bad in &["no-at-sign", "@no-local.com", "no-domain@", "spaces in@email.com"] {
            assert!(
                probes_accept(EmailProbe, MValue::Text((*bad).into())).is_none(),
                "should reject '{}'", bad,
            );
        }
    }

    #[test]
    fn url_probe() {
        assert!(probes_accept(UrlProbe, MValue::Text("https://example.com/path".into())).is_some());
        assert!(probes_accept(UrlProbe, MValue::Text("http://x.io".into())).is_some());
        assert!(probes_accept(UrlProbe, MValue::Text("just a string".into())).is_none());
    }

    #[test]
    fn iso_8601_date_strict() {
        assert!(probes_accept(Iso8601DateProbe, MValue::Text("2024-01-15".into())).is_some());
        assert!(probes_accept(Iso8601DateProbe, MValue::Text("2024-1-5".into())).is_none());
        assert!(probes_accept(Iso8601DateProbe, MValue::Text("2024/01/15".into())).is_none());
    }

    #[test]
    fn iso_8601_datetime() {
        assert!(probes_accept(Iso8601DateTimeProbe, MValue::Text("2024-01-15T10:30:00".into())).is_some());
        assert!(probes_accept(Iso8601DateTimeProbe, MValue::Text("2024-01-15T10:30:00Z".into())).is_some());
        assert!(probes_accept(Iso8601DateTimeProbe, MValue::Text("hello".into())).is_none());
    }

    #[test]
    fn ipv4_probe() {
        assert!(probes_accept(Ipv4Probe, MValue::Text("192.168.0.1".into())).is_some());
        assert!(probes_accept(Ipv4Probe, MValue::Text("256.0.0.1".into())).is_none());
        assert!(probes_accept(Ipv4Probe, MValue::Text("1.2.3".into())).is_none());
    }

    #[test]
    fn ipv6_probe() {
        assert!(probes_accept(Ipv6Probe, MValue::Text("::1".into())).is_some());
        assert!(probes_accept(Ipv6Probe, MValue::Text("2001:db8::1".into())).is_some());
    }

    #[test]
    fn epoch_seconds_plausibility() {
        assert!(probes_accept(EpochSecondsPlausibility, MValue::Int(1_700_000_000)).is_some());
        assert!(probes_accept(EpochSecondsPlausibility, MValue::Int(42)).is_none());
    }

    #[test]
    fn epoch_millis_plausibility() {
        assert!(probes_accept(EpochMillisPlausibility, MValue::Int(1_700_000_000_000)).is_some());
        assert!(probes_accept(EpochMillisPlausibility, MValue::Int(1_700_000_000)).is_none());
    }

    #[test]
    fn boolean_literal() {
        for ok in &["true", "False", "Yes", "no", "1", "0", "T", "f"] {
            assert!(
                probes_accept(BooleanLiteralProbe, MValue::Text((*ok).into())).is_some(),
                "should accept '{}'", ok,
            );
        }
        assert!(probes_accept(BooleanLiteralProbe, MValue::Text("maybe".into())).is_none());
    }

    #[test]
    fn composite_identifier() {
        match probes_accept(CompositeIdentifierProbe, MValue::Text("USR_00123".into())).unwrap() {
            SemanticType::Identifier(IdentifierKind::Composite { prefix }) => {
                assert_eq!(prefix.as_deref(), Some("USR"));
            }
            other => panic!("got {:?}", other),
        }
        assert!(probes_accept(CompositeIdentifierProbe, MValue::Text("free text".into())).is_none());
    }

    #[test]
    fn hex_fixed_width() {
        assert!(probes_accept(HexFixedWidthProbe, MValue::Text("deadbeefcafebabe".into())).is_some());
        assert!(probes_accept(HexFixedWidthProbe, MValue::Text("0xdeadbeef".into())).is_some());
        assert!(probes_accept(HexFixedWidthProbe, MValue::Text("not-hex".into())).is_none());
    }

    #[test]
    fn magic_byte_recognizes_png() {
        let png_header = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0xff];
        assert!(matches!(
            probes_accept(MagicByteProbe, MValue::Bytes(png_header)),
            Some(SemanticType::Binary(BinaryKind::Magic))
        ));
    }

    #[test]
    fn json_probe() {
        assert!(probes_accept(JsonProbe, MValue::Text(r#"{"k":1}"#.into())).is_some());
        assert!(probes_accept(JsonProbe, MValue::Text(r#"["a","b"]"#.into())).is_some());
        assert!(probes_accept(JsonProbe, MValue::Text("{not-json".into())).is_none());
    }

    #[test]
    fn currency_probe() {
        assert!(probes_accept(CurrencyProbe, MValue::Text("$12.50".into())).is_some());
        assert!(probes_accept(CurrencyProbe, MValue::Text("€1,000.00".into())).is_some());
        assert!(probes_accept(CurrencyProbe, MValue::Text("12.50".into())).is_none());
    }

    #[test]
    fn run_probes_picks_highest_match() {
        let probes = default_probes();
        let samples: Vec<MValue> = (0..50)
            .map(|i| MValue::Text(format!("user_{:04}@example.com", i)))
            .collect();
        let (tallies, best) = run_probes(&probes, &samples, 0.95);
        let (kind, verdict, rate) = best.expect("expected a winner");
        assert_eq!(kind, "EmailProbe");
        assert!((rate - 1.0).abs() < 1e-9);
        match verdict {
            SemanticType::Structured(StructuredKind::Email) => {}
            other => panic!("expected Email, got {:?}", other),
        }
        // Some probes should still have non-trivial match rates
        // (CompositeIdentifierProbe might match emails partially).
        let _ = tallies;
    }

    #[test]
    fn run_probes_returns_none_when_no_match() {
        let probes = default_probes();
        let samples: Vec<MValue> = (0..50)
            .map(|i| MValue::Text(format!("free text {}", i)))
            .collect();
        let (_tallies, best) = run_probes(&probes, &samples, 0.95);
        assert!(best.is_none());
    }
}
