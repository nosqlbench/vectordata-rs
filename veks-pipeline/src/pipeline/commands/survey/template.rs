// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pass-1 exploration state and Pass-2 template synthesis.
//!
//! The [`ExplorationProbe`] is what Pass 1 accumulates per field —
//! a cheap, bounded pilot that tracks just enough information to
//! decide the field's classified type and cardinality regime at the
//! end of the pass. Its output, the [`FieldTemplate`], is the
//! contract handed to Pass 2 telling it which measures to
//! instantiate.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use super::measure::MeasureKind;
use super::probes::{default_probes, run_probes, ProbeTally};
use super::sketches::Reservoir;
use super::types::{
    CardinalityRegime, NumberKind, NumericWidth, SemanticType, WireEncoding, WireEncodingKind,
};

/// Per-field exploration state used during Pass 1.
///
/// Bounded-memory accumulator: each component is either O(1) or a
/// small fixed-capacity buffer.
pub struct ExplorationProbe {
    /// Configured template thresholds.
    cfg: TemplateConfig,
    // -- presence --
    present: u64,
    null_count: u64,
    // -- wire encoding --
    tag_counts: IndexMap<String, u64>,
    /// Kinds observed across all non-null observations. A field
    /// with more than one kind transitions to `Unstable`.
    kinds: HashMap<WireEncodingKind, u64>,
    // -- numeric pilot (only updated for Numeric/Bool kinds) --
    numeric_min: f64,
    numeric_max: f64,
    numeric_saw_any: bool,
    /// Most-restrictive width needed to hold the observed range
    /// without loss. Updated lazily; finalized at end of Pass 1.
    numeric_narrowest: Option<NumericWidth>,
    /// Whether any negative numeric value was observed.
    numeric_negative_seen: bool,
    // -- string length pilot --
    strlen_min: usize,
    strlen_max: usize,
    strlen_saw_any: bool,
    // -- bounded distinct tracking --
    distinct: IndexMap<String, u64>,
    distinct_overflow: bool,
    // -- representative sample --
    /// Sampled raw values used by §13.3.3 semantic probes during
    /// `finalize`. Kept as MValues (not pre-rendered to JSON) so
    /// probes can match on tag-specific shapes (Bytes magic, direct
    /// UUIDs, etc.).
    reservoir: Reservoir<MValue>,
}

/// Operator-tunable thresholds for template synthesis.
#[derive(Debug, Clone)]
pub struct TemplateConfig {
    /// Cardinality at-or-below which a field is `LowCard` and gets
    /// an exact frequency table.
    pub low_card_threshold: u32,
    /// Cardinality above which a field is `HighCardOrUnique`.
    pub mid_card_threshold: u32,
    /// Bounded distinct-tracker capacity during Pass 1. Should be
    /// at least `mid_card_threshold` so that mid-card fields can be
    /// counted exactly during exploration.
    pub distinct_cap: u32,
    /// Reservoir size carried through the template into Pass 2.
    pub reservoir_size: usize,
    /// Reservoir seed (reproducibility).
    pub reservoir_seed: u64,
    /// Minimum semantic-probe match rate required to commit to a
    /// non-encoding-floor semantic type. Probes below this rate are
    /// ignored; the encoding-only verdict stands. Sysref §13.3.3.
    pub semantic_confidence: f64,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        TemplateConfig {
            low_card_threshold: 64,
            mid_card_threshold: 4_096,
            distinct_cap: 4_096,
            reservoir_size: 1_024,
            reservoir_seed: 0xC011EC70,
            semantic_confidence: 0.95,
        }
    }
}

impl ExplorationProbe {
    pub fn new(cfg: TemplateConfig) -> Self {
        let reservoir_size = cfg.reservoir_size;
        let reservoir_seed = cfg.reservoir_seed;
        ExplorationProbe {
            cfg,
            present: 0,
            null_count: 0,
            tag_counts: IndexMap::new(),
            kinds: HashMap::new(),
            numeric_min: f64::INFINITY,
            numeric_max: f64::NEG_INFINITY,
            numeric_saw_any: false,
            numeric_narrowest: None,
            numeric_negative_seen: false,
            strlen_min: usize::MAX,
            strlen_max: 0,
            strlen_saw_any: false,
            distinct: IndexMap::new(),
            distinct_overflow: false,
            reservoir: Reservoir::new(reservoir_size, reservoir_seed),
        }
    }

    /// True iff no observations have been recorded for this field.
    pub fn is_empty(&self) -> bool {
        self.present == 0
    }

    /// Number of non-null observations.
    pub fn present(&self) -> u64 {
        self.present
    }

    /// Number of `MValue::Null` observations.
    pub fn null_count(&self) -> u64 {
        self.null_count
    }

    /// Total observations (present + null + absent).
    pub fn observed(&self) -> u64 {
        self.present
    }

    /// Record one observation for this field.
    pub fn observe(&mut self, value: &MValue) {
        self.present += 1;
        let tag_name = format!("{:?}", value.tag());
        *self.tag_counts.entry(tag_name).or_insert(0) += 1;
        if matches!(value, MValue::Null) {
            self.null_count += 1;
            return;
        }
        let kind = WireEncoding::kind_for(value);
        *self.kinds.entry(kind).or_insert(0) += 1;
        // Type-specific pilots
        match kind {
            WireEncodingKind::Numeric | WireEncodingKind::Bool => self.update_numeric(value),
            WireEncodingKind::Textual => self.update_strlen(value),
            _ => {}
        }
        // Bounded distinct tracker (representation: `Debug` form
        // truncated to keep memory bounded).
        if !self.distinct_overflow {
            let repr = repr_for_distinct(value);
            if self.distinct.contains_key(&repr) {
                *self.distinct.get_mut(&repr).unwrap() += 1;
            } else if (self.distinct.len() as u32) < self.cfg.distinct_cap {
                self.distinct.insert(repr, 1);
            } else {
                self.distinct_overflow = true;
            }
        }
        // Reservoir of raw values
        self.reservoir.add(value.clone());
    }

    fn update_numeric(&mut self, value: &MValue) {
        let Some(x) = super::measures::numeric::mvalue_as_f64(value) else { return };
        if x.is_nan() { return }
        self.numeric_saw_any = true;
        if x < self.numeric_min { self.numeric_min = x }
        if x > self.numeric_max { self.numeric_max = x }
        if x < 0.0 { self.numeric_negative_seen = true; }
    }

    fn update_strlen(&mut self, value: &MValue) {
        let s: Option<&str> = match value {
            MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s)
            | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => Some(s.as_str()),
            _ => None,
        };
        if let Some(s) = s {
            let len = s.len();
            self.strlen_saw_any = true;
            if len < self.strlen_min { self.strlen_min = len; }
            if len > self.strlen_max { self.strlen_max = len; }
        }
    }

    /// End-of-Pass-1 conversion: synthesize the [`FieldTemplate`]
    /// from accumulated state. Encoding family, semantic type, and
    /// cardinality regime are all decided here.
    pub fn finalize(self, _field_name: &str) -> FieldTemplate {
        // Always-null fields stay in the conceptual `Unknown` root
        // — see §13.2. We materialize them as a template with no
        // measures other than Presence, so the orchestrator can
        // still report counts.
        if self.present == 0 || self.present == self.null_count {
            return FieldTemplate {
                wire_encoding: WireEncoding {
                    kind: WireEncodingKind::Null,
                    storage_width: None,
                    narrowest_width: None,
                    tag_histogram: histogram_from(self.tag_counts, self.present),
                    mixed: false,
                },
                // No semantic verdict; nothing committed.
                semantic_type: None,
                semantic_confidence: 0.0,
                cardinality_regime: CardinalityRegime::Unknown,
                measures: vec![MeasureKind::Presence],
                probe_tallies: Vec::new(),
            };
        }

        let mixed_kinds = self.kinds.len() > 1;
        let kind = if mixed_kinds {
            WireEncodingKind::Null // placeholder, overridden below for the mixed-encoding case
        } else {
            *self.kinds.keys().next().expect("at least one kind by construction")
        };
        let (mut semantic, mut semantic_confidence) = if mixed_kinds {
            (Some(SemanticType::Unstable), 0.0)
        } else {
            classify_semantic(kind, self.numeric_negative_seen, self.numeric_min, self.numeric_max)
        };

        // Run semantic probes against the reservoir samples and
        // upgrade the verdict if any probe clears the confidence
        // threshold. The encoding-only verdict is kept as the floor.
        let reservoir_samples: Vec<MValue> = self.reservoir.items().to_vec();
        let (probe_tallies, best_probe) =
            run_probes(&default_probes(), &reservoir_samples, self.cfg.semantic_confidence);
        if !mixed_kinds
            && let Some((_kind_label, verdict, rate)) = best_probe {
                semantic = Some(verdict);
                semantic_confidence = rate;
            }

        let cardinality = classify_cardinality(
            semantic.as_ref(),
            &self.distinct,
            self.distinct_overflow,
            self.present.saturating_sub(self.null_count),
            self.cfg.low_card_threshold,
            self.cfg.mid_card_threshold,
        );

        let narrowest_width = if self.numeric_saw_any {
            Some(narrowest_numeric_width(
                self.numeric_min,
                self.numeric_max,
                self.numeric_negative_seen,
            ))
        } else {
            None
        };

        let wire = WireEncoding {
            kind: if mixed_kinds {
                // Surface the actual mix in `tag_histogram`; the
                // top-level `kind` for a mixed field has no single
                // truthful answer, so we pick the dominant one.
                *self.kinds.iter().max_by_key(|(_, c)| *c).map(|(k, _)| k).unwrap()
            } else {
                kind
            },
            storage_width: narrowest_width, // storage_width = narrowest at end of Pass 1
            narrowest_width,
            tag_histogram: histogram_from(self.tag_counts, self.present),
            mixed: mixed_kinds,
        };

        let measures = default_measures_for(semantic.as_ref(), &cardinality);

        FieldTemplate {
            wire_encoding: wire,
            semantic_type: semantic,
            semantic_confidence,
            cardinality_regime: cardinality,
            measures,
            probe_tallies,
        }
    }
}

/// Synthesis output of Pass 1, consumed by Pass 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldTemplate {
    /// Wire-encoding verdict.
    pub wire_encoding: WireEncoding,
    /// Semantic-type verdict. `None` means "field has only `Null`
    /// observations — no verdict possible."
    pub semantic_type: Option<SemanticType>,
    /// Confidence in the semantic verdict — fraction of non-null
    /// observations accepted by the chosen probe.
    pub semantic_confidence: f64,
    /// Cardinality regime.
    pub cardinality_regime: CardinalityRegime,
    /// Ordered list of measures Pass 2 should instantiate for this
    /// field.
    pub measures: Vec<MeasureKind>,
    /// Per-semantic-probe match tallies collected during Pass 1. The
    /// orchestrator's `ProbeAttemptReport` measure for Unstable fields
    /// renders these for operator diagnosis.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probe_tallies: Vec<ProbeTally>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn histogram_from(counts: IndexMap<String, u64>, total: u64) -> IndexMap<String, f64> {
    let mut out = IndexMap::with_capacity(counts.len());
    let total_f = total.max(1) as f64;
    for (k, c) in counts {
        out.insert(k, c as f64 / total_f);
    }
    out
}

/// Map a `(kind, numeric range)` tuple to an initial semantic
/// verdict. This is the cheap, encoding-only path used by Pass 1's
/// pilot. The richer textual probe table (§13.3.3) is run during
/// Pass 2 when the orchestrator wires up its parsers; the verdict
/// here is the floor.
fn classify_semantic(
    kind: WireEncodingKind,
    negative_seen: bool,
    nmin: f64,
    nmax: f64,
) -> (Option<SemanticType>, f64) {
    match kind {
        WireEncodingKind::Bool => (Some(SemanticType::Boolean), 1.0),
        WireEncodingKind::Numeric => {
            let bit_width_hint = narrowest_numeric_width(nmin, nmax, negative_seen);
            (
                Some(SemanticType::Number(NumberKind::Integer {
                    signed: negative_seen,
                    bit_width_hint,
                })),
                1.0,
            )
        }
        WireEncodingKind::Textual => (Some(SemanticType::FreeText), 0.5),
        WireEncodingKind::Bytes => (
            Some(SemanticType::Binary(super::types::BinaryKind::Opaque)),
            0.5,
        ),
        WireEncodingKind::Identifier => (
            Some(SemanticType::Identifier(super::types::IdentifierKind::Opaque)),
            1.0,
        ),
        WireEncodingKind::Collection => (Some(SemanticType::Unstable), 0.0),
        WireEncodingKind::Null => (None, 0.0),
    }
}

fn classify_cardinality(
    semantic: Option<&SemanticType>,
    distinct: &IndexMap<String, u64>,
    overflow: bool,
    non_null: u64,
    low: u32,
    mid: u32,
) -> CardinalityRegime {
    if matches!(semantic, Some(SemanticType::Unstable)) || non_null == 0 {
        return CardinalityRegime::Unknown;
    }
    if !overflow {
        let d = distinct.len() as u32;
        match d {
            0 => CardinalityRegime::Unknown,
            1 => CardinalityRegime::Constant,
            2 => CardinalityRegime::Binary,
            d if d <= low => CardinalityRegime::LowCard { exact_distinct: d },
            d if d <= mid => CardinalityRegime::MidCard {
                hll_estimate_at_pass1: d as f64,
            },
            _ => CardinalityRegime::HighCardOrUnique {
                uniqueness_ratio: distinct.len() as f64 / non_null.max(1) as f64,
            },
        }
    } else {
        CardinalityRegime::HighCardOrUnique {
            uniqueness_ratio: distinct.len() as f64 / non_null.max(1) as f64,
        }
    }
}

/// Narrowest signed/unsigned width that covers the observed range.
///
/// Signed types use their actual asymmetric ranges (e.g. i8 is
/// `-128..=127`), not symmetric absolute-value bounds. A bare
/// `|min|` check would reject the legal i8 value -128 because its
/// absolute value exceeds i8::MAX.
fn narrowest_numeric_width(min: f64, max: f64, negative: bool) -> NumericWidth {
    if min.is_nan() || max.is_nan() {
        return NumericWidth::I64;
    }
    if !negative {
        if max <= u8::MAX as f64 { return NumericWidth::I8 }
        if max <= u16::MAX as f64 { return NumericWidth::I16 }
        if max <= u32::MAX as f64 { return NumericWidth::I32 }
        return NumericWidth::I64
    }
    if min >= i8::MIN as f64 && max <= i8::MAX as f64 { return NumericWidth::I8 }
    if min >= i16::MIN as f64 && max <= i16::MAX as f64 { return NumericWidth::I16 }
    if min >= i32::MIN as f64 && max <= i32::MAX as f64 { return NumericWidth::I32 }
    NumericWidth::I64
}

/// Default measure selection. The full §13.6 policy table belongs
/// here; for this step we land enough to make the universal +
/// numeric trio reachable. Subsequent build-plan steps extend this
/// function as new measures land.
fn default_measures_for(
    semantic: Option<&SemanticType>,
    regime: &CardinalityRegime,
) -> Vec<MeasureKind> {
    let mut out = vec![
        MeasureKind::Presence,
        MeasureKind::TypeStability,
        MeasureKind::ReservoirSample,
    ];
    let unstable = matches!(semantic, Some(SemanticType::Unstable));
    if unstable {
        // Opaque-only set per §13.5.8. ReservoirSample remains the
        // primary diagnostic alongside the encoding histogram, a
        // simple length range, and the probe-attempt diagnostic
        // showing which semantic probes came closest.
        out.push(MeasureKind::WireEncodingHistogram);
        out.push(MeasureKind::ByteOrCharLengthRange);
        out.push(MeasureKind::ProbeAttempt);
        return out;
    }
    // Numeric measures
    if matches!(
        semantic,
        Some(SemanticType::Number(_)) | Some(SemanticType::Boolean)
    ) {
        out.push(MeasureKind::ExactExtrema);
        out.push(MeasureKind::ExactMoments);
        // QuantileSketch is the rank-aware companion to moments —
        // gives p50, IQR, p99 for predicate-selectivity work.
        out.push(MeasureKind::QuantileSketch);
        // Equi-width histogram derived from the observed range.
        out.push(MeasureKind::HistogramFromQuantiles);
        // Trend / monotonicity over observation order.
        out.push(MeasureKind::Monotonicity);
        // Integer-specific: bit-width / density / popcount and the
        // epoch-plausibility detector.
        if matches!(semantic, Some(SemanticType::Number(super::types::NumberKind::Integer { .. }))) {
            out.push(MeasureKind::BitWidth);
            out.push(MeasureKind::EpochPlausibility);
        }
        // Float-specific: "actually-integer" detector.
        if matches!(semantic, Some(SemanticType::Number(super::types::NumberKind::Floating))) {
            out.push(MeasureKind::DiscreteIndicator);
        }
    }
    // Temporal range applies to direct temporal types AND to any
    // numeric field where EpochPlausibility might trigger.
    if matches!(semantic, Some(SemanticType::Temporal(_))) {
        out.push(MeasureKind::TemporalRange);
    }
    // Bytes-specific: entropy classifier.
    if matches!(semantic, Some(SemanticType::Binary(_))) {
        out.push(MeasureKind::ByteEntropy);
    }
    // Textual / Identifier / Structured / FreeText fields: shape data
    // on the string representation (length, char-class) is useful
    // regardless of the semantic verdict.
    if matches!(
        semantic,
        Some(SemanticType::FreeText)
            | Some(SemanticType::Identifier(_))
            | Some(SemanticType::Structured(_))
            | Some(SemanticType::Temporal(_))
            | Some(SemanticType::Categorical(_))
    ) {
        out.push(MeasureKind::ExactLengthMoments);
        out.push(MeasureKind::LengthQuantiles);
        out.push(MeasureKind::CharClassMix);
        out.push(MeasureKind::PatternSkeleton);
    }
    // Trigram heavy hitters: only for text-shaped fields where
    // substring/`MATCHES` predicate synthesis makes sense.
    // Identifiers (UUID/ULID) and pure numerics are excluded — the
    // shape data wouldn't drive useful regex predicates and the
    // memory budget is better spent elsewhere.
    if matches!(
        semantic,
        Some(SemanticType::FreeText)
            | Some(SemanticType::Structured(_))
            | Some(SemanticType::Categorical(_))
    ) {
        out.push(MeasureKind::TrigramHeavyHitters);
    }
    // Labelset heavy hitters: only when Pass 1 committed a
    // labelset verdict. Operates on the *flattened* label space
    // (comma-separated chunks), distinct from the whole-string
    // distribution that `HeavyHitters` / `ExactFrequencyTable`
    // track.
    if matches!(
        semantic,
        Some(SemanticType::Categorical(super::types::CategoricalKind::Labelset)),
    ) {
        out.push(MeasureKind::LabelsetHeavyHitters);
    }
    // Cardinality measures gated by regime.
    match regime {
        CardinalityRegime::Constant | CardinalityRegime::Binary | CardinalityRegime::LowCard { .. } => {
            // Exact frequency table when we know the cardinality is
            // small. HeavyHitters is redundant here.
            out.push(MeasureKind::ExactFrequencyTable);
        }
        CardinalityRegime::MidCard { .. } => {
            // HLL gives the count, HeavyHitters surfaces the top-K.
            out.push(MeasureKind::HyperLogLog);
            out.push(MeasureKind::HeavyHitters);
        }
        CardinalityRegime::HighCardOrUnique { .. } => {
            // HLL is all that survives at high cardinality. Heavy
            // hitters would mostly chase noise; skip.
            out.push(MeasureKind::HyperLogLog);
        }
        CardinalityRegime::Unknown => {}
    }
    out
}

fn repr_for_distinct(value: &MValue) -> String {
    // Compact stable representation; bounded length to avoid blowing
    // up the bounded distinct tracker. Per-MValue display is enough
    // for the cardinality decision at end of Pass 1. Truncate on
    // the largest UTF-8 char boundary `<= 256` bytes so multi-byte
    // characters (em-dashes, ellipses, non-ASCII text) don't trigger
    // a slice-boundary panic.
    let s = format!("{:?}", value);
    if s.len() <= 256 { return s; }
    let mut cut = 256;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s[..cut].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::commands::survey::types::{NumberKind, NumericWidth};

    fn make_probe() -> ExplorationProbe {
        ExplorationProbe::new(TemplateConfig::default())
    }

    #[test]
    fn empty_probe_is_empty() {
        let p = make_probe();
        assert!(p.is_empty());
        let t = p.finalize("x");
        assert!(t.semantic_type.is_none());
        assert_eq!(t.cardinality_regime, CardinalityRegime::Unknown);
        assert!(t.measures.contains(&MeasureKind::Presence));
    }

    #[test]
    fn pure_integer_field_classifies_as_number_integer() {
        let mut p = make_probe();
        for v in 0..50 {
            p.observe(&MValue::Int32(v));
        }
        let t = p.finalize("count");
        let st = t.semantic_type.expect("verdict expected");
        match st {
            SemanticType::Number(NumberKind::Integer { signed, bit_width_hint }) => {
                assert!(!signed, "no negatives observed");
                // Range 0..50 fits in I8.
                assert_eq!(bit_width_hint, NumericWidth::I8);
            }
            other => panic!("unexpected verdict: {:?}", other),
        }
        // 50 distinct integers, under low_card_threshold=64 →
        // LowCard regime.
        match t.cardinality_regime {
            CardinalityRegime::LowCard { exact_distinct } => assert_eq!(exact_distinct, 50),
            other => panic!("unexpected regime: {:?}", other),
        }
        // ExactMoments + ExactExtrema should appear.
        assert!(t.measures.contains(&MeasureKind::ExactMoments));
        assert!(t.measures.contains(&MeasureKind::ExactExtrema));
    }

    #[test]
    fn negative_integers_widen_storage() {
        let mut p = make_probe();
        for v in [-200i32, -1, 0, 1, 200] {
            p.observe(&MValue::Int32(v));
        }
        let t = p.finalize("delta");
        if let Some(SemanticType::Number(NumberKind::Integer { signed, bit_width_hint })) =
            t.semantic_type
        {
            assert!(signed, "negatives observed");
            // Range fits in I16 (200 > i8::MAX = 127).
            assert_eq!(bit_width_hint, NumericWidth::I16);
        } else {
            panic!("expected Integer verdict");
        }
    }

    #[test]
    fn mixed_kinds_collapse_to_unstable() {
        let mut p = make_probe();
        p.observe(&MValue::Int(1));
        p.observe(&MValue::Text("oops".into()));
        let t = p.finalize("messy");
        assert_eq!(t.semantic_type, Some(SemanticType::Unstable));
        assert!(t.wire_encoding.mixed);
        assert_eq!(t.cardinality_regime, CardinalityRegime::Unknown);
    }

    #[test]
    fn constant_stream_classified_as_constant_regime() {
        let mut p = make_probe();
        for _ in 0..1_000 {
            p.observe(&MValue::Int(42));
        }
        let t = p.finalize("constant_field");
        assert_eq!(t.cardinality_regime, CardinalityRegime::Constant);
    }

    #[test]
    fn binary_field_detected() {
        let mut p = make_probe();
        for i in 0..1_000 {
            p.observe(&MValue::Int(i % 2));
        }
        let t = p.finalize("flag");
        assert_eq!(t.cardinality_regime, CardinalityRegime::Binary);
    }

    #[test]
    fn high_cardinality_above_thresholds() {
        let cfg = TemplateConfig {
            low_card_threshold: 4,
            mid_card_threshold: 8,
            distinct_cap: 8,
            ..TemplateConfig::default()
        };
        let mut p = ExplorationProbe::new(cfg);
        for i in 0..1_000 {
            p.observe(&MValue::Text(format!("v-{}", i)));
        }
        let t = p.finalize("hi");
        match t.cardinality_regime {
            CardinalityRegime::HighCardOrUnique { .. } => {}
            other => panic!("expected HighCardOrUnique, got {:?}", other),
        }
    }

    #[test]
    fn always_null_field_stays_unverdicted() {
        let mut p = make_probe();
        for _ in 0..10 {
            p.observe(&MValue::Null);
        }
        let t = p.finalize("nullable");
        assert!(t.semantic_type.is_none());
        // Only Presence runs.
        assert_eq!(t.measures, vec![MeasureKind::Presence]);
    }

    #[test]
    fn narrowest_width_unsigned_thresholds() {
        assert_eq!(narrowest_numeric_width(0.0, 255.0, false), NumericWidth::I8);
        assert_eq!(narrowest_numeric_width(0.0, 256.0, false), NumericWidth::I16);
        assert_eq!(narrowest_numeric_width(0.0, 65_535.0, false), NumericWidth::I16);
        assert_eq!(narrowest_numeric_width(0.0, 65_536.0, false), NumericWidth::I32);
    }

    #[test]
    fn narrowest_width_signed_thresholds() {
        assert_eq!(narrowest_numeric_width(-128.0, 127.0, true), NumericWidth::I8);
        assert_eq!(narrowest_numeric_width(-129.0, 127.0, true), NumericWidth::I16);
        assert_eq!(narrowest_numeric_width(-32_768.0, 32_767.0, true), NumericWidth::I16);
        assert_eq!(narrowest_numeric_width(-32_769.0, 0.0, true), NumericWidth::I32);
    }
}
