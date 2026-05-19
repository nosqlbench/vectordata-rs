# 13. Incremental Metadata Survey

**Status:** LANDED. All 11 build-plan steps from §13.13 are
implemented under `veks-pipeline/src/pipeline/commands/survey/`. The
single-pass legacy survey command was removed; `analyze survey`
now refers exclusively to the two-pass type-driven orchestrator
specified by this document. Downstream consumers
(`generate predicates --mode=survey`, the bootstrap wizard's
synthesis-mode-2 path) consume the §13.8 JSON shape directly.

This document specifies a richer, multi-pass survey that builds a
typed, distributional, and relational model of a metadata corpus
record-by-record. The goal is to give downstream synthesis (predicate
generation, oracle partition design, dataset documentation) a model
expressive enough to drive selectivity-controlled compound predicates
and statistically realistic synthetic data — not just min/max bounds.

The proposal deliberately keeps memory bounded: every accumulator is
either O(1) per field or a sublinear streaming sketch with a declared
budget.

### Scope — ANode → MNode only

This version surveys **ANode-encoded slab files**, decoded as MNodes
(`veks-core::formats::anode::decode` → `ANode::MNode(MNode { fields })`).
Each MNode is one record; each `(name, MValue)` pair in `MNode.fields`
is one field observation. Non-MNode ANode variants (e.g. `PNode`) are
counted as `non_mnode_count` and skipped, matching today's
`survey_slab` behavior.

Out of scope:

- `.ivec` / `.ivvec` / `.i32vvec` metadata input (the legacy
  `survey_ivec` path in `slab.rs:2251`). Metadata in the new survey
  is ANode → MNode only. The ivec input path remains reachable via
  `analyze legacy-survey` until that command is removed.
- Parquet, NPY, HDF5 metadata, JSON-Lines, CSV — none of these
  decode to MNode, so they would require a separate adapter layer
  before they could feed this pipeline.
- Synthetic in-memory MNode streams that don't originate from a slab
  reader. (Tests will mock the reader, but the user-facing command
  always reads a slab.)

The shape of the survey model — WireEncoding (§13.3.1), SemanticType
(§13.3.2), per-MValue probes (§13.3.3) — is deliberately built on
the MValue tag set so the orchestrator doesn't accumulate
abstractions for input formats it cannot actually consume. Adding
other input formats later means writing an MNode-shaped adapter (or
extending the WireEncoding lattice), not retrofitting the survey
state machine.

---

## 13.1 Motivation and Limits of the Current Survey

The existing `analyze survey` accumulates, per field name:

- type-tag histogram
- `min`, `max`, `sum`, `count` (numeric)
- `min/max/sum/count` of string length
- `min/max/sum/count` of bytes length
- a bounded distinct-value map (default cap 20, gen_predicates passes 100)
- an overflow flag once the distinct cap is hit

Everything is decided in a single pass and treated as univariate. As a
result the survey cannot:

- distinguish a uniform integer field from a heavily skewed one
- expose quantiles, percentiles, IQR, mode, or any sketch of the CDF
- detect monotonicity, repetition runs, or near-uniqueness
- recognize structural patterns in strings (UUID-like, date-like,
  fixed-width-id, …)
- estimate cardinality past the distinct cap
- describe any cross-field relationship
- adapt its measurement plan to what each field actually is

The new design fixes each of these in a structured way.

---

## 13.2 Core Model: Field Exploration State Machine

Every field name maps to a **FieldExplorationState**, a lattice whose
root is `Unknown`. The state advances monotonically as records are
observed. **Type determination lives near the root.** Each subsequent
level adds measure structure specific to what was learned.

```text
                                   Unknown
                                      │
                                      │    first non-null observation
                                      ▼
                                TypeHypothesis
                                      │
       ┌──────────┬───────────────────┼───────────────────┬──────────────┐
       │          │                   │                   │              │
       │          │                   │                   │              │
       ▼          ▼                   ▼                   ▼              ▼
      Bool     Numeric             Textual              Bytes        Unstable
                  │                   │                                  │
                  │                   │                                  │
        ┌─────────┼─────────┐         ├──── Ascii                        ▼
        │         │         │         │
        │         │         │         ├──── EnumStr                opaque-only
        ▼         ▼         ▼         │                              (§13.5.8)
     Integer    Float     Millis      ├──── Text
                                      │
                                      ├──── Date
                                      │
                                      ├──── Time
                                      │
                                      └──── DateTime


       (Bool · Numeric · Textual · Bytes branches all converge here ↓)


                              CardinalityRegime
                                      +
                              DistributionShape
```

A field that is observed but never carries a non-null value stays in
`Unknown` for the duration of the survey — it has no TypeHypothesis
to make. The report records the presence/null counts via
`PresenceMeasure` (the only measure that runs against `Unknown`) so
always-null fields are still visible to the operator without
requiring a dedicated `Null` type at the state-machine root.

Transitions:

- **First observation** picks the leaf based on the `MValue` tag.
- **Type-widening** observation (e.g. a record with `Int` after we had
  only seen `Int32`) promotes the field to a wider numeric type, or
  collapses to `Unstable` when no covering type exists.
- **Type-mixing** observation (e.g. `Int` after `Text`) promotes
  immediately to `Unstable`. An unstable field is treated as
  **opaque** for the rest of the survey: only encoding-agnostic
  measures run on it (§13.5.8). The survey does not try to be clever
  about per-tag sub-profiles — heterogeneous semantics in one field
  is a data-quality problem to surface, not a thing to model.
- **Cardinality regime** is decided at the end of Pass 1 based on
  distinct-tracker state and a small reservoir, not greedily.

The state machine is the contract between Pass 1 (template discovery)
and Pass 2 (full profiling). Pass 1 emits, for every field, a
**FieldTemplate** describing exactly which measures Pass 2 should
instantiate.

---

## 13.3 Two-Layer Type Model: Encoding vs Semantic

A field's "type" has two **independent** dimensions, and the survey
tracks both:

- **`WireEncoding`** — how the bytes arrived, in terms of the storage
  `MValue` tag. Truthful, no interpretation. The same record can have
  multiple wire encodings for the same field across runs (e.g. an
  importer might emit `Text` one day, `Int` the next).
- **`SemanticType`** — what the value *means*. Determined by Pass 1
  through cheap parsing probes. `"2143"` (wire `Text`) and `2143`
  (wire `Int`) share the same `SemanticType::Number(Integer)`; an
  `Int` holding a Unix-millis-since-epoch and a `Text "2024-01-15"`
  share `SemanticType::Temporal(Date|DateTime)`; a fixed-width
  hex-string identifier shares `SemanticType::Identifier(HashLike)`
  with a `Bytes` field whose values are 16 raw bytes.

Measure selection (§13.6) is keyed on `SemanticType` so that a
correlation between a string-of-digits column and an integer column
"just works". Encoding-specific reporting still happens — operators
need to know the field "*reads* like an integer but is *stored* as a
4-byte ASCII string", because that affects storage cost, sorting,
and synthesis output format.

### 13.3.1 WireEncoding lattice

```text
WireEncoding
│
├── Numeric
│   │
│   ├── Int8  /  Int16  /  Int32  /  Int64        (signed; UInt* mirror if all ≥ 0)
│   │
│   ├── Float16  /  Float32  /  Float64
│   │
│   └── Millis                                    (i64 epoch-millis convention)
│
├── Textual
│   │
│   ├── Ascii  /  Text  /  EnumStr
│   │
│   └── Date  /  Time  /  DateTime                (still strings on the wire)
│
├── Bytes
│
├── Bool
│
├── Null
│
└── MixedEncodings { tag_histogram }              (a field's encoding genuinely
                                                   varies across records)
```

Width-narrowing (Int64 → Int32, etc.) is reported as
`WireEncoding::Numeric { storage_width, narrowest_width }` so callers
can choose more compact storage on derived datasets.

### 13.3.2 SemanticType lattice

```text
SemanticType
├── Number
│   ├── Integer    { signed, bit_width_hint }
│   ├── Decimal    { precision_hint, scale_hint }
│   └── Floating
├── Boolean                                  ("true"/"false", 0/1, "yes"/"no")
├── Temporal
│   ├── Date                                 (ISO 8601 day, days-since-epoch, …)
│   ├── Time
│   ├── DateTime    { has_timezone }
│   ├── Timestamp   { granularity: sec|ms|µs|ns }
│   └── Duration
├── Identifier
│   ├── UUID
│   ├── Sequential                           (densely packed integer ID)
│   ├── HashLike                             (fixed-width hex/base64, uniform)
│   ├── Composite   { prefix, body, suffix } (e.g. "USR_00123")
│   └── Opaque
├── Categorical
│   ├── Enum                                 (closed small set)
│   └── OpenSet                              (bounded mid-card)
├── Structured
│   ├── Email
│   ├── URL
│   ├── IPAddress   { v4|v6 }
│   ├── PhoneNumber { country_hint }
│   ├── Geocode                              ("lat,lng" or similar)
│   ├── Currency    { currency_code }
│   └── Json                                 (recurse: sub-survey)
├── FreeText
├── Binary
│   ├── Compressed                           (high entropy)
│   ├── MagicTyped  { magic: "PNG" | "gzip" | … }
│   └── Opaque
└── Unstable                                 (no semantic verdict — only
                                              opaque-value measures run;
                                              §13.5.8)
```

Always-null fields don't get a `SemanticType` at all — they stay in
the state-machine's `Unknown` root and surface in the report through
`PresenceMeasure` counts. `MValue::Null` as an observation simply
increments the null counter; it never transitions the field forward.

A field's full classification is the triple
`(WireEncoding, SemanticType, Confidence)`. Confidence is the
fraction of non-null observations that the chosen semantic probe
accepted (default commit threshold: 0.95). Below threshold the field
classifies as `SemanticType::Unstable` and only the opaque-value
measure set (§13.5.8) runs. The top-N candidate probes and their
match rates are still retained in the report as diagnostic
information, but they do not unlock semantic-typed measures —
unstable means "we don't trust ourselves to interpret this field".

### 13.3.3 Semantic probes

A `SemanticProbe` returns a parsed canonical form when it accepts a
value. Probes are tried in **cost order** during Pass 1: cheap
structural checks first, expensive parsers only on values that
survive the cheap filter. The default suite:

| Probe                       | Inputs          | Cost   | Accepts                                          | Parsed form |
|-----------------------------|-----------------|--------|--------------------------------------------------|-------------|
| `IntegerLiteralProbe`       | Textual         | O(len) | `^-?\d+$` (with width fit)                       | `i64` |
| `DecimalLiteralProbe`       | Textual         | O(len) | `^-?\d+(\.\d+)?$`                                | `(i64 mantissa, scale)` |
| `FloatLiteralProbe`         | Textual         | O(len) | `^-?\d+(\.\d+)?([eE][+-]?\d+)?$`                 | `f64` |
| `BooleanLiteralProbe`       | Textual,Numeric | O(1)   | true/false, t/f, yes/no, y/n, 1/0                | `bool` |
| `UuidProbe`                 | Textual,Bytes   | O(len) | RFC 4122 form (string) or 16-byte length         | `Uuid` |
| `EmailProbe`                | Textual         | O(len) | simple "_@_._" with length & class checks        | `(local, domain)` |
| `UrlProbe`                  | Textual         | O(len) | `^[a-z]+://…`                                    | parsed URL |
| `Ipv4Probe` / `Ipv6Probe`   | Textual,Bytes   | O(len) | dotted-quad / colon-hex form, or 4/16 bytes      | `IpAddr` |
| `Iso8601DateProbe`          | Textual         | O(len) | `YYYY-MM-DD`                                     | `Date` |
| `Iso8601DateTimeProbe`      | Textual         | O(len) | full / partial ISO 8601                          | `DateTime` |
| `EpochSecondsPlausibility`  | Numeric         | O(1)   | i64 in `[946684800, 4102444800)`                 | `Timestamp(sec)` |
| `EpochMillisPlausibility`   | Numeric         | O(1)   | i64 in `[946684800e3, 4102444800e3)`             | `Timestamp(ms)` |
| `CurrencyProbe`             | Textual         | O(len) | `^[\$€£¥]?\d+(\.\d{1,2})?$`                      | `(currency, amount)` |
| `GeocodeProbe`              | Textual         | O(len) | `^-?\d+\.\d+,\s*-?\d+\.\d+$`                     | `(lat, lng)` |
| `PhoneNumberProbe`          | Textual         | O(len) | E.164 / formatted national patterns              | `Phone` |
| `HexFixedWidthProbe`        | Textual         | O(len) | `^[0-9a-fA-F]{N}$` for stable N                  | bytes |
| `Base64FixedWidthProbe`     | Textual         | O(len) | well-formed b64 with stable length               | bytes |
| `CompositeIdentifierProbe`  | Textual         | O(len) | `^[A-Z]+_\d+$` and similar; reuses PatternSkeleton voting | `(prefix, body)` |
| `JsonProbe`                 | Textual,Bytes   | O(len) | starts with `{` or `[`, parses                   | recursive |
| `MagicByteProbe`            | Bytes           | O(1)   | known magic-number prefixes                      | format tag |

Each probe is a small trait impl that costs ~zero when the wire
encoding makes it impossible (e.g. `IntegerLiteralProbe` skips
Numeric and Bytes inputs immediately).

### 13.3.4 How (encoding, semantic, confidence) drive everything else

- **Measure selection** (§13.6) keys on `SemanticType` so a textual
  Integer field gets the same numeric measure suite as a wire
  Integer field — operating on the parsed `i64`, not the raw string.
- **Encoding-specific reporting** stays in the report: textual
  measures (length, char-class mix, pattern skeleton) still apply to
  string-encoded numerics and identifiers because the encoding shape
  matters for storage and synthesis decisions.
- **Cross-field correlation** (§13.7) aligns by `SemanticType`. A
  Pearson r between a `Text "2143"` field and a `Int 2143` field is
  meaningful and computable.
- **Synthesis** (predicate generation, gen metadata) reads
  `(SemanticType, regime)` to choose value spaces and `WireEncoding`
  to choose `Comparand` variants and storage formats. The two-layer
  split fixes the current conflation that forces string-encoded
  numerics into string comparands.

---

## 13.4 Two-Pass Orchestration

### Input contract

Both passes read records through the same slab pipeline used by today's
survey:

```text
SlabReader::open(path)
   → page_entries[i] → read_data_page(entry)
       → page.record_count() / page.get_record(i) → raw bytes
           → anode::decode(bytes) → ANode::MNode(node)
               → node.fields: IndexMap<String, MValue>
                   → orchestrator dispatches each (name, MValue)
                     to the field's current FieldExplorationState
```

The orchestrator only ever sees `MNode.fields`. `ANode::PNode` and
decode errors increment `non_mnode_count` / `decode_errors` (same
counters today's survey reports) and the record is skipped. No
adapter layer for other input formats is in scope.

### Pass 1 — Discovery (cheap, bounded)

Reads every sampled record once. For every observed field, advances
its `ExplorationProbe`:

```rust
struct ExplorationProbe {
    presence: PresenceCounter,           // count, null_count, absent_in_record
    tag_histogram: TagHistogram,
    bounded_distinct: BoundedDistinct,   // cap K_template; on overflow, retains freq for retained values
    numeric_pilot: Option<MomentsPilot>, // m1..m4, min, max
    strlen_pilot: Option<LenPilot>,
    byteslen_pilot: Option<LenPilot>,
    monotonic_check: MonotonicTracker,   // ascending/descending/violation counts
    run_length_pilot: RunLengthPilot,    // current run, max run, mean run
    reservoir: Reservoir<MValue>,        // size R_template, e.g. 256
    pattern_probe: PatternProbe,         // for textual: char-class skeleton voting
}
```

The pilot is intentionally lossy — its purpose is to drive the
*planning* decision in §13.6, not to publish.

Pass 1 also runs a **co-presence pair counter** for every observed
field pair (bounded by `max_pairs`, default 256² = 65 536; with
heavy-hitter pruning beyond that). The co-presence matrix is itself a
useful output and seeds Pass 2's cross-field analyzer eligibility.

### Pass 1 outputs

1. Per-field `FieldTemplate` (§13.6).
2. Cross-field measurement plan: list of pair-analyzers to run in Pass 2.
3. Total record count, sampling regime used, sampling-bias estimate.

### Pass 2 — Profiling (rich, type-targeted)

Re-reads from the first record. For every field, instantiates the
measure suite declared by its `FieldTemplate`. Each measure
implements:

```rust
trait Measure {
    fn observe(&mut self, value: &MValue, ctx: &MeasureCtx);
    fn finalize(self: Box<Self>) -> MeasureReport;
    fn memory_budget(&self) -> usize;
    fn kind(&self) -> MeasureKind;
}
```

`MeasureCtx` carries the current record index (for position-correlated
measures) and the field's classified type (so a measure written for
`Numeric` can assume the value is castable).

Cross-field analyzers receive paired observations directly from the
orchestrator, which dispatches to them after the per-field observers
fire.

### Pass 2 outputs

A structured JSON report (`survey.json`) — see §13.8.

### When sampling is enabled

Both passes use the **same sample set** (deterministic page indices,
as today). The two-pass cost is therefore ~2× a single-pass survey
on the sample, not on the full corpus. For corpora small enough to
survey exhaustively (`samples >= total_records`), the cost is
exactly 2× a full scan. Two passes are the survey shape; there is
no single-pass mode.

---

## 13.5 Per-Type Measure Catalog

Each measure here is independent and addressable by `MeasureKind`. A
`FieldTemplate` is a `Vec<MeasureKind>` plus type and regime metadata.

### 13.5.1 Universal (all types)

- **PresenceMeasure**: present/absent/null counts, presence rate.
- **TypeStabilityMeasure**: verifies Pass 1's type verdict; emits
  surprise count if Pass 2 sees a tag absent from Pass 1.
- **ReservoirSample**: bounded reservoir (default size 1024) of
  representative values for the report's "examples" section.

### 13.5.2 Numeric (Integer and Float)

- **ExactMoments**: count, sum, sum², sum³, sum⁴ → mean, variance,
  stddev, skewness, kurtosis. All in f64.
- **ExactExtrema**: min, max (exact, free with moments).
- **QuantileSketch**: KLL sketch (`k=200`, ~4 KB) → percentiles,
  median, IQR, full CDF approximation with ε≈1%.
- **HistogramFromQuantiles**: derives equi-depth and equi-width
  histograms (configurable bin count, default 32) from the quantile
  sketch.
- **DiscreteIndicator**: ratio of integer-valued observations to
  total; for `Float` fields, flags "actually-integer" cases.
- **MonotonicityReport**: ascending/descending/violation counts, plus
  Mann-Kendall trend statistic τ over the reservoir.
- **RepetitionReport**: run-length distribution (mean, max,
  count-of-runs-of-length≥2).
- **DistributionFitHints**: chi-square goodness-of-fit against
  {uniform, normal (moments-derived), exponential, log-normal,
  zipfian}. Reports the best-fit family with p-value, plus residuals
  summary.
- **BitWidthReport** (Integer only): minimum signed/unsigned width
  that covers the observed range; flags "looks like packed enum"
  (small contiguous range), "looks like ID" (large sparse range),
  "looks like hash" (uniform across full width).

### 13.5.3 Cardinality and Frequency (all types)

- **HyperLogLog**: cardinality estimate (~2% accuracy, 4 KB sketch).
  Necessary when distinct count exceeds Pass 1's bounded tracker.
- **HeavyHitters**: Misra-Gries with `k = top_k` (default 64). Exact
  for low cardinality; converges to top-k for high cardinality.
- **ExactFrequencyTable**: enabled only when Pass 1 verified
  cardinality ≤ `low_card_threshold` (default 64). Provides exact
  mode, entropy, Gini, full distinct count.

### 13.5.4 Temporal (Millis, parseable Date/Time/DateTime)

- **TemporalRange**: parsed min/max as ISO 8601, plus inferred
  granularity (seconds / millis / microseconds / nanos / days).
- **CalendarHistograms**: hour-of-day, day-of-week, month-of-year,
  year-bucket. Equi-width bins.
- **GapDistribution**: deltas between consecutive sorted values →
  median gap, p99 gap; flags "looks periodic at period P" if the
  autocorrelation of gaps is high.
- **EpochPlausibility**: for `Int` fields, evaluates "is this a
  Unix-seconds-since-1970 plausible timestamp" and "is it Unix
  millis". Reports confidence.

### 13.5.5 Textual (Ascii / Text / EnumStr / Date / Time / DateTime)

- **ExactLengthMoments**: same shape as ExactMoments, applied to
  `len(s)` (UTF-8 byte length and char-count are both reported).
- **LengthQuantiles**: KLL over lengths.
- **CharClassMix**: per-record, fraction of {alpha, digit, punct,
  whitespace, other}; reports the cross-record mean/stddev.
- **PatternSkeleton**: collapses each value to a regex-like skeleton
  (`A` for letter, `9` for digit, `.` for punct, etc.) and runs
  HeavyHitters over the skeletons. Top-K skeletons surface as
  "patterns this field tends to follow".
- **PrefixDistribution**: longest-common-prefix length distribution
  over the reservoir; flags ID-like prefixes (e.g. all values start
  with `usr_`).
- **TokenStats**: split-on-whitespace token count distribution; useful
  to distinguish single-token codes from sentences/paragraphs.
- **StructuredFormatDetectors**: independent yes/no probes for UUID,
  hex-string, base64, ISO date, ISO datetime, URL, email,
  semver-like. Reports per-detector match rate.
- **CardinalityFamily** (same as §13.5.3): HLL, HeavyHitters,
  optional ExactFrequencyTable.

### 13.5.6 Bytes

- **ExactLengthMoments** + **LengthQuantiles** over `bytes.len()`.
- **ByteEntropy**: Shannon entropy over the byte-value histogram;
  flags "high entropy → likely compressed/encrypted/random",
  "low entropy → likely text or structured".
- **MagicByteVotes**: HeavyHitters over the first N bytes (default 8)
  to surface common magic numbers (PNG, gzip, parquet, …).

### 13.5.7 Bool

- True/false counts, ratio.

### 13.5.8 Unstable (opaque-value measures only)

When Pass 1 cannot commit to a `SemanticType` (mixed wire encodings,
or no probe clears the confidence threshold), the field is treated
as **opaque**: a sequence of values whose internal structure the
survey refuses to interpret. Only encoding-agnostic measures run:

- **PresenceMeasure** — present / absent / null counts.
- **WireEncodingHistogram** — the actual `MValue` tag mix that made
  the field unstable in the first place. This is the most important
  output for unstable fields, because it tells the operator *why*
  the survey gave up.
- **ProbeAttemptReport** — for each candidate semantic probe, the
  match rate and the most common rejection reason. Diagnostic only;
  no semantic interpretation is committed to the report.
- **ReservoirSample** — bounded reservoir of raw values so the
  operator can eyeball the field and decide whether to clean it up.
- **ByteOrCharLengthRange** — encoding-agnostic length: bytes-length
  for binary, char-count for textual, "n/a" otherwise. Min / max /
  count only; no quantiles, no distributional fit.

Explicitly **not** run on unstable fields: distinct tracking,
cardinality estimation, frequency tables, quantile sketches,
correlations, heavy-hitter analysis, pattern skeletons. These all
require a semantic interpretation, and committing to one for an
unstable field would propagate noise into the report.

---

## 13.6 Template Synthesis (End of Pass 1)

For each field, Pass 1 produces:

```rust
struct FieldTemplate {
    wire_encoding: WireEncoding,                  // observed encoding(s)
    semantic_type: SemanticType,                  // Pass-1 verdict
    semantic_confidence: f64,                     // fraction accepted by chosen probe
    storage_width_hint: Option<NumericWidth>,     // for Integer/Float semantics
    cardinality_regime: CardinalityRegime,        // see below
    monotonicity_hint: MonotonicityHint,
    measures: Vec<MeasureKind>,
    sketch_budgets: SketchBudgets,                // per-sketch sizing
}

enum CardinalityRegime {
    Constant,                                     // 1 distinct
    Binary,                                       // 2 distinct
    LowCard { exact_distinct: u32 },              // ≤ low_card_threshold, fully enumerable
    MidCard { hll_estimate_at_pass1: f64 },       // between thresholds
    HighCardOrUnique { uniqueness_ratio: f64 },   // distinct/count near 1.0
}
```

**Default measure-selection policy** (configurable):

| Regime / Type combination                        | Measures activated |
|--------------------------------------------------|--------------------|
| Constant or Binary, any type                     | Universal + ExactFrequencyTable |
| LowCard, any type                                | Universal + ExactFrequencyTable + HeavyHitters |
| MidCard, Numeric                                 | Universal + ExactMoments + ExactExtrema + QuantileSketch + HistogramFromQuantiles + HyperLogLog + HeavyHitters + DistributionFitHints + MonotonicityReport + RepetitionReport (+ BitWidthReport for Integer) |
| HighCard, Numeric                                | as MidCard + DiscreteIndicator |
| MidCard / HighCard, Textual                      | Universal + ExactLengthMoments + LengthQuantiles + CharClassMix + PatternSkeleton + PrefixDistribution + TokenStats + StructuredFormatDetectors + HyperLogLog + HeavyHitters |
| Temporal-promoted Integer                        | as Numeric + TemporalRange + CalendarHistograms + GapDistribution + EpochPlausibility |
| Bytes                                            | Universal + ExactLengthMoments + LengthQuantiles + ByteEntropy + MagicByteVotes + HyperLogLog |
| Unstable, any encoding                           | §13.5.8 opaque-only set: Presence + WireEncodingHistogram + ProbeAttemptReport + Reservoir + ByteOrCharLengthRange. No other measures run. |

Operators override via:

```sh
analyze survey \
  --measures FieldName=ExactMoments,QuantileSketch,Histogram \
  --measures TagField=ExactFrequencyTable
```

---

## 13.7 Cross-Field Analysis

Pass 1 produces an **eligibility list** for pair analyzers, capped by
`max_pair_analyses` (default 1024) and prioritized by Pass 1's
co-presence and uniqueness metrics:

| Pair regime                                  | Analyzer                                                |
|----------------------------------------------|---------------------------------------------------------|
| Numeric × Numeric                            | Pearson r (exact) + Spearman ρ (rank via QuantileSketch) + linear-fit residuals summary |
| LowCard × LowCard                            | Exact contingency table → χ², Cramér's V, mutual information |
| LowCard × Numeric                            | One-way ANOVA / η² (variance-explained), per-category quantile summary |
| Numeric × RecordIndex                        | Position trend: Pearson r vs index, Mann-Kendall τ      |
| Any × Any                                    | Co-presence: P(A present │ B present), P(A null │ B present), Jaccard over presence sets |
| (no row for Unstable)                        | Unstable fields are excluded from cross-field analysis. Including them would require committing to a semantic interpretation the survey refused to make. |
| LowCard × Any (functional-dep probe)         | Approximate functional dependency: P(unique B per group of A); reports candidate FDs |

Cross-field measures share the streaming pipeline — they observe
pairs as the orchestrator emits them, then publish in §13.8's report.

### Output: correlation matrices

The report includes pre-rendered matrices for the common cases:

- `numeric_correlation`: square matrix of Pearson r (and a parallel
  Spearman ρ matrix) over all numeric × numeric eligible pairs.
- `categorical_association`: square matrix of Cramér's V over all
  LowCard × LowCard eligible pairs.
- `copresence`: square matrix of co-presence rates over **all** field
  pairs that survived eligibility pruning.

Cells outside the eligibility set are omitted (sparse JSON) rather
than padded with NaN, to keep the report scannable for high-arity
schemas.

---

## 13.8 Output Schema (`survey.json`)

Single canonical envelope. `schema_version` is a forward-evolution
marker for future refinements, not a compatibility dispatcher:
there is no prior version of this schema to read.

```jsonc
{
  "schema_version": 1,
  "produced_by": "veks-pipeline analyze survey",
  "source": {
    "path": "metadata_content.slab",
    "format": "slab",
    "total_records": 10000000,
    "sampled_records": 100000,
    "sampling": { "mode": "page_stride", "page_count": 1024 }
  },
  "fields": {
    "field_0": {
      "wire_encoding": { "kind": "Numeric", "storage_width": "I32", "narrowest_width": "I8" },
      "semantic_type": { "kind": "Number", "subkind": "Integer", "signed": false, "bit_width_hint": 8 },
      "semantic_confidence": 1.00,
      "cardinality_regime": { "kind": "LowCard", "exact_distinct": 7 },
      "monotonicity_hint": "Random",
      "presence":   { "present": 100000, "null": 12, "absent_in_record": 0 },
      "measures": {
        "ExactMoments":        { "mean": …, "stddev": …, "skewness": …, "kurtosis": … },
        "ExactExtrema":        { "min": 0, "max": 6 },
        "ExactFrequencyTable": { "0": 14200, "1": 14210, …, "entropy": 2.807, "gini": 0.857 },
        "ReservoirSample":     [0,3,1,4,1,5,9,2,6,5,3,5]
      }
    },
    "string_encoded_int": {
      "wire_encoding": { "kind": "Textual", "subkind": "Text" },
      "semantic_type": { "kind": "Number", "subkind": "Integer", "signed": true, "bit_width_hint": 32 },
      "semantic_confidence": 0.998,
      "cardinality_regime": { "kind": "MidCard", "hll_estimate_at_pass1": 12480.0 },
      "presence": { … },
      "measures": {
        "ExactMoments":        { "mean": 1024.4, "stddev": 612.1, … },
        "QuantileSketch":      { "p50": 1018, "p90": 1840, "p99": 2110 },
        "ExactLengthMoments":  { "mean": 4.1, "min": 1, "max": 6 },
        "PatternSkeleton":     { "top": [["9+", 0.998], ["-9+", 0.002]] }
      }
    },
    "user_email": {
      "wire_encoding": { "kind": "Textual", "subkind": "Text" },
      "semantic_type": { "kind": "Structured", "subkind": "Email" },
      "semantic_confidence": 1.00,
      "cardinality_regime": { "kind": "HighCardOrUnique", "uniqueness_ratio": 0.992 },
      "presence": { … },
      "measures": {
        "ExactLengthMoments":          { "mean": 24.1, "stddev": 6.3, "min": 9, "max": 64 },
        "LengthQuantiles":             { "p50": 23, "p90": 32, "p99": 48 },
        "CharClassMix":                { "alpha": 0.78, "digit": 0.08, "punct": 0.14 },
        "PatternSkeleton":             { "top": [["A+@A+.A+", 0.98], ["A+.A+@A+.A+", 0.02]] },
        "StructuredFormatDetectors":   { "email": 1.00, "uuid": 0.00, "url": 0.00 },
        "HyperLogLog":                 { "cardinality_estimate": 99124, "stderr": 0.012 },
        "HeavyHitters":                [ … top-64 with frequencies … ]
      }
    },
    "tags": {
      "wire_encoding": { "kind": "MixedEncodings", "tag_histogram": { "Text": 0.7, "Int": 0.3 } },
      "semantic_type": { "kind": "Unstable" },
      "semantic_confidence": 0.0,
      "cardinality_regime": { "kind": "Unknown" },
      "presence": { … },
      "measures": {
        "WireEncodingHistogram": { "Text": 0.7, "Int": 0.3 },
        "ProbeAttemptReport":    {
          "IntegerLiteralProbe": { "match_rate": 0.30, "top_reject_reason": "not_numeric_encoding" },
          "EmailProbe":          { "match_rate": 0.00, "top_reject_reason": "no_at_sign" }
        },
        "ByteOrCharLengthRange": { "min": 1, "max": 36, "count": 100000 },
        "ReservoirSample":       ["tag_a", 17, "tag_b", 42, …]
      }
    }
  },
  "cross_field": {
    "numeric_correlation": {
      "fields": ["field_0","field_1","field_2"],
      "pearson":  [[1.0, 0.02, -0.41], [0.02, 1.0, 0.13], [-0.41, 0.13, 1.0]],
      "spearman": [[1.0, 0.04, -0.39], [0.04, 1.0, 0.10], [-0.39, 0.10, 1.0]]
    },
    "categorical_association": { "fields": [...], "cramers_v": [...] },
    "copresence":              { "fields": [...], "matrix":    [...] },
    "functional_dependencies": [
      { "lhs": "country_code", "rhs": "currency", "support": 0.998 }
    ]
  },
  "warnings": [ ... per-field surprises, decode errors, etc. ... ]
}
```

`generate predicates --survey path/to/survey.json` consumes this
schema directly. The richer selectivity-driven synthesis described
in `gen_predicates.rs:600-678` operates against the
`QuantileSketch`, `HeavyHitters`, `ExactFrequencyTable`, and
`categorical_association` fields. The legacy survey JSON shape is
removed in the same change as the legacy survey command.

---

## 13.9 CLI and Pipeline Integration

### Command

`analyze survey` is the command. Before this lands, the existing
single-pass implementation is renamed to **`analyze legacy-survey`**
so the new design owns the unqualified name with no risk of
confusion. `legacy-survey` is removed once the new survey covers its
use cases.

Every CLI flag below has a congruent YAML key under
`dataset.yaml`'s `survey:` block, and vice-versa. Precedence is
**CLI > YAML > built-in default**.

| CLI flag | YAML key (`survey.*`) | Default | Purpose |
|----------|-----------------------|---------|---------|
| `--source` | `source` | required | metadata slab file (ANode → MNode records); other formats are not supported |
| `--output` | `output` | `survey.json` | structured report path |
| `--samples` | `samples` | `100000` | per-pass sample cap |
| `--sampling` | `sampling` | `stride` | `stride` / `reservoir` / `exhaustive` |
| `--low-card-threshold` | `low-card-threshold` | `64` | ExactFrequencyTable cutoff |
| `--mid-card-threshold` | `mid-card-threshold` | `4096` | HLL fallback cutoff |
| `--top-k` | `top-k` | `64` | HeavyHitters width |
| `--quantile-k` | `quantile-k` | `200` | KLL sketch parameter |
| `--hll-precision` | `hll-precision` | `12` | HLL register count = 2^p |
| `--reservoir-size` | `reservoir-size` | `1024` | per-field reservoir |
| `--max-pair-analyses` | `max-pair-analyses` | `1024` | cross-field budget |
| `--correlations` | `correlations` | `auto` | `auto` / `none` / `numeric` / `categorical` / `all` |
| `--measures` | `measures` (map) | `auto` | per-field measure overrides (`field: [Measure, …]`) |
| `--semantic-confidence` | `semantic-confidence` | `0.95` | per-field threshold for committing to a `SemanticType` |
| `--force-semantic-type` | `force-semantic-type` (map) | empty | force a field's `SemanticType` (`field: Number(Integer)`); bypasses the probe verdict |
| `--memory-budget-mb` | `memory-budget-mb` | `256` | initial sketch-memory ceiling; the governor (`--resources mem=…`) supersedes this at runtime if set |
| `--batch-size` | `batch-size` | governor `segmentsize` or auto | records per batch; batch boundary is where the governor is consulted, sketches are checked, and progress reports update |
| `--findings-markdown` | `findings-markdown` | `survey.findings.md` next to `--output` | operator-readable Markdown findings (§13.11.4); empty string disables |
| `--findings-json` | `findings-json` | `survey.findings.json` next to `--output` | machine-readable curated findings (§13.11.4); empty string disables |
| `--findings-severity` | `findings-severity` | `info` | minimum severity included in findings outputs (`info`/`notable`/`warning`/`error`) |
| `--progress-log-interval` | `progress-log-interval` | `1` interactive / `10` headless | batches between milestone log lines |
| `--no-progress` | `no-progress` | off | suppress the live progress bar (still emits milestone log lines and StatusSource updates) |

Example `dataset.yaml` block:

```yaml
survey:
  samples: 250000
  sampling: reservoir
  semantic-confidence: 0.99
  force-semantic-type:
    user_id: Identifier(Sequential)
    legacy_count_str: Number(Integer)
  measures:
    user_email: [StructuredFormatDetectors, HyperLogLog, HeavyHitters]
  findings-severity: notable
```

### Programmatic API

```rust
pub fn survey(
    path: &Path,
    config: SurveyConfig,
    ui: Option<&UiHandle>,
) -> Result<SurveyReport, String>;
```

`SurveyReport` and all `MeasureReport` variants are
`serde::Serialize + Deserialize`, sharing the §13.8 JSON layout.

### Downstream consumers

- `generate predicates`: uses `QuantileSketch` for range-bucket
  selection, `ExactFrequencyTable` or `HeavyHitters` for value
  picking, and `categorical_association` to choose **correlated**
  field combinations for conjugates. The current `gen_predicates`
  independence assumption (n-th-root rule for joint selectivity) is
  replaced with cross-field-aware selection in the same change that
  lands this survey.
- `gen metadata` (synthesis): when we generate metadata with a
  survey-shaped distribution (skewed values, ID-like prefixes,
  etc.), the report is the
  natural input format.
- `analyze describe-dataset`: includes a survey-derived section
  describing field distributions in human-readable prose.

---

## 13.10 Implementation Sketch (proposed module layout)

```
veks-pipeline/src/pipeline/commands/survey/
├── mod.rs                      // SurveyOp CommandOp + factory
├── orchestrator.rs             // two-pass driver, sampling, pair dispatch
├── exploration.rs              // ExplorationProbe + state machine
├── template.rs                 // FieldTemplate synthesis from probe
├── measures/
│   ├── mod.rs                  // Measure trait + MeasureKind enum
│   ├── universal.rs            // Presence, TypeStability, Reservoir
│   ├── numeric.rs              // Moments, Quantile, Histogram, Monotonicity, Repetition, BitWidth
│   ├── cardinality.rs          // HLL, HeavyHitters, ExactFrequencyTable
│   ├── temporal.rs             // TemporalRange, CalendarHistograms, GapDistribution, EpochPlausibility
│   ├── textual.rs              // LengthMoments, CharClassMix, PatternSkeleton, PrefixDistribution, TokenStats, StructuredFormatDetectors
│   ├── bytes.rs                // ByteEntropy, MagicByteVotes
│   └── distribution_fit.rs     // χ² goodness-of-fit family
├── crossfield/
│   ├── mod.rs                  // PairAnalyzer trait + eligibility planner
│   ├── numeric_corr.rs         // Pearson, Spearman, residuals
│   ├── categorical_assoc.rs    // Contingency, Cramér's V, mutual information
│   ├── copresence.rs           // Presence-set intersection / Jaccard
│   ├── trend.rs                // Numeric × RecordIndex, Mann-Kendall
│   └── functional_dep.rs       // Approximate FD detection
├── sketches/
│   ├── mod.rs                  // Sketch trait + memory budget plumbing
│   ├── kll.rs                  // KLL quantile sketch
│   ├── hll.rs                  // HyperLogLog
│   ├── misra_gries.rs          // Heavy hitters
│   └── reservoir.rs            // Reservoir sampling
├── governor.rs                 // ResourceGovernor adapter — batch loop, downscale policy, throttle handling, demand offers
├── progress.rs                 // UiHandle progress bars + spinners + milestone logger; StatusSource impl for `veks status`
├── report.rs                   // SurveyReport + JSON envelope (machine-readable §13.8)
└── findings.rs                 // SurveyReport → survey.findings.md + survey.findings.json (curated, operator and machine; §13.11.4)
```

The trait surface stays small:

```rust
trait Measure: Send {
    fn observe(&mut self, value: &MValue, ctx: &MeasureCtx);
    fn finalize(self: Box<Self>) -> MeasureReport;
    fn kind(&self) -> MeasureKind;
}

trait PairAnalyzer: Send {
    fn observe_pair(&mut self, a: &MValue, b: &MValue, ctx: &MeasureCtx);
    fn finalize(self: Box<Self>) -> PairReport;
    fn kind(&self) -> PairAnalyzerKind;
}

trait Sketch {
    type Snapshot;
    fn memory_bytes(&self) -> usize;
    fn snapshot(&self) -> Self::Snapshot;   // for finalize
}
```

External crates considered for sketches: `hyperloglog-rs`, `tdigest`,
`kll`-style implementations exist on crates.io. The proposal is to
**vendor minimal implementations** for the four sketches we need
(KLL, HLL, Misra-Gries, reservoir) — they total a few hundred lines
each, removing a transitive dependency risk and giving us control
over the snapshot/serialization format that ends up in
`survey.json`.

---

## 13.11 Operational Concerns: Governor, Batching, Progress, Findings

The survey is a long-running, RAM-hungry pipeline command that has to
share a host with other pipeline steps and stay interactive for the
operator. Four cross-cutting concerns are first-class in the design,
not bolted on after the fact: **resource governance**, **batched
work**, **progress and status indicators**, and a **plain-language
findings report**.

### 13.11.1 Resource Governor Integration

The survey is a `CommandOp` like every other pipeline command — it
receives a `ResourceGovernor` from `StreamContext.governor`
(`veks-pipeline/src/pipeline/resource.rs:1406`) and obeys the same
contract every other command does:

- **Read effective values** at each batch boundary via
  `governor.current("mem")`, `governor.current("threads")`,
  `governor.current_or("segmentsize", default)`.
- **Honor `should_throttle()` / emergency flags** by pausing
  observation between records, not mid-record. Throttling collapses
  to a sleep-and-recheck loop with a bounded back-off.
- **Surface demand** with `governor.offer_demand("mem", current,
  desired)` whenever the orchestrator detects it could use more RAM
  productively (e.g. on a large schema where increasing
  `quantile_k` or `top_k` would tighten estimates).
- **Call `governor.checkpoint()`** at well-defined points (end of
  each batch, end of each pass, after cross-field finalize) so the
  governor's background thread can re-evaluate and adjust effective
  values.

Resources the survey claims:

| Resource         | How it's used                                                    |
|------------------|------------------------------------------------------------------|
| `mem`            | Caps aggregate sketch memory, reservoirs, cross-field accumulators (see §13.11.5) |
| `threads`        | Parallelism for cross-field pair analyzers and the Pass 2 measure-suite finalize phase. Within a record, per-field observation is sequential — parallelism only enters at pair-group and finalize boundaries. |
| `segmentsize`    | Default batch size in records. When absent, the orchestrator auto-sizes from `mem / sketch_memory_per_field × heuristic`. |

The survey does **not** invent its own resource concepts. Everything
flows through the existing governor so `veks run --resources …` and
the budgets in `dataset.yaml` work the same way they do for KNN
compute and sort.

### 13.11.2 Batched Work

Records are processed in **batches** keyed by `segmentsize`. The
batch is the unit at which:

1. The governor is consulted (`current`, `checkpoint`,
   `should_throttle`).
2. Sketch memory is checked against `mem` ceiling and **adaptive
   downsizing** kicks in if RSS climbs past the budget. Downsizing
   priority (cheapest to most painful):
   - shrink per-field reservoirs (1024 → 512 → 256)
   - drop the bottom-priority cross-field pair analyzers
   - reduce KLL `k` parameter (200 → 100 → 50; accuracy degrades
     gracefully)
   - reduce HLL precision (12 → 10; stderr widens)
   - finally, drop measures from the lowest-priority fields by
     uniqueness rank
3. Progress reports update with batch-level stats.
4. Backpressure: if the governor signals throttle during a batch,
   the orchestrator finishes the in-progress record, then pauses
   between records (never mid-MNode).

The two-pass shape composes cleanly with batching: each pass is a
sequence of batches; sketches accumulate **across** batches; the
Pass 1 → Pass 2 transition is a single template-synthesis step
between the last Pass 1 batch and the first Pass 2 batch. The same
adaptive downsizing rules apply in both passes, with the Pass 1
budget reserved against the larger Pass 2 budget (e.g. Pass 1 gets
20% of `mem`, Pass 2 gets 80%).

### 13.11.3 Progress and Status Indicators

The survey uses the standard `UiHandle` pipeline (`veks-core/src/ui`)
so its output integrates with every other command's progress shell —
plain terminal, ratatui TUI, headless log, all work the same.

Live progress signals during a run:

- **Primary record-count bar**: shows pass index, records processed,
  rate, ETA. Format:
  ```
  Pass 1/2 — surveying:  4 234 112 / 10 000 000   312k rec/s   ETA 18s
  ```
- **Per-batch milestone log lines** for events worth surfacing:
  - sketch downscale events ("mem ceiling 64% — shrinking reservoirs from 1024 → 512")
  - per-field type commitments at first observation
  - Pass-1 template synthesis summary ("412 fields, 3 Unstable, 24 LowCard, 38 MidCard, 347 HighCard, 184 cross-field pairs scheduled")
  - start of Pass 2 with the measure plan
  - cross-field finalize start/end ("computing 184 pair analyzers across 12 threads")
- **Resource status line** sampled from
  `ResourceGovernor::status_source().status_line()` every N seconds
  (default 5s). Same format as every other pipeline command.
- **StatusSource implementation** on the orchestrator so external
  `veks status` polling sees:
  ```
  pass=1/2  records=4,234,112/10,000,000  rate=312k/s  ETA=18s
  fields_classified=412  unstable=3  mem=2.1GB/4GB  threads=8/12
  ```

Spinners are reserved for phases where the record count isn't the
right denominator: cross-field pair finalize ("finalizing 482
analyzers…"), template synthesis ("classifying 412 fields…"),
findings report rendering.

### 13.11.4 Findings Report (operator and machine summaries)

The survey emits **three** artifacts at the end:

1. **`survey.json`** — the structured machine-readable report from
   §13.8. Full per-field measure outputs, cross-field matrices,
   warnings. Consumed by `generate predicates`, `analyze
   describe-dataset`, and downstream tooling. Not what a human
   reads first.
2. **`survey.findings.md`** — the plain-language summary an operator
   actually opens. Hand-tuned to surface what's interesting and
   leave the noise in the JSON. Renders cleanly in terminals,
   GitHub PRs, and documentation pipelines.
3. **`survey.findings.json`** — the **same curated findings** as the
   Markdown report, serialized as a structured list. Lets CI
   integrations, dashboards, and downstream tooling consume the
   curated subset without parsing the full `survey.json` and
   replaying the curation heuristics. The two findings artifacts
   are produced by the same renderer pass — they cannot drift.

The findings report (in both forms) is **not** a pretty-printed dump
of `survey.json`; it is a curated narrative organized by what
matters for downstream decisions. Default sections (Markdown headings;
JSON `section` field):

| Section                             | What it contains                                                                                  |
|-------------------------------------|---------------------------------------------------------------------------------------------------|
| **Overview**                        | Record count, sampled count, sampling regime, runtime, peak RSS, governor adjustments observed.   |
| **Schema at a glance**              | Field count by `(WireEncoding, SemanticType)` cell. One table, dense.                             |
| **Unstable fields**                 | Each unstable field with its `WireEncodingHistogram`, top probe-attempt rates, sample values, and a one-line "why this is unstable" diagnosis. |
| **Partition-candidate fields**      | Fields whose `(SemanticType, regime)` make them good partition keys: LowCard categorical, integer enums, ID-prefix composites. Each row shows distinct count, distribution entropy, and the joint-selectivity table when combined with the top-correlated peer. |
| **Predicate-candidate fields**      | Numeric and Temporal fields suitable for range predicates, ranked by (1) absence of skew that would collapse a uniform range to a corner of the distribution and (2) availability of a quantile sketch for selectivity targeting. |
| **High-signal numeric fields**      | Fields with notable distribution shapes — strong skew, multimodality, monotonic-in-record-order, unusual run-length structure. |
| **Identifier fields**               | UUIDs, sequential IDs, hash-like fields, composite identifiers (`USR_00123`-style). Useful for join planning and synthetic ID generation. |
| **Cross-field highlights**          | Top-K Pearson, top-K Spearman, top-K Cramér's V, top-K mutual-information pairs, top-K candidate functional dependencies. Each entry shows the strength and the eligibility-pruning rank so the operator can see what was *not* compared. |
| **Data-quality flags**              | High null rates, decode errors, type-tag instability, mixed encodings on field that *did* get a semantic verdict but with low confidence. Severity-tagged. |
| **Pass 1 vs Pass 2 surprises**      | Cases where Pass 2 found something Pass 1's pilot didn't predict (new type tag, distribution shift, cardinality misestimation). Surfaced as warnings. |
| **Recommendations**                 | Concrete suggestions: "use field X as partition key", "field Y looks like it should be timestamp but is stored as Text", "fields A and B are 0.99 functionally dependent — consider deriving one from the other". |

Each finding is **severity-tagged** (`info`, `notable`, `warning`,
`error`) and carries a pointer (`json_path`) into `survey.json` so
a curious operator can drill down without grepping. The Markdown
renders the pointer as a `<sub>↳ json:…</sub>` trailer; the JSON
form exposes it as the `json_path` field on each finding.

`survey.findings.json` schema (sketch):

```jsonc
{
  "schema_version": 1,
  "produced_by": "veks-pipeline analyze survey (findings)",
  "source": { "path": "metadata_content.slab", "survey_json": "survey.json" },
  "findings": [
    {
      "section": "Partition-candidate fields",
      "severity": "notable",
      "field": "country_code",
      "title": "country_code is a strong partition-key candidate",
      "body": "LowCard categorical (24 distinct), nearly uniform (entropy 4.51 of 4.58 max). Co-presence with `currency` is 0.998 and Cramér's V 0.997 — partitioning on `country_code` cleanly subdivides `currency`.",
      "samples": ["US", "GB", "DE", "JP", "BR"],
      "json_path": "fields.country_code.measures.ExactFrequencyTable"
    },
    {
      "section": "Data-quality flags",
      "severity": "warning",
      "field": "user_email",
      "title": "user_email contains 0.4% non-email values",
      "body": "StructuredFormatDetectors.email = 0.996. Sample rejects: \"unknown\", \"-\", \"N/A\".",
      "samples": ["unknown", "-", "N/A"],
      "json_path": "fields.user_email.measures.StructuredFormatDetectors"
    }
  ]
}
```

The renderer is a single module (`survey::findings`) that consumes
`SurveyReport` and emits both forms in one pass — they share a
curated-findings tree internally so they cannot drift. This keeps
the JSON producer focused on facts, the findings producer focused
on curation, and lets us version the findings template
independently of the survey schema.

### 13.11.5 Memory and Time Budget

For typical schemas (≤ 256 fields, ≤ 1024 cross-field pairs):

| Component                                  | Memory per field | Aggregate |
|--------------------------------------------|------------------|-----------|
| Per-field exact moments (m1..m4 + min/max) | 64 B             | 16 KB     |
| KLL quantile sketch (k=200)                | ~4 KB            | 1 MB      |
| HyperLogLog (p=12)                         | ~4 KB            | 1 MB      |
| Misra-Gries (k=64)                         | ~2 KB            | 0.5 MB    |
| Reservoir (size=1024)                      | depends on type  | typ. ~50 KB total |
| Cross-field accumulators (avg)             | ~256 B per pair  | 256 KB    |
| **Total typical**                          |                  | **~3–5 MB** |

Time cost: Pass 1 has the same wall-clock as today's survey on the
sample set (the probe is cheaper than the existing FieldStats
observer for most types). Pass 2 adds a second traversal of the
sample, plus per-record sketch updates that are O(log N) for KLL and
O(1) for the others. End-to-end the two-pass survey runs at roughly
1.5× the current survey on typical metadata.

---

## 13.12 Settled Decisions and Remaining Open Questions

The decisions below are settled (greenfield posture: pick the
structurally correct answer, don't carry parallel options for
backwards-compat reasons). They are documented here as **settled**
rather than left implicit so a future reader can see why the spec
makes the choices it does.

### Settled

- **Command name.** New command is `analyze survey`. The existing
  single-pass command is renamed to `analyze legacy-survey` before
  the new code lands, removed once parity is reached.
- **No single-pass mode.** Two passes are the survey shape. A
  template-only mode would be a different command (and isn't
  proposed here).
- **Sketch vendoring.** KLL, HLL, Misra-Gries, and reservoir are
  vendored in-tree. Snapshot stability + JSON layout control beat
  crate reuse.
- **Schema layout.** Per-field `measures: { MeasureKind: report }`
  open map. Forward-compat over compile-time type safety. New
  measures land without bumping `schema_version`; structural changes
  to a measure's payload do bump it.
- **Synthesis-mode-2 wiring.** Bootstrap emits
  `gen metadata → analyze survey → generate predicates
  --mode=survey --strategy=compound`. No new mode in
  `gen predicates`; the survey pipeline is agnostic to whether the
  slab was hand-curated or freshly synthesized.
- **Sampling default.** Page-stride sampling stays the default for
  reproducibility. `--sampling reservoir|exhaustive` is exposed for
  cases where the operator suspects ordering bias.
- **Downscale priority.** The order in §13.11.2 is internal: shrink
  reservoirs → drop low-priority pair analyzers → reduce KLL k →
  reduce HLL precision → drop low-priority fields. Not exposed as a
  knob; revisit if specific use-cases demand override.
- **Headless progress granularity.** Every-batch logging in
  interactive mode, every-N-batches (N=10) in headless. Override via
  `--progress-log-interval`.

### Additional settled (from inline review)

- **Pair-analyzer heuristic.** Default ranking
  `min(uniqueness(A), uniqueness(B)) × copresence(A,B)`, capped at
  `--max-pair-analyses` / `survey.max-pair-analyses`. Single-key
  ranking; no second-order tiebreaker. The ranking is rendered in
  the report so an operator can see *what was not compared*.
- **Unstable threshold override is dual-surfaced.** Per the project's
  CLI/YAML mirror principle, the confidence threshold (default
  `0.95`) and any per-field forced `SemanticType` are reachable
  from both `dataset.yaml`'s `survey:` block **and** the CLI
  (`--semantic-confidence`, `--force-semantic-type field=Integer`).
  Precedence: CLI > YAML > built-in default.
- **Findings emitted in both Markdown and JSON.** The findings
  renderer writes `survey.findings.md` (operator artifact) **and**
  `survey.findings.json` (machine-readable, same curated content
  with the same `json_path` pointers, suitable for CI integrations
  and downstream tooling that wants the curated subset without
  parsing the full `survey.json`).

### Remaining open questions

(None at the moment — flag any new ones inline in the doc and they
will move into Settled or get a recommendation in the next pass.)
---

## 13.13 Build Plan (post-acceptance)

1. **Rename existing command** to `analyze legacy-survey`. Update
   `gen predicates --survey` and any other internal callers to keep
   working against `legacy-survey`'s output until the new survey
   replaces it. This step ships independently.
2. Land vendored sketches (`kll`, `hll`, `misra_gries`, `reservoir`)
   with unit tests against textbook reference outputs.
3. Land `ExplorationProbe`, `FieldTemplate`, and the `Measure` trait
   with universal measures + `ExactMoments` / `ExactExtrema`.
4. Land the **governor adapter** and **progress / StatusSource
   scaffolding** (§13.10 `governor.rs`, `progress.rs`) before the
   measure suites, so every subsequent step has working resource
   shaping and live progress from the start.
5. Land the new `analyze survey` orchestrator (two-pass driver,
   sampling, template synthesis) producing `survey.json`.
6. Land the per-type measure suites (numeric, textual, cardinality,
   temporal, bytes) one family at a time, each with golden-output
   tests on fixture slabs.
7. Land cross-field analyzers (`numeric_corr`, `categorical_assoc`,
   `copresence`, `trend`, `functional_dep`).
8. Land the **findings renderer** (§13.10 `findings.rs`) producing
   `survey.findings.md` *and* `survey.findings.json` from
   `SurveyReport` in a single pass over a shared curated-findings
   tree. Each report section is independently shippable.
9. Switch `gen predicates --mode=survey` over to the new
   `survey.json` schema. Remove the legacy reader path.
10. Wire the bootstrap wizard's synthesis-mode-2 option to
    `gen metadata → analyze survey → generate predicates
    --mode=survey --strategy=compound`.
11. Remove `analyze legacy-survey` and the `survey_slab` /
    `survey_ivec` code it backs.

Each step is independently shippable. The orchestrator handles a
partial measure set by emitting reports only for kinds that are
implemented; the findings renderer emits only the sections backed
by data present in the report. Integration tests at each step run
the full pipeline end-to-end against fixture slabs with numerical
verification (per project memory `feedback_integration_tests`).
