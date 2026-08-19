# PVS Corpus Selection Annex

Status: DRAFT — criteria and aspects elaborated; matrix scores are projected
pending sample-survey confirmation (§4.4)
Date: 2026-08-14
Companion to: [`amazon-reviews-2023-pvs-plan.md`](amazon-reviews-2023-pvs-plan.md)

This annex holds the corpus-selection criteria for predicated-vector-search
(PVS) dataset construction and the survey of candidate corpora evaluated
against them. The main plan targets the reference benchmark's corpus (Amazon
Reviews 2023 items); that corpus is thin as PVS raw material — effectively two
usable filter fields, no multi-valued fields, no temporal texture — so this
annex defines what "rich enough" means, how coverage is *scored*, and catalogs
alternatives with comparable accessibility. Survey facts verified 2026-08-14/15.

---

## 1. Purpose and scope

Three uses:

1. **Selection** — choose successor/companion corpora for PVS testing layers,
   scored against explicit criteria rather than ad-hoc appeal.
2. **Specification** — the texture aspects (§3) double as a requirements
   vocabulary for the predicate-generation and metadata tooling: each aspect a
   corpus provides is an aspect the pipeline must be able to survey, predicate
   over, and verify.
3. **Layer planning** — testing layers declare required aspects at minimum
   FOC levels (§4.5); a corpus is *ready* for a layer iff it meets all minima.

---

## 2. Selection criteria

Six criteria, each scored on an anchored 0–4 scale. **C1 and C2 are gates**:
a candidate scoring below 2 on either is disqualified regardless of other
scores. C3–C6 are trade-offs weighted per purpose. **C5 is derived** from the
aspect scores (§4.3), not judged independently.

Each criterion states: what it measures, which plan phase it gates, what
evidence scores it, and the level anchors.

### C1 — Legal / IP posture

**Measures**: whether we may acquire the corpus, build the facets, and
*redistribute* what we build.
**Gates**: plan Phase 1 (acquisition) and Phase 5 (publish/push).
**Evidence**: the license text itself; any access-time riders; presence of
takedown/dispute history; whether derived facets (embeddings, metadata slabs,
predicate sets, ground truth) can be published. Embeddings are generally
low-risk derived works (precedent: paragraph embeddings of CC-BY-SA text
published under Apache-2.0), but share-alike and research-only terms can
encumber the **metadata facet**, which carries source values verbatim.

| Level | Anchor |
|---|---|
| 4 | CC0 / public domain; no attribution, no restrictions; all derived facets freely publishable |
| 3 | Attribution-only (CC-BY, ODC-BY, Apache-style); derived facets publishable with credit |
| 2 | Share-alike (CC-BY-SA, ODbL) or minor caveats (e.g. text-copyright carve-outs that facet design can route around); publishable with license propagation on affected facets |
| 1 | Research-only / non-commercial terms, or post-hoc riders conflicting with the underlying license; publication posture ambiguous |
| 0 | Approval-gated, revocable, or prohibitive terms; publication effectively barred |

### C2 — Access mechanics

**Measures**: friction and reliability of acquiring the corpus at 100GB+
scale.
**Gates**: plan Phase 1; also re-acquisition reproducibility (provenance).
**Evidence**: gating (none → registration/API key → click-through →
approval); transport (bulk HTTP(S)/S3 > torrent > paginated API);
resumability and `download huggingface` / `download bulk` compatibility;
hosting stability (institutional, versioned snapshots > community mirrors);
format friction as a noted modifier (parquet/npy native to veks-core >
jsonl/CSV one preprocessing hop > XML heavier).

| Level | Anchor |
|---|---|
| 4 | Ungated, scriptable, resumable bulk download from stable versioned hosting |
| 3 | Free registration/API key, or ungated with meaningful format/assembly friction |
| 2 | Click-through agreement, torrent-only distribution, or unstable/unofficial mirrors |
| 1 | Approval-gated with plausible grants; or official channel restricted with only archived copies |
| 0 | Effectively unobtainable at scale |

### C3 — Scale and headroom

**Measures**: item cardinality relative to the reference benchmark's 15.3M,
and headroom to grow *the same corpus* (same schema, same distributions)
toward 100M+.
**Gates**: plan §2 cardinality targets; strata design.
**Evidence**: documented corpus size, **stated as the cardinality of the
primary record type** — the unit that becomes one base vector (work, paper,
article, image, paragraph, release, post, pair, item). Counts at different
granularities are not comparable: 250M *paragraphs* is not 250M *works*.
Where a corpus offers multiple granularities (work vs abstract-bearing
subset, release vs track, item vs review), score on the granularity intended
as the vector unit and note the alternates — a finer granularity that
multiplies the count only upgrades C3 if it is genuinely the unit we would
embed and search. Also note natural scale points that map onto sized strata,
and per-item text length (embedding cost / cluster realism modifier).

| Level | Anchor |
|---|---|
| 4 | ≥100M items |
| 3 | ≥30M |
| 2 | ≥15M (parity with the reference large corpus) |
| 1 | 5–15M |
| 0 | <5M |

### C4 — Embeddability

**Measures**: cost and quality of obtaining full-cardinality vectors.
**Gates**: plan §3 (vector-source decision), Phase 2.
**Evidence**: precomputed embeddings (coverage, dimensionality, license,
model era); else local embedding cost (items × tokens on one GPU) and whether
the text is semantically load-bearing (title-only corpora embed cheaply but
yield weak cluster structure, which weakens T9).

| Level | Anchor |
|---|---|
| 4 | Precomputed embeddings: full cardinality, permissive license, credible model |
| 3 | Precomputed with caveats (partial coverage, dated model, intersection work, license friction) |
| 2 | No embeddings, but full substantive text available for local embedding |
| 1 | Weak text only (titles/short strings) — embeddable but semantically thin |
| 0 | No embeddable content |

### C5 — Metadata richness (derived)

**Measures**: coverage of the texture aspects in §3.
**Gates**: which testing layers the corpus can serve at all.
**Evidence**: the aspect score vector (§4); C5 is computed, not judged:

> C5 = round(mean of the ten FOC aspect scores), i.e. `round(Σsₐ / 10)`.

Field-quality factors (population rate, parse cleanliness, controlled
vocabularies, documented semantics, joinability to sibling datasets) enter
through the FOC gates in §4, so they are captured here without double
counting.

### C6 — Ecosystem comparability

**Measures**: external reference points on the same corpus.
**Gates**: credibility of results; availability of *external oracles* to
cross-check our facet computation (especially F).
**Evidence**: published ANN/filtered-search ground truth; benchmark suites
using the corpus; prevalence as an IR/vector-DB evaluation corpus.

| Level | Anchor |
|---|---|
| 4 | Published vector-search GT on this corpus (usable as an external oracle) |
| 3 | Corpus is a standard benchmark substrate (IR suites, shared tasks) without vector GT |
| 2 | Common demo/eval corpus in the vector ecosystem |
| 1 | General research use only |
| 0 | No ecosystem presence |

---

## 3. Texture aspects

The predicate machinery (PNode ops `GT/LT/EQ/NE/GE/LE/IN/MATCHES`,
survey-calibrated selectivity via `analyze survey` → `generate predicates`)
can express ten texture aspects. Summary first; per-aspect elaboration in
§3.1–§3.10. Each aspect defines **qualification gates** — what a field must
be for the aspect to count as present at all — which the scoring system in §4
builds on.

| # | Aspect | PNode form | One-line why |
|---|---|---|---|
| T1 | Low-cardinality categorical | `EQ`/`NE` | canonical partition filter; selectivity set by class priors |
| T2 | High-cardinality categorical | `EQ`, `IN` | long-tail selectivity; stresses per-value posting structures |
| T3 | Continuous numeric (skewed) | range ops | selectivity dialable via quantiles; skew breaks uniform planners |
| T4 | Temporal | range over date/epoch | freshness filters; insertion-order correlation |
| T5 | Boolean | `EQ` | two-class edge case at both balanced and extreme priors |
| T6 | Multi-valued set | `IN` over sets | the tag/label model; varying per-item cardinality |
| T7 | String pattern | `MATCHES` | non-hashable predicates; pattern selectivity estimation |
| T8 | Zipfian distribution | any op on skewed field | head ≈ unselective, tail ≈ needle, one field |
| T9 | Vector-correlation spectrum | any | the primary PVS difficulty axis |
| T10 | Compound / joint selectivity | `AND`/`OR` trees | joint ≠ product of marginals under correlation |

### T1 — Low-cardinality categorical

- **Definition**: a single-valued field with a small closed vocabulary.
- **Qualification gates**: 2–~200 distinct values; population ≥95% (or
  explicit missing semantics); deterministic normalization (no free-text
  variants of the same class).
- **Operational measures**: class priors spanning at least two selectivity
  decades (largest vs smallest class); prior stability across strata windows.
- **Difficulty contribution**: models the classic "category filter"; with O
  facet, doubles as the partition label.
- **Examples**: product category, language, work type.

### T2 — High-cardinality categorical

- **Definition**: single-valued field with an open/long-tail vocabulary.
- **Qualification gates**: ≥10⁴ distinct values; population ≥90%; stable
  identity (ids or canonicalized names, not free text).
- **Operational measures**: tail mass (share of items in values below rank
  1000); achievable `EQ` selectivities reaching ≤10⁻⁵; `IN`-set construction
  yields dialable union selectivity.
- **Difficulty contribution**: per-value posting lists too small to index
  independently; forces shared structures; tail predicates approximate
  worst-case pre-filtering.
- **Examples**: store, venue, label, author, user id.

### T3 — Continuous numeric (skewed)

- **Definition**: orderable numeric field with enough resolution for
  quantile-calibrated range predicates.
- **Qualification gates**: numeric type after deterministic parse
  (parse-clean rate ≥95%, failures explicitly encoded — cf. plan §4 Phase 1
  price sentinel); ≥10³ distinct values; population ≥90%.
- **Operational measures**: quantile resolution supporting target
  selectivities across ≥3 decades; skew measured (e.g. P99/P50 ratio, tail
  index); missing-value mass isolated.
- **Difficulty contribution**: arbitrary-precision selectivity dialing; skew
  defeats uniform-assumption selectivity estimation.
- **Examples**: price, citation count, score, helpful votes.

### T4 — Temporal

- **Definition**: date/epoch field with meaningful span and resolution.
- **Qualification gates**: span ≥10 years (or corpus-appropriate);
  resolution ≤1 day preferred (year-only acceptable, noted); population
  ≥90%.
- **Operational measures**: volume-over-time profile (growth curves make
  recent-window predicates selectivity-unstable across strata — measure it);
  correlation with insertion/ordinal order (after our shuffle, ordinal
  correlation is destroyed by construction — the *field* remains available as
  a low-T9 filter).
- **Difficulty contribution**: freshness filters are the most common
  real-world predicate; typically low vector-correlation, giving scattered
  passing sets.
- **Examples**: publication date, creation date, review timestamp.

### T5 — Boolean

- **Definition**: two-valued field (or deterministically derivable flag).
- **Qualification gates**: population ≥95%; both values present.
- **Operational measures**: minority-class share — ideally *multiple*
  boolean fields spanning near-50% (balanced) and ≤1% (rare-flag) priors,
  since the two regimes stress different code paths.
- **Difficulty contribution**: degenerate categorical; rare-flag booleans are
  natural high-selectivity `EQ` predicates, balanced ones nearly free.
- **Examples**: open-access, verified-purchase, accepted-answer, retracted.

### T6 — Multi-valued set

- **Definition**: field whose value is a *set* of labels per item.
- **Qualification gates**: set-valued representation preserved through
  ingestion (not flattened/first-only); vocabulary ≥10² distinct labels;
  per-item cardinality ≥1 for ≥90% of items.
- **Operational measures**: vocabulary size; per-item set-size distribution;
  label frequency skew; membership (`IN`) selectivity dialable from head
  labels (~10⁻¹) to tail labels (≤10⁻⁴); co-occurrence structure for
  intersection predicates.
- **Difficulty contribution**: the tag/label model of the filtered-ANN
  literature; union vs intersection semantics; the aspect the reference
  corpus wholly lacks. **Tooling-gated** — see §8.
- **Examples**: MeSH terms, SE tags, styles, topics, formats.

### T7 — String pattern

- **Definition**: free-ish text field suitable for `MATCHES` predicates.
- **Qualification gates**: string type; population ≥90%; value length and
  alphabet suitable for prefix/substring patterns (identifiers and names, not
  full prose).
- **Operational measures**: prefix-selectivity spread (selectivity as a
  function of pattern length spans ≥3 decades); pattern-evaluation cost
  bounded at corpus scale.
- **Difficulty contribution**: predicates with no precomputable posting
  list; worst case for filter acceleration, upper anchor for pre-filter scan
  cost.
- **Examples**: journal name, store name, title.

### T8 — Zipfian value distribution

- **Definition**: cross-cutting property — at least one field whose value
  frequencies are heavy-tailed enough that one field yields the full
  selectivity spectrum.
- **Qualification gates**: a qualifying T1/T2/T6 field exists.
- **Operational measures**: top-1 value share ≥5% while tail (rank >10³)
  retains ≥10% mass; rank-frequency slope in the Zipf-plausible band.
- **Difficulty contribution**: head values give cheap unselective filters,
  tail values give needle-in-haystack — a single field sweeps the spectrum
  without changing predicate shape.
- **Examples**: tags, labels, authors.

### T9 — Vector-correlation spectrum

- **Definition**: availability of filter fields at *both ends* of the
  filter↔embedding correlation spectrum — correlated fields (topic, category)
  concentrate the passing set in embedding space; uncorrelated ones
  (timestamp) scatter it.
- **Qualification gates**: at least one plausibly-correlated and one
  plausibly-uncorrelated qualifying field.
- **Operational measures** (requires pilot embeddings — plan Phase 3):
  **neighborhood lift** — for sample queries, the fraction of top-100
  neighbors sharing the query item's field value, divided by the field
  value's marginal selectivity. Lift ≫1 = correlated; ≈1 = uncorrelated. A
  corpus scores high when its fields span a wide, measured lift range.
- **Difficulty contribution**: the primary PVS difficulty axis: at fixed
  selectivity, scattered passing sets defeat graph locality while
  concentrated ones defeat naive pruning — both regimes must be testable.
- **Examples**: topics/MeSH (high lift) vs dates (≈1).

### T10 — Compound / joint selectivity

- **Definition**: field pairs/triples whose joint distribution deviates from
  independence, expressed as `AND`/`OR` PNode trees.
- **Qualification gates**: ≥2 qualifying fields of distinct aspects.
- **Operational measures**: pairwise lift `P(A∧B)/(P(A)·P(B))` measured for
  candidate pairs (survey cross-field statistics); availability of both
  positively-correlated pairs (lift >2) and near-independent pairs
  (lift ≈1); compound predicates achieving target joint selectivities across
  ≥3 decades.
- **Difficulty contribution**: correlated fields make joint selectivity
  unpredictable from marginals — stresses selectivity estimation, plan
  choice, and our own `selectivity-max` calibration.
- **Examples**: category × price, field × year × venue, tag co-occurrence.

### Cross-cutting difficulty dimensions

Applying to every aspect, and referenced by the operational gates above:

- **Selectivity spectrum** — target predicates spanning ~10⁻⁵ … 0.5 of the
  corpus; the reference benchmark tested a single point (~3%).
- **Passing-set geometry** — at fixed selectivity, whether passing items form
  one tight cluster, several clusters, or uniform scatter (driven by T9).
- **Strata stability** — sized profiles are prefix windows of the shuffled
  corpus; a predicate band survives down-stratum only if its selectivity is
  scale-invariant and its expected match count stays ≥k. Define each
  predicate set's **minimum stratum floor** = smallest stratum where
  `selectivity × stratum_size ≥ maxk`.
- **Per-query vs pooled predicates** — paired (query, predicate) sets vs a
  shared predicate pool across queries; layer-level choice (§9 TODO).

---

## 4. Coverage scoring system (FOC)

The **Functional/Operational Coverage** score ranks each aspect per corpus on
a five-level ordinal scale. "Functional" levels assert what the corpus
structurally provides; "Operational" levels additionally require measured
evidence at corpus scale.

### 4.1 The scale

| Level | Glyph | Name | Meaning | Evidence standard |
|---|---|---|---|---|
| 0 | ❌ | **Absent** | No field qualifies for the aspect | desk review |
| 1 | ☐ | **Nominal** | A candidate field exists but fails a qualification gate (population, cardinality band, parse determinism, flattened sets) or requires an unbuilt join | desk review |
| 2 | ☑ | **Functional** | ≥1 field passes all §3 qualification gates: predicates of this aspect *can be generated and evaluated* | desk review + spot check |
| 3 | ✅ | **Operational** | Field(s) pass the aspect's §3 operational measures on a ≥1M-record sample survey: selectivity dialable across the aspect's target band, distributions verified, strata-stable | measured (§4.4) |
| 4 | ⭐ | **Rich** | Level 3, **plus** ≥2 independent qualifying fields for the aspect, and (where T9-relevant) options at more than one point of the correlation spectrum — enabling compound layering without leaving the aspect | measured (§4.4) |

The glyph gradient ❌ ☐ ☑ ✅ ⭐ renders every 0–4 score in this document's
matrices (criteria scores use the same gradient against their §2 anchors):
the box appears when a field exists (☐), gets checked when it functions (☑),
turns verified-green when measured (✅), and gold when rich (⭐); ❌ means
nothing qualifies.

Rules:

- Levels are cumulative: 4 implies 3 implies 2.
- A field may score under multiple aspects (a Zipfian tag field counts for T6
  and T8), but a *join-dependent* field caps at 1 until the join is built and
  measured.
- Desk review may assign at most level 2. Levels 3–4 require the measurement
  protocol (§4.4). Projected 3/4 scores are marked provisional until then.

### 4.2 Per-corpus aggregates

For aspect score vector S = (s_T1 … s_T10):

- **Breadth** `B = |{a : sₐ ≥ 2}|` — how many aspects are at least
  Functional (0–10). The headline "how many textures can I pick here."
- **Depth** `D = Σ sₐ / 40` — normalized total coverage (0–1).
- **C5 (derived)** `= round(Σ sₐ / 10)` — feeds the criteria matrix.

Breadth and depth are reported separately, never merged: a corpus with one
world-class aspect (YFCC tags) and a broadly-adequate corpus are different
tools, and a single scalar would hide that.

### 4.3 Criteria composite

No overall scalar across C1–C6 either. Instead: **gate first** (C1 ≥2 and
C2 ≥2), then compare candidates on the criteria vector under per-purpose
weightings (§9 TODO defines the weightings; §7 gives qualitative per-purpose
recommendations meanwhile).

### 4.4 Measurement protocol (operationalizing levels 3–4)

To confirm any score ≥3, run on a ≥1M-record uniform sample (full corpus
where cheap):

1. **Field survey** — `analyze survey` per candidate field: population rate,
   distinct count, top-k value shares, tail mass, quantile table, per-item
   set-size distribution (T6), parse-failure mass (T3).
2. **Selectivity band check** — from the survey, compute achievable predicate
   selectivities per aspect; require the aspect's decade span; record each
   predicate band's minimum stratum floor (§3 cross-cutting).
3. **Cross-field lift** — pairwise `P(A∧B)/(P(A)P(B))` for nominated pairs
   (T10) from survey cross-field statistics.
4. **Neighborhood lift** (T9; requires pilot vectors — plan Phase 3): for
   ≥1k sample queries, top-100 neighbor label-agreement ÷ marginal
   selectivity, per field; report the min/max lift spread across fields.
5. **Strata stability** — repeat (1)–(2) on a prefix window one decade
   smaller; distributions must agree within tolerance.

Outputs land next to the survey artifacts and upgrade the matrix scores from
projected to measured. Steps (1)–(3) are `analyze survey` work (with the §8
set-valued extension); step (4) is a small new probe worth adding to
`analyze` when a T9-sensitive layer is scheduled.

### 4.5 Layer-readiness specs

A testing layer declares minimum FOC levels; a corpus is ready iff all minima
are met. Format: `layer: {aspect: min-level, …}`. Seed examples:

- `correlated-compound-range`: {T1≥3, T3≥3, T9≥3, T10≥3} — the reference
  benchmark's category×price shape, generalized. The incumbent corpus
  qualifies.
- `tag-membership`: {T6≥3, T8≥3} — filtered-ANN tag model. Requires
  OpenAlex / S2AG / PubMed / SE / Discogs / YFCC class corpora, and the §8
  tooling extension.
- `temporal-scatter`: {T4≥3, T9≥3 with a measured low-lift field} — freshness
  filters over scattered passing sets. Incumbent qualifies only with the
  review-side join (§5.9).
- `adversarial-selectivity-sweep`: {T8≥3, any of T2/T6≥3} — one predicate
  shape swept 10⁻⁵…0.5.

The full per-layer requirement set is §9 TODO.

---

## 5. Candidate profiles

Numeric scores use the §2 anchors; texture vectors are in §6.2. All scores
≥3 are projected pending §4.4 measurement.

### 5.1 OpenAlex

- **What**: open scholarly catalog, ~250M works (articles, books, preprints…).
- **C1=4**: CC0, no restrictions; derived facets freely publishable.
- **C2=4**: monthly full snapshots on S3 (anonymous `--no-sign-request`, no
  account) plus API; 2026+ snapshots publish **parquet alongside JSONL** —
  parquet is veks-native, so the format hop disappears.
- **C3=4**: 250M ceiling; natural strata: works-with-abstracts (~100M+) vs
  title-only (250M).
- **C4=2**: no precomputed embeddings — embed title+abstract locally (plan
  §3 Option B). Abstracts stored as reconstructable inverted index; coverage
  partial.
- **C5=4** (derived): richest typed metadata of any candidate — type,
  language, dates, venue/source, hierarchical multi-valued topics, citation
  counts, OA/retraction flags, authors/institutions/countries.
- **C6=1**: bibliometrics standard; no vector GT.

### 5.2 Semantic Scholar S2AG

- **What**: academic graph, ~200M papers, monthly bulk snapshots.
- **C1=3**: ODC-BY (attribution + linkback + citation ask).
- **C2=3**: free API key; gzipped JSONL bulk files via Datasets API.
- **C3=4**: ~200M.
- **C4=4**: **precomputed SPECTER embeddings ship as a bulk dataset** — full
  corpus, zero embedding cost (model era noted: SPECTER, not current-gen).
- **C5=3** (derived): year, venue, multi-valued fields of study,
  citation/influence counts, abstracts (subset); topics flatter than
  OpenAlex.
- **C6=2**: SPECTER/SciDocs ecosystem.

### 5.3 PubMed + MedCPT embeddings

- **What**: ~34M biomedical citations; NCBI publishes MedCPT article
  embeddings (768-d) for the whole corpus on its FTP server.
- **C1=3**: NLM terms; metadata effectively open; abstract text carries
  copyright caveats — distributing embeddings + metadata routes around text
  redistribution.
- **C2=4**: plain FTP/HTTPS bulk `.npy` + PMID json.
- **C3=3**: 34M — 2× the reference benchmark; no headroom beyond.
- **C4=4**: precomputed, turnkey.
- **C5=3** (derived): **MeSH** is the best multi-valued predicate vocabulary
  available anywhere — curated, hierarchical, Zipfian, major/minor
  qualifiers; plus year, journal, publication types, language; numeric T3
  only via NIH iCite join (caps at 1 until built).
- **C6=3**: standard biomedical IR substrate.

### 5.4 big-ann NeurIPS'23 filtered track (YFCC-10M)

- **What**: 10M YFCC100M images, CLIP embeddings, bag-of-tags metadata
  (200,386-word vocabulary: description words, camera, year, country), with
  **published filtered queries and ground truth**.
- **C1=3 / C2=4**: public benchmark artifacts, standard research use.
- **C3=1**: 10M, fixed — but scale is not its role here.
- **C4=4**: provided.
- **C5=1** (derived): single texture — everything flattened into the tag
  bag.
- **C6=4**: **published filtered GT = external oracle for
  `compute prefiltered-knn`** — its primary value.

### 5.5 Cohere Wikipedia 2023-11 embeddings

- **What**: ~250M paragraph embeddings (embed-multilingual-v3, 1024-d, plus
  int8/binary variants), all-language Wikipedia, ungated on HF.
- **C1=3**: embeddings Apache-2.0; underlying text CC-BY-SA (share-alike
  attaches to text carried into the metadata facet — score reflects
  facet-routable caveat).
- **C2=4**: ungated HF parquet — native ingestion.
- **C3=4**: ~250M.
- **C4=4**: precomputed — **the cheapest path to >100M vectors**.
- **C5=2** (derived): language (300+, extreme skew — a T1×T8×T9 triple, as
  language correlates strongly with multilingual embeddings), article title
  grouping, paragraph position; numeric/temporal/set textures only via joins
  (pageviews, Wikidata).
- **C6=2**: common vector-DB demo corpus.

### 5.6 Discogs monthly dumps

- **What**: music-release catalog, ~17M releases, monthly XML dumps.
- **C1=4**: **CC0.**
- **C2=3**: plain HTTPS monthly dumps; XML preprocessing.
- **C3=2**: ~17M releases (more at track granularity).
- **C4=1**: no embeddings; short text (artist + title + tracklist) — cheap
  to embed, semantically thin clusters.
- **C5=3** (derived): genre + styles (multi-valued), year, country, format
  (multi-valued), label/artist (high-card, Zipfian).
- **C6=0**: none.

### 5.7 Stack Exchange data dumps

- **What**: ~60M posts network-wide (Stack Overflow ~24M questions).
- **C1=2**: content CC-BY-SA, but official distribution moved behind a login
  with an anti-LLM-training rider that critics argue conflicts with the
  license; archived torrents exist (e.g. 2025-12-31 on Academic Torrents).
- **C2=2**: login-gated official download or torrent; XML.
- **C3=3**: 24–60M.
- **C4=2**: no embeddings; title+body is rich text.
- **C5=4** (derived): tags (multi-valued, ~65k Zipfian vocab, strongly
  vector-correlated), signed skewed score (negative values give natural
  `GT 0`/`NE` textures), view counts, dates, accepted/answered booleans,
  site.
- **C6=2**: common IR/QA corpus.

### 5.8 Re-LAION-5B

- **What**: 5.5B image-text pairs, relaunch of LAION-5B after CSAM removal.
- **C1=1 / C2=1**: Apache-2.0 but approval-gated on HF; takedown history;
  URL-based content with link-rot and content-risk exposure. **Fails the C1/C2
  gate.** Included for completeness; not recommended.

### 5.9 Enriching the incumbent (Amazon Reviews 2023)

Not an alternative but a floor-raiser: the *review* side of the same dataset
(571M reviews: rating, verified-purchase boolean, helpful-vote skewed count,
timestamp, high-cardinality user/item ids) joins onto the item corpus already
planned, adding T2/T3/T4/T5 textures without changing corpus, license
posture, or pipeline. The item records' `categories` path field can supply a
modest T6 (currently Nominal — messy paths, needs normalization). The
cheapest richness upgrade if we stay on the main plan's corpus.

### 5.10 S2ORC full text (passage-level)

- **What**: structured full text for ~15.7M open-access papers (2026 release)
  in the S2AG universe; **the passage is the vector unit** — ~60–120
  passages/paper ≈ **1–2B passages**, each inheriting its parent paper's
  graph metadata plus passage-local structure (section type, position).
- **C1=3**: ODC-BY.
- **C2=3**: bulk via the Datasets API (free key), JSONL; passage extraction
  is our preprocessing.
- **C3=4**: ~1–2B passages — legitimate under the §2 C3 rule when
  passage-retrieval is the intended search unit.
- **C4=2**: no passage embeddings exist; rich text, ~250 tokens/passage.
- **C5=4** (derived): parent metadata (venue, year, fields, citation counts)
  + section type (a *correlated* T1) + parent-paper id (natural T2 grouping).
- **C6=2**: standard corpus in pretraining/IR research.
- **DOI-joinable to OpenAlex**: CC0 topics/OA-status/institution fields can
  be grafted onto every passage.

### 5.11 PMC Open Access full text (passage-level)

- **What**: ~6M+ OA biomedical articles (JATS XML) → **~0.6–1B passages**
  with MeSH-rich inherited metadata. Effectively the full-text extension of
  §5.3.
- **C1=3**: per-article CC licenses (majority CC-BY; the commercial-use
  partition can be selected for clean redistribution).
- **C2=4**: ungated bulk on AWS S3 and NCBI FTP.
- **C3=4**: ~0.6–1B passages.
- **C4=2**: full text, no embeddings (MedCPT covers only title+abstract at
  article level).
- **C5=3** (derived): MeSH (T6 exemplar) + journal/year + section type; T3
  via iCite join. Biomedical-only narrows the T9 spread.
- **C6=3**: standard biomedical corpus.

### 5.12 US patent full text (passage-level)

- **What**: USPTO full-text grants (1976+) and applications (2001+), ~15M
  documents → **~2–4B passages/claims**; the worldwide bibliographic
  universe is 120M+ documents (Google Patents / IFI).
- **C1=4**: US patent text is public domain — the cleanest C1 at any scale.
- **C2=3**: ungated USPTO bulk XML (weekly files; era-varying schemas make
  assembly real work); BigQuery mirror for metadata joins.
- **C3=4**: ~2–4B passages — the absolute cardinality ceiling at
  ⭐-richness (§6.4).
- **C4=2**: long technical text; passage embedding is the program's dominant
  cost line.
- **C5=4** (derived): CPC/IPC hierarchical multi-valued codes (MeSH-grade
  T6), assignee/inventor (T2), multiple date fields (T4), citation/family/
  claim counts (T3), claim-vs-description passage type (correlated T1).
- **C6=2**: patent-IR benchmarks exist (CLEF-IP, HUPD).
- Domain-monotone (all patents) — narrows T9 diversity relative to
  scholarly corpora.

---

## 6. Comparison matrices

All scores ≥3 are **projected** (desk review); confirmation requires the
§4.4 protocol.

### 6.1 Criteria (0–4 per §2 anchors; C5 derived from §6.2)

| Candidate | C1 legal | C2 access | C3 scale (primary records) | C4 embed | C5 richness | C6 ecosystem | Gate |
|---|---|---|---|---|---|---|---|
| [OpenAlex](https://help.openalex.org/) | ⭐ | ⭐ | ⭐ ~250M works¹ | ☑ | ⭐ | ☐ | pass |
| [S2AG](https://www.semanticscholar.org/product/api) | ✅ | ✅ | ⭐ ~200M papers² | ⭐ | ✅ | ☑ | pass |
| [PubMed+MedCPT](https://github.com/ncbi/MedCPT) | ✅ | ⭐ | ✅ ~34M articles | ⭐ | ✅ | ✅ | pass |
| [big-ann YFCC-10M](https://big-ann-benchmarks.com/neurips23.html) | ✅ | ⭐ | ☐ 10M images (fixed) | ⭐ | ☐ | ⭐ | pass |
| [Cohere Wikipedia](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3) | ✅ | ⭐ | ⭐ ~250M paragraphs³ | ⭐ | ☑ | ☑ | pass |
| [Discogs](https://data.discogs.com/) | ⭐ | ✅ | ☑ ~17M releases⁴ | ☐ | ✅ | ❌ | pass |
| [Stack Exchange](https://archive.org/details/stackexchange) | ☑ | ☑ | ✅ ~60M posts⁵ | ☑ | ⭐ | ☑ | pass |
| [Re-LAION-5B](https://laion.ai/blog/relaion-5b/) | ☐ | ☐ | ⭐ ~5.5B image-text pairs | ✅ | ☐ | ☑ | **fail** |
| [S2ORC full text](https://github.com/allenai/s2orc) | ✅ | ✅ | ⭐ ~1–2B passages⁸ | ☑ | ⭐ | ☑ | pass |
| [PMC OA full text](https://registry.opendata.aws/ncbi-pmc/) | ✅ | ⭐ | ⭐ ~0.6–1B passages⁹ | ☑ | ✅ | ✅ | pass |
| [US patent full text](https://github.com/google/patents-public-data) | ⭐ | ✅ | ⭐ ~2–4B passages¹⁰ | ☑ | ⭐ | ☑ | pass |
| [Amazon items (incumbent)](https://amazon-reviews-2023.github.io/) | ✅ | ⭐ | ☑ 15.3M cleaned items⁶ | ☑ | ☑ | ☑ | pass |
| [Amazon items+reviews](https://amazon-reviews-2023.github.io/) | ✅ | ⭐ | ☑ 15.3M items⁷ | ☑ | ✅ | ☑ | pass |

Scale: ❌ 0 · ☐ 1 · ☑ 2 · ✅ 3 · ⭐ 4 (per-criterion anchors in §2; gate
requires C1 ≥ ☑ and C2 ≥ ☑). C3 cells state the cardinality of the
**primary record type** — the unit that becomes one base vector (§2 C3).

Granularity notes:
¹ works; ~100M+ carry abstracts (title-only embedding beyond that).
² papers, one SPECTER vector each; abstracts on a subset.
³ the *paragraph/passage* is the vector unit here (article-level cardinality
  is far smaller, across 300+ languages).
⁴ releases; track-level granularity raises the count several-fold but with
  even thinner text per record.
⁵ questions + answers network-wide (~24M Stack Overflow questions).
⁶ ~48.2M raw items reduce to 15.3M after the reference benchmark's cleaning;
  items are the vector unit.
⁷ item vectors unchanged at 15.3M — the 571M reviews contribute *metadata*
  textures, not vectors; embedding reviews as records would be a different
  corpus (571M review vectors) with its own C3.
⁸ ~15.7M full-text papers × ~60–120 passages; the passage is the vector
  unit; parent-paper metadata inherited per passage.
⁹ ~6M+ OA articles × ~100–150 passages; biomedical subset of the S2ORC
  universe.
¹⁰ ~15M US full-text documents (grants 1976+, applications 2001+) ×
  ~100–300 passages/claims; worldwide bibliographic universe 120M+ documents.

(Incumbent C1: research-published corpus, HF-hosted, widely redistributed;
C6: the reference benchmark itself used it, but published no reusable GT.)

### 6.2 Texture coverage (FOC 0–4 per §4.1; B = breadth, D = depth)

| Candidate | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | B | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [OpenAlex](https://help.openalex.org/) | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ✅ | ⭐ | ⭐ | ⭐ | 10 | .97 |
| [S2AG](https://www.semanticscholar.org/product/api) | ✅ | ✅ | ⭐ | ⭐ | ✅ | ✅ | ✅ | ⭐ | ⭐ | ✅ | 10 | .85 |
| [PubMed+MedCPT](https://github.com/ncbi/MedCPT) | ✅ | ✅ | ☐ʲ | ⭐ | ☑ | ⭐ | ✅ | ⭐ | ⭐ | ⭐ | 9 | .80 |
| [big-ann YFCC-10M](https://big-ann-benchmarks.com/neurips23.html) | ☐ | ❌ | ❌ | ☐ | ❌ | ⭐ | ❌ | ✅ | ☑ | ☑ | 4 | .33 |
| [Cohere Wikipedia](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3) | ✅ | ✅ | ☐ʲ | ☐ʲ | ❌ | ☐ʲ | ✅ | ✅ | ✅ | ☑ | 6 | .50 |
| [Discogs](https://data.discogs.com/) | ✅ | ✅ | ☐ | ✅ | ☐ | ⭐ | ✅ | ✅ | ☑ | ✅ | 8 | .65 |
| [Stack Exchange](https://archive.org/details/stackexchange) | ✅ | ✅ | ⭐ | ⭐ | ⭐ | ⭐ | ✅ | ⭐ | ⭐ | ⭐ | 10 | .93 |
| [Re-LAION-5B](https://laion.ai/blog/relaion-5b/) | ☑ | ❌ | ☑ | ❌ | ☐ | ❌ | ☐ | ☑ | ☐ | ☐ | 3 | .25 |
| [S2ORC full text](https://github.com/allenai/s2orc) | ⭐ | ⭐ | ⭐ | ⭐ | ✅ | ✅ | ✅ | ⭐ | ⭐ | ⭐ | 10 | .93 |
| [PMC OA full text](https://registry.opendata.aws/ncbi-pmc/) | ⭐ | ⭐ | ☐ʲ | ⭐ | ☑ | ⭐ | ✅ | ⭐ | ⭐ | ⭐ | 9 | .85 |
| [US patent full text](https://github.com/google/patents-public-data) | ⭐ | ⭐ | ⭐ | ⭐ | ✅ | ⭐ | ✅ | ⭐ | ⭐ | ⭐ | 10 | .95 |
| [Amazon items (incumbent)](https://amazon-reviews-2023.github.io/) | ✅ | ✅ | ✅ | ❌ | ❌ | ☐ | ✅ | ✅ | ✅ | ✅ | 7 | .55 |
| [Amazon items+reviews](https://amazon-reviews-2023.github.io/) | ✅ | ⭐ | ⭐ | ⭐ | ✅ | ☐ | ✅ | ⭐ | ✅ | ⭐ | 9 | .82 |

Scale: ❌ absent · ☐ nominal · ☑ functional · ✅ operational · ⭐ rich
(§4.1). ʲ = join-dependent field, capped at Nominal until the join is built
and measured (PubMed & PMC T3: iCite citation counts; Cohere Wikipedia
T3/T4/T6: pageviews / Wikidata).

Reading: OpenAlex, Stack Exchange, and S2AG are the broad-spectrum corpora;
PubMed+MedCPT is nearly as broad *with vectors included*; the incumbent
jumps from B=7/D=.55 to B=9/D=.82 with the review-side join — the single
cheapest depth upgrade available. The passage-level rows (S2ORC, PMC OA,
patents) match or exceed that breadth while multiplying cardinality 5–15×
over any document-level corpus — see §6.4.

### 6.3 Info pages and download locations

The info-page links below are the same targets used by the candidate row
headers in §6.1/§6.2.

| Candidate | Info page | Download location | Access notes |
|---|---|---|---|
| OpenAlex | [help.openalex.org](https://help.openalex.org/) | [snapshot download guide](https://help.openalex.org/download/download-to-machine) · [AWS Open Data registry](https://registry.opendata.aws/openalex/) — bucket `s3://openalex`, anonymous via `--no-sign-request` | monthly snapshots; JSONL **and parquet** prefixes (2026+); ~660 GB both formats |
| S2AG | [Semantic Scholar API](https://www.semanticscholar.org/product/api) | [Datasets API](https://api.semanticscholar.org/api-docs/datasets) | free API key; signed bulk-file URLs per release |
| PubMed+MedCPT | [MedCPT repo](https://github.com/ncbi/MedCPT) | [embeddings FTP](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/) · [PubMed baseline](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/) | plain HTTPS/FTP, `.npy` + PMID json; metadata from the baseline XML |
| big-ann YFCC-10M | [NeurIPS'23 track](https://big-ann-benchmarks.com/neurips23.html) | via [`benchmark/datasets.py`](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py) in the harness repo | canonical URLs encoded in the harness; includes filtered queries + GT |
| Cohere Wikipedia | [HF dataset card](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3) | same HF repo (parquet) · [int8/binary variant](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary) | ungated; `download huggingface`-compatible |
| Discogs | [data.discogs.com](https://data.discogs.com/) | [data.discogs.com](https://data.discogs.com/) (per-year directories) | monthly XML dumps: releases, artists, labels, masters |
| Stack Exchange | [archive.org item](https://archive.org/details/stackexchange) | [archive.org item](https://archive.org/details/stackexchange) · [2025-12-31 torrent](https://academictorrents.com/details/0d1d597fa7809f0e85f127b5eb3088219ddbad39) | official channel login-gated with rider (§5.7); archives carry CC-BY-SA content |
| Re-LAION-5B | [release post](https://laion.ai/blog/relaion-5b/) | [HF collection](https://huggingface.co/collections/laion/re-laion-5b-research-safe-67e311013ba899a938569e32) | approval-gated (fails C1/C2 gate) |
| S2ORC full text | [allenai/s2orc](https://github.com/allenai/s2orc) | [S2AG Datasets API](https://api.semanticscholar.org/api-docs/datasets) (`s2orc` dataset) | free API key; JSONL bulk; passage extraction is ours |
| PMC OA full text | [AWS registry: NCBI PMC](https://registry.opendata.aws/ncbi-pmc/) | same registry (S3 buckets) · [NCBI FTP](https://ftp.ncbi.nlm.nih.gov/pub/pmc/) | ungated; JATS XML; per-article CC licenses |
| US patent full text | [google/patents-public-data](https://github.com/google/patents-public-data) | [USPTO bulk data](https://bulkdata.uspto.gov/) · BigQuery `patents-public-data` | public-domain text; weekly XML since 1976, era-varying schemas |
| Amazon Reviews 2023 (items / items+reviews) | [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) | [HF dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | `raw_meta_<Category>` + `raw_review_<Category>` configs; parquet via HF auto-conversion |

### 6.4 Cardinality ceiling (passage-level scaling)

Question: what is the highest primary-embedding cardinality available at
OpenAlex-grade richness? Two-part answer:

1. **At native document granularity, OpenAlex (~250M works) is already the
   ceiling.** The only larger gate-passing corpus of native records is
   Amazon Reviews at review granularity (571M), which gives up hierarchical
   multi-valued texture (T6). Everything bigger at document level is
   web-scale (FineWeb-class, ~25B documents) whose metadata collapses to
   ☑-richness — failing the constraint.
2. **To go higher, make the passage the vector unit** (the same choice
   Cohere Wikipedia embodies; legitimate under the §2 C3 rule when passage
   retrieval is the intended search behavior). Passages *inherit* their
   parent document's full metadata and add passage-local textures
   (section/claim type — a correlated T1; parent-document id — a natural
   T2).

The ladder, gate-passing candidates only:

| Corpus | Vector unit | Cardinality | Richness note |
|---|---|---|---|
| OpenAlex | work | ~250M (~100M w/ abstracts) | ⭐ native ceiling at document level |
| Amazon reviews-as-records | review | 571M | high, no hierarchical T6 |
| PMC OA full text | passage | ~0.6–1B | MeSH-rich, biomedical-only |
| S2ORC full text | passage | **~1–2B** | full S2AG metadata + section type; **recommended** |
| US patent full text | passage/claim | **~2–4B** | CPC-rich, public domain; **absolute ceiling** |

Engineering realities (from the plan's stage model, which scales linearly
in N):

- **Embedding wall-clock is the binding constraint, not storage**: 1.5B
  passages × ~250 tokens ≈ 375B tokens — roughly 1–2 weeks on 8×L40S with a
  0.6B-class embedder (1024-d native), months with an 8B-class model. At
  this tier the small embedder (or an H100 burst) is effectively forced.
- **Storage at 1.5B × 1024-d f32**: ~6.2 TB per vector copy → ~21 TB modeled
  peak → provision ~32–40 TB (striped gp3 stays feasible).
- **Passage corpora structural bonus**: the parent-document hierarchy gives
  a *principled* strata ladder (sample by parent, not by passage) and
  per-parent dedup/grouping semantics that document corpora cannot express.

Recommendation: **US patent full text is the absolute ceiling (~2–4B,
cleanest license at any scale)**; **S2ORC (~1–2B) is the recommended
high-cardinality corpus** — the same scholarly-richness family as OpenAlex
and DOI-joinable to it, so CC0 OpenAlex fields can be grafted onto every
passage. The combined S2ORC×OpenAlex design — entity graph, denormalization
star, and subset constellations — is specified in
[`pvs-s2orc-openalex-union.md`](pvs-s2orc-openalex-union.md).

---

## 7. Recommendations by purpose

- **Best single successor corpus: OpenAlex.** Passes gates at the maximum,
  tops both breadth and depth, CC0 removes every encumbrance question, and
  250M gives headroom well past 15.3M. Cost concentrated in C4=2: we embed it
  ourselves — exactly the plan's §3 Option-B path.
- **Best turnkey (vectors included + rich labels): PubMed+MedCPT** (34M,
  B=9); same idea at 200M scale: S2AG+SPECTER (B=10).
- **Pipeline validation: big-ann YFCC-10M filtered** — C6=4 is the point:
  ingest at 10M and use its published filtered GT as an external oracle for
  `compute prefiltered-knn`, regardless of which corpus we standardize on.
- **Cheapest >100M vectors: Cohere Wikipedia**, accepting D=.50 with
  join-gated upside.
- **Cheapest depth upgrade: stay on the incumbent and join the review side**
  (§5.9): B 7→9, D .55→.82, no license/pipeline change; satisfies the
  `temporal-scatter` layer spec (§4.5) that the items-only corpus fails.
- **Highest primary-embedding cardinality at OpenAlex-grade richness
  (§6.4)**: US patent full text is the absolute ceiling (~2–4B passages,
  public domain); **S2ORC passages (~1–2B) recommended** — same richness
  family as OpenAlex and DOI-joinable to it for CC0 metadata enrichment;
  design in [`pvs-s2orc-openalex-union.md`](pvs-s2orc-openalex-union.md).

---

## 8. Tooling implications

- **Multi-valued fields (T6)**: before any tag-rich corpus is committed,
  verify the M/P path end-to-end for set-valued fields — MNode array typing,
  `analyze survey` statistics over sets (vocab size, per-item cardinality,
  co-occurrence), and IN-predicate generation against them. Every T6 score
  ≥2 in §6.2 is conditional on this. Treat as a Phase-0 check/extension in
  the main plan.
- **Survey extensions for the §4.4 protocol**: cross-field lift (step 3) and
  the neighborhood-lift probe (step 4) are small additions to `analyze`;
  scope them with the first layer that needs T9/T10 at level ≥3.
- **Join-derived fields** (iCite counts, pageviews, Wikidata): joins happen
  in Phase-1 preprocessing; the ordinal-alignment invariant (plan §4 Phase 2)
  is unchanged, but provenance should record the join inputs. Unbuilt joins
  cap FOC at 1 (§4.1).
- **Passage-unit corpora** (S2ORC, PMC OA, patents): Phase-1 preprocessing
  gains a chunker; every passage must carry its parent-document id and
  inherited metadata into M; strata must sample **by parent document**, not
  by passage (else parent bleed across strata boundaries); dedup policy must
  choose passage- vs parent-granularity. Scope with the first passage-level
  layer.
- **`MATCHES` (T7) and `IN` (T6) selectivity calibration**: survey-based
  `generate predicates` currently calibrates comparand quantiles for numeric
  ranges; pattern and set predicates need their own calibration strategies —
  scope when a T6/T7-heavy layer is scheduled.
- **Negative textures** (`NE`, exclusion): near-1.0 selectivity predicates
  invert the F-facet cost profile (passing set ≈ whole corpus); make sure
  `selectivity-max` handling and R-facet sizing account for them before
  generating such layers.

---

## 9. Expansion TODOs

- [x] Scoring rubric — FOC scale (§4.1), aggregates (§4.2), criteria anchors
      (§2). ~~Numeric scoring rubric over C1–C6 with per-purpose
      weightings~~ → remaining: the per-purpose *weightings* over the
      criteria vector (§4.3).
- [ ] Per-testing-layer texture requirements — extend the §4.5 seed specs to
      the full layer catalog (which T-aspects, at which FOC minima and
      selectivity bands, paired vs pooled predicates, per layer).
- [ ] Run the §4.4 measurement protocol on the top candidates (OpenAlex,
      S2AG or PubMed, incumbent+reviews) to convert projected 3/4 scores to
      measured; includes abstract-coverage and field-population measurement
      for OpenAlex/S2AG.
- [ ] MNode/PNode set-valued support audit (§8 first bullet).
- [ ] Negative/exclusion textures (`NE`, `NOT IN`) as first-class aspects or
      as modifiers on T1/T2/T6 — decide placement, then score.
- [ ] Null/missing-value semantics per aspect (extend the plan's price
      sentinel ruling into a general policy).

---

## 10. Sources

- [S2AG datasets API & license](https://api.semanticscholar.org/license/);
  [Semantic Scholar Open Data Platform](https://arxiv.org/pdf/2301.10140)
- [OpenAlex snapshot download guide](https://help.openalex.org/download/download-to-machine);
  [OpenAlex on the AWS Open Data registry](https://registry.opendata.aws/openalex/);
  [OpenAlex overview](https://catalysiseducation.substack.com/p/openalex-the-free-open-and-massive)
- [MedCPT repo](https://github.com/ncbi/MedCPT) (embeddings under
  `ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/`)
- [big-ann NeurIPS'23](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md);
  [Big-ANN results paper](https://arxiv.org/pdf/2409.17424)
- [Cohere Wikipedia embeddings](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3)
- [Discogs Data](https://data.discogs.com/)
- [SE dump restrictions](https://devclass.com/2024/07/30/stack-exchange-restricts-access-to-dump-of-user-contributed-data-as-critics-complain-license-permits-reuse-for-any-purpose/);
  [state of SE dumps](https://search.feep.dev/blog/post/2025-02-20-state-of-stackexchange);
  [SE dump 2025-12-31 torrent](https://academictorrents.com/details/0d1d597fa7809f0e85f127b5eb3088219ddbad39)
- [Re-LAION-5B](https://laion.ai/blog/relaion-5b/)
- [S2ORC corpus](https://github.com/allenai/s2orc);
  [S2ORC paper](https://arxiv.org/abs/1911.02782)
- [PMC article datasets on AWS](https://registry.opendata.aws/ncbi-pmc/)
- [Google Patents Public Datasets](https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data);
  [google/patents-public-data](https://github.com/google/patents-public-data);
  [USPTO bulk data](https://bulkdata.uspto.gov/)
