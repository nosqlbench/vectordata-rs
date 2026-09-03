# SRD — Topic-stratified predicate sampling

**Status:** proposed
**Scope:** metadata enrichment, predicate generation, and the pipeline
steps that produce them. Adds commands to `veks run`; designed to
augment an existing dataset in place rather than rebuild it.

**Depends on**
[metadata-facets-and-layout-namespace.md](metadata-facets-and-layout-namespace.md)
for the M / P / R facet shapes, and
[prefilter-postfilter-facets.md](prefilter-postfilter-facets.md) for
what E and F measure and why the difference between them matters, and
[sysref §13](../sysref/13-metadata-survey.md) for the survey this design
extends. Query-relative pivots for the non-topical families and a
per-pivot control share are deferred to
[srd-predicate-pivot-regimes.md](srd-predicate-pivot-regimes.md).

## 1. Problem

`generate predicates --strategy compound --selectivity 0.001` does not
deliver 0.1%. Measured on tessera at the 110m profile, across 32
predicates sampled from `metadata_results.slab`:

| statistic | value | vs target |
|---|---|---|
| target | 0.100% | — |
| mean | 0.219% | 2.2× looser |
| median | 0.018% | 5.5× tighter |
| max | 1.10% | 11× looser |
| zero matches | 9 of 32 (28%) | unusable |

The spread covers roughly two and a half decades, and better than a
quarter of the predicate set selects nothing at all — a filtered search
over an empty set exercises nothing and contributes no signal.

### 1.1 Cause one: the metadata is paper-level, the corpus is passage-level

Row *i* of `metadata.parquet` carries the parent **paper's** fields,
copied onto every passage of that paper. Only `section` varies within a
paper. tessera has 531,869,985 passages from 17,457,121 papers — a mean
of **30.5 passages per paper**.

So a predicate on `year`, `venue`, `field`, `citationcount` or
`corpusid` draws from 17.5M items, not 531.9M, and each hit brings
~30.5 rows with it. Writing `b` for the block size and `s` for the
per-paper probability:

```
X = b · B,   B ~ Binomial(P, s),   P = 17.46M papers

E[X]   = b·P·s = N·s            ← the mean is still correct
Var[X] = b²·P·s(1−s) ≈ b·E[X]   ← the spread is not
sd(X)  ≈ √(b · E[X])            ← inflated by √30.5 ≈ 5.5×
```

Match counts are also quantised to multiples of ~30.5. A target of 10
matches can only return 0 or 30. **That is the 28% empty rate**, not a
separate phenomenon beside it.

### 1.2 Cause two: the eligible fields are correlated and skewed

Measured over 300,000 metadata rows (9,528 papers):

| field | distinct | empty | level | problem |
|---|---:|---:|---|---|
| `section` | 72,977 | 3.4% | passage | raw headings, no taxonomy |
| `year` | 116 | 0% | paper | correlates with `citationcount` |
| `citationcount` | — | 0% | paper | correlates with `year` |
| `isopenaccess` | 2 | 0% | paper | selectivity floor of 34% |
| `field` | 25 | 22.8% | paper | finest single `Eq` is ~4% |
| `venue` | 4,195 | 21.9% | paper | near-determined by `field` |

`generate_compound_predicate` splits a target as
`per_child = target^(1/arity)`, which is correct arithmetic only under
independence — and the function's own comment concedes *"Independence
is not enforced."* A venue determines its field, so
`field=Biology AND venue=PLoS ONE` is barely more selective than the
venue alone (overshoot); `field=Physics AND venue=PLoS ONE` selects
nothing (empty). Empty string is the most common value in both columns.

### 1.3 Cause three: the constraint that rules out the obvious fix

Uniform hash buckets and random hyperplanes over the embedding solve
the distribution problem completely and exactly. They also produce
predicates like `WHERE bucket_1000 = 37`, which nobody would ever
write. A benchmark whose filters are visibly reverse-engineered from
the measurement is difficult to defend as evidence about anything.

**TS-1.** Every predicate's selectivity must be known before any query
runs, not estimated from a formula and hoped for.

**TS-2.** Every predicate must be a filter a person could plausibly
have meant, expressible in the vernacular of the corpus. A family that
cannot meet this may be retained **only** if it is labelled and
reported as a control.

**TS-3.** The predicate set must cover a stated range of selectivity
decades, with predicates present in every decade.

**TS-4.** Match cardinality must have a floor **above a stated
reliability threshold** (§3.3). Below that threshold the design makes
no cardinality promise, and a thin or absent semantic family is an
expected outcome rather than a defect. The floor is a target the
sampler works to, not an admissibility gate that rejects predicates.

**TS-5.** The design must augment the existing tessera dataset without
re-extracting base vectors or recomputing unfiltered KNN ground truth.

## 2. Why control and credibility are not in tension

Control is a property of how an attribute is **distributed**.
Credibility is a property of what it **means**. The two are
independent, so the resolution is not to choose between them — it is to
engineer the distribution of attributes that already mean something.

**TS-6.** Predicates are drawn from four *families*, spanning a second
axis the current generator ignores entirely: how strongly the filter
correlates with the query.

| family | correlation with query | role |
|---|---|---|
| **topical** | high — the query usually sits inside the filter | headline instrument |
| **structural** | mild | breaks paper-blocking |
| **bibliographic** | near zero | the realistic majority |
| **hash control** | zero by construction | null hypothesis |

**TS-7.** Selectivity alone does not characterise a filtered search.
Two predicates admitting the same fraction pose different problems
depending on *where* that fraction sits relative to the query: a
bibliographic filter thins a neighbourhood roughly uniformly, while a
topical filter either contains it or excludes it wholesale. Results
must therefore be reported per `(family × decade)` cell, never averaged
across families at a fixed selectivity — averaging hides the effect
that E and F exist to expose.

## 3. The sampling model

This is the substance of the proposal: how to obtain reliable decade
coverage from clusters whose sizes we do not control.

### 3.1 Measure, do not balance

**TS-8.** Topic clusters must **not** be size-balanced. Forcing equal
sizes distorts clusters into shapes no topic model would produce,
reintroducing exactly the artificiality this design removes.

**TS-9.** Instead, cluster freely, then record every cluster's exact
member count. Selectivity becomes a *measured table*, read before any
query runs — a stronger guarantee than the current scheme offers even
in principle, and it satisfies TS-1 directly.

The uneven size distribution stops being a defect and becomes the menu
the sampler chooses from.

### 3.2 Profiles are uniform samples, so selectivity is scale-free

tessera's base is shuffled with seed 42, and every sized profile is a
**prefix** `[0..N)` of that shuffle. A prefix of a shuffled corpus is a
uniform random sample of it.

**TS-10.** A topic with global selectivity *s* therefore has
`≈ s · N` members in the profile of base count *N*, for every profile.
Selectivity is constant across the ladder; only the absolute count
moves.

This has a consequence that must be designed for rather than
discovered: tessera's profiles span 100,000 to 495,930,736 — a factor
of **4,959**. A predicate at *s* = 10⁻⁴ yields 10 matches at the `100k`
profile and 49,593 at `default`.

### 3.3 The reliability threshold, and the floor above it

Projective reliability is not free, and it is not worth buying at every
scale. A profile of 100,000 passages cannot host a topic that is both
fine-grained enough to be interesting and populous enough to measure —
there is no clustering that makes that true, so requiring it would only
force the design into contortions that serve the small end at the
expense of the large one.

**TS-46.** A **reliability threshold** *N*ᵣ (default **10,000,000**)
divides the profile ladder:

| range | expectation |
|---|---|
| *N* ≥ *N*ᵣ | stratified coverage holds; selectivity is measured and predicates populate every decade above the floor |
| *N* < *N*ᵣ | best effort. Semantic families may be sparse or absent; no cardinality promise is made |

The threshold is configuration, not a constant. It exists so that the
design states where its guarantees apply instead of implying they apply
everywhere.

**TS-11.** For a profile with *N* ≥ *N*ᵣ and configured minimum match
count *M*, the sampler targets predicates satisfying

```
s · N  ≥  M + 3·√M
```

The `3√M` term is binomial headroom: matches are `Binomial(N, s)` with
`sd ≈ √(sN)`, so relative precision is `1/√M` and the bound holds the
realised count above *M* with ~3σ of margin. This is a **target the
sampler works to**, not a gate that rejects predicates — a predicate
falling below it is admitted and its realised count recorded, because
the count is measured either way (TS-9) and a known-small predicate is
more useful than an absent one.

**TS-12.** Predicates are generated **once** into a master set and
*selected* per profile, not sampled per profile — see §3.7, which the
decade ladder makes possible. The R facet remains per-profile, since
match sets differ by window. Above the threshold, the floor implies a
smallest well-supported decade:

| profile | N | floor at M = 100 | decades with support |
|---|---:|---:|---|
| `100k` | 100,000 | — | below threshold; hash family only |
| `1m` | 1,000,000 | — | below threshold; hash family only |
| `10m` | 10,000,000 | 1.3×10⁻⁵ | 10⁻¹ … 10⁻⁵ |
| `100m` | 100,000,000 | 1.3×10⁻⁶ | 10⁻¹ … 10⁻⁶ |
| `default` | 495,930,736 | 2.6×10⁻⁷ | 10⁻¹ … 10⁻⁷ |

**TS-13.** A stratum with no candidates is reported as *unpopulated*
for that profile, with its shortfall recorded (TS-16). Above the
threshold this is a finding worth acting on; below it, it is the
expected case and must not be reported as an error. The number of
decades a profile supports is data about the profile.

**TS-47.** Below the threshold, the **hash family carries the
profile**. Its advantage there is availability, not precision: a bucket
of any modulus *K* can be defined on demand, whereas no clustering can
be asked to produce a topic of arbitrary size. The statistical spread
is identical — a bucket and a topic of the same selectivity both give
`Binomial(N, s)` — so the hash family does not rescue small-*N*
variance, and this design does not claim it does. What it provides is a
predicate set that exists and has exactly known selectivity at every
scale, so the small profiles still exercise the filtered-search path
rather than being skipped.

**TS-48.** Results from sub-threshold profiles must be labelled as
such. They measure filtered-search mechanics at small scale; they do
not carry the `(family × decade)` comparison of TS-7, because the
families are not all present.

### 3.4 Two-dimensional stratified sampling

**TS-14.** The sampler draws from strata indexed by
`(family, decade)`. For a profile with base count *N*, target decade
set *D* (those above the floor), and per-cell count *c*:

```
for each family f in {topical, structural, bibliographic, control}:
  for each decade d in D:
    candidates = { p : family(p) = f  and  sel(p) ∈ [d/√10, d·√10) }
    draw c predicates uniformly from candidates, seeded
```

The half-decade band `[d/√10, d·√10)` tiles the selectivity axis
without gaps or overlap, so every candidate falls in exactly one bin.

**TS-15.** Drawing is seeded and reproducible: the same dataset, seed
and configuration must produce the same predicate set.

**TS-16.** When a `(family, decade)` cell has fewer candidates than
*c*, the sampler takes what exists and records the shortfall in the
generation report. It must not substitute predicates from another cell,
which would silently distort the family mix the results are grouped by.

### 3.5 Reaching decades below what clusters provide

Level-3 topics average ~10⁻⁴. Deeper decades need conjunction — but
*deliberate* conjunction with a factor known to be independent, rather
than the blind ANDing that caused §1.2.

**TS-17.** A topical predicate may be conjoined with **at most one**
bibliographic predicate to reach a lower decade. The two are drawn from
different families whose independence is a design property, not an
assumption: a paper's citation percentile within its year carries no
information about which embedding cluster its passages fall into.

**TS-18.** The realised selectivity of such a conjunction is
**measured**, not predicted. The candidate is admitted to whichever
decade band its measured selectivity lands in — so an unexpected
correlation demotes the predicate to a different bin rather than
corrupting the bin it was intended for.

This is also the more believable form. *"Passages about grid
integration of renewables, from well-cited papers"* is a more natural
query than either conjunct alone.

### 3.6 Query placement is a parameter, not an accident

**TS-19.** For a topical predicate, whether the query vector lies
inside or outside the filtered topic determines whether the case
measures high-recall filtering or the adversarial extreme. This was
left to chance. The relation is a property of the **pair** — the query
at ordinal *i* and the predicate at ordinal *i*, which is what the
filtered ground truth evaluates (TS-156) — never of the predicate
against the query set as a whole. The generator must decide it per
pair from the query's own descent (TS-137), record it beside the pair,
and draw a configured mix of both.

**TS-156.** **One predicate per query ordinal.** The filtered facets
pair query *i* with predicate *i*, so `predicates.slab` holds exactly
as many records as there are queries, in query order, and record *i*
is *the* predicate of query *i*. A predicate set of another size
leaves queries without a predicate, which the filtered KNN would
silently evaluate as an empty match set. The stratification is
therefore over **query slots**: the slots are shared equally by the
families and split over the decades by the per-cell spec (TS-159);
each cell draws its distinct predicates and pairs them with slots, a
distinct predicate repeating only when the cell's pool is smaller than
its slots; and every slot no cell can fill takes a control predicate,
so no query is without one. The families namespace records the
distinct predicate a record carries so results can be grouped by
predicate as well as by query.

**TS-163.** **The families share the query slots evenly**, the control
family included, in one global hash space: 2,500 of tessera's 10,000
queries carry a hash predicate. That is an even share for diagnostics,
not a weighting of the null case against the semantic case it
accompanies; weighting the control share per semantic pivot, and
pivoting the structural and bibliographic families on the query's own
passage, are deferred to
[srd-predicate-pivot-regimes.md](srd-predicate-pivot-regimes.md)
until the dataset inventory can hold several predicate regimes as
distinct artifacts.

**TS-157.** **Placement is decided per pair.** A topical cell whose
queries' topics are known fills an in-topic slot by pairing a predicate
with a free query whose descent lands in the predicate's topic, and an
out-of-topic slot with one whose descent does not; the mix is the
configured placement, in-topic first, and a share that cannot be met
falls to the other side and is reported. Topical cells fill before the
other families, finest decade first, because they are the ones that
need particular queries — a level-3 topic holds about one query in ten
thousand. The other families pair with queries in seeded draw order,
which is the zero correlation they exist to measure (TS-6), and their
records say so by carrying no placement.

### 3.7 Sampling for a decade ladder

This dataset's profiles are the decade detents only — `1…9 × 10^k` from
100k to 400m, 31 in all, plus `default` at the full 495,930,736. Other
strata (`mul`, `fib`, `linear`) are dropped.

**TS-49.** The ladder has two regions under TS-46: **18 of the 31
detents sit below the 10M threshold** (100k–900k and 1m–9m), and 13 sit
above it (10m–90m, 100m–400m), plus `default`. More than half the
ladder is therefore hash-family only, which is a deliberate consequence
of TS-46 and not a gap to be closed.

**TS-50.** Predicates are drawn **once**, into a master set generated
against the full base. A profile's set is the subset whose selectivity
clears its floor — a *selection*, not a fresh sample.

This is what makes the ladder legible. Because selectivity is scale-free
(TS-10), predicate *p* means the same thing at 10m and at 400m, so a
difference in measured behaviour between two profiles is attributable to
scale alone. Sampling independently per profile would confound the two,
and there are 31 profiles to confound. It also matches the layout
already on disk: `predicates.slab` is a single shared artifact under
`profiles/base/`, not a per-profile one.

**TS-51.** Per-profile applicability is **computed, not sampled**: a
predicate applies to profile *N* when it clears that profile's floor
(TS-11). The master set spans decades 10⁻¹ … 10⁻⁷ so that the top of
the ladder is served; at 10m only 10⁻¹ … 10⁻⁵ apply.

**TS-52.** The nine mantissa steps within a decade share the predicates
of their decade anchor. Consecutive detents differ by at most 2× (and
by 1.125× at the top of a decade), which is a scaling question, not a
predicate-behaviour question. **The mantissa sweep measures how cost
grows with N; the decade anchors measure how it varies with the
predicate.** Generating different predicates per mantissa step would
add no information and would break TS-50's comparability.

### 3.8 Predicate counts are the R-facet storage dial

**TS-53.** The R facet's size obeys

```
bytes per base ordinal  =  4 · Σ selectivity over all predicates
```

Validated against tessera: 10,000 predicates at a measured mean
selectivity of 0.00219 give a mass of 21.9 and predict 87.6 B/base
against **87.45 B/base measured — 0.17% agreement**.

The consequence is that **the coarsest decade dominates**. Ten
predicates at 10⁻¹ carry the same storage mass as ten thousand at 10⁻⁴.
A flat count per cell spends nearly all of the budget at the top of the
range, where predicates are also the least interesting.

**TS-54.** Per-cell counts are therefore configured **per decade**, not
flat, and tapered upward as selectivity falls. For the decade ladder,
with four families:

| per-cell count | mass | B/base | R over the ladder |
|---|---:|---:|---:|
| flat 50 | 22.2 | 88.9 | 177 GB |
| flat 10 | 4.4 | 17.8 | 35 GB |
| tapered 10 / 20 / 50… | 5.0 | 20.1 | **40 GB** |

The tapered row uses 10 predicates at 10⁻¹, 20 at 10⁻², and 50 at each
decade from 10⁻³ down — the same total storage as a flat count of 10,
but five times the predicates where the resolution matters. Against
today's 174.5 GB for the same ladder, that is a **4.3× reduction**
alongside the coverage improvement.

**TS-159.** The mass is a property of the **records**, not of the
distinct predicates: with one record per query (TS-156) it is
Σ over records of selectivity, and 10,000 records apportioned in the
proportions above would put 356 of them in the 10⁻¹ decade and spend
285 GB there alone. The per-cell spec is therefore read as **slots per
decade per family** with two forms: numbers alone are weights, and a
list with `rest` names absolute counts for the decades it numbers and
shares the remaining slots equally among the `rest` decades. `tapered`
is `10, 20, 50, rest…`: the three coarsest decades keep exactly the
counts above, whose predicates are the expensive ones, and every
decade below 10⁻³ takes an equal share of the rest — for tessera's four
families and 10,000 queries, 605 slots each — where a record costs
next to nothing. The mass is then 5.3, 21 B/base, and 42 GB over the
ladder: the budget of the table, with every query paired.

## 4. Derived columns, encoding, and adjunct data

### 4.1 The columns

**TS-20.** The following are added to the metadata facet. Everything
except the topic levels is derivable from data already on disk.

| column | MNode type | wire B | derived from | family |
|---|---|---:|---|---|
| `topic_l1` | text label | 29 | embeddings, k-means | topical |
| `topic_l2` | text label | 41 | embeddings, k-means | topical |
| `topic_l3` | text label | 47 | embeddings, k-means | topical |
| `section_class` | text label | 29 | `section` heading | structural |
| `citation_percentile` | int16 | 24 | `citationcount` × `year` | bibliographic |
| `passage_position` | int16 | 21 | `ordinal` ÷ paper length | structural |
| `word_count` | int16 | 15 | `char_end − char_start` | structural |
| `sample_bucket` | int32 | 20 | hash of `(corpusid, ordinal)` | control |

**226 bytes per record** — 120 GB across 531,869,985 passages, taking
the M facet from 85.7 GB to **206 GB**.

**TS-21.** `citation_percentile` is computed **within publication
year**, so it means "well-cited for its age". This is both the more
meaningful quantity and the one that breaks the `year`/`citationcount`
correlation identified in §1.2.

**TS-22.** `section_class` normalises the raw headings (72,977 distinct
in the 10M pilot; 82,165,334 in the 550M corpus, measured 2026-09-02)
to a small closed taxonomy (`introduction`, `background`, `methods`,
`results`, `discussion`, `conclusion`, `references`, `other`).

**TS-23.** `passage_position`, `word_count` and `section_class` are
passage-level, so predicates over them draw from 531.9M independent
items rather than 17.5M blocked ones. §1.1's variance inflation and
quantisation do not apply to this family.

### 4.2 Stored natively as MNode fields

An MNode record carries every field's name on the wire — `[name_len
u16][name][tag u8][value]` — so a field's name is repeated in all
531,869,985 records. Of the 226 bytes above, 119 are names.

**TS-55.** The derived columns are nonetheless stored as ordinary MNode
fields in the M facet. The overhead is real and is accepted.

The trade is favourable because of where the cost lands. **The M facet
is shared across every profile** — sized profiles window into one slab
rather than copying it — so this is paid once, not 31 times:

| artifact | size | multiplicity |
|---|---:|---|
| `base_vectors` | 2,033 GB | once |
| `metadata_content` | 85.7 → **206 GB** | once, windowed |
| `metadata_results` (R) | ~40 GB | per profile |

120 GB against a 2 TB base, on a volume with 4.2 TB free, to avoid a
second storage format and everything that would have to understand it.

**TS-56.** Coded values are stored as their **label**, not as a code —
`section_class = "methods"`, `topic_l2 =
"photovoltaic-grid-integration"`. This is what makes the encoding
decision and the credibility requirement (TS-2) the same decision: the
value a predicate compares against is the value on the wire, so no
dimension table, decode step or catalog stands between the stored data
and the query a person would write.

Scanning cost is unaffected: `check_condition_raw` compares text as a
length check plus a byte comparison, without allocating.

### 4.3 Rejected: packed columns with dimension tables

**TS-57.** Storing each column as a `ScalarPacked` facet with a
code→label dimension table was considered and rejected. It costs 15
bytes per record instead of 226 — 8 GB instead of 120 — and it is the
wrong trade here.

What it would have required: a layout-facet extension describing where
each field's values live; a resolver presenting the union of MNode
fields and column facets as one logical record; label→code resolution
at predicate-compile time and code→label decoding for every report; and
changes to the survey, the generator and the evaluator. Every consumer
of the dataset would need to understand a second storage format to read
a field.

**112 GB is not worth that on a dataset whose base facet is 2 TB.** If
a future corpus makes the ratio unfavourable — many more columns, or a
much smaller base — this is the design to revisit.

### 4.4 What ships, and what it looks like

The statistical machinery lives in the **sampler**, not in the
predicate. Density conditioning happens when a predicate is *selected*
(§3); it leaves no residue in the predicate that is *published*.

**TS-58.** A retained predicate **record** is an ordinary PNode over
named fields with literal comparands, and nothing else. It does not
record the cluster size that qualified it or the stratum it filled —
that belongs in the generation report (TS-34). What a predicate *is*
and what it *does* is annotated separately, in a sibling namespace
(§4.5), which a consumer reading filters can ignore entirely.

So the predicate facet a consumer reads contains only this:

```
topic_l2 = 'photovoltaic-grid-integration'
year >= 2020 AND section_class = 'methods'
citation_percentile >= 90
topic_l2 = 'transient-flow-in-pipelines' AND citation_percentile >= 90
passage_position < 20
word_count BETWEEN 120 AND 200
```

Each renders through the existing vernacular codecs as a `WHERE` clause
in whichever dialect a consumer wants, because these are PNodes like
any other.

**TS-59.** The one exception is the control family, whose predicates
are not prosaic and are not meant to be:

```
sample_bucket < 16778
```

It must be labelled as a control wherever the predicate set is
published, so that a consumer cannot mistake it for a realistic query
(TS-2, TS-48). Everything else in the corpus is a sentence somebody
could have meant, and reads as one without reference to this document —
which is the acceptance test in TS-45.

### 4.5 Marking the predicate family

TS-7 requires results be reported per `(family × decade)` cell, so the
grouping key has to survive into the published dataset. Nothing marks
it today.

**TS-60.** The family is recorded in a **sibling namespace of
`predicates.slab`**, one record per predicate ordinal: record *i* of the
annotation namespace describes predicate *i* of the content namespace.

The facet spec already models this — `StandardFacet::namespaces()`
returns `["layout", ""]` for the layout facet — so
`MetadataPredicates` gains an annotation namespace beside its default
one. The reader exists too: `RecordFacet::namespace(name)` opens a named
namespace as an ordinal-addressed record facet.

**TS-61.** Annotation records are MNode. The name tax that mattered at
531,869,985 passages (§4.2) is irrelevant across a few thousand
predicates.

**TS-62.** The annotation carries what a predicate **is** and what it
**does**, not why it was selected:

| field | type | notes |
|---|---|---|
| `family` | text | `topical` / `structural` / `bibliographic` / `control` |
| `selectivity` | float64 | the fraction measured when the predicate was admitted |
| `topic_level` | int | 1 / 2 / 3, topical family only |
| `conjunct` | bool | topical only: a topic conjoined with one bibliographic qualifier (TS-17) |
| `query_placement` | text | `in-topic` / `out-of-topic` (TS-19), topical only, present only when queries were given |

**The decade is derived, not stored.** It is `⌊log₁₀ selectivity⌋`, so
recording the continuous value and computing the bin keeps the artifact
describing the predicate rather than the experiment that produced it —
which is what reconciles TS-7 with TS-58.

**TS-63.** Absence of the namespace means **one unlabelled family**, not
zero predicates. Every dataset written before this design has no such
namespace and must keep working unchanged. This follows the `forms`
namespace precedent exactly, where absence means one implicit form
rather than none.

**TS-64.** The annotation must live in the same file as the predicates
it describes. A sibling facet or a side-car report can be lost,
truncated or updated independently; a namespace in the same slab cannot
drift from the records it annotates.

#### Rejected alternatives

**TS-65.** *Annotate inside the PNode.* A PNode has slots for a field,
an operator and comparands, and nothing else. The only way to attach a
label within one is to conjoin a synthetic predicate — which changes
what the predicate matches and therefore its selectivity. It would
corrupt the very quantity the annotation exists to report.

**TS-66.** *A separate facet.* Same information, an extra declaration,
an extra file, and the ability for the two to disagree. Namespaces
exist for precisely this relationship.

**TS-67.** *Encode by ordinal range* — predicates 0–999 topical,
1000–1999 structural. Implicit, undiscoverable, and broken by any
change to per-cell counts (TS-54), which are configuration.

**TS-68.** *The generation report alone.* The report is where the
selection process belongs (TS-34), but it is a side artifact. A
consumer holding only the published dataset could not group results by
family, and TS-7 would be unsatisfiable from the dataset itself. The
dataset must be self-describing.

### 4.6 Adjunct provenance and diagnostics

§4.5 solves one instance of a general problem. Stated once, so the next
one does not get argued from scratch.

**TS-69.** The M facet is the **query surface**. Intermediate models,
causal traces and provenance must not be added to it. Two reasons, and
the second is the one that matters: every added field inflates all
531,869,985 records for readers who will never query it (§4.2), and
anything present in M is something a predicate can name — so putting
diagnostics there blurs the boundary between what the benchmark
*measures* and what it *explains*.

**TS-70.** They are nonetheless worth keeping. A dataset that can
explain an anomalous result is worth more than one that cannot, and the
cost of retaining a trace is far lower than the cost of regenerating a
corpus to recover it. Adjunct data is **retained in the dataset,
ordinal-aligned**, so it joins to the records it describes without a
key.

**TS-71.** Two carriers, chosen by scale and cardinality:

| relationship | carrier | example |
|---|---|---|
| 1:1 with a slab's records, thousands of rows | **namespace** in that slab | predicate families (§4.5) |
| 1:1 with passages, hundreds of millions | **adjunct facet**, ordinal-aligned | cluster-assignment margin |
| dataset-level, one value | **namespace** or dataset attribute | embedding model revision |

**TS-72.** Adjuncts are **optional**. Absence must be legal and must
not fail a read — the `forms` precedent again (TS-63). A consumer that
wants only the benchmark reads M, P, R, G, D, E, F and nothing else.

**TS-73.** Adjuncts are **never required** to evaluate a predicate or
reproduce ground truth. They explain; they do not define. Anything that
a result *depends* on belongs in a facet, not an adjunct.

**TS-74.** The packed-column encoding rejected in TS-57 is the right
encoding **here**. It was wrong for the query surface because it would
have put a second storage format between predicates and the fields they
name; nothing writes a predicate against a diagnostic, so that objection
does not apply, and the density does:

| per-passage cluster-margin trace | per record | total |
|---|---:|---:|
| as MNode fields on M | 64 B | 34.0 GB |
| as an `mvecs` adjunct facet, dim 2 | 8 B | **4.25 GB** |

Eight times smaller, and it keeps M unchanged. `ScalarPacked` would be
denser still at 4 B, but it has no float extension (`u8`…`i64` only), so
it would mean a fixed-point scale convention every reader has to know.
The self-describing form is worth 2 GB here. → TS-89.

**TS-75.** Adjuncts worth retaining for this design:

- **Topic centroids** — the fitted model, 10,310 × 1024 f32 = 42 MB.
  Already required by TS-26 for reproducibility; this is where it lives.
- **Cluster-assignment margin** — distance to the assigned centroid and
  to the runner-up, per passage. This is the trace that explains
  *filter crispness*: a passage near a boundary belongs to its topic by
  a hair, so a topical filter includes it almost arbitrarily. When
  pre-filter and post-filter results diverge in a way the selectivity
  does not account for, this is the column that says why.
- **Predicate generation trace** — per predicate, the candidate pool it
  was drawn from and the stratum it filled. The residue TS-58 keeps out
  of the predicate record, kept where it can still be read.
- **Embedding provenance** — model and resolved revision. tessera's are
  currently recorded only in a markdown file beside the corpus, so
  nothing in the dataset says which weights produced its vectors. That
  is exactly the gap this requirement exists to close.

#### Rejected

**TS-76.** *Keeping traces outside the dataset* — in a report, a
notebook, or a file beside the corpus. This is the status quo for
tessera's embedding revision, and its own provenance notes concede the
failure mode: "nothing in dataset.log or runlog.jsonl would show the
difference" if another host resolved the model tag differently. A trace
that does not travel with the data it describes is a trace that will
eventually describe something else.

### 4.7 Only a passage's own data is ever attached to its ordinal

**TS-158.** Every field a base ordinal carries in the M facet is either
the passage's **source metadata** — the row of the metadata table that
is row-aligned with its embedding — or a **derivative computed from
that passage's own data**: its embedding (the topic labels, through the
same centroid model that will place the queries), its text (the word
count), its heading (the section class), its position in its paper and
its paper's records (the citation percentile, ranked among the papers
of the same year). Nothing is synthesised, sampled from elsewhere, or
assigned by position. The one column that is not a property of the
passage, `sample_bucket`, is a seeded hash whose only purpose is the
control family (TS-115), is named for what it is, and is never read by
a semantic predicate (TS-149). A pipeline change that attaches a column
to an ordinal by any other route than the passage's own data violates
this document.

**TS-162.** The same holds across the query ↔ predicate boundary: the
predicate at ordinal *i* is evaluated against those authentic fields
and its relation to query *i* is decided from query *i*'s own
embedding (TS-157), or it is drawn independently of the query by
design and recorded as such (TS-6). Both are measurements of something
real; a predicate affixed to a query by position alone is neither, and
TS-161 refuses it.

**TS-165.** **The queries' own metadata rows are a facet**,
`profiles/base/query_metadata.slab`, in query order: row *i* is the
source-order metadata row of the passage query *i* was derived from,
carried across the bridge by the same shuffle over `[0, query_count)`
that `extract-queries` applies to the vectors. It is produced by
`transform extract` exactly as the base metadata is (TS-130), so it
cannot disagree with it, and it is retained (TS-84): it is what makes
any query-relative statement checkable without the generator.

**TS-166.** **Every pair is labelled against its own query.** When the
query metadata is given, the generator evaluates record *i*'s predicate
against row *i* and records `query_in_filter` in the families
namespace for every family, not only the topical one: whether the
query's own passage would pass its own filter. For a topical single
predicate this agrees with the descent-based placement except at a
margin the perturbed query vector crossed (TS-89); for the other
families it is the one relation the pair has to the query, and it is
recorded rather than assumed. The generation report tallies both sides
per cell. Selection is unchanged by the label: the structural and
bibliographic draws stay independent of the query (TS-6), so the
label measures how often independence happens to coincide with the
query's own passage, which is the base rate any query-relative regime
(the deferred SRD) would be read against.

## 5. Topic hierarchy

**TS-24.** Three levels, branching ~10 / ~30 / ~33, giving 10 / 300 /
10,000 clusters and mean selectivities of ~10⁻¹ / ~3×10⁻³ / ~10⁻⁴.

**TS-25.** Centroids are fitted on a sample (default 5M passages) and
the full corpus is assigned by descent through the hierarchy: 10
comparisons, then ~30 within the chosen branch, then ~33 within that —
about 73 dot products of width 1024 per passage, rather than 10,000.

**TS-26.** Assignment is a pure function of the vector and the fitted
centroids, so it is deterministic and re-runnable, and the centroids
are published as an artifact so a third party can reproduce the
labelling.

**TS-27.** Each cluster is labelled from the most distinctive terms of
its member passages. Labels are cosmetic to the measurement and carry
the entire credibility argument of TS-2; they must be generated, and
they must not be load-bearing for correctness.

## 6. Pipeline commands to add

Four new commands, one command extended (`analyze survey`, which gains
the census), one reused (`transform extract`, for the margin adjunct),
and two ordinal spaces that every one of them must be explicit about.
Existing `evaluate-predicates` is unchanged.

**TS-130.** Two ordinal spaces exist in this dataset, and each artifact
below belongs to exactly one:

| space | population | rows | what lives in it |
|---|---|---:|---|
| **source order** | every passage as chunked | 531,869,985 | `_base_all.fvecs`, `_metadata.parquet`, `passages.parquet` — row-aligned |
| **base order** | shuffled, minus 10,000 queries and 35,929,249 duplicates | 495,930,736 | everything under `profiles/base/` |

`extract-metadata` is the bridge: `transform extract` applies
`shuffle.ivecs` over `[query_count, clean_count)` to carry a
source-order artifact into base order. **Topic assignment and
enrichment happen in source order**, because they feed
`convert-metadata` (TS-32) and because the join to the passage table is
a row-aligned join in that order and in no other. **Adjuncts published
under `profiles/base/` are in base order**, because "ordinal-aligned"
(TS-70) means nothing unless the ordinal is the facet's own. An adjunct
computed in source order crosses the bridge the same way M does — the
margin through the mvec mode of `transform extract` — never by a
mapping of its own.

**TS-131.** Consequently `compute topics` **assigns** over
`_base_all.fvecs`, not `profiles/base/base_vectors.fvecs`, and emits one
assignment per source row.

**TS-150.** The **fit** reads a different file: the first `sample-size`
rows of the shuffled base, `profiles/base/base_vectors.fvecs`. A prefix
of a shuffled facet is a uniform sample of the base population (TS-10),
it excludes the duplicates and queries the base excludes, and it is one
sequential read of 20 GB. *Rejected:* sampling the source-order file,
whose prefix is one corner of the corpus and whose uniform sample is
five million scattered 4 KB reads across 2.18 TB. The command keeps
both forms — `sample-order: prefix` for a shuffled facet, `strided` for
one in corpus order — so a dataset without a shuffled base still fits
on a uniform sample, just more slowly.

### 6.1 `compute topics`

**TS-28.** Fits a hierarchical clustering over a base facet and emits
per-passage assignments.

Full surface in §10.1.

**TS-29.** The assignment pass reads the base facet sequentially and
must not require the whole facet resident — the same incremental
contract every other reader here holds.

**TS-77.** The same pass emits the **cluster-assignment margin**
(TS-75): the distance to the assigned centroid and to the runner-up,
per passage, as a packed `f16` adjunct facet (TS-74). Both distances are
already computed to make the assignment; recovering them later would
cost a second pass over 531,869,985 × 1024 floats. Optional output —
absence is legal (TS-72) — but it is nearly free at the point where the
numbers exist and unaffordable anywhere else. It is written in source
order under `.cache/` and carried into base order by `transform extract`
(TS-130); the retained facet is the extracted one.

### 6.2 `analyze survey`: the census pass

**TS-30.** The survey gains a third, **exhaustive** pass that emits the
exact member count of every value of every enumerable predicate target
over the full base population — each topic at each level, each
`section_class`, each `year`, and the distributions behind the range
families — structured so that the fitting phase (§11) reads selectivity
by lookup and never scans M. This is the table the sampler selects from
(TS-9) and the artifact that makes TS-1 true. The survey is the
dataset's content-and-distribution assay and the one artifact the
generator and the evaluator already read; exact counting is an
extension of it, not a sibling of it (D-19).

**TS-124.** **The survey's sampled passes cannot supply this, and cannot
be configured into supplying it.** Three of their properties are
structural, not defaults to raise:

| survey property | value | consequence for selectivity mapping |
|---|---|---|
| `samples` | 100,000 | 0.019% of the corpus |
| `low-card-threshold` | 64 | exact counts only at or below 64 distinct values |
| `top-k` | 64 | above that, only the 64 most frequent values are kept |

tessera's own pipeline sets `samples: 10000`, a tenth of the default,
so the first row reads 0.0019% there.

Of the 10,310 topics needing an exact count, the sampled passes can
enumerate 64. `venue` already demonstrates the failure on tessera
today: 4,195 distinct values, classified `MidCard`, and only its heavy
hitters retained.

Sampling is the harder limit. A topic's expected membership in a
100,000-row sample, against the decades this design targets:

| selectivity | expected members in sample | relative error |
|---|---:|---|
| 10⁻³ | 100 | 10% |
| 10⁻⁴ | 10 | 32% |
| 10⁻⁵ | 1 | 100% |
| 10⁻⁶ | 0.1 | invisible |
| 10⁻⁷ | 0.01 | invisible |

The design's floor reaches 10⁻⁷ (§3.3). A sampled pass cannot see three
of those decades at all, and TS-1 requires the selectivity be *known*,
not estimated.

**TS-125.** Raising the sampled passes' caps is rejected rather than
untried. It would disable the sampling that makes the survey
affordable, widen every bounded tracker, and still run a dozen measures
— trigram statistics, character-class mixes, semantic probes,
cross-field pair analyses — that exact counting does not need, over
every record. Worse, Pass 1 decides a field's *regime* from a sample
and Pass 2 selects its measures from that verdict (sysref §13.6), so a
misclassification there cannot be corrected by a larger cap later.
Passes 1 and 2 stay what they are. The census is a **third pass with
its own measures**, exhaustive by definition and bounded by declaration
rather than by a sampled verdict.

**TS-139.** **What the census pass counts is declared, with a default.**
Three declaration kinds, each an option of the step (§10.2) and
therefore a YAML key:

| declaration | form | yields |
|---|---|---|
| census field | `census: auto` (default), `none`, or a field list | an exact value table per field |
| hierarchy | `hierarchy: topic_l1>topic_l2>topic_l3` | an exact nested tree with a count at every node |
| pair | `census-pair: topic_l3:citation_percentile` | an exact joint table for the pair |

`auto` censuses every field whose Pass 1 regime is `Constant`, `Binary`,
`LowCard` or `MidCard` — fields the sample already showed to be
enumerable — so every survey gains exact counts for its enumerable
fields without being told. A **listed** field is censused regardless of
regime; this is how `topic_l3` (10,000 distinct, `HighCardOrUnique` by
sample) enters, and `auto` may be combined with a list. Hierarchies and
pairs are declared only: they are the parts of the census whose cost is
a product, and nothing in a sample says which products a consumer
wants.

**TS-140.** Two exact measures, chosen by wire encoding:

| measure | fields | report |
|---|---|---|
| `ExactValueCensus` | text, bool, and any non-integer enumerable | `population`, `distinct`, `counts: value → n` ordered by `n` descending |
| `ExactIntegerHistogram` | integers | `population`, `min`, `max`, `counts[]` dense from `min` to `max` |

Value keys use the survey's canonical value key — the same rendering
`ExactFrequencyTable` uses and the generator already decodes — so the
substitution TS-144 promises holds byte for byte. The histogram is
dense and ordered so a range's selectivity is a prefix-sum difference —
exact, with no interpolation (TS-128). An integer field gets both
tables. Each is bounded by `census-cap` (default 65,536 distinct values
or histogram width). For a field the operator **listed**, more distinct
values than the cap is an **error**: the operator asserted the field is
enumerable, the data disagrees, and a truncated table would be a wrong
selectivity presented as an exact one. For an `auto` field it is a
`Warning` finding and the field leaves the census — the sample
misjudged it, which the survey reports rather than fails on. A
histogram whose range alone exceeds the cap is dropped with a warning
while the value table stands: enumerability is the gate, density a
convenience.

**TS-141.** A **hierarchy** is counted as exact path tuples — each
`(l1, l2, l3)` combination that occurs, with its count — folded into a
tree whose every node carries its own count. The pass also **verifies
nesting**: every value at level *k*+1 must occur under exactly one path
at level *k*. A violation is an `Error` finding and fails the step,
because a declared hierarchy is an invariant of the data that produced
it — for tessera, of `enrich-metadata` — and a tree that is not a tree
would give the sampler two selectivities for one label. The tree is
what makes "multi-layer" concrete for the fitting phase: a topic at any
level is a node, its selectivity is the node's count, and a query's
placement (TS-117) at any level is a walk up from its L3 assignment.

**TS-142.** A **pair** is counted as an exact joint table: `a_values`,
`b_values`, and a dense row-major `counts[|a|][|b|]`. Its memory is the
product of two cardinalities, so the pass enforces `pair-cells-cap`
(default 4,194,304 cells — 32 MB of `u64`) and errors when a table would
exceed it. For tessera the nine declared pairs — each topic level
against `citation_percentile`, `year` and `isopenaccess` — total ~2.25M
cells, dominated by `topic_l3 × year` (1.16M) and
`topic_l3 × citation_percentile` (1.0M). This is what makes a
conjunction's selectivity **exact rather than estimated** (TS-18,
TS-116): `topic_l3 = X AND citation_percentile ≥ t` is row *X* summed
over the columns at or above *t*.

**TS-126.** The survey moves to read **`profiles/base/metadata_content.slab`**
— the M facet, after `extract-metadata` — rather than the source-order
slab it reads today. Passes 1 and 2 are indifferent to which; the
census is not (TS-132). One pass then covers topics and every other
derived field at once, keyed by the **label** a predicate compares
against (TS-56), so no code-to-label reconciliation stands between the
survey and the sampler.

**TS-132.** Counting M is a **population** choice, not a convenience. M
holds the base population: the source rows minus the 10,000 that became
queries and the 35,929,249 that `prepare-vectors` found to be
duplicates — 495,930,736 of 531,869,985 (TS-130). That is the population
every predicate is evaluated over and every profile is a prefix of
(TS-10). A census of the source parquet would count 6.75% of rows that
no predicate can ever match, and there is no reason to expect them to
be spread evenly over topics or sections — a boilerplate passage
repeated across papers is exactly the kind that clusters — so it would
carry a bias the base does not have, and TS-43 would then be checking
the sampler against the wrong denominator. Walking 495,930,736 records
is one pass, and with TS-148's extraction it costs less than a minute
on tessera — 57.8 s measured over the current M facet on 128 threads,
against 615.6 s for the same count done single-threaded with full
record decoding — which is a small price for counting the right thing.

**TS-127.** The census is exact and **global**. Per-profile counts are
not censused: selectivity is scale-free (TS-10), so a profile's expected
count is `s · N`, and its *realised* count is known exactly from the R
facet after evaluation, which is where TS-43 checks it.

**TS-143.** **The report is structured for the fitting phase.**
Selectivity is `count / source.total_records`, with the denominator
recorded once in the report and never a stored fraction, so nothing
rounds. Where each candidate of §11 reads:

| candidate | reads | selectivity |
|---|---|---|
| `field = v` | `fields[field].measures.ExactValueCensus.counts[v]` | `n / N` |
| `field BETWEEN lo AND hi` | `fields[field].measures.ExactIntegerHistogram.counts` | prefix-sum difference `/ N` |
| topic at level *k* | the node in `hierarchies` | node count `/ N` |
| `a = x AND b ≥ t` | `pair_census[a:b].counts[x][t..]` | row-sum `/ N` |
| `sample_bucket < m` | nothing — uniform by construction (TS-115) | `m / K`, verified by TS-43 |

`hierarchies` and `pair_census` are new top-level sections of
`survey.json` beside `fields` and `cross_field` (sysref §13.8), absent
when nothing was declared. A field's census measures live in its
`measures` map like every other measure, keyed by kind, so the map's
key-routed deserialisation gains two variants and nothing else changes
shape.

**TS-144.** A censused field's `cardinality_regime` is replaced by
**`Censused { exact_distinct }`** and its profile marked `censused`.
Pass 1's verdict was an estimate; the census is the fact, and a
consumer choosing an operator family from the regime must see the
fact. Its `presence` is exact too — the generator divides a value's
count by the field's presence, so both must come from the same
population — and its sampled cardinality measures
(`ExactFrequencyTable`, `HeavyHitters`, `HyperLogLog`) are **removed**
rather than left beside the census: two answers, one of them an
estimate, is worse than one. Every consumer that accepts an
`ExactFrequencyTable` accepts an `ExactValueCensus` in its place — the
existing strategies' eligibility checks read measure presence — and
this contract is what lets `eq` and `compound` gain exact selectivity
without change.

**TS-145.** The census pass declares its memory to the governor from
the caps (sysref §13.11.1) — `census-cap` × entry size per censused
field plus `pair-cells-cap` × 8 B — before it begins, and drives the
standard progress sink by page, because it is a full scan of the
largest record facet in the dataset and must say so while it runs. The
number of `auto` fields is unknown until Pass 1, so the declaration
allows for a fixed 32 of them beyond those listed; a budget that
cannot grant the ceiling fails the step before the pass rather than
during it.

**TS-148.** **What the pass scans, and how.** It reads every page of M
in ordinal order — the same reader Passes 1 and 2 use, without the page
stride that samples them — and walks each record once with the
zero-allocation MNode scanner, extracting **only the declared fields**
into a per-page slot array and one text arena: two allocations per
page, none per record. Per record the consumer then does one thing per
declaration: looks up each censused field's value in that field's
table, folds each hierarchy's value tuple into its tree, and
increments each pair's cell. Values are **interned on first sight** —
a label maps to a small integer id in a per-field table, looked up by
the borrowed arena slice — so the hierarchy and pair accumulators
index by id and the per-record cost is one hash probe per declared
field, not a string copy; canonical keys are built once per distinct
value and written out only when the report is serialised. Extraction,
which is the cost, runs on the governor's `threads` in parallel across
pages; counting applies extracted pages strictly in page order on one
thread, so interned ids, first-seen parents and report order never
depend on scheduling and the report is identical for any thread count.
The pass needs Pass 1 only for `auto` (TS-139) and nothing from Pass 2, so
a survey configured with `census: none` is exactly today's survey, and
one with declarations adds one sequential read of M — 206 GB after
enrichment, a single slab under the 1 TB shard cap, opened as the
survey opens any slab today.

**TS-146.** sysref §13 is updated in the same change: the pass (§13.4),
the two measures (§13.5.3), the hierarchy and pair census (§13.7), the
report sections (§13.8) and the options (§13.9). The survey has a
system reference, and a survey extension the reference does not
describe is not finished.

### 6.3 `transform enrich-metadata`

**TS-31.** Joins the derived columns of §4 onto the metadata source,
emitting an enriched parquet in source order (TS-130) that
`convert-metadata` then reads in place of the original. Inputs, all
row-aligned or keyed by `corpusid`: `metadata.parquet` (the paper
fields), `passages.parquet` (`ordinal`, `text`), `parents.parquet`
(`passage_count` and `row_start` per paper), the topic assignments, and
the topic labels.

**TS-135.** The labels input is **optional**. Without it, enrichment
writes the positional label (TS-102) for every cluster, so the whole
chain can be exercised — and the census taken — before labelling has
run. Re-running enrichment once labels exist is what the option's
`Input` role is for: M, the survey with its census tables, and the
predicates are all marked stale, which is correct, because every
topical comparand changes.

**TS-78.** The command is **not a pure row-wise map**, and the plan must
say so rather than let an implementation discover it. Five of the six
derivations are row-local given a table; one needs an aggregate over
the papers first:

| column | derivation | pass |
|---|---|---|
| `word_count` | whitespace tokens of `text` (TS-106) | row-local |
| `sample_bucket` | `hash(corpusid ‖ ordinal) mod K` | row-local |
| `section_class` | heading → class lookup | row-local, given the table |
| `topic_l1/l2/l3` | assignment lookup by row, then code → label | row-local, given assignments and labels |
| `passage_position` | `ordinal ÷ passage_count` | row-local, given `parents.parquet` |
| `citation_percentile` | rank of `citationcount` within `year` | **needs the per-year distribution over papers** |

`parents.parquet` already carries `passage_count` and `row_start` for
each of the 17,457,121 papers, so the paper-length aggregate an earlier
draft planned is a 200 MB lookup, not a pass. The one real aggregate is
the per-year citation distribution over **papers** (TS-104): one
columnar read of `corpusid`, `year`, `citationcount`, taking the first
row of each paper — rows are grouped by paper, which is what `row_start`
records — into 116 per-year tables. Then a single pass over the three
parquet files in row order emits the enriched rows.

**TS-136.** That single pass reads the `text` column, which is nearly
all of the 241 GB `passages.parquet`. This is the dominant cost of
enrichment and is paid for `word_count` alone. It is accepted because
TS-106 wants the unit a person filters in, and it is stated here so the
estimate in TS-123 is not read as a parquet-to-parquet copy.

**TS-79.** The heading → `section_class` mapping is published as an
adjunct (TS-71). It is a judgement call applied 531,869,985 times across
72,977 distinct headings, and anyone assessing a `section_class`
predicate will want to audit it. Headings the table does not cover fall
to `other`; an unmapped heading is never an error.

**TS-80.** The hash keyed for `sample_bucket` is over
`(corpusid, ordinal)` — the **passage**, not the paper. Keying it on
`corpusid` would reproduce the paper-blocking of §1.1 in the one family
that exists to be free of it.

**TS-149.** `sample_bucket` is an **auxiliary** element of M.
Enrichment always computes it — it is the one column whose distribution
is exactly uniform by construction, and it costs one hash per row — but
**no semantic family may reference it**. Topical, structural and
bibliographic predicates, and the conjunctions of TS-17, are drawn from
the labelled and derived fields alone; the control family is the only
consumer of the hash, and it is labelled as such wherever it is
published (TS-59). The default predicate set is therefore
semantic-first: the hash carries the sub-threshold profiles (TS-47) and
supplies the labelled null hypothesis above them (D-3), and nothing a
person would read as a query depends on it.

**TS-32.** Enrichment happens **upstream of** `convert-metadata`, so
the M facet, the survey, and everything downstream flow from it without
special-casing.

### 6.4 `generate predicates --strategy stratified`

**TS-33.** A third strategy alongside `eq` and `compound`, implementing
§3.4.

Full surface in §10.4.

**TS-34.** The command emits a generation report alongside the
predicate facet: per-cell counts, realised selectivity distribution,
and any shortfalls from TS-16.

**TS-81.** The same command writes the **families namespace** (§4.5)
into `predicates.slab` in the same operation. It is the only step that
knows a predicate's family, and writing the two together is what makes
TS-64's co-location hold by construction rather than by discipline.

**TS-82.** The per-predicate portion of the generation report — the
candidate pool a predicate was drawn from and the stratum it filled —
is additionally retained as an ordinal-aligned adjunct (TS-75), so the
selection process travels with the dataset rather than only with the
run that produced it. Its record carries `cell` (`family:1e-d`), `pool`
(candidates in that cell), `source` (which census table: hierarchy,
histogram, values, pair, or control), `expected_count` (the exact
matches over the census population), `vernacular` (the predicate
rendered) and `backfill` (whether the record took a control predicate
because no cell could fill its slot, TS-156), so a reader can check any
predicate's claim against the R facet without the generator. The
families record carries, besides the family and selectivity, the
distinct `predicate` index (TS-156), and for a topical record its
`topic` label, `query_placement` and the query's own `query_topic` at
that level (TS-157), so the pair's relation is auditable from the
record alone.

**TS-167.** A verification step, `verify predicate-strata`, holds the
generator's claims against the answer keys rather than trusting them:
one record per query and both namespaces the same length (TS-156,
TS-111); at the profile whose base is the census population, every
censused record's realised match count equals its recorded
`expected_count` exactly, and a control record's lies in its band
(TS-115); at every other profile above the reliability threshold the
realised selectivity lies in the record's half-decade band and no
record is empty (TS-42, TS-43), while below the threshold only the
control family is held to a non-empty result; and every
`query_in_filter` label re-derives from the query's own row (TS-166).
Realised counts are the results records' lengths, so the pass reads
each profile's R facet once and decodes nothing. It writes a report
per profile and per family and fails the pipeline on any violation,
which is what TS-121 asked for: the acceptance is checked against R,
not against the generator.

### 6.5 Recording embedding provenance

**TS-83.** The model identity and **resolved** revision that produced
the base vectors are recorded in the dataset, as the attributes
`model`, `model_revision` (the commit the request resolved to, read
from the hub cache's `snapshots/<sha>/` layout) and
`model_revision_requested` (what the build asked for, so a floating tag
is visible as such). For tessera this is a back-fill: the values exist
only in a markdown file beside the corpus, and its own notes concede
that nothing in `dataset.log` or `runlog.jsonl` would show the
difference if another host resolved the model tag differently (TS-76).
The back-fill route is `state set` with `attribute: true`, which writes
a dataset attribute rather than a pipeline variable and refuses a key
the dataset does not define. For future builds the embed command emits
all three at the point it resolves the weights, which is the only point
where they are known to be correct.

### 6.6 What is cache and what is retained

**TS-84.** Intermediates that can be recomputed live under `.cache/`
and are removable by `veks prepare cache-gc`. Retained adjuncts live in
the dataset and are not:

| artifact | where | why |
|---|---|---|
| topic assignments | `.cache/` | consumed by `enrich-metadata`; the values then live in M |
| enriched metadata | `.cache/` | consumed by `convert-metadata`; the values then live in M |
| survey, with census tables | `.cache/` | recomputable from M in one pass; the selectivity that matters is retained per predicate (TS-62) |
| topic labels | dataset | the comparand every topical predicate names (TS-56) |
| topic centroids | dataset | reproducing the labelling requires them (TS-26) |
| topic model report | dataset | levels, seed and sample, without which the centroids cannot be re-fitted (TS-152) |
| assignment margin | dataset | explains filter crispness; a second pass to recover |
| heading → class table | dataset | a judgement call worth auditing |
| query metadata | dataset | every query-relative claim is checked against it (TS-165) |
| families namespace | in `predicates.slab` | TS-64 |
| generation trace | dataset | selection provenance |
| embed provenance | dataset | irrecoverable once the host cache turns over |

**TS-85.** An adjunct must never be written under `.cache/`. It would
be correct until the first `cache-gc`, and its loss would be silent —
the failure mode TS-76 rejects, arrived at by a different route.

### 6.7 `compute topic-labels`

**TS-133.** Labelling is its **own command**, not a mode of `compute
topics`. It reads text where `compute topics` reads vectors, it needs
memory for term tables where the fit needs the GPU, and it is the one
phase whose output can be absent without blocking anything (TS-102,
TS-135). Coupling it to the fit would make the largest GPU step re-run
whenever a labelling parameter changed.

**TS-138.** Labelling reads a **seeded subset of row groups**, not a
sample of rows. `passages.parquet` is 508 row groups of ~1.05M rows, so
2,000 uniformly sampled members of a typical L3 cluster fall in ~20
distinct groups, and a sample that touches every group is a full read of
241 GB (TS-136). Instead the command reads `row-groups` groups chosen by
seed (default 64, an eighth of the file), visits each group's rows in a
seeded order so a cluster's cap does not fill from one corner of the
group, and accepts a passage for its cluster at every level whose cap
has room, up to `sample-per-cluster` per cluster. Acceptance is decided
sequentially because it depends on the caps filling; the accepted
passages are then tokenised in parallel and their terms added in the
same order, so the labels are identical for any thread count. The cap
is a ceiling, not a floor: a cluster at 10⁻⁶ meets ~67 members in 67M
rows, which is enough for term statistics, and a cluster that meets
fewer than `min-sample` (default 20) is given a positional label rather
than a label fitted to noise.

Full surface in §10.5.

## 7. Augmenting tessera in place

**TS-35.** The augmentation must not invalidate `extract-base` or any
`compute-knn` step. This holds because unfiltered KNN depends on
`count-base` and `extract-queries` only — not on metadata — so under
the `config-only` provenance selector the 85 completed KNN profiles
stay fresh.

What re-runs is the metadata and predicate chain:

```
compute-topics             NEW    fit on the shuffled base prefix (TS-150),
                                  then one CPU pass over _base_all.fvecs,
                                  source order (TS-131) -> assignments +
                                  margin (.cache), centroids + model
                                  report (dataset)
compute-topic-labels       NEW    seeded row-group sample of passages.parquet
                                  (TS-138) -> topic_labels.slab (dataset)
enrich-metadata            NEW    one paper-level aggregate, two lookups,
                                  one pass over the three parquet files
                                  -> .cache/metadata_enriched.parquet
record-embed-provenance    NEW    back-fill; dataset attribute only
convert-metadata           re-run source repointed at the enriched
                                  parquet; ~300 s (measured)
extract-metadata           re-run ~207 s per pass (measured)
extract-topic-margin       NEW    transform extract, mvec mode, with the
                                  shuffle: margin from source order to
                                  base order (TS-130)
survey-metadata            re-run moved after extract-metadata, onto M;
                                  passes 1–2 as today plus the exhaustive
                                  census pass with tessera's declarations
                                  (TS-126, TS-139)
generate-predicates        re-run --strategy stratified
evaluate-predicates        re-run per profile   ← the expensive one
```

**TS-36.** Steps are appended to `upstream.steps` with `after:`
declaring the order above: `compute-topics` → `compute-topic-labels` →
`enrich-metadata` → `convert-metadata`; `extract-topic-margin` after
`compute-topics` and `generate-shuffle`; `survey-metadata` after
`extract-metadata`; `generate-predicates` after `survey-metadata` as
today. Exactly three existing step definitions change:
`convert-metadata`'s `source` moves from `_metadata.parquet` to the
enriched parquet (TS-32); `survey-metadata` gains `after:
extract-metadata`, its `source` becomes the M facet, and it carries the
census declarations (TS-139); and `generate-predicates`' options change
for the new strategy. Each change is what correctly marks that step and
its dependents stale; nothing above `convert-metadata` in the metadata
chain, and nothing in the vector chain, is touched.

**TS-37.** The R facet must be regenerated for every profile. Its size
scales linearly with base count at 87.45 B/base under the current
0.001 configuration; at the lower selectivities this design targets it
falls proportionally. The existing per-profile `metadata_results.slab`
files and their `.cache/*.predkeys.slab` segments become garbage and
should be removed by `veks prepare cache-gc` before the run to reclaim
space.

**TS-169.** **Removing a profile invalidates nothing but itself.**
The per-profile answer keys are computed one profile at a time in size
order, and the runner orders them with a chain from each profile's
step to the previous profile's. That chain is sequencing, not
dependence: a profile's neighbours' answer keys do not change because
it was added, removed or recomputed. So the chain is carried as
`sequence_after`, which orders the run and contributes nothing to
provenance, and the staleness comparison is made over the upstreams a
step declares *now*, so a recorded node that still names a sequencing
edge from an older build, or a neighbour since removed by
`veks prepare cleanup-profiles`, compares equal. *Measured on tessera,
2026-09-03:* dropping the 36 profiles of the removed `linear` stratum
had marked the nine largest KNN answer keys stale — 128mi through
`default` — purely through the chain; with this, all 49 remain fresh.

**TS-170.** **A source rewritten in place is a different source.**
Repointing `convert-metadata` at the enriched table rewrites
`${cache}/metadata_all.slab` under its old path (TS-32); the extract
that carries it into base order resumes from cached partitions when
it can, and those partitions must be keyed on the source's
**identity** — its provenance address, from the sidecar its producer
wrote or from its size and mtime — and on the output they belong to,
never on the path alone. *Measured on tessera, 2026-09-03:* the first
run after enrichment resumed both partitions cached from the
un-enriched slab, produced a base facet byte-identical to the old one
(85,697,052,998 bytes) beneath a 209 GB source, and the survey
failed on the missing `topic_l3` — the failure that caught it. The
extract now records both identities and keeps one cache per output; a
facet produced by the earlier behaviour is not detected by
freshness, since its step's record is intact, and must be removed by
hand for the step to run again.

## 8. Artifact register

Every artifact this design introduces, named, located and formatted.
Nothing below is inferable from §§4–6, and an implementation cannot
start without it.

**TS-86.**

| artifact | path | order (TS-130) | format | lifetime |
|---|---|---|---|---|
| topic assignments | `.cache/topic_assign.u16vecs` | source | u16 xvec, dim = levels (TS-151) | cache |
| topic centroids | `profiles/base/topic_centroids.fvecs` | by cluster | `FloatXvec` f32, dim 1024 | retained |
| topic model report | `profiles/base/topic_centroids.json` | — | JSON (TS-152) | retained |
| topic labels | `profiles/base/topic_labels.slab` | by cluster | `Slab`, MNode | retained |
| cluster margin, as computed | `.cache/topic_margin_all.mvecs` | source | `FloatXvec` f16, dim 2 | cache |
| cluster margin | `profiles/base/topic_margin.mvecs` | base | `FloatXvec` f16, dim 2 | retained |
| enriched metadata | `.cache/metadata_enriched.parquet` | source | parquet | cache |
| section-class map | `profiles/base/section_class_map.slab` | by heading | `Slab`, MNode | retained |
| survey, with census tables | `${cache}/metadata_survey.json` | counts over base | JSON, sysref §13.8 | cache |
| query metadata | `profiles/base/query_metadata.slab` | query | `Slab`, MNode (TS-165) | retained |
| predicate families | `predicates.slab` ns `families` | by query (TS-156) | `Slab`, MNode | retained |
| generation trace | `predicates.slab` ns `generation` | by query (TS-156) | `Slab`, MNode | retained |
| strata verification | `${cache}/verify_predicate_strata.json` | per profile | JSON (TS-167) | cache |
| embed provenance | `dataset.yaml` attributes | — | yaml | retained |

The census tables are cache rather than retained by TS-84's rule: they
are recomputable from M in one pass, exactly, and the selectivity each
admitted predicate was measured at is retained where it matters — in
the families namespace (TS-62). They live inside `metadata_survey.json`,
which the pipeline already keeps under `${cache}` for the same reason.
The centroids and labels are keyed by cluster, not passage — record *i*
of each is cluster *i* in level order — so they carry no ordinal space
and are the same file whichever side of the bridge reads them.

**TS-87.** The retained adjuncts are declared as **ordinary views with
non-standard keys**, not as new `StandardFacet` variants.

Conformance leaves a view whose key is not a standard facet alone, so
this works without touching the spec. It is also the right category:
the facet spec exists to say what a dataset *must* have and what shape
it must take, and TS-73 states that nothing depends on an adjunct. A
single-letter facet code is a scarce namespace that should not be spent
on optional diagnostics. The cost is that `veks check` will not validate
them, which is the correct trade for artifacts whose absence is legal.

**TS-88.** `predicates.slab` gains **two** namespaces, not one:

| namespace | holds | why separate |
|---|---|---|
| `families` | what a predicate *is* (TS-62) | read by anyone grouping results |
| `generation` | why it was *selected* (TS-82) | the residue TS-58 keeps out of the record |

Kept apart so a consumer can read the family without also reading the
experimental design. They join the `schema` and `survey` namespaces
the slab already carries — the generation template and the survey the
predicates were drawn from — so a stratified `predicates.slab` holds
five namespaces. `StandardFacet::MetadataPredicates::namespaces()`
changes from `[""]` to `["families", "generation", ""]`, and each name
is a published constant beside `SCHEMA_NAMESPACE` and
`SURVEY_NAMESPACE`.

**TS-89.** The cluster margin is one `mvecs` facet of dim 2 — the
distance to the assigned L3 centroid and to the runner-up — rather than
two scalar facets. Both numbers are read together or not at all, and
`ScalarPacked` has no float extension.

**TS-90.** No artifact in this register is required to evaluate a
predicate, compute ground truth, or run the benchmark. A consumer that
reads only M, P, R, G, D, E and F is complete and correct (TS-73).

## 9. Algorithms

### 9.1 Clustering

**TS-91.** **Spherical k-means.** The vectors are unit-normalised
(measured: max |1−‖v‖| = 2.99×10⁻⁸), so cosine similarity is the inner
product and assignment is an argmax of `v · c`. Centroids are
re-normalised to unit length after each update; without that step the
centroids drift off the sphere and the argmax stops corresponding to
cosine.

**TS-92.** **Initialisation is k-means++ over the fitted sample**,
seeded. Random initialisation on a corpus this skewed leaves empty
clusters, and an empty cluster is a topic with no members and no label.

**TS-93.** **Convergence** is a cap on iterations (default 50) or a
mean centroid movement below a threshold (default 10⁻⁴ cosine),
whichever comes first. The cap is not a failure — a clustering that has
not fully converged still partitions the space, and TS-9 measures what
it actually produced rather than trusting it.

**TS-94.** **Determinism under threading.** Summing vectors to form a
centroid is a floating-point reduction, and reduction order changes the
result. Fitting must therefore use a fixed reduction order independent
of thread count and scheduling — a deterministic tree reduction over a
fixed partition of the sample, not an unordered atomic accumulation.
Without this, TS-26's "pure function of the vector and the fitted
centroids" holds for assignment but not for the fit, and a re-run on a
different machine produces different topics.

**TS-95.** An empty or single-member cluster at any level is **split
from the largest sibling** rather than left in place, up to a bounded
number of repair rounds. A cluster of one is not a topic and would
occupy a stratum slot that no predicate can usefully fill.

### 9.2 Assignment and margin

**TS-96.** Assignment descends the hierarchy: argmax over 10 L1
centroids, then over the ~30 L2 children of that branch, then over the
~33 L3 children of that. About 73 inner products of width 1024 per
passage rather than 10,310.

**TS-97.** Descent is greedy and therefore **not equivalent to a flat
argmax over all 10,000 L3 centroids**. A passage near an L1 boundary can
be assigned to an L3 cluster that is not its global nearest. This is
accepted — it is what makes assignment affordable — and it is
*measurable*, which is what the margin facet is for. It must be stated
so nobody later reads a topic as "the nearest cluster".

**TS-98.** The margin records the two L3 distances at the point of
assignment: to the chosen centroid and to the best alternative among its
siblings.

### 9.3 Labelling

**TS-99.** Labels come from a **class-based TF-IDF** over each
cluster's member passages: term frequencies aggregated per cluster,
weighted against their frequency across all clusters at the same level,
top terms joined with hyphens into a slug.

**TS-100.** Labelling reads `passages.parquet`, not the metadata — the
passage text is not in M and never will be (TS-69). It runs on a
**sample** of each cluster's members (default 2,000, drawn from a seeded
subset of row groups — TS-138), because term statistics converge long
before the member list is exhausted.

**TS-101.** Labels must be **unique within a level**, since TS-56 stores
them as the predicate's comparand and two clusters sharing a label would
be two different filters that read identically. Collisions are broken by
appending the next distinguishing term, then by ordinal.

**TS-102.** A cluster whose label cannot be generated — no usable terms
— gets a stable positional label (`l3-04187`). It remains a valid
predicate target; it is simply not a believable one, and TS-16's
shortfall reporting is where that surfaces.

### 9.4 Derived-column derivations

**TS-103.** `section_class`. The raw heading is lowercased, stripped of
leading numbering (`3.`, `iii.`, `A)`) and trailing punctuation, then
matched against an ordered prefix table to one of `introduction`,
`background`, `methods`, `results`, `discussion`, `conclusion`,
`references`, `other`. First match wins; order is significant because
`results and discussion` must not match `results` before the compound
rule is tried. The table is published (TS-79).

**TS-104.** `citation_percentile`. The rank of a paper's
`citationcount` **within its publication year**, expressed as an integer
0–99 over *papers*, not passages — otherwise papers with many passages
would dominate their own percentile. Ties take the **midpoint** rank, so
the large mass of zero-citation papers maps to a single value rather
than being spread arbitrarily. A year with fewer than 100 papers still
produces valid percentiles; the resolution is simply coarser.

**TS-105.** `passage_position`. `⌊100 · ordinal / passages_in_paper⌋`,
so 0–99. A single-passage paper yields 0, which is correct: its one
passage is at the start.

**TS-106.** `word_count` is whitespace-delimited tokens of the passage
text, not `char_end − char_start`. Character span is a proxy that
diverges with language and markup, and the field exists so a person can
filter on "long passages" in the unit they think in. It is computed in
enrichment's pass over `passages.parquet` (TS-78, TS-136), not in
labelling's, which reads only a subset of the file (TS-138).

## 10. Command surfaces

Complete option tables. `role` is the `OptionRole` the runner uses —
`Input` participates in the "input newer than output" staleness check,
`Config` is provenance-bearing, `Output` is what `check_artifact`
inspects.

### 10.1 `compute topics`

| option | role | req | default | notes |
|---|---|---|---|---|
| `base` | Input | yes | — | `_base_all.fvecs`, **source order** (TS-131); accepts a series (SH-35) |
| `sample` | Input | no | `base` | facet to fit on; the shuffled base with `sample-order: prefix` (TS-150) |
| `sample-size` | Config | no | `5000000` | rows fitted |
| `sample-order` | Config | no | `strided` | `prefix` for a shuffled facet, `strided` for corpus order (TS-150) |
| `levels` | Config | no | `10,30,33` | branching per level |
| `iterations` | Config | no | `50` | convergence cap (TS-93) |
| `tolerance` | Config | no | `1e-4` | mean centroid movement |
| `seed` | Config | no | `42` | k-means++ and sampling |
| `normalize` | Config | no | `true` | unit-normalise before fitting and assigning (TS-91) |
| `centroids` | Output | yes | — | `topic_centroids.fvecs`, every level in order |
| `output` | Output | yes | — | `.cache/topic_assign.u16vecs`: one record per base vector, one code per level (TS-151) |
| `margin` | Output | no | — | `.cache/topic_margin_all.mvecs`, source order; omit to skip (TS-72) |
| `model` | Output | no | beside `centroids`, `.json` | the model report (TS-152) |

**TS-107.** `check_artifact` is Complete when the centroid file holds
`Σ levels` records of the base's dimension, the assignment facet holds
one record per source row with one code per level (TS-131, TS-151), the
margin — if declared — holds one dim-2 record per source row, and the
model report parses with the configured levels. Any one alone is
insufficient: centroids without assignments is a fit that never
finished, and assignments without centroids cannot be reproduced or
extended.

**TS-151.** The assignments are **one facet**, `topic_assign.u16vecs`,
a u16 xvec whose dimension is the number of levels, rather than one
packed file per level. The runner anchors a step's freshness on one
`output`, a reader wants every level of a row in one record, and a
leaf code determines its ancestors anyway; three files would be three
ways to be inconsistent. Codes are positional per level — level *l*'s
code `c` is child `c mod k` of level *l*−1's `c div k` — so the tree
is implicit in the branching.

**TS-152.** A **model report** is written beside the centroids,
`topic_centroids.json`, and retained: levels, seed, the sample and how
it was taken, iteration cap and tolerance, the kernel, and per level
the cluster count, empties, runs converged, largest final movement and
repairs. The centroid file carries none of this, and TS-26's
reproducibility promise needs all of it.

**TS-108.** Declares `mem` and `threads` to the governor. The fit holds
`sample-size × 1024 × 4` bytes — 20 GB at the default — and must
request it rather than discover it.

### 10.2 `analyze survey` — census options

Added to the existing surface (sysref §13.9). Nothing existing changes.

| option | role | req | default | notes |
|---|---|---|---|---|
| `census` | Config | no | `auto` | `auto`, `none`, or a field list; `auto` may be combined with a list (TS-139) |
| `census-cap` | Config | no | `65536` | distinct values or histogram width per field (TS-140) |
| `hierarchy` | Config | no | — | comma-separated `a>b>c` declarations (TS-141) |
| `census-pair` | Config | no | — | comma-separated `a:b` declarations (TS-142) |
| `pair-cells-cap` | Config | no | `4194304` | joint-table cells per pair (TS-142) |

For tessera: `source` becomes the M facet (TS-126); `census:
auto,topic_l3`; `hierarchy: topic_l1>topic_l2>topic_l3`; and the nine
`census-pair` declarations of TS-142.

**TS-109.** For every censused field the report carries, per value, the
exact member count over the base population; the fraction is derived
(TS-143). That fraction is the `selectivity` the sampler stratifies on
(TS-14) and the value written into the families namespace (TS-62).

**TS-128.** For the numeric range families — `citation_percentile`,
`passage_position`, `word_count` — the `ExactIntegerHistogram` is the
census, so a range predicate's selectivity is a sum over bins rather
than an interpolation from a quantile sketch. All three have small
integer domains (0–99, 0–99, and word counts bounded by the chunker at
230), so exactness is affordable.

**TS-129.** `check_artifact` for the survey step is Complete when the
report's schema version carries the census sections, every declared
field, hierarchy and pair is present, and for every censused field
`population + missing` equals `source.total_records`. A field whose
counts do not sum to the population has silently dropped values, and a
sampler reading it would stratify on a distribution that does not
exist.

### 10.3 `transform enrich-metadata`

| option | role | req | default | notes |
|---|---|---|---|---|
| `metadata` | Input | yes | — | source parquet |
| `passages` | Input | yes | — | `passages.parquet`: `ordinal`, `text` |
| `parents` | Input | yes | — | `parents.parquet`: `passage_count`, `row_start` |
| `assignments` | Input | yes | — | `topic_assign.u16vecs`, source order (TS-131, TS-151) |
| `labels` | Input | no | positional | code → label, from `compute topic-labels` (TS-135) |
| `paper-column` | Config | no | `corpusid` | metadata column identifying the paper |
| `section-column` | Config | no | `section` | metadata column holding the heading |
| `year-column` | Config | no | `year` | metadata column holding the year |
| `citations-column` | Config | no | `citationcount` | metadata column holding the citation count |
| `buckets` | Config | no | `16777216` | *K* for `sample_bucket`, 2²⁴ (TS-115) |
| `seed` | Config | no | `42` | hash seed |
| `output` | Output | yes | — | enriched parquet, source order |
| `section-map-out` | Output | no | beside `output`, `section_class_map.slab` | every distinct heading with its class and count (TS-79) |
| `report` | Output | no | beside `output`, `.json` | rows, papers, years, distinct headings, share classed `other` |

**TS-154.** The prefix table itself (TS-103) is a constant of the
command and appears in its documentation; what is published beside the
output is its **outcome** on this corpus — one record per distinct
heading with the class it received and how many passages carry it —
because that, not the rules, is what an auditor of a `section_class`
predicate needs to read. `check_artifact` is Complete when the enriched
table holds exactly as many rows as the metadata and carries every
derived column, typed as §4.1 says, for the assignments' depth.

**TS-155.** The map runs in parallel over the passage table's row
groups — each worker reads one group's `ordinal` and `text`, the
aligned metadata rows by row range whatever the metadata's own row
groups are, and the aligned assignments — and one consumer writes the
enriched rows in group order, so the output is identical for any
thread count. Workers are bounded independently of the governor's
thread count because each holds a row group of text.

*Measured on tessera, 2026-09-02 (128 cores, 32 workers, outputs in a
scratch directory):* 531,869,985 rows enriched in 10 min 11 s wall
(583.6 s inside the command) at ~1.5 M rows/s, peak resident set
101 GB; 17,457,121 papers over 231 distinct years; 82,165,334 distinct
headings, of which the prefix table classes 67.5 % of passages as
`other`. The published heading map (TS-154) is therefore 8.9 GB —
larger than the enriched table's own increment — because most
headings are singletons. Whether the map should keep every heading or
only those carrying at least two passages is an open question for the
dataset owner; the command writes what TS-154 says until that is
decided.

**TS-110.** Fails if the assignment count does not equal the metadata
row count. A silent off-by-one here mislabels every passage after the
divergence and would surface only as inexplicable benchmark results.

### 10.4 `generate predicates --strategy stratified`

| option | role | req | default | notes |
|---|---|---|---|---|
| `survey` | Input | yes | — | schema, distributions **and** the census tables (TS-143) |
| `base-count` | Config | no | the census population | *N*, for the floors in the report |
| `families` | Config | no | all four | which families, in output order; the semantic three never read the hash (TS-149) |
| `topic-fields` | Config | no | the survey's first hierarchy | topic fields, outermost first |
| `bibliographic-fields` | Config | no | `citation_percentile,year,isopenaccess` | censused paper-level fields |
| `structural-fields` | Config | no | `section_class,passage_position,word_count` | censused passage-level fields |
| `control-field` | Config | no | `sample_bucket` | the hash field (TS-115) |
| `buckets` | Config | no | `16777216` | its modulus |
| `count` | Config | no | the number of `queries` | records to write, one per query ordinal (TS-156); required without `queries` |
| `decades` | Config | no | `1e-1..1e-7` | target range, or a comma list |
| `per-cell` | Config | no | tapered | a family's slots per decade (TS-159): `tapered`, one weight, or one entry per decade — numbers are weights, with `rest` they are counts |
| `min-matches` | Config | no | `100` | *M* in TS-11 |
| `reliability-threshold` | Config | no | `10000000` | *N*ᵣ (TS-46) |
| `query-placement` | Config | no | mixed | mix of topical pairs whose query lies inside its predicate's topic: `mixed`, `in-topic`, `out-of-topic` or `any` (TS-19, TS-157); needs `queries` |
| `queries` | Input | no | — | the query vectors; record *i* is query *i*'s predicate and placement is decided per pair |
| `query-metadata` | Input | no | — | the queries' own metadata rows in query order (TS-165); every pair gets `query_in_filter` (TS-166) |
| `centroids` | Input | no | — | with `queries` (TS-137) |
| `model` | Input | no | — | with `queries`: the model report, for the branching |
| `labels` | Input | no | — | with `queries`: code → label, so placement is by label |
| `report` | Output | no | beside `output`, `.json` | generation report (TS-34) |

The strategy is a branch of `generate predicates` — `--strategy
stratified` — implemented in its own module, since its surface and its
outputs share nothing with `eq` and `compound` beyond the slab they
write. The other strategies' options are ignored under it.

**TS-111.** `check_artifact` is Complete when the content namespace
holds exactly the expected number of predicates — the number of
`queries`, or `count` (TS-156) — **and** the `families` and
`generation` namespaces hold as many records each. An unequal count
means the pairing with queries or the annotation is off, which TS-64
exists to prevent.

**TS-137.** Query placement needs the fitted model, not only the query
vectors. The queries are perturbed copies of source vectors
(`extract-queries`), so their topics cannot be looked up from the
source-order assignments; each is assigned by the descent of TS-96
against `centroids`, structured by the model report's branching, and
its codes are turned into labels through the label slab, because a
topical predicate names a label. `queries` without all three of
`centroids`, `model` and `labels` is a configuration error, not a
silent omission.

### 10.5 `compute topic-labels`

| option | role | req | default | notes |
|---|---|---|---|---|
| `passages` | Input | yes | — | `passages.parquet`, source order |
| `assignments` | Input | yes | — | `topic_assign.u16vecs` from `compute topics` (TS-151) |
| `model` | Input | yes | — | the model report (TS-152), for the branching per level |
| `text-column` | Config | no | `text` | column holding the passage text |
| `row-groups` | Config | no | `64` | seeded subset read (TS-138) |
| `sample-per-cluster` | Config | no | `2000` | cap on members per cluster, per level |
| `min-sample` | Config | no | `20` | below this, positional label (TS-102) |
| `top-terms` | Config | no | `3` | terms joined into the slug |
| `seed` | Config | no | `42` | row-group and row-order selection |
| `output` | Output | yes | — | `topic_labels.slab` |
| `report` | Output | no | beside `output`, `.json` | row groups read, rows visited, per-level positional and collision counts, sample sizes |

**TS-134.** `check_artifact` is Complete when the slab holds exactly
`Σ levels` records — one per cluster, in level then code order,
matching the model report's branching — and every label is unique
within its level (TS-101). A record carries `level`, `code`, `label`,
`terms` (the ranked terms the slug was cut from), `sample_size` and
`positional`, so a reader can see how much evidence stood behind each
name and whether it is a name at all.

**TS-153.** Terms are lower-cased alphabetic tokens of three to thirty
characters with a short list of function words and academic
boilerplate removed, plus the bigrams of tokens that were adjacent in
the text — a hyphen or apostrophe keeps adjacency, other punctuation
ends a phrase. Class-based TF-IDF scores each term against the level:
its frequency within the cluster's sampled passages, times the log of
one plus the level's mean token count over the term's total count
across the level, so a term that is everywhere scores nothing. The
slug takes the top terms without repeating a word — or a stem: a word
that is a prefix of one already used, or shares its first five
characters, so `robot`, `robots` and `robotic` do not make a label
between them — and a collision within the level extends the slug with
the next ranked terms and, failing that, appends the code.

### 10.6 `verify predicate-strata`

| option | role | req | default | notes |
|---|---|---|---|---|
| `predicates` | Input | yes | — | the stratified facet, with both namespaces |
| `results` | Config | no | `metadata_results.slab` | the results facet's file name under each profile |
| `queries` | Input | no | — | to check one record per query |
| `query-metadata` | Input | no | — | to re-derive every `query_in_filter` label (TS-166) |
| `reliability-threshold` | Config | no | `10000000` | *N*ᵣ (TS-46): above it every family holds its band and is non-empty |
| `output` | Output | yes | — | report: per profile, per family, first violations |

Profiles come from `dataset.yaml`, partition profiles excluded, or
from the `profiles/` directory when the file does not load, with a
sized profile's base count read from its name; the census profile is
the one with no declared count. The step fails on any violation
(TS-167).

## 11. Candidate enumeration

The sampler draws from candidate pools (TS-14). Nothing yet says what
is in them.

**TS-112.** **Topical.** Every cluster at every level is a candidate:
10 + 300 + 10,000 = 10,310, each a node of the hierarchy census
(TS-141) with its exact count. No enumeration cost — the tree *is* the
pool.

**TS-113.** **Bibliographic.** Enumerated from the survey's census
tables (TS-143) as threshold
and range predicates over `citation_percentile`, `year` and
`isopenaccess`. `citation_percentile` is uniform by construction, so a
threshold at *t* has selectivity near `(100−t)/100` and the pool is
dense across the coarse decades; the exact figure comes from the
`ExactIntegerHistogram` (TS-128), which is what tie-rounding (TS-104)
makes necessary. `year` ranges are enumerated from the census's
per-year counts, which are skewed, so their selectivities are read
rather than assumed. The sampled measures contribute field types and
nothing else (TS-124).

**TS-114.** **Structural.** Enumerated over `section_class` values
(eight, censused), `passage_position` ranges (uniform by construction,
census-exact), and `word_count` ranges (from the census histogram).

**TS-115.** **Control.** Generated, not enumerated. `sample_bucket`
holds a seeded hash of the passage reduced modulo *K* = 2²⁴ (§10.3),
uniform by construction, so a target selectivity *s* is the threshold
`sample_bucket < ⌈s · K⌉`, at a resolution of 6×10⁻⁸ — below the
design's finest decade. Distinct predicates for one cell are disjoint
ranges `sample_bucket BETWEEN a AND b` of the same width, drawn seeded.
A single modulus of 1,000 with equality predicates, as an earlier draft
had it, reaches 10⁻³ and nothing finer. The column is **not censused**:
its width exceeds `census-cap` by design, and its selectivity is known
by construction rather than by count — which is exactly why TS-43
verifies it against R like every other family. The only family that can
fill any cell on demand, which is what TS-47 relies on.

**TS-116.** **Conjunctions** (TS-17) are formed only after the single-
field pools are exhausted for a cell, pairing a topical candidate with a
bibliographic one, and **only over pairs the survey censused** (TS-142).
They belong to the topical family — the topic is what they are about,
the qualifier is how they reach a lower decade — and are flagged
`conjunct` in the families namespace. Their selectivity is **read, not
multiplied** (TS-18): for an integer qualifier every threshold
`topic = x AND b ≥ t` is a suffix sum over the joint table's row, for a
boolean or text qualifier every `topic = x AND b = v` is a cell, and the
candidate is admitted to whichever band that lands in. A pair that was
not declared yields no conjunctions and is reported as a shortfall
(TS-16) rather than estimated.

**TS-117.** Query placement (TS-19) requires the query vectors and the
centroids (TS-137). Each query is assigned to a topic at every level by
the same descent as TS-96, and a topical record is labelled `in-topic`
when *its own* query falls in the predicate's topic, `out-of-topic`
otherwise (TS-157); the query's own label at that level is recorded
with it. Without `queries` the generator omits the fields rather than
guessing, and needs `count` to know how many records to write.

## 12. Test strategy

**TS-118.** Each of §9's algorithms is tested against a **synthetic
corpus with known structure** — planted clusters at known sizes,
metadata with known distributions — so expected outputs are derivable
rather than golden. Determinism (TS-94) is tested by fitting the same
sample at 1, 4 and 16 threads and comparing centroids bit-for-bit.

**TS-119.** The sampler is tested without any clustering: given a
synthetic census tables, every `(family, decade)` cell above the floor is
populated, cells below it are reported unpopulated rather than filled,
and the same seed reproduces the same set (TS-15, TS-44).

**TS-147.** The census pass is tested on a synthetic slab with known
counts: every `auto` field's table sums to the population; a listed
field over `census-cap` errors and an `auto` field over it warns and
leaves the census; a declared hierarchy with a planted nesting
violation fails the step; a pair over `pair-cells-cap` errors; a
censused field reports the `Censused` regime; and the report
round-trips through JSON with both new sections present, and with
neither when nothing was declared.

**TS-120.** End-to-end, on a small dataset built by the existing
fixtures: the pipeline runs, the enriched M facet carries all eight
columns, `predicates.slab` carries both namespaces with matching
counts, and every generated predicate returns a non-empty match set at
a profile above the threshold (TS-42). *Realised 2026-09-02* as
`veks/tests/e2e_topic_predicates.rs`: a planted corpus of 200 passages
in 50 papers around six leaf directions with matching vocabulary and
metadata is bootstrapped by `veks prepare bootstrap`, extended exactly
as tessera's definition is, and run through the binary; every retained
adjunct of §8 exists, the M facet carries every derived column, the
query metadata facet has one row per query, the predicate facet one
labelled record per query, the strata verification reports no
violation, and the dataset's own `veks check` passes. Building it
exposed three defects outside the design that would have failed the
tessera build's acceptance: the SQLite oracle declared every column
`TEXT` and so compared numeric ranges lexicographically; the runner
recorded output paths relative to the process's working directory
rather than the workspace; and the manifest projection behind the
extraneous-files check did not read the variables a run records, so
every step naming `${vector_count}` or `${base_count}` was silently
dropped from it. All three are fixed and tested.

**TS-121.** The acceptance criterion of TS-43 — realised selectivity
within the assigned band — is checked **against the R facet after
evaluation**, not against the generator's own estimate. A generator
that mis-measures would otherwise validate itself.

**TS-160.** Pairing is tested with a planted topic model whose leaves
are unambiguous directions and queries built from them, so every
query's topic is known in advance: the predicate facet holds one record
per query, every in-topic record's query lies in its predicate's topic
and every out-of-topic record's does not, the recorded `query_topic`
is the query's own, a `count` that disagrees with the queries is
refused, and without either there is an error rather than a guess.

**TS-168.** The verification is tested on answer keys computed by the
test itself over the fixture rows: a census profile and a sized
profile pass; one ordinal removed from one censused record fails the
census profile with exactly one mismatch; the same rows shifted by one
query fail the label check; a query facet of another size fails the
count check; and the control family, whose count is by construction,
is held to its band rather than to an exact count.

## 13. Work breakdown

Ordered by dependency. Each phase is independently testable and leaves
the tree green.

**Phase 1 — clustering.** `compute topics` (§10.1) with §9.1–9.2:
spherical k-means, deterministic reduction, hierarchical assignment,
margin output, the fit on the shuffled prefix (TS-150) and the
assignment in source order (TS-131). The largest single piece. It runs
on the CPU with the KNN engines' AVX kernels — the fit is about 19
TFLOP and the assignment is bound by reading 2.18 TB, so a GPU would
buy nothing. Testable in isolation against planted clusters (TS-118).

**Phase 2 — labelling.** `compute topic-labels` (§10.5) with §9.3.
Independent of everything after it: positional labels (TS-102, TS-135)
let the chain proceed while this is unfinished, at the price of
re-running the chain once it is.

**Phase 3 — enrichment.** `transform enrich-metadata` (§10.3) with
§9.4, plus the `extract-topic-margin` wiring that carries the margin
into base order (TS-130). One paper-level aggregate, two lookups, one
pass. No new formats.

**Phase 4 — survey census pass.** `analyze survey` extended (§6.2,
§10.2): the exhaustive pass, `ExactValueCensus` and
`ExactIntegerHistogram`, the hierarchy and pair census, the `Censused`
regime, the two report sections, and sysref §13 (TS-146). Larger than
the sibling command an earlier draft proposed, because it must fit the
survey's measure, template, governor and findings machinery rather than
stand beside it — which is the point. It must follow extraction
(TS-126), which is why it is not Phase 2.

**Phase 5 — sampling.** `generate predicates --strategy stratified`
(§10.4) with §11, plus the two namespaces (TS-88). The most intricate
logic and the least I/O.

**Phase 6 — provenance.** Embedding provenance (TS-83) and the
cache/retained split (TS-84). Small, and separable.

**TS-122.** Phases 1–4 can each be run against tessera by hand and
checked without regenerating any predicate: nothing consumes their
outputs until the steps are wired in. The first irreversible act is the
wiring of TS-36 — repointing `convert-metadata` marks M and everything
below it stale — and the first expensive one is Phase 5, whose new
predicate set invalidates `evaluate-predicates` across every profile.

**TS-123.** Estimated cost on tessera, from measured comparable steps:

| phase | pass | estimate |
|---|---|---|
| 1 fit | 5M × 1024, 50 iterations, ≈19 TFLOP | 106.7 s measured on tessera, 128 threads, AVX-512 |
| 1 assign | 531.9M × 73 inner products | 746.1 s measured: one sequential pass over 2.18 TB at ~2.9 GB/s |
| 2 | text of 64 row groups, ~67M passages | ~30 GB read; minutes |
| 3 aggregate | 3 columns of `metadata.parquet`, first row per paper | minutes |
| 3 map | 531.9M rows including the 241 GB `text` column | tens of minutes, I/O bound (TS-136) |
| 3 margin | `transform extract`, mvec, 495.9M × 4 B | comparable to `extract-metadata` |
| 4 | census pass over 495.9M records of M | 58 s measured on tessera, 128 threads (TS-148) |
| 5 | sampling by lookup over the census tables (TS-143) | seconds |
| — | `evaluate-predicates` re-run | per profile; the dominant cost |

## 14. Open questions

**TS-38.** ~~*L3 cluster count interacts with the smallest profile.*~~
**Resolved by TS-46.** The cluster count is now constrained by the
threshold profile rather than the smallest one, and 10,000 L3 clusters
clear it comfortably: at *N* = 10M a mean selectivity of ~10⁻⁴ yields
~1,000 matches, a relative spread of ~3%. The question that forced a
choice before fitting no longer does.

**TS-39.** *Do topic labels need to be good?* §9.3 now specifies a
mechanism — class-based TF-IDF, sampled, uniqueness-enforced — so the
question is no longer "how" but "how good is good enough". **Measured
on tessera, 2026-09-02:** 64 of 508 row groups read in 12 min 51 s;
every level-1 and level-2 cluster labelled from a full 2,000-passage
sample; 9,897 of 9,900 leaves labelled, 3 positional, 135 collisions
resolved by extension. Level 1 reads `theorem-proof-lemma`,
`patients-disease-risk`, `energy-field-model`, `species-water-soil`,
`cells-expression-protein`, and one cluster of non-English passages;
leaves read `biosensors-aptamer-sers`, `controller-robot-tracking`,
`absorbance-spectrophotometer-vis`. Good enough to be read as a query
by someone in the field, which is the bar TS-45 sets; still open is
whether a reviewer outside the field reads them the same way.

**TS-40.** ~~*Should the control family ship?*~~ **Settled by TS-47:
it must.** It is what gives sub-threshold profiles a predicate set at
all, so it is no longer optional — which raises the remaining question
in its place: how to label it so a consumer cannot mistake a control
for a realistic query. The dataset carries no field today that marks a
predicate's family, and TS-7 requires results be grouped by one.

**TS-41.** ~~*Sampling the fit.*~~ **Measured on tessera, 2026-09-02.**
A fit on the first 5M rows of the shuffled base (TS-150), 10 / 30 / 33,
converged at every level with no empty cluster and no repair, in
106.7 s; assigning all 531,869,985 source rows took 746.1 s, bound by
reading 2.18 TB at ~2.9 GB/s. The assignment distribution over the
source population:

| level | clusters | smallest | median | largest | selectivity range |
|---|---:|---:|---:|---:|---|
| 1 | 10 | 39.3M | 50.8M | 74.0M | 7.4×10⁻² … 1.4×10⁻¹ |
| 2 | 300 | 614,819 | 1,657,167 | 3,762,990 | 1.2×10⁻³ … 7.1×10⁻³ |
| 3 | 9,900 | 42 | 48,917 | 623,987 | 7.9×10⁻⁸ … 1.2×10⁻³ |

9,884 of the 9,900 leaves fall in the 10⁻⁴ and 10⁻⁵ decades (4,237
and 5,647); the tail is 10 leaves at 10⁻⁶, four at 10⁻⁷ and one at
10⁻⁸ — the unstable small clusters this question anticipated, present
but few, and exactly what the census reports and the sampler skips or
labels. The topical family therefore serves 10⁻¹, 10⁻³, 10⁻⁴ and 10⁻⁵
on its own, with 10⁻² and below 10⁻⁵ left to conjunctions (TS-17) and
the control family. The leaf margin confirms the greedy descent is
worth measuring: the median gap between a passage's leaf and its best
sibling is 0.029 in cosine distance, and 20.8% of passages sit within
0.01 of a sibling.

**TS-164.** *Should the non-topical families pivot on the query's own
passage, and should the control share follow the pivot?* Yes to both
in principle — every query is a source passage whose enriched row
exists, so any predicate can be labelled or constructed against it —
but each is a further predicate construction regime, and regimes need
inventory support to be distinct artifacts before they are worth
building. Deferred to
[srd-predicate-pivot-regimes.md](srd-predicate-pivot-regimes.md); the
even global hash share stands meanwhile (TS-163).

## 15. Acceptance

**TS-42.** At every profile with *N* ≥ *N*ᵣ: no predicate that applies
to the profile (TS-51) returns zero matches, and every
`(family, decade)` cell above the floor is populated
or its shortfall reported. Below *N*ᵣ the acceptance is weaker and
deliberately so — the hash family is present and returns non-empty
results, and nothing else is required.

**TS-43.** Realised selectivity of each predicate lies within its
assigned half-decade band, verified against the R facet after
evaluation rather than asserted at generation.

**TS-44.** Regenerating with the same seed and configuration produces
an identical predicate facet, byte for byte.

**TS-161.** Every query ordinal has a predicate, and for every topical
record the recorded placement is what the query's own descent says.
A filtered result whose predicate was affixed to its query by position
alone is not a measurement of anything, so this is checked before the
R facet is built, not inferred from it.

**TS-45.** A reader of the predicate set, given only the vernacular
form of each predicate, can state what each one is asking for without
reference to this document — the operational test for TS-2.

## 16. Decision record

The requirements above state *what* the design does. These are the
judgement calls behind them, with the alternative that was rejected —
recorded because a reader who disagrees needs the reasoning, not just
the conclusion, and because several of these looked settled and were
not.

**D-1 — Believability is a hard requirement, not a preference.**
*Rejected:* a scheme of uniform hash buckets and random hyperplanes over
the embedding, which solves the distribution problem exactly and at
lower cost than anything here. It produces predicates like
`WHERE bucket_1000 = 37` and `WHERE hplane_bits & 0x0F = 0x0A`. A
benchmark whose filters are visibly reverse-engineered from the
measurement cannot be offered as evidence about anything, so the
statistical merit was not decisive. → TS-2.

**D-2 — Control and credibility are orthogonal.** The reframe that made
D-1 survivable. Control is a property of how an attribute is
*distributed*; credibility is a property of what it *means*. Nothing
requires them to travel together, so the move is not to choose between
them but to engineer the distribution of attributes that already mean
something. → §2.

**D-3 — The control family is demoted, not deleted.** A filter with
exactly known selectivity and provably zero semantic correlation is a
legitimate null hypothesis. Keeping it is what lets a cost difference be
attributed to *semantics* rather than to filtering as such. It is
labelled, never presented as a realistic query. → TS-2, TS-59.

**D-4 — Reliability is promised above a threshold, not everywhere.**
*Rejected:* a hard floor enforced at every profile. A 100,000-passage
profile cannot host a topic both fine-grained enough to be interesting
and populous enough to measure — no clustering makes that true, and
requiring it would have distorted the design to serve the least
interesting end of the ladder. **More than half the ladder (18 of 31
detents) sits below the threshold, and that is accepted rather than
worked around.** → TS-46, TS-49.

**D-5 — Below the threshold the hash family carries the profile, and
this buys availability rather than precision.** A bucket of any modulus
can be defined on demand; no clustering can be asked for a topic of
arbitrary size. The statistical spread is identical — both give
`Binomial(N, s)` — so the small profiles still have a predicate set
with known selectivity, but nothing about small-*N* variance improves.
Stated explicitly so the guarantee is not read as stronger than it is.
→ TS-47.

**D-6 — Clusters are measured, not balanced.** *Rejected:* balanced
k-means. Forcing equal sizes distorts clusters into shapes no topic
model would produce, reintroducing exactly the artificiality D-1
rejects. Letting them fall naturally and selecting from the resulting
size distribution gives real topics and exact selectivity at once; the
unevenness becomes the menu rather than the problem. → TS-8, TS-9.

**D-7 — Only the decade detents are built for this dataset.** The
`mul`, `fib` and `linear` strata are dropped, leaving 31 profiles. →
§3.7.

**D-8 — One master predicate set, selected per profile.** *Rejected:*
sampling independently per profile, which the per-profile R facet would
have permitted. Because selectivity is scale-free, predicate *p* means
the same thing at 10m and at 400m, so a difference between two profiles
is attributable to scale alone; independent sampling would confound the
two across 31 profiles. It also matches what is already on disk —
`predicates.slab` is a single shared artifact. → TS-50, TS-51.

**D-9 — Predicates are anchored at decade boundaries; mantissa steps
inherit them.** Consecutive detents differ by at most 2×. The mantissa
sweep measures how cost grows with *N*; the decade anchors measure how
it varies with the predicate. Generating different predicates per step
would add no information and would break D-8. → TS-52.

**D-10 — Per-decade predicate counts are tapered.** Follows from
`R bytes/base = 4 · Σ selectivity`, which validates to 0.17% against the
measured facet. Ten predicates at 10⁻¹ carry the same storage mass as
ten thousand at 10⁻⁴, so a flat count spends the budget at the coarse
end where predicates are least interesting. → TS-53, TS-54.

**D-11 — Derived columns are stored natively as MNode fields.**
*Rejected:* packed `ScalarPacked` columns with code→label dimension
tables, which costs 15 bytes a record instead of 226 — 8 GB instead of
120. Rejected because the M facet is shared across profiles and paid
once, so 112 GB on a dataset with a 2 TB base does not justify a second
storage format that the layout facet, the survey, the generator, the
evaluator and every external consumer would have to understand. → TS-55,
TS-57.

**D-12 — Coded values store their label, not a code.** The consequence
of D-11 that matters most: the value a predicate compares against is
the value on the wire, so nothing stands between the stored data and
the query a person would write, and D-1 needs no decode step to hold.
→ TS-56.

**D-13 — The sampler's machinery leaves no residue in the artifact.**
Density conditioning happens when a predicate is *selected*; the
published predicate records nothing of the decade it was drawn for or
the cluster size that qualified it. That belongs in the generation
report. A consumer reading the predicate facet sees filters, not an
experimental design. → TS-58.

**D-14 — The predicate family is a sibling namespace, not a field, a
facet, or a report.** *Rejected:* annotating inside the PNode, which
would change what the predicate matches; a separate facet, which can
disagree with what it describes; ordinal-range conventions, which are
implicit and break when counts change; and the generation report alone,
which would leave a consumer holding the dataset unable to satisfy TS-7
from it. The namespace keeps the filter clean and the dataset
self-describing at once, and it reuses machinery that already exists —
the spec models per-facet namespaces, the reader opens them, and the
`forms` namespace already establishes that absence means a default
rather than an error. **The decade is derived from the recorded
selectivity rather than stored**, which is what lets TS-7 and TS-58 both
hold. → TS-60 … TS-68.

**D-15 — Provenance and diagnostics are retained, ordinal-aligned,
and never in the metadata facet.** Generalises D-14 from the predicate
family to the pattern. M is the query surface, so anything in it is
something a predicate can name and something every reader pays for;
traces belong beside the data, joined by ordinal, and optional.
*Rejected:* keeping traces outside the dataset entirely, which is
tessera's current handling of its embedding revision and which its own
provenance notes admit is undetectable when it goes wrong. This also
gives the packed-column encoding rejected in D-11 its proper home — the
objection there was a second format between predicates and the fields
they name, and nothing writes predicates against a diagnostic. →
TS-69 … TS-76.

**D-16 — Adjunct data is written by the command that already computes
it, not by passes of its own.** The cluster margin is emitted by
`compute topics` because both distances exist at assignment time and
recovering them later costs a second pass over half a billion 1024-wide
vectors; the families namespace is written by `generate predicates`
because it is the only step that knows a family, which makes
co-location structural rather than a rule someone must remember.
*Rejected:* separate analysis passes for each adjunct, which would be
tidier to specify and would pay for the tidiness in full re-reads of the
largest facet in the dataset. → TS-77, TS-81, TS-82.

**D-17 — Adjuncts are non-standard view keys, not new facet-spec
variants.** *Rejected:* registering each as a `StandardFacet`.
Conformance leaves non-standard keys alone, so registration is not
needed to declare them — and the facet spec's job is to say what a
dataset *must* have, while TS-73 says nothing depends on an adjunct. A
single-letter facet code is a scarce namespace and should not be spent
on optional diagnostics. The cost, accepted knowingly, is that
`veks check` will not validate them. → TS-87.

**D-18 — Hierarchical assignment is greedy, and says so.** Descending
10 → 30 → 33 costs 73 inner products per passage instead of 10,310, and
is therefore not equivalent to a flat argmax over all L3 centroids: a
passage near an L1 boundary can land in a cluster that is not its global
nearest. Accepted for the 140× saving, but stated explicitly so nobody
later reads a topic assignment as "the nearest cluster" — and the margin
facet is what makes the discrepancy measurable rather than invisible.
→ TS-97, TS-98.

**D-19 — Selectivity is censused by the survey's own exhaustive pass:
not by its sampled passes, and not by a separate instrument.** The
sampled passes are a capped content-and-distribution assay: 100,000
rows (10,000 on tessera), exact frequency tables only at or below 64
distinct values, heavy-hitters top-64 above that. Of 10,310 topics they
can enumerate 64, and a topic at 10⁻⁵ has one expected member in the
sample. *Rejected:* raising their caps, which would disable the sampling
that makes the survey affordable, widen every bounded tracker, run a
dozen measures exact counting does not need over half a billion records
— and still leave the regime decided from a sample before the measures
are chosen. *Also rejected:* a separate census command, which an earlier
draft proposed. The survey is the dataset's content-and-distribution
assay and the one artifact the generator and the evaluator already
read; a second instrument would describe the same fields in a second
file with a second schema and a second staleness chain, and the fitting
phase would have to reconcile the two. The census is therefore a third
pass inside the survey, exhaustive by definition, bounded by
declaration, with hierarchy and pair tables shaped for lookup during
fitting. **The survey also moves downstream, to the M facet after
extraction**, so that it counts the base population rather than the
source (D-21). → TS-30, TS-124 … TS-129, TS-139 … TS-146.

**D-20 — Labelling is its own command, and it samples row groups, not
rows.** *Rejected:* a mode of `compute topics`, which couples a text
pass to the GPU step so that a labelling change re-fits the model; and a
uniform row sample, which touches every row group and turns a 20M-row
sample into a 241 GB read. A seeded subset of row groups bounds the read
at an eighth of the file and still meets ~67 members of a 10⁻⁶ cluster.
→ TS-133, TS-134, TS-138.

**D-21 — Compute in source order, publish in base order, count in base
order.** The three parquet files and `_base_all.fvecs` are row-aligned
in source order and in no other, so assignment and enrichment happen
there. `profiles/base/` is base order, and "ordinal-aligned" has no
other meaning for an adjunct, so the margin crosses the same bridge M
does, by `transform extract`. The survey's census pass counts M because
M is the population predicates are evaluated over: the source holds
35,929,249
duplicates and 10,000 queries that no predicate can match, and nothing
says they are spread evenly. *Rejected:* a mapping of the pipeline's own
for the margin (a second way to do what `extract-metadata` already
does), and censusing the enriched parquet (cheaper by a columnar read,
wrong by 6.75% of rows unevenly distributed). → TS-130 … TS-132.

**D-22 — Fit on the shuffled prefix, assign in source order, one
assignment facet.** The shuffled base's prefix is a uniform sample of
the base population and a sequential 20 GB read; the source-order
file's prefix is a corner of the corpus and its uniform sample is five
million scattered reads. Assignment still runs over the source-order
file, because that is the order enrichment joins in (TS-130). The
assignments are one u16 xvec with one code per level per record rather
than a packed file per level: the runner anchors freshness on one
output, a reader wants a row's levels together, and codes are
positional so the tree is implicit. A model report is retained beside
the centroids because reproducing the fit needs the levels, seed and
sample the centroid file cannot carry. → TS-150 … TS-152.

**D-23 — One predicate per query, placement per pair.** The facet
contract every consumer relies on pairs query *i* with predicate *i*.
An earlier draft sized the predicate set by the per-cell counts alone
(≈1,120 records for the ladder) and labelled placement against the
query set as a whole; the filtered KNN would have evaluated 8,880
queries against nothing and called it a result. The stratification is
now over query slots, placement is decided per pair from the query's
own descent, distinct predicates repeat only under pool exhaustion, and
the control family backfills any slot no cell can fill. → TS-156,
TS-157, TS-159, TS-161.

**D-24 — Authenticity is a standing requirement, not a property of the
current columns.** Every field attached to a base ordinal is that
passage's own source metadata or a derivative of that passage's own
data, and every predicate paired with a query relates to that query's
own data or is independent of it by design and says so. The control
hash is the single, labelled exception. → TS-158, TS-162.

**D-25 — Even control share now, pivot-weighted later.** The hash
family takes the same share of query slots as each semantic family, in
one global hash space, because an even share is the simplest to read
in diagnostics and the alternative — a control share weighted per
semantic pivot — is one of several predicate regimes that need the
inventory to hold them as distinct artifacts first. → TS-163, TS-164,
and the deferred SRD.

### 16.1 What changed while this was written

**TS-38** asked whether the L3 cluster count should be set by the
smallest profile. D-4 resolved it: the count is now bounded by the
threshold profile, and 10,000 clusters clear it comfortably.

**TS-40** asked whether the control family should ship. D-5 settled it —
it must, because it is what gives sub-threshold profiles a predicate set
at all. That surfaced the question now in its place — how a predicate's family
reaches a consumer — which §4.5 and D-14 now answer.
