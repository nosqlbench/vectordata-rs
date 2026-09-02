# SRD — Topic-stratified predicate sampling

**Status:** proposed
**Scope:** metadata enrichment, predicate generation, and the pipeline
steps that produce them. Adds commands to `veks run`; designed to
augment an existing dataset in place rather than rebuild it.

**Depends on**
[metadata-facets-and-layout-namespace.md](metadata-facets-and-layout-namespace.md)
for the M / P / R facet shapes, and
[prefilter-postfilter-facets.md](prefilter-postfilter-facets.md) for
what E and F measure and why the difference between them matters.

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
measures high-recall filtering or the adversarial extreme. This is
currently left to chance. The generator must label each topical
predicate with the relation between its topic and the query set, and
the sampler must draw a configured mix of both.

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

**TS-22.** `section_class` normalises the 72,977 raw headings to a
small closed taxonomy (`introduction`, `background`, `methods`,
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
sample_bucket = 37
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
| `query_placement` | text | `in-topic` / `out-of-topic` (TS-19), topical only |

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

Four new commands. Existing `evaluate-predicates` is unchanged.

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
numbers exist and unaffordable anywhere else.

### 6.2 `analyze topic-sizes`

**TS-30.** Reads the assignments and emits the measured size of every
cluster at every level, as the table the sampler selects from (TS-9).
This is a counting pass, cheap, and its output is the artifact that
makes TS-1 true.

### 6.3 `transform enrich-metadata`

**TS-31.** Joins the derived columns of §4 onto the metadata source,
emitting an enriched artifact in the same format. Inputs: the metadata
parquet, the topic assignments, and the passage table for `ordinal`,
`char_start`, `char_end`.

**TS-78.** The command is **not a pure row-wise map**, and the plan must
say so rather than let an implementation discover it. Three of the six
derivations are row-local; two need an aggregate over the corpus first:

| column | derivation | pass |
|---|---|---|
| `word_count` | `char_end − char_start` | row-local |
| `sample_bucket` | `hash(corpusid ‖ ordinal) mod K` | row-local |
| `section_class` | heading → class lookup | row-local, given the table |
| `topic_l1/l2/l3` | assignment lookup by ordinal | row-local, given assignments |
| `passage_position` | `ordinal ÷ passages_in_paper` | **needs a group-by on `corpusid`** |
| `citation_percentile` | rank of `citationcount` within `year` | **needs a group-by on `year`** |

Both aggregates are small — 17,457,121 paper lengths and 116 per-year
citation distributions — so each is one counting pass and an in-memory
table, not a sort of the corpus.

**TS-79.** The heading → `section_class` mapping is published as an
adjunct (TS-71). It is a judgement call applied 531,869,985 times across
72,977 distinct headings, and anyone assessing a `section_class`
predicate will want to audit it. Headings the table does not cover fall
to `other`; an unmapped heading is never an error.

**TS-80.** The hash keyed for `sample_bucket` is over
`(corpusid, ordinal)` — the **passage**, not the paper. Keying it on
`corpusid` would reproduce the paper-blocking of §1.1 in the one family
that exists to be free of it.

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
run that produced it.

### 6.5 Recording embedding provenance

**TS-83.** The model identity and **resolved** revision that produced
the base vectors are recorded in the dataset. For tessera this is a
back-fill: the values exist only in a markdown file beside the corpus,
and its own notes concede that nothing in `dataset.log` or
`runlog.jsonl` would show the difference if another host resolved the
model tag differently (TS-76). For future builds the embed command
emits them at the point it resolves them, which is the only point where
they are known to be correct.

### 6.6 What is cache and what is retained

**TS-84.** Intermediates that can be recomputed live under `.cache/`
and are removable by `veks prepare cache-gc`. Retained adjuncts live in
the dataset and are not:

| artifact | where | why |
|---|---|---|
| topic assignments | `.cache/` | consumed by `enrich-metadata`; the values then live in M |
| topic centroids | dataset | reproducing the labelling requires them (TS-26) |
| assignment margin | dataset | explains filter crispness; a second pass to recover |
| heading → class table | dataset | a judgement call worth auditing |
| families namespace | in `predicates.slab` | TS-64 |
| generation trace | dataset | selection provenance |
| embed provenance | dataset | irrecoverable once the host cache turns over |

**TS-85.** An adjunct must never be written under `.cache/`. It would
be correct until the first `cache-gc`, and its loss would be silent —
the failure mode TS-76 rejects, arrived at by a different route.

## 7. Augmenting tessera in place

**TS-35.** The augmentation must not invalidate `extract-base` or any
`compute-knn` step. This holds because unfiltered KNN depends on
`count-base` and `extract-queries` only — not on metadata — so under
the `config-only` provenance selector the 85 completed KNN profiles
stay fresh.

What re-runs is the metadata and predicate chain:

```
compute-topics          NEW    one GPU pass over base_vectors
                               -> assignments (.cache), centroids +
                                  margins (dataset)
analyze-topic-sizes     NEW    counting pass over assignments
enrich-metadata         NEW    two counting passes (paper lengths,
                               per-year citation ranks) then one
                               pass over metadata + passages parquet
record-embed-provenance NEW    back-fill; dataset attribute only
convert-metadata        re-run ~300 s (measured)
extract-metadata        re-run ~2 passes (measured ~207 s each)
survey-metadata         re-run
generate-predicates     re-run with --strategy stratified
evaluate-predicates     re-run per profile   ← the expensive one
```

**TS-36.** Steps are appended to `upstream.steps` with `after:`
declaring `compute-topics` before `enrich-metadata`, and
`enrich-metadata` before `convert-metadata`. No existing step
definition changes except `generate-predicates`' options, which is what
correctly marks it and its dependents stale.

**TS-37.** The R facet must be regenerated for every profile. Its size
scales linearly with base count at 87.45 B/base under the current
0.001 configuration; at the lower selectivities this design targets it
falls proportionally. The existing per-profile `metadata_results.slab`
files and their `.cache/*.predkeys.slab` segments become garbage and
should be removed by `veks prepare cache-gc` before the run to reclaim
space.

## 8. Artifact register

Every artifact this design introduces, named, located and formatted.
Nothing below is inferable from §§4–6, and an implementation cannot
start without it.

**TS-86.**

| artifact | path | format | lifetime |
|---|---|---|---|
| topic assignments | `.cache/topic_assign_l{1,2,3}.u16` | `ScalarPacked` u16 | cache |
| topic centroids | `profiles/base/topic_centroids.fvecs` | `FloatXvec` f32, dim 1024 | retained |
| cluster margin | `profiles/base/topic_margin.mvecs` | `FloatXvec` f16, dim 2 | retained |
| section-class map | `profiles/base/section_class_map.slab` | `Slab`, MNode | retained |
| predicate families | `predicates.slab` ns `families` | `Slab`, MNode | retained |
| generation trace | `predicates.slab` ns `generation` | `Slab`, MNode | retained |
| embed provenance | `dataset.yaml` attributes | yaml | retained |

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
experimental design. `StandardFacet::MetadataPredicates::namespaces()`
changes from `[""]` to `["families", "generation", ""]`, and each name
is a published constant beside `FORMS_NAMESPACE`.

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
**sample** of each cluster's members (default 2,000), because term
statistics converge long before the member list is exhausted.

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
filter on "long passages" in the unit they think in. It is computed
during the same pass that reads `passages.parquet` for labelling.

## 10. Command surfaces

Complete option tables. `role` is the `OptionRole` the runner uses —
`Input` participates in the "input newer than output" staleness check,
`Config` is provenance-bearing, `Output` is what `check_artifact`
inspects.

### 10.1 `compute topics`

| option | role | req | default | notes |
|---|---|---|---|---|
| `base` | Input | yes | — | base vectors; accepts a series (SH-35) |
| `levels` | Config | no | `10,30,33` | branching per level |
| `sample-size` | Config | no | `5000000` | passages used to fit |
| `iterations` | Config | no | `50` | convergence cap (TS-93) |
| `tolerance` | Config | no | `1e-4` | mean centroid movement |
| `seed` | Config | no | `42` | k-means++ and sampling |
| `centroids` | Output | yes | — | `topic_centroids.fvecs` |
| `assignments` | Output | yes | — | `.cache/topic_assign_l*.u16` |
| `margin` | Output | no | — | omit to skip (TS-72) |

**TS-107.** `check_artifact` is Complete when the centroid file holds
`Σ levels` records of the base's dimension **and** every assignment
column has one record per base ordinal. Either alone is insufficient:
centroids without assignments is a fit that never finished, and
assignments without centroids cannot be reproduced or extended.

**TS-108.** Declares `mem` and `threads` to the governor. The fit holds
`sample-size × 1024 × 4` bytes — 20 GB at the default — and must
request it rather than discover it.

### 10.2 `analyze topic-sizes`

| option | role | req | default | notes |
|---|---|---|---|---|
| `assignments` | Input | yes | — | from `compute topics` |
| `output` | Output | yes | — | size table per level |
| `min-members` | Config | no | `0` | report clusters below this |

**TS-109.** Emits, per cluster: level, code, member count, and the
member count as a fraction of the corpus. That fraction is the
`selectivity` the sampler stratifies on (TS-14) and the value written
into the families namespace (TS-62).

### 10.3 `transform enrich-metadata`

| option | role | req | default | notes |
|---|---|---|---|---|
| `metadata` | Input | yes | — | source parquet |
| `passages` | Input | yes | — | for `ordinal`, text, word count |
| `assignments` | Input | yes | — | topic codes |
| `labels` | Input | yes | — | code → label, from `compute topics` |
| `section-map` | Input | no | built-in | override the prefix table |
| `buckets` | Config | no | `1000` | *K* for `sample_bucket` |
| `seed` | Config | no | `42` | hash seed |
| `output` | Output | yes | — | enriched parquet |
| `section-map-out` | Output | no | — | the table actually applied (TS-79) |

**TS-110.** Fails if the assignment count does not equal the metadata
row count. A silent off-by-one here mislabels every passage after the
divergence and would surface only as inexplicable benchmark results.

### 10.4 `generate predicates --strategy stratified`

| option | role | req | default | notes |
|---|---|---|---|---|
| `survey` | Input | yes | — | existing metadata survey |
| `base-count` | Config | yes | — | *N*, for the per-profile floor |
| `query-placement` | Config | no | mixed | in-topic / out-of-topic mix (TS-19) |
| `topic-sizes` | Input | yes | — | from `analyze topic-sizes` |
| `queries` | Input | no | — | required for `query-placement` |
| `families` | Config | no | all four | which families, and their mix |
| `per-cell` | Config | no | tapered | per-decade counts (TS-54) |
| `decades` | Config | no | `1e-1..1e-7` | target range |
| `min-matches` | Config | no | `100` | *M* in TS-11 |
| `reliability-threshold` | Config | no | `10000000` | *N*ᵣ (TS-46) |
| `report` | Output | no | — | generation report (TS-34) |

**TS-111.** `check_artifact` is Complete when the content namespace
holds at least one predicate **and** the `families` namespace holds
exactly as many records. An unequal count means annotation and
predicates disagree, which TS-64 exists to prevent.

## 11. Candidate enumeration

The sampler draws from candidate pools (TS-14). Nothing yet says what
is in them.

**TS-112.** **Topical.** Every cluster at every level is a candidate:
10 + 300 + 10,000 = 10,310, each with a measured selectivity from
TS-109. No enumeration cost — the size table *is* the pool.

**TS-113.** **Bibliographic.** Enumerated from the survey as threshold
and range predicates over `citation_percentile`, `year` and
`isopenaccess`. `citation_percentile` is uniform by construction, so a
threshold at *t* has selectivity `(100−t)/100` exactly and the pool is
dense across the coarse decades. `year` ranges are enumerated from the
observed distribution, which is skewed, so their selectivities are
measured rather than assumed.

**TS-114.** **Structural.** Enumerated over `section_class` values
(eight, measured), `passage_position` ranges (uniform by construction),
and `word_count` ranges (measured from the distribution).

**TS-115.** **Control.** Generated, not enumerated: for a target
selectivity *s*, emit `sample_bucket = c` with `K = round(1/s)` and *c*
drawn seeded. The only family that can fill any cell on demand, which is
what TS-47 relies on.

**TS-116.** **Conjunctions** (TS-17) are formed only after the single-
field pools are exhausted for a cell, pairing a topical candidate with a
bibliographic one. Their selectivity is **measured, not multiplied**
(TS-18): evaluated over a uniform sample of the corpus (default 10M
passages) sized so the estimate's relative error at the cell's target
decade is under 10%, then admitted to whichever band the estimate lands
in.

**TS-117.** Query placement (TS-19) requires the query vectors. Each
query is assigned to a topic by the same descent as TS-96, and a
topical predicate is labelled `in-topic` when at least one query falls
in its cluster, `out-of-topic` otherwise. Without `queries` the
generator omits the field rather than guessing.

## 12. Test strategy

**TS-118.** Each of §9's algorithms is tested against a **synthetic
corpus with known structure** — planted clusters at known sizes,
metadata with known distributions — so expected outputs are derivable
rather than golden. Determinism (TS-94) is tested by fitting the same
sample at 1, 4 and 16 threads and comparing centroids bit-for-bit.

**TS-119.** The sampler is tested without any clustering: given a
synthetic size table, every `(family, decade)` cell above the floor is
populated, cells below it are reported unpopulated rather than filled,
and the same seed reproduces the same set (TS-15, TS-44).

**TS-120.** End-to-end, on a small dataset built by the existing
fixtures: the pipeline runs, the enriched M facet carries all eight
columns, `predicates.slab` carries both namespaces with matching
counts, and every generated predicate returns a non-empty match set at
a profile above the threshold (TS-42).

**TS-121.** The acceptance criterion of TS-43 — realised selectivity
within the assigned band — is checked **against the R facet after
evaluation**, not against the generator's own estimate. A generator
that mis-measures would otherwise validate itself.

## 13. Work breakdown

Ordered by dependency. Each phase is independently testable and leaves
the tree green.

**Phase 1 — clustering.** `compute topics` (§10.1) with §9.1–9.2:
spherical k-means, deterministic reduction, hierarchical assignment,
margin output. The largest single piece and the only one needing GPU
work. Testable in isolation against planted clusters (TS-118).

**Phase 2 — measurement.** `analyze topic-sizes` (§10.2). Small, and it
unblocks the sampler.

**Phase 3 — labelling.** §9.3, as a mode of `compute topics` or a
sibling command. Independent of Phases 4–6: positional labels (TS-102)
let everything downstream proceed while this is unfinished.

**Phase 4 — enrichment.** `transform enrich-metadata` (§10.3) with
§9.4. Two counting passes plus a map. No new formats.

**Phase 5 — sampling.** `generate predicates --strategy stratified`
(§10.4) with §11, plus the two namespaces (TS-88). The most intricate
logic and the least I/O.

**Phase 6 — provenance.** Embedding provenance (TS-83) and the
cache/retained split (TS-84). Small, and separable.

**TS-122.** Phases 1–2 and 4 can be validated on tessera without
regenerating any predicate: they produce artifacts that nothing yet
consumes. The first irreversible step is Phase 5, because it invalidates
`evaluate-predicates` across every profile.

**TS-123.** Estimated cost on tessera, from measured comparable steps:

| phase | pass | estimate |
|---|---|---|
| 1 fit | 5M × 1024, 50 iterations | GPU minutes |
| 1 assign | 531.9M × 73 inner products | one sequential pass over 2.03 TB |
| 2 | count 531.9M u16 | minutes |
| 3 | text of 10,310 × 2,000 passages | one indexed pass over `passages.parquet` |
| 4 | 2 counting passes + 1 map over 531.9M | comparable to `convert-metadata`, ~300 s measured |
| 5 | sampling + conjunction measurement on 10M | minutes |
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
question is no longer "how" but "how good is good enough". Still open,
and TS-102's positional fallback means it does not block anything.

**TS-40.** ~~*Should the control family ship?*~~ **Settled by TS-47:
it must.** It is what gives sub-threshold profiles a predicate set at
all, so it is no longer optional — which raises the remaining question
in its place: how to label it so a consumer cannot mistake a control
for a realistic query. The dataset carries no field today that marks a
predicate's family, and TS-7 requires results be grouped by one.

**TS-41.** *Sampling the fit.* 5M of 531.9M is ~1%. Whether that is
enough to place 10,000 L3 centroids stably is untested; the failure
mode is unstable small clusters, which the size table would expose.

## 15. Acceptance

**TS-42.** At every profile with *N* ≥ *N*ᵣ: no predicate returns zero
matches, and every `(family, decade)` cell above the floor is populated
or its shortfall reported. Below *N*ᵣ the acceptance is weaker and
deliberately so — the hash family is present and returns non-empty
results, and nothing else is required.

**TS-43.** Realised selectivity of each predicate lies within its
assigned half-decade band, verified against the R facet after
evaluation rather than asserted at generation.

**TS-44.** Regenerating with the same seed and configuration produces
an identical predicate facet, byte for byte.

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

### 16.1 What changed while this was written

**TS-38** asked whether the L3 cluster count should be set by the
smallest profile. D-4 resolved it: the count is now bounded by the
threshold profile, and 10,000 clusters clear it comfortably.

**TS-40** asked whether the control family should ship. D-5 settled it —
it must, because it is what gives sub-threshold profiles a predicate set
at all. That surfaced the question now in its place — how a predicate's family
reaches a consumer — which §4.5 and D-14 now answer.
