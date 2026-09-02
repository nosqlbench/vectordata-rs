# SRD — Predicate pivot regimes

**Status:** to be done — deferred until the dataset inventory can hold
several predicate regimes as distinct, named artifacts.
**Scope:** how a query's predicate relates to the query's *own* passage
for every family, not only the topical one; how several such regimes
coexist in one dataset, separately or in a combined form; and how the
control family's share is weighted per pivot rather than globally.

**Depends on**
[srd-topic-stratified-predicates.md](srd-topic-stratified-predicates.md)
(TS-6 for the families, TS-19 / TS-157 for per-pair placement, TS-156
for one predicate per query, TS-158 / TS-162 for the authenticity
requirement) and on
[prefilter-postfilter-facets.md](prefilter-postfilter-facets.md) for
the F and E facets a regime is measured through.

## 1. What is settled, and what this defers

The topic-stratified design pairs query *i* with predicate *i*
(TS-156) and, for the **topical** family, decides the pair's relation
from the query's own descent: *in-topic* when the query lies in the
predicate's topic, *out-of-topic* when it does not (TS-157). That is a
**pivot**: the predicate is anchored to something real about the
query, and the anchor is named in the record.

For the **structural** and **bibliographic** families the predicate is
drawn independently of the query, by design (TS-6), and the record
carries no placement. That is a measurement of the uncorrelated case,
and it stays. What it is *not* is a pivot: nothing relates the
predicate to the query passage's own section, year or citation
standing, although every query is a source passage whose enriched
metadata row exists (TS-137, TS-158).

The **control** family is a seeded hash, never a semantic predicate
(TS-115, TS-149). Its share of the query slots is the same as every
other family's, an even quarter (TS-163), which is right for
diagnostics and wrong as a *weighting*: the null case should be sized
against the pivot it is the null of, not against the whole.

This document names the regimes that would complete the picture and
the inventory support they need. None of it is built, and none of it
should be built until the inventory can tell one regime from another.

## 2. Regimes

**PR-1.** A **regime** is a rule for choosing the predicate of query
*i* given query *i*'s own passage (its embedding and its enriched
metadata row) and the census. The current facet realises two regimes
in one artifact — topical-with-placement and independent — and records
which applies per record.

**PR-2. Query-native pivot.** For any family, the predicate is drawn
from the census as now and the pair is labelled by evaluating the
predicate against the **query's own metadata row**: *query-in-filter*
when the query passage itself satisfies the predicate, *query-out-of-
filter* otherwise. This generalises TS-157 from the topical family to
all of them at no cost beyond one predicate evaluation per pair, and
gives the structural and bibliographic families a meaningful
relationship to pivot on: a filter on the query's own year, or its own
section class, is the realistic "search within my kind of paper" case.
*The labelling half of this is built* (TS-165, TS-166): the query
metadata facet exists and every record carries `query_in_filter`.
What remains of PR-2 is drawing *toward* the label, which is PR-3.

**PR-3. Anchored construction.** Beyond labelling, a regime may
**construct** the predicate from the query's own row — the query's
section class, its year, its citation band, its topic — so the pair is
in-filter by construction, at the selectivity the census reports for
that value. This is the query-relative counterpart of the topical
in-topic draw, and it is what makes a conjunction across families
(`topic = t AND year ≥ y`, with both taken from the query's own row)
a single, authentic case rather than two independent draws.

**PR-4. Combined regimes.** Regimes may be realised as **distinct
predicate facets** in one dataset — one per regime, each with its own
families and generation namespaces and its own F and E facets — or as
**one facet whose records carry a projection**: for each field the
predicate touches, the relation between the predicate's comparand and
the query's own value (equal, contains, above, below, absent), so that
a single answer key can be sliced by regime after the fact. The
projective form is the numerically cheaper one, since the answer keys
are computed once; the distinct form is the operationally simpler one,
since every consumer already understands one facet per profile.

**PR-5. Per-pivot control share.** For each semantic pivot point (a
topic, a section class, a year band) the control family's share is a
weight attached to that pivot, so the null case is sized against the
semantic case it accompanies: a pivot with many in-filter pairs gets a
proportionate number of hash predicates at the same selectivities, a
pivot with none gets none. The global even share of TS-163 is the
degenerate case with one pivot.

## 3. What the inventory must support first

**PR-6.** A dataset must be able to hold **more than one predicate
facet per profile**, each named for its regime, and the catalog,
`veks check`, the explorer and the verification tools must treat them
as distinct rather than as one `metadata_predicates` facet with a
default name. Today's facet spec binds one predicate facet, one
results facet and one pair of filtered ground-truth facets to a
profile.

**PR-7.** Each such facet carries its own **families** and
**generation** namespaces (TS-88), and its results and filtered
ground-truth facets are named to match, so a regime's answer keys can
never be read against another regime's predicates.

**PR-8.** The dataset attributes and the catalog entry state which
regimes are present and, for the projective form, which projection
fields a record carries, so a consumer can select a regime without
opening the slab.

**PR-9.** The pipeline scaffolding must run the predicate, evaluation,
F and E steps **per regime** as it runs them per profile today, with
the regime a parameter of the step rather than a copy of it.

## 4. Interim decision

**PR-10.** Until PR-6 through PR-9 exist, the dataset carries one
predicate facet realising the topical-with-placement and independent
regimes side by side, and the control family takes an **even share of
the query slots** in one global hash space (TS-163). The record-level
fields already written — `family`, `topic`, `query_placement`,
`query_topic`, `predicate`, `backfill` — are the ones a later
projective form would extend, so nothing written now needs to be
re-shaped to adopt PR-4.

## 5. Acceptance, when built

**PR-11.** For every regime facet, record *i* is query *i*'s predicate
and the recorded relation is what evaluating the predicate against
query *i*'s own row says (PR-2), or what the construction guarantees
(PR-3).

**PR-12.** Answer keys of one regime are never consumed against the
predicates of another; a mismatch is a check failure, not a warning.

**PR-13.** A per-pivot control share sums, over pivots, to the
configured global share, and each pivot's hash predicates match its
semantic predicates decade for decade.

## 6. Open questions

**PR-14.** Whether the projective form (PR-4) should be the only form,
with distinct facets derived from it on demand, or whether both must
be first-class. The answer depends on how consumers slice results,
which the first regime facets will show.

**PR-15.** Whether anchored construction (PR-3) should draw the
anchoring value from the query's row directly or from the census node
that contains it, when the two disagree at a boundary (a year at the
edge of a range, a citation count at a tie).
