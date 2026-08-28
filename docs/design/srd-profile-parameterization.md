# SRD — Parameterized profiles

**Status:** proposed
**Scope:** `ProfileConfig` / `DSProfile`, default inheritance, the
`sized:` generator, conformance, `describe` / `list`.

**Supersedes** `srd-facet-variants.md`, removed in the same change. That
document proposed a fourth addressing coordinate — a "facet variant" —
on the premise that expressing a family of selectivities as profiles
would duplicate `base_vectors` across them. The premise was false:
`apply_default_inheritance` has always let a non-default profile omit
the shared facets and inherit them. What the coordinate would have
provided, a profile already is. See §7 for what was actually missing.

## 1. Problem

A dataset needs precomputed filtered ground truth at several
selectivities at once: at 0.1%, at 1%, at 10%, with a benchmark naming
one of them.

Those facets **co-vary**. At a given selectivity the predicates that
define the filter, the result index recording which base ordinals each
predicate matches, and the pre- and post-filtered ground truth derived
from those are four files that only mean anything together. Pairing 1%
predicates with 10% results is not a degraded answer, it is a
meaningless one that produces numbers looking like measurements.

A **profile is already exactly that**: a named group of facets, selected
as a whole, sharing invariant facets with `default` by inheritance. One
selectivity is one profile. Mixing is unexpressible because there is
nothing to mix — you open a profile and get its facets.

So this is not a new mechanism. It is the removal of three specific
things that stop the existing one carrying a second axis.

## 2. What already works

Stated so the gaps below are read against it, and so nothing here is
rebuilt:

- **Inheritance.** A non-default profile omitting `base_vectors`,
  `query_vectors`, `metadata_content`, `metadata_predicates`,
  `predicate_results` or `metadata_layout` inherits them from `default`
  (`apply_default_inheritance`). No duplication.
- **Generators.** `sized:` expands a spec into a family of profiles,
  interpolating `${profile}` and `${range}` into facet path templates.
- **Family membership.** `series_by_spec` already records which profiles
  each generator spec produced, in generator order.
- **Selection.** `dataset:profile` addresses one. Nothing new is needed
  to *choose* a parameterization.

## 3. The axis problem

**P-1.** Profiles are the **only** parameterization axis. There is no
per-facet parameter and no fourth addressing coordinate. A family of
selectivities is a family of profiles, and the tuple stays
`(dataset, profile, facet)`.

**P-2.** Inheritance must follow **the axis a family varies along**, not
a fixed list.

Today `apply_default_inheritance` hardcodes one axis. `neighbor_indices`
and `neighbor_distances` never inherit, which is correct for the *size*
axis — unfiltered ground truth depends on `base_count`, so a 1m profile
and a 10m profile cannot share it.

On a **selectivity** axis that is backwards. Every member shares one
`base_count`, so the unfiltered ground truth is invariant and should
inherit; what varies is `metadata_predicates`, `metadata_results`, and
the filtered ground truth. Under today's rule each selectivity profile
must re-declare `neighbor_indices` pointing at the same file — a
declaration duplicated per member, and a drift hazard the moment one is
edited.

**P-3.** A generator therefore declares **which facets it overrides**,
and everything else inherits. The override set is a property of the
spec, not of the facet table. `sized:` becomes one instance of the
general mechanism — its override set is the size-dependent facets —
rather than the only shape inheritance knows.

## 4. Attributes

**P-4.** `ProfileConfig` gains `attributes:`, an open key/value map,
mirroring the dataset-level `attributes` that `DatasetConfig` already
carries.

This is the substantive gap. A profile can state `base_count` and
`maxk`, but a selectivity has nowhere to live — so a family is
machine-readable **only through its member names**, and "which profile
is nearest 1% selectivity" is answerable only by parsing strings. Names
are identifiers; attributes carry meaning.

**P-5.** Conventional keys for a filtered family are `selectivity` (a
fraction in `0.0..=1.0`), `predicate_count`, and `k`. Conventions, not
requirements: unknown keys are preserved and reported, never rejected.

**P-6.** Attributes are **recorded from what a run realized**, never
asserted from what it was asked for. A sweep targeting 1% that achieves
1.2% on the corpus records `0.012`. A profile whose attributes describe
its request rather than its result is a benchmark scoring against a
number nobody measured — and the error is invisible, because the
declaration looks right.

**P-7.** A profile with no attributes reports as **undescribed**, not as
zero-valued. Absent is not `selectivity: 0.0`.

## 5. Families

**P-8.** Family membership is **exposed**, not inferred. `series_by_spec`
already records it; nothing surfaces it, so two profiles that are the
same corpus at different selectivities are indistinguishable from two
unrelated profiles. A consumer must be able to ask for a family's
members and their parameter values without parsing names.

**P-9.** A generated profile name that will be interpolated into a
filename must **not be all-digits**. The shard field is four digits and
always last before the extension, so `…__0010__0000.ivecs` has two
readings and neither is decidable
([srd-multifile-facet-shards.md](srd-multifile-facet-shards.md),
SH-101).

**P-10.** A family member is produced **whole**: one run of a sweep
writes every facet in the generator's override set, and the profile is
declared only after all of them are durable. A half-written member is
worse than an absent one — absent is a visible gap, half-written is a
selectable profile whose ground truth does not match its predicates.

## 6. Validation and reporting

**P-11.** `veks check` verifies that members of one family agree on
their **invariants**: same `base_count`, same inherited facets, differing
only in the generator's override set. A member that quietly differs in
`base_vectors` is not a parameterization of the same corpus, and
comparing results across it is meaningless.

**P-12.** `describe` and `list` report a profile's attributes, and for a
generated profile the family it belongs to and the parameter value that
distinguishes it. What a reader needs is *what switching profiles
changes*, which is exactly the override set plus the attribute that
varies.

## 7. What was actually missing

For the record, since a fourth coordinate was proposed to supply it:

| Need | Already there | Gap |
|---|---|---|
| Group of co-varying facets, selected as one | a profile | — |
| No duplication of shared facets | inheritance | — |
| Impossible to mix parameterizations | a profile is atomic | — |
| Addressing | `dataset:profile` | — |
| Machine-readable parameter value | — | **P-4** |
| Inheritance that follows the axis | one hardcoded axis | **P-2, P-3** |
| Discoverable family membership | `series_by_spec` | **P-8** (exposure) |

Three gaps, none of them a coordinate.

## 8. Declaration shape

```yaml
name: amazon-reviews-2023
profiles:
  default:
    base_vectors: base_vectors.fvec
    query_vectors: query_vectors.fvec
    metadata_content: metadata_content.slab

  1m:
    base_count: 1000000
    neighbor_indices: profiles/1m/neighbor_indices.ivecs
    neighbor_distances: profiles/1m/neighbor_distances.fvecs

  1m-sel001:
    base_count: 1000000
    attributes: { selectivity: 0.001, predicate_count: 1000, k: 100 }
    metadata_predicates: profiles/1m-sel001/metadata_predicates.slab
    metadata_results: profiles/1m-sel001/metadata_results.ivvec
    postfiltered_neighbor_indices: profiles/1m-sel001/postfiltered_neighbor_indices.ivecs
    postfiltered_neighbor_distances: profiles/1m-sel001/postfiltered_neighbor_distances.fvecs

  1m-sel010:
    base_count: 1000000
    attributes: { selectivity: 0.01, predicate_count: 1000, k: 100 }
    metadata_predicates: profiles/1m-sel010/metadata_predicates.slab
    metadata_results: profiles/1m-sel010/metadata_results.ivvec
    postfiltered_neighbor_indices: profiles/1m-sel010/postfiltered_neighbor_indices.ivecs
    postfiltered_neighbor_distances: profiles/1m-sel010/postfiltered_neighbor_distances.fvecs
```

`base_vectors` and `query_vectors` are inherited, not repeated. Under
P-2, `neighbor_indices` would be inherited from `1m` as well rather than
restated, since the selectivity axis does not change it.

```
veks datasets precache amazon-reviews-2023:1m-sel001
```

One profile, one coherent set. No new flag, no new coordinate.

## 9. Non-goals

- A fourth addressing coordinate, in any spelling.
- Selecting different parameterizations for different facets of one
  profile — unreachable by construction, and the reason profiles are the
  right unit.
- Parameter **axes**. A profile is a named point. If selectivity and `k`
  vary independently, each combination is its own profile with its own
  name; a product space would need a rule for what an unspecified axis
  means, and every such rule silently picks data for the caller.
- Rewriting `sized:`. P-3 generalizes what it hardcodes; the spec
  language and existing datasets are unchanged.

## 10. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | selectivity family declared as profiles | shared facets inherited, not duplicated |
| 2 | profile `attributes:` | round-trips through load, save, and catalog |
| 3 | attributes absent | reports undescribed, never `0.0` |
| 4 | recorded attributes | match the run's realized value, not its target |
| 5 | family membership | members and their parameter values enumerable without parsing names |
| 6 | generator override set | facets outside it inherit; facets inside do not |
| 7 | selectivity family | `neighbor_indices` inherited, not restated per member (P-2) |
| 8 | size family | `neighbor_indices` still per-profile — the axis decides |
| 9 | all-digit generated profile name | rejected (shard-field ambiguity, P-9) |
| 10 | sweep producing part of a member | member not declared |
| 11 | family member differing in `base_vectors` | reported by `check` |
| 12 | every dataset predating this | loads and reads unchanged |

**P-13.** Case 12 is the gate. Inheritance is what every sized dataset
in existence already depends on, so generalizing it (P-2, P-3) is the
one change here that can break data in the field. It ships only with
that case green.

## 11. Open

**The shape of the override declaration (P-3).** Whether a generator
names the facets it overrides explicitly, or derives them from which
facets its templates mention, is unsettled. Deriving is less to write
and cannot fall out of step with the templates; naming is explicit and
survives a template that mentions a facet it does not actually vary.
Worth deciding against a real `sized:` spec before implementing.
