# SRD — Profile layers and explicit inheritance

**Status:** implemented in the Rust reference implementation
(2026-09-05); the Java client (PL-8) is trued up separately. Decisions
taken during implementation are in §10.
**Scope:** `dataset.yaml` profile inheritance (`inherits:`), a new
format version that makes every parent explicit, the intermediate
*layer* profile, the content checks that hold a predicate set's members
together, the generators that write layered datasets, `veks check`,
and tessera's migration to a second series of predicate sets.

**Builds on** [srd-profile-parameterization.md](srd-profile-parameterization.md)
(a profile may name its parent with `inherits:`; inheritance follows
the axis a family varies along, P-2), [srd-profile-selectors.md](srd-profile-selectors.md)
(tags, naming, the required `predicates` tag, PS-23), and
[srd-dataset-format-version.md](srd-dataset-format-version.md) (a
version is a distinct type that adds cases, V-16 to V-19).

## 1. Problem

tessera's predicate set is declared once, on `default`, as
`profiles/base/predicates.slab`, and every sized profile inherits it.
That was right when there was one set: it is the same 10,000 predicates
at every size, and only the results index and the filtered ground truth
are per size.

The next series needs **different predicate sets at different
selectivities over the same metadata**: the uniform-form sets of PS-23,
several per size. They cannot live on `default`, because `default` is
what every profile inherits, and a profile can have one
`metadata_predicates`. Today a new set can only be expressed by
overriding the inherited facet in a profile that also has to redeclare
the size window and the unfiltered ground truth it shares with the
sized profile beside it, because there is nothing between `default`
and a sized profile to inherit those from.

Two things are missing:

- an **intermediate profile** that holds what the predicate sets share
  — the size window, the metadata window, the unfiltered ground truth —
  and nothing they differ on, so a set inherits from *it* rather than
  from a `default` that happens to carry another set;
- a parent that is **stated**. `inherits:` exists, but absent means
  `default`, so a reader of `dataset.yaml` cannot tell a profile that
  builds on `10m` from one that forgot to say so, and the loader's
  fallback is what let the predicate set hide on `default` in the first
  place.

## 2. What already works

- `inherits: <name>` names a parent; absent, the parent is `default`;
  `partition: true` has no parent; an unknown or self parent falls back
  to `default`; a cycle leaves its members with what they declare
  (parameterization §11).
- Inheritance follows the axis (P-2): across the size axis
  `base_vectors` and `metadata_content` inherit under the child's
  `base_count` window and the neighbor facets do not; across any other
  axis every facet inherits as is.
- `metadata_predicates` and `metadata_layout` inherit plainly on every
  axis; a child that declares one overrides it.
- Tags never inherit, and every generated profile carries its full tag
  set (PS-19).

Two facts about the reference implementation that this document
changes, found in review:

- the size axis is decided by **parent identity** — the size rules
  apply exactly when the parent is `default` — so a sized profile that
  named another sized profile would take that profile's ground truth;
- `metadata_results` inherits plainly on every axis, so a sized profile
  that omitted it would silently take `default`'s results at the wrong
  size. Every sized profile in the field declares its results, which is
  why nothing has been served wrong; the hole is closed first, on its
  own, before any of this lands (PL-14).

## 3. Layers

**PL-1.** A **layer** is a profile that exists to be inherited from. It
is an ordinary profile — it is openable, selectable, and precachable
like any other — and it is a layer by what it holds, not by a flag: it
declares the facets its children share and none they differ on. The
**size layer** `10m` holds the base window, the metadata window, and
the unfiltered ground truth for ten million records, and no predicate
facet. Opening it is the unfiltered 10m benchmark, which is a real
thing to open.

**PL-2.** A **predicate set** is a profile that names a size layer as
its parent and declares the **predicate group** whole:
`metadata_predicates`, `metadata_results`, and the pre- and
post-filtered ground truth, with the tags PS-23 and PS-11 require
(`predicates`, `selectivity` or `selectivity_ladder`, `family`, and for
a uniform set `form`). Everything else it inherits from the layer, and
the inheritance is on the selectivity axis, so the unfiltered ground
truth comes across unchanged (P-2).

**PL-3.** The predicate group is guarded by **content, not by
declaration shape**. A profile inherits whichever members it does not
declare — the natural mechanism, which is stable because every leaf
ends up a dense superset of its parent — and `veks check` verifies
that the members a profile ends up with belong together: the results
index holds one row per predicate of the slab it resolves to, each
filtered ground truth was computed from those results according to the
step records that produced it, and their row counts agree with the
profile's query set. A mismatch is reported naming the profile and the
facets that disagree. No combination of declared and inherited members
is refused at load, so tessera's sized profiles — slab inherited,
results declared — are exactly as legal as they are correct.

**PL-4.** A layer carries no predicate group **of its own**, with one
exception that is the shape tessera already has: `default` may declare
a facet that is invariant across the size axis — the mixed slab is one
file at every size — and every sized profile inherits it. `veks check`
reports a parent only when a **direct** child overrides a facet the
parent declares, since that is the arrangement in which a parent's
declaration is a decoy. A uniform set overriding `default`'s slab from
two levels down, through its size layer, is not reported: the layer
between them inherits the slab unchanged and is the unfiltered
benchmark either way.

**PL-5.** Layers may stack. `default` is the base layer; `10m` builds
on it across the size axis; `10m-uniform-2-1e-2` builds on `10m` across
the selectivity axis. Each step names its parent (PL-6), and the axis
of a step is **derived from `base_count`**: a step is on the size axis
when the child's `base_count` differs from the parent's effective one,
whatever the parent is called, and on the axis of the facets it
declares otherwise. That replaces the parent-identity rule (§2), so a
`20m` that names `10m` as its parent gets the size rules, and a test
pins it.

## 4. Explicit inheritance

**PL-6.** In `format_version` **3**, every profile other than `default`
states `inherits:`. Absent is a load refusal — "profile `10m-mixed`
names no parent; state `inherits: default` or the layer it builds on" —
not a fallback. `partition: true` remains the spelling for a profile
that builds on nothing and states no `inherits:`; a profile that
states both is refused as a contradiction. An unknown parent, a self
parent, and a cycle are refused too, each naming the profiles
involved: a stated 3 is a claim that every parent is real and every
profile has one meaning. Versions 1 and 2 keep their fallbacks
(parameterization §11), and `veks check` reports an implicit parent on
them as advisory, so the coming rule is seen before it refuses. An
**unversioned** file is held to version 1 (V-24): it is read as
pre-inheritance and pre-tag, so a named parent in it is refused with
the version to declare, never read generously.

Version 3 also carries the **tag schema** of PS-19: every profile in a
version-3 dataset carries every schema tag, `veks check` refuses one
that does not, and a reader below 3 presumes no schema exists. The two
promises — stated parents and stated tags — are one version, because a
dataset that names its parents is the same dataset that is addressed by
its tags.

**PL-7.** Version 3 is **derived from the structure** (V-19): a dataset
requires 3 when any profile names a parent other than `default`. A
reader that predates `inherits:` would fall back to `default` and serve
the wrong facets without a word, which is exactly the misread V-9
exists to refuse, so a named parent is a structural requirement, not a
promise. Version 3 adds one case to version 2 — *profiles that name
their parent, and therefore all state one* — and redefines nothing
below it (V-18). Every dataset in circulation stays at 1 or 2, because
none names a parent; a new dataset that does requires the updated
clients, which is the trade the project has made: existing clients
read existing datasets, new datasets may need new clients. A
version-3 dataset with no named parent does not exist under this
rule; one whose only parents are `default` downgrades by dropping
those lines (V-20), and one with a named parent does not downgrade at
all, since no lower version can say what it says.

**PL-8.** The Rust loader is the reference. The Java client is trued
up to it after this lands, as its compatibility contract already
provides for; nothing here is specified against Java, and the axis
rule it records today is superseded by PL-5.

## 5. Generators

**PL-9.** Bootstrap writes a layered dataset: `default` as the base
layer, one size layer per rung of the ladder, and one predicate-set
profile per (size, `predicates`, level) the spec asks for, each with
`inherits:` stated and its predicate group under `profiles/<name>/`. A
predicate slab that is invariant across every size — a mixed set is
one file at every size — is declared once, on the layer above the
profiles that use it (PL-4), and each of them declares its own
results and filtered ground truth. A slab that varies per set is
declared by the set.

**PL-10.** A pipeline step is refused **at plan time** when a facet it
writes is a predicate-group facet and the profile it names declares no
predicate facet: `evaluate-predicates` pointed at `10m-unfiltered` is
refused before anything runs, with the facet and the profile named,
while `compute-knn` on that layer is what the layer is for. The rule
follows the facet the step writes, so no command needs a declaration
of its own.

## 6. tessera

**PL-11.** tessera's existing profiles stay as they are: `10m` keeps
its inherited mixed set and its consumers, under the no-rename rule
(PS-22), and gains its tags and an explicit `inherits: default` on its
next run (PL-12). The new series is **additive**:

```yaml
format_version: 3                    # derived: a profile names 10m-unfiltered

profiles:
  default:
    attributes: { size: 495m, predicates: mixed, family: stratified }
    base_vectors: ...
    metadata_content: profiles/base/metadata_content.slab
    metadata_predicates: profiles/base/predicates.slab   # invariant across sizes, stays here (PL-4)

  10m:                                 # today's profile, content unchanged
    inherits: default                  # stated, as v3 requires
    base_count: 10000000
    attributes: { size: 10m, predicates: mixed, family: stratified, selectivity_ladder: [...] }
    metadata_results: profiles/10m/metadata_results.slab
    neighbor_indices: profiles/10m/neighbor_indices.ivecs
    ...

  10m-unfiltered:                      # the size layer for the new series (PL-1), hand-named (PL-13)
    inherits: default
    base_count: 10000000
    attributes: { size: 10m }
    neighbor_indices: profiles/10m/neighbor_indices.ivecs   # the same files 10m declares
    neighbor_distances: profiles/10m/neighbor_distances.fvecs

  10m-uniform-2-1e-2:                  # a predicate set (PL-2)
    inherits: 10m-unfiltered
    attributes: { size: 10m, predicates: uniform-2, selectivity: 1e-2, family: uniform, form: topic_l3.eq_citation_percentile.range }
    metadata_predicates: profiles/10m-uniform-2-1e-2/predicates.slab
    metadata_results: profiles/10m-uniform-2-1e-2/metadata_results.slab
    prefiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/prefiltered_neighbor_indices.ivecs
    prefiltered_neighbor_distances: profiles/10m-uniform-2-1e-2/prefiltered_neighbor_distances.fvecs
    postfiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/postfiltered_neighbor_indices.ivecs
    postfiltered_neighbor_distances: profiles/10m-uniform-2-1e-2/postfiltered_neighbor_distances.fvecs
```

`tessera:10m` means what it meant. `tessera:10m-unfiltered` is the
layer; it inherits `default`'s slab unchanged and is not reported
(PL-4). `tessera:size=10m,predicates=uniform*` is the new series.

**PL-12.** The migration is one idempotent finalize step that edits
`dataset.yaml` **textually** (PS-19's rule), checking before each edit
whether the line is already present: it writes the PS-19 tags onto
every existing profile and `inherits: default` onto every profile
other than `default` and the partitions. It moves no facet and no
file. The `format_version: 3` line is written by the same rule every
writer follows (V-5), the moment a profile names a parent other than
`default` — that is, when the new series is declared — and not before.
The step records what it wrote (PS-13); it edits the file that defines
the pipeline running it, which is why it is idempotent and runs in the
finalize pass, after every producing step.

**PL-13.** Profiles that a migration or an author writes are
**hand-named**, `10m-unfiltered` included; PS-21's naming rule governs
the sets a generator creates. A dataset bootstrapped under this
document names its size layers plainly — `10m` *is* the layer, and the
mixed set is `10m-mixed` — because there is no prior meaning of `10m`
to keep. No tag exists to produce a migration name, and the schema is
the same in every dataset.

**PL-14.** Landed first, on its own: `metadata_results` no longer
crosses the size axis. Across a size step it is declared or absent,
never inherited, with a test that a sized child of a parent carrying
results has none of its own unless it declares them. Every dataset in
the field declares its results per size, so nothing changes for them.

## 7. Non-goals

- Multiple parents, or mixins. A profile has one parent and the axis
  rule decides what crosses.
- An `abstract:` flag. A layer is openable; hiding it would remove the
  unfiltered benchmark it is.
- Renaming tessera's existing profiles. PS-22 forbids it, and the
  additive series does not need it.

## 8. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | v3 profile without `inherits:` | load refused naming the profile and the fix |
| 2 | v3 unknown, self, or cyclic parent; `partition: true` with `inherits:` | load refused naming the profiles; in v1 and v2 the fallback stands and `veks check` reports |
| 3 | `partition: true` alone in v3 | no parent, no `inherits:`, accepted |
| 4 | `inherits: 10m-unfiltered` from a set | base window, metadata window, and unfiltered ground truth inherited; predicate group its own |
| 5 | results index with the wrong predicate count, or filtered ground truth not derived from the profile's results | `veks check` reports the profile and the facets that disagree |
| 6 | a direct child overriding a facet its parent declares | `veks check` reports the parent; an override two levels down is not reported |
| 7 | two sets declaring one shared predicate slab | both load; the file is one, the declarations two |
| 8 | `20m` with `inherits: 10m` | the size rules apply: windows re-cut, ground truth not inherited |
| 9 | stacked layers `default` → `10m` → set | each step's axis derived from `base_count`; tags not inherited |
| 10 | a v2 dataset with any named parent | `min_format_version` is 3; a reader at 2 refuses it |
| 11 | v3 whose only parents are `default` | downgrades by dropping those lines; one with a named parent does not downgrade |
| 12 | `evaluate-predicates` on a layer | refused at plan time naming the facet and profile; `compute-knn` on the layer runs |
| 13 | tessera migration step | tags and `inherits: default` written by textual edit, comments intact, no facet moved, re-run writes nothing; `tessera:10m` opens the same facets |
| 14 | a sized child of a parent carrying `metadata_results` | inherits none (PL-14) |
| 15 | `veks check` on a v2 dataset with implicit parents | advisory report, load unchanged |
| 16 | an unversioned `dataset.yaml` with a named parent | load refused naming `format_version: 3`; `veks check` refuses the missing version outright (V-24, V-25) |
| 17 | a v3 profile missing a schema tag | `veks check` refuses it; the loader accepts |

## 9. Open

- Whether `tessera:10m` should one day be re-pointed at the layer, with
  the mixed set as `10m-mixed`, in a fresh dataset version rather than
  in place. PS-22 keeps the name where it is for this dataset.

## 10. Settled in implementation

**Declared means "not the parent's".** A profile declares a facet when
it carries a view whose file is not its parent's effective one; a
windowed cut of the parent's file is inheritance under the child's
count, not a declaration. This one predicate is what the template
gating (PL-9), the plan-time refusal (PL-10), the decoy report (PL-4)
and the `predicates` requirement (PS-23) read, so they cannot drift.

**Layered is version 3.** A dataset at `format_version` 3 is layered:
its generated rungs split into a size layer and a mixed set beside it
(`10m`, `10m-mixed`, PL-13), and a per-profile template runs only where
the profile declares the facet its command produces. Below 3 every
sized profile takes every template and holds every per-profile facet,
exactly as before, which is what keeps tessera and every dataset in
the field unchanged. The split is one idempotent pass over generated
rungs, run at load after expansion and by `stratify` before it saves,
so a compact file and an expanded file yield the same profiles.

**A set restates its count and takes the slab's tags.** The set
carries `base_count` equal to its layer's — the writer-side group
resolves no inheritance, and every consumer of a count reads the
profile — so the step from layer to set is at one size (PL-5). The
tags that describe the slab (`predicates`, `family`, `forms`,
`selectivity_ladder`, `form`, `form_shape`) move from the rung to the
set; the layer keeps `size` alone, as PL-11 draws it, and the migration
and generator tagging skip layers.

**Dependencies follow the parent chain.** A template dependency whose
instance was not emitted for a profile resolves to the nearest
ancestor's instance, so a set's evaluation waits on its layer's
unfiltered ground truth; a shared step fans in only to instances that
exist.

**Names that YAML reads as numbers.** A rung such as `100` is written
quoted where it names a parent and read back as a name by both
loaders, so a layered dataset with all-digit rungs states its parents.

**Schema tags at version 3.** `veks check` refuses a profile lacking a
schema tag that has a default; a naming tag declared `~` may be absent,
as the selectors SRD settled. The loader accepts either.

**Downgrade** is `veks prepare downgrade --to N`, a textual edit with
a backup: to 2 it drops `inherits: default` lines and the schema and
refuses a named parent; to 1 it also refuses a multi-file facet.

**Uniform sets are declared by a command, not by bootstrap.** `veks
prepare predicate-sets --form F --levels L [--sizes S]` declares one
set per size and level under its size layer — the rung itself in a
layered dataset, a hand-named `<rung>-unfiltered` beside a rung that
carries its own group (PL-11) — with the predicate group under
`profiles/<set>/` and one `generate predicates --strategy uniform` step
per set, by textual edit, lifting a file below 3 to 3 with the standard
schema. A per-profile template then reads the profile's own facet: an
option naming a file `default` declares becomes the path the profile
reads that facet from, and a step declared for the profile beside a
shared one is the instance it waits on. The strata verifier checks only
the profiles whose predicate facet is the slab it was given.
