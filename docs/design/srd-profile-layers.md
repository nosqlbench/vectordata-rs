# SRD — Profile layers and explicit inheritance

**Status:** draft
**Scope:** `dataset.yaml` profile inheritance (`inherits:`), a new
format version that makes every parent explicit, the intermediate
*layer* profile, the rule that a predicate set moves as one group, the
generators that write layered datasets, `veks check`, and tessera's
migration to a second series of predicate sets.

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
overriding the inherited facet in every profile that wants it, which
works by accident of override order and leaves the parent carrying a
set the child silently replaces.

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
  axis every facet inherits as is. The Java client implements the same
  rule.
- `metadata_predicates`, `metadata_results` (`predicate_results`), and
  `metadata_layout` inherit plainly on every axis; a child that declares
  one overrides it.
- Tags never inherit, and every generated profile carries its full tag
  set (PS-19).

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

**PL-3.** The predicate group moves **as one**. A profile either
declares every member of the group or inherits every member; declaring
`metadata_predicates` without `metadata_results`, or a filtered ground
truth without the predicates it was computed against, pairs files that
only mean anything together with files from another set (parameterization
§1), and the loader refuses it naming the members present and absent.
This is P-10's whole-member rule stated for inheritance: half a set is
worse than none.

**PL-4.** A layer carries no predicate group, and `veks check` reports
a profile that other profiles inherit from and that carries one, unless
every child inherits the group unchanged. A parent whose group is
overridden by a child is the arrangement that hid tessera's set on
`default`, and it is reported rather than allowed to look like a
design.

**PL-5.** Layers may stack. `default` is the base layer; `10m` builds
on it across the size axis; `10m-uniform-2-1e-2` builds on `10m` across
the selectivity axis. Each step names its parent (PL-6), and each
step's axis is decided as P-2 decides it: a child with its own
`base_count` is on the size axis, otherwise on the axis of the facets
it declares.

## 4. Explicit inheritance

**PL-6.** In `format_version` **3**, every profile other than `default`
states `inherits:`. Absent is a load refusal — "profile `10m-mixed`
names no parent; state `inherits: default` or the layer it builds on" —
not a fallback. `partition: true` remains the spelling for a profile
that builds on nothing, and it states no `inherits:`. An unknown or
self parent is likewise a refusal in version 3, where in versions 1 and
2 it falls back to `default` and is reported by `veks check`
(parameterization §11): a stated version is a claim, and version 3
claims that every parent is real.

**PL-7.** Version 3 is a distinct type (V-16) that adds one case to
version 2: *profiles whose parents are all stated*. It redefines
nothing below it (V-18), and a version-3 dataset downgrades to 2 by
deleting no information, since `inherits: default` is what version 2
assumed (V-20). The version is not derived from the structure (V-19)
because explicitness is not a shape; it is a **promise a writer
makes**, and bootstrap makes it for every dataset it creates from now
on. A version-2 reader refuses a version-3 dataset (V-9), which is the
cost of the promise being checkable; the clients in use both read
version 3 before any dataset states it.

**PL-8.** The loader's inheritance and the Java client's agree on the
version-3 rule, and the compatibility contract records it beside the
axis rule it already carries.

## 5. Generators

**PL-9.** Bootstrap writes a layered dataset: `default` as the base
layer with no predicate group, one size layer per rung of the ladder,
and one predicate-set profile per (size, `predicates`, level) the spec
asks for, each with `inherits:` stated and its predicate group declared
whole under `profiles/<name>/`. A predicate slab shared by several sets
— the mixed set is one file at every size — is one file **declared by
each set**; sharing is a fact about files, and the declaration is per
profile (SH-66's rule for shards, applied to facets).

**PL-10.** The stratified generator and the uniform-form generator fill
a predicate-set profile, never a layer: the step's profile is the set,
its outputs land under the set's directory, and the tags it writes are
the set's. A step whose profile is a layer is refused at plan time.

## 6. tessera

**PL-11.** tessera's existing profiles stay as they are: `10m` keeps
its inherited mixed set and its consumers, under the no-rename rule
(PS-22), and gains the tags of PS-19 on its next run. The new series is
**additive**:

```yaml
format_version: 3

profiles:
  default:
    attributes: { size: 495m, family: stratified }   # no predicate group after migration (PL-12)
    base_vectors: ...
    metadata_content: profiles/base/metadata_content.slab

  10m:                                 # today's profile, unchanged in content
    inherits: default
    base_count: 10000000
    attributes: { size: 10m, predicates: mixed, family: stratified, selectivity_ladder: [...] }
    metadata_predicates: profiles/base/predicates.slab      # declared, no longer inherited (PL-12)
    metadata_results: profiles/10m/metadata_results.slab
    neighbor_indices: profiles/10m/neighbor_indices.ivecs
    ...

  10m-unfiltered:                      # the size layer for the new series (PL-1)
    inherits: default
    base_count: 10000000
    attributes: { size: 10m }
    neighbor_indices: profiles/10m/neighbor_indices.ivecs   # the same files 10m declares
    neighbor_distances: profiles/10m/neighbor_distances.fvecs

  10m-uniform-2-1e-2:                  # a predicate set (PL-2)
    inherits: 10m-unfiltered
    attributes: { size: 10m, predicates: uniform-2, selectivity: 1e-2, family: uniform, form: topic_l3.eq+citation_percentile.range }
    metadata_predicates: profiles/10m-uniform-2-1e-2/predicates.slab
    metadata_results: profiles/10m-uniform-2-1e-2/metadata_results.slab
    prefiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/prefiltered_neighbor_indices.ivecs
    prefiltered_neighbor_distances: profiles/10m-uniform-2-1e-2/prefiltered_neighbor_distances.fvecs
    postfiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/postfiltered_neighbor_indices.ivecs
    postfiltered_neighbor_distances: profiles/10m-uniform-2-1e-2/postfiltered_neighbor_distances.fvecs
```

`tessera:10m` means what it meant. `tessera:10m-unfiltered` is the
layer. `tessera:size=10m,predicates=uniform*` is the new series, and
`tessera:size=10m,not(predicates=*)` is the layer by what it lacks.

**PL-12.** Moving the mixed set off `default` is a textual edit of
`dataset.yaml` (PS-19's rule): the `metadata_predicates` line moves
from `default` into each sized profile that inherited it, with the same
path, so no profile changes meaning and no file moves. After it,
`default` is a base layer and PL-4 holds. The edit and the
`format_version: 3` line are one `veks run` finalize step, recorded
like any other, so a rerun does not repeat them.

**PL-13.** A dataset bootstrapped under this document names its size
layers plainly — `10m` *is* the layer, and the mixed set is
`10m-mixed` — because there is no prior meaning of `10m` to keep. The
`-unfiltered` suffix is tessera's migration spelling only, and PS-21's
naming rule produces it from a schema tag `role: unfiltered` that the
migration writes and a fresh bootstrap never needs.

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
| 2 | v3 unknown or self parent | load refused; in v1 and v2 the fallback and `veks check` report as before |
| 3 | `partition: true` in v3 | no parent, no `inherits:`, accepted |
| 4 | `inherits: 10m-unfiltered` from a set | base window, metadata window, and unfiltered ground truth inherited; predicate group its own |
| 5 | set declaring `metadata_predicates` alone | refused naming the absent group members |
| 6 | layer carrying a group a child overrides | `veks check` reports it |
| 7 | two sets declaring one shared predicate slab | both load; the file is one, the declarations two |
| 8 | stacked layers `default` → `10m` → set | each step's axis decided by P-2; tags not inherited |
| 9 | v3 downgraded to v2 | `inherits: default` lines dropped, nothing else changed, loads identically |
| 10 | a generator step whose profile is a layer | refused at plan time |
| 11 | tessera migration step | `metadata_predicates` moved by textual edit, comments intact, `format_version: 3` added, `tessera:10m` opens the same facets |
| 12 | Java client | agrees on cases 1 to 9 |

## 9. Open

- Whether the migration spelling `10m-unfiltered` should instead be the
  layer name `10m` with today's `10m` renamed to `10m-mixed`, taking a
  one-time break of `tessera:10m` for a cleaner long-term layout. PS-22
  says no; the cost of `-unfiltered` is one odd name per size.
- Whether version 3 should also require `attributes:` on every profile
  (PS-19's full tag set), making the two promises one version, or leave
  tags to `veks check`.
