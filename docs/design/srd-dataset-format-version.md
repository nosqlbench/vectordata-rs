# SRD — `dataset.yaml` format version

**Status:** proposed
**Scope:** `DatasetConfig` (both loaders), catalog entries, `derive` /
`push` output, `veks check`.

## 1. Problem

`dataset.yaml` has no format version. Until now it did not need one:
every change to the schema was additive, and a reader meeting an
unfamiliar key ignored it and still read the data correctly.

Multi-file facets end that. A sharded declaration is the first thing a
dataset can say that an older reader **cannot** act on
([srd-multifile-facet-shards.md](srd-multifile-facet-shards.md)). That
SRD requires such a declaration to fail loudly rather than be misread
(SH-74), and it does — but as a *symptom*: a type error on `source`
where a string was expected, or `no such file: base_vectors__NNNN.fvec`.
Neither says what is actually wrong, which is that the reader is too
old.

**A version field turns that symptom into a diagnosis.**

## 2. Why now

Not because anything is broken today, but because a version rule is only
cheap to state **before** the format diverges. Introduced now, the rule
is "absent means 1, and 1 is everything that exists" — a statement with
no exceptions. Introduced after divergent datasets are in circulation,
it has to describe them retroactively, and every dataset written in
between is a special case.

There is one thing it cannot do, and the SRD says so plainly rather than
implying otherwise: **the field cannot help readers that already
exist.** A build shipped before this lands does not know to look for
`format_version` and will ignore it, exactly as it ignores every other
unfamiliar key. The failure it gets is still SH-74's symptom. What the
field buys is that every reader *from now on* gets the diagnosis — which
is why the cost of adding it only ever goes up.

## 3. The field

**V-1.** `dataset.yaml` carries an optional top-level
`format_version:`, a single non-negative integer, sibling to `name:`,
`attributes:` and `profiles:`.

**V-2.** **Absent means 1.** Every dataset in circulation omits it, and
they are all version 1 — not "unversioned", not a distinct state to
handle. A reader that special-cased absence would be writing a third
case for a set with no members.

**V-3.** A single integer, not `major.minor`. The version answers one
question — *is this reader capable of reading this dataset correctly* —
and there is no partial answer. A change an older reader can safely
ignore does not need a version at all; the `attributes` map already
absorbs additions without one.

## 4. What the number means

**V-4.** The version is a **minimum reader requirement**, not a
timestamp. It says "a reader needs at least this much to read me
correctly", not "this is when I was written".

**V-5.** A writer emits the **lowest version that describes the dataset
it actually wrote**, not the highest it knows. A new build writing an
unsharded dataset emits nothing, and that dataset stays readable by
every build that ever existed.

This mirrors the shard SRD's central property: sharding is generative,
not retroactive (SH-95). A new writer must not make old readers fail on
data they could have read, and a version stamped by capability rather
than by use would do exactly that.

**V-6.** A version is bumped only by a change that makes a dataset
**unreadable or misreadable** by the previous version's readers. Additive
changes that older readers ignore harmlessly do not bump it. Changes
that older readers would *misread* must bump it — and if the change
cannot be made to fail loudly on an old reader, the bump is not
sufficient and the change needs rethinking (SH-74).

## 5. The versions

**V-7.** Defined so far:

| Version | Introduced by | An older reader |
|---|---|---|
| 1 | everything before multi-file facets | — |
| 2 | a sharded facet declaration (uniform or explicit) | cannot resolve the facet; fails at load or open |

**V-8.** Version 2 is emitted **only when a dataset actually contains a
sharded facet**. A dataset all of whose facets are single files is
version 1 whatever wrote it (V-5).

## 6. Reading

**V-9.** A reader declares the highest version it supports and refuses
anything above it, naming both numbers:

```
dataset 'amazon-reviews-2023' requires format_version 2;
this build supports up to 1. Upgrade vectordata to read it.
```

That is the whole point of the field: a sentence naming the cause, in
place of a type error on `source` or a missing `__NNNN` file.

**V-10.** The refusal is at **load**, before any facet is opened. A
dataset the reader cannot understand must not be half-read — a profile
whose sharded facet failed while its unsharded ones succeeded is a view
with a hole in it, and a caller that checks only what it touched will
not find the hole.

**V-11.** A reader **must not** attempt a dataset above its version, even
when the parts it needs look familiar. "The facets I want are all
version 1" is a judgement the reader is not equipped to make: the
version exists precisely because it cannot tell what it is missing.

## 7. Writing and validation

**V-12.** Both loaders read the field identically —
`DatasetConfig`'s deserializer and the catalog path — for the same
reason shard realization is shared and not mirrored
([srd-multifile-facet-shards.md](srd-multifile-facet-shards.md), SH-90).
A dataset accepted by one route and refused by the other would make the
transport decide whether it is readable.

**V-13.** Catalog entries carry the version, so a consumer can refuse a
dataset **before fetching** rather than after.

**V-14.** A **stated** version lower than the content requires is a
declaration contradicting itself, and is refused at load — the same
class of fault as a record count that disagrees with its shards (SH-8).

**V-22.** An **absent** field is not a claim. It means 1 for the version
gate (V-2), but a dataset that never declared a version has not
*understated* one, and a reader new enough to notice the omission is new
enough to read the data. So absence plus higher-version content is a
note from `veks check` — "declare `format_version: 2` so older readers
get a diagnosis" — never a load failure. Refusing it would reject every
hand-written dataset for a field that helps no reader capable of
reading it anyway.

The distinction matters because V-2 makes absence *mean* 1. Without
V-22, that equivalence would turn every unannotated sharded dataset into
a self-contradiction, which is the opposite of what V-2 is for.

**V-23.** A version *higher* than the content requires is merely
conservative, and is reported as a note rather than an error.

## 8. The version is structural

A version carried only as an integer is a claim a writer makes about
itself, and a claim can drift from the thing it describes. The
requirements below make it a **property of the shape**, so it cannot.

**V-16.** Each version is a **distinct type**, not a flag on one shape.
A v1 declaration and a v2 declaration are different things in the model,
distinguishable without inspecting whether an optional field happens to
be set.

The alternative — one struct whose extra fields are `Option`, `None`
meaning v1 — cannot tell the two apart structurally. Every consumer
would re-derive the version by probing options, and each probe is a
restatement of the rule that can disagree with the others.

**V-17.** Version *n+1* **contains** version *n* by composition. A v1
declaration **is** a v2 declaration, spelled as the case that holds it —
not converted to one, not re-encoded. This is what makes the version a
tower rather than a series of unrelated schemas, and it is why a v1
dataset needs no migration to be read by a v2 reader.

**V-18.** A higher version **adds cases; it never redefines a lower
version's**. If a change would alter what an existing construct means,
it is not an extension and composition is a lie about it — the change
needs a different shape, or the lower construct needs deprecating in the
open.

**V-19.** The required version is **derived from the structure**, never
asserted. Each construct answers what it minimally requires; a
dataset's requirement is the maximum over its constructs. V-5's rule —
emit the lowest version that describes what was written — is then a fold
over the tree rather than a flag a writer sets, and a writer cannot
drift from it because it does not restate it.

**V-20.** **Downgrade is a typed operation**, and it succeeds exactly
when nothing in the dataset needs the higher version. `try_into_v1`
returning `Some` *is* the proof that a v1 reader can read it; there is
no separate check to keep in step, and no way to claim compatibility a
type does not support.

**V-21.** Malformed intermediate states should be **unrepresentable**
rather than validated. A sharded declaration requires its stride, its
count and its record count together (SH-47); modelling those as required
fields of the sharded case makes "`NNNN` without `shard_stride`" a
parse failure rather than a rule someone has to remember to check.

## 9. Non-goals

- Recording which build wrote a dataset. That is `veks_version`, an
  attribute, and it already exists — provenance, not compatibility.
- Content versioning. This says nothing about whether the *data*
  changed; two datasets with different vectors are both format 1.
- Ranges or negotiation. A reader supports `1..=N` for one `N`; there is
  no "supports 1 and 3 but not 2", because a version is cumulative by
  construction (V-4).
- Downgrade. A version-2 dataset cannot be read as version 1 by omitting
  the parts that need 2 — see V-11.

## 10. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | dataset with no `format_version` | loads as version 1, unchanged |
| 2 | `format_version: 1` explicit | identical behaviour to absent |
| 3 | version above what the build supports | refused at load, naming both numbers |
| 4 | refusal | no facet opened, no partial view |
| 5 | unsharded dataset written by a new build | emits no version; readable by older builds |
| 6 | sharded dataset written | emits `format_version: 2` |
| 7 | sharded content *stating* version 1 | refused at load — a declaration cannot understate itself |
| 7a | sharded content with no version field | loads; `check` notes the omission |
| 8 | version higher than content needs | a note, not an error |
| 9 | both loaders | accept and refuse identically |
| 10 | catalog entry | carries the version; refusal before fetch |
| 11 | every dataset predating this | loads and reads unchanged |
| 12 | v1 declaration in a v2 model | held as the v1 case, not re-encoded (V-17) |
| 13 | required version | derived by folding the tree, never read from a field (V-19) |
| 14 | `try_into_v1` on an unsharded dataset | `Some` — and that *is* the compatibility proof (V-20) |
| 15 | `try_into_v1` on a sharded dataset | `None`, with no partial conversion |
| 16 | `NNNN` without stride/count | a parse failure, not a validation rule (V-21) |

**V-15.** Case 11 is the gate, and case 5 is what makes it durable: the
field is worthless if adding it changes what older builds can read.
