# SRD — Profile selectors

**Status:** draft
**Scope:** the `dataset:profile` spec everywhere it is accepted (the
`vectordata` and `veks` command lines, `explore`, catalogs, completions,
`series_by_spec`), profile `attributes:` as the selection axis, the
generators that record them, `veks check`, and the Java client's
compatibility contract.

**Builds on** [srd-profile-parameterization.md](srd-profile-parameterization.md),
which gave a profile an open `attributes:` map (P-4) and required that
family membership be exposed rather than inferred from names (P-8).
This document is the selection language over that map, and the rule
for how generated profiles are tagged and named. It adds no coordinate:
a selector names a *set of profiles*, and a profile is still the unit
that is opened.

**Departs from P-6 for tags.** P-6 records an attribute from what a run
realised. A tag here describes what a profile was **asked to be** — the
level a set was generated for — because a name derived from a tag must
exist at bootstrap and never change, and a realised value does neither.
What a run realised is measured and reported by verification
(`verify-predicate-strata`, `analyze predicate-summary`), not stored as
a tag. A tag is a plan; a report is a measurement; the two are never
confused because only one of them is a tag.

## 1. Problem

A dataset directory is about to hold more than one family of profiles
at once. tessera today has one family, the size ladder (`100k` … `400m`),
and one predicate set per size, and its 10,000 predicates take 22
distinct forms ([analyze predicate-forms](../sysref/commands/analyze-predicate-forms.md)).
That many forms in one set makes a result hard to reason about: a
recall difference at 1% selectivity may be an index choice, a form the
engine handles badly, or the mix of both. The next step is predicate
sets of **uniform form** at **several selectivity levels**, side by side
in one dataset, so a run can hold every axis fixed but one.

Profiles are the right unit for that (P-1: profiles are the only
parameterization axis). What is missing is the ability to *address*
them by what they are rather than by what they are called. Today a
spec is `dataset:profile`, one literal name, and a family of
twenty-odd profiles at three selectivities and two forms is reachable
only by knowing sixty names. A consumer that wants "the 10m profiles at
selectivity between 1e-3 and 1e-2, uniform form" has to parse names,
which P-4 already identified as the wrong place for meaning.

Two smaller faults ride along and are fixed here because the grammar
cannot be specified without settling them:

- The spec is split at a colon by four different rules. `describe` and
  `explore` split at the **first** colon; `precache` classifies a URL
  head before splitting; the Java client splits at the **last** colon.
  A URL with a port or a path already reads differently across them.
- tessera's profiles carry no attributes at all. `base_count` is a
  field, not an attribute, and selectivity lives nowhere, so even the
  existing family is machine-readable only through its names (P-4's
  gap, still open for this dataset).

## 2. What already works

- `ProfileConfig.attributes` and `DSProfile.attributes`: an open map of
  YAML scalars, preserved through load, save, and the catalog (P-4,
  acceptance case 2).
- `TestDataGroup::attribute(name)` reads a dataset-level attribute;
  profiles have the map but no accessor beyond the struct field.
- `series_by_spec` records which generator produced which profiles
  (P-8), unsurfaced.
- `profile_names()` returns declaration order, size-sorted by the
  generators that write sized families.
- `describe` and `list` print a profile's attributes (P-12).

## 3. The spec

**PS-1.** A dataset spec is `<dataset>` or `<dataset>:<selector>`. The
dataset part is what it is today: a catalog name, a directory, or a
URL. The selector part is either a **profile name** or a **selector
expression**. A spec with no selector means the profile `default` on
**every** surface. `precache` without a profile iterates every profile
today; that exception ends, and "every profile" is spelled
`dataset:profile=*`, the same on every surface. The old form is
refused with a message naming the new spelling rather than silently
meaning less than it used to.

**PS-2.** The dataset part is found by its **shape**, and the selector
begins at the first colon after it. A URL is recognised by scheme and
authority, so its `:` in `https://` and in a port belong to the head; a
path is recognised by its separators or a drive letter, which the
precache resolver already protects; anything else is a catalog name,
which contains no colon. So `https://host:8080/ds` is a dataset with no
selector, `https://host/ds:10m` selects `10m`, and
`tessera:profile=^a:b$` selects on the regex `^a:b$` because the head
ended at the first colon. A malformed selector is reported **as a
selector error**, with the position and what was expected, never as a
dataset that could not be found. One rule for every surface, Java
included; the four rules in §1 go.

**PS-3.** The grammar:

```
spec      := dataset [ ":" selector ]
selector  := name | expr
expr      := term { "," term }                 -- "," is AND
term      := "and" "(" expr ")"
           | "or"  "(" expr ")"
           | "not" "(" expr ")"
           | atom
           | name                                 -- shorthand for profile=name
atom      := key op value
key       := ident { "." ident }                -- dotted keys reach into a map-valued attribute
op        := "=" | "!=" | "<" | "<=" | ">" | ">="
value     := quoted | bare
name      := ident
ident     := [A-Za-z0-9_][A-Za-z0-9_.-]*
```

`quoted` is single- or double-quoted, and inside quotes every character
is literal, commas, parentheses, and colons included. `bare` runs to
the next `,` or `)` at the current nesting depth. Whitespace around
tokens is ignored.

**PS-4.** Every profile carries an **automatic tag** `profile` whose
value is its name; nothing declares it and nothing can override it. A
selector that is a bare `name` is the atom `profile=<name>` with a
**literal** value: `tessera:10m` selects exactly the profile called
`10m`, as it always has. `profile` is otherwise a key like any other,
so `tessera:profile=10m*` and `tessera:profile=^1m-.*$` are selectors
on the profile's name, and `and(profile=10m*,family=uniform)` composes
it with the declared attributes. A bare name is accepted **anywhere** a
term is, as the same shorthand, so `or(10m,20m)` is
`or(profile=10m,profile=20m)`; a token with no operator can only be a
name, because an atom always carries one.

**PS-5.** How a value is read is decided by its spelling, in this
order, so that every value has exactly one reading:

| Spelling | Reading |
|---|---|
| starts with `^` or ends with `$` | regular expression, matched against the whole canonical text |
| contains `*`, `?`, or `[` | glob: `*` any run, `?` one character, `[...]` a class |
| `lo..hi` where both parse as numbers | numeric half-open interval `[lo, hi)` |
| parses as a number | number |
| `true` / `false` | boolean |
| anything else | literal string |

A **number** is read by the window grammar's count rule, so it may
carry a count suffix: `100k`, `10m`, `128mi`, `1g` are numbers, as
`1e-3` and `0.001` are. A literal that would otherwise read as one of
the above is quoted: `profile='^literal'`. Intervals use the same `..`
the window grammar uses, with the same half-open meaning, so a reader
who knows one knows both.

Comparison is **case-insensitive** throughout: keys, literal values,
globs, and regular expressions fold case, and so do count suffixes, so
`family=Uniform` matches `uniform` and `10M` is `10m`. Two attribute
keys or values that differ only in case are therefore the same key or
value, and `veks check` reports a dataset that declares both.

The regular-expression dialect is the **RE2 subset as implemented by
the Rust `regex` crate**, which is normative: no lookaround, no
backreferences. A glob is `*`, `?`, and `[...]` classes, nothing more.
The Java client validates a pattern against that subset and refuses one
outside it rather than accept a selector Rust cannot evaluate (PS-17).

**PS-6.** An atom matches a profile by comparing the value against the
profile's attribute of that key, under the operator:

- `=` and `!=` with a literal, glob, or regex compare the attribute's
  **canonical text**: strings as written, numbers as YAML would
  serialise them, booleans as `true`/`false`.
- `=` with a number matches when the attribute is numeric and the two
  are equal after parsing (`1e-3` matches `0.001`); with an interval,
  when the attribute is numeric and inside it. `!=` is the negation.
  An attribute is **numeric** when it is a YAML number or a string that
  parses under the count rule, so `size: 10m` compares as ten million
  and `size>=100m` selects the rungs from `100m` up.
- `<`, `<=`, `>`, `>=` require a numeric value and a numeric
  attribute; a string attribute that is not a count makes the atom
  false, never an error, because a family may legitimately mix
  described and undescribed members (P-7).
- An attribute whose value is a **list** matches when any element
  matches. Each atom is existential on its own: `ladder>=1e-3,ladder<1e-2`
  is satisfied by two different elements, and the one-element test is
  the interval `ladder=1e-3..1e-2`. An attribute whose value is a
  **map** is reached with a dotted key and matches nothing as a whole.
- An **absent** attribute matches nothing under any operator,
  including `!=`. Absent is not a value (P-7). `not(key=x)` is the way
  to say "anything but x, including undescribed".

**PS-7.** Some of what describes a profile is a field, not an
attribute. These are exposed to selectors as **structural keys** and
read before the attribute map: `profile` (the automatic name tag of
PS-4), `base_count`, `maxk`, `partition`, and `inherits`. An attribute
declared under one of those keys is shadowed and reported by `veks
check` (PS-15); it is never silently the one that wins. `name` is not
reserved: an attribute called `name` is an ordinary attribute.
Structural keys are read from the profile **as loaded**, after
inheritance, so an inherited `base_count` is as selectable as a
declared one.

**PS-8.** `,` and `and(...)` are the same conjunction; `or(...)` is a
disjunction over its comma-separated terms; `not(...)` negates the
conjunction of its terms. Terms nest without limit. There is no
operator precedence to learn because there are no infix operators:
`a=1,or(b=2,c=3)` is the only way to write that expression.

A key may appear **any number of times** in a selector. Each occurrence
is an independent atom, and the junction it sits in decides how the
occurrences combine: under `,` or `and(...)` every atom on the key must
hold, under `or(...)` any one of them. So
`selectivity>=1e-4,selectivity<1e-2` is an intersection spelled as two
bounds, `or(size=10m,size=100m)` is a union of two sizes, and
`or(profile=10m*,profile=^1m-.*$)` unions two name patterns. A repeated
key is never a map lookup and never overrides an earlier occurrence; a
selector is a formula, not a form. On a scalar attribute two different
`=` atoms on one key under `,` match nothing, which is the truthful
answer and the reason `or(...)` exists.

## 4. Cardinality

**PS-9.** A selector yields the **set** of matching profiles, in the
dataset's declaration order. Which surface receives it decides what a
set means:

- **Set surfaces** act on every match: `precache`, `ping`, `purge`,
  `list`, the explorer's picker (the selector is the row filter), and
  the Java client's `openProfiles`. Zero matches is an error that
  prints the profiles and attributes the dataset does have. `purge`
  over a set removes only the files that **no unmatched profile
  references**: the base shards every profile shares survive a purge of
  one family, and the command reports what it kept and which profile
  kept it. Freeing everything is `profile=*`.
- **Single surfaces** need exactly one: `describe`, `explore
  --dataset`, the `explain-*` commands, a `veks` pipeline's `--profile`,
  and the Java client's `openProfile`. More than one match is an error
  that lists the matches and says to narrow the selector or name one;
  it never picks the first. Zero is the same error as above.

**PS-10.** A bare name is never ambiguous: it matches one profile or
none. Existing specs therefore behave exactly as before on every
surface, which is the compatibility guarantee of this document.

## 5. Recording the axes

**PS-11.** A selector is only as good as the attributes it reads, and
here they come from the plan: bootstrap and the generators write what
each profile was generated **for**, and verification measures what
came out. This document adds the conventional keys the two families
that motivate it need, extending P-5:

| Key | Type | Meaning |
|---|---|---|
| `size` | count | the size ladder rung as spelled in the name (`10m`); numeric under the count rule |
| `predicates` | identifier | **required** on any profile that declares a predicate facet: the structural class of that facet, `mixed` or `uniform-<n>` (PS-23) |
| `selectivity` | number | the selectivity level a single-level predicate set was generated **for** (`1e-2`), not what it realised |
| `selectivity_ladder` | list of numbers | the decades a stratified set was generated for, most to least selective |
| `form` | identifier | the predicate form of a uniform set as a **derived id**: the fields and access kinds of the form in canonical order joined by `_`, `topic_l3.eq_citation_percentile.range`, so it is an identifier PS-21 can name with; `stratified` for a stratified set |
| `form_shape` | string | the rendered form, `(citation_percentile >= ? AND topic_l3 = ?)`, descriptive and never a naming tag |
| `forms` | number | how many distinct forms the set holds |
| `family` | string | the generator family the profile belongs to (P-8, surfaced) |
| `predicate_count`, `k` | number | as in P-5 |

Conventions, not requirements: unknown keys are preserved and
selectable (P-5). Realised selectivities are reported by verification
and by `analyze predicate-summary`; they are measurements, not tags.

**PS-12.** The sized-profile derivation writes `size` for every member
it produces; a predicate generator writes `family` for the set it
fills, the stratified one adding `selectivity_ladder` and `forms`, a
uniform-form one `selectivity`, `form`, and `form_shape`. tessera's
next `veks prepare stratify` and pipeline run therefore give its
existing profiles the tags they lack (§1), with no change to their
names or files.

**PS-13.** A step that writes attributes records the values it wrote in
its step record, beside its outputs. On the next run a difference
between the record and the yaml, or a changed input that would change
a written value, marks the step stale exactly as a changed output
would, and the run reports the attribute by name. A hand edit to a
generated tag is therefore reported, never silently overwritten and
never silently kept.

**PS-23.** Every profile that declares a predicate facet
(`metadata_predicates`) carries the standard tag `predicates`, and
`veks check` refuses a profile that declares the facet without it. The
value names the **structural class** of the facet, which is what a
result can be reasoned about against:

| Value | Meaning |
|---|---|
| `mixed` | more than one predicate form is present; tessera's stratified sets today, 22 forms |
| `uniform-<n>` | every predicate takes one and the same form, and that form has exactly `n` **parts** in its junction, conjunctive or disjunctive: `uniform-1` is single-part predicates, `uniform-2` two-part junctions, `uniform-3` three |

A uniform set may include predicates that act as no-ops to hold the
form constant; they still take the form, and the class still holds. The
tag is a plan like every tag (PS-11): the generator writes the class it
was asked for. Verification then holds the facet to it: `veks check`
runs the form census of `analyze predicate-forms` over the facet and
refuses `uniform-<n>` when the facet holds more than one form or its
form has other than `n` parts, and refuses `mixed` when it holds
exactly one — a set that is uniform in fact must say so, because a
consumer selecting `predicates=mixed` is asking for the hard case.
`predicates` is a naming tag, ordered in the schema after `size` and
before the level, so a set is `10m-uniform-2-1e-2`; `form` (PS-11)
still says *which* form a uniform set takes and is not a naming tag
unless a dataset holds several uniform forms at one level, in which
case the schema lists it too.

## 6. Tags at creation

**PS-19.** A dataset created by `veks prepare bootstrap` declares a
**tag schema**: the ordered list of naming tags and a default value for
each, kept in `dataset.yaml` beside the profiles. `veks run` applies
it: every profile it declares or fills carries **every** schema tag,
the defaults filled in where the profile does not set one, and a
generator writes the tags of the set it was asked to produce. Tags are
**denormalised**: reading one profile answers every key without
consulting `default` or the schema, and a selector never meets an
absent schema tag on a generated profile. Keys outside the schema are
still free (P-5); they are simply not part of a name. The schema, the
naming rule, and the required tags are **version-3** semantics
([srd-dataset-format-version.md](srd-dataset-format-version.md) V-7,
V-24): a reader presumes no schema on a dataset below 3 or without a
version, whose `attributes:` remain plain, selectable data (P-4) and
nothing more. Attributes **never inherit**: a profile carries exactly the tags written on it,
whatever its `inherits:` parent carries, and the loader pins that with
a test. Writing tags into an existing `dataset.yaml` is a **textual
edit** of the profile's own lines, preserving every comment and every
other line, never a serializer round trip.

**PS-20.** `size` is a schema tag. The size-ladder rung that was the
whole of a sized profile's name — `10m` — is the value of its `size`
tag, spelled exactly as the name was. The `default` profile carries
`size` too, valued as the rung spelling of its own `base_count` — for
tessera that is `495m`, which is what the base holds, whatever the
dataset is called — so `size>=100m` selects it like any rung under the
count rule.

**PS-21.** A generated profile is **named by its tags**: the values of
the naming tags it carries, in schema order, joined by `-`. A
`{size: 10m, predicates: uniform-2, selectivity: 1e-2}` profile is
`10m-uniform-2-1e-2`; a sized profile whose only naming tag is `size`
is `10m`, as before. Every naming tag is a **planned** value known at
bootstrap (the level, the family, the form id), which is what lets the
name exist before a generator runs and stay put after it. The name is
derived and never parsed — the tags are the truth (P-4), which is why
a value may itself contain `-`. Values
that reach a name obey P-9: filename-safe characters, and an all-digit
name is refused where a shard field would follow it. Two profiles whose
naming values are identical would have one name, and the generator
refuses to write the second rather than invent a suffix (it is also
the PS-15 case a selector cannot tell apart). `default` keeps its name;
it is the one name every spec without a selector relies on. The rule
governs profiles a **generator** creates; a profile an author or a
migration writes by hand carries the name it is given.

**PS-22.** Names are **stable**. A profile already declared is never
renamed by a later run, whatever its tags say: `veks run` on a dataset
created before its schema existed adds the schema tags to every profile
idempotently and leaves every name alone, so tessera's `10m` stays
`10m` while gaining `size: 10m` and `family: stratified`. The naming
rule governs profiles created once the schema exists. A dataset that
wants the new names is derived or bootstrapped afresh.

## 7. Surfaces

**PS-14.** The selector is parsed once, by one parser in `vectordata`,
and every surface that accepts a dataset spec calls it: the `vectordata`
subcommands, `explore`, every `veks` command that takes `--dataset` or a
`dataset:profile` argument, catalog resolution, and completions. The
explorer's picker uses the same parser for its filter box, so what a
user types there is what they can paste on a command line. On a
command line a selector with a regex, a glob, or a quoted value is
itself double-quoted, `"tessera:form='topic_l3.eq_citation_percentile.range'"`,
and every command's help shows that idiom once.

**PS-15.** `veks check` reports, per dataset:

- an attribute whose key is a structural key (PS-7), shadowed;
- an attribute whose value is a map, which a selector cannot compare;
- a family (P-8) whose members do not all carry the key that
  distinguishes them, since the family is then addressable only by
  name, which is the fault this document exists to remove;
- two profiles whose attributes are identical on every key, which a
  selector can never tell apart;
- a profile declaring a predicate facet without `predicates`, or
  whose `predicates` value the facet's form census contradicts (PS-23).

**PS-16.** Completion offers, after `dataset:`, the profile names, the
automatic `profile=`, and the attribute keys of that dataset's profiles
each followed by `=`;
after `key=`, the distinct values that key takes, which for a level tag
are the levels; after `,`, the same again. The completion spec ([11a](../sysref/11a-completion-spec.md))
gains these as one more scope over the catalog data it already holds.

**PS-17.** The Java client implements the same grammar and cardinality
rules and records them in its compatibility contract, with
`openProfile` refusing more than one match and a new `openProfiles`
returning the set. Its last-colon split is replaced by PS-2, so the two
runtimes agree on every spec, including ones with a URL head.

## 8. Declaration shape

The family this document is for, once the generators of §5 have run:

```yaml
profile_tags:                       # the schema: naming order, then defaults
  size: ~                           # written per profile from base_count
  predicates: ~                     # required wherever a predicate facet is declared (PS-23)
  selectivity: ~                    # no default: only single-level sets carry it
  family: stratified                # not a naming tag: the generator family

profiles:
  default:
    attributes: { size: 495m, predicates: mixed, family: stratified }
    base_vectors: { source: profiles/base/base_vectors__NNNN.fvecs, shard_stride: 100000000, shard_count: 5, record_count: 495930736 }
    ...

  10m:
    base_count: 10000000
    base_vectors: { source: profiles/base/base_vectors__NNNN.fvecs, shard_stride: 100000000, shard_count: 5, record_count: 495930736, window: "[0..10000000]" }
    neighbor_indices: profiles/10m/neighbor_indices.ivecs
    metadata_predicates: profiles/base/predicates.slab
    metadata_results: profiles/10m/metadata_results.slab
    attributes: { size: 10m, predicates: mixed, family: stratified, selectivity_ladder: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], forms: 22 }

  10m-uniform-2-1e-2:               # named by its tags (PS-21, PS-23)
    inherits: 10m
    attributes: { size: 10m, predicates: uniform-2, selectivity: 1e-2, family: uniform, form: topic_l3.eq_citation_percentile.range, form_shape: "(citation_percentile >= ? AND topic_l3 = ?)" }
    metadata_predicates: profiles/10m-uniform-2-1e-2/predicates.slab
    metadata_results: profiles/10m-uniform-2-1e-2/metadata_results.slab
    postfiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/postfiltered_neighbor_indices.ivecs
    postfiltered_neighbor_distances: profiles/10m-uniform-2-1e-2/postfiltered_neighbor_distances.fvecs

  10m-uniform-2-1e-3:
    inherits: 10m
    attributes: { size: 10m, predicates: uniform-2, selectivity: 1e-3, family: uniform, form: topic_l3.eq_citation_percentile.range, form_shape: "(citation_percentile >= ? AND topic_l3 = ?)" }
    ...
```

And the specs that address it:

```
tessera:10m                                          the stratified set, as today
tessera:size=10m,predicates=uniform*                 both uniform sets
tessera:predicates=mixed                             every mixed-form set, the hard case
tessera:size=10m,selectivity=1e-3..1e-2              the uniform set nearest 1%
tessera:family=uniform,selectivity<1e-3              every uniform set below 0.1%, any size
tessera:profile=^10m-uniform-2.*$                    the same two, by name
tessera:not(family=uniform)                          the stratified sets and anything undescribed
tessera:or(size=10m,size=100m),predicates=uniform-2  both sizes' two-leaf uniform sets
tessera:selectivity>=1e-4,selectivity<1e-2           two bounds on one key, an intersection
```

The single-surface rule (PS-9) makes the second spec an error on
`describe` and a filter on the picker, which is the difference between
those surfaces stated once.

## 9. Non-goals

- Selecting facets within a profile. A profile is opened whole (P-1);
  a selector chooses profiles.
- Selecting across datasets. The dataset part of a spec is one dataset;
  a catalog-wide query is a different tool.
- Arithmetic or functions over attribute values beyond comparison and
  intervals.
- A default selector per dataset. Absent means `default`, as today.
- Making predicate sets uniform. That is the follow-on this document
  exists to make addressable, and has its own SRD.

## 10. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | `ds:10m` on every surface | the profile named `10m`, exactly as before |
| 2 | `ds` with no selector | `default` on every surface; `precache ds` is refused naming `ds:profile=*` |
| 3 | `https://h:8080/ds`, `https://h/ds:10m`, `ds:profile=^a:b$` | dataset alone; `10m`; regex `^a:b$` (PS-2) |
| 4 | literal, glob, regex, number, interval, boolean values | each read per PS-5, quoted literal not reinterpreted |
| 5 | `selectivity=1e-3` against `0.001`; `size>=100m` against `size: 128mi` | match numerically under the count rule |
| 5a | `Family=UNIFORM`, `10M` | case folds on keys, values, and suffixes |
| 5b | a lookaround regex | refused by both runtimes as outside the subset |
| 6 | `selectivity<1e-2` against a string attribute | false, no error |
| 7 | absent attribute under `!=` | no match; `not(key=x)` matches it |
| 8 | list-valued attribute | matches on any element |
| 9 | `a=1,or(b=2,c=3)`, `not(a=1,b=2)`, nested | the expected sets |
| 9a | a key repeated: `s>=1e-4,s<1e-2`; `or(size=10m,size=100m)`; `size=10m,size=100m` | intersection; union; empty on a scalar attribute |
| 10 | structural keys | `profile`, `base_count`, `maxk`, `partition`, `inherits` select; a shadowing attribute is reported |
| 11 | two matches on `describe` | error listing both; on `precache` both are fetched |
| 11a | `purge ds:family=uniform` | shared base shards kept and reported; the family's own files removed |
| 12 | zero matches | error listing the dataset's profiles and attributes |
| 13 | sized derivation and stratified generation | members carry the keys of PS-11 |
| 14 | attributes survive load, save, catalog | as P-4 case 2, now including lists |
| 15 | completion after `ds:`, `ds:key=`, `ds:a=1,` | names and keys; values; both again |
| 16 | Java contract | same sets as Rust for every case above |
| 17 | bootstrap + run | every profile carries every schema tag, defaults filled, the generators' planned values written; `default` carries `size` |
| 18 | generated names | `{size: 10m, predicates: uniform-2, selectivity: 1e-2}` is `10m-uniform-2-1e-2`; a `size`-only profile is `10m`; identical naming values are refused |
| 18a | `predicates` tag | a predicate facet without it is refused; `uniform-2` on a 22-form facet is refused; `mixed` on a one-form facet is refused; tessera's sets check as `mixed` |
| 19 | run on a pre-schema dataset | tags added by textual edit with comments intact, no profile renamed |
| 20 | child of a tagged parent | carries only its own tags |
| 21 | `generate-catalog` | profile attributes present in `catalog.json` and `catalog.yaml` |
| 22 | a hand-edited generated tag | the writing step reports stale, naming the attribute |
| 23 | `or(10m,20m)` | the two named profiles |

**PS-18.** Case 1 is the gate. Every dataset in the field is addressed
by name today, and a selector language that changed what one of those
specs means would be a regression dressed as a feature.

## 11. Open

- Whether a selector may follow a `--profile` flag as well as a spec
  suffix. The draft says yes wherever `--profile` is accepted, with the
  same cardinality rule as the spec.
- Whether the picker should show a profile's attributes as columns once
  they exist, so a filter typed there has something to look at.
- What the uniform-form generator's `form` attribute should be when a
  set deliberately includes predicates that act as no-ops to keep the
  form constant, which is the question of the follow-on SRD.
