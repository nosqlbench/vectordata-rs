# SRD — Facet variants (the fourth addressing coordinate)

**Status:** proposed
**Scope:** `vectordata` view/config model, `dataset.yaml`, catalogs,
`veks` pipeline, CLI facet selection, conformance.

## 1. Problem

Addressing today is a 3-tuple:

```
(dataset, dataset-profile, facet)
```

That is one facet of one kind per profile. It cannot express a facet
that exists in several *parameterizations at once* — most immediately,
precomputed post-filtered ground truth at several selectivities. A
profile wants `postfiltered_neighbor_indices` at 0.1%, at 1%, and at 10%
simultaneously, and a benchmark wants to name one of them.

The workarounds are both bad. Minting a dataset profile per selectivity
duplicates `base_vectors` and every other shared facet across profiles
that differ in one derived artifact. Inventing facet keys
(`postfiltered_neighbor_indices_sel001`) puts a parameter inside a name
that the facet spec treats as a closed vocabulary, and it defeats
`StandardFacet::classify`.

## 2. The fourth coordinate

```
(dataset, dataset-profile, facet, facet-variant)
```

**FV-1.** A **variant** is a named parameterization of one facet within
one dataset profile. This is the settled term for code, CLI, and YAML.

The coordinate was originally described as *profile_of_facet*, and
**facet profile** was the alternative spelling — it honors that phrasing
and keeps the four coordinates named symmetrically. `variant` was chosen
instead because *profile* already means the second coordinate: every
sentence would need a qualifier to stay unambiguous, "profile" versus
"facet profile", and that distinction is exactly the kind that erodes in
code comments and error messages until the two are used
interchangeably. A distinct word cannot erode into the wrong one.

`profile_of_facet` and *facet profile* name this same coordinate
wherever they still appear in discussion.

**FV-2.** A variant is a property of a facet **within a profile**, not
of the dataset. Two profiles of one dataset may offer different variant
sets for the same facet.

**FV-3.** Variants are **orthogonal to windows**. A window selects a
range of ordinals within a facet; a variant selects *which facet data*
the ordinals index into. Both may apply at once.

## 3. What a variant is not

**FV-4.** A variant is not a free-form label. It carries **attributes**
that describe the parameterization — for the motivating case,
selectivity. Attributes are declared, machine-readable, and reportable;
a consumer must be able to ask "which variant is nearest 1%
selectivity" without parsing the variant's name.

**FV-5.** Variant names are therefore identifiers, not encodings. The
name is for humans and for stable addressing; the attributes carry
meaning.

## 4. Naming and syntax

**FV-6.** A variant token matches `[a-z][a-z0-9_]*` — lowercase, starts
with a letter, **never all digits**. The digit prohibition is not
cosmetic: in the uniform shard layout the shard suffix is four digits
and always last before the extension (see
[srd-multifile-facet-shards.md](srd-multifile-facet-shards.md), SH-1 and
SH-2), so an all-digit variant would make
`postfiltered_neighbor_indices__0001__0000.ivecs` ambiguous. The
constraint applies only to that layout: explicit-form filenames are read
from the declaration and never parsed for a shard field, so they carry
no naming constraint (SH-50).

**FV-7.** The qualified facet name is `<facet>@<variant>`, e.g.
`postfiltered_neighbor_indices@sel001`. `@` is chosen because `:` is
already the path/namespace/window separator in source strings and the
`dataset:profile` separator in CLI specs, and `#` is the slab-namespace
fragment marker.

`@` is not new to the workspace: `vecd` already uses it as a version
selector on URL path segments (`…/@<sel>/…`), as a system-role owner
prefix (`@<level>`), and in `<ns>@<version>` cleanup targets. That is a
*compatible* homonym — every use reads "the qualifier follows" — and
there is no mechanical collision, because under FV-9 a variant never
appears in a filename or a URL path segment. `vecd` matches a segment
that *starts with* `@`; a qualified facet name embeds one. The two
grammars occupy different argument positions and never parse the same
token.

**FV-8.** The unqualified name `<facet>` remains legal and resolves to
the facet's **default variant**. Every existing call site, CLI
invocation, and stored config keeps working unchanged and unedited.

**FV-9.** On-disk filenames encode the variant as
`<basename>__<variant>.<ext>`, keeping the standard basename first so
`StandardFacet::classify` can strip the variant suffix the same way it
strips the shard suffix. Stripping order is: shard suffix (all-digits)
first, then variant suffix (non-digits), then `IDXFOR__` prefix.

## 5. Declaration in `dataset.yaml`

**FV-10.** `FacetConfig` gains a third shape alongside `Simple` and
`Detailed`:

```yaml
profiles:
  default:
    base_vectors: base_vectors.fvec          # unchanged
    postfiltered_neighbor_indices:
      default: sel010
      variants:
        sel001:
          source: postfiltered_neighbor_indices__sel001.ivecs
          attributes: { selectivity: 0.001, predicate_count: 1000 }
        sel010:
          source: postfiltered_neighbor_indices__sel010.ivecs
          attributes: { selectivity: 0.01, predicate_count: 1000 }
        sel100:
          source: postfiltered_neighbor_indices__sel100.ivecs
          attributes: { selectivity: 0.1, predicate_count: 1000 }
```

**FV-11.** Each variant body is a full facet declaration — it may carry
`window`, and it may be sharded in either form from the shard SRD. It is
also bound by the collapse rule: a variant backed by one file is spelled
as one file, never as a one-element series
([srd-multifile-facet-shards.md](srd-multifile-facet-shards.md), SH-4,
SH-72). A variant is a facet, not a filename.

**FV-12.** `default:` names the variant an unqualified reference
resolves to. It is **required** when a facet declares variants; there is
no implicit "first" or "only" default, because map order is not stable
and a one-variant facet will one day be a two-variant facet.

**FV-13.** The three `FacetConfig` shapes are mutually exclusive. A
declaration carrying both `source:` and `variants:` is an error.

## 6. Attributes

**FV-14.** `attributes` is an open key/value map, consistent with the
dataset-level `attributes` already in `DatasetConfig`. The schema is not
closed — a variant may describe itself with whatever the producing
pipeline knows.

**FV-15.** Conventional keys for filtered-result facets are
`selectivity` (fraction in `0.0..=1.0`), `predicate_count`, and
`k`. Conventions, not requirements: unknown keys are preserved and
reported, never rejected.

**FV-16.** Attributes are descriptive, never load-bearing for
resolution. Resolution is by variant name only. A consumer may *search*
attributes to pick a name, but the layer never resolves a facet by
attribute value — that would make the addressing space depend on
floating-point comparison.

## 7. Inheritance

**FV-17.** Default-profile inheritance (`apply_default_inheritance`)
carries a variant-bearing facet as a whole: the variant set and the
default together. A child profile does not inherit a single variant.

**FV-18.** A child profile may override the whole facet, or override
`default:` alone to select a different variant from the inherited set.
It may not add a variant to an inherited set — partial merge of a
variant map is a silent-divergence hazard for no expressive gain.

**FV-19.** Window inheritance (`inherit_with_window`) applies to every
variant in the set. A sized profile windows all of them identically,
because they index the same ordinal space.

## 8. API surface

**FV-20.** Every string-keyed facet entry point accepts a qualified
name: `facet(name)`, `facet_source(name)`, `facet_element_type(name)`,
`open_facet_typed(view, name)`, `open_facet_storage(name)`,
`prefetch_plan(facet, window)`, `prefetch(facet, window, fallback)`.
No new parameter is threaded through any of them — the fourth
coordinate travels inside the key that is already there.

**FV-21.** `facet_manifest()` reports one entry per **resolvable
qualified name**, plus one entry per unqualified name aliasing its
default. A consumer iterating the manifest to precache everything gets
every variant exactly once, and one alias it can recognize as such.

**FV-22.** New surface for enumeration and choice:

- `facet_variants(facet) -> Vec<VariantDescriptor>` — names, attributes,
  and which is default.
- `FacetDescriptor` gains `variant: Option<String>` so a manifest entry
  says which one it is.

**FV-23.** Requesting a variant that does not exist is
`UnknownVariant { facet, variant, available }` — and it lists what is
available, because the failure is almost always a typo or a stale
benchmark config.

**FV-24.** Requesting `<facet>@<variant>` on a facet with no variants is
an error, not a silent fallback to the single file. Silently ignoring a
qualifier the caller wrote is how a benchmark measures the wrong data.

## 9. CLI surface

**FV-25.** `--facets` accepts qualified names:

```
--facets base_vectors,postfiltered_neighbor_indices@sel001
```

**FV-26.** New selection flags, mirroring the YAML exactly:

| `dataset.yaml` | CLI |
|---|---|
| `variants:` / `default:` | `--facet-variant <facet>=<variant>` (repeatable) |
| *(all variants)* | `--all-variants` |

**FV-27.** `--facet-variant` and an `@`-qualified name in `--facets` are
the same operation by two spellings. Specifying both for one facet with
different values is an error, not a precedence puzzle.

**FV-28.** `describe` and `list` report the variant set with attributes.
The explore TUI presents variants as a selectable dimension of a facet,
not as separate facets.

**FV-29.** Precache and prefetch of an unqualified facet name operate on
the **default variant only**, never on all of them. Fetching three
selectivities because the user named the facet once is a surprise
measured in terabytes. `--all-variants` is how the user asks for that,
explicitly.

## 10. Creation

Variants exist to hold results someone **precomputed**. Something has to
compute them, and until this section existed the SRD described a shape
with no way to produce it.

**FV-35.** A variant set is produced by a **parameter sweep**: one
computation run per point, each writing its own file and its own entry.
Adding a third selectivity to a facet that has two is a run that
produces one file and one entry — it does not recompute or rewrite the
existing variants, which are finished artifacts (see
[srd-multifile-facet-shards.md](srd-multifile-facet-shards.md), SH-91).

**FV-36.** Variant **names are given, not derived**. A producer is told
`sel001`; it does not compute that string from `0.001`. Deriving names
from parameter values means encoding a float in an identifier, and that
encoding has no good answer for `0.0015`, for two parameters that round
alike, or for a value whose formatting changes between releases. FV-5
already says names are identifiers and attributes carry meaning; this is
that rule applied at the moment of writing.

**FV-37.** Attributes are **recorded from what the run actually did**,
never asserted from what it was asked to do. A sweep targeting 1%
selectivity that realizes 1.2% on the corpus records `0.012`. A variant
whose attributes describe its request rather than its result is a
benchmark scoring against a number nobody measured — and the error is
invisible, because the declaration looks right.

**FV-38.** The **first** variant a producer writes becomes `default:`.
Later runs add variants and **do not** change it. A default that moved
when a set grew would silently repoint every unqualified reference,
including ones in benchmark configs written months earlier (FV-8,
FV-12).

**FV-39.** Every variant of a facet is written in the **same format and
element type** (FV-31). A sweep that would produce a variant of a
different shape is producing a different facet, and must say so.

**FV-40.** Variant files follow FV-9's naming, and a variant that is
itself sharded follows the shard SRD — collapse rule included: one file
means one file, never a one-element series (SH-4, SH-72).

**FV-41.** Writing is atomic per variant, and the declaration entry is
added **after** the file and any sidecars are durable — the ordering the
shard SRD requires of a series (SH-37). A reader must never see a
variant named in a declaration whose file is not yet there.

## 11. Conformance and validation

**FV-30.** `veks check` verifies:

- every declared variant's source exists and classifies to the declared
  facet after suffix stripping
- `default:` names a variant that is present
- variant tokens match FV-6, including the not-all-digits rule
- no two variants of one facet resolve to the same source
- attribute values that use conventional keys have plausible types
  (`selectivity` numeric and within `0.0..=1.0`)
- every variant of a facet shares the same format and element type —
  variants differ in *content*, never in *shape*
- a variant carrying no attributes reports as **undescribed**, not as
  zero-valued: absent is not `selectivity: 0.0` (FV-37)

**FV-31.** FV-30's last clause is an invariant, not a nicety: a caller
that switches variants must not have to re-derive the element type.

## 12. Cache, publication, transport

**FV-32.** Variants are ordinary files; the cache layout
`<cache_root>/<dataset>/<relpath>` is unchanged. The relpath-collision
guard already rejects two facets mapping to one cache file, which now
also covers two variants.

**FV-33.** `push` publishes every variant and lists each in
`SHA256SUMS`. Publishing a subset of a declared variant set is invalid
for the same reason a partial shard series is.

**FV-34.** Catalog entries carry the variant set and its attributes, so
a remote consumer can enumerate and choose before fetching anything.

## 13. Error taxonomy

| Error | Raised when |
|---|---|
| `UnknownVariant{facet, variant, available}` | qualified name not in the set |
| `VariantsWithoutDefault{facet}` | `variants:` present, `default:` absent |
| `DefaultVariantMissing{facet, default}` | `default:` names an absent variant |
| `VariantOnUnvariantedFacet{facet, variant}` | qualifier on a plain facet |
| `MalformedVariantToken{token}` | fails FV-6 (including all-digits) |
| `MixedFacetDeclaration{facet}` | both `source:` and `variants:` |
| `VariantShapeDisagreement{facet, a, b}` | variants differ in format/elem type |
| `DuplicateVariantSource{facet, source}` | two variants, one file |
| `VariantDefaultReassigned{facet}` | a producer moved `default:` on an existing set |

## 14. Non-goals

- Resolving a facet by attribute value rather than by name.
- Variants that differ in element type or record shape.
- Partial merge of an inherited variant map.
- Variants of the dataset itself (that is a dataset profile).
- Cross-variant joins or automatic nearest-selectivity selection at the
  access layer. A consumer may implement that on top of
  `facet_variants()`; the layer will not guess.

## 15. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | unqualified name, varianted facet | resolves to `default:` |
| 2 | unqualified name, plain facet | unchanged from today |
| 3 | `@variant` resolves | correct source, window, shards |
| 4 | unknown variant | `UnknownVariant`, lists available |
| 5 | `@variant` on plain facet | error, never silent fallback |
| 6 | `variants:` without `default:` | rejected at load |
| 7 | all-digit variant token | rejected (shard ambiguity) |
| 8 | variant + window | both apply |
| 9 | variant + shard series | `__<variant>__<NNNN>.<ext>` resolves |
| 10 | inheritance | child inherits set + default |
| 11 | child overrides `default:` only | inherited set, new default |
| 12 | sized profile window | applies to every variant |
| 13 | `facet_manifest()` | each variant once, plus the alias |
| 14 | precache unqualified | default only, not all variants |
| 15 | `--all-variants` | every variant |
| 16 | conformance | every violation reported |
| 17 | variants differ in elem type | rejected |
| 18 | round trip | declare → push → catalog → open, variants intact |
| 19 | adding a variant to an existing set | existing files and entries untouched |
| 20 | first variant written | becomes `default:` |
| 21 | later variants written | `default:` unchanged |
| 22 | recorded attributes | match the run's realized values, not its targets |
| 23 | single-file variant | spelled as one file, never a one-element series |

## 16. Worked example

```yaml
name: amazon-reviews-2023
profiles:
  1m:
    base_count: 1000000
    base_vectors: base_vectors.fvec
    query_vectors: query_vectors.fvec
    neighbor_indices: neighbor_indices.ivecs
    metadata_predicates: metadata_predicates.slab
    postfiltered_neighbor_indices:
      default: sel010
      variants:
        sel001:
          source: postfiltered_neighbor_indices__sel001.ivecs
          attributes: { selectivity: 0.001, predicate_count: 1000, k: 100 }
        sel010:
          source: postfiltered_neighbor_indices__sel010.ivecs
          attributes: { selectivity: 0.01, predicate_count: 1000, k: 100 }
```

```
veks datasets precache amazon-reviews-2023:1m \
  --facets base_vectors,postfiltered_neighbor_indices@sel001
```

One dataset, one profile, shared base vectors fetched once, and the
benchmark names exactly the ground truth it intends to score against.
