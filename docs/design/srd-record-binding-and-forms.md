# SRD — Record binding, and forms

**Status:** proposed
**Scope:** `vectordata::records`, `formats::anode_vernacular`, the slab
namespace set, `metadata_schema`.

**Depends on**
[metadata-facets-and-layout-namespace.md](metadata-facets-and-layout-namespace.md)
(Stage 4 — the record reader) and
[srd-multifile-facet-shards.md](srd-multifile-facet-shards.md) (SH-96,
SH-98 — sharded slabs).

## 1. Problem

A workload generator addresses a metadata or predicate facet **per
cycle**: cycle → ordinal → record → a bound operation. The reader
delivers the first three. The fourth has no contract.

What exists in its place is the vernacular layer, and it is the wrong
currency:

```
render(&node, Vernacular::Cql)  ->  (7, 'a''b')
```

Literals inlined, strings hand-quoted, and the field **names dropped** —
`to_cql` iterates `(_name, value)` and emits positionally. For a
prepared statement that means no statement reuse, escaping standing in
for the bind protocol, and an allocation per cycle. It is the right
output for `veks inspect` and the wrong one for driving load.

An integrator who notices this reaches past `Vernacular` into
`MNode.fields`, which is correct and is also a sign the layer is
mis-signalling: the codec named `Cql` is the one a CQL workload must
not use.

**OT-1.** Binding and rendering are **different contracts**, and both
are legitimate. Rendering produces text for a human. Binding produces
named, typed values for a driver. Neither is a degraded form of the
other, and a caller must not have to discover which one a codec gives.

## 2. What already works

Stated so the requirements below are read against it, and so none of it
is rebuilt:

- **Positional field access with no allocation.**
  `veks_anode::mnode::scan` — `discover_schema`, `skip_value`,
  `check_condition_raw`, `CompiledScanPredicates` — walks raw MNode
  bytes, resolves names to positions once, and evaluates per record by
  index. Built for predicate scanning; it is the binding hot path
  unchanged.
- **Stable positions.** `MNode.fields` is an `IndexMap` preserving
  insertion order to match the wire format, so position survives encode
  and decode.
- **Structural identity.** `ANode::fingerprint` and `is_congruent`
  answer "is this the same shape, differing only in values".
- **A record says what it is.** The dialect leader byte distinguishes
  MNode from PNode without reference to the facet it came from.
- **An encoding identifier.** `PredicateSchema.wire_format` names the
  record encoding and is already documented as reserved for a future
  one.
- **Named-by-name resolution.** `Vernacular::parse` and
  `records::codec_by_name` establish how a name from a setting reaches
  the same code as a name in a type signature.
- **An exhaustive type mapping.** `cql_type` covers all 29 `TypeTag`
  variants with no wildcard arm, so a new tag breaks the build rather
  than falling through. The tag assignments are themselves *"stable and
  match the Java `datatools-vectordata` implementation"* — a
  cross-language contract, not a local enum.
- **Predicate evaluation.** `eval::evaluate(&PNode, &MNode)` answers
  whether a record satisfies a predicate.
- **Structural identity at every level.** `fingerprint` and
  `is_congruent` exist on `ANode`, `MNode`, `MValue` and `PNode`.
- **Stream framing.** `MNode::encode` writes a length prefix and
  `from_buffer` reads from any `Read`, for callers that stream rather
  than address.

## 3. Two asymmetries the design must respect

Found by reviewing the wire types rather than assumed, and both would
have produced a wrong binding layer.

**OT-A. The existing type mapping is for rendering, not binding.**
`cql_type` is exhaustive and correct as DDL, and wrong as a bind-type
source in three places: `Half → smallint` types a 16-bit float as an
integer, `Null → text` gives an absent value a type it does not have,
and `List|Array → list<text>`, `Set → set<text>`,
`Map|TypedMap → map<text, text>` collapse element types. A binder
reusing it would mistype `Half` silently and bind container elements as
strings. Binding needs its own mapping from `TypeTag`, equally
exhaustive, and the two must not be conflated — the same rendering/
binding split as OT-1, one layer down.

**OT-B. A comparand is not a field value.** `MValue` has 29 variants;
`Comparand` has six — Int, Float, Text, Bool, Bytes, Null. So a
predicate over a `DateTime` field carries an `Int` comparand, and the
driver type for that parameter must come from the **field's** tag, not
the comparand's variant. A binder that types parameters from the
comparand collapses every temporal and UUID column to a bigint or a
string.

Relatedly, **indexed mode is not a general positional encoding.**
`FieldRef::Index` exists, but `Comparand::as_i64` is documented as
*"used by indexed mode which only supports integer comparands"* — which
is why facets carry `pnode:named`. Indexed mode is not the positional
binding form it looks like from the outside, and OT-7's positional
access comes from the discovered schema, not from it.


## 4. The variance problem

Today a metadata facet has one record encoding and one plausible
operation shape. Neither will stay true.

**OT-2.** A source may produce **variant record types**, and a facet may
offer **more than one op-template form**. A row insert, a document put
and an edge upsert are three forms of the same records; a future
encoding is a different form of the same facet. A design that assumes
one form per facet has to be reopened for the first counterexample.

**OT-3.** Variance is **declared, not inferred**. Forms are enumerated
in the facet itself — a namespace of the metadata or predicate file —
so a consumer asks what a facet offers rather than deducing it from a
name, an extension, or a version number.

This is the extension point the format already uses. `schema`, `layout`
and `survey` are namespaces; a fourth costs nothing structurally, and
puts the enumeration where the records are rather than in a sidecar
that can drift from them.

**OT-4.** The precedent to extend is `wire_format`, not to replace it.
`PredicateSchema` already carries *"a stable identifier — currently
always `pnode:named` — reserved for a future indexed-mode encoding"*.
That field answers "how is this record encoded". A form answers "what
operation can this record become". They are different questions and
both are needed; a form names the wire format it consumes.

## 5. Absence is not a variant

**OT-5.** A facet with no forms namespace offers **exactly one** form:
the implicit one its records already have. Absence is not an empty set
and not an error — it is every dataset in existence, and they must read
unchanged.

The same rule the format version takes (V-2: absent means 1, which is
not a distinct unversioned state) and profile attributes take (P-7:
absent means undescribed, not zero). Stated here because the failure
mode is specific: a reader that treats "no forms declared" as "no forms
available" makes every current dataset unbindable.

**OT-6.** An **unknown** form is preserved and reported, never rejected.
A writer that records a form this build does not implement is
recording, not misbehaving (P-5). A consumer asking for a form it does
not have gets a refusal naming what *is* offered — the
`WrongFacetShape` pattern, which names the door rather than describing
the symptom.

## 6. The binding contract

**This section was drafted against the wrong mechanism.** It proposed
`MNode.fields` as the binding surface and left allocation open. Both
were answered before this document existed, in
`veks_anode::mnode::scan` — re-exported as
`vectordata::formats::mnode::scan` — which was built for zero-allocation
predicate evaluation and is the same shape binding needs.

**OT-7.** Binding is **positional against a discovered schema**, not
name-keyed per record. The wire layout is
`[dialect][u16 field_count]` then per field
`[u16 name_len][name][tag][value]`, and `MNode.fields` is an `IndexMap`
*"to preserve insertion order, matching the wire format's"* — so a
field's position survives the round trip and is stable across every
record a source produces.

`discover_schema(bytes) -> RecordSchema` captures the ordered field
names once, skipping values without allocating. Per record the name is
**skipped entirely** — `pos += 2 + name_len` — and the value reached as
`(value_pos, tag)`, which `check_condition_raw` already consumes
without materializing an `MValue`. That is the hot path, and it exists.

**OT-8.** Field names are the **metadata names**, preserved as the
schema presents them. They are not renamed, normalized or generated:
the record carries them, the schema restates them in the same order,
and a binder resolves each to a position once.

**OT-9.** An op-template runtime **may override a name for
substitution** — binding a field to a differently-named parameter is
the runtime's business. The override is a mapping applied at compile
time, against the discovered positions, so it costs nothing per record
and cannot drift into the data. Absent an override, the metadata name
is the parameter name.

**OT-10.** A form is **validated against the discovered schema**, not
assumed to fit. `ANode::fingerprint` replaces values with type-default
placeholders while preserving field names and structure, and
`is_congruent` compares two records for the same structure differing
only in values — so "does this record match the form I prepared for" is
already answerable, and a mismatched record is caught rather than bound
into the wrong positions.

**OT-11.** A schema is obtained **once**, before the first record. From
the `schema` namespace where a facet publishes one, or discovered from
the first record where it does not. Both yield the same ordered names;
the namespace is authoritative when present, because a facet whose
first record happens to omit an optional field must not define the
layout for the rest.

**OT-12.** Binding allocates **nothing per field name per record**.
Names are identical across a facet and resolved at compile time. This
is a requirement rather than an aspiration because the mechanism is
already built and proven — a binder that materializes an `MNode` per
cycle has bypassed it, not failed to have it.

**OT-13.** The dialect leader byte remains the authority on what a
record *is*. A form says what a record can become; it does not override
`0x01`/`0x02`. A facet holding a mix still decodes correctly, and a form
that disagrees with a record's dialect is an error against the form.

**OT-14.** Predicate facets bind too. A filtered workload needs a WHERE
clause with **parameters**, not a rendered fragment with comparands
inlined — the same distinction as OT-1, against `PNode`. `flatten_and`
already reduces a predicate to `(field, op, comparands)` triples, which
is the parameter list in all but name.

## 7. Selection

**OT-15.** A form is selected **by name**, resolved through one
implementation. `Vernacular::parse` and `records::codec_by_name` already
establish this: a name from a setting and a name in a type signature
reach the same code. A form registry is the same shape, and must not
become a second dispatch that can disagree with the first.

**OT-16.** Cycle-to-ordinal mapping belongs to the caller. `count()` is
exact and cheap (from the tail index, no scan), so wrapping, hashing or
striding is the generator's policy. The contract states this rather than
leaving an integrator to find out that `get(count())` errors.

## 8. Declaration shape

A forms namespace, one record per form:

```yaml
# metadata_content.slab :forms   (one record per form)
- name: row
  kind: metadata
  wire_format: mnode:v1
  operation: insert
  fields: [id, bucket, tag]        # subset/order this form binds
- name: document
  kind: metadata
  wire_format: mnode:v1
  operation: put
  fields: [id, payload]
```

Nothing here is required for a facet to be readable. A facet without
this namespace binds through its schema, as one unnamed form.

## 9. Non-goals

- **Rendering going away.** The text vernaculars stay; they are what
  `inspect`, `describe` and a human want.
- **vectordata knowing any driver.** It yields names and `MValue`s; the
  mapping to a CQL, SQL or document type is the adapter's, exactly as
  the vernacular renderers keep their dialect knowledge local.
- **A form implying a statement.** A form names what an operation binds;
  composing the statement is the generator's.
- **Retrofitting forms onto existing data.** OT-5 makes that
  unnecessary.

## 10. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | facet with no forms namespace | one implicit form; binds unchanged |
| 2 | facet with a forms namespace | forms enumerable by name |
| 3 | unknown form requested | refused, naming the forms offered |
| 4 | unrecognised form declared | preserved and reported, not rejected |
| 5 | bound record | names and typed values, in declared order |
| 6 | schema | obtained once, before any record is bound |
| 7 | predicate facet | binds parameters, not an inlined fragment |
| 8 | mixed-dialect facet | decodes per record; a form cannot override the leader byte |
| 9 | binding N records | no per-record allocation of field names |
| 10 | every dataset predating this | reads and binds unchanged |
| 11 | sharded facet | forms and schema read from the series, not shard 0 |
| 12 | remote facet | binding stays incremental — a record costs its page |
| 13 | a `Half` field | binds as a float, not as a smallint (OT-A) |
| 14 | a container field | element types preserved, not collapsed to text (OT-A) |
| 15 | predicate over a temporal field | parameter typed from the field's tag, not the comparand's (OT-B) |
| 16 | a new `TypeTag` | binding mapping fails to compile until it is handled |

**OT-17.** Case 10 is the gate, and case 12 is the one that regresses
quietly: a forms or schema lookup that reads more than the tail would
reintroduce the whole-facet fetch that Stage 4 removed.

## 11. Open

**~~The allocation strategy~~ — settled.** Not open, and should not have
been listed as open: `mnode::scan` answers it. Names resolve to
positions once, the per-record path skips names and reads values raw.
No change to `MNode` is needed, and a borrowing record would have been
a second mechanism competing with a working one.

**~~Whether a form names its fields~~ — mostly settled.** It does not
need to. The names are the metadata names, carried by the record in
wire order and restated by the schema, so a form that listed them would
be a second copy that can drift (P-3's ruling, again). What a form
*may* carry is a **substitution override** — a parameter name differing
from the field name — which is a mapping, not a restatement. Still open
only in the narrow sense of whether a form may also bind a *subset*, or
whether a subset is the template's business.

**Namespace name — settled.** `forms`. slabtastic reserves namespace
*index* 0 as invalid and reserves no names, and the workspace uses only
`schema`, `layout` and `survey` today, so there is nothing to collide
with. Kept in this section rather than deleted because it was a real
question and the answer is a fact someone will want to re-check.
