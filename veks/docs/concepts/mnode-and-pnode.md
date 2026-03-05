<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MNode and PNode

MNode and PNode are the two binary record types used in predicated datasets.
They serve complementary roles: MNode describes *what* a record contains, and
PNode describes *how* to filter records.

## MNode — Metadata Node

An MNode is a self-describing key-value record. Each field has a name (UTF-8
string), a type tag (one of 29 types), and a value encoded according to its
type. MNode is used for the `metadata_content` facet in predicated datasets.

Key properties:

- **Self-describing**: Every field carries its own type tag, so the record can
  be decoded without external schema knowledge.
- **Order-preserving**: Fields maintain their insertion order (backed by
  `IndexMap`).
- **Rich type system**: Supports scalars (text, int, float, bool, null),
  collections (list, map, set, array, typed_map), temporal types (millis,
  nanos, date, time, datetime), identifiers (uuid_v1, uuid_v7, ulid), and
  numeric variants (int32, short, float32, half, ascii, enum).
- **Nestable**: The `Map` variant contains a nested MNode, enabling
  hierarchical records.

## PNode — Predicate Node

A PNode is a boolean predicate tree used for the `metadata_predicates` facet.
It expresses filter conditions over metadata fields.

Key properties:

- **Tree structure**: Composed of leaf nodes (`PredicateNode`) and interior
  nodes (`ConjugateNode` with AND/OR).
- **Two addressing modes**: Fields can be referenced by index (`u8`) or by
  name (`String`), supporting both compact wire encoding and human-readable
  formats.
- **Eight operators**: GT, LT, EQ, NE, GE, LE, IN, MATCHES.
- **Multi-value comparands**: Each predicate can compare against multiple `i64`
  values (used by the IN operator).
- **Pre-order encoding**: The tree is serialized parent-first, enabling
  single-pass streaming decode.

## How they relate

In a predicated dataset, metadata records (MNode) describe properties of each
vector, and predicate trees (PNode) express filter conditions that select
subsets of vectors based on their metadata. The dataset facets look like:

| Facet                 | Format | Contains         |
|-----------------------|--------|------------------|
| `metadata_content`    | slab   | MNode records    |
| `metadata_predicates` | slab   | PNode records    |

Both record types are stored as opaque byte slices in slab files. The dialect
leader byte (`0x01` for MNode, `0x02` for PNode) allows tools to identify and
decode records without prior knowledge of the slab's contents.

## ANode — the unified view

The ANode enum wraps both types behind a single interface, enabling generic
operations (decode, render, inspect) that work on any record type. See
[Two-Stage Codec](../explanation/two-stage-codec.md) for the architecture.
