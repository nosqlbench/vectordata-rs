<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The Two-Stage Codec Architecture

The ANode codec uses a two-stage pipeline to convert between raw binary records
and human-readable text:

```
bytes ←→ [Stage 1: ANode binary codec] ←→ ANode ←→ [Stage 2: Vernacular codec] ←→ text
```

## Why two stages?

Combining binary codec and text rendering into a single layer would create a
combinatorial explosion: each binary format (MNode, PNode, and any future
formats) would need its own JSON renderer, SQL renderer, CDDL renderer, and so
on. With two stages, the concerns are cleanly separated:

- **Stage 1** knows about wire formats, byte layouts, and dialect detection.
- **Stage 2** knows about human-readable syntax, quoting rules, and type
  mappings.

Adding a new binary format requires only a Stage 1 codec and a few lines in
Stage 2 to dispatch rendering. Adding a new text format requires only Stage 2
changes, with no binary format knowledge needed.

## Stage 1: ANode binary codec

Stage 1 answers the question: *"What kind of record is this, and what does it
contain?"*

Every binary record begins with a **dialect leader byte**:

| Byte   | Record type |
|--------|-------------|
| `0x00` | Invalid     |
| `0x01` | MNode       |
| `0x02` | PNode       |

The `anode::decode()` function reads this byte and dispatches to the
appropriate decoder (`MNode::from_bytes` or `PNode::from_bytes_named`). The
`anode::encode()` function delegates to the corresponding `to_bytes` method,
which prepends the leader byte automatically.

The ANode enum itself is intentionally simple — it adds no new data
representation, just wraps the existing MNode and PNode types:

```rust
pub enum ANode {
    MNode(MNode),
    PNode(PNode),
}
```

## Stage 2: Vernacular codec

Stage 2 answers the question: *"How should this record look in format X?"*

The `render()` function takes an ANode and a Vernacular variant and produces a
string. It dispatches to format-specific renderers, many of which delegate to
the pre-existing vernacular adapters in `mnode::vernacular` and
`pnode::vernacular`.

The `parse()` function handles the reverse direction for formats where it makes
sense: JSON, SQL VALUES, CQL VALUES, CDDL groups, YAML, and readout. Not all
formats are parseable — schema-only formats like `sql-schema` and output-only
formats like `display` return errors.

## Design decisions

### Why not serde?

The JSON rendering uses a custom `JsonValue` enum rather than `serde_json`
serialization. This avoids adding serde derives to MNode/PNode (which would
couple the wire format types to a serialization framework) and gives full
control over the output shape — for example, PNode predicates are rendered as
`{"type": "predicate", "field": "age", "op": ">", "value": 18}` rather than
mirroring the Rust enum structure.

### Why delegate to existing vernacular modules?

The SQL, CQL, and CDDL renderers in `mnode::vernacular` and
`pnode::vernacular` predate ANode. Rather than duplicating that logic, Stage 2
delegates to them. The new formats (JSON, YAML, readout) are implemented
directly in `anode_vernacular.rs` because they need to handle the full ANode
dispatch.

### Why is the leader byte part of to_bytes/from_bytes?

Putting the leader byte inside MNode's `to_bytes()` and PNode's
`to_bytes_named()` means every encoded payload carries its dialect tag. This
makes individual records self-identifying without requiring external metadata,
which is important for slab files where records are opaque byte slices.
