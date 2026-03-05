<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dialect Leader Byte

## The problem

MNode and PNode records are stored as opaque byte slices in slab files. Without
external metadata, there is no way to tell whether a given byte sequence is a
metadata record (MNode) or a predicate tree (PNode). This makes generic tooling
— slab inspectors, debuggers, validators — impossible without prior knowledge
of the slab's contents.

## The solution

Every MNode and PNode payload now begins with a single **dialect leader byte**
that identifies the record type:

| Byte   | Dialect   | Meaning                  |
|--------|-----------|--------------------------|
| `0x00` | Invalid   | Reserved, never valid     |
| `0x01` | MNode     | Metadata record follows   |
| `0x02` | PNode     | Predicate tree follows    |

This byte is prepended by `to_bytes()` / `to_bytes_named()` /
`to_bytes_indexed()` and verified+stripped by the corresponding `from_bytes()`
methods.

## Wire format impact

### MNode

Before:

```
[field_count: u16 LE][fields...]
```

After:

```
[0x01][field_count: u16 LE][fields...]
```

Every MNode payload is 1 byte larger.

### PNode

Before:

```
[conjugate_type: u8][...]
```

After:

```
[0x02][conjugate_type: u8][...]
```

Every PNode payload is 1 byte larger.

## Why it works without ambiguity

The leader byte values `0x01` and `0x02` were chosen to avoid collisions with
the natural first bytes of each format:

- An MNode without a leader byte starts with a `u16 LE` field count. A field
  count of 1 (`0x01 0x00`) could theoretically collide, but the leader byte is
  always followed by the field count, so the two-byte sequence `0x01 0x01 0x00`
  (leader + count=256) vs `0x01 0x00` (count=1) are structurally distinct at
  the decoder level.

- A PNode without a leader byte starts with a `ConjugateType` discriminant
  (`0x00`=PRED, `0x01`=AND, `0x02`=OR). The leader byte `0x02` is consumed
  first, and the remaining bytes are decoded normally.

In practice, the decoder simply reads byte 0, dispatches by value, and passes
the remainder to the format-specific decoder.

## Backward compatibility

The leader byte is a breaking change to the wire format. All producers and
consumers of MNode/PNode binary data must be updated simultaneously. The
`from_bytes` methods now reject data that lacks the correct leader byte,
providing clear error messages:

```
expected MNode dialect leader 0x01, got 0x00
```

The `CompiledMnodeWriter` (used for Parquet → MNode conversion) was also
updated to prepend `0x01` in its `write_row()` method, maintaining roundtrip
compatibility with `MNode::from_bytes()`.
