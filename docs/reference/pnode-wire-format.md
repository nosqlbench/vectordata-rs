<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PNode Wire Format Reference

Module: `vectordata::formats::pnode`

Source: `vectordata/src/formats/pnode/mod.rs`

## Wire layout

All PNode payloads begin with the dialect leader byte `0x02`, followed by a
recursive pre-order tree encoding.

### Dialect leader byte

`DIALECT_PNODE = 0x02`

### ConjugateType discriminant

The first byte after the leader byte identifies the node type:

| Byte | Type |
|------|------|
| 0    | PRED (PredicateNode) |
| 1    | AND (ConjugateNode)  |
| 2    | OR (ConjugateNode)   |

### ConjugateNode (AND/OR)

```
[conjugate_type: u8 (1 or 2)][child_count: u8][children...]
```

Children are encoded recursively in sequence.

### PredicateNode — indexed mode

```
[PRED=0: u8][field_index: u8][op: u8][comparand_count: i16 LE][comparands: i64 LE * n]
```

### PredicateNode — named mode

```
[PRED=0: u8][name_len: u16 LE][name: UTF-8][op: u8][comparand_count: i16 LE][comparands: i64 LE * n]
```

### Full payload (named mode example)

```
[0x02][node encoding...]
```

### Full payload (indexed mode example)

```
[0x02][node encoding...]
```

## Operator types

| Ordinal | Name    | Symbol    |
|---------|---------|-----------|
| 0       | GT      | `>`       |
| 1       | LT      | `<`       |
| 2       | EQ      | `=`       |
| 3       | NE      | `!=`      |
| 4       | GE      | `>=`      |
| 5       | LE      | `<=`      |
| 6       | IN      | `IN`      |
| 7       | MATCHES | `MATCHES` |

## Encoding modes

PNode supports two encoding modes, determined at serialization time:

- **Indexed mode**: Fields are referenced by `u8` index. Compact but requires a
  schema mapping to interpret. Produced by `to_bytes_indexed()` / decoded by
  `from_bytes_indexed()`.

- **Named mode**: Fields are referenced by UTF-8 name. Self-describing but
  larger. Produced by `to_bytes_named()` / decoded by `from_bytes_named()`.

Both modes use the same `0x02` dialect leader byte. The ANode codec uses named
mode by default.

## Public API

```rust
pub const DIALECT_PNODE: u8 = 0x02;

impl PNode {
    pub fn to_bytes_indexed(&self) -> Vec<u8>
    pub fn to_bytes_named(&self) -> Vec<u8>
    pub fn from_bytes_indexed(data: &[u8]) -> io::Result<Self>
    pub fn from_bytes_named(data: &[u8]) -> io::Result<Self>
}
```

## Wire size examples

Indexed predicate with 1 comparand (named mode would be larger):

```
leader(1) + PRED(1) + field_index(1) + op(1) + count(2) + comparand(8) = 14 bytes
```
