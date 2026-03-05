<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ANode API Reference

Module: `veks::formats::anode`

Source: `src/formats/anode.rs`

## Constants

### `DIALECT_INVALID`

```rust
pub const DIALECT_INVALID: u8 = 0x00;
```

Reserved dialect leader byte. Never appears in valid records.

### `DIALECT_MNODE`

```rust
pub const DIALECT_MNODE: u8 = 0x01;
```

Dialect leader byte for MNode (metadata) records. Re-exported from
`mnode::DIALECT_MNODE`.

### `DIALECT_PNODE`

```rust
pub const DIALECT_PNODE: u8 = 0x02;
```

Dialect leader byte for PNode (predicate) records. Re-exported from
`pnode::DIALECT_PNODE`.

## Types

### `ANode`

```rust
pub enum ANode {
    MNode(mnode::MNode),
    PNode(pnode::PNode),
}
```

Unified wrapper for MNode and PNode records. Implements `Debug`, `Clone`,
`PartialEq`, and `Display`.

Display delegates to the inner type's `Display` implementation.

## Functions

### `decode`

```rust
pub fn decode(data: &[u8]) -> Result<ANode, String>
```

Decode raw bytes, auto-detecting the dialect from the leader byte.

- `0x01` → decodes as MNode
- `0x02` → decodes as PNode (named mode)
- Any other value → returns an error

**Errors**: Returns `Err` for empty data, unknown leader bytes, or malformed
payloads.

### `decode_mnode`

```rust
pub fn decode_mnode(data: &[u8]) -> Result<ANode, String>
```

Decode raw bytes, forcing MNode interpretation. The data must begin with the
`0x01` leader byte. Equivalent to calling `MNode::from_bytes(data)` and
wrapping the result.

### `decode_pnode`

```rust
pub fn decode_pnode(data: &[u8]) -> Result<ANode, String>
```

Decode raw bytes, forcing PNode interpretation (named mode). The data must
begin with the `0x02` leader byte.

### `encode`

```rust
pub fn encode(node: &ANode) -> Vec<u8>
```

Encode an ANode to bytes, including the dialect leader byte.

- `ANode::MNode` → delegates to `MNode::to_bytes()`
- `ANode::PNode` → delegates to `PNode::to_bytes_named()`
