<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MNode Wire Format Reference

Module: `vectordata::formats::mnode`

Source: `vectordata/src/formats/mnode/mod.rs`

## Wire layout

### Unframed payload

```
[dialect_leader: u8 = 0x01]
[field_count: u16 LE]
per field:
  [name_len: u16 LE][name_utf8: N bytes]
  [type_tag: u8]
  [value_bytes: variable per tag]
```

### Framed payload (for stream embedding)

```
[payload_len: u32 LE][payload...]
```

The framed length prefix covers the entire payload including the dialect leader
byte.

## Dialect leader byte

`DIALECT_MNODE = 0x01`

Always the first byte of any MNode payload produced by `to_bytes()` or
`CompiledMnodeWriter::write_row()`. Verified and stripped by `from_bytes()`.

## Type tags

| Tag | Name            | Value encoding                                          |
|-----|-----------------|---------------------------------------------------------|
| 0   | text            | u32 LE length + UTF-8 bytes                             |
| 1   | int             | i64 LE                                                  |
| 2   | float           | f64 LE                                                  |
| 3   | bool            | u8 (0 = false, 1 = true)                                |
| 4   | bytes           | u32 LE length + raw bytes                               |
| 5   | null            | (no value bytes)                                        |
| 6   | enum_str        | u32 LE length + UTF-8 bytes                             |
| 7   | enum_ord        | i32 LE                                                  |
| 8   | list            | u32 LE count + tagged values                            |
| 9   | map             | u32 LE payload length + MNode payload (no leader byte)  |
| 10  | text_validated  | u32 LE length + UTF-8 bytes                             |
| 11  | ascii           | u32 LE length + ASCII bytes                             |
| 12  | int32           | i32 LE                                                  |
| 13  | short           | i16 LE                                                  |
| 14  | decimal         | i32 LE scale + u32 LE length + big-endian bytes         |
| 15  | varint          | u32 LE length + big-endian two's complement bytes       |
| 16  | float32         | f32 LE                                                  |
| 17  | half            | u16 LE (IEEE 754 half-precision bits)                   |
| 18  | millis          | i64 LE (epoch milliseconds)                             |
| 19  | nanos           | i64 LE epoch_seconds + i32 LE nano_adjust               |
| 20  | date            | u32 LE length + ISO-8601 date string                    |
| 21  | time            | u32 LE length + ISO-8601 time string                    |
| 22  | datetime        | u32 LE length + ISO-8601 datetime string                |
| 23  | uuid_v1         | 16 bytes (raw UUID)                                     |
| 24  | uuid_v7         | 16 bytes (raw UUID)                                     |
| 25  | ulid            | 16 bytes (raw ULID)                                     |
| 26  | array           | u8 element_tag + u32 LE count + untagged values         |
| 27  | set             | u32 LE count + tagged values                            |
| 28  | typed_map       | u32 LE count + (tagged key + tagged value) pairs        |

### Collection encoding notes

- **list** and **set**: each element is a fully tagged value (tag byte +
  value bytes), supporting heterogeneous elements.
- **array**: elements share a single tag (homogeneous), so each element
  contains only value bytes (no per-element tag byte).
- **map**: the payload is a nested MNode payload (field_count + fields), but
  without a dialect leader byte.
- **typed_map**: each entry is a pair of fully tagged values.

## Public API

```rust
pub const DIALECT_MNODE: u8 = 0x01;

impl MNode {
    pub fn new() -> Self
    pub fn insert(&mut self, name: String, value: MValue)
    pub fn to_bytes(&self) -> Vec<u8>               // unframed, with leader byte
    pub fn encode(&self) -> Vec<u8>                  // framed (u32 LE prefix + payload)
    pub fn from_bytes(data: &[u8]) -> io::Result<Self>    // unframed, verifies leader byte
    pub fn from_buffer(reader: &mut impl Read) -> io::Result<Self>  // framed
}
```
