<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 02 — Data Model

## 2.1 Vector Formats (xvec family)

Fixed-dimension vector files where each record is:

```
[dimension: i32 LE][element_0..element_dim-1: T]
```

| Extension | Element type | Element size | Use case |
|-----------|-------------|--------------|----------|
| `.fvec` | f32 | 4 bytes | Standard float vectors |
| `.ivec` | i32 | 4 bytes | Integer vectors, neighbor indices |
| `.bvec` | u8 | 1 byte (padded to 4) | Byte vectors |
| `.dvec` | f64 | 8 bytes | Double-precision vectors |
| `.hvec` | f16 | 2 bytes | Half-precision vectors |
| `.svec` | i16 | 2 bytes | Short integer vectors |

Access pattern: memory-mapped via `MmapVectorReader<T>`. O(1) random access
by ordinal. Record size = 4 + dimension * element_size.

## 2.2 NumPy Format (.npy)

Standard NumPy array format. Used as source data for import. Parsed by the
npy reader in `veks/src/formats/reader/npy.rs`. Supports dtype detection and
streaming reads.

## 2.3 Apache Parquet (.parquet)

Columnar format used for metadata import. Source data is read via the arrow
and parquet crates. The parquet reader (`reader/parquet.rs`) supports
streaming row-group reads and an optimized `parquet_mnode` path that compiles
a schema-aware MNode writer for zero-copy field extraction.

## 2.4 Slab Format (.slab)

Page-aligned record container provided by the `slabtastic` crate.

### Physical layout

```
[Page 0][Page 1]...[Page N-1][PagesPage]
```

- **Page**: Fixed-size (configurable, typically 64 KiB) containing packed
  variable-length records with a directory of (offset, length) entries.
- **PagesPage**: Final page in the file. Contains an array of `PageEntry`
  records mapping `(start_ordinal, file_offset)` for each data page.

### Record addressing

Records are addressed by sequential ordinal (0-based). The reader:
1. Reads the PagesPage to build a page index
2. Binary-searches the index to find which page contains ordinal N
3. Reads that page and extracts the record at the correct directory slot

### Writer configuration

```rust
WriterConfig::new(
    page_alignment,     // typically 512 or 4096
    page_data_capacity, // typically 65536 (64 KiB)
    max_record_size,    // typically u32::MAX
    allow_spanning,     // whether records can span multiple pages
)
```

### Namespace support

A slab file can contain multiple namespaces, each with its own page index.
Used for multi-facet storage within a single file.

### Open progress callback

SlabReader reports opening progress via `OpenProgress`:
- page index loading phase
- total page count and record count

## 2.5 MNode — Metadata Node

Self-describing key-value record for vector metadata. Wire format:

```
[0x01 dialect leader][field_count: u16 LE]
per field:
  [name_len: u16 LE][name: UTF-8][type_tag: u8][value: variable]
```

### Type system (29 types)

| Tag | Type | Value encoding |
|-----|------|----------------|
| 0 | text | u32 len + UTF-8 |
| 1 | int | i64 LE |
| 2 | float | f64 LE |
| 3 | bool | u8 |
| 4 | bytes | u32 len + raw |
| 5 | null | (empty) |
| 6 | enum_str | u32 len + UTF-8 |
| 7 | enum_ord | i32 LE |
| 8 | list | u32 count + tagged values |
| 9 | map | u32 len + nested MNode (no leader) |
| 10 | text_validated | u32 len + UTF-8 |
| 11 | ascii | u32 len + ASCII |
| 12 | int32 | i32 LE |
| 13 | short | i16 LE |
| 14 | decimal | i32 scale + u32 len + big-endian bytes |
| 15 | varint | u32 len + big-endian two's complement |
| 16 | float32 | f32 LE |
| 17 | half | u16 LE |
| 18 | millis | i64 LE |
| 19 | nanos | i64 seconds + i32 nano_adjust |
| 20 | date | u32 len + ISO-8601 string |
| 21 | time | u32 len + ISO-8601 string |
| 22 | datetime | u32 len + ISO-8601 string |
| 23 | uuid_v1 | 16 bytes |
| 24 | uuid_v7 | 16 bytes |
| 25 | ulid | 16 bytes |
| 26 | array | u8 elem_tag + u32 count + untagged values |
| 27 | set | u32 count + tagged values |
| 28 | typed_map | u32 count + (tagged key + tagged value) pairs |

### API

```rust
impl MNode {
    fn new() -> Self
    fn insert(&mut self, name: String, value: MValue)
    fn to_bytes(&self) -> Vec<u8>         // with 0x01 leader
    fn encode(&self) -> Vec<u8>           // framed: u32 LE len + payload
    fn from_bytes(data: &[u8]) -> Result  // verifies 0x01 leader
    fn from_buffer(reader: &mut Read) -> Result  // framed
}
```

## 2.6 PNode — Predicate Node

Boolean predicate tree for metadata filtering. Wire format:

```
[0x02 dialect leader][tree encoding...]
```

### Node types

- **PredicateNode** (leaf): field reference + operator + comparand values
- **ConjugateNode** (interior): AND/OR + child count + children

### Comparand types

PNode supports typed comparands (signaled by a `0xFF` marker byte after
the dialect leader):

| Tag | Type | Encoding |
|-----|------|----------|
| 0 | Int | i64 LE |
| 1 | Float | f64 LE |
| 2 | Text | u16 LE len + UTF-8 |
| 3 | Bool | u8 |
| 4 | Bytes | u32 LE len + raw |
| 5 | Null | (empty) |

Legacy mode (no `0xFF` marker) uses i64-only comparands.

### Encoding modes

- **Named mode**: fields referenced by UTF-8 name (self-describing).
  Supports both legacy (i64 comparands) and typed (mixed comparands).
- **Indexed mode**: fields referenced by u8 position (compact)

### Operators

GT (`>`), LT (`<`), EQ (`=`), NE (`!=`), GE (`>=`), LE (`<=`), IN, MATCHES

### API

```rust
impl PNode {
    fn to_bytes_indexed(&self) -> Vec<u8>
    fn to_bytes_named(&self) -> Vec<u8>
    fn from_bytes_indexed(data: &[u8]) -> Result
    fn from_bytes_named(data: &[u8]) -> Result
}
```

## 2.7 ANode — Unified Codec

Wraps MNode and PNode behind a single interface:

```rust
enum ANode {
    MNode(MNode),
    PNode(PNode),
}
```

### Two-stage codec architecture

```
bytes ←→ [Stage 1: binary codec] ←→ ANode ←→ [Stage 2: vernacular] ←→ text
```

- **Stage 1**: Dialect detection (leader byte), binary encode/decode
- **Stage 2**: Human-readable rendering/parsing in 13 vernacular formats

### Vernacular formats

| Format | Direction | MNode rendering | PNode rendering |
|--------|-----------|-----------------|-----------------|
| cddl | render + parse | Type group | Structure |
| cddl-value | render only | With literal values | Same as cddl |
| sql | render + parse | VALUES tuple | WHERE clause |
| sql-schema | render only | Column definitions | WHERE clause |
| sqlite | render + parse | Delegates to sql | Delegates to sql |
| sqlite-schema | render only | Delegates to sql-schema | Delegates to sql |
| cql | render + parse | VALUES tuple | WHERE clause |
| cql-schema | render only | Column definitions | WHERE clause |
| json | render + parse | Pretty JSON object | JSON predicate tree |
| jsonl | render + parse | Compact JSON | Compact JSON |
| yaml | render + parse | YAML key-value | YAML tree |
| readout | render + parse | Tab-indented display | Indented infix |
| display | render only | Rust Display | Rust Display |

## 2.8 Dataset Facets

A complete predicated dataset contains:

| Facet | Format | Description |
|-------|--------|-------------|
| `base_vectors` | fvec/hvec | Corpus vectors to search |
| `query_vectors` | fvec/hvec | Query vectors to run |
| `neighbor_indices` | ivec | Ground-truth neighbor indices |
| `neighbor_distances` | fvec | Ground-truth neighbor distances |
| `metadata_content` | slab (MNode) | Metadata records per vector |
| `metadata_predicates` | slab (PNode) | Predicate trees per query |
| `metadata_layout` | slab | Metadata field layout schema |
| `predicate_results` | slab | Predicate evaluation result bitmaps |
| `filtered_neighbor_indices` | ivec | Filtered ground-truth indices |
| `filtered_neighbor_distances` | fvec | Filtered ground-truth distances |

## 2.9 Distance Metrics

| Metric | Implementation | Notes |
|--------|---------------|-------|
| L2 (Euclidean) | simsimd | AVX-512, AVX2, NEON, SVE dispatched |
| Cosine | simsimd | Hardware-dispatched |
| DotProduct | simsimd | Hardware-dispatched |
| L1 (Manhattan) | Hand-rolled | AVX2/AVX-512, simsimd lacks L1 |

All metrics support both f32 and f16 element types via `select_distance_fn`
and `select_distance_fn_f16`.
