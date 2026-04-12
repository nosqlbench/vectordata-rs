# 10. ANode Codec System

Binary codecs and human-readable renderers for structured metadata
(MNode) and predicate trees (PNode). Provided by the `veks-anode` crate.

---

## 10.1 Two-Stage Architecture

```
bytes  ←→  [Stage 1: binary codec]  ←→  ANode  ←→  [Stage 2: vernacular]  ←→  text
```

**Stage 1** handles dialect detection and binary serialization.
The dialect leader byte (`0x01` = MNode, `0x02` = PNode) makes
records self-identifying in slab files.

**Stage 2** handles human-readable rendering and parsing.
Each vernacular format knows how to render both MNode and PNode
records in its own syntax.

This separation avoids combinatorial explosion: adding a new binary
format requires only Stage 1 changes, adding a new text format
requires only Stage 2 changes.

### ANode enum

```rust
pub enum ANode {
    MNode(MNode),
    PNode(PNode),
}
```

### Core API

```rust
use veks_anode::anode;
use veks_anode::anode_vernacular::{self, Vernacular};

// Stage 1: bytes ↔ ANode
let bytes = node.to_bytes();              // encode with leader byte
let decoded = anode::decode(&bytes)?;     // auto-detect type

// Stage 2: ANode ↔ text
let text = anode_vernacular::render(&decoded, Vernacular::Json);
let parsed = anode_vernacular::parse(text, Vernacular::Json)?;
```

---

## 10.2 Wire Formats

Wire format details are in [Data Model §1.4](./01-data-model.md#14-wire-formats-mnode--pnode).

Summary:
- **MNode**: `[0x01][field_count: u16][fields...]` — 28 type tags
- **PNode**: `[0x02][tree...]` — recursive pre-order, indexed or named mode
- **Framed**: `[payload_len: u32][payload...]` for stream embedding

---

## 10.3 Vernacular Formats

| Format | MNode rendering | PNode rendering | Parseable |
|--------|----------------|-----------------|-----------|
| `json` | JSON object | JSON tree | yes |
| `jsonl` | compact single-line JSON | compact JSON | yes |
| `yaml` | YAML mapping | YAML tree | yes |
| `sql` | `VALUES (...)` tuple | `WHERE ...` clause | yes |
| `sqlite` | SQLite VALUES | SQLite WHERE | yes |
| `cql` | CQL VALUES | CQL WHERE | yes |
| `cddl` | CDDL group `{ field: type }` | CDDL predicate | yes |
| `readout` | colon-aligned key: value | indented tree | yes |
| `sql-schema` | `CREATE TABLE` DDL | — | no |
| `sqlite-schema` | SQLite DDL | — | no |
| `cql-schema` | CQL DDL | — | no |
| `cddl-value` | CDDL with values | — | no |
| `display` | compact debug | compact debug | no |

### Type inference during parsing

| Format | String indicator | Integer | Float | Boolean | Null |
|--------|-----------------|---------|-------|---------|------|
| JSON/YAML | `"quoted"` | bare number | decimal point | `true`/`false` | `null` |
| SQL/CQL | `'single quoted'` | bare number | decimal point | — | `NULL` |
| CDDL | `"quoted"` | bare number | decimal point | `true`/`false` | `null` |

---

## 10.4 MNode Usage

```rust
use veks_anode::mnode::{MNode, MValue};
use veks_anode::anode::ANode;
use veks_anode::anode_vernacular::{self, Vernacular};

// Build a record
let mut node = MNode::new();
node.insert("name".into(), MValue::Text("alice".into()));
node.insert("age".into(), MValue::Int(30));
node.insert("score".into(), MValue::Float(99.5));

// Encode → decode → render
let bytes = node.to_bytes();
let decoded = ANode::MNode(node);
let json = anode_vernacular::render(&decoded, Vernacular::Json);
let sql = anode_vernacular::render(&decoded, Vernacular::Sql);

// Parse back
let parsed = anode_vernacular::parse(&json, Vernacular::Json).unwrap();
```

### MValue → JSON type mapping

| MValue | JSON |
|--------|------|
| Text, Ascii, Date, Time, DateTime | string |
| Int, Int32, Short, Millis | integer |
| Float, Float32 | number |
| Bool | boolean |
| Null | null |
| List, Set, Array | array |
| Map | object |
| Bytes, Ulid | string (hex) |
| UuidV1, UuidV7 | string (UUID) |

### Roundtrip verification

```rust
let original = MNode::new();
// ... populate fields ...
let bytes = original.to_bytes();
let decoded = anode::decode(&bytes).unwrap();
let json = anode_vernacular::render(&decoded, Vernacular::Json);
let parsed = anode_vernacular::parse(&json, Vernacular::Json).unwrap();
let re_encoded = anode::encode(&parsed);
assert_eq!(bytes, re_encoded);
```

---

## 10.5 PNode Usage

```rust
use veks_anode::pnode::*;
use veks_anode::anode::ANode;
use veks_anode::anode_vernacular::{self, Vernacular};

let predicate = PNode::Conjugate(ConjugateNode {
    conjugate_type: ConjugateType::And,
    children: vec![
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named("age".into()),
            op: OpType::Gt,
            comparands: vec![Comparand::Int(18)],
        }),
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named("status".into()),
            op: OpType::In,
            comparands: vec![Comparand::Int(1), Comparand::Int(2)],
        }),
    ],
});

// Render as SQL WHERE clause
let sql = anode_vernacular::render(&ANode::PNode(predicate.clone()), Vernacular::Sql);
// → (age > 18 AND status IN (1, 2))

// Render as CQL
let cql = anode_vernacular::render(&ANode::PNode(predicate.clone()), Vernacular::Cql);

// Encode to binary (named mode, self-describing)
let bytes = predicate.to_bytes_named();
assert_eq!(bytes[0], 0x02); // DIALECT_PNODE

// Encode to binary (indexed mode, compact)
let bytes = predicate.to_bytes_indexed();
```

---

## 10.6 CLI Access

```bash
# Inspect slab records in any vernacular
veks pipeline slab inspect --source metadata.slab --ordinals "0,1,2" --format json
veks pipeline slab inspect --source predicates.slab --ordinals "0" --format sql

# Explain predicates with metadata cross-reference
veks analyze explain-predicates --ordinal 42 --vernacular sql
```

---

## 10.7 Extending: Adding a New Vernacular

1. Add variant to `Vernacular` enum in `anode_vernacular.rs`
2. Add string mapping in `Vernacular::from_str`
3. Add MNode rendering in `render_mnode()` match arm
4. Add PNode rendering in `render_pnode()` match arm
5. Add parsing in `parse()` if the format is bidirectional
6. Add tests — the format is automatically available in `slab inspect`
