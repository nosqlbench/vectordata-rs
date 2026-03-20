<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Building an ANode Codec Pipeline in Rust

This tutorial shows how to use the two-stage ANode codec in your own Rust code
to encode, decode, and render metadata and predicate records.

## Prerequisites

- The `vectordata` crate as a dependency (or working within the vectordata-rs source tree)
- Basic familiarity with MNode and PNode structures

## Step 1: Create an MNode record

```rust
use vectordata::formats::mnode::{MNode, MValue};

let mut node = MNode::new();
node.insert("name".into(), MValue::Text("alice".into()));
node.insert("age".into(), MValue::Int(30));
node.insert("score".into(), MValue::Float(99.5));
node.insert("active".into(), MValue::Bool(true));
```

## Step 2: Encode to bytes (Stage 1)

The `to_bytes()` method produces a byte vector with the `0x01` dialect leader
byte followed by the MNode payload:

```rust
let bytes = node.to_bytes();
assert_eq!(bytes[0], 0x01); // DIALECT_MNODE leader byte
```

## Step 3: Auto-detect and decode via ANode

Use the `anode` module to decode bytes without knowing the record type in
advance:

```rust
use vectordata::formats::anode;

let decoded = anode::decode(&bytes).unwrap();
match &decoded {
    anode::ANode::MNode(m) => println!("Got MNode with {} fields", m.fields.len()),
    anode::ANode::PNode(p) => println!("Got PNode: {}", p),
}
```

## Step 4: Render to human-readable text (Stage 2)

```rust
use vectordata::formats::anode_vernacular::{self, Vernacular};

// Render as JSON
let json = anode_vernacular::render(&decoded, Vernacular::Json);
println!("{}", json);

// Render as SQL VALUES
let sql = anode_vernacular::render(&decoded, Vernacular::Sql);
println!("{}", sql);

// Render as CDDL schema
let cddl = anode_vernacular::render(&decoded, Vernacular::Cddl);
println!("{}", cddl);
```

## Step 5: Parse text back to ANode (reverse direction)

```rust
let json_text = r#"{"name": "bob", "age": 25, "active": true}"#;
let parsed = anode_vernacular::parse(json_text, Vernacular::Json).unwrap();

// Re-encode to bytes
let re_encoded = anode::encode(&parsed);
```

## Step 6: Work with PNode records

```rust
use vectordata::formats::pnode::*;

let predicate = PNode::Conjugate(ConjugateNode {
    conjugate_type: ConjugateType::And,
    children: vec![
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named("age".into()),
            op: OpType::Gt,
            comparands: vec![18],
        }),
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named("status".into()),
            op: OpType::In,
            comparands: vec![1, 2, 3],
        }),
    ],
});

let bytes = predicate.to_bytes_named();
assert_eq!(bytes[0], 0x02); // DIALECT_PNODE leader byte

let decoded = anode::decode(&bytes).unwrap();
let sql = anode_vernacular::render(&decoded, Vernacular::Sql);
println!("{}", sql);
// Output: (age > 18 AND status IN (1, 2, 3))
```

## The full pipeline

```
bytes ←→ [Stage 1: anode::decode/encode] ←→ ANode ←→ [Stage 2: vernacular render/parse] ←→ text
```

Both stages are independently useful. Stage 1 handles dialect detection and
binary serialization. Stage 2 provides the human-readable interface.
