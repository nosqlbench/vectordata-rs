<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Working with ANode Codecs in Rust

Encode, decode, and render metadata (MNode) and predicate (PNode)
records using the `veks-anode` crate.

## Prerequisites

```toml
[dependencies]
veks-anode = "0.17"
```

## Create an MNode record

```rust
use veks_anode::mnode::{MNode, MValue};

let mut node = MNode::new();
node.insert("name".into(), MValue::Text("alice".into()));
node.insert("age".into(), MValue::Int(30));
node.insert("score".into(), MValue::Float(99.5));
node.insert("active".into(), MValue::Bool(true));
```

## Encode to bytes

```rust
let bytes = node.to_bytes();
assert_eq!(bytes[0], 0x01); // DIALECT_MNODE leader byte
```

## Decode via ANode (auto-detect type)

```rust
use veks_anode::anode;

let decoded = anode::decode(&bytes).unwrap();
match &decoded {
    anode::ANode::MNode(m) => println!("MNode: {} fields", m.fields.len()),
    anode::ANode::PNode(p) => println!("PNode: {}", p),
}
```

## Render to text

```rust
use veks_anode::anode_vernacular::{self, Vernacular};

let json = anode_vernacular::render(&decoded, Vernacular::Json);
let sql = anode_vernacular::render(&decoded, Vernacular::Sql);
let cddl = anode_vernacular::render(&decoded, Vernacular::Cddl);
```

## Parse text back to ANode

```rust
let json_text = r#"{"name": "bob", "age": 25, "active": true}"#;
let parsed = anode_vernacular::parse(json_text, Vernacular::Json).unwrap();
let re_encoded = anode::encode(&parsed);
```

## PNode predicates

```rust
use veks_anode::pnode::*;

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
            comparands: vec![Comparand::Int(1), Comparand::Int(2), Comparand::Int(3)],
        }),
    ],
});

let bytes = predicate.to_bytes_named();
let decoded = anode::decode(&bytes).unwrap();
let sql = anode_vernacular::render(&decoded, Vernacular::Sql);
// Output: (age > 18 AND status IN (1, 2, 3))
```

## The codec pipeline

```
bytes  <-->  [anode::decode/encode]  <-->  ANode  <-->  [vernacular render/parse]  <-->  text
```

Stage 1 handles dialect detection and binary serialization.
Stage 2 provides the human-readable interface.
