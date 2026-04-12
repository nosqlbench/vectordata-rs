<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example: MNode ↔ JSON Roundtrip

This example demonstrates the full roundtrip: build an MNode, encode it to
bytes, decode via ANode, render as JSON, parse the JSON back, and verify
equality.

```rust
use vectordata::formats::mnode::{MNode, MValue};
use vectordata::formats::anode::{self, ANode};
use vectordata::formats::anode_vernacular::{self, Vernacular};

// Build an MNode with various types
let mut node = MNode::new();
node.insert("name".into(), MValue::Text("alice".into()));
node.insert("age".into(), MValue::Int(30));
node.insert("score".into(), MValue::Float(99.5));
node.insert("active".into(), MValue::Bool(true));
node.insert("tags".into(), MValue::List(vec![
    MValue::Text("rust".into()),
    MValue::Text("vectors".into()),
]));

// Stage 1: Encode to bytes
let bytes = node.to_bytes();
assert_eq!(bytes[0], 0x01); // MNode dialect leader

// Stage 1: Decode via ANode
let decoded = anode::decode(&bytes).unwrap();

// Stage 2: Render as pretty JSON
let json = anode_vernacular::render(&decoded, Vernacular::Json);
println!("{}", json);
// Output:
// {
//   "name": "alice",
//   "age": 30,
//   "score": 99.5,
//   "active": true,
//   "tags": [
//     "rust",
//     "vectors"
//   ]
// }

// Stage 2: Parse JSON back to ANode
let reparsed = anode_vernacular::parse(&json, Vernacular::Json).unwrap();

// Verify the roundtrip preserved values
match &reparsed {
    ANode::MNode(m) => {
        assert_eq!(m.fields["name"], MValue::Text("alice".into()));
        assert_eq!(m.fields["age"], MValue::Int(30));
        assert_eq!(m.fields["active"], MValue::Bool(true));
    }
    _ => panic!("expected MNode"),
}

// Re-encode to bytes
let re_bytes = anode::encode(&reparsed);
assert_eq!(re_bytes[0], 0x01); // Still MNode
```

## Compact JSON (JSONL)

For streaming or log output, use `Vernacular::Jsonl`:

```rust
let jsonl = anode_vernacular::render(&decoded, Vernacular::Jsonl);
// Output: {"name":"alice","age":30,"score":99.5,"active":true,"tags":["rust","vectors"]}
```
