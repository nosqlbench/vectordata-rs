<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How to Render an MNode as JSON

## From the CLI

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0"
      format: json
'
```

For single-line (JSONL) output suitable for streaming:

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0..100"
      format: jsonl
'
```

## From Rust code

```rust
use vectordata::formats::anode::{self, ANode};
use vectordata::formats::anode_vernacular::{self, Vernacular};
use vectordata::formats::mnode::{MNode, MValue};

let mut node = MNode::new();
node.insert("name".into(), MValue::Text("alice".into()));
node.insert("age".into(), MValue::Int(30));

let anode = ANode::MNode(node);

// Pretty-printed JSON
let json = anode_vernacular::render(&anode, Vernacular::Json);

// Compact single-line JSON
let jsonl = anode_vernacular::render(&anode, Vernacular::Jsonl);
```

## Type mapping

| MValue variant    | JSON type        |
|-------------------|------------------|
| Text, Ascii, Date, Time, DateTime | string |
| Int, Int32, Short, Millis | number (integer) |
| Float, Float32    | number (float)   |
| Bool              | boolean          |
| Null              | null             |
| List, Set, Array  | array            |
| Map               | object           |
| Bytes, Ulid       | string (hex)     |
| UuidV1, UuidV7    | string (UUID)    |
| Nanos             | object `{epoch_seconds, nano_adjust}` |
