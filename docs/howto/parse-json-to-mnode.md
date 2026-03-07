<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How to Parse JSON Text into an MNode

The vernacular codec can parse JSON objects into MNode records, enabling a text
→ binary pipeline.

## From Rust code

```rust
use veks::formats::anode::{self, ANode};
use veks::formats::anode_vernacular::{self, Vernacular};

let json = r#"{"name": "alice", "age": 30, "active": true, "tags": [1, 2, 3]}"#;

let anode = anode_vernacular::parse(json, Vernacular::Json).unwrap();

// Encode to binary
let bytes = anode::encode(&anode);

// The result is a valid MNode with dialect leader byte
assert_eq!(bytes[0], 0x01);
```

## Type inference from JSON

| JSON type  | MValue variant |
|------------|----------------|
| string     | Text           |
| integer    | Int            |
| float      | Float          |
| boolean    | Bool           |
| null       | Null           |
| array      | List           |
| object     | Map (nested MNode) |

## Supported parse formats

The following vernaculars support parsing text → ANode:

| Format   | Input shape                          | Example                       |
|----------|--------------------------------------|-------------------------------|
| json     | JSON object                          | `{"k": "v", "n": 42}`        |
| jsonl    | Same as json                         | `{"k":"v","n":42}`            |
| sql      | SQL VALUES tuple                     | `('alice', 42, TRUE, NULL)`   |
| cql      | CQL VALUES tuple                     | `('alice', 42, true, null)`   |
| cddl     | CDDL group                           | `{ name : tstr, count : int }`|
| yaml     | YAML key-value lines                 | `name: alice\nage: 30`        |
| readout  | Colon-aligned key-value lines        | `name : 'alice'\nage  : 30`   |

Unsupported parse directions (e.g., `display`, `sql-schema`) return an error.
