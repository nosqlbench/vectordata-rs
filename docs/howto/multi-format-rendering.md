<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example: Rendering One Record in All Formats

This example shows the same MNode record rendered in every available vernacular
format.

```rust
use vectordata::formats::mnode::{MNode, MValue};
use vectordata::formats::anode::ANode;
use vectordata::formats::anode_vernacular::{self, Vernacular};

let mut node = MNode::new();
node.insert("name".into(), MValue::Text("alice".into()));
node.insert("age".into(), MValue::Int(30));
node.insert("score".into(), MValue::Float(99.5));
node.insert("active".into(), MValue::Bool(true));

let anode = ANode::MNode(node);
```

## CDDL

```
{
  name : tstr,
  age : int,
  score : float,
  active : bool
}
```

## CDDL Value

```
{
  name : "alice",
  age : 30,
  score : 99.5,
  active : true
}
```

## SQL

```
('alice', 30, 99.5, TRUE)
```

## SQL Schema

```
(
  name TEXT,
  age BIGINT,
  score DOUBLE PRECISION,
  active BOOLEAN
)
```

## CQL

```
('alice', 30, 99.5, true)
```

## CQL Schema

```
(
  name text,
  age bigint,
  score double,
  active boolean
)
```

## JSON

```json
{
  "name": "alice",
  "age": 30,
  "score": 99.5,
  "active": true
}
```

## JSONL

```json
{"name":"alice","age":30,"score":99.5,"active":true}
```

## YAML

```yaml
name: alice
age: 30
score: 99.5
active: true
```

## Readout

```
name   : 'alice'
age    : 30
score  : 99.5
active : true
```

## Display

```
{name: 'alice', age: 30, score: 99.5, active: true}
```

## Generating all formats programmatically

```rust
let formats = [
    Vernacular::Cddl, Vernacular::CddlValue,
    Vernacular::Sql, Vernacular::SqlSchema,
    Vernacular::Cql, Vernacular::CqlSchema,
    Vernacular::Json, Vernacular::Jsonl,
    Vernacular::Yaml, Vernacular::Readout,
    Vernacular::Display,
];

for fmt in &formats {
    println!("--- {:?} ---", fmt);
    println!("{}", anode_vernacular::render(&anode, *fmt));
    println!();
}
```
