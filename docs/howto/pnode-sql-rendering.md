<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example: PNode → SQL WHERE Clause

This example builds a predicate tree and renders it as a SQL WHERE clause and
other formats.

```rust
use vectordata::formats::pnode::*;
use vectordata::formats::anode::{self, ANode};
use vectordata::formats::anode_vernacular::{self, Vernacular};

// Build a predicate: (age > 18 AND (status IN (1, 2, 3) OR score <= 100))
let tree = PNode::Conjugate(ConjugateNode {
    conjugate_type: ConjugateType::And,
    children: vec![
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named("age".into()),
            op: OpType::Gt,
            comparands: vec![18],
        }),
        PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::Or,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("status".into()),
                    op: OpType::In,
                    comparands: vec![1, 2, 3],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("score".into()),
                    op: OpType::Le,
                    comparands: vec![100],
                }),
            ],
        }),
    ],
});

let anode = ANode::PNode(tree);

// SQL
let sql = anode_vernacular::render(&anode, Vernacular::Sql);
println!("SQL:  {}", sql);
// SQL:  (age > 18 AND (status IN (1, 2, 3) OR score <= 100))

// CQL
let cql = anode_vernacular::render(&anode, Vernacular::Cql);
println!("CQL:  {}", cql);
// CQL:  (age > 18 AND (status IN (1, 2, 3) OR score <= 100))

// CDDL
let cddl = anode_vernacular::render(&anode, Vernacular::Cddl);
println!("CDDL: {}", cddl);
// CDDL: { and: [{ field: "age", op: "gt", value: 18 }, { or: [...] }] }

// JSON
let json = anode_vernacular::render(&anode, Vernacular::Json);
println!("JSON:\n{}", json);
// JSON:
// {
//   "type": "and",
//   "children": [
//     {
//       "type": "predicate",
//       "field": "age",
//       "op": ">",
//       "value": 18
//     },
//     ...
//   ]
// }
```

## Encode, store, and re-render

```rust
// Encode to binary (named mode, with leader byte)
let bytes = tree.to_bytes_named();
assert_eq!(bytes[0], 0x02); // PNode dialect

// Later: decode from slab
let decoded = anode::decode(&bytes).unwrap();
let sql_again = anode_vernacular::render(&decoded, Vernacular::Sql);
```
