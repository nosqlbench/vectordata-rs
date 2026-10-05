# veks-anode

Self-describing binary wire formats for structured metadata records (MNode),
predicate expression trees (PNode), and annotation nodes (ANode).

## Types

### MNode — Metadata Records

An ordered map of named, typed fields. Each field carries a `TypeTag`
discriminant (29 types: scalars, temporals, containers, UUIDs) and a value
from the `MValue` enum. Records are self-describing — no external schema needed.

```rust
use vectordata_wire::mnode::{MNode, MValue};

let mut node = MNode::new();
node.insert("name".into(), MValue::Text("alice".into()));
node.insert("age".into(), MValue::Int(30));
node.insert("score".into(), MValue::Float32(99.5));

let bytes = node.to_bytes();
let decoded = MNode::from_bytes(&bytes).unwrap();
assert_eq!(node, decoded);
```

### PNode — Predicate Trees

Boolean expression trees with field references, comparison operators, and
typed comparands. Used for filtered vector search queries.

```rust
use vectordata_wire::pnode::*;

let tree = PNode::Conjugate(ConjugateNode {
    conjugate_type: ConjugateType::And,
    children: vec![
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named("age".into()),
            op: OpType::Ge,
            comparands: vec![Comparand::Int(18)],
        }),
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named("status".into()),
            op: OpType::In,
            comparands: vec![Comparand::Int(1), Comparand::Int(2)],
        }),
    ],
});

let bytes = tree.to_bytes_named();
let decoded = PNode::from_bytes_named(&bytes).unwrap();
```

## Modules

| Module | Purpose |
|--------|---------|
| `mnode` | MNode, MValue, TypeTag — core metadata record types + codec |
| `mnode::scan` | Zero-allocation binary scanner — evaluate predicates on raw MNode bytes without deserialization |
| `mnode::vernacular` | Human-readable output: SQL, CQL, CDDL representations of MNode records |
| `pnode` | PNode, PredicateNode, ConjugateNode, OpType, Comparand, FieldRef — core predicate tree types + codec |
| `pnode::eval` | Evaluate a PNode predicate tree against an MNode record |
| `pnode::vernacular` | Human-readable output: SQL, CQL, CDDL representations of predicate trees |

## Wire Format

Both types use a dialect leader byte for identification in mixed streams:
- `0x01` — MNode record
- `0x02` — PNode predicate

PNode supports two field reference modes:
- **Indexed** — fields by positional u8 index (compact, for schema-compiled evaluation)
- **Named** — fields by UTF-8 string name (self-describing, for interchange)

## Dependencies

Only `byteorder` and `indexmap`. No serde, no networking, no heavy frameworks.

## License

Apache-2.0
