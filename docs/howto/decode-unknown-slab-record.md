<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How to Decode an Unknown Slab Record

When you have a slab file and don't know whether its records are MNode
(metadata) or PNode (predicates), the ANode auto-detection handles this for
you.

## From the CLI

```
veks pipeline run --steps '
  - slab inspect:
      input: unknown.slab
      ordinals: "0"
      format: json
'
```

The `codec: auto` default reads the first byte of each record to determine the
type:

- `0x01` → MNode (metadata record)
- `0x02` → PNode (predicate tree)

## From Rust code

```rust
use vectordata::formats::anode;

let bytes: Vec<u8> = reader.get(ordinal).unwrap();

match anode::decode(&bytes) {
    Ok(anode::ANode::MNode(m)) => {
        println!("Metadata: {} fields", m.fields.len());
    }
    Ok(anode::ANode::PNode(p)) => {
        println!("Predicate: {}", p);
    }
    Err(e) => {
        eprintln!("Not an ANode record: {}", e);
    }
}
```

## Forcing a specific codec

If you know the record type and want to skip auto-detection (or get a more
specific error message on failure):

```rust
let mnode_result = anode::decode_mnode(&bytes);
let pnode_result = anode::decode_pnode(&bytes);
```

From the CLI, use `codec: mnode` or `codec: pnode`.
