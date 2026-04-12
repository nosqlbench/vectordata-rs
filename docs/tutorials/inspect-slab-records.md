<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Inspecting Slab Records

Decode and render slab records in human-readable formats.

## Prerequisites

- A built `veks` binary
- A slab file containing MNode or PNode records (e.g., `metadata.slab`)

## Inspect records

```bash
veks pipeline slab inspect --source metadata.slab --ordinals "0,1,2"
```

## Output formats

### JSON

```bash
veks pipeline slab inspect --source metadata.slab --ordinals "0" --format json
```

### SQL VALUES

```bash
veks pipeline slab inspect --source metadata.slab --ordinals "0" --format sql
```

### YAML

```bash
veks pipeline slab inspect --source metadata.slab --ordinals "0" --format yaml
```

### Readout (tab-indented, colon-aligned)

```bash
veks pipeline slab inspect --source metadata.slab --ordinals "0" --format readout
```

## Ordinal ranges

```bash
veks pipeline slab inspect --source metadata.slab --ordinals "0..5,10,20..25"
```

Inspects ordinals 0-4, 10, and 20-24.

## Force a specific codec

```bash
veks pipeline slab inspect --source predicates.slab --ordinals "0" --codec pnode
```

## Compact output for scripting

```bash
veks pipeline slab inspect --source metadata.slab --ordinals "0..1000" --format jsonl
```

Each record on a single line — pipe to `jq` or other tools.

## Decoding unknown records from Rust

When you don't know if a slab contains MNode or PNode records,
use ANode auto-detection:

```rust
use veks_anode::anode;

let bytes: Vec<u8> = reader.get(ordinal).unwrap();

match anode::decode(&bytes) {
    Ok(anode::ANode::MNode(m)) => println!("Metadata: {} fields", m.fields.len()),
    Ok(anode::ANode::PNode(p)) => println!("Predicate: {}", p),
    Err(e) => eprintln!("Unknown record type: {}", e),
}
```

The first byte determines the type: `0x01` = MNode, `0x02` = PNode.

## Available formats

`cddl` (default), `json`, `jsonl`, `yaml`, `sql`, `cql`, `readout`
