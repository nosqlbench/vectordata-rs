<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `slab inspect` Command Reference

Pipeline command: `slab inspect`

Source: `src/pipeline/commands/slab.rs` — `SlabInspectOp`

## Description

Decode and render slab records as human-readable text using the two-stage ANode
codec. Records are auto-detected as MNode or PNode by their dialect leader byte,
then rendered in the specified vernacular format.

## Options

| Option     | Type   | Required | Default | Description                                           |
|------------|--------|----------|---------|-------------------------------------------------------|
| `input`    | Path   | yes      | —       | Slab file path                                        |
| `ordinals` | String | yes      | —       | Comma-separated ordinals or ranges (e.g. `0,1,5..10`) |
| `codec`    | enum   | no       | `auto`  | Codec selection: `auto`, `mnode`, `pnode`              |
| `format`   | enum   | no       | `cddl`  | Vernacular format (see below)                         |

### Ordinal syntax

- Single ordinal: `5`
- Comma-separated: `0,1,5`
- Range (exclusive end): `5..10` (produces 5, 6, 7, 8, 9)
- Mixed: `0,5..10,15`

### Codec values

| Value   | Behavior                                                          |
|---------|-------------------------------------------------------------------|
| `auto`  | Read the first byte to determine MNode (`0x01`) or PNode (`0x02`) |
| `mnode` | Force MNode decoding (error if leader byte is not `0x01`)         |
| `pnode` | Force PNode decoding in named mode (error if leader byte is not `0x02`) |

### Format values

`cddl`, `cddl-value`, `sql`, `sql-schema`, `sqlite`, `sqlite-schema`, `cql`,
`cql-schema`, `json`, `jsonl`, `yaml`, `readout`, `display`

See [Vernacular Formats](../explanation/vernacular-formats.md) for details.

## Output

Each record is printed to stderr as:

```
[ordinal]: rendered_output
```

Decode errors are printed as:

```
[ordinal]: DECODE ERROR: message
```

Missing ordinals are printed as:

```
[ordinal]: NOT FOUND (message)
```

## Return status

| Status    | Condition                           |
|-----------|-------------------------------------|
| `Ok`      | All requested ordinals decoded      |
| `Warning` | One or more errors or missing records |

## Example

```yaml
- slab inspect:
    input: metadata.slab
    ordinals: "0..3"
    format: json
```

Output:

```
[0]: {
  "name": "alice",
  "age": 30
}
[1]: {
  "name": "bob",
  "age": 25
}
[2]: {
  "name": "carol",
  "age": 35
}
```
