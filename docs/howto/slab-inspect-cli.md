<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How-to: `slab inspect` CLI Usage

The `slab inspect` command allows for direct, on-demand decoding of binary slab records.

## Inspect metadata records as CDDL (default)

```bash
veks pipeline slab inspect --input metadata_content.slab --ordinals "0"
```

## Inspect a range as JSON

```bash
veks pipeline slab inspect --input metadata_content.slab --ordinals "0..3" --format json
```

## Inspect predicates as SQL

```bash
veks pipeline slab inspect --input metadata_predicates.slab --ordinals "0,1" --codec pnode --format sql
```

## Mixed ordinals with ranges

```bash
veks pipeline slab inspect --input data.slab --ordinals "0,5..8,100" --format yaml
```

This inspects ordinals 0, 5, 6, 7, and 100.

## Compact output for scripting (JSONL)

```bash
veks pipeline slab inspect --input metadata_content.slab --ordinals "0..1000" --format jsonl
```

Each record is output on a single line, making it suitable for piping to `jq` or other CLI tools.

## Error Handling

- **Decode Error**: Occurs when a record's dialect leader byte is unknown or the payload is malformed.
- **Not Found**: Occurs when an ordinal is outside the slab's valid range.

The command returns a `Warning` status if any individual records fail to decode, but `Ok` if all requested records are processed successfully.
