<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Inspecting Slab Records with `slab inspect`

This tutorial walks you through using `veks pipeline slab inspect` to decode and render records in various human-readable formats.

## Prerequisites

- A built `veks` binary.
- A slab file containing MNode or PNode records (e.g., `metadata.slab`).

## Step 1: Inspect records with the default format

The simplest invocation retrieves specific records by ordinal and renders them as CDDL (the default format).

```bash
veks pipeline slab inspect --input metadata.slab --ordinals "0,1,2"
```

Output:

```
[0]: {
  name : tstr,
  age : int,
  score : float
}
```

## Step 2: Try different vernacular formats

### JSON (pretty-printed)

```bash
veks pipeline slab inspect --input metadata.slab --ordinals "0" --format json
```

### SQL VALUES

```bash
veks pipeline slab inspect --input metadata.slab --ordinals "0" --format sql
```

### YAML

```bash
veks pipeline slab inspect --input metadata.slab --ordinals "0" --format yaml
```

### Readout (tab-indented, colon-aligned)

```bash
veks pipeline slab inspect --input metadata.slab --ordinals "0" --format readout
```

## Step 3: Use ordinal ranges

You can specify ranges with the `..` syntax (exclusive end):

```bash
veks pipeline slab inspect --input metadata.slab --ordinals "0..5,10,20..25" --format jsonl
```

This inspects ordinals 0–4, 10, and 20–24.

## Step 4: Force a specific codec

By default, `slab inspect` auto-detects the record type from the dialect leader byte. You can override this if needed:

```bash
veks pipeline slab inspect --input predicates.slab --ordinals "0" --codec pnode --format sql
```

## Summary

- Use `veks pipeline slab inspect` for direct record decoding.
- Available formats: `cddl` (default), `sql`, `cql`, `json`, `jsonl`, `yaml`, `readout`.
- Ordinals can be single numbers, lists, or ranges.
