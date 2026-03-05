<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Inspecting Slab Records with `slab inspect`

This tutorial walks you through creating a slab file containing MNode metadata
records and using `slab inspect` to decode and render them in various
human-readable formats.

## Prerequisites

- A built `veks` binary
- Familiarity with the pipeline command system

## Step 1: Create a slab file with MNode records

First, use the pipeline to import some data into a slab file. For this tutorial
we will build MNode records programmatically and write them to a slab.

If you already have a slab file containing MNode-encoded metadata (for example
from a predicated dataset's `metadata_content` facet), skip to Step 2.

```
veks pipeline run --steps '
  - slab import:
      from: metadata_source.txt
      to: metadata.slab
'
```

## Step 2: Inspect records with the default format

The simplest invocation retrieves specific records by ordinal and renders them
as CDDL (the default format):

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0,1,2"
'
```

Output:

```
[0]: {
  name : tstr,
  age : int,
  score : float
}
```

## Step 3: Try different vernacular formats

### JSON (pretty-printed)

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0"
      format: json
'
```

### SQL VALUES

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0"
      format: sql
'
```

### YAML

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0"
      format: yaml
'
```

### Readout (tab-indented, colon-aligned)

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0"
      format: readout
'
```

## Step 4: Use ordinal ranges

You can specify ranges with the `..` syntax (exclusive end):

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata.slab
      ordinals: "0..5,10,20..25"
      format: jsonl
'
```

This inspects ordinals 0–4, 10, and 20–24.

## Step 5: Force a specific codec

By default, `slab inspect` auto-detects the record type from the dialect leader
byte. You can override this:

```
veks pipeline run --steps '
  - slab inspect:
      input: predicates.slab
      ordinals: "0"
      codec: pnode
      format: sql
'
```

## What you learned

- How to use `slab inspect` to decode binary slab records
- The available vernacular formats: cddl, sql, cql, json, jsonl, yaml, readout, display
- How ordinal ranges work
- How to force a specific codec (mnode or pnode)
