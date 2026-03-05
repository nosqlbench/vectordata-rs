<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example: `slab inspect` CLI Usage

## Inspect metadata records as CDDL (default)

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata_content.slab
      ordinals: "0"
'
```

Output:

```
[0]: {
  name : tstr,
  age : int,
  score : float,
  active : bool
}
```

## Inspect a range as JSON

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata_content.slab
      ordinals: "0..3"
      format: json
'
```

Output:

```
[0]: {
  "name": "alice",
  "age": 30,
  "score": 99.5,
  "active": true
}
[1]: {
  "name": "bob",
  "age": 25,
  "score": 88.0,
  "active": false
}
[2]: {
  "name": "carol",
  "age": 35,
  "score": 95.2,
  "active": true
}
```

## Inspect predicates as SQL

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata_predicates.slab
      ordinals: "0,1"
      codec: pnode
      format: sql
'
```

Output:

```
[0]: (age > 18 AND status IN (1, 2, 3))
[1]: (score >= 90 OR category = 5)
```

## Mixed ordinals with ranges

```
veks pipeline run --steps '
  - slab inspect:
      input: data.slab
      ordinals: "0,5..8,100"
      format: yaml
'
```

Inspects ordinals 0, 5, 6, 7, and 100.

## Compact output for scripting

```
veks pipeline run --steps '
  - slab inspect:
      input: metadata_content.slab
      ordinals: "0..1000"
      format: jsonl
'
```

Each record on a single line, suitable for piping to `jq` or other tools.

## Handling errors

When a record cannot be decoded (e.g., it's not an MNode or PNode):

```
[42]: DECODE ERROR: unknown dialect leader byte: 0x48
```

When an ordinal doesn't exist:

```
[999]: NOT FOUND (ordinal 999 out of range)
```

The command returns `Warning` status if any errors occur, `Ok` if all records
decoded successfully.
