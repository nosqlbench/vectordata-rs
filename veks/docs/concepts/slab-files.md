<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Slab Files

Slab files (`.slab`) are page-aligned record containers provided by the
`slabtastic` crate. They are the storage format for metadata records (MNode),
predicate trees (PNode), and other binary facets in predicated datasets.

## Key properties

- **Page-aligned**: Records are organized into fixed-size pages for efficient
  I/O. Pages can be read independently without scanning the entire file.
- **Ordinal-addressed**: Each record has a sequential ordinal (starting at 0)
  that serves as its address. Records can be retrieved by ordinal via
  `reader.get(ordinal)`.
- **Opaque payloads**: The slab format treats record contents as raw byte
  slices. It does not know or care about MNode/PNode encoding — that
  interpretation happens at the codec layer above.
- **Append-friendly**: New records can be appended to an existing slab without
  rewriting the entire file.
- **Namespace support**: A slab file can contain multiple namespaces, each with
  its own page index.

## Slab commands

The pipeline provides commands for working with slab files:

| Command            | Purpose                                           |
|--------------------|---------------------------------------------------|
| `slab import`      | Import records from text/cstring/slab source       |
| `slab export`      | Export records as text, hex, raw, or JSON          |
| `slab append`      | Append records from one slab to another            |
| `slab rewrite`     | Rewrite a slab with clean page alignment           |
| `slab check`       | Validate structural integrity                      |
| `slab get`          | Extract specific records by ordinal               |
| `slab analyze`     | Report statistics (pages, records, sizes)           |
| `slab explain`     | Display page layout with ASCII diagrams            |
| `slab namespaces`  | List namespaces in a slab file                     |
| `slab inspect`     | Decode and render records as ANode vernacular text |

## The inspect command

`slab inspect` bridges the gap between raw slab storage and human
understanding. It combines `slab get` (retrieve by ordinal) with the ANode
codec (decode binary → render text). See the
[slab inspect reference](../reference/slab-inspect.md) for details.
