<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Predicated Datasets

A predicated dataset extends a standard vector dataset with metadata and filter
predicates, enabling *filtered* approximate nearest neighbor (ANN) search.

## Dataset facets

A complete predicated dataset contains these facets:

| Facet                         | Format | Description                               |
|-------------------------------|--------|-------------------------------------------|
| `base_vectors`                | fvec   | The corpus of vectors to search            |
| `query_vectors`               | fvec   | Query vectors to run                       |
| `neighbor_indices`            | ivec   | Ground-truth neighbor indices              |
| `neighbor_distances`          | fvec   | Ground-truth neighbor distances            |
| `metadata_content`            | slab   | MNode metadata records per vector          |
| `metadata_predicates`         | slab   | PNode predicate trees per query            |
| `metadata_layout`             | slab   | Metadata field layout schema               |
| `predicate_results`           | slab   | Predicate evaluation result bitmaps        |
| `filtered_neighbor_indices`   | ivec   | Ground-truth filtered neighbor indices     |
| `filtered_neighbor_distances` | fvec   | Ground-truth filtered neighbor distances   |

## The role of MNode and PNode

Each base vector has an associated metadata record (MNode in the
`metadata_content` slab). This record describes properties of the vector —
for example, a product's category, price, creation date, or tags.

Each query has an associated predicate tree (PNode in the
`metadata_predicates` slab). This tree expresses a boolean filter condition —
for example, `(category = 3 AND price <= 100)`.

During filtered ANN search, only vectors whose metadata satisfies the query's
predicate are considered as potential neighbors.

## Slab storage

Both MNode and PNode records are stored in slab files (`.slab`), a
page-aligned record container format. Each record is an opaque byte slice from
the slab's perspective. The dialect leader byte (`0x01` or `0x02`) at the start
of each record enables type identification without external metadata.

## Inspecting predicated datasets

The `slab inspect` command can decode and render both MNode and PNode records
from slab files:

```
# Inspect metadata records
veks pipeline run --steps '
  - slab inspect:
      input: metadata_content.slab
      ordinals: "0..5"
      format: json
'

# Inspect predicate trees
veks pipeline run --steps '
  - slab inspect:
      input: metadata_predicates.slab
      ordinals: "0..5"
      format: sql
'
```

This is useful for validating that metadata and predicates are correctly encoded
after import, or for debugging filtered search results.
