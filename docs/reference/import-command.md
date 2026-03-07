<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Import Command Reference

The `veks import` command converts source data into the preferred internal
format for each dataset facet type. Unlike `veks convert`, which performs
arbitrary format-to-format conversion, `import` automatically selects the
output format based on the facet type.

## Facets

| Facet | Preferred Format | Description |
|-------|-----------------|-------------|
| `base_vectors` | fvec | The corpus vectors to search |
| `query_vectors` | fvec | The query vectors to run |
| `neighbor_indices` | ivec | Ground-truth neighbor indices |
| `neighbor_distances` | fvec | Ground-truth neighbor distances |
| `metadata_content` | slab (MNode) | Metadata content records |
| `metadata_predicates` | slab (PNode) | Predicate filter trees |
| `predicate_results` | slab | Predicate filter result bitmaps |
| `metadata_layout` | slab | Metadata field layout schema |
| `filtered_neighbor_indices` | ivec | Filtered ground-truth neighbor indices |
| `filtered_neighbor_distances` | fvec | Filtered ground-truth neighbor distances |

## Single facet mode

```
veks import ./source_data.npy --facet base-vectors -o base_vectors.fvec
```

When `--output` is omitted, the output filename is derived from the facet
key and preferred format extension (e.g. `base_vectors.fvec`).

## Dataset YAML mode

Import can process multiple facets at once from a `dataset.yaml` file:

```
veks import --dataset dataset.yaml
```

The dataset.yaml lists all facets with their output paths and optional
upstream source references:

```yaml
name: my-dataset
description: Example predicated dataset
facets:
  base_vectors:
    path: base_vectors.fvec
    format: fvec
    upstream:
      source: ./raw/base_vectors.npy
      format: npy
  query_vectors:
    path: query_vectors.fvec
    format: fvec
    upstream:
      source: ./raw/queries.npy
  metadata_content:
    path: metadata_content.slab
    format: slab
    upstream:
      source: ./raw/metadata/
      slab_page_size: 65536
      slab_namespace: 1
```

The `upstream` field has the same semantic structure as the facet's output
configuration but describes where to import data from. This allows full
import pipelines to be described declaratively.

Subsequent imports will not replace previously processed source data if it
is complete. Use `--force` to replace them anyway.

## Scaffold generation

Generate an empty dataset.yaml scaffold:

```
veks import --scaffold my-dataset -o dataset.yaml
```

This creates a dataset.yaml with all facet keys pre-populated with default
paths and formats, ready to be filled in. In a scaffold, `xvec` is used as
the source file extension instead of `fvec`, `ivec`, etc., allowing import
to pick the correct variant based on element size.

## Fail-fast validation

When importing from a dataset.yaml, the import command performs two-phase
validation:

1. **Configuration validation**: All facet keys are checked for validity,
   all upstream source paths are verified to exist, and format
   compatibility is checked.
2. **Source probing**: Each upstream source is opened to verify it can be
   read (dimension, record count, format). Any errors here abort before
   any data is written.

Only after both phases pass does the actual import begin, with per-facet
progress indicators.
