# compute evaluate-predicates

Evaluate predicates against metadata, producing a variable-length
result file mapping each predicate to its matching base ordinals.

## Usage (pipeline step)

```yaml
- id: evaluate-predicates
  run: compute evaluate-predicates
  per_profile: true
  phase: 1
  source: profiles/base/metadata_content.u8
  predicates: profiles/base/predicates.u8
  mode: simple-int-eq
  fields: 1
  range: "[0,1000000)"
  output: metadata_indices.ivvecs
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | Metadata file (scalar or slab) |
| `--predicates` | yes | Predicates file |
| `--mode` | yes | Evaluation mode (simple-int-eq or survey) |
| `--output` | yes | Output file (.ivvecs or .slab) |
| `--fields` | no | Number of metadata fields |
| `--range` | no | Base vector ordinal range |

Automatically builds an IDXFOR__ offset index after writing vvec output.

## Segment cache

The evaluation walks the metadata facet in segments at fixed ordinal
boundaries (`segment_size`, one million rows by default, reduced when
the memory budget demands it), writes each segment's match lists to
`.cache/` as a predicate-keys slab with a provenance sidecar, and
recombines the output from them, so an interrupted run resumes at a
segment boundary.

A segment is keyed on what determines its content: the source and
predicate facets' provenance, the binary, and the options that change
a match list (`mode`, `fields`, `limit`). The range, the output, the
step id and the segment sizing do not enter the key, so every profile
of a dataset evaluating the same predicates over the same facet names
the same segments, and a larger profile reuses every full segment a
smaller one already produced. Give each profile the range of the base
it holds, `[0,${base_count})`, so its end coincides with the segment
boundaries.
