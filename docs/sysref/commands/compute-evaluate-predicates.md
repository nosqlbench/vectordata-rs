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
  output: metadata_indices.ivvec
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | Metadata file (scalar or slab) |
| `--predicates` | yes | Predicates file |
| `--mode` | yes | Evaluation mode (simple-int-eq or survey) |
| `--output` | yes | Output file (.ivvec or .slab) |
| `--fields` | no | Number of metadata fields |
| `--range` | no | Base vector ordinal range |

Automatically builds an IDXFOR__ offset index after writing vvec output.
