# generate vectors

Generate random vectors with configurable distribution.

## Usage (pipeline step)

```yaml
- id: generate-base
  run: generate vectors
  output: profiles/base/base_vectors.fvec
  dimension: 128
  count: 1000000
  seed: 42
  distribution: gaussian
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--output` | yes | Output file path |
| `--dimension` | yes | Vector dimensionality |
| `--count` | yes | Number of vectors |
| `--seed` | no | Random seed (default: 0) |
| `--distribution` | no | gaussian or uniform (default: gaussian) |
