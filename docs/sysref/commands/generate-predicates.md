# generate predicates

Generate random equality predicates.

## Usage (pipeline step)

```yaml
- id: generate-predicates
  run: generate predicates
  output: profiles/base/predicates.u8
  count: 10000
  seed: 42
  mode: simple-int-eq
  fields: 1
  range-min: 0
  range-max: 12
  format: u8
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--output` | yes | Output file path |
| `--count` | yes | Number of predicates |
| `--mode` | yes | simple-int-eq or survey |
| `--seed` | no | Random seed |
| `--format` | no | Output format |
