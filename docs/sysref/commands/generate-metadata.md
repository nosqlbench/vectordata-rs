# generate metadata

Generate random integer metadata labels.

## Usage (pipeline step)

```yaml
- id: generate-metadata
  run: generate metadata
  output: profiles/base/metadata_content.u8
  count: 1000000
  fields: 1
  range-min: 0
  range-max: 12
  seed: 42
  format: u8
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--output` | yes | Output file path |
| `--count` | yes | Number of records |
| `--fields` | no | Fields per record (default: 1) |
| `--range-min` | no | Minimum value (default: 0) |
| `--range-max` | no | Maximum value |
| `--seed` | no | Random seed |
| `--format` | no | Output format: u8, i8, u16, i16, u32, i32, u64, i64, slab |
