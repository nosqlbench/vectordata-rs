# analyze select

Display specific records by ordinal.

## Usage

```bash
veks pipeline analyze select --source <file> --range <spec> [--format text|json|csv]
```

## Examples

### Single vector (text)

```bash
veks pipeline analyze select --source profiles/base/base_vectors.fvec --range 0
```

```
[0] 0.593, 0.809, 0.020, -0.255, ...  (128 values)
```

### Scalar metadata values

```bash
veks pipeline analyze select --source profiles/base/metadata_content.u8 --range "0..5"
```

```
[0] 9
[1] 3
[2] 11
[3] 8
[4] 9
```

### Variable-length record (ivvec)

```bash
veks pipeline analyze select --source profiles/default/metadata_indices.ivvec --range 0
```

```
[0] 0, 4, 24, 36, 39, 48, 53, 67, 94, 126, ...  (81 ordinals)
```

### Ground truth neighbors

```bash
veks pipeline analyze select --source profiles/default/neighbor_indices.ivec --range 0
```

```
[0] 288, 434, 291, 153, 283, 26, 22, 648, ...  (100 neighbors)
```

### JSON output

```bash
veks pipeline analyze select --source profiles/base/base_vectors.fvec --range 0 --format json
```

## Range syntax

| Syntax | Meaning |
|--------|---------|
| `42` | Single ordinal |
| `0..5` | Range 0-4 (exclusive end) |
| `[0,10)` | Range 0-9 |
| `0,5,10` | Specific ordinals |
| `0..3,10,20..25` | Mixed |
