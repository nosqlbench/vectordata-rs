# analyze describe

File format, dimensions, record count, and structure detection.

## Usage

```bash
veks pipeline analyze describe --source <file> [--scan true]
```

## Examples

### Uniform vector file (fvec)

```bash
veks pipeline analyze describe --source profiles/base/base_vectors.fvec
```

```
File:        ./profiles/base/base_vectors.fvec
Format:      fvec
Dimensions:  128
Element:     4 bytes (f32/i32)
Records:     1000
Record size: 516 bytes
File size:   503.9 KB
Structure:   uniform (all records dim=128, 1000 records)
```

### Variable-length vector file (ivvec)

```bash
veks pipeline analyze describe --source profiles/default/metadata_indices.ivvec
```

```
File:        ./profiles/default/metadata_indices.ivvec
Format:      ivvec
Dimensions:  81
Element:     4 bytes (f32/i32)
Records:     102
Record size: 328 bytes
File size:   33.0 KB
Structure:   variable-length (records have different dimensions)
```

### Scalar file

```bash
veks pipeline analyze describe --source profiles/base/metadata_content.u8
```

```
File:        ./profiles/base/metadata_content.u8
Format:      u8
Dimensions:  1
Element:     1 bytes (u8/i8)
Records:     1000
Record size: 5 bytes
File size:   1000 B
```

### Scan mode (record length histogram)

```bash
veks pipeline analyze describe --source profiles/default/metadata_indices.ivvec --scan true
```

Walks all records and reports dimension distribution for variable-length files.

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | File to describe |
| `--format` | no | Format override (auto-detected from extension) |
| `--scan` | no | Scan all records for dimension histogram (vvec only) |
