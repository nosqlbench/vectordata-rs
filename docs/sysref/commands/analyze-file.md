# analyze file

Low-level file metadata.

## Usage

```bash
veks pipeline analyze file --source <file>
```

## Example

```bash
veks pipeline analyze file --source profiles/base/base_vectors.fvec
```

```
File: ./profiles/base/base_vectors.fvec
  Size:       503.9 KB (516000 bytes)
  Format:     .fvec (float32)
  Element:    4 bytes per value
  Dimensions: 128
  Vectors:    1000
  Per-vector: 516 bytes (4 header + 512 data)
  Header overhead: 0.8%
```
