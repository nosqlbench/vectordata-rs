# analyze find-zeros

Scan vectors for near-zero entries (L2 norm below threshold).

## Usage

```bash
veks pipeline analyze find-zeros --source <file> [--threshold 1e-6]
```

## Example

```bash
veks pipeline analyze find-zeros --source profiles/base/base_vectors.fvec
```

```
find-zeros: scanning 1000 f32 vectors (dim=128, threshold=1e-6, 128 threads)

  0 near-zero vectors found (L2 < 1e-6) out of 1000
```

Sets `zero_count` and `source_zero_count` pipeline variables.
