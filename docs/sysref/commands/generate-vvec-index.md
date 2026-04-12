# generate vvec-index

Build IDXFOR__ offset index files for variable-length vector files.

## Usage (pipeline step)

```yaml
- id: generate-vvec-index
  run: generate vvec-index
  source: .
```

## Behavior

Scans the workspace for `.ivvec`, `.fvvec`, and other vvec files.
For each, creates `IDXFOR__<name>.<i32|i64>` — a flat-packed array
of byte offsets enabling O(1) random access.

Also checks legacy `.ivec` files: if they have variable-length
records (file size not divisible by stride), builds an index for
those too.
