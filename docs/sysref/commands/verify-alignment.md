# verify alignment

Verify row-count (ordinal) alignment between two artifacts. Asserts that a
vectors artifact (`source`) has exactly as many records as a reference
artifact (`reference`), optionally asserting the vector dimensionality, and
fails the step on any mismatch.

This is the enforcement point for foreign-input embedding contracts: row i
of an embedded vectors file must embed row i of the passage table it was
derived from. Place it between the embed stage and `prepare bootstrap` so a
truncated, duplicated, or re-ordered vectors artifact fails fast instead of
becoming a dataset facet.

## Usage (pipeline step)

```yaml
- id: verify-alignment
  run: verify alignment
  source: upstream/base_all.npy
  reference: upstream/passages.parquet
  dim: 1024
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | Vectors artifact to check (npy file/dir, xvec, hdf5, slab, vector parquet) |
| `--reference` | yes | Reference artifact whose row count must match (e.g. `passages.parquet`) |
| `--dim` | no | Also assert the source vector dimensionality |

## Notes

- A `.parquet` reference is counted from footer metadata only, so it works
  for non-vector tables such as `passages.parquet`.
- Row-count equality is necessary but not sufficient for ordinal identity —
  it cannot detect a same-length permutation. The embed contract (read in
  row order, write in row order) carries the rest.
