# verify similarity

Compare two vector artifacts row-by-row. Streams two f32 artifacts of
identical shape (npy file/dir, xvec, hdf5, slab, vector parquet) through
the veks-core readers and reports per-row cosine statistics (min, mean,
worst row index) plus the maximum deviation of `source` rows from unit
L2 norm. With `--min-cosine` set, the step fails when any row's cosine
falls below the threshold.

## Usage (pipeline step)

```yaml
- id: verify-similarity
  run: verify similarity
  source: upstream/vectors/base_all_v2.npy
  reference: upstream/vectors/base_all.npy
  min-cosine: 0.995
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | Vector artifact under test |
| `--reference` | yes | Golden vector artifact of identical shape |
| `--min-cosine` | no | Fail when any row's cosine falls below this threshold |

## Notes

- Embedding outputs are floating-point and implementation-shaped: kernel
  fusion, dtype, batching, and device changes all reorder reductions, so
  byte-identity is the wrong equivalence for "same embedding". This
  command makes the right equivalence — bounded per-row cosine drift — a
  first-class, recordable pipeline step (e.g. gate a re-embed with a new
  binary revision against the previous artifact before it replaces a
  dataset facet). Same-binary, same-options re-runs of `generate embed`
  are bit-identical and score cosine exactly 1.0.
- Calibration from the S2OA pilot set (bf16, Qwen3-Embedding-0.6B):
  cross-revision drift lands around min cosine 0.9995–0.9999; anything
  below ~0.995 warrants investigation.
- Accumulation is f64; throughput is reader-bound (~250k rows/s at
  1024-d).
