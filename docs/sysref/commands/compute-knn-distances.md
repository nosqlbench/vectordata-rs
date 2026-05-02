# compute knn-distances

Materialize `neighbor_distances.fvecs` from a pre-existing
`neighbor_indices.ivecs` plus the base and query vectors.

## When to use

Some published datasets ship ground-truth neighbor *indices* but no
*distances*. When the dataset.yaml aliases such a source through the
`Identity` slot, the default profile ends up with indices but no
distances, and any downstream consumer that expects the full
`neighbor_distances` facet (verifiers, prebufferers) fails with
"facet not present" / 403 errors.

This command fills that gap. For each query, it walks the recorded
indices, reads the corresponding base vectors, computes the configured
metric, and writes a row-major `.fvecs` output that matches the FAISS
publication convention used by `compute knn` — so verifiers can compare
bit-for-bit between sources that compute distances and sources that
recover them this way.

The bootstrapper auto-emits this step when ground truth is provided as
Identity, no separate distances file is supplied, and there are no
sized profiles or partition oracles (whose own KNN templates would
already cover distances).

## Usage

```bash
veks pipeline compute knn-distances \
  --base    profiles/base/base_vectors.fvec \
  --query   profiles/base/query_vectors.fvec \
  --indices profiles/default/neighbor_indices.ivec \
  --output  profiles/default/neighbor_distances.fvecs \
  --metric  IP
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--base` | yes | — | Base vectors (`.fvec` / `.mvec`) |
| `--query` | yes | — | Query vectors (`.fvec` / `.mvec`) |
| `--indices` | yes | — | Existing neighbor indices (`.ivec`) |
| `--output` | yes | — | Output distances file (`.fvec`) |
| `--metric` | no | `IP` | `L2`, `IP` / `DOT`, `COSINE`, or `L1` |
| `--assume_normalized_like_faiss=true` | when `metric=COSINE` | — | Treat inputs as pre-normalized; evaluate cosine as inner product (FAISS / numpy / knn_utils convention). Exactly one of this and `use_proper_cosine_metric` must be set when `metric=COSINE`. |
| `--use_proper_cosine_metric=true` | when `metric=COSINE` | — | Compute cosine in-kernel from raw vectors via dot/(\|q\|×\|b\|). Use when inputs are not pre-normalized. |

## Element types

Both base and query may be stored as `.fvec` (f32) or `.mvec` (f16).
The command dispatches to a monomorphized inner loop per (base, query)
element-type pair — `(f32, f32)`, `(f32, f16)`, `(f16, f32)`,
`(f16, f16)` — with no dynamic dispatch in the hot path. The sgemm
kernel itself runs in f32; f16 inputs are upcast at read time.

## Output format

Row-major `.fvecs`: one record per query, each record is `(k as i32)
+ k × f32` little-endian. Distances follow the FAISS publication
convention:

| Metric | On-disk value |
|--------|---------------|
| `L2` | squared L2 (positive, smaller = nearer) |
| `IP` / `DOT` | dot product (larger = more similar) |
| `COSINE` | cosine similarity (larger = more similar) |
| `L1` | L1 (Manhattan) distance (positive) |

Padding: out-of-range / missing index entries (rare but possible if
the indices file has `-1` placeholders) write `f32::INFINITY` for
that slot.
