# compute knn

Brute-force exact KNN ground truth computation.

## Usage (pipeline step)

```yaml
- id: compute-knn
  run: compute knn
  per_profile: true
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  indices: neighbor_indices.ivec
  distances: neighbor_distances.fvec
  neighbors: 100
  metric: L2
  normalized: false
```

## Direct invocation

```bash
veks pipeline compute knn \
  --base profiles/base/base_vectors.fvec \
  --query profiles/base/query_vectors.fvec \
  --indices neighbor_indices.ivec \
  --distances neighbor_distances.fvec \
  --neighbors 100 \
  --metric L2
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--base` | yes | Base vectors file |
| `--query` | yes | Query vectors file |
| `--indices` | yes | Output neighbor indices |
| `--distances` | yes | Output neighbor distances |
| `--neighbors` | yes | k (number of neighbors) |
| `--metric` | yes | L2, Cosine, or DotProduct |
| `--normalized` | no | Vectors are L2-normalized (default: false) |
