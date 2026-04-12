# compute filtered-knn

KNN with predicate pre-filtering. Only base vectors matching the
query's predicate are considered as candidates.

## Usage (pipeline step)

```yaml
- id: compute-filtered-knn
  run: compute filtered-knn
  per_profile: true
  phase: 2
  after: [verify-predicates-sqlite]
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  metadata-indices: metadata_indices.ivvec
  indices: filtered_neighbor_indices.ivec
  distances: filtered_neighbor_distances.fvec
  neighbors: 100
  metric: L2
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--base` | yes | Base vectors file |
| `--query` | yes | Query vectors file |
| `--metadata-indices` | yes | Predicate result ordinals (ivvec or slab) |
| `--indices` | yes | Output filtered neighbor indices |
| `--distances` | yes | Output filtered neighbor distances |
| `--neighbors` | yes | k |
| `--metric` | yes | Distance metric |
