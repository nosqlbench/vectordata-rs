# verify knn-consolidated

Multi-threaded brute-force KNN verification across all profiles.

## Usage (pipeline step)

```yaml
- id: verify-knn
  run: verify knn-consolidated
  after: [compute-knn]
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  metric: L2
  normalized: false
  sample: 100
  seed: 42
  output: "${cache}/verify_knn.json"
```

## Behavior

Samples queries, recomputes KNN by brute force, and compares against
stored results. Tie-breaks at the k-th boundary are handled correctly
(duplicate-distance vectors at the boundary count as passes).
