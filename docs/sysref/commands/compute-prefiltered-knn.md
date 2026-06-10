# compute prefiltered-knn

Brute-force pre-filter KNN ground truth — produces the **F facet**
(`prefiltered_neighbor_indices` / `prefiltered_neighbor_distances`).
Equivalent to ACORN's `G_K`: top-K of `X_p` (the predicate-passing
base vectors) by distance, perfect recall by construction, `|F| = K`
whenever `|X_p| ≥ K`.

This is ACORN-paper "pre-filtering" (§3.2): filter the candidate set
first, then take top-K. For the sparse post-filter sibling
(`G ∩ R`), see [`compute postfiltered-knn`](compute-postfiltered-knn.md).

The legacy command name `compute filtered-knn` is retained as an alias
to this command — existing pipeline.yaml files keep working unchanged.

## Usage (pipeline step)

```yaml
- id: compute-prefiltered-knn
  run: compute prefiltered-knn
  per_profile: true
  phase: 2
  after: [verify-predicates-sqlite]
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  metadata-indices: metadata_indices.ivvecs
  indices: prefiltered_neighbor_indices.ivecs
  distances: prefiltered_neighbor_distances.fvecs
  neighbors: 100
  metric: L2
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--base` | yes | Base vectors file |
| `--query` | yes | Query vectors file |
| `--metadata-indices` | yes | Predicate result ordinals (ivvec or slab) |
| `--indices` | yes | Output prefiltered neighbor indices (F facet) |
| `--distances` | yes | Output prefiltered neighbor distances (F facet) |
| `--neighbors` | yes | k |
| `--metric` | yes | Distance metric |

## Output semantics

- **`|F| = K`** whenever the predicate matches at least K base vectors
  for that query (sentinel `-1` / `+∞` rows only when `|X_p| < K`).
- **Perfect recall** against an exact filtered ground truth: by
  construction, F contains the K nearest base vectors that pass the
  predicate.
- **Distances** are in FAISS publication convention (smaller is better
  for L2, larger is better for IP/cosine).

## Role in evaluation

F is the verification target for any filtered ANN engine that aspires
to perfect recall — pre-filter scan engines, ACORN-style predicate-
subgraph engines, oracle-partition HNSW. To evaluate engines that
post-filter without rescue scope (a common naive implementation),
use the **E** facet from `compute postfiltered-knn` instead.
