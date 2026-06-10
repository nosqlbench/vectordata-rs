# compute postfiltered-knn

Derive the **E facet** — post-filter KNN ground truth — as
`G ∩ R`: the unfiltered top-K of each query intersected with the
predicate-matching base vectors. Cheap (no base/query rereads, no
distance recompute), single-pass; survivors keep G's rank order,
remaining slots are sentinel-padded.

This is ACORN-paper "post-filtering" (§3.2) specialised to the **no-
scope-expansion** case: the search scope is exactly the unfiltered
top-K, and predicate failures are dropped without rescue. The output
matches what a naive post-filter ANN engine would return for a query
whose search scope equals the ground-truth top-K.

For the perfect-recall pre-filter sibling (ACORN `G_K`, full K), see
[`compute prefiltered-knn`](compute-prefiltered-knn.md).

## Usage (pipeline step)

```yaml
- id: compute-postfiltered-knn
  run: compute postfiltered-knn
  per_profile: true
  phase: 2
  after: [compute-knn, compute-evaluate-predicates]
  ground-truth: profiles/{profile}/neighbor_indices.ivecs
  ground-truth-distances: profiles/{profile}/neighbor_distances.fvecs
  metadata-indices: profiles/{profile}/metadata_indices.ivvecs
  indices: postfiltered_neighbor_indices.ivecs
  distances: postfiltered_neighbor_distances.fvecs
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--ground-truth` | yes | Unfiltered top-K indices (G facet, ivec/ivecs) |
| `--ground-truth-distances` | no | Unfiltered top-K distances (D facet). When supplied, survivor distances are copied verbatim into the E output. |
| `--metadata-indices` | yes | Predicate-matching base ordinals per query (R facet, slab or ivvec) |
| `--indices` | yes | Output post-filter neighbor indices (E facet) |
| `--distances` | no | Output post-filter neighbor distances. Requires `--ground-truth-distances`. |

## Algorithm

```
for each query q:
    G_q ← unfiltered top-K neighbors of q          # from --ground-truth
    R_q ← base ordinals passing q's predicate      # from --metadata-indices
    E_q ← [o for o in G_q if o ∈ R_q]              # preserve G rank order
    pad E_q to length K with sentinel (-1 / +inf)
    distances for survivors copied from D
```

## Output semantics

- **Sparse possible.** `|E_q| ∈ [0, K]`; rows that don't fill K are
  padded with `-1` indices and `+∞` distances.
- **Rank-preserving.** Survivors come out in the same order as in G,
  so E remains sorted by distance.
- **Distances** are passed through verbatim from D — same sign
  convention as G (FAISS publication convention).
- **`E_q = ∅`** when none of G's top-K passes the predicate. This is
  the **realistic** signal that a post-filter engine will struggle
  with that query without scope expansion.

## Role in evaluation

E is the verification target for ANN engines that do post-filter
search without rescue scope. Comparing such an engine against the F
facet would conflate algorithm error with semantic mismatch (F is
*always* full K; the engine cannot deliver K when the predicate
excludes its top-K candidates without expanding the scope).

## Cost

Single-pass O(|R_q|) per query for set construction plus O(K) per
query for the membership tests. The producer does **not** open base
or query vectors and does **not** recompute distances. On a 10K-query
dataset this typically completes in under a second.
