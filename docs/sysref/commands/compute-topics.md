# compute topics

Hierarchical spherical k-means over a vector facet: fit the tree on a
sample, then assign every base vector one code per level. The
assignments are the topical dimension of the stratified predicate
families (`docs/design/srd-topic-stratified-predicates.md`, §10.1).

## Usage (pipeline step)

```yaml
- id: compute-topics
  run: compute topics
  base: _base_all.fvecs                      # source order; a sharded series is accepted
  sample: profiles/base/base_vectors.fvecs   # shuffled, so a prefix is a uniform sample
  sample-order: prefix
  sample-size: 5000000
  levels: 10,30,33
  seed: ${seed}
  centroids: profiles/base/topic_centroids.fvecs
  output: ${cache}/topic_assign.u16vecs
  margin: ${cache}/topic_margin_all.mvecs
```

The assignments follow the order of `base`; when that is the raw
source order (as above), `transform extract` over the shuffle carries
`output` and `margin` into base order alongside the metadata.

## Direct invocation

```bash
veks pipeline compute topics \
  --base profiles/label_00/base_vectors.fvec \
  --levels 4,4 \
  --centroids topic_centroids.fvec \
  --output topic_assign.u16vec \
  --margin topic_margin.mvec
```

## Example

Run against the synthetic-1k fixture's `label_00` profile (80 vectors,
dim 128):

```
topics: fitting levels [4, 4] on 80 of 80 rows from base_vectors.fvec (Strided), dim 128, 128 threads, AVX-512 kernel
topics: level 1 — 4 clusters, 0 empty, 1/1 runs converged, max movement 1.34e-7, 0 repairs
topics: level 2 — 16 clusters, 0 empty, 4/4 runs converged, max movement 8.94e-8, 0 repairs
20 centroids over 2 levels fitted on 80 rows in 0.0s; 80 vectors assigned in 0.0s
Produced:
  topic_centroids.fvec
  topic_assign.u16vec
  topic_margin.mvec
  topic_centroids.json
```

The model report written beside the centroids:

```json
{
 "schema_version": 1,
 "dim": 128,
 "levels": [4, 4],
 "total_centroids": 20,
 "sample": "profiles/label_00/base_vectors.fvec",
 "sample_size": 80,
 "sample_order": "strided",
 "seed": 42,
 "iterations": 50,
 "tolerance": 0.0001,
 "normalize": true,
 "kernel": "AVX-512",
 "fit_seconds": 0.000182356,
 "per_level": [
  {"branching": 4, "clusters": 4, "empty": 0, "runs": 1, "converged": 1,
   "max_final_movement": 1.3411045e-07, "repairs": 0},
  {"branching": 4, "clusters": 16, "empty": 0, "runs": 4, "converged": 4,
   "max_final_movement": 8.940697e-08, "repairs": 0}
 ],
 "assignment": {
  "base": "profiles/label_00/base_vectors.fvec",
  "records": 80,
  "seconds": 0.001534219,
  "margin_written": true
 }
}
```

## Options

| Option | Role | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--base` | input | yes | — | Vector facet to assign, in the order the assignments must follow; accepts a sharded series |
| `--sample` | input | no | `base` | Vector facet to fit on; a shuffled facet with `sample-order: prefix` is the cheap uniform sample |
| `--sample-size` | config | no | `5000000` | Rows fitted |
| `--sample-order` | config | no | `strided` | `prefix` (first rows; right for a shuffled facet) or `strided` (evenly spaced; right for corpus order) |
| `--levels` | config | no | `10,30,33` | Branching per level, outermost first |
| `--iterations` | config | no | `50` | Iteration cap per k-means run |
| `--tolerance` | config | no | `1e-4` | Mean centroid movement (cosine distance) below which a run has converged |
| `--seed` | config | no | `42` | Seed for k-means++ and sampling |
| `--normalize` | config | no | `true` | Unit-normalise vectors before fitting and assigning |
| `--centroids` | output | yes | — | Centroid facet, fvecs, all levels in order |
| `--output` | output | yes | — | Assignments, u16vecs: one record per base vector, one code per level |
| `--margin` | output | no | — | Leaf margin facet, mvecs dim 2: distance to the chosen leaf and to its best sibling; omit to skip |
| `--model` | output | no | beside `centroids`, `.json` | Model report JSON |

## Determinism

The fit reduces in fixed 4096-row chunks in f64, so the centroids are
bit-identical for any thread count; the assignment is a pure function
of the centroids. Rerunning with the same seed and inputs reproduces
the outputs exactly.
