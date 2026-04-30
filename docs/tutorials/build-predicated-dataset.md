<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Building a Predicated Search Dataset

Create a complete dataset with metadata labels, equality predicates,
and filtered KNN ground truth — everything needed to benchmark
predicated vector search.

## What you'll build

| Facet | What it is |
|-------|-----------|
| B | 1M base vectors (random gaussian, dim=128) |
| Q | 10K query vectors |
| G | Exact KNN ground truth (k=100) |
| M | Random integer labels per base vector (0..12) |
| P | Random equality predicates per query (0..12) |
| R | Matching base ordinals per predicate (variable-length) |
| F | Filtered KNN (neighbors that pass the predicate) |

## Option A: Use the bootstrap wizard

```shell
mkdir my-predicated-dataset && cd my-predicated-dataset

# If you have source vectors:
cp /path/to/base.fvec _base_vectors.fvec
cp /path/to/queries.fvec _query_vectors.fvec

veks bootstrap -i
```

The wizard will:
- Detect the source files
- Ask about metadata synthesis (choose "simple-int-eq")
- Configure label range, predicate count, selectivity
- Generate the full pipeline

```shell
veks run dataset.yaml
```

## Option B: Write the pipeline directly

Create `dataset.yaml`:

```yaml
name: my-predicated-dataset

attributes:
  distance_function: L2

upstream:
  defaults:
    seed: 42

  steps:
    - id: generate-base
      run: generate vectors
      output: profiles/base/base_vectors.fvec
      dimension: 128
      count: 1000000
      seed: 42
      distribution: gaussian

    - id: generate-queries
      run: generate vectors
      output: profiles/base/query_vectors.fvec
      dimension: 128
      count: 10000
      seed: 1337
      distribution: gaussian

    - id: scan-zeros
      run: analyze find-zeros
      after: [generate-base]
      source: profiles/base/base_vectors.fvec

    - id: scan-duplicates
      run: analyze find-duplicates
      after: [generate-base]
      source: profiles/base/base_vectors.fvec

    - id: compute-knn
      run: compute knn
      per_profile: true
      after: [generate-base, generate-queries]
      base: profiles/base/base_vectors.fvec
      query: profiles/base/query_vectors.fvec
      indices: neighbor_indices.ivec
      distances: neighbor_distances.fvec
      neighbors: 100
      metric: L2
      normalized: false

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

    - id: generate-metadata
      run: generate metadata
      after: [generate-base]
      output: profiles/base/metadata_content.u8
      count: 1000000
      fields: 1
      range-min: 0
      range-max: 12
      seed: 42
      format: u8

    - id: generate-predicates
      run: generate predicates
      after: [generate-metadata]
      output: profiles/base/predicates.u8
      count: 10000
      seed: 42
      mode: simple-int-eq
      fields: 1
      range-min: 0
      range-max: 12
      format: u8

    - id: evaluate-predicates
      run: compute evaluate-predicates
      per_profile: true
      phase: 1
      after: [generate-predicates, generate-metadata]
      source: profiles/base/metadata_content.u8
      predicates: profiles/base/predicates.u8
      mode: simple-int-eq
      fields: 1
      range: "[0,1000000)"
      output: metadata_indices.ivvec

    - id: verify-predicates-sqlite
      run: verify predicates-sqlite
      per_profile: true
      phase: 1
      after: [evaluate-predicates]
      metadata: profiles/base/metadata_content.u8
      predicates: profiles/base/predicates.u8
      results: metadata_indices.ivvec
      fields: 1
      output: "${cache}/verify_predicates_sqlite.json"

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

    - id: generate-dataset-json
      run: generate dataset-json
      after: [compute-filtered-knn, verify-predicates-sqlite]

    - id: generate-variables-json
      run: generate variables-json
      after: [generate-dataset-json]

    - id: generate-dataset-log-jsonl
      run: generate dataset-log-jsonl
      after: [generate-dataset-json]

    - id: generate-vvec-index
      run: generate vvec-index
      after: [generate-variables-json, generate-dataset-log-jsonl]
      source: .

    - id: generate-merkle
      run: merkle create
      after: [generate-vvec-index]
      source: .
      min-size: 0

    - id: generate-catalog
      run: catalog generate
      after: [generate-merkle]
      source: .

profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
    neighbor_indices: profiles/default/neighbor_indices.ivec
    neighbor_distances: profiles/default/neighbor_distances.fvec
    metadata_content: profiles/base/metadata_content.u8
    metadata_predicates: profiles/base/predicates.u8
    metadata_indices: profiles/default/metadata_indices.ivvec
    filtered_neighbor_indices: profiles/default/filtered_neighbor_indices.ivec
    filtered_neighbor_distances: profiles/default/filtered_neighbor_distances.fvec
```

Run it:

```shell
veks run dataset.yaml
```

## Inspect the results

```shell
# Verify everything
veks check

# Look at a specific predicate's matches
veks analyze explain-predicates --ordinal 42

# Trace a full filtered KNN query
veks analyze explain-filtered-knn --ordinal 42

# Describe file structure
veks analyze describe --source profiles/default/metadata_indices.ivvec
veks analyze describe --source profiles/default/metadata_indices.ivvec --scan true
```

## Access from Rust

```rust
use vectordata::{open_facet_typed, TestDataGroup, TestDataView, TypedReader};

let group = TestDataGroup::load("./my-predicated-dataset/")?;
let view = group.profile("default").unwrap();

// Verify: every filtered neighbor passes its predicate
let fki = view.filtered_neighbor_indices()?;
let mi  = view.metadata_indices()?;       // Arc<dyn VvecReader<i32>>
let meta: TypedReader<u8> = open_facet_typed(&*view, "metadata_content")?;
let pred: TypedReader<u8> = open_facet_typed(&*view, "metadata_predicates")?;

for qi in 0..10 {
    let pred_val = pred.get_native(qi);
    let neighbors = fki.get(qi)?;
    for &ord in &neighbors {
        if ord < 0 { continue; }
        assert_eq!(meta.get_native(ord as usize), pred_val);
    }
    // (mi.get(qi)? gives the variable-length list of base ordinals
    // matching the predicate, useful for cross-checking coverage.)
    let _ = mi.get(qi)?;
}
```
