<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Dataset Recipes

Common patterns for constructing datasets from external sources.
Each recipe produces a directory with a `dataset.yaml` manifest and
the associated data files.

---

## Recipe 1: Base Vectors Only

Starting point: a single vector file.

```bash
mkdir my-dataset && cd my-dataset
cp /path/to/vectors.fvec _base_vectors.fvec
veks bootstrap -i
```

The wizard detects the base vectors and generates a minimal pipeline:
count → scan-zeros → scan-duplicates → prepare → extract → compute-knn
→ verify → merkle → catalog.

```bash
veks run dataset.yaml
```

---

## Recipe 2: Base + Queries + Pre-computed Ground Truth

When you already have KNN results:

```bash
mkdir sift1m && cd sift1m
cp /path/to/base.fvecs _sift_base.fvecs
cp /path/to/query.fvecs _sift_query.fvecs
cp /path/to/gt.ivecs _sift_groundtruth.ivecs
veks bootstrap -i
```

The wizard detects all three roles. With pre-computed GT:
- Shuffle defaults to OFF (seed=0)
- Normalization defaults to OFF
- No extract steps needed (Identity artifacts → symlinks)
- verify-knn still runs to confirm GT correctness

---

## Recipe 3: Adding Sized Profiles

After the default profile is built, add profiles at multiple scales:

```bash
veks prepare stratify
veks run dataset.yaml
```

Sized profiles (10K, 100K, 1M, etc.) share the same base vectors
file but compute independent KNN for each subset. The pipeline's
`per_profile` expansion handles this automatically.

Benefits:
- No disk duplication — profiles reference the same source files
- Merkle-based transfer fetches only needed chunks
- KNN cache segments from smaller profiles are reused by larger ones

---

## Recipe 4: Adding Metadata and Predicates

### Synthesized (simple-int-eq mode)

For random integer labels and equality predicates:

```bash
veks bootstrap -i
# When prompted for metadata synthesis, choose "simple-int-eq"
# Configure: fields=1, range 0..12, format u8
```

This adds:
- `generate metadata` → `profiles/base/metadata_content.u8`
- `generate predicates` → `profiles/base/predicates.u8`
- `evaluate-predicates` → `profiles/default/metadata_indices.ivvec`
  (or `.slab` for complex predicates)
- `verify-predicates-sqlite` → SQLite oracle verification
- `compute filtered-knn` → `profiles/default/filtered_neighbor_indices.ivec`

Predicate results format depends on the synthesis mode:
- **simple-int-eq**: `.ivvec` — variable-length ordinal lists, no index
  file needed for sequential access
- **survey (slab)**: `.slab` — supports arbitrary PNode predicate trees
  with complex conjunctions, no separate offset index needed

### From external metadata

When you have existing metadata (parquet, slab):

```bash
cp /path/to/metadata/ _metadata/
veks bootstrap -i
# The wizard detects _metadata/ and assigns the M role
```

---

## Recipe 5: Self-Search (No Separate Queries)

When base and query vectors come from the same source:

```bash
mkdir glove && cd glove
cp /path/to/glove.fvec _base_vectors.fvec
veks bootstrap -i
# Choose self_search=true, query_count=10000
```

The pipeline:
1. Shuffles the base vectors (randomized train/test split)
2. Extracts the first `query_count` vectors as queries
3. Extracts the remainder as base vectors
4. Computes KNN from queries against base

---

## Recipe 6: HDF5 Import

```bash
mkdir hdf5-dataset && cd hdf5-dataset
cp /path/to/data.hdf5 _source.hdf5
veks bootstrap -i
# The wizard auto-detects HDF5 and prompts for dataset paths:
#   base: _source.hdf5#train
#   query: _source.hdf5#test
#   gt:    _source.hdf5#neighbors (optional)
```

HDF5 datasets are extracted during the pipeline's convert step.

---

## Recipe 7: Fully Synthetic (No Source Data)

Generate everything from scratch using the pipeline:

```yaml
# dataset.yaml
name: synthetic-128d
upstream:
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

    # ... (add KNN, metadata, predicates as needed)
```

See `veks/tests/fixtures/synthetic-1k/dataset.yaml` for a complete
example with all BQGDMPRF facets.

---

## Key Rules

- **Ordinal correspondence**: `metadata[i]` describes `base_vectors[i]`.
  When shuffling, the same permutation must be applied to both.
- **Underscore prefix**: source files named `_foo.fvec` are excluded
  from publishing. The wizard handles this automatically.
- **Idempotent pipelines**: `veks run` is resumable. Only stale steps
  re-execute. Use `--clean` for a full reset.
