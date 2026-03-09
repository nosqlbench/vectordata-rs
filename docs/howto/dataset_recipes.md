# Dataset Recipes

Constructing datasets from external sources can be trivial or complex. The dataflow recipes in this section are meant to
be used as a starting point and practical reference.

The goal of sourcing a dataset for the vectordata system is to get it into the internal manifest form which is
universally recognized by any vectordata client. In it simplest form, this is just a directory containing the basic
files (ivec, slab, sqlite, or any recognized encoding form) and a `dataset.yaml` descriptor which is the canonical "
dataset" ToC.

In its richest form, the dataset.yaml provides some or all of the following:

- dataset provenance and related details
- metric and dimensionality
- profiles
    - a profile is a subset of the data by range, with its own unique answer keys (ground truth, predicate indices)
- data preparation steps from upstream sources

The last item is your "easy button" for preparing datasets. By defining the upstream sources and preparation steps, the
whole pipeline becomes a declarative property of the dataset.yaml consumable form itself. This means that processes are
repeatable, idempotent, and iterable in a development sense.

It also means that you should be primarily concerned with filling out the `upstream` section of dataset.yaml as the way
you source and prepare datasets. All of of the available processing commands are also available directly as CLI options
for diagnostic, experimentation and exploration.

## Scenario: You have only base vectors

1. Scrub the base vectors to avoid zero-vectors
2. Create a total-ordering over the vectors lexicographically to eliminate duplicates
3. Create a shuffled order over the base vector ordinal in a persisted file.
4. Extract the first 10K (for example) vectors as the canonical `query_vectors`.
5. Extract the remaining vectors into a file for the canonical `base_vectors`
6. Determine the distance metrics used. If the vectors are verifiably L2-normalized, then use dot product. If not, then
   the user will be required to specify the distance function to use.
7. Using the query_vectors, base_vectors, and distance function, construct a KNN ground-truth set for some K (usually
    100)
8. Save the neighbor distances canonically as `neighbor_distances`.
9. Save the neighbor indices canonically as `neighbor_indices`.
10. Use validation tools to do a few samples of brute-force knn to verify the results.

For a knn-ground-truth dataset, this is the essential flow. The veks command line tools can assist in automating this
process.

## Scenario: Adding Profiles

Profiles allow multiple sizes of the same dataset to coexist without
duplicating base vector files. A sized profile specifies `base_count` and
references the default profile's base vectors with a window.

### Window-based profiles

After the default profile's base vectors are prepared, add sized profiles
to `dataset.yaml`:

```yaml
profiles:
  default:
    base_vectors: profiles/default/base_vectors.fvec
    query_vectors: profiles/default/query_vectors.fvec
    neighbor_indices: profiles/default/neighbor_indices.ivec
  1m:
    base_count: 1000000
    base_vectors: "profiles/default/base_vectors.fvec[0..1M)"
    query_vectors: profiles/default/query_vectors.fvec
  10m:
    base_count: 10000000
    base_vectors: "profiles/default/base_vectors.fvec[0..10M)"
    query_vectors: profiles/default/query_vectors.fvec
```

Each sized profile shares the same physical base vectors file but only
reads the first `base_count` records. KNN ground truth must still be
computed separately per profile since the neighbor set depends on the
base vector count:

```yaml
upstream:
  steps:
    - id: compute-knn
      run: compute knn
      per_profile: true
      base: "${profile_dir}base_vectors.fvec"
      query: query_vectors.fvec
      indices: "${profile_dir}neighbor_indices.ivec"
      distances: "${profile_dir}neighbor_distances.fvec"
      neighbors: 100
      metric: L2
```

When `per_profile: true` is set, the step is expanded for each profile.
For sized profiles, the `base` option will include the window from the
profile's view, so the KNN computation only considers the windowed subset.

### Benefits

- **Disk space**: Sized profiles do not duplicate base vectors.
- **Partial download**: Merkle-based transfer fetches only the pages
  needed for the requested window.
- **Consistency**: All profiles reference the same shuffled base data.

## Scenario: Adding metadata

Metadata is stored in slab files (`metadata_content.slab`) and provides
per-record attributes (captions, labels, scores, URLs, dimensions, etc.)
for the base vectors. The critical rule is:

> **Ordinal correspondence**: `metadata_content.slab[i]` must describe
> `base_vectors[i]`.

How you maintain this depends on the dataset type.

### Cross-modal datasets (no shuffle)

When base and query vectors come from different embedding spaces (e.g.,
image embeddings as base, text embeddings as queries), there is no shuffle
— vectors are imported in shard order and metadata is imported in the same
shard order. Correspondence is preserved automatically:

```yaml
steps:
  - id: import-base
    run: import
    source: img_emb/
    output: base_vectors.hvec

  - id: import-metadata
    run: import
    source: metadata/
    from: parquet
    output: metadata_content.slab
```

Both imports read shards in the same order, so `base_vectors[i]` and
`metadata_content.slab[i]` describe the same record.

For sized profiles, both base vectors and metadata use the same window:

```yaml
profiles:
  10m:
    base_count: 10M
    base_vectors: "base_vectors.hvec[0..10M]"
    metadata_content: "metadata_content.slab[0..10M]"
```

### Self-search datasets (shuffle required)

When base and query vectors are split from the same embedding space via a
shuffle, the base vectors end up in a **shuffled order** that no longer
matches the original import order of the metadata. You must apply the same
shuffle permutation and range to metadata.

```yaml
steps:
  # 1. Import all vectors and metadata in original shard order
  - id: import-all
    run: import
    source: embeddings/
    output: all_vectors.hvec

  - id: import-metadata
    run: import
    source: metadata/
    from: parquet
    output: ${scratch}/metadata_all.slab

  # 2. Generate a reproducible shuffle
  - id: generate-shuffle
    run: generate ivec-shuffle
    after: [import-all]
    output: shuffle.ivec
    interval: "${vector_count}"
    seed: "${seed}"

  # 3. Extract query vectors (first N shuffled indices)
  - id: extract-query
    run: transform hvec-extract
    after: [import-all, generate-shuffle]
    hvec-file: all_vectors.hvec
    ivec-file: shuffle.ivec
    output: query_vectors.hvec
    range: "[0,${query_count})"

  # 4. Extract base vectors (remainder of shuffled indices)
  - id: extract-base
    run: transform hvec-extract
    per_profile: true
    after: [import-all, generate-shuffle]
    hvec-file: all_vectors.hvec
    ivec-file: shuffle.ivec
    output: base_vectors.hvec
    range: "[${query_count},${base_end})"

  # 5. Extract metadata with the SAME shuffle and range
  - id: extract-metadata
    run: transform slab-extract
    per_profile: true
    after: [import-metadata, generate-shuffle]
    slab-file: ${scratch}/metadata_all.slab
    ivec-file: shuffle.ivec
    output: metadata_content.slab
    range: "[${query_count},${base_end})"
```

The `transform slab-extract` step applies the same ivec permutation file
and the same range as `extract-base`, so the output `metadata_content.slab`
is in the same shuffled order as `base_vectors.hvec`. Without this step,
`metadata_content.slab[i]` would refer to the *original* i-th record, not
the i-th base vector.

### Why not let consumers handle the indirection?

Forcing downstream consumers to look up `metadata[shuffle[query_count + i]]`
instead of `metadata[i]` would:

- Complicate every client that uses metadata
- Break the simple ordinal-correspondence contract
- Make windowed profiles impossible (you'd need the full shuffle file)

By extracting metadata into shuffled order during dataset preparation, all
consumers see a simple 1:1 positional correspondence.

## Scenario: Adding predicates

Predicates define metadata filter conditions for filtered KNN evaluation.
They are generated from a statistical survey of the metadata:

```yaml
steps:
  - id: analyze-metadata
    run: survey
    after: [extract-metadata]
    input: metadata_content.slab
    output: ${scratch}/metadata_survey.json
    samples: 10000
    max-distinct: 100

  - id: synthesize-predicates
    run: synthesize predicates
    after: [analyze-metadata]
    input: metadata_content.slab
    survey: ${scratch}/metadata_survey.json
    output: ${scratch}/predicates.slab
    count: 10000
    selectivity: 0.0001
    seed: 42
```

The survey analyzes field distributions and cardinalities. The predicate
generator then produces random filter expressions calibrated to the target
selectivity. Survey and predicates are placed in `${scratch}` since they
are intermediates, not part of the final hosted dataset.

## Scenario: Computing metadata indices

Metadata indices record which base vector ordinals satisfy each predicate.
They are used by `compute filtered-knn` to restrict the candidate set:

```yaml
steps:
  - id: evaluate-predicates
    run: evaluate predicates
    per_profile: true
    after: [extract-metadata, synthesize-predicates, analyze-metadata]
    input: metadata_content.slab
    predicates: ${scratch}/predicates.slab
    survey: ${scratch}/metadata_survey.json
    selectivity: 0.0001
    output: metadata_indices.slab
```

Each record in `metadata_indices.slab` is a packed `[i32 LE]*` array of
base ordinals matching the corresponding predicate. The `per_profile: true`
flag ensures that each sized profile gets its own metadata indices computed
against only the base vectors in that profile's window.

## Scenario: Filtered KNN

With metadata indices computed, filtered KNN finds the K nearest neighbors
among only the base vectors that satisfy each predicate:

```yaml
steps:
  - id: compute-filtered-knn
    run: compute filtered-knn
    per_profile: true
    after: [extract-base, extract-query, evaluate-predicates]
    base: base_vectors.hvec
    query: query_vectors.hvec
    metadata-indices: metadata_indices.slab
    indices: filtered_neighbor_indices.ivec
    distances: filtered_neighbor_distances.fvec
    neighbors: 100
    metric: L2
```

## HDF5 (ann-benchmarks) import

## Raw fvec or ivec import directly

