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
cp /path/to/vectors.fvecs _base_vectors.fvecs
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
mkdir my-dataset && cd my-dataset
cp /path/to/base.fvecs _base.fvecs
cp /path/to/query.fvecs _query.fvecs
cp /path/to/gt.ivecs _gt.ivecs
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
- `evaluate-predicates` → `profiles/default/metadata_indices.ivvecs`
  (or `.slab` for complex predicates)
- `verify-predicates-sqlite` → SQLite oracle verification
- `compute filtered-knn` → `profiles/default/filtered_neighbor_indices.ivecs`

Predicate results format depends on the synthesis mode:
- **simple-int-eq**: `.ivvecs` — variable-length ordinal lists, no index
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
mkdir my-dataset && cd my-dataset
cp /path/to/source.fvecs _base_vectors.fvecs
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
      output: profiles/base/base_vectors.fvecs
      dimension: 128
      count: 1000000
      seed: 42
      distribution: gaussian

    - id: generate-queries
      run: generate vectors
      output: profiles/base/query_vectors.fvecs
      dimension: 128
      count: 10000
      seed: 1337
      distribution: gaussian

    # ... (add KNN, metadata, predicates as needed)
```

See `veks/tests/fixtures/synthetic-1k/dataset.yaml` for a complete
example with all BQGDMPRF facets.

---

## Recipe 8: Text Corpus → Embedded Predicated Dataset (GPU)

Building a dataset from raw text rather than existing vectors: acquire a
corpus, chunk it into passages, embed them, join metadata, then bootstrap.
Every stage is a pipeline step, so the whole upstream half is one
`veks run` with provenance, resume, and `dataset.log`.

Build with the CUDA features — the embedding stage is the long pole and
the fused path is roughly three orders of magnitude faster than CPU:

```bash
cargo install --path veks --features embed-cuda-flash   # needs nvcc
```

`upstream/dataset.yaml`:

```yaml
name: corpus-upstream
upstream:
  steps:
    - id: download-corpus
      run: download s2ag
      release: 2026-08-11          # pin: 'latest' is rejected
      dataset-name: s2orc
      api-key-file: keys.yaml      # never inline the key
      output: sources/s2orc_v2
    - id: generate-passages
      run: generate passages
      after: [download-corpus]
      source: sources/s2orc_v2
      files: first:8               # shard subset — a provenance axis
      output: upstream/passages/passages.parquet
      doc-limit: 350000
      doc-order: corpusid
      chunker: para-v1
      seed: 42
    - id: generate-embed
      run: generate embed
      after: [generate-passages]
      source: upstream/passages/passages.parquet
      output: upstream/vectors/base_all.npy
      model: Qwen/Qwen3-Embedding-0.6B
      dtype: bf16                  # pin explicitly: an identity axis
      # device/batch-size default to every visible GPU at batch 128
    - id: verify-alignment
      run: verify alignment
      after: [generate-embed]
      source: upstream/vectors/base_all.npy
      reference: upstream/passages/passages.parquet
      dim: 1024
    - id: generate-metadata        # M facet: parent metadata per passage
      run: generate passage-metadata
      after: [download-papers, generate-passages]
      source: sources/papers
      passages: upstream/passages/passages.parquet
      output: upstream/metadata/metadata.parquet
```

Then the dataset itself — the two raw artifacts are all the bootstrap
needs, and it infers the predicated facets from them:

```bash
veks run upstream/dataset.yaml
veks prepare bootstrap --name corpus-10m --output datasets/corpus-10m \
    --base-vectors upstream/vectors/base_all.npy \
    --metadata upstream/metadata/metadata.parquet \
    --self-search --query-count 10000 \
    --metric Cosine --assume-normalized-like-faiss \
    --neighbors 100 --seed 42 --required-facets BQGMPRF
veks run datasets/corpus-10m/dataset.yaml --output batch
veks check datasets/corpus-10m --check-integrity
```

Notes:

- **Ordinal identity is the whole contract**: row i of the passage table,
  the vectors, and the metadata table all describe the same passage.
  `verify alignment` gates it after each producing stage; run it against
  the metadata table too, not just the vectors.
- **Metric**: Qwen3 embeddings are unit-normalized, so
  `--metric Cosine --assume-normalized-like-faiss` evaluates cosine as
  inner product exactly, with no extra norm work.
- **Deduplicate expectations**: real corpora repeat boilerplate — a 10M
  passage set deduped to 9.76M (5.7%) in practice. Base counts shrink;
  that is the dedup stage working, not data loss.
- **Sizing the dataset**: `--base-count N` takes exactly N base vectors;
  `--base-fraction` takes a share. They are mutually exclusive, and a
  count larger than the source is refused with the source's actual size
  rather than being clamped — a dataset silently smaller than asked for
  invalidates every number derived from it. The count is applied as a
  `limit` on the import, and the row-aligned metadata takes the identical
  subset.
- **Interrupted runs**: split very large embeds into 1–2M-passage steps.
  Text, token ids, and vectors are all resident per step (~5 GB per 1M
  passages at 1024-d), and a step boundary is a resume point.

---

## Key Rules

- **Ordinal correspondence**: `metadata[i]` describes `base_vectors[i]`.
  When shuffling, the same permutation must be applied to both.
- **Underscore prefix**: source files named `_foo.fvecs` are excluded
  from publishing. The wizard handles this automatically.
- **Idempotent pipelines**: `veks run` is resumable. Only stale steps
  re-execute. Use `--clean` for a full reset.
