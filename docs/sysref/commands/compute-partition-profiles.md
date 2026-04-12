# compute partition-profiles

Create per-label partition profiles from an existing predicated dataset.

Each partition contains only the base vectors matching a specific
predicate value, with KNN ground truth computed within the partition's
own ordinal space. This provides oracle-quality recall references for
label-specific search evaluation.

## Usage (pipeline step)

```yaml
- id: partition-profiles
  run: compute partition-profiles
  after: [evaluate-predicates]
  base: profiles/base/base_vectors.fvec
  query: profiles/base/query_vectors.fvec
  metadata: profiles/base/metadata_content.u8
  metadata-indices: profiles/default/metadata_indices.ivvec
  neighbors: 100
  metric: L2
```

## Direct invocation

```bash
veks pipeline compute partition-profiles \
  --base profiles/base/base_vectors.fvec \
  --query profiles/base/query_vectors.fvec \
  --metadata profiles/base/metadata_content.u8 \
  --metadata-indices profiles/default/metadata_indices.ivvec \
  --neighbors 100 \
  --metric L2
```

## What it produces

For a dataset with metadata labels 0..12, creates 13 partition profiles:

```
profiles/
├── label-0/
│   ├── base_vectors.fvec          # ~8.3% of base vectors (label == 0)
│   ├── query_vectors.fvec         # symlink → ../../base/query_vectors.fvec
│   ├── neighbor_indices.ivec      # KNN within label-0 partition
│   └── neighbor_distances.fvec    # distances within label-0 partition
├── label-1/
│   ├── base_vectors.fvec
│   ├── ...
```

Each partition profile is added to `dataset.yaml`:

```yaml
profiles:
  label-0:
    base_vectors: profiles/label-0/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
    neighbor_indices: profiles/label-0/neighbor_indices.ivec
    neighbor_distances: profiles/label-0/neighbor_distances.fvec
  label-1:
    ...
```

## Ordinal remapping

Partition base vectors use their own ordinal space (0..N where N is
the partition size). KNN indices reference partition ordinals, not
global ordinals. This matches how a label-specific index would work
in practice — the index only contains vectors with that label.

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--base` | yes | Base vectors file (full dataset) |
| `--query` | yes | Query vectors file |
| `--metadata` | yes | Metadata labels file (scalar) |
| `--metadata-indices` | yes | Predicate results (ivvec or slab) |
| `--neighbors` | yes | k for KNN computation |
| `--metric` | yes | L2, Cosine, or DotProduct |
| `--prefix` | no | Profile name prefix (default: "label") |
| `--labels` | no | Specific label values to partition (default: all) |

## Query handling

All partition profiles share the same query vectors — a symlink is
created rather than copying the file. Each query is evaluated against
only the partition's base vectors, producing the "what if my index
only contained vectors with this label" ground truth.

## When to use

- **Recall benchmarking**: compare ANN recall on partitioned vs full index
- **Oracle ground truth**: exact filtered KNN without approximation
- **Label-specific analysis**: understand how data distribution affects search quality
