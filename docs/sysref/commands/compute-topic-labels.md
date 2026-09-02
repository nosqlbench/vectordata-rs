# compute topic-labels

Name every cluster of a `compute topics` model with a readable slug
drawn from the passages assigned to it, so a topical predicate reads
`topic_l2 == "wind-turbine-blade"` rather than a code.

There is no synthetic-1k example: the command reads a passage table
(`passages.parquet`, as written by `generate passages`).

## Usage (pipeline step)

```yaml
- id: compute-topic-labels
  run: compute topic-labels
  after: compute-topics
  passages: _passages.parquet
  assignments: ${cache}/topic_assign.u16vecs
  model: profiles/base/topic_centroids.json
  seed: ${seed}
  output: profiles/base/topic_labels.slab
```

## How labels are chosen

A seeded subset of the passage table's row groups is read (never the
whole table); within it, up to `sample-per-cluster` passages per
cluster are tokenised into unigrams and bigrams. Each cluster's terms
are ranked by class TF-IDF against its siblings at the same level, and
the top `top-terms` are joined into a slug. Slugs are unique per level;
a term that is a stem variant of one already in the slug is skipped
(`robot` / `robots` / `robotic`). A cluster with fewer than
`min-sample` sampled passages gets a positional label (`l2-00017`) and
is counted in the report.

The output slab holds one MNode per cluster (`level`, `code`, `label`,
`terms`, `sample_size`, `positional`), in level then code order.

## Options

| Option | Role | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--passages` | input | yes | — | Passage table (parquet), row-aligned with the assignments |
| `--assignments` | input | yes | — | Topic assignments from `compute topics` (u16vecs, one code per level) |
| `--model` | input | yes | — | Model report from `compute topics` (JSON), for the branching per level |
| `--text-column` | config | no | `text` | Column of the passage table holding the text |
| `--row-groups` | config | no | `64` | Row groups read, chosen by seed |
| `--sample-per-cluster` | config | no | `2000` | Cap on passages sampled per cluster at each level |
| `--min-sample` | config | no | `20` | Below this many passages a cluster gets a positional label |
| `--top-terms` | config | no | `3` | Terms joined into the slug |
| `--seed` | config | no | `42` | Row-group and row-order selection |
| `--output` | output | yes | — | Label slab, one MNode per cluster in level then code order |
| `--report` | output | no | beside `output`, `.json` | Labelling report JSON |
