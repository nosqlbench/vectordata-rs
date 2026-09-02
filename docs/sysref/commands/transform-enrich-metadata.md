# transform enrich-metadata

Add the derived columns the stratified predicate families draw on to a
passage-level metadata table, in source order, one output row per
input row:

| column | type | from |
|---|---|---|
| `topic_l1` … `topic_lN` | string | the assignments, named through the label slab |
| `section_class` | string | the heading, through an ordered prefix table |
| `citation_percentile` | int16 | the paper's rank among papers of the same year |
| `passage_position` | int16 | the passage's percent position within its paper |
| `word_count` | int16 | the passage text |
| `sample_bucket` | int32 | `splitmix64(seed, paper, ordinal) mod buckets` — the control family's hash field, never a semantic predicate |

Beside the output it writes a slab of every distinct heading with the
class it received and how many passages carry it, which is what an
auditor of a `section_class` predicate needs to read.

There is no synthetic-1k example: the command joins the passage and
parent tables written by `generate passages`.

## Usage (pipeline step)

```yaml
- id: enrich-metadata
  run: transform enrich-metadata
  after: compute-topic-labels
  metadata: _metadata.parquet
  passages: _passages.parquet
  parents: _parents.parquet
  assignments: ${cache}/topic_assign.u16vecs
  labels: profiles/base/topic_labels.slab
  seed: ${seed}
  output: ${cache}/metadata_enriched.parquet
  section-map-out: profiles/base/section_class_map.slab
```

`convert-metadata` then reads the enriched parquet instead of the
source table, so every downstream facet carries the new columns.

## Options

| Option | Role | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--metadata` | input | yes | — | Source metadata table (parquet), one row per passage |
| `--passages` | input | yes | — | Passage table (parquet): `ordinal` and `text` |
| `--parents` | input | yes | — | Parent table (parquet): `passage_count` and `row_start` per paper |
| `--assignments` | input | yes | — | Topic assignments (u16vecs, one code per level), row-aligned |
| `--labels` | input | no | positional | Topic label slab from `compute topic-labels`; absent means positional labels |
| `--paper-column` | config | no | `corpusid` | Metadata column identifying the paper |
| `--section-column` | config | no | `section` | Metadata column holding the section heading |
| `--year-column` | config | no | `year` | Metadata column holding the publication year |
| `--citations-column` | config | no | `citationcount` | Metadata column holding the citation count |
| `--buckets` | config | no | `16777216` | Modulus of `sample_bucket` (2^24 by default) |
| `--seed` | config | no | `42` | Hash seed for `sample_bucket` |
| `--output` | output | yes | — | Enriched metadata table (parquet) |
| `--section-map-out` | output | no | beside `output`, `section_class_map.slab` | Slab of every distinct heading with its class and count |
| `--report` | output | no | beside `output`, `.json` | Enrichment report JSON: rows, papers, years, distinct headings, share classed `other` |

## Behaviour

Workers map the passage table's row groups in parallel, each reading
the aligned metadata rows by row range and the aligned assignments; one
consumer writes the enriched rows in group order, so the output is
identical for any thread count. The command fails if the assignment
count differs from the metadata row count.
