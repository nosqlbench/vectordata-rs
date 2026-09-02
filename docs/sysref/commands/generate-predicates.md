# generate predicates

Generate random equality predicates.

## Usage (pipeline step)

```yaml
- id: generate-predicates
  run: generate predicates
  output: profiles/base/predicates.u8
  count: 10000
  seed: 42
  mode: simple-int-eq
  fields: 1
  range-min: 0
  range-max: 12
  format: u8
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--output` | yes | Output file path |
| `--count` | yes | Number of predicates |
| `--mode` | yes | simple-int-eq or survey |
| `--seed` | no | Random seed |
| `--format` | no | Output format |

## Strategy `stratified`

With `strategy: stratified` the predicates are drawn from the survey's
census tables (`analyze survey` with `census`, `hierarchy` and
`census-pair` declared) so that every family covers every selectivity
decade with a known exact match count. Four families are drawn:
topical (the topic hierarchy), structural (passage-level fields),
bibliographic (paper-level fields) and control (threshold and range
predicates over the hash field `sample_bucket`, never a semantic
predicate). See `docs/design/srd-topic-stratified-predicates.md`.

The facet holds **one record per query ordinal**: record *i* is the
predicate the filtered ground truth evaluates for query *i*. The query
slots are shared equally by the families and split over the decades by
`per-cell`; each cell draws its distinct predicates and pairs them with
slots, repeating a predicate only when its pool is smaller than its
slots, and any slot no cell can fill takes a control predicate. When
the queries are given (with the topic centroids, model and labels),
every topical pair's placement is decided from that query's own
descent: an in-topic pair's query lies in the predicate's topic, an
out-of-topic pair's does not. The `families` namespace records, per
query, the family, selectivity, distinct predicate index, topic label,
placement and the query's own topic; the `generation` namespace records
the cell, pool, census source, expected count, vernacular form and
whether the record was backfilled.

```yaml
- id: generate-predicates
  run: generate predicates
  after: survey-metadata
  output: profiles/base/predicates.slab
  seed: ${seed}
  survey: ${cache}/metadata_survey.json
  strategy: stratified
  base-count: ${base_count}
  queries: profiles/base/query_vectors.fvecs
  centroids: profiles/base/topic_centroids.fvecs
  model: profiles/base/topic_centroids.json
  labels: profiles/base/topic_labels.slab
```

| Option | Role | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--survey` | input | yes | — | Survey report carrying the census tables |
| `--count` | config | no | the number of `queries` | Records to write, one per query ordinal; required without `queries` |
| `--base-count` | config | no | the census population | N of the full base, for the reliability floors in the report |
| `--families` | config | no | `topical,structural,bibliographic,control` | Families to draw, and their order in the output |
| `--topic-fields` | config | no | the survey's first hierarchy | Topic fields outermost first |
| `--bibliographic-fields` | config | no | `citation_percentile,year,isopenaccess` | Censused paper-level fields |
| `--structural-fields` | config | no | `section_class,passage_position,word_count` | Censused passage-level fields |
| `--control-field` | config | no | `sample_bucket` | The hash field of the control family |
| `--buckets` | config | no | `16777216` | Modulus of the control field |
| `--decades` | config | no | `1e-1..1e-7` | Target decades, as a range or a comma list |
| `--per-cell` | config | no | `tapered` | A family's query slots per decade, coarsest first: `tapered` (10, 20, 50, the rest shared by the decades below), one weight, or one entry per decade; numbers alone are weights, with `rest` they are counts |
| `--min-matches` | config | no | `100` | M in the floor s·N ≥ M + 3√M |
| `--reliability-threshold` | config | no | `10000000` | Base count above which the floor is promised |
| `--query-placement` | config | no | `mixed` | Mix of topical pairs whose query lies inside its predicate's topic: `mixed`, `in-topic`, `out-of-topic` or `any`; needs `queries` |
| `--queries` | input | no | — | The query vectors; record i is query i's predicate, and placement is decided per pair |
| `--centroids` | input | no | — | Topic centroids, required with `queries` |
| `--model` | input | no | — | Topic model report, required with `queries` |
| `--labels` | input | no | — | Topic label slab, required with `queries` |
| `--report` | output | no | beside `output`, `.json` | Generation report JSON: per-cell counts, floors, placement |
