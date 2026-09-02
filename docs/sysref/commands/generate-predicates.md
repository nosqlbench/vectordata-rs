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
decade with a known exact match count. Four families are written, each
as its own namespace in `predicates.slab` (`families`, `generation`):
topical (the topic hierarchy), structural (passage-level fields),
bibliographic (paper-level fields) and control (threshold and range
predicates over the hash field `sample_bucket`, never a semantic
predicate). See `docs/design/srd-topic-stratified-predicates.md`.

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
| `--base-count` | config | no | the census population | N of the full base, for the reliability floors in the report |
| `--families` | config | no | `topical,structural,bibliographic,control` | Families to draw, and their order in the output |
| `--topic-fields` | config | no | the survey's first hierarchy | Topic fields outermost first |
| `--bibliographic-fields` | config | no | `citation_percentile,year,isopenaccess` | Censused paper-level fields |
| `--structural-fields` | config | no | `section_class,passage_position,word_count` | Censused passage-level fields |
| `--control-field` | config | no | `sample_bucket` | The hash field of the control family |
| `--buckets` | config | no | `16777216` | Modulus of the control field |
| `--decades` | config | no | `1e-1..1e-7` | Target decades, as a range or a comma list |
| `--per-cell` | config | no | `tapered` | Predicates per (family, decade) cell: `tapered` (10, 20, then 50), one count, or one per decade coarsest first |
| `--min-matches` | config | no | `100` | M in the floor s·N ≥ M + 3√M |
| `--reliability-threshold` | config | no | `10000000` | Base count above which the floor is promised |
| `--query-placement` | config | no | `mixed` | `mixed`, `in-topic`, `out-of-topic` or `any`; needs `queries` |
| `--queries` | input | no | — | Query vectors, for in-topic / out-of-topic placement |
| `--centroids` | input | no | — | Topic centroids, required with `queries` |
| `--model` | input | no | — | Topic model report, required with `queries` |
| `--labels` | input | no | — | Topic label slab, required with `queries` |
| `--report` | output | no | beside `output`, `.json` | Generation report JSON: per-cell counts, floors, placement |
