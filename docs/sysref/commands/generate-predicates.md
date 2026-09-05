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
whether the record was backfilled. With `query-metadata` (the queries'
own metadata rows, as `transform extract` writes them over the query
range of the shuffle), every record also carries `query_in_filter`:
whether the query's own passage satisfies its predicate. That is the
one relation a structural or bibliographic pair has to its query, and
it is recorded rather than assumed; `verify predicate-strata`
re-derives it.

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
| `--query-metadata` | input | no | — | The queries' own metadata rows, in query order; every pair is labelled `query_in_filter` by evaluating its predicate against its query's row |
| `--centroids` | input | no | — | Topic centroids, required with `queries` |
| `--model` | input | no | — | Topic model report, required with `queries` |
| `--labels` | input | no | — | Topic label slab, required with `queries` |
| `--report` | output | no | beside `output`, `.json` | Generation report JSON: per-cell counts, floors, placement |

## Uniform strategy

`--strategy uniform` writes a set in which every predicate takes **one
form at one level** (PS-23, PL-2): the form is `field.access` parts
joined by `+` (a conjunction) or `|` (a disjunction), the level is the
selectivity the set is planned for, and every predicate lands in the
half-decade band around it. Literals come from the survey's census; a
two-part conjunction the census tabulated as a pair takes its exact
count, anything else is estimated from the parts' marginals under
independence and the record says so. A part may be a no-op that holds
the form constant. The set's profile is tagged `family: uniform`,
`predicates: uniform-<n>`, `form`, `form_shape` and `selectivity`, and
`veks check` holds the facet to a form census.

```bash
veks generate predicates --strategy uniform \
  --form topic_l3.eq+citation_percentile.range --selectivity 1e-2 \
  --count 10000 --survey .cache/metadata_survey.json \
  --output profiles/10m-uniform-2-1e-2/predicates.slab
```

| Option | Required | Description |
|---|---|---|
| `--form` | yes | The one form: `field.eq`, `field.range` (a lower bound) and `field.le` parts under `+` or `\|` |
| `--selectivity` | yes | The level, e.g. `1e-2`; kept as spelled in the `selectivity` tag and the set's name |
| `--count` | yes | Records to write, one per query ordinal |
| `--survey` | yes | The censused survey the literals come from |
| `--band` | no | Band factor around the level (default √10) |
| `--report` | no | Generation report (default: beside the output as `.json`) |

Sets are declared with `veks prepare predicate-sets`, which writes one
profile per size and level under its size layer and one generator step
per set.
