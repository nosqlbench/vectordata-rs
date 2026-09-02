# verify predicate-strata

Hold a stratified predicate set's claims against the answer keys it
produced, at every profile that has them.

`generate predicates --strategy stratified` records, per query ordinal,
the family, the cell, the exact match count it expects over the census
population and, when the queries' own metadata rows were given, whether
the query's own passage satisfies its predicate. This step reads those
claims back and compares them with what `compute evaluate-predicates`
wrote:

- one record per query, and both annotation namespaces the same length;
- at the profile whose base is the census population, every censused
  record's realised match count equals the recorded one exactly, and a
  control record's lies in its half-decade band;
- at every other profile above the reliability threshold, the realised
  selectivity lies in the record's band and no record is empty; below
  the threshold only the control family must be non-empty;
- every `query_in_filter` label agrees with evaluating the predicate
  against the query's own row.

Realised counts are the results records' lengths, so the pass reads each
profile's results facet once and decodes nothing. It fails on any
violation and writes a JSON report either way.

There is no synthetic-1k example: the fixture has no stratified
predicate set.

## Usage (pipeline step)

```yaml
- id: verify-predicate-strata
  run: verify predicate-strata
  after: evaluate-predicates
  predicates: profiles/base/predicates.slab
  queries: profiles/base/query_vectors.fvecs
  query-metadata: profiles/base/query_metadata.slab
  reliability-threshold: 10000000
  output: ${cache}/verify_predicate_strata.json
```

## Options

| Option | Role | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--predicates` | input | yes | — | Stratified predicate facet, with families and generation namespaces |
| `--results` | config | no | `metadata_results.slab` | Results facet file name under each profile directory |
| `--queries` | input | no | — | Query vectors, to check that there is one predicate per query |
| `--query-metadata` | input | no | — | The queries' own metadata rows, to re-derive every `query_in_filter` label |
| `--reliability-threshold` | config | no | `10000000` | Base count from which every family must hold its band and be non-empty |
| `--output` | output | yes | — | JSON report: per profile, per family, first violations |

## Report

```json
{
  "predicates": 10000,
  "query_count": 10000,
  "census_population": 495930736,
  "label_checks": 10000,
  "label_disagreements": 0,
  "profiles": [
    {"profile": "100k", "base_count": 100000, "census_profile": false, "above_threshold": false,
     "records": 10000, "exact_mismatches": 0, "band_violations": 0, "zero_matches": 0,
     "control_zero_below_threshold": 0,
     "per_family": {"control": {"records": 2500, "mean_claimed_selectivity": 0.0053, "mean_realised_selectivity": 0.0053}}}
  ],
  "violations": 0
}
```

Profiles come from `dataset.yaml` (partition profiles excluded) or, when
it does not load, from the `profiles/` directory with a sized profile's
base count read from its name; the census profile is the one with no
declared count.
