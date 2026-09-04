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
  control record's is credible under its binomial construction;
- at every other profile, every record's count is **credible** under
  its sampling model: the base is a seeded shuffle, so a censused
  predicate's matches in the first *N* rows are a hypergeometric draw
  of its census count from the population, and a control predicate's
  are a binomial draw at its constructed selectivity. A count is
  credible when it lies inside the model's two-sided `1 − 1e-9`
  region, computed exactly, so an empty record is credible up to about
  twenty-one expected matches and a count six sigma from a mean of a
  million is not;
- above the reliability threshold, a record that clears the profile's
  floor `M + 3√M` (`min-matches`) is non-empty;
- at the census profile, every record's claimed selectivity lies in
  its cell's half-decade band, the generator's own invariant; and at
  every profile above the threshold, every cell whose decade clears
  the floor holds at least one record realised inside its band, so the
  ladder is populated where the design promises it. A cell that does
  not is **uncovered**;
- every `query_in_filter` label agrees with evaluating the predicate
  against the query's own row.

Sampling noise around a cell's half-decade band and empties where few
matches are expected are what the models predict at a sized profile.
They are reported — the empties against the number the models predict
— and not failed. A control record at 1e-7 expects 0.01 matches at a
100k base and is empty ninety-nine times in a hundred.

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

`min-matches` defaults to the sampler's default; give it the sampler's
value when that was changed, so both agree on which records apply to a
profile.

## Options

| Option | Role | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--predicates` | input | yes | — | Stratified predicate facet, with families and generation namespaces |
| `--results` | config | no | `metadata_results.slab` | Results facet file name under each profile directory |
| `--queries` | input | no | — | Query vectors, to check that there is one predicate per query |
| `--query-metadata` | input | no | — | The queries' own metadata rows, to re-derive every `query_in_filter` label |
| `--reliability-threshold` | config | no | `10000000` | Base count from which a record that clears the floor must be non-empty |
| `--min-matches` | config | no | `100` | `M` in the floor `s·N ≥ M + 3√M` that decides which records apply to a profile |
| `--output` | output | yes | — | JSON report: per profile, per family, first violations |

## Report

```json
{
  "schema_version": 2,
  "predicates": 10000,
  "query_count": 10000,
  "census_population": 495930736,
  "reliability_threshold": 10000000,
  "min_matches": 100,
  "label_checks": 10000,
  "label_disagreements": 0,
  "profiles": [
    {"profile": "100k", "base_count": 100000, "census_profile": false, "above_threshold": false,
     "floor": 130.0, "records": 10000,
     "exact_mismatches": 0, "incredible_counts": 0,
     "applicable": 0, "applicable_empty": 0,
     "empties": 1769, "empties_expected": 1761.4, "out_of_band": 4312,
     "band_violations": 0, "uncovered_cells": 0,
     "cells": {"control:1e-4": {"records": 605, "applicable": false, "in_band": 590, "below_band": 15, "above_band": 0, "empty": 0}},
     "per_family": {"control": {"records": 3710, "mean_claimed_selectivity": 0.005536, "mean_realised_selectivity": 0.005539,
                                "out_of_band": 1180, "empties": 1769}},
     "first_violations": []}
  ],
  "violations": 0
}
```

`exact_mismatches`, `incredible_counts`, `applicable_empty`,
`band_violations` and `uncovered_cells` are violations. `empties`,
`empties_expected`, `out_of_band`, `applicable` and the per-cell
`cells` table describe the profile: below the threshold nothing
applies, and at a small base most of the finest cells are empty, as
predicted. The `cells` table is where a cell's realised spread at a
size is read: how many of its records landed in the band, below it,
above it, or empty.

Profiles come from `dataset.yaml` (partition profiles excluded) or, when
it does not load, from the `profiles/` directory with a sized profile's
base count read from its name; the census profile is the one with no
declared count.
