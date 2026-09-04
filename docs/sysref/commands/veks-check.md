# veks check

Preflight verification for dataset readiness.

## Usage

```bash
veks check [--check-all] [--json] [--quiet]
```

## Example

```bash
veks check
```

```
✓ check pipeline-execution: ok
    1 dataset(s), 18/18 steps fresh
✓ check pipeline-coverage: ok
    all publishable files have pipeline coverage
✓ check dataset-attributes: ok
    all required attributes present
✓ check merkle: ok
    10 file(s) >= 0 B, all have current .mref
✓ check integrity: ok
    10 data file(s) checked, all valid
✓ check extraneous-files: ok
    all publishable files are accounted for by the pipeline
```

## Individual checks

```bash
veks check --check-integrity
veks check --check-merkle
veks check --check-pipelines
veks check --check-publish
veks check --check-catalogs
veks check --check-extraneous
```

## Cleanup

```bash
veks check --clean          # list extraneous files
veks check --clean-files    # remove extraneous files
```

## What counts as accounted for

A publishable file passes the extraneous-files check when a pipeline
step or profile view names it, when it is known infrastructure, when it
is static payload — `LICENSE`, `NOTICE`, `README`, `CITATION` files an
operator places by hand, which ship with the data, are merkled like any
content, and are never cleaned or asked of a step — or when
it is a derivative of something named: a `.mref` of it, an `IDXFOR__`
index of it, or a shard of it. The shards of a sharded facet
(`base_vectors__0000.fvecs`, `base_vectors__0001.fvecs`, …) are
accounted for by the series the manifest knows as `base_vectors.fvecs`
or declares as `base_vectors__NNNN.fvecs`.

A profile directory the definition no longer names — typically one a
removed stratum generated — shows up here as extraneous; `veks prepare
cleanup-profiles` is the remedy, since it also drops the profile's entry from
the definition.

The manifest is projected from the pipeline definition with the same
variables a run sees: the upstream defaults, the dataset's `variables:`
section, and `variables.yaml`. A step whose options name a variable a
run records, such as `${vector_count}`, is therefore projected once the
run has recorded it, and cannot be before.
