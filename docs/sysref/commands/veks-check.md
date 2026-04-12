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
