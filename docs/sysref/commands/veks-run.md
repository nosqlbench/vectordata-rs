# veks run

Execute the pipeline defined in dataset.yaml.

## Usage

```bash
veks run [dataset.yaml] [--clean] [--dry-run]
```

## Example (dry run)

```bash
veks run dataset.yaml --dry-run
```

```
Pipeline: 18 steps in topological order (profile: all)
  1. generate-base (generate vectors)
  2. generate-queries (generate vectors)
  3. scan-zeros (analyze find-zeros)
  4. scan-duplicates (analyze find-duplicates)
  5. generate-metadata (generate metadata)
  6. generate-predicates (generate predicates)
  7. compute-knn (compute knn)
  8. verify-knn (verify knn-consolidated)
  9. evaluate-predicates (compute evaluate-predicates)
  10. verify-predicates-sqlite (verify predicates-sqlite)
  11. compute-filtered-knn (compute filtered-knn)
  12. verify-filtered-knn (verify filtered-knn-consolidated)
  13. generate-dataset-json (generate dataset-json)
  14. generate-variables-json (generate variables-json)
  15. generate-dataset-log-jsonl (generate dataset-log-jsonl)
  16. generate-vvec-index (generate vvec-index)
  17. generate-merkle (merkle create)
  18. generate-catalog (catalog generate)
```

## Options

| Option | Description |
|--------|-------------|
| `--clean` | Reset pipeline (remove generated artifacts, preserve symlinks) |
| `--dry-run` | Show plan without executing |
| `--output tui\|basic\|batch` | Display mode |
| `--resources 'mem:25%-50%'` | Resource governance |
| `--governor maximize\|conservative\|fixed` | Governor strategy |
