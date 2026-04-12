# veks-pipeline

Pipeline engine and command implementations for the vectordata-rs workspace.

This crate provides:
- The DAG execution engine that processes `dataset.yaml` pipelines
- 50+ pipeline commands (analyze, compute, generate, verify, transform, merkle)
- Step lifecycle management (freshness, fingerprinting, resumption)
- Per-profile expansion for sized dataset profiles
- Resource governance (memory/thread budgets)
- UI eventing layer (progress bars, logging)

## Usage

This crate is not intended for direct consumption. It is used by the
[veks](../veks/) CLI binary. Pipeline commands are invoked via:

```bash
veks run dataset.yaml              # execute pipeline
veks pipeline <group> <command>    # direct command invocation
```

## Command Groups

| Group | Commands |
|-------|----------|
| analyze | describe, stats, select, find-zeros, find-duplicates, explain-predicates, explain-filtered-knn |
| compute | knn, filtered-knn, evaluate-predicates, sort |
| generate | vectors, metadata, predicates, vvec-index, shuffle, dataset-json, merkle |
| verify | knn-consolidated, predicates-sqlite, filtered-knn-consolidated |
| transform | extract, convert, ordinals |
| merkle | create, verify, diff, summary |

## Documentation

- [Command Reference](../docs/sysref/commands/README.md) — per-command examples with real output
- [Pipeline Engine](../docs/sysref/04-pipeline.md) — DAG execution, variables, profiles
- [Architecture](../docs/sysref/08-architecture.md) — CommandOp trait, resource governance, UI layer
