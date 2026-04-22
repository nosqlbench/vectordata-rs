# veks-pipeline

Pipeline engine and command implementations for the vectordata-rs workspace.

This crate provides:
- The DAG execution engine that processes `dataset.yaml` pipelines
- 50+ pipeline commands (analyze, compute, generate, verify, transform, merkle)
- Step lifecycle management (freshness, fingerprinting, resumption)
- Per-profile expansion for sized dataset profiles
- Resource governance (memory/thread budgets)
- UI eventing layer (progress bars, logging)

## Verified against FAISS and numpy

This crate hosts all four KNN engines — `compute_knn` (SimSIMD),
`compute_knn_stdarch` (pure `std::arch`), `compute_knn_blas`
(`cblas_sgemm`), and `compute_knn_faiss` (FAISS) — plus the shared
[`knn_compare`](src/pipeline/commands/knn_compare.rs) classifier
and the end-to-end [`verify_dataset_knnutils`](src/pipeline/commands/verify_dataset_knnutils.rs)
command. In-tree conformance tests assert *zero* differing neighbors across
engines on deterministic fixtures; the `verify engine-parity`
demo command defaults to `--boundary-tolerance 0` so any disagreement
fails the verdict. The `BOUNDARY_THRESHOLD = 5` constant in
`knn_compare.rs` is a defensive ceiling for the multi-threaded BLAS
regime at billion-vector scale, not a slack used at unit-test or
default-demo sizes. See the
[conformance section](../docs/sysref/12-knn-utils-verification.md#127-cross-engine-conformance-testing)
for observed numbers, the test catalog, and the two degenerate
regimes where users might opt into `--boundary-tolerance > 0`.

**Live demo** of cross-engine parity (runs every available engine
side-by-side, prints neighbor tables + classification):

```sh
veks pipeline verify engine-parity --synthetic \
  --dim 32 --base-count 500 --query-count 20 --neighbors 5
```

In-tree conformance suite:

```sh
cargo test -p veks-pipeline --lib pipeline::commands::compute_knn
cargo test -p veks-pipeline --features knnutils,faiss \
  --lib pipeline::commands::compute_knn
cargo test -p veks-pipeline --features knnutils \
  --lib pipeline::commands::verify_dataset_knnutils
```

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
