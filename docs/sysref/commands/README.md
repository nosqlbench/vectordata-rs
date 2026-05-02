# Command Reference

Every command with a working example run against the
[synthetic-1k](../../veks/tests/fixtures/synthetic-1k/) test fixture.

## Analyze

| Command | Description |
|---------|-------------|
| [analyze describe](./analyze-describe.md) | File format, dimensions, record count, structure |
| [analyze stats](./analyze-stats.md) | Per-dimension statistics |
| [analyze select](./analyze-select.md) | Display specific records by ordinal |
| [analyze find-zeros](./analyze-find-zeros.md) | Scan for near-zero vectors |
| [analyze find-duplicates](./analyze-find-duplicates.md) | Scan for duplicate vectors |
| [analyze explain-predicates](./analyze-explain-predicates.md) | Trace predicate → matching metadata |
| [analyze explain-filtered-knn](./analyze-explain-filtered-knn.md) | Full query trace through all pipeline stages |
| [analyze file](./analyze-file.md) | Low-level file metadata |
| [analyze explain-partitions](./analyze-explain-partitions.md) | Trace query through partition oracle creation |
| [analyze check-endian](./analyze-check-endian.md) | Verify byte order |

## Compute

| Command | Description |
|---------|-------------|
| [compute knn](./compute-knn.md) | Brute-force exact KNN ground truth |
| [compute knn-distances](./compute-knn-distances.md) | Recover `neighbor_distances.fvecs` from existing indices + base + query (used when the source ships indices but no distances) |
| [compute filtered-knn](./compute-filtered-knn.md) | KNN with predicate pre-filtering |
| [compute evaluate-predicates](./compute-evaluate-predicates.md) | Evaluate predicates against metadata |
| [compute partition-profiles](./compute-partition-profiles.md) | Per-label partition profiles with partitioned KNN |

## Generate

| Command | Description |
|---------|-------------|
| [generate vectors](./generate-vectors.md) | Random vector generation |
| [generate metadata](./generate-metadata.md) | Random integer metadata labels |
| [generate predicates](./generate-predicates.md) | Random equality predicates |
| [generate vvec-index](./generate-vvec-index.md) | Build offset indices for vvec files |

## Verify

| Command | Description |
|---------|-------------|
| [verify knn-consolidated](./verify-knn-consolidated.md) | Multi-threaded KNN verification |
| [verify predicates-sqlite](./verify-predicates-sqlite.md) | SQLite oracle verification |

## Merkle

| Command | Description |
|---------|-------------|
| [merkle summary](./merkle-summary.md) | Merkle tree file summary |
| [merkle create](./merkle-create.md) | Generate merkle hash trees |

## Pipeline Control

| Command | Description |
|---------|-------------|
| [state set](./state-set.md) | Set pipeline variable |
| [veks check](./veks-check.md) | Preflight checks |
| [veks run](./veks-run.md) | Execute pipeline |
