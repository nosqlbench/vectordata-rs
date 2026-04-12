# Oracle Partition Profiles — Design

## Overview

Oracle partition profiles are deferred profiles that expand after
the metadata facets (M, P, R) are produced. Each unique label value
becomes a profile with its own extracted base vectors. The existing
`per_profile` pipeline steps (compute-knn, verify-knn) then naturally
run for each partition profile.

## Pipeline phases

```
Phase 1 (core):
  generate-base → generate-queries → generate-metadata → generate-predicates
  → evaluate-predicates → verify-predicates-sqlite
  → compute-knn (default) → verify-knn (default)
  → compute-filtered-knn (default)

Phase 2 (partition expansion):
  prepare-partitions:
    1. Read metadata labels (M facet)
    2. For each viable label (count >= k):
       a. Extract matching base vectors → profiles/label-N/base_vectors.fvec
       b. Symlink query vectors → profiles/label-N/query_vectors.fvec
       c. Register profile "label-N" in dataset.yaml with paths
    3. Set variable partition_count = N

  Re-expansion:
    per_profile templates expand for label-0, label-1, ..., label-N
    → compute-knn (label-0), compute-knn (label-1), ...
    → verify-knn runs across all profiles including partitions

Phase 3 (finalize):
  generate-dataset-json → generate-vvec-index → generate-merkle → generate-catalog
```

## Key design points

1. **One step creates all partition base vectors** — `prepare-partitions`
   is NOT per_profile. It runs once, reads M, and writes all partition
   base vector files.

2. **KNN is per_profile** — existing compute-knn template expands
   naturally for each partition profile. No special KNN code needed.

3. **Deferred expansion** — partition profiles are added to dataset.yaml
   mid-pipeline. Phase 2 re-expansion picks them up.

4. **No per-partition pipeline steps** — the pipeline definition stays
   compact. Partitions are profiles, not steps.

## Profile registration

`prepare-partitions` adds profiles to `config.profiles.profiles`:

```yaml
profiles:
  label-0:
    base_vectors: profiles/label-0/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
  label-1:
    base_vectors: profiles/label-1/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
```

These are concrete profiles (not deferred) — they have fixed paths,
no `base_count`, no `${variable}` references.
