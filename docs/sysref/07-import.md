# 7. Dataset Import

---

## 7.1 Bootstrap Wizard

`veks bootstrap -i` launches an interactive wizard that:

1. Scans the working directory for data files
2. Auto-detects file roles (base, query, GT, metadata) from filenames
3. Prompts for confirmation and configuration
4. Generates a complete `dataset.yaml` with the pipeline DAG
5. Creates `profiles/base/` symlinks for source files
6. Offers to underscore-prefix source files (excludes from publishing)
7. Offers to underscore-prefix unassigned files

### Auto mode

```bash
veks bootstrap -y       # accept all defaults
veks bootstrap --auto   # strict mode: all files must have recognizable roles
```

---

## 7.2 Source File Detection

Files are assigned roles by keyword matching on the filename:

| Role | Keywords |
|------|----------|
| Base vectors | `base`, `train` |
| Query vectors | `query`, `queries`, `test` |
| Neighbor indices (GT) | `groundtruth`, `gt`, `indices`, `neighbors` |
| Neighbor distances | `distances` |
| Metadata content | `metadata`, `content` |

Files with a leading `_` are still detected (the prefix is stripped
before matching). Unrecognized files are listed as "unassigned" and
offered underscore-prefix renaming.

---

## 7.3 Artifact Resolution

Each data facet resolves to one of two artifact types:

- **Identity** — source file used as-is. A symlink is created in
  `profiles/base/` pointing to the source. No pipeline step needed.
- **Materialized** — a pipeline step produces this artifact. The step
  writes to a path under `profiles/` or `.cache/`.

### When Identity is used

| Facet | Identity when |
|-------|--------------|
| Base vectors | no dedup, no normalize, no shuffle |
| Query vectors | no format conversion, no overlap removal |
| Ground truth | pre-computed GT provided |

### The O facet (oracle partitions)

The O facet is never auto-inferred — it must be explicitly requested.
It carries **sub-facets** that control what is computed within each
partition:

| Spec | Main facets | Per-partition facets |
|------|-------------|---------------------|
| `BQGDMPRFO` | BQGDMPRF | BQG (default) |
| `BQGDMPRFObqg` | BQGDMPRF | BQG (explicit) |
| `BQGDMPRFObqgd` | BQGDMPRF | BQGD (with distances) |
| `BQGDMPRFObqgmprf` | BQGDMPRF | BQGMPRF (full) |
| `+O` (wizard) | inferred | BQG (default) |

Characters after `O` (upper or lower case) specify which facets to
compute within each partition. When no sub-facets follow `O`, the
default is `bqg` — partitioned base vectors, shared queries, and
KNN per partition.

```
veks bootstrap -i                                  # wizard offers +O
veks bootstrap --required-facets BQGDMPRFO         # O with default bqg
veks bootstrap --required-facets BQGDMPRFObqgmprf  # full facets per partition
```

In the wizard, users add O to the inferred facets with `+` syntax:

```
Facets to include in dataset (* for all, +X to add) [BQGMPRF]: +O
  → Facets: BQGMPRFO  (+O added to BQGMPRF, partitions get BQG)
```

The `partition-profiles` step reads metadata labels (scalar or slab
format), identifies unique label values, extracts base vectors per
label, and registers a partition profile in `dataset.yaml`. Pipeline
Phase 3 (see §4.4) then re-expands `per_profile` templates for each
partition — but only for the sub-facets specified after `O`.

The O facet requires metadata content (M) to be available — it reads
the label values to determine partition membership. It does NOT require
filtered KNN (F) on the main dataset.

### Profile view paths

Profile views always use canonical paths regardless of artifact type:

```yaml
# Identity: symlink at profiles/base/base_vectors.fvec → ../../source.fvecs
base_vectors: profiles/base/base_vectors.fvec

# Materialized: step output
neighbor_indices: profiles/default/neighbor_indices.ivec
```

---

## 7.4 Pipeline Generation

The wizard generates a DAG of steps based on the resolved artifacts:

### Minimal pipeline (B only)

```
count-vectors → generate-vvec-index → generate-merkle → generate-catalog
```

### Self-search (B, no Q)

```
count-vectors → generate-shuffle → extract-queries → extract-base
→ compute-knn → verify-knn → ... → generate-merkle → generate-catalog
```

### Full predicated (B+Q+GT, synthesize metadata)

```
count-vectors → scan-zeros → scan-duplicates
             → generate-metadata → generate-predicates
             → evaluate-predicates → verify-predicates-sqlite
             → compute-filtered-knn → verify-filtered-knn
             → generate-dataset-json → generate-variables-json
             → generate-dataset-log-jsonl → generate-vvec-index
             → generate-merkle → generate-catalog
```

### Key decisions

| Decision | Condition |
|----------|-----------|
| Shuffle | `seed != 0` (disabled for pre-computed GT) |
| Normalize | explicit opt-in (disabled by default with pre-computed GT) |
| Dedup | `!no_dedup` (default: on) |
| Filtered KNN | metadata present and `!no_filtered` |
| Scan steps | emitted when `prepare-vectors` is skipped (Identity base) |
| Zero removal | zeros detected and excluded from ordinals during sort/dedup |

---

## 7.5 --clean Behavior

`veks run dataset.yaml --clean` resets the pipeline:

1. Removes progress log
2. Removes `.cache/` directory
3. Removes `.scratch/` directory
4. Cleans `profiles/` — removes generated files but **preserves
   Identity symlinks** (source data references)
5. Removes generated log/variable/json files

Identity symlinks survive `--clean` so subsequent runs can find the
source data without re-bootstrapping.

---

## 7.6 Zero Vector Handling

Zero vectors (all-component-zero) are detected and removed from the ordinal set during the sort/dedup step. The `clean_count` variable excludes both duplicates and zeros, ensuring ordinal congruency between base vectors and any metadata extractions (predicates, labels). This prevents downstream KNN and filtered-KNN computations from operating on degenerate vectors that would produce undefined distance results.

---

## 7.7 DatasetConfig Round-Trip Serialization

`DatasetConfig.save()` writes `dataset.yaml` with canonical field ordering: `name` → `description` → `attributes` → `upstream` → `profiles` → `variables`. It preserves `sized:` compact syntax (e.g., `sized: 100K`) and is round-trip safe — load, modify, save without format loss. Pipeline commands like `update_dataset_attributes` and `compute partition-profiles` use `DatasetConfig.save()` instead of text surgery, ensuring consistent formatting.

```yaml
# Compact sized syntax preserved on round-trip
profiles:
  100K:
    sized: 100K
    base_count: 100000
```
