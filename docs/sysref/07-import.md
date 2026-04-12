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
