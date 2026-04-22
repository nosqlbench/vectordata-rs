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

### Strata prompts

When the dataset is large enough to warrant sub-size profiles, the
wizard offers a choice of generator strategies (`mul:`, `fib:`,
`linear:` every-10M, and `decade`). The user can accept any
combination; the resulting strings are written to the root-level
`strata:` block (see §7.7 and §1.7) and expanded into concrete
sized-profile entries the first time the dataset is loaded or
published.

- The `linear:` strategy is only offered when `base_count ≥ 10M`
  (below that, every-10M would yield zero or one profile).
- The `decade` strategy is only offered when `base_count ≥ 1M`
  (the default-start of 100k makes no meaningful sweep below that).

### Zero / duplicate count assertions

The wizard normally schedules `scan-zeros` and `scan-duplicates` to
run as part of the pipeline. If the user declines removal — for a
dataset they already trust — the wizard then asks for the *known*
zero-count and duplicate-count to assert against. Asserted values
are written to `variables.yaml` and the matching scan steps are
skipped, keeping pipeline output complete (the published
`is_zero_vector_free` / `is_duplicate_vector_free` attributes still
get set) without re-reading the data.

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
| Self-search | `Q ∈ required ∧ Q ∉ provided` (see §7.4.1) |
| Shuffle | `seed != 0` — *user-supplied no never silently overridden* (see §7.4.1) |
| Normalize | explicit opt-in (disabled by default with pre-computed GT) |
| Dedup | `!no_dedup` (default: on) |
| Filtered KNN | metadata present and `!no_filtered` |
| Scan steps | emitted when `prepare-vectors` is skipped (Identity base) |
| Zero removal | zeros detected and excluded from ordinals during sort/dedup |

### 7.4.1 Self-search and shuffle

Self-search is **defined by the input set, not by detection fallback**. Two
facet sets matter:

- **provided**: facets the user brought as inputs (what's on disk, what's
  passed via `--base-vectors` / `--query-vectors`, what `detect_roles`
  identified, what the user confirmed in the wizard).
- **required**: facets the pipeline must produce (from `--required-facets`
  or the wizard's facet confirmation step).

Self-search applies iff `Q ∈ required ∧ Q ∉ provided`. In that case the
pipeline must derive queries from base via shuffle + extract.

If `Q ∈ provided` — regardless of whether detection found it, the user
typed the path, or it came in via CLI — **self-search is categorically
excluded**. The wizard must not pick it, the import generator must not
emit `extract-queries`, and the shuffle is not forced.

The wizard always asks "Shuffle base vectors?" and the user's answer
drives `seed` (`No → seed=0 → shuffle disabled`). The user's answer is
**never silently overridden**:

- If the chosen mode does not require shuffle, the answer is honored
  directly: `seed=0` produces no `generate-shuffle` step.
- If the chosen mode is self-search and the user answered No, the
  configuration is internally contradictory (self-search needs shuffle
  to pick the train/test split). The import generator fails fast with
  an error directing the user to either provide a separate query file
  or enable shuffle. It does **not** silently materialize a shuffle the
  user said they did not want.

This rule applies more broadly: anywhere a user-supplied flag could
collide with a mode-implied requirement, the pipeline fails fast rather
than overriding the user.

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

`DatasetConfig.save()` writes `dataset.yaml` with canonical field ordering: `name` → `description` → `attributes` → `upstream` → `strata` → `profiles` → `variables`. It is round-trip safe — load, modify, save without format loss. Pipeline commands like `update_dataset_attributes` and `compute partition-profiles` use `DatasetConfig.save()` instead of text surgery, ensuring consistent formatting.

`update_dataset_attributes` is the *single* writer of `dataset.yaml` after pipeline runs — earlier commands return their changes via the `DatasetConfig` model rather than touching the file directly. This avoids the multi-writer races that were causing "fixes don't stick" issues where `generate-dataset-json` recorded the file as an output and then `update_dataset_attributes` overwrote it with different content.

Sized profiles live under the root-level `strata:` block as generator strings (`decade`, `mul:1m..16m/2`, `fib:1m`, etc.); see §1.7 for the full list of strategies. Both the unexpanded `strata:` and the resulting per-size profile entries are persisted, so consumers that don't run the loader still see the concrete profile list.

```yaml
# Strata generators at the root, expanded per-size profiles under profiles:
strata:
  - "mul:1m..16m/2"
profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvec
  1m:
    base_count: 1000000
    base_vectors: "profiles/base/base_vectors.fvec[0..1000000)"
  2m:
    base_count: 2000000
    base_vectors: "profiles/base/base_vectors.fvec[0..2000000)"
  # … 4m, 8m, 16m
```

The `path[lo..hi)` suffix on per-vector shared facets (e.g. `base_vectors`) is the canonical sub-ordinal window notation. Sized profiles inherit `base_vectors` from `default` and clip it to their first `base_count` rows; the `vectordata` reader honors the suffix on both `Detailed{source,window}` and `Simple("path[0..N)")` forms, so a consumer reading a `1m` profile sees exactly 1,000,000 rows even though the underlying file is the full base.
