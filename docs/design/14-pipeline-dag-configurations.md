# 14 Pipeline DAG Configurations

The `prepare bootstrap` command generates a `dataset.yaml` containing a
pipeline of steps whose shape depends on the inputs provided. This document
describes the DAG configurations that arise from different input
combinations, and serves as the specification for the integration tests in
`veks/tests/dag_configurations.rs`.

---

## 14.1 Relationship to SRD 12

This document is the **test specification companion** to
[12-dataset-import-flowchart.md](12-dataset-import-flowchart.md), which
defines the authoritative slot model, resolution rules, and pipeline
emission algorithm. This document enumerates the **concrete DAG shapes**
that arise from different input combinations, provides ASCII diagrams for
each, and specifies the integration tests in `veks/tests/dag_configurations.rs`.

For the slot model, identity collapse rules, and ordinal congruency
invariants, see SRD §12.1–12.3. For variable reference rules (when to
use `${clean_count}` vs `${vector_count}` vs `${base_end}`), see
SRD §12.1 ordinal congruency.

The configurations below are parameterized by the facet code system
defined in SRD §2.8 (`BQGDMPRF`). Each configuration shows which
facets are active and which pipeline steps result.

---

## 14.2 Configuration Matrix

| Config | base | query | meta | dedup | zeros | filtered | fraction | extras | Key feature |
|--------|------|-------|------|-------|-------|----------|----------|--------|-------------|
| 1. Minimal | fvec | self | - | yes | yes | - | 100% | - | Smallest self-search |
| 2. No cleaning | fvec | self | - | no | no | - | 100% | - | Skip dedup+zeros |
| 3. Separate query | fvec | fvec | - | yes | yes | - | 100% | - | No shuffle |
| 4. With metadata | fvec | self | parquet | yes | yes | yes | 100% | - | Full predicate chain |
| 5. Base fraction | fvec | self | - | yes | yes | - | 50% | - | Capped base set |
| 6. No dedup + meta | fvec | self | parquet | no | yes | yes | 100% | - | Zero-only cleaning |
| 7. Meta no filtered | fvec | self | parquet | yes | yes | no | 100% | - | Skip filtered KNN |
| 8. Foreign base | npy | self | - | yes | yes | - | 100% | - | convert-vectors step |
| 9. Foreign queries | fvec | npy | - | yes | yes | - | 100% | - | convert-queries step |
| 10. Pre-computed GT | fvec | fvec | - | yes | yes | - | 100% | gt_ivec | Skip compute-knn |
| 11. Normalize | fvec | self | - | yes | yes | - | 100% | normalize | Extract normalization |
| 12. Fraction + meta | fvec | self | parquet | yes | yes | yes | 50% | - | base_end in all ranges |
| 13. Convert format | fvec | self | - | yes | yes | - | 100% | mvec convert | Precision conversion |
| 14. Sized profiles | fvec | self | - | yes | yes | - | 100% | sized spec | Multi-scale YAML |
| 15. Native metadata | fvec | self | slab | yes | yes | yes | 100% | - | Identity metadata |
| 16. No base vectors | - | - | - | - | - | - | - | - | Empty pipeline |
| 17. GT + distances | fvec | fvec | - | yes | yes | - | 100% | gt+dist | Both pre-computed |
| 18. Full pipeline | npy | self | parquet | yes | yes | yes | 75% | normalize,sized | Everything enabled |

---

## 14.3 DAG Diagrams

### Config 1: Minimal Self-Search

Inputs: `base_vectors.fvec` (native format, no conversion needed)

```
count-vectors
      |
sort-and-dedup --> count-duplicates
      |
find-zeros ------> count-zeros
      |
filter-ordinals --> count-clean
      |
generate-shuffle
      |
  +---+---+
  |       |
extract   extract
-queries  -base --> count-base
  |       |
  +---+---+
      |
 compute-knn [per-profile]
      |
 verify-knn [per-profile]
      |
 generate-catalog
      |
 generate-merkle
```

**Key invariants:**
- `generate-shuffle` depends on `count-clean` (not `count-vectors`)
- `shuffle.interval = ${clean_count}`
- `extract-base.range = [${query_count}, ${clean_count})`
- No `convert-vectors` (native fvec is identity)
- No metadata chain

### Config 2: No Dedup, No Zero Check

```
count-vectors
      |
generate-shuffle
      |
  +---+---+
  |       |
extract   extract
-queries  -base --> count-base
  |       |
  +---+---+
      |
 compute-knn [per-profile]
      |
 verify-knn [per-profile]
      |
 generate-catalog
      |
 generate-merkle
```

**Key invariants:**
- No `sort-and-dedup`, `find-zeros`, `filter-ordinals`
- `generate-shuffle` depends on `count-vectors` directly
- `shuffle.interval = ${vector_count}`
- `extract-base.range = [${query_count}, ${vector_count})`

### Config 3: Separate Queries

Inputs: `base_vectors.fvec` + `query_vectors.fvec`

```
count-vectors
      |
sort-and-dedup --> count-duplicates
      |
find-zeros ------> count-zeros
      |
filter-ordinals --> count-clean
      |
 compute-knn [per-profile]
      |
 verify-knn [per-profile]
      |
 generate-catalog
      |
 generate-merkle
```

**Key invariants:**
- No `generate-shuffle`, `extract-queries`, `extract-base`, `count-base`
- Query and base vectors are provided directly (identity slots)
- KNN references the provided files

### Config 4: Full Metadata Pipeline

Inputs: `base_vectors.fvec` + metadata (parquet)

```
count-vectors
      |                          convert-metadata
sort-and-dedup                        |
      |                    +----------+----------+
find-zeros                 |                     |
      |              survey-metadata        extract-metadata
filter-ordinals            |
      |              generate-predicates
generate-shuffle           |
      |              evaluate-predicates [per-profile]
  +---+---+                |
  |       |          compute-filtered-knn [per-profile]
extract   extract          |
-queries  -base      verify-predicates [per-profile]
  |       |
  +---+---+
      |
 compute-knn [per-profile]
      |
 verify-knn [per-profile]
      |
 generate-catalog  (after verify-knn AND verify-predicates)
      |
 generate-merkle
```

**Key invariants:**
- `extract-metadata` depends on `[convert-metadata, generate-shuffle]`
- `extract-metadata.range = [${query_count}, ${clean_count})`
- `evaluate-predicates` depends on `[generate-predicates, extract-metadata]`
- `compute-filtered-knn` depends on `[evaluate-predicates, count-base, extract-queries]`
- `generate-catalog` depends on `[verify-knn, verify-predicates]`

### Config 5: Base Fraction (50%)

Same as Config 1 but with a `compute-base-end` step inserted:

```
... count-clean ...
        |
  compute-base-end  (state set: base_end = scale:${clean_count}*0.5)
        |
  generate-shuffle
        |
    +---+---+
    |       |
  extract   extract
  -queries  -base
```

**Key invariants:**
- `compute-base-end` depends on `count-clean`
- `compute-base-end.value` = `scale:${clean_count}*0.5`
- `extract-queries` and `extract-base` depend on `compute-base-end`
- `extract-base.range = [${query_count}, ${base_end})`
- Metadata extraction (if present) also uses `${base_end}`

### Config 6: No Dedup + Metadata

When dedup is disabled but zero check is still on:

```
count-vectors
      |                          convert-metadata
find-zeros --> count-zeros            |
      |                    +----------+----------+
filter-ordinals            |                     |
      |              survey-metadata        extract-metadata
count-clean                |
      |              generate-predicates
generate-shuffle           |
      |              evaluate-predicates [per-profile]
  +---+---+
  ... (rest same as Config 4)
```

**Key invariants:**
- No `sort-and-dedup`
- `filter-ordinals` depends on `find-zeros` only (no sort)
- `shuffle.interval = ${clean_count}` (clean ordinals still exist from zero check)
- `extract-metadata.range = [${query_count}, ${clean_count})`

### Config 7: Metadata Without Filtered KNN

Same as Config 4 but without `compute-filtered-knn` and `verify-predicates`:

```
  ... (same vector chain as Config 4) ...

  convert-metadata --> survey-metadata --> generate-predicates
        |                                        |
  extract-metadata                      evaluate-predicates [per-profile]

  compute-knn [per-profile]
        |
  verify-knn [per-profile]
        |
  generate-catalog  (after verify-knn only — no verify-predicates)
        |
  generate-merkle
```

**Key invariants:**
- `generate-predicates` and `evaluate-predicates` still present (metadata indexing)
- No `compute-filtered-knn`
- No `verify-predicates`
- `generate-catalog` depends only on `verify-knn`

### Config 8: Foreign Base (npy)

Inputs: directory of npy files (requires conversion)

```
convert-vectors (transform convert, npy -> mvec)
      |
count-vectors
      |
... (same chain as Config 1) ...
```

**Key invariants:**
- `convert-vectors` present with `run: transform convert`
- `convert-vectors` has `from: npy` option
- All downstream steps depend on `convert-vectors`

### Config 9: Foreign Queries (npy)

Inputs: `base_vectors.fvec` + `query_vectors` as npy directory

```
  convert-queries (transform convert, npy -> mvec)
       |
  ... (separate-query chain, no shuffle) ...
```

**Key invariants:**
- `convert-queries` or equivalent import step present
- No `generate-shuffle` (separate queries)
- `compute-knn` references the converted query file

### Config 10: Pre-computed Ground Truth

Inputs: `base_vectors.fvec` + `query_vectors.fvec` + `ground_truth.ivec`

```
count-vectors
      |
sort-and-dedup --> ...
      |
filter-ordinals
      |
verify-knn (references provided GT, NOT per-profile)
      |
generate-catalog
```

**Key invariants:**
- No `compute-knn` step
- `verify-knn` present, NOT per-profile
- `verify-knn` indices option references the GT file path

### Config 11: Normalize

Same as Config 1 but `extract-queries` and `extract-base` have `normalize: true`.

**Key invariants:**
- `extract-queries` options include `normalize: true`
- `extract-base` options include `normalize: true`

### Config 12: Base Fraction + Metadata

Combines Config 5 (fraction) with Config 4 (metadata):

```
... count-clean ...
        |
  compute-base-end
        |
  generate-shuffle           convert-metadata
        |                         |
    +---+---+             +-------+-------+
    |       |             |               |
  extract   extract   survey-meta    extract-metadata
  -queries  -base         |          (range: ${base_end})
                    gen-predicates
                          |
                    eval-predicates
```

**Key invariants:**
- `extract-metadata.range = [${query_count}, ${base_end})`
- `extract-base.range = [${query_count}, ${base_end})`
- Both metadata and vector extraction use the same capped upper bound

### Config 13: Base Convert Format

Inputs: `base_vectors.fvec` + `base_convert_format: "mvec"`

```
convert-vectors (identity/symlink for native fvec)
      |
convert-base-format (transform convert, fvec -> mvec)
      |
count-vectors
      |
... (standard chain) ...
```

**Key invariants:**
- An additional conversion step appears after the base import
- The converted format is used by downstream steps

### Config 14: Sized Profiles

Same as Config 1 but YAML includes `sized:` section.

**Key invariants:**
- YAML contains `sized:` under `profiles:`
- `ranges:` contains the parsed spec
- `facets:` maps each facet with `${range}` or `${profile}` templates

### Config 15: Native Metadata (slab)

Inputs: `base_vectors.fvec` + `metadata.slab` (native, no conversion)

**Key invariants:**
- No `convert-metadata` step (identity)
- `survey-metadata` references the slab directly
- `extract-metadata` still present (self-search requires ordinal realignment)

### Config 16: No Base Vectors

No inputs at all — empty pipeline.

**Key invariants:**
- No upstream steps (or only catalog/merkle)
- `dataset.yaml` exists with `profiles:` section

### Config 17: Pre-computed GT + Distances

Inputs: `base_vectors.fvec` + `query_vectors.fvec` + `ground_truth.ivec` + `ground_truth_distances.fvec`

**Key invariants:**
- No `compute-knn`
- Both indices and distances are identity slots
- `verify-knn` references both provided files

### Config 18: Full Pipeline

Inputs: npy base + parquet metadata + self-search + fraction 75% + normalize + sized profiles

```
convert-vectors (npy -> mvec)
      |
count-vectors
      |
sort-and-dedup --> count-duplicates
      |
find-zeros --> count-zeros
      |
filter-ordinals --> count-clean
      |
compute-base-end (scale: 0.75)
      |
generate-shuffle          convert-metadata
      |                        |
  +---+---+             +------+------+
  |       |             |             |
extract   extract   survey-meta  extract-meta
-queries  -base         |        (${base_end})
  |  (norm)  |    gen-predicates
  |  (norm)  |          |
  +---+---+       eval-predicates [pp]
      |                 |
 compute-knn [pp]  filt-knn [pp]
      |                 |
 verify-knn [pp]   verify-pred [pp]
      |                 |
      +--------+--------+
               |
         generate-catalog
               |
         generate-merkle
```

**Key invariants:**
- All steps present
- `convert-vectors` (npy is foreign)
- `compute-base-end` with `*0.75`
- Extract steps have `normalize: true`
- Sized profiles in YAML
- All ranges use `${base_end}`
- Total ~25 steps

### Configuration 21: Early Stratification with Deferred Profiles

**Scenario**: Bootstrap with `--sized 'mul:1m..${base_count}/2'` — profiles
declared at bootstrap but expanded after core stages produce `base_count`.

**Input**: fvec base vectors, self-search, sized spec with variable reference

**DAG (Phase 1 — core stages)**:
```
import → count → sort → zeros → clean-ordinals → shuffle → extract-queries → extract-base → count-base → compute-knn → verify-knn
```

**DAG (Phase 2 — deferred profile stages, after base_count known)**:
Per-profile stages expand dynamically. For a 100K dataset with
`mul:1m..${base_count}/2`, if `base_count=95000`, no profiles are generated
(all would exceed the base count). For `base_count=10000000`, profiles
`1m, 2m, 4m` are generated and processed smallest-first.

**Key properties**:
- Phase 1 is identical to Configuration 1 (minimal self-search)
- Phase 2 steps are generated at runtime after `base_count` is known
- Profile expansion uses the same `expand_per_profile_steps()` as static profiles
- Cache reuse: 1m partition segments reused by 2m and 4m profiles

### Configuration 22: Base Fraction with Early Stratification

**Scenario**: `--base-fraction '5%' --sized 'mul:1m..${base_count}/2'`

**Input**: fvec base vectors, self-search, 5% fraction, sized spec

**Key property**: The fraction is applied first (during import/subset),
then cleaning runs on the subset, then `base_count` is computed from
the cleaned subset. Profiles are derived from this cleaned count.
Changing the fraction would require a full re-bootstrap.

---

## 14.4 Variable Reference Rules

See SRD §12.1 (ordinal congruency) for the authoritative specification.
Summary:

| Condition | shuffle.interval | extract range upper bound |
|-----------|------------------|--------------------------|
| dedup or zeros active | `${clean_count}` | `${clean_count}` |
| no cleaning | `${vector_count}` | `${vector_count}` |
| base_fraction < 1.0 | `${clean_count}` | `${base_end}` |

When `base_fraction < 1.0`, the `convert-vectors` step also receives a
`fraction` option so it only imports the needed subset of source data.
The `compute-base-end` step computes `base_end` using a `scale:` expression
with `:roundN` for clean dataset sizes (see `--round-digits`).

---

### Configuration 21: Early Stratification

**Scenario**: Bootstrap with `--sized '50,100'` on 200-vector source data.
Profiles are declared at bootstrap time with concrete values.

**Facets**: BQGD (no metadata)

**Key properties**:
- The `sized:` key appears in `dataset.yaml` at bootstrap time
- Core stages (import, sort, dedup, shuffle, extract, KNN) run as normal
- Per-profile steps expand dynamically at `veks run` time
- Adding more profiles later (via stratify) does not invalidate core steps

**Test**: `dag_21_early_stratification`

---

### Configuration 22: Base Fraction with Early Stratification

**Scenario**: `--base-fraction 50% --sized '20,40'`

**Facets**: BQGD (no metadata)

**Key properties**:
- The fraction is applied first via `subset-vectors`
- Cleaning runs on the subset
- `base_count` is computed from the cleaned subset
- Sized profiles operate within this reduced universe
- **Invariant**: changing the fraction invalidates all artifacts (§3.13)

**Test**: `dag_22_fraction_with_early_stratification`

---

### Configuration 23: Full Pipeline with Early Stratification

**Scenario**: All facets (BQGDMPRF) with metadata and `--sized '50,100'`

**Key properties**:
- Full facet chain including metadata conversion, predicate synthesis,
  filtered KNN
- Sized profiles get per-profile KNN and filtered KNN steps
- Metadata is shared (not per-profile); KNN and filtered KNN are per-profile

**Test**: `dag_23_full_with_early_stratification`

---

## 14.5 Adversarial Test Coverage

The following adversarial conditions are tested in
`veks/tests/e2e_pipeline.rs` to verify pipeline robustness:

| Test | Condition | Expected behavior |
|------|-----------|-------------------|
| `adversarial_idempotent_rerun` | Re-run with same config | All steps skipped (fresh) |
| `adversarial_knn_indices_in_range` | KNN output validation | All indices in `[0, base_count)` |
| `adversarial_empty_base_vectors` | 0-byte input file | Graceful error, no hang |
| `adversarial_query_count_exceeds_vectors` | `query_count > vector_count` | No panic; cap or error |
| `adversarial_k_exceeds_base_count` | `k > base_count` | No panic; cap or error |
| `adversarial_zero_fraction` | `--base-fraction 0%` | Graceful handling |
| `adversarial_sized_profile_exceeds_base` | Profile size > base count | Window clamped, no error |
| `adversarial_different_seeds_valid_knn` | Same data, different seeds | Both produce valid results |

---

## 14.6 Test Execution

Tests are in `veks/tests/dag_configurations.rs`. Each test:

1. Creates a temporary directory under `target/tmp/` (not `/tmp`)
2. Writes small synthetic data files (200 vectors, dim=4)
3. Calls `import::run()` with the appropriate `ImportArgs`
4. Parses the generated `dataset.yaml`
5. Asserts step presence/absence and dependency edges
6. Asserts variable references in range options

Run all DAG tests:
```
cargo test -p veks dag_configurations
```

Run a specific test:
```
cargo test -p veks dag_configurations::dag_with_metadata
```
