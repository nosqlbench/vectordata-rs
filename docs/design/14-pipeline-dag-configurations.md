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
| 19. Pre-computed GT (separate B+Q) | fvec | fvec | - | no | no | - | 100% | gt_ivec, no_shuffle | No shuffle, no dedup, identity B+Q |
| 20. Simple-int-eq synthesis | fvec | fvec | synth(u8) | no | no | yes | 100% | gt_ivec, no_shuffle | Full predicate chain with SQLite oracle |

---

## 14.3a Facet Inference and Pipeline Gating

The pipeline DAG shape is determined by which facets are active. Facets
are resolved in a two-stage process:

1. **Inference**: `resolve_facets()` evaluates preconditions to produce a
   default facet set from the available inputs.
2. **Override**: The user may accept or modify the inferred set via the
   wizard prompt or `--required-facets`.

The inferred set gates all downstream DAG decisions. This section
specifies the inference rules as the authoritative reference.

### Facet Inference Precondition Table

Each facet is evaluated in order. A facet is inferred as **required** if
its precondition is met. The user can always override this by specifying
`--required-facets` or editing the wizard prompt.

| Facet | Code | Precondition for inference | Default | Rationale |
|-------|------|---------------------------|---------|-----------|
| Base vectors | B | `base_vectors` input provided | required if any input exists | Foundation — everything depends on B |
| Query vectors | Q | B is required | always if B | Self-search or separate queries; Q is always needed for KNN |
| KNN indices | G | B+Q are required | always if B | Ground truth is always produced (computed or provided) |
| KNN distances | D | `ground_truth_distances` file explicitly provided | only if provided | Computing D requires full KNN pass; expensive and unnecessary unless distances file is given |
| Metadata | M | `metadata` file provided OR G is provided (pre-computed GT implies full dataset) | if M provided or BQG all present | When BQG are all provided, M can be synthesized |
| Predicates | P | M is required | if M | P depends on M (predicates test metadata fields) |
| Predicate results | R | P is required | if P | R depends on P (evaluation of predicates against metadata) |
| Filtered KNN | F | R is required AND `no_filtered` is false | if R and not no_filtered | F depends on R (KNN restricted to matching ordinals) |

### Evaluation Order and Cascading

Facets are evaluated in `BQGDMPRF` order. Each facet's precondition
references only earlier facets or input files:

```
B → provided(base_vectors)
Q → required(B)
G → required(B) ∧ required(Q)
D → provided(ground_truth_distances)
M → provided(metadata) ∨ required(G)
P → required(M)
R → required(P)
F → required(R) ∧ ¬no_filtered
```

### D Facet: Why Not Inferred by Default

D (KNN ground-truth distances) is deliberately NOT inferred from B+Q+G.
Computing distances requires a full brute-force KNN pass against all base
and query vectors — the most expensive operation in the pipeline. When
the user provides pre-computed GT indices (G), they typically don't need
distances unless they're doing:

- Stratified dataset profiling (distance distributions across scales)
- Distance-based quality metrics (e.g., average distance to k-th neighbor)
- Dataset difficulty analysis (intrinsic dimensionality estimation)

If D is needed, the user explicitly provides a distances file or adds D
to the required facets. The pipeline then either uses the provided file
(Identity) or computes distances alongside KNN (Materialized).

### M Facet: Synthesis Gate

When M is required but no metadata file is provided, the wizard offers
synthesis. The synthesis mode determines:

- **simple-int-eq**: Metadata and predicates are flat packed integers
  (u8, ivec, etc.). No survey step. SQLite oracle verification.
- **slab** (future: conjugate): MNode/PNode structured records.
  Survey step discovers schema. Consolidated verification.

The synthesis decision happens AFTER facet inference but BEFORE pipeline
emission. It does not change which facets are active — it changes HOW
the M/P/R/F chain is implemented.

### Input Combination → Inferred Facets (Quick Reference)

| Inputs detected | Inferred facets | Notes |
|----------------|-----------------|-------|
| B only | BQG | Minimal: self-search KNN |
| B + metadata | BQGMPRF | Full predicate chain |
| B + Q + G | BQGMPRF | MPRF included (M synthesizable), D omitted |
| B + Q + G + D | BQGDMPRF | D included (distances file provided) |
| B + Q (no G) | BQG | G computed, no pre-computed GT |
| (no B) | (empty) | No pipeline |

### Shuffle Rule

Shuffle is disabled (seed=0) by default when B, Q, and G are all
provided. Pre-computed GT was computed against a specific vector ordering;
shuffling would invalidate the GT ordinals.

### Normalization Rule

Normalization defaults to OFF when pre-computed GT is provided.
Normalizing vectors changes distances, which invalidates GT neighbors.
The wizard warns: "normalizing would change distances and invalidate the
GT neighbors."

### Simple-Int-Eq Synthesis Mode

When metadata is required but not provided, the wizard offers synthesis.
In simple-int-eq mode:

- Metadata (M): flat packed scalar values (u8, ivec, etc.)
- Predicates (P): flat packed scalar values (same format as M)
- Results (R): ivec of matching ordinals per predicate
- No survey step (schema is known from config)
- Verification via SQLite oracle (verify-predicates-sqlite)

### Scalar Format Schema Convention

When metadata (M) and predicates (P) use scalar or ivec formats (not
slab), the schema is inferred from the format and the `fields` parameter.
This convention is shared by all commands in the predicate chain and
the SQLite oracle.

#### Field naming

Fields are named `field_0`, `field_1`, ..., `field_{N-1}` where N is the
`fields` count. This naming is used:

- By `generate metadata` when writing slab-format MNode records
- By `verify-predicates-sqlite` when creating the SQLite table schema
- By the SQLite oracle when constructing SQL WHERE clauses

For scalar and ivec formats, the field names are positional — the first
value in each record is `field_0`, the second is `field_1`, etc.

#### Format interpretation

| Format | Element | Record layout | As M (metadata) | As P (predicate) |
|--------|---------|---------------|-----------------|-----------------|
| `.u8` | uint8 | `fields` × 1 byte, flat packed | `field_i` is the i-th byte, unsigned | `field_i == value` (equality) |
| `.i8` | int8 | `fields` × 1 byte, flat packed | `field_i` is the i-th byte, signed | `field_i == value` |
| `.u16` | uint16 | `fields` × 2 bytes LE, flat packed | `field_i` is the i-th u16 | `field_i == value` |
| `.i16` | int16 | `fields` × 2 bytes LE, flat packed | `field_i` is the i-th i16 | `field_i == value` |
| `.u32` | uint32 | `fields` × 4 bytes LE, flat packed | `field_i` is the i-th u32 | `field_i == value` |
| `.i32` | int32 | `fields` × 4 bytes LE, flat packed | `field_i` is the i-th i32 | `field_i == value` |
| `.u64` | uint64 | `fields` × 8 bytes LE, flat packed | `field_i` is the i-th u64 | `field_i == value` |
| `.i64` | int64 | `fields` × 8 bytes LE, flat packed | `field_i` is the i-th i64 | `field_i == value` |
| `.ivec` | int32 | `[dim:i32, val0:i32, ..., val_{dim-1}:i32]` | `field_i` is i-th i32 after dim header | `field_i == value` |
| `.slab` | MNode | Self-describing typed fields | Named fields from MNode schema | PNode predicate tree (arbitrary ops) |

#### Predicate semantics

For scalar and ivec formats, predicates are always **equality**:

- **1 field**: `field_0 == value` (single match criterion)
- **N fields**: `field_0 == v0 AND field_1 == v1 AND ... AND field_{N-1} == v_{N-1}` (conjunction)

This is the "simple-int-eq" mode. There is no support for range
predicates, IN-lists, or OR-disjunction in scalar formats. Those
require slab format with PNode predicate trees.

#### Record count inference

| Format | Record count | How determined |
|--------|-------------|----------------|
| Scalar (u8, i16, etc.) | `file_size / (fields × element_size)` | No header; purely from file size |
| ivec | Count of `[dim, data...]` records | Walk dim headers sequentially |
| slab | `SlabReader::total_records()` | Slab page index |

#### Results (R) format

Predicate results are **always ivec**, regardless of the M/P format.
Each record in R is a variable-length list of base vector ordinals that
match the corresponding predicate:

```
R[i] = [dim:i32, ord0:i32, ord1:i32, ..., ord_{dim-1}:i32]
```

Where `dim` is the number of matching ordinals and each `ord_j` is a
0-based index into the base vector file. This is inherently
variable-length (different predicates match different numbers of
records), so a flat scalar format cannot represent it.

#### SQLite schema correspondence

The SQLite oracle creates a table that directly mirrors the scalar schema:

```sql
CREATE TABLE metadata (
    ordinal INTEGER PRIMARY KEY,
    field_0 INTEGER,
    field_1 INTEGER,
    ...
    field_{N-1} INTEGER
);
CREATE INDEX idx_field_0 ON metadata (field_0);
CREATE INDEX idx_field_1 ON metadata (field_1);
...
```

Each metadata record's ordinal is the primary key. Predicate verification
executes:

```sql
SELECT ordinal FROM metadata
WHERE field_0 = ? AND field_1 = ? ...
ORDER BY ordinal
```

The resulting ordinal list is compared byte-for-byte against R[i]. An
exact match confirms the evaluate-predicates command produced the correct
result using a completely independent code path (SQL execution vs.
integer comparison loop).

### Predicate Chain Dependency Order

```
generate-metadata (M)
  → generate-predicates (P)
    → evaluate-predicates (R)
      → verify-predicates-sqlite (SQLite oracle: M×P=R)
        → compute-filtered-knn (F: KNN restricted to R)
          → verify-filtered-knn (brute-force recomputation of F)
```

Key ordering invariant: `compute-filtered-knn` depends on
`verify-predicates-sqlite` (not just `evaluate-predicates`). R must be
verified correct before using it to filter KNN. If the predicate
evaluation is wrong, filtered KNN results would be silently corrupted.

### Verify-Predicates Step Variants

| Step ID | Command | When used | What it does |
|---------|---------|-----------|-------------|
| verify-predicates-sqlite | verify predicates-sqlite | simple-int-eq | Load M+P+R into SQLite, independently evaluate every predicate via SQL, compare against R |
| verify-predicates | verify predicates-consolidated | slab mode | Load metadata+predicates from slab, translate PNode→SQL, sample-verify against stored indices |
| verify-filtered-knn | verify filtered-knn-consolidated | always with F | Brute-force recompute filtered KNN for sampled queries, compare against stored F |

### Step Presence Rules

| Condition | verify-predicates | verify-predicates-sqlite | compute-filtered-knn | verify-filtered-knn |
|-----------|------------------|-------------------------|---------------------|-------------------|
| M present, slab mode | yes | no | if F facet | if F facet |
| M present, simple-int-eq | no | yes | if F facet | if F facet |
| M absent | no | no | no | no |
| no_filtered=true | still present (verifies R) | still present | no | no |

---

## 14.3b Verification Exemplars

Each verification step must produce output that demonstrates real work —
not just "pass" or "checked". This section shows what correct
verification output looks like, annotated to explain what each line proves.

These exemplars also define the log output contract: the pipeline log
must include these details so that post-hoc audits can confirm
verification actually occurred.

### Exemplar 1: verify-knn (KNN Ground Truth Verification)

**What it does**: For a sample of queries, brute-force scan ALL base
vectors, compute exact distances, find true top-k, compare against
stored GT.

**Console output (pass)**:
```
verify-knn-consolidated: 1 profiles, 100 sample queries, metric=L2, threads=128
  loaded GT for 1 profiles, 1 sample queries, k=100
  scanning 1000000 base vectors (1 sample queries, 128 threads)
  profile 'default' (base_count=1000000): 1/1 pass, 0 fail, recall@100=1.0000
```

**What each line proves**:
- `100 sample queries` — verification tested 100 queries, not just 1
- `scanning 1000000 base vectors` — brute-force computed distances to
  ALL base vectors, not a shortcut
- `recall@100=1.0000` — every stored GT neighbor is confirmed correct
  by independent computation

**Console output (fail)**:
```
  MISMATCH query 0 (sample #0): 2 of 100 neighbors differ
    boundary distance: 0.32305700, 1 neighbors at this distance
      idx=989762 dist=0.32305700 in GT
    computed-only: idx=670103 dist=0.3229170144 (rank 97)
    computed-only: idx=929750 dist=0.3230350018 (rank 98)
    expected-only: idx=63349 (GT rank 96)
    expected-only: idx=398306 (GT rank 97)
  profile 'default': 0/1 pass, 1 fail, recall@100=0.0000
```

**What a failure proves**:
- `boundary distance: 0.32305700` — shows the exact distance where
  tie-breaking differs, proving this is a real distance computation
- `computed-only` / `expected-only` — lists the specific ordinals that
  differ, with their computed distances and ranks
- This level of detail lets the user diagnose whether the mismatch is
  a real error or a floating-point tie-break at the boundary

### Exemplar 2: verify-predicates-sqlite (SQLite Oracle)

**What it does**: Loads ALL metadata values and predicate values into
an in-memory SQLite database. For each predicate, constructs and executes
a SQL query independently of the pipeline evaluation code, then compares
the SQL results against the stored predicate results (R).

**Console output (pass)**:
```
  verify predicates-sqlite: metadata=profiles/base/metadata_content.u8
    predicates=profiles/base/predicates.u8 results=profiles/default/metadata_indices.ivec
  loaded 1000000 metadata records into SQLite (1 fields)
  loaded 10000 predicates
  loaded 10000 result records
  10000 pass, 0 fail (10000 checked of 10000)
```

**What each line proves**:
- `loaded 1000000 metadata records into SQLite` — every metadata record
  was inserted into a real SQL table, not skipped or sampled
- `loaded 10000 predicates` — every predicate was loaded
- `10000 checked of 10000` — every single predicate was verified, not
  a sample. The verification is exhaustive.
- The SQLite path shares ZERO code with the evaluate-predicates command.
  It uses `rusqlite` SQL execution, not the pipeline's integer matching
  loop. If both agree, the result is independently confirmed.

**Console output (fail)**:
```
  MISMATCH: predicate 42 (field_0 = 7): expected 91205 matches, got 91100
  MISMATCH: predicate 1337 (field_0 = 3): expected 90800 matches, got 0
  9998 pass, 2 fail (10000 checked of 10000)
```

**What a failure proves**:
- `predicate 42 (field_0 = 7)` — shows the exact predicate expression,
  so the user can manually verify: "how many records have field_0 = 7?"
- `expected 91205 matches, got 91100` — shows the magnitude of the
  discrepancy. 105 missing matches suggests a boundary or range error,
  not a total failure.
- `got 0` — a total mismatch indicates the evaluate step is reading the
  wrong file or using the wrong field index.

**Why SQLite is the right oracle**: SQLite's query execution is a
completely independent code path from the pipeline's integer comparison
loop. SQLite parses SQL, builds a query plan, scans the index, and
returns rows. The pipeline reads flat-packed bytes and compares integers.
If both produce the same ordinal sets, the result is correct by
construction — no shared code, no shared bugs.

### Exemplar 3: verify-filtered-knn (Filtered KNN Verification)

**What it does**: For sampled queries, independently computes filtered
KNN by brute-force: get the matching ordinals from R, compute distances
only to those base vectors, find top-k, compare against stored filtered
GT (F).

**Console output (pass)**:
```
  verify-filtered-knn-consolidated: 1 profiles, 50 sample queries,
    1000000 base, 10000 queries
  verify filtered-knn 'default': 50/50 pass (k=100)
  profile 'default' (base_count=1000000): 50 pass, 0 fail
```

**What each line proves**:
- `50 sample queries` — tested a meaningful subset, not 1
- `50/50 pass` — for every sampled query, the stored filtered GT exactly
  matches the brute-force recomputation

**Console output (fail)**:
```
  profile 'default' (base_count=1000000): 48 pass, 2 fail
```

**What a failure means**: The stored filtered KNN result for 2 queries
does not match the brute-force recomputation. This could indicate:
- The predicate evaluation (R) was wrong (some ordinals missing/extra)
- The filtered KNN search had a bug (wrong distance computation)
- Floating-point tie-breaking at the k-th boundary (allowed up to 99%
  recall before flagging as failure)

### Log Output Contract

All verification steps MUST include in their pipeline log output:

1. **Input counts**: How many records were loaded (metadata, predicates,
   base vectors, queries)
2. **Sample size**: How many items were verified (all, or N of M sampled)
3. **Method**: What independent computation was performed (brute-force
   scan, SQL query, etc.)
4. **Per-item results**: Pass/fail counts with concrete numbers
5. **Failure details**: On failure, the specific item (query index,
   predicate expression, ordinal) and the nature of the discrepancy
   (expected vs got, with values)

A verification step that outputs only "verified" or "pass" without these
details is considered a **stub** and must be replaced with a real
implementation before the pipeline configuration is considered complete.

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

### Configuration 24: Pre-computed GT with Separate B+Q (No Shuffle)

**Scenario**: Separate base + query fvec files with pre-computed GT ivec.
No shuffle, no dedup, no normalization. Minimal pipeline.

**Facets**: BQG (no D — would require re-computing KNN)

**DAG**:
```
count-vectors
  → verify-knn (pre-computed GT, not per-profile)
    → generate-dataset-json → generate-merkle → generate-catalog
```

**Key properties**:
- NO shuffle, extract-queries, extract-base, count-base steps
- Base and query vectors are Identity (direct file references)
- `verify-knn` references provided GT file, not per-profile
- `combined_bq = false` (pre-computed GT means queries are independent)
- `is_shuffled = false`, `self_search = false`

---

### Configuration 25: Simple-Int-Eq Predicate Synthesis (Full Chain)

**Scenario**: Separate B+Q with pre-computed GT, plus synthesized
simple-integer-equality metadata and predicates in u8 format.

**Facets**: BQGMPRF (D omitted — GT provided, no re-computation)

**DAG**:
```
count-vectors
  ├→ generate-metadata (M: u8 flat packed)
  │   → generate-predicates (P: u8 flat packed, mode=simple-int-eq)
  │     → evaluate-predicates (R: ivec matching ordinals)
  │       → verify-predicates-sqlite (SQLite oracle: M×P=R)
  │         → compute-filtered-knn (F: KNN restricted to R)
  │           → verify-filtered-knn (brute-force recomputation)
  └→ verify-knn (independent: verifies G)
→ generate-dataset-json → generate-merkle → generate-catalog
```

**Key properties**:
- No survey-metadata step (schema known from config)
- Metadata generates directly to `profiles/base/metadata_content.u8`
- No extract-metadata step (no shuffle = no reordering needed)
- `evaluate-predicates` in mode=simple-int-eq reads u8 files
- `verify-predicates-sqlite` loads M+P+R into SQLite, independently
  evaluates every predicate via SQL `WHERE field_0 = X`, compares R
- `compute-filtered-knn` reads ivec predicate indices via PredicateIndices
- `verify-filtered-knn` brute-force recomputes filtered KNN for sampled
  queries against eligible base vectors

**Predicate chain ordering invariant**: `compute-filtered-knn` depends on
`verify-predicates-sqlite`, not on `evaluate-predicates` directly. R must
be verified correct before using it as a KNN filter.

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
