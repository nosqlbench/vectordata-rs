<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 12 — Dataset Import Flowchart

This document defines the universal pipeline generation logic for
`veks datasets import`. The pipeline is modeled as an idempotent flow
graph where every slot always exists. Slots that require no work
collapse to identity (alias to an upstream artifact), and optional data
axes prune dependent subgraphs when absent.

---

## 12.1 Design Principles

**Superset graph, identity collapse.** The generator always builds the
same graph topology. Each slot is resolved to either a *materialized
step* (a real pipeline command that produces a new artifact) or an
*identity alias* (the slot's output is bound directly to an earlier
artifact, and no step is emitted). This avoids conditional branching in
the generator; instead, each slot independently evaluates whether it
needs to do work.

**Type-safe optional axes.** Metadata and query vectors are modeled as
`Option`-typed slots. When `None`, the slot and all transitively
dependent slots are pruned from the emitted pipeline. This is not a
runtime check — it is a structural property of the graph: a `None` slot
produces no artifact, so downstream slots that require it cannot resolve
and are not emitted.

**Idempotent flow state.** Every slot is a pure function of its inputs.
Given the same user-provided files and options, the generator always
produces the same `dataset.yaml`. Re-running `datasets import` with
identical inputs is a no-op (modulo file timestamps).

**Symlink-when-possible.** When a source file is already in the target
xvec format (no conversion, no count limit, single file), the import
step creates a symlink instead of copying bytes. This eliminates
redundant I/O for large native-format files.

**Ordinal congruency.** When paired data files (vectors + metadata)
are provided, the pipeline must maintain pairwise ordinal correspondence
throughout all reordering and elision operations. Specifically:
- The same shuffle permutation (`shuffle.ivec`) is applied to both
  vector extraction and metadata extraction.
- The same clean ordinals index (`clean_ordinals.ivec`) determines which
  records survive dedup/zero elision for both vectors and metadata.
- The same range windows (`[query_count, vector_count)`) are used for
  both base vector extraction and metadata content extraction.
This ensures that after all pipeline transformations, record N in the
output base vectors file corresponds to record N in the output metadata
content file. Violating this invariant would silently corrupt filtered
KNN results, since predicate evaluation indexes metadata by the same
ordinal used to address base vectors.

---

## 12.2 Slot Model

The pipeline is a DAG of **slots**. Each slot has:

- A **name** (e.g., `all_vectors`, `base_vectors`, `query_vectors`)
- A **type** describing the artifact kind (`Vectors`, `Metadata`, `Ordinals`, `GroundTruth`)
- A **resolution**: either `Materialized(step)` or `Identity(upstream_slot)`
- **Dependencies**: other slots whose artifacts this slot consumes

```
Slot<T> = Materialized { step_id, command, options, output_path }
        | Identity { alias_to: Path }     // no step emitted
        | Absent                           // Option::None — prunes dependents
```

When a slot resolves to `Identity`, its `output_path` is set to the
same path as the upstream artifact it aliases. No pipeline step is
emitted. Downstream slots see the same path regardless of whether the
upstream was materialized or aliased.

When a slot resolves to `Absent`, it and all transitively dependent
slots are omitted from the emitted pipeline and profile views.

---

## 12.3 Slot Resolution Rules

Each slot has an identity predicate — the condition under which it
collapses to a pass-through. The following table defines every slot in
the superset graph and its resolution rule.

### Vector slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `fetch_vectors` | Vectors | No URL provided | `fetch bulkdl` or `fetch dlhf` |
| `import_vectors` | Vectors | Source is native xvec single file (symlinked) | `import` (npy/parquet/dir → xvec) |
| `all_vectors` | Vectors | *(terminal — always resolves to import or source)* | alias chain: fetch → import → source |
| `convert_base_precision` | Vectors | No precision conversion requested | `convert` (e.g., fvec→mvec or mvec→fvec) |
| `sort` | Ordinals | `--no-sort` | `compute dedup` — lexicographic external merge-sort producing sorted ordinal index + duplicate report as a byproduct. Duplicate detection retains **one representative** of each duplicate set in the sorted index; only the extra copies are recorded in the duplicates file. The sort enables binary search for zero vectors and produces the canonical ordinal ordering for all downstream steps. |
| `zero_check` | Ordinals | `--no-zero-check` | `analyze zeros` — binary search the lexicographically sorted ordinal index for the zero vector `[0,0,...,0]` |
| `clean_ordinals` | Ordinals | No sort and no zero-check | `transform clean-ordinals` — combine duplicate + zero exclusion ordinals, filter the sorted index to produce the clean ordinal set used by shuffle and extraction |

### Count and statistics slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `vector_count` | Variable | Always materialized | `set variable` — record count of all_vectors |
| `duplicate_count` | Variable | `--no-dedup` | `set variable` — record count of dedup_duplicates.ivec (number of elided duplicates) |
| `zero_count` | Variable | `--no-zero-check` | `set variable` — record count of zero_ordinals.ivec (number of zero vectors removed) |
| `clean_count` | Variable | Always needed when dedup or zero-check active | `set variable` on clean_ordinals |
| `base_count` | Variable | No self-search (base = all_vectors) | `set variable` on base_vectors |

All count/statistics variables are persisted to `variables.yaml` and
available to downstream steps via `${name}` interpolation. After a
pipeline run, `variables.yaml` contains a complete record of dataset
statistics including `vector_count`, `duplicate_count`, `zero_count`,
`clean_count`, and (if self-search) `base_count`.

### Query slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `shuffle` | Ordinals | No self-search (separate query file or no queries) | `generate ivec-shuffle` over clean_count |
| `query_vectors` | Vectors | Provided as native xvec (symlinked) | `import` or `transform *-extract` (self-search via clean_ordinals; `--normalize` option applies L2 normalization in-flight during extraction) |
| `convert_query_precision` | Vectors | No precision conversion requested | `convert` (e.g., fvec→mvec or dvec→fvec) |
| `base_vectors` | Vectors | Not self-search (all_vectors used directly) | `transform *-extract` (self-search range via clean_ordinals; `--normalize` option applies L2 normalization in-flight during extraction) |

### Metadata slots (all `Absent` when `--metadata` not provided)

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `import_metadata` | Metadata | Source is native slab | `import` (parquet/dir → slab) |
| `metadata_all` | Metadata | *(terminal)* | alias: import → source |
| `extract_metadata` | Metadata | No self-search (ordinals already aligned) | `transform slab-extract` |
| `metadata_content` | Metadata | *(terminal)* | alias: extract → metadata_all |
| `survey` | JSON | Never when metadata present | `survey` |
| `predicates` | Metadata | Never when metadata present | `synthesize predicates` |
| `predicate_indices` | Metadata | Never when metadata present | `compute predicates` (per_profile) |

### Ground truth slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `knn` | GroundTruth | Pre-computed GT provided, or no query vectors | `compute knn` (per_profile, `--compress-cache` for partition artifacts) |
| `filtered_knn` | GroundTruth | No metadata, `--no-filtered`, or no queries | `compute filtered-knn` (per_profile, `--compress-cache` for partition artifacts) |

### Verification slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `verify_knn` | Report | No KNN computed, or `--no-verify` | `verify knn` (per_profile, sparse-sample brute-force recomputation) |
| `verify_predicates` | Report | No filtered KNN, no metadata, or `--no-verify` | `verify predicates` (per_profile, SQLite-backed sparse-sample evaluation) |

### Merkle slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `merkle` | HashTree | Always materialized | `merkle create` — walks all publishable data files and produces `.mref` hash trees for integrity verification and incremental transfer |

### Catalog slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `catalog` | Index | Always materialized (final step) | `catalog generate` — produces `catalog.json` and `catalog.yaml` for the local dataset directory so the dataset is discoverable by catalog queries |

---

## 12.4 Universal Flow Graph

The full superset graph with identity collapse annotations. Slots that
resolve to `Identity` are shown with dashed borders. The graph is always
this shape; only the resolution of each slot changes.

```mermaid
flowchart TD
    %% ── Inputs (user-provided) ───────────────────────────────────
    SRC_VEC([/"vector source"/]):::input
    SRC_QUERY([/"query source<br/>Option"/]):::input
    SRC_META([/"metadata source<br/>Option"/]):::input
    SRC_GT([/"ground truth<br/>Option"/]):::input

    %% ── Vector chain ─────────────────────────────────────────────
    SRC_VEC --> FETCH["fetch_vectors<br/><i>Identity if no URL</i>"]:::slot
    FETCH --> IMPORT_V["import_vectors<br/><i>Identity if native xvec<br/>(symlinked when possible)</i>"]:::slot
    IMPORT_V --> ALL_V["all_vectors"]:::artifact

    ALL_V --> CONV_BASE["convert_base_precision<br/><i>Identity if no conversion<br/>requested; up-convert or<br/>down-convert via IEEE 754</i>"]:::slot
    CONV_BASE --> CONV_V["converted_vectors<br/>(or alias to all_vectors)"]:::artifact

    CONV_V --> SORT["compute dedup<br/><i>lexicographic external<br/>merge-sort + duplicate<br/>detection as byproduct</i>"]:::slot
    SORT --> SORTED["sorted_ordinals"]:::artifact
    SORT --> DUPS["duplicate_ordinals"]:::artifact

    SORTED --> ZERO["analyze zeros<br/><i>binary search sorted<br/>index for [0,0,...,0]</i>"]:::slot
    CONV_V --> ZERO
    ZERO --> ZEROS["zero_ordinals"]:::artifact

    SORTED --> CLEAN["clean_ordinals<br/><i>Identity if no dups<br/>and no zeros</i>"]:::slot
    DUPS --> CLEAN
    ZEROS --> CLEAN
    CLEAN --> CLEAN_ORD["clean_ordinals"]:::artifact
    CLEAN_ORD --> SET_CC["clean_count<br/>set variable"]:::slot

    %% ── Query chain ──────────────────────────────────────────────
    SRC_QUERY --> IMPORT_Q["import_query<br/><i>Identity if native xvec<br/>(symlinked when possible)</i>"]:::slot

    SET_CC --> SHUFFLE["shuffle<br/><i>Identity if separate query</i>"]:::slot
    CONV_V --> EXT_Q["extract_query<br/><i>Identity if separate query<br/>normalize option</i>"]:::slot
    CLEAN_ORD --> EXT_Q
    SHUFFLE --> EXT_Q
    IMPORT_Q --> QUERY_V

    EXT_Q --> QUERY_RAW["query_vectors<br/>(pre-convert)"]:::artifact
    QUERY_RAW --> CONV_Q["convert_query_precision<br/><i>Identity if no conversion</i>"]:::slot
    CONV_Q --> QUERY_V["query_vectors"]:::artifact

    CONV_V --> EXT_B["extract_base<br/><i>Identity if not self-search<br/>normalize option</i>"]:::slot
    CLEAN_ORD --> EXT_B
    SHUFFLE --> EXT_B
    EXT_B --> BASE_V["base_vectors"]:::artifact
    BASE_V --> SET_BC["base_count<br/><i>Identity if not self-search</i>"]:::slot

    %% ── Metadata chain (entire subgraph Absent when no metadata) ─
    SRC_META --> IMPORT_M["import_metadata<br/><i>Identity if native slab</i>"]:::slot
    IMPORT_M --> META_ALL["metadata_all"]:::artifact

    META_ALL --> EXT_META["extract_metadata<br/><i>Identity if not self-search<br/>uses clean_ordinals + shuffle</i>"]:::slot
    CLEAN_ORD --> EXT_META
    SHUFFLE --> EXT_META
    EXT_META --> META_ALIGNED["metadata_content<br/>(ordinal-aligned)"]:::artifact

    META_ALL --> SURVEY["survey"]:::slot
    SURVEY --> SYNTH["synthesize predicates"]:::slot
    META_ALL --> SYNTH
    SYNTH --> PREDS["predicates.slab"]:::artifact

    PREDS --> EVAL["compute predicates<br/><i>per_profile</i>"]:::slot
    META_ALIGNED --> EVAL
    EVAL --> PRED_IDX["metadata_indices"]:::artifact

    %% ── Ground truth chain ───────────────────────────────────────
    SRC_GT --> KNN
    SET_BC --> KNN["compute knn<br/><i>Identity if GT provided<br/>Absent if no queries<br/>--compress-cache for partitions</i>"]:::slot
    QUERY_V --> KNN
    BASE_V --> KNN
    KNN --> GT["neighbor_indices<br/>neighbor_distances"]:::artifact

    PRED_IDX --> FKNN["compute filtered-knn<br/><i>Absent if no metadata<br/>or --no-filtered<br/>--compress-cache for partitions</i>"]:::slot
    SET_BC --> FKNN
    QUERY_V --> FKNN
    BASE_V --> FKNN
    FKNN --> FGT["filtered_neighbor_indices<br/>filtered_neighbor_distances"]:::artifact

    %% ── Verification chain ───────────────────────────────────────
    GT --> VKNN["verify knn<br/><i>per_profile<br/>sparse-sample brute-force<br/>recomputation</i>"]:::slot
    BASE_V --> VKNN
    QUERY_V --> VKNN
    VKNN --> VKNN_RPT["${cache}/profiles/${profile}/\nverify_knn.json"]:::artifact

    FGT --> VPRED["verify predicates<br/><i>per_profile<br/>SQLite-backed sparse-sample<br/>evaluation</i>"]:::slot
    META_ALIGNED --> VPRED
    PREDS --> VPRED
    PRED_IDX --> VPRED
    VPRED --> VPRED_RPT["${cache}/profiles/${profile}/\nverify_predicates.json"]:::artifact

    %% ── Merkle hash trees (final pipeline step) ───────────────────
    VKNN_RPT --> MERKLE["merkle create<br/><i>always last step<br/>produces .mref files</i>"]:::slot
    VPRED_RPT --> MERKLE
    GT --> MERKLE
    FGT --> MERKLE
    MERKLE --> MREF[".mref files"]:::artifact

    %% ── Catalog generation (final step) ──────────────────────────
    MREF --> CATGEN["catalog generate<br/><i>always last step<br/>produces catalog.json +<br/>catalog.yaml</i>"]:::slot
    CATGEN --> CATFILES["catalog.json<br/>catalog.yaml"]:::artifact

    %% ── Cache compression (offline, post-pipeline) ───────────────
    GT -.-> CACHE_CMP["cache-compress<br/><i>offline retroactive<br/>compression of eligible<br/>cache artifacts</i>"]:::offline
    FGT -.-> CACHE_CMP
    SORTED -.-> CACHE_CMP

    %% ── Profile assembly ─────────────────────────────────────────
    BASE_V --> PROF[/"dataset.yaml<br/>profiles"/]:::output
    QUERY_V --> PROF
    META_ALIGNED --> PROF
    PREDS --> PROF
    GT --> PROF
    FGT --> PROF
    PRED_IDX --> PROF
    VKNN_RPT --> PROF
    VPRED_RPT --> PROF

    %% ── Styling ──────────────────────────────────────────────────
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef slot fill:#fff3e0,stroke:#ef6c00,stroke-dasharray:5 5
    classDef artifact fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef offline fill:#fce4ec,stroke:#c62828,stroke-dasharray:3 3
```

**Reading the graph:**
- **Parallelogram** nodes are user-provided inputs.
- **Dashed** nodes are slots — each independently resolves to either a
  materialized step or an identity alias.
- **Solid green** nodes are artifacts — the resolved output path,
  regardless of whether it was materialized or aliased.
- **Dotted red** nodes are offline operations that can be run after the
  pipeline completes (e.g., `veks datasets cache-compress`).
- The *italic annotation* on each slot states its identity condition.
- When an input is `Option::None`, its entire downstream subgraph
  vanishes from the emitted YAML.

---

## 12.5 Identity Collapse Examples

### Example 1: Minimal — native fvec, no queries, no metadata

```
fetch_vectors          → Identity (no URL)
import_vectors         → Identity (native fvec, symlinked)
all_vectors            → /data/base.fvec (alias to source)
convert_base_precision → Identity (no conversion requested)
sort                   → Materialized (compute dedup → lexicographic sort + duplicate detection)
zero_check             → Materialized (binary search sorted index for zero vector)
clean_ordinals         → Materialized (transform clean-ordinals, exclude dups + zeros)
clean_count            → Materialized (set variable)
shuffle                → Identity (no self-search)
extract_query          → Absent (no queries)
extract_base           → Identity (base = all_vectors)
base_vectors           → /data/base.fvec (alias)
metadata chain         → Absent (no metadata)
knn                    → Absent (no queries)
filtered_knn           → Absent (no metadata)
```

Emitted steps: `dedup-vectors`, `zero-check`, `clean-ordinals`,
`set-clean-count`. Four steps total.

### Example 2: Maximal — foreign format, self-search, metadata, all GT

```
fetch_vectors          → Materialized (fetch bulkdl)
import_vectors         → Materialized (import npy → mvec)
all_vectors            → ${cache}/all_vectors.mvec
convert_base_precision → Identity (no conversion requested)
sort                   → Materialized (compute dedup → lexicographic sort + duplicate detection)
zero_check             → Materialized (binary search sorted index for zero vector)
clean_ordinals         → Materialized (transform clean-ordinals, exclude dups + zeros)
clean_count            → Materialized (set variable)
shuffle                → Materialized (generate ivec-shuffle over clean_count)
extract_query          → Materialized (transform mvec-extract [0, query_count) via clean_ordinals)
convert_query_prec     → Identity (no conversion requested)
extract_base           → Materialized (transform mvec-extract [query_count, N) via clean_ordinals)
base_count             → Materialized (set variable)
import_metadata        → Materialized (import parquet → slab)
extract_metadata       → Materialized (transform slab-extract, clean_ordinals + shuffle aligned)
survey                 → Materialized (survey)
predicates             → Materialized (synthesize predicates)
pred_indices           → Materialized (compute predicates, per_profile)
knn                    → Materialized (compute knn, per_profile, compress_cache=true)
filtered_knn           → Materialized (compute filtered-knn, per_profile, compress_cache=true)
verify_knn             → Materialized (verify knn, per_profile, sparse sample=100)
verify_predicates      → Materialized (verify predicates, per_profile, sparse sample=50, metadata-sample=100K)
```

All slots materialized. This is the laion400m-img-search pattern.

### Example 3: Native xvec base + separate native query, with metadata

```
import_vectors         → Identity (native, symlinked)
all_vectors            → base.fvec (alias)
convert_base_precision → Identity (no conversion)
sort                   → Materialized (compute dedup — lexicographic sort + dup detection)
zero_check             → Materialized (binary search sorted index)
clean_ordinals         → Materialized (transform clean-ordinals)
clean_count            → Materialized (set variable)
shuffle                → Identity (separate query, no self-search)
import_query           → Identity (native xvec, symlinked)
query_vectors          → query.fvec (alias)
convert_query_prec     → Identity (no conversion)
extract_base           → Identity (not self-search)
base_vectors           → base.fvec (alias)
import_metadata        → Materialized (parquet → slab)
extract_metadata       → Identity (no shuffle, ordinals aligned)
metadata_content       → ${cache}/metadata_all.slab (alias)
survey                 → Materialized
predicates             → Materialized
pred_indices           → Materialized (per_profile)
knn                    → Materialized (per_profile)
filtered_knn           → Materialized (per_profile)
verify_knn             → Materialized (per_profile)
verify_predicates      → Materialized (per_profile)
```

Import steps collapse; shuffle/extract collapse; metadata extract
collapses. Dedup, zero-check, and clean-ordinals always run.
All compute steps still run.

### Example 4: Precision conversion — f32 source, target f16

```
import_vectors         → Identity (native fvec, symlinked)
all_vectors            → /data/base.fvec (alias)
convert_base_precision → Materialized (convert fvec → mvec, lossy IEEE 754 round-to-nearest-even)
converted_vectors      → ${cache}/all_vectors.mvec
dedup                  → Materialized (compute dedup on converted mvec)
...                    → (remainder of graph uses converted mvec)
```

The wizard detects the precision mismatch and automatically inserts
the convert step. Down-conversion (f32→f16) is lossy; up-conversion
(f16→f32) is lossless but does not improve accuracy.

---

## 12.6 Type-Safe Optional Axes

The three optional axes — query vectors, metadata, and ground truth —
use a typed `Option` model in the generator:

```rust
struct PipelineSlots {
    // Required
    all_vectors: Artifact,
    dedup: Artifact,            // compute dedup → sorted ordinals + duplicates
    zero_check: Artifact,       // binary search sorted index for zero vector
    clean_ordinals: Artifact,   // transform clean-ordinals (exclude dups + zeros)
    clean_count: Artifact,      // set variable on clean_ordinals

    // Query axis (None → no KNN, no filtered-KNN)
    self_search: bool,
    shuffle: Option<Artifact>,
    query_vectors: Option<Artifact>,
    base_vectors: Artifact,     // = all_vectors when no self-search
    base_count: Option<Artifact>,

    // Metadata axis (None → entire predicate chain pruned)
    metadata: Option<MetadataSlots>,

    // Ground truth axis
    knn: Option<Artifact>,           // None when no queries or GT provided
    filtered_knn: Option<Artifact>,  // None when no metadata or --no-filtered
}

struct MetadataSlots {
    metadata_all: Artifact,
    metadata_content: Artifact,  // = metadata_all when no self-search
    survey: Artifact,
    predicates: Artifact,
    predicate_indices: Artifact,
}

enum Artifact {
    /// A real pipeline step produces this artifact.
    Materialized { step_id: String, output: PathBuf },
    /// No step needed — artifact is an alias to an existing path.
    Identity { path: PathBuf },
}
```

When `query_vectors` is `None`:
- `knn` is forced to `None` (nothing to query against)
- `filtered_knn` is forced to `None`
- `shuffle`, `extract_query`, `extract_base` are all `Identity`

When `metadata` is `None`:
- The entire `MetadataSlots` struct is absent
- `filtered_knn` is forced to `None`
- No survey, predicate synthesis, or evaluation steps are emitted

When ground truth is provided externally:
- `knn` resolves to `Identity` (alias to the provided files)
- `filtered_knn` may still be `Materialized` if metadata is present

---

## 12.7 Pipeline Emission Algorithm

The generator walks the slot graph in topological order and emits a
step for each `Materialized` slot:

```
fn emit_pipeline(slots: &PipelineSlots, args: &ImportArgs) -> Vec<StepDef> {
    let mut steps = Vec::new();
    let mut last_vector_step = "";

    // Vector chain — emit in dependency order
    if let Materialized { .. } = slots.all_vectors {
        steps.push(/* import step */);
        last_vector_step = "import-vectors";
    }

    // Precision conversion (e.g., fvec→mvec or mvec→fvec)
    if args.base_convert_format.is_some() {
        steps.push(/* convert step */);
        last_vector_step = "convert-base-precision";
    }

    if let Materialized { .. } = slots.dedup {
        steps.push(/* compute dedup → sorted ordinals + duplicates */);
    }
    if let Materialized { .. } = slots.zero_check {
        steps.push(/* binary search sorted index for zero vector */);
    }
    if let Materialized { .. } = slots.clean_ordinals {
        steps.push(/* transform clean-ordinals — exclude dups + zeros */);
    }
    steps.push(/* set-clean-count — always */);

    // Query chain
    if let Some(ref qv) = slots.query_vectors {
        match qv {
            Materialized { .. } => {
                // Self-search path
                steps.push(/* generate ivec-shuffle */);
                steps.push(/* transform extract query */);
                steps.push(/* transform extract base */);
                steps.push(/* set-base-count */);
            }
            Identity { .. } => {
                // Separate query — no shuffle/extract steps
            }
        }
    }

    // Metadata chain — only when Some
    if let Some(ref meta) = slots.metadata {
        if let Materialized { .. } = meta.metadata_all {
            steps.push(/* import metadata */);
        }
        if let Materialized { .. } = meta.metadata_content {
            steps.push(/* transform slab-extract */);
        }
        steps.push(/* survey */);
        steps.push(/* synthesize predicates */);
        steps.push(/* compute predicates (per_profile) */);
    }

    // Ground truth
    if let Some(Materialized { .. }) = slots.knn {
        steps.push(/* compute knn (per_profile) */);
    }
    if let Some(Materialized { .. }) = slots.filtered_knn {
        steps.push(/* compute filtered-knn (per_profile) */);
    }

    steps
}
```

Profile views are assembled from the resolved `output` path of each
slot, regardless of whether it is `Materialized` or `Identity`.
`Absent` slots are omitted from the profiles entirely.

---

## 12.8 CLI Options for `datasets import`

| Option | Type | Description |
|--------|------|-------------|
| `--name` | string | Dataset name (required unless `--interactive`) |
| `-o, --output` | path | Output directory (required unless `--interactive`) |
| `-i, --interactive` | flag | Launch interactive wizard mode |
| `-y, --yes` | flag | Accept all wizard defaults without prompting (use with `-i`) |
| `-r, --restart` | flag | Remove dataset.yaml, variables.yaml, and progress log before proceeding (implies `--force`) |
| `--auto` | flag | Fully automatic mode — implies `-i -r -y`. Detects file roles by filename keywords, accepts all defaults, starts fresh. Refuses to continue if any candidate file cannot be assigned a role (see §12.16). |
| `--base-vectors` | path | Base vector source (file or directory) |
| `--query-vectors` | path | Separate query vectors (file or directory) |
| `--self-search` | flag | Extract queries from base via shuffle (default when no `--query-vectors`) |
| `--query-count` | int | Number of query vectors in self-search mode (default: 10000) |
| `--metadata` | path | Metadata source (file or directory) |
| `--ground-truth` | path | Pre-computed ground truth indices (ivec) |
| `--ground-truth-distances` | path | Pre-computed ground truth distances (fvec) |
| `--metric` | string | Distance metric for KNN: L2, Cosine, DotProduct, L1 (required) |
| `--neighbors` | int | Number of neighbors for KNN (default: 100) |
| `--seed` | int | Random seed for shuffle (default: 42) |
| `--description` | string | Dataset description |
| `--no-dedup` | flag | Collapse dedup slot to identity |
| `--no-zero-check` | flag | Collapse zero_check slot to identity |
| `--no-filtered` | flag | Force filtered_knn to Absent |
| `--normalize` | flag | L2-normalize vectors during extraction |
| `--compress-cache` | flag | Enable gzip compression for eligible cache artifacts (default: true) |
| `--force` | flag | Overwrite existing dataset.yaml |

### Implied defaults

- `--auto` → `-i -r -y` (interactive + restart + auto-accept)
- `--base-vectors` without `--query-vectors` → self-search mode
- `--metadata` present → full predicate chain unless `--no-filtered`
- `--ground-truth` → KNN slot collapses to Identity (alias to provided files)
- `--metric` is required and has no default — user must declare intent

---

## 12.9 Starting Scenarios

The `datasets import` command can be invoked from several well-defined
initial conditions. Each produces a valid `dataset.yaml` and pipeline:

### Scenario 1: Fresh start — no existing state

The most common case. The user has source data files and wants to create
a new dataset from scratch.

**Preconditions**: no `dataset.yaml`, no `variables.yaml`, no `.cache/`
**Invocation**: `veks datasets import -i` (interactive) or with explicit flags
**Behavior**: scans for data files, prompts for options, generates `dataset.yaml`

### Scenario 2: Restart — discard existing state

The user has a prior `dataset.yaml` and computed artifacts but wants to
start over (e.g., changed source data, different options).

**Preconditions**: existing `dataset.yaml`, possibly `variables.yaml` and `.cache/`
**Invocation**: `veks datasets import -i --restart`
**Behavior**: removes `dataset.yaml`, `variables.yaml`, and `.cache/.upstream.progress.yaml`,
then proceeds as Scenario 1. Cached artifacts in `.cache/` are NOT deleted —
the pipeline runner's skip-if-fresh logic will recompute invalidated steps.

### Scenario 3: Auto-accept — scripted defaults

Non-interactive generation using all default answers, suitable for
scripting or when the source data is well-understood.

**Preconditions**: source data files present in working directory
**Invocation**: `veks datasets import -i -y` (or `-iry`)
**Behavior**: runs the wizard with all prompts returning their default
values. Still prints each decision to stderr for auditability.

### Scenario 3a: Fully automatic — `--auto`

One-command dataset bootstrap. Detects file roles by filename keywords,
auto-accepts all defaults, and starts fresh.

**Preconditions**: source data files in working directory with
recognizable role keywords in filenames (see §12.16)
**Invocation**: `veks datasets import --auto`
**Behavior**: equivalent to `-iry` but additionally enforces that every
candidate data file in the directory has a recognized role assignment.
If any file cannot be mapped to a role, the command refuses to continue
and prints a table of recognized keywords so the user can rename files.
Source files are automatically renamed with a `_` prefix (the default
source-location option). This is the recommended mode for directories
that follow the naming conventions.

### Scenario 4: CLI flags — fully explicit

No wizard. All options specified via command-line flags.

**Preconditions**: source data files at known paths
**Invocation**: `veks datasets import --name my-ds -o . --base-vectors base.mvec --metric L2`
**Behavior**: resolves slots directly from flags, no user interaction.
Fails with an error if required options are missing.

### Scenario 5: Native xvec — identity collapse

Source data is already in the target xvec format. The import step
collapses to an identity alias (symlink), avoiding redundant I/O.

**Preconditions**: source is a single `.fvec`, `.mvec`, `.dvec`, etc. file
**Behavior**: `all_vectors` slot resolves to `Identity` pointing at the
source file. Downstream steps (sort, shuffle, extract) read directly from
the source via mmap.

### Scenario 6: Foreign format — full import

Source data requires format conversion (npy directory, parquet, etc.).

**Preconditions**: source is a directory of `.npy` files, a `.parquet` file, etc.
**Behavior**: `all_vectors` slot materializes as an `import` step that
converts to xvec format. The converted file goes to `${cache}/all_vectors.mvec`.

### Scenario 7: Pre-computed ground truth

The user provides pre-computed KNN indices instead of computing them.

**Preconditions**: `--ground-truth` flag points to an existing `.ivec` file
**Behavior**: `knn` slot resolves to `Identity` (alias to provided file).
The `compute knn` step is not emitted. Verification still runs against
the provided indices.

### Scenario 8: With metadata — full predicate chain

Source data includes metadata (parquet, slab) alongside vectors.

**Preconditions**: `--metadata` flag points to metadata source
**Behavior**: full metadata chain materializes: import → survey →
synthesize predicates → compute predicates → filtered KNN → verify.
Metadata is ordinal-aligned with vectors through the same shuffle/extract
steps.

---

## 12.10 Interactive Wizard

The `--interactive` flag launches a guided import wizard that walks the
user through each decision point.

### 12.10.1 Reasonable-Default Requirement

**Every interactive prompt MUST provide a reasonable default value that
produces a correct result when auto-accepted.** This is a normative
requirement: the interactive wizard is the backbone of `--auto` mode,
which runs the wizard with all prompts returning their defaults. If any
prompt's default produces an invalid or incomplete result, `--auto` mode
is broken. Specifically:

- Every yes/no confirmation must have a default (Y or N) that produces
  the correct pipeline for the most common case.
- Every multiple-choice prompt must have a default selection that is
  stable — it must not depend on the ordering of candidates, which may
  vary between runs.
- When filename-keyword role detection assigns a file to a role, that
  assignment becomes the default for the corresponding prompt, bypassing
  unstable multi-select menus.
- The default for source-file location is "rename with `_` prefix"
  (option 1), ensuring source files are marked as non-publishable.

This requirement ensures that the interactive wizard, auto-accept mode
(`-iy`), and fully automatic mode (`--auto`) produce identical results
for the same input files, differing only in whether the user is prompted
to confirm each step.

### 12.10.2 Wizard Flow

The wizard proceeds through these phases:

1. **Scans the working directory** for recognized data files (xvec,
   npy, parquet, slab) and presents detected candidates.
2. **Filename-keyword role detection** (§12.16): examines each
   candidate's filename for role-hinting keywords and presents a
   single confirmation prompt for all detected assignments. In
   `--auto` mode, unrecognized files cause a hard stop with guidance.
3. **Prompts for each option** incrementally, with sensible defaults
   derived from detected files. When role detection has assigned a
   file, the corresponding prompt is pre-filled and skips multi-select.
4. **Precision confirmation**: for floating-point xvec sources (fvec,
   mvec, dvec), the wizard probes the file and displays its current
   precision (element size, dimensions, record count), then asks whether
   the user wants to keep the native precision or convert:
   - **Accept as-is** (default): no conversion step emitted.
   - **Up-convert** (e.g., f16→f32): warns that extra precision bits are
     zero-filled and accuracy is not improved; emits a `convert` step.
   - **Down-convert** (e.g., f32→f16): warns that this is lossy with
     IEEE 754 round-to-nearest-even semantics and values outside the
     target range saturate to ±Inf; emits a `convert` step.
5. **Normalization detection**: samples vectors and reports mean L2 norm.
   If not normalized, offers to add `--normalize` to extraction steps.
6. **Cache compression**: asks whether to enable gzip compression for
   eligible cache artifacts (default: yes). This adds `compress_cache:
   true` to pipeline steps that produce sequential-only intermediates.
7. **Displays a summary** of all resolved options and asks for
   confirmation before generating `dataset.yaml`.

The wizard produces the same `ImportArgs` struct as the CLI flags,
including `base_convert_format` and `query_convert_format` when
precision conversion is requested.

### 12.10.3 Import Provenance Log

When `datasets import` generates a `dataset.yaml`, it also writes a
provenance entry to `dataset.log` in the output directory. This entry
records:

- **Timestamp** of the import
- **All input arguments**: paths, flags, metric, seed, etc.
- **Detected starting scenario**: a human-readable label describing the
  combination of inputs (e.g., "full (base + query + metadata, compute
  GT)" or "base vectors only (no queries, no metadata)")
- **Dataflow graph entry points**: the resolution of every slot
  (Materialized, Identity, or Absent) with output paths

This ensures that `dataset.log` provides a complete audit trail from
the moment of import, before any pipeline step has run. Subsequent
pipeline runs append their own entries (step outcomes, timing, errors)
to the same file.

---

## 12.11 Precision Conversion

The `convert_base_precision` and `convert_query_precision` slots handle
element type conversion between the six xvec floating-point and integer
formats:

| Format | Element | Size | Notes |
|--------|---------|------|-------|
| `dvec` | f64 | 8 bytes | IEEE 754 double |
| `fvec` | f32 | 4 bytes | IEEE 754 single |
| `mvec` | f16 | 2 bytes | IEEE 754 half |
| `ivec` | i32 | 4 bytes | Two's complement signed |
| `svec` | i16 | 2 bytes | Two's complement signed |
| `bvec` | u8 | 1 byte | Unsigned byte |

### Conversion semantics

**Up-conversion** (narrower → wider, e.g., f16→f32, f32→f64):
- Lossless. Every value in the source type is exactly representable in
  the target type.
- The additional precision bits are zero-filled.
- Does not improve the accuracy of the original data. Useful only for
  compatibility with tools that require a specific element width.

**Down-conversion** (wider → narrower, e.g., f32→f16, f64→f32):
- **Lossy.** IEEE 754 round-to-nearest-even rules apply.
- Values outside the target type's representable range saturate to ±Inf.
- Subnormal values may flush to zero depending on the target format.
- The `convert` pipeline step reports precision loss statistics
  (max error, mean error, saturation count) so the user can assess
  whether the conversion is acceptable.

### When conversion is emitted

- The wizard sets `base_convert_format` or `query_convert_format` in
  `ImportArgs` when the user selects a different precision.
- The pipeline generator emits a `convert` step immediately after the
  `import`/`identity` step for the affected vector source.
- All downstream steps (dedup, extract, KNN, etc.) operate on the
  converted vectors.

---

## 12.12 Zero-Length Intermediate Products

A pipeline step may legitimately produce a **zero-byte output file**.
This is a defined result, not an error condition. Examples:

- `analyze zeros` finds no zero vectors → writes a 0-byte
  `zero_ordinals.ivec` (0 records)
- `clean-ordinals` with no exclusions → the full sorted index passes
  through unchanged (though in practice this means all records survive)

The pipeline runner must treat 0-byte xvec files as `Complete`, not
`Partial`. The artifact checker recognizes that a 0-byte file in a
recognized xvec format (fvec, ivec, mvec, dvec, bvec, svec) represents
0 records and is structurally valid.

Downstream steps that consume these files must handle count = 0
gracefully. The vector reader returns count = 0 and dim = `DIM_UNDEFINED`
(a sentinel, not 0) for empty files. Callers must check `count > 0`
before relying on `dim`.

---

## 12.13 Import Optimizations

### Symlink fast path

When the source file meets all of these conditions:
- Already in the target xvec format (source format = target format)
- No `--count` limit specified (importing all records)
- Source is a single file (not a directory of npy/parquet shards)

The `import` command creates a symlink to the source file instead of
copying bytes. This eliminates redundant I/O for large native-format
files. If symlink creation fails (e.g., cross-filesystem), the command
falls back to a normal byte copy.

### Compressed cache artifacts

**Core principle:** Only cache files that are consumed in sequential or
in-memory-segment mode are eligible for compression. These files are
never mmap'd for random access — they are always read entirely into
memory before use (e.g., sorted runs are streamed through a merge heap,
KNN partition caches are loaded row-by-row per query). Compressing them
has zero impact on access patterns and saves significant disk space,
often 3-10x for integer ordinal data and 1.5-3x for floating-point
partition caches.

Files that require random-access mmap (all_vectors, shuffle indices,
dedup ordinals, clean ordinals, and all final output files) are **never
compressed**, because decompressing them would require loading the
entire file into memory and would defeat the O(1) seek property.

Pipeline commands that produce eligible cache artifacts support a
`compress_cache` option. When enabled:

- Eligible files are gzip-compressed in memory before writing to disk.
- Compressed files use a `.gz` suffix appended to the original filename.
- The original uncompressed file is removed after successful compression.
- File timestamps (mtime) are preserved on the compressed file for
  provenance tracking.
- Loading transparently decompresses in memory via the `gz_cache` module.
- Compression level defaults to 9 (maximum) — these files are written
  once and read many times, so upfront CPU cost is amortized.

The `veks datasets cache-compress` and `cache-uncompress` commands
provide **offline retroactive compression** as a convenience. They walk
the cache directory, identify eligible files by name pattern, and
compress or decompress them in parallel using rayon. This is useful for
reducing disk footprint on existing caches that were produced before
`compress_cache` was enabled. The offline commands apply the same
eligibility rules and never touch random-access files.

**Eligible for compression** (sequential or in-memory segment access):
- Dedup sorted runs (`dedup_runs/run_*.bin` → `.bin.gz`)
- KNN partition caches (`*.neighbors.ivec` → `.ivec.gz`, `*.distances.fvec` → `.fvec.gz`)
- Predicate segment caches (`*.predkeys.slab` → `.slab.gz`)

**Never eligible** (require mmap random access from disk):
- `all_vectors.mvec` — random access by ordinal
- `shuffle.ivec` — random access by index
- `dedup_ordinals.ivec` — binary search
- `clean_ordinals.ivec` — random access
- Any final output file referenced by profile views

### KNN partition cache reuse

KNN partition cache keys are derived from `base_stem.query_stem.range.k.metric`
— deliberately excluding the pipeline step ID. This means:
- Partitions computed for a smaller profile (e.g., 10M base vectors)
  are automatically reused by larger profiles (50M, 150M) that share
  the same base file, query file, k, and metric.
- Ordering profiles smallest-to-largest in the pipeline maximizes cache
  reuse. Later profiles skip all overlapping partition ranges.

---

## 12.14 Output Convention

Resolved artifact paths follow the workspace layout (§5.7):

| Artifact | Path | Cache/final |
|----------|------|-------------|
| Imported vectors | `${cache}/all_vectors.mvec` | cache |
| Converted vectors | `${cache}/all_vectors.<ext>` | cache |
| Sorted ordinals (dedup) | `${cache}/dedup_ordinals.ivec` | cache |
| Duplicate ordinals | `${cache}/dedup_duplicates.ivec` | cache |
| Dedup report | `${cache}/dedup_report.json` | cache |
| Zero ordinals | `${cache}/zero_ordinals.ivec` | cache |
| Clean ordinals | `${cache}/clean_ordinals.ivec` | cache |
| Dedup sorted runs | `${cache}/dedup_runs/` | cache |
| Shuffle permutation | `${cache}/shuffle.ivec` | cache |
| Imported metadata | `${cache}/metadata_all.slab` | cache |
| Metadata survey | `${cache}/metadata_survey.json` | cache |
| Base vectors | `profiles/base/base_vectors.<ext>` | final |
| Query vectors | `profiles/base/query_vectors.<ext>` | final |
| Metadata content | `profiles/base/metadata_content.slab` | final |
| Predicates | `predicates.slab` | final |
| KNN indices | `profiles/${profile}/neighbor_indices.ivec` | final |
| KNN distances | `profiles/${profile}/neighbor_distances.fvec` | final |
| Filtered indices | `profiles/${profile}/filtered_neighbor_indices.ivec` | final |
| Filtered distances | `profiles/${profile}/filtered_neighbor_distances.fvec` | final |
| Predicate indices | `profiles/${profile}/metadata_indices.slab` | final |
| KNN verification report | `${cache}/profiles/${profile_name}/verify_knn.json` | cache |
| Predicate verification report | `${cache}/profiles/${profile_name}/verify_predicates.json` | cache |

### 12.14.1 The `profiles/base/` Directory

The `profiles/base/` directory is the **canonical location for
full-dataset artifacts** that are shared by all profiles. Named
profiles (e.g., `default`, `10m`, `50m`) reference these base
artifacts, potentially with range windows (e.g.,
`profiles/base/base_vectors.mvec[0..10000000)`).

All profile view paths in `dataset.yaml` point into this canonical
structure — never to raw source file paths. This ensures:

- **Consistent layout**: every dataset has the same directory structure
  regardless of how it was created (self-search vs. separate query,
  native format vs. imported).
- **Clean publishing**: all publishable data lives under `profiles/`
  and the dataset root. Source files with `_` prefix are excluded.
- **Symlink transparency**: when a source file requires no conversion,
  `profiles/base/<facet>.<ext>` is a symlink to the source. This
  avoids copying while maintaining the canonical layout.

### 12.14.2 Symlink Semantics

When a source file is already in native format and no transformation
is needed (Identity resolution), the import command creates a symlink
at the canonical `profiles/base/` location pointing to the source file:

```
profiles/base/base_vectors.mvec  →  _base_vectors.mvec
profiles/base/query_vectors.fvec →  _query_vectors.fvec
profiles/base/metadata_content.slab → _metadata.slab
```

When the pipeline runs an extract or import step (Materialized
resolution), the step writes a real file at the canonical path and
no symlink is needed.

**Publishing and transport**: Symlinks are **not preserved** through
any transport or publishing mechanism. Publishing tools (rsync, S3
upload, archive creation) MUST resolve symlinks and copy the actual
data. The `_`-prefixed source files are excluded from publishing by
convention (the `_` prefix marks non-publishable files). Since
publishing resolves symlinks to real data, the published dataset
contains only real files at the canonical paths — consumers never see
symlinks.

---

## 12.15 Post-Pipeline Verification

After the compute phases produce KNN ground truth and predicate answer
keys, two verification pipeline phases perform sparse-sample spot-checks
to confirm correctness. These are the last steps in the pipeline before
the dataset is considered publication-ready.

### 12.15.1 KNN Verification (`verify knn`)

Recomputes brute-force exact KNN for a sparse random sample of query
vectors and compares against the stored ground-truth results. This
catches data corruption, byte-order issues, shuffle bugs, off-by-one
errors, and distance function mismatches.

**Algorithm:**

1. **Sample selection**: Choose `sample_count` (default: 100) query
   indices uniformly at random from the full query set. Use a
   deterministic seed derived from the dataset seed for reproducibility.

2. **Parallel brute-force recomputation**: For each sampled query,
   recompute exact top-k nearest neighbors against the full base vector
   set. This uses the same optimized infrastructure as `compute knn`:
   - SIMD-accelerated distance functions (simsimd for f32/f16/f64)
   - Governor-limited rayon thread pool
   - Batched query processing
   - For large base sets, the existing partitioned computation with
     cache reuse applies — verification queries reuse the same partition
     caches as the original computation.

3. **Comparison**: For each sampled query, compare the recomputed top-k
   index set against the stored indices from the ground-truth ivec file.
   - **Exact match**: all k indices are identical → pass
   - **Distance-tie match**: indices differ but all discrepant indices
     have distances within `phi` tolerance of the k-th true neighbor →
     pass (ties are expected in quantized or low-dimensional data)
   - **Mismatch**: true failure → report query index, expected vs actual
     indices, and distance comparison

4. **Report**: Write `verify_knn.json` with:
   - `sample_count`, `pass_count`, `fail_count`, `tie_count`
   - `recall@k` (fraction of sampled queries with correct results)
   - Per-failure detail (query index, mismatched indices, distances)
   - Elapsed time

**Pipeline step:**

```yaml
- id: verify-knn
  run: verify knn
  after: [compute-knn]
  per_profile: true
  base: profiles/base/base_vectors.mvec
  query: profiles/base/query_vectors.mvec
  indices: neighbor_indices.ivec
  distances: neighbor_distances.fvec
  metric: L2
  sample: 100
  seed: "${seed}"
  output: "${cache}/profiles/${profile_name}/verify_knn.json"
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base` | Path | required | Base vectors file |
| `query` | Path | required | Query vectors file |
| `indices` | Path | required | Precomputed neighbor indices (ivec) |
| `distances` | Path | optional | Precomputed neighbor distances (fvec) |
| `metric` | enum | L2 | Distance metric |
| `sample` | int | 100 | Number of query vectors to spot-check |
| `seed` | int | 42 | Random seed for sample selection |
| `phi` | float | 0.001 | Distance tolerance for tie detection |
| `output` | Path | required | Verification report (JSON) |

### 12.15.2 Predicate Verification (`verify predicates`)

Evaluates a sparse random sample of predicates against the full metadata
set using SQLite as an independent source of truth, and compares the
resulting ordinal sets against the stored predicate evaluation results
(`metadata_indices.slab`).

This provides a completely independent verification path: the pipeline
computes predicate results using the custom slab-based evaluator, and
the verification step recomputes them using SQL queries against an
in-memory SQLite database. Any discrepancy indicates a bug in either
the predicate evaluator or the metadata import.

**Algorithm:**

1. **SQLite loading (sampled)**: Load a uniform random sample of metadata
   records (default: 100K rows, configurable via `metadata-sample`) into
   an in-memory SQLite database. Each sampled MNode record becomes a row
   with its original ordinal preserved. This bounds memory to ~100K rows
   regardless of dataset size. The sample size is configurable to trade
   off verification thoroughness vs. memory/time.

2. **Sample selection**: Choose `sample_count` (default: 50) predicate
   indices uniformly at random from the predicates slab.

3. **PNode → SQL translation**: Each sampled PNode predicate tree is
   translated to an equivalent SQL WHERE clause:
   - `EQ(field, value)` → `field = value`
   - `GT(field, value)` → `field > value`
   - `LT(field, value)` → `field < value`
   - `AND(a, b)` → `(a) AND (b)`
   - `OR(a, b)` → `(a) OR (b)`
   - `IN(field, values)` → `field IN (v1, v2, ...)`
   - `BETWEEN(field, lo, hi)` → `field BETWEEN lo AND hi`

4. **SQL evaluation**: Execute `SELECT ordinal FROM metadata WHERE <clause>`
   to get the set of matching ordinals from SQLite.

5. **Comparison**: For each sampled predicate, compare the SQLite result
   set against the stored ordinal set from `metadata_indices.slab`:
   - Extra ordinals in stored results → false positives
   - Missing ordinals from stored results → false negatives
   - Both are failures that would corrupt filtered KNN results

6. **Filtered KNN spot-check** (optional, when filtered ground truth
   exists): For a subset of sampled predicates, also verify that the
   filtered KNN indices are consistent with the predicate results. For
   each sampled predicate, take its first query, apply the predicate
   filter to the base vectors, and recompute brute-force KNN over the
   filtered subset. Compare against stored filtered KNN indices.

7. **Report**: Write `verify_predicates.json` with:
   - `sample_count`, `pass_count`, `fail_count`
   - `false_positive_count`, `false_negative_count`
   - Per-failure detail (predicate index, SQL clause, expected vs actual ordinals)
   - SQLite load time, evaluation time

**Pipeline step:**

```yaml
- id: verify-predicates
  run: verify predicates
  after: [compute-filtered-knn]
  per_profile: true
  metadata: profiles/base/metadata_content.slab
  predicates: predicates.slab
  metadata-indices: metadata_indices.slab
  survey: ${cache}/metadata_survey.json
  sample: 50
  seed: "${seed}"
  output: "${cache}/profiles/${profile_name}/verify_predicates.json"
  # Optional: also verify filtered KNN
  filtered-indices: filtered_neighbor_indices.ivec
  base: profiles/base/base_vectors.mvec
  query: profiles/base/query_vectors.mvec
  metric: L2
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `metadata` | Path | required | Metadata content slab |
| `predicates` | Path | required | Predicates slab (PNode records) |
| `metadata-indices` | Path | required | Precomputed predicate results slab |
| `survey` | Path | required | Metadata survey JSON (for schema) |
| `sample` | int | 50 | Number of predicates to spot-check |
| `seed` | int | 42 | Random seed for sample selection |
| `output` | Path | required | Verification report (JSON) |
| `filtered-indices` | Path | optional | Filtered KNN indices for cross-check |
| `base` | Path | optional | Base vectors (for filtered KNN cross-check) |
| `query` | Path | optional | Query vectors (for filtered KNN cross-check) |
| `metric` | enum | L2 | Distance metric (for filtered KNN cross-check) |

### 12.15.3 Design Rationale

**Why sparse sampling?** Full recomputation of all queries against all
base vectors is prohibitively expensive for large datasets (407M × 10K
queries = 4 trillion distance computations). Sparse sampling of 100
queries provides high confidence: if any systematic error exists (wrong
byte order, off-by-one in shuffle, incorrect distance function), even a
single sampled query will detect it. Random errors (bit flips, partial
corruption) are detected with probability proportional to the sample
fraction.

**Why SQLite for predicates?** The predicate evaluator in the pipeline
uses a custom slab-based engine optimized for throughput. SQLite provides
a completely independent implementation of the same logical operations.
If both agree, we have high confidence the results are correct. If they
disagree, the SQL query serves as a human-readable reference for
debugging.

**Why reuse existing infrastructure?** The verification queries are
computationally identical to the original KNN computation — they just
run on fewer queries. Reusing the partitioned computation, SIMD distance
functions, and governor controls means verification benefits from the
same optimizations and runs within the same resource constraints as the
original pipeline.

---

## 12.16 Filename-Keyword Role Detection

The import wizard classifies candidate data files into dataset roles
based on keyword substrings in their filenames. This eliminates
unstable multi-select prompts and makes auto-accept mode (`-y`) and
fully automatic mode (`--auto`) produce valid results when filenames
follow conventional naming.

### 12.16.1 Detection Algorithm

After `scan_candidates()` finds all recognized data files in the
working directory, `detect_roles()` examines each file's **stem**
(filename without extension, lowercased) for keyword substrings and
assigns it to a `StandardFacet` role.

**Preprocessing:**
- The filename stem is lowercased.
- A leading `_` prefix is stripped before matching. This ensures that
  source files renamed by a prior import (e.g., `_base_vectors.mvec`)
  are still recognized.
- The stem is split on delimiters (`_`, `-`, `.`) into tokens for
  word-boundary matching.

**Matching modes:**
- **Token match**: short, ambiguous keywords (`base`, `train`, `query`,
  `test`, `gt`, `content`, `filter`, `filtered`) are matched as whole
  tokens only (delimited by `_`, `-`, `.`). This prevents false
  positives like `test` matching inside `base_test` or `gt` matching
  inside `weight`.
- **Substring match**: longer, unambiguous keywords (`groundtruth`,
  `metadata`, `neighbors`, `predicate`, `distance`, `result`) are
  matched as substrings of the full stem.

### 12.16.2 Role Assignment Rules

Detection is evaluated top-to-bottom; the first matching rule wins.
Filtered variants take priority over non-filtered. Within the vector
roles, `base` takes priority over `query`/`test` when both keywords
are present in the same filename.

| Role | Keywords | Format constraint |
|------|----------|-------------------|
| Filtered neighbor indices | `filtered`∣`filter` AND (`indices`∣`neighbors`∣`gt`∣`groundtruth`) | ivec |
| Filtered neighbor distances | `filtered`∣`filter` AND `distance*` | fvec, dvec, mvec |
| Neighbor indices (GT) | `groundtruth`, `gt`, `indices`, `neighbors` | ivec |
| Neighbor distances (GT) | `distance*` (but not also matching indices keywords) | fvec, dvec, mvec |
| Metadata predicates | `predicate*` | slab |
| Metadata results | `result*` | slab |
| Metadata content | `metadata`, `content` (but not also matching predicates/results keywords) | slab, parquet |
| Base vectors | `base`, `train` | any vector format |
| Query vectors | `query`, `queries`, `test` | any vector format |

**Vector formats**: fvec, ivec, mvec, bvec, dvec, svec, npy
**Float vector formats**: fvec, dvec, mvec

### 12.16.3 Ambiguity Resolution

If two or more candidate files claim the same role, **neither is
assigned** — the role is left unresolved and falls through to the
wizard's manual selection prompt. This prevents silent misassignment
when, e.g., two files both contain `base` in their names.

Files that match no role are placed in the **unassigned** list.

### 12.16.4 `--auto` Mode Enforcement

In `--auto` mode, the unassigned list must be empty. If any candidate
data file cannot be mapped to a role, the command prints an error with:

1. The list of unrecognized files
2. A table of recognized keywords per role
3. Examples of correctly-named files

The command then exits with a non-zero status. The user must rename
their files to include appropriate keywords before retrying.

This strictness is deliberate: `--auto` mode is designed for
directories where the user has already named files according to
convention. Ambiguous directories should use interactive mode (`-i`)
where the user can manually resolve assignments.

### 12.16.5 Wizard Integration

When role detection succeeds, the wizard presents a single confirmation
prompt showing all detected assignments:

```
Detected file roles:
  Base vectors:       _base_vectors.mvec
  Query vectors:      query_test.mvec
  Metadata content:   metadata.slab

Use detected assignments? [Y/n]:
```

When accepted (or auto-accepted with `-y`), the detected files are used
directly for the corresponding wizard sections, bypassing the unstable
multi-select menus. When rejected (user answers N), the wizard falls
through to its existing per-role prompts.

### 12.16.6 Source File Renaming

All source files are automatically renamed with a `_` prefix (the
wizard's default source-location option) to mark them as
non-publishable. The `_` prefix is stripped during role detection, so
files renamed by a prior import retain their role assignments across
restart cycles.

### 12.16.7 Examples

```
Directory contents:          Detected roles:
─────────────────────────    ───────────────────────────
base_vectors.fvec            → Base vectors
query_vectors.fvec           → Query vectors
gt_neighbors.ivec            → Neighbor indices
gt_distances.fvec            → Neighbor distances
metadata_content.slab        → Metadata content
predicates.slab              → Metadata predicates
filtered_neighbors.ivec      → Filtered neighbor indices
filtered_distances.fvec      → Filtered neighbor distances
```

```
Directory after import:      Role preserved:
────────────────────────     ─────────────────
_base_vectors.fvec           → Base vectors (still detected)
_query_vectors.fvec          → Query vectors (still detected)
dataset.yaml                 (not a candidate — skipped)
```
