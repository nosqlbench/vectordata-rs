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
| `import_vectors` | Vectors | Source is native xvec single file | `import` (npy/parquet/dir → xvec) |
| `all_vectors` | Vectors | *(terminal — always resolves to import or source)* | alias chain: fetch → import → source |
| `dedup` | Ordinals | `--no-dedup` | `compute dedup` |

### Count slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `vector_count` | Variable | Never (always needed) | `set variable` on all_vectors |
| `base_count` | Variable | No self-search (base = all_vectors) | `set variable` on base_vectors |

### Query slots

| Slot | Type | Identity when | Materialized as |
|------|------|---------------|-----------------|
| `shuffle` | Ordinals | No self-search (separate query file or no queries) | `generate ivec-shuffle` |
| `query_vectors` | Vectors | Provided as native xvec | `import` or `transform *-extract` (self-search) |
| `base_vectors` | Vectors | Not self-search (all_vectors used directly) | `transform *-extract` (self-search range) |

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
| `knn` | GroundTruth | Pre-computed GT provided, or no query vectors | `compute knn` (per_profile) |
| `filtered_knn` | GroundTruth | No metadata, `--no-filtered`, or no queries | `compute filtered-knn` (per_profile) |

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
    FETCH --> IMPORT_V["import_vectors<br/><i>Identity if native xvec</i>"]:::slot
    IMPORT_V --> ALL_V["all_vectors"]:::artifact

    ALL_V --> DEDUP["dedup<br/><i>Identity if --no-dedup</i>"]:::slot
    ALL_V --> SET_VC["vector_count<br/>set variable"]:::slot

    %% ── Query chain ──────────────────────────────────────────────
    SRC_QUERY --> IMPORT_Q["import_query<br/><i>Identity if native xvec</i>"]:::slot

    SET_VC --> SHUFFLE["shuffle<br/><i>Identity if separate query</i>"]:::slot
    ALL_V --> EXT_Q["extract_query<br/><i>Identity if separate query</i>"]:::slot
    SHUFFLE --> EXT_Q
    IMPORT_Q --> QUERY_V

    EXT_Q --> QUERY_V["query_vectors"]:::artifact

    ALL_V --> EXT_B["extract_base<br/><i>Identity if not self-search</i>"]:::slot
    SHUFFLE --> EXT_B
    EXT_B --> BASE_V["base_vectors"]:::artifact
    BASE_V --> SET_BC["base_count<br/><i>Identity if not self-search</i>"]:::slot

    %% ── Metadata chain (entire subgraph Absent when no metadata) ─
    SRC_META --> IMPORT_M["import_metadata<br/><i>Identity if native slab</i>"]:::slot
    IMPORT_M --> META_ALL["metadata_all"]:::artifact

    META_ALL --> EXT_META["extract_metadata<br/><i>Identity if not self-search</i>"]:::slot
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
    SET_BC --> KNN["compute knn<br/><i>Identity if GT provided<br/>Absent if no queries</i>"]:::slot
    QUERY_V --> KNN
    BASE_V --> KNN
    KNN --> GT["neighbor_indices<br/>neighbor_distances"]:::artifact

    PRED_IDX --> FKNN["compute filtered-knn<br/><i>Absent if no metadata<br/>or --no-filtered</i>"]:::slot
    SET_BC --> FKNN
    QUERY_V --> FKNN
    BASE_V --> FKNN
    FKNN --> FGT["filtered_neighbor_indices<br/>filtered_neighbor_distances"]:::artifact

    %% ── Profile assembly ─────────────────────────────────────────
    BASE_V --> PROF[/"dataset.yaml<br/>profiles"/]:::output
    QUERY_V --> PROF
    META_ALIGNED --> PROF
    PREDS --> PROF
    GT --> PROF
    FGT --> PROF
    PRED_IDX --> PROF
    DEDUP --> PROF

    %% ── Styling ──────────────────────────────────────────────────
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef slot fill:#fff3e0,stroke:#ef6c00,stroke-dasharray:5 5
    classDef artifact fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

**Reading the graph:**
- **Parallelogram** nodes are user-provided inputs.
- **Dashed** nodes are slots — each independently resolves to either a
  materialized step or an identity alias.
- **Solid green** nodes are artifacts — the resolved output path,
  regardless of whether it was materialized or aliased.
- The *italic annotation* on each slot states its identity condition.
- When an input is `Option::None`, its entire downstream subgraph
  vanishes from the emitted YAML.

---

## 12.5 Identity Collapse Examples

### Example 1: Minimal — native fvec, no queries, no metadata

```
fetch_vectors     → Identity (no URL)
import_vectors    → Identity (native fvec)
all_vectors       → /data/base.fvec (alias to source)
dedup             → Materialized (compute dedup)
vector_count      → Materialized (set variable)
shuffle           → Identity (no self-search)
extract_query     → Absent (no queries)
extract_base      → Identity (base = all_vectors)
base_vectors      → /data/base.fvec (alias)
metadata chain    → Absent (no metadata)
knn               → Absent (no queries)
filtered_knn      → Absent (no metadata)
```

Emitted steps: `set-vector-count`, `dedup-vectors`. Two steps total.

### Example 2: Maximal — foreign format, self-search, metadata, all GT

```
fetch_vectors     → Materialized (fetch bulkdl)
import_vectors    → Materialized (import npy → mvec)
all_vectors       → ${cache}/all_vectors.mvec
dedup             → Materialized (compute dedup)
vector_count      → Materialized (set variable)
shuffle           → Materialized (generate ivec-shuffle)
extract_query     → Materialized (transform mvec-extract [0, query_count))
extract_base      → Materialized (transform mvec-extract [query_count, N))
base_count        → Materialized (set variable)
import_metadata   → Materialized (import parquet → slab)
extract_metadata  → Materialized (transform slab-extract, shuffle-aligned)
survey            → Materialized (survey)
predicates        → Materialized (synthesize predicates)
pred_indices      → Materialized (compute predicates, per_profile)
knn               → Materialized (compute knn, per_profile)
filtered_knn      → Materialized (compute filtered-knn, per_profile)
```

All slots materialized. This is the laion400m-img-search pattern.

### Example 3: Native xvec base + separate native query, with metadata

```
import_vectors    → Identity (native)
all_vectors       → base.fvec (alias)
dedup             → Materialized
shuffle           → Identity (separate query, no self-search)
extract_query     → Identity (query provided directly)
query_vectors     → query.fvec (alias)
extract_base      → Identity (not self-search)
base_vectors      → base.fvec (alias)
import_metadata   → Materialized (parquet → slab)
extract_metadata  → Identity (no shuffle, ordinals aligned)
metadata_content  → ${cache}/metadata_all.slab (alias)
survey            → Materialized
predicates        → Materialized
pred_indices      → Materialized (per_profile)
knn               → Materialized (per_profile)
filtered_knn      → Materialized (per_profile)
```

Import steps collapse; shuffle/extract collapse; metadata extract
collapses. But all compute steps still run.

---

## 12.6 Type-Safe Optional Axes

The three optional axes — query vectors, metadata, and ground truth —
use a typed `Option` model in the generator:

```rust
struct PipelineSlots {
    // Required
    all_vectors: Artifact,
    vector_count: Artifact,
    dedup: Artifact,

    // Query axis (None → no KNN, no filtered-KNN)
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
fn emit_pipeline(slots: &PipelineSlots) -> Vec<StepDef> {
    let mut steps = Vec::new();

    // Vector chain — emit in dependency order
    if let Materialized { .. } = slots.all_vectors {
        steps.push(/* import step */);
    }
    steps.push(/* set-vector-count — always */);
    if let Materialized { .. } = slots.dedup {
        steps.push(/* compute dedup */);
    }

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
| `--name` | string | Dataset name (required) |
| `-o, --output` | path | Output directory (required) |
| `--base-vectors` | path | Base vector source (file or directory) |
| `--query-vectors` | path | Separate query vectors (file or directory) |
| `--self-search` | flag | Extract queries from base via shuffle (default when no `--query-vectors`) |
| `--query-count` | int | Number of query vectors in self-search mode (default: 10000) |
| `--metadata` | path | Metadata source (file or directory) |
| `--ground-truth` | path | Pre-computed ground truth indices (ivec) |
| `--ground-truth-distances` | path | Pre-computed ground truth distances (fvec) |
| `--metric` | string | Distance metric for KNN (default: L2) |
| `--neighbors` | int | Number of neighbors for KNN (default: 100) |
| `--seed` | int | Random seed for shuffle (default: 42) |
| `--description` | string | Dataset description |
| `--no-dedup` | flag | Collapse dedup slot to identity |
| `--no-filtered` | flag | Force filtered_knn to Absent |
| `--force` | flag | Overwrite existing dataset.yaml |

### Implied defaults

- `--base-vectors` without `--query-vectors` → self-search mode
- `--metadata` present → full predicate chain unless `--no-filtered`
- `--ground-truth` → KNN slot collapses to Identity (alias to provided files)

---

## 12.9 Output Convention

Resolved artifact paths follow the workspace layout (§5.7):

| Artifact | Path | Cache/final |
|----------|------|-------------|
| Imported vectors | `${cache}/all_vectors.mvec` | cache |
| Dedup ordinals | `${cache}/dedup_ordinals.ivec` | cache |
| Dedup report | `${cache}/dedup_report.json` | cache |
| Shuffle permutation | `${cache}/shuffle.ivec` | cache |
| Imported metadata | `${cache}/metadata_all.slab` | cache |
| Metadata survey | `${cache}/metadata_survey.json` | cache |
| Base vectors | `profiles/base/base_vectors.mvec` | final |
| Query vectors | `profiles/base/query_vectors.mvec` | final |
| Metadata content | `profiles/base/metadata_content.slab` | final |
| Predicates | `predicates.slab` | final |
| KNN indices | `profiles/${profile}/neighbor_indices.ivec` | final |
| KNN distances | `profiles/${profile}/neighbor_distances.fvec` | final |
| Filtered indices | `profiles/${profile}/filtered_neighbor_indices.ivec` | final |
| Filtered distances | `profiles/${profile}/filtered_neighbor_distances.fvec` | final |
| Predicate indices | `profiles/${profile}/metadata_indices.slab` | final |
