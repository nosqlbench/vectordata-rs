# 4. Pipeline Engine

---

## 4.1 Overview

The pipeline engine executes the `upstream.steps` section of
`dataset.yaml` as a directed acyclic graph (DAG). Each step runs a
registered command with options, producing artifacts that downstream
steps depend on.

```bash
veks run dataset.yaml              # execute all pending steps
veks run dataset.yaml --clean      # reset and re-execute
veks run dataset.yaml --dry-run    # show what would run
```

---

## 4.2 Step Definition

```yaml
steps:
  - id: compute-knn
    description: Compute brute-force exact KNN
    run: compute knn
    after: [generate-base, generate-queries]
    per_profile: true
    phase: 0
    base: profiles/base/base_vectors.fvec
    query: profiles/base/query_vectors.fvec
    indices: neighbor_indices.ivec
    distances: neighbor_distances.fvec
    neighbors: 100
    metric: L2
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Unique step identifier |
| `run` | yes | Command path (e.g., `compute knn`) |
| `after` | no | Dependencies (step IDs) |
| `per_profile` | no | Expand for each profile (default: false) |
| `phase` | no | Ordering group within per-profile expansion |
| `description` | no | Human-readable purpose |
| `finalize` | no | If `true`, step runs in the finalization pass (default: false) |
| All others | — | Passed as command options |

Steps marked `finalize: true` are separated from compute steps and run in a dedicated final pass after all compute phases complete. This ensures finalization steps (e.g., `generate-dataset-json`, `generate-variables-json`, `generate-dataset-log-jsonl`, `generate-merkle`, `generate-catalog`) see the full set of profiles and artifacts, including any profiles created by partition expansion.

```yaml
  - id: generate-catalog
    run: generate catalog
    after: [generate-merkle]
    finalize: true
```

---

## 4.3 Execution Model

1. **Parse** — Load `dataset.yaml`, resolve variables
2. **Expand** — `per_profile: true` steps are cloned for each profile,
   with output paths prefixed by `profiles/<name>/`
3. **Topologize** — Order steps by `after` dependencies
4. **Execute** — Run each step sequentially, skip if output is fresh
5. **Sync** — Write pipeline variables to `dataset.yaml`

### Variable interpolation

Step options can reference variables:

```yaml
count: "${vector_count}"          # from variables section
range: "[0,${base_count})"        # computed by earlier step
output: "${cache}/sorted.ivec"    # .cache/ directory
```

### Freshness checking

Staleness is fingerprint-based, not mtime-based. A step is skipped when:
- The output file exists and has non-zero size
- The step's configuration fingerprint (which chains through the DAG and includes the build version) matches the progress log

The fingerprint incorporates all step options, dependency fingerprints, and the build version, so any code change or option change propagates staleness through downstream steps. `--clean` removes all generated artifacts and forces re-execution.

---

## 4.4 Execution Phases

The pipeline executes in four phases:

1. **Phase 1 — Core + resolved per-profile steps.** All steps whose dependencies are already satisfiable run first. This includes non-per-profile steps and per-profile steps for profiles that exist at bootstrap time.

2. **Phase 2 — Deferred sized expansion.** When a step like `count-vectors` resolves `base_count`, size-bucketed profiles (e.g., `100K`, `250K`) become concrete. Per-profile templates are re-expanded for these new profiles and appended to the DAG.

3. **Phase 3 — Partition expansion.** When `partition-profiles` creates new profiles from metadata labels, per-profile templates (e.g., `compute-knn`) are expanded again for each partition profile. The engine calls `build_dag_partial` to splice new steps into the running DAG.

4. **Finalization.** Steps with `finalize: true` are held back and run once after all three compute phases complete. They are added to the DAG via a final `build_dag_partial` call, ensuring they see every profile and artifact.

---

## 4.5 Per-Profile Expansion

Steps with `per_profile: true` are templates. The engine expands them
once per profile, prefixing output paths:

```
evaluate-predicates (template, output: metadata_indices.ivvec)
  → evaluate-predicates       (default, output: profiles/default/metadata_indices.ivvec)
  → evaluate-predicates-100K  (100K,    output: profiles/100K/metadata_indices.ivvec)
```

The `phase` field controls ordering within expansion: all phase-0
steps for all profiles run before any phase-1 steps.

---

## 4.6 Resource Governance

The resource governor limits memory and thread usage per step:

```yaml
# In step options
mem: 4G           # memory budget
threads: 16       # thread limit
```

Steps declare resource requirements via `describe_resources()`.
The governor prevents system lockups during large-scale operations
(e.g., 1M × 1M KNN computation).

---

## 4.7 Build Versioning

Each command exposes `build_version()` returning a string of the form `{CARGO_PKG_VERSION}+{git_hash}`. This version is included in the step fingerprint — recompiling with code changes automatically invalidates cached results without manual `--clean`. At bootstrap time, `veks_version` and `veks_build` are stamped into `dataset.yaml` attributes so consumers can trace which build produced a dataset.

```yaml
attributes:
  veks_version: "0.9.0"
  veks_build: "0.9.0+a3f7c2d"
```

---

## 4.8 Progress and Logging

- **Progress bars** — per-step progress via the UI handle
- **dataset.log** — timestamped log of all step output
- **runlog.jsonl** — machine-readable step execution log
- **variables.yaml** — accumulated pipeline state

Each step's log output is captured in a step buffer, then flushed
to `dataset.log` with timestamps after the step completes.
