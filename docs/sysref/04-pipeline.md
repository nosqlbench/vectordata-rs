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
| All others | — | Passed as command options |

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

Steps are skipped when:
- The output file exists
- The output is newer than all inputs
- The step's configuration fingerprint matches the progress log

`--clean` removes all generated artifacts and forces re-execution.

---

## 4.4 Per-Profile Expansion

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

## 4.5 Resource Governance

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

## 4.6 Progress and Logging

- **Progress bars** — per-step progress via the UI handle
- **dataset.log** — timestamped log of all step output
- **runlog.jsonl** — machine-readable step execution log
- **variables.yaml** — accumulated pipeline state

Each step's log output is captured in a step buffer, then flushed
to `dataset.log` with timestamps after the step completes.
