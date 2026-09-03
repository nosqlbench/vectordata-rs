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
    base: profiles/base/base_vectors.fvecs
    query: profiles/base/query_vectors.fvecs
    indices: neighbor_indices.ivecs
    distances: neighbor_distances.fvecs
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
output: "${cache}/sorted.ivecs"    # .cache/ directory
```

### Freshness checking — structured provenance (v6)

Staleness is provenance-based, not mtime-based. Each step's progress
record points at a **`ProvenanceNode`** capturing every component that
*could* be used to decide whether the step is stale:

- **Identity** — `step_id`, `command_path`
- **Binary version** — `version_major`, `version_minor`, `version_patch`,
  `git_hash`, `dirty` (parsed from `{CARGO_PKG_VERSION}+{git_hash}[+dirty]`)
- **Resolved options** — sorted `BTreeMap<String, String>`
- **Upstream provenance** — the *address* of each upstream step's node
  in the log's graph, so a relaxed selector cascades correctly through
  the DAG (the hash recurses through addresses). Upstreams are the
  steps a definition names in `after:`. The chain that runs sized
  profiles one at a time and the phase boundaries between them are
  **sequencing edges** (`sequence_after`, filled in memory by
  per-profile expansion, never written): they order the run and are not
  provenance, so dropping a profile or re-running one never makes its
  neighbours stale.

A step is fresh when:
- The output file exists with the recorded size (output paths are
  recorded relative to the workspace, so a record reads the same from
  any working directory), AND
- The step's node hashes under the **active selector** to the same
  value as the node recorded for it, compared over the upstreams
  **both** nodes declare: an upstream only the recorded node names (a
  dependency since removed, or a sequencing edge an older build
  recorded as one) cannot make the step stale, a declared one that
  changed does, and one declared since the record is judged by the
  next rule, not by a hash the record never had, AND
- No step it declares as an upstream (`after`) wrote a recorded
  output after the step's own record — an upstream that ran again,
  in this session or another, or one declared since and run later:
  `after` is the data dependency, so what the upstream wrote is what
  this step reads. An upstream without outputs, a variable set,
  reaches its dependents through their resolved options instead;
  sequencing edges do not count, AND
- No input-role path is newer than the output (the Make rule), which
  carries a cascade across sessions: an upstream rebuilt yesterday
  makes its dependents stale today.

A dry run applies the same rules to its plan: a planned step with
outputs stands for outputs newer than its dependents' records, and a
step whose input a planned step produces is shown as following it.

#### Selectors and presets

The selector is a `ProvenanceFlags` bitset over the components above.
Three presets cover the common cases:

| Preset | Components | Use when |
|--------|------------|----------|
| `config-only` (default) | identity + options + upstream | a rebuilt binary never invalidates a completed step; only configuration and upstreams do |
| `version-aware` | identity + `version_major` + options + upstream | a major version bump should invalidate, minor/patch should not |
| `strict` | every component | the binary itself is part of the key |

Pick a selector at the CLI:

```bash
veks run --provenance config-only     # default: ignore the binary version
veks run --provenance version-aware   # ignore minor/patch/git/dirty
veks run --provenance strict          # every component, binary included
veks run --provenance step_id,command_path,version_major,options
                                      # custom comma-separated component list
```

Tab-completion suggests presets up front; once a comma is typed it
switches to suggesting individual components and excludes any already
chosen.

#### Why the full node is stored

Storing the full node (rather than a single opaque hash) means a later
run can pick a *different* selector against the same record without
re-execution. The `upstream` addresses are what make the relaxation
cascade: when the head step's hash is computed under selector S, each
upstream's node is *also* hashed under S — otherwise the head's hash
would still pull in strictly-computed leaves and the relaxation would
be silently neutered.

#### The provenance graph

Nodes live in one flat, content-addressed table per progress log — the
**`ProvenanceGraph`**. A node's address is its hash under `strict`, so
equal content has equal address, a subtree shared by many dependents
is stored once, and a step that runs again with different inputs gets
a *new* node while the node its dependents were built on stays for as
long as they reference it. Nothing nests: the graph has the same shape
in memory, in the log, and in a sidecar, and its depth on disk is
constant however long an `after:` chain runs.

```yaml
schema_version: 6
provenance:
  1f0c…:              # address = strict hash of the node
    step_id: compute-knn-90m
    command_path: compute knn
    options: {base: profiles/90m/base_vectors.fvecs, …}
    upstream: {compute-knn-89m: 9a41…, count-base: 77e2…, extract-queries: 03bd…}
steps:
  compute-knn-90m:
    status: ok
    provenance: 1f0c…
```

Sidecars (`<cache>/provenance/…provenance.json`) hold a `root` address
and the `nodes` it reaches, so a consumer that keys a cache on its
inputs absorbs them into its own graph and compares roots.

This shape matters: schema 5 nested each upstream's full map inside
every dependent, which made the file quadratic in the length of an
`after:` chain and — past 64 chained steps — deeper than the 128
levels the YAML and JSON parsers will read. A per-profile KNN chain
across 85 profiles reached that limit, at which point the log could not
be loaded and every step looked unrecorded.

#### Schema version

The progress log carries `schema_version: 6`. A schema-5 log is
**migrated** on load, not cleared: its records are kept, and each
record's node is rebuilt with, for every upstream, the *snapshot* the
record carried — the upstream's own components as they were when the
record ran, read one level deep with anything nested further skipped
unparsed. A snapshot that matches the upstream's own record is the
same node; one that differs stays in the graph as the version the
record was built on, so the record hashes exactly as it did before
under every selector. The load message names records built on an
earlier upstream version. Schema 5 also recorded a sharded output under
its logical name with size 0; the migration re-measures such outputs as
their shards. The runner writes the migrated log back at once and keeps
the original beside it as `.upstream.progress.v5.yaml`. A v4-or-earlier
log is still invalidated wholesale (records cleared) so the next run
starts from a known-correct state.

#### Tools

- **`veks run --rerun STEP[,STEP...]`** — run the named steps again
  whether or not they look fresh. Their records are set aside for the
  run, so the freshness check finds nothing recorded and the step
  executes; every step that declares it as an upstream follows, and
  so does anything whose input it rewrites. This is the tool for a build that
  changed what one step computes without changing any option: the
  config-only selector cannot see it, and `strict` would re-run the
  whole ladder, ground truth included.
- **`veks run --explain-staleness`** — walk every step under the active
  selector and print `fresh` / `STALE` plus per-component diff lines
  (`binary_version_major: 1 → 2`, `option 'k': "100" → "200"`,
  `upstream 'extract' provenance changed`, …) without executing
  anything. Upstream maps in the walk use the in-flight currents so the
  cascade reflects the planned re-run order, not a misleading snapshot.
- **`--clean`** removes all generated artifacts and forces
  re-execution from scratch.

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
evaluate-predicates (template, output: metadata_indices.ivvecs)
  → evaluate-predicates       (default, output: profiles/default/metadata_indices.ivvecs)
  → evaluate-predicates-100K  (100K,    output: profiles/100K/metadata_indices.ivvecs)
```

The `phase` field controls ordering within expansion: all phase-0
steps for all profiles run before any phase-1 steps.

An expanded step is **scoped to its profile**: any facet it resolves
by default — a `ground-truth` it does not name, say — comes from that
profile's views, not from `default`. An explicit `profile` option
still wins. Before this, the post-filter step of every sized profile
intersected the census profile's neighbours with its own predicate
results (SRD TS-176).

A step that is not expanded but names a template as its upstream —
a verifier that reads every profile's answer keys, a finalize step
that publishes them all — depends on **every instance** of it: its
`after` fans in to the instances in profile order. Left as the
template's bare id it would name the default instance alone, and a
sized profile added or recomputed later would reach nothing
downstream.

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

Each command exposes `build_version()` returning a string of the form `{CARGO_PKG_VERSION}+{git_hash}[+dirty]`. The runner parses this into the `BinaryVersion` axes (major/minor/patch/git_hash/dirty) recorded in each step's `ProvenanceMap`, so the staleness check can ignore or honor each axis independently per the active selector (see §4.3). At bootstrap time, `veks_version` and `veks_build` are stamped into `dataset.yaml` attributes so consumers can trace which build produced a dataset.

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
