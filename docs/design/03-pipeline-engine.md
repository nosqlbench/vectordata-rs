<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 03 — Pipeline Engine

## 3.1 Architecture

The pipeline engine executes a DAG of processing steps defined in
`dataset.yaml`. It is the primary execution mode for dataset preparation.

### Components

```
dataset.yaml
    ↓ parse
schema.rs → StepDef[]
    ↓ resolve dependencies
dag.rs → ResolvedStep[] (topologically sorted)
    ↓ interpolate variables
interpolate.rs → resolved options per step
    ↓ execute
runner.rs → step-by-step execution with progress tracking
    ↓ delegate
registry.rs → CommandRegistry → CommandOp implementations
    ↓ record
progress.rs → .upstream.progress.yaml
```

## 3.2 Step Lifecycle

Each step in the pipeline goes through:

1. **Option interpolation** — Variables like `${scratch}` are resolved from
   context defaults and workspace paths.

2. **Freshness check** — The progress log is consulted. A step is skipped
   only when all of: status is OK, outputs exist with matching sizes, resolved
   options are unchanged, and no input file has a modification time newer than
   the step's completion timestamp. If stale, the reason is logged (e.g.,
   "stale: input 'source' (base_vectors.hvec) is newer than last run").

3. **Command resolution** — The `run` field (e.g., `"compute knn"`) is looked
   up in the `CommandRegistry` to find the factory function.

4. **Artifact state check** — If the step declares an output path, its
   current state is checked:
   - `Complete` → skip (record in progress log)
   - `Partial` → restart (delete) or resume, depending on `on_partial`
   - `Absent` → proceed normally

5. **Option validation** — Required options from `describe_options()` are
   verified. Missing required options with no defaults cause an error (or a
   dry-run warning).

6. **Buffer-write setup** — If the artifact is `Absent` (or `Partial` with
   restart), the runner rewrites the `output` option to a `_buffer` path
   (e.g., `data.slab` → `data_buffer.slab`). The command writes to the buffer
   path without knowing it. This prevents partially-written files from being
   mistaken for valid artifacts by other processes or a re-run after a crash.

7. **Execution** — `cmd.execute(&options, &mut ctx)` is called. The command
   has full control during execution.

8. **Buffer-write commit** — On success (status Ok or Warning), the runner
   renames the buffer file to the final output path. This is an atomic
   filesystem operation, so the final path either contains the complete
   artifact or does not exist. On error, the buffer is left in place for
   diagnostics and cleaned up on the next run.

9. **Result recording** — The `CommandResult` (status, message, produced
   files, elapsed time) is recorded in the progress log.

10. **Post-execution check** — The artifact state of the final output path
    is re-checked. A non-`Complete` state is an error.

11. **Progress save** — The progress log is atomically persisted to disk.

## 3.3 StreamContext

The shared execution context available to all steps:

```rust
pub struct StreamContext {
    pub workspace: PathBuf,    // directory containing dataset.yaml
    pub scratch: PathBuf,      // temporary files (cleaned after success)
    pub cache: PathBuf,        // persistent intermediates across runs
    pub defaults: IndexMap<String, String>,  // interpolation variables
    pub dry_run: bool,
    pub progress: ProgressLog,
    pub threads: usize,        // available thread count
    pub step_id: String,       // current step identifier
    pub governor: ResourceGovernor,  // resource governance (see §3.12)
}
```

### Workspace directories

- **workspace**: The "home" directory for the dataset. All relative paths
  in step options resolve against this.
- **scratch**: `${workspace}/.scratch/` — temporary files that can be
  deleted after a successful pipeline run. Scratch data is truly disposable.
- **cache**: `${workspace}/.cache/` — persistent intermediates that survive
  across runs. Used by partitioned KNN and segmented predicate evaluation.

  **Cache data is not disposable.** Cached artifacts may be expensive to
  acquire or compute (e.g., KNN partition results over hundreds of GB of
  vectors, or predicate evaluation segments over hundreds of millions
  of metadata records). While cache data is not carried forward into the
  hosted views of the dataset (it is not a dataset facet), it remains
  valuable in-situ because forward rendering stages may be re-executed in
  the future to fill in missing pieces — and cached segments allow those
  re-executions to skip already-completed work. Deleting cache data forces
  full recomputation of all intermediate stages.

  **Cache keys are derived from input file stems, not step IDs.** This
  enables cross-profile segment reuse: when profiles share overlapping
  vector ranges, the smaller profile's cached segments are valid for the
  larger profile. For example, `compute knn` for a 50M profile can reuse
  partition caches originally computed for a 10M profile because both
  process the same base/query vector files. Cache file naming by command:

  | Command | Cache file pattern |
  |---------|-------------------|
  | `compute knn` | `{base_stem}.{query_stem}.range_{start}_{end}.k{k}.{metric}.{ext}` |
  | `evaluate predicates` | `{input_stem}.{pred_stem}.seg_{start:010}_{end:010}.predkeys.slab` |

  Profile ordering (smallest to largest, see §3.6.1) ensures that when
  the 110M profile runs, segments `[0,10M)` through `[99M,100M)` are
  already cached by profiles 10M through 100M.

  **Cache invalidation by mtime**: when an input file's mtime is newer
  than the step's `completed_at` timestamp, the step is considered stale
  and will re-run, potentially recomputing cache segments.

## 3.4 CommandOp Trait

```rust
pub trait CommandOp: Send {
    fn command_path(&self) -> &str;
    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult;
    fn describe_options(&self) -> Vec<OptionDesc>;
    fn command_doc(&self) -> CommandDoc;
    fn describe_resources(&self) -> Vec<ResourceDesc>;
    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState;
}
```

Commands are registered by name in the `CommandRegistry` via factory functions.
The registry maps command paths (e.g., `"compute knn"`) to `fn() -> Box<dyn CommandOp>`.

### `command_doc()`

Returns a `CommandDoc` with `summary` (one-line, used in completions and `.about()`)
and `body` (full markdown, used in `--help` via `.long_about()`). The default
implementation generates a basic doc from the command path and options table, but
all 66+ commands override this with hand-written documentation. See
[07-command-documentation.md](07-command-documentation.md) for the specification.

### `describe_resources()`

Returns a `Vec<ResourceDesc>` declaring which resource types (mem, threads,
readahead, segmentsize, etc.) the command consumes. Used by the CLI completion
system for per-command resource suggestions, by the runner for governance
enforcement warnings, and by the documentation system for the `## Resources`
section. 37 commands that process dataset-scale files provide resource
declarations; the remainder return an empty vector. See
[06-resource-management.md](06-resource-management.md) §6.4.4 for the full list.

## 3.5 Progress Tracking

### File format

Stored as `.upstream.progress.yaml` adjacent to `dataset.yaml`:

```yaml
schema_version: 2
steps:
  import-base:
    status: ok
    message: imported 407314954 records to base_vectors.hvec
    completed_at: '2026-03-05T05:59:50.266068879Z'
    elapsed_secs: 280.878825819
    outputs:
    - path: base_vectors.hvec
      size: 418719772712
      mtime: '2026-03-05T06:00:01.123456789Z'
    resolved_options:
      from: npy
      output: base_vectors.hvec
      source: ../sourcedata/embeddings/text_emb
      facet: base_vectors
```

### Schema version

The progress log carries a `schema_version` field
(`PROGRESS_SCHEMA_VERSION` constant in `progress.rs`). When the version
in the file differs from the code's constant, **all step records are
cleared** and the version is updated. This provides automatic
invalidation when internal algorithms change in ways that affect cache
naming, segment layout, or output content — without requiring the user
to manually delete the progress file.

The schema version is bumped whenever:
- Cache key naming conventions change (e.g., step-ID-based →
  file-stem-based).
- Segment boundary calculation changes.
- Output file formats change in incompatible ways.

### Freshness algorithm

A step is "fresh" when ALL of:
1. It has a progress record with status `ok`
2. All recorded output files exist on disk
3. All recorded output file sizes match the recorded values
4. The resolved options match the recorded options (or the recorded
   options are empty, for backward compatibility)
5. No input file (option value that resolves to an existing file, excluding
   `output`) has a modification time newer than the step's `completed_at`
   timestamp

Additionally, two whole-log invalidation mechanisms exist:

- **Schema version mismatch**: If `schema_version` in the file differs
  from `PROGRESS_SCHEMA_VERSION`, all step records are cleared and the
  user is informed (e.g., "Progress log schema version 1 differs from
  current 2 — clearing all step records").

- **Pipeline definition mtime**: If `dataset.yaml` has a newer mtime
  than the progress log file itself, the entire progress log is
  invalidated (all step records cleared) before any steps are evaluated.
  This ensures that edits to the pipeline definition trigger full
  re-evaluation.

When a step is stale, `check_step_freshness()` returns a reason string
(e.g., "options changed", "output 'foo.ivec' missing", "input 'source'
(base_vectors.hvec) is newer than last run") that is displayed in the
pipeline log.

### Output mtime recording

Step records store the modification time of each output file at
completion as an RFC 3339 timestamp in the `mtime` field. This enables
downstream freshness checks to detect when an input (which is another
step's output) has been regenerated.

### Atomic persistence

Progress is saved atomically via write-to-temp + rename. This prevents
corruption from interrupted writes.

## 3.6 DAG Resolution

Steps declare dependencies via `after: [step-id, ...]`. The DAG resolver
(`dag.rs`, using `petgraph`):

1. Resolves effective IDs (explicit `id` field or derived from `run` field)
2. Checks for duplicate step IDs
3. Detects output path collisions (two steps writing the same file)
4. Builds explicit `after` edges
5. Builds **implicit file-based edges**: if step B's option value matches
   step A's output path, B depends on A (reduces boilerplate)
6. Performs topological sort (detects and reports cycles)
7. Returns `ResolvedStep[]` in execution order

Step ID auto-derivation: `run: "generate ivec-shuffle"` →
`id: "generate-ivec-shuffle"` when no explicit `id` is provided.

### 3.6.1 Profile Expansion and Barriers

Steps marked `per_profile: true` in `dataset.yaml` are expanded into
per-profile copies by `expand_per_profile_steps()`:

1. **Sorted expansion**: Sized profiles are sorted ascending by
   `base_count` (10M, 20M, 30M, …), then `default` (full dataset) last.

2. **ID suffixing**: Template step `evaluate-predicates` becomes
   `evaluate-predicates-10m`, `evaluate-predicates-20m`, etc.

3. **Auto-prefixing**: `output` values are prefixed with
   `profiles/{name}/`. Non-output values that match template output
   filenames are also prefixed (cross-references between per-profile
   steps). Shared inputs (e.g., `metadata_content.slab` produced by a
   non-per-profile step) are NOT prefixed — all profiles read the same
   shared file.

4. **Range injection**: Sized profiles automatically receive a `range`
   option `[0, query_count + base_count)` if one is not already present.

5. **Barrier insertion**: `insert_profile_barriers()` adds synthetic
   barrier steps between profile groups so that ALL steps for one profile
   complete before any step of the next profile begins. This ensures
   cache segments are available for reuse: the 20M profile's
   `evaluate-predicates-20m` can reuse segments `[0,10M)` cached by
   `evaluate-predicates-10m` because the 10M barrier guarantees all 10M
   steps completed first.

## 3.7 Variable Interpolation

Options can reference variables using `${name}` syntax:

| Variable | Source | Example value |
|----------|--------|---------------|
| `${scratch}` | `ctx.scratch` | `.scratch/` |
| `${cache}` | `ctx.cache` | `.cache/` |
| Custom | `ctx.defaults` | User-defined |

Interpolation happens before execution, so commands see fully resolved strings.

## 3.8 Dry-Run Mode

When `--dry-run` is active:
- Steps are evaluated but not executed
- Option validation still runs (catches missing required options)
- Each step prints `WOULD RUN: <command> [options]`
- No files are created, deleted, or modified
- Useful for validating pipeline configuration before committing to execution

## 3.9 Artifact Bound Checks

Format-aware completeness checking via `bound.rs`:

| Format | Check method |
|--------|-------------|
| xvec (fvec, ivec, hvec, ...) | File exists + non-empty = Complete |
| slab | `SlabReader::probe()` — checks pages page exists and has pages |
| Recognized but unchecked (npy, parquet) | `Unknown` — error: no completeness check for this format |
| Unrecognized extension | `Unknown` — error: cannot verify completeness |
| Empty file | Always Partial |
| Missing file | Always Absent |

The `ArtifactState` enum has four variants:

| Variant | Meaning | Pipeline behavior |
|---------|---------|-------------------|
| `Complete` | Passes format-specific bound check | Skip step |
| `Partial` | Exists but incomplete or corrupt | Restart or resume |
| `Absent` | Does not exist | Execute step |
| `Unknown(reason)` | Format unrecognized or has no completeness check | Error (halt pipeline) |

Unrecognized formats produce an error rather than silently assuming
completeness. Commands can override `check_artifact()` to provide
format-specific checks for their particular output types.

## 3.10 Error Handling

- Step failure (status = Error) stops the pipeline immediately
- Progress is saved before returning the error, enabling resume
- The error message is recorded in the progress log's `error` field
- Re-running the pipeline skips completed steps and retries the failed step

## 3.11 Resource Governance Integration

The pipeline engine integrates with the `ResourceGovernor` from
[06-resource-management.md](06-resource-management.md):

### Governor lifecycle

1. **CLI parsing**: `--resources 'mem:25%-50%,threads:4-8'` is parsed into a
   `ResourceBudget` in `run_pipeline()`.
2. **Governor creation**: `ResourceGovernor::new(budget, workspace)` initializes
   effective values at midpoints, creates the `.governor.log`, and installs the
   default `MaximizeUtilizationStrategy`.
3. **Per-step setup**: Before each step, the runner calls
   `ctx.governor.set_step_id(&step.id)` so log entries are correlated.
4. **Enforcement check**: The runner warns if a command processes files >1 GiB
   without `describe_resources()` declarations.
5. **Command interaction**: Commands query `ctx.governor.current("mem")`,
   call `ctx.governor.checkpoint()` at segment boundaries, and respond to
   `ctx.governor.should_throttle()`.

### Governor API (read-only for commands)

| Method | Purpose |
|--------|---------|
| `current(name)` | Get the current effective value for a resource |
| `current_or(name, default)` | Get effective value or a fallback |
| `request(name, preferred)` | Request a specific value; governor clamps to budget range |
| `checkpoint()` | Trigger evaluation if interval elapsed; returns throttle state |
| `should_throttle()` | Check if the governor is signaling throttle |
| `is_emergency()` | Check if emergency flush is active |

### Implementation status

The governor infrastructure (parsing, strategies, logging, evaluation) and
per-command resource declarations (37 commands) are implemented. Command-level
integration — where commands actually consult the governor for buffer sizes,
segment sizing, and thread counts — is planned but not yet complete.

## 3.12 Current Limitations

**Sequential execution only**: Steps execute one at a time. Independent steps
that could safely run in parallel are serialized.

**No timeout mechanism**: A step that hangs or runs indefinitely has no
timeout or watchdog.

**Thread count is static**: `ctx.threads` is set once at pipeline start and
never adjusted based on system load. The governor can publish updated thread
counts, but commands do not yet query them.

**Governor evaluation is synchronous**: The governor evaluates on `checkpoint()`
calls from commands, not on a background thread. A background governor thread
is planned (REQ-RM-12).

**Telemetry**: `SystemSnapshot` samples RSS, page faults, CPU utilization,
I/O throughput, I/O queue depth, active thread count, per-core CPU
percentages, and system page cache size from `/proc` on Linux. All metrics
are charted in the TUI (see §8.6.3).
