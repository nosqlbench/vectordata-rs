<!-- Copyright (c) nosqlbench contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 01 — System Overview

## 1.1 Purpose

Veks is a CLI tool for the full lifecycle of vector dataset preparation for ANN
benchmarking. It handles downloading source data, converting between formats,
importing into canonical layouts, computing ground truth (brute-force KNN),
generating predicated datasets with metadata and filter predicates, and
analyzing results.

## 1.2 Workspace Structure

```
vectordata-rs/                     Cargo workspace root
├── Cargo.toml                     workspace members: vectordata, veks, slabtastic
│
├── vectordata/src/                shared types, I/O, dataset spec, format codecs
│   ├── dataset/                   dataset.yaml schema, profiles, facets, catalogs, windows
│   ├── formats/                   binary codecs and vernacular renderers
│   │   ├── mnode/                 MNode — metadata record codec (29 type tags)
│   │   └── pnode/                 PNode — predicate tree codec with eval engine
│   ├── merkle/                    Merkle tree core (.mref/.mrkl wire-compatible)
│   ├── transport/                 HTTP transport with retry, chunked Range downloads
│   └── cache/                     CachedChannel — read-through cache with Merkle verify
│
├── slabtastic/src/                page-aligned record container library
│   └── cli/                       `slab` CLI subcommands (import, export, check, etc.)
│
└── veks/src/                      the CLI application
    ├── cli/                       shell completion generation
    ├── formats/                   format codecs, readers (npy, parquet, slab, xvec), writers
    │   ├── reader/                format-specific readers
    │   └── writer/                format-specific writers
    ├── datasets/                  dataset consumer commands (list, cache, curlify, prebuffer)
    ├── prepare/                   dataset producer commands (import, stratify, catalog, cache-compress)
    ├── catalog/                   dataset catalog generation and resolution
    ├── ui/                        UI-agnostic eventing layer (plain, test, ratatui sinks)
    └── pipeline/                  DAG execution engine
        ├── analyze/               analysis and diagnostic commands
        ├── bulkdl/                bulk parallel file downloader (URL template expansion)
        ├── convert/               format-to-format conversion
        ├── import/                facet-aware data import orchestration
        ├── predicate/             predicate generation subsystem
        └── commands/              67 CommandOp implementations (see §1.5)
```

## 1.3 Crate Responsibilities

### vectordata

The shared foundation crate. Provides:

- **`VectorReader<T>` trait** — uniform access to vector files by ordinal
  (`get_slice(index) → &[T]`, `count() → usize`, `dimension() → usize`)
- **`MmapVectorReader<T>`** — memory-mapped implementation for xvec files
  (fvec, ivec, mvec, etc.). Opens the file once, mmaps it, and provides O(1)
  random access to any vector by ordinal. Provides `get_slice()` for
  zero-copy access and `prefetch_range()` / `prefetch_pages()` with
  `MADV_WILLNEED` for page cache warming.
- **`HttpVectorReader<T>`** — remote HTTP(S) reader using Range requests for
  lazy vector access from URLs. Reads header + size on open, then per-vector
  Range requests on access.
- **`TestDataGroup`** / **`TestDataView`** — high-level dataset access API.
  Loads `dataset.yaml`, resolves profiles, and provides `Arc<dyn VectorReader>`
  accessors for each facet. Transparently handles filesystem and HTTP sources.
- **`DataSource` enum** — `FileSystem(PathBuf)` | `Http(Url)` for unified
  source resolution
- **Format codecs** — MNode/PNode/ANode binary codecs and vernacular text
  renderers/parsers. These live in `vectordata::formats` and are re-exported
  by the veks crate for CLI use.
- **Merkle core** — `MerkleRef` and `MerkleState` for wire-compatible integrity
  verification (.mref/.mrkl files). Java-compatible BitSet encoding.
- **Chunked transport** — `HttpTransport` with retry policy, connection pooling,
  and progress tracking.
- **Cache-backed file channel** — `CachedChannel` for transparent read-through
  caching with Merkle verification and crash recovery.
- **`dataset` module** — `dataset.yaml` parsing, `DatasetConfig`, `DSProfile`,
  profile inheritance, views, aliases, windows, sized expansion, facet
  definitions, pipeline step schema, and catalog support.

### slabtastic

Page-aligned record container library. Provides:

- **`SlabWriter`** — appends variable-length records into fixed-size pages,
  building a page index ("pages page") for ordinal-addressed retrieval
- **`SlabReader`** — opens a slab file, reads the page index, and retrieves
  records by ordinal via `get(ordinal) → Result<Vec<u8>>`
- **`Page`** — a single fixed-size page containing packed records with a
  directory of (offset, length) entries
- **`PagesPage`** — the file-level table of contents mapping ordinal ranges
  to page file offsets
- **`slab` CLI** — standalone binary for slab file operations (import, export,
  append, rewrite, check, get, explain, analyze, namespaces)

Key properties: page-aligned I/O, ordinal-addressed records, opaque payloads,
append-friendly, namespace support, probe-based completeness checks.

Four reader access modes:
1. **Point get** — interpolation search (O(1) expected) + mmap slice
2. **Batch iteration** — sequential page reads for streaming
3. **Sink read** — sequential write of all records to any `Write` sink
4. **Multi-batch concurrent** — scoped thread parallel batch reads

### veks

The CLI application. Encompasses:

- **CLI layer** — clap-based command parsing with root-level subcommands:
  `run`, `script`, `pipeline`, `datasets`, `completions`, `help`
- **UI eventing layer** — decoupled event algebra (`UiEvent` enum) dispatched
  to pluggable sinks (`UiSink` trait) via an ergonomic handle facade
  (`UiHandle`). No pipeline code imports rendering libraries directly.
  Three concrete sinks: `PlainSink` (non-TTY), `TestSink` (test harness),
  `RatatuiSink` (full-screen TUI with resource charts).
- **Format codecs** — Re-exports MNode/PNode/ANode codecs from vectordata.
  Adds parquet-to-MNode compiler (arrow-dependent), format readers (npy,
  parquet, slab, xvec), and format writers.
- **Pipeline engine** — DAG-ordered step execution with skip-if-fresh,
  artifact bound checks, progress tracking, dry-run, variable interpolation
  (with `$$` escape for literal `$`), resource governance via
  `ResourceGovernor`
- **67 pipeline commands** — import, convert, compute KNN, generate
  predicates, slab operations, analysis, merkle trees, etc. Each provides
  built-in markdown documentation (`command_doc()`) and resource declarations
  (`describe_resources()`).

## 1.4 Execution Modes

| Mode                          | Entry point      | Description                                            |
|-------------------------------|------------------|--------------------------------------------------------|
| `veks run dataset.yaml`      | Pipeline runner  | Execute DAG-ordered steps from dataset.yaml            |
| `veks pipeline <group> <cmd>`| Direct command   | Execute a single pipeline command with CLI args        |
| `veks script dataset.yaml`   | Script export    | Emit a pipeline as an equivalent shell script          |
| `veks prepare <subcmd>`      | Preparation      | Import, stratify, catalog, compress datasets           |
| `veks datasets <subcmd>`     | Inventory        | Browse, search, cache, prebuffer, curlify datasets     |
| `veks check`                 | Validation       | Pre-flight checks for dataset readiness                |
| `veks publish`               | Publishing       | Publish dataset to S3                                  |
| `veks help [command]`        | Documentation    | Display help for pipeline commands and groups          |
| `veks completions`           | Shell setup      | Generate shell completions for bash/zsh/fish           |

### 1.4.1 Shorthand Dispatch and Global Name Uniqueness

**All subcommand names MUST be globally unique across all command
groups.** This is a normative requirement that enables shorthand
dispatch: any subcommand can be invoked directly from the root without
its group prefix when it is unambiguous.

For example, `veks import` dispatches to `veks prepare import` because
`import` is a unique subcommand name. Similarly, `veks list` dispatches
to `veks datasets list`.

The dispatch order is:

1. Match against known root-level commands (`run`, `prepare`, etc.)
2. Look up in the shorthand table (subcommands of `prepare` and `datasets`)
3. Fall through to the pipeline command registry (`veks pipeline`)

**Shorthand names are NOT included in shell completions** — tab
completion only shows the canonical group-prefixed forms. This keeps
the completion output clean while still allowing experienced users to
type the shorter form.

**Enforcement**: when adding a new subcommand to any group, verify
that its name does not collide with any existing subcommand in any
other group. If a collision is detected, one of the conflicting
commands must be renamed before the change is accepted.

Current subcommand mapping:

| Shorthand | Canonical form |
|-----------|---------------|
| `import` | `prepare import` |
| `stratify` | `prepare stratify` |
| `catalog` | `prepare catalog` |
| `cache-compress` | `prepare cache-compress` |
| `cache-uncompress` | `prepare cache-uncompress` |
| `list` / `ls` | `datasets list` |
| `cache` | `datasets cache` |
| `curlify` | `datasets curlify` |
| `prebuffer` | `datasets prebuffer` |

## 1.5 CLI Design Principles

The following principles govern the naming, grouping, and behavior of all
veks commands. These are normative requirements — new commands and
refactoring of existing commands must conform.

### 1.5.1 Structural Consistency

- **Group names are verbs** that describe the action category: `analyze`,
  `compute`, `generate`, `transform`, `verify`, `download`, `query`.
- **Subcommand names are nouns or noun-phrases** that specify the target:
  `knn`, `shuffle`, `ordinals`, `knn-groundtruth`.
- **Command paths read as verb-noun phrases**: `compute knn`, `analyze
  stats`, `verify knn-groundtruth`, `transform extract`.
- **Arity consistency**: every pipeline command has a group. No bare
  commands at the pipeline root. Single-operation groups (like `merkle`)
  are acceptable when the group represents a cohesive tool domain.

### 1.5.2 Semantic Consistency

- **One concept, one name, one location.** If a name is used, it means
  the same thing everywhere. `import` does not mean "bootstrap a dataset"
  in one context and "convert a file format" in another.
- **Leaf commands are self-describing without parent context.** `verify
  knn-groundtruth` is clear anywhere; `verify knn` requires knowing
  what aspect of KNN is being verified.
- **Format-specific variants collapse to parameterized commands.** One
  `transform extract` with format auto-detection, not N format-specific
  commands. Exceptions: when the operation is structurally different
  (e.g., `transform extract-slab` vs `transform extract`).

### 1.5.3 Perceptual Consistency

- **Pipeline commands vs workflow commands never share a namespace.**
  Pipeline commands are building blocks (`CommandOp` implementations).
  Workflow commands (`prepare bootstrap`, `prepare stratify`) orchestrate
  those building blocks. They live in separate command trees.
- **Infrastructure commands are visually separated from data processing.**
  `state set`, `state clear`, `config show` are meta-operations, not
  data transformations.
- **Usage modes lay out naturally.** A user exploring the tool encounters
  workflow commands first (`prepare`, `datasets`, `run`), then discovers
  pipeline building blocks (`pipeline analyze`, `pipeline compute`).

### 1.5.4 Zero Ambiguity

- **All subcommand names are globally unique** across all command groups
  (see §1.4.1).
- **Sibling subcommands MUST NOT share a common prefix.** Names like
  `list` and `list-cache` under the same parent create ambiguous
  tab-completion: typing `list<TAB>` cannot complete unambiguously.
  Instead, use flags on the base command (e.g., `list --cached`) or
  choose distinct names. This rule applies at every level of the
  command tree.
- **Argument semantics are uniform for a given type.** A `--source` path
  means the same thing in every command that accepts one. Semantic
  disambiguation through naming is acceptable (e.g., `--base` vs
  `--query` for different vector roles).

## 1.6 Command Categories

### Workflow commands

High-level commands that orchestrate pipeline steps, manage dataset
configuration, and provide user-facing workflows. These are the primary
entry points for dataset producers and consumers.

| Group | Commands | Purpose |
|---|---|---|
| `prepare` | `bootstrap`, `stratify`, `catalog` | Create and configure datasets |
| `datasets` | `list`, `list-cache`, `prebuffer`, `script curl` | Browse, cache, download datasets |
| `config` | `show`, `init`, `list-mounts` | Tool and storage configuration |
| `run` | *(top-level)* | Execute pipeline from dataset.yaml |
| `script` | *(top-level)* | Emit pipeline as shell script |
| `check` | *(top-level)* | Pre-flight validation |
| `publish` | *(top-level)* | Publish to S3 |

### Pipeline commands

Building-block commands that implement `CommandOp` and can be composed
into DAG-ordered pipelines via `dataset.yaml`. Also invocable directly
via `veks pipeline <group> <command>`.

| Group | Purpose | Commands |
|---|---|---|
| `analyze` | Examine data, produce reports, inspect structure | `check-endian`, `compare`, `compute-info`, `describe`, `explore`, `file`, `find`, `flamegraph`, `histogram`, `model-diff`, `plot`, `predicate`, `profile`, `select`, `slice`, `stats`, `survey`, `verify-knn`, `verify-profiles`, `zeros` |
| `cache` | Pipeline cache management | `compress`, `uncompress` |
| `compute` | Expensive numerical computation | `evaluate-predicates`, `filtered-knn`, `knn`, `sort` |
| `download` | Fetch from external sources | `bulk`, `huggingface` |
| `generate` | Create new data from parameters/models | `dataset`, `derive`, `from-model`, `predicated`, `predicates`, `shuffle`, `sketch`, `vectors` |
| `merkle` | Merkle integrity tree operations | `create`, `diff`, `path`, `spoilbits`, `spoilchunks`, `summary`, `treeview`, `verify` |
| `query` | Search and filter records | `json`, `records` |
| `state` | Pipeline variable management | `clear`, `set` |
| `transform` | Reshape, convert, subset data | `convert`, `extract`, `ordinals` |
| `verify` | Correctness validation | `knn-groundtruth`, `predicate-results` |
| `pipeline` | Pipeline orchestration | `require` |

### Pipeline dependencies (`pipeline require`)

Pipeline files can declare dependencies on other pipeline files using
the `pipeline require` command. This enables multi-file pipeline
architectures where shared work (e.g., downloading source data) is
defined once and required by multiple dataset pipelines.

```yaml
# dataset.yaml
upstream:
  steps:
    - id: require-downloads
      run: pipeline require
      description: Ensure source data is downloaded
      file: ../_sourcedata/download.yaml

    - id: convert-vectors
      run: transform convert
      after: [require-downloads]
      source: ../_sourcedata/embeddings/img_emb/
      output: ${cache}/all_vectors.mvec
```

**Semantics:**

- The required pipeline runs in its own workspace (the directory
  containing the required YAML file), with its own `dataset.log`
  progress state, `.scratch`, `.cache`, and `variables.yaml`.
- If all steps in the required pipeline are already complete (from a
  previous run, whether invoked directly or via `pipeline require`),
  the command returns immediately.
- If steps are incomplete, the full pipeline runner is invoked on
  the required file. The required pipeline's internal resume logic
  picks up where it left off.
- The required pipeline's artifact state is `PartialResumable` when
  incomplete — the runner never deletes its output.
- Running the required pipeline directly (`veks run download.yaml`)
  and having it invoked via `pipeline require` produce identical
  progress state. They are interchangeable.

**Use case: shared source data**

A download pipeline fetches hundreds of GB of source files that feed
multiple dataset preparation pipelines:

```
_sourcedata/
├── download.yaml          ← downloads img/text/metadata shards
├── embeddings/img_emb/    ← 410 .npy files
├── embeddings/text_emb/   ← 410 .npy files
└── embeddings/metadata/   ← 410 .parquet files

img-search/
├── dataset.yaml           ← requires download.yaml, then processes
└── ...

text-search/
├── dataset.yaml           ← also requires download.yaml
└── ...
```

### Artifact integrity and failsafe checks

The pipeline runner MUST NOT report success for a step unless the
output has been affirmatively verified. Specifically:

- **xvec files** (fvec, mvec, ivec, etc.): The bound checker reads
  the dimension from the first 4 bytes, computes the record stride
  (`4 + dim × element_size`), and verifies the file size is an exact
  multiple. A file with trailing bytes is `Partial` — the conversion
  was interrupted.

- **Verified count**: Every pipeline command that produces an xvec or
  slab output file **must** write a `verified_count:<filename>` entry
  to `variables.yaml` (via `variables::set_and_save`) after successful
  completion. The bound checker reads this entry and compares it against
  the file's actual record count. A record-aligned file without a
  verified count is treated as `Partial` (failsafe: the write may have
  been interrupted at a record boundary). The workspace is located by
  walking up from the output file's directory until `variables.yaml`
  is found (supporting outputs in `.cache/`, `profiles/name/`, etc.).

- **Post-write verification**: After writing an xvec file, commands
  verify that `file_size == count × stride` before reporting success.
  A mismatch is an error.

- **Progress log authority**: The progress log records success or
  failure affirmatively. A step without a progress record is treated
  as incomplete regardless of what the output file looks like on disk.
  The bound checker provides a secondary check that catches truncated
  files from killed processes.

- **Failsafe principle**: False positives (reporting success when
  data is incomplete) are worse than false negatives (re-running a
  step that was already complete). All checks err on the side of
  re-running.

### Parallel mmap conversion

When converting a directory of npy files to an xvec format, the
`transform convert` command uses a parallel mmap write path:

1. Scan all npy file headers → per-file row counts and cumulative offsets
2. Pre-allocate the output file to exact size via `posix_fallocate` + mmap
3. Load, convert, and write files in parallel via rayon — each file writes
   to its computed byte offset in the mmap, with no coordination needed
4. Periodic `msync(MS_ASYNC)` flushes dirty pages asynchronously
5. Final `flush()` ensures all data reaches disk

The `SharedMmapWriter` is safe for concurrent disjoint writes because
each record ordinal maps to a non-overlapping byte range in the file.

### Artifact state model

Pipeline commands report the state of their output artifacts:

| State | Meaning | Runner action |
|-------|---------|---------------|
| `Complete` | Output exists and is valid | Skip step |
| `Partial` | Incomplete, not resumable | Delete output, restart |
| `PartialResumable` | Incomplete, command handles resume | Keep output, re-run |
| `Absent` | Output does not exist | Run step |

Commands that accumulate output across runs (like `download bulk` and
`pipeline require`) return `PartialResumable` so the runner never
destroys partially-complete work.

### Root-level command summary

| Command          | Category             | Description                                        |
|------------------|----------------------|----------------------------------------------------|
| `veks run`       | Pipeline runner      | Execute a full pipeline from dataset.yaml          |
| `veks pipeline`  | Pipeline (direct)    | Execute a single pipeline command with CLI args    |
| `veks script`    | Pipeline (export)    | Emit a pipeline as an equivalent shell script      |
| `veks prepare`   | Preparation          | Bootstrap, stratify, catalog datasets              |
| `veks datasets`  | Consumer             | Browse, search, cache, download datasets           |
| `veks config`    | Configuration        | Tool and storage configuration                     |
| `veks check`     | Validation           | Pre-flight checks for dataset readiness            |
| `veks publish`   | Publishing           | Publish dataset to S3                              |
| `veks help`      | Documentation        | Pipeline command documentation                     |
| `veks completions`| Shell setup         | Shell completion generation                        |

## 1.6 Visualization Command Requirements

The `veks visualize` command group (`explore-histograms`, `plot`, `pca`)
provides interactive TUI-based data exploration. These commands are
normatively required to satisfy the following:

### 1.6.1 Incremental Streaming with TUI Preview

All visualization commands MUST be incremental and TUI-friendly.
Users must see results appear progressively — not wait for an entire
computation to finish before seeing anything. Specifically:

- Data is processed in **partitions** (e.g., 1M vectors per partition).
- After each partition, the TUI updates with the current approximation.
- The user can press `q` at any time to stop with the current result.
- For PCA: eigenvectors are recomputed after each partition; the scatter
  plot refines as more data is processed.
- For histograms: bins update as more samples are incorporated.

### 1.6.2 Cached Intermediate Artifacts

All visualization commands MUST cache their intermediate products so
that recomputation is avoided when the source data has not changed:

- Cache location: `.cache/viz/` next to the source file, or in the
  centrally configured cache directory (from `settings.yaml`) for
  catalog datasets.
- Per-partition statistics are cached individually so that adding more
  partitions only computes the new ones.
- Final projected/binned results are cached for instant reload.
- Cache invalidation: mtime comparison — if the source file is newer
  than the cache, the cache is stale and recomputed.

### 1.6.3 SIMD Optimization

All visualization commands that perform vector arithmetic (dot products,
distance computations, accumulations) MUST use `simsimd` for the inner
loops where applicable. The key hot paths are:

- **PCA covariance-vector product**: the `(x - μ) · v` dot product in
  power iteration is called O(N × iterations) times and dominates
  runtime for large datasets.
- **PCA projection**: the `(x - μ) · eigenvector` dot products.
- **Histogram binning**: not SIMD-critical (single scalar per vector).

Use `simsimd::SpatialSimilarity::dot` for f32 vectors. For f16 (mvec)
sources, convert individual vectors to f64 on the fly via the
`AnyVectorReader` abstraction — do NOT bulk-convert the entire file.

### 1.6.4 Format-Agnostic Access

All visualization commands MUST support all primary vector formats
(fvec, mvec, dvec) through the `AnyVectorReader` abstraction. Format
is auto-detected from the file extension. Individual vectors are
converted to f64 at read time — no temporary file conversion.

### 1.6.5 Progress and Abort

All visualization commands MUST show a progress indicator during data
loading and computation phases:

- Progress updates at **1–4 Hz** (at least once per second, at most
  4 times per second).
- Progress shows: phase name, items processed / total, percentage,
  elapsed time, and estimated time remaining where computable.
- Progress is rendered to stderr so it doesn't interfere with TUI
  rendering once the alternate screen is entered.

All visualization commands MUST handle abort signals:

- **Ctrl-C** (SIGINT): caught and interpreted as a clean abort. The
  command restores the terminal, prints a summary of what was computed
  so far, and exits with status 130.
- **Escape**: caught during TUI interaction and during loading phases
  (via non-blocking key poll). Treated as "stop early with current
  results" — the TUI displays whatever has been computed so far.
- **`?` key**: all TUI modes MUST support `?` as a "show keystroke
  help" toggle. When pressed, an overlay or footer displays all
  available key bindings and their actions.

### 1.6.6 Remote Dataset Access

All visualization commands MUST accept `dataset:profile[:facet]`
specifiers in addition to local file paths. Remote datasets are
accessed through the vectordata data access layer (SRD §13) which
provides:

- **Lazy on-demand chunk fetching** with merkle verification
- **Local caching** in the configured cache directory
- **Automatic promotion** from channel-backed to mmap-backed access
  after full caching
- **Windowed access** for profile-defined subsets of remote files

Visualization commands MUST NOT implement their own download logic.
They MUST use the `DatasetLoader` API (§13.2) to obtain typed views
that handle transport, caching, and verification transparently.

### 1.6.7 Path Handling Requirements

**Absolute paths MUST NOT appear in any user-facing output, persisted
configuration, or stored state.** This is a system-wide requirement
that applies to ALL commands, checks, error messages, and advisories.

All paths shown to users MUST be relative — either to the current
working directory, the workspace directory, or the publish root,
whichever is most natural for the context.

Specific rules:

- **User-facing output** (println, eprintln, log messages, check results,
  advisory messages): All paths MUST be displayed relative to the current
  working directory or to a context-appropriate root (e.g., publish root
  for publish checks, workspace for pipeline messages). Use `rel_display()`
  or equivalent to strip the cwd prefix.
- **Symlinks** created during import MUST use relative targets computed
  from the link's parent directory to the source file.
- **dataset.yaml** MUST contain only relative paths for source references.
  Paths are relative to the directory containing dataset.yaml.
- **Pipeline steps** resolve paths at runtime via `${workspace}` and
  `${cache}` interpolation. Persisted configuration MUST NOT contain
  absolute paths.
- **Catalog entries** store paths relative to the catalog file location.
- **Check messages** (pipeline, catalogs, publish, merkle, integrity):
  paths MUST be relative to the check root or publish root, not absolute.
- **variables.yaml** and **progress log**: paths stored within MUST be
  relative to the workspace.
- **`std::fs::canonicalize()`** MAY be used internally for path
  comparison (e.g., detecting whether two paths refer to the same
  directory) but MUST NOT appear in user-facing output or persisted
  state. When canonicalization is needed for upward traversal, the
  results MUST be relativized before display.

The intent is that:
1. A dataset directory can be moved, renamed, or shared across machines
   without breaking internal references.
2. The system works correctly without full filesystem path traversal
   privileges.
3. User-facing output is concise and readable.

### 1.6.8 TUI Thread Architecture

Any UI/console/TUI logic MUST have a **dedicated rendering thread** that
never blocks on background processes, I/O, or computation:

- The TUI thread's only blocking operation is `event::poll()` with a
  short timeout (10-100ms). Everything else is non-blocking.
- All computation (mean accumulation, eigenvector iteration, projection)
  and all I/O (vector reads, HTTP chunk downloads) MUST run on background
  threads.
- Background threads communicate results to the TUI via `mpsc::channel`.
  The TUI polls with `try_recv()` — never `recv()`.
- Ctrl-C/q/Esc MUST be responsive within one poll cycle (~10ms during
  computation, ~100ms when idle), regardless of what background work is
  in progress.
- Status lines MUST update every frame with live progress: elapsed time,
  throughput rates, cache download stats for remote data, and in-flight
  request counts.

This architecture ensures the TUI never appears frozen, even when
background operations take seconds per unit of work (e.g., remote chunk
downloads over slow connections).

### 1.6.9 Wizard Defaults

The interactive wizard (`veks prepare bootstrap`) SHOULD default to
enabling L2-normalization when vectors are not already normalized.
Normalized vectors provide consistent distance semantics across metrics
and are the expected input for most ANN benchmark configurations.

### 1.6.10 Pre-Computed Ground Truth Handling

When the user provides pre-computed ground truth (neighbor_indices and
optionally neighbor_distances), the wizard MUST:

- **Infer k** from the ground truth ivec dimensionality (each row has k
  indices). Do NOT prompt the user for neighbor count.
- **Skip KNN computation** — no `compute-knn` step is generated.
- **Include KNN verification** — a `verify-knn` step validates a sparse
  sample of the provided ground truth against recomputed brute-force
  distances.
- **Make dedup/zero checks advisory only** — the ground truth was computed
  against the original base ordinal space. Generating ordinal exclusion
  filters (which reindex vectors) would invalidate the ground truth. The
  checks still run and report findings, but no exclusion ordinals are
  produced. Users are warned about this in the wizard output.
- **Infer metric** from normalization status when possible (see §1.6.11).

### 1.6.11 Interactive Configuration Flags

Commands that configure datasets support a consistent set of flags
for controlling interactivity:

| Flag | Short | Meaning |
|------|-------|---------|
| `--interactive` | `-i` | Launch guided wizard with step-by-step prompts |
| `--yes` | `-y` | Accept all defaults without prompting |
| `--restart` | `-r` | Delete existing config and start fresh |
| `--auto` | — | Short-circuit: combines `-i -y -r` behavior |

#### Flag semantics

**`--interactive` (`-i`)**: Launches a multi-step wizard that explains
each option, shows previews, and lets the user choose between
strategies. Example: `veks prepare stratify --interactive` walks
through multiplicative/fibonacci/linear/custom sizing strategies.

**`--yes` (`-y`)**: Accepts the standard defaults for every prompt
without waiting for input. Useful for scripted/CI pipelines. The
defaults are the same values the wizard would show as `[default]`.

**`--restart` (`-r`)**: Deletes existing configuration (`dataset.yaml`,
`variables.yaml`, progress log) and starts fresh. Without this flag,
existing configuration is preserved and the command either resumes
or reports an error if it would conflict.

**`--auto`**: Equivalent to `--interactive --yes --restart` — detects
source files automatically, accepts all standard defaults, and
overwrites any existing configuration. This is the "do everything
from scratch with sensible defaults" mode.

#### Combination matrix

| Flags | Behavior |
|-------|----------|
| (none) | Simple prompt with default shown, Enter to accept |
| `-i` | Full wizard with strategy choices and previews |
| `-y` | Use standard defaults, no prompts |
| `-i -y` | Run wizard logic to compute defaults, skip prompts |
| `-r` | Delete existing config, then prompt |
| `--auto` | Delete existing, detect files, accept all defaults |

#### Standard specs

When computing default values for sized profiles, the **standard
spec** combines three strategies for comprehensive scale coverage:

```
fib:1m..{max}, mul:1m..{max}/2, 0m..{max}/10m
```

This produces profiles at fibonacci points (1m, 2m, 3m, 5m, 8m, 13m,
21m, 34m...), powers of 2 (1m, 2m, 4m, 8m, 16m, 32m...), and linear
10m steps (10m, 20m, 30m...) — all deduped and sorted. The resulting
set gives dense coverage at small scales (important for development
iteration) and regular large-scale checkpoints.

This standard spec is used by:
- `veks prepare stratify --yes` / `--auto`
- `veks prepare stratify --interactive` (option 1, "Standard")
- `veks prepare bootstrap --auto` when sized profiles are enabled

## 1.7 Technology Stack

| Component          | Technology                                                      |
|--------------------|-----------------------------------------------------------------|
| Language           | Rust (edition 2021)                                             |
| CLI framework      | clap v4 with clap_complete for shell completions                |
| SIMD distance      | simsimd + hand-rolled AVX2/AVX-512 for L1                      |
| Parallelism        | std::thread::scope, rayon (in some commands)                    |
| Memory-mapped I/O  | memmap2 crate (via MmapVectorReader)                            |
| UI eventing        | Custom event algebra, sink trait, ratatui alternate-screen TUI  |
| Serialization      | serde + serde_yaml, byteorder for wire formats                  |
| Parquet support    | arrow + parquet crates                                          |
| Half-precision     | half crate (f16)                                                |
| HTTP transport     | reqwest (blocking, connection-pooled)                           |
| Integrity          | sha2 crate for Merkle tree hashing                              |

## 1.7 Scale Characteristics

Observed from LAION-400M processing (407M vectors, 512 dimensions, f16):

| Metric                                   | Value                    |
|------------------------------------------|--------------------------|
| Base vectors file size                   | 418 GB (mvec)            |
| Query vectors file size                  | 418 GB (mvec)            |
| Metadata slab file size                  | 207 GB (407M MNode records)|
| Metadata import time                     | ~250 seconds             |
| Base vector import time                  | ~280 seconds             |
| Brute-force KNN (10K queries, k=100)    | ~2100 seconds            |
| Total records processed                  | 407,314,954              |

These workloads can saturate system resources (memory, file descriptors, disk
I/O, CPU). The `ResourceGovernor` (see [06-resource-management.md](06-resource-management.md))
provides adaptive resource governance to prevent system lockups and OOM
conditions during such workloads.
