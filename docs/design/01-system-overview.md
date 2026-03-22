<!-- Copyright (c) DataStax, Inc. -->
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

## 1.5 Command Categories

Veks commands are organized into three categories based on their purpose and
the nature of their output.

### Pipeline commands

Commands that produce structured, reproducible outputs suitable for data
processing workflows. These commands can analyze, process, filter, convert,
compute, generate, import, or otherwise transform data in a deterministic and
composable way. Pipeline commands live under the `veks pipeline` namespace and
implement the `CommandOp` trait so they can be composed into DAG-ordered
pipelines via `dataset.yaml`.

Pipeline commands are invocable two ways:

- **Via pipeline**: `veks run dataset.yaml` — the runner executes steps in DAG
  order, passing each command a `StreamContext` with workspace, scratch space,
  variables, and resource governance.
- **Directly**: `veks pipeline <group> <command> [options]` — a single
  command is executed with CLI-provided options and an ad-hoc `StreamContext`.

Pipeline command groups:

| Group         | Purpose                                                                      |
|---------------|------------------------------------------------------------------------------|
| `import`      | Import source data into preferred internal format by facet type              |
| `convert`     | Format-to-format conversion of vector data                                   |
| `compute`     | Brute-force KNN, filtered KNN, sorting                                       |
| `generate`    | Synthetic vectors, predicates, metadata indices, shuffles, sketches          |
| `transform`   | Extract subsets from vector and slab files                                   |
| `slab`        | Slab file operations (import, export, append, rewrite, check, get, inspect)  |
| `analyze`     | Stats, histograms, verify-knn, compare, select, slice, zeros, find, plots   |
| `merkle`      | Merkle tree creation, verification, diff, summary, proof paths              |
| `datasets`    | List, plan, cache, curlify, prebuffer                                        |
| `cleanup`     | Repair or remove malformed data files                                        |
| `config`      | Show, init, list-mounts                                                      |
| `info`        | File metadata, compute capabilities                                          |
| `json`        | Record-oriented JSON querying (jjq, rjq)                                     |
| `fetch`       | Download from external sources (HuggingFace, bulk URL templates)             |
| `synthesize`  | Generate synthetic predicates from metadata survey                           |
| `inspect`     | Decode and render predicate records                                          |
| `set`         | Set and clear pipeline variables                                             |
| `barrier`     | Synchronization barriers between pipeline phases                             |
| `survey`      | Sample records and analyze field distributions                               |

### Exploration commands

Commands that help the user navigate datasets, understand data, and illustrate
results in a human-oriented, often unstructured way. Their output is designed
for human consumption rather than downstream processing.

| Command                  | Purpose                                                            |
|--------------------------|--------------------------------------------------------------------|
| `veks help [command]`    | Display documentation for pipeline commands and groups             |

### Inventory commands

Commands for listing and finding basic elements — datasets, formats,
capabilities, configuration. Their output enumerates what is available rather
than processing data.

| Command              | Purpose                                                        |
|----------------------|----------------------------------------------------------------|
| `veks datasets`     | Browse, search, and inspect datasets from configured catalogs  |
| `veks completions`  | Generate shell completions for bash/zsh/fish                   |

### Root-level command summary

| Command          | Category             | Description                                        |
|------------------|----------------------|----------------------------------------------------|
| `veks run`       | Pipeline runner      | Execute a full pipeline from dataset.yaml          |
| `veks pipeline`  | Pipeline (direct)    | Execute a single pipeline command with CLI args    |
| `veks script`    | Pipeline (export)    | Emit a pipeline as an equivalent shell script      |
| `veks prepare`   | Preparation          | Import, stratify, catalog, compress datasets       |
| `veks datasets`  | Inventory            | Browse and search datasets from configured catalogs|
| `veks check`     | Validation           | Pre-flight checks for dataset readiness            |
| `veks publish`   | Publishing           | Publish dataset to S3                              |
| `veks help`      | Documentation        | Pipeline command documentation                     |
| `veks completions`| Inventory           | Shell completion generation                        |

## 1.6 Technology Stack

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
