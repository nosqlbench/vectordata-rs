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
vectordata-rs/               Cargo workspace root
├── Cargo.toml               workspace members: vectordata, veks, slabtastic
├── vectordata/              shared types and I/O abstractions
│   └── src/
│       ├── lib.rs           VectorReader trait, format types
│       └── io.rs            MmapVectorReader<T> — mmap-backed vector access
├── slabtastic/              page-aligned record container library
│   └── src/
│       ├── lib.rs           SlabWriter, WriterConfig, PageEntry, OpenProgress
│       ├── page.rs          Page — fixed-size page with record packing
│       ├── pages_page.rs    PagesPage — page index (table of contents)
│       └── reader.rs        SlabReader — ordinal-addressed record retrieval
└── veks/                    the CLI application
    └── src/
        ├── main.rs          CLI entry point (clap), top-level commands
        ├── lib.rs           library facade (if present)
        ├── formats/         data format codecs
        │   ├── mod.rs       VecFormat enum, format detection
        │   ├── anode.rs     ANode — unified MNode/PNode wrapper, decode/encode
        │   ├── anode_vernacular.rs  Vernacular enum, render/parse Stage 2
        │   ├── mnode/       MNode codec: wire format, scan, vernacular
        │   │   ├── mod.rs   MNode, MValue, to_bytes/from_bytes
        │   │   └── scan.rs  zero-alloc record scanning for predicate eval
        │   ├── pnode/       PNode codec: wire format, eval, vernacular
        │   │   ├── mod.rs   PNode, PredicateNode, ConjugateNode
        │   │   ├── eval.rs  tree-walk predicate evaluation
        │   │   └── vernacular.rs  SQL/CQL/CDDL rendering
        │   ├── reader/      format readers (npy, parquet, slab, xvec)
        │   └── writer/      format writers (xvec)
        ├── ui/              UI-agnostic eventing layer
        │   ├── mod.rs       module root and re-exports
        │   ├── event.rs     UiEvent enum, ProgressId, ProgressKind
        │   ├── sink.rs      UiSink trait
        │   ├── handle.rs    UiHandle facade, ProgressHandle RAII guard
        │   ├── plain_sink.rs  non-TTY sink (stdout text, no progress)
        │   └── test_sink.rs   test harness sink (records events)
        ├── import/          facet-aware import logic
        ├── convert/         format-to-format conversion
        └── pipeline/        DAG execution engine
            ├── mod.rs       module root, RunArgs, run_pipeline()
            ├── command.rs   CommandOp trait, Options, StreamContext, CommandDoc
            ├── runner.rs    step execution loop, resource enforcement
            ├── progress.rs  ProgressLog — persistent .upstream.progress.yaml
            ├── resource.rs  ResourceGovernor, strategies, budget parsing
            ├── bound.rs     artifact completeness checks
            ├── simd_distance.rs  SIMD-accelerated distance functions
            ├── dag.rs       topological sort, dependency resolution
            ├── schema.rs    dataset.yaml parsing
            ├── interpolate.rs  variable interpolation (${scratch}, etc.)
            ├── registry.rs  CommandRegistry — command name → factory mapping
            ├── cli.rs       direct CLI invocation, clap command tree
            └── commands/    ~66 CommandOp implementations
```

## 1.3 Crate Responsibilities

### vectordata

The shared foundation crate. Provides:

- **`VectorReader<T>` trait** — uniform access to vector files by ordinal
  (`get_slice(index) → &[T]`, `count() → usize`, `dimension() → usize`)
- **`MmapVectorReader<T>`** — memory-mapped implementation for xvec files
  (fvec, ivec, hvec, etc.). Opens the file once, mmaps it, and provides O(1)
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

Key properties: page-aligned I/O, ordinal-addressed records, opaque payloads,
append-friendly, namespace support, probe-based completeness checks.

Four reader access modes:
1. **Point get** — interpolation search (O(1) expected) + mmap slice
2. **Batch iteration** — sequential page reads for streaming
3. **Sink read** — sequential write of all records to any `Write` sink
4. **Multi-batch concurrent** — scoped thread parallel batch reads

### veks

The CLI application. Encompasses:

- **CLI layer** — clap-based command parsing (`run`, `pipeline`, `import`,
  `convert`, `analyze`, `bulkdl`, `completions`)
- **UI eventing layer** — decoupled event algebra (`UiEvent` enum) dispatched
  to pluggable sinks (`UiSink` trait) via an ergonomic handle facade
  (`UiHandle`). No pipeline code imports rendering libraries directly.
- **Format codecs** — MNode/PNode/ANode binary codecs, vernacular text
  renderers/parsers (JSON, SQL, CQL, CDDL, YAML, readout)
- **Pipeline engine** — DAG-ordered step execution with skip-if-fresh,
  artifact bound checks, progress tracking, dry-run, variable interpolation,
  resource governance via `ResourceGovernor`
- **~66 pipeline commands** — import, convert, compute KNN, generate
  predicates, slab operations, analysis, merkle trees, etc. Each provides
  built-in markdown documentation (`command_doc()`) and resource declarations
  (`describe_resources()`).

## 1.4 Execution Modes

| Mode | Entry point | Description |
|------|-------------|-------------|
| `veks run dataset.yaml` | Pipeline runner | Execute DAG-ordered steps from dataset.yaml |
| `veks pipeline <group> <cmd>` | Direct command | Execute a single pipeline command with CLI args |
| `veks import` | Import facade | Facet-aware import (auto-selects output format) |
| `veks convert` | Convert facade | Format-to-format conversion |
| `veks bulkdl config.yaml` | Bulk downloader | Parallel file downloads with token expansion |
| `veks analyze <cmd>` | Analysis | Inspect, validate, and characterize vector data |
| `veks datasets` | Inventory | Browse and search datasets from configured catalogs |

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

| Group | Purpose |
|-------|---------|
| `import` | Import source data into preferred internal format by facet type |
| `convert` | Format-to-format conversion of vector data |
| `compute` | Brute-force KNN, filtered KNN, sorting |
| `generate` | Synthetic vectors, predicates, predicate keys, shuffles, extracts, sketches |
| `slab` | Slab file operations (import, export, append, rewrite, check, get, inspect, survey) |
| `analyze` | Stats, histograms, verify-knn, compare, select, slice, zeros, find, profiles, plots |
| `merkle` | Merkle tree creation, verification, diff, summary, proof paths |
| `datasets` | List, plan, cache, curlify, prebuffer |
| `cleanup` | Repair or remove malformed data files |
| `config` | Show, init, list-mounts |
| `info` | File metadata, compute capabilities |
| `json` | Record-oriented JSON querying (jjq, rjq) |
| `fetch` | Download from external sources (HuggingFace) |
| `catalog` | Generate dataset catalog |
| `bulkdl` | Bulk parallel file downloads with token expansion |

### Exploration commands

Commands that help the user navigate datasets, understand data, and illustrate
results in a human-oriented, often unstructured way. Their output is designed
for human consumption rather than downstream processing. Exploration commands
are root-level subcommands of `veks`.

| Command | Purpose |
|---------|---------|
| `veks analyze describe` | Describe vector file metadata (dimensions, element type, normalization) |
| `veks help [command]` | Display documentation for pipeline commands and groups |

### Inventory commands

Commands for listing and finding basic elements — datasets, formats,
capabilities, configuration. Their output enumerates what is available rather
than processing data. Inventory commands are root-level subcommands of `veks`.

| Command | Purpose |
|---------|---------|
| `veks datasets` | Browse, search, and inspect datasets addressable by the catalog system |
| `veks completions` | Generate shell completions for bash/zsh/fish |

### Root-level command summary

Root-level commands (`veks <command>`) include pipeline commands with
root-level shortcuts, exploration commands, and inventory commands:

| Command | Category | Description |
|---------|----------|-------------|
| `veks run` | Pipeline runner | Execute a full pipeline from dataset.yaml |
| `veks pipeline` | Pipeline (direct) | Execute a single pipeline command with CLI args |
| `veks script` | Pipeline (export) | Emit a pipeline as an equivalent shell script |
| `veks import` | Pipeline | Facet-aware import into preferred internal format |
| `veks convert` | Pipeline | Format-to-format conversion of vector data |
| `veks bulkdl` | Pipeline | Parallel file downloads with token expansion |
| `veks analyze` | Exploration | Inspect and describe vector data |
| `veks help` | Exploration | Pipeline command documentation |
| `veks datasets` | Inventory | Browse and search datasets from configured catalogs |
| `veks completions` | Inventory | Shell completion generation |

## 1.6 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust (edition 2021) |
| CLI framework | clap v4 |
| SIMD distance | simsimd + hand-rolled AVX2/AVX-512 for L1 |
| Parallelism | std::thread::scope, rayon (in some commands) |
| Memory-mapped I/O | memmap2 crate (via MmapVectorReader) |
| UI eventing | Custom event algebra (`ui/event.rs`), sink trait (`ui/sink.rs`), ratatui (planned) |
| Serialization | serde + serde_yaml, byteorder for wire formats |
| Parquet support | arrow + parquet crates |
| Half-precision | half crate (f16) |

## 1.7 Scale Characteristics

Observed from LAION-400M processing (407M vectors, 512 dimensions, f16):

| Metric | Value |
|--------|-------|
| Base vectors file size | 418 GB (hvec) |
| Query vectors file size | 418 GB (hvec) |
| Metadata slab file size | 207 GB (407M MNode records) |
| Metadata import time | ~250 seconds |
| Base vector import time | ~280 seconds |
| Brute-force KNN (10K queries, k=100) | ~2100 seconds |
| Total records processed | 407,314,954 |

These workloads can saturate system resources (memory, file descriptors, disk
I/O, CPU). The `ResourceGovernor` (see [06-resource-management.md](06-resource-management.md))
provides adaptive resource governance to prevent system lockups and OOM
conditions during such workloads.
