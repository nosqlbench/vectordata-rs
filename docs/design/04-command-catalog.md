<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 04 — Command Catalog

## 4.1 Registration

All commands are registered in `commands/mod.rs::register_all()`. The
registry maps command path strings to factory functions that produce
`Box<dyn CommandOp>` instances.

Each command implements:

- `describe_options()` — accepted key-value options
- `command_doc()` — built-in markdown documentation with summary and body
  (see [07-command-documentation.md](07-command-documentation.md))
- `describe_resources()` — resource types the command consumes (mem, threads,
  readahead, etc.). 37 commands that process dataset-scale files provide
  resource declarations; lightweight commands return an empty vector.
  (see [06-resource-management.md](06-resource-management.md) §6.4.4)

## 4.2 Command Groups

### Import

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `import facet` | Import source data into preferred internal format by facet type | **I/O-heavy**: streaming read + write. Memory proportional to buffer sizes. For parquet→slab, uses compiled MNode writer. |

### Convert

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `convert file` | Convert between vector formats | **I/O-bound**: streaming read + write. Memory: single vector buffer. |

### Compute

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `compute knn` | Brute-force exact K-nearest neighbors | **CPU + memory intensive**: mmaps base vectors (potentially hundreds of GB). Multi-threaded query processing. Partitioned mode caches per-partition results in `.cache/`. |
| `compute filtered-knn` | Predicate-filtered exact KNN | **CPU + memory intensive**: mmaps base vectors. Reads predicate-key ordinals from slab. Multi-threaded. |
| `compute sort` | Sort vectors by some criterion | I/O-bound |

### Generate

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `generate vectors` | Generate synthetic random vectors | CPU + I/O |
| `generate ivec-shuffle` | Generate shuffled ordinal permutation | Memory: full ordinal array |
| `generate fvec-extract` | Extract vector subset | I/O-bound |
| `generate ivec-extract` | Extract index subset | I/O-bound |
| `generate hvec-extract` | Extract half-precision subset | I/O-bound |
| `generate sketch` | Generate vector sketches | CPU |
| `generate from-model` | Generate vectors from an ML model | CPU + GPU |
| `generate dataset` | Generate synthetic dataset | Mixed |
| `generate derive` | Derive new facets from existing | Mixed |
| `generate predicated` | Generate predicated dataset | Mixed |
| `generate predicates` | Generate random filter predicates from metadata survey | **Light**: reads survey JSON, outputs small slab. Fast. |
| `generate predicate-keys` | Evaluate predicates against all metadata records | **CRITICAL: highest resource risk**. Scans entire metadata slab (207 GB for LAION-400M). Segmented processing with cache. Per-segment: reads all pages, evaluates all predicates against every record. Memory: match ordinal vectors per predicate × segment. THIS IS THE COMMAND THAT CAUSED THE SYSTEM LOCKUP. |

### Slab Operations

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `slab import` | Import records into slab | I/O-bound |
| `slab export` | Export records from slab | I/O-bound |
| `slab append` | Append records between slabs | I/O-bound |
| `slab rewrite` | Rewrite slab with clean alignment | I/O-bound |
| `slab check` | Validate structural integrity | I/O-bound |
| `slab get` | Extract records by ordinal | Random I/O |
| `slab analyze` | Report statistics | Light |
| `slab explain` | Display page layout | Light |
| `slab namespaces` | List namespaces | Light |
| `slab inspect` | Decode + render records via ANode | Light per record |
| `slab survey` | Sample records and analyze field distributions | Memory: sample set |

### Analysis

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `analyze describe` | Describe vector file metadata | Light |
| `analyze stats` | Compute statistical summary | Streaming |
| `analyze histogram` | Generate dimension histograms | Streaming |
| `analyze verify-knn` | Validate KNN ground truth | CPU-intensive (re-computes distances) |
| `analyze compare` | Compare two vector files | I/O: reads both files |
| `analyze select` | Select vectors by predicate | Streaming |
| `analyze slice` | Extract ordinal range | I/O-bound |
| `analyze check-endian` | Detect endian issues | Light |
| `analyze zeros` | Find zero vectors | Streaming |
| `analyze explore` | Interactive exploration | Light |
| `analyze find` | Search for vectors | Streaming |
| `analyze profile` | Performance profiling | CPU-intensive |
| `analyze model-diff` | Compare model outputs | CPU-intensive |
| `analyze verify-profiles` | Validate dataset profiles | Mixed |
| `analyze plot` | Generate plots | Light |
| `analyze flamegraph` | Generate flamegraphs | CPU profiling |

### Merkle

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `merkle create` | Build merkle tree from file | Streaming + CPU |
| `merkle verify` | Verify merkle tree | Streaming + CPU |
| `merkle diff` | Compare merkle trees | Light |
| `merkle summary` | Summarize merkle tree | Light |
| `merkle treeview` | Display merkle tree | Light |
| `merkle path` | Show merkle proof path | Light |
| `merkle spoilbits` | Corrupt specific bits | Light |
| `merkle spoilchunks` | Corrupt specific chunks | Light |

### Datasets

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `datasets list` | List available datasets | Light |
| `datasets plan` | Show dataset preparation plan | Light |
| `datasets cache` | Manage dataset cache | I/O |
| `datasets curlify` | Generate curl commands for downloads | Light |
| `datasets prebuffer` | Pre-buffer dataset into page cache | I/O-heavy |

### Other

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `config show` | Show configuration | Light |
| `config init` | Initialize configuration | Light |
| `config list-mounts` | List mount points | Light |
| `info file` | Show file metadata | Light |
| `info compute` | Show compute capabilities | Light |
| `cleanup cleanfvec` | Remove invalid vectors | I/O + CPU |
| `json jjq` | JSON query (jq-like) | Light |
| `json rjq` | Record-oriented JSON query | Light |
| `fetch dlhf` | Download from HuggingFace | Network + I/O |
| `catalog generate` | Generate dataset catalog | Light |

## 4.3 Resource Risk Matrix

| Risk level | Commands | Primary concern |
|-----------|----------|-----------------|
| **Critical** | `generate predicate-keys` | Memory: accumulates match ordinals for all predicates across all records. At 407M records × 10K predicates, matching ordinal vectors can consume tens of GB. Segmented, but segment size and match density determine peak memory. |
| **High** | `compute knn`, `compute filtered-knn` | Memory: mmaps full base vector file. CPU: brute-force distance for every query × every base vector. Thread count × working set can overwhelm system. |
| **Medium** | `import facet` (parquet→slab), `datasets prebuffer` | I/O: sustained sequential reads and writes at multi-GB/s. Can saturate disk bandwidth and page cache. |
| **Low** | Most analysis, slab, merkle, config commands | Bounded resource usage. |

## 4.4 generate predicate-keys — Deep Dive

This is the command that caused the system lockup. Its execution model:

### Phase 1: Load predicates

Read all PNode records from `predicates.slab`. For LAION-400M, this is 10,000
predicates (~800 KB total). Light.

### Phase 2: Memoize predicates

Flatten AND-only predicate trees into field-indexed condition lists. OR trees
and indexed-field predicates fall back to full tree evaluation. Light.

### Phase 3: Open metadata slab

Open the metadata slab (207 GB for LAION-400M). The `SlabReader` reads the
page index. For 407M records, the page index alone can be significant.

### Phase 4: Plan segments

Divide the metadata into segments (default 1M records). Each segment maps to
a contiguous range of slab pages. Cache paths are computed for each segment.

### Phase 5: Process segments

For each segment:
1. Allocate `Vec<Vec<i32>>` — one vector per predicate for match ordinals
2. Read all pages in the segment
3. For each record in each page, scan fields and evaluate all predicates
4. Records that match a predicate get their ordinal appended to that
   predicate's match vector

**The memory risk**: If a predicate has high selectivity (matches many
records), its match vector grows to hold millions of i32 ordinals. With 10K
predicates and up to millions of matches per predicate, memory consumption
can spike dramatically within a single segment.

### Phase 6: Write segment results to cache

Segment match results are written to cache slab files. Each cache record is
a packed `[i32 LE]*` of matching ordinals.

### Phase 7: Merge cached segments

All segment caches are merged into the final output slab.

### Failure mode

The system lockup occurred because:
- 10,000 predicates × millions of matching ordinals per predicate
- No memory limit on per-predicate match vectors
- No monitoring of system memory pressure
- No backpressure or throttling when RSS approaches system limits
- The process grew until OOM killed it or swapping made the system unresponsive

## 4.5 Root-Level Commands

In addition to pipeline commands (accessed via `veks pipeline` or
`veks run`), veks provides root-level commands for exploration, inventory,
and convenience access to common pipeline operations.

### `veks datasets`

Root-level inventory command for browsing and searching datasets addressable
by the catalog system. This mirrors the equivalent command in the Java
companion project.

| Subcommand | Description | Status |
|------------|-------------|--------|
| `veks datasets list` | List all datasets known to configured catalogs | Planned |
| `veks datasets search <query>` | Search datasets by name, tag, or description | Planned |
| `veks datasets info <name>` | Show detailed metadata for a dataset (dimensions, element type, record count, profiles, facets) | Planned |
| `veks datasets catalogs` | List configured catalog sources | Planned |

The root-level `veks datasets` command differs from the pipeline
`datasets list` / `datasets plan` commands in scope and intent:

- **Pipeline `datasets` commands** operate within the context of a pipeline
  execution — they plan steps, manage caches, and prebuffer files for a
  specific workspace.
- **Root-level `veks datasets`** is a standalone exploration tool for
  discovering what datasets exist across all configured catalogs, independent
  of any pipeline or workspace. It is the entry point for a user who wants to
  answer "what datasets can I work with?"

The catalog system provides the backing data. Catalogs are configured via
`veks config` and can reference local directories, remote indexes, or
HuggingFace repositories. The `veks datasets` command queries these catalogs
and presents results in a human-readable format.
