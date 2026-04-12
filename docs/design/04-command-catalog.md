<!-- Copyright (c) Jonathan Shook -->
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
| `import` | Import source data into preferred internal format by facet type | **I/O-heavy**: streaming read + write. Memory proportional to buffer sizes. For parquet→slab, uses compiled MNode writer. |

### Convert

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `convert file` | Convert between vector formats | **I/O-bound**: streaming read + write. Memory: single vector buffer. |

### Compute

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `compute knn` | Brute-force exact K-nearest neighbors. Supports `--compress-cache` to gzip partition cache files. | **CPU + memory intensive**: mmaps base vectors (potentially hundreds of GB). Multi-threaded query processing. Partitioned mode caches per-partition results in `.cache/`. |
| `compute filtered-knn` | Predicate-filtered exact KNN | **CPU + memory intensive**: mmaps base vectors. Reads matching metadata ordinals from slab. Multi-threaded. |
| `compute sort` | Sort vectors by some criterion | I/O-bound |
| `compute dedup` | Lexicographic sort + duplicate detection via external merge-sort. Produces sorted ordinal index (ivec), duplicate ordinals (ivec), and JSON report. Adaptive prefix-key width avoids full-vector I/O for non-duplicates. Cached sorted runs enable resume. Supports `--compress-cache` to gzip sorted run files. | **CPU + I/O**: multi-threaded parallel sort. Memory proportional to batch size (default 1M vectors). Intermediate sorted runs written to `.cache/dedup_runs/`. Governor-aware batch sizing. |

### Generate

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `generate vectors` | Generate synthetic random vectors. Supports `--zeros-ratio` and `--duplicates-ratio` for injecting test data artifacts (zero vectors, duplicates) at controlled ratios using Vose's alias method for O(1) per-vector disposition sampling. `--append` mode adds vectors to an existing file, matching its dimensionality. All count/dimension options accept ISO suffixes (K, M, G, KiB, MiB, etc.). Generation is embarrassingly parallel via per-chunk deterministic RNGs; duplicates are sourced intra-batch (random earlier vector within the same chunk) so no cross-thread state is needed. Supports all 6 element types (f32, f16, f64, i32, i16, u8). | CPU + I/O (parallel) |
| `generate ivec-shuffle` | Generate shuffled ordinal permutation | Memory: full ordinal array |
| `generate sketch` | Generate vector sketches | CPU |
| `generate from-model` | Generate vectors from an ML model | CPU + GPU |
| `generate dataset` | Generate synthetic dataset | Mixed |
| `generate derive` | Derive new facets from existing | Mixed |
| `generate predicated` | Generate predicated dataset | Mixed |
| `synthesize predicates` | Generate random filter predicates from metadata survey | **Light**: reads survey JSON, outputs small slab. Fast. |
| `compute predicates` | Compute predicates against all metadata records. Supports `--compress-cache` to gzip segment cache files. | **CRITICAL: highest resource risk**. Scans entire metadata slab (207 GB for LAION-400M). Segmented processing with cache. Per-segment: reads all pages, evaluates all predicates against every record. Memory: match ordinal vectors per predicate × segment. THIS IS THE COMMAND THAT CAUSED THE SYSTEM LOCKUP. |

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
| `survey` | Sample records and analyze field distributions | Memory: sample set |

### Transform

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `transform extract` | **Generic extract**: auto-detects source format from file extension and delegates to the appropriate format-specific command. Preferred entry point — avoids format mismatches. | I/O-bound |
| `transform fvec-extract` | Extract vector subset from fvec file. Validates `.fvec` extension. Supports both **index-based** (with `--ivec-file`) and **range-based** (identity, without `--ivec-file`) extraction. | I/O-bound |
| `transform ivec-extract` | Extract index subset from ivec file. Supports index-based and range-based modes. | I/O-bound |
| `transform mvec-extract` | Extract half-precision vector subset from mvec file. Supports index-based and range-based modes. | I/O-bound |
| `transform slab-extract` | Extract record subset from slab file. Supports index-based and range-based modes. | I/O-bound |
| `transform clean-ordinals` | Filter ordinal index by excluding duplicate and zero-vector ordinals. Reads a sorted index (from `compute dedup`) plus exclusion lists and writes a clean index suitable for downstream shuffle and extraction. | **Memory**: loads exclusion sets into HashSet. I/O: single streaming pass over input index. |

### Analysis

All analysis commands support **all 6 xvec element types** (f32, f16, f64,
i32, i16, u8) via `ElementType` dispatch. Values are converted to f64
for statistical accumulation. Format is auto-detected from file extension.

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `analyze describe` | Describe vector file metadata (format, dim, count, file size) | Light |
| `analyze stats` | Compute statistical summary | Streaming |
| `analyze histogram` | Generate dimension histograms | Streaming |
| `analyze verify-knn` | Validate KNN ground truth (legacy, full range) | CPU-intensive (re-computes distances) |
| `verify knn` | **Pipeline verification**: sparse-sample KNN spot-check using parallel batched SIMD distance recomputation. Samples N queries uniformly at random, recomputes brute-force top-k, compares against stored ground truth with tie tolerance. Writes JSON report. See §12.13.1. | CPU-intensive (parallel, SIMD) |
| `verify predicates` | **Pipeline verification**: sparse-sample predicate evaluation using SQLite as independent oracle. Loads N sampled metadata rows into in-memory SQLite, translates PNode predicates to SQL WHERE clauses, compares against stored results. Writes JSON report. See §12.13.2. | Memory (SQLite) + CPU |
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
| `datasets cache-compress` | Retroactively gzip-compress eligible `.cache/` artifacts (dedup runs, KNN partition caches, predicate segment caches). Parallel via rayon. See §5.7 for eligibility rules. | I/O + CPU (parallel) |
| `datasets cache-uncompress` | Reverse of `cache-compress`: decompress `.gz` cache files back to originals. Parallel via rayon. Preserves file timestamps. | I/O + CPU (parallel) |
| `datasets import` | Bootstrap a new dataset directory from source files. Interactive wizard (`-i`), auto-accept (`-y`), restart (`--restart`). See §12. | Light |

### Pipeline Control

| Command | Description | Resource profile |
|---------|-------------|------------------|
| `barrier` | Synchronization barrier between pipeline phases | None — no execution, ordering only |
| `set variable` | Set a pipeline variable for downstream steps | Light |
| `clear variables` | Clear all pipeline variables | Light |
| `inspect predicate` | Decode and render a PNode predicate record | Light |

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
| `fetch bulkdl` | Bulk parallel file downloads with token expansion | Network + I/O |
| `catalog generate` | Generate dataset catalog | Light |

## 4.3 Resource Risk Matrix

| Risk level | Commands | Primary concern |
|-----------|----------|-----------------|
| **Critical** | `compute predicates` | Memory: accumulates match ordinals for all predicates across all records. At 407M records × 10K predicates, matching ordinal vectors can consume tens of GB. Segmented, but segment size and match density determine peak memory. |
| **High** | `compute knn`, `compute filtered-knn` | Memory: mmaps full base vector file. CPU: brute-force distance for every query × every base vector. Thread count × working set can overwhelm system. |
| **Medium** | `import` (parquet→slab), `datasets prebuffer` | I/O: sustained sequential reads and writes at multi-GB/s. Can saturate disk bandwidth and page cache. |
| **Low** | Most analysis, slab, merkle, config commands | Bounded resource usage. |

## 4.4 compute predicates — Deep Dive

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

### `veks datasets catalog generate`

Scans a directory tree for `dataset.yaml` files and writes hierarchical
`catalog.json` / `catalog.yaml` index files. Each directory level between
the scan root and each dataset gets its own catalog containing entries for
all datasets below it.

#### Catalog root detection (`.catalog_root`)

A `.catalog_root` file is a zero-byte sentinel that marks the top of the
catalog hierarchy. When `catalog generate` is run from any directory below
a `.catalog_root`, the generator automatically uses the directory containing
`.catalog_root` as the scan root and generates catalogs at every level from
there down.

This enables a common pattern where the remote publish URL has an
uncataloged leading path:

```
/data/                         ← .publish_url (s3://bucket/data/)
/data/public/                  ← no catalog here (leading path)
/data/public/vectordata/       ← .catalog_root (catalog hierarchy starts here)
/data/public/vectordata/ds-a/  ← dataset.yaml
/data/public/vectordata/ds-b/  ← dataset.yaml
```

Running `veks datasets catalog generate` from `/data/public/vectordata/ds-a/`
detects `.catalog_root` at `/data/public/vectordata/` and generates:
- `/data/public/vectordata/catalog.json` (lists ds-a and ds-b)
- `/data/public/vectordata/ds-a/catalog.json` (lists ds-a only)
- `/data/public/vectordata/ds-b/catalog.json` (lists ds-b only)

No catalog is generated at `/data/public/` or `/data/` because those
directories are above the `.catalog_root`.

#### `.publish_url` warning

When no `.catalog_root` is present and a `.publish_url` file is detected
above the scan directory, the command warns that catalog coverage may be
incomplete. The warning suggests using `--for-publish-url` to regenerate
the full publish hierarchy, or placing a `.catalog_root` file at the
desired catalog top level.

When `.catalog_root` is detected, the warning is suppressed because the
user has explicitly declared the catalog boundary.

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT` | `.` | Root directory to scan |
| `--basename` | `catalog` | Filename stem for catalog files |
| `--for-publish-url` | off | Walk up to `.publish_url` and generate from that root |
| `--update` | on | Only update existing catalog files; skip directories with no catalog yet |
| `--no-update` | — | Generate at all hierarchy levels regardless of existing files |

#### Resolution priority

When determining the scan root:

1. **`--for-publish-url`**: walks up to `.publish_url`, generates from that root
2. **`.catalog_root` in parent path**: uses that directory, generates full hierarchy
3. **`--update` (default)**: uses `INPUT`, updates only existing catalog files
4. **`--no-update`**: uses `INPUT`, generates at all hierarchy levels

### `veks check`

Context-aware pre-flight verification. The checks performed depend on
where the command is run:

#### Context detection

`veks check` examines the target directory and determines the context:

1. **Dataset directory** — contains `dataset.yaml`. Checks focus on the
   local dataset: pipeline execution, file integrity, merkle coverage,
   extraneous files.

2. **Publish path** — no `dataset.yaml` in the current directory, but a
   `.publish_url` or `.catalog_root` is found in the parent path. This
   indicates the directory is part of a publishable hierarchy. Checks
   include publish URL binding and catalog chain freshness across the
   full hierarchy.

3. **Unrecognized** — neither a dataset directory nor part of a publish
   path. `veks check` exits with an error.

#### Check categories by context

| Check | Dataset dir | Publish path | Description |
|-------|:-----------:|:------------:|-------------|
| `pipeline-execution` | yes | — | All pipeline steps fresh |
| `pipeline-coverage` | yes | — | Every publishable file covered by a pipeline step |
| `publish` | — | yes | `.publish_url` is valid and transport is supported |
| `merkle` | yes | — | Every publishable file above threshold has a current `.mref` |
| `integrity` | yes | — | Every data file passes format-specific structural validation |
| `catalogs` | — | yes | `catalog.json`/`catalog.yaml` present and current at every directory level from `.catalog_root` (or `.publish_url`) down; parent catalogs not older than child catalogs |
| `extraneous` | yes | — | No publishable files outside the pipeline manifest |

#### Design rationale

This separation means:
- Running `veks check` in a dataset directory after `veks run` validates
  that the dataset is internally consistent — pipeline complete, files
  intact, merkle hashes current.
- Running `veks check` from a publishing root (or any intermediate
  directory with `.catalog_root` or `.publish_url`) validates the
  catalog chain and publish binding — everything needed before
  `veks publish`.
- The two contexts do not overlap: a dataset directory never checks
  parent catalog chains (that's the publish path's job), and a publish
  path never checks individual pipeline step freshness.
