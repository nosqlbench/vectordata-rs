<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 06 — Resource Management

## 6.1 The Problem

On 2026-03-05, processing the LAION-400M img2text dataset's
`compute predicates` step caused a complete system lockup. The step
was evaluating 10,000 predicates against 407 million metadata records
(207 GB slab file). The system became unresponsive, requiring a hard reset.

### Root cause analysis

The `compute predicates` command accumulates per-predicate match
ordinal vectors (`Vec<Vec<i32>>`) across metadata records. With 10,000
predicates and high-match-rate predicates on a 407M-record corpus:

- **Worst case per predicate**: 407M × 4 bytes = ~1.6 GB per predicate
- **Aggregate worst case**: 10,000 predicates × 1.6 GB = 16 TB
  (theoretical maximum; actual depends on predicate selectivity)
- **Actual observed behavior**: predicates with selectivity targets near
  0.01% still produce ~40K matches per predicate. At 10K predicates ×
  40K × 4 bytes = 1.6 GB per segment — manageable in isolation, but
  multiple segments in flight or high-selectivity predicates can spike well
  beyond available RAM.

Contributing factors:
1. No memory budget or RSS monitoring during execution
2. No per-command resource limits or quotas
3. No system-level health checks during long-running operations
4. No graceful degradation path (smaller segments, fewer concurrent
   predicates, spilling to disk)
5. Thread pool sizes are fixed, not adaptive to memory pressure

## 6.2 Existing Resource-Aware Patterns

Before detailing gaps, it is important to note patterns already in the
codebase that demonstrate resource awareness:

- **xvec reader** (`RawXvecReader`): Uses 4 MiB read buffers and calls
  `posix_fadvise(FADV_DONTNEED)` every 64 MiB to prevent page cache
  accumulation during sequential scans.
- **xvec writer** (`XvecSink`): Two-phase page cache management:
  1. `sync_file_range(SYNC_FILE_RANGE_WRITE)` for async writeback on new pages
  2. `sync_file_range(SYNC_FILE_RANGE_WAIT_BEFORE)` + `posix_fadvise(FADV_DONTNEED)`
     to release clean pages. Triggered every 64 MiB.
- **SlabReader**: Hints `madvise(MADV_HUGEPAGE)` on Unix for 2 MiB transparent
  huge pages to reduce TLB misses.
- **MmapVectorReader**: Provides `prefetch_range()` and `prefetch_pages()`
  methods using `MADV_WILLNEED` for page cache warming.
- **KNN partitioned computation**: Caches per-partition results to `.cache/`
  so only one partition's results are in memory at a time.
- **Predicate evaluation segmentation**: Processes metadata in 1M-record segments
  with per-segment disk caching.

These patterns are *local* to individual components. The gap is that there
is no *global* coordination or enforcement across the pipeline.

## 6.3 Current Resource Usage Patterns

### Memory

| Component | Pattern | Risk |
|-----------|---------|------|
| `MmapVectorReader` | Memory-mapped files. Kernel manages page cache. | Low direct risk — kernel evicts pages under pressure. But: competing mmap regions can thrash. |
| `SlabReader` page index | Loaded into heap on open. Proportional to page count. | Medium — 407M records at ~6500 records/page ≈ 62K pages × ~16 bytes = ~1 MB. Manageable. |
| KNN partition caches | Cached to disk. Only one partition's results in memory at a time. | Low — partitioned design is sound. |
| Predicate match vectors | `Vec<Vec<i32>>` — one per predicate, growing during segment scan. | **Critical** — unbounded growth within a segment. |
| Parquet→MNode import | Buffered row-group reads + MNode encoding. | Medium — row group size determines peak. |

### CPU

| Component | Pattern | Risk |
|-----------|---------|------|
| Brute-force KNN | `queries × base_vectors × dim` distance computations | High — scales quadratically with dataset size. SIMD-accelerated but still O(N²). |
| Predicate scanning | `predicates × records × fields` condition evaluations | High — though the compiled scan path skips non-targeted fields. |
| SIMD distance | AVX-512/AVX2/NEON dispatched, single-threaded per query | Low per-call, high aggregate. |

### I/O

| Component | Pattern | Risk |
|-----------|---------|------|
| Vector mmap reads | Sequential scan for KNN, random for filtered KNN | Depends on page cache. Cold reads on 418 GB file = sustained I/O. |
| Slab sequential scan | Sequential page reads during predicate evaluation | 207 GB sequential reads. Saturates disk bandwidth. |
| Cache file writes | Per-segment slab writes to `.cache/` | Moderate — many small files can stress filesystem. |
| Progress log writes | Small YAML file, atomic write+rename | Negligible. |

### File descriptors

| Component | Pattern | Risk |
|-----------|---------|------|
| Mmap'd vector files | 1-2 FDs per mmap'd file | Low. |
| Slab files | 1 FD per open SlabReader/SlabWriter | Low for typical pipelines. |
| Cache segment files | 1 FD per segment being read during merge | Medium — hundreds of segments could be opened during merge. |

## 6.4 Requirements: Active Resource Management

### REQ-RM-01: Memory budget awareness

The pipeline engine MUST be able to operate within a configurable memory
budget. Commands MUST be able to query the available memory budget and
adjust their behavior accordingly (e.g., smaller segments, fewer concurrent
allocations, disk-based spilling).

### REQ-RM-02: RSS monitoring

The system MUST periodically monitor its own RSS (resident set size) during
execution of resource-intensive commands. If RSS exceeds a configurable
threshold (absolute or percentage of system RAM), the system MUST take
corrective action before OOM.

### REQ-RM-03: Graceful degradation

When memory pressure is detected, resource-intensive commands MUST be able to:
- Reduce segment sizes (process fewer records per batch)
- Spill intermediate results to disk earlier
- Reduce thread parallelism to lower peak memory
- Flush and release completed intermediate buffers eagerly

### REQ-RM-04: Per-command resource declarations

Each `CommandOp` SHOULD declare its expected resource profile:
- Estimated peak memory (function of input sizes)
- I/O pattern (sequential, random, mixed)
- CPU intensity (light, medium, heavy)
- Whether it can operate in a reduced-memory mode

### REQ-RM-05: Thread pool adaptivity

The thread count SHOULD be adjustable during execution based on observed
memory pressure. Reducing threads reduces peak working set when each thread
holds independent buffers.

### REQ-RM-06: Progress-based checkpointing

Long-running operations MUST checkpoint progress frequently enough that
an OOM kill or system reset loses at most one segment's worth of work.
The predicate evaluation already uses segment caching — this pattern
should be formalized and required for all resource-intensive commands.

### REQ-RM-07: System health telemetry

The pipeline engine SHOULD emit periodic telemetry during execution:
- RSS and virtual memory size
- CPU utilization (user + system, per-core)
- I/O throughput (cumulative read/write bytes)
- I/O queue depth (inflight read/write from `/proc/diskstats`)
- Active thread count
- System page cache size and hit ratio
- Major/minor page fault counts

This telemetry supports both real-time monitoring and post-mortem analysis.

Telemetry is published through two channels:

1. **`ResourceStatusSource::status_line_with_metrics()`** — produces a
   formatted text line for display AND a `ResourceMetrics` struct with
   raw numeric values. The `UiEvent::ResourceStatus` event carries both,
   so the TUI renders charts directly from structured data without
   re-parsing the text representation.

2. **Governor log** (`.cache/.governor.log`) — JSON-line entries with
   observation, decision, throttle, request, and ignored records.

### REQ-RM-08: Configurable resource limits

Resource limits are specified through a unified `--resources` CLI option.
All resource types use a consistent naming and value scheme, including
support for ranges that the governor thread can use for dynamic adjustment.

#### 6.4.1 The `--resources` option

All resource configuration is passed via a single `--resources` flag:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8,segmentsize:500000'
```

**Short form** (inline key:value pairs):

```sh
--resources 'mem:32GiB,threads:8,iothreads:4'
--resources 'mem:25%-50%,segments:5-20'
```

**Long form** (individual flags per resource, rewritten by autocomplete):

```sh
--mem '25%-50%' --threads '4-8' --segmentsize '500000-2000000'
```

When a user types `--mem 32GiB` or `--threads 8`, shell autocomplete
MUST rewrite these to the canonical `--resources 'mem:32GiB,threads:8'`
form. The long-form flags exist solely as ergonomic aliases resolved at
completion time — the parser only needs to handle `--resources`.

#### 6.4.2 Resource types

| Resource | Description | Value syntax | Default |
|----------|-------------|-------------|---------|
| `mem` | Memory budget (RSS ceiling) | absolute or % of system RAM | `80%` |
| `threads` | CPU thread pool size | integer | `num_cpus` |
| `segments` | Maximum concurrent segments in flight | integer | `1` |
| `segmentsize` | Records per processing segment | integer | `1000000` |
| `iothreads` | Concurrent I/O operations | integer | `4` |
| `cache` | Maximum disk space for `.cache/` | absolute size | `unlimited` |
| `readahead` | Read-ahead buffer / prefetch window | absolute size | `64MiB` |

**Note on cache budget**: Cache data is expensive to compute and not
disposable. A cache budget should trigger eviction of the *oldest*
completed segments only when strictly necessary, and should warn the
operator before doing so. Eviction forces recomputation.

#### 6.4.3 Value syntax

Values support absolute sizes, percentages, and ranges:

| Form | Example | Meaning |
|------|---------|---------|
| Absolute | `32GB`, `32GiB`, `1024MB` | Fixed value (SI and IEC units) |
| Percentage | `50%` | Fraction of system capacity |
| Fixed (single) | `8` | Integer count |
| Range (absolute) | `16GiB-48GiB` | Governor adjusts within bounds |
| Range (%) | `25%-50%` | Governor adjusts within bounds |
| Range (count) | `4-8` | Governor adjusts within bounds |

When a single value is given (e.g., `mem:32GiB`), it is treated as both
the minimum and maximum — the governor cannot adjust it. When a range is
given (e.g., `mem:25%-50%`), the governor starts at the midpoint and
dynamically adjusts between the floor and ceiling based on observed
system pressure.

Unit suffixes for memory-like resources:

| Suffix | Base | Multiplier |
|--------|------|-----------|
| `B` | — | 1 |
| `KB` | SI | 1,000 |
| `MB` | SI | 1,000,000 |
| `GB` | SI | 1,000,000,000 |
| `TB` | SI | 1,000,000,000,000 |
| `KiB` | IEC | 1,024 |
| `MiB` | IEC | 1,048,576 |
| `GiB` | IEC | 1,073,741,824 |
| `TiB` | IEC | 1,099,511,627,776 |

#### 6.4.4 Per-command resource applicability

Not all resources apply to every command. Each `CommandOp` declares which
resource types it consumes via a new trait method:

```rust
fn describe_resources(&self) -> Vec<ResourceDesc> {
    vec![]  // default: no resource declarations
}
```

Where `ResourceDesc` names the resource and provides a hint:

```rust
pub struct ResourceDesc {
    /// Resource name (e.g., "mem", "threads", "segmentsize").
    pub name: String,
    /// Human-readable description of how this command uses the resource.
    pub description: String,
    /// Whether the command can dynamically adjust this resource mid-execution.
    pub adjustable: bool,
}
```

**CLI completion filtering**: The dynamic completion system (which already
builds the command tree from `describe_options()`) MUST also consult
`describe_resources()`. When the user has typed a specific pipeline
command, only the resource types declared by that command are offered as
completions after `--resources` or as long-form aliases. For example:

```sh
# `compute knn` declares: mem, threads, readahead
$ veks pipeline compute knn --resources '<TAB>'
mem:  threads:  readahead:

# `slab analyze` declares nothing resource-intensive
$ veks pipeline slab analyze --resources '<TAB>'
# (no suggestions — command has no resource declarations)
```

**Validation**: If a user specifies a resource that is not declared by
the command, the pipeline SHOULD emit a warning (not an error), since
the governor may still use system-level resources like `mem` for global
monitoring even when the command itself does not declare it.

#### 6.4.4a Unrecognized resource types

If a resource limit is applied to a command that does not recognize the
resource type, the limit MUST be silently ignored at runtime — the
command proceeds normally without error. However, this situation MUST be
recorded as a debug-level entry in the governor utilization log so that
operators can detect misconfiguration during post-mortem analysis:

```json
{
  "type": "ignored",
  "ts": "2026-03-05T06:14:30.125Z",
  "step_id": "import-base",
  "resource": "segmentsize",
  "reason": "command 'import' does not declare resource 'segmentsize'; limit ignored"
}
```

This applies to both unknown resource names (typos, custom names) and
known `ResourceType` variants that a specific command does not declare
in its `describe_resources()`. The governor continues to track and
adjust the resource for its own bookkeeping — only the command-level
application is skipped.

#### 6.4.5 Example resource declarations by command

| Command | Applicable resources |
|---------|---------------------|
| `compute knn` | `mem`, `threads`, `readahead` |
| `compute filtered-knn` | `mem`, `threads`, `readahead` |
| `compute sort` | `mem`, `threads`, `readahead` |
| `compute predicates` | `mem`, `threads`, `segments`, `segmentsize` |
| `generate predicated` | `mem`, `threads`, `readahead` |
| `generate derive` | `mem`, `threads`, `readahead` |
| `transform fvec-extract` | `readahead` |
| `import` | `threads`, `iothreads`, `readahead` |
| `convert file` | `iothreads`, `readahead` |
| `slab import` | `iothreads`, `readahead` |
| `slab export` | `readahead` |
| `slab rewrite` | `mem`, `readahead` |
| `slab append` | `readahead` |
| `fetch dlhf` | `iothreads`, `mem` |
| `json rjq` | `mem`, `readahead` |
| `json jjq` | `mem`, `readahead` |
| `analyze stats` | `mem`, `readahead` |
| `analyze histogram` | `mem`, `readahead` |
| `analyze compare` | `mem`, `readahead` |
| `analyze verify-knn` | `mem`, `threads`, `readahead` |
| `analyze explore` | `mem`, `readahead` |
| `analyze profile` | `mem`, `threads`, `readahead` |
| `analyze model-diff` | `mem`, `readahead` |
| `analyze verify-profiles` | `mem`, `readahead` |
| `analyze plot` | `mem`, `readahead` |
| `merkle create` | `readahead` |
| `merkle verify` | `readahead` |
| `merkle diff` | `readahead` |
| `cleanup cleanfvec` | `mem`, `readahead` |
| `datasets prebuffer` | `readahead` |
| `synthesize predicates` | (none — lightweight) |
| `slab inspect` | (none — lightweight) |

#### 6.4.6 Autocomplete rewrite behavior

The dynamic completion system extends the existing `build_augmented_cli()`
mechanism. When long-form resource flags are used:

1. User types: `veks pipeline compute knn --mem 32GiB --threads 8 --base ...`
2. Completion engine recognizes `--mem` and `--threads` as resource aliases
3. On submission, the parser rewrites to:
   `veks pipeline compute knn --resources 'mem:32GiB,threads:8' --base ...`

This keeps the internal API surface small (only `--resources` is parsed)
while providing a natural CLI experience. The long-form flags:
- Are generated dynamically from `describe_resources()`
- Only appear in completions for commands that declare them
- Are never passed through to the command — always rewritten

### REQ-RM-09: OOM prevention for predicate evaluation

The `compute predicates` command specifically MUST:
1. Estimate total memory for match vectors before starting each segment
2. If the estimate exceeds budget, split the predicates into batches
   (process predicate subsets sequentially to bound peak memory)
3. Monitor actual allocation growth during the segment scan
4. If growth rate projects exceeding budget, flush current segment early
   and continue with a new segment

### REQ-RM-10: Mmap coordination

When multiple mmap regions are active (base vectors + query vectors +
metadata slab), the system SHOULD use `madvise` hints:
- `MADV_SEQUENTIAL` for sequential scans
- `MADV_WILLNEED` for pre-faulting upcoming pages
- `MADV_DONTNEED` for releasing completed scan regions

This reduces page cache thrashing when working sets exceed physical RAM.

### REQ-RM-11: Mandatory governance for unbounded-data commands

Every pipeline command that processes arbitrarily large data files MUST
integrate with the resource governor and resource management system
described in this specification. This is not optional — it is a hard
requirement for any `CommandOp` whose input size is not inherently bounded.

#### Classification

A command processes "arbitrarily large data" if any of its inputs or
outputs are files whose size scales with the dataset — or if the command
opens, mmaps, or streams files that could be dataset-scale regardless of
how much data the command itself consumes from them. The classification
errs on the side of inclusion: if a command accepts a file path argument
that could refer to a dataset-scale file, it MUST comply unless the file
is inherently small (configuration, survey JSON, catalog index, etc.).

#### Mandatory obligations

Commands classified as unbounded-data MUST:

1. **Declare resources** via `describe_resources()` — at minimum `mem`,
   plus any other applicable resource types from section 6.4.2.
2. **Consult the governor** before sizing buffers, thread pools, segment
   sizes, or any allocation that scales with input size. The command MUST
   call `ctx.governor.current("resource")` rather than using hardcoded
   defaults or per-command option values directly.
3. **Checkpoint periodically** via `ctx.governor.checkpoint()` at
   segment or partition boundaries. This serves two purposes: it allows
   the governor to update effective resource values (dynamic adjustment),
   and it ensures progress is persisted for crash recovery.
4. **Respond to throttle signals** — if `ctx.governor.should_throttle()`
   returns true, the command MUST reduce its resource consumption before
   continuing (e.g., flush buffers, shrink segment size, reduce thread
   count for subsequent work units).
5. **Report resource consumption** in the `CommandResult` or via the
   governor's telemetry interface, so that per-step resource usage can
   be recorded in the progress log.

#### Commands that MUST comply

Any command that opens, mmaps, or streams a file that could be
dataset-scale. Organized by category:

**Compute commands** — all process dataset-scale vector files:

| Command | Reason |
|---------|--------|
| `compute knn` | Mmaps full base vector file; multi-threaded distance computation |
| `compute filtered-knn` | Mmaps full base vector file; reads metadata indices slab |
| `compute sort` | Reads and reorders arbitrarily large vector files |

**Import / convert / export** — stream arbitrarily large files:

| Command | Reason |
|---------|--------|
| `import` | Streams arbitrarily large source files (npy, parquet) to output |
| `convert file` | Streams arbitrarily large source files to output |
| `slab import` | Reads source records of arbitrary size into a slab |
| `slab export` | Reads slab records of arbitrary count |
| `slab append` | Reads source slab of arbitrary size |
| `slab rewrite` | Rewrites entire slab file |
| `fetch dlhf` | Downloads files of arbitrary size from HuggingFace Hub |

**Generate commands** — read or produce dataset-scale files:

| Command | Reason |
|---------|--------|
| `compute predicates` | Scans entire metadata slab; accumulates match ordinals |
| `generate predicated` | Processes full dataset to produce predicated output |
| `generate derive` | Derives new facets from existing large files |
| `transform fvec-extract` | Reads from arbitrarily large source vectors |
| `transform ivec-extract` | Reads from arbitrarily large source vectors |
| `transform mvec-extract` | Reads from arbitrarily large source vectors |

**Analyze commands** — open and read dataset-scale files:

| Command | Reason |
|---------|--------|
| `analyze stats` | Mmaps and streams entire vector file |
| `analyze histogram` | Mmaps and streams entire vector file |
| `analyze compare` | Mmaps and reads two vector files |
| `analyze verify-knn` | Mmaps base vectors; re-computes distances |
| `analyze select` | Mmaps entire vector file |
| `analyze slice` | Mmaps arbitrarily large source file |
| `analyze zeros` | Mmaps and streams entire vector file |
| `analyze find` | Mmaps and streams entire vector file |
| `analyze explore` | Mmaps full vector file for interactive access |
| `analyze profile` | Mmaps full vector file for distance benchmarking |
| `analyze describe` | Mmaps vector file (full mmap even if only reading metadata) |
| `analyze model-diff` | Mmaps two vector files for distribution comparison |
| `analyze verify-profiles` | Mmaps vector files and model data |
| `analyze plot` | Mmaps vector file for visualization |

**JSON/slab query commands** — iterate over potentially large files:

| Command | Reason |
|---------|--------|
| `json jjq` | Reads and iterates over JSON files of arbitrary size |
| `json rjq` | Opens and iterates over slab files of arbitrary size |

**Integrity commands** — stream full files:

| Command | Reason |
|---------|--------|
| `merkle create` | Streams entire file for hashing |
| `merkle verify` | Streams entire file for verification |
| `merkle diff` | Reads two files for chunk-level comparison |
| `cleanup cleanfvec` | Reads and rewrites vector file |
| `datasets prebuffer` | Reads entire dataset into page cache |

#### Commands exempt from this requirement

Commands that operate on fixed-size inputs, inherently small files, or
do not perform I/O on dataset-scale files:

- `synthesize predicates` — reads a small survey JSON, writes a small slab
- `generate vectors` — output size is user-specified, not input-driven
- `generate ivec-shuffle` — output size is user-specified
- `generate sketch` — output size is user-specified
- `generate from-model` — reads a model file (small), writes user-specified count
- `generate dataset` — orchestrates other commands, does not directly stream data
- `slab inspect`, `slab get`, `slab analyze`, `slab explain`,
  `slab namespaces`, `slab check` — point lookups or metadata-only
  operations that do not stream the full file
- `survey` — samples a bounded number of records (not a full scan)
- `config show`, `config init`, `config list-mounts` — configuration only
- `info file` — reads file header metadata only
- `info compute` — no file I/O
- `datasets list`, `datasets plan`, `datasets curlify`, `datasets cache`
  — read small configuration/catalog files
- `catalog generate` — reads small catalog index
- `merkle summary`, `merkle treeview`, `merkle path` — read compact
  merkle tree files (logarithmic in source file size)
- `merkle spoilbits`, `merkle spoilchunks` — read compact merkle data
- `analyze flamegraph` — reads profiling data (not dataset-scale)
- `analyze check-endian` — reads only a few bytes from the file header

#### Enforcement

The pipeline runner SHOULD verify at step startup that any command
processing files above a configurable size threshold (default: 1 GiB)
has a non-empty `describe_resources()`. If a command processes large
files without resource declarations, the runner MUST emit a warning:

```
WARNING: step 'my-step' (convert file) processes 207 GB input but
declares no resource requirements. Resource governance is disabled
for this step.
```

This warning serves as a prompt for developers to add resource
declarations when implementing or modifying commands.

### REQ-RM-12: Exclusive governor authority over runtime resource adjustment

Resource settings (memory budgets, thread counts, segment sizes, etc.) MUST
only be adjusted at runtime by the resource governor thread. No other code
path — not commands, not the pipeline runner, not user callbacks — is
permitted to mutate effective resource values during execution.

#### Invariants

1. **Single writer**: The `AdaptiveController` (governor thread) is the sole
   entity that writes to the shared `EffectiveResources` state. Commands and
   the runner hold read-only views.

2. **User-specified values are bounds, not directives**: The `--resources`
   values (whether single values or ranges) define the *envelope* within
   which the governor operates. They do not directly set the runtime values.
   Even a single value like `mem:32GiB` is treated as a fixed range
   (`32GiB-32GiB`) that the governor honors — the governor still decides
   *when* and *how* to apply it during execution phases.

3. **Commands request, governor grants**: Commands call
   `ctx.governor.request("segmentsize", preferred)` to express a preference.
   The governor evaluates this against current system state and the
   configured range, then publishes an effective value that the command reads
   on its next `checkpoint()` call. The command MUST NOT assume its
   preference was granted.

4. **No side-channel mutation**: Resources embedded in `Options` (e.g., a
   legacy `threads` option in a step definition) are treated as initial
   hints only. Once the governor is active, the governor's published values
   take precedence. Commands MUST read effective values from the governor,
   not from their own options.

#### Pluggable governor strategies

The governor accepts a pluggable strategy trait:

```rust
pub trait GovernorStrategy: Send + Sync {
    /// Called periodically by the governor thread.
    /// Returns updated effective values for any resources that should change.
    fn evaluate(
        &mut self,
        snapshot: &SystemSnapshot,
        budget: &ResourceBudget,
        current: &EffectiveResources,
    ) -> ResourceAdjustments;
}
```

Where `SystemSnapshot` includes:
- Current RSS and virtual memory size
- CPU utilization (per-core and aggregate)
- I/O throughput (read/write bytes/sec)
- Page fault rate (major + minor)
- Active thread count
- Elapsed time since last evaluation

And `ResourceAdjustments` is a map of resource names to new effective values
(only resources that should change need to be included).

**Default strategy: `MaximizeUtilizationStrategy`**

The built-in default governor strategy endeavors to maximize system
utilization to a high degree without saturating the system. Its heuristics:

1. **Memory headroom targeting**: Aims to keep RSS between 70% and 90% of
   the `mem` ceiling. Below 70%, it increases `segmentsize` or permits
   more concurrent work. Above 90%, it decreases `segmentsize` and signals
   throttle. Above 95%, it triggers emergency flush.

2. **CPU saturation detection**: Monitors per-core utilization. If all
   cores are saturated (>95% user+system) and memory permits, the strategy
   holds steady. If cores are underutilized (<50% aggregate), it may
   increase `threads` toward the ceiling to improve throughput.

3. **I/O bandwidth balancing**: Monitors read/write throughput. If I/O
   is the bottleneck (high iowait, low CPU), it may increase `readahead`
   or `iothreads`. If I/O is saturated and causing memory pressure (dirty
   page accumulation), it reduces write concurrency.

4. **Page fault rate monitoring**: A spike in major page faults indicates
   page cache thrashing. The strategy responds by reducing mmap working
   set (via `MADV_DONTNEED` on completed regions) and potentially reducing
   `segmentsize` to lower the active data footprint.

5. **Ramp-up / ramp-down damping**: Changes are applied gradually (no
   more than 25% adjustment per evaluation cycle) to avoid oscillation.
   The strategy tracks whether its last adjustment improved or worsened
   the target metrics and adjusts its damping factor accordingly.

#### Default utilization targets

The `MaximizeUtilizationStrategy` uses the following default targets and
thresholds. All values are tunable via strategy configuration but these
defaults represent the out-of-box behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Memory** | | |
| `mem_target_low` | 70% of `mem` ceiling | Below this: scale up (increase segmentsize, permit more work) |
| `mem_target_high` | 85% of `mem` ceiling | Target operating range ceiling; above this: begin scaling down |
| `mem_throttle` | 90% of `mem` ceiling | Above this: signal throttle to active commands |
| `mem_emergency` | 95% of `mem` ceiling | Above this: emergency flush — all commands must immediately release buffers |
| `mem_evaluation_interval` | 500 ms | How often RSS is sampled |
| **CPU** | | |
| `cpu_target` | 85% aggregate utilization | Target utilization across all cores |
| `cpu_saturated` | 95% aggregate | Above this: hold steady, do not add more threads |
| `cpu_underutilized` | 50% aggregate | Below this: consider increasing `threads` |
| `cpu_evaluation_interval` | 1 s | How often CPU utilization is sampled |
| **I/O** | | |
| `io_target_utilization` | 80% of device bandwidth | Target I/O throughput relative to measured device capacity |
| `io_dirty_page_limit` | 10% of `mem` ceiling | Maximum dirty page accumulation before reducing write concurrency |
| `io_evaluation_interval` | 2 s | How often I/O metrics are sampled |
| **Page faults** | | |
| `major_fault_spike` | 100 major faults/sec sustained over 3 evaluations | Threshold for triggering mmap working set reduction |
| **Damping** | | |
| `max_adjustment_pct` | 25% | Maximum resource adjustment per evaluation cycle |
| `damping_increase` | 1.5× | Damping factor applied when last adjustment worsened metrics |
| `damping_decrease` | 0.75× | Damping factor relaxation when last adjustment improved metrics |
| `min_damping` | 1.0 | Floor for damping factor (no damping) |
| `max_damping` | 4.0 | Ceiling for damping factor (most conservative) |
| **General** | | |
| `ramp_up_delay` | 5 s | After startup or throttle recovery, wait this long before scaling up |
| `stable_window` | 3 evaluations | Number of consecutive stable evaluations before scaling up |

#### Operating bands

The strategy partitions system state into five bands and takes
corresponding action:

```
 0%─────50%──────70%──────85%──────90%──────95%────100%  (of mem ceiling)
  │  UNDERUSED  │  NOMINAL  │ CAUTION │THROTTLE│EMERGENCY│
  │ scale UP    │ hold      │scale dn │ signal │  flush  │
```

- **UNDERUSED** (RSS < `mem_target_low`): The system is well below
  capacity. The governor increases `segmentsize` (up to the configured
  ceiling) and may increase `threads` if CPU is also underutilized.
  Ramp-up is subject to `ramp_up_delay` and `stable_window` to avoid
  oscillation after recent throttling.

- **NOMINAL** (`mem_target_low` ≤ RSS < `mem_target_high`): The system
  is within the desired operating range. No adjustments are made. This
  is the steady-state target.

- **CAUTION** (`mem_target_high` ≤ RSS < `mem_throttle`): RSS is
  approaching limits. The governor begins reducing `segmentsize` and
  `threads` by up to `max_adjustment_pct` per cycle. No throttle signal
  is sent — commands continue normally but with smaller work units.

- **THROTTLE** (`mem_throttle` ≤ RSS < `mem_emergency`): The governor
  sets `should_throttle() = true`. Active commands MUST reduce
  consumption on their next `checkpoint()` — flush completed buffers,
  shrink segments, reduce thread counts. The governor continues
  reducing effective resource values.

- **EMERGENCY** (RSS ≥ `mem_emergency`): The governor sends an
  emergency flush signal. Commands MUST immediately release all
  non-essential memory — write intermediate results to disk, drop
  caches, shrink to minimum segment size. If RSS remains above
  emergency after two evaluation cycles, the governor logs a critical
  warning (the system may be headed for OOM and the only remaining
  mitigation is for the command to abort its current segment).

Users may provide custom governor strategies by implementing the
`GovernorStrategy` trait and registering them via the CLI:

```sh
veks run dataset.yaml --resources 'mem:25%-50%' --governor conservative
```

Built-in strategies:

| Strategy | Behavior |
|----------|----------|
| `maximize` (default) | Aggressively uses available resources up to the configured ceiling |
| `conservative` | Starts at floor values and only increases when sustained low utilization is observed |
| `fixed` | Uses the midpoint of each range, never adjusts. Useful for benchmarking. |

### REQ-RM-13: Governor utilization log

The resource governor MUST maintain a separate utilization log that records
its observations, decisions, and the measurements that drove those decisions.
This log enables operators to understand *why* resources were adjusted and
supports post-mortem tuning of resource ranges and governor strategies.

#### Log location

```
${workspace}/.governor.log
```

This is a line-oriented log file (not YAML/JSON) for append efficiency.
Each line is a self-contained JSON object.

#### Log entry types

**Observation** — periodic system snapshot:

```json
{
  "type": "observation",
  "ts": "2026-03-05T06:14:30.123Z",
  "rss_bytes": 34359738368,
  "rss_pct": 42.5,
  "cpu_user_pct": 78.2,
  "cpu_system_pct": 4.1,
  "io_read_mbps": 1240.5,
  "io_write_mbps": 320.1,
  "major_faults": 12,
  "minor_faults": 84200,
  "active_threads": 8,
  "step_id": "evaluate-predicates"
}
```

**Decision** — resource adjustment:

```json
{
  "type": "decision",
  "ts": "2026-03-05T06:14:30.125Z",
  "step_id": "evaluate-predicates",
  "resource": "segmentsize",
  "old_value": 1000000,
  "new_value": 750000,
  "reason": "RSS at 91% of ceiling (32GiB); reducing segment size to lower peak memory",
  "heuristic": "memory_headroom",
  "trigger_metric": "rss_pct",
  "trigger_value": 91.2
}
```

**Throttle** — signaling a command to reduce consumption:

```json
{
  "type": "throttle",
  "ts": "2026-03-05T06:14:35.001Z",
  "step_id": "evaluate-predicates",
  "reason": "RSS exceeded 95% of ceiling; emergency flush requested",
  "resources_affected": ["segmentsize", "threads"]
}
```

**Request/Grant** — command resource negotiation:

```json
{
  "type": "request",
  "ts": "2026-03-05T06:14:36.100Z",
  "step_id": "evaluate-predicates",
  "resource": "segmentsize",
  "requested": 1500000,
  "granted": 750000,
  "reason": "requested value exceeds current effective limit under memory pressure"
}
```

#### Log retention

The governor log is overwritten at the start of each pipeline run (it is
not a cumulative log across runs). If the operator needs historical data,
they should archive the log before re-running the pipeline.

#### Relation to progress log

The progress log (`.cache/.upstream.progress.yaml`) records per-step *summary*
resource consumption (peak RSS, total I/O, CPU time). The governor log
records the *detailed timeline* of observations and decisions within each
step. Together they provide both the "what" and the "why" of resource usage.

### REQ-RM-14: Resource affinity protocol (demand-pull)

Commands (resource users) and the governor (resource manager) communicate
via a **demand-pull** protocol. The knowledge of which resources can
actually benefit a workload lives in the command; the knowledge of whether
granting more resources is safe lives in the governor. Neither side should
make the other's decision.

#### Protocol

1. **Command declares affinity**: At segment or checkpoint boundaries, a
   command MAY signal to the governor that it could make productive use of
   a specific resource type by calling `ctx.governor.offer_demand()`:

   ```rust
   ctx.governor.offer_demand("threads", current_threads, max_useful_threads);
   ctx.governor.offer_demand("iothreads", current_io, max_useful_io);
   ```

   The call declares: "I am currently using `current`, and I could
   productively use up to `desired`." This is purely informational —
   the command does NOT change its own behavior yet.

2. **Governor evaluates and publishes**: On the next evaluation cycle, the
   governor considers all outstanding demand offers against system state
   (CPU idle, I/O queue depth, memory headroom, storage saturation
   thresholds). If the resource is not scarce, the governor increases the
   effective value for that resource toward (but not exceeding) the
   demanded ceiling:

   ```rust
   // Governor internal logic (simplified)
   if snapshot.cpu_user_pct < cpu_underutilized && !throttle {
       let demand = demands.get("threads");
       let ceiling = budget.get("threads").ceiling();
       let new_eff = current_eff.min(demand.desired).min(ceiling);
       effective.insert("threads", new_eff);
   }
   ```

3. **Command reads updated effective value**: On its next `checkpoint()`,
   the command reads `ctx.governor.current("threads")` and adjusts its
   thread pool, segment concurrency, or I/O parallelism accordingly. The
   command MUST NOT exceed the granted effective value.

4. **Governor may revoke**: If system pressure increases, the governor
   reduces the effective value. Commands observe this on their next
   checkpoint and MUST comply by reducing consumption.

#### Design principles

- **Commands know their workload**: An import command knows that adding
  more I/O threads will help because it is I/O-bound. A compute command
  knows that more CPU threads will help because it is compute-bound. Only
  the command has this knowledge.

- **Governor knows the system**: The governor knows the RSS headroom, CPU
  utilization, I/O queue depth, storage type, and page cache hit ratio.
  Only the governor can make safe decisions about whether granting more
  resources will help or harm overall system health.

- **Neither side overreaches**: Commands don't blindly grab resources.
  The governor doesn't blindly assign resources without demand.

#### Examples

**Example 1: Import metadata (I/O bound)**

The `import` command processes parquet shards. It detects that its I/O
wait time exceeds compute time and offers demand for more I/O concurrency:

```rust
// In import command's processing loop, at checkpoint boundary:
let io_threads = ctx.governor.current_or("iothreads", 4) as usize;
let could_use = (io_threads * 2).min(32); // I'm I/O bound, I could use 2x
ctx.governor.offer_demand("iothreads", io_threads as u64, could_use as u64);
```

The governor sees that I/O queue depth is low (under the storage's
saturation threshold from auto-detected NVMe) and grants the increase.
On the next checkpoint, the import command reads the new effective value
and spawns additional I/O worker threads.

**Example 2: KNN computation (CPU bound)**

The `compute knn` command finds that SIMD distance computations leave
some cores idle due to memory stalls:

```rust
let threads = ctx.governor.current_or("threads", 8) as usize;
let could_use = (threads + 4).min(num_cpus); // Hyperthreading helps here
ctx.governor.offer_demand("threads", threads as u64, could_use as u64);
```

The governor checks CPU utilization and memory headroom. If cores are
idle and memory is in the NOMINAL band, it grants additional threads.

**Example 3: Extract (I/O bound, sequential read)**

The `transform mvec-extract` command uses sorted-index extraction. The
sequential read pattern benefits from increased readahead:

```rust
let readahead = ctx.governor.current_or("readahead", 64 * 1024 * 1024);
let could_use = 256 * 1024 * 1024; // 256 MiB readahead helps sequential
ctx.governor.offer_demand("readahead", readahead, could_use);
```

The governor checks page cache pressure and memory headroom. If the
system has ample free RAM and the page cache hit ratio is high, it
grants the larger readahead window.

**Example 4: Governor declines (memory pressure)**

Same import command offers demand for more I/O threads, but this time
RSS is in the CAUTION band (85% of ceiling). The governor keeps the
effective value unchanged or reduces it. The command continues at its
current concurrency level without any special handling — the protocol
is entirely non-blocking from the command's perspective.

#### Demand expiry

Demand offers are transient — they expire after one governor evaluation
cycle. Commands must re-offer demand on each checkpoint if they still
want more resources. This prevents stale demands from keeping resources
elevated after a command's workload characteristics change.

#### Logging

Demand offers and governor responses are logged in `.cache/.governor.log`:

```json
{
  "type": "demand",
  "ts": "2026-03-07T14:22:01.500Z",
  "step_id": "import-metadata",
  "resource": "iothreads",
  "current": 4,
  "desired": 8,
  "granted": 6,
  "reason": "io_queue_depth 12 < saturation_threshold 128 (local NVMe); scaling up"
}
```

### REQ-RM-15: Storage type detection

The resource governor MUST detect the backing storage type for the
workspace directory at startup. This information informs I/O-related
resource decisions (queue depth saturation thresholds, readahead sizing,
I/O concurrency limits).

#### Detection method (Linux)

1. Resolve the workspace path to its backing block device via
   `stat(2)` device number and `/sys/block/*/dev` matching.
2. For device-mapper (LVM/RAID) devices, follow
   `/sys/block/dm-*/slaves/` to the underlying physical device.
3. Read sysfs attributes:
   - `/sys/block/<dev>/queue/rotational` — 0=SSD/NVMe, 1=spinning
   - `/sys/block/<dev>/device/model` — identifies cloud storage types
   - `/sys/block/<dev>/queue/nr_requests` — hardware queue depth
   - Transport (inferred from device name or uevent)
4. Classify into storage tiers:

| Tier | Detection | Saturation depth |
|------|-----------|-----------------|
| `LocalNvme` | NVMe transport + "Instance Storage" model or bare NVMe | 128 |
| `NetworkBlock` | "Elastic Block Store", virtio, xvd/vd prefix | 32 |
| `SataSsd` | SATA transport, non-rotational | 32 |
| `Hdd` | rotational=1 | 4 |

The storage type and saturation threshold are exposed via
`governor.storage_type()` and `governor.io_saturation_depth()` for
use by the demand-pull protocol and governor strategy heuristics.

### REQ-RM-16: Page cache observability

The system MUST provide page cache performance metrics for monitoring
and governor decision-making:

1. **System page cache size**: Read from `/proc/meminfo` (Cached + Buffers).
   Displayed in the resource status line as `pcache: <size>`.

2. **Page cache hit ratio**: Computed from process-level minor faults
   (cache hits) and major faults (cache misses) between sampling
   intervals. Displayed as `pcache: <size> hit:<pct>%` when available.

These metrics help identify whether the workload is I/O-bound due to
cache misses (low hit ratio → increase readahead, reduce working set)
or compute-bound with good cache behavior (high hit ratio → scale up
CPU resources).

## 6.5 Current State vs Requirements

| Requirement | Current state | Gap |
|-------------|--------------|-----|
| REQ-RM-01 | `ResourceBudget` parsed from `--resources`; commands query `governor.current()` / `current_or()` for thread count and segment sizing; memory-aware segment/partition sizing via `mem_ceiling()` in compute knn and gen metadata-indices | **Done** |
| REQ-RM-02 | `SystemSnapshot::sample()` reads RSS, page faults, CPU times (with raw ticks), I/O bytes, thread count from `/proc/self/stat`, `/proc/self/io`, `/proc/self/status` | **Done** |
| REQ-RM-03 | Governor publishes throttle/emergency signals; `checkpoint()` called at boundaries in compute knn, filtered-knn, gen metadata-indices, compute sort, convert, import, gen predicated, cleanup cleanfvec, analyze verifyknn | **Done** |
| REQ-RM-04 | 37 commands implement `describe_resources()` with resource declarations | **Done** |
| REQ-RM-05 | Commands query `governor.current_or("threads", ...)` for thread count (compute knn, filtered-knn, gen metadata-indices, import, convert) | **Done** |
| REQ-RM-06 | Segment caching formalized with **file-stem-based keys** for cross-profile reuse: compute predicates uses `.cache/{input_stem}.{pred_stem}.seg_{start}_{end}.predkeys.slab`; compute knn uses `.cache/{base_stem}.{query_stem}.range_{start}_{end}.k{k}.{metric}.{ext}`. Cache keys are derived from input file stems (not step IDs) so that overlapping ordinal ranges across profiles share cached segments. Profile barriers ensure smallest-to-largest execution order so cached segments are available for reuse. | **Done** |
| REQ-RM-07 | Per-step `ResourceSummary` (peak RSS, CPU user/system seconds, I/O read/write bytes) captured by runner and stored in `.cache/.upstream.progress.yaml`; governor log writes JSON-line observation/decision/throttle/request/ignored entries | **Done** |
| REQ-RM-08 | `--resources` and `--governor` args on both `run_pipeline()` and per-command direct CLI; long-form resource aliases (e.g., `--mem`, `--readahead`) generated from `describe_resources()` with conflict avoidance; completion filtering shows only applicable resource types per command | **Done** |
| REQ-RM-09 | Memory-aware segment/partition sizing: gen metadata-indices estimates per-segment memory from selectivity × predicates × 4B × threads and scales down if budget exceeded; compute knn estimates result set + page cache pressure and reduces partition_size accordingly | **Done** |
| REQ-RM-10 | `MmapVectorReader` provides `advise_sequential()` (MADV_SEQUENTIAL), `prefetch_range()` / `prefetch_pages()` (MADV_WILLNEED), and `release_range()` (MADV_DONTNEED); compute knn uses sequential advice at start and releases completed partitions; gen metadata-indices has prefetch thread with MADV_WILLNEED per segment | **Done** |
| REQ-RM-11 | Runner warns for >1 GiB files without resource declarations; 37 commands comply; runner logs ignored resources for budget items not declared by command | **Done** |
| REQ-RM-12 | Background governor thread (`governor-bg`) monitors RSS vs mem ceiling, adjusts effective values, and sets throttle/emergency flags between checkpoint() calls; only started when explicit `--resources` with mem budget is provided; 3 built-in strategies (maximize, conservative, fixed) | **Done** |
| REQ-RM-13 | `GovernorLog` writes to `.cache/.governor.log` with JSON-line entries including `Ignored` variant | **Done** |

## 6.6 Proposed Architecture: ResourceGovernor

A new `ResourceGovernor` component that sits between the pipeline runner
and command execution:

```
--resources 'mem:25%-50%,threads:4-8,segmentsize:500K-2M'
    ↓ parse
ResourceBudget (per-resource floor/ceiling pairs)
    ↓
StreamContext
  └── ResourceGovernor
        ├── MemoryMonitor       (periodic RSS sampling)
        ├── ResourceBudget      (parsed from --resources)
        ├── HealthReporter      (telemetry emission)
        └── AdaptiveController  (adjusts resources within declared ranges)
```

### Integration points

1. **CLI layer**: Parses `--resources` (or rewrites long-form aliases).
   Constructs `ResourceBudget` with floor/ceiling for each resource type.

2. **Pipeline runner**: Creates `ResourceGovernor` from `ResourceBudget`.
   Passes it to commands via `StreamContext`.

3. **CommandOp trait extension**:
   ```rust
   fn describe_resources(&self) -> Vec<ResourceDesc> {
       vec![]  // default: no resource declarations
   }
   ```

4. **Commands**: Query `ctx.governor.current("mem")` to get the current
   effective value within the configured range. Call
   `ctx.governor.checkpoint()` periodically. Respond to
   `ctx.governor.should_throttle()`.

5. **AdaptiveController**: Background governor thread (see REQ-RM-12) that
   is the *sole entity* permitted to adjust resource values at runtime.
   - Reads RSS from `/proc/self/status` (Linux)
   - Compares against `mem` range
   - Adjusts `threads`, `segmentsize`, and other adjustable resources
     toward floor when pressure is high, toward ceiling when pressure
     is low
   - Publishes updated effective values that commands read on next
     checkpoint
   - Uses a pluggable `GovernorStrategy` (default: `MaximizeUtilizationStrategy`)
   - Writes all observations, decisions, and throttle events to the
     governor utilization log (`.cache/.governor.log`, see REQ-RM-13)

6. **Dynamic completion**: `build_augmented_cli()` consults
   `describe_resources()` to populate per-command resource completions
   and generate long-form alias flags.

### Minimal viable implementation

For immediate relief, the minimum changes needed:

1. Add `--resources 'mem:...'` CLI parsing with single-value support
2. Add RSS check in `compute predicates` between segments
3. If RSS exceeds the `mem` ceiling, reduce `segmentsize` by 50% for
   remaining segments
4. Add `madvise(MADV_SEQUENTIAL)` to `MmapVectorReader` for scan access
5. Log RSS at segment boundaries

Range support and the full adaptive governor can follow incrementally.
This addresses the immediate crash scenario without requiring the full
framework.

## 6.7 Failure Scenarios and Mitigations

### Scenario: Predicate evaluation OOM

- **Trigger**: High-selectivity predicates on large metadata corpus
- **Current behavior**: System lockup / OOM kill
- **Mitigation**: Memory-bounded match vectors with disk spilling

### Scenario: KNN mmap thrashing

- **Trigger**: Base + query + metadata files exceed physical RAM
- **Current behavior**: Extreme slowdown from page cache thrashing
- **Mitigation**: Sequential madvise hints, memory-aware partition sizing

### Scenario: Cache directory growth

- **Trigger**: Many segments × many partitions generating thousands of
  cache files
- **Current behavior**: Filesystem metadata overhead, unbounded growth
- **Mitigation**: Cache budget limits with oldest-first eviction (with
  operator warning), aggregated cache files. Cache is not disposable —
  eviction forces recomputation of affected segments.

The `--compress-cache` option provides an additional resource trade-off
for cache files: gzip compression reduces disk I/O bandwidth and storage
consumption at the cost of CPU time and memory for in-memory
compression/decompression. This is beneficial when cache storage is the
bottleneck (slow or limited disk) but CPU headroom exists. See
[05-dataset-specification.md](05-dataset-specification.md) §5.7 for
which cache artifacts are eligible for compression.

### Scenario: Concurrent pipelines

- **Trigger**: User runs multiple dataset preparations simultaneously
- **Current behavior**: No coordination — each pipeline assumes full system
- **Mitigation**: System-level resource accounting, advisory locks

## 6.8 Observability Requirements

### Real-time

- Progress bars (already implemented via indicatif)
- RSS and memory pressure indicators in progress output
- Step-level resource consumption summary on completion

### Post-mortem

- Per-step resource usage in `.cache/.upstream.progress.yaml`:
  - peak RSS
  - total I/O (read + write bytes)
  - CPU time (user + system)
- Segment-level timing and memory in cache metadata
- System event correlation (dmesg OOM messages)
