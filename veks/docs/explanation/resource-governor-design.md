<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The Resource Governor

The resource governor is a runtime component that dynamically adjusts memory
budgets, thread counts, segment sizes, and other resource allocations within
user-specified bounds. It exists because static resource limits cannot safely
handle the range of workloads that veks pipelines encounter.

## The problem it solves

On 2026-03-05, processing the LAION-400M img2text dataset locked up a machine
completely. The `generate predicate-keys` step was evaluating 10,000 predicates
against 407 million metadata records stored in a 207 GB slab file. Each
predicate accumulates a `Vec<i32>` of matching ordinals. With high-match-rate
predicates, worst-case memory scales to 10,000 predicates times 1.6 GB per
predicate — a theoretical 16 TB. Even with the existing 1M-record segment
caching, predicates with moderate selectivity produced 1.6 GB per segment
across multiple segments in flight. Memory grew without bound, the kernel
thrashed, and the system required a hard reset.

The contributing factors were:

1. No memory budget or RSS monitoring during execution
2. No per-command resource limits or quotas
3. No system-level health checks during long-running operations
4. No graceful degradation path (smaller segments, fewer concurrent predicates,
   spilling to disk)
5. Thread pool sizes were fixed and not adaptive to memory pressure

## Why static limits are not enough

A single fixed memory limit is either too conservative or too aggressive,
depending on the workload:

- **Import pipelines** stream data through small buffers. A 50% memory ceiling
  wastes half the system for no benefit.
- **KNN computation** memory-maps large vector files. The kernel manages page
  cache automatically, but competing mmap regions can thrash if the working set
  exceeds physical RAM.
- **Predicate-key generation** accumulates per-predicate match vectors that
  grow during a segment scan. Peak memory depends on predicate selectivity,
  which varies per dataset.

System capacity also varies. The same pipeline runs on a 64 GB workstation
during development and a 512 GB cloud instance in production. A fixed value
that is safe on one machine is wasteful on another.

Workloads also change *during* a single pipeline run. Early steps may be
I/O-bound imports that need minimal memory, while later steps are
compute-heavy KNN passes that benefit from every available byte of page cache.
A fixed limit set for the worst step penalizes every other step.

## The governor model

Instead of a single fixed limit, the user specifies *ranges* — an envelope
of acceptable values:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8,segmentsize:500000-2000000'
```

The resource governor operates within these ranges. It monitors live system
metrics (RSS, CPU utilization, I/O throughput, page fault rates) and
dynamically adjusts effective resource values. When the system has headroom, the
governor scales up toward the ceiling for better throughput. When pressure
builds, it scales down toward the floor to avoid OOM.

A single value like `mem:32GiB` is treated as a fixed range (`32GiB-32GiB`).
The governor honors it but cannot adjust — this gives the user a static limit
when that is what they want.

## Exclusive governor authority (REQ-RM-12)

The governor thread is the *sole entity* permitted to write effective resource
values at runtime. This is not a convenience — it is a correctness requirement.
If commands could independently adjust their own thread counts or segment sizes
in response to pressure, their adjustments would conflict: one command reducing
threads while another increases them, or two commands simultaneously growing
their segments past the combined memory budget.

The invariants are:

1. **Single writer.** The `AdaptiveController` (governor thread) is the only
   code that writes to `EffectiveResources`. Commands and the pipeline runner
   hold read-only views.

2. **User values are bounds, not directives.** The `--resources` values define
   the envelope. The governor decides the effective values within that envelope.

3. **Commands request, governor grants.** A command calls
   `ctx.governor.request("segmentsize", preferred)` to express a preference.
   The governor evaluates it against current system state and the configured
   range, then publishes an effective value that the command reads on its next
   `checkpoint()`. The command must not assume its preference was granted.

4. **No side-channel mutation.** Legacy per-command options (like a `threads`
   field in a step definition) are treated as initial hints only. Once the
   governor is active, its published values take precedence.

## The five operating bands

The governor partitions system state into five bands based on current RSS
relative to the memory ceiling:

```
 0%─────50%──────70%──────85%──────90%──────95%────100%  (of mem ceiling)
  │  UNDERUSED  │  NOMINAL  │ CAUTION │THROTTLE│EMERGENCY│
  │ scale UP    │ hold      │scale dn │ signal │  flush  │
```

- **UNDERUSED** (RSS below 70% of ceiling): The system is well below capacity.
  The governor increases `segmentsize` and may increase `threads` if CPU is
  also underutilized. Ramp-up is subject to delay and stability requirements
  to avoid oscillation after recent throttling.

- **NOMINAL** (70%–85%): The system is within the target operating range. No
  adjustments are made. This is the steady-state goal.

- **CAUTION** (85%–90%): RSS is approaching limits. The governor begins
  reducing `segmentsize` and `threads` by up to 25% per evaluation cycle.
  Commands continue normally but with smaller work units.

- **THROTTLE** (90%–95%): The governor sets `should_throttle()` to true.
  Active commands must reduce consumption on their next `checkpoint()` — flush
  completed buffers, shrink segments, reduce thread counts.

- **EMERGENCY** (above 95%): The governor sends an emergency flush signal.
  Commands must immediately release all non-essential memory — write
  intermediate results to disk, drop caches, shrink to minimum segment size.
  If RSS remains above emergency after two evaluation cycles, the governor
  logs a critical warning indicating the system may be headed for OOM.

## Strategy pluggability

The governor accepts a pluggable strategy via the `GovernorStrategy` trait:

```rust
pub trait GovernorStrategy: Send + Sync {
    fn evaluate(
        &mut self,
        snapshot: &SystemSnapshot,
        budget: &ResourceBudget,
        current: &EffectiveResources,
    ) -> ResourceAdjustments;
}
```

Three built-in strategies ship with veks:

| Strategy | Behavior |
|----------|----------|
| `maximize` (default) | Aggressively uses available resources up to the ceiling |
| `conservative` | Starts at floor values, grows only under sustained low utilization |
| `fixed` | Uses the midpoint of each range, never adjusts — useful for reproducible benchmarks |

Users select a strategy with `--governor`:

```sh
veks run dataset.yaml --resources 'mem:25%-50%' --governor conservative
```

## MaximizeUtilizationStrategy deep dive

The default `maximize` strategy aims to keep the system running as close to
full capacity as possible without tipping into pressure. It evaluates four
dimensions:

**Memory headroom targeting.** The strategy aims to keep RSS between 70% and
85% of the memory ceiling. Below 70%, it increases `segmentsize` or permits
more concurrent work. Above 85%, it begins scaling down. Above 90%, it signals
throttle. Above 95%, it triggers emergency flush.

**CPU saturation detection.** The strategy monitors per-core utilization. When
all cores are saturated (above 95% user+system) and memory permits, it holds
steady. When cores are underutilized (below 50% aggregate), it may increase
`threads` toward the ceiling.

**I/O bandwidth balancing.** The strategy monitors read/write throughput. When
I/O is the bottleneck (high iowait, low CPU), it may increase `readahead` or
`iothreads`. When I/O is saturated and causing dirty page accumulation, it
reduces write concurrency.

**Page fault rate monitoring.** A spike in major page faults (above 100/sec
sustained over three evaluation cycles) indicates page cache thrashing. The
strategy responds by triggering `MADV_DONTNEED` on completed mmap regions and
potentially reducing `segmentsize` to lower the active data footprint.

**Ramp-up / ramp-down damping.** Changes are applied gradually — no more than
25% adjustment per evaluation cycle. The strategy tracks whether its last
adjustment improved or worsened the target metrics and adjusts a damping factor
accordingly (increasing damping after a bad adjustment, relaxing it after a
good one). This prevents oscillation between scale-up and scale-down.

## The governor log

The governor writes a line-oriented log to `${workspace}/.governor.log`.
Each line is a self-contained JSON object. The log is overwritten at the start
of each pipeline run.

The log exists for two reasons:

1. **Post-mortem tuning.** After a run completes, the operator can read the
   log to understand whether the resource ranges were well-chosen. If the
   governor spent the entire run in UNDERUSED, the floor should be raised. If
   it repeatedly hit EMERGENCY, the ceiling should be lowered or the workload
   should be restructured.

2. **Understanding decisions.** When throughput drops or a step takes longer
   than expected, the log shows exactly what the governor observed and why it
   adjusted resources. This replaces guesswork with data.

The log records four entry types:

- **Observation** — periodic system snapshots (RSS, CPU, I/O, page faults,
  active threads, current step).
- **Decision** — resource adjustments with the triggering metric, old value,
  new value, and the heuristic that drove the change.
- **Throttle** — notifications sent to commands to reduce consumption, with
  the reason and affected resources.
- **Request/Grant** — command resource negotiations showing what was requested,
  what was granted, and why the grant differed from the request.

The progress log (`.upstream.progress.yaml`) records per-step *summary*
resource consumption. The governor log records the *detailed timeline* within
each step. Together they provide both the "what" and the "why" of resource
usage.

## Existing resource-aware patterns in the codebase

The governor does not replace local resource management — it coordinates it.
Several components already implement resource-aware patterns:

- **xvec reader** (`RawXvecReader`): Uses 4 MiB read buffers and calls
  `posix_fadvise(FADV_DONTNEED)` every 64 MiB to prevent page cache
  accumulation during sequential scans.

- **xvec writer** (`XvecSink`): Two-phase page cache management —
  `sync_file_range` for async writeback on new pages, then
  `posix_fadvise(FADV_DONTNEED)` to release clean pages. Triggered every
  64 MiB.

- **SlabReader**: Hints `madvise(MADV_HUGEPAGE)` on Unix for 2 MiB transparent
  huge pages to reduce TLB misses.

- **MmapVectorReader**: Provides `prefetch_range()` and `prefetch_pages()`
  methods using `MADV_WILLNEED` for page cache warming.

- **KNN partitioned computation**: Caches per-partition results to `.cache/`
  so only one partition's worth of results is in memory at a time.

- **Predicate-key segmentation**: Processes metadata in 1M-record segments
  with per-segment disk caching.

These patterns are *local* to individual components. Each makes sensible
decisions in isolation, but none knows what the others are doing. The xvec
reader releases pages that the KNN partition cache might need. The
predicate-key segmentation picks a fixed 1M-record segment size regardless of
how much memory is available. The governor provides the global view that lets
these local decisions work together — coordinating page cache hints, sizing
segments to fit within the memory budget, and adjusting thread counts across
components that share the same CPU.
