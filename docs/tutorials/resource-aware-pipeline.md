<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Running a Pipeline with Resource Management

This tutorial walks you through running a veks pipeline with resource
governance enabled. You will start with no resource controls, add static
limits, upgrade to dynamic governor ranges, and learn to read the governor
log to tune your configuration.

## Prerequisites

- A built `veks` binary
- A dataset directory with source data large enough to exercise resource
  pressure (at least a few GB; the examples use a LAION-style predicated
  dataset)
- Familiarity with the pipeline command system and `dataset.yaml` format

## Step 1: Start with a compute-heavy dataset.yaml

Create or use a `dataset.yaml` that includes resource-intensive steps. This
example imports metadata, generates predicates and metadata-indices, and
computes KNN ground truth:

```yaml
name: laion-sample
metric: cosine
dimensions: 768

upstream:
  steps:
    - import:
        from: metadata.parquet
        to: metadata_content
        encoding: mnode

    - synthesize predicates:
        survey: survey.json
        count: 1000
        to: predicates

    - compute predicates:
        metadata: metadata_content
        predicates: predicates
        survey: survey.json
        to: metadata_indices

    - compute knn:
        base: base_vectors.fvec
        query: query_vectors.fvec
        k: 100
        to: ground_truth
```

The `compute predicates` step scans the entire metadata slab once per
segment, evaluating every predicate against every record. The `compute knn`
step memory-maps the full base vector file and runs brute-force distance
computations. Both are resource-intensive.

## Step 2: Run without resource controls

Run the pipeline with no `--resources` flag:

```sh
veks run dataset.yaml
```

With default settings, the pipeline uses the system default memory ceiling
(80% of system RAM) and `num_cpus` threads. For small datasets this works
fine. For large datasets — hundreds of millions of records, tens of thousands
of predicates — you may see:

- System slowdown as RSS climbs toward physical RAM limits
- Page cache thrashing when mmap regions compete for memory
- In the worst case, OOM kills or system lockup

If the run completes, note the elapsed time. You will compare against
governed runs in later steps.

## Step 3: Set a fixed memory ceiling

Add a static memory limit at 50% of system RAM:

```sh
veks run dataset.yaml --resources 'mem:50%'
```

This tells the governor "do not let RSS exceed 50% of system RAM." The
governor treats a single value as a fixed range (`50%-50%`), so it cannot
adjust — it enforces the ceiling but cannot scale down further or up.

If your dataset fits comfortably within 50%, you will see similar performance
to the uncontrolled run. If it does not fit, the governor will throttle
commands and trigger emergency flushes, which slows execution but prevents
OOM.

## Step 4: Let the governor adjust within a range

Replace the fixed ceiling with a range and add thread governance:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8'
```

Now the governor can dynamically adjust:

- **Memory**: The governor starts at the midpoint (37.5% of system RAM) and
  adjusts between 25% and 50% based on observed RSS pressure.
- **Threads**: The governor starts at 6 threads and adjusts between 4 and 8
  based on CPU utilization and memory headroom.

During the `import` step (I/O-bound, low memory), the governor will
likely stay in the UNDERUSED band and scale toward the ceiling. During
`compute predicates` (memory-intensive), the governor will scale down
if RSS climbs into the CAUTION or THROTTLE bands.

## Step 5: Examine the governor log

After the run completes, open the governor log:

```sh
cat .cache/.governor.log | head -20
```

Each line is a JSON object. The key entry types are:

**Observation entries** show what the governor saw:

```json
{
  "type": "observation",
  "ts": "2026-03-05T06:14:30.123Z",
  "rss_bytes": 17179869184,
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

**Decision entries** show what the governor changed and why:

```json
{
  "type": "decision",
  "ts": "2026-03-05T06:14:30.125Z",
  "step_id": "evaluate-predicates",
  "resource": "segmentsize",
  "old_value": 1000000,
  "new_value": 750000,
  "reason": "RSS at 91% of ceiling; reducing segment size to lower peak memory",
  "heuristic": "memory_headroom",
  "trigger_metric": "rss_pct",
  "trigger_value": 91.2
}
```

**Throttle entries** show when the governor told a command to slow down:

```json
{
  "type": "throttle",
  "ts": "2026-03-05T06:14:35.001Z",
  "step_id": "evaluate-predicates",
  "reason": "RSS exceeded 90% of ceiling; throttle requested",
  "resources_affected": ["segmentsize", "threads"]
}
```

**Request/Grant entries** show resource negotiations:

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

To get a quick summary of how the governor behaved, filter by entry type:

```sh
# Count entries by type
cat .cache/.governor.log | jq -r '.type' | sort | uniq -c

# Show only decisions
cat .cache/.governor.log | jq 'select(.type == "decision")'

# Show throttle events
cat .cache/.governor.log | jq 'select(.type == "throttle")'
```

## Step 6: Tune based on what the log shows

The governor log tells you whether your ranges are well-sized.

**If the governor spent most of the run in UNDERUSED** (observations show
`rss_pct` consistently below 50% of ceiling), your floor is too low. The
governor is wasting time ramping up. Raise the floor:

```sh
veks run dataset.yaml --resources 'mem:40%-50%,threads:6-8'
```

**If the governor frequently hit THROTTLE or EMERGENCY** (many throttle
entries, decisions repeatedly reducing segment sizes), your ceiling is too
high for the available memory or the workload is inherently too large for the
range. Lower the ceiling or widen the range:

```sh
veks run dataset.yaml --resources 'mem:20%-40%,threads:4-8'
```

**If you see many request/grant entries where the granted value is much lower
than the requested value**, the command wants more resources than the governor
can safely provide. This is normal under pressure but if it happens
consistently, consider whether the step needs restructuring (more segments,
smaller predicate batches).

**If decision entries oscillate** (segment size going up and down repeatedly),
the governor is hunting. This usually means the range is narrow and the
workload is variable. Widen the range to give the governor more room, or
switch to the `conservative` strategy which applies larger damping.

## Step 7: Try different governor strategies

The `--governor` flag selects the governor strategy:

**maximize (default)** — Aggressively uses available resources. Best for
throughput when you trust the ranges:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8' --governor maximize
```

**conservative** — Starts at floor values and only scales up under sustained
low utilization. Best for shared machines or when you are unsure about the
workload's memory profile:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8' --governor conservative
```

Compare the governor logs from `maximize` and `conservative` runs. The
`conservative` log will show fewer decisions and longer periods in the
UNDERUSED band before scaling up. The `maximize` log will show faster
ramp-up and more time in NOMINAL.

**fixed** — Uses the midpoint of each range and never adjusts. Best for
reproducible benchmarks where you want identical resource conditions across
runs:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8' --governor fixed
```

With `fixed`, the governor log will contain only observation entries — no
decisions or throttle events. This confirms that resources were held constant.

## Summary

| What you did | Why |
|-------------|-----|
| Ran without `--resources` | Established a baseline with system defaults |
| Added `--resources 'mem:50%'` | Set a fixed ceiling to prevent OOM |
| Added `--resources 'mem:25%-50%,threads:4-8'` | Gave the governor room to adapt |
| Read `.cache/.governor.log` | Understood what the governor observed and decided |
| Tuned floor and ceiling | Matched the resource envelope to the actual workload |
| Tried `--governor conservative` and `--governor fixed` | Selected a strategy that matches the operational context |

The governor log is the primary tool for understanding resource behavior. Read
it after every significant run, especially when tuning a new dataset or
running on new hardware.
