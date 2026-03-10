<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How to Configure Resource Management

The `--resources` option controls how veks manages memory, threads, and
other system resources during pipeline execution. This guide covers the
most common configuration scenarios.

## Setting a memory budget

Use the `mem` resource to cap the resident set size (RSS) ceiling:

```sh
veks run dataset.yaml --resources 'mem:32GiB'
```

A single value creates a fixed ceiling -- the governor cannot adjust it.
To let the governor dynamically adjust memory usage between a floor and
ceiling, specify a range:

```sh
veks run dataset.yaml --resources 'mem:25%-50%'
```

Percentages are relative to total system RAM. On a 128 GiB machine,
`25%-50%` means the governor operates between 32 GiB and 64 GiB,
starting at the midpoint (48 GiB) and adjusting based on observed
pressure.

## Controlling thread count

Use the `threads` resource to limit CPU thread pool size:

```sh
veks run dataset.yaml --resources 'threads:4-8'
```

A range lets the governor scale threads down under memory pressure and
back up when headroom is available. A single value pins the thread count:

```sh
veks run dataset.yaml --resources 'threads:8'
```

## Combining multiple resources

Separate resource declarations with commas in a single `--resources`
value:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8,segmentsize:500000-2000000'
```

All three resources are parsed together and passed to the governor. You
can mix fixed values and ranges:

```sh
veks run dataset.yaml --resources 'mem:32GiB,threads:4-8,segmentsize:500000'
```

Here `mem` and `segmentsize` are fixed while `threads` is adjustable.

## Choosing a governor strategy

The governor strategy controls how aggressively the system uses available
resources. Select a strategy with `--governor`:

```sh
veks run dataset.yaml --resources 'mem:25%-50%' --governor maximize
```

| Strategy | Behavior |
|----------|----------|
| `maximize` (default) | Aggressively uses available resources up to the configured ceiling. Scales up when headroom exists, scales down quickly under pressure. |
| `conservative` | Starts at floor values and only increases when sustained low utilization is observed over multiple evaluation cycles. |
| `fixed` | Uses the midpoint of each range and never adjusts. Useful for benchmarking or when you want predictable resource usage. |

For production workloads on shared machines, `conservative` avoids
competing with other processes. For dedicated machines running a single
pipeline, `maximize` (the default) extracts the best throughput.

## Reading the governor utilization log

During execution the governor writes a line-oriented JSON log to:

```
${workspace}/.governor.log
```

Each line is a self-contained JSON object with a `type` field. The four
entry types are:

**observation** -- periodic system snapshot:

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
  "active_threads": 8,
  "step_id": "evaluate-predicates"
}
```

**decision** -- resource adjustment:

```json
{
  "type": "decision",
  "ts": "2026-03-05T06:14:30.125Z",
  "step_id": "evaluate-predicates",
  "resource": "segmentsize",
  "old_value": 1000000,
  "new_value": 750000,
  "reason": "RSS at 91% of ceiling (32GiB); reducing segment size"
}
```

**throttle** -- command told to reduce consumption:

```json
{
  "type": "throttle",
  "ts": "2026-03-05T06:14:35.001Z",
  "step_id": "evaluate-predicates",
  "reason": "RSS exceeded 95% of ceiling; emergency flush requested",
  "resources_affected": ["segmentsize", "threads"]
}
```

**request** -- command resource negotiation:

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

Use these entries to understand why the governor changed resource levels.
If you see frequent throttle entries, your floor values may be too high
for the available system memory.

The log is overwritten at the start of each pipeline run. Archive it
before re-running if you need historical data.

## Tuning for specific workloads

Different commands have different resource profiles. Not all resources
apply to every command -- each command declares which resources it
consumes via `describe_resources()`.

### Predicate evaluation

The `compute predicates` command accumulates per-predicate match
ordinal vectors in memory. With many predicates and a large metadata
corpus, memory can grow rapidly within a segment. Use smaller segment
sizes to bound peak memory:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,segmentsize:500000'
```

Smaller segments mean more disk I/O for segment caching, but the
per-segment memory footprint stays manageable.

### KNN computation

The `compute knn` command mmaps the full base vector file. The kernel
manages page cache, but competing mmap regions (base + query + metadata)
can cause thrashing if the working set exceeds physical RAM. Give the
governor a generous memory range so it can coordinate:

```sh
veks run dataset.yaml --resources 'mem:50%-80%,threads:4-8,readahead:128MiB'
```

A larger `readahead` window helps sequential scan performance when
sufficient memory is available.

### Import and convert

Import and convert commands are I/O-bound. Increasing `iothreads` and
`readahead` improves throughput more than adding CPU threads:

```sh
veks run dataset.yaml --resources 'iothreads:4-8,readahead:64MiB-256MiB'
```

## Example: LAION-400M predicate evaluation

The LAION-400M img2text dataset has 407 million metadata records in a
207 GB slab file. Evaluating 10,000 predicates against this corpus
requires careful resource management to avoid system lockup.

A safe configuration for a machine with 128 GiB RAM:

```sh
veks run laion-400m.yaml \
  --resources 'mem:25%-50%,threads:4-8,segmentsize:500000' \
  --governor conservative
```

This configuration:

- Keeps memory between 32 GiB and 64 GiB (25%-50% of 128 GiB)
- Limits threads to 4-8, reducing per-thread buffer accumulation
- Uses 500K-record segments instead of the default 1M, halving
  per-segment memory for match ordinal vectors
- Uses the `conservative` governor to avoid aggressive scale-up that
  could spike RSS on high-selectivity predicates

Monitor the governor log during execution:

```sh
tail -f laion-400m/.governor.log | jq 'select(.type == "decision")'
```

If you see repeated segment size reductions, the floor may still be too
high. Lower `segmentsize` to 250000 or reduce the `mem` ceiling.
