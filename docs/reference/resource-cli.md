<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `--resources` CLI Reference

The `--resources` option configures how veks manages system resources
during pipeline execution. It controls memory budgets, thread counts,
segment sizes, and other tunable parameters. A background governor
thread monitors system state and dynamically adjusts resource values
within the configured bounds.

## Syntax

```
--resources 'key:value,key:value,...'
```

Multiple resource declarations are separated by commas within a single
quoted string. Each declaration is a resource name followed by a colon
and a value or range.

Examples:

```sh
--resources 'mem:32GiB'
--resources 'mem:25%-50%,threads:4-8'
--resources 'mem:25%-50%,threads:4-8,segmentsize:500000-2000000'
```

## Resource types

| Resource | Description | Value syntax | Default |
|----------|-------------|-------------|---------|
| `mem` | Memory budget (RSS ceiling) | Absolute size or % of system RAM | `80%` |
| `threads` | CPU thread pool size | Integer or integer range | `num_cpus` |
| `segments` | Maximum concurrent segments in flight | Integer | `1` |
| `segmentsize` | Records per processing segment | Integer or integer range | `1000000` |
| `iothreads` | Concurrent I/O operations | Integer or integer range | `4` |
| `cache` | Maximum disk space for `.cache/` directory | Absolute size | `unlimited` |
| `readahead` | Read-ahead buffer / prefetch window size | Absolute size or size range | `64MiB` |

## Value syntax

| Form | Example | Meaning |
|------|---------|---------|
| Absolute | `32GB`, `32GiB`, `1024MB` | Fixed value using SI or IEC units |
| Percentage | `50%` | Fraction of system capacity (RAM for `mem`) |
| Fixed count | `8` | Integer count (threads, segments, etc.) |
| Range (absolute) | `16GiB-48GiB` | Governor adjusts within bounds |
| Range (%) | `25%-50%` | Governor adjusts within bounds |
| Range (count) | `4-8` | Governor adjusts within bounds |

**Single value = fixed range.** When you specify a single value (e.g.,
`mem:32GiB`), it is treated as both the minimum and maximum. The
governor cannot adjust it -- the resource is pinned.

**Range = governor adjustable.** When you specify a range (e.g.,
`mem:25%-50%`), the governor starts at the midpoint and dynamically
adjusts between the floor and ceiling based on observed system pressure.

## Unit suffixes

Memory-like resources (`mem`, `cache`, `readahead`) accept size values
with unit suffixes. Both SI (base-1000) and IEC (base-1024) units are
supported.

### SI units (base 1000)

| Suffix | Multiplier |
|--------|-----------|
| `B` | 1 |
| `KB` | 1,000 |
| `MB` | 1,000,000 |
| `GB` | 1,000,000,000 |
| `TB` | 1,000,000,000,000 |

### IEC units (base 1024)

| Suffix | Multiplier |
|--------|-----------|
| `KiB` | 1,024 |
| `MiB` | 1,048,576 |
| `GiB` | 1,073,741,824 |
| `TiB` | 1,099,511,627,776 |

## Governor strategies

The `--governor` option selects the strategy the governor thread uses to
adjust resources within their configured ranges.

```sh
--governor maximize
```

| Strategy | Description |
|----------|-------------|
| `maximize` (default) | Aggressively uses available resources up to the configured ceiling. Scales up when headroom exists, scales down quickly under pressure. Targets 70%-85% of the memory ceiling as the nominal operating range. |
| `conservative` | Starts at floor values and only increases when sustained low utilization is observed over multiple evaluation cycles. Preferred for shared machines or unpredictable workloads. |
| `fixed` | Uses the midpoint of each range and never adjusts. No dynamic scaling. Useful for benchmarking where repeatable resource usage is needed. |

## Governor log format

The governor writes a line-oriented log of its observations and
decisions during execution.

### Location

```
${workspace}/.cache/.governor.log
```

The log is overwritten at the start of each pipeline run. Each line is
a self-contained JSON object with a `type` field.

### Entry types

**observation** -- periodic system snapshot recorded by the governor:

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

**decision** -- the governor adjusted a resource value:

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

**throttle** -- the governor signaled a command to reduce consumption:

```json
{
  "type": "throttle",
  "ts": "2026-03-05T06:14:35.001Z",
  "step_id": "evaluate-predicates",
  "reason": "RSS exceeded 95% of ceiling; emergency flush requested",
  "resources_affected": ["segmentsize", "threads"]
}
```

**request** -- a command requested a resource value and the governor
granted (or denied) it:

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

## Per-command resource applicability

Not all resources apply to every command. Each `CommandOp` declares
which resource types it consumes via `describe_resources()`. The
pipeline runner uses this information for validation and completion
filtering.

If you specify a resource that is not declared by the command, the
pipeline emits a warning (not an error) -- the governor may still use
system-level resources like `mem` for global monitoring even when the
command itself does not declare it.

Use `describe_resources()` on a command to see its applicable resource
list. In the CLI, tab-completion after `--resources` only suggests
resources declared by the current command.

### Example resource declarations by command

| Command | Applicable resources |
|---------|---------------------|
| `compute knn` | `mem`, `threads`, `readahead` |
| `compute filtered-knn` | `mem`, `threads`, `readahead` |
| `compute predicates` | `mem`, `threads`, `segments`, `segmentsize` |
| `generate predicated` | `mem`, `threads`, `readahead` |
| `import` | `threads`, `iothreads`, `readahead` |
| `convert file` | `iothreads`, `readahead` |
| `slab import` | `iothreads`, `readahead` |
| `analyze stats` | `mem`, `readahead` |
| `synthesize predicates` | (none -- lightweight) |
| `slab inspect` | (none -- lightweight) |
