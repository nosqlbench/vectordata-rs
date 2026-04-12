<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Running a Pipeline with Resource Governance

Control memory and thread usage during large-scale pipeline runs.

## Prerequisites

- A built `veks` binary
- A dataset with resource-intensive steps (e.g., 1M+ vectors with
  metadata synthesis and filtered KNN)

## Step 1: Run without controls (baseline)

```shell
veks run dataset.yaml
```

Defaults: 80% of system RAM, all available CPU threads. Fine for small
datasets. For large ones (hundreds of millions of records), you may
see system slowdown or OOM kills.

## Step 2: Set a fixed memory ceiling

```shell
veks run dataset.yaml --resources 'mem:50%'
```

Caps RSS at 50% of system RAM. Commands throttle batch sizes to stay
within the limit.

## Step 3: Use dynamic ranges

```shell
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8'
```

The governor adjusts within the range based on observed pressure:
- **Memory**: starts at midpoint, scales based on RSS
- **Threads**: starts at midpoint, scales based on CPU utilization

## Step 4: Examine the governor log

```shell
cat .cache/.governor.log | head -20
```

Each line is JSON. Key entry types:

| Type | What it shows |
|------|--------------|
| `observation` | RSS, CPU, I/O, major faults, active threads |
| `decision` | Resource adjustments with reason and trigger metric |
| `throttle` | When a command was told to slow down |
| `request` | Resource negotiation (requested vs granted) |

Filter by type:

```shell
cat .cache/.governor.log | jq 'select(.type == "decision")'
cat .cache/.governor.log | jq 'select(.type == "throttle")'
```

## Step 5: Tune

| Log pattern | Action |
|-------------|--------|
| Mostly UNDERUSED (RSS well below ceiling) | Raise floor |
| Frequent THROTTLE/EMERGENCY | Lower ceiling or widen range |
| Granted << requested consistently | Step needs restructuring |
| Oscillating decisions | Widen range or use conservative strategy |

## Step 6: Governor strategies

```shell
# Maximize throughput (default)
veks run dataset.yaml --resources 'mem:25%-50%' --governor maximize

# Conservative (slow ramp-up, good for shared machines)
veks run dataset.yaml --resources 'mem:25%-50%' --governor conservative

# Fixed (no adjustment, reproducible benchmarks)
veks run dataset.yaml --resources 'mem:25%-50%' --governor fixed
```

---

## Resource value syntax

| Value | Meaning |
|-------|---------|
| `mem:32GiB` | Fixed 32 GiB ceiling |
| `mem:50%` | Fixed at 50% of system RAM |
| `mem:25%-50%` | Governor adjusts between 25% and 50% |
| `threads:8` | Fixed 8 threads |
| `threads:4-8` | Governor adjusts between 4 and 8 |
| `segmentsize:500000` | Fixed 500K records per segment |
| `readahead:64MiB` | Fixed 64 MiB read-ahead buffer |

Combine with commas: `--resources 'mem:25%-50%,threads:4-8,segmentsize:500000'`

---

## Tuning for specific workloads

### Predicate evaluation

Accumulates per-predicate match ordinals in memory. Use smaller
segments to bound peak memory:

```shell
veks run dataset.yaml --resources 'mem:25%-50%,segmentsize:500000'
```

### KNN computation

Mmaps the full base vector file. Give the governor generous memory
so it can coordinate competing mmap regions:

```shell
veks run dataset.yaml --resources 'mem:50%-80%,threads:4-8'
```

### Import and convert

I/O-bound. Increase read-ahead more than CPU threads:

```shell
veks run dataset.yaml --resources 'iothreads:4-8,readahead:64MiB-256MiB'
```
