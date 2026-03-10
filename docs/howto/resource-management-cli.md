<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example: Resource Management CLI Usage

## Fixed memory ceiling

Limit the pipeline to 32 GiB of memory:

```sh
veks run dataset.yaml --resources 'mem:32GiB'
```

## Dynamic range with governor adjustment

Let the governor adjust memory between 25% and 50% of system RAM, and
threads between 4 and 8:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8'
```

The governor starts at the midpoint (37.5% memory, 6 threads) and adjusts
based on observed RSS and system pressure.

## Predicate evaluation with bounded segments

For the resource-intensive `compute predicates` command, limit segment
size to prevent memory spikes:

```sh
veks run dataset.yaml --resources 'mem:25%-50%,segmentsize:500000-2000000'
```

## Direct command invocation with resources

```sh
veks pipeline compute knn \
  --base base_vectors.fvec \
  --query query_vectors.fvec \
  --indices neighbor_indices.ivec \
  --distances neighbor_distances.fvec \
  --neighbors 100 \
  --metric L2 \
  --resources 'mem:25%-50%,threads:8'
```

## Reading the governor log

After a pipeline run, inspect `.governor.log` for resource decisions:

```sh
# Show all throttle events
grep '"type":"throttle"' .governor.log | jq .

# Show resource adjustments
grep '"type":"decision"' .governor.log | jq '.resource, .old_value, .new_value, .reason'

# Show RSS over time
grep '"type":"observation"' .governor.log | jq '{ts: .ts, rss_pct: .rss_pct}'
```

## Resource value syntax examples

| Value | Meaning |
|-------|---------|
| `mem:32GiB` | Fixed 32 GiB memory ceiling |
| `mem:50%` | Fixed at 50% of system RAM |
| `mem:25%-50%` | Governor adjusts between 25% and 50% of system RAM |
| `threads:8` | Fixed 8 threads |
| `threads:4-8` | Governor adjusts between 4 and 8 threads |
| `segmentsize:500000` | Fixed 500K records per segment |
| `segmentsize:500000-2000000` | Governor adjusts segment size from 500K to 2M |
| `readahead:64MiB` | Fixed 64 MiB read-ahead buffer |
