# SPLAT Step 1 — Segment

Divide the **output** ordinal space into contiguous segments, each
small enough that a fully assembled segment fits in the memory budget.
The segment count is the pass count: everything after this step repeats
once per segment.

Segments never partition the input. A pass owns one contiguous slice of
the *output* and pulls the records that belong there from arbitrary
positions all over the source. That asymmetry is the design: the write
side is contiguous for free, and every later step exists to tame the
read side — [L](./03-linearize.md) sorts it into one ascending sweep,
[A](./04-assemble.md) absorbs what remains of the scatter in RAM. See
[02-plan.md](./02-plan.md#segments-partition-the-output-not-the-input)
for why this is the side worth partitioning.

## Sizing

```
records_per_segment = max(mem_budget / record_bytes, 1)
num_segments        = max(ceil(extract_count / records_per_segment), 2)
segment_size        = ceil(extract_count / num_segments)

 output ordinal space, extract_count records
 ┌────────────────┬────────────────┬──────────────┐
 │   segment 0    │   segment 1    │  segment 2   │   ← one pass each
 │ [0, s)         │ [s, 2s)        │ [2s, count)  │
 └────────────────┴────────────────┴──────────────┘
   s = segment_size          buffer = s × record_bytes ≤ mem_budget
```

The floor of two segments keeps the resident buffer at or below half
the output size even when the budget would cover the whole extract in
one pass.

## Where the budget comes from

The memory budget is negotiated with the resource governor
(`ctx.governor.offer_demand("mem", …)` — see
[08-architecture.md §8.3](../08-architecture.md)), so `--resources
mem=4G` and memory-pressure bands both shape the pass count. The
per-variant defaults differ:

| Variant | Default budget |
|---------|----------------|
| mvec | Half of system RAM |
| fvec | Conservative: reserve page-cache headroom for the source and output (capped at a quarter of RAM each), then take 10% of the remainder, floor 256 MiB |
| slab | Half of system RAM, divided by a *sampled* mean record size (up to 1000 records probed evenly across the slab, floor 64 bytes/record) |

The fvec sizing is deliberately stingy: the segment buffer competes
with the source and output page cache, and a buffer that evicts the
source's pages between passes defeats the point of sequential reads.

## Output preallocation

Fixed-stride variants preallocate the full output with `set_len`
before the first pass, so each pass can seek to its segment offset
and write without extending the file. On most filesystems this is a
near-instant sparse allocation; the surrounding log lines exist
because some configurations stall in it for seconds.

The mvec variant also allocates its segment buffer once, sized for
the largest segment, and reuses it across passes (re-zeroing per
pass) — repeatedly allocating and freeing a tens-of-GiB buffer costs
real time in kernel page zeroing.

## Where in code

`sorted_index_extract_{fvec,mvec,slab}` in
`veks-pipeline/src/pipeline/commands/gen_extract.rs`, the block
labeled "Determine partition count from memory budget" (and the
fvec-specific "Partition sizing" block).

Next: [02-plan.md](./02-plan.md) — collecting the segment's read plan.
