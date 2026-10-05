# gsplat Step 1 — Segment

Divide the **output** ordinal space into contiguous segments, each
small enough that a fully assembled segment fits in the memory budget.
The segment count is the pass count: everything after this step repeats
once per segment.

Segments never partition the input. A pass owns one contiguous slice of
the *output* and pulls the records that belong there from arbitrary
positions all over the source. That asymmetry is the design — the write
side becomes contiguous for free, and the read side is left unordered,
which [L](./03-linearize.md) can fix by sorting. Scattered writes
cannot be un-scattered; see
[02-plan.md](./02-plan.md#segments-partition-the-output-not-the-input).

## Sizing

```
 records_per_segment = max(memory_budget / record_bytes, 1)
 num_segments        = max(ceil(output_count / records_per_segment), 2)
 segment_size        = ceil(output_count / num_segments)

 output ordinal space, output_count records
 ┌────────────────┬────────────────┬──────────────┐
 │   segment 0    │   segment 1    │  segment 2   │   ← one pass each
 │ [0, s)         │ [s, 2s)        │ [2s, count)  │
 └────────────────┴────────────────┴──────────────┘
   s = segment_size        buffer = s × record_bytes ≤ memory_budget
```

The floor of two segments keeps the resident buffer at or below half
the output size even when the budget would cover the whole rewrite in
one pass. A host that would rather special-case "it all fits" should do
that explicitly — read, permute in memory, write — rather than letting
this step produce a single pass.

**The pass count is the algorithm's primary cost lever.** Total work
grows with `P` until a threshold set by the storage tier's fetch
granularity, after which it flattens. Halving the budget can double the
I/O; sizing this step generously is the cheapest tuning available. The
model is in [cost-model.md](./cost-model.md).

## Where the budget comes from

Any of these, in descending order of preference:

- an admission controller or resource manager that hands out a byte
  budget and can revise it between passes,
- a container or cgroup memory limit, minus headroom,
- a fraction of physical memory,
- a caller-supplied constant.

Two subtleties are worth encoding wherever the number is chosen:

- **Leave the storage cache room.** The segment buffer competes with
  the host's page or block cache. A buffer that evicts the source's
  cached pages between passes defeats the sequential reads it was
  meant to enable. Reserving headroom for the source and output, then
  taking a fraction of the remainder, is a defensible default — but see
  the note above about `P`, because an over-conservative reserve buys
  cache hits at the price of extra passes, and passes usually cost
  more.
- **Count every resident structure.** The buffer is not the only
  resident state: the plan for the active pass is `sizeof(entry) ×
  segment_size` alongside it, and any per-worker scratch adds more.

## Variable-length records

With no fixed stride there is no exact `record_bytes`. Sample the store
— a bounded number of records spread evenly across it — to estimate a
mean size, and floor the estimate so a pathological sample cannot
produce absurd segment sizes. Segments are then sized in *expected*
bytes, so the buffer must tolerate overshoot: either size to the
estimate and grow on demand, or size below budget deliberately.

## Preparing the output

Fixed-stride sinks should preallocate the full output before the first
pass, so each pass writes into an existing extent rather than extending
the file. On most filesystems this is a near-instant sparse allocation;
some configurations stall in it long enough that it is worth logging
around.

Allocate the segment buffer once, sized for the largest segment, and
reuse it across passes. Repeatedly allocating and releasing a
multi-gigabyte buffer costs real time in kernel page zeroing, and on
some allocators it also fragments. If the buffer must be cleared
between passes, clear it in chunks with progress reporting: a single
zeroing of tens of gigabytes blocks long enough to look like a hang.

Next: [02-plan.md](./02-plan.md) — selecting and reversing the map
window for one segment.
