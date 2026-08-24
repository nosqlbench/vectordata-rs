# gsplat Step 4 — Assemble

Walk the linearized plan, reading source records in ascending order and
scattering each into the segment buffer at its final position
(`segment_local_ordinal × record_bytes`). This is the transpose at the
heart of the rewrite: the permutation's randomness is spent here, in
memory, where seeks are free.

```
 source (read ascending)               segment buffer (memory)
 ┌────┬────┬────┬────┬────┬────┐      ┌──────┬──────┬──────┐
 │ .. │ 3  │ .. │71442 .. 902117      │local0│local1│local2│
 └────┴─┬──┴────┴──┬─┴────┴─┬──┘      └──────┴──────┴──────┘
        │          │        │             ▲      ▲      ▲
        └──────────┼────────┼─────────────┘      │      │
                   └────────┼────────────────────┘      │
                            └───────────────────────────┘
        sequential gather            random scatter — in memory,
                                     where seeks are free
```

## Lock-free parallelism

Each output position occurs exactly once in the plan, so concurrent
records land in **disjoint** buffer regions and no locking is required.
Workers need a way to express disjoint mutable views of one buffer; if
the host language cannot, give each worker private scratch and
concatenate in output order afterward
([host-interface.md](./host-interface.md#concurrency-requirement)).

Issue reads in plan order across the pool, so the aggregate access
pattern stays ascending even as individual workers interleave. A
relaxed counter is enough for progress; it does not need to be
sequentially consistent with the data writes, which are disjoint.

## Reads: positional, not mapped

Prefer a positional read primitive over memory-mapped slices for the
source. Faulting shuffled records through a mapping pulls every touched
page into the process's resident set — on a multi-terabyte source that
is a resident set measured in hundreds of gigabytes, and it evicts the
very cache the algorithm depends on. A positional read moves the same
bytes through the same cache without mapping them into the process, so
residency stays bounded by the segment buffer.

Where the storage tier charges per request (object storage), coalesce
adjacent plan entries into single ranged reads. Linearizing has already
made the candidates adjacent; the merge is a linear scan over the
sorted plan with a gap threshold.

## Per-record work

Everything that must touch each record belongs here, while its bytes
are already in cache — a separate pass over the output would cost
another full traversal:

| Hook | Shape | Notes |
|------|-------|-------|
| Reshape / re-encode | `bytes -> bytes` | Header rewrites, width conversion, compression |
| Normalize / rescale | in-place on the buffer | Can run after the scatter completes, over the contiguous buffer, which vectorizes better than per-record |
| Predicate / filter | `bytes -> bool` | Makes the output shorter than the plan; see compaction |
| Statistics | accumulate per worker | Merge per pass; keep accumulators associative so worker order does not change results |

**Sampling elision.** If a bounded sample proves a hook is a no-op for
this input — a transform that would leave every record unchanged —
skip it for all records and record that decision. The sample cost is
constant; the saving is `Θ(N)`.

**Compaction.** When a predicate drops records, the segment is shorter
than planned. Each worker writes into its own scratch, and the scratch
buffers are compacted in output order so the gaps close; the running
output total then drives the write position in
[T](./05-transfer.md), and the surviving-ordinal correspondence should
be recorded if downstream consumers need it. Compaction is the point
at which the output ordinal space stops being the map's ordinal space —
worth stating loudly wherever the result is documented.

## Variable-length records

With no fixed stride there is no transpose position to scatter into:
the buffer cannot know where record `l` starts until every record
before it is known. Assemble therefore collects `(local_ordinal,
bytes)` pairs while reading in source order, then sorts them by
local ordinal before handing the segment to the writer. That second
sort is memory-only and preserves both storage-facing invariants —
reads still ascended, and the write is still one ordered run.

## Buffer hygiene

If the segment buffer must be cleared between passes, clear it in
chunks with progress reporting. Zeroing tens of gigabytes in one call
blocks long enough to be indistinguishable from a hang, and on many
allocators a fresh allocation per pass silently pays that cost anyway.

Next: [05-transfer.md](./05-transfer.md) — flushing the segment.
